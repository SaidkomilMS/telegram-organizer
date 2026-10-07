"""llm/: wire shapes per dialect and provider (httpx.MockTransport, no network), the retry and
out-of-credit policy, the monthly budget and its one notification, OpenRouter settlement,
self-hosted thinking suppression, the prompts' defensive parsers, and the disabled LLM."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, FakeClock
from tg_curator import contracts
from tg_curator.config import LlmSettings
from tg_curator.db.store import Store
from tg_curator.domain import ProviderInfo, Topic, TopicName
from tg_curator.llm import base, budget, factory, prompts, providers, registry
from tg_curator.llm.base import (
    ChatLLM,
    DisabledLLM,
    LLMAuthError,
    LLMCreditError,
    LLMRequestError,
    LLMUnavailableError,
    Usage,
)
from tg_curator.llm.budget import Budget, Price
from tg_curator.llm.providers import (
    AnthropicBackend,
    GoogleBackend,
    OpenAIChatBackend,
    build_backend,
    normalise_base_url,
)
from tg_curator.runtime import Runtime

POST = (
    "Центробанк Узбекистана сохранил основную ставку на уровне 14% годовых, сославшись на "
    "замедление инфляции до 8,9% в сентябре и стабильный курс сума."
)
SUMMARY = "ЦБ Узбекистана сохранил ставку 14% на фоне замедления инфляции до 8,9%."


# --- helpers -----------------------------------------------------------------------------------


@dataclass
class Recorder:
    """A MockTransport handler that stores every request and serves scripted responses."""

    responses: list[httpx.Response | Callable[[httpx.Request], httpx.Response]]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return item(request) if callable(item) else item

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ok(payload: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None):
    return httpx.Response(status, json=payload, headers=headers)


def openai_answer(text: str | None, **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    message.update(extra.pop("message", {}))
    body = {
        "id": "chatcmpl-1",
        "model": "gpt-6-luna",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }
    body.update(extra)
    return body


class Sleeps:
    """Records the retry delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


async def zero_price(_model: str | None) -> Price:
    return budget.ZERO_PRICE


def luna_price() -> Callable[[str | None], Any]:
    async def lookup(_model: str | None) -> Price:
        return Price(0.10, 0.50)

    return lookup


def make_budget(
    store: Store, clock: FakeClock, *, cap: float = 0.0, price: Any = None, notifier: Any = None
) -> Budget:
    return Budget(
        store,
        clock,
        cap_usd=cap,
        timezone="Asia/Tashkent",
        price_lookup=price or zero_price,
        notifier=notifier,
    )


def cfg(**kwargs: Any) -> LlmSettings:
    return LlmSettings(**kwargs)


# --- one round trip per dialect and provider ---------------------------------------------------


async def test_openai_request_shape_and_parse() -> None:
    usage_in = {
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    rec = Recorder([ok(openai_answer(SUMMARY, usage=usage_in))])
    backend = build_backend(
        cfg(mode="provider", provider="openai", model="gpt-6-luna", api_key="sk-test"),
        client=client_for(rec),
    )
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=False)
    req = rec.requests[0]
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = rec.body()
    assert body == {
        "model": "gpt-6-luna",
        "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        "temperature": 0.2,
        "max_completion_tokens": 200,
        "reasoning_effort": "none",
    }
    assert "max_tokens" not in body
    assert text == SUMMARY
    assert usage == Usage(120, 40, reasoning_tokens=5, model="gpt-6-luna")
    assert backend.read_timeout == 60.0


async def test_openai_json_mode_adds_response_format() -> None:
    rec = Recorder([ok(openai_answer('{"name": "A", "description": "b"}'))])
    backend = build_backend(
        cfg(mode="provider", provider="openai", model="gpt-6-luna", api_key="k"),
        client=client_for(rec),
    )
    await backend.complete("S", "U", max_tokens=400, json_mode=True)
    assert rec.body()["response_format"] == {"type": "json_object"}


async def test_anthropic_haiku_omits_thinking_and_keeps_temperature() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "id": "msg_1",
                    "type": "message",
                    "model": "claude-haiku-4-5",
                    "content": [{"type": "text", "text": SUMMARY}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 35},
                }
            )
        ]
    )
    backend = build_backend(
        cfg(mode="provider", provider="anthropic", model="claude-haiku-4-5", api_key="sk-ant"),
        client=client_for(rec),
    )
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=False)
    req = rec.requests[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-ant"
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in req.headers
    assert rec.body() == {
        "model": "claude-haiku-4-5",
        "max_tokens": 200,
        "temperature": 0.2,
        "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
    }
    assert text == SUMMARY
    assert usage == Usage(100, 35, model="claude-haiku-4-5")


async def test_anthropic_sonnet_between_tools_low_effort_no_sampling_and_json_schema() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "type": "message",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "text", "text": '{"name": "Crypto", "description": "d"}'},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 300,
                        "output_tokens": 60,
                        "output_tokens_details": {"thinking_tokens": 12},
                    },
                }
            )
        ]
    )
    backend = AnthropicBackend("claude-sonnet-5-5", api_key="k", client=client_for(rec))
    text, usage = await backend.complete("S", "U", max_tokens=400, json_mode=True)
    body = rec.body()
    assert body["thinking"] == {"type": "between_tools"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": prompts.TOPIC_JSON_SCHEMA,
    }
    assert "temperature" not in body and "top_p" not in body
    assert text == '{"name": "Crypto", "description": "d"}'
    assert usage.reasoning_tokens == 12 and usage.output_tokens == 60


async def test_anthropic_refusal_is_an_empty_answer_with_usage() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "I cannot help"}],
                    "stop_reason": "refusal",
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                }
            )
        ]
    )
    backend = AnthropicBackend("claude-haiku-4-5", api_key="k", client=client_for(rec))
    text, usage = await backend.complete("S", "U", max_tokens=200, json_mode=False)
    assert text == "" and usage.output_tokens == 4


async def test_google_native_request_shape_and_thinking_accounting() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "ignored thought", "thought": True},
                                    {"text": SUMMARY},
                                ],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": "gemini-3.5-flash-lite",
                    "usageMetadata": {
                        "promptTokenCount": 150,
                        "candidatesTokenCount": 40,
                        "thoughtsTokenCount": 25,
                    },
                }
            )
        ]
    )
    backend = build_backend(
        cfg(mode="provider", provider="google", model="gemini-3.5-flash-lite", api_key="g-key"),
        client=client_for(rec),
    )
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=False)
    req = rec.requests[0]
    assert (
        str(req.url) == "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash-lite:generateContent"
    )
    assert req.headers["x-goog-api-key"] == "g-key"
    assert "key=" not in str(req.url)
    assert rec.body() == {
        "systemInstruction": {"parts": [{"text": "SYS"}]},
        "contents": [{"role": "user", "parts": [{"text": "USER"}]}],
        "generationConfig": {
            "maxOutputTokens": 200 + providers.GEMINI_THINKING_HEADROOM,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    assert text == SUMMARY
    assert usage == Usage(150, 65, reasoning_tokens=25, model="gemini-3.5-flash-lite")
    assert backend.output_budget(200) == 200 + providers.GEMINI_THINKING_HEADROOM


async def test_google_flash_uses_low_json_mime_and_safety_block_is_empty() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "promptFeedback": {"blockReason": "SAFETY"},
                    "usageMetadata": {"promptTokenCount": 9},
                }
            )
        ]
    )
    backend = GoogleBackend("gemini-3.8-flash", api_key="k", client=client_for(rec))
    text, usage = await backend.complete("S", "U", max_tokens=400, json_mode=True)
    config = rec.body()["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert config["responseMimeType"] == "application/json"
    assert "temperature" not in config
    assert text == "" and usage.input_tokens == 9 and usage.output_tokens is None


async def test_mistral_request_shape_and_chunk_list_content() -> None:
    rec = Recorder(
        [
            ok(
                {
                    "model": "mistral-small-2603",
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {
                                        "type": "thinking",
                                        "thinking": [{"type": "text", "text": "hmm"}],
                                    },
                                    {"type": "text", "text": SUMMARY},
                                ]
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 90, "completion_tokens": 31},
                }
            )
        ]
    )
    backend = build_backend(
        cfg(mode="provider", provider="mistral", model="mistral-small-latest", api_key="m"),
        client=client_for(rec),
    )
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=False)
    assert str(rec.requests[0].url) == "https://api.mistral.ai/v1/chat/completions"
    body = rec.body()
    assert body["max_tokens"] == 200 and "max_completion_tokens" not in body
    assert body["reasoning_effort"] == "none" and body["temperature"] == 0.2
    assert text == SUMMARY
    assert usage.model == "mistral-small-2603"  # the dated id prices the settlement


async def test_openrouter_headers_reasoning_and_cost() -> None:
    rec = Recorder(
        [
            ok(
                openai_answer(
                    SUMMARY,
                    model="openai/gpt-6-luna",
                    usage={"prompt_tokens": 100, "completion_tokens": 30, "cost": 0.000025},
                )
            )
        ]
    )
    backend = build_backend(
        cfg(mode="provider", provider="openrouter", model="openai/gpt-6-luna", api_key="sk-or"),
        client=client_for(rec),
    )
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=False)
    req = rec.requests[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert "http-referer" not in req.headers
    assert req.headers["x-openrouter-title"] == "tg-curator"
    body = rec.body()
    assert body["reasoning"] == {"effort": "none", "exclude": True}
    assert body["max_tokens"] == 200 and "reasoning_effort" not in body
    assert usage.reported_cost_usd == pytest.approx(0.000025)
    assert text == SUMMARY


async def test_openrouter_mandatory_reasoning_sends_lowest_effort(clock: FakeClock) -> None:
    catalogue = registry.OpenRouterCatalogue(clock)
    listing = {
        "data": [
            {
                "id": "google/gemini-3.5-flash-lite",
                "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
                "reasoning": {
                    "mandatory": True,
                    "supported_efforts": ["high", "medium", "low", "minimal"],
                },
            },
            {
                "id": "openai/gpt-6-luna",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
                "reasoning": {"mandatory": False, "supported_efforts": ["none", "low"]},
            },
        ]
    }
    assert await catalogue.refresh(client_for(Recorder([ok(listing)])))
    assert catalogue.effort("google/gemini-3.5-flash-lite") == "minimal"
    assert catalogue.effort("openai/gpt-6-luna") == "none"
    assert catalogue.effort("unknown/model") == "none"
    assert catalogue.price("google/gemini-3.5-flash-lite") == Price(
        0.3, 2.5, "https://openrouter.ai/api/v1/models"
    )
    rec = Recorder([ok(openai_answer(SUMMARY))])
    backend = OpenAIChatBackend(
        "google/gemini-3.5-flash-lite",
        flavour="openrouter",
        base_url=providers.PROVIDERS["openrouter"].base_url,
        api_key="k",
        effort_lookup=catalogue.effort,
        client=client_for(rec),
    )
    await backend.complete("S", "U", max_tokens=200, json_mode=False)
    assert rec.body()["reasoning"] == {"effort": "minimal", "exclude": True}


async def test_openrouter_200_with_error_body_is_not_an_answer() -> None:
    rec = Recorder([ok({"error": {"code": 502, "message": "upstream died"}})])
    backend = build_backend(
        cfg(mode="provider", provider="openrouter", model="openai/gpt-6-luna", api_key="k"),
        client=client_for(rec),
        sleep=Sleeps(),
    )
    with pytest.raises(LLMUnavailableError):
        await backend.complete("S", "U", max_tokens=200, json_mode=False)
    assert len(rec.requests) == 3  # a 502 in the body is retried like a 502 status
    rec = Recorder([ok({"error": {"code": 402, "message": "Insufficient credits"}})])
    backend = build_backend(
        cfg(mode="provider", provider="openrouter", model="openai/gpt-6-luna", api_key="k"),
        client=client_for(rec),
    )
    with pytest.raises(LLMCreditError):
        await backend.complete("S", "U", max_tokens=200, json_mode=False)
    assert len(rec.requests) == 1


async def test_selfhosted_request_shape_think_stripping_and_reasoning_fields() -> None:
    rec = Recorder(
        [
            ok(
                openai_answer(
                    f"<think>\nlet me think\n</think>\n{SUMMARY}",
                    model="gemma4:12b",
                    message={"reasoning": "secret thoughts", "reasoning_content": "more"},
                    usage=None,
                )
            )
        ]
    )
    backend = build_backend(
        cfg(mode="selfhosted", base_url="localhost:11434", model="gemma4:12b", api_key=""),
        client=client_for(rec),
    )
    assert backend.selfhosted and backend.read_timeout == 180.0
    text, usage = await backend.complete("SYS", "USER", max_tokens=200, json_mode=True)
    req = rec.requests[0]
    assert str(req.url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in req.headers
    body = rec.body()
    assert body["reasoning_effort"] == "none"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 200
    assert "response_format" not in body  # prompt-only JSON for local servers
    assert text == SUMMARY
    assert usage.estimated and usage.input_tokens is None and usage.model == "gemma4:12b"


async def test_selfhosted_token_and_presets() -> None:
    rec = Recorder([ok(openai_answer(SUMMARY))])
    backend = build_backend(
        cfg(mode="selfhosted", base_url="https://llm.example/v1/", model="m", api_key="tok"),
        client=client_for(rec),
    )
    await backend.complete("S", "U", max_tokens=5, json_mode=False)
    assert rec.requests[0].headers["authorization"] == "Bearer tok"
    assert str(rec.requests[0].url) == "https://llm.example/v1/chat/completions"
    assert normalise_base_url("http://host:8000/v1") == "http://host:8000/v1"
    assert normalise_base_url("host:1234") == "http://host:1234/v1"
    presets = {p.key: p for p in registry.selfhosted_presets()}
    assert (
        presets["ollama"].base_url == "http://localhost:11434/v1" and presets["ollama"].key_ignored
    )
    assert presets["vllm"].base_url == "http://localhost:8000/v1"
    assert presets["lmstudio"].base_url == "http://localhost:1234/v1"
    assert presets["llamacpp"].base_url == "http://localhost:8080/v1"
    assert presets["llamacpp"].health_path == "/health"
    assert all(p.key_optional for p in presets.values())


# --- retry policy ------------------------------------------------------------------------------


async def test_retry_on_429_honours_retry_after_then_succeeds() -> None:
    sleeps = Sleeps()
    slow_down = {"error": {"message": "slow down", "type": "rate_limit_error", "code": "slow_down"}}
    rec = Recorder(
        [
            ok(slow_down, 429, {"retry-after": "7"}),
            ok({"error": {"message": "overloaded"}}, 529),
            ok(openai_answer(SUMMARY)),
        ]
    )
    backend = OpenAIChatBackend(
        "gpt-6-luna",
        flavour="openai",
        base_url="https://api.openai.com/v1",
        api_key="k",
        client=client_for(rec),
        sleep=sleeps,
    )
    text, _ = await backend.complete("S", "U", max_tokens=200, json_mode=False)
    assert text == SUMMARY
    assert len(rec.requests) == 3
    assert sleeps.delays[0] == 7.0  # Retry-After wins over the backoff
    assert 1.5 <= sleeps.delays[1] <= 2.5  # second retry: the 2 s step with ±25 % jitter


async def test_retries_are_bounded_and_connection_errors_count() -> None:
    sleeps = Sleeps()

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rec = Recorder([boom])
    backend = OpenAIChatBackend(
        "m", flavour="selfhosted", base_url="http://x/v1", api_key="", client=client_for(rec),
        sleep=sleeps,
    )  # fmt: skip
    with pytest.raises(LLMUnavailableError):
        await backend.complete("S", "U", max_tokens=5, json_mode=False)
    assert len(rec.requests) == base.MAX_ATTEMPTS
    assert len(sleeps.delays) == 2  # two sleeps between three attempts
    assert 0.75 <= sleeps.delays[0] <= 1.25 and 1.5 <= sleeps.delays[1] <= 2.5


async def test_timeout_is_reported_as_timeout() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    backend = OpenAIChatBackend(
        "m", flavour="selfhosted", base_url="http://x/v1", api_key="", client=client_for(slow),
        sleep=Sleeps(),
    )  # fmt: skip
    with pytest.raises(base.LLMTimeoutError):
        await backend.complete("S", "U", max_tokens=5, json_mode=False)


@pytest.mark.parametrize(
    ("status", "body", "exc"),
    [
        (402, {"error": {"message": "billing"}}, LLMCreditError),
        (400, {"error": {"message": "bad", "type": "invalid_request_error"}}, LLMRequestError),
        (401, {"error": {"message": "invalid key"}}, LLMAuthError),
        (403, {"error": {"message": "forbidden"}}, LLMAuthError),
        (404, {"error": {"message": "model: nope"}}, LLMRequestError),
        (413, {"error": {"message": "too large"}}, LLMRequestError),
        (422, {"object": "error", "message": "validation"}, LLMRequestError),
        (429, {"error": {"message": "quota", "code": "insufficient_quota"}}, LLMCreditError),
        (429, {"error": {"message": "q", "code": "credit_balance_exhausted"}}, LLMCreditError),
        (
            429,
            {"error": {"message": "s", "details": {"error_code": "enforced_spend_limit_reached"}}},
            LLMCreditError,
        ),
        (
            400,
            {"error": {"message": "You have reached your specified API usage limits for ..."}},
            LLMCreditError,
        ),
    ],
)  # fmt: skip
async def test_no_retry_on_credit_and_request_errors(
    status: int, body: dict[str, Any], exc: type[Exception]
) -> None:
    sleeps = Sleeps()
    rec = Recorder([ok(body, status)])
    backend = OpenAIChatBackend(
        "m", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec), sleep=sleeps,
    )  # fmt: skip
    with pytest.raises(exc) as info:
        await backend.complete("S", "U", max_tokens=5, json_mode=False)
    assert len(rec.requests) == 1 and sleeps.delays == []
    assert "k" not in str(info.value).replace("key", "")  # the message carries no secret
    assert info.value.status == status


# --- the budget --------------------------------------------------------------------------------


async def test_budget_reserve_settle_release(store: Store, clock: FakeClock) -> None:
    b = make_budget(store, clock, cap=1.0, price=luna_price())
    assert b.month() == "2026-10"
    res = await b.reserve(400, 200)
    assert res is not None and res.cost_usd == pytest.approx((400 * 0.10 + 200 * 0.50) / 1e6)
    usage = await store.get_usage("2026-10")
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (1, 400, 200)
    settled = await b.settle(res, input_tokens=380, output_tokens=40)
    assert not settled.estimated
    usage = await store.get_usage("2026-10")
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (1, 380, 40)
    assert usage.cost_usd == pytest.approx((380 * 0.10 + 40 * 0.50) / 1e6)
    res2 = await b.reserve(100, 50)
    assert res2 is not None
    await b.release(res2)
    usage = await store.get_usage("2026-10")
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (2, 380, 40)
    res3 = await b.reserve(100, 50)
    assert res3 is not None
    await b.release(res3, keep_estimate=True)  # a timeout: the estimate stays
    usage = await store.get_usage("2026-10")
    assert usage.input_tokens == 480 and usage.cost_usd > (380 * 0.10 + 40 * 0.50) / 1e6


async def test_budget_estimated_settlement_and_reported_cost(
    store: Store, clock: FakeClock
) -> None:
    b = make_budget(store, clock, price=luna_price())
    res = await b.reserve(100, 200)
    assert res is not None
    settled = await b.settle(res, input_tokens=None, output_tokens=None)
    assert settled.estimated and settled.cost_usd == pytest.approx(res.cost_usd * 1.5)
    res = await b.reserve(100, 200)
    assert res is not None
    settled = await b.settle(res, input_tokens=90, output_tokens=20, reported_cost_usd=0.000123)
    assert settled.cost_usd == 0.000123
    assert (await store.get_usage("2026-10")).cost_usd == pytest.approx(
        res.cost_usd * 1.5 + 0.000123
    )


async def test_cap_reached_returns_none_and_notifies_once(
    rt: Runtime, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    async def pricey(_model: str | None) -> Price:
        return Price(1_000_000.0, 1_000_000.0)  # $1 per token: the second call crosses $1

    b = Budget(
        rt.store, clock, cap_usd=1.0, timezone="UTC", price_lookup=pricey, notifier=rt.notifier
    )
    rec = Recorder([ok(openai_answer(SUMMARY, usage={"prompt_tokens": 1, "completion_tokens": 0}))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, b, concurrency=4)
    assert isinstance(llm, contracts.LLM) and llm.enabled
    # a reservation of ~(len/2.5 + 200) tokens at $1 each is far above the $1 cap: refused
    assert await llm.summarise_line(POST, 180) is None
    assert rec.requests == []
    # nothing spent, but the cap is reached for the month: it stays closed until it ends
    assert await b.exhausted("2026-10") is True
    assert (await rt.store.get_usage("2026-10")).cap_notified
    sent = bot_gw.sent(OWNER_ID)
    assert len(sent) == 1 and "2026-10" in sent[0].text and "$1.00" in sent[0].text
    assert await llm.summarise_line(POST, 180) is None
    assert await llm.name_topic([POST]) is None
    assert len(bot_gw.sent(OWNER_ID)) == 1  # once a month, whatever happens next


async def test_exhausted_after_spending_the_cap(store: Store, clock: FakeClock) -> None:
    b = make_budget(store, clock, cap=0.001, price=luna_price())
    res = await b.reserve(100, 10)
    assert res is not None
    await b.settle(res, input_tokens=5000, output_tokens=1000)  # $0.0005 + $0.0005
    assert await b.exhausted("2026-10")
    assert await b.reserve(1, 1) is None
    assert (await store.get_usage("2026-10")).cap_notified
    clock.advance(31 * 24 * 3600)
    assert b.month() == "2026-11" and not await b.exhausted()


async def test_out_of_credit_pauses_the_provider_for_the_month(
    rt: Runtime, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    b = Budget(
        rt.store,
        clock,
        cap_usd=0.0,
        timezone="UTC",
        price_lookup=luna_price(),
        notifier=rt.notifier,
    )
    rec = Recorder([ok({"error": {"message": "no credits"}}, 402)])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, b, concurrency=4)
    assert await llm.summarise_line(POST, 180) is None
    assert len(rec.requests) == 1
    assert b.paused_month == "2026-10" and await b.exhausted()
    assert await llm.summarise_line(POST, 180) is None
    assert len(rec.requests) == 1  # paused: no second request this month
    [notice] = bot_gw.sent(OWNER_ID)
    # the provider's own words (§17.5), not a cap line showing a $0.00 cap nobody set
    assert notice.text.startswith("OpenAI reports the account is out of credit")
    assert "until Oct 31" in notice.text and "$" not in notice.text
    usage = await rt.store.get_usage("2026-10")
    assert usage.requests == 1 and usage.cost_usd == 0.0  # the reservation was released
    assert usage.cap_notified  # shared with the cap line: one notice per month at most
    clock.advance(31 * 24 * 3600)
    assert not await b.exhausted()


async def test_openrouter_cost_settlement_through_the_llm(store: Store, clock: FakeClock) -> None:
    b = make_budget(store, clock, cap=5.0, price=luna_price())
    usage_in = {"prompt_tokens": 100, "completion_tokens": 30, "cost": 0.0042}
    rec = Recorder([ok(openai_answer(SUMMARY, usage=usage_in))])
    backend = OpenAIChatBackend(
        "openai/gpt-6-luna", flavour="openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="k", client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, b, concurrency=4)
    assert await llm.summarise_line(POST, 180) == SUMMARY
    usage = await store.get_usage("2026-10")
    assert usage.cost_usd == pytest.approx(0.0042)
    assert (usage.input_tokens, usage.output_tokens, usage.requests) == (100, 30, 1)


async def test_selfhosted_costs_nothing_but_is_counted(store: Store, clock: FakeClock) -> None:
    settings_like = type("S", (), {})()
    settings_like.llm = cfg(mode="selfhosted", base_url="http://localhost:8080", model="m")
    settings_like.general = type("G", (), {"timezone": "UTC"})()
    rec = Recorder([ok(openai_answer(SUMMARY, usage=None))])
    llm = factory.make_llm(settings_like, store, clock=clock, client=client_for(rec))
    assert isinstance(llm, ChatLLM) and llm.budget.cap_usd == 0.0
    assert llm._semaphore._value == 1  # self-hosted: one request at a time
    assert await llm.summarise_line(POST, 180) == SUMMARY
    usage = await store.get_usage("2026-10")
    assert usage.requests == 1 and usage.cost_usd == 0.0 and usage.input_tokens > 0


# --- the LLM over the backend ------------------------------------------------------------------


async def test_summarise_line_truncates_to_max_chars(store: Store, clock: FakeClock) -> None:
    long = "Summary: " + " ".join(["слово"] * 60) + "\n\nsecond paragraph the model added"
    rec = Recorder([ok(openai_answer(long))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    line = await llm.summarise_line(POST, 80, source="Kun.uz")
    assert line is not None and len(line) <= 80 and line.endswith("…")
    assert not line.startswith("Summary") and "\n" not in line
    body = rec.body()
    assert body["messages"][0]["content"] == prompts.SUMMARY_SYSTEM.format(max_chars=80)
    assert body["messages"][1]["content"] == f'<post source="Kun.uz">\n{POST}\n</post>'


async def test_empty_answer_is_none_and_still_settled(store: Store, clock: FakeClock) -> None:
    rec = Recorder([ok(openai_answer("", usage={"prompt_tokens": 50, "completion_tokens": 200}))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock, price=luna_price()), concurrency=4)
    assert await llm.summarise_line(POST, 180) is None
    usage = await store.get_usage("2026-10")
    assert usage.output_tokens == 200  # billed even though empty


async def test_name_topic_parses_sloppy_json_and_retries_once(
    store: Store, clock: FakeClock
) -> None:
    sloppy = 'Sure! Here you go:\n```json\n{"name": "Crypto & Exchanges", "description": "Posts about exchanges and coins."}\n```'  # noqa: E501
    rec = Recorder([ok(openai_answer(sloppy))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    assert await llm.name_topic(["p1", "p2"], existing=["Fintech"], category_label="Crypto") == (
        TopicName("Crypto & Exchanges", "Posts about exchanges and coins.")
    )
    user = rec.body()["messages"][1]["content"]
    assert user.startswith(
        "Existing topics: Fintech\nSuggested category from the classifier: Crypto"
    )
    assert "<post>p1</post>\n<post>p2</post>" in user
    # garbage, then a valid answer on the single retry
    rec = Recorder(
        [ok(openai_answer("no json here")), ok(openai_answer('{"name":"X","description":"y"}'))]
    )
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    assert await llm.name_topic(["p"]) == TopicName("X", "y")
    assert len(rec.requests) == 2
    assert rec.body()["messages"][1]["content"].endswith(
        f"no json here\n{prompts.TOPIC_RETRY_SUFFIX}"
    )
    # garbage twice -> None (the caller falls back to the category label)
    rec = Recorder([ok(openai_answer("{broken"))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    assert await llm.name_topic(["p"]) is None
    assert len(rec.requests) == 2


def test_parse_topic_name_validation() -> None:
    assert (
        prompts.parse_topic_name('{"name": "Sport", "description": "d"}', existing=["sport"])
        is None
    )
    assert prompts.parse_topic_name('{"name": "", "description": "d"}') is None
    assert prompts.parse_topic_name('{"name": "x" * 1, "description": 5}') is None
    assert prompts.parse_topic_name('{"name": "' + "a" * 41 + '", "description": "d"}') is None
    assert prompts.parse_topic_name('{"name": " Real  Estate ", "description": "d"}') == TopicName(
        "Real Estate", "d"
    )
    assert prompts.parse_topic_name('<think>x</think>{"name":"A","description":"b"}') == TopicName(
        "A", "b"
    )


async def test_second_opinion_yes_no_and_prompt(
    store: Store, clock: FakeClock, rt: Runtime
) -> None:
    topic = Topic(
        id=1, key="ml", name="ML & AI", description="Machine learning.", created_at=clock.now()
    )
    other = Topic(id=2, key="sci", name="Science", description=None, created_at=clock.now())
    rec = Recorder([ok(openai_answer("Yes.")), ok(openai_answer("NO")), ok(openai_answer("Maybe"))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    assert await llm.second_opinion(POST, topic, competing=[other]) is True
    assert rec.body()["max_completion_tokens"] == prompts.OPINION_MAX_TOKENS
    assert rec.body()["messages"][0]["content"] == prompts.OPINION_SYSTEM
    assert rec.body()["messages"][1]["content"] == (
        'Topic "ML & AI": Machine learning.\n'
        'Other topics that compete for this post: "Science": \n'
        f"<post>{POST}</post>"
    )
    assert await llm.second_opinion(POST, topic) is False
    assert await llm.second_opinion(POST, topic) is None


async def test_disabled_llm(rt: Runtime) -> None:
    llm = DisabledLLM()
    assert isinstance(llm, contracts.LLM) and llm.enabled is False
    assert await llm.summarise_line(POST, 100) is None
    assert await llm.name_topic([POST]) is None
    topic = Topic(id=1, key="k", name="K", created_at=rt.clock.now())
    assert await llm.second_opinion(POST, topic) is None
    assert isinstance(factory.make_llm(rt.settings, rt.store), DisabledLLM)


async def test_factory_builds_a_provider_llm_and_attaches_the_notifier(rt: Runtime) -> None:
    settings = rt.settings.model_copy(deep=True)
    settings.llm = cfg(
        mode="provider", provider="anthropic", model="claude-haiku-4-5", api_key="k",
        monthly_cap_usd=2.0, timeout_seconds=90,
    )  # fmt: skip
    llm = factory.make_llm(settings, rt.store, clock=rt.clock)
    assert isinstance(llm, ChatLLM) and isinstance(llm.backend, AnthropicBackend)
    assert llm.backend.read_timeout == 90.0 and llm.budget.cap_usd == 2.0
    assert llm._semaphore._value == 4 and llm.budget.notifier is None
    factory.attach_notifier(llm, rt.notifier)
    assert llm.budget.notifier is rt.notifier
    price = await llm.budget._price_lookup(None)
    assert price == registry.PRICES[("anthropic", "claude-haiku-4-5")][0]
    await llm.aclose()


# --- the registry ------------------------------------------------------------------------------


def test_providers_short_lists_and_privacy_lines() -> None:
    infos = registry.providers()
    assert [p.key for p in infos] == ["anthropic", "openai", "google", "mistral", "openrouter"]
    assert all(isinstance(p, ProviderInfo) for p in infos)
    by_key = {p.key: p for p in infos}
    assert by_key["anthropic"].models == ["claude-haiku-4-5", "claude-sonnet-5-5"]
    assert by_key["openai"].models == ["gpt-6-luna", "gpt-6-sol"]
    assert by_key["google"].models == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert by_key["mistral"].models == [
        "mistral-small-latest",
        "ministral-8b-2512",
        "mistral-medium-3-5",
    ]
    assert by_key["openrouter"].models == [
        "openai/gpt-6-luna",
        "mistralai/mistral-small-2603",
        "google/gemini-3.5-flash-lite",
        "anthropic/claude-sonnet-5.5",
    ]
    line = by_key["openai"].privacy_line
    assert line.startswith("Sent to OpenAI: the text of posts picked for a digest")
    assert "{test_post}" in line and "never chat names or identifiers" in line
    assert "{base_url}" in registry.PRIVACY_SELFHOSTED
    assert registry.PRICES_LAST_VERIFIED == date(2026, 10, 6)
    assert not registry.prices_stale(date(2026, 12, 1)) and registry.prices_stale(date(2027, 2, 1))


def test_price_table_lookup_rules() -> None:
    today = date(2026, 10, 6)
    assert (
        registry.price_for("anthropic", "claude-haiku-4-5", today=today).input_usd_per_mtok == 1.0
    )
    assert registry.price_for("openai", "gpt-6-luna", today=today) == Price(
        0.10, 0.50, "https://developers.openai.com/api/docs/pricing"
    )
    assert registry.price_for("google", "gemini-3.8-flash", today=today).input_usd_per_mtok == 0.75
    assert (
        registry.price_for("google", "gemini-3.8-flash", today=date(2027, 1, 1)).input_usd_per_mtok
        == 1.5
    )
    assert (
        registry.price_for("mistral", "mistral-small-latest", today=today).output_usd_per_mtok
        == 0.60
    )
    assert (
        registry.price_for("mistral", "mistral-medium-latest", today=today).output_usd_per_mtok
        == 7.5
    )
    assert (
        registry.price_for(
            "mistral", "fresh-2701", today=today, aliases={"fresh-2701": "mistral-large-2512"}
        ).input_usd_per_mtok
        == 0.5
    )
    unknown = registry.price_for("openai", "gpt-9-mystery", today=today)
    assert (unknown.input_usd_per_mtok, unknown.output_usd_per_mtok) == (2.0, 10.0)
    assert "unknown model" in unknown.note
    assert registry.price_for("selfhosted", "anything", today=today) == budget.ZERO_PRICE
    nothing = registry.price_for("openrouter", "x/y", today=today)
    assert (nothing.input_usd_per_mtok, nothing.output_usd_per_mtok) == (4.0, 20.0)


async def test_openrouter_catalogue_refresh_and_fallback(clock: FakeClock) -> None:
    catalogue = registry.OpenRouterCatalogue(clock)
    assert catalogue.stale()
    assert not await catalogue.refresh(client_for(Recorder([ok({"error": "down"}, 503)])))
    assert catalogue.entries == {}
    listing = {"data": [{"id": "a/b", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}
    rec = Recorder([ok(listing)])
    assert await catalogue.refresh(client_for(rec))
    assert str(rec.requests[0].url) == "https://openrouter.ai/api/v1/models"
    assert "authorization" not in rec.requests[0].headers  # the catalogue needs no key
    assert not catalogue.stale()
    assert registry.price_for(
        "openrouter", "a/b", today=clock.now().date(), catalogue=catalogue
    ) == Price(1.0, 2.0, "https://openrouter.ai/api/v1/models")
    rec2 = Recorder([ok({"data": []})])
    await catalogue.ensure_fresh(client_for(rec2))
    assert rec2.requests == []  # fresh: not fetched again
    clock.advance(2 * 24 * 3600)
    assert catalogue.stale()
    await catalogue.ensure_fresh(client_for(rec2))
    assert len(rec2.requests) == 1 and catalogue.price("a/b") is not None  # empty: old copy kept


async def test_validate_key_and_list_models_per_provider() -> None:
    rec = Recorder([ok({"data": [{"id": "gpt-6-luna"}, {"id": "gpt-6-sol"}]})])
    check = await registry.validate_key("openai", "sk", client=client_for(rec))
    assert check.models == ["gpt-6-luna", "gpt-6-sol"]
    assert str(rec.requests[0].url) == "https://api.openai.com/v1/models"
    assert rec.requests[0].headers["authorization"] == "Bearer sk"

    rec = Recorder([ok({"data": [{"id": "claude-haiku-4-5"}], "has_more": False})])
    assert await registry.list_models("anthropic", "ak", client=client_for(rec)) == [
        "claude-haiku-4-5"
    ]
    assert str(rec.requests[0].url) == "https://api.anthropic.com/v1/models"
    assert rec.requests[0].headers["x-api-key"] == "ak"
    assert rec.requests[0].headers["anthropic-version"] == "2023-06-01"

    rec = Recorder(
        [ok({"models": [{"name": "models/gemini-3.8-flash"}, {"name": "models/embedding-001"}]})]
    )
    assert await registry.list_models("google", "gk", client=client_for(rec)) == [
        "gemini-3.8-flash",
        "embedding-001",
    ]
    assert str(rec.requests[0].url) == "https://generativelanguage.googleapis.com/v1beta/models"
    assert rec.requests[0].headers["x-goog-api-key"] == "gk" and "key=" not in str(
        rec.requests[0].url
    )

    rec = Recorder(
        [ok({"data": [{"id": "mistral-small-2603", "aliases": ["mistral-small-latest"]}]})]
    )
    check = await registry.validate_key("mistral", "mk", client=client_for(rec))
    assert check.models == ["mistral-small-2603"]
    assert check.aliases == {"mistral-small-latest": "mistral-small-2603"}
    assert str(rec.requests[0].url) == "https://api.mistral.ai/v1/models"

    rec = Recorder(
        [
            ok({"data": {"label": "sk-or-v1-abc", "usage": 25.5, "limit": 100}}),
            ok(
                {
                    "data": [
                        {"id": "openai/gpt-6-luna", "pricing": {"prompt": "0", "completion": "0"}}
                    ]
                }
            ),
        ]
    )
    check = await registry.validate_key("openrouter", "ok", client=client_for(rec))
    assert str(rec.requests[0].url) == "https://openrouter.ai/api/v1/key"
    assert str(rec.requests[1].url) == "https://openrouter.ai/api/v1/models"
    assert (check.spend_usd, check.limit_usd, check.models) == (25.5, 100.0, ["openai/gpt-6-luna"])

    rec = Recorder([ok({"data": [{"id": "gemma4:12b", "owned_by": "library"}]})])
    check = await registry.validate_key(
        "selfhosted", "tok", base_url="localhost:11434", client=client_for(rec)
    )
    assert check.models == ["gemma4:12b"]
    assert str(rec.requests[0].url) == "http://localhost:11434/v1/models"
    assert rec.requests[0].headers["authorization"] == "Bearer tok"

    rec = Recorder([ok({"error": {"message": "Invalid API Key", "code": 401}}, 401)])
    with pytest.raises(LLMAuthError):
        await registry.validate_key(
            "selfhosted", "", base_url="http://localhost:8080/v1", client=client_for(rec)
        )
    assert "authorization" not in rec.requests[0].headers
    rec = Recorder([ok({}, 404)])
    assert (
        await registry.validate_key(
            "selfhosted", "", base_url="http://proxy/v1", client=client_for(rec)
        )
    ).models == []

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailableError):
        await registry.validate_key(
            "selfhosted", "", base_url="http://down:1/v1", client=client_for(refused)
        )
    with pytest.raises(LLMAuthError):
        await registry.validate_key("openai", "bad", client=client_for(Recorder([ok({}, 401)])))
    with pytest.raises(LLMRequestError):
        await registry.validate_key("nope", "k", client=client_for(Recorder([ok({})])))


async def test_test_prompt_times_the_call(store: Store, clock: FakeClock) -> None:
    rec = Recorder([ok(openai_answer(SUMMARY))])
    backend = OpenAIChatBackend(
        "gpt-6-luna", flavour="openai", base_url="https://api.openai.com/v1", api_key="k",
        client=client_for(rec),
    )  # fmt: skip
    llm = ChatLLM(backend, make_budget(store, clock), concurrency=4)
    ticks = iter([10.0, 12.5])
    line, seconds = await registry.test_prompt(llm, POST, 180, timer=lambda: next(ticks))
    assert line == SUMMARY and seconds == 2.5
    line, seconds = await registry.test_prompt(DisabledLLM(), POST, 180)
    assert line is None and seconds >= 0.0
