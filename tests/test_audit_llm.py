"""Audit fixes in llm/: the connection test names the cause of a failure (S1) and a month closed
at the spending cap stays closed until it ends (TUN-7)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fakes import FakeBotGateway, FakeClock, make_bot_message, make_callback
from tests.test_bot_llm import (
    KEY,
    LONG_TEXT,
    Server,
    Ticker,
    app,  # noqa: F401 - fixture
    owner_texts,
    provider_index,
    stored_post,
)
from tests.test_cli import curator, fakes  # noqa: F401 - fixture
from tg_curator import service
from tg_curator.bot import llm as llm_mod
from tg_curator.bot.core import BotApp
from tg_curator.config import ENV_OVERRIDES, SettingsFile, write_template
from tg_curator.db.store import Store
from tg_curator.llm import base, budget, factory, registry
from tg_curator.llm.base import ChatLLM
from tg_curator.llm.budget import Budget, Price
from tg_curator.llm.providers import OpenAIChatBackend
from tg_curator.ml import models
from tg_curator.runtime import Runtime
from tg_curator.service import Doubles

POST = (
    "Центробанк Узбекистана сохранил основную ставку на уровне 14% годовых, сославшись на "
    "замедление инфляции до 8,9% в сентябре и стабильный курс сума."
)
SUMMARY = "ЦБ Узбекистана сохранил ставку 14% на фоне замедления инфляции до 8,9%."


def answer(text: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "model": "gpt-6-luna",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    )


async def no_sleep(_seconds: float) -> None:
    return None


async def luna(_model: str | None) -> Price:
    return Price(0.10, 0.50)


def chat_llm(
    store: Store,
    clock: FakeClock,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    cap: float = 0.0,
    selfhosted: bool = False,
) -> ChatLLM:
    backend = OpenAIChatBackend(
        "qwen3:8b" if selfhosted else "gpt-6-luna",
        flavour="selfhosted" if selfhosted else "openai",
        base_url="http://192.168.1.20:8000/v1" if selfhosted else "https://api.openai.com/v1",
        api_key="k",
        provider="selfhosted" if selfhosted else "openai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=no_sleep,
    )
    b = Budget(store, clock, cap_usd=cap, timezone="UTC", price_lookup=luna)
    return ChatLLM(backend, b, concurrency=1)


def status(http_status: int, message: str = "no", **error: Any) -> Callable[[httpx.Request], Any]:
    return lambda _r: httpx.Response(http_status, json={"error": {"message": message, **error}})


def raises(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise exc

    return handler


# --- S1: the test says which cause it was ---------------------------------------------------


@pytest.mark.parametrize(
    ("handler", "kind"),
    [
        (status(401, "invalid api key"), "auth"),
        (status(404, "The model `gpt-6-luna` does not exist"), "model"),
        (status(400, "Invalid model: gpt-6-luna"), "model"),
        (status(400, "max_tokens is too large"), "request"),
        (raises(httpx.ReadTimeout("slow")), "timeout"),
        (raises(httpx.ConnectError("refused")), "unreachable"),
        (status(503, "overloaded"), "unavailable"),
        (status(402, "no credits"), "credit"),
        (status(429, "q", code="insufficient_quota"), "credit"),
        (lambda _r: answer(""), "empty"),
    ],
)
async def test_probe_names_the_cause(
    store: Store,
    clock: FakeClock,
    handler: Callable[[httpx.Request], httpx.Response],
    kind: str,
) -> None:
    llm = chat_llm(store, clock, handler)
    ticks = iter([1.0, 3.0])
    outcome = await registry.probe(llm, POST, 180, timer=lambda: next(ticks))
    assert outcome.line is None and outcome.seconds == 2.0
    assert outcome.failure is not None and outcome.failure.kind == kind
    assert outcome.failure.provider == "OpenAI" and outcome.failure.model == "gpt-6-luna"
    assert "Bearer" not in outcome.failure.detail
    # the digest path is unchanged: the same failure is a quiet None
    assert await chat_llm(store, clock, handler).summarise_line(POST, 180) is None


async def test_probe_reports_the_cap_and_the_success(store: Store, clock: FakeClock) -> None:
    await store.add_usage("2026-10", 1, 0, 0, 1.0)
    capped = chat_llm(store, clock, lambda _r: answer(SUMMARY), cap=1.0)
    outcome = await registry.probe(capped, POST, 180)
    assert outcome.failure is not None and outcome.failure.kind == "cap"
    assert outcome.failure.cap_usd == 1.0

    good = await registry.probe(chat_llm(store, clock, lambda _r: answer(SUMMARY)), POST, 180)
    assert good.line == SUMMARY and good.failure is None
    line, _ = await registry.test_prompt(
        chat_llm(store, clock, lambda _r: answer(SUMMARY)), POST, 180
    )
    assert line == SUMMARY  # the contract's two-value form is unchanged


async def test_selfhosted_failures_name_the_server(store: Store, clock: FakeClock) -> None:
    llm = chat_llm(store, clock, raises(httpx.ReadTimeout("slow")), selfhosted=True)
    outcome = await registry.probe(llm, POST, 180)
    assert outcome.failure is not None
    assert outcome.failure.kind == "timeout" and outcome.failure.timeout_seconds == 180.0
    assert outcome.failure.provider == "192.168.1.20"
    t = translator()
    assert "No answer within 180 s" in llm_mod.failure_reason(t, outcome.failure)
    assert "llm.timeout_seconds" in llm_mod.failure_reason(t, outcome.failure)

    down = chat_llm(store, clock, raises(httpx.ConnectError("refused")), selfhosted=True)
    failure = (await registry.probe(down, POST, 180)).failure
    assert "Cannot reach http://192.168.1.20:8000/v1" in llm_mod.failure_reason(t, failure)


def translator() -> Callable[..., str]:
    from tg_curator.i18n import LOCALES_DIR, Translator

    return Translator(locales_dir=LOCALES_DIR).t


def test_every_failure_kind_has_a_message() -> None:
    t = translator()
    for kind, key in llm_mod.FAILURE_KEYS.items():
        text = llm_mod.failure_reason(
            t, registry.ProbeFailure(kind, provider="OpenAI", model="m<1>", detail="x")
        )
        assert text != key and "{" not in text, kind
        assert "m<1>" not in text  # values are escaped for the bot's HTML


class SlowServer(Server):
    """``/chat/completions`` times out once ``timeout`` is set."""

    timeout = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.timeout and request.url.path.endswith("/chat/completions"):
            self.requests.append(request)
            raise httpx.ReadTimeout("slow", request=request)
        return super().__call__(request)


@pytest.fixture
def server() -> SlowServer:
    return SlowServer()


@pytest.fixture
async def client(server: SlowServer) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as c:
        yield c


@pytest.fixture
def ticker() -> Ticker:
    return Ticker()


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.HttpBackend, "_backoff", lambda self, attempt: 0.0)


async def test_bot_names_an_unknown_model(
    app: BotApp,  # noqa: F811
    rt: Runtime,
    bot_gw: FakeBotGateway,
    server: SlowServer,
) -> None:
    await stored_post(rt, LONG_TEXT)
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    await bot_gw.say(make_bot_message(KEY, message_id=9))
    server.chat_status = 404

    await bot_gw.say(make_bot_message("gpt-6-nonexistent"))

    text = owner_texts(bot_gw)[-1]
    assert "nothing was saved" in text
    assert "gpt-6-nonexistent is not served by OpenAI. Type another model name." in text
    assert "service log" not in text and "refused" not in text
    assert rt.settings.llm.mode == "none"


async def test_bot_names_a_timeout_of_a_selfhosted_model(
    app: BotApp,  # noqa: F811
    rt: Runtime,
    bot_gw: FakeBotGateway,
    server: SlowServer,
) -> None:
    server.timeout = True
    await bot_gw.press(make_callback("lm:m:1"))
    await bot_gw.press(make_callback("lm:p:0"))
    await bot_gw.press(make_callback("lm:u"))
    await bot_gw.press(make_callback("lm:nt"))
    await bot_gw.press(make_callback("lm:md:0"))
    text = owner_texts(bot_gw)[-1]
    assert "nothing was saved" in text and "No answer within 180 s" in text
    assert "llm.timeout_seconds" in text
    assert rt.settings.llm.mode == "none"


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A CLI home with the template and the three [telegram] values (as in test_cli.py)."""
    for var in (*ENV_OVERRIDES, service.WAIT_ENV, "TG_CURATOR_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(models.FAKE_MODELS_ENV, "1")
    monkeypatch.setattr(models, "_embedder", None)
    cli_home = tmp_path / "cli-home"
    path = cli_home / "settings.toml"
    write_template(path)
    settings_file = SettingsFile(path, env={})
    settings_file.load_sync()

    def fill(doc: Any) -> None:
        doc["telegram"]["api_id"] = 12345
        doc["telegram"]["api_hash"] = "0123456789abcdef"
        doc["telegram"]["bot_token"] = "1:token"

    asyncio.run(settings_file.update(fill))
    return cli_home


def test_cli_names_a_refused_key(
    configured: Path,
    fakes: Doubles,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = factory.make_llm
    seen: list[str] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    def make_llm(settings: Any, store: Any, **kw: Any) -> Any:
        kw["client"] = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
        return real(settings, store, **kw)

    monkeypatch.setattr(factory, "make_llm", make_llm)
    args = ["--mode", "provider", "--provider", "openai", "--model", "gpt-6-luna", "--key-stdin"]
    result = curator(configured, "llm", *args, input="sk-wrong\n")
    assert result.exit_code == 1
    assert seen == ["/v1/chat/completions"]
    assert "nothing was saved" in result.output
    assert "OpenAI refused the key or token" in result.output
    assert "check the key, the model name and the address" not in result.output
    assert "sk-wrong" not in result.output


def test_cli_names_a_missing_model(
    configured: Path,
    fakes: Doubles,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = factory.make_llm

    def make_llm(settings: Any, store: Any, **kw: Any) -> Any:
        handler = status(404, 'model "qwen9" not found, try pulling it first')
        kw["client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return real(settings, store, **kw)

    monkeypatch.setattr(factory, "make_llm", make_llm)
    args = ["--mode", "selfhosted", "--base-url", "http://localhost:11434/v1", "--model", "qwen9"]
    result = curator(configured, "llm", *args, "--key-stdin", input="\n")
    assert result.exit_code == 1
    assert "qwen9 is not served by localhost. Type another model name." in result.output


# --- TUN-7: the month stays closed once the cap is reached -----------------------------------


class Notices:
    def __init__(self) -> None:
        self.caps: list[str] = []

    async def llm_cap_reached(self, month: str, cap_usd: float) -> None:
        self.caps.append(month)

    async def llm_out_of_credit(self, provider: str) -> None:
        raise AssertionError("not this notice")


def capped(store: Store, clock: FakeClock, cap: float, notices: Notices) -> Budget:
    async def dollar(_model: str | None) -> Price:
        return Price(1_000.0, 1_000.0)  # $0.001 per token

    return Budget(store, clock, cap_usd=cap, timezone="UTC", price_lookup=dollar, notifier=notices)


async def test_a_refusal_at_the_cap_closes_the_month(store: Store, clock: FakeClock) -> None:
    notices = Notices()
    await store.add_usage("2026-10", 3, 0, 0, 0.995)
    b = capped(store, clock, 1.0, notices)
    assert await b.reserve(100, 200) is None  # $0.30 would cross $1.00
    assert notices.caps == ["2026-10"]
    assert await b.exhausted()
    assert await b.reserve(1, 1) is None  # $0.002 would still fit, but the month is closed
    assert notices.caps == ["2026-10"]  # said once
    usage = await store.get_usage("2026-10")
    assert usage.cost_usd == pytest.approx(0.995) and usage.requests == 3

    # a restart (a new Budget with the same cap) keeps it closed
    again = capped(store, clock, 1.0, notices)
    assert await again.exhausted() and await again.reserve(1, 1) is None
    assert await store.kv_get(budget.CAP_CLOSED_KEY) == {"month": "2026-10", "cap_usd": 1.0}

    # raising the cap through /llm (a new Budget) opens the month again
    raised = capped(store, clock, 2.0, notices)
    assert not await raised.exhausted()
    assert await raised.reserve(1, 1) is not None

    # and the next month starts open
    clock.advance(31 * 24 * 3600)
    assert b.month() == "2026-11"
    assert not await b.exhausted() and await b.reserve(1, 1) is not None


async def test_a_settlement_that_reaches_the_cap_closes_the_month(
    store: Store, clock: FakeClock
) -> None:
    notices = Notices()
    b = capped(store, clock, 0.01, notices)
    res = await b.reserve(1, 1)
    assert res is not None
    await b.settle(res, input_tokens=6, output_tokens=5)  # $0.011 > $0.01
    assert await b.exhausted() and notices.caps == ["2026-10"]
    # a refund after a release would bring usage back under the cap; the month stays closed
    await store.add_usage("2026-10", 0, 0, 0, -0.005)
    assert await b.exhausted() and await b.reserve(1, 1) is None


async def test_no_model_lines_after_the_cap_notice(store: Store, clock: FakeClock) -> None:
    """End to end: a long post is refused at the cap, a short one later is not sent either."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return answer(SUMMARY)

    notices = Notices()
    await store.add_usage("2026-10", 1, 0, 0, 0.996)
    backend = OpenAIChatBackend(
        "gpt-6-luna",
        flavour="openai",
        base_url="https://api.openai.com/v1",
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=no_sleep,
    )

    async def pricey(_model: str | None) -> Price:
        return Price(2.5, 10.0)

    b = Budget(store, clock, cap_usd=1.0, timezone="UTC", price_lookup=pricey, notifier=notices)
    llm = ChatLLM(backend, b, concurrency=1)
    long_post = POST * 400
    assert await llm.summarise_line(long_post, 180) is None
    assert notices.caps == ["2026-10"] and calls == []
    assert await llm.summarise_line("Short post.", 180) is None
    assert calls == []
    assert await store.kv_get(budget.CAP_CLOSED_KEY) == {"month": "2026-10", "cap_usd": 1.0}
