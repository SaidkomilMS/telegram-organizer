"""The wire dialects and the provider presets (DESIGN §15 "Language models", research §2–§4).

Three request shapes cover every supported endpoint: OpenAI Chat Completions (OpenAI, Mistral,
OpenRouter and every self-hosted server, each with its own parameter rules), Anthropic
Messages, and Google's native ``generateContent``. Reasoning is forced to its floor on every
call because a one-sentence answer gains nothing from thinking and a thinking model can spend
the whole output budget on it; the parsers tolerate chunk-list content, inline ``<think>``
blocks, reasoning fields, refusals and missing usage because real servers do all of that.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from tg_curator.config import LlmSettings
from tg_curator.llm import prompts
from tg_curator.llm.base import (
    PROVIDER_READ_TIMEOUT,
    SELFHOSTED_READ_TIMEOUT,
    HttpBackend,
    LLMRequestError,
    Sleep,
    Usage,
)

Dialect = Literal["openai", "anthropic", "google"]
Flavour = Literal["openai", "mistral", "openrouter", "selfhosted"]

TEMPERATURE = 0.2
GEMINI_THINKING_HEADROOM = 1024
"""Extra ``maxOutputTokens`` on Gemini: thinking tokens count toward the cap and a tight cap
yields an empty, still-billed answer."""
OPENROUTER_REASONING_HEADROOM = 1024
"""Extra ``max_tokens`` on OpenRouter models whose reasoning is mandatory: reasoning tokens
count against ``max_tokens`` (research §2.2/§3.5), and Anthropic models through OpenRouter
reserve at least 1024 thinking tokens, which ``max_tokens`` must strictly exceed."""
OPENAI_MIN_LOW_EFFORT_MODELS = ("gpt-6.1-sol", "gpt-6-astra")
"""OpenAI models that reject ``reasoning_effort: "none"`` (minimum ``low``, research §3.2);
with any effort other than ``none`` OpenAI also rejects ``temperature``."""
ANTHROPIC_VERSION = "2023-06-01"
OPENROUTER_TITLE = "tg-curator"
"""OpenRouter app attribution: the generic program name only. No HTTP-Referer is sent, so
requests never point at any particular repository, author or server (spec: open-source hygiene)."""

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EffortLookup = Callable[[str], str]
"""OpenRouter: the lowest reasoning effort a model accepts (``"none"`` unless mandatory)."""


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    base_url: str
    dialect: Dialect
    flavour: Flavour | None
    models: tuple[str, ...]  # the short list, cheapest first


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        "anthropic",
        "Anthropic",
        "https://api.anthropic.com",
        "anthropic",
        None,
        ("claude-haiku-4-5", "claude-sonnet-5-5"),
    ),
    "openai": ProviderSpec(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        "openai",
        "openai",
        ("gpt-6-luna", "gpt-6-sol"),
    ),
    "google": ProviderSpec(
        "google",
        "Google",
        "https://generativelanguage.googleapis.com/v1beta",
        "google",
        None,
        ("gemini-3.5-flash-lite", "gemini-3.8-flash"),
    ),
    "mistral": ProviderSpec(
        "mistral",
        "Mistral",
        "https://api.mistral.ai/v1",
        "openai",
        "mistral",
        ("mistral-small-latest", "ministral-8b-2512", "mistral-medium-3-5"),
    ),
    "openrouter": ProviderSpec(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "openai",
        "openrouter",
        (
            "openai/gpt-6-luna",
            "mistralai/mistral-small-2603",
            "google/gemini-3.5-flash-lite",
            "anthropic/claude-sonnet-5.5",
        ),
    ),
}


@dataclass(frozen=True)
class SelfHostedPreset:
    key: str
    label: str
    base_url: str
    key_optional: bool  # a token may be required by the server's configuration
    key_ignored: bool  # any bearer value is accepted and ignored (Ollama)
    reasoning_field: str  # where the server puts thinking it did not suppress
    health_path: str | None  # a public liveness endpoint outside /v1, when one exists


SELFHOSTED_PRESETS: tuple[SelfHostedPreset, ...] = (
    SelfHostedPreset(
        "ollama", "Ollama", "http://localhost:11434/v1", True, True, "reasoning", None
    ),
    SelfHostedPreset("vllm", "vLLM", "http://localhost:8000/v1", True, False, "reasoning", None),
    SelfHostedPreset(
        "lmstudio", "LM Studio", "http://localhost:1234/v1", True, False, "reasoning_content", None
    ),
    SelfHostedPreset(
        "llamacpp",
        "llama.cpp",
        "http://localhost:8080/v1",
        True,
        False,
        "reasoning_content",
        "/health",
    ),
)


def normalise_base_url(address: str) -> str:
    """``host:port`` / ``http://host:port`` / with or without ``/v1`` -> ``http://host:port/v1``.

    The scheme typed by the user is kept (never downgraded); a missing one is plain http,
    which is what every local server speaks.
    """
    raw = address.strip().rstrip("/")
    if not raw:
        raise LLMRequestError("the server address is empty")
    if "://" not in raw:
        raw = "http://" + raw
    if not raw.endswith("/v1"):
        raw += "/v1"
    return raw


_SNAPSHOT_DATE_RE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")


def undated_model_id(model: str) -> str:
    """``model`` without a trailing snapshot date: the id a server answers with is often the
    resolved snapshot (``claude-haiku-4-5-20251001``, ``gpt-6-luna-2026-05-18``) of the id that
    was asked for. Mistral's ``-2603``-style version suffixes are left alone."""
    return _SNAPSHOT_DATE_RE.sub("", model)


# --- the OpenAI Chat Completions dialect -----------------------------------------------------


class OpenAIChatBackend(HttpBackend):
    """OpenAI, Mistral, OpenRouter and self-hosted servers; ``flavour`` picks the rules."""

    def __init__(
        self,
        model: str,
        *,
        flavour: Flavour,
        base_url: str,
        api_key: str,
        provider: str | None = None,
        effort_lookup: EffortLookup | None = None,
        client: httpx.AsyncClient | None = None,
        read_timeout: float | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        selfhosted = flavour == "selfhosted"
        timeout = read_timeout or (SELFHOSTED_READ_TIMEOUT if selfhosted else PROVIDER_READ_TIMEOUT)
        super().__init__(model, client=client, read_timeout=timeout, sleep=sleep)
        self.provider = provider or flavour
        self.flavour = flavour
        self.selfhosted = selfhosted
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._effort_lookup = effort_lookup

    def _openrouter_effort(self) -> str:
        """The lowest effort the OpenRouter model accepts (``"none"`` unless mandatory)."""
        if self.flavour != "openrouter" or self._effort_lookup is None:
            return "none"
        return self._effort_lookup(self.model)

    def output_budget(self, max_tokens: int) -> int:
        # Mandatory reasoning on OpenRouter cannot be switched off and counts against
        # ``max_tokens``: without headroom a 5-token second opinion comes back empty, billed.
        if self._openrouter_effort() != "none":
            return max_tokens + OPENROUTER_REASONING_HEADROOM
        return max_tokens

    def _build(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
        }
        if self.flavour == "openai":
            body["max_completion_tokens"] = max_tokens
            if undated_model_id(self.model) in OPENAI_MIN_LOW_EFFORT_MODELS:
                body["reasoning_effort"] = "low"
                del body["temperature"]
            else:
                body["reasoning_effort"] = "none"
        elif self.flavour == "mistral":
            body["max_tokens"] = max_tokens
            body["reasoning_effort"] = "none"
        elif self.flavour == "openrouter":
            effort = self._openrouter_effort()
            body["max_tokens"] = self.output_budget(max_tokens)
            body["reasoning"] = {"effort": effort, "exclude": True}
        else:  # selfhosted
            body["max_tokens"] = max_tokens
            body["reasoning_effort"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode and not self.selfhosted:  # many local servers reject response_format
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.flavour == "openrouter":
            headers["X-OpenRouter-Title"] = OPENROUTER_TITLE
        return f"{self._base_url}/chat/completions", headers, body

    def _parse(self, body: dict[str, Any]) -> tuple[str, Usage]:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMRequestError(f"{self.provider}: answer without choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        # ``reasoning`` / ``reasoning_content`` fields are ignored on purpose; an empty content
        # with ``finish_reason: "length"`` (reasoning ate the budget) is simply an empty answer.
        text = prompts.strip_thinking(_join_content(message.get("content"))).strip()
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
        cost = usage.get("cost") if self.flavour == "openrouter" else None
        return text, Usage(
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            reasoning_tokens=int(reasoning or 0),
            reported_cost_usd=float(cost) if isinstance(cost, int | float) else None,
            model=body.get("model") if isinstance(body.get("model"), str) else None,
        )


def _join_content(content: Any) -> str:
    """A string, or Mistral-style chunks whose ``text`` parts are joined (thinking skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c["text"]
            for c in content
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str)
        ]
        return "".join(parts)
    return ""


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


# --- Anthropic Messages ----------------------------------------------------------------------

ANTHROPIC_NO_THINKING_MODELS = ("claude-haiku-4-5",)
"""Models where ``thinking`` is simply omitted and ``temperature`` is still accepted; every
other Claude is treated like Sonnet 5.5 (``between_tools`` + effort low, no sampling params)."""


class AnthropicBackend(HttpBackend):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = PROVIDERS["anthropic"].base_url,
        client: httpx.AsyncClient | None = None,
        read_timeout: float = PROVIDER_READ_TIMEOUT,
        sleep: Sleep | None = None,
    ) -> None:
        super().__init__(model, client=client, read_timeout=read_timeout, sleep=sleep)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _build(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if self.model.startswith(ANTHROPIC_NO_THINKING_MODELS):
            body["temperature"] = TEMPERATURE
        else:
            body["thinking"] = {"type": "between_tools"}
            output_config["effort"] = "low"
        if json_mode:
            output_config["format"] = {"type": "json_schema", "schema": prompts.TOPIC_JSON_SCHEMA}
        if output_config:
            body["output_config"] = output_config
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        return f"{self._base_url}/v1/messages", headers, body

    def _parse(self, body: dict[str, Any]) -> tuple[str, Usage]:
        content = body.get("content")
        if not isinstance(content, list):
            raise LLMRequestError("anthropic: answer without content")
        text = "" if body.get("stop_reason") == "refusal" else _join_content(content).strip()
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        details = usage.get("output_tokens_details")
        thinking = details.get("thinking_tokens", 0) if isinstance(details, dict) else 0
        return text, Usage(
            input_tokens=_int_or_none(usage.get("input_tokens")),
            output_tokens=_int_or_none(usage.get("output_tokens")),
            reasoning_tokens=int(thinking or 0),
            model=body.get("model") if isinstance(body.get("model"), str) else None,
        )


# --- Google generateContent ------------------------------------------------------------------


def gemini_thinking_level(model: str) -> str:
    """The floor per model family: Flash-Lite allows MINIMAL, the Flash models stop at LOW."""
    return "MINIMAL" if "flash-lite" in model else "LOW"


class GoogleBackend(HttpBackend):
    provider = "google"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = PROVIDERS["google"].base_url,
        client: httpx.AsyncClient | None = None,
        read_timeout: float = PROVIDER_READ_TIMEOUT,
        sleep: Sleep | None = None,
    ) -> None:
        super().__init__(model, client=client, read_timeout=read_timeout, sleep=sleep)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def output_budget(self, max_tokens: int) -> int:
        return max_tokens + GEMINI_THINKING_HEADROOM

    def _build(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        config: dict[str, Any] = {
            "maxOutputTokens": self.output_budget(max_tokens),
            "thinkingConfig": {"thinkingLevel": gemini_thinking_level(self.model)},
        }
        if json_mode:
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        return f"{self._base_url}/models/{self.model}:generateContent", headers, body

    def _parse(self, body: dict[str, Any]) -> tuple[str, Usage]:
        text = ""
        feedback = body.get("promptFeedback")
        blocked = isinstance(feedback, dict) and feedback.get("blockReason")
        candidates = body.get("candidates")
        if not blocked and isinstance(candidates, list) and candidates:
            candidate = candidates[0] if isinstance(candidates[0], dict) else {}
            if candidate.get("finishReason") != "SAFETY":
                content = (
                    candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
                )
                parts = content.get("parts") if isinstance(content.get("parts"), list) else []
                text = "".join(
                    p["text"]
                    for p in parts
                    if isinstance(p, dict)
                    and isinstance(p.get("text"), str)
                    and not p.get("thought")
                ).strip()
        elif not blocked and "candidates" not in body:
            raise LLMRequestError("google: answer without candidates")
        meta = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        thoughts = int(meta.get("thoughtsTokenCount") or 0)
        answer = _int_or_none(meta.get("candidatesTokenCount"))
        return text, Usage(
            input_tokens=_int_or_none(meta.get("promptTokenCount")),
            output_tokens=None if answer is None else answer + thoughts,
            reasoning_tokens=thoughts,
            model=body.get("modelVersion") if isinstance(body.get("modelVersion"), str) else None,
        )


# --- construction ----------------------------------------------------------------------------


def build_backend(
    cfg: LlmSettings,
    *,
    client: httpx.AsyncClient | None = None,
    effort_lookup: EffortLookup | None = None,
    sleep: Sleep | None = None,
) -> HttpBackend:
    """The backend for a validated ``[llm]`` section (mode selfhosted or provider).

    The read timeout is the §15 floor for the mode (60 s providers, 180 s self-hosted) or
    ``llm.timeout_seconds`` when the user set it higher; a CPU-hosted model that needs minutes
    is the case the setting exists for, never a reason to time out sooner than the floor.
    """
    if cfg.mode == "selfhosted":
        return OpenAIChatBackend(
            cfg.model,
            flavour="selfhosted",
            base_url=normalise_base_url(cfg.base_url),
            api_key=cfg.api_key,
            provider="selfhosted",
            client=client,
            read_timeout=max(float(cfg.timeout_seconds), SELFHOSTED_READ_TIMEOUT),
            sleep=sleep,
        )
    if cfg.mode != "provider":
        raise LLMRequestError(f"llm.mode {cfg.mode!r} has no backend")
    spec = PROVIDERS.get(cfg.provider)
    if spec is None:
        raise LLMRequestError(
            f"llm.provider {cfg.provider!r} is unknown; one of {', '.join(PROVIDERS)}"
        )
    timeout = max(float(cfg.timeout_seconds), PROVIDER_READ_TIMEOUT)
    if spec.dialect == "anthropic":
        return AnthropicBackend(
            cfg.model, api_key=cfg.api_key, client=client, read_timeout=timeout, sleep=sleep
        )
    if spec.dialect == "google":
        return GoogleBackend(
            cfg.model, api_key=cfg.api_key, client=client, read_timeout=timeout, sleep=sleep
        )
    assert spec.flavour is not None
    return OpenAIChatBackend(
        cfg.model,
        flavour=spec.flavour,
        base_url=spec.base_url,
        api_key=cfg.api_key,
        provider=spec.key,
        effort_lookup=effort_lookup,
        client=client,
        read_timeout=timeout,
        sleep=sleep,
    )


def lowest_effort(supported: list[str] | tuple[str, ...], mandatory: bool) -> str:
    """OpenRouter: ``"none"`` unless reasoning is mandatory, then the lowest listed effort."""
    if not mandatory:
        return "none"
    ranked = [e for e in EFFORT_ORDER if e in supported and e != "none"]
    return ranked[0] if ranked else "low"
