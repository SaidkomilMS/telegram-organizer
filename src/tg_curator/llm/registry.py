"""What the ``/llm`` wizard and the CLI need to know about providers (DESIGN §8, §11.2, §15).

The short model lists, the privacy sentence per choice, the shipped price table with the date
it was last checked, OpenRouter's catalogue fetched at runtime (its prices move day to day and
are never shipped), the self-hosted presets, key validation, model listing and the one-post
connection test. Budgeting an unknown model at the provider's most expensive listed price is
deliberate: a wrong guess then errs on the side of stopping early.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from tg_curator.domain import ProviderInfo
from tg_curator.llm.base import (
    CONNECT_TIMEOUT,
    LLMAuthError,
    LLMBudgetError,
    LLMCreditError,
    LLMEmptyAnswerError,
    LLMError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    provider_label,
)
from tg_curator.llm.budget import ZERO_PRICE, Price
from tg_curator.llm.providers import (
    ANTHROPIC_VERSION,
    OPENROUTER_TITLE,
    PROVIDERS,
    SELFHOSTED_PRESETS,
    ProviderSpec,
    SelfHostedPreset,
    lowest_effort,
    normalise_base_url,
    undated_model_id,
)

if TYPE_CHECKING:
    from tg_curator.clock import Clock
    from tg_curator.contracts import LLM

log = logging.getLogger(__name__)

PRICES_LAST_VERIFIED = date(2026, 10, 6)
"""When the maintainers last read the vendor pricing pages; the wizard warns past 90 days."""
PRICES_STALE_DAYS = 90
PROBE_TIMEOUT = 10.0
CATALOGUE_MAX_AGE = timedelta(days=1)

_ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
_OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"
_GOOGLE_PRICING = "https://ai.google.dev/gemini-api/docs/pricing"
_MISTRAL_PRICING = "https://mistral.ai/pricing"

PRICES: dict[tuple[str, str], tuple[Price, ...]] = {
    ("anthropic", "claude-haiku-4-5"): (Price(1.00, 5.00, _ANTHROPIC_PRICING),),
    ("anthropic", "claude-sonnet-5-5"): (Price(2.00, 10.00, _ANTHROPIC_PRICING),),
    ("anthropic", "claude-sonnet-5"): (Price(2.00, 10.00, _ANTHROPIC_PRICING),),
    ("anthropic", "claude-opus-5-5"): (Price(4.00, 20.00, _ANTHROPIC_PRICING),),
    ("openai", "gpt-6-luna"): (Price(0.10, 0.50, _OPENAI_PRICING),),
    ("openai", "gpt-6-sol"): (Price(2.00, 10.00, _OPENAI_PRICING),),
    ("openai", "gpt-6.1-sol"): (Price(2.00, 10.00, _OPENAI_PRICING),),
    ("google", "gemini-3.5-flash-lite"): (Price(0.30, 2.50, _GOOGLE_PRICING),),
    ("google", "gemini-3.8-flash"): (
        Price(0.75, 3.75, _GOOGLE_PRICING),
        Price(1.50, 7.50, _GOOGLE_PRICING, effective_from=date(2027, 1, 1)),
    ),
    ("google", "gemini-3.1-flash-lite"): (
        Price(0.25, 1.50, _GOOGLE_PRICING, note="shutdown 2027-05-07"),
    ),
    ("mistral", "mistral-small-2603"): (Price(0.15, 0.60, _MISTRAL_PRICING),),
    ("mistral", "ministral-8b-2512"): (Price(0.15, 0.15, _MISTRAL_PRICING),),
    ("mistral", "mistral-large-2512"): (Price(0.50, 1.50, _MISTRAL_PRICING),),
    ("mistral", "mistral-medium-3-5"): (Price(1.50, 7.50, _MISTRAL_PRICING),),
}

ALIASES: dict[tuple[str, str], str] = {
    ("mistral", "mistral-small-latest"): "mistral-small-2603",
    ("mistral", "ministral-8b-latest"): "ministral-8b-2512",
    ("mistral", "mistral-large-latest"): "mistral-large-2512",
    ("mistral", "mistral-medium-latest"): "mistral-medium-3-5",
}
"""Static alias bindings (plausible, not officially published); ``validate_key`` re-resolves
them from Mistral's ``/v1/models`` and the settlement prices by the id the API answers with."""

PRIVACY_PROVIDER = (
    "Sent to {provider}: the text of posts picked for a digest, the example posts behind a "
    "proposed topic, one post for this test ({test_post}), and — only if you turn on second "
    "opinion — borderline posts. Nothing else, never chat names or identifiers."
)
PRIVACY_SELFHOSTED = (
    "Sent to your server at {base_url}: the text of posts picked for a digest, the example "
    "posts behind a proposed topic, one post for this test ({test_post}), and — only if you "
    "turn on second opinion — borderline posts. Nothing else, never chat names or identifiers."
)
PRIVACY_NONE = "Nothing leaves the server: digest lines are the first line of each post."
"""The three privacy sentences of §11.2; ``{test_post}`` is "<source, date>" of the post the
wizard will send, filled in by the bot."""


def providers() -> list[ProviderInfo]:
    """The provider choices of the wizard, in the order they are shown (§8 module function)."""
    return [
        ProviderInfo(
            key=spec.key,
            label=spec.label,
            models=list(spec.models),
            privacy_line=PRIVACY_PROVIDER.format(provider=spec.label, test_post="{test_post}"),
        )
        for spec in PROVIDERS.values()
    ]


def selfhosted_presets() -> list[SelfHostedPreset]:
    return list(SELFHOSTED_PRESETS)


def prices_stale(today: date) -> bool:
    return (today - PRICES_LAST_VERIFIED).days > PRICES_STALE_DAYS


# --- prices ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueEntry:
    price: Price
    reasoning_mandatory: bool
    supported_efforts: tuple[str, ...]


class OpenRouterCatalogue:
    """``GET /api/v1/models`` held in memory, refreshed at most daily, kept on failure.

    Prices are strings in USD per token; reasoning flags say whether ``effort: "none"`` would
    be rejected. The catalogue needs no key, so a stale or missing copy only costs a more
    conservative reservation, never a failed request.
    """

    def __init__(self, clock: Clock, *, base_url: str = PROVIDERS["openrouter"].base_url) -> None:
        self._clock = clock
        self._url = f"{base_url.rstrip('/')}/models"
        self._entries: dict[str, CatalogueEntry] = {}
        self.fetched_at: datetime | None = None

    @property
    def entries(self) -> dict[str, CatalogueEntry]:
        return dict(self._entries)

    def stale(self) -> bool:
        return self.fetched_at is None or self._clock.now() - self.fetched_at > CATALOGUE_MAX_AGE

    async def refresh(self, client: httpx.AsyncClient) -> bool:
        """Fetch the catalogue; ``False`` (and the old copy kept) when it cannot be reached."""
        try:
            response = await client.get(
                self._url,
                headers={"X-OpenRouter-Title": OPENROUTER_TITLE},
                timeout=httpx.Timeout(PROBE_TIMEOUT, connect=CONNECT_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json().get("data")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            log.warning("llm: OpenRouter model catalogue not refreshed: %s", exc)
            return False
        entries = {}
        for item in data if isinstance(data, list) else []:
            entry = _catalogue_entry(item)
            if entry is not None:
                entries[item["id"]] = entry
        if not entries:
            log.warning("llm: OpenRouter model catalogue came back empty; keeping the old one")
            return False
        self._entries = entries
        self.fetched_at = self._clock.now()
        return True

    async def ensure_fresh(self, client: httpx.AsyncClient) -> None:
        if self.stale():
            await self.refresh(client)

    def price(self, model: str) -> Price | None:
        entry = self._entries.get(model)
        return None if entry is None else entry.price

    def effort(self, model: str) -> str:
        entry = self._entries.get(model)
        if entry is None:
            return "none"
        return lowest_effort(entry.supported_efforts, entry.reasoning_mandatory)


def _catalogue_entry(item: Any) -> CatalogueEntry | None:
    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
        return None
    pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
    try:
        prompt = float(pricing.get("prompt", 0)) * 1e6
        completion = float(pricing.get("completion", 0)) * 1e6
    except (TypeError, ValueError):
        return None
    reasoning = item.get("reasoning") if isinstance(item.get("reasoning"), dict) else {}
    efforts = reasoning.get("supported_efforts")
    return CatalogueEntry(
        Price(prompt, completion, "https://openrouter.ai/api/v1/models"),
        bool(reasoning.get("mandatory")),
        tuple(e for e in efforts if isinstance(e, str)) if isinstance(efforts, list) else (),
    )


def known_price(
    provider: str,
    model: str,
    *,
    today: date,
    aliases: dict[str, str] | None = None,
    catalogue: OpenRouterCatalogue | None = None,
) -> Price | None:
    """The listed price of ``model``, or ``None`` when nothing lists it (no flagship guess).

    Exact row valid today -> alias (the runtime ``aliases``, then the static ones) ->
    OpenRouter catalogue; self-hosted is always zero.
    """
    if provider == "selfhosted":
        return ZERO_PRICE
    rows = PRICES.get((provider, model))
    if rows is None:
        target = (aliases or {}).get(model) or ALIASES.get((provider, model))
        if target is not None:
            rows = PRICES.get((provider, target))
    if rows:
        valid = [p for p in rows if p.effective_from is None or p.effective_from <= today]
        return valid[-1] if valid else rows[0]
    if provider == "openrouter" and catalogue is not None:
        return catalogue.price(model)
    return None


def price_for(
    provider: str,
    model: str,
    *,
    today: date,
    aliases: dict[str, str] | None = None,
    catalogue: OpenRouterCatalogue | None = None,
) -> Price:
    """The price to budget ``model`` at (research §7.1 lookup order).

    ``known_price`` (exact row, alias, OpenRouter catalogue, self-hosted zero), else the
    provider's most expensive listed input and output prices (an id the table does not know,
    so the guess errs towards stopping early).
    """
    price = known_price(provider, model, today=today, aliases=aliases, catalogue=catalogue)
    return price if price is not None else most_expensive(provider, today)


def settlement_price(
    provider: str,
    reported: str | None,
    configured: str,
    *,
    today: date,
    aliases: dict[str, str] | None = None,
    catalogue: OpenRouterCatalogue | None = None,
) -> Price:
    """The price of a finished request whose answer named ``reported`` as its model.

    The reported id wins when it is listed (a Mistral ``-latest`` alias resolves to a dated
    id), then the same id without a snapshot date (Anthropic and OpenAI answer with the
    resolved snapshot, ``claude-haiku-4-5-20251001``), then the configured id — which still
    lands on the flagship guess when the table does not know it either.
    """
    for candidate in dict.fromkeys(filter(None, (reported, undated_model_id(reported or "")))):
        price = known_price(provider, candidate, today=today, aliases=aliases, catalogue=catalogue)
        if price is not None:
            return price
    return price_for(provider, configured, today=today, aliases=aliases, catalogue=catalogue)


def most_expensive(provider: str, today: date) -> Price:
    """The flagship guess for an unknown id: the provider's highest input and output prices
    (OpenRouter has no shipped rows, so it is budgeted at the dearest model of the table)."""
    candidates = [
        price_for(p, m, today=today)
        for (p, m) in PRICES
        if provider == "openrouter" or p == provider
    ]
    if not candidates:
        candidates = [price_for(p, m, today=today) for (p, m) in PRICES]
    return Price(
        max(c.input_usd_per_mtok for c in candidates),
        max(c.output_usd_per_mtok for c in candidates),
        note=f"unknown model: budgeted at the most expensive {provider} price",
    )


# --- key validation and model listing --------------------------------------------------------


@dataclass(frozen=True)
class KeyCheck:
    """What a successful probe learnt: the ids, Mistral alias bindings, OpenRouter spend."""

    models: list[str]
    aliases: dict[str, str]
    spend_usd: float | None = None
    limit_usd: float | None = None


async def validate_key(
    provider: str,
    api_key: str,
    *,
    base_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> KeyCheck:
    """Probe the key (or the self-hosted address) with the cheapest read-only call.

    Raises ``LLMAuthError`` for a refused key, ``LLMUnavailableError`` when nothing answers,
    ``LLMRequestError`` for anything else; a self-hosted ``/models`` that is simply missing
    (404 on some proxies) passes with an empty list.
    """
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        if provider == "selfhosted":
            return await _probe_selfhosted(client, normalise_base_url(base_url), api_key)
        spec = PROVIDERS.get(provider)
        if spec is None:
            raise LLMRequestError(f"unknown provider {provider!r}")
        if provider == "openrouter":
            return await _probe_openrouter(client, spec, api_key)
        return await _probe_models(client, spec, api_key)
    finally:
        if own:
            await client.aclose()


async def list_models(
    provider: str,
    api_key: str,
    *,
    base_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """The model ids the key (or server) can use; same errors as ``validate_key``."""
    return (await validate_key(provider, api_key, base_url=base_url, client=client)).models


def _headers(spec: ProviderSpec, api_key: str) -> dict[str, str]:
    if spec.dialect == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    if spec.dialect == "google":
        return {"x-goog-api-key": api_key}
    headers = {"Authorization": f"Bearer {api_key}"}
    if spec.key == "openrouter":
        headers["X-OpenRouter-Title"] = OPENROUTER_TITLE
    return headers


async def _get(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> httpx.Response:
    try:
        return await client.get(
            url, headers=headers, timeout=httpx.Timeout(PROBE_TIMEOUT, connect=CONNECT_TIMEOUT)
        )
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(
            f"no answer from {_host(url)} ({exc.__class__.__name__})"
        ) from None
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(f"cannot reach {_host(url)} ({exc.__class__.__name__})") from None


def _check(response: httpx.Response, what: str, *, dialect: str = "openai") -> dict[str, Any]:
    status = response.status_code
    if status in (401, 403) or (
        status == 400 and dialect == "google" and _google_key_invalid(response)
    ):
        raise LLMAuthError(f"{what}: the key was refused (HTTP {status})", status=status)
    if status >= 500:
        raise LLMUnavailableError(f"{what}: HTTP {status}", status=status)
    if status != 200:
        raise LLMRequestError(f"{what}: HTTP {status}", status=status)
    try:
        body = response.json()
    except ValueError:
        raise LLMRequestError(f"{what}: the answer is not JSON") from None
    if not isinstance(body, dict):
        raise LLMRequestError(f"{what}: unexpected answer shape")
    return body


def _google_key_invalid(response: httpx.Response) -> bool:
    """Gemini answers a bad key with 400 ``INVALID_ARGUMENT`` / ``API_KEY_INVALID``; other 400s
    (``FAILED_PRECONDITION``: billing off, region unsupported) are not about the key."""
    try:
        body = response.json()
    except ValueError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    details = error.get("details")
    if isinstance(details, list) and any(
        isinstance(d, dict) and d.get("reason") == "API_KEY_INVALID" for d in details
    ):
        return True
    message = error.get("message")
    return (
        error.get("status") == "INVALID_ARGUMENT"
        and isinstance(message, str)
        and "api key" in message.lower()
    )


async def _probe_models(client: httpx.AsyncClient, spec: ProviderSpec, api_key: str) -> KeyCheck:
    path = "/v1/models" if spec.dialect == "anthropic" else "/models"
    response = await _get(client, spec.base_url + path, _headers(spec, api_key))
    body = _check(response, spec.label, dialect=spec.dialect)
    if spec.dialect == "google":
        raw = body.get("models")
        names = [m.get("name") for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []
        ids = [n.removeprefix("models/") for n in names if isinstance(n, str)]
        return KeyCheck(ids, {})
    raw = body.get("data")
    items = [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []
    ids = [m["id"] for m in items if isinstance(m.get("id"), str)]
    aliases: dict[str, str] = {}
    for item in items:
        for alias in item.get("aliases") or []:
            if isinstance(alias, str) and isinstance(item.get("id"), str):
                aliases[alias] = item["id"]
    return KeyCheck(ids, aliases)


async def _probe_openrouter(
    client: httpx.AsyncClient, spec: ProviderSpec, api_key: str
) -> KeyCheck:
    headers = _headers(spec, api_key)
    key_body = _check(await _get(client, spec.base_url + "/key", headers), spec.label)
    data = key_body.get("data") if isinstance(key_body.get("data"), dict) else {}
    models = await _probe_models(client, spec, api_key)
    return KeyCheck(
        models.models,
        {},
        spend_usd=_float_or_none(data.get("usage")),
        limit_usd=_float_or_none(data.get("limit")),
    )


async def _probe_selfhosted(client: httpx.AsyncClient, base_url: str, token: str) -> KeyCheck:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = await _get(client, base_url + "/models", headers)
    if response.status_code == 404:
        return KeyCheck([], {})  # some proxies lack /models; the real test is the post
    body = _check(response, _host(base_url))
    raw = body.get("data")
    items = [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []
    return KeyCheck([m["id"] for m in items if isinstance(m.get("id"), str)], {})


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _host(url: str) -> str:
    return httpx.URL(url).host or url


# --- the connection test ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeFailure:
    """Why the connection test produced no line, specific enough to say what to fix.

    ``kind`` is one of ``auth`` (the key/token was refused), ``model`` (the model is not
    served there), ``timeout``, ``unreachable`` (no connection to ``base_url``),
    ``unavailable`` (429/5xx after retries), ``credit`` (out of credit or over a spend limit),
    ``cap`` (``llm.monthly_cap_usd`` is reached), ``request`` (another refusal; ``detail``
    has the server's words) and ``empty`` (the model answered with nothing usable).
    """

    kind: str
    provider: str = ""  # display name, or the host of a self-hosted server
    model: str = ""
    base_url: str = ""
    detail: str = ""  # the error as logged (never contains the key)
    timeout_seconds: float = 0.0
    cap_usd: float = 0.0


@dataclass(frozen=True)
class ProbeOutcome:
    line: str | None
    seconds: float
    failure: ProbeFailure | None = None


async def probe(
    llm: LLM, text: str, max_chars: int, *, timer: Callable[[], float] = time.monotonic
) -> ProbeOutcome:
    """Summarise one real post and time it; on failure, say why (``ProbeFailure``).

    A ``ChatLLM`` is asked through ``try_summarise`` so the backend's error reaches the owner;
    anything else (a disabled or fake LLM) only has ``summarise_line``, whose ``None`` is
    reported as an empty answer.
    """
    started = timer()
    try_summarise = getattr(llm, "try_summarise", None)
    if try_summarise is None:
        line = await llm.summarise_line(text, max_chars)
        seconds = timer() - started
        return ProbeOutcome(line, seconds, None if line else ProbeFailure("empty"))
    try:
        line = await try_summarise(text, max_chars)
    except LLMError as exc:
        return ProbeOutcome(None, timer() - started, describe_failure(exc, llm))
    return ProbeOutcome(line, timer() - started)


def describe_failure(exc: LLMError, llm: Any) -> ProbeFailure:
    """Map the error of a test request to the cause the owner can act on."""
    backend = getattr(llm, "backend", None)
    provider_key = str(getattr(backend, "provider", ""))
    base_url = str(getattr(backend, "_base_url", ""))
    selfhosted = bool(getattr(backend, "selfhosted", False)) or provider_key == "selfhosted"
    provider = _host(base_url) if selfhosted and base_url else provider_label(provider_key)
    budget = getattr(llm, "budget", None)
    context = {
        "provider": provider,
        "model": str(getattr(backend, "model", "")),
        "base_url": base_url,
        "detail": str(exc),
        "timeout_seconds": float(getattr(backend, "read_timeout", 0.0) or 0.0),
        "cap_usd": float(getattr(budget, "cap_usd", 0.0) or 0.0),
    }
    return ProbeFailure(_failure_kind(exc), **context)


def _failure_kind(exc: LLMError) -> str:
    if isinstance(exc, LLMBudgetError):
        return "credit" if exc.paused else "cap"
    if isinstance(exc, LLMEmptyAnswerError):
        return "empty"
    if isinstance(exc, LLMCreditError):
        return "credit"
    if isinstance(exc, LLMAuthError):
        return "auth"
    if isinstance(exc, LLMTimeoutError):
        return "timeout"
    if isinstance(exc, LLMUnavailableError):
        return "unreachable" if exc.status is None else "unavailable"
    if exc.status == 404 or (exc.status in (400, 422) and "model" in str(exc).lower()):
        return "model"
    return "request"


async def test_prompt(
    llm: LLM, text: str, max_chars: int, *, timer: Callable[[], float] = time.monotonic
) -> tuple[str | None, float]:
    """Summarise one real post and time it: ``(line, elapsed seconds)``; ``None`` on failure.
    ``probe`` is the same test with the reason for a failure (the wizard and the CLI use it)."""
    outcome = await probe(llm, text, max_chars, timer=timer)
    return outcome.line, outcome.seconds
