"""The ``LLM`` implementation and the HTTP transport every dialect shares (DESIGN §8, §15).

``ChatLLM`` is what the rest of the curator calls: it builds the prompt, reserves budget,
runs one short request through a ``Backend`` and settles. Every failure becomes ``None`` —
the digest always has the first line to fall back on, so nothing here may raise into a loop.
The one exception is ``try_summarise``, the connection test's path, which raises the reason.
``HttpBackend`` holds the retry policy of §15 (408/409/429/5xx/529 and connection errors,
three attempts, 1-2-4 s with jitter, ``Retry-After`` honoured; never 400/401/402/403/404/413/
422) and the out-of-credit detection that pauses a provider for the month; the wire dialects
themselves live in ``providers.py``. ``DisabledLLM`` is ``llm.mode = "none"``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from tg_curator.domain import Topic, TopicName
from tg_curator.errors import CuratorError
from tg_curator.llm import prompts
from tg_curator.llm.budget import Budget, Reservation, estimate_input_tokens

log = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
NEVER_RETRY_STATUSES = frozenset({400, 401, 402, 403, 404, 413, 422})
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
RETRY_AFTER_MAX = 60.0
CONNECT_TIMEOUT = 10.0
PROVIDER_READ_TIMEOUT = 60.0
SELFHOSTED_READ_TIMEOUT = 180.0
PROVIDER_CONCURRENCY = 4
SELFHOSTED_CONCURRENCY = 1

OPENAI_CREDIT_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "insufficient_quota",
    }
)
ANTHROPIC_SPEND_LIMIT_CODE = "enforced_spend_limit_reached"
ANTHROPIC_SPEND_LIMIT_PREFIX = "You have reached your specified API usage limits"


# --- errors ----------------------------------------------------------------------------------


class LLMError(CuratorError):
    """Base of every language-model failure; the message never contains the key."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMAuthError(LLMError):
    """401/403: the key or token is wrong, or the model is not allowed for it."""


class LLMCreditError(LLMError):
    """The provider refused for lack of credit; retrying will not help this month."""


class LLMRequestError(LLMError):
    """A non-retryable request error (400/404/413/422, bad JSON, unknown shape)."""


class LLMUnavailableError(LLMError):
    """Retries exhausted on 429/5xx or connection errors."""


class LLMTimeoutError(LLMUnavailableError):
    """Retries exhausted and the last failure was a timeout (the provider may have billed)."""


class LLMBudgetError(LLMError):
    """No request was sent: the monthly cap is reached (or a request would cross it), or —
    ``paused`` — the provider said earlier this month that the account is out of credit."""

    def __init__(self, message: str, *, paused: bool = False) -> None:
        super().__init__(message)
        self.paused = paused


class LLMEmptyAnswerError(LLMError):
    """The model answered, but with nothing usable (a refusal, a safety block, only noise)."""


# --- what a backend returns ------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Token accounting as reported; ``None`` tokens mean the server sent no usage."""

    input_tokens: int | None
    output_tokens: int | None  # INCLUDING reasoning/thinking tokens
    reasoning_tokens: int = 0
    reported_cost_usd: float | None = None  # OpenRouter ``usage.cost``
    model: str | None = None  # the model id the server says it used

    @property
    def estimated(self) -> bool:
        return self.input_tokens is None or self.output_tokens is None


class Backend(Protocol):
    """One wire dialect bound to one model; ``complete`` is a single system+user turn."""

    provider: str
    model: str
    selfhosted: bool

    async def complete(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, Usage]:
        """The cleaned answer text (``""`` on refusal/empty/safety block) and its usage."""
        ...

    def output_budget(self, max_tokens: int) -> int:
        """Tokens to reserve for the answer (Gemini needs headroom for thinking)."""
        ...

    async def aclose(self) -> None: ...


Sleep = Callable[[float], Awaitable[None]]


class HttpBackend:
    """The shared transport: one ``httpx.AsyncClient``, the retry policy, error classes.

    Subclasses implement ``_build`` (url, headers, body) and ``_parse`` (text, usage) for
    their dialect; everything about failure is decided here once.
    """

    provider: str
    model: str
    selfhosted: bool = False

    def __init__(
        self,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
        read_timeout: float = PROVIDER_READ_TIMEOUT,
        sleep: Sleep | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._own_client = client is None
        self.read_timeout = read_timeout
        self._timeout = httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT)
        self._sleep = sleep or asyncio.sleep

    # --- the dialect hooks ---

    def _build(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _parse(self, body: dict[str, Any]) -> tuple[str, Usage]:
        raise NotImplementedError

    def output_budget(self, max_tokens: int) -> int:
        return max_tokens

    # --- the public call ---

    async def complete(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool
    ) -> tuple[str, Usage]:
        url, headers, body = self._build(system, user, max_tokens=max_tokens, json_mode=json_mode)
        data = await self.post_json(url, headers, body)
        return self._parse(data)

    async def aclose(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    # --- transport ---

    async def post_json(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST with the §15 retry policy; returns the decoded JSON body."""
        last: LLMError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self.client.post(
                    url, json=body, headers=headers, timeout=self._timeout
                )
            except httpx.TimeoutException as exc:
                last = LLMTimeoutError(f"{self.provider}: timed out ({exc.__class__.__name__})")
                delay = self._backoff(attempt)
            except httpx.HTTPError as exc:
                last = LLMUnavailableError(f"{self.provider}: {exc.__class__.__name__}")
                delay = self._backoff(attempt)
            else:
                outcome = self._classify(response)
                if isinstance(outcome, dict):
                    return outcome
                last = outcome
                if not isinstance(outcome, LLMUnavailableError):
                    raise outcome
                delay = self._retry_after(response) or self._backoff(attempt)
            if attempt + 1 < MAX_ATTEMPTS:
                log.debug("llm: %s; retrying in %.1f s", last, delay)
                await self._sleep(delay)
        assert last is not None
        raise last

    def _classify(self, response: httpx.Response) -> dict[str, Any] | LLMError:
        """A decoded body, or the error to raise (an ``LLMUnavailableError`` is retried)."""
        status = response.status_code
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        error = body.get("error")
        if status == 200 and error is not None and not self._has_answer(body):
            # OpenRouter (and some proxies) answer 200 with an error body when the upstream
            # failed after the headers went out; treat it by the status the body carries.
            status = error.get("code") if isinstance(error, dict) else None
            status = status if isinstance(status, int) and status >= 400 else 502
        if status == 200:
            return body
        message = _error_message(body) or response.reason_phrase or f"HTTP {status}"
        text = f"{self.provider}: HTTP {status}: {message[:200]}"
        if status == 402 or out_of_credit(status, body):
            return LLMCreditError(text, status=status)
        if status in (401, 403):
            return LLMAuthError(text, status=status)
        if status in RETRYABLE_STATUSES or status >= 500:
            return LLMUnavailableError(text, status=status)
        return LLMRequestError(text, status=status)

    @staticmethod
    def _has_answer(body: dict[str, Any]) -> bool:
        return any(k in body for k in ("choices", "content", "candidates"))

    def _backoff(self, attempt: int) -> float:
        base = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
        return base * random.uniform(0.75, 1.25)  # ±25 % jitter

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            return None
        return seconds if 0 <= seconds <= RETRY_AFTER_MAX else None


def out_of_credit(status: int, body: dict[str, Any]) -> bool:
    """The non-retryable billing shapes of §15 (OpenAI codes, Anthropic's two forms)."""
    error = body.get("error")
    if not isinstance(error, dict):
        return False
    code = error.get("code")
    if isinstance(code, str) and code in OPENAI_CREDIT_CODES:
        return True
    details = error.get("details")
    if (
        status == 429
        and isinstance(details, dict)
        and details.get("error_code") == ANTHROPIC_SPEND_LIMIT_CODE
    ):
        return True
    message = error.get("message")
    return (
        status == 400
        and isinstance(message, str)
        and message.startswith(ANTHROPIC_SPEND_LIMIT_PREFIX)
    )


def _error_message(body: dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    message = body.get("message")  # Mistral puts it at the top level
    return message if isinstance(message, str) else ""


# --- the LLM ---------------------------------------------------------------------------------


class ChatLLM:
    """The connected model (``llm.mode`` selfhosted or provider), see the module docstring."""

    enabled = True

    def __init__(self, backend: Backend, budget: Budget, *, concurrency: int) -> None:
        self.backend = backend
        self.budget = budget
        self._semaphore = asyncio.Semaphore(concurrency)

    async def summarise_line(
        self, text: str, max_chars: int, *, source: str | None = None
    ) -> str | None:
        try:
            return await self.try_summarise(text, max_chars, source=source)
        except LLMError:
            return None  # already logged by ``_request``; the digest uses the first line

    async def try_summarise(self, text: str, max_chars: int, *, source: str | None = None) -> str:
        """``summarise_line`` that says why it failed instead of answering ``None``: the
        connection test (``registry.probe``) needs the reason to tell the owner what to fix.

        Raises the backend's ``LLMError`` subclass, ``LLMBudgetError`` when no request may be
        sent this month, or ``LLMEmptyAnswerError`` when the answer holds no line.
        """
        system, user = prompts.summary_prompt(text, max_chars, source=source)
        answer = await self._request(system, user, max_tokens=prompts.SUMMARY_MAX_TOKENS)
        line = prompts.clean_line(answer, max_chars) if answer else ""
        if not line:
            raise LLMEmptyAnswerError(
                f"{provider_label(self.backend.provider)}: {self.backend.model} returned an "
                "empty answer"
            )
        return line

    async def name_topic(
        self,
        examples: Sequence[str],
        *,
        existing: Sequence[str] = (),
        category_label: str | None = None,
    ) -> TopicName | None:
        system, user = prompts.topic_prompt(
            examples, existing=existing, category_label=category_label
        )
        answer = await self._ask(system, user, max_tokens=prompts.TOPIC_MAX_TOKENS, json_mode=True)
        if answer is None:
            return None
        name = prompts.parse_topic_name(answer, existing=existing)
        if name is not None:
            return name
        # One retry with the model's own answer quoted back, then the caller falls back.
        retry_user = f"{user}\n{answer}\n{prompts.TOPIC_RETRY_SUFFIX}"
        answer = await self._ask(
            system, retry_user, max_tokens=prompts.TOPIC_MAX_TOKENS, json_mode=True
        )
        return None if answer is None else prompts.parse_topic_name(answer, existing=existing)

    async def second_opinion(
        self, text: str, topic: Topic, *, competing: Sequence[Topic] = ()
    ) -> bool | None:
        system, user = prompts.opinion_prompt(text, topic, competing=competing)
        answer = await self._ask(system, user, max_tokens=prompts.OPINION_MAX_TOKENS)
        return None if answer is None else prompts.parse_yes_no(answer)

    async def aclose(self) -> None:
        await self.backend.aclose()

    # --- one request, budgeted ---

    async def _ask(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool = False
    ) -> str | None:
        try:
            text = await self._request(system, user, max_tokens=max_tokens, json_mode=json_mode)
        except LLMError:
            return None  # logged by ``_request``; every caller has a fallback
        return text or None

    async def _request(
        self, system: str, user: str, *, max_tokens: int, json_mode: bool = False
    ) -> str:
        """One budgeted request; the answer text (``""`` when empty) or the ``LLMError`` that
        stopped it. The budget is released or kept exactly as §15 says for each failure."""
        if await self.budget.exhausted():
            raise self._budget_error()
        reservation = await self.budget.reserve(
            estimate_input_tokens(system, user), self.backend.output_budget(max_tokens)
        )
        if reservation is None:
            raise self._budget_error()
        async with self._semaphore:
            try:
                text, usage = await self.backend.complete(
                    system, user, max_tokens=max_tokens, json_mode=json_mode
                )
            except LLMCreditError as exc:
                log.warning("llm: %s", exc)
                await self.budget.release(reservation)
                await self.budget.out_of_credit(
                    provider_label(self.backend.provider), reservation.month
                )
                raise
            except LLMTimeoutError as exc:
                log.warning("llm: %s", exc)
                await self.budget.release(reservation, keep_estimate=True)
                raise
            except LLMError as exc:
                log.warning("llm: %s", exc)
                await self.budget.release(reservation)
                raise
        await self._settle(reservation, usage)
        return text

    def _budget_error(self) -> LLMBudgetError:
        paused = self.budget.paused_month == self.budget.month()
        what = "out of credit" if paused else "the monthly cap is reached"
        return LLMBudgetError(
            f"{provider_label(self.backend.provider)}: {what}; no request sent", paused=paused
        )

    async def _settle(self, reservation: Reservation, usage: Usage) -> None:
        await self.budget.settle(
            reservation,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=usage.model,
            reported_cost_usd=usage.reported_cost_usd,
        )


def provider_label(provider: str) -> str:
    """The name the owner knows a provider by ("OpenAI" for ``openai``); a self-hosted server
    or an unknown key is shown as configured."""
    # Imported here: providers.py builds on this module, so a module-level import would cycle.
    from tg_curator.llm.providers import PROVIDERS

    spec = PROVIDERS.get(provider)
    return spec.label if spec is not None else provider


class DisabledLLM:
    """``llm.mode = "none"``: nothing leaves the server and every call answers ``None``."""

    enabled = False

    async def summarise_line(self, text: str, max_chars: int) -> str | None:
        return None

    async def name_topic(
        self,
        examples: Sequence[str],
        *,
        existing: Sequence[str] = (),
        category_label: str | None = None,
    ) -> TopicName | None:
        return None

    async def second_opinion(
        self, text: str, topic: Topic, *, competing: Sequence[Topic] = ()
    ) -> bool | None:
        return None

    async def aclose(self) -> None:
        return None
