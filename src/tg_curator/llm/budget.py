"""The monthly spending ledger (DESIGN §8 LLM, §15 "Budget ledger", research §7.2).

Reserve before the call, settle after it: the reservation is a conservative estimate
(``chars / 2.5`` input tokens plus the whole output budget), so the month can never be
overspent by a burst of parallel requests, and the settlement replaces it with what the
provider reported (OpenRouter: its ``usage.cost``). Self-hosted servers cost nothing, so their
requests are counted but never capped. When the cap is crossed — or the provider itself says
the account is out of credit — the owner hears about it once per month (one of the two lines,
whichever comes first: both share ``llm_usage.cap_notified``), and the curator falls back to
first-line digests until the month ends.

"Until the month ends" is kept literally: once a request is refused at the cap (or settling one
reaches it) the month is *closed* — later, smaller requests that would still fit under the cap
are refused too, so model lines do not reappear after the owner was told they stopped. The
closure is stored in ``kv llm.cap_closed`` (``{"month", "cap_usd"}``) so it survives a restart,
and it holds only while the cap is not raised above the one it was closed at: raising the cap
through ``/llm`` (a new ``Budget``) opens the month again.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from tg_curator.domain import KV

if TYPE_CHECKING:
    from tg_curator.clock import Clock
    from tg_curator.contracts import Notifier
    from tg_curator.db.store import Store

log = logging.getLogger(__name__)

CAP_CLOSED_KEY = KV.LLM_CAP_CLOSED
"""``kv`` key: ``{"month": "YYYY-MM", "cap_usd": float}`` of the month closed at the cap."""

CHARS_PER_TOKEN = 2.5
"""Conservative chars-per-token for the reservation (Uzbek Latin measured at 3.1)."""
ESTIMATE_MARGIN = 1.5
"""Settlements without reported usage (self-hosted mostly) are scaled up by this factor."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens; the output price also applies to reasoning/thinking tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    source: str = ""
    effective_from: date | None = None
    note: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        ) / 1e6


ZERO_PRICE = Price(0.0, 0.0, note="self-hosted")

PriceLookup = Callable[[str | None], Awaitable[Price]]
"""``await lookup(model)``: the price to use; ``None`` = the configured model."""


@dataclass(frozen=True)
class Reservation:
    month: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Settled:
    """What a request finally cost; ``estimated`` when the server reported no usage."""

    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool


def estimate_input_tokens(*texts: str) -> int:
    return math.ceil(sum(len(t) for t in texts) / CHARS_PER_TOKEN)


class Budget:
    """The ledger over ``llm_usage`` (§6) for one configured model."""

    def __init__(
        self,
        store: Store,
        clock: Clock,
        *,
        cap_usd: float,
        timezone: str,
        price_lookup: PriceLookup,
        notifier: Notifier | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self.cap_usd = cap_usd
        self._tz = ZoneInfo(timezone)
        self._price_lookup = price_lookup
        self.notifier = notifier
        self._paused_month: str | None = None
        self._closed_month: str | None = None
        # The cap check and the booking must be one step, or parallel reservations all read
        # the same total and all pass. One Budget per process, so a process lock suffices.
        self._reserve_lock = asyncio.Lock()

    # --- months ---

    def month(self) -> str:
        """``YYYY-MM`` of now in the user's timezone: people think in local months."""
        return self._clock.now().astimezone(self._tz).strftime("%Y-%m")

    @property
    def capped(self) -> bool:
        return self.cap_usd > 0

    # --- the state callers ask about ---

    async def exhausted(self, month: str | None = None) -> bool:
        """True when nothing more may be spent this month: the cap is reached, or the provider
        reported it is out of credit."""
        month = month or self.month()
        if self._paused_month == month:
            return True
        if not self.capped:
            return False
        if await self._closed(month):
            return True
        usage = await self._store.get_usage(month)
        return usage.cost_usd >= self.cap_usd

    # --- reserve / settle / release ---

    async def reserve(self, input_tokens: int, output_tokens: int) -> Reservation | None:
        """Book a conservative estimate; ``None`` (and the once-a-month notice) when it would
        cross the cap or the provider is paused."""
        month = self.month()
        if self._paused_month == month or await self._closed(month):
            return None
        price = await self._price_lookup(None)
        cost = price.cost(input_tokens, output_tokens)
        async with self._reserve_lock:
            over = False
            if self.capped:
                usage = await self._store.get_usage(month)
                over = usage.cost_usd + cost > self.cap_usd
            if not over:
                await self._store.add_usage(month, 1, input_tokens, output_tokens, cost)
        if over:
            # The cap is reached for this month: smaller requests that would still fit must
            # not bring model lines back after the owner is told digests show first lines.
            await self._close(month)
            await self._notify_cap(month)
            return None
        return Reservation(month, input_tokens, output_tokens, cost)

    async def settle(
        self,
        reservation: Reservation,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        model: str | None = None,
        reported_cost_usd: float | None = None,
    ) -> Settled:
        """Replace the reservation with what was actually used.

        Missing usage (``None`` tokens) keeps the estimate scaled by ``ESTIMATE_MARGIN``; a
        reported cost (OpenRouter) wins over the price table; the model actually used (a
        Mistral ``-latest`` alias resolves to a dated id) prices the settlement.
        """
        estimated = input_tokens is None or output_tokens is None
        in_tokens = reservation.input_tokens if input_tokens is None else input_tokens
        out_tokens = reservation.output_tokens if output_tokens is None else output_tokens
        if reported_cost_usd is not None:
            cost = reported_cost_usd
        else:
            price = await self._price_lookup(model)
            cost = price.cost(in_tokens, out_tokens)
            if estimated:
                cost *= ESTIMATE_MARGIN
        await self._store.add_usage(
            reservation.month,
            0,
            in_tokens - reservation.input_tokens,
            out_tokens - reservation.output_tokens,
            cost - reservation.cost_usd,
        )
        if self.capped and await self.exhausted(reservation.month):
            await self._close(reservation.month)
            await self._notify_cap(reservation.month)
        return Settled(in_tokens, out_tokens, cost, estimated)

    async def release(self, reservation: Reservation, *, keep_estimate: bool = False) -> None:
        """The request failed: nothing was billed (unless ``keep_estimate`` — a timeout, where
        the provider may have charged anyway); the request itself stays counted."""
        if keep_estimate:
            return
        await self._store.add_usage(
            reservation.month,
            0,
            -reservation.input_tokens,
            -reservation.output_tokens,
            -reservation.cost_usd,
        )

    async def out_of_credit(self, provider: str, month: str | None = None) -> None:
        """``provider`` (its display name) refused for lack of credit: pause it for the rest of
        the month and tell the owner once, in the provider's own words rather than as a cap
        the owner never set (§17.5)."""
        month = month or self.month()
        self._paused_month = month
        log.warning("llm: %s is out of credit; paused until the end of %s", provider, month)
        if await self._first_notice(month) and self.notifier is not None:
            await self.notifier.llm_out_of_credit(provider)

    @property
    def paused_month(self) -> str | None:
        return self._paused_month

    # --- internals ---

    async def _closed(self, month: str) -> bool:
        """The month was closed at the cap (by this process or, per ``kv``, an earlier one)
        and the cap has not been raised since."""
        if not self.capped:
            return False
        if self._closed_month == month:
            return True
        raw = await self._store.kv_get(CAP_CLOSED_KEY)
        if not isinstance(raw, dict) or raw.get("month") != month:
            return False
        closed_at = raw.get("cap_usd")
        if not isinstance(closed_at, int | float) or self.cap_usd > closed_at:
            return False  # the owner raised the cap since: the month is open again
        self._closed_month = month
        return True

    async def _close(self, month: str) -> None:
        if self._closed_month == month:
            return
        self._closed_month = month
        await self._store.kv_set(CAP_CLOSED_KEY, {"month": month, "cap_usd": self.cap_usd})

    async def _notify_cap(self, month: str) -> None:
        if not await self._first_notice(month):
            return
        log.warning("llm: monthly budget exhausted for %s; digests fall back to first lines", month)
        if self.notifier is not None:
            await self.notifier.llm_cap_reached(month, self.cap_usd)

    async def _first_notice(self, month: str) -> bool:
        """Claim the month's one notice: ``True`` the first time it is asked for a month."""
        usage = await self._store.get_usage(month)
        if usage.cap_notified:
            return False
        await self._store.set_cap_notified(month, True)
        return True
