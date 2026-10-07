"""``make_llm``: the one place that turns ``[llm]`` settings into an ``LLM`` (DESIGN §8).

The runtime holds ``rt.llm`` before the notifier exists (the notifier is built from the
runtime), so the once-a-month cap notice cannot be wired at construction: pass ``notifier``
when you have one, or call ``attach_notifier`` afterwards — ``service.py`` and the CLI do the
latter right after building the ``Runtime``. Without a notifier the budget still stops at the
cap; the owner just does not hear about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from tg_curator.clock import Clock, SystemClock
from tg_curator.contracts import LLM, Notifier
from tg_curator.llm.base import (
    PROVIDER_CONCURRENCY,
    SELFHOSTED_CONCURRENCY,
    ChatLLM,
    DisabledLLM,
)
from tg_curator.llm.budget import Budget, Price
from tg_curator.llm.providers import build_backend
from tg_curator.llm.registry import OpenRouterCatalogue, settlement_price

if TYPE_CHECKING:
    from tg_curator.config import Settings
    from tg_curator.db.store import Store


def make_llm(
    settings: Settings,
    store: Store,
    *,
    notifier: Notifier | None = None,
    clock: Clock | None = None,
    client: httpx.AsyncClient | None = None,
) -> LLM:
    """A ``DisabledLLM`` for ``mode = "none"``, else a budgeted ``ChatLLM`` over the backend.

    ``notifier`` is optional because of the construction order described in the module
    docstring; ``clock`` defaults to the wall clock and ``client`` to a client the LLM owns.
    """
    cfg = settings.llm
    if cfg.mode == "none":
        return DisabledLLM()
    clock = clock or SystemClock()
    selfhosted = cfg.mode == "selfhosted"
    provider = "selfhosted" if selfhosted else cfg.provider
    catalogue = OpenRouterCatalogue(clock) if provider == "openrouter" else None
    backend = build_backend(
        cfg, client=client, effort_lookup=catalogue.effort if catalogue else None
    )

    async def lookup(model: str | None) -> Price:
        # ``model`` is the id the answer reported (``None`` for a reservation); a dated
        # snapshot of the configured model is priced as that model, never as an unknown one.
        if catalogue is not None:
            await catalogue.ensure_fresh(backend.client)
        return settlement_price(
            provider, model, cfg.model, today=clock.now().date(), catalogue=catalogue
        )

    budget = Budget(
        store,
        clock,
        cap_usd=0.0 if selfhosted else cfg.monthly_cap_usd,
        timezone=settings.general.timezone,
        price_lookup=lookup,
        notifier=notifier,
    )
    concurrency = SELFHOSTED_CONCURRENCY if selfhosted else PROVIDER_CONCURRENCY
    return ChatLLM(backend, budget, concurrency=concurrency)


def attach_notifier(llm: LLM, notifier: Notifier) -> None:
    """Give an already built ``ChatLLM`` its notifier (a no-op for ``DisabledLLM``)."""
    if isinstance(llm, ChatLLM):
        llm.budget.notifier = notifier
