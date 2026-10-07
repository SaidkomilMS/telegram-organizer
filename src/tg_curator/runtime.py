"""The container handed to every service, and the in-process event bus (DESIGN §8).

``Runtime`` is a plain dataclass so that wiring is visible in one place (``service.py`` and
the CLI) and tests can hand a service exactly the fakes it needs. Services are attributes that
start as ``None`` and are set by the wiring: a service reads ``rt.sorter`` at call time, never
in its constructor, so construction order does not matter.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tg_curator import notify
from tg_curator.clock import Clock
from tg_curator.config import Settings, SettingsFile
from tg_curator.contracts import (
    LLM,
    AccountService,
    ActionExecutor,
    Backfill,
    DigestService,
    Discovery,
    Embedder,
    FolderManager,
    Intake,
    Learning,
    Notifier,
    PreviewService,
    Publisher,
    ReviewService,
    Sorter,
    StatsService,
    TopicClassifier,
    TopicsService,
)
from tg_curator.i18n import Translator
from tg_curator.telegram.gateway import Account, BotGateway, UserGateway

if TYPE_CHECKING:
    from tg_curator.db.store import Store

log = logging.getLogger(__name__)

# --- events ----------------------------------------------------------------------------------

EVENT_SETTINGS_CHANGED = "settings_changed"
"""The settings file was written (bot edit) or re-read (``/reload``); no payload."""
EVENT_TOPICS_CHANGED = "topics_changed"
"""A topic was created, merged, removed or deactivated; no payload."""
EVENT_ACCOUNT_BOUND = "account_bound"
"""The account finished signing in; the service starts intake without a restart."""
EVENT_SESSION_LOST = "session_lost"
"""The account session is gone; payload ``reason: str`` (a ``SessionLostReason``)."""
EVENT_EXAMPLES_CHANGED = "examples_changed"
"""Training data changed; payload ``reason: ExamplesChangedReason`` (always present)."""

EVENTS = frozenset(
    {
        EVENT_SETTINGS_CHANGED,
        EVENT_TOPICS_CHANGED,
        EVENT_ACCOUNT_BOUND,
        EVENT_SESSION_LOST,
        EVENT_EXAMPLES_CHANGED,
    }
)

ExamplesChangedReason = Literal["correction", "examples", "topics"]
"""Why ``examples_changed`` fired: ``"correction"`` from ``Learning.correct`` (§9.8),
``"examples"`` from ``TopicsService.add_examples``, ``add_example_channel`` and
``update(description=...)``, ``"topics"`` from ``TopicsService.create``, ``merge``, ``remove``
and ``Discovery.accept`` (which already ran ``classifier.reload`` and
``sorter.resort_unsorted`` themselves). The retrain loop re-sorts only for ``"examples"``."""
EXAMPLES_CHANGED_REASONS = frozenset({"correction", "examples", "topics"})

EventHandler = Callable[..., Awaitable[None]]
"""``async def handler(**payload)``; it receives exactly the keyword payload of ``emit``."""


class EventBus:
    """A tiny in-process bus: ``on(name, handler)``, ``await emit(name, **payload)``.

    Handlers run sequentially in registration order so an emitter can rely on "after emit
    returns, every handler ran". A failing handler is logged and skipped — an event is a
    notification, and the emitter (a bot command, the sorter) must not die because a
    subscriber did.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, name: str, handler: EventHandler) -> None:
        if name not in EVENTS:
            log.warning("events: subscribing to unknown event %r", name)
        self._handlers[name].append(handler)

    def handlers(self, name: str) -> list[EventHandler]:
        """The registered handlers for ``name`` (a copy; for tests and ``/status``)."""
        return list(self._handlers.get(name, ()))

    async def emit(self, name: str, **payload: Any) -> None:
        if name not in EVENTS:
            log.warning("events: emitting unknown event %r", name)
        if name == EVENT_EXAMPLES_CHANGED and payload.get("reason") not in EXAMPLES_CHANGED_REASONS:
            log.warning(
                "events: examples_changed without a valid reason: %r", payload.get("reason")
            )
        for handler in self.handlers(name):
            try:
                await handler(**payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("events: handler %s for %s failed", _handler_name(handler), name)


def _handler_name(handler: EventHandler) -> str:
    return getattr(handler, "__qualname__", None) or repr(handler)


# --- the runtime -----------------------------------------------------------------------------


@dataclass
class LoopState:
    """What the supervisor in ``service.py`` knows about one loop, for ``/status`` (RD-5).

    ``health`` alone records successes, so a loop that failed on every tick since start-up
    left no trace there and a loop held back by setup mode looked the same as a broken one;
    this record keeps both apart. ``interval`` is ``None`` for a body that serves until it is
    cancelled (the control socket): it has no ticks, so it cannot fall behind.
    ``failing_since`` is the first failure of the current run of failures (``None`` once a
    tick succeeds); ``last_error_at``/``last_error`` keep the latest one, as a short text.
    ``paused`` means stopped on purpose until the account is bound (§11.3).
    """

    interval: float | None
    started_at: datetime | None = None
    first_delay: float = 0.0
    last_ok_at: datetime | None = None
    failing_since: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None
    running: bool = False
    paused: bool = False


@dataclass
class Runtime:
    """Everything a service needs, in one object; see the module docstring.

    ``settings`` is read at use time (``rt.settings.sorting.confidence``) so bot edits apply
    live. ``notifier`` is built here because it needs the runtime it is part of (the bot, the
    owner id and the catalogue) — pass nothing for it. ``health`` holds the last successful
    tick per loop or on-demand job name; ``loops`` holds the supervisor's full view of each
    supervised loop (``LoopState``). Both are written by ``service.py`` and read by ``/status``.
    """

    home: Path
    settings_file: SettingsFile
    store: Store
    clock: Clock
    events: EventBus
    t: Translator
    embedder: Embedder
    classifier: TopicClassifier
    llm: LLM
    user: UserGateway | None = None
    bot: BotGateway | None = None
    bot_account: Account | None = None
    health: dict[str, datetime] = field(default_factory=dict)
    loops: dict[str, LoopState] = field(default_factory=dict)
    notifier: Notifier = field(init=False)

    # services, set by the wiring in service.py / cli.py
    intake: Intake | None = None
    backfill: Backfill | None = None
    sorter: Sorter | None = None
    preview: PreviewService | None = None
    publisher: Publisher | None = None
    digest: DigestService | None = None
    topics: TopicsService | None = None
    learning: Learning | None = None
    stats: StatsService | None = None
    review: ReviewService | None = None
    actions: ActionExecutor | None = None
    folders: FolderManager | None = None
    discovery: Discovery | None = None
    account: AccountService | None = None

    def __post_init__(self) -> None:
        self.notifier = notify.Notifier(self)
        # §4: after any settings write the bus hears settings_changed; the file cannot import
        # the runtime, so the hook is connected here (an explicit hook set earlier is kept).
        if self.settings_file.on_change is None:
            self.settings_file.on_change = self._settings_written

    @property
    def settings(self) -> Settings:
        return self.settings_file.current

    async def _settings_written(self, settings: Settings) -> None:
        await self.events.emit(EVENT_SETTINGS_CHANGED)
