"""``curator run``: the wiring of every component and the supervised background loops (§13).

One asyncio process holds the account session, the bot and every loop. ``build_runtime``
constructs the real components (or the test doubles handed in) and wires every service and
bot module; ``Service`` runs the start sequence of §13 and keeps the loops alive; ``run`` adds
the parts that belong to a process — the service lock, the settings checks of a first start,
the model download and clean shutdown on SIGTERM/SIGINT.

Setup mode (§11.3): while the account is not bound — on a first start, or after the session
was revoked — only the bot and the account-free loops run; ``account_bound`` brings the rest
up without a restart, the same way as at start.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import tomllib
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from tg_curator import control
from tg_curator.account import WATCHDOG_INTERVAL, AccountService
from tg_curator.bot.bind import logged_in_line
from tg_curator.bot.control import day_word, has_postable_topic
from tg_curator.bot.core import BotApp
from tg_curator.clock import Clock, SystemClock
from tg_curator.config import SETTINGS_FILENAME, Settings, SettingsFile, ensure_home, write_template
from tg_curator.db import schema
from tg_curator.db.store import Store, sqlite_url
from tg_curator.domain import KV
from tg_curator.errors import (
    ConfigError,
    CuratorError,
    FloodWait,
    SessionLost,
    TelegramUnavailable,
)
from tg_curator.i18n import Translator
from tg_curator.llm.factory import attach_notifier, make_llm
from tg_curator.ml import models
from tg_curator.pipeline.backfill import BackfillService
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.engine import embedding_to_bytes
from tg_curator.pipeline.intake import GROUP_RETENTION, IntakeService
from tg_curator.pipeline.preview import PreviewService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import (
    EVENT_ACCOUNT_BOUND,
    EVENT_EXAMPLES_CHANGED,
    EVENT_SESSION_LOST,
    EVENT_SETTINGS_CHANGED,
    EventBus,
    LoopState,
    Runtime,
)
from tg_curator.subscriptions.actions import ActionExecutor
from tg_curator.subscriptions.discovery import Discovery
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.topics.learning import Learning
from tg_curator.topics.service import TopicsService

if TYPE_CHECKING:
    import numpy as np

    from tg_curator.contracts import LLM, Embedder, TopicClassifier
    from tg_curator.telegram.gateway import BotGateway, IncomingMessage, UserGateway

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Progress = Callable[[str], None]

BOT_MODULES = (
    "setup",
    "bind",
    "control",
    "topics",
    "corrections",
    "preview",
    "llm",
    "settings",
    "reports",
    "review",
)
"""The feature modules under ``tg_curator.bot``; each exposes ``register(app)`` (§11.2)."""

WAIT_ENV = "TG_CURATOR_WAIT_FOR_SETTINGS"
SETTINGS_POLL_SECONDS = 5.0
RESTART_DELAY = 5.0
ALBUM_FLUSH_TIMEOUT = 15.0
"""How long a stop waits for the albums still settling to be submitted (RD-6)."""
BRING_UP_MAX_DELAY = 300.0
RETRAIN_DEBOUNCE = 5.0
SETTINGS_DEBOUNCE = 2.0
EMBED_BATCH = 64
LLM_USAGE_KEEP = timedelta(days=366)

# Loop name -> interval in seconds (§13). The account loops stop in setup mode (§11.3).
CORE_LOOPS = {"sorter": 30.0, "publisher": 2.0, "digest": 30.0, "review": 60.0, "link": 60.0}
ACCOUNT_LOOPS = {
    "intake": 5.0,
    "actions": 20.0,
    "chats": 1800.0,
    "folders": 1800.0,
    "watchdog": WATCHDOG_INTERVAL.total_seconds(),
    "topic_channels": 60.0,
}
HOUSEKEEPING_SECONDS = 86400.0
ERROR_TEXT_LIMIT = 120
"""Characters of a loop's last error kept for ``/status``; the full traceback is in the log."""

SETUP_STEPS = """\
tg-curator: created {path} from the template.
Next steps:
  1. Create a bot with @BotFather in Telegram and put its token in [telegram] bot_token.
  2. Get api_id and api_hash for your account at https://my.telegram.org (API development
     tools) and put them in [telegram]. With Docker, the three values can go into .env
     instead (TG_CURATOR_API_ID, TG_CURATOR_API_HASH, TG_CURATOR_BOT_TOKEN).
  3. Start the service again (`curator run`, `systemctl start tg-curator`, or with Docker
     `docker compose up -d`). It prints a one-time claim code.
  4. Open your bot in Telegram, send /start with the claim code, then /setup."""


# --- wiring ----------------------------------------------------------------------------------


@dataclass
class Doubles:
    """Replacements for the real components (tests); every field left ``None`` is real."""

    user: UserGateway | None = None
    bot: BotGateway | None = None
    clock: Clock | None = None
    embedder: Embedder | None = None
    classifier: TopicClassifier | None = None
    llm: LLM | None = None


doubles_factory: Callable[[], Doubles] | None = None
"""The test hook: when set, ``build_runtime`` without explicit ``fakes`` uses what it returns,
so ``run()`` and every CLI command can be driven by the fakes without touching Telegram."""


class _ModelsNotLoaded:
    """Stands in for the embedder and the classifier in commands that never sort (login,
    chats, stats, digest, llm): loading a 700 MB model for them would be pure waste. Any use is
    a wiring bug and says so."""

    id = "not-loaded"
    dim = 0

    def __getattr__(self, name: str) -> Any:
        raise CuratorError(f"the local models are not loaded in this command (wanted {name})")


def build_runtime(
    home: Path,
    *,
    settings_file: SettingsFile,
    store: Store | None = None,
    fakes: Doubles | None = None,
    models_loaded: bool = True,
) -> tuple[Runtime, BotApp]:
    """The Runtime with every service and the BotApp with every bot module registered.

    ``settings_file`` must be loaded. Telethon is imported only when a real gateway is needed.
    With ``models_loaded`` (the default) the model files must already be on disk
    (``load_models``); ``False`` wires a placeholder for commands that never embed.
    """
    if fakes is None and doubles_factory is not None:
        fakes = doubles_factory()
    fakes = fakes or Doubles()
    settings = settings_file.current
    clock = fakes.clock or SystemClock()
    store = store or Store(
        settings.storage.database_url or sqlite_url(home / "curator.db"), clock=clock
    )
    embedder, classifier = _models(home, settings, fakes, models_loaded)
    rt = Runtime(
        home=home,
        settings_file=settings_file,
        store=store,
        clock=clock,
        events=EventBus(),
        t=Translator(settings.general.language, home=home),
        embedder=embedder,
        classifier=classifier,
        llm=fakes.llm or make_llm(settings, store, clock=clock),
        user=fakes.user or _user_gateway(home, settings, clock),
        bot=fakes.bot or _bot_gateway(home, settings_file),
    )
    attach_notifier(rt.llm, rt.notifier)
    rt.account = AccountService(rt)
    rt.intake = IntakeService(rt)
    rt.backfill = BackfillService(rt)
    rt.sorter = Sorter(rt)
    rt.preview = PreviewService(rt)
    rt.publisher = Publisher(rt)
    rt.digest = DigestService(rt)
    rt.topics = TopicsService(rt)
    rt.learning = Learning(rt)
    rt.stats = StatsService(rt)
    rt.review = ReviewService(rt)
    rt.actions = ActionExecutor(rt)
    rt.folders = FolderManager(rt)
    rt.discovery = Discovery(rt)
    app = BotApp(rt)
    register_bot_modules(app)
    return rt, app


def register_bot_modules(app: BotApp) -> None:
    """Every feature module's ``register(app)``, in the order of §11.2."""
    import importlib

    for name in BOT_MODULES:
        importlib.import_module(f"tg_curator.bot.{name}").register(app)


def _models(
    home: Path, settings: Settings, fakes: Doubles, loaded: bool
) -> tuple[Embedder, TopicClassifier]:
    if fakes.embedder is not None:
        embedder = fakes.embedder
    elif loaded:
        embedder = models.make_embedder(home, settings)
    else:
        placeholder: Any = _ModelsNotLoaded()
        return placeholder, placeholder
    return embedder, fakes.classifier or models.make_classifier(embedder)


def _user_gateway(home: Path, settings: Settings, clock: Clock) -> UserGateway:
    from tg_curator.telegram.user_client import TelethonUserGateway

    return TelethonUserGateway(
        home, settings.telegram.api_id, settings.telegram.api_hash, clock=clock
    )


def _bot_gateway(home: Path, settings_file: SettingsFile) -> BotGateway:
    from tg_curator.telegram.bot_client import TelethonBotGateway

    settings = settings_file.current
    return TelethonBotGateway(
        home,
        api_id=settings.telegram.api_id,
        api_hash=settings.telegram.api_hash,
        bot_token=settings.telegram.bot_token,
        owner_id=lambda: settings_file.current.telegram.owner_id,
    )


async def load_models(home: Path, settings: Settings, progress: Progress | None) -> None:
    """Download the pinned model files once (§10); a failure is one sentence, never a crash."""
    try:
        await asyncio.to_thread(models.ensure_models, home, progress, settings=settings)
    except Exception as exc:
        raise CuratorError(
            f"cannot download the local models from huggingface.co: {exc}; check the network "
            "connection and start again"
        ) from exc


# --- settings at start -----------------------------------------------------------------------


def _blank_settings(path: Path) -> bool:
    """True when the file parses as TOML and holds no keys at all; a file that does not parse
    is left to the normal error path (it is not blank, it is broken)."""
    try:
        return not tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False


def check_settings(settings_file: SettingsFile) -> Settings | tuple[int, str]:
    """The settings, or ``(exit code, what to print)`` for the three first-start cases (§13):
    no file (the template is written, exit 0), an invalid file or blank ``[telegram]`` values
    (exit 2).

    A missing file is not the end of the start when the environment already carries the
    ``[telegram]`` values (the Docker ``.env``, §3): they are written into the fresh template
    and the start goes on, so both install paths end with the same file and nobody has to
    restart a container that was set up correctly.
    """
    # An empty file (0 bytes, or only comments) is a first start too: it holds nothing the
    # template would overwrite, and the owner needs the steps (SPEC: "the first start with an
    # empty settings file ... prints the setup steps and stops").
    first_start = not settings_file.exists() or _blank_settings(settings_file.path)
    if first_start:
        write_template(settings_file.path)
    try:
        settings = settings_file.load_sync()
    except ConfigError as exc:
        return 2, f"{exc}\nFix {settings_file.path} and start again."
    if first_start and settings.missing_telegram():
        return 0, SETUP_STEPS.format(path=settings_file.path)
    missing = settings.missing_telegram()
    if missing:
        return 2, (
            f"settings: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} empty in "
            f"{settings_file.path}: fill in api_id and api_hash from https://my.telegram.org "
            "and bot_token from @BotFather, then start again."
        )
    return settings


async def ready_settings(
    settings_file: SettingsFile, *, wait: bool, sleep: Sleep = asyncio.sleep
) -> Settings | int:
    """``check_settings`` for a start: prints the reason and returns the exit code, or with
    ``wait`` (the Docker image) prints it once and polls the file's mtime until it is fixed,
    so a container never crash-loops."""
    shown: str | None = None
    while True:
        outcome = check_settings(settings_file)
        if isinstance(outcome, Settings):
            return outcome
        code, message = outcome
        if not wait:
            _say(message)
            return code
        if message != shown:
            _say(f"{message}\nWaiting for {settings_file.path} to change…")
            shown = message
        await wait_for_change(settings_file.path, sleep=sleep)


async def wait_for_change(path: Path, *, sleep: Sleep = asyncio.sleep) -> None:
    """Return once ``path``'s mtime differs from what it is now (polled every 5 s)."""
    before = _mtime(path)
    while _mtime(path) == before:
        await sleep(SETTINGS_POLL_SECONDS)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _say(text: str) -> None:
    """Terminal output of a start (setup steps, claim code): stdout, so it reaches the
    journal and ``docker logs`` alongside the log lines."""
    print(text, flush=True)


async def go_live_report(rt: Runtime) -> tuple[bool, list[str]]:
    """What ``/go`` checks and confirms, as terminal lines for ``curator run --live``:
    ``(True, [live, schedule, how to pause])`` when something can be posted, else
    ``(False, [the one line naming the missing step])``."""
    if await logged_in_line(rt) is None:
        return False, [
            "live, but nothing can be posted yet: no account is bound — run `curator login` "
            "(or send /bind to the bot)"
        ]
    if not await has_postable_topic(rt):
        return False, [
            "live, but nothing can be posted yet: no topic has a channel the bot can post "
            "into — `curator topics add NAME --create-channel` (or `--channel REF` with the "
            "bot as an admin of that channel)"
        ]
    settings = rt.settings
    schedule = rt.t(
        "control_go_schedule",
        when=day_word(rt, rt.digest.next_run()) if rt.digest else rt.t("control_every_day"),
        time=f"{settings.digest.hour:02d}:{settings.digest.minute:02d}",
        tz=settings.general.timezone,
        weekday=rt.t(f"control_weekday_{settings.review.weekday}"),
        review_hour=f"{settings.review.hour:02d}:00",
    )
    if await rt.store.kv_get(KV.SERVICE_PAUSED):
        tail = "publishing is paused: send /resume in the bot to start posting"
    else:
        tail = "stop posting with /pause in the bot (intake keeps running)"
    return True, ["publishing is live", control.plain(schedule), tail]


# --- loops -----------------------------------------------------------------------------------


class Supervisor:
    """Runs each loop as a task that never dies: a failure is logged and the loop restarts
    after ``restart_delay``; every successful tick is written to ``rt.health[name]``.

    ``rt.loops[name]`` (a ``LoopState``) carries the rest of what ``/status`` needs to give a
    verdict (RD-5): whether the task runs, when it started, the current run of failures and
    the last error, and whether it is held back on purpose (``hold``) rather than broken.

    ``TelegramUnavailable`` is an outage, not a bug: it is logged once per outage at WARNING
    without a traceback (the first loop to hit it opens the outage, the last one to recover
    closes it) and the loop simply tries again on its next tick. A ``FloodWait`` is Telegram's
    own instruction: one WARNING, and the next try comes after the wait.

    Each loop body runs under its loop's gate; ``paused(names)`` holds the gates so a one-off
    job (the outbox reconcile after a /bind) never runs alongside a tick of those loops.
    """

    def __init__(
        self, rt: Runtime, *, sleep: Sleep = asyncio.sleep, restart_delay: float = RESTART_DELAY
    ) -> None:
        self._rt = rt
        self._sleep = sleep
        self._restart_delay = restart_delay
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._unreachable: set[str] = set()
        self._gates: dict[str, asyncio.Lock] = {}

    def start(
        self,
        name: str,
        body: Callable[[], Awaitable[Any]],
        interval: float,
        *,
        first_delay: float = 0.0,
        serves: bool = False,
    ) -> None:
        """Start loop ``name`` unless it is already running.

        ``serves`` marks a body that runs until cancelled (a server) instead of ticking;
        ``interval`` is then only the pause before it is started again after a failure.
        """
        if self.running(name):
            return
        state = LoopState(
            interval=None if serves else interval,
            started_at=self._rt.clock.now(),
            first_delay=first_delay,
            running=True,
        )
        self._rt.loops[name] = state
        task = asyncio.create_task(
            self._loop(name, state, body, interval, first_delay), name=f"loop:{name}"
        )
        # Set from the task itself, so the state is right however the task ended (a stop, or
        # a BaseException the loop does not catch) before anyone awaiting the stop resumes.
        task.add_done_callback(lambda _: setattr(state, "running", False))
        self._tasks[name] = task

    def running(self, name: str) -> bool:
        task = self._tasks.get(name)
        return task is not None and not task.done()

    def names(self) -> list[str]:
        return [name for name in self._tasks if self.running(name)]

    async def stop(self, names: Iterable[str] | None = None) -> None:
        """Cancel the named loops (all when ``None``) and wait for them to end."""
        wanted = set(self._tasks) if names is None else set(names) & set(self._tasks)
        tasks = [self._tasks.pop(name) for name in wanted]
        self._unreachable -= wanted
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def hold(self, loops: Mapping[str, float], *, paused: bool) -> None:
        """Record the loops of ``loops`` that are not running as held back on purpose
        (``paused``: setup mode, they wait for the account) or as due to start (``False``:
        the account is bound and its loops come up once the bring-up succeeds)."""
        for name, interval in loops.items():
            if self.running(name):
                continue
            state = self._rt.loops.setdefault(name, LoopState(interval=interval))
            state.paused = paused

    @asynccontextmanager
    async def paused(self, names: Iterable[str]) -> AsyncIterator[None]:
        """Wait for the running ticks of ``names`` to end and keep new ones from starting
        until the block is left (a tick due meanwhile runs right after)."""
        gates = [self._gate(name) for name in sorted(set(names))]
        held: list[asyncio.Lock] = []
        try:
            for gate in gates:
                await gate.acquire()
                held.append(gate)
            yield
        finally:
            for gate in reversed(held):
                gate.release()

    def _gate(self, name: str) -> asyncio.Lock:
        return self._gates.setdefault(name, asyncio.Lock())

    async def _loop(
        self,
        name: str,
        state: LoopState,
        body: Callable[[], Awaitable[Any]],
        interval: float,
        first_delay: float,
    ) -> None:
        if first_delay:
            await self._sleep(first_delay)
        gate = self._gate(name)
        while True:
            delay = interval
            try:
                async with gate:
                    await body()
            except asyncio.CancelledError:
                raise
            except TelegramUnavailable as exc:
                self._outage(name, exc)
                self._failed(state, exc)
            except FloodWait as exc:
                log.warning(
                    "loop %s: Telegram asks to wait %d s; next try after that", name, exc.seconds
                )
                self._failed(state, exc)
                delay = max(float(exc.seconds), interval)
            except Exception as exc:
                log.exception("loop %s failed; restarting in %.0f s", name, self._restart_delay)
                self._failed(state, exc)
                delay = self._restart_delay
            else:
                self._recovered(name)
                now = self._rt.clock.now()
                self._rt.health[name] = now
                state.last_ok_at = now
                state.failing_since = None
            await self._sleep(delay)

    def _failed(self, state: LoopState, exc: Exception) -> None:
        now = self._rt.clock.now()
        if state.failing_since is None:
            state.failing_since = now
        state.last_error_at = now
        state.last_error = short_error(exc)

    def _outage(self, name: str, exc: TelegramUnavailable) -> None:
        if not self._unreachable:
            log.warning("Telegram is unreachable (%s): %s; retrying on every tick", name, exc)
        self._unreachable.add(name)

    def _recovered(self, name: str) -> None:
        if name in self._unreachable:
            self._unreachable.discard(name)
            if not self._unreachable:
                log.info("Telegram is reachable again")


def short_error(exc: BaseException) -> str:
    """One line for ``/status``: the message of our own errors (written for the owner), the
    class name plus message of anything else, cut at ``ERROR_TEXT_LIMIT`` characters."""
    text = " ".join(str(exc).split())
    if not text:
        text = type(exc).__name__
    elif not isinstance(exc, CuratorError):
        text = f"{type(exc).__name__}: {text}"
    if len(text) > ERROR_TEXT_LIMIT:
        text = text[: ERROR_TEXT_LIMIT - 1].rstrip() + "…"
    return text


class Debouncer:
    """Runs ``action`` once, ``delay`` seconds after the last ``trigger``; the tags of every
    trigger in between are handed over together. A trigger during the action schedules one
    more run, so nothing is lost and a burst costs one run."""

    def __init__(
        self,
        rt: Runtime,
        name: str,
        delay: float,
        action: Callable[[set[str]], Awaitable[None]],
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._rt = rt
        self._name = name
        self._delay = delay
        self._action = action
        self._sleep = sleep
        self._tags: set[str] = set()
        self._generation = 0
        self._task: asyncio.Task[None] | None = None

    def trigger(self, tag: str | None = None) -> None:
        if tag is not None:
            self._tags.add(tag)
        self._generation += 1
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"debounce:{self._name}")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            seen = self._generation
            await self._sleep(self._delay)
            if seen != self._generation:
                continue
            tags, self._tags = self._tags, set()
            try:
                await self._action(tags)
            except asyncio.CancelledError:
                raise
            except TelegramUnavailable as exc:
                log.warning("%s: Telegram is unreachable: %s", self._name, exc)
            except Exception:
                log.exception("%s failed", self._name)
            else:
                self._rt.health[self._name] = self._rt.clock.now()
            if seen == self._generation:
                return


# --- the service -----------------------------------------------------------------------------


class Service:
    """The start sequence of §13 and everything that keeps running afterwards."""

    def __init__(
        self,
        rt: Runtime,
        app: BotApp,
        *,
        control_socket: Path | None = None,
        went_live: bool = False,
        live_requested: bool = False,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.rt = rt
        self.app = app
        self.supervisor = Supervisor(rt, sleep=sleep)
        self._sleep = sleep
        self._control_socket = control_socket
        self._went_live = went_live
        self._live_requested = live_requested
        self._retrain = Debouncer(rt, "retrain", RETRAIN_DEBOUNCE, self._retrain_now, sleep=sleep)
        self._resync = Debouncer(rt, "settings", SETTINGS_DEBOUNCE, self._resync_now, sleep=sleep)
        self._background: set[asyncio.Task[None]] = set()
        self._bring_up = asyncio.Lock()
        self._bind_task: asyncio.Task[None] | None = None
        self._listening = False
        rt.events.on(EVENT_ACCOUNT_BOUND, self._on_account_bound)
        rt.events.on(EVENT_SESSION_LOST, self._on_session_lost)
        rt.events.on(EVENT_EXAMPLES_CHANGED, self._on_examples_changed)
        rt.events.on(EVENT_SETTINGS_CHANGED, self._on_settings_changed)

    @property
    def setup_mode(self) -> bool:
        return self._account().setup_mode

    async def start(self) -> None:
        """Store, bot, claim code, account; then (when bound) owned channels, topics, staging,
        chats; reconcile the outboxes; start the loops — in that order (§13)."""
        rt = self.rt
        await rt.store.start()
        if self._went_live:
            await rt.store.kv_set(KV.SERVICE_WENT_LIVE_AT, rt.clock.now().isoformat())
        await refresh_embeddings(rt)
        await rt.store.kv_delete(KV.FOLDERS_DISABLED)  # §12: the next start tries folders again
        await self._start_bot()
        authorised = await self._user().connect()
        await register_owned(rt)
        if authorised:
            await rt.store.kv_delete(KV.ACCOUNT_LOSS_NOTIFIED)  # bound again (`curator login`)
            await self._bring_up_account()  # the topic sync also loads the classifier
        else:
            self._account().setup_mode = True
            self.supervisor.hold(ACCOUNT_LOOPS, paused=True)
            log.warning(
                "the account is not bound: setup mode (bot only). Send /bind to the bot, or "
                "stop the service and run `curator login`"
            )
            await self._warn_session_expired()
            await _learning(rt).retrain()
        # The stale-queued rule (§14.6) runs here, before any publisher tick: what `/go`
        # runs before its live switch also covers `curator run --live`.
        await self._reconcile()
        await _need(rt.sorter).start()
        self._start_core_loops()
        if authorised:
            self._start_account_loops()
        await self._report_live()

    async def _warn_session_expired(self) -> None:
        """A session revoked or expired while the service was down (§11.3): Telethon reports
        it as "not authorised" rather than as an error, so the loss procedure never ran. An
        account that was bound before gets the session-lost warning — once, until the next
        bind — so the owner hears that intake stopped (spec "Running")."""
        rt = self.rt
        if await rt.store.kv_get(KV.ACCOUNT_ID) is None:
            return  # a first start: nothing was ever bound, setup mode is expected
        log.warning("the stored account session is no longer authorised (revoked or expired)")
        if not rt.settings.telegram.owner_id:
            return
        if await rt.store.kv_get(KV.ACCOUNT_LOSS_NOTIFIED) is not None:
            return
        await rt.notifier.intake_warning("session_lost")
        await rt.store.kv_set(KV.ACCOUNT_LOSS_NOTIFIED, rt.clock.now().isoformat())

    async def _report_live(self) -> None:
        """``curator run --live`` confirms like ``/go`` (spec "Shipping it": every step works
        or says what to fix): the schedule and how to pause, or the missing step. A service
        that was already live says only what keeps it from posting."""
        rt = self.rt
        if not rt.settings.publishing.live:
            return
        ready, lines = await go_live_report(rt)
        if ready and not (self._went_live or self._live_requested):
            return
        for line in lines:
            _say(line)

    async def shutdown(self) -> None:
        """Cancel every task, then close the clients and the store (SIGTERM/SIGINT)."""
        rt = self.rt
        await self._retrain.close()
        await self._resync.close()
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
        await self._flush_albums()
        await self.supervisor.stop()
        await close_runtime(rt)
        log.info("stopped")

    async def _flush_albums(self) -> None:
        """Submit the live albums still settling, so a clean stop never drops one (what a
        crash leaves is read back from ``kv intake.open_albums`` at the next start)."""
        if self.rt.intake is None:
            return
        try:
            await asyncio.wait_for(self.rt.intake.flush_albums(), timeout=ALBUM_FLUSH_TIMEOUT)
        except Exception:
            log.warning("albums still settling were not submitted; the next start does it")

    # --- start steps ---

    async def _start_bot(self) -> None:
        await start_bot(self.rt)
        await self.app.start()
        code = self.app.claim_code
        if code is not None:
            _say(
                f"Claim code: {code}\nOpen @{_bot_username(self.rt)} in Telegram and send "
                f"/start {code} — whoever sends it becomes the owner."
            )

    async def _bring_up_account(self) -> None:
        """The account-side start steps; at start and again on every ``account_bound``."""
        rt = self.rt
        async with self._bring_up:
            await register_owned(rt)
            account = await self._user().me()
            if account is not None:
                log.info("logged in as %s (@%s)", account.name, account.username or "-")
            for line in control.topics_summary(await _need(rt.topics).sync_from_settings()):
                log.info("topics: %s", line)
            await self._ensure_staging()
            await _need(rt.intake).sync_chats()
            if not self._listening:
                self._user().on_message(self._on_message)
                self._listening = True

    async def _ensure_staging(self) -> None:
        """The staging channel exists whenever posting is live in repost style (§8, §14.2)."""
        rt = self.rt
        publishing = rt.settings.publishing
        if not publishing.live or publishing.style != "repost":
            return
        try:
            await _need(rt.topics).ensure_staging_channel()
        except CuratorError as exc:
            log.warning("cannot set up the staging channel: %s; media goes as text + link", exc)

    async def _reconcile(self) -> None:
        await _need(self.rt.publisher).reconcile()
        await _need(self.rt.digest).reconcile()

    def _start_core_loops(self) -> None:
        rt = self.rt
        bodies: dict[str, Callable[[], Awaitable[Any]]] = {
            "sorter": _need(rt.sorter).tick,
            "publisher": _need(rt.publisher).tick,
            "digest": _need(rt.digest).tick,
            "review": _need(rt.review).tick,
            "link": self._keep_linked,
        }
        for name, interval in CORE_LOOPS.items():
            self.supervisor.start(name, bodies[name], interval)
        self.supervisor.start("housekeeping", lambda: housekeeping(rt), HOUSEKEEPING_SECONDS)
        if self._control_socket is not None:
            path = self._control_socket
            self.supervisor.start(
                "control", lambda: control.serve(rt, path), RESTART_DELAY, serves=True
            )

    async def _keep_linked(self) -> None:
        """Reconnect a client Telethon gave up on (§17.3), so live updates (the account's
        messages, the owner's commands) resume without waiting for an outgoing request."""
        rt = self.rt
        if rt.bot is not None:
            await rt.bot.ensure_connected()
        if rt.user is not None and not self.setup_mode:
            try:
                await rt.user.ensure_connected()
            except SessionLost:
                return  # the gateway's on_session_lost handler takes over

    def _start_account_loops(self) -> None:
        rt = self.rt
        bodies: dict[str, Callable[[], Awaitable[Any]]] = {
            "intake": _need(rt.intake).tick,
            "actions": _need(rt.actions).tick,
            "chats": _need(rt.intake).sync_chats,
            "folders": _need(rt.folders).sync,
            "watchdog": self._account().watchdog_tick,
            "topic_channels": _need(rt.topics).tick,
        }
        for name, interval in ACCOUNT_LOOPS.items():
            # The dialog list was synced moments ago by the bring-up; the next sweep is due
            # one interval later.
            first_delay = interval if name == "chats" else 0.0
            self.supervisor.start(name, bodies[name], interval, first_delay=first_delay)

    # --- events ---

    async def _on_message(self, msg: IncomingMessage) -> None:
        try:
            await _need(self.rt.intake).handle_message(msg)
        except asyncio.CancelledError:
            raise
        except TelegramUnavailable as exc:
            log.warning("intake: Telegram is unreachable: %s", exc)
        except Exception:
            log.exception("intake: message %d in chat %d failed", msg.message_id, msg.chat.id)
        else:
            self.rt.health["intake_events"] = self.rt.clock.now()

    async def _on_account_bound(self, **_: Any) -> None:
        self._account().setup_mode = False  # the account says so already; bound is bound
        self.supervisor.hold(ACCOUNT_LOOPS, paused=False)  # due now: /status flags a failure
        await self.rt.store.kv_delete(KV.ACCOUNT_LOSS_NOTIFIED)  # a later loss warns again
        self._cancel_bind()
        self._bind_task = self._spawn(self._after_bind(), "account-bound")

    async def _after_bind(self) -> None:
        """The start steps of a bound account, retried with backoff until they succeed or the
        session is lost again: /bind already told the owner they are logged in, and nothing
        else would ever start intake, the actions or the watchdog (§11.3)."""
        log.info("account bound: starting intake")
        delay = RESTART_DELAY
        while not self.setup_mode:
            try:
                await self._bring_up_account()
                # The publisher and the digest kept running in setup mode: their outboxes are
                # reconciled between ticks, never alongside one (a row mid-send is not stale).
                async with self.supervisor.paused(("publisher", "digest")):
                    await self._reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                wait = delay
                if isinstance(exc, FloodWait):
                    wait = max(wait, float(exc.seconds))
                if isinstance(exc, TelegramUnavailable | FloodWait):
                    log.warning("account bring-up failed: %s; retrying in %.0f s", exc, wait)
                else:
                    log.exception("account bring-up failed; retrying in %.0f s", wait)
                await self._sleep(wait)
                delay = min(delay * 2, BRING_UP_MAX_DELAY)
                continue
            self._start_account_loops()
            return

    async def _on_session_lost(self, **payload: Any) -> None:
        log.warning("session lost (%s): setup mode until /bind", payload.get("reason", "?"))
        # The account service warned the owner already; a restart before /bind does not.
        await self.rt.store.kv_set(KV.ACCOUNT_LOSS_NOTIFIED, self.rt.clock.now().isoformat())
        self._cancel_bind()
        self._spawn(self._pause_account_loops(), "session-lost")

    async def _pause_account_loops(self) -> None:
        """Stop the account loops and record them as waiting for /bind, not as broken."""
        await self.supervisor.stop(ACCOUNT_LOOPS)
        self.supervisor.hold(ACCOUNT_LOOPS, paused=True)

    def _cancel_bind(self) -> None:
        if self._bind_task is not None and not self._bind_task.done():
            self._bind_task.cancel()
        self._bind_task = None

    async def _on_examples_changed(self, **payload: Any) -> None:
        self._retrain.trigger(str(payload.get("reason", "")))

    async def _on_settings_changed(self, **_: Any) -> None:
        self._resync.trigger()

    async def _retrain_now(self, reasons: set[str]) -> None:
        """§13 retrain loop: rebuild the user layer; re-sort only for new examples (§9.3)."""
        rt = self.rt
        await _learning(rt).retrain()
        if "examples" in reasons:
            since = rt.clock.now() - timedelta(days=rt.settings.review.cluster_window_days)
            moved = await _need(rt.sorter).resort_unsorted(since)
            log.info("retrain: %d unsorted posts re-sorted with the new examples", moved)

    async def _resync_now(self, _: set[str]) -> None:
        """``settings_changed`` -> ``topics.sync_from_settings`` (§17.7), then the staging
        channel when live in repost style (§8); nothing to resolve in setup mode."""
        if self.setup_mode:
            return
        result = await _need(self.rt.topics).sync_from_settings()
        if result.total or result.unresolved:
            for line in control.topics_summary(result):
                log.info("topics: %s", line)
        await self._ensure_staging()

    # --- helpers ---

    def _spawn(self, work: Awaitable[None], name: str) -> asyncio.Task[None]:
        async def guarded() -> None:
            try:
                await work
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s failed", name)

        task = asyncio.create_task(guarded(), name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _user(self) -> UserGateway:
        return _need(self.rt.user)

    def _account(self) -> AccountService:
        account = _need(self.rt.account)
        if not isinstance(account, AccountService):
            raise CuratorError("the account service is not the one service.py wires")
        return account


def _need[T](component: T | None) -> T:
    if component is None:
        raise CuratorError("the runtime is not fully wired (see service.build_runtime)")
    return component


def _learning(rt: Runtime) -> Learning:
    learning = _need(rt.learning)
    if not isinstance(learning, Learning):
        raise CuratorError("the learning service is not the one service.py wires")
    return learning


def _bot_username(rt: Runtime) -> str:
    return rt.bot_account.username if rt.bot_account and rt.bot_account.username else "your bot"


# --- shared start/stop steps (service and CLI) -----------------------------------------------


async def start_bot(rt: Runtime) -> None:
    """``bot.start()`` -> ``rt.bot_account`` and its mirror in ``kv`` (§8)."""
    account = await _need(rt.bot).start()
    rt.bot_account = account
    await rt.store.kv_set(KV.BOT_ID, account.id)
    await rt.store.kv_set(KV.BOT_USERNAME, account.username)


async def register_owned(rt: Runtime) -> None:
    """Every output and staging channel becomes writable for the account (§1, §13)."""
    user = _need(rt.user)
    for role in ("output", "staging"):
        for chat in await rt.store.list_chats(role=role):
            user.register_owned(chat.id)


async def close_runtime(rt: Runtime) -> None:
    """Disconnect both clients, close the LLM client and the store; failures only logged."""
    steps: list[tuple[str, Callable[[], Awaitable[Any]] | None]] = [
        ("bot", rt.bot.stop if rt.bot is not None else None),
        ("account", rt.user.disconnect if rt.user is not None else None),
        ("llm", getattr(rt.llm, "aclose", None)),
        ("store", rt.store.close),
    ]
    for what, step in steps:
        if step is None:
            continue
        try:
            await step()
        except Exception as exc:
            log.warning("closing %s: %s", what, exc)


@asynccontextmanager
async def open_runtime(
    home: Path,
    settings_file: SettingsFile,
    *,
    needs: Collection[str],
    progress: Progress | None = None,
) -> AsyncIterator[Runtime]:
    """A runtime for one standalone CLI command (the caller holds ``service.lock``).

    ``needs`` names what the command uses: ``"models"`` loads the local models and the
    classifier's topics, ``"bot"`` starts the bot, ``"user"`` connects the account (which must
    be signed in) and registers the owned channels. Everything is closed on the way out.
    """
    with_models = "models" in needs
    if with_models:
        await load_models(home, settings_file.current, progress)
    rt, _ = build_runtime(home, settings_file=settings_file, models_loaded=with_models)
    await rt.store.start()
    try:
        if "bot" in needs:
            await start_bot(rt)
        if "user" in needs:
            if not await _need(rt.user).connect():
                raise CuratorError("the account is not signed in: run `curator login` first")
            await register_owned(rt)
        if with_models:
            await _learning(rt).retrain()
        yield rt
    finally:
        await close_runtime(rt)


# --- housekeeping ----------------------------------------------------------------------------


async def housekeeping(rt: Runtime) -> None:
    """The daily purge (§13): group buffer, old embeddings, old posts, old LLM months."""
    store, storage = rt.store, rt.settings.storage
    now = rt.clock.now()
    buffered = await store.purge_group_messages(now - GROUP_RETENTION)
    posts = schema.posts
    cleared = await store.execute(
        sa.update(posts)
        .where(
            posts.c.embedding.is_not(None),
            posts.c.posted_at < now - timedelta(days=storage.keep_embeddings_days),
        )
        .values(embedding=None)
    )
    removed = 0
    if storage.keep_posts_days > 0:
        removed = await _purge_posts(rt, now - timedelta(days=storage.keep_posts_days))
    oldest_month = (now - LLM_USAGE_KEEP).strftime("%Y-%m")
    await store.execute(sa.delete(schema.llm_usage).where(schema.llm_usage.c.month < oldest_month))
    log.info(
        "housekeeping: %d buffered group messages, %s embeddings and %d posts purged",
        buffered,
        cleared,
        removed,
    )


async def _purge_posts(rt: Runtime, before: datetime) -> int:
    """Posts older than ``keep_posts_days`` go, with the rows that point at them; examples
    keep their text and embedding (only the link to the post is dropped)."""
    posts = schema.posts
    old = sa.select(posts.c.id).where(posts.c.posted_at < before).scalar_subquery()
    store = rt.store
    async with store.begin() as conn:
        await store.execute(
            sa.delete(schema.digest_items).where(schema.digest_items.c.post_id.in_(old)), conn
        )
        await store.execute(
            sa.delete(schema.publications).where(schema.publications.c.post_id.in_(old)), conn
        )
        await store.execute(
            sa.update(posts).where(posts.c.duplicate_of.in_(old)).values(duplicate_of=None), conn
        )
        await store.execute(
            sa.update(schema.examples)
            .where(schema.examples.c.post_id.in_(old))
            .values(post_id=None),
            conn,
        )
        removed = await store.execute(sa.delete(posts).where(posts.c.posted_at < before), conn)
    return int(removed) if isinstance(removed, int) else 0


async def refresh_embeddings(rt: Runtime) -> None:
    """Recompute stored embeddings when the embedder changed since the last start (§10)."""
    stored = await rt.store.kv_get(KV.ML_EMBEDDER_ID)
    current = rt.embedder.id
    if stored == current:
        return
    if stored is not None:
        posts = await _reembed(rt, schema.posts, schema.posts.c.embedding.is_not(None))
        examples = await _reembed(rt, schema.examples, sa.true())
        log.info(
            "embedder changed (%s -> %s): recomputed %d post and %d example embeddings",
            stored,
            current,
            posts,
            examples,
        )
    await rt.store.kv_set(KV.ML_EMBEDDER_ID, current)


async def _reembed(rt: Runtime, table: sa.Table, condition: Any) -> int:
    result = await rt.store.execute(sa.select(table.c.id, table.c.text).where(condition))
    rows = result if isinstance(result, list) else []
    for start in range(0, len(rows), EMBED_BATCH):
        chunk = rows[start : start + EMBED_BATCH]
        vectors: np.ndarray = await asyncio.to_thread(rt.embedder.embed, [r.text for r in chunk])
        async with rt.store.begin() as conn:
            for row, vector in zip(chunk, vectors, strict=True):
                await rt.store.execute(
                    sa.update(table)
                    .where(table.c.id == row.id)
                    .values(embedding=embedding_to_bytes(vector)),
                    conn,
                )
    return len(rows)


# --- the process -----------------------------------------------------------------------------


async def run(home: Path, *, live: bool = False) -> int:
    """``curator run``: the whole service until SIGTERM/SIGINT; returns the exit code."""
    ensure_home(home)
    lock = control.ServiceLock(home)
    if not lock.acquire():
        _say(f"tg-curator is already running for {home}; stop it first (one service per home)")
        return 1
    try:
        return await _run_locked(home, live=live)
    finally:
        lock.release()


async def _run_locked(home: Path, *, live: bool) -> int:
    settings_file = SettingsFile(home / SETTINGS_FILENAME)
    wait = os.environ.get(WAIT_ENV, "") == "1"
    outcome = await ready_settings(settings_file, wait=wait)
    if isinstance(outcome, int):
        return outcome
    went_live = live and not outcome.publishing.live
    if went_live:
        await settings_file.set_value("publishing.live", True)
        log.info("publishing is live (curator run --live)")
    try:
        await load_models(home, settings_file.current, _say)
    except CuratorError as exc:  # the network, not the settings: systemd and Docker retry
        _say(str(exc))
        return 1
    code = await _serve(home, settings_file, went_live=went_live, live_requested=live)
    if code == 2 and wait:
        # Telegram refused a value from the file (a mistyped token or api_id). Restarting the
        # container would only repeat the refused login in a loop; wait for the fix instead,
        # then exit so the restart policy starts afresh with the corrected file.
        _say(f"Waiting for {settings_file.path} to change…")
        await wait_for_change(settings_file.path)
    return code


async def _serve(
    home: Path, settings_file: SettingsFile, *, went_live: bool, live_requested: bool = False
) -> int:
    """Build the runtime, start the service and keep it running until SIGTERM/SIGINT."""
    rt, app = build_runtime(home, settings_file=settings_file)
    service = Service(
        rt,
        app,
        control_socket=control.socket_path(home),
        went_live=went_live,
        live_requested=live_requested,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await service.start()
        log.info("running; stop with SIGTERM or Ctrl-C")
        await stop.wait()
    except ConfigError as exc:  # e.g. a revoked bot token: restarting would not help
        _say(str(exc))
        return 2
    except CuratorError as exc:  # e.g. Telegram unreachable at start: systemd retries
        _say(f"cannot start: {exc}")
        return 1
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        await service.shutdown()
    return 0
