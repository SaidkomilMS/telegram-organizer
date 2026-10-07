"""service.py: the wiring, the supervised loops, the start sequence, setup mode (DESIGN §13)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import (
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeUserGateway,
)
from tg_curator import service
from tg_curator.account import AccountService
from tg_curator.bot.core import BotApp
from tg_curator.config import SettingsFile
from tg_curator.db.store import Store
from tg_curator.domain import KV, NewPost, PostStatus
from tg_curator.errors import TelegramUnavailable
from tg_curator.ml import models
from tg_curator.ml.embedder import HashEmbedder
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED, Runtime
from tg_curator.service import Doubles, Service, Supervisor, build_runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

SERVICES = (
    "intake",
    "backfill",
    "sorter",
    "preview",
    "publisher",
    "digest",
    "topics",
    "learning",
    "stats",
    "review",
    "actions",
    "folders",
    "discovery",
    "account",
)
BOT_COMMANDS = {
    "start",
    "setup",
    "bind",
    "go",
    "pause",
    "resume",
    "status",
    "reload",
    "help",
    "topics",
    "preview",
    "llm",
    "settings",
    "stats",
    "digest",
    "review",
    "undo",
}
OUTPUT = -1_001_000_007_001
STAGING = -1_001_000_007_002


def info(chat_id: int, title: str) -> ChatInfo:
    return ChatInfo(
        id=chat_id,
        kind="channel",
        title=title,
        username=None,
        noforwards=False,
        is_creator=True,
        is_admin=True,
        archived=False,
        muted_until=None,
    )


async def parked(_: float) -> None:
    """A sleep that never ends: every loop runs its body exactly once."""
    await asyncio.Event().wait()


async def settle(condition: Callable[[], bool] = lambda: False, rounds: int = 50) -> None:
    """Let background tasks run (the store answers from a worker thread, so a little real
    time passes per round) until ``condition`` holds or the rounds are used up."""
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.005)


@pytest.fixture
def doubles(clock: FakeClock, user_gw: FakeUserGateway, bot_gw: FakeBotGateway) -> Doubles:
    return Doubles(
        user=user_gw,
        bot=bot_gw,
        clock=clock,
        embedder=FakeEmbedder(),
        classifier=FakeClassifier(),
        llm=FakeLLM(),
    )


@pytest.fixture
def wired(
    home: Path, settings_file: SettingsFile, store: Store, doubles: Doubles
) -> tuple[Runtime, BotApp]:
    return build_runtime(home, settings_file=settings_file, store=store, fakes=doubles)


# --- wiring ------------------------------------------------------------------------------------


def test_build_runtime_wires_every_service_and_every_bot_command(
    wired: tuple[Runtime, BotApp], doubles: Doubles
) -> None:
    rt, app = wired
    for name in SERVICES:
        assert getattr(rt, name) is not None, name
    assert isinstance(rt.account, AccountService)
    assert rt.user is doubles.user and rt.bot is doubles.bot and rt.llm is doubles.llm
    assert app.rt is rt
    assert BOT_COMMANDS <= set(app.commands)
    assert {"wt", "mv", "rv", "ds", "lm"} <= set(app.callbacks)


def test_build_runtime_uses_the_hash_embedder_under_fake_models(
    home: Path,
    settings_file: SettingsFile,
    store: Store,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(models.FAKE_MODELS_ENV, "1")
    monkeypatch.setattr(models, "_embedder", None)
    monkeypatch.setattr(service, "doubles_factory", lambda: Doubles(user=user_gw, bot=bot_gw))
    rt, _ = build_runtime(home, settings_file=settings_file, store=store)
    assert isinstance(rt.embedder, HashEmbedder)
    assert rt.user is user_gw  # the test hook reached build_runtime without explicit fakes
    unloaded, _ = build_runtime(home, settings_file=settings_file, store=store, models_loaded=False)
    with pytest.raises(Exception, match="not loaded"):
        unloaded.embedder.embed(["x"])


def test_the_shipped_unit_file_is_the_deploy_copy() -> None:
    root = Path(__file__).resolve().parents[1]
    shipped = (root / "src" / "tg_curator" / "data" / "tg-curator.service").read_text()
    assert shipped == (root / "deploy" / "tg-curator.service").read_text()
    for line in (
        "User=tg-curator",
        "StateDirectory=tg-curator",
        "Environment=TG_CURATOR_HOME=/var/lib/tg-curator",
        "Restart=on-failure",
        "RestartSec=10",
        "RestartPreventExitStatus=2",
        "ProtectSystem=strict",
        "PrivateTmp=yes",
        "NoNewPrivileges=yes",
    ):
        assert line in shipped.splitlines(), line


# --- the supervisor ----------------------------------------------------------------------------


async def test_loops_update_health_and_one_failing_loop_does_not_stop_the_others(
    rt: Runtime, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    async def tick(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    counts = {"good": 0, "bad": 0, "flaky": 0}

    async def good() -> None:
        counts["good"] += 1

    async def bad() -> None:
        counts["bad"] += 1
        raise RuntimeError("boom")

    async def flaky() -> None:
        counts["flaky"] += 1
        if counts["flaky"] <= 3:
            raise TelegramUnavailable("connection reset")

    supervisor = Supervisor(rt, sleep=tick)
    caplog.set_level(logging.INFO, logger="tg_curator.service")
    started = clock.now()
    supervisor.start("good", good, 2.0)
    supervisor.start("bad", bad, 2.0)
    supervisor.start("flaky", flaky, 2.0)
    await settle(rounds=40)
    try:
        assert counts["good"] > 5 and counts["bad"] > 5 and counts["flaky"] > 5
        assert rt.health["good"] > started
        assert "bad" not in rt.health
        assert rt.health["flaky"] > started
        assert supervisor.running("bad") and supervisor.running("good")
        crashes = [r for r in caplog.records if "loop bad failed" in r.getMessage()]
        assert crashes and crashes[0].exc_info is not None
        outages = [r for r in caplog.records if "unreachable" in r.getMessage()]
        assert len(outages) == 1  # once per outage, not once per failed tick
        assert outages[0].levelno == logging.WARNING and outages[0].exc_info is None
        assert any("reachable again" in r.getMessage() for r in caplog.records)
    finally:
        await supervisor.stop()
    assert supervisor.names() == []


# --- the start sequence ------------------------------------------------------------------------


async def seed_owned_channels(rt: Runtime) -> None:
    await rt.store.start()
    await rt.store.upsert_chat(info(OUTPUT, "ML & AI"), role="output")
    await rt.store.upsert_chat(info(STAGING, "tg-curator media"), role="staging")


def trace(rt: Runtime, service_: Service, user_gw: FakeUserGateway, monkeypatch: Any) -> None:
    """Put loop starts and reconciles into the account fake's call log, in order."""
    original = Supervisor.start

    def start(self: Supervisor, name: str, *args: Any, **kwargs: Any) -> None:
        user_gw.calls.append(("loop", {"name": name}))
        original(self, name, *args, **kwargs)

    monkeypatch.setattr(Supervisor, "start", start)
    for owner in ("publisher", "digest"):
        component = getattr(rt, owner)
        reconcile = component.reconcile

        async def traced(_reconcile: Callable[[], Awaitable[None]] = reconcile, _o: str = owner):
            user_gw.calls.append((f"{_o}.reconcile", {}))
            await _reconcile()

        monkeypatch.setattr(component, "reconcile", traced)


def names(user_gw: FakeUserGateway) -> list[str]:
    return [name for name, _ in user_gw.calls]


async def test_the_start_sequence_runs_in_the_order_of_section_13(
    wired: tuple[Runtime, BotApp],
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rt, app = wired
    await seed_owned_channels(rt)
    user_gw.add_chat(info(-1_001_000_007_100, "A source"))
    svc = Service(rt, app, sleep=parked)
    trace(rt, svc, user_gw, monkeypatch)
    await svc.start()
    try:
        calls = names(user_gw)
        first_loop = calls.index("loop")
        assert calls[0] == "connect"
        assert calls.index("register_owned") < first_loop
        assert {OUTPUT, STAGING} <= user_gw.owned
        assert calls.index("list_chats") < calls.index("publisher.reconcile")
        assert calls.index("publisher.reconcile") < calls.index("digest.reconcile") < first_loop
        loops = [kw["name"] for name, kw in user_gw.calls if name == "loop"]
        assert set(service.CORE_LOOPS) | set(service.ACCOUNT_LOOPS) <= set(loops)
        assert bot_gw.calls[0][0] == "start"
        assert rt.bot_account is not None
        assert await rt.store.kv_get(KV.BOT_USERNAME) == rt.bot_account.username
        assert await rt.store.kv_get(KV.ML_EMBEDDER_ID) == rt.embedder.id
        assert "Claim code:" in capsys.readouterr().out  # no owner yet in the template
        assert user_gw._message_handlers  # live intake subscribed
        assert not svc.setup_mode
    finally:
        await svc.shutdown()
    assert "disconnect" in names(user_gw)
    assert bot_gw.calls[-1][0] == "stop"


async def test_the_link_loop_reconnects_both_clients_and_the_account_only_when_bound(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    """§17.3: a client Telethon gave up on is connected again on a timer, so the account's
    live updates and the owner's commands come back without an outgoing request."""
    rt, app = wired
    svc = Service(rt, app, sleep=parked)
    assert service.CORE_LOOPS["link"] == 60.0
    await svc._keep_linked()
    assert "ensure_connected" in [c[0] for c in bot_gw.calls]
    assert "ensure_connected" in names(user_gw)
    user_gw.calls.clear()
    bot_gw.calls.clear()
    rt.account.setup_mode = True  # type: ignore[union-attr]
    await svc._keep_linked()
    assert "ensure_connected" in [c[0] for c in bot_gw.calls]
    assert "ensure_connected" not in names(user_gw)  # a login may be in progress


async def test_setup_mode_until_account_bound_then_intake_starts(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, app = wired
    await seed_owned_channels(rt)
    user_gw.authorised = False
    svc = Service(rt, app, sleep=parked)
    trace(rt, svc, user_gw, monkeypatch)
    await svc.start()
    try:
        assert svc.setup_mode
        assert "list_chats" not in names(user_gw)
        assert not svc.supervisor.running("intake")
        assert svc.supervisor.running("publisher") and svc.supervisor.running("sorter")
        assert {OUTPUT, STAGING} <= user_gw.owned  # before any loop, bound or not

        user_gw.authorised = True  # what a successful /bind leaves behind
        await rt.events.emit("account_bound")
        await settle(lambda: svc.supervisor.running("intake"))
        assert svc.supervisor.running("intake") and svc.supervisor.running("actions")
        calls = names(user_gw)
        bound_at = calls.index("list_chats")
        assert calls.index("publisher.reconcile", bound_at) < calls.index("loop", bound_at)
        assert user_gw._message_handlers

        await user_gw.lose_session("revoked")
        await settle(lambda: not svc.supervisor.running("intake"))
        assert svc.setup_mode
        assert not svc.supervisor.running("intake") and not svc.supervisor.running("watchdog")
        assert svc.supervisor.running("publisher")
    finally:
        await svc.shutdown()


async def test_folders_disabled_is_cleared_at_start(
    wired: tuple[Runtime, BotApp],
) -> None:
    rt, app = wired
    await rt.store.kv_set(KV.FOLDERS_DISABLED, "limit")
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        assert await rt.store.kv_get(KV.FOLDERS_DISABLED) is None
    finally:
        await svc.shutdown()


# --- event-driven work -------------------------------------------------------------------------


async def test_retrain_is_debounced_and_resorts_only_for_new_examples(
    wired: tuple[Runtime, BotApp], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, app = wired
    calls: list[str] = []

    async def retrain() -> None:
        calls.append("retrain")

    async def resort(since: Any) -> int:
        calls.append("resort")
        return 0

    async def tick(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(rt.learning, "retrain", retrain)
    monkeypatch.setattr(rt.sorter, "resort_unsorted", resort)
    svc = Service(rt, app, sleep=tick)
    await rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="correction")
    await rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="correction")
    await settle(lambda: "retrain" in calls)
    await settle(rounds=10)
    assert calls == ["retrain"]  # two corrections, one retrain, no re-sort

    calls.clear()
    await rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="correction")
    await rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="examples")
    await settle(lambda: "resort" in calls)
    assert calls == ["retrain", "resort"]
    assert "retrain" in rt.health
    await svc.shutdown()


async def test_settings_changes_resync_topics_debounced(
    wired: tuple[Runtime, BotApp], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, app = wired
    synced: list[int] = []
    original = rt.topics.sync_from_settings

    async def sync() -> Any:
        synced.append(1)
        return await original()

    async def tick(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(rt.topics, "sync_from_settings", sync)
    svc = Service(rt, app, sleep=tick)
    await rt.store.start()
    await rt.settings_file.set_value("digest.hour", 20)
    await rt.settings_file.set_value("digest.minute", 30)
    await settle(lambda: bool(synced))
    await settle(rounds=10)
    assert len(synced) == 1
    await svc.shutdown()


# --- housekeeping ------------------------------------------------------------------------------


async def test_housekeeping_drops_old_embeddings_and_old_posts(rt: Runtime) -> None:
    await rt.settings_file.set_value("storage.keep_posts_days", 60)
    await rt.store.upsert_chat(info(-1_001_000_007_200, "Old news"))
    now = rt.clock.now()

    async def post(mid: int, age_days: int) -> int:
        text = f"post number {mid}"
        new = NewPost(
            chat_id=-1_001_000_007_200,
            message_id=mid,
            kind="post",
            message_ids=[mid],
            posted_at=now - timedelta(days=age_days),
            via="live",
            text=text,
            text_hash=text_hash(text),
            urls=[],
            embedding=b"\x00\x00\x80\x3f",
        )
        stored = await rt.store.insert_post(new, status=PostStatus.dropped)
        assert stored is not None
        return stored.id

    fresh, older, ancient = await post(1, 1), await post(2, 40), await post(3, 90)
    await service.housekeeping(rt)
    kept = await rt.store.get_post(fresh)
    assert kept is not None and kept.embedding is not None
    aged = await rt.store.get_post(older)
    assert aged is not None and aged.embedding is None
    assert await rt.store.get_post(ancient) is None


async def test_a_new_embedder_recomputes_the_stored_embeddings(rt: Runtime) -> None:
    from tg_curator.domain import Example

    await rt.store.kv_set(KV.ML_EMBEDDER_ID, "an-older-model")
    await rt.store.upsert_chat(info(-1_001_000_007_300, "Source"))
    text = "a post embedded by the older model"
    stored = await rt.store.insert_post(
        NewPost(
            chat_id=-1_001_000_007_300,
            message_id=1,
            kind="post",
            message_ids=[1],
            posted_at=rt.clock.now(),
            via="live",
            text=text,
            text_hash=text_hash(text),
            urls=[],
            embedding=b"\x00" * 8,
        ),
        status=PostStatus.unsorted,
    )
    assert stored is not None
    example = await rt.store.add_example(
        Example(
            id=0,
            topic_id=None,
            kind="example",
            text="an example",
            embedding=b"\x00" * 8,
            created_at=rt.clock.now(),
        )
    )
    await service.refresh_embeddings(rt)
    post = await rt.store.get_post(stored.id)
    assert post is not None and len(post.embedding or b"") == rt.embedder.dim * 4
    examples = await rt.store.list_examples()
    assert [len(e.embedding) for e in examples if e.id == example.id] == [rt.embedder.dim * 4]
    assert await rt.store.kv_get(KV.ML_EMBEDDER_ID) == rt.embedder.id


# --- settings at start -------------------------------------------------------------------------


async def test_missing_settings_write_the_template_and_print_the_steps(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sf = SettingsFile(home / "settings.toml", env={})
    assert await service.ready_settings(sf, wait=False) == 0
    assert sf.exists()
    out = capsys.readouterr().out
    assert "@BotFather" in out and "my.telegram.org" in out
    assert await service.ready_settings(sf, wait=False) == 2  # template blanks -> exit 2
    assert "telegram.api_id" in capsys.readouterr().out


async def test_a_first_start_with_the_telegram_values_in_the_environment_goes_on(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = {
        "TG_CURATOR_API_ID": "12345",
        "TG_CURATOR_API_HASH": "abc",
        "TG_CURATOR_BOT_TOKEN": "1:xyz",
    }
    sf = SettingsFile(home / "settings.toml", env=env)
    settings = await service.ready_settings(sf, wait=True)  # the Docker first start
    assert not isinstance(settings, int)
    assert settings.telegram.api_id == 12345 and settings.telegram.bot_token == "1:xyz"
    assert "api_id = 12345" in sf.path.read_text()  # written into the file (§3)
    assert "Next steps" not in capsys.readouterr().out


async def test_wait_mode_polls_until_the_file_is_fixed(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sf = SettingsFile(home / "settings.toml", env={})
    polls = 0

    async def poll(_: float) -> None:
        nonlocal polls
        polls += 1
        if polls == 2:
            text = sf.path.read_text()
            text = text.replace("api_id = 0", "api_id = 12345", 1)
            text = text.replace('api_hash = ""', 'api_hash = "abc"', 1)
            text = text.replace('bot_token = ""', 'bot_token = "1:xyz"', 1)
            sf.path.write_text(text)

    settings = await service.ready_settings(sf, wait=True, sleep=poll)
    assert not isinstance(settings, int)
    assert settings.telegram.api_id == 12345
    out = capsys.readouterr().out
    assert out.count("Waiting for") == 1  # printed once, not once per poll
