"""RD-5: the supervisor's per-loop record (``rt.loops``) and the verdicts ``/status`` gives.

``rt.health`` only records successes, so a loop failing on every tick since start-up and a
loop held back by setup mode were invisible to ``/status``; these tests drive the real
``Supervisor`` and ``Service`` and read the reply the owner gets.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeUserGateway,
    make_bot_message,
)
from tests.fakes import plain_text as plain
from tg_curator.account import AccountService
from tg_curator.bot import control
from tg_curator.bot.core import BotApp
from tg_curator.config import SettingsFile
from tg_curator.db.store import Store
from tg_curator.errors import CuratorError, FloodWait, TelegramUnavailable
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.service import (
    ACCOUNT_LOOPS,
    ERROR_TEXT_LIMIT,
    Doubles,
    Service,
    Supervisor,
    build_runtime,
    short_error,
)


async def parked(_: float) -> None:
    """A sleep that never ends: every loop runs its body exactly once."""
    await asyncio.Event().wait()


async def settle(condition: Callable[[], bool] = lambda: False, rounds: int = 50) -> None:
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.005)


@pytest.fixture
async def app(rt: Runtime) -> BotApp:
    rt.account = AccountService(rt)
    rt.publisher = Publisher(rt)
    rt.digest = DigestService(rt)
    app = BotApp(rt)
    control.register(app)
    await app.start()
    return app


async def status(bot_gw: FakeBotGateway) -> list[str]:
    await bot_gw.say(make_bot_message("/status"))
    return plain(bot_gw.sent(OWNER_ID)[-1].html).splitlines()


# --- the supervisor's record -------------------------------------------------------------------


async def test_a_loop_failing_since_start_up_is_flagged_with_its_error(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    async def good() -> None:
        return None

    async def bad() -> None:
        raise RuntimeError("database is locked")

    supervisor = Supervisor(rt, sleep=parked)
    supervisor.start("good", good, 30.0)
    supervisor.start("bad", bad, 30.0)
    await settle(lambda: rt.loops["bad"].last_error is not None and "good" in rt.health)
    try:
        assert "bad" not in rt.health  # what /status could not see before
        state = rt.loops["bad"]
        assert state.running and not state.paused and state.last_ok_at is None
        assert state.failing_since == rt.clock.now() == state.last_error_at
        assert state.last_error == "RuntimeError: database is locked"
        lines = await status(bot_gw)
        assert lines[1] == "⚠ 1 problem: bad — see Loops below"
        assert "• bad: failing since 12:00: RuntimeError: database is locked" in lines
        assert "• good: ok (last tick 0 s ago)" in lines
        assert lines.index("• bad: failing since 12:00: RuntimeError: database is locked") < (
            lines.index("• good: ok (last tick 0 s ago)")
        )
    finally:
        await supervisor.stop()


async def test_the_failing_run_starts_at_the_first_failure_and_ends_at_a_success(
    rt: Runtime, clock: FakeClock
) -> None:
    calls = {"n": 0}
    release = asyncio.Event()

    async def flaky() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TelegramUnavailable("connection reset")
        if calls["n"] == 2:
            raise FloodWait(1)
        await release.wait()  # the third tick succeeds once the test has looked

    async def tick(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    first = clock.now()
    supervisor = Supervisor(rt, sleep=tick)
    supervisor.start("flaky", flaky, 60.0)
    await settle(lambda: calls["n"] >= 3)
    try:
        state = rt.loops["flaky"]
        assert state.failing_since == first  # the run began with the first failure
        assert state.last_error_at is not None and state.last_error_at > first
        assert state.last_error == "Telegram asks to wait 1 s"
        release.set()
        await settle(lambda: calls["n"] >= 3 and state.last_ok_at is not None)
        assert state.failing_since is None
        assert state.last_error is not None and state.last_error_at is not None
        assert rt.health["flaky"] == state.last_ok_at  # rt.health keeps working
    finally:
        await supervisor.stop()


async def test_a_stopped_loop_reads_not_started_and_a_held_one_paused(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    async def body() -> None:
        return None

    supervisor = Supervisor(rt, sleep=parked)
    supervisor.start("intake", body, 5.0)
    await settle(lambda: "intake" in rt.health)
    await supervisor.stop(["intake"])
    assert rt.loops["intake"].running is False
    lines = await status(bot_gw)
    assert "• intake: not started" in lines
    assert lines[1] == "⚠ 1 problem: intake — see Loops below"

    supervisor.hold({"intake": 5.0, "watchdog": 3600.0}, paused=True)
    lines = await status(bot_gw)
    assert "• intake: paused (account not bound — /bind)" in lines
    assert "• watchdog: paused (account not bound — /bind)" in lines
    assert lines[1] == "⏸ Running in setup mode: what needs the account waits for /bind"

    supervisor.hold({"intake": 5.0}, paused=False)  # bound: due to start
    assert "• intake: not started" in await status(bot_gw)


async def test_hold_leaves_a_running_loop_alone(rt: Runtime) -> None:
    async def body() -> None:
        return None

    supervisor = Supervisor(rt, sleep=parked)
    supervisor.start("intake", body, 5.0)
    try:
        supervisor.hold({"intake": 5.0}, paused=True)
        assert rt.loops["intake"].paused is False and rt.loops["intake"].running
    finally:
        await supervisor.stop()


async def test_a_loop_whose_first_tick_never_ends_turns_stalled(
    app: BotApp, rt: Runtime, clock: FakeClock, bot_gw: FakeBotGateway
) -> None:
    async def hangs() -> None:
        await asyncio.Event().wait()

    supervisor = Supervisor(rt, sleep=parked)
    supervisor.start("sorter", hangs, 30.0)
    supervisor.start("chats", hangs, 1800.0, first_delay=1800.0)
    try:
        lines = await status(bot_gw)
        assert "• sorter: ok (waiting for its first tick)" in lines
        assert lines[1] == "✓ Everything is running"
        clock.advance(timedelta(minutes=3))
        lines = await status(bot_gw)
        assert "• sorter: stalled (no tick since it started 3 min ago)" in lines
        assert "• chats: ok (waiting for its first tick)" in lines  # first tick due at +30 min
        assert lines[1] == "⚠ 1 problem: sorter — see Loops below"
    finally:
        await supervisor.stop()


async def test_a_serving_loop_is_ok_while_it_listens_and_after_it_recovers(
    app: BotApp, rt: Runtime, clock: FakeClock, bot_gw: FakeBotGateway
) -> None:
    attempts = {"n": 0}

    async def serve() -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise CuratorError("cannot open the control socket")
        rt.health["control"] = rt.clock.now()  # what control.serve writes once listening
        await asyncio.Event().wait()

    async def tick(seconds: float) -> None:
        clock.advance(seconds)
        await asyncio.sleep(0)

    supervisor = Supervisor(rt, sleep=tick)
    supervisor.start("control", serve, 5.0, serves=True)
    try:
        await settle(lambda: "control" in rt.health)
        state = rt.loops["control"]
        assert state.interval is None and state.failing_since is not None
        clock.advance(timedelta(hours=1))  # no ticks, yet never stalled
        lines = await status(bot_gw)
        assert "• control: ok (running since 12:00)" in lines
        assert lines[1] == "✓ Everything is running"
    finally:
        await supervisor.stop()


async def test_the_error_text_is_escaped_for_the_html_reply(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    async def bad() -> None:
        raise ValueError("<b>bold</b> & co")

    supervisor = Supervisor(rt, sleep=parked)
    supervisor.start("bad", bad, 30.0)
    await settle(lambda: rt.loops["bad"].last_error is not None)
    try:
        await bot_gw.say(make_bot_message("/status"))
        html = bot_gw.sent(OWNER_ID)[-1].html
        assert "ValueError: &lt;b&gt;bold&lt;/b&gt; &amp; co" in html
    finally:
        await supervisor.stop()


def test_short_error_is_one_short_line() -> None:
    assert short_error(CuratorError("the account is not signed in")) == (
        "the account is not signed in"
    )
    assert short_error(OSError("disk\nfull")) == "OSError: disk full"
    assert short_error(TimeoutError()) == "TimeoutError"
    long = short_error(RuntimeError("x" * 500))
    assert len(long) == ERROR_TEXT_LIMIT and long.endswith("…")


# --- the service -------------------------------------------------------------------------------


@pytest.fixture
def wired(
    home: Path,
    settings_file: SettingsFile,
    store: Store,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
) -> tuple[Runtime, BotApp]:
    doubles = Doubles(
        user=user_gw,
        bot=bot_gw,
        clock=clock,
        embedder=FakeEmbedder(),
        classifier=FakeClassifier(),
        llm=FakeLLM(),
    )
    return build_runtime(home, settings_file=settings_file, store=store, fakes=doubles)


async def test_setup_mode_records_the_account_loops_as_paused_until_bound(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway
) -> None:
    rt, app = wired
    user_gw.authorised = False
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        assert all(rt.loops[name].paused for name in ACCOUNT_LOOPS)
        assert not any(rt.loops[name].running for name in ACCOUNT_LOOPS)
        assert rt.loops["sorter"].running and not rt.loops["sorter"].paused

        user_gw.authorised = True
        await rt.events.emit("account_bound")
        await settle(lambda: svc.supervisor.running("intake"))
        assert rt.loops["intake"].running and not rt.loops["intake"].paused

        await user_gw.lose_session("revoked")
        await settle(lambda: rt.loops["intake"].paused)
        assert rt.loops["intake"].paused and not rt.loops["intake"].running
        assert rt.loops["watchdog"].paused
    finally:
        await svc.shutdown()


async def test_a_bound_account_whose_bring_up_fails_is_not_shown_as_paused(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway
) -> None:
    rt, app = wired
    user_gw.authorised = False
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        user_gw.authorised = True
        user_gw.fail_next("me", RuntimeError("bring-up broke"))
        await rt.events.emit("account_bound")
        await settle(rounds=10)
        assert not svc.supervisor.running("intake")
        assert rt.loops["intake"].paused is False  # "not started": a problem, not setup mode
    finally:
        await svc.shutdown()
