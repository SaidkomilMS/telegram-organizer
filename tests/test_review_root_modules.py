"""Regression tests for the review findings in the root modules (service, config, textutil,
cli/control)."""

# The fixtures imported from the other test modules are parameters here, which ruff reads as
# redefinitions.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import pytest
import tomlkit

from tests.fakes import OWNER_ID, FakeBotGateway, FakeUserGateway
from tests.test_bot_control import app, hand_edit, say  # noqa: F401 - fixture re-used
from tests.test_service import (  # noqa: F401 - fixtures re-used
    doubles,
    parked,
    seed_owned_channels,
    settle,
    wired,
)
from tg_curator import control
from tg_curator.bot.core import BotApp
from tg_curator.config import SettingsFile, validate_settings
from tg_curator.errors import ConfigError, FloodWait, TelegramUnavailable
from tg_curator.runtime import Runtime
from tg_curator.service import Service, Supervisor
from tg_curator.textutil import _TAG_RE, canonical_urls, split_html, url_key

# --- service: the bring-up after /bind ---------------------------------------------------------


async def fast(_: float) -> None:
    await asyncio.sleep(0)


async def bound_service(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway
) -> tuple[Runtime, Service]:
    rt, app = wired
    await seed_owned_channels(rt)
    user_gw.authorised = False
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    assert svc.setup_mode
    return rt, svc


@pytest.mark.parametrize("error", [TelegramUnavailable("blip"), FloodWait(300)])
async def test_a_failed_bring_up_after_bind_is_retried_until_intake_runs(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, error: Exception
) -> None:
    rt, svc = await bound_service(wired, user_gw)
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        await asyncio.sleep(0)

    svc._sleep = sleep  # the retry's sleep; the loops stay parked
    try:
        user_gw.authorised = True
        user_gw.fail_next("list_chats", error)
        await rt.events.emit("account_bound")
        await settle(lambda: svc.supervisor.running("watchdog"))
        assert svc.supervisor.running("intake") and svc.supervisor.running("watchdog")
        assert user_gw._message_handlers
        assert len(waits) == 1
        if isinstance(error, FloodWait):
            assert waits[0] >= 300  # Telegram's wait is honoured
    finally:
        await svc.shutdown()


async def test_a_session_lost_during_the_retries_stops_them(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway
) -> None:
    rt, svc = await bound_service(wired, user_gw)
    retried = asyncio.Event()

    async def sleep(_: float) -> None:
        retried.set()
        await asyncio.Event().wait()

    svc._sleep = sleep
    try:
        user_gw.authorised = True
        user_gw.fail_next("list_chats", TelegramUnavailable("blip"))
        await rt.events.emit("account_bound")
        await asyncio.wait_for(retried.wait(), 2)
        await user_gw.lose_session("revoked")
        await settle(rounds=10)
        assert svc.setup_mode
        assert not svc.supervisor.running("intake") and not svc.supervisor.running("watchdog")
        assert svc._bind_task is None
    finally:
        await svc.shutdown()


async def test_the_reconcile_after_bind_waits_for_a_running_publisher_tick(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, app = wired
    await seed_owned_channels(rt)
    user_gw.authorised = False
    log: list[str] = []
    release = asyncio.Event()
    assert rt.publisher is not None

    async def tick() -> None:
        log.append("tick")
        await release.wait()  # a send in flight
        log.append("tick done")

    async def reconcile() -> None:
        log.append("reconcile")

    monkeypatch.setattr(rt.publisher, "tick", tick)
    monkeypatch.setattr(rt.publisher, "reconcile", reconcile)
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        await settle(lambda: "tick" in log)
        log.clear()
        log.append("tick")  # still in flight from the start
        user_gw.authorised = True
        await rt.events.emit("account_bound")
        await settle(rounds=20)
        assert "reconcile" not in log  # never alongside the tick
        release.set()
        await settle(lambda: svc.supervisor.running("intake"))
        assert log == ["tick", "tick done", "reconcile"]
    finally:
        await svc.shutdown()


# --- service: the supervisor -------------------------------------------------------------------


async def test_a_flood_wait_in_a_loop_is_one_warning_and_waits_it_out(
    rt: Runtime, caplog: pytest.LogCaptureFixture
) -> None:
    waits: list[float] = []
    calls = {"n": 0}

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) >= 2:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    async def body() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloodWait(600)

    caplog.set_level(logging.INFO, logger="tg_curator.service")
    supervisor = Supervisor(rt, sleep=sleep)
    supervisor.start("folders", body, 30.0)
    await settle(lambda: len(waits) >= 2)
    try:
        assert waits[0] >= 600
        floods = [r for r in caplog.records if "asks to wait 600" in r.getMessage()]
        assert len(floods) == 1
        assert floods[0].levelno == logging.WARNING and floods[0].exc_info is None
        assert not any("failed" in r.getMessage() for r in caplog.records)
        assert calls["n"] == 2 and "folders" in rt.health  # the second try succeeded
    finally:
        await supervisor.stop()


async def test_paused_waits_for_the_running_tick_and_holds_the_next(rt: Runtime) -> None:
    release = asyncio.Event()
    log: list[str] = []

    async def body() -> None:
        log.append("start")
        await release.wait()
        log.append("end")

    supervisor = Supervisor(rt, sleep=fast)
    supervisor.start("publisher", body, 2.0)
    await settle(lambda: "start" in log)

    async def job() -> None:
        async with supervisor.paused(["publisher"]):
            log.append("job")
            await asyncio.sleep(0)
            log.append("job done")

    task = asyncio.create_task(job())
    await settle(rounds=5)
    assert "job" not in log
    release.set()
    await asyncio.wait_for(task, 2)
    await settle(lambda: log.count("start") >= 2)
    await supervisor.stop()
    assert log[:5] == ["start", "end", "job", "job done", "start"]


# --- config: writes keep hand edits; /reload keeps [telegram]/[storage] ------------------------


def _edit_file(sf: SettingsFile, edit: Any) -> None:
    doc = tomlkit.parse(sf.path.read_text(encoding="utf-8"))
    edit(doc)
    sf.path.write_text(tomlkit.dumps(doc), encoding="utf-8")


async def test_a_bot_write_keeps_a_pending_hand_edit(settings_file: SettingsFile) -> None:
    assert settings_file.current.digest.hour == 21

    def edit(doc: Any) -> None:
        doc["digest"]["hour"] = 7

    _edit_file(settings_file, edit)
    await settings_file.set_value("publishing.live", True)
    on_disk = tomlkit.parse(settings_file.path.read_text(encoding="utf-8"))
    assert on_disk["digest"]["hour"] == 7
    assert on_disk["publishing"]["live"] is True
    assert "# local hour" in settings_file.path.read_text(encoding="utf-8")  # comments kept
    # The running service changes by the bot's write only; the hand edit waits for /reload.
    assert settings_file.current.publishing.live is True
    assert settings_file.current.digest.hour == 21
    assert (await settings_file.load()).digest.hour == 7


async def test_a_bot_write_over_an_invalid_hand_edit_writes_nothing(
    settings_file: SettingsFile,
) -> None:
    def edit(doc: Any) -> None:
        doc["digest"]["hour"] = 99

    _edit_file(settings_file, edit)
    before = settings_file.path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match=r"edited by hand.*digest\.hour"):
        await settings_file.set_value("publishing.live", True)
    assert settings_file.path.read_text(encoding="utf-8") == before
    assert settings_file.current.publishing.live is False


async def test_a_write_after_reload_keeps_the_running_telegram_and_storage(
    settings_file: SettingsFile,
) -> None:
    await settings_file.set_value("telegram.owner_id", OWNER_ID)

    def edit(doc: Any) -> None:
        doc["telegram"]["owner_id"] = 999
        doc["storage"]["database_url"] = "postgresql+asyncpg://elsewhere/db"

    _edit_file(settings_file, edit)
    fresh = await settings_file.load()
    assert fresh.telegram.owner_id == 999  # the caller can tell a restart is needed
    assert settings_file.current.telegram.owner_id == OWNER_ID
    await settings_file.set_value("digest.hour", 20)
    current = settings_file.current
    assert current.digest.hour == 20
    assert current.telegram.owner_id == OWNER_ID
    assert current.storage.database_url == ""
    on_disk = tomlkit.parse(settings_file.path.read_text(encoding="utf-8"))
    assert on_disk["telegram"]["owner_id"] == 999  # the hand edit applies at the next start
    # A write to the pinned section itself (the owner claim) does take effect.
    await settings_file.set_value("telegram.owner_id", 4242)
    assert settings_file.current.telegram.owner_id == 4242
    assert settings_file.current.storage.database_url == ""


async def test_reload_then_a_settings_write_keeps_obeying_the_running_owner(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
) -> None:
    def edit(doc: Any) -> None:
        doc["telegram"]["owner_id"] = 999
        doc["storage"]["database_url"] = "postgresql+asyncpg://elsewhere/db"

    hand_edit(rt, edit)
    await say(bot_gw, "/reload")
    await rt.settings_file.set_value("digest.hour", 20)
    assert rt.settings.telegram.owner_id == OWNER_ID
    assert rt.settings.storage.database_url == ""
    assert "Status" in await say(bot_gw, "/status")
    on_disk = tomlkit.parse(rt.settings_file.path.read_text(encoding="utf-8"))
    assert on_disk["telegram"]["owner_id"] == 999


# --- config: error sentences -------------------------------------------------------------------


def test_a_bad_database_url_never_shows_its_password() -> None:
    with pytest.raises(ConfigError) as caught:
        validate_settings({"storage": {"database_url": "mysql://curator:S3cretPW@db/curator"}})
    message = str(caught.value)
    assert "S3cretPW" not in message
    assert "storage.database_url" in message and "mysql://curator:***@db/curator" in message


def test_the_plain_database_url_schemes_name_the_async_driver() -> None:
    settings = validate_settings({"storage": {"database_url": "postgresql://u:p@h/db"}})
    assert settings.storage.database_url == "postgresql+asyncpg://u:p@h/db"
    settings = validate_settings({"storage": {"database_url": "postgres://u:p@h/db"}})
    assert settings.storage.database_url == "postgresql+asyncpg://u:p@h/db"
    settings = validate_settings({"storage": {"database_url": "sqlite:////var/x.db"}})
    assert settings.storage.database_url == "sqlite+aiosqlite:////var/x.db"


@pytest.mark.parametrize(
    "data",
    [
        {"telegram": {"bot_token": 123456789}},
        {"telegram": {"api_hash": 123456789}},
        {"llm": {"api_key": 123456789}},
    ],
)
def test_a_wrong_typed_secret_is_named_but_not_echoed(data: dict[str, Any]) -> None:
    with pytest.raises(ConfigError) as caught:
        validate_settings(data)
    assert "123456789" not in str(caught.value)


def test_model_level_errors_read_without_a_stray_colon() -> None:
    topics = [{"key": "a", "name": "A"}, {"key": "a", "name": "B"}]
    with pytest.raises(ConfigError) as caught:
        validate_settings({"topics": topics})
    assert str(caught.value).startswith('settings: topics: key "a" appears twice')
    sources = [{"chat": "@kunuz"}, {"chat": "kunuz"}]
    with pytest.raises(ConfigError) as caught:
        validate_settings({"sources": sources})
    assert str(caught.value).startswith('settings: sources: chat "kunuz" is listed twice')


# --- textutil ----------------------------------------------------------------------------------


def _balanced(part: str) -> bool:
    stack: list[str] = []
    for m in _TAG_RE.finditer(part):
        closing, name = m.group(1), m.group(2).lower()
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


@pytest.mark.parametrize(
    ("markup", "limit", "expected"),
    [
        (
            "First paragraph text here.\n\n<blockquote>\nQuoted words follow and go on."
            "</blockquote>",
            40,
            [
                "First paragraph text here.",
                "<blockquote>Quoted words follow and go on.</blockquote>",
            ],
        ),
        ("aa <b> bb</b>", 4, ["aa", "<b>bb</b>"]),
        ("aaaa<b>bbbb</b>", 4, ["aaaa", "<b>bbbb</b>"]),
        (
            "x" * 10 + '<a href="u">' + "y" * 10 + "</a>",
            10,
            ["x" * 10, '<a href="u">' + "y" * 10 + "</a>"],
        ),
    ],
)
def test_an_open_tag_at_a_cut_starts_the_next_part(
    markup: str, limit: int, expected: list[str]
) -> None:
    parts = split_html(markup, limit)
    assert parts == expected
    assert all(_balanced(p) for p in parts)


def test_every_part_is_balanced_whatever_the_limit() -> None:
    markup = (
        "Intro line one.\n\n<b>\nBold body <i>with italics</i> and more</b> plain "
        '<a href="https://x.y/z">a link text</a>\n<blockquote> quoted <code>c()</code></blockquote>'
    )
    for limit in range(1, 60):
        parts = split_html(markup, limit)
        assert all(_balanced(p) for p in parts), (limit, parts)
        text = "".join(re.sub(r"<[^>]+>|\s", "", p) for p in parts)
        assert text == re.sub(r"<[^>]+>|\s", "", markup), limit


def test_a_link_with_a_bad_port_is_not_a_link() -> None:
    assert url_key("https://example.com:abc/x") is None
    assert url_key("http://example.com:99999/x") is None
    assert url_key("http://example.com:8080/x") == "example.com:8080/x"
    assert canonical_urls(["https://example.com:abc/x", "https://good.example/story"]) == [
        "good.example/story"
    ]


# --- control: topics add / remove run inside the service ---------------------------------------


async def test_topics_add_and_remove_run_as_socket_commands(
    wired: tuple[Runtime, BotApp],
) -> None:
    rt, _ = wired
    await rt.store.start()
    lines: list[str] = []

    async def out(line: str) -> None:
        lines.append(line)

    code = await control.run_command(
        rt, "topics-add", {"name": "Crypto & exchanges", "category": "crypto"}, out
    )
    assert code == control.EXIT_OK, lines
    assert lines[-1].endswith("added, channel: none yet (tracked only)")
    key = next(t.key for t in rt.settings.topics if t.name == "Crypto & exchanges")

    code = await control.run_command(rt, "topics-add", {"name": "crypto & EXCHANGES"}, out)
    assert code == control.EXIT_FAILED and "edit the existing topic" in lines[-1]

    code = await control.run_command(rt, "topics-remove", {"key": key}, out)
    assert code == control.EXIT_OK
    assert lines[-1] == f"topic {key} removed; its channel and history are kept"
    assert rt.settings.topic(key) is None
