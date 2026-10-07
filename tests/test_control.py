"""control.py: the service lock, the control socket and the shared command bodies (§13)."""

from __future__ import annotations

import asyncio
import os
import socket
import stat
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import FakeUserGateway
from tg_curator import control
from tg_curator.clock import local_date
from tg_curator.domain import TopicSyncResult
from tg_curator.errors import CuratorError
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

SOURCE = -1_001_000_008_001


def info(chat_id: int, title: str, username: str | None = None) -> ChatInfo:
    return ChatInfo(
        id=chat_id,
        kind="channel",
        title=title,
        username=username,
        noforwards=False,
        is_creator=False,
        is_admin=False,
        archived=False,
        muted_until=None,
    )


@pytest.fixture
def sock(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A relative socket path inside ``home``: pytest's temp paths on macOS are longer than
    the 104 bytes an AF_UNIX address may have."""
    monkeypatch.chdir(home)
    return Path(control.SOCKET_NAME)


@pytest.fixture
async def server(rt: Runtime, sock: Path) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(control.serve(rt, sock))
    for _ in range(200):
        if sock.exists():
            break
        await asyncio.sleep(0.005)
    yield task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def echo_command(rt: Runtime, args: Mapping[str, Any], out: control.Out) -> int:
    for word in args["words"]:
        await out(word)
    return 3


async def failing_command(rt: Runtime, args: Mapping[str, Any], out: control.Out) -> int:
    await out("starting")
    raise CuratorError("the account is not bound: /bind")


# --- the lock ----------------------------------------------------------------------------------


def test_the_lock_is_exclusive_and_the_probe_does_not_keep_it(home: Path) -> None:
    first, second = control.ServiceLock(home), control.ServiceLock(home)
    assert not control.service_running(home)
    assert first.acquire()
    try:
        assert control.service_running(home)
        assert not second.acquire()
    finally:
        first.release()
    assert not control.service_running(home)
    assert second.acquire()
    second.release()
    assert stat.S_IMODE(os.stat(control.lock_path(home)).st_mode) == 0o600


# --- the socket --------------------------------------------------------------------------------


async def test_round_trip_streams_lines_and_the_exit_code(
    rt: Runtime, sock: Path, server: asyncio.Task[None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(control.COMMANDS, "echo", echo_command)
    monkeypatch.setitem(control.COMMANDS, "fail", failing_command)
    assert stat.S_IMODE(os.stat(sock).st_mode) == 0o600
    assert "control" in rt.health

    lines: list[str] = []
    assert await control.request(sock, "echo", {"words": ["one", "two"]}, lines.append) == 3
    assert lines == ["one", "two"]

    lines.clear()
    assert await control.request(sock, "fail", {}, lines.append) == control.EXIT_FAILED
    assert lines == ["starting", "the account is not bound: /bind"]

    lines.clear()
    assert await control.request(sock, "nope", {}, lines.append) == control.EXIT_USAGE
    assert "unknown command" in lines[0]


async def test_a_bad_request_gets_a_usage_exit(sock: Path, server: asyncio.Task[None]) -> None:
    reader, writer = await asyncio.open_unix_connection(os.fspath(sock))
    writer.write(b"this is not json\n")
    await writer.drain()
    replies = [line async for line in reader]
    writer.close()
    assert b'"exit": 2' in replies[-1]


async def test_the_socket_file_goes_away_with_the_server(
    sock: Path, server: asyncio.Task[None]
) -> None:
    server.cancel()
    await asyncio.gather(server, return_exceptions=True)
    assert not sock.exists()
    with pytest.raises(control.ServiceUnreachableError):
        await control.request(sock, "stats", {}, print)


async def test_a_stale_socket_is_replaced(
    rt: Runtime, sock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(control.COMMANDS, "echo", echo_command)
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(os.fspath(sock))
    stale.close()  # the file stays, nothing listens: a crashed service's leftover
    task = asyncio.create_task(control.serve(rt, sock))
    try:
        for _ in range(200):
            if "control" in rt.health:
                break
            await asyncio.sleep(0.005)
        lines: list[str] = []
        assert await control.request(sock, "echo", {"words": ["alive"]}, lines.append) == 3
        assert lines == ["alive"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_nothing_listening_is_reported_as_unreachable(sock: Path) -> None:
    with pytest.raises(control.ServiceUnreachableError):
        await control.request(sock, "stats", {}, print)


# --- the command bodies ------------------------------------------------------------------------


async def collect(rt: Runtime, name: str, args: Mapping[str, Any]) -> tuple[int, list[str]]:
    lines: list[str] = []

    async def out(line: str) -> None:
        lines.append(line)

    return await control.run_command(rt, name, args, out), lines


async def test_stats_lists_every_source_chat(rt: Runtime) -> None:
    from tg_curator.subscriptions.stats import StatsService

    rt.stats = StatsService(rt)
    await rt.store.upsert_chat(info(SOURCE, "Loud channel"))
    day = local_date(rt.clock.now(), rt.settings.general.timezone)
    await rt.store.bump_chat_daily(SOURCE, day, 40)
    code, lines = await collect(rt, "stats", {"days": 7})
    assert code == 0
    assert any("Loud channel" in line and " 40 " in line for line in lines)
    assert lines[-1] == "stats: 1 chats over the last 7 days"


async def test_chats_syncs_the_dialog_list_first(rt: Runtime, user_gw: FakeUserGateway) -> None:
    from tg_curator.pipeline.intake import IntakeService

    rt.intake = IntakeService(rt)
    user_gw.add_chat(info(SOURCE, "Kun.uz", username="kunuz"))
    code, lines = await collect(rt, "chats", {})
    assert code == 0
    assert any(str(SOURCE) in line and "@kunuz" in line for line in lines)
    assert lines[-1] == "found 1 chats, 0 are output channels"


async def test_digest_preview_without_channels_says_so(rt: Runtime) -> None:
    from tg_curator.pipeline.digest import DigestService

    rt.digest = DigestService(rt)
    code, lines = await collect(rt, "digest-preview", {})
    assert code == 0
    assert lines == ["no topic has a channel yet: there is no digest to show"]


async def test_a_command_without_its_service_fails_with_a_sentence(rt: Runtime) -> None:
    code, lines = await collect(rt, "review", {})
    assert code == control.EXIT_FAILED
    assert lines == ["review is not wired in this process"]


def test_topics_summary_confirms_or_lists_the_problems() -> None:
    assert control.topics_summary(TopicSyncResult(4, 0, [])) == [
        "4 topics, all channels resolved, 4 without a channel yet (tracked only)"
    ]
    assert control.topics_summary(TopicSyncResult(3, 3, [])) == ["3 topics, all channels resolved"]
    lines = control.topics_summary(TopicSyncResult(2, 1, ["topic x: channel @x cannot be found"]))
    assert lines == ["2 topics, 1 with a problem:", "  topic x: channel @x cannot be found"]


def test_plain_turns_bot_html_into_terminal_text() -> None:
    assert control.plain("<b>Daily digest</b> · a &amp; b") == "Daily digest · a & b"


def test_every_socket_command_says_what_it_needs() -> None:
    assert set(control.NEEDS) == set(control.COMMANDS)
    assert all(need <= {"models", "user", "bot"} for need in control.NEEDS.values())
