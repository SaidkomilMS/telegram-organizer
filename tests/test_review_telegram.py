"""Regression tests for the review findings on the Telegram gateways.

- A link Telethon gave up on is connected again (both gateways), instead of every request
  failing with "Cannot send requests while disconnected" until a restart.
- ``AuthKeyNotFound`` (the transport's 404) is translated: the account's session is retired and
  the service starts unbound; the bot signs in a new session with the same token.
- ``leave`` treats a chat the account can no longer see as already left.
- Listing folders never makes a user's folder writable, and pinned chats inside a folder are
  read back and kept out of ``include_peers``.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from telethon import TelegramClient, functions, types, utils
from telethon import errors as tl_errors

from tests.fakes import START, FakeClock, FakeUserGateway
from tests.test_bot_client import OWNER
from tests.test_bot_client import StubClient as BotStub
from tests.test_user_client import (
    CHANNEL_BARE,
    CHANNEL_ID,
    GROUP_BARE,
    GROUP_ID,
    OUTPUT_BARE,
    OUTPUT_ID,
    basic_group,
    channel,
)
from tests.test_user_client import StubClient as UserStub
from tg_curator.domain import KV, PROPOSAL_APPROVED, PROPOSAL_DONE
from tg_curator.errors import NotOwnedError, TelegramUnavailable
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.actions import ActionExecutor
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram import bot_client, user_client
from tg_curator.telegram.bot_client import TelethonBotGateway
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.telegram.user_client import TelethonUserGateway

GIVEN_UP = "Cannot send requests while disconnected"


class LinkStub(UserStub):
    """The user stub with a counted ``connect`` that can be told to fail."""

    def __init__(self) -> None:
        super().__init__()
        self.connects = 0
        self.connect_errors: list[BaseException] = []

    async def connect(self) -> None:
        self.connects += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        self.connected = True


@pytest.fixture
def links() -> list[LinkStub]:
    """Every client the user gateway built, oldest first."""
    return []


@pytest.fixture
def ugw(home: Path, clock: FakeClock, links: list[LinkStub]) -> TelethonUserGateway:
    def factory() -> LinkStub:
        stub = LinkStub()
        stub.authorised = not links  # a session made after a loss is empty
        stub.entities[CHANNEL_ID] = channel()
        stub.entities[GROUP_ID] = basic_group()
        stub.entities[OUTPUT_ID] = channel(OUTPUT_BARE, title="Output", username=None)
        stub.responses[functions.users.GetUsersRequest] = [stub.me]
        stub.responses[functions.messages.UpdateDialogFilterRequest] = True
        links.append(stub)
        return stub

    return TelethonUserGateway(home, 1, "hash", clock=clock, client_factory=factory)


def _drop(stub: Any) -> None:
    """What Telethon leaves behind after its reconnect attempts failed: ``_disconnect(error)``
    turned ``is_connected()`` False for good."""
    stub.connected = False


# --- the account's link -----------------------------------------------------------------------


async def test_a_request_reconnects_the_account_after_telethon_gave_up(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    await ugw.connect()
    stub = links[0]
    assert stub.connects == 1
    _drop(stub)
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters([])
    assert await ugw.list_folders() == []
    assert stub.connects == 2 and stub.is_connected()


async def test_ping_and_ensure_connected_reconnect_and_report_a_lasting_outage(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    await ugw.connect()
    stub = links[0]
    _drop(stub)
    outage = ConnectionError("Connection to Telegram failed 5 time(s)")
    stub.connect_errors = [outage, outage]
    assert await ugw.ping() is False  # still unreachable: a "no", tried again next time
    with pytest.raises(TelegramUnavailable):
        await ugw.ensure_connected()
    assert await ugw.ping() is True  # the network is back: the same client is connected again
    assert stub.connects == 4 and stub.is_connected()
    await ugw.ensure_connected()  # connected: nothing to do
    assert stub.connects == 4


async def test_no_reconnect_before_connect_or_after_disconnect(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    await ugw.ensure_connected()
    assert await ugw.ping() is False
    assert links[0].connects == 0
    await ugw.connect()
    await ugw.disconnect()
    await ugw.ensure_connected()
    assert await ugw.ping() is False
    assert links[0].connects == 1


def test_real_clients_retry_a_finite_number_of_times_then_reconnect_on_use(home: Path) -> None:
    """Endless retries would keep every request waiting through an outage; finite ones plus
    ``ensure_connected`` end each tick with ``TelegramUnavailable`` and recover afterwards."""
    clients: list[TelegramClient] = [
        TelethonUserGateway(home, 1, "hash", clock=FakeClock(START))._client,
        TelethonBotGateway(
            home, api_id=1, api_hash="h", bot_token="1:t", owner_id=OWNER
        )._build_client(),
    ]
    for client in clients:
        assert client._connection_retries == 5 == user_client.CONNECTION_RETRIES
        assert client._retry_delay == 5 == bot_client.RETRY_DELAY
        assert client._auto_reconnect is True


# --- AuthKeyNotFound (transport 404) on the account ----------------------------------------


async def test_a_forgotten_key_at_connect_starts_unbound_instead_of_crashing(
    ugw: TelethonUserGateway, links: list[LinkStub], home: Path, clock: FakeClock
) -> None:
    (home / "user.session").write_bytes(b"old key")
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    ugw.on_session_lost(lost)
    links[0].connect_errors = [tl_errors.AuthKeyNotFound()]
    assert await ugw.connect() is False  # the service enters setup mode
    assert reasons == ["unregistered"]
    stamp = clock.now().strftime("%Y%m%dT%H%M%SZ")
    assert (home / f"user.session.revoked-{stamp}").read_bytes() == b"old key"
    assert len(links) == 2 and links[1].is_connected()  # the fresh session is open for /bind


async def test_a_forgotten_key_while_running_is_a_session_loss(
    ugw: TelethonUserGateway, links: list[LinkStub], home: Path
) -> None:
    (home / "user.session").write_bytes(b"old key")
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    ugw.on_session_lost(lost)
    await ugw.connect()
    _drop(links[0])  # Telethon got a 404 and disconnected
    links[0].connect_errors = [tl_errors.AuthKeyNotFound()]
    assert await ugw.ping() is False
    assert reasons == ["unregistered"]
    assert not (home / "user.session").exists()
    await ugw.ensure_connected()  # a lost session is never reconnected behind /bind's back
    assert links[1].connects == 0


async def test_a_forgotten_key_from_the_update_loop_is_a_session_loss(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    ugw.on_session_lost(lost)
    await ugw.connect()
    links[0]._updates_error = tl_errors.AuthKeyNotFound()
    assert await ugw.ping() is False
    assert reasons == ["unregistered"]


# --- the bot's link ---------------------------------------------------------------------------


@pytest.fixture
def bot_stubs() -> list[BotStub]:
    return [BotStub()]


@pytest.fixture
def bot(home: Path, bot_stubs: list[BotStub]) -> TelethonBotGateway:
    def factory() -> BotStub:
        bot_stubs.append(BotStub())
        return bot_stubs[-1]

    return TelethonBotGateway(
        home,
        api_id=1,
        api_hash="h",
        bot_token="1:token",
        owner_id=OWNER,
        client=bot_stubs[0],
        client_factory=factory,
    )


async def test_the_bot_reconnects_after_telethon_gave_up(
    bot: TelethonBotGateway, bot_stubs: list[BotStub]
) -> None:
    await bot.start()
    stub = bot_stubs[0]
    _drop(stub)
    assert await bot.send_text(OWNER, "hi") > 0
    assert [c.name for c in stub.calls] == [
        "connect", "is_user_authorized", "get_me", "connect", "send_message",
    ]  # fmt: skip


async def test_the_bot_reports_a_lasting_outage_and_recovers(
    bot: TelethonBotGateway, bot_stubs: list[BotStub]
) -> None:
    await bot.start()
    stub = bot_stubs[0]
    _drop(stub)
    stub.fail_next("connect", ConnectionError("Connection to Telegram failed 5 time(s)"))
    with pytest.raises(TelegramUnavailable):
        await bot.send_text(OWNER, "hi")
    assert not stub.calls_of("send_message")
    await bot.ensure_connected()  # what the service's timer does: the network is back
    assert stub.is_connected()
    await bot.send_text(OWNER, "hi")
    assert len(stub.calls_of("send_message")) == 1


async def test_the_bot_does_not_reconnect_before_start_or_after_stop(
    bot: TelethonBotGateway, bot_stubs: list[BotStub]
) -> None:
    await bot.ensure_connected()
    assert bot_stubs[0].calls == []
    await bot.start()
    await bot.stop()
    await bot.ensure_connected()
    assert len(bot_stubs[0].calls_of("connect")) == 1


async def test_a_forgotten_bot_key_at_start_signs_in_a_new_session(
    bot: TelethonBotGateway, bot_stubs: list[BotStub], home: Path
) -> None:
    (home / "bot.session").write_bytes(b"old key")
    bot_stubs[0].fail_next("connect", tl_errors.AuthKeyNotFound())
    account = await bot.start()
    assert account.username == "curator_test_bot"
    old, new = bot_stubs
    assert [c.name for c in old.calls] == ["connect", "disconnect"]
    assert not (home / "bot.session").exists()
    assert [c.name for c in new.calls] == ["connect", "sign_in", "is_user_authorized", "get_me"]
    assert new.calls_of("sign_in")[0].kwargs == {"bot_token": "1:token"}
    assert len(new.handlers) == 2 and old.handlers == []


async def test_a_forgotten_bot_key_while_running_signs_in_a_new_session(
    bot: TelethonBotGateway, bot_stubs: list[BotStub]
) -> None:
    await bot.start()
    _drop(bot_stubs[0])
    bot_stubs[0].fail_next("connect", tl_errors.AuthKeyNotFound())
    await bot.send_text(OWNER, "hi")
    new = bot_stubs[1]
    assert [c.name for c in new.calls] == ["connect", "sign_in", "send_message"]
    assert len(new.handlers) == 2  # the owner's commands reach the bot on the new session


# --- leave: an inaccessible chat is already left --------------------------------------------


async def test_leave_treats_a_forbidden_stub_as_already_left(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    stub = links[0]
    stub.entities[CHANNEL_ID] = types.ChannelForbidden(id=CHANNEL_BARE, access_hash=1, title="X")
    stub.entities[GROUP_ID] = types.ChatForbidden(id=GROUP_BARE, title="Y")
    await ugw.leave(CHANNEL_ID)
    await ugw.leave(GROUP_ID)
    assert stub.requests_of(functions.channels.LeaveChannelRequest) == []
    assert stub.requests_of(functions.messages.DeleteChatUserRequest) == []


async def test_leave_treats_a_private_channel_as_already_left(
    ugw: TelethonUserGateway, links: list[LinkStub], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def private(ref: Any) -> Any:
        raise tl_errors.ChannelPrivateError(request=None)

    monkeypatch.setattr(links[0], "get_entity", private)
    await ugw.leave(CHANNEL_ID)
    assert links[0].requests_of(functions.channels.LeaveChannelRequest) == []


async def test_an_approved_leave_of_a_chat_already_left_marks_it_inactive(
    rt: Runtime,
    ugw: TelethonUserGateway,
    links: list[LinkStub],
    make_chat: Callable[..., ChatInfo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.stats = StatsService(rt)
    rt.folders = FolderManager(rt)
    rt.review = ReviewService(rt)
    executor = ActionExecutor(rt)
    executor._rng = random.Random(7)  # type: ignore[attr-defined]
    info = make_chat(kind="channel")
    fake = rt.user
    assert isinstance(fake, FakeUserGateway)
    fake.add_chat(info)
    await rt.store.upsert_chat(info, role="source")
    p = await rt.store.create_proposal(
        "leave", reason="leave", review_day=START.date(), chat_id=info.id
    )
    await rt.review.send()
    await rt.review.decide(p.id, "approve")
    await rt.review.decide(p.id, "confirm")
    assert (await rt.store.get_proposal(p.id)).state == PROPOSAL_APPROVED  # type: ignore[union-attr]
    # the owner was removed from the channel by hand in the meantime
    bare, _ = utils.resolve_id(info.id)
    links[0].entities[info.id] = types.ChannelForbidden(id=bare, access_hash=1, title="X")
    rt.user = ugw
    await executor.tick()
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and not chat.active
    assert (await rt.store.get_proposal(p.id)).state == PROPOSAL_DONE  # type: ignore[union-attr]


# --- folders ----------------------------------------------------------------------------------


def _folder(folder_id: int, title: str, *, pinned: list[Any], include: list[Any]) -> Any:
    return types.DialogFilter(
        folder_id, types.TextWithEntities(title, []), pinned, include, [], emoticon="📁"
    )


async def _output_chat(rt: Runtime) -> None:
    info = ChatInfo(
        id=OUTPUT_ID, kind="channel", title="Output", username=None, noforwards=False,
        is_creator=True, is_admin=True, archived=False, muted_until=None,
    )  # fmt: skip
    await rt.store.upsert_chat(info, role="output")


async def test_sync_never_writes_a_user_folder_that_reused_the_stored_id(
    rt: Runtime, ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    """The curator's "Curated" (kv id 2) was deleted by hand and the user's new "Work" folder
    got id 2. Listing it must not make it writable, whatever folders.py then decides."""
    stub = links[0]
    rt.user = ugw
    await _output_chat(rt)
    await rt.store.kv_set(KV.FOLDERS_CURATED_ID, 2)
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_folder(2, "Work", pinned=[], include=[types.InputPeerChannel(CHANNEL_BARE, 1)])]
    )
    with contextlib.suppress(NotOwnedError):
        await FolderManager(rt).sync()
    writes = stub.requests_of(functions.messages.UpdateDialogFilterRequest)
    assert all(w.id != 2 for w in writes)


async def test_get_folder_reports_pinned_chats_first(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    pinned = [types.InputPeerChannel(OUTPUT_BARE, 1)]
    include = [types.InputPeerChannel(CHANNEL_BARE, 1), types.InputPeerChat(GROUP_BARE)]
    links[0].responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_folder(2, "Curated", pinned=pinned, include=include)]
    )
    assert await ugw.get_folder(2) == [OUTPUT_ID, CHANNEL_ID, GROUP_ID]


async def test_save_folder_keeps_pinned_and_included_disjoint_and_a_hand_pinned_chat(
    ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    stub = links[0]
    hand_pinned = types.InputPeerChat(GROUP_BARE)  # the user pinned their own group in there
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [
            _folder(
                2,
                "Curated",
                pinned=[hand_pinned],
                include=[types.InputPeerChannel(CHANNEL_BARE, 1)],
            )
        ]
    )
    ugw.register_own_folder(2)
    current = await ugw.get_folder(2)
    assert current == [GROUP_ID, CHANNEL_ID]
    await ugw.save_folder(2, "Curated", [*current, OUTPUT_ID])
    [request] = stub.requests_of(functions.messages.UpdateDialogFilterRequest)
    pinned = [utils.get_peer_id(p) for p in request.filter.pinned_peers]
    included = [utils.get_peer_id(p) for p in request.filter.include_peers]
    assert pinned == [GROUP_ID]
    assert included == [CHANNEL_ID, OUTPUT_ID]


async def test_a_pinned_topic_channel_does_not_make_every_sync_rewrite_the_folder(
    rt: Runtime, ugw: TelethonUserGateway, links: list[LinkStub]
) -> None:
    stub = links[0]
    rt.user = ugw
    await _output_chat(rt)
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters([])
    folders = FolderManager(rt)
    await folders.sync()  # creates "Curated" with the topic channel
    [created] = stub.requests_of(functions.messages.UpdateDialogFilterRequest)
    title = rt.settings.folders.curated_name
    # the owner pins the topic channel inside the folder: Telegram moves it to pinned_peers
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_folder(created.id, title, pinned=[types.InputPeerChannel(OUTPUT_BARE, 1)], include=[])]
    )
    await folders.sync()
    await folders.sync()
    assert len(stub.requests_of(functions.messages.UpdateDialogFilterRequest)) == 1
