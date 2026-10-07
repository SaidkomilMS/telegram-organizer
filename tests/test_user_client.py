"""Unit tests for the Telethon user gateway (DESIGN §5) without a network.

The pure parts (TL object -> dataclass conversion, reference parsing, error translation) are
tested on TL objects built here; the guard rules, request shapes and the session-loss
procedure are tested against a stub client that records every request it is handed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import TelegramClient, errors, functions, types, utils

from tests.fakes import FakeClock
from tg_curator.errors import (
    ChatGone,
    FloodWait,
    FolderLimit,
    ForwardsRestricted,
    LoginError,
    MediaUnavailable,
    NotAllowed,
    NotOwnedError,
    SessionLost,
    TelegramUnavailable,
)
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage, UserGateway
from tg_curator.telegram.user_client import (
    FLOOD_SLEEP_THRESHOLD,
    MUTE_FOREVER,
    TelethonUserGateway,
    chat_info_from_dialog,
    chat_info_from_entity,
    extract_urls,
    login_error,
    media_kind,
    message_to_incoming,
    mute_timestamp,
    parse_chat_ref,
    translate_error,
)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
CHANNEL_BARE = 1_234_567_890
CHANNEL_ID = -1_000_000_000_000 - CHANNEL_BARE
GROUP_BARE = 987
GROUP_ID = -GROUP_BARE
OUTPUT_BARE = 555
OUTPUT_ID = -1_000_000_000_000 - OUTPUT_BARE
BOT_USER = types.User(id=42, bot=True, access_hash=7, first_name="Curator", username="curator_bot")


def channel(
    bare: int = CHANNEL_BARE,
    *,
    title: str = "Source",
    broadcast: bool = True,
    megagroup: bool = False,
    gigagroup: bool = False,
    username: str | None = "source_chan",
    noforwards: bool = False,
    creator: bool = False,
    admin_rights: types.ChatAdminRights | None = None,
    left: bool = False,
    monoforum: bool = False,
    usernames: list[types.Username] | None = None,
) -> types.Channel:
    return types.Channel(
        id=bare,
        title=title,
        photo=types.ChatPhotoEmpty(),
        date=NOW,
        broadcast=broadcast,
        megagroup=megagroup,
        gigagroup=gigagroup,
        username=username,
        noforwards=noforwards,
        creator=creator,
        admin_rights=admin_rights,
        left=left,
        monoforum=monoforum,
        usernames=usernames,
        access_hash=99,
    )


def basic_group(
    bare: int = GROUP_BARE, *, creator: bool = False, deactivated: bool = False
) -> types.Chat:
    return types.Chat(
        id=bare,
        title="Group",
        photo=types.ChatPhotoEmpty(),
        participants_count=3,
        date=NOW,
        version=1,
        creator=creator,
        deactivated=deactivated,
    )


def tl_message(
    message_id: int = 10,
    *,
    peer: types.TypePeer | None = None,
    text: str = "hello",
    entities: list[Any] | None = None,
    media: Any = None,
    out: bool = False,
    from_id: types.TypePeer | None = None,
    fwd_from: types.MessageFwdHeader | None = None,
    reply_to: types.MessageReplyHeader | None = None,
    grouped_id: int | None = None,
    views: int | None = None,
    forwards: int | None = None,
    noforwards: bool = False,
    date: datetime | None = NOW,
) -> types.Message:
    return types.Message(
        id=message_id,
        peer_id=peer or types.PeerChannel(CHANNEL_BARE),
        date=date,
        message=text,
        entities=entities,
        media=media,
        out=out,
        from_id=from_id,
        fwd_from=fwd_from,
        reply_to=reply_to,
        grouped_id=grouped_id,
        views=views,
        forwards=forwards,
        noforwards=noforwards,
    )


def photo_media() -> types.MessageMediaPhoto:
    return types.MessageMediaPhoto(
        photo=types.Photo(id=1, access_hash=2, file_reference=b"ref", date=NOW, sizes=[], dc_id=2)
    )


def document_media(
    *, video: bool = False, round_message: bool = False
) -> types.MessageMediaDocument:
    attributes: list[Any] = [types.DocumentAttributeFilename("a.bin")]
    if video or round_message:
        attributes.append(types.DocumentAttributeVideo(1.0, 10, 10, round_message=round_message))
    return types.MessageMediaDocument(
        document=types.Document(
            id=3,
            access_hash=4,
            file_reference=b"ref",
            date=NOW,
            mime_type="application/octet-stream",
            size=1,
            dc_id=2,
            attributes=attributes,
        ),
        video=video,
        round=round_message,
    )


def chat_info(**changes: Any) -> ChatInfo:
    base = ChatInfo(
        id=CHANNEL_ID,
        kind="channel",
        title="Source",
        username="source_chan",
        noforwards=False,
        is_creator=False,
        is_admin=False,
        archived=False,
        muted_until=None,
    )
    return replace(base, **changes)


# --- the stub client ------------------------------------------------------------------------


class StubClient:
    """Records everything the gateway asks of it; answers from small tables."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.handlers: list[tuple[Any, Any]] = []
        self.responses: dict[type, Any] = {}
        self.entities: dict[Any, Any] = {}
        self.history: dict[int, list[Any]] = {}
        self.dialogs: dict[int, list[Any]] = {0: [], 1: []}
        self.send_file_errors: list[Exception] = []
        self.connected = False
        self.gone: set[int] = set()  # message ids forward_messages finds deleted
        self.authorised = True
        self.me = types.User(id=1001, first_name="Owner", last_name="One", phone="998901234512")
        self.sign_in_error: Exception | None = None
        self.send_code_error: Exception | None = None
        self._next_id = 100
        self._updates_error: Exception | None = None

    async def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        answer = self.responses.get(type(request))
        if callable(answer) and not isinstance(answer, type):
            answer = answer(request)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def requests_of(self, cls: type) -> list[Any]:
        return [r for r in self.requests if isinstance(r, cls)]

    def add_event_handler(self, callback: Any, event: Any = None) -> None:
        self.handlers.append((callback, event))

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def is_user_authorized(self) -> bool:
        return self.authorised

    async def get_me(self) -> types.User | None:
        return self.me if self.authorised else None

    async def send_code_request(self, phone: str) -> types.auth.SentCode:
        self.calls.append(("send_code_request", {"phone": phone}))
        if self.send_code_error is not None:
            raise self.send_code_error
        return types.auth.SentCode(types.auth.SentCodeTypeApp(5), "hash-1")

    async def sign_in(self, phone: str | None = None, code: str | None = None, **kw: Any) -> Any:
        self.calls.append(("sign_in", {"phone": phone, "code": code, **kw}))
        if self.sign_in_error is not None:
            raise self.sign_in_error
        self.authorised = True
        return self.me

    async def get_input_entity(self, ref: Any) -> Any:
        self.calls.append(("get_input_entity", {"ref": ref}))
        if isinstance(ref, str):
            entity = self.entities.get(ref.lstrip("@").lower())
            if entity is None:
                raise ValueError(f"No user has {ref!r} as username")
            return utils.get_input_peer(entity)
        if isinstance(ref, int):
            bare, kind = utils.resolve_id(ref)
            if kind is types.PeerChannel:
                return types.InputPeerChannel(bare, 1)
            if kind is types.PeerChat:
                return types.InputPeerChat(bare)
            return types.InputPeerUser(bare, 1)
        return ref

    async def get_entity(self, ref: Any) -> Any:
        self.calls.append(("get_entity", {"ref": ref}))
        key = ref
        if isinstance(ref, str):
            key = ref.lstrip("@").lower()
        elif isinstance(ref, types.InputPeerChannel):
            key = utils.get_peer_id(ref)
        elif isinstance(ref, types.InputPeerChat):
            key = utils.get_peer_id(ref)
        if key not in self.entities:
            raise ValueError(f"Could not find the input entity for {ref!r}")
        return self.entities[key]

    def iter_messages(self, entity: Any, limit: Any = None, **kw: Any) -> AsyncIterator[Any]:
        self.calls.append(("iter_messages", {"entity": entity, "limit": limit, **kw}))
        messages = list(self.history.get(utils.get_peer_id(entity), []))
        messages.sort(key=lambda m: m.id, reverse=not kw.get("reverse"))
        offset = kw.get("offset_date")
        if offset is not None and kw.get("reverse"):
            messages = [m for m in messages if m.date > offset]

        async def gen() -> AsyncIterator[Any]:
            for i, msg in enumerate(messages):
                if limit is not None and i >= limit:
                    return
                yield msg

        return gen()

    async def get_messages(self, entity: Any, ids: list[int] | None = None, **kw: Any) -> list[Any]:
        self.calls.append(("get_messages", {"entity": entity, "ids": ids}))
        known = {m.id: m for m in self.history.get(utils.get_peer_id(entity), [])}
        return [known.get(i) for i in ids or []]

    async def send_file(self, entity: Any, file: Any, **kw: Any) -> Any:
        self.calls.append(("send_file", {"entity": entity, "file": file}))
        if self.send_file_errors:
            raise self.send_file_errors.pop(0)
        self._next_id += 1
        return SimpleNamespace(id=self._next_id)

    async def forward_messages(
        self, entity: Any, messages: list[int], from_peer: Any = None, **kw: Any
    ) -> list[Any]:
        self.calls.append(
            ("forward_messages", {"entity": entity, "messages": messages, "from_peer": from_peer})
        )
        out: list[Any] = []
        for mid in messages:
            if mid in self.gone:
                out.append(None)  # Telegram forwards nothing for a deleted message
                continue
            self._next_id += 1
            out.append(SimpleNamespace(id=self._next_id))
        return out

    def iter_dialogs(self, folder: int | None = None, **kw: Any) -> AsyncIterator[Any]:
        self.calls.append(("iter_dialogs", {"folder": folder}))

        async def gen() -> AsyncIterator[Any]:
            for dialog in self.dialogs.get(folder or 0, []):
                yield dialog

        return gen()


def dialog(entity: Any, *, archived: bool = False, mute_until: datetime | None = None) -> Any:
    settings = types.PeerNotifySettings(mute_until=mute_until)
    return SimpleNamespace(
        entity=entity, archived=archived, dialog=SimpleNamespace(notify_settings=settings)
    )


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


@pytest.fixture
def gw(home: Path, clock: FakeClock, stub: StubClient) -> TelethonUserGateway:
    factory_calls = {"n": 0}

    def factory() -> StubClient:
        factory_calls["n"] += 1
        return stub if factory_calls["n"] == 1 else StubClient()

    gateway = TelethonUserGateway(home, 1, "hash", clock=clock, client_factory=factory)
    stub.entities[CHANNEL_ID] = channel()
    stub.entities[GROUP_ID] = basic_group()
    stub.entities[OUTPUT_ID] = channel(OUTPUT_BARE, title="Output", username=None, creator=True)
    stub.entities["curator_bot"] = BOT_USER
    return gateway


# --- pure conversions: chats --------------------------------------------------------------


def test_broadcast_channel_classification() -> None:
    info = chat_info_from_entity(channel(noforwards=True))
    assert info == ChatInfo(
        id=CHANNEL_ID,
        kind="channel",
        title="Source",
        username="source_chan",
        noforwards=True,
        is_creator=False,
        is_admin=False,
        archived=False,
        muted_until=None,
    )


def test_megagroup_is_a_group_and_gigagroup_a_channel() -> None:
    assert chat_info_from_entity(channel(broadcast=False, megagroup=True)).kind == "group"
    assert (
        chat_info_from_entity(channel(broadcast=False, megagroup=True, gigagroup=True)).kind
        == "channel"
    )


def test_is_admin_needs_post_rights_in_a_channel() -> None:
    no_post = types.ChatAdminRights(change_info=True)
    assert chat_info_from_entity(channel(admin_rights=no_post)).is_admin is False
    assert chat_info_from_entity(
        channel(admin_rights=types.ChatAdminRights(post_messages=True))
    ).is_admin
    assert chat_info_from_entity(channel(creator=True)).is_creator is True
    # in a group any admin rights count: there is no post right to check
    group = channel(broadcast=False, megagroup=True, admin_rights=no_post)
    assert chat_info_from_entity(group).is_admin is True


def test_username_falls_back_to_the_first_active_collectible() -> None:
    names = [types.Username("old", active=False), types.Username("fresh", active=True)]
    assert chat_info_from_entity(channel(username=None, usernames=names)).username == "fresh"
    assert chat_info_from_entity(channel(username=None)).username is None


def test_basic_group_and_skipped_entities() -> None:
    info = chat_info_from_entity(basic_group(creator=True))
    assert info is not None and (info.id, info.kind, info.is_creator) == (GROUP_ID, "group", True)
    assert chat_info_from_entity(basic_group(deactivated=True)) is None
    assert chat_info_from_entity(channel(monoforum=True)) is None
    assert chat_info_from_entity(types.User(id=5)) is None
    # a left channel is still a readable entity (public channels work without membership)
    assert chat_info_from_entity(channel(left=True)) is not None


def test_dialog_carries_archive_and_mute_and_drops_left_chats() -> None:
    later = NOW + timedelta(days=3)
    info = chat_info_from_dialog(dialog(channel(), archived=True, mute_until=later), now=NOW)
    assert info is not None and info.archived and info.muted_until == later
    expired = chat_info_from_dialog(dialog(channel(), mute_until=NOW - timedelta(days=1)), now=NOW)
    assert expired is not None and expired.muted_until is None
    assert chat_info_from_dialog(dialog(channel(left=True)), now=NOW) is None


# --- pure conversions: messages -----------------------------------------------------------


def test_message_conversion_with_entities_urls_and_counters() -> None:
    text = "Bold news https://example.com/a?utm_source=x see www.other.org/p."
    entities = [
        types.MessageEntityBold(0, 4),
        types.MessageEntityUrl(10, 34),
        types.MessageEntityTextUrl(45, 3, "https://linked.example/q"),
    ]
    fwd = types.MessageFwdHeader(date=NOW, from_id=types.PeerChannel(77), channel_post=5)
    msg = tl_message(
        text=text,
        entities=entities,
        media=photo_media(),
        from_id=types.PeerUser(500),
        fwd_from=fwd,
        reply_to=types.MessageReplyHeader(reply_to_msg_id=9),
        grouped_id=123,
        views=1500,
        forwards=12,
    )
    incoming = message_to_incoming(msg, chat_info(), now=NOW)
    assert incoming == IncomingMessage(
        chat=chat_info(),
        message_id=10,
        date=NOW,
        text=text,
        html=(
            '<strong>Bold</strong> news <a href="https://example.com/a?utm_source=x">'
            'https://example.com/a?utm_source=x</a> <a href="https://linked.example/q">see</a>'
            " www.other.org/p."
        ),
        sender_id=500,
        is_outgoing=False,
        is_service=False,
        reply_to_id=9,
        grouped_id=123,
        media="photo",
        urls=("https://example.com/a?utm_source=x", "https://linked.example/q", "www.other.org/p"),
        views=1500,
        forwards=12,
        fwd_from_chat_id=-1_000_000_000_077,
        fwd_from_message_id=5,
        noforwards=False,
    )


def test_a_channels_copy_in_its_discussion_group_is_an_automatic_forward() -> None:
    source = types.PeerChannel(77)
    group = types.PeerChannel(88)
    auto = tl_message(
        30,
        peer=group,
        from_id=source,
        fwd_from=types.MessageFwdHeader(
            date=NOW, from_id=source, channel_post=5, saved_from_peer=source, saved_from_msg_id=5
        ),
    )
    got = message_to_incoming(auto, chat_info(), now=NOW)
    assert got is not None and got.is_automatic_forward
    assert (got.fwd_from_chat_id, got.fwd_from_message_id) == (-1_000_000_000_077, 5)
    # a member forwarding the same post is an ordinary forward
    member = tl_message(
        31,
        peer=group,
        from_id=types.PeerUser(500),
        fwd_from=types.MessageFwdHeader(date=NOW, from_id=source, channel_post=5),
    )
    assert not message_to_incoming(member, chat_info(), now=NOW).is_automatic_forward
    # ... and so is one the member saved from that channel
    saved = tl_message(
        32,
        peer=group,
        from_id=types.PeerUser(500),
        fwd_from=types.MessageFwdHeader(
            date=NOW, from_id=source, channel_post=5, saved_from_peer=source, saved_from_msg_id=5
        ),
    )
    assert not message_to_incoming(saved, chat_info(), now=NOW).is_automatic_forward
    assert not message_to_incoming(tl_message(), chat_info(), now=NOW).is_automatic_forward


def test_plain_message_has_no_html_and_marks_outgoing_and_noforwards() -> None:
    msg = tl_message(text="plain", out=True, noforwards=True)
    incoming = message_to_incoming(msg, chat_info(), now=NOW)
    assert incoming is not None
    assert incoming.html is None and incoming.urls == ()
    assert incoming.is_outgoing is True and incoming.noforwards is True
    assert incoming.sender_id is None and incoming.reply_to_id is None
    # noforwards also follows the chat
    assert message_to_incoming(tl_message(), chat_info(noforwards=True), now=NOW).noforwards is True


def test_service_messages_are_never_converted() -> None:
    service = types.MessageService(
        id=3, peer_id=types.PeerChannel(CHANNEL_BARE), action=types.MessageActionPinMessage()
    )
    assert message_to_incoming(service, chat_info(), now=NOW) is None


def test_forum_topic_root_is_not_a_reply_but_a_reply_inside_a_topic_is() -> None:
    root = types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=1)
    assert message_to_incoming(tl_message(reply_to=root), chat_info(), now=NOW).reply_to_id is None
    inner = types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=8, reply_to_top_id=1)
    assert message_to_incoming(tl_message(reply_to=inner), chat_info(), now=NOW).reply_to_id == 8


def test_forwards_from_users_carry_no_origin() -> None:
    fwd = types.MessageFwdHeader(date=NOW, from_id=types.PeerUser(5))
    incoming = message_to_incoming(tl_message(fwd_from=fwd), chat_info(), now=NOW)
    assert (incoming.fwd_from_chat_id, incoming.fwd_from_message_id) == (None, None)


@pytest.mark.parametrize(
    ("media", "kind"),
    [
        (None, None),
        (types.MessageMediaEmpty(), None),
        (photo_media(), "photo"),
        (document_media(video=True), "video"),
        (document_media(), "file"),
        (document_media(round_message=True), "file"),
        (types.MessageMediaWebPage(types.WebPageEmpty(id=1)), "other"),
        (types.MessageMediaStory(types.PeerChannel(1), 2), "other"),
        (types.MessageMediaPaidMedia(1, []), "other"),
        (types.MessageMediaGeo(types.GeoPointEmpty()), "other"),
    ],
)
def test_media_kinds(media: Any, kind: str | None) -> None:
    assert media_kind(media) == kind


def test_extract_urls_handles_surrogates_and_trailing_punctuation() -> None:
    text = "😀 look https://a.b/c, and https://a.b/c again"
    entities = [types.MessageEntityUrl(8, 13)]
    assert extract_urls(text, entities) == ("https://a.b/c",)


# --- pure conversions: references, errors, timestamps -------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (CHANNEL_ID, ("id", CHANNEL_ID)),
        (str(CHANNEL_ID), ("id", CHANNEL_ID)),
        ("@KunUz", ("username", "kunuz")),
        ("kunuz", ("username", "kunuz")),
        ("https://t.me/kunuz", ("username", "kunuz")),
        ("t.me/kunuz/123", ("username", "kunuz")),
        ("https://telegram.me/kunuz?start=1", ("username", "kunuz")),
        ("https://t.me/c/1234567890/55", ("id", CHANNEL_ID)),
        ("https://t.me/+AbCdEf", ("invite", "AbCdEf")),
        ("t.me/joinchat/XyZ", ("invite", "XyZ")),
        ("tg://join?invite=QQ", ("invite", "QQ")),
        ("tg://resolve?domain=Kunuz", ("username", "kunuz")),
    ],
)
def test_parse_chat_ref(ref: str | int, expected: tuple[str, int | str]) -> None:
    assert parse_chat_ref(ref) == expected


def _rpc(cls: type, message: str = "X", **kw: Any) -> Exception:
    try:
        return cls(request=None, **kw)
    except TypeError:
        return cls(request=None, message=message)


@pytest.mark.parametrize(
    ("exc", "expected_type", "reason"),
    [
        (_rpc(errors.AuthKeyUnregisteredError), SessionLost, "unregistered"),
        (_rpc(errors.AuthKeyInvalidError), SessionLost, "invalid"),
        (_rpc(errors.SessionRevokedError), SessionLost, "revoked"),
        (_rpc(errors.SessionExpiredError), SessionLost, "expired"),
        (_rpc(errors.UserDeactivatedError), SessionLost, "deactivated"),
        (_rpc(errors.UserDeactivatedBanError), SessionLost, "banned"),
        (_rpc(errors.AuthKeyDuplicatedError), SessionLost, "duplicated"),
        (errors.FloodWaitError(request=None, capture=300), FloodWait, None),
        (_rpc(errors.ChatForwardsRestrictedError), ForwardsRestricted, None),
        (_rpc(errors.FileReferenceExpiredError), MediaUnavailable, None),
        (_rpc(errors.MessageIdInvalidError), MediaUnavailable, None),  # deleted at the source
        (_rpc(errors.MessageIdsEmptyError), MediaUnavailable, None),
        (_rpc(errors.ChannelPrivateError), ChatGone, None),
        (_rpc(errors.UsernameNotOccupiedError), ChatGone, None),
        (_rpc(errors.InviteHashExpiredError), ChatGone, None),
        (_rpc(errors.UserCreatorError), NotAllowed, "creator"),
        (_rpc(errors.FreshChangeAdminsForbiddenError), NotAllowed, "fresh_session"),
        (
            errors.BadRequestError(request=None, message="DIALOG_FILTERS_TOO_MUCH"),
            FolderLimit,
            None,
        ),
        (errors.BadRequestError(request=None, message="FILTER_INCLUDE_EMPTY"), ValueError, None),
        (_rpc(errors.ChatAdminRequiredError), NotAllowed, "other"),
        (ConnectionError("Cannot send requests while disconnected"), TelegramUnavailable, None),
        (TimeoutError(), TelegramUnavailable, None),
        (OSError("Network is unreachable"), TelegramUnavailable, None),
        (errors.ServerError(request=None, message="INTERNAL"), TelegramUnavailable, None),
        (_rpc(errors.HistoryGetFailedError), TelegramUnavailable, None),
        (errors.RPCError(None, "No workers running", -500), TelegramUnavailable, None),
        (errors.TimedOutError(None, "Timeout", -503), TelegramUnavailable, None),
        (errors.AuthKeyNotFound(), SessionLost, "unregistered"),
    ],
)
def test_error_translation_table(exc: Exception, expected_type: type, reason: str | None) -> None:
    translated = translate_error(exc)
    assert isinstance(translated, expected_type)
    if reason is not None:
        assert translated.reason == reason
    if isinstance(translated, FloodWait):
        assert translated.seconds == 300


def test_password_needed_and_foreign_errors_are_not_translated() -> None:
    assert translate_error(_rpc(errors.SessionPasswordNeededError)) is None
    assert translate_error(ValueError("x")) is None


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (_rpc(errors.PhoneCodeInvalidError), "bad_code"),
        (_rpc(errors.PhoneCodeExpiredError), "expired_code"),
        (_rpc(errors.PasswordHashInvalidError), "bad_password"),
        (_rpc(errors.PhoneNumberInvalidError), "bad_phone"),
        (errors.FloodWaitError(request=None, capture=30), "flood"),
    ],
)
def test_login_error_table(exc: Exception, reason: str) -> None:
    assert login_error(exc).reason == reason


def test_mute_timestamps() -> None:
    assert mute_timestamp(None) == 0
    assert mute_timestamp(datetime.max) == MUTE_FOREVER
    assert mute_timestamp(datetime.max.replace(tzinfo=UTC)) == MUTE_FOREVER
    assert mute_timestamp(NOW) == int(NOW.timestamp())


# --- the real client is built with the contract's flags ------------------------------------


def test_real_client_flags(home: Path) -> None:
    gateway = TelethonUserGateway(home, 1, "hash", clock=FakeClock(NOW))
    client = gateway._client
    assert isinstance(client, TelegramClient)
    assert client._catch_up is False
    assert client.flood_sleep_threshold == FLOOD_SLEEP_THRESHOLD == 120
    assert client._raise_last_call_error is True  # an outage surfaces as itself, not ValueError
    assert client._init_request.device_model == "tg-curator"
    assert client._init_request.app_version.startswith("tg-curator")
    assert (home / "user.session").exists()
    assert isinstance(gateway, UserGateway)


# --- login flow ---------------------------------------------------------------------------


async def test_login_flow_keeps_the_hash_in_memory(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.authorised = False
    assert await gw.connect() is False
    assert await gw.me() is None
    await gw.send_code("+998901234512")
    assert gw._phone_code_hash == "hash-1"
    stub.responses[functions.auth.ResendCodeRequest] = types.auth.SentCode(
        types.auth.SentCodeTypeSms(5), "hash-2"
    )
    await gw.resend_code()
    assert stub.requests_of(functions.auth.ResendCodeRequest)[0].phone_code_hash == "hash-1"
    stub.sign_in_error = _rpc(errors.SessionPasswordNeededError)
    assert await gw.sign_in("12345") == "password_needed"
    assert stub.calls[-1] == (
        "sign_in",
        {"phone": "+998901234512", "code": "12345", "phone_code_hash": "hash-2"},
    )
    stub.sign_in_error = None
    await gw.sign_in_password("secret")
    assert gw._phone_code_hash is None
    me = await gw.me()
    assert me is not None and (me.id, me.name, me.phone) == (1001, "Owner One", "998901234512")


async def test_sign_in_without_a_pending_code_is_a_login_error(gw: TelethonUserGateway) -> None:
    with pytest.raises(LoginError) as info:
        await gw.sign_in("1 2 3")
    assert info.value.reason == "other"


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (_rpc(errors.PhoneCodeInvalidError), "bad_code"),
        (_rpc(errors.PhoneCodeExpiredError), "expired_code"),
        (errors.FloodWaitError(request=None, capture=600), "flood"),
    ],
)
async def test_sign_in_errors_are_login_errors(
    gw: TelethonUserGateway, stub: StubClient, exc: Exception, reason: str
) -> None:
    await gw.send_code("+998901234512")
    stub.sign_in_error = exc
    with pytest.raises(LoginError) as info:
        await gw.sign_in("12345")
    assert info.value.reason == reason


async def test_bad_phone_and_bad_password(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.send_code_error = _rpc(errors.PhoneNumberInvalidError)
    with pytest.raises(LoginError) as info:
        await gw.send_code("+1")
    assert info.value.reason == "bad_phone"
    stub.sign_in_error = _rpc(errors.PasswordHashInvalidError)
    with pytest.raises(LoginError) as info:
        await gw.sign_in_password("wrong")
    assert info.value.reason == "bad_password"


# --- session loss -------------------------------------------------------------------------


async def test_reset_session_deletes_the_files_and_starts_an_empty_client(
    gw: TelethonUserGateway, stub: StubClient, home: Path
) -> None:
    (home / "user.session").write_bytes(b"old key")
    (home / "user.session-journal").write_bytes(b"")
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gw.on_session_lost(lost)
    await gw.connect()
    await gw.send_code("+998901234512")
    await gw.reset_session()
    assert stub.connected is False
    assert not (home / "user.session").exists() and not (home / "user.session-journal").exists()
    assert gw._client is not stub  # the old auth key leaves with the old client
    assert len(gw._client.handlers) == 2  # live intake keeps working after the new login
    assert reasons == []  # a reset is the owner's choice, not a session loss
    with pytest.raises(LoginError):
        await gw.sign_in("12345")  # the old code request is gone with the old session


async def test_a_stray_401_during_a_pending_login_keeps_the_code(
    gw: TelethonUserGateway, stub: StubClient, home: Path
) -> None:
    """A loop still calling the account while /bind waits for the code gets SessionLost, but
    the unauthorised session is not retired: the pending code stays usable."""
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gw.on_session_lost(lost)
    stub.authorised = False
    assert await gw.connect() is False
    (home / "user.session").write_bytes(b"login")
    await gw.send_code("+998901234512")
    stub.responses[functions.messages.GetDialogFiltersRequest] = _rpc(
        errors.AuthKeyUnregisteredError
    )
    with pytest.raises(SessionLost):
        await gw.list_folders()
    assert reasons == []
    assert gw._client is stub and gw._phone_code_hash == "hash-1"
    assert (home / "user.session").read_bytes() == b"login"
    assert await gw.sign_in("12345") == "ok"


async def test_session_loss_renames_the_file_and_tells_once(
    gw: TelethonUserGateway, stub: StubClient, home: Path, clock: FakeClock
) -> None:
    (home / "user.session").write_bytes(b"key")
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gw.on_session_lost(lost)
    await gw.connect()
    stub.responses[functions.users.GetUsersRequest] = _rpc(errors.AuthKeyUnregisteredError)
    assert await gw.ping() is False
    assert reasons == ["unregistered"]
    assert not (home / "user.session").exists()
    stamp = clock.now().strftime("%Y%m%dT%H%M%SZ")
    assert (home / f"user.session.revoked-{stamp}").read_bytes() == b"key"
    assert stub.connected is False
    assert gw._client is not stub  # a fresh session was created
    assert len(gw._client.handlers) == 2  # handlers re-installed on the fresh client
    # a second error from the dead client does not tell the handler again
    await gw._session_lost("unregistered", "again")
    assert reasons == ["unregistered"]


async def test_duplicated_key_and_update_loop_errors_count(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gw.on_session_lost(lost)
    await gw.connect()
    stub._updates_error = _rpc(errors.AuthKeyDuplicatedError)
    assert await gw.ping() is False
    assert reasons == ["duplicated"]


async def test_write_errors_raise_session_lost(gw: TelethonUserGateway, stub: StubClient) -> None:
    await gw.connect()
    stub.responses[functions.channels.CreateChannelRequest] = _rpc(errors.SessionRevokedError)
    with pytest.raises(SessionLost) as info:
        await gw.create_channel("Topic")
    assert info.value.reason == "revoked"


async def test_ping_true_when_the_session_answers(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    assert await gw.ping() is False  # not connected yet
    await gw.connect()
    stub.responses[functions.users.GetUsersRequest] = [stub.me]
    assert await gw.ping() is True
    request = stub.requests_of(functions.users.GetUsersRequest)[0]
    assert isinstance(request.id[0], types.InputUserSelf)


# --- re-login beside the working session ----------------------------------------------------


class HangingStub(StubClient):
    """A working client whose folder reads hang until Telethon would fail them: the key's
    error once the session is logged out, a cancelled wait once the client disconnects."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting: list[asyncio.Future[Any]] = []

    async def __call__(self, request: Any) -> Any:
        if isinstance(request, functions.messages.GetDialogFiltersRequest):
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self.waiting.append(future)
            return await future
        if isinstance(request, functions.auth.LogOutRequest) and self.waiting:
            self.waiting.pop(0).set_exception(_rpc(errors.AuthKeyUnregisteredError))
        return await super().__call__(request)

    async def disconnect(self) -> None:
        for future in self.waiting:
            future.cancel()
        await super().disconnect()


class Relogins:
    """The clients a gateway builds: the working ones in order, and each re-login's."""

    def __init__(self, first: StubClient) -> None:
        self.main: list[StubClient] = [first]
        self.pending: list[StubClient] = []
        self._first_used = False

    def make_main(self) -> StubClient:
        if not self._first_used:
            self._first_used = True
            return self.main[0]
        client = StubClient()
        self.main.append(client)
        return client

    def make_pending(self) -> StubClient:
        client = StubClient()
        client.authorised = False
        self.pending.append(client)
        return client


def relogin_gateway(home: Path, clock: FakeClock, first: StubClient) -> tuple[Any, Relogins]:
    clients = Relogins(first)
    gateway = TelethonUserGateway(
        home,
        1,
        "hash",
        clock=clock,
        client_factory=clients.make_main,
        pending_client_factory=clients.make_pending,
    )
    return gateway, clients


def test_the_pending_client_is_the_same_application_on_its_own_file(home: Path) -> None:
    gateway = TelethonUserGateway(home, 1, "hash", clock=FakeClock(NOW))
    client = gateway._make_pending_client()
    try:
        assert client.session.filename == str(home / "user.session.pending")
        assert (home / "user.session.pending").exists()
        assert not (home / "user.session.pending.session").exists()
        assert (client.api_id, client.api_hash) == (1, "hash")
        assert client._init_request.device_model == "tg-curator"
        assert client._catch_up is False and client._no_updates is True
    finally:
        client.session.close()


async def test_a_relogin_asks_for_the_code_on_its_own_client(
    home: Path, clock: FakeClock, stub: StubClient
) -> None:
    gateway, clients = relogin_gateway(home, clock, stub)
    (home / "user.session").write_bytes(b"old key")
    assert await gateway.connect() is True
    await gateway.begin_relogin()
    pending_file = home / "user.session.pending"
    assert pending_file.stat().st_mode & 0o777 == 0o600
    pending = clients.pending[0]
    assert pending.connected
    await gateway.send_code("+998901234512")
    assert pending.calls == [("send_code_request", {"phone": "+998901234512"})]
    assert not [c for c in stub.calls if c[0] == "send_code_request"]
    assert gateway._phone_code_hash is None  # the hash lives on the pending login only
    assert gateway._pending is not None and gateway._pending.phone_code_hash == "hash-1"
    pending.responses[functions.auth.ResendCodeRequest] = types.auth.SentCode(
        types.auth.SentCodeTypeSms(5), "hash-2"
    )
    await gateway.resend_code()
    assert pending.requests_of(functions.auth.ResendCodeRequest)[0].phone_code_hash == "hash-1"
    assert stub.requests_of(functions.auth.ResendCodeRequest) == []
    me = await gateway.me()  # the working session still answers for the account
    assert me is not None and me.id == 1001
    assert gateway._client is stub and stub.connected


async def test_errors_of_the_pending_login_never_touch_the_working_session(
    home: Path, clock: FakeClock, stub: StubClient
) -> None:
    gateway, clients = relogin_gateway(home, clock, stub)
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gateway.on_session_lost(lost)
    (home / "user.session").write_bytes(b"old key")
    await gateway.connect()
    await gateway.begin_relogin()
    clients.pending[0].send_code_error = _rpc(errors.AuthKeyUnregisteredError)
    with pytest.raises(LoginError) as refused:  # a failed step of the new login, no loss
        await gateway.send_code("+998901234512")
    assert refused.value.reason == "other"
    clients.pending[0].send_code_error = None
    await gateway.send_code("+998901234512")
    clients.pending[0].sign_in_error = _rpc(errors.PhoneCodeInvalidError)
    with pytest.raises(LoginError) as info:
        await gateway.sign_in("11111")
    assert info.value.reason == "bad_code"
    assert reasons == [] and gateway._client is stub and stub.connected
    assert (home / "user.session").read_bytes() == b"old key"
    assert (home / "user.session.pending").exists()  # the owner may still type the right one


async def test_a_swap_that_cannot_store_the_file_keeps_the_working_session(
    home: Path, clock: FakeClock, stub: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, clients = relogin_gateway(home, clock, stub)
    (home / "user.session").write_bytes(b"old key")
    await gateway.connect()
    await gateway.begin_relogin()
    await gateway.send_code("+998901234512")

    def refuse(src: Any, dst: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr("tg_curator.telegram.user_client.os.replace", refuse)
    with pytest.raises(LoginError) as info:
        await gateway.sign_in("12345")
    monkeypatch.undo()
    assert info.value.reason == "other"
    assert gateway._client is stub and stub.connected and gateway._pending is None
    assert stub.requests_of(functions.auth.LogOutRequest) == []
    assert (home / "user.session").read_bytes() == b"old key"
    assert not (home / "user.session.pending").exists()
    assert len(clients.main) == 1


async def test_requests_cut_off_by_the_swap_are_an_outage_not_a_loss(
    home: Path, clock: FakeClock
) -> None:
    """The loops keep running through a re-bind: what the swap does to their requests in
    flight on the old client must not look like a lost session (which would retire the new
    one) nor kill the loop with a CancelledError."""
    working = HangingStub()
    gateway, clients = relogin_gateway(home, clock, working)
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gateway.on_session_lost(lost)
    (home / "user.session").write_bytes(b"old key")
    await gateway.connect()

    cancelled = asyncio.create_task(gateway.list_folders())
    while len(working.waiting) < 1:
        await asyncio.sleep(0)
    cancelled.cancel()  # a real cancellation of the caller still cancels it
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    working.waiting.clear()

    at_logout = asyncio.create_task(gateway.list_folders())
    at_disconnect = asyncio.create_task(gateway.list_folders())
    while len(working.waiting) < 2:
        await asyncio.sleep(0)
    await gateway.begin_relogin()
    (home / "user.session.pending").write_bytes(b"new key")
    await gateway.send_code("+998901234512")
    assert await gateway.sign_in("12345") == "ok"
    for task in (at_logout, at_disconnect):
        with pytest.raises(TelegramUnavailable):
            await task
    assert reasons == []
    new = clients.main[-1]
    assert gateway._client is new and new.connected and len(new.handlers) == 2
    assert (home / "user.session").read_bytes() == b"new key"
    assert working.requests_of(functions.auth.LogOutRequest) and not working.connected


async def test_disconnect_and_reset_abandon_a_pending_relogin(
    home: Path, clock: FakeClock, stub: StubClient
) -> None:
    gateway, clients = relogin_gateway(home, clock, stub)
    (home / "user.session").write_bytes(b"old key")
    await gateway.connect()
    await gateway.begin_relogin()
    await gateway.disconnect()  # a clean stop: the login dies with the process
    assert not (home / "user.session.pending").exists() and not clients.pending[0].connected
    await gateway.connect()
    await gateway.begin_relogin()
    await gateway.begin_relogin()  # /bind again: the earlier one is abandoned first
    assert not clients.pending[1].connected and clients.pending[2].connected
    await gateway.reset_session()
    assert gateway._pending is None and not (home / "user.session.pending").exists()
    assert not clients.pending[2].connected


# --- guards ---------------------------------------------------------------------------------


async def test_content_writes_need_an_owned_target(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    with pytest.raises(NotOwnedError):
        await gw.rename_channel(OUTPUT_ID, "New")
    with pytest.raises(NotOwnedError):
        await gw.add_bot_admin(OUTPUT_ID, "curator_bot")
    with pytest.raises(NotOwnedError):
        await gw.copy_media(CHANNEL_ID, [1], OUTPUT_ID)
    with pytest.raises(NotOwnedError):
        await gw.forward(CHANNEL_ID, [1], OUTPUT_ID)
    with pytest.raises(NotOwnedError):
        await gw.find_message(OUTPUT_ID, contains="x")
    assert stub.requests == [] and stub.calls == []


async def test_subscription_writes_refuse_owned_chats(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    with pytest.raises(NotOwnedError):
        await gw.mute(OUTPUT_ID, None)
    with pytest.raises(NotOwnedError):
        await gw.set_archived(OUTPUT_ID, True)
    with pytest.raises(NotOwnedError):
        await gw.leave(OUTPUT_ID)
    assert stub.requests == [] and stub.calls == []


async def test_save_folder_refuses_foreign_ids_and_bad_input(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    with pytest.raises(NotOwnedError):
        await gw.save_folder(7, "Curated", [CHANNEL_ID])
    with pytest.raises(ValueError):
        await gw.save_folder(None, "Curated", [])
    with pytest.raises(ValueError):
        await gw.save_folder(None, "A title that is too long", [CHANNEL_ID])
    assert stub.requests == []


# --- writes into owned chats ----------------------------------------------------------------


async def test_create_channel_is_a_private_broadcast(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    created = channel(OUTPUT_BARE, title="ML & AI", username=None, creator=True)
    stub.responses[functions.channels.CreateChannelRequest] = types.Updates(
        [], [], [created], NOW, 1
    )
    info = await gw.create_channel("ML & AI", about="topic")
    request = stub.requests_of(functions.channels.CreateChannelRequest)[0]
    assert (request.title, request.about, request.broadcast, request.megagroup) == (
        "ML & AI",
        "topic",
        True,
        False,
    )
    assert info.id == OUTPUT_ID and info.is_creator and info.kind == "channel"
    assert OUTPUT_ID not in gw._owned  # TopicsService registers it, not the gateway


async def test_create_channel_flood_is_raised(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.responses[functions.channels.CreateChannelRequest] = errors.FloodWaitError(
        request=None, capture=900
    )
    with pytest.raises(FloodWait) as info:
        await gw.create_channel("Topic")
    assert info.value.seconds == 900


async def test_add_bot_admin_grants_post_and_edit_only(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.responses[functions.channels.EditAdminRequest] = types.Updates([], [], [], NOW, 1)
    await gw.add_bot_admin(OUTPUT_ID, "@curator_bot")
    request = stub.requests_of(functions.channels.EditAdminRequest)[0]
    assert request.channel.channel_id == OUTPUT_BARE and request.user_id.user_id == 42
    rights = request.admin_rights.to_dict()
    granted = {k for k, v in rights.items() if v is True}
    assert granted == {"post_messages", "edit_messages"}


async def test_add_bot_admin_fresh_session_is_not_allowed(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.responses[functions.channels.EditAdminRequest] = _rpc(
        errors.FreshChangeAdminsForbiddenError
    )
    with pytest.raises(NotAllowed) as info:
        await gw.add_bot_admin(OUTPUT_ID, "curator_bot")
    assert info.value.reason == "fresh_session"


async def test_rename_channel(gw: TelethonUserGateway, stub: StubClient) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.responses[functions.channels.EditTitleRequest] = types.Updates([], [], [], NOW, 1)
    await gw.rename_channel(OUTPUT_ID, "Renamed")
    request = stub.requests_of(functions.channels.EditTitleRequest)[0]
    assert (request.channel.channel_id, request.title) == (OUTPUT_BARE, "Renamed")


async def test_copy_media_sends_by_reference_per_message(
    gw: TelethonUserGateway, stub: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tg_curator.telegram.user_client.COPY_ITEM_GAP_SECONDS", 0)
    gw.register_owned(OUTPUT_ID)
    stub.history[CHANNEL_ID] = [
        tl_message(1, media=photo_media()),
        tl_message(2, media=document_media(video=True)),
    ]
    ids = await gw.copy_media(CHANNEL_ID, [1, 2], OUTPUT_ID)
    assert ids == [101, 102]
    sends = [c for c in stub.calls if c[0] == "send_file"]
    assert [type(c[1]["file"]) for c in sends] == [
        types.MessageMediaPhoto,
        types.MessageMediaDocument,
    ]
    assert all(c[1]["entity"].channel_id == OUTPUT_BARE for c in sends)


async def test_copy_media_refetches_once_then_gives_up(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.history[CHANNEL_ID] = [tl_message(1, media=photo_media())]
    stub.send_file_errors = [_rpc(errors.FileReferenceExpiredError)]
    assert await gw.copy_media(CHANNEL_ID, [1], OUTPUT_ID) == [101]
    fetches = [c for c in stub.calls if c[0] == "get_messages"]
    assert len(fetches) == 2  # the initial fetch and the one re-fetch
    stub.send_file_errors = [
        _rpc(errors.FileReferenceExpiredError),
        _rpc(errors.FileReferenceExpiredError),
    ]
    with pytest.raises(MediaUnavailable):
        await gw.copy_media(CHANNEL_ID, [1], OUTPUT_ID)


async def test_copy_media_refuses_protected_and_uncopyable(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.history[CHANNEL_ID] = [
        tl_message(1, media=photo_media(), noforwards=True),
        tl_message(2, media=types.MessageMediaWebPage(types.WebPageEmpty(id=1))),
    ]
    with pytest.raises(ForwardsRestricted):
        await gw.copy_media(CHANNEL_ID, [1], OUTPUT_ID)
    with pytest.raises(MediaUnavailable):
        await gw.copy_media(CHANNEL_ID, [2], OUTPUT_ID)
    with pytest.raises(MediaUnavailable):
        await gw.copy_media(CHANNEL_ID, [3], OUTPUT_ID)  # gone
    stub.history[CHANNEL_ID] = [tl_message(4, media=photo_media())]
    stub.send_file_errors = [_rpc(errors.ChatForwardsRestrictedError)]
    with pytest.raises(ForwardsRestricted):
        await gw.copy_media(CHANNEL_ID, [4], OUTPUT_ID)
    assert not [c for c in stub.calls if c[0] == "forward_messages"]


async def test_forward_keeps_album_ids_together(gw: TelethonUserGateway, stub: StubClient) -> None:
    gw.register_owned(OUTPUT_ID)
    assert await gw.forward(CHANNEL_ID, [7, 8, 9], OUTPUT_ID) == [101, 102, 103]
    call = [c for c in stub.calls if c[0] == "forward_messages"][0][1]
    assert call["messages"] == [7, 8, 9]
    assert call["from_peer"].channel_id == CHANNEL_BARE and call["entity"].channel_id == OUTPUT_BARE


async def test_forward_of_a_deleted_source_raises_media_unavailable(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    stub.gone = {7, 8}
    with pytest.raises(MediaUnavailable):
        await gw.forward(CHANNEL_ID, [7, 8], OUTPUT_ID)
    assert len(await gw.forward(CHANNEL_ID, [8, 9], OUTPUT_ID)) == 1  # partly there: kept


# --- reading --------------------------------------------------------------------------------


async def test_list_chats_sweeps_both_folders_and_unions_by_id(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    later = NOW + timedelta(days=1)
    stub.dialogs[0] = [dialog(channel()), dialog(types.User(id=9)), dialog(basic_group())]
    stub.dialogs[1] = [
        dialog(channel(), archived=True, mute_until=later),
        dialog(channel(2, left=True)),
    ]
    chats = await gw.list_chats()
    assert [c[1]["folder"] for c in stub.calls if c[0] == "iter_dialogs"] == [0, 1]
    assert {c.id for c in chats} == {CHANNEL_ID, GROUP_ID}
    archived = next(c for c in chats if c.id == CHANNEL_ID)
    assert archived.archived is True and archived.muted_until == later


async def test_resolve_chat_by_id_username_and_link(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.entities["source_chan"] = channel()
    stub.responses[functions.messages.GetPeerDialogsRequest] = types.messages.PeerDialogs(
        dialogs=[
            types.Dialog(
                types.PeerChannel(CHANNEL_BARE),
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                types.PeerNotifySettings(),
                folder_id=1,
            )
        ],
        messages=[],
        chats=[],
        users=[],
        state=types.updates.State(1, 1, NOW, 1, 1),
    )
    info = await gw.resolve_chat("https://t.me/source_chan/15")
    assert info.id == CHANNEL_ID and info.archived is True
    assert (await gw.resolve_chat(CHANNEL_ID)).id == CHANNEL_ID
    assert (await gw.resolve_chat("@source_chan")).username == "source_chan"
    with pytest.raises(ChatGone):
        await gw.resolve_chat("@nobody")
    stub.entities[9] = types.User(id=9)
    with pytest.raises(ChatGone):
        await gw.resolve_chat(9)


async def test_resolve_invite_link_only_checks_and_never_joins(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.responses[functions.messages.CheckChatInviteRequest] = types.ChatInviteAlready(channel())
    info = await gw.resolve_chat("https://t.me/+AbC")
    assert info.id == CHANNEL_ID
    assert stub.requests_of(functions.messages.CheckChatInviteRequest)[0].hash == "AbC"
    stub.responses[functions.messages.CheckChatInviteRequest] = types.ChatInvite(
        title="Closed", photo=types.PhotoEmpty(1), participants_count=1, color=0
    )
    with pytest.raises(NotAllowed) as denied:
        await gw.resolve_chat("t.me/joinchat/XyZ")
    assert denied.value.reason == "not_a_member"
    stub.responses[functions.messages.CheckChatInviteRequest] = _rpc(errors.InviteHashExpiredError)
    with pytest.raises(ChatGone):
        await gw.resolve_chat("t.me/+expired")
    # only the check and the dialog-state read: nothing that joins
    allowed = (functions.messages.CheckChatInviteRequest, functions.messages.GetPeerDialogsRequest)
    assert all(isinstance(r, allowed) for r in stub.requests)


async def test_history_is_oldest_first_and_skips_service_messages(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    since = NOW - timedelta(days=1)
    stub.history[CHANNEL_ID] = [
        tl_message(3, text="three", date=NOW),
        tl_message(1, text="one", date=NOW - timedelta(hours=2)),
        types.MessageService(
            id=2,
            peer_id=types.PeerChannel(CHANNEL_BARE),
            date=NOW - timedelta(hours=1),
            action=types.MessageActionPinMessage(),
        ),
        tl_message(0, text="old", date=NOW - timedelta(days=2)),
    ]
    got = [m async for m in gw.history(CHANNEL_ID, since=since)]
    assert [m.message_id for m in got] == [1, 3]
    assert all(m.chat.id == CHANNEL_ID for m in got)
    call = [c for c in stub.calls if c[0] == "iter_messages"][0][1]
    assert (call["limit"], call["reverse"], call["wait_time"], call["offset_date"]) == (
        2000,
        True,
        1,
        since,
    )
    assert not [m async for m in gw.history(CHANNEL_ID, since=since, limit=0)]


async def test_get_views_reads_without_incrementing(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    def answer(request: Any) -> Any:
        assert request.increment is False
        views = [
            types.MessageViews(views=i * 10) if i != 2 else types.MessageViews() for i in request.id
        ]
        return types.messages.MessageViews(views=views, chats=[], users=[])

    stub.responses[functions.messages.GetMessagesViewsRequest] = answer
    assert await gw.get_views(CHANNEL_ID, [1, 2, 3]) == {1: 10, 3: 30}
    assert await gw.get_views(GROUP_ID, [1, 2]) == {}
    assert len(stub.requests_of(functions.messages.GetMessagesViewsRequest)) == 1


async def test_get_views_skips_invalid_ids(
    gw: TelethonUserGateway, stub: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def answer(request: Any) -> Any:
        if 2 in request.id:
            return errors.BadRequestError(request=None, message="MSG_ID_INVALID")
        return types.messages.MessageViews(
            views=[types.MessageViews(views=i) for i in request.id], chats=[], users=[]
        )

    stub.responses[functions.messages.GetMessagesViewsRequest] = answer
    assert await gw.get_views(CHANNEL_ID, [1, 2, 3, 4]) == {1: 1, 3: 3, 4: 4}
    assert sleeps and all(s == 1 for s in sleeps)


async def test_find_message_matches_text_entity_href_forward_and_since(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    gw.register_owned(OUTPUT_ID)
    link = "https://t.me/source_chan/15"
    fwd = types.MessageFwdHeader(date=NOW, from_id=types.PeerChannel(CHANNEL_BARE), channel_post=15)
    stub.history[OUTPUT_ID] = [
        tl_message(30, peer=types.PeerChannel(OUTPUT_BARE), text="Daily digest", date=NOW),
        tl_message(
            20,
            peer=types.PeerChannel(OUTPUT_BARE),
            text="Source · source",
            entities=[types.MessageEntityTextUrl(9, 6, link)],
            date=NOW - timedelta(minutes=5),
        ),
        tl_message(
            10,
            peer=types.PeerChannel(OUTPUT_BARE),
            text="fwd",
            fwd_from=fwd,
            date=NOW - timedelta(hours=2),
        ),
    ]
    assert await gw.find_message(OUTPUT_ID, contains=link) == 20
    assert await gw.find_message(OUTPUT_ID, contains="Daily digest") == 30
    assert await gw.find_message(OUTPUT_ID, fwd_of=(CHANNEL_ID, 15)) == 10
    assert (
        await gw.find_message(OUTPUT_ID, fwd_of=(CHANNEL_ID, 15), since=NOW - timedelta(hours=1))
        is None
    )
    assert await gw.find_message(OUTPUT_ID, contains="nothing") is None
    assert [c[1]["limit"] for c in stub.calls if c[0] == "iter_messages"] == [50] * 5


async def test_find_media_matches_the_staging_copies_by_media_id(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    def photo(pid: int) -> types.MessageMediaPhoto:
        return types.MessageMediaPhoto(
            photo=types.Photo(
                id=pid, access_hash=2, file_reference=b"ref", date=NOW, sizes=[], dc_id=2
            )
        )

    gw.register_owned(OUTPUT_ID)
    peer = types.PeerChannel(OUTPUT_BARE)
    fwd = types.MessageFwdHeader(date=NOW, from_id=types.PeerChannel(CHANNEL_BARE), channel_post=1)
    stub.history[CHANNEL_ID] = [  # the staging copies
        tl_message(51, text="", media=photo(501)),
        tl_message(52, text="", media=photo(502)),
    ]
    stub.history[OUTPUT_ID] = [  # newest first, as getHistory returns them
        tl_message(9, peer=peer, text="the text part", date=NOW),
        tl_message(8, peer=peer, text="", media=photo(502), date=NOW),
        tl_message(7, peer=peer, text="", media=photo(501), date=NOW),
        tl_message(6, peer=peer, text="", media=photo(999), date=NOW),
        tl_message(5, peer=peer, text="", media=photo(501), fwd_from=fwd, date=NOW),
        tl_message(4, peer=peer, text="↪ moved", media=photo(501), date=NOW),
        tl_message(3, peer=peer, text="", media=photo(501), date=NOW - timedelta(hours=2)),
    ]
    with pytest.raises(NotOwnedError):  # the staging channel must be owned too
        await gw.find_media(OUTPUT_ID, copies_of=(CHANNEL_ID, [51]))
    gw.register_owned(CHANNEL_ID)
    recent = NOW - timedelta(hours=1)
    assert await gw.find_media(OUTPUT_ID, copies_of=(CHANNEL_ID, [51, 52]), since=recent) == [7, 8]
    assert await gw.find_media(OUTPUT_ID, copies_of=(CHANNEL_ID, [51])) == [3]
    assert await gw.find_media(OUTPUT_ID, copies_of=(CHANNEL_ID, [51]), after_id=7) == []
    assert await gw.find_media(OUTPUT_ID, copies_of=(CHANNEL_ID, [52, 51])) == []


# --- subscriptions --------------------------------------------------------------------------


async def test_mute_uses_update_notify_settings(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.responses[functions.account.UpdateNotifySettingsRequest] = True
    until = NOW + timedelta(days=30)
    await gw.mute(CHANNEL_ID, until)
    await gw.mute(CHANNEL_ID, None)
    await gw.mute(CHANNEL_ID, datetime.max)
    requests = stub.requests_of(functions.account.UpdateNotifySettingsRequest)
    assert [r.settings.mute_until for r in requests] == [int(until.timestamp()), 0, MUTE_FOREVER]
    assert all(r.peer.peer.channel_id == CHANNEL_BARE for r in requests)


async def test_set_archived_uses_peer_folders(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.responses[functions.folders.EditPeerFoldersRequest] = types.Updates([], [], [], NOW, 1)
    await gw.set_archived(CHANNEL_ID, True)
    await gw.set_archived(CHANNEL_ID, False)
    requests = stub.requests_of(functions.folders.EditPeerFoldersRequest)
    assert [r.folder_peers[0].folder_id for r in requests] == [1, 0]


async def test_leave_uses_the_right_request_per_chat_type(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.responses[functions.channels.LeaveChannelRequest] = types.Updates([], [], [], NOW, 1)
    stub.responses[functions.messages.DeleteChatUserRequest] = types.Updates([], [], [], NOW, 1)
    await gw.leave(CHANNEL_ID)
    await gw.leave(GROUP_ID)
    leave = stub.requests_of(functions.channels.LeaveChannelRequest)[0]
    assert leave.channel.channel_id == CHANNEL_BARE
    delete = stub.requests_of(functions.messages.DeleteChatUserRequest)[0]
    assert delete.chat_id == GROUP_BARE and isinstance(delete.user_id, types.InputUserSelf)


async def test_leave_refuses_creators_and_tolerates_already_gone(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.entities[CHANNEL_ID] = channel(creator=True)
    with pytest.raises(NotAllowed) as info:
        await gw.leave(CHANNEL_ID)
    assert info.value.reason == "creator"
    assert stub.requests == []
    stub.entities[CHANNEL_ID] = channel()
    stub.responses[functions.channels.LeaveChannelRequest] = _rpc(errors.UserCreatorError)
    with pytest.raises(NotAllowed) as info:
        await gw.leave(CHANNEL_ID)
    assert info.value.reason == "creator"
    stub.responses[functions.channels.LeaveChannelRequest] = _rpc(errors.ChannelPrivateError)
    await gw.leave(CHANNEL_ID)  # already kicked: nothing to do


# --- folders --------------------------------------------------------------------------------


def _filter(folder_id: int, title: str, *peers: Any, chatlist: bool = False) -> Any:
    text = types.TextWithEntities(title, [])
    if chatlist:
        return types.DialogFilterChatlist(folder_id, text, [], list(peers))
    return types.DialogFilter(folder_id, text, [], list(peers), [], emoticon="📁", color=3)


async def test_list_and_get_folders(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [
            types.DialogFilterDefault(),
            _filter(2, "Curated", types.InputPeerChannel(OUTPUT_BARE, 1)),
            _filter(3, "Shared", chatlist=True),
        ]
    )
    assert await gw.list_folders() == [(2, "Curated"), (3, "Shared")]
    assert await gw.get_folder(2) == [OUTPUT_ID]
    assert await gw.get_folder(9) is None


async def test_save_folder_creates_with_a_free_id_and_keeps_user_flags(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_filter(2, "Mine"), _filter(3, "Shared", chatlist=True)]
    )
    stub.responses[functions.messages.UpdateDialogFilterRequest] = True
    assert await gw.save_folder(None, "Low signal", [CHANNEL_ID, GROUP_ID]) == 4
    request = stub.requests_of(functions.messages.UpdateDialogFilterRequest)[0]
    assert request.id == 4 and isinstance(request.filter, types.DialogFilter)
    assert (
        isinstance(request.filter.title, types.TextWithEntities)
        and request.filter.title.text == "Low signal"
    )
    assert [utils.get_peer_id(p) for p in request.filter.include_peers] == [CHANNEL_ID, GROUP_ID]
    # a folder that was merely listed stays the user's: listing never makes it writable
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_filter(2, "Mine", types.InputPeerChannel(OUTPUT_BARE, 1))]
    )
    assert await gw.list_folders() == [(2, "Mine")]
    assert await gw.get_folder(2) == [OUTPUT_ID]
    with pytest.raises(NotOwnedError):
        await gw.save_folder(2, "Curated", [OUTPUT_ID])
    with pytest.raises(NotOwnedError):
        await gw.delete_folder(2)
    assert len(stub.requests_of(functions.messages.UpdateDialogFilterRequest)) == 1
    # a registered id (one stored in kv folders.*) is updated and keeps emoticon and colour
    gw.register_own_folder(2)
    assert await gw.save_folder(2, "Curated", [OUTPUT_ID]) == 2
    updated = stub.requests_of(functions.messages.UpdateDialogFilterRequest)[1].filter
    assert (updated.id, updated.title.text, updated.emoticon, updated.color) == (
        2,
        "Curated",
        "📁",
        3,
    )


async def test_save_folder_limit(gw: TelethonUserGateway, stub: StubClient) -> None:
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters([])
    stub.responses[functions.messages.UpdateDialogFilterRequest] = errors.BadRequestError(
        request=None, message="DIALOG_FILTERS_TOO_MUCH"
    )
    with pytest.raises(FolderLimit):
        await gw.save_folder(None, "Curated", [CHANNEL_ID])


async def test_delete_folder_sends_an_update_without_filter(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.responses[functions.messages.GetDialogFiltersRequest] = types.messages.DialogFilters(
        [_filter(4, "Low signal", types.InputPeerChannel(CHANNEL_BARE, 1))]
    )
    stub.responses[functions.messages.UpdateDialogFilterRequest] = True
    gw.register_own_folder(4)
    await gw.delete_folder(4)
    [request] = stub.requests_of(functions.messages.UpdateDialogFilterRequest)
    assert request.id == 4 and request.filter is None
    # the id is forgotten: a second delete (or a save) of it is refused until registered again
    with pytest.raises(NotOwnedError):
        await gw.delete_folder(4)


async def test_delete_folder_refuses_ids_the_gateway_never_returned(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    with pytest.raises(NotOwnedError):
        await gw.delete_folder(7)
    assert stub.requests_of(functions.messages.UpdateDialogFilterRequest) == []


# --- outages (§17.3) --------------------------------------------------------------------------


async def test_outages_reach_the_caller_as_telegram_unavailable(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    await gw.connect()
    stub.responses[functions.channels.CreateChannelRequest] = errors.ServerError(
        request=None, message="INTERNAL"
    )
    with pytest.raises(TelegramUnavailable):
        await gw.create_channel("Topic")
    stub.responses[functions.messages.GetDialogFiltersRequest] = ConnectionError("down")
    with pytest.raises(TelegramUnavailable):
        await gw.list_folders()


async def test_resolve_chat_does_not_mistake_an_outage_for_no_dialog(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.responses[functions.messages.GetPeerDialogsRequest] = TimeoutError()
    with pytest.raises(TelegramUnavailable):
        await gw.resolve_chat(CHANNEL_ID)


async def test_login_steps_keep_an_outage_an_outage(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    stub.send_code_error = ConnectionError("down")
    with pytest.raises(TelegramUnavailable):
        await gw.send_code("+998901234512")


async def test_ping_is_false_while_telegram_is_unreachable(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    await gw.connect()
    stub.responses[functions.users.GetUsersRequest] = OSError("Network is unreachable")
    assert await gw.ping() is False
    stub.responses[functions.users.GetUsersRequest] = [stub.me]
    assert await gw.ping() is True


# --- live delivery ----------------------------------------------------------------------------


async def test_delivery_converts_dedupes_and_flags_outgoing(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    seen: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        seen.append(msg)

    gw.on_message(handler)
    msg = tl_message(5, text="post", grouped_id=1)
    msg._chat = channel()
    await gw._on_new_message(SimpleNamespace(message=msg))
    await gw._on_album(SimpleNamespace(messages=[msg]))  # the Album event repeats the item
    assert [m.message_id for m in seen] == [5] and seen[0].grouped_id == 1
    assert seen[0].chat.id == CHANNEL_ID
    own = tl_message(6, text="mine", out=True)
    own._chat = channel()
    await gw._on_new_message(SimpleNamespace(message=own))
    await gw._on_album(SimpleNamespace(messages=[own]))
    # delivered once, flagged: intake drops it in a group and keeps it in a channel
    assert [m.message_id for m in seen] == [5, 6] and seen[1].is_outgoing
    private = tl_message(7, peer=types.PeerUser(9))
    private._chat = types.User(id=9)
    await gw._on_new_message(SimpleNamespace(message=private))
    assert len(seen) == 2


async def test_delivery_survives_a_failing_handler(gw: TelethonUserGateway) -> None:
    calls: list[int] = []

    async def bad(msg: IncomingMessage) -> None:
        raise RuntimeError("boom")

    async def good(msg: IncomingMessage) -> None:
        calls.append(msg.message_id)

    gw.on_message(bad)
    gw.on_message(good)
    msg = tl_message(8)
    msg._chat = channel()
    await gw._on_new_message(SimpleNamespace(message=msg))
    assert calls == [8]


def test_gateway_registers_new_message_and_album_handlers(
    gw: TelethonUserGateway, stub: StubClient
) -> None:
    kinds = [type(event).__name__ for _, event in stub.handlers]
    assert kinds == ["NewMessage", "Album"]
    new_message = stub.handlers[0][1]
    assert new_message.incoming is True
