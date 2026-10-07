"""The Telethon bot gateway without a network (DESIGN §5 BotGateway, §16).

The conversions are exercised with TL objects built here; the gateway methods run against a
stub client that records every Telethon call and raises what a test tells it to, so the
composition of ``send_copy``, the error translation, the owner-only delete guard and the
``MESSAGE_NOT_MODIFIED`` swallow are all checked without Telegram.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import errors as tl_errors
from telethon import functions, types

from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    ConfigError,
    CuratorError,
    FloodWait,
    ForwardsRestricted,
    MediaUnavailable,
    NotAllowed,
    TelegramUnavailable,
)
from tg_curator.telegram import bot_client
from tg_curator.telegram.bot_client import (
    CALLBACK_DATA_MAX_BYTES,
    TelethonBotGateway,
    can_post_from,
    input_media_of,
    to_bot_callback,
    to_bot_message,
    to_keyboard,
    translate_error,
)
from tg_curator.telegram.gateway import BotCallback, BotGateway, BotMessage, Button

OWNER = 1001
CHANNEL = -1_001_000_000_777
STAGING = -1_001_000_000_555
NOW = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)


class _Request:
    """A Telethon request as the stub sees it: the method name, its args and kwargs."""

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name, self.args, self.kwargs = name, args, kwargs


def _message(mid: int, media: Any = None) -> types.Message:
    return types.Message(
        id=mid, peer_id=types.PeerChannel(1_000_000_555), date=NOW, message="", media=media
    )


def _photo(mid: int, ref: bytes = b"ref") -> types.MessageMediaPhoto:
    return types.MessageMediaPhoto(
        photo=types.Photo(id=mid, access_hash=99, file_reference=ref, date=NOW, sizes=[], dc_id=2)
    )


def _document(mid: int) -> types.MessageMediaDocument:
    return types.MessageMediaDocument(
        document=types.Document(
            id=mid,
            access_hash=77,
            file_reference=b"doc",
            date=NOW,
            mime_type="video/mp4",
            size=10,
            dc_id=2,
            attributes=[],
        )
    )


class StubClient:
    """Records every call the gateway makes and fails the ones a test queues a failure for."""

    def __init__(self) -> None:
        self.calls: list[_Request] = []
        self.failures: dict[str, list[BaseException]] = {}
        self.authorised = True
        self.me = types.User(
            id=7_000_000_001, first_name="curator bot", username="curator_test_bot"
        )
        self.staged: dict[int, types.Message | None] = {}
        self.channels: dict[int, Any] = {}
        self.next_id = 100
        self.handlers: list[tuple[Any, Any]] = []
        self.parse_mode: str | None = None
        self.connected = False

    def fail_next(self, method: str, exc: BaseException) -> None:
        self.failures.setdefault(method, []).append(exc)

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_Request(name, *args, **kwargs))
        queue = self.failures.get(name)
        if queue:
            raise queue.pop(0)

    def calls_of(self, name: str) -> list[_Request]:
        return [c for c in self.calls if c.name == name]

    def _sent(self) -> types.Message:
        self.next_id += 1
        return _message(self.next_id)

    # --- the Telethon surface the gateway uses ---

    async def connect(self) -> None:
        self._record("connect")
        self.connected = True

    async def is_user_authorized(self) -> bool:
        self._record("is_user_authorized")
        return self.authorised

    async def sign_in(self, **kwargs: Any) -> types.User:
        self._record("sign_in", **kwargs)
        self.authorised = True
        return self.me

    async def get_me(self) -> types.User:
        self._record("get_me")
        return self.me

    def add_event_handler(self, callback: Any, event: Any) -> None:
        self.handlers.append((callback, event))

    async def disconnect(self) -> None:
        self._record("disconnect")
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def send_message(self, entity: Any, message: str, **kwargs: Any) -> types.Message:
        self._record("send_message", entity, message, **kwargs)
        return self._sent()

    async def get_messages(self, entity: Any, *, ids: list[int]) -> list[types.Message | None]:
        self._record("get_messages", entity, ids=ids)
        return [self.staged.get(i) for i in ids]

    async def send_file(self, entity: Any, file: Any, **kwargs: Any) -> Any:
        self._record("send_file", entity, file, **kwargs)
        if isinstance(file, list):
            return [self._sent() for _ in file]
        return self._sent()

    async def edit_message(self, entity: Any, message: int, text: Any, **kwargs: Any) -> Any:
        self._record("edit_message", entity, message, text, **kwargs)
        return self._sent()

    async def delete_messages(self, entity: Any, message_ids: list[int], **kwargs: Any) -> None:
        self._record("delete_messages", entity, message_ids, **kwargs)

    async def __call__(self, request: Any) -> Any:
        self._record("call", request)
        if isinstance(request, functions.channels.GetChannelsRequest):
            chat = self.channels[request.id[0]]
            if isinstance(chat, BaseException):
                raise chat
            return types.messages.Chats(chats=[chat])
        return None


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


@pytest.fixture
def gateway(home: Path, stub: StubClient) -> TelethonBotGateway:
    return TelethonBotGateway(
        home, api_id=1, api_hash="h", bot_token="1:token", owner_id=OWNER, client=stub
    )


@pytest.fixture
async def started(gateway: TelethonBotGateway) -> TelethonBotGateway:
    await gateway.start()
    return gateway


# --- conversions -----------------------------------------------------------------------------


def test_gateway_satisfies_the_protocol(gateway: TelethonBotGateway) -> None:
    assert isinstance(gateway, BotGateway)


def test_keyboard_rows_become_inline_buttons() -> None:
    rows = to_keyboard(
        [[Button("Wrong topic", data="wt:42")], [Button("Open", url="https://t.me/x/1")]]
    )
    assert rows is not None and len(rows) == 2
    first, second = rows[0][0], rows[1][0]
    assert isinstance(first, types.KeyboardInlineButton)
    assert isinstance(first.type, types.InlineButtonTypeCallback) and first.type.data == b"wt:42"
    assert first.text == "Wrong topic"
    assert isinstance(second.type, types.InlineButtonTypeUrl)
    assert second.type.url == "https://t.me/x/1"


def test_keyboard_none_for_nothing_to_show() -> None:
    assert to_keyboard(None) is None
    assert to_keyboard([]) is None
    assert to_keyboard([[]]) is None


def test_keyboard_enforces_the_64_byte_limit() -> None:
    assert to_keyboard([[Button("ok", data="x" * CALLBACK_DATA_MAX_BYTES)]]) is not None
    with pytest.raises(ValueError, match="65 bytes"):
        to_keyboard([[Button("too long", data="x" * (CALLBACK_DATA_MAX_BYTES + 1))]])
    # bytes, not characters: 22 Cyrillic letters are 44 bytes, 33 are 66
    assert to_keyboard([[Button("ok", data="я" * 22)]]) is not None
    with pytest.raises(ValueError):
        to_keyboard([[Button("ok", data="я" * 33)]])


def test_keyboard_rejects_a_button_without_payload() -> None:
    with pytest.raises(ValueError, match="neither data nor url"):
        to_keyboard([[Button("empty")]])


def test_private_message_converts() -> None:
    msg = types.Message(
        id=5,
        peer_id=types.PeerUser(OWNER),
        from_id=types.PeerUser(OWNER),
        date=NOW,
        message="/start abc",
        reply_to=types.MessageReplyHeader(reply_to_msg_id=4),
    )
    got = to_bot_message(msg)
    assert got == BotMessage(
        chat_id=OWNER,
        message_id=5,
        sender_id=OWNER,
        text="/start abc",
        is_private=True,
        fwd_from_chat_id=None,
        fwd_from_title=None,
        reply_to_id=4,
    )


def test_forwarded_channel_post_carries_source_id_and_title() -> None:
    source = types.Channel(id=1_000_000_123, title="Kun.uz", photo=types.ChatPhotoEmpty(), date=NOW)
    msg = types.Message(
        id=6,
        peer_id=types.PeerUser(OWNER),
        from_id=types.PeerUser(OWNER),
        date=NOW,
        message="example post",
        fwd_from=types.MessageFwdHeader(
            date=NOW, from_id=types.PeerChannel(1_000_000_123), channel_post=900
        ),
    )
    got = to_bot_message(msg, {-1_001_000_000_123: source})
    assert got.fwd_from_chat_id == -1_001_000_000_123
    assert got.fwd_from_title == "Kun.uz"
    assert got.is_private and got.text == "example post"
    # without the entity the id is still known, the title is not
    assert to_bot_message(msg).fwd_from_title is None
    assert to_bot_message(msg).fwd_from_chat_id == -1_001_000_000_123


def test_a_media_post_without_caption_is_marked_as_media_not_text() -> None:
    photo = types.Message(
        id=8,
        peer_id=types.PeerUser(OWNER),
        date=NOW,
        message="",
        media=types.MessageMediaPhoto(photo=types.PhotoEmpty(id=1)),
    )
    got = to_bot_message(photo)
    assert got.has_media and got.text == ""
    preview = types.Message(
        id=9,
        peer_id=types.PeerUser(OWNER),
        date=NOW,
        message="https://example.org",
        media=types.MessageMediaWebPage(webpage=types.WebPageEmpty(id=1)),
    )
    assert not to_bot_message(preview).has_media  # a link preview is not media


def test_forward_from_hidden_sender_keeps_only_the_name() -> None:
    msg = types.Message(
        id=7,
        peer_id=types.PeerUser(OWNER),
        date=NOW,
        message="hi",
        fwd_from=types.MessageFwdHeader(date=NOW, from_name="Someone"),
    )
    got = to_bot_message(msg)
    assert got.fwd_from_chat_id is None and got.fwd_from_title == "Someone"
    assert got.sender_id == OWNER  # incoming private messages have no from_id since layer 119


def test_channel_post_is_not_private_and_the_channel_is_the_sender() -> None:
    msg = types.Message(
        id=8, peer_id=types.PeerChannel(1_000_000_555), date=NOW, message="", post=True
    )
    got = to_bot_message(msg)
    assert got.chat_id == STAGING and got.sender_id == STAGING
    assert not got.is_private and got.text == "" and got.reply_to_id is None


def test_callback_converts_with_decoded_data() -> None:
    update = types.UpdateBotCallbackQuery(
        query_id=123456789012345678,
        user_id=OWNER,
        peer=types.PeerChannel(1_000_000_777),
        msg_id=42,
        chat_instance=1,
        data=b"wt:42",
    )
    assert to_bot_callback(update) == BotCallback(
        query_id="123456789012345678", sender_id=OWNER, chat_id=CHANNEL, message_id=42, data="wt:42"
    )
    update.data = None
    assert to_bot_callback(update).data == ""


def test_input_media_of_photo_and_document_by_reference() -> None:
    photo = input_media_of(_message(1, _photo(11, b"abc")))
    assert isinstance(photo, types.InputMediaPhoto)
    assert photo.id.id == 11 and photo.id.file_reference == b"abc"
    doc = input_media_of(_message(2, _document(22)))
    assert isinstance(doc, types.InputMediaDocument) and doc.id.id == 22


@pytest.mark.parametrize(
    "media",
    [
        None,
        types.MessageMediaEmpty(),
        types.MessageMediaUnsupported(),
        types.MessageMediaWebPage(webpage=types.WebPageEmpty(id=1)),
    ],
)
def test_input_media_of_refuses_what_cannot_be_resent(media: Any) -> None:
    with pytest.raises(MediaUnavailable):
        input_media_of(_message(1, media))


def test_input_media_of_missing_message() -> None:
    with pytest.raises(MediaUnavailable):
        input_media_of(None)


def _channel(**kwargs: Any) -> types.Channel:
    return types.Channel(
        id=1_000_000_777, title="ML & AI", photo=types.ChatPhotoEmpty(), date=NOW, **kwargs
    )


def test_can_post_from_reads_admin_rights_and_left() -> None:
    assert can_post_from(_channel(admin_rights=types.ChatAdminRights(post_messages=True)))
    assert not can_post_from(_channel(admin_rights=types.ChatAdminRights(edit_messages=True)))
    assert not can_post_from(_channel(admin_rights=None))
    assert not can_post_from(
        _channel(admin_rights=types.ChatAdminRights(post_messages=True), left=True)
    )
    assert not can_post_from(types.ChannelForbidden(id=1, access_hash=2, title="gone"))
    assert not can_post_from(None)


# --- error translation -----------------------------------------------------------------------


def _rpc(cls: type[Exception], **kwargs: Any) -> Exception:
    return cls(request=None, **kwargs)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_rpc(tl_errors.ChatWriteForbiddenError), BotCannotPost),
        (_rpc(tl_errors.ChatAdminRequiredError), BotCannotPost),
        (_rpc(tl_errors.UserBannedInChannelError), BotCannotPost),
        (_rpc(tl_errors.ChannelPrivateError), ChatGone),
        (_rpc(tl_errors.ChannelInvalidError), ChatGone),
        (_rpc(tl_errors.PeerIdInvalidError), ChatGone),
        (_rpc(tl_errors.UserIsBlockedError), ChatGone),
        (_rpc(tl_errors.ChatForwardsRestrictedError), ForwardsRestricted),
        (_rpc(tl_errors.FileReferenceExpiredError), MediaUnavailable),
        (_rpc(tl_errors.MediaEmptyError), MediaUnavailable),
        (_rpc(tl_errors.MessageDeleteForbiddenError), NotAllowed),
        (_rpc(tl_errors.AccessTokenInvalidError), ConfigError),
        (_rpc(tl_errors.AuthKeyUnregisteredError), ConfigError),
        (_rpc(tl_errors.MessageTooLongError), CuratorError),
        (ValueError("Could not find the input entity for PeerChannel(...)"), ChatGone),
        (ConnectionError("Cannot send requests while disconnected"), TelegramUnavailable),
        (TimeoutError(), TelegramUnavailable),
        (OSError("Network is unreachable"), TelegramUnavailable),
        (_rpc(tl_errors.ServerError, message="INTERNAL"), TelegramUnavailable),
        (_rpc(tl_errors.HistoryGetFailedError), TelegramUnavailable),
        (_rpc(tl_errors.RPCError, message="No workers running", code=-500), TelegramUnavailable),
        (_rpc(tl_errors.TimedOutError, message="Timeout", code=-503), TelegramUnavailable),
        # the transport's 404 (Telegram forgot the session key): the token is still good
        (tl_errors.AuthKeyNotFound(), TelegramUnavailable),
    ],
)
def test_error_translation_table(exc: Exception, expected: type[Exception]) -> None:
    translated = translate_error(exc)
    assert type(translated) is expected


def test_flood_wait_keeps_the_seconds() -> None:
    translated = translate_error(tl_errors.FloodWaitError(request=None, capture=300))
    assert isinstance(translated, FloodWait) and translated.seconds == 300
    slow = translate_error(tl_errors.SlowModeWaitError(request=None, capture=7))
    assert isinstance(slow, FloodWait) and slow.seconds == 7


def test_our_own_errors_and_unrelated_exceptions_pass_through() -> None:
    mine = ChatGone("already ours")
    assert translate_error(mine) is mine
    assert translate_error(ValueError("Failed to parse message")) is None
    assert translate_error(KeyError("x")) is None


def test_real_client_flags(tmp_path: Path) -> None:
    """The client the gateway builds sleeps short floods, replays nothing, and lets an outage
    that outlived Telethon's retries surface as the server error (-> TelegramUnavailable)."""
    gateway = TelethonBotGateway(
        tmp_path, api_id=1, api_hash="hash", bot_token="1:token", owner_id=OWNER
    )
    client = gateway._build_client()
    assert client.flood_sleep_threshold == 120
    assert client._catch_up is False
    assert client._raise_last_call_error is True
    assert client._init_request.device_model == "tg-curator"


async def test_an_outage_reaches_the_caller(gateway: TelethonBotGateway, stub: StubClient) -> None:
    await gateway.start()
    stub.fail_next("send_message", ConnectionError("Cannot send requests while disconnected"))
    with pytest.raises(TelegramUnavailable):
        await gateway.send_text(OWNER, "hi")


# --- start / stop ----------------------------------------------------------------------------


async def test_start_signs_in_only_when_the_session_is_fresh(
    gateway: TelethonBotGateway, stub: StubClient
) -> None:
    stub.authorised = False
    account = await gateway.start()
    assert [c.name for c in stub.calls] == ["connect", "is_user_authorized", "sign_in", "get_me"]
    assert stub.calls_of("sign_in")[0].kwargs == {"bot_token": "1:token"}
    assert account.id == 7_000_000_001 and account.username == "curator_test_bot"
    assert account.name == "curator bot" and account.phone is None
    assert len(stub.handlers) == 2
    await gateway.stop()
    assert stub.calls[-1].name == "disconnect"


async def test_start_reuses_an_authorised_session(
    gateway: TelethonBotGateway, stub: StubClient
) -> None:
    await gateway.start()
    assert not stub.calls_of("sign_in")
    await gateway.start()
    assert len(stub.handlers) == 2, "handlers are registered once"


async def test_bad_token_is_a_config_error(gateway: TelethonBotGateway, stub: StubClient) -> None:
    stub.authorised = False
    stub.fail_next("sign_in", _rpc(tl_errors.AccessTokenInvalidError))
    with pytest.raises(ConfigError, match="telegram.bot_token"):
        await gateway.start()


async def test_methods_before_start_fail_plainly(gateway: TelethonBotGateway) -> None:
    with pytest.raises(CuratorError, match="not started"):
        await gateway.send_text(OWNER, "hi")
    await gateway.stop()  # a never-started gateway stops silently


# --- update dispatch -------------------------------------------------------------------------


async def test_updates_reach_every_handler_even_when_one_fails(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    seen: list[Any] = []

    async def bad(item: Any) -> None:
        raise RuntimeError("boom")

    async def good(item: Any) -> None:
        seen.append(item)

    started.on_message(bad)
    started.on_message(good)
    started.on_callback(bad)
    started.on_callback(good)
    new_message, callback = (handler for handler, _ in stub.handlers)
    msg = types.Message(
        id=1,
        peer_id=types.PeerUser(OWNER),
        from_id=types.PeerUser(OWNER),
        date=NOW,
        message="/status",
    )
    await new_message(SimpleNamespace(message=msg))
    query = types.UpdateBotCallbackQuery(
        query_id=1,
        user_id=OWNER,
        peer=types.PeerUser(OWNER),
        msg_id=9,
        chat_instance=0,
        data=b"ct:pause",
    )
    await callback(SimpleNamespace(query=query))
    await callback(
        SimpleNamespace(
            query=types.UpdateInlineBotCallbackQuery(
                query_id=2,
                user_id=OWNER,
                msg_id=types.InputBotInlineMessageID(dc_id=2, id=1, access_hash=1),
                chat_instance=0,
            )
        )
    )
    assert [type(i) for i in seen] == [BotMessage, BotCallback]
    assert seen[0].text == "/status" and seen[1].data == "ct:pause"


# --- send_text ---------------------------------------------------------------------------------


async def test_send_text_uses_html_no_preview_and_buttons(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    mid = await started.send_text(
        CHANNEL,
        "<b>hi</b>",
        buttons=[[Button("Wrong topic", data="wt:1")]],
        reply_to=3,
        silent=True,
    )
    call = stub.calls_of("send_message")[0]
    assert call.args == (CHANNEL, "<b>hi</b>")
    assert call.kwargs["parse_mode"] == "html"
    assert call.kwargs["link_preview"] is False
    assert call.kwargs["reply_to"] == 3 and call.kwargs["silent"] is True
    assert call.kwargs["buttons"][0][0].type.data == b"wt:1"
    assert mid == 101
    await started.send_text(OWNER, "plain")
    assert stub.calls_of("send_message")[1].kwargs["buttons"] is None


async def test_send_text_translates_rights_and_flood(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.fail_next("send_message", _rpc(tl_errors.ChatWriteForbiddenError))
    with pytest.raises(BotCannotPost):
        await started.send_text(CHANNEL, "x")
    stub.fail_next("send_message", tl_errors.FloodWaitError(request=None, capture=900))
    with pytest.raises(FloodWait) as info:
        await started.send_text(CHANNEL, "x")
    assert info.value.seconds == 900
    stub.fail_next("send_message", _rpc(tl_errors.ChannelPrivateError))
    with pytest.raises(ChatGone):
        await started.send_text(CHANNEL, "x")


async def test_send_text_checks_callback_data_before_any_request(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    with pytest.raises(ValueError):
        await started.send_text(CHANNEL, "x", buttons=[[Button("b", data="d" * 65)]])
    assert not stub.calls_of("send_message")


# --- send_copy ---------------------------------------------------------------------------------


async def test_send_copy_single_media_with_caption_and_buttons(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.staged[10] = _message(10, _photo(11, b"r1"))
    ids = await started.send_copy(
        STAGING,
        [10],
        CHANNEL,
        caption_html="<b>src</b>",
        buttons=[[Button("Wrong topic", data="wt:5")]],
    )
    fetch = stub.calls_of("get_messages")[0]
    assert fetch.args == (STAGING,) and fetch.kwargs == {"ids": [10]}
    send = stub.calls_of("send_file")[0]
    assert send.args[0] == CHANNEL
    assert (
        isinstance(send.args[1], types.InputMediaPhoto) and send.args[1].id.file_reference == b"r1"
    )
    assert send.kwargs["caption"] == "<b>src</b>" and send.kwargs["parse_mode"] == "html"
    assert send.kwargs["buttons"][0][0].type.data == b"wt:5"
    assert ids == [101]


async def test_send_copy_album_is_one_multimedia_send_without_buttons_or_caption(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.staged[10] = _message(10, _photo(11))
    stub.staged[11] = _message(11, _document(22))
    ids = await started.send_copy(STAGING, [10, 11], CHANNEL, caption_html="ignored for albums")
    send = stub.calls_of("send_file")[0]
    assert isinstance(send.args[1], list) and len(send.args[1]) == 2
    assert isinstance(send.args[1][0], types.InputMediaPhoto)
    assert isinstance(send.args[1][1], types.InputMediaDocument)
    assert send.kwargs["caption"] == "" and "buttons" not in send.kwargs
    assert ids == [101, 102]


async def test_send_copy_refuses_buttons_on_albums_before_fetching(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    with pytest.raises(ValueError, match="albums cannot carry buttons"):
        await started.send_copy(STAGING, [1, 2], CHANNEL, buttons=[[Button("b", data="x")]])
    assert await started.send_copy(STAGING, [], CHANNEL) == []
    assert not stub.calls_of("get_messages") and not stub.calls_of("send_file")


async def test_send_copy_missing_or_unsendable_staged_message(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    with pytest.raises(MediaUnavailable):
        await started.send_copy(STAGING, [404], CHANNEL)
    stub.staged[12] = _message(12, types.MessageMediaWebPage(webpage=types.WebPageEmpty(id=1)))
    with pytest.raises(MediaUnavailable):
        await started.send_copy(STAGING, [12], CHANNEL)
    assert not stub.calls_of("send_file")


async def test_send_copy_refetches_once_on_an_expired_file_reference(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.staged[10] = _message(10, _photo(11, b"old"))
    stub.fail_next("send_file", _rpc(tl_errors.FileReferenceExpiredError))

    async def refreshed(entity: Any, *, ids: list[int]) -> list[types.Message | None]:
        stub._record("get_messages", entity, ids=ids)
        return [_message(10, _photo(11, b"new"))]

    stub.get_messages = refreshed  # type: ignore[method-assign]
    ids = await started.send_copy(STAGING, [10], CHANNEL)
    assert ids == [101]
    assert len(stub.calls_of("get_messages")) == 2
    sends = stub.calls_of("send_file")
    assert [s.args[1].id.file_reference for s in sends] == [b"new", b"new"]


async def test_send_copy_gives_up_after_the_second_expiry(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.staged[10] = _message(10, _photo(11))
    stub.fail_next("send_file", _rpc(tl_errors.FileReferenceExpiredError))
    stub.fail_next("send_file", _rpc(tl_errors.FileReferenceExpiredError))
    with pytest.raises(MediaUnavailable):
        await started.send_copy(STAGING, [10], CHANNEL)
    assert len(stub.calls_of("send_file")) == 2


async def test_send_copy_translates_target_errors(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.staged[10] = _message(10, _photo(11))
    stub.fail_next("send_file", _rpc(tl_errors.ChatAdminRequiredError))
    with pytest.raises(BotCannotPost):
        await started.send_copy(STAGING, [10], CHANNEL)
    stub.fail_next("send_file", _rpc(tl_errors.ChatForwardsRestrictedError))
    with pytest.raises(ForwardsRestricted):
        await started.send_copy(STAGING, [10], CHANNEL)
    stub.fail_next("get_messages", _rpc(tl_errors.ChannelPrivateError))
    with pytest.raises(ChatGone):
        await started.send_copy(STAGING, [10], CHANNEL)


# --- edits -------------------------------------------------------------------------------------


async def test_edit_text_sends_caption_or_text_with_keyboard(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    await started.edit_text(
        CHANNEL, 5, "<b>+3 more</b>", buttons=[[Button("Wrong topic", data="wt:1")]]
    )
    call = stub.calls_of("edit_message")[0]
    assert call.args == (CHANNEL, 5, "<b>+3 more</b>")
    assert call.kwargs["parse_mode"] == "html" and call.kwargs["link_preview"] is False
    assert call.kwargs["buttons"][0][0].type.data == b"wt:1"
    await started.edit_text(CHANNEL, 5, "stub")
    assert stub.calls_of("edit_message")[1].kwargs["buttons"] is None


async def test_edit_buttons_edits_only_the_markup(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    await started.edit_buttons(CHANNEL, 5, [[Button("Approve", data="rv:a:1")]])
    call = stub.calls_of("edit_message")[0]
    assert call.args == (CHANNEL, 5, None) and call.kwargs["parse_mode"] is None
    assert call.kwargs["buttons"][0][0].text == "Approve"
    await started.edit_buttons(CHANNEL, 5, None)
    assert stub.calls_of("edit_message")[1].kwargs["buttons"] is None


async def test_edits_swallow_message_not_modified(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.fail_next("edit_message", _rpc(tl_errors.MessageNotModifiedError))
    await started.edit_text(CHANNEL, 5, "same")
    stub.fail_next("edit_message", _rpc(tl_errors.MessageNotModifiedError))
    await started.edit_buttons(CHANNEL, 5, None)
    stub.fail_next("edit_message", _rpc(tl_errors.MessageAuthorRequiredError))
    with pytest.raises(NotAllowed):
        await started.edit_text(CHANNEL, 5, "not mine")


# --- delete guard ----------------------------------------------------------------------------


async def test_delete_only_in_the_owners_private_chat(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    await started.delete_message(OWNER, 7)
    call = stub.calls_of("delete_messages")[0]
    assert call.args == (OWNER, [7]) and call.kwargs == {"revoke": True}
    for chat_id in (CHANNEL, STAGING, OWNER + 1, 0):
        with pytest.raises(NotAllowed):
            await started.delete_message(chat_id, 7)
    assert len(stub.calls_of("delete_messages")) == 1


async def test_delete_guard_reads_a_callable_owner_at_use_time(
    home: Path, stub: StubClient
) -> None:
    owner = {"id": 0}
    gw = TelethonBotGateway(
        home, api_id=1, api_hash="h", bot_token="t", owner_id=lambda: owner["id"], client=stub
    )
    await gw.start()
    with pytest.raises(NotAllowed):
        await gw.delete_message(OWNER, 1)
    owner["id"] = OWNER  # the owner claimed the bot after start
    await gw.delete_message(OWNER, 1)
    assert len(stub.calls_of("delete_messages")) == 1


async def test_delete_forbidden_is_not_allowed(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.fail_next("delete_messages", _rpc(tl_errors.MessageDeleteForbiddenError))
    with pytest.raises(NotAllowed):
        await started.delete_message(OWNER, 7)


# --- answer_callback and can_post ------------------------------------------------------------


async def test_answer_callback_builds_the_request_and_ignores_stale_queries(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    await started.answer_callback("123", "Moved", alert=True)
    request = stub.calls_of("call")[0].args[0]
    assert isinstance(request, functions.messages.SetBotCallbackAnswerRequest)
    assert request.query_id == 123 and request.message == "Moved" and request.alert is True
    assert request.cache_time == 0
    stub.fail_next("call", _rpc(tl_errors.QueryIdInvalidError))
    await started.answer_callback("124")


async def test_can_post_reads_the_bots_rights(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.channels[CHANNEL] = _channel(admin_rights=types.ChatAdminRights(post_messages=True))
    assert await started.can_post(CHANNEL)
    request = stub.calls_of("call")[0].args[0]
    assert isinstance(request, functions.channels.GetChannelsRequest) and request.id == [CHANNEL]
    stub.channels[CHANNEL] = _channel(admin_rights=types.ChatAdminRights(edit_messages=True))
    assert not await started.can_post(CHANNEL)
    stub.channels[CHANNEL] = types.ChannelForbidden(id=1_000_000_777, access_hash=1, title="x")
    assert not await started.can_post(CHANNEL)


async def test_can_post_is_false_before_the_bot_is_added_and_gone_when_invalid(
    started: TelethonBotGateway, stub: StubClient
) -> None:
    stub.channels[CHANNEL] = _rpc(tl_errors.ChannelPrivateError)
    assert not await started.can_post(CHANNEL)
    stub.channels[CHANNEL] = ValueError("Could not find the input entity for PeerChannel(1)")
    assert not await started.can_post(CHANNEL)
    stub.channels[CHANNEL] = _rpc(tl_errors.ChannelInvalidError)
    with pytest.raises(ChatGone):
        await started.can_post(CHANNEL)
    stub.channels[CHANNEL] = ValueError("Failed to parse message")
    with pytest.raises(ValueError):
        await started.can_post(CHANNEL)


# --- hygiene -----------------------------------------------------------------------------------


async def test_the_token_is_never_logged(
    gateway: TelethonBotGateway, stub: StubClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    stub.authorised = False
    stub.fail_next("send_message", _rpc(tl_errors.ChatWriteForbiddenError))
    await gateway.start()
    with pytest.raises(BotCannotPost):
        await gateway.send_text(CHANNEL, "x")
    await gateway.stop()
    assert "1:token" not in caplog.text


def test_the_bot_never_forwards() -> None:
    """§5: the bot copies media from the staging channel and never forwards anything;
    forwarding is the account's job (``UserGateway.forward``) and would leak the source."""
    source = Path(str(bot_client.__file__)).read_text(encoding="utf-8")
    for forbidden in ("forward_messages", "ForwardMessagesRequest"):
        assert forbidden not in source, forbidden
