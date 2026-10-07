"""The Telethon implementation of ``BotGateway`` (DESIGN §5, §15 "Telegram").

The bot is a second ``TelegramClient`` on its own session file (``<home>/bot.session``)
started with the token from @BotFather. It does every write into topic channels and every
exchange with the owner; the account never posts anywhere a person could see.

Only this module and ``user_client.py`` import Telethon, so every Telethon object is turned
into the contract's dataclasses right here, and every Telethon exception is translated into the
``errors.py`` vocabulary before it leaves a method. The pure conversion functions
(``to_keyboard``, ``to_bot_message``, ``to_bot_callback``, ``input_media_of``,
``can_post_from``, ``translate_error``) are module-level so they can be tested with TL objects
built in a test, without a network.

Media never travels as bytes: ``send_copy`` fetches the staged messages by id and re-sends
``InputMediaPhoto`` / ``InputMediaDocument`` references (``messages.sendMedia`` for one item,
``messages.sendMultiMedia`` for an album), which the server copies on its side; a forward is
never used, so the topic channel shows the bot as the author and the bot can edit later.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from telethon import Button as TlButton
from telethon import TelegramClient, events, functions, types, utils
from telethon import errors as tl_errors

from tg_curator import __version__
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
from tg_curator.telegram.gateway import Account, BotCallback, BotMessage, Buttons

__all__ = [
    "CALLBACK_DATA_MAX_BYTES",
    "FLOOD_SLEEP_THRESHOLD",
    "SESSION_FILENAME",
    "OwnerId",
    "TelethonBotGateway",
    "can_post_from",
    "input_media_of",
    "to_bot_callback",
    "to_bot_message",
    "to_keyboard",
    "translate_error",
    "translating",
]

log = logging.getLogger(__name__)

CALLBACK_DATA_MAX_BYTES = 64
"""Telegram's limit on inline-button callback data (§11.1, §15)."""
FLOOD_SLEEP_THRESHOLD = 120
"""Waits up to this many seconds are slept by Telethon itself; longer ones raise ``FloodWait``."""
SESSION_FILENAME = "bot.session"
"""The bot's Telethon session inside ``home`` (§3)."""
CONNECTION_RETRIES = 5
RETRY_DELAY = 5
"""Telethon's own reconnection after a dropped link: attempts and seconds between them. Finite
on purpose (an endless retry keeps every request waiting through the whole outage); once
Telethon gives up, ``ensure_connected`` (run before every request) connects again."""

MessageHandler = Callable[[BotMessage], Awaitable[None]]
CallbackHandler = Callable[[BotCallback], Awaitable[None]]
OwnerId = int | Callable[[], int]
"""The owner's user id, or a callable that reads it at use time: the owner is claimed *after*
the bot starts, so the service passes ``lambda: rt.settings.telegram.owner_id``."""

# Telethon raises ValueError (not an RPC error) when it cannot build an input entity for an
# id the bot has never seen; the message always mentions the entity.
_ENTITY_VALUE_ERROR_MARK = "entity"


# --- pure conversions --------------------------------------------------------------------------


def to_keyboard(buttons: Buttons | None) -> list[list[types.KeyboardInlineButton]] | None:
    """``Buttons`` rows -> Telethon inline keyboard rows; ``None`` when there is nothing to show.

    The 64-byte check is made here, before any request, so a too-long callback payload is a
    ``ValueError`` naming the button rather than a ``BUTTON_DATA_INVALID`` from Telegram.
    """
    if not buttons:
        return None
    rows: list[list[types.KeyboardInlineButton]] = []
    for row in buttons:
        tl_row: list[types.KeyboardInlineButton] = []
        for button in row:
            if button.data is not None:
                payload = button.data.encode("utf-8")
                if len(payload) > CALLBACK_DATA_MAX_BYTES:
                    raise ValueError(
                        f"callback data of button {button.text!r} is {len(payload)} bytes; "
                        f"Telegram allows {CALLBACK_DATA_MAX_BYTES}"
                    )
                tl_row.append(TlButton.inline(button.text, payload))
            elif button.url is not None:
                tl_row.append(TlButton.url(button.text, button.url))
            else:
                raise ValueError(f"button {button.text!r} has neither data nor url")
        if tl_row:
            rows.append(tl_row)
    return rows or None


def to_bot_message(message: types.Message, chats: Mapping[int, Any] | None = None) -> BotMessage:
    """A received ``types.Message`` -> ``BotMessage``.

    ``chats`` maps marked peer ids to the entities Telethon attached to the update, so a
    forwarded channel post gets its title; a forward from a sender who hides their account
    only carries ``fwd_from.from_name``. A channel post has no ``from_id``: its sender is the
    channel itself (``BotApp`` only ever acts on private messages, §11.1).
    """
    chat_id = message.chat_id if message.chat_id is not None else utils.get_peer_id(message.peer_id)
    sender_id = message.sender_id if message.sender_id is not None else chat_id
    fwd_chat_id: int | None = None
    fwd_title: str | None = None
    fwd = message.fwd_from
    if isinstance(fwd, types.MessageFwdHeader):
        if fwd.from_id is not None:
            fwd_chat_id = utils.get_peer_id(fwd.from_id)
            entity = (chats or {}).get(fwd_chat_id)
            if entity is not None:
                fwd_title = utils.get_display_name(entity) or None
        if fwd_title is None:
            fwd_title = fwd.from_name or None
    return BotMessage(
        chat_id=chat_id,
        message_id=message.id,
        sender_id=sender_id,
        text=message.message or "",
        is_private=isinstance(message.peer_id, types.PeerUser),
        fwd_from_chat_id=fwd_chat_id,
        fwd_from_title=fwd_title,
        reply_to_id=message.reply_to_msg_id,
        has_media=message.media is not None
        and not isinstance(message.media, types.MessageMediaWebPage),
    )


def to_bot_callback(update: types.UpdateBotCallbackQuery) -> BotCallback:
    """``UpdateBotCallbackQuery`` -> ``BotCallback`` with the payload decoded as UTF-8."""
    data = update.data or b""
    return BotCallback(
        query_id=str(update.query_id),
        sender_id=update.user_id,
        chat_id=utils.get_peer_id(update.peer),
        message_id=update.msg_id,
        data=bytes(data).decode("utf-8", errors="replace"),
    )


def input_media_of(message: types.Message | None) -> types.TypeInputMedia:
    """The re-sendable reference of a staged message's media.

    Photos, documents (video, file, voice, sticker, ...) and the other castable kinds become
    ``InputMedia*`` by id + access hash + file reference; a missing message, a web-page
    preview (a property of the text, not a file) and anything Telethon cannot cast raise
    ``MediaUnavailable`` so the publisher falls back to text + link.
    """
    if message is None:
        raise MediaUnavailable("the staged message is gone")
    media = message.media
    if media is None or isinstance(
        media, types.MessageMediaWebPage | types.MessageMediaEmpty | types.MessageMediaUnsupported
    ):
        raise MediaUnavailable(f"message {message.id} carries no re-sendable media")
    try:
        return utils.get_input_media(media)
    except TypeError as exc:
        raise MediaUnavailable(f"message {message.id}: {exc}") from exc


def can_post_from(chat: Any) -> bool:
    """Whether a ``channels.getChannels`` result says the bot may post there: a ``Channel``
    it has not left whose own ``admin_rights`` include Post Messages. A ``ChannelForbidden``
    (kicked or banned) or anything else is a no."""
    if not isinstance(chat, types.Channel) or chat.left:
        return False
    rights = chat.admin_rights
    return rights is not None and bool(rights.post_messages)


def translate_error(exc: BaseException) -> CuratorError | None:
    """The §5 error for a Telethon exception, ``None`` when it is not Telegram's to translate.

    Rate limits become ``FloodWait`` (Telethon already slept the short ones); missing rights
    ``BotCannotPost``; an unreachable peer ``ChatGone``; a refused token or api_id a
    ``ConfigError`` naming the key; an outage (connection, OS-level or timeout error, or a 5xx
    answer) ``TelegramUnavailable``; anything else Telegram refused a plain ``CuratorError``.
    """
    if isinstance(exc, CuratorError):
        return exc
    if isinstance(exc, tl_errors.AuthKeyNotFound):
        # The transport's 404: Telegram forgot the session's key. The token is still good, so
        # this is not a ConfigError; the next ``ensure_connected`` signs in a new session.
        return TelegramUnavailable(
            "Telegram no longer knows the bot's session key; a new session is signed in with "
            "the same token on the next connect"
        )
    if isinstance(exc, tl_errors.FloodError):
        return FloodWait(int(getattr(exc, "seconds", 0) or 0))
    if isinstance(exc, tl_errors.ChatForwardsRestrictedError):
        return ForwardsRestricted(str(exc))
    if isinstance(exc, _CANNOT_POST):
        return BotCannotPost(str(exc))
    if isinstance(exc, _GONE):
        return ChatGone(str(exc))
    if isinstance(exc, _MEDIA):
        return MediaUnavailable(str(exc))
    if isinstance(exc, _NOT_ALLOWED):
        return NotAllowed("other", str(exc))
    if isinstance(exc, tl_errors.AccessTokenInvalidError | tl_errors.AccessTokenExpiredError):
        return ConfigError(
            "telegram.bot_token was refused by Telegram: paste the token from @BotFather again"
        )
    if isinstance(exc, tl_errors.ApiIdInvalidError | tl_errors.ApiIdPublishedFloodError):
        return ConfigError(
            "telegram.api_id / telegram.api_hash were refused by Telegram: "
            "copy both again from https://my.telegram.org (API development tools)"
        )
    if isinstance(exc, tl_errors.UnauthorizedError | tl_errors.AuthKeyError):
        return ConfigError(
            "telegram.bot_token: Telegram no longer accepts the bot session; "
            "get the token from @BotFather again and restart"
        )
    if _is_outage(exc):
        return TelegramUnavailable(f"Telegram is unavailable: {exc or type(exc).__name__}")
    if isinstance(exc, tl_errors.RPCError):
        return CuratorError(f"Telegram refused the request: {type(exc).__name__}: {exc}")
    if isinstance(exc, ValueError) and _ENTITY_VALUE_ERROR_MARK in str(exc):
        return ChatGone(str(exc))
    return None


def _is_outage(exc: BaseException) -> bool:
    """A transport error (``ConnectionError`` and ``TimeoutError`` are ``OSError``s), Telegram's
    ``ServerError``/``TimedOutError``, or any RPC error with a 5xx code (Telegram also sends
    them negative, e.g. -500 "No workers running")."""
    if isinstance(exc, OSError | tl_errors.ServerError | tl_errors.TimedOutError):
        return True
    code = getattr(exc, "code", None) if isinstance(exc, tl_errors.RPCError) else None
    return isinstance(code, int) and abs(code) >= 500


_CANNOT_POST = (
    tl_errors.ChatWriteForbiddenError,
    tl_errors.ChatAdminRequiredError,
    tl_errors.ChatRestrictedError,
    tl_errors.UserBannedInChannelError,
    tl_errors.ChatSendMediaForbiddenError,
    tl_errors.ChatSendPhotosForbiddenError,
)
_GONE = (
    tl_errors.ChannelPrivateError,
    tl_errors.ChannelInvalidError,
    tl_errors.PeerIdInvalidError,
    tl_errors.ChatIdInvalidError,
    tl_errors.UserIsBlockedError,
    tl_errors.InputUserDeactivatedError,
    tl_errors.UserIsBotError,
)
_MEDIA = (
    tl_errors.FileReferenceExpiredError,
    tl_errors.MediaEmptyError,
    tl_errors.MediaInvalidError,
    tl_errors.WebpageMediaEmptyError,
)
_NOT_ALLOWED = (
    tl_errors.MessageDeleteForbiddenError,
    tl_errors.MessageAuthorRequiredError,
    tl_errors.MessageIdInvalidError,
)


@contextmanager
def translating() -> Iterator[None]:
    """Translate whatever escapes the block (§5: no Telethon exception leaves the gateway)."""
    try:
        yield
    except CuratorError:
        raise
    except Exception as exc:
        translated = translate_error(exc)
        if translated is None:
            raise
        raise translated from exc


def _forward_entities(message: Any) -> dict[int, Any]:
    """The entities Telethon resolved for a message's forward header, keyed by marked id."""
    forward = getattr(message, "forward", None)
    if forward is None:
        return {}
    out: dict[int, Any] = {}
    for entity in (forward.chat, forward.sender):
        if entity is not None:
            out[utils.get_peer_id(entity)] = entity
    return out


# --- the gateway ------------------------------------------------------------------------------


class TelethonBotGateway:
    """``BotGateway`` over a Telethon ``TelegramClient`` logged in with the bot token.

    ``client`` and ``client_factory`` are only for tests: a stub that records the calls listed
    in the methods below, and what builds the next one when the session has to be renewed.
    The token is held for ``start()`` and never logged or repeated anywhere.
    """

    def __init__(
        self,
        home: Path,
        *,
        api_id: int,
        api_hash: str,
        bot_token: str,
        owner_id: OwnerId,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._home = home
        self._api_id = api_id
        self._api_hash = api_hash
        self._bot_token = bot_token
        self._owner = owner_id
        self._client = client
        self._client_factory = client_factory or self._build_client
        self._started = False
        self._handlers_registered = False
        self._link = asyncio.Lock()
        self._drop_noticed = False
        self._message_handlers: list[MessageHandler] = []
        self._callback_handlers: list[CallbackHandler] = []

    # --- lifecycle ---

    async def start(self) -> Account:
        """Connect, sign in with the token when the session is fresh, hook the update handlers
        and return the bot identity. Stores nothing: ``service.py`` keeps the ``Account``."""
        client = self._client or self._client_factory()
        with translating():
            try:
                await client.connect()
            except tl_errors.AuthKeyNotFound:
                client = await self._renew_session(client)
            if not await client.is_user_authorized():
                await client.sign_in(bot_token=self._bot_token)
            me = await client.get_me()
        if me is None:
            raise ConfigError("telegram.bot_token: the bot could not sign in; check the token")
        self._client = client
        self._started = True
        self._drop_noticed = False
        self._install_handlers(client)
        account = Account(
            id=me.id, name=utils.get_display_name(me), username=me.username, phone=None
        )
        log.info("bot connected as @%s (id %s)", account.username, account.id)
        return account

    async def stop(self) -> None:
        if not self._started:
            return
        client = self._require_client()
        self._started = False
        with translating():
            await client.disconnect()

    def _install_handlers(self, client: Any) -> None:
        if not self._handlers_registered:
            client.add_event_handler(self._handle_new_message, events.NewMessage(incoming=True))
            client.add_event_handler(self._handle_callback, events.CallbackQuery())
            self._handlers_registered = True

    async def ensure_connected(self) -> None:
        """Connect the started bot again when Telegram dropped the link and Telethon gave up
        reconnecting (it disconnects for good after ``CONNECTION_RETRIES`` failed attempts,
        and nothing else would ever connect again: the bot would stay deaf and mute). A
        no-op while connected, before ``start()`` and after ``stop()``; a failure is raised
        as ``TelegramUnavailable`` and retried on the next call."""
        if not self._started or self._client is None or self._client.is_connected():
            return
        async with self._link:
            client = self._client
            if not self._started or client is None or client.is_connected():
                return  # another caller reconnected meanwhile
            if not self._drop_noticed:
                self._drop_noticed = True
                log.warning("the bot's connection to Telegram dropped: reconnecting")
            with translating():
                try:
                    await client.connect()
                except tl_errors.AuthKeyNotFound:
                    await self._renew_session(client)
            self._drop_noticed = False
            log.info("the bot is connected to Telegram again")

    async def _renew_session(self, old: Any) -> Any:
        """Telegram no longer knows the session's auth key (the transport's 404), so the key
        and the file holding it are useless. The token still is not: drop ``bot.session``,
        build a fresh client and sign in again with the same token."""
        log.warning(
            "Telegram no longer knows the bot's session key: starting a new bot session with "
            "the same token"
        )
        with contextlib.suppress(Exception):
            await old.disconnect()
        for name in (SESSION_FILENAME, f"{SESSION_FILENAME}-journal"):
            try:
                (self._home / name).unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove %s", name, exc_info=True)
        client = self._client_factory()
        self._client = client
        self._handlers_registered = False
        await client.connect()
        await client.sign_in(bot_token=self._bot_token)
        if self._started:
            self._install_handlers(client)
        return client

    def _build_client(self) -> TelegramClient:
        """A client whose session lives in ``home`` and that sleeps short flood waits itself.
        The device/app strings mirror the account client's (§5) so both sessions are
        recognisable as tg-curator; ``catch_up=False`` keeps downtime from being replayed;
        ``raise_last_call_error=True`` makes a request that failed through all of Telethon's
        retries raise the server error itself (translated to ``TelegramUnavailable``) rather
        than a bare ``ValueError``."""
        client = TelegramClient(
            str(self._home / SESSION_FILENAME),
            self._api_id,
            self._api_hash,
            flood_sleep_threshold=FLOOD_SLEEP_THRESHOLD,
            raise_last_call_error=True,
            connection_retries=CONNECTION_RETRIES,
            retry_delay=RETRY_DELAY,
            auto_reconnect=True,
            catch_up=False,
            device_model="tg-curator",
            system_version="bot",
            app_version=__version__,
        )
        client.parse_mode = "html"
        return client

    # --- updates ---

    def on_message(self, handler: MessageHandler) -> None:
        self._message_handlers.append(handler)

    def on_callback(self, handler: CallbackHandler) -> None:
        self._callback_handlers.append(handler)

    async def _handle_new_message(self, event: Any) -> None:
        message = event.message
        try:
            converted = to_bot_message(message, _forward_entities(message))
        except Exception:
            log.exception("could not read an incoming bot message")
            return
        await self._dispatch(self._message_handlers, converted)

    async def _handle_callback(self, event: Any) -> None:
        # Inline-mode queries (UpdateInlineBotCallbackQuery) have no chat and no message id;
        # the bot never offers inline results, so they are ignored.
        if not isinstance(event.query, types.UpdateBotCallbackQuery):
            return
        await self._dispatch(self._callback_handlers, to_bot_callback(event.query))

    @staticmethod
    async def _dispatch(handlers: Sequence[Callable[[Any], Awaitable[None]]], item: Any) -> None:
        """Every handler gets the update even when another one fails: one bad command must
        not silence the bot."""
        for handler in list(handlers):
            try:
                await handler(item)
            except Exception:
                log.exception("bot handler %s failed", getattr(handler, "__name__", handler))

    # --- sending ---

    async def send_text(
        self,
        chat_id: int,
        html: str,
        *,
        buttons: Buttons | None = None,
        reply_to: int | None = None,
        silent: bool = False,
    ) -> int:
        keyboard = to_keyboard(buttons)
        client = await self._ready()
        with translating():
            sent = await client.send_message(
                chat_id,
                html,
                parse_mode="html",
                link_preview=False,
                buttons=keyboard,
                reply_to=reply_to,
                silent=silent,
            )
        return int(sent.id)

    async def send_copy(
        self,
        from_chat_id: int,
        message_ids: Sequence[int],
        to_chat_id: int,
        *,
        caption_html: str | None = None,
        buttons: Buttons | None = None,
    ) -> list[int]:
        """Fetch the staged messages and re-send their media references; a file reference
        that expired between staging and sending is refreshed once by re-fetching."""
        ids = [int(i) for i in message_ids]
        if not ids:
            return []
        if len(ids) > 1 and buttons:
            raise ValueError(
                "albums cannot carry buttons (Telegram forbids reply markup on grouped media)"
            )
        keyboard = to_keyboard(buttons)
        media = await self._fetch_media(from_chat_id, ids)
        with translating():
            try:
                return await self._send_media(to_chat_id, media, caption_html, keyboard)
            except tl_errors.FileReferenceExpiredError:
                log.info("file reference expired for %s in %s: refetching once", ids, from_chat_id)
            media = await self._fetch_media(from_chat_id, ids)
            return await self._send_media(to_chat_id, media, caption_html, keyboard)

    async def _fetch_media(self, chat_id: int, ids: list[int]) -> list[types.TypeInputMedia]:
        """``channels.getMessages`` by id (Telethon keeps the result aligned with ``ids`` and
        puts ``None`` where a message is gone)."""
        client = await self._ready()
        with translating():
            messages = await client.get_messages(chat_id, ids=ids)
        return [input_media_of(message) for message in messages]

    async def _send_media(
        self,
        to_chat_id: int,
        media: list[types.TypeInputMedia],
        caption_html: str | None,
        keyboard: list[list[types.KeyboardInlineButton]] | None,
    ) -> list[int]:
        """One item -> ``messages.sendMedia`` with caption and buttons; several -> one
        ``messages.sendMultiMedia`` album, which can carry neither (§5)."""
        client = self._require_client()
        if len(media) == 1:
            sent = await client.send_file(
                to_chat_id,
                media[0],
                caption=caption_html or "",
                parse_mode="html",
                buttons=keyboard,
            )
            return [int(sent.id)]
        sent_all = await client.send_file(to_chat_id, media, caption="", parse_mode="html")
        return [int(m.id) for m in sent_all]

    # --- editing ---

    async def edit_text(
        self, chat_id: int, message_id: int, html: str, *, buttons: Buttons | None = None
    ) -> None:
        """One ``messages.editMessage`` changes text or caption and the keyboard together;
        ``buttons=None`` leaves the request without ``reply_markup``, which removes it."""
        keyboard = to_keyboard(buttons)
        client = await self._ready()
        with translating():
            try:
                await client.edit_message(
                    chat_id,
                    message_id,
                    html,
                    parse_mode="html",
                    link_preview=False,
                    buttons=keyboard,
                )
            except tl_errors.MessageNotModifiedError:
                log.debug("message %s in %s already had that text", message_id, chat_id)

    async def edit_buttons(self, chat_id: int, message_id: int, buttons: Buttons | None) -> None:
        keyboard = to_keyboard(buttons)
        client = await self._ready()
        with translating():
            try:
                await client.edit_message(
                    chat_id, message_id, None, parse_mode=None, buttons=keyboard
                )
            except tl_errors.MessageNotModifiedError:
                log.debug("message %s in %s already had those buttons", message_id, chat_id)

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        """Only in the owner's private chat (the chat id of a private chat is the user id):
        the curator deletes nothing anywhere else (§1)."""
        owner = self._owner() if callable(self._owner) else self._owner
        if chat_id <= 0 or chat_id != owner:
            raise NotAllowed("other", "the bot deletes messages only in the owner's private chat")
        client = await self._ready()
        with translating():
            await client.delete_messages(chat_id, [message_id], revoke=True)

    # --- callbacks and rights ---

    async def answer_callback(
        self, query_id: str, text: str | None = None, *, alert: bool = False
    ) -> None:
        """``messages.setBotCallbackAnswer``; a query Telegram already expired is not an
        error worth raising — the press was handled, only the spinner is gone."""
        request = functions.messages.SetBotCallbackAnswerRequest(
            query_id=int(query_id), cache_time=0, alert=alert, message=text
        )
        client = await self._ready()
        with translating():
            try:
                await client(request)
            except tl_errors.QueryIdInvalidError:
                log.debug("callback %s expired before it was answered", query_id)

    async def can_post(self, chat_id: int) -> bool:
        """``channels.getChannels`` -> the bot's own ``admin_rights`` and ``left``.

        A channel the bot has never seen (Telethon cannot build the input entity) or may not
        access (``CHANNEL_PRIVATE``) is a plain ``False``: that is the state before
        ``add_bot_admin`` runs (§8 ``link_channel``). ``CHANNEL_INVALID`` / ``PEER_ID_INVALID``
        on a channel the bot knows mean it is gone: ``ChatGone``.
        """
        client = await self._ready()
        with translating():
            try:
                result = await client(functions.channels.GetChannelsRequest([chat_id]))
            except tl_errors.ChannelPrivateError:
                return False
            except ValueError as exc:
                if _ENTITY_VALUE_ERROR_MARK not in str(exc):
                    raise
                return False
        chats = list(getattr(result, "chats", []) or [])
        return bool(chats) and can_post_from(chats[0])

    def _require_client(self) -> Any:
        if not self._started or self._client is None:
            raise CuratorError("the bot is not started")
        return self._client

    async def _ready(self) -> Any:
        """The started client, connected again first if Telegram dropped the link."""
        self._require_client()
        await self.ensure_connected()
        return self._require_client()
