"""The Telethon implementation of ``UserGateway`` (DESIGN §5, §15 "Telegram").

This module and ``bot_client.py`` are the only places that import Telethon. Everything the
account does to Telegram goes through here, which is what makes the safety rules of DESIGN §1
structural: there is no method that sends a message, joins, reacts or marks anything as read,
every content write checks the owned-chat registry, every subscription write checks the
opposite, and every Telethon exception is translated into the ``errors.py`` vocabulary before
it leaves. The pure parts (TL object -> dataclass conversion, reference parsing, error
translation) are module functions so they can be unit-tested without a client.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from telethon import TelegramClient, errors, events, functions, helpers, types, utils
from telethon.extensions import html as tl_html
from telethon.sessions import SQLiteSession

from tg_curator import __version__
from tg_curator.clock import Clock, SystemClock
from tg_curator.errors import (
    ChatGone,
    CuratorError,
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
from tg_curator.logging_setup import mask_phone
from tg_curator.telegram.gateway import Account, ChatInfo, IncomingMessage, MediaKind

log = logging.getLogger("tg_curator.telegram.user")

SESSION_FILENAME = "user.session"
PENDING_FILENAME = f"{SESSION_FILENAME}.pending"
"""The session a re-login signs in on, beside the working one, until it is swapped in."""
DEVICE_MODEL = "tg-curator"
FLOOD_SLEEP_THRESHOLD = 120
LOGOUT_TIMEOUT = 10.0
"""How long logging out a replaced session may hold up the end of a re-login."""
# Telethon's own reconnection after a dropped link: this many attempts, this many seconds
# apart. It is deliberately finite (an endless retry keeps every request waiting for as long
# as the outage lasts, so no loop tick ever ends); once Telethon gives up, the next request,
# ``ping()`` or ``ensure_connected()`` connects again (see ``_reconnect_if_dropped``).
CONNECTION_RETRIES = 5
RETRY_DELAY = 5
HISTORY_LIMIT = 2000
VIEWS_BATCH = 100
COPY_ITEM_GAP_SECONDS = 1.0
"""Pause between the account's media copies when they cannot go as one album (§14)."""
FOLDER_TITLE_MAX = 12
FOLDER_ID_RANGE = range(2, 256)
MUTE_FOREVER = 2**31 - 1
_DELIVERED_MEMORY = 4096

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'()]+", re.IGNORECASE)
_URL_TRAIL = ".,;:!?"
_TME_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me|telegram\.dog)/(?P<path>[^?#]+)",
    re.IGNORECASE,
)
_TG_SCHEME_RE = re.compile(r"^tg://(?P<kind>join|resolve)\?(?P<query>.*)$", re.IGNORECASE)
_CHANNEL_MARK = -1_000_000_000_000

# Errors that end the session (DESIGN §5); SessionPasswordNeededError shares the base and is
# the one UnauthorizedError that must NOT count.
_SESSION_LOST: dict[type[Exception], str] = {
    errors.AuthKeyUnregisteredError: "unregistered",
    errors.AuthKeyInvalidError: "invalid",
    errors.SessionRevokedError: "revoked",
    errors.SessionExpiredError: "expired",
    errors.UserDeactivatedError: "deactivated",
    errors.UserDeactivatedBanError: "banned",
    errors.AuthKeyDuplicatedError: "duplicated",
    # Not an RPC error: the transport's 404, "the server does not know this auth key". Telethon
    # fails every pending request with it and disconnects; the key is useless from then on,
    # so it ends the session like AUTH_KEY_UNREGISTERED does.
    errors.AuthKeyNotFound: "unregistered",
}
_FOLDER_LIMIT_CODES = frozenset({"DIALOG_FILTERS_TOO_MUCH", "FILTER_INCLUDE_TOO_MUCH"})
_FOLDER_VALUE_CODES = frozenset({"FILTER_INCLUDE_EMPTY", "FILTER_TITLE_EMPTY"})
_FLOOD_ERRORS = (errors.FloodWaitError, errors.FloodPremiumWaitError, errors.SlowModeWaitError)
# Telegram's own trouble (5xx, including the -500/-503 variants) and the transport's: nothing is
# wrong with the request, so it becomes TelegramUnavailable and is retried on the next tick.
# ConnectionError and TimeoutError are OSError subclasses.
_OUTAGE_ERRORS = (OSError, errors.ServerError, errors.TimedOutError)
_FILE_REFERENCE_ERRORS = (
    errors.FileReferenceExpiredError,
    errors.FileReferenceEmptyError,
    errors.FileReferenceInvalidError,
)
_MEDIA_ERRORS = (*_FILE_REFERENCE_ERRORS, errors.MediaEmptyError, errors.MediaInvalidError)
_SOURCE_GONE_ERRORS = (errors.MessageIdInvalidError, errors.MessageIdsEmptyError)
_GONE_ERRORS = (
    errors.ChannelPrivateError,
    errors.ChannelInvalidError,
    errors.ChatIdInvalidError,
    errors.PeerIdInvalidError,
    errors.UsernameNotOccupiedError,
    errors.UsernameInvalidError,
    errors.InviteHashExpiredError,
    errors.InviteHashInvalidError,
)
_LOGIN_ERRORS: list[tuple[type[Exception], str]] = [
    (errors.PhoneCodeInvalidError, "bad_code"),
    (errors.PhoneCodeEmptyError, "bad_code"),
    (errors.PhoneCodeExpiredError, "expired_code"),
    (errors.PasswordHashInvalidError, "bad_password"),
    (errors.PhoneNumberInvalidError, "bad_phone"),
    (errors.PhoneNumberBannedError, "bad_phone"),
    (errors.PhoneNumberUnoccupiedError, "bad_phone"),
    (errors.PhoneNumberFloodError, "flood"),
    (errors.PhonePasswordFloodError, "flood"),
    (errors.FloodWaitError, "flood"),
]

_in_handler: contextvars.ContextVar[bool] = contextvars.ContextVar("in_handler", default=False)

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]
LostHandler = Callable[[str], Awaitable[None]]
ChatRef = tuple[Literal["id", "username", "invite"], int | str]


# --- pure conversions ---------------------------------------------------------------------


def translate_error(exc: BaseException) -> Exception | None:
    """The ``errors.py`` equivalent of a Telethon exception, ``None`` when there is none.

    Kept as a table so the boundary is reviewable in one place: the session-ending errors
    become ``SessionLost``, waits longer than the client sleeps itself become ``FloodWait``,
    the two folder-content refusals become the ``ValueError`` the contract names, an outage
    (connection, OS-level or timeout error, or a 5xx answer) becomes ``TelegramUnavailable``,
    and every other ``RPCError`` becomes a ``NotAllowed("other")`` carrying the code, so no
    Telethon class ever reaches another module.
    """
    if isinstance(exc, errors.SessionPasswordNeededError):
        return None
    for cls, reason in _SESSION_LOST.items():
        if isinstance(exc, cls):
            return SessionLost(reason, str(exc))
    if isinstance(exc, errors.UnauthorizedError):
        return SessionLost("other", str(exc))
    if isinstance(exc, _FLOOD_ERRORS):
        return FloodWait(exc.seconds)
    if isinstance(exc, errors.ChatForwardsRestrictedError):
        return ForwardsRestricted(str(exc))
    if isinstance(exc, _MEDIA_ERRORS):
        return MediaUnavailable(str(exc))
    if isinstance(exc, _SOURCE_GONE_ERRORS):
        # The message was deleted at its source: nothing can be forwarded or copied any more
        # (the publisher reposts the version first seen instead, SPEC "Channels and groups").
        return MediaUnavailable(f"the source message is gone: {exc}")
    if isinstance(exc, _GONE_ERRORS):
        return ChatGone(str(exc))
    if isinstance(exc, errors.UserCreatorError):
        return NotAllowed("creator", "the account created this chat and cannot leave it")
    if isinstance(exc, errors.FreshChangeAdminsForbiddenError):
        return NotAllowed("fresh_session", "Telegram does not let a new session change admins yet")
    if is_outage(exc):
        return TelegramUnavailable(f"Telegram is unavailable: {exc or type(exc).__name__}")
    if isinstance(exc, errors.RPCError):
        code = getattr(exc, "message", "") or ""
        if code in _FOLDER_LIMIT_CODES:
            return FolderLimit(code)
        if code in _FOLDER_VALUE_CODES:
            return ValueError(f"the folder was refused: {code}")
        return NotAllowed("other", f"{code or type(exc).__name__}: {exc}")
    return None


def is_outage(exc: BaseException) -> bool:
    """Whether ``exc`` says Telegram is unreachable or failing on its side, not that the
    request was wrong: a transport error, ``ServerError``/``TimedOutError``, or any RPC error
    whose code is 5xx (Telegram also reports them negative, e.g. -500 "No workers running")."""
    if isinstance(exc, _OUTAGE_ERRORS):
        return True
    code = getattr(exc, "code", None) if isinstance(exc, errors.RPCError) else None
    return isinstance(code, int) and abs(code) >= 500


def login_error(exc: BaseException) -> LoginError | None:
    """Map a login-step failure to ``LoginError(reason)`` (DESIGN §5), ``None`` otherwise."""
    for cls, reason in _LOGIN_ERRORS:
        if isinstance(exc, cls):
            return LoginError(reason, str(exc))
    return None


def parse_chat_ref(ref: str | int) -> ChatRef:
    """Classify a chat reference the owner typed: marked id, username or invite hash.

    ``t.me/c/<bare>/<msg>`` links are turned into the marked id, ``t.me/<name>/<msg>`` into
    the username, ``t.me/+hash`` and ``t.me/joinchat/hash`` into an invite hash. Telethon's own
    parser does not handle message links, which is why this exists.
    """
    if isinstance(ref, int):
        return ("id", ref)
    text = ref.strip()
    if re.fullmatch(r"-?\d+", text):
        return ("id", int(text))
    scheme = _TG_SCHEME_RE.match(text)
    if scheme:
        params = dict(p.split("=", 1) for p in scheme.group("query").split("&") if "=" in p)
        if scheme.group("kind").lower() == "join" and params.get("invite"):
            return ("invite", params["invite"])
        if params.get("domain"):
            return ("username", params["domain"].lower())
        raise ChatGone(f"{ref!r} is not a chat link")
    link = _TME_RE.match(text)
    if link:
        parts = [p for p in link.group("path").split("/") if p]
        if not parts:
            raise ChatGone(f"{ref!r} is not a chat link")
        head = parts[0]
        if head.startswith("+"):
            return ("invite", head[1:])
        if head.lower() == "joinchat" and len(parts) > 1:
            return ("invite", parts[1])
        if head.lower() == "c" and len(parts) > 1 and parts[1].isdigit():
            return ("id", _CHANNEL_MARK - int(parts[1]))
        return ("username", head.lower())
    return ("username", text.lstrip("@").lower())


def chat_info_from_entity(
    entity: Any, *, archived: bool = False, muted_until: datetime | None = None
) -> ChatInfo | None:
    """``types.Channel`` / ``types.Chat`` -> ``ChatInfo``; ``None`` for anything that is not a
    channel or group the account is in (users, left or deactivated chats, monoforums,
    forbidden stubs).

    Gigagroups are broadcast-like (only admins post) and count as channels. ``is_admin`` is
    true only with posting rights in a channel; in a group any admin rights object counts,
    since groups have no post right and the flag there only matters for the review wording.
    """
    if isinstance(entity, types.Channel):
        if entity.monoforum:
            return None
        kind = "channel" if (entity.broadcast or entity.gigagroup) else "group"
        rights = entity.admin_rights
        if kind == "channel":
            is_admin = bool(rights is not None and rights.post_messages)
        else:
            is_admin = rights is not None
        return ChatInfo(
            id=utils.get_peer_id(entity),
            kind=kind,
            title=entity.title,
            username=_username(entity),
            noforwards=bool(entity.noforwards),
            is_creator=bool(entity.creator),
            is_admin=is_admin,
            archived=archived,
            muted_until=muted_until,
        )
    if isinstance(entity, types.Chat):
        if entity.deactivated or entity.migrated_to is not None:
            return None
        return ChatInfo(
            id=utils.get_peer_id(entity),
            kind="group",
            title=entity.title,
            username=None,
            noforwards=bool(entity.noforwards),
            is_creator=bool(entity.creator),
            is_admin=entity.admin_rights is not None,
            archived=archived,
            muted_until=muted_until,
        )
    return None


def chat_info_from_dialog(dialog: Any, *, now: datetime) -> ChatInfo | None:
    """A ``Dialog`` from ``iter_dialogs`` -> ``ChatInfo`` with its archive and mute state.

    A dialog can linger for a chat the account has left (``left`` flag); those are not
    subscriptions and are dropped here, while ``chat_info_from_entity`` keeps them so a public
    channel can still be read without membership (``add_example_channel``, §8).
    """
    if getattr(dialog.entity, "left", False):
        return None
    settings = getattr(getattr(dialog, "dialog", None), "notify_settings", None)
    return chat_info_from_entity(
        dialog.entity,
        archived=bool(getattr(dialog, "archived", False)),
        muted_until=mute_state(settings, now=now),
    )


def mute_state(settings: Any, *, now: datetime) -> datetime | None:
    """``PeerNotifySettings.mute_until`` as "muted until", ``None`` when not muted now."""
    until = getattr(settings, "mute_until", None)
    if until is None or until <= now:
        return None
    return until


def media_kind(media: Any) -> MediaKind | None:
    """Map a message's media to the curator's vocabulary.

    Photos and documents can be re-sent by reference; everything else (web previews, stories,
    paid media, polls, invoices, giveaways, locations...) is ``"other"``: noted in the post,
    never copied. ``"album"`` is assigned by intake once the grouped messages are merged.
    """
    if media is None or isinstance(media, types.MessageMediaEmpty):
        return None
    if isinstance(media, types.MessageMediaPhoto):
        return "photo"
    if isinstance(media, types.MessageMediaDocument):
        if media.video and not media.round:
            return "video"
        document = media.document
        attributes = getattr(document, "attributes", None) or []
        for attribute in attributes:
            if isinstance(attribute, types.DocumentAttributeVideo) and not attribute.round_message:
                return "video"
        return "file"
    return "other"


def media_key(media: Any) -> tuple[str, int] | None:
    """The identity of a photo or document, which survives a re-send by reference; ``None``
    for anything else."""
    if isinstance(media, types.MessageMediaPhoto) and media.photo is not None:
        return ("photo", media.photo.id)
    if isinstance(media, types.MessageMediaDocument) and media.document is not None:
        return ("document", media.document.id)
    return None


def copyable_media(media: Any) -> bool:
    """Whether the media can be re-sent by reference (a photo or a document)."""
    return isinstance(media, types.MessageMediaPhoto | types.MessageMediaDocument)


def extract_urls(text: str, entities: Sequence[Any] | None) -> tuple[str, ...]:
    """Links of a message: text-link entities, auto-detected url entities and anything that
    looks like a URL in the plain text, deduplicated in order of appearance."""
    found: list[str] = []
    surrogate = helpers.add_surrogate(text) if entities else ""
    for entity in entities or ():
        if isinstance(entity, types.MessageEntityTextUrl):
            found.append(entity.url)
        elif isinstance(entity, types.MessageEntityUrl):
            found.append(
                helpers.del_surrogate(surrogate[entity.offset : entity.offset + entity.length])
            )
    for match in _URL_RE.finditer(text):
        found.append(match.group(0).rstrip(_URL_TRAIL))
    out: list[str] = []
    for url in found:
        url = url.strip()
        if url and url not in out:
            out.append(url)
    return tuple(out)


def forward_origin(fwd: Any) -> tuple[int | None, int | None]:
    """``(chat id, message id)`` of a forwarded channel post, ``(None, None)`` otherwise.

    Forwards from users or groups carry no original message id; reporting the sender alone
    would make the engine treat every forward from that sender as the same story.
    """
    if fwd is None or fwd.channel_post is None or fwd.from_id is None:
        return (None, None)
    return (utils.get_peer_id(fwd.from_id), fwd.channel_post)


def reply_target(reply_to: Any) -> int | None:
    """The replied-to message id, ignoring the implicit "reply" to a forum topic root.

    In forums every message in a topic carries ``reply_to`` pointing at the topic; only when
    ``reply_to_top_id`` is set is ``reply_to_msg_id`` a reply to a specific message.
    """
    if not isinstance(reply_to, types.MessageReplyHeader):
        return None
    if reply_to.forum_topic and reply_to.reply_to_top_id is None:
        return None
    return reply_to.reply_to_msg_id


def _one_album(messages: Sequence[Any]) -> bool:
    """Do all ``messages`` belong to one source album (so they can be re-sent as one)?"""
    groups = {getattr(m, "grouped_id", None) for m in messages}
    return len(groups) == 1 and None not in groups


def topic_of(reply_to: Any) -> int | None:
    """The forum topic (thread top) a message belongs to; ``None`` for the General topic.

    A reply inside a topic carries the topic in ``reply_to_top_id``; a plain message in a
    topic carries it in ``reply_to_msg_id`` with ``forum_topic`` set (see ``reply_target``).
    """
    if not isinstance(reply_to, types.MessageReplyHeader):
        return None
    if reply_to.reply_to_top_id is not None:
        return reply_to.reply_to_top_id
    if reply_to.forum_topic:
        return reply_to.reply_to_msg_id
    return None


def message_to_incoming(msg: Any, chat: ChatInfo, *, now: datetime) -> IncomingMessage | None:
    """``types.Message`` -> ``IncomingMessage``; ``None`` for service messages (never
    delivered). Built from the raw TL fields only, so it needs no client."""
    if not isinstance(msg, types.Message):
        return None
    text = msg.message or ""
    entities = list(msg.entities or ())
    fwd_chat, fwd_msg = forward_origin(msg.fwd_from)
    return IncomingMessage(
        chat=chat,
        message_id=msg.id,
        date=msg.date or now,
        text=text,
        html=tl_html.unparse(text, entities) if entities else None,
        sender_id=utils.get_peer_id(msg.from_id) if msg.from_id is not None else None,
        is_outgoing=bool(msg.out),
        is_service=False,
        reply_to_id=reply_target(msg.reply_to),
        grouped_id=msg.grouped_id,
        media=media_kind(msg.media),
        urls=extract_urls(text, entities),
        views=msg.views,
        forwards=msg.forwards,
        fwd_from_chat_id=fwd_chat,
        fwd_from_message_id=fwd_msg,
        noforwards=bool(chat.noforwards or msg.noforwards),
        topic_id=topic_of(msg.reply_to),
        is_automatic_forward=is_automatic_forward(msg),
    )


def is_automatic_forward(msg: Any) -> bool:
    """Is ``msg`` the copy Telegram puts into a channel's linked discussion group?

    That copy is sent by the channel itself (``from_id``), forwarded from the channel
    (``fwd_from.from_id``) and saved from it (``fwd_from.saved_from_peer``): all three name
    the same channel. A member forwarding a channel post has their own ``from_id``.
    """
    fwd = getattr(msg, "fwd_from", None)
    if not isinstance(fwd, types.MessageFwdHeader):
        return False
    if msg.from_id is None or fwd.from_id is None or fwd.saved_from_peer is None:
        return False
    if not isinstance(fwd.from_id, types.PeerChannel):
        return False
    channel = utils.get_peer_id(fwd.from_id)
    return (
        utils.get_peer_id(msg.from_id) == channel
        and utils.get_peer_id(fwd.saved_from_peer) == channel
    )


def mute_timestamp(until: datetime | None) -> int:
    """``mute_until`` for ``InputPeerNotifySettings``: 0 unmutes, ``datetime.max`` is forever."""
    if until is None:
        return 0
    if until.year >= 9999:
        return MUTE_FOREVER
    return min(int(until.timestamp()), MUTE_FOREVER)


def _username(entity: types.Channel) -> str | None:
    if entity.username:
        return entity.username
    for extra in entity.usernames or ():
        if extra.active:
            return extra.username
    return None


def _account(user: types.User) -> Account:
    name = " ".join(part for part in (user.first_name, user.last_name) if part)
    return Account(
        id=user.id,
        name=name or user.username or str(user.id),
        username=user.username,
        phone=user.phone,
    )


# --- the gateway --------------------------------------------------------------------------


class _ExactPathSession(SQLiteSession):
    """A Telethon SQLite session stored at exactly ``path``.

    Telethon appends ``.session`` to any file name that does not end in it, which would put
    the pending session at ``user.session.pending.session``: start-up looks for (and deletes)
    ``user.session.pending``, so the name must be the one given. ``filename`` is pinned; the
    two values Telethon's constructor assigns are ignored."""

    def __init__(self, path: Path) -> None:
        self._exact = str(path)
        super().__init__(self._exact)

    @property
    def filename(self) -> str:
        return self._exact

    @filename.setter
    def filename(self, value: str) -> None:
        return  # Telethon's ":memory:" default and its ".session"-suffixed name


@dataclass
class _PendingLogin:
    """A re-login on its own client beside the working session (§11.2): what Telegram handed
    back for the code lives here, in memory only, and dies with an abandoned login."""

    client: Any
    phone: str | None = None
    phone_code_hash: str | None = None


class TelethonUserGateway:
    """``UserGateway`` on a ``TelegramClient`` over ``<home>/user.session``.

    ``client_factory`` and ``pending_client_factory`` exist for tests, which substitute
    recording stubs; the defaults build the real clients with ``catch_up=False`` (downtime is
    never replayed, §5), ``flood_sleep_threshold=120`` (shorter waits are slept, longer ones
    raised as ``FloodWait``), ``raise_last_call_error=True`` (after its own retries Telethon
    otherwise raises a bare ``ValueError``, which would read as "chat not found" instead of an
    outage) and a recognisable device name for the owner's devices list. The pending client of
    a re-login is the same client over ``user.session.pending``, minus the update stream.
    """

    def __init__(
        self,
        home: Path,
        api_id: int,
        api_hash: str,
        *,
        clock: Clock | None = None,
        client_factory: Callable[[], Any] | None = None,
        pending_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._home = home
        self._api_id = api_id
        self._api_hash = api_hash
        self._clock: Clock = clock or SystemClock()
        self._client_factory = client_factory or self._make_client
        self._pending_client_factory = pending_client_factory or self._make_pending_client
        self._pending: _PendingLogin | None = None
        # True while a signed-in re-login replaces the working session; ``_replaced`` is the
        # client it replaced. A session error from either is the swap, not a loss.
        self._swapping = False
        self._replaced: Any = None
        self._owned: set[int] = set()
        self._own_folders: set[int] = set()
        self._chat_cache: dict[int, ChatInfo] = {}
        self._message_handlers: list[MessageHandler] = []
        self._lost_handlers: list[LostHandler] = []
        self._delivered: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._phone: str | None = None
        self._phone_code_hash: str | None = None
        self._losing: asyncio.Task[None] | None = None
        self._lost = False
        # True between a successful ``connect()`` and ``disconnect()`` / a session loss: the
        # link is wanted, so a client Telethon gave up on is connected again on next use.
        self._keep_connected = False
        self._drop_noticed = False
        self._client: Any = self._client_factory()
        self._install_handlers()

    @property
    def session_path(self) -> Path:
        return self._home / SESSION_FILENAME

    @property
    def pending_path(self) -> Path:
        return self._home / PENDING_FILENAME

    def _make_client(self) -> TelegramClient:
        return self._telegram_client(self.session_path, receive_updates=True)

    def _make_pending_client(self) -> TelegramClient:
        """The re-login's client: same API id, hash and device name, so Telegram sees the
        same application; no update stream, since nothing listens to it before the swap."""
        return self._telegram_client(_ExactPathSession(self.pending_path), receive_updates=False)

    def _telegram_client(
        self, session: Path | SQLiteSession, *, receive_updates: bool
    ) -> TelegramClient:
        return TelegramClient(
            session,
            self._api_id,
            self._api_hash,
            catch_up=False,
            receive_updates=receive_updates,
            flood_sleep_threshold=FLOOD_SLEEP_THRESHOLD,
            raise_last_call_error=True,
            connection_retries=CONNECTION_RETRIES,
            retry_delay=RETRY_DELAY,
            auto_reconnect=True,
            device_model=DEVICE_MODEL,
            app_version=f"{DEVICE_MODEL} {__version__}",
        )

    def _install_handlers(self) -> None:
        self._client.add_event_handler(self._on_new_message, events.NewMessage(incoming=True))
        self._client.add_event_handler(self._on_album, events.Album())

    # --- the boundary ---

    @contextlib.asynccontextmanager
    async def _boundary(self, *, reconnect: bool = True) -> AsyncIterator[None]:
        """Translate every Telethon exception; a session-ending one also runs the loss
        procedure (disconnect, rename the session, fresh session, handler once).

        Before the request, a client whose link dropped for good is connected again
        (``reconnect``), so a failed reconnect is translated like any other error.

        The loops keep running while a re-login is swapped in, so two things that look like
        failures are the swap instead: a request whose client was disconnected under it (its
        wait is cancelled although this task is not) and a session error from the client
        being replaced. Both reach the caller as ``TelegramUnavailable`` — retried on the next
        tick — and neither runs the loss procedure on the new session."""
        client = self._client
        try:
            if reconnect:
                await self._reconnect_if_dropped()
            yield
        except CuratorError:
            raise
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is None or task.cancelling():
                raise  # this task really is being cancelled
            raise TelegramUnavailable("the request was cut off when its link closed") from exc
        except Exception as exc:
            translated = translate_error(exc)
            if translated is None:
                raise
            if isinstance(translated, SessionLost):
                if self._swapping or client is self._replaced:
                    raise TelegramUnavailable(
                        f"the request went to the session being replaced: {translated}"
                    ) from exc
                if self._phone_code_hash is None and client is self._client:
                    # While a login code is pending the session is unauthorised on purpose:
                    # a stray request (a loop still running during /bind) must not retire it
                    # and drop the code the owner is about to type.
                    await self._session_lost(translated.reason, str(exc))
            raise translated from exc

    async def _reconnect_if_dropped(self) -> None:
        """Connect again when the link is wanted but Telethon has given up on it.

        Telethon reconnects by itself ``CONNECTION_RETRIES`` times; when those fail it
        disconnects for good, and every request after that fails with "Cannot send requests
        while disconnected" while nothing else ever connects again. ``connect()`` also restarts
        Telethon's update loop, so live intake comes back with the link. Not after
        ``disconnect()``, before the first ``connect()`` or once the session is lost."""
        if not self._keep_connected or self._lost or self._losing is not None:
            return
        if self._client.is_connected():
            return
        if not self._drop_noticed:
            self._drop_noticed = True
            log.warning("the account's connection to Telegram dropped: reconnecting")
        await self._client.connect()
        self._drop_noticed = False
        log.info("the account is connected to Telegram again")

    async def ensure_connected(self) -> None:
        """Reconnect the account's client if Telegram dropped it (§17.3); a no-op while the
        link is up, before the first ``connect()``, after ``disconnect()`` and after a session
        loss. A failure is raised translated (``TelegramUnavailable``, ``SessionLost``)."""
        async with self._boundary(reconnect=False):
            await self._reconnect_if_dropped()

    async def _session_lost(self, reason: str, detail: str) -> None:
        if self._losing is None:
            self._losing = asyncio.create_task(self._lose_session(reason, detail))
        if not _in_handler.get():
            # Inside a Telethon event handler ``disconnect()`` cancels the running handler
            # task, so there the procedure is left to its own task; elsewhere it is awaited.
            await self._losing

    async def _lose_session(self, reason: str, detail: str) -> None:
        if self._lost:
            return
        self._lost = True
        self._keep_connected = False
        self._phone_code_hash = None
        log.warning("account session lost (%s): %s", reason, detail)
        try:
            await self._client.disconnect()
        except Exception:
            # The session is dead already; nothing here may stop the rename below.
            log.debug("disconnect after session loss failed", exc_info=True)
        self._retire_session_file()
        self._client = self._client_factory()
        self._install_handlers()
        for handler in list(self._lost_handlers):
            try:
                await handler(reason)
            except Exception:
                log.exception("session-lost handler failed")

    def _retire_session_file(self) -> None:
        path = self.session_path
        if not path.exists():
            return
        stamp = self._clock.now().strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.name}.revoked-{stamp}")
        try:
            os.replace(path, target)
            log.info("session file moved to %s", target.name)
        except OSError:
            log.warning("could not move the revoked session file %s", path, exc_info=True)

    def _chmod_session(self) -> None:
        """Telethon creates the session with the process umask; 0600 is enforced here too so
        a permissive umask never leaves the account readable (DESIGN §3)."""
        try:
            os.chmod(self.session_path, 0o600)
        except OSError:
            log.debug("could not chmod %s", self.session_path, exc_info=True)

    # --- guards ---

    def _require_owned(self, chat_id: int) -> None:
        if chat_id not in self._owned:
            raise NotOwnedError(f"chat {chat_id} is not registered as owned")

    def _require_not_owned(self, chat_id: int) -> None:
        if chat_id in self._owned:
            raise NotOwnedError(f"chat {chat_id} is an output/staging channel")

    # --- entity helpers (ValueError from Telethon's cache lookups means "cannot resolve") ---

    async def _input_peer(self, chat_id: int) -> Any:
        try:
            async with self._boundary():
                return await self._client.get_input_entity(chat_id)
        except ValueError as exc:
            raise ChatGone(f"chat {chat_id} cannot be found: {exc}") from exc

    async def _entity(self, ref: int | str) -> Any:
        try:
            async with self._boundary():
                return await self._client.get_entity(ref)
        except ValueError as exc:
            raise ChatGone(f"chat {ref} cannot be found: {exc}") from exc

    async def _dialog_state(self, peer: Any) -> tuple[bool, datetime | None]:
        """Archive and mute state of one dialog; defaults when Telegram has no dialog for
        it (a public channel the account only looked at). An outage is not "no dialog": it
        is raised like everywhere else."""
        try:
            async with self._boundary():
                result = await self._client(
                    functions.messages.GetPeerDialogsRequest([types.InputDialogPeer(peer)])
                )
        except TelegramUnavailable:
            raise
        except CuratorError:
            return (False, None)
        for dialog in getattr(result, "dialogs", ()):
            if isinstance(dialog, types.Dialog):
                return (dialog.folder_id == 1, mute_state(dialog.notify_settings, now=self._now()))
        return (False, None)

    def _now(self) -> datetime:
        return self._clock.now()

    def _remember(self, info: ChatInfo) -> ChatInfo:
        self._chat_cache[info.id] = info
        return info

    # --- session / login ---

    async def connect(self) -> bool:
        """Open the session; ``True`` if it is authorised.

        When Telegram no longer knows the session's auth key (a 404 from the transport while
        connecting), the loss procedure runs (file renamed, fresh session, handler told) and
        the fresh, empty session is connected instead: the account is then simply not bound,
        so the service starts in setup mode rather than failing on every restart.

        A ``user.session.pending`` found here belongs to a re-login a crash or a restart cut
        short (this process has none under way): it can never finish, so it is deleted."""
        if self._pending is None:
            self._discard_pending_files()
        try:
            return await self._connect_once()
        except SessionLost as exc:
            log.warning("the stored session is unusable (%s): continuing unbound", exc.reason)
        return await self._connect_once()

    async def _connect_once(self) -> bool:
        async with self._boundary(reconnect=False):
            await self._client.connect()
            self._chmod_session()
            self._lost = False
            self._losing = None
            self._keep_connected = True
            self._drop_noticed = False
            return bool(await self._client.is_user_authorized())

    async def disconnect(self) -> None:
        await self.cancel_relogin()  # a login never outlives the process (§11.2)
        self._keep_connected = False
        with contextlib.suppress(Exception):
            await self._client.disconnect()

    async def reset_session(self) -> None:
        """Forget the account entirely: the in-memory auth key goes with the old client, the
        files with it (a pending re-login too), and the next ``connect`` starts an empty
        session (§11.2)."""
        await self.disconnect()
        for name in (SESSION_FILENAME, f"{SESSION_FILENAME}-journal"):
            try:
                (self._home / name).unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove %s", name, exc_info=True)
        self._phone = None
        self._phone_code_hash = None
        self._chat_cache.clear()  # access hashes belong to the old account
        self._lost = False
        self._losing = None
        self._client = self._client_factory()
        self._install_handlers()
        log.info("account session reset: the next login starts from an empty session")

    async def me(self) -> Account | None:
        async with self._boundary():
            if not await self._client.is_user_authorized():
                return None
            user = await self._client.get_me()
        return _account(user) if user is not None else None

    async def send_code(self, phone: str) -> None:
        log.info("requesting a login code for %s", mask_phone(phone))
        try:
            async with self._login_boundary() as client:
                sent = await client.send_code_request(phone)
        except CuratorError as exc:
            raise self._as_login_error(exc) from exc
        self._keep_code(phone, sent.phone_code_hash)

    async def resend_code(self) -> None:
        phone, code_hash = self._code_state()
        if phone is None or code_hash is None:
            raise LoginError("other", "no code was requested")
        try:
            async with self._login_boundary() as client:
                sent = await client(
                    functions.auth.ResendCodeRequest(phone_number=phone, phone_code_hash=code_hash)
                )
        except CuratorError as exc:
            raise self._as_login_error(exc) from exc
        self._keep_code(phone, sent.phone_code_hash)

    async def sign_in(self, code: str) -> Literal["ok", "password_needed"]:
        phone, code_hash = self._code_state()
        if phone is None or code_hash is None:
            raise LoginError("other", "no code was requested")
        try:
            async with self._login_boundary() as client:
                await client.sign_in(phone, code, phone_code_hash=code_hash)
        except errors.SessionPasswordNeededError:
            return "password_needed"
        except CuratorError as exc:
            raise self._as_login_error(exc) from exc
        await self._finish_login()
        return "ok"

    async def sign_in_password(self, password: str) -> None:
        try:
            async with self._login_boundary() as client:
                await client.sign_in(password=password)
        except CuratorError as exc:
            raise self._as_login_error(exc) from exc
        await self._finish_login()

    async def _finish_login(self) -> None:
        pending = self._pending
        if pending is None:
            self._phone_code_hash = None
            self._chmod_session()
            log.info("account signed in as %s", mask_phone(self._phone))
            return
        log.info("account signed in again as %s: replacing the session", mask_phone(pending.phone))
        await self._adopt(pending)

    def _code_state(self) -> tuple[str | None, str | None]:
        """The phone and ``phone_code_hash`` of the login under way: the re-login's when one
        is pending, else the working client's own (a first bind)."""
        if self._pending is not None:
            return self._pending.phone, self._pending.phone_code_hash
        return self._phone, self._phone_code_hash

    def _keep_code(self, phone: str, code_hash: str) -> None:
        if self._pending is not None:
            self._pending.phone, self._pending.phone_code_hash = phone, code_hash
        else:
            self._phone, self._phone_code_hash = phone, code_hash

    @contextlib.asynccontextmanager
    async def _login_boundary(self) -> AsyncIterator[Any]:
        """The boundary of one login step; yields the client the step goes to.

        Without a re-login that is the working client under the usual boundary. During one it
        is the pending client: its errors are translated the same way, but none of them is a
        loss of the working session, so the loss procedure never runs for it and a session
        error there is reported as the failed login step it is."""
        pending = self._pending
        if pending is None:
            async with self._boundary():
                yield self._client
            return
        try:
            if not pending.client.is_connected():
                await pending.client.connect()
            yield pending.client
        except CuratorError:
            raise
        except Exception as exc:
            translated = translate_error(exc)
            if translated is None:
                raise
            if isinstance(translated, SessionLost):
                raise LoginError("other", f"the new login's session was refused: {exc}") from exc
            raise translated from exc

    # --- re-login beside the working session ---

    async def begin_relogin(self) -> None:
        await self.cancel_relogin()  # one re-login at a time: /bind again starts afresh
        try:
            _create_private(self.pending_path)
        except OSError as exc:
            raise LoginError("other", f"the new session file cannot be created: {exc}") from exc
        self._pending = _PendingLogin(self._pending_client_factory())
        try:
            async with self._login_boundary() as client:
                await client.connect()
        except BaseException:
            await self.cancel_relogin()
            raise
        log.info("re-login started beside the working session")

    async def cancel_relogin(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            with contextlib.suppress(Exception):
                await pending.client.disconnect()
            log.info("re-login abandoned: the working session stays as it was")
        self._discard_pending_files()

    def _discard_pending_files(self) -> None:
        for path in (self.pending_path, self._home / f"{PENDING_FILENAME}-journal"):
            try:
                if path.exists():
                    path.unlink()
                    log.info("removed %s", path.name)
            except OSError:
                log.warning("could not remove %s", path.name, exc_info=True)

    async def _adopt(self, pending: _PendingLogin) -> None:
        """Swap a re-login that has just signed in for the working session (§11.2).

        Until ``os.replace`` succeeds the working session is untouched, so a failure there
        leaves the old account bound. After it the replaced key is logged out (nobody holds it
        any more, and it must not linger in the owner's devices list) and disconnected before
        the new client opens the moved file, so the old client's last writes never meet the
        new session. The message handlers are installed on the new client; the loops, which
        kept running on the old one, simply carry on with it."""
        self._pending = None
        with contextlib.suppress(Exception):
            await pending.client.disconnect()  # commits and closes user.session.pending
        try:
            os.replace(self.pending_path, self.session_path)
        except OSError as exc:
            log.error("the new session could not be moved into place: %s", exc)
            self._discard_pending_files()
            raise LoginError("other", f"the new session could not be stored: {exc}") from exc
        old = self._client
        self._swapping = True
        try:
            self._keep_connected = False  # nothing connects the old client again
            try:
                await asyncio.wait_for(old(functions.auth.LogOutRequest()), LOGOUT_TIMEOUT)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is None or task.cancelling():
                    raise
                log.info("the replaced session was not logged out (its link closed)")
            except Exception as exc:
                log.info("the replaced session was not logged out (%s)", type(exc).__name__)
            with contextlib.suppress(Exception):
                await old.disconnect()
            self._replaced = old
            self._phone = None
            self._phone_code_hash = None
            self._chat_cache.clear()  # access hashes belong to the old session
            self._lost = False
            self._losing = None
            self._client = self._client_factory()
            self._install_handlers()
        finally:
            self._swapping = False
        try:
            await self._connect_once()
        except TelegramUnavailable as exc:
            self._keep_connected = True  # the next request or the link loop connects it
            log.warning("the new session is in place but not connected yet: %s", exc)
        log.info("the new session replaced the old one")

    @staticmethod
    def _as_login_error(exc: CuratorError) -> CuratorError:
        cause = exc.__cause__
        mapped = login_error(cause) if cause is not None else None
        if mapped is not None:
            return mapped
        if isinstance(exc, FloodWait):
            return LoginError("flood", str(exc))
        if isinstance(exc, SessionLost | LoginError | TelegramUnavailable):
            return exc
        return LoginError("other", str(exc))

    async def ping(self) -> bool:
        # The update loop reports its fatal error here and disconnects (Telethon never calls
        # back); checking it first turns a quiet loop death into a proper session loss.
        loop_error = getattr(self._client, "_updates_error", None)
        translated = translate_error(loop_error) if loop_error is not None else None
        if isinstance(translated, SessionLost):
            await self._session_lost(translated.reason, str(loop_error))
            return False
        if not self._client.is_connected() and not self._keep_connected:
            return False  # never connected, deliberately disconnected, or the session is gone
        try:
            # The boundary first connects again a client Telethon gave up on, so a dropped
            # link is repaired here rather than reported "no" forever.
            async with self._boundary():
                await self._client(functions.users.GetUsersRequest([types.InputUserSelf()]))
        except (SessionLost, TelegramUnavailable):
            # ping answers a question rather than doing work: an unreachable Telegram is a
            # "no", and the watchdog turns an hour of "no" into the owner warning (§17.3).
            return False
        return True

    # --- reading ---

    def on_message(self, handler: MessageHandler) -> None:
        self._message_handlers.append(handler)

    def on_session_lost(self, handler: LostHandler) -> None:
        self._lost_handlers.append(handler)

    async def _on_new_message(self, event: Any) -> None:
        await self._deliver(event.message)

    async def _on_album(self, event: Any) -> None:
        # Albums arrive item by item through NewMessage too; the Album event only adds items
        # that were somehow not delivered (it collates across data centres). The account's own
        # messages are delivered too, flagged ``is_outgoing``: intake ignores them in groups
        # only, since in a channel every post is a candidate (SPEC "Channels and groups").
        for msg in event.messages:
            await self._deliver(msg)

    async def _deliver(self, msg: Any) -> None:
        if not self._message_handlers or not isinstance(msg, types.Message):
            return
        key = (utils.get_peer_id(msg.peer_id), msg.id)
        if key in self._delivered:
            return
        self._delivered[key] = None
        while len(self._delivered) > _DELIVERED_MEMORY:
            self._delivered.popitem(last=False)
        token = _in_handler.set(True)
        try:
            entity = msg.chat
            if entity is None:
                try:
                    async with self._boundary():
                        entity = await msg.get_chat()
                except CuratorError as exc:
                    log.warning("chat of message %s unavailable: %s", key, exc)
                    return
            cached = self._chat_cache.get(key[0])
            info = chat_info_from_entity(
                entity,
                archived=cached.archived if cached else False,
                muted_until=cached.muted_until if cached else None,
            )
            if info is None:
                return
            incoming = message_to_incoming(msg, self._remember(info), now=self._now())
            if incoming is None:
                return
            for handler in list(self._message_handlers):
                try:
                    await handler(incoming)
                except Exception:
                    # An intake failure is logged, never allowed to kill the update loop.
                    log.exception("message handler failed for %s/%s", key[0], key[1])
        finally:
            _in_handler.reset(token)

    async def list_chats(self) -> list[ChatInfo]:
        found: dict[int, ChatInfo] = {}
        async with self._boundary():
            for folder in (0, 1):
                async for dialog in self._client.iter_dialogs(folder=folder):
                    info = chat_info_from_dialog(dialog, now=self._now())
                    if info is not None:
                        found[info.id] = self._remember(info)
        log.info("dialog sweep: %d channels and groups", len(found))
        return list(found.values())

    async def resolve_chat(self, ref: str | int) -> ChatInfo:
        kind, value = parse_chat_ref(ref)
        if kind == "invite":
            entity = await self._check_invite(str(value))
        else:
            entity = await self._entity(value)
        if not isinstance(entity, types.Channel | types.Chat):
            raise ChatGone(f"{ref} is not a channel or group")
        try:
            peer = await self._input_peer(utils.get_peer_id(entity))
        except ChatGone:
            archived, muted = False, None
        else:
            archived, muted = await self._dialog_state(peer)
        info = chat_info_from_entity(entity, archived=archived, muted_until=muted)
        if info is None:
            raise ChatGone(f"the account is no longer in {ref}")
        return self._remember(info)

    async def _check_invite(self, invite_hash: str) -> Any:
        """``messages.checkChatInvite`` only: it never joins. Anything but "already a member"
        is reported as such, so the owner joins by hand and sends the link again."""
        async with self._boundary():
            result = await self._client(functions.messages.CheckChatInviteRequest(invite_hash))
        if isinstance(result, types.ChatInviteAlready):
            return result.chat
        raise NotAllowed("not_a_member", "the account is not a member of that chat")

    def history(
        self, chat_id: int, *, since: datetime, limit: int | None = None
    ) -> AsyncIterator[IncomingMessage]:
        cap = HISTORY_LIMIT if limit is None else limit

        async def gen() -> AsyncIterator[IncomingMessage]:
            peer = await self._input_peer(chat_id)
            chat = await self._chat_info(chat_id, peer)
            n = 0
            async with self._boundary():
                async for msg in self._client.iter_messages(
                    peer, limit=cap, offset_date=since, reverse=True, wait_time=1
                ):
                    incoming = message_to_incoming(msg, chat, now=self._now())
                    if incoming is None or incoming.date <= since:
                        continue
                    n += 1
                    yield incoming
            log.info("history: %d messages read from %s", n, chat_id)

        return gen()

    async def _chat_info(self, chat_id: int, peer: Any) -> ChatInfo:
        cached = self._chat_cache.get(chat_id)
        if cached is not None:
            return cached
        entity = await self._entity(peer)
        info = chat_info_from_entity(entity)
        if info is None:
            raise ChatGone(f"chat {chat_id} is not a channel or group the account is in")
        return self._remember(info)

    async def get_views(self, chat_id: int, message_ids: Sequence[int]) -> dict[int, int]:
        peer = await self._input_peer(chat_id)
        if not isinstance(peer, types.InputPeerChannel):
            return {}
        ids = list(dict.fromkeys(message_ids))
        out: dict[int, int] = {}
        pending = [ids[i : i + VIEWS_BATCH] for i in range(0, len(ids), VIEWS_BATCH)]
        first = True
        while pending:
            batch = pending.pop(0)
            if not first:
                await asyncio.sleep(1)
            first = False
            try:
                async with self._boundary():
                    result = await self._client(
                        functions.messages.GetMessagesViewsRequest(
                            peer=peer, id=batch, increment=False
                        )
                    )
            except NotAllowed as exc:
                if "MSG_ID_INVALID" not in str(exc):
                    raise
                if len(batch) == 1:
                    log.debug("views: message %s/%s is gone", chat_id, batch[0])
                    continue
                half = len(batch) // 2
                pending[:0] = [batch[:half], batch[half:]]
                continue
            for mid, item in zip(batch, result.views, strict=False):
                if item.views is not None:
                    out[mid] = item.views
        return out

    async def find_message(
        self,
        chat_id: int,
        *,
        contains: str | None = None,
        fwd_of: tuple[int, int] | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> int | None:
        self._require_owned(chat_id)
        peer = await self._input_peer(chat_id)
        async with self._boundary():
            async for msg in self._client.iter_messages(peer, limit=limit):
                if not isinstance(msg, types.Message):
                    continue
                if since is not None and msg.date is not None and msg.date <= since:
                    break
                if contains is not None and not _message_contains(msg, contains):
                    continue
                if fwd_of is not None and forward_origin(msg.fwd_from) != fwd_of:
                    continue
                return msg.id
        return None

    async def find_media(
        self,
        chat_id: int,
        *,
        copies_of: tuple[int, Sequence[int]],
        after_id: int | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[int]:
        self._require_owned(chat_id)
        staging_chat, staging_ids = copies_of
        self._require_owned(staging_chat)
        peer = await self._input_peer(chat_id)
        staging = await self._input_peer(staging_chat)
        originals = await self._fetch(staging, list(staging_ids))
        keys = [media_key(m.media) if m is not None else None for m in originals]
        if not keys or any(k is None for k in keys):
            return []
        bare: list[Any] = []
        async with self._boundary():
            async for msg in self._client.iter_messages(peer, limit=limit):
                if not isinstance(msg, types.Message):
                    continue
                if since is not None and msg.date is not None and msg.date <= since:
                    break
                if after_id is not None and msg.id <= after_id:
                    break
                if msg.media is not None and not msg.message and msg.fwd_from is None:
                    bare.append(msg)
        bare.reverse()  # oldest first
        found: list[int] = []
        for key in keys:
            match = next(
                (
                    m.id
                    for m in bare
                    if (not found or m.id > found[-1]) and media_key(m.media) == key
                ),
                None,
            )
            if match is None:
                return []
            found.append(match)
        return found

    # --- writes into owned chats ---

    async def create_channel(self, title: str, about: str = "") -> ChatInfo:
        async with self._boundary():
            updates = await self._client(
                functions.channels.CreateChannelRequest(
                    title=title, about=about, broadcast=True, megagroup=False
                )
            )
        channel = next((c for c in updates.chats if isinstance(c, types.Channel)), None)
        info = chat_info_from_entity(channel) if channel is not None else None
        if info is None:
            raise ChatGone("Telegram returned no channel for the create request")
        log.info("created private channel %r (%s)", title, info.id)
        return self._remember(info)

    async def rename_channel(self, chat_id: int, title: str) -> None:
        self._require_owned(chat_id)
        peer = await self._input_peer(chat_id)
        async with self._boundary():
            await self._client(
                functions.channels.EditTitleRequest(utils.get_input_channel(peer), title)
            )
        cached = self._chat_cache.get(chat_id)
        if cached is not None:
            self._chat_cache[chat_id] = replace(cached, title=title)

    async def add_bot_admin(self, chat_id: int, bot_username: str) -> None:
        self._require_owned(chat_id)
        peer = await self._input_peer(chat_id)
        bot = await self._input_peer_for(bot_username)
        rights = types.ChatAdminRights(post_messages=True, edit_messages=True)
        async with self._boundary():
            await self._client(
                functions.channels.EditAdminRequest(
                    channel=utils.get_input_channel(peer),
                    user_id=utils.get_input_user(bot),
                    admin_rights=rights,
                    rank="",
                )
            )
        log.info("%s is now an admin of %s (post + edit)", bot_username, chat_id)

    async def _input_peer_for(self, username: str) -> Any:
        handle = username if username.startswith("@") else f"@{username}"
        try:
            async with self._boundary():
                return await self._client.get_input_entity(handle)
        except ValueError as exc:
            raise ChatGone(f"{handle} cannot be found: {exc}") from exc

    async def copy_media(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        """Copy by reference; one source album goes out as one sendMultiMedia call, anything
        else one message at a time, ``COPY_ITEM_GAP_SECONDS`` apart, so the account never
        bursts into the staging channel (§9.4 pacing, §14)."""
        self._require_owned(to_chat_id)
        source = await self._input_peer(from_chat_id)
        target = await self._input_peer(to_chat_id)
        ids = list(message_ids)
        messages = await self._fetch(source, ids)
        for mid, msg in zip(ids, messages, strict=False):
            self._check_copyable(from_chat_id, mid, msg)

        async def send(batch: list[int], media: list[Any]) -> list[int]:
            # The account's only upload-like call: the media goes by reference (id, hash, file
            # reference), nothing is downloaded. File references expire; one re-fetch gets
            # fresh ones (§5), a second failure means the media is unavailable.
            for attempt in (1, 2):
                try:
                    async with self._boundary():
                        sent = await self._client.send_file(
                            target, file=media if len(media) > 1 else media[0]
                        )
                    break
                except TypeError as exc:
                    raise MediaUnavailable(f"media cannot be re-sent: {exc}") from exc
                except MediaUnavailable:
                    if attempt == 2:
                        raise
                    fresh = await self._fetch(source, batch)
                    if any(f is None or not copyable_media(f.media) for f in fresh):
                        raise
                    media = [f.media for f in fresh]
            sent_list = sent if isinstance(sent, list) else [sent]
            out = [m.id for m in sent_list if m is not None]
            if len(out) != len(batch):
                raise MediaUnavailable(
                    f"chat {from_chat_id}: {len(out)} of {len(batch)} media copies came back"
                )
            return out

        if len(ids) > 1 and _one_album(messages):
            return await send(ids, [m.media for m in messages])
        copied: list[int] = []
        for i, (mid, msg) in enumerate(zip(ids, messages, strict=False)):
            if i:
                await asyncio.sleep(COPY_ITEM_GAP_SECONDS)
            copied += await send([mid], [msg.media])
        return copied

    @staticmethod
    def _check_copyable(chat_id: int, mid: int, msg: Any) -> None:
        if msg is None:
            raise MediaUnavailable(f"message {chat_id}/{mid} is gone")
        if msg.noforwards or getattr(msg.chat, "noforwards", False):
            raise ForwardsRestricted(f"chat {chat_id} forbids saving content")
        if not copyable_media(msg.media):
            raise MediaUnavailable(f"message {chat_id}/{mid} has no copyable media")

    async def _fetch(self, peer: Any, ids: list[int]) -> list[Any]:
        async with self._boundary():
            result = await self._client.get_messages(peer, ids=ids)
        return list(result)

    async def forward(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        self._require_owned(to_chat_id)
        source = await self._input_peer(from_chat_id)
        target = await self._input_peer(to_chat_id)
        async with self._boundary():
            sent = await self._client.forward_messages(target, list(message_ids), from_peer=source)
        ids = [m.id for m in sent if m is not None]
        if message_ids and not ids:
            # Telegram forwarded nothing: the source messages were deleted since they were read.
            raise MediaUnavailable(f"the source message(s) {list(message_ids)} are gone")
        return ids

    def register_owned(self, chat_id: int) -> None:
        self._owned.add(chat_id)

    # --- subscriptions (never on an owned chat) ---

    async def mute(self, chat_id: int, until: datetime | None) -> None:
        self._require_not_owned(chat_id)
        peer = await self._input_peer(chat_id)
        async with self._boundary():
            await self._client(
                functions.account.UpdateNotifySettingsRequest(
                    peer=types.InputNotifyPeer(peer),
                    settings=types.InputPeerNotifySettings(mute_until=mute_timestamp(until)),
                )
            )
        log.info("muted %s until %s", chat_id, until.isoformat() if until else "now (unmuted)")

    async def set_archived(self, chat_id: int, archived: bool) -> None:
        self._require_not_owned(chat_id)
        peer = await self._input_peer(chat_id)
        async with self._boundary():
            await self._client(
                functions.folders.EditPeerFoldersRequest(
                    [types.InputFolderPeer(peer, folder_id=1 if archived else 0)]
                )
            )
        log.info("%s %s", "archived" if archived else "unarchived", chat_id)

    async def leave(self, chat_id: int) -> None:
        self._require_not_owned(chat_id)
        try:
            entity = await self._entity(chat_id)
        except ChatGone:
            # CHANNEL_PRIVATE and friends: the account can no longer see the chat, which is
            # what leaving would achieve.
            log.info("leave %s: already gone", chat_id)
            self._chat_cache.pop(chat_id, None)
            return
        if isinstance(entity, types.ChannelForbidden | types.ChatForbidden):
            # Kicked or banned: Telegram only returns a stub, and there is nothing to leave.
            log.info("leave %s: no longer accessible, already left", chat_id)
            self._chat_cache.pop(chat_id, None)
            return
        if not isinstance(entity, types.Channel | types.Chat):
            raise NotAllowed("other", f"{chat_id} is not a channel or group")
        if entity.creator:
            raise NotAllowed("creator", "the account created this chat and cannot leave it")
        bare, peer_type = utils.resolve_id(chat_id)
        if peer_type is types.PeerChannel:
            request: Any = functions.channels.LeaveChannelRequest(utils.get_input_channel(entity))
        else:
            request = functions.messages.DeleteChatUserRequest(
                chat_id=bare, user_id=types.InputUserSelf()
            )
        try:
            async with self._boundary():
                await self._client(request)
        except ChatGone:
            log.info("leave %s: already gone", chat_id)
        except NotAllowed as exc:
            if "USER_NOT_PARTICIPANT" not in str(exc):
                raise
            log.info("leave %s: already left", chat_id)
        self._chat_cache.pop(chat_id, None)
        log.info("left %s", chat_id)

    # --- folders (dialog filters; folders.py is the only caller) ---

    async def _dialog_filters(self) -> list[Any]:
        async with self._boundary():
            result = await self._client(functions.messages.GetDialogFiltersRequest())
        return list(result.filters)

    async def list_folders(self) -> list[tuple[int, str]]:
        """Read only: listing a folder never makes it writable (that is ``register_own_folder``
        or ``save_folder(None, ...)``), so the guard below still protects the user's folders."""
        out: list[tuple[int, str]] = []
        for item in await self._dialog_filters():
            if isinstance(item, types.DialogFilter | types.DialogFilterChatlist):
                out.append((item.id, item.title.text))
        return out

    async def get_folder(self, folder_id: int) -> list[int] | None:
        """Every chat in the folder: the pinned ones first, then the included ones. Telegram
        keeps the two lists disjoint (pinning a chat inside a folder moves it from
        ``include_peers`` to ``pinned_peers``), so reading only ``include_peers`` would miss
        a pinned chat and make every sync rewrite the folder."""
        for item in await self._dialog_filters():
            if isinstance(item, types.DialogFilter) and item.id == folder_id:
                ids = [utils.get_peer_id(p) for p in (*item.pinned_peers, *item.include_peers)]
                return list(dict.fromkeys(ids))
        return None

    def register_own_folder(self, folder_id: int) -> None:
        """Mark a folder id as the curator's own (an id stored in ``kv folders.*``), which makes
        ``save_folder``/``delete_folder`` accept it after a restart. Ids created by
        ``save_folder(None, ...)`` are registered by it already."""
        self._own_folders.add(folder_id)

    async def save_folder(self, folder_id: int | None, title: str, chat_ids: Sequence[int]) -> int:
        if not chat_ids:
            raise ValueError("a folder needs at least one chat (FILTER_INCLUDE_EMPTY)")
        if len(title) > FOLDER_TITLE_MAX:
            raise ValueError(f"a folder title is at most {FOLDER_TITLE_MAX} characters")
        if folder_id is not None and folder_id not in self._own_folders:
            raise NotOwnedError(f"folder {folder_id} is not one of the curator's folders")
        filters = await self._dialog_filters()
        taken = {item.id for item in filters if hasattr(item, "id")}
        existing = next(
            (f for f in filters if isinstance(f, types.DialogFilter) and f.id == folder_id), None
        )
        if folder_id is None:
            folder_id = next((i for i in FOLDER_ID_RANGE if i not in taken), None)
            if folder_id is None:
                raise FolderLimit("no free folder id in 2..255")
        # A chat pinned inside the folder stays pinned (and only there: Telegram keeps the
        # pinned and included lists disjoint); every other chat goes to include_peers.
        wanted = set(chat_ids)
        pinned = [
            p for p in (existing.pinned_peers if existing else ()) if utils.get_peer_id(p) in wanted
        ]
        pinned_ids = {utils.get_peer_id(p) for p in pinned}
        peers = []
        for chat_id in dict.fromkeys(chat_ids):
            if chat_id in pinned_ids:
                continue
            try:
                peers.append(await self._input_peer(chat_id))
            except ChatGone:
                log.warning("folder %r: chat %s cannot be resolved, left out", title, chat_id)
        if not peers and not pinned:
            raise ValueError("none of the chats could be resolved (FILTER_INCLUDE_EMPTY)")
        dialog_filter = types.DialogFilter(
            id=folder_id,
            title=types.TextWithEntities(title, []),
            pinned_peers=pinned,
            include_peers=peers,
            exclude_peers=list(existing.exclude_peers) if existing else [],
            contacts=existing.contacts if existing else None,
            non_contacts=existing.non_contacts if existing else None,
            groups=existing.groups if existing else None,
            broadcasts=existing.broadcasts if existing else None,
            bots=existing.bots if existing else None,
            exclude_muted=existing.exclude_muted if existing else None,
            exclude_read=existing.exclude_read if existing else None,
            exclude_archived=existing.exclude_archived if existing else None,
            emoticon=existing.emoticon if existing else None,
            color=existing.color if existing else None,
        )
        async with self._boundary():
            await self._client(
                functions.messages.UpdateDialogFilterRequest(id=folder_id, filter=dialog_filter)
            )
        self._own_folders.add(folder_id)
        log.info("folder %r saved with %d chats", title, len(pinned) + len(peers))
        return folder_id

    async def delete_folder(self, folder_id: int) -> None:
        """An update without a filter deletes the folder (layer 229: ``filter`` is the only
        optional field, so ``UpdateDialogFilterRequest(id=...)`` alone is the delete)."""
        if folder_id not in self._own_folders:
            raise NotOwnedError(f"folder {folder_id} is not one of the curator's folders")
        async with self._boundary():
            await self._client(functions.messages.UpdateDialogFilterRequest(id=folder_id))
        self._own_folders.discard(folder_id)
        log.info("folder %s deleted", folder_id)


def _create_private(path: Path) -> None:
    """Create ``path`` empty with mode 0600 before Telethon opens it, so a session file is
    never readable by others, whatever the umask (DESIGN §3); SQLite takes an empty file as
    an empty database."""
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)


def _message_contains(msg: Any, needle: str) -> bool:
    if needle in (msg.message or ""):
        return True
    return any(
        isinstance(e, types.MessageEntityTextUrl) and e.url == needle for e in msg.entities or ()
    )
