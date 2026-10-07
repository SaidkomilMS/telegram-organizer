"""The two Telegram gateways every module talks through (DESIGN §5).

Only ``telegram/user_client.py`` and ``telegram/bot_client.py`` import Telethon; everything
else sees these Protocols and frozen dataclasses, so the whole pipeline runs against the fakes
in ``tests/fakes.py``. The account's safety rules are structural: ``UserGateway`` has no way to
send a message, join, react, mark as read or edit a profile, and every write it does have is
guarded by the owned-chat rules written in the docstrings below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from tg_curator.domain import ChatKind, MediaKind

__all__ = [
    "Account",
    "BotCallback",
    "BotGateway",
    "BotMessage",
    "Button",
    "Buttons",
    "ChatInfo",
    "ChatKind",
    "IncomingMessage",
    "MediaKind",
    "UserGateway",
]


@dataclass(frozen=True)
class ChatInfo:
    """A channel or group as the account sees it; ``id`` is the marked id (``-100…``).

    ``is_admin`` is true only when the account's admin rights include posting.
    """

    id: int
    kind: ChatKind
    title: str
    username: str | None
    noforwards: bool
    is_creator: bool
    is_admin: bool
    archived: bool
    muted_until: datetime | None


@dataclass(frozen=True)
class Account:
    """The identity of the user account or of the bot."""

    id: int
    name: str
    username: str | None
    phone: str | None


@dataclass(frozen=True)
class IncomingMessage:
    """One message read by the account; service messages and edits never become one."""

    chat: ChatInfo
    message_id: int
    date: datetime
    text: str
    html: str | None
    sender_id: int | None
    is_outgoing: bool
    is_service: bool
    reply_to_id: int | None
    grouped_id: int | None
    media: MediaKind | None
    urls: tuple[str, ...]
    views: int | None
    forwards: int | None
    fwd_from_chat_id: int | None
    fwd_from_message_id: int | None
    noforwards: bool
    topic_id: int | None = None
    """The forum topic (or comment thread) the message belongs to; ``None`` outside forums
    and for a forum's General topic. Conversation units never cross topics (§14)."""
    is_automatic_forward: bool = False
    """Telegram's own copy of a channel post in the channel's linked discussion group (the
    root of its comment section): not a message of the group, and not a second chat carrying
    the story. Intake ignores it; the comments under it are judged on their own text."""


@dataclass(frozen=True)
class Button:
    """One inline button: ``data`` for a callback (<= 64 bytes, §11.1) or ``url`` for a link."""

    text: str
    data: str | None = None
    url: str | None = None


Buttons = list[list[Button]]
"""Inline keyboard rows: the type of every ``buttons=`` parameter, of ``Notifier.owner`` and of
``ctx.reply`` / ``ctx.edit`` (§11.1)."""


@dataclass(frozen=True)
class BotMessage:
    """A message the bot received (only the owner's private messages are ever handled)."""

    chat_id: int
    message_id: int
    sender_id: int
    text: str
    is_private: bool
    fwd_from_chat_id: int | None
    fwd_from_title: str | None
    reply_to_id: int | None
    has_media: bool = False
    """A photo, video, document ... (not a link preview): a media post without a caption
    arrives with ``text == ""``, which is "no text", not "short text"."""


@dataclass(frozen=True)
class BotCallback:
    """An inline-button press; ``data`` is the full callback payload (``"wt:42"``)."""

    query_id: str
    sender_id: int
    chat_id: int
    message_id: int
    data: str


@runtime_checkable
class UserGateway(Protocol):
    """The user account. It reads, copies media into chats the user owns, and carries out
    approved subscription actions; it never sends a message to anyone, never joins, never
    reacts, never marks as read and never edits a profile — there are no such methods.

    Guard rules (DESIGN §1), enforced by the Telethon implementation and by the fake alike:

    - Content writes (``create_channel``, ``rename_channel``, ``add_bot_admin``,
      ``copy_media``, ``forward``, ``find_message``, ``find_media`` — checked on the
      destination / the chat read) require the target chat to be registered as owned
      (``register_owned``, i.e. ``chats.role`` ``output`` or ``staging``), else
      ``NotOwnedError``.
    - Subscription writes (``mute``, ``set_archived``, ``leave``) require the target NOT to be
      owned (an output or staging channel is never muted, archived or left), else
      ``NotOwnedError``; ``leave`` additionally refuses when ``ChatInfo.is_creator``
      (``NotAllowed("creator")``). They are callable only from ``subscriptions/actions.py`` and
      ``subscriptions/folders.py``, which act only on ``proposals`` rows in state ``approved``
      (a ``leave`` reaches ``approved`` only through ``confirming``).
    - ``save_folder`` and ``delete_folder`` may only be called with a folder id the gateway
      itself created or that was registered with ``register_own_folder`` (the ids in
      ``kv folders.*``; ``save_folder`` also takes ``None`` to create); ids merely seen by
      ``list_folders``/``get_folder`` stay refused. ``folders.py`` is the only caller (§12,
      §17.2).
    - ``register_owned`` is called only by ``TopicsService.sync_from_settings()``, ``create()``,
      ``link_channel()`` and ``ensure_staging_channel()``, only for chats where
      ``resolve_chat`` reports ``is_creator`` or ``is_admin`` with post rights, and by
      ``service.py`` at start for every ``chats`` row with role ``output``/``staging`` (§13).

    Every Telethon exception is translated into the ``errors.py`` vocabulary at this boundary:
    ``FloodWait`` (waits up to 120 s are slept by the implementation itself; only longer ones
    are raised), ``NotOwnedError``, ``ForwardsRestricted``, ``MediaUnavailable``, ``ChatGone``,
    ``FolderLimit``, ``NotAllowed(reason)``, ``SessionLost(reason)``, ``LoginError(reason)``,
    and ``TelegramUnavailable`` for connection, OS-level and timeout errors and for Telegram's
    5xx answers (§17.3) — always raised to the caller, never swallowed here.
    """

    # --- session / login (no interactive prompts) ---

    async def connect(self) -> bool:
        """Open the session; ``True`` if it is already authorised.

        Never uses Telethon's ``catch_up``: updates missed while disconnected are not replayed;
        ``backfill`` is the only way to recover them. A ``user.session.pending`` left behind by
        a re-login that a crash or a restart interrupted is deleted first (unless a re-login
        of this process is under way): that login can never finish, and the file holds a
        half-made session nobody may keep.
        """
        ...

    async def disconnect(self) -> None:
        """Close the link; a re-login still pending is abandoned (``cancel_relogin``)."""
        ...

    async def reset_session(self) -> None:
        """Drop the session completely before a login when there is no working session to keep
        (a first bind, or after a loss; ``/bind`` "always starts fresh", §11.2): disconnect,
        delete ``user.session`` (and its journal) and continue with a fresh, empty session.
        Deleting the file alone is not enough: the open client keeps the old auth key in
        memory. A pending re-login is abandoned too. Sends nothing to anyone."""
        ...

    async def begin_relogin(self) -> None:
        """Start a re-login beside the working session (``/bind`` on a bound account; spec
        "every step can be repeated", "/bind sets it up again").

        A second client over ``<home>/user.session.pending`` (created ``0600``; an earlier
        pending re-login is abandoned first), with the same ``api_id``/``api_hash`` and device
        model, takes ``send_code``, ``resend_code``, ``sign_in`` and ``sign_in_password`` from
        here on, and its ``phone_code_hash`` lives on that client only; the working session
        keeps serving every other method, intake included. When ``sign_in`` (or, with 2FA,
        ``sign_in_password``) succeeds, the new session is swapped in: the old one is logged
        out and disconnected, the pending file replaces ``user.session``, the new client is
        connected with the message handlers, and ``me()`` reports the new account. Until
        then, a wrong code, a failed step or an abandoned login changes nothing. A request
        cut off by the swap is raised as ``TelegramUnavailable``, never as a session loss."""
        ...

    async def cancel_relogin(self) -> None:
        """Abandon a pending re-login: its client is disconnected and
        ``user.session.pending`` (with its journal) deleted; the working session is not
        touched. Also removes a stale pending file when no re-login is under way."""
        ...

    async def me(self) -> Account | None:
        """The bound account, ``None`` when the session is not authorised."""
        ...

    async def send_code(self, phone: str) -> None:
        """Ask Telegram for a login code; ``phone_code_hash`` is kept in process memory only
        (during a re-login, on its pending client)."""
        ...

    async def resend_code(self) -> None:
        """``auth.resendCode`` for the pending phone."""
        ...

    async def sign_in(self, code: str) -> Literal["ok", "password_needed"]:
        """Submit the login code; ``"password_needed"`` when 2FA is on. Raises ``LoginError``."""
        ...

    async def sign_in_password(self, password: str) -> None:
        """Submit the 2FA password. Raises ``LoginError("bad_password")``."""
        ...

    async def ping(self) -> bool:
        """``users.getUsers(InputUserSelf)``: the session-loss detector used by the watchdog.

        Telethon gives no callback when the session is revoked from another device: the
        update loop simply goes quiet and the next request fails, which this call provokes.
        A client whose link Telegram dropped is connected again first (``ensure_connected``),
        so ``False`` means the account really does not answer.
        """
        ...

    async def ensure_connected(self) -> None:
        """Connect again if the link to Telegram dropped and Telethon gave up reconnecting
        (§17.3: an outage must end when the network comes back, not stop the account for
        good). A no-op while connected, before the first ``connect()``, after ``disconnect()``
        and after a session loss. Every request does this first; the service is meant to call
        it on a timer as well, so live updates resume without waiting for a request. Raises
        ``TelegramUnavailable`` while Telegram is still unreachable, ``SessionLost`` when it
        no longer knows the session."""
        ...

    # --- reading ---

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register the handler for new incoming messages (service messages and edits are
        never delivered)."""
        ...

    def on_session_lost(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """Register the handler called exactly once with the reason when the session is
        lost: the gateway has then disconnected, renamed ``user.session`` to
        ``user.session.revoked-<timestamp>`` and created a fresh empty session."""
        ...

    async def list_chats(self) -> list[ChatInfo]:
        """Channels and groups only. Sweeps the main list (folder=0) AND the archive
        (folder=1) as two explicit ``iter_dialogs`` passes and unions them by id (archived
        inclusion by default is unverified), 100 per page."""
        ...

    async def resolve_chat(self, ref: str | int) -> ChatInfo:
        """id, ``@username`` or t.me link -> ``ChatInfo``. Never joins.

        Numeric ids and usernames go through ``get_entity`` / ``contacts.resolveUsername``;
        invite links (``t.me/+…``, ``t.me/joinchat/…``) use ``messages.checkChatInvite`` ONLY:
        ``ChatInviteAlready`` -> ``ChatInfo``; anything else -> ``NotAllowed("not_a_member")``
        ("join the chat in Telegram first, then send the link again"). Never
        ``ImportChatInviteRequest``, never ``JoinChannelRequest``. A chat that cannot be
        resolved raises ``ChatGone``.
        """
        ...

    def history(
        self, chat_id: int, *, since: datetime, limit: int | None = None
    ) -> AsyncIterator[IncomingMessage]:
        """Messages dated after ``since``, oldest first; ``messages.getHistory`` 100 per page,
        1 s between pages, at most 2000 messages per chat per run (the default limit)."""
        ...

    async def get_views(self, chat_id: int, message_ids: Sequence[int]) -> dict[int, int]:
        """``messages.getMessagesViews(peer, ids, increment=False)`` — NEVER ``increment=True``.

        <= 100 ids per call, one call per second; groups and chats without views return ``{}``;
        ``MSG_ID_INVALID`` entries are skipped; forwards are in the result but only views are
        returned.
        """
        ...

    async def find_message(
        self,
        chat_id: int,
        *,
        contains: str | None = None,
        fwd_of: tuple[int, int] | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> int | None:
        """Outbox reconciliation, owned chats only (``NotOwnedError`` otherwise).

        Reads the newest ``limit`` messages with ``getHistory`` (never ``messages.search``) and
        matches on entities: ``contains`` matches when it occurs in the plain text or caption
        OR equals the url of any ``MessageEntityTextUrl``; ``fwd_of=(chat_id, message_id)``
        matches ``fwd_from``; ``since`` restricts to messages dated after it. Returns the
        newest matching id.
        """
        ...

    async def find_media(
        self,
        chat_id: int,
        *,
        copies_of: tuple[int, Sequence[int]],
        after_id: int | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[int]:
        """Outbox reconciliation of the media a repost sends first, without a caption (§9.4);
        both chats must be owned (``NotOwnedError`` otherwise).

        ``copies_of=(staging_chat_id, staging_ids)`` names the staging copies the media was
        sent from. Reads the newest ``limit`` messages of ``chat_id`` with ``getHistory`` and
        considers those without text that are not forwards, dated after ``since`` and with an
        id above ``after_id``; for each staging copy in order it takes the oldest such message
        carrying the same photo or document (Telegram keeps its id when media is re-sent by
        reference). Returns those ids in order, or ``[]`` unless every copy is matched.
        """
        ...

    # --- writes: only into owned channels ---

    async def create_channel(self, title: str, about: str = "") -> ChatInfo:
        """``channels.createChannel`` with ``broadcast=True, megagroup=False`` (private, no
        username). Flood-limited by Telegram: raises ``FloodWait``; the caller
        (``TopicsService``) paces, see §8."""
        ...

    async def rename_channel(self, chat_id: int, title: str) -> None:
        """Owned channels only (``NotOwnedError``)."""
        ...

    async def add_bot_admin(self, chat_id: int, bot_username: str) -> None:
        """``channels.editAdmin`` with ``post_messages`` + ``edit_messages`` ONLY; never
        ``delete_messages``, ``add_admins``, ``invite_users``. Owned channels only."""
        ...

    async def copy_media(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        """Re-send the media of the source messages into an owned chat (never a forward);
        one new message id per source id.

        On ``FileReferenceExpiredError`` re-fetch the source messages once
        (``channels.getMessages`` / ``messages.getMessages``) and retry; on the second failure
        raise ``MediaUnavailable`` so the publisher falls back to text + link. ``noforwards``
        sources raise ``ForwardsRestricted``.
        """
        ...

    async def forward(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        """Forward the messages (a whole album) into an owned chat; one new id per source id.
        ``noforwards`` sources raise ``ForwardsRestricted``."""
        ...

    def register_owned(self, chat_id: int) -> None:
        """Mark a chat as owned (role ``output`` / ``staging``) for the guard rules above."""
        ...

    # --- subscriptions (only after the owner approved a proposal; target must NOT be owned) ---

    async def mute(self, chat_id: int, until: datetime | None) -> None:
        """``account.updateNotifySettings(InputNotifyPeer(peer),
        InputPeerNotifySettings(mute_until=<ts>))``; ``None`` -> ``mute_until=0`` (unmute);
        "forever" = ``datetime.max`` -> ``mute_until=2**31-1``. Never on an owned chat."""
        ...

    async def set_archived(self, chat_id: int, archived: bool) -> None:
        """``folders.editPeerFolders(folder_id=1 / 0)``. Never on an owned chat."""
        ...

    async def leave(self, chat_id: int) -> None:
        """``channels.leaveChannel`` for channels/supergroups,
        ``messages.deleteChatUser(InputUserSelf)`` for basic groups, never ``delete_dialog``;
        ``NotAllowed("creator")`` when ``is_creator``. Never on an owned chat."""
        ...

    async def list_folders(self) -> list[tuple[int, str]]:
        """``messages.getDialogFilters``, read only: ``(id, title)`` pairs. Listing a folder
        never makes it writable for ``save_folder``/``delete_folder``."""
        ...

    async def get_folder(self, folder_id: int) -> list[int] | None:
        """Every chat in the folder (``pinned_peers`` first, then ``include_peers``, which
        Telegram keeps disjoint), ``None`` if the folder is gone."""
        ...

    def register_own_folder(self, folder_id: int) -> None:
        """Mark a folder id stored in ``kv folders.*`` as the curator's own, so
        ``save_folder``/``delete_folder`` accept it (after a restart the gateway knows only
        the ids it created itself). Called by ``folders.py`` only, and only for a stored id
        whose folder it verified is still the curator's."""
        ...

    async def save_folder(self, folder_id: int | None, title: str, chat_ids: Sequence[int]) -> int:
        """``messages.updateDialogFilter``; ``None`` = create with a free id in 2..255.

        ``title`` <= 12 chars (at layer 229 ``DialogFilter.title`` is a ``TextWithEntities``);
        raises ``FolderLimit`` on ``DIALOG_FILTERS_TOO_MUCH``, ``ValueError`` on empty
        ``chat_ids`` (``FILTER_INCLUDE_EMPTY``); callable only with ids this gateway returned
        (``kv folders.*``) or ``None``. Returns the folder id.
        """
        ...

    async def delete_folder(self, folder_id: int) -> None:
        """``messages.updateDialogFilter(id=folder_id)`` with no filter, which deletes it.

        Telegram refuses a folder without chats, so this is the only way to "empty" one of the
        curator's folders (§17.2). Guarded like ``save_folder``: only ids this gateway
        returned (``kv folders.*``), else ``NotOwnedError``; a user's folder is never touched.
        """
        ...


@runtime_checkable
class BotGateway(Protocol):
    """The bot: it does all posting into topic channels and all talking to the owner.

    Telethon exceptions are translated at this boundary like the account's (``FloodWait``,
    ``BotCannotPost``, ``ChatGone``, ...), including ``TelegramUnavailable`` for connection,
    OS-level and timeout errors and 5xx answers (§17.3), which is always raised, never
    swallowed.
    """

    async def start(self) -> Account:
        """Connect and return the bot identity (stored in ``rt.bot_account`` and in
        ``kv bot.id`` / ``bot.username``)."""
        ...

    async def stop(self) -> None: ...

    async def ensure_connected(self) -> None:
        """Connect the started bot again if the link to Telegram dropped and Telethon gave
        up reconnecting; a no-op while connected or before ``start()``/after ``stop()``.
        Every request does this first; the service is meant to call it on a timer as well,
        so the owner's commands are heard again without waiting for an outgoing message.
        When Telegram no
        longer knows the bot's session key, a new session is signed in with the same token.
        Raises ``TelegramUnavailable`` while Telegram is still unreachable."""
        ...

    def on_message(self, handler: Callable[[BotMessage], Awaitable[None]]) -> None: ...

    def on_callback(self, handler: Callable[[BotCallback], Awaitable[None]]) -> None: ...

    async def send_text(
        self,
        chat_id: int,
        html: str,
        *,
        buttons: Buttons | None = None,
        reply_to: int | None = None,
        silent: bool = False,
    ) -> int:
        """Send an HTML message; returns its id. ``BotCannotPost`` when the bot is not an
        admin with Post Messages there, ``ChatGone`` when the chat is unreachable."""
        ...

    async def send_copy(
        self,
        from_chat_id: int,
        message_ids: Sequence[int],
        to_chat_id: int,
        *,
        caption_html: str | None = None,
        buttons: Buttons | None = None,
    ) -> list[int]:
        """Fetch the messages by id from ``from_chat_id`` (``channels.getMessages``; the bot
        must be an admin there, i.e. the staging channel) and re-send their media with
        ``messages.sendMedia`` / ``sendMultiMedia`` — never a forward.

        ``caption_html`` and ``buttons`` are applied to single media only; albums never carry
        buttons (Telegram forbids reply markup on grouped media) and the caller passes
        ``buttons=None``. One new message id per source id.
        """
        ...

    async def edit_text(
        self, chat_id: int, message_id: int, html: str, *, buttons: Buttons | None = None
    ) -> None:
        """Edit the text; on a media message edits the caption (the media stays);
        ``MESSAGE_NOT_MODIFIED`` is swallowed. Only the bot's own messages can be edited."""
        ...

    async def edit_buttons(self, chat_id: int, message_id: int, buttons: Buttons | None) -> None:
        """Replace (or with ``None`` remove) the inline keyboard of one of the bot's messages."""
        ...

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        """Private owner chat only: the owner's own secret-bearing messages (login code, 2FA
        password, API key). The curator never deletes messages anywhere else."""
        ...

    async def answer_callback(
        self, query_id: str, text: str | None = None, *, alert: bool = False
    ) -> None:
        """Acknowledge a button press; called FIRST in every callback handler because
        Telegram invalidates the query after a few seconds (§11.1)."""
        ...

    async def can_post(self, chat_id: int) -> bool:
        """Whether the bot is an admin with Post Messages in that channel
        (``channels.getChannels`` -> ``admin_rights``, ``left``)."""
        ...
