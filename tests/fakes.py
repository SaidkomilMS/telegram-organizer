"""Test doubles that are part of the build contract (DESIGN §2 "Wave 0", §16).

Every fake records its calls, returns increasing ids and can be told to fail the next call of
a method, so a module test can assert exactly what reached Telegram and how a failure was
handled. The two gateway fakes enforce the §1 guard rules the same way the Telethon
implementation does, so a module that writes where it should not fails its own tests. When
they share one ``FakeWorld`` (as the ``rt`` fixture arranges) the account sees what the bot
sent, which is what ``reconcile()`` needs.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np

from tg_curator.domain import Example, MediaKind, Topic, TopicName, TopicScore
from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    FolderLimit,
    ForwardsRestricted,
    LoginError,
    MediaUnavailable,
    NotAllowed,
    NotOwnedError,
)
from tg_curator.telegram.gateway import (
    Account,
    BotCallback,
    BotMessage,
    Buttons,
    ChatInfo,
    IncomingMessage,
)
from tg_curator.textutil import normalise

START = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
OWNER_ID = 1001
BOT_ACCOUNT = Account(id=7_000_000_001, name="curator bot", username="curator_test_bot", phone=None)
USER_ACCOUNT = Account(id=OWNER_ID, name="Test Owner", username="test_owner", phone="+998901234567")

# The provisional built-in categories named in the settings template (ml/categories.py owns
# the real list); the fake classifier scores them uniformly low unless told otherwise.
PROVISIONAL_CATEGORIES = (
    "tech", "science", "finance", "crypto", "politics", "world", "sport", "health", "education",
    "culture", "real_estate", "jobs", "travel", "auto", "society", "disaster", "religion",
    "lifestyle", "humor", "ads",
)  # fmt: skip

_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r"""href=["']([^"']*)["']""")


def plain_text(markup: str | None) -> str:
    """The text Telegram would show for an HTML message: tags removed, entities decoded."""
    if not markup:
        return ""
    return html_lib.unescape(_TAG_RE.sub("", markup))


def hrefs(markup: str | None) -> list[str]:
    """The urls of the ``<a href>`` entities of an HTML message."""
    return _HREF_RE.findall(markup or "")


# --- clock -----------------------------------------------------------------------------------


class FakeClock:
    """A clock that moves only when a test says so."""

    def __init__(self, start: datetime = START) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs an aware datetime")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta | float) -> datetime:
        """Move forward by a ``timedelta`` or a number of seconds; returns the new time."""
        if not isinstance(delta, timedelta):
            delta = timedelta(seconds=delta)
        self._now = self._now + delta
        return self._now

    def set(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("FakeClock needs an aware datetime")
        self._now = at


# --- the shared message store ----------------------------------------------------------------


@dataclass
class FakeMessage:
    """One message in a fake chat: sent by the bot, by the account, or read from a source."""

    chat_id: int
    message_id: int
    sender: Literal["bot", "user", "source"]
    date: datetime
    text: str = ""
    html: str | None = None
    media: MediaKind | None = None
    buttons: Buttons | None = None
    fwd_of: tuple[int, int] | None = None
    copied_from: tuple[int, int] | None = None
    reply_to: int | None = None
    silent: bool = False
    noforwards: bool = False
    edits: int = 0

    @property
    def hrefs(self) -> list[str]:
        return hrefs(self.html)


class FakeWorld:
    """Messages of every fake chat plus the per-chat id counters, shared by both gateways."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock or FakeClock()
        self.messages: dict[int, list[FakeMessage]] = defaultdict(list)
        self.bot_blocked: set[int] = set()  # chats the bot cannot post into
        self._next_id: dict[int, int] = defaultdict(lambda: 1)

    def next_id(self, chat_id: int) -> int:
        mid = self._next_id[chat_id]
        self._next_id[chat_id] = mid + 1
        return mid

    def add(
        self,
        chat_id: int,
        *,
        sender: Literal["bot", "user", "source"],
        message_id: int | None = None,
        html: str | None = None,
        text: str | None = None,
        media: MediaKind | None = None,
        buttons: Buttons | None = None,
        fwd_of: tuple[int, int] | None = None,
        copied_from: tuple[int, int] | None = None,
        reply_to: int | None = None,
        silent: bool = False,
        noforwards: bool = False,
        date: datetime | None = None,
    ) -> FakeMessage:
        if message_id is None:
            message_id = self.next_id(chat_id)
        elif message_id >= self._next_id[chat_id]:
            self._next_id[chat_id] = message_id + 1
        msg = FakeMessage(
            chat_id=chat_id,
            message_id=message_id,
            sender=sender,
            date=date or self.clock.now(),
            text=plain_text(html) if text is None else text,
            html=html,
            media=media,
            buttons=buttons,
            fwd_of=fwd_of,
            copied_from=copied_from,
            reply_to=reply_to,
            silent=silent,
            noforwards=noforwards,
        )
        self.messages[chat_id].append(msg)
        return msg

    def get(self, chat_id: int, message_id: int) -> FakeMessage | None:
        for msg in self.messages.get(chat_id, ()):
            if msg.message_id == message_id:
                return msg
        return None

    def sent(self, chat_id: int) -> list[FakeMessage]:
        """The messages of a chat in send order (a copy)."""
        return list(self.messages.get(chat_id, ()))


class _Recorder:
    """Call recording and failure injection shared by the two gateway fakes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._failures: dict[str, list[Exception]] = defaultdict(list)

    def fail_next(self, method: str, exc: Exception) -> None:
        """Raise ``exc`` from the next call of ``method`` (queued: call twice for two).

        Any ``errors.py`` class can be injected, ``TelegramUnavailable`` included: that is how
        a test simulates an outage, since the fakes never fail on their own.
        """
        self._failures[method].append(exc)

    def calls_of(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def _pop_failure(self, method: str) -> Exception | None:
        queue = self._failures.get(method)
        return queue.pop(0) if queue else None

    def _call(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))
        failure = self._pop_failure(method)
        if failure is not None:
            raise failure


# --- the user account ------------------------------------------------------------------------


@dataclass
class FakeRelogin:
    """The second client of a re-login (``begin_relogin``): only the phone a code went to."""

    phone: str | None = None


class FakeUserGateway(_Recorder):
    """``UserGateway`` double. ``chats`` is what ``list_chats()`` returns; ``owned`` is the
    registry behind the §1 guard rules; ``history_data`` feeds ``history()``; ``views`` feeds
    ``get_views``; ``invites`` maps invite links to chat ids the account is already in.

    A re-login (``begin_relogin``) runs on ``relogin`` beside the working session: the login
    steps go there and obey the same ``valid_code``/``valid_password``/``password_needed``,
    while ``authorised`` and ``account`` stay untouched until a sign-in completes. Then the
    new session is swapped in: ``account`` becomes ``relogin_account`` (default: the same
    account) and ``session_generation`` goes up, which is how a test sees the old session go.
    """

    def __init__(
        self,
        world: FakeWorld | None = None,
        *,
        authorised: bool = True,
        account: Account | None = USER_ACCOUNT,
    ) -> None:
        super().__init__()
        self.world = world or FakeWorld()
        self.chats: list[ChatInfo] = []
        self.owned: set[int] = set()
        self.history_data: dict[int, list[IncomingMessage]] = defaultdict(list)
        self.views: dict[tuple[int, int], int] = {}
        self.invites: dict[str, int] = {}
        self.folders: dict[int, tuple[str, list[int]]] = {}
        self.folder_limit = 10
        self.created: list[ChatInfo] = []
        self.admins: set[tuple[int, str]] = set()
        self.mutes: dict[int, datetime | None] = {}
        self.archived: dict[int, bool] = {}
        self.left: list[int] = []
        self.deleted_sources: set[tuple[int, int]] = set()  # deleted at the source since
        self.connected = False
        self.authorised = authorised
        self.account = account
        self.alive = True
        self.pending_phone: str | None = None
        self.codes_sent: list[str] = []
        self.valid_code: str | None = None
        self.valid_password: str | None = None
        self.password_needed = False
        self.relogin: FakeRelogin | None = None
        self.relogin_account: Account | None = None
        self.session_generation = 1
        self._message_handlers: list[Callable[[IncomingMessage], Awaitable[None]]] = []
        self._lost_handlers: list[Callable[[str], Awaitable[None]]] = []
        self._own_folders: set[int] = set()
        self._next_chat = -1_002_000_000_000

    # --- test helpers ---

    def add_chat(self, chat: ChatInfo) -> ChatInfo:
        self.chats = [c for c in self.chats if c.id != chat.id] + [chat]
        return chat

    def chat(self, chat_id: int) -> ChatInfo | None:
        for chat in self.chats:
            if chat.id == chat_id:
                return chat
        return None

    def seed(self, msg: IncomingMessage) -> FakeMessage:
        """Put a source message into the chat's history (and the world) without delivering it."""
        self.history_data[msg.chat.id].append(msg)
        fwd = None
        if msg.fwd_from_chat_id is not None and msg.fwd_from_message_id is not None:
            fwd = (msg.fwd_from_chat_id, msg.fwd_from_message_id)
        return self.world.add(
            msg.chat.id,
            sender="source",
            message_id=msg.message_id,
            html=msg.html,
            text=msg.text,
            media=msg.media,
            fwd_of=fwd,
            reply_to=msg.reply_to_id,
            noforwards=msg.noforwards or msg.chat.noforwards,
            date=msg.date,
        )

    async def deliver(self, msg: IncomingMessage) -> None:
        """Push a message through every ``on_message`` handler (it is seeded first)."""
        self.seed(msg)
        for handler in list(self._message_handlers):
            await handler(msg)

    async def lose_session(self, reason: str = "revoked") -> None:
        """What the real gateway does on a session error: drop authorisation, tell once."""
        self.authorised = False
        self.connected = False
        for handler in list(self._lost_handlers):
            await handler(reason)

    def own_folder(self, folder_id: int, title: str, chat_ids: Sequence[int]) -> None:
        """Pre-seed a folder as one this gateway created (as if stored in ``kv folders.*``)."""
        self.folders[folder_id] = (title, list(chat_ids))
        self._own_folders.add(folder_id)

    def _replace_chat(self, chat_id: int, **changes: Any) -> None:
        self.chats = [replace(c, **changes) if c.id == chat_id else c for c in self.chats]

    # --- session / login ---

    async def connect(self) -> bool:
        self._call("connect")
        self.connected = True
        return self.authorised

    async def disconnect(self) -> None:
        self._call("disconnect")
        self.connected = False
        self.relogin = None

    async def reset_session(self) -> None:
        """The real gateway deletes ``user.session`` and starts an empty client."""
        self._call("reset_session")
        self.connected = False
        self.authorised = False
        self.pending_phone = None
        self.relogin = None

    async def begin_relogin(self) -> None:
        self._call("begin_relogin")
        self.relogin = FakeRelogin()

    async def cancel_relogin(self) -> None:
        self._call("cancel_relogin")
        self.relogin = None

    async def me(self) -> Account | None:
        self._call("me")
        return self.account if self.authorised else None

    async def send_code(self, phone: str) -> None:
        self._call("send_code", phone=phone)
        if self.relogin is not None:
            self.relogin.phone = phone
        else:
            self.pending_phone = phone
        self.codes_sent.append(phone)

    async def resend_code(self) -> None:
        self._call("resend_code")
        phone = self._login_phone()
        if phone is None:
            raise LoginError("other", "no code was requested")
        self.codes_sent.append(phone)

    async def sign_in(self, code: str) -> Literal["ok", "password_needed"]:
        self._call("sign_in", code=code)
        if self._login_phone() is None:
            raise LoginError("other", "no code was requested")
        if self.valid_code is not None and code != self.valid_code:
            raise LoginError("bad_code")
        if self.password_needed:
            return "password_needed"
        self._signed_in()
        return "ok"

    async def sign_in_password(self, password: str) -> None:
        self._call("sign_in_password", password=password)
        if self.valid_password is not None and password != self.valid_password:
            raise LoginError("bad_password")
        self._signed_in()

    def _login_phone(self) -> str | None:
        return self.relogin.phone if self.relogin is not None else self.pending_phone

    def _signed_in(self) -> None:
        if self.relogin is not None:  # the swap: the new session replaces the working one
            self.relogin = None
            self.account = self.relogin_account or self.account
            self.session_generation += 1
        self.authorised = True

    async def ping(self) -> bool:
        self._call("ping")
        return self.alive and self.authorised

    async def ensure_connected(self) -> None:
        """The fake never drops its link; an outage is injected with ``fail_next``."""
        self._call("ensure_connected")

    # --- reading ---

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        self._message_handlers.append(handler)

    def on_session_lost(self, handler: Callable[[str], Awaitable[None]]) -> None:
        self._lost_handlers.append(handler)

    async def list_chats(self) -> list[ChatInfo]:
        self._call("list_chats")
        return list(self.chats)

    async def resolve_chat(self, ref: str | int) -> ChatInfo:
        self._call("resolve_chat", ref=ref)
        if isinstance(ref, int) or ref.lstrip("-").isdigit():
            chat = self.chat(int(ref))
            if chat is None:
                raise ChatGone(f"chat {ref} cannot be found")
            return chat
        text = ref.strip()
        if "t.me/" in text or "telegram.me/" in text:
            path = text.split(".me/", 1)[1].strip("/")
            if path.startswith("+") or path.startswith("joinchat/"):
                chat_id = self.invites.get(text) or self.invites.get(path)
                chat = self.chat(chat_id) if chat_id is not None else None
                if chat is None:
                    raise NotAllowed("not_a_member", "the account is not a member of that chat")
                return chat
            text = path.split("/", 1)[0]
        username = text.lstrip("@").lower()
        for chat in self.chats:
            if chat.username and chat.username.lower() == username:
                return chat
        raise ChatGone(f"chat {ref} cannot be found")

    def history(
        self, chat_id: int, *, since: datetime, limit: int | None = None
    ) -> AsyncIterator[IncomingMessage]:
        self.calls.append(("history", {"chat_id": chat_id, "since": since, "limit": limit}))
        failure = self._pop_failure("history")
        messages = sorted(self.history_data.get(chat_id, ()), key=lambda m: (m.date, m.message_id))

        async def gen() -> AsyncIterator[IncomingMessage]:
            if failure is not None:
                raise failure
            n = 0
            for msg in messages:
                if msg.date <= since:
                    continue
                if limit is not None and n >= limit:
                    return
                n += 1
                yield msg

        return gen()

    async def get_views(self, chat_id: int, message_ids: Sequence[int]) -> dict[int, int]:
        self._call("get_views", chat_id=chat_id, message_ids=list(message_ids))
        chat = self.chat(chat_id)
        if chat is not None and chat.kind == "group":
            return {}
        return {
            mid: self.views[(chat_id, mid)] for mid in message_ids if (chat_id, mid) in self.views
        }

    async def find_message(
        self,
        chat_id: int,
        *,
        contains: str | None = None,
        fwd_of: tuple[int, int] | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> int | None:
        self._call(
            "find_message",
            chat_id=chat_id,
            contains=contains,
            fwd_of=fwd_of,
            since=since,
            limit=limit,
        )
        self._require_owned(chat_id)
        newest = sorted(self.world.sent(chat_id), key=lambda m: m.message_id, reverse=True)[:limit]
        for msg in newest:
            if since is not None and msg.date <= since:
                continue
            if contains is not None and contains not in msg.text and contains not in msg.hrefs:
                continue
            if fwd_of is not None and msg.fwd_of != fwd_of:
                continue
            return msg.message_id
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
        """The fake's media identity is ``copied_from``: the bot's copies of a staging message
        point at it, as a re-sent Telegram photo keeps its photo id."""
        self._call(
            "find_media",
            chat_id=chat_id,
            copies_of=(copies_of[0], list(copies_of[1])),
            after_id=after_id,
            since=since,
            limit=limit,
        )
        self._require_owned(chat_id)
        self._require_owned(copies_of[0])
        newest = sorted(self.world.sent(chat_id), key=lambda m: m.message_id, reverse=True)[:limit]
        bare = [
            m
            for m in reversed(newest)
            if (since is None or m.date > since)
            and (after_id is None or m.message_id > after_id)
            and m.media is not None
            and not m.text
            and m.fwd_of is None
        ]
        found: list[int] = []
        for staging_id in copies_of[1]:
            match = next(
                (
                    m.message_id
                    for m in bare
                    if (not found or m.message_id > found[-1])
                    and m.copied_from == (copies_of[0], staging_id)
                ),
                None,
            )
            if match is None:
                return []
            found.append(match)
        return found

    # --- writes into owned chats ---

    async def create_channel(self, title: str, about: str = "") -> ChatInfo:
        self._call("create_channel", title=title, about=about)
        self._next_chat -= 1
        chat = ChatInfo(
            id=self._next_chat,
            kind="channel",
            title=title,
            username=None,
            noforwards=False,
            is_creator=True,
            is_admin=True,
            archived=False,
            muted_until=None,
        )
        self.add_chat(chat)
        self.created.append(chat)
        return chat

    async def rename_channel(self, chat_id: int, title: str) -> None:
        self._call("rename_channel", chat_id=chat_id, title=title)
        self._require_owned(chat_id)
        self._replace_chat(chat_id, title=title)

    async def add_bot_admin(self, chat_id: int, bot_username: str) -> None:
        self._call("add_bot_admin", chat_id=chat_id, bot_username=bot_username)
        self._require_owned(chat_id)
        self.admins.add((chat_id, bot_username))
        self.world.bot_blocked.discard(chat_id)

    async def copy_media(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        self._call(
            "copy_media",
            from_chat_id=from_chat_id,
            message_ids=list(message_ids),
            to_chat_id=to_chat_id,
        )
        self._require_owned(to_chat_id)
        out = []
        for mid in message_ids:
            src = self._source(from_chat_id, mid)
            new = self.world.add(
                to_chat_id,
                sender="user",
                media=src.media if src else None,
                copied_from=(from_chat_id, mid),
            )
            out.append(new.message_id)
        return out

    async def forward(
        self, from_chat_id: int, message_ids: Sequence[int], to_chat_id: int
    ) -> list[int]:
        self._call(
            "forward",
            from_chat_id=from_chat_id,
            message_ids=list(message_ids),
            to_chat_id=to_chat_id,
        )
        self._require_owned(to_chat_id)
        if all((from_chat_id, mid) in self.deleted_sources for mid in message_ids):
            # as the real gateway: Telegram forwards nothing (MESSAGE_ID_INVALID)
            raise MediaUnavailable(f"the source message(s) {list(message_ids)} are gone")
        out = []
        for mid in message_ids:
            if (from_chat_id, mid) in self.deleted_sources:
                continue
            src = self._source(from_chat_id, mid)
            new = self.world.add(
                to_chat_id,
                sender="user",
                html=src.html if src else None,
                text=src.text if src else "",
                media=src.media if src else None,
                fwd_of=(from_chat_id, mid),
            )
            out.append(new.message_id)
        return out

    def register_owned(self, chat_id: int) -> None:
        self.calls.append(("register_owned", {"chat_id": chat_id}))
        self.owned.add(chat_id)

    # --- subscriptions ---

    async def mute(self, chat_id: int, until: datetime | None) -> None:
        self._call("mute", chat_id=chat_id, until=until)
        self._require_not_owned(chat_id)
        self.mutes[chat_id] = until
        self._replace_chat(chat_id, muted_until=until)

    async def set_archived(self, chat_id: int, archived: bool) -> None:
        self._call("set_archived", chat_id=chat_id, archived=archived)
        self._require_not_owned(chat_id)
        self.archived[chat_id] = archived
        self._replace_chat(chat_id, archived=archived)

    async def leave(self, chat_id: int) -> None:
        self._call("leave", chat_id=chat_id)
        self._require_not_owned(chat_id)
        chat = self.chat(chat_id)
        if chat is None:
            raise ChatGone(f"chat {chat_id} cannot be found")
        if chat.is_creator:
            raise NotAllowed("creator", "the account created this chat and cannot leave it")
        self.chats = [c for c in self.chats if c.id != chat_id]
        self.left.append(chat_id)

    async def list_folders(self) -> list[tuple[int, str]]:
        self._call("list_folders")
        return [(fid, title) for fid, (title, _) in sorted(self.folders.items())]

    async def get_folder(self, folder_id: int) -> list[int] | None:
        self._call("get_folder", folder_id=folder_id)
        entry = self.folders.get(folder_id)
        return list(entry[1]) if entry else None

    def register_own_folder(self, folder_id: int) -> None:
        self.calls.append(("register_own_folder", {"folder_id": folder_id}))
        self._own_folders.add(folder_id)

    async def save_folder(self, folder_id: int | None, title: str, chat_ids: Sequence[int]) -> int:
        self._call("save_folder", folder_id=folder_id, title=title, chat_ids=list(chat_ids))
        if not chat_ids:
            raise ValueError("a folder needs at least one chat (FILTER_INCLUDE_EMPTY)")
        if len(title) > 12:
            raise ValueError("a folder title is at most 12 characters")
        if folder_id is None:
            if len(self.folders) >= self.folder_limit:
                raise FolderLimit("DIALOG_FILTERS_TOO_MUCH")
            folder_id = next(i for i in range(2, 256) if i not in self.folders)
            self._own_folders.add(folder_id)
        elif folder_id not in self._own_folders:
            raise NotOwnedError(f"folder {folder_id} was not created by this gateway")
        self.folders[folder_id] = (title, list(chat_ids))
        return folder_id

    async def delete_folder(self, folder_id: int) -> None:
        self._call("delete_folder", folder_id=folder_id)
        if folder_id not in self._own_folders:
            raise NotOwnedError(f"folder {folder_id} was not created by this gateway")
        self.folders.pop(folder_id, None)
        self._own_folders.discard(folder_id)

    # --- guards ---

    def _require_owned(self, chat_id: int) -> None:
        if chat_id not in self.owned:
            raise NotOwnedError(f"chat {chat_id} is not registered as owned")

    def _require_not_owned(self, chat_id: int) -> None:
        if chat_id in self.owned:
            raise NotOwnedError(f"chat {chat_id} is an output/staging channel")

    def _source(self, chat_id: int, message_id: int) -> FakeMessage | None:
        chat = self.chat(chat_id)
        src = self.world.get(chat_id, message_id)
        if (chat is not None and chat.noforwards) or (src is not None and src.noforwards):
            raise ForwardsRestricted(f"chat {chat_id} forbids saving content")
        return src


# --- the bot ---------------------------------------------------------------------------------


class FakeBotGateway(_Recorder):
    """``BotGateway`` double. Messages land in the shared world (``sent(chat_id)`` to read
    them back); ``cannot_post`` holds the channels the bot is not an admin of."""

    def __init__(self, world: FakeWorld | None = None, *, account: Account = BOT_ACCOUNT) -> None:
        super().__init__()
        self.world = world or FakeWorld()
        self.account = account
        self.started = False
        self.answers: list[tuple[str, str | None, bool]] = []
        self.deleted: list[tuple[int, int]] = []
        self._message_handlers: list[Callable[[BotMessage], Awaitable[None]]] = []
        self._callback_handlers: list[Callable[[BotCallback], Awaitable[None]]] = []

    @property
    def cannot_post(self) -> set[int]:
        return self.world.bot_blocked

    # --- test helpers ---

    def sent(self, chat_id: int) -> list[FakeMessage]:
        return self.world.sent(chat_id)

    async def say(self, msg: BotMessage) -> None:
        """Deliver a message to every ``on_message`` handler (what the owner typed)."""
        for handler in list(self._message_handlers):
            await handler(msg)

    async def press(self, cb: BotCallback) -> None:
        """Deliver a button press to every ``on_callback`` handler."""
        for handler in list(self._callback_handlers):
            await handler(cb)

    # --- the protocol ---

    async def start(self) -> Account:
        self._call("start")
        self.started = True
        return self.account

    async def stop(self) -> None:
        self._call("stop")
        self.started = False

    async def ensure_connected(self) -> None:
        """The fake never drops its link; an outage is injected with ``fail_next``."""
        self._call("ensure_connected")

    def on_message(self, handler: Callable[[BotMessage], Awaitable[None]]) -> None:
        self._message_handlers.append(handler)

    def on_callback(self, handler: Callable[[BotCallback], Awaitable[None]]) -> None:
        self._callback_handlers.append(handler)

    async def send_text(
        self,
        chat_id: int,
        html: str,
        *,
        buttons: Buttons | None = None,
        reply_to: int | None = None,
        silent: bool = False,
    ) -> int:
        self._call(
            "send_text",
            chat_id=chat_id,
            html=html,
            buttons=buttons,
            reply_to=reply_to,
            silent=silent,
        )
        self._require_can_post(chat_id)
        msg = self.world.add(
            chat_id, sender="bot", html=html, buttons=buttons, reply_to=reply_to, silent=silent
        )
        return msg.message_id

    async def send_copy(
        self,
        from_chat_id: int,
        message_ids: Sequence[int],
        to_chat_id: int,
        *,
        caption_html: str | None = None,
        buttons: Buttons | None = None,
    ) -> list[int]:
        self._call(
            "send_copy",
            from_chat_id=from_chat_id,
            message_ids=list(message_ids),
            to_chat_id=to_chat_id,
            caption_html=caption_html,
            buttons=buttons,
        )
        self._require_can_post(to_chat_id)
        if len(message_ids) > 1 and buttons:
            raise ValueError(
                "albums cannot carry buttons (Telegram forbids reply markup on grouped media)"
            )
        single = len(message_ids) == 1
        out = []
        for mid in message_ids:
            src = self.world.get(from_chat_id, mid)
            if src is None:
                raise MediaUnavailable(f"message {mid} not found in chat {from_chat_id}")
            msg = self.world.add(
                to_chat_id,
                sender="bot",
                html=caption_html if single else None,
                media=src.media,
                buttons=buttons if single else None,
                copied_from=(from_chat_id, mid),
            )
            out.append(msg.message_id)
        return out

    async def edit_text(
        self, chat_id: int, message_id: int, html: str, *, buttons: Buttons | None = None
    ) -> None:
        self._call("edit_text", chat_id=chat_id, message_id=message_id, html=html, buttons=buttons)
        msg = self._own_message(chat_id, message_id)
        if msg.html == html and msg.buttons == buttons:
            return  # MESSAGE_NOT_MODIFIED is swallowed
        msg.html, msg.text, msg.buttons = html, plain_text(html), buttons
        msg.edits += 1

    async def edit_buttons(self, chat_id: int, message_id: int, buttons: Buttons | None) -> None:
        self._call("edit_buttons", chat_id=chat_id, message_id=message_id, buttons=buttons)
        msg = self._own_message(chat_id, message_id)
        if msg.buttons == buttons:
            return
        msg.buttons = buttons
        msg.edits += 1

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self._call("delete_message", chat_id=chat_id, message_id=message_id)
        if chat_id <= 0:
            raise NotAllowed("other", "the bot deletes messages only in the private owner chat")
        self.world.messages[chat_id] = [
            m for m in self.world.messages.get(chat_id, ()) if m.message_id != message_id
        ]
        self.deleted.append((chat_id, message_id))

    async def answer_callback(
        self, query_id: str, text: str | None = None, *, alert: bool = False
    ) -> None:
        self._call("answer_callback", query_id=query_id, text=text, alert=alert)
        self.answers.append((query_id, text, alert))

    async def can_post(self, chat_id: int) -> bool:
        self._call("can_post", chat_id=chat_id)
        return chat_id not in self.world.bot_blocked

    # --- guards ---

    def _require_can_post(self, chat_id: int) -> None:
        if chat_id in self.world.bot_blocked:
            raise BotCannotPost(f"the bot cannot post into {chat_id}")

    def _own_message(self, chat_id: int, message_id: int) -> FakeMessage:
        msg = self.world.get(chat_id, message_id)
        if msg is None:
            raise ValueError(f"message {message_id} does not exist in chat {chat_id}")
        if msg.sender != "bot":
            raise NotAllowed("other", "a bot can only edit its own messages")
        return msg


# --- models and the language model -----------------------------------------------------------


class FakeEmbedder:
    """Deterministic hashed bag-of-words unit vectors: identical texts give identical vectors,
    texts that share most words give a high cosine, unrelated texts a low one."""

    id = "fake-embedder-64"
    dim = 64

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._vector(text)
        return out

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in normalise(text).split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            n = int.from_bytes(digest, "little")
            v[n % self.dim] += 1.0 if (n >> 8) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v[0] = 1.0
            return v
        return v / norm


class FakeClassifier:
    """Returns the confidences a test sets: ``scores`` (topic id -> confidence) for every
    text, ``text_scores`` (exact text -> scores) for particular texts; every active topic from
    the last ``reload`` gets ``default`` when not listed."""

    def __init__(self, scores: dict[int, float] | None = None, *, default: float = 0.1) -> None:
        self.scores: dict[int, float] = dict(scores or {})
        self.text_scores: dict[str, dict[int, float]] = {}
        self.default = default
        self.topics: list[Topic] = []
        self.examples: list[Example] = []
        self.learned: list[Example] = []
        self.reloads = 0
        self.categories: list[tuple[str, float]] = [(k, default) for k in PROVISIONAL_CATEGORIES]

    def reload(self, topics: Sequence[Topic], examples: Sequence[Example]) -> None:
        self.topics = [t for t in topics if t.active]
        self.examples = list(examples)
        self.reloads += 1

    def predict(self, text: str, embedding: np.ndarray) -> list[TopicScore]:
        scores = self.text_scores.get(text, self.scores)
        topic_ids = [t.id for t in self.topics] or list(scores)
        result = [TopicScore(tid, float(scores.get(tid, self.default))) for tid in topic_ids]
        return sorted(result, key=lambda s: (-s.confidence, s.topic_id))

    def learn(self, example: Example) -> None:
        self.learned.append(example)
        self.examples.append(example)

    def category_scores(self, text: str, embedding: np.ndarray) -> list[tuple[str, float]]:
        return sorted(self.categories, key=lambda kv: -kv[1])


CANNED_TOPIC_NAME = TopicName("Canned topic", "A description written by the fake.")


class FakeLLM:
    """Canned answers; ``enabled=False`` (or ``fail=True``) makes every call return ``None``."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        summary: str | None = "Canned one-line summary.",
        topic_name: TopicName | None = CANNED_TOPIC_NAME,
        opinion: bool | None = True,
    ) -> None:
        self.enabled = enabled
        self.summary = summary
        self.summaries: dict[str, str] = {}
        self.topic_name = topic_name
        self.opinion = opinion
        self.fail = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def summarise_line(self, text: str, max_chars: int) -> str | None:
        self.calls.append(("summarise_line", {"text": text, "max_chars": max_chars}))
        if not self.enabled or self.fail:
            return None
        line = self.summaries.get(text, self.summary)
        return None if line is None else line[:max_chars]

    async def name_topic(
        self,
        examples: Sequence[str],
        *,
        existing: Sequence[str] = (),
        category_label: str | None = None,
    ) -> TopicName | None:
        self.calls.append(
            (
                "name_topic",
                {
                    "examples": list(examples),
                    "existing": list(existing),
                    "category_label": category_label,
                },
            )
        )
        if not self.enabled or self.fail:
            return None
        return self.topic_name

    async def second_opinion(
        self, text: str, topic: Topic, *, competing: Sequence[Topic] = ()
    ) -> bool | None:
        self.calls.append(
            ("second_opinion", {"text": text, "topic": topic, "competing": list(competing)})
        )
        if not self.enabled or self.fail:
            return None
        return self.opinion


# --- bot-side factories ----------------------------------------------------------------------


def make_bot_message(
    text: str,
    *,
    sender_id: int = OWNER_ID,
    chat_id: int | None = None,
    message_id: int = 1,
    is_private: bool = True,
    fwd_from_chat_id: int | None = None,
    fwd_from_title: str | None = None,
    reply_to_id: int | None = None,
    has_media: bool = False,
) -> BotMessage:
    """A message to the bot; by default the owner's, in the private chat."""
    return BotMessage(
        chat_id=sender_id if chat_id is None else chat_id,
        message_id=message_id,
        sender_id=sender_id,
        text=text,
        is_private=is_private,
        fwd_from_chat_id=fwd_from_chat_id,
        fwd_from_title=fwd_from_title,
        reply_to_id=reply_to_id,
        has_media=has_media,
    )


def make_callback(
    data: str,
    *,
    sender_id: int = OWNER_ID,
    chat_id: int | None = None,
    message_id: int = 1,
    query_id: str = "q1",
) -> BotCallback:
    """A button press; by default the owner's, on a message in the private chat."""
    return BotCallback(
        query_id=query_id,
        sender_id=sender_id,
        chat_id=sender_id if chat_id is None else chat_id,
        message_id=message_id,
        data=data,
    )
