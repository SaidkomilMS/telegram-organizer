"""Telegram messages -> candidates (DESIGN §9.1): filtering, counting, albums, group units.

Intake decides *what is a post*. Channels are publications, so every message is a candidate
and albums collapse into the message that carries the caption; groups are conversations, so
messages are buffered in ``group_messages``, stitched into conversation units and only the
units that reach the floor become candidates. Counting rules live here too because they
depend on "was the message new": groups are counted when ``add_group_message`` inserts a row,
channels when the sorter's ``insert_post`` does (the sorter bumps those, §9.3).

Nothing here calls the sorter except ``handle_message`` and ``tick``; ``collect`` and
``close_units`` only build candidates so ``Backfill`` can order them globally before
submitting.

Live intake and a backfill run in the same process at the same time, so the two keep out of
each other's way: a group chat a backfill ``collect`` has buffered into belongs to that
backfill until its ``close_units(chat, ..., force=True)`` (the live ``tick`` leaves it alone and
does not purge meanwhile), backfill albums live only inside their own ``collect``, and every
read-then-write of one chat's open units holds that chat's lock.

Live album parts wait ``ALBUM_SETTLE`` in memory, so they are also written to
``kv intake.open_albums`` until the album is submitted: a stop or a crash in those seconds does
not lose the album, the next process submits it (``flush_albums`` submits them at a clean stop).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from tg_curator.clock import local_date
from tg_curator.db.store import GroupMessage
from tg_curator.domain import KV, Candidate, Chat, ChatSyncResult, Via
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage
from tg_curator.textutil import canonical_urls, html_escape

log = logging.getLogger(__name__)

ALBUM_SETTLE = timedelta(seconds=2)
"""How long after its last part an album waits before it becomes one candidate (§9.1)."""
LATE_DELIVERY = timedelta(minutes=10)
"""A live message dated this far before the process started is a reconnect gap-fill (§9.1)."""
UNIT_MAX_AGE = timedelta(minutes=30)
"""A conversation unit never spans more than this from its root (§9.1)."""
GROUP_RETENTION = timedelta(days=3)
"""Buffered group messages older than this are purged (§6)."""
PURGE_INTERVAL = timedelta(hours=1)
"""``tick`` runs every few seconds; the purge is a DELETE and needs no such pace."""
LATE_UNIT = UNIT_MAX_AGE + LATE_DELIVERY
"""A unit the live ``tick`` closes this long after its root was not closed on time: it was
buffered by a backfill or before downtime, so like a late delivery it never goes out in real
time and is submitted as ``via="backfill"``."""
SUBMIT_ATTEMPTS = 3
"""How often ``tick`` submits a candidate whose submission raised before it gives up."""
THREAD_WINDOW = timedelta(hours=24)
"""How long after its root a reply still joins its thread (spec: "the replies that hang off a
message" are one piece of text), even past the same-sender gap and once the thread's unit was
closed below the floor (§14). Shorter than the digest window, so a slow thread that only
reaches the floor late still makes the digest."""
FORWARD_FLOOR_SHARE = 4
"""A post forwarded from a channel passes at ``min_chars // 4`` characters (spec: "the
articles people forward into the group" pass)."""
LINK_FLOOR_SHARE = 20
"""A shared outside link passes when the words around it reach ``min_chars // 20`` characters:
"Good long read on this: <link>" passes, "lol <link>" does not."""

_AlbumKey = tuple[Via, int, int]


@dataclass
class _Album:
    """Parts of one channel album, waiting for the stream to settle."""

    via: Via
    chat: Chat
    parts: list[IncomingMessage]
    last_seen: datetime


@dataclass
class _Unit:
    """An open conversation unit: the root and every member buffered so far."""

    root_id: int
    root_date: datetime
    members: list[GroupMessage]

    @property
    def last(self) -> GroupMessage:
        return self.members[-1]

    def accepts(self, at: datetime, gap: timedelta) -> bool:
        """Would the unit still be open for a message dated ``at``?"""
        return at - self.last.date < gap and at - self.root_date < UNIT_MAX_AGE

    def accepts_reply(self, at: datetime) -> bool:
        """A reply joins its thread for ``THREAD_WINDOW`` after the root, whatever the gap."""
        return timedelta(0) <= at - self.root_date < THREAD_WINDOW

    def closes(self, now: datetime, gap: timedelta) -> bool:
        return now - self.last.date >= gap or now - self.root_date >= UNIT_MAX_AGE


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def passes_floor(candidate: Candidate, min_chars: int) -> bool:
    """Does a unit reach the group floor (§9.1, §14)? Long text always does; a post forwarded
    from a channel needs a quarter of the floor, a shared outside link a few words of its own.
    """
    text = candidate.text
    if len(text) >= min_chars:
        return True
    if candidate.fwd_from_chat_id is not None and len(text) >= min_chars // FORWARD_FLOOR_SHARE:
        return True
    if canonical_urls(candidate.urls):
        own = text
        for url in candidate.urls:
            own = own.replace(url, " ")
        return len(" ".join(own.split())) >= max(1, min_chars // LINK_FLOOR_SHARE)
    return False


def _album_part(msg: IncomingMessage) -> dict[str, Any]:
    """A live album part as JSON for ``kv intake.open_albums`` (the chat is stored once)."""
    return {
        "message_id": msg.message_id,
        "date": msg.date.isoformat(),
        "text": msg.text,
        "html": msg.html,
        "sender_id": msg.sender_id,
        "reply_to_id": msg.reply_to_id,
        "grouped_id": msg.grouped_id,
        "media": msg.media,
        "urls": list(msg.urls),
        "views": msg.views,
        "forwards": msg.forwards,
        "fwd_from_chat_id": msg.fwd_from_chat_id,
        "fwd_from_message_id": msg.fwd_from_message_id,
        "noforwards": msg.noforwards,
        "topic_id": msg.topic_id,
    }


def _album_message(chat: Chat, data: dict[str, Any]) -> IncomingMessage:
    info = ChatInfo(
        id=chat.id,
        kind=chat.kind,
        title=chat.title,
        username=chat.username,
        noforwards=chat.noforwards,
        is_creator=chat.is_creator,
        is_admin=False,
        archived=False,
        muted_until=None,
    )
    return IncomingMessage(
        chat=info,
        message_id=int(data["message_id"]),
        date=datetime.fromisoformat(data["date"]),
        text=data.get("text") or "",
        html=data.get("html"),
        sender_id=data.get("sender_id"),
        is_outgoing=False,
        is_service=False,
        reply_to_id=data.get("reply_to_id"),
        grouped_id=data.get("grouped_id"),
        media=data.get("media"),
        urls=tuple(data.get("urls") or ()),
        views=data.get("views"),
        forwards=data.get("forwards"),
        fwd_from_chat_id=data.get("fwd_from_chat_id"),
        fwd_from_message_id=data.get("fwd_from_message_id"),
        noforwards=bool(data.get("noforwards")),
        topic_id=data.get("topic_id"),
    )


def _build_units(rows: list[GroupMessage]) -> dict[int, _Unit]:
    """Open units of one chat from its buffered rows (oldest first, as the Store orders)."""
    units: dict[int, _Unit] = {}
    for row in rows:
        root_id = row.unit_root_id if row.unit_root_id is not None else row.message_id
        unit = units.get(root_id)
        if unit is None:
            units[root_id] = _Unit(root_id=root_id, root_date=row.date, members=[row])
        else:
            unit.members.append(row)
            if row.message_id == root_id:
                unit.root_date = row.date
    return units


class IntakeService:
    """The ``Intake`` service of §8; constructed once by ``service.py`` / the CLI.

    ``started_at`` anchors the late-delivery guard: by default the moment of construction,
    which is the moment the process started for the service.
    """

    def __init__(self, rt: Runtime, *, started_at: datetime | None = None) -> None:
        self._rt = rt
        self._started_at = started_at or rt.clock.now()
        self._albums: dict[_AlbumKey, _Album] = {}
        """Live albums only; a backfill's albums stay local to its ``collect``."""
        self._purged_at: datetime | None = None
        self._backfilling: set[int] = set()
        """Group chats a backfill buffered into and has not force-closed yet."""
        self._forced: dict[tuple[int, int], datetime] = {}
        """``(chat_id, root_id) -> root date`` of the recent units a forced close returned: if
        one is reopened after a failed submit, a later live close still labels it a backfill
        (an older one is labelled so by its age, ``LATE_UNIT``)."""
        self._group_locks: dict[int, asyncio.Lock] = {}
        self._group_gen: dict[int, int] = {}
        """Bumped on every change of a chat's open units, so a backfill's cache notices."""
        self._retry: list[Candidate] = []
        self._attempts: dict[tuple[int, int], int] = {}
        self._albums_loaded = False
        """``kv intake.open_albums`` of a previous process read back (once, lazily)."""
        self._submitting: dict[_AlbumKey, _Album] = {}
        """Live albums flushed into a candidate whose submit has not succeeded yet: they stay
        in ``kv intake.open_albums`` until it does (or is given up)."""
        self._albums_dirty = False

    # --- the protocol ------------------------------------------------------------------------

    async def handle_message(self, msg: IncomingMessage) -> None:
        """One live message: collect it and submit what it yields at once."""

        async def one() -> AsyncIterator[IncomingMessage]:
            yield msg

        await self._submit(await self.collect(one(), via="live"))

    async def collect(
        self, messages: AsyncIterator[IncomingMessage], *, via: Via
    ) -> list[Candidate]:
        """Run the stream through the filters, the album buffer and the group buffer.

        Albums of a backfill stream are complete when the stream ends, so they are flushed
        here from a buffer of this call alone (the live ``tick`` never sees them, and a read
        that fails mid-album leaves nothing behind); live albums arrive one part per call and
        wait for ``tick``. Group units are never closed here (``close_units`` does), so a
        group chat yields nothing from ``collect`` itself; a backfill claims every group chat
        it buffers into until its forced ``close_units`` releases it.
        """
        out: list[Candidate] = []
        chats: dict[int, Chat | None] = {}
        units: dict[int, tuple[int, dict[int, _Unit]]] = {}
        if via == "live":
            await self._load_albums()
        albums = self._albums if via == "live" else {}
        album_keys: list[_AlbumKey] = []
        async for msg in messages:
            chat = await self._source_chat(msg, via, chats)
            if chat is None:
                continue
            if chat.kind == "group":
                await self._buffer_group_message(msg, chat, units, via)
            elif msg.grouped_id is not None:
                album_keys.append(self._buffer_album_part(msg, chat, via, albums))
            else:
                out.append(self._post_candidate(msg, chat, via))
                log.debug("intake: post chat=%d message=%d via=%s", chat.id, msg.message_id, via)
            if via == "live":
                await self._rt.store.kv_set(KV.INTAKE_LAST_MESSAGE_AT, msg.date.isoformat())
        if via == "backfill":
            for key in dict.fromkeys(album_keys):
                out.append(self._flush_album(key, albums))
        elif album_keys:
            self._albums_dirty = True
            await self._save_albums()
        return out

    async def close_units(
        self, chat_id: int | None, now: datetime, *, force: bool = False
    ) -> list[Candidate]:
        """Close the units that are due at ``now`` (all of them with ``force``).

        Every closed unit is marked in ``group_messages``; only those that reach
        ``groups.min_chars`` come back as candidates — the rest exist as raw volume only.
        A forced close is the backfill's, so its candidates carry ``via="backfill"``, and it
        releases the chat(s) for the live ``tick`` again even when it fails. A non-forced
        close skips the chats a backfill is still filling, and labels a unit it closes late
        (``LATE_UNIT``) or that a forced close returned before as ``via="backfill"``.
        """
        try:
            return await self._close_units(chat_id, now, force)
        finally:
            if force:
                if chat_id is None:
                    self._backfilling.clear()
                else:
                    self._backfilling.discard(chat_id)

    async def _close_units(
        self, chat_id: int | None, now: datetime, force: bool
    ) -> list[Candidate]:
        store = self._rt.store
        settings = self._rt.settings
        gap = timedelta(minutes=settings.groups.unit_gap_minutes)
        by_chat: dict[int, list[GroupMessage]] = {}
        for row in await store.open_group_messages(chat_id):
            by_chat.setdefault(row.chat_id, []).append(row)
        out: list[Candidate] = []
        for cid, rows in by_chat.items():
            if not force and (
                cid in self._backfilling
                or not any(u.closes(now, gap) for u in _build_units(rows).values())
            ):
                continue
            # the snapshot above only says something may close; decide again under the lock
            # so a message attached to a unit meanwhile is closed with it, never left behind
            async with self._group_lock(cid):
                if not force and cid in self._backfilling:
                    continue
                units = _build_units(await store.open_group_messages(cid))
                closing = [u for u in units.values() if force or u.closes(now, gap)]
                if not closing:
                    continue
                chat = await store.get_chat(cid)
                async with store.begin() as conn:
                    for unit in closing:
                        await store.mark_group_messages(
                            cid,
                            [m.message_id for m in unit.members],
                            unit_root_id=unit.root_id,
                            closed=True,
                            conn=conn,
                        )
                self._changed(cid)
            for unit in closing:
                via = self._close_via(cid, unit, now, force)
                candidate = self._unit_candidate(unit, chat, via)
                if passes_floor(candidate, settings.groups.min_chars):
                    out.append(candidate)
                    if force and self._rt.clock.now() - unit.root_date < LATE_UNIT:
                        self._forced[(cid, unit.root_id)] = unit.root_date
                    log.info(
                        "intake: unit chat=%d root=%d members=%d chars=%d via=%s",
                        cid,
                        unit.root_id,
                        len(unit.members),
                        len(candidate.text),
                        via,
                    )
                else:
                    log.debug(
                        "intake: unit chat=%d root=%d below the floor (%d chars)",
                        cid,
                        unit.root_id,
                        len(candidate.text),
                    )
        return out

    def _close_via(self, chat_id: int, unit: _Unit, now: datetime, force: bool) -> Via:
        if force:
            return "backfill"
        if (chat_id, unit.root_id) in self._forced or now - unit.root_date >= LATE_UNIT:
            return "backfill"
        return "live"

    async def tick(self) -> None:
        """Retry failed posts, flush settled albums, close due units, submit; purge the group
        buffer (not while a backfill is filling it: its rows may be older than 3 days)."""
        now = self._rt.clock.now()
        await self._load_albums()
        candidates, self._retry = self._retry, []
        settled = self._flush_settled_albums(now)
        candidates += settled
        candidates += await self.close_units(None, now)
        await self._submit(candidates)
        await self._save_albums()  # only after the submit: a crash before it keeps them
        if self._backfilling:
            return
        if self._purged_at is None or now - self._purged_at >= PURGE_INTERVAL:
            purged = await self._rt.store.purge_group_messages(now - GROUP_RETENTION)
            self._purged_at = now
            self._forced = {k: d for k, d in self._forced.items() if now - d < LATE_UNIT}
            if purged:
                log.info("intake: purged %d buffered group messages older than 3 days", purged)

    async def sync_chats(self) -> ChatSyncResult:
        """Mirror the dialog list into ``chats``: new dialogs become sources, vanished source
        chats are marked left (their rows, history and statistics stay)."""
        rt = self._rt
        if rt.user is None:
            raise RuntimeError("intake.sync_chats needs the user gateway")
        dialogs = await rt.user.list_chats()
        known = {c.id: c for c in await rt.store.list_chats()}
        seen: set[int] = set()
        new: list[int] = []
        outputs = 0
        for info in dialogs:
            seen.add(info.id)
            old = known.get(info.id)
            chat = await rt.store.upsert_chat(info)
            if old is None or not old.active:
                new.append(info.id)
            if chat.role == "output":
                outputs += 1
        now = rt.clock.now()
        left: list[int] = []
        for cid, chat in known.items():
            if cid in seen or not chat.active or chat.role != "source":
                continue
            await rt.store.set_chat_fields(cid, active=False, left_at=now)
            left.append(cid)
        log.info(
            "intake: found %d chats, %d are output channels (%d new, %d left)",
            len(dialogs),
            outputs,
            len(new),
            len(left),
        )
        return ChatSyncResult(total=len(dialogs), new=new, left=left, outputs=outputs)

    # --- filtering ---------------------------------------------------------------------------

    async def _source_chat(
        self, msg: IncomingMessage, via: Via, cache: dict[int, Chat | None]
    ) -> Chat | None:
        """The source chat row for a message, or ``None`` when the message is ignored."""
        # The user's own messages are ignored in groups; in a channel every post is a
        # candidate, the account's own posts included (SPEC "Channels and groups").
        if msg.is_service or (msg.is_outgoing and msg.chat.kind == "group"):
            return None
        if msg.is_automatic_forward:
            # A channel's post copied into its discussion group: not a message of the group
            # (no volume, no unit root, no "other chat" corroborating the channel post).
            log.debug(
                "intake: automatic forward chat=%d message=%d ignored", msg.chat.id, msg.message_id
            )
            return None
        if via == "live" and msg.date < self._started_at - LATE_DELIVERY:
            log.debug(
                "intake: dropping late delivery chat=%d message=%d dated %s",
                msg.chat.id,
                msg.message_id,
                msg.date.isoformat(),
            )
            return None
        store = self._rt.store
        if msg.chat.id not in cache:
            chat = await store.get_chat(msg.chat.id)
            if chat is None:
                chat = await store.upsert_chat(msg.chat)
                log.info("intake: new source chat %d (%s)", chat.id, chat.title)
            elif not chat.active:
                chat = await store.upsert_chat(msg.chat)
                log.info("intake: source chat %d (%s) is active again", chat.id, chat.title)
            cache[msg.chat.id] = chat if chat.role == "source" else None
        return cache[msg.chat.id]

    # --- channels ----------------------------------------------------------------------------

    @staticmethod
    def _post_candidate(msg: IncomingMessage, chat: Chat, via: Via) -> Candidate:
        return Candidate(
            chat_id=chat.id,
            message_id=msg.message_id,
            message_ids=[msg.message_id],
            kind="post",
            posted_at=msg.date,
            text=msg.text,
            html=msg.html,
            urls=_unique(msg.urls),
            media=msg.media,
            grouped_id=None,
            views=msg.views,
            forwards=msg.forwards,
            fwd_from_chat_id=msg.fwd_from_chat_id,
            fwd_from_message_id=msg.fwd_from_message_id,
            noforwards=msg.noforwards or chat.noforwards,
            via=via,
        )

    def _buffer_album_part(
        self, msg: IncomingMessage, chat: Chat, via: Via, albums: dict[_AlbumKey, _Album]
    ) -> _AlbumKey:
        key = (via, chat.id, msg.grouped_id or 0)
        now = self._rt.clock.now()
        album = albums.get(key)
        if album is None:
            albums[key] = _Album(via=via, chat=chat, parts=[msg], last_seen=now)
        else:
            album.parts.append(msg)
            album.last_seen = now
        return key

    async def flush_albums(self) -> int:
        """Submit every buffered live album now, settled or not (a clean stop calls this:
        Telegram sends an album's parts within milliseconds, so none is cut short)."""
        await self._load_albums()
        candidates = [self._flush_live_album(k) for k in list(self._albums)]
        await self._submit(candidates)
        await self._save_albums()
        return len(candidates)

    async def _save_albums(self) -> None:
        """Mirror the live albums not yet stored (buffered, or flushed and not yet submitted)
        into ``kv intake.open_albums``; the key is deleted when there are none."""
        if not self._albums_dirty:
            return
        store = self._rt.store
        pending = {**self._submitting, **self._albums}
        try:
            if not pending:
                await store.kv_delete(KV.INTAKE_OPEN_ALBUMS)
            else:
                value = [
                    {
                        "chat_id": album.chat.id,
                        "grouped_id": key[2],
                        "parts": [_album_part(m) for m in album.parts],
                    }
                    for key, album in pending.items()
                ]
                await store.kv_set(KV.INTAKE_OPEN_ALBUMS, value)
            self._albums_dirty = False
        except Exception:
            log.exception("intake: could not record the open albums")

    async def _load_albums(self) -> None:
        """Albums a stopped or crashed process left settling come back into the buffer;
        the next ``tick`` submits them (the sorter dedups one that was submitted after all)."""
        if self._albums_loaded:
            return
        self._albums_loaded = True
        store = self._rt.store
        try:
            saved = await store.kv_get(KV.INTAKE_OPEN_ALBUMS)
            now = self._rt.clock.now()
            for entry in saved or []:
                chat = await store.get_chat(int(entry["chat_id"]))
                if chat is None or chat.role != "source":
                    continue
                for data in entry.get("parts") or []:
                    msg = _album_message(chat, data)
                    key = ("live", chat.id, msg.grouped_id or 0)
                    album = self._albums.get(key)
                    if album is None:
                        # buffered before the restart: settled, the next tick submits it
                        self._albums[key] = _Album("live", chat, [msg], now - ALBUM_SETTLE)
                    elif all(p.message_id != msg.message_id for p in album.parts):
                        album.parts.append(msg)
            if saved:
                self._albums_dirty = True
                log.info("intake: %d albums from before the restart are submitted", len(saved))
        except Exception:
            log.exception("intake: could not read back the open albums")

    def _flush_settled_albums(self, now: datetime) -> list[Candidate]:
        """Live albums whose last part is ``ALBUM_SETTLE`` old (backfill ones are never here)."""
        due = [k for k, a in self._albums.items() if now - a.last_seen >= ALBUM_SETTLE]
        return [self._flush_live_album(k) for k in due]

    def _flush_live_album(self, key: _AlbumKey) -> Candidate:
        """``_flush_album`` for a live album, remembered until its submit succeeds."""
        self._submitting[key] = self._albums[key]
        self._albums_dirty = True
        return self._flush_album(key, self._albums)

    def _album_done(self, c: Candidate) -> None:
        """A live album candidate was stored (or given up): it leaves ``kv``."""
        if c.kind == "post" and c.media == "album" and c.via == "live":
            if self._submitting.pop(("live", c.chat_id, c.grouped_id or 0), None) is not None:
                self._albums_dirty = True

    @staticmethod
    def _flush_album(key: _AlbumKey, albums: dict[_AlbumKey, _Album]) -> Candidate:
        """One candidate for the whole album: the caption carrier wins, else the first part."""
        album = albums.pop(key)
        # a part delivered twice (a re-delivery) is still one part
        unique = {m.message_id: m for m in album.parts}
        parts = sorted(unique.values(), key=lambda m: m.message_id)
        carrier = next((m for m in parts if m.text.strip()), parts[0])
        log.debug(
            "intake: album chat=%d carrier=%d parts=%d via=%s",
            album.chat.id,
            carrier.message_id,
            len(parts),
            album.via,
        )
        return Candidate(
            chat_id=album.chat.id,
            message_id=carrier.message_id,
            message_ids=[m.message_id for m in parts],
            kind="post",
            posted_at=carrier.date,
            text=carrier.text,
            html=carrier.html,
            urls=_unique(u for m in parts for u in m.urls),
            media="album",
            grouped_id=carrier.grouped_id,
            views=carrier.views,
            forwards=carrier.forwards,
            fwd_from_chat_id=carrier.fwd_from_chat_id,
            fwd_from_message_id=carrier.fwd_from_message_id,
            noforwards=carrier.noforwards or album.chat.noforwards,
            via=album.via,
        )

    # --- groups ------------------------------------------------------------------------------

    async def _buffer_group_message(
        self,
        msg: IncomingMessage,
        chat: Chat,
        units: dict[int, tuple[int, dict[int, _Unit]]],
        via: Via,
    ) -> None:
        """Store the message, count it when new, and attach it to its unit (§9.1 rules).

        ``units`` caches the chat's open units for the rest of a ``collect`` stream, tagged
        with the chat's change generation: when anyone else changed them meanwhile (a close,
        a live message during a backfill) they are read again. Choosing the unit and the
        insert happen under the chat's lock, so a concurrent close never misses the row.
        """
        store = self._rt.store
        if via == "backfill":
            self._backfilling.add(chat.id)
        async with self._group_lock(chat.id):
            cached = units.get(chat.id)
            if cached is None or cached[0] != self._group_gen.get(chat.id, 0):
                cached = (
                    self._group_gen.get(chat.id, 0),
                    _build_units(await store.open_group_messages(chat.id)),
                )
            chat_units = cached[1]
            unit = self._unit_for(msg, chat_units)
            reopened = False
            if unit is None and msg.reply_to_id is not None:
                unit = await self._closed_thread(msg, chat.id, chat_units)
                reopened = unit is not None
            root_id = unit.root_id if unit is not None else msg.message_id
            row = GroupMessage(
                chat_id=chat.id,
                message_id=msg.message_id,
                sender_id=msg.sender_id,
                reply_to_id=msg.reply_to_id,
                date=msg.date,
                text=msg.text,
                html=msg.html,
                urls=_unique(msg.urls),
                media=msg.media,
                fwd_from_chat_id=msg.fwd_from_chat_id,
                fwd_from_message_id=msg.fwd_from_message_id,
                unit_root_id=root_id,
                topic_id=msg.topic_id,
                grouped_id=msg.grouped_id,
            )
            async with store.begin() as conn:
                new = await store.add_group_message(
                    row.chat_id,
                    row.message_id,
                    date=row.date,
                    text=row.text,
                    sender_id=row.sender_id,
                    reply_to_id=row.reply_to_id,
                    html=row.html,
                    urls=row.urls,
                    media=row.media,
                    fwd_from_chat_id=row.fwd_from_chat_id,
                    fwd_from_message_id=row.fwd_from_message_id,
                    unit_root_id=root_id,
                    topic_id=row.topic_id,
                    grouped_id=row.grouped_id,
                    conn=conn,
                )
                if new:
                    # an album counts once (§9.1): only its first buffered part is counted
                    if self._counts(msg, via) and not (
                        msg.grouped_id is not None
                        and await store.album_buffered(
                            chat.id, msg.grouped_id, besides=msg.message_id, conn=conn
                        )
                    ):
                        day = local_date(msg.date, self._rt.settings.general.timezone)
                        await store.bump_chat_daily(chat.id, day, conn=conn)
                    await store.touch_chat(chat.id, msg.date, conn=conn)
                    if reopened and unit is not None:
                        await store.mark_group_messages(
                            chat.id, [m.message_id for m in unit.members], closed=False, conn=conn
                        )
            if new:
                if reopened and unit is not None:
                    chat_units[unit.root_id] = unit
                if unit is None:
                    chat_units[root_id] = _Unit(root_id=root_id, root_date=msg.date, members=[row])
                else:
                    unit.members.append(row)
                self._changed(chat.id)
            units[chat.id] = (self._group_gen.get(chat.id, 0), chat_units)

    async def _closed_thread(
        self, msg: IncomingMessage, chat_id: int, units: dict[int, _Unit]
    ) -> _Unit | None:
        """The closed unit a reply hangs off, when it never reached the floor (so it was never
        submitted) and its root is within ``THREAD_WINDOW``: the reply reopens it, so a slow
        thread is still one piece of text linking to its first message (§14). A unit that
        passed the floor was submitted as it stood; a reply to it starts a new unit."""
        assert msg.reply_to_id is not None
        if any(any(m.message_id == msg.reply_to_id for m in u.members) for u in units.values()):
            return None  # the target's unit is open; ``_unit_for`` already said no
        rows = await self._rt.store.group_unit_of(chat_id, msg.reply_to_id)
        if not rows or not all(r.closed for r in rows):
            return None
        target = next(r for r in rows if r.message_id == msg.reply_to_id)
        if target.topic_id != msg.topic_id:
            return None  # replies stay inside their forum topic
        root_id = target.unit_root_id if target.unit_root_id is not None else target.message_id
        unit = _build_units(rows).get(root_id)
        if unit is None or not unit.accepts_reply(msg.date):
            return None
        if passes_floor(
            self._unit_candidate(unit, None, "live"), self._rt.settings.groups.min_chars
        ):
            return None
        log.debug("intake: reply %d reopens the thread of root %d", msg.message_id, root_id)
        return unit

    def _counts(self, msg: IncomingMessage, via: Via) -> bool:
        """Does a newly inserted row prove the message was never counted (§9.1)?

        Not for a backfilled message older than the buffer's retention: its row may have been
        counted and purged since, so the insert proves nothing. ``Backfill`` counts the days
        it fully covered itself (``chat_daily = max(existing, read)``), which cannot double
        count; a partly covered oldest day may stay undercounted, never overcounted.
        """
        return via == "live" or msg.date >= self._rt.clock.now() - GROUP_RETENTION

    def _group_lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._group_locks.get(chat_id)
        if lock is None:
            lock = self._group_locks[chat_id] = asyncio.Lock()
        return lock

    def _changed(self, chat_id: int) -> None:
        self._group_gen[chat_id] = self._group_gen.get(chat_id, 0) + 1

    def _unit_for(self, msg: IncomingMessage, units: dict[int, _Unit]) -> _Unit | None:
        """(a) the open unit of the reply target, else (b) the open unit whose last message
        is by the same sender in the same forum topic within the gap, else ``None`` (a new
        root). Units never cross forum topics (§14); replies stay inside their topic."""
        gap = timedelta(minutes=self._rt.settings.groups.unit_gap_minutes)
        if msg.grouped_id is not None:
            # an album's parts are one message whoever sent them (an anonymous admin too)
            for unit in units.values():
                if any(m.grouped_id == msg.grouped_id for m in unit.members):
                    return unit
        if msg.reply_to_id is not None:
            for unit in units.values():
                if any(m.message_id == msg.reply_to_id for m in unit.members):
                    ok = unit.accepts(msg.date, gap) or unit.accepts_reply(msg.date)
                    return unit if ok else None
        if msg.sender_id is None:
            return None
        best: _Unit | None = None
        for unit in units.values():
            if unit.last.sender_id != msg.sender_id or unit.last.topic_id != msg.topic_id:
                continue
            if not timedelta(0) <= msg.date - unit.last.date < gap:
                continue
            if not unit.accepts(msg.date, gap):
                continue
            if best is None or unit.last.date > best.last.date:
                best = unit
        return best

    @staticmethod
    def _unit_candidate(unit: _Unit, chat: Chat | None, via: Via) -> Candidate:
        """Texts in date order joined by blank lines, urls united, the link points at the root.

        A unit whose root is an album is an album post like a channel's (§9.1): ``media`` is
        ``album`` and the link points at the part that carries the caption; ``root_id`` stays
        the unit's key in ``group_messages``.
        """
        members = sorted(unit.members, key=lambda m: (m.date, m.message_id))
        root = next((m for m in members if m.message_id == unit.root_id), members[0])
        album = (
            [m for m in members if m.grouped_id == root.grouped_id]
            if root.grouped_id is not None
            else []
        )
        link = root
        if len(album) > 1:
            link = next((m for m in album if m.text.strip()), album[0])
        texts = [m.text for m in members if m.text.strip()]
        html: str | None = None
        if any(m.html for m in members):
            html = "\n\n".join(
                m.html if m.html else html_escape(m.text) for m in members if m.text.strip()
            )
        return Candidate(
            chat_id=unit.members[0].chat_id,
            message_id=link.message_id,
            message_ids=[m.message_id for m in members],
            kind="unit",
            posted_at=unit.root_date,
            text="\n\n".join(texts),
            html=html,
            urls=_unique(u for m in members for u in m.urls),
            media="album" if len(album) > 1 else root.media,  # type: ignore[arg-type]
            grouped_id=root.grouped_id if len(album) > 1 else None,
            views=None,
            forwards=None,
            fwd_from_chat_id=root.fwd_from_chat_id,
            fwd_from_message_id=root.fwd_from_message_id,
            noforwards=chat.noforwards if chat is not None else False,
            via=via,
        )

    # --- submission --------------------------------------------------------------------------

    async def _submit(self, candidates: list[Candidate]) -> None:
        """Submit one by one; a failure never stops the rest (see ``_submit_failed``)."""
        if not candidates:
            return
        sorter = self._rt.sorter
        if sorter is None:
            log.warning("intake: no sorter wired, %d candidates dropped", len(candidates))
            return
        for candidate in candidates:
            try:
                await sorter.submit(candidate)
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._submit_failed(candidate)
            else:
                self._attempts.pop((candidate.chat_id, candidate.message_id), None)
                self._album_done(candidate)

    async def _submit_failed(self, c: Candidate) -> None:
        """Keep a candidate whose submit raised for the next ``tick``, ``SUBMIT_ATTEMPTS``
        times in all: a unit is reopened in ``group_messages`` (the next close rebuilds it),
        a channel post or album waits in memory. The sorter dedups, so a retry of a candidate
        that was stored before the error is harmless."""
        key = (c.chat_id, c.message_id)
        attempts = self._attempts.get(key, 0) + 1
        if attempts >= SUBMIT_ATTEMPTS:
            self._attempts.pop(key, None)
            self._album_done(c)
            log.exception(
                "intake: giving up on %s %d/%d after %d failed submits",
                c.kind,
                c.chat_id,
                c.message_id,
                attempts,
            )
            return
        self._attempts[key] = attempts
        log.exception(
            "intake: submit of %s %d/%d failed, retrying on the next tick",
            c.kind,
            c.chat_id,
            c.message_id,
        )
        if c.kind != "unit":
            self._retry.append(c)
            return
        try:
            async with self._group_lock(c.chat_id):
                # the rows keep the unit_root_id the close gave them (an album unit's
                # message_id is its caption part, not its root)
                await self._rt.store.mark_group_messages(c.chat_id, c.message_ids, closed=False)
                self._changed(c.chat_id)
        except Exception:
            log.exception("intake: could not reopen unit %d/%d", c.chat_id, c.message_id)
