"""Backfill: the last few days of every source chat, read gently, submitted in order (§9.1).

Chats are read one at a time with a pause between them and every ``FloodWait`` slept, so the
account looks like a person scrolling; the candidates of all chats are then submitted in
global ``posted_at`` order so that "first seen" matches what actually happened across chats.
Running it twice is harmless: posts and group messages dedup on ``(chat_id, message_id)`` and
counters are bumped only for new rows. A group message older than the group buffer's 3-day
retention is never bumped by intake, though: its row may have been counted and purged since,
so a fresh insert proves nothing. Instead, for every day the read fully covered,
``chat_daily`` is raised to the count read from history (``max(existing, read)``), which
repairs an undercounted day and cannot double count, however long the backfill.

A failure stays local: a chat whose read fails is skipped, but what it buffered is still
force-closed as a backfill; a candidate whose submit fails is logged (a unit is reopened for a
later close) and the rest are still submitted. A ``FloodWait`` is never a failure: the curator
waits as long as it is told and reads on (spec "Pacing"); only a chat whose waits add up to
more than ``FLOOD_DEFER_AFTER`` is put back until the other chats are read, then read again.

One run at a time: a second caller (``curator backfill`` during a ``/preview`` read, two CLI
runs) waits for the first, so chats are always read one at a time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import sqlalchemy as sa

from tg_curator.clock import local_date
from tg_curator.contracts import ProgressCallback
from tg_curator.db import schema
from tg_curator.domain import KV, BackfillResult, Candidate, Chat
from tg_curator.errors import ChatGone, FloodWait, NotAllowed
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import IncomingMessage

log = logging.getLogger(__name__)

PAUSE_BETWEEN_CHATS = 3.0
"""Seconds between two chats (spec "Pacing": history is read one chat at a time with pauses)."""
FLOOD_DEFER_AFTER = 6 * 3600.0
"""Seconds of ``FloodWait`` one chat may cost in one pass. Every wait is slept and the read
resumed; past this total the chat is put back and read again after the other chats, and only
skipped when that second pass is throttled as long again."""
FULL_RUN_DAYS = 3
"""A run over every source chat and at least this many days is what ``/preview`` and setup
step 6 rely on; only such a run writes ``kv backfill.last_run_at``."""
HISTORY_LIMIT = 2000
"""The gateway's per-chat cap (§5): when a read hits it, the oldest day read was cut short."""

Sleep = Callable[[float], Awaitable[None]]


class _ThrottledError(Exception):
    """One chat's ``FloodWait``s of this pass add up to more than ``FLOOD_DEFER_AFTER``."""


@dataclass
class _ChatRead:
    """What one chat's history read saw, for the counting of §9.1."""

    tz: str
    seen: set[int] = field(default_factory=set)
    newest: datetime | None = None
    per_day: dict[date, set[tuple[str, int]]] = field(default_factory=dict)

    @property
    def messages(self) -> int:
        return len(self.seen)

    def note(self, msg: IncomingMessage) -> bool:
        """Record a message; ``False`` when a retried read delivered it a second time."""
        if msg.message_id in self.seen:
            return False
        self.seen.add(msg.message_id)
        if self.newest is None or msg.date > self.newest:
            self.newest = msg.date
        ignored = (
            msg.is_service
            or (msg.is_outgoing and msg.chat.kind == "group")
            or msg.is_automatic_forward
        )
        if not ignored:
            key = ("album", msg.grouped_id) if msg.grouped_id is not None else ("m", msg.message_id)
            self.per_day.setdefault(local_date(msg.date, self.tz), set()).add(key)
        return True

    def covered_counts(self, since: datetime, now: datetime) -> dict[date, int]:
        """Message counts for the days the read fully covered: strictly between the local
        dates of ``since`` and ``now``, and never the oldest day when the cap was hit."""
        first, last = local_date(since, self.tz), local_date(now, self.tz)
        counts = {d: len(keys) for d, keys in self.per_day.items() if first < d < last}
        if self.messages >= HISTORY_LIMIT and self.per_day:
            counts.pop(min(self.per_day), None)
        return counts


class BackfillService:
    """The ``Backfill`` service of §8. ``sleep`` is injectable so tests never wait."""

    def __init__(self, rt: Runtime, *, sleep: Sleep = asyncio.sleep) -> None:
        self._rt = rt
        self._sleep = sleep
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """A run is under way; another ``run`` would wait for it."""
        return self._lock.locked()

    async def run(
        self,
        days: int = 3,
        *,
        chat_ids: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> BackfillResult:
        """Read, then submit in global order. Runs are serialised (a second caller waits)."""
        async with self._lock:
            return await self._run(days, chat_ids, progress)

    async def _run(
        self,
        days: int,
        chat_ids: Sequence[int] | None,
        progress: ProgressCallback | None,
    ) -> BackfillResult:
        rt = self._rt
        if rt.intake is None or rt.sorter is None or rt.user is None:
            raise RuntimeError("backfill needs intake, sorter and the user gateway wired")
        now = rt.clock.now()
        since = now - timedelta(days=days)
        tz = rt.settings.general.timezone
        chats, skipped = await self._target_chats(chat_ids)
        candidates: list[Candidate] = []
        reads: dict[int, _ChatRead] = {}
        deferred: list[Chat] = []
        for done, chat in enumerate(chats, start=1):
            if done > 1:
                await self._sleep(PAUSE_BETWEEN_CHATS)
            if not await self._read_chat(chat, since, now, tz, candidates, reads, skipped):
                deferred.append(chat)
            if progress is not None:
                await progress(done, len(chats), chat)
        for chat in deferred:
            # throttled for hours in the first pass: read once more after the others
            await self._sleep(PAUSE_BETWEEN_CHATS)
            log.info(
                "backfill: reading chat %d (%s) again after the long waits", chat.id, chat.title
            )
            if not await self._read_chat(chat, since, now, tz, candidates, reads, skipped):
                log.warning(
                    "backfill: skipping chat %d (%s): throttled too long", chat.id, chat.title
                )
                skipped.append(chat.id)
        submitted = failed = 0
        for candidate in sorted(candidates, key=lambda c: (c.posted_at, c.chat_id, c.message_id)):
            try:
                post = await rt.sorter.submit(candidate)
            except asyncio.CancelledError:
                raise
            except Exception:
                failed += 1
                log.exception(
                    "backfill: submit of %s %d/%d failed",
                    candidate.kind,
                    candidate.chat_id,
                    candidate.message_id,
                )
                await self._reopen(candidate)
                continue
            if post is not None:
                submitted += 1
        for chat_id, read in reads.items():
            await self._reconcile_daily(chat_id, read.covered_counts(since, now))
        if chat_ids is None and days >= FULL_RUN_DAYS:
            # a partial or short run must not stand in for the three-day read of every chat
            await rt.store.kv_set(KV.BACKFILL_LAST_RUN_AT, rt.clock.now().isoformat())
        result = BackfillResult(
            chats=len(reads),
            messages=sum(r.messages for r in reads.values()),
            candidates=len(candidates),
            submitted=submitted,
            skipped_chats=skipped,
            per_chat={cid: r.messages for cid, r in reads.items()},
        )
        log.info(
            "backfill: %d chats, %d messages, %d candidates, %d new posts, %d failed, "
            "%d chats skipped",
            result.chats,
            result.messages,
            result.candidates,
            result.submitted,
            failed,
            len(result.skipped_chats),
        )
        return result

    async def _read_chat(
        self,
        chat: Chat,
        since: datetime,
        now: datetime,
        tz: str,
        candidates: list[Candidate],
        reads: dict[int, _ChatRead],
        skipped: list[int],
    ) -> bool:
        """Read one chat into ``candidates``; ``False`` when it was throttled too long and
        should be read again later (what it buffered is closed either way)."""
        intake = self._rt.intake
        assert intake is not None
        read = _ChatRead(tz=tz)
        try:
            candidates += await intake.collect(self._history(chat.id, since, read), via="backfill")
        except _ThrottledError:
            log.warning(
                "backfill: chat %d (%s) asked for more than %d s of waits; reading it again later",
                chat.id,
                chat.title,
                int(FLOOD_DEFER_AFTER),
            )
            return False
        except (ChatGone, NotAllowed) as exc:
            log.warning("backfill: skipping chat %d (%s): %s", chat.id, chat.title, exc)
            skipped.append(chat.id)
        except Exception:
            log.exception("backfill: skipping chat %d (%s)", chat.id, chat.title)
            skipped.append(chat.id)
        else:
            reads[chat.id] = read
            log.info(
                "backfill: read %d messages from chat %d (%s)",
                read.messages,
                chat.id,
                chat.title,
            )
        finally:
            # even after a failed read: what was buffered closes as a backfill now and
            # the chat is released, so the live tick never takes it as live (§9.1)
            candidates += await self._close_chat(chat.id, read.newest or now)
        return True

    async def _close_chat(self, chat_id: int, newest: datetime) -> list[Candidate]:
        """Force-close the chat's units; a failure here is logged, never fatal to the run."""
        intake = self._rt.intake
        assert intake is not None
        try:
            return await intake.close_units(chat_id, newest, force=True)
        except Exception:
            log.exception("backfill: closing the units of chat %d failed", chat_id)
            return []

    async def _reopen(self, candidate: Candidate) -> None:
        """A unit whose submit failed is reopened, so a later close (the live tick, as a
        backfill since intake remembers it, or a re-run) builds and submits it again; a channel
        post needs nothing, a re-run reads it again."""
        if candidate.kind != "unit":
            return
        try:
            # the rows keep the unit_root_id the close gave them (an album unit's
            # message_id is its caption part, not its root)
            await self._rt.store.mark_group_messages(
                candidate.chat_id, candidate.message_ids, closed=False
            )
        except Exception:
            log.exception(
                "backfill: could not reopen unit %d/%d", candidate.chat_id, candidate.message_id
            )

    # --- which chats -------------------------------------------------------------------------

    async def _target_chats(self, chat_ids: Sequence[int] | None) -> tuple[list[Chat], list[int]]:
        """Active source chats, or exactly the requested ones (resolved when not yet known)."""
        rt = self._rt
        if chat_ids is None:
            return await rt.store.list_chats(role="source", active=True), []
        chats: list[Chat] = []
        skipped: list[int] = []
        for chat_id in dict.fromkeys(chat_ids):
            chat = await rt.store.get_chat(chat_id)
            if chat is None:
                try:
                    info = await rt.user.resolve_chat(chat_id)  # type: ignore[union-attr]
                except (ChatGone, NotAllowed) as exc:
                    log.warning("backfill: chat %d cannot be read: %s", chat_id, exc)
                    skipped.append(chat_id)
                    continue
                chat = await rt.store.upsert_chat(info)
            if chat.role != "source":
                log.warning("backfill: chat %d is an %s channel, not a source", chat_id, chat.role)
                skipped.append(chat_id)
                continue
            chats.append(chat)
        return chats, skipped

    # --- reading -----------------------------------------------------------------------------

    async def _history(
        self, chat_id: int, since: datetime, read: _ChatRead
    ) -> AsyncIterator[IncomingMessage]:
        """The chat's history with every ``FloodWait`` slept and the read resumed from the
        start; ``read`` drops the messages a retry delivers again. Raises ``_ThrottledError`` once
        the waits of this pass would pass ``FLOOD_DEFER_AFTER``."""
        waited = 0.0
        while True:
            try:
                async for msg in self._rt.user.history(chat_id, since=since):  # type: ignore[union-attr]
                    if read.note(msg):
                        yield msg
                return
            except FloodWait as exc:
                if waited + exc.seconds > FLOOD_DEFER_AFTER:
                    raise _ThrottledError(chat_id) from exc
                waited += exc.seconds
                log.info("backfill: chat %d: waiting %d s as Telegram asks", chat_id, exc.seconds)
                await self._sleep(float(exc.seconds))

    # --- counting ----------------------------------------------------------------------------

    async def _reconcile_daily(self, chat_id: int, counts: dict[date, int]) -> None:
        """``chat_daily.messages = max(existing, counted)`` for the fully covered days.

        Runs after submission, so the sorter's bumps for new posts are already in
        ``existing``; in the normal case this changes nothing and it only repairs a day that
        was undercounted (a lost buffer, an interrupted earlier run).
        """
        if not counts:
            return
        store = self._rt.store
        table = schema.chat_daily
        async with store.begin() as conn:
            for day, counted in counts.items():
                if counted == 0:
                    continue
                where = sa.and_(table.c.chat_id == chat_id, table.c.day == day)
                rows = await store.execute(sa.select(table.c.messages).where(where), conn)
                if not rows:
                    await store.bump_chat_daily(chat_id, day, counted, conn=conn)
                elif rows[0].messages < counted:
                    await store.execute(
                        sa.update(table).where(where).values(messages=counted), conn
                    )
                    log.info(
                        "backfill: chat %d day %s counted %d messages, had %d",
                        chat_id,
                        day.isoformat(),
                        counted,
                        rows[0].messages,
                    )
