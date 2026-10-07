"""The daily digest: ranking, composition, sending and schedule (DESIGN §9.5).

One run handles one topic and is the same code for the scheduled tick and a manual send;
every run is serialised by one lock so two of them never fight over the same pool. The
invariants that matter are crash-safety ones: a post enters a digest only through the
``UPDATE … WHERE status='digest'`` guard inside the composition transaction, so a post promoted
to real time meanwhile is excluded rather than shown twice; message ids are committed part by
part, so a restart resends only what Telegram never acknowledged; and a part is never resent
without first looking for its unique header in the channel.

Views are read by the account only to compare each candidate with what its source normally
gets (``chats.views_baseline``); nothing is ever incremented and a failed read simply gives
no attention score, so the digest still goes out.
"""

from __future__ import annotations

import asyncio
import html
import logging
import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from tg_curator.clock import local_date, scheduled_moment
from tg_curator.db import schema
from tg_curator.domain import (
    DIGEST_FAILED,
    DIGEST_PENDING,
    DIGEST_SENDING,
    DIGEST_SENT,
    KV,
    Chat,
    Digest,
    DigestDraft,
    DigestLine,
    DigestResult,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.errors import BotCannotPost, ChatGone, CuratorError, FloodWait
from tg_curator.pipeline import render
from tg_curator.textutil import first_line

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

VIEWS_BATCH = 100
"""Ids per ``get_views`` call (Telegram's limit, §5)."""
BASELINE_MIN_SAMPLES = 5
BASELINE_MIN_AGE = timedelta(hours=12)
"""A post younger than this is still collecting views and would drag the median down; for the
same reason only a read taken at least this long after posting is a baseline sample (the count
stored at intake, minutes after posting, is not)."""
BASELINE_LOOKBACK = timedelta(days=3)
BASELINE_REFRESH_PER_CHAT = 30
"""Older posts (12 h – 3 days) without a mature read whose views are re-read per chat and run,
so even a busy channel has enough samples after a run or two; they share the candidates'
``get_views`` calls (100 ids each)."""
MAX_DIGEST_PARTS = 2
"""A digest longer than one message is split into two, never more: the lowest-ranked items are
left out until it fits (they are dropped like any other leftover)."""
SUMMARY_CONCURRENCY = 4
ATTENTION_MIN, ATTENTION_MAX = -2.0, 3.0
AGE_SCALE_HOURS = 8.0
RETRY_BASE_SECONDS = 30
RETRY_CAP_SECONDS = 600
CANNOT_POST_RETRY = timedelta(minutes=10)

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class _Draft:
    """Steps 1–4 of a run: what would be sent, before anything is written."""

    candidates: list[Post]
    older_ids: list[int]
    lines: list[DigestLine]
    parts: list[str]
    chats: dict[int, Chat]
    posts: dict[int, Post]


@dataclass
class _Outcome:
    """One run's result plus what the owner line needs: the topic name and whether the
    message really reached the channel (a composed-but-failed digest is not announced)."""

    result: DigestResult
    topic_name: str
    manual: bool
    sent: bool


def _header_text(part: str) -> str:
    """The plain text of a part's first line: the unique per-part header ``reconcile`` looks
    for (§9.5 step 4)."""
    first = part.split("\n", 1)[0]
    return html.unescape(_TAG_RE.sub("", first)).strip()


def _kv_datetime(value: Any) -> datetime | None:
    """``kv service.went_live_at`` as written by ``/go``: an ISO-8601 string (naive = UTC) or an
    epoch number; anything else counts as "never went live", which only skips a digest."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, UTC)
    return None


def _aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; every stored datetime is UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _mature(posted_at: datetime, views_at: datetime | None) -> bool:
    """Whether a stored view count was read late enough to stand for what the post got."""
    return views_at is not None and _aware(views_at) - _aware(posted_at) >= BASELINE_MIN_AGE


def _chunks(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class DigestService:
    """Ranking, composition, sending and schedule of the daily digest (§9.5)."""

    VIEWS_GAP_SECONDS = 1.0
    """One ``get_views`` call per second (§5); a class attribute so tests can zero it."""
    PART_GAP_SECONDS = 3.0
    """The publisher's global gap between two sends, honoured between digest parts (§9.4)."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._lock = asyncio.Lock()
        self._warned_channels: set[int] = set()
        self._last_views_call: float | None = None
        self._last_part_sent: float | None = None
        self._views_until: datetime | None = None

    # --- schedule ------------------------------------------------------------------------------

    def _moment(self, day: date) -> datetime:
        settings = self._rt.settings
        return scheduled_moment(
            day, settings.digest.hour, settings.digest.minute, settings.general.timezone
        )

    def _due_day(self, now: datetime) -> date:
        """``today`` once the hour has passed, else ``yesterday`` (whose run may be missing)."""
        today = local_date(now, self._rt.settings.general.timezone)
        return today if now >= self._moment(today) else today - timedelta(days=1)

    def next_run(self) -> datetime:
        now = self._rt.clock.now()
        today = local_date(now, self._rt.settings.general.timezone)
        moment = self._moment(today)
        return moment if moment > now else self._moment(today + timedelta(days=1))

    # --- the public entry points ---------------------------------------------------------------

    async def preview(self, topic_key: str | None = None) -> list[DigestDraft]:
        """Tonight's digest as drafts: steps 1–4 only, no row, no status change (the summary
        cache and the views baseline may be filled, as they would be by the real run)."""
        day = local_date(self.next_run(), self._rt.settings.general.timezone)
        drafts: list[DigestDraft] = []
        for topic in await self._topics(topic_key):
            async with self._lock:
                draft = await self._compose(topic, day, seq=0)
            drafts.append(DigestDraft(topic.key, day, draft.parts, draft.lines))
        return drafts

    async def send(self, topic_key: str | None = None) -> list[DigestResult]:
        """The manual digest: a new ``seq >= 1`` row per topic over the current pool.

        Refused with ``CuratorError`` carrying the one catalogue sentence when publishing is
        not live or paused. Leftover candidates stay in the pool for the scheduled run.
        """
        if not self._rt.settings.publishing.live:
            raise CuratorError(self._rt.t("not_live"))
        if await self._paused():
            raise CuratorError(self._rt.t("paused"))
        day = local_date(self._rt.clock.now(), self._rt.settings.general.timezone)
        outcomes = [
            await self._run(topic, day, manual=True) for topic in await self._topics(topic_key)
        ]
        await self._owner_lines(outcomes)
        return [o.result for o in outcomes]

    async def tick(self) -> None:
        """Retry failed rows, then compose the scheduled digest of every topic that is due and
        has none yet (today's once the hour passed, else yesterday's — an outage never loses a
        day). Nothing happens while publishing is not live or paused."""
        if not self._rt.settings.publishing.live or await self._paused():
            return
        now = self._rt.clock.now()
        outcomes = await self._retry_failed(now)
        due = self._due_day(now)
        moment = self._moment(due)
        went_live_at = _kv_datetime(await self._rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT))
        if went_live_at is not None and went_live_at > moment:
            log.debug("digest: went live after the %s hour; first digest tomorrow", due)
        else:
            for topic in await self._topics(None):
                if topic.created_at > moment:
                    continue  # created after today's hour: its first digest is tomorrow
                if await self._rt.store.get_digest_by_key(topic.id, due, seq=0) is not None:
                    continue  # already composed (any state; a failed one was retried above)
                outcomes.append(await self._run(topic, due, manual=False))
        await self._owner_lines(outcomes)

    async def reconcile(self) -> None:
        """After a restart: rows caught between composition and ``sent`` are finished, each
        missing part first looked up by its header so a part Telegram acknowledged before the
        crash is never sent twice."""
        for row in await self._rt.store.digests_in_state([DIGEST_SENDING, DIGEST_PENDING]):
            topic = await self._rt.store.get_topic(row.topic_id)
            async with self._lock:
                await self._send(row, topic, verify=True)

    # --- one run for one topic -----------------------------------------------------------------

    async def _run(self, topic: Topic, day: date, *, manual: bool) -> _Outcome:
        async with self._lock:
            nothing = DigestResult(topic.key, day, 0, 0, [], self._rt.t("nothing_to_do"))
            if topic.channel_id is None:
                return _Outcome(nothing, topic.name, manual, sent=False)
            seq = await self._rt.store.next_manual_seq(topic.id, day) if manual else 0
            # a scheduled run covers the window that ends at its own hour, even when it runs
            # late after an outage: yesterday's digest stays about yesterday
            until = None if manual else self._moment(day)
            draft = await self._compose(topic, day, seq, until=until)
            digest = await self._commit(topic, day, seq, manual, draft)
            if digest is None or digest.item_count == 0:
                return _Outcome(replace(nothing, seq=seq), topic.name, manual, sent=False)
            sent = await self._send(digest, topic, verify=False)
            stored = await self._rt.store.get_digest(digest.id)
            message_ids = stored.message_ids if stored is not None else []
            result = DigestResult(topic.key, day, seq, digest.item_count, message_ids, None)
            return _Outcome(result, topic.name, manual, sent)

    async def _compose(
        self, topic: Topic, day: date, seq: int, *, until: datetime | None = None
    ) -> _Draft:
        """Steps 1–4: candidates, views, rank, lines, render. Gateway and LLM awaits happen
        here, outside any transaction.

        The window is the ``window_hours`` before ``until``: the scheduled moment of ``day``
        for a scheduled run (so a run composed late still covers that day), now for a manual
        run or a preview. Posts newer than ``until`` are neither candidates nor leftovers:
        they stay in the pool for the next digest.
        """
        settings = self._rt.settings
        now = self._rt.clock.now()
        until = now if until is None or until > now else until
        since = until - timedelta(hours=settings.digest.window_hours)
        pool = await self._rt.store.posts_by_status(PostStatus.digest, topic_id=topic.id)
        candidates = [p for p in pool if since <= p.posted_at <= until]
        older_ids = [p.id for p in pool if p.posted_at < since]
        chats = await self._chats_of(candidates)
        fresh_views = await self._refresh_views(candidates, chats, now)
        ranked = sorted(
            candidates,
            key=lambda p: (
                -self._rank(p, chats.get(p.chat_id), fresh_views.get(p.id), now),
                -p.posted_at.timestamp(),
                -p.id,
            ),
        )
        top = ranked[: settings.digest.items]
        lines = await self._lines(top)
        posts = {p.id: p for p in top}
        parts = self._render(day, topic, lines, chats, posts, seq)
        while len(parts) > MAX_DIGEST_PARTS and len(lines) > 1:
            lines = lines[:-1]  # the lowest-ranked item; it is a leftover like the rest
            parts = self._render(day, topic, lines, chats, posts, seq)
        if len(lines) < len(top):
            log.info(
                "digest: topic %s: %d item(s) left out to keep the digest within %d messages",
                topic.key,
                len(top) - len(lines),
                MAX_DIGEST_PARTS,
            )
            posts = {line.post_id: posts[line.post_id] for line in lines}
        log.info(
            "digest: topic %s day %s seq %d: %d candidates, %d picked",
            topic.key,
            day,
            seq,
            len(candidates),
            len(lines),
        )
        return _Draft(candidates, older_ids, lines, parts, chats, posts)

    def _render(
        self,
        day: date,
        topic: Topic,
        lines: list[DigestLine],
        chats: dict[int, Chat],
        posts: dict[int, Post],
        seq: int,
    ) -> list[str]:
        if not lines:
            return []
        return render.digest_message(
            day, topic.name, lines, chats, seq=seq, posts=posts, t=self._rt.t
        )

    async def _commit(
        self, topic: Topic, day: date, seq: int, manual: bool, draft: _Draft
    ) -> Digest | None:
        """Step 5 in one transaction: the status guard, the row, the items, the drop.

        An item whose post is no longer ``digest`` in this topic (promoted, or corrected into
        another topic meanwhile) is removed and the body re-rendered before the commit. A
        scheduled run drops what it did not pick, so nothing rolls over; a manual run leaves
        the rest for the scheduled one (§14.4). An empty scheduled run records a ``sent`` row
        with 0 items; an empty manual run writes nothing.
        """
        store = self._rt.store
        now = self._rt.clock.now()
        assert topic.channel_id is not None  # checked by _run; the invariant of §9.7
        async with store.begin() as conn:
            kept: list[DigestLine] = []
            for line in draft.lines:
                flipped = await store.execute(
                    sa.update(schema.posts)
                    .where(schema.posts.c.id == line.post_id)
                    .where(schema.posts.c.status == PostStatus.digest.value)
                    .where(schema.posts.c.topic_id == topic.id)  # a correction meanwhile wins
                    .values(status=PostStatus.digested.value, decided_at=now),
                    conn,
                )
                if flipped:
                    kept.append(line)
            parts = draft.parts
            if len(kept) != len(draft.lines):
                log.info(
                    "digest: topic %s: %d item(s) promoted during composition, left out",
                    topic.key,
                    len(draft.lines) - len(kept),
                )
                kept = [replace(line, position=i) for i, line in enumerate(kept, 1)]
                parts = self._render(day, topic, kept, draft.chats, draft.posts, seq)
            if not manual:
                await self._drop_rest(topic, draft, conn)
            if not kept:
                if manual:
                    return None
                row = await store.create_digest(
                    topic.id,
                    topic.channel_id,
                    day,
                    body=[],
                    item_count=0,
                    seq=seq,
                    manual=False,
                    state=DIGEST_SENT,
                    conn=conn,
                )
                await store.set_digest_fields(row.id, conn=conn, sent_at=now)
                return replace(row, sent_at=now)
            row = await store.create_digest(
                topic.id,
                topic.channel_id,
                day,
                body=parts,
                item_count=len(kept),
                seq=seq,
                manual=manual,
                conn=conn,
            )
            await store.add_digest_items(row.id, kept, conn=conn)
        return row

    async def _drop_rest(self, topic: Topic, draft: _Draft, conn: AsyncConnection) -> None:
        leftover = [p.id for p in draft.candidates] + draft.older_ids
        if not leftover:
            return
        dropped = await self._rt.store.execute(
            sa.update(schema.posts)
            .where(schema.posts.c.id.in_(leftover))
            .where(schema.posts.c.status == PostStatus.digest.value)
            .where(schema.posts.c.topic_id == topic.id)  # corrected away meanwhile: not ours
            .values(status=PostStatus.dropped.value),
            conn,
        )
        if dropped:
            log.info("digest: %d post(s) did not make the cut and were dropped", dropped)

    # --- step 1: views and the per-chat baseline -----------------------------------------------

    async def _chats_of(self, posts: Sequence[Post]) -> dict[int, Chat]:
        chats: dict[int, Chat] = {}
        for chat_id in {p.chat_id for p in posts}:
            chat = await self._rt.store.get_chat(chat_id)
            if chat is not None:
                chats[chat_id] = chat
        return chats

    async def _refresh_views(
        self, candidates: Sequence[Post], chats: dict[int, Chat], now: datetime
    ) -> dict[int, int]:
        """Re-read the views of the candidates (plus a few older posts per chat for the
        baseline) through the account; returns ``post_id -> views`` for the posts whose read
        succeeded — only those get an attention score. ``chats`` is updated in place with any
        new baseline."""
        user = self._rt.user
        fresh: dict[int, int] = {}
        if user is None:
            return fresh
        if self._views_until is not None and self._rt.clock.now() < self._views_until:
            log.info("digest: views not read until %s (Telegram asked to wait)", self._views_until)
            return fresh
        by_chat: dict[int, list[Post]] = {}
        for post in candidates:
            by_chat.setdefault(post.chat_id, []).append(post)
        for chat_id, posts in by_chat.items():
            chat = chats.get(chat_id)
            if chat is None or chat.kind != "channel":
                continue  # groups have no views; an unknown chat has no baseline to keep
            by_message = {p.message_id: p.id for p in posts}
            for post_id, message_id in await self._baseline_posts(chat_id, now):
                by_message.setdefault(message_id, post_id)
            got: dict[int, int] = {}
            flood: FloodWait | None = None
            try:
                for chunk in _chunks(list(by_message), VIEWS_BATCH):
                    await self._pace("_last_views_call", self.VIEWS_GAP_SECONDS)
                    got.update(await user.get_views(chat_id, chunk))
            except FloodWait as exc:
                # Telegram asks the account to slow down: no further read until the wait is
                # over, in this run or the next; what was read so far is kept
                flood = exc
                self._views_until = self._rt.clock.now() + timedelta(seconds=exc.seconds)
                log.warning(
                    "digest: Telegram asked to wait %d s; views skipped until %s, attention 0",
                    exc.seconds,
                    self._views_until,
                )
            except CuratorError as exc:
                log.warning("digest: views of chat %d not read (%s); attention 0", chat_id, exc)
                continue
            if got:
                await self._store_views(chat, posts, by_message, got, chats, fresh, now)
            if flood is not None:
                break
        return fresh

    async def _store_views(
        self,
        chat: Chat,
        posts: list[Post],
        by_message: dict[int, int],
        got: dict[int, int],
        chats: dict[int, Chat],
        fresh: dict[int, int],
        now: datetime,
    ) -> None:
        """Commit one chat's fresh views and its new baseline; ``fresh`` gets the candidates'."""
        chat_id = chat.id
        candidate_ids = {p.id for p in posts}
        async with self._rt.store.begin() as conn:
            for message_id, views in got.items():
                post_id = by_message.get(message_id)
                if post_id is None:
                    continue
                await self._rt.store.set_post_fields(post_id, conn=conn, views=views, views_at=now)
                if post_id in candidate_ids:
                    fresh[post_id] = views
            baseline = await self._baseline(chat_id, now, conn)
            if baseline is not None:
                await self._rt.store.set_chat_fields(
                    chat_id, conn=conn, views_baseline=baseline, baseline_at=now
                )
                chats[chat_id] = replace(chat, views_baseline=baseline, baseline_at=now)

    async def _baseline_posts(self, chat_id: int, now: datetime) -> list[tuple[int, int]]:
        """Up to ``BASELINE_REFRESH_PER_CHAT`` posts of the chat aged 12 h – 3 days that have
        no mature read yet (newest first), re-read so they become baseline samples."""
        stmt = (
            sa.select(
                schema.posts.c.id,
                schema.posts.c.message_id,
                schema.posts.c.posted_at,
                schema.posts.c.views_at,
            )
            .where(schema.posts.c.chat_id == chat_id)
            .where(schema.posts.c.posted_at >= now - BASELINE_LOOKBACK)
            .where(schema.posts.c.posted_at < now - BASELINE_MIN_AGE)
            .order_by(schema.posts.c.posted_at.desc(), schema.posts.c.id.desc())
        )
        rows = await self._rt.store.execute(stmt)
        assert isinstance(rows, list)
        immature = [row for row in rows if not _mature(row.posted_at, row.views_at)]
        return [(row.id, row.message_id) for row in immature[:BASELINE_REFRESH_PER_CHAT]]

    async def _baseline(self, chat_id: int, now: datetime, conn: AsyncConnection) -> float | None:
        """Median views of the chat's posts older than 12 h (within the lookback), counting
        only reads taken at least 12 h after posting; ``None`` below five samples, in which
        case the stored baseline is left as it is."""
        stmt = (
            sa.select(schema.posts.c.views, schema.posts.c.posted_at, schema.posts.c.views_at)
            .where(schema.posts.c.chat_id == chat_id)
            .where(schema.posts.c.views.is_not(None))
            .where(schema.posts.c.posted_at >= now - BASELINE_LOOKBACK)
            .where(schema.posts.c.posted_at < now - BASELINE_MIN_AGE)
        )
        rows = await self._rt.store.execute(stmt, conn)
        assert isinstance(rows, list)
        samples = [int(row.views) for row in rows if _mature(row.posted_at, row.views_at)]
        if len(samples) < BASELINE_MIN_SAMPLES:
            return None
        return float(statistics.median(samples))

    # --- step 2: the rank ----------------------------------------------------------------------

    def _rank(self, post: Post, chat: Chat | None, views: int | None, now: datetime) -> float:
        """``2·ln(1+corroboration) + attention + 0.5·(trust − neutral)`` (§9.5 step 2)."""
        neutral = self._rt.settings.sorting.neutral_trust
        trust = chat.trust if chat is not None and chat.trust is not None else neutral
        attention = 0.0
        baseline = chat.views_baseline if chat is not None else None
        if views is not None and baseline:
            age_hours = max((now - post.posted_at).total_seconds() / 3600.0, 0.0)
            age_factor = 1.0 - math.exp(-age_hours / AGE_SCALE_HOURS)
            if age_factor > 0.0:
                if views <= 0:
                    attention = ATTENTION_MIN
                else:
                    ratio = views / (baseline * age_factor)
                    attention = min(max(math.log2(ratio), ATTENTION_MIN), ATTENTION_MAX)
        return 2.0 * math.log1p(post.corroboration) + attention + 0.5 * (trust - neutral)

    # --- step 3: the lines ---------------------------------------------------------------------

    async def _lines(self, top: Sequence[Post]) -> list[DigestLine]:
        """One line per pick: with a model, the cached summary or the LLM's sentence (cached,
        at most four requests in flight); without one (or when it fails), the first line of
        the post. Every line is cut at the current ``line_chars``, so a summary cached under
        a longer setting never shows at its old length, and without a model the digest is
        extractive whatever the cache holds."""
        max_chars = self._rt.settings.digest.line_chars
        llm = self._rt.llm
        semaphore = asyncio.Semaphore(SUMMARY_CONCURRENCY)

        async def line_of(post: Post) -> str:
            if llm.enabled:
                if post.summary:
                    return first_line(post.summary, max_chars)
                async with semaphore:
                    summary = await llm.summarise_line(post.text, max_chars)
                if summary:
                    await self._rt.store.set_post_fields(post.id, summary=summary)
                    return first_line(summary, max_chars)
            return first_line(post.text, max_chars)

        texts = await asyncio.gather(*(line_of(p) for p in top))
        pairs = zip(top, texts, strict=True)
        return [DigestLine(i, p.id, text) for i, (p, text) in enumerate(pairs, 1)]

    # --- step 6: sending -----------------------------------------------------------------------

    async def _send(self, digest: Digest, topic: Topic | None, *, verify: bool) -> bool:
        """Send the parts not yet in ``message_ids``, committing each id before the next part.

        ``verify`` (retries and ``reconcile``) looks each missing part up by its header first,
        so a part whose id was lost to a crash is adopted rather than sent again; a lookup
        that fails counts as a transient failure, never as "not found". Returns whether the
        row reached ``sent``.
        """
        store = self._rt.store
        await store.set_digest_fields(digest.id, state=DIGEST_SENDING)
        message_ids = list(digest.message_ids)
        try:
            bot = self._rt.bot
            if bot is None:
                raise CuratorError("the bot is not running")
            for index in range(len(message_ids), len(digest.body)):
                part = digest.body[index]
                message_id = await self._find_part(digest, part) if verify else None
                if message_id is None:
                    await self._pace("_last_part_sent", self.PART_GAP_SECONDS)
                    message_id = await bot.send_text(digest.channel_id, part)
                message_ids.append(message_id)
                await store.set_digest_fields(digest.id, message_ids=message_ids)
        except (BotCannotPost, ChatGone) as exc:
            await self._fail(digest, exc, CANNOT_POST_RETRY)
            await self._warn_cannot_post(digest.channel_id, topic)
            return False
        except FloodWait as exc:
            await self._fail(digest, exc, timedelta(seconds=exc.seconds))
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            backoff = min(RETRY_BASE_SECONDS * 2**digest.attempts, RETRY_CAP_SECONDS)
            await self._fail(digest, exc, timedelta(seconds=backoff))
            return False
        now = self._rt.clock.now()
        await store.set_digest_fields(digest.id, state=DIGEST_SENT, sent_at=now, last_error=None)
        self._warned_channels.discard(digest.channel_id)
        log.info(
            "digest: %s sent %d item(s) in %d part(s) to channel %d (day %s, seq %d)",
            topic.key if topic is not None else f"topic {digest.topic_id}",
            digest.item_count,
            len(message_ids),
            digest.channel_id,
            digest.day,
            digest.seq,
        )
        return True

    async def _find_part(self, digest: Digest, part: str) -> int | None:
        user = self._rt.user
        if user is None:
            raise CuratorError("no account session to look for an already sent digest part")
        # ``since`` means "dated after": a part sent within the same second as the row was
        # created must still be found, so the search starts one second earlier.
        return await user.find_message(
            digest.channel_id,
            contains=_header_text(part),
            since=digest.created_at - timedelta(seconds=1),
        )

    async def _fail(self, digest: Digest, exc: BaseException, retry_in: timedelta) -> None:
        now = self._rt.clock.now()
        attempts = digest.attempts + 1
        await self._rt.store.set_digest_fields(
            digest.id,
            state=DIGEST_FAILED,
            attempts=attempts,
            next_attempt_at=now + retry_in,
            last_error=str(exc)[:200],
        )
        log.warning(
            "digest: sending to channel %d failed (attempt %d, retry in %s): %s",
            digest.channel_id,
            attempts,
            retry_in,
            exc,
        )

    async def _warn_cannot_post(self, channel_id: int, topic: Topic | None) -> None:
        """Once per channel until a send into it succeeds (§11.4 item 4)."""
        if channel_id in self._warned_channels:
            return
        self._warned_channels.add(channel_id)
        chat = await self._rt.store.get_chat(channel_id)
        title = chat.title if chat is not None else topic.name if topic else str(channel_id)
        await self._rt.notifier.cannot_post(title)

    async def _retry_failed(self, now: datetime) -> list[_Outcome]:
        """Re-send ``failed`` rows past ``next_attempt_at`` (scheduled and manual alike); the
        body is never dropped and the items stay ``digested``. Successful retries join the
        owner line, since their digest was never announced."""
        outcomes: list[_Outcome] = []
        for row in await self._rt.store.digests_in_state(DIGEST_FAILED):
            if row.next_attempt_at is not None and row.next_attempt_at > now:
                continue
            topic = await self._rt.store.get_topic(row.topic_id)
            async with self._lock:
                sent = await self._send(row, topic, verify=True)
            if sent and topic is not None:
                stored = await self._rt.store.get_digest(row.id)
                result = DigestResult(
                    topic.key,
                    row.day,
                    row.seq,
                    row.item_count,
                    stored.message_ids if stored is not None else [],
                    None,
                )
                outcomes.append(_Outcome(result, topic.name, row.manual, sent=True))
        return outcomes

    # --- step 7: the owner line ----------------------------------------------------------------

    async def _owner_lines(self, outcomes: Sequence[_Outcome]) -> None:
        """``ML & AI: 15 posts; Fintech: 9`` for the topics that really went out, ``(manual)``
        prefixed for manual runs; topics with nothing sent are omitted."""
        for manual in (False, True):
            counted = [o for o in outcomes if o.manual == manual and o.sent and o.result.item_count]
            if not counted:
                continue
            pieces = [f"{counted[0].topic_name}: {counted[0].result.item_count} posts"]
            pieces += [f"{o.topic_name}: {o.result.item_count}" for o in counted[1:]]
            summary = "; ".join(pieces)
            if manual:
                summary = f"(manual) {summary}"
            await self._rt.notifier.digest_line(summary)

    # --- helpers -------------------------------------------------------------------------------

    async def _topics(self, key: str | None) -> list[Topic]:
        """Active topics with a channel, or the one named; an unknown key is refused with the
        catalogue's "not available" sentence."""
        topics = await self._rt.store.list_topics(active=True)
        if key is None:
            return [t for t in topics if t.channel_id is not None]
        for topic in topics:
            if topic.key == key:
                return [topic]
        raise CuratorError(self._rt.t("unknown_choice"))

    async def _paused(self) -> bool:
        return bool(await self._rt.store.kv_get(KV.SERVICE_PAUSED, False))

    async def _pace(self, attr: str, gap: float) -> None:
        """Sleep until ``gap`` seconds passed since the last call tracked in ``attr``; the loop
        clock is used so a frozen test clock never forces a real wait."""
        loop = asyncio.get_running_loop()
        last: float | None = getattr(self, attr)
        if last is not None:
            wait = gap - (loop.time() - last)
            if wait > 0:
                await asyncio.sleep(wait)
        setattr(self, attr, loop.time())
