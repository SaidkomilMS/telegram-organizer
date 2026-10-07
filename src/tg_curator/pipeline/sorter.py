"""The live sorter: persists the engine's decisions, routes, holds and promotes (DESIGN §9.3).

One worker processes candidates in arrival order because decisions depend on order (the first
copy of a story is the root, later ones corroborate it). Each submit is: embed in a thread (a
burst is embedded as one batch), decide, persist the post with its decision and the volume
count in one transaction, index it, and only after the commit talk to the publisher. The
in-memory ``RecentIndex`` is rebuilt from the database at start, so a restart changes nothing.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import date, datetime, timedelta

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from tg_curator.clock import local_date
from tg_curator.config import Settings
from tg_curator.contracts import Publisher
from tg_curator.db import schema
from tg_curator.domain import Candidate, Chat, Decision, NewPost, Post, PostStatus
from tg_curator.pipeline.engine import (
    DecisionEngine,
    Features,
    RecentIndex,
    candidate_from_post,
    detect_language,
    embedding_from_bytes,
    embedding_to_bytes,
    features_of,
    fwd_key_of,
    numbers_in,
)
from tg_curator.runtime import Runtime
from tg_curator.textutil import canonical_urls

log = logging.getLogger(__name__)

EMBED_BATCH = 32
"""Candidates embedded per model call during a burst; bounds the latency of the first one."""
PROMOTABLE = (PostStatus.held, PostStatus.digest)
"""Statuses a live root may be promoted from (§9.3): never digested, queued or published."""
SELF_HEAL = (PostStatus.held, PostStatus.digest, PostStatus.queued)
"""Statuses the self-healing pass looks at for a live post that should be in the outbox."""

_Item = tuple[Candidate, asyncio.Future[Post | None]]


class Sorter:
    """``Sorter`` of §8: ``submit``, ``tick`` and ``resort_unsorted`` over the runtime."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._index: RecentIndex | None = None
        self._lock = asyncio.Lock()
        self._queue: deque[_Item] = deque()

    @property
    def index(self) -> RecentIndex | None:
        """The recent index once built (``None`` before the first submit, tick or ``start``)."""
        return self._index

    async def start(self) -> None:
        """Build the index from the stored window now rather than on the first submit."""
        async with self._lock:
            await self._ensure_index()

    # --- submit ------------------------------------------------------------------------------

    async def submit(self, c: Candidate) -> Post | None:
        """Full decision, persisted, routed; ``None`` when the candidate already existed.

        The lock is the single worker: whoever holds it drains the whole queue, so a burst of
        submits is embedded in one batch and decided in arrival order while the other callers
        simply wait for their result.
        """
        future: asyncio.Future[Post | None] = asyncio.get_running_loop().create_future()
        self._queue.append((c, future))
        async with self._lock:
            if not future.done():
                await self._drain()
        return await future

    async def _drain(self) -> None:
        while self._queue:
            batch = [self._queue.popleft() for _ in range(min(len(self._queue), EMBED_BATCH))]
            try:
                await self._ensure_index()
                await self._process_batch(batch)
            except Exception as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for _, future in batch:
                    if not future.done():
                        future.cancel()

    async def _process_batch(self, batch: list[_Item]) -> None:
        store = self._rt.store
        work: list[_Item] = []
        for c, future in batch:
            if await store.get_post_by_message(c.chat_id, c.message_id) is not None:
                log.debug("sorter: %d/%d already stored, nothing to do", c.chat_id, c.message_id)
                future.set_result(None)
            else:
                work.append((c, future))
        texts = [c.text for c, _ in work if c.text.strip()]
        vectors = iter(await asyncio.to_thread(self._rt.embedder.embed, texts) if texts else ())
        for c, future in work:
            try:
                embedding = next(vectors) if c.text.strip() else None
                post = await self._process(c, embedding)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("sorter: %d/%d failed", c.chat_id, c.message_id)
                future.set_exception(exc)
            else:
                future.set_result(post)

    async def _process(self, c: Candidate, embedding: np.ndarray | None) -> Post | None:
        rt = self._rt
        now = rt.clock.now()
        settings = rt.settings
        day = local_date(c.posted_at, settings.general.timezone)
        if embedding is None:
            async with rt.store.begin() as conn:
                post = await rt.store.insert_post(
                    self._new_post(c, None, None, now),
                    conn=conn,
                    status=PostStatus.ignored,
                    ignore_reason="no_text",
                    decided_at=now,
                )
                if post is not None:
                    await self._count(c, day, conn)
            log.info("ingested %d/%d: no text, ignored", c.chat_id, c.message_id)
            return post
        feats = features_of(c)
        chat = await rt.store.get_chat(c.chat_id)
        topics = await rt.store.list_topics(active=True)
        engine = self._engine(settings)
        decision = await engine.decide(c, embedding, chat, topics, now, features=feats)
        async with rt.store.begin() as conn:
            post = await rt.store.insert_post(
                self._new_post(c, feats, embedding, now),
                conn=conn,
                **decision_columns(decision, now),
            )
            if post is None:
                return None
            await self._count(c, day, conn)
        self._index_post(post, feats, embedding, decision.duplicate_of)
        # Readable in the journal (SPEC: "what was sorted where"): the chat's title and the
        # topic's key next to the ids, which stay so a line can be matched to its row.
        log.info(
            "sorted %s (%d/%d): %s%s",
            _quoted(chat.title if chat is not None else None),
            post.chat_id,
            post.message_id,
            post.status.value,
            _detail(decision, {t.id: t.key for t in topics}),
        )
        await self._route(post, decision, engine)
        return post

    async def _route(self, post: Post, decision: Decision, engine: DecisionEngine) -> None:
        """What happens after the commit: outbox, or corroboration of the root (§9.3)."""
        if decision.status == PostStatus.queued:
            await self._enqueue(post.id)
            return
        if decision.status != PostStatus.duplicate or decision.duplicate_of is None:
            return
        count = await self._rt.store.add_corroboration(decision.duplicate_of, post.chat_id)
        if count is not None:
            await self._corroborated(decision.duplicate_of, count, engine)

    async def _corroborated(self, root_id: int, count: int, engine: DecisionEngine) -> None:
        """Recompute the root's strength with its chat's *current* trust; promote or note +N."""
        store = self._rt.store
        root = await store.get_post(root_id)
        if root is None:
            return
        strength, would_realtime = engine.strength(root, await store.get_chat(root.chat_id), count)
        await store.set_post_fields(root.id, strength=strength, would_realtime=would_realtime)
        chat = await store.get_chat(root.chat_id)
        topic = await store.get_topic(root.topic_id) if root.topic_id is not None else None
        log.info(
            "corroborated %s (%d/%d)%s: %d other chats, strength %.2f",
            _quoted(chat.title if chat is not None else None),
            root.chat_id,
            root.message_id,
            f" in {topic.key}" if topic is not None else "",
            count,
            strength,
        )
        if root.status == PostStatus.published:
            publisher = self._publisher()
            if publisher is not None:
                await publisher.note_corroboration(root.id)
        elif would_realtime and root.via == "live" and root.status in PROMOTABLE:
            await self._enqueue(root.id)

    # --- tick --------------------------------------------------------------------------------

    async def tick(self) -> None:
        """Hold expiry, self-healing of the outbox, index trimming (§9.3)."""
        async with self._lock:
            await self._ensure_index()
            rt = self._rt
            now = rt.clock.now()
            settings = rt.settings
            engine = self._engine(settings)
            for post in await rt.store.posts_by_status(PostStatus.held, due_before=now):
                await self._expire(post, engine)
            cutoff = now - timedelta(minutes=settings.sorting.hold_minutes)
            for post_id, status, posted_at in await self._orphans(cutoff):
                if posted_at >= cutoff:
                    log.warning("sorter: post %d was due for the outbox but never queued", post_id)
                    await self._enqueue(post_id)
                elif status == PostStatus.queued.value:
                    # §9.4/§14.6: an old queued post goes to the digest, never out late
                    await rt.store.set_post_fields(post_id, status=PostStatus.digest)
                    log.info(
                        "sorter: post %d was queued but never enqueued and is stale: digest",
                        post_id,
                    )
            assert self._index is not None
            window = timedelta(days=settings.sorting.dedup_window_days)
            self._index.remove_older_than(now - window)

    async def _expire(self, post: Post, engine: DecisionEngine) -> None:
        """A hold ran out: strength with today's settings and trust decides queue or digest."""
        store = self._rt.store
        strength, would_realtime = engine.strength(
            post, await store.get_chat(post.chat_id), post.corroboration
        )
        if would_realtime:
            await store.set_post_fields(post.id, strength=strength, would_realtime=True)
            if await self._enqueue(post.id):
                return
        await store.set_post_fields(
            post.id, status=PostStatus.digest, strength=strength, would_realtime=would_realtime
        )
        chat = await store.get_chat(post.chat_id)
        topic = await store.get_topic(post.topic_id) if post.topic_id is not None else None
        log.info(
            "hold expired %s (%d/%d): digest%s",
            _quoted(chat.title if chat is not None else None),
            post.chat_id,
            post.message_id,
            f" of {topic.key}" if topic is not None else "",
        )

    async def _orphans(self, cutoff: datetime) -> list[tuple[int, str, datetime]]:
        """Live posts that should be in the outbox but have no ``publications`` row, as
        ``(id, status, posted_at)``: the fresh ones (posted at or after ``cutoff``) and the
        stale ``queued`` ones.

        A crash between the post's commit and ``publisher.enqueue`` leaves exactly this trace;
        a ``cancelled`` row is a row, so a post that left the real-time path is never revived.
        Only a fresh orphan is enqueued: one older than ``hold_minutes`` is what the stale
        rule of §9.4 sends to the digest, and ``publisher.reconcile`` (which applies that rule
        at a restart) only sees posts that have a row. A stale ``held``/``digest`` orphan is
        already on its way to the digest and is not selected at all.
        """
        posts, pubs = schema.posts, schema.publications
        stmt = (
            sa.select(posts.c.id, posts.c.status, posts.c.posted_at)
            .select_from(posts.outerjoin(pubs, pubs.c.post_id == posts.c.id))
            .where(posts.c.status.in_([s.value for s in SELF_HEAL]))
            .where(posts.c.would_realtime.is_(True))
            .where(posts.c.via == "live")
            .where(pubs.c.id.is_(None))
            .where(sa.or_(posts.c.posted_at >= cutoff, posts.c.status == PostStatus.queued.value))
            .order_by(posts.c.id)
        )
        rows = await self._rt.store.execute(stmt)
        if not isinstance(rows, list):
            return []
        return [(row.id, str(row.status), row.posted_at) for row in rows]

    # --- resort ------------------------------------------------------------------------------

    async def resort_unsorted(self, since: datetime) -> int:
        """Questions 2–4 again on ``unsorted`` posts since ``since`` (never corrected ones).

        Sorted like a backfill: digest inside the window, else dropped, tracked without a
        channel — nothing old is ever published. A re-sorted post is stored as
        ``via="backfill"`` too, so neither the self-healing of ``tick`` nor a later
        corroboration can put it on the real-time path (§9.2); ``would_realtime`` is still
        recorded for preview. Runs without a language model: a re-sort can cover a week of
        posts and must not cost a request per borderline one.
        """
        async with self._lock:
            rt = self._rt
            now = rt.clock.now()
            engine = DecisionEngine(rt.settings, rt.classifier, RecentIndex(1), None)
            topics = await rt.store.list_topics(active=True)
            posts = [
                p
                for p in await rt.store.recent_posts(
                    since, statuses=[PostStatus.unsorted], with_embeddings=True
                )
                if not p.corrected
            ]
            vectors = await self._vectors(posts)
            chats: dict[int, Chat | None] = {}
            changed = 0
            for post, embedding in zip(posts, vectors, strict=True):
                if post.chat_id not in chats:
                    chats[post.chat_id] = await rt.store.get_chat(post.chat_id)
                decision = await engine.classify(
                    candidate_from_post(post),
                    embedding,
                    chats[post.chat_id],
                    topics,
                    now,
                    realtime=False,
                )
                if decision.status == PostStatus.unsorted:
                    continue
                await rt.store.set_post_fields(
                    post.id, **decision_columns(decision, now), via="backfill"
                )
                changed += 1
                chat = chats[post.chat_id]
                log.info(
                    "re-sorted %s (%d/%d): %s%s",
                    _quoted(chat.title if chat is not None else None),
                    post.chat_id,
                    post.message_id,
                    decision.status.value,
                    _detail(decision, {t.id: t.key for t in topics}),
                )
            return changed

    async def _vectors(self, posts: list[Post]) -> list[np.ndarray]:
        """Stored embeddings, with the missing ones computed in one batch."""
        vectors = [embedding_from_bytes(p.embedding) for p in posts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            computed = await asyncio.to_thread(
                self._rt.embedder.embed, [posts[i].text for i in missing]
            )
            for i, vector in zip(missing, computed, strict=True):
                vectors[i] = vector
        return [v for v in vectors if v is not None]

    # --- internals ---------------------------------------------------------------------------

    def _engine(self, settings: Settings) -> DecisionEngine:
        assert self._index is not None
        return DecisionEngine(settings, self._rt.classifier, self._index, self._rt.llm)

    def _publisher(self) -> Publisher | None:
        publisher = self._rt.publisher
        if publisher is None:
            log.debug("sorter: no publisher wired, nothing queued")
        return publisher

    async def _enqueue(self, post_id: int) -> bool:
        publisher = self._publisher()
        return publisher is not None and await publisher.enqueue(post_id)

    async def _count(self, c: Candidate, day: date, conn: AsyncConnection) -> None:
        """§9.1 counting for a channel post the sorter just inserted (albums once).

        A group unit is not counted here: intake already counted each of its messages when
        ``add_group_message`` inserted them, and a unit spans several of those rows.
        """
        if c.kind != "post":
            return
        await self._rt.store.bump_chat_daily(c.chat_id, day, conn=conn)
        await self._rt.store.touch_chat(c.chat_id, c.posted_at, conn=conn)

    async def _ensure_index(self) -> None:
        """Rebuild the window from the database once; every submit then adds to it."""
        if self._index is not None:
            return
        rt = self._rt
        window = rt.settings.sorting.dedup_window_days
        index = RecentIndex(window)
        since = rt.clock.now() - timedelta(days=window)
        for post in await rt.store.recent_posts(since, with_embeddings=True):
            if post.status == PostStatus.ignored:
                continue
            index.add(
                post.id,
                post.chat_id,
                post.message_id,
                post.duplicate_of,
                post.text_hash,
                tuple(canonical_urls(post.urls)),  # every link, not only posts.url_key
                fwd_key_of(post.fwd_from_chat_id, post.fwd_from_message_id),
                embedding_from_bytes(post.embedding),
                posted_at=post.posted_at,
                lang=post.lang or detect_language(post.text),
                numbers=numbers_in(post.text),
            )
        self._index = index
        log.info("sorter: recent index holds %d posts of the last %d days", len(index), window)

    def _index_post(
        self, post: Post, feats: Features, embedding: np.ndarray, root_id: int | None
    ) -> None:
        assert self._index is not None
        self._index.add(
            post.id,
            post.chat_id,
            post.message_id,
            root_id,
            feats.text_hash,
            feats.url_keys,
            feats.fwd_key,
            embedding,
            posted_at=post.posted_at,
            lang=feats.lang,
            numbers=feats.numbers,
        )

    def _new_post(
        self, c: Candidate, feats: Features | None, embedding: np.ndarray | None, now: datetime
    ) -> NewPost:
        return NewPost(
            chat_id=c.chat_id,
            message_id=c.message_id,
            kind=c.kind,
            message_ids=list(c.message_ids),
            grouped_id=c.grouped_id,
            posted_at=c.posted_at,
            via=c.via,
            text=c.text,
            html=c.html,
            lang=None if feats is None else feats.lang,
            text_hash="" if feats is None else feats.text_hash,
            url_key=feats.url_keys[0] if feats is not None and feats.url_keys else None,
            urls=list(c.urls),
            media=c.media,
            noforwards=c.noforwards,
            fwd_from_chat_id=c.fwd_from_chat_id,
            fwd_from_message_id=c.fwd_from_message_id,
            embedding=None if embedding is None else embedding_to_bytes(embedding),
            views=c.views,
            forwards=c.forwards,
            views_at=now if c.views is not None else None,
        )


def decision_columns(decision: Decision, now: datetime) -> dict[str, object]:
    """The decision as ``posts`` columns, for ``insert_post(**...)`` and ``set_post_fields``."""
    return {
        "status": decision.status,
        "ignore_reason": decision.ignore_reason,
        "duplicate_of": decision.duplicate_of,
        "dup_kind": decision.dup_kind,
        "dup_score": decision.dup_score,
        "topic_id": decision.topic_id,
        "confidence": decision.confidence,
        "topic_scores": [
            {"topic_id": s.topic_id, "confidence": s.confidence} for s in decision.topic_scores
        ]
        or None,
        "strength": decision.strength,
        "would_realtime": decision.would_realtime,
        "hold_until": decision.hold_until,
        "decided_at": now,
    }


def _quoted(title: str | None) -> str:
    """A chat title for a log line: quoted, on one line, never empty."""
    text = " ".join((title or "?").split())
    return f'"{text}"'


def _detail(decision: Decision, topic_keys: dict[int, str] | None = None) -> str:
    if decision.status == PostStatus.duplicate:
        return f" of #{decision.duplicate_of} ({decision.dup_kind} {decision.dup_score:.2f})"
    if decision.topic_id is not None:
        key = (topic_keys or {}).get(decision.topic_id, f"topic {decision.topic_id}")
        return (
            f" -> {key} ({decision.confidence:.2f}), strength "
            f"{decision.strength:.2f}, immediate={'yes' if decision.would_realtime else 'no'}"
        )
    if decision.confidence is not None:
        return f" (best {decision.confidence:.2f})"
    return ""
