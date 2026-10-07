"""Corrections and retraining of the per-user layer (DESIGN §9.8).

A correction is the one thing the owner does to teach the curator, so it has to be cheap,
immediate and idempotent: one ``examples`` row per post (a second tap replaces the first),
``classifier.learn`` at once, the post's label and status settled in the same transaction, and
only then the publisher asked to move what was already posted. Retraining rebuilds the user
layer from the database, which is also how a replaced correction row is un-learned.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from tg_curator.domain import (
    PUB_CANCELLED,
    CorrectionResult,
    Example,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.errors import ConfigError, CuratorError
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED, Runtime

log = logging.getLogger(__name__)

# Statuses a correction cannot apply to: the post is not a story of its own (§9.8 step 1).
_REFUSED = (PostStatus.ignored, PostStatus.duplicate)
# Statuses whose post is in flight to a channel and keeps its status while the new topic has a
# channel (§9.8 step 4, second bullet).
_IN_FLIGHT = (PostStatus.held, PostStatus.digest, PostStatus.queued)


def embedding_bytes(vector: np.ndarray) -> bytes:
    """Little-endian float32 bytes, the storage form of every embedding (§6)."""
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


def embedding_array(raw: bytes) -> np.ndarray:
    """The inverse of :func:`embedding_bytes`."""
    return np.frombuffer(raw, dtype="<f4")


async def retrain_classifier(rt: Runtime) -> None:
    """Rebuild the user layer from the active topics and every stored example.

    Shared by ``Learning.retrain`` and by every ``TopicsService`` path that changes the topic
    set, so the classifier never sees a topic that is no longer active.
    """
    topics = await rt.store.list_topics(active=True)
    examples = await rt.store.list_examples()
    rt.classifier.reload(topics, examples)
    log.info("classifier reloaded: %d topics, %d examples", len(topics), len(examples))


class Learning:
    """The ``Learning`` service of §8: ``correct`` and ``retrain``."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    async def correct(self, post_id: int, new_topic_id: int | None) -> CorrectionResult:
        """Move a post to another topic (``None`` = "not for me") and learn from it (§9.8)."""
        store = self._rt.store
        post = await store.get_post(post_id)
        if post is None:
            raise CuratorError(f"post {post_id} does not exist")
        old_topic_id = post.topic_id
        if post.status in _REFUSED:
            log.info("correction of post %d refused: it is %s", post_id, post.status.value)
            return CorrectionResult(post_id, old_topic_id, new_topic_id, moved=False)
        new_topic = await self._target_topic(new_topic_id)

        vector = await self._embedding_of(post)
        example = Example(
            id=0,
            topic_id=new_topic_id,
            kind="correction",
            post_id=post.id,
            wrong_topic_id=old_topic_id,
            text=post.text,
            embedding=embedding_bytes(vector),
            created_at=self._rt.clock.now(),
        )
        async with store.begin() as conn:
            stored = await store.add_example(example, conn=conn)
            publication = await store.get_publication(post.id, conn=conn)
            has_row = publication is not None and publication.state != PUB_CANCELLED
            fields = {"topic_id": new_topic_id, "corrected": True, "confidence": 1.0}
            if post.embedding is None:
                fields["embedding"] = example.embedding
            status = _settled_status(post.status, new_topic, has_row)
            if status is not None:
                fields["status"] = status
            await store.set_post_fields(post.id, conn=conn, **fields)
        self._rt.classifier.learn(stored)

        moved = False
        if has_row:
            moved = await self._move(post.id, new_topic_id)
        log.info(
            "post %d corrected: topic %s -> %s (status %s, moved=%s)",
            post.id,
            old_topic_id,
            new_topic_id,
            status.value if status is not None else "kept for the publisher",
            moved,
        )
        await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="correction")
        return CorrectionResult(post.id, old_topic_id, new_topic_id, moved)

    async def retrain(self) -> None:
        """Rebuild the user layer from the DB (also un-learns replaced correction rows)."""
        await retrain_classifier(self._rt)

    # --- internals ---------------------------------------------------------------------------

    async def _target_topic(self, topic_id: int | None) -> Topic | None:
        if topic_id is None:
            return None
        topic = await self._rt.store.get_topic(topic_id)
        if topic is None or not topic.active:
            raise ConfigError(f"topic {topic_id} does not exist any more; pick another one")
        return topic

    async def _embedding_of(self, post: Post) -> np.ndarray:
        """The stored embedding, or a fresh one for a post that was never embedded."""
        if post.embedding is not None:
            return embedding_array(post.embedding)
        matrix = await asyncio.to_thread(self._rt.embedder.embed, [post.text])
        return matrix[0]

    async def _move(self, post_id: int, new_topic_id: int | None) -> bool:
        """Hand the post to ``publisher.move``; ``True`` when the publisher was asked.

        ``CorrectionResult.moved`` means "the publication was touched": a pending row that was
        only re-pointed counts too, since the post will now go out elsewhere.
        """
        publisher = self._rt.publisher
        if publisher is None:
            log.warning("post %d has a publication but no publisher is wired; not moved", post_id)
            return False
        result = await publisher.move(post_id, new_topic_id)
        log.debug("publisher.move(%d): %s", post_id, result)
        return True


def _settled_status(
    status: PostStatus, new_topic: Topic | None, has_row: bool
) -> PostStatus | None:
    """The status rule of §9.8 step 4; ``None`` leaves the status to ``publisher.move``."""
    if new_topic is None:
        return PostStatus.rejected
    if has_row:
        return None
    if status in _IN_FLIGHT:
        return None if new_topic.channel_id is not None else PostStatus.tracked
    if status in (PostStatus.unsorted, PostStatus.rejected):
        return PostStatus.tracked
    return None
