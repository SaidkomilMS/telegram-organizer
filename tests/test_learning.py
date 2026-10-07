"""Learning.correct / retrain (DESIGN §9.8) against the fakes and the real Store."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from tests.fakes import FakeUserGateway
from tg_curator.domain import (
    PUB_RETRACTED,
    CorrectionResult,
    MoveResult,
    NewPost,
    PostStatus,
    Topic,
)
from tg_curator.errors import ConfigError, CuratorError
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.learning import Learning, embedding_array, embedding_bytes
from tg_curator.topics.service import TopicsService


class FakePublisher:
    """Records ``move`` calls; the real one settles the final status itself (§9.4)."""

    def __init__(self) -> None:
        self.moves: list[tuple[int, int | None]] = []

    async def enqueue(self, post_id: int) -> bool:
        raise NotImplementedError

    async def tick(self) -> None:
        raise NotImplementedError

    async def reconcile(self) -> None:
        raise NotImplementedError

    async def note_corroboration(self, post_id: int) -> None:
        raise NotImplementedError

    async def move(self, post_id: int, new_topic_id: int | None) -> MoveResult:
        self.moves.append((post_id, new_topic_id))
        return MoveResult(republished=False, stubbed=True, new_message_ids=[])


class Topics:
    ml: Topic  # has a channel
    fin: Topic  # has a channel
    foot: Topic  # no channel


@pytest.fixture
async def topics(
    rt: Runtime, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> Topics:
    """The template topics, two of them with channels, plus a wired fake publisher."""
    rt.publisher = FakePublisher()
    rt.learning = Learning(rt)
    svc = TopicsService(rt)
    ml_chan = user_gw.add_chat(make_chat(title="ML", username="ml", is_creator=True))
    fin_chan = user_gw.add_chat(make_chat(title="Fin", username="fin", is_creator=True))
    await rt.settings_file.upsert_topic("ml-ai", channel=ml_chan.id)
    await rt.settings_file.upsert_topic("fintech", channel=fin_chan.id)
    await svc.sync_from_settings()
    out = Topics()
    for attr, key in (("ml", "ml-ai"), ("fin", "fintech"), ("foot", "football")):
        topic = await rt.store.get_topic_by_key(key)
        assert topic is not None
        setattr(out, attr, topic)
    assert out.ml.channel_id and out.fin.channel_id and out.foot.channel_id is None
    return out


def reasons(seen: list[dict[str, Any]]) -> list[str]:
    return [p["reason"] for p in seen]


@pytest.fixture
def examples_changed(rt: Runtime) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def handler(**payload: Any) -> None:
        seen.append(payload)

    rt.events.on("examples_changed", handler)
    return seen


async def add_post(
    rt: Runtime,
    message_id: int,
    status: PostStatus,
    topic_id: int | None,
    *,
    with_embedding: bool = True,
    text: str | None = None,
) -> int:
    text = text or f"post {message_id} about something or other"
    embedding = embedding_bytes(rt.embedder.embed([text])[0]) if with_embedding else None
    new = NewPost(
        chat_id=-1_001_000_000_555,
        message_id=message_id,
        kind="post",
        message_ids=[message_id],
        posted_at=rt.clock.now() - timedelta(minutes=10),
        via="live",
        text=text,
        text_hash=f"h{message_id}",
        urls=[],
        embedding=embedding,
    )
    row = await rt.store.insert_post(new, status=status, topic_id=topic_id, confidence=0.6)
    assert row is not None
    return row.id


async def test_published_post_is_moved(
    rt: Runtime, topics: Topics, examples_changed: list[dict[str, Any]]
) -> None:
    post_id = await add_post(rt, 1, PostStatus.published, topics.ml.id)
    pub = await rt.store.create_publication(post_id, topics.ml.id)
    assert pub is not None
    await rt.store.set_publication_fields(pub.id, state="sent", message_ids=[7])

    result = await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]

    assert result == CorrectionResult(post_id, topics.ml.id, topics.fin.id, moved=True)
    assert rt.publisher.moves == [(post_id, topics.fin.id)]  # type: ignore[union-attr]
    post = await rt.store.get_post(post_id)
    assert post is not None
    assert (post.topic_id, post.corrected, post.confidence) == (topics.fin.id, True, 1.0)
    assert post.status == PostStatus.published, "the status is the publisher's to settle"
    rows = await rt.store.list_examples(topic_id=topics.fin.id)
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.post_id, row.wrong_topic_id, row.weight) == (
        "correction",
        post_id,
        topics.ml.id,
        1.0,
    )
    assert row.text == post.text and row.embedding == post.embedding
    assert [e.id for e in rt.classifier.learned] == [row.id]
    assert reasons(examples_changed) == ["correction"]


async def test_held_post_only_changes_label(rt: Runtime, topics: Topics) -> None:
    post_id = await add_post(rt, 2, PostStatus.held, topics.ml.id)

    result = await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]

    assert result.moved is False and rt.publisher.moves == []  # type: ignore[union-attr]
    post = await rt.store.get_post(post_id)
    assert post is not None and (post.status, post.topic_id) == (PostStatus.held, topics.fin.id)


async def test_held_post_into_a_channel_less_topic_is_tracked(rt: Runtime, topics: Topics) -> None:
    held = await add_post(rt, 3, PostStatus.held, topics.ml.id)
    digest = await add_post(rt, 4, PostStatus.digest, topics.ml.id)
    for pid in (held, digest):
        await rt.learning.correct(pid, topics.foot.id)  # type: ignore[union-attr]
        post = await rt.store.get_post(pid)
        assert post is not None
        assert (post.status, post.topic_id) == (PostStatus.tracked, topics.foot.id)


async def test_cancelled_publication_counts_as_no_row(rt: Runtime, topics: Topics) -> None:
    post_id = await add_post(rt, 5, PostStatus.digest, topics.ml.id)
    pub = await rt.store.create_publication(post_id, topics.ml.id)
    assert pub is not None
    await rt.store.set_publication_fields(pub.id, state="cancelled")

    result = await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]

    assert result.moved is False and rt.publisher.moves == []  # type: ignore[union-attr]
    assert (await rt.store.get_post(post_id)).status == PostStatus.digest  # type: ignore[union-attr]


async def test_unsorted_post_becomes_tracked_and_is_embedded_now(
    rt: Runtime, topics: Topics
) -> None:
    post_id = await add_post(rt, 6, PostStatus.unsorted, None, with_embedding=False)

    result = await rt.learning.correct(post_id, topics.ml.id)  # type: ignore[union-attr]

    assert result == CorrectionResult(post_id, None, topics.ml.id, moved=False)
    post = await rt.store.get_post(post_id)
    assert post is not None and (post.status, post.topic_id) == (PostStatus.tracked, topics.ml.id)
    assert post.embedding is not None
    expected = rt.embedder.embed([post.text])[0]
    assert np.allclose(embedding_array(post.embedding), expected)
    row = (await rt.store.list_examples(topic_id=topics.ml.id))[0]
    assert row.wrong_topic_id is None and row.embedding == post.embedding


async def test_label_only_statuses_keep_their_status(rt: Runtime, topics: Topics) -> None:
    for mid, status in ((7, PostStatus.digested), (8, PostStatus.dropped), (9, PostStatus.tracked)):
        pid = await add_post(rt, mid, status, topics.ml.id)
        await rt.learning.correct(pid, topics.fin.id)  # type: ignore[union-attr]
        post = await rt.store.get_post(pid)
        assert post is not None and (post.status, post.topic_id) == (status, topics.fin.id)
    assert rt.publisher.moves == []  # type: ignore[union-attr]


async def test_not_for_me_rejects_and_retracts(
    rt: Runtime, topics: Topics, examples_changed: list[dict[str, Any]]
) -> None:
    published = await add_post(rt, 10, PostStatus.published, topics.ml.id)
    pub = await rt.store.create_publication(published, topics.ml.id)
    assert pub is not None
    await rt.store.set_publication_fields(pub.id, state="sent")
    held = await add_post(rt, 11, PostStatus.held, topics.ml.id)

    moved = await rt.learning.correct(published, None)  # type: ignore[union-attr]
    kept = await rt.learning.correct(held, None)  # type: ignore[union-attr]

    assert (moved.moved, kept.moved) == (True, False)
    assert rt.publisher.moves == [(published, None)]  # type: ignore[union-attr]
    for pid in (published, held):
        post = await rt.store.get_post(pid)
        assert post is not None and (post.status, post.topic_id) == (PostStatus.rejected, None)
    negatives = await rt.store.list_examples(topic_id=None)
    assert sorted(e.post_id for e in negatives) == [published, held]  # type: ignore[type-var]
    assert all(e.wrong_topic_id == topics.ml.id for e in negatives)
    assert reasons(examples_changed) == ["correction", "correction"]


async def test_retracted_post_corrected_back_is_republished_via_move(
    rt: Runtime, topics: Topics
) -> None:
    post_id = await add_post(rt, 12, PostStatus.rejected, None)
    pub = await rt.store.create_publication(post_id, topics.ml.id)
    assert pub is not None
    await rt.store.set_publication_fields(pub.id, state=PUB_RETRACTED)

    result = await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]

    assert result.moved is True and rt.publisher.moves == [(post_id, topics.fin.id)]  # type: ignore[union-attr]
    post = await rt.store.get_post(post_id)
    assert post is not None and post.topic_id == topics.fin.id
    assert post.status == PostStatus.rejected, "move() settles it; Learning does not guess"


async def test_second_correction_replaces_the_first_row(rt: Runtime, topics: Topics) -> None:
    post_id = await add_post(rt, 13, PostStatus.tracked, topics.ml.id)

    await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]
    first = await rt.store.list_examples()
    await rt.learning.correct(post_id, topics.foot.id)  # type: ignore[union-attr]
    second = await rt.store.list_examples()

    assert len(first) == 1 and len(second) == 1
    assert second[0].id == first[0].id
    assert (second[0].topic_id, second[0].wrong_topic_id) == (topics.foot.id, topics.fin.id)
    assert len(rt.classifier.learned) == 2
    post = await rt.store.get_post(post_id)
    assert post is not None and post.topic_id == topics.foot.id


async def test_ignored_and_duplicate_posts_are_refused(
    rt: Runtime, topics: Topics, examples_changed: list[dict[str, Any]]
) -> None:
    ignored = await add_post(rt, 14, PostStatus.ignored, None)
    root = await add_post(rt, 15, PostStatus.digest, topics.ml.id)
    duplicate = await add_post(rt, 16, PostStatus.duplicate, None)
    await rt.store.set_post_fields(duplicate, duplicate_of=root)

    for pid in (ignored, duplicate):
        result = await rt.learning.correct(pid, topics.fin.id)  # type: ignore[union-attr]
        assert result.moved is False
        post = await rt.store.get_post(pid)
        assert post is not None and post.topic_id is None and post.corrected is False
    assert await rt.store.list_examples() == []
    assert rt.classifier.learned == [] and examples_changed == []


async def test_bad_targets(rt: Runtime, topics: Topics) -> None:
    post_id = await add_post(rt, 17, PostStatus.held, topics.ml.id)
    with pytest.raises(ConfigError):
        await rt.learning.correct(post_id, 99_999)  # type: ignore[union-attr]
    with pytest.raises(CuratorError):
        await rt.learning.correct(99_999, topics.ml.id)  # type: ignore[union-attr]
    assert await rt.store.list_examples() == []


async def test_retrain_rebuilds_from_the_database(rt: Runtime, topics: Topics) -> None:
    post_id = await add_post(rt, 18, PostStatus.tracked, topics.ml.id)
    await rt.learning.correct(post_id, topics.fin.id)  # type: ignore[union-attr]
    await rt.store.set_topic_fields(topics.foot.id, active=False)
    before = rt.classifier.reloads

    await rt.learning.retrain()  # type: ignore[union-attr]

    assert rt.classifier.reloads == before + 1
    assert [t.key for t in rt.classifier.topics] == ["ml-ai", "fintech", "uzbekistan"]
    assert [e.post_id for e in rt.classifier.examples] == [post_id]
