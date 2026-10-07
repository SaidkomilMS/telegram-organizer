"""discovery.py: proposals from synthetic clusters, naming, accept, merges, the auto rule."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeClock, FakeEmbedder, FakeLLM
from tg_curator.domain import (
    PROPOSAL_DONE,
    PROPOSAL_PROPOSED,
    Category,
    Cluster,
    Example,
    NewPost,
    Post,
    PostStatus,
    Topic,
    TopicPair,
)
from tg_curator.errors import TopicExists
from tg_curator.notify import Notifier
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED, Runtime
from tg_curator.subscriptions.discovery import Discovery, category_label
from tg_curator.subscriptions.review import ReviewService
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

EMBEDDER = FakeEmbedder()


class FakeCluster:
    """A stand-in for ``ml/cluster.py`` with the §10 signatures: every ``min_size`` rows in
    order form one cluster (``groups`` caps how many), and ``competition`` returns ``pairs``."""

    def __init__(self) -> None:
        self.groups = 1
        self.pairs: list[TopicPair] = []
        self.calls: list[tuple[str, Any]] = []
        self.langs: Sequence[str | None] | None = None
        self.existing: np.ndarray | None = None

    def find_clusters(
        self,
        embeddings: np.ndarray,
        min_size: int,
        tightness: float,
        *,
        langs: Sequence[str | None] | None = None,
        existing: np.ndarray | None = None,
    ) -> list[Cluster]:
        self.calls.append(("find_clusters", (embeddings.shape, min_size, tightness)))
        self.langs, self.existing = langs, existing
        out = []
        for g in range(self.groups):
            ids = list(range(g * min_size, (g + 1) * min_size))
            if ids[-1] >= len(embeddings):
                break
            out.append(
                Cluster(member_ids=ids, centroid=embeddings[ids].mean(axis=0), tightness=0.9)
            )
        return out

    def competition(self, posts: Sequence[Post], margin: float) -> list[TopicPair]:
        self.calls.append(("competition", (len(posts), margin)))
        return list(self.pairs)


class FakeTopics:
    """Only what discovery calls on the topics service: ``create`` and ``merge``."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self.calls: list[tuple[str, Any]] = []
        self.fail: Exception | None = None
        self.channel_id: int | None = None  # the channel create() gets; None = paced
        self.wait: int | None = None  # what channel_wait_minutes() reports

    def channel_wait_minutes(self) -> int | None:
        return self.wait

    async def create(self, name: str, **kw: Any) -> Topic:
        self.calls.append(("create", {"name": name, **kw}))
        if self.fail is not None:
            raise self.fail
        key = name.lower().replace(" ", "-")[:32]
        return await self._rt.store.upsert_topic(
            Topic(
                id=0,
                key=key,
                name=name,
                category=kw.get("category"),
                description=kw.get("description"),
                origin=kw.get("origin", "user"),
                channel_id=self.channel_id,
                created_at=self._rt.clock.now(),
            )  # fmt: skip
        )

    async def merge(self, src_key: str, dst_key: str) -> Topic:
        self.calls.append(("merge", (src_key, dst_key)))
        src = await self._rt.store.get_topic_by_key(src_key)
        dst = await self._rt.store.get_topic_by_key(dst_key)
        assert src is not None and dst is not None
        await self._rt.store.set_topic_fields(src.id, active=False)
        return dst


class FakeSorter:
    """Moves (up to) ``moves`` unsorted posts since ``since`` to ``tracked``, like a re-sort
    that found a topic for them, and returns how many it moved."""

    def __init__(self, rt: Runtime, moves: int = 7) -> None:
        self._rt = rt
        self.moves = moves
        self.calls: list[datetime] = []

    async def resort_unsorted(self, since: datetime) -> int:
        self.calls.append(since)
        store = self._rt.store
        posts = await store.recent_posts(since, statuses=[PostStatus.unsorted])
        moved = [p for p in posts if not p.corrected][: self.moves]
        for post in moved:
            await store.set_post_fields(post.id, status=PostStatus.tracked)
        return len(moved)


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    fake = FakeCluster()
    module = ModuleType("tg_curator.ml.cluster")
    module.find_clusters = fake.find_clusters  # type: ignore[attr-defined]
    module.competition = fake.competition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tg_curator.ml.cluster", module)
    categories = ModuleType("tg_curator.ml.categories")
    categories.all = lambda: [  # type: ignore[attr-defined]
        Category("crypto", "Crypto & exchanges"), Category("tech", "Technology")
    ]  # fmt: skip
    monkeypatch.setitem(sys.modules, "tg_curator.ml.categories", categories)
    return fake


@pytest.fixture
def svc(rt: Runtime, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch) -> Discovery:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.topics = FakeTopics(rt)  # type: ignore[assignment]
    rt.sorter = FakeSorter(rt)  # type: ignore[assignment]
    rt.discovery = Discovery(rt)
    rt.review = ReviewService(rt)
    rt.classifier.categories = [("crypto", 0.8), ("tech", 0.3)]  # type: ignore[attr-defined]
    return rt.discovery


async def unsorted_posts(
    rt: Runtime, info: ChatInfo, n: int, *, status: PostStatus = PostStatus.unsorted,
    age: timedelta = timedelta(hours=1), start: int = 1, topic_id: int | None = None,
) -> list[Post]:  # fmt: skip
    if await rt.store.get_chat(info.id) is None:
        await rt.store.upsert_chat(info)
    out = []
    for i in range(start, start + n):
        text = f"crypto exchange story number {i} about bitcoin and tether"
        new = NewPost(
            chat_id=info.id, message_id=i, kind="post", message_ids=[i],
            posted_at=rt.clock.now() - age, via="live", text=text, text_hash=text_hash(text),
            urls=[], embedding=EMBEDDER.embed([text])[0].tobytes(),
        )  # fmt: skip
        post = await rt.store.insert_post(new, status=status, topic_id=topic_id)
        assert post is not None
        out.append(post)
    return out


# --- proposing new topics --------------------------------------------------------------------


async def test_a_tight_cluster_becomes_a_new_topic_proposal(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo]
) -> None:
    posts = await unsorted_posts(rt, make_chat(username="cryptonews"), 30)
    [p] = await svc.propose()
    assert p.kind == "new_topic" and p.state == PROPOSAL_PROPOSED and p.chat_id is None
    assert p.review_day == START.date()
    assert p.reason == "30 unsorted posts in the last 7 days look like one topic"
    assert sorted(p.payload["member_post_ids"]) == [q.id for q in posts]
    examples = p.payload["example_post_ids"]
    assert len(examples) == 5 and set(examples) <= set(p.payload["member_post_ids"])
    assert p.payload["name"] == "Canned topic"  # the fake LLM is connected by default
    assert p.payload["description"] == "A description written by the fake."
    assert p.payload["category"] == "crypto"
    [(_, (shape, min_size, tightness))] = cluster.calls[:1]
    assert shape == (30, 64) and min_size == 30 and tightness == 0.60
    [call] = rt.llm.calls  # type: ignore[attr-defined]
    assert call[0] == "name_topic" and len(call[1]["examples"]) == 5


async def test_the_model_is_told_the_existing_topics_and_the_category_hint(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo]
) -> None:
    """The name must differ from the active topics, and the closest category is a hint."""
    await rt.store.upsert_topic(Topic(id=0, key="ml", name="ML", created_at=START))
    await rt.store.upsert_topic(Topic(id=0, key="old", name="Old", active=False, created_at=START))
    await unsorted_posts(rt, make_chat(username="cryptonews"), 30)
    await svc.propose()
    [call] = [c for c in rt.llm.calls if c[0] == "name_topic"]  # type: ignore[attr-defined]
    assert call[1]["existing"] == ["ML"]
    assert call[1]["category_label"] == category_label("crypto")


async def test_existing_topic_prototypes_and_languages_reach_the_clustering(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo]
) -> None:
    """§10 drops a cluster that is "more of an existing topic": ``find_clusters`` is handed one
    unit-norm prototype per topic that has examples (the mean of its example vectors; a topic
    without examples and the "not for me" negatives contribute nothing) and each post's
    language for the duplicate collapse."""
    topic = await rt.store.upsert_topic(Topic(id=0, key="ml", name="ML", created_at=START))
    await rt.store.upsert_topic(Topic(id=0, key="ai", name="AI", created_at=START))
    vectors = np.asarray(EMBEDDER.embed(["machine learning paper", "neural network training"]))
    for i, vector in enumerate(vectors):
        await rt.store.add_example(
            Example(
                id=0,
                topic_id=topic.id,
                kind="post",
                text=f"example {i}",
                embedding=vector.astype("<f4").tobytes(),
                created_at=START,
            )  # fmt: skip
        )
    await rt.store.add_example(
        Example(
            id=0,
            topic_id=None,
            kind="correction",
            text="not for me",
            embedding=vectors[0].astype("<f4").tobytes(),
            created_at=START,
        )  # fmt: skip
    )
    await unsorted_posts(rt, make_chat(), 30)
    await svc.propose()
    assert cluster.existing is not None and cluster.existing.shape == (1, 64)
    expected = vectors.mean(axis=0)
    assert np.allclose(cluster.existing[0], expected / np.linalg.norm(expected), atol=1e-6)
    assert cluster.langs is not None and len(cluster.langs) == 30


async def test_without_a_language_model_the_name_is_the_closest_category(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo]
) -> None:
    rt.llm = FakeLLM(enabled=False)
    await unsorted_posts(rt, make_chat(), 30)
    [p] = await svc.propose()
    assert p.payload["name"] == "Crypto & exchanges" and p.payload["description"] is None


async def test_rejected_old_and_too_few_posts_never_cluster(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo]
) -> None:
    chat = make_chat()
    await unsorted_posts(rt, chat, 20)
    await unsorted_posts(rt, chat, 20, status=PostStatus.rejected, start=100)
    await unsorted_posts(rt, chat, 20, age=timedelta(days=8), start=200)
    assert await svc.propose() == []
    assert cluster.calls == []  # fewer than cluster_min_posts: the models are not even asked
    await unsorted_posts(rt, chat, 10, start=300)
    [p] = await svc.propose()
    assert len(p.payload["member_post_ids"]) == 30


async def test_posts_of_an_open_proposal_are_not_clustered_again(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo]
) -> None:
    await unsorted_posts(rt, make_chat(), 30)
    assert len(await svc.propose()) == 1
    assert await svc.propose() == []


async def test_the_review_sends_a_new_topic_with_five_example_links(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await unsorted_posts(rt, make_chat(username="cryptonews"), 30)
    [p] = await rt.review.build()  # type: ignore[union-attr]
    assert await rt.review.send() == 1  # type: ignore[union-attr]
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.startswith("✨ New topic: Canned topic")
    assert len(msg.hrefs) == 5 and all(h.startswith("https://t.me/cryptonews/") for h in msg.hrefs)
    assert [[b.data for b in row] for row in msg.buttons or []] == [
        [f"ds:create:{p.id}", f"ds:rename:{p.id}"], [f"ds:dismiss:{p.id}"],
    ]  # fmt: skip


async def test_the_proposal_shows_the_models_short_description(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await unsorted_posts(rt, make_chat(username="cryptonews"), 30)
    await rt.review.build()  # type: ignore[union-attr]
    await rt.review.send()  # type: ignore[union-attr]
    [msg] = bot_gw.sent(OWNER_ID)
    lines = msg.text.splitlines()
    assert "A description written by the fake." in lines
    # ... above the five examples
    assert lines.index("A description written by the fake.") < lines.index(
        next(line for line in lines if line.startswith("1. "))
    )
    assert "<i>A description written by the fake.</i>" in msg.html


async def test_without_a_language_model_no_description_line_appears(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    rt.llm = FakeLLM(enabled=False)
    await unsorted_posts(rt, make_chat(username="cryptonews"), 30)
    await rt.review.build()  # type: ignore[union-attr]
    await rt.review.send()  # type: ignore[union-attr]
    [msg] = bot_gw.sent(OWNER_ID)
    assert "<i>" not in msg.html
    assert msg.text.split("\n\n")[1].startswith("1. ")


# --- accepting -------------------------------------------------------------------------------


async def test_accept_creates_the_topic_stores_cluster_examples_retrains_and_resorts(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    events: list[dict[str, Any]] = []

    async def on_examples(**payload: Any) -> None:
        events.append(payload)

    rt.events.on(EVENT_EXAMPLES_CHANGED, on_examples)
    await unsorted_posts(rt, make_chat(), 30)
    [p] = await rt.review.build()  # type: ignore[union-attr]
    await rt.review.send()  # type: ignore[union-attr]
    reloads = rt.classifier.reloads  # type: ignore[attr-defined]
    topic = await svc.accept(p.id)
    assert topic.name == "Canned topic" and topic.origin == "discovered"
    [(_, create)] = rt.topics.calls  # type: ignore[attr-defined]
    assert create == {
        "name": "Canned topic", "category": "crypto", "create_channel": True,
        "description": "A description written by the fake.", "origin": "discovered",
    }  # fmt: skip
    examples = await rt.store.list_examples(topic.id)
    assert len(examples) == 30 and {e.kind for e in examples} == {"cluster"}
    assert rt.classifier.reloads == reloads + 1  # type: ignore[attr-defined]
    assert rt.sorter.calls == [START - timedelta(days=7)]  # type: ignore[attr-defined]
    done = await rt.store.get_proposal(p.id)
    assert done is not None and done.state == PROPOSAL_DONE and done.result == topic.key
    assert done.payload["topic_id"] == topic.id and done.payload["absorbed"] == 7
    assert events == [{"reason": "topics"}]
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.endswith("Created the topic Canned topic ✓") and msg.buttons is None
    assert bot_gw.sent(OWNER_ID)[0].message_id == done.bot_message_id


async def test_absorbed_counts_what_the_topics_own_resort_moved_too(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo]
) -> None:
    """``TopicsService.create`` re-sorts the window itself (the new topic's category prior
    alone can take posts) before discovery stores the cluster examples and re-sorts again:
    the absorbed count covers both passes, not only the last one."""
    await unsorted_posts(rt, make_chat(), 40)
    [p] = await rt.review.build()  # type: ignore[union-attr]
    topics = rt.topics
    assert isinstance(topics, FakeTopics) and isinstance(rt.sorter, FakeSorter)
    first_pass = FakeSorter(rt, moves=20)
    plain_create = topics.create

    async def create_and_resort(name: str, **kw: Any) -> Topic:
        topic = await plain_create(name, **kw)
        await first_pass.resort_unsorted(rt.clock.now() - timedelta(days=7))
        return topic

    topics.create = create_and_resort  # type: ignore[method-assign]
    await svc.accept(p.id)
    done = await rt.store.get_proposal(p.id)
    assert done is not None and done.payload["absorbed"] == 27  # 20 + 7, not 7


async def test_accept_with_a_name_renames(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo]
) -> None:
    await unsorted_posts(rt, make_chat(), 30)
    [p] = await svc.propose()
    topic = await svc.accept(p.id, name="Stablecoins")
    assert topic.name == "Stablecoins"
    assert rt.topics.calls[0][1]["name"] == "Stablecoins"  # type: ignore[attr-defined]


async def test_accept_refuses_a_decided_proposal(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo]
) -> None:
    await unsorted_posts(rt, make_chat(), 30)
    [p] = await svc.propose()
    await rt.review.decide(p.id, "skip")  # type: ignore[union-attr]
    with pytest.raises(Exception, match="not available"):
        await svc.accept(p.id)


# --- merges ----------------------------------------------------------------------------------


async def test_competing_topics_become_a_merge_proposal_once(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo],
    bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    a = await rt.store.upsert_topic(Topic(id=0, key="ml", name="ML", created_at=START))
    b = await rt.store.upsert_topic(Topic(id=0, key="ai", name="AI", created_at=START))
    await unsorted_posts(rt, make_chat(), 3, status=PostStatus.digested, topic_id=a.id)
    cluster.pairs = [TopicPair(a.id, b.id, 12)]
    [p] = await svc.propose()
    assert p.kind == "merge_topics" and p.payload == {"a_id": a.id, "b_id": b.id, "shared": 12}
    assert p.reason == "ML and AI competed for 12 of the same posts"
    assert await svc.propose() == []
    assert await rt.review.send() == 1  # type: ignore[union-attr]
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.startswith("🔀 Merge topics · ML + AI")
    merged = await svc.accept(p.id)
    assert merged.key == "ai" and rt.topics.calls == [("merge", ("ml", "ai"))]  # type: ignore[attr-defined]
    assert (await rt.store.get_proposal(p.id)).state == PROPOSAL_DONE  # type: ignore[union-attr]
    assert msg.text.endswith("Merged into AI ✓")


# --- automatic mode --------------------------------------------------------------------------


async def test_auto_mode_creates_at_most_one_topic_per_day_and_tells_the_owner(
    rt: Runtime, svc: Discovery, cluster: FakeCluster, make_chat: Callable[..., ChatInfo],
    bot_gw: FakeBotGateway, clock: FakeClock,
) -> None:  # fmt: skip
    await rt.settings_file.set_value("review.auto_create_topics", True)
    cluster.groups = 2
    await unsorted_posts(rt, make_chat(), 60)
    assert isinstance(rt.llm, FakeLLM)
    [p] = await svc.propose()  # two clusters, one topic: the other waits
    assert p.state == PROPOSAL_DONE and len(rt.topics.calls) == 1  # type: ignore[attr-defined]
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.startswith("Created the topic Canned topic")
    # The waiting cluster's examples never went to the language model.
    assert len([c for c in rt.llm.calls if c[0] == "name_topic"]) == 1
    assert await svc.propose() == []  # same day: nothing more is created or proposed
    assert len([c for c in rt.llm.calls if c[0] == "name_topic"]) == 1
    clock.advance(timedelta(days=1))
    rt.topics.fail = TopicExists("Canned topic")  # type: ignore[attr-defined]
    [second] = await svc.propose()
    assert second.state == PROPOSAL_PROPOSED  # the name exists: ask instead of failing
    assert len(bot_gw.sent(OWNER_ID)) == 1


async def test_auto_mode_says_when_the_channel_is_not_created_yet(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("review.auto_create_topics", True)
    rt.topics.wait = 42  # type: ignore[attr-defined]  # channel pacing: no channel yet
    await unsorted_posts(rt, make_chat(), 30)
    await svc.propose()
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.startswith("Created the topic Canned topic (automatic mode)")
    assert "and its private channel" not in msg.text
    assert "wait 42 min" in msg.text and "/topics" in msg.text


async def test_auto_mode_names_the_channel_when_it_was_created(
    rt: Runtime, svc: Discovery, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("review.auto_create_topics", True)
    rt.topics.channel_id = -100_777  # type: ignore[attr-defined]
    await unsorted_posts(rt, make_chat(), 30)
    await svc.propose()
    [msg] = bot_gw.sent(OWNER_ID)
    assert msg.text.startswith("Created the topic Canned topic and its private channel")
