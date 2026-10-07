"""The live sorter (DESIGN §9.3): persistence, routing, corroboration, holds, self-healing."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
import sqlalchemy as sa

from tests.fakes import START, FakeClassifier, FakeClock, FakeEmbedder
from tg_curator.clock import local_date
from tg_curator.db import schema
from tg_curator.db.store import Store
from tg_curator.domain import (
    PUB_CANCELLED,
    PUB_PENDING,
    Candidate,
    MoveResult,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.pipeline import engine as engine_mod
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

CHANNEL = -1_001_000_000_900
A = (
    "The central bank raised its key rate to fourteen percent today, citing stubborn inflation "
    "and a weak currency."
)
A_FOOTER = A + "\n\n🔥 https://t.me/kunuz"
A_REWORDED = (
    "Today the central bank raised its key rate to fourteen percent, citing stubborn inflation "
    "and a weak currency outlook."
)
A_LINKED = A + "\nhttps://example.com/news/rate"
A_URL = "https://example.com/news/rate"
U1 = (
    "Key rate up to fourteen percent: the central bank cites inflation and a weak currency. "
    "https://example.com/news/rate?utm_source=tg"
)
U1_URL = "https://example.com/news/rate?utm_source=tg"
B = "Barcelona beat Real Madrid in the derby after a late goal from the substitute striker."
C = "A new metro line opened in the capital this morning with twelve stations."


def fake_language(text: str) -> str:
    return "ru" if re.search(r"[Ѐ-ӿ]", text) else "en"


def fake_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


@pytest.fixture(autouse=True)
def language_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "detect_language", fake_language)
    monkeypatch.setattr(engine_mod, "numbers_in", fake_numbers)


class StubPublisher:
    """What the sorter sees of the outbox: ``enqueue`` writes the row like the real one."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.enqueued: list[int] = []
        self.noted: list[int] = []
        self.refuse = False

    async def enqueue(self, post_id: int) -> bool:
        self.enqueued.append(post_id)
        if self.refuse:
            return False
        post = await self.store.get_post(post_id)
        assert post is not None and post.topic_id is not None
        async with self.store.begin() as conn:
            row = await self.store.create_publication(post_id, post.topic_id, conn=conn)
            if row is None:
                return False
            await self.store.set_post_fields(post_id, status=PostStatus.queued, conn=conn)
        return True

    async def tick(self) -> None:
        return None

    async def reconcile(self) -> None:
        return None

    async def note_corroboration(self, post_id: int) -> None:
        self.noted.append(post_id)

    async def move(self, post_id: int, new_topic_id: int | None) -> MoveResult:
        return MoveResult(False, False, [])


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(len(texts))
        return super().embed(texts)


@pytest.fixture
async def publisher(rt: Runtime) -> StubPublisher:
    pub = StubPublisher(rt.store)
    rt.publisher = pub  # type: ignore[assignment]
    return pub


@pytest.fixture
async def topic(rt: Runtime) -> Topic:
    """One topic with a channel that the fake classifier is sure about."""
    t = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=rt.clock.now())
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {t.id: 0.9}
    return t


@pytest.fixture
async def chats(rt: Runtime, make_chat: Callable[..., ChatInfo]) -> list[int]:
    """Four source chats stored in the DB, neutral trust."""
    ids = []
    for _ in range(4):
        info = make_chat()
        await rt.store.upsert_chat(info)
        ids.append(info.id)
    return ids


@pytest.fixture
def sorter(rt: Runtime, publisher: StubPublisher) -> Sorter:
    return Sorter(rt)


def cand(
    chat_id: int,
    message_id: int,
    text: str = A,
    *,
    urls: tuple[str, ...] = (),
    fwd: tuple[int, int] | None = None,
    via: str = "live",
    posted_at: datetime | None = None,
    media: str | None = None,
) -> Candidate:
    return Candidate(
        chat_id=chat_id,
        message_id=message_id,
        message_ids=[message_id],
        kind="post",
        posted_at=posted_at or START,
        text=text,
        html=None,
        urls=list(urls),
        media=media,  # type: ignore[arg-type]
        grouped_id=None,
        views=None,
        forwards=None,
        fwd_from_chat_id=None if fwd is None else fwd[0],
        fwd_from_message_id=None if fwd is None else fwd[1],
        noforwards=False,
        via=via,  # type: ignore[arg-type]
    )


async def trust(rt: Runtime, chat_id: int, level: float | None) -> None:
    await rt.store.set_chat_fields(chat_id, trust=level)


async def daily(store: Store, chat_id: int) -> int:
    rows = await store.execute(
        sa.select(schema.chat_daily.c.messages).where(schema.chat_daily.c.chat_id == chat_id)
    )
    return sum(r.messages for r in rows) if isinstance(rows, list) else 0


async def fresh(store: Store, post: Post) -> Post:
    got = await store.get_post(post.id)
    assert got is not None
    return got


# --- submit basics ------------------------------------------------------------------------------


async def test_no_text_is_ignored_counted_and_never_indexed(
    rt: Runtime, sorter: Sorter, chats: list[int]
) -> None:
    post = await sorter.submit(cand(chats[0], 1, "", media="photo"))
    assert post is not None and post.status == PostStatus.ignored
    assert post.ignore_reason == "no_text" and post.embedding is None and post.lang is None
    assert await daily(rt.store, chats[0]) == 1
    chat = await rt.store.get_chat(chats[0])
    assert chat is not None and chat.last_message_at == START
    assert sorter.index is not None and post.id not in sorter.index


async def test_sorted_post_is_persisted_with_decision_lang_and_embedding(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    post = await sorter.submit(cand(chats[0], 1, "Банк повысил ставку до 14 процентов."))
    assert post is not None
    assert post.status == PostStatus.held and post.topic_id == topic.id
    assert post.confidence == 0.9 and post.lang == "ru" and post.text_hash
    assert post.topic_scores == [{"topic_id": topic.id, "confidence": 0.9}]
    assert post.strength == 1.0 and post.would_realtime is False
    assert post.hold_until == START + timedelta(minutes=45) and post.decided_at == START
    assert post.embedding is not None and len(post.embedding) == 64 * 4
    assert await daily(rt.store, chats[0]) == 1
    assert sorter.index is not None and post.id in sorter.index


async def test_duplicate_submission_is_a_no_op(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    await trust(rt, chats[0], 3.0)
    first = await sorter.submit(cand(chats[0], 1))
    assert first is not None and first.status == PostStatus.queued
    assert publisher.enqueued == [first.id]
    again = await sorter.submit(cand(chats[0], 1))
    assert again is None
    assert await daily(rt.store, chats[0]) == 1
    assert publisher.enqueued == [first.id]
    assert (await fresh(rt.store, first)).status == PostStatus.queued
    rows = await rt.store.execute(sa.select(sa.func.count()).select_from(schema.posts))
    assert isinstance(rows, list) and rows[0][0] == 1


async def test_a_burst_is_embedded_in_one_batch_and_decided_in_order(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    embedder = CountingEmbedder()
    rt.embedder = embedder
    posts = await asyncio.gather(
        sorter.submit(cand(chats[0], 1, A)),
        sorter.submit(cand(chats[1], 1, A_FOOTER)),
        sorter.submit(cand(chats[2], 1, "")),
        sorter.submit(cand(chats[3], 1, B)),
    )
    # the first one starts at once; the three that arrived meanwhile are one batch of two
    # texts (the empty one is never embedded)
    assert embedder.batches == [1, 2]
    assert [p.status for p in posts if p] == [
        PostStatus.held,
        PostStatus.duplicate,
        PostStatus.ignored,
        PostStatus.held,
    ]
    assert posts[1] is not None and posts[0] is not None
    assert posts[1].duplicate_of == posts[0].id


async def test_a_failing_candidate_does_not_block_the_others(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    bad = cand(chats[0], 1, A, posted_at=START.replace(tzinfo=None))  # naive: refused by the DB
    results = await asyncio.gather(
        sorter.submit(bad), sorter.submit(cand(chats[1], 1, B)), return_exceptions=True
    )
    assert isinstance(results[0], Exception)
    assert isinstance(results[1], Post) and results[1].status == PostStatus.held


# --- every dedup kind through the sorter ----------------------------------------------------------


async def test_exact_repeat_with_a_footer_link_and_emoji(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    dup = await sorter.submit(cand(chats[1], 1, A_FOOTER, urls=("https://t.me/kunuz",)))
    assert root is not None and dup is not None
    assert dup.status == PostStatus.duplicate and dup.dup_kind == "exact"
    assert dup.duplicate_of == root.id and dup.topic_id is None
    assert (await fresh(rt.store, root)).corroboration == 1


async def test_forward_of_a_held_message_and_two_forwards_of_one_original(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 5, A))
    assert root is not None
    forward = await sorter.submit(cand(chats[1], 1, B, fwd=(chats[0], 5)))
    assert forward is not None and forward.dup_kind == "forward"
    assert forward.duplicate_of == root.id
    first_fwd = await sorter.submit(cand(chats[2], 1, C, fwd=(-1_001_999, 42)))
    assert first_fwd is not None and first_fwd.status == PostStatus.held
    second_fwd = await sorter.submit(cand(chats[3], 1, "other text", fwd=(-1_001_999, 42)))
    assert second_fwd is not None and second_fwd.dup_kind == "forward"
    assert second_fwd.duplicate_of == first_fwd.id


async def test_url_repeat_needs_similar_text(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A_LINKED, urls=(A_URL,)))
    assert root is not None and root.url_key == "example.com/news/rate"
    dup = await sorter.submit(cand(chats[1], 1, U1, urls=(U1_URL,)))
    assert dup is not None and dup.dup_kind == "url" and dup.duplicate_of == root.id
    assert dup.dup_score is not None and 0.6 <= dup.dup_score < 0.9
    footer = await sorter.submit(cand(chats[2], 1, B + " " + A_URL, urls=(A_URL,)))
    assert footer is not None and footer.status == PostStatus.held


async def test_semantic_repeat_is_a_reworded_post(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    dup = await sorter.submit(cand(chats[1], 1, A_REWORDED))
    assert root is not None and dup is not None
    assert dup.dup_kind == "semantic" and dup.duplicate_of == root.id
    assert dup.dup_score is not None and dup.dup_score >= 0.9
    other = await sorter.submit(cand(chats[2], 1, B))
    assert other is not None and other.status == PostStatus.held


async def test_repeat_chain_points_at_the_root(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    mid = await sorter.submit(cand(chats[1], 1, A_REWORDED))
    last = await sorter.submit(cand(chats[2], 1, A_REWORDED + " 🔥"))
    assert root and mid and last
    assert mid.duplicate_of == root.id and last.duplicate_of == root.id
    assert last.dup_kind == "exact"


# --- corroboration and promotion ---------------------------------------------------------


async def test_corroboration_counts_one_per_other_chat(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    assert root is not None
    await sorter.submit(cand(chats[0], 2, A_FOOTER))  # own chat: no corroboration
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    await sorter.submit(cand(chats[1], 2, A_REWORDED))  # same chat twice: once
    got = await fresh(rt.store, root)
    assert got.corroboration == 1 and got.corroborating_chats == [chats[1]]
    assert got.strength == 2.0 and got.would_realtime is False
    await sorter.submit(cand(chats[2], 1, A_REWORDED))
    got = await fresh(rt.store, root)
    assert got.corroboration == 2 and got.strength == 3.0 and got.would_realtime is True


async def test_trust_levels_route_exactly_as_the_design_says(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    await trust(rt, chats[0], 0.0)
    await trust(rt, chats[1], 1.0)
    await trust(rt, chats[2], 2.0)
    await trust(rt, chats[3], 3.0)
    digest_only = await sorter.submit(cand(chats[0], 1, A))
    neutral = await sorter.submit(cand(chats[1], 1, B))
    mild = await sorter.submit(cand(chats[2], 1, C))
    trusted = await sorter.submit(cand(chats[3], 1, "Fourth distinct story about taxes."))
    assert digest_only and neutral and mild and trusted
    assert digest_only.status == PostStatus.digest
    assert neutral.status == PostStatus.held
    assert mild.status == PostStatus.held and mild.strength == 2.0
    assert trusted.status == PostStatus.queued
    assert publisher.enqueued == [trusted.id]
    row = await rt.store.get_publication(trusted.id)
    assert row is not None and row.state == PUB_PENDING
    # one other chat is enough for trust 2
    await sorter.submit(cand(chats[1], 2, C + " 🔥"))
    assert (await fresh(rt.store, mild)).status == PostStatus.queued
    assert publisher.enqueued == [trusted.id, mild.id]
    # the neutral one needs two
    await sorter.submit(cand(chats[2], 2, B + " 🔥"))
    assert (await fresh(rt.store, neutral)).status == PostStatus.held
    await sorter.submit(cand(chats[3], 2, B + " 🔥🔥"))
    assert (await fresh(rt.store, neutral)).status == PostStatus.queued
    # a digest-only source is never promoted however much it spreads
    for i, chat_id in enumerate(chats[1:], start=3):
        await sorter.submit(cand(chat_id, i, A_FOOTER))
    got = await fresh(rt.store, digest_only)
    assert got.corroboration == 3 and got.status == PostStatus.digest
    assert got.would_realtime is False
    assert digest_only.id not in publisher.enqueued


async def test_promotion_uses_the_root_chats_current_trust(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    assert root is not None and root.status == PostStatus.held
    await trust(rt, chats[0], 0.0)
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    await sorter.submit(cand(chats[2], 1, A_FOOTER))
    got = await fresh(rt.store, root)
    assert got.corroboration == 2 and got.status == PostStatus.held
    assert got.would_realtime is False and publisher.enqueued == []
    rt.clock.advance(timedelta(minutes=46))  # type: ignore[attr-defined]
    await sorter.tick()
    assert (await fresh(rt.store, root)).status == PostStatus.digest
    assert publisher.enqueued == []


async def test_backfilled_roots_gain_corroboration_but_are_never_promoted(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    root = await sorter.submit(
        cand(chats[0], 1, A, via="backfill", posted_at=START - timedelta(hours=2))
    )
    assert root is not None and root.status == PostStatus.digest
    old = await sorter.submit(
        cand(chats[0], 2, B, via="backfill", posted_at=START - timedelta(hours=30))
    )
    assert old is not None and old.status == PostStatus.dropped
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    await sorter.submit(cand(chats[2], 1, A_REWORDED))
    got = await fresh(rt.store, root)
    assert got.corroboration == 2 and got.strength == 3.0 and got.would_realtime is True
    assert got.status == PostStatus.digest and publisher.enqueued == []
    await sorter.tick()
    assert (await fresh(rt.store, root)).status == PostStatus.digest
    assert publisher.enqueued == []


async def test_live_duplicate_of_a_backfilled_root_is_still_a_duplicate(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A, via="backfill"))
    dup = await sorter.submit(cand(chats[1], 1, A_FOOTER))
    assert root and dup and dup.duplicate_of == root.id


async def test_hold_expiry_goes_to_digest_or_out(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    clock: FakeClock = rt.clock  # type: ignore[assignment]
    quiet = await sorter.submit(cand(chats[0], 1, A))
    spread = await sorter.submit(cand(chats[1], 1, B))
    assert quiet and spread
    clock.advance(timedelta(minutes=44))
    await sorter.tick()
    assert (await fresh(rt.store, quiet)).status == PostStatus.held
    # a trust raise during the hold counts at expiry
    await trust(rt, chats[1], 3.0)
    clock.advance(timedelta(minutes=2))
    await sorter.tick()
    assert (await fresh(rt.store, quiet)).status == PostStatus.digest
    got = await fresh(rt.store, spread)
    assert got.status == PostStatus.queued and got.would_realtime is True and got.strength == 3.0
    assert publisher.enqueued == [spread.id]


async def test_hold_expiry_refused_by_the_outbox_falls_into_the_digest(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    post = await sorter.submit(cand(chats[0], 1, A))
    assert post is not None
    await trust(rt, chats[0], 3.0)
    publisher.refuse = True
    rt.clock.advance(timedelta(minutes=46))  # type: ignore[attr-defined]
    await sorter.tick()
    assert (await fresh(rt.store, post)).status == PostStatus.digest


async def test_promotion_while_digest_but_not_after_digested(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    clock: FakeClock = rt.clock  # type: ignore[assignment]
    root = await sorter.submit(cand(chats[0], 1, A))
    assert root is not None
    clock.advance(timedelta(minutes=46))
    await sorter.tick()
    assert (await fresh(rt.store, root)).status == PostStatus.digest
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    assert (await fresh(rt.store, root)).status == PostStatus.digest
    await sorter.submit(cand(chats[2], 1, A_REWORDED))
    got = await fresh(rt.store, root)
    assert got.status == PostStatus.queued and publisher.enqueued == [root.id]

    other = await sorter.submit(cand(chats[0], 2, B))
    assert other is not None
    clock.advance(timedelta(minutes=46))
    await sorter.tick()
    await rt.store.set_post_fields(other.id, status=PostStatus.digested)  # composed meanwhile
    await sorter.submit(cand(chats[1], 2, B + " 🔥"))
    await sorter.submit(cand(chats[2], 2, B + " 🔥🔥"))
    got = await fresh(rt.store, other)
    assert got.corroboration == 2 and got.status == PostStatus.digested
    assert other.id not in publisher.enqueued
    await sorter.tick()
    assert other.id not in publisher.enqueued


async def test_published_root_gets_a_plus_n_note(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    await trust(rt, chats[0], 3.0)
    root = await sorter.submit(cand(chats[0], 1, A))
    assert root is not None
    await rt.store.set_post_fields(root.id, status=PostStatus.published)
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    assert publisher.noted == [root.id]
    assert (await fresh(rt.store, root)).corroboration == 1
    assert publisher.enqueued == [root.id]  # only the original enqueue


async def test_self_healing_queues_posts_a_crash_left_without_a_row(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    await trust(rt, chats[0], 3.0)
    publisher.refuse = True  # the "crash" between the commit and the enqueue
    post = await sorter.submit(cand(chats[0], 1, A))
    assert post is not None and post.status == PostStatus.queued
    assert await rt.store.get_publication(post.id) is None
    held = await sorter.submit(cand(chats[1], 1, B))
    assert held is not None
    await rt.store.set_post_fields(held.id, status=PostStatus.digest, would_realtime=True)
    backfilled = await sorter.submit(cand(chats[2], 1, C, via="backfill"))
    assert backfilled is not None
    await rt.store.set_post_fields(backfilled.id, would_realtime=True)
    publisher.refuse = False
    publisher.enqueued.clear()
    await sorter.tick()
    assert publisher.enqueued == [post.id, held.id]
    assert (await fresh(rt.store, held)).status == PostStatus.queued
    assert await rt.store.get_publication(post.id) is not None
    await sorter.tick()
    assert publisher.enqueued == [post.id, held.id]


async def test_self_healing_never_revives_a_cancelled_publication(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic, publisher: StubPublisher
) -> None:
    await trust(rt, chats[0], 3.0)
    post = await sorter.submit(cand(chats[0], 1, A))
    assert post is not None
    row = await rt.store.get_publication(post.id)
    assert row is not None
    await rt.store.set_publication_fields(row.id, state=PUB_CANCELLED)
    await rt.store.set_post_fields(post.id, status=PostStatus.digest)
    publisher.enqueued.clear()
    await sorter.tick()
    assert publisher.enqueued == []
    await sorter.submit(cand(chats[1], 1, A_FOOTER))
    assert publisher.enqueued == [post.id]  # a promotion attempt the outbox refuses
    assert (await fresh(rt.store, post)).status == PostStatus.digest


async def test_without_a_publisher_nothing_is_queued_and_nothing_breaks(
    rt: Runtime, chats: list[int], topic: Topic
) -> None:
    rt.publisher = None
    sorter = Sorter(rt)
    await trust(rt, chats[0], 3.0)
    post = await sorter.submit(cand(chats[0], 1, A))
    assert post is not None and post.status == PostStatus.queued
    await sorter.tick()
    assert await rt.store.get_publication(post.id) is None


# --- the index across restarts -----------------------------------------------------------


async def test_index_is_rebuilt_from_the_store_at_start(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    forwarded = await sorter.submit(cand(chats[1], 1, C, fwd=(-1_001_999, 42)))
    ignored = await sorter.submit(cand(chats[2], 1, ""))
    assert root and forwarded and ignored
    restarted = Sorter(rt)
    await restarted.start()
    assert restarted.index is not None
    assert len(restarted.index) == 2 and ignored.id not in restarted.index
    dup = await restarted.submit(cand(chats[2], 2, A_REWORDED))
    assert dup is not None and dup.duplicate_of == root.id and dup.dup_kind == "semantic"
    twin = await restarted.submit(cand(chats[3], 1, "x", fwd=(-1_001_999, 42)))
    assert twin is not None and twin.duplicate_of == forwarded.id


async def test_duplicates_are_indexed_with_their_root(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    root = await sorter.submit(cand(chats[0], 1, A))
    mid = await sorter.submit(cand(chats[1], 1, A_REWORDED))
    assert root and mid
    restarted = Sorter(rt)
    last = await restarted.submit(cand(chats[2], 1, A_REWORDED + " 🔥"))
    assert last is not None and last.duplicate_of == root.id


async def test_tick_trims_the_index_to_the_window(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    clock: FakeClock = rt.clock  # type: ignore[assignment]
    old = await sorter.submit(cand(chats[0], 1, A))
    assert old is not None
    clock.advance(timedelta(days=3, minutes=1))
    await sorter.tick()
    assert sorter.index is not None and old.id not in sorter.index
    again = await sorter.submit(cand(chats[1], 1, A_FOOTER, posted_at=clock.now()))
    assert again is not None and again.status == PostStatus.held


# --- resort ---------------------------------------------------------------------------------------


async def test_resort_unsorted_sorts_old_posts_without_publishing(
    rt: Runtime, sorter: Sorter, chats: list[int], publisher: StubPublisher
) -> None:
    clock: FakeClock = rt.clock  # type: ignore[assignment]
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    await trust(rt, chats[0], 3.0)
    stale = await sorter.submit(cand(chats[0], 1, B, posted_at=START - timedelta(hours=30)))
    recent = await sorter.submit(cand(chats[0], 2, A))
    assert stale and recent and stale.status == recent.status == PostStatus.unsorted
    corrected = await sorter.submit(cand(chats[1], 1, C))
    assert corrected is not None
    await rt.store.set_post_fields(corrected.id, corrected=True)
    rejected = await sorter.submit(cand(chats[2], 1, "A rejected one about nothing."))
    assert rejected is not None
    await rt.store.set_post_fields(rejected.id, status=PostStatus.rejected)

    topic = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=clock.now())
    )
    classifier.scores = {topic.id: 0.9}
    n = await sorter.resort_unsorted(clock.now() - timedelta(days=7))
    assert n == 2
    assert (await fresh(rt.store, recent)).status == PostStatus.digest
    assert (await fresh(rt.store, stale)).status == PostStatus.dropped
    assert (await fresh(rt.store, corrected)).status == PostStatus.unsorted
    assert (await fresh(rt.store, rejected)).status == PostStatus.rejected
    got = await fresh(rt.store, recent)
    assert got.topic_id == topic.id and got.would_realtime is True and got.decided_at == START
    assert got.via == "backfill"  # sorted like a backfill, so never on the real-time path
    assert publisher.enqueued == []
    await sorter.tick()  # the self-healing pass must not pick it up either
    assert publisher.enqueued == []
    assert (await fresh(rt.store, recent)).status == PostStatus.digest

    tracked_topic = await rt.store.upsert_topic(
        Topic(id=0, key="other", name="Other", channel_id=None, created_at=clock.now())
    )
    classifier.scores = {topic.id: 0.1, tracked_topic.id: 0.9}
    await rt.store.set_post_fields(recent.id, status=PostStatus.unsorted, topic_id=None)
    assert await sorter.resort_unsorted(clock.now() - timedelta(days=7)) == 1
    assert (await fresh(rt.store, recent)).status == PostStatus.tracked
    assert await sorter.resort_unsorted(clock.now() - timedelta(days=7)) == 0


async def test_chat_daily_is_bumped_on_the_local_day_of_the_post(
    rt: Runtime, sorter: Sorter, chats: list[int], topic: Topic
) -> None:
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")
    late = START.replace(hour=20)  # 01:00 next day in Tashkent
    post = await sorter.submit(cand(chats[0], 1, A, posted_at=late))
    assert post is not None
    rows = await rt.store.execute(
        sa.select(schema.chat_daily.c.day).where(schema.chat_daily.c.chat_id == chats[0])
    )
    assert isinstance(rows, list) and rows[0].day == local_date(late, "Asia/Tashkent")
    assert rows[0].day == START.date() + timedelta(days=1)


async def test_a_group_unit_is_not_counted_again_by_the_sorter(
    rt: Runtime, sorter: Sorter, make_chat: Callable[..., ChatInfo], topic: Topic
) -> None:
    """§9.1: intake counted every message of a conversation unit when it stored them in
    ``group_messages``; the unit itself (one candidate spanning several messages) must not
    bump ``chat_daily`` or ``last_message_at`` a second time."""
    group = make_chat(kind="group")
    await rt.store.upsert_chat(group)
    unit = replace(cand(group.id, 1, A), kind="unit", message_ids=[1, 2, 3])
    post = await sorter.submit(unit)
    assert post is not None and post.kind == "unit"
    assert await daily(rt.store, group.id) == 0
    chat = await rt.store.get_chat(group.id)
    assert chat is not None and chat.last_message_at is None
