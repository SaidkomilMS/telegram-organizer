"""Store: every shared helper round-trips, upserts behave, transactions and locks hold (§7)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, StatementError

from tg_curator.db import schema
from tg_curator.db.store import ANY, Store, sqlite_url
from tg_curator.domain import (
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PROPOSAL_PROPOSED,
    PUB_FAILED,
    PUB_PENDING,
    PUB_SENT,
    DigestLine,
    Example,
    NewPost,
    Post,
    PostStatus,
    Topic,
)

PG_URL = os.environ.get("TG_CURATOR_TEST_PG_URL")
BACKENDS = [
    "sqlite",
    pytest.param(
        "postgres", marks=pytest.mark.skipif(not PG_URL, reason="TG_CURATOR_TEST_PG_URL not set")
    ),
]

T0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current

    def advance(self, **delta: int) -> datetime:
        self.current += timedelta(**delta)
        return self.current


@dataclass(frozen=True)
class Info:
    """Stand-in for ``telegram.gateway.ChatInfo`` (§5), field for field."""

    id: int
    kind: str = "channel"
    title: str = "Example"
    username: str | None = None
    noforwards: bool = False
    is_creator: bool = False
    is_admin: bool = False
    archived: bool = False
    muted_until: datetime | None = None


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=BACKENDS)
async def store(
    request: pytest.FixtureRequest, tmp_path: Path, clock: FakeClock
) -> AsyncIterator[Store]:
    url = sqlite_url(tmp_path / "curator.db") if request.param == "sqlite" else str(PG_URL)
    s = Store(url, clock=clock)
    await s.start()
    yield s
    if request.param == "postgres":
        async with s.engine.begin() as conn:
            await conn.run_sync(schema.metadata.drop_all)
    await s.close()


def new_post(chat_id: int = -1001, message_id: int = 1, **over: Any) -> NewPost:
    values: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "kind": "post",
        "message_ids": [message_id],
        "posted_at": T0 - timedelta(minutes=5),
        "via": "live",
        "text": "Hello world, this is a post",
        "text_hash": "abc",
        "urls": ["https://example.org/a"],
    }
    values.update(over)
    return NewPost(**values)


def topic(key: str = "ml-ai", **over: Any) -> Topic:
    values: dict[str, Any] = {"id": 0, "key": key, "name": key.upper(), "created_at": T0}
    values.update(over)
    return Topic(**values)


# --- kv -------------------------------------------------------------------------------------


async def test_kv_roundtrip(store: Store) -> None:
    assert await store.kv_get("missing") is None
    assert await store.kv_get("missing", 7) == 7
    await store.kv_set("setup.step", "phone")
    await store.kv_set("bot.conversation", {"flow": "topic_add", "step": 2, "data": [1, "x"]})
    assert await store.kv_get("setup.step") == "phone"
    assert await store.kv_get("bot.conversation") == {
        "flow": "topic_add",
        "step": 2,
        "data": [1, "x"],
    }
    await store.kv_set("setup.step", 4)
    assert await store.kv_get("setup.step") == 4
    await store.kv_delete("setup.step")
    assert await store.kv_get("setup.step") is None
    await store.kv_delete("setup.step")


# --- chats ----------------------------------------------------------------------------------


async def test_upsert_chat_insert_and_refresh(store: Store, clock: FakeClock) -> None:
    chat = await store.upsert_chat(Info(-1001, title="Kun.uz", username="kunuz"))
    assert chat.id == -1001
    assert chat.role == "source"
    assert chat.active is True
    assert chat.first_seen_at == T0
    assert chat.trust is None and chat.keep is False and chat.left_at is None

    await store.set_chat_fields(-1001, active=False, left_at=T0, trust=2.0, keep=True)
    clock.advance(hours=1)
    again = await store.upsert_chat(
        Info(-1001, title="Kun.uz news", archived=True, muted_until=T0 + timedelta(days=30))
    )
    assert again.title == "Kun.uz news"
    assert again.first_seen_at == T0, "first_seen_at is kept on refresh"
    assert again.active is True and again.left_at is None, "a chat seen again is active"
    assert again.trust == 2.0 and again.keep is True, "our own columns survive a refresh"
    assert again.role == "source", "role is kept when not given"
    assert again.archived is True
    assert again.muted_until == T0 + timedelta(days=30)

    output = await store.upsert_chat(Info(-1001), role="output")
    assert output.role == "output"
    kept = await store.upsert_chat(Info(-1001))
    assert kept.role == "output"


async def test_get_and_list_chats(store: Store) -> None:
    assert await store.get_chat(1) is None
    await store.upsert_chat(Info(-1), role="source")
    await store.upsert_chat(Info(-2, kind="group"), role="source")
    await store.upsert_chat(Info(-3), role="output")
    await store.set_chat_fields(-2, active=False)
    assert [c.id for c in await store.list_chats()] == [-3, -2, -1]
    assert [c.id for c in await store.list_chats(role="source")] == [-2, -1]
    assert [c.id for c in await store.list_chats(role="source", active=True)] == [-1]
    assert [c.id for c in await store.list_chats(active=False)] == [-2]
    got = await store.get_chat(-2)
    assert got is not None and got.kind == "group"


async def test_set_fields_rejects_unknown_and_primary_key(store: Store) -> None:
    await store.upsert_chat(Info(-1))
    with pytest.raises(TypeError, match="no column"):
        await store.set_chat_fields(-1, colour="red")
    with pytest.raises(TypeError, match="primary key"):
        await store.set_chat_fields(-1, id=5)
    assert await store.set_chat_fields(-1) == 0
    assert await store.set_chat_fields(-999, title="x") == 0


async def test_bump_chat_daily_and_touch_chat(store: Store) -> None:
    await store.upsert_chat(Info(-1))
    day = date(2026, 10, 6)
    await store.bump_chat_daily(-1, day)
    await store.bump_chat_daily(-1, day, n=4)
    await store.bump_chat_daily(-1, date(2026, 10, 7))
    rows = await store.execute(
        sa.select(schema.chat_daily.c.day, schema.chat_daily.c.messages).order_by(
            schema.chat_daily.c.day
        )
    )
    assert [(r.day, r.messages) for r in rows] == [(day, 5), (date(2026, 10, 7), 1)]

    await store.touch_chat(-1, T0)
    await store.touch_chat(-1, T0 - timedelta(days=1))
    chat = await store.get_chat(-1)
    assert chat is not None and chat.last_message_at == T0
    await store.touch_chat(-1, T0 + timedelta(hours=1))
    chat = await store.get_chat(-1)
    assert chat is not None and chat.last_message_at == T0 + timedelta(hours=1)


# --- topics ---------------------------------------------------------------------------------


async def test_topics_upsert_get_list(store: Store) -> None:
    created = await store.upsert_topic(topic("ml-ai", name="ML & AI", category="tech"))
    assert created.id > 0 and created.key == "ml-ai" and created.active is True
    assert created.origin == "user" and created.created_at == T0

    updated = await store.upsert_topic(
        topic(
            "ml-ai", name="Machine learning", channel_id=-100500, created_at=T0 + timedelta(days=9)
        )
    )
    assert updated.id == created.id, "upsert by key keeps the row"
    assert updated.created_at == T0, "created_at is kept"
    assert updated.name == "Machine learning" and updated.channel_id == -100500
    assert updated.category is None, "the given Topic is the whole truth"

    other = await store.upsert_topic(topic("fintech", origin="discovered", strictness=0.7))
    assert other.origin == "discovered" and other.strictness == 0.7
    await store.set_topic_fields(other.id, active=False)

    assert [t.key for t in await store.list_topics()] == ["ml-ai"]
    assert [t.key for t in await store.list_topics(active=None)] == ["ml-ai", "fintech"]
    assert [t.key for t in await store.list_topics(active=False)] == ["fintech"]
    assert (await store.get_topic(other.id)) == await store.get_topic_by_key("fintech")
    assert await store.get_topic(9999) is None
    assert await store.get_topic_by_key("nope") is None


# --- posts ----------------------------------------------------------------------------------


async def test_insert_post_roundtrip_and_duplicate(store: Store) -> None:
    post = await store.insert_post(new_post(media="photo", embedding=b"\x00\x01", views=10))
    assert isinstance(post, Post)
    assert post.id > 0
    assert post.ingested_at == T0
    assert post.status is PostStatus.unsorted
    assert post.corroboration == 0 and post.corroborating_chats == []
    assert post.message_ids == [1] and post.urls == ["https://example.org/a"]
    assert post.embedding == b"\x00\x01" and post.media == "photo" and post.views == 10
    assert post.topic_scores is None and post.decided_at is None

    assert await store.insert_post(new_post(text="same ids, other text")) is None
    assert await store.get_post(post.id) == post
    assert await store.get_post_by_message(-1001, 1) == post
    assert await store.get_post_by_message(-1001, 2) is None
    assert await store.get_post(post.id + 100) is None


async def test_insert_post_with_decision(store: Store) -> None:
    post = await store.insert_post(
        new_post(),
        status=PostStatus.held,
        topic_id=3,
        confidence=0.8,
        topic_scores=[{"topic_id": 3, "confidence": 0.8}],
        strength=1.25,
        hold_until=T0 + timedelta(minutes=45),
        decided_at=T0,
    )
    assert post is not None
    assert post.status is PostStatus.held and post.topic_id == 3
    assert post.topic_scores == [{"topic_id": 3, "confidence": 0.8}]
    assert post.hold_until == T0 + timedelta(minutes=45)
    with pytest.raises(TypeError, match="no column"):
        await store.insert_post(new_post(message_id=2), bogus=1)


async def test_set_post_fields_with_enum(store: Store) -> None:
    post = await store.insert_post(new_post())
    assert post is not None
    assert await store.set_post_fields(post.id, status=PostStatus.digest, topic_id=1) == 1
    got = await store.get_post(post.id)
    assert got is not None and got.status is PostStatus.digest and got.topic_id == 1
    raw = await store.execute(sa.select(schema.posts.c.status).where(schema.posts.c.id == post.id))
    assert raw[0].status == "digest"


async def test_recent_posts(store: Store) -> None:
    old = await store.insert_post(new_post(message_id=1, posted_at=T0 - timedelta(days=5)))
    a = await store.insert_post(
        new_post(message_id=2, posted_at=T0 - timedelta(hours=2), embedding=b"\x01"),
        status=PostStatus.digest,
    )
    b = await store.insert_post(
        new_post(message_id=3, posted_at=T0 - timedelta(hours=1), embedding=b"\x02"),
        status=PostStatus.ignored,
    )
    assert old and a and b
    since = T0 - timedelta(days=3)
    assert [p.id for p in await store.recent_posts(since)] == [a.id, b.id]
    assert all(p.embedding is None for p in await store.recent_posts(since))
    with_emb = await store.recent_posts(since, with_embeddings=True)
    assert [p.embedding for p in with_emb] == [b"\x01", b"\x02"]
    only = await store.recent_posts(since, statuses=[PostStatus.digest, "held"])
    assert [p.id for p in only] == [a.id]


async def test_posts_by_status(store: Store) -> None:
    h1 = await store.insert_post(
        new_post(message_id=1, posted_at=T0 - timedelta(hours=3)),
        status=PostStatus.held,
        topic_id=1,
        hold_until=T0 - timedelta(minutes=1),
    )
    h2 = await store.insert_post(
        new_post(message_id=2, posted_at=T0 - timedelta(hours=2)),
        status=PostStatus.held,
        topic_id=2,
        hold_until=T0 + timedelta(minutes=30),
    )
    d1 = await store.insert_post(
        new_post(message_id=3, posted_at=T0 - timedelta(hours=1)),
        status=PostStatus.digest,
        topic_id=1,
    )
    assert h1 and h2 and d1
    assert [p.id for p in await store.posts_by_status(PostStatus.held)] == [h1.id, h2.id]
    assert [p.id for p in await store.posts_by_status("held", due_before=T0)] == [h1.id]
    assert [p.id for p in await store.posts_by_status("held", topic_id=2)] == [h2.id]
    assert [p.id for p in await store.posts_by_status(["held", "digest"], limit=2)] == [
        h1.id,
        h2.id,
    ]
    assert [p.id for p in await store.posts_by_status("digest", topic_id=1)] == [d1.id]
    assert await store.posts_by_status("queued") == []


async def test_add_corroboration_counts_each_chat_once(store: Store) -> None:
    root = await store.insert_post(new_post(chat_id=-1, message_id=1))
    assert root is not None
    assert await store.add_corroboration(root.id, -2) == 1
    assert await store.add_corroboration(root.id, -2) is None
    assert await store.add_corroboration(root.id, -3) == 2
    assert await store.add_corroboration(root.id, -1) is None, "the root's own chat never counts"
    assert await store.add_corroboration(root.id + 99, -4) is None
    got = await store.get_post(root.id)
    assert got is not None
    assert got.corroboration == 2 and got.corroborating_chats == [-2, -3]


# --- examples -------------------------------------------------------------------------------


def example(**over: Any) -> Example:
    values: dict[str, Any] = {
        "id": 0,
        "topic_id": 1,
        "kind": "example",
        "text": "an example post",
        "embedding": b"\x00" * 8,
        "created_at": T0,
    }
    values.update(over)
    return Example(**values)


async def test_examples_add_list_delete(store: Store) -> None:
    e1 = await store.add_example(example())
    e2 = await store.add_example(example(text="another"))
    assert e1.id != e2.id, "examples without a post are never merged"
    neg = await store.add_example(
        example(topic_id=None, kind="correction", post_id=5, wrong_topic_id=1)
    )
    assert neg.topic_id is None and neg.wrong_topic_id == 1 and neg.weight == 1.0

    assert [e.id for e in await store.list_examples()] == [e1.id, e2.id, neg.id]
    assert [e.id for e in await store.list_examples(topic_id=1)] == [e1.id, e2.id]
    assert [e.id for e in await store.list_examples(topic_id=None)] == [neg.id]
    assert [e.id for e in await store.list_examples(ANY)] == [e1.id, e2.id, neg.id]
    assert await store.list_examples(topic_id=42) == []

    assert await store.delete_examples(1) == 2
    assert [e.id for e in await store.list_examples()] == [neg.id]


async def test_second_correction_replaces_the_first(store: Store) -> None:
    first = await store.add_example(
        example(kind="correction", post_id=7, topic_id=1, wrong_topic_id=2)
    )
    second = await store.add_example(
        example(
            kind="correction",
            post_id=7,
            topic_id=None,
            wrong_topic_id=1,
            text="new",
            embedding=b"\x01",
        )
    )
    assert second.id == first.id
    assert second.topic_id is None and second.wrong_topic_id == 1
    assert second.text == "new" and second.embedding == b"\x01"
    cluster = await store.add_example(example(kind="cluster", post_id=7, topic_id=3))
    assert cluster.id != first.id, "uniqueness is per (post, kind)"
    assert len(await store.list_examples()) == 2


# --- publications ---------------------------------------------------------------------------


async def test_publications(store: Store, clock: FakeClock) -> None:
    post = await store.insert_post(new_post())
    assert post is not None
    pub = await store.create_publication(post.id, topic_id=1)
    assert pub is not None
    assert pub.state == PUB_PENDING and pub.channel_id is None and pub.style is None
    assert pub.message_ids == [] and pub.staging_ids == [] and pub.attempts == 0
    assert pub.created_at == T0 and pub.moved_from is None
    assert await store.create_publication(post.id, topic_id=2) is None, "idempotent per post"
    assert await store.get_publication(post.id) == pub
    assert await store.get_publication(post.id + 1) is None

    clock.advance(minutes=1)
    assert (
        await store.set_publication_fields(
            pub.id,
            state=PUB_SENT,
            channel_id=-100,
            style="repost",
            message_ids=[10, 11],
            sent_at=clock.now(),
            moved_from=[{"channel_id": -99, "message_ids": [1], "topic_id": 3}],
        )
        == 1
    )
    sent = await store.get_publication(post.id)
    assert sent is not None
    assert sent.state == PUB_SENT and sent.message_ids == [10, 11] and sent.sent_at == clock.now()
    assert sent.moved_from == [{"channel_id": -99, "message_ids": [1], "topic_id": 3}]

    other = await store.insert_post(new_post(message_id=2))
    assert other is not None
    pub2 = await store.create_publication(other.id, topic_id=1)
    assert pub2 is not None
    await store.set_publication_fields(pub2.id, state=PUB_FAILED)
    assert [p.id for p in await store.publications_in_state(PUB_PENDING)] == []
    assert [p.id for p in await store.publications_in_state([PUB_SENT, PUB_FAILED])] == [
        pub.id,
        pub2.id,
    ]
    assert [p.id for p in await store.publications_in_state([PUB_SENT, PUB_FAILED], limit=1)] == [
        pub.id
    ]


# --- digests --------------------------------------------------------------------------------


async def test_digests(store: Store) -> None:
    day = date(2026, 10, 6)
    assert await store.next_manual_seq(1, day) == 1, (
        "first manual run gets seq 1 without a scheduled row"
    )
    p1 = await store.insert_post(new_post(message_id=1))
    p2 = await store.insert_post(new_post(message_id=2))
    assert p1 and p2

    scheduled = await store.create_digest(1, -100, day, body=["part 1", "part 2"], item_count=2)
    assert scheduled.seq == 0 and scheduled.manual is False and scheduled.state == "pending"
    assert scheduled.body == ["part 1", "part 2"] and scheduled.message_ids == []
    assert scheduled.day == day and scheduled.created_at == T0 and scheduled.attempts == 0
    await store.add_digest_items(
        scheduled.id, [DigestLine(1, p1.id, "first"), DigestLine(2, p2.id, "second")]
    )
    items = await store.list_digest_items(scheduled.id)
    assert [(i.position, i.post_id, i.line) for i in items] == [
        (1, p1.id, "first"),
        (2, p2.id, "second"),
    ]
    await store.add_digest_items(scheduled.id, [])

    assert await store.next_manual_seq(1, day) == 1
    manual = await store.create_digest(1, -100, day, body=["m"], item_count=0, seq=1, manual=True)
    assert manual.seq == 1 and manual.manual is True
    assert await store.next_manual_seq(1, day) == 2
    assert await store.next_manual_seq(2, day) == 1

    assert await store.get_digest(scheduled.id) == scheduled
    assert await store.get_digest_by_key(1, day) == scheduled
    assert await store.get_digest_by_key(1, day, seq=1) == manual
    assert await store.get_digest_by_key(1, date(2026, 10, 7)) is None
    assert await store.get_digest(999) is None

    await store.set_digest_fields(scheduled.id, state="sending", message_ids=[5])
    assert [d.id for d in await store.digests_in_state("sending")] == [scheduled.id]
    assert [d.id for d in await store.digests_in_state(["pending", "sending"])] == [
        scheduled.id,
        manual.id,
    ]
    assert [d.id for d in await store.digests_in_state("pending", limit=5)] == [manual.id]

    with pytest.raises(IntegrityError):
        await store.create_digest(1, -100, day, body=[], item_count=0, seq=1, manual=True)
    with pytest.raises(IntegrityError):
        await store.add_digest_items(manual.id, [DigestLine(1, p1.id, "again")])


# --- proposals ------------------------------------------------------------------------------


async def test_proposals(store: Store, clock: FakeClock) -> None:
    day = date(2026, 10, 4)
    p = await store.create_proposal(
        "mute", chat_id=-1, reason="41 posts, 0 reached a topic", review_day=day
    )
    assert p.id > 0 and p.state == PROPOSAL_PROPOSED and p.payload == {} and p.kind == "mute"
    assert p.review_day == day and p.created_at == T0 and p.bot_message_id is None
    nt = await store.create_proposal(
        "new_topic",
        reason="40 unsorted posts look alike",
        review_day=day,
        payload={"member_post_ids": [1, 2], "name": "Crypto"},
    )
    assert nt.chat_id is None and nt.payload["member_post_ids"] == [1, 2]
    assert await store.get_proposal(p.id) == p
    assert await store.get_proposal(p.id + 50) is None

    assert await store.open_proposal_for_chat(-1) == p
    assert await store.open_proposal_for_chat(-2) is None
    clock.advance(minutes=3)
    await store.set_proposal_fields(
        p.id, state=PROPOSAL_APPROVED, decided_at=clock.now(), bot_message_id=77
    )
    got = await store.open_proposal_for_chat(-1)
    assert got is not None and got.state == PROPOSAL_APPROVED and got.bot_message_id == 77
    await store.set_proposal_fields(p.id, state=PROPOSAL_DONE, result="Muted until Nov 4")
    assert await store.open_proposal_for_chat(-1) is None

    assert [x.id for x in await store.proposals_by_state(PROPOSAL_PROPOSED)] == [nt.id]
    assert [x.id for x in await store.proposals_by_state([PROPOSAL_PROPOSED, PROPOSAL_DONE])] == [
        p.id,
        nt.id,
    ]
    assert [x.id for x in await store.proposals_by_state(PROPOSAL_PROPOSED, kind="mute")] == []
    assert [x.id for x in await store.proposals_by_state(PROPOSAL_DONE, review_day=day)] == [p.id]
    assert await store.proposals_by_state(PROPOSAL_DONE, review_day=date(2026, 1, 1)) == []
    assert [x.id for x in await store.proposals_by_state(PROPOSAL_PROPOSED, limit=1)] == [nt.id]


# --- llm_usage ------------------------------------------------------------------------------


async def test_llm_usage(store: Store) -> None:
    empty = await store.get_usage("2026-10")
    assert (empty.requests, empty.input_tokens, empty.output_tokens, empty.cost_usd) == (
        0,
        0,
        0,
        0.0,
    )
    assert empty.cap_notified is False
    usage = await store.add_usage("2026-10", 1, 120, 30, 0.004)
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (1, 120, 30)
    usage = await store.add_usage("2026-10", 2, 80, 20, 0.001)
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (3, 200, 50)
    assert usage.cost_usd == pytest.approx(0.005)
    settled = await store.add_usage("2026-10", 0, -20, -5, -0.001)
    assert (settled.input_tokens, settled.output_tokens) == (180, 45)
    await store.set_cap_notified("2026-10")
    assert (await store.get_usage("2026-10")).cap_notified is True
    await store.set_cap_notified("2026-10", False)
    assert (await store.get_usage("2026-10")).cap_notified is False
    await store.set_cap_notified("2026-11")
    nov = await store.get_usage("2026-11")
    assert nov.cap_notified is True and nov.requests == 0


# --- group_messages -------------------------------------------------------------------------


async def test_group_messages(store: Store) -> None:
    t = T0 - timedelta(minutes=10)
    assert await store.add_group_message(
        -5, 1, date=t, text="root", sender_id=42, urls=["https://x.y/z"]
    )
    assert not await store.add_group_message(-5, 1, date=t, text="again", sender_id=42)
    assert await store.add_group_message(
        -5, 2, date=t + timedelta(minutes=1), text="reply", reply_to_id=1
    )
    assert await store.add_group_message(-6, 1, date=t + timedelta(minutes=2), text="other chat")

    opened = await store.open_group_messages()
    assert [(m.chat_id, m.message_id) for m in opened] == [(-5, 1), (-5, 2), (-6, 1)]
    assert (
        opened[0].urls == ["https://x.y/z"]
        and opened[0].sender_id == 42
        and opened[0].closed is False
    )
    assert opened[1].reply_to_id == 1 and opened[1].date == t + timedelta(minutes=1)
    assert [m.message_id for m in await store.open_group_messages(-5)] == [1, 2]

    assert await store.mark_group_messages(-5, [1, 2], unit_root_id=1) == 2
    assert await store.mark_group_messages(-5, [], closed=True) == 0
    assert await store.mark_group_messages(-5, [1, 2], closed=True) == 2
    assert await store.open_group_messages(-5) == []
    rows = await store.execute(
        sa.select(schema.group_messages.c.unit_root_id).where(schema.group_messages.c.chat_id == -5)
    )
    assert [r.unit_root_id for r in rows] == [1, 1]
    with pytest.raises(TypeError):
        await store.mark_group_messages(-5, [1], text="no")

    assert await store.purge_group_messages(t + timedelta(minutes=2)) == 2
    assert [m.chat_id for m in await store.open_group_messages()] == [-6]


# --- types ----------------------------------------------------------------------------------


async def test_datetimes_come_back_as_utc(store: Store) -> None:
    tashkent = timezone(timedelta(hours=5))
    local = datetime(2026, 10, 6, 17, 30, 15, 123456, tzinfo=tashkent)
    post = await store.insert_post(new_post(posted_at=local, views_at=local))
    assert post is not None
    for value in (post.posted_at, post.views_at, post.ingested_at):
        assert value is not None and value.tzinfo is not None
        assert value.utcoffset() == timedelta(0)
    assert post.posted_at == local
    assert post.posted_at.hour == 12 and post.posted_at.microsecond == 123456
    rows = await store.execute(
        sa.select(schema.posts.c.id).where(schema.posts.c.posted_at == local.astimezone(UTC))
    )
    assert [r.id for r in rows] == [post.id]


async def test_naive_datetime_is_refused(store: Store) -> None:
    with pytest.raises(StatementError, match="naive"):
        await store.insert_post(new_post(posted_at=datetime(2026, 10, 6, 12, 0)))
    assert await store.get_post_by_message(-1001, 1) is None


async def test_jsontext(store: Store) -> None:
    value = {"name": "Криптo & биржалар", "ids": [1, 2, [3]], "nested": {"ok": True, "none": None}}
    await store.kv_set("bot.conversation", value)
    assert await store.kv_get("bot.conversation") == value
    await store.kv_set("bot.conversation", None)
    assert await store.kv_get("bot.conversation", "default") is None, "NULL is a stored None"
    raw = await store.execute(
        sa.select(schema.kv.c.value).where(schema.kv.c.key == "bot.conversation")
    )
    assert raw[0].value is None
    await store.kv_set("x", "plain string")
    assert await store.kv_get("x") == "plain string"


# --- transactions and locking ---------------------------------------------------------------


async def test_helpers_share_the_callers_connection(store: Store) -> None:
    """Helpers called with the block's conn neither deadlock nor commit on their own."""

    async def block() -> None:
        async with store.begin() as conn:
            post = await store.insert_post(new_post(), conn=conn)
            assert post is not None
            await store.set_post_fields(post.id, status=PostStatus.held, conn=conn)
            await store.bump_chat_daily(-1001, date(2026, 10, 6), conn=conn)
            await store.kv_set("k", 1, conn=conn)
            assert await store.get_post(post.id, conn=conn) is not None
            assert await store.kv_get("k", conn=conn) == 1
            assert await store.add_corroboration(post.id, -2, conn=conn) == 1
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        await asyncio.wait_for(block(), timeout=5)
    assert await store.get_post_by_message(-1001, 1) is None, "nothing of the block was committed"
    assert await store.kv_get("k") is None

    async with store.begin() as conn:
        post = await store.insert_post(new_post(), conn=conn)
        assert post is not None
        await store.set_post_fields(post.id, status=PostStatus.queued, conn=conn)
    got = await store.get_post_by_message(-1001, 1)
    assert got is not None and got.status is PostStatus.queued


async def test_nested_begin_in_one_task_is_refused(store: Store) -> None:
    async def nested() -> None:
        async with store.begin():
            await store.kv_set("k", 1)

    with pytest.raises(RuntimeError, match="nested"):
        await asyncio.wait_for(nested(), timeout=5)
    assert await store.kv_get("k") is None
    await store.kv_set("k", 2)
    assert await store.kv_get("k") == 2


async def test_write_transactions_are_serialised_and_reads_are_not(store: Store) -> None:
    inside = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def writer_a() -> None:
        async with store.begin() as conn:
            await store.kv_set("a", 1, conn=conn)
            order.append("a-in")
            inside.set()
            await release.wait()
            order.append("a-out")

    async def writer_b() -> None:
        await inside.wait()
        async with store.begin() as conn:
            order.append("b-in")
            await store.kv_set("b", 1, conn=conn)

    async def reader() -> Any:
        await inside.wait()
        value = await store.kv_get("a", "unset")
        order.append("read")
        return value

    task_a = asyncio.create_task(writer_a())
    task_b = asyncio.create_task(writer_b())
    task_r = asyncio.create_task(reader())
    read_value = await asyncio.wait_for(task_r, timeout=5)
    if store.is_sqlite:
        assert "b-in" not in order, "the second writer waits for the first on SQLite"
        assert read_value == "unset", "the read ran before the first writer committed"
    release.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)
    if store.is_sqlite:
        assert order.index("a-out") < order.index("b-in")
    assert await store.kv_get("a") == 1 and await store.kv_get("b") == 1


async def test_execute_escape_hatch(store: Store) -> None:
    await store.upsert_chat(Info(-1, title="one"))
    await store.upsert_chat(Info(-2, title="two"))
    rows = await store.execute(sa.select(schema.chats.c.title).order_by(schema.chats.c.id))
    assert [r.title for r in rows] == ["two", "one"]
    count = await store.execute(sa.update(schema.chats).values(keep=True))
    assert count == 2
    assert all(c.keep for c in await store.list_chats())
    async with store.begin() as conn:
        assert (
            await store.execute(sa.delete(schema.chats).where(schema.chats.c.id == -1), conn) == 1
        )
        rows = await store.execute(sa.select(schema.chats.c.id), conn)
        assert [r.id for r in rows] == [-2]
    assert await store.get_chat(-1) is None


async def test_store_must_be_started(tmp_path: Path) -> None:
    s = Store(sqlite_url(tmp_path / "x.db"))
    with pytest.raises(RuntimeError, match="not started"):
        await s.kv_get("k")
    await s.close()


async def test_sqlite_pragmas(tmp_path: Path) -> None:
    s = Store(sqlite_url(tmp_path / "x.db"))
    await s.start()
    async with s.connect() as conn:
        assert (await conn.execute(sa.text("PRAGMA journal_mode"))).scalar() == "wal"
        assert (await conn.execute(sa.text("PRAGMA foreign_keys"))).scalar() == 1
        assert (await conn.execute(sa.text("PRAGMA busy_timeout"))).scalar() == 5000
    await s.close()


async def test_foreign_keys_are_enforced(store: Store) -> None:
    with pytest.raises(IntegrityError):
        await store.create_publication(12345, topic_id=1)
    assert await store.get_publication(12345) is None
