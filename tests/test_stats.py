"""stats.py: the §12 definitions, the two worked examples, the window and the exclusions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

import pytest

from tests.fakes import START, FakeClock
from tg_curator.domain import ChatStats, NewPost, Post, PostStatus
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash


@pytest.fixture
def stats(rt: Runtime) -> StatsService:
    rt.stats = StatsService(rt)
    return rt.stats


async def source(rt: Runtime, info: ChatInfo, *, days: int = 30, role: str = "source") -> ChatInfo:
    """A chat row first seen ``days`` ago."""
    await rt.store.upsert_chat(info, role=role)  # type: ignore[arg-type]
    await rt.store.set_chat_fields(info.id, first_seen_at=rt.clock.now() - timedelta(days=days))
    return info


async def post(
    rt: Runtime,
    chat_id: int,
    message_id: int,
    *,
    status: PostStatus = PostStatus.unsorted,
    topic_id: int | None = None,
    duplicate_of: int | None = None,
    posted_at: datetime | None = None,
    message_ids: list[int] | None = None,
) -> Post:
    text = f"post {chat_id} {message_id}"
    new = NewPost(
        chat_id=chat_id,
        message_id=message_id,
        kind="post",
        message_ids=message_ids or [message_id],
        posted_at=posted_at or rt.clock.now(),
        via="live",
        text=text,
        text_hash=text_hash(text),
        urls=[],
    )
    created = await rt.store.insert_post(
        new, status=status, topic_id=topic_id, duplicate_of=duplicate_of
    )
    assert created is not None
    return created


def by_id(rows: list[ChatStats], chat_id: int) -> ChatStats:
    return next(s for s in rows if s.chat_id == chat_id)


# --- the worked examples of §12 --------------------------------------------------------------


async def test_group_example_500_raw_6_sorted_4_repeats(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    group = await source(rt, make_chat(kind="group", title="Loud group"))
    other = await source(rt, make_chat(username="kunuz"))
    today = rt.clock.now().date()
    for day, n in ((today, 200), (today - timedelta(days=1), 300)):
        await rt.store.bump_chat_daily(group.id, day, n)
    for i in range(6):
        await post(rt, group.id, 100 + i, status=PostStatus.digested, topic_id=1)
    for i in range(4):
        root = await post(rt, other.id, 500 + i, status=PostStatus.published, topic_id=1)
        await post(rt, group.id, 200 + i, status=PostStatus.duplicate, duplicate_of=root.id)
    [g] = [s for s in await stats.chat_stats() if s.chat_id == group.id]
    assert g.volume == 500 and g.sorted == 6 and g.duplicates == 4
    assert g.signal == pytest.approx(0.012) and g.duplicate_share == pytest.approx(0.008)
    assert g.published == 6 and g.top_repeated_chat_id == other.id and g.title == "Loud group"


async def test_channel_example_40_albums_and_10_text_posts_have_volume_50(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    channel = await source(rt, make_chat())
    # Intake counts an album once in chat_daily however many messages it spans (§9.1).
    await rt.store.bump_chat_daily(channel.id, rt.clock.now().date(), 50)
    for i in range(40):
        first = 1000 + 3 * i
        await post(rt, channel.id, first, message_ids=[first, first + 1, first + 2])
    for i in range(10):
        await post(rt, channel.id, 2000 + i)
    assert by_id(await stats.chat_stats(), channel.id).volume == 50


# --- definitions -----------------------------------------------------------------------------


async def test_zero_volume_is_zero_safe(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    silent = await source(rt, make_chat(title="Silent"))
    s = by_id(await stats.chat_stats(), silent.id)
    assert s == ChatStats(
        chat_id=silent.id,
        title="Silent",
        volume=0,
        sorted=0,
        signal=0.0,
        duplicates=0,
        duplicate_share=0.0,
        published=0,
        observed_days=30,
        top_repeated_chat_id=None,
    )


async def test_sorted_counts_any_topic_status_but_never_rejected_or_duplicates(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    chat = await source(rt, make_chat())
    other = await source(rt, make_chat())
    await rt.store.bump_chat_daily(chat.id, rt.clock.now().date(), 10)
    statuses = (
        PostStatus.tracked, PostStatus.held, PostStatus.queued, PostStatus.published,
        PostStatus.digest, PostStatus.digested, PostStatus.dropped,
    )  # fmt: skip
    for i, status in enumerate(statuses):
        await post(rt, chat.id, i + 1, status=status, topic_id=3)
    await post(rt, chat.id, 50, status=PostStatus.rejected, topic_id=3)
    await post(rt, chat.id, 51, status=PostStatus.unsorted)
    root = await post(rt, other.id, 1, status=PostStatus.published, topic_id=3)
    await post(rt, chat.id, 52, status=PostStatus.duplicate, topic_id=3, duplicate_of=root.id)
    s = by_id(await stats.chat_stats(), chat.id)
    assert s.sorted == len(statuses) and s.signal == pytest.approx(0.7)
    assert s.published == 2  # published + digested, a moved post counts once
    assert s.duplicates == 1


async def test_repeats_of_the_chats_own_posts_do_not_count(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    chat = await source(rt, make_chat())
    await rt.store.bump_chat_daily(chat.id, rt.clock.now().date(), 2)
    root = await post(rt, chat.id, 1, status=PostStatus.digest, topic_id=1)
    await post(rt, chat.id, 2, status=PostStatus.duplicate, duplicate_of=root.id)
    s = by_id(await stats.chat_stats(), chat.id)
    assert s.duplicates == 0 and s.duplicate_share == 0.0 and s.top_repeated_chat_id is None


async def test_top_repeated_chat_is_the_most_repeated_root_chat(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    chat = await source(rt, make_chat())
    a = await source(rt, make_chat(username="a_news"))
    b = await source(rt, make_chat(username="b_news"))
    await rt.store.bump_chat_daily(chat.id, rt.clock.now().date(), 10)
    for i in range(3):
        root = await post(rt, a.id, i + 1, status=PostStatus.published, topic_id=1)
        await post(rt, chat.id, 10 + i, status=PostStatus.duplicate, duplicate_of=root.id)
    root = await post(rt, b.id, 1, status=PostStatus.published, topic_id=1)
    await post(rt, chat.id, 20, status=PostStatus.duplicate, duplicate_of=root.id)
    s = by_id(await stats.chat_stats(), chat.id)
    assert s.duplicates == 4 and s.top_repeated_chat_id == a.id


# --- the window ------------------------------------------------------------------------------


async def test_window_is_the_last_n_local_days_for_volume_and_posts(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo], clock: FakeClock
) -> None:
    chat = await source(rt, make_chat(), days=60)
    today: date = clock.now().date()
    await rt.store.bump_chat_daily(chat.id, today, 1)
    await rt.store.bump_chat_daily(chat.id, today - timedelta(days=29), 10)  # inside (30 days)
    await rt.store.bump_chat_daily(chat.id, today - timedelta(days=30), 100)  # outside
    inside = clock.now() - timedelta(days=29)
    outside = clock.now() - timedelta(days=31)
    await post(rt, chat.id, 1, status=PostStatus.digested, topic_id=1, posted_at=inside)
    await post(rt, chat.id, 2, status=PostStatus.digested, topic_id=1, posted_at=outside)
    s = by_id(await stats.chat_stats(), chat.id)
    assert s.volume == 11 and s.sorted == 1 and s.published == 1 and s.observed_days == 30
    week = by_id(await stats.chat_stats(days=7), chat.id)
    assert week.volume == 1 and week.sorted == 0 and week.observed_days == 7


async def test_observed_days_is_capped_by_the_window(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    old = await source(rt, make_chat(), days=45)
    young = await source(rt, make_chat(), days=7)
    rows = await stats.chat_stats()
    assert by_id(rows, old.id).observed_days == 30
    assert by_id(rows, young.id).observed_days == 7


# --- which chats -----------------------------------------------------------------------------


async def test_left_chats_only_on_request_and_output_staging_never(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    active = await source(rt, make_chat())
    left = await source(rt, make_chat())
    await rt.store.set_chat_fields(left.id, active=False, left_at=rt.clock.now())
    output = await source(rt, make_chat(), role="output")
    staging = await source(rt, make_chat(), role="staging")
    ids = {s.chat_id for s in await stats.chat_stats()}
    assert ids == {active.id}
    ids_all = {s.chat_id for s in await stats.chat_stats(include_left=True)}
    assert ids_all == {active.id, left.id}
    assert output.id not in ids_all and staging.id not in ids_all


async def test_rows_are_ordered_by_volume_then_title(
    rt: Runtime, stats: StatsService, make_chat: Callable[..., ChatInfo]
) -> None:
    small = await source(rt, make_chat(title="b small"))
    big = await source(rt, make_chat(title="a big"))
    quiet = await source(rt, make_chat(title="A quiet"))
    await rt.store.bump_chat_daily(small.id, START.date(), 1)
    await rt.store.bump_chat_daily(big.id, START.date(), 5)
    rows = await stats.chat_stats()
    assert [s.chat_id for s in rows] == [big.id, small.id, quiet.id]
