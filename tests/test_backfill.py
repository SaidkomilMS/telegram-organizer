"""Backfill (DESIGN §9.1): paced reads, global order, idempotence, chat_daily reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from tests.fakes import START, FakeClock, FakeUserGateway
from tests.test_intake import REPLY_TEXT, ROOT_TEXT, FakeSorter, daily, group_rows
from tg_curator.db import schema
from tg_curator.domain import KV, Chat
from tg_curator.errors import ChatGone, FloodWait
from tg_curator.pipeline.backfill import PAUSE_BETWEEN_CHATS, BackfillService
from tg_curator.pipeline.intake import IntakeService
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

DAY = START.date()


class Sleeps:
    """Records the pauses a backfill asked for instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sorter(rt: Runtime) -> FakeSorter:
    rt.sorter = FakeSorter(rt.store)
    return rt.sorter


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
def backfill(rt: Runtime, sorter: FakeSorter, sleeps: Sleeps) -> BackfillService:
    rt.intake = IntakeService(rt)
    rt.backfill = BackfillService(rt, sleep=sleeps)
    return rt.backfill


async def seed_source(rt: Runtime, user_gw: FakeUserGateway, info: ChatInfo) -> Chat:
    user_gw.add_chat(info)
    return await rt.store.upsert_chat(info)


async def posts_in_order(rt: Runtime) -> list[tuple[int, int]]:
    rows = await rt.store.execute(
        sa.select(schema.posts.c.chat_id, schema.posts.c.message_id).order_by(schema.posts.c.id)
    )
    return [(r.chat_id, r.message_id) for r in rows]


# --- ordering and pacing -----------------------------------------------------------------------


async def test_submits_in_global_posted_at_order_across_chats(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    sleeps: Sleeps,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    a = await seed_source(rt, user_gw, make_chat(title="A"))
    b = await seed_source(rt, user_gw, make_chat(title="B"))
    hours = {"a1": 30, "b1": 20, "a2": 10, "b2": 5}
    for key, back in hours.items():
        chat = a if key[0] == "a" else b
        user_gw.seed(make_message(chat, text=f"post {key}", date=START - timedelta(hours=back)))
    seen: list[tuple[int, int, int]] = []

    async def progress(done: int, total: int, chat: Chat) -> None:
        seen.append((done, total, chat.id))

    result = await backfill.run(days=3, progress=progress)

    assert [c.text for c in sorter.submitted] == ["post a1", "post b1", "post a2", "post b2"]
    assert all(c.via == "backfill" for c in sorter.submitted)
    assert await posts_in_order(rt) == [(a.id, 1), (b.id, 1), (a.id, 2), (b.id, 2)]
    # chats are read in `list_chats` order (by id); ids from the fixture count downwards
    assert seen == [(1, 2, b.id), (2, 2, a.id)]
    assert sleeps.calls == [PAUSE_BETWEEN_CHATS]
    assert result.chats == 2 and result.messages == 4
    assert result.candidates == 4 and result.submitted == 4 and result.skipped_chats == []
    assert result.per_chat == {a.id: 2, b.id: 2}
    assert [c["since"] for c in user_gw.calls_of("history")] == [START - timedelta(days=3)] * 2
    assert await rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT) == START.isoformat()


async def test_running_twice_changes_nothing(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel = await seed_source(rt, user_gw, make_chat())
    group = await seed_source(rt, user_gw, make_chat(kind="group"))
    day_before = START - timedelta(days=1)
    user_gw.seed(make_message(channel, text="one", date=day_before))
    user_gw.seed(make_message(channel, text="", media="photo", grouped_id=3, date=day_before))
    user_gw.seed(make_message(channel, text="album", media="photo", grouped_id=3, date=day_before))
    user_gw.seed(make_message(group, text=ROOT_TEXT, sender_id=500, date=day_before))
    user_gw.seed(
        make_message(group, text=REPLY_TEXT, sender_id=500, date=day_before + timedelta(minutes=2))
    )
    user_gw.seed(
        make_message(group, text="chatter", sender_id=501, date=day_before + timedelta(hours=1))
    )

    first = await backfill.run(days=3)
    assert first.messages == 6 and first.candidates == 3 and first.submitted == 3
    posts = await posts_in_order(rt)
    assert sorted(posts) == sorted([(channel.id, 1), (channel.id, 3), (group.id, 1)])
    assert await daily(rt.store, channel.id) == {DAY - timedelta(days=1): 2}
    assert await daily(rt.store, group.id) == {DAY - timedelta(days=1): 3}
    assert all(r.closed for r in await group_rows(rt.store, group.id))

    second = await backfill.run(days=3)
    assert second.messages == 6 and second.candidates == 2 and second.submitted == 0
    assert len(sorter.submitted) == 5
    assert await posts_in_order(rt) == posts
    assert await daily(rt.store, channel.id) == {DAY - timedelta(days=1): 2}
    assert await daily(rt.store, group.id) == {DAY - timedelta(days=1): 3}
    assert [r.unit_root_id for r in await group_rows(rt.store, group.id)] == [1, 1, 3]


# --- counting ----------------------------------------------------------------------------------


async def test_chat_daily_is_raised_to_the_count_for_fully_covered_days(
    rt: Runtime,
    backfill: BackfillService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel = await seed_source(rt, user_gw, make_chat())
    covered = datetime(2026, 10, 3, 15, 0, tzinfo=START.tzinfo)
    for i in range(3):
        user_gw.seed(make_message(channel, text=f"old {i}", date=covered + timedelta(minutes=i)))
    user_gw.seed(make_message(channel, text="today", date=START - timedelta(hours=1)))
    user_gw.seed(make_message(channel, text="service", is_service=True, date=covered))
    # the account's own post in a channel counts (only own messages in groups are ignored)
    user_gw.seed(make_message(channel, text="mine", is_outgoing=True, date=covered))

    await backfill.run(days=3)
    assert await daily(rt.store, channel.id) == {date(2026, 10, 3): 4, DAY: 1}

    # an undercounted covered day is repaired; the partial day (today) is left alone
    await rt.store.execute(
        sa.update(schema.chat_daily)
        .where(schema.chat_daily.c.chat_id == channel.id)
        .values(messages=1)
    )
    await rt.store.execute(
        sa.delete(schema.chat_daily)
        .where(schema.chat_daily.c.chat_id == channel.id)
        .where(schema.chat_daily.c.day == date(2026, 10, 3))
    )
    await backfill.run(days=3)
    assert await daily(rt.store, channel.id) == {date(2026, 10, 3): 4, DAY: 1}

    # never lowered
    await rt.store.bump_chat_daily(channel.id, date(2026, 10, 3), 5)
    await backfill.run(days=3)
    assert await daily(rt.store, channel.id) == {date(2026, 10, 3): 9, DAY: 1}


# --- failures and selection --------------------------------------------------------------------


async def test_flood_wait_is_slept_and_the_chat_re_read(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    sleeps: Sleeps,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel = await seed_source(rt, user_gw, make_chat())
    user_gw.seed(make_message(channel, text="post", date=START - timedelta(hours=2)))
    user_gw.fail_next("history", FloodWait(300))

    result = await backfill.run(days=1)
    assert sleeps.calls == [300.0]
    assert result.messages == 1 and result.submitted == 1 and result.skipped_chats == []
    assert len(user_gw.calls_of("history")) == 2


async def test_a_gone_chat_is_skipped_and_the_rest_proceed(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    fine = await seed_source(rt, user_gw, make_chat(title="Fine"))
    gone = await seed_source(rt, user_gw, make_chat(title="Gone"))  # lower id: read first
    user_gw.seed(make_message(fine, text="post", date=START - timedelta(hours=2)))
    user_gw.fail_next("history", ChatGone("kicked"))

    result = await backfill.run(days=1)
    assert result.skipped_chats == [gone.id] and result.chats == 1
    assert result.per_chat == {fine.id: 1} and result.submitted == 1


async def test_chat_ids_select_the_chats_and_resolve_unknown_ones(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    known = await seed_source(rt, user_gw, make_chat(title="Known"))
    other = await seed_source(rt, user_gw, make_chat(title="Other"))
    unknown = make_chat(title="Unknown")
    user_gw.add_chat(unknown)
    output = make_chat(title="Topic")
    await rt.store.upsert_chat(output, role="output")
    for chat in (known, other, unknown):
        user_gw.seed(
            make_message(chat, text=f"post in {chat.title}", date=START - timedelta(hours=1))
        )

    result = await backfill.run(days=1, chat_ids=[known.id, unknown.id, output.id, -5])

    assert sorted(c.text for c in sorter.submitted) == ["post in Known", "post in Unknown"]
    assert result.skipped_chats == [output.id, -5]
    assert result.per_chat == {known.id: 1, unknown.id: 1}
    stored = await rt.store.get_chat(unknown.id)
    assert stored is not None and stored.role == "source"
    assert other.id not in result.per_chat


async def test_backfilled_group_units_are_closed_by_force(
    rt: Runtime,
    backfill: BackfillService,
    sorter: FakeSorter,
    user_gw: FakeUserGateway,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = await seed_source(rt, user_gw, make_chat(kind="group"))
    recent = START - timedelta(minutes=1)  # younger than the gap: only force closes it
    user_gw.seed(make_message(group, text=ROOT_TEXT + " " + REPLY_TEXT, date=recent))

    result = await backfill.run(days=1)
    assert result.candidates == 1 and result.submitted == 1
    unit = sorter.submitted[0]
    assert unit.kind == "unit" and unit.via == "backfill" and unit.posted_at == recent
    assert [r.closed for r in await group_rows(rt.store, group.id)] == [True]
