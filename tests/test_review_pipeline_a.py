"""Regression tests for the pipeline review findings (intake, backfill, sorter).

Live intake and a backfill share one ``IntakeService`` in the running service, so most of
these interleave the live ``tick`` with a backfill that is still reading.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

import pytest

from tests.fakes import START, FakeClassifier, FakeClock, FakeUserGateway
from tests.test_intake import (
    FOLLOW_TEXT,
    REPLY_TEXT,
    ROOT_TEXT,
    FakeSorter,
    daily,
    group_rows,
    stream,
)
from tests.test_sorter import (
    A_FOOTER,
    CHANNEL,
    A,
    StubPublisher,
    cand,
    fresh,
    language_stand_ins,  # noqa: F401 - autouse fixture, used by the sorter tests
    trust,
)
from tg_curator.domain import KV, Candidate, Post, PostStatus, Topic
from tg_curator.errors import ChatGone, FloodWait
from tg_curator.pipeline.backfill import BackfillService
from tg_curator.pipeline.intake import SUBMIT_ATTEMPTS, IntakeService
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage

LONG = ROOT_TEXT + " " + REPLY_TEXT  # one message that passes the 400-char floor


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FailingSorter(FakeSorter):
    """A ``FakeSorter`` whose submit raises for the chosen ``(chat_id, message_id)`` keys."""

    def __init__(self, store: Any) -> None:
        super().__init__(store)
        self.fail: set[tuple[int, int]] = set()
        self.attempts: list[tuple[int, int]] = []

    async def submit(self, c: Candidate) -> Post | None:
        self.attempts.append((c.chat_id, c.message_id))
        if (c.chat_id, c.message_id) in self.fail:
            raise RuntimeError("embedder exploded")
        return await super().submit(c)


@pytest.fixture
def fake_sorter(rt: Runtime) -> FailingSorter:
    rt.sorter = FailingSorter(rt.store)
    return rt.sorter


@pytest.fixture
def intake(rt: Runtime, fake_sorter: FailingSorter) -> IntakeService:
    rt.intake = IntakeService(rt, started_at=START - timedelta(days=10))
    return rt.intake


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
def backfill(rt: Runtime, intake: IntakeService, sleeps: Sleeps) -> BackfillService:
    rt.backfill = BackfillService(rt, sleep=sleeps)
    return rt.backfill


async def seed_source(rt: Runtime, user_gw: FakeUserGateway, info: ChatInfo) -> None:
    user_gw.add_chat(info)
    await rt.store.upsert_chat(info)


def history_with(
    user_gw: FakeUserGateway,
    chat_id: int,
    between: Callable[[IncomingMessage], Any],
    *,
    fail_after: int | None = None,
) -> None:
    """Replace the fake history of one chat: after each message ``await between(msg)`` (a
    Telegram page fetch, during which the live loops run); optionally raise ``ChatGone``
    after ``fail_after`` messages."""
    original = user_gw.history

    def history(
        cid: int, *, since: datetime, limit: int | None = None
    ) -> AsyncIterator[IncomingMessage]:
        inner = original(cid, since=since, limit=limit)
        if cid != chat_id:
            return inner

        async def gen() -> AsyncIterator[IncomingMessage]:
            n = 0
            async for msg in inner:
                if fail_after is not None and n >= fail_after:
                    raise ChatGone("kicked")
                yield msg
                n += 1
                await between(msg)
            if fail_after is not None:
                raise ChatGone("kicked")

        return gen()

    user_gw.history = history  # type: ignore[method-assign]


# --- live-tick-steals-backfill-units -----------------------------------------------------------


async def test_live_tick_leaves_units_a_backfill_is_filling(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    day_ago = START - timedelta(days=1)
    msgs = [
        make_message(group, text=ROOT_TEXT, sender_id=500, date=day_ago),
        make_message(group, text=REPLY_TEXT, sender_id=500, date=day_ago + timedelta(minutes=1)),
    ]
    assert await intake.collect(stream(msgs), via="backfill") == []
    await intake.tick()
    assert fake_sorter.submitted == []
    assert [r.closed for r in await group_rows(rt.store, group.id)] == [False, False]

    found = await intake.close_units(group.id, msgs[-1].date, force=True)
    assert [(c.via, c.message_ids) for c in found] == [("backfill", [1, 2])]
    # released: the chat's later live units close on the live tick again
    await intake.tick()
    assert fake_sorter.submitted == []


async def test_backfill_with_a_concurrent_tick_keeps_the_unit_whole_and_backfilled(
    rt: Runtime,
    backfill: BackfillService,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await seed_source(rt, user_gw, group)
    day_ago = START - timedelta(days=1)
    user_gw.seed(make_message(group, text=ROOT_TEXT, sender_id=500, date=day_ago))
    user_gw.seed(make_message(group, text=REPLY_TEXT, sender_id=501, reply_to_id=1, date=day_ago))

    async def page(msg: IncomingMessage) -> None:
        clock.advance(timedelta(seconds=10))
        await intake.tick()  # the intake loop runs while the next page is fetched

    history_with(user_gw, group.id, page)
    result = await backfill.run(days=3)
    assert [(c.via, c.message_ids) for c in fake_sorter.submitted] == [("backfill", [1, 2])]
    assert result.candidates == 1 and result.submitted == 1


async def test_a_chat_failing_mid_read_still_closes_its_units_as_backfill(
    rt: Runtime,
    backfill: BackfillService,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await seed_source(rt, user_gw, group)
    user_gw.seed(make_message(group, text=LONG, date=START - timedelta(hours=5)))
    user_gw.seed(make_message(group, text=LONG, date=START - timedelta(hours=1)))

    async def nothing(msg: IncomingMessage) -> None:
        return None

    history_with(user_gw, group.id, nothing, fail_after=1)
    result = await backfill.run(days=1)
    assert result.skipped_chats == [group.id]
    assert [(c.via, c.message_ids) for c in fake_sorter.submitted] == [("backfill", [1])]
    assert [r.closed for r in await group_rows(rt.store, group.id)] == [True]
    await intake.tick()
    assert len(fake_sorter.submitted) == 1


async def test_purge_waits_while_a_backfill_is_filling(
    rt: Runtime,
    intake: IntakeService,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    old = make_message(group, text=LONG, date=START - timedelta(days=5))
    await intake.collect(stream([old]), via="backfill")
    await intake.tick()
    assert len(await group_rows(rt.store, group.id)) == 1
    found = await intake.close_units(group.id, old.date, force=True)
    assert [c.message_ids for c in found] == [[1]]


async def test_a_unit_closed_long_after_its_root_is_never_live(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    """Rows a crashed backfill (or the time before downtime) left open are not real time."""
    group = make_chat(kind="group")
    await rt.store.upsert_chat(group)
    await rt.store.add_group_message(group.id, 1, date=START - timedelta(hours=3), text=LONG)
    await rt.store.add_group_message(group.id, 2, date=START - timedelta(minutes=6), text=LONG)
    await intake.tick()
    assert sorted((c.message_id, c.via) for c in fake_sorter.submitted) == [
        (1, "backfill"),
        (2, "live"),
    ]


# --- backfill-album-keyerror -------------------------------------------------------------------


async def test_tick_mid_backfill_stream_never_touches_its_albums(
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    chat = make_chat()
    old = START - timedelta(days=1)
    parts = [
        make_message(chat, text="", media="photo", grouped_id=9, date=old),
        make_message(chat, text="caption", media="photo", grouped_id=9, date=old),
    ]
    later = make_message(chat, text="later post", date=old + timedelta(hours=1))

    async def messages() -> AsyncIterator[IncomingMessage]:
        for part in parts:
            yield part
        clock.advance(timedelta(seconds=3))
        await intake.tick()
        yield later

    found = await intake.collect(messages(), via="backfill")
    assert sorted((c.message_id, c.message_ids, c.via) for c in found) == [
        (2, [1, 2], "backfill"),
        (3, [3], "backfill"),
    ]
    assert fake_sorter.submitted == []


async def test_flood_wait_after_an_album_does_not_break_the_backfill(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    chat = make_chat()
    await seed_source(rt, user_gw, chat)
    old = START - timedelta(hours=5)
    user_gw.seed(make_message(chat, text="", media="photo", grouped_id=9, date=old))
    user_gw.seed(make_message(chat, text="caption", media="photo", grouped_id=9, date=old))
    user_gw.seed(make_message(chat, text="later", date=old + timedelta(hours=1)))
    flooded = {"done": False}

    async def page(msg: IncomingMessage) -> None:
        if msg.message_id == 2 and not flooded["done"]:
            flooded["done"] = True
            raise FloodWait(30)

    async def sleep(seconds: float) -> None:
        clock.advance(timedelta(seconds=seconds))
        await intake.tick()  # the live loop runs while the backfill sleeps

    history_with(user_gw, chat.id, page)
    rt.backfill = BackfillService(rt, sleep=sleep)
    result = await rt.backfill.run(days=1)
    assert sorted((c.message_id, c.message_ids, c.via) for c in fake_sorter.submitted) == [
        (2, [1, 2], "backfill"),
        (3, [3], "backfill"),
    ]
    assert result.submitted == 2
    # a one-day run is not the full read /preview relies on (S11)
    assert await rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT) is None


# --- group-double-count-after-purge ------------------------------------------------------------


async def test_backfill_longer_than_the_buffer_retention_never_double_counts(
    rt: Runtime,
    backfill: BackfillService,
    intake: IntakeService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await seed_source(rt, user_gw, group)
    when = START - timedelta(days=5)
    user_gw.seed(make_message(group, text="old message", date=when))
    await backfill.run(days=7)
    assert await daily(rt.store, group.id) == {when.date(): 1}
    for _ in range(2):
        await rt.store.purge_group_messages(START - timedelta(days=3))
        await backfill.run(days=7)
        assert await daily(rt.store, group.id) == {when.date(): 1}


async def test_live_counted_group_rows_purged_then_backfilled_count_once(
    rt: Runtime,
    backfill: BackfillService,
    intake: IntakeService,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await seed_source(rt, user_gw, group)
    when = START - timedelta(days=5)
    clock.set(when)
    msg = make_message(group, text="live message", date=when)
    user_gw.seed(msg)
    rt.intake = live = IntakeService(rt)
    await live.handle_message(msg)
    assert await daily(rt.store, group.id) == {when.date(): 1}
    clock.set(START)
    await rt.store.purge_group_messages(START - timedelta(days=3))
    rt.backfill = BackfillService(rt, sleep=Sleeps())
    await rt.backfill.run(days=7)
    assert await daily(rt.store, group.id) == {when.date(): 1}


# --- submit-failure-drops-closed-units ---------------------------------------------------------


async def test_a_failing_submit_in_a_tick_does_not_lose_the_other_units(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    groups = [make_chat(kind="group") for _ in range(3)]
    for group in groups:
        await intake.handle_message(make_message(group, text=LONG, date=START))
    fake_sorter.fail = {(groups[1].id, 1)}
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    assert sorted(c.chat_id for c in fake_sorter.submitted) == sorted([groups[0].id, groups[2].id])
    assert [r.closed for r in await group_rows(rt.store, groups[1].id)] == [False]

    fake_sorter.fail = set()
    await intake.tick()
    assert sorted(c.chat_id for c in fake_sorter.submitted) == sorted(g.id for g in groups)
    assert fake_sorter.submitted[-1].via == "live"
    assert [r.closed for r in await group_rows(rt.store, groups[1].id)] == [True]


async def test_a_failing_post_is_retried_and_then_given_up(
    intake: IntakeService,
    fake_sorter: FailingSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    chat = make_chat()
    fake_sorter.fail = {(chat.id, 1)}
    await intake.handle_message(make_message(chat, text="first"))
    await intake.handle_message(make_message(chat, text="second"))
    assert [c.message_id for c in fake_sorter.submitted] == [2]
    for _ in range(SUBMIT_ATTEMPTS + 2):
        await intake.tick()
    assert fake_sorter.attempts.count((chat.id, 1)) == SUBMIT_ATTEMPTS


async def test_a_failing_submit_in_a_backfill_does_not_stop_the_rest(
    rt: Runtime,
    backfill: BackfillService,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel, group = make_chat(), make_chat(kind="group")
    await seed_source(rt, user_gw, channel)
    await seed_source(rt, user_gw, group)
    user_gw.seed(make_message(channel, text="one", date=START - timedelta(hours=3)))
    user_gw.seed(make_message(group, text=LONG, date=START - timedelta(hours=2)))
    user_gw.seed(make_message(channel, text="three", date=START - timedelta(hours=1)))
    fake_sorter.fail = {(group.id, 1)}

    result = await backfill.run(days=1)
    assert [c.text for c in fake_sorter.submitted] == ["one", "three"]
    assert result.submitted == 2
    # a one-day run is not the full read /preview relies on (S11)
    assert await rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT) is None
    assert [r.closed for r in await group_rows(rt.store, group.id)] == [False]

    fake_sorter.fail = set()
    await intake.tick()  # the reopened unit is closed again, and still as a backfill
    assert [(c.chat_id, c.via) for c in fake_sorter.submitted[2:]] == [(group.id, "backfill")]


# --- close-units-race-orphans-member -----------------------------------------------------------


async def test_a_reply_arriving_while_its_unit_closes_is_never_orphaned(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FailingSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = make_chat(kind="group")
    await intake.handle_message(make_message(group, text=ROOT_TEXT, sender_id=500, date=START))
    reply = make_message(
        group,
        text=FOLLOW_TEXT,
        sender_id=501,
        reply_to_id=1,
        date=START + timedelta(minutes=5) - timedelta(milliseconds=200),
    )
    clock.set(START + timedelta(minutes=5))

    closing, delivered = asyncio.Event(), asyncio.Event()
    original = rt.store.get_chat
    calls = {"n": 0}

    async def slow_get_chat(chat_id: int, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:  # the close's own lookup, between reading and marking
            closing.set()
            # let the reply through if nothing holds it back (the unfixed race)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(delivered.wait(), timeout=0.3)
        return await original(chat_id, *args, **kwargs)

    monkeypatch.setattr(rt.store, "get_chat", slow_get_chat)

    async def deliver() -> None:
        await closing.wait()
        await intake.handle_message(reply)
        delivered.set()

    await asyncio.gather(intake.tick(), deliver())
    rows = {r.message_id: r for r in await group_rows(rt.store, group.id)}
    in_unit = [c.message_ids for c in fake_sorter.submitted if c.message_id == 1]
    # the reply either made the close, or (the unit closed below the floor) reopened its
    # thread (CG-2): both rows open again under root 1, to be closed together
    reopened = rows[2].unit_root_id == 1 and not rows[1].closed and not rows[2].closed
    assert (rows[1].closed and in_unit == [[1, 2]]) or reopened


# --- resort-self-heal-publishes-old / self-heal-revives-stale-orphans --------------------------


@pytest.fixture
def publisher(rt: Runtime) -> StubPublisher:
    pub = StubPublisher(rt.store)
    rt.publisher = pub  # type: ignore[assignment]
    return pub


@pytest.fixture
async def chats(rt: Runtime, make_chat: Callable[..., ChatInfo]) -> Sequence[int]:
    ids = []
    for _ in range(3):
        info = make_chat()
        await rt.store.upsert_chat(info)
        ids.append(info.id)
    return ids


@pytest.fixture
def real_sorter(rt: Runtime, publisher: StubPublisher) -> Sorter:
    return Sorter(rt)


async def test_resorted_live_posts_never_reach_the_outbox(
    rt: Runtime,
    real_sorter: Sorter,
    chats: Sequence[int],
    publisher: StubPublisher,
    clock: FakeClock,
) -> None:
    await trust(rt, chats[0], 3.0)
    await trust(rt, chats[1], 3.0)
    post = await real_sorter.submit(cand(chats[0], 1, A))
    assert post is not None and post.status == PostStatus.unsorted
    clock.advance(timedelta(hours=10))
    topic = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=clock.now())
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {topic.id: 0.9}
    assert await real_sorter.resort_unsorted(clock.now() - timedelta(days=7)) == 1
    got = await fresh(rt.store, post)
    assert got.status == PostStatus.digest and got.would_realtime is True

    await real_sorter.tick()
    assert publisher.enqueued == []
    assert (await fresh(rt.store, post)).status == PostStatus.digest

    dup = await real_sorter.submit(cand(chats[1], 1, A_FOOTER))
    assert dup is not None and dup.duplicate_of == post.id
    assert publisher.enqueued == []
    assert (await fresh(rt.store, post)).corroboration == 1


async def test_self_healing_sends_a_stale_queued_orphan_to_the_digest(
    rt: Runtime,
    real_sorter: Sorter,
    chats: Sequence[int],
    publisher: StubPublisher,
) -> None:
    topic = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=rt.clock.now())
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {topic.id: 0.9}
    await trust(rt, chats[0], 3.0)
    publisher.refuse = True  # the crash between the commit and the enqueue
    old = await real_sorter.submit(cand(chats[0], 1, A, posted_at=START - timedelta(hours=5)))
    new = await real_sorter.submit(
        cand(chats[0], 2, "A different story entirely about trains and metro lines.")
    )
    assert old is not None and new is not None
    await rt.store.set_post_fields(old.id, status=PostStatus.queued, would_realtime=True)
    assert (await fresh(rt.store, new)).status == PostStatus.queued
    publisher.refuse = False
    publisher.enqueued.clear()

    await real_sorter.tick()
    assert publisher.enqueued == [new.id]
    assert (await fresh(rt.store, old)).status == PostStatus.digest
    assert await rt.store.get_publication(old.id) is None
    await real_sorter.tick()
    assert publisher.enqueued == [new.id]
