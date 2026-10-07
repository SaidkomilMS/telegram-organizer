"""Intake (DESIGN §9.1): filtering, counting, albums, conversation units, chat sync."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from tests.fakes import START, FakeClock, FakeUserGateway
from tg_curator.clock import local_date
from tg_curator.db import schema
from tg_curator.db.store import Store
from tg_curator.domain import KV, Candidate, NewPost, Post
from tg_curator.pipeline.intake import IntakeService
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage
from tg_curator.textutil import text_hash

DAY = START.date()
MORNING = datetime(2026, 10, 5, 10, 0, tzinfo=UTC)

ROOT_TEXT = ("The root message of a substantive thread about something. " * 5).strip()
REPLY_TEXT = ("A reply that adds detail and argument to the thread. " * 4).strip()
FOLLOW_TEXT = ("And a follow-up by the same person with more of the same. " * 3).strip()


class FakeSorter:
    """Records submissions and does the sorter's half of the counting rule (§9.3): the post
    row is inserted and ``chat_daily`` bumped only when the row is new."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.submitted: list[Candidate] = []

    async def submit(self, c: Candidate) -> Post | None:
        self.submitted.append(c)
        new = NewPost(
            chat_id=c.chat_id,
            message_id=c.message_id,
            kind=c.kind,
            message_ids=list(c.message_ids),
            grouped_id=c.grouped_id,
            posted_at=c.posted_at,
            via=c.via,
            text=c.text,
            html=c.html,
            text_hash=text_hash(c.text),
            urls=list(c.urls),
            media=c.media,
            noforwards=c.noforwards,
            fwd_from_chat_id=c.fwd_from_chat_id,
            fwd_from_message_id=c.fwd_from_message_id,
            views=c.views,
            forwards=c.forwards,
        )
        async with self.store.begin() as conn:
            post = await self.store.insert_post(new, conn=conn)
            if post is not None and c.kind == "post":  # group units were counted by intake
                await self.store.bump_chat_daily(
                    c.chat_id, local_date(c.posted_at, "UTC"), conn=conn
                )
                await self.store.touch_chat(c.chat_id, c.posted_at, conn=conn)
        return post

    async def tick(self) -> None:
        return None

    async def resort_unsorted(self, since: datetime) -> int:
        return 0


async def daily(store: Store, chat_id: int) -> dict[date, int]:
    rows = await store.execute(
        sa.select(schema.chat_daily).where(schema.chat_daily.c.chat_id == chat_id)
    )
    return {r.day: r.messages for r in rows}


async def group_rows(store: Store, chat_id: int) -> list[Any]:
    return await store.execute(
        sa.select(schema.group_messages)
        .where(schema.group_messages.c.chat_id == chat_id)
        .order_by(schema.group_messages.c.message_id)
    )


async def post_count(store: Store, chat_id: int) -> int:
    rows = await store.execute(
        sa.select(sa.func.count())
        .select_from(schema.posts)
        .where(schema.posts.c.chat_id == chat_id)
    )
    return int(rows[0][0])


async def stream(messages: Sequence[IncomingMessage]) -> AsyncIterator[IncomingMessage]:
    for msg in messages:
        yield msg


@pytest.fixture
def sorter(rt: Runtime) -> FakeSorter:
    rt.sorter = FakeSorter(rt.store)
    return rt.sorter


@pytest.fixture
def intake(rt: Runtime, sorter: FakeSorter) -> IntakeService:
    rt.intake = IntakeService(rt)
    return rt.intake


@pytest.fixture
def morning_intake(rt: Runtime, sorter: FakeSorter, clock: FakeClock) -> IntakeService:
    """An intake whose process started at 10:00, so the §9.1 example times are not late."""
    clock.set(MORNING)
    rt.intake = IntakeService(rt)
    return rt.intake


# --- channels ----------------------------------------------------------------------------------


async def test_channel_post_is_a_candidate_and_counted_once(
    rt: Runtime, intake: IntakeService, sorter: FakeSorter, make_message: Callable[..., Any]
) -> None:
    msg = make_message(text="A post with enough words to be a post.", urls=("https://a.example/x",))
    await intake.handle_message(msg)

    assert len(sorter.submitted) == 1
    c = sorter.submitted[0]
    assert (c.kind, c.message_id, c.message_ids, c.via) == ("post", 1, [1], "live")
    assert c.urls == ["https://a.example/x"] and c.media is None and c.grouped_id is None
    chat = await rt.store.get_chat(msg.chat.id)
    assert chat is not None and chat.role == "source" and chat.active
    assert await daily(rt.store, msg.chat.id) == {DAY: 1}
    assert await rt.store.kv_get(KV.INTAKE_LAST_MESSAGE_AT) == msg.date.isoformat()

    await intake.handle_message(msg)  # re-delivery: submitted again, counted never twice
    assert len(sorter.submitted) == 2
    assert await daily(rt.store, msg.chat.id) == {DAY: 1}
    assert await post_count(rt.store, msg.chat.id) == 1


async def test_post_without_text_is_still_submitted(
    intake: IntakeService, sorter: FakeSorter, make_message: Callable[..., Any]
) -> None:
    await intake.handle_message(make_message(text="", media="photo"))
    assert [c.text for c in sorter.submitted] == [""]


async def test_album_becomes_one_candidate_with_the_caption_carrier(
    rt: Runtime,
    intake: IntakeService,
    sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    chat = make_chat()
    parts = [
        make_message(chat, text="", media="photo", grouped_id=42, urls=("https://a.example/1",)),
        make_message(chat, text="The caption of the album", media="photo", grouped_id=42),
        make_message(chat, text="", media="photo", grouped_id=42, urls=("https://a.example/2",)),
    ]
    for part in parts:
        await intake.handle_message(part)
    assert sorter.submitted == []

    clock.advance(1)
    await intake.tick()
    assert sorter.submitted == []  # not settled yet

    clock.advance(2)
    await intake.tick()
    assert len(sorter.submitted) == 1
    c = sorter.submitted[0]
    assert c.message_id == 2 and c.message_ids == [1, 2, 3]
    assert c.media == "album" and c.grouped_id == 42 and c.text == "The caption of the album"
    assert c.urls == ["https://a.example/1", "https://a.example/2"]
    assert await daily(rt.store, chat.id) == {DAY: 1}

    await intake.tick()  # the buffer is empty now
    assert len(sorter.submitted) == 1


async def test_album_without_caption_uses_the_first_part(
    intake: IntakeService,
    sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    chat = make_chat()
    await intake.handle_message(make_message(chat, text="", media="photo", grouped_id=7))
    await intake.handle_message(make_message(chat, text="", media="video", grouped_id=7))
    clock.advance(3)
    await intake.tick()
    assert [(c.message_id, c.message_ids) for c in sorter.submitted] == [(1, [1, 2])]


# --- what is dropped ---------------------------------------------------------------------------


async def test_service_outgoing_non_source_and_late_messages_are_dropped(
    rt: Runtime,
    intake: IntakeService,
    sorter: FakeSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    output = make_chat(title="Topic channel")
    await rt.store.upsert_chat(output, role="output")
    group = make_chat(kind="group")
    dropped = [
        make_message(is_service=True),
        make_message(output),
        make_message(date=START - timedelta(minutes=11)),
        make_message(group, is_service=True),
        make_message(group, is_outgoing=True, sender_id=1001),
        make_message(group, date=START - timedelta(minutes=11)),
    ]
    for msg in dropped:
        await intake.handle_message(msg)

    assert sorter.submitted == []
    for msg in dropped:
        assert await daily(rt.store, msg.chat.id) == {}
    assert await group_rows(rt.store, group.id) == []
    assert await rt.store.kv_get(KV.INTAKE_LAST_MESSAGE_AT) is None

    # just inside the window is fine
    await intake.handle_message(make_message(date=START - timedelta(minutes=9)))
    assert len(sorter.submitted) == 1

    # the account's own post in a channel is a candidate like any other (only the user's own
    # messages in groups are ignored)
    own = make_message(text="my own channel post", is_outgoing=True)
    await intake.handle_message(own)
    assert len(sorter.submitted) == 2
    assert sorter.submitted[-1].message_id == own.message_id
    assert sum((await daily(rt.store, own.chat.id)).values()) >= 1


async def test_backfill_stream_is_not_subject_to_the_late_guard(
    intake: IntakeService, make_message: Callable[..., Any]
) -> None:
    old = make_message(date=START - timedelta(days=2))
    found = await intake.collect(stream([old]), via="backfill")
    assert [c.via for c in found] == ["backfill"]


async def test_left_chat_becomes_active_again_on_a_message(
    rt: Runtime, intake: IntakeService, make_message: Callable[..., Any]
) -> None:
    msg = make_message()
    await rt.store.upsert_chat(msg.chat)
    await rt.store.set_chat_fields(msg.chat.id, active=False, left_at=START)
    await intake.handle_message(msg)
    chat = await rt.store.get_chat(msg.chat.id)
    assert chat is not None and chat.active and chat.left_at is None


# --- groups ------------------------------------------------------------------------------------


async def test_group_example_closes_at_10_08_with_three_members(
    rt: Runtime,
    morning_intake: IntakeService,
    sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    intake = morning_intake
    group = make_chat(kind="group")
    at = {m: MORNING + timedelta(minutes=m) for m in (0, 2, 3, 7, 8, 20, 25)}

    clock.set(at[0])
    await intake.handle_message(make_message(group, text=ROOT_TEXT, sender_id=500, date=at[0]))
    clock.set(at[2])
    reply = make_message(group, text=REPLY_TEXT, sender_id=501, reply_to_id=1, date=at[2])
    await intake.handle_message(reply)
    clock.set(at[3])
    await intake.handle_message(make_message(group, text=FOLLOW_TEXT, sender_id=501, date=at[3]))

    rows = await group_rows(rt.store, group.id)
    assert [(r.message_id, r.unit_root_id, r.closed) for r in rows] == [
        (1, 1, False),
        (2, 1, False),
        (3, 1, False),
    ]
    assert await daily(rt.store, group.id) == {DAY: 3}
    await intake.handle_message(reply)  # re-delivery
    assert await daily(rt.store, group.id) == {DAY: 3}
    assert sorter.submitted == []  # units are never submitted from collect

    assert await intake.close_units(None, at[7]) == []
    assert all(not r.closed for r in await group_rows(rt.store, group.id))

    clock.set(at[8])
    await intake.tick()
    assert len(sorter.submitted) == 1
    unit = sorter.submitted[0]
    assert unit.kind == "unit" and unit.message_id == 1 and unit.message_ids == [1, 2, 3]
    assert unit.posted_at == at[0] and unit.via == "live" and unit.chat_id == group.id
    assert unit.text == "\n\n".join([ROOT_TEXT, REPLY_TEXT, FOLLOW_TEXT])
    assert all(r.closed and r.unit_root_id == 1 for r in await group_rows(rt.store, group.id))
    assert await daily(rt.store, group.id) == {DAY: 3}  # the unit itself is not counted again

    # a reply to the closed unit starts its own (sub-floor) unit
    clock.set(at[20])
    late_reply = make_message(group, text="Late reply.", sender_id=502, reply_to_id=1, date=at[20])
    await intake.handle_message(late_reply)
    row = (await group_rows(rt.store, group.id))[-1]
    assert (row.message_id, row.unit_root_id, row.closed) == (4, 4, False)

    assert await intake.close_units(group.id, at[25]) == []
    row = (await group_rows(rt.store, group.id))[-1]
    assert (row.message_id, row.unit_root_id, row.closed) == (4, 4, True)
    assert len(sorter.submitted) == 1
    assert await daily(rt.store, group.id) == {DAY: 4}


async def test_unit_below_the_floor_is_closed_and_not_submitted(
    rt: Runtime,
    morning_intake: IntakeService,
    sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await morning_intake.handle_message(make_message(group, text="short", date=MORNING))
    clock.set(MORNING + timedelta(minutes=6))
    await morning_intake.tick()
    assert sorter.submitted == []
    rows = await group_rows(rt.store, group.id)
    assert [(r.unit_root_id, r.closed) for r in rows] == [(1, True)]
    assert await daily(rt.store, group.id) == {DAY: 1}


async def test_min_chars_setting_is_the_floor(
    rt: Runtime,
    morning_intake: IntakeService,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    await rt.settings_file.set_value("groups.min_chars", 5)
    group = make_chat(kind="group")
    await morning_intake.handle_message(make_message(group, text="short", date=MORNING))
    found = await morning_intake.close_units(group.id, MORNING + timedelta(minutes=5))
    assert [c.text for c in found] == ["short"]


async def test_same_sender_run_and_the_30_minute_cap(
    rt: Runtime,
    morning_intake: IntakeService,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    # every 4 minutes from 10:00: joins while under the cap, 10:32 is a new root
    for minutes in range(0, 33, 4):
        at = MORNING + timedelta(minutes=minutes)
        clock.set(at)
        await morning_intake.handle_message(make_message(group, text="x", sender_id=500, date=at))
    roots = [r.unit_root_id for r in await group_rows(rt.store, group.id)]
    assert roots == [1] * 8 + [9]

    # a gap of more than unit_gap_minutes between same-sender messages starts a new root
    at = MORNING + timedelta(minutes=38)
    clock.set(at)
    await morning_intake.handle_message(make_message(group, text="x", sender_id=500, date=at))
    roots = [r.unit_root_id for r in await group_rows(rt.store, group.id)]
    assert roots[-2:] == [9, 10]

    # a different sender without a reply is its own root
    at = MORNING + timedelta(minutes=39)
    clock.set(at)
    await morning_intake.handle_message(make_message(group, text="x", sender_id=501, date=at))
    roots = [r.unit_root_id for r in await group_rows(rt.store, group.id)]
    assert roots[-1] == 11


async def test_reply_joins_the_thread_of_a_different_sender(
    rt: Runtime,
    morning_intake: IntakeService,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    clock.set(MORNING)
    await morning_intake.handle_message(make_message(group, text="root", sender_id=500))
    clock.set(MORNING + timedelta(minutes=4))
    await morning_intake.handle_message(
        make_message(group, text="r1", sender_id=501, reply_to_id=1)
    )
    clock.set(MORNING + timedelta(minutes=8))
    await morning_intake.handle_message(
        make_message(group, text="r2", sender_id=502, reply_to_id=2)
    )
    roots = [r.unit_root_id for r in await group_rows(rt.store, group.id)]
    assert roots == [1, 1, 1]


async def test_unit_candidate_joins_html_and_unites_urls(
    rt: Runtime,
    morning_intake: IntakeService,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    await rt.settings_file.set_value("groups.min_chars", 1)
    group = make_chat(kind="group", noforwards=True)
    clock.set(MORNING)
    await morning_intake.handle_message(
        make_message(
            group,
            text="See https://a.example/1",
            html='See <a href="https://a.example/1">link</a>',
            urls=("https://a.example/1",),
            media="file",
        )
    )
    clock.set(MORNING + timedelta(minutes=1))
    await morning_intake.handle_message(
        make_message(group, text="And <this>", urls=("https://a.example/1", "https://b.example/"))
    )
    found = await morning_intake.close_units(group.id, MORNING + timedelta(minutes=6))
    assert len(found) == 1
    c = found[0]
    assert c.html == 'See <a href="https://a.example/1">link</a>\n\nAnd &lt;this&gt;'
    assert c.urls == ["https://a.example/1", "https://b.example/"]
    assert c.media == "file" and c.noforwards and c.views is None


async def test_forced_close_returns_backfill_units_without_waiting(
    rt: Runtime,
    intake: IntakeService,
    sorter: FakeSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    msgs = [
        make_message(group, text=ROOT_TEXT, sender_id=500, date=START - timedelta(days=1)),
        make_message(
            group, text=REPLY_TEXT, sender_id=500, date=START - timedelta(days=1, minutes=-1)
        ),
    ]
    assert await intake.collect(stream(msgs), via="backfill") == []
    assert sorter.submitted == []
    found = await intake.close_units(group.id, msgs[-1].date, force=True)
    assert [(c.kind, c.via, c.message_ids) for c in found] == [("unit", "backfill", [1, 2])]
    assert await daily(rt.store, group.id) == {DAY - timedelta(days=1): 2}


async def test_a_channels_automatic_forward_into_its_discussion_group_is_not_a_unit(
    rt: Runtime,
    intake: IntakeService,
    sorter: FakeSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    """The copy Telegram puts into a channel's comment section is neither counted in the
    group's volume nor a unit root; the comments are judged on their own text."""
    import dataclasses

    group = make_chat(kind="group")
    when = START - timedelta(days=1)
    auto = dataclasses.replace(
        make_message(
            group,
            text=ROOT_TEXT,
            sender_id=-1_001_000_000_777,
            date=when,
            fwd_from_chat_id=-1_001_000_000_777,
            fwd_from_message_id=5,
        ),
        is_automatic_forward=True,
    )
    comment = make_message(
        group, text=f"{ROOT_TEXT} {REPLY_TEXT}", sender_id=500, reply_to_id=auto.message_id,
        date=when + timedelta(minutes=1),
    )  # fmt: skip
    assert await intake.collect(stream([auto, comment]), via="backfill") == []
    found = await intake.close_units(group.id, comment.date, force=True)
    assert all(auto.message_id not in c.message_ids for c in found)
    assert all(c.fwd_from_chat_id is None for c in found)
    assert [c.message_ids for c in found] == [[comment.message_id]]
    assert [r.message_id for r in await group_rows(rt.store, group.id)] == [comment.message_id]
    assert await daily(rt.store, group.id) == {DAY - timedelta(days=1): 1}


async def test_backfill_collect_flushes_albums_at_the_end_of_the_stream(
    intake: IntakeService, make_chat: Callable[..., ChatInfo], make_message: Callable[..., Any]
) -> None:
    chat = make_chat()
    msgs = [
        make_message(chat, text="first"),
        make_message(chat, text="", media="photo", grouped_id=5),
        make_message(chat, text="album caption", media="photo", grouped_id=5),
        make_message(chat, text="last"),
    ]
    found = await intake.collect(stream(msgs), via="backfill")
    assert sorted((c.message_id, c.text, c.via) for c in found) == [
        (1, "first", "backfill"),
        (3, "album caption", "backfill"),
        (4, "last", "backfill"),
    ]


async def test_tick_purges_group_messages_older_than_three_days(
    rt: Runtime, intake: IntakeService, make_chat: Callable[..., ChatInfo]
) -> None:
    group = make_chat(kind="group")
    await rt.store.upsert_chat(group)
    await rt.store.add_group_message(
        group.id, 1, date=START - timedelta(days=4), text="old", closed=True, unit_root_id=1
    )
    await rt.store.add_group_message(
        group.id, 2, date=START - timedelta(days=2), text="recent", closed=True, unit_root_id=2
    )
    await intake.tick()
    assert [r.message_id for r in await group_rows(rt.store, group.id)] == [2]


# --- chat sync ---------------------------------------------------------------------------------


async def test_sync_chats_marks_a_vanished_chat_left_and_keeps_its_row(
    rt: Runtime,
    intake: IntakeService,
    user_gw: FakeUserGateway,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
) -> None:
    channel = make_chat(title="News", username="news")
    group = make_chat(kind="group", title="Chatter", archived=True)
    output = make_chat(title="ML & AI", is_creator=True)
    await rt.store.upsert_chat(output, role="output")
    for info in (channel, group, output):
        user_gw.add_chat(info)

    first = await intake.sync_chats()
    assert first.total == 3 and first.outputs == 1 and first.left == []
    assert sorted(first.new) == sorted([channel.id, group.id])
    stored = await rt.store.get_chat(group.id)
    assert stored is not None and stored.role == "source" and stored.archived

    user_gw.chats = [c for c in user_gw.chats if c.id != group.id]
    clock.advance(1800)
    second = await intake.sync_chats()
    assert second.total == 2 and second.new == [] and second.left == [group.id]
    gone = await rt.store.get_chat(group.id)
    assert gone is not None and not gone.active and gone.left_at == clock.now()
    assert gone.title == "Chatter" and gone.first_seen_at == START
    assert [c.id for c in await rt.store.list_chats(role="source", active=True)] == [channel.id]

    # a message from a left chat is not a source message any more
    user_gw.add_chat(group)
    third = await intake.sync_chats()
    assert third.new == [group.id] and third.left == []
    back = await rt.store.get_chat(group.id)
    assert back is not None and back.active and back.left_at is None

    # a renamed chat is refreshed; an output channel that vanished is never marked left
    user_gw.add_chat(ChatInfo(**{**channel.__dict__, "title": "News (renamed)"}))
    user_gw.chats = [c for c in user_gw.chats if c.id != output.id]
    fourth = await intake.sync_chats()
    assert fourth.left == [] and fourth.outputs == 0
    renamed = await rt.store.get_chat(channel.id)
    assert renamed is not None and renamed.title == "News (renamed)"
    kept = await rt.store.get_chat(output.id)
    assert kept is not None and kept.active and kept.role == "output"
