"""Regression tests for the publisher review findings (DESIGN §9.4, §14.1, §14.6).

Each test pins one race or crash window the review reproduced: a move or a reconcile that
interleaves with a send or a ``+N`` edit, a multi-message post that fails or crashes halfway,
a stub Telegram refused, and the bookkeeping a reconcile leaves behind.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from tests.fakes import FakeBotGateway, FakeClock, FakeUserGateway, FakeWorld
from tests.test_publisher import CHANNEL_A, CHANNEL_B, LONG_TEXT, Harness, wrong_topic_button
from tg_curator.domain import (
    PUB_CANCELLED,
    PUB_FAILED,
    PUB_PENDING,
    PUB_RETRACTED,
    PUB_SENDING,
    PUB_SENT,
    PostStatus,
)
from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    CuratorError,
    FloodWait,
    ForwardsRestricted,
    NotAllowed,
    TelegramUnavailable,
)
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

SPLIT_TEXT = "\n\n".join(f"paragraph {i} " + "y" * 300 for i in range(30))  # three messages


class Crash(BaseException):
    """The process dying: nothing in the publisher may catch it."""


@pytest.fixture
async def h(
    rt: Runtime,
    clock: FakeClock,
    world: FakeWorld,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> Harness:
    harness = Harness(rt, clock, world, user_gw, bot_gw, make_chat)
    await harness.setup()
    return harness


def flaky_send_text(
    h: Harness, monkeypatch: pytest.MonkeyPatch, fail_on: int, exc: Exception, *, deliver: bool
) -> None:
    """Make the ``fail_on``-th ``send_text`` call raise ``exc``, after delivering the message
    when ``deliver`` (a timeout after Telegram accepted it)."""
    original = h.bot.send_text
    count = 0

    async def send_text(chat_id: int, html: str, **kw: Any) -> int:
        nonlocal count
        count += 1
        if count == fail_on:
            if deliver:
                await original(chat_id, html, **kw)
            raise exc
        return await original(chat_id, html, **kw)

    monkeypatch.setattr(h.bot, "send_text", send_text)


def texts(h: Harness, channel: int = CHANNEL_A) -> list[str]:
    return [m.text for m in h.sent(channel)]


# --- tick-move-race ----------------------------------------------------------------------------


async def test_retract_between_the_due_list_and_the_lock_is_not_overwritten(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post("unique race story")
    await h.pub.enqueue(post.id)
    due_rows = h.pub._due_rows

    async def due_then_move() -> Any:
        rows = await due_rows()
        await h.pub.move(post.id, None)  # the owner's "not for me" lands mid-tick
        return rows

    monkeypatch.setattr(h.pub, "_due_rows", due_then_move)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(post)
    assert row.state == PUB_RETRACTED
    assert (await h.fresh(post)).status == PostStatus.rejected
    assert h.sent(CHANNEL_A) == [] and h.sent(CHANNEL_B) == []


async def test_repoint_between_the_due_list_and_the_lock_goes_to_the_new_channel(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post("unique race story")
    await h.pub.enqueue(post.id)
    due_rows = h.pub._due_rows
    moved = False

    async def due_then_move() -> Any:
        nonlocal moved
        rows = await due_rows()
        if not moved:
            moved = True
            await h.pub.move(post.id, h.topic_b.id)
        return rows

    monkeypatch.setattr(h.pub, "_due_rows", due_then_move)
    for _ in range(2):
        h.clock.advance(5)
        await h.pub.tick()
    assert h.sent(CHANNEL_A) == []
    assert len(h.sent(CHANNEL_B)) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and row.topic_id == h.topic_b.id
    assert row.channel_id == CHANNEL_B and row.message_ids == [h.sent(CHANNEL_B)[0].message_id]


async def test_concurrent_tick_and_retract_never_leave_the_post_visible(h: Harness) -> None:
    post = await h.post("unique race story")
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await asyncio.gather(h.pub.tick(), h.pub.move(post.id, None))
    assert (await h.row(post)).state == PUB_RETRACTED
    assert (await h.fresh(post)).status == PostStatus.rejected
    assert not any("unique race story" in t for t in texts(h))


# --- edit-unstubs-moved-post -------------------------------------------------------------------


async def test_move_between_the_edit_select_and_the_lock_keeps_the_stub(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post("unique edit story")
    post = await h.publish(post)
    old = h.sent()[0]
    for k in range(3):
        await h.rt.store.add_corroboration(post.id, -7 - k)
    h.clock.advance(30)
    execute = h.rt.store.execute
    fired = False

    async def execute_then_move(stmt: Any, conn: Any = None) -> Any:
        nonlocal fired
        result = await execute(stmt, conn)
        if not fired:
            fired = True
            await h.pub.move(post.id, h.topic_b.id)
            # the stub writes alone would keep this channel's budget busy; take them out so
            # the edit really reaches the row the move just changed
            h.pub._channel_writes[CHANNEL_A].clear()
        return result

    monkeypatch.setattr(h.rt.store, "execute", execute_then_move)
    await h.pub._edit_corroboration()
    monkeypatch.setattr(h.rt.store, "execute", execute)
    msg = h.world.get(CHANNEL_A, old.message_id)
    assert msg.text == "↪ moved to Fintech" and msg.buttons is None
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.shown_corroboration == 0 and row.edited_at is None
    h.clock.advance(5)
    await h.pub.tick()
    assert not any("unique edit story" in t for t in texts(h, CHANNEL_A))
    assert sum("unique edit story" in t for t in texts(h, CHANNEL_B)) == 1


async def test_move_during_an_edit_in_flight_waits_and_stubs_after_it(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post("unique edit story")
    post = await h.publish(post)
    old = h.sent()[0]
    await h.rt.store.add_corroboration(post.id, -7)
    h.clock.advance(30)
    edit_text = h.bot.edit_text
    move_task: asyncio.Task[Any] | None = None

    async def slow_edit(chat_id: int, message_id: int, html: str, **kw: Any) -> None:
        nonlocal move_task
        if move_task is None:
            move_task = asyncio.create_task(h.pub.move(post.id, h.topic_b.id))
            await asyncio.wait({move_task}, timeout=0.3)
            assert not move_task.done()  # the move waits for the per-post lock
        await edit_text(chat_id, message_id, html, **kw)

    monkeypatch.setattr(h.bot, "edit_text", slow_edit)
    await h.pub.tick()
    assert move_task is not None
    await move_task
    msg = h.world.get(CHANNEL_A, old.message_id)
    assert msg.text == "↪ moved to Fintech" and msg.buttons is None


async def test_move_while_the_edit_reads_the_post_cannot_be_undone_by_it(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post("unique edit story")
    post = await h.publish(post)
    old = h.sent()[0]
    await h.rt.store.add_corroboration(post.id, -7)
    h.clock.advance(30)
    get_chat = h.rt.store.get_chat
    move_task: asyncio.Task[Any] | None = None

    async def get_chat_while_moving(chat_id: int, **kw: Any) -> Any:
        nonlocal move_task
        if move_task is None:
            move_task = asyncio.create_task(h.pub.move(post.id, h.topic_b.id))
            await asyncio.wait({move_task}, timeout=0.3)
        return await get_chat(chat_id, **kw)

    monkeypatch.setattr(h.rt.store, "get_chat", get_chat_while_moving)
    await h.pub._edit_corroboration()
    assert move_task is not None
    await move_task
    msg = h.world.get(CHANNEL_A, old.message_id)
    assert msg.text == "↪ moved to Fintech" and msg.buttons is None
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.channel_id == CHANNEL_B


# --- partial-send-duplicates -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "wait"), [(TelegramUnavailable("blip"), 31), (FloodWait(200), 200)]
)
async def test_failure_on_part_two_resumes_without_resending_part_one(
    h: Harness, monkeypatch: pytest.MonkeyPatch, exc: Exception, wait: int
) -> None:
    flaky_send_text(h, monkeypatch, 2, exc, deliver=False)
    post = await h.post(SPLIT_TEXT)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(post)
    first = h.sent()
    assert row.state == PUB_FAILED and len(first) == 1
    assert row.message_ids == [first[0].message_id]  # committed as soon as it went out
    h.clock.advance(wait)
    await h.pub.tick()
    msgs = h.sent()
    assert len(msgs) >= 3 and len({m.text for m in msgs}) == len(msgs)
    assert wrong_topic_button(msgs[-1]) and not any(wrong_topic_button(m) for m in msgs[:-1])
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [m.message_id for m in msgs]


async def test_part_delivered_before_a_timeout_is_found_not_resent(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky_send_text(h, monkeypatch, 2, TelegramUnavailable("timeout"), deliver=True)
    post = await h.post(SPLIT_TEXT)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_FAILED and len(h.sent()) == 2
    h.clock.advance(31)
    await h.pub.tick()
    msgs = h.sent()
    assert len({m.text for m in msgs}) == len(msgs)
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [m.message_id for m in msgs]


async def test_media_first_text_failure_does_not_copy_or_send_the_media_again(
    h: Harness,
) -> None:
    h.bot.fail_next("send_text", TelegramUnavailable("blip"))
    post = await h.post(LONG_TEXT, media="video")
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(post)
    media = h.sent()
    assert row.state == PUB_FAILED and len(media) == 1 and media[0].media == "video"
    assert row.message_ids == [media[0].message_id] and len(row.staging_ids) == 1
    h.clock.advance(31)
    await h.pub.tick()
    assert len(h.user.calls_of("copy_media")) == 1
    assert len(h.bot.calls_of("send_copy")) == 1
    msgs = h.sent()
    assert len(msgs) == 2 and msgs[1].media is None and wrong_topic_button(msgs[1])
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [m.message_id for m in msgs]


async def test_move_of_a_partly_sent_row_stubs_what_went_out(h: Harness) -> None:
    h.bot.fail_next("send_text", TelegramUnavailable("blip"))
    post = await h.post(LONG_TEXT, media="video")
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    orphan = h.sent()[0]
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and result.stubbed
    assert h.world.get(CHANNEL_A, orphan.message_id).text == "↪ moved"
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent(CHANNEL_B)) == 2 and len(h.sent(CHANNEL_A)) == 1
    assert len(h.user.calls_of("copy_media")) == 1  # the staging copy is reused


async def test_cancelling_a_partly_sent_row_stubs_the_fragment(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky_send_text(h, monkeypatch, 2, TelegramUnavailable("blip"), deliver=False)
    post = await h.post(SPLIT_TEXT)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    fragment = h.sent()[0]
    await h.rt.store.set_topic_fields(h.topic_a.id, active=False)
    h.clock.advance(31)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.unsorted
    assert h.world.get(CHANNEL_A, fragment.message_id).text == "↪ moved"
    assert len(h.sent()) == 1


# --- move-ignores-stub-failure -----------------------------------------------------------------


async def test_failed_stub_blocks_the_republish_until_it_is_retried(h: Harness) -> None:
    post = await h.post("unique story text")
    post = await h.publish(post)
    old = h.sent()[0]
    h.bot.fail_next("edit_text", TelegramUnavailable("blip"))
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and not result.stubbed
    row = await h.row(post)
    assert row.moved_from[0]["pending_stubs"] == [[old.message_id, "↪ moved to <b>Fintech</b>"]]
    h.clock.advance(5)
    await h.pub.tick()  # channel A's budget is busy with the failed edit: nothing yet
    assert h.sent(CHANNEL_B) == []
    assert "unique story text" in h.world.get(CHANNEL_A, old.message_id).text
    restarted = Publisher(h.rt)  # the pending stub is stored, not held in memory
    h.clock.advance(25)
    await restarted.tick()
    msg = h.world.get(CHANNEL_A, old.message_id)
    assert msg.text == "↪ moved to Fintech" and msg.buttons is None
    assert len(h.sent(CHANNEL_B)) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and "pending_stubs" not in row.moved_from[0]


async def test_a_deleted_old_message_does_not_block_the_republish(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    h.bot.fail_next("edit_text", NotAllowed("other", "MESSAGE_ID_INVALID"))
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and result.stubbed
    assert "pending_stubs" not in (await h.row(post)).moved_from[0]
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent(CHANNEL_B)) == 1


# --- reconcile-drops-earlier-ids ---------------------------------------------------------------


async def _crash_after(h: Harness, post: Any, committed: int) -> list[int]:
    """Put a published row back into ``sending`` with only the first ``committed`` ids."""
    row = await h.row(post)
    ids = list(row.message_ids)
    await h.rt.store.set_publication_fields(
        row.id, state=PUB_SENDING, message_ids=ids[:committed], sent_at=None
    )
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued, published_at=None)
    return ids


async def test_reconcile_of_a_split_post_keeps_every_id_and_move_stubs_them_all(
    h: Harness,
) -> None:
    post = await h.publish(await h.post(SPLIT_TEXT))
    ids = await _crash_after(h, post, committed=len(h.sent()) - 1)
    assert len(ids) >= 3
    await Publisher(h.rt).reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == ids
    await h.pub.move(post.id, h.topic_b.id)
    assert all(h.world.get(CHANNEL_A, mid).text == "↪ moved to Fintech" for mid in ids)


async def test_reconcile_of_a_media_first_post_keeps_the_media(h: Harness) -> None:
    post = await h.publish(await h.post(LONG_TEXT, media="video"))
    staging = list((await h.row(post)).staging_ids)
    ids = await _crash_after(h, post, committed=1)
    await Publisher(h.rt).reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == ids and row.staging_ids == staging
    await h.pub.move(post.id, h.topic_b.id)
    assert h.world.get(CHANNEL_A, ids[0]).text == "↪ moved"
    assert h.world.get(CHANNEL_A, ids[1]).text == "↪ moved to Fintech"


async def test_reconcile_requeues_only_the_missing_tail(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky_send_text(h, monkeypatch, 2, TelegramUnavailable("blip"), deliver=False)
    post = await h.post(SPLIT_TEXT)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING)  # crash after part 1
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.message_ids == [h.sent()[0].message_id]
    h.clock.advance(5)
    await restarted.tick()
    msgs = h.sent()
    assert len({m.text for m in msgs}) == len(msgs) >= 3
    assert (await h.row(post)).message_ids == [m.message_id for m in msgs]


# --- failed-edit-hot-loop ----------------------------------------------------------------------


async def _corroborated(h: Harness) -> Any:
    post = await h.publish(await h.post())
    await h.rt.store.add_corroboration(post.id, -7)
    h.clock.advance(30)
    return post


async def _ticks(h: Harness, n: int, every: int = 2) -> None:
    for _ in range(n):
        h.clock.advance(every)
        await h.pub.tick()


async def test_a_permanently_failing_edit_is_tried_once(h: Harness) -> None:
    post = await _corroborated(h)
    writes = len(h.pub._channel_writes[CHANNEL_A])
    h.bot.fail_next("edit_text", NotAllowed("other", "MESSAGE_ID_INVALID"))
    await h.pub.tick()
    await _ticks(h, 10)
    assert len(h.bot.calls_of("edit_text")) == 1
    assert len(h.pub._channel_writes[CHANNEL_A]) == writes + 1
    row = await h.row(post)
    assert row.shown_corroboration == 1 and row.edited_at is not None


async def test_a_flood_wait_on_an_edit_waits_it_out(h: Harness) -> None:
    await _corroborated(h)
    h.bot.fail_next("edit_text", FloodWait(300))
    queued = await h.post("next story")
    await h.pub.enqueue(queued.id)
    await h.pub.tick()  # the send goes first, the edit waits for the budget
    h.clock.advance(20)
    await h.pub.tick()
    assert len(h.bot.calls_of("edit_text")) == 1
    later = await h.post("blocked story")
    await h.pub.enqueue(later.id)
    await _ticks(h, 10)
    assert len(h.bot.calls_of("edit_text")) == 1
    assert not any("blocked story" in t for t in texts(h))  # the channel waits too
    h.clock.advance(300)
    await h.pub.tick()
    await _ticks(h, 15)
    assert any("blocked story" in t for t in texts(h))
    assert len(h.bot.calls_of("edit_text")) == 2


async def test_a_transient_edit_failure_backs_off_by_the_cooldown(h: Harness) -> None:
    await _corroborated(h)
    h.bot.fail_next("edit_text", CuratorError("hiccup"))
    await h.pub.tick()
    await _ticks(h, 25)  # 50 s
    assert len(h.bot.calls_of("edit_text")) == 1
    await _ticks(h, 6)
    assert len(h.bot.calls_of("edit_text")) == 2
    assert "+1 more" in h.sent()[0].text


# --- reconcile-stuck-sending -------------------------------------------------------------------


async def _stuck_sending(h: Harness) -> Any:
    post = await h.post()
    await h.pub.enqueue(post.id)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(
        row.id, state=PUB_SENDING, channel_id=CHANNEL_A, style="repost", attempts=1
    )
    return post


async def test_a_lookup_that_failed_at_start_is_retried_by_the_tick(h: Harness) -> None:
    post = await _stuck_sending(h)
    h.user.fail_next("find_message", TelegramUnavailable("down"))
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    assert (await h.row(post)).state == PUB_SENDING
    h.clock.advance(30)
    await restarted.tick()
    assert (await h.row(post)).state == PUB_PENDING
    h.clock.advance(5)
    await restarted.tick()
    assert (await h.row(post)).state == PUB_SENT and len(h.sent()) == 1


async def test_an_unverifiable_sending_row_falls_into_the_digest_after_six_hours(
    h: Harness,
) -> None:
    post = await _stuck_sending(h)
    h.rt.user = None
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    h.clock.advance(3600)
    await restarted.tick()
    assert (await h.row(post)).state == PUB_SENDING
    h.clock.advance(5 * 3600 + 1)
    await restarted.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest
    assert h.sent() == []


async def test_a_channel_that_is_gone_cancels_the_sending_row_at_once(h: Harness) -> None:
    post = await _stuck_sending(h)
    h.user.fail_next("find_message", ChatGone("deleted"))
    await Publisher(h.rt).reconcile()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest


# --- reconcile-unlocked-while-live -------------------------------------------------------------


async def test_reconcile_waits_for_a_send_in_flight(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    entered, gate = asyncio.Event(), asyncio.Event()
    send_text = h.bot.send_text

    async def slow_send(chat_id: int, html: str, **kw: Any) -> int:
        entered.set()
        await gate.wait()
        return await send_text(chat_id, html, **kw)

    monkeypatch.setattr(h.bot, "send_text", slow_send)
    h.clock.advance(5)
    tick = asyncio.create_task(h.pub.tick())
    await entered.wait()
    reconcile = asyncio.create_task(h.pub.reconcile())  # /go or a re-bind while live
    for _ in range(20):
        await asyncio.sleep(0)
    assert not reconcile.done()
    gate.set()
    await tick
    await reconcile
    assert h.user.calls_of("find_message") == []
    assert (await h.row(post)).state == PUB_SENT
    h.clock.advance(30)
    await h.pub.tick()
    assert len(h.sent()) == 1


# --- reconcile-wrong-style-after-fallback ------------------------------------------------------


async def test_crash_after_a_forward_fell_back_to_a_repost_is_found_by_the_link(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    h.user.fail_next("forward", ForwardsRestricted("protected now"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    send_text = h.bot.send_text

    async def deliver_then_crash(chat_id: int, html: str, **kw: Any) -> int:
        await send_text(chat_id, html, **kw)
        raise Crash

    monkeypatch.setattr(h.bot, "send_text", deliver_then_crash)
    h.clock.advance(5)
    with pytest.raises(Crash):
        await h.pub.tick()
    row = await h.row(post)
    assert row.state == PUB_SENDING and row.style == "repost" and row.message_ids == []
    monkeypatch.setattr(h.bot, "send_text", send_text)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.style == "repost"
    assert row.message_ids == [h.sent()[0].message_id]
    assert h.user.calls_of("find_message")[0]["contains"] == "https://t.me/kunuz/1"
    h.clock.advance(30)
    await restarted.tick()
    assert len(h.sent()) == 1


# --- reconcile-shown-corroboration -------------------------------------------------------------


async def test_reconcile_records_the_count_that_was_rendered(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    send_text = h.bot.send_text

    async def corroborate_send_crash(chat_id: int, html: str, **kw: Any) -> int:
        await h.rt.store.add_corroboration(post.id, -7)  # the sorter commits mid-send
        await send_text(chat_id, html, **kw)
        raise Crash

    monkeypatch.setattr(h.bot, "send_text", corroborate_send_crash)
    h.clock.advance(5)
    with pytest.raises(Crash):
        await h.pub.tick()
    monkeypatch.setattr(h.bot, "send_text", send_text)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.shown_corroboration == 0
    h.clock.advance(61)
    await restarted.tick()
    edits = h.bot.calls_of("edit_text")
    assert len(edits) == 1
    assert h.sent()[0].text.startswith("Kun.uz · source · +1 more\n")
    assert (await h.row(post)).shown_corroboration == 1


# --- bot cannot post: a partly sent row given up after 6 h -------------------------------------


async def test_giving_up_on_a_partly_sent_row_stubs_the_fragment(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky_send_text(h, monkeypatch, 2, BotCannotPost("rights lost"), deliver=False)
    post = await h.post(SPLIT_TEXT)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    fragment = h.sent()[0]
    h.bot.cannot_post.add(CHANNEL_A)
    h.clock.advance(int(timedelta(hours=6).total_seconds()) + 1)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest
    assert h.world.get(CHANNEL_A, fragment.message_id).text == "↪ moved"
