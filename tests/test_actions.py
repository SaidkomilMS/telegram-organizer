"""actions.py: the exact semantics of every action, pacing, FloodWait, outcome edits, undo."""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeClock, FakeUserGateway
from tg_curator.domain import (
    KV,
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PROPOSAL_FAILED,
    PROPOSAL_UNDONE,
    Proposal,
)
from tg_curator.errors import CuratorError, FloodWait, NotAllowed, TelegramUnavailable
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.actions import MUTE_FOREVER, ActionExecutor
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo


@pytest.fixture
def executor(rt: Runtime, monkeypatch: pytest.MonkeyPatch) -> ActionExecutor:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.stats = StatsService(rt)
    rt.folders = FolderManager(rt)
    rt.review = ReviewService(rt)
    rt.actions = ActionExecutor(rt)
    rt.actions._rng = random.Random(7)  # type: ignore[attr-defined]
    return rt.actions  # type: ignore[return-value]


async def approved(
    rt: Runtime, info: ChatInfo, kind: str, *, role: str = "source", send: bool = True
) -> Proposal:
    """A proposal of ``kind`` for ``info``, sent to the owner and approved (confirmed for a
    leave), exactly as the review would leave it for the executor."""
    rt.user.add_chat(info)  # type: ignore[union-attr]
    await rt.store.upsert_chat(info, role=role)  # type: ignore[arg-type]
    p = await rt.store.create_proposal(
        kind, reason=f"{kind} {info.title}", review_day=START.date(), chat_id=info.id
    )
    if send:
        await rt.review.send()
    await rt.review.decide(p.id, "approve")
    if kind == "leave":
        await rt.review.decide(p.id, "confirm")
    refreshed = await rt.store.get_proposal(p.id)
    assert refreshed is not None and refreshed.state == PROPOSAL_APPROVED
    return refreshed


def message(bot_gw: FakeBotGateway, p: Proposal) -> tuple[str, list[list[str | None]]]:
    msg = next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == p.bot_message_id)
    return msg.text, [[b.data for b in row] for row in msg.buttons or []]


async def state(rt: Runtime, p: Proposal) -> Proposal:
    refreshed = await rt.store.get_proposal(p.id)
    assert refreshed is not None
    return refreshed


# --- the four actions ------------------------------------------------------------------------


async def test_mute_until_now_plus_mute_days_with_undo_button(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    info = make_chat(title="Noisy")
    p = await approved(rt, info, "mute")
    await executor.tick()
    until = START + timedelta(days=30)
    assert user_gw.mutes == {info.id: until}
    assert (await rt.store.get_chat(info.id)).muted_until == until  # type: ignore[union-attr]
    done = await state(rt, p)
    assert done.state == PROPOSAL_DONE and done.executed_at == START
    text, buttons = message(bot_gw, p)
    assert text.endswith("Muted until Nov 4 ✓") and buttons == [[f"rv:{p.id}:undo"]]


async def test_archive_is_mute_forever_plus_archive(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    info = make_chat()
    p = await approved(rt, info, "archive")
    await executor.tick()
    assert user_gw.mutes == {info.id: MUTE_FOREVER} and user_gw.archived == {info.id: True}
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and chat.archived and chat.muted_until == MUTE_FOREVER
    text, buttons = message(bot_gw, p)
    assert text.endswith("Archived and muted ✓") and buttons == [[f"rv:{p.id}:undo"]]


async def test_folder_flags_the_chat_and_syncs_the_folder(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    info = make_chat()
    p = await approved(rt, info, "folder")
    await executor.tick()
    assert (await rt.store.get_chat(info.id)).in_low_signal  # type: ignore[union-attr]
    [save] = user_gw.calls_of("save_folder")
    assert save["title"] == "Low signal" and save["chat_ids"] == [info.id]
    assert user_gw.mutes == {} and user_gw.archived == {}
    text, buttons = message(bot_gw, p)
    assert text.endswith("Moved to “Low signal” ✓") and buttons == [[f"rv:{p.id}:undo"]]


async def test_leave_after_confirmation_is_not_undoable(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    info = make_chat(kind="group")
    p = await approved(rt, info, "leave")
    await executor.tick()
    assert user_gw.left == [info.id]
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and not chat.active and chat.left_at == START
    text, buttons = message(bot_gw, p)
    assert text.endswith("Left ✓") and buttons == []
    assert await rt.store.kv_get(KV.LEAVE_LOG) == [START.isoformat()]
    with pytest.raises(CuratorError, match="cannot be undone"):
        await rt.review.undo(p.id)


async def test_only_approved_rows_run_and_output_chats_are_never_touched(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    proposed = make_chat()
    rt.user.add_chat(proposed)  # type: ignore[union-attr]
    await rt.store.upsert_chat(proposed)
    await rt.store.create_proposal("mute", reason="x", review_day=START.date(), chat_id=proposed.id)
    output = make_chat(title="ML & AI")
    p = await approved(rt, output, "mute", role="output")
    await executor.tick()
    assert user_gw.calls_of("mute") == [] and user_gw.mutes == {}
    failed = await state(rt, p)
    assert failed.state == PROPOSAL_FAILED
    text, buttons = message(bot_gw, p)
    assert text.endswith("could not mute: this chat is one of the curator's own channels")
    assert buttons == []


# --- pacing ----------------------------------------------------------------------------------


async def test_one_action_per_tick_with_a_random_gap_of_20_to_90_seconds(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    first = await approved(rt, make_chat(), "mute")
    second = await approved(rt, make_chat(), "archive")
    await executor.tick()
    assert len(user_gw.calls_of("mute")) == 1 and (await state(rt, first)).state == PROPOSAL_DONE
    clock.advance(19)
    await executor.tick()
    assert (await state(rt, second)).state == PROPOSAL_APPROVED
    clock.advance(90 - 19)
    await executor.tick()
    assert (await state(rt, second)).state == PROPOSAL_DONE


async def test_two_leaves_are_at_least_leave_interval_apart(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    a = await approved(rt, make_chat(kind="group"), "leave")
    b = await approved(rt, make_chat(kind="group"), "leave")
    mute = await approved(rt, make_chat(), "mute")
    await executor.tick()
    assert user_gw.left == [a.chat_id]
    clock.advance(timedelta(minutes=5))
    await executor.tick()  # the mute goes out; the second leave waits
    assert user_gw.left == [a.chat_id] and (await state(rt, mute)).state == PROPOSAL_DONE
    clock.advance(timedelta(minutes=24, seconds=59))
    await executor.tick()
    assert user_gw.left == [a.chat_id]
    clock.advance(1)
    await executor.tick()
    assert user_gw.left == [a.chat_id, b.chat_id]
    assert (await state(rt, b)).executed_at == START + timedelta(minutes=30)


async def test_leaves_per_day_cap(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    await rt.settings_file.set_value("review.leaves_per_day", 2)
    chats = [make_chat(kind="group") for _ in range(3)]
    for info in chats:
        await approved(rt, info, "leave")
    for _ in range(3):
        await executor.tick()
        clock.advance(timedelta(minutes=30))
    assert user_gw.left == [chats[0].id, chats[1].id]
    clock.advance(timedelta(hours=22, minutes=29))  # 23 h 59 min after the first leave
    await executor.tick()
    assert len(user_gw.left) == 2
    clock.advance(timedelta(minutes=2))
    await executor.tick()
    assert user_gw.left == [c.id for c in chats]
    assert len(await rt.store.kv_get(KV.LEAVE_LOG)) == 2  # the log keeps the last 24 h only


# --- failures --------------------------------------------------------------------------------


async def test_flood_wait_longer_than_the_gateway_sleeps_is_scheduled_not_slept(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway, clock: FakeClock,
) -> None:  # fmt: skip
    p = await approved(rt, make_chat(), "mute")
    user_gw.fail_next("mute", FloodWait(600))
    await executor.tick()
    waiting = await state(rt, p)
    assert waiting.state == PROPOSAL_APPROVED
    assert waiting.payload["next_attempt_at"] == (START + timedelta(seconds=600)).isoformat()
    text, buttons = message(bot_gw, p)
    assert text.endswith("Telegram asked to wait, retrying at 12:10") and buttons == []
    clock.advance(599)
    await executor.tick()
    assert (await state(rt, p)).state == PROPOSAL_APPROVED
    clock.advance(1)
    await executor.tick()
    assert (await state(rt, p)).state == PROPOSAL_DONE and user_gw.mutes[p.chat_id] is not None


async def test_leave_refused_for_a_creator_is_reported_in_the_message(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    p = await approved(rt, make_chat(kind="group", is_creator=True), "leave")
    await executor.tick()
    failed = await state(rt, p)
    assert failed.state == PROPOSAL_FAILED and user_gw.left == []
    text, buttons = message(bot_gw, p)
    assert text.endswith("could not leave: you created this chat") and buttons == []
    assert await rt.store.kv_get(KV.LEAVE_LOG) is None


async def test_other_gateway_errors_fail_the_proposal_once(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    p = await approved(rt, make_chat(), "mute")
    user_gw.fail_next("mute", NotAllowed("not_a_member"))
    await executor.tick()
    assert (await state(rt, p)).state == PROPOSAL_FAILED
    text, _ = message(bot_gw, p)
    assert text.endswith("could not mute: the account is not in this chat any more")


async def test_nothing_happens_without_an_account(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    p = await approved(rt, make_chat(), "mute")
    rt.user = None
    await executor.tick()
    assert (await state(rt, p)).state == PROPOSAL_APPROVED and user_gw.mutes == {}


# --- undo ------------------------------------------------------------------------------------


async def test_undo_mute_archive_and_folder(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway, clock: FakeClock,
) -> None:  # fmt: skip
    mute = await approved(rt, make_chat(), "mute")
    archive = await approved(rt, make_chat(), "archive")
    folder = await approved(rt, make_chat(), "folder")
    for _ in range(3):
        await executor.tick()
        clock.advance(90)
    assert [p.state for p in await rt.store.proposals_by_state(PROPOSAL_DONE)] == ["done"] * 3

    undone = await rt.review.undo(mute.id)
    assert undone.state == PROPOSAL_UNDONE and user_gw.mutes[mute.chat_id] is None
    assert (await rt.store.get_chat(mute.chat_id)).muted_until is None  # type: ignore[union-attr]

    await rt.review.undo(archive.id)
    assert user_gw.mutes[archive.chat_id] is None and user_gw.archived[archive.chat_id] is False
    chat = await rt.store.get_chat(archive.chat_id)
    assert chat is not None and not chat.archived and chat.muted_until is None

    await rt.review.undo(folder.id)
    assert not (await rt.store.get_chat(folder.chat_id)).in_low_signal  # type: ignore[union-attr]
    # Telegram cannot hold an empty folder: with no chat flagged any more, "Low signal" is
    # deleted and its kv id forgotten (DESIGN §17.2); the next flagged chat recreates it.
    assert user_gw.folders == {} and user_gw.calls_of("delete_folder")
    assert await rt.store.kv_get("folders.low_signal_id") is None
    for p in (mute, archive, folder):
        text, buttons = message(bot_gw, p)
        assert text.endswith("Undone ✓") and buttons == []
    with pytest.raises(CuratorError):
        await rt.review.undo(mute.id)  # already undone


async def test_undo_archive_restores_the_mute_the_chat_had_before(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    # §18.6: an archive can be proposed for a chat that is already muted; undoing it must give
    # back that mute, not unmute the chat.
    info = make_chat()
    await approved(rt, info, "mute")
    await executor.tick()
    clock.advance(90)
    earlier = (await rt.store.get_chat(info.id)).muted_until  # type: ignore[union-attr]
    assert earlier is not None and earlier > clock.now()

    # The chat as the dialogs now report it (muted), as a sync would store it.
    synced = next(c for c in user_gw.chats if c.id == info.id)
    archive = await approved(rt, synced, "archive")
    await executor.tick()
    assert (await state(rt, archive)).state == PROPOSAL_DONE
    assert user_gw.mutes[info.id] == MUTE_FOREVER and user_gw.archived[info.id] is True

    await rt.review.undo(archive.id)
    assert user_gw.mutes[info.id] == earlier and user_gw.archived[info.id] is False
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and not chat.archived and chat.muted_until == earlier


async def test_undo_archive_keeps_the_earlier_mute_across_an_outage_retry(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    # The first attempt mutes forever, then the outage hits; a dialog sync then reads the
    # forever-mute back into the store. The retry must not take that for the earlier mute.
    earlier = START + timedelta(days=3)
    info = make_chat(muted_until=earlier)
    archive = await approved(rt, info, "archive")
    user_gw.fail_next("set_archived", TelegramUnavailable("down"))
    with pytest.raises(TelegramUnavailable):
        await executor.tick()
    assert (await state(rt, archive)).state == PROPOSAL_APPROVED
    await rt.store.set_chat_fields(info.id, muted_until=MUTE_FOREVER)

    clock.advance(90)
    await executor.tick()
    assert (await state(rt, archive)).state == PROPOSAL_DONE
    await rt.review.undo(archive.id)
    assert user_gw.mutes[info.id] == earlier
    assert (await rt.store.get_chat(info.id)).muted_until == earlier  # type: ignore[union-attr]


async def test_undo_archive_unmutes_when_the_earlier_mute_has_run_out(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, clock: FakeClock,
) -> None:  # fmt: skip
    info = make_chat(muted_until=START + timedelta(hours=1))
    archive = await approved(rt, info, "archive")
    await executor.tick()
    clock.advance(2 * 3600)
    await rt.review.undo(archive.id)
    assert user_gw.mutes[info.id] is None and user_gw.archived[info.id] is False
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and not chat.archived and chat.muted_until is None


async def test_undo_before_execution_is_refused(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await approved(rt, make_chat(), "mute")
    with pytest.raises(CuratorError):
        await executor.undo(p.id)
    with pytest.raises(CuratorError):
        await executor.undo(404)


async def test_outcome_without_a_sent_message_does_not_edit(
    rt: Runtime, executor: ActionExecutor, make_chat: Callable[..., ChatInfo],
    bot_gw: FakeBotGateway, user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    p = await approved(rt, make_chat(), "mute", send=False)
    await executor.tick()
    assert (await state(rt, p)).state == PROPOSAL_DONE and len(user_gw.mutes) == 1
    assert bot_gw.calls_of("edit_text") == []


def test_mute_forever_is_aware_utc() -> None:
    assert MUTE_FOREVER.tzinfo is UTC and MUTE_FOREVER.year == datetime.max.year
