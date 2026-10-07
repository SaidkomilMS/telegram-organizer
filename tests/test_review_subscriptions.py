"""Review fixes in the subscriptions group: folder ownership by title, hand edits of the
curator's folders, no re-proposal of a level already in effect, outages during an action."""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import timedelta

import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeClock, FakeUserGateway
from tests.test_review import chat_with_numbers
from tg_curator.domain import (
    KV,
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PROPOSAL_FAILED,
    PROPOSAL_UNDONE,
    Proposal,
)
from tg_curator.errors import NotAllowed, TelegramUnavailable
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.actions import MUTE_FOREVER, ActionExecutor
from tg_curator.subscriptions.discovery import Discovery
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo


@pytest.fixture
def services(rt: Runtime, monkeypatch: pytest.MonkeyPatch) -> Runtime:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.stats = StatsService(rt)
    rt.folders = FolderManager(rt)
    rt.discovery = Discovery(rt)
    rt.review = ReviewService(rt)
    rt.actions = ActionExecutor(rt)
    rt.actions._rng = random.Random(7)  # type: ignore[attr-defined]
    return rt


def adopt_like_the_real_gateway(user_gw: FakeUserGateway, folder_id: int) -> None:
    """``TelethonUserGateway.list_folders`` marks every listed id as its own, so its
    ``NotOwnedError`` guard lets a write to a user's folder through; only ``folders.py``'s
    own check stands between the curator and that folder."""
    user_gw._own_folders.add(folder_id)  # type: ignore[attr-defined]


async def source(rt: Runtime, info: ChatInfo, *, role: str = "source") -> int:
    rt.user.add_chat(info)  # type: ignore[union-attr]
    await rt.store.upsert_chat(info, role=role)  # type: ignore[arg-type]
    return info.id


async def run_folder_move(rt: Runtime, chat_id: int) -> Proposal:
    """A folder proposal for ``chat_id`` sent, approved and carried out, as in a real review."""
    review, actions = rt.review, rt.actions
    assert review is not None and actions is not None
    p = await rt.store.create_proposal(
        "folder", reason="noisy", review_day=START.date(), chat_id=chat_id
    )
    await review.send()
    await review.decide(p.id, "approve")
    actions._not_before = None  # type: ignore[attr-defined]
    await actions.tick()
    done = await rt.store.get_proposal(p.id)
    assert done is not None and done.state == PROPOSAL_DONE
    return done


async def get(rt: Runtime, p: Proposal) -> Proposal:
    refreshed = await rt.store.get_proposal(p.id)
    assert refreshed is not None
    return refreshed


# --- a reused folder id belongs to the user ------------------------------------------------


async def test_a_reused_id_holding_a_user_folder_is_neither_renamed_nor_merged(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    low = await source(rt, make_chat())
    await run_folder_move(rt, low)
    old_id = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert user_gw.folders[old_id] == ("Low signal", [low])
    # The user deletes "Low signal" by hand and creates "Work", which gets the same id.
    user_gw.folders[old_id] = ("Work", [424242])
    adopt_like_the_real_gateway(user_gw, old_id)
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders[old_id] == ("Work", [424242])
    new_id = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert new_id not in (None, old_id) and user_gw.folders[new_id] == ("Low signal", [low])
    assert old_id not in [c["folder_id"] for c in user_gw.calls_of("save_folder")[1:]]


async def test_a_reused_id_holding_a_flag_only_user_folder_is_not_deleted(
    services: Runtime, user_gw: FakeUserGateway
) -> None:
    rt = services
    await rt.store.kv_set(KV.FOLDERS_LOW_SIGNAL_ID, 3)  # the curator's, deleted by hand since
    user_gw.folders[3] = ("Channels", [])  # the user's own: broadcasts=True, no include_peers
    adopt_like_the_real_gateway(user_gw, 3)
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders == {3: ("Channels", [])}
    assert user_gw.calls_of("delete_folder") == [] and user_gw.calls_of("save_folder") == []
    assert user_gw.calls_of("get_folder") == []  # the user's folder is not even read
    assert await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None


async def test_a_title_renamed_in_the_config_still_finds_the_curators_folder(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    out = await source(rt, make_chat(), role="output")
    await rt.folders.sync()  # type: ignore[union-attr]
    fid = await rt.store.kv_get(KV.FOLDERS_CURATED_ID)
    await rt.settings_file.set_value("folders.curated_name", "Topics")
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders == {fid: ("Topics", [out])}
    assert [c["folder_id"] for c in user_gw.calls_of("save_folder")] == [None, fid]


# --- chats the user takes out by hand stay out ---------------------------------------------


async def test_a_chat_taken_out_of_low_signal_by_hand_stays_out_and_is_undone(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
) -> None:  # fmt: skip
    rt = services
    a = await source(rt, make_chat())
    b = await source(rt, make_chat())
    pa = await run_folder_move(rt, a)
    pb = await run_folder_move(rt, b)
    fid = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert user_gw.folders[fid] == ("Low signal", [a, b])
    user_gw.folders[fid] = ("Low signal", [b])  # the owner drags ``a`` out in Telegram
    saves = len(user_gw.calls_of("save_folder"))
    for _ in range(3):
        await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders[fid] == ("Low signal", [b])
    assert len(user_gw.calls_of("save_folder")) == saves  # nothing to write back
    assert (await rt.store.get_chat(a)).in_low_signal is False  # type: ignore[union-attr]
    assert (await rt.store.get_chat(b)).in_low_signal is True  # type: ignore[union-attr]
    assert (await get(rt, pa)).state == PROPOSAL_UNDONE
    assert (await get(rt, pb)).state == PROPOSAL_DONE
    msg = next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == pa.bot_message_id)
    assert msg.text.endswith("Undone ✓") and msg.buttons is None


async def test_a_channel_taken_out_of_curated_by_hand_stays_out_until_put_back(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    one = await source(rt, make_chat(), role="output")
    two = await source(rt, make_chat(), role="output")
    await rt.folders.sync()  # type: ignore[union-attr]
    fid = await rt.store.kv_get(KV.FOLDERS_CURATED_ID)
    user_gw.folders[fid] = ("Curated", [two])  # taken out by hand
    await rt.folders.sync()  # type: ignore[union-attr]
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders[fid] == ("Curated", [two])
    user_gw.folders[fid] = ("Curated", [two, one])  # put back by hand
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders[fid] == ("Curated", [two, one])
    assert await rt.store.kv_get("folders.curated_hand_removed") is None
    three = await source(rt, make_chat(), role="output")
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders[fid] == ("Curated", [two, one, three])


async def test_a_folder_deleted_by_hand_is_recreated_with_its_chats_still_flagged(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    """Deleting the whole folder is not taking every chat out of it (DESIGN §12 recreates)."""
    rt = services
    a = await source(rt, make_chat())
    p = await run_folder_move(rt, a)
    fid = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    del user_gw.folders[fid]
    await rt.folders.sync()  # type: ignore[union-attr]
    new_id = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert user_gw.folders[new_id] == ("Low signal", [a])
    assert (await rt.store.get_chat(a)).in_low_signal is True  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_DONE


# --- a level already in effect is not proposed again ---------------------------------------


async def test_a_done_folder_move_is_not_proposed_again_next_week(
    services: Runtime, make_chat: Callable[..., ChatInfo], clock: FakeClock
) -> None:
    rt = services
    info = await chat_with_numbers(rt, make_chat(), days=40, volume=100, sorted_=4, repeats=0)
    [p] = await rt.review.build()  # type: ignore[union-attr]
    assert p.kind == "folder"
    await rt.store.set_proposal_fields(p.id, state=PROPOSAL_DONE)
    await rt.store.set_chat_fields(info.id, in_low_signal=True)
    clock.advance(timedelta(days=7))
    assert await rt.review.build() == []  # type: ignore[union-attr]


async def test_a_chat_in_low_signal_can_still_be_offered_a_stronger_level(
    services: Runtime, make_chat: Callable[..., ChatInfo]
) -> None:
    rt = services
    kunuz = make_chat(username="kunuz")
    info = await chat_with_numbers(rt, make_chat(is_creator=True), days=30, repeated=kunuz)
    await rt.store.set_chat_fields(info.id, in_low_signal=True)
    [p] = await rt.review.build()  # type: ignore[union-attr]
    assert (p.kind, p.chat_id) == ("archive", info.id)


async def test_a_running_mute_and_an_archive_are_not_proposed_again(
    services: Runtime, make_chat: Callable[..., ChatInfo], clock: FakeClock
) -> None:
    rt = services
    kunuz = make_chat(username="kunuz")
    muted = await chat_with_numbers(rt, make_chat(), days=14, repeated=kunuz)
    archived = await chat_with_numbers(rt, make_chat(is_creator=True), days=30, repeated=kunuz)
    await rt.store.set_chat_fields(muted.id, muted_until=START + timedelta(days=10))
    await rt.store.set_chat_fields(archived.id, archived=True, muted_until=MUTE_FOREVER)
    assert await rt.review.build() == []  # type: ignore[union-attr]
    clock.advance(timedelta(days=11))  # the mute ran out: muting again is a real proposal
    assert [(p.chat_id, p.kind) for p in await rt.review.build()] == [  # type: ignore[union-attr]
        (muted.id, "mute")
    ]


# --- an outage during an action is retried, a failure leaves nothing half done ---------------


async def approved(rt: Runtime, info: ChatInfo, kind: str) -> Proposal:
    await source(rt, info)
    p = await rt.store.create_proposal(kind, reason=kind, review_day=START.date(), chat_id=info.id)
    await rt.review.send()  # type: ignore[union-attr]
    await rt.review.decide(p.id, "approve")  # type: ignore[union-attr]
    return await get(rt, p)


async def test_an_outage_between_mute_and_archive_keeps_the_row_approved(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway, clock: FakeClock,
) -> None:  # fmt: skip
    rt = services
    info = make_chat()
    p = await approved(rt, info, "archive")
    user_gw.fail_next("set_archived", TelegramUnavailable("connection reset"))
    with pytest.raises(TelegramUnavailable):  # up to the Supervisor, logged once per outage
        await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_APPROVED
    clock.advance(timedelta(minutes=2))
    await rt.actions.tick()  # type: ignore[union-attr]
    done = await get(rt, p)
    assert done.state == PROPOSAL_DONE
    assert user_gw.mutes[info.id] == MUTE_FOREVER and user_gw.archived[info.id] is True
    msg = next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == p.bot_message_id)
    assert [[b.data for b in row] for row in msg.buttons or []] == [[f"rv:{p.id}:undo"]]


async def test_an_outage_on_a_mute_keeps_the_approval(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    p = await approved(rt, make_chat(), "mute")
    user_gw.fail_next("mute", TelegramUnavailable("timeout"))
    with pytest.raises(TelegramUnavailable):
        await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_APPROVED
    await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_DONE


async def test_an_archive_that_fails_for_good_is_unmuted_again(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    info = make_chat()
    p = await approved(rt, info, "archive")
    user_gw.fail_next("set_archived", NotAllowed("other", "refused"))
    await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_FAILED
    assert user_gw.mutes[info.id] is None and not user_gw.archived.get(info.id)
    chat = await rt.store.get_chat(info.id)
    assert chat is not None and chat.muted_until is None and chat.archived is False


async def test_a_folder_move_whose_sync_fails_for_good_clears_the_flag(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    info = make_chat()
    p = await approved(rt, info, "folder")
    user_gw.fail_next("list_folders", NotAllowed("other", "refused"))
    await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_FAILED
    assert (await rt.store.get_chat(info.id)).in_low_signal is False  # type: ignore[union-attr]
    await rt.folders.sync()  # type: ignore[union-attr]
    assert user_gw.folders == {}  # the periodic sync does not move it behind the owner's back


async def test_a_folder_move_hit_by_an_outage_is_retried(
    services: Runtime, make_chat: Callable[..., ChatInfo], user_gw: FakeUserGateway
) -> None:
    rt = services
    info = make_chat()
    p = await approved(rt, info, "folder")
    user_gw.fail_next("list_folders", TelegramUnavailable("timeout"))
    with pytest.raises(TelegramUnavailable):
        await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_APPROVED
    await rt.actions.tick()  # type: ignore[union-attr]
    assert (await get(rt, p)).state == PROPOSAL_DONE
    assert list(user_gw.folders.values()) == [("Low signal", [info.id])]
