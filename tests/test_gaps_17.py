"""DESIGN §17 items 2–6, end to end across the modules they touch.

Each module's own test file covers its half; these tests pin the seams: the folder deletion as
the folder manager and the fake gateway see it together, an outage injected into a fake
reaching the caller untouched, the effective LLM read timeout built from the settings, the
out-of-credit notice sharing the month's one notice with the cap, and the review opening with
the absorbed-volume lines of the topics accepted since the last review.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeClock, FakeUserGateway
from tg_curator.config import LlmSettings
from tg_curator.domain import KV, PROPOSAL_DONE, NewPost, PostStatus, Proposal, Topic
from tg_curator.errors import CuratorError, NotOwnedError, TelegramUnavailable
from tg_curator.llm.budget import Budget, Price
from tg_curator.llm.providers import build_backend
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ABSORBED_REPORTED, ReviewService
from tg_curator.telegram.gateway import ChatInfo, UserGateway
from tg_curator.textutil import text_hash

# --- §17.2 delete_folder ---------------------------------------------------------------------


async def test_the_fake_deletes_only_its_own_folders(user_gw: FakeUserGateway) -> None:
    assert isinstance(user_gw, UserGateway)
    folder_id = await user_gw.save_folder(None, "Low signal", [-1001])
    user_gw.folders[9] = ("Work", [-1002])  # the user's own folder
    with pytest.raises(NotOwnedError):
        await user_gw.delete_folder(9)
    await user_gw.delete_folder(folder_id)
    assert list(user_gw.folders) == [9]
    with pytest.raises(NotOwnedError):  # forgotten: not a valid id any more
        await user_gw.save_folder(folder_id, "Low signal", [-1001])


async def test_sync_deletes_low_signal_once_the_last_flag_is_undone(
    rt: Runtime, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    folders = FolderManager(rt)
    info = make_chat()
    user_gw.add_chat(info)
    await rt.store.upsert_chat(info)
    await rt.store.set_chat_fields(info.id, in_low_signal=True)
    await rt.store.create_proposal(
        "folder", reason="r", review_day=START.date(), chat_id=info.id, state=PROPOSAL_DONE
    )
    await folders.sync()
    folder_id = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert folder_id is not None
    await rt.store.set_chat_fields(info.id, in_low_signal=False)
    await folders.sync()
    assert user_gw.calls_of("delete_folder") == [{"folder_id": folder_id}]
    assert user_gw.folders == {} and await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None


# --- §17.3 TelegramUnavailable ---------------------------------------------------------------


def test_telegram_unavailable_is_a_curator_error() -> None:
    assert issubclass(TelegramUnavailable, CuratorError)


async def test_an_outage_reaches_the_folder_loop_untouched(
    rt: Runtime, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    """The supervisor, not the folder manager, decides what an outage means (§17.3)."""
    info = make_chat()
    await rt.store.upsert_chat(info, role="output")
    user_gw.fail_next("list_folders", TelegramUnavailable("down"))
    with pytest.raises(TelegramUnavailable):
        await FolderManager(rt).sync()
    assert await rt.store.kv_get(KV.FOLDERS_DISABLED) is None  # an outage is not the limit


# --- §17.4 the effective LLM read timeout ----------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "setting", "expected"),
    [
        ("provider", 30, 60.0),
        ("provider", 90, 90.0),
        ("selfhosted", 30, 180.0),
        ("selfhosted", 600, 600.0),
    ],
)
def test_timeout_setting_only_raises_the_floor(mode: str, setting: int, expected: float) -> None:
    fields: dict[str, Any] = {"mode": mode, "model": "m", "timeout_seconds": setting}
    if mode == "provider":
        fields.update(provider="openai", api_key="k")
    else:
        fields.update(base_url="http://localhost:11434/v1")
    backend = build_backend(LlmSettings(**fields))
    assert backend.read_timeout == expected


# --- §17.5 out of credit ---------------------------------------------------------------------


async def _zero(_model: str | None) -> Price:
    return Price(0.0, 0.0)


async def test_out_of_credit_and_the_cap_share_one_notice_a_month(
    rt: Runtime, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    budget = Budget(
        rt.store, clock, cap_usd=1.0, timezone="UTC", price_lookup=_zero, notifier=rt.notifier
    )
    await budget.out_of_credit("Mistral")
    await budget.out_of_credit("Mistral")  # a second refusal the same month: still one notice
    await rt.store.add_usage(budget.month(), 0, 0, 0, 5.0)  # and the cap is crossed as well
    assert await budget.reserve(1, 1) is None
    [notice] = bot_gw.sent(OWNER_ID)
    assert notice.text.startswith("Mistral reports the account is out of credit")
    clock.advance(timedelta(days=31))
    await budget.out_of_credit("Mistral")  # a new month: told again
    assert len(bot_gw.sent(OWNER_ID)) == 2


# --- §17.6 absorbed volume -------------------------------------------------------------------


@pytest.fixture
def review(rt: Runtime, monkeypatch: pytest.MonkeyPatch) -> ReviewService:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.review = ReviewService(rt)
    return rt.review


async def _post(rt: Runtime, chat_id: int, mid: int, *, age: timedelta, status: PostStatus) -> None:
    text = f"post {mid}"
    new = NewPost(
        chat_id=chat_id,
        message_id=mid,
        kind="post",
        message_ids=[mid],
        posted_at=rt.clock.now() - age,
        via="live",
        text=text,
        text_hash=text_hash(text),
        urls=[],
    )
    assert await rt.store.insert_post(new, status=status) is not None


async def _accepted(rt: Runtime, name: str, *, absorbed: int, ago: timedelta) -> Proposal:
    """A ``new_topic`` proposal the owner accepted ``ago``, as discovery leaves it."""
    topic = await rt.store.upsert_topic(
        Topic(id=0, key=name.lower(), name=name, created_at=rt.clock.now() - ago)
    )
    at = rt.clock.now() - ago
    proposal = await rt.store.create_proposal(
        "new_topic",
        reason="31 unsorted posts",
        review_day=(at - timedelta(days=1)).date(),
        payload={"name": name, "member_post_ids": [], "topic_id": topic.id, "absorbed": absorbed},
        state=PROPOSAL_DONE,
    )
    await rt.store.set_proposal_fields(proposal.id, decided_at=at, executed_at=at)
    found = await rt.store.get_proposal(proposal.id)
    assert found is not None
    return found


async def test_review_opens_with_the_absorbed_lines_once(
    rt: Runtime, review: ReviewService, bot_gw: FakeBotGateway
) -> None:
    accepted = await _accepted(rt, "Crypto", absorbed=34, ago=timedelta(days=5))
    # inside the re-sorted window (acceptance - 7 days .. acceptance): 4 still unsorted
    for i, age in enumerate((6, 8, 10, 11)):
        await _post(rt, -1001, i + 1, age=timedelta(days=age), status=PostStatus.unsorted)
    await _post(rt, -1001, 50, age=timedelta(days=13), status=PostStatus.unsorted)  # before it
    await _post(rt, -1001, 51, age=timedelta(days=1), status=PostStatus.unsorted)  # after it
    await _post(rt, -1001, 52, age=timedelta(days=7), status=PostStatus.digest)  # sorted
    await review.send()
    [opening] = bot_gw.sent(OWNER_ID)
    since = (accepted.executed_at - timedelta(days=7)).date()  # type: ignore[operator]
    assert opening.text == (
        f"📥 Crypto absorbed 34 of 38 unsorted posts since {since:%b} {since.day}"
    )
    refreshed = await rt.store.get_proposal(accepted.id)
    assert (
        refreshed is not None and refreshed.payload[ABSORBED_REPORTED] == START.date().isoformat()
    )
    assert refreshed.payload["absorbed"] == 34  # discovery's figure is kept as it was
    await review.send()  # the next review does not repeat it
    assert len(bot_gw.sent(OWNER_ID)) == 1


async def test_a_topic_accepted_today_waits_for_the_next_review(
    rt: Runtime, review: ReviewService, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    await _accepted(rt, "Crypto", absorbed=3, ago=timedelta(minutes=5))
    await review.send()
    assert bot_gw.sent(OWNER_ID) == []
    clock.advance(timedelta(days=7))
    await review.send()
    [opening] = bot_gw.sent(OWNER_ID)
    assert opening.text.startswith("📥 Crypto absorbed 3 of 3 unsorted posts since")


async def test_the_absorbed_lines_come_before_the_proposals_and_wait_for_an_owner(
    rt: Runtime, review: ReviewService, bot_gw: FakeBotGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await _accepted(rt, "Real estate", absorbed=0, ago=timedelta(days=3))
    await _accepted(rt, "Gone", absorbed=9, ago=timedelta(days=3))
    gone = await rt.store.get_topic_by_key("gone")
    assert gone is not None
    await rt.store.set_topic_fields(gone.id, active=False)  # removed since: no line for it
    await rt.store.create_proposal(
        "merge_topics",
        reason="they compete",
        review_day=START.date(),
        payload={"a_id": gone.id, "b_id": accepted.payload["topic_id"], "shared": 5},
    )
    monkeypatch.setattr(rt, "bot", None)  # no bot yet: nothing is sent, nothing marked
    await review.send()
    still = await rt.store.get_proposal(accepted.id)
    assert still is not None and ABSORBED_REPORTED not in still.payload
    monkeypatch.setattr(rt, "bot", bot_gw)
    assert await review.send() == 1
    opening, merge = bot_gw.sent(OWNER_ID)
    assert len(opening.text.splitlines()) == 1
    assert opening.text.startswith("📥 Real estate absorbed 0 of 0 unsorted posts")
    assert "Merge topics" in merge.text
