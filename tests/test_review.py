"""review.py: level selection, reasons, idempotency, delivery, the decision machine, the clock."""

from __future__ import annotations

import inspect
import itertools
import re
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeClock
from tg_curator.domain import (
    KV,
    PROPOSAL_APPROVED,
    PROPOSAL_CONFIRMING,
    PROPOSAL_NEVER,
    PROPOSAL_PROPOSED,
    PROPOSAL_SKIPPED,
    NewPost,
    PostStatus,
    Proposal,
)
from tg_curator.errors import CuratorError
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions import actions, discovery, review
from tg_curator.subscriptions.actions import ActionExecutor
from tg_curator.subscriptions.discovery import Discovery
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash


@pytest.fixture
def svc(rt: Runtime, monkeypatch: pytest.MonkeyPatch) -> ReviewService:
    """The subscription services wired into the runtime; owner messages without the 1 s gap."""
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.stats = StatsService(rt)
    rt.folders = FolderManager(rt)
    rt.actions = ActionExecutor(rt)
    rt.discovery = Discovery(rt)
    rt.review = ReviewService(rt)
    return rt.review


async def chat_with_numbers(
    rt: Runtime,
    info: ChatInfo,
    *,
    days: int,
    volume: int = 41,
    sorted_: int = 0,
    repeats: int = 35,
    repeated: ChatInfo | None = None,
) -> ChatInfo:
    """A source chat observed ``days`` days with ``volume`` messages, ``sorted_`` posts that
    reached a topic and ``repeats`` repeats of ``repeated`` (another chat)."""
    store = rt.store
    now = rt.clock.now()
    rt.user.add_chat(info)  # type: ignore[union-attr]
    await store.upsert_chat(info)
    await store.set_chat_fields(info.id, first_seen_at=now - timedelta(days=days))
    if volume:
        await store.bump_chat_daily(info.id, now.date(), volume)
    for i in range(sorted_):
        await add_post(rt, info.id, 100 + i, status=PostStatus.digested, topic_id=1)
    if repeats:
        assert repeated is not None
        if await store.get_chat(repeated.id) is None:
            await store.upsert_chat(repeated)
        for i in range(repeats):
            root = await add_post(rt, repeated.id, next(_ROOT_IDS), status=PostStatus.published)
            await add_post(rt, info.id, 300 + i, status=PostStatus.duplicate, duplicate_of=root.id)
    return info


_ROOT_IDS = itertools.count(5000)


async def add_post(rt: Runtime, chat_id: int, mid: int, **decision: Any) -> Any:
    text = f"post {chat_id} {mid}"
    new = NewPost(
        chat_id=chat_id,
        message_id=mid,
        kind="post",
        message_ids=[mid],
        posted_at=rt.clock.now(),
        via="live",
        text=text,
        text_hash=text_hash(text),
        urls=[],
    )
    created = await rt.store.insert_post(new, **decision)
    assert created is not None
    return created


def data_of(bot_gw: FakeBotGateway, message_id: int) -> list[list[str | None]]:
    msg = next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == message_id)
    return [[b.data for b in row] for row in msg.buttons or []]


def text_of(bot_gw: FakeBotGateway, message_id: int) -> str:
    return next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == message_id).text


# --- the catalogue ---------------------------------------------------------------------------


def test_every_review_key_used_ships_in_english_and_nothing_else() -> None:
    used: set[str] = set()
    for module in (review, actions, discovery):
        used |= set(re.findall(r'"(review_[a-z_]+)"', inspect.getsource(module)))
    used |= {f"review_verb_{k}" for k in ("folder", "mute", "archive", "leave")}
    with (LOCALES_DIR / "en" / "review.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    assert shipped <= Translator(locales_dir=LOCALES_DIR).english_keys()


# --- building: the level per observation period ----------------------------------------------


@pytest.mark.parametrize(
    ("days", "kind"), [(6, None), (7, "folder"), (14, "mute"), (21, "mute"), (30, "leave")]
)
async def test_strongest_level_by_observation_period(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], days: int, kind: str | None
) -> None:
    kunuz = make_chat(username="kunuz", title="Kun.uz")
    chat = await chat_with_numbers(rt, make_chat(title="Reposter"), days=days, repeated=kunuz)
    proposals = await svc.build()
    if kind is None:
        assert proposals == []
        return
    [p] = proposals
    assert p.kind == kind and p.chat_id == chat.id and p.state == PROPOSAL_PROPOSED
    assert p.review_day == START.date()
    # The day count is how long the chat was observed (at most the 30-day window).
    observed = min(days, 30)
    assert p.reason.startswith(
        f"41 posts in {observed} days, 0 reached a topic, 85% were repeats of @kunuz"
    )


async def test_reason_counts_the_days_actually_observed(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    kunuz = make_chat(username="kunuz")
    young = await chat_with_numbers(
        rt, make_chat(title="Young"), days=10, volume=300, sorted_=6, repeats=210, repeated=kunuz
    )
    old = await chat_with_numbers(rt, make_chat(title="Old"), days=31, repeated=kunuz)
    by_chat = {p.chat_id: p for p in await svc.build()}
    assert by_chat[young.id].reason.startswith("300 posts in 10 days")
    assert by_chat[old.id].reason.startswith("41 posts in 30 days")


async def test_folder_proposal_names_the_configured_folder(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("folders.low_signal_name", "Noise")
    kunuz = make_chat(username="kunuz")
    await chat_with_numbers(rt, make_chat(title="Fold me"), days=7, repeated=kunuz)
    [p] = await svc.build()
    rendered = await review.render_proposal(rt, p)
    assert rendered.html.startswith("📁 <b>Move to the “Noise” folder</b> · Fold me")
    await svc.send()
    assert bot_gw.sent(OWNER_ID)[0].text.startswith("📁 Move to the “Noise” folder · Fold me")
    assert "Low signal" not in bot_gw.sent(OWNER_ID)[0].text


async def test_leave_is_never_proposed_for_a_chat_the_account_created(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    kunuz = make_chat(username="kunuz")
    await chat_with_numbers(rt, make_chat(is_creator=True), days=30, repeated=kunuz)
    [p] = await svc.build()
    assert p.kind == "archive"
    assert p.reason.endswith("archive and mute")


async def test_admin_suffix_on_leave_and_source_without_username(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    source = make_chat(title="Private Source")
    await chat_with_numbers(rt, make_chat(is_admin=True), days=30, repeated=source)
    [p] = await svc.build()
    assert p.kind == "leave"
    assert p.reason == (
        "41 posts in 30 days, 0 reached a topic, 85% were repeats of Private Source"
        " (you are an admin; leaving drops that)"
    )


async def test_mute_needs_repeats_otherwise_folder_and_leave_needs_zero_sorted(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    quiet = await chat_with_numbers(rt, make_chat(), days=14, repeats=0)
    useful = await chat_with_numbers(rt, make_chat(), days=30, volume=100, sorted_=1, repeats=0)
    kinds = {p.chat_id: p.kind for p in await svc.build()}
    assert kinds == {quiet.id: "folder", useful.id: "archive"}  # 1 % signal, not zero


async def test_too_little_volume_gives_no_proposal(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    await chat_with_numbers(rt, make_chat(), days=30, volume=5, repeats=0)
    assert await svc.build() == []


async def test_keep_left_output_and_open_proposals_are_skipped(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    kept = await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    await rt.store.set_chat_fields(kept.id, keep=True)
    left = await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    await rt.store.set_chat_fields(left.id, active=False)
    busy = await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    await rt.store.create_proposal(
        "mute", reason="old", review_day=START.date() - timedelta(days=7), chat_id=busy.id,
        state=PROPOSAL_APPROVED,
    )  # fmt: skip
    fresh = await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    assert [p.chat_id for p in await svc.build()] == [fresh.id]


async def test_build_is_idempotent_per_review_day(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], clock: FakeClock
) -> None:
    await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    [p] = await svc.build()
    assert await svc.build() == []
    await svc.decide(p.id, "skip")
    assert await svc.build() == []  # decided today: no second row today
    clock.advance(timedelta(days=7))
    [again] = await svc.build()  # skipped proposals come back when the numbers still hold
    assert again.id != p.id and again.kind == "leave"


async def test_no_folder_proposal_while_folders_are_disabled(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    kunuz = make_chat(username="kunuz")
    await chat_with_numbers(rt, make_chat(), days=7, repeated=kunuz)
    muted = await chat_with_numbers(rt, make_chat(), days=14, repeated=kunuz)
    await rt.store.kv_set(KV.FOLDERS_DISABLED, "limit")
    assert [(p.chat_id, p.kind) for p in await svc.build()] == [(muted.id, "mute")]


async def test_build_then_calls_discovery(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo]
) -> None:
    calls: list[str] = []

    class Spy:
        async def propose(self) -> list[Proposal]:
            calls.append("propose")
            return [
                await rt.store.create_proposal(
                    "new_topic", reason="cluster", review_day=START.date(), payload={"name": "X"}
                )
            ]

    rt.discovery = Spy()  # type: ignore[assignment]
    await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    proposals = await svc.build()
    assert calls == ["propose"] and [p.kind for p in proposals] == ["leave", "new_topic"]


# --- sending ---------------------------------------------------------------------------------


async def test_send_groups_caps_stores_ids_and_sets_buttons(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    kunuz = make_chat(username="kunuz")
    leave = await chat_with_numbers(rt, make_chat(title="Leave me"), days=30, repeated=kunuz)
    folder = await chat_with_numbers(rt, make_chat(title="Fold me"), days=7, repeated=kunuz)
    mute = await chat_with_numbers(rt, make_chat(title="Mute me"), days=14, repeated=kunuz)
    await rt.settings_file.set_value("review.max_proposals", 2)
    await svc.build()
    assert await svc.send() == 2
    sent = bot_gw.sent(OWNER_ID)
    assert [m.text.splitlines()[0] for m in sent[:2]] == [
        "📁 Move to the “Low signal” folder · Fold me",
        "🔕 Mute · Mute me",
    ]
    assert sent[2].text == "1 more proposals are held back until the next review."
    by_chat = {p.chat_id: p for p in await rt.store.proposals_by_state(PROPOSAL_PROPOSED)}
    assert by_chat[folder.id].bot_message_id == 1 and by_chat[mute.id].bot_message_id == 2
    assert by_chat[leave.id].bot_message_id is None
    pid = by_chat[folder.id].id
    assert data_of(bot_gw, 1) == [[f"rv:{pid}:approve", f"rv:{pid}:skip"], [f"rv:{pid}:never"]]
    assert "85% were repeats of @kunuz" in sent[0].text
    # the held-back one goes out with the next send, and nothing is sent twice
    await rt.settings_file.set_value("review.max_proposals", 20)
    assert await svc.send() == 1
    assert await svc.send() == 0
    assert [m.text.splitlines()[0] for m in bot_gw.sent(OWNER_ID)[3:]] == ["🚪 Leave · Leave me"]


# --- decisions -------------------------------------------------------------------------------


async def test_leave_confirmation_flow(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await chat_with_numbers(rt, make_chat(title="Old group"), days=30, repeats=0)
    [p] = await svc.build()
    await svc.send()
    mid = (await rt.store.get_proposal(p.id)).bot_message_id  # type: ignore[union-attr]
    assert mid is not None

    confirming = await svc.decide(p.id, "approve")
    assert confirming.state == PROPOSAL_CONFIRMING and confirming.decided_at is None
    assert data_of(bot_gw, mid) == [[f"rv:{p.id}:confirm", f"rv:{p.id}:cancel"]]
    assert "Leave Old group? This cannot be undone" in text_of(bot_gw, mid)

    cancelled = await svc.decide(p.id, "cancel")
    assert cancelled.state == PROPOSAL_PROPOSED
    assert data_of(bot_gw, mid) == [[f"rv:{p.id}:approve", f"rv:{p.id}:skip"], [f"rv:{p.id}:never"]]
    assert "cannot be undone without a new invitation." in text_of(bot_gw, mid)
    assert "Leave Old group?" not in text_of(bot_gw, mid)

    await svc.decide(p.id, "approve")
    approved = await svc.decide(p.id, "confirm")
    assert approved.state == PROPOSAL_APPROVED and approved.decided_at == START
    assert data_of(bot_gw, mid) == [] and "Approved ✓" in text_of(bot_gw, mid)
    with pytest.raises(CuratorError):
        await svc.decide(p.id, "confirm")  # a stale tap


async def test_approve_skip_and_never(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    kunuz = make_chat(username="kunuz")
    a = await chat_with_numbers(rt, make_chat(), days=14, repeated=kunuz)
    b = await chat_with_numbers(rt, make_chat(), days=14, repeated=kunuz)
    c = await chat_with_numbers(rt, make_chat(), days=14, repeated=kunuz)
    by_chat = {p.chat_id: p for p in await svc.build()}
    await svc.send()
    approved = await svc.decide(by_chat[a.id].id, "approve")
    assert approved.state == PROPOSAL_APPROVED and approved.decided_at == START
    skipped = await svc.decide(by_chat[b.id].id, "skip")
    assert skipped.state == PROPOSAL_SKIPPED
    never = await svc.decide(by_chat[c.id].id, "never")
    assert never.state == PROPOSAL_NEVER
    assert (await rt.store.get_chat(c.id)).keep is True  # type: ignore[union-attr]
    texts = [m.text for m in bot_gw.sent(OWNER_ID)]
    assert any("Approved ✓ — will be done within minutes." in t for t in texts)
    assert any("Skipped — it comes back" in t for t in texts)
    assert any("Never asked again" in t for t in texts)
    assert all(m.buttons is None for m in bot_gw.sent(OWNER_ID))
    with pytest.raises(ValueError):
        await svc.decide(by_chat[a.id].id, "maybe")
    with pytest.raises(CuratorError):
        await svc.decide(999, "approve")


async def test_decide_without_a_sent_message_still_records(
    rt: Runtime, svc: ReviewService, make_chat: Callable[..., ChatInfo], bot_gw: FakeBotGateway
) -> None:
    await chat_with_numbers(rt, make_chat(), days=30, repeats=0)
    [p] = await svc.build()
    assert (await svc.decide(p.id, "skip")).state == PROPOSAL_SKIPPED
    assert bot_gw.calls_of("edit_text") == []


# --- the weekly clock ------------------------------------------------------------------------


async def test_tick_runs_a_missed_week_once_then_waits_for_the_next_moment(
    rt: Runtime, svc: ReviewService, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[datetime] = []

    async def build() -> list[Proposal]:
        runs.append(clock.now())
        return []

    monkeypatch.setattr(svc, "build", build)
    assert START.weekday() == 0  # Monday: Sunday 11:00 passed while "down"
    await svc.tick()
    assert runs == [START]
    assert await rt.store.kv_get(KV.REVIEW_LAST_DAY) == "2026-10-04"
    await svc.tick()
    assert len(runs) == 1
    clock.set(datetime(2026, 10, 11, 10, 59, tzinfo=UTC))
    await svc.tick()
    assert len(runs) == 1
    clock.set(datetime(2026, 10, 11, 11, 0, tzinfo=UTC))
    await svc.tick()
    assert len(runs) == 2 and await rt.store.kv_get(KV.REVIEW_LAST_DAY) == "2026-10-11"


async def test_tick_honours_the_review_day_and_timezone(
    rt: Runtime, svc: ReviewService, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    await rt.settings_file.set_value("review.weekday", "monday")
    await rt.settings_file.set_value("review.hour", 18)
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")  # UTC+5
    await rt.store.kv_set(KV.REVIEW_LAST_DAY, "2026-09-28")
    runs: list[datetime] = []

    async def build() -> list[Proposal]:
        runs.append(clock.now())
        return []

    monkeypatch.setattr(svc, "build", build)
    clock.set(datetime(2026, 10, 5, 12, 59, tzinfo=UTC))  # 17:59 local Monday
    await svc.tick()
    assert runs == []
    clock.set(datetime(2026, 10, 5, 13, 0, tzinfo=UTC))  # 18:00 local
    await svc.tick()
    assert len(runs) == 1 and await rt.store.kv_get(KV.REVIEW_LAST_DAY) == "2026-10-05"
