"""Audit gaps of the pipeline-a group: S11, TC-8, SORT-1/TUN-3, SORT-3, TUN-2, TUN-5, CG-1,
CG-2, CG-5, RD-6, SAFE-4, SAFE-5 (each test names the gap it closes)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from tests.fakes import START, FakeClassifier, FakeClock, FakeUserGateway
from tests.test_bot_preview import Scene, scene  # noqa: F401 - the /preview scene fixture
from tests.test_intake import ROOT_TEXT, FakeSorter, daily, group_rows
from tests.test_preview import CHANNEL, cand
from tg_curator.bot import setup
from tg_curator.bot.preview import backfill_due
from tg_curator.bot.reports import stats_parts
from tg_curator.config import Settings
from tg_curator.control import stats_command
from tg_curator.domain import KV, Candidate, Chat, Post, PostStatus, Topic
from tg_curator.errors import FloodWait
from tg_curator.pipeline import backfill as backfill_mod
from tg_curator.pipeline.backfill import BackfillService
from tg_curator.pipeline.engine import DecisionEngine, RecentIndex, features_of
from tg_curator.pipeline.intake import IntakeService, passes_floor
from tg_curator.pipeline.preview import PreviewService, render_text
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage

LONG = "A long enough post about the central bank and its key rate decision this week. " * 2


async def no_sleep(_: float) -> None:
    return None


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def fake_sorter(rt: Runtime) -> FakeSorter:
    rt.sorter = FakeSorter(rt.store)
    return rt.sorter


@pytest.fixture
def intake(rt: Runtime, fake_sorter: FakeSorter) -> IntakeService:
    rt.intake = IntakeService(rt)
    return rt.intake


async def source(rt: Runtime, user_gw: FakeUserGateway, info: ChatInfo) -> Chat:
    user_gw.add_chat(info)
    return await rt.store.upsert_chat(info)


async def stream(messages: Sequence[IncomingMessage]) -> AsyncIterator[IncomingMessage]:
    for msg in messages:
        yield msg


# --- S11: only a full read stands in for setup step 6 ------------------------------------------


async def test_s11_a_partial_backfill_neither_skips_the_read_nor_ticks_step_6(
    scene: Scene,  # noqa: F811 - the imported fixture
) -> None:
    rt = scene.rt
    chats = await rt.store.list_chats(role="source", active=True)
    assert rt.backfill is not None
    await rt.backfill.run(days=1, chat_ids=[chats[0].id])  # `curator backfill --chat X --days 1`
    assert await rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT) is None
    assert await backfill_due(rt) is True
    assert await setup._confirmation(rt, "preview", None) is None  # /setup stops at step 6

    reads = scene.history_reads()
    await scene.drv.say("/preview")  # what step 6 runs: every chat is read, the report is sent
    assert scene.history_reads() == reads + len(chats)
    assert "Preview of the last 3 days" in scene.drv.sent()[-1].text
    assert await rt.store.kv_get(KV.SETUP_PREVIEW_SHOWN) == rt.clock.now().isoformat()
    line = await setup._confirmation(rt, "preview", None)
    assert line is not None and line.startswith("preview:")


async def test_s11_a_full_three_day_run_still_records_the_read(
    scene: Scene,  # noqa: F811
) -> None:
    rt = scene.rt
    assert rt.backfill is not None
    await rt.backfill.run(days=3)
    assert await backfill_due(rt) is False
    # but the walkthrough still waits for the report itself
    assert await setup._confirmation(rt, "preview", None) is None


# --- SAFE-4: every FloodWait is slept, the chat is read --------------------------------------


async def test_safe4_many_flood_waits_are_all_slept_and_the_chat_is_read(
    rt: Runtime,
    fake_sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    sleeps = Sleeps()
    rt.intake = IntakeService(rt)
    service = BackfillService(rt, sleep=sleeps)
    chat = await source(rt, user_gw, make_chat())
    user_gw.seed(make_message(chat, text="post", date=START - timedelta(hours=2)))
    for _ in range(5):
        user_gw.fail_next("history", FloodWait(300))

    result = await service.run(days=1)
    assert sleeps.calls == [300.0] * 5
    assert result.skipped_chats == [] and result.messages == 1 and result.submitted == 1


async def test_safe4_a_chat_throttled_for_hours_is_read_again_after_the_others(
    rt: Runtime,
    fake_sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill_mod, "FLOOD_DEFER_AFTER", 1000.0)
    sleeps = Sleeps()
    rt.intake = IntakeService(rt)
    service = BackfillService(rt, sleep=sleeps)
    fine = await source(rt, user_gw, make_chat(title="Fine"))
    slow = await source(rt, user_gw, make_chat(title="Slow"))  # lower id: read first
    user_gw.seed(make_message(slow, text="slow post", date=START - timedelta(hours=2)))
    user_gw.seed(make_message(fine, text="fine post", date=START - timedelta(hours=1)))
    user_gw.fail_next("history", FloodWait(600))
    user_gw.fail_next("history", FloodWait(600))  # 1200 s > 1000 s: put back

    result = await service.run(days=1)
    assert result.skipped_chats == []
    assert sorted(c.text for c in fake_sorter.submitted) == ["fine post", "slow post"]
    assert [c["chat_id"] for c in user_gw.calls_of("history")] == [
        slow.id,
        slow.id,
        fine.id,
        slow.id,
    ]


# --- SAFE-5: one backfill at a time ----------------------------------------------------------


async def test_safe5_two_runs_never_read_two_histories_at_once(
    rt: Runtime,
    fake_sorter: FakeSorter,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    rt.intake = IntakeService(rt)

    async def yielding(_: float) -> None:
        await asyncio.sleep(0)

    service = BackfillService(rt, sleep=yielding)
    for _ in range(3):
        chat = await source(rt, user_gw, make_chat())
        for i in range(3):
            user_gw.seed(make_message(chat, text=f"p{i}", date=START - timedelta(hours=i + 1)))
    open_now = {"n": 0, "max": 0}
    original = user_gw.history

    def counting(chat_id: int, **kwargs: Any) -> AsyncIterator[IncomingMessage]:
        async def gen() -> AsyncIterator[IncomingMessage]:
            open_now["n"] += 1
            open_now["max"] = max(open_now["max"], open_now["n"])
            try:
                async for msg in original(chat_id, **kwargs):
                    await asyncio.sleep(0)
                    yield msg
            finally:
                open_now["n"] -= 1

        return gen()

    user_gw.history = counting  # type: ignore[method-assign]
    started = asyncio.Event()

    async def first() -> Any:
        started.set()
        return await service.run(days=3)

    async def second() -> Any:
        await started.wait()
        await asyncio.sleep(0)
        assert service.running
        return await service.run(days=3)

    one, two = await asyncio.gather(first(), second())
    assert open_now["max"] == 1
    assert one.messages == two.messages == 9
    assert not service.running


async def test_safe5_preview_refuses_while_a_cli_backfill_reads(
    scene: Scene,  # noqa: F811
) -> None:
    rt = scene.rt
    assert rt.backfill is not None
    lock: asyncio.Lock = rt.backfill._lock  # type: ignore[attr-defined]
    async with lock:  # a `curator backfill` is reading
        reads = scene.history_reads()
        await scene.drv.say("/preview")
        assert scene.history_reads() == reads
        assert "already" in scene.drv.sent()[-1].text.lower()


# --- TUN-2: a fully trusted source is always immediate ---------------------------------------


def _topic(channel: int | None = CHANNEL) -> Topic:
    return Topic(id=1, key="ml", name="ML", channel_id=channel, created_at=START)


def _chat(trust: float | None) -> Chat:
    return Chat(id=-100, kind="channel", title="T", first_seen_at=START, trust=trust)


@pytest.mark.parametrize(
    "overrides",
    [{"realtime_strength": 4.0}, {"neutral_trust": 3.5}, {"realtime_strength": 10.0}],
)
async def test_tun2_trust_3_is_queued_whatever_the_thresholds(
    rt: Runtime, overrides: dict[str, float]
) -> None:
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings.sorting, key, value)
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {1: 0.9}
    engine = DecisionEngine(settings, classifier, RecentIndex(3), None)
    decision = await engine.classify(
        cand(-100, 1, "short text"), np.ones(64, np.float32) / 8, _chat(3.0), [_topic()], START
    )
    assert decision.status == PostStatus.queued and decision.would_realtime is True
    # an ordinary source still waits under the same settings (the 0-2 trust levels keep their
    # threshold rule, and a high neutral_trust does not make every chat fully trusted)
    ordinary = await engine.classify(
        cand(-100, 2, "short text"), np.ones(64, np.float32) / 8, _chat(2.0), [_topic()], START
    )
    assert ordinary.status in (PostStatus.held, PostStatus.digest)
    assert ordinary.would_realtime is False


async def test_tun2_a_held_trust_3_post_goes_out_when_its_hold_expires(
    rt: Runtime, make_chat: Callable[..., ChatInfo]
) -> None:
    """Held under an old rule (or trust raised meanwhile): expiry re-asks the engine."""
    info = make_chat()
    await rt.store.upsert_chat(info)
    topic = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=START)
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {topic.id: 0.9}
    queued: list[int] = []

    class Pub:
        async def enqueue(self, post_id: int) -> bool:
            queued.append(post_id)
            return True

    rt.publisher = Pub()  # type: ignore[assignment]
    sorter = Sorter(rt)
    post = await sorter.submit(cand(info.id, 1, LONG))
    assert post is not None and post.status == PostStatus.held
    await rt.store.set_chat_fields(info.id, trust=3.0)
    await rt.settings_file.set_value("sorting.realtime_strength", 4.0)
    rt.clock.advance(timedelta(minutes=46))  # type: ignore[attr-defined]
    await sorter.tick()
    assert queued == [post.id]


# --- SORT-3: every link of a stored post is indexed -------------------------------------------


def _unit(*xs: float) -> np.ndarray:
    v = np.zeros(64, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def test_sort3_a_later_post_sharing_the_second_link_is_a_url_repeat() -> None:
    index = RecentIndex(3)
    stored = dataclasses.replace(
        cand(-1, 1, "first"),
        urls=["https://example.com/a-story", "https://news.org/the-article?utm_source=x"],
    )
    feats = features_of(stored)
    assert len(feats.url_keys) == 2
    index.add(1, -1, 1, None, feats.text_hash, feats.url_keys, None, _unit(1.0, 0.0))
    later = dataclasses.replace(cand(-2, 1, "second"), urls=["https://news.org/the-article"])
    embedding = _unit(0.75, float(np.sqrt(1 - 0.75**2)))  # cosine 0.75: below semantic
    match = index.match_url(features_of(later).url_keys, embedding)
    assert match is not None and match.root_id == 1
    assert index.match_semantic(embedding) is None


class MapEmbedder:
    """Fixed vectors per text, so a test sets the cosine between two posts."""

    id = "map"
    dim = 64

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.vectors[t] for t in texts])


async def test_sort3_the_index_rebuilt_at_start_holds_every_link(
    rt: Runtime, make_chat: Callable[..., ChatInfo]
) -> None:
    a, b = make_chat(), make_chat()
    for info in (a, b):
        await rt.store.upsert_chat(info)
    first_text, second_text = "The bank story, two links.", "Same article, other words."
    rt.embedder = MapEmbedder(  # type: ignore[assignment]
        {first_text: _unit(1.0, 0.0), second_text: _unit(0.75, float(np.sqrt(1 - 0.75**2)))}
    )
    rt.publisher = None
    first = dataclasses.replace(
        cand(a.id, 1, first_text),
        urls=["https://example.com/a-story", "https://news.org/the-article"],
    )
    root = await Sorter(rt).submit(first)
    assert root is not None
    second = dataclasses.replace(
        cand(b.id, 1, second_text, posted_at=START + timedelta(minutes=5)),
        urls=["https://news.org/the-article"],
    )
    repeat = await Sorter(rt).submit(second)  # a fresh sorter: the index comes from the store
    assert repeat is not None and repeat.status == PostStatus.duplicate
    assert repeat.dup_kind == "url" and repeat.duplicate_of == root.id


# --- SORT-1 / TUN-3: the preview counts corroboration --------------------------------------


@pytest.fixture
async def neutral_scene(rt: Runtime, make_chat: Callable[..., ChatInfo]) -> tuple[list[int], Topic]:
    topic = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML", channel_id=CHANNEL, created_at=START)
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {topic.id: 0.9}
    chats = []
    for _ in range(3):
        info = make_chat()
        await rt.store.upsert_chat(info)
        chats.append(info.id)
    rt.publisher = None
    return chats, topic


async def _story(rt: Runtime, chats: list[int], repeats: int) -> list[Post]:
    sorter = Sorter(rt)
    posts = []
    for i in range(repeats + 1):
        post = await sorter.submit(
            cand(chats[i], 1, LONG, posted_at=START + timedelta(minutes=i * 2))
        )
        assert post is not None
        posts.append(post)
    return posts


async def test_sort1_a_neutral_story_repeated_twice_shows_immediate_yes(
    rt: Runtime, neutral_scene: tuple[list[int], Topic]
) -> None:
    chats, topic = neutral_scene
    posts = await _story(rt, chats, repeats=2)
    stored = await rt.store.get_post(posts[0].id)
    assert stored is not None and stored.would_realtime is True and stored.corroboration == 2

    report = await PreviewService(rt).replay()
    root = report.decisions[0]
    assert root.post_id == posts[0].id and root.corroboration == 2
    assert root.would_realtime is True and root.status == PostStatus.queued
    assert root.strength == stored.strength
    assert (
        "ML 0.90  immediate: yes  [queued]"
        in render_text(report, topic_names={topic.id: "ML"}).splitlines()[1]
    )


async def test_sort1_lowering_realtime_strength_flips_a_once_repeated_story(
    rt: Runtime, neutral_scene: tuple[list[int], Topic]
) -> None:
    chats, _ = neutral_scene
    await _story(rt, chats, repeats=1)
    service = PreviewService(rt)
    now = await service.replay()
    assert now.decisions[0].would_realtime is False  # 1 + 0.25 + 1 = 2.25 < 3
    then = await service.replay(overrides={"sorting.realtime_strength": 2})
    assert then.decisions[0].would_realtime is True
    assert then.decisions[0].status == PostStatus.queued
    weighted = await service.replay(overrides={"sorting.corroboration_weight": 2})
    assert weighted.decisions[0].would_realtime is True


async def test_sort1_promotion_follows_the_sorter_around_the_hold_and_the_digest(
    rt: Runtime, neutral_scene: tuple[list[int], Topic]
) -> None:
    chats, _ = neutral_scene
    sorter = Sorter(rt)
    # repeats after the hold ran out, before the evening digest: promoted from the digest
    for i, minutes in enumerate((0, 50, 52)):
        await sorter.submit(cand(chats[i], 1, LONG, posted_at=START + timedelta(minutes=minutes)))
    root = (await PreviewService(rt).replay()).decisions[0]
    assert root.status == PostStatus.queued and root.would_realtime is True


async def test_sort1_a_repeat_after_the_digest_was_composed_does_not_promote(
    rt: Runtime, neutral_scene: tuple[list[int], Topic]
) -> None:
    chats, _ = neutral_scene
    sorter = Sorter(rt)
    for i, minutes in enumerate((0, 26 * 60, 26 * 60 + 2)):
        await sorter.submit(cand(chats[i], 1, LONG, posted_at=START + timedelta(minutes=minutes)))
    rt.clock.advance(timedelta(days=1, hours=3))  # type: ignore[attr-defined]
    root = (await PreviewService(rt).replay()).decisions[0]
    assert root.status == PostStatus.digest and root.would_realtime is True


# --- TC-8: topics in the statistics ------------------------------------------------------------


async def test_tc8_a_channel_less_topic_shows_in_stats(
    rt: Runtime, neutral_scene: tuple[list[int], Topic]
) -> None:
    chats, topic = neutral_scene
    trial = await rt.store.upsert_topic(
        Topic(id=0, key="trial", name="Trial", channel_id=None, created_at=START)
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.text_scores["A trial topic post about gardens and tomatoes in October."] = {
        topic.id: 0.1,
        trial.id: 0.9,
    }
    sorter = Sorter(rt)
    await sorter.submit(cand(chats[0], 1, LONG))
    await sorter.submit(
        cand(chats[1], 2, "A trial topic post about gardens and tomatoes in October.")
    )
    rt.stats = StatsService(rt)
    rows = {t.key: t for t in await rt.stats.topic_stats()}
    assert (rows["ml"].posts, rows["ml"].has_channel) == (1, True)
    assert (rows["trial"].posts, rows["trial"].has_channel) == (1, False)

    stats = await rt.stats.chat_stats()
    text = "\n".join(stats_parts(rt, stats, days=30, topics=list(rows.values())))
    assert "Trial (no channel yet): 1 posts" in text and "ML: 1 posts" in text

    lines: list[str] = []

    async def out(line: str) -> None:
        lines.append(line)

    await stats_command(rt, {}, out)
    assert any(line.startswith("topic Trial (no channel yet): 1 posts") for line in lines), lines


# --- CG-1: albums in groups count once --------------------------------------------------------


async def test_cg1_a_group_album_counts_once_and_links_its_caption(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    for i in range(5):
        text = ROOT_TEXT + " " + ROOT_TEXT if i == 3 else ""
        await intake.handle_message(
            make_message(group, text=text, media="photo", grouped_id=99, sender_id=500)
        )
    assert await daily(rt.store, group.id) == {START.date(): 1}
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    (unit,) = fake_sorter.submitted
    assert unit.media == "album" and unit.grouped_id == 99
    assert unit.message_id == 4 and unit.message_ids == [1, 2, 3, 4, 5]
    rows = await group_rows(rt.store, group.id)
    assert {r.unit_root_id for r in rows} == {1} and all(r.closed for r in rows)


async def test_cg1_an_anonymous_album_is_one_unit(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    for i in range(3):
        msg = make_message(group, text=ROOT_TEXT * 2 if i == 1 else "", grouped_id=7)
        await intake.handle_message(_anonymous(msg))
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    assert [(c.message_id, c.message_ids) for c in fake_sorter.submitted] == [(2, [1, 2, 3])]
    assert await daily(rt.store, group.id) == {START.date(): 1}


def _anonymous(msg: IncomingMessage) -> IncomingMessage:
    return dataclasses.replace(msg, sender_id=None)


async def test_cg1_a_backfill_of_a_live_album_keeps_the_count(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    info = make_chat(kind="group")
    group = await source(rt, user_gw, info)
    parts = [
        make_message(
            info,
            text="",
            media="photo",
            grouped_id=5,
            date=START - timedelta(hours=1),
        )
        for _ in range(4)
    ]
    rows = await intake.collect(stream(parts), via="backfill")
    assert rows == []
    await intake.close_units(group.id, START, force=True)
    assert await daily(rt.store, group.id) == {(START - timedelta(hours=1)).date(): 1}
    again = await intake.collect(stream(parts), via="backfill")  # a second backfill
    assert again == []
    assert await daily(rt.store, group.id) == {(START - timedelta(hours=1)).date(): 1}


# --- CG-2: slow threads stay one piece -------------------------------------------------------


async def test_cg2_a_slow_thread_is_one_candidate_linking_its_first_message(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    # 100 + 3 x 120 characters: only the whole thread reaches the 400 floor
    await intake.handle_message(make_message(group, text="R" * 100, sender_id=1))
    for n in range(3):
        clock.advance(timedelta(minutes=6))
        await intake.tick()  # the unit closes below the floor before each reply
        await intake.handle_message(
            make_message(group, text="x" * 120, sender_id=10 + n, reply_to_id=1)
        )
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    (unit,) = fake_sorter.submitted
    assert unit.message_id == 1 and unit.message_ids == [1, 2, 3, 4]
    assert unit.via == "live"
    rows = await group_rows(rt.store, group.id)
    assert {r.unit_root_id for r in rows} == {1} and all(r.closed for r in rows)


async def test_cg2_a_reply_to_a_submitted_unit_starts_a_new_one(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    await intake.handle_message(make_message(group, text="L" * 500, sender_id=1))
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    assert [c.message_id for c in fake_sorter.submitted] == [1]
    await intake.handle_message(make_message(group, text="y" * 50, sender_id=2, reply_to_id=1))
    rows = {r.message_id: r for r in await group_rows(rt.store, group.id)}
    assert rows[1].closed and rows[2].unit_root_id == 2 and not rows[2].closed


# --- CG-5: forwards and link shares pass a lower floor ----------------------------------------


async def test_cg5_short_forwards_and_link_shares_pass_short_chatter_does_not(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    forwarded = (
        "Breaking: the parliament passed the budget bill after a night-long session, "
        "the speaker said; the vote was 201 to 97."
    )
    assert 100 <= len(forwarded) < 400
    url = "https://www.ft.com/content/0f8e7d6c-budget"
    await intake.handle_message(
        make_message(
            group,
            text=forwarded,
            sender_id=1,
            fwd_from_chat_id=-100_777,
            fwd_from_message_id=5,
        )
    )
    await intake.handle_message(
        make_message(group, text=f"Good long read on this: {url}", sender_id=2, urls=(url,))
    )
    await intake.handle_message(make_message(group, text=f"lol {url}", sender_id=3, urls=(url,)))
    await intake.handle_message(make_message(group, text="short chatter " * 5, sender_id=4))
    clock.advance(timedelta(minutes=6))
    await intake.tick()
    assert sorted(c.message_id for c in fake_sorter.submitted) == [1, 2]


def test_cg5_the_floor_rule() -> None:
    def c(text: str, *, fwd: int | None = None, urls: tuple[str, ...] = ()) -> Candidate:
        return dataclasses.replace(cand(-1, 1, text), fwd_from_chat_id=fwd, urls=list(urls))

    assert passes_floor(c("x" * 400), 400)
    assert not passes_floor(c("x" * 399), 400)
    assert passes_floor(c("x" * 100, fwd=-100), 400)
    assert not passes_floor(c("x" * 99, fwd=-100), 400)
    assert passes_floor(c("see https://t.me/x"), 0)  # min_chars 0: everything passes
    assert not passes_floor(c("see https://t.me/c/1/2 now", urls=("https://t.me/c/1/2",)), 400)


# --- RD-6: live albums survive a stop or a crash ----------------------------------------------


async def test_rd6_an_album_buffered_before_a_crash_is_submitted_after_the_restart(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel = make_chat()
    for i in range(3):
        await intake.handle_message(
            make_message(channel, text="caption" if i == 1 else "", media="photo", grouped_id=3)
        )
    await intake.handle_message(make_message(channel, text="a single post"))
    assert [c.message_id for c in fake_sorter.submitted] == [4]
    assert await rt.store.kv_get(KV.INTAKE_OPEN_ALBUMS)

    restarted = IntakeService(rt)  # the crash: the old buffer is gone
    clock.advance(timedelta(seconds=60))
    await restarted.tick()
    await restarted.tick()
    assert [(c.message_id, c.message_ids, c.media) for c in fake_sorter.submitted[1:]] == [
        (2, [1, 2, 3], "album")
    ]
    assert await rt.store.kv_get(KV.INTAKE_OPEN_ALBUMS) is None


async def test_rd6_flush_albums_submits_what_is_settling_at_a_clean_stop(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    channel = make_chat()
    for i in range(2):
        await intake.handle_message(
            make_message(channel, text="cap" if i == 0 else "", media="photo", grouped_id=8)
        )
    assert await intake.flush_albums() == 1
    assert [(c.message_id, c.media) for c in fake_sorter.submitted] == [(1, "album")]
    assert await rt.store.kv_get(KV.INTAKE_OPEN_ALBUMS) is None


async def test_rd6_an_album_whose_submit_failed_stays_recorded(
    rt: Runtime,
    intake: IntakeService,
    fake_sorter: FakeSorter,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = make_chat()
    for i in range(2):
        await intake.handle_message(
            make_message(channel, text="cap" if i == 0 else "", media="photo", grouped_id=8)
        )

    async def boom(c: Candidate) -> Post | None:
        raise RuntimeError("embedder down")

    monkeypatch.setattr(fake_sorter, "submit", boom)
    clock.advance(timedelta(seconds=5))
    await intake.tick()
    saved = await rt.store.kv_get(KV.INTAKE_OPEN_ALBUMS)
    assert saved and [p["message_id"] for p in saved[0]["parts"]] == [1, 2]
