"""The preview (DESIGN §9.6): a replay that reproduces the live decisions and writes nothing."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from tests.fakes import START, FakeClassifier, FakeLLM
from tg_curator.config import Settings
from tg_curator.db import schema
from tg_curator.db.store import Store
from tg_curator.domain import Candidate, Decision, Post, PostStatus, PreviewReport, Topic
from tg_curator.errors import ConfigError
from tg_curator.pipeline import engine as engine_mod
from tg_curator.pipeline.preview import (
    PreviewService,
    apply_overrides,
    render_text,
    split_trust_overrides,
)
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

CHANNEL = -1_001_000_000_900
A = (
    "The central bank raised its key rate to fourteen percent today, citing stubborn inflation "
    "and a weak currency."
)
A_FOOTER = A + "\n\n🔥 https://t.me/kunuz"
A_REWORDED = (
    "Today the central bank raised its key rate to fourteen percent, citing stubborn inflation "
    "and a weak currency outlook."
)
B = "Barcelona beat Real Madrid in the derby after a late goal from the substitute striker."
C = "A new metro line opened in the capital this morning with twelve stations."
D = "Шахматный турнир завершился победой семнадцатилетнего гроссмейстера."


def fake_language(text: str) -> str:
    return "ru" if re.search(r"[Ѐ-ӿ]", text) else "en"


def fake_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


@pytest.fixture(autouse=True)
def language_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "detect_language", fake_language)
    monkeypatch.setattr(engine_mod, "numbers_in", fake_numbers)


def cand(
    chat_id: int,
    message_id: int,
    text: str,
    *,
    via: str = "live",
    posted_at: datetime | None = None,
) -> Candidate:
    return Candidate(
        chat_id=chat_id,
        message_id=message_id,
        message_ids=[message_id],
        kind="post",
        posted_at=posted_at or START,
        text=text,
        html=None,
        urls=[],
        media=None,
        grouped_id=None,
        views=None,
        forwards=None,
        fwd_from_chat_id=None,
        fwd_from_message_id=None,
        noforwards=False,
        via=via,  # type: ignore[arg-type]
    )


async def dump(store: Store) -> dict[str, list[tuple[Any, ...]]]:
    """Every row of every table, for "nothing was written" assertions."""
    out = {}
    for table in schema.metadata.sorted_tables:
        rows = await store.execute(sa.select(table).order_by(*table.primary_key.columns))
        out[table.name] = [tuple(r) for r in rows] if isinstance(rows, list) else []
    return out


@pytest.fixture
async def scene(
    rt: Runtime, make_chat: Callable[..., ChatInfo]
) -> tuple[Sorter, list[int], Topic, Topic]:
    """Two topics (one without a channel), three chats, the second trusted."""
    ml = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML & AI", channel_id=CHANNEL, created_at=START)
    )
    sport = await rt.store.upsert_topic(
        Topic(id=0, key="sport", name="Sport", channel_id=None, created_at=START)
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {ml.id: 0.8, sport.id: 0.2}
    classifier.text_scores[B] = {ml.id: 0.1, sport.id: 0.7}
    classifier.text_scores[C] = {ml.id: 0.45, sport.id: 0.3}
    chats = []
    for _ in range(3):
        info = make_chat()
        await rt.store.upsert_chat(info)
        chats.append(info.id)
    await rt.store.set_chat_fields(chats[1], trust=3.0)
    rt.publisher = None
    return Sorter(rt), chats, ml, sport


async def live_run(rt: Runtime, sorter: Sorter, chats: list[int]) -> list[Post]:
    clock = rt.clock
    posts = []
    for i, (chat, text, via) in enumerate(
        [
            (chats[0], A, "live"),
            (chats[1], A_FOOTER, "live"),
            (chats[2], A_REWORDED, "live"),
            (chats[1], B, "live"),
            (chats[0], C, "live"),
            (chats[2], D, "backfill"),
            (chats[0], "", "live"),
        ]
    ):
        at = clock.now() + timedelta(minutes=i)
        post = await sorter.submit(cand(chat, i + 1, text, via=via, posted_at=at))
        assert post is not None
        posts.append(post)
    return posts


# --- replay ---------------------------------------------------------------------------------------


async def test_replay_reproduces_the_live_decisions_and_writes_nothing(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    posts = await live_run(rt, sorter, chats)
    rt.clock.advance(timedelta(hours=1))  # type: ignore[attr-defined]
    before = await dump(rt.store)

    report = await PreviewService(rt).replay(days=3)

    assert await dump(rt.store) == before
    assert report.days == 3 and report.ignored == 1
    assert report.repeats == 2 and report.unsorted == 1
    assert report.per_topic == {"ml": 2, "sport": 1}
    by_id = {d.post_id: d for d in report.decisions}
    assert list(by_id) == [p.id for p in posts[:-1]]
    for submitted in posts[:-1]:
        # the stored row, after the later repeats corroborated it — not the submit snapshot
        post = await rt.store.get_post(submitted.id)
        assert post is not None
        d = by_id[post.id]
        assert d.candidate.chat_id == post.chat_id and d.candidate.text == post.text
        assert (d.topic_id, d.dup_kind, d.duplicate_of, d.would_realtime) == (
            post.topic_id,
            post.dup_kind,
            post.duplicate_of,
            post.would_realtime,
        )
        assert (d.strength, d.corroboration) == (post.strength, post.corroboration)
        if post.id != posts[0].id:
            assert d.status == post.status
    root, exact, semantic, sport_post, borderline, backfilled = (by_id[p.id] for p in posts[:-1])
    # two other chats repeated the held root before its hold ran out: the sorter promoted it
    # (this scene wires no publisher, so the stored row could not move to the outbox)
    assert root.corroboration == 2 and root.would_realtime is True
    assert root.status == PostStatus.queued and root.hold_until is None
    assert exact.dup_kind == "exact" and semantic.dup_kind == "semantic"
    assert sport_post.status == PostStatus.tracked and sport_post.would_realtime is True
    assert borderline.status == PostStatus.unsorted and borderline.confidence == 0.45
    assert backfilled.status == PostStatus.digest


async def test_overrides_apply_to_a_copy_and_change_the_outcome(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    await live_run(rt, sorter, chats)
    before = await dump(rt.store)
    service = PreviewService(rt)

    looser = await service.replay(overrides={"sorting.confidence": "0.4"})
    assert looser.unsorted == 0 and looser.per_topic == {"ml": 3, "sport": 1}
    stricter = await service.replay(overrides={"sorting.confidence": 0.85})
    assert stricter.per_topic == {"ml": 0, "sport": 0} and stricter.unsorted == 4
    per_topic = await service.replay(overrides={"topics.ml.strictness": 0.85})
    assert per_topic.per_topic == {"ml": 0, "sport": 1}
    no_dups = await service.replay(
        overrides={"sorting.duplicate_similarity": 0.99, "sorting.realtime_strength": 1.0}
    )
    assert no_dups.repeats == 1  # the exact repeat stays, the reworded one is its own post
    assert all(d.would_realtime for d in no_dups.decisions if d.topic_id == ml.id)

    assert rt.settings.sorting.confidence == 0.5
    assert await dump(rt.store) == before
    assert rt.settings_file.path.read_text().count("confidence = 0.5") == 1


async def test_a_trust_override_changes_what_would_go_out_at_once(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    await live_run(rt, sorter, chats)
    before = await dump(rt.store)
    service = PreviewService(rt)
    now = await service.replay()
    source = next(d.candidate.chat_id for d in now.decisions if d.topic_id == sport.id)
    trusted = await service.replay(overrides={f"sources.{source}.trust": 3})
    digest_only = await service.replay(overrides={f"sources.{source}.trust": "0"})
    from_source = [d for d in trusted.decisions if d.candidate.chat_id == source and d.topic_id]
    assert from_source and all(d.would_realtime for d in from_source)
    assert not any(d.would_realtime for d in digest_only.decisions if d.candidate.chat_id == source)
    assert await dump(rt.store) == before  # nothing written, chats.trust untouched


def test_trust_overrides_are_split_and_checked() -> None:
    rest, trust = split_trust_overrides({"sources.-100123.trust": 2, "sorting.confidence": 0.6})
    assert rest == {"sorting.confidence": 0.6} and trust == {-100123: 2.0}
    with pytest.raises(ConfigError, match="0, 1, 2 or 3"):
        split_trust_overrides({"sources.-100123.trust": 5})
    with pytest.raises(ConfigError, match="numeric id"):
        split_trust_overrides({"sources.@kunuz.trust": 1})
    with pytest.raises(ConfigError, match="sources.<chat id>.trust"):
        split_trust_overrides({"sources.-100123.chat": 1})


def test_keys_the_replay_cannot_reflect_are_refused() -> None:
    settings = Settings()
    with pytest.raises(ConfigError, match="applied at intake"):
        apply_overrides(settings, {"groups.min_chars": 10})
    with pytest.raises(ConfigError, match="nothing to preview"):
        apply_overrides(settings, {"review.hour": 9})
    with pytest.raises(ConfigError, match="nothing to preview"):
        apply_overrides(settings, {"publishing.style": "forward"})
    copy, _ = apply_overrides(settings, {"digest.hour": 20, "general.timezone": "Asia/Tashkent"})
    assert copy.digest.hour == 20 and copy.general.timezone == "Asia/Tashkent"


async def test_bad_overrides_are_config_errors_naming_the_key(rt: Runtime) -> None:
    settings = Settings()
    with pytest.raises(ConfigError, match="sorting.confidence"):
        apply_overrides(settings, {"sorting.confidence": "high"})
    with pytest.raises(ConfigError, match="sorting.confidence"):
        apply_overrides(settings, {"sorting.confidence": 1.5})
    with pytest.raises(ConfigError, match="sorting.nope"):
        apply_overrides(settings, {"sorting.nope": 1})
    with pytest.raises(ConfigError, match="confidence"):
        apply_overrides(settings, {"confidence": 1})
    with pytest.raises(ConfigError, match="topics.ml.strictness"):
        apply_overrides(settings, {"topics.ml.strictness": "x"})
    with pytest.raises(ConfigError, match="topics.ml.name"):
        apply_overrides(settings, {"topics.ml.name": "New"})
    copy, strictness = apply_overrides(
        settings,
        {"sorting.hold_minutes": "10", "topics.ml.strictness": 0, "topics.x.strictness": 0.7},
    )
    assert copy.sorting.hold_minutes == 10 and settings.sorting.hold_minutes == 45
    assert strictness == {"ml": None, "x": 0.7}
    with pytest.raises(ConfigError, match="nope"):
        await PreviewService(rt).replay(topic_key="nope")


async def test_second_opinions_never_cost_a_request_in_a_preview(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    await live_run(rt, sorter, chats)
    llm: FakeLLM = rt.llm  # type: ignore[assignment]
    llm.calls.clear()
    report = await PreviewService(rt).replay(overrides={"sorting.second_opinion": True})
    assert llm.calls == []
    assert report.unsorted == 1


async def test_topic_filter_keeps_the_counts_and_narrows_the_decisions(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    await live_run(rt, sorter, chats)
    report = await PreviewService(rt).replay(topic_key="sport")
    assert report.per_topic == {"ml": 2, "sport": 1} and report.repeats == 2
    assert [d.topic_id for d in report.decisions] == [sport.id]


async def test_rejected_posts_are_replayed_but_reported_as_such(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    posts = await live_run(rt, sorter, chats)
    await rt.store.set_post_fields(posts[0].id, status=PostStatus.rejected)
    report = await PreviewService(rt).replay()
    first = report.decisions[0]
    assert first.status == PostStatus.rejected and first.topic_id == ml.id
    assert report.per_topic == {"ml": 1, "sport": 1}
    assert report.decisions[1].duplicate_of == posts[0].id  # still the root of its repeats


async def test_replay_window_and_missing_embeddings(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    old = await sorter.submit(cand(chats[0], 1, A, posted_at=START - timedelta(days=4)))
    recent = await sorter.submit(cand(chats[2], 1, A_REWORDED))
    assert old and recent and recent.duplicate_of == old.id
    await rt.store.set_post_fields(recent.id, embedding=None)
    before = await dump(rt.store)
    report = await PreviewService(rt).replay(days=3)
    assert [d.post_id for d in report.decisions] == [recent.id]
    assert report.decisions[0].status == PostStatus.held  # the root is outside the replay
    # a wider replay sees the root, but it falls out of the 3-day dedup window before the
    # repeat is decided — exactly what the live sorter's trimming would have done
    wider = await PreviewService(rt).replay(days=5)
    # (the old root's hold ran out days ago, so like the sorter's tick it is in the digest)
    assert [d.status for d in wider.decisions] == [PostStatus.digest, PostStatus.held]
    close = await PreviewService(rt).replay(days=5, overrides={"sorting.dedup_window_days": 5})
    assert [d.status for d in close.decisions] == [PostStatus.digest, PostStatus.duplicate]
    assert close.decisions[0].would_realtime is False  # one repeat: 2.0 < 3.0
    assert await dump(rt.store) == before  # the missing embedding was computed in memory only


# --- render_text ----------------------------------------------------------------------------------


async def test_render_text_prints_one_line_per_decision(
    rt: Runtime, scene: tuple[Sorter, list[int], Topic, Topic]
) -> None:
    sorter, chats, ml, sport = scene
    posts = await live_run(rt, sorter, chats)
    report = await PreviewService(rt).replay()
    text = render_text(report)
    lines = text.splitlines()
    assert len(lines) == 1 + len(report.decisions)
    assert lines[0].startswith("preview: last 3 days, 6 decisions — ml: 2, sport: 1")
    assert "repeats: 2 · unsorted: 1 · ignored: 1" in lines[0]
    # every line starts with the post's number, so "repeat of post #N" points at a line
    assert lines[1].startswith(f"#{posts[0].id}  2026-10-05 12:00  chat ")
    assert f"topic #{ml.id} 0.80  immediate: yes  [queued]" in lines[1]
    assert (
        f"repeat of post #{posts[0].id} (chat {chats[0]}: "
        "The central bank raised its key rate to…) (exact 1.00)  [duplicate]"
    ) in lines[2]
    assert f"unsorted (best topic #{ml.id} 0.45)  [unsorted]" in lines[5]
    assert "immediate: yes  [tracked]" in lines[4]

    named = render_text(
        report, topic_names={ml.id: "ML & AI", sport.id: "Sport"}, chat_names={chats[0]: "Kun.uz"}
    )
    assert "  Kun.uz  The central bank raised its key rate" in named.splitlines()[1]
    assert f"repeat of post #{posts[0].id} (Kun.uz: The central" in named.splitlines()[3]
    assert "ML & AI 0.80" in named and "Sport 0.70" in named and "best ML & AI 0.45" in named


def test_render_text_of_an_empty_report_and_a_rejected_decision() -> None:
    empty = PreviewReport(days=1, per_topic={}, repeats=0, unsorted=0, ignored=0, decisions=[])
    assert render_text(empty) == (
        "preview: last 1 days, 0 decisions — no topics · repeats: 0 · unsorted: 0 · ignored: 0"
    )
    rejected = Decision(post_id=1, candidate=cand(-1, 1, "x"), status=PostStatus.rejected)
    no_text = Decision(post_id=2, candidate=cand(-1, 2, " "), status=PostStatus.unsorted)
    report = PreviewReport(
        days=1, per_topic={"a": 0}, repeats=0, unsorted=1, ignored=0, decisions=[rejected, no_text]
    )
    lines = render_text(report).splitlines()
    assert lines[1].endswith("x  | not for me  [rejected]")
    assert lines[2].endswith("(no text)  | unsorted  [unsorted]")
