"""The pure decision engine (DESIGN §9.2): the recent index, the four questions, strength."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.fakes import START, FakeClassifier, FakeLLM
from tg_curator.config import Settings
from tg_curator.domain import Candidate, Chat, PostStatus, Topic
from tg_curator.pipeline import engine as engine_mod
from tg_curator.pipeline.engine import (
    DecisionEngine,
    RecentIndex,
    candidate_from_post,
    embedding_from_bytes,
    embedding_to_bytes,
    features_of,
)
from tg_curator.textutil import text_hash

DIM = 8
CHANNEL = -1_001_000_000_777


def fake_language(text: str) -> str:
    return "ru" if re.search(r"[Ѐ-ӿ]", text) else "en"


def fake_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


@pytest.fixture(autouse=True)
def language_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's rules are tested with deterministic stand-ins for the ml helpers."""
    monkeypatch.setattr(engine_mod, "detect_language", fake_language)
    monkeypatch.setattr(engine_mod, "numbers_in", fake_numbers)


def vec(cosine: float = 1.0, axis: int = 1) -> np.ndarray:
    """A unit vector whose cosine with ``vec(1.0)`` is exactly ``cosine``."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cosine
    v[axis] = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    return v


def cand(
    text: str = "Some post text about something.",
    *,
    chat_id: int = -1_001_000_000_001,
    message_id: int = 1,
    urls: tuple[str, ...] = (),
    fwd: tuple[int, int] | None = None,
    via: str = "live",
    posted_at: datetime = START,
) -> Candidate:
    return Candidate(
        chat_id=chat_id,
        message_id=message_id,
        message_ids=[message_id],
        kind="post",
        posted_at=posted_at,
        text=text,
        html=None,
        urls=list(urls),
        media=None,
        grouped_id=None,
        views=None,
        forwards=None,
        fwd_from_chat_id=None if fwd is None else fwd[0],
        fwd_from_message_id=None if fwd is None else fwd[1],
        noforwards=False,
        via=via,  # type: ignore[arg-type]
    )


def topic(id: int = 1, *, channel: int | None = CHANNEL, strictness: float | None = None) -> Topic:  # noqa: A002
    return Topic(
        id=id,
        key=f"t{id}",
        name=f"Topic {id}",
        channel_id=channel,
        strictness=strictness,
        created_at=START,
    )


def chat(trust: float | None = None, chat_id: int = -1_001_000_000_001) -> Chat:
    return Chat(id=chat_id, kind="channel", title="Source", first_seen_at=START, trust=trust)


def settings(**sorting: object) -> Settings:
    s = Settings()
    for key, value in sorting.items():
        setattr(s.sorting, key, value)
    return s


def make_engine(
    *,
    scores: dict[int, float] | None = None,
    index: RecentIndex | None = None,
    llm: FakeLLM | None = None,
    **sorting: object,
) -> tuple[DecisionEngine, FakeClassifier]:
    classifier = FakeClassifier(scores or {1: 0.9})
    eng = DecisionEngine(settings(**sorting), classifier, index or RecentIndex(3), llm)
    return eng, classifier


def add(
    index: RecentIndex,
    post_id: int,
    *,
    chat_id: int = -1_001_000_000_001,
    message_id: int | None = None,
    root_id: int | None = None,
    text: str = "",
    url_key: str | None = None,
    fwd_key: tuple[int, int] | None = None,
    embedding: np.ndarray | None = None,
    posted_at: datetime | None = START,
    lang: str | None = "en",
    numbers: tuple[str, ...] = (),
) -> None:
    index.add(
        post_id,
        chat_id,
        post_id if message_id is None else message_id,
        root_id,
        text_hash(text),
        url_key,
        fwd_key,
        embedding,
        posted_at=posted_at,
        lang=lang,
        numbers=numbers,
    )


# --- RecentIndex: the four kinds ---------------------------------------------------------------


def test_forward_of_a_stored_message_matches() -> None:
    index = RecentIndex(3)
    add(index, 10, chat_id=-5, message_id=77, text="original")
    match = index.match_forward((-5, 77))
    assert match is not None and match.root_id == 10 and match.chat_id == -5
    assert index.match_forward((-5, 78)) is None
    assert index.match_forward(None) is None


def test_two_forwards_of_the_same_original_match_each_other() -> None:
    index = RecentIndex(3)
    add(index, 10, text="forwarded once", fwd_key=(-9, 5))
    match = index.match_forward((-9, 5))
    assert match is not None and match.post_id == 10


def test_exact_matches_same_normalised_text_and_empty_hash_never_matches() -> None:
    index = RecentIndex(3)
    add(index, 1, text="Rate up to 14%. 🔥")
    add(index, 2, text="🔥🔥 https://t.me/x")  # normalises to nothing: hash ""
    assert index.match_exact(text_hash("rate up to 14%")) is not None
    assert index.match_exact(text_hash("https://t.me/y 🔥")) is None
    assert index.match_exact("") is None


def test_url_match_needs_cosine_of_at_least_0_60() -> None:
    index = RecentIndex(3)
    add(index, 1, text="story one", url_key="example.com/news/rate", embedding=vec(1.0))
    assert index.match_url(["example.com/news/rate"], vec(0.7)) is not None
    assert index.match_url(["example.com/news/rate"], vec(0.5)) is None
    assert index.match_url(["example.com/other"], vec(1.0)) is None


def test_footer_link_shared_by_unrelated_posts_does_not_merge_them() -> None:
    index = RecentIndex(3)
    add(index, 1, text="weather tomorrow", url_key="kun.uz/subscribe", embedding=vec(1.0))
    eng, _ = make_engine(index=index)
    feats = features_of(cand("football tonight", urls=("https://kun.uz/subscribe",)))
    assert feats.url_keys == ("kun.uz/subscribe",)
    assert eng.repeat(feats, vec(0.0)) is None


def test_semantic_same_language_needs_0_90_and_cross_language_0_86() -> None:
    index = RecentIndex(3)
    add(index, 1, text="english story", embedding=vec(1.0), lang="en")
    assert index.match_semantic(vec(0.88), lang="en") is None
    assert index.match_semantic(vec(0.91), lang="en") is not None
    assert index.match_semantic(vec(0.88), lang="ru") is not None
    assert index.match_semantic(vec(0.85), lang="ru") is None


def test_unknown_language_uses_the_stricter_threshold() -> None:
    index = RecentIndex(3)
    add(index, 1, text="x", embedding=vec(1.0), lang=None)
    assert index.match_semantic(vec(0.88), lang="ru") is None
    assert index.match_semantic(vec(0.88), lang=None) is None
    assert index.match_semantic(vec(0.95), lang=None) is not None


def test_number_veto_two_disjoint_number_sets_below_0_97() -> None:
    index = RecentIndex(3)
    add(index, 1, text="3-1 at 90", embedding=vec(1.0), numbers=("3", "1", "90"))
    assert index.match_semantic(vec(0.95), lang="en", numbers=("2", "0")) is None
    assert index.match_semantic(vec(0.95), lang="en", numbers=("2", "90")) is not None
    assert index.match_semantic(vec(0.98), lang="en", numbers=("2", "0")) is not None
    assert index.match_semantic(vec(0.95), lang="en", numbers=("2",)) is not None


def test_semantic_picks_the_best_surviving_entry_not_just_the_top_cosine() -> None:
    index = RecentIndex(3)
    add(index, 1, text="vetoed", embedding=vec(1.0), numbers=("1", "2"))
    add(index, 2, text="ok", embedding=vec(0.95, axis=2), numbers=())
    match = index.match_semantic(vec(0.97, axis=2), lang="en", numbers=("8", "9"))
    assert match is not None and match.post_id == 2


def test_chains_resolve_to_the_root_in_one_step() -> None:
    index = RecentIndex(3)
    add(index, 1, text="the story")
    add(index, 2, text="the story again", root_id=1)
    add(index, 3, text="the story again", root_id=1)
    match = index.match_exact(text_hash("the story again"))
    assert match is not None and match.post_id == 2 and match.root_id == 1


def test_remove_older_than_trims_and_keeps_matching_working() -> None:
    index = RecentIndex(3)
    add(index, 1, text="old", embedding=vec(1.0), posted_at=START - timedelta(days=4))
    add(index, 2, text="new", embedding=vec(1.0, axis=2), posted_at=START, numbers=("5",))
    add(index, 3, text="undated", embedding=None, posted_at=None)
    assert len(index) == 3
    index.remove_older_than(START - timedelta(days=3))
    assert len(index) == 2 and 1 not in index and 2 in index and 3 in index
    assert index.match_exact(text_hash("old")) is None
    match = index.match_semantic(vec(1.0, axis=2), lang="en")
    assert match is not None and match.post_id == 2
    index.remove_older_than(START - timedelta(days=3))  # nothing to do: no rebuild
    assert len(index) == 2


def test_add_is_idempotent_and_missing_embeddings_never_match() -> None:
    index = RecentIndex(3)
    add(index, 1, text="x", embedding=None)
    add(index, 1, text="x", embedding=vec(1.0))
    assert len(index) == 1
    assert index.match_semantic(vec(1.0), lang="en") is None
    add(index, 2, text="y", embedding=vec(1.0))
    assert index.match_semantic(vec(1.0), lang="en") is not None


def test_empty_index_matches_nothing() -> None:
    index = RecentIndex(3)
    assert index.match_semantic(vec(1.0)) is None
    assert index.match_url(["a.com/x"], vec(1.0)) is None


def test_index_grows_past_its_initial_capacity() -> None:
    index = RecentIndex(3)
    rng = np.random.default_rng(1)
    for i in range(50):
        v = rng.standard_normal(DIM).astype(np.float32)
        add(index, i, text=f"post {i}", embedding=v / np.linalg.norm(v))
    add(index, 99, text="needle", embedding=vec(1.0))
    match = index.match_semantic(vec(1.0), lang="en")
    assert match is not None and match.post_id == 99


# --- DecisionEngine ------------------------------------------------------------------------------


async def test_a_repeat_stops_at_question_one() -> None:
    index = RecentIndex(3)
    add(index, 5, chat_id=-2, text="same text here")
    eng, classifier = make_engine(index=index)
    d = await eng.decide(cand("Same text here!"), vec(0.0), chat(), [topic()], START)
    assert d.status == PostStatus.duplicate
    assert d.duplicate_of == 5 and d.dup_kind == "exact" and d.dup_score == 1.0
    assert d.topic_id is None and d.strength is None


async def test_dedup_order_forward_then_exact_then_url_then_semantic() -> None:
    index = RecentIndex(3)
    add(index, 1, chat_id=-2, message_id=9, text="a", embedding=vec(0.0, axis=3))
    add(index, 2, chat_id=-3, text="same words", embedding=vec(0.0, axis=4))
    add(index, 3, chat_id=-4, text="b", url_key="example.com/p", embedding=vec(1.0))
    add(index, 4, chat_id=-5, text="c", embedding=vec(0.0, axis=5))
    eng, _ = make_engine(index=index)
    forward = await eng.decide(
        cand("same words", urls=("https://example.com/p",), fwd=(-2, 9)),
        vec(1.0),
        chat(),
        [topic()],
        START,
    )
    assert (forward.dup_kind, forward.duplicate_of) == ("forward", 1)
    exact = await eng.decide(
        cand("same words", urls=("https://example.com/p",)), vec(1.0), chat(), [topic()], START
    )
    assert (exact.dup_kind, exact.duplicate_of) == ("exact", 2)
    url = await eng.decide(
        cand("other words", urls=("https://example.com/p",)), vec(1.0), chat(), [topic()], START
    )
    assert (url.dup_kind, url.duplicate_of) == ("url", 3)
    semantic = await eng.decide(cand("other words"), vec(0.0, axis=5), chat(), [topic()], START)
    assert (semantic.dup_kind, semantic.duplicate_of) == ("semantic", 4)


async def test_cross_language_repeat_uses_the_lower_threshold() -> None:
    index = RecentIndex(3)
    add(index, 1, text="The bank raised the rate", embedding=vec(1.0), lang="en")
    eng, _ = make_engine(index=index)
    russian = cand("Банк повысил ставку")
    d = await eng.decide(russian, vec(0.88), chat(), [topic()], START)
    assert d.status == PostStatus.duplicate and d.dup_kind == "semantic"
    same_lang = await eng.decide(cand("The bank raised"), vec(0.88), chat(), [topic()], START)
    assert same_lang.status != PostStatus.duplicate


async def test_translation_with_locale_formatted_thousands_is_a_semantic_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SORT-2: "1,500 ... 2,300" and "1 500 ... 2 300" share their numbers, so no veto."""
    from tg_curator.ml import embedder

    monkeypatch.setattr(engine_mod, "numbers_in", embedder.numbers_in)
    english = "Floods displaced 1,500 families and damaged 2,300 homes"
    russian = "Наводнение оставило без крова 1 500 семей и разрушило 2 300 домов"
    index = RecentIndex(3)
    feats = features_of(cand(english))
    assert len(feats.numbers) == 2
    add(
        index,
        1,
        chat_id=-2,
        text=english,
        embedding=vec(1.0),
        lang="en",
        numbers=tuple(feats.numbers),
    )
    add(index, 2, chat_id=-3, text="later copy", embedding=vec(0.5, axis=3), root_id=1)
    eng, _ = make_engine(index=index)
    d = await eng.decide(cand(russian), vec(0.88), chat(), [topic()], START)
    assert d.status == PostStatus.duplicate and d.dup_kind == "semantic"
    assert d.duplicate_of == 1 and d.dup_score == 0.88
    other_story = await eng.decide(
        cand("Наводнение: 1 700 семей и 2 900 домов"), vec(0.88), chat(), [topic()], START
    )
    assert other_story.status != PostStatus.duplicate  # the veto still splits different figures


async def test_number_veto_reaches_the_engine_through_features() -> None:
    index = RecentIndex(3)
    add(index, 1, text="Score 3-1", embedding=vec(1.0), numbers=("3", "1"))
    eng, _ = make_engine(index=index)
    d = await eng.decide(cand("Score 2 to 0"), vec(0.95), chat(), [topic()], START)
    assert d.status != PostStatus.duplicate


async def test_unsorted_below_the_threshold_keeps_the_best_score() -> None:
    eng, _ = make_engine(scores={1: 0.4, 2: 0.3})
    d = await eng.decide(cand(), vec(), chat(), [topic(1), topic(2)], START)
    assert d.status == PostStatus.unsorted and d.topic_id is None
    assert d.confidence == 0.4 and [s.topic_id for s in d.topic_scores] == [1, 2]
    assert d.strength is None and d.would_realtime is False


async def test_per_topic_strictness_overrides_the_global_confidence() -> None:
    eng, _ = make_engine(scores={1: 0.6})
    strict = await eng.decide(cand(), vec(), chat(), [topic(strictness=0.7)], START)
    assert strict.status == PostStatus.unsorted
    loose = await eng.decide(cand(), vec(), chat(), [topic(strictness=0.55)], START)
    assert loose.topic_id == 1
    zero_means_global = await eng.decide(cand(), vec(), chat(), [topic(strictness=0.0)], START)
    assert zero_means_global.topic_id == 1


async def test_scores_of_inactive_or_unknown_topics_are_ignored() -> None:
    eng, _ = make_engine(scores={1: 0.95, 2: 0.9, 3: 0.8, 4: 0.7})
    inactive = topic(1)
    inactive.active = False
    d = await eng.decide(cand(), vec(), chat(), [inactive, topic(2), topic(3), topic(4)], START)
    assert d.topic_id == 2
    assert [s.topic_id for s in d.topic_scores] == [2, 3, 4]
    none = await eng.decide(cand(), vec(), chat(), [], START)
    assert none.status == PostStatus.unsorted and none.topic_scores == []


async def test_second_opinion_only_for_borderline_posts_with_the_switch_on() -> None:
    llm = FakeLLM(opinion=True)
    eng, _ = make_engine(scores={1: 0.45}, llm=llm, second_opinion=True)
    d = await eng.decide(cand("borderline"), vec(), chat(), [topic()], START)
    assert d.topic_id == 1 and len(llm.calls) == 1
    assert llm.calls[0][1]["text"] == "borderline" and llm.calls[0][1]["topic"].id == 1

    llm.opinion = False
    d = await eng.decide(cand(), vec(), chat(), [topic()], START)
    assert d.status == PostStatus.unsorted

    # The runner-up topics the classifier also scored are handed over as the competition.
    rivals, _ = make_engine(scores={1: 0.45, 2: 0.3, 3: 0.2, 4: 0.1}, llm=llm, second_opinion=True)
    await rivals.decide(cand(), vec(), chat(), [topic(1), topic(2), topic(3), topic(4)], START)
    assert [t.id for t in llm.calls[-1][1]["competing"]] == [2, 3]

    far = DecisionEngine(
        settings(second_opinion=True), FakeClassifier({1: 0.3}), RecentIndex(3), llm
    )
    calls = len(llm.calls)
    assert (await far.decide(cand(), vec(), chat(), [topic()], START)).status == PostStatus.unsorted
    assert len(llm.calls) == calls


async def test_no_second_opinion_when_switched_off_or_without_a_model() -> None:
    llm = FakeLLM(opinion=True)
    off, _ = make_engine(scores={1: 0.45}, llm=llm)
    assert (await off.decide(cand(), vec(), chat(), [topic()], START)).status == PostStatus.unsorted
    assert llm.calls == []
    disabled = FakeLLM(enabled=False)
    eng, _ = make_engine(scores={1: 0.45}, llm=disabled, second_opinion=True)
    assert (await eng.decide(cand(), vec(), chat(), [topic()], START)).status == PostStatus.unsorted
    assert disabled.calls == []
    none, _ = make_engine(scores={1: 0.45}, llm=None, second_opinion=True)
    assert (
        await none.decide(cand(), vec(), chat(), [topic()], START)
    ).status == PostStatus.unsorted


# --- strength ------------------------------------------------------------------------------------


def test_strength_defaults_give_the_spec_behaviour() -> None:
    eng, _ = make_engine()
    assert eng.strength(cand(), chat(None), 0) == (1.0, False)
    assert eng.strength(cand(), chat(None), 2) == (3.0, True)
    assert eng.strength(cand(), chat(3.0), 0) == (3.0, True)
    assert eng.strength(cand(), chat(2.0), 1) == (3.0, True)
    assert eng.strength(cand(), chat(2.0), 0) == (2.0, False)
    assert eng.strength(cand(), chat(0.0), 5) == (5.0, False)
    assert eng.strength(cand(), None, 2) == (3.0, True)


def test_strength_bonuses_for_links_and_length() -> None:
    eng, _ = make_engine()
    linked = cand("x", urls=("https://example.com/a",))
    assert eng.strength(linked, chat(), 0)[0] == 1.25
    telegram_only = cand("x", urls=("https://t.me/kunuz/5",))
    assert eng.strength(telegram_only, chat(), 0)[0] == 1.0
    long = cand("w" * 600)
    assert eng.strength(long, chat(), 0)[0] == 1.25
    both = cand("w" * 600, urls=("https://example.com/a",))
    assert eng.strength(both, chat(), 1) == (2.5, False)


def test_strength_reads_the_settings_it_was_built_with() -> None:
    eng, _ = make_engine(realtime_strength=2.0, corroboration_weight=0.5, neutral_trust=1.5)
    assert eng.strength(cand(), chat(None), 1) == (2.0, True)
    assert eng.strength(cand(), chat(1.0), 2) == (2.0, False)  # below neutral: never


# --- routing (question 4) ------------------------------------------------------------------------


async def test_topic_without_channel_is_tracked() -> None:
    eng, _ = make_engine()
    d = await eng.decide(cand(), vec(), chat(3.0), [topic(channel=None)], START)
    assert d.status == PostStatus.tracked and d.would_realtime is True and d.hold_until is None


async def test_trust_levels_route_queued_held_digest() -> None:
    eng, _ = make_engine()
    immediate = await eng.decide(cand(), vec(), chat(3.0), [topic()], START)
    assert immediate.status == PostStatus.queued and immediate.strength == 3.0
    held = await eng.decide(cand(), vec(), chat(None), [topic()], START)
    assert held.status == PostStatus.held
    assert held.hold_until == START + timedelta(minutes=45)
    mild = await eng.decide(cand(), vec(), chat(2.0), [topic()], START)
    assert mild.status == PostStatus.held
    digest_only = await eng.decide(cand(), vec(), chat(0.0), [topic()], START)
    assert digest_only.status == PostStatus.digest and digest_only.hold_until is None


async def test_backfilled_posts_go_to_digest_inside_the_window_else_dropped() -> None:
    eng, _ = make_engine()
    inside = cand(via="backfill", posted_at=START - timedelta(hours=25))
    d = await eng.decide(inside, vec(), chat(3.0), [topic()], START)
    assert d.status == PostStatus.digest and d.would_realtime is True
    outside = cand(via="backfill", posted_at=START - timedelta(hours=27))
    d = await eng.decide(outside, vec(), chat(3.0), [topic()], START)
    assert d.status == PostStatus.dropped
    no_channel = await eng.decide(inside, vec(), chat(), [topic(channel=None)], START)
    assert no_channel.status == PostStatus.tracked


async def test_realtime_false_sorts_a_live_post_like_a_backfill() -> None:
    eng, _ = make_engine()
    d = await eng.classify(cand(), vec(), chat(3.0), [topic()], START, realtime=False)
    assert d.status == PostStatus.digest
    d = await eng.decide(cand(), vec(), chat(3.0), [topic()], START, realtime=False)
    assert d.status == PostStatus.digest


# --- helpers ------------------------------------------------------------------------------------


def test_features_and_embedding_round_trip() -> None:
    feats = features_of(
        cand(
            "Ставка 14,5% и 3 пункта https://example.com/a?utm_source=x",
            urls=("https://example.com/a?utm_source=x", "https://t.me/kunuz"),
            fwd=(-7, 3),
        )
    )
    assert feats.lang == "ru" and feats.fwd_key == (-7, 3)
    assert feats.url_keys == ("example.com/a",) and feats.numbers == {"14,5", "3"}
    v = vec(0.6)
    assert np.allclose(embedding_from_bytes(embedding_to_bytes(v)), v)
    assert embedding_from_bytes(None) is None and embedding_from_bytes(b"") is None


def test_candidate_from_post_keeps_what_the_engine_reads() -> None:
    from tg_curator.domain import Post

    post = Post(
        id=4,
        chat_id=-1,
        message_id=2,
        kind="unit",
        message_ids=[2, 3],
        posted_at=START,
        ingested_at=START,
        via="backfill",
        text="t",
        text_hash="h",
        urls=["https://a.com/x"],
        status=PostStatus.unsorted,
        fwd_from_chat_id=-9,
        fwd_from_message_id=8,
    )
    c = candidate_from_post(post)
    assert (c.chat_id, c.message_id, c.message_ids, c.kind, c.via) == (
        -1,
        2,
        [2, 3],
        "unit",
        "backfill",
    )
    assert c.urls == ["https://a.com/x"] and (c.fwd_from_chat_id, c.fwd_from_message_id) == (-9, 8)


def test_ml_embedder_exposes_the_two_language_helpers() -> None:
    embedder = pytest.importorskip("tg_curator.ml.embedder")
    helpers: list[Callable[[str], object]] = [embedder.detect_language, embedder.numbers_in]
    assert all(callable(h) for h in helpers)
    assert isinstance(embedder.detect_language("hello world"), str)
    assert set(map(str, embedder.numbers_in("1 and 2"))) >= {"1", "2"}
