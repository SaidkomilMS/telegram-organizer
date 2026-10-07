"""The local models without the real models (DESIGN §10, §16): the deterministic
``HashEmbedder``, a base model built from the seeds, the per-user layer, discovery maths and
the model factory. Nothing here downloads, trains on public data or loads onnxruntime."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import PROVISIONAL_CATEGORIES
from tg_curator.contracts import Embedder, TopicClassifier
from tg_curator.domain import Example, Post, PostStatus, Topic
from tg_curator.ml import categories, models
from tg_curator.ml.base_model import (
    FLOOR,
    BaseModel,
    fit_head,
    load_for,
    shipped_path,
    topic_confidence,
)
from tg_curator.ml.classifier import (
    DEFAULT_CONFIDENCE,
    FORCED_SCORE,
    VETOED_SCORE,
    LocalTopicClassifier,
)
from tg_curator.ml.cluster import average_linkage, competition, find_clusters
from tg_curator.ml.embedder import (
    HashEmbedder,
    clean,
    detect_language,
    is_uz_cyrillic,
    numbers_in,
    numbers_shared,
    prepare,
    uz_cyr_to_lat,
)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)

CRYPTO_EXAMPLES = [
    "Bitcoin price hits a new high; ETF inflows and liquidations rise",
    "Crypto exchange hacked, millions in tokens stolen overnight",
    "Ethereum and altcoins fall as stablecoin regulation tightens",
    "Биткоин обновил максимум; притоки в ETF и ликвидации растут",
]
SPORT_EXAMPLES = [
    "Full time: the match ended 2-1, goals and the league table updated",
    "Tennis final result; Grand Slam title decided in five sets",
    "UFC fighter defends the title; boxing and the Olympics ahead",
    "Матч закончился со счётом 2:1, голы и турнирная таблица",
]
SPORT_POST = (
    "The striker scored twice in the final minutes and the match ended 3-1 with the league "
    "table changing again tonight"
)
SPORT_POST_NEAR = SPORT_POST.replace("tonight", "today")
UNRELATED_POST = "Recipe of the day: five habits for better sleep and a tidy home"


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def base(embedder: HashEmbedder) -> BaseModel:
    return BaseModel.from_embedder(embedder)


@pytest.fixture
def make_topic() -> Callable[..., Topic]:
    def factory(
        topic_id: int, key: str, *, category: str | None = None, description: str | None = None
    ) -> Topic:
        return Topic(
            id=topic_id,
            key=key,
            name=key,
            category=category,
            description=description,
            created_at=NOW,
        )

    return factory


@pytest.fixture
def make_example(embedder: HashEmbedder) -> Callable[..., Example]:
    counter = {"n": 0}

    def factory(
        topic_id: int | None,
        text: str,
        *,
        kind: str = "example",
        wrong_topic_id: int | None = None,
        post_id: int | None = None,
        embedding: bytes | None = None,
    ) -> Example:
        counter["n"] += 1
        if embedding is None:
            embedding = embedder.embed([text])[0].astype("<f4").tobytes()
        return Example(
            id=counter["n"],
            topic_id=topic_id,
            kind=kind,  # type: ignore[arg-type]
            post_id=post_id,
            wrong_topic_id=wrong_topic_id,
            text=text,
            embedding=embedding,
            created_at=NOW,
        )

    return factory


def _predict(clf: LocalTopicClassifier, embedder: Embedder, text: str) -> dict[int, float]:
    return {s.topic_id: s.confidence for s in clf.predict(text, embedder.embed([text])[0])}


# --- text helpers ----------------------------------------------------------------------------


def test_clean_strips_links_tags_and_emoji() -> None:
    text = "⚡️ Rate cut! https://example.com/x #fed @channel t.me/foo  more   text 😂"
    assert clean(text) == "Rate cut! more text"


def test_uzbek_cyrillic_is_transliterated_only_when_uzbek() -> None:
    uz = "Марказий банк асосий ставкани ўзгартирди, қарор ҳақида"
    assert is_uz_cyrillic(uz)
    assert prepare(uz) == "Markaziy bank asosiy stavkani o‘zgartirdi, qaror haqida"
    ru = "Центробанк изменил ключевую ставку"
    assert not is_uz_cyrillic(ru)
    assert prepare(ru) == ru
    assert uz_cyr_to_lat("Ёш Ўзбекистон") == "Yosh O‘zbekiston"


def test_detect_language() -> None:
    assert detect_language("Центробанк изменил ключевую ставку") == "ru"
    assert detect_language("Марказий банк ставкани ўзгартирди, қарор ҳақида") == "uz"
    assert detect_language("The central bank changed its key rate") == "en"
    assert detect_language("Markaziy bank va hukumat bu qaror bilan yilni yakunladi") == "uz"
    assert detect_language("12 345 — 🙂") == "other"
    assert detect_language("") == "other"


def test_numbers_in() -> None:
    assert numbers_in("rate 3,75% to 4.00% on 25 bp; score 2:1") == {"3.75", "4.00", "25", "2:1"}
    assert numbers_in("no digits here") == set()
    assert numbers_in("12 500 so‘m") == {"12500"}


def test_numbers_in_matches_thousands_across_locales() -> None:
    """SORT-2: an EN/RU translation writes the same figures as 1,500 / 1 500 / 1500."""
    english = numbers_in("Floods displaced 1,500 families and damaged 2,300 homes")
    assert len(english) == 2  # one element per source number, whatever its readings
    for other in (
        "Наводнение: 1 500 семей и 2 300 домов",
        "1\u00a0500 и 2\u202f300",
        "1500 и 2300",
        "1.500 и 2.300",
    ):
        assert numbers_shared(english, numbers_in(other)), other
    assert numbers_in("1 500 и 2 300") == {"1500", "2300"}
    assert numbers_shared(numbers_in("3,5 and 7"), numbers_in("3.5 и 8"))
    assert numbers_in("3,5") == numbers_in("3.5") == {"3.5"}
    assert not numbers_shared(numbers_in("1,500"), numbers_in("1,600"))
    assert numbers_in("1,500,000 or 1 500 000 or 1.500.000") == {"1500000"}
    assert numbers_in("1,500.50 и 1 500,50") == {"1500.50"}
    assert numbers_in("в 2024 1 500") == {"2024", "1500"}
    assert numbers_in("07.10.2026") == {"07.10.2026"}


# --- HashEmbedder ----------------------------------------------------------------------------


def test_hash_embedder_is_a_deterministic_unit_norm_embedder(embedder: HashEmbedder) -> None:
    assert isinstance(embedder, Embedder)
    vectors = embedder.embed([SPORT_POST, SPORT_POST, SPORT_POST_NEAR, UNRELATED_POST, ""])
    assert vectors.shape == (5, embedder.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(vectors[0], vectors[1])
    assert np.array_equal(vectors[0], HashEmbedder().embed([SPORT_POST])[0])
    assert float(vectors[0] @ vectors[2]) >= 0.9  # one word changed
    assert float(vectors[0] @ vectors[3]) < 0.5
    assert embedder.embed([]).shape == (0, embedder.dim)


def test_hash_embedder_ignores_links_emoji_and_script(embedder: HashEmbedder) -> None:
    a, b = embedder.embed([SPORT_POST, "🔥 " + SPORT_POST + " https://t.me/x/1 #football"])
    assert float(a @ b) > 0.999
    uz_cyr, uz_lat = embedder.embed(
        [
            "Марказий банк ставкани ўзгартирди, қарор ҳақида",
            "Markaziy bank stavkani o‘zgartirdi, qaror haqida",
        ]
    )
    assert float(uz_cyr @ uz_lat) > 0.999


# --- categories and the base model -----------------------------------------------------------


def test_categories_are_the_twenty_of_the_design() -> None:
    keys = [c.key for c in categories.all()]
    assert keys == list(PROVISIONAL_CATEGORIES) == list(categories.KEYS)
    assert all(c.label for c in categories.all())
    texts, seed_keys = categories.seed_texts()
    assert len(texts) == 180 and len(set(texts)) == 180
    assert all(len(categories.SEEDS[k]) == 9 for k in categories.KEYS)
    assert seed_keys[:9] == ["tech"] * 9
    assert categories.label("real_estate") == "Real estate"
    assert "mortgages" in categories.describe("real_estate")
    assert all(a in categories.KEYS and b in categories.KEYS for a, b in categories.NEIGHBOUR_PAIRS)


def test_base_model_probabilities_and_scores(base: BaseModel, embedder: HashEmbedder) -> None:
    assert base.keys == categories.KEYS and base.dim == embedder.dim
    probs = base.probabilities(embedder.embed(["Bitcoin price hits a new high; ETF inflows"])[0])
    assert probs.shape == (20,) and abs(float(probs.sum()) - 1.0) < 1e-5
    scores = base.scores(embedder.embed(["Bitcoin price hits a new high; ETF inflows"])[0])
    assert scores[0][0] == "crypto" and scores[0][1] > 0.5
    assert [s for _, s in scores] == sorted((s for _, s in scores), reverse=True)
    assert base.scores(embedder.embed(["Матч закончился со счётом 2:1"])[0])[0][0] == "sport"


def test_base_model_roundtrips_through_npz(base: BaseModel, tmp_path: Path) -> None:
    base.save(tmp_path / "m.npz")
    loaded = BaseModel.load(tmp_path / "m.npz")
    x = HashEmbedder().embed(["Vacancy: developer wanted, salary range"])[0]
    assert loaded.keys == base.keys and loaded.embedder_id == base.embedder_id
    assert np.allclose(loaded.probabilities(x), base.probabilities(x))


def test_shipped_model_exists_and_matches_the_real_embedder() -> None:
    assert shipped_path().exists()
    shipped = BaseModel.load()
    assert shipped.keys == categories.KEYS
    assert shipped.dim == 384 and shipped.head_w.shape == (384, 20)
    assert shipped.embedder_id == models.EMBEDDER_ID
    assert shipped_path().stat().st_size < 200_000  # it is a small data file, not a download


def test_load_for_falls_back_to_seed_prototypes_for_another_embedder(
    embedder: HashEmbedder,
) -> None:
    model = load_for(embedder)
    assert model.embedder_id == embedder.id and model.dim == embedder.dim


def test_fit_head_is_deterministic_and_fits_its_training_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.repeat(np.arange(3), 20)
    x[:, :3] += np.eye(3, dtype=np.float32)[y] * 3
    w1, b1 = fit_head(x, y, 3)
    w2, b2 = fit_head(x, y, 3)
    assert np.array_equal(w1, w2) and np.array_equal(b1, b2)
    assert ((x @ w1 + b1).argmax(1) == y).mean() > 0.95


def test_topic_confidence_puts_the_base_floor_on_the_default_threshold() -> None:
    assert topic_confidence(0.0) == 0.0 and topic_confidence(1.0) == 1.0
    assert topic_confidence(FLOOR) == pytest.approx(DEFAULT_CONFIDENCE)
    assert topic_confidence(FLOOR - 1e-6) < DEFAULT_CONFIDENCE <= topic_confidence(FLOOR + 1e-9)
    grid = np.linspace(0.0, 1.0, 101)
    mapped = topic_confidence(grid)
    assert isinstance(mapped, np.ndarray) and mapped.shape == grid.shape
    assert np.all(np.diff(mapped) > 0)  # strictly monotone: the base model's ranking is kept
    assert np.all((mapped >= 0.0) & (mapped <= 1.0))
    assert isinstance(topic_confidence(np.float32(0.5)), float)
    assert topic_confidence(0.5, floor=0.4) == pytest.approx(0.5 + 0.5 * 0.1 / 0.6)


# --- the per-user layer ----------------------------------------------------------------------


def test_classifier_is_a_topic_classifier(embedder: HashEmbedder, base: BaseModel) -> None:
    assert isinstance(LocalTopicClassifier(embedder, base), TopicClassifier)


def test_category_only_topics_get_sensible_scores(
    embedder: HashEmbedder, base: BaseModel, make_topic: Callable[..., Topic]
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    clf.reload(
        [make_topic(1, "crypto", category="crypto"), make_topic(2, "sport", category="sport")], []
    )
    crypto = _predict(clf, embedder, "Bitcoin price hits a new high as ETF inflows grow")
    assert crypto[1] >= DEFAULT_CONFIDENCE > crypto[2]
    crypto_ru = _predict(clf, embedder, "Биткоин обновил максимум; притоки в ETF")
    assert crypto_ru[1] >= DEFAULT_CONFIDENCE > crypto_ru[2]
    sport = _predict(clf, embedder, "Full time: the match ended 2-1, goals and the league table")
    assert sport[2] >= DEFAULT_CONFIDENCE > sport[1]
    none = _predict(clf, embedder, UNRELATED_POST)
    assert max(none.values()) < DEFAULT_CONFIDENCE  # "none of the topics"
    # with no posts of its own a topic is its category, calibrated onto the threshold
    for text in (SPORT_POST, UNRELATED_POST, "Bitcoin price hits a new high as ETF inflows grow"):
        x = embedder.embed([text])[0]
        probs = base.probabilities(x)
        got = _predict(clf, embedder, text)
        assert got[1] == pytest.approx(topic_confidence(probs[base.keys.index("crypto")]))
        assert got[2] == pytest.approx(topic_confidence(probs[base.keys.index("sport")]))


def test_category_only_topics_survive_examples_for_another_topic(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    """The template's first step: examples for the topic without a category, nothing else.
    The fit must neither silence the category-only topic nor send every post to the topic
    that has examples (the earlier layer did both: crypto 0.0, sport 1.0 for every post)."""
    suffixes = ("today", "this morning", "- details inside")
    examples = [make_example(2, f"{t} {s}") for t in SPORT_EXAMPLES[:3] for s in suffixes]
    alone = LocalTopicClassifier(embedder, base)
    alone.reload([make_topic(1, "crypto", category="crypto")], [])
    clf = LocalTopicClassifier(embedder, base)
    clf.reload([make_topic(1, "crypto", category="crypto"), make_topic(2, "sport")], examples)
    for text in ("Bitcoin price hits a new high as ETF inflows grow", "Биткоин обновил максимум"):
        scores = _predict(clf, embedder, text)
        assert scores[1] >= DEFAULT_CONFIDENCE > scores[2]
        assert scores[1] == _predict(alone, embedder, text)[1]  # untouched by the other fit
    scores = _predict(clf, embedder, f"{SPORT_EXAMPLES[1]} tonight")
    assert scores[2] >= DEFAULT_CONFIDENCE > scores[1]
    for text in (UNRELATED_POST, "Parliament passed the budget law after a long debate"):
        assert max(_predict(clf, embedder, text).values()) < DEFAULT_CONFIDENCE


def test_a_category_topic_moves_to_the_layer_with_its_first_own_post(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    topics = [make_topic(1, "crypto", category="crypto"), make_topic(2, "sport", category="sport")]
    clf = LocalTopicClassifier(embedder, base)
    clf.reload(topics, [])
    probe = "Ethereum and altcoins fall as stablecoin regulation tightens this week"
    x = embedder.embed([probe])[0]
    calibrated = topic_confidence(base.probabilities(x)[base.keys.index("sport")])
    assert _predict(clf, embedder, probe)[2] == pytest.approx(calibrated)
    # the owner confirms a crypto-looking post in sport: it and its near-twin flip at once,
    # and sport is now scored by the layer (its own post counts), crypto still by its category
    clf.learn(make_example(2, probe, kind="correction", wrong_topic_id=1, post_id=5))
    for text in (probe, f"{probe} now"):
        after = _predict(clf, embedder, text)
        assert after[2] >= FORCED_SCORE and after[1] <= VETOED_SCORE
    unrelated = _predict(clf, embedder, UNRELATED_POST)
    probs = base.probabilities(embedder.embed([UNRELATED_POST])[0])
    assert unrelated[1] == pytest.approx(topic_confidence(probs[base.keys.index("crypto")]))
    assert unrelated[2] != pytest.approx(topic_confidence(probs[base.keys.index("sport")]))
    assert max(unrelated.values()) < DEFAULT_CONFIDENCE


def test_predict_scores_every_active_topic_best_first_in_unit_range(
    embedder: HashEmbedder, base: BaseModel, make_topic: Callable[..., Topic]
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    inactive = make_topic(3, "old", category="tech")
    inactive.active = False
    clf.reload(
        [
            make_topic(1, "crypto", category="crypto"),
            make_topic(2, "sport", category="sport"),
            make_topic(4, "plain"),
            make_topic(5, "described", description="Housing prices and mortgages in Tashkent"),
            inactive,
        ],
        [],
    )
    for text in (SPORT_POST, UNRELATED_POST, "", "12345"):
        scores = clf.predict(text, embedder.embed([text])[0])
        assert [s.topic_id for s in scores] and set(s.topic_id for s in scores) == {1, 2, 4, 5}
        assert all(0.0 <= s.confidence <= 1.0 for s in scores)
        assert [s.confidence for s in scores] == sorted(
            (s.confidence for s in scores), reverse=True
        )
    assert clf.predict(SPORT_POST, embedder.embed([SPORT_POST])[0])[0].topic_id == 2
    empty = LocalTopicClassifier(embedder, base)
    assert empty.predict(SPORT_POST, embedder.embed([SPORT_POST])[0]) == []


def test_category_scores_cover_every_category_best_first(
    embedder: HashEmbedder, base: BaseModel
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    scores = clf.category_scores(
        "Vacancy: developer wanted, send your CV", embedder.embed(["x"])[0]
    )
    assert [k for k, _ in scores] != list(categories.KEYS)  # sorted, not in list order
    assert sorted(k for k, _ in scores) == sorted(categories.KEYS)
    assert abs(sum(s for _, s in scores) - 1.0) < 1e-5
    text = "Vacancy: developer wanted, salary range, send your CV"
    assert clf.category_scores(text, embedder.embed([text])[0])[0][0] == "jobs"


def test_examples_teach_a_topic_and_one_correction_flips_near_identical_posts(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    topics = [make_topic(1, "crypto", category="crypto"), make_topic(2, "sport", category="sport")]
    examples = [make_example(1, t) for t in CRYPTO_EXAMPLES]
    examples += [make_example(2, t) for t in SPORT_EXAMPLES]
    clf.reload(topics, examples)
    before = _predict(clf, embedder, SPORT_POST)
    assert before[2] > before[1]
    # the owner moves the post to crypto: the post itself and a near-identical repost flip at once
    clf.learn(make_example(1, SPORT_POST, kind="correction", wrong_topic_id=2, post_id=77))
    for text in (SPORT_POST, SPORT_POST_NEAR):
        after = _predict(clf, embedder, text)
        assert after[1] >= FORCED_SCORE and after[2] <= VETOED_SCORE
    # a second correction of the same post replaces the first: "not for me" vetoes every topic
    clf.learn(make_example(None, SPORT_POST, kind="correction", wrong_topic_id=1, post_id=77))
    for text in (SPORT_POST, SPORT_POST_NEAR):
        assert max(_predict(clf, embedder, text).values()) <= VETOED_SCORE
    assert sum(e.post_id == 77 for e in clf._examples) == 1
    # an unrelated post is untouched by the override and stays unsorted
    assert max(_predict(clf, embedder, UNRELATED_POST).values()) < DEFAULT_CONFIDENCE


def test_example_rows_make_a_topic_without_a_category_usable(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    topics = [make_topic(1, "crypto"), make_topic(2, "sport")]
    # nine forwarded posts per topic; the hash embedder has no semantics, so the posts of a
    # topic must share words the way reposts of a channel do
    suffixes = ("today", "this morning", "- details inside")
    examples = [make_example(1, f"{t} {s}") for t in CRYPTO_EXAMPLES[:3] for s in suffixes]
    examples += [
        make_example(2, f"{t} {s}", kind="channel") for t in SPORT_EXAMPLES[:3] for s in suffixes
    ]
    clf.reload(topics, examples)
    scores = _predict(clf, embedder, f"{CRYPTO_EXAMPLES[1]} tonight")
    assert scores[1] >= DEFAULT_CONFIDENCE > scores[2]
    scores = _predict(clf, embedder, f"{SPORT_EXAMPLES[1]} tonight")
    assert scores[2] >= DEFAULT_CONFIDENCE > scores[1]
    assert max(_predict(clf, embedder, UNRELATED_POST).values()) < DEFAULT_CONFIDENCE


def test_retrain_is_deterministic_and_examples_of_other_topics_are_ignored(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    topics = [make_topic(1, "crypto", category="crypto"), make_topic(2, "sport", category="sport")]
    examples = [make_example(1, t) for t in CRYPTO_EXAMPLES]
    examples += [make_example(2, t) for t in SPORT_EXAMPLES]
    examples.append(make_example(9, "An example of a topic that no longer exists"))
    a = LocalTopicClassifier(embedder, base)
    b = LocalTopicClassifier(embedder, base)
    a.reload(topics, examples)
    b.reload(topics, examples)
    a.reload(topics, examples)  # a second retrain lands on exactly the same weights
    assert np.array_equal(a._layer.w, b._layer.w) and a._layer.b == b._layer.b
    for text in (SPORT_POST, UNRELATED_POST, CRYPTO_EXAMPLES[0]):
        x = embedder.embed([text])[0]
        assert a.predict(text, x) == b.predict(text, x)
    # the order of the rows only moves floating-point noise
    b.reload(topics, list(reversed(examples)))
    assert np.allclose(a._layer.w, b._layer.w) and abs(a._layer.b - b._layer.b) < 1e-9
    assert all(e.topic_id in (1, 2) for e in a._examples)


def test_examples_without_an_embedding_are_embedded_from_their_text(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    stale = [make_example(1, t, embedding=b"\x00\x01") for t in CRYPTO_EXAMPLES]
    fresh = [make_example(1, t) for t in CRYPTO_EXAMPLES]
    clf.reload([make_topic(1, "crypto")], stale)
    stale_layer = clf._layer
    clf.reload([make_topic(1, "crypto")], fresh)
    assert np.allclose(
        stale_layer.topics[1].positives.vectors, clf._layer.topics[1].positives.vectors
    )


def test_layer_is_swapped_atomically(
    embedder: HashEmbedder, base: BaseModel, make_topic: Callable[..., Topic]
) -> None:
    clf = LocalTopicClassifier(embedder, base)
    clf.reload([make_topic(1, "a", category="crypto")], [])
    first = clf._layer
    clf.reload([make_topic(1, "a", category="crypto"), make_topic(2, "b", category="sport")], [])
    assert clf._layer is not first and first.topic_ids == (1,) and clf._layer.topic_ids == (1, 2)


def test_export_and_import_of_the_fitted_weights(
    embedder: HashEmbedder,
    base: BaseModel,
    make_topic: Callable[..., Topic],
    make_example: Callable[..., Example],
) -> None:
    topics = [make_topic(1, "crypto", category="crypto"), make_topic(2, "sport", category="sport")]
    examples = [make_example(1, t) for t in CRYPTO_EXAMPLES]
    examples += [make_example(2, t) for t in SPORT_EXAMPLES]
    trained = LocalTopicClassifier(embedder, base)
    trained.reload(topics, examples)
    blob = trained.export_layer()
    fresh = LocalTopicClassifier(embedder, base)
    fresh.reload(topics, [])
    assert not np.array_equal(fresh._layer.w, trained._layer.w)
    assert fresh.import_layer(blob)
    assert np.array_equal(fresh._layer.w, trained._layer.w) and fresh._layer.b == trained._layer.b
    other = LocalTopicClassifier(embedder, base)
    other.reload([make_topic(3, "x")], [])
    assert not other.import_layer(blob)
    assert not other.import_layer(b"not a layer")


# --- discovery -------------------------------------------------------------------------------


def _blobs(seed: int = 0) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, 64))
    points, labels = [], []
    for c in range(3):
        for _ in range(40):
            points.append(centres[c] + 0.6 * rng.normal(size=64))
            labels.append(c)
    for _ in range(30):
        points.append(rng.normal(size=64))
        labels.append(-1)
    return np.array(points, dtype=np.float32), labels


def test_find_clusters_finds_synthetic_blobs_and_ignores_scatter() -> None:
    x, labels = _blobs()
    clusters = find_clusters(x, 10, 0.6)
    assert len(clusters) == 3
    found = [sorted({labels[i] for i in c.member_ids}) for c in clusters]
    assert sorted(found) == [[0], [1], [2]]
    for c in clusters:
        assert len(c.member_ids) == 40 and c.tightness >= 0.6
        assert c.centroid.shape == (64,) and abs(float(np.linalg.norm(c.centroid)) - 1) < 1e-5
    assert find_clusters(x, 41, 0.6) == []
    assert find_clusters(x, 10, 0.99) == []
    assert find_clusters(np.zeros((0, 64), dtype=np.float32), 1, 0.5) == []


def test_find_clusters_maps_ids_collapses_duplicates_and_skips_existing_topics() -> None:
    x, labels = _blobs()
    ids = [1000 + i for i in range(len(x))]
    clusters = find_clusters(x, 10, 0.6, ids=ids)
    assert all(m >= 1000 for c in clusters for m in c.member_ids)
    # twenty copies of one scattered post are one story, never a cluster
    copies = np.repeat(x[-1:], 20, axis=0)
    clusters = find_clusters(np.vstack([x, copies]), 10, 0.6)
    assert len(clusters) == 3 and all(len(c.member_ids) == 40 for c in clusters)
    # a blob that is "more of" an existing topic is not proposed
    blob0 = x[[i for i, lab in enumerate(labels) if lab == 0]].mean(0)
    clusters = find_clusters(x, 10, 0.6, existing=blob0[None, :])
    assert sorted(sorted({labels[i] for i in c.member_ids}) for c in clusters) == [[1], [2]]
    with pytest.raises(ValueError):
        find_clusters(x, 10, 0.6, ids=[1, 2])


def test_average_linkage_matches_a_naive_implementation() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    chain = sorted(sorted(c) for c in average_linkage(x, 0.2))
    # naive: repeatedly merge the pair with the highest average cosine
    sims = x @ x.T
    clusters: list[list[int]] = [[i] for i in range(40)]
    while True:
        best, pair = -2.0, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sims[np.ix_(clusters[i], clusters[j])].mean())
                if s > best:
                    best, pair = s, (i, j)
        if pair is None or best < 0.2:
            break
        i, j = pair
        clusters[i] += clusters.pop(j)
    assert chain == sorted(sorted(c) for c in clusters)


def _post(post_id: int, topic_id: int, embedding: np.ndarray, scores: dict[int, float]) -> Post:
    return Post(
        id=post_id,
        chat_id=-1,
        message_id=post_id,
        kind="post",
        message_ids=[post_id],
        posted_at=NOW,
        via="live",
        text="x",
        text_hash="h",
        urls=[],
        embedding=embedding.astype("<f4").tobytes(),
        ingested_at=NOW,
        status=PostStatus.digest,
        topic_id=topic_id,
        topic_scores=[{"topic_id": t, "confidence": c} for t, c in scores.items()],
    )


def test_competition_detects_two_near_identical_topics() -> None:
    rng = np.random.default_rng(5)
    direction_a = rng.normal(size=32)
    direction_b = rng.normal(size=32)
    posts = []
    direction_c = rng.normal(size=32)
    for i in range(8):
        # topics 1 and 2 keep competing for the same posts although they point elsewhere
        posts.append(_post(i, 1, direction_a + 0.3 * rng.normal(size=32), {1: 0.9, 2: 0.7}))
        posts.append(_post(10 + i, 2, direction_b + 0.3 * rng.normal(size=32), {2: 0.9, 1: 0.6}))
        # topic 3 is its own thing
        posts.append(_post(20 + i, 3, direction_c + 0.3 * rng.normal(size=32), {3: 0.9, 1: 0.1}))
    pairs = competition(posts, 0.3)
    assert [(p.a_id, p.b_id) for p in pairs] == [(1, 2)]
    assert pairs[0].shared == 16
    assert competition(posts, 1.01) == []  # a share no overlap can reach
    # too few posts say nothing
    assert competition(posts[:6], 0.3) == []


def test_competition_by_meaning_without_score_overlap() -> None:
    rng = np.random.default_rng(7)
    direction = rng.normal(size=32)
    posts = [_post(i, 1, direction + 0.2 * rng.normal(size=32), {1: 0.9}) for i in range(6)]
    posts += [_post(10 + i, 2, direction + 0.2 * rng.normal(size=32), {2: 0.9}) for i in range(6)]
    other = rng.normal(size=32)
    posts += [_post(20 + i, 3, other + 0.2 * rng.normal(size=32), {3: 0.9}) for i in range(6)]
    assert [(p.a_id, p.b_id, p.shared) for p in competition(posts, 0.3)] == [(1, 2, 0)]


# --- models.py -------------------------------------------------------------------------------


def test_fake_models_select_the_hash_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(models.FAKE_MODELS_ENV, "1")
    monkeypatch.setattr(models, "_embedder", None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    models.ensure_models(tmp_path, lambda line: pytest.fail(f"unexpected progress: {line}"))
    assert "HF_HUB_OFFLINE" not in os_environ()
    embedder = models.make_embedder(tmp_path)
    assert isinstance(embedder, HashEmbedder) and models.make_embedder(tmp_path) is embedder
    clf = models.make_classifier(embedder)
    assert isinstance(clf, LocalTopicClassifier)
    assert (
        clf.category_scores(
            "Bitcoin hits a new high", embedder.embed(["Bitcoin hits a new high"])[0]
        )[0][0]
        == "crypto"
    )
    assert models.make_base_model(embedder).embedder_id == embedder.id


def test_real_embedder_needs_the_downloaded_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(models.FAKE_MODELS_ENV, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(models, "_embedder", None)
    assert models.models_dir(tmp_path) == tmp_path / "models"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert models.models_dir(tmp_path) == tmp_path / "hf"
    with pytest.raises(FileNotFoundError):
        models.make_embedder(tmp_path)


def test_embedder_id_is_pinned() -> None:
    assert models.EMBEDDER_REPO == "intfloat/multilingual-e5-small"
    assert models.EMBEDDER_REVISION == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert models.EMBEDDER_ID.startswith(models.EMBEDDER_REPO + "@")
    assert models.EMBEDDER_INT8_FILE == "onnx/model_qint8_avx512_vnni.onnx"
    assert models.EMBEDDER_FP32_FILE == "onnx/model.onnx"


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)
