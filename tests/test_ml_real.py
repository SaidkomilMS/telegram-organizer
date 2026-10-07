"""The real models against the research evaluation set (DESIGN §16, ``research/models.md``).

Skipped unless ``TG_CURATOR_TEST_MODELS=1``. Needs the downloaded model files in ``HF_HOME``
(default: ``hf-cache`` next to the eval directory; the module is skipped rather than
downloading when neither exists) and the research eval set: ``TG_CURATOR_EVAL_DIR``, else
``.build/research/eval`` in the repository (``posts.json``, ``user_examples.json``,
``categories.py``). The two tests that compare with the research's cached embeddings and
training data are skipped when ``cache/`` or ``train/`` are absent. Every test prints the
numbers it measured next to the research figure it reproduces.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from tg_curator.contracts import Embedder
from tg_curator.domain import Example, Topic
from tg_curator.ml import categories, models
from tg_curator.ml.base_model import FLOOR, HEAD_SCALE, BaseModel, fit_head
from tg_curator.ml.classifier import (
    DEFAULT_CONFIDENCE,
    FORCED_SCORE,
    VETOED_SCORE,
    LocalTopicClassifier,
)
from tg_curator.ml.cluster import competition, find_clusters
from tg_curator.ml.embedder import detect_language, numbers_in, numbers_shared, prepare

pytestmark = pytest.mark.models

ENABLED = os.environ.get("TG_CURATOR_TEST_MODELS") == "1"
EVAL_DIR = Path(
    os.environ.get("TG_CURATOR_EVAL_DIR")
    or Path(__file__).resolve().parents[1] / ".build" / "research" / "eval"
)
HF_CACHE = Path(os.environ.get("HF_HOME") or EVAL_DIR.parent / "hf-cache")
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
TOPIC_CATEGORY = {"ml_ai": "tech", "fintech": "finance", "uzbekistan": None, "football": "sport"}
TOPIC_DESCRIPTION = {
    "ml_ai": "Machine learning and AI: new models, LLMs, research papers, AI companies",
    "fintech": "Fintech: banks' apps, payments, neobanks, payment systems, transfers, cards",
    "uzbekistan": "News from Uzbekistan: Tashkent, government decrees, economy, society, sport",
    "football": "Football: matches, results, transfers, leagues, national teams",
}
EVAL_TO_BUILTIN = {"technology": "tech"}
LANGS = {"en": ("en",), "ru": ("ru",), "uz": ("uz_latn", "uz_cyrl")}

if not ENABLED:
    pytest.skip("set TG_CURATOR_TEST_MODELS=1 to run the real-model tests", allow_module_level=True)
if not (EVAL_DIR / "posts.json").exists():
    pytest.skip("TG_CURATOR_EVAL_DIR must point at research/eval", allow_module_level=True)
if not HF_CACHE.is_dir():
    pytest.skip(f"no model cache at {HF_CACHE}: set HF_HOME", allow_module_level=True)


def _needs(*parts: str) -> Path:
    path = EVAL_DIR.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"{path} is absent (the research's cached embeddings / training data)")
    return path


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ.pop(models.FAKE_MODELS_ENV, None)
    models.ensure_models(Path(os.environ["HF_HOME"]).parent, print)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    return models.make_embedder(Path(os.environ["HF_HOME"]).parent)


@pytest.fixture(scope="module")
def posts() -> list[dict]:
    return json.load(open(EVAL_DIR / "posts.json", encoding="utf-8"))


@pytest.fixture(scope="module")
def vectors(embedder: Embedder, posts: list[dict]) -> np.ndarray:
    return embedder.embed([p["text"] for p in posts])


@pytest.fixture(scope="module")
def base(embedder: Embedder) -> BaseModel:
    return models.make_base_model(embedder)


def _f1(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, int, int]:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), p, r, fp, fn


def test_embedder_reproduces_the_research_embeddings(
    embedder: Embedder, vectors: np.ndarray
) -> None:
    assert embedder.id == models.EMBEDDER_ID and embedder.dim == 384
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    cached = np.load(_needs("cache", "e5-small-q-vnni.posts.npy"))
    cos = (vectors * cached).sum(1)
    print(f"cosine to the research cache: min {cos.min():.4f} median {np.median(cos):.4f}")
    assert cos.min() > 0.99
    assert models.make_embedder(Path(os.environ["HF_HOME"]).parent) is embedder  # loaded once


def test_duplicate_rule_separates_same_story_pairs(posts: list[dict], vectors: np.ndarray) -> None:
    """Research §4.3: language-aware 0.90/0.86 + soft number veto -> F1 0.943 (15 FP / 15 FN)."""
    n = len(posts)
    iu = np.triu_indices(n, 1)
    story = np.array([p["story"] for p in posts])
    truth = story[iu[0]] == story[iu[1]]
    texts = [prepare(p["text"]) for p in posts]
    langs = np.array([detect_language(t) for t in texts])
    true_lang = np.array([p["lang"].split("_")[0] for p in posts])
    lang_acc = float((langs == true_lang).mean())
    nums = [numbers_in(t) for t in texts]
    sims = (vectors @ vectors.T)[iu]
    same = langs[iu[0]] == langs[iu[1]]
    veto = np.array(
        [
            len(nums[i]) >= 2 and len(nums[j]) >= 2 and not numbers_shared(nums[i], nums[j])
            for i, j in zip(*iu, strict=True)
        ]
    )
    rule = np.where(same, sims >= 0.90, sims >= 0.86) & ~(veto & (sims < 0.97))
    f1, p, r, fp, fn = _f1(rule, truth)
    best_global = max((_f1(sims >= t, truth)[0], t) for t in np.arange(0.84, 0.95, 0.005))
    pos_same = sims[truth & same]
    pos_cross = sims[truth & ~same]
    neg = sims[~truth]
    print(
        f"language id accuracy {lang_acc:.3f}; rule F1 {f1:.3f} P {p:.3f} R {r:.3f} "
        f"fp {fp} fn {fn}; best single threshold F1 {best_global[0]:.3f} @ {best_global[1]:.3f}; "
        f"same-language positives p5 {np.percentile(pos_same, 5):.3f}, cross-language positives "
        f"p5 {np.percentile(pos_cross, 5):.3f}, negatives p99 {np.percentile(neg, 99):.3f} "
        f"p99.9 {np.percentile(neg, 99.9):.3f} max {neg.max():.3f}"
    )
    assert lang_acc >= 0.9
    assert f1 >= 0.93 and p >= 0.92 and r >= 0.92
    assert best_global[0] < f1 - 0.08  # one global threshold is clearly worse
    assert np.percentile(neg, 99) < 0.87 < np.percentile(pos_cross, 25)


def test_category_model_accuracy(posts: list[dict], vectors: np.ndarray, base: BaseModel) -> None:
    """Research §6.3: ens-mean 0.912 overall; conf >= 0.4 keeps 83 % at 0.98."""
    assert base.embedder_id == models.EMBEDDER_ID  # the shipped weights, not a fallback
    clf = LocalTopicClassifier(models.make_embedder(Path(os.environ["HF_HOME"]).parent), base)
    pred = [clf.category_scores(p["text"], x)[0] for p, x in zip(posts, vectors, strict=True)]
    ok = np.array(
        [
            k == EVAL_TO_BUILTIN.get(p["cat"], p["cat"])
            for (k, _), p in zip(pred, posts, strict=True)
        ]
    )
    conf = np.array([c for _, c in pred])
    per_lang = {
        lang: round(float(ok[[p["lang"] == lang for p in posts]].mean()), 3)
        for lang in sorted({p["lang"] for p in posts})
    }
    keep = conf >= 0.4
    print(
        f"category accuracy {ok.mean():.3f} {per_lang}; at conf >= 0.4: coverage "
        f"{keep.mean():.3f} accuracy {ok[keep].mean():.3f}"
    )
    assert ok.mean() >= 0.90
    assert all(v >= 0.85 for v in per_lang.values())
    assert keep.mean() >= 0.8 and ok[keep].mean() >= 0.97


def test_shipped_head_is_reproducible_from_the_research_data() -> None:
    _needs("train")
    _needs("cache")
    sys.path.insert(0, str(EVAL_DIR))
    try:
        import categories as research  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    keys = list(categories.KEYS)
    assert keys == list(research.CATEGORIES)
    items: list[str] = []
    mappers = (
        ("mnds", lambda r: research.MNDS_MAP.get(r["label"])),
        (
            "lenta",
            lambda r: research.LENTA_TAG_MAP.get(
                r.get("tag") or "", research.LENTA_TOPIC_MAP.get(r["label"])
            ),
        ),
        ("uz", lambda r: research.UZ_MAP.get(r["label"])),
        ("huff", lambda r: research.HUFF_MAP.get(r["label"])),
    )
    for name, mapper in mappers:
        for line in open(EVAL_DIR / "train" / f"{name}.jsonl", encoding="utf-8"):
            key = mapper(json.loads(line))
            if key:
                items.append(key)
    items += [k for k in keys for _ in categories.SEEDS[k]]
    x = np.load(_needs("cache", f"e5-small-q-vnni.train{len(items)}.npy"))
    y = np.array([keys.index(k) for k in items])
    w, b = fit_head(x * HEAD_SCALE, y, len(keys))
    shipped = BaseModel.load()
    assert np.array_equal(w, shipped.head_w) and np.array_equal(b, shipped.head_b)


def _topics() -> list[Topic]:
    return [
        Topic(
            id=i + 1,
            key=name,
            name=name,
            category=TOPIC_CATEGORY[name],
            description=TOPIC_DESCRIPTION[name],
            created_at=NOW,
        )
        for i, name in enumerate(TOPIC_CATEGORY)
    ]


def _scores_of(
    clf: LocalTopicClassifier, posts: list[dict], vectors: np.ndarray, i: int
) -> dict[int, float]:
    return {s.topic_id: s.confidence for s in clf.predict(posts[i]["text"], vectors[i])}


def _lang_mask(posts: list[dict], lang: str) -> np.ndarray:
    return np.array([p["lang"] in LANGS[lang] for p in posts])


def _category(post: dict) -> str:
    return EVAL_TO_BUILTIN.get(post["cat"], post["cat"])


def _stats(
    clf: LocalTopicClassifier, posts: list[dict], vectors: np.ndarray
) -> tuple[dict[str, dict[str, float]], list[dict[int, float]]]:
    scores = [
        {s.topic_id: s.confidence for s in clf.predict(p["text"], x)}
        for p, x in zip(posts, vectors, strict=True)
    ]
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TOPIC_CATEGORY):
        tid = i + 1
        truth = np.array([name in p["topics"] for p in posts])
        s = np.array([sc[tid] for sc in scores])
        f1, p, r, fp, fn = _f1(s >= DEFAULT_CONFIDENCE, truth)
        auc = float((s[truth][:, None] > s[~truth][None, :]).mean())
        out[name] = {"P": round(p, 2), "R": round(r, 2), "F1": round(f1, 2), "AUC": round(auc, 3)}
    none_true = np.array([not p["topics"] for p in posts])
    none_pred = np.array([max(sc.values()) < DEFAULT_CONFIDENCE for sc in scores])
    declared = {c for c in TOPIC_CATEGORY.values() if c}
    off_category = none_true & np.array([_category(p) not in declared for p in posts])
    out["none"] = {
        "untopical_as_none": round(float(none_pred[none_true].mean()), 2),
        # untopical posts outside every topic's category: what a category-only topic must
        # leave alone (a tech post that is not about ML *is* in ml_ai's declared category)
        "off_category_as_none": round(float(none_pred[off_category].mean()), 2),
        "topical_as_none": round(float(none_pred[~none_true].mean()), 2),
    }
    return out, scores


def _category_topics(keys: list[str]) -> list[Topic]:
    return [Topic(id=i + 1, key=k, name=k, category=k, created_at=NOW) for i, k in enumerate(keys)]


def test_category_only_topics_reach_the_base_operating_point(
    embedder: Embedder, base: BaseModel, posts: list[dict], vectors: np.ndarray
) -> None:
    """One topic per built-in category present in the eval set, no description, no examples:
    at the default threshold 0.5 every language keeps recall >= 0.80 at precision ~0.95.
    Measured (calibration report): before the fix P 1.00 / R 0.20 overall (en 0.17, ru 0.25,
    uz 0.18 recall: the layer prior needed a category probability of 0.75); with
    ``topic_confidence`` en 0.96/0.92, ru 0.97/0.86, uz 0.95/0.82, overall 0.96/0.87 (the
    research's top-1 floor 0.4 gives 0.98/0.82 overall but uz 0.94/0.73)."""
    keys = sorted({_category(p) for p in posts})
    clf = LocalTopicClassifier(embedder, base)
    clf.reload(_category_topics(keys), [])
    scores = np.array(
        [
            [s.confidence for s in sorted(clf.predict(p["text"], x), key=lambda s: s.topic_id)]
            for p, x in zip(posts, vectors, strict=True)
        ]
    )
    truth = np.array([[_category(p) == k for k in keys] for p in posts])
    pred = scores >= DEFAULT_CONFIDENCE
    # the score is the category model's own probability, only moved onto the threshold
    probs = np.array([base.probabilities(x) for x in vectors])
    cols = [base.keys.index(k) for k in keys]
    assert np.array_equal(pred, probs[:, cols] >= FLOOR)
    _, p_all, r_all, fp_all, _ = _f1(pred.ravel(), truth.ravel())
    per_lang = {}
    for lang in LANGS:
        m = _lang_mask(posts, lang)
        _, p, r, fp, fn = _f1(pred[m].ravel(), truth[m].ravel())
        per_lang[lang] = {"P": round(p, 3), "R": round(r, 3), "fp": fp, "fn": fn}
    print(
        f"category-only topics at {DEFAULT_CONFIDENCE}: P {p_all:.3f} R {r_all:.3f} "
        f"({fp_all} wrong of {int(pred.sum())}); {per_lang}"
    )
    assert p_all >= 0.95 and r_all >= 0.85
    for lang, m in per_lang.items():
        assert m["R"] >= 0.80, lang
        # uz: 37 of 39 right; the two misses (an Nvidia earnings post labelled finance that
        # lands in tech, a one-line exchange rate taken for humour) sit at 0.47 on the wrong
        # category, so no monotone calibration removes them without losing a third of uz
        assert m["P"] >= (0.94 if lang == "uz" else 0.95), lang
    # posts outside a smaller set of category topics stay unsorted
    subset = ["tech", "finance", "sport", "politics"]
    clf.reload(_category_topics(subset), [])
    outside = [
        max(s.confidence for s in clf.predict(p["text"], x)) < DEFAULT_CONFIDENCE
        for p, x in zip(posts, vectors, strict=True)
        if _category(p) not in subset
    ]
    print(f"posts outside {subset} left unsorted: {np.mean(outside):.3f} of {len(outside)}")
    assert np.mean(outside) >= 0.95


def test_user_layer_stages_and_the_on_the_spot_flip(
    embedder: Embedder, base: BaseModel, posts: list[dict], vectors: np.ndarray
) -> None:
    """Research §7.2: A (category only) ml_ai R 0.88 / football R 0.77, uzbekistan 0 — now
    the calibrated category (R 1.00 / 1.00); M and B3 (a mix of example-backed and
    category-only topics, not in the research); B (6 examples) every topic usable, 3 %
    topical posts wrongly "none"; D: one confirmed Revolut post flips its English original
    0.66 -> 0.95; one removed post -> 0.05."""
    examples_by_topic = json.load(open(EVAL_DIR / "user_examples.json", encoding="utf-8"))
    topics = _topics()
    make: Callable[..., Example] = lambda i, tid, text, **kw: Example(  # noqa: E731
        id=i,
        topic_id=tid,
        kind=kw.get("kind", "example"),
        post_id=kw.get("post_id"),
        wrong_topic_id=kw.get("wrong_topic_id"),
        text=text,
        embedding=embedder.embed([text])[0].astype("<f4").tobytes(),
        created_at=NOW,
    )
    clf = LocalTopicClassifier(embedder, base)
    clf.reload(topics, [])
    stage_a, scores_a = _stats(clf, posts, vectors)
    print("A. zero examples:", stage_a)
    # before the calibration fix: ml_ai 0.75/0.71, football 0.89/0.77 (research 0.68/0.88,
    # 0.81/0.77); now the whole declared category: ml_ai 0.63/1.00, football 0.85/1.00.
    # fintech stays 0/0: its nine posts are the ones the category model calls tech/crypto.
    assert stage_a["ml_ai"]["R"] >= 0.95 and stage_a["football"]["R"] >= 0.95
    assert stage_a["football"]["P"] >= 0.8
    assert stage_a["uzbekistan"]["R"] < 0.3  # a description alone is weak (§10)
    assert stage_a["none"]["off_category_as_none"] >= 0.95

    # A'. corrections on category-only topics flip at once (kNN override on top of the map)
    football = 4
    outside = [i for i, p in enumerate(posts) if "football" not in p["topics"]]
    top_a = max(outside, key=lambda i: scores_a[i][football])
    clf.learn(
        make(
            898,
            None,
            posts[top_a]["text"],
            kind="correction",
            post_id=top_a,
            wrong_topic_id=football,
        )
    )
    assert _scores_of(clf, posts, vectors, top_a)[football] <= VETOED_SCORE
    inside = [i for i, p in enumerate(posts) if "ml_ai" in p["topics"]]
    low_a = min(inside, key=lambda i: scores_a[i][1])
    clf.learn(make(899, 1, posts[low_a]["text"], kind="correction", post_id=low_a))
    after_a = _scores_of(clf, posts, vectors, low_a)
    print(
        f"A'. removed {posts[top_a]['story']} from football {scores_a[top_a][football]:.2f} -> "
        f"<= {VETOED_SCORE}; confirmed {posts[low_a]['story']} in ml_ai "
        f"{scores_a[low_a][1]:.2f} -> {after_a[1]:.2f}"
    )
    assert after_a[1] >= FORCED_SCORE

    examples = []
    for i, name in enumerate(TOPIC_CATEGORY):
        examples += [make(len(examples) + 1, i + 1, t) for t in examples_by_topic[name]]
    # M. the template's first step: examples only for the topic without a category. Before
    # the fix ml_ai and football fell to recall 0 and uzbekistan took every post (P 0.31,
    # 0 % of untopical posts unsorted); now 0.43/0.98 with 23 % unsorted.
    clf.reload(topics, [e for e in examples if e.topic_id == 3])
    stage_m, _ = _stats(clf, posts, vectors)
    print("M. examples for uzbekistan only:", stage_m)
    assert stage_m["ml_ai"] == stage_a["ml_ai"] and stage_m["football"] == stage_a["football"]
    assert stage_m["uzbekistan"]["R"] >= 0.9 and stage_m["uzbekistan"]["P"] >= 0.4
    assert stage_m["none"]["untopical_as_none"] >= 0.2
    # B3. three topics with examples, football still category-only (before: football 0/0)
    clf.reload(topics, [e for e in examples if e.topic_id != football])
    stage_b3, _ = _stats(clf, posts, vectors)
    print("B3. examples for all but football:", stage_b3)
    assert stage_b3["football"] == stage_a["football"]
    assert stage_b3["none"]["untopical_as_none"] >= 0.45

    clf.reload(topics, examples)
    stage_b, scores_b = _stats(clf, posts, vectors)
    print("B. six examples per topic:", stage_b)
    # identical to the layer before the calibration fix (research: 0.63/1.00, 0.19/0.89,
    # 0.81/0.89, 0.85/1.00, 53 % / 3 %): ml_ai 0.61/1.00, fintech 0.20/1.00,
    # uzbekistan 0.80/0.93, football 0.85/1.00, 54 % untopical unsorted, 2 % topical
    assert all(stage_b[n]["R"] >= 0.8 for n in TOPIC_CATEGORY)
    assert stage_b["football"]["F1"] >= 0.8 and stage_b["uzbekistan"]["F1"] >= 0.7
    assert stage_b["football"]["P"] >= 0.8 and stage_b["uzbekistan"]["P"] >= 0.75
    assert stage_b["none"]["topical_as_none"] <= 0.1
    assert stage_b["none"]["untopical_as_none"] >= 0.5
    assert all(stage_b[n]["AUC"] >= 0.8 for n in TOPIC_CATEGORY)

    # D. confirm the lowest-scoring fintech post: its same-story twins flip at once
    fintech = 2
    in_topic = [i for i, p in enumerate(posts) if "fintech" in p["topics"]]
    worst = min(in_topic, key=lambda i: scores_b[i][fintech])
    twins = [i for i, p in enumerate(posts) if p["story"] == posts[worst]["story"] and i != worst]
    clf.learn(
        make(
            900,
            fintech,
            posts[worst]["text"],
            kind="correction",
            post_id=worst,
            wrong_topic_id=None,
        )
    )
    after = {i: _scores_of(clf, posts, vectors, i) for i in [worst, *twins]}
    print(
        f"D. confirmed post {worst} ({posts[worst]['lang']}, {posts[worst]['story']}): "
        f"{scores_b[worst][fintech]:.2f} -> {after[worst][fintech]:.2f}; twins: "
        + ", ".join(
            f"{posts[i]['lang']}/{posts[i]['kind']} {scores_b[i][fintech]:.2f}->"
            f"{after[i][fintech]:.2f}"
            for i in twins
        )
    )
    assert after[worst][fintech] >= FORCED_SCORE
    assert twins and all(after[i][fintech] >= FORCED_SCORE for i in twins)

    # D'. remove the highest-scoring non-fintech post: the twin that is a near-duplicate flips
    out_topic = [i for i, p in enumerate(posts) if "fintech" not in p["topics"]]
    top = max(out_topic, key=lambda i: scores_b[i][fintech])
    twins = [i for i, p in enumerate(posts) if p["story"] == posts[top]["story"] and i != top]
    clf.learn(
        make(901, None, posts[top]["text"], kind="correction", post_id=top, wrong_topic_id=fintech)
    )
    removed = {i: _scores_of(clf, posts, vectors, i) for i in [top, *twins]}
    print(
        f"D'. removed post {top} ({posts[top]['lang']}, {posts[top]['story']}): "
        f"{scores_b[top][fintech]:.2f} -> {removed[top][fintech]:.2f}; twins: "
        + ", ".join(
            f"{posts[i]['lang']}/{posts[i]['kind']} {scores_b[i][fintech]:.2f}->"
            f"{removed[i][fintech]:.2f}"
            for i in twins
        )
    )
    assert removed[top][fintech] <= VETOED_SCORE
    near = [
        i
        for i in twins
        if float(vectors[i] @ vectors[top])
        >= (
            0.90
            if detect_language(prepare(posts[i]["text"]))
            == detect_language(prepare(posts[top]["text"]))
            else 0.86
        )
    ]
    assert all(removed[i][fintech] <= VETOED_SCORE for i in near)


def test_discovery_on_the_eval_set_and_competing_topics(
    posts: list[dict], vectors: np.ndarray
) -> None:
    """Research §8: 181 posts collapse to about 53 stories; the eval set holds ~3 unrelated
    stories per category, so no cluster of four stories is tight; fintech/uzbekistan and
    uzbekistan/football compete by overlap."""
    from tg_curator.domain import Post, PostStatus
    from tg_curator.ml.cluster import _components

    langs = [p["lang"].split("_")[0] for p in posts]
    clusters = find_clusters(vectors, 4, 0.6, langs=langs)
    stories = (
        int(_components((vectors @ vectors.T >= 0.90) & ~np.eye(len(posts), dtype=bool)).max()) + 1
    )
    print(f"stories at the same-language threshold: {stories} (true 59); clusters: {len(clusters)}")
    assert clusters == []
    assert find_clusters(vectors, 4, 0.6) == []

    names = list(TOPIC_CATEGORY)
    rows = []
    for i, p in enumerate(posts):
        if not p["topics"]:
            continue
        scores = {names.index(n) + 1: 0.9 for n in p["topics"]}
        rows.append(
            Post(
                id=i,
                chat_id=-1,
                message_id=i,
                kind="post",
                message_ids=[i],
                posted_at=NOW,
                via="live",
                text=p["text"],
                text_hash="h",
                urls=[],
                embedding=vectors[i].astype("<f4").tobytes(),
                ingested_at=NOW,
                status=PostStatus.digest,
                topic_id=names.index(p["topics"][0]) + 1,
                topic_scores=[{"topic_id": t, "confidence": c} for t, c in scores.items()],
            )
        )
    pairs = {(p.a_id, p.b_id): p.shared for p in competition(rows, 0.3)}
    print("competing topics (a, b) -> shared:", pairs)
    assert (2, 3) in pairs and (3, 4) in pairs  # fintech/uzbekistan, uzbekistan/football
    assert (1, 4) not in pairs  # ml_ai vs football
    # the sanity of the centroid rule: no two eval topics mean the same thing
    assert all((a, b) in pairs for a, b in combinations((2, 3), 2))
