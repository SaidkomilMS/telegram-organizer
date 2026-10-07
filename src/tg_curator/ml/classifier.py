"""The per-user topic layer on top of the shipped category model (DESIGN §10, research §7).

A topic that names a built-in category and has no posts of its own yet (no examples, no
confirmed posts) is scored straight from the category model: the ensemble probability of its
category through ``base_model.topic_confidence``, which puts the base model's measured floor on
the default threshold 0.5. The layer below cannot score such a topic well: it has no positive
row to learn from, and its hand-set prior was tuned on the sharper prototype-only probabilities
of the research script, so clearly on-topic posts sat at 0.38-0.54 (measured with the real
models: recall 0.20 at 0.5 for one category-only topic per category, 0.87 with the map).

Once a topic has posts of its own, one tiny logistic regression shared by every such topic
takes over. For a post ``x`` and a topic ``t`` it sees six numbers — how close ``x`` is to the
topic's best example, to its top three, to its description, how likely the topic's built-in
category is, how close ``x`` is to a post the owner removed from ``t``, and whether ``t`` has
a category at all — plus a one-hot term per topic. Ten-odd weights fit in milliseconds from
the owner's own examples, so every correction moves all topics at once. Only topics with
positives take part in the fit (a topic with none would only add negative rows, which drove
its one-hot term, and every category-only topic, to zero as soon as another topic got
examples); a topic with no contrast yet — the only one with examples, nothing removed —
gets the built-in category prototypes as background negatives so the fit does not learn
"everything is this topic". A topic with neither a category nor posts (a description only)
stays on the layer and is weak, as the research measured.

Corrections also act instantly through a kNN override: a post whose raw cosine to a corrected
post passes the duplicate thresholds inherits that correction (0.95 / 0.05) before the
logistic score is used, so a repost or a translation of something the owner just moved does
not wait for the refit.

Thread safety: ``reload``/``learn`` build a complete ``_Layer`` and then swap it in with one
assignment; ``predict`` reads ``self._layer`` once at entry, so the sorter's worker thread never
sees a half-built layer (no locks on the read path).
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tg_curator.contracts import Embedder
from tg_curator.domain import Example, Topic, TopicScore
from tg_curator.ml.base_model import BaseModel, topic_confidence
from tg_curator.ml.embedder import detect_language, unit_norm

log = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.5
"""The decision point the scores are calibrated around (``sorting.confidence``): the layer's
probabilities are fitted around it and ``topic_confidence`` maps the base floor onto it."""
NEAR_SAME_LANGUAGE = 0.90
NEAR_CROSS_LANGUAGE = 0.86
"""Raw-cosine thresholds of the kNN override: the duplicate rule of §9.2 reused."""
FORCED_SCORE = 0.95
VETOED_SCORE = 0.05
N_CORE_FEATURES = 6
PRIOR_W = np.array([3.0, 2.0, 2.0, 4.0, -3.0, 0.0])
PRIOR_B = -3.0
"""Hand-set weights used until there are enough examples to fit on (research §7.1)."""
MIN_FIT_ROWS = 3
"""The fit needs this many positive and negative rows; a topic with fewer negatives than this
gets the category prototypes as background negatives."""
LAYER_FORMAT = 1


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(
    features: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    iters: int = 500,
    lr: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Deterministic numpy logistic regression (Adam, class-balanced, L2 on the six shared
    features only — the per-topic one-hot terms are free so a topic can be strict or lax)."""
    w = np.zeros(features.shape[1])
    b = 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b = v_b = 0.0
    positives = y.sum()
    pos_weight = (len(y) - positives) / max(positives, 1)
    sample_w = np.where(y == 1, pos_weight, 1.0)
    l2_vec = np.zeros(features.shape[1])
    l2_vec[:N_CORE_FEATURES] = l2
    for t in range(1, iters + 1):
        p = _sigmoid(features @ w + b)
        g = (p - y) * sample_w / len(y)
        g_w = features.T @ g + l2_vec * w
        g_b = float(g.sum())
        m_w = 0.9 * m_w + 0.1 * g_w
        v_w = 0.999 * v_w + 0.001 * g_w * g_w
        m_b = 0.9 * m_b + 0.1 * g_b
        v_b = 0.999 * v_b + 0.001 * g_b * g_b
        w -= lr * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9**t)) / (np.sqrt(v_b / (1 - 0.999**t)) + 1e-8)
    return w, float(b)


# --- the immutable layer ----------------------------------------------------------------------


@dataclass
class _Memory:
    """Raw unit vectors with the language of their text, for the kNN override."""

    vectors: np.ndarray  # (k, dim)
    langs: np.ndarray  # (k,) str

    @classmethod
    def empty(cls, dim: int) -> _Memory:
        return cls(np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype="<U8"))

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def near(self, x: np.ndarray, lang: str, same: float, cross: float) -> bool:
        if not len(self):
            return False
        sims = self.vectors @ x
        return bool(np.any(sims >= np.where(self.langs == lang, same, cross)))


@dataclass
class _TopicState:
    topic_id: int
    category_index: int | None
    description: np.ndarray | None  # centred unit vector
    positives: _Memory
    negatives: _Memory
    forced: _Memory
    centred_positives: np.ndarray = field(init=False)
    centred_negatives: np.ndarray = field(init=False)


@dataclass
class _Layer:
    topic_ids: tuple[int, ...]
    topics: dict[int, _TopicState]
    not_for_me: _Memory
    centred_not_for_me: np.ndarray
    w: np.ndarray
    b: float


class LocalTopicClassifier:
    """``TopicClassifier`` (contracts.py) built from the shipped base model and the owner's
    examples. Constructed by ``service.py`` / ``cli.py`` through ``ml.models``."""

    def __init__(
        self,
        embedder: Embedder,
        base_model: BaseModel,
        *,
        near_same: float = NEAR_SAME_LANGUAGE,
        near_cross: float = NEAR_CROSS_LANGUAGE,
    ) -> None:
        self._embedder = embedder
        self._base = base_model
        self._near_same = near_same
        self._near_cross = near_cross
        self._lock = threading.Lock()  # serialises the writers (reload/learn), never predict
        self._topics: dict[int, Topic] = {}
        self._examples: list[Example] = []
        self._layer = self._build()

    # --- TopicClassifier ---

    def reload(self, topics: Sequence[Topic], examples: Sequence[Example]) -> None:
        with self._lock:
            self._topics = {t.id: t for t in topics if t.active}
            self._examples = [e for e in examples if self._relevant(e)]
            layer = self._build()
        self._layer = layer

    def learn(self, example: Example) -> None:
        """A new example or correction; a correction for a post replaces its older row."""
        with self._lock:
            if not self._relevant(example):
                return
            if example.post_id is not None:
                self._examples = [
                    e
                    for e in self._examples
                    if not (e.post_id == example.post_id and e.kind == example.kind)
                ]
            self._examples.append(example)
            layer = self._build()
        self._layer = layer

    def predict(self, text: str, embedding: np.ndarray) -> list[TopicScore]:
        layer = self._layer
        if not layer.topic_ids:
            return []
        x = unit_norm(np.asarray(embedding, dtype=np.float32))
        lang = detect_language(text)
        base_probs = self._base.probabilities(x)
        xc = self._base.centre(x)
        scores = []
        for tid in layer.topic_ids:
            state = layer.topics[tid]
            s = self._score(layer, state, x, xc, base_probs)
            if state.negatives.near(x, lang, self._near_same, self._near_cross) or (
                layer.not_for_me.near(x, lang, self._near_same, self._near_cross)
            ):
                s = min(s, VETOED_SCORE)
            if state.forced.near(x, lang, self._near_same, self._near_cross):
                s = max(s, FORCED_SCORE)
            scores.append(TopicScore(tid, min(max(s, 0.0), 1.0)))
        return sorted(scores, key=lambda sc: (-sc.confidence, sc.topic_id))

    def category_scores(self, text: str, embedding: np.ndarray) -> list[tuple[str, float]]:
        return self._base.scores(unit_norm(np.asarray(embedding, dtype=np.float32)))

    def _score(
        self,
        layer: _Layer,
        state: _TopicState,
        x: np.ndarray,
        xc: np.ndarray,
        base_probs: np.ndarray,
    ) -> float:
        """The category model alone until the topic has posts of its own, then the layer
        (which still sees the category probability as a feature). Blending the two by the
        number of examples was measured and rejected: it pulls a sub-topic such as fintech
        back towards its whole category (fintech recall 1.00 -> 0.89 with six examples)."""
        if state.category_index is not None and not len(state.positives):
            return float(topic_confidence(base_probs[state.category_index]))
        return float(_sigmoid(self._features(layer, state, x, xc, base_probs) @ layer.w + layer.b))

    # --- persistence of the fitted weights (numpy only) ---

    def export_layer(self) -> bytes:
        """The fitted weights as ``.npz`` bytes (``kv ml.user_layer`` holds them base64-encoded
        so a restart can predict before the first retrain finishes)."""
        layer = self._layer
        buf = io.BytesIO()
        np.savez(
            buf,
            format=np.array(LAYER_FORMAT),
            topic_ids=np.array(layer.topic_ids, dtype=np.int64),
            w=layer.w,
            b=np.array(layer.b),
        )
        return buf.getvalue()

    def import_layer(self, data: bytes) -> bool:
        """Restore weights exported for the same set of topics; ``False`` (and nothing
        changed) when the topics differ or the bytes are not ours."""
        try:
            with np.load(io.BytesIO(data)) as saved:
                if int(saved["format"]) != LAYER_FORMAT:
                    return False
                topic_ids = tuple(int(i) for i in saved["topic_ids"])
                w = np.asarray(saved["w"], dtype=np.float64)
                b = float(saved["b"])
        except (OSError, ValueError, KeyError):
            return False
        with self._lock:
            current = self._layer
            if current.topic_ids != topic_ids or w.shape != current.w.shape:
                return False
            layer = _Layer(
                topic_ids=current.topic_ids,
                topics=current.topics,
                not_for_me=current.not_for_me,
                centred_not_for_me=current.centred_not_for_me,
                w=w,
                b=b,
            )
        self._layer = layer
        return True

    # --- building ---

    def _relevant(self, example: Example) -> bool:
        if example.topic_id is None:
            return example.kind == "correction"
        return example.topic_id in self._topics

    def _vector(self, example: Example) -> np.ndarray:
        dim = self._embedder.dim
        if example.embedding and len(example.embedding) == dim * 4:
            return unit_norm(np.frombuffer(example.embedding, dtype="<f4").copy())
        log.debug("example %s has no usable embedding; embedding its text", example.id)
        return unit_norm(self._embedder.embed([example.text])[0])

    def _build(self) -> _Layer:
        dim = self._embedder.dim
        topic_ids = tuple(sorted(self._topics))
        cat_index = {k: i for i, k in enumerate(self._base.keys)}
        pos: dict[int, list[tuple[np.ndarray, str]]] = {t: [] for t in topic_ids}
        neg: dict[int, list[tuple[np.ndarray, str]]] = {t: [] for t in topic_ids}
        forced: dict[int, list[tuple[np.ndarray, str]]] = {t: [] for t in topic_ids}
        desc: dict[int, np.ndarray] = {}
        not_for_me: list[tuple[np.ndarray, str]] = []
        for ex in self._examples:
            vec = self._vector(ex)
            lang = detect_language(ex.text)
            if ex.kind == "description" and ex.topic_id in pos:
                desc[ex.topic_id] = vec
                continue
            if ex.topic_id is None:
                not_for_me.append((vec, lang))
            elif ex.topic_id in pos:
                pos[ex.topic_id].append((vec, lang))
                if ex.kind == "correction":
                    forced[ex.topic_id].append((vec, lang))
            if ex.kind == "correction" and ex.wrong_topic_id in neg:
                neg[ex.wrong_topic_id].append((vec, lang))
        for tid, topic in self._topics.items():
            if tid not in desc and topic.description:
                desc[tid] = unit_norm(self._embedder.embed([topic.description])[0])

        states: dict[int, _TopicState] = {}
        for tid in topic_ids:
            topic = self._topics[tid]
            state = _TopicState(
                topic_id=tid,
                category_index=cat_index.get(topic.category or ""),
                description=self._base.centre(desc[tid]) if tid in desc else None,
                positives=self._memory(pos[tid], dim),
                negatives=self._memory(neg[tid], dim),
                forced=self._memory(forced[tid], dim),
            )
            state.centred_positives = self._base.centre(state.positives.vectors)
            state.centred_negatives = self._base.centre(state.negatives.vectors)
            states[tid] = state
        nfm = self._memory(not_for_me, dim)
        layer = _Layer(
            topic_ids=topic_ids,
            topics=states,
            not_for_me=nfm,
            centred_not_for_me=self._base.centre(nfm.vectors),
            w=np.concatenate([PRIOR_W, np.zeros(len(topic_ids))]),
            b=PRIOR_B,
        )
        self._fit(layer)
        return layer

    @staticmethod
    def _memory(items: list[tuple[np.ndarray, str]], dim: int) -> _Memory:
        if not items:
            return _Memory.empty(dim)
        return _Memory(
            np.stack([v for v, _ in items]).astype(np.float32),
            np.array([lang for _, lang in items], dtype="<U8"),
        )

    def _features(
        self,
        layer: _Layer,
        state: _TopicState,
        x: np.ndarray,
        xc: np.ndarray,
        base_probs: np.ndarray,
        *,
        exclude: int | None = None,
    ) -> np.ndarray:
        """The research's six features plus the topic one-hot; ``exclude`` leaves one positive
        out so an example is not scored against itself during the fit."""
        sims = state.centred_positives @ xc
        if exclude is not None:
            sims = np.delete(sims, exclude)
        if sims.size:
            f1 = float(sims.max())
            f2 = float(np.sort(sims)[-3:].mean())
        else:
            f1 = f2 = 0.0
        f3 = float(state.description @ xc) if state.description is not None else 0.0
        f4 = float(base_probs[state.category_index]) if state.category_index is not None else 0.0
        negs = state.centred_negatives @ xc
        f5 = float(negs.max()) if negs.size else 0.0
        f6 = 1.0 if state.category_index is not None else 0.0
        onehot = np.zeros(len(layer.topic_ids))
        onehot[layer.topic_ids.index(state.topic_id)] = 1.0
        return np.concatenate([[f1, f2, f3, f4, f5, f6], onehot])

    def _fit(self, layer: _Layer) -> None:
        """Leave-one-out positives, every other topic's examples as negatives, removed posts
        and "not for me" posts as negatives; keeps the prior when there is too little data.

        Only topics with positives take part (the others are scored from their category or,
        without one, by the shared weights alone). A topic with fewer than ``MIN_FIT_ROWS``
        negative rows gets the category prototypes (all but its own category's) as background
        negatives: with examples for one topic only and nothing removed, the fit otherwise
        has no contrast and learned to put every post into that topic (measured: precision
        0.31, 0 % of untopical posts left unsorted; 0.43 / 23 % with the background).
        """
        rows: list[np.ndarray] = []
        y: list[int] = []
        negatives = dict.fromkeys(layer.topic_ids, 0)
        fitted = [t for t in layer.topic_ids if len(layer.topics[t].positives)]

        def add(
            tid: int,
            item: tuple[np.ndarray, np.ndarray, np.ndarray],
            label: int,
            exclude: int | None = None,
        ) -> None:
            vec, xc, probs = item
            rows.append(self._features(layer, layer.topics[tid], vec, xc, probs, exclude=exclude))
            y.append(label)
            negatives[tid] += 1 - label

        def prepared(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return vec, self._base.centre(vec), self._base.probabilities(vec)

        for tid in fitted:
            state = layer.topics[tid]
            for i, vec in enumerate(state.positives.vectors):
                item = prepared(vec)
                add(tid, item, 1, exclude=i)
                for other in fitted:
                    if other != tid:
                        add(other, item, 0)
            for vec in state.negatives.vectors:
                add(tid, prepared(vec), 0)
        for vec in layer.not_for_me.vectors:
            item = prepared(vec)
            for tid in fitted:
                add(tid, item, 0)
        background = [prepared(vec) for vec in self._base.prototypes]
        for tid in fitted:
            if negatives[tid] < MIN_FIT_ROWS:
                own = layer.topics[tid].category_index
                for k, item in enumerate(background):
                    if k != own:
                        add(tid, item, 0)
        positives = sum(y)
        if positives < MIN_FIT_ROWS or len(y) - positives < MIN_FIT_ROWS:
            return
        layer.w, layer.b = fit_logreg(np.array(rows), np.array(y, dtype=float))
