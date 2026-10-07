"""The shipped category model (DESIGN §10, ``research/models.md`` §5–§6).

Two cheap classifiers on the same 384-d embedding, averaged: (a) cosine to one prototype per
category, computed on *centred* vectors because e5 squeezes every news post into a narrow
cone and the common direction has to go before the softmax means anything; (b) a softmax
head trained offline on public news-category data and shipped as a 30 KB ``.npz`` inside the
package, so a user's server never trains and never downloads a second model. The prototypes
carry crypto, ads, jobs and humor (no public data), the head carries the world/disaster side
and the English and Cyrillic-Uzbek posts; together they score 0.91 on the research eval set.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np

from tg_curator.contracts import Embedder
from tg_curator.ml import categories
from tg_curator.ml.embedder import unit_norm

log = logging.getLogger(__name__)

DATA_FILE = "category_model.npz"
PROTO_TEMPERATURE = 0.05
"""Softmax temperature over centred cosines; measured to give usable confidences."""
HEAD_SCALE = 10.0
"""e5 cosines live in a narrow band; the head was trained on embeddings scaled by this."""
FLOOR = 1 / 3
"""The ensemble probability at which a category guess is trusted; ``topic_confidence`` maps it
onto 0.5, the default ``sorting.confidence``. Measured with the real models on the research
eval set, one category-only topic per category (precision / recall per language):

    floor 0.40 (the research's top-1 floor): en 0.98/0.90  ru 1.00/0.81  uz 0.94/0.73
    floor 1/3:                               en 0.96/0.92  ru 0.97/0.86  uz 0.95/0.82

Every floor from 0.33 to 0.45 has the same worst-language precision (a story-level bootstrap
gives a median of 0.94 for each: two Uzbek posts sit at 0.47 on a wrong category, which no
monotone map can remove), while the worst-language recall falls from 0.82 to 0.64; below
0.33 Russian precision drops to 0.94 and lower. 1/3 is the lowest floor before that drop."""


def topic_confidence(p: np.ndarray | float, floor: float = FLOOR) -> np.ndarray | float:
    """The confidence of a topic that only names a built-in category, from that category's
    ensemble probability: piecewise linear through (0, 0), (floor, 0.5) and (1, 1).

    The ensemble's probabilities are flat (the head is unsure, the mean of two softmaxes
    halves every peak), so a clearly on-topic post often sits at 0.35-0.6; taken as they are
    they would leave most obvious posts unsorted at the default threshold of 0.5. The map is
    monotone, so the base model's ranking is untouched; it only moves the decision point onto
    the measured floor (``FLOOR``), and an owner who lowers ``sorting.confidence`` to 0.4 moves
    it to 0.27. Asserted with the real models in ``tests/test_ml_real.py``.
    """
    q = np.asarray(p, dtype=np.float64)
    out = np.where(q < floor, 0.5 * q / floor, 0.5 + 0.5 * (q - floor) / (1.0 - floor))
    out = np.clip(out, 0.0, 1.0)
    return float(out) if np.ndim(out) == 0 else out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def shipped_path() -> Path:
    """Where the packaged weights live."""
    return Path(str(files("tg_curator.ml") / "data" / DATA_FILE))


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    l2: float = 1e-2,
    epochs: int = 400,
    lr: float = 0.05,
    max_class_weight: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch multinomial logistic regression with Adam, numpy only; deterministic.

    Class weights are capped so the nine-seed classes do not dominate the gradient. This is
    the exact recipe of ``research/eval/classify_eval.py`` and exists in the package so the
    shipped weights can be regenerated and verified from the research embeddings.
    """
    n, d = x.shape
    w = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    targets = np.eye(n_classes, dtype=np.float32)[y]
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1
    sample_w = np.minimum(n / (n_classes * counts), max_class_weight)[y]
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        probs = softmax(x @ w + b)
        grad = (probs - targets) * sample_w[:, None] / n
        g_w = x.T @ grad + l2 * w
        g_b = grad.sum(0)
        m_w = b1 * m_w + (1 - b1) * g_w
        v_w = b2 * v_w + (1 - b2) * g_w * g_w
        m_b = b1 * m_b + (1 - b1) * g_b
        v_b = b2 * v_b + (1 - b2) * g_b * g_b
        w -= lr * (m_w / (1 - b1**epoch)) / (np.sqrt(v_w / (1 - b2**epoch)) + eps)
        b -= lr * (m_b / (1 - b1**epoch)) / (np.sqrt(v_b / (1 - b2**epoch)) + eps)
    return w, b


class BaseModel:
    """Category probabilities for one embedding: the mean of the prototype and head softmaxes.

    ``mu`` (the mean prototype) is also what the per-user layer centres its cosines on, so
    both classifiers agree on what "the common news direction" is.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prototypes: np.ndarray,
        head_w: np.ndarray,
        head_b: np.ndarray,
        *,
        embedder_id: str,
    ) -> None:
        if len(keys) != prototypes.shape[0] or head_w.shape[1] != len(keys):
            raise ValueError("category model: keys, prototypes and head do not agree")
        self.keys = tuple(keys)
        self.embedder_id = embedder_id
        self.prototypes = unit_norm(prototypes)
        self.mu = self.prototypes.mean(0).astype(np.float32)
        self.centred_prototypes = unit_norm(self.prototypes - self.mu)
        self.head_w = np.asarray(head_w, dtype=np.float32)
        self.head_b = np.asarray(head_b, dtype=np.float32)
        self.dim = int(self.prototypes.shape[1])

    def centre(self, vectors: np.ndarray) -> np.ndarray:
        """Centred, re-normalised copy of raw unit vectors (one or many)."""
        return unit_norm(np.asarray(vectors, dtype=np.float32) - self.mu)

    def probabilities(self, embedding: np.ndarray) -> np.ndarray:
        """``(n_categories,)`` probabilities in ``self.keys`` order, summing to one."""
        x = np.asarray(embedding, dtype=np.float32)
        proto = softmax((self.centred_prototypes @ self.centre(x)) / PROTO_TEMPERATURE)
        head = softmax(x * HEAD_SCALE @ self.head_w + self.head_b)
        return (proto + head) / 2

    def scores(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """``(key, probability)`` best first."""
        probs = self.probabilities(embedding)
        order = np.argsort(-probs, kind="stable")
        return [(self.keys[i], float(probs[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            keys=np.array(self.keys),
            prototypes=self.prototypes,
            head_w=self.head_w,
            head_b=self.head_b,
            embedder_id=np.array(self.embedder_id),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> BaseModel:
        """The shipped weights (or another ``.npz`` written by ``save``)."""
        with np.load(path or shipped_path()) as data:
            return cls(
                [str(k) for k in data["keys"]],
                data["prototypes"],
                data["head_w"],
                data["head_b"],
                embedder_id=str(data["embedder_id"]),
            )

    @classmethod
    def from_embedder(
        cls,
        embedder: Embedder,
        *,
        head: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> BaseModel:
        """Prototypes from the seed phrases through ``embedder``; ``head`` ``None`` fits the
        head on the seeds alone, which is what the test double gets (no public data)."""
        texts, keys = categories.seed_texts()
        vectors = unit_norm(embedder.embed(texts))
        key_arr = np.array(keys)
        protos = np.stack([vectors[key_arr == k].mean(0) for k in categories.KEYS])
        if head is None:
            labels = np.array([categories.KEYS.index(k) for k in keys])
            head = fit_head(vectors * HEAD_SCALE, labels, len(categories.KEYS))
        return cls(categories.KEYS, protos, head[0], head[1], embedder_id=embedder.id)


def load_for(embedder: Embedder) -> BaseModel:
    """The model to run with ``embedder``: the shipped weights when they were made for it,
    otherwise seed prototypes computed through the embedder with a flat head (the test double,
    or a future embedder whose head has not been retrained — logged once)."""
    path = shipped_path()
    if path.exists():
        model = BaseModel.load(path)
        if model.embedder_id == embedder.id and model.dim == embedder.dim:
            return model
        log.info("category model was trained for %s, not %s", model.embedder_id, embedder.id)
    return BaseModel.from_embedder(embedder)
