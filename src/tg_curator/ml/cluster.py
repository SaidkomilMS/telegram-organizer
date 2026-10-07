"""Discovery maths: tight clusters of unsorted posts and topics that compete (DESIGN §10, §12).

Numpy only. Raw e5 cosines between unrelated news posts already sit at 0.78–0.86, so every
similarity here is taken on vectors centred on the mean of the batch and re-normalised, which
spreads unrelated stories to about -0.1..0.25 and lets a plain cut (0.30, average linkage)
separate a real theme from background. Duplicates are collapsed first so a repost storm counts
as one story, not thirty.

Average linkage is run as a nearest-neighbour chain, which is exact for average linkage
(a reducible criterion) and O(n²) in time, so a week of 5,000 unsorted posts clusters in
seconds on a 100 MB similarity matrix instead of minutes.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from tg_curator.domain import Cluster, Post, TopicPair
from tg_curator.ml.embedder import unit_norm

LINK_CUT = 0.30
"""Average-linkage cut on centred cosine (research §8.1: 0.30 found both real lenta themes)."""
DUPLICATE_COSINE = 0.90
DUPLICATE_COSINE_CROSS = 0.86
"""Raw cosines at which two unsorted posts are one story: the dedup thresholds of §9.2 for
two posts in the same language and for two whose languages differ or are unknown."""
EXISTING_TOPIC_COSINE = 0.75
"""A cluster this close to an existing topic is "more of that topic", not a new one."""
CENTROID_COMPETE_COSINE = 0.80
"""Two topics whose accepted posts point the same way are competing on meaning."""
MIN_COMPETE_POSTS = 5
"""Below this many accepted posts an overlap share says nothing."""


def _components(adjacent: np.ndarray) -> np.ndarray:
    """Connected-component label per row of a boolean adjacency matrix."""
    n = adjacent.shape[0]
    label = -np.ones(n, dtype=np.int64)
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = current
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adjacent[u]):
                if label[v] < 0:
                    label[v] = current
                    stack.append(int(v))
        current += 1
    return label


def average_linkage(vectors: np.ndarray, cut: float) -> list[list[int]]:
    """Clusters of row indices whose average pairwise cosine stays at or above ``cut``.

    Nearest-neighbour chain: follow nearest neighbours until two clusters are each other's
    nearest; merge them when their linkage reaches ``cut``, otherwise neither can ever merge
    at or above ``cut`` again (linkage only falls as clusters grow), so both are final.
    """
    n = vectors.shape[0]
    if n == 0:
        return []
    sims = (vectors @ vectors.T).astype(np.float32)
    np.fill_diagonal(sims, -2.0)
    sizes = np.ones(n, dtype=np.float32)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    open_ = np.ones(n, dtype=bool)  # still able to merge
    chain: list[int] = []
    while int(open_.sum()) >= 2:
        if not chain:
            chain.append(int(np.flatnonzero(open_)[0]))
        i = chain[-1]
        row = np.where(open_, sims[i], -2.0)
        j = int(row.argmax())
        if len(chain) >= 2 and j == chain[-2]:
            chain.pop()
            chain.pop()
            if sims[i, j] >= cut:
                merged = (sims[i] * sizes[i] + sims[j] * sizes[j]) / (sizes[i] + sizes[j])
                sims[i] = merged
                sims[:, i] = merged
                sims[i, i] = -2.0
                sizes[i] += sizes[j]
                members[i].extend(members.pop(j))
                open_[j] = False
            else:
                open_[i] = False
                open_[j] = False
        else:
            chain.append(j)
    return list(members.values())


def find_clusters(
    embeddings: np.ndarray,
    min_size: int,
    tightness: float,
    *,
    ids: Sequence[int] | None = None,
    langs: Sequence[str | None] | None = None,
    existing: np.ndarray | None = None,
    link_cut: float = LINK_CUT,
) -> list[Cluster]:
    """Tight groups of at least ``min_size`` distinct stories among ``embeddings``.

    ``ids`` names the rows (post ids; row indices by default) in ``Cluster.member_ids``;
    ``langs`` (``posts.lang``, one per row) lets the duplicate collapse use the same-language
    threshold where it applies — without it every pair uses the lower cross-language one, which
    merges more and proposes less; ``existing`` holds the prototypes of the current topics
    (raw unit vectors, one row each) and a cluster within ``EXISTING_TOPIC_COSINE`` of one is
    dropped. ``Cluster.centroid`` is the raw unit-norm mean of the members, ready for
    ``category_scores``; ``tightness`` is the mean centred cosine of the stories to their
    centroid. Largest cluster first.
    """
    x = unit_norm(np.asarray(embeddings, dtype=np.float32))
    n = x.shape[0]
    names = list(ids) if ids is not None else list(range(n))
    if len(names) != n or (langs is not None and len(langs) != n):
        raise ValueError("ids and langs must name every embedding row")
    if n == 0 or min_size < 1:
        return []
    # 1. collapse duplicates into stories
    threshold = np.full((n, n), DUPLICATE_COSINE_CROSS, dtype=np.float32)
    if langs is not None:
        lang_arr = np.array([lang or "" for lang in langs])
        same = (lang_arr[:, None] == lang_arr[None, :]) & (lang_arr != "")[:, None]
        threshold[same] = DUPLICATE_COSINE
    adjacent = (x @ x.T >= threshold) & ~np.eye(n, dtype=bool)
    story_of = _components(adjacent)
    n_stories = int(story_of.max()) + 1
    stories = unit_norm(np.stack([x[story_of == s].mean(0) for s in range(n_stories)]))
    # 2. centre on the batch mean and cluster
    mu = x.mean(0)
    centred = unit_norm(stories - mu)
    centred_existing = (
        unit_norm(np.asarray(existing, dtype=np.float32) - mu)
        if existing is not None and len(existing)
        else None
    )
    out: list[Cluster] = []
    for story_rows in average_linkage(centred, link_cut):
        if len(story_rows) < min_size:
            continue
        centre = unit_norm(centred[story_rows].mean(0))
        tight = float((centred[story_rows] @ centre).mean())
        if tight < tightness:
            continue
        if centred_existing is not None and float((centred_existing @ centre).max()) >= (
            EXISTING_TOPIC_COSINE
        ):
            continue
        rows = np.flatnonzero(np.isin(story_of, story_rows))
        out.append(
            Cluster(
                member_ids=[names[r] for r in rows],
                centroid=unit_norm(x[rows].mean(0)),
                tightness=tight,
            )
        )
    out.sort(key=lambda c: (-len(c.member_ids), -c.tightness))
    return out


def _accepted_by(post: Post, threshold: float) -> set[int]:
    """The topics that would have taken the post: the one it went to, plus any other whose
    confidence reached the threshold (``topic_scores`` keeps the top three)."""
    topics: set[int] = set()
    if post.topic_id is not None:
        topics.add(post.topic_id)
    for entry in post.topic_scores or ():
        tid, conf = entry.get("topic_id"), entry.get("confidence")
        if tid is not None and conf is not None and float(conf) >= threshold:
            topics.add(int(tid))
    return topics


def competition(
    posts: Sequence[Post],
    margin: float,
    *,
    threshold: float = 0.5,
    min_posts: int = MIN_COMPETE_POSTS,
) -> list[TopicPair]:
    """Pairs of topics that keep competing for the same accepted posts.

    A pair competes when the posts both would accept are at least ``margin`` of the smaller
    topic's accepted posts ("same posts, different reasons"), or when the centroids of their
    sorted posts have cosine >= 0.80 ("same meaning, two names"). ``shared`` counts the posts
    both accept. ``threshold`` is the confidence at which a runner-up score counts as
    acceptance (the layer's decision point).
    """
    accepted: dict[int, set[int]] = {}
    sorted_vectors: dict[int, list[np.ndarray]] = {}
    for post in posts:
        for tid in _accepted_by(post, threshold):
            accepted.setdefault(tid, set()).add(post.id)
        if post.topic_id is not None and post.embedding:
            vec = np.frombuffer(post.embedding, dtype="<f4")
            sorted_vectors.setdefault(post.topic_id, []).append(vec)
    all_vectors = [v for vs in sorted_vectors.values() for v in vs]
    mu = np.mean(all_vectors, axis=0) if all_vectors else None
    centroids = {
        tid: unit_norm(np.mean(vs, axis=0) - mu)
        for tid, vs in sorted_vectors.items()
        if mu is not None
    }
    pairs: list[TopicPair] = []
    for a, b in combinations(sorted(accepted), 2):
        shared = accepted[a] & accepted[b]
        smaller = min(len(accepted[a]), len(accepted[b]))
        if smaller < min_posts:
            continue
        by_overlap = len(shared) / smaller >= margin
        by_meaning = (
            a in centroids
            and b in centroids
            and float(centroids[a] @ centroids[b]) >= CENTROID_COMPETE_COSINE
        )
        if by_overlap or by_meaning:
            pairs.append(TopicPair(a_id=a, b_id=b, shared=len(shared)))
    pairs.sort(key=lambda p: (-p.shared, p.a_id, p.b_id))
    return pairs
