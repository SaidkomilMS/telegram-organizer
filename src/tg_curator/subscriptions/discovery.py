"""New topics found in the unsorted posts, and merges of competing ones (DESIGN §12).

Unsorted posts keep their embeddings so that a tight cluster of them can become a topic
proposal with a name the owner can judge: the closest built-in category, or a sharper name and
description from the connected language model, plus the five posts nearest the centre. Nothing
is created without a tap unless ``auto_create_topics`` is on, and even then at most one topic a
day, so a bad week of news cannot flood the account with channels. ``propose()`` runs only
inside ``ReviewService.build()``; the clustering maths lives in ``ml/cluster.py`` and is
imported when first needed so this module stays testable without the models.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from tg_curator.clock import local_date
from tg_curator.domain import (
    OPEN_PROPOSAL_STATES,
    PROPOSAL_DONE,
    Cluster,
    Example,
    Post,
    PostStatus,
    Proposal,
    Topic,
)
from tg_curator.errors import CuratorError, TopicExists
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED
from tg_curator.subscriptions.review import edit_proposal_message
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

EXAMPLE_COUNT = 5
SORTED_STATUSES = (
    PostStatus.tracked, PostStatus.held, PostStatus.queued, PostStatus.published,
    PostStatus.digest, PostStatus.digested, PostStatus.dropped,
)  # fmt: skip
"""Posts a topic accepted: what ``competition()`` compares two topics over."""


def cluster_module() -> ModuleType:
    """``tg_curator.ml.cluster`` (``find_clusters`` / ``competition``, §10), loaded on use."""
    return importlib.import_module("tg_curator.ml.cluster")


def category_label(key: str) -> str:
    """The English label of a built-in category (``ml/categories.py``), or the key itself
    made readable when the category module cannot answer."""
    try:
        categories = importlib.import_module("tg_curator.ml.categories")
        for category in categories.all():
            if category.key == key:
                return str(category.label)
    except ImportError:
        pass
    return key.replace("_", " ").capitalize()


class Discovery:
    """``propose()`` and ``accept()`` of DESIGN §8."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    # --- proposing ---------------------------------------------------------------------------

    async def propose(self) -> list[Proposal]:
        rt = self._rt
        now = rt.clock.now()
        review_day = local_date(now, rt.settings.general.timezone)
        since = now - timedelta(days=rt.settings.review.cluster_window_days)
        out = await self._propose_topics(review_day, now, since)
        out += await self._propose_merges(review_day, since)
        return out

    async def _propose_topics(
        self, review_day: date, now: datetime, since: datetime
    ) -> list[Proposal]:
        rt = self._rt
        store = rt.store
        s = rt.settings.review
        taken = await self._open_members()
        posts = [
            p
            for p in await store.recent_posts(
                since, statuses=[PostStatus.unsorted], with_embeddings=True
            )
            if p.embedding is not None and p.id not in taken
        ]
        if len(posts) < s.cluster_min_posts:
            return []
        matrix = np.stack([np.frombuffer(p.embedding, dtype="<f4") for p in posts])
        existing = _prototypes(await store.list_topics(), await store.list_examples())
        clusters: list[Cluster] = cluster_module().find_clusters(
            matrix,
            s.cluster_min_posts,
            s.cluster_tightness,
            langs=[p.lang for p in posts],
            existing=existing,
        )
        auto = s.auto_create_topics
        created_today = auto and await self._created_today(now)
        out: list[Proposal] = []
        for cluster in clusters:
            # Checked before naming (``created_today`` implies auto): a cluster that is not
            # proposed sends nothing to the language model (spec: only a proposed topic's posts).
            if created_today:
                log.info(
                    "discovery: cluster of %d posts waits for the next review",
                    len(cluster.member_ids),
                )
                continue
            members = [posts[i] for i in cluster.member_ids]
            examples = _closest(members, cluster.centroid, EXAMPLE_COUNT)
            name, description, category = await self._name(members, examples)
            proposal = await store.create_proposal(
                "new_topic",
                reason=rt.t(
                    "review_reason_cluster", count=len(members), days=s.cluster_window_days
                ),
                review_day=review_day,
                payload={
                    "member_post_ids": [p.id for p in members],
                    "example_post_ids": [p.id for p in examples],
                    "name": name,
                    "description": description,
                    "category": category,
                },
            )
            log.info("discovery: proposed topic %r from %d posts", name, len(members))
            if auto:
                try:
                    topic = await self.accept(proposal.id)
                except TopicExists:
                    log.info("discovery: a topic named %r exists; asking instead", name)
                else:
                    created_today = True
                    if topic.channel_id is None:
                        # Channel pacing (§8) kept the channel from being created yet.
                        wait = rt.topics.channel_wait_minutes() if rt.topics is not None else None
                        await rt.notifier.topic_created(
                            topic.name, has_channel=False, wait_minutes=wait
                        )
                    else:
                        await rt.notifier.topic_created(topic.name)
                    refreshed = await store.get_proposal(proposal.id)
                    proposal = refreshed if refreshed is not None else proposal
            out.append(proposal)
        return out

    async def _propose_merges(self, review_day: date, since: datetime) -> list[Proposal]:
        rt = self._rt
        store = rt.store
        posts = [
            p
            for p in await store.recent_posts(
                since, statuses=list(SORTED_STATUSES), with_embeddings=True
            )
            if p.topic_id is not None
        ]
        if not posts:
            return []
        pairs = cluster_module().competition(posts, rt.settings.review.merge_margin)
        if not pairs:
            return []
        topics = {t.id: t for t in await store.list_topics()}
        open_pairs = {
            frozenset((int(p.payload["a_id"]), int(p.payload["b_id"])))
            for p in await store.proposals_by_state(OPEN_PROPOSAL_STATES, kind="merge_topics")
        }
        out: list[Proposal] = []
        for pair in pairs:
            a, b = topics.get(pair.a_id), topics.get(pair.b_id)
            if a is None or b is None or frozenset((a.id, b.id)) in open_pairs:
                continue
            out.append(
                await store.create_proposal(
                    "merge_topics",
                    reason=rt.t("review_reason_merge", a=a.name, b=b.name, shared=pair.shared),
                    review_day=review_day,
                    payload={"a_id": a.id, "b_id": b.id, "shared": pair.shared},
                )
            )
            log.info("discovery: proposed merging %s into %s", a.key, b.key)
        return out

    async def _open_members(self) -> set[int]:
        """Posts inside an open or accepted new-topic proposal are not clustered again: the
        open one is still being asked about, the accepted one already became a topic (its
        posts are re-sorted by the sorter; a skipped one is free to come back)."""
        taken: set[int] = set()
        states = (*OPEN_PROPOSAL_STATES, PROPOSAL_DONE)
        for p in await self._rt.store.proposals_by_state(states, kind="new_topic"):
            taken.update(int(i) for i in p.payload.get("member_post_ids", []))
        return taken

    async def _created_today(self, now: datetime) -> bool:
        tz = self._rt.settings.general.timezone
        today = local_date(now, tz)
        return any(
            t.origin == "discovered" and local_date(t.created_at, tz) == today
            for t in await self._rt.store.list_topics(None)
        )

    async def _name(
        self, members: Sequence[Post], examples: Sequence[Post]
    ) -> tuple[str, str | None, str | None]:
        """(name, description, category key): the closest built-in category over the whole
        cluster, sharpened by the language model from the example posts when one is connected."""
        rt = self._rt
        totals: dict[str, float] = {}
        for post in members:
            embedding = np.frombuffer(post.embedding or b"", dtype="<f4")
            for key, score in rt.classifier.category_scores(post.text, embedding):
                totals[key] = totals.get(key, 0.0) + score
        category = max(totals, key=lambda k: totals[k]) if totals else None
        name = category_label(category) if category else rt.t("unknown")
        description: str | None = None
        if rt.llm is not None and rt.llm.enabled:
            named = await rt.llm.name_topic(
                [p.text for p in examples],
                existing=[t.name for t in await rt.store.list_topics(active=True)],
                category_label=category_label(category) if category else None,
            )
            if named is not None and named.name.strip():
                name, description = named.name.strip(), named.description.strip() or None
        return name, description, category

    # --- accepting ---------------------------------------------------------------------------

    async def accept(self, proposal_id: int, name: str | None = None) -> Topic:
        rt = self._rt
        proposal = await rt.store.get_proposal(proposal_id)
        if proposal is None or proposal.state not in OPEN_PROPOSAL_STATES:
            raise CuratorError(rt.t("unknown_choice"))
        if proposal.kind == "new_topic":
            return await self._accept_topic(proposal, name)
        if proposal.kind == "merge_topics":
            return await self._accept_merge(proposal)
        raise CuratorError(rt.t("unknown_choice"))

    async def _accept_topic(self, proposal: Proposal, name: str | None) -> Topic:
        rt = self._rt
        store = rt.store
        topics = rt.topics
        if topics is None:
            raise CuratorError(rt.t("error_generic", error="topics service not available"))
        payload = proposal.payload
        since = rt.clock.now() - timedelta(days=rt.settings.review.cluster_window_days)
        # Absorbed = what left ``unsorted`` across the acceptance: create() re-sorts once with
        # the new topic (category prior only) and the re-sort below again with its examples.
        unsorted_before = await self._unsorted_count(since)
        topic = await topics.create(
            name or str(payload.get("name", "")),
            category=payload.get("category"),
            description=payload.get("description"),
            create_channel=True,
            origin="discovered",
        )
        now = rt.clock.now()
        stored = 0
        for post_id in payload.get("member_post_ids", []):
            post = await store.get_post(int(post_id))
            if post is None or post.embedding is None:
                continue
            await store.add_example(
                Example(
                    id=0,
                    topic_id=topic.id,
                    kind="cluster",
                    post_id=post.id,
                    text=post.text,
                    embedding=post.embedding,
                    created_at=now,
                )
            )
            stored += 1
        rt.classifier.reload(await store.list_topics(), await store.list_examples())
        if rt.sorter is not None:
            await rt.sorter.resort_unsorted(since)
        absorbed = max(0, unsorted_before - await self._unsorted_count(since))
        await store.set_proposal_fields(
            proposal.id,
            state=PROPOSAL_DONE,
            decided_at=now,
            executed_at=now,
            result=topic.key,
            payload={**payload, "topic_id": topic.id, "absorbed": absorbed},
        )
        await rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="topics")
        await edit_proposal_message(
            rt, proposal, rt.t("review_outcome_topic_created", name=html_escape(topic.name)), None
        )
        log.info(
            "discovery: created topic %s with %d cluster examples; %d posts re-sorted",
            topic.key, stored, absorbed,
        )  # fmt: skip
        return topic

    async def _unsorted_count(self, since: datetime) -> int:
        """Posts since ``since`` a re-sort may still move: unsorted and not corrected (the
        same filter as ``Sorter.resort_unsorted``)."""
        posts = await self._rt.store.recent_posts(since, statuses=[PostStatus.unsorted])
        return sum(1 for p in posts if not p.corrected)

    async def _accept_merge(self, proposal: Proposal) -> Topic:
        rt = self._rt
        store = rt.store
        topics = rt.topics
        if topics is None:
            raise CuratorError(rt.t("error_generic", error="topics service not available"))
        src = await store.get_topic(int(proposal.payload["a_id"]))
        dst = await store.get_topic(int(proposal.payload["b_id"]))
        if src is None or dst is None or not (src.active and dst.active):
            raise CuratorError(rt.t("unknown_choice"))
        merged = await topics.merge(src.key, dst.key)
        now = rt.clock.now()
        await store.set_proposal_fields(
            proposal.id, state=PROPOSAL_DONE, decided_at=now, executed_at=now, result=merged.key
        )
        await edit_proposal_message(
            rt, proposal, rt.t("review_outcome_merged", name=html_escape(merged.name)), None
        )
        log.info("discovery: merged topic %s into %s", src.key, merged.key)
        return merged


def _prototypes(topics: Sequence[Topic], examples: Sequence[Example]) -> np.ndarray | None:
    """One unit-norm row per active topic with example embeddings: the mean of them.

    §10 drops a cluster that sits within 0.75 of an existing topic's prototype ("more of that
    topic, not a new one"); ``find_clusters`` can only do that when it is handed the
    prototypes, and the examples are the one description of a topic every topic has once it
    sorted anything. A topic without examples has no prototype and cannot veto a cluster.
    """
    active = {t.id for t in topics}
    by_topic: dict[int, list[np.ndarray]] = {}
    for example in examples:
        if example.topic_id in active and example.embedding:
            vector = np.frombuffer(example.embedding, dtype="<f4")
            by_topic.setdefault(example.topic_id, []).append(vector)
    if not by_topic:
        return None
    rows = np.stack([np.mean(vectors, axis=0) for vectors in by_topic.values()])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norms, 1e-12)).astype(np.float32)


def _closest(members: Sequence[Post], centroid: np.ndarray, n: int) -> list[Post]:
    """The ``n`` posts nearest the cluster centre: the ones that show best what it is about."""
    vectors = np.stack([np.frombuffer(p.embedding or b"", dtype="<f4") for p in members])
    centre = np.asarray(centroid, dtype=np.float32)
    norm = float(np.linalg.norm(centre))
    scores = vectors @ (centre / norm if norm else centre)
    order = np.argsort(-scores, kind="stable")
    return [members[int(i)] for i in order[:n]]
