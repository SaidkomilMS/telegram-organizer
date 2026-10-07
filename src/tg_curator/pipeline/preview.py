"""Replay of the stored backlog with the current (or overridden) settings; nothing written
(DESIGN §9.6).

The preview runs the very same ``DecisionEngine`` as the live sorter over the posts of the
last few days, in ``posted_at`` order, against a fresh in-memory index — so what it prints is
what the sorter would have decided with these settings. What the sorter does *after* a
decision is replayed too (§9.3): a repeat from another chat corroborates its original, whose
strength is recomputed and which is promoted to real time while it is still held (or waiting
for a digest not yet composed), and a hold that runs out sends the post to the digest. So
"immediate" is the post's final state, not its state the moment it arrived. Overrides land on
a copy of the settings, the language model is never asked, and the database is only read.
"""

from __future__ import annotations

import asyncio
import heapq
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from tg_curator.clock import local_date, scheduled_moment
from tg_curator.config import Settings
from tg_curator.domain import (
    Candidate,
    Chat,
    Decision,
    Post,
    PostStatus,
    PreviewReport,
    Topic,
)
from tg_curator.errors import ConfigError
from tg_curator.pipeline.engine import (
    DecisionEngine,
    RecentIndex,
    candidate_from_post,
    embedding_from_bytes,
    features_of,
)
from tg_curator.runtime import Runtime
from tg_curator.textutil import first_line

LINE_CHARS = 60
"""How much of a post the text report shows; the issue template wants decisions, not posts."""
ROOT_CHARS = 40
"""How much of the original a repeat's line quotes."""
PROMOTABLE = (PostStatus.held, PostStatus.digest)
"""Statuses a live original may be promoted from (the sorter's ``PROMOTABLE``)."""


@dataclass
class _Root:
    """An original post of the replay and what the sorter would do to it later."""

    decision: Decision
    chat: Chat | None
    counted: set[int] = field(default_factory=set)
    """Chats carrying the story, its own included (corroboration = len - 1)."""
    digest_at: datetime | None = None
    """When the digest that would carry it is composed (it can be promoted until then)."""


class PreviewService:
    """``PreviewService`` of §8."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    async def replay(
        self,
        *,
        days: int = 3,
        topic_key: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> PreviewReport:
        rt = self._rt
        rest, trust = split_trust_overrides(overrides or {})
        settings, strictness = apply_overrides(rt.settings, rest)
        now = rt.clock.now()
        topics = [
            replace(t, strictness=strictness[t.key]) if t.key in strictness else t
            for t in await rt.store.list_topics(active=True)
        ]
        only = _topic_filter(topics, topic_key)
        posts = await rt.store.recent_posts(now - timedelta(days=days), with_embeddings=True)
        ignored = sum(1 for p in posts if p.status == PostStatus.ignored)
        replayed = [p for p in posts if p.status != PostStatus.ignored]
        vectors = await self._vectors(replayed)

        index = RecentIndex(settings.sorting.dedup_window_days)
        engine = DecisionEngine(settings, rt.classifier, index, None)
        chats: dict[int, Chat | None] = {}
        per_topic: Counter[str] = Counter({t.key: 0 for t in topics})
        names = {t.id: t.key for t in topics}
        repeats = unsorted = 0
        decisions: list[Decision] = []
        later = _Aftermath(engine, settings)
        for post, embedding in zip(replayed, vectors, strict=True):
            later.expire(post.posted_at)
            index.remove_older_than(post.posted_at - index.window)
            if post.chat_id not in chats:
                chat = await rt.store.get_chat(post.chat_id)
                if chat is not None and post.chat_id in trust:
                    chat = replace(chat, trust=trust[post.chat_id])
                chats[post.chat_id] = chat
            candidate = candidate_from_post(post)
            feats = features_of(candidate)
            decision = await engine.decide(
                candidate, embedding, chats[post.chat_id], topics, post.posted_at, features=feats
            )
            decision.post_id = post.id
            if post.status == PostStatus.rejected:
                decision.status = PostStatus.rejected
            if decision.status == PostStatus.duplicate:
                later.corroborate(decision, post.chat_id)
            else:
                later.original(post.id, decision, chats[post.chat_id])
            index.add(
                post.id,
                post.chat_id,
                post.message_id,
                decision.duplicate_of,
                feats.text_hash,
                feats.url_keys,
                feats.fwd_key,
                embedding,
                posted_at=post.posted_at,
                lang=feats.lang,
                numbers=feats.numbers,
            )
            if decision.status == PostStatus.duplicate:
                repeats += 1
            elif decision.status == PostStatus.unsorted:
                unsorted += 1
            elif decision.topic_id is not None and decision.status != PostStatus.rejected:
                per_topic[names[decision.topic_id]] += 1
            if only is None or decision.topic_id == only:
                decisions.append(decision)
        later.expire(now)
        return PreviewReport(
            days=days,
            per_topic=dict(per_topic),
            repeats=repeats,
            unsorted=unsorted,
            ignored=ignored,
            decisions=decisions,
        )

    async def _vectors(self, posts: list[Post]) -> list[np.ndarray]:
        """Stored embeddings; the missing ones are computed in memory, never written."""
        vectors = [embedding_from_bytes(p.embedding) for p in posts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            computed = await asyncio.to_thread(
                self._rt.embedder.embed, [posts[i].text for i in missing]
            )
            for i, vector in zip(missing, computed, strict=True):
                vectors[i] = vector
        return [v for v in vectors if v is not None]


# --- what the sorter does after a decision (§9.3) ------------------------------------------------


class _Aftermath:
    """Corroboration, promotion and hold expiry as ``Sorter`` does them, in memory."""

    def __init__(self, engine: DecisionEngine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings
        self._roots: dict[int, _Root] = {}
        self._holds: list[tuple[datetime, int]] = []

    def original(self, post_id: int, decision: Decision, chat: Chat | None) -> None:
        root = self._roots[post_id] = _Root(decision, chat, {decision.candidate.chat_id})
        if decision.status == PostStatus.held and decision.hold_until is not None:
            heapq.heappush(self._holds, (decision.hold_until, post_id))
        elif decision.status == PostStatus.digest:
            root.digest_at = _digest_moment(self._settings, decision.candidate.posted_at)

    def corroborate(self, repeat: Decision, chat_id: int) -> None:
        """The sorter's ``_corroborated``: a repeat from a chat not yet counted adds one to
        its original, whose strength is recomputed; a live original still held, or waiting
        for a digest not yet composed, goes out immediately once it is strong enough."""
        root = self._roots.get(repeat.duplicate_of or 0)
        if root is None or chat_id in root.counted:
            return
        root.counted.add(chat_id)
        d = root.decision
        d.corroboration = len(root.counted) - 1
        d.strength, d.would_realtime = self._engine.strength(
            d.candidate, root.chat, d.corroboration
        )
        if not d.would_realtime or d.candidate.via != "live" or d.status not in PROMOTABLE:
            return
        at = repeat.candidate.posted_at
        if d.status == PostStatus.held and d.hold_until is not None and at >= d.hold_until:
            return  # the hold ran out at this very instant: expiry decides
        if d.status == PostStatus.digest and root.digest_at is not None and at >= root.digest_at:
            return  # already in a composed digest
        d.status, d.hold_until = PostStatus.queued, None

    def expire(self, at: datetime) -> None:
        """The sorter's hold expiry (``Sorter._expire``) for every hold that ran out by ``at``."""
        while self._holds and self._holds[0][0] <= at:
            hold_until, post_id = heapq.heappop(self._holds)
            root = self._roots[post_id]
            d = root.decision
            if d.status != PostStatus.held or d.hold_until != hold_until:
                continue  # promoted (or rejected) meanwhile
            d.strength, d.would_realtime = self._engine.strength(
                d.candidate, root.chat, d.corroboration
            )
            if d.would_realtime:
                d.status = PostStatus.queued
            else:
                d.status = PostStatus.digest
                root.digest_at = _digest_moment(self._settings, hold_until)
            d.hold_until = None


def _digest_moment(settings: Settings, at: datetime) -> datetime:
    """The first scheduled digest at or after ``at`` (§9.5)."""
    tz = settings.general.timezone
    hour, minute = settings.digest.hour, settings.digest.minute
    day = local_date(at, tz)
    moment = scheduled_moment(day, hour, minute, tz)
    if moment < at:
        moment = scheduled_moment(day + timedelta(days=1), hour, minute, tz)
    return moment


# --- overrides ---------------------------------------------------------------------------------

PREVIEWABLE_SECTIONS = frozenset({"sorting"})
PREVIEWABLE_KEYS = frozenset(
    {"digest.hour", "digest.minute", "digest.window_hours", "general.timezone"}
)
"""The settings a replay of stored posts reflects: the engine's ``[sorting]`` and the digest
schedule that decides how long a post can still be promoted. Anything else (intake floors,
review, posting style, ...) cannot change a decision, so a preview of it would silently show
the same numbers; such keys are refused instead (SPEC "Every change can be previewed")."""


def previewable(key: str) -> bool:
    """Whether ``apply_overrides`` takes the ``"section.key"`` setting (see above)."""
    return key.split(".", 1)[0] in PREVIEWABLE_SECTIONS or key in PREVIEWABLE_KEYS


def split_trust_overrides(
    overrides: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, float]]:
    """Take ``"sources.<chat_id>.trust"`` (0–3, as in ``[[sources]]``) out of ``overrides``.

    Trust lives in ``chats.trust`` (mirrored from the file), not in a settings section, so
    the replay applies it to the chat rows it loads; the rest goes to ``apply_overrides``.
    """
    rest: dict[str, Any] = {}
    trust: dict[int, float] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "sources" or parts[2] != "trust":
            if parts[0] == "sources":
                raise ConfigError(f'preview: cannot override "{key}"; use sources.<chat id>.trust')
            rest[key] = value
            continue
        try:
            chat_id = int(parts[1])
        except ValueError:
            raise ConfigError(
                f'preview: "{key}": the chat must be given by its numeric id'
            ) from None
        try:
            level = float(str(value).strip()) if not isinstance(value, bool) else -1.0
        except ValueError:
            raise ConfigError(f"preview: {key} = {value!r} is not a number") from None
        if level not in (0.0, 1.0, 2.0, 3.0):
            raise ConfigError(f"preview: {key} = {value!r}: must be 0, 1, 2 or 3")
        trust[chat_id] = level
    return rest, trust


def apply_overrides(
    settings: Settings, overrides: Mapping[str, Any]
) -> tuple[Settings, dict[str, float | None]]:
    """A deep copy of ``settings`` with dotted overrides applied, plus per-topic strictness.

    ``"section.key"`` sets a settings value (validated like the file is);
    ``"topics.<key>.strictness"`` is returned separately because the engine reads strictness
    from the topic rows, not from the file. Anything else is a ``ConfigError`` naming the key.
    """
    copy = settings.model_copy(deep=True)
    strictness: dict[str, float | None] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "topics" and parts[2] == "strictness":
            try:
                level = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"preview: {key} = {value!r} is not a number") from None
            if not 0.0 <= level <= 1.0:
                raise ConfigError(f"preview: {key} = {value!r}: must be between 0 and 1")
            strictness[parts[1]] = level or None
            continue
        if len(parts) != 2:
            raise ConfigError(
                f'preview: cannot override "{key}"; use "section.key" or "topics.<key>.strictness"'
            )
        section = getattr(copy, parts[0], None)
        if not isinstance(section, BaseModel) or parts[1] not in type(section).model_fields:
            raise ConfigError(f'preview: "{key}" is not a settings key')
        if not previewable(key):
            if parts[0] == "groups":
                raise ConfigError(
                    f'preview: "{key}" is applied at intake and cannot be previewed on stored '
                    "posts; it takes effect for new messages"
                )
            raise ConfigError(
                f'preview: "{key}" does not change what is sorted or sent at once, so there '
                "is nothing to preview; set it directly"
            )
        try:
            setattr(section, parts[1], value)
        except ValidationError as exc:
            reason = "; ".join(e["msg"] for e in exc.errors())
            raise ConfigError(f"preview: {key} = {value!r}: {reason}") from None
    return copy, strictness


def _topic_filter(topics: list[Topic], topic_key: str | None) -> int | None:
    if topic_key is None:
        return None
    for topic in topics:
        if topic.key == topic_key:
            return topic.id
    raise ConfigError(f'preview: no active topic has the key "{topic_key}"; see `curator topics`')


# --- text rendering ----------------------------------------------------------------------------


def render_text(
    report: PreviewReport,
    *,
    topic_names: Mapping[int, str] | None = None,
    chat_names: Mapping[int, str] | None = None,
) -> str:
    """One line per decision (§9.6): post number, time, source, first line, verdict, status,
    immediate. A repeat names its original: the original's number (which starts the
    original's own line), its source and first words.

    Topics and chats are shown by name when the caller hands in the mappings (the CLI has
    them from the store) and by id otherwise — the report itself carries only ids.
    """
    topics = topic_names or {}
    chats = chat_names or {}
    counts = ", ".join(f"{key}: {n}" for key, n in report.per_topic.items()) or "no topics"
    head = (
        f"preview: last {report.days} days, {len(report.decisions)} decisions — {counts} · "
        f"repeats: {report.repeats} · unsorted: {report.unsorted} · ignored: {report.ignored}"
    )
    lines = [head]
    originals = {d.post_id: d.candidate for d in report.decisions if d.post_id is not None}
    for d in report.decisions:
        c = d.candidate
        source = chats.get(c.chat_id, f"chat {c.chat_id}")
        text = first_line(c.text, LINE_CHARS) or "(no text)"
        number = f"#{d.post_id}  " if d.post_id is not None else ""
        verdict = _verdict(d, topics, chats, originals)
        lines.append(
            f"{number}{_stamp(c.posted_at)}  {source}  {text}  | {verdict}  [{d.status.value}]"
        )
    return "\n".join(lines)


def _stamp(at: datetime) -> str:
    return at.strftime("%Y-%m-%d %H:%M")


def _verdict(
    d: Decision,
    topics: Mapping[int, str],
    chats: Mapping[int, str],
    originals: Mapping[int, Candidate],
) -> str:
    if d.status == PostStatus.duplicate:
        score = f"({d.dup_kind} {d.dup_score or 0.0:.2f})"
        original = originals.get(d.duplicate_of or 0)
        if original is None:
            return f"repeat of post #{d.duplicate_of} {score}"
        source = chats.get(original.chat_id, f"chat {original.chat_id}")
        quote = first_line(original.text, ROOT_CHARS) or "(no text)"
        return f"repeat of post #{d.duplicate_of} ({source}: {quote}) {score}"
    if d.status == PostStatus.rejected:
        return "not for me"
    if d.topic_id is None:
        if d.topic_scores:
            best = d.topic_scores[0]
            label = topics.get(best.topic_id, f"topic #{best.topic_id}")
            return f"unsorted (best {label} {best.confidence:.2f})"
        return "unsorted"
    label = topics.get(d.topic_id, f"topic #{d.topic_id}")
    immediate = "yes" if d.would_realtime else "no"
    return f"{label} {d.confidence:.2f}  immediate: {immediate}"
