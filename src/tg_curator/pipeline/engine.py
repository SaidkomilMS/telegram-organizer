"""The four questions, answered without touching the database or Telegram (DESIGN §9.2).

``RecentIndex`` is the in-memory window of recent posts the repeat check runs against;
``DecisionEngine`` turns a candidate into a ``Decision``. Both are pure so the live sorter and
the preview run the very same code: the sorter persists what the engine says, the preview only
prints it. Everything the engine needs from the outside (settings, the classifier, the index,
an optional language model) is handed in, so a preview can swap in overridden settings and
``llm=None`` without the engine knowing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from tg_curator.config import Settings
from tg_curator.contracts import LLM, TopicClassifier
from tg_curator.domain import (
    Candidate,
    Chat,
    Decision,
    DupKind,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.ml.embedder import detect_language, numbers_in, numbers_shared
from tg_curator.textutil import canonical_urls, text_hash

URL_MATCH_MIN_COSINE = 0.60
"""Same outside link counts as a repeat only when the texts also resemble each other: a
channel's footer link would otherwise merge everything it posts (§9.2)."""
NUMBER_VETO_MAX_COSINE = 0.97
"""Below this cosine, two number-heavy texts sharing no number are two different reports."""
NUMBER_VETO_MIN_COUNT = 2
SECOND_OPINION_MARGIN = 0.10
"""A borderline post is one whose best score sits within this much below the threshold."""
TOP_SCORES = 3
"""How many topic scores a decision keeps (``posts.topic_scores``, §6)."""
FULL_TRUST = 3.0
"""A fully trusted source (§14.3): every sorted post goes out immediately, whatever
``realtime_strength`` and ``neutral_trust`` are set to (spec "Tuning")."""

FwdKey = tuple[int, int]


# --- text features ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Features:
    """What the repeat check needs from a text, computed once per candidate."""

    text_hash: str
    url_keys: tuple[str, ...]
    fwd_key: FwdKey | None
    lang: str
    numbers: frozenset[str]


def fwd_key_of(fwd_from_chat_id: int | None, fwd_from_message_id: int | None) -> FwdKey | None:
    """``(chat, message)`` of the forwarded original, or ``None`` when not a forward."""
    if fwd_from_chat_id is None or fwd_from_message_id is None:
        return None
    return (fwd_from_chat_id, fwd_from_message_id)


def features_of(candidate: Candidate) -> Features:
    return Features(
        text_hash=text_hash(candidate.text),
        url_keys=tuple(canonical_urls(candidate.urls)),
        fwd_key=fwd_key_of(candidate.fwd_from_chat_id, candidate.fwd_from_message_id),
        lang=detect_language(candidate.text),
        numbers=frozenset(str(n) for n in numbers_in(candidate.text)),
    )


def candidate_from_post(post: Post) -> Candidate:
    """A stored post as the candidate it once was, for replays and re-sorts."""
    return Candidate(
        chat_id=post.chat_id,
        message_id=post.message_id,
        message_ids=list(post.message_ids),
        kind=post.kind,
        posted_at=post.posted_at,
        text=post.text,
        html=post.html,
        urls=list(post.urls),
        media=post.media,
        grouped_id=post.grouped_id,
        views=post.views,
        forwards=post.forwards,
        fwd_from_chat_id=post.fwd_from_chat_id,
        fwd_from_message_id=post.fwd_from_message_id,
        noforwards=post.noforwards,
        via=post.via,
    )


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Little-endian float32 bytes, the storage format of ``posts.embedding`` (§6)."""
    return np.asarray(embedding, dtype="<f4").tobytes()


def embedding_from_bytes(raw: bytes | None) -> np.ndarray | None:
    if raw is None or len(raw) == 0:
        return None
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


# --- the recent index --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexMatch:
    """A stored post the candidate repeats; ``root_id`` is what ``duplicate_of`` points to."""

    post_id: int
    root_id: int
    chat_id: int
    score: float


@dataclass(slots=True)
class _Entry:
    post_id: int
    chat_id: int
    message_id: int
    root_id: int
    text_hash: str
    url_keys: tuple[str, ...]
    fwd_key: FwdKey | None
    posted_at: datetime | None
    lang: str | None
    numbers: frozenset[str]
    row: int


class RecentIndex:
    """Every non-ignored post of the last ``window_days``, duplicates included (§9.2).

    Duplicates are stored with their root so a chain of reposts resolves to the original in
    one lookup. Embeddings live in one unit-norm matrix (grown by doubling, compacted on
    trim) so the semantic check is a single matrix-vector product.
    """

    def __init__(self, window_days: int) -> None:
        self.window = timedelta(days=window_days)
        self._entries: list[_Entry] = []
        self._by_post: dict[int, _Entry] = {}
        self._by_message: dict[FwdKey, _Entry] = {}
        self._by_fwd: dict[FwdKey, list[_Entry]] = {}
        self._by_hash: dict[str, list[_Entry]] = {}
        self._by_url: dict[str, list[_Entry]] = {}
        self._matrix: np.ndarray | None = None
        self._rows = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, post_id: int) -> bool:
        return post_id in self._by_post

    def add(
        self,
        post_id: int,
        chat_id: int,
        message_id: int,
        root_id: int | None,
        text_hash: str,
        url_keys: Iterable[str] | str | None,
        fwd_key: FwdKey | None,
        embedding: np.ndarray | None,
        *,
        posted_at: datetime | None = None,
        lang: str | None = None,
        numbers: Iterable[str] = (),
    ) -> None:
        """Index one post; adding a post that is already there is a no-op.

        ``url_keys`` are every canonical outside link of the post (a single key or ``None``
        is accepted too), so a later post sharing any one of them is a ``url`` repeat;
        ``posted_at`` drives ``remove_older_than`` (an entry without it is never trimmed);
        ``lang`` and ``numbers`` feed the language-aware semantic rule and the number veto.
        """
        if post_id in self._by_post:
            return
        entry = _Entry(
            post_id=post_id,
            chat_id=chat_id,
            message_id=message_id,
            root_id=root_id or post_id,
            text_hash=text_hash,
            url_keys=_keys(url_keys),
            fwd_key=fwd_key,
            posted_at=posted_at,
            lang=lang,
            numbers=frozenset(str(n) for n in numbers),
            row=self._append_row(embedding),
        )
        self._entries.append(entry)
        self._link(entry)

    def remove_older_than(self, ts: datetime) -> None:
        """Drop entries posted before ``ts`` and compact the matrix."""
        keep = [e for e in self._entries if e.posted_at is None or e.posted_at >= ts]
        if len(keep) == len(self._entries):
            return
        old_matrix = self._matrix
        self._entries, self._rows, self._matrix = [], 0, None
        self._by_post, self._by_message = {}, {}
        self._by_fwd, self._by_hash, self._by_url = {}, {}, {}
        for entry in keep:
            vector = None if old_matrix is None else old_matrix[entry.row]
            entry.row = self._append_row(vector)
            self._entries.append(entry)
            self._link(entry)

    # --- the four kinds of repeat ---

    def match_forward(self, fwd_key: FwdKey | None) -> IndexMatch | None:
        """A forward of a message we hold, or a second forward of the same original."""
        if fwd_key is None:
            return None
        entry = self._by_message.get(fwd_key)
        if entry is None:
            twins = self._by_fwd.get(fwd_key)
            entry = twins[0] if twins else None
        return None if entry is None else self._match(entry, 1.0)

    def match_exact(self, text_hash: str) -> IndexMatch | None:
        """Same normalised text; an empty hash (nothing but links and emoji) never matches."""
        if not text_hash:
            return None
        entries = self._by_hash.get(text_hash)
        return None if not entries else self._match(entries[0], 1.0)

    def match_url(self, url_keys: Sequence[str], embedding: np.ndarray) -> IndexMatch | None:
        """Same canonical outside link and a text that resembles it (cosine >= 0.60)."""
        best: tuple[float, _Entry] | None = None
        for key in dict.fromkeys(url_keys):
            for entry in self._by_url.get(key, ()):
                score = self._cosine(entry, embedding)
                if score >= URL_MATCH_MIN_COSINE and (best is None or score > best[0]):
                    best = (score, entry)
        return None if best is None else self._match(best[1], best[0])

    def match_semantic(
        self,
        embedding: np.ndarray,
        *,
        lang: str | None = None,
        numbers: Iterable[str] = (),
        threshold: float = 0.90,
        cross_threshold: float = 0.86,
    ) -> IndexMatch | None:
        """The best cosine that passes the language-aware threshold and the number veto.

        Same-language pairs need ``threshold``, pairs in different languages the lower
        ``cross_threshold`` (translations score lower than two different stories in one
        domain). Two texts that both carry two or more numbers and share none are vetoed below
        ``NUMBER_VETO_MAX_COSINE`` whatever the wording: two match reports, two rate decisions.
        """
        if self._matrix is None or self._rows == 0:
            return None
        own_numbers = frozenset(str(n) for n in numbers)
        scores = self._matrix[: self._rows] @ np.asarray(embedding, dtype=np.float32)
        floor = min(threshold, cross_threshold)
        for row in np.argsort(-scores):
            score = float(scores[row])
            if score < floor:
                break
            entry = self._entries[row]
            needed = threshold if _same_language(lang, entry.lang) else cross_threshold
            if score < needed or _number_veto(own_numbers, entry.numbers, score):
                continue
            return self._match(entry, score)
        return None

    # --- internals ---

    def _match(self, entry: _Entry, score: float) -> IndexMatch:
        return IndexMatch(
            post_id=entry.post_id, root_id=entry.root_id, chat_id=entry.chat_id, score=score
        )

    def _link(self, entry: _Entry) -> None:
        self._by_post[entry.post_id] = entry
        self._by_message[(entry.chat_id, entry.message_id)] = entry
        if entry.fwd_key is not None:
            self._by_fwd.setdefault(entry.fwd_key, []).append(entry)
        if entry.text_hash:
            self._by_hash.setdefault(entry.text_hash, []).append(entry)
        for key in entry.url_keys:
            self._by_url.setdefault(key, []).append(entry)

    def _append_row(self, embedding: np.ndarray | None) -> int:
        """Store one unit vector and return its row; a missing one is zeros and never matches."""
        vector = None if embedding is None else np.asarray(embedding, dtype=np.float32).ravel()
        if self._matrix is None:
            if vector is None:
                self._rows += 1
                return self._rows - 1
            self._matrix = np.zeros((max(16, self._rows * 2 + 1), vector.shape[0]), np.float32)
        if self._rows >= self._matrix.shape[0]:
            grown = np.zeros((self._matrix.shape[0] * 2, self._matrix.shape[1]), np.float32)
            grown[: self._rows] = self._matrix[: self._rows]
            self._matrix = grown
        if vector is not None and vector.shape[0] == self._matrix.shape[1]:
            self._matrix[self._rows] = vector
        self._rows += 1
        return self._rows - 1

    def _cosine(self, entry: _Entry, embedding: np.ndarray) -> float:
        if self._matrix is None:
            return 0.0
        return float(self._matrix[entry.row] @ np.asarray(embedding, dtype=np.float32))


def _keys(url_keys: Iterable[str] | str | None) -> tuple[str, ...]:
    if url_keys is None:
        return ()
    if isinstance(url_keys, str):
        return (url_keys,)
    return tuple(dict.fromkeys(k for k in url_keys if k))


def _same_language(a: str | None, b: str | None) -> bool:
    """Unknown languages are treated as the same so the stricter threshold applies."""
    return a is None or b is None or a == b


def _number_veto(a: frozenset[str], b: frozenset[str], score: float) -> bool:
    """Each element is one source number; an ambiguous one (``1,500``) holds both readings."""
    return (
        len(a) >= NUMBER_VETO_MIN_COUNT
        and len(b) >= NUMBER_VETO_MIN_COUNT
        and score < NUMBER_VETO_MAX_COSINE
        and not numbers_shared(a, b)
    )


# --- the decision engine -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Route:
    status: PostStatus
    hold_until: datetime | None = None


class DecisionEngine:
    """Answers the four questions of §9.2 for one candidate at a time.

    Built fresh from ``rt.settings`` by every caller that must see live setting changes; the
    object itself is cheap, the index it is given is the shared state.
    """

    def __init__(
        self,
        settings: Settings,
        classifier: TopicClassifier,
        index: RecentIndex,
        llm: LLM | None,
    ) -> None:
        self.settings = settings
        self.classifier = classifier
        self.index = index
        self.llm = llm

    async def decide(
        self,
        candidate: Candidate,
        embedding: np.ndarray,
        chat: Chat | None,
        topics: Sequence[Topic],
        now: datetime,
        *,
        features: Features | None = None,
        realtime: bool | None = None,
    ) -> Decision:
        """Repeat? Then topic, strength and wait (``classify``).

        ``features`` lets a caller that computed them already (to index the post afterwards)
        hand them in; ``realtime`` overrides the ``via`` rule — ``False`` sorts like a backfill
        (digest inside the window, else dropped), which is what ``resort_unsorted`` needs.
        """
        feats = features or features_of(candidate)
        repeat = self.repeat(feats, embedding)
        if repeat is not None:
            match, kind = repeat
            return Decision(
                post_id=None,
                candidate=candidate,
                status=PostStatus.duplicate,
                duplicate_of=match.root_id,
                dup_kind=kind,
                dup_score=round(match.score, 4),
            )
        return await self.classify(candidate, embedding, chat, topics, now, realtime=realtime)

    def repeat(self, feats: Features, embedding: np.ndarray) -> tuple[IndexMatch, DupKind] | None:
        """Question 1 against the index, first hit wins: forward, exact, url, semantic."""
        sorting = self.settings.sorting
        match = self.index.match_forward(feats.fwd_key)
        if match is not None:
            return match, "forward"
        match = self.index.match_exact(feats.text_hash)
        if match is not None:
            return match, "exact"
        match = self.index.match_url(feats.url_keys, embedding)
        if match is not None:
            return match, "url"
        match = self.index.match_semantic(
            embedding,
            lang=feats.lang,
            numbers=feats.numbers,
            threshold=sorting.duplicate_similarity,
            cross_threshold=sorting.duplicate_similarity_cross,
        )
        if match is not None:
            return match, "semantic"
        return None

    async def classify(
        self,
        candidate: Candidate,
        embedding: np.ndarray,
        chat: Chat | None,
        topics: Sequence[Topic],
        now: datetime,
        *,
        realtime: bool | None = None,
    ) -> Decision:
        """Questions 2–4 only (the post is known not to be a repeat, or is being re-sorted)."""
        if realtime is None:
            realtime = candidate.via == "live"
        by_id = {t.id: t for t in topics if t.active}
        scores = [
            s for s in self.classifier.predict(candidate.text, embedding) if s.topic_id in by_id
        ]
        top = scores[:TOP_SCORES]
        if not scores:
            return Decision(post_id=None, candidate=candidate, status=PostStatus.unsorted)
        best = scores[0]
        topic = by_id[best.topic_id]
        threshold = topic.strictness or self.settings.sorting.confidence
        sure = best.confidence >= threshold or await self._second_opinion(
            candidate.text,
            topic,
            best.confidence,
            threshold,
            competing=[by_id[s.topic_id] for s in top[1:]],
        )
        if not sure:
            return Decision(
                post_id=None,
                candidate=candidate,
                status=PostStatus.unsorted,
                confidence=best.confidence,
                topic_scores=top,
            )
        strength, would_realtime = self.strength(candidate, chat, 0)
        route = self._route(topic, would_realtime, self.trust(chat), realtime, candidate, now)
        return Decision(
            post_id=None,
            candidate=candidate,
            status=route.status,
            topic_id=topic.id,
            confidence=best.confidence,
            topic_scores=top,
            strength=strength,
            would_realtime=would_realtime,
            hold_until=route.hold_until,
        )

    def strength(
        self, post_or_candidate: Post | Candidate, chat: Chat | None, corroboration: int
    ) -> tuple[float, bool]:
        """``(strength, would_realtime)`` from the source's trust as it is *now* (§9.2 Q3).

        A source below neutral trust never goes out immediately however strong the post; one
        the owner fully trusts (``chats.trust >= FULL_TRUST``) always does, whatever the two
        thresholds say — a high ``neutral_trust`` does not make every chat fully trusted.
        """
        sorting = self.settings.sorting
        trust = self.trust(chat)
        has_link = bool(canonical_urls(post_or_candidate.urls))
        is_long = len(post_or_candidate.text) >= sorting.length_bonus_chars
        strength = (
            trust
            + sorting.link_bonus * has_link
            + sorting.length_bonus * is_long
            + sorting.corroboration_weight * corroboration
        )
        trusted = chat is not None and chat.trust is not None and chat.trust >= FULL_TRUST
        would_realtime = trusted or (
            strength >= sorting.realtime_strength and trust >= sorting.neutral_trust
        )
        return strength, would_realtime

    def trust(self, chat: Chat | None) -> float:
        """``chats.trust`` when set, else neutral; a chat not yet stored is neutral too."""
        if chat is None or chat.trust is None:
            return self.settings.sorting.neutral_trust
        return chat.trust

    async def _second_opinion(
        self,
        text: str,
        topic: Topic,
        confidence: float,
        threshold: float,
        *,
        competing: Sequence[Topic] = (),
    ) -> bool:
        """Ask the connected model about a borderline post when the switch is on (§9.2 Q2)."""
        sorting = self.settings.sorting
        if not sorting.second_opinion or self.llm is None or not self.llm.enabled:
            return False
        if confidence < threshold - SECOND_OPINION_MARGIN:
            return False
        return bool(await self.llm.second_opinion(text, topic, competing=competing))

    def _route(
        self,
        topic: Topic,
        would_realtime: bool,
        trust: float,
        realtime: bool,
        candidate: Candidate,
        now: datetime,
    ) -> _Route:
        """Question 4, with the backfill rule of §9.2's last paragraph."""
        if topic.channel_id is None:
            return _Route(PostStatus.tracked)
        if not realtime:
            window = timedelta(hours=self.settings.digest.window_hours)
            inside = candidate.posted_at >= now - window
            return _Route(PostStatus.digest if inside else PostStatus.dropped)
        if would_realtime:
            return _Route(PostStatus.queued)
        if trust < self.settings.sorting.neutral_trust:
            return _Route(PostStatus.digest)
        hold = timedelta(minutes=self.settings.sorting.hold_minutes)
        return _Route(PostStatus.held, hold_until=now + hold)
