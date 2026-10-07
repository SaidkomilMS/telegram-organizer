"""Dataclasses and enums shared by every module (DESIGN §6, §7, §8).

Table-backed dataclasses carry exactly the columns of their table, same names and types, so
the Store can build them by column name and every module reads the same shape. They are
keyword-only: column order is irrelevant and defaults can sit wherever the schema puts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

    from tg_curator.telegram.gateway import Account

# --- literal vocabularies -------------------------------------------------------------------

ChatKind = Literal["channel", "group"]
MediaKind = Literal["photo", "video", "file", "album", "other"]
ChatRole = Literal["source", "output", "staging"]
TopicOrigin = Literal["user", "discovered"]
PostKind = Literal["post", "unit"]
Via = Literal["live", "backfill"]
DupKind = Literal["forward", "exact", "url", "semantic"]
ExampleKind = Literal["example", "channel", "correction", "cluster", "description"]
ProposalKind = Literal["folder", "mute", "archive", "leave", "new_topic", "merge_topics"]
PublishStyle = Literal["repost", "forward"]


class PostStatus(StrEnum):
    """Where a post stands in the pipeline (§8); ``rejected`` = "not for me", counts as nothing."""

    ignored = "ignored"
    duplicate = "duplicate"
    unsorted = "unsorted"
    tracked = "tracked"
    held = "held"
    queued = "queued"
    published = "published"
    digest = "digest"
    digested = "digested"
    dropped = "dropped"
    rejected = "rejected"


# Publication states (§6 publications.state).
PUB_PENDING = "pending"
PUB_SENDING = "sending"
PUB_SENT = "sent"
PUB_FAILED = "failed"
PUB_CANCELLED = "cancelled"
PUB_RETRACTED = "retracted"
PUBLICATION_STATES = (PUB_PENDING, PUB_SENDING, PUB_SENT, PUB_FAILED, PUB_CANCELLED, PUB_RETRACTED)

# Digest states (§6 digests.state).
DIGEST_PENDING = "pending"
DIGEST_SENDING = "sending"
DIGEST_SENT = "sent"
DIGEST_FAILED = "failed"
DIGEST_CANCELLED = "cancelled"
DIGEST_STATES = (DIGEST_PENDING, DIGEST_SENDING, DIGEST_SENT, DIGEST_FAILED, DIGEST_CANCELLED)

# Proposal states (§6 proposals.state).
PROPOSAL_PROPOSED = "proposed"
PROPOSAL_CONFIRMING = "confirming"
PROPOSAL_APPROVED = "approved"
PROPOSAL_DONE = "done"
PROPOSAL_FAILED = "failed"
PROPOSAL_SKIPPED = "skipped"
PROPOSAL_NEVER = "never"
PROPOSAL_UNDONE = "undone"
PROPOSAL_STATES = (
    PROPOSAL_PROPOSED,
    PROPOSAL_CONFIRMING,
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PROPOSAL_FAILED,
    PROPOSAL_SKIPPED,
    PROPOSAL_NEVER,
    PROPOSAL_UNDONE,
)
OPEN_PROPOSAL_STATES = (PROPOSAL_PROPOSED, PROPOSAL_CONFIRMING, PROPOSAL_APPROVED)


class KV:
    """The namespaced ``kv`` keys (§7). Only these names are ever written to the table."""

    CLAIM_CODE_HASH = "claim.code_hash"
    CLAIM_ATTEMPTS = "claim.attempts"
    BOT_CONVERSATION = "bot.conversation"
    BOT_ID = "bot.id"
    BOT_USERNAME = "bot.username"
    SETUP_STEP = "setup.step"
    SETUP_PREVIEW_SHOWN = "setup.preview_shown"
    """When the bot last sent the /preview report: what marks setup step 6 done (§14.15)."""
    ACCOUNT_ID = "account.id"
    ACCOUNT_LOSS_NOTIFIED = "account.loss_notified"
    SERVICE_PAUSED = "service.paused"
    SERVICE_WENT_LIVE_AT = "service.went_live_at"
    INTAKE_LAST_MESSAGE_AT = "intake.last_message_at"
    INTAKE_OPEN_ALBUMS = "intake.open_albums"
    """Live album parts still settling, so a stop or crash does not lose them (§9.1)."""
    WATCHDOG_WARNED = "watchdog.warned"
    BACKFILL_LAST_RUN_AT = "backfill.last_run_at"
    REVIEW_LAST_DAY = "review.last_day"
    FOLDERS_CURATED_ID = "folders.curated_id"
    FOLDERS_LOW_SIGNAL_ID = "folders.low_signal_id"
    FOLDERS_DISABLED = "folders.disabled"
    ML_EMBEDDER_ID = "ml.embedder_id"
    ML_USER_LAYER = "ml.user_layer"
    LEAVE_LOG = "leave.log"
    TOPICS_CREATE_LOG = "topics.create_log"
    LLM_CAP_CLOSED = "llm.cap_closed"
    """``{"month": "YYYY-MM", "cap_usd": float}``: the month the LLM cap closed (§7)."""


# --- table-backed dataclasses (§6) -----------------------------------------------------------


@dataclass(kw_only=True)
class Chat:
    id: int
    kind: ChatKind
    title: str
    username: str | None = None
    noforwards: bool = False
    role: ChatRole = "source"
    active: bool = True
    first_seen_at: datetime
    left_at: datetime | None = None
    last_message_at: datetime | None = None
    is_creator: bool = False
    is_admin: bool = False
    trust: float | None = None
    keep: bool = False
    muted_until: datetime | None = None
    archived: bool = False
    in_low_signal: bool = False
    views_baseline: float | None = None
    baseline_at: datetime | None = None


@dataclass(kw_only=True)
class Topic:
    id: int
    key: str
    name: str
    channel_id: int | None = None
    category: str | None = None
    description: str | None = None
    example_channel: str | None = None
    strictness: float | None = None
    active: bool = True
    origin: TopicOrigin = "user"
    created_at: datetime


@dataclass(kw_only=True)
class NewPost:
    """What intake and the sorter hand to ``insert_post``: a ``Post`` before it has an id,
    an ingestion time or a decision (§8)."""

    chat_id: int
    message_id: int
    kind: PostKind
    message_ids: list[int]
    grouped_id: int | None = None
    posted_at: datetime
    via: Via
    text: str
    html: str | None = None
    lang: str | None = None
    text_hash: str
    url_key: str | None = None
    urls: list[str]
    media: MediaKind | None = None
    noforwards: bool = False
    fwd_from_chat_id: int | None = None
    fwd_from_message_id: int | None = None
    embedding: bytes | None = None
    views: int | None = None
    forwards: int | None = None
    views_at: datetime | None = None


@dataclass(kw_only=True)
class Post(NewPost):
    id: int
    ingested_at: datetime
    status: PostStatus
    ignore_reason: str | None = None
    duplicate_of: int | None = None
    dup_kind: DupKind | None = None
    dup_score: float | None = None
    topic_id: int | None = None
    confidence: float | None = None
    topic_scores: list[dict[str, Any]] | None = None
    corrected: bool = False
    corroboration: int = 0
    corroborating_chats: list[int] = field(default_factory=list)
    strength: float | None = None
    would_realtime: bool = False
    hold_until: datetime | None = None
    decided_at: datetime | None = None
    published_at: datetime | None = None
    summary: str | None = None


@dataclass(kw_only=True)
class Publication:
    id: int
    post_id: int
    topic_id: int
    channel_id: int | None = None
    style: PublishStyle | None = None
    state: str
    message_ids: list[int] = field(default_factory=list)
    staging_ids: list[int] = field(default_factory=list)
    shown_corroboration: int = 0
    edited_at: datetime | None = None
    attempts: int = 0
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    sent_at: datetime | None = None
    moved_from: list[dict[str, Any]] | None = None


@dataclass(kw_only=True)
class Digest:
    id: int
    topic_id: int
    channel_id: int
    day: date
    seq: int = 0
    manual: bool = False
    state: str
    body: list[str]
    message_ids: list[int] = field(default_factory=list)
    item_count: int
    attempts: int = 0
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    sent_at: datetime | None = None


@dataclass(kw_only=True)
class DigestItem:
    digest_id: int
    position: int
    post_id: int
    line: str


@dataclass(kw_only=True)
class Example:
    id: int
    topic_id: int | None
    kind: ExampleKind
    post_id: int | None = None
    wrong_topic_id: int | None = None
    text: str
    embedding: bytes
    weight: float = 1.0
    created_at: datetime


@dataclass(kw_only=True)
class Proposal:
    id: int
    kind: ProposalKind
    chat_id: int | None = None
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)
    state: str
    bot_message_id: int | None = None
    review_day: date
    created_at: datetime
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    result: str | None = None


# --- pipeline value objects (§8) -------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Candidate:
    """What intake hands to the sorter: one channel post (album = one) or one group unit."""

    chat_id: int
    message_id: int
    message_ids: list[int]
    kind: PostKind
    posted_at: datetime
    text: str
    html: str | None
    urls: list[str]
    media: MediaKind | None
    grouped_id: int | None
    views: int | None
    forwards: int | None
    fwd_from_chat_id: int | None
    fwd_from_message_id: int | None
    noforwards: bool
    via: Via


@dataclass(frozen=True)
class TopicScore:
    topic_id: int
    confidence: float


@dataclass(kw_only=True)
class Decision:
    """The engine's answer to the four questions; also the unit the preview prints."""

    post_id: int | None
    candidate: Candidate
    status: PostStatus
    ignore_reason: str | None = None
    duplicate_of: int | None = None
    dup_kind: DupKind | None = None
    dup_score: float | None = None
    topic_id: int | None = None
    confidence: float | None = None
    topic_scores: list[TopicScore] = field(default_factory=list)
    strength: float | None = None
    corroboration: int = 0
    would_realtime: bool = False
    hold_until: datetime | None = None


# --- result objects (§8, exact field lists) --------------------------------------------------


@dataclass(frozen=True)
class ChatSyncResult:
    total: int
    new: list[int]
    left: list[int]
    outputs: int


@dataclass(frozen=True)
class BackfillResult:
    chats: int
    messages: int
    candidates: int
    submitted: int
    skipped_chats: list[int]
    per_chat: dict[int, int]


@dataclass(frozen=True)
class TopicSyncResult:
    total: int
    resolved: int
    unresolved: list[str]


@dataclass(frozen=True)
class PreviewReport:
    days: int
    per_topic: dict[str, int]
    repeats: int
    unsorted: int
    ignored: int
    decisions: list[Decision]


@dataclass(frozen=True)
class DigestLine:
    """A draft line; not table-backed, it exists before any ``digests`` row does."""

    position: int
    post_id: int
    line: str


@dataclass(frozen=True)
class DigestDraft:
    topic_key: str
    day: date
    parts: list[str]
    items: list[DigestLine]


@dataclass(frozen=True)
class DigestResult:
    topic_key: str
    day: date
    seq: int
    item_count: int
    message_ids: list[int]
    skipped_reason: str | None


@dataclass(frozen=True)
class MoveResult:
    republished: bool
    stubbed: bool
    new_message_ids: list[int]


@dataclass(frozen=True)
class CorrectionResult:
    post_id: int
    old_topic_id: int | None
    new_topic_id: int | None
    moved: bool


@dataclass(frozen=True)
class AccountStatus:
    bound: bool
    account: Account | None
    step: Literal["none", "phone", "code", "password", "ok"]


@dataclass(frozen=True)
class TopicName:
    name: str
    description: str


@dataclass(eq=False)
class Cluster:
    """A tight group of unsorted posts; ``eq`` is off because the centroid is an array."""

    member_ids: list[int]
    centroid: np.ndarray
    tightness: float


@dataclass(frozen=True)
class TopicPair:
    a_id: int
    b_id: int
    shared: int


@dataclass(frozen=True)
class ChatStats:
    chat_id: int
    title: str
    volume: int
    sorted: int
    signal: float
    duplicates: int
    duplicate_share: float
    published: int
    observed_days: int
    top_repeated_chat_id: int | None


@dataclass(frozen=True)
class RenderedPost:
    header: str
    media_line: str | None
    parts: list[str]


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    label: str
    models: list[str]
    privacy_line: str


@dataclass(frozen=True)
class Category:
    key: str
    label: str
