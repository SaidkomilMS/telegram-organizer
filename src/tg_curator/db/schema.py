"""The database schema (DESIGN §6) as SQLAlchemy Core tables.

Everything here is portable between SQLite and PostgreSQL on purpose: only the types listed in
§6 are used, JSON is text, and datetimes go through ``UTCDateTime`` so a value read back is an
aware UTC datetime on both engines (SQLite has no timezone-aware column type; the decorator
normalises on the way in and re-attaches UTC on the way out).

Modules that need a query of their own build it against these tables and run it through
``Store.begin()`` / ``Store.execute()`` (§7).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator

metadata = sa.MetaData()


class JSONText(TypeDecorator[Any]):
    """JSON stored as text, so SQLite and PostgreSQL behave identically (§6).

    ``None`` is stored as SQL NULL, not as the JSON literal ``null``: columns such as
    ``posts.topic_scores`` use NULL for "no value".
    """

    impl = sa.Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def process_result_value(self, value: str | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        return json.loads(value)


class UTCDateTime(TypeDecorator[datetime]):
    """``DateTime(timezone=True)`` that is always UTC in Python (§1, §6).

    SQLite stores the wall-clock fields and forgets the offset, so a value is converted to UTC
    before it is written and gets UTC re-attached when it is read; PostgreSQL's ``timestamptz``
    returns aware values, which are normalised to UTC as well. Naive datetimes are refused on
    write because a naive value is always a bug in this code base.
    """

    impl = sa.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime written to the database; all datetimes are aware UTC")
        value = value.astimezone(UTC)
        if dialect.name == "sqlite":
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _bool(default: bool) -> dict[str, Any]:
    """Column keyword arguments for a boolean with a portable server default."""
    return {"default": default, "server_default": sa.true() if default else sa.false()}


meta = sa.Table(
    "meta",
    metadata,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("value", sa.Text, nullable=False),
)

kv = sa.Table(
    "kv",
    metadata,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("value", JSONText, nullable=True),
    sa.Column("updated_at", UTCDateTime, nullable=False),
)

chats = sa.Table(
    "chats",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("title", sa.Text, nullable=False),
    sa.Column("username", sa.Text, nullable=True),
    sa.Column("noforwards", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("role", sa.Text, nullable=False, default="source", server_default="source"),
    sa.Column("active", sa.Boolean, nullable=False, **_bool(True)),
    sa.Column("first_seen_at", UTCDateTime, nullable=False),
    sa.Column("left_at", UTCDateTime, nullable=True),
    sa.Column("last_message_at", UTCDateTime, nullable=True),
    sa.Column("is_creator", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("is_admin", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("trust", sa.Float, nullable=True),
    sa.Column("keep", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("muted_until", UTCDateTime, nullable=True),
    sa.Column("archived", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("in_low_signal", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("views_baseline", sa.Float, nullable=True),
    sa.Column("baseline_at", UTCDateTime, nullable=True),
)

chat_daily = sa.Table(
    "chat_daily",
    metadata,
    sa.Column("chat_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("day", sa.Date, primary_key=True),
    sa.Column("messages", sa.Integer, nullable=False, default=0, server_default="0"),
)

group_messages = sa.Table(
    "group_messages",
    metadata,
    sa.Column("chat_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("message_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("sender_id", sa.BigInteger, nullable=True),
    sa.Column("reply_to_id", sa.BigInteger, nullable=True),
    sa.Column("date", UTCDateTime, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("html", sa.Text, nullable=True),
    sa.Column("urls", JSONText, nullable=False),
    sa.Column("media", sa.Text, nullable=True),
    sa.Column("fwd_from_chat_id", sa.BigInteger, nullable=True),
    sa.Column("fwd_from_message_id", sa.BigInteger, nullable=True),
    sa.Column("unit_root_id", sa.BigInteger, nullable=True),
    sa.Column("topic_id", sa.BigInteger, nullable=True),  # forum topic; NULL = General / none
    sa.Column("grouped_id", sa.BigInteger, nullable=True),  # album id: an album counts once
    sa.Column("closed", sa.Boolean, nullable=False, **_bool(False)),
    sa.Index("ix_group_messages_chat_closed", "chat_id", "closed"),
    sa.Index("ix_group_messages_date", "date"),
)

topics = sa.Table(
    "topics",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("key", sa.Text, nullable=False, unique=True),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("channel_id", sa.BigInteger, nullable=True),
    sa.Column("category", sa.Text, nullable=True),
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("example_channel", sa.Text, nullable=True),
    sa.Column("strictness", sa.Float, nullable=True),
    sa.Column("active", sa.Boolean, nullable=False, **_bool(True)),
    sa.Column("origin", sa.Text, nullable=False, default="user", server_default="user"),
    sa.Column("created_at", UTCDateTime, nullable=False),
)

posts = sa.Table(
    "posts",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("chat_id", sa.BigInteger, nullable=False),
    sa.Column("message_id", sa.BigInteger, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("message_ids", JSONText, nullable=False),
    sa.Column("grouped_id", sa.BigInteger, nullable=True),
    sa.Column("posted_at", UTCDateTime, nullable=False),
    sa.Column("ingested_at", UTCDateTime, nullable=False),
    sa.Column("via", sa.Text, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("html", sa.Text, nullable=True),
    sa.Column("lang", sa.Text, nullable=True),
    sa.Column("text_hash", sa.Text, nullable=False, index=True),
    sa.Column("url_key", sa.Text, nullable=True, index=True),
    sa.Column("urls", JSONText, nullable=False),
    sa.Column("media", sa.Text, nullable=True),
    sa.Column("noforwards", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("fwd_from_chat_id", sa.BigInteger, nullable=True),
    sa.Column("fwd_from_message_id", sa.BigInteger, nullable=True),
    sa.Column("embedding", sa.LargeBinary, nullable=True),
    sa.Column("views", sa.Integer, nullable=True),
    sa.Column("forwards", sa.Integer, nullable=True),
    sa.Column("views_at", UTCDateTime, nullable=True),
    sa.Column("status", sa.Text, nullable=False, index=True),
    sa.Column("ignore_reason", sa.Text, nullable=True),
    sa.Column("duplicate_of", sa.Integer, sa.ForeignKey("posts.id"), nullable=True, index=True),
    sa.Column("dup_kind", sa.Text, nullable=True),
    sa.Column("dup_score", sa.Float, nullable=True),
    sa.Column("topic_id", sa.Integer, nullable=True),
    sa.Column("confidence", sa.Float, nullable=True),
    sa.Column("topic_scores", JSONText, nullable=True),
    sa.Column("corrected", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("corroboration", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("corroborating_chats", JSONText, nullable=False),
    sa.Column("strength", sa.Float, nullable=True),
    sa.Column("would_realtime", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("hold_until", UTCDateTime, nullable=True),
    sa.Column("decided_at", UTCDateTime, nullable=True),
    sa.Column("published_at", UTCDateTime, nullable=True),
    sa.Column("summary", sa.Text, nullable=True),
    sa.UniqueConstraint("chat_id", "message_id", name="uq_posts_chat_message"),
    sa.Index("ix_posts_posted_at", "posted_at"),
    sa.Index("ix_posts_chat_posted", "chat_id", "posted_at"),
    sa.Index("ix_posts_chat_status", "chat_id", "status"),
    sa.Index("ix_posts_status_hold", "status", "hold_until"),
    sa.Index("ix_posts_topic_status_posted", "topic_id", "status", "posted_at"),
)

publications = sa.Table(
    "publications",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("post_id", sa.Integer, sa.ForeignKey("posts.id"), nullable=False, unique=True),
    sa.Column("topic_id", sa.Integer, nullable=False),
    sa.Column("channel_id", sa.BigInteger, nullable=True),
    sa.Column("style", sa.Text, nullable=True),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("message_ids", JSONText, nullable=False),
    sa.Column("staging_ids", JSONText, nullable=False),
    sa.Column("shown_corroboration", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("edited_at", UTCDateTime, nullable=True),
    sa.Column("attempts", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("next_attempt_at", UTCDateTime, nullable=True),
    sa.Column("last_error", sa.Text, nullable=True),
    sa.Column("created_at", UTCDateTime, nullable=False),
    sa.Column("sent_at", UTCDateTime, nullable=True),
    sa.Column("moved_from", JSONText, nullable=True),
    sa.Index("ix_publications_state_next", "state", "next_attempt_at"),
)

digests = sa.Table(
    "digests",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("topic_id", sa.Integer, nullable=False),
    sa.Column("channel_id", sa.BigInteger, nullable=False),
    sa.Column("day", sa.Date, nullable=False),
    sa.Column("seq", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("manual", sa.Boolean, nullable=False, **_bool(False)),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("body", JSONText, nullable=False),
    sa.Column("message_ids", JSONText, nullable=False),
    sa.Column("item_count", sa.Integer, nullable=False),
    sa.Column("attempts", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("next_attempt_at", UTCDateTime, nullable=True),
    sa.Column("last_error", sa.Text, nullable=True),
    sa.Column("created_at", UTCDateTime, nullable=False),
    sa.Column("sent_at", UTCDateTime, nullable=True),
    sa.UniqueConstraint("topic_id", "day", "seq", name="uq_digests_topic_day_seq"),
)

digest_items = sa.Table(
    "digest_items",
    metadata,
    sa.Column("digest_id", sa.Integer, primary_key=True, autoincrement=False),
    sa.Column("position", sa.Integer, primary_key=True, autoincrement=False),
    sa.Column("post_id", sa.Integer, sa.ForeignKey("posts.id"), nullable=False, unique=True),
    sa.Column("line", sa.Text, nullable=False),
)

examples = sa.Table(
    "examples",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("topic_id", sa.Integer, nullable=True),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("post_id", sa.Integer, nullable=True),
    sa.Column("wrong_topic_id", sa.Integer, nullable=True),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("embedding", sa.LargeBinary, nullable=False),
    sa.Column("weight", sa.Float, nullable=False, default=1.0, server_default="1.0"),
    sa.Column("created_at", UTCDateTime, nullable=False),
    sa.UniqueConstraint("post_id", "kind", name="uq_examples_post_kind"),
    sa.Index("ix_examples_topic", "topic_id"),
)

proposals = sa.Table(
    "proposals",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("chat_id", sa.BigInteger, nullable=True),
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("payload", JSONText, nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("bot_message_id", sa.BigInteger, nullable=True),
    sa.Column("review_day", sa.Date, nullable=False),
    sa.Column("created_at", UTCDateTime, nullable=False),
    sa.Column("decided_at", UTCDateTime, nullable=True),
    sa.Column("executed_at", UTCDateTime, nullable=True),
    sa.Column("result", sa.Text, nullable=True),
    sa.Index("ix_proposals_state", "state"),
    sa.Index("ix_proposals_chat_state", "chat_id", "state"),
)

llm_usage = sa.Table(
    "llm_usage",
    metadata,
    sa.Column("month", sa.Text, primary_key=True),
    sa.Column("requests", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("input_tokens", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("output_tokens", sa.Integer, nullable=False, default=0, server_default="0"),
    sa.Column("cost_usd", sa.Float, nullable=False, default=0.0, server_default="0"),
    sa.Column("cap_notified", sa.Boolean, nullable=False, **_bool(False)),
)
