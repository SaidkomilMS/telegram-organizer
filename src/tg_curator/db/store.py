"""The Store: engine, transactions and the shared query API (DESIGN §7).

One ``Store`` per process owns the engine. Every write helper takes an optional ``conn`` from
``begin()`` so a module can put several helpers into one transaction (the "one transaction"
rules of §9); without it the helper opens and commits its own. On SQLite, ``begin()`` also takes
one asyncio lock so write transactions never collide (WAL lets readers through meanwhile);
on PostgreSQL the lock is a no-op.

Rows become the dataclasses of ``domain.py`` in exactly one place (``_row_to``), so every
module sees the same shape. Upserts use the dialect's ``INSERT ... ON CONFLICT`` chosen at
start, which keeps every statement portable without a second code path.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.sql.expression import Executable, UpdateBase

from tg_curator.clock import Clock, SystemClock
from tg_curator.db import migrations, schema
from tg_curator.domain import (
    OPEN_PROPOSAL_STATES,
    PROPOSAL_PROPOSED,
    PUB_PENDING,
    Chat,
    ChatRole,
    Digest,
    DigestItem,
    DigestLine,
    Example,
    NewPost,
    Post,
    PostStatus,
    Proposal,
    ProposalKind,
    Publication,
    Topic,
)
from tg_curator.errors import ConfigError

if TYPE_CHECKING:
    from tg_curator.telegram.gateway import ChatInfo

log = logging.getLogger(__name__)

T = TypeVar("T")


class _Any:
    """Sentinel type: "no filter" where ``None`` itself is a meaningful filter value."""


ANY = _Any()

SQLITE_PRAGMAS = ("PRAGMA journal_mode=WAL", "PRAGMA foreign_keys=ON", "PRAGMA busy_timeout=5000")


def sqlite_url(path: Path | str) -> str:
    """The database URL for the default SQLite file in the home directory (§3)."""
    return f"sqlite+aiosqlite:///{path}"


# --- rows without a dataclass in domain.py -------------------------------------------------


@dataclass(kw_only=True)
class GroupMessage:
    """One row of the ``group_messages`` rolling buffer (§6); only intake reads these."""

    chat_id: int
    message_id: int
    sender_id: int | None = None
    reply_to_id: int | None = None
    date: datetime
    text: str
    html: str | None = None
    urls: list[str] = dataclasses.field(default_factory=list)
    media: str | None = None
    fwd_from_chat_id: int | None = None
    fwd_from_message_id: int | None = None
    unit_root_id: int | None = None
    topic_id: int | None = None
    grouped_id: int | None = None
    closed: bool = False


@dataclass(kw_only=True)
class LlmUsage:
    """One month of the budget ledger (§6 ``llm_usage``)."""

    month: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cap_notified: bool = False


def _row_to[T](cls: type[T], row: Row[Any] | Mapping[str, Any], **extra: Any) -> T:
    """The one row -> dataclass mapping: columns and fields share names by contract (§8)."""
    data = dict(row._mapping) if isinstance(row, Row) else dict(row)
    data.update(extra)
    if cls is Post:
        data["status"] = PostStatus(data["status"])
    return cls(**data)


def _plain(value: Any) -> Any:
    """Enum members are stored by value; everything else passes through."""
    return value.value if isinstance(value, Enum) else value


def _fields(table: sa.Table, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``**fields`` of a ``set_*_fields`` helper against the table's columns."""
    unknown = [k for k in fields if k not in table.c]
    if unknown:
        raise TypeError(f"{table.name} has no column {', '.join(unknown)}")
    for key in fields:
        if table.c[key].primary_key:
            raise TypeError(f"{table.name}.{key} is the primary key and cannot be changed")
    return {k: _plain(v) for k, v in fields.items()}


def _states(state: str | Sequence[str]) -> list[str]:
    return [_plain(state)] if isinstance(state, str) else [_plain(s) for s in state]


class Store:
    """Engine owner and shared query API (§7)."""

    def __init__(self, database_url: str, *, clock: Clock | None = None) -> None:
        self._url = sa.make_url(database_url)
        self._clock: Clock = clock or SystemClock()
        self._engine: AsyncEngine | None = None
        self._lock: asyncio.Lock | None = None
        self._writers: set[asyncio.Task[Any]] = set()
        self._insert = sqlite_insert if self.is_sqlite else pg_insert

    @property
    def is_sqlite(self) -> bool:
        return self._url.get_backend_name() == "sqlite"

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("store not started: call await store.start() first")
        return self._engine

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> list[int]:
        """Create the engine and bring the schema up to date; returns the migrations applied."""
        if self._engine is None:
            self._engine = self._make_engine()
            self._lock = asyncio.Lock() if self.is_sqlite else None
        async with self.begin() as conn:
            applied = await migrations.apply_migrations(conn)
        if applied:
            log.info("database: schema at version %d", applied[-1])
        return applied

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    def _make_engine(self) -> AsyncEngine:
        if not self.is_sqlite:
            try:
                return create_async_engine(self._url)
            except ModuleNotFoundError as exc:
                # The driver is the optional `postgres` extra; a traceback (and, under a restart
                # policy, a crash loop) would hide the one command that fixes it.
                raise ConfigError(
                    "database_url points to PostgreSQL but the driver is missing: "
                    "pip install 'tg-curator[postgres]'"
                ) from exc
        engine = create_async_engine(self._url)

        # The sqlite3 driver begins a transaction only before DML and commits on its own before
        # DDL, which would leave a failed migration half-applied. Taking over BEGIN (the
        # SQLAlchemy-documented pattern) makes every transaction, DDL included, really atomic.
        @sa.event.listens_for(engine.sync_engine, "connect")
        def _configure(dbapi_connection: Any, _record: Any) -> None:
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            for pragma in SQLITE_PRAGMAS:
                cursor.execute(pragma)
            cursor.close()

        @sa.event.listens_for(engine.sync_engine, "begin")
        def _begin(conn: sa.Connection) -> None:
            conn.exec_driver_sql("BEGIN")

        return engine

    # --- transactions ------------------------------------------------------------------------

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[AsyncConnection]:
        """A connection inside a write transaction, committed on exit.

        A task that is already inside ``begin()`` must pass its ``conn`` to the helpers it
        calls; opening a second transaction from the same task would wait on its own lock
        forever on SQLite, so it is refused at once on both engines.
        """
        task = asyncio.current_task()
        if task in self._writers:
            raise RuntimeError("nested store.begin() in one task: pass conn= to the helper")
        engine = self.engine
        if self._lock is None:
            async with engine.begin() as conn:
                self._writers.add(task)  # type: ignore[arg-type]
                try:
                    yield conn
                finally:
                    self._writers.discard(task)  # type: ignore[arg-type]
            return
        async with self._lock:
            self._writers.add(task)  # type: ignore[arg-type]
            try:
                async with engine.begin() as conn:
                    yield conn
            finally:
                self._writers.discard(task)  # type: ignore[arg-type]

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """A connection for reads: no write lock, nothing committed."""
        async with self.engine.connect() as conn:
            yield conn

    @asynccontextmanager
    async def _tx(
        self, conn: AsyncConnection | None, write: bool
    ) -> AsyncIterator[AsyncConnection]:
        """The caller's connection when given, else a transaction (write) or a plain read."""
        if conn is not None:
            yield conn
            return
        if write:
            async with self.begin() as own:
                yield own
        else:
            async with self.connect() as own:
                yield own

    async def execute(
        self, stmt: Executable, conn: AsyncConnection | None = None
    ) -> list[Row[Any]] | int:
        """Escape hatch for a module's own query over the schema tables.

        Returns the rows when the statement produces any (SELECT or RETURNING), else the row
        count. Without ``conn`` an INSERT/UPDATE/DELETE (or raw text) runs in its own write
        transaction; anything else runs as an unlocked read.
        """
        write = isinstance(stmt, UpdateBase) or not isinstance(stmt, sa.Select)
        async with self._tx(conn, write) as c:
            result = await c.execute(stmt)
            if result.returns_rows:
                return list(result.all())
            return result.rowcount

    # --- kv ----------------------------------------------------------------------------------

    async def kv_get(
        self, key: str, default: Any = None, *, conn: AsyncConnection | None = None
    ) -> Any:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.kv.c.value).where(schema.kv.c.key == key))
            ).one_or_none()
        return default if row is None else row.value

    async def kv_set(self, key: str, value: Any, *, conn: AsyncConnection | None = None) -> None:
        now = self._clock.now()
        stmt = self._insert(schema.kv).values(key=key, value=value, updated_at=now)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"value": value, "updated_at": now}
        )
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)

    async def kv_delete(self, key: str, *, conn: AsyncConnection | None = None) -> None:
        async with self._tx(conn, write=True) as c:
            await c.execute(sa.delete(schema.kv).where(schema.kv.c.key == key))

    # --- chats -------------------------------------------------------------------------------

    async def upsert_chat(
        self, info: ChatInfo, role: ChatRole | None = None, *, conn: AsyncConnection | None = None
    ) -> Chat:
        """Insert or refresh a chat from what Telegram reports about it.

        A chat seen again is active (a dialog that is present was not left); ``role`` is kept
        when not given, so an output channel never silently turns back into a source.
        """
        values: dict[str, Any] = {
            "id": info.id,
            "kind": info.kind,
            "title": info.title,
            "username": info.username,
            "noforwards": info.noforwards,
            "role": role or "source",
            "active": True,
            "first_seen_at": self._clock.now(),
            "is_creator": info.is_creator,
            "is_admin": info.is_admin,
            "archived": info.archived,
            "muted_until": info.muted_until,
        }
        refresh = {
            k: values[k]
            for k in (
                "kind",
                "title",
                "username",
                "noforwards",
                "is_creator",
                "is_admin",
                "archived",
                "muted_until",
                "active",
            )
        }
        refresh["left_at"] = None
        if role is not None:
            refresh["role"] = role
        stmt = self._insert(schema.chats).values(**values)
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=refresh)
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)
            row = (
                await c.execute(sa.select(schema.chats).where(schema.chats.c.id == info.id))
            ).one()
        return _row_to(Chat, row)

    async def get_chat(self, chat_id: int, *, conn: AsyncConnection | None = None) -> Chat | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.chats).where(schema.chats.c.id == chat_id))
            ).one_or_none()
        return None if row is None else _row_to(Chat, row)

    async def list_chats(
        self,
        role: ChatRole | None = None,
        active: bool | None = None,
        *,
        conn: AsyncConnection | None = None,
    ) -> list[Chat]:
        stmt = sa.select(schema.chats).order_by(schema.chats.c.id)
        if role is not None:
            stmt = stmt.where(schema.chats.c.role == role)
        if active is not None:
            stmt = stmt.where(schema.chats.c.active == active)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Chat, r) for r in rows]

    async def set_chat_fields(
        self, chat_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(schema.chats, schema.chats.c.id == chat_id, fields, conn)

    async def bump_chat_daily(
        self, chat_id: int, day: date, n: int = 1, *, conn: AsyncConnection | None = None
    ) -> None:
        stmt = self._insert(schema.chat_daily).values(chat_id=chat_id, day=day, messages=n)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chat_id", "day"],
            set_={"messages": schema.chat_daily.c.messages + stmt.excluded.messages},
        )
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)

    async def touch_chat(
        self, chat_id: int, at: datetime, *, conn: AsyncConnection | None = None
    ) -> None:
        """Move ``last_message_at`` forward to ``at``, never back (backfill reads old posts)."""
        stmt = (
            sa.update(schema.chats)
            .where(schema.chats.c.id == chat_id)
            .where(
                sa.or_(
                    schema.chats.c.last_message_at.is_(None), schema.chats.c.last_message_at < at
                )
            )
            .values(last_message_at=at)
        )
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)

    # --- topics ------------------------------------------------------------------------------

    async def list_topics(
        self, active: bool | None = True, *, conn: AsyncConnection | None = None
    ) -> list[Topic]:
        stmt = sa.select(schema.topics).order_by(schema.topics.c.id)
        if active is not None:
            stmt = stmt.where(schema.topics.c.active == active)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Topic, r) for r in rows]

    async def get_topic(
        self, topic_id: int, *, conn: AsyncConnection | None = None
    ) -> Topic | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.topics).where(schema.topics.c.id == topic_id))
            ).one_or_none()
        return None if row is None else _row_to(Topic, row)

    async def get_topic_by_key(
        self, key: str, *, conn: AsyncConnection | None = None
    ) -> Topic | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.topics).where(schema.topics.c.key == key))
            ).one_or_none()
        return None if row is None else _row_to(Topic, row)

    async def upsert_topic(self, topic: Topic, *, conn: AsyncConnection | None = None) -> Topic:
        """Insert or update by ``key``; ``id`` and ``created_at`` of an existing row are kept."""
        values = {
            "key": topic.key,
            "name": topic.name,
            "channel_id": topic.channel_id,
            "category": topic.category,
            "description": topic.description,
            "example_channel": topic.example_channel,
            "strictness": topic.strictness,
            "active": topic.active,
            "origin": topic.origin,
            "created_at": topic.created_at,
        }
        refresh = {k: v for k, v in values.items() if k not in ("key", "created_at")}
        stmt = self._insert(schema.topics).values(**values)
        stmt = stmt.on_conflict_do_update(index_elements=["key"], set_=refresh)
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)
            row = (
                await c.execute(sa.select(schema.topics).where(schema.topics.c.key == topic.key))
            ).one()
        return _row_to(Topic, row)

    async def set_topic_fields(
        self, topic_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(schema.topics, schema.topics.c.id == topic_id, fields, conn)

    # --- posts -------------------------------------------------------------------------------

    async def insert_post(
        self, new: NewPost, *, conn: AsyncConnection | None = None, **decision: Any
    ) -> Post | None:
        """Insert a post; ``None`` when ``(chat_id, message_id)`` already exists.

        ``decision`` may carry decision columns (``status`` ... ``summary``) so the sorter
        writes the row and its decision in one statement; without it the post starts
        ``unsorted`` with no decision. The duplicate check is ``ON CONFLICT DO NOTHING`` rather
        than a caught IntegrityError because a failed statement would abort the caller's
        PostgreSQL transaction.
        """
        values = {f.name: _plain(getattr(new, f.name)) for f in dataclasses.fields(NewPost)}
        values.update(
            ingested_at=self._clock.now(),
            status=PostStatus.unsorted.value,
            corroborating_chats=[],
        )
        values.update(_fields(schema.posts, decision))
        stmt = self._insert(schema.posts).values(**values)
        stmt = stmt.on_conflict_do_nothing(index_elements=["chat_id", "message_id"])
        async with self._tx(conn, write=True) as c:
            inserted = (await c.execute(stmt)).rowcount
            if inserted == 0:
                return None
            row = (
                await c.execute(
                    sa.select(schema.posts)
                    .where(schema.posts.c.chat_id == new.chat_id)
                    .where(schema.posts.c.message_id == new.message_id)
                )
            ).one()
        return _row_to(Post, row)

    async def get_post(self, post_id: int, *, conn: AsyncConnection | None = None) -> Post | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.posts).where(schema.posts.c.id == post_id))
            ).one_or_none()
        return None if row is None else _row_to(Post, row)

    async def get_post_by_message(
        self, chat_id: int, message_id: int, *, conn: AsyncConnection | None = None
    ) -> Post | None:
        stmt = (
            sa.select(schema.posts)
            .where(schema.posts.c.chat_id == chat_id)
            .where(schema.posts.c.message_id == message_id)
        )
        async with self._tx(conn, write=False) as c:
            row = (await c.execute(stmt)).one_or_none()
        return None if row is None else _row_to(Post, row)

    async def set_post_fields(
        self, post_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(schema.posts, schema.posts.c.id == post_id, fields, conn)

    async def recent_posts(
        self,
        since: datetime,
        *,
        statuses: Sequence[str | PostStatus] | None = None,
        with_embeddings: bool = False,
        conn: AsyncConnection | None = None,
    ) -> list[Post]:
        """Posts with ``posted_at >= since``, oldest first; embeddings only on request."""
        columns = [c for c in schema.posts.c if with_embeddings or c.name != "embedding"]
        stmt = sa.select(*columns).where(schema.posts.c.posted_at >= since)
        if statuses is not None:
            stmt = stmt.where(schema.posts.c.status.in_(_states(statuses)))
        stmt = stmt.order_by(schema.posts.c.posted_at, schema.posts.c.id)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        if with_embeddings:
            return [_row_to(Post, r) for r in rows]
        return [_row_to(Post, r, embedding=None) for r in rows]

    async def posts_by_status(
        self,
        status: str | PostStatus | Sequence[str | PostStatus],
        *,
        topic_id: int | None = None,
        due_before: datetime | None = None,
        limit: int | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[Post]:
        """Posts in ``status`` (one or several), oldest first; ``due_before`` = ``hold_until``."""
        stmt = sa.select(schema.posts).where(schema.posts.c.status.in_(_states(status)))
        if topic_id is not None:
            stmt = stmt.where(schema.posts.c.topic_id == topic_id)
        if due_before is not None:
            stmt = stmt.where(schema.posts.c.hold_until <= due_before)
        stmt = stmt.order_by(schema.posts.c.posted_at, schema.posts.c.id)
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Post, r) for r in rows]

    async def add_corroboration(
        self, root_id: int, chat_id: int, *, conn: AsyncConnection | None = None
    ) -> int | None:
        """Count ``chat_id`` as one more chat carrying the root's story.

        Returns the new count, or ``None`` when that chat is already counted — the root's own
        chat included, since corroboration counts *other* chats (§6). The row is locked for
        the update on PostgreSQL; on SQLite the write lock of ``begin()`` does the same.
        """
        stmt = (
            sa.select(schema.posts.c.chat_id, schema.posts.c.corroborating_chats)
            .where(schema.posts.c.id == root_id)
            .with_for_update()
        )
        async with self._tx(conn, write=True) as c:
            row = (await c.execute(stmt)).one_or_none()
            if row is None:
                return None
            counted = list(row.corroborating_chats or [])
            if chat_id == row.chat_id or chat_id in counted:
                return None
            counted.append(chat_id)
            await c.execute(
                sa.update(schema.posts)
                .where(schema.posts.c.id == root_id)
                .values(corroboration=len(counted), corroborating_chats=counted)
            )
        return len(counted)

    # --- examples ----------------------------------------------------------------------------

    async def add_example(
        self, example: Example, *, conn: AsyncConnection | None = None
    ) -> Example:
        """Insert an example; one with a ``post_id`` replaces the row of the same (post, kind)."""
        values = {
            "topic_id": example.topic_id,
            "kind": _plain(example.kind),
            "post_id": example.post_id,
            "wrong_topic_id": example.wrong_topic_id,
            "text": example.text,
            "embedding": example.embedding,
            "weight": example.weight,
            "created_at": example.created_at,
        }
        async with self._tx(conn, write=True) as c:
            if example.post_id is None:
                result = await c.execute(sa.insert(schema.examples).values(**values))
                where = schema.examples.c.id == result.inserted_primary_key[0]
            else:
                refresh = {k: v for k, v in values.items() if k not in ("kind", "post_id")}
                stmt = self._insert(schema.examples).values(**values)
                stmt = stmt.on_conflict_do_update(index_elements=["post_id", "kind"], set_=refresh)
                await c.execute(stmt)
                where = sa.and_(
                    schema.examples.c.post_id == example.post_id,
                    schema.examples.c.kind == values["kind"],
                )
            row = (await c.execute(sa.select(schema.examples).where(where))).one()
        return _row_to(Example, row)

    async def list_examples(
        self, topic_id: int | None | _Any = ANY, *, conn: AsyncConnection | None = None
    ) -> list[Example]:
        """All examples by default; ``topic_id=None`` selects the "not for me" negatives."""
        stmt = sa.select(schema.examples).order_by(schema.examples.c.id)
        if topic_id is None:
            stmt = stmt.where(schema.examples.c.topic_id.is_(None))
        elif not isinstance(topic_id, _Any):
            stmt = stmt.where(schema.examples.c.topic_id == topic_id)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Example, r) for r in rows]

    async def delete_examples(self, topic_id: int, *, conn: AsyncConnection | None = None) -> int:
        async with self._tx(conn, write=True) as c:
            result = await c.execute(
                sa.delete(schema.examples).where(schema.examples.c.topic_id == topic_id)
            )
        return result.rowcount

    # --- publications ------------------------------------------------------------------------

    async def create_publication(
        self, post_id: int, topic_id: int, *, conn: AsyncConnection | None = None
    ) -> Publication | None:
        """A ``pending`` outbox row; ``None`` when the post already has one in any state."""
        stmt = self._insert(schema.publications).values(
            post_id=post_id,
            topic_id=topic_id,
            state=PUB_PENDING,
            message_ids=[],
            staging_ids=[],
            created_at=self._clock.now(),
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["post_id"])
        async with self._tx(conn, write=True) as c:
            if (await c.execute(stmt)).rowcount == 0:
                return None
            row = (
                await c.execute(
                    sa.select(schema.publications).where(schema.publications.c.post_id == post_id)
                )
            ).one()
        return _row_to(Publication, row)

    async def get_publication(
        self, post_id: int, *, conn: AsyncConnection | None = None
    ) -> Publication | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(
                    sa.select(schema.publications).where(schema.publications.c.post_id == post_id)
                )
            ).one_or_none()
        return None if row is None else _row_to(Publication, row)

    async def set_publication_fields(
        self, publication_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(
            schema.publications, schema.publications.c.id == publication_id, fields, conn
        )

    async def publications_in_state(
        self,
        state: str | Sequence[str],
        limit: int | None = None,
        *,
        conn: AsyncConnection | None = None,
    ) -> list[Publication]:
        stmt = (
            sa.select(schema.publications)
            .where(schema.publications.c.state.in_(_states(state)))
            .order_by(schema.publications.c.created_at, schema.publications.c.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Publication, r) for r in rows]

    # --- digests -----------------------------------------------------------------------------

    async def create_digest(
        self,
        topic_id: int,
        channel_id: int,
        day: date,
        *,
        body: list[str],
        item_count: int,
        seq: int = 0,
        manual: bool = False,
        state: str = "pending",
        conn: AsyncConnection | None = None,
    ) -> Digest:
        """Insert a digest row; a ``(topic, day, seq)`` collision raises (the caller chose seq)."""
        values = {
            "topic_id": topic_id,
            "channel_id": channel_id,
            "day": day,
            "seq": seq,
            "manual": manual,
            "state": state,
            "body": list(body),
            "message_ids": [],
            "item_count": item_count,
            "created_at": self._clock.now(),
        }
        async with self._tx(conn, write=True) as c:
            result = await c.execute(sa.insert(schema.digests).values(**values))
            digest_id = result.inserted_primary_key[0]
            row = (
                await c.execute(sa.select(schema.digests).where(schema.digests.c.id == digest_id))
            ).one()
        return _row_to(Digest, row)

    async def get_digest(
        self, digest_id: int, *, conn: AsyncConnection | None = None
    ) -> Digest | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(sa.select(schema.digests).where(schema.digests.c.id == digest_id))
            ).one_or_none()
        return None if row is None else _row_to(Digest, row)

    async def get_digest_by_key(
        self, topic_id: int, day: date, seq: int = 0, *, conn: AsyncConnection | None = None
    ) -> Digest | None:
        """The row of ``(topic, day, seq)``; ``seq=0`` is the scheduled digest (§9.5)."""
        stmt = (
            sa.select(schema.digests)
            .where(schema.digests.c.topic_id == topic_id)
            .where(schema.digests.c.day == day)
            .where(schema.digests.c.seq == seq)
        )
        async with self._tx(conn, write=False) as c:
            row = (await c.execute(stmt)).one_or_none()
        return None if row is None else _row_to(Digest, row)

    async def set_digest_fields(
        self, digest_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(
            schema.digests, schema.digests.c.id == digest_id, fields, conn
        )

    async def digests_in_state(
        self,
        state: str | Sequence[str],
        limit: int | None = None,
        *,
        conn: AsyncConnection | None = None,
    ) -> list[Digest]:
        stmt = (
            sa.select(schema.digests)
            .where(schema.digests.c.state.in_(_states(state)))
            .order_by(schema.digests.c.created_at, schema.digests.c.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Digest, r) for r in rows]

    async def add_digest_items(
        self,
        digest_id: int,
        items: Sequence[DigestLine | DigestItem],
        *,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Store the items of a composed digest; ``DigestLine`` drafts are accepted directly."""
        if not items:
            return
        rows = [
            {"digest_id": digest_id, "position": i.position, "post_id": i.post_id, "line": i.line}
            for i in items
        ]
        async with self._tx(conn, write=True) as c:
            await c.execute(sa.insert(schema.digest_items), rows)

    async def list_digest_items(
        self, digest_id: int, *, conn: AsyncConnection | None = None
    ) -> list[DigestItem]:
        stmt = (
            sa.select(schema.digest_items)
            .where(schema.digest_items.c.digest_id == digest_id)
            .order_by(schema.digest_items.c.position)
        )
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(DigestItem, r) for r in rows]

    async def next_manual_seq(
        self, topic_id: int, day: date, *, conn: AsyncConnection | None = None
    ) -> int:
        """``COALESCE(MAX(seq), 0) + 1`` over the rows of that topic and day (§9.5)."""
        stmt = (
            sa.select(sa.func.coalesce(sa.func.max(schema.digests.c.seq), 0) + 1)
            .where(schema.digests.c.topic_id == topic_id)
            .where(schema.digests.c.day == day)
        )
        async with self._tx(conn, write=False) as c:
            return int(await c.scalar(stmt))

    # --- proposals ---------------------------------------------------------------------------

    async def create_proposal(
        self,
        kind: ProposalKind,
        *,
        reason: str,
        review_day: date,
        chat_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
        state: str = PROPOSAL_PROPOSED,
        bot_message_id: int | None = None,
        conn: AsyncConnection | None = None,
    ) -> Proposal:
        values = {
            "kind": _plain(kind),
            "chat_id": chat_id,
            "reason": reason,
            "payload": dict(payload or {}),
            "state": state,
            "bot_message_id": bot_message_id,
            "review_day": review_day,
            "created_at": self._clock.now(),
        }
        async with self._tx(conn, write=True) as c:
            result = await c.execute(sa.insert(schema.proposals).values(**values))
            proposal_id = result.inserted_primary_key[0]
            row = (
                await c.execute(
                    sa.select(schema.proposals).where(schema.proposals.c.id == proposal_id)
                )
            ).one()
        return _row_to(Proposal, row)

    async def get_proposal(
        self, proposal_id: int, *, conn: AsyncConnection | None = None
    ) -> Proposal | None:
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(
                    sa.select(schema.proposals).where(schema.proposals.c.id == proposal_id)
                )
            ).one_or_none()
        return None if row is None else _row_to(Proposal, row)

    async def set_proposal_fields(
        self, proposal_id: int, *, conn: AsyncConnection | None = None, **fields: Any
    ) -> int:
        return await self._set_fields(
            schema.proposals, schema.proposals.c.id == proposal_id, fields, conn
        )

    async def proposals_by_state(
        self,
        state: str | Sequence[str],
        *,
        kind: ProposalKind | None = None,
        review_day: date | None = None,
        limit: int | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[Proposal]:
        stmt = (
            sa.select(schema.proposals)
            .where(schema.proposals.c.state.in_(_states(state)))
            .order_by(schema.proposals.c.created_at, schema.proposals.c.id)
        )
        if kind is not None:
            stmt = stmt.where(schema.proposals.c.kind == _plain(kind))
        if review_day is not None:
            stmt = stmt.where(schema.proposals.c.review_day == review_day)
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(Proposal, r) for r in rows]

    async def open_proposal_for_chat(
        self, chat_id: int, *, conn: AsyncConnection | None = None
    ) -> Proposal | None:
        """The chat's proposal in ``proposed``/``confirming``/``approved``, if any (§12)."""
        stmt = (
            sa.select(schema.proposals)
            .where(schema.proposals.c.chat_id == chat_id)
            .where(schema.proposals.c.state.in_(list(OPEN_PROPOSAL_STATES)))
            .order_by(schema.proposals.c.id.desc())
            .limit(1)
        )
        async with self._tx(conn, write=False) as c:
            row = (await c.execute(stmt)).one_or_none()
        return None if row is None else _row_to(Proposal, row)

    # --- llm_usage ---------------------------------------------------------------------------

    async def get_usage(self, month: str, *, conn: AsyncConnection | None = None) -> LlmUsage:
        """The month's ledger; an all-zero record (not stored) when nothing was spent yet."""
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(
                    sa.select(schema.llm_usage).where(schema.llm_usage.c.month == month)
                )
            ).one_or_none()
        return LlmUsage(month=month) if row is None else _row_to(LlmUsage, row)

    async def add_usage(
        self,
        month: str,
        requests: int,
        in_tokens: int,
        out_tokens: int,
        cost: float,
        *,
        conn: AsyncConnection | None = None,
    ) -> LlmUsage:
        """Add to the month's counters (negative values settle an earlier reservation)."""
        stmt = self._insert(schema.llm_usage).values(
            month=month,
            requests=requests,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["month"],
            set_={
                "requests": schema.llm_usage.c.requests + stmt.excluded.requests,
                "input_tokens": schema.llm_usage.c.input_tokens + stmt.excluded.input_tokens,
                "output_tokens": schema.llm_usage.c.output_tokens + stmt.excluded.output_tokens,
                "cost_usd": schema.llm_usage.c.cost_usd + stmt.excluded.cost_usd,
            },
        )
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)
            row = (
                await c.execute(
                    sa.select(schema.llm_usage).where(schema.llm_usage.c.month == month)
                )
            ).one()
        return _row_to(LlmUsage, row)

    async def set_cap_notified(
        self, month: str, notified: bool = True, *, conn: AsyncConnection | None = None
    ) -> None:
        stmt = self._insert(schema.llm_usage).values(month=month, cap_notified=notified)
        stmt = stmt.on_conflict_do_update(index_elements=["month"], set_={"cap_notified": notified})
        async with self._tx(conn, write=True) as c:
            await c.execute(stmt)

    # --- group_messages ----------------------------------------------------------------------

    async def add_group_message(
        self,
        chat_id: int,
        message_id: int,
        *,
        date: datetime,
        text: str,
        sender_id: int | None = None,
        reply_to_id: int | None = None,
        html: str | None = None,
        urls: Sequence[str] = (),
        media: str | None = None,
        fwd_from_chat_id: int | None = None,
        fwd_from_message_id: int | None = None,
        unit_root_id: int | None = None,
        topic_id: int | None = None,
        grouped_id: int | None = None,
        closed: bool = False,
        conn: AsyncConnection | None = None,
    ) -> bool:
        """Buffer a group message; ``False`` when ``(chat_id, message_id)`` was already there."""
        stmt = self._insert(schema.group_messages).values(
            chat_id=chat_id,
            message_id=message_id,
            sender_id=sender_id,
            reply_to_id=reply_to_id,
            date=date,
            text=text,
            html=html,
            urls=list(urls),
            media=_plain(media),
            fwd_from_chat_id=fwd_from_chat_id,
            fwd_from_message_id=fwd_from_message_id,
            unit_root_id=unit_root_id,
            topic_id=topic_id,
            grouped_id=grouped_id,
            closed=closed,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["chat_id", "message_id"])
        async with self._tx(conn, write=True) as c:
            return (await c.execute(stmt)).rowcount == 1

    async def open_group_messages(
        self, chat_id: int | None = None, *, conn: AsyncConnection | None = None
    ) -> list[GroupMessage]:
        """Buffered messages not yet in a closed unit, oldest first."""
        stmt = sa.select(schema.group_messages).where(schema.group_messages.c.closed == False)  # noqa: E712
        if chat_id is not None:
            stmt = stmt.where(schema.group_messages.c.chat_id == chat_id)
        stmt = stmt.order_by(schema.group_messages.c.date, schema.group_messages.c.message_id)
        async with self._tx(conn, write=False) as c:
            rows = (await c.execute(stmt)).all()
        return [_row_to(GroupMessage, r) for r in rows]

    async def album_buffered(
        self,
        chat_id: int,
        grouped_id: int,
        *,
        besides: int,
        conn: AsyncConnection | None = None,
    ) -> bool:
        """Is another part (not message ``besides``) of this album already buffered?"""
        gm = schema.group_messages.c
        stmt = (
            sa.select(gm.message_id)
            .where(gm.chat_id == chat_id, gm.grouped_id == grouped_id, gm.message_id != besides)
            .limit(1)
        )
        async with self._tx(conn, write=False) as c:
            return (await c.execute(stmt)).first() is not None

    async def group_unit_of(
        self, chat_id: int, message_id: int, *, conn: AsyncConnection | None = None
    ) -> list[GroupMessage]:
        """Every buffered row of the unit that holds ``message_id`` (open or closed), oldest
        first; empty when the message is not buffered (or was purged)."""
        gm = schema.group_messages.c
        async with self._tx(conn, write=False) as c:
            row = (
                await c.execute(
                    sa.select(gm.unit_root_id).where(
                        gm.chat_id == chat_id, gm.message_id == message_id
                    )
                )
            ).one_or_none()
            if row is None:
                return []
            root = row.unit_root_id if row.unit_root_id is not None else message_id
            stmt = (
                sa.select(schema.group_messages)
                .where(
                    gm.chat_id == chat_id,
                    sa.or_(gm.unit_root_id == root, gm.message_id == root),
                )
                .order_by(gm.date, gm.message_id)
            )
            rows = (await c.execute(stmt)).all()
        return [_row_to(GroupMessage, r) for r in rows]

    async def mark_group_messages(
        self,
        chat_id: int,
        message_ids: Sequence[int],
        *,
        conn: AsyncConnection | None = None,
        **fields: Any,
    ) -> int:
        """Set ``unit_root_id`` and/or ``closed`` on the given messages of one chat.

        Only the two unit-building columns may change: a buffered message's content is the
        version first seen, by the spec's rule that edits are not followed.
        """
        other = set(fields) - {"unit_root_id", "closed"}
        if other:
            raise TypeError(f"mark_group_messages cannot change {', '.join(sorted(other))}")
        if not message_ids:
            return 0
        where = sa.and_(
            schema.group_messages.c.chat_id == chat_id,
            schema.group_messages.c.message_id.in_(list(message_ids)),
        )
        return await self._set_fields(schema.group_messages, where, fields, conn)

    async def purge_group_messages(
        self, before: datetime, *, conn: AsyncConnection | None = None
    ) -> int:
        """Drop buffered messages older than ``before`` (the 3-day retention of §6)."""
        async with self._tx(conn, write=True) as c:
            result = await c.execute(
                sa.delete(schema.group_messages).where(schema.group_messages.c.date < before)
            )
        return result.rowcount

    # --- internals ---------------------------------------------------------------------------

    async def _set_fields(
        self,
        table: sa.Table,
        where: Any,
        fields: Mapping[str, Any],
        conn: AsyncConnection | None,
    ) -> int:
        values = _fields(table, fields)
        if not values:
            return 0
        async with self._tx(conn, write=True) as c:
            result = await c.execute(sa.update(table).where(where).values(**values))
        return result.rowcount
