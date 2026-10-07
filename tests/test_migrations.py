"""Migrations: from empty, idempotent, transactional, and refusing a newer database."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from tg_curator.db import migrations, schema
from tg_curator.db.store import Store, sqlite_url
from tg_curator.errors import CuratorError

PG_URL = os.environ.get("TG_CURATOR_TEST_PG_URL")
BACKENDS = [
    "sqlite",
    pytest.param(
        "postgres", marks=pytest.mark.skipif(not PG_URL, reason="TG_CURATOR_TEST_PG_URL not set")
    ),
]


def _url(backend: str, tmp_path: Path) -> str:
    return sqlite_url(tmp_path / "curator.db") if backend == "sqlite" else str(PG_URL)


async def _drop_all(store: Store) -> None:
    async with store.engine.begin() as conn:
        await conn.run_sync(schema.metadata.drop_all)


@pytest.fixture(params=BACKENDS)
async def url(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[str]:
    database_url = _url(request.param, tmp_path)
    yield database_url
    if request.param == "postgres":
        store = Store(database_url)
        store._engine = store._make_engine()
        await _drop_all(store)
        await store.close()


async def _table_names(conn: AsyncConnection) -> set[str]:
    return set(await conn.run_sync(lambda sync: sa.inspect(sync).get_table_names()))


async def _index_names(conn: AsyncConnection, table: str) -> set[str]:
    indexes = await conn.run_sync(lambda sync: sa.inspect(sync).get_indexes(table))
    return {i["name"] for i in indexes if i["name"]}


async def test_from_empty_creates_everything(url: str) -> None:
    store = Store(url)
    applied = await store.start()
    assert applied == [1]
    async with store.connect() as conn:
        assert await migrations.current_version(conn) == migrations.LATEST_VERSION
        tables = await _table_names(conn)
        assert set(schema.metadata.tables) <= tables
        assert {
            "ix_posts_posted_at",
            "ix_posts_chat_posted",
            "ix_posts_chat_status",
            "ix_posts_status_hold",
            "ix_posts_topic_status_posted",
            "ix_posts_text_hash",
            "ix_posts_url_key",
            "ix_posts_status",
            "ix_posts_duplicate_of",
        } <= await _index_names(conn, "posts")
        assert "ix_publications_state_next" in await _index_names(conn, "publications")
        assert {"ix_proposals_state", "ix_proposals_chat_state"} <= await _index_names(
            conn, "proposals"
        )
        assert {"ix_group_messages_chat_closed", "ix_group_messages_date"} <= await _index_names(
            conn, "group_messages"
        )
        assert "ix_examples_topic" in await _index_names(conn, "examples")
    await store.close()


async def test_start_is_idempotent(url: str) -> None:
    store = Store(url)
    assert await store.start() == [1]
    assert await store.start() == []
    await store.close()
    again = Store(url)
    assert await again.start() == []
    async with again.connect() as conn:
        version = await conn.scalar(
            sa.select(schema.meta.c.value).where(schema.meta.c.key == migrations.SCHEMA_VERSION_KEY)
        )
    assert version == str(migrations.LATEST_VERSION)
    await again.close()


async def test_newer_database_is_refused(url: str) -> None:
    store = Store(url)
    await store.start()
    async with store.begin() as conn:
        await conn.execute(
            sa.update(schema.meta)
            .where(schema.meta.c.key == migrations.SCHEMA_VERSION_KEY)
            .values(value="999")
        )
    await store.close()
    with pytest.raises(CuratorError, match="newer"):
        await Store(url).start()


async def test_failed_migration_rolls_back(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A migration that fails half-way leaves the version (and its own DDL) untouched."""
    store = Store(url)
    await store.start()
    await store.close()

    async def _boom(conn: AsyncConnection) -> None:
        await conn.execute(sa.text("CREATE TABLE half_way (id INTEGER PRIMARY KEY)"))
        raise RuntimeError("boom")

    monkeypatch.setattr(migrations, "MIGRATIONS", [*migrations.MIGRATIONS, (2, "boom", _boom)])
    monkeypatch.setattr(migrations, "LATEST_VERSION", 2)
    broken = Store(url)
    with pytest.raises(RuntimeError, match="boom"):
        await broken.start()
    await broken.close()

    monkeypatch.undo()
    store = Store(url)
    assert await store.start() == []
    async with store.connect() as conn:
        assert await migrations.current_version(conn) == 1
        assert "half_way" not in await _table_names(conn)
    await store.close()


async def test_migrations_are_ordered_and_contiguous() -> None:
    versions = [v for v, _, _ in migrations.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert migrations.LATEST_VERSION == versions[-1]
