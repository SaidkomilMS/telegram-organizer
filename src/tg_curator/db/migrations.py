"""Versioned schema migrations, applied on ``Store.start()`` (DESIGN §6).

The list is append-only: every released schema change is a new ``(version, name, fn)`` entry,
and the current version lives in ``meta.schema_version``. ``Store.start()`` runs the pending
entries inside one transaction, so an interrupted upgrade leaves the database at the previous
version instead of half-way (both SQLite and PostgreSQL have transactional DDL).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from tg_curator.db import schema
from tg_curator.errors import CuratorError

log = logging.getLogger(__name__)

SCHEMA_VERSION_KEY = "schema_version"

Migration = tuple[int, str, Callable[[AsyncConnection], Awaitable[None]]]


async def _v1_create_everything(conn: AsyncConnection) -> None:
    """Version 1: every table, constraint and index of §6 as the schema module defines them."""
    await conn.run_sync(schema.metadata.create_all)


MIGRATIONS: list[Migration] = [
    (1, "create everything", _v1_create_everything),
]

LATEST_VERSION = MIGRATIONS[-1][0]


async def current_version(conn: AsyncConnection) -> int:
    """The schema version recorded in ``meta`` (0 for an empty database)."""
    await conn.run_sync(schema.meta.create, checkfirst=True)
    value = await conn.scalar(
        sa.select(schema.meta.c.value).where(schema.meta.c.key == SCHEMA_VERSION_KEY)
    )
    return int(value) if value is not None else 0


async def apply_migrations(conn: AsyncConnection) -> list[int]:
    """Apply every migration newer than the recorded version; return the versions applied.

    A database written by a newer release is refused rather than migrated backwards: its
    tables may hold columns this code does not know, and silently running on it would be
    worse than a clear message telling the user to upgrade.
    """
    version = await current_version(conn)
    if version > LATEST_VERSION:
        raise CuratorError(
            f"the database is at schema version {version}, newer than this version of "
            f"tg-curator (schema {LATEST_VERSION}): upgrade tg-curator"
        )
    applied: list[int] = []
    for number, name, fn in MIGRATIONS:
        if number <= version:
            continue
        log.info("database: applying migration %d (%s)", number, name)
        await fn(conn)
        await _record_version(conn, number)
        applied.append(number)
    return applied


async def _record_version(conn: AsyncConnection, version: int) -> None:
    updated = await conn.execute(
        sa.update(schema.meta)
        .where(schema.meta.c.key == SCHEMA_VERSION_KEY)
        .values(value=str(version))
    )
    if updated.rowcount == 0:
        await conn.execute(
            sa.insert(schema.meta).values(key=SCHEMA_VERSION_KEY, value=str(version))
        )
