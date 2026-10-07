"""Per-chat statistics for ``/stats`` and the weekly review (DESIGN §12).

Volume is read from ``chat_daily`` rather than counted from ``posts`` because intake counts
every new message there — albums once, group chatter included, the owner's own and service
messages never — while ``posts`` holds only what became a candidate. That difference is the
point: a loud group with no substance shows its real volume next to its tiny signal, which is
what makes the review's proposals fair. The other counters are counts over ``posts`` in the
same window, and every ratio is zero-safe so a silent chat never divides by zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import sqlalchemy as sa

from tg_curator.clock import local_date, scheduled_moment
from tg_curator.db import schema
from tg_curator.domain import ChatStats, PostStatus

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

PUBLISHED_STATUSES = (PostStatus.published.value, PostStatus.digested.value)
"""A post reached its channel: posted in real time or listed in a digest (a moved post once)."""


@dataclass(frozen=True)
class TopicStats:
    """One active topic over the statistics window: how many original posts it caught and
    how many of those were strong enough to go out immediately. A topic without a channel is
    counted the same way, so a new topic can be tried out before it gets a home (spec)."""

    topic_id: int
    key: str
    name: str
    has_channel: bool
    posts: int
    immediate: int


class StatsService:
    """``chat_stats()``: one ``ChatStats`` per source chat over the review window;
    ``topic_stats()``: one ``TopicStats`` per active topic over the same window."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    def _since(self, days: int | None) -> tuple[int, date, datetime]:
        settings = self._rt.settings
        window = days or settings.review.window_days
        tz = settings.general.timezone
        first_day = local_date(self._rt.clock.now(), tz) - timedelta(days=window - 1)
        return window, first_day, scheduled_moment(first_day, 0, 0, tz)

    async def topic_stats(self, days: int | None = None) -> list[TopicStats]:
        """Per active topic: original posts sorted into it (not rejected), and how many of
        them would have gone out immediately, over the window ``chat_stats`` uses."""
        _, _, since = self._since(days)
        c = schema.posts.c
        stmt = (
            sa.select(
                c.topic_id,
                sa.func.count(),
                sa.func.coalesce(sa.func.sum(sa.case((c.would_realtime.is_(True), 1), else_=0)), 0),
            )
            .where(c.posted_at >= since)
            .where(_sorted_condition())
            .group_by(c.topic_id)
        )
        counts = {row[0]: (int(row[1]), int(row[2])) for row in await self._rows(stmt)}
        out = []
        for topic in await self._rt.store.list_topics(active=True):
            posts, immediate = counts.get(topic.id, (0, 0))
            out.append(
                TopicStats(
                    topic_id=topic.id,
                    key=topic.key,
                    name=topic.name,
                    has_channel=topic.channel_id is not None,
                    posts=posts,
                    immediate=immediate,
                )
            )
        return out

    async def chat_stats(
        self, days: int | None = None, *, include_left: bool = False
    ) -> list[ChatStats]:
        rt = self._rt
        settings = rt.settings
        window = days or settings.review.window_days
        tz = settings.general.timezone
        now = rt.clock.now()
        # The window is `window` local calendar days ending today; posts are cut at the local
        # midnight that opens it, so chat_daily (dated locally) and posts agree on the span.
        first_day = local_date(now, tz) - timedelta(days=window - 1)
        since = scheduled_moment(first_day, 0, 0, tz)

        chats = [c for c in await rt.store.list_chats(role="source") if include_left or c.active]
        volume = await self._volume(first_day)
        sorted_ = await self._count(_sorted_condition(), since)
        published = await self._count(schema.posts.c.status.in_(PUBLISHED_STATUSES), since)
        duplicates = await self._duplicates(since)

        out: list[ChatStats] = []
        for chat in chats:
            vol = volume.get(chat.id, 0)
            by_root = duplicates.get(chat.id, {})
            dups = sum(by_root.values())
            top = max(by_root, key=lambda root: (by_root[root], -root)) if by_root else None
            observed = min(window, max(0, (now - chat.first_seen_at).days))
            out.append(
                ChatStats(
                    chat_id=chat.id,
                    title=chat.title,
                    volume=vol,
                    sorted=sorted_.get(chat.id, 0),
                    signal=sorted_.get(chat.id, 0) / vol if vol else 0.0,
                    duplicates=dups,
                    duplicate_share=dups / vol if vol else 0.0,
                    published=published.get(chat.id, 0),
                    observed_days=observed,
                    top_repeated_chat_id=top,
                )
            )
        out.sort(key=lambda s: (-s.volume, s.title.casefold(), s.chat_id))
        log.debug("stats: %d chats over %d days", len(out), window)
        return out

    # --- the aggregations (§12; not in the Store on purpose, wave-0 report item 10) ---

    async def _volume(self, first_day: date) -> dict[int, int]:
        c = schema.chat_daily.c
        stmt = (
            sa.select(c.chat_id, sa.func.coalesce(sa.func.sum(c.messages), 0))
            .where(c.day >= first_day)
            .group_by(c.chat_id)
        )
        return {row[0]: int(row[1]) for row in await self._rows(stmt)}

    async def _count(self, condition: sa.ColumnElement[bool], since: datetime) -> dict[int, int]:
        c = schema.posts.c
        stmt = (
            sa.select(c.chat_id, sa.func.count())
            .where(c.posted_at >= since)
            .where(condition)
            .group_by(c.chat_id)
        )
        return {row[0]: int(row[1]) for row in await self._rows(stmt)}

    async def _duplicates(self, since: datetime) -> dict[int, dict[int, int]]:
        """``{chat_id: {root_chat_id: count}}`` — repeats of *another* chat's posts only."""
        p = schema.posts.alias("p")
        r = schema.posts.alias("r")
        stmt = (
            sa.select(p.c.chat_id, r.c.chat_id, sa.func.count())
            .select_from(p.join(r, r.c.id == p.c.duplicate_of))
            .where(p.c.status == PostStatus.duplicate.value)
            .where(p.c.posted_at >= since)
            .where(r.c.chat_id != p.c.chat_id)
            .group_by(p.c.chat_id, r.c.chat_id)
        )
        out: dict[int, dict[int, int]] = {}
        for chat_id, root_chat, n in await self._rows(stmt):
            out.setdefault(chat_id, {})[root_chat] = int(n)
        return out

    async def _rows(self, stmt: sa.Select[tuple]) -> list[sa.Row]:
        rows = await self._rt.store.execute(stmt)
        return rows if isinstance(rows, list) else []


def _sorted_condition() -> sa.ColumnElement[bool]:
    """An original post of the chat that was given a topic and was not rejected (§12)."""
    c = schema.posts.c
    return sa.and_(
        c.duplicate_of.is_(None),
        c.topic_id.is_not(None),
        c.status != PostStatus.rejected.value,
    )
