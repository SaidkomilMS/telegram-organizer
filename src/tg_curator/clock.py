"""Time access and the one local-time calculation the curator makes.

Every component reads the time through an injected ``Clock`` so tests can move it; the only
place local time matters is deciding *when* a digest or review is due (DESIGN §9.5), and that
is computed here once so the digest and the review agree.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    """Source of the current time; always an aware UTC datetime."""

    def now(self) -> datetime: ...


class SystemClock:
    """The wall clock, in UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def scheduled_moment(day: date, hour: int, minute: int, tz: str) -> datetime:
    """The UTC instant of ``hour:minute`` on local ``day`` in timezone ``tz``.

    ``fold=0`` picks the first occurrence of an ambiguous local time; a nonexistent one (the
    DST gap) is normalised by zoneinfo, so every scheduled moment is a real instant and every
    comparison can be made in UTC.
    """
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(tz), fold=0)
    return local.astimezone(UTC)


def local_date(now: datetime, tz: str) -> date:
    """The calendar date of ``now`` in timezone ``tz`` (``now`` must be aware)."""
    return now.astimezone(ZoneInfo(tz)).date()
