"""clock.py: aware UTC everywhere, local schedule -> UTC instant, DST edge cases."""

from datetime import UTC, date, datetime, timedelta

from tg_curator.clock import SystemClock, local_date, scheduled_moment


def test_system_clock_is_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is UTC
    assert abs(now - datetime.now(UTC)) < timedelta(seconds=5)


def test_scheduled_moment_converts_local_to_utc() -> None:
    moment = scheduled_moment(date(2026, 10, 6), 21, 0, "Asia/Tashkent")  # UTC+5, no DST
    assert moment.tzinfo is UTC
    assert moment == datetime(2026, 10, 6, 16, 0, tzinfo=UTC)


def test_scheduled_moment_utc_identity() -> None:
    assert scheduled_moment(date(2026, 1, 1), 0, 30, "UTC") == datetime(
        2026, 1, 1, 0, 30, tzinfo=UTC
    )


def test_scheduled_moment_in_dst_gap_is_a_real_instant() -> None:
    # Europe/Berlin springs forward 2026-03-29 at 02:00 -> 03:00; 02:30 does not exist.
    moment = scheduled_moment(date(2026, 3, 29), 2, 30, "Europe/Berlin")
    assert moment.tzinfo is UTC
    # zoneinfo keeps the wall time and applies the pre-transition offset (+01:00).
    assert moment == datetime(2026, 3, 29, 1, 30, tzinfo=UTC)
    # whatever the normalisation, the moment lies inside the transition hour
    assert (
        datetime(2026, 3, 29, 0, 0, tzinfo=UTC) < moment < datetime(2026, 3, 29, 3, 0, tzinfo=UTC)
    )


def test_scheduled_moment_ambiguous_time_takes_first_occurrence() -> None:
    # Europe/Berlin falls back 2026-10-25 at 03:00 -> 02:00; 02:30 happens twice. fold=0 is
    # the first one (still CEST, +02:00), so 02:30 local = 00:30 UTC.
    moment = scheduled_moment(date(2026, 10, 25), 2, 30, "Europe/Berlin")
    assert moment == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)


def test_scheduled_moments_compare_in_utc_across_zones() -> None:
    tashkent = scheduled_moment(date(2026, 6, 1), 9, 0, "Asia/Tashkent")
    berlin = scheduled_moment(date(2026, 6, 1), 9, 0, "Europe/Berlin")
    assert tashkent < berlin  # 09:00 comes earlier in the east


def test_local_date_crosses_midnight() -> None:
    now = datetime(2026, 10, 6, 22, 30, tzinfo=UTC)
    assert local_date(now, "UTC") == date(2026, 10, 6)
    assert local_date(now, "Asia/Tashkent") == date(2026, 10, 7)
    assert local_date(now, "America/Los_Angeles") == date(2026, 10, 6)
