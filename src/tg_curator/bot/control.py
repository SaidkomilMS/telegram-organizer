"""``/go``, ``/pause``, ``/resume``, ``/status`` and ``/reload``: the remote control (DESIGN §11.2).

These commands change no pipeline logic; they flip the two switches the pipeline reads
(``publishing.live`` in the settings file, ``kv service.paused``) and report what the loops
wrote down (``rt.loops``, ``rt.health``, ``kv intake.last_message_at``, the digest
schedule). Their care is in the order of the writes:

- ``/go`` refuses with the one sentence that names the missing step, makes the staging
  channel exist before anything can be posted, and applies the stale-queued rule (§14.6)
  *before* the live switch so nothing that waited too long goes out late;
- ``/resume`` likewise runs the publisher's restart hook before lifting the pause.

That hook (``publisher.reconcile()``) is crash recovery: it takes a row in ``sending`` for a
send a crash interrupted. It therefore runs only while the publisher is stopped (not live, or
paused) and after any tick still in flight has finished — a ``/go`` while already live
changes nothing and does not run it — so a send in progress is never re-queued and posted
twice (§16).

``/help`` is the core's: it lists every registered command with its help line.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from tg_curator.bot.bind import logged_in_line
from tg_curator.domain import KV, TopicSyncResult
from tg_curator.errors import CuratorError
from tg_curator.llm.factory import make_llm
from tg_curator.runtime import EVENT_SETTINGS_CHANGED
from tg_curator.textutil import html_escape
from tg_curator.topics.learning import retrain_classifier

if TYPE_CHECKING:
    from tg_curator.bot.core import BotApp, Ctx
    from tg_curator.runtime import LoopState, Runtime

log = logging.getLogger(__name__)


def register(app: BotApp) -> None:
    app.command("go", go_command, help_key="control_help_go")
    app.command("pause", pause_command, help_key="control_help_pause")
    app.command("resume", resume_command, help_key="control_help_resume")
    app.command("status", status_command, help_key="control_help_status")
    app.command("reload", reload_command, help_key="control_help_reload")


# --- /go ---------------------------------------------------------------------------------------


async def go_command(ctx: Ctx, args: str) -> None:
    rt = ctx.rt
    if await logged_in_line(rt) is None:
        await ctx.reply("control_go_no_account")
        return
    if not await has_postable_topic(rt):
        await ctx.reply("control_go_no_channel")
        return
    lines: list[str] = []
    if rt.settings.publishing.style == "repost":
        lines.append(await _staging_line(rt))
    already_live = rt.settings.publishing.live
    if not already_live:
        # §14.6: stale queued posts go to the digest, not out late. Publishing is still off
        # here, so no tick can start a send while the hook runs.
        await _restart_hook(rt)
    if not already_live:
        await rt.settings_file.set_value("publishing.live", True)
    if not already_live or await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT) is None:
        await rt.store.kv_set(KV.SERVICE_WENT_LIVE_AT, rt.clock.now().isoformat())
    log.info("publishing is live%s", " (it already was)" if already_live else "")
    head = rt.t("control_go_already_live" if already_live else "control_go_live")
    schedule = rt.t(
        "control_go_schedule",
        when=day_word(rt, rt.digest.next_run()) if rt.digest else rt.t("control_every_day"),
        time=f"{rt.settings.digest.hour:02d}:{rt.settings.digest.minute:02d}",
        tz=html_escape(rt.settings.general.timezone),
        weekday=rt.t(f"control_weekday_{rt.settings.review.weekday}"),
        review_hour=f"{rt.settings.review.hour:02d}:00",
    )
    tail = [rt.t("control_go_pause_hint")]
    if await rt.store.kv_get(KV.SERVICE_PAUSED):
        tail = [rt.t("paused")]
    await ctx.reply("\n\n".join([head, schedule, *lines, *tail]))


async def has_postable_topic(rt: Runtime) -> bool:
    """At least one active topic has a channel the bot can post into (the /go precondition)."""
    if rt.bot is None:
        return False
    for topic in await rt.store.list_topics(active=True):
        if topic.channel_id is None:
            continue
        try:
            if await rt.bot.can_post(topic.channel_id):
                return True
        except CuratorError as exc:
            log.info("topic %s: the bot cannot check its channel: %s", topic.key, exc)
    return False


async def _staging_line(rt: Runtime) -> str:
    """Make the staging channel exist (§14.2); a failure does not block going live — media
    goes out as text + link until the service creates the channel at a later start."""
    if rt.topics is None:
        return rt.t("control_go_staging_failed", error=rt.t("unknown"))
    try:
        await rt.topics.ensure_staging_channel()
    except CuratorError as exc:
        log.warning("could not set up the staging channel: %s", exc)
        return rt.t("control_go_staging_failed", error=html_escape(str(exc)))
    return rt.t("control_go_staging")


def day_word(rt: Runtime, moment: datetime) -> str:
    """Today, tomorrow or the date of ``moment`` (the next digest), in local time."""
    tz = ZoneInfo(rt.settings.general.timezone)
    today = rt.clock.now().astimezone(tz).date()
    day = moment.astimezone(tz).date()
    if day == today:
        return rt.t("control_today")
    if day == today + timedelta(days=1):
        return rt.t("control_tomorrow")
    return _date_text(day)


# --- /pause and /resume --------------------------------------------------------------------------


async def pause_command(ctx: Ctx, args: str) -> None:
    rt = ctx.rt
    if await rt.store.kv_get(KV.SERVICE_PAUSED):
        await ctx.reply("control_pause_already")
        return
    await rt.store.kv_set(KV.SERVICE_PAUSED, rt.clock.now().isoformat())
    log.info("publishing paused by the owner")
    key = "control_paused" if rt.settings.publishing.live else "control_paused_not_live"
    await ctx.reply(key)


async def resume_command(ctx: Ctx, args: str) -> None:
    rt = ctx.rt
    if not await rt.store.kv_get(KV.SERVICE_PAUSED):
        await ctx.reply("control_resume_not_paused")
        return
    await _restart_hook(rt)  # §14.6, before the pause is lifted
    await rt.store.kv_delete(KV.SERVICE_PAUSED)
    log.info("publishing resumed by the owner")
    if not rt.settings.publishing.live:
        await ctx.reply("control_resumed_not_live")
        return
    await ctx.reply("control_resumed", minutes=rt.settings.sorting.hold_minutes)


async def _restart_hook(rt: Runtime) -> None:
    """``publisher.reconcile()`` once no tick is in flight.

    Callers run it only while the publisher's own ``_active()`` check is false (not live, or
    paused), so no new tick starts; a tick that started before ``/pause`` may still be
    sending, and its row in ``sending`` would look like a crash leftover. Waiting for the
    tick's lock to come free settles that: the tick takes the lock right after its check.
    """
    publisher = rt.publisher
    if publisher is None:
        return
    tick_lock = getattr(publisher, "_tick_lock", None)
    if isinstance(tick_lock, asyncio.Lock):
        async with tick_lock:
            pass
    await publisher.reconcile()


# --- /status -----------------------------------------------------------------------------------


async def status_command(ctx: Ctx, args: str) -> None:
    rt = ctx.rt
    overall, loops = _loops_status(rt)
    lines = [
        rt.t("control_status_title"),
        overall,
        await _publishing_status(rt),
        await _account_status(rt, await logged_in_line(rt)),
        await _intake_status(rt),
        await _digest_status(rt),
        await _folders_status(rt),
        *loops,
    ]
    await ctx.reply("\n".join(lines))


async def _publishing_status(rt: Runtime) -> str:
    paused = await rt.store.kv_get(KV.SERVICE_PAUSED)
    if paused:
        return rt.t("control_status_paused", since=when_text(rt, parse_kv_time(paused)))
    if rt.settings.publishing.live:
        went_live = parse_kv_time(await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT))
        return rt.t("control_status_live", since=when_text(rt, went_live))
    return rt.t("control_status_not_live")


async def _account_status(rt: Runtime, line: str | None) -> str:
    if line is not None:
        return rt.t("control_status_account", line=line)
    if await rt.store.kv_get(KV.ACCOUNT_ID) is not None:
        return rt.t("control_status_account_lost")
    return rt.t("control_status_account_none")


async def _intake_status(rt: Runtime) -> str:
    last = parse_kv_time(await rt.store.kv_get(KV.INTAKE_LAST_MESSAGE_AT))
    if last is None:
        return rt.t("control_status_intake_none")
    return rt.t("control_status_intake", when=when_text(rt, last))


async def _digest_status(rt: Runtime) -> str:
    """The next digest that will actually go out: ``DigestService.tick`` posts nothing while
    publishing is paused or not live yet, so the clock's next digest hour is shown only when
    neither switch holds it back."""
    if rt.digest is None:
        return rt.t("control_status_digest_off")
    if await rt.store.kv_get(KV.SERVICE_PAUSED):
        return rt.t("control_status_digest_paused")
    if not rt.settings.publishing.live:
        return rt.t("control_status_digest_not_live")
    return rt.t("control_status_digest", when=when_text(rt, rt.digest.next_run()))


async def _folders_status(rt: Runtime) -> str:
    if await rt.store.kv_get(KV.FOLDERS_DISABLED) is not None:
        return rt.t("control_status_folders_limit")
    folders = rt.settings.folders
    if not (folders.curated or folders.low_signal):
        return rt.t("control_status_folders_off")
    return rt.t("control_status_folders_on")


STALL_FACTOR = 3
"""A loop whose last successful tick is older than this many intervals is reported stalled."""
STALL_FLOOR = timedelta(minutes=2)
"""...but never sooner than this (the fast loops tick every few seconds)."""


def _loops_status(rt: Runtime) -> tuple[str, list[str]]:
    """The overall line ("is everything running", shown right under the title so the owner
    sees it at a glance) and the verdict of every supervised loop, problems first, then the
    last run of the on-demand jobs (debounced work, live intake events).

    The verdicts come from ``rt.loops``, the supervisor's record (RD-5): ``rt.health`` alone
    could not flag a loop that has failed on every tick since start-up, nor tell a loop held
    back by setup mode from a broken one.
    """
    now = rt.clock.now()
    verdicts = {name: _loop_verdict(rt, name, state, now) for name, state in rt.loops.items()}
    problems = sorted(name for name, (problem, _) in verdicts.items() if problem)
    if not verdicts:
        overall = rt.t("control_status_overall_none")
    elif len(problems) == 1:
        overall = rt.t("control_status_overall_problem", names=html_escape(problems[0]))
    elif problems:
        overall = rt.t(
            "control_status_overall_problems",
            count=len(problems),
            names=html_escape(", ".join(problems)),
        )
    elif any(state.paused for state in rt.loops.values()):
        overall = rt.t("control_status_overall_setup")
    else:
        overall = rt.t("control_status_overall_ok")
    lines = [rt.t("control_status_loops")] if verdicts else []
    for name in sorted(verdicts, key=lambda n: (not verdicts[n][0], n)):
        lines.append(rt.t("control_status_loop", name=html_escape(name), verdict=verdicts[name][1]))
    jobs = sorted((name, at) for name, at in rt.health.items() if name not in rt.loops)
    if jobs:
        listed = ", ".join(
            rt.t("control_status_job", name=html_escape(name), ago=_ago(rt, now - at))
            for name, at in jobs
        )
        lines.append(rt.t("control_status_jobs", jobs=listed))
    return overall, lines


def _loop_verdict(rt: Runtime, name: str, state: LoopState, now: datetime) -> tuple[bool, str]:
    """``(is a problem, the verdict text)`` for one loop.

    A serving loop (``interval is None``, the control socket) never completes a tick: its
    ``rt.health`` entry, written when it listens, newer than its last error means it
    recovered. A ticking loop is *stalled* when its last success — or, before the first one,
    the moment its first tick was due — is older than ``STALL_FACTOR`` intervals (at least
    ``STALL_FLOOR``): it hangs, or fails in a way the supervisor never sees.
    """
    if state.paused:
        return False, rt.t("control_status_loop_paused")
    if not state.running:
        return True, rt.t("control_status_loop_not_started")
    if state.failing_since is not None and not _listening_again(rt, name, state):
        return True, rt.t(
            "control_status_loop_failing",
            since=clock_text(rt, state.failing_since),
            error=html_escape(state.last_error or rt.t("unknown")),
        )
    started = state.started_at or now
    if state.interval is None:
        return False, rt.t("control_status_loop_serving", since=clock_text(rt, started))
    limit = max(timedelta(seconds=STALL_FACTOR * state.interval), STALL_FLOOR)
    if state.last_ok_at is None:
        due = started + timedelta(seconds=state.first_delay)
        if now - due > limit:
            return True, rt.t("control_status_loop_never_ticked", ago=_ago(rt, now - started))
        return False, rt.t("control_status_loop_waiting")
    ago = _ago(rt, now - state.last_ok_at)
    if now - state.last_ok_at > limit:
        return True, rt.t("control_status_loop_stalled", ago=ago)
    return False, rt.t("control_status_loop_ok", ago=ago)


def _listening_again(rt: Runtime, name: str, state: LoopState) -> bool:
    if state.interval is not None or state.last_error_at is None:
        return False
    listening = rt.health.get(name)
    return listening is not None and listening > state.last_error_at


# --- /reload -----------------------------------------------------------------------------------


async def reload_command(ctx: Ctx, args: str) -> None:
    """§4: re-read the file (the running ``[telegram]``/``[storage]`` stay), tell everyone,
    re-sync topics and sources, reload the classifier, and retry the folders."""
    rt = ctx.rt
    running = rt.settings
    # ``load()`` returns the file's values; ``rt.settings`` keeps the running [telegram] and
    # [storage] — the clients and the database were built from them — here and in every
    # later settings write until a restart (SettingsFile pins them).
    fresh = await rt.settings_file.load()
    restart_needed = fresh.telegram != running.telegram or fresh.storage != running.storage
    if fresh.llm != running.llm:
        await _rebuild_llm(rt)
    await rt.events.emit(EVENT_SETTINGS_CHANGED)
    result = await rt.topics.sync_from_settings() if rt.topics is not None else None
    # sync_from_settings reloads the classifier too, but returns early when it is re-entered
    # from a running sync; the explicit reload keeps /reload's promise either way.
    await retrain_classifier(rt)
    await rt.store.kv_delete(KV.FOLDERS_DISABLED)
    log.info("settings reloaded by the owner")
    lines = [rt.t("control_reloaded")]
    if result is not None:
        lines.append(sync_result_text(rt, result))
    if restart_needed:
        lines.append(rt.t("control_reload_restart_needed"))
    await ctx.reply("\n".join(lines))


async def _rebuild_llm(rt: Runtime) -> None:
    """A hand-edited ``[llm]`` applies now: the model object holds its provider, key and cap
    from when it was built, so reading ``rt.settings`` at use time is not enough (§4)."""
    old, rt.llm = (
        rt.llm,
        make_llm(rt.settings, rt.store, notifier=rt.notifier, clock=rt.clock),
    )
    close = getattr(old, "aclose", None)
    if close is not None:
        await close()
    log.info("language model rebuilt from the re-read [llm] settings (%s)", rt.settings.llm.mode)


def sync_result_text(rt: Runtime, result: TopicSyncResult) -> str:
    """The line "4 topics, all channels resolved", or the unresolved ones (spec "Shipping
    it": each step confirms itself)."""
    if result.unresolved:
        problems = "\n".join(f"• {html_escape(line)}" for line in result.unresolved)
        return rt.t(
            "control_topics_unresolved",
            total=result.total,
            resolved=result.resolved,
            problems=problems,
        )
    text = rt.t("control_topics_resolved", total=result.total)
    if result.resolved < result.total:
        text += " " + rt.t("control_topics_tracked", count=result.total - result.resolved)
    return text


# --- time formatting ---------------------------------------------------------------------------


def parse_kv_time(value: object) -> datetime | None:
    """A kv timestamp as the writers store it (ISO-8601, naive = UTC); else ``None``."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def when_text(rt: Runtime, at: datetime | None) -> str:
    """Local time of ``at`` (see ``clock_text``) plus how long ago or how far ahead it is."""
    if at is None:
        return rt.t("unknown")
    now = rt.clock.now()
    stamp = clock_text(rt, at)
    if at <= now:
        return rt.t("control_when_past", stamp=stamp, ago=_ago(rt, now - at))
    return rt.t("control_when_future", stamp=stamp, ahead=_ago(rt, at - now))


def clock_text(rt: Runtime, at: datetime) -> str:
    """Local time of ``at``: "14:05", or "Oct 4, 14:05" when it is not today."""
    tz = ZoneInfo(rt.settings.general.timezone)
    local = at.astimezone(tz)
    stamp = local.strftime("%H:%M")
    if local.date() != rt.clock.now().astimezone(tz).date():
        stamp = f"{_date_text(local.date())}, {stamp}"
    return stamp


def _ago(rt: Runtime, delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return rt.t("control_seconds", n=seconds)
    if seconds < 3600:
        return rt.t("control_minutes", n=seconds // 60)
    if seconds < 2 * 86400:
        return rt.t("control_hours", n=seconds // 3600, m=(seconds % 3600) // 60)
    return rt.t("control_days", n=seconds // 86400)


def _date_text(day: date) -> str:
    return f"{day.strftime('%b')} {day.day}"
