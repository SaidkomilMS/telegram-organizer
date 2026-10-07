"""``/start`` and ``/setup``: the seven-step walkthrough of spec "Setting up" (DESIGN §11.2).

The walkthrough keeps no script of its own. Every time it runs it asks the real state which
step is the first one not done — is the account bound, does a topic have a channel, was a
language model chosen, did the preview run, is publishing live — so it continues from
wherever the owner stopped, survives restarts, and running it twice changes nothing. Each
finished step confirms itself in the spec's words ("logged in as …", "4 topics, all channels
resolved"), and the current step is handed to the module that owns it: the bind flow, ``/topics
add``, ``/llm``, ``/preview``, ``/go``. The owner comes back with ``/setup`` or the [Continue
setup] button; a step that finishes inside the same call (the preview) continues at once.

One choice cannot be read from state: "no language model" is the default, so choosing it
leaves no trace. ``kv setup.step`` therefore records the furthest step the walkthrough
reached; once it is past the model step, that step counts as done whatever the mode is.
"""

from __future__ import annotations

import logging
import os
import zoneinfo
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tg_curator.bot.bind import begin_bind, logged_in_line
from tg_curator.bot.control import parse_kv_time, sync_result_text, when_text
from tg_curator.domain import KV, TopicSyncResult
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.bot.core import BotApp, Ctx
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

STEPS = ("bind", "topics", "llm", "preview", "go")
"""The steps the bot walks through, in order; install and claim are done before it can talk."""
DONE = "done"
NUMBERS = {"bind": 3, "topics": 4, "llm": 5, "preview": 6, "go": 7}
"""Each step's number in spec "Setting up" (1 = install, 2 = claim)."""
TOTAL_STEPS = 7

CONTINUE = "su:next"
CHOOSE_LLM = "su:llm"
SKIP_LLM = "su:skipllm"
GO_LIVE = "su:go"
TZ_SERVER = "su:tzsrv"
TZ_KEEP = "su:tzkeep"
TIMEZONE_KEY = "general.timezone"
LOCALTIME = Path("/etc/localtime")


def register(app: BotApp) -> None:
    app.command("start", start_command, help_key="setup_help_start")
    app.command("setup", setup_command, help_key="setup_help_setup")
    app.callback("su", setup_callback)


# --- where the owner is ------------------------------------------------------------------------


@dataclass
class Progress:
    """The finished steps (with their confirmation lines) and the first unfinished one."""

    done: list[tuple[str, str]] = field(default_factory=list)
    current: str = DONE


async def read_progress(rt: Runtime) -> Progress:
    """Walk the steps in order against the real state; stop at the first unfinished one."""
    progress = Progress()
    furthest = await _furthest(rt)
    for step in STEPS:
        line = await _confirmation(rt, step, furthest)
        if line is None:
            progress.current = step
            return progress
        progress.done.append((step, line))
    return progress


async def _confirmation(rt: Runtime, step: str, furthest: str | None) -> str | None:
    """The step's confirmation sentence when it is done, else ``None``."""
    if step == "bind":
        return await logged_in_line(rt)
    if step == "topics":
        return await _topics_line(rt)
    if step == "llm":
        return _llm_line(rt, furthest)
    if step == "preview":
        # the report sent, not any backfill: a CLI or one-chat read never ticks step 6
        shown = parse_kv_time(await rt.store.kv_get(KV.SETUP_PREVIEW_SHOWN))
        return None if shown is None else rt.t("setup_preview_done", when=when_text(rt, shown))
    if not rt.settings.publishing.live:
        return None
    went_live = parse_kv_time(await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT))
    line = rt.t("setup_live_since", when=when_text(rt, went_live))
    if await rt.store.kv_get(KV.SERVICE_PAUSED):
        line += " " + rt.t("setup_live_paused")
    return line


async def _topics_line(rt: Runtime) -> str | None:
    """Done once a topic has a channel (what /go needs); channel-less topics are tracked."""
    topics = await rt.store.list_topics(active=True)
    with_channel = sum(1 for t in topics if t.channel_id is not None)
    if with_channel == 0:
        return None
    return sync_result_text(
        rt, TopicSyncResult(total=len(topics), resolved=with_channel, unresolved=[])
    )


def _llm_line(rt: Runtime, furthest: str | None) -> str | None:
    llm = rt.settings.llm
    if llm.mode == "provider":
        return rt.t(
            "setup_llm_provider", provider=html_escape(llm.provider), model=html_escape(llm.model)
        )
    if llm.mode == "selfhosted":
        return rt.t(
            "setup_llm_selfhosted", model=html_escape(llm.model), url=html_escape(llm.base_url)
        )
    if furthest is not None and _index(furthest) > _index("llm"):
        return rt.t("setup_llm_none")
    return None


async def _furthest(rt: Runtime) -> str | None:
    stored = await rt.store.kv_get(KV.SETUP_STEP)
    return stored if isinstance(stored, str) and stored in (*STEPS, DONE) else None


async def _record(rt: Runtime, step: str) -> None:
    """Remember the furthest step reached (never moves back: a lost session later does not
    un-choose the language model)."""
    furthest = await _furthest(rt)
    if furthest is None or _index(step) > _index(furthest):
        await rt.store.kv_set(KV.SETUP_STEP, step)
        log.info("setup: reached step %s", step)


async def mark_llm_chosen(rt: Runtime) -> bool:
    """``/llm`` -> None while the walkthrough waits at the model step: that step is done.

    ``True`` when the walkthrough moved on (the caller offers [Continue setup]). A ``/llm``
    outside the walkthrough, or before it reached this step, records nothing.
    """
    if await _furthest(rt) != "llm":
        return False
    await _record(rt, STEPS[STEPS.index("llm") + 1])
    return True


def _index(step: str) -> int:
    return (*STEPS, DONE).index(step)


# --- /start and /setup -------------------------------------------------------------------------


async def start_command(ctx: Ctx, args: str) -> None:
    progress = await read_progress(ctx.rt)
    if progress.current == DONE:
        await ctx.reply("setup_welcome_done")
        return
    await ctx.reply("setup_welcome", number=NUMBERS[progress.current], total=TOTAL_STEPS)


async def setup_command(ctx: Ctx, args: str) -> None:
    await walkthrough(ctx)


async def walkthrough(ctx: Ctx) -> None:
    """Show what is done, then hand the first unfinished step to its module.

    A step that completes within the hand-off (nothing left waiting for the owner) moves the
    walkthrough on at once; otherwise the owner comes back with /setup or [Continue setup].
    """
    rt = ctx.rt
    progress = await read_progress(rt)
    await _record(rt, progress.current)
    await ctx.reply(_checklist(rt, progress))
    while progress.current != DONE:
        step = progress.current
        if not await _hand_off(ctx, step):
            return
        if await ctx.app.conversation() is not None:
            return
        progress = await read_progress(rt)
        if _index(progress.current) <= _index(step):
            return
        await _record(rt, progress.current)
        await ctx.reply("setup_line_done", number=NUMBERS[step], line=dict(progress.done)[step])
    await ctx.reply("setup_complete")


def _checklist(rt: Runtime, progress: Progress) -> str:
    lines = [
        rt.t("setup_title"),
        rt.t("setup_line_done", number=1, line=rt.t("setup_installed")),
        rt.t("setup_line_done", number=2, line=rt.t("setup_claimed")),
    ]
    lines += [rt.t("setup_line_done", number=NUMBERS[s], line=line) for s, line in progress.done]
    if progress.current != DONE:
        lines.append(
            rt.t(
                "setup_line_current",
                number=NUMBERS[progress.current],
                line=rt.t(f"setup_title_{progress.current}"),
            )
        )
    return "\n".join(lines)


async def _hand_off(ctx: Ctx, step: str) -> bool:
    """Start the step. ``True`` when it ran to its end inside this call (it may have finished
    the step), ``False`` when it now waits for the owner (a flow, a button)."""
    rt = ctx.rt
    number = NUMBERS[step]
    if step == "bind":
        await ctx.reply("setup_bind_intro", number=number, total=TOTAL_STEPS)
        await begin_bind(ctx, then=CONTINUE)
        return False
    if step == "topics":
        await ctx.reply("setup_topics_intro", number=number, total=TOTAL_STEPS)
        await _run_command(ctx, "topics", "add")
        return False
    if step == "llm":
        buttons: Buttons = [
            [Button(rt.t("setup_llm_choose"), data=CHOOSE_LLM)],
            [Button(rt.t("setup_llm_skip"), data=SKIP_LLM)],
        ]
        await ctx.reply("setup_llm_intro", number=number, total=TOTAL_STEPS, buttons=buttons)
        return False
    if step == "preview":
        await ctx.reply("setup_preview_intro", number=number, total=TOTAL_STEPS)
        return await _run_command(ctx, "preview")
    if rt.settings.general.timezone == "UTC":
        # The digest and review hours are local to general.timezone; a bot-only setup would
        # otherwise never choose it (SPEC: the first digest arrives "that evening").
        await _ask_timezone(ctx)
        return False
    await _go_intro(ctx)
    return False


async def _go_intro(ctx: Ctx) -> None:
    rt = ctx.rt
    buttons = [[Button(rt.t("setup_go_button"), data=GO_LIVE)]]
    await ctx.reply("setup_go_intro", number=NUMBERS["go"], total=TOTAL_STEPS, buttons=buttons)


async def _ask_timezone(ctx: Ctx) -> None:
    """Which zone the digest hour is in: the server's zone in one tap, another one typed in
    the /settings prompt, or UTC kept."""
    rt = ctx.rt
    rows: Buttons = []
    server = server_timezone()
    if server is not None and server != "UTC":
        rows.append([Button(rt.t("setup_timezone_server", zone=server), data=TZ_SERVER)])
    other = _settings_item_data(TIMEZONE_KEY)
    if other is not None:
        rows.append([Button(rt.t("setup_timezone_other"), data=other)])
    rows.append([Button(rt.t("setup_timezone_keep"), data=TZ_KEEP)])
    hour = f"{rt.settings.digest.hour:02d}:{rt.settings.digest.minute:02d}"
    await ctx.reply("setup_timezone_prompt", buttons=rows, hour=hour)


def _settings_item_data(key: str) -> str | None:
    """The /settings button that edits ``key`` (``st:k:<i>``), when that module is present."""
    try:
        from tg_curator.bot.settings import ITEMS, PREFIX
    except ImportError:  # pragma: no cover - the settings module ships with the bot
        return None
    index = next((i for i, item in enumerate(ITEMS) if item.key == key), None)
    return None if index is None else f"{PREFIX}:k:{index}"


def server_timezone() -> str | None:
    """The server's IANA zone without a new dependency: ``$TZ`` when it names a zone, else
    the target of the ``/etc/localtime`` symlink below a ``zoneinfo`` directory."""
    candidates: list[str] = []
    env = os.environ.get("TZ", "").lstrip(":").strip()
    if env:
        candidates.append(env)
    try:
        target = LOCALTIME.resolve(strict=True)
    except (OSError, RuntimeError):
        target = None
    if target is not None:
        parts = target.parts
        if "zoneinfo" in parts:
            index = len(parts) - 1 - parts[::-1].index("zoneinfo")
            candidates.append("/".join(parts[index + 1 :]))
    for name in candidates:
        name = name.removeprefix("posix/").removeprefix("right/")
        if not name or name.startswith("/"):
            continue
        try:
            zoneinfo.ZoneInfo(name)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
            continue
        return "UTC" if name in ("Etc/UTC", "Etc/UCT", "UCT", "Zulu", "Etc/Zulu") else name
    return None


async def _run_command(ctx: Ctx, name: str, args: str = "") -> bool:
    """Run another module's command in this context, as if the owner had typed it; ``False``
    (with a hint to type it) when that module is not part of this installation."""
    entry = ctx.app.commands.get(name)
    if entry is None:
        await ctx.reply("setup_send_command", command=f"/{name} {args}".strip())
        return False
    await ctx.end_flow()  # as for a typed command: it cancels the active flow (§11.1)
    await entry[0](ctx, args)
    return True


# --- buttons -----------------------------------------------------------------------------------


async def setup_callback(ctx: Ctx, data: str) -> None:
    action = f"su:{data}"
    if action == CONTINUE:
        await walkthrough(ctx)
    elif action == CHOOSE_LLM:
        await _run_command(ctx, "llm")
    elif action == SKIP_LLM:
        await _record(ctx.rt, STEPS[STEPS.index("llm") + 1])
        await walkthrough(ctx)
    elif action == TZ_SERVER:
        zone = server_timezone()
        if zone is None:
            await ctx.reply("unknown_choice")
            return
        await ctx.rt.settings_file.set_value(TIMEZONE_KEY, zone)
        await ctx.reply("setup_timezone_saved", zone=html_escape(zone))
        await _go_intro(ctx)
    elif action == TZ_KEEP:
        await _go_intro(ctx)
    elif action == GO_LIVE:
        await _run_command(ctx, "go")
        if (await read_progress(ctx.rt)).current == DONE:
            await walkthrough(ctx)
    else:
        await ctx.reply("unknown_choice")
