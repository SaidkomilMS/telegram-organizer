"""/preview: what the curator would have done with the last three days (DESIGN §11.2, §9.6).

``/preview`` first pulls the last three days from every chat when the stored backlog is stale
(older than a day, or never read; ``/preview refresh`` forces it), then replays it with the
current settings and sends a short summary with examples. Nothing is posted: the replay writes
nothing and the backfill only stores posts (§14.15). ``/preview <topic>`` pages through the
individual decisions of one topic, each with the same [Wrong topic] menu a channel post has.

The backfill reads every chat with pauses between them, which takes minutes; meanwhile the
owner chat stays usable (``ctx.unlocked()``: ``/pause`` or a button never waits behind the
read), and a second ``/preview`` that would read again is told the read is under way.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from tg_curator.bot.core import BotApp, Ctx
from tg_curator.bot.corrections import wrong_topic_buttons
from tg_curator.domain import KV, Chat, Decision, PostStatus, PreviewReport, Topic
from tg_curator.errors import CuratorError
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import first_line, html_escape, split_html

log = logging.getLogger(__name__)

PREFIX = "pv"
DAYS = 3
"""The spec's "last three days"."""
BACKFILL_MAX_AGE = timedelta(hours=24)
EXAMPLES = 3
"""Example lines under each count of the summary."""
PAGE_SIZE = 5
LINE_CHARS = 90
MESSAGE_LIMIT = 4096

_READING: weakref.WeakSet[BotApp] = weakref.WeakSet()
"""Apps whose owner chat has a backfill under way (one read at a time)."""


def register(app: BotApp) -> None:
    app.command("preview", preview_command, help_key="preview_help")
    app.callback(PREFIX, on_callback)


async def preview_command(ctx: Ctx, args: str) -> None:
    arg = args.strip()
    if not arg:
        await run_preview(ctx)
    elif arg.casefold() == "refresh":
        await run_preview(ctx, refresh=True)
    else:
        topic = await _find_topic(ctx.rt, arg)
        if topic is None:
            await ctx.reply(
                "preview_unknown_topic", buttons=await _topic_buttons(ctx.rt), name=html_escape(arg)
            )
            return
        await send_page(ctx, topic, 0)


async def run_preview(ctx: Ctx, *, refresh: bool = False) -> None:
    """Backfill when stale (or ``refresh``), replay, send the summary; setup step 6 uses it."""
    rt = ctx.rt
    if rt.preview is None:
        raise RuntimeError("the preview service is not wired")
    if refresh or await backfill_due(rt):
        reading = rt.backfill is not None and rt.backfill.running
        if ctx.app in _READING or reading:
            # one read at a time, whoever started it (another /preview, `curator backfill`)
            await ctx.reply("preview_already_reading")
            return
        _READING.add(ctx.app)
        try:
            await _backfill(ctx)
        finally:
            _READING.discard(ctx.app)
    report = await rt.preview.replay(days=DAYS)
    parts = split_html(await summary_html(rt, report), MESSAGE_LIMIT)
    buttons = await _topic_buttons(rt)
    for i, part in enumerate(parts):
        await ctx.reply(part, buttons=buttons if i == len(parts) - 1 else None)
    # setup step 6 is done once the report was actually sent, not when any backfill ran
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())


async def backfill_due(rt: Runtime) -> bool:
    """True when ``kv backfill.last_run_at`` is absent or older than 24 h (only a run over
    every chat and three days writes it, so a ``curator backfill --chat X`` never counts)."""
    raw = await rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT)
    if not raw:
        return True
    try:
        last = datetime.fromisoformat(str(raw))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return rt.clock.now() - last > BACKFILL_MAX_AGE


async def _backfill(ctx: Ctx) -> None:
    """Run the backfill, editing one progress message as chats are read."""
    rt = ctx.rt
    if rt.backfill is None or rt.user is None:
        await ctx.reply("preview_cannot_read")
        return
    total = len(await rt.store.list_chats(role="source", active=True))
    message_id = await ctx.reply("preview_reading", done=0, total=total)

    async def progress(done: int, total: int, chat: Chat) -> None:
        try:
            await ctx.edit("preview_reading", message_id=message_id, done=done, total=total)
        except CuratorError as exc:  # a missed progress line must not stop the read
            log.debug("preview: progress not shown: %s", exc)

    async with ctx.unlocked():  # minutes of reading: the owner chat stays responsive
        result = await rt.backfill.run(days=DAYS, progress=progress)
    await ctx.edit(
        "preview_read_done",
        message_id=message_id,
        messages=result.messages,
        chats=result.chats,
        new=result.submitted,
    )


# --- the summary -------------------------------------------------------------------------------


async def summary_html(rt: Runtime, report: PreviewReport) -> str:
    """Counts per topic, repeats, unsorted and ignored, each count with up to three examples
    drawn from ``report.decisions`` (the report carries no example field, §8)."""
    topics = await rt.store.list_topics(active=True)
    titles = _ChatTitles(rt)
    blocks = [rt.t("preview_summary_head", days=report.days)]
    for topic in topics:
        picked = [d for d in report.decisions if _counted_for(d) and d.topic_id == topic.id]
        head = rt.t(
            "preview_summary_topic",
            name=html_escape(topic.name),
            count=report.per_topic.get(topic.key, 0),
        )
        if topic.channel_id is None:
            head += rt.t("preview_summary_tracked")
        blocks.append(await _with_examples(rt, head, picked, titles))
    sections: list[tuple[str, int, Callable[[Decision], bool]]] = [
        ("preview_summary_repeats", report.repeats, lambda d: d.status == PostStatus.duplicate),
        ("preview_summary_unsorted", report.unsorted, lambda d: d.status == PostStatus.unsorted),
    ]
    for key, count, wanted in sections:
        picked = [d for d in report.decisions if wanted(d)]
        blocks.append(await _with_examples(rt, rt.t(key, count=count), picked, titles))
    blocks.append(rt.t("preview_summary_ignored", count=report.ignored))
    blocks.append(rt.t("preview_summary_foot"))
    return "\n\n".join(blocks)


def _counted_for(d: Decision) -> bool:
    """The decisions ``PreviewReport.per_topic`` counts (see ``PreviewService.replay``)."""
    return d.topic_id is not None and d.status not in (
        PostStatus.duplicate,
        PostStatus.unsorted,
        PostStatus.rejected,
    )


async def _with_examples(
    rt: Runtime, head: str, decisions: list[Decision], titles: _ChatTitles
) -> str:
    lines = [head]
    for d in decisions[-EXAMPLES:]:  # the newest ones read most familiar
        lines.append(
            rt.t(
                "preview_example_line",
                text=html_escape(first_line(d.candidate.text, LINE_CHARS)),
                source=html_escape(await titles.get(d.candidate.chat_id)),
            )
        )
    return "\n".join(lines)


# --- /preview <topic> --------------------------------------------------------------------------


async def send_page(ctx: Ctx, topic: Topic, page: int) -> None:
    """Five decisions of ``topic``, one message each with [Wrong topic], then ‹ ›."""
    rt = ctx.rt
    if rt.preview is None:
        raise RuntimeError("the preview service is not wired")
    report = await rt.preview.replay(days=DAYS, topic_key=topic.key)
    decisions = [d for d in report.decisions if d.post_id is not None]
    if not decisions:
        await ctx.reply("preview_topic_empty", name=html_escape(topic.name), days=DAYS)
        return
    pages = -(-len(decisions) // PAGE_SIZE)
    page = min(max(page, 0), pages - 1)
    first = page * PAGE_SIZE
    titles = _ChatTitles(rt)
    tz = ZoneInfo(rt.settings.general.timezone)
    for n, d in enumerate(decisions[first : first + PAGE_SIZE], start=first + 1):
        assert d.post_id is not None
        await ctx.reply(
            await _item_html(rt, n, d, topic, titles, tz),
            buttons=wrong_topic_buttons(rt, d.post_id),
        )
    nav: list[Button] = []
    if page > 0:
        nav.append(Button(rt.t("preview_prev"), data=f"{PREFIX}:p:{topic.id}:{page - 1}"))
    if page < pages - 1:
        nav.append(Button(rt.t("preview_next"), data=f"{PREFIX}:p:{topic.id}:{page + 1}"))
    await ctx.reply(
        "preview_page",
        buttons=[nav] if nav else None,
        name=html_escape(topic.name),
        first=first + 1,
        last=min(first + PAGE_SIZE, len(decisions)),
        total=len(decisions),
    )


async def _item_html(
    rt: Runtime, n: int, d: Decision, topic: Topic, titles: _ChatTitles, tz: ZoneInfo
) -> str:
    c = d.candidate
    key = "preview_item_immediate" if d.would_realtime else "preview_item"
    return rt.t(
        key,
        n=n,
        when=c.posted_at.astimezone(tz).strftime("%d %b %H:%M"),
        source=html_escape(await titles.get(c.chat_id)),
        text=html_escape(first_line(c.text, LINE_CHARS)),
        topic=html_escape(topic.name),
        confidence=f"{d.confidence or 0.0:.2f}",
    )


async def on_callback(ctx: Ctx, data: str) -> None:
    """``pv:p:<topic_id>:<page>``: another page (the old navigation buttons go away)."""
    verb, *rest = data.split(":")
    if verb != "p" or len(rest) != 2 or not all(p.isdigit() for p in rest):
        await ctx.reply("unknown_choice")
        return
    topic = await ctx.rt.store.get_topic(int(rest[0]))
    if topic is None or not topic.active:
        await ctx.reply("unknown_choice")
        return
    cb = ctx.callback
    if cb is not None and cb.chat_id == ctx.chat_id and ctx.rt.bot is not None:
        await ctx.rt.bot.edit_buttons(cb.chat_id, cb.message_id, None)
    await send_page(ctx, topic, int(rest[1]))


# --- helpers -----------------------------------------------------------------------------------


async def _find_topic(rt: Runtime, ref: str) -> Topic | None:
    """A topic by key or by name, case-insensitively."""
    wanted = ref.casefold()
    for topic in await rt.store.list_topics(active=True):
        if wanted in (topic.key.casefold(), topic.name.casefold()):
            return topic
    return None


async def _topic_buttons(rt: Runtime) -> Buttons | None:
    """One [name] button per topic opening its decisions, two per row."""
    topics = await rt.store.list_topics(active=True)
    buttons = [Button(t.name, data=f"{PREFIX}:p:{t.id}:0") for t in topics]
    return [buttons[i : i + 2] for i in range(0, len(buttons), 2)] or None


class _ChatTitles:
    """Source titles for one rendering, read from the store once per chat."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._titles: dict[int, str] = {}

    async def get(self, chat_id: int) -> str:
        if chat_id not in self._titles:
            chat = await self._rt.store.get_chat(chat_id)
            self._titles[chat_id] = chat.title if chat is not None else str(chat_id)
        return self._titles[chat_id]
