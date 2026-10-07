"""``/stats`` and ``/digest`` in the bot (DESIGN §11.2 reports.py, §9.5, §12).

``/stats`` is the cleanup list of the spec: every chat with its volume, signal, duplicate
share, published count and how long it has been observed, the weakest first so the chats
worth dropping head the list. It is a monospace table because the numbers are compared
down the columns. Below it, one line per topic says how many posts the topic caught, a topic
without a channel included (it is tracked in the statistics, spec "Topic channels").
``/digest`` sends the manual digest now (a new, clearly marked digest per topic; the
scheduled one still runs) and ``/digest preview`` shows tonight's digest to the
owner only, every part prefixed so it is never mistaken for the real thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from tg_curator.bot.core import BotApp, Ctx
from tg_curator.domain import ChatStats, DigestDraft, DigestResult
from tg_curator.errors import ConfigError, CuratorError
from tg_curator.pipeline.render import TEXT_LIMIT
from tg_curator.textutil import html_escape, split_html

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime
    from tg_curator.subscriptions.stats import TopicStats

TITLE_CHARS = 24
"""Chat titles are cut so a row stays on one line of a phone screen."""


# --- /stats ----------------------------------------------------------------------------------


def percent(share: float) -> str:
    """``0.023`` -> ``2%``; a non-zero share below one percent is not shown as ``0%``."""
    if 0 < share < 0.01:
        return "<1%"
    return f"{share * 100:.0f}%"


def stats_row(s: ChatStats) -> str:
    title = s.title if len(s.title) <= TITLE_CHARS else s.title[: TITLE_CHARS - 1] + "…"
    return (
        f"{percent(s.signal):>4} {percent(s.duplicate_share):>4} {s.volume:>5} "
        f"{s.published:>4} {s.observed_days:>3}  {title}"
    )


def stats_parts(
    rt: Runtime,
    stats: list[ChatStats],
    *,
    days: int,
    topics: Sequence[TopicStats] = (),
) -> list[str]:
    """The table as HTML parts of at most one message each, weakest signal first, then the
    per-topic block."""
    ordered = sorted(stats, key=lambda s: (s.signal, -s.volume, s.title.casefold(), s.chat_id))
    rows = [rt.t("reports_stats_columns")] + [stats_row(s) for s in ordered]
    table = html_escape("\n".join(rows))
    text = rt.t("reports_stats", days=days, count=len(stats), table=table)
    if topics:
        text += "\n\n" + topic_block(rt, topics, days=days)
    return split_html(text, TEXT_LIMIT)


def topic_block(rt: Runtime, topics: Sequence[TopicStats], *, days: int) -> str:
    """One line per active topic; a topic without a channel is marked as tracked only."""
    lines = []
    for t in topics:
        name = html_escape(t.name)
        if not t.has_channel:
            name += rt.t("reports_stats_topic_no_channel")
        lines.append(
            rt.t("reports_stats_topic_line", name=name, posts=t.posts, immediate=t.immediate)
        )
    return rt.t("reports_stats_topics", days=days, lines="\n".join(lines))


async def stats_command(ctx: Ctx, args: str) -> None:
    rt = ctx.rt
    if args not in ("", "all"):
        await ctx.reply("reports_stats_usage")
        return
    service = rt.stats
    if service is None:
        raise ConfigError("statistics are not running in this process")
    stats = await service.chat_stats(include_left=args == "all")
    if not stats:
        await ctx.reply("reports_stats_empty")
        return
    days = rt.settings.review.window_days
    topics = await service.topic_stats(days)
    for part in stats_parts(rt, stats, days=days, topics=topics):
        await ctx.reply(part)


# --- /digest ---------------------------------------------------------------------------------


async def digest_command(ctx: Ctx, args: str) -> None:
    """``/digest [topic]`` sends now; ``/digest preview [topic]`` shows tonight's drafts."""
    rt = ctx.rt
    service = rt.digest
    if service is None:
        raise ConfigError("the digest is not running in this process")
    words = args.split()
    preview = bool(words) and words[0] == "preview"
    topic_key = words[1 if preview else 0] if len(words) > (1 if preview else 0) else None
    try:
        if preview:
            await _send_preview(ctx, await service.preview(topic_key))
        else:
            await _report_results(ctx, await service.send(topic_key))
    except ConfigError:
        raise
    except CuratorError as exc:
        # The service refuses with one catalogue sentence (not live, paused, unknown topic).
        await ctx.reply(html_escape(str(exc)))


async def _report_results(ctx: Ctx, results: list[DigestResult]) -> None:
    sent = [r for r in results if r.item_count > 0]
    if not sent:
        await ctx.reply("reports_digest_nothing")
        return
    names = await _topic_names(ctx.rt)
    lines = []
    for r in sent:
        name = html_escape(names.get(r.topic_key, r.topic_key))
        key = "reports_digest_line" if r.message_ids else "reports_digest_line_retry"
        lines.append(ctx.rt.t(key, topic=name, count=r.item_count))
    await ctx.reply("reports_digest_sent", lines="\n".join(lines))


async def _send_preview(ctx: Ctx, drafts: list[DigestDraft]) -> None:
    rt = ctx.rt
    names = await _topic_names(rt)
    shown = [d for d in drafts if d.items]
    if not shown:
        await ctx.reply("reports_preview_nothing")
        return
    prefix = rt.t("reports_preview_prefix")
    for draft in shown:
        for part in draft.parts:
            for chunk in split_html(prefix + part, TEXT_LIMIT):
                await ctx.reply(chunk)
    empty = [html_escape(names.get(d.topic_key, d.topic_key)) for d in drafts if not d.items]
    if empty:
        await ctx.reply("reports_preview_empty_topics", topics=", ".join(empty))


async def _topic_names(rt: Runtime) -> dict[str, str]:
    return {t.key: t.name for t in await rt.store.list_topics(active=None)}


def register(app: BotApp) -> None:
    """Wire ``/stats`` and ``/digest`` into ``app``."""
    app.command("stats", stats_command, help_key="reports_help_stats")
    app.command("digest", digest_command, help_key="reports_help_digest")
