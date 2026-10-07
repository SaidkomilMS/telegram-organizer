"""The [Wrong topic] button: pick the right topic, the curator learns (DESIGN §11.2, §9.8).

The menu replaces the buttons of the very message that was tapped — a post in a topic channel
or an item of ``/preview <topic>`` — so a correction is two taps and no extra messages in a
channel that must stay readable. A choice calls ``rt.learning.correct`` and nothing else:
stubbing and re-publishing a channel post is ``publisher.move``'s job, which ``correct`` calls.
The core already drops every tap that is not the owner's (§11.1).
"""

from __future__ import annotations

import logging

from tg_curator.bot.core import BotApp, Ctx, check_buttons
from tg_curator.domain import PUB_CANCELLED, Post, PostStatus
from tg_curator.errors import ConfigError
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import Button, Buttons

log = logging.getLogger(__name__)

OPEN = "wt"
CHOOSE = "mv"
NOT_FOR_ME = "0"
CANCEL = "x"


def register(app: BotApp) -> None:
    app.callback(OPEN, on_wrong_topic)
    app.callback(CHOOSE, on_choice)


def wrong_topic_buttons(rt: Runtime, post_id: int) -> Buttons:
    """The single [Wrong topic] row; ``/preview`` puts it under every listed decision."""
    return [[Button(rt.t("corrections_wrong_topic"), data=f"{OPEN}:{post_id}")]]


async def menu_buttons(rt: Runtime, post: Post) -> Buttons:
    """Every other active topic (two per row), then [Not for me] [Cancel]."""
    others = [t for t in await rt.store.list_topics(active=True) if t.id != post.topic_id]
    topics = [Button(t.name, data=f"{CHOOSE}:{post.id}:{t.id}") for t in others]
    rows = [topics[i : i + 2] for i in range(0, len(topics), 2)]
    rows.append(
        [
            Button(rt.t("corrections_not_for_me"), data=f"{CHOOSE}:{post.id}:{NOT_FOR_ME}"),
            Button(rt.t("cancel"), data=f"{CHOOSE}:{post.id}:{CANCEL}"),
        ]
    )
    check_buttons(rows)
    return rows


async def on_wrong_topic(ctx: Ctx, data: str) -> None:
    """``wt:<post_id>``: swap the message's buttons for the menu, in place."""
    post = await _post(ctx, data)
    if post is None:
        return
    await _set_buttons(ctx, await menu_buttons(ctx.rt, post))


async def on_choice(ctx: Ctx, data: str) -> None:
    """``mv:<post_id>:<topic_id | 0 | x>``: correct, or put [Wrong topic] back on cancel."""
    post_part, _, choice = data.partition(":")
    post = await _post(ctx, post_part)
    if post is None:
        return
    rt = ctx.rt
    if choice == CANCEL:
        await _set_buttons(ctx, wrong_topic_buttons(rt, post.id))
        return
    if not choice.isdigit():
        await ctx.answer(rt.t("unknown_choice"))
        return
    new_topic_id = int(choice) or None
    if rt.learning is None:
        raise RuntimeError("the learning service is not wired")
    try:
        result = await rt.learning.correct(post.id, new_topic_id)
    except ConfigError:
        await _set_buttons(ctx, wrong_topic_buttons(rt, post.id))
        await ctx.answer(rt.t("corrections_topic_gone"), alert=True)
        return
    # In a topic channel a moved post was stubbed (its buttons removed) by publisher.move; a
    # post nobody moved, and every /preview item in the private chat, keep the button so the
    # owner can correct again.
    private = ctx.callback is not None and ctx.callback.chat_id == ctx.chat_id
    if private or not result.moved:
        await _set_buttons(ctx, wrong_topic_buttons(rt, post.id))
    # The callback's one answer carries the outcome (the core answers silently only when
    # this takes too long, and the text then comes as a message). A refusal (§9.8 step 1) is
    # an alert, which stays until the owner closes it: nothing else shows that nothing moved.
    refused = post.status in _REFUSED
    await ctx.answer(await _outcome(rt, post, new_topic_id), alert=refused)


_REFUSED = (PostStatus.duplicate, PostStatus.ignored)
"""Posts ``Learning.correct`` refuses to move (§9.8 step 1)."""


async def _outcome(rt: Runtime, post: Post, new_topic_id: int | None) -> str:
    """The toast that says what happened (plain text: callback answers carry no HTML)."""
    if post.status in _REFUSED:
        return rt.t("corrections_refused")
    if new_topic_id is None:
        return rt.t("corrections_not_for_me_done")
    topic = await rt.store.get_topic(new_topic_id)
    name = topic.name if topic else new_topic_id
    # A forward that went out cannot be re-sent elsewhere (§14.1, §9.4): publisher.move only
    # relabels it, so it stays in this channel, cancelled under the new topic.
    row = await rt.store.get_publication(post.id)
    if (
        row is not None
        and row.style == "forward"
        and row.state == PUB_CANCELLED
        and row.topic_id == new_topic_id
        and row.message_ids
    ):
        return rt.t("corrections_relabelled", topic=name)
    return rt.t("corrections_moved", topic=name)


async def _post(ctx: Ctx, raw: str) -> Post | None:
    post = await ctx.rt.store.get_post(int(raw)) if raw.isdigit() else None
    if post is None:
        log.warning("corrections: no post for callback data %r", raw)
        await ctx.answer(ctx.rt.t("unknown_choice"))
    return post


async def _set_buttons(ctx: Ctx, buttons: Buttons) -> None:
    """Edit only the keyboard of the tapped message; its text is never ours to change here."""
    cb = ctx.callback
    if cb is None or ctx.rt.bot is None:
        return
    await ctx.rt.bot.edit_buttons(cb.chat_id, cb.message_id, buttons)
