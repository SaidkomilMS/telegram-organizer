"""The weekly review in the bot: ``/review``, the proposal buttons and ``/undo`` (DESIGN §11.2).

Everything that changes state lives in the services: ``ReviewService.decide``/``undo`` move a
proposal through its states and rewrite its message, ``Discovery.accept`` creates or merges
topics. This module only routes the owner's taps and replies to them, so the bot and
``curator review`` can never disagree about what a tap does. What it adds is the conversation
around the taps: the short acknowledgements, the explicit "Yes, leave" label on the one action
that cannot be undone, the name prompt of Rename, and the line saying whether a new topic's
channel is ready.

The services announce a refusal (a stale tap, an action that cannot be undone) by raising a
plain ``CuratorError`` whose message is already a catalogue sentence; those are shown to the
owner as they are. Gateway failures are ``CuratorError`` subclasses and go to the core, which
renders them (§11.1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import sqlalchemy as sa

from tg_curator.bot.core import Ctx, Flow
from tg_curator.db import schema
from tg_curator.domain import (
    OPEN_PROPOSAL_STATES,
    PROPOSAL_CONFIRMING,
    PROPOSAL_DONE,
    Proposal,
    Topic,
)
from tg_curator.errors import CuratorError, TopicExists
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.bot.core import BotApp
    from tg_curator.contracts import Discovery
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

RENAME_FLOW = "review_rename"
REVIEW_ACTIONS = ("approve", "confirm", "cancel", "skip", "never", "undo")
DISCOVERY_ACTIONS = ("create", "rename", "dismiss")
DISCOVERY_KINDS = ("new_topic", "merge_topics")


def register(app: BotApp) -> None:
    app.command("review", review_command, help_key="review_bot_help_review")
    app.command("undo", undo_command, help_key="review_bot_help_undo")
    app.callback("rv", review_callback)
    app.callback("ds", discovery_callback)
    app.flow(RENAME_FLOW, RenameFlow)


# --- /review -----------------------------------------------------------------------------------


async def review_command(ctx: Ctx, args: str) -> None:
    """Build and send the review now; the proposals arrive as their own messages above."""
    rt = ctx.rt
    if rt.review is None:
        await ctx.reply("review_bot_unavailable")
        return
    built = await rt.review.build()
    sent = await rt.review.send()
    log.info("bot: /review built %d proposals, sent %d", len(built), sent)
    if sent == 0:
        await ctx.reply("review_bot_nothing")
    elif sent == 1:
        await ctx.reply("review_bot_sent_one")
    else:
        await ctx.reply("review_bot_sent_many", count=sent)


# --- rv:<proposal_id>:<action> -----------------------------------------------------------------


async def review_callback(ctx: Ctx, data: str) -> None:
    """Approve / Confirm / Cancel / Skip / Never ask again / Undo on a chat proposal."""
    rt = ctx.rt
    raw_id, _, action = data.partition(":")
    if rt.review is None or not raw_id.isdigit() or action not in REVIEW_ACTIONS:
        await ctx.answer(rt.t("unknown_choice"))
        return
    proposal_id = int(raw_id)
    try:
        if action == "undo":
            proposal = await rt.review.undo(proposal_id)
        else:
            proposal = await rt.review.decide(proposal_id, action)
    except CuratorError as exc:
        if not _is_refusal(exc):
            raise
        await ctx.answer(str(exc), alert=True)
        return
    if proposal.state == PROPOSAL_CONFIRMING:
        await _label_leave_confirmation(ctx, proposal)
    await ctx.answer(rt.t(_acknowledgement(action, proposal)))


def _acknowledgement(action: str, proposal: Proposal) -> str:
    """The toast key for a decision that went through."""
    if action == "approve":
        return "review_bot_ack_confirm" if proposal.kind == "leave" else "review_bot_ack_approve"
    if action == "confirm":
        return "review_bot_ack_leave"
    return f"review_bot_ack_{action}"


async def _label_leave_confirmation(ctx: Ctx, proposal: Proposal) -> None:
    """Put [Yes, leave] [Cancel] under the leave confirmation.

    ``ReviewService.decide`` already wrote the confirmation text with generic Confirm/Cancel
    buttons; leaving is the one action that cannot be undone, so the button says what it does.
    A failed edit leaves the generic buttons, which carry the same callbacks.
    """
    bot = ctx.rt.bot
    if bot is None or proposal.bot_message_id is None:
        return
    buttons: Buttons = [
        [
            Button(ctx.rt.t("review_bot_btn_yes_leave"), data=f"rv:{proposal.id}:confirm"),
            Button(ctx.rt.t("cancel"), data=f"rv:{proposal.id}:cancel"),
        ]
    ]
    try:
        await bot.edit_buttons(ctx.chat_id, proposal.bot_message_id, buttons)
    except CuratorError as exc:
        log.warning("review: could not relabel the leave confirmation of %d: %s", proposal.id, exc)


# --- /undo -------------------------------------------------------------------------------------


async def undo_command(ctx: Ctx, args: str) -> None:
    """``/undo`` sent as a reply to a proposal message reverses what that proposal did."""
    rt = ctx.rt
    if rt.review is None:
        await ctx.reply("review_bot_unavailable")
        return
    reply_to = ctx.message.reply_to_id if ctx.message is not None else None
    if reply_to is None:
        await ctx.reply("review_bot_undo_how")
        return
    proposal = await _proposal_of_message(rt, reply_to)
    if proposal is None:
        await ctx.reply("review_bot_undo_not_proposal")
        return
    if proposal.state != PROPOSAL_DONE:
        await ctx.reply("review_bot_undo_nothing")
        return
    try:
        await rt.review.undo(proposal.id)
    except CuratorError as exc:
        if not _is_refusal(exc):
            raise
        await ctx.reply("review_bot_refused", reason=str(exc))
        return
    await ctx.reply("review_bot_undone")


async def _proposal_of_message(rt: Runtime, message_id: int) -> Proposal | None:
    """The proposal whose message is ``message_id`` in the owner chat (every proposal goes
    there, so the id is unambiguous; the newest row wins should a message id ever repeat)."""
    proposals = schema.proposals
    rows = await rt.store.execute(
        sa.select(proposals.c.id)
        .where(proposals.c.bot_message_id == message_id)
        .order_by(proposals.c.id.desc())
        .limit(1)
    )
    if not isinstance(rows, list) or not rows:
        return None
    return await rt.store.get_proposal(int(rows[0][0]))


# --- ds:<create|rename|dismiss>:<proposal_id> --------------------------------------------------


async def discovery_callback(ctx: Ctx, data: str) -> None:
    """Create / Rename / Dismiss on a new-topic proposal, Merge / Dismiss on a merge."""
    rt = ctx.rt
    action, _, raw_id = data.partition(":")
    proposal = await rt.store.get_proposal(int(raw_id)) if raw_id.isdigit() else None
    if (
        proposal is None
        or proposal.kind not in DISCOVERY_KINDS
        or action not in DISCOVERY_ACTIONS
        or (action == "rename" and proposal.kind != "new_topic")
        or rt.review is None
        or rt.discovery is None
    ):
        await ctx.answer(rt.t("unknown_choice"))
        return
    try:
        if action == "dismiss":
            await rt.review.decide(proposal.id, "skip")
            await ctx.answer(rt.t("review_bot_ack_dismiss"))
        elif action == "rename":
            await ctx.start_flow(RENAME_FLOW, proposal_id=proposal.id)
        else:
            await _create(ctx, rt.discovery, proposal)
    except CuratorError as exc:
        if not _is_refusal(exc):
            raise
        await ctx.answer(str(exc), alert=True)


async def _create(ctx: Ctx, discovery: Discovery, proposal: Proposal) -> None:
    """Accept a new topic under its proposed name, or carry out a merge."""
    rt = ctx.rt
    if proposal.kind == "merge_topics":
        await discovery.accept(proposal.id)
        await ctx.answer(rt.t("review_bot_ack_merged"))
        return
    name = str(proposal.payload.get("name", ""))
    try:
        topic = await discovery.accept(proposal.id)
    except TopicExists:
        await ctx.reply("review_bot_name_taken_create", name=html_escape(name))
        return
    await ctx.reply(await created_message(rt, topic, proposal.id))


async def created_message(rt: Runtime, topic: Topic, proposal_id: int) -> str:
    """The confirmation of a created topic: its name, whether its channel is ready, and how
    much of the unsorted volume it took over (``payload["absorbed"]``, set by accept)."""
    lines = [rt.t("review_bot_created", name=html_escape(topic.name))]
    lines.append(await _channel_status(rt, topic))
    accepted = await rt.store.get_proposal(proposal_id)
    absorbed = int(accepted.payload.get("absorbed", 0)) if accepted is not None else 0
    if absorbed:
        days = rt.settings.review.cluster_window_days
        lines.append(rt.t("review_bot_absorbed", count=absorbed, days=days))
    return "\n".join(lines)


async def _channel_status(rt: Runtime, topic: Topic) -> str:
    """One line on the new topic's channel: ready, missing the bot's rights, or not created
    yet because Telegram paces channel creation (§8, ``channel_wait_minutes``)."""
    if topic.channel_id is None:
        wait = rt.topics.channel_wait_minutes() if rt.topics is not None else None
        if wait is not None:
            return rt.t("review_bot_channel_wait", minutes=wait)
        return rt.t("review_bot_channel_none")
    chat = await rt.store.get_chat(topic.channel_id)
    title = html_escape(chat.title if chat is not None else topic.name)
    if not await _bot_can_post(rt, topic.channel_id):
        bot = rt.bot_account.username if rt.bot_account is not None else None
        return rt.t("review_bot_channel_no_bot", channel=title, bot=bot or "")
    return rt.t("review_bot_channel_ready", channel=title)


async def _bot_can_post(rt: Runtime, channel_id: int) -> bool:
    """A channel the bot cannot reach counts as one it cannot post into: either way the owner
    has to add the bot by hand, and the topic itself was created regardless."""
    if rt.bot is None:
        return False
    try:
        return await rt.bot.can_post(channel_id)
    except CuratorError as exc:
        log.warning("review: cannot check the bot's rights in %d: %s", channel_id, exc)
        return False


class RenameFlow(Flow):
    """Rename: ask for the new topic's name, then accept the proposal under that name.

    Only the proposal id is persisted; a name that is taken keeps the conversation at the
    same step so the owner simply sends another one.
    """

    first_step = "name"

    async def start(self) -> None:
        proposal = await self._proposal()
        if proposal is None:
            await self.end()
            await self.ctx.reply("unknown_choice")
            return
        suggested = html_escape(str(proposal.payload.get("name", "")))
        await self.ctx.reply("review_bot_rename_prompt", name=suggested)

    @Flow.step("name")
    async def name(self, text: str) -> None:
        rt = self.ctx.rt
        name = text.strip()
        if not name:
            await self.ctx.reply("review_bot_rename_empty")
            return
        proposal = await self._proposal()
        if proposal is None or rt.discovery is None:
            await self.end()
            await self.ctx.reply("unknown_choice")
            return
        try:
            topic = await rt.discovery.accept(proposal.id, name=name)
        except TopicExists:
            await self.ctx.reply("review_bot_name_taken_rename", name=html_escape(name))
            return
        except CuratorError as exc:
            if not _is_refusal(exc):
                raise
            await self.end()
            await self.ctx.reply("review_bot_refused", reason=str(exc))
            return
        await self.end()
        await self.ctx.reply(await created_message(rt, topic, proposal.id))

    async def _proposal(self) -> Proposal | None:
        """The proposal being renamed, while it is still an open new-topic proposal."""
        proposal = await self.ctx.rt.store.get_proposal(int(self.data.get("proposal_id", 0)))
        if (
            proposal is None
            or proposal.kind != "new_topic"
            or proposal.state not in OPEN_PROPOSAL_STATES
        ):
            return None
        return proposal


def _is_refusal(exc: CuratorError) -> bool:
    """A service's refusal (plain ``CuratorError`` carrying a catalogue sentence), as opposed
    to a gateway failure (a subclass), which the core reports in its own words."""
    return type(exc) is CuratorError
