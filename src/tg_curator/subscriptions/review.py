"""The weekly review: proposals, their delivery, the owner's decisions (DESIGN §12).

A proposal is a row first and a message second: ``build()`` persists what the numbers justify
(idempotently per review day, so a crash between building and sending never doubles a
proposal), ``send()`` turns the unsent rows into one message each, and ``decide()`` moves a
row through its states on the owner's taps. Nothing here touches the account: an approved row
is picked up by ``actions.py``, which is the only place the subscription writes happen.

The message body is rendered here as well as sent through ``Notifier.proposal`` (the same two
lines of formatting) because an edit — "Left ✓", a confirmation, "Undone" — must reproduce
the body it appends to, and the Notifier does not hand back what it sent.

A review opens with one line per topic accepted from a ``new_topic`` proposal since the last
review: how many of the unsorted posts it absorbed (spec "New topics, found for you", §17.6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import sqlalchemy as sa

from tg_curator.clock import local_date, scheduled_moment
from tg_curator.db import schema
from tg_curator.domain import (
    KV,
    OPEN_PROPOSAL_STATES,
    PROPOSAL_APPROVED,
    PROPOSAL_CONFIRMING,
    PROPOSAL_DONE,
    PROPOSAL_NEVER,
    PROPOSAL_PROPOSED,
    PROPOSAL_SKIPPED,
    PROPOSAL_STATES,
    Chat,
    ChatStats,
    PostStatus,
    Proposal,
    ProposalKind,
)
from tg_curator.errors import CuratorError
from tg_curator.notify import AbsorbedLine
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.telegram.links import permalink
from tg_curator.textutil import first_line, html_escape

if TYPE_CHECKING:
    from tg_curator.config import ReviewSettings
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

CHAT_KINDS: tuple[ProposalKind, ...] = ("folder", "mute", "archive", "leave")
"""The kinds that act on one chat through the account (``rv:`` callbacks)."""
GROUP_ORDER: tuple[ProposalKind, ...] = (*CHAT_KINDS, "new_topic", "merge_topics")
"""Delivery order of the review: the four chat groups, then new topics, then merges."""
DECISIONS = ("approve", "confirm", "cancel", "skip", "never")
EXAMPLE_LINE_CHARS = 100
ABSORBED_REPORTED = "absorbed_reported_on"
"""Payload key of an accepted ``new_topic`` proposal: the review day that reported it."""


def proposal_level(
    stats: ChatStats, chat: Chat, s: ReviewSettings, *, folders_ok: bool
) -> ProposalKind | None:
    """The strongest proposal level whose rule holds for a chat, or ``None`` (§12, §14.10).

    ``leave`` needs the full observation period and nothing unique in it, and is never
    offered for a chat the account created (it could not leave anyway); ``archive`` needs
    the same period; ``mute`` is for chats that mostly repeat others; ``folder`` is the mild
    first step. A chat that posted fewer than ``folder_min_posts`` messages in the window
    says nothing about itself yet and gets no proposal at all.
    """
    if stats.volume < s.folder_min_posts:
        return None
    observed = stats.observed_days
    if observed >= s.leave_min_days and stats.sorted == 0 and not chat.is_creator:
        return "leave"
    if observed >= s.archive_min_days and stats.signal <= s.archive_max_signal:
        return "archive"
    if (
        observed >= s.mute_min_days
        and stats.signal <= s.mute_max_signal
        and stats.duplicate_share >= s.mute_min_duplicates
    ):
        return "mute"
    if folders_ok and observed >= s.folder_min_days and stats.signal <= s.folder_max_signal:
        return "folder"
    return None


def level_in_effect(chat: Chat, now: datetime) -> int:
    """Index into ``CHAT_KINDS`` of the strongest level already in effect for a chat, or -1.

    A chat archived and muted is archived (an archived chat that is not muted un-archives
    itself on its next message, so that alone is not); a mute counts until it runs out; then
    the "Low signal" flag. This holds as much for the owner's own moves in Telegram as for
    the curator's, since ``muted_until`` and ``archived`` are synced from the dialogs.
    """
    muted = chat.muted_until is not None and chat.muted_until > now
    if chat.archived and muted:
        return CHAT_KINDS.index("archive")
    if muted:
        return CHAT_KINDS.index("mute")
    if chat.in_low_signal:
        return CHAT_KINDS.index("folder")
    return -1


@dataclass(frozen=True)
class RenderedProposal:
    """What one proposal looks like to the owner: the plain parts and the HTML body."""

    title: str
    reason: str
    details: str | None
    html: str


async def render_proposal(rt: Runtime, proposal: Proposal) -> RenderedProposal:
    """Title, reason, details and the body exactly as ``Notifier.proposal`` renders it."""
    details = None
    if proposal.kind in CHAT_KINDS:
        chat = await rt.store.get_chat(proposal.chat_id) if proposal.chat_id is not None else None
        title = chat.title if chat is not None else str(proposal.chat_id)
    elif proposal.kind == "new_topic":
        title = str(proposal.payload.get("name", ""))
        details = await _example_lines(rt, proposal.payload.get("example_post_ids", []))
        # The short description the language model wrote from the examples (SPEC: "a sharper
        # name and a short description"), shown above the examples when there is one.
        description = str(proposal.payload.get("description") or "").strip()
        if description:
            line = rt.t("review_topic_description", description=html_escape(description))
            details = f"{line}\n\n{details}" if details else line
    else:
        a = await rt.store.get_topic(int(proposal.payload.get("a_id", 0)))
        b = await rt.store.get_topic(int(proposal.payload.get("b_id", 0)))
        title = f"{a.name if a else '?'} + {b.name if b else '?'}"
    html = rt.t(
        f"notify_proposal_{proposal.kind}",
        title=html_escape(title),
        reason=html_escape(proposal.reason),
        folder=html_escape(rt.settings.folders.low_signal_name),
    )
    if details:
        html = f"{html}\n\n{details}"
    return RenderedProposal(title=title, reason=proposal.reason, details=details, html=html)


async def _example_lines(rt: Runtime, post_ids: list[int]) -> str | None:
    lines: list[str] = []
    for post_id in post_ids:
        post = await rt.store.get_post(int(post_id))
        if post is None:
            continue
        chat = await rt.store.get_chat(post.chat_id)
        line = html_escape(first_line(post.text, EXAMPLE_LINE_CHARS))
        url = permalink(post.chat_id, chat.username if chat else None, post.message_id)
        n = len(lines) + 1
        if url:
            lines.append(rt.t("review_example_link", n=n, url=url, line=line))
        else:
            lines.append(rt.t("review_example_plain", n=n, line=line))
    return "\n".join(lines) or None


def proposal_buttons(rt: Runtime, proposal: Proposal) -> Buttons:
    """The initial keyboard: Approve / Skip / Never ask again, or Create / Rename / Dismiss."""
    t = rt.t
    pid = proposal.id
    if proposal.kind in CHAT_KINDS:
        return [
            [
                Button(t("approve"), data=f"rv:{pid}:approve"),
                Button(t("skip"), data=f"rv:{pid}:skip"),
            ],
            [Button(t("never_ask_again"), data=f"rv:{pid}:never")],
        ]
    if proposal.kind == "new_topic":
        return [
            [
                Button(t("review_btn_create"), data=f"ds:create:{pid}"),
                Button(t("review_btn_rename"), data=f"ds:rename:{pid}"),
            ],
            [Button(t("review_btn_dismiss"), data=f"ds:dismiss:{pid}")],
        ]
    return [
        [
            Button(t("review_btn_merge"), data=f"ds:create:{pid}"),
            Button(t("review_btn_dismiss"), data=f"ds:dismiss:{pid}"),
        ]
    ]


def undo_buttons(rt: Runtime, proposal: Proposal) -> Buttons:
    return [[Button(rt.t("undo"), data=f"rv:{proposal.id}:undo")]]


async def edit_proposal_message(
    rt: Runtime, proposal: Proposal, outcome: str | None, buttons: Buttons | None
) -> None:
    """Rewrite the proposal's message as its body plus ``outcome`` (HTML) and new buttons.

    A failed edit is logged, never raised: the decision or action already happened and the
    row records it; the message is only the owner's view of it.
    """
    bot = rt.bot
    if bot is None or proposal.bot_message_id is None:
        return
    rendered = await render_proposal(rt, proposal)
    html = rendered.html if outcome is None else f"{rendered.html}\n\n{outcome}"
    try:
        await bot.edit_text(
            rt.settings.telegram.owner_id, proposal.bot_message_id, html, buttons=buttons
        )
    except CuratorError as exc:
        log.warning("review: could not update the message of proposal %d: %s", proposal.id, exc)


class ReviewService:
    """``build`` / ``send`` / ``decide`` / ``undo`` / ``tick`` of DESIGN §8 and §12."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    # --- building ----------------------------------------------------------------------------

    async def build(self) -> list[Proposal]:
        rt = self._rt
        store = rt.store
        settings = rt.settings
        now = rt.clock.now()
        review_day = local_date(now, settings.general.timezone)
        stats = await rt.stats.chat_stats() if rt.stats is not None else []
        chats = {c.id: c for c in await store.list_chats()}
        folders_ok = settings.folders.low_signal and await store.kv_get(KV.FOLDERS_DISABLED) is None
        # One open proposal per chat at most, and never a second row for the same review day
        # (a decided one included): that is what makes build() safe to run twice.
        blocked = {
            p.chat_id
            for p in await store.proposals_by_state(OPEN_PROPOSAL_STATES)
            if p.chat_id is not None
        }
        blocked |= {
            p.chat_id
            for p in await store.proposals_by_state(PROPOSAL_STATES, review_day=review_day)
            if p.chat_id is not None
        }
        out: list[Proposal] = []
        for st in stats:
            chat = chats.get(st.chat_id)
            if chat is None or chat.keep or chat.role != "source" or not chat.active:
                continue
            if chat.id in blocked:
                continue
            kind = proposal_level(st, chat, settings.review, folders_ok=folders_ok)
            # A level already in effect (a done move, or the owner's own) is never proposed
            # again; only a stronger one is.
            if kind is None or CHAT_KINDS.index(kind) <= level_in_effect(chat, now):
                continue
            reason = self._reason(st, kind, chat, chats)
            out.append(
                await store.create_proposal(
                    kind, reason=reason, review_day=review_day, chat_id=chat.id
                )
            )
            log.info("review: proposed %s for chat %d (%s)", kind, chat.id, reason)
        if rt.discovery is not None:
            out.extend(await rt.discovery.propose())
        return out

    def _reason(self, st: ChatStats, kind: ProposalKind, chat: Chat, chats: dict[int, Chat]) -> str:
        t = self._rt.t
        # The numbers cover the days the chat was actually observed, not the whole window: a
        # folder or mute can be proposed after a week, and "300 posts in 30 days" would then
        # understate the chat's pace (SPEC: the reason names the numbers in plain words).
        window = self._rt.settings.review.window_days
        days = max(1, min(window, st.observed_days))
        parts = [t("review_reason", volume=st.volume, days=days, sorted=st.sorted)]
        if st.duplicates and st.top_repeated_chat_id is not None:
            source = chats.get(st.top_repeated_chat_id)
            if source is None:
                label = str(st.top_repeated_chat_id)
            elif source.username:
                label = f"@{source.username}"
            else:
                label = source.title
            percent = round(100 * st.duplicate_share)
            parts.append(t("review_reason_repeats", percent=percent, source=label))
        reason = ", ".join(parts)
        if kind == "archive":
            reason += t("review_reason_archive")
        if kind == "leave" and chat.is_admin:
            reason += t("review_reason_admin")
        return reason

    # --- sending -----------------------------------------------------------------------------

    async def send(self) -> int:
        rt = self._rt
        store = rt.store
        await self._report_absorbed()
        cap = rt.settings.review.max_proposals
        pending = [
            p for p in await store.proposals_by_state(PROPOSAL_PROPOSED) if p.bot_message_id is None
        ]
        pending.sort(key=lambda p: (GROUP_ORDER.index(p.kind), p.created_at, p.id))
        to_send, held = pending[:cap], pending[cap:]
        sent = 0
        for proposal in to_send:
            rendered = await render_proposal(rt, proposal)
            # The Notifier spaces the owner's messages one second apart (§8).
            mid = await rt.notifier.proposal(
                proposal.kind,
                rendered.title,
                rendered.reason,
                details=rendered.details,
                buttons=proposal_buttons(rt, proposal),
            )
            if mid is None:
                break  # no owner or no bot yet: the rows stay unsent for the next /review
            await store.set_proposal_fields(proposal.id, bot_message_id=mid)
            sent += 1
        if sent and held:
            await rt.notifier.owner(rt.t("review_held_back", count=len(held)))
        log.info("review: sent %d proposals, %d held back", sent, len(held))
        return sent

    async def _report_absorbed(self) -> None:
        """One line per accepted ``new_topic`` proposal not reported yet: the topic absorbed N
        of M unsorted posts since the start of the window it was re-sorted over.

        N is ``payload["absorbed"]``, what discovery's re-sort moved at acceptance; M adds the
        posts of that window that are still unsorted. A topic accepted today waits for the
        next review (in automatic mode it is created inside this very review), and a row is
        marked reported only once the message went out, so a review without an owner or bot
        reports it later instead of never.
        """
        rt = self._rt
        store = rt.store
        tz = rt.settings.general.timezone
        today = local_date(rt.clock.now(), tz)
        window = timedelta(days=rt.settings.review.cluster_window_days)
        due: list[Proposal] = []
        lines: list[AbsorbedLine] = []
        for proposal in await store.proposals_by_state(PROPOSAL_DONE, kind="new_topic"):
            payload = proposal.payload
            accepted = proposal.executed_at
            if ABSORBED_REPORTED in payload or "absorbed" not in payload or accepted is None:
                continue
            if local_date(accepted, tz) >= today:
                continue
            due.append(proposal)
            topic = await store.get_topic(int(payload["topic_id"]))
            if topic is None or not topic.active:
                continue  # removed since: nothing left to report about it
            since = accepted - window
            absorbed = int(payload["absorbed"])
            left = await self._still_unsorted(since, accepted)
            lines.append(AbsorbedLine(topic.name, absorbed, absorbed + left, local_date(since, tz)))
        if lines and await rt.notifier.review_absorbed(lines) is None:
            return
        for proposal in due:
            await store.set_proposal_fields(
                proposal.id, payload={**proposal.payload, ABSORBED_REPORTED: today.isoformat()}
            )
        if lines:
            log.info("review: reported the absorbed volume of %d new topics", len(lines))

    async def _still_unsorted(self, since: datetime, until: datetime) -> int:
        """Posts dated in ``[since, until)`` that are unsorted now: what the re-sort left."""
        c = schema.posts.c
        stmt = (
            sa.select(sa.func.count())
            .select_from(schema.posts)
            .where(c.status == PostStatus.unsorted.value)
            .where(c.posted_at >= since)
            .where(c.posted_at < until)
        )
        rows = await self._rt.store.execute(stmt)
        return int(rows[0][0]) if isinstance(rows, list) and rows else 0

    # --- decisions ---------------------------------------------------------------------------

    async def decide(self, proposal_id: int, decision: str) -> Proposal:
        rt = self._rt
        store = rt.store
        t = rt.t
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}")
        proposal = await self._get(proposal_id)
        now = rt.clock.now()
        state = proposal.state
        fields: dict[str, object] = {}
        outcome: str | None = None
        buttons: Buttons | None = None
        if decision == "approve" and state == PROPOSAL_PROPOSED and proposal.kind in CHAT_KINDS:
            if proposal.kind == "leave":
                rendered = await render_proposal(rt, proposal)
                fields["state"] = PROPOSAL_CONFIRMING
                outcome = t("review_confirm_leave", title=html_escape(rendered.title))
                buttons = [
                    [
                        Button(t("confirm"), data=f"rv:{proposal.id}:confirm"),
                        Button(t("cancel"), data=f"rv:{proposal.id}:cancel"),
                    ]
                ]
            else:
                fields.update(state=PROPOSAL_APPROVED, decided_at=now)
                outcome = t("review_approved")
        elif decision == "confirm" and state == PROPOSAL_CONFIRMING:
            fields.update(state=PROPOSAL_APPROVED, decided_at=now)
            outcome = t(
                "review_approved_leave",
                per_day=rt.settings.review.leaves_per_day,
                minutes=rt.settings.review.leave_interval_minutes,
            )
        elif decision == "cancel" and state == PROPOSAL_CONFIRMING:
            fields["state"] = PROPOSAL_PROPOSED
            buttons = proposal_buttons(rt, proposal)
        elif decision == "skip" and state in (PROPOSAL_PROPOSED, PROPOSAL_CONFIRMING):
            fields.update(state=PROPOSAL_SKIPPED, decided_at=now)
            outcome = t("review_skipped" if proposal.kind in CHAT_KINDS else "review_dismissed")
        elif decision == "never" and state in (PROPOSAL_PROPOSED, PROPOSAL_CONFIRMING):
            fields.update(state=PROPOSAL_NEVER, decided_at=now)
            if proposal.chat_id is not None:
                await store.set_chat_fields(proposal.chat_id, keep=True)
            outcome = t("review_never")
        else:
            raise CuratorError(t("unknown_choice"))
        await store.set_proposal_fields(proposal.id, **fields)
        await edit_proposal_message(rt, proposal, outcome, buttons)
        log.info("review: proposal %d %s -> %s", proposal.id, decision, fields.get("state"))
        return await self._get(proposal.id)

    async def undo(self, proposal_id: int) -> Proposal:
        """Reverse a done folder/mute/archive action; the account write lives in actions.py
        (§1: only ``actions.py`` and ``folders.py`` call ``mute``/``set_archived``)."""
        actions = self._rt.actions
        if actions is None:
            raise CuratorError(self._rt.t("unknown_choice"))
        return await actions.undo(proposal_id)

    # --- schedule ----------------------------------------------------------------------------

    async def tick(self) -> None:
        """Run the review once per week at the configured moment; a week whose moment passed
        while the service was down is run at the next tick, and ``kv review.last_day``
        remembers the week so a restart later the same day does not run it twice."""
        rt = self._rt
        s = rt.settings.review
        tz = rt.settings.general.timezone
        now = rt.clock.now()
        today = local_date(now, tz)
        day = today - timedelta(days=(today.weekday() - s.weekday_index) % 7)
        if now < scheduled_moment(day, s.hour, 0, tz):
            day -= timedelta(days=7)
        key = day.isoformat()
        if await rt.store.kv_get(KV.REVIEW_LAST_DAY) == key:
            return
        log.info("review: running the review of %s", key)
        await self.build()
        await self.send()
        await rt.store.kv_set(KV.REVIEW_LAST_DAY, key)

    # --- internals ---------------------------------------------------------------------------

    async def _get(self, proposal_id: int) -> Proposal:
        proposal = await self._rt.store.get_proposal(proposal_id)
        if proposal is None:
            raise CuratorError(self._rt.t("unknown_choice"))
        return proposal
