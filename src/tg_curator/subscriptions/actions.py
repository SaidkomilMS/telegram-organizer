"""Carrying out approved proposals through the account, slowly (DESIGN §12).

Everything here is paced on purpose: one action per tick with a random 20–90 s gap, leaves no
faster than one per ``leave_interval_minutes`` and at most ``leaves_per_day`` in any 24 hours,
so from Telegram's side the account looks like a person tidying up. Each step is "mark, then
act, then mark" with no transaction open across the gateway call (§7), and every outcome is
written back into the proposal's own message so the owner sees what happened without a second
notification. This module and ``folders.py`` are the only callers of ``mute``,
``set_archived`` and ``leave`` (§1), and they act only on rows in state ``approved``.

An outage (``TelegramUnavailable``) is transient (§17.3): it leaves the row ``approved`` and
goes up to the Supervisor, and the next tick runs the whole action again, which is safe because
every step is idempotent. An action that fails for good never leaves half of itself behind:
an archive whose second step failed is unmuted again, and a folder move whose sync failed
clears the chat's flag, so the owner is never left with an effect the message says failed.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from tg_curator.domain import (
    KV,
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PROPOSAL_FAILED,
    PROPOSAL_UNDONE,
    Chat,
    Proposal,
)
from tg_curator.errors import (
    ChatGone,
    CuratorError,
    FloodWait,
    NotAllowed,
    SessionLost,
    TelegramUnavailable,
)
from tg_curator.subscriptions.review import CHAT_KINDS, edit_proposal_message, undo_buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime
    from tg_curator.telegram.gateway import UserGateway

log = logging.getLogger(__name__)

MUTE_FOREVER = datetime.max.replace(tzinfo=UTC)
"""The "forever" mute of an archive (§5 ``mute``: ``datetime.max`` -> ``mute_until=2**31-1``),
made aware so it can also be stored in ``chats.muted_until``."""
SPACING_SECONDS = (20, 90)
LEAVE_WINDOW = timedelta(hours=24)
REVERSIBLE = ("folder", "mute", "archive")
PREVIOUS_MUTE_KEY = "previous_muted_until"
"""Archive payload key: the mute the chat had before the archive (ISO time or null)."""


class ActionExecutor:
    """``tick()`` runs one due action; ``undo()`` reverses a done reversible one."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._rng = random.Random()
        self._not_before: datetime | None = None

    async def tick(self) -> None:
        rt = self._rt
        user = rt.user
        if user is None:
            return
        now = rt.clock.now()
        if self._not_before is not None and now < self._not_before:
            return
        leave_ok = await self._leave_allowed(now)
        for proposal in await rt.store.proposals_by_state(PROPOSAL_APPROVED):
            if proposal.kind not in CHAT_KINDS or not _due(proposal, now):
                continue
            if proposal.kind == "leave" and not leave_ok:
                continue
            await self._execute(user, proposal, now)
            gap = self._rng.uniform(*SPACING_SECONDS)
            self._not_before = now + timedelta(seconds=gap)
            return

    async def undo(self, proposal_id: int) -> Proposal:
        """Reverse a ``done`` folder/mute/archive action (``/undo`` or the [Undo] button)."""
        rt = self._rt
        store = rt.store
        t = rt.t
        proposal = await store.get_proposal(proposal_id)
        if proposal is None:
            raise CuratorError(t("unknown_choice"))
        if proposal.kind == "leave":
            raise CuratorError(t("review_not_undoable"))
        if proposal.kind not in REVERSIBLE or proposal.state != PROPOSAL_DONE:
            raise CuratorError(t("unknown_choice"))
        user = rt.user
        chat_id = proposal.chat_id
        if user is None or chat_id is None:
            raise CuratorError(t("error_session_lost"))
        if proposal.kind == "folder":
            await store.set_chat_fields(chat_id, in_low_signal=False)
            if rt.folders is not None:
                await rt.folders.sync()
        elif proposal.kind == "mute":
            await user.mute(chat_id, None)
            await store.set_chat_fields(chat_id, muted_until=None)
        else:
            # The mute the chat had before the archive (the owner's own, or an earlier approved
            # one) comes back if it is still running; otherwise the chat is unmuted.
            previous = _previous_mute(proposal, rt.clock.now())
            await user.set_archived(chat_id, False)
            await user.mute(chat_id, previous)
            await store.set_chat_fields(chat_id, archived=False, muted_until=previous)
        await store.set_proposal_fields(proposal.id, state=PROPOSAL_UNDONE, result="undone")
        await edit_proposal_message(rt, proposal, t("review_outcome_undone"), None)
        log.info("actions: undid %s for chat %d", proposal.kind, chat_id)
        refreshed = await store.get_proposal(proposal.id)
        return refreshed if refreshed is not None else proposal

    # --- one action --------------------------------------------------------------------------

    async def _execute(self, user: UserGateway, proposal: Proposal, now: datetime) -> None:
        rt = self._rt
        t = rt.t
        chat = await rt.store.get_chat(proposal.chat_id) if proposal.chat_id is not None else None
        if chat is None or chat.role != "source":
            # §1: an output or staging channel is never muted, archived or left.
            await self._finish(proposal, PROPOSAL_FAILED, self._failed(proposal.kind, None))
            return
        try:
            outcome = await self._perform(user, proposal, chat, now)
        except FloodWait as exc:
            retry_at = now + timedelta(seconds=exc.seconds)
            payload = {**proposal.payload, "next_attempt_at": retry_at.isoformat()}
            await rt.store.set_proposal_fields(proposal.id, payload=payload)
            text = t("review_outcome_flood", time=self._local(retry_at).strftime("%H:%M"))
            await edit_proposal_message(rt, proposal, text, None)
            log.warning("actions: %s of chat %d waits until %s", proposal.kind, chat.id, retry_at)
            return
        except (SessionLost, TelegramUnavailable):
            # An outage is transient (§17.3): the row stays ``approved`` and the next tick
            # tries again (every step is idempotent); the Supervisor logs it once per outage.
            raise
        except CuratorError as exc:
            await self._finish(proposal, PROPOSAL_FAILED, self._failed(proposal.kind, exc))
            log.warning("actions: %s of chat %d failed: %s", proposal.kind, chat.id, exc)
            return
        await self._finish(proposal, PROPOSAL_DONE, outcome)
        log.info("actions: %s of chat %d done", proposal.kind, chat.id)

    async def _perform(
        self, user: UserGateway, proposal: Proposal, chat: Chat, now: datetime
    ) -> str:
        """Do the action; returns the outcome text (HTML) for the message."""
        rt = self._rt
        store = rt.store
        t = rt.t
        settings = rt.settings
        if proposal.kind == "folder":
            await store.set_chat_fields(chat.id, in_low_signal=True)
            if rt.folders is not None:
                try:
                    await rt.folders.sync()
                except CuratorError as exc:
                    if not _retried(exc):
                        # The move failed for good: the flag must not let the next periodic
                        # sync put the chat into the folder behind the owner's back.
                        await store.set_chat_fields(chat.id, in_low_signal=False)
                    raise
            return t("review_outcome_folder", folder=html_escape(settings.folders.low_signal_name))
        if proposal.kind == "mute":
            until = now + timedelta(days=settings.review.mute_days)
            await user.mute(chat.id, until)
            await store.set_chat_fields(chat.id, muted_until=until)
            local = self._local(until)
            return t("review_outcome_mute", date=f"{local:%b} {local.day}")
        if proposal.kind == "archive":
            # The mute running before the archive is recorded once, before the first attempt
            # mutes forever (a retry would otherwise read back its own forever-mute synced from
            # the dialogs), so that undo and a failed archive can put it back as it was.
            if PREVIOUS_MUTE_KEY not in proposal.payload:
                running = chat.muted_until is not None and chat.muted_until > now
                previous = chat.muted_until if running else None
                proposal.payload = {
                    **proposal.payload,
                    PREVIOUS_MUTE_KEY: previous.isoformat() if previous is not None else None,
                }
                await store.set_proposal_fields(proposal.id, payload=proposal.payload)
            # An archived chat that is not muted un-archives itself on its next message.
            await user.mute(chat.id, MUTE_FOREVER)
            try:
                await user.set_archived(chat.id, True)
            except CuratorError as exc:
                if not _retried(exc):
                    previous = _previous_mute(proposal, now)
                    await self._unmute_after_failed_archive(user, chat.id, previous)
                raise
            await store.set_chat_fields(chat.id, muted_until=MUTE_FOREVER, archived=True)
            return t("review_outcome_archive")
        await user.leave(chat.id)
        await store.set_chat_fields(chat.id, active=False, left_at=now)
        await self._log_leave(now)
        return t("review_outcome_leave")

    async def _unmute_after_failed_archive(
        self, user: UserGateway, chat_id: int, previous: datetime | None
    ) -> None:
        """Best effort: an archive that failed for good leaves no forever-mute behind; a mute
        the chat already had (the owner's own, or an earlier one) is put back as it was."""
        try:
            await user.mute(chat_id, previous)
        except CuratorError as exc:
            log.warning(
                "actions: could not unmute chat %d after a failed archive: %s", chat_id, exc
            )

    async def _finish(self, proposal: Proposal, state: str, outcome: str) -> None:
        rt = self._rt
        now = rt.clock.now()
        await rt.store.set_proposal_fields(
            proposal.id, state=state, executed_at=now, result=outcome
        )
        reversible = state == PROPOSAL_DONE and proposal.kind in REVERSIBLE
        buttons = undo_buttons(rt, proposal) if reversible else None
        await edit_proposal_message(rt, proposal, outcome, buttons)

    def _failed(self, kind: str, exc: CuratorError | None) -> str:
        t = self._rt.t
        if exc is None:
            error = t("review_error_not_source")
        elif isinstance(exc, NotAllowed) and exc.reason == "creator":
            error = t("review_error_creator")
        elif isinstance(exc, NotAllowed) and exc.reason == "not_a_member":
            error = t("review_error_not_member")
        elif isinstance(exc, ChatGone):
            error = t("review_error_chat_gone")
        else:
            error = html_escape(str(exc))
        return t("review_outcome_failed", verb=t(f"review_verb_{kind}"), error=error)

    # --- leave pacing ------------------------------------------------------------------------

    async def _leave_allowed(self, now: datetime) -> bool:
        s = self._rt.settings.review
        done = await self._recent_leaves(now)
        if len(done) >= s.leaves_per_day:
            return False
        return not done or max(done) + timedelta(minutes=s.leave_interval_minutes) <= now

    async def _recent_leaves(self, now: datetime) -> list[datetime]:
        raw = await self._rt.store.kv_get(KV.LEAVE_LOG, [])
        stamps = [datetime.fromisoformat(s) for s in raw]
        return [ts for ts in stamps if ts > now - LEAVE_WINDOW]

    async def _log_leave(self, now: datetime) -> None:
        recent = await self._recent_leaves(now) + [now]
        await self._rt.store.kv_set(KV.LEAVE_LOG, [ts.isoformat() for ts in recent])

    def _local(self, at: datetime) -> datetime:
        return at.astimezone(ZoneInfo(self._rt.settings.general.timezone))


def _retried(exc: CuratorError) -> bool:
    """Errors after which the row stays ``approved`` and the whole action runs again."""
    return isinstance(exc, (FloodWait, SessionLost, TelegramUnavailable))


def _previous_mute(proposal: Proposal, now: datetime) -> datetime | None:
    """The mute an archive recorded as running before it muted the chat forever, if it is
    still running at ``now`` (one that has run out since is no mute at all)."""
    raw = proposal.payload.get(PREVIOUS_MUTE_KEY)
    until = datetime.fromisoformat(str(raw)) if raw else None
    return until if until is not None and until > now else None


def _due(proposal: Proposal, now: datetime) -> bool:
    """False while a FloodWait longer than the gateway sleeps itself is still running."""
    raw = proposal.payload.get("next_attempt_at")
    return raw is None or datetime.fromisoformat(str(raw)) <= now
