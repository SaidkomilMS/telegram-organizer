"""Owner notifications (DESIGN §11.4).

The spec promises the owner hears from the curator only for a handful of reasons, so this
module is the one place that can message the owner unprompted and it exposes exactly six
named kinds (two of them with a second form: the absorbed-volume lines that open a review and
the provider's own out-of-credit notice, §17.5–6): no other module can invent a notification
without adding a method here (and a string to ``locales/en/notify.toml``). ``owner()`` is
the primitive underneath; it serialises sends so the private chat gets at most one message per
second however many loops fire at once, and it is a quiet no-op until the bot is claimed and
started.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

from tg_curator.clock import local_date
from tg_curator.domain import ProposalKind
from tg_curator.errors import ConfigError, CuratorError
from tg_curator.telegram.gateway import Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

IntakeWarningReason = Literal["stalled", "session_lost"]


@dataclass(frozen=True)
class AbsorbedLine:
    """What a topic found by discovery took in: ``absorbed`` of the ``unsorted`` posts dated
    on or after ``since`` (§17.6)."""

    topic: str
    absorbed: int
    unsorted: int
    since: date


_PROPOSAL_KEYS: dict[str, str] = {
    "folder": "notify_proposal_folder",
    "mute": "notify_proposal_mute",
    "archive": "notify_proposal_archive",
    "leave": "notify_proposal_leave",
    "new_topic": "notify_proposal_new_topic",
    "merge_topics": "notify_proposal_merge_topics",
}


class Notifier:
    """Sends to the owner's private chat through ``rt.bot``; see the module docstring."""

    MIN_GAP_SECONDS = 1.0

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._lock = asyncio.Lock()
        self._last_send: float | None = None

    # --- the primitive ---

    async def owner(self, html: str, *, buttons: Buttons | None = None) -> int | None:
        """Send ``html`` to the owner; returns the message id.

        ``None`` (logged at DEBUG) when there is no owner id yet or no bot; also ``None`` when
        Telegram refuses the send — a notification is best effort and must never take down
        the loop that wanted to send it.
        """
        bot = self._rt.bot
        owner_id = self._owner_id()
        if bot is None or owner_id == 0:
            log.debug(
                "owner notification skipped (no %s): %.60s", "bot" if bot is None else "owner", html
            )
            return None
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._last_send is not None:
                wait = self.MIN_GAP_SECONDS - (loop.time() - self._last_send)
                if wait > 0:
                    await asyncio.sleep(wait)
            try:
                return await bot.send_text(owner_id, html, buttons=buttons)
            except CuratorError as exc:
                log.warning("owner notification not delivered: %s", exc)
                return None
            finally:
                self._last_send = loop.time()

    # --- the six kinds (§11.4) ---

    async def digest_line(self, summary: str) -> int | None:
        """(1) One line per digest run, e.g. ``ML & AI: 15 posts; Fintech: 9``."""
        return await self.owner(self._rt.t("notify_digest", summary=html_escape(summary)))

    async def proposal(
        self,
        kind: ProposalKind,
        title: str,
        reason: str,
        *,
        details: str | None = None,
        buttons: Buttons | None = None,
    ) -> int | None:
        """(2) One message per proposal. ``title`` and ``reason`` are plain text (escaped here);
        ``details`` is HTML already rendered by the caller (example lines with links)."""
        text = self._rt.t(
            _PROPOSAL_KEYS[kind],
            title=html_escape(title),
            reason=html_escape(reason),
            folder=html_escape(self._rt.settings.folders.low_signal_name),
        )
        if details:
            text = f"{text}\n\n{details}"
        return await self.owner(text, buttons=buttons)

    async def review_absorbed(self, lines: Sequence[AbsorbedLine]) -> int | None:
        """(2) The opening message of a review: one line per topic accepted from a proposal
        since the last one, saying how much of the unsorted volume it absorbed (§17.6)."""
        text = "\n".join(
            self._rt.t(
                "notify_review_absorbed",
                topic=html_escape(line.topic),
                absorbed=line.absorbed,
                unsorted=line.unsorted,
                since=_day(line.since),
            )
            for line in lines
        )
        return await self.owner(text)

    async def intake_warning(self, reason: IntakeWarningReason) -> int | None:
        """(3) Intake stopped for over an hour (and ``ping()`` fails), or the session is lost."""
        key = "notify_session_lost" if reason == "session_lost" else "notify_intake_stalled"
        return await self.owner(self._rt.t(key))

    async def cannot_post(self, channel_title: str) -> int | None:
        """(4) The bot cannot post into a topic channel; the caller sends it once per channel
        until that channel recovers."""
        return await self.owner(
            self._rt.t("notify_cannot_post", channel=html_escape(channel_title))
        )

    async def llm_cap_reached(self, month: str, cap: float) -> int | None:
        """(5) The monthly spending cap was crossed (once a month, from ``llm/budget.py``)."""
        return await self.owner(self._rt.t("notify_llm_cap_reached", month=month, cap=cap))

    async def llm_out_of_credit(self, provider: str) -> int | None:
        """(5) The provider itself refused for lack of credit; the curator pauses it until the
        month ends. Sent at most once a month together with the cap line (``llm/budget.py``
        shares ``llm_usage.cap_notified`` between the two)."""
        today = local_date(self._rt.clock.now(), self._rt.settings.general.timezone)
        month_end = today.replace(day=calendar.monthrange(today.year, today.month)[1])
        return await self.owner(
            self._rt.t(
                "notify_llm_out_of_credit", provider=html_escape(provider), until=_day(month_end)
            )
        )

    async def topic_created(
        self, name: str, *, has_channel: bool = True, wait_minutes: int | None = None
    ) -> int | None:
        """(6) Automatic mode created a topic without asking (from ``Discovery``). Without a
        channel (channel pacing, §8) the message says so instead of claiming one exists."""
        if has_channel:
            key = "notify_topic_created"
        elif wait_minutes is not None:
            key = "notify_topic_created_wait"
        else:
            key = "notify_topic_created_no_channel"
        return await self.owner(self._rt.t(key, name=html_escape(name), minutes=wait_minutes))

    # --- internals ---

    def _owner_id(self) -> int:
        try:
            return self._rt.settings.telegram.owner_id
        except ConfigError:
            return 0


def _day(day: date) -> str:
    """``Oct 31``: the short form the spec's own messages use ("Muted until Nov 4")."""
    return f"{day:%b} {day.day}"
