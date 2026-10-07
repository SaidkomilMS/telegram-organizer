"""The outbox: repost/forward styles, pacing, +N edits and moves (DESIGN §9.4).

The ``publications`` table is the only memory this module has. A row goes ``pending`` ->
``sending`` -> ``sent`` with every Telegram await outside any transaction, and a post counts
as published only once Telegram returned a message id; a crash in between leaves a row in
``sending`` that ``reconcile()`` settles at the next start by asking the account what the
channel really shows. The dirty flag for ``+N`` edits is likewise a comparison between two
columns, so nothing is lost when the process dies.

A repost that takes several messages (media first, or text split into parts) is sent step by
step, and every bot message id is committed to ``message_ids`` as soon as Telegram returns it,
the way the digest outbox commits its parts (§9.5). The plan is rendered with the
corroboration count stored on the row before the first message goes out, so a retry or a
restart rebuilds the same split and resumes after the last committed message instead of
sending the post again from the start. Messages that went out but are no longer wanted
(a move, a cancelled partial send) are stubbed; a stub Telegram refused for a transient reason
is kept on the ``moved_from`` entry as ``pending_stubs`` and retried by every tick, and the row
is not sent anywhere else until all of them went through (§14.1).

Every piece of work on one post — the send, a ``+N`` edit, a move, a reconcile — holds the
per-post lock and re-reads the row inside it, so none of them acts on a stale snapshot.

Pacing never sleeps: a tick sends what the per-channel and global budgets allow right now and
leaves the rest for the next tick, so a frozen test clock and a busy news day behave alike.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import logging
import math
import re
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import sqlalchemy as sa
from sqlalchemy.engine import Row

from tg_curator.db import schema
from tg_curator.domain import (
    KV,
    PUB_CANCELLED,
    PUB_FAILED,
    PUB_PENDING,
    PUB_RETRACTED,
    PUB_SENDING,
    PUB_SENT,
    Chat,
    MoveResult,
    Post,
    PostStatus,
    Publication,
    PublishStyle,
    Topic,
)
from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    CuratorError,
    FloodWait,
    ForwardsRestricted,
    MediaUnavailable,
    NotAllowed,
    NotOwnedError,
    TelegramUnavailable,
)
from tg_curator.pipeline import render
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.telegram.links import permalink

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime
    from tg_curator.telegram.gateway import UserGateway

log = logging.getLogger(__name__)

GLOBAL_GAP = timedelta(seconds=3)
"""Minimum spacing between any two sends, whatever the channel (§9.4 pacing)."""
MAX_WRITES_PER_MINUTE = 15
"""Per-channel ceiling on messages written (sends and edits) per minute; Telegram allows 20.
It counts messages, not calls: an album of 10 is 10 writes. The staging channel, which the
account fills with media copies for every topic channel, has the same budget (§14)."""
EDIT_COOLDOWN = timedelta(seconds=60)
"""A ``+N`` header is edited at most once a minute per post."""
BACKOFF_BASE_SECONDS = 30
BACKOFF_CAP_SECONDS = 600
CANNOT_POST_RETRY = timedelta(minutes=10)
CANNOT_POST_GIVE_UP = timedelta(hours=6)
"""After this long a row the bot cannot post goes ``cancelled`` and the post to the digest; the
same limit applies to a row left in ``sending`` that the account cannot verify."""
RECONCILE_MATCH_CHARS = 100
"""Length of the text snippet used to find a sent message that carries no source link."""
ORPHAN_CHECK_EVERY = timedelta(minutes=1)
"""How often a tick retries the lookup for rows a failed reconcile left in ``sending``."""
PERMANENT_EDIT_ERRORS = (NotAllowed, ChatGone, BotCannotPost)
"""Edit failures that will not go away by retrying: the message is gone or not the bot's, the
channel is gone, or the bot lost its rights there."""

WRONG_TOPIC_KEY = "publisher_wrong_topic"
WRONG_TOPIC_DEFAULT = "Wrong topic"
"""Used only while ``locales/en/publisher.toml`` does not define the key."""

PENDING_STUBS = "pending_stubs"
"""Key of a ``moved_from`` entry: ``[[message_id, stub_html], ...]`` still to be applied."""


class DeliveryUnverifiable(TelegramUnavailable):
    """A retry could not ask the account whether an earlier attempt reached the channel.

    Transient like any outage: the row stays ``failed`` with backoff and the check runs again
    on the next attempt, because sending blindly could post twice (§16); after 6 h the row is
    cancelled and the post goes to the digest, like a ``sending`` row nobody can verify.
    """


class AccountFloodWait(FloodWait):
    """A flood wait the account got while copying media into staging: it holds back the
    media copies until it is over, not the bot's writes into the topic channel."""


NOTHING_SENT_ERRORS = ("FloodWait", "AccountFloodWait", "BotCannotPost", "ChatGone")
"""``last_error`` types that prove the failed attempt delivered nothing: Telegram refused the
write itself. Any other failure may have come after delivery (a timeout), so the retry checks
the channel first."""

_TAG_RE = re.compile(r"<[^>]+>")


def _publications(rows: Sequence[Row[Any]] | int) -> list[Publication]:
    """Rows of this module's own SELECTs over ``publications`` as dataclasses (columns and
    fields share names by contract, §8); an int is the rowcount of a non-SELECT."""
    if isinstance(rows, int):
        return []
    return [Publication(**dict(r._mapping)) for r in rows]


def _plain(markup: str) -> str:
    """The text Telegram shows for ``markup``: what ``find_message`` matches against."""
    return html_lib.unescape(_TAG_RE.sub("", markup)).strip()


def _has_pending_stubs(row: Publication) -> bool:
    return any(entry.get(PENDING_STUBS) for entry in row.moved_from or [])


def _may_have_delivered(row: Publication) -> bool:
    """Whether an earlier attempt of ``row`` (the snapshot before ``attempts`` is incremented)
    may have reached the channel without its id being committed."""
    if row.attempts <= 0:
        return False
    error = row.last_error or ""
    return not any(error.startswith(f"{name}:") for name in NOTHING_SENT_ERRORS)


@dataclass(frozen=True)
class _Sent:
    """What one delivery produced: bot message ids in send order, the staging copies used
    (empty when the media stayed at the source), the style that was actually applied and the
    corroboration count the header was rendered with."""

    message_ids: list[int]
    staging_ids: list[int]
    style: PublishStyle
    shown: int


@dataclass(frozen=True)
class _Step:
    """One bot write of a repost: the media copy (with or without the caption) or one text
    part. ``count`` is how many message ids it produces (an album yields one per item)."""

    kind: Literal["media", "caption", "text"]
    html: str | None
    last: bool
    staging_ids: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.staging_ids) if self.kind == "media" else 1


class Publisher:
    """The outbox service of §9.4; see the module docstring."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._locks: dict[int, asyncio.Lock] = {}
        self._tick_lock = asyncio.Lock()
        self._channel_writes: dict[int, deque[datetime]] = defaultdict(deque)
        self._flood_until: dict[int, datetime] = {}
        self._last_send: datetime | None = None
        self._staging_warned = False
        self._staging_verified: int | None = None
        self._cannot_post_notified: set[int] = set()
        self._stubs_dirty = True  # scan once after start: stubs may be pending from before
        self._next_orphan_check: datetime | None = None
        self._unverified_warned: set[int] = set()

    # --- the protocol ------------------------------------------------------------------------

    async def enqueue(self, post_id: int) -> bool:
        """Insert the ``pending`` row and set the post ``queued`` in one transaction.

        An existing row in any state means ``False``: a ``cancelled`` one says the post left
        the real-time path for good and a later promotion must not bring it back.
        """
        store = self._rt.store
        post = await store.get_post(post_id)
        if post is None or post.topic_id is None:
            return False
        async with store.begin() as conn:
            row = await store.create_publication(post_id, post.topic_id, conn=conn)
            if row is None:
                return False
            await store.set_post_fields(post_id, status=PostStatus.queued, conn=conn)
        log.info("queued post %d %s", post_id, await self._log_names(post))
        return True

    async def tick(self) -> None:
        """Retry pending stubs, send what is due within the budgets, catch up ``+N``
        headers, then retry the lookup for rows a failed reconcile left in ``sending``.

        Nothing happens unless publishing is live and not paused; a tick that overlaps a
        running one is skipped rather than queued.
        """
        if not await self._active() or self._tick_lock.locked():
            return
        async with self._tick_lock:
            await self._retry_stubs()
            await self._send_due()
            await self._edit_corroboration()
            await self._retry_orphaned()

    async def reconcile(self) -> None:
        """Settle rows left in ``sending`` by a crash, then apply the stale-queued rule.

        The account reads the channel: the committed ``message_ids`` are what surely went
        out, and the message of the next step, found by its text or the source link (or the
        forward origin), means that one went out too. A row whose every step is accounted for
        becomes ``sent``; otherwise it goes back to ``pending`` with what was found, and the
        next tick sends only the missing tail. Then queued posts older than ``hold_minutes``
        fall into the digest instead of being posted late (§14.6).

        Safe while the loops run (``/go``, ``/resume``, a re-bind): every row is handled under
        its per-post lock and re-read there, so a send in flight is waited for, not undone.
        """
        store = self._rt.store
        for stale in await store.publications_in_state(PUB_SENDING):
            async with self._lock(stale.post_id):
                row = await store.get_publication(stale.post_id)
                if row is None or row.id != stale.id or row.state != PUB_SENDING:
                    continue  # a send in this process (or a move) settled it meanwhile
                await self._reconcile_sending(row)
        await self._retry_stubs()
        await self._cancel_stale()

    async def note_corroboration(self, post_id: int) -> None:
        """Nothing to store: ``posts.corroboration > publications.shown_corroboration`` is
        the dirty flag ``tick`` reads, which is what makes the edit crash-safe."""
        row = await self._rt.store.get_publication(post_id)
        if row is not None and row.state == PUB_SENT and row.style in ("repost", "forward"):
            log.debug("post %d: +N edit pending (shown %d)", post_id, row.shown_corroboration)

    async def move(self, post_id: int, new_topic_id: int | None) -> MoveResult:
        """Stub the old messages and re-point, retract or cancel the row (§9.4).

        A ``sending`` row waits for its send under the per-post lock, so what is stubbed is
        what really went out. ``republished`` means the row was re-pointed to a fresh
        ``pending`` and the next live tick sends it; the new ids exist only then, so
        ``new_message_ids`` is empty here. ``stubbed`` is true only when every old message
        was stubbed; the ones Telegram refused for a transient reason are kept as pending
        stubs, retried by every tick, and the row is not sent again until they went through.

        A forward that went out cannot be edited or deleted, so it is never published a second
        time: moved to a topic with a channel, its row is ``cancelled`` with the post still
        ``published`` where it is, and only the label changes (§14.1); its button message is
        stubbed to ``↪ belongs in <b>{topic}</b>``.
        """
        store = self._rt.store
        async with self._lock(post_id):
            row = await store.get_publication(post_id)
            if row is None or row.state == PUB_CANCELLED:
                return MoveResult(republished=False, stubbed=False, new_message_ids=[])
            new_topic = None if new_topic_id is None else await store.get_topic(new_topic_id)
            relocatable = (
                new_topic is not None and new_topic.active and new_topic.channel_id is not None
            )
            forward_out = row.style == "forward" and bool(row.message_ids)
            stubbed = False
            failed: list[list[Any]] = []
            plan: list[list[Any]] = []
            if row.state != PUB_RETRACTED and row.channel_id is not None and row.message_ids:
                if row.style == "repost":
                    # a sent repost, or the part of one a failed or interrupted send left behind
                    plan = self._stub_plan(row, new_topic)
                elif row.style == "forward" and row.state == PUB_SENT:
                    # the forwards stay as they are; the bot's button message is the last id
                    kind: Literal["moved", "not_for_me", "belongs"] = (
                        "not_for_me" if new_topic is None else "belongs" if relocatable else "moved"
                    )
                    name = None if new_topic is None else new_topic.name
                    plan = [[row.message_ids[-1], render.stub(kind, name, t=self._rt.t)]]
            if plan:
                assert row.channel_id is not None
                failed = await self._apply_stubs(row.channel_id, post_id, plan)
                stubbed = not failed
                if failed:
                    log.warning(
                        "post %d: %d message(s) in %d not stubbed yet; retried before it is "
                        "sent anywhere else",
                        post_id,
                        len(failed),
                        row.channel_id,
                    )
            fields: dict[str, Any] = {}
            if row.state != PUB_RETRACTED:
                entry: dict[str, Any] = {
                    "channel_id": row.channel_id,
                    "message_ids": list(row.message_ids),
                    "topic_id": row.topic_id,
                }
                if failed:
                    entry[PENDING_STUBS] = failed
                    self._stubs_dirty = True
                fields["moved_from"] = list(row.moved_from or []) + [entry]
            republished = False
            if new_topic_id is None:
                fields["state"] = PUB_RETRACTED
                status = PostStatus.rejected
            elif new_topic is None or not relocatable:
                fields["state"] = PUB_CANCELLED
                status = PostStatus.tracked
            elif forward_out:
                # never in two topic channels: the forward stays, labelled with the new topic
                assert new_topic is not None
                fields.update(state=PUB_CANCELLED, topic_id=new_topic.id)
                status = PostStatus.published
            else:
                fields.update(
                    topic_id=new_topic.id,
                    channel_id=new_topic.channel_id,
                    style=None,
                    state=PUB_PENDING,
                    message_ids=[],
                    shown_corroboration=0,
                    edited_at=None,
                    attempts=0,
                    next_attempt_at=None,
                    last_error=None,
                    sent_at=None,
                )
                status = PostStatus.queued
                republished = True
            async with store.begin() as conn:
                await store.set_publication_fields(row.id, conn=conn, **fields)
                await store.set_post_fields(post_id, status=status, conn=conn)
        log.info(
            "moved post %d: row %s -> %s, post %s", post_id, row.state, fields["state"], status
        )
        return MoveResult(republished=republished, stubbed=stubbed, new_message_ids=[])

    # --- sending -----------------------------------------------------------------------------

    async def _active(self) -> bool:
        if not self._rt.settings.publishing.live:
            return False
        return not bool(await self._rt.store.kv_get(KV.SERVICE_PAUSED))

    async def _due_rows(self) -> list[Publication]:
        now = self._rt.clock.now()
        p = schema.publications
        stmt = (
            sa.select(p)
            .where(
                sa.or_(
                    p.c.state == PUB_PENDING,
                    sa.and_(p.c.state == PUB_FAILED, p.c.next_attempt_at <= now),
                )
            )
            .order_by(p.c.created_at, p.c.id)
        )
        return _publications(await self._rt.store.execute(stmt))

    @staticmethod
    def _is_due(row: Publication, now: datetime) -> bool:
        if row.state == PUB_PENDING:
            return True
        return (
            row.state == PUB_FAILED
            and row.next_attempt_at is not None
            and row.next_attempt_at <= now
        )

    async def _send_due(self) -> None:
        for candidate in await self._due_rows():
            async with self._lock(candidate.post_id):
                stop = await self._send_one(candidate)
            if stop:
                break

    async def _send_one(self, candidate: Publication) -> bool:
        """Send one due row under its per-post lock; ``True`` when the global gap stops the
        tick. Everything is resolved from the row as it is now, not from the tick's list:
        a ``move()`` that ran in between may have retracted, cancelled or re-pointed it."""
        store = self._rt.store
        row = await store.get_publication(candidate.post_id)
        if row is None or row.id != candidate.id or not self._is_due(row, self._rt.clock.now()):
            return False
        if _has_pending_stubs(row):
            return False  # its old messages are not stubbed yet (§14.1)
        post = await store.get_post(row.post_id)
        if post is None:
            await self._cancel(row, None, error="post missing")
            return False
        topic = await store.get_topic(row.topic_id)
        if topic is None or not topic.active:
            await self._cancel(row, PostStatus.unsorted, error="topic inactive")
            return False
        if topic.channel_id is None:
            await self._cancel(row, PostStatus.tracked, error="topic has no channel")
            return False
        channel = topic.channel_id
        if row.style == "forward" and row.message_ids and row.channel_id is not None:
            # the forward is out: its button message follows it, wherever the topic points now
            channel = row.channel_id
        now = self._rt.clock.now()
        if not self._global_free(now):
            return True
        if not self._channel_free(channel, now):
            return False
        if row.message_ids and row.channel_id is not None and row.channel_id != channel:
            # the topic changed its channel after part of the post went out: stub that part
            # and start over in the new channel once the stubs went through
            await self._drop_partial(row)
            return False
        chat = await self._source_chat(post)
        # a partly sent row resumes in the style it started with
        style: PublishStyle = (
            (row.style or "repost") if row.message_ids else self._style(post, chat)
        )
        if not self._channel_free(channel, now, need=self._expected_messages(row, post, style)):
            return False  # the album would not fit this minute's budget; a later tick sends it
        staged = self._expected_staging(row, post, style)
        staging = self._rt.settings.publishing.staging_channel
        if staged and not self._channel_free(staging, now, need=staged, gap=GLOBAL_GAP):
            return False  # the account's copies into staging wait for room too
        await store.set_publication_fields(
            row.id, state=PUB_SENDING, channel_id=channel, style=style, attempts=row.attempts + 1
        )
        try:
            sent = await self._deliver(row, post, chat, channel, style)
        except Exception as exc:
            if not isinstance(exc, CuratorError):
                log.exception("publishing post %d: unexpected error", post.id)
            if isinstance(exc, FloodWait) and not isinstance(exc, AccountFloodWait):
                self._flood(channel, exc)
            after = await store.get_publication(row.post_id)
            if after is not None and len(after.message_ids) > len(row.message_ids):
                written = len(after.message_ids) - len(row.message_ids)
                self._record_write(channel, self._rt.clock.now(), send=True, count=written)
            await self._failed(row, topic, exc)
            return False
        await self._mark_sent(row, post, channel, sent)
        written = max(1, len(sent.message_ids) - len(row.message_ids))
        self._record_write(channel, self._rt.clock.now(), send=True, count=written)
        return False

    @staticmethod
    def _expected_messages(row: Publication, post: Post, style: PublishStyle) -> int:
        """How many messages the send will still write into the topic channel (an estimate:
        the album or forwarded messages plus one text; long texts may add a part)."""
        if style == "forward":
            total = len(post.message_ids) + 1  # the forwards and the bot's button message
        elif post.media in render.COPYABLE_MEDIA and not post.noforwards:
            total = len(post.message_ids) + 1
        else:
            total = 1
        return max(1, total - len(row.message_ids))

    def _expected_staging(self, row: Publication, post: Post, style: PublishStyle) -> int:
        """How many media copies the account will put into staging for this send (0: none)."""
        if style != "repost" or row.message_ids or row.staging_ids:
            return 0
        if post.media not in render.COPYABLE_MEDIA or post.noforwards or self._rt.user is None:
            return 0
        if not self._rt.settings.publishing.staging_channel:
            return 0
        return len(post.message_ids)

    async def _deliver(
        self, row: Publication, post: Post, chat: Chat, channel: int, style: PublishStyle
    ) -> _Sent:
        """Forward or repost; a forward that Telegram refuses falls back to a repost.

        The fallback is committed as ``style='repost'`` before the bot sends anything, so a
        crash afterwards is reconciled by the source link, not by a forward origin the repost
        does not carry.
        """
        if style == "forward" and (self._rt.user is not None or row.message_ids):
            try:
                return await self._forward(row, post, chat, channel)
            except ForwardsRestricted:
                log.info("post %d: source forbids forwarding, reposting instead", post.id)
            except MediaUnavailable as exc:
                # Deleted at the source since it was read: the stored text still goes out as
                # first seen (SPEC: "deleted posts are not removed"), with its source link.
                log.info("post %d: source is gone (%s), reposting instead", post.id, exc)
        if style != "repost":
            await self._rt.store.set_publication_fields(row.id, style="repost")
        return await self._repost(row, post, chat, channel)

    async def _forward(self, row: Publication, post: Post, chat: Chat, channel: int) -> _Sent:
        """The account forwards the original, then the bot sends the post's button message:
        the header with the source link and ``+N`` and the ``[Wrong topic]`` button (§9.4).

        The forward ids are committed before the button message goes out, and the button
        message's id is always the LAST id, so ``+N`` edits and stubs target it as in repost
        style. A retry first asks the account whether the earlier attempt's forward (or button
        message) reached the channel anyway, so a timeout after delivery never forwards twice;
        a check that cannot run keeps the row waiting (``DeliveryUnverifiable``).
        """
        store = self._rt.store
        done = list(row.message_ids)
        verify = _may_have_delivered(row)
        if done:
            corroboration = row.shown_corroboration
        else:
            if verify:
                done = await self._checked(
                    post, lambda user: self._lookup_forward(user, row, post, channel)
                )
            if not done:
                user = self._rt.user
                assert user is not None  # checked by _deliver
                done = await user.forward(post.chat_id, post.message_ids, channel)
            corroboration = post.corroboration
            await store.set_publication_fields(
                row.id, message_ids=list(done), shown_corroboration=corroboration
            )
            verify = False  # the button message only ever follows committed forward ids
        companion_html = render.forward_companion(post, chat, corroboration, t=self._rt.t)
        step = _Step("text", companion_html, True)
        if verify and permalink(chat.id, chat.username, post.message_id) is not None:
            found = await self._find_delivered(row, post, chat, channel, step, done)
            if found:
                return _Sent(done + found, [], "forward", corroboration)
        bot = self._rt.bot
        assert bot is not None and step.html is not None
        companion = await bot.send_text(
            channel, step.html, buttons=self._wrong_topic(post.id), silent=True
        )
        return _Sent(done + [companion], [], "forward", corroboration)

    async def _lookup_forward(
        self, user: UserGateway, row: Publication, post: Post, channel: int
    ) -> list[int]:
        """The ids of the post's forwards in ``channel`` (one per source message, by the
        forward origin); ``[]`` when the first one is not there."""
        since = row.created_at - timedelta(seconds=1)
        found: list[int] = []
        for mid in post.message_ids or [post.message_id]:
            fid = await user.find_message(channel, fwd_of=(post.chat_id, mid), since=since)
            if fid is None:
                break
            found.append(fid)
        return sorted(found)

    async def _repost(self, row: Publication, post: Post, chat: Chat, channel: int) -> _Sent:
        """Send the repost step by step, committing every message id as it comes back.

        A fresh start copies the media into staging (unless the row already has copies) and
        commits the copies and the corroboration count to render with before the first send.
        A retried row renders with the stored count, so the split is the one the earlier
        attempt used, and resumes after the committed ids; it first asks the account whether
        the step that failed last time reached the channel anyway (a timeout after delivery),
        and waits rather than sends when that check cannot run (``DeliveryUnverifiable``).
        The header's ``+N`` is caught up by the next edit once the row is ``sent``.
        """
        store = self._rt.store
        done = list(row.message_ids)
        staging_ids = list(row.staging_ids)
        verify = _may_have_delivered(row)
        # a retry renders with the count the earlier attempt used, so a message found by the
        # delivery check below and the parts sent after it belong to the same split
        corroboration = row.shown_corroboration if done or verify else post.corroboration
        if not done:
            if not staging_ids:
                staging_ids = await self._copy_to_staging(post)
            await store.set_publication_fields(
                row.id, staging_ids=staging_ids, shown_corroboration=corroboration
            )
        steps = self._plan(post, chat, staging_ids, corroboration)
        if not await self._run_steps(row, post, chat, channel, steps, done, verify=verify):
            staging_ids = []
            await store.set_publication_fields(row.id, staging_ids=[])
            steps = self._plan(post, chat, [], corroboration)
            await self._run_steps(row, post, chat, channel, steps, done, verify=False)
        return _Sent(done, staging_ids, "repost", corroboration)

    def _plan(
        self, post: Post, chat: Chat, staging_ids: list[int], corroboration: int
    ) -> list[_Step]:
        """The bot writes of a repost in send order (§9.4): the media with the whole post as
        its caption when it fits a single caption; else the media first and the text after
        it; text alone without media. The header sits on the last step."""
        parts: list[str]
        steps: list[_Step] = []
        if staging_ids:
            if len(staging_ids) == 1:
                rendered = render.realtime_post(
                    post, chat, corroboration, caption=True, media_attached=True, t=self._rt.t
                )
                if len(rendered.parts) == 1:
                    return [_Step("caption", rendered.parts[0], True, list(staging_ids))]
            steps.append(_Step("media", None, False, list(staging_ids)))
            parts = render.realtime_post(
                post, chat, corroboration, media_attached=True, t=self._rt.t
            ).parts
        else:
            parts = render.realtime_post(
                post, chat, corroboration, media_attached=False, t=self._rt.t
            ).parts
        last = len(parts) - 1
        steps += [_Step("text", part, i == last) for i, part in enumerate(parts)]
        return steps

    @staticmethod
    def _resume_index(steps: list[_Step], sent: int) -> int:
        """The first step not covered by ``sent`` committed message ids."""
        covered = 0
        for i, step in enumerate(steps):
            if covered >= sent:
                return i
            covered += step.count
        return len(steps)

    async def _run_steps(
        self,
        row: Publication,
        post: Post,
        chat: Chat,
        channel: int,
        steps: list[_Step],
        done: list[int],
        *,
        verify: bool,
    ) -> bool:
        """Send the steps not yet in ``done``, appending and committing each id.

        ``False`` when the media copy turned out unusable before anything went out: the
        caller then falls back to text + link.
        """
        bot = self._rt.bot
        assert bot is not None
        store = self._rt.store
        buttons = self._wrong_topic(post.id)
        for step in steps[self._resume_index(steps, len(done)) :]:
            if verify:
                verify = False
                found = await self._find_delivered(row, post, chat, channel, step, done)
                if found:
                    done.extend(found)
                    await store.set_publication_fields(row.id, message_ids=list(done))
                    continue
            if step.kind == "text":
                assert step.html is not None
                ids = [
                    await bot.send_text(channel, step.html, buttons=buttons if step.last else None)
                ]
            else:
                staging = self._rt.settings.publishing.staging_channel
                try:
                    ids = await bot.send_copy(
                        staging,
                        step.staging_ids,
                        channel,
                        caption_html=step.html,
                        buttons=buttons if step.kind == "caption" else None,
                    )
                except MediaUnavailable as exc:
                    if done:
                        raise
                    log.warning("post %d: media copy unusable (%s); text + link", post.id, exc)
                    return False
            done.extend(ids)
            await store.set_publication_fields(row.id, message_ids=list(done))
        return True

    async def _find_delivered(
        self,
        row: Publication,
        post: Post,
        chat: Chat,
        channel: int,
        step: _Step,
        done: list[int],
    ) -> list[int]:
        """The ids of ``step``'s messages when an earlier attempt delivered them after all."""
        return await self._checked(
            post, lambda user: self._lookup_step(user, row, post, chat, channel, step, done)
        )

    async def _checked(
        self, post: Post, lookup: Callable[[UserGateway], Awaitable[list[int]]]
    ) -> list[int]:
        """Run a before-retry delivery check through the account; it fails closed.

        A check that cannot run (no account session, Telegram unreachable, a flood wait) is
        no proof that nothing went out, so it raises ``DeliveryUnverifiable`` and the row waits
        with backoff instead of sending blindly; ``ChatGone``/``NotOwnedError`` pass through.
        """
        user = self._rt.user
        if user is None:
            raise DeliveryUnverifiable("no account to check the channel before a retry")
        try:
            found = await lookup(user)
        except (ChatGone, NotOwnedError, DeliveryUnverifiable):
            raise
        except CuratorError as exc:
            raise DeliveryUnverifiable(f"cannot check the channel before a retry: {exc}") from exc
        if found:
            log.info("post %d: message(s) %s had gone out before the retry", post.id, found)
        return found

    async def _lookup_step(
        self,
        user: UserGateway,
        row: Publication,
        post: Post,
        chat: Chat,
        channel: int,
        step: _Step,
        done: list[int],
    ) -> list[int]:
        """Ask the account for ``step``'s messages: the last step by the source link, an
        earlier text part by its first characters, media sent first by its staging copies
        (the same photo or document, every item of an album). Only messages newer than the
        committed ones count; ``[]`` when they are not there."""
        since = row.created_at - timedelta(seconds=1)
        after = max(done) if done else None
        if step.kind == "media":
            staging = self._rt.settings.publishing.staging_channel
            if not staging or not step.staging_ids:
                return []
            return await user.find_media(
                channel, copies_of=(staging, step.staging_ids), after_id=after, since=since
            )
        if step.html is None:
            return []
        plain = _plain(step.html)
        needle: str | None = None
        if step.last:
            needle = permalink(chat.id, chat.username, post.message_id) or (
                plain[-RECONCILE_MATCH_CHARS:].strip() or None
            )
        else:
            needle = plain[:RECONCILE_MATCH_CHARS].strip() or None
        if needle is None:
            return []
        found = await user.find_message(channel, contains=needle, since=since)
        if found is None or (after is not None and found <= after):
            return []
        return [found]

    async def _copy_to_staging(self, post: Post) -> list[int]:
        """The account's media copy into the staging channel; a failure means no media.

        A flood wait is not a failure: Telegram asks the account to slow down, so the post
        waits for it (``AccountFloodWait`` reaches ``_failed``, which retries when the wait is
        over) and no media copy is tried until then (§9.4 "Failures").
        """
        if post.media not in render.COPYABLE_MEDIA or post.noforwards:
            return []
        user = self._rt.user
        if user is None:
            log.info("post %d: no account session, media stays at the source", post.id)
            return []
        staging = await self._staging()
        if staging is None:
            return []
        now = self._rt.clock.now()
        until = self._flood_until.get(staging)
        if until is not None and now < until:
            raise AccountFloodWait(max(1, math.ceil((until - now).total_seconds())))
        if not self._channel_free(staging, now, need=len(post.message_ids), gap=GLOBAL_GAP):
            # only a forward that fell back to a repost gets here unchecked (§14 pacing)
            log.info("post %d: staging channel busy, media stays at the source", post.id)
            return []
        try:
            ids = await user.copy_media(post.chat_id, post.message_ids, staging)
        except (ForwardsRestricted, MediaUnavailable) as exc:
            log.info("post %d: media not copied (%s); text + link", post.id, exc)
            return []
        except FloodWait as exc:
            self._flood(staging, exc)
            log.warning("post %d: media copy hit a flood wait (%s); the post waits", post.id, exc)
            raise AccountFloodWait(exc.seconds) from exc
        except CuratorError as exc:
            log.warning("post %d: media copy failed (%s); text + link", post.id, exc)
            return []
        self._record_write(staging, self._rt.clock.now(), send=False, count=len(ids))
        return ids

    async def _staging(self) -> int | None:
        """The staging channel when it is configured and the bot can post there.

        The publisher never creates it (§14.2); while it is missing, media is skipped and
        one WARNING per start says so.
        """
        staging = self._rt.settings.publishing.staging_channel
        if staging and self._staging_verified == staging:
            return staging
        usable = False
        if staging and self._rt.bot is not None:
            try:
                usable = await self._rt.bot.can_post(staging)
            except CuratorError as exc:
                log.warning("staging channel %d cannot be checked: %s", staging, exc)
        if usable:
            self._staging_verified = staging
            return staging
        if not self._staging_warned:
            self._staging_warned = True
            why = "not configured" if not staging else "the bot cannot post into it"
            log.warning(
                "no usable staging channel (%s): media is skipped, posts go out as text + link",
                why,
            )
        return None

    async def _log_names(self, post: Post | int, *, topic: Topic | None = None) -> str:
        """``from "Chat title" for <topic key>``: the names a journal reader knows, next to
        the ids of the log line (SPEC: what was posted, what failed and why)."""
        store = self._rt.store
        try:
            found = await store.get_post(post) if isinstance(post, int) else post
            if found is None:
                return ""
            chat = await store.get_chat(found.chat_id)
            if topic is None and found.topic_id is not None:
                topic = await store.get_topic(found.topic_id)
        except Exception:  # noqa: BLE001 - a log line must never break a send
            return ""
        title = " ".join((chat.title if chat is not None else str(found.chat_id)).split())
        key = topic.key if topic is not None else f"topic {found.topic_id}"
        return f'from "{title}" for {key}'

    async def _mark_sent(self, row: Publication, post: Post, channel: int, sent: _Sent) -> None:
        store = self._rt.store
        now = self._rt.clock.now()
        async with store.begin() as conn:
            await store.set_publication_fields(
                row.id,
                state=PUB_SENT,
                style=sent.style,
                message_ids=sent.message_ids,
                staging_ids=sent.staging_ids,
                shown_corroboration=sent.shown,
                sent_at=now,
                next_attempt_at=None,
                last_error=None,
                conn=conn,
            )
            await store.set_post_fields(
                post.id, status=PostStatus.published, published_at=now, conn=conn
            )
        self._cannot_post_notified.discard(channel)
        self._unverified_warned.discard(post.id)
        log.info(
            "published post %d %s into channel %d as %d message(s), %s style",
            post.id,
            await self._log_names(post),
            channel,
            len(sent.message_ids),
            sent.style,
        )

    async def _failed(self, row: Publication, topic: Topic, exc: BaseException) -> None:
        """Backoff for transient errors; cannot-post handling with the 6 h give-up.

        What already went out stays committed in ``message_ids``; the retry resumes there.
        """
        now = self._rt.clock.now()
        error = f"{type(exc).__name__}: {exc}"[:200]
        if isinstance(exc, FloodWait):
            next_at = now + timedelta(seconds=exc.seconds)
        elif isinstance(exc, (BotCannotPost, ChatGone)):
            if now - row.created_at >= CANNOT_POST_GIVE_UP:
                log.warning("post %d: the bot could not post for 6 h; digest instead", row.post_id)
                await self._cancel(row, PostStatus.digest, error=error)
                return
            next_at = now + CANNOT_POST_RETRY
            await self._notify_cannot_post(topic)
        elif (
            isinstance(exc, (DeliveryUnverifiable, NotOwnedError))
            and now - row.created_at >= CANNOT_POST_GIVE_UP
        ):
            # the same limit as a ``sending`` row nobody can verify (§17 item 4)
            log.warning("post %d: delivery unverifiable for 6 h; digest instead", row.post_id)
            await self._cancel(row, PostStatus.digest, error="unverifiable after 6 h")
            return
        else:
            delay = min(BACKOFF_BASE_SECONDS * 2**row.attempts, BACKOFF_CAP_SECONDS)
            next_at = now + timedelta(seconds=delay)
        log.warning(
            "publishing post %d %s failed (%s); retry at %s",
            row.post_id,
            await self._log_names(row.post_id, topic=topic),
            error,
            next_at,
        )
        await self._rt.store.set_publication_fields(
            row.id, state=PUB_FAILED, next_attempt_at=next_at, last_error=error
        )

    async def _cancel(
        self, row: Publication, status: PostStatus | None, *, error: str | None = None
    ) -> None:
        """Row ``cancelled`` (and the post to ``status``). Messages a partial send left in
        the channel are stubbed first, so no fragment of the post stays visible there."""
        store = self._rt.store
        current = await store.get_publication(row.post_id) or row
        fields: dict[str, Any] = {
            "state": PUB_CANCELLED,
            "next_attempt_at": None,
            "last_error": error,
        }
        if (
            current.state not in (PUB_SENT, PUB_RETRACTED)
            and current.style == "repost"
            and current.channel_id is not None
            and current.message_ids
        ):
            log.info(
                "post %d: stubbing %d message(s) of an unfinished send",
                row.post_id,
                len(current.message_ids),
            )
            failed = await self._apply_stubs(
                current.channel_id, row.post_id, self._stub_plan(current, None, orphan=True)
            )
            if failed:
                fields["moved_from"] = list(current.moved_from or []) + [
                    {
                        "channel_id": current.channel_id,
                        "message_ids": list(current.message_ids),
                        "topic_id": current.topic_id,
                        PENDING_STUBS: failed,
                    }
                ]
                self._stubs_dirty = True
        async with store.begin() as conn:
            await store.set_publication_fields(row.id, conn=conn, **fields)
            if status is not None:
                await store.set_post_fields(row.post_id, status=status, conn=conn)
        log.info(
            "cancelled publication of post %d %s (%s)",
            row.post_id,
            await self._log_names(row.post_id),
            error or "no reason",
        )

    async def _drop_partial(self, row: Publication) -> None:
        """Stub the part of a post that went into a channel its topic no longer uses and
        reset the row, so the next tick starts the post afresh in the topic's channel."""
        assert row.channel_id is not None
        failed = await self._apply_stubs(
            row.channel_id, row.post_id, self._stub_plan(row, None, orphan=True)
        )
        entry: dict[str, Any] = {
            "channel_id": row.channel_id,
            "message_ids": list(row.message_ids),
            "topic_id": row.topic_id,
        }
        if failed:
            entry[PENDING_STUBS] = failed
            self._stubs_dirty = True
        await self._rt.store.set_publication_fields(
            row.id,
            moved_from=list(row.moved_from or []) + [entry],
            message_ids=[],
            shown_corroboration=0,
        )
        log.info("post %d: the topic changed channel mid-send; starting over", row.post_id)

    async def _notify_cannot_post(self, topic: Topic) -> None:
        """§11.4 item 4: once per channel until a write into that channel succeeds."""
        channel = topic.channel_id
        if channel is None or channel in self._cannot_post_notified:
            return
        self._cannot_post_notified.add(channel)
        chat = await self._rt.store.get_chat(channel)
        await self._rt.notifier.cannot_post(chat.title if chat is not None else topic.name)

    # --- +N edits ----------------------------------------------------------------------------

    async def _edit_corroboration(self) -> None:
        store = self._rt.store
        now = self._rt.clock.now()
        p, posts = schema.publications, schema.posts
        stmt = (
            sa.select(p)
            .join(posts, posts.c.id == p.c.post_id)
            .where(
                p.c.state == PUB_SENT,
                p.c.style.in_(("repost", "forward")),  # a forward's button message is the bot's
                posts.c.status == PostStatus.published.value,
                posts.c.corroboration > p.c.shown_corroboration,
                sa.or_(p.c.edited_at.is_(None), p.c.edited_at <= now - EDIT_COOLDOWN),
            )
            .order_by(p.c.sent_at, p.c.id)
        )
        for candidate in _publications(await store.execute(stmt)):
            channel = candidate.channel_id
            if channel is None or not candidate.message_ids or not self._channel_free(channel, now):
                continue
            async with self._lock(candidate.post_id):
                await self._edit_one(candidate, channel, now)

    async def _edit_one(self, candidate: Publication, channel: int, now: datetime) -> None:
        """The ``+N`` edit of one row, under its per-post lock and only while the row is
        still the sent message the SELECT saw: a ``move()`` that ran in between has stubbed
        it, and the full text must not come back over the stub."""
        store = self._rt.store
        row = await store.get_publication(candidate.post_id)
        if (
            row is None
            or row.id != candidate.id
            or row.state != PUB_SENT
            or row.style not in ("repost", "forward")
            or row.channel_id != channel
            or row.message_ids != candidate.message_ids
        ):
            return
        post = await store.get_post(row.post_id)
        if (
            post is None
            or post.status != PostStatus.published
            or post.corroboration <= row.shown_corroboration
        ):
            return
        html = self._last_message_html(row, post, await self._source_chat(post))
        if html is not None:
            bot = self._rt.bot
            assert bot is not None
            try:
                await bot.edit_text(
                    channel, row.message_ids[-1], html, buttons=self._wrong_topic(post.id)
                )
            except FloodWait as exc:
                # the attempt used the budget; nothing in this channel until the wait is over
                self._record_write(channel, now, send=False)
                self._flood(channel, exc)
                log.warning("post %d: +N edit hit a flood wait of %d s", post.id, exc.seconds)
                await store.set_publication_fields(
                    row.id, edited_at=now + timedelta(seconds=exc.seconds) - EDIT_COOLDOWN
                )
                return
            except PERMANENT_EDIT_ERRORS as exc:
                # the message is gone or no longer editable: stop trying for this count
                self._record_write(channel, now, send=False)
                log.warning("post %d: +N edit impossible, given up: %s", post.id, exc)
                await store.set_publication_fields(
                    row.id, shown_corroboration=post.corroboration, edited_at=now
                )
                return
            except CuratorError as exc:
                # transient: the 60 s edit cooldown is the backoff
                self._record_write(channel, now, send=False)
                log.warning("post %d: +N edit failed: %s", post.id, exc)
                await store.set_publication_fields(row.id, edited_at=now)
                return
            self._record_write(channel, now, send=False)
            log.info("post %d: header updated to +%d", post.id, post.corroboration)
        await store.set_publication_fields(
            row.id, shown_corroboration=post.corroboration, edited_at=now
        )

    def _last_message_html(self, row: Publication, post: Post, chat: Chat) -> str | None:
        """The last message re-rendered with the current ``+N``: a forward's button message,
        or a repost's header part.

        The first render reserved room for the longest ``+N`` (``render.MORE_RESERVE``), so the
        split never changes; ``None`` is only a safety net for a re-render that would not line
        up with what was sent. The count is then recorded as shown without an edit rather
        than overwriting a message with a shorter tail.
        """
        if row.style == "forward":
            return render.forward_companion(post, chat, post.corroboration, t=self._rt.t)
        attached = bool(row.staging_ids)
        caption = attached and len(row.message_ids) == 1
        rendered = render.realtime_post(
            post,
            chat,
            post.corroboration,
            caption=caption,
            media_attached=attached,
            t=self._rt.t,
        )
        media_messages = 0 if caption or not attached else len(row.staging_ids)
        if len(rendered.parts) != len(row.message_ids) - media_messages:
            log.warning("post %d: +N edit skipped, the text would no longer fit", post.id)
            return None
        return rendered.parts[-1]

    # --- reconcile ---------------------------------------------------------------------------

    async def _reconcile_sending(self, row: Publication) -> None:
        """Settle one ``sending`` row; the caller holds its per-post lock."""
        store = self._rt.store
        post = await store.get_post(row.post_id)
        if post is None:
            await self._cancel(row, None, error="post missing")
            return
        user = self._rt.user
        channel = row.channel_id
        if user is None or channel is None:
            await self._unverifiable(row, "no account to verify it with")
            return
        done = list(row.message_ids)
        shown = row.shown_corroboration
        try:
            chat = await self._source_chat(post)
            if row.style == "forward":
                if not done:
                    # the button message only ever follows committed forward ids, so a
                    # forward found here is all that went out; the next tick adds the rest
                    done = await self._lookup_forward(user, row, post, channel)
                    shown = post.corroboration
                elif permalink(chat.id, chat.username, post.message_id) is not None:
                    html = render.forward_companion(post, chat, shown, t=self._rt.t)
                    step = _Step("text", html, True)
                    found = await self._lookup_step(user, row, post, chat, channel, step, done)
                    if found:
                        sent = _Sent(done + found, [], "forward", shown)
                        await self._mark_sent(row, post, channel, sent)
                        log.info("post %d: found in %d after a restart", post.id, channel)
                        return
            else:
                steps = self._plan(post, chat, list(row.staging_ids), row.shown_corroboration)
                i = self._resume_index(steps, len(done))
                while i < len(steps):
                    found = await self._lookup_step(user, row, post, chat, channel, steps[i], done)
                    if not found:
                        break
                    done.extend(found)
                    i += 1
                if i >= len(steps):
                    # shown = the count the found messages were rendered with (stored on the
                    # row before the first send), so a later corroboration still edits it
                    sent = _Sent(done, list(row.staging_ids), "repost", row.shown_corroboration)
                    await self._mark_sent(row, post, channel, sent)
                    log.info("post %d: found in %d as %s after a restart", post.id, channel, done)
                    return
        except (ChatGone, NotOwnedError) as exc:
            log.warning("post %d: channel %d cannot be read (%s); digest", post.id, channel, exc)
            await self._cancel(row, PostStatus.digest, error=f"unverifiable: {exc}"[:200])
            return
        except CuratorError as exc:
            await self._unverifiable(row, f"cannot read the channel: {exc}")
            return
        await store.set_publication_fields(
            row.id, state=PUB_PENDING, message_ids=done, shown_corroboration=shown
        )
        self._unverified_warned.discard(post.id)
        log.info(
            "post %d: %d of its message(s) found in %d after a restart, the rest re-queued",
            post.id,
            len(done),
            channel,
        )

    async def _unverifiable(self, row: Publication, why: str) -> None:
        """A ``sending`` row the account cannot check: left alone (sending blindly could
        post twice) and retried by the ticks, until 6 h after it was created; then it is
        cancelled and the post goes to the digest, like a row the bot cannot post (§9.4)."""
        if self._rt.clock.now() - row.created_at >= CANNOT_POST_GIVE_UP:
            log.warning("post %d: still unverifiable after 6 h (%s); digest", row.post_id, why)
            await self._cancel(row, PostStatus.digest, error="unverifiable after 6 h")
            self._unverified_warned.discard(row.post_id)
            return
        if row.post_id in self._unverified_warned:
            log.debug("post %d: still in 'sending' (%s)", row.post_id, why)
            return
        self._unverified_warned.add(row.post_id)
        log.warning("post %d: left in 'sending' (%s); retried every minute", row.post_id, why)

    async def _retry_orphaned(self) -> None:
        """Retry the lookup for ``sending`` rows no send in this process holds — the ones a
        reconcile could not settle (no account yet, Telegram unreachable at start)."""
        now = self._rt.clock.now()
        if self._next_orphan_check is not None and now < self._next_orphan_check:
            return
        self._next_orphan_check = now + ORPHAN_CHECK_EVERY
        store = self._rt.store
        cutoff = now - timedelta(minutes=self._rt.settings.sorting.hold_minutes)
        for stale in await store.publications_in_state(PUB_SENDING):
            lock = self._locks.get(stale.post_id)
            if lock is not None and lock.locked():
                continue  # a move is working on it
            async with self._lock(stale.post_id):
                row = await store.get_publication(stale.post_id)
                if row is None or row.id != stale.id or row.state != PUB_SENDING:
                    continue
                await self._reconcile_sending(row)
                row = await store.get_publication(stale.post_id)
                if row is not None and row.state == PUB_PENDING:
                    # the restart's reconcile finishing late: the stale rule still applies
                    await self._cancel_if_stale(row, cutoff)

    async def _cancel_stale(self) -> None:
        """§14.6: a queued post older than ``hold_minutes`` goes to the digest, not out late."""
        store = self._rt.store
        cutoff = self._rt.clock.now() - timedelta(minutes=self._rt.settings.sorting.hold_minutes)
        for stale in await store.publications_in_state([PUB_PENDING, PUB_FAILED]):
            async with self._lock(stale.post_id):
                row = await store.get_publication(stale.post_id)
                if row is None or row.state not in (PUB_PENDING, PUB_FAILED):
                    continue
                await self._cancel_if_stale(row, cutoff)

    async def _cancel_if_stale(self, row: Publication, cutoff: datetime) -> None:
        post = await self._rt.store.get_post(row.post_id)
        if post is None or post.status != PostStatus.queued or post.posted_at >= cutoff:
            return
        if row.style == "forward" and row.message_ids:
            return  # the forward is already out: its button message follows, not the digest
        await self._cancel(row, PostStatus.digest, error="stale at restart")

    # --- moves and stubs ---------------------------------------------------------------------

    def _stub_plan(
        self, row: Publication, new_topic: Topic | None, *, orphan: bool = False
    ) -> list[list[Any]]:
        """``[message_id, stub_html]`` for every message of the row.

        Media that went first without a caption gets the short ``↪ moved`` caption; a media
        message that carried the caption, and every text message, gets the full stub. The
        media itself stays (the bot can neither remove it by an edit nor delete it, §14.1).
        ``orphan`` is the part of an unfinished send that is cancelled: every message gets the
        short stub. A row that is not ``sent`` with staging copies is such a partial send, so
        all of its leading media messages went first.
        """
        if orphan:
            full = short = render.stub("moved", None, t=self._rt.t)
        elif new_topic is None:
            full = short = render.stub("not_for_me", None, t=self._rt.t)
        else:
            full = render.stub("moved", new_topic.name, t=self._rt.t)
            short = render.stub("moved", None, t=self._rt.t)
        n_ids, n_staged = len(row.message_ids), len(row.staging_ids)
        media_first = (
            min(n_staged, n_ids) if n_staged and (n_ids > n_staged or row.state != PUB_SENT) else 0
        )
        return [[mid, short if i < media_first else full] for i, mid in enumerate(row.message_ids)]

    async def _apply_stubs(
        self, channel: int, post_id: int, stubs: list[list[Any]]
    ) -> list[list[Any]]:
        """Edit each message into its stub, buttons removed; the stubs still to apply.

        A message that is gone, not the bot's, or in a channel the bot lost counts as done:
        nothing of the post is visible there any more, or nothing ever will be editable. A
        transient error (Telegram unreachable, a flood wait) keeps the stub for a retry.
        """
        bot = self._rt.bot
        if bot is None:
            return [list(s) for s in stubs]
        remaining: list[list[Any]] = []
        for i, (mid, stub_html) in enumerate(stubs):
            try:
                await bot.edit_text(channel, mid, stub_html, buttons=None)
            except FloodWait as exc:
                self._record_write(channel, self._rt.clock.now(), send=False)
                self._flood(channel, exc)
                log.warning("post %d: stubbing hit a flood wait of %d s", post_id, exc.seconds)
                remaining.extend([m, s] for m, s in stubs[i:])
                break
            except PERMANENT_EDIT_ERRORS as exc:
                log.warning("post %d: message %d cannot be stubbed: %s", post_id, mid, exc)
            except CuratorError as exc:
                log.warning("post %d: message %d not stubbed yet: %s", post_id, mid, exc)
                remaining.append([mid, stub_html])
            self._record_write(channel, self._rt.clock.now(), send=False)
        return remaining

    async def _retry_stubs(self) -> None:
        """Apply the stubs a move or a cancel could not, within each channel's budget."""
        if not self._stubs_dirty:
            return
        store = self._rt.store
        p = schema.publications
        stmt = (
            sa.select(p)
            .where(sa.cast(p.c.moved_from, sa.Text).like(f'%"{PENDING_STUBS}"%'))
            .order_by(p.c.id)
        )
        left = False
        for candidate in _publications(await store.execute(stmt)):
            async with self._lock(candidate.post_id):
                row = await store.get_publication(candidate.post_id)
                if row is None or not _has_pending_stubs(row):
                    continue
                now = self._rt.clock.now()
                entries = [dict(e) for e in row.moved_from or []]
                changed = False
                for entry in entries:
                    stubs = entry.get(PENDING_STUBS)
                    if not stubs:
                        continue
                    channel = entry["channel_id"]
                    if not self._channel_free(channel, now):
                        left = True
                        continue
                    remaining = await self._apply_stubs(channel, row.post_id, stubs)
                    changed = True
                    if remaining:
                        entry[PENDING_STUBS] = remaining
                        left = True
                    else:
                        del entry[PENDING_STUBS]
                        log.info("post %d: pending stubs in %d applied", row.post_id, channel)
                if changed:
                    await store.set_publication_fields(row.id, moved_from=entries)
        self._stubs_dirty = left

    # --- pacing ------------------------------------------------------------------------------

    def _channel_free(
        self, channel: int, now: datetime, *, need: int = 1, gap: timedelta | None = None
    ) -> bool:
        """May ``need`` more messages go into ``channel`` now? No flood wait pending, room
        for them within the minute's budget and ``gap`` (default ``min_gap_seconds``) since
        the last write."""
        until = self._flood_until.get(channel)
        if until is not None:
            if now < until:
                return False
            del self._flood_until[channel]
        writes = self._channel_writes[channel]
        while writes and now - writes[0] >= timedelta(minutes=1):
            writes.popleft()
        need = min(max(need, 1), MAX_WRITES_PER_MINUTE)
        if len(writes) + need > MAX_WRITES_PER_MINUTE:
            return False
        if gap is None:
            gap = timedelta(seconds=self._rt.settings.publishing.min_gap_seconds)
        return not writes or now - writes[-1] >= gap

    def _global_free(self, now: datetime) -> bool:
        return self._last_send is None or now - self._last_send >= GLOBAL_GAP

    def _record_write(self, channel: int, now: datetime, *, send: bool, count: int = 1) -> None:
        """``count`` messages written into ``channel`` (an album counts once per item)."""
        self._channel_writes[channel].extend([now] * max(count, 1))
        if send:
            self._last_send = now

    def _flood(self, channel: int, exc: FloodWait) -> None:
        """No write into ``channel`` until Telegram's wait is over."""
        until = self._rt.clock.now() + timedelta(seconds=exc.seconds)
        self._flood_until[channel] = max(until, self._flood_until.get(channel, until))

    # --- helpers -----------------------------------------------------------------------------

    def _lock(self, post_id: int) -> asyncio.Lock:
        lock = self._locks.get(post_id)
        if lock is None:
            lock = self._locks[post_id] = asyncio.Lock()
        return lock

    async def _source_chat(self, post: Post) -> Chat:
        chat = await self._rt.store.get_chat(post.chat_id)
        if chat is not None:
            return chat
        # A post whose chat row is gone still renders: the title is then the id.
        return Chat(
            id=post.chat_id,
            kind="channel",
            title=str(post.chat_id),
            noforwards=post.noforwards,
            first_seen_at=self._rt.clock.now(),
        )

    def _style(self, post: Post, chat: Chat) -> PublishStyle:
        """``publishing.style``, except that protected sources and group threads are always
        reposted: a conversation unit is one post linking to its first message (§9.1), which
        forwarding its members one by one would not be."""
        if post.noforwards or chat.noforwards or post.kind == "unit":
            return "repost"
        return self._rt.settings.publishing.style

    def _wrong_topic(self, post_id: int) -> Buttons:
        t = self._rt.t
        text = t(WRONG_TOPIC_KEY) if t.has(WRONG_TOPIC_KEY) else WRONG_TOPIC_DEFAULT
        return [[Button(text, data=f"wt:{post_id}")]]
