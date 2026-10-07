"""Topics: create, edit, merge, remove, examples, and the sync with the settings file
(DESIGN §8 ``TopicsService``, §4, §9.7, §14.13).

The settings file is the source of truth for *which* topics exist (identity = ``key``); the
``topics`` table mirrors it so the pipeline never parses TOML. Everything that touches
Telegram on behalf of a topic — resolving and adopting an output channel, creating one,
reading an example channel, the staging channel — lives here so that the §1 guard rules
(``register_owned`` only for channels the account owns) have exactly one home.
"""

from __future__ import annotations

import asyncio
import logging
import math
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from tomlkit import TOMLDocument

from tg_curator.clock import local_date, scheduled_moment
from tg_curator.config import TopicSettings
from tg_curator.db import schema
from tg_curator.domain import (
    DIGEST_CANCELLED,
    DIGEST_PENDING,
    DIGEST_SENDING,
    KV,
    PUB_CANCELLED,
    PUB_FAILED,
    PUB_PENDING,
    PUB_SENDING,
    ChatRole,
    Example,
    ExampleKind,
    PostStatus,
    Topic,
    TopicOrigin,
    TopicSyncResult,
)
from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    ConfigError,
    CuratorError,
    FloodWait,
    NotAllowed,
    TopicExists,
)
from tg_curator.ml import categories
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED, EVENT_TOPICS_CHANGED, Runtime
from tg_curator.telegram.gateway import ChatInfo, UserGateway
from tg_curator.textutil import html_escape
from tg_curator.topics.learning import embedding_bytes, retrain_classifier

log = logging.getLogger(__name__)

STAGING_TITLE = "tg-curator media"
STAGING_ABOUT = (
    "private staging channel used by tg-curator to hand media to the bot; safe to ignore"
)

# Channel creation pacing (§8): Telegram flood-limits channels.createChannel and the
# community figure is about 50 a day, so the curator stays far below it.
CREATE_INTERVAL = timedelta(seconds=60)
CREATES_PER_DAY = 5

# Example channels: how far back and how many messages to read for up to 50 examples.
EXAMPLE_CHANNEL_DAYS = 30
EXAMPLE_CHANNEL_READ_LIMIT = 200
EXAMPLE_CHANNEL_MAX = 50

# Statuses that §9.7 moves away from a deactivated topic; every other status keeps its
# topic_id for history.
_MOVABLE = (PostStatus.held, PostStatus.queued, PostStatus.digest, PostStatus.tracked)

# Cyrillic -> ASCII for slugs. Russian plus the Uzbek Cyrillic letters (ғ қ ҳ ў); keys are
# only identifiers, so a readable approximation is all that is needed.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "ғ": "g", "қ": "q", "ҳ": "h", "ў": "o", "ї": "yi", "є": "ye", "і": "i",
    "ґ": "g",
}  # fmt: skip
_KEY_MAX = 32


def slugify(name: str) -> str:
    """A valid topic key: ``"ML & AI"`` -> ``"ml-ai"``, ``"Футбол"`` -> ``"futbol"``."""
    lowered = unicodedata.normalize("NFKC", name).casefold()
    ascii_chars: list[str] = []
    for ch in lowered:
        if ch in _TRANSLIT:
            ascii_chars.append(_TRANSLIT[ch])
            continue
        decomposed = unicodedata.normalize("NFKD", ch)
        ascii_chars.append("".join(c for c in decomposed if ord(c) < 128))
    out = ""
    for ch in "".join(ascii_chars):
        out += ch if ch.isalnum() and ch.isascii() else "-"
    parts = [p for p in out.split("-") if p]
    slug = "-".join(parts)[:_KEY_MAX].rstrip("-")
    return slug or "topic"


def _unique_key(base: str, taken: set[str]) -> str:
    """``base``, else ``base-2``, ``base-3`` … within the 32-character key limit."""
    if base not in taken:
        return base
    n = 2
    while True:
        suffix = f"-{n}"
        key = base[: _KEY_MAX - len(suffix)].rstrip("-") + suffix
        if key not in taken:
            return key
        n += 1


def _blank_to_none(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _category(value: str | None) -> str | None:
    """A blank category is none; anything else must be a built-in key (``ConfigError`` naming
    close matches otherwise), so a topic never silently sorts nothing."""
    value = _blank_to_none(value)
    return categories.require_key(value) if value is not None else None


def _strictness(value: float | None) -> float | None:
    """Settings ``0.0`` means "use sorting.confidence", stored as NULL (wave-0 gap 4)."""
    return None if not value else float(value)


class TopicsService:
    """The ``TopicsService`` of §8."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._create_log: dict[str, Any] | None = None
        self._syncing = False

    # --- sync with the settings file ---------------------------------------------------------

    async def sync_from_settings(self) -> TopicSyncResult:
        """Mirror ``[[topics]]`` and ``[[sources]]`` into the database (§4, §8).

        Every resolved ``@username``/link is rewritten to its numeric id in ONE settings write
        at the end, so the ``settings_changed`` that write raises reaches a sync that has
        nothing left to rewrite; a sync re-entered from its own write returns at once.
        """
        if self._syncing:
            log.debug("topics: sync re-entered from its own settings write; skipped")
            return TopicSyncResult(total=0, resolved=0, unresolved=[])
        self._syncing = True
        try:
            return await self._sync()
        finally:
            self._syncing = False

    async def _sync(self) -> TopicSyncResult:
        store = self._rt.store
        settings = self._rt.settings
        await self._load_create_log()
        unresolved: list[str] = []
        resolved = 0
        channel_rewrites: dict[str, int] = {}
        changed_set = False
        examples_stored = 0
        stored = {t.key: t for t in await store.list_topics(active=None)}

        # Removed keys first: a channel a removed topic used is free for the topics that remain
        # (one channel per topic, see ``_check_channel_free``).
        file_keys = {t.key for t in settings.topics}
        deactivated = 0
        for topic in stored.values():
            if topic.active and topic.key not in file_keys:
                await self._deactivate(topic, None)
                deactivated += 1
                changed_set = True
                log.info("topic %s removed from the settings file: deactivated", topic.key)

        for entry in settings.topics:
            existing = stored.get(entry.key)
            if existing is None or not existing.active:
                changed_set = True
            row = await store.upsert_topic(self._row_from_settings(entry, existing))
            if entry.category and entry.category not in categories.CATEGORIES:
                # Not a reason to reject the whole file: the topic still sorts by its
                # examples, and the owner is told what to fix (``curator topics``, /reload).
                unresolved.append(
                    f"topic {entry.key}: {categories.unknown_key_message(entry.category)}"
                )
            if entry.channel == 0:
                if existing is not None and existing.active and existing.channel_id is not None:
                    unresolved.append(
                        f"topic {entry.key}: a channel cannot be removed from a topic: "
                        "remove the topic, or /pause to stop posting"
                    )
            else:
                try:
                    linked = await self._link(row, entry.channel, write_settings=False)
                    resolved += 1
                    if isinstance(entry.channel, str) and linked.channel_id is not None:
                        channel_rewrites[entry.key] = linked.channel_id
                except CuratorError as exc:
                    unresolved.append(f"topic {entry.key}: {_reason(exc)}")
            # ``row.example_channel`` is the ref whose posts were actually read (see
            # ``_row_from_settings``): it only takes the file's value once ingestion succeeds,
            # so a failed read (not a member yet, an outage) is retried on the next sync, and
            # a re-added topic — whose examples §9.7 deleted — reads its channel again.
            new_example_channel = _blank_to_none(entry.example_channel)
            if new_example_channel is None:
                if row.example_channel is not None:
                    await store.set_topic_fields(row.id, example_channel=None)
            elif new_example_channel != row.example_channel:
                try:
                    examples_stored += await self._ingest_example_channel(row, new_example_channel)
                except CuratorError as exc:
                    unresolved.append(f"topic {entry.key}: example channel: {_reason(exc)}")
                else:
                    await store.set_topic_fields(row.id, example_channel=new_example_channel)

        source_rewrites = await self._sync_sources(unresolved)
        if channel_rewrites or source_rewrites:
            await self._rt.settings_file.update(
                lambda doc: _rewrite_refs(doc, channel_rewrites, source_rewrites)
            )

        await retrain_classifier(self._rt)
        if changed_set:
            await self._rt.events.emit(EVENT_TOPICS_CHANGED)
        if deactivated:
            await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="topics")
        if examples_stored:
            await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="examples")
        total = len(settings.topics)
        log.info(
            "topics: %d in the settings file, %d channels resolved, %d unresolved",
            total,
            resolved,
            len(unresolved),
        )
        return TopicSyncResult(total=total, resolved=resolved, unresolved=unresolved)

    def _row_from_settings(self, entry: TopicSettings, existing: Topic | None) -> Topic:
        """The table row for a ``[[topics]]`` entry (the settings <-> DB mapping of gap 4).

        The stored channel is kept until ``link_channel`` replaces it: a channel is never
        removed through the file (§9.7 invariant), and an unresolvable new value must not
        wipe a working one. ``example_channel`` likewise keeps the stored value (the channel
        whose posts were read) and ``_sync`` writes the file's value only after reading it
        succeeded; a deactivated topic starts from none, since §9.7 deleted its examples.
        """
        keep = existing is not None and existing.active
        return Topic(
            id=existing.id if existing else 0,
            key=entry.key,
            name=entry.name,
            channel_id=existing.channel_id if keep and existing else None,
            category=entry.category,
            description=entry.description,
            example_channel=existing.example_channel if keep and existing else None,
            strictness=_strictness(entry.strictness),
            active=True,
            origin=existing.origin if existing else "user",
            created_at=existing.created_at if existing else self._rt.clock.now(),
        )

    async def _sync_sources(self, unresolved: list[str]) -> dict[str, int]:
        """Mirror ``[[sources]]`` into ``chats.trust``; returns ``{written ref: numeric id}``."""
        store = self._rt.store
        rewrites: dict[str, int] = {}
        listed: set[int] = set()
        for source in self._rt.settings.sources:
            if isinstance(source.chat, int):
                listed.add(source.chat)
            try:
                info = await self._user().resolve_chat(source.chat)
            except CuratorError as exc:
                unresolved.append(f"source {source.chat}: {_reason(exc)}")
                continue
            listed.add(info.id)
            if isinstance(source.chat, str):
                rewrites[source.chat] = info.id
            async with store.begin() as conn:
                # A known chat only gets its trust: ``upsert_chat`` would mark a chat the
                # account left active again (resolve_chat still finds a public channel after
                # leaving), and only intake.sync_chats / a live message re-activates a chat.
                if await store.get_chat(info.id, conn=conn) is None:
                    await store.upsert_chat(info, conn=conn)
                await store.set_chat_fields(info.id, trust=float(source.trust), conn=conn)
        reset = (
            sa.update(schema.chats)
            .where(schema.chats.c.trust.is_not(None))
            .where(schema.chats.c.id.not_in(list(listed)) if listed else sa.true())
            .values(trust=None)
        )
        await store.execute(reset)
        return rewrites

    # --- channels ----------------------------------------------------------------------------

    async def link_channel(self, key: str, ref: str | int) -> Topic:
        """Resolve, check, adopt and record an output channel for ``key`` (§8)."""
        topic = await self._active_topic(key)
        return await self._link(topic, ref, write_settings=True)

    async def _link(self, topic: Topic, ref: str | int, *, write_settings: bool) -> Topic:
        info = await self._resolve_output_channel(ref, topic_id=topic.id)
        store = self._rt.store
        await store.set_topic_fields(topic.id, channel_id=info.id)
        if write_settings:
            entry = self._rt.settings.topic(topic.key)
            if entry is not None and entry.channel != info.id:
                await self._rt.settings_file.upsert_topic(topic.key, channel=info.id)
        log.info("topic %s posts into channel %d (%s)", topic.key, info.id, info.title)
        return await self._refresh(topic.id)

    async def _resolve_output_channel(
        self, ref: str | int, *, topic_id: int | None = None
    ) -> ChatInfo:
        """Steps 1–5 of ``link_channel``, before anything about the topic is written.

        ``topic_id`` is the topic being linked (``None`` for one being created), so linking a
        topic to the channel it already has is not refused as "already used".
        """
        try:
            info = await self._user().resolve_chat(ref)
        except ChatGone:
            raise ConfigError(
                f"topic channel {ref} cannot be found: check the link or that your account is in it"
            ) from None
        if info.kind != "channel" or not (info.is_creator or info.is_admin):
            raise ConfigError(
                f"channel {info.title} is not yours: the account must be its creator or an admin"
            )
        await self._check_channel_free(info, topic_id)
        await self._adopt_channel(info, role="output")
        return info

    async def _check_channel_free(self, info: ChatInfo, topic_id: int | None) -> None:
        """One channel per topic, and never the staging channel (SPEC: a topic channel gets
        only its own posts and one digest a day). Checked before the role is overwritten."""
        staging = self._rt.settings.publishing.staging_channel
        chat = await self._rt.store.get_chat(info.id)
        if (staging and info.id == staging) or (chat is not None and chat.role == "staging"):
            raise ConfigError(
                f"channel {info.title} is the curator's private media channel; pick or create "
                "another one"
            )
        for other in await self._rt.store.list_topics(active=True):
            if other.channel_id == info.id and other.id != topic_id:
                raise ConfigError(
                    f"channel {info.title} is already the channel of topic {other.name}"
                )

    async def _adopt_channel(self, info: ChatInfo, *, role: ChatRole) -> None:
        """Register a channel the account owns and make sure the bot can post into it."""
        await self._rt.store.upsert_chat(info, role=role)
        self._user().register_owned(info.id)
        await self._ensure_bot_can_post(info)

    async def _ensure_bot_can_post(self, info: ChatInfo, *, promote: bool = False) -> None:
        """``add_bot_admin`` when the bot cannot post into an owned channel (``promote``: try
        it without asking first, for a channel just created); ``BotCannotPost`` if it still
        cannot. Safe to call again: a promotion that failed is retried, never replaced."""
        bot = self._rt.bot
        if bot is None:
            log.warning("no bot wired: cannot check that it can post into %s", info.title)
            return
        if not promote and await bot.can_post(info.id):
            return
        bot_username = self._bot_username()
        try:
            await self._user().add_bot_admin(info.id, bot_username)
        except CuratorError as exc:
            log.warning("could not add @%s as admin of %s: %s", bot_username, info.title, exc)
        if not await bot.can_post(info.id):
            raise BotCannotPost(
                f"the bot cannot post into {info.title}: add @{bot_username} as an admin with "
                "Post Messages"
            )

    def channel_wait_minutes(self) -> int | None:
        """Minutes until ``create_channel`` is allowed again; ``None`` = now (§8 pacing)."""
        log_ = self._create_log or {}
        now = self._rt.clock.now()
        tz = self._rt.settings.general.timezone
        created = [datetime.fromisoformat(s) for s in log_.get("created", ())]
        until: datetime | None = None
        if created and max(created) + CREATE_INTERVAL > now:
            until = max(created) + CREATE_INTERVAL
        today = local_date(now, tz)
        if sum(1 for c in created if local_date(c, tz) == today) >= CREATES_PER_DAY:
            midnight = scheduled_moment(today + timedelta(days=1), 0, 0, tz)
            until = midnight if until is None else max(until, midnight)
        wait_until = log_.get("wait_until")
        if wait_until:
            flood = datetime.fromisoformat(wait_until)
            if flood > now:
                until = flood if until is None else max(until, flood)
        if until is None:
            return None
        return max(1, math.ceil((until - now).total_seconds() / 60))

    async def _create_channel(self, title: str, about: str) -> ChatInfo:
        """``user.create_channel`` recorded in ``kv topics.create_log`` (a FloodWait too)."""
        log_ = await self._load_create_log()
        now = self._rt.clock.now()
        try:
            info = await self._user().create_channel(title, about=about)
        except FloodWait as exc:
            log_["wait_until"] = (now + timedelta(seconds=exc.seconds)).isoformat()
            await self._save_create_log()
            raise
        keep_after = now - timedelta(days=2)
        created = [s for s in log_.get("created", ()) if datetime.fromisoformat(s) > keep_after]
        created.append(now.isoformat())
        log_["created"] = created
        log_["wait_until"] = None
        await self._save_create_log()
        return info

    async def _load_create_log(self) -> dict[str, Any]:
        """The pacing record, read from ``kv`` once per process and kept in memory so the
        synchronous ``channel_wait_minutes()`` can answer without a query."""
        if self._create_log is None:
            stored = await self._rt.store.kv_get(KV.TOPICS_CREATE_LOG)
            self._create_log = dict(stored) if stored else {"created": [], "wait_until": None}
        return self._create_log

    async def _save_create_log(self) -> None:
        await self._rt.store.kv_set(KV.TOPICS_CREATE_LOG, self._create_log)

    async def ensure_staging_channel(self) -> int:
        """The private channel the account copies media into for the bot (§8, §14.2).

        A new channel is recorded (role ``staging``, ``publishing.staging_channel``) before the
        bot is made its admin, so a failed promotion is retried against the same channel on
        the next call instead of leaving an unrecorded channel and creating another one.
        Creation follows the same pacing as topic channels (``FloodWait`` while it says wait).
        """
        store = self._rt.store
        current = self._rt.settings.publishing.staging_channel
        if current:
            try:
                info = await self._user().resolve_chat(current)
            except ChatGone:
                log.warning("staging channel %d is gone: creating a new one", current)
            else:
                if info.kind != "channel" or not (info.is_creator or info.is_admin):
                    # §1: only a channel the account owns may be registered as owned.
                    raise ConfigError(
                        f"publishing.staging_channel = {current} is not a channel your account "
                        "owns: set it to 0 and the curator creates one"
                    )
                for topic in await store.list_topics(active=True):
                    if topic.channel_id == info.id:
                        raise ConfigError(
                            f"publishing.staging_channel = {current} is the channel of topic "
                            f"{topic.name}: set it to 0 and the curator creates one"
                        )
                await store.upsert_chat(info, role="staging")
                self._user().register_owned(info.id)
                await self._ensure_bot_can_post(info)
                return info.id
        if self._rt.bot is not None:
            self._bot_username()  # no bot username yet: fail before creating anything
        await self._load_create_log()
        wait = self.channel_wait_minutes()
        if wait is not None:
            log.info("staging channel not created: Telegram pacing asks to wait %d min", wait)
            raise FloodWait(wait * 60)
        info = await self._create_channel(STAGING_TITLE, STAGING_ABOUT)
        self._user().register_owned(info.id)
        await store.upsert_chat(info, role="staging")
        await self._rt.settings_file.set_value("publishing.staging_channel", info.id)
        log.info("staging channel created: %d", info.id)
        await self._ensure_bot_can_post(info, promote=True)
        return info.id

    # --- create / update / remove / merge ----------------------------------------------------

    async def create(
        self,
        name: str,
        *,
        category: str | None = None,
        description: str | None = None,
        channel: str | int | None = None,
        create_channel: bool = False,
        example_channel: str | None = None,
        origin: TopicOrigin = "user",
    ) -> Topic:
        """Create a topic: slug key, optional channel (existing or created), examples (§8).

        Everything that can fail against Telegram (an existing channel, an example channel) is
        resolved BEFORE the first write, so a refusal leaves nothing half-created. Channel
        creation is the exception: a FloodWait or the daily cap saves the topic with channel 0
        and ``channel_wait_minutes()`` says how long to wait.
        """
        store = self._rt.store
        name = name.strip()
        if not name:
            raise ConfigError("a topic needs a name")
        await self._ensure_name_free(name, except_key=None)
        taken = {t.key for t in await store.list_topics(active=None)}
        taken.update(t.key for t in self._rt.settings.topics)
        key = _unique_key(slugify(name), taken)
        category = _category(category)

        channel_info: ChatInfo | None = None
        if channel is not None and channel != 0 and channel != "":
            channel_info = await self._resolve_output_channel(channel)
        example_ref = _blank_to_none(example_channel)
        example_texts = await self._read_example_channel(example_ref) if example_ref else []
        deferred = False
        if channel_info is None and create_channel:
            channel_info = await self.create_topic_channel(name)
            deferred = channel_info is None

        topic = await store.upsert_topic(
            Topic(
                id=0,
                key=key,
                name=name,
                channel_id=channel_info.id if channel_info else None,
                category=category,
                description=_blank_to_none(description),
                example_channel=example_ref,
                strictness=None,
                active=True,
                origin=origin,
                created_at=self._rt.clock.now(),
            )
        )
        if example_texts:
            await self._store_examples(topic, example_texts, kind="channel")
        await self._rt.settings_file.upsert_topic(
            key,
            name=name,
            channel=topic.channel_id or 0,
            category=topic.category or "",
            description=topic.description or "",
            example_channel=example_ref or "",
            strictness=0.0,
        )
        if deferred:
            await self.want_channel(key)
        log.info("topic %s (%s) created, channel %s", key, name, topic.channel_id or "none")
        await self._after_topic_set_changed()
        return topic

    async def create_topic_channel(self, name: str) -> ChatInfo | None:
        """A private channel named after a topic, within the pacing rules; ``None`` = wait.

        Public because the bot creates channels for topics that already exist (the
        [Create channel] button after a FloodWait, §11.2); the caller links it with
        ``link_channel``. Keeping the creation here keeps the pacing log in one place.
        """
        # The stored log, not only this process's memory: a CLI run never syncs first.
        await self._load_create_log()
        wait = self.channel_wait_minutes()
        if wait is not None:
            log.info("channel for %r not created: Telegram pacing asks to wait %d min", name, wait)
            return None
        try:
            info = await self._create_channel(name, "")
        except FloodWait as exc:
            log.warning("channel for %r not created: FloodWait %d s", name, exc.seconds)
            return None
        try:
            await self._adopt_channel(info, role="output")
        except BotCannotPost as exc:
            # The channel exists and is the topic's; the publisher reports the missing rights
            # once per channel (§11.4) until the owner adds the bot by hand.
            log.warning("%s", exc)
        return info

    async def want_channel(self, key: str) -> None:
        """Remember that ``key`` still wants a created channel: Telegram's pacing (or the
        curator's own) deferred it, and ``tick`` creates it once the wait is over (SPEC:
        "the curator waits as long as it is told and continues"). Kept in
        ``kv topics.create_log`` next to the pacing record it depends on."""
        log_ = await self._load_create_log()
        wanted = list(log_.get("wanted") or [])
        if key not in wanted:
            wanted.append(key)
            log_["wanted"] = wanted
            await self._save_create_log()
        log.info("topic %s: its channel is created once Telegram's wait is over", key)

    async def wanted_channels(self) -> list[str]:
        """The topic keys whose channel creation is deferred (see ``want_channel``)."""
        return list((await self._load_create_log()).get("wanted") or [])

    async def tick(self) -> None:
        """Create one deferred topic channel when the pacing allows it (a periodic loop).

        One per run keeps to the 60 s spacing; a new FloodWait keeps the mark and the next
        run waits again. A key that was removed, merged or got a channel otherwise (the
        [Create channel] button, a link) is dropped. The owner is told once it is done.
        """
        log_ = await self._load_create_log()
        wanted = list(log_.get("wanted") or [])
        if not wanted or self._rt.user is None or self.channel_wait_minutes() is not None:
            return
        store = self._rt.store
        remaining = list(wanted)
        for key in wanted:
            topic = await store.get_topic_by_key(key)
            if topic is None or not topic.active or topic.channel_id is not None:
                remaining.remove(key)
                continue
            try:
                info = await self.create_topic_channel(topic.name)
            except CuratorError as exc:
                log.warning("topic %s: channel not created: %s; retrying later", key, exc)
                break
            if info is None:
                break  # Telegram asked to wait again: the mark stays
            remaining.remove(key)
            linked = await self._attach_created(topic, info)
            await self._announce_created(linked, info)
            break
        if remaining != wanted:
            log_["wanted"] = remaining
            await self._save_create_log()

    async def _attach_created(self, topic: Topic, info: ChatInfo) -> Topic:
        """Record a channel ``create_topic_channel`` made (and adopted) on ``topic``."""
        await self._rt.store.set_topic_fields(topic.id, channel_id=info.id)
        entry = self._rt.settings.topic(topic.key)
        if entry is not None and entry.channel != info.id:
            await self._rt.settings_file.upsert_topic(topic.key, channel=info.id)
        log.info("topic %s posts into the new channel %d (%s)", topic.key, info.id, info.title)
        return await self._refresh(topic.id)

    async def _announce_created(self, topic: Topic, info: ChatInfo) -> None:
        rt = self._rt
        can_post = rt.bot is None or await rt.bot.can_post(info.id)
        username = rt.bot_account.username if rt.bot_account is not None else None
        text = rt.t(
            "topics_channel_auto_created" if can_post else "topics_channel_auto_created_no_rights",
            name=html_escape(topic.name),
            channel=html_escape(info.title),
            bot=html_escape(f"@{username}" if username else "the bot"),
        )
        await rt.notifier.owner(text)  # best effort: never raises

    async def update(self, key: str, **changes: Any) -> Topic:
        """Edit name / category / description / channel / example_channel / strictness."""
        allowed = {"name", "category", "description", "channel", "example_channel", "strictness"}
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(f"update() cannot change {', '.join(sorted(unknown))}")
        topic = await self._active_topic(key)
        fields: dict[str, Any] = {}
        file_fields: dict[str, Any] = {}
        if "category" in changes:  # checked first: a refused edit leaves nothing changed
            value = _category(changes["category"])
            fields["category"] = value
            file_fields["category"] = value or ""
        if "name" in changes:
            name = str(changes["name"]).strip()
            if not name:
                raise ConfigError("a topic needs a name")
            await self._ensure_name_free(name, except_key=key)
            fields["name"] = file_fields["name"] = name
        if "channel" in changes:
            ref = changes["channel"]
            if ref in (None, 0, ""):
                raise ConfigError(
                    "a channel cannot be removed from a topic: remove the topic, or /pause to "
                    "stop posting"
                )
            info = await self._resolve_output_channel(ref, topic_id=topic.id)
            fields["channel_id"] = file_fields["channel"] = info.id
        if "description" in changes:
            value = _blank_to_none(changes["description"])
            fields["description"] = value
            file_fields["description"] = value or ""
        if "strictness" in changes:
            value = _strictness(changes["strictness"])
            fields["strictness"] = value
            file_fields["strictness"] = value or 0.0
        example_texts: list[str] = []
        if "example_channel" in changes:
            ref = _blank_to_none(changes["example_channel"])
            if ref and ref != topic.example_channel:
                example_texts = await self._read_example_channel(ref)
            fields["example_channel"] = ref
            file_fields["example_channel"] = ref or ""

        if fields:
            await self._rt.store.set_topic_fields(topic.id, **fields)
        if example_texts:
            await self._store_examples(topic, example_texts, kind="channel")
        if file_fields:
            await self._rt.settings_file.upsert_topic(key, **file_fields)
        await retrain_classifier(self._rt)
        if example_texts or "description" in changes:
            await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="examples")
        log.info("topic %s updated: %s", key, ", ".join(sorted(fields)) or "nothing")
        return await self._refresh(topic.id)

    async def remove(self, key: str) -> None:
        """§9.7 deactivation + removal from the settings file; the channel is never touched."""
        topic = await self._active_topic(key)
        await self._deactivate(topic, None)
        await self._rt.settings_file.remove_topic(key)
        log.info("topic %s removed (deactivated; its channel is untouched)", key)
        await self._after_topic_set_changed()

    async def merge(self, src_key: str, dst_key: str) -> Topic:
        """§9.7 with ``dst``: posts, open publications and examples move to ``dst``."""
        if src_key == dst_key:
            raise ConfigError("a topic cannot be merged into itself")
        src = await self._active_topic(src_key)
        dst = await self._active_topic(dst_key)
        await self._deactivate(src, dst)
        await self._rt.settings_file.remove_topic(src_key)
        log.info("topic %s merged into %s", src_key, dst_key)
        await self._after_topic_set_changed()
        return await self._refresh(dst.id)

    async def _deactivate(self, topic: Topic, dst: Topic | None) -> None:
        """Steps 1–5 of §9.7 in one transaction (step 6 is the caller's)."""
        store = self._rt.store
        posts, pubs, digests, examples = (
            schema.posts,
            schema.publications,
            schema.digests,
            schema.examples,
        )
        # A post whose publication is being sent right now finishes in the old channel (§9.7
        # step 3), so its status is left to the publisher.
        sending = sa.select(pubs.c.post_id).where(pubs.c.state == PUB_SENDING)
        move_posts = (
            sa.update(posts)
            .where(posts.c.topic_id == topic.id)
            .where(posts.c.status.in_([s.value for s in _MOVABLE]))
            .where(posts.c.id.not_in(sending))
        )
        open_pubs = (
            sa.update(pubs)
            .where(pubs.c.topic_id == topic.id)
            .where(pubs.c.state.in_([PUB_PENDING, PUB_FAILED]))
        )
        dst_has_channel = dst is not None and dst.channel_id is not None
        async with store.begin() as conn:
            await store.set_topic_fields(topic.id, active=False, conn=conn)
            if dst is None:
                await conn.execute(
                    move_posts.values(
                        status=PostStatus.unsorted.value, topic_id=None, confidence=None
                    )
                )
            elif dst_has_channel:
                await conn.execute(move_posts.values(topic_id=dst.id))
            else:
                await conn.execute(
                    move_posts.values(topic_id=dst.id, status=PostStatus.tracked.value)
                )
            if dst_has_channel:
                await conn.execute(open_pubs.values(topic_id=dst.id))
            else:
                await conn.execute(open_pubs.values(state=PUB_CANCELLED))
            await conn.execute(
                sa.update(digests)
                .where(digests.c.topic_id == topic.id)
                .where(digests.c.state.in_([DIGEST_PENDING, DIGEST_SENDING]))
                .values(state=DIGEST_CANCELLED)
            )
            if dst is None:
                await store.delete_examples(topic.id, conn=conn)
            else:
                await conn.execute(
                    sa.update(examples)
                    .where(examples.c.topic_id == topic.id)
                    .values(topic_id=dst.id)
                )

    async def _after_topic_set_changed(self) -> None:
        """§9.7 step 6 / the tail of ``create``: reload, re-sort, tell everyone."""
        await retrain_classifier(self._rt)
        sorter = self._rt.sorter
        if sorter is not None:
            since = self._rt.clock.now() - timedelta(
                days=self._rt.settings.review.cluster_window_days
            )
            n = await sorter.resort_unsorted(since)
            log.info("re-sorted %d unsorted posts after the topic change", n)
        await self._rt.events.emit(EVENT_TOPICS_CHANGED)
        await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="topics")

    # --- examples ----------------------------------------------------------------------------

    async def add_examples(self, key: str, texts: Sequence[str]) -> int:
        """Store example posts (``kind='example'``) and announce them."""
        topic = await self._active_topic(key)
        clean = [t.strip() for t in texts if t and t.strip()]
        if not clean:
            return 0
        n = await self._store_examples(topic, clean, kind="example")
        await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="examples")
        log.info("topic %s: %d example posts added", key, n)
        return n

    async def add_example_channel(self, key: str, ref: str) -> int:
        """Read up to 50 recent posts of a channel as examples (``kind='channel'``)."""
        topic = await self._active_topic(key)
        ref = ref.strip()
        texts = await self._read_example_channel(ref)
        n = await self._store_examples(topic, texts, kind="channel") if texts else 0
        await self._rt.store.set_topic_fields(topic.id, example_channel=ref)
        await self._rt.settings_file.upsert_topic(key, example_channel=ref)
        await self._rt.events.emit(EVENT_EXAMPLES_CHANGED, reason="examples")
        log.info("topic %s: %d examples read from %s", key, n, ref)
        return n

    async def _ingest_example_channel(self, topic: Topic, ref: str) -> int:
        texts = await self._read_example_channel(ref)
        return await self._store_examples(topic, texts, kind="channel") if texts else 0

    async def _read_example_channel(self, ref: str) -> list[str]:
        """The texts of the most recent posts of ``ref`` (``NotAllowed``/``ChatGone`` propagate)."""
        info = await self._user().resolve_chat(ref)
        since = self._rt.clock.now() - timedelta(days=EXAMPLE_CHANNEL_DAYS)
        texts: list[str] = []
        async for msg in self._user().history(
            info.id, since=since, limit=EXAMPLE_CHANNEL_READ_LIMIT
        ):
            if (
                msg.is_service
                or (msg.is_outgoing and msg.chat.kind == "group")
                or not msg.text.strip()
            ):
                continue
            texts.append(msg.text.strip())
        log.info("example channel %s: %d posts with text read", info.title, len(texts))
        return texts[-EXAMPLE_CHANNEL_MAX:]

    async def _store_examples(
        self, topic: Topic, texts: Sequence[str], *, kind: ExampleKind
    ) -> int:
        matrix = await asyncio.to_thread(self._rt.embedder.embed, list(texts))
        now = self._rt.clock.now()
        store = self._rt.store
        async with store.begin() as conn:
            for text, vector in zip(texts, matrix, strict=True):
                await store.add_example(
                    Example(
                        id=0,
                        topic_id=topic.id,
                        kind=kind,
                        text=text,
                        embedding=embedding_bytes(vector),
                        created_at=now,
                    ),
                    conn=conn,
                )
        return len(texts)

    # --- helpers -----------------------------------------------------------------------------

    def _user(self) -> UserGateway:
        if self._rt.user is None:
            raise ConfigError("no account is bound: send /bind to sign in first")
        return self._rt.user

    def _bot_username(self) -> str:
        account = self._rt.bot_account
        if account is None or not account.username:
            raise ConfigError("the bot has no username yet; start the service first")
        return account.username

    async def _active_topic(self, key: str) -> Topic:
        topic = await self._rt.store.get_topic_by_key(key)
        if topic is None or not topic.active:
            raise ConfigError(f"topic {key} does not exist; /topics lists the current ones")
        return topic

    async def _refresh(self, topic_id: int) -> Topic:
        topic = await self._rt.store.get_topic(topic_id)
        assert topic is not None
        return topic

    async def _ensure_name_free(self, name: str, *, except_key: str | None) -> None:
        wanted = name.casefold()
        for topic in await self._rt.store.list_topics(active=True):
            if topic.key != except_key and topic.name.casefold() == wanted:
                raise TopicExists(f'a topic named "{topic.name}" already exists ({topic.key})')
        for entry in self._rt.settings.topics:
            if entry.key != except_key and entry.name.casefold() == wanted:
                raise TopicExists(f'a topic named "{entry.name}" already exists ({entry.key})')


def _reason(exc: CuratorError) -> str:
    """The sentence a sync result shows for a failure the §8 wording does not already cover."""
    if isinstance(exc, NotAllowed) and exc.reason == "not_a_member":
        return "join the chat in Telegram first, then send the link again"
    return str(exc)


def _rewrite_refs(doc: TOMLDocument, channels: dict[str, int], sources: dict[str, int]) -> None:
    """Replace resolved ``@username``/link values with numeric ids, comments untouched."""
    for table in doc.get("topics") or []:
        key = table.get("key")
        if key in channels:
            table["channel"] = channels[key]
    for table in doc.get("sources") or []:
        chat = table.get("chat")
        if isinstance(chat, str) and chat in sources:
            table["chat"] = sources[chat]
