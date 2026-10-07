"""/topics: list, add, edit, merge and remove topics from the bot (DESIGN §11.2, §4, §14.13/14).

Everything here is conversation; the work is done by ``rt.topics`` (``TopicsService``), whose
error sentences are shown as they are because §11.2 wants exactly those wordings. A new topic
is created as soon as it has a name, and every later answer (category, example posts, example
channel, description, channel) is applied the moment it arrives: the owner sees each one
confirmed, and an abandoned conversation leaves a valid tracked-only topic rather than half a
form in ``kv``. Callback data carries topic ids and list indexes only (§11.1).
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache

from tg_curator.bot.core import BotApp, Ctx, Flow
from tg_curator.config import TEMPLATE_PATH, TopicSettings, validate_settings
from tg_curator.contracts import TopicsService
from tg_curator.domain import Category, Topic
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
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import html_escape

log = logging.getLogger(__name__)

PREFIX = "tp"
CATEGORIES_PER_PAGE = 10
"""Five rows of two: the 20 built-in categories fit on two pages."""
EXAMPLE_MIN_CHARS = 20
"""A message at least this long during the "what is it" step is an example post."""
STRICTNESS_CHOICES = (0, 40, 50, 60, 70, 80)
"""Per-topic confidence thresholds offered, in percent; 0 = use ``sorting.confidence``."""

FLOW_ADD = "topic_add"
FLOW_EDIT = "topic_edit"


def register(app: BotApp) -> None:
    app.command("topics", topics_command, help_key="topics_help")
    app.callback(PREFIX, on_callback)
    app.flow(FLOW_ADD, TopicAddFlow)
    app.flow(FLOW_EDIT, TopicEditFlow)


# --- /topics ---------------------------------------------------------------------------------


async def topics_command(ctx: Ctx, args: str) -> None:
    """``/topics`` lists the topics, ``/topics add`` starts the add conversation."""
    if args.strip().casefold() == "add":
        await ctx.start_flow(FLOW_ADD)
        return
    await send_list(ctx)


async def send_list(ctx: Ctx) -> None:
    """A header with [Add a topic], then one card per active topic with its own buttons."""
    rt = ctx.rt
    topics = await rt.store.list_topics(active=True)
    add = [[Button(rt.t("topics_add_button"), data=f"{PREFIX}:add")]]
    if not topics:
        await ctx.reply("topics_list_empty", buttons=add)
        return
    await ctx.reply("topics_list_header", buttons=add, count=len(topics))
    for topic in topics:
        await ctx.reply(await card_text(rt, topic), buttons=card_buttons(rt, topic, len(topics)))


async def card_text(rt: Runtime, topic: Topic) -> str:
    examples = len(await rt.store.list_examples(topic_id=topic.id))
    return rt.t(
        "topics_card",
        name=html_escape(topic.name),
        channel=await _channel_label(rt, topic),
        category=_category_label(rt, topic.category),
        examples=examples,
        description=html_escape(topic.description) if topic.description else rt.t("topics_none"),
        strictness=_strictness_label(rt, topic.strictness),
    )


def card_buttons(rt: Runtime, topic: Topic, n_topics: int) -> Buttons:
    row = [Button(rt.t("edit"), data=f"{PREFIX}:e:{topic.id}")]
    if n_topics > 1:
        row.append(Button(rt.t("topics_merge_button"), data=f"{PREFIX}:m:{topic.id}"))
    row.append(Button(rt.t("remove"), data=f"{PREFIX}:r:{topic.id}"))
    rows = [row]
    if topic.channel_id is None:
        rows.append([Button(rt.t("topics_create_channel_button"), data=f"{PREFIX}:cn:{topic.id}")])
    return rows


# --- the four example topics of the template (§4, §14.14) -----------------------------------


@cache
def _template_topics() -> dict[str, TopicSettings]:
    with TEMPLATE_PATH.open("rb") as fh:
        return {t.key: t for t in validate_settings(tomllib.load(fh)).topics}


async def untouched_examples(rt: Runtime) -> list[Topic]:
    """Template topics still exactly as shipped: same settings, no channel, no examples."""
    template = _template_topics()
    out: list[Topic] = []
    for entry in rt.settings.topics:
        if template.get(entry.key) != entry:
            continue
        topic = await rt.store.get_topic_by_key(entry.key)
        if topic is None or not topic.active or topic.channel_id is not None:
            continue
        if await rt.store.list_examples(topic_id=topic.id):
            continue
        out.append(topic)
    return out


async def offer_examples(ctx: Ctx) -> int:
    """One message per untouched example topic with [Keep] [Edit] [Remove]; also used by
    setup step 4. Returns how many were offered."""
    rt = ctx.rt
    offered = await untouched_examples(rt)
    if offered:
        await ctx.reply("topics_examples_intro", count=len(offered))
    for topic in offered:
        buttons = [
            [
                Button(rt.t("keep"), data=f"{PREFIX}:k:{topic.id}"),
                Button(rt.t("edit"), data=f"{PREFIX}:e:{topic.id}"),
                Button(rt.t("remove"), data=f"{PREFIX}:r:{topic.id}"),
            ]
        ]
        await ctx.reply(await card_text(rt, topic), buttons=buttons)
    return len(offered)


# --- the add conversation ----------------------------------------------------------------------


class _TopicFlow(Flow):
    """A conversation about one topic, ``data["topic_id"]``."""

    async def _topic(self) -> Topic | None:
        topic = await self.ctx.rt.store.get_topic(int(self.data.get("topic_id", 0)))
        if topic is None or not topic.active:
            await self.end()
            await self.ctx.reply("unknown_choice")
            return None
        return topic

    async def _take_example(self, topic: Topic, text: str) -> None:
        """One answer to "give me examples": a forwarded or pasted post of at least
        ``EXAMPLE_MIN_CHARS`` characters is an example post, a channel ref that was not
        forwarded is an example channel, anything else is too short. ``data["examples"]``
        counts what was added in this conversation."""
        text = text.strip()
        msg = self.ctx.message
        forwarded = msg is not None and msg.fwd_from_chat_id is not None
        if not text and msg is not None and (forwarded or msg.fwd_from_title or msg.has_media):
            # A forwarded photo or video without a caption: not "too short", it has no text
            # the classifier could learn from (SPEC: "example posts forwarded to the bot").
            await self.ctx.reply("topics_example_no_text")
        elif not forwarded and _is_channel_ref(text):
            await self._example_channel(topic, text)
        elif len(text) >= EXAMPLE_MIN_CHARS:
            added = await _service(self.ctx.rt).add_examples(topic.key, [text])
            self.data["examples"] = int(self.data.get("examples", 0)) + added
            await self.save()
            await self.ctx.reply("topics_example_saved", count=self.data["examples"])
        else:
            await self.ctx.reply("topics_example_too_short", chars=EXAMPLE_MIN_CHARS)

    async def _example_channel(self, topic: Topic, ref: str) -> bool:
        """Read ``ref`` as the topic's example channel; ``False`` when it could not be read
        (the owner was told why)."""
        try:
            read = await _service(self.ctx.rt).add_example_channel(topic.key, ref)
        except NotAllowed as exc:
            if exc.reason != "not_a_member":
                raise
            await self.ctx.reply("topics_join_first")
            return False
        except ChatGone:
            await self.ctx.reply("topics_example_channel_gone", ref=html_escape(ref))
            return False
        self.data["examples"] = int(self.data.get("examples", 0)) + read
        await self.save()
        await self.ctx.reply("topics_example_channel_read", count=read, ref=html_escape(ref))
        return True


class TopicAddFlow(_TopicFlow):
    """``/topics add``: name -> what it is -> channel. ``data``: ``topic_id``, ``examples``."""

    first_step = "name"

    async def start(self) -> None:
        await offer_examples(self.ctx)
        await self.ctx.reply("topics_add_ask_name")

    @Flow.step("name")
    async def name(self, text: str) -> None:
        rt = self.ctx.rt
        name = text.strip()
        if not name:
            await self.ctx.reply("topics_add_ask_name")
            return
        try:
            topic = await _service(rt).create(name)
        except TopicExists:
            await _reply_exists(self.ctx, name)
            return
        await self.go("what", topic_id=topic.id, examples=0)
        await self.show_what(topic)

    @Flow.step("what")
    async def what(self, text: str) -> None:
        topic = await self._topic()
        if topic is not None:
            await self._take_example(topic, text)

    @Flow.step("description")
    async def description(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        topic = await _service(self.ctx.rt).update(topic.key, description=text.strip())
        await self.go("what")
        await self.ctx.reply("topics_description_saved")
        await self.show_what(topic)

    @Flow.step("channel")
    async def channel(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        try:
            topic = await _service(self.ctx.rt).link_channel(topic.key, text.strip())
        except (ConfigError, BotCannotPost, NotAllowed, FloodWait) as exc:
            await reply_channel_failure(self.ctx, exc)
            return
        await self.finish(topic)

    # --- driven by buttons (``tp:a…``) ---

    async def show_what(self, topic: Topic, page: int = 0, *, edit: bool = False) -> None:
        rt = self.ctx.rt
        rows = category_rows(
            rt, page, pick=lambda i: f"{PREFIX}:ac:{i}", turn=lambda p: f"{PREFIX}:ap:{p}"
        )
        rows.append(
            [
                Button(rt.t("topics_description_button"), data=f"{PREFIX}:ad"),
                Button(rt.t("done"), data=f"{PREFIX}:af"),
            ]
        )
        fmt = {
            "name": html_escape(topic.name),
            "category": _category_label(rt, topic.category),
            "chars": EXAMPLE_MIN_CHARS,
        }
        if edit:
            await self.ctx.edit("topics_add_what", buttons=rows, **fmt)
        else:
            await self.ctx.reply("topics_add_what", buttons=rows, **fmt)

    async def pick_category(self, index: int) -> None:
        topic = await self._topic()
        category = _category_at(index)
        if topic is None or category is None:
            await self.ctx.reply("unknown_choice")
            return
        topic = await _service(self.ctx.rt).update(topic.key, category=category.key)
        await self.show_what(topic, index // CATEGORIES_PER_PAGE, edit=True)

    async def turn_page(self, page: int) -> None:
        topic = await self._topic()
        if topic is not None:
            await self.show_what(topic, page, edit=True)

    async def ask_description(self) -> None:
        await self.go("description")
        await self.ctx.reply("topics_add_ask_description")

    async def done_with_what(self) -> None:
        topic = await self._topic()
        if topic is None:
            return
        if topic.category is None and not await self.ctx.rt.store.list_examples(topic_id=topic.id):
            await self.ctx.reply("topics_add_sorts_nothing", name=html_escape(topic.name))
        await self.go("channel")
        rt = self.ctx.rt
        buttons = [
            [Button(rt.t("topics_channel_create_button"), data=f"{PREFIX}:an")],
            [Button(rt.t("topics_channel_link_button"), data=f"{PREFIX}:al")],
            [Button(rt.t("topics_channel_none_button"), data=f"{PREFIX}:a0")],
        ]
        await self.ctx.reply(
            "topics_add_ask_channel", buttons=buttons, name=html_escape(topic.name)
        )

    async def create_channel(self) -> None:
        topic = await self._topic()
        if topic is None:
            return
        result = await create_channel_for(self.ctx, topic)
        if result.topic is not None and result.can_post:
            await self.finish(result.topic)
        elif result.topic is not None or result.must_wait:
            # Linked, but the bot has no rights yet (the reply said what to do; the publisher
            # reports it once per channel, §11.4) — or saved with channel 0 and /topics
            # offers [Create channel] later. Either way the conversation is over, so the
            # [Create a private one] button cannot make a second channel.
            await self.end()

    async def ask_link(self) -> None:
        await self.ctx.reply("topics_channel_ask_link")

    async def no_channel(self) -> None:
        topic = await self._topic()
        if topic is None:
            return
        await self.end()
        await self.ctx.reply("topics_add_done_no_channel", name=html_escape(topic.name))

    async def finish(self, topic: Topic) -> None:
        await self.end()
        await self.ctx.reply(
            "topics_add_done",
            name=html_escape(topic.name),
            channel=await _channel_label(self.ctx.rt, topic),
        )


# --- the edit conversation (text answers only; choices are buttons) ---------------------------


_EDIT_PROMPTS = {
    "name": "topics_edit_ask_name",
    "description": "topics_edit_ask_description",
    "channel": "topics_channel_ask_link",
    "examples": "topics_edit_ask_examples",
    "example_channel": "topics_edit_ask_example_channel",
}


class TopicEditFlow(_TopicFlow):
    """Text answers for one existing topic: ``data`` = ``topic_id`` and the ``field`` asked
    for. Every field but ``examples`` takes one answer; ``examples`` (spec "Tuning": forward
    a topic a few more examples) takes posts until [Done] (``tp:ef``) returns to the card."""

    async def start(self) -> None:
        field = str(self.data.get("field"))
        if field not in _EDIT_PROMPTS:
            await self.end()
            await self.ctx.reply("unknown_choice")
            return
        await self.go(field)
        if field not in ("examples", "example_channel"):
            await self.ctx.reply(_EDIT_PROMPTS[field])
            return
        topic = await self._topic()
        if topic is None:
            return
        buttons = None
        if field == "examples":
            self.data["examples"] = 0
            await self.save()
            buttons = [[Button(self.ctx.rt.t("done"), data=f"{PREFIX}:ef:{topic.id}")]]
        await self.ctx.reply(
            _EDIT_PROMPTS[field],
            buttons=buttons,
            name=html_escape(topic.name),
            chars=EXAMPLE_MIN_CHARS,
            current=html_escape(topic.example_channel or "") or self.ctx.rt.t("topics_none"),
        )

    @Flow.step("name")
    async def name(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        try:
            topic = await _service(self.ctx.rt).update(topic.key, name=text.strip())
        except TopicExists:
            await _reply_exists(self.ctx, text.strip())
            return
        await self._done(topic)

    @Flow.step("description")
    async def description(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        value = "" if text.strip() == "-" else text.strip()
        await self._done(await _service(self.ctx.rt).update(topic.key, description=value))

    @Flow.step("channel")
    async def channel(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        try:
            topic = await _service(self.ctx.rt).link_channel(topic.key, text.strip())
        except (ConfigError, BotCannotPost, NotAllowed, FloodWait) as exc:
            await reply_channel_failure(self.ctx, exc)
            return
        await self._done(topic)

    @Flow.step("examples")
    async def examples(self, text: str) -> None:
        topic = await self._topic()
        if topic is not None:
            await self._take_example(topic, text)

    @Flow.step("example_channel")
    async def example_channel(self, text: str) -> None:
        topic = await self._topic()
        if topic is None:
            return
        ref = text.strip()
        forwarded = self.ctx.message is not None and self.ctx.message.fwd_from_chat_id is not None
        if forwarded or not _is_channel_ref(ref):
            await self.ctx.reply("topics_edit_example_channel_not_a_ref")
            return
        if await self._example_channel(topic, ref):
            await self._done(await self.ctx.rt.store.get_topic(topic.id) or topic)

    async def _done(self, topic: Topic) -> None:
        await self.end()
        rt = self.ctx.rt
        n = len(await rt.store.list_topics(active=True))
        await self.ctx.reply(await card_text(rt, topic), buttons=card_buttons(rt, topic, n))


# --- channels ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelResult:
    """What ``create_channel_for`` did: ``topic`` is the linked topic (``None`` when no
    channel was created), ``must_wait`` says Telegram's pacing stopped the creation, and
    ``can_post`` whether the bot can post into the new channel already."""

    topic: Topic | None
    must_wait: bool = False
    can_post: bool = False


async def create_channel_for(ctx: Ctx, topic: Topic) -> ChannelResult:
    """Create a private channel for an existing topic and link it.

    For a topic that already exists (the [Create channel] button of §11.2, and the add
    conversation, which creates the topic at the name step) ``TopicsService``'s paced creator
    is used, so the pacing and the ``kv topics.create_log`` bookkeeping stay in one place.

    A created channel is the topic's whatever the bot's rights are, as in
    ``TopicsService.create(create_channel=True)``: the creator already registered it and
    tried to make the bot an admin. When that failed (Telegram refuses admin changes from a
    session younger than a day) the reply says what to do, and the publisher reports the
    missing rights once per channel (§11.4) until the owner fixes them — a channel left
    unlinked would be an orphan, and every retry would make another.
    """
    rt = ctx.rt
    service = _service(rt)
    wait = service.channel_wait_minutes()
    info = None
    if wait is None:
        info = await service.create_topic_channel(topic.name)
        wait = service.channel_wait_minutes() if info is None else None
    if info is None:
        await service.want_channel(topic.key)  # created by TopicsService.tick after the wait
        await ctx.reply("topics_channel_wait", minutes=wait or 1)
        return ChannelResult(None, must_wait=True)
    linked = await _attach_created_channel(rt, topic, info.id)
    if await _bot_can_post(ctx, linked, info.id):
        return ChannelResult(linked, can_post=True)
    return ChannelResult(linked)


async def _attach_created_channel(rt: Runtime, topic: Topic, channel_id: int) -> Topic:
    """Record a channel ``create_topic_channel`` made (and adopted) on the topic, in the
    database and the settings file — ``link_channel`` would adopt it again and refuse."""
    await rt.store.set_topic_fields(topic.id, channel_id=channel_id)
    entry = rt.settings.topic(topic.key)
    if entry is not None and entry.channel != channel_id:
        await rt.settings_file.upsert_topic(topic.key, channel=channel_id)
    log.info("topic %s posts into the new channel %d", topic.key, channel_id)
    return await rt.store.get_topic(topic.id) or topic


async def _bot_can_post(ctx: Ctx, topic: Topic, channel_id: int) -> bool:
    """``True`` when the bot can post into the new channel; otherwise the owner is told why.

    The creator's attempt to make the bot an admin is logged, not returned, so it is made
    once more here to learn the reason: a session younger than a day gets the §15 sentence.
    """
    rt = ctx.rt
    bot = rt.bot
    if bot is None or await bot.can_post(channel_id):
        return True
    username = rt.bot_account.username if rt.bot_account is not None else None
    fresh_session = False
    if rt.user is not None and username:
        try:
            await rt.user.add_bot_admin(channel_id, username)
        except NotAllowed as exc:
            fresh_session = exc.reason == "fresh_session"
            log.info("topic %s: the bot is not an admin of its channel: %s", topic.key, exc)
        except CuratorError as exc:
            log.info("topic %s: the bot is not an admin of its channel: %s", topic.key, exc)
        else:
            if await bot.can_post(channel_id):
                return True
    await ctx.reply(
        "topics_channel_no_rights_fresh" if fresh_session else "topics_channel_no_rights",
        name=html_escape(topic.name),
        channel=await _channel_label(rt, topic),
        bot=html_escape(f"@{username}" if username else "the bot"),
    )
    return False


async def reply_channel_failure(ctx: Ctx, exc: Exception) -> None:
    """The §11.2 wordings: ``link_channel``'s own sentences, "join first", the flood wait."""
    if isinstance(exc, NotAllowed) and exc.reason == "not_a_member":
        await ctx.reply("topics_join_first")
    elif isinstance(exc, FloodWait):
        await ctx.reply("topics_flood_wait", minutes=max(1, -(-exc.seconds // 60)))
    else:
        await ctx.reply("topics_channel_error", error=html_escape(str(exc)))


# --- button handlers -------------------------------------------------------------------------


async def on_callback(ctx: Ctx, data: str) -> None:
    verb, *rest = data.split(":")
    entry = _CALLBACKS.get(verb)
    try:
        args = [int(part) for part in rest]
    except ValueError:
        entry = None
    if entry is None or len(args) != entry[1]:
        log.warning("topics: unknown callback %r", data)
        await ctx.reply("unknown_choice")
        return
    await entry[0](ctx, *args)


async def _add(ctx: Ctx) -> None:
    await ctx.start_flow(FLOW_ADD)


_DESCRIBING = ("what", "description")
_CHOOSING_CHANNEL = ("channel",)


async def _with_add_flow(
    ctx: Ctx, steps: tuple[str, ...], action: Callable[[TopicAddFlow], Awaitable[None]]
) -> None:
    """Run ``action`` on the active add conversation when it is at one of ``steps``; a tap on
    an old prompt of a finished or different conversation is refused."""
    flow = await ctx.flow()
    if not isinstance(flow, TopicAddFlow) or flow.current_step not in steps:
        await ctx.reply("unknown_choice")
        return
    await action(flow)


async def _add_category(ctx: Ctx, index: int) -> None:
    await _with_add_flow(ctx, _DESCRIBING, lambda f: f.pick_category(index))


async def _add_page(ctx: Ctx, page: int) -> None:
    await _with_add_flow(ctx, _DESCRIBING, lambda f: f.turn_page(page))


async def _add_description(ctx: Ctx) -> None:
    await _with_add_flow(ctx, _DESCRIBING, lambda f: f.ask_description())


async def _add_done(ctx: Ctx) -> None:
    await _with_add_flow(ctx, _DESCRIBING, lambda f: f.done_with_what())


async def _add_channel_new(ctx: Ctx) -> None:
    await _with_add_flow(ctx, _CHOOSING_CHANNEL, lambda f: f.create_channel())


async def _add_channel_link(ctx: Ctx) -> None:
    await _with_add_flow(ctx, _CHOOSING_CHANNEL, lambda f: f.ask_link())


async def _add_channel_none(ctx: Ctx) -> None:
    await _with_add_flow(ctx, _CHOOSING_CHANNEL, lambda f: f.no_channel())


async def _open(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is not None:
        n = len(await ctx.rt.store.list_topics(active=True))
        await ctx.edit(await card_text(ctx.rt, topic), buttons=card_buttons(ctx.rt, topic, n))


async def _keep(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is not None:
        await ctx.edit("topics_example_kept", name=html_escape(topic.name))


async def _edit_menu(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    t = ctx.rt.t
    buttons = [
        [
            Button(t("topics_field_name"), data=f"{PREFIX}:en:{topic.id}"),
            Button(t("topics_field_description"), data=f"{PREFIX}:ed:{topic.id}"),
        ],
        [
            Button(t("topics_field_category"), data=f"{PREFIX}:ec:{topic.id}:0"),
            Button(t("topics_field_strictness"), data=f"{PREFIX}:et:{topic.id}"),
        ],
        [
            Button(t("topics_field_examples"), data=f"{PREFIX}:ee:{topic.id}"),
            Button(t("topics_field_example_channel"), data=f"{PREFIX}:ev:{topic.id}"),
        ],
        [
            Button(t("topics_field_channel"), data=f"{PREFIX}:el:{topic.id}"),
            Button(t("back"), data=f"{PREFIX}:o:{topic.id}"),
        ],
    ]
    await ctx.edit(await card_text(ctx.rt, topic), buttons=buttons)


async def _edit_examples(ctx: Ctx, topic_id: int) -> None:
    if await _active(ctx, topic_id) is not None:
        await ctx.start_flow(FLOW_EDIT, topic_id=topic_id, field="examples")


async def _edit_example_channel(ctx: Ctx, topic_id: int) -> None:
    if await _active(ctx, topic_id) is not None:
        await ctx.start_flow(FLOW_EDIT, topic_id=topic_id, field="example_channel")


async def _examples_done(ctx: Ctx, topic_id: int) -> None:
    """[Done] under "forward more examples": close that conversation, show the card."""
    flow = await ctx.flow()
    if (
        isinstance(flow, TopicEditFlow)
        and flow.current_step == "examples"
        and int(flow.data.get("topic_id", 0)) == topic_id
    ):
        await flow.end()
    await _open(ctx, topic_id)


async def _edit_name(ctx: Ctx, topic_id: int) -> None:
    if await _active(ctx, topic_id) is not None:
        await ctx.start_flow(FLOW_EDIT, topic_id=topic_id, field="name")


async def _edit_description(ctx: Ctx, topic_id: int) -> None:
    if await _active(ctx, topic_id) is not None:
        await ctx.start_flow(FLOW_EDIT, topic_id=topic_id, field="description")


async def _edit_category_page(ctx: Ctx, topic_id: int, page: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    rows = category_rows(
        ctx.rt,
        page,
        pick=lambda i: f"{PREFIX}:es:{topic_id}:{i}",
        turn=lambda p: f"{PREFIX}:ec:{topic_id}:{p}",
    )
    rows.append(
        [
            Button(ctx.rt.t("topics_category_none_button"), data=f"{PREFIX}:ex:{topic_id}"),
            Button(ctx.rt.t("back"), data=f"{PREFIX}:o:{topic_id}"),
        ]
    )
    await ctx.edit(
        "topics_edit_category",
        buttons=rows,
        name=html_escape(topic.name),
        category=_category_label(ctx.rt, topic.category),
    )


async def _edit_category_set(ctx: Ctx, topic_id: int, index: int) -> None:
    topic = await _active(ctx, topic_id)
    category = _category_at(index)
    if topic is None or category is None:
        await ctx.reply("unknown_choice")
        return
    await _service(ctx.rt).update(topic.key, category=category.key)
    await _open(ctx, topic_id)


async def _edit_category_clear(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is not None:
        await _service(ctx.rt).update(topic.key, category="")
        await _open(ctx, topic_id)


async def _edit_strictness(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    rt = ctx.rt
    choices = [
        Button(_strictness_label(rt, pct / 100 or None), data=f"{PREFIX}:ep:{topic_id}:{pct}")
        for pct in STRICTNESS_CHOICES
    ]
    rows = [choices[i : i + 3] for i in range(0, len(choices), 3)]
    rows.append([Button(rt.t("back"), data=f"{PREFIX}:o:{topic_id}")])
    await ctx.edit(
        "topics_edit_strictness",
        buttons=rows,
        name=html_escape(topic.name),
        current=_strictness_label(rt, topic.strictness),
    )


async def _strictness_proposed(ctx: Ctx, topic_id: int, pct: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    if pct not in STRICTNESS_CHOICES:
        await ctx.reply("unknown_choice")
        return
    rt = ctx.rt
    buttons = [
        [
            Button(rt.t("topics_preview_button"), data=f"{PREFIX}:eq:{topic_id}:{pct}"),
            Button(rt.t("topics_apply_button"), data=f"{PREFIX}:ea:{topic_id}:{pct}"),
        ],
        [Button(rt.t("cancel"), data=f"{PREFIX}:o:{topic_id}")],
    ]
    await ctx.edit(
        "topics_strictness_proposed",
        buttons=buttons,
        name=html_escape(topic.name),
        value=_strictness_label(rt, pct / 100 or None),
    )


async def _strictness_preview(ctx: Ctx, topic_id: int, pct: int) -> None:
    """What the proposed strictness would have done to the last three days (§11.2)."""
    topic = await _active(ctx, topic_id)
    rt = ctx.rt
    if topic is None or rt.preview is None or pct not in STRICTNESS_CHOICES:
        await ctx.reply("unknown_choice")
        return
    now = await rt.preview.replay(days=3)
    then = await rt.preview.replay(days=3, overrides={f"topics.{topic.key}.strictness": pct / 100})
    await ctx.reply(
        "topics_strictness_preview",
        name=html_escape(topic.name),
        value=_strictness_label(rt, pct / 100 or None),
        now=now.per_topic.get(topic.key, 0),
        then=then.per_topic.get(topic.key, 0),
        unsorted_now=now.unsorted,
        unsorted_then=then.unsorted,
    )


async def _strictness_apply(ctx: Ctx, topic_id: int, pct: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    if pct not in STRICTNESS_CHOICES:
        await ctx.reply("unknown_choice")
        return
    await _service(ctx.rt).update(topic.key, strictness=pct / 100)
    await _open(ctx, topic_id)


async def _channel_menu(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    t = ctx.rt.t
    buttons = [
        [Button(t("topics_channel_create_button"), data=f"{PREFIX}:cn:{topic_id}")],
        [Button(t("topics_channel_link_button"), data=f"{PREFIX}:cl:{topic_id}")],
        [Button(t("back"), data=f"{PREFIX}:o:{topic_id}")],
    ]
    await ctx.edit(
        "topics_edit_channel",
        buttons=buttons,
        name=html_escape(topic.name),
        channel=await _channel_label(ctx.rt, topic),
    )


async def _channel_create(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    result = await create_channel_for(ctx, topic)
    if result.topic is not None and result.can_post:
        await ctx.reply(
            "topics_channel_created",
            name=html_escape(result.topic.name),
            channel=await _channel_label(ctx.rt, result.topic),
        )


async def _channel_link(ctx: Ctx, topic_id: int) -> None:
    if await _active(ctx, topic_id) is not None:
        await ctx.start_flow(FLOW_EDIT, topic_id=topic_id, field="channel")


async def _merge_menu(ctx: Ctx, src_id: int) -> None:
    src = await _active(ctx, src_id)
    if src is None:
        return
    others = [t for t in await ctx.rt.store.list_topics(active=True) if t.id != src_id]
    rows = pairs([Button(t.name, data=f"{PREFIX}:mm:{src_id}:{t.id}") for t in others])
    rows.append([Button(ctx.rt.t("cancel"), data=f"{PREFIX}:o:{src_id}")])
    await ctx.edit("topics_merge_pick", buttons=rows, name=html_escape(src.name))


async def _merge_confirm(ctx: Ctx, src_id: int, dst_id: int) -> None:
    src, dst = await _active(ctx, src_id), await _active(ctx, dst_id)
    if src is None or dst is None or src_id == dst_id:
        return
    buttons = [
        [
            Button(ctx.rt.t("topics_merge_button"), data=f"{PREFIX}:md:{src_id}:{dst_id}"),
            Button(ctx.rt.t("cancel"), data=f"{PREFIX}:o:{src_id}"),
        ]
    ]
    await ctx.edit(
        "topics_merge_confirm",
        buttons=buttons,
        src=html_escape(src.name),
        dst=html_escape(dst.name),
        channel=await _channel_label(ctx.rt, src),
    )


async def _merge_do(ctx: Ctx, src_id: int, dst_id: int) -> None:
    src, dst = await _active(ctx, src_id), await _active(ctx, dst_id)
    if src is None or dst is None:
        return
    await _service(ctx.rt).merge(src.key, dst.key)
    await ctx.edit("topics_merged", src=html_escape(src.name), dst=html_escape(dst.name))


async def _remove_confirm(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    buttons = [
        [
            Button(ctx.rt.t("remove"), data=f"{PREFIX}:rd:{topic_id}"),
            Button(ctx.rt.t("cancel"), data=f"{PREFIX}:o:{topic_id}"),
        ]
    ]
    await ctx.edit(
        "topics_remove_confirm",
        buttons=buttons,
        name=html_escape(topic.name),
        channel=await _channel_label(ctx.rt, topic),
    )


async def _remove_do(ctx: Ctx, topic_id: int) -> None:
    topic = await _active(ctx, topic_id)
    if topic is None:
        return
    channel = await _channel_label(ctx.rt, topic)
    await _service(ctx.rt).remove(topic.key)
    key = "topics_removed" if topic.channel_id is not None else "topics_removed_no_channel"
    await ctx.edit(key, name=html_escape(topic.name), channel=channel)


_CALLBACKS: dict[str, tuple[Callable[..., Awaitable[None]], int]] = {
    "add": (_add, 0),
    "ac": (_add_category, 1),
    "ap": (_add_page, 1),
    "ad": (_add_description, 0),
    "af": (_add_done, 0),
    "an": (_add_channel_new, 0),
    "al": (_add_channel_link, 0),
    "a0": (_add_channel_none, 0),
    "o": (_open, 1),
    "k": (_keep, 1),
    "e": (_edit_menu, 1),
    "en": (_edit_name, 1),
    "ed": (_edit_description, 1),
    "ec": (_edit_category_page, 2),
    "es": (_edit_category_set, 2),
    "ex": (_edit_category_clear, 1),
    "et": (_edit_strictness, 1),
    "ep": (_strictness_proposed, 2),
    "eq": (_strictness_preview, 2),
    "ea": (_strictness_apply, 2),
    "ee": (_edit_examples, 1),
    "ev": (_edit_example_channel, 1),
    "ef": (_examples_done, 1),
    "el": (_channel_menu, 1),
    "cn": (_channel_create, 1),
    "cl": (_channel_link, 1),
    "m": (_merge_menu, 1),
    "mm": (_merge_confirm, 2),
    "md": (_merge_do, 2),
    "r": (_remove_confirm, 1),
    "rd": (_remove_do, 1),
}


# --- shared helpers --------------------------------------------------------------------------


def pairs(buttons: list[Button]) -> Buttons:
    """Rows of at most two buttons (topic and category names are long, §11.2)."""
    return [buttons[i : i + 2] for i in range(0, len(buttons), 2)]


def category_rows(
    rt: Runtime, page: int, *, pick: Callable[[int], str], turn: Callable[[int], str]
) -> Buttons:
    """One page of built-in categories (by index, two per row) plus ‹ › when paged."""
    every = categories.all()
    pages = max(1, -(-len(every) // CATEGORIES_PER_PAGE))
    page = min(max(page, 0), pages - 1)
    first = page * CATEGORIES_PER_PAGE
    shown = every[first : first + CATEGORIES_PER_PAGE]
    rows = pairs([Button(c.label, data=pick(first + i)) for i, c in enumerate(shown)])
    nav: list[Button] = []
    if page > 0:
        nav.append(Button(rt.t("topics_page_prev"), data=turn(page - 1)))
    if page < pages - 1:
        nav.append(Button(rt.t("topics_page_next"), data=turn(page + 1)))
    if nav:
        rows.append(nav)
    return rows


def _category_at(index: int) -> Category | None:
    every = categories.all()
    return every[index] if 0 <= index < len(every) else None


def _category_label(rt: Runtime, key: str | None) -> str:
    if not key:
        return rt.t("topics_none")
    try:
        return html_escape(categories.label(key))
    except KeyError:
        return html_escape(key)


def _strictness_label(rt: Runtime, value: float | None) -> str:
    if not value:
        return rt.t("topics_strictness_default", value=f"{rt.settings.sorting.confidence:.2f}")
    return f"{value:.2f}"


async def _channel_label(rt: Runtime, topic: Topic) -> str:
    if topic.channel_id is None:
        return rt.t("topics_no_channel")
    chat = await rt.store.get_chat(topic.channel_id)
    return html_escape(chat.title) if chat is not None else str(topic.channel_id)


def _is_channel_ref(text: str) -> bool:
    """``@name`` or a t.me link, alone on the line: an example channel, not an example post."""
    if not text or any(ch.isspace() for ch in text):
        return False
    return text.startswith("@") or "t.me/" in text or "telegram.me/" in text


async def _reply_exists(ctx: Ctx, name: str) -> None:
    """``TopicExists`` -> "edit the existing one?" with a button when the row is known."""
    wanted = name.casefold()
    existing = next(
        (t for t in await ctx.rt.store.list_topics(active=True) if t.name.casefold() == wanted),
        None,
    )
    buttons = None
    if existing is not None:
        buttons = [
            [Button(ctx.rt.t("topics_edit_existing_button"), data=f"{PREFIX}:e:{existing.id}")]
        ]
    await ctx.reply("topics_exists", buttons=buttons, name=html_escape(name))


async def _active(ctx: Ctx, topic_id: int) -> Topic | None:
    topic = await ctx.rt.store.get_topic(topic_id)
    if topic is None or not topic.active:
        await ctx.reply("unknown_choice")
        return None
    return topic


def _service(rt: Runtime) -> TopicsService:
    if rt.topics is None:
        raise RuntimeError("the topics service is not wired")
    return rt.topics
