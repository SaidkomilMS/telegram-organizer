"""``/settings``: the tunable keys from the bot, in sync with the settings file (DESIGN §11.2).

Every change goes through ``SettingsFile`` so the bot and the file never disagree and the
user's comments survive. The keys that decide what gets sorted and what goes out at once —
the ``[sorting]`` section, a topic's own strictness and a trusted source's trust — are shown
as a proposal first: the preview replays the last few days with the proposed value next to
the current counts, and only [Apply] writes the file. Everything else cannot change a sorting
decision, is cheap to undo and applies immediately.
Validation is the settings file's own, so a bad value is answered with the same sentence the
file would produce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from tg_curator.bot.core import BotApp, Ctx, Flow
from tg_curator.config import WEEKDAYS, Settings, validate_settings
from tg_curator.domain import PreviewReport, Topic
from tg_curator.errors import ConfigError, CuratorError
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime

PREFIX = "st"
FLOW = "settings"
PREVIEW_DAYS = 3


@dataclass(frozen=True)
class Item:
    """One key of the menu. ``label``/``hint`` are catalogue keys; ``parse`` turns the typed
    answer into the value (a number unless ``text``); ``preview`` marks the keys that are
    replayed before they apply (§11.2)."""

    key: str
    label: str
    hint: str
    parse: Callable[[str], Any]
    preview: bool = False
    text: bool = False


def _float(text: str) -> float:
    return float(text.replace(",", "."))


ITEMS: tuple[Item, ...] = (
    # The zone the digest and review hours are local to: settable from the bot, so a user who
    # sets up only through the bot does not get the 21:00 digest at 02:00 (SPEC "a day").
    Item(
        "general.timezone",
        "settings_label_timezone",
        "settings_hint_timezone",
        str.strip,
        text=True,
    ),
    Item("digest.hour", "settings_label_digest_hour", "settings_hint_digest_hour", int),
    Item("digest.minute", "settings_label_digest_minute", "settings_hint_digest_minute", int),
    Item("digest.items", "settings_label_digest_items", "settings_hint_digest_items", int),
    Item("digest.line_chars", "settings_label_digest_line_chars", "settings_hint_line_chars", int),
    Item(
        "sorting.confidence",
        "settings_label_confidence",
        "settings_hint_confidence",
        _float,
        preview=True,
    ),
    Item(
        "sorting.realtime_strength",
        "settings_label_realtime_strength",
        "settings_hint_realtime_strength",
        _float,
        preview=True,
    ),
    Item(
        "sorting.duplicate_similarity",
        "settings_label_duplicate_similarity",
        "settings_hint_duplicate_similarity",
        _float,
        preview=True,
    ),
    # Not previewed: a replay of stored posts promotes a corroborated post until its digest
    # is composed, whatever the hold, so the Preview would always show the same numbers. It
    # only moves *when* a lone post joins the digest pool; it applies at once (the hint says so).
    Item(
        "sorting.hold_minutes",
        "settings_label_hold_minutes",
        "settings_hint_hold_minutes",
        int,
    ),
    Item("groups.min_chars", "settings_label_min_chars", "settings_hint_min_chars", int),
    Item("review.hour", "settings_label_review_hour", "settings_hint_review_hour", int),
    Item(
        "review.leaves_per_day",
        "settings_label_leaves_per_day",
        "settings_hint_leaves_per_day",
        int,
    ),
    Item(
        "review.leave_interval_minutes",
        "settings_label_leave_interval",
        "settings_hint_leave_interval",
        int,
    ),
)
"""The menu, addressed by index in callback data (``st:k:<i>``, §11.1)."""

WEEKDAY_LABELS = (
    "settings_weekday_monday",
    "settings_weekday_tuesday",
    "settings_weekday_wednesday",
    "settings_weekday_thursday",
    "settings_weekday_friday",
    "settings_weekday_saturday",
    "settings_weekday_sunday",
)
TRUST_LABELS = ("settings_trust_0", "settings_trust_1", "settings_trust_2", "settings_trust_3")


def current_value(settings: Settings, dotted: str) -> Any:
    section, name = dotted.split(".")
    return getattr(getattr(settings, section), name)


def _shown(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


# --- the flow --------------------------------------------------------------------------------


class SettingsFlow(Flow):
    """Waiting for a typed value, a proposal's decision, or a trusted source to add."""

    first_step = "value"
    menu: ClassVar[SettingsMenu]

    @Flow.step("value")
    async def on_value(self, text: str) -> None:
        await self.menu.value(self, text)

    @Flow.step("strictness")
    async def on_strictness(self, text: str) -> None:
        await self.menu.value(self, text)

    @Flow.step("confirm")
    async def on_confirm(self, text: str) -> None:
        await self.menu.value(self, text)  # a new value typed instead of a button: re-propose

    @Flow.step("source")
    async def on_source(self, text: str) -> None:
        await self.menu.source(self, text)

    @Flow.step("trust")
    async def on_trust(self, text: str) -> None:
        await self.ctx.reply("settings_pick_button")


class SettingsMenu:
    """Handlers of ``/settings`` and its ``st:`` buttons."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt

    # --- /settings ---

    async def command(self, ctx: Ctx, args: str) -> None:
        settings = self.rt.settings
        t = self.rt.t
        pairs = [(t(item.label), _shown(current_value(settings, item.key))) for item in ITEMS]
        pairs += [
            (t("settings_label_style"), t(_style_label(settings.publishing.style))),
            (t("settings_label_weekday"), t(WEEKDAY_LABELS[settings.review.weekday_index])),
            (t("settings_label_sources"), str(len(settings.sources))),
        ]
        lines = [t("settings_menu_line", label=label, value=value) for label, value in pairs]
        buttons: Buttons = [
            [Button(t(item.label), f"{PREFIX}:k:{i}") for i, item in enumerate(ITEMS[j : j + 2], j)]
            for j in range(0, len(ITEMS), 2)
        ]
        buttons.append(
            [
                Button(t("settings_label_style"), f"{PREFIX}:sy"),
                Button(t("settings_label_weekday"), f"{PREFIX}:wd"),
            ]
        )
        buttons.append(
            [
                Button(t("settings_label_strictness"), f"{PREFIX}:ts"),
                Button(t("settings_label_sources"), f"{PREFIX}:src"),
            ]
        )
        await ctx.reply("settings_menu", buttons=buttons, lines="\n".join(lines))

    # --- the buttons ---

    async def callback(self, ctx: Ctx, data: str) -> None:
        action, _, arg = data.partition(":")
        handlers: dict[str, Callable[[Ctx, str], Any]] = {
            "k": self._edit,
            "pv": self._preview,
            "ap": self._apply,
            "cx": self._cancel,
            "sy": self._style,
            "wd": self._weekday,
            "ts": self._strictness,
            "src": self._sources,
            "sa": self._add_source,
            "sr": self._remove_source,
            "tr": self._trust,
        }
        handler = handlers.get(action)
        if handler is None:
            await ctx.reply("unknown_choice")
            return
        await handler(ctx, arg)

    async def _flow(self, ctx: Ctx, *steps: str) -> SettingsFlow | None:
        flow = await ctx.flow()
        if isinstance(flow, SettingsFlow) and flow.current_step in steps:
            return flow
        await ctx.reply("unknown_choice")
        return None

    # --- plain keys ---

    async def _edit(self, ctx: Ctx, arg: str) -> None:
        index = _index(arg, len(ITEMS))
        if index is None:
            await ctx.reply("unknown_choice")
            return
        item = ITEMS[index]
        flow = await ctx.start_flow(FLOW, item=index)
        await flow.go("value")
        await ctx.reply(
            "settings_value_prompt",
            label=self.rt.t(item.label),
            current=_shown(current_value(self.rt.settings, item.key)),
            hint=self.rt.t(item.hint),
        )

    async def value(self, flow: SettingsFlow, text: str) -> None:
        """A typed value: parsed, validated like the file, then applied or proposed."""
        topic = await self._flow_topic(flow)
        item = None if topic is not None else _item(flow.data.get("item"))
        if topic is None and item is None:
            await flow.end()
            await flow.ctx.reply("unknown_choice")
            return
        try:
            value = (item.parse if item is not None else _float)(text.strip())
        except ValueError:
            await flow.ctx.reply("settings_not_number")
            return
        try:
            if topic is not None:
                _validated_strictness(self.rt.settings, topic.key, value)
            else:
                assert item is not None
                _validated(self.rt.settings, item.key, value)
        except ConfigError as exc:
            await flow.ctx.reply("settings_invalid", error=html_escape(str(exc)))
            return
        if item is not None and not item.preview:
            await self.rt.settings_file.set_value(item.key, value)
            await flow.end()
            await flow.ctx.reply("settings_saved", label=self.rt.t(item.label), value=_shown(value))
            return
        await flow.go("confirm", value=value)
        label, current = self._describe(item, topic)
        await flow.ctx.reply(
            "settings_propose",
            buttons=_decision_buttons(self.rt, preview=True),
            label=label,
            current=current,
            value=_shown(value),
            days=PREVIEW_DAYS,
        )

    def _describe(self, item: Item | None, topic: Topic | None) -> tuple[str, str]:
        """The label and current value of what a proposal changes."""
        if topic is not None:
            label = self.rt.t("settings_strictness_label", topic=html_escape(topic.name))
            return label, self._strictness_text(topic)
        assert item is not None
        return self.rt.t(item.label), _shown(current_value(self.rt.settings, item.key))

    def _strictness_text(self, topic: Topic) -> str:
        if topic.strictness:
            return _shown(topic.strictness)
        return self.rt.t(
            "settings_strictness_general", confidence=_shown(self.rt.settings.sorting.confidence)
        )

    # --- proposals: preview / apply / cancel ---

    async def _preview(self, ctx: Ctx, arg: str) -> None:
        flow = await self._flow(ctx, "confirm")
        if flow is None:
            return
        override = await self._override(flow)
        if override is None:
            return
        key, value = override
        preview = self.rt.preview
        if preview is None:
            raise ConfigError("the preview is not running in this process")
        await ctx.reply("working")
        now = await preview.replay(days=PREVIEW_DAYS)
        then = await preview.replay(days=PREVIEW_DAYS, overrides={key: value})
        label, current, shown = await self._proposal(flow)
        await ctx.reply(
            "settings_preview",
            buttons=_decision_buttons(self.rt, preview=False),
            days=PREVIEW_DAYS,
            label=label,
            current=current,
            value=shown,
            lines=await self._comparison(now, then),
        )

    async def _proposal(self, flow: SettingsFlow) -> tuple[str, str, str]:
        """Label, current value and proposed value of the pending proposal, as shown."""
        trust = flow.data.get("trust")
        if isinstance(trust, int):
            return self._trust_description(flow.data, trust)
        topic = await self._flow_topic(flow)
        label, current = self._describe(_item(flow.data.get("item")), topic)
        return label, current, _shown(flow.data.get("value"))

    def _trust_description(self, data: dict[str, Any], trust: int) -> tuple[str, str, str]:
        t = self.rt.t
        chat_id = data.get("chat_id")
        title = html_escape(str(data.get("title", chat_id)))
        now = next((s.trust for s in self.rt.settings.sources if s.chat == chat_id), None)
        current = t(TRUST_LABELS[now]) if now is not None else t("settings_trust_unlisted")
        return t("settings_trust_label", title=title), current, t(TRUST_LABELS[trust])

    async def _comparison(self, now: PreviewReport, then: PreviewReport) -> str:
        names = {t.key: t.name for t in await self.rt.store.list_topics(active=True)}
        t = self.rt.t
        lines = [
            t(
                "settings_preview_line",
                name=html_escape(names.get(key, key)),
                current=now.per_topic.get(key, 0),
                proposed=then.per_topic.get(key, 0),
            )
            for key in sorted(set(now.per_topic) | set(then.per_topic))
        ]
        lines.append(
            t(
                "settings_preview_line",
                name=t("settings_preview_repeats"),
                current=now.repeats,
                proposed=then.repeats,
            )
        )
        lines.append(
            t(
                "settings_preview_line",
                name=t("settings_preview_unsorted"),
                current=now.unsorted,
                proposed=then.unsorted,
            )
        )
        realtime_now = sum(1 for d in now.decisions if d.would_realtime and d.topic_id)
        realtime_then = sum(1 for d in then.decisions if d.would_realtime and d.topic_id)
        lines.append(
            t(
                "settings_preview_line",
                name=t("settings_preview_realtime"),
                current=realtime_now,
                proposed=realtime_then,
            )
        )
        return "\n".join(lines)

    async def _apply(self, ctx: Ctx, arg: str) -> None:
        flow = await self._flow(ctx, "confirm")
        if flow is None:
            return
        override = await self._override(flow)
        if override is None:
            return
        key, value = override
        if isinstance(flow.data.get("trust"), int):
            await self._save_trust(ctx, flow)
            return
        topic = await self._flow_topic(flow)
        if topic is not None:
            topics = self.rt.topics
            if topics is None:
                raise ConfigError("the topics service is not running in this process")
            await topics.update(topic.key, strictness=value)
        else:
            await self.rt.settings_file.set_value(key, value)
        await flow.end()
        label, _ = self._describe(_item(flow.data.get("item")), topic)
        await ctx.reply("settings_saved", label=label, value=_shown(value))

    async def _cancel(self, ctx: Ctx, arg: str) -> None:
        await ctx.end_flow()
        await ctx.reply("flow_cancelled")

    async def _override(self, flow: SettingsFlow) -> tuple[str, Any] | None:
        """The dotted key and value of the pending proposal (§8 ``replay(overrides=…)``)."""
        trust, chat_id = flow.data.get("trust"), flow.data.get("chat_id")
        if isinstance(trust, int) and isinstance(chat_id, int):
            return f"sources.{chat_id}.trust", trust
        value = flow.data.get("value")
        topic = await self._flow_topic(flow)
        if topic is not None:
            return f"topics.{topic.key}.strictness", value
        item = _item(flow.data.get("item"))
        if item is None or value is None:
            await flow.end()
            await flow.ctx.reply("unknown_choice")
            return None
        return item.key, value

    async def _flow_topic(self, flow: SettingsFlow) -> Topic | None:
        topic_id = flow.data.get("topic_id")
        if not isinstance(topic_id, int):
            return None
        topic = await self.rt.store.get_topic(topic_id)
        return topic if topic is not None and topic.active else None

    # --- per-topic strictness ---

    async def _strictness(self, ctx: Ctx, arg: str) -> None:
        topics = await self.rt.store.list_topics(active=True)
        if not arg:
            if not topics:
                await ctx.reply("settings_no_topics")
                return
            buttons = [
                [Button(t.name, f"{PREFIX}:ts:{t.id}") for t in topics[j : j + 2]]
                for j in range(0, len(topics), 2)
            ]
            await ctx.reply("settings_strictness_pick", buttons=buttons)
            return
        topic = next((t for t in topics if str(t.id) == arg), None)
        if topic is None:
            await ctx.reply("unknown_choice")
            return
        flow = await ctx.start_flow(FLOW, topic_id=topic.id)
        await flow.go("strictness")
        await ctx.reply(
            "settings_strictness_prompt",
            topic=html_escape(topic.name),
            current=self._strictness_text(topic),
        )

    # --- posting style ---

    async def _style(self, ctx: Ctx, arg: str) -> None:
        if not arg:
            style = self.rt.settings.publishing.style
            buttons = [
                [
                    Button(self.rt.t("settings_style_repost"), f"{PREFIX}:sy:0"),
                    Button(self.rt.t("settings_style_forward"), f"{PREFIX}:sy:1"),
                ]
            ]
            await ctx.reply(
                "settings_style_prompt", buttons=buttons, current=self.rt.t(_style_label(style))
            )
            return
        if arg == "1":
            await self.rt.settings_file.set_value("publishing.style", "forward")
            await ctx.reply("settings_style_forward_saved")
            return
        if arg != "0":
            await ctx.reply("unknown_choice")
            return
        await self.rt.settings_file.set_value("publishing.style", "repost")
        if not self.rt.settings.publishing.live:
            await ctx.reply("settings_style_repost_saved")
            return
        topics = self.rt.topics
        if topics is None:
            raise ConfigError("the topics service is not running in this process")
        await topics.ensure_staging_channel()  # repost hands media over through it (§14.2)
        await ctx.reply("settings_style_repost_live")

    # --- review day ---

    async def _weekday(self, ctx: Ctx, arg: str) -> None:
        if not arg:
            labels = [
                Button(self.rt.t(k), f"{PREFIX}:wd:{i}") for i, k in enumerate(WEEKDAY_LABELS)
            ]
            buttons = [labels[j : j + 4] for j in range(0, len(labels), 4)]
            await ctx.reply("settings_weekday_prompt", buttons=buttons)
            return
        index = _index(arg, len(WEEKDAYS))
        if index is None:
            await ctx.reply("unknown_choice")
            return
        await self.rt.settings_file.set_value("review.weekday", WEEKDAYS[index])
        await ctx.reply(
            "settings_saved",
            label=self.rt.t("settings_label_weekday"),
            value=self.rt.t(WEEKDAY_LABELS[index]),
        )

    # --- trusted sources ---

    async def _sources(self, ctx: Ctx, arg: str) -> None:
        sources = self.rt.settings.sources
        lines = []
        buttons: Buttons = [[Button(self.rt.t("settings_source_add"), f"{PREFIX}:sa")]]
        for i, src in enumerate(sources):
            title = await self._source_title(src.chat)
            lines.append(
                self.rt.t(
                    "settings_source_line",
                    title=html_escape(title),
                    trust=self.rt.t(TRUST_LABELS[src.trust]),
                )
            )
            buttons.append(
                [Button(self.rt.t("settings_source_remove", title=title), f"{PREFIX}:sr:{i}")]
            )
        listing = "\n".join(lines) if lines else self.rt.t("settings_sources_none")
        await ctx.reply("settings_sources", buttons=buttons, lines=listing)

    async def _source_title(self, chat: int | str) -> str:
        if isinstance(chat, int):
            row = await self.rt.store.get_chat(chat)
            return row.title if row is not None else str(chat)
        return chat

    async def _add_source(self, ctx: Ctx, arg: str) -> None:
        flow = await ctx.start_flow(FLOW)
        await flow.go("source")
        await ctx.reply("settings_source_prompt")

    async def source(self, flow: SettingsFlow, text: str) -> None:
        ref = text.strip()
        if self.rt.user is None:
            await flow.ctx.reply("settings_source_no_account")
            return
        info = await self.rt.user.resolve_chat(ref)  # ChatGone / not a member: the core says so
        await flow.go("trust", chat_id=info.id, ref=ref, title=info.title)
        buttons = [[Button(self.rt.t(k), f"{PREFIX}:tr:{i}")] for i, k in enumerate(TRUST_LABELS)]
        await flow.ctx.reply(
            "settings_trust_prompt", buttons=buttons, title=html_escape(info.title)
        )

    async def _trust(self, ctx: Ctx, arg: str) -> None:
        flow = await self._flow(ctx, "trust")
        if flow is None:
            return
        trust = _index(arg, len(TRUST_LABELS))
        chat_id = flow.data.get("chat_id")
        if trust is None or not isinstance(chat_id, int):
            await ctx.reply("unknown_choice")
            return
        # Trust decides what goes out at once (trust 3 is always immediate, 0 digest only):
        # a proposal with [Preview] like the sorting keys, written only on [Apply].
        await flow.go("confirm", trust=trust)
        label, current, shown = self._trust_description(flow.data, trust)
        await ctx.reply(
            "settings_propose",
            buttons=_decision_buttons(self.rt, preview=True),
            label=label,
            current=current,
            value=shown,
            days=PREVIEW_DAYS,
        )

    async def _save_trust(self, ctx: Ctx, flow: SettingsFlow) -> None:
        trust, chat_id = flow.data.get("trust"), flow.data.get("chat_id")
        assert isinstance(trust, int) and isinstance(chat_id, int)
        settings_file = self.rt.settings_file
        for ref in await self._other_refs_of(chat_id):
            await settings_file.remove_source(ref)  # one entry per chat, written as its id
        await settings_file.upsert_source(chat_id, trust)
        unresolved = await self._sync()
        await flow.end()
        await ctx.reply(
            "settings_source_added",
            title=html_escape(str(flow.data.get("title", chat_id))),
            trust=self.rt.t(TRUST_LABELS[trust]),
            unresolved=unresolved,
        )

    async def _other_refs_of(self, chat_id: int) -> list[str]:
        """``[[sources]]`` entries naming ``chat_id`` by @username or link (a hand edit, or one
        the sync could not rewrite yet); they would become a second entry for the same chat."""
        refs = []
        for src in self.rt.settings.sources:
            if isinstance(src.chat, str) and self.rt.user is not None:
                try:
                    same = (await self.rt.user.resolve_chat(src.chat)).id == chat_id
                except CuratorError:
                    same = False
                if same:
                    refs.append(src.chat)
        return refs

    async def _remove_source(self, ctx: Ctx, arg: str) -> None:
        sources = self.rt.settings.sources
        index = _index(arg, len(sources))
        if index is None:
            await ctx.reply("unknown_choice")
            return
        chat = sources[index].chat
        title = await self._source_title(chat)
        await self.rt.settings_file.remove_source(chat)
        unresolved = await self._sync()
        await ctx.reply("settings_source_removed", title=html_escape(title), unresolved=unresolved)

    async def _sync(self) -> str:
        """Mirror ``[[sources]]`` into ``chats.trust`` now (§8 ``sync_from_settings``) and
        return the lines it could not resolve, ready to append."""
        topics = self.rt.topics
        if topics is None:
            raise ConfigError("the topics service is not running in this process")
        result = await topics.sync_from_settings()
        return "".join(f"\n{html_escape(line)}" for line in result.unresolved)


# --- helpers ---------------------------------------------------------------------------------


def _decision_buttons(rt: Runtime, *, preview: bool) -> Buttons:
    row = [
        Button(rt.t("settings_apply_button"), f"{PREFIX}:ap"),
        Button(rt.t("cancel"), f"{PREFIX}:cx"),
    ]
    if preview:
        row.insert(0, Button(rt.t("settings_preview_button"), f"{PREFIX}:pv"))
    return [row]


def _validated(settings: Settings, dotted: str, value: Any) -> Settings:
    """Raise the settings file's own ``ConfigError`` sentence when ``value`` is not allowed."""
    data = settings.model_dump()
    section, name = dotted.split(".")
    data[section][name] = value
    return validate_settings(data)


def _validated_strictness(settings: Settings, key: str, value: float) -> Settings:
    data = settings.model_dump()
    for entry in data["topics"]:
        if entry["key"] == key:
            entry["strictness"] = value
    return validate_settings(data)


def _item(index: Any) -> Item | None:
    return ITEMS[index] if isinstance(index, int) and 0 <= index < len(ITEMS) else None


def _index(arg: str, size: int) -> int | None:
    return int(arg) if arg.isdigit() and int(arg) < size else None


def _style_label(style: str) -> str:
    return "settings_style_forward" if style == "forward" else "settings_style_repost"


def register(app: BotApp) -> SettingsMenu:
    """Wire ``/settings``, the ``st:`` buttons and the ``settings`` flow into ``app``."""
    menu = SettingsMenu(app.rt)
    app.command("settings", menu.command, help_key="settings_help_settings")
    app.callback(PREFIX, menu.callback)
    app.flow(FLOW, type("BoundSettingsFlow", (SettingsFlow,), {"menu": menu}))
    return menu
