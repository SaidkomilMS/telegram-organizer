"""bot/topics.py: /topics list, the add conversation, edit, merge, remove (DESIGN §11.2)."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import (
    BOT_ACCOUNT,
    OWNER_ID,
    FakeBotGateway,
    FakeMessage,
    FakeUserGateway,
    FakeWorld,
    make_bot_message,
    make_callback,
)
from tg_curator.bot import corrections, preview, topics
from tg_curator.bot.core import BotApp
from tg_curator.domain import KV, Topic
from tg_curator.errors import FloodWait, NotAllowed
from tg_curator.i18n import Translator
from tg_curator.ml import categories
from tg_curator.pipeline.preview import PreviewService
from tg_curator.runtime import EVENT_EXAMPLES_CHANGED, Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage
from tg_curator.topics.learning import Learning
from tg_curator.topics.service import TopicsService

TEMPLATE_KEYS = ["ml-ai", "fintech", "uzbekistan", "football"]
CALLBACK_RE = re.compile(r"^(tp|wt|mv|pv)(:[a-z0-9]+)*$")
EXAMPLE_1 = "Bitcoin crossed a new all-time high as spot ETF inflows kept growing this week."
EXAMPLE_2 = "Ethereum developers scheduled the next network upgrade for the end of the month."


class Driver:
    """Types and taps as the owner and reads back what the bot sent to the private chat."""

    def __init__(self, rt: Runtime, bot: FakeBotGateway) -> None:
        self.rt = rt
        self.bot = bot
        self._next_id = 1000

    async def say(self, text: str, **kw: Any) -> None:
        self._next_id += 1
        await self.bot.say(make_bot_message(text, message_id=self._next_id, **kw))

    async def press(self, data: str, message: FakeMessage | int, **kw: Any) -> None:
        mid = message if isinstance(message, int) else message.message_id
        chat_id = kw.pop("chat_id", OWNER_ID)
        await self.bot.press(make_callback(data, message_id=mid, chat_id=chat_id, **kw))

    def sent(self) -> list[FakeMessage]:
        return self.bot.sent(OWNER_ID)

    def last(self) -> FakeMessage:
        return self.sent()[-1]

    def with_button(self, data: str) -> FakeMessage:
        """The newest owner-chat message carrying a button with exactly this payload."""
        for msg in reversed(self.sent()):
            if data in button_data(msg):
                return msg
        raise AssertionError(f"no message with button {data!r}")

    def texts_since(self, n: int) -> list[str]:
        return [m.text for m in self.sent()[n:]]

    async def conversation(self) -> Any:
        return await self.rt.store.kv_get(KV.BOT_CONVERSATION)


def button_data(msg: FakeMessage) -> list[str]:
    return [b.data for row in msg.buttons or [] for b in row if b.data is not None]


def button_text(msg: FakeMessage, data: str) -> str:
    for row in msg.buttons or []:
        for b in row:
            if b.data == data:
                return b.text
    raise AssertionError(f"no button {data!r}")


def wire(rt: Runtime) -> BotApp:
    """The real services of wave 1 plus the three bot modules on one BotApp."""
    rt.topics = TopicsService(rt)
    rt.learning = Learning(rt)
    rt.preview = PreviewService(rt)
    app = BotApp(rt)
    topics.register(app)
    corrections.register(app)
    preview.register(app)
    return app


@pytest.fixture
async def drv(rt: Runtime, bot_gw: FakeBotGateway) -> Driver:
    app = wire(rt)
    await app.start()
    assert rt.topics is not None
    await rt.topics.sync_from_settings()  # the four example topics of the template
    return Driver(rt, bot_gw)


async def topic_by_name(rt: Runtime, name: str) -> Topic:
    for topic in await rt.store.list_topics(active=None):
        if topic.name == name:
            return topic
    raise AssertionError(f"no topic {name!r}")


def settings_on_disk(rt: Runtime) -> dict[str, dict[str, Any]]:
    data = tomllib.loads(Path(rt.settings_file.path).read_text())
    return {t["key"]: t for t in data.get("topics", [])}


async def start_add(drv: Driver, name: str) -> Topic:
    await drv.say("/topics add")
    await drv.say(name)
    return await topic_by_name(drv.rt, name)


# --- the add conversation, end to end ---------------------------------------------------------


async def test_add_flow_creates_a_topic_with_category_examples_and_a_new_channel(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway, world: FakeWorld
) -> None:
    await drv.say("/topics add")
    assert "What should the new topic be called" in drv.last().text
    await drv.say("Crypto news")
    topic = await topic_by_name(rt, "Crypto news")
    prompt = drv.with_button("tp:af")
    assert "What is Crypto news about" in prompt.text

    crypto = categories.KEYS.index("crypto")
    await drv.press(f"tp:ac:{crypto}", prompt)
    assert "now: Crypto" in prompt.text  # the prompt was edited in place
    await drv.say(EXAMPLE_1, fwd_from_chat_id=-100555, fwd_from_title="Some channel")
    await drv.say(EXAMPLE_2)
    assert "2 so far" in drv.last().text

    await drv.press("tp:af", prompt)
    channel_prompt = drv.with_button("tp:an")
    assert "Where should" in channel_prompt.text
    assert not any("will not sort anything" in t for t in drv.texts_since(0))

    new_channel_id = user_gw._next_chat - 1
    world.bot_blocked.add(new_channel_id)  # a fresh channel: the bot is not an admin yet
    await drv.press("tp:an", channel_prompt)

    assert [c.title for c in user_gw.created] == ["Crypto news"]
    assert {"chat_id": new_channel_id} in user_gw.calls_of("register_owned")
    assert {"chat_id": new_channel_id, "bot_username": BOT_ACCOUNT.username} in user_gw.calls_of(
        "add_bot_admin"
    )
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == new_channel_id and row.category == "crypto"
    chat = await rt.store.get_chat(new_channel_id)
    assert chat is not None and chat.role == "output"
    examples = await rt.store.list_examples(topic_id=topic.id)
    assert sorted(e.text for e in examples) == sorted([EXAMPLE_1, EXAMPLE_2])
    assert {e.kind for e in examples} == {"example"}
    on_disk = settings_on_disk(rt)[topic.key]
    assert on_disk["name"] == "Crypto news"
    assert on_disk["channel"] == new_channel_id
    assert on_disk["category"] == "crypto"
    assert "is ready and posts into Crypto news" in drv.last().text
    assert await drv.conversation() is None


async def test_add_offers_the_untouched_example_topics_first(drv: Driver, rt: Runtime) -> None:
    await drv.say("/topics add")
    offers = [m for m in drv.sent() if any(d.startswith("tp:k:") for d in button_data(m))]
    names = [m.text.splitlines()[0] for m in offers]
    assert names == [t.name for t in rt.settings.topics]
    first = offers[0]
    topic = await rt.store.get_topic_by_key(TEMPLATE_KEYS[0])
    assert topic is not None
    assert button_data(first) == [f"tp:k:{topic.id}", f"tp:e:{topic.id}", f"tp:r:{topic.id}"]
    assert [button_text(first, d) for d in button_data(first)] == ["Keep", "Edit", "Remove"]

    await drv.press(f"tp:k:{topic.id}", first)
    assert "Kept" in first.text and first.buttons is None


async def test_touched_example_topics_are_not_offered(drv: Driver, rt: Runtime) -> None:
    assert rt.topics is not None
    await rt.topics.update("ml-ai", description="Only my favourite AI labs.")
    await rt.topics.add_examples("fintech", [EXAMPLE_1])
    await drv.say("/topics add")
    offered = {m.text.splitlines()[0] for m in drv.sent() if "tp:k:" in " ".join(button_data(m))}
    assert offered == {"Узбекистан: новости", "Футбол"}


async def test_removing_an_example_topic_from_the_offer(drv: Driver, rt: Runtime) -> None:
    await drv.say("/topics add")
    topic = await rt.store.get_topic_by_key("football")
    assert topic is not None
    offer = drv.with_button(f"tp:r:{topic.id}")
    await drv.press(f"tp:r:{topic.id}", offer)
    assert "not touched" in offer.text
    await drv.press(f"tp:rd:{topic.id}", offer)
    row = await rt.store.get_topic(topic.id)
    assert row is not None and not row.active
    assert "football" not in settings_on_disk(rt)
    assert "was removed" in offer.text


async def test_an_existing_name_offers_to_edit_the_existing_topic(drv: Driver, rt: Runtime) -> None:
    await drv.say("/topics add")
    await drv.say("fintech")  # case-insensitive match with "Fintech"
    topic = await rt.store.get_topic_by_key("fintech")
    assert topic is not None
    reply = drv.last()
    assert "already exists" in reply.text
    assert button_data(reply) == [f"tp:e:{topic.id}"]
    assert (await drv.conversation())["step"] == "name"
    assert len(await rt.store.list_topics(active=True)) == 4

    await drv.press(f"tp:e:{topic.id}", reply)
    assert f"tp:en:{topic.id}" in button_data(reply)


async def test_a_topic_with_neither_category_nor_examples_gets_the_warning(drv: Driver) -> None:
    await start_add(drv, "Misc")
    await drv.press("tp:ad", drv.with_button("tp:ad"))
    await drv.say("Anything I find interesting.")
    assert "Description saved" in drv.texts_since(-2)[0]
    await drv.press("tp:af", drv.with_button("tp:af"))
    assert any("will not sort anything" in t for t in drv.texts_since(-2))
    await drv.press("tp:a0", drv.with_button("tp:a0"))
    assert "tracked only" in drv.last().text
    assert await drv.conversation() is None


async def test_short_text_is_not_an_example(drv: Driver, rt: Runtime) -> None:
    topic = await start_add(drv, "Space")
    await drv.say("too short")
    assert "too short" in drv.last().text
    assert await rt.store.list_examples(topic_id=topic.id) == []


@pytest.mark.parametrize(
    "kw",
    [
        {"fwd_from_chat_id": -100555, "fwd_from_title": "Some channel"},
        {"fwd_from_title": "Hidden user"},
        {"has_media": True},
    ],
)
async def test_a_post_without_text_is_named_as_such_not_too_short(
    drv: Driver, rt: Runtime, kw: dict[str, Any]
) -> None:
    topic = await start_add(drv, "Space")
    await drv.say("", **kw)
    texts = [m.text for m in drv.sent()]
    assert any("no text the classifier can learn from" in t for t in texts)
    assert not any("too short" in t for t in texts)
    assert await rt.store.list_examples(topic_id=topic.id) == []


async def test_example_channel_is_read_and_a_private_one_needs_joining_first(
    drv: Driver,
    rt: Runtime,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> None:
    public = user_gw.add_chat(make_chat(username="spacenews", title="Space News"))
    for i in range(3):
        user_gw.seed(make_message(public, text=f"Rocket launch number {i} went well today."))
    topic = await start_add(drv, "Space")

    await drv.say("https://t.me/+secretinvite")
    assert drv.last().text == "Join the chat in Telegram first, then send the link again."
    await drv.say("@nosuchchannel")
    assert "Cannot find @nosuchchannel" in drv.last().text
    await drv.say("@spacenews")
    assert "Read 3 recent posts of @spacenews" in drv.last().text
    examples = await rt.store.list_examples(topic_id=topic.id)
    assert len(examples) == 3 and {e.kind for e in examples} == {"channel"}
    assert settings_on_disk(rt)[topic.key]["example_channel"] == "@spacenews"


async def test_category_pages(drv: Driver) -> None:
    await start_add(drv, "Space")
    prompt = drv.with_button("tp:af")
    first_page = [d for d in button_data(prompt) if d.startswith("tp:ac:")]
    assert first_page == [f"tp:ac:{i}" for i in range(topics.CATEGORIES_PER_PAGE)]
    assert all(len(row) <= 2 for row in prompt.buttons or [])
    await drv.press("tp:ap:1", prompt)
    second_page = [d for d in button_data(prompt) if d.startswith("tp:ac:")]
    assert second_page == [f"tp:ac:{i}" for i in range(10, len(categories.all()))]
    assert "tp:ap:0" in button_data(prompt)


# --- channel wordings (§11.2) ------------------------------------------------------------------


async def channel_step(drv: Driver, name: str) -> Topic:
    topic = await start_add(drv, name)
    await drv.press("tp:af", drv.with_button("tp:af"))
    return topic


async def test_link_channel_error_wordings(
    drv: Driver,
    rt: Runtime,
    user_gw: FakeUserGateway,
    world: FakeWorld,
    make_chat: Callable[..., ChatInfo],
) -> None:
    topic = await channel_step(drv, "Space")
    await drv.press("tp:al", drv.with_button("tp:al"))
    assert "Send the @username or the link" in drv.last().text

    await drv.say("@nope")
    assert drv.last().text == (
        "topic channel @nope cannot be found: check the link or that your account is in it"
    )
    user_gw.add_chat(make_chat(username="notmine", title="Not Mine"))
    await drv.say("@notmine")
    assert drv.last().text == (
        "channel Not Mine is not yours: the account must be its creator or an admin"
    )
    await drv.say("https://t.me/+private")
    assert drv.last().text == "Join the chat in Telegram first, then send the link again."

    mine = user_gw.add_chat(make_chat(username="mine", title="Mine", is_creator=True))
    world.bot_blocked.add(mine.id)
    user_gw.fail_next("add_bot_admin", NotAllowed("fresh_session"))
    await drv.say("@mine")
    assert drv.last().text == (
        f"the bot cannot post into Mine: add @{BOT_ACCOUNT.username} as an admin with Post Messages"
    )
    assert (await drv.conversation())["step"] == "channel"

    await drv.say("@mine")  # the owner fixed it (the fake's add_bot_admin now works)
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == mine.id
    assert settings_on_disk(rt)[topic.key]["channel"] == mine.id
    assert await drv.conversation() is None


async def test_flood_wait_on_channel_creation_saves_the_topic_without_channel(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway
) -> None:
    topic = await channel_step(drv, "Space")
    user_gw.fail_next("create_channel", FloodWait(600))
    await drv.press("tp:an", drv.with_button("tp:an"))
    assert drv.last().text == (
        "Telegram asks to wait 10 min before another channel can be created; "
        "the channel will be created automatically then, and I will let you know."
    )
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.active and row.channel_id is None
    assert settings_on_disk(rt)[topic.key]["channel"] == 0
    assert await drv.conversation() is None
    assert rt.topics is not None
    assert await rt.topics.wanted_channels() == [topic.key]  # type: ignore[attr-defined]

    await drv.say("/topics")
    card = drv.with_button(f"tp:cn:{topic.id}")
    assert button_text(card, f"tp:cn:{topic.id}") == "Create channel"
    await drv.press(f"tp:cn:{topic.id}", card)  # still inside Telegram's wait
    assert "Telegram asks to wait" in drv.last().text
    rt.clock.advance(601)  # type: ignore[attr-defined]
    await drv.press(f"tp:cn:{topic.id}", card)
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == user_gw.created[-1].id
    assert "now posts into Space" in drv.last().text


# --- /topics: list, edit, merge, remove -------------------------------------------------------


async def test_list_has_per_topic_buttons(drv: Driver, rt: Runtime) -> None:
    await drv.say("/topics")
    header = drv.sent()[0]
    assert button_data(header) == ["tp:add"]
    cards = drv.sent()[1:]
    assert len(cards) == 4
    for card, key in zip(cards, TEMPLATE_KEYS, strict=True):
        topic = await rt.store.get_topic_by_key(key)
        assert topic is not None
        assert button_data(card) == [
            f"tp:e:{topic.id}",
            f"tp:m:{topic.id}",
            f"tp:r:{topic.id}",
            f"tp:cn:{topic.id}",
        ]
        assert "none yet (tracked only)" in card.text


async def test_edit_name_description_category_and_strictness(drv: Driver, rt: Runtime) -> None:
    topic = await rt.store.get_topic_by_key("fintech")
    assert topic is not None
    await drv.say("/topics")
    card = drv.with_button(f"tp:e:{topic.id}")
    await drv.press(f"tp:e:{topic.id}", card)

    await drv.press(f"tp:en:{topic.id}", card)
    await drv.say("Payments")
    assert rt.settings.topic("fintech").name == "Payments"  # type: ignore[union-attr]
    assert drv.last().text.startswith("Payments")

    await drv.press(f"tp:ed:{topic.id}", card)
    await drv.say("-")
    assert settings_on_disk(rt)["fintech"].get("description", "") == ""

    await drv.press(f"tp:ec:{topic.id}:0", card)
    assert f"tp:ex:{topic.id}" in button_data(card)
    crypto = categories.KEYS.index("crypto")
    await drv.press(f"tp:es:{topic.id}:{crypto}", card)
    assert settings_on_disk(rt)["fintech"]["category"] == "crypto"
    assert "category: Crypto" in card.text

    await drv.press(f"tp:et:{topic.id}", card)
    assert f"tp:ep:{topic.id}:70" in button_data(card)
    await drv.press(f"tp:ep:{topic.id}:70", card)
    await drv.press(f"tp:eq:{topic.id}:70", card)
    assert "with strictness 0.70" in drv.last().text
    await drv.press(f"tp:ea:{topic.id}:70", card)
    assert settings_on_disk(rt)["fintech"]["strictness"] == 0.7
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.strictness == 0.7


async def test_merge_picks_the_destination_by_buttons(drv: Driver, rt: Runtime) -> None:
    src = await rt.store.get_topic_by_key("football")
    dst = await rt.store.get_topic_by_key("ml-ai")
    assert src is not None and dst is not None
    await drv.say("/topics")
    card = drv.with_button(f"tp:m:{src.id}")
    await drv.press(f"tp:m:{src.id}", card)
    choices = [d for d in button_data(card) if d.startswith("tp:mm:")]
    assert len(choices) == 3 and f"tp:mm:{src.id}:{dst.id}" in choices
    assert all(len(row) <= 2 for row in card.buttons or [])
    await drv.press(f"tp:mm:{src.id}:{dst.id}", card)
    assert "not touched" in card.text
    await drv.press(f"tp:md:{src.id}:{dst.id}", card)
    row = await rt.store.get_topic(src.id)
    assert row is not None and not row.active
    assert "football" not in settings_on_disk(rt)
    assert "was merged into" in card.text


async def test_remove_explains_the_channel_is_untouched(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    assert rt.topics is not None
    channel = user_gw.add_chat(make_chat(title="Ball", is_creator=True))
    await rt.topics.link_channel("football", channel.id)
    topic = await rt.store.get_topic_by_key("football")
    assert topic is not None
    await drv.say("/topics")
    card = drv.with_button(f"tp:r:{topic.id}")
    await drv.press(f"tp:r:{topic.id}", card)
    assert "Ball" in card.text and "not touched" in card.text
    await drv.press(f"tp:o:{topic.id}", card)  # cancel restores the card
    assert f"tp:e:{topic.id}" in button_data(card)
    await drv.press(f"tp:r:{topic.id}", card)
    await drv.press(f"tp:rd:{topic.id}", card)
    assert "Its channel Ball was not touched" in card.text
    assert (await rt.store.get_chat(channel.id)).role == "output"  # type: ignore[union-attr]
    assert channel.id in user_gw.owned


async def test_unknown_or_stale_callbacks_are_refused(drv: Driver) -> None:
    await drv.press("tp:e:9999", 1)
    assert drv.last().text == "That choice is not available any more."
    await drv.press("tp:zz:1", 1)
    await drv.press("tp:ac:1", 1)  # no add conversation running
    assert [m.text for m in drv.sent()[-2:]] == ["That choice is not available any more."] * 2


# --- contract checks ---------------------------------------------------------------------------


async def test_callback_data_is_ids_and_indexes_within_64_bytes(drv: Driver) -> None:
    await drv.say("/topics")
    await drv.say("/topics add")
    await drv.say("A topic with a rather long name to stress the buttons")
    for msg in drv.sent():
        for data in button_data(msg):
            assert CALLBACK_RE.match(data), data
            assert len(data.encode()) <= 64


@pytest.mark.parametrize("module", ["topics", "corrections", "preview"])
def test_every_catalogue_key_used_exists(module: str) -> None:
    source = (Path(topics.__file__).parent / f"{module}.py").read_text()
    used = set(re.findall(rf'"({module}_[a-z0-9_]+)"', source))
    used |= set(re.findall(r'\b(?:reply|edit|t)\(\s*"([a-z][a-z0-9_]*)"', source))
    english = Translator("en").english_keys()
    assert used and not used - english
    own = tomllib.loads(
        (Path(topics.__file__).parents[1] / "locales" / "en" / f"{module}.toml").read_text()
    )
    assert all(key.startswith(f"{module}_") for key in own)


async def test_old_prompts_of_another_step_are_refused(drv: Driver) -> None:
    await channel_step(drv, "Space")
    await drv.press("tp:ac:0", drv.with_button("tp:ac:0"))  # the "what is it" prompt is past
    assert drv.last().text == "That choice is not available any more."
    assert (await drv.conversation())["step"] == "channel"


async def test_edit_channel_by_link(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    channel = user_gw.add_chat(make_chat(username="mlchan", title="ML Channel", is_admin=True))
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    await drv.say("/topics")
    card = drv.with_button(f"tp:e:{topic.id}")
    await drv.press(f"tp:el:{topic.id}", card)
    assert button_data(card) == [f"tp:cn:{topic.id}", f"tp:cl:{topic.id}", f"tp:o:{topic.id}"]
    await drv.press(f"tp:cl:{topic.id}", card)
    await drv.say("https://t.me/mlchan")
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == channel.id
    assert settings_on_disk(rt)["ml-ai"]["channel"] == channel.id
    assert "channel: ML Channel" in drv.last().text
    assert channel.id in user_gw.owned


# --- more examples for an existing topic (spec "Tuning") -----------------------------------------


def record_examples_changed(rt: Runtime) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def on_examples(**payload: Any) -> None:
        seen.append(payload)

    rt.events.on(EVENT_EXAMPLES_CHANGED, on_examples)
    return seen


async def test_forward_more_examples_to_an_existing_topic(drv: Driver, rt: Runtime) -> None:
    topic = await rt.store.get_topic_by_key("fintech")
    assert topic is not None
    before = len(await rt.store.list_examples(topic_id=topic.id))
    events = record_examples_changed(rt)
    await drv.say("/topics")
    card = drv.with_button(f"tp:e:{topic.id}")
    await drv.press(f"tp:e:{topic.id}", card)
    assert f"tp:ee:{topic.id}" in button_data(card)
    assert f"tp:ev:{topic.id}" in button_data(card)
    assert button_text(card, f"tp:ee:{topic.id}") == "Add examples"

    await drv.press(f"tp:ee:{topic.id}", card)
    prompt = drv.with_button(f"tp:ef:{topic.id}")
    assert "Forward a few more posts" in prompt.text
    await drv.say(EXAMPLE_1, fwd_from_chat_id=-100555, fwd_from_title="Some channel")
    assert "1 so far" in drv.last().text
    examples = await rt.store.list_examples(topic_id=topic.id)
    assert len(examples) == before + 1
    assert EXAMPLE_1 in {e.text for e in examples}
    assert events == [{"reason": "examples"}]

    await drv.say("too short")  # stays in the step
    assert "too short" in drv.last().text
    await drv.say(EXAMPLE_2)  # pasted, not forwarded: an example too
    assert "2 so far" in drv.last().text
    assert len(await rt.store.list_examples(topic_id=topic.id)) == before + 2
    assert (await drv.conversation())["step"] == "examples"

    await drv.press(f"tp:ef:{topic.id}", prompt)
    assert await drv.conversation() is None
    assert f"tp:e:{topic.id}" in button_data(prompt)  # back on the topic card
    assert "Fintech" in prompt.text


async def test_examples_step_reads_a_channel_ref_as_an_example_channel(
    drv: Driver,
    rt: Runtime,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> None:
    public = user_gw.add_chat(make_chat(username="paynews", title="Pay News"))
    for i in range(2):
        user_gw.seed(make_message(public, text=f"Card payments story number {i} for today."))
    topic = await rt.store.get_topic_by_key("fintech")
    assert topic is not None
    events = record_examples_changed(rt)
    await drv.say("/topics")
    await drv.press(f"tp:ee:{topic.id}", drv.with_button(f"tp:e:{topic.id}"))
    await drv.say("@paynews")
    assert "Read 2 recent posts of @paynews" in drv.last().text
    examples = await rt.store.list_examples(topic_id=topic.id)
    assert {e.kind for e in examples} == {"channel"} and len(examples) == 2
    assert settings_on_disk(rt)["fintech"]["example_channel"] == "@paynews"
    assert events == [{"reason": "examples"}]


async def test_example_channel_button_sets_the_example_channel(
    drv: Driver,
    rt: Runtime,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> None:
    public = user_gw.add_chat(make_chat(username="ballnews", title="Ball News"))
    user_gw.seed(make_message(public, text="The derby ended two all after a late equaliser."))
    topic = await rt.store.get_topic_by_key("football")
    assert topic is not None
    await drv.say("/topics")
    card = drv.with_button(f"tp:e:{topic.id}")
    await drv.press(f"tp:ev:{topic.id}", card)
    assert drv.last().text.startswith("Example channel of Футбол now")
    await drv.say("A long sentence that is certainly not a channel reference.")
    assert "That is not a channel" in drv.last().text
    await drv.say("@nosuchchannel")
    assert "Cannot find @nosuchchannel" in drv.last().text
    assert (await drv.conversation())["step"] == "example_channel"
    await drv.say("https://t.me/ballnews")
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.example_channel == "https://t.me/ballnews"
    assert settings_on_disk(rt)["football"]["example_channel"] == "https://t.me/ballnews"
    assert len(await rt.store.list_examples(topic_id=topic.id)) == 1
    assert await drv.conversation() is None
    assert f"tp:e:{topic.id}" in button_data(drv.last())  # the card again


async def test_stale_examples_done_does_not_end_another_conversation(drv: Driver) -> None:
    topic = await start_add(drv, "Space")
    await drv.press(f"tp:ef:{topic.id}", 1)
    assert (await drv.conversation())["step"] == "what"
