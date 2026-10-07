"""bot/settings.py: immediate keys, preview-then-apply for sorting keys and topic strictness,
the posting-style wording, the review day and trusted sources mirrored by the topics sync."""

from __future__ import annotations

import inspect
import re
import tomllib
import zoneinfo
from collections.abc import Callable
from datetime import UTC, timedelta

import pytest

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeClassifier,
    FakeUserGateway,
    make_bot_message,
    make_callback,
    plain_text,
)
from tg_curator.bot import settings as settings_mod
from tg_curator.bot.core import BotApp
from tg_curator.domain import NewPost, PostStatus, Topic
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.pipeline.preview import PreviewService
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash
from tg_curator.topics.service import TopicsService

SOURCE = -1_001_000_000_301


def item_index(key: str) -> int:
    return next(i for i, item in enumerate(settings_mod.ITEMS) if item.key == key)


def file_text(rt: Runtime) -> str:
    return rt.settings_file.path.read_text(encoding="utf-8")


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


def last_buttons(bot_gw: FakeBotGateway) -> list[str]:
    msg = bot_gw.sent(OWNER_ID)[-1]
    return [b.data or "" for row in msg.buttons or [] for b in row]


@pytest.fixture
async def app(rt: Runtime) -> BotApp:
    rt.topics = TopicsService(rt)
    rt.preview = PreviewService(rt)
    await rt.topics.sync_from_settings()  # the four template topics become rows
    app = BotApp(rt)
    settings_mod.register(app)
    return app


async def ml_topic(rt: Runtime) -> Topic:
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    return topic


async def seed_posts(rt: Runtime, make_chat: Callable[..., ChatInfo], n: int = 3) -> None:
    """``n`` distinct posts of the last day from one source chat."""
    chat = make_chat(SOURCE, title="Source")
    await rt.store.upsert_chat(chat)
    for i in range(n):
        text = f"Story number {i} about completely different matter {i * 7} here"
        await rt.store.insert_post(
            NewPost(
                chat_id=SOURCE,
                message_id=i + 1,
                kind="post",
                message_ids=[i + 1],
                posted_at=rt.clock.now() - timedelta(hours=10 - i),
                via="live",
                text=text,
                text_hash=text_hash(text),
                urls=[],
            ),
            status=PostStatus.unsorted,
        )


# --- immediate keys --------------------------------------------------------------------------


async def test_menu_lists_the_keys_with_their_values(app: BotApp, bot_gw: FakeBotGateway) -> None:
    await bot_gw.say(make_bot_message("/settings"))
    text = owner_texts(bot_gw)[-1]
    assert "Digest hour: 21" in text
    assert "Sorting confidence: 0.5" in text
    assert "Posting style: Repost" in text
    assert "Review day: Sunday" in text
    assert "Trusted sources: 0" in text
    assert f"st:k:{item_index('digest.hour')}" in last_buttons(bot_gw)


async def test_a_plain_key_applies_immediately_and_bad_values_get_the_config_sentence(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback(f"st:k:{item_index('digest.hour')}"))
    assert "Digest hour — now 21" in owner_texts(bot_gw)[-1]

    await bot_gw.say(make_bot_message("25"))
    assert owner_texts(bot_gw)[-1].startswith("Not saved: settings: digest.hour = 25")
    assert rt.settings.digest.hour == 21

    await bot_gw.say(make_bot_message("soon"))
    assert "not a number" in owner_texts(bot_gw)[-1]

    await bot_gw.say(make_bot_message("20"))
    assert owner_texts(bot_gw)[-1] == "Saved: Digest hour = 20."
    assert rt.settings.digest.hour == 20
    assert re.search(r"(?m)^hour = 20", file_text(rt))
    assert await rt.store.kv_get("bot.conversation") is None


async def test_the_timezone_is_set_from_the_bot_and_the_digest_follows_it(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    from tg_curator.pipeline.digest import DigestService

    await bot_gw.say(make_bot_message("/settings"))
    assert "Timezone: UTC" in owner_texts(bot_gw)[-1]
    assert f"st:k:{item_index('general.timezone')}" in last_buttons(bot_gw)
    await bot_gw.press(make_callback(f"st:k:{item_index('general.timezone')}"))
    prompt = owner_texts(bot_gw)[-1]
    assert "Timezone — now UTC" in prompt and "IANA" in prompt

    await bot_gw.say(make_bot_message("Mars/Olympus"))
    assert "is not an IANA timezone name" in owner_texts(bot_gw)[-1]
    assert rt.settings.general.timezone == "UTC"

    await bot_gw.say(make_bot_message("  Asia/Tashkent "))
    assert owner_texts(bot_gw)[-1] == "Saved: Timezone = Asia/Tashkent."
    assert rt.settings.general.timezone == "Asia/Tashkent"
    assert re.search(r'(?m)^timezone = "Asia/Tashkent"', file_text(rt))
    # 21:00 in Tashkent (UTC+5) is 16:00 UTC: the digest arrives in the owner's evening
    nxt = DigestService(rt).next_run()
    assert nxt.astimezone(zoneinfo.ZoneInfo("Asia/Tashkent")).hour == 21
    assert nxt.utcoffset() is None or nxt.astimezone(UTC).hour == 16


async def test_hour_hints_say_they_are_in_the_chosen_timezone(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback(f"st:k:{item_index('digest.hour')}"))
    assert "in your timezone (see Timezone)" in owner_texts(bot_gw)[-1]


# --- preview, then apply ---------------------------------------------------------------------


async def test_confidence_is_previewed_and_written_only_on_apply(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> None:
    await seed_posts(rt, make_chat)
    topic = await ml_topic(rt)
    assert isinstance(rt.classifier, FakeClassifier)
    rt.classifier.scores = {topic.id: 0.6}
    before = file_text(rt)

    await bot_gw.press(make_callback(f"st:k:{item_index('sorting.confidence')}"))
    await bot_gw.say(make_bot_message("0.7"))
    assert "Sorting confidence: 0.5 → 0.7" in owner_texts(bot_gw)[-1]
    assert last_buttons(bot_gw) == ["st:pv", "st:ap", "st:cx"]
    assert file_text(rt) == before

    await bot_gw.press(make_callback("st:pv"))
    preview = owner_texts(bot_gw)[-1]
    assert "ML & AI: 3 → 0" in preview
    assert "unsorted: 0 → 3" in preview
    assert last_buttons(bot_gw) == ["st:ap", "st:cx"]
    assert file_text(rt) == before
    assert rt.settings.sorting.confidence == 0.5

    await bot_gw.press(make_callback("st:ap"))
    assert owner_texts(bot_gw)[-1] == "Saved: Sorting confidence = 0.7."
    assert rt.settings.sorting.confidence == 0.7
    assert re.search(r"(?m)^confidence = 0.7", file_text(rt))


async def test_cancel_and_invalid_proposals_write_nothing(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    before = file_text(rt)
    await bot_gw.press(make_callback(f"st:k:{item_index('sorting.confidence')}"))
    await bot_gw.say(make_bot_message("1.5"))
    assert owner_texts(bot_gw)[-1].startswith("Not saved: settings: sorting.confidence = 1.5")
    await bot_gw.say(make_bot_message("0.6"))
    await bot_gw.press(make_callback("st:cx"))
    assert owner_texts(bot_gw)[-1] == plain_text(rt.t("flow_cancelled"))
    await bot_gw.press(make_callback("st:ap"))  # stale button: the proposal is gone
    assert owner_texts(bot_gw)[-1] == plain_text(rt.t("unknown_choice"))
    assert file_text(rt) == before


async def test_topic_strictness_is_previewed_then_applied_to_file_and_row(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> None:
    await seed_posts(rt, make_chat, n=2)
    topic = await ml_topic(rt)
    assert isinstance(rt.classifier, FakeClassifier)
    rt.classifier.scores = {topic.id: 0.6}
    before = file_text(rt)

    await bot_gw.press(make_callback("st:ts"))
    assert f"st:ts:{topic.id}" in last_buttons(bot_gw)
    await bot_gw.press(make_callback(f"st:ts:{topic.id}"))
    assert "now the general 0.5" in owner_texts(bot_gw)[-1]
    await bot_gw.say(make_bot_message("0.8"))
    await bot_gw.press(make_callback("st:pv"))
    assert "ML & AI: 2 → 0" in owner_texts(bot_gw)[-1]
    assert file_text(rt) == before

    await bot_gw.press(make_callback("st:ap"))
    assert owner_texts(bot_gw)[-1] == "Saved: Strictness of ML & AI = 0.8."
    entry = rt.settings.topic("ml-ai")
    assert entry is not None and entry.strictness == 0.8
    assert (await ml_topic(rt)).strictness == 0.8


# --- posting style ---------------------------------------------------------------------------


async def test_switching_to_forward_explains_what_forwards_cannot_do(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback("st:sy"))
    assert last_buttons(bot_gw) == ["st:sy:0", "st:sy:1"]
    await bot_gw.press(make_callback("st:sy:1"))
    assert rt.settings.publishing.style == "forward"
    assert owner_texts(bot_gw)[-1] == (
        "Saved: forward style. Forwards keep Telegram's attribution; the bot adds a short line "
        "with the source, +N and the [Wrong topic] button under each. A forward cannot be "
        "moved: after a correction it stays where it is and only its label changes. Group "
        "threads and protected channels are always reposted."
    )


async def test_switching_to_repost_while_live_ensures_the_staging_channel(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    await rt.settings_file.set_value("publishing.style", "forward")
    await rt.settings_file.set_value("publishing.live", True)

    await bot_gw.press(make_callback("st:sy:0"))

    assert rt.settings.publishing.style == "repost"
    assert [c.title for c in user_gw.created] == ["tg-curator media"]
    assert rt.settings.publishing.staging_channel == user_gw.created[0].id
    assert "'tg-curator media' hands media to the bot" in owner_texts(bot_gw)[-1]


async def test_switching_to_repost_before_go_creates_nothing(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    await rt.settings_file.set_value("publishing.style", "forward")
    await bot_gw.press(make_callback("st:sy:0"))
    assert user_gw.created == []
    assert owner_texts(bot_gw)[-1] == "Saved: repost style."


async def test_review_weekday_from_buttons(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback("st:wd"))
    assert "st:wd:0" in last_buttons(bot_gw)
    await bot_gw.press(make_callback("st:wd:0"))
    assert rt.settings.review.weekday == "monday"
    assert owner_texts(bot_gw)[-1] == "Saved: Review day = Monday."


# --- trusted sources -------------------------------------------------------------------------


async def test_trusted_source_add_and_remove_update_the_file_and_chat_trust(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
) -> None:
    kun = user_gw.add_chat(make_chat(SOURCE, title="Kun.uz", username="kunuz"))

    await bot_gw.press(make_callback("st:src"))
    assert "None yet" in owner_texts(bot_gw)[-1]
    await bot_gw.press(make_callback("st:sa"))
    await bot_gw.say(make_bot_message("@kunuz"))
    assert "How much should Kun.uz be trusted?" in owner_texts(bot_gw)[-1]
    await bot_gw.press(make_callback("st:tr:3"))
    # trust decides what goes out at once: a proposal first, nothing written yet
    assert owner_texts(bot_gw)[-1].startswith(
        "Trust of Kun.uz: not listed (neutral) → 3 · always immediate"
    )
    assert last_buttons(bot_gw) == ["st:pv", "st:ap", "st:cx"]
    assert rt.settings.sources == []
    await bot_gw.press(make_callback("st:ap"))

    assert owner_texts(bot_gw)[-1] == "Saved: Kun.uz — 3 · always immediate."
    assert [(s.chat, s.trust) for s in rt.settings.sources] == [(kun.id, 3)]
    chat = await rt.store.get_chat(kun.id)
    assert chat is not None and chat.trust == 3.0

    await bot_gw.press(make_callback("st:src"))
    assert "Kun.uz — 3 · always immediate" in owner_texts(bot_gw)[-1]
    assert "st:sr:0" in last_buttons(bot_gw)
    await bot_gw.press(make_callback("st:sr:0"))

    assert owner_texts(bot_gw)[-1] == "Removed Kun.uz; it is neutral again."
    assert rt.settings.sources == []
    chat = await rt.store.get_chat(kun.id)
    assert chat is not None and chat.trust is None


async def test_re_adding_a_source_written_as_username_keeps_one_entry(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
) -> None:
    kun = user_gw.add_chat(make_chat(SOURCE, title="Kun.uz", username="kunuz"))
    await rt.settings_file.upsert_source("@kunuz", 1)

    await bot_gw.press(make_callback("st:sa"))
    await bot_gw.say(make_bot_message("https://t.me/kunuz"))
    await bot_gw.press(make_callback("st:tr:2"))
    await bot_gw.press(make_callback("st:ap"))

    assert [(s.chat, s.trust) for s in rt.settings.sources] == [(kun.id, 2)]


async def test_a_trust_change_is_previewed_before_it_applies(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
) -> None:
    await seed_posts(rt, make_chat)
    user_gw.add_chat(make_chat(SOURCE, title="Source", username="source"))
    await bot_gw.press(make_callback("st:sa"))
    await bot_gw.say(make_bot_message("@source"))
    await bot_gw.press(make_callback("st:tr:0"))
    await bot_gw.press(make_callback("st:pv"))
    preview = owner_texts(bot_gw)[-1]
    assert preview.startswith("Preview, last 3 days — Trust of Source: not listed (neutral)")
    assert "would go out immediately:" in preview and "Nothing was changed yet." in preview
    assert last_buttons(bot_gw) == ["st:ap", "st:cx"]
    assert rt.settings.sources == []
    chat = await rt.store.get_chat(SOURCE)
    assert chat is not None and chat.trust is None
    await bot_gw.press(make_callback("st:cx"))
    assert rt.settings.sources == []


async def test_an_unknown_source_is_refused_and_nothing_is_written(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback("st:sa"))
    await bot_gw.say(make_bot_message("@nobody_here"))
    assert owner_texts(bot_gw)[-1] == plain_text(rt.t("error_chat_gone"))
    assert rt.settings.sources == []


# --- the catalogue ---------------------------------------------------------------------------


def test_every_settings_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(settings_mod)
    used = set(re.findall(r'"(settings_[a-z0-9_]+)"', source))
    with (LOCALES_DIR / "en" / "settings.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = {"unknown_choice", "flow_cancelled", "working", "cancel"}
    assert common <= Translator(locales_dir=LOCALES_DIR).english_keys()


async def test_hold_time_applies_at_once_and_says_why_it_has_no_preview(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback(f"st:k:{item_index('sorting.hold_minutes')}"))
    assert "Applies at once" in owner_texts(bot_gw)[-1]
    await bot_gw.say(make_bot_message("30"))
    assert owner_texts(bot_gw)[-1] == "Saved: Hold time = 30."
    assert rt.settings.sorting.hold_minutes == 30


async def test_group_minimum_cannot_be_previewed_on_stored_posts(rt: Runtime) -> None:
    from tg_curator.errors import ConfigError

    with pytest.raises(ConfigError, match="applied at intake"):
        await PreviewService(rt).replay(overrides={"groups.min_chars": 10})
