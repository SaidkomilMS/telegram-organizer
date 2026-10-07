"""bot/reports.py: /stats rendered from seeded statistics, /digest and /digest preview."""

from __future__ import annotations

import inspect
import re
import tomllib
from datetime import timedelta

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, FakeLLM, make_bot_message, plain_text
from tg_curator.bot import reports
from tg_curator.bot.core import BotApp
from tg_curator.clock import local_date
from tg_curator.domain import NewPost, Post, PostStatus, Topic
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.render import TEXT_LIMIT
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash, utf16_len

CHANNEL = -1_001_000_009_001
LOUD = -1_001_000_000_201
GOOD = -1_001_000_000_202
GONE = -1_001_000_000_203


def info(chat_id: int, title: str, *, kind: str = "channel") -> ChatInfo:
    return ChatInfo(
        id=chat_id,
        kind=kind,  # type: ignore[arg-type]
        title=title,
        username=None,
        noforwards=False,
        is_creator=False,
        is_admin=False,
        archived=False,
        muted_until=None,
    )


class Seed:
    """Chats, topics and posts for the two commands, one call each."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.mid: dict[int, int] = {}

    async def topic(self, key: str = "ml-ai", name: str = "ML & AI") -> Topic:
        await self.rt.store.upsert_chat(info(CHANNEL, f"{name} channel"), role="output")
        self.rt.user.register_owned(CHANNEL)
        return await self.rt.store.upsert_topic(
            Topic(id=0, key=key, name=name, channel_id=CHANNEL, created_at=self.rt.clock.now())
        )

    async def chat(self, chat_id: int, title: str, volume: int, *, kind: str = "channel") -> None:
        await self.rt.store.upsert_chat(info(chat_id, title, kind=kind))
        day = local_date(self.rt.clock.now(), self.rt.settings.general.timezone)
        await self.rt.store.bump_chat_daily(chat_id, day, volume)

    async def post(
        self,
        chat_id: int,
        *,
        topic_id: int | None,
        status: PostStatus,
        text: str | None = None,
    ) -> Post:
        mid = self.mid.get(chat_id, 0) + 1
        self.mid[chat_id] = mid
        text = text or f"Post {mid} of chat {chat_id} with a few words."
        new = NewPost(
            chat_id=chat_id,
            message_id=mid,
            kind="post",
            message_ids=[mid],
            posted_at=self.rt.clock.now() - timedelta(hours=1),
            via="live",
            text=text,
            text_hash=text_hash(text),
            urls=[],
        )
        post = await self.rt.store.insert_post(new, status=status, topic_id=topic_id)
        assert post is not None
        return post


@pytest.fixture
async def app(rt: Runtime) -> BotApp:
    await rt.settings_file.set_value("general.timezone", "UTC")
    rt.stats = StatsService(rt)
    digest = DigestService(rt)
    digest.VIEWS_GAP_SECONDS = 0.0
    digest.PART_GAP_SECONDS = 0.0
    rt.digest = digest
    rt.llm = FakeLLM(enabled=False)
    rt.notifier.MIN_GAP_SECONDS = 0.0  # type: ignore[attr-defined]
    app = BotApp(rt)
    reports.register(app)
    return app


@pytest.fixture
def seed(rt: Runtime) -> Seed:
    return Seed(rt)


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


# --- /stats ----------------------------------------------------------------------------------


async def test_stats_is_a_table_sorted_by_signal_ascending(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    topic = await seed.topic()
    await seed.chat(LOUD, "Loud group", 300, kind="group")
    await seed.chat(GOOD, "Good channel", 10)
    for _ in range(6):
        await seed.post(LOUD, topic_id=topic.id, status=PostStatus.dropped)
    for _ in range(5):
        await seed.post(GOOD, topic_id=topic.id, status=PostStatus.published)

    await bot_gw.say(make_bot_message("/stats"))

    [html] = [m.html for m in bot_gw.sent(OWNER_ID)]
    assert "<pre>" in html and "</pre>" in html
    text = plain_text(html)
    rows = [line for line in text.splitlines() if "Loud group" in line or "Good channel" in line]
    assert [r.split("  ")[-1] for r in rows] == ["Loud group", "Good channel"]
    loud, good = rows
    assert loud.split()[:5] == ["2%", "0%", "300", "0", "0"]
    assert good.split()[:5] == ["50%", "0%", "10", "5", "0"]
    assert "last 30 days" in text
    assert " sig  dup   vol  pub day  chat" in text


async def test_stats_hides_left_chats_unless_all(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    await seed.chat(GOOD, "Good channel", 10)
    await seed.chat(GONE, "Left & gone", 4)
    await rt.store.set_chat_fields(GONE, active=False, left_at=rt.clock.now())

    await bot_gw.say(make_bot_message("/stats"))
    assert "Left & gone" not in owner_texts(bot_gw)[-1]
    await bot_gw.say(make_bot_message("/stats all"))
    html = bot_gw.sent(OWNER_ID)[-1].html or ""
    assert "Left &amp; gone" in html  # titles are escaped inside the <pre> table


async def test_long_stats_are_split_into_parts_that_each_fit(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    for n in range(160):
        await seed.chat(-1_001_000_100_000 - n, f"Channel number {n:03d} with a long name", n + 1)

    await bot_gw.say(make_bot_message("/stats"))

    sent = bot_gw.sent(OWNER_ID)
    assert len(sent) >= 2
    for msg in sent:
        assert utf16_len(plain_text(msg.html)) <= TEXT_LIMIT
        assert (msg.html or "").count("<pre>") == 1
    titles = [line for m in sent for line in plain_text(m.html).splitlines() if "Channel" in line]
    assert len(titles) == 160


async def test_stats_without_chats_and_bad_arguments(app: BotApp, bot_gw: FakeBotGateway) -> None:
    await bot_gw.say(make_bot_message("/stats"))
    await bot_gw.say(make_bot_message("/stats everything"))
    t = Translator(locales_dir=LOCALES_DIR)
    assert owner_texts(bot_gw) == [
        plain_text(t("reports_stats_empty")),
        plain_text(t("reports_stats_usage")),
    ]


# --- /digest ---------------------------------------------------------------------------------


async def test_digest_is_refused_with_one_sentence_when_not_live(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    topic = await seed.topic()
    await seed.chat(GOOD, "Good channel", 3)
    await seed.post(GOOD, topic_id=topic.id, status=PostStatus.digest)

    await bot_gw.say(make_bot_message("/digest"))

    assert owner_texts(bot_gw) == [plain_text(rt.t("not_live"))]
    assert bot_gw.sent(CHANNEL) == []


async def test_digest_sends_now_and_replies_with_the_results(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("publishing.live", True)
    topic = await seed.topic()
    await seed.chat(GOOD, "Good channel", 3)
    posts = [await seed.post(GOOD, topic_id=topic.id, status=PostStatus.digest) for _ in range(2)]

    await bot_gw.say(make_bot_message("/digest"))

    channel = [plain_text(m.html) for m in bot_gw.sent(CHANNEL)]
    assert len(channel) == 1 and "Digest (manual 1)" in channel[0]
    texts = owner_texts(bot_gw)
    assert "Digest sent:\nML & AI: 2 posts" in texts
    for post in posts:
        stored = await rt.store.get_post(post.id)
        assert stored is not None and stored.status == PostStatus.digested


async def test_digest_with_an_empty_pool_says_nothing_to_send(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("publishing.live", True)
    await seed.topic()

    await bot_gw.say(make_bot_message("/digest"))

    assert owner_texts(bot_gw)[-1] == plain_text(rt.t("reports_digest_nothing"))
    assert bot_gw.sent(CHANNEL) == []


async def test_digest_for_an_unknown_topic_is_refused(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("publishing.live", True)
    await seed.topic()

    await bot_gw.say(make_bot_message("/digest no-such-topic"))

    assert owner_texts(bot_gw)[-1] == plain_text(rt.t("unknown_choice"))


async def test_digest_preview_sends_the_drafts_to_the_owner_only(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    topic = await seed.topic()
    await rt.store.upsert_topic(
        Topic(id=0, key="empty", name="Empty topic", channel_id=-1_001_000_009_002,
              created_at=rt.clock.now())
    )  # fmt: skip
    await seed.chat(GOOD, "Good channel", 3)
    post = await seed.post(
        GOOD, topic_id=topic.id, status=PostStatus.digest, text="Rates held at 14 percent."
    )

    await bot_gw.say(make_bot_message("/digest preview"))

    texts = owner_texts(bot_gw)
    previews = [t for t in texts if t.startswith("preview — ")]
    assert len(previews) == 1
    assert "Daily digest" in previews[0] and "Rates held at 14 percent." in previews[0]
    assert texts[-1] == "Nothing tonight for: Empty topic"
    assert bot_gw.sent(CHANNEL) == []
    stored = await rt.store.get_post(post.id)
    assert stored is not None and stored.status == PostStatus.digest


async def test_digest_preview_with_nothing_waiting(
    app: BotApp, rt: Runtime, seed: Seed, bot_gw: FakeBotGateway
) -> None:
    await seed.topic()

    await bot_gw.say(make_bot_message("/digest preview"))

    assert owner_texts(bot_gw) == [plain_text(rt.t("reports_preview_nothing"))]


# --- the catalogue ---------------------------------------------------------------------------


def test_every_reports_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(reports)
    used = set(re.findall(r'"(reports_[a-z_]+)"', source))
    with (LOCALES_DIR / "en" / "reports.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = {"unknown_choice", "not_live"}
    assert common <= Translator(locales_dir=LOCALES_DIR).english_keys()
