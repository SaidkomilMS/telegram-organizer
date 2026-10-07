"""Spec gaps closed in the topics group: category keys (S2), one channel per topic and never
the staging channel (TC-6), and deferred channel creation that resumes by itself (SAFE-6)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, FakeClock, FakeUserGateway
from tests.test_cli import configured, curator, fakes, home  # noqa: F401 - CLI fixtures
from tg_curator import service as service_module
from tg_curator.config import SettingsFile
from tg_curator.domain import KV
from tg_curator.errors import ConfigError, FloodWait
from tg_curator.ml import categories
from tg_curator.runtime import Runtime
from tg_curator.service import Doubles
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import TopicsService


class _Sorter:
    async def submit(self, c: Any) -> None:
        raise NotImplementedError

    async def tick(self) -> None:
        raise NotImplementedError

    async def resort_unsorted(self, since: datetime) -> int:
        return 0


@pytest.fixture
def svc(rt: Runtime) -> TopicsService:
    rt.sorter = _Sorter()
    svc = TopicsService(rt)
    rt.topics = svc
    return svc


@pytest.fixture
def chan(user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]) -> ChatInfo:
    return user_gw.add_chat(make_chat(title="ML channel", username="mlchan", is_creator=True))


# --- S2: a category must be a built-in key ---------------------------------------------------


def test_require_key_suggests_the_key_for_the_spec_word() -> None:
    assert categories.require_key(" Tech ") == "tech"
    with pytest.raises(ConfigError) as err:
        categories.require_key("technology")
    assert str(err.value) == (
        "unknown category 'technology' (did you mean 'tech'?); "
        "`curator topics categories` lists the keys"
    )
    assert categories.suggest("sports") == ["sport"]
    assert categories.suggest("zzz") == []


def test_cli_topics_add_refuses_an_unknown_category(configured: Path, fakes: Doubles) -> None:  # noqa: F811
    result = curator(configured, "topics", "add", "Tech stuff", "--category", "technology")
    assert result.exit_code != 0, result.output
    assert "unknown category 'technology'" in result.output
    assert "'tech'" in result.output
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    assert all(t.name != "Tech stuff" for t in settings.topics), "nothing half-created"


async def test_create_and_update_check_the_category_before_writing(
    rt: Runtime, svc: TopicsService
) -> None:
    await svc.sync_from_settings()
    with pytest.raises(ConfigError, match="did you mean 'tech'"):
        await svc.create("Tech stuff", category="technology")
    assert await rt.store.get_topic_by_key("tech-stuff") is None
    assert rt.settings.topic("tech-stuff") is None

    topic = await svc.create("Tech stuff", category="Tech")
    assert topic.category == "tech"
    assert rt.settings.topic("tech-stuff").category == "tech"  # type: ignore[union-attr]

    with pytest.raises(ConfigError, match="unknown category 'sports'"):
        await svc.update("tech-stuff", category="sports", name="Renamed")
    row = await rt.store.get_topic_by_key("tech-stuff")
    assert row is not None and row.category == "tech" and row.name == "Tech stuff"
    assert rt.settings.topic("tech-stuff").category == "tech"  # type: ignore[union-attr]

    updated = await svc.update("tech-stuff", category="")
    assert updated.category is None


async def test_sync_reports_a_hand_edited_unknown_category(rt: Runtime, svc: TopicsService) -> None:
    await rt.settings_file.upsert_topic("ml-ai", category="technology")
    result = await svc.sync_from_settings()
    assert result.unresolved == [
        "topic ml-ai: unknown category 'technology' (did you mean 'tech'?); "
        "`curator topics categories` lists the keys"
    ]
    assert result.total == 4, "the rest of the file still loads"


# --- TC-6: one channel per topic, never the staging channel ----------------------------------


async def test_a_channel_belongs_to_one_topic(
    rt: Runtime, svc: TopicsService, chan: ChatInfo
) -> None:
    await svc.sync_from_settings()
    await svc.link_channel("ml-ai", "@mlchan")
    again = await svc.link_channel("ml-ai", "@mlchan")  # its own channel again: fine
    assert again.channel_id == chan.id

    with pytest.raises(ConfigError) as err:
        await svc.link_channel("fintech", "@mlchan")
    assert str(err.value) == "channel ML channel is already the channel of topic ML & AI"
    fintech = await rt.store.get_topic_by_key("fintech")
    assert fintech is not None and fintech.channel_id is None
    ml = await rt.store.get_topic_by_key("ml-ai")
    assert ml is not None and ml.channel_id == chan.id

    with pytest.raises(ConfigError, match="already the channel of topic"):
        await svc.create("Robotics", channel="@mlchan")
    assert await rt.store.get_topic_by_key("robotics") is None
    with pytest.raises(ConfigError, match="already the channel of topic"):
        await svc.update("fintech", channel=chan.id)

    # Once ML & AI is removed its channel is free again.
    await svc.remove("ml-ai")
    linked = await svc.link_channel("fintech", "@mlchan")
    assert linked.channel_id == chan.id


async def test_the_staging_channel_is_never_a_topic_channel(
    rt: Runtime, svc: TopicsService
) -> None:
    await svc.sync_from_settings()
    staging_id = await svc.ensure_staging_channel()
    with pytest.raises(ConfigError) as err:
        await svc.link_channel("football", staging_id)
    assert "private media channel" in str(err.value)
    chat = await rt.store.get_chat(staging_id)
    assert chat is not None and chat.role == "staging"
    football = await rt.store.get_topic_by_key("football")
    assert football is not None and football.channel_id is None


async def test_the_staging_setting_cannot_name_a_topic_channel(
    rt: Runtime, svc: TopicsService, chan: ChatInfo
) -> None:
    await svc.sync_from_settings()
    await svc.link_channel("ml-ai", chan.id)
    await rt.settings_file.set_value("publishing.staging_channel", chan.id)
    with pytest.raises(ConfigError, match="is the channel of topic ML & AI"):
        await svc.ensure_staging_channel()
    chat = await rt.store.get_chat(chan.id)
    assert chat is not None and chat.role == "output"


async def test_sync_reports_a_duplicate_channel_and_keeps_the_stored_ones(
    rt: Runtime, svc: TopicsService, chan: ChatInfo
) -> None:
    await svc.sync_from_settings()
    await svc.link_channel("ml-ai", chan.id)
    await rt.settings_file.upsert_topic("fintech", channel=chan.id)
    result = await svc.sync_from_settings()
    assert result.unresolved == [
        "topic fintech: channel ML channel is already the channel of topic ML & AI"
    ]
    ml = await rt.store.get_topic_by_key("ml-ai")
    fintech = await rt.store.get_topic_by_key("fintech")
    assert ml is not None and ml.channel_id == chan.id
    assert fintech is not None and fintech.channel_id is None


async def test_sync_frees_the_channel_of_a_removed_topic_first(
    rt: Runtime, svc: TopicsService, chan: ChatInfo
) -> None:
    await svc.sync_from_settings()
    await svc.link_channel("ml-ai", chan.id)

    def move(doc: Any) -> None:
        tables = doc["topics"]
        for i, table in enumerate(tables):
            if table["key"] == "ml-ai":
                del tables[i]
                break
        for table in tables:
            if table["key"] == "fintech":
                table["channel"] = chan.id

    await rt.settings_file.update(move)
    result = await svc.sync_from_settings()
    assert result.unresolved == []
    fintech = await rt.store.get_topic_by_key("fintech")
    assert fintech is not None and fintech.channel_id == chan.id


# --- SAFE-6: a deferred channel is created once the wait is over ------------------------------


async def test_deferred_channel_is_created_after_the_wait(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
) -> None:
    await svc.sync_from_settings()
    user_gw.fail_next("create_channel", FloodWait(600))
    topic = await svc.create("Space", create_channel=True)
    assert topic.channel_id is None
    assert await svc.wanted_channels() == ["space"]
    stored = await rt.store.kv_get(KV.TOPICS_CREATE_LOG)
    assert stored["wanted"] == ["space"], "the mark survives a restart"

    await svc.tick()  # still inside Telegram's wait
    assert user_gw.created == []

    clock.advance(timedelta(minutes=10))
    before = len(bot_gw.sent(OWNER_ID))
    fresh = TopicsService(rt)  # a restart in between: the mark is read back from kv
    rt.topics = fresh
    await fresh.tick()

    created = user_gw.created[-1]
    assert created.title == "Space"
    row = await rt.store.get_topic_by_key("space")
    assert row is not None and row.channel_id == created.id
    assert rt.settings.topic("space").channel == created.id  # type: ignore[union-attr]
    assert await fresh.wanted_channels() == []
    told = bot_gw.sent(OWNER_ID)[before:]
    assert len(told) == 1
    assert "wait is over: Space now posts into its new channel Space" in told[0].text
    assert "cannot post" not in told[0].text

    await fresh.tick()  # nothing left to do
    assert len(user_gw.created) == 1


async def test_deferred_channel_waits_again_and_tells_about_missing_rights(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
) -> None:
    await svc.sync_from_settings()
    user_gw.fail_next("create_channel", FloodWait(600))
    await svc.create("Space", create_channel=True)
    clock.advance(timedelta(minutes=10))
    user_gw.fail_next("create_channel", FloodWait(300))
    await svc.tick()
    assert await svc.wanted_channels() == ["space"] and svc.channel_wait_minutes() == 5

    clock.advance(timedelta(minutes=5))
    user_gw.fail_next("add_bot_admin", ConfigError("refused"))
    original = user_gw.create_channel

    async def create_blocked(title: str, about: str = "") -> ChatInfo:
        info = await original(title, about=about)
        bot_gw.world.bot_blocked.add(info.id)
        return info

    user_gw.create_channel = create_blocked  # type: ignore[method-assign]
    await svc.tick()
    assert await svc.wanted_channels() == []
    last = bot_gw.sent(OWNER_ID)[-1].text
    assert "cannot post into it" in last and "@curator_test_bot" in last


async def test_a_removed_or_linked_topic_drops_its_mark(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, clock: FakeClock, chan: ChatInfo
) -> None:
    await svc.sync_from_settings()
    await svc.want_channel("ml-ai")
    await svc.want_channel("fintech")
    await svc.link_channel("ml-ai", chan.id)
    await svc.remove("fintech")
    await svc.tick()
    assert user_gw.created == []
    assert await svc.wanted_channels() == []


def test_the_service_runs_the_deferred_channel_loop() -> None:
    assert "topic_channels" in service_module.ACCOUNT_LOOPS
