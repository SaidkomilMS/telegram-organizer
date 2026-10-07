"""Regressions for the review findings on ``TopicsService`` (staging channel, creation pacing,
example-channel retries, ``[[sources]]`` mirroring)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from tests.fakes import FakeClock, FakeUserGateway, FakeWorld
from tests.test_topics import Events, FakeSorter
from tg_curator.domain import KV
from tg_curator.errors import BotCannotPost, ConfigError, FloodWait, TelegramUnavailable
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import TopicsService


@pytest.fixture
def svc(rt: Runtime) -> TopicsService:
    rt.sorter = FakeSorter()
    svc = TopicsService(rt)
    rt.topics = svc
    return svc


# --- staging channel: a failed promotion never leaves an unrecorded channel ------------------


async def test_failed_bot_promotion_keeps_the_staging_channel_and_retries(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, world: FakeWorld
) -> None:
    next_id = user_gw._next_chat - 1  # the id the fake gives the next created channel
    world.bot_blocked.add(next_id)  # a new channel: the bot cannot post until promoted
    user_gw.fail_next("add_bot_admin", TelegramUnavailable("Telegram is not reachable"))

    with pytest.raises(BotCannotPost):
        await svc.ensure_staging_channel()

    assert [c.id for c in user_gw.created] == [next_id]
    assert rt.settings.publishing.staging_channel == next_id, "recorded before the bot step"
    chat = await rt.store.get_chat(next_id)
    assert chat is not None and chat.role == "staging"
    assert next_id in user_gw.owned

    # The next trigger repairs the promotion on the same channel, unpaced or not.
    assert await svc.ensure_staging_channel() == next_id
    assert len(user_gw.created) == 1, "no second channel"
    assert [c["chat_id"] for c in user_gw.calls_of("add_bot_admin")] == [next_id, next_id]
    assert (next_id, "curator_test_bot") in user_gw.admins

    # Once the bot can post, a further call touches nothing.
    assert await svc.ensure_staging_channel() == next_id
    assert len(user_gw.calls_of("add_bot_admin")) == 2


async def test_staging_channel_without_a_bot_username_creates_nothing(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway
) -> None:
    rt.bot_account = None
    for _ in range(3):
        with pytest.raises(ConfigError):
            await svc.ensure_staging_channel()
    assert user_gw.created == []
    assert rt.settings.publishing.staging_channel == 0


async def test_staging_channel_creation_is_paced(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, clock: FakeClock
) -> None:
    await svc.create("Alpha", create_channel=True)
    assert len(user_gw.created) == 1

    with pytest.raises(FloodWait) as wait:
        await svc.ensure_staging_channel()
    assert wait.value.seconds == 60 and len(user_gw.created) == 1
    assert rt.settings.publishing.staging_channel == 0

    clock.advance(61)
    staging = await svc.ensure_staging_channel()
    assert staging == user_gw.created[-1].id and len(user_gw.created) == 2


# --- creation pacing reads the stored log in a fresh process ---------------------------------


async def test_a_fresh_service_respects_the_stored_create_log(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, clock: FakeClock
) -> None:
    await svc.create("Alpha", create_channel=True)
    assert len(user_gw.created) == 1

    fresh = TopicsService(rt)  # a CLI run: no sync_from_settings before creating
    beta = await fresh.create("Beta", create_channel=True)
    assert beta.channel_id is None and len(user_gw.created) == 1
    assert fresh.channel_wait_minutes() == 1


async def test_a_fresh_service_respects_a_stored_flood_wait(
    rt: Runtime, user_gw: FakeUserGateway, clock: FakeClock
) -> None:
    rt.sorter = FakeSorter()
    first = TopicsService(rt)
    user_gw.fail_next("create_channel", FloodWait(86_400))
    assert await first.create_topic_channel("Alpha") is None
    stored = await rt.store.kv_get(KV.TOPICS_CREATE_LOG)
    assert stored is not None and stored["wait_until"]

    clock.advance(timedelta(hours=1))
    fresh = TopicsService(rt)
    assert await fresh.create_topic_channel("Beta") is None
    assert len(user_gw.created) == 0
    assert fresh.channel_wait_minutes() == 23 * 60
    after = await rt.store.kv_get(KV.TOPICS_CREATE_LOG)
    assert after == stored, "the recorded FloodWait is kept"


# --- example channels set in the file are retried until they are read ------------------------


def _fill_history(
    user_gw: FakeUserGateway,
    chat: ChatInfo,
    make_message: Callable[..., Any],
    clock: FakeClock,
    n: int = 3,
) -> None:
    for i in range(n):
        user_gw.history_data[chat.id].append(
            make_message(chat, text=f"Kun news {i}", date=clock.now() - timedelta(hours=i + 1))
        )


async def _channel_examples(rt: Runtime, key: str) -> int:
    topic = await rt.store.get_topic_by_key(key)
    assert topic is not None
    return sum(1 for e in await rt.store.list_examples(topic_id=topic.id) if e.kind == "channel")


async def test_an_example_channel_that_failed_is_read_on_the_next_sync(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    clock: FakeClock,
) -> None:
    await svc.sync_from_settings()
    await rt.settings_file.upsert_topic("uzbekistan", example_channel="@kunuz")

    first = await svc.sync_from_settings()
    assert any("example channel" in line for line in first.unresolved)
    assert await _channel_examples(rt, "uzbekistan") == 0
    topic = await rt.store.get_topic_by_key("uzbekistan")
    assert topic is not None and topic.example_channel is None, "not read yet"

    kun = user_gw.add_chat(make_chat(title="Kun.uz", username="kunuz"))
    _fill_history(user_gw, kun, make_message, clock)
    events = Events(rt)

    second = await svc.sync_from_settings()
    assert not any("example channel" in line for line in second.unresolved)
    assert await _channel_examples(rt, "uzbekistan") == 3
    topic = await rt.store.get_topic_by_key("uzbekistan")
    assert topic is not None and topic.example_channel == "@kunuz"
    assert "examples" in events.reasons()

    await svc.sync_from_settings()
    assert await _channel_examples(rt, "uzbekistan") == 3, "read once, not on every sync"


async def test_a_re_added_topic_reads_its_example_channel_again(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    clock: FakeClock,
) -> None:
    kun = user_gw.add_chat(make_chat(title="Kun.uz", username="kunuz"))
    _fill_history(user_gw, kun, make_message, clock)
    await svc.sync_from_settings()
    entry = rt.settings.topic("uzbekistan")
    assert entry is not None
    await rt.settings_file.upsert_topic("uzbekistan", example_channel="@kunuz")
    await svc.sync_from_settings()
    assert await _channel_examples(rt, "uzbekistan") == 3

    await rt.settings_file.remove_topic("uzbekistan")
    await svc.sync_from_settings()
    assert await _channel_examples(rt, "uzbekistan") == 0, "§9.7 deleted them"

    await rt.settings_file.upsert_topic(
        "uzbekistan", name=entry.name, category=entry.category, example_channel="@kunuz"
    )
    result = await svc.sync_from_settings()
    assert not any("example channel" in line for line in result.unresolved)
    assert await _channel_examples(rt, "uzbekistan") == 3


async def test_clearing_the_example_channel_in_the_file_keeps_the_examples(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    clock: FakeClock,
) -> None:
    kun = user_gw.add_chat(make_chat(title="Kun.uz", username="kunuz"))
    _fill_history(user_gw, kun, make_message, clock)
    await svc.sync_from_settings()
    await rt.settings_file.upsert_topic("uzbekistan", example_channel="@kunuz")
    await svc.sync_from_settings()

    await rt.settings_file.upsert_topic("uzbekistan", example_channel="")
    await svc.sync_from_settings()
    topic = await rt.store.get_topic_by_key("uzbekistan")
    assert topic is not None and topic.example_channel is None
    assert await _channel_examples(rt, "uzbekistan") == 3


# --- [[sources]] never re-activates a chat the account left ----------------------------------


async def test_sources_sync_does_not_resurrect_a_left_chat(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    clock: FakeClock,
) -> None:
    foo = user_gw.add_chat(make_chat(title="Foo", username="foo"))
    await rt.store.upsert_chat(foo)
    left_at = clock.now() - timedelta(days=1)
    await rt.store.set_chat_fields(foo.id, active=False, left_at=left_at)
    await rt.settings_file.upsert_source("@foo", 3)

    result = await svc.sync_from_settings()
    assert result.unresolved == []
    chat = await rt.store.get_chat(foo.id)
    assert chat is not None
    assert chat.active is False and chat.left_at == left_at
    assert chat.trust == 3.0, "trust still mirrored"

    await svc.sync_from_settings()
    chat = await rt.store.get_chat(foo.id)
    assert chat is not None and chat.active is False and chat.left_at == left_at


async def test_sources_sync_registers_an_unknown_listed_chat(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    bar = user_gw.add_chat(make_chat(title="Bar", username="bar"))
    await rt.settings_file.upsert_source("@bar", 2)
    await svc.sync_from_settings()
    chat = await rt.store.get_chat(bar.id)
    assert chat is not None and chat.active and chat.trust == 2.0
