"""TopicsService (DESIGN §8, §4, §9.7, §14.13) against the fakes and the real Store."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest

from tests.fakes import FakeClock, FakeUserGateway
from tg_curator.domain import (
    DIGEST_CANCELLED,
    KV,
    PUB_CANCELLED,
    PUB_PENDING,
    Example,
    NewPost,
    PostStatus,
)
from tg_curator.errors import BotCannotPost, ConfigError, FloodWait, NotAllowed, TopicExists
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import STAGING_ABOUT, STAGING_TITLE, TopicsService, slugify

TEMPLATE_KEYS = ["ml-ai", "fintech", "uzbekistan", "football"]


class FakeSorter:
    def __init__(self) -> None:
        self.resorts: list[datetime] = []

    async def submit(self, c: Any) -> None:
        raise NotImplementedError

    async def tick(self) -> None:
        raise NotImplementedError

    async def resort_unsorted(self, since: datetime) -> int:
        self.resorts.append(since)
        return 0


class Events:
    """Records every event the bus delivered, in order."""

    def __init__(self, rt: Runtime) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []
        for name in ("topics_changed", "examples_changed", "settings_changed"):

            async def handler(_name: str = name, **payload: Any) -> None:
                self.seen.append((_name, payload))

            rt.events.on(name, handler)

    def names(self) -> list[str]:
        return [n for n, _ in self.seen]

    def reasons(self) -> list[str]:
        return [p["reason"] for n, p in self.seen if n == "examples_changed"]


@pytest.fixture
def svc(rt: Runtime) -> TopicsService:
    rt.sorter = FakeSorter()
    svc = TopicsService(rt)
    rt.topics = svc
    return svc


@pytest.fixture
def output_channel(user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]) -> ChatInfo:
    return user_gw.add_chat(make_chat(title="ML channel", username="mlchan", is_creator=True))


def post(rt: Runtime, message_id: int, **over: Any) -> NewPost:
    values: dict[str, Any] = {
        "chat_id": -1_001_000_000_777,
        "message_id": message_id,
        "kind": "post",
        "message_ids": [message_id],
        "posted_at": rt.clock.now() - timedelta(minutes=5),
        "via": "live",
        "text": f"post number {message_id} with a few words",
        "text_hash": f"hash{message_id}",
        "urls": [],
    }
    values.update(over)
    return NewPost(**values)


async def add_post(rt: Runtime, message_id: int, status: PostStatus, topic_id: int | None) -> int:
    row = await rt.store.insert_post(post(rt, message_id), status=status, topic_id=topic_id)
    assert row is not None
    return row.id


# --- slugs -----------------------------------------------------------------------------------


def test_slugify_transliterates_and_cleans() -> None:
    assert slugify("ML & AI") == "ml-ai"
    assert slugify("Узбекистан: новости") == "uzbekistan-novosti"
    assert slugify("Футбол") == "futbol"
    assert slugify("Qo'qon va Toshkent!") == "qo-qon-va-toshkent"
    assert slugify("Crème brûlée") == "creme-brulee"
    assert slugify("!!!") == "topic"
    assert len(slugify("a" * 80)) == 32


# --- sync ------------------------------------------------------------------------------------


async def test_sync_creates_the_template_topics(rt: Runtime, svc: TopicsService) -> None:
    events = Events(rt)
    result = await svc.sync_from_settings()

    assert (result.total, result.resolved, result.unresolved) == (4, 0, [])
    topics = {t.key: t for t in await rt.store.list_topics()}
    assert list(topics) == TEMPLATE_KEYS
    uz = topics["uzbekistan"]
    assert uz.category is None and uz.example_channel is None and uz.strictness is None
    assert topics["ml-ai"].category == "tech" and topics["ml-ai"].channel_id is None
    assert rt.classifier.reloads == 1 and [t.key for t in rt.classifier.topics] == TEMPLATE_KEYS
    assert events.names() == ["topics_changed"]

    # A second sync changes nothing and announces nothing.
    await svc.sync_from_settings()
    assert events.names() == ["topics_changed"]
    assert len(await rt.store.list_topics(active=None)) == 4


async def test_sync_resolves_a_username_channel_and_rewrites_the_file(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo, user_gw: FakeUserGateway
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel="@mlchan")

    result = await svc.sync_from_settings()

    assert (result.resolved, result.unresolved) == (1, [])
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None and topic.channel_id == output_channel.id
    assert rt.settings.topic("ml-ai").channel == output_channel.id  # type: ignore[union-attr]
    text = rt.settings_file.path.read_text()
    assert f"channel = {output_channel.id}" in text
    assert "# from my.telegram.org" in text, "the user's comments survive the rewrite"
    chat = await rt.store.get_chat(output_channel.id)
    assert chat is not None and chat.role == "output"
    assert output_channel.id in user_gw.owned
    assert rt.bot.calls_of("can_post") == [{"chat_id": output_channel.id}]  # type: ignore[union-attr]


async def test_sync_does_not_loop_on_its_own_rewrite(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel="@mlchan")
    calls = 0

    async def service_handler(**_: Any) -> None:
        nonlocal calls
        calls += 1
        await svc.sync_from_settings()

    rt.events.on("settings_changed", service_handler)
    await svc.sync_from_settings()
    assert calls == 1, "one settings_changed from the rewrite, re-entered sync returned at once"
    assert rt.settings.topic("ml-ai").channel == output_channel.id  # type: ignore[union-attr]


async def test_sync_collects_channel_failures_instead_of_raising(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    group = user_gw.add_chat(make_chat(kind="group", title="A group", username="agroup"))
    blocked = user_gw.add_chat(make_chat(title="Blocked", username="blocked", is_admin=True))
    rt.bot.cannot_post.add(blocked.id)  # type: ignore[union-attr]
    user_gw.fail_next("add_bot_admin", NotAllowed("fresh_session"))
    await rt.settings_file.upsert_topic("ml-ai", channel="@nobody")
    await rt.settings_file.upsert_topic("fintech", channel="@agroup")
    await rt.settings_file.upsert_topic("football", channel="@blocked")
    await rt.settings_file.upsert_topic("uzbekistan", channel="https://t.me/+secret")

    result = await svc.sync_from_settings()

    assert result.resolved == 0
    assert result.unresolved == [
        "topic ml-ai: topic channel @nobody cannot be found: check the link or that your "
        "account is in it",
        "topic fintech: channel A group is not yours: the account must be its creator or an admin",
        "topic uzbekistan: join the chat in Telegram first, then send the link again",
        "topic football: the bot cannot post into Blocked: add @curator_test_bot as an admin "
        "with Post Messages",
    ]
    assert user_gw.calls_of("add_bot_admin") == [
        {"chat_id": blocked.id, "bot_username": "curator_test_bot"}
    ]
    assert group.id not in user_gw.owned
    assert (await rt.store.get_topic_by_key("football")).channel_id is None  # type: ignore[union-attr]


async def test_sync_keeps_a_channel_the_file_set_back_to_zero(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel=output_channel.id)
    await svc.sync_from_settings()
    await rt.settings_file.upsert_topic("ml-ai", channel=0)

    result = await svc.sync_from_settings()

    assert result.unresolved == [
        "topic ml-ai: a channel cannot be removed from a topic: remove the topic, or /pause to "
        "stop posting"
    ]
    assert (await rt.store.get_topic_by_key("ml-ai")).channel_id == output_channel.id  # type: ignore[union-attr]


async def test_sync_deactivates_a_removed_key_and_cascades(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel=output_channel.id)
    await svc.sync_from_settings()
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    store = rt.store
    held = await add_post(rt, 1, PostStatus.held, topic.id)
    queued = await add_post(rt, 2, PostStatus.queued, topic.id)
    digest = await add_post(rt, 3, PostStatus.digest, topic.id)
    tracked = await add_post(rt, 4, PostStatus.tracked, topic.id)
    published = await add_post(rt, 5, PostStatus.published, topic.id)
    digested = await add_post(rt, 6, PostStatus.digested, topic.id)
    pub = await store.create_publication(queued, topic.id)
    assert pub is not None
    sent_pub = await store.create_publication(published, topic.id)
    assert sent_pub is not None
    await store.set_publication_fields(sent_pub.id, state="sent")
    digest_row = await store.create_digest(
        topic.id, output_channel.id, rt.clock.now().date(), body=["x"], item_count=1
    )
    await store.add_example(
        Example(
            id=0,
            topic_id=topic.id,
            kind="example",
            text="an example",
            embedding=b"\x00" * 4,
            created_at=rt.clock.now(),
        )
    )
    events = Events(rt)

    await rt.settings_file.remove_topic("ml-ai")
    result = await svc.sync_from_settings()

    assert result.total == 3
    row = await store.get_topic(topic.id)
    assert row is not None and row.active is False and row.channel_id == output_channel.id
    statuses = {}
    for pid in (held, queued, digest, tracked, published, digested):
        p = await store.get_post(pid)
        assert p is not None
        statuses[pid] = (p.status, p.topic_id, p.confidence)
    for pid in (held, queued, digest, tracked):
        assert statuses[pid] == (PostStatus.unsorted, None, None)
    assert statuses[published] == (PostStatus.published, topic.id, None)
    assert statuses[digested] == (PostStatus.digested, topic.id, None)
    assert (await store.get_publication(queued)).state == PUB_CANCELLED  # type: ignore[union-attr]
    assert (await store.get_publication(published)).state == "sent"  # type: ignore[union-attr]
    assert (await store.get_digest(digest_row.id)).state == DIGEST_CANCELLED  # type: ignore[union-attr]
    assert await store.list_examples(topic_id=topic.id) == []
    chat = await store.get_chat(output_channel.id)
    assert chat is not None and chat.role == "output", "the channel is never touched"
    assert [t.key for t in rt.classifier.topics] == ["fintech", "uzbekistan", "football"]
    assert events.names()[:2] == ["settings_changed", "topics_changed"]
    assert events.reasons() == ["topics"]

    # Re-adding the key reactivates the same row, learning afresh.
    await rt.settings_file.upsert_topic("ml-ai", name="ML & AI", channel=0, category="tech")
    await svc.sync_from_settings()
    again = await store.get_topic_by_key("ml-ai")
    assert again is not None and again.id == topic.id and again.active and again.channel_id is None


async def test_sync_mirrors_sources_into_chat_trust(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    kun = user_gw.add_chat(make_chat(title="Kun.uz", username="kunuz"))
    other = user_gw.add_chat(make_chat(title="Other"))
    await rt.store.upsert_chat(other)
    await rt.store.set_chat_fields(other.id, trust=2.0)
    await rt.settings_file.upsert_source("@kunuz", 3)
    await rt.settings_file.upsert_source("@ghost", 0)

    result = await svc.sync_from_settings()

    assert result.unresolved == ["source @ghost: chat @ghost cannot be found"]
    assert (await rt.store.get_chat(kun.id)).trust == 3.0  # type: ignore[union-attr]
    assert (await rt.store.get_chat(other.id)).trust is None, "no longer listed -> reset"  # type: ignore[union-attr]
    assert rt.settings.source_for(kun.id) is not None, "rewritten to the numeric id"
    text = rt.settings_file.path.read_text()
    assert f"chat = {kun.id}" in text and 'chat = "@kunuz"' not in text
    assert "# Trusted sources" in text

    await rt.settings_file.remove_source(kun.id)
    await svc.sync_from_settings()
    assert (await rt.store.get_chat(kun.id)).trust is None  # type: ignore[union-attr]


# --- link_channel ----------------------------------------------------------------------------


async def test_link_channel_error_messages(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await svc.sync_from_settings()
    with pytest.raises(ConfigError) as gone:
        await svc.link_channel("ml-ai", "@missing")
    assert str(gone.value) == (
        "topic channel @missing cannot be found: check the link or that your account is in it"
    )

    member = user_gw.add_chat(make_chat(title="Not mine", username="notmine"))
    with pytest.raises(ConfigError) as mine:
        await svc.link_channel("ml-ai", member.id)
    assert str(mine.value) == (
        "channel Not mine is not yours: the account must be its creator or an admin"
    )

    blocked = user_gw.add_chat(make_chat(title="Blocked", username="blocked", is_creator=True))
    rt.bot.cannot_post.add(blocked.id)  # type: ignore[union-attr]
    user_gw.fail_next("add_bot_admin", NotAllowed("fresh_session"))
    with pytest.raises(BotCannotPost) as cannot:
        await svc.link_channel("ml-ai", "@blocked")
    assert str(cannot.value) == (
        "the bot cannot post into Blocked: add @curator_test_bot as an admin with Post Messages"
    )
    assert (await rt.store.get_topic_by_key("ml-ai")).channel_id is None  # type: ignore[union-attr]

    # The bot gets its admin rights on the second try and the link succeeds.
    topic = await svc.link_channel("ml-ai", "@blocked")
    assert topic.channel_id == blocked.id
    assert rt.settings.topic("ml-ai").channel == blocked.id  # type: ignore[union-attr]
    assert user_gw.calls_of("add_bot_admin")[-1] == {
        "chat_id": blocked.id,
        "bot_username": "curator_test_bot",
    }


# --- create ----------------------------------------------------------------------------------


async def test_create_slugs_collides_and_refuses_a_known_name(
    rt: Runtime, svc: TopicsService
) -> None:
    await svc.sync_from_settings()
    events = Events(rt)

    with pytest.raises(TopicExists):
        await svc.create("ml & ai")

    first = await svc.create("Крипто биржи", category="crypto", description="  ")
    assert (first.key, first.name, first.category, first.description) == (
        "kripto-birzhi",
        "Крипто биржи",
        "crypto",
        None,
    )
    second = await svc.create("ML AI")
    third = await svc.create("ml-ai!")
    assert (second.key, third.key) == ("ml-ai-2", "ml-ai-3")

    entry = rt.settings.topic("kripto-birzhi")
    assert entry is not None and entry.channel == 0 and entry.category == "crypto"
    text = rt.settings_file.path.read_text()
    assert 'key = "ml-ai-3"' in text
    assert "# Topics. Four examples to keep" in text, "the template comment survives"
    assert rt.sorter.resorts == [rt.clock.now() - timedelta(days=7)] * 3  # type: ignore[union-attr]
    assert events.names().count("topics_changed") == 3
    assert events.reasons() == ["topics"] * 3
    assert {t.key for t in rt.classifier.topics} >= {"kripto-birzhi", "ml-ai-2", "ml-ai-3"}


async def test_create_with_an_existing_channel_resolves_before_writing(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await svc.sync_from_settings()
    with pytest.raises(ConfigError):
        await svc.create("Nope", channel="@missing")
    assert await rt.store.get_topic_by_key("nope") is None
    assert rt.settings.topic("nope") is None

    topic = await svc.create("Robotics", channel="@mlchan")
    assert topic.channel_id == output_channel.id
    assert rt.settings.topic("robotics").channel == output_channel.id  # type: ignore[union-attr]


async def test_create_channel_pacing(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, clock: FakeClock
) -> None:
    await svc.sync_from_settings()
    assert svc.channel_wait_minutes() is None

    first = await svc.create("Alpha", create_channel=True)
    created = user_gw.created[-1]
    assert first.channel_id == created.id and created.title == "Alpha"
    assert created.id in user_gw.owned
    assert (await rt.store.get_chat(created.id)).role == "output"  # type: ignore[union-attr]
    assert svc.channel_wait_minutes() == 1

    second = await svc.create("Beta", create_channel=True)
    assert second.channel_id is None, "one channel per 60 s: saved with channel 0"
    assert rt.settings.topic("beta").channel == 0  # type: ignore[union-attr]
    assert len(user_gw.created) == 1

    for name in ("Gamma", "Delta", "Epsilon", "Zeta"):
        clock.advance(61)
        assert svc.channel_wait_minutes() is None
        topic = await svc.create(name, create_channel=True)
        assert topic.channel_id is not None
    assert len(user_gw.created) == 5
    clock.advance(61)
    # 12:05:05 UTC -> next local (UTC) midnight is 11 h 54 min 55 s away, rounded up.
    assert svc.channel_wait_minutes() == 715, "five per day: wait until local midnight"
    eta = await svc.create("Eta", create_channel=True)
    assert eta.channel_id is None and len(user_gw.created) == 5

    log = await rt.store.kv_get(KV.TOPICS_CREATE_LOG)
    assert len(log["created"]) == 5

    clock.advance(timedelta(hours=12))
    assert svc.channel_wait_minutes() is None
    user_gw.fail_next("create_channel", FloodWait(600))
    theta = await svc.create("Theta", create_channel=True)
    assert theta.channel_id is None
    assert svc.channel_wait_minutes() == 10
    clock.advance(timedelta(minutes=10))
    assert svc.channel_wait_minutes() is None


async def test_create_log_survives_a_restart(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway
) -> None:
    await svc.sync_from_settings()
    await svc.create("Alpha", create_channel=True)
    fresh = TopicsService(rt)
    await fresh.sync_from_settings()
    assert fresh.channel_wait_minutes() == 1


# --- staging channel -------------------------------------------------------------------------


async def test_ensure_staging_channel_sequence(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, clock: FakeClock
) -> None:
    staging_id = await svc.ensure_staging_channel()

    created = user_gw.created[-1]
    assert staging_id == created.id
    names = [name for name, _ in user_gw.calls]
    assert names == ["create_channel", "register_owned", "add_bot_admin"]
    assert user_gw.calls_of("create_channel") == [{"title": STAGING_TITLE, "about": STAGING_ABOUT}]
    assert user_gw.calls_of("add_bot_admin") == [
        {"chat_id": staging_id, "bot_username": "curator_test_bot"}
    ]
    assert staging_id in user_gw.owned
    assert (await rt.store.get_chat(staging_id)).role == "staging"  # type: ignore[union-attr]
    assert rt.settings.publishing.staging_channel == staging_id

    assert await svc.ensure_staging_channel() == staging_id
    assert len(user_gw.created) == 1

    user_gw.chats = [c for c in user_gw.chats if c.id != staging_id]
    with pytest.raises(FloodWait):
        await svc.ensure_staging_channel()  # recreation follows the 60 s pacing too
    assert len(user_gw.created) == 1
    clock.advance(61)
    recreated = await svc.ensure_staging_channel()
    assert recreated != staging_id and len(user_gw.created) == 2
    assert rt.settings.publishing.staging_channel == recreated


async def test_staging_channel_flood_wait_propagates(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway
) -> None:
    user_gw.fail_next("create_channel", FloodWait(300))
    with pytest.raises(FloodWait):
        await svc.ensure_staging_channel()
    assert rt.settings.publishing.staging_channel == 0
    assert svc.channel_wait_minutes() == 5


# --- update / remove / merge -----------------------------------------------------------------


async def test_update_rules(rt: Runtime, svc: TopicsService, output_channel: ChatInfo) -> None:
    await svc.sync_from_settings()
    events = Events(rt)

    with pytest.raises(ConfigError) as removed:
        await svc.update("ml-ai", channel=0)
    assert str(removed.value) == (
        "a channel cannot be removed from a topic: remove the topic, or /pause to stop posting"
    )
    with pytest.raises(TopicExists):
        await svc.update("ml-ai", name="fintech")
    with pytest.raises(TypeError):
        await svc.update("ml-ai", key="other")

    topic = await svc.update(
        "ml-ai", channel="@mlchan", description="Models and papers", strictness=0.7, category=""
    )
    assert topic.channel_id == output_channel.id
    assert (topic.description, topic.strictness, topic.category) == ("Models and papers", 0.7, None)
    entry = rt.settings.topic("ml-ai")
    assert entry is not None
    assert (entry.channel, entry.description, entry.strictness, entry.category) == (
        output_channel.id,
        "Models and papers",
        0.7,
        None,
    )
    assert events.reasons() == ["examples"]

    renamed = await svc.update("ml-ai", name="  Machine learning  ")
    assert renamed.name == "Machine learning" and rt.settings.topic("ml-ai").name == renamed.name  # type: ignore[union-attr]
    with pytest.raises(ConfigError):
        await svc.update("ghost", name="x")


async def test_remove_deactivates_and_leaves_the_channel(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo, user_gw: FakeUserGateway
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel=output_channel.id)
    await svc.sync_from_settings()
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    held = await add_post(rt, 1, PostStatus.held, topic.id)
    events = Events(rt)

    await svc.remove("ml-ai")

    assert rt.settings.topic("ml-ai") is None
    row = await rt.store.get_topic(topic.id)
    assert row is not None and not row.active and row.channel_id == output_channel.id
    assert (await rt.store.get_post(held)).status == PostStatus.unsorted  # type: ignore[union-attr]
    assert output_channel.id in user_gw.owned
    assert (await rt.store.get_chat(output_channel.id)).role == "output"  # type: ignore[union-attr]
    assert "topics_changed" in events.names() and events.reasons() == ["topics"]
    with pytest.raises(ConfigError):
        await svc.remove("ml-ai")


async def test_merge_repoints_into_a_topic_with_a_channel(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await rt.settings_file.upsert_topic("fintech", channel=output_channel.id)
    await svc.sync_from_settings()
    src = await rt.store.get_topic_by_key("ml-ai")
    dst = await rt.store.get_topic_by_key("fintech")
    assert src is not None and dst is not None
    await rt.store.set_topic_fields(src.id, channel_id=output_channel.id)
    src = await rt.store.get_topic(src.id)
    assert src is not None
    held = await add_post(rt, 1, PostStatus.held, src.id)
    queued = await add_post(rt, 2, PostStatus.queued, src.id)
    published = await add_post(rt, 3, PostStatus.published, src.id)
    pub = await rt.store.create_publication(queued, src.id)
    assert pub is not None
    await rt.store.add_example(
        Example(
            id=0,
            topic_id=src.id,
            kind="correction",
            post_id=held,
            wrong_topic_id=dst.id,
            text="x",
            embedding=b"\x00" * 4,
            created_at=rt.clock.now(),
        )
    )

    result = await svc.merge("ml-ai", "fintech")

    assert result.id == dst.id
    assert not (await rt.store.get_topic(src.id)).active  # type: ignore[union-attr]
    assert rt.settings.topic("ml-ai") is None and rt.settings.topic("fintech") is not None
    for pid, status in ((held, PostStatus.held), (queued, PostStatus.queued)):
        p = await rt.store.get_post(pid)
        assert p is not None and (p.status, p.topic_id) == (status, dst.id)
    assert (await rt.store.get_post(published)).topic_id == src.id  # type: ignore[union-attr]
    moved_pub = await rt.store.get_publication(queued)
    assert moved_pub is not None and (moved_pub.state, moved_pub.topic_id) == (PUB_PENDING, dst.id)
    examples = await rt.store.list_examples(topic_id=dst.id)
    assert [e.post_id for e in examples] == [held]
    assert rt.sorter.resorts  # type: ignore[union-attr]
    with pytest.raises(ConfigError):
        await svc.merge("fintech", "fintech")


async def test_merge_into_a_channel_less_topic_tracks(
    rt: Runtime, svc: TopicsService, output_channel: ChatInfo
) -> None:
    await rt.settings_file.upsert_topic("ml-ai", channel=output_channel.id)
    await svc.sync_from_settings()
    src = await rt.store.get_topic_by_key("ml-ai")
    dst = await rt.store.get_topic_by_key("football")
    assert src is not None and dst is not None and dst.channel_id is None
    held = await add_post(rt, 1, PostStatus.held, src.id)
    queued = await add_post(rt, 2, PostStatus.queued, src.id)
    digest = await add_post(rt, 3, PostStatus.digest, src.id)
    assert await rt.store.create_publication(queued, src.id) is not None

    await svc.merge("ml-ai", "football")

    for pid in (held, queued, digest):
        p = await rt.store.get_post(pid)
        assert p is not None and (p.status, p.topic_id) == (PostStatus.tracked, dst.id)
    assert (await rt.store.get_publication(queued)).state == PUB_CANCELLED  # type: ignore[union-attr]


# --- examples --------------------------------------------------------------------------------


async def test_add_examples_embeds_and_announces(rt: Runtime, svc: TopicsService) -> None:
    await svc.sync_from_settings()
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    events = Events(rt)

    assert await svc.add_examples("ml-ai", ["  ", ""]) == 0
    n = await svc.add_examples("ml-ai", ["A new transformer paper", " GPU prices drop "])

    assert n == 2
    rows = await rt.store.list_examples(topic_id=topic.id)
    assert [(e.kind, e.text) for e in rows] == [
        ("example", "A new transformer paper"),
        ("example", "GPU prices drop"),
    ]
    assert len(rows[0].embedding) == 64 * 4 and rows[0].post_id is None
    assert events.reasons() == ["examples"]


async def test_add_example_channel_reads_up_to_fifty_posts(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    clock: FakeClock,
) -> None:
    await svc.sync_from_settings()
    topic = await rt.store.get_topic_by_key("uzbekistan")
    assert topic is not None
    source = user_gw.add_chat(make_chat(title="Kun.uz", username="kunuz"))
    for i in range(60):
        date = clock.now() - timedelta(days=3, minutes=60 - i)
        user_gw.history_data[source.id].append(
            make_message(source, text=f"News item {i}", date=date)
        )
    user_gw.history_data[source.id].append(make_message(source, text="   ", date=clock.now()))
    user_gw.history_data[source.id].append(
        make_message(source, text="too old", date=clock.now() - timedelta(days=40))
    )

    n = await svc.add_example_channel("uzbekistan", "@kunuz")

    assert n == 50
    rows = await rt.store.list_examples(topic_id=topic.id)
    assert {e.kind for e in rows} == {"channel"}
    assert rows[0].text == "News item 10" and rows[-1].text == "News item 59"
    assert (await rt.store.get_topic(topic.id)).example_channel == "@kunuz"  # type: ignore[union-attr]
    assert rt.settings.topic("uzbekistan").example_channel == "@kunuz"  # type: ignore[union-attr]
    assert user_gw.calls_of("history")[0]["limit"] == 200

    with pytest.raises(NotAllowed) as not_member:
        await svc.add_example_channel("uzbekistan", "https://t.me/+private")
    assert not_member.value.reason == "not_a_member"


async def test_create_with_example_channel_stores_examples(
    rt: Runtime,
    svc: TopicsService,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
    clock: FakeClock,
) -> None:
    await svc.sync_from_settings()
    source = user_gw.add_chat(make_chat(title="Daryo", username="daryo"))
    user_gw.history_data[source.id].append(
        make_message(source, text="Tashkent metro opens", date=clock.now() - timedelta(days=1))
    )

    topic = await svc.create("Tashkent", example_channel="@daryo")

    rows = await rt.store.list_examples(topic_id=topic.id)
    assert [e.text for e in rows] == ["Tashkent metro opens"]
    assert topic.example_channel == "@daryo"
    assert rt.settings.topic("tashkent").example_channel == "@daryo"  # type: ignore[union-attr]


async def test_staging_channel_setting_must_be_owned(
    rt: Runtime, svc: TopicsService, user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    foreign = user_gw.add_chat(make_chat(title="Someone else's"))
    await rt.settings_file.set_value("publishing.staging_channel", foreign.id)
    with pytest.raises(ConfigError):
        await svc.ensure_staging_channel()
    assert foreign.id not in user_gw.owned
