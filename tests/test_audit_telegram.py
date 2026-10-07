"""Audit gaps of the telegram group: forum topics never merge into one unit (CG-9) and the
account's media copies into staging are paced like the bot's posts (SAFE-8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import errors, types

from tests.fakes import FakeBotGateway, FakeClock, FakeUserGateway, FakeWorld
from tests.test_intake import MORNING, FakeSorter, group_rows
from tests.test_publisher import CHANNEL_A, CHANNEL_B, STAGING, Harness
from tests.test_user_client import (
    CHANNEL_ID,
    OUTPUT_BARE,
    OUTPUT_ID,
    StubClient,
    _rpc,
    channel,
    chat_info,
    document_media,
    photo_media,
    tl_message,
)
from tg_curator.pipeline import publisher as publisher_mod
from tg_curator.pipeline.intake import IntakeService
from tg_curator.runtime import Runtime
from tg_curator.telegram import user_client
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.telegram.user_client import TelethonUserGateway, message_to_incoming, topic_of

NOW = MORNING

# --- CG-9: forum topics ----------------------------------------------------------------------


def test_topic_of_reads_the_forum_topic_from_the_reply_header() -> None:
    # a plain message in topic 10: the implicit "reply" to the topic root
    assert topic_of(types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=10)) == 10
    # a reply to message 55 inside topic 10
    in_topic = types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=55, reply_to_top_id=10)
    assert topic_of(in_topic) == 10
    # the General topic and ordinary groups: no topic
    assert topic_of(types.MessageReplyHeader(reply_to_msg_id=55)) is None
    assert topic_of(None) is None


def test_message_conversion_carries_the_topic_and_drops_the_implicit_reply() -> None:
    plain = tl_message(reply_to=types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=10))
    incoming = message_to_incoming(plain, chat_info(), now=NOW)
    assert incoming is not None and (incoming.topic_id, incoming.reply_to_id) == (10, None)
    reply = tl_message(
        reply_to=types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=55, reply_to_top_id=10)
    )
    incoming = message_to_incoming(reply, chat_info(), now=NOW)
    assert incoming is not None and (incoming.topic_id, incoming.reply_to_id) == (10, 55)
    incoming = message_to_incoming(tl_message(), chat_info(), now=NOW)
    assert incoming is not None and incoming.topic_id is None


@pytest.fixture
def forum_intake(rt: Runtime, clock: FakeClock) -> IntakeService:
    rt.sorter = FakeSorter(rt.store)
    clock.set(MORNING)
    rt.intake = IntakeService(rt)
    return rt.intake


async def test_one_sender_in_two_forum_topics_makes_two_units(
    rt: Runtime,
    forum_intake: IntakeService,
    clock: FakeClock,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., Any],
) -> None:
    group = make_chat(kind="group")
    plan = [(0, 10), (2, 20), (3, 10), (4, 20), (5, None)]  # (minute, topic)
    for minute, topic in plan:
        at = MORNING + timedelta(minutes=minute)
        clock.set(at)
        msg = make_message(group, text="x", sender_id=500, date=at)
        await forum_intake.handle_message(replace(msg, topic_id=topic))
    rows = await group_rows(rt.store, group.id)
    assert [(r.message_id, r.unit_root_id, r.topic_id) for r in rows] == [
        (1, 1, 10),
        (2, 2, 20),
        (3, 1, 10),
        (4, 2, 20),
        (5, 5, None),
    ]


# --- SAFE-8: the account's copies are paced --------------------------------------------------


class AlbumStub(StubClient):
    """Answers a list ``file`` like Telethon does: one message per album item."""

    async def send_file(self, entity: Any, file: Any, **kw: Any) -> Any:
        if not isinstance(file, list):
            return await super().send_file(entity, file, **kw)
        self.calls.append(("send_file", {"entity": entity, "file": file}))
        if self.send_file_errors:
            raise self.send_file_errors.pop(0)
        out = []
        for _ in file:
            self._next_id += 1
            out.append(SimpleNamespace(id=self._next_id))
        return out


@pytest.fixture
def album_stub() -> AlbumStub:
    return AlbumStub()


@pytest.fixture
def album_gw(home: Path, clock: FakeClock, album_stub: AlbumStub) -> TelethonUserGateway:
    gateway = TelethonUserGateway(home, 1, "hash", clock=clock, client_factory=lambda: album_stub)
    album_stub.entities[CHANNEL_ID] = channel()
    album_stub.entities[OUTPUT_ID] = channel(
        OUTPUT_BARE, title="Output", username=None, creator=True
    )
    gateway.register_owned(OUTPUT_ID)
    return gateway


async def test_an_album_is_copied_in_one_call(
    album_gw: TelethonUserGateway, album_stub: AlbumStub
) -> None:
    album_stub.history[CHANNEL_ID] = [
        tl_message(i, media=photo_media(), grouped_id=77) for i in (1, 2, 3)
    ]
    assert await album_gw.copy_media(CHANNEL_ID, [1, 2, 3], OUTPUT_ID) == [101, 102, 103]
    sends = [c for c in album_stub.calls if c[0] == "send_file"]
    assert len(sends) == 1 and len(sends[0][1]["file"]) == 3


async def test_an_expired_album_is_refetched_once(
    album_gw: TelethonUserGateway, album_stub: AlbumStub
) -> None:
    album_stub.history[CHANNEL_ID] = [
        tl_message(i, media=photo_media(), grouped_id=77) for i in (1, 2)
    ]
    album_stub.send_file_errors = [_rpc(errors.FileReferenceExpiredError)]
    assert await album_gw.copy_media(CHANNEL_ID, [1, 2], OUTPUT_ID) == [101, 102]
    fetches = [c for c in album_stub.calls if c[0] == "get_messages"]
    assert [c[1]["ids"] for c in fetches] == [[1, 2], [1, 2]]


async def test_separate_items_are_spaced(
    album_gw: TelethonUserGateway, album_stub: AlbumStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    pauses: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(user_client.asyncio, "sleep", fake_sleep)
    album_stub.history[CHANNEL_ID] = [
        tl_message(1, media=photo_media()),
        tl_message(2, media=document_media(video=True)),
        tl_message(3, media=photo_media()),
    ]
    assert await album_gw.copy_media(CHANNEL_ID, [1, 2, 3], OUTPUT_ID) == [101, 102, 103]
    assert len([c for c in album_stub.calls if c[0] == "send_file"]) == 3
    assert pauses == [user_client.COPY_ITEM_GAP_SECONDS] * 2
    assert user_client.COPY_ITEM_GAP_SECONDS >= 1


@pytest.fixture
async def h(
    rt: Runtime,
    clock: FakeClock,
    world: FakeWorld,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> Harness:
    harness = Harness(rt, clock, world, user_gw, bot_gw, make_chat)
    await harness.setup()
    return harness


def _max_per_minute(dates: list[datetime]) -> int:
    dates = sorted(dates)
    return max(
        (sum(1 for d in dates if start <= d < start + timedelta(minutes=1)) for start in dates),
        default=0,
    )


async def test_album_posts_across_channels_stay_under_the_per_minute_cap(h: Harness) -> None:
    for i in range(8):
        first = 100 + i * 10
        post = await h.post(
            f"album story {i}",
            media="album",
            message_id=first,
            message_ids=list(range(first, first + 10)),
            topic=h.topic_a if i % 2 == 0 else h.topic_b,
        )
        assert await h.pub.enqueue(post.id)
    for _ in range(150):  # five minutes of 2-second ticks
        h.clock.advance(2)
        await h.pub.tick()
    staging = h.world.sent(STAGING)
    cap = publisher_mod.MAX_WRITES_PER_MINUTE
    assert len(staging) >= 40  # the media did go out, only spaced
    assert _max_per_minute([m.date for m in staging]) <= cap
    for target in (CHANNEL_A, CHANNEL_B):
        sent = h.sent(target)
        assert sent and _max_per_minute([m.date for m in sent]) <= cap
    # every album that went out kept all of its media
    for target in (CHANNEL_A, CHANNEL_B):
        assert len([m for m in h.sent(target) if m.media == "album"]) % 10 == 0
