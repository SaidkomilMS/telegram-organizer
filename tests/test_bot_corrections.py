"""bot/corrections.py: the [Wrong topic] menu on channel posts and preview items (§11.2, §9.8)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, FakeMessage, FakeUserGateway, make_callback
from tests.test_bot_topics import Driver, button_data, button_text, wire
from tg_curator.domain import CorrectionResult, NewPost, Post, PostStatus, Topic
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

CHANNEL_A = -1_009_000_000_001
CHANNEL_B = -1_009_000_000_002
STRANGER = 4242
TEXT = "The central bank kept its key rate unchanged and signalled cuts later in the year."


class Scene:
    """Three topics (A and B with channels, C without), one published post of A."""

    def __init__(self, rt: Runtime, drv: Driver, a: Topic, b: Topic, c: Topic) -> None:
        self.rt, self.drv, self.a, self.b, self.c = rt, drv, a, b, c
        self.corrections: list[tuple[int, int | None]] = []
        assert isinstance(rt.publisher, Publisher)
        self.publisher = rt.publisher

    async def post(self, status: PostStatus = PostStatus.held, message_id: int = 1) -> Post:
        new = NewPost(
            chat_id=-1_009_000_000_100,
            message_id=message_id,
            kind="post",
            message_ids=[message_id],
            posted_at=self.rt.clock.now(),
            via="live",
            text=TEXT,
            text_hash=text_hash(TEXT),
            urls=[],
        )
        post = await self.rt.store.insert_post(new, status=status, topic_id=self.a.id)
        assert post is not None
        return post

    async def published(self) -> tuple[Post, FakeMessage]:
        post = await self.post()
        assert await self.publisher.enqueue(post.id)
        self.rt.clock.advance(5)  # type: ignore[attr-defined]
        await self.publisher.tick()
        sent = self.drv.bot.sent(CHANNEL_A)
        assert len(sent) == 1 and button_data(sent[0]) == [f"wt:{post.id}"]
        return post, sent[0]

    async def tap(self, data: str, msg: FakeMessage, *, sender_id: int = OWNER_ID) -> None:
        await self.drv.bot.press(
            make_callback(data, chat_id=msg.chat_id, message_id=msg.message_id, sender_id=sender_id)
        )

    def answer(self) -> str | None:
        return self.drv.bot.answers[-1][1]


@pytest.fixture
async def scene(
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
) -> Scene:
    app = wire(rt)
    await app.start()
    source = make_chat(-1_009_000_000_100, title="Bank News")
    await rt.store.upsert_chat(source)
    topics = []
    for key, name, channel in (("a", "Finance", CHANNEL_A), ("b", "Markets", CHANNEL_B)):
        info = make_chat(channel, title=name, is_creator=True)
        await rt.store.upsert_chat(info, role="output")
        user_gw.register_owned(channel)
        topics.append(
            Topic(id=0, key=key, name=name, channel_id=channel, created_at=rt.clock.now())
        )
    topics.append(Topic(id=0, key="c", name="Tracked", channel_id=None, created_at=rt.clock.now()))
    a, b, c = [await rt.store.upsert_topic(t) for t in topics]
    rt.publisher = Publisher(rt)
    await rt.settings_file.set_value("publishing.live", True)
    s = Scene(rt, Driver(rt, bot_gw), a, b, c)

    assert rt.learning is not None
    real = rt.learning.correct

    async def spy(post_id: int, new_topic_id: int | None) -> CorrectionResult:
        s.corrections.append((post_id, new_topic_id))
        return await real(post_id, new_topic_id)

    rt.learning.correct = spy  # type: ignore[method-assign]
    return s


async def test_wrong_topic_opens_the_menu_in_place(scene: Scene) -> None:
    post, msg = await scene.published()
    before = msg.text
    await scene.tap(f"wt:{post.id}", msg)
    assert msg.text == before  # only the keyboard changed
    assert msg.buttons is not None
    assert [[b.data for b in row] for row in msg.buttons] == [
        [f"mv:{post.id}:{scene.b.id}", f"mv:{post.id}:{scene.c.id}"],
        [f"mv:{post.id}:0", f"mv:{post.id}:x"],
    ]
    assert button_text(msg, f"mv:{post.id}:0") == "Not for me"
    assert button_text(msg, f"mv:{post.id}:x") == "Cancel"
    assert scene.drv.sent() == []  # nothing is written to the owner chat or the channel
    assert len(scene.drv.bot.sent(CHANNEL_A)) == 1


async def test_cancel_restores_the_button(scene: Scene) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:x", msg)
    assert button_data(msg) == [f"wt:{post.id}"]
    assert button_text(msg, f"wt:{post.id}") == "Wrong topic"
    assert scene.corrections == []


async def test_a_choice_calls_correct_and_leaves_the_message_to_the_publisher(
    scene: Scene,
) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", msg)
    assert scene.corrections == [(post.id, scene.b.id)]
    assert "moved to" in msg.text and msg.buttons is None  # publisher.move stubbed it
    assert scene.answer() == "Now in Markets. The curator learns from this."
    fresh = await scene.rt.store.get_post(post.id)
    assert fresh is not None and fresh.topic_id == scene.b.id and fresh.corrected

    scene.rt.clock.advance(30)  # type: ignore[attr-defined]
    await scene.publisher.tick()
    moved = scene.drv.bot.sent(CHANNEL_B)
    assert len(moved) == 1 and TEXT in moved[0].text


async def test_not_for_me_rejects(scene: Scene) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:0", msg)
    assert scene.corrections == [(post.id, None)]
    fresh = await scene.rt.store.get_post(post.id)
    assert fresh is not None and fresh.status == PostStatus.rejected
    assert "not for me" in msg.text and msg.buttons is None
    assert scene.answer() == "Marked not for me. The curator learns from this."


async def test_only_the_owner_is_obeyed(scene: Scene) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg, sender_id=STRANGER)
    assert button_data(msg) == [f"wt:{post.id}"]
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", msg, sender_id=STRANGER)
    assert scene.corrections == []
    assert f"mv:{post.id}:{scene.b.id}" in button_data(msg)


async def test_a_repeat_is_refused_and_keeps_its_button(scene: Scene) -> None:
    post = await scene.post(PostStatus.duplicate)
    item = scene.drv.bot.world.add(OWNER_ID, sender="bot", html="an item")
    await scene.tap(f"wt:{post.id}", item)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", item)
    assert scene.corrections == [(post.id, scene.b.id)]
    assert (
        scene.answer() == "This post is a repeat of an earlier one; correct the original instead."
    )
    assert button_data(item) == [f"wt:{post.id}"]


async def test_in_the_private_chat_the_button_comes_back_after_a_choice(scene: Scene) -> None:
    post = await scene.post(PostStatus.digest)
    item = scene.drv.bot.world.add(OWNER_ID, sender="bot", html="a preview item")
    await scene.tap(f"wt:{post.id}", item)
    await scene.tap(f"mv:{post.id}:{scene.c.id}", item)
    assert scene.corrections == [(post.id, scene.c.id)]
    assert button_data(item) == [f"wt:{post.id}"]
    fresh = await scene.rt.store.get_post(post.id)
    assert fresh is not None and fresh.status == PostStatus.tracked  # C has no channel
    assert scene.answer() == "Now in Tracked. The curator learns from this."


async def test_a_topic_removed_meanwhile_is_answered(scene: Scene) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg)
    await scene.rt.store.set_topic_fields(scene.b.id, active=False)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", msg)
    assert scene.answer() == "That topic does not exist any more; pick another one."
    assert button_data(msg) == [f"wt:{post.id}"]


async def test_garbage_data_is_answered_not_crashed(scene: Scene) -> None:
    post, msg = await scene.published()
    await scene.tap("wt:abc", msg)
    await scene.tap(f"mv:{post.id}:zz", msg)
    assert scene.answer() == "That choice is not available any more."
    assert scene.corrections == []


async def test_a_sent_forward_is_relabelled_and_the_toast_says_it_stays(scene: Scene) -> None:
    # publisher.move never re-sends a forward that went out (§14.1): the toast must not
    # claim the post is now in the other topic's channel.
    await scene.rt.settings_file.set_value("publishing.style", "forward")
    post = await scene.post()
    assert await scene.publisher.enqueue(post.id)
    scene.rt.clock.advance(5)  # type: ignore[attr-defined]
    await scene.publisher.tick()
    row = await scene.rt.store.get_publication(post.id)
    assert row is not None and row.style == "forward" and row.message_ids
    msg = scene.drv.bot.sent(CHANNEL_A)[-1]  # the bot's companion carries [Wrong topic]
    assert button_data(msg) == [f"wt:{post.id}"]
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", msg)
    assert scene.answer() == (
        "A forwarded post cannot be moved: it stays here, marked as belonging in Markets. "
        "The curator learns from this."
    )
    scene.rt.clock.advance(30)  # type: ignore[attr-defined]
    await scene.publisher.tick()
    assert scene.drv.bot.sent(CHANNEL_B) == []
