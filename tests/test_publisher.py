"""The outbox (DESIGN §9.4, §14.1, §14.6, §16 invariants) against the fakes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeClock,
    FakeUserGateway,
    FakeWorld,
    hrefs,
    plain_text,
)
from tg_curator.domain import (
    KV,
    PUB_CANCELLED,
    PUB_FAILED,
    PUB_PENDING,
    PUB_RETRACTED,
    PUB_SENDING,
    PUB_SENT,
    NewPost,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.errors import CuratorError, FloodWait, MediaUnavailable, TelegramUnavailable
from tg_curator.pipeline import render
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

CHANNEL_A = -1_001_000_000_901
CHANNEL_B = -1_001_000_000_902
STAGING = -1_001_000_000_950
LONG_TEXT = "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(6))  # > 1024 units
BACKOFF_CAP = 600  # seconds: past any retry's backoff


class Harness:
    """A live runtime with two topics (each with a channel), a staging channel and a source."""

    def __init__(
        self,
        rt: Runtime,
        clock: FakeClock,
        world: FakeWorld,
        user_gw: FakeUserGateway,
        bot_gw: FakeBotGateway,
        make_chat: Callable[..., ChatInfo],
    ) -> None:
        self.rt, self.clock, self.world = rt, clock, world
        self.user, self.bot = user_gw, bot_gw
        self.make_chat = make_chat
        self.pub = Publisher(rt)
        self.topic_a: Topic
        self.topic_b: Topic
        self.topic_none: Topic
        self.source: ChatInfo

    async def setup(self, *, live: bool = True, staging: int = STAGING) -> None:
        store = self.rt.store
        sf = self.rt.settings_file
        await sf.set_value("publishing.live", live)
        await sf.set_value("publishing.staging_channel", staging)
        now = self.clock.now()
        self.topic_a = await store.upsert_topic(
            Topic(id=0, key="ml-ai", name="ML & AI", channel_id=CHANNEL_A, created_at=now)
        )
        self.topic_b = await store.upsert_topic(
            Topic(id=0, key="fintech", name="Fintech", channel_id=CHANNEL_B, created_at=now)
        )
        self.topic_none = await store.upsert_topic(
            Topic(id=0, key="tracked", name="Tracked only", created_at=now)
        )
        for cid, title in [(CHANNEL_A, "ML & AI"), (CHANNEL_B, "Fintech"), (STAGING, "media")]:
            info = self.make_chat(cid, title=title, is_creator=True, is_admin=True)
            self.user.add_chat(info)
            await store.upsert_chat(info, role="staging" if cid == STAGING else "output")
            self.user.register_owned(cid)
        self.source = self.make_chat(title="Kun.uz", username="kunuz")
        self.user.add_chat(self.source)
        await store.upsert_chat(self.source)

    async def post(
        self,
        text: str = "A fresh story from the source.",
        *,
        media: str | None = None,
        message_id: int | None = None,
        message_ids: list[int] | None = None,
        chat: ChatInfo | None = None,
        topic: Topic | None = None,
        corroboration: int = 0,
        noforwards: bool = False,
        posted_at: Any = None,
        html: str | None = None,
    ) -> Post:
        chat = chat or self.source
        mid = message_id or self.world.next_id(chat.id)
        ids = message_ids or [mid]
        for i in ids:
            self.world.add(
                chat.id,
                sender="source",
                message_id=i,
                html=html,
                text=text,
                media=media,
                noforwards=noforwards or chat.noforwards,
            )
        new = NewPost(
            chat_id=chat.id,
            message_id=mid,
            kind="post",
            message_ids=ids,
            posted_at=posted_at or self.clock.now(),
            via="live",
            text=text,
            html=html,
            text_hash=text_hash(text),
            urls=[],
            media=media,
            noforwards=noforwards or chat.noforwards,
        )
        post = await self.rt.store.insert_post(
            new,
            status=PostStatus.held,
            topic_id=(topic or self.topic_a).id,
            corroboration=corroboration,
            corroborating_chats=[-1 - k for k in range(corroboration)],
        )
        assert post is not None
        return post

    async def publish(self, post: Post, *, ticks: int = 1) -> Post:
        """Enqueue, move the clock a little and tick until the post is sent."""
        assert await self.pub.enqueue(post.id)
        for _ in range(ticks):
            self.clock.advance(5)
            await self.pub.tick()
        return await self.fresh(post)

    async def fresh(self, post: Post) -> Post:
        p = await self.rt.store.get_post(post.id)
        assert p is not None
        return p

    async def row(self, post: Post) -> Any:
        return await self.rt.store.get_publication(post.id)

    def sent(self, channel: int = CHANNEL_A) -> list[Any]:
        return self.bot.sent(channel)


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


def wrong_topic_button(msg: Any) -> bool:
    return bool(msg.buttons) and msg.buttons[0][0].data.startswith("wt:")


# --- enqueue ---------------------------------------------------------------------------------


async def test_enqueue_is_idempotent_and_sets_queued(h: Harness) -> None:
    post = await h.post()
    assert await h.pub.enqueue(post.id) is True
    assert (await h.fresh(post)).status == PostStatus.queued
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.channel_id is None and row.style is None
    assert await h.pub.enqueue(post.id) is False


async def test_enqueue_refuses_a_cancelled_row(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, state=PUB_CANCELLED)
    await h.rt.store.set_post_fields(post.id, status=PostStatus.digest)
    assert await h.pub.enqueue(post.id) is False
    assert (await h.fresh(post)).status == PostStatus.digest


async def test_enqueue_without_topic_is_refused(h: Harness) -> None:
    post = await h.post()
    await h.rt.store.set_post_fields(post.id, topic_id=None)
    assert await h.pub.enqueue(post.id) is False


# --- live / paused -----------------------------------------------------------------------------


async def test_nothing_is_posted_before_go(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.live", False)
    post = await h.post()
    await h.publish(post, ticks=3)
    assert h.sent() == []
    assert (await h.fresh(post)).status == PostStatus.queued
    await h.rt.settings_file.set_value("publishing.live", True)
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent()) == 1


async def test_nothing_is_posted_while_paused(h: Harness) -> None:
    await h.rt.store.kv_set(KV.SERVICE_PAUSED, True)
    post = await h.post()
    await h.publish(post, ticks=2)
    assert h.sent() == []
    await h.rt.store.kv_delete(KV.SERVICE_PAUSED)
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent()) == 1


# --- repost style ------------------------------------------------------------------------------


async def test_text_only_post_is_one_message_with_button(h: Harness) -> None:
    post = await h.post(corroboration=2)
    post = await h.publish(post)
    assert post.status == PostStatus.published and post.published_at == h.clock.now()
    msgs = h.sent()
    assert len(msgs) == 1
    assert msgs[0].text.startswith("Kun.uz · source · +2 more\nA fresh story")
    assert hrefs(msgs[0].html) == ["https://t.me/kunuz/1"]
    assert msgs[0].buttons == [[msgs[0].buttons[0][0]]]
    assert msgs[0].buttons[0][0].data == f"wt:{post.id}" and msgs[0].buttons[0][0].text
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [msgs[0].message_id]
    assert row.channel_id == CHANNEL_A and row.style == "repost"
    assert row.shown_corroboration == 2 and row.sent_at == h.clock.now()
    assert row.staging_ids == []
    assert h.user.calls_of("copy_media") == []


async def test_media_with_short_text_is_one_captioned_message(h: Harness) -> None:
    post = await h.post(media="photo")
    post = await h.publish(post)
    copies = h.user.calls_of("copy_media")
    assert copies == [
        {"from_chat_id": h.source.id, "message_ids": [post.message_id], "to_chat_id": STAGING}
    ]
    staged = h.world.sent(STAGING)
    assert len(staged) == 1 and staged[0].media == "photo"
    msgs = h.sent()
    assert len(msgs) == 1
    assert msgs[0].media == "photo" and msgs[0].copied_from == (STAGING, staged[0].message_id)
    assert "📎 photo attached" in msgs[0].text
    assert wrong_topic_button(msgs[0])
    row = await h.row(post)
    assert row.message_ids == [msgs[0].message_id]
    assert row.staging_ids == [staged[0].message_id]
    assert h.bot.calls_of("send_text") == []


async def test_long_text_with_media_goes_media_first_then_text(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="video")
    post = await h.publish(post)
    msgs = h.sent()
    assert len(msgs) == 2
    media, text = msgs
    assert media.media == "video" and media.html is None and media.buttons is None
    assert text.media is None and wrong_topic_button(text)
    assert "🎬 video attached" in text.text and "Kun.uz · source" in text.text
    row = await h.row(post)
    assert row.message_ids == [media.message_id, text.message_id]


async def test_very_long_text_is_split_with_the_button_on_the_last(h: Harness) -> None:
    body = "\n\n".join(f"paragraph {i} " + "y" * 300 for i in range(40))
    post = await h.post(body, media="photo")
    post = await h.publish(post)
    msgs = h.sent()
    assert msgs[0].media == "photo"
    texts = msgs[1:]
    assert len(texts) >= 3
    assert all(not wrong_topic_button(m) for m in texts[:-1])
    assert wrong_topic_button(texts[-1])
    assert texts[-1].text.startswith("Kun.uz · source\n📎 photo attached\n")
    assert all(render.plain_units(m.html) <= render.TEXT_LIMIT for m in texts)
    row = await h.row(post)
    assert row.message_ids == [m.message_id for m in msgs]
    assert row.message_ids[-1] == texts[-1].message_id


async def test_album_never_carries_buttons(h: Harness) -> None:
    post = await h.post(media="album", message_ids=[10, 11, 12], message_id=10)
    post = await h.publish(post)
    assert h.user.calls_of("copy_media")[0]["message_ids"] == [10, 11, 12]
    msgs = h.sent()
    assert len(msgs) == 4
    assert all(m.media == "album" and m.buttons is None and m.html is None for m in msgs[:3])
    assert wrong_topic_button(msgs[3]) and "🖼 album attached" in msgs[3].text
    row = await h.row(post)
    assert len(row.staging_ids) == 3 and row.message_ids[-1] == msgs[3].message_id


async def test_protected_source_is_text_and_link(h: Harness, make_chat: Any) -> None:
    protected = make_chat(title="Secret", noforwards=True)
    h.user.add_chat(protected)
    await h.rt.store.upsert_chat(protected)
    post = await h.post(media="photo", chat=protected)
    await h.publish(post)
    assert h.user.calls_of("copy_media") == []
    msgs = h.sent()
    assert len(msgs) == 1 and msgs[0].media is None
    assert "📎 photo at source" in msgs[0].text
    assert hrefs(msgs[0].html) == [f"https://t.me/c/{-1_000_000_000_000 - protected.id}/1"]
    assert wrong_topic_button(msgs[0])


async def test_media_copy_failure_falls_back_to_text(h: Harness) -> None:
    h.user.fail_next("copy_media", MediaUnavailable("expired"))
    post = await h.post(media="file")
    post = await h.publish(post)
    assert post.status == PostStatus.published
    msgs = h.sent()
    assert len(msgs) == 1 and msgs[0].media is None
    assert "📎 file at source" in msgs[0].text
    assert (await h.row(post)).staging_ids == []


async def test_bot_copy_failure_falls_back_to_text(h: Harness) -> None:
    h.bot.fail_next("send_copy", MediaUnavailable("gone from staging"))
    post = await h.post(media="photo")
    post = await h.publish(post)
    msgs = h.sent()
    assert len(msgs) == 1 and msgs[0].media is None and "at source" in msgs[0].text
    assert (await h.row(post)).staging_ids == []


async def test_missing_staging_channel_skips_media_with_one_warning(
    h: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    await h.rt.settings_file.set_value("publishing.staging_channel", 0)
    caplog.set_level("WARNING")
    p1 = await h.post(media="photo")
    p2 = await h.post(media="photo")
    await h.publish(p1)
    h.clock.advance(30)
    await h.publish(p2)
    assert h.user.calls_of("copy_media") == []
    assert all(m.media is None for m in h.sent())
    warnings = [r for r in caplog.records if "staging" in r.getMessage()]
    assert len(warnings) == 1


async def test_bot_not_admin_of_staging_skips_media(h: Harness) -> None:
    h.bot.cannot_post.add(STAGING)
    post = await h.post(media="photo")
    post = await h.publish(post)
    assert post.status == PostStatus.published
    assert h.user.calls_of("copy_media") == []
    assert h.sent()[0].media is None


async def test_other_media_is_noted_not_copied(h: Harness) -> None:
    post = await h.post(media="other")
    await h.publish(post)
    assert h.user.calls_of("copy_media") == []
    assert "📎 media at source" in h.sent()[0].text


# --- forward style -----------------------------------------------------------------------------


async def test_forward_style_forwards_then_adds_the_button_message(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post(media="album", message_ids=[20, 21], message_id=20, corroboration=3)
    post = await h.publish(post)
    assert h.user.calls_of("forward") == [
        {"from_chat_id": h.source.id, "message_ids": [20, 21], "to_chat_id": CHANNEL_A}
    ]
    *forwards, companion = h.sent()
    assert len(forwards) == 2
    assert all(m.sender == "user" and m.buttons is None and m.fwd_of for m in forwards)
    # the bot's message under the forward carries what a forward cannot: +N and the button
    assert companion.sender == "bot" and wrong_topic_button(companion) and companion.silent
    assert companion.text == "Kun.uz · source · +3 more"
    assert hrefs(companion.html) == ["https://t.me/kunuz/20"]
    assert h.bot.calls_of("send_copy") == []
    row = await h.row(post)
    assert row.style == "forward" and row.state == PUB_SENT
    assert row.message_ids == [m.message_id for m in forwards] + [companion.message_id]


async def test_forward_style_falls_back_to_repost_for_protected_sources(
    h: Harness, make_chat: Any
) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    protected = make_chat(title="Secret", noforwards=True)
    h.user.add_chat(protected)
    await h.rt.store.upsert_chat(protected)
    post = await h.post(chat=protected)
    post = await h.publish(post)
    assert h.user.calls_of("forward") == []
    assert (await h.row(post)).style == "repost"
    assert wrong_topic_button(h.sent()[0])


async def test_forward_style_reposts_the_version_first_seen_when_the_source_is_gone(
    h: Harness,
) -> None:
    """Deleted at the source during the hold: the stored text still goes out (SPEC: deleted
    posts are not removed), as a repost committed before the bot sends it (§18.4)."""
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post("The story as it was first seen.")
    h.user.deleted_sources.add((post.chat_id, post.message_id))
    post = await h.publish(post)
    assert post.status == PostStatus.published
    row = await h.row(post)
    assert row.style == "repost" and row.state == PUB_SENT
    [message] = h.sent()
    assert message.sender == "bot" and "The story as it was first seen." in message.text
    assert wrong_topic_button(message)


async def test_forward_rows_get_plus_n_on_their_button_message(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post()
    post = await h.publish(post)
    forward, companion = h.sent()
    await h.rt.store.add_corroboration(post.id, -5)
    await h.pub.note_corroboration(post.id)
    h.clock.advance(120)
    await h.pub.tick()
    edits = h.bot.calls_of("edit_text")
    assert [e["message_id"] for e in edits] == [companion.message_id]
    assert h.world.get(CHANNEL_A, companion.message_id).text == "Kun.uz · source · +1 more"
    assert wrong_topic_button(h.world.get(CHANNEL_A, companion.message_id))
    assert h.world.get(CHANNEL_A, forward.message_id).edits == 0
    assert (await h.row(post)).shown_corroboration == 1


# --- pacing ------------------------------------------------------------------------------------


async def test_same_channel_posts_are_spaced_by_min_gap(h: Harness) -> None:
    p1, p2 = await h.post("one"), await h.post("two")
    await h.pub.enqueue(p1.id)
    await h.pub.enqueue(p2.id)
    await h.pub.tick()
    assert len(h.sent()) == 1
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent()) == 1  # min_gap_seconds = 20 not reached
    h.clock.advance(15)
    await h.pub.tick()
    assert len(h.sent()) == 2
    assert [plain_text(m.html).splitlines()[-1] for m in h.sent()] == ["one", "two"]


async def test_global_gap_between_sends_into_different_channels(h: Harness) -> None:
    p1 = await h.post("a")
    p2 = await h.post("b", topic=h.topic_b)
    await h.pub.enqueue(p1.id)
    await h.pub.enqueue(p2.id)
    await h.pub.tick()
    assert len(h.sent(CHANNEL_A)) == 1 and h.sent(CHANNEL_B) == []
    h.clock.advance(2)
    await h.pub.tick()
    assert h.sent(CHANNEL_B) == []
    h.clock.advance(1)
    await h.pub.tick()
    assert len(h.sent(CHANNEL_B)) == 1


async def test_per_channel_ceiling_of_fifteen_writes_per_minute(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.min_gap_seconds", 1)
    posts = [await h.post(f"story {i}") for i in range(17)]
    for p in posts:
        await h.pub.enqueue(p.id)
    for _ in range(16):
        h.clock.advance(3)
        await h.pub.tick()
    assert len(h.sent()) == 15
    h.clock.advance(30)
    await h.pub.tick()
    assert len(h.sent()) == 16


# --- +N edits ----------------------------------------------------------------------------------


async def test_corroboration_edits_the_last_message_header(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="photo")
    post = await h.publish(post)
    media, text = h.sent()
    assert "more" not in text.text
    assert await h.rt.store.add_corroboration(post.id, -7) == 1
    await h.pub.note_corroboration(post.id)
    h.clock.advance(30)
    await h.pub.tick()
    edits = h.bot.calls_of("edit_text")
    assert len(edits) == 1 and edits[0]["message_id"] == text.message_id
    assert edits[0]["buttons"][0][0].data == f"wt:{post.id}"
    text = h.world.get(CHANNEL_A, text.message_id)
    assert text.text.startswith("Kun.uz · source · +1 more\n📎 photo attached\n")
    assert text.text.endswith("x" * 300)
    assert h.world.get(CHANNEL_A, media.message_id).edits == 0
    row = await h.row(post)
    assert row.shown_corroboration == 1 and row.edited_at == h.clock.now()


async def test_captioned_media_edit_keeps_the_caption(h: Harness) -> None:
    post = await h.post(media="photo")
    post = await h.publish(post)
    await h.rt.store.add_corroboration(post.id, -7)
    await h.rt.store.add_corroboration(post.id, -8)
    h.clock.advance(30)
    await h.pub.tick()
    msg = h.sent()[0]
    assert msg.media == "photo" and msg.text.startswith("Kun.uz · source · +2 more\n📎 photo")


async def test_edit_waits_for_the_cooldown_and_the_budget(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    await h.rt.store.add_corroboration(post.id, -7)
    await h.pub.tick()  # the send just happened: channel budget busy, no edit yet
    assert h.bot.calls_of("edit_text") == []
    h.clock.advance(20)
    await h.pub.tick()
    assert len(h.bot.calls_of("edit_text")) == 1
    await h.rt.store.add_corroboration(post.id, -8)
    h.clock.advance(30)
    await h.pub.tick()  # within the 60 s edit cooldown
    assert len(h.bot.calls_of("edit_text")) == 1
    h.clock.advance(31)
    await h.pub.tick()
    assert len(h.bot.calls_of("edit_text")) == 2
    assert "+2 more" in h.sent()[0].text


async def test_edit_is_a_write_that_shares_the_channel_budget(h: Harness) -> None:
    p1 = await h.post("first")
    p1 = await h.publish(p1)
    p2 = await h.post("second")
    await h.pub.enqueue(p2.id)
    await h.rt.store.add_corroboration(p1.id, -7)
    h.clock.advance(20)
    await h.pub.tick()  # the send wins the budget; the edit waits
    assert len(h.sent()) == 2 and h.bot.calls_of("edit_text") == []
    h.clock.advance(20)
    await h.pub.tick()
    assert len(h.bot.calls_of("edit_text")) == 1


async def test_retracted_and_digested_rows_are_never_edited(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    await h.pub.move(post.id, None)
    await h.rt.store.add_corroboration(post.id, -7)
    h.clock.advance(120)
    await h.pub.tick()
    edits = h.bot.calls_of("edit_text")
    assert len(edits) == 1 and plain_text(edits[0]["html"]) == "✕ not for me"


async def test_message_not_modified_is_ignored(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, shown_corroboration=0)
    await h.rt.store.set_post_fields(post.id, corroboration=0)
    h.clock.advance(60)
    await h.pub.tick()
    assert h.bot.calls_of("edit_text") == []


# --- failures ----------------------------------------------------------------------------------


async def test_bot_cannot_post_notifies_once_and_retries_every_ten_minutes(h: Harness) -> None:
    h.bot.cannot_post.add(CHANNEL_A)
    p1, p2 = await h.post("a"), await h.post("b")
    await h.pub.enqueue(p1.id)
    await h.pub.enqueue(p2.id)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(p1)
    assert row.state == PUB_FAILED and row.attempts == 1
    assert row.next_attempt_at == h.clock.now() + timedelta(minutes=10)
    assert "BotCannotPost" in row.last_error
    assert (await h.fresh(p1)).status == PostStatus.queued
    owner = h.bot.sent(OWNER_ID)
    assert len(owner) == 1 and "ML &amp; AI" in owner[0].html
    h.clock.advance(5)
    await h.pub.tick()  # p2 fails too: no second notification for the same channel
    assert (await h.row(p2)).state == PUB_FAILED
    assert len(h.bot.sent(OWNER_ID)) == 1
    h.clock.advance(60)
    await h.pub.tick()  # not due yet
    assert (await h.row(p1)).attempts == 1
    h.clock.advance(600)
    await h.pub.tick()
    assert (await h.row(p1)).attempts == 2
    assert len(h.bot.sent(OWNER_ID)) == 1
    h.bot.cannot_post.discard(CHANNEL_A)
    h.clock.advance(601)
    await h.pub.tick()
    assert (await h.row(p1)).state == PUB_SENT
    assert (await h.fresh(p1)).status == PostStatus.published
    h.bot.cannot_post.add(CHANNEL_A)
    h.clock.advance(601)
    await h.pub.tick()  # the channel broke again: a fresh notification
    assert len(h.bot.sent(OWNER_ID)) == 2


async def test_after_six_hours_the_post_falls_into_the_digest(h: Harness) -> None:
    h.bot.cannot_post.add(CHANNEL_A)
    post = await h.post()
    await h.pub.enqueue(post.id)
    for _ in range(35):  # 35 × 601 s ≈ 5.8 h of retries every ten minutes
        h.clock.advance(601)
        await h.pub.tick()
    assert (await h.row(post)).state == PUB_FAILED
    h.clock.advance(601)
    await h.pub.tick()  # past six hours
    row = await h.row(post)
    assert row.state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest
    assert h.sent() == []


async def test_long_flood_wait_sets_next_attempt_without_sleeping(h: Harness) -> None:
    h.bot.fail_next("send_text", FloodWait(300))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    row = await h.row(post)
    assert row.state == PUB_FAILED
    assert row.next_attempt_at == h.clock.now() + timedelta(seconds=300)
    h.clock.advance(299)
    await h.pub.tick()
    assert h.sent() == []
    h.clock.advance(1)
    await h.pub.tick()
    assert len(h.sent()) == 1
    assert h.bot.sent(OWNER_ID) == []


async def test_transient_failures_back_off_exponentially(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    expected = [30, 60, 120]
    for attempt, delay in enumerate(expected, 1):
        h.bot.fail_next("send_text", CuratorError("hiccup"))
        h.clock.advance(delay + 1)
        await h.pub.tick()
        row = await h.row(post)
        assert row.state == PUB_FAILED and row.attempts == attempt
        assert row.next_attempt_at == h.clock.now() + timedelta(seconds=delay)
    h.clock.advance(121)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_SENT
    assert len(h.sent()) == 1


async def test_topic_without_channel_cancels_into_tracked(h: Harness) -> None:
    post = await h.post(topic=h.topic_none)
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.tracked


async def test_inactive_topic_cancels_into_unsorted(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    await h.rt.store.set_topic_fields(h.topic_a.id, active=False)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.unsorted
    assert h.sent() == []


# --- reconcile: never published twice ----------------------------------------------------------


async def test_crash_after_send_is_reconciled_by_the_link(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    row = await h.row(post)
    sent_id = row.message_ids[0]
    # simulate the crash between Telegram's answer and the commit
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, message_ids=[], sent_at=None)
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued, published_at=None)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [sent_id]
    assert (await h.fresh(post)).status == PostStatus.published
    find = h.user.calls_of("find_message")
    assert find[0]["contains"] == "https://t.me/kunuz/1" and find[0]["chat_id"] == CHANNEL_A
    h.clock.advance(30)
    await restarted.tick()
    assert len(h.sent()) == 1


async def test_crash_before_send_goes_back_to_pending_and_is_sent_once(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(
        row.id, state=PUB_SENDING, channel_id=CHANNEL_A, style="repost", attempts=1
    )
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    assert (await h.row(post)).state == PUB_PENDING
    h.clock.advance(30)
    await restarted.tick()
    await restarted.tick()
    assert len(h.sent()) == 1
    assert (await h.row(post)).state == PUB_SENT


async def test_forward_crash_is_reconciled_by_the_forward_origin(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post()
    post = await h.publish(post)
    row = await h.row(post)
    fwd_id, companion_id = row.message_ids
    # the crash came between the forward and the commit of its id: no button message yet
    h.world.messages[CHANNEL_A] = [m for m in h.sent() if m.message_id != companion_id]
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, message_ids=[])
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.message_ids == [fwd_id]
    find = h.user.calls_of("find_message")
    assert find[0]["fwd_of"] == (h.source.id, post.message_id)
    h.clock.advance(30)
    await restarted.tick()
    assert len(h.user.calls_of("forward")) == 1  # never forwarded again
    forward, companion = h.sent()
    assert forward.message_id == fwd_id and wrong_topic_button(companion)
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [fwd_id, companion.message_id]


async def test_forward_crash_after_the_button_message_is_reconciled(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post()
    post = await h.publish(post)
    row = await h.row(post)
    fwd_id, companion_id = row.message_ids
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, message_ids=[fwd_id])
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [fwd_id, companion_id]
    h.clock.advance(30)
    await restarted.tick()
    assert len(h.sent()) == 2


async def test_stale_queued_posts_fall_into_the_digest_at_restart(h: Harness) -> None:
    old = await h.post("old", posted_at=h.clock.now() - timedelta(minutes=50))
    fresh = await h.post("fresh", posted_at=h.clock.now() - timedelta(minutes=10))
    await h.pub.enqueue(old.id)
    await h.pub.enqueue(fresh.id)
    await Publisher(h.rt).reconcile()
    assert (await h.row(old)).state == PUB_CANCELLED
    assert (await h.fresh(old)).status == PostStatus.digest
    assert (await h.row(fresh)).state == PUB_PENDING
    assert (await h.fresh(fresh)).status == PostStatus.queued
    h.clock.advance(5)
    await h.pub.tick()
    assert [plain_text(m.html).splitlines()[-1] for m in h.sent()] == ["fresh"]
    assert await h.pub.enqueue(old.id) is False


async def test_reconcile_without_an_account_leaves_sending_alone(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, channel_id=CHANNEL_A)
    h.rt.user = None
    await Publisher(h.rt).reconcile()
    assert (await h.row(post)).state == PUB_SENDING


# --- move ----------------------------------------------------------------------------------------


async def test_move_stubs_a_captioned_post_and_republishes_in_the_new_channel(
    h: Harness,
) -> None:
    post = await h.post(media="photo")
    post = await h.publish(post)
    old_msg = h.sent()[0]
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and result.stubbed and result.new_message_ids == []
    old_msg = h.world.get(CHANNEL_A, old_msg.message_id)
    assert old_msg.text == "↪ moved to Fintech" and old_msg.buttons is None
    assert old_msg.media == "photo"
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.topic_id == h.topic_b.id
    assert row.channel_id == CHANNEL_B and row.message_ids == [] and row.attempts == 0
    assert len(row.staging_ids) == 1
    assert row.moved_from == [
        {"channel_id": CHANNEL_A, "message_ids": [old_msg.message_id], "topic_id": h.topic_a.id}
    ]
    assert (await h.fresh(post)).status == PostStatus.queued
    h.clock.advance(5)
    await h.pub.tick()
    new = h.sent(CHANNEL_B)
    assert len(new) == 1 and new[0].media == "photo" and wrong_topic_button(new[0])
    assert h.user.calls_of("copy_media") and len(h.user.calls_of("copy_media")) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [new[0].message_id]
    assert (await h.fresh(post)).status == PostStatus.published


async def test_move_media_first_post_captions_media_and_stubs_text(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="album", message_ids=[30, 31], message_id=30)
    post = await h.publish(post)
    m1, m2, text = h.sent()
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.stubbed
    assert h.world.get(CHANNEL_A, m1.message_id).text == "↪ moved"
    assert h.world.get(CHANNEL_A, m2.message_id).text == "↪ moved"
    stubbed = h.world.get(CHANNEL_A, text.message_id)
    assert stubbed.text == "↪ moved to Fintech" and stubbed.buttons is None


async def test_move_not_for_me_retracts_and_rejects(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="photo")
    post = await h.publish(post)
    media, text = h.sent()
    result = await h.pub.move(post.id, None)
    assert result.stubbed and not result.republished
    assert h.world.get(CHANNEL_A, media.message_id).text == "✕ not for me"
    assert h.world.get(CHANNEL_A, text.message_id).text == "✕ not for me"
    row = await h.row(post)
    assert row.state == PUB_RETRACTED
    assert row.message_ids == [media.message_id, text.message_id]
    assert row.moved_from[0]["topic_id"] == h.topic_a.id
    assert (await h.fresh(post)).status == PostStatus.rejected
    h.clock.advance(60)
    await h.pub.tick()
    assert h.sent(CHANNEL_B) == [] and len(h.sent()) == 2


async def test_retracted_post_can_be_republished_later(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    await h.pub.move(post.id, None)
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and not result.stubbed
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.topic_id == h.topic_b.id
    assert len(row.moved_from) == 1
    assert (await h.fresh(post)).status == PostStatus.queued
    h.clock.advance(5)
    await h.pub.tick()
    assert len(h.sent(CHANNEL_B)) == 1
    assert len(h.bot.calls_of("edit_text")) == 1  # only the one stub


async def test_move_to_a_topic_without_channel_cancels_into_tracked(h: Harness) -> None:
    post = await h.post()
    post = await h.publish(post)
    result = await h.pub.move(post.id, h.topic_none.id)
    assert result.stubbed and not result.republished
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.tracked
    assert plain_text(h.sent()[0].html) == "↪ moved to Tracked only"


async def test_move_pending_row_just_repoints(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result.republished and not result.stubbed
    assert h.bot.calls_of("edit_text") == []
    row = await h.row(post)
    assert row.topic_id == h.topic_b.id and row.state == PUB_PENDING
    h.clock.advance(5)
    await h.pub.tick()
    assert h.sent(CHANNEL_A) == [] and len(h.sent(CHANNEL_B)) == 1


async def test_move_cancelled_row_is_left_alone(h: Harness) -> None:
    post = await h.post()
    await h.pub.enqueue(post.id)
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, state=PUB_CANCELLED)
    await h.rt.store.set_post_fields(post.id, status=PostStatus.digest)
    result = await h.pub.move(post.id, h.topic_b.id)
    assert result == type(result)(republished=False, stubbed=False, new_message_ids=[])
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest


async def test_move_without_a_row_does_nothing(h: Harness) -> None:
    post = await h.post()
    result = await h.pub.move(post.id, h.topic_b.id)
    assert not result.republished and not result.stubbed


async def test_moved_forward_stays_and_is_relabelled_never_republished(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post("unique story text")
    post = await h.publish(post)
    forward, companion = h.sent()
    result = await h.pub.move(post.id, h.topic_b.id)
    assert not result.republished and result.stubbed
    assert [e["message_id"] for e in h.bot.calls_of("edit_text")] == [companion.message_id]
    stub = h.world.get(CHANNEL_A, companion.message_id)
    assert stub.text == "↪ belongs in Fintech" and stub.buttons is None
    assert h.world.get(CHANNEL_A, forward.message_id).text == "unique story text"
    row = await h.row(post)
    assert row.state == PUB_CANCELLED and row.topic_id == h.topic_b.id
    assert (await h.fresh(post)).status == PostStatus.published
    h.clock.advance(60)
    await h.pub.tick()
    assert h.sent(CHANNEL_B) == [] and len(h.user.calls_of("forward")) == 1


async def test_moved_forward_not_for_me_stubs_its_button_message(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    post = await h.post()
    post = await h.publish(post)
    _, companion = h.sent()
    result = await h.pub.move(post.id, None)
    assert result.stubbed and not result.republished
    assert h.world.get(CHANNEL_A, companion.message_id).text == "✕ not for me"
    assert (await h.row(post)).state == PUB_RETRACTED
    # a later correction to a topic with a channel still never forwards it a second time
    result = await h.pub.move(post.id, h.topic_b.id)
    assert not result.republished
    h.clock.advance(60)
    await h.pub.tick()
    assert h.sent(CHANNEL_B) == [] and len(h.user.calls_of("forward")) == 1


@pytest.mark.parametrize("style", ["repost", "forward"])
async def test_post_is_never_in_two_channels_at_once(h: Harness, style: str) -> None:
    await h.rt.settings_file.set_value("publishing.style", style)
    post = await h.post("unique story text")
    post = await h.publish(post)
    await h.pub.move(post.id, h.topic_b.id)
    h.clock.advance(5)
    await h.pub.tick()
    in_a = [m for m in h.sent(CHANNEL_A) if "unique story text" in m.text]
    in_b = [m for m in h.sent(CHANNEL_B) if "unique story text" in m.text]
    assert len(in_a) + len(in_b) == 1
    if style == "repost":
        assert in_a == [] and len(in_b) == 1


# --- misc --------------------------------------------------------------------------------------


async def test_tick_is_a_noop_without_rows(h: Harness) -> None:
    await h.pub.tick()
    assert h.bot.calls == [("start", {})]


async def test_send_failure_keeps_the_post_queued_and_the_row_retryable(h: Harness) -> None:
    h.bot.fail_next("send_text", CuratorError("boom"))
    post = await h.post()
    post = await h.publish(post)
    assert post.status == PostStatus.queued
    row = await h.row(post)
    assert row.state == PUB_FAILED and row.channel_id == CHANNEL_A and row.style == "repost"


# --- never twice, even when a send times out after delivery (§16) --------------------------------


def deliver_then_fail(gateway: Any, method: str, exc: Exception) -> None:
    """The next ``method`` call reaches Telegram and then raises ``exc`` (a lost answer)."""
    original = getattr(gateway, method)

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        setattr(gateway, method, original)
        await original(*args, **kwargs)
        raise exc

    setattr(gateway, method, wrapper)


async def test_forward_delivered_then_timeout_is_not_forwarded_again(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    deliver_then_fail(h.user, "forward", TelegramUnavailable("timeout after delivery"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_FAILED
    h.clock.advance(60)
    await h.pub.tick()
    forwards = [m for m in h.sent() if m.fwd_of]
    assert len(forwards) == 1
    assert len(h.user.calls_of("forward")) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids[0] == forwards[0].message_id
    assert len(row.message_ids) == 2 and wrong_topic_button(h.sent()[-1])


async def test_forward_retry_waits_while_the_check_cannot_run(h: Harness) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    deliver_then_fail(h.user, "forward", TelegramUnavailable("timeout after delivery"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    h.user.fail_next("find_message", TelegramUnavailable("still flapping"))
    h.clock.advance(60)
    await h.pub.tick()
    row = await h.row(post)
    assert row.state == PUB_FAILED and row.last_error.startswith("DeliveryUnverifiable")
    assert len(h.user.calls_of("forward")) == 1 and len(h.sent()) == 1
    h.clock.advance(120)
    await h.pub.tick()
    row = await h.row(post)
    assert row.state == PUB_SENT and len(h.user.calls_of("forward")) == 1
    assert len([m for m in h.sent() if m.fwd_of]) == 1


async def test_text_retry_with_a_failed_check_never_resends(h: Harness) -> None:
    deliver_then_fail(h.bot, "send_text", TelegramUnavailable("timeout after delivery"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    for _ in range(3):  # every retry's check fails: the row waits, it never sends blindly
        h.user.fail_next("find_message", TelegramUnavailable("flapping"))
        h.clock.advance(BACKOFF_CAP)
        await h.pub.tick()
    assert len(h.sent()) == 1
    row = await h.row(post)
    assert row.state == PUB_FAILED and row.last_error.startswith("DeliveryUnverifiable")
    assert (await h.fresh(post)).status == PostStatus.queued
    h.user.fail_next("find_message", TelegramUnavailable("flapping"))
    h.clock.set(row.created_at + timedelta(hours=6, minutes=1))
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_CANCELLED
    assert (await h.fresh(post)).status == PostStatus.digest
    assert len(h.sent()) == 1


async def test_text_retry_without_an_account_waits(h: Harness) -> None:
    deliver_then_fail(h.bot, "send_text", TelegramUnavailable("timeout after delivery"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    h.rt.user = None
    h.clock.advance(BACKOFF_CAP)
    await h.pub.tick()
    assert len(h.sent()) == 1 and (await h.row(post)).state == PUB_FAILED


async def test_text_retry_after_a_found_delivery_is_marked_sent(h: Harness) -> None:
    deliver_then_fail(h.bot, "send_text", TelegramUnavailable("timeout after delivery"))
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    h.clock.advance(BACKOFF_CAP)
    await h.pub.tick()
    assert len(h.sent()) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [h.sent()[0].message_id]


async def test_retry_after_a_refused_send_needs_no_check(h: Harness) -> None:
    h.bot.fail_next("send_text", FloodWait(300))  # Telegram refused: nothing went out
    post = await h.post()
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    h.rt.user = None  # no account to check with, and no check is needed
    h.clock.advance(300)
    await h.pub.tick()
    assert len(h.sent()) == 1 and (await h.row(post)).state == PUB_SENT


async def test_media_first_delivered_then_timeout_is_not_resent(h: Harness) -> None:
    deliver_then_fail(h.bot, "send_copy", TelegramUnavailable("timeout after delivery"))
    post = await h.post(LONG_TEXT, media="photo")
    await h.pub.enqueue(post.id)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.row(post)).state == PUB_FAILED
    h.clock.advance(BACKOFF_CAP)
    await h.pub.tick()
    media = [m for m in h.sent() if m.media]
    assert len(media) == 1 and len(h.sent()) == 2
    assert len(h.bot.calls_of("send_copy")) == 1
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == [m.message_id for m in h.sent()]
    check = h.user.calls_of("find_media")[0]
    assert check["copies_of"] == (STAGING, row.staging_ids) and check["after_id"] is None


async def test_album_crash_before_the_commit_is_reconciled_once(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="album", message_ids=[40, 41], message_id=40)
    post = await h.publish(post)
    m1, m2, text = h.sent()
    row = await h.row(post)
    # the crash came between Telegram's answer to the album and the commit of its ids
    h.world.messages[CHANNEL_A] = [m1, m2]
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, message_ids=[], sent_at=None)
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued, published_at=None)
    restarted = Publisher(h.rt)
    await restarted.reconcile()
    row = await h.row(post)
    assert row.state == PUB_PENDING and row.message_ids == [m1.message_id, m2.message_id]
    h.clock.advance(30)
    await restarted.tick()
    msgs = h.sent()
    assert [m.media for m in msgs] == ["album", "album", None]
    assert len(h.bot.calls_of("send_copy")) == 1
    assert (await h.row(post)).state == PUB_SENT


async def test_album_and_text_found_after_a_crash_is_sent(h: Harness) -> None:
    post = await h.post(LONG_TEXT, media="album", message_ids=[40, 41], message_id=40)
    post = await h.publish(post)
    ids = [m.message_id for m in h.sent()]
    row = await h.row(post)
    await h.rt.store.set_publication_fields(row.id, state=PUB_SENDING, message_ids=[], sent_at=None)
    await h.rt.store.set_post_fields(post.id, status=PostStatus.queued, published_at=None)
    await Publisher(h.rt).reconcile()
    row = await h.row(post)
    assert row.state == PUB_SENT and row.message_ids == ids


# --- group threads, flood waits on the media copy -------------------------------------------------


async def test_a_thread_is_one_repost_even_in_forward_style(h: Harness, make_chat: Any) -> None:
    await h.rt.settings_file.set_value("publishing.style", "forward")
    group = make_chat(id=-1_001_000_000_777, title="Dev chat", username="devchat", kind="group")
    h.user.add_chat(group)
    await h.rt.store.upsert_chat(group)
    for i, who in zip([100, 101, 102], ["alice", "bob", "carol"], strict=True):
        h.world.add(group.id, sender="source", message_id=i, text=f"{who} says something")
    text = "alice says something\nbob says something\ncarol says something"
    new = NewPost(
        chat_id=group.id,
        message_id=100,
        kind="unit",
        message_ids=[100, 101, 102],
        posted_at=h.clock.now(),
        via="live",
        text=text,
        html=None,
        text_hash=text_hash(text),
        urls=[],
        media=None,
        noforwards=False,
    )
    post = await h.rt.store.insert_post(new, status=PostStatus.held, topic_id=h.topic_a.id)
    assert post is not None
    post = await h.publish(post)
    assert h.user.calls_of("forward") == []
    (msg,) = h.sent()
    assert msg.sender == "bot" and hrefs(msg.html) == ["https://t.me/devchat/100"]
    assert "carol says something" in msg.text and wrong_topic_button(msg)
    row = await h.row(post)
    assert row.style == "repost" and row.message_ids == [msg.message_id]


async def test_flood_wait_on_the_media_copy_holds_the_post_back(h: Harness) -> None:
    h.user.fail_next("copy_media", FloodWait(600))
    first = await h.post("first story", media="photo")
    await h.pub.enqueue(first.id)
    h.clock.advance(5)
    await h.pub.tick()
    assert (await h.fresh(first)).status == PostStatus.queued
    row = await h.row(first)
    assert row.state == PUB_FAILED and row.staging_ids == []
    assert row.next_attempt_at == h.clock.now() + timedelta(seconds=600)
    assert h.sent() == []
    second = await h.post("second story", media="photo")
    await h.pub.enqueue(second.id)
    h.clock.advance(60)
    await h.pub.tick()
    assert len(h.user.calls_of("copy_media")) == 1  # no copy while the wait lasts
    assert h.sent() == []
    h.clock.advance(600)
    for _ in range(5):
        await h.pub.tick()
        h.clock.advance(60)
    msgs = h.sent()
    assert len(msgs) == 2 and all(m.media == "photo" for m in msgs)
    assert all("at source" not in m.text for m in msgs)
    assert (await h.fresh(first)).status == PostStatus.published
    assert (await h.fresh(second)).status == PostStatus.published


# --- +N always fits what was sent -----------------------------------------------------------------


async def test_plus_n_fits_a_caption_near_the_limit(h: Harness) -> None:
    body = "x" * (render.CAPTION_LIMIT - 40)
    post = await h.post(body, media="photo")
    post = await h.publish(post)
    await h.rt.store.add_corroboration(post.id, -5)
    await h.rt.store.add_corroboration(post.id, -6)
    h.clock.advance(120)
    await h.pub.tick()
    last = h.sent()[-1]
    assert "+2 more" in last.text and wrong_topic_button(last)
    captions = [m for m in h.sent() if m.media]
    assert all(render.plain_units(m.html or "") <= render.CAPTION_LIMIT for m in captions)


async def test_plus_n_fits_a_text_post_at_the_limit(h: Harness) -> None:
    header_units = len("Kun.uz · source")
    post = await h.post("y" * (render.TEXT_LIMIT - header_units - 1))  # fits without +N
    post = await h.publish(post)
    await h.rt.store.add_corroboration(post.id, -5)
    await h.rt.store.add_corroboration(post.id, -6)
    h.clock.advance(120)
    await h.pub.tick()
    last = h.sent()[-1]
    assert "+2 more" in last.text and wrong_topic_button(last)
    assert all(render.plain_units(m.html or "") <= render.TEXT_LIMIT for m in h.sent())
    assert (await h.row(post)).shown_corroboration == 2
