"""The fakes behave as the contract says: guards, failure injection, ids, find_message."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.fakes import (
    OWNER_ID,
    START,
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeUserGateway,
    make_bot_message,
    make_callback,
)
from tg_curator.contracts import LLM, Embedder, TopicClassifier
from tg_curator.domain import Example, Topic, TopicScore
from tg_curator.errors import (
    BotCannotPost,
    ChatGone,
    FloodWait,
    FolderLimit,
    ForwardsRestricted,
    LoginError,
    MediaUnavailable,
    NotAllowed,
    NotOwnedError,
    SessionLost,
)
from tg_curator.telegram.gateway import (
    BotGateway,
    Button,
    ChatInfo,
    IncomingMessage,
    UserGateway,
)

OUT = -1_001_900_000_001  # an output channel


# --- protocols -------------------------------------------------------------------------------


def test_fakes_satisfy_the_protocols(user_gw: FakeUserGateway, bot_gw: FakeBotGateway) -> None:
    assert isinstance(user_gw, UserGateway)
    assert isinstance(bot_gw, BotGateway)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeClassifier(), TopicClassifier)
    assert isinstance(FakeLLM(), LLM)


# --- clock -----------------------------------------------------------------------------------


def test_fake_clock_moves_only_on_advance() -> None:
    clock = FakeClock(START)
    assert clock.now() == START
    assert clock.advance(90) == START + timedelta(seconds=90)
    assert clock.advance(timedelta(hours=1)) == START + timedelta(seconds=90, hours=1)
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 1, 1))


# --- content-write guards --------------------------------------------------------------------


async def test_content_writes_need_an_owned_target(user_gw: FakeUserGateway) -> None:
    with pytest.raises(NotOwnedError):
        await user_gw.copy_media(-100_1, [1], OUT)
    with pytest.raises(NotOwnedError):
        await user_gw.forward(-100_1, [1], OUT)
    with pytest.raises(NotOwnedError):
        await user_gw.rename_channel(OUT, "x")
    with pytest.raises(NotOwnedError):
        await user_gw.add_bot_admin(OUT, "bot")
    with pytest.raises(NotOwnedError):
        await user_gw.find_message(OUT, contains="x")
    user_gw.register_owned(OUT)
    assert await user_gw.copy_media(-100_1, [1], OUT) == [1]
    assert ("register_owned", {"chat_id": OUT}) in user_gw.calls


async def test_created_channel_is_not_owned_until_registered(user_gw: FakeUserGateway) -> None:
    chat = await user_gw.create_channel("ML & AI")
    assert chat.is_creator and chat.is_admin and chat.kind == "channel"
    assert chat in user_gw.chats and user_gw.created == [chat]
    with pytest.raises(NotOwnedError):
        await user_gw.rename_channel(chat.id, "renamed")
    user_gw.register_owned(chat.id)
    await user_gw.rename_channel(chat.id, "renamed")
    assert user_gw.chat(chat.id).title == "renamed"


# --- subscription-write guards ---------------------------------------------------------------


async def test_subscription_writes_refuse_owned_chats(
    user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    user_gw.add_chat(make_chat(OUT, title="Output"))
    user_gw.register_owned(OUT)
    with pytest.raises(NotOwnedError):
        await user_gw.mute(OUT, None)
    with pytest.raises(NotOwnedError):
        await user_gw.set_archived(OUT, True)
    with pytest.raises(NotOwnedError):
        await user_gw.leave(OUT)


async def test_leave_rules(user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]) -> None:
    mine = user_gw.add_chat(make_chat(is_creator=True))
    other = user_gw.add_chat(make_chat())
    with pytest.raises(NotAllowed) as exc:
        await user_gw.leave(mine.id)
    assert exc.value.reason == "creator"
    with pytest.raises(ChatGone):
        await user_gw.leave(-100_42)
    await user_gw.leave(other.id)
    assert user_gw.left == [other.id] and user_gw.chat(other.id) is None
    assert other not in await user_gw.list_chats()


async def test_mute_and_archive_update_the_dialog(
    user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo], clock: FakeClock
) -> None:
    chat = user_gw.add_chat(make_chat())
    until = clock.now() + timedelta(days=30)
    await user_gw.mute(chat.id, until)
    await user_gw.set_archived(chat.id, True)
    assert user_gw.chat(chat.id).muted_until == until and user_gw.chat(chat.id).archived
    assert user_gw.mutes[chat.id] == until and user_gw.archived[chat.id] is True
    await user_gw.mute(chat.id, None)
    assert user_gw.chat(chat.id).muted_until is None


# --- failure injection and recording ---------------------------------------------------------


async def test_fail_next_raises_once_and_records_the_call(bot_gw: FakeBotGateway) -> None:
    bot_gw.fail_next("send_text", FloodWait(300))
    bot_gw.fail_next("send_text", BotCannotPost("no"))
    with pytest.raises(FloodWait) as exc:
        await bot_gw.send_text(OUT, "a")
    assert exc.value.seconds == 300
    with pytest.raises(BotCannotPost):
        await bot_gw.send_text(OUT, "a")
    assert await bot_gw.send_text(OUT, "b") == 1
    assert [name for name, _ in bot_gw.calls] == ["send_text"] * 3
    assert bot_gw.calls_of("send_text")[0]["html"] == "a"
    assert bot_gw.calls[-1] == (
        "send_text",
        {"chat_id": OUT, "html": "b", "buttons": None, "reply_to": None, "silent": False},
    )


async def test_fail_next_on_history_raises_while_iterating(user_gw: FakeUserGateway) -> None:
    user_gw.fail_next("history", SessionLost("revoked"))
    stream = user_gw.history(-100_1, since=START - timedelta(days=1))
    with pytest.raises(SessionLost):
        async for _ in stream:
            pass
    assert user_gw.calls[-1][0] == "history"


async def test_every_user_write_can_fail(user_gw: FakeUserGateway) -> None:
    user_gw.register_owned(OUT)
    user_gw.fail_next("copy_media", MediaUnavailable("gone"))
    with pytest.raises(MediaUnavailable):
        await user_gw.copy_media(-100_1, [1], OUT)
    user_gw.fail_next("create_channel", FloodWait(900))
    with pytest.raises(FloodWait):
        await user_gw.create_channel("x")
    user_gw.fail_next("ping", SessionLost("expired"))
    with pytest.raises(SessionLost):
        await user_gw.ping()


# --- ids and channel contents ----------------------------------------------------------------


async def test_message_ids_increase_per_chat(
    bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    user_gw.register_owned(OUT)
    assert await bot_gw.send_text(OUT, "one") == 1
    assert await bot_gw.send_text(OUT, "two") == 2
    assert await bot_gw.send_text(OWNER_ID, "private") == 1  # another chat, its own counter
    assert await user_gw.copy_media(-100_1, [10, 11, 12], OUT) == [3, 4, 5]
    assert await user_gw.forward(-100_1, [10, 11], OUT) == [6, 7]
    sent = bot_gw.sent(OUT)
    assert [m.message_id for m in sent] == [1, 2, 3, 4, 5, 6, 7]
    assert [m.sender for m in sent] == ["bot", "bot", "user", "user", "user", "user", "user"]
    assert sent[2].copied_from == (-100_1, 10) and sent[5].fwd_of == (-100_1, 10)


async def test_copy_and_forward_carry_the_source_media_and_text(
    user_gw: FakeUserGateway, make_message: Callable[..., IncomingMessage]
) -> None:
    msg = make_message(text="photo post", media="photo")
    user_gw.seed(msg)
    user_gw.register_owned(OUT)
    [copied] = await user_gw.copy_media(msg.chat.id, [msg.message_id], OUT)
    [forwarded] = await user_gw.forward(msg.chat.id, [msg.message_id], OUT)
    copy, fwd = user_gw.world.get(OUT, copied), user_gw.world.get(OUT, forwarded)
    assert copy.media == "photo" and copy.text == "" and copy.fwd_of is None
    assert fwd.media == "photo" and fwd.text == "photo post"
    assert fwd.fwd_of == (msg.chat.id, msg.message_id)


async def test_noforwards_sources_raise_forwards_restricted(
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> None:
    chat = user_gw.add_chat(make_chat(noforwards=True))
    msg = make_message(chat)
    user_gw.seed(msg)
    user_gw.register_owned(OUT)
    with pytest.raises(ForwardsRestricted):
        await user_gw.copy_media(chat.id, [msg.message_id], OUT)
    with pytest.raises(ForwardsRestricted):
        await user_gw.forward(chat.id, [msg.message_id], OUT)


# --- find_message ----------------------------------------------------------------------------


async def test_find_message_matches_text_href_and_forward(
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    user_gw.register_owned(OUT)
    link = "https://t.me/kunuz/123"
    await bot_gw.send_text(OUT, f'<b>Kun.uz</b> · <a href="{link}">source</a>\nText one')
    clock.advance(60)
    since = clock.now()
    clock.advance(60)
    await bot_gw.send_text(OUT, "Second message with &amp; entity")
    [fwd_id] = await user_gw.forward(-100_7, [55], OUT)

    assert await user_gw.find_message(OUT, contains=link) == 1
    assert await user_gw.find_message(OUT, contains="Text one") == 1
    assert await user_gw.find_message(OUT, contains="source") == 1  # plain text, not the tag
    assert await user_gw.find_message(OUT, contains="with & entity") == 2
    assert await user_gw.find_message(OUT, contains="<b>") is None  # tags are not text
    assert await user_gw.find_message(OUT, fwd_of=(-100_7, 55)) == fwd_id
    assert await user_gw.find_message(OUT, fwd_of=(-100_7, 56)) is None
    assert await user_gw.find_message(OUT, contains="Text one", since=since) is None
    assert await user_gw.find_message(OUT, contains="entity", since=since) == 2
    assert await user_gw.find_message(OUT) == fwd_id  # newest of all
    assert await user_gw.find_message(OUT, contains="Text one", limit=1) is None


# --- reading ---------------------------------------------------------------------------------


async def test_deliver_reaches_handlers_and_seeds_history(
    user_gw: FakeUserGateway, make_message: Callable[..., IncomingMessage], clock: FakeClock
) -> None:
    seen: list[int] = []

    async def handler(msg: IncomingMessage) -> None:
        seen.append(msg.message_id)

    user_gw.on_message(handler)
    first = make_message(text="first")
    clock.advance(10)
    second = make_message(first.chat, text="second")
    await user_gw.deliver(second)
    await user_gw.deliver(first)
    assert seen == [2, 1]
    got = [m async for m in user_gw.history(first.chat.id, since=START - timedelta(days=1))]
    assert [m.message_id for m in got] == [1, 2]  # oldest first whatever the delivery order
    got = [m async for m in user_gw.history(first.chat.id, since=first.date)]
    assert [m.message_id for m in got] == [2]  # strictly after `since`
    got = [
        m async for m in user_gw.history(first.chat.id, since=START - timedelta(days=1), limit=1)
    ]
    assert [m.message_id for m in got] == [1]


async def test_resolve_chat_never_joins(
    user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    public = user_gw.add_chat(make_chat(username="kunuz"))
    private = user_gw.add_chat(make_chat())
    user_gw.invites["https://t.me/+abc"] = private.id
    assert await user_gw.resolve_chat(public.id) == public
    assert await user_gw.resolve_chat(str(public.id)) == public
    assert await user_gw.resolve_chat("@KunUz") == public
    assert await user_gw.resolve_chat("https://t.me/kunuz") == public
    assert await user_gw.resolve_chat("https://t.me/+abc") == private
    with pytest.raises(NotAllowed) as exc:
        await user_gw.resolve_chat("https://t.me/+unknown")
    assert exc.value.reason == "not_a_member"
    with pytest.raises(ChatGone):
        await user_gw.resolve_chat("@nobody")
    with pytest.raises(ChatGone):
        await user_gw.resolve_chat(-100_99)


async def test_get_views_only_for_channels(
    user_gw: FakeUserGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    channel = user_gw.add_chat(make_chat())
    group = user_gw.add_chat(make_chat(kind="group"))
    user_gw.views[(channel.id, 1)] = 120
    user_gw.views[(group.id, 1)] = 5
    assert await user_gw.get_views(channel.id, [1, 2]) == {1: 120}
    assert await user_gw.get_views(group.id, [1]) == {}


# --- folders ---------------------------------------------------------------------------------


async def test_folders(user_gw: FakeUserGateway) -> None:
    user_gw.folders[1] = ("Work", [-100_5])  # a folder the user made
    fid = await user_gw.save_folder(None, "Curated", [OUT])
    assert fid == 2 and await user_gw.get_folder(fid) == [OUT]
    assert await user_gw.list_folders() == [(1, "Work"), (2, "Curated")]
    assert await user_gw.save_folder(fid, "Curated", [OUT, -100_9]) == fid
    with pytest.raises(NotOwnedError):
        await user_gw.save_folder(1, "Work", [-100_5])  # not ours
    with pytest.raises(ValueError):
        await user_gw.save_folder(fid, "Curated", [])
    with pytest.raises(ValueError):
        await user_gw.save_folder(fid, "A title that is too long", [OUT])
    assert await user_gw.get_folder(7) is None
    user_gw.folder_limit = 2
    with pytest.raises(FolderLimit):
        await user_gw.save_folder(None, "Low signal", [-100_5])
    user_gw.own_folder(9, "Low signal", [-100_5])
    assert await user_gw.save_folder(9, "Low signal", [-100_6]) == 9


# --- login and session -----------------------------------------------------------------------


async def test_login_flow(user_gw: FakeUserGateway) -> None:
    user_gw.authorised = False
    user_gw.password_needed = True
    user_gw.valid_code = "12345"
    assert await user_gw.connect() is False
    assert await user_gw.me() is None
    with pytest.raises(LoginError) as exc:
        await user_gw.sign_in("1")
    assert exc.value.reason == "other"  # no code requested yet
    await user_gw.send_code("+998901234567")
    await user_gw.resend_code()
    assert user_gw.codes_sent == ["+998901234567"] * 2
    with pytest.raises(LoginError) as exc:
        await user_gw.sign_in("99999")
    assert exc.value.reason == "bad_code"
    assert await user_gw.sign_in("12345") == "password_needed"
    assert await user_gw.me() is None
    await user_gw.sign_in_password("secret")
    assert (await user_gw.me()).id == OWNER_ID
    assert await user_gw.ping() is True


async def test_lose_session_tells_handlers_once(user_gw: FakeUserGateway) -> None:
    reasons: list[str] = []

    async def handler(reason: str) -> None:
        reasons.append(reason)

    user_gw.on_session_lost(handler)
    await user_gw.lose_session("duplicated")
    assert reasons == ["duplicated"] and user_gw.authorised is False
    assert await user_gw.ping() is False


# --- the bot ---------------------------------------------------------------------------------


async def test_bot_say_and_press_reach_handlers(bot_gw: FakeBotGateway) -> None:
    texts: list[str] = []
    datas: list[str] = []

    async def on_msg(msg) -> None:  # noqa: ANN001 - the handler shape is the protocol's
        texts.append(msg.text)

    async def on_cb(cb) -> None:  # noqa: ANN001
        datas.append(cb.data)
        await bot_gw.answer_callback(cb.query_id, "ok", alert=True)

    bot_gw.on_message(on_msg)
    bot_gw.on_callback(on_cb)
    assert (await bot_gw.start()).username == "curator_test_bot"
    await bot_gw.say(make_bot_message("/start 1234"))
    await bot_gw.press(make_callback("wt:42", chat_id=OUT, message_id=7))
    assert texts == ["/start 1234"] and datas == ["wt:42"]
    assert bot_gw.answers == [("q1", "ok", True)]
    msg = make_bot_message("hi")
    assert msg.chat_id == OWNER_ID and msg.sender_id == OWNER_ID and msg.is_private


async def test_bot_cannot_post_until_made_admin(
    bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    bot_gw.cannot_post.add(OUT)
    assert await bot_gw.can_post(OUT) is False
    with pytest.raises(BotCannotPost):
        await bot_gw.send_text(OUT, "x")
    user_gw.register_owned(OUT)
    await user_gw.add_bot_admin(OUT, "curator_test_bot")
    assert await bot_gw.can_post(OUT) is True
    assert await bot_gw.send_text(OUT, "x") == 1


async def test_bot_send_copy_rules(bot_gw: FakeBotGateway, user_gw: FakeUserGateway) -> None:
    staging = -1_001_900_000_002
    user_gw.register_owned(staging)
    media_ids = await user_gw.copy_media(-100_1, [1, 2], staging)
    with pytest.raises(ValueError):
        await bot_gw.send_copy(
            staging, media_ids, OUT, buttons=[[Button("Wrong topic", data="wt:1")]]
        )
    ids = await bot_gw.send_copy(staging, media_ids, OUT, caption_html="ignored for albums")
    assert len(ids) == 2
    assert all(m.html is None and m.buttons is None for m in bot_gw.sent(OUT))
    [single] = await bot_gw.send_copy(
        staging, media_ids[:1], OUT, caption_html="<b>cap</b>", buttons=[[Button("A", data="a")]]
    )
    sent = bot_gw.world.get(OUT, single)
    assert sent.text == "cap" and sent.buttons == [[Button("A", data="a")]]
    assert sent.copied_from == (staging, media_ids[0])
    with pytest.raises(MediaUnavailable):
        await bot_gw.send_copy(staging, [999], OUT)


async def test_bot_edits_only_its_own_messages(
    bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    user_gw.register_owned(OUT)
    mid = await bot_gw.send_text(OUT, "<b>A</b> · +1 more", buttons=[[Button("W", data="wt:1")]])
    await bot_gw.edit_text(OUT, mid, "<b>A</b> · +1 more", buttons=[[Button("W", data="wt:1")]])
    assert bot_gw.world.get(OUT, mid).edits == 0  # MESSAGE_NOT_MODIFIED swallowed
    await bot_gw.edit_text(OUT, mid, "<b>A</b> · +2 more", buttons=[[Button("W", data="wt:1")]])
    await bot_gw.edit_buttons(OUT, mid, None)
    msg = bot_gw.world.get(OUT, mid)
    assert msg.text == "A · +2 more" and msg.buttons is None and msg.edits == 2
    [fwd] = await user_gw.forward(-100_1, [1], OUT)
    with pytest.raises(NotAllowed):
        await bot_gw.edit_text(OUT, fwd, "stub")
    with pytest.raises(ValueError):
        await bot_gw.edit_text(OUT, 99, "stub")


async def test_bot_deletes_only_in_the_private_chat(bot_gw: FakeBotGateway) -> None:
    mid = await bot_gw.send_text(OWNER_ID, "code 1 2 3")
    await bot_gw.delete_message(OWNER_ID, mid)
    assert bot_gw.sent(OWNER_ID) == [] and bot_gw.deleted == [(OWNER_ID, mid)]
    channel_mid = await bot_gw.send_text(OUT, "post")
    with pytest.raises(NotAllowed):
        await bot_gw.delete_message(OUT, channel_mid)


# --- embedder --------------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def test_fake_embedder_is_deterministic_and_unit_norm() -> None:
    emb = FakeEmbedder()
    assert emb.id == "fake-embedder-64" and emb.dim == 64
    base = "OpenAI releases a new model for code generation with longer context windows today"
    near = "OpenAI releases a new model for code generation with longer context windows now"
    other = "Футбол: Пахтакор обыграл Навбахор со счётом два ноль в чемпионате Узбекистана"
    vecs = emb.embed([base, base, near, other, ""])
    assert vecs.shape == (5, 64) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(vecs[0], vecs[1])
    assert _cos(vecs[0], vecs[2]) > 0.9
    assert _cos(vecs[0], vecs[3]) < 0.5
    assert np.array_equal(emb.embed([base])[0], vecs[0])  # same across calls


# --- classifier ------------------------------------------------------------------------------


def _topic(tid: int, key: str, active: bool = True) -> Topic:
    return Topic(id=tid, key=key, name=key.upper(), active=active, created_at=START)


def test_fake_classifier_scores() -> None:
    clf = FakeClassifier({1: 0.9, 2: 0.3})
    clf.reload(
        [_topic(1, "a"), _topic(2, "b"), _topic(3, "c"), _topic(4, "gone", active=False)], []
    )
    scores = clf.predict("anything", np.zeros(64))
    assert scores == [TopicScore(1, 0.9), TopicScore(2, 0.3), TopicScore(3, 0.1)]
    clf.text_scores["special"] = {3: 0.8}
    assert clf.predict("special", np.zeros(64))[0] == TopicScore(3, 0.8)
    cats = clf.category_scores("x", np.zeros(64))
    assert len(cats) == 20 and all(s == 0.1 for _, s in cats)
    clf.categories = [("sport", 0.2), ("politics", 0.7)]
    assert clf.category_scores("x", np.zeros(64))[0] == ("politics", 0.7)
    example = Example(
        id=1, topic_id=2, kind="correction", text="t", embedding=b"", created_at=START
    )
    clf.learn(example)
    assert clf.learned == [example] and clf.reloads == 1


def test_fake_classifier_without_reload_uses_the_score_keys() -> None:
    clf = FakeClassifier({5: 0.6})
    assert clf.predict("x", np.zeros(64)) == [TopicScore(5, 0.6)]
    assert FakeClassifier().predict("x", np.zeros(64)) == []


# --- llm -------------------------------------------------------------------------------------


async def test_fake_llm() -> None:
    llm = FakeLLM()
    assert llm.enabled
    assert await llm.summarise_line("a long post", max_chars=6) == "Canned"
    llm.summaries["a long post"] = "Specific line."
    assert await llm.summarise_line("a long post", max_chars=100) == "Specific line."
    assert (await llm.name_topic(["x"])).name == "Canned topic"
    assert await llm.second_opinion("x", _topic(1, "a")) is True
    assert [name for name, _ in llm.calls] == ["summarise_line"] * 2 + [
        "name_topic",
        "second_opinion",
    ]
    off = FakeLLM(enabled=False)
    assert await off.summarise_line("x", 10) is None
    assert (
        await off.name_topic(["x"]) is None
        and await off.second_opinion("x", _topic(1, "a")) is None
    )
    llm.fail = True
    assert await llm.summarise_line("x", 10) is None
