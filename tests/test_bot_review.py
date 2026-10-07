"""bot/review.py: /review, the rv: and ds: buttons, the Rename conversation, /undo, the owner guard.

Proposals are seeded through the real ReviewService / Discovery / ActionExecutor /
TopicsService on the fake clock; only the clustering maths is replaced (as in
test_discovery.py) so a new-topic proposal forms without the local models.
"""

from __future__ import annotations

import inspect
import itertools
import re
import sys
import tomllib
from collections.abc import Callable, Sequence
from datetime import timedelta
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeEmbedder,
    FakeUserGateway,
    make_bot_message,
    make_callback,
)
from tg_curator.bot import review as bot_review
from tg_curator.bot.core import BotApp
from tg_curator.domain import (
    PROPOSAL_APPROVED,
    PROPOSAL_CONFIRMING,
    PROPOSAL_DONE,
    PROPOSAL_NEVER,
    PROPOSAL_PROPOSED,
    PROPOSAL_SKIPPED,
    PROPOSAL_UNDONE,
    Category,
    Cluster,
    NewPost,
    Post,
    PostStatus,
    Proposal,
    TopicPair,
)
from tg_curator.errors import FloodWait
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.notify import Notifier
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.actions import ActionExecutor
from tg_curator.subscriptions.discovery import Discovery
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.subscriptions.review import ReviewService
from tg_curator.subscriptions.stats import StatsService
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash
from tg_curator.topics.service import TopicsService

STRANGER = 4242
EMBEDDER = FakeEmbedder()
_IDS = itertools.count(5000)


class FakeCluster:
    """``ml/cluster.py`` stand-in: the first ``min_size`` rows form one cluster; ``competition``
    returns ``pairs``."""

    def __init__(self) -> None:
        self.pairs: list[TopicPair] = []

    def find_clusters(
        self, embeddings: np.ndarray, min_size: int, tightness: float, **_: Any
    ) -> list[Cluster]:
        if len(embeddings) < min_size:
            return []
        ids = list(range(min_size))
        return [Cluster(member_ids=ids, centroid=embeddings[ids].mean(axis=0), tightness=0.9)]

    def competition(self, posts: Sequence[Post], margin: float, **_: Any) -> list[TopicPair]:
        return list(self.pairs)


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    fake = FakeCluster()
    module = ModuleType("tg_curator.ml.cluster")
    module.find_clusters = fake.find_clusters  # type: ignore[attr-defined]
    module.competition = fake.competition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tg_curator.ml.cluster", module)
    categories = ModuleType("tg_curator.ml.categories")
    categories.all = lambda: [Category("crypto", "Crypto & exchanges")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tg_curator.ml.categories", categories)
    return fake


@pytest.fixture
def app(rt: Runtime, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch) -> BotApp:
    """The real subscription and topic services, the bot app with this module registered."""
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    rt.stats = StatsService(rt)
    rt.folders = FolderManager(rt)
    rt.actions = ActionExecutor(rt)
    rt.discovery = Discovery(rt)
    rt.review = ReviewService(rt)
    rt.topics = TopicsService(rt)
    rt.classifier.categories = [("crypto", 0.8)]  # type: ignore[attr-defined]
    bot_app = BotApp(rt)
    bot_review.register(bot_app)
    return bot_app


# --- seeding -----------------------------------------------------------------------------------


async def add_post(rt: Runtime, chat_id: int, mid: int, **decision: Any) -> Post:
    text = f"post {chat_id} {mid}"
    new = NewPost(
        chat_id=chat_id, message_id=mid, kind="post", message_ids=[mid],
        posted_at=rt.clock.now(), via="live", text=text, text_hash=text_hash(text), urls=[],
    )  # fmt: skip
    post = await rt.store.insert_post(new, **decision)
    assert post is not None
    return post


async def low_signal_chat(
    rt: Runtime, info: ChatInfo, *, days: int, repeated: ChatInfo | None = None
) -> ChatInfo:
    """A source chat observed ``days`` days: 41 messages, nothing sorted, and — when
    ``repeated`` is given — 35 repeats of that chat (what makes a mute proposal)."""
    store = rt.store
    now = rt.clock.now()
    rt.user.add_chat(info)  # type: ignore[union-attr]
    await store.upsert_chat(info)
    await store.set_chat_fields(info.id, first_seen_at=now - timedelta(days=days))
    await store.bump_chat_daily(info.id, now.date(), 41)
    if repeated is not None:
        if await store.get_chat(repeated.id) is None:
            await store.upsert_chat(repeated)
        for i in range(35):
            root = await add_post(rt, repeated.id, next(_IDS), status=PostStatus.published)
            await add_post(rt, info.id, 300 + i, status=PostStatus.duplicate, duplicate_of=root.id)
    return info


async def unsorted_posts(rt: Runtime, info: ChatInfo, n: int = 30) -> None:
    """``n`` unsorted posts with embeddings: what discovery clusters into a new topic."""
    await rt.store.upsert_chat(info)
    for i in range(1, n + 1):
        text = f"crypto exchange story number {i} about bitcoin and tether"
        new = NewPost(
            chat_id=info.id, message_id=i, kind="post", message_ids=[i],
            posted_at=rt.clock.now() - timedelta(hours=1), via="live", text=text,
            text_hash=text_hash(text), urls=[], embedding=EMBEDDER.embed([text])[0].tobytes(),
        )  # fmt: skip
        assert await rt.store.insert_post(new, status=PostStatus.unsorted) is not None


async def sent_review(rt: Runtime, bot_gw: FakeBotGateway) -> dict[str, Proposal]:
    """Run /review and return the proposals by kind, re-read after sending."""
    await bot_gw.say(make_bot_message("/review"))
    out: dict[str, Proposal] = {}
    for p in await rt.store.proposals_by_state(PROPOSAL_PROPOSED):
        assert p.bot_message_id is not None
        out[p.kind] = p
    return out


async def press(bot_gw: FakeBotGateway, data: str, *, sender_id: int = OWNER_ID) -> None:
    await bot_gw.press(make_callback(data, sender_id=sender_id))


async def state_of(rt: Runtime, proposal: Proposal) -> str:
    fresh = await rt.store.get_proposal(proposal.id)
    assert fresh is not None
    return fresh.state


def message(bot_gw: FakeBotGateway, message_id: int | None) -> Any:
    return next(m for m in bot_gw.sent(OWNER_ID) if m.message_id == message_id)


def button_data(bot_gw: FakeBotGateway, message_id: int | None) -> list[list[str | None]]:
    return [[b.data for b in row] for row in message(bot_gw, message_id).buttons or []]


def button_labels(bot_gw: FakeBotGateway, message_id: int | None) -> list[list[str]]:
    return [[b.text for b in row] for row in message(bot_gw, message_id).buttons or []]


def last_text(bot_gw: FakeBotGateway) -> str:
    return str(bot_gw.sent(OWNER_ID)[-1].text)


def toasts(bot_gw: FakeBotGateway) -> list[str | None]:
    return [text for _, text, _ in bot_gw.answers if text is not None]


# --- the catalogue -----------------------------------------------------------------------------


def test_every_key_used_ships_in_english_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(bot_review)
    used = set(re.findall(r'"(review_bot_[a-z_]+)"', source))
    used |= {f"review_bot_ack_{a}" for a in ("cancel", "skip", "never", "undo")}  # f-string
    with (LOCALES_DIR / "en" / "review_bot.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    english = Translator(locales_dir=LOCALES_DIR).english_keys()
    assert shipped <= english
    generic = set(re.findall(r'"([a-z_]+)"', source)) & {"unknown_choice", "cancel"}
    assert generic <= english


# --- /review -----------------------------------------------------------------------------------


async def test_review_builds_sends_and_replies_with_the_count(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    kunuz = make_chat(username="kunuz")
    await low_signal_chat(rt, make_chat(title="Fold me"), days=7, repeated=kunuz)
    await low_signal_chat(rt, make_chat(title="Mute me"), days=14, repeated=kunuz)
    by_kind = await sent_review(rt, bot_gw)
    assert set(by_kind) == {"folder", "mute"}
    sent = bot_gw.sent(OWNER_ID)
    assert [m.text.splitlines()[0] for m in sent[:2]] == [
        "📁 Move to the “Low signal” folder · Fold me",
        "🔕 Mute · Mute me",
    ]
    assert sent[-1].text == "2 proposals sent above. Nothing happens until you tap Approve on one."
    # a second /review finds nothing new to send
    await bot_gw.say(make_bot_message("/review"))
    assert last_text(bot_gw).startswith("Nothing to propose right now")


async def test_review_with_a_single_proposal_says_one(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await low_signal_chat(rt, make_chat(), days=7)
    await sent_review(rt, bot_gw)
    assert last_text(bot_gw) == "One proposal sent above. Nothing happens until you tap Approve."


# --- rv: decisions -----------------------------------------------------------------------------


async def test_approve_a_folder_move_edits_the_message_and_acknowledges(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await low_signal_chat(rt, make_chat(), days=7)
    p = (await sent_review(rt, bot_gw))["folder"]
    await press(bot_gw, f"rv:{p.id}:approve")
    assert await state_of(rt, p) == PROPOSAL_APPROVED
    edited = message(bot_gw, p.bot_message_id)
    assert edited.text.endswith("Approved ✓ — will be done within minutes.")
    assert edited.buttons is None
    assert toasts(bot_gw) == ["Approved. It will be done within minutes."]


async def test_approve_leave_asks_to_confirm_then_confirm_approves(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await low_signal_chat(rt, make_chat(title="Old group"), days=30)
    p = (await sent_review(rt, bot_gw))["leave"]
    await press(bot_gw, f"rv:{p.id}:approve")
    assert await state_of(rt, p) == PROPOSAL_CONFIRMING
    assert "Leave Old group? This cannot be undone" in message(bot_gw, p.bot_message_id).text
    assert button_labels(bot_gw, p.bot_message_id) == [["Yes, leave", "Cancel"]]
    assert button_data(bot_gw, p.bot_message_id) == [[f"rv:{p.id}:confirm", f"rv:{p.id}:cancel"]]
    assert toasts(bot_gw)[-1] == "Leaving cannot be undone: confirm below."

    await press(bot_gw, f"rv:{p.id}:confirm")
    assert await state_of(rt, p) == PROPOSAL_APPROVED
    assert message(bot_gw, p.bot_message_id).text.endswith(
        "leaving soon: leaves are paced at most 3 a day and 30 min apart so the account "
        "never looks automated."
    )
    assert "within the hour" not in message(bot_gw, p.bot_message_id).text
    assert message(bot_gw, p.bot_message_id).buttons is None
    assert toasts(bot_gw)[-1] == "Approved. Leaving soon (paced)."
    assert rt.user.left == []  # type: ignore[union-attr]  # the executor leaves, paced, later


async def test_cancel_on_the_leave_confirmation_restores_the_proposal(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await low_signal_chat(rt, make_chat(title="Old group"), days=30)
    p = (await sent_review(rt, bot_gw))["leave"]
    await press(bot_gw, f"rv:{p.id}:approve")
    await press(bot_gw, f"rv:{p.id}:cancel")
    assert await state_of(rt, p) == PROPOSAL_PROPOSED
    assert button_data(bot_gw, p.bot_message_id) == [
        [f"rv:{p.id}:approve", f"rv:{p.id}:skip"], [f"rv:{p.id}:never"],
    ]  # fmt: skip
    assert "Leave Old group?" not in message(bot_gw, p.bot_message_id).text
    assert toasts(bot_gw)[-1] == "Cancelled. Nothing was changed."


async def test_skip_and_never_acknowledge_and_never_keeps_the_chat(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    kunuz = make_chat(username="kunuz")
    await low_signal_chat(rt, make_chat(), days=7, repeated=kunuz)
    kept = await low_signal_chat(rt, make_chat(), days=14, repeated=kunuz)
    by_kind = await sent_review(rt, bot_gw)
    skip, never = by_kind["folder"], by_kind["mute"]

    await press(bot_gw, f"rv:{skip.id}:skip")
    assert await state_of(rt, skip) == PROPOSAL_SKIPPED
    assert "Skipped — it comes back" in message(bot_gw, skip.bot_message_id).text
    assert toasts(bot_gw)[-1] == "Skipped."

    await press(bot_gw, f"rv:{never.id}:never")
    assert await state_of(rt, never) == PROPOSAL_NEVER
    assert (await rt.store.get_chat(kept.id)).keep is True  # type: ignore[union-attr]
    assert "Never asked again" in message(bot_gw, never.bot_message_id).text
    assert toasts(bot_gw)[-1] == "This chat will not be proposed again."


async def test_a_stale_or_malformed_tap_changes_nothing_and_says_so(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await low_signal_chat(rt, make_chat(), days=7)
    p = (await sent_review(rt, bot_gw))["folder"]
    await press(bot_gw, f"rv:{p.id}:approve")
    sent_before = len(bot_gw.sent(OWNER_ID))
    await press(bot_gw, f"rv:{p.id}:approve")  # a double tap
    await press(bot_gw, f"rv:{p.id}:maybe")
    await press(bot_gw, "rv:x:approve")
    assert await state_of(rt, p) == PROPOSAL_APPROVED
    assert toasts(bot_gw)[-3:] == ["That choice is not available any more."] * 3
    # Each tap had exactly one answer, and it carried the alert (§11.1).
    assert bot_gw.answers[-3] == ("q1", "That choice is not available any more.", True)
    assert len(bot_gw.answers) == 4
    assert len(bot_gw.sent(OWNER_ID)) == sent_before  # no error message, only the toast


# --- undo --------------------------------------------------------------------------------------


async def done_proposal(
    rt: Runtime, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo], kind: str
) -> Proposal:
    """A proposal of ``kind`` approved through the bot and carried out by the executor."""
    days = {"folder": 7, "mute": 14, "leave": 30}[kind]
    repeated = make_chat(username="kunuz") if kind == "mute" else None
    await low_signal_chat(rt, make_chat(), days=days, repeated=repeated)
    p = (await sent_review(rt, bot_gw))[kind]
    await press(bot_gw, f"rv:{p.id}:approve")
    if kind == "leave":
        await press(bot_gw, f"rv:{p.id}:confirm")
    await rt.actions.tick()  # type: ignore[union-attr]
    assert await state_of(rt, p) == PROPOSAL_DONE
    return p


async def test_undo_button_reverses_a_done_mute(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    p = await done_proposal(rt, bot_gw, make_chat, "mute")
    assert button_data(bot_gw, p.bot_message_id) == [[f"rv:{p.id}:undo"]]
    assert user_gw.mutes[p.chat_id] is not None  # type: ignore[index]
    await press(bot_gw, f"rv:{p.id}:undo")
    assert await state_of(rt, p) == PROPOSAL_UNDONE
    assert user_gw.mutes[p.chat_id] is None  # type: ignore[index]
    assert message(bot_gw, p.bot_message_id).text.endswith("Undone ✓")
    assert toasts(bot_gw)[-1] == "Undone."


async def test_undo_command_in_reply_to_the_proposal_message(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await done_proposal(rt, bot_gw, make_chat, "folder")
    assert (await rt.store.get_chat(p.chat_id)).in_low_signal is True  # type: ignore[arg-type,union-attr]
    await bot_gw.say(make_bot_message("/undo", message_id=90, reply_to_id=p.bot_message_id))
    assert await state_of(rt, p) == PROPOSAL_UNDONE
    assert (await rt.store.get_chat(p.chat_id)).in_low_signal is False  # type: ignore[arg-type,union-attr]
    assert message(bot_gw, p.bot_message_id).text.endswith("Undone ✓")
    assert last_text(bot_gw) == "Undone ✓"


async def test_undo_command_explains_itself_when_it_cannot_undo(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await bot_gw.say(make_bot_message("/undo"))
    assert last_text(bot_gw).startswith("Send /undo as a reply to the proposal message")
    await bot_gw.say(make_bot_message("/undo", reply_to_id=777))
    assert last_text(bot_gw).startswith("That message is not a proposal.")
    await low_signal_chat(rt, make_chat(), days=7)
    p = (await sent_review(rt, bot_gw))["folder"]
    await bot_gw.say(make_bot_message("/undo", reply_to_id=p.bot_message_id))
    assert last_text(bot_gw).startswith("Nothing was carried out for that proposal")
    assert await state_of(rt, p) == PROPOSAL_PROPOSED


async def test_undo_of_a_leave_is_refused_with_the_reason(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await done_proposal(rt, bot_gw, make_chat, "leave")
    assert message(bot_gw, p.bot_message_id).buttons is None  # no [Undo] on a leave
    await bot_gw.say(make_bot_message("/undo", reply_to_id=p.bot_message_id))
    assert last_text(bot_gw) == "Leaving cannot be undone without a new invitation."
    assert await state_of(rt, p) == PROPOSAL_DONE


# --- ds: new topics and merges -----------------------------------------------------------------


async def topic_proposal(
    rt: Runtime, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> Proposal:
    await unsorted_posts(rt, make_chat(username="cryptonews"))
    p = (await sent_review(rt, bot_gw))["new_topic"]
    assert button_data(bot_gw, p.bot_message_id) == [
        [f"ds:create:{p.id}", f"ds:rename:{p.id}"], [f"ds:dismiss:{p.id}"],
    ]  # fmt: skip
    return p


async def test_create_accepts_the_topic_with_its_channel_and_confirms(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    p = await topic_proposal(rt, bot_gw, make_chat)
    await press(bot_gw, f"ds:create:{p.id}")
    done = await rt.store.get_proposal(p.id)
    assert done is not None and done.state == PROPOSAL_DONE
    topic = await rt.store.get_topic(done.payload["topic_id"])
    assert topic is not None and topic.name == "Canned topic" and topic.origin == "discovered"
    assert topic.channel_id is not None and topic.channel_id in user_gw.owned
    assert len(await rt.store.list_examples(topic.id)) == 30
    assert message(bot_gw, p.bot_message_id).text.endswith("Created the topic Canned topic ✓")
    assert last_text(bot_gw) == (
        "Created the topic Canned topic.\n"
        "Its private channel Canned topic is ready: new posts on this topic go there."
    )


async def test_create_while_telegram_paces_channel_creation_says_how_long(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    p = await topic_proposal(rt, bot_gw, make_chat)
    user_gw.fail_next("create_channel", FloodWait(600))
    await press(bot_gw, f"ds:create:{p.id}")
    assert await state_of(rt, p) == PROPOSAL_DONE
    assert last_text(bot_gw).splitlines()[1] == (
        "Telegram asks to wait 10 min before another channel can be created; the channel "
        "will be created automatically then, and I will let you know. Until then the topic is "
        "tracked in the statistics only."
    )
    assert (
        rt.topics is not None and len(await rt.topics.wanted_channels()) == 1
    )  # what tick() creates


async def test_create_with_a_name_that_exists_points_to_rename(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await rt.topics.create("Canned topic")  # type: ignore[union-attr]
    p = await topic_proposal(rt, bot_gw, make_chat)
    await press(bot_gw, f"ds:create:{p.id}")
    assert await state_of(rt, p) == PROPOSAL_PROPOSED
    assert last_text(bot_gw) == (
        "A topic named Canned topic already exists. Tap Rename to choose another name, or Dismiss."
    )


async def test_rename_asks_for_a_name_and_creates_the_topic_under_it(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await topic_proposal(rt, bot_gw, make_chat)
    await press(bot_gw, f"ds:rename:{p.id}")
    assert last_text(bot_gw) == "Send the name for the new topic (suggested: Canned topic)."
    assert await state_of(rt, p) == PROPOSAL_PROPOSED

    await bot_gw.say(make_bot_message("   "))
    assert last_text(bot_gw).startswith("The name cannot be empty.")
    await bot_gw.say(make_bot_message("ML & AI"))  # one of the template's example topics
    assert last_text(bot_gw) == "A topic named ML & AI already exists. Send another name."
    assert "<b>ML &amp; AI</b>" in bot_gw.sent(OWNER_ID)[-1].html  # the name is escaped
    assert await state_of(rt, p) == PROPOSAL_PROPOSED

    await bot_gw.say(make_bot_message("Stablecoins"))
    done = await rt.store.get_proposal(p.id)
    assert done is not None and done.state == PROPOSAL_DONE
    topic = await rt.store.get_topic(done.payload["topic_id"])
    assert topic is not None and topic.name == "Stablecoins"
    assert last_text(bot_gw).startswith("Created the topic Stablecoins.\n")
    assert await app.conversation() is None  # the conversation is over

    await bot_gw.say(make_bot_message("anything"))
    assert last_text(bot_gw).startswith("Nothing is waiting for a reply")


async def test_rename_of_a_decided_proposal_is_refused(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await topic_proposal(rt, bot_gw, make_chat)
    await press(bot_gw, f"ds:dismiss:{p.id}")
    await press(bot_gw, f"ds:rename:{p.id}")
    assert last_text(bot_gw) == "That choice is not available any more."
    assert await app.conversation() is None


async def test_dismiss_skips_the_new_topic_proposal(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    p = await topic_proposal(rt, bot_gw, make_chat)
    await press(bot_gw, f"ds:dismiss:{p.id}")
    assert await state_of(rt, p) == PROPOSAL_SKIPPED
    assert message(bot_gw, p.bot_message_id).text.endswith("Dismissed.")
    assert message(bot_gw, p.bot_message_id).buttons is None
    assert toasts(bot_gw)[-1] == "Dismissed."
    assert await rt.store.list_topics() == []  # nothing was created


async def test_merge_carries_out_topics_merge(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, cluster: FakeCluster,
    make_chat: Callable[..., ChatInfo],
) -> None:  # fmt: skip
    a = await rt.topics.create("Machine learning")  # type: ignore[union-attr]
    b = await rt.topics.create("Artificial intelligence")  # type: ignore[union-attr]
    chat = make_chat()
    await rt.store.upsert_chat(chat)
    await add_post(rt, chat.id, 1, status=PostStatus.digested, topic_id=a.id)
    cluster.pairs = [TopicPair(a.id, b.id, 12)]
    p = (await sent_review(rt, bot_gw))["merge_topics"]
    assert button_data(bot_gw, p.bot_message_id) == [[f"ds:create:{p.id}", f"ds:dismiss:{p.id}"]]
    await press(bot_gw, f"ds:rename:{p.id}")  # a merge has no Rename
    assert await state_of(rt, p) == PROPOSAL_PROPOSED
    await press(bot_gw, f"ds:create:{p.id}")
    assert await state_of(rt, p) == PROPOSAL_DONE
    assert (await rt.store.get_topic(a.id)).active is False  # type: ignore[union-attr]
    assert rt.settings.topic(a.key) is None and rt.settings.topic(b.key) is not None
    assert message(bot_gw, p.bot_message_id).text.endswith("Merged into Artificial intelligence ✓")
    assert toasts(bot_gw)[-1] == "Merged."


# --- the owner guard ---------------------------------------------------------------------------


async def test_presses_and_commands_from_anyone_but_the_owner_do_nothing(
    rt: Runtime, app: BotApp, bot_gw: FakeBotGateway, make_chat: Callable[..., ChatInfo]
) -> None:
    await bot_gw.say(make_bot_message("/review", sender_id=STRANGER))
    assert bot_gw.sent(OWNER_ID) == [] and bot_gw.sent(STRANGER) == []
    assert await rt.store.proposals_by_state(PROPOSAL_PROPOSED) == []

    await low_signal_chat(rt, make_chat(), days=7)
    p = (await sent_review(rt, bot_gw))["folder"]
    edits = len(bot_gw.calls_of("edit_text"))
    await press(bot_gw, f"rv:{p.id}:approve", sender_id=STRANGER)
    await press(bot_gw, f"rv:{p.id}:never", sender_id=STRANGER)
    await bot_gw.say(make_bot_message("/undo", sender_id=STRANGER, reply_to_id=p.bot_message_id))
    assert await state_of(rt, p) == PROPOSAL_PROPOSED
    assert len(bot_gw.calls_of("edit_text")) == edits
    assert bot_gw.answers[-2:] == [("q1", None, False), ("q1", None, False)]
    assert bot_gw.sent(STRANGER) == []
