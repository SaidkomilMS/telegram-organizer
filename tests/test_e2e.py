"""One realistic day of the whole curator, end to end (DESIGN §9, §11, §12, §16).

Everything real is wired by ``service.build_runtime`` and started by ``Service.start`` — the
store, every service, every bot module — over the fakes: the account and the bot share one
``FakeWorld``, the models are the hashing embedder and a classifier told which text belongs to
which topic, and no language model is connected. The loops run their first tick at start and
then stay parked, so the test moves the fake clock and calls each tick itself, in the order a
day would. Telegram is observed through the fakes (what each channel shows), the database
through the store.

The day: two topics with channels, three sources (an ordinary public channel, a channel that
forbids forwarding and is fully trusted, a group), the account bound, ``/go``. A story spreads
from the channel to the protected channel (reworded) and the group (forwarded) and goes out
once with ``+2``; a trusted post goes out alone; lone posts wait 45 minutes and land in the
evening digest; the owner corrects a post; a crash between send and commit is reconciled;
``/pause`` + ``/resume`` turns a stale queued post into a digest item; ``/stats`` and a
``/review`` with one proposal that the executor carries out.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import tomlkit

from tests.fakes import (
    OWNER_ID,
    START,
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeMessage,
    FakeUserGateway,
    make_bot_message,
    make_callback,
    plain_text,
)
from tg_curator.config import SettingsFile
from tg_curator.db.store import Store
from tg_curator.domain import (
    KV,
    PROPOSAL_APPROVED,
    PROPOSAL_DONE,
    PUB_CANCELLED,
    PUB_SENDING,
    PUB_SENT,
    Post,
    PostStatus,
)
from tg_curator.notify import Notifier
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.service import ACCOUNT_LOOPS, CORE_LOOPS, Doubles, Service, build_runtime
from tg_curator.telegram.gateway import Button, ChatInfo, IncomingMessage
from tg_curator.telegram.links import permalink

TECH_CH = -1_001_000_009_001
MARKETS_CH = -1_001_000_009_002
LINK = "https://example.org/research/open-model-release?utm_source=telegram"

S1 = (
    "Researchers at an open laboratory released a new open source language model that runs on "
    "a single laptop without a graphics card. The team says the model matches much larger "
    "systems on reasoning benchmarks while using a fraction of the memory, and the weights are "
    "published under a permissive licence so anyone can study or adapt them. Independent "
    "testers who tried the release this morning report fast answers in several languages and a "
    "small download size that makes local use practical for hobbyists and schools."
)
S1_REWORDED = (
    "A research laboratory has released a new open source language model that runs on one "
    "laptop without a graphics card. According to the team the model matches much larger "
    "systems on reasoning benchmarks while using a fraction of the memory, and the weights are "
    "published under a permissive licence so anyone can study or adapt them. Testers who tried "
    "it this morning report quick answers in several languages and a small download that makes "
    "local use practical."
)
T1 = (
    "The central bank kept its key rate unchanged today and signalled that it may cut borrowing "
    "costs before the end of the year if inflation keeps slowing. Analysts had expected the "
    "pause, and the national currency barely moved after the announcement."
)
T2 = (
    "Shares of the largest national airline jumped after the company reported record summer "
    "traffic and announced plans to buy new long haul aircraft over the next decade."
)
T3 = (
    "Oil prices slipped in early trading as traders weighed rising inventories against signs of "
    "stronger demand from factories in Asia."
)
L1 = (
    "A popular messaging app added end to end encrypted backups for every user, closing a gap "
    "that security researchers had criticised for years."
)
L2 = (
    "Engineers at a European chip startup showed a prototype processor built for running neural "
    "networks on phones, claiming it can translate speech and recognise images while drawing "
    "very little power. The company plans to license the design to handset makers rather than "
    "build its own devices, and says the first phones using it could reach shops next spring if "
    "testing goes well. Several investors joined a new funding round after the demonstration."
)
L3 = (
    "A browser maker announced that its next release will block tracking cookies by default and "
    "warn users when a website tries to fingerprint their device."
)


def chat(
    chat_id: int,
    title: str,
    *,
    kind: str = "channel",
    username: str | None = None,
    noforwards: bool = False,
    mine: bool = False,
) -> ChatInfo:
    return ChatInfo(
        id=chat_id,
        kind=kind,  # type: ignore[arg-type]
        title=title,
        username=username,
        noforwards=noforwards,
        is_creator=mine,
        is_admin=mine,
        archived=False,
        muted_until=None,
    )


ORDINARY = chat(-1_001_000_000_101, "Daily Wire", username="dailywire")
PROTECTED = chat(-1_001_000_000_102, "Closed Desk", noforwards=True)
GROUP = chat(-1_001_000_000_103, "Readers Club", kind="group")


def at(hour: int, minute: int = 0, *, day: int = 0) -> datetime:
    return START.replace(hour=hour, minute=minute) + timedelta(days=day)


async def parked(_: float) -> None:
    """A loop sleep that never ends: each loop runs its first tick at start, then waits."""
    await asyncio.Event().wait()


async def settle(condition: Callable[[], bool], rounds: int = 200) -> None:
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("background work did not settle")


def texts(messages: list[FakeMessage]) -> list[str]:
    return [m.text for m in messages]


class Day:
    """The wired service plus the handles the scenario needs."""

    def __init__(
        self,
        rt: Runtime,
        svc: Service,
        clock: FakeClock,
        user_gw: FakeUserGateway,
        bot_gw: FakeBotGateway,
        make_message: Callable[..., IncomingMessage],
    ) -> None:
        self.rt = rt
        self.svc = svc
        self.clock = clock
        self.user_gw = user_gw
        self.bot_gw = bot_gw
        self.make_message = make_message

    @property
    def store(self) -> Store:
        return self.rt.store

    async def say(self, text: str) -> str:
        """The owner types ``text``; returns the bot's last reply in the private chat."""
        before = len(self.bot_gw.sent(OWNER_ID))
        await self.bot_gw.say(make_bot_message(text))
        replies = self.bot_gw.sent(OWNER_ID)[before:]
        assert replies, f"no reply to {text}"
        return replies[-1].text

    async def source_post(self, source: ChatInfo, text: str, **fields: Any) -> IncomingMessage:
        """A source publishes ``text`` now; the account's update handler delivers it."""
        msg = self.make_message(source, text=text, **fields)
        await self.user_gw.deliver(msg)
        return msg

    async def stored(self, msg: IncomingMessage) -> Post:
        post = await self.store.get_post_by_message(msg.chat.id, msg.message_id)
        assert post is not None, f"message {msg.message_id} of {msg.chat.title} was not stored"
        return post

    async def publish(self) -> None:
        """One publisher tick, a minute after the last one (pacing never sleeps, §9.4)."""
        self.clock.advance(60)
        publisher = self.rt.publisher
        assert publisher is not None
        await publisher.tick()


@pytest.fixture
async def day(
    home: Path,
    settings_file: SettingsFile,
    store: Store,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_message: Callable[..., IncomingMessage],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.0)
    monkeypatch.setattr(DigestService, "VIEWS_GAP_SECONDS", 0.0)
    monkeypatch.setattr(DigestService, "PART_GAP_SECONDS", 0.0)

    # The settings the owner ended setup with: two topics with their channels, the protected
    # channel fully trusted, two digest items so the cut is visible.
    def configure(doc: Any) -> None:
        doc["telegram"]["owner_id"] = OWNER_ID
        doc["digest"]["items"] = 2
        topics = tomlkit.aot()
        for key, name, category, channel in (
            ("tech", "Tech", "tech", TECH_CH),
            ("markets", "Markets", "finance", MARKETS_CH),
        ):
            entry = tomlkit.table()
            entry.update({"key": key, "name": name, "category": category, "channel": channel})
            topics.append(entry)
        doc["topics"] = topics
        sources = tomlkit.aot()
        source = tomlkit.table()
        source.update({"chat": PROTECTED.id, "trust": 3})
        sources.append(source)
        doc["sources"] = sources

    await settings_file.update(configure)
    for info in (
        ORDINARY,
        PROTECTED,
        GROUP,
        chat(TECH_CH, "Tech feed", mine=True),
        chat(MARKETS_CH, "Markets feed", mine=True),
    ):
        user_gw.add_chat(info)

    # Weeks before today: the group has been read for ten days and chats a lot; last
    # Sunday's review already ran.
    await store.upsert_chat(GROUP)
    await store.set_chat_fields(GROUP.id, first_seen_at=START - timedelta(days=10))
    for back in range(1, 11):
        await store.bump_chat_daily(GROUP.id, (START - timedelta(days=back)).date(), n=25)
    await store.kv_set(KV.REVIEW_LAST_DAY, "2026-10-04")

    classifier = FakeClassifier()
    doubles = Doubles(
        user=user_gw,
        bot=bot_gw,
        clock=clock,
        embedder=FakeEmbedder(),
        classifier=classifier,
        llm=FakeLLM(enabled=False),  # no language model: digest lines are first lines
    )
    rt, app = build_runtime(home, settings_file=settings_file, store=store, fakes=doubles)
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    first_ticks = (set(CORE_LOOPS) | set(ACCOUNT_LOOPS) | {"housekeeping"}) - {"chats"}
    await settle(lambda: first_ticks <= set(rt.health))

    tech = await store.get_topic_by_key("tech")
    markets = await store.get_topic_by_key("markets")
    assert tech is not None and markets is not None
    for text, topic in ((S1, tech), (L1, tech), (L2, tech), (L3, tech)):
        classifier.text_scores[text] = {topic.id: 0.9}
    for text in (T1, T2, T3):
        classifier.text_scores[text] = {markets.id: 0.9}
    try:
        yield Day(rt, svc, clock, user_gw, bot_gw, make_message)
    finally:
        await svc.shutdown()


async def test_one_day_with_the_curator(
    day: Day, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="tg_curator")
    rt, store, clock = day.rt, day.store, day.clock
    user_gw, bot_gw = day.user_gw, day.bot_gw
    tech = await store.get_topic_by_key("tech")
    markets = await store.get_topic_by_key("markets")
    assert tech is not None and markets is not None and tech.channel_id == TECH_CH
    assert user_gw.owned >= {TECH_CH, MARKETS_CH}
    assert (await store.get_chat(PROTECTED.id)).trust == 3.0  # type: ignore[union-attr]

    # --- going live --------------------------------------------------------------------------
    reply = await day.say("/go")
    assert "You are live" in reply and "tg-curator media" in reply
    assert rt.settings.publishing.live
    staging = rt.settings.publishing.staging_channel
    assert staging and staging in user_gw.owned
    assert (await store.get_chat(staging)).role == "staging"  # type: ignore[union-attr]
    assert await store.kv_get(KV.SERVICE_WENT_LIVE_AT) is not None

    # --- (a) one story from three sources: posted once, with media, +2 ------------------------
    clock.set(at(12, 1))
    s1_msg = await day.source_post(ORDINARY, S1, media="photo", urls=(LINK,))
    s1 = await day.stored(s1_msg)
    assert s1.status == PostStatus.held and s1.topic_id == tech.id
    assert s1.hold_until == at(12, 46)

    clock.set(at(12, 5))
    reworded = await day.stored(
        await day.source_post(PROTECTED, S1_REWORDED, urls=(LINK.split("?")[0],))
    )
    assert reworded.status == PostStatus.duplicate and reworded.duplicate_of == s1.id
    assert reworded.dup_kind == "url"
    s1 = await store.get_post(s1.id)  # type: ignore[assignment]
    assert s1.corroboration == 1 and s1.status == PostStatus.held  # 2.25 < 3: still waiting

    clock.set(at(12, 8))
    forward = day.make_message(
        GROUP,
        text=S1,
        urls=(LINK,),
        media="photo",
        fwd_from_chat_id=ORDINARY.id,
        fwd_from_message_id=s1_msg.message_id,
    )
    await user_gw.deliver(forward)
    clock.set(at(12, 9))
    for sender, chatter in ((501, "hi all"), (502, "anyone tried the new model?")):
        await user_gw.deliver(day.make_message(GROUP, text=chatter, sender_id=sender))
    assert await store.get_post_by_message(GROUP.id, forward.message_id) is None  # unit open

    clock.set(at(12, 14))
    assert rt.intake is not None
    await rt.intake.tick()  # the units close after five quiet minutes
    unit = await day.stored(forward)
    assert unit.kind == "unit" and unit.status == PostStatus.duplicate
    assert unit.duplicate_of == s1.id and unit.dup_kind == "forward"
    s1 = await store.get_post(s1.id)  # type: ignore[assignment]
    assert s1.corroboration == 2 and s1.status == PostStatus.queued  # promoted: it spread

    await day.publish()
    tech_msgs = bot_gw.sent(TECH_CH)
    assert len(tech_msgs) == 1
    s1_out = tech_msgs[0]
    assert s1_out.media == "photo" and s1_out.copied_from is not None
    assert s1_out.copied_from[0] == staging  # the account's copy, handed over by the bot
    assert s1_out.buttons == [[Button("Wrong topic", data=f"wt:{s1.id}")]]
    header = s1_out.text.split("\n", 1)[0]
    assert header == "Daily Wire · source · +2 more"
    assert "📎 photo attached" in s1_out.text and S1 in s1_out.text
    assert permalink(ORDINARY.id, ORDINARY.username, s1_msg.message_id) in s1_out.hrefs
    s1 = await store.get_post(s1.id)  # type: ignore[assignment]
    assert s1.status == PostStatus.published
    pub = await store.get_publication(s1.id)
    assert pub is not None and pub.state == PUB_SENT and pub.message_ids == [s1_out.message_id]
    for repeat in (reworded, unit):
        assert await store.get_publication(repeat.id) is None  # a repeat is never published
    assert len(user_gw.calls_of("copy_media")) == 1

    # --- (b) a trusted source goes out alone ---------------------------------------------------
    clock.set(at(12, 20))
    t1 = await day.stored(await day.source_post(PROTECTED, T1, media="photo"))
    assert t1.status == PostStatus.queued and t1.would_realtime and t1.corroboration == 0
    await day.publish()
    (t1_out,) = bot_gw.sent(MARKETS_CH)
    assert t1_out.text.split("\n", 1)[0] == "Closed Desk · source"  # no +N: it went alone
    assert "📎 photo at source" in t1_out.text and T1 in t1_out.text  # protected: text + link
    assert t1_out.media is None and t1_out.buttons == [[Button("Wrong topic", f"wt:{t1.id}")]]
    assert len(user_gw.calls_of("copy_media")) == 1  # nothing copied from a protected chat

    # --- (c) lone posts wait 45 minutes, then join the digest pool -----------------------------
    clock.set(at(12, 30))
    l1 = await day.stored(await day.source_post(ORDINARY, L1))
    clock.set(at(12, 35))
    l2_msg = await day.source_post(ORDINARY, L2)
    clock.set(at(12, 40))
    l2_copy = day.make_message(GROUP, text=L2, sender_id=503)
    await user_gw.deliver(l2_copy)
    clock.set(at(12, 46))
    await rt.intake.tick()
    assert (await day.stored(l2_copy)).dup_kind == "exact"
    l2 = await day.stored(l2_msg)
    assert l2.corroboration == 1 and l2.status == PostStatus.held  # one other chat: not enough
    clock.set(at(12, 50))
    l3 = await day.stored(await day.source_post(ORDINARY, L3))
    assert {p.status for p in (l1, l3)} == {PostStatus.held}

    assert rt.sorter is not None
    clock.set(at(13, 0))
    await rt.sorter.tick()
    assert (await store.get_post(l1.id)).status == PostStatus.held  # type: ignore[union-attr]
    clock.set(at(13, 16))
    await rt.sorter.tick()
    assert (await store.get_post(l1.id)).status == PostStatus.digest  # type: ignore[union-attr]
    assert (await store.get_post(l3.id)).status == PostStatus.held  # type: ignore[union-attr]

    # --- (d) the owner moves a post to another topic --------------------------------------------
    clock.set(at(13, 20))
    await bot_gw.press(make_callback(f"wt:{s1.id}", chat_id=TECH_CH, message_id=s1_out.message_id))
    s1_out = bot_gw.world.get(TECH_CH, s1_out.message_id)  # type: ignore[assignment]
    assert s1_out.buttons is not None
    assert [[b.data for b in row] for row in s1_out.buttons] == [
        [f"mv:{s1.id}:{markets.id}"],
        [f"mv:{s1.id}:0", f"mv:{s1.id}:x"],
    ]
    await bot_gw.press(
        make_callback(f"mv:{s1.id}:{markets.id}", chat_id=TECH_CH, message_id=s1_out.message_id)
    )
    s1_out = bot_gw.world.get(TECH_CH, s1_out.message_id)  # type: ignore[assignment]
    assert s1_out.text == "↪ moved to Markets" and not s1_out.buttons  # stub, nothing deleted
    assert any(text and text.startswith("Now in Markets") for _, text, _ in bot_gw.answers)
    corrections = [e for e in await store.list_examples() if e.kind == "correction"]
    assert [(e.post_id, e.topic_id, e.wrong_topic_id) for e in corrections] == [
        (s1.id, markets.id, tech.id)
    ]
    classifier = rt.classifier
    assert isinstance(classifier, FakeClassifier)
    assert [e.post_id for e in classifier.learned] == [s1.id]
    s1 = await store.get_post(s1.id)  # type: ignore[assignment]
    assert s1.topic_id == markets.id and s1.corrected and s1.status == PostStatus.queued

    await day.publish()
    markets_msgs = bot_gw.sent(MARKETS_CH)
    assert len(markets_msgs) == 2
    moved = markets_msgs[-1]
    assert moved.copied_from == s1_out.copied_from  # the staged copy is reused, not re-copied
    assert len(user_gw.calls_of("copy_media")) == 1
    assert moved.text.split("\n", 1)[0] == "Daily Wire · source · +2 more"
    assert moved.buttons == [[Button("Wrong topic", data=f"wt:{s1.id}")]]
    pub = await store.get_publication(s1.id)
    assert pub is not None and pub.state == PUB_SENT and pub.channel_id == MARKETS_CH
    assert pub.moved_from == [
        {"channel_id": TECH_CH, "message_ids": [s1_out.message_id], "topic_id": tech.id}
    ]
    assert not any(S1 in text for text in texts(bot_gw.sent(TECH_CH)))  # one channel only

    # --- (e) a crash between the send and its commit, then the restart's reconcile -------------
    clock.set(at(13, 30))
    t2_msg = await day.source_post(PROTECTED, T2)
    t2 = await day.stored(t2_msg)
    assert t2.status == PostStatus.queued
    crashing = rt.publisher
    assert isinstance(crashing, Publisher)

    async def crash(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the process died before the commit")

    monkeypatch.setattr(crashing, "_mark_sent", crash)
    with pytest.raises(RuntimeError):
        await day.publish()
    t2_link = permalink(PROTECTED.id, None, t2_msg.message_id)
    assert t2_link is not None
    shown = [m for m in bot_gw.sent(MARKETS_CH) if t2_link in m.hrefs]
    assert len(shown) == 1  # Telegram has it ...
    pub = await store.get_publication(t2.id)
    assert pub is not None and pub.state == PUB_SENDING  # ... the database does not know yet

    rt.publisher = Publisher(rt)  # the restarted process
    await rt.publisher.reconcile()
    pub = await store.get_publication(t2.id)
    assert pub is not None and pub.state == PUB_SENT and pub.message_ids == [shown[0].message_id]
    assert (await store.get_post(t2.id)).status == PostStatus.published  # type: ignore[union-attr]
    await day.publish()
    assert len([m for m in bot_gw.sent(MARKETS_CH) if t2_link in m.hrefs]) == 1  # no duplicate

    clock.set(at(13, 40))
    await rt.sorter.tick()  # the remaining holds ran out: digest
    for waited in (l2, l3):
        stored = await store.get_post(waited.id)
        assert stored is not None and stored.status == PostStatus.digest

    # --- (f) /pause, a post that would go out, /resume: it goes to the digest -------------------
    clock.set(at(14, 0))
    assert "Paused" in await day.say("/pause")
    clock.set(at(14, 1))
    t3 = await day.stored(await day.source_post(PROTECTED, T3))
    assert t3.status == PostStatus.queued
    sent_before = len(bot_gw.sent(MARKETS_CH))
    await day.publish()
    assert len(bot_gw.sent(MARKETS_CH)) == sent_before  # nothing is posted while paused
    clock.set(at(15, 0))
    assert "Resumed" in await day.say("/resume")
    t3 = await store.get_post(t3.id)  # type: ignore[assignment]
    assert t3.status == PostStatus.digest  # stale: never posted late (§14.6)
    assert (await store.get_publication(t3.id)).state == PUB_CANCELLED  # type: ignore[union-attr]
    await day.publish()
    assert len(bot_gw.sent(MARKETS_CH)) == sent_before

    # --- (g) /stats, then a review with one proposal that the account carries out --------------
    clock.set(at(15, 10))
    stats = await day.say("/stats")
    for title in ("Daily Wire", "Closed Desk", "Readers Club"):
        assert title in stats
    assert "Tech feed" not in stats and "tg-curator media" not in stats
    rows = {s.chat_id: s for s in await rt.stats.chat_stats()}  # type: ignore[union-attr]
    assert rows[ORDINARY.id].volume == 4 and rows[ORDINARY.id].sorted == 4
    assert rows[PROTECTED.id].volume == 4 and rows[PROTECTED.id].duplicates == 1
    assert rows[GROUP.id].volume == 250 + 4 and rows[GROUP.id].sorted == 0
    assert rows[GROUP.id].duplicates == 2 and rows[GROUP.id].top_repeated_chat_id == ORDINARY.id

    owner_before = len(bot_gw.sent(OWNER_ID))
    assert "One proposal sent" in await day.say("/review")
    (proposal,) = await store.proposals_by_state("proposed")
    assert proposal.kind == "folder" and proposal.chat_id == GROUP.id
    message = bot_gw.world.get(OWNER_ID, proposal.bot_message_id or 0)
    assert message is not None and bot_gw.sent(OWNER_ID).index(message) >= owner_before
    assert "Readers Club" in message.text and "254 posts in 10 days, 0 reached a topic" in (
        message.text
    )
    await bot_gw.press(make_callback(f"rv:{proposal.id}:approve", message_id=message.message_id))
    approved = await store.get_proposal(proposal.id)
    assert approved is not None and approved.state == PROPOSAL_APPROVED
    assert not user_gw.calls_of("save_folder") or all(
        GROUP.id not in call["chat_ids"] for call in user_gw.calls_of("save_folder")
    )  # approving does nothing by itself: the executor acts, paced

    clock.set(at(15, 11))
    assert rt.actions is not None
    await rt.actions.tick()
    done = await store.get_proposal(proposal.id)
    assert done is not None and done.state == PROPOSAL_DONE and done.executed_at == at(15, 11)
    assert (await store.get_chat(GROUP.id)).in_low_signal  # type: ignore[union-attr]
    assert ("Low signal", [GROUP.id]) in user_gw.folders.values()
    edited = bot_gw.world.get(OWNER_ID, message.message_id)
    assert edited is not None and edited.text.endswith("Moved to “Low signal” ✓")
    assert edited.buttons == [[Button("Undo", data=f"rv:{proposal.id}:undo")]]
    folder_writes = len(user_gw.calls_of("save_folder"))
    await rt.actions.tick()  # nothing else approved: the next tick has nothing to do
    assert len(user_gw.calls_of("save_folder")) == folder_writes

    # --- the digest hour ------------------------------------------------------------------------
    clock.set(at(21, 0))
    assert rt.digest is not None
    owner_before = len(bot_gw.sent(OWNER_ID))
    await rt.digest.tick()
    tech_digest = bot_gw.sent(TECH_CH)[-1]
    lines = tech_digest.text.split("\n\n")
    assert lines[0] == "Daily digest · 2026-10-05 · Tech"
    assert lines[1].startswith("1. Engineers at a European chip startup")  # spread: ranks first
    assert lines[1].endswith("— Daily Wire · link · +1")
    assert lines[2].startswith("2. A browser maker announced")  # tie broken newest first
    assert len(lines) == 3 and "messaging app" not in tech_digest.text
    markets_digest = bot_gw.sent(MARKETS_CH)[-1]
    assert markets_digest.text.startswith("Daily digest · 2026-10-05 · Markets\n\n1. Oil prices")
    assert markets_digest.buttons is None
    statuses = {}
    for post in (l1, l2, l3, t3):
        stored = await store.get_post(post.id)
        assert stored is not None
        statuses[post.id] = stored.status
    assert statuses == {
        l1.id: PostStatus.dropped,  # did not make the cut: dropped, never rolled over
        l2.id: PostStatus.digested,
        l3.id: PostStatus.digested,
        t3.id: PostStatus.digested,
    }
    owner_lines = [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)[owner_before:]]
    assert owner_lines == ["Digest sent — Tech: 2 posts; Markets: 1"]

    # --- the next evening: nothing rolls over ----------------------------------------------------
    tech_count, markets_count = len(bot_gw.sent(TECH_CH)), len(bot_gw.sent(MARKETS_CH))
    clock.set(at(21, 0, day=1))
    await rt.digest.tick()
    assert (len(bot_gw.sent(TECH_CH)), len(bot_gw.sent(MARKETS_CH))) == (
        tech_count,
        markets_count,
    )
    empty = await store.get_digest_by_key(tech.id, date(2026, 10, 6), seq=0)
    assert empty is not None and empty.item_count == 0 and empty.state == "sent"
    assert (await store.get_post(l1.id)).status == PostStatus.dropped  # type: ignore[union-attr]

    # --- the invariants over the whole day ------------------------------------------------------
    sent_posts = await store.posts_by_status(PostStatus.published)
    assert sorted(p.id for p in sent_posts) == sorted([s1.id, t1.id, t2.id])
    for post in sent_posts:
        username = ORDINARY.username if post.chat_id == ORDINARY.id else None
        link = permalink(post.chat_id, username, post.message_id)
        carrying = [
            m
            for channel in (TECH_CH, MARKETS_CH)
            for m in bot_gw.sent(channel)
            if link in m.hrefs and "Daily digest" not in m.text
        ]
        assert len(carrying) == 1, f"post {post.id} is shown {len(carrying)} times"

    # --- the journal names what was sorted where, not only row and chat ids (SPEC "Logs") -------
    journal = [r.getMessage() for r in caplog.records]
    sorted_lines = [line for line in journal if line.startswith("sorted ")]
    assert any(
        line.startswith('sorted "Daily Wire" (') and "-> tech (" in line for line in sorted_lines
    ), sorted_lines
    assert any(re.match(r'published post \d+ from "[^"]+" for (tech|markets) into', line)
               for line in journal)  # fmt: skip
    assert any(re.match(r"queued post \d+ from \"[^\"]+\" for (tech|markets)$", line)
               for line in journal)  # fmt: skip
    assert any(line.startswith("digest: tech sent ") for line in journal)
