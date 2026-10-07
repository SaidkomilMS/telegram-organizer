"""DigestService (DESIGN §9.5): ranking, composition, sending, schedule and reconciliation.

Everything runs on the fakes of ``tests/fakes.py`` and the fake clock: no Telegram, no model.
``render.digest_message`` belongs to the publisher; where a test must control how a digest is
split it installs a small chunking stub, and if the real module is missing the same stub
stands in so these tests do not depend on another owner's file being there yet.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from tests.fakes import OWNER_ID, START, FakeLLM, plain_text
from tg_curator.db import schema
from tg_curator.domain import (
    DIGEST_FAILED,
    DIGEST_SENDING,
    DIGEST_SENT,
    KV,
    Chat,
    DigestLine,
    NewPost,
    Post,
    PostStatus,
    Topic,
)
from tg_curator.errors import CuratorError, FloodWait
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import first_line, html_escape, text_hash


def chunking_digest_message(per_part: int) -> Callable[..., list[str]]:
    """A ``render.digest_message`` double that puts ``per_part`` items into each part, with
    the unique per-part header of §9.5, so a test decides exactly where the split falls."""

    def digest_message(
        day: date, topic_name: str, items: list[DigestLine], chats: dict[int, Chat], **kw: Any
    ) -> list[str]:
        seq = kw.get("seq", 0)
        title = "Daily digest" if seq == 0 else f"Digest (manual {seq})"
        header = f"<b>{title}</b> · {day.isoformat()} · {html_escape(topic_name)}"
        lines = [f"{it.position}. {html_escape(it.line)}" for it in items]
        if not lines:
            return [header]
        groups = [lines[i : i + per_part] for i in range(0, len(lines), per_part)]
        if len(groups) == 1:
            return [header + "\n\n" + "\n\n".join(groups[0])]
        n = len(groups)
        return [f"{header} ({i}/{n})\n\n" + "\n\n".join(g) for i, g in enumerate(groups, 1)]

    return digest_message


try:
    import tg_curator.pipeline.render  # noqa: F401
except Exception:  # the publisher's module is not there yet: stand in with the stub
    _stub = types.ModuleType("tg_curator.pipeline.render")
    _stub.digest_message = chunking_digest_message(10_000)  # type: ignore[attr-defined]
    sys.modules["tg_curator.pipeline.render"] = _stub

from tg_curator.pipeline import digest as digest_mod  # noqa: E402
from tg_curator.pipeline.digest import DigestService  # noqa: E402

ML_CHANNEL = -1_001_000_009_001
FIN_CHANNEL = -1_001_000_009_002
SRC_A = -1_001_000_000_101
SRC_B = -1_001_000_000_102
SRC_C = -1_001_000_000_103
SRC_D = -1_001_000_000_104
H21_OCT5 = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)


class Fixture:
    """Rows and fake-world state a digest test needs, built with one call each."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.store = rt.store
        self._mid: dict[int, int] = {}

    async def topic(
        self, key: str, name: str, channel_id: int, *, created_at: datetime | None = None
    ) -> Topic:
        info = ChatInfo(
            id=channel_id,
            kind="channel",
            title=f"{name} channel",
            username=None,
            noforwards=False,
            is_creator=True,
            is_admin=True,
            archived=False,
            muted_until=None,
        )
        await self.store.upsert_chat(info, role="output")
        self.rt.user.add_chat(info)
        self.rt.user.register_owned(channel_id)
        return await self.store.upsert_topic(
            Topic(
                id=0,
                key=key,
                name=name,
                channel_id=channel_id,
                created_at=created_at or self.rt.clock.now(),
            )
        )

    async def source(
        self,
        chat_id: int,
        title: str,
        *,
        kind: str = "channel",
        trust: float | None = None,
        username: str | None = None,
    ) -> Chat:
        info = ChatInfo(
            id=chat_id,
            kind=kind,  # type: ignore[arg-type]
            title=title,
            username=username,
            noforwards=False,
            is_creator=False,
            is_admin=False,
            archived=False,
            muted_until=None,
        )
        chat = await self.store.upsert_chat(info)
        self.rt.user.add_chat(info)
        if trust is not None:
            await self.store.set_chat_fields(chat_id, trust=trust)
        return chat

    async def post(
        self,
        chat_id: int,
        *,
        topic_id: int | None,
        posted_at: datetime | None = None,
        corroboration: int = 0,
        status: PostStatus = PostStatus.digest,
        text: str | None = None,
        views: int | None = None,
    ) -> Post:
        mid = self._mid.get(chat_id, 0) + 1
        self._mid[chat_id] = mid
        text = text or f"Post {mid} of chat {chat_id} with a few words.\nSecond line."
        new = NewPost(
            chat_id=chat_id,
            message_id=mid,
            kind="post",
            message_ids=[mid],
            posted_at=posted_at or self.rt.clock.now(),
            via="live",
            text=text,
            text_hash=text_hash(text),
            urls=[],
        )
        if views is not None:
            self.rt.user.views[(chat_id, mid)] = views
        post = await self.store.insert_post(
            new, status=status, topic_id=topic_id, corroboration=corroboration
        )
        assert post is not None
        return post

    async def statuses(self, *ids: int) -> list[PostStatus]:
        out = []
        for post_id in ids:
            post = await self.store.get_post(post_id)
            assert post is not None
            out.append(post.status)
        return out

    async def items(self, digest_id: int) -> list[int]:
        return [i.post_id for i in await self.store.list_digest_items(digest_id)]

    def owner_texts(self) -> list[str]:
        return [plain_text(m.html) for m in self.rt.bot.sent(OWNER_ID)]

    def channel_texts(self, channel_id: int) -> list[str]:
        return [plain_text(m.html) for m in self.rt.bot.sent(channel_id)]


@pytest.fixture
async def fx(rt: Runtime) -> Fixture:
    """Live, UTC, no language model: lines are first lines, so a test can find its texts."""
    await rt.settings_file.set_value("publishing.live", True)
    await rt.settings_file.set_value("general.timezone", "UTC")
    rt.llm = FakeLLM(enabled=False)
    rt.notifier.MIN_GAP_SECONDS = 0.0  # type: ignore[attr-defined]
    return Fixture(rt)


@pytest.fixture
def service(rt: Runtime) -> DigestService:
    svc = DigestService(rt)
    svc.VIEWS_GAP_SECONDS = 0.0
    svc.PART_GAP_SECONDS = 0.0
    rt.digest = svc
    return svc


@pytest.fixture
def chunked(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """``chunked(n)``: render ``n`` items per part for the rest of the test."""

    def install(per_part: int) -> None:
        monkeypatch.setattr(digest_mod.render, "digest_message", chunking_digest_message(per_part))

    return install


# --- ranking ---------------------------------------------------------------------------------


async def test_ranking_corroboration_attention_trust_and_ties(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    now = H21_OCT5  # ages below are measured from the tick that composes the digest
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Corroborated source")
    await fx.source(SRC_B, "Hot source")
    await fx.source(SRC_C, "Trusted source", trust=3)
    await fx.source(SRC_D, "Plain source")
    # five older posts of the hot source give it a baseline of 100 views
    for _ in range(5):
        await fx.post(
            SRC_B,
            topic_id=topic.id,
            posted_at=now - timedelta(days=2),
            status=PostStatus.published,
            views=100,
        )
    corroborated = await fx.post(
        SRC_A, topic_id=topic.id, posted_at=now - timedelta(hours=2), corroboration=3
    )
    hot = await fx.post(SRC_B, topic_id=topic.id, posted_at=now - timedelta(hours=20), views=800)
    trusted = await fx.post(SRC_C, topic_id=topic.id, posted_at=now - timedelta(hours=1))
    plain_new = await fx.post(SRC_D, topic_id=topic.id, posted_at=now - timedelta(minutes=30))
    plain_old = await fx.post(SRC_D, topic_id=topic.id, posted_at=now - timedelta(hours=3))

    rt.clock.set(H21_OCT5)
    await service.tick()

    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENT and row.item_count == 5
    # attention (clamped +3) > 2·ln(4) > trust bonus 1.0 > nothing; ties newest first
    assert await fx.items(row.id) == [
        hot.id,
        corroborated.id,
        trusted.id,
        plain_new.id,
        plain_old.id,
    ]
    chat_b = await rt.store.get_chat(SRC_B)
    assert chat_b is not None and chat_b.views_baseline == 100.0 and chat_b.baseline_at == H21_OCT5
    refreshed = await rt.store.get_post(hot.id)
    assert refreshed is not None and refreshed.views == 800 and refreshed.views_at == H21_OCT5
    calls = rt.user.calls_of("get_views")
    hot_calls = [c for c in calls if c["chat_id"] == SRC_B]
    assert len(hot_calls) == 1 and hot.message_id in hot_calls[0]["message_ids"]
    assert len(hot_calls[0]["message_ids"]) == 6  # the candidate plus the five older posts
    assert all(len(c["message_ids"]) <= 100 for c in calls)


async def test_views_read_in_batches_of_at_most_100(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Busy source")
    for _ in range(150):
        await fx.post(SRC_A, topic_id=topic.id, views=10)
    await service.preview()
    sizes = [len(c["message_ids"]) for c in rt.user.calls_of("get_views")]
    assert sizes == [100, 50]


async def test_groups_and_failed_reads_give_no_attention(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "A group", kind="group")
    await fx.source(SRC_B, "A channel")
    group_post = await fx.post(SRC_A, topic_id=topic.id, posted_at=START - timedelta(hours=2))
    channel_post = await fx.post(SRC_B, topic_id=topic.id, posted_at=START - timedelta(hours=1))
    rt.user.fail_next("get_views", CuratorError("session lost"))
    drafts = await service.preview()
    # both rank 0, so the newer channel post comes first; nothing was read for the group
    assert [i.post_id for i in drafts[0].items] == [channel_post.id, group_post.id]
    assert all(c["chat_id"] != SRC_A for c in rt.user.calls_of("get_views"))


# --- window, split, lines --------------------------------------------------------------------


async def test_window_edge_and_leftovers_dropped(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    window = timedelta(hours=rt.settings.digest.window_hours)
    inside = await fx.post(SRC_A, topic_id=topic.id, posted_at=H21_OCT5 - window)
    outside = await fx.post(
        SRC_A, topic_id=topic.id, posted_at=H21_OCT5 - window - timedelta(seconds=1)
    )
    rt.clock.set(H21_OCT5)
    await service.tick()
    assert await fx.statuses(inside.id, outside.id) == [PostStatus.digested, PostStatus.dropped]
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and await fx.items(row.id) == [inside.id]


async def test_split_into_two_parts_never_inside_an_item(
    fx: Fixture, service: DigestService, rt: Runtime, chunked: Callable[[int], None]
) -> None:
    chunked(2)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    posts = [
        await fx.post(SRC_A, topic_id=topic.id, text=f"Story number {n} here.") for n in range(3)
    ]
    rt.clock.set(H21_OCT5)
    await service.tick()
    texts = fx.channel_texts(ML_CHANNEL)
    assert len(texts) == 2
    assert texts[0].startswith("Daily digest · 2026-10-05 · ML & AI (1/2)")
    assert texts[1].startswith("Daily digest · 2026-10-05 · ML & AI (2/2)")
    for n in range(3):
        assert sum(f"Story number {n} here." in t for t in texts) == 1
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.message_ids == [m.message_id for m in rt.bot.sent(ML_CHANNEL)]
    assert await fx.items(row.id) == [p.id for p in reversed(posts)]  # ties: newest first


async def test_lines_from_llm_are_cached_and_fall_back_to_first_line(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    summarised = await fx.post(SRC_A, topic_id=topic.id, text="Long story.\nMore text.")
    others = [await fx.post(SRC_A, topic_id=topic.id) for _ in range(5)]
    rt.llm = FakeLLM()
    rt.llm.summaries["Long story.\nMore text."] = "One sentence about the long story."
    drafts = await service.preview()
    lines = {i.post_id: i.line for i in drafts[0].items}
    assert lines[summarised.id] == "One sentence about the long story."
    assert all(lines[p.id] == "Canned one-line summary." for p in others)
    for post in [summarised, *others]:
        stored = await rt.store.get_post(post.id)
        assert stored is not None and stored.summary == lines[post.id]
    assert len(rt.llm.calls) == 6
    assert all(c[1]["max_chars"] == rt.settings.digest.line_chars for c in rt.llm.calls)
    # the cache is used: a second preview asks the model nothing
    await service.preview()
    assert len(rt.llm.calls) == 6
    # without a model (or on a failed request) the first line of the post is used
    plain = await fx.post(SRC_A, topic_id=topic.id, text="First line of the post.\nSecond.")
    rt.llm = FakeLLM(enabled=False)
    drafts = await service.preview()
    lines = {i.post_id: i.line for i in drafts[0].items}
    assert lines[plain.id] == first_line("First line of the post.\nSecond.", 180)
    # without a model the digest is extractive: the cached summary is not used
    assert lines[summarised.id] == first_line("Long story.\nMore text.", 180)


async def test_preview_changes_no_status_and_writes_no_row(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    post = await fx.post(SRC_A, topic_id=topic.id)
    drafts = await service.preview("ml-ai")
    assert len(drafts) == 1 and drafts[0].topic_key == "ml-ai"
    assert drafts[0].day == date(2026, 10, 5) and len(drafts[0].parts) == 1
    assert drafts[0].items[0].position == 1 and drafts[0].items[0].post_id == post.id
    assert await fx.statuses(post.id) == [PostStatus.digest]
    assert await rt.store.digests_in_state(["pending", "sending", "sent", "failed"]) == []
    assert rt.bot.sent(ML_CHANNEL) == []
    with pytest.raises(CuratorError):
        await service.preview("no-such-topic")


# --- crash safety ----------------------------------------------------------------------------


async def test_crash_after_part_one_reconcile_sends_only_part_two(
    fx: Fixture, service: DigestService, rt: Runtime, chunked: Callable[[int], None]
) -> None:
    chunked(1)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    for n in range(2):
        await fx.post(SRC_A, topic_id=topic.id, text=f"Story {n} for the crash test.")
    rt.clock.set(H21_OCT5)
    real_send = rt.bot.send_text
    sends = 0

    async def crashing_send(chat_id: int, html: str, **kw: Any) -> int:
        nonlocal sends
        sends += 1
        if sends == 2:
            raise asyncio.CancelledError()  # the process dies between part 1 and part 2
        return await real_send(chat_id, html, **kw)

    rt.bot.send_text = crashing_send  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await service.tick()
    rt.bot.send_text = real_send  # type: ignore[method-assign]
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENDING and len(row.message_ids) == 1
    # the worst case: part 1 reached Telegram but its id was never committed
    await rt.store.set_digest_fields(row.id, message_ids=[])

    restarted = DigestService(rt)
    restarted.PART_GAP_SECONDS = 0.0
    await restarted.reconcile()

    texts = fx.channel_texts(ML_CHANNEL)
    assert len(texts) == 2  # part 1 was adopted, only part 2 was sent
    assert texts[0].startswith("Daily digest · 2026-10-05 · ML & AI (1/2)")
    assert texts[1].startswith("Daily digest · 2026-10-05 · ML & AI (2/2)")
    finished = await rt.store.get_digest(row.id)
    assert finished is not None and finished.state == DIGEST_SENT
    assert finished.message_ids == [m.message_id for m in rt.bot.sent(ML_CHANNEL)]
    looked_for = [c["contains"] for c in rt.user.calls_of("find_message")]
    assert looked_for == [
        "Daily digest · 2026-10-05 · ML & AI (1/2)",
        "Daily digest · 2026-10-05 · ML & AI (2/2)",
    ]


async def test_a_language_change_between_send_and_reconcile_sends_nothing_twice(
    fx: Fixture,
    service: DigestService,
    rt: Runtime,
    chunked: Callable[[int], None],
    tmp_path: Any,
) -> None:
    """A sent part is found by the header stored with its body, not by re-rendering it in
    the language now in effect (SHIP-1: the channel text is translatable)."""
    from tg_curator.i18n import Translator

    chunked(1)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    for n in range(2):
        await fx.post(SRC_A, topic_id=topic.id, text=f"Story {n} for the language test.")
    rt.clock.set(H21_OCT5)
    real_send = rt.bot.send_text
    sends = 0

    async def crashing_send(chat_id: int, html: str, **kw: Any) -> int:
        nonlocal sends
        sends += 1
        if sends == 2:
            raise asyncio.CancelledError()
        return await real_send(chat_id, html, **kw)

    rt.bot.send_text = crashing_send  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await service.tick()
    rt.bot.send_text = real_send  # type: ignore[method-assign]
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None
    await rt.store.set_digest_fields(row.id, message_ids=[])
    (tmp_path / "messages.toml").write_text('render_digest_daily = "Kunlik dayjest"\n', "utf-8")
    rt.t = Translator(home=tmp_path)  # the wording changed while the service was down

    restarted = DigestService(rt)
    restarted.PART_GAP_SECONDS = 0.0
    await restarted.reconcile()
    texts = fx.channel_texts(ML_CHANNEL)
    assert len(texts) == 2
    assert all(text.startswith("Daily digest · 2026-10-05") for text in texts)


async def test_promotion_race_excludes_the_promoted_post(
    fx: Fixture, service: DigestService, rt: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    posts = [
        await fx.post(SRC_A, topic_id=topic.id, text=f"Race story {n} text.") for n in range(3)
    ]
    promoted = posts[1]
    compose = service._compose

    async def compose_then_promote(*args: Any, **kw: Any) -> Any:
        draft = await compose(*args, **kw)
        await rt.store.set_post_fields(promoted.id, status=PostStatus.queued)
        return draft

    monkeypatch.setattr(service, "_compose", compose_then_promote)
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENT and row.item_count == 2
    assert promoted.id not in await fx.items(row.id)
    assert [i.position for i in await rt.store.list_digest_items(row.id)] == [1, 2]
    assert await fx.statuses(promoted.id) == [PostStatus.queued]
    text = fx.channel_texts(ML_CHANNEL)[0]
    assert "Race story 1 text." not in text
    assert "Race story 0 text." in text and "Race story 2 text." in text


async def test_a_correction_during_composition_wins(
    fx: Fixture, service: DigestService, rt: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner moves a candidate to another topic while the digest is being composed: it is
    neither listed nor dropped here; it goes out in the other topic's digest instead."""
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    other = await fx.topic("fintech", "Fintech", -1009000000099)
    await fx.source(SRC_A, "Source")
    picked = await fx.post(SRC_A, topic_id=topic.id, text="Moved story text.")
    await fx.post(SRC_A, topic_id=topic.id, text="Staying story text.")
    compose = service._compose

    async def compose_then_correct(*args: Any, **kw: Any) -> Any:
        draft = await compose(*args, **kw)
        await rt.store.set_post_fields(picked.id, topic_id=other.id)
        return draft

    monkeypatch.setattr(service, "_compose", compose_then_correct)
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.item_count == 1
    assert picked.id not in await fx.items(row.id)
    assert "Moved story text." not in fx.channel_texts(ML_CHANNEL)[0]
    other_row = await rt.store.get_digest_by_key(other.id, date(2026, 10, 5))
    assert other_row is not None and picked.id in await fx.items(other_row.id)


# --- schedule --------------------------------------------------------------------------------


async def test_scheduled_run_after_outage_runs_once(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    rt.clock.set(datetime(2026, 10, 5, 21, 30, tzinfo=UTC))  # restart after the hour passed
    await service.tick()
    assert len(rt.bot.sent(ML_CHANNEL)) == 1
    rt.clock.set(datetime(2026, 10, 6, 9, 0, tzinfo=UTC))  # next morning: due = yesterday
    await fx.post(SRC_A, topic_id=topic.id)
    await service.tick()
    assert len(rt.bot.sent(ML_CHANNEL)) == 1
    rt.clock.set(datetime(2026, 10, 6, 21, 0, tzinfo=UTC))
    await service.tick()
    assert len(rt.bot.sent(ML_CHANNEL)) == 2
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 6))) is not None


async def test_topic_created_after_the_hour_waits_for_tomorrow(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    created = datetime(2026, 10, 5, 21, 10, tzinfo=UTC)
    rt.clock.set(created)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL, created_at=created)
    await fx.source(SRC_A, "Source")
    post = await fx.post(SRC_A, topic_id=topic.id)
    rt.clock.set(datetime(2026, 10, 5, 21, 30, tzinfo=UTC))
    await service.tick()
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))) is None
    assert await fx.statuses(post.id) == [PostStatus.digest]
    rt.clock.set(datetime(2026, 10, 6, 21, 0, tzinfo=UTC))
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 6))
    assert row is not None and await fx.items(row.id) == [post.id]


async def test_went_live_after_the_hour_waits_for_tomorrow(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    went_live = datetime(2026, 10, 5, 21, 15, tzinfo=UTC)
    await rt.store.kv_set(KV.SERVICE_WENT_LIVE_AT, went_live.isoformat())
    rt.clock.set(datetime(2026, 10, 5, 21, 30, tzinfo=UTC))
    await service.tick()
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))) is None
    rt.clock.set(datetime(2026, 10, 6, 21, 0, tzinfo=UTC))
    await service.tick()
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 6))) is not None


async def test_next_run_in_local_time(fx: Fixture, service: DigestService, rt: Runtime) -> None:
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")
    # START is 12:00 UTC = 17:00 in Tashkent (UTC+5); 21:00 local today is 16:00 UTC, ahead
    assert service.next_run() == datetime(2026, 10, 5, 16, 0, tzinfo=UTC)
    rt.clock.set(datetime(2026, 10, 5, 16, 0, tzinfo=UTC))  # the moment itself is not ahead
    assert service.next_run() == datetime(2026, 10, 6, 16, 0, tzinfo=UTC)
    rt.clock.set(datetime(2026, 10, 5, 20, 0, tzinfo=UTC))  # 01:00 local next day
    assert service.next_run() == datetime(2026, 10, 6, 16, 0, tzinfo=UTC)


async def test_nothing_while_paused_or_not_live(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    rt.clock.set(H21_OCT5)
    await rt.store.kv_set(KV.SERVICE_PAUSED, True)
    await service.tick()
    with pytest.raises(CuratorError, match=rt.t("paused")):
        await service.send()
    await rt.store.kv_delete(KV.SERVICE_PAUSED)
    await rt.settings_file.set_value("publishing.live", False)
    await service.tick()
    with pytest.raises(CuratorError, match=rt.t("not_live")):
        await service.send()
    assert rt.bot.sent(ML_CHANNEL) == []
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))) is None


# --- manual sends and the empty-run rules ----------------------------------------------------


async def test_manual_send_then_scheduled_run_with_the_rest(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    await rt.settings_file.set_value("digest.items", 2)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    posts = [await fx.post(SRC_A, topic_id=topic.id) for _ in range(5)]

    results = await service.send()
    assert len(results) == 1
    assert (results[0].seq, results[0].item_count, results[0].skipped_reason) == (1, 2, None)
    manual = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5), seq=1)
    assert manual is not None and manual.manual and manual.state == DIGEST_SENT
    assert fx.channel_texts(ML_CHANNEL)[0].startswith("Digest (manual 1) · 2026-10-05 · ML & AI")
    assert (
        await fx.statuses(*(p.id for p in posts))
        == [PostStatus.digest] * 3 + [PostStatus.digested] * 2
    )
    assert fx.owner_texts()[-1] == "Digest sent — (manual) ML & AI: 2 posts"

    rt.clock.set(H21_OCT5)
    await service.tick()
    scheduled = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert scheduled is not None and not scheduled.manual and scheduled.item_count == 2
    assert fx.channel_texts(ML_CHANNEL)[1].startswith("Daily digest · 2026-10-05 · ML & AI")
    # the two next-newest went out, the last one did not make the cut and never rolls over
    assert await fx.statuses(*(p.id for p in posts)) == [
        PostStatus.dropped,
        PostStatus.digested,
        PostStatus.digested,
        PostStatus.digested,
        PostStatus.digested,
    ]
    assert fx.owner_texts()[-1] == "Digest sent — ML & AI: 2 posts"
    # a second manual send the same day gets seq 2, seq 0 stays the scheduled row's
    await fx.post(SRC_A, topic_id=topic.id)
    assert (await service.send("ml-ai"))[0].seq == 2
    rt.clock.set(datetime(2026, 10, 6, 21, 0, tzinfo=UTC))
    await service.tick()
    assert (await rt.store.get_digest_by_key(topic.id, date(2026, 10, 6))) is not None
    assert len(rt.bot.sent(ML_CHANNEL)) == 3  # nothing came back: the empty run sends nothing


async def test_manual_empty_run_writes_nothing(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    results = await service.send()
    assert len(results) == 1 and results[0].item_count == 0 and results[0].message_ids == []
    assert results[0].skipped_reason == rt.t("nothing_to_do")
    assert await rt.store.digests_in_state(["pending", "sending", "sent", "failed"]) == []
    assert rt.bot.sent(ML_CHANNEL) == [] and fx.owner_texts() == []


async def test_scheduled_empty_run_records_a_sent_row_with_zero_items(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENT and row.item_count == 0
    assert row.body == [] and row.message_ids == [] and row.sent_at == H21_OCT5
    assert rt.bot.sent(ML_CHANNEL) == [] and fx.owner_texts() == []
    await service.tick()  # the row exists: not composed again
    assert (await rt.store.next_manual_seq(topic.id, date(2026, 10, 5))) == 1


async def test_owner_line_format(fx: Fixture, service: DigestService, rt: Runtime) -> None:
    ml = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    fin = await fx.topic("fintech", "Fintech", FIN_CHANNEL)
    empty = await fx.topic("football", "Football", -1_001_000_009_003)
    await fx.source(SRC_A, "Source")
    for _ in range(15):
        await fx.post(SRC_A, topic_id=ml.id)
    for _ in range(9):
        await fx.post(SRC_A, topic_id=fin.id)
    rt.clock.set(H21_OCT5)
    await service.tick()
    assert fx.owner_texts() == ["Digest sent — ML & AI: 15 posts; Fintech: 9"]
    row = await rt.store.get_digest_by_key(empty.id, date(2026, 10, 5))
    assert row is not None and row.item_count == 0


# --- failures --------------------------------------------------------------------------------


async def test_cannot_post_fails_notifies_once_and_retries_every_10_minutes(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    post = await fx.post(SRC_A, topic_id=topic.id)
    rt.bot.cannot_post.add(ML_CHANNEL)
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_FAILED and row.attempts == 1
    assert row.next_attempt_at == H21_OCT5 + timedelta(minutes=10)
    assert row.body and row.item_count == 1  # the body is never dropped
    assert await fx.statuses(post.id) == [PostStatus.digested]
    assert fx.owner_texts() == [plain_text(rt.t("notify_cannot_post", channel="ML & AI channel"))]

    rt.clock.advance(timedelta(minutes=5))
    await service.tick()  # not due yet
    rt.clock.advance(timedelta(minutes=6))
    await service.tick()  # due, still blocked: no second warning
    row = await rt.store.get_digest(row.id)
    assert row is not None and row.state == DIGEST_FAILED and row.attempts == 2
    assert len(fx.owner_texts()) == 1

    rt.bot.cannot_post.discard(ML_CHANNEL)
    rt.clock.advance(timedelta(minutes=11))
    await service.tick()
    row = await rt.store.get_digest(row.id)
    assert row is not None and row.state == DIGEST_SENT and len(row.message_ids) == 1
    assert len(rt.bot.sent(ML_CHANNEL)) == 1
    assert fx.owner_texts()[-1] == "Digest sent — ML & AI: 1 posts"
    # the (topic, day, 0) row was re-sent, never recomposed
    assert len(await rt.store.digests_in_state([DIGEST_SENT, DIGEST_FAILED])) == 1


async def test_transient_failure_backs_off_and_verifies_before_resending(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    rt.bot.fail_next("send_text", RuntimeError("connection reset"))
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_FAILED
    assert row.next_attempt_at == H21_OCT5 + timedelta(seconds=30)
    assert row.last_error == "connection reset"
    assert rt.user.calls_of("find_message") == []
    rt.clock.advance(timedelta(seconds=30))
    await service.tick()
    row = await rt.store.get_digest(row.id)
    assert row is not None and row.state == DIGEST_SENT and len(row.message_ids) == 1
    assert len(rt.user.calls_of("find_message")) == 1  # looked before resending
    assert len(rt.bot.sent(ML_CHANNEL)) == 1


async def test_long_flood_wait_sets_the_retry_without_sleeping(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    rt.bot.fail_next("send_text", FloodWait(300))
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_FAILED
    assert row.next_attempt_at == H21_OCT5 + timedelta(seconds=300)
    assert fx.owner_texts() == []  # a flood wait is not a "cannot post"


async def test_lookup_failure_never_resends_blindly(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    await fx.post(SRC_A, topic_id=topic.id)
    rt.clock.set(H21_OCT5)
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENT
    await rt.store.set_digest_fields(row.id, state=DIGEST_SENDING, message_ids=[])
    rt.user.fail_next("find_message", CuratorError("session lost"))
    await service.reconcile()
    row = await rt.store.get_digest(row.id)
    assert row is not None and row.state == DIGEST_FAILED
    assert len(rt.bot.sent(ML_CHANNEL)) == 1  # nothing was sent again


async def test_contract_shape(service: DigestService) -> None:
    from tg_curator.contracts import DigestService as Protocol

    assert isinstance(service, Protocol)
    assert schema.digests.c.seq.default is not None  # seq 0 is the scheduled row's


# --- audit fixes: baseline, late runs, two parts at most, lines, flood waits -------------------


async def test_busy_channel_baseline_counts_only_mature_reads(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    now = H21_OCT5
    rt.clock.set(now)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL, created_at=now - timedelta(days=5))
    await fx.source(SRC_A, "Big channel")
    await fx.source(SRC_B, "Small channel")

    async def history(chat_id: int, ages: list[timedelta], real: int) -> None:
        for age in ages:
            posted = now - age
            post = await fx.post(
                chat_id, topic_id=topic.id, posted_at=posted, status=PostStatus.published
            )
            rt.user.views[(chat_id, post.message_id)] = real
            # the count stored at intake, minutes after posting
            await rt.store.set_post_fields(
                post.id, views=10, views_at=posted + timedelta(minutes=5)
            )

    await history(SRC_A, [timedelta(minutes=30 * k) for k in range(25, 144)], 5000)
    await history(SRC_B, [timedelta(hours=13 + 12 * k) for k in range(5)], 300)
    ordinary = await fx.post(SRC_A, topic_id=topic.id, posted_at=now - timedelta(hours=10))
    rt.user.views[(SRC_A, ordinary.message_id)] = 5000
    stand_out = await fx.post(SRC_B, topic_id=topic.id, posted_at=now - timedelta(hours=10))
    rt.user.views[(SRC_B, stand_out.message_id)] = 900

    drafts = await service.preview()

    assert [i.post_id for i in drafts[0].items] == [stand_out.id, ordinary.id]
    big = await rt.store.get_chat(SRC_A)
    small = await rt.store.get_chat(SRC_B)
    assert big is not None and big.views_baseline == 5000.0
    assert small is not None and small.views_baseline == 300.0


async def test_late_scheduled_run_covers_its_own_day(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    too_old = await fx.post(
        SRC_A, topic_id=topic.id, posted_at=datetime(2026, 10, 4, 17, tzinfo=UTC)
    )
    yesterday = await fx.post(
        SRC_A, topic_id=topic.id, posted_at=datetime(2026, 10, 5, 9, tzinfo=UTC)
    )
    today = await fx.post(SRC_A, topic_id=topic.id, posted_at=datetime(2026, 10, 6, 10, tzinfo=UTC))
    # an outage across the Oct 5 digest hour; the service is back on Oct 6 at 18:00
    rt.clock.set(datetime(2026, 10, 6, 18, 0, tzinfo=UTC))
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.state == DIGEST_SENT and row.item_count == 1
    assert await fx.items(row.id) == [yesterday.id]
    assert await fx.statuses(too_old.id, yesterday.id, today.id) == [
        PostStatus.dropped,
        PostStatus.digested,
        PostStatus.digest,  # after the Oct 5 hour: it waits for today's digest
    ]
    texts = fx.channel_texts(ML_CHANNEL)
    assert len(texts) == 1 and texts[0].startswith("Daily digest · 2026-10-05 · ML & AI")
    assert fx.owner_texts() == ["Digest sent — ML & AI: 1 posts"]
    rt.clock.set(datetime(2026, 10, 6, 21, 0, tzinfo=UTC))
    await service.tick()
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 6))
    assert row is not None and await fx.items(row.id) == [today.id]


async def test_a_digest_is_never_more_than_two_messages(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    await rt.settings_file.set_value("digest.items", 60)
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "A source with a rather long title, as some channels have " * 2)
    posts = [
        await fx.post(SRC_A, topic_id=topic.id, text=f"Story {n} " + "word " * 60)
        for n in range(60)
    ]
    drafts = await service.preview()
    assert len(drafts[0].parts) == 2 and 15 < len(drafts[0].items) < 60
    rt.clock.set(H21_OCT5)
    await service.tick()
    assert len(rt.bot.sent(ML_CHANNEL)) == 2
    row = await rt.store.get_digest_by_key(topic.id, date(2026, 10, 5))
    assert row is not None and row.item_count == len(await fx.items(row.id)) < 60
    statuses = await fx.statuses(*(p.id for p in posts))
    assert statuses.count(PostStatus.digested) == row.item_count
    assert statuses.count(PostStatus.dropped) == 60 - row.item_count


async def test_cached_summaries_follow_the_current_line_length(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "Source")
    post = await fx.post(SRC_A, topic_id=topic.id, text="Long story.\nMore text.")
    rt.llm = FakeLLM()
    summary = "A summary sentence " * 9  # 171 characters
    rt.llm.summaries["Long story.\nMore text."] = summary.strip()
    drafts = await service.preview()
    assert drafts[0].items[0].line == summary.strip()
    await rt.settings_file.set_value("digest.line_chars", 60)
    drafts = await service.preview()
    assert drafts[0].items[0].line == first_line(summary, 60)
    assert len(drafts[0].items[0].line) <= 60
    assert len(rt.llm.calls) == 1  # the cache is still used
    stored = await rt.store.get_post(post.id)
    assert stored is not None and stored.summary == summary.strip()


async def test_views_flood_wait_stops_reads_until_it_is_over(
    fx: Fixture, service: DigestService, rt: Runtime
) -> None:
    topic = await fx.topic("ml-ai", "ML & AI", ML_CHANNEL)
    await fx.source(SRC_A, "First channel")
    await fx.source(SRC_B, "Second channel")
    await fx.post(SRC_A, topic_id=topic.id, views=100)
    await fx.post(SRC_B, topic_id=topic.id, views=100)
    rt.user.fail_next("get_views", FloodWait(900))
    drafts = await service.preview()
    assert len(drafts[0].items) == 2  # the digest still composes, without attention
    assert len(rt.user.calls_of("get_views")) == 1  # no read after Telegram asked to wait
    rt.clock.advance(600)
    await service.preview()
    assert len(rt.user.calls_of("get_views")) == 1
    rt.clock.advance(301)
    await service.preview()
    assert len(rt.user.calls_of("get_views")) == 3
