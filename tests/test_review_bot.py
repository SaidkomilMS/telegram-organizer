"""Regression tests for the review findings in ``bot/`` (one section per finding)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import tomlkit

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeClock,
    FakeUserGateway,
    FakeWorld,
    make_bot_message,
    make_callback,
    plain_text,
)
from tests.test_bot_corrections import CHANNEL_A as CHANNEL_A_CORR
from tests.test_bot_corrections import CHANNEL_B
from tests.test_bot_corrections import Scene as CorrectionScene
from tests.test_bot_llm import Server, provider_index
from tests.test_bot_topics import Driver, channel_step, settings_on_disk, wire
from tests.test_publisher import CHANNEL_A, Harness
from tg_curator.account import AccountService
from tg_curator.bot import bind, control, core, setup
from tg_curator.bot import llm as llm_mod
from tg_curator.bot.core import BotApp, Ctx
from tg_curator.domain import KV, PUB_SENDING, PUB_SENT, PostStatus, Topic
from tg_curator.errors import NotAllowed
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import EVENT_SESSION_LOST, Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import TopicsService


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


async def settle(task: asyncio.Task[Any], *, rounds: int = 50) -> None:
    """Let ``task`` run until it blocks (the store answers from a worker thread)."""
    for _ in range(rounds):
        if task.done():
            return
        await asyncio.sleep(0.01)


# --- fixtures (the other modules' fixtures, rebuilt under names of this module) -------------


@pytest.fixture
async def h(
    rt: Runtime,
    clock: FakeClock,
    world: FakeWorld,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> Harness:
    """tests/test_publisher.py's live outbox harness."""
    harness = Harness(rt, clock, world, user_gw, bot_gw, make_chat)
    await harness.setup()
    return harness


@pytest.fixture
async def drv(rt: Runtime, bot_gw: FakeBotGateway) -> Driver:
    """tests/test_bot_topics.py's driver, with the template's four topics."""
    app = wire(rt)
    await app.start()
    assert rt.topics is not None
    await rt.topics.sync_from_settings()
    return Driver(rt, bot_gw)


@pytest.fixture
async def scene(
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
) -> CorrectionScene:
    """tests/test_bot_corrections.py's scene: topics A and B with channels, C without."""
    app = wire(rt)
    await app.start()
    await rt.store.upsert_chat(make_chat(-1_009_000_000_100, title="Bank News"))
    topics = []
    for key, name, channel in (("a", "Finance", CHANNEL_A_CORR), ("b", "Markets", CHANNEL_B)):
        await rt.store.upsert_chat(make_chat(channel, title=name, is_creator=True), role="output")
        user_gw.register_owned(channel)
        topics.append(
            Topic(id=0, key=key, name=name, channel_id=channel, created_at=rt.clock.now())
        )
    topics.append(Topic(id=0, key="c", name="Tracked", channel_id=None, created_at=rt.clock.now()))
    a, b, c = [await rt.store.upsert_topic(t) for t in topics]
    rt.publisher = Publisher(rt)
    await rt.settings_file.set_value("publishing.live", True)
    return CorrectionScene(rt, Driver(rt, bot_gw), a, b, c)


# --- go-resume-reconcile-race ------------------------------------------------------------------


async def control_app(h: Harness) -> BotApp:
    rt = h.rt
    rt.account = AccountService(rt)
    rt.topics = TopicsService(rt)
    rt.publisher = h.pub
    rt.digest = DigestService(rt)
    app = BotApp(rt)
    control.register(app)
    await app.start()
    return app


class SlowSend:
    """Holds the bot's send into the topic channel until released (a Bot API round trip)."""

    def __init__(self, h: Harness) -> None:
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.real = h.bot.send_text
        self.h = h
        h.bot.send_text = self  # type: ignore[method-assign]

    async def __call__(self, chat_id: int, *args: Any, **kw: Any) -> int:
        if chat_id == CHANNEL_A:
            self.entered.set()
            await self.release.wait()
        return await self.real(chat_id, *args, **kw)


async def test_resume_waits_for_a_tick_in_flight_so_a_post_is_sent_once(h: Harness) -> None:
    await control_app(h)
    post = await h.post()
    assert await h.pub.enqueue(post.id)
    send = SlowSend(h)
    h.clock.advance(5)
    tick = asyncio.create_task(h.pub.tick())
    await asyncio.wait_for(send.entered.wait(), 2)
    assert (await h.row(post)).state == PUB_SENDING

    await h.bot.say(make_bot_message("/pause"))  # the tick keeps running
    resume = asyncio.create_task(h.bot.say(make_bot_message("/resume", message_id=2)))
    await settle(resume)
    assert not resume.done()  # waits for the send, does not treat it as a crash leftover
    assert h.user.calls_of("find_message") == []

    send.release.set()
    await tick
    await resume
    row = await h.row(post)
    assert row.state == PUB_SENT
    assert await h.rt.store.kv_get(KV.SERVICE_PAUSED) is None
    for _ in range(3):
        h.clock.advance(30)
        await h.pub.tick()
    assert len(h.sent(CHANNEL_A)) == 1
    assert (await h.row(post)).message_ids == row.message_ids


async def test_go_while_live_does_not_run_crash_recovery(h: Harness) -> None:
    await control_app(h)
    post = await h.post()
    assert await h.pub.enqueue(post.id)
    send = SlowSend(h)
    h.clock.advance(5)
    tick = asyncio.create_task(h.pub.tick())
    await asyncio.wait_for(send.entered.wait(), 2)

    await h.bot.say(make_bot_message("/go"))
    assert "already live" in owner_texts(h.bot)[-1]
    assert h.user.calls_of("find_message") == []
    assert (await h.row(post)).state == PUB_SENDING  # untouched: the tick owns it

    send.release.set()
    await tick
    for _ in range(3):
        h.clock.advance(30)
        await h.pub.tick()
    assert len(h.sent(CHANNEL_A)) == 1
    assert (await h.row(post)).state == PUB_SENT


async def test_go_before_live_still_applies_the_stale_rule(h: Harness) -> None:
    await control_app(h)
    await h.rt.settings_file.set_value("publishing.live", False)
    post = await h.post(posted_at=h.clock.now())
    assert await h.pub.enqueue(post.id)
    h.clock.advance(h.rt.settings.sorting.hold_minutes * 60 + 60)
    await h.bot.say(make_bot_message("/go"))
    fresh = await h.fresh(post)
    assert fresh.status == PostStatus.digest  # §14.6: not posted late
    assert h.rt.settings.publishing.live


# --- create-channel-orphans --------------------------------------------------------------------


def fresh_session(user_gw: FakeUserGateway, world: FakeWorld, times: int = 6) -> None:
    """A session younger than a day: Telegram refuses every admin change."""
    real = user_gw.create_channel

    async def create(title: str, about: str = "") -> ChatInfo:
        info = await real(title, about)
        world.bot_blocked.add(info.id)
        return info

    user_gw.create_channel = create  # type: ignore[method-assign]
    for _ in range(times):
        user_gw.fail_next(
            "add_bot_admin",
            NotAllowed("fresh_session", "Telegram does not let a new session change admins yet"),
        )


async def test_create_private_channel_on_a_fresh_session_links_it_and_ends_the_flow(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway, world: FakeWorld
) -> None:
    topic = await channel_step(drv, "Crypto news")
    fresh_session(user_gw, world)
    button = drv.with_button("tp:an")
    await drv.press("tp:an", button)

    assert len(user_gw.created) == 1
    channel = user_gw.created[0].id
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == channel
    assert settings_on_disk(rt)[topic.key]["channel"] == channel
    assert await drv.conversation() is None
    reply = drv.last().text
    assert "Telegram does not let a new session change admins yet" in reply
    assert "try again tomorrow or add @" in reply

    rt.clock.advance(61)  # type: ignore[attr-defined]
    await drv.press("tp:an", button)  # the old prompt cannot make a second channel
    assert len(user_gw.created) == 1
    assert drv.last().text == "That choice is not available any more."


async def test_create_channel_button_links_the_channel_and_names_other_causes(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway, world: FakeWorld
) -> None:
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None and topic.channel_id is None
    real = user_gw.create_channel

    async def create(title: str, about: str = "") -> ChatInfo:
        info = await real(title, about)
        world.bot_blocked.add(info.id)
        return info

    user_gw.create_channel = create  # type: ignore[method-assign]
    for _ in range(2):
        user_gw.fail_next("add_bot_admin", NotAllowed("other", "CHAT_ADMIN_REQUIRED"))
    await drv.press(f"tp:cn:{topic.id}", 1)

    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == user_gw.created[0].id
    reply = drv.last().text
    assert "cannot post into it: add @" in reply and "new session" not in reply


async def test_create_channel_when_the_bot_gets_its_rights_says_created(
    drv: Driver, rt: Runtime, user_gw: FakeUserGateway
) -> None:
    topic = await rt.store.get_topic_by_key("ml-ai")
    assert topic is not None
    await drv.press(f"tp:cn:{topic.id}", 1)
    row = await rt.store.get_topic(topic.id)
    assert row is not None and row.channel_id == user_gw.created[0].id
    assert "the bot is an admin there" in drv.last().text
    assert user_gw.calls_of("add_bot_admin") == []  # the bot could post: nothing retried


# --- rebind-loops-kill-login (superseded by S12) ---------------------------------------------


async def test_rebind_keeps_the_working_session_and_never_enters_setup_mode(
    rt: Runtime, bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    """/bind on a bound account logs in beside the working session (S12): the loops keep
    their session, so nothing is lost, setup mode stays off and nothing is reset."""
    rt.account = account = AccountService(rt)
    app = BotApp(rt)
    bind.register(app)
    await app.start()
    events: list[str] = []

    async def on_lost(**payload: Any) -> None:
        events.append(payload["reason"])

    rt.events.on(EVENT_SESSION_LOST, on_lost)
    await bot_gw.say(make_bot_message("/bind"))
    assert events == [] and not account.setup_mode  # an abandoned /bind changes nothing
    user_gw.valid_code = "73915"
    await bot_gw.say(make_bot_message("+998901234567", message_id=2))
    assert events == [] and not account.setup_mode
    assert user_gw.calls_of("begin_relogin") and user_gw.calls_of("reset_session") == []
    assert not any("session is lost" in t for t in owner_texts(bot_gw))

    await bot_gw.say(make_bot_message("7 3 9 1 5", message_id=3))
    assert user_gw.calls_of("sign_in") == [{"code": "73915"}]
    assert user_gw.session_generation == 2  # the new login was swapped in on success
    assert events == [] and not account.setup_mode


# --- second-callback-answer-lost ---------------------------------------------------------------


async def test_a_correction_toast_is_the_one_answer_of_its_query(
    scene: CorrectionScene,
) -> None:
    post, msg = await scene.published()
    await scene.tap(f"wt:{post.id}", msg)
    await scene.tap(f"mv:{post.id}:{scene.b.id}", msg)
    # make_callback reuses "q1": the menu tap had one silent answer, the choice one toast
    assert scene.drv.bot.answers == [
        ("q1", None, False),
        ("q1", "Now in Markets. The curator learns from this.", False),
    ]


async def test_a_duplicate_refusal_is_an_alert(scene: CorrectionScene) -> None:
    post = await scene.post(PostStatus.duplicate)
    mid = await scene.drv.bot.send_text(OWNER_ID, "a /preview item")
    await scene.drv.bot.press(
        make_callback(f"mv:{post.id}:{scene.b.id}", message_id=mid, query_id="r1")
    )
    assert scene.drv.bot.answers == [
        ("r1", "This post is a repeat of an earlier one; correct the original instead.", True)
    ]


async def test_a_slow_handler_is_answered_silently_and_its_toast_becomes_a_message(
    rt: Runtime, bot_gw: FakeBotGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "CALLBACK_ANSWER_DEADLINE", 0.02)
    app = BotApp(rt)
    await app.start()
    release = asyncio.Event()

    async def slow(ctx: Ctx, data: str) -> None:
        await release.wait()
        await ctx.answer("Done <now>", alert=True)

    app.callback("zz", slow)
    press = asyncio.create_task(bot_gw.press(make_callback("zz:1", query_id="s1")))
    for _ in range(100):
        if bot_gw.answers:
            break
        await asyncio.sleep(0.01)
    assert bot_gw.answers == [("s1", None, False)]  # in time, before the handler finished
    release.set()
    await press
    assert bot_gw.answers == [("s1", None, False)]  # never a second answer
    assert owner_texts(bot_gw)[-1] == "Done <now>"


# --- backfill-blocks-owner-chat ----------------------------------------------------------------


@dataclass
class _Result:
    messages: int = 0
    chats: int = 0
    submitted: int = 0


class BlockedBackfill:
    def __init__(self) -> None:
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.runs = 0

    @property
    def running(self) -> bool:  # contracts.Backfill: a run is under way
        return self.entered.is_set() and not self.release.is_set()

    async def run(self, *, days: int, progress: Any = None) -> _Result:
        self.runs += 1
        self.entered.set()
        await self.release.wait()
        return _Result()


async def test_pause_is_obeyed_while_preview_reads_and_a_second_read_is_refused(
    rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    app = wire(rt)
    control.register(app)
    await app.start()
    rt.backfill = backfill = BlockedBackfill()  # type: ignore[assignment]

    reading = asyncio.create_task(bot_gw.say(make_bot_message("/preview")))
    await asyncio.wait_for(backfill.entered.wait(), 2)

    await asyncio.wait_for(bot_gw.say(make_bot_message("/pause", message_id=2)), 2)
    assert await rt.store.kv_get(KV.SERVICE_PAUSED) is not None
    await asyncio.wait_for(bot_gw.say(make_bot_message("/preview", message_id=3)), 2)
    assert owner_texts(bot_gw)[-1].startswith("The last three days are being read already")
    assert backfill.runs == 1

    backfill.release.set()
    await reading
    assert "Preview of the last 3 days" in owner_texts(bot_gw)[-1]


# --- slash-password-not-deleted ----------------------------------------------------------------


@pytest.fixture
async def bind_app(rt: Runtime, user_gw: FakeUserGateway) -> BotApp:
    rt.account = AccountService(rt)
    app = BotApp(rt)
    bind.register(app)
    control.register(app)
    rt.publisher = Publisher(rt)
    await app.start()
    user_gw.authorised = False
    user_gw.valid_code = "73915"
    user_gw.password_needed = True
    return app


async def at_password_step(bot_gw: FakeBotGateway, rt: Runtime) -> None:
    await bot_gw.say(make_bot_message("/bind", message_id=1))
    await bot_gw.say(make_bot_message("+998901234567", message_id=2))
    await bot_gw.say(make_bot_message("7 3 9 1 5", message_id=3))
    assert (await rt.store.kv_get(KV.BOT_CONVERSATION))["step"] == "password"


@pytest.mark.parametrize("password", ["/Secr3t", "/my pass", "/x@y z"])
async def test_a_password_that_looks_like_a_command_is_deleted_and_passed_verbatim(
    bind_app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    password: str,
) -> None:
    user_gw.valid_password = password
    await at_password_step(bot_gw, rt)
    await bot_gw.say(make_bot_message(password, message_id=9))
    assert (OWNER_ID, 9) in bot_gw.deleted
    assert user_gw.calls_of("sign_in_password") == [{"password": password}]
    assert "logged in as" in owner_texts(bot_gw)[-1]


async def test_a_bare_command_at_the_password_step_cancels_but_is_deleted(
    bind_app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    await at_password_step(bot_gw, rt)
    await bot_gw.say(make_bot_message("/status", message_id=9))
    assert (OWNER_ID, 9) in bot_gw.deleted
    assert user_gw.calls_of("sign_in_password") == []
    assert await rt.store.kv_get(KV.BOT_CONVERSATION) is None
    assert owner_texts(bot_gw)[-1].startswith("Status")


async def test_commands_outside_a_secret_step_are_unchanged(
    bind_app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.say(make_bot_message("/bind", message_id=1))
    await bot_gw.say(make_bot_message("/nonsense", message_id=2))  # phone step: not secret
    assert (OWNER_ID, 2) not in bot_gw.deleted
    assert owner_texts(bot_gw)[-1] == "Unknown command. Send /help for the list."
    assert await rt.store.kv_get(KV.BOT_CONVERSATION) is None


@pytest.fixture
async def llm_client() -> AsyncIterator[tuple[Server, httpx.AsyncClient]]:
    server = Server()
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        yield server, client


async def test_an_api_key_that_looks_like_a_command_is_deleted(
    rt: Runtime, bot_gw: FakeBotGateway, llm_client: tuple[Server, httpx.AsyncClient]
) -> None:
    server, client = llm_client
    server.models_status = 401
    app = BotApp(rt)
    llm_mod.register(app, client=client)
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    await bot_gw.say(make_bot_message("/sk-test", message_id=12))
    assert (OWNER_ID, 12) in bot_gw.deleted
    assert owner_texts(bot_gw)[-1].startswith("OpenAI refused that key")
    assert server.requests[-1].headers["authorization"] == "Bearer /sk-test"


# --- setup-llm-none-not-recorded ---------------------------------------------------------------


async def test_choosing_none_in_llm_from_setup_completes_step_5(
    rt: Runtime, bot_gw: FakeBotGateway, user_gw: FakeUserGateway
) -> None:
    rt.account = AccountService(rt)
    rt.topics = TopicsService(rt)
    rt.publisher = Publisher(rt)
    rt.digest = DigestService(rt)
    app = BotApp(rt)
    for module in (setup, bind, control):
        module.register(app)
    llm_mod.register(app)
    await app.start()
    await rt.store.upsert_topic(
        Topic(
            id=0,
            key="ml-ai",
            name="ML & AI",
            channel_id=-1_003_000_000_001,
            created_at=rt.clock.now(),
        )
    )
    await bot_gw.say(make_bot_message("/setup"))
    assert await rt.store.kv_get(KV.SETUP_STEP) == "llm"

    await bot_gw.press(make_callback(setup.CHOOSE_LLM))
    await bot_gw.press(make_callback("lm:m:0"))
    last = bot_gw.sent(OWNER_ID)[-1]
    assert plain_text(last.html).startswith("Saved: no language model")
    assert [b.data for row in last.buttons or [] for b in row] == [setup.CONTINUE]
    assert await rt.store.kv_get(KV.SETUP_STEP) == "preview"

    await bot_gw.say(make_bot_message("/setup", message_id=2))
    lines = [t for t in owner_texts(bot_gw) if t.startswith("Setup\n")][-1].splitlines()
    assert lines[5].startswith("✓ 5. no language model")
    assert lines[6] == "→ 6. preview the last three days"


async def test_none_outside_setup_records_nothing(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    app = BotApp(rt)
    llm_mod.register(app)
    await bot_gw.press(make_callback("lm:m:0"))
    assert await rt.store.kv_get(KV.SETUP_STEP) is None
    assert bot_gw.sent(OWNER_ID)[-1].buttons is None


# --- reload-telegram-leaks-on-next-write -------------------------------------------------------


async def test_reload_keeps_the_running_owner_through_later_writes(
    rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    app = BotApp(rt)
    control.register(app)
    await app.start()
    path = rt.settings_file.path
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    doc["telegram"]["owner_id"] = 999  # type: ignore[index]
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    await bot_gw.say(make_bot_message("/reload"))
    assert "next start" in owner_texts(bot_gw)[-1]
    await rt.settings_file.set_value("digest.hour", 20)

    assert rt.settings.telegram.owner_id == OWNER_ID
    assert app.owner_id == OWNER_ID
    assert rt.settings.digest.hour == 20
    on_disk = tomlkit.parse(path.read_text(encoding="utf-8"))
    assert on_disk["telegram"]["owner_id"] == 999  # type: ignore[index]
