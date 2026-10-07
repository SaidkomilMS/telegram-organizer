"""bot/core.py: owner guard, claim, routing, callbacks, flows, serialisation, catalogue."""

from __future__ import annotations

import asyncio
import inspect
import re
import tomllib
from collections.abc import Callable
from typing import Any

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, make_bot_message, make_callback, plain_text
from tg_curator.bot import core
from tg_curator.bot.core import CLAIM_MAX_ATTEMPTS, BotApp, Ctx, Flow
from tg_curator.db.store import Store
from tg_curator.domain import KV
from tg_curator.errors import ChatGone, CuratorError, FloodWait, TelegramUnavailable
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.runtime import EVENT_SETTINGS_CHANGED, Runtime
from tg_curator.telegram.gateway import Button

STRANGER = 4242
CHANNEL = -1_001_000_000_777


class Recorder:
    """Collects what handlers were called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def command(self, name: str) -> Callable[[Ctx, str], Any]:
        async def handler(ctx: Ctx, args: str) -> None:
            self.calls.append((name, args))

        return handler

    def callback(self, prefix: str) -> Callable[[Ctx, str], Any]:
        async def handler(ctx: Ctx, data: str) -> None:
            self.calls.append((prefix, data))
            await ctx.reply("saved")

        return handler


class DemoFlow(Flow):
    """A two-step conversation used by the flow tests."""

    first_step = "name"
    seen: list[tuple[str, str, dict[str, Any]]] = []

    async def start(self) -> None:
        await self.ctx.reply("working")

    @Flow.step("name")
    async def name(self, text: str) -> None:
        DemoFlow.seen.append(("name", text, dict(self.data)))
        await self.go("channel", name=text)

    @Flow.step("channel")
    async def channel(self, text: str) -> None:
        DemoFlow.seen.append(("channel", text, dict(self.data)))
        await self.end()


def make_app(rt: Runtime, rec: Recorder) -> BotApp:
    app = BotApp(rt)
    app.command("topics", rec.command("topics"), help_key="core_help_help")
    app.command("go", rec.command("go"))
    app.callback("tp", rec.callback("tp"))
    app.flow("demo", DemoFlow)
    return app


@pytest.fixture(autouse=True)
def _reset_flow_log() -> None:
    DemoFlow.seen = []


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
async def app(rt: Runtime, rec: Recorder) -> BotApp:
    app = make_app(rt, rec)
    await app.start()
    return app


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


# --- the owner guard -------------------------------------------------------------------------


async def test_private_message_from_a_stranger_is_ignored_silently(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.say(make_bot_message("/topics", sender_id=STRANGER))
    assert rec.calls == []
    assert bot_gw.calls == [("start", {})]
    assert bot_gw.sent(STRANGER) == []


async def test_owner_message_outside_the_private_chat_is_ignored(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.say(make_bot_message("/topics", chat_id=CHANNEL, is_private=False))
    assert rec.calls == []
    assert bot_gw.calls == [("start", {})]


async def test_stranger_callback_gets_an_empty_answer_and_nothing_else(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.press(make_callback("tp:1", sender_id=STRANGER, chat_id=CHANNEL, query_id="q9"))
    assert rec.calls == []
    assert bot_gw.answers == [("q9", None, False)]
    assert [name for name, _ in bot_gw.calls] == ["start", "answer_callback"]


async def test_owner_callback_on_a_topic_channel_button_is_obeyed_and_answered_privately(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.press(make_callback("tp:7:3", chat_id=CHANNEL))
    assert rec.calls == [("tp", "7:3")]
    assert bot_gw.sent(CHANNEL) == []
    assert owner_texts(bot_gw) == ["Saved."]


# --- claiming the bot ------------------------------------------------------------------------


@pytest.fixture
async def unclaimed(make_runtime: Callable[..., Any], store: Store, rec: Recorder) -> BotApp:
    rt = await make_runtime(store, owner_id=0)
    app = make_app(rt, rec)
    await app.start()
    return app


async def test_claim_code_is_generated_and_only_its_hash_is_stored(
    unclaimed: BotApp, store: Store
) -> None:
    code = unclaimed.claim_code
    assert code is not None and re.fullmatch(r"[A-Z2-9]{4}-[A-Z2-9]{4}", code)
    stored = await store.kv_get(KV.CLAIM_CODE_HASH)
    assert stored and code not in stored and code.replace("-", "") not in stored
    assert await store.kv_get(KV.CLAIM_ATTEMPTS) == 0


async def test_no_claim_code_when_an_owner_exists(app: BotApp, store: Store) -> None:
    assert app.claim_code is None
    assert await store.kv_get(KV.CLAIM_CODE_HASH) is None


async def test_commands_are_not_obeyed_before_the_claim(
    unclaimed: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.say(make_bot_message("/topics", sender_id=STRANGER))
    await bot_gw.say(make_bot_message("/topics"))
    assert rec.calls == []
    assert bot_gw.sent(OWNER_ID) == [] and bot_gw.sent(STRANGER) == []


async def test_bare_start_prompts_for_the_code(unclaimed: BotApp, bot_gw: FakeBotGateway) -> None:
    await bot_gw.say(make_bot_message("/start", sender_id=STRANGER))
    assert len(bot_gw.sent(STRANGER)) == 1
    assert "claim code" in plain_text(bot_gw.sent(STRANGER)[0].html)
    assert unclaimed.claim_code is not None


async def test_claim_with_start_and_code_makes_the_sender_the_owner(
    unclaimed: BotApp, bot_gw: FakeBotGateway, store: Store, rec: Recorder
) -> None:
    rt = unclaimed.rt
    fired: list[str] = []

    async def on_changed(**payload: Any) -> None:
        fired.append("settings_changed")

    rt.events.on(EVENT_SETTINGS_CHANGED, on_changed)
    await bot_gw.say(make_bot_message(f"/start {unclaimed.claim_code}", sender_id=STRANGER))
    assert rt.settings.telegram.owner_id == STRANGER
    assert fired == ["settings_changed"]
    assert unclaimed.claim_code is None
    assert await store.kv_get(KV.CLAIM_CODE_HASH) is None
    assert await store.kv_get(KV.CLAIM_ATTEMPTS) is None
    assert "owner" in plain_text(bot_gw.sent(STRANGER)[-1].html)
    # from now on the new owner is obeyed and everyone else is not
    await bot_gw.say(make_bot_message("/topics list", sender_id=STRANGER))
    await bot_gw.say(make_bot_message("/topics list", sender_id=OWNER_ID))
    assert rec.calls == [("topics", "list")]


async def test_bare_code_in_any_case_or_spacing_claims_too(
    unclaimed: BotApp, bot_gw: FakeBotGateway
) -> None:
    code = unclaimed.claim_code
    assert code is not None
    await bot_gw.say(make_bot_message(f"  {code.replace('-', '').lower()} ", sender_id=STRANGER))
    assert unclaimed.rt.settings.telegram.owner_id == STRANGER


async def test_five_wrong_attempts_regenerate_the_code(
    unclaimed: BotApp, bot_gw: FakeBotGateway, store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    first = unclaimed.claim_code
    first_hash = await store.kv_get(KV.CLAIM_CODE_HASH)
    assert first is not None
    for n in range(1, CLAIM_MAX_ATTEMPTS):
        await bot_gw.say(make_bot_message("/start AAAA-BBBB", sender_id=STRANGER))
        assert await store.kv_get(KV.CLAIM_ATTEMPTS) == n
        assert unclaimed.claim_code == first
    assert f"{CLAIM_MAX_ATTEMPTS - 1} attempts left" in plain_text(bot_gw.sent(STRANGER)[0].html)
    with caplog.at_level("WARNING", logger="tg_curator.bot.core"):
        await bot_gw.say(make_bot_message("ZZZZ-ZZZZ", sender_id=STRANGER))
    second = unclaimed.claim_code
    assert second is not None and second != first
    assert await store.kv_get(KV.CLAIM_CODE_HASH) not in (None, first_hash)
    assert await store.kv_get(KV.CLAIM_ATTEMPTS) == 0
    assert second in caplog.text and first not in caplog.text
    assert "new claim code" in plain_text(bot_gw.sent(STRANGER)[-1].html)
    # the old code is dead, the new one works
    await bot_gw.say(make_bot_message(first, sender_id=STRANGER))
    assert unclaimed.rt.settings.telegram.owner_id == 0
    await bot_gw.say(make_bot_message(second, sender_id=STRANGER))
    assert unclaimed.rt.settings.telegram.owner_id == STRANGER


async def test_chatter_before_the_claim_does_not_burn_attempts(
    unclaimed: BotApp, bot_gw: FakeBotGateway, store: Store
) -> None:
    await bot_gw.say(make_bot_message("hello, who are you?", sender_id=STRANGER))
    await bot_gw.say(make_bot_message("/help", sender_id=STRANGER))
    assert await store.kv_get(KV.CLAIM_ATTEMPTS) == 0
    assert len(bot_gw.sent(STRANGER)) == 1  # the prompt for the chatter, silence for /help


# --- command routing -------------------------------------------------------------------------


async def test_command_dispatch_with_args(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.say(make_bot_message("/topics add  ML & AI "))
    await bot_gw.say(make_bot_message("/go"))
    await bot_gw.say(make_bot_message("/Topics@curator_test_bot list"))
    assert rec.calls == [("topics", "add  ML & AI"), ("go", ""), ("topics", "list")]


async def test_unknown_command_and_idle_text_get_a_hint(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.say(make_bot_message("/nothing"))
    await bot_gw.say(make_bot_message("just some text"))
    assert rec.calls == []
    assert owner_texts(bot_gw) == [
        "Unknown command. Send /help for the list.",
        "Nothing is waiting for a reply right now. Send /help for the list of commands.",
    ]


async def test_help_lists_registered_commands(app: BotApp, bot_gw: FakeBotGateway) -> None:
    await bot_gw.say(make_bot_message("/help"))
    text = owner_texts(bot_gw)[0]
    assert text.splitlines() == ["Commands:", "/go", "/help — this list", "/topics — this list"]


async def test_handler_failures_are_reported_and_do_not_stop_the_bot(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    async def flood(ctx: Ctx, args: str) -> None:
        raise FloodWait(300)

    async def gone(ctx: Ctx, args: str) -> None:
        raise ChatGone("x")

    async def crash(ctx: Ctx, args: str) -> None:
        raise RuntimeError("boom")

    app.command("flood", flood)
    app.command("gone", gone)
    app.command("crash", crash)
    await bot_gw.say(make_bot_message("/flood"))
    await bot_gw.say(make_bot_message("/gone"))
    await bot_gw.say(make_bot_message("/crash"))
    await bot_gw.say(make_bot_message("/go"))
    assert owner_texts(bot_gw) == [
        "Telegram asks to wait 5 min before trying again.",
        "That chat cannot be found: check the link or that your account is in it.",
        "Something went wrong: RuntimeError",
    ]
    assert rec.calls == [("go", "")]


async def test_a_service_refusal_is_shown_as_is_and_an_outage_says_so(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    async def refused(ctx: Ctx, args: str) -> None:
        raise CuratorError(ctx.rt.t("paused"))  # how digest.send / review.decide refuse

    async def outage(ctx: Ctx, args: str) -> None:
        raise TelegramUnavailable("connection reset")

    app.command("refused", refused)
    app.command("outage", outage)
    await bot_gw.say(make_bot_message("/refused"))
    await bot_gw.say(make_bot_message("/outage"))
    assert owner_texts(bot_gw) == [
        "Publishing is paused. Send /resume to continue.",
        "Telegram cannot be reached right now; nothing was changed. Try again in a minute.",
    ]


# --- callbacks -------------------------------------------------------------------------------


async def test_callback_prefix_routing_answers_exactly_once(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.press(make_callback("tp:12:x", query_id="q5"))
    assert rec.calls == [("tp", "12:x")]
    # The handler did not answer, so the core answers silently once it returns: a query
    # takes exactly one answer (§11.1).
    assert bot_gw.answers == [("q5", None, False)]


async def test_unknown_callback_prefix_is_answered_and_dropped(
    app: BotApp, bot_gw: FakeBotGateway, rec: Recorder
) -> None:
    await bot_gw.press(make_callback("zz:1"))
    assert rec.calls == []
    assert len(bot_gw.answers) == 1
    assert bot_gw.sent(OWNER_ID) == []


async def test_ctx_edit_targets_the_pressed_message_and_answer_can_toast(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    mid = await bot_gw.send_text(OWNER_ID, "menu", buttons=[[Button("A", data="tp:1")]])

    async def handler(ctx: Ctx, data: str) -> None:
        await ctx.edit("saved", buttons=None)
        await ctx.answer("done", alert=True)

    app.callback("tp", handler)
    await bot_gw.press(make_callback("tp:1", message_id=mid, query_id="q2"))
    msg = bot_gw.sent(OWNER_ID)[0]
    assert plain_text(msg.html) == "Saved." and msg.buttons is None
    assert bot_gw.answers == [("q2", "done", True)]  # the toast is the one answer


async def test_callback_data_over_64_bytes_is_refused(app: BotApp, bot_gw: FakeBotGateway) -> None:
    ctx = Ctx(app, message=make_bot_message("x"), chat_id=OWNER_ID)
    ok = [[Button("ok", data="x" * 64)]]
    await ctx.reply("saved", buttons=ok)
    with pytest.raises(ValueError, match="65 bytes"):
        await ctx.reply("saved", buttons=[[Button("no", data="x" * 65)]])
    with pytest.raises(ValueError):
        await ctx.reply("saved", buttons=[[Button("no", data="ё" * 33)]])
    with pytest.raises(ValueError):
        await ctx.edit("saved", message_id=1, buttons=[[Button("no", data="x" * 65)]])
    assert len(bot_gw.sent(OWNER_ID)) == 1


async def test_reply_resolves_keys_and_passes_html_through(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    ctx = Ctx(app, message=make_bot_message("x"), chat_id=OWNER_ID)
    await ctx.reply("error_generic", error="E")
    await ctx.reply("<b>raw</b> html")
    assert [m.html for m in bot_gw.sent(OWNER_ID)] == ["Something went wrong: E", "<b>raw</b> html"]
    with pytest.raises(TypeError):
        await ctx.reply("<b>raw</b>", error="E")


async def test_delete_incoming_removes_the_owner_message(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    async def handler(ctx: Ctx, args: str) -> None:
        await ctx.delete_incoming()

    app.command("secret", handler)
    await bot_gw.say(make_bot_message("/secret 1 2 3", message_id=77))
    assert bot_gw.deleted == [(OWNER_ID, 77)]


# --- flows -----------------------------------------------------------------------------------


async def test_flow_is_persisted_and_survives_a_new_app_instance(
    rt: Runtime, rec: Recorder, bot_gw: FakeBotGateway, store: Store
) -> None:
    app1 = make_app(rt, rec)

    async def add(ctx: Ctx, args: str) -> None:
        await ctx.start_flow("demo", origin="command")

    app1.command("add", add)
    await app1.handle_message(make_bot_message("/add"))
    assert owner_texts(bot_gw) == ["Working…"]
    assert await store.kv_get(KV.BOT_CONVERSATION) == {
        "flow": "demo",
        "step": "name",
        "data": {"origin": "command"},
    }
    # "restart": a fresh BotApp on the same store continues the conversation
    app2 = make_app(rt, rec)
    await app2.handle_message(make_bot_message("ML & AI"))
    assert DemoFlow.seen == [("name", "ML & AI", {"origin": "command"})]
    assert await store.kv_get(KV.BOT_CONVERSATION) == {
        "flow": "demo",
        "step": "channel",
        "data": {"origin": "command", "name": "ML & AI"},
    }
    await app2.handle_message(make_bot_message("@mychannel"))
    assert DemoFlow.seen[-1] == ("channel", "@mychannel", {"origin": "command", "name": "ML & AI"})
    assert await store.kv_get(KV.BOT_CONVERSATION) is None
    assert rec.calls == []


async def test_any_command_cancels_the_active_flow(
    app: BotApp, bot_gw: FakeBotGateway, store: Store, rec: Recorder
) -> None:
    ctx = Ctx(app, message=make_bot_message("x"), chat_id=OWNER_ID)
    await ctx.start_flow("demo")
    assert await store.kv_get(KV.BOT_CONVERSATION) is not None
    await bot_gw.say(make_bot_message("/go"))
    assert await store.kv_get(KV.BOT_CONVERSATION) is None
    assert rec.calls == [("go", "")]
    await bot_gw.say(make_bot_message("text after the cancel"))
    assert DemoFlow.seen == []


async def test_a_conversation_without_a_handler_is_cancelled_cleanly(
    app: BotApp, bot_gw: FakeBotGateway, store: Store
) -> None:
    await store.kv_set(KV.BOT_CONVERSATION, {"flow": "gone", "step": "x", "data": {}})
    await bot_gw.say(make_bot_message("hello"))
    assert await store.kv_get(KV.BOT_CONVERSATION) is None
    assert owner_texts(bot_gw) == ["Cancelled; nothing was changed."]


async def test_ctx_flow_gives_a_callback_handler_the_active_flow(
    app: BotApp, bot_gw: FakeBotGateway, store: Store
) -> None:
    ctx = Ctx(app, message=make_bot_message("x"), chat_id=OWNER_ID)
    await ctx.start_flow("demo")

    async def handler(ctx: Ctx, data: str) -> None:
        flow = await ctx.flow()
        assert isinstance(flow, DemoFlow) and flow.current_step == "name"
        await flow.go("channel", name=data)

    app.callback("tp", handler)
    await bot_gw.press(make_callback("tp:Chosen"))
    state = await store.kv_get(KV.BOT_CONVERSATION)
    assert state["step"] == "channel" and state["data"] == {"name": "Chosen"}


def test_unknown_flow_or_non_flow_class_is_refused(rt: Runtime) -> None:
    app = BotApp(rt)
    with pytest.raises(TypeError):
        app.flow("bad", object)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        app.flow_class("missing")


# --- serialisation ---------------------------------------------------------------------------


async def test_owner_messages_never_interleave(app: BotApp, bot_gw: FakeBotGateway) -> None:
    order: list[str] = []
    gate = asyncio.Event()

    async def slow(ctx: Ctx, args: str) -> None:
        order.append("slow-start")
        await gate.wait()
        order.append("slow-end")

    async def fast(ctx: Ctx, args: str) -> None:
        order.append("fast")

    app.command("slow", slow)
    app.command("fast", fast)
    t1 = asyncio.create_task(bot_gw.say(make_bot_message("/slow")))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(bot_gw.say(make_bot_message("/fast")))
    await asyncio.sleep(0.01)
    assert order == ["slow-start"]
    gate.set()
    await asyncio.gather(t1, t2)
    assert order == ["slow-start", "slow-end", "fast"]


# --- the catalogue ---------------------------------------------------------------------------


def test_every_core_key_used_exists_and_nothing_unused_is_shipped() -> None:
    used = set(re.findall(r'"(core_[a-z_]+)"', inspect.getsource(core)))
    with (LOCALES_DIR / "en" / "core.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    t = Translator(locales_dir=LOCALES_DIR)
    common = set(re.findall(r'"((?:error|unknown|flow)_[a-z_]+)"', inspect.getsource(core)))
    assert common and common <= t.english_keys()
