"""bot/setup.py: /start, and /setup resuming at the first unfinished step of the seven."""

from __future__ import annotations

import inspect
import re
import tomllib
from typing import Any

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway, FakeUserGateway, make_bot_message, make_callback
from tests.fakes import plain_text as plain
from tg_curator.account import AccountService
from tg_curator.bot import bind, control, setup
from tg_curator.bot.core import BotApp, Ctx
from tg_curator.domain import KV, Topic
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import TopicsService

CHANNEL = -1_003_000_000_001
CODE = "24680"


class Others:
    """Stand-ins for the commands other modules register (/topics, /llm, /preview)."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.calls: list[tuple[str, str]] = []
        self.preview_runs = True

    def register(self, app: BotApp, names: tuple[str, ...] = ("topics", "llm", "preview")) -> None:
        for name in names:
            app.command(name, getattr(self, name))

    async def topics(self, ctx: Ctx, args: str) -> None:
        self.calls.append(("topics", args))
        await ctx.reply("<i>topics module</i>")

    async def llm(self, ctx: Ctx, args: str) -> None:
        self.calls.append(("llm", args))
        await ctx.reply("<i>llm module</i>")

    async def preview(self, ctx: Ctx, args: str) -> None:
        self.calls.append(("preview", args))
        if self.preview_runs:
            await self.rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, self.rt.clock.now().isoformat())
        await ctx.reply("<i>preview report</i>")


@pytest.fixture
async def others(rt: Runtime) -> Others:
    return Others(rt)


@pytest.fixture
async def app(rt: Runtime, others: Others) -> BotApp:
    wire(rt)
    app = BotApp(rt)
    for module in (setup, bind, control):
        module.register(app)
    others.register(app)
    await app.start()
    return app


def wire(rt: Runtime) -> None:
    rt.account = AccountService(rt)
    rt.topics = TopicsService(rt)
    rt.publisher = Publisher(rt)
    rt.digest = DigestService(rt)


def texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain(m.html) for m in bot_gw.sent(OWNER_ID)]


async def say(bot_gw: FakeBotGateway, text: str, message_id: int = 1) -> None:
    await bot_gw.say(make_bot_message(text, message_id=message_id))


def checklist(bot_gw: FakeBotGateway) -> list[str]:
    """The lines of the most recent checklist message."""
    for text in reversed(texts(bot_gw)):
        if text.startswith("Setup\n"):
            return text.splitlines()[1:]
    raise AssertionError("no checklist was sent")


async def with_channel(rt: Runtime, user_gw: FakeUserGateway) -> None:
    user_gw.add_chat(
        ChatInfo(
            id=CHANNEL,
            kind="channel",
            title="ML & AI",
            username=None,
            noforwards=False,
            is_creator=True,
            is_admin=True,
            archived=False,
            muted_until=None,
        )
    )
    for key, name, channel in (("ml-ai", "ML & AI", CHANNEL), ("fintech", "Fintech", None)):
        await rt.store.upsert_topic(
            Topic(id=0, key=key, name=name, channel_id=channel, created_at=rt.clock.now())
        )


async def connect_provider(rt: Runtime) -> None:
    def edit(doc: Any) -> None:
        doc["llm"]["mode"] = "provider"
        doc["llm"]["provider"] = "openai"
        doc["llm"]["model"] = "gpt-6-luna"
        doc["llm"]["api_key"] = "sk-test-not-shown"

    await rt.settings_file.update(edit)


def buttons_of(bot_gw: FakeBotGateway) -> list[str | None]:
    last = bot_gw.sent(OWNER_ID)[-1]
    return [b.data for row in last.buttons or [] for b in row]


# --- resuming at the right step ----------------------------------------------------------------


async def test_unbound_account_starts_at_step_3_with_the_bind_flow(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    user_gw.authorised = False
    await say(bot_gw, "/setup")
    assert checklist(bot_gw) == [
        "✓ 1. installed and running",
        "✓ 2. bot claimed: you are its owner",
        "→ 3. bind your Telegram account",
    ]
    assert "phone number" in texts(bot_gw)[-1]
    assert (await rt.store.kv_get(KV.BOT_CONVERSATION))["flow"] == "bind"
    assert await rt.store.kv_get(KV.SETUP_STEP) == "bind"
    assert others.calls == []


async def test_bound_without_a_topic_channel_hands_off_to_topics_add(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, others: Others
) -> None:
    await say(bot_gw, "/setup")
    assert checklist(bot_gw)[2:] == [
        "✓ 3. logged in as Test Owner (@test_owner)",
        "→ 4. create the topics",
    ]
    assert others.calls == [("topics", "add")]
    assert "send /setup to continue" in texts(bot_gw)[-2]
    assert await rt.store.kv_get(KV.SETUP_STEP) == "topics"


async def test_a_topic_with_a_channel_moves_on_to_the_language_model_step(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await say(bot_gw, "/setup")
    assert checklist(bot_gw)[3:] == [
        "✓ 4. 2 topics, all channels resolved. 1 without a channel yet (tracked only).",
        "→ 5. connect a language model, or skip this",
    ]
    assert buttons_of(bot_gw) == [setup.CHOOSE_LLM, setup.SKIP_LLM]
    assert others.calls == []

    await bot_gw.press(make_callback(setup.CHOOSE_LLM))
    assert others.calls == [("llm", "")]


async def test_skipping_the_model_continues_to_the_preview_and_then_go(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await say(bot_gw, "/setup")
    await bot_gw.press(make_callback(setup.SKIP_LLM))

    assert others.calls == [("preview", "")]
    lines = checklist(bot_gw)
    assert lines[4] == (
        "✓ 5. no language model: the digest shows the first line of each post and nothing "
        "leaves the server"
    )
    assert lines[5] == "→ 6. preview the last three days"
    after = texts(bot_gw)
    assert after[-3] == "preview report"
    assert after[-2] == "✓ 6. preview: the last three days were read 12:00 (0 s ago)"
    # still on UTC: the zone of the digest hour is asked before going live
    assert "which timezone" in after[-1].lower()
    assert setup.TZ_KEEP in buttons_of(bot_gw)
    await bot_gw.press(make_callback(setup.TZ_KEEP))
    assert "go live" in texts(bot_gw)[-1].lower()
    assert buttons_of(bot_gw) == [setup.GO_LIVE]
    assert rt.settings.general.timezone == "UTC"
    assert await rt.store.kv_get(KV.SETUP_STEP) == "go"


async def test_the_go_step_offers_the_servers_timezone_in_one_tap(
    app: BotApp,
    rt: Runtime,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tg_curator.bot import settings as settings_mod

    settings_mod.register(app)
    monkeypatch.setattr(setup, "server_timezone", lambda: "Asia/Tashkent")
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())
    await say(bot_gw, "/setup")
    tz_index = next(i for i, it in enumerate(settings_mod.ITEMS) if it.key == "general.timezone")
    assert buttons_of(bot_gw) == [setup.TZ_SERVER, f"st:k:{tz_index}", setup.TZ_KEEP]
    assert "Use Asia/Tashkent (the server's zone)" in str(bot_gw.sent(OWNER_ID)[-1].buttons)
    await bot_gw.press(make_callback(setup.TZ_SERVER))
    assert rt.settings.general.timezone == "Asia/Tashkent"
    assert "Timezone: Asia/Tashkent" in texts(bot_gw)[-2]
    assert buttons_of(bot_gw) == [setup.GO_LIVE]
    # a /setup now goes straight to the go button: the zone is chosen
    await say(bot_gw, "/setup")
    assert buttons_of(bot_gw) == [setup.GO_LIVE]


def test_the_servers_zone_is_read_from_the_localtime_link(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    zone_file = tmp_path / "usr" / "share" / "zoneinfo" / "Asia" / "Tashkent"
    zone_file.parent.mkdir(parents=True)
    zone_file.write_bytes(b"")
    link = tmp_path / "localtime"
    link.symlink_to(zone_file)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(setup, "LOCALTIME", link)
    assert setup.server_timezone() == "Asia/Tashkent"
    monkeypatch.setattr(setup, "LOCALTIME", tmp_path / "missing")
    assert setup.server_timezone() is None
    monkeypatch.setenv("TZ", "Europe/Berlin")
    assert setup.server_timezone() == "Europe/Berlin"
    monkeypatch.setenv("TZ", "Etc/UTC")
    assert setup.server_timezone() == "UTC"


async def test_a_connected_model_counts_as_done_and_names_itself(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    others.preview_runs = False  # the preview did not finish: the walkthrough waits there
    await say(bot_gw, "/setup")
    lines = checklist(bot_gw)
    assert lines[4] == "✓ 5. language model: openai, gpt-6-luna"
    assert lines[5] == "→ 6. preview the last three days"
    assert "sk-test-not-shown" not in "\n".join(texts(bot_gw))
    assert others.calls == [("preview", "")]
    assert texts(bot_gw)[-1] == "preview report"


async def test_everything_done_but_live_offers_go_and_the_button_finishes_setup(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")
    await say(bot_gw, "/setup")
    assert checklist(bot_gw)[-1] == "→ 7. go live"
    assert buttons_of(bot_gw) == [setup.GO_LIVE]
    assert others.calls == []

    await bot_gw.press(make_callback(setup.GO_LIVE))
    assert rt.settings.publishing.live is True
    assert any("You are live" in t for t in texts(bot_gw))
    assert checklist(bot_gw)[-1] == "✓ 7. live since 17:00 (0 s ago)"  # Tashkent time
    assert texts(bot_gw)[-1].startswith("Setup is complete.")
    assert await rt.store.kv_get(KV.SETUP_STEP) == "done"


async def test_a_refused_go_does_not_claim_completion(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())
    bot_gw.cannot_post.add(CHANNEL)
    await say(bot_gw, "/setup")
    await bot_gw.press(make_callback(setup.GO_LIVE))
    assert rt.settings.publishing.live is False
    assert texts(bot_gw)[-1] == "No topic has a channel the bot can post into: /topics"


async def test_setup_when_everything_is_done_confirms_all_seven_steps(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())
    await say(bot_gw, "/go")
    await rt.store.kv_set(KV.SERVICE_PAUSED, rt.clock.now().isoformat())
    await say(bot_gw, "/setup")
    lines = checklist(bot_gw)
    assert [line[:4] for line in lines] == [f"✓ {n}." for n in range(1, 8)]
    assert lines[-1] == "✓ 7. live since 12:00 (0 s ago) (paused — /resume)"
    assert texts(bot_gw)[-1].startswith("Setup is complete.")
    assert others.calls == []


async def test_running_setup_twice_is_harmless(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await say(bot_gw, "/setup")
    first = texts(bot_gw)
    await say(bot_gw, "/setup")
    assert texts(bot_gw)[len(first) :] == first
    assert await rt.store.kv_get(KV.SETUP_STEP) == "llm"


async def test_the_model_choice_is_remembered_when_an_earlier_step_breaks(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    await with_channel(rt, user_gw)
    await say(bot_gw, "/setup")
    await bot_gw.press(make_callback(setup.SKIP_LLM))  # "no model"; the preview runs too
    await user_gw.lose_session("revoked")
    await say(bot_gw, "/setup")
    assert checklist(bot_gw)[-1] == "→ 3. bind your Telegram account"
    assert await rt.store.kv_get(KV.SETUP_STEP) == "go"  # progress never moves back

    user_gw.authorised = True
    await say(bot_gw, "/setup")
    assert checklist(bot_gw)[-1] == "→ 7. go live"


async def test_binding_from_setup_offers_continue_and_comes_back(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, others: Others
) -> None:
    user_gw.authorised = False
    user_gw.valid_code = CODE
    await say(bot_gw, "/setup", 1)
    await say(bot_gw, "+998901234567", 2)
    await say(bot_gw, " ".join(CODE), 3)
    assert (OWNER_ID, 3) in bot_gw.deleted
    assert "logged in as Test Owner" in texts(bot_gw)[-1]
    assert buttons_of(bot_gw) == [setup.CONTINUE]

    await bot_gw.press(make_callback(setup.CONTINUE))
    assert checklist(bot_gw)[-1] == "→ 4. create the topics"
    assert others.calls == [("topics", "add")]


async def test_a_missing_module_gets_a_hint_instead(
    rt: Runtime, bot_gw: FakeBotGateway, others: Others
) -> None:
    wire(rt)
    app = BotApp(rt)
    for module in (setup, bind, control):
        module.register(app)
    await app.start()
    await say(bot_gw, "/setup")
    assert texts(bot_gw)[-1] == "Send /topics add to do this step, then /setup to continue."


# --- /start ------------------------------------------------------------------------------------


async def test_start_welcomes_the_owner_and_points_to_setup(
    app: BotApp, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    user_gw.authorised = False
    await say(bot_gw, "/start")
    assert "Setup is at step 3 of 7: send /setup to continue" in texts(bot_gw)[-1]


async def test_start_when_everything_is_set_up(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await with_channel(rt, user_gw)
    await connect_provider(rt)
    await rt.store.kv_set(KV.SETUP_PREVIEW_SHOWN, rt.clock.now().isoformat())
    await rt.settings_file.set_value("publishing.live", True)
    await say(bot_gw, "/start")
    assert texts(bot_gw)[-1].startswith("Hello! Everything is set up.")


async def test_an_unknown_setup_button_is_refused(app: BotApp, bot_gw: FakeBotGateway) -> None:
    await bot_gw.press(make_callback("su:nonsense"))
    assert texts(bot_gw)[-1] == "That choice is not available any more."


# --- the catalogue -----------------------------------------------------------------------------


def test_every_setup_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(setup)
    used = set(re.findall(r'"(setup_[a-z_]+)"', source))
    used |= {f"setup_title_{step}" for step in setup.STEPS}  # looked up by step name
    with (LOCALES_DIR / "en" / "setup.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = set(re.findall(r'"(unknown_[a-z_]+)"', source))
    assert common and common <= Translator(locales_dir=LOCALES_DIR).english_keys()
