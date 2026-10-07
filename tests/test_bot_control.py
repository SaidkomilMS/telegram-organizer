"""bot/control.py: /go preconditions and replies, /pause and /resume, /status, /reload."""

from __future__ import annotations

import inspect
import re
import tomllib
from datetime import timedelta
from typing import Any

import pytest
import tomlkit

from tests.fakes import OWNER_ID, FakeBotGateway, FakeClassifier, FakeUserGateway, make_bot_message
from tests.fakes import plain_text as plain
from tg_curator.account import AccountService
from tg_curator.bot import control
from tg_curator.bot.core import BotApp
from tg_curator.config import WEEKDAYS
from tg_curator.domain import KV, Topic
from tg_curator.errors import FloodWait
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.pipeline.digest import DigestService
from tg_curator.pipeline.publisher import Publisher
from tg_curator.runtime import EVENT_SETTINGS_CHANGED, LoopState, Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.topics.service import STAGING_TITLE, TopicsService

CHANNEL = -1_003_000_000_001


@pytest.fixture
async def app(rt: Runtime) -> BotApp:
    rt.account = AccountService(rt)
    rt.topics = TopicsService(rt)
    rt.publisher = Publisher(rt)
    rt.digest = DigestService(rt)
    app = BotApp(rt)
    control.register(app)
    await app.start()
    return app


def ticked(rt: Runtime, interval: float, ago: timedelta) -> LoopState:
    """A running loop whose last successful tick was ``ago`` before now."""
    now = rt.clock.now()
    return LoopState(
        interval=interval, started_at=now - timedelta(days=1), last_ok_at=now - ago, running=True
    )


async def say(bot_gw: FakeBotGateway, text: str) -> str:
    await bot_gw.say(make_bot_message(text))
    return plain(bot_gw.sent(OWNER_ID)[-1].html)


async def add_topic(rt: Runtime, user_gw: FakeUserGateway, *, channel: int | None) -> Topic:
    if channel is not None:
        user_gw.add_chat(
            ChatInfo(
                id=channel,
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
    return await rt.store.upsert_topic(
        Topic(id=0, key="ml-ai", name="ML & AI", channel_id=channel, created_at=rt.clock.now())
    )


class Reconciles:
    """Records each publisher.reconcile() with the switches as they were at that moment."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.seen: list[dict[str, Any]] = []
        assert rt.publisher is not None
        self.real = rt.publisher.reconcile
        rt.publisher.reconcile = self  # type: ignore[method-assign]

    async def __call__(self) -> None:
        self.seen.append(
            {
                "live": self.rt.settings.publishing.live,
                "paused": bool(await self.rt.store.kv_get(KV.SERVICE_PAUSED)),
            }
        )
        await self.real()


# --- /go ---------------------------------------------------------------------------------------


async def test_go_refuses_without_a_bound_account(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    user_gw.authorised = False
    await add_topic(rt, user_gw, channel=CHANNEL)
    assert await say(bot_gw, "/go") == "No account bound: /bind"
    assert rt.settings.publishing.live is False
    assert await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT) is None


async def test_go_refuses_when_no_topic_has_a_channel(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=None)
    assert await say(bot_gw, "/go") == "No topic has a channel the bot can post into: /topics"
    assert rt.settings.publishing.live is False


async def test_go_refuses_when_the_bot_cannot_post_into_any_channel(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    bot_gw.cannot_post.add(CHANNEL)
    assert await say(bot_gw, "/go") == "No topic has a channel the bot can post into: /topics"
    assert rt.settings.publishing.live is False
    assert user_gw.calls_of("create_channel") == []


async def test_go_sets_live_creates_staging_and_replies_with_the_local_schedule(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")  # UTC+5: 17:00 now
    await add_topic(rt, user_gw, channel=CHANNEL)
    reconciles = Reconciles(rt)
    reply = await say(bot_gw, "/go")

    assert rt.settings.publishing.live is True
    assert await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT) == rt.clock.now().isoformat()
    assert [c["title"] for c in user_gw.calls_of("create_channel")] == [STAGING_TITLE]
    staging = rt.settings.publishing.staging_channel
    assert staging != 0 and staging in user_gw.owned
    assert reconciles.seen == [{"live": False, "paused": False}]  # stale rule before going live

    assert "You are live" in reply
    assert "The first digest arrives today at 21:00 (Asia/Tashkent)" in reply
    assert "weekly review comes on Sundays at 11:00" in reply
    assert "tg-curator media" in reply and "keep it, nobody else sees it" in reply
    assert "Send /pause to stop posting without stopping intake" in reply


async def test_go_after_the_digest_hour_names_tomorrow(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("digest.hour", 9)
    await add_topic(rt, user_gw, channel=CHANNEL)
    assert "arrives tomorrow at 09:00 (UTC)" in await say(bot_gw, "/go")


async def test_go_in_forward_style_creates_no_staging_channel(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await rt.settings_file.set_value("publishing.style", "forward")
    await add_topic(rt, user_gw, channel=CHANNEL)
    reply = await say(bot_gw, "/go")
    assert rt.settings.publishing.live is True
    assert user_gw.calls_of("create_channel") == []
    assert "tg-curator media" not in reply


async def test_go_twice_keeps_the_first_went_live_moment(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    await say(bot_gw, "/go")
    first = await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT)
    rt.clock.advance(timedelta(hours=1))  # type: ignore[attr-defined]
    reply = await say(bot_gw, "/go")
    assert "already live" in reply
    assert await rt.store.kv_get(KV.SERVICE_WENT_LIVE_AT) == first
    assert len(user_gw.calls_of("create_channel")) == 1  # the staging channel is reused


async def test_go_while_paused_says_so(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    await say(bot_gw, "/pause")
    reply = await say(bot_gw, "/go")
    assert rt.settings.publishing.live is True
    assert "Publishing is paused. Send /resume" in reply


# --- /pause and /resume ------------------------------------------------------------------------


async def test_pause_and_resume_flip_the_flag_and_resume_reconciles_first(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    await say(bot_gw, "/go")
    reconciles = Reconciles(rt)

    assert "Paused" in await say(bot_gw, "/pause")
    assert await rt.store.kv_get(KV.SERVICE_PAUSED)
    assert "already paused" in await say(bot_gw, "/pause")

    reply = await say(bot_gw, "/resume")
    assert "Resumed" in reply and "45 min" in reply
    assert not await rt.store.kv_get(KV.SERVICE_PAUSED)
    assert reconciles.seen == [{"live": True, "paused": True}]  # stale rule before the lift
    assert await say(bot_gw, "/resume") == "Publishing is not paused."


async def test_pause_before_go_is_kept_and_explained(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    assert "not live yet" in await say(bot_gw, "/pause")
    assert await rt.store.kv_get(KV.SERVICE_PAUSED)
    assert "send /go" in await say(bot_gw, "/resume")


# --- /status -----------------------------------------------------------------------------------


async def test_status_lists_every_section(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    await say(bot_gw, "/go")
    now = rt.clock.now()
    rt.loops.update(
        {
            "intake": ticked(rt, 5.0, timedelta(seconds=5)),
            "publisher": ticked(rt, 2.0, timedelta(0)),
        }
    )
    rt.health.update({"intake": now - timedelta(seconds=5), "publisher": now})
    rt.health["retrain"] = now - timedelta(minutes=3)  # on demand: listed, no verdict
    await rt.store.kv_set(KV.INTAKE_LAST_MESSAGE_AT, (now - timedelta(minutes=12)).isoformat())
    await rt.store.kv_set(KV.FOLDERS_DISABLED, "limit")

    reply = await say(bot_gw, "/status")
    lines = reply.splitlines()
    assert lines[0] == "Status"
    assert lines[1] == "✓ Everything is running"
    assert "Publishing: live since 12:00 (0 s ago)" in lines
    assert "Account: logged in as Test Owner (@test_owner)" in lines
    assert "Last post ingested: 11:48 (12 min ago)" in lines
    assert "Next digest: 21:00 (in 9 h 0 min)" in lines
    assert any(
        line.startswith("Folders: off — Telegram's folder limit is reached") for line in lines
    )
    assert "• intake: ok (last tick 5 s ago)" in lines
    assert "• publisher: ok (last tick 0 s ago)" in lines
    assert "On demand (last run): retrain 3 min ago" in lines


async def test_status_before_anything_ran(app: BotApp, rt: Runtime, bot_gw: FakeBotGateway) -> None:
    reply = await say(bot_gw, "/status")
    assert "Publishing: not live yet — send /go" in reply
    assert "Last post ingested: none yet" in reply
    assert "Folders: on" in reply
    assert "The loops have not started yet" in reply


async def test_status_shows_pause_and_a_lost_session(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await rt.store.kv_set(KV.ACCOUNT_ID, OWNER_ID)
    await say(bot_gw, "/pause")
    await user_gw.lose_session("revoked")
    reply = await say(bot_gw, "/status")
    assert "Publishing: paused since 12:00 (0 s ago) — send /resume" in reply
    assert "Account: session lost — run /bind" in reply


async def test_status_names_no_digest_while_publishing_is_not_live(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    lines = (await say(bot_gw, "/status")).splitlines()
    assert "Next digest: none — publishing is not live yet (send /go)" in lines
    assert not any(line.startswith("Next digest: 21:00") for line in lines)


async def test_status_names_no_digest_while_paused(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    await say(bot_gw, "/go")
    await say(bot_gw, "/pause")
    lines = (await say(bot_gw, "/status")).splitlines()
    assert "Next digest: none — publishing is paused (send /resume)" in lines
    await say(bot_gw, "/resume")
    lines = (await say(bot_gw, "/status")).splitlines()
    assert "Next digest: 21:00 (in 9 h 0 min)" in lines


async def test_status_flags_a_stalled_loop(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    rt.loops.update(
        {
            "intake": ticked(rt, 5.0, timedelta(hours=6)),
            "sorter": ticked(rt, 30.0, timedelta(seconds=40)),  # within 3 intervals: fine
            "chats": ticked(rt, 1800.0, timedelta(minutes=50)),  # 1800 s interval: fine
        }
    )
    rt.health["retrain"] = rt.clock.now() - timedelta(days=3)  # on demand: no verdict
    lines = (await say(bot_gw, "/status")).splitlines()
    assert lines[1] == "⚠ 1 problem: intake — see Loops below"
    assert "• intake: stalled (last tick 6 h 0 min ago)" in lines
    assert "• sorter: ok (last tick 40 s ago)" in lines
    assert lines.index("• intake: stalled (last tick 6 h 0 min ago)") < lines.index(
        "• chats: ok (last tick 50 min ago)"
    )  # problems first


async def test_status_reports_account_loops_paused_after_a_lost_session(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await rt.store.kv_set(KV.ACCOUNT_ID, OWNER_ID)
    rt.loops.update(
        {
            "intake": LoopState(interval=5.0, paused=True),
            "sorter": ticked(rt, 30.0, timedelta(seconds=3)),
        }
    )
    await user_gw.lose_session("revoked")
    lines = (await say(bot_gw, "/status")).splitlines()
    assert lines[1] == "⏸ Running in setup mode: what needs the account waits for /bind"
    assert "• intake: paused (account not bound — /bind)" in lines
    assert "Account: session lost — run /bind" in lines


async def test_status_for_an_account_never_bound(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    user_gw.authorised = False
    assert "Account: not bound — run /bind" in await say(bot_gw, "/status")


# --- /reload -----------------------------------------------------------------------------------


def hand_edit(rt: Runtime, edit: Any) -> None:
    """Change the file on disk the way a user with an editor would (no settings_changed)."""
    path = rt.settings_file.path
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


async def test_reload_rereads_the_file_and_resyncs_everything(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    fired: list[str] = []

    async def on_changed(**payload: Any) -> None:
        fired.append(EVENT_SETTINGS_CHANGED)

    rt.events.on(EVENT_SETTINGS_CHANGED, on_changed)
    await rt.store.kv_set(KV.FOLDERS_DISABLED, "limit")
    assert isinstance(rt.classifier, FakeClassifier)
    reloads = rt.classifier.reloads

    def edit(doc: Any) -> None:
        doc["digest"]["hour"] = 20

    hand_edit(rt, edit)
    reply = await say(bot_gw, "/reload")

    assert rt.settings.digest.hour == 20
    assert fired == [EVENT_SETTINGS_CHANGED]
    assert [t.key for t in await rt.store.list_topics()] == [
        "ml-ai",
        "fintech",
        "uzbekistan",
        "football",
    ]
    assert rt.classifier.reloads >= reloads + 2  # the sync's reload and the explicit one
    assert await rt.store.kv_get(KV.FOLDERS_DISABLED) is None
    assert "Settings file re-read." in reply
    assert "4 topics, all channels resolved. 4 without a channel yet (tracked only)." in reply
    assert "next start" not in reply


async def test_reload_lists_unresolved_channels(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    def edit(doc: Any) -> None:
        doc["topics"][0]["channel"] = "@no_such_channel"

    hand_edit(rt, edit)
    reply = await say(bot_gw, "/reload")
    assert "4 topics, 0 channels resolved. To fix:" in reply
    assert "• topic ml-ai: topic channel @no_such_channel cannot be found" in reply


async def test_reload_keeps_the_running_telegram_and_storage_values(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    def edit(doc: Any) -> None:
        doc["telegram"]["owner_id"] = 999
        doc["storage"]["database_url"] = "postgresql+asyncpg://elsewhere/db"
        doc["general"]["timezone"] = "Europe/Berlin"

    hand_edit(rt, edit)
    reply = await say(bot_gw, "/reload")
    assert rt.settings.general.timezone == "Europe/Berlin"
    assert rt.settings.telegram.owner_id == OWNER_ID
    assert rt.settings.storage.database_url == ""
    assert "take effect at the next start" in reply
    assert "Status" in await say(bot_gw, "/status")  # the running owner is still obeyed


async def test_reload_rebuilds_the_language_model_when_llm_changed(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    before = rt.llm
    await say(bot_gw, "/reload")
    assert rt.llm is before  # nothing in [llm] changed: the running model stays

    def edit(doc: Any) -> None:
        doc["llm"]["mode"] = "selfhosted"
        doc["llm"]["base_url"] = "http://localhost:11434/v1"
        doc["llm"]["model"] = "qwen3:8b"

    hand_edit(rt, edit)
    await say(bot_gw, "/reload")
    assert rt.llm is not before and rt.llm.enabled
    await rt.llm.aclose()  # type: ignore[attr-defined]


async def test_a_broken_file_is_reported_and_nothing_changes(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    def edit(doc: Any) -> None:
        doc["digest"]["hour"] = 99

    hand_edit(rt, edit)
    reply = await say(bot_gw, "/reload")
    assert reply.startswith("Settings problem:") and "digest.hour" in reply
    assert rt.settings.digest.hour == 21


# --- /help and the catalogue -------------------------------------------------------------------


async def test_help_lists_the_control_commands(app: BotApp, bot_gw: FakeBotGateway) -> None:
    reply = await say(bot_gw, "/help")
    for name in ("go", "pause", "resume", "status", "reload", "help"):
        assert f"/{name} — " in reply


def test_every_control_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(control)
    used = set(re.findall(r'"(control_[a-z_]+)"', source))
    used |= {f"control_weekday_{day}" for day in WEEKDAYS}  # looked up by the settings value
    with (LOCALES_DIR / "en" / "control.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = set(re.findall(r'"(paused|unknown)"', source))
    assert common == {"paused", "unknown"}
    assert common <= Translator(locales_dir=LOCALES_DIR).english_keys()


async def test_a_staging_failure_does_not_block_going_live(
    app: BotApp, rt: Runtime, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await add_topic(rt, user_gw, channel=CHANNEL)
    user_gw.fail_next("create_channel", FloodWait(3600))
    reply = await say(bot_gw, "/go")
    assert rt.settings.publishing.live is True
    assert rt.settings.publishing.staging_channel == 0
    assert "media channel could not be set up" in reply
    assert "text with a link to the original" in reply
