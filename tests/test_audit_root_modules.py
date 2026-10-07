"""Regression tests for the spec-audit gaps closed in the root modules: ``curator run --live``
confirms like ``/go`` (S3), the preview reads hand edits of the file (TUN-4), and a session
that expired while the service was down is announced (RD-2)."""

# The fixtures imported from the other test modules are parameters here, which ruff reads as
# redefinitions.
# ruff: noqa: F811

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import tomlkit

from tests.fakes import OWNER_ID, FakeBotGateway, FakeClock, FakeUserGateway
from tests.test_bot_control import hand_edit
from tests.test_cli import (  # noqa: F401 - fixtures re-used
    SOURCE,
    configured,
    curator,
    fakes,
    seed,
)
from tests.test_cli import info as source_info
from tests.test_service import (  # noqa: F401 - fixtures re-used
    OUTPUT,
    doubles,
    info,
    parked,
    seed_owned_channels,
    wired,
)
from tg_curator import control, service
from tg_curator.bot.core import BotApp
from tg_curator.config import ENV_OVERRIDES
from tg_curator.db.store import Store
from tg_curator.domain import KV, NewPost, PostStatus, Topic
from tg_curator.ml import models
from tg_curator.runtime import EVENT_ACCOUNT_BOUND, Runtime
from tg_curator.service import Doubles, Service
from tg_curator.textutil import text_hash

# --- S3: `curator run --live` checks and confirms like /go -------------------------------------


async def go_live(wired: tuple[Runtime, BotApp], capsys: pytest.CaptureFixture[str]) -> str:
    rt, app = wired
    await seed_owned_channels(rt)
    await rt.settings_file.set_value("publishing.live", True)  # what _run_locked wrote
    svc = Service(rt, app, sleep=parked, went_live=True, live_requested=True)
    capsys.readouterr()
    await svc.start()
    try:
        return capsys.readouterr().out
    finally:
        await svc.shutdown()


async def test_run_live_without_a_bound_account_names_the_login_step(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, capsys: pytest.CaptureFixture[str]
) -> None:
    user_gw.authorised = False
    out = await go_live(wired, capsys)
    assert "live, but nothing can be posted yet: no account is bound" in out
    assert "curator login" in out and "/bind" in out
    assert "/pause" not in out


async def test_run_live_with_only_channel_less_topics_names_the_topics_step(
    wired: tuple[Runtime, BotApp], capsys: pytest.CaptureFixture[str]
) -> None:
    out = await go_live(wired, capsys)  # the template's example topics have channel = 0
    assert "no topic has a channel the bot can post into" in out
    assert "curator topics add NAME --create-channel" in out
    assert "The first digest arrives" not in out


async def test_run_live_with_a_postable_topic_confirms_the_schedule_and_the_pause(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, capsys: pytest.CaptureFixture[str]
) -> None:
    rt, _ = wired
    user_gw.add_chat(info(OUTPUT, "ML & AI"))

    def link(doc: Any) -> None:
        doc["topics"][0]["channel"] = OUTPUT

    await rt.settings_file.update(link)
    out = await go_live(wired, capsys)
    assert "publishing is live" in out
    assert "The first digest arrives" in out and "(UTC)" in out
    assert "weekly review comes on" in out
    assert "stop posting with /pause in the bot" in out
    assert "nothing can be posted" not in out


async def test_a_service_already_live_reports_only_what_keeps_it_from_posting(
    wired: tuple[Runtime, BotApp], capsys: pytest.CaptureFixture[str]
) -> None:
    rt, app = wired
    await seed_owned_channels(rt)
    await rt.settings_file.set_value("publishing.live", True)
    svc = Service(rt, app, sleep=parked)  # a plain restart, no --live
    capsys.readouterr()
    await svc.start()
    try:
        assert "no topic has a channel the bot can post into" in capsys.readouterr().out
    finally:
        await svc.shutdown()


# --- TUN-4: the preview replays the file as edited ----------------------------------------------

POST_TEXT = "A new open model beats the benchmark on reasoning tasks"


def only_ml_topic(strictness: float) -> Any:
    def edit(doc: Any) -> None:
        topics = tomlkit.aot()
        entry = tomlkit.table()
        entry.update({"key": "ml-ai", "name": "ML", "category": "tech", "strictness": strictness})
        topics.append(entry)
        doc["topics"] = topics

    return edit


def test_standalone_preview_uses_the_files_topic_strictness(
    configured: Path, clock: FakeClock, fakes: Doubles, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the CLI tests' own ``home`` fixture sets up (this module's ``home`` is conftest's).
    for var in (*ENV_OVERRIDES, service.WAIT_ENV, "TG_CURATOR_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(models.FAKE_MODELS_ENV, "1")
    monkeypatch.setattr(models, "_embedder", None)

    async def work(store: Store) -> None:
        await store.upsert_chat(source_info(SOURCE, "Tech news"))
        await store.upsert_topic(
            Topic(id=0, key="ml-ai", name="ML", category="tech", created_at=clock.now())
        )
        await store.insert_post(
            NewPost(
                chat_id=SOURCE,
                message_id=1,
                kind="post",
                message_ids=[1],
                posted_at=clock.now() - timedelta(hours=1),
                via="live",
                text=POST_TEXT,
                text_hash=text_hash(POST_TEXT),
                urls=[],
            ),
            status=PostStatus.unsorted,
        )

    seed(configured, clock, work)
    path = configured / "settings.toml"

    def write(strictness: float) -> None:
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        only_ml_topic(strictness)(doc)
        doc["sorting"]["confidence"] = 0.05
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    write(0.0)  # the file matches the topic row
    before = curator(configured, "preview")
    assert before.exit_code == 0, before.output
    assert "ml-ai: 1" in before.output and "unsorted: 0" in before.output
    assert "note:" not in before.output

    write(0.99)  # the hand edit the next start would apply
    after = curator(configured, "preview")
    assert after.exit_code == 0, after.output
    assert "ml-ai: 0" in after.output and "unsorted: 1" in after.output
    assert "topics.ml-ai.strictness" in after.output.splitlines()[0]


async def test_preview_in_the_running_service_takes_the_hand_edits_and_says_so(
    wired: tuple[Runtime, BotApp], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, _ = wired
    await rt.store.start()
    await rt.store.upsert_topic(
        Topic(id=0, key="ml-ai", name="ML", category="tech", created_at=clock.now())
    )
    await rt.store.upsert_chat(source_info(SOURCE, "Tech news", username="technews"))
    seen: dict[str, Any] = {}

    async def replay(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return await real(**kwargs)

    assert rt.preview is not None
    real = rt.preview.replay
    monkeypatch.setattr(rt.preview, "replay", replay)

    def edit(doc: Any) -> None:
        only_ml_topic(0.7)(doc)
        topic = doc["topics"][0]
        topic["name"] = "ML"
        doc["topics"].append(tomlkit.table())
        doc["topics"][1].update({"key": "fintech", "name": "Fintech"})
        doc["sorting"]["confidence"] = 0.42
        doc["sources"] = tomlkit.aot()
        doc["sources"].append(tomlkit.table())
        doc["sources"][0].update({"chat": "@technews", "trust": 3})

    hand_edit(rt, edit)  # no /reload: rt.settings still has the old values
    lines: list[str] = []

    async def out(line: str) -> None:
        lines.append(line)

    code = await control.preview_command(rt, {"set": {"digest.hour": 7}}, out)
    assert code == control.EXIT_OK
    assert seen["overrides"] == {
        "sorting.confidence": 0.42,
        "topics.ml-ai.strictness": 0.7,
        f"sources.{SOURCE}.trust": 3,  # a known chat's trust is replayed (TUN-6 override)
        "digest.hour": 7,
    }
    assert "sorting.confidence" in lines[0] and "/reload" in lines[0]
    assert f"sources.{SOURCE}.trust" in lines[0]
    assert "new topic fintech" in lines[1] and "trust of source" not in lines[1]
    assert lines[2].startswith("preview: last 3 days")
    assert rt.settings.sorting.confidence != 0.42  # nothing applied to the running service

    lines.clear()
    await control.preview_command(rt, {"set": {"topics.ml-ai.strictness": 0.1}}, out)
    assert seen["overrides"]["topics.ml-ai.strictness"] == 0.1  # --set wins over the file


async def test_hand_edits_a_replay_cannot_show_are_named_not_passed_to_the_replay(
    wired: tuple[Runtime, BotApp], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # apply_overrides refuses intake, review, LLM, ... keys; taking them from the file as
    # overrides made every `curator preview` fail once one of them was hand-edited.
    rt, _ = wired
    await rt.store.start()
    seen: dict[str, Any] = {}
    assert rt.preview is not None
    real = rt.preview.replay

    async def replay(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(rt.preview, "replay", replay)

    def edit(doc: Any) -> None:
        doc["groups"]["unit_gap_minutes"] = 9
        doc["review"]["hour"] = 9
        doc["sorting"]["confidence"] = 0.42

    hand_edit(rt, edit)
    lines: list[str] = []

    async def out(line: str) -> None:
        lines.append(line)

    assert await control.preview_command(rt, {}, out) == control.EXIT_OK
    assert seen["overrides"] == {"sorting.confidence": 0.42}
    assert "sorting.confidence" in lines[0]
    assert "groups.unit_gap_minutes" in lines[1] and "review.hour" in lines[1]
    assert "cannot show" in lines[1]
    assert any(line.startswith("preview: last 3 days") for line in lines)


# --- RD-2: a session that expired while the service was down is announced ----------------------


async def start_expired(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway
) -> tuple[Runtime, Service]:
    rt, app = wired
    await seed_owned_channels(rt)
    user_gw.authorised = False
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    return rt, svc


def lost_warnings(bot_gw: FakeBotGateway) -> list[str]:
    return [m.text for m in bot_gw.sent(OWNER_ID) if "session is lost" in m.text]


async def test_a_session_expired_while_down_warns_the_owner_once_until_the_next_bind(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    rt, app = wired
    await rt.settings_file.set_value("telegram.owner_id", OWNER_ID)
    await rt.store.start()
    await rt.store.kv_set(KV.ACCOUNT_ID, 4242)  # it was bound before the service stopped

    rt, svc = await start_expired(wired, user_gw)
    try:
        assert svc.setup_mode
        assert len(lost_warnings(bot_gw)) == 1
        assert await rt.store.kv_get(KV.ACCOUNT_LOSS_NOTIFIED) is not None
    finally:
        await svc.shutdown()

    again = Service(rt, app, sleep=parked)  # restarted with no /bind in between
    await rt.store.start()
    await again.start()
    try:
        assert len(lost_warnings(bot_gw)) == 1
        user_gw.authorised = True
        await rt.events.emit(EVENT_ACCOUNT_BOUND)
        assert await rt.store.kv_get(KV.ACCOUNT_LOSS_NOTIFIED) is None  # the next loss warns
    finally:
        await again.shutdown()


async def test_a_fresh_install_in_setup_mode_sends_no_session_warning(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    rt, _ = wired
    await rt.settings_file.set_value("telegram.owner_id", OWNER_ID)
    rt, svc = await start_expired(wired, user_gw)
    try:
        assert svc.setup_mode
        assert lost_warnings(bot_gw) == []
    finally:
        await svc.shutdown()


async def test_a_loss_at_runtime_is_not_announced_again_at_the_next_start(
    wired: tuple[Runtime, BotApp], user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    rt, app = wired
    await rt.settings_file.set_value("telegram.owner_id", OWNER_ID)
    await seed_owned_channels(rt)
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        await rt.store.kv_set(KV.ACCOUNT_ID, 4242)
        await user_gw.lose_session("revoked")
        assert len(lost_warnings(bot_gw)) == 1  # the account service's own warning
        assert await rt.store.kv_get(KV.ACCOUNT_LOSS_NOTIFIED) is not None
    finally:
        await svc.shutdown()

    again = Service(rt, app, sleep=parked)
    await rt.store.start()
    await again.start()
    try:
        assert len(lost_warnings(bot_gw)) == 1
    finally:
        await again.shutdown()
