"""notify.py: the six notification kinds, the no-op cases, the one-per-second serialisation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import tomllib
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import pytest

from tests.fakes import OWNER_ID, FakeBotGateway
from tg_curator import contracts, notify
from tg_curator.errors import BotCannotPost
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.notify import AbsorbedLine, Notifier
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import Button

CATALOGUE_KEYS = {
    "notify_digest",
    "notify_proposal_folder",
    "notify_proposal_mute",
    "notify_proposal_archive",
    "notify_proposal_leave",
    "notify_proposal_new_topic",
    "notify_proposal_merge_topics",
    "notify_review_absorbed",
    "notify_intake_stalled",
    "notify_session_lost",
    "notify_cannot_post",
    "notify_llm_cap_reached",
    "notify_llm_out_of_credit",
    "notify_topic_created",
    "notify_topic_created_wait",
    "notify_topic_created_no_channel",
}


@pytest.fixture
async def rt(make_runtime: Callable[..., Any]) -> Runtime:
    """A Runtime without a database: the notifier never touches the store."""
    return await make_runtime()


# --- the catalogue ---------------------------------------------------------------------------


def test_every_notification_string_ships_in_english() -> None:
    t = Translator(locales_dir=LOCALES_DIR)
    assert CATALOGUE_KEYS <= t.english_keys()
    assert (LOCALES_DIR / "en" / "notify.toml").is_file()


def test_every_key_used_by_the_module_exists_and_nothing_else_is_shipped() -> None:
    """§11.5: a bot module never uses a key missing from English — read off the source, so a
    key added to notify.py without a string (or a string nobody uses) fails here."""
    used = set(re.findall(r'"(notify_[a-z_]+)"', inspect.getsource(notify)))
    assert used == CATALOGUE_KEYS
    with (LOCALES_DIR / "en" / "notify.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert shipped == CATALOGUE_KEYS


def test_only_the_six_kinds_exist() -> None:
    public = {n for n, v in vars(Notifier).items() if callable(v) and not n.startswith("_")}
    assert public == {
        "owner", "digest_line", "proposal", "review_absorbed", "intake_warning", "cannot_post",
        "llm_cap_reached", "llm_out_of_credit", "topic_created",
    }  # fmt: skip
    assert Notifier.MIN_GAP_SECONDS == 1.0


async def test_notifier_satisfies_the_protocol(rt: Runtime) -> None:
    assert isinstance(rt.notifier, contracts.Notifier)


# --- no-op cases -----------------------------------------------------------------------------


async def test_noop_without_owner(
    make_runtime: Callable[..., Any], bot_gw: FakeBotGateway, caplog: pytest.LogCaptureFixture
) -> None:
    rt: Runtime = await make_runtime(owner_id=0)
    with caplog.at_level(logging.DEBUG, logger="tg_curator.notify"):
        assert await rt.notifier.owner("<b>hello</b>") is None
        assert await rt.notifier.digest_line("ML & AI: 3 posts") is None
    assert bot_gw.calls_of("send_text") == []
    assert any("skipped (no owner)" in r.getMessage() for r in caplog.records)


async def test_noop_without_bot(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    rt.bot = None
    assert await rt.notifier.owner("hello") is None
    assert bot_gw.calls_of("send_text") == []


async def test_noop_before_settings_are_loaded(
    make_runtime: Callable[..., Any], bot_gw: FakeBotGateway
) -> None:
    rt: Runtime = await make_runtime()
    rt.settings_file._current = None  # as before load(): .current raises ConfigError
    assert await rt.notifier.owner("hello") is None
    assert bot_gw.calls_of("send_text") == []


# --- sending ---------------------------------------------------------------------------------


async def test_owner_sends_to_the_private_chat(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    buttons = [[Button("Approve", data="rv:1:approve")]]
    mid = await rt.notifier.owner("<b>hi</b>", buttons=buttons)
    assert mid == 1
    [sent] = bot_gw.sent(OWNER_ID)
    assert sent.html == "<b>hi</b>" and sent.buttons == buttons and sent.sender == "bot"


async def test_failures_are_swallowed(
    rt: Runtime, bot_gw: FakeBotGateway, caplog: pytest.LogCaptureFixture
) -> None:
    bot_gw.fail_next("send_text", BotCannotPost("blocked"))
    with caplog.at_level(logging.WARNING, logger="tg_curator.notify"):
        assert await rt.notifier.owner("x") is None
    assert any("not delivered" in r.getMessage() for r in caplog.records)
    assert await rt.notifier.owner("y") == 1


async def test_sends_are_serialised_one_per_gap(
    rt: Runtime, bot_gw: FakeBotGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Notifier, "MIN_GAP_SECONDS", 0.2)
    loop = asyncio.get_running_loop()
    started = loop.time()
    ids = await asyncio.gather(
        rt.notifier.owner("one"), rt.notifier.owner("two"), rt.notifier.owner("three")
    )
    elapsed = loop.time() - started
    assert ids == [1, 2, 3]
    assert elapsed >= 0.38, elapsed  # two gaps between three sends
    assert [m.text for m in bot_gw.sent(OWNER_ID)] == ["one", "two", "three"]


# --- the six kinds ---------------------------------------------------------------------------


async def test_digest_line_escapes_html(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    await rt.notifier.digest_line("ML & AI: 15 posts; Fintech: 9")
    [sent] = bot_gw.sent(OWNER_ID)
    assert "ML &amp; AI: 15 posts; Fintech: 9" in sent.html
    assert sent.text == "Digest sent — ML & AI: 15 posts; Fintech: 9"


@pytest.mark.parametrize(
    "kind", ["folder", "mute", "archive", "leave", "new_topic", "merge_topics"]
)
async def test_proposal_per_kind(rt: Runtime, bot_gw: FakeBotGateway, kind: str) -> None:
    buttons = [[Button("Approve", data="rv:7:approve"), Button("Skip", data="rv:7:skip")]]
    mid = await rt.notifier.proposal(
        kind, "News <Uz>", "41 posts in 30 days, 0 reached a topic", buttons=buttons
    )
    assert mid == 1
    [sent] = bot_gw.sent(OWNER_ID)
    assert sent.buttons == buttons
    assert "News &lt;Uz&gt;" in sent.html and "41 posts in 30 days" in sent.text
    assert not sent.html.startswith("notify_")  # the key resolved to a string
    if kind == "leave":
        assert "cannot be undone" in sent.text


async def test_proposal_details_are_appended_as_html(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    details = '1. <a href="https://t.me/c/1/2">example</a>'
    await rt.notifier.proposal("new_topic", "Crypto", "40 unsorted posts", details=details)
    [sent] = bot_gw.sent(OWNER_ID)
    assert sent.html.endswith("\n\n" + details) and sent.hrefs == ["https://t.me/c/1/2"]


async def test_intake_warnings(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    await rt.notifier.intake_warning("stalled")
    await rt.notifier.intake_warning("session_lost")
    stalled, lost = bot_gw.sent(OWNER_ID)
    assert "over an hour" in stalled.text and "/bind" in lost.text and stalled.text != lost.text


async def test_cannot_post_llm_cap_and_topic_created(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    await rt.notifier.cannot_post("ML & AI")
    await rt.notifier.llm_cap_reached("2026-10", 5)
    await rt.notifier.topic_created("Crypto & exchanges")
    cannot, cap, created = bot_gw.sent(OWNER_ID)
    assert "<b>ML &amp; AI</b>" in cannot.html and "Post Messages" in cannot.text
    assert "$5.00" in cap.text and "2026-10" in cap.text
    assert "<b>Crypto &amp; exchanges</b>" in created.html and "/topics" in created.text


async def test_review_absorbed_is_one_message_with_a_line_per_topic(
    rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await rt.notifier.review_absorbed(
        [
            AbsorbedLine("Crypto & exchanges", 34, 51, date(2026, 9, 28)),
            AbsorbedLine("Real estate", 0, 12, date(2026, 10, 1)),
        ]
    )
    [sent] = bot_gw.sent(OWNER_ID)
    assert sent.text.splitlines() == [
        "📥 Crypto & exchanges absorbed 34 of 51 unsorted posts since Sep 28",
        "📥 Real estate absorbed 0 of 12 unsorted posts since Oct 1",
    ]
    assert "<b>Crypto &amp; exchanges</b>" in sent.html


async def test_llm_out_of_credit_names_the_provider_and_the_month_end(
    rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    rt.clock.set(datetime(2026, 2, 10, 9, 0, tzinfo=UTC))  # type: ignore[attr-defined]
    await rt.notifier.llm_out_of_credit("OpenAI")
    [sent] = bot_gw.sent(OWNER_ID)
    assert sent.text.startswith("OpenAI reports the account is out of credit")
    assert "until Feb 28" in sent.text and "/llm" in sent.text


async def test_llm_out_of_credit_month_end_is_local(rt: Runtime, bot_gw: FakeBotGateway) -> None:
    await rt.settings_file.set_value("general.timezone", "Asia/Tashkent")
    rt.clock.set(datetime(2026, 10, 31, 20, 0, tzinfo=UTC))  # type: ignore[attr-defined]
    await rt.notifier.llm_out_of_credit("Anthropic")  # already November 1st in Tashkent
    [sent] = bot_gw.sent(OWNER_ID)
    assert "until Nov 30" in sent.text
