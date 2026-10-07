"""The gateway contract (§5), the service contracts (§8) and the runtime (§8): the fakes match
the Protocols signature for signature, the dataclasses are frozen, the bus behaves."""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.fakes import FakeBotGateway, FakeUserGateway
from tg_curator import contracts
from tg_curator.config import SettingsFile
from tg_curator.notify import Notifier
from tg_curator.runtime import (
    EVENT_EXAMPLES_CHANGED,
    EVENT_SETTINGS_CHANGED,
    EVENTS,
    EventBus,
    Runtime,
)
from tg_curator.telegram import gateway
from tg_curator.telegram.gateway import (
    Account,
    BotCallback,
    BotGateway,
    BotMessage,
    Button,
    Buttons,
    ChatInfo,
    IncomingMessage,
    UserGateway,
)

# --- the gateway protocols -------------------------------------------------------------------

USER_METHODS = {
    "connect", "disconnect", "reset_session", "begin_relogin", "cancel_relogin", "me",
    "send_code", "resend_code", "sign_in",
    "sign_in_password", "ping", "ensure_connected", "on_message", "on_session_lost",
    "list_chats", "resolve_chat",
    "history", "get_views", "find_message", "find_media", "create_channel", "rename_channel",
    "add_bot_admin",
    "copy_media", "forward", "register_owned", "mute", "set_archived", "leave",
    "list_folders", "get_folder", "register_own_folder", "save_folder", "delete_folder",
}  # fmt: skip
BOT_METHODS = {
    "start", "stop", "ensure_connected", "on_message", "on_callback", "send_text", "send_copy",
    "edit_text",
    "edit_buttons", "delete_message", "answer_callback", "can_post",
}  # fmt: skip
FORBIDDEN_ON_THE_ACCOUNT = {
    "send_message", "send_text", "join", "join_channel", "import_invite", "react",
    "mark_read", "read", "delete_message", "delete_messages", "delete_dialog", "set_profile",
}  # fmt: skip


def _protocol_methods(proto: type) -> dict[str, Callable[..., Any]]:
    return {
        name: obj for name, obj in vars(proto).items() if callable(obj) and not name.startswith("_")
    }


def test_user_gateway_has_exactly_the_contract_methods() -> None:
    assert set(_protocol_methods(UserGateway)) == USER_METHODS
    assert not USER_METHODS & FORBIDDEN_ON_THE_ACCOUNT


def test_bot_gateway_has_exactly_the_contract_methods() -> None:
    assert set(_protocol_methods(BotGateway)) == BOT_METHODS


@pytest.mark.parametrize(
    ("proto", "fake"), [(UserGateway, FakeUserGateway), (BotGateway, FakeBotGateway)]
)
def test_fakes_match_every_signature(proto: type, fake: type) -> None:
    for name, method in _protocol_methods(proto).items():
        impl = getattr(fake, name)
        want, have = inspect.signature(method), inspect.signature(impl)
        assert list(want.parameters) == list(have.parameters), name
        for pname, p in want.parameters.items():
            q = have.parameters[pname]
            assert (p.kind, p.default) == (q.kind, q.default), f"{name}({pname})"
        assert inspect.iscoroutinefunction(method) == inspect.iscoroutinefunction(impl), name


def test_gateway_module_does_not_import_telethon() -> None:
    tree = ast.parse(inspect.getsource(gateway))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.split(".")[0] == "telethon" for name in imported), imported


# --- dataclasses -----------------------------------------------------------------------------


def test_dataclasses_are_frozen_with_the_contract_fields() -> None:
    chat = ChatInfo(
        id=-1001, kind="channel", title="T", username=None, noforwards=False,
        is_creator=False, is_admin=False, archived=False, muted_until=None,
    )  # fmt: skip
    with pytest.raises(FrozenInstanceError):
        chat.title = "x"  # type: ignore[misc]
    assert [f.name for f in fields(ChatInfo)] == [
        "id", "kind", "title", "username", "noforwards", "is_creator", "is_admin",
        "archived", "muted_until",
    ]  # fmt: skip
    assert [f.name for f in fields(Account)] == ["id", "name", "username", "phone"]
    assert [f.name for f in fields(IncomingMessage)] == [
        "chat", "message_id", "date", "text", "html", "sender_id", "is_outgoing", "is_service",
        "reply_to_id", "grouped_id", "media", "urls", "views", "forwards", "fwd_from_chat_id",
        "fwd_from_message_id", "noforwards", "topic_id", "is_automatic_forward",
    ]  # fmt: skip
    assert [f.name for f in fields(BotMessage)] == [
        "chat_id", "message_id", "sender_id", "text", "is_private", "fwd_from_chat_id",
        "fwd_from_title", "reply_to_id", "has_media",
    ]  # fmt: skip
    assert [f.name for f in fields(BotCallback)] == [
        "query_id", "sender_id", "chat_id", "message_id", "data",
    ]  # fmt: skip
    button = Button("Approve", data="rv:1:approve")
    assert button.url is None
    with pytest.raises(FrozenInstanceError):
        button.text = "x"  # type: ignore[misc]
    rows: Buttons = [[button, Button("Docs", url="https://example.org")]]
    assert Buttons == list[list[Button]] and isinstance(rows[0][1].url, str)


# --- the service contracts -------------------------------------------------------------------

SERVICE_METHODS = {
    contracts.Embedder: {"embed"},
    contracts.TopicClassifier: {"reload", "predict", "learn", "category_scores"},
    contracts.LLM: {"summarise_line", "name_topic", "second_opinion"},
    contracts.Notifier: {
        "owner",
        "digest_line",
        "proposal",
        "intake_warning",
        "cannot_post",
        "llm_cap_reached",
        "llm_out_of_credit",
        "review_absorbed",
        "topic_created",
    },  # fmt: skip
    contracts.Intake: {
        "handle_message",
        "collect",
        "close_units",
        "tick",
        "sync_chats",
        "flush_albums",
    },  # fmt: skip
    contracts.Backfill: {"run"},
    contracts.Sorter: {"submit", "tick", "resort_unsorted"},
    contracts.PreviewService: {"replay"},
    contracts.Publisher: {"enqueue", "tick", "reconcile", "note_corroboration", "move"},
    contracts.DigestService: {"preview", "send", "tick", "reconcile", "next_run"},
    contracts.TopicsService: {
        "sync_from_settings",
        "link_channel",
        "create",
        "channel_wait_minutes",
        "create_topic_channel",
        "want_channel",
        "tick",
        "update",
        "remove",
        "merge",
        "add_examples",
        "add_example_channel",
        "ensure_staging_channel",
    },  # fmt: skip
    contracts.Learning: {"correct", "retrain"},
    contracts.StatsService: {"chat_stats", "topic_stats"},
    contracts.ReviewService: {"build", "send", "decide", "undo", "tick"},
    contracts.ActionExecutor: {"tick", "undo"},
    contracts.FolderManager: {"sync"},
    contracts.Discovery: {"propose", "accept"},
    contracts.AccountService: {"status", "begin", "resend", "code", "password"},
}


@pytest.mark.parametrize(
    ("proto", "methods"), SERVICE_METHODS.items(), ids=lambda p: getattr(p, "__name__", "")
)
def test_service_protocols_carry_the_contract_methods(proto: type, methods: set[str]) -> None:
    assert set(_protocol_methods(proto)) == methods
    for name, method in _protocol_methods(proto).items():
        assert method.__doc__ or name in {"status", "stop"}, (
            f"{proto.__name__}.{name} has no docstring"
        )


def test_the_key_behavioural_rules_are_in_the_docstrings() -> None:
    assert "never" in contracts.Sorter.resort_unsorted.__doc__.lower()
    assert "ONLY by" in contracts.Publisher.move.__doc__
    assert "Idempotent" in contracts.Publisher.enqueue.__doc__
    assert "seq = COALESCE(MAX(seq), 0) + 1" in contracts.DigestService.send.__doc__
    assert "TopicExists" in contracts.TopicsService.create.__doc__
    assert "ONLY from" in contracts.Discovery.propose.__doc__
    assert "increment=True" in UserGateway.get_views.__doc__
    assert "never joins" in UserGateway.resolve_chat.__doc__.lower()
    assert "NotOwnedError" in UserGateway.__doc__ and 'NotAllowed("creator")' in UserGateway.__doc__
    assert "Telegram forbids reply markup on grouped media" in BotGateway.send_copy.__doc__


# --- the event bus ---------------------------------------------------------------------------


async def test_bus_runs_handlers_in_order_with_the_payload() -> None:
    bus = EventBus()
    seen: list[tuple[str, dict[str, Any]]] = []

    async def first(**payload: Any) -> None:
        seen.append(("first", payload))

    async def second(**payload: Any) -> None:
        await asyncio.sleep(0)
        seen.append(("second", payload))

    bus.on(EVENT_EXAMPLES_CHANGED, first)
    bus.on(EVENT_EXAMPLES_CHANGED, second)
    await bus.emit(EVENT_EXAMPLES_CHANGED, reason="correction")
    assert seen == [("first", {"reason": "correction"}), ("second", {"reason": "correction"})]
    await bus.emit(EVENT_SETTINGS_CHANGED)  # no handlers: fine
    assert bus.handlers(EVENT_SETTINGS_CHANGED) == []


async def test_bus_swallows_handler_errors_and_keeps_going(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    ran: list[str] = []

    async def broken(**_: Any) -> None:
        raise RuntimeError("boom")

    async def fine(**_: Any) -> None:
        ran.append("fine")

    bus.on(EVENT_SETTINGS_CHANGED, broken)
    bus.on(EVENT_SETTINGS_CHANGED, fine)
    with caplog.at_level(logging.ERROR, logger="tg_curator.runtime"):
        await bus.emit(EVENT_SETTINGS_CHANGED)
    assert ran == ["fine"]
    assert any("broken" in r.getMessage() and "failed" in r.getMessage() for r in caplog.records)


async def test_bus_lets_cancellation_through() -> None:
    bus = EventBus()

    async def cancelled(**_: Any) -> None:
        raise asyncio.CancelledError

    bus.on(EVENT_SETTINGS_CHANGED, cancelled)
    with pytest.raises(asyncio.CancelledError):
        await bus.emit(EVENT_SETTINGS_CHANGED)


async def test_bus_warns_on_unknown_events_and_missing_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    with caplog.at_level(logging.WARNING, logger="tg_curator.runtime"):
        await bus.emit("no_such_event")
        await bus.emit(EVENT_EXAMPLES_CHANGED)
        bus.on("typo_event", _noop)
    messages = [r.getMessage() for r in caplog.records]
    assert any("unknown event 'no_such_event'" in m for m in messages)
    assert any("examples_changed without a valid reason" in m for m in messages)
    assert any("unknown event 'typo_event'" in m for m in messages)
    assert EVENTS == {
        "settings_changed", "topics_changed", "account_bound", "session_lost", "examples_changed",
    }  # fmt: skip


async def _noop(**_: Any) -> None:
    pass


# --- the runtime -----------------------------------------------------------------------------


async def test_runtime_wiring(
    make_runtime: Callable[..., Any], settings_file: SettingsFile
) -> None:
    rt: Runtime = await make_runtime()
    assert rt.settings is settings_file.current
    assert rt.settings.telegram.owner_id == 1001
    assert rt.bot_account is not None and rt.bot_account.username == "curator_test_bot"
    assert isinstance(rt.notifier, Notifier) and isinstance(rt.notifier, contracts.Notifier)
    assert rt.health == {}
    for name in (
        "intake", "backfill", "sorter", "preview", "publisher", "digest", "topics", "learning",
        "stats", "review", "actions", "folders", "discovery", "account",
    ):  # fmt: skip
        assert getattr(rt, name) is None, name
    assert isinstance(rt.clock.now(), datetime) and rt.clock.now().tzinfo is UTC
    assert rt.t("yes") == "Yes"


async def test_rt_fixture_has_a_started_store(rt: Runtime) -> None:
    """The full fixture chain: temp home, template settings, fakes and a started Store."""
    assert (rt.home / "curator.db").is_file()
    await rt.store.kv_set("smoke", {"ok": True})
    assert await rt.store.kv_get("smoke") == {"ok": True}
    assert await rt.store.kv_get("missing", default=0) == 0


async def test_settings_writes_emit_settings_changed(make_runtime: Callable[..., Any]) -> None:
    rt: Runtime = await make_runtime()
    fired: list[dict[str, Any]] = []

    async def handler(**payload: Any) -> None:
        fired.append(payload)

    rt.events.on(EVENT_SETTINGS_CHANGED, handler)
    await rt.settings_file.set_value("digest.hour", 20)
    assert fired == [{}] and rt.settings.digest.hour == 20
    await rt.settings_file.load()  # a plain re-read does not fire (a /reload emits itself)
    assert fired == [{}]
