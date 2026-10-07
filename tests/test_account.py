"""account.py: the login state machine, LoginError mapping, the watchdog and session loss."""

from __future__ import annotations

import inspect
import re
import tomllib
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from tests.fakes import (
    OWNER_ID,
    USER_ACCOUNT,
    FakeBotGateway,
    FakeClock,
    FakeUserGateway,
    plain_text,
)
from tg_curator import account as account_module
from tg_curator import contracts
from tg_curator.account import RELOGIN_TTL, STALL_AFTER, AccountService
from tg_curator.db import schema
from tg_curator.db.store import Store
from tg_curator.domain import KV
from tg_curator.errors import FloodWait, LoginError, TelegramUnavailable
from tg_curator.i18n import LOCALES_DIR
from tg_curator.runtime import EVENT_ACCOUNT_BOUND, EVENT_SESSION_LOST, Runtime
from tg_curator.telegram.gateway import Account

PHONE = "+998901234567"
CODE = "12345"
PASSWORD = "hunter2-secret"
OTHER = Account(id=2002, name="Second Owner", username="second", phone="+998901112233")


@pytest.fixture
def events(rt: Runtime) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []

    async def bound(**payload: Any) -> None:
        seen.append((EVENT_ACCOUNT_BOUND, payload))

    async def lost(**payload: Any) -> None:
        seen.append((EVENT_SESSION_LOST, payload))

    rt.events.on(EVENT_ACCOUNT_BOUND, bound)
    rt.events.on(EVENT_SESSION_LOST, lost)
    return seen


@pytest.fixture
def unbound(user_gw: FakeUserGateway) -> FakeUserGateway:
    user_gw.authorised = False
    user_gw.valid_code = CODE
    return user_gw


@pytest.fixture
def service(rt: Runtime, unbound: FakeUserGateway) -> AccountService:
    svc = AccountService(rt)
    svc.setup_mode = True  # what service.py sets when the account is not bound at start
    rt.account = svc
    return svc


async def kv_dump(store: Store) -> str:
    rows = await store.execute(sa.select(schema.kv.c.key, schema.kv.c.value))
    return repr(rows)


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


# --- status ----------------------------------------------------------------------------------


async def test_status_before_and_after_binding(service: AccountService, rt: Runtime) -> None:
    assert isinstance(service, contracts.AccountService)
    before = await service.status()
    assert (before.bound, before.account, before.step) == (False, None, "phone")
    await service.begin(PHONE)
    assert (await service.status()).step == "code"
    assert await service.code(CODE) == "ok"
    after = await service.status()
    assert (after.bound, after.account, after.step) == (True, USER_ACCOUNT, "ok")


async def test_status_without_a_user_client(make_runtime: Callable[..., Any]) -> None:
    rt = await make_runtime()
    rt.user = None
    status = await AccountService(rt).status()
    assert (status.bound, status.step) == (False, "none")


# --- the happy paths -------------------------------------------------------------------------


async def test_happy_path_without_2fa(
    service: AccountService,
    rt: Runtime,
    unbound: FakeUserGateway,
    store: Store,
    events: list[tuple[str, dict[str, Any]]],
) -> None:
    await service.begin(PHONE)
    assert unbound.calls_of("connect") and unbound.calls_of("send_code") == [{"phone": PHONE}]
    assert await service.code(CODE) == "ok"
    assert unbound.authorised
    assert await store.kv_get(KV.ACCOUNT_ID) == USER_ACCOUNT.id
    assert events == [(EVENT_ACCOUNT_BOUND, {})]
    assert rt.settings.telegram.owner_id == OWNER_ID  # already set: untouched
    assert CODE not in await kv_dump(store)


async def test_happy_path_with_2fa_and_nothing_secret_persisted(
    service: AccountService,
    unbound: FakeUserGateway,
    store: Store,
    events: list[tuple[str, dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    unbound.password_needed = True
    unbound.valid_password = PASSWORD
    with caplog.at_level("DEBUG"):
        await service.begin(PHONE)
        assert await service.code(CODE) == "password_needed"
        assert (await service.status()).step == "password"
        assert events == []
        await service.password(PASSWORD)
    assert unbound.authorised and events == [(EVENT_ACCOUNT_BOUND, {})]
    dump = await kv_dump(store)
    assert CODE not in dump and PASSWORD not in dump
    assert CODE not in caplog.text and PASSWORD not in caplog.text and PHONE not in caplog.text
    assert "+9989***67" in caplog.text


async def test_binding_sets_the_owner_when_there_is_none(
    make_runtime: Callable[..., Any], store: Store, unbound: FakeUserGateway
) -> None:
    rt = await make_runtime(store, owner_id=0)
    svc = AccountService(rt)
    await svc.begin(PHONE)
    await svc.code(CODE)
    assert rt.settings.telegram.owner_id == USER_ACCOUNT.id


async def test_resend_only_while_a_code_is_pending(
    service: AccountService, unbound: FakeUserGateway, rt: Runtime
) -> None:
    with pytest.raises(LoginError) as info:
        await service.resend()
    assert str(info.value) == rt.t("account_error_no_code_pending")
    await service.begin(PHONE)
    await service.resend()
    assert unbound.codes_sent == [PHONE, PHONE]


async def test_steps_out_of_order_are_refused_with_plain_sentences(
    service: AccountService, rt: Runtime
) -> None:
    with pytest.raises(LoginError) as info:
        await service.code(CODE)
    assert str(info.value) == rt.t("account_error_no_code_pending")
    with pytest.raises(LoginError) as info:
        await service.password(PASSWORD)
    assert str(info.value) == rt.t("account_error_no_password_pending")


# --- LoginError mapping ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "reason"),
    [
        ("send_code", "bad_phone"),
        ("send_code", "flood"),
        ("send_code", "other"),
        ("sign_in", "bad_code"),
        ("sign_in", "expired_code"),
        ("sign_in_password", "bad_password"),
    ],
)
async def test_every_login_error_becomes_its_catalogue_sentence(
    service: AccountService, unbound: FakeUserGateway, rt: Runtime, method: str, reason: str
) -> None:
    unbound.fail_next(method, LoginError(reason, "raw telethon detail"))
    with pytest.raises(LoginError) as info:
        if method == "send_code":
            await service.begin(PHONE)
        elif method == "sign_in":
            await service.begin(PHONE)
            await service.code("0")
        else:
            unbound.password_needed = True
            await service.begin(PHONE)
            await service.code(CODE)
            await service.password("x")
    assert info.value.reason == reason
    expected = rt.t(f"account_error_{reason}", detail="raw telethon detail")
    assert str(info.value) == expected
    assert expected != f"account_error_{reason}"  # the key exists in English
    if reason != "other":
        assert "raw telethon detail" not in str(info.value)


async def test_the_fake_bad_code_path_maps_too(
    service: AccountService, unbound: FakeUserGateway, rt: Runtime
) -> None:
    await service.begin(PHONE)
    with pytest.raises(LoginError) as info:
        await service.code("99999")
    assert info.value.reason == "bad_code"
    assert str(info.value) == rt.t("account_error_bad_code")
    assert (await service.status()).step == "code"  # still waiting for the right one


async def test_flood_wait_becomes_a_flood_login_error(
    service: AccountService, unbound: FakeUserGateway, rt: Runtime
) -> None:
    unbound.fail_next("send_code", FloodWait(500))
    with pytest.raises(LoginError) as info:
        await service.begin(PHONE)
    assert info.value.reason == "flood"
    assert str(info.value) == rt.t("account_error_flood", minutes=9)


# --- the watchdog ----------------------------------------------------------------------------


@pytest.fixture
def bound(rt: Runtime, user_gw: FakeUserGateway) -> AccountService:
    svc = AccountService(rt)
    rt.account = svc
    return svc


async def test_watchdog_warns_once_only_when_quiet_and_ping_fails(
    bound: AccountService,
    rt: Runtime,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
    store: Store,
) -> None:
    await bound.watchdog_tick()  # arms at the first tick
    assert user_gw.calls_of("ping") == [] and owner_texts(bot_gw) == []
    clock.advance(STALL_AFTER - timedelta(minutes=1))
    await bound.watchdog_tick()
    assert user_gw.calls_of("ping") == []  # not quiet long enough: no ping at all
    clock.advance(timedelta(minutes=2))
    await bound.watchdog_tick()
    assert len(user_gw.calls_of("ping")) == 1 and owner_texts(bot_gw) == []  # ping ok: silent
    user_gw.alive = False
    await bound.watchdog_tick()
    assert owner_texts(bot_gw) == [rt.t("notify_intake_stalled")]
    assert await store.kv_get(KV.WATCHDOG_WARNED)
    clock.advance(timedelta(hours=3))
    await bound.watchdog_tick()
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 1  # once


async def test_watchdog_rearms_when_intake_resumes(
    bound: AccountService,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
    store: Store,
) -> None:
    user_gw.alive = False
    await bound.watchdog_tick()
    clock.advance(STALL_AFTER)
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 1
    await store.kv_set(KV.INTAKE_LAST_MESSAGE_AT, clock.now().isoformat())
    await bound.watchdog_tick()
    assert await store.kv_get(KV.WATCHDOG_WARNED) is None  # re-armed
    assert len(owner_texts(bot_gw)) == 1
    clock.advance(STALL_AFTER)
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 2


async def test_watchdog_counts_from_the_last_ingested_message(
    bound: AccountService,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
    store: Store,
) -> None:
    user_gw.alive = False
    await bound.watchdog_tick()
    clock.advance(timedelta(minutes=50))
    await store.kv_set(KV.INTAKE_LAST_MESSAGE_AT, clock.now().isoformat())
    clock.advance(timedelta(minutes=50))
    await bound.watchdog_tick()
    assert owner_texts(bot_gw) == []  # only 50 min since the last message
    clock.advance(timedelta(minutes=11))
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 1


async def test_watchdog_is_not_armed_in_setup_mode(
    bound: AccountService, user_gw: FakeUserGateway, bot_gw: FakeBotGateway, clock: FakeClock
) -> None:
    bound.setup_mode = True
    user_gw.alive = False
    await bound.watchdog_tick()
    clock.advance(STALL_AFTER * 2)
    await bound.watchdog_tick()
    assert user_gw.calls_of("ping") == [] and owner_texts(bot_gw) == []


# --- session loss ----------------------------------------------------------------------------


async def test_session_lost_warns_once_sets_setup_mode_and_emits(
    bound: AccountService,
    rt: Runtime,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    clock: FakeClock,
    events: list[tuple[str, dict[str, Any]]],
) -> None:
    await bound.watchdog_tick()
    await user_gw.lose_session("revoked")
    assert bound.setup_mode is True
    assert owner_texts(bot_gw) == [rt.t("notify_session_lost")]
    assert events == [(EVENT_SESSION_LOST, {"reason": "revoked"})]
    status = await bound.status()
    assert (status.bound, status.step) == (False, "phone")
    # the stalled warning does not pile on top while in setup mode
    clock.advance(STALL_AFTER * 2)
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 1
    # /bind brings it back: setup mode off, watchdog armed afresh
    user_gw.valid_code = CODE
    await bound.begin(PHONE)
    await bound.code(CODE)
    assert bound.setup_mode is False
    assert events[-1] == (EVENT_ACCOUNT_BOUND, {})
    await bound.watchdog_tick()
    assert len(owner_texts(bot_gw)) == 1


# --- binding again (spec "every step can be repeated") -----------------------------------------


async def test_begin_on_a_working_session_logs_in_beside_it(
    bound: AccountService,
    user_gw: FakeUserGateway,
    store: Store,
    events: list[tuple[str, dict[str, Any]]],
) -> None:
    user_gw.valid_code = CODE
    user_gw.relogin_account = OTHER
    await bound.begin(PHONE)
    assert user_gw.calls_of("begin_relogin") and user_gw.calls_of("send_code") == [{"phone": PHONE}]
    assert user_gw.calls_of("reset_session") == [] and user_gw.calls_of("connect") == []
    status = await bound.status()
    assert (status.bound, status.account, status.step) == (True, USER_ACCOUNT, "code")
    assert events == [] and bound.setup_mode is False  # nothing stops while the code is awaited
    with pytest.raises(LoginError):
        await bound.code("99999")
    assert user_gw.account == USER_ACCOUNT and user_gw.session_generation == 1
    assert await bound.code(CODE) == "ok"
    assert user_gw.account == OTHER and user_gw.session_generation == 2
    assert await store.kv_get(KV.ACCOUNT_ID) == OTHER.id
    assert events == [(EVENT_ACCOUNT_BOUND, {})]


async def test_a_refused_code_request_abandons_the_relogin(
    bound: AccountService, user_gw: FakeUserGateway, rt: Runtime
) -> None:
    user_gw.fail_next("send_code", LoginError("bad_phone", "PHONE_NUMBER_INVALID"))
    with pytest.raises(LoginError) as info:
        await bound.begin(PHONE)
    assert str(info.value) == rt.t("account_error_bad_phone")
    assert user_gw.calls_of("cancel_relogin") and user_gw.relogin is None
    status = await bound.status()
    assert (status.bound, status.step) == (True, "ok")


async def test_an_unanswered_session_check_drops_nothing(
    bound: AccountService, user_gw: FakeUserGateway, rt: Runtime
) -> None:
    """Telegram unreachable is not "no working session": the session must not be reset."""
    user_gw.fail_next("me", TelegramUnavailable("Telegram is unavailable"))
    with pytest.raises(LoginError) as info:
        await bound.begin(PHONE)
    assert str(info.value) == rt.t("account_error_unreachable")
    assert user_gw.calls_of("reset_session") == [] and user_gw.calls_of("begin_relogin") == []
    assert user_gw.authorised and bound.setup_mode is False


async def test_an_unfinished_relogin_expires_and_the_account_stays(
    bound: AccountService, user_gw: FakeUserGateway, rt: Runtime, clock: FakeClock
) -> None:
    await bound.begin(PHONE)
    clock.advance(RELOGIN_TTL - timedelta(minutes=1))
    assert (await bound.status()).step == "code"
    clock.advance(timedelta(minutes=1))
    with pytest.raises(LoginError) as info:
        await bound.code(CODE)
    assert str(info.value) == rt.t("account_error_no_code_pending")
    assert user_gw.calls_of("sign_in") == [] and user_gw.relogin is None
    status = await bound.status()
    assert (status.bound, status.account, status.step) == (True, USER_ACCOUNT, "ok")


async def test_a_session_loss_during_a_relogin_keeps_the_pending_code(
    bound: AccountService,
    user_gw: FakeUserGateway,
    events: list[tuple[str, dict[str, Any]]],
) -> None:
    """The working session dies while the owner types the code: finishing the re-login is
    exactly what binds the account again, so the code stays usable."""
    user_gw.valid_code = CODE
    await bound.begin(PHONE)
    await user_gw.lose_session("revoked")
    assert bound.setup_mode is True
    assert (await bound.status()).step == "code"
    assert await bound.code(CODE) == "ok"
    assert bound.setup_mode is False and user_gw.authorised
    assert events == [(EVENT_SESSION_LOST, {"reason": "revoked"}), (EVENT_ACCOUNT_BOUND, {})]


# --- the catalogue ---------------------------------------------------------------------------


def test_every_account_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(account_module)
    used = set(re.findall(r'"(account_[a-z_]+)"', source))
    with (LOCALES_DIR / "en" / "account.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
