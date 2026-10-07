"""Binding again never costs the bound account (spec "every step can be repeated at any time;
nothing is lost by running one twice", "/bind sets it up again").

The service-level tests run the whole curator on the fakes: a re-bind runs beside the working
session, intake keeps delivering through it, and only a fully successful login replaces it.
The gateway-level tests run the same flow through the Telethon gateway on stub clients, where
the files and the second client are real enough to look at.
"""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telethon import errors, functions, types

from tests.fakes import (
    OWNER_ID,
    USER_ACCOUNT,
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeUserGateway,
    make_bot_message,
    plain_text,
)
from tests.test_user_client import StubClient, _rpc, channel, tl_message
from tg_curator.account import RELOGIN_TTL, AccountService
from tg_curator.config import SettingsFile
from tg_curator.db.store import Store
from tg_curator.domain import KV
from tg_curator.errors import LoginError
from tg_curator.runtime import EVENT_SESSION_LOST, Runtime
from tg_curator.service import Doubles, Service, build_runtime
from tg_curator.telegram.gateway import Account, ChatInfo, IncomingMessage
from tg_curator.telegram.user_client import TelethonUserGateway

PHONE = "+998901234567"
CODE = "73915"
SPACED_CODE = "7 3 9 1 5"
PASSWORD = "correct horse battery"
OTHER = Account(id=2002, name="Second Owner", username="second", phone="+998901112233")


async def parked(_: float) -> None:
    """A sleep that never ends: every loop runs its body exactly once."""
    await asyncio.Event().wait()


async def settle(condition: Callable[[], bool], rounds: int = 100) -> None:
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.005)


class Curator:
    """The running service on the fakes, with the owner's chat to the bot."""

    def __init__(
        self, rt: Runtime, svc: Service, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
    ) -> None:
        self.rt = rt
        self.svc = svc
        self.user_gw = user_gw
        self.bot_gw = bot_gw
        self._next_id = 100

    async def say(self, text: str) -> None:
        self._next_id += 1
        await self.bot_gw.say(make_bot_message(text, message_id=self._next_id))

    def last(self) -> str:
        return plain_text(self.bot_gw.sent(OWNER_ID)[-1].html)

    def account_loops_running(self) -> bool:
        return all(self.svc.supervisor.running(name) for name in ("intake", "watchdog", "chats"))

    async def intake_delivers(self, msg: IncomingMessage) -> bool:
        """Push one live message through the account's handlers; True when intake took it."""
        await self.rt.store.kv_delete(KV.INTAKE_LAST_MESSAGE_AT)
        await self.user_gw.deliver(msg)
        return await self.rt.store.kv_get(KV.INTAKE_LAST_MESSAGE_AT) == msg.date.isoformat()


@pytest.fixture
async def curator(
    home: Path,
    settings_file: SettingsFile,
    store: Store,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
) -> Any:
    await settings_file.set_value("telegram.owner_id", OWNER_ID)
    user_gw.add_chat(make_chat())
    user_gw.valid_code = CODE
    doubles = Doubles(
        user=user_gw,
        bot=bot_gw,
        clock=clock,
        embedder=FakeEmbedder(),
        classifier=FakeClassifier(),
        llm=FakeLLM(),
    )
    rt, app = build_runtime(home, settings_file=settings_file, store=store, fakes=doubles)
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    try:
        yield Curator(rt, svc, user_gw, bot_gw)
    finally:
        await svc.shutdown()


@pytest.fixture
def source(user_gw: FakeUserGateway) -> ChatInfo:
    return user_gw.chats[0]


# --- the service on the fakes ------------------------------------------------------------------


async def test_an_abandoned_rebind_keeps_the_account_and_intake(
    curator: Curator,
    source: ChatInfo,
    make_message: Callable[..., IncomingMessage],
    clock: FakeClock,
) -> None:
    rt, user_gw = curator.rt, curator.user_gw
    lost: list[dict[str, Any]] = []

    async def on_lost(**payload: Any) -> None:
        lost.append(payload)

    rt.events.on(EVENT_SESSION_LOST, on_lost)
    assert curator.account_loops_running()
    await curator.say("/bind")
    await curator.say(PHONE)
    assert "space between every digit" in curator.last()
    assert user_gw.calls_of("begin_relogin") and user_gw.relogin is not None
    assert user_gw.calls_of("reset_session") == [] and lost == []
    assert user_gw.authorised and not curator.svc.setup_mode
    assert curator.account_loops_running()  # the loops keep running on the old session
    assert await curator.intake_delivers(make_message(source))  # ... and so does intake

    await curator.say("/help")  # the owner gives up: any command ends the flow
    status = await rt.account.status()  # type: ignore[union-attr]
    assert (status.bound, status.account) == (True, USER_ACCOUNT)
    assert await curator.intake_delivers(make_message(source))
    assert user_gw.session_generation == 1  # the old session was never replaced

    # nobody comes back for it: the watchdog loop closes the second client in the end
    clock.advance(RELOGIN_TTL)
    await rt.account.watchdog_tick()  # type: ignore[union-attr]
    assert user_gw.calls_of("cancel_relogin") and user_gw.relogin is None
    assert user_gw.authorised and curator.account_loops_running() and lost == []


async def test_a_rebind_with_a_wrong_code_leaves_the_old_session_intact(
    curator: Curator, source: ChatInfo, make_message: Callable[..., IncomingMessage]
) -> None:
    user_gw = curator.user_gw
    await curator.say("/bind")
    await curator.say(PHONE)
    await curator.say("1 1 1 1 1")
    assert "probably sent without spaces" in curator.last()
    assert user_gw.calls_of("sign_in") == [{"code": "11111"}]
    assert user_gw.authorised and user_gw.account == USER_ACCOUNT
    assert user_gw.session_generation == 1
    assert curator.account_loops_running() and not curator.svc.setup_mode
    assert await curator.intake_delivers(make_message(source))
    # a second wrong try, then a new /bind: the earlier pending login is abandoned first
    await curator.say("2 2 2 2 2")
    await curator.say("/bind")
    await curator.say(PHONE)
    assert len(user_gw.calls_of("begin_relogin")) == 2
    assert user_gw.calls_of("cancel_relogin")
    assert user_gw.session_generation == 1 and user_gw.authorised


async def test_a_successful_rebind_swaps_in_the_new_account(
    curator: Curator,
    source: ChatInfo,
    make_message: Callable[..., IncomingMessage],
) -> None:
    rt, user_gw = curator.rt, curator.user_gw
    user_gw.relogin_account = OTHER
    user_gw.password_needed = True
    user_gw.valid_password = PASSWORD
    await curator.say("/bind")
    await curator.say(PHONE)
    await curator.say(SPACED_CODE)
    assert "two-factor password" in curator.last()
    assert user_gw.account == USER_ACCOUNT and user_gw.session_generation == 1  # not yet
    await curator.say(PASSWORD)
    assert "logged in as Second Owner (@second)" in curator.last()
    assert user_gw.account == OTHER and user_gw.session_generation == 2  # the old one is gone
    assert await rt.store.kv_get(KV.ACCOUNT_ID) == OTHER.id
    assert not curator.svc.setup_mode
    await settle(lambda: curator.svc._bind_task is not None and curator.svc._bind_task.done())
    assert curator.account_loops_running()
    assert await curator.intake_delivers(make_message(source))


async def test_the_first_bind_still_starts_from_an_empty_session(
    home: Path,
    settings_file: SettingsFile,
    store: Store,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> None:
    await settings_file.set_value("telegram.owner_id", OWNER_ID)
    chat = user_gw.add_chat(make_chat())
    user_gw.authorised = False
    user_gw.valid_code = CODE
    doubles = Doubles(
        user=user_gw,
        bot=bot_gw,
        clock=clock,
        embedder=FakeEmbedder(),
        classifier=FakeClassifier(),
        llm=FakeLLM(),
    )
    rt, app = build_runtime(home, settings_file=settings_file, store=store, fakes=doubles)
    svc = Service(rt, app, sleep=parked)
    await svc.start()
    curator = Curator(rt, svc, user_gw, bot_gw)
    try:
        assert svc.setup_mode and not svc.supervisor.running("intake")
        user_gw.calls.clear()
        await curator.say("/bind")
        await curator.say(PHONE)
        names = [name for name, _ in user_gw.calls]
        assert "begin_relogin" not in names
        assert names.index("reset_session") < names.index("connect") < names.index("send_code")
        await curator.say(SPACED_CODE)
        assert "logged in as Test Owner" in curator.last()
        await settle(lambda: svc.supervisor.running("intake"))
        assert curator.account_loops_running() and not svc.setup_mode
        assert await curator.intake_delivers(make_message(chat))
    finally:
        await svc.shutdown()


# --- the Telethon gateway on stub clients ------------------------------------------------------


class Stubs:
    """Every client the gateway builds: the working ones in order, and the pending ones."""

    def __init__(self) -> None:
        self.main: list[StubClient] = []
        self.pending: list[StubClient] = []
        self.next_me: types.User | None = None  # who the next working client is signed in as

    def make_main(self) -> StubClient:
        stub = StubClient()
        if self.next_me is not None:
            stub.me = self.next_me
        self.main.append(stub)
        return stub

    def make_pending(self) -> StubClient:
        stub = StubClient()
        stub.authorised = False
        stub.me = types.User(id=OTHER.id, first_name="Second", last_name="Owner", phone="1")
        self.pending.append(stub)
        return stub


@pytest.fixture
def stubs() -> Stubs:
    return Stubs()


@pytest.fixture
async def telethon_rt(
    rt: Runtime, home: Path, clock: FakeClock, stubs: Stubs
) -> tuple[Runtime, TelethonUserGateway]:
    gateway = TelethonUserGateway(
        home,
        1,
        "hash",
        clock=clock,
        client_factory=stubs.make_main,
        pending_client_factory=stubs.make_pending,
    )
    rt.user = gateway
    rt.account = AccountService(rt)
    (home / "user.session").write_bytes(b"old key")
    assert await gateway.connect() is True
    return rt, gateway


async def test_a_stale_pending_file_is_deleted_when_the_session_opens(
    home: Path, clock: FakeClock, stubs: Stubs
) -> None:
    """A re-login a crash or restart cut short leaves ``user.session.pending``: service.py's
    start opens the account with ``connect()``, which removes it."""
    (home / "user.session.pending").write_bytes(b"half-made session")
    (home / "user.session.pending-journal").write_bytes(b"")
    gateway = TelethonUserGateway(home, 1, "hash", clock=clock, client_factory=stubs.make_main)
    assert (home / "user.session.pending").exists()  # building the gateway touches nothing
    await gateway.connect()
    assert not (home / "user.session.pending").exists()
    assert not (home / "user.session.pending-journal").exists()


async def test_rebind_abandoned_on_telethon_keeps_the_working_client(
    telethon_rt: tuple[Runtime, TelethonUserGateway], home: Path, stubs: Stubs
) -> None:
    rt, gateway = telethon_rt
    account = rt.account
    assert isinstance(account, AccountService)
    working = stubs.main[0]
    await account.begin(PHONE)
    pending_file = home / "user.session.pending"
    assert stat.S_IMODE(os.stat(pending_file).st_mode) == 0o600
    pending = stubs.pending[0]
    assert pending.calls == [("send_code_request", {"phone": PHONE})]
    assert ("send_code_request", {"phone": PHONE}) not in working.calls
    assert gateway._phone_code_hash is None  # the hash lives on the pending login
    # the working session still serves: a reconnect-free request goes to it
    assert gateway._client is working and working.connected
    assert (await account.status()).bound
    await gateway.connect()  # a connect while the re-login is under way keeps its file
    assert pending_file.exists()

    await gateway.cancel_relogin()
    assert not pending_file.exists() and pending.connected is False
    assert gateway._client is working and (home / "user.session").read_bytes() == b"old key"
    assert len(stubs.main) == 1


async def test_rebind_wrong_code_on_telethon_leaves_the_working_session(
    telethon_rt: tuple[Runtime, TelethonUserGateway], home: Path, stubs: Stubs
) -> None:
    rt, gateway = telethon_rt
    account = rt.account
    assert isinstance(account, AccountService)
    reasons: list[str] = []

    async def lost(reason: str) -> None:
        reasons.append(reason)

    gateway.on_session_lost(lost)
    await account.begin(PHONE)
    pending = stubs.pending[0]
    pending.sign_in_error = _rpc(errors.PhoneCodeInvalidError)
    with pytest.raises(LoginError) as info:
        await account.code("11111")
    assert info.value.reason == "bad_code"
    # an unregistered key on the pending client is no loss of the working session either
    pending.sign_in_error = _rpc(errors.AuthKeyUnregisteredError)
    with pytest.raises(LoginError) as refused:
        await account.code("22222")
    assert refused.value.reason == "other"
    assert reasons == [] and gateway._client is stubs.main[0]
    assert (home / "user.session").read_bytes() == b"old key"
    me = await gateway.me()
    assert me is not None and me.id == 1001
    assert (await account.status()).step == "code"  # the owner may still type the right one


async def test_rebind_success_on_telethon_swaps_the_session_and_the_handlers(
    telethon_rt: tuple[Runtime, TelethonUserGateway], home: Path, stubs: Stubs
) -> None:
    rt, gateway = telethon_rt
    account = rt.account
    assert isinstance(account, AccountService)
    delivered: list[IncomingMessage] = []

    async def on_message(msg: IncomingMessage) -> None:
        delivered.append(msg)

    gateway.on_message(on_message)
    old = stubs.main[0]
    await account.begin(PHONE)
    pending = stubs.pending[0]
    (home / "user.session.pending").write_bytes(b"new key")  # what Telethon wrote
    pending.sign_in_error = _rpc(errors.SessionPasswordNeededError)
    assert await account.code(CODE) == "password_needed"
    assert gateway._client is old and (home / "user.session").read_bytes() == b"old key"
    pending.sign_in_error = None
    stubs.next_me = pending.me  # the moved file now holds the new account's key
    await account.password(PASSWORD)
    new = stubs.main[-1]
    assert new is not old and gateway._client is new
    assert (home / "user.session").read_bytes() == b"new key"
    assert not (home / "user.session.pending").exists()
    assert old.requests_of(functions.auth.LogOutRequest) and old.connected is False
    assert pending.connected is False and new.connected
    assert len(new.handlers) == 2  # live intake is subscribed on the new client
    on_new_message = new.handlers[0][0]
    msg = tl_message(5, text="after the swap")
    msg._chat = channel()
    await on_new_message(SimpleNamespace(message=msg))
    assert [m.text for m in delivered] == ["after the swap"]
    assert await rt.store.kv_get(KV.ACCOUNT_ID) == OTHER.id
    status = await account.status()
    assert (status.bound, status.step) == (True, "ok") and status.account is not None
    assert status.account.id == OTHER.id
