"""bot/bind.py: /bind phone -> code -> 2FA, deletions, no secrets kept, resend, restart."""

from __future__ import annotations

import inspect
import json
import logging
import re
import tomllib

import pytest
import sqlalchemy as sa

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeUserGateway,
    make_bot_message,
    make_callback,
    plain_text,
)
from tg_curator.account import AccountService
from tg_curator.bot import bind
from tg_curator.bot.core import BotApp, Ctx
from tg_curator.db import schema
from tg_curator.db.store import Store
from tg_curator.domain import KV
from tg_curator.errors import NotAllowed
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.runtime import EVENT_SESSION_LOST, Runtime

PHONE = "+998901234567"
CODE = "73915"
SPACED_CODE = "7 3 9 1 5"
PASSWORD = "correct horse battery"


@pytest.fixture
async def app(rt: Runtime) -> BotApp:
    rt.account = AccountService(rt)
    app = BotApp(rt)
    bind.register(app)
    await app.start()
    return app


@pytest.fixture
def unbound(user_gw: FakeUserGateway) -> FakeUserGateway:
    user_gw.authorised = False
    user_gw.valid_code = CODE
    return user_gw


class Chat:
    """Types as the owner, with increasing message ids, and reads the bot's replies."""

    def __init__(self, bot_gw: FakeBotGateway) -> None:
        self.bot_gw = bot_gw
        self.next_id = 100

    async def say(self, text: str) -> int:
        self.next_id += 1
        await self.bot_gw.say(make_bot_message(text, message_id=self.next_id))
        return self.next_id

    def texts(self) -> list[str]:
        return [plain_text(m.html) for m in self.bot_gw.sent(OWNER_ID)]

    def last(self) -> str:
        return self.texts()[-1]


@pytest.fixture
def chat(bot_gw: FakeBotGateway) -> Chat:
    return Chat(bot_gw)


async def kv_dump(store: Store) -> str:
    rows = await store.execute(sa.select(schema.kv.c.key, schema.kv.c.value))
    return json.dumps([[r.key, r.value] for r in rows])


async def conversation(store: Store) -> dict | None:
    return await store.kv_get(KV.BOT_CONVERSATION)


# --- the happy path ----------------------------------------------------------------------------


async def test_bind_happy_path_deletes_the_code_and_logs_in(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await chat.say("/bind")
    assert "phone number" in chat.last()
    assert unbound.calls_of("reset_session") == []  # nothing is dropped before the phone arrives

    await chat.say("+998 90 123-45-67")
    names = [name for name, _ in unbound.calls]
    # the gateway drops the old session (file and in-memory key) before the new login
    assert names.index("reset_session") < names.index("connect") < names.index("send_code")
    assert unbound.calls_of("send_code") == [{"phone": PHONE}]
    assert "space between every digit" in chat.last()
    assert (await conversation(rt.store)) == {
        "flow": "bind",
        "step": "code",
        "data": {"then": None},
    }

    code_msg = await chat.say(SPACED_CODE)
    assert (OWNER_ID, code_msg) in bot_gw.deleted
    assert unbound.calls_of("sign_in") == [{"code": CODE}]
    assert "logged in as Test Owner (@test_owner)" in chat.last()
    assert await conversation(rt.store) is None
    assert await rt.store.kv_get(KV.ACCOUNT_ID) == OWNER_ID


async def test_letters_and_separators_are_stripped_from_the_code(
    app: BotApp, chat: Chat, unbound: FakeUserGateway
) -> None:
    await chat.say("/bind")
    await chat.say(PHONE)
    await chat.say("c7-3.9 1_5")
    assert unbound.calls_of("sign_in") == [{"code": CODE}]


async def test_two_factor_password_is_deleted_and_passed_verbatim(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    unbound.password_needed = True
    unbound.valid_password = PASSWORD
    await chat.say("/bind")
    await chat.say(PHONE)
    code_msg = await chat.say(SPACED_CODE)
    assert "two-factor password" in chat.last()
    assert (await conversation(rt.store))["step"] == "password"

    pw_msg = await chat.say(PASSWORD)
    assert (OWNER_ID, code_msg) in bot_gw.deleted and (OWNER_ID, pw_msg) in bot_gw.deleted
    assert unbound.calls_of("sign_in_password") == [{"password": PASSWORD}]
    assert "logged in as Test Owner" in chat.last()
    # the deleted messages are gone from the chat
    assert all(m.message_id not in (code_msg, pw_msg) for m in bot_gw.sent(OWNER_ID))


async def test_a_code_and_password_telegram_refuses_to_delete_are_flagged_to_the_owner(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    unbound.password_needed = True
    unbound.valid_password = PASSWORD
    await chat.say("/bind")
    await chat.say(PHONE)
    bot_gw.fail_next("delete_message", NotAllowed("other", "message can't be deleted"))
    await chat.say(SPACED_CODE)
    assert bot_gw.deleted == []
    assert any(
        "could not delete your message with the login code" in t and "yourself" in t
        for t in chat.texts()
    )
    assert "two-factor password" in chat.last()  # the flow goes on
    bot_gw.fail_next("delete_message", NotAllowed("other", "message can't be deleted"))
    await chat.say(PASSWORD)
    assert any("could not delete your message with the password" in t for t in chat.texts())
    assert "logged in as Test Owner" in chat.last()
    assert all(PASSWORD not in t for t in chat.texts())


async def test_no_code_or_password_reaches_kv_or_the_logs(
    app: BotApp,
    rt: Runtime,
    chat: Chat,
    unbound: FakeUserGateway,
    caplog: pytest.LogCaptureFixture,
) -> None:
    unbound.password_needed = True
    unbound.valid_password = PASSWORD
    secrets = (CODE, SPACED_CODE, PASSWORD)
    with caplog.at_level(logging.DEBUG):
        await chat.say("/bind")
        await chat.say(PHONE)
        await chat.say(SPACED_CODE)
        mid_flow = await kv_dump(rt.store)
        await chat.say("wrong password")
        await chat.say(PASSWORD)
    final = await kv_dump(rt.store)
    for secret in (*secrets, "wrong password"):
        assert secret not in mid_flow
        assert secret not in final
        assert secret not in caplog.text
    assert PHONE not in caplog.text  # the phone appears only masked
    assert "logged in as" in chat.last()


async def test_wrong_two_factor_password_is_explained_and_asked_again(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway
) -> None:
    unbound.password_needed = True
    unbound.valid_password = PASSWORD
    await chat.say("/bind")
    await chat.say(PHONE)
    await chat.say(SPACED_CODE)
    await chat.say("nope")
    assert "two-factor password is not right" in chat.last()
    assert (await conversation(rt.store))["step"] == "password"
    await chat.say(PASSWORD)
    assert "logged in as" in chat.last()


# --- refusals ----------------------------------------------------------------------------------


async def test_bad_code_offers_a_new_code_and_the_flow_stays_at_the_code(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await chat.say("/bind")
    await chat.say(PHONE)
    bad = await chat.say("1 1 1 1 1")
    assert (OWNER_ID, bad) in bot_gw.deleted
    reply = bot_gw.sent(OWNER_ID)[-1]
    assert "probably sent without spaces" in plain_text(reply.html)
    assert reply.buttons is not None
    assert [(b.text, b.data) for b in reply.buttons[0]] == [("Send a new code", bind.RESEND)]
    assert (await conversation(rt.store))["step"] == "code"

    await bot_gw.press(make_callback(bind.RESEND, message_id=reply.message_id))
    assert len(unbound.calls_of("resend_code")) == 1
    assert "new code is on its way" in chat.last()

    await chat.say(SPACED_CODE)
    assert "logged in as" in chat.last()


async def test_a_message_without_digits_is_not_sent_to_telegram(
    app: BotApp, chat: Chat, unbound: FakeUserGateway
) -> None:
    await chat.say("/bind")
    await chat.say(PHONE)
    await chat.say("what code?")
    assert unbound.calls_of("sign_in") == []
    assert "No digits" in chat.last()


async def test_an_invalid_phone_is_asked_again(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway
) -> None:
    await chat.say("/bind")
    await chat.say("my phone")
    assert "does not look like a phone number" in chat.last()
    assert unbound.calls_of("send_code") == []
    assert (await conversation(rt.store))["step"] == "phone"


async def test_a_resend_without_a_pending_code_is_refused(
    app: BotApp, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback(bind.RESEND))
    assert unbound.calls_of("resend_code") == []
    assert "not available" in chat.last()


# --- restarts and re-linking -------------------------------------------------------------------


async def test_a_restart_between_code_request_and_code_resumes_at_the_phone_step(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    await chat.say("/bind")
    await chat.say(PHONE)
    rt.account = AccountService(rt)  # a new process: the pending code died with the old one
    msg = await chat.say(SPACED_CODE)
    assert (OWNER_ID, msg) in bot_gw.deleted
    assert unbound.calls_of("sign_in") == []
    assert "restarted" in chat.last()
    assert (await conversation(rt.store))["step"] == "phone"

    await chat.say(PHONE)
    await chat.say(SPACED_CODE)
    assert "logged in as" in chat.last()


async def test_bind_when_bound_names_the_account_and_drops_nothing_yet(
    app: BotApp, chat: Chat, user_gw: FakeUserGateway
) -> None:
    await chat.say("/bind")
    texts = chat.texts()
    assert "Currently logged in as Test Owner" in texts[-2]
    assert "phone number" in texts[-1]
    assert user_gw.calls_of("reset_session") == []


async def test_bind_on_a_bound_account_logs_in_beside_the_working_session(
    app: BotApp, rt: Runtime, chat: Chat, user_gw: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    lost: list[dict[str, object]] = []

    async def on_lost(**payload: object) -> None:
        lost.append(payload)

    rt.events.on(EVENT_SESSION_LOST, on_lost)
    user_gw.valid_code = CODE
    await chat.say("/bind")
    await chat.say(PHONE)
    assert user_gw.calls_of("begin_relogin") and user_gw.calls_of("reset_session") == []
    assert lost == [] and rt.account.setup_mode is False  # type: ignore[union-attr]
    assert "space between every digit" in chat.last()
    code_msg = await chat.say(SPACED_CODE)
    assert (OWNER_ID, code_msg) in bot_gw.deleted
    assert user_gw.session_generation == 2  # swapped in only now
    assert "logged in as Test Owner" in chat.last()


async def test_a_command_mid_flow_cancels_the_bind(
    app: BotApp, rt: Runtime, chat: Chat, unbound: FakeUserGateway
) -> None:
    await chat.say("/bind")
    await chat.say(PHONE)
    await chat.say("/help")
    assert await conversation(rt.store) is None


async def test_the_continue_button_is_offered_when_the_caller_asks_for_it(
    rt: Runtime, chat: Chat, unbound: FakeUserGateway, bot_gw: FakeBotGateway
) -> None:
    rt.account = AccountService(rt)
    app = BotApp(rt)
    bind.register(app)

    async def via_setup(ctx: Ctx, args: str) -> None:  # a stand-in for /setup
        await bind.begin_bind(ctx, then="su:next")

    app.command("setup", via_setup)
    await chat.say("/setup")
    await chat.say(PHONE)
    await chat.say(SPACED_CODE)
    done = bot_gw.sent(OWNER_ID)[-1]
    assert done.buttons is not None and done.buttons[0][0].data == "su:next"


# --- the catalogue -----------------------------------------------------------------------------


def test_every_bind_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(bind)
    used = set(re.findall(r'"(bind_[a-z_]+)"', source))
    with (LOCALES_DIR / "en" / "bind.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = set(re.findall(r'"(unknown_[a-z_]+)"', source))
    assert common and common <= Translator(locales_dir=LOCALES_DIR).english_keys()
