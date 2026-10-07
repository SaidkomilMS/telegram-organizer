"""bot/llm.py: the three /llm paths, key deletion and in-memory keeping, nothing written before
a successful test and everything after, the test-post choice and the sample fallback, the
restart that asks for the key again, the monthly cap and the second-opinion switch.

The language models are real ``ChatLLM`` objects talking to an ``httpx.MockTransport``
"server", so the wizard is exercised down to the request it would send; no network.
"""

from __future__ import annotations

import inspect
import json
import re
import tomllib
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import httpx
import pytest

from tests.fakes import (
    OWNER_ID,
    FakeBotGateway,
    FakeWorld,
    make_bot_message,
    make_callback,
    plain_text,
)
from tg_curator.bot import llm as llm_mod
from tg_curator.bot.core import BotApp
from tg_curator.domain import KV, NewPost, Post, PostStatus
from tg_curator.i18n import LOCALES_DIR, Translator
from tg_curator.llm import registry
from tg_curator.llm.base import ChatLLM, DisabledLLM
from tg_curator.runtime import EVENT_SETTINGS_CHANGED, Runtime
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

SOURCE = -1_001_000_000_401
KEY = "sk-test-secret-0123456789"
TOKEN = "local-token-abc"
LONG_TEXT = (
    "Центробанк Узбекистана сохранил основную ставку на уровне 14% годовых, сославшись на "
    "замедление инфляции до 8,9% в сентябре и стабильный курс сума. Регулятор ожидает, что "
    "инфляция вернётся к целевому уровню в следующем году, и не исключает снижения ставки."
)
SUMMARY = "ЦБ Узбекистана сохранил ставку 14% на фоне замедления инфляции."


class Server:
    """Scripted OpenAI-compatible endpoints: ``/models`` and ``/chat/completions``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.models_status = 200
        self.chat_status = 200
        self.answer: str | None = SUMMARY

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.models_status != 200:
                return httpx.Response(self.models_status, json={"error": {"message": "no"}})
            return httpx.Response(200, json={"data": [{"id": "qwen3:8b"}, {"id": "llama3.2"}]})
        if request.url.path.endswith("/chat/completions"):
            if self.chat_status != 200:
                return httpx.Response(
                    self.chat_status, json={"error": {"message": "model not found"}}
                )
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": self.answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                },
            )
        return httpx.Response(404)

    def completions(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/completions")]


class Ticker:
    """A monotonic timer that advances ``step`` seconds per reading."""

    def __init__(self, step: float = 2.5) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


@pytest.fixture
def server() -> Server:
    return Server()


@pytest.fixture
async def client(server: Server) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as c:
        yield c


@pytest.fixture
def ticker() -> Ticker:
    return Ticker()


@pytest.fixture
async def app(rt: Runtime, client: httpx.AsyncClient, ticker: Ticker) -> BotApp:
    await rt.settings_file.set_value("general.timezone", "UTC")
    app = BotApp(rt)
    llm_mod.register(app, client=client, timer=ticker)
    return app


@pytest.fixture
def changes(rt: Runtime) -> list[str]:
    seen: list[str] = []

    async def on_change(**_: Any) -> None:
        seen.append(rt.settings.llm.mode)

    rt.events.on(EVENT_SETTINGS_CHANGED, on_change)
    return seen


async def stored_post(
    rt: Runtime,
    text: str,
    *,
    status: PostStatus = PostStatus.digest,
    hours_ago: float = 2,
    message_id: int = 1,
) -> Post:
    await rt.store.upsert_chat(
        ChatInfo(
            id=SOURCE,
            kind="channel",
            title="Kun.uz",
            username="kunuz",
            noforwards=False,
            is_creator=False,
            is_admin=False,
            archived=False,
            muted_until=None,
        )
    )
    post = await rt.store.insert_post(
        NewPost(
            chat_id=SOURCE,
            message_id=message_id,
            kind="post",
            message_ids=[message_id],
            posted_at=rt.clock.now() - timedelta(hours=hours_ago),
            via="live",
            text=text,
            text_hash=text_hash(text),
            urls=[],
        ),
        status=status,
    )
    assert post is not None
    return post


def owner_texts(bot_gw: FakeBotGateway) -> list[str]:
    return [plain_text(m.html) for m in bot_gw.sent(OWNER_ID)]


def last_buttons(bot_gw: FakeBotGateway) -> list[str]:
    msg = bot_gw.sent(OWNER_ID)[-1]
    return [b.data or "" for row in msg.buttons or [] for b in row]


def provider_index(key: str) -> int:
    return next(i for i, p in enumerate(registry.providers()) if p.key == key)


def file_text(rt: Runtime) -> str:
    return rt.settings_file.path.read_text(encoding="utf-8")


async def no_secret_anywhere(rt: Runtime, bot_gw: FakeBotGateway, secret: str) -> None:
    assert secret not in json.dumps(await rt.store.kv_get(KV.BOT_CONVERSATION))
    assert all(secret not in (m.html or "") for m in bot_gw.sent(OWNER_ID))


# --- the menu --------------------------------------------------------------------------------


async def test_menu_shows_three_choices_with_cost_and_privacy(
    app: BotApp, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.say(make_bot_message("/llm"))
    text = owner_texts(bot_gw)[-1]
    assert "now: none" in text
    assert "None — free. Nothing leaves the server" in text
    assert "Self-hosted — free per request" in text
    assert "Provider — a few cents a month" in text
    assert last_buttons(bot_gw) == ["lm:m:0", "lm:m:1", "lm:m:2"]


# --- provider --------------------------------------------------------------------------------


async def test_provider_path_deletes_the_key_tests_then_saves_everything_at_once(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    server: Server,
    changes: list[str],
) -> None:
    await stored_post(rt, LONG_TEXT)
    before = file_text(rt)

    await bot_gw.press(make_callback("lm:m:2"))
    assert f"lm:pr:{provider_index('openai')}" in last_buttons(bot_gw)
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    assert (
        "Sent to OpenAI: the text of posts picked for a digest, the example posts behind a "
        "proposed topic, one post for this test (Kun.uz, 2026-10-05), and — only if you turn "
        "on second opinion — borderline posts. Nothing else, never chat names or identifiers."
    ) in owner_texts(bot_gw)[-1]

    await bot_gw.say(make_bot_message(KEY, message_id=55))
    assert (OWNER_ID, 55) in bot_gw.deleted
    assert server.requests[-1].headers["authorization"] == f"Bearer {KEY}"  # key probe
    assert last_buttons(bot_gw) == ["lm:md:0", "lm:md:1"]
    await no_secret_anywhere(rt, bot_gw, KEY)
    assert file_text(rt) == before and changes == []

    await bot_gw.press(make_callback("lm:md:0"))

    [body] = server.completions()
    assert body["model"] == "gpt-6-luna"
    assert LONG_TEXT in json.dumps(body, ensure_ascii=False)
    texts = owner_texts(bot_gw)
    assert any(f"answered in 2.5 s:\n{SUMMARY}" in t for t in texts)
    cfg = rt.settings.llm
    assert (cfg.mode, cfg.provider, cfg.model, cfg.api_key) == (
        "provider",
        "openai",
        "gpt-6-luna",
        KEY,
    )
    assert changes == ["provider"]  # one update carried every [llm] key
    assert isinstance(rt.llm, ChatLLM)
    await no_secret_anywhere(rt, bot_gw, KEY)

    assert "Monthly spending cap" in texts[-1]
    await bot_gw.say(make_bot_message("$2.50"))
    assert rt.settings.llm.monthly_cap_usd == 2.5
    assert isinstance(rt.llm, ChatLLM) and rt.llm.budget.cap_usd == 2.5
    assert owner_texts(bot_gw)[-1] == "Monthly cap: $2.50."
    assert await rt.store.kv_get(KV.BOT_CONVERSATION) is None


async def test_nothing_is_written_before_a_successful_test(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server, changes: list[str]
) -> None:
    await stored_post(rt, LONG_TEXT)
    before = file_text(rt)
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    await bot_gw.say(make_bot_message(KEY, message_id=9))
    server.chat_status = 404

    await bot_gw.say(make_bot_message("gpt-6-nonexistent"))

    assert "nothing was saved" in owner_texts(bot_gw)[-1]
    assert last_buttons(bot_gw) == ["lm:t"]
    assert file_text(rt) == before and changes == []
    assert rt.settings.llm.mode == "none"

    server.chat_status = 200
    await bot_gw.press(make_callback("lm:t"))

    assert rt.settings.llm.model == "gpt-6-nonexistent"
    assert rt.settings.llm.api_key == KEY
    assert changes == ["provider"]


async def test_a_key_telegram_refuses_to_delete_is_flagged_and_the_flow_goes_on(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server
) -> None:
    from tg_curator.errors import NotAllowed

    await stored_post(rt, LONG_TEXT)
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    bot_gw.fail_next("delete_message", NotAllowed("other", "message can't be deleted"))
    await bot_gw.say(make_bot_message(KEY, message_id=56))
    assert (OWNER_ID, 56) not in bot_gw.deleted
    assert any(
        "could not delete your message with the API key" in t and "yourself" in t
        for t in owner_texts(bot_gw)
    )
    assert last_buttons(bot_gw) == ["lm:md:0", "lm:md:1"]  # the flow goes on
    assert all(KEY not in t for t in owner_texts(bot_gw))


async def test_a_refused_key_is_not_kept_and_is_asked_for_again(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server
) -> None:
    server.models_status = 401
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))

    await bot_gw.say(make_bot_message(KEY, message_id=12))

    assert (OWNER_ID, 12) in bot_gw.deleted
    assert owner_texts(bot_gw)[-1].startswith("OpenAI refused that key")
    conversation = await rt.store.kv_get(KV.BOT_CONVERSATION)
    assert conversation["step"] == "key"
    assert rt.settings.llm.mode == "none"


async def test_a_restart_mid_flow_asks_for_the_key_again(
    app: BotApp,
    rt: Runtime,
    bot_gw: FakeBotGateway,
    world: FakeWorld,
    client: httpx.AsyncClient,
    server: Server,
) -> None:
    await stored_post(rt, LONG_TEXT)
    await bot_gw.press(make_callback("lm:m:2"))
    await bot_gw.press(make_callback(f"lm:pr:{provider_index('openai')}"))
    await bot_gw.say(make_bot_message(KEY, message_id=20))

    # The process restarts: a new bot, a new app; only kv bot.conversation survived.
    restarted = FakeBotGateway(world)
    rt.bot = restarted
    app2 = BotApp(rt)
    llm_mod.register(app2, client=client, timer=Ticker())
    await restarted.press(make_callback("lm:md:0"))

    assert "The service restarted, so the key is gone" in plain_text(
        restarted.sent(OWNER_ID)[-1].html
    )
    assert server.completions() == []
    assert rt.settings.llm.mode == "none"

    await restarted.say(make_bot_message(KEY, message_id=21))
    assert (OWNER_ID, 21) in restarted.deleted
    await restarted.press(make_callback("lm:md:1"))
    assert rt.settings.llm.model == "gpt-6-sol"
    assert rt.settings.llm.api_key == KEY


# --- self-hosted -----------------------------------------------------------------------------


async def test_selfhosted_path_with_a_token_lists_models_and_saves_after_the_test(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server, changes: list[str]
) -> None:
    await stored_post(rt, LONG_TEXT)

    await bot_gw.press(make_callback("lm:m:1"))
    assert last_buttons(bot_gw) == ["lm:p:0", "lm:p:1", "lm:p:2", "lm:p:3", "lm:p:4"]
    await bot_gw.press(make_callback("lm:p:0"))  # Ollama
    assert last_buttons(bot_gw) == ["lm:u"]
    await bot_gw.press(make_callback("lm:u"))
    assert (
        "Sent to your server at http://localhost:11434/v1: the text of posts picked for a "
        "digest, the example posts behind a proposed topic, one post for this test (Kun.uz, "
        "2026-10-05)"
    ) in owner_texts(bot_gw)[-1]

    await bot_gw.say(make_bot_message(TOKEN, message_id=31))

    assert (OWNER_ID, 31) in bot_gw.deleted
    probe = server.requests[-1]
    assert str(probe.url) == "http://localhost:11434/v1/models"
    assert probe.headers["authorization"] == f"Bearer {TOKEN}"
    assert last_buttons(bot_gw) == ["lm:md:0", "lm:md:1"]
    await no_secret_anywhere(rt, bot_gw, TOKEN)
    assert changes == []

    await bot_gw.press(make_callback("lm:md:1"))

    cfg = rt.settings.llm
    assert (cfg.mode, cfg.base_url, cfg.model, cfg.api_key, cfg.provider) == (
        "selfhosted",
        "http://localhost:11434/v1",
        "llama3.2",
        TOKEN,
        "",
    )
    assert changes == ["selfhosted"]
    assert owner_texts(bot_gw)[-1] == "Digests will now use it for one-line summaries."
    assert last_buttons(bot_gw) == ["lm:so"]
    assert await rt.store.kv_get(KV.BOT_CONVERSATION) is None


async def test_without_a_long_stored_post_the_bundled_sample_is_used_and_named(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server
) -> None:
    await stored_post(rt, "Too short to say anything.")

    await bot_gw.press(make_callback("lm:m:1"))
    await bot_gw.press(make_callback("lm:p:4"))  # Other
    await bot_gw.say(make_bot_message("192.168.1.20:8000"))
    assert "one post for this test (a bundled sample paragraph" in owner_texts(bot_gw)[-1]
    await bot_gw.press(make_callback("lm:nt"))
    assert "authorization" not in server.requests[-1].headers
    await bot_gw.say(make_bot_message("my-model"))

    [body] = server.completions()
    sample = rt.t("test_sample")
    assert len(sample) >= llm_mod.TEST_MIN_CHARS
    assert sample in json.dumps(body, ensure_ascii=False)
    assert "the bundled sample paragraph was used" in "\n".join(owner_texts(bot_gw))
    assert rt.settings.llm.base_url == "http://192.168.1.20:8000/v1"
    assert rt.settings.llm.api_key == ""


async def test_a_slow_model_is_saved_with_a_warning(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, ticker: Ticker
) -> None:
    ticker.step = 25.0
    await bot_gw.press(make_callback("lm:m:1"))
    await bot_gw.press(make_callback("lm:p:0"))
    await bot_gw.press(make_callback("lm:u"))
    await bot_gw.press(make_callback("lm:nt"))
    await bot_gw.press(make_callback("lm:md:0"))
    assert any("That took 25 s for one line" in t for t in owner_texts(bot_gw))
    assert rt.settings.llm.mode == "selfhosted"


async def test_an_unreachable_server_asks_for_the_address_again(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway, server: Server
) -> None:
    server.models_status = 503
    await bot_gw.press(make_callback("lm:m:1"))
    await bot_gw.press(make_callback("lm:p:1"))
    await bot_gw.press(make_callback("lm:u"))
    await bot_gw.press(make_callback("lm:nt"))
    assert "send the address again" in owner_texts(bot_gw)[-1]
    conversation = await rt.store.kv_get(KV.BOT_CONVERSATION)
    assert conversation["step"] == "address"
    assert rt.settings.llm.mode == "none"


# --- none, second opinion, stale buttons ------------------------------------------------------


async def configure_provider(rt: Runtime) -> None:
    def mutate(doc: Any) -> None:
        doc["llm"]["mode"] = "provider"
        doc["llm"]["provider"] = "openai"
        doc["llm"]["model"] = "gpt-6-luna"
        doc["llm"]["api_key"] = KEY

    await rt.settings_file.update(mutate)


async def test_none_clears_the_model_and_the_key(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await configure_provider(rt)
    await bot_gw.say(make_bot_message("/llm"))
    assert "now: OpenAI gpt-6-luna, monthly cap: none" in owner_texts(bot_gw)[-1]
    assert last_buttons(bot_gw)[-2:] == ["lm:cap", "lm:so"]

    await bot_gw.press(make_callback("lm:m:0"))

    cfg = rt.settings.llm
    assert (cfg.mode, cfg.provider, cfg.model, cfg.api_key) == ("none", "", "", "")
    assert KEY not in file_text(rt)
    assert isinstance(rt.llm, DisabledLLM)
    assert owner_texts(bot_gw)[-1].startswith("Saved: no language model")


async def test_second_opinion_switch_repeats_the_privacy_clause_when_turned_on(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback("lm:so"))
    assert "needs a connected model" in owner_texts(bot_gw)[-1]
    assert rt.settings.sorting.second_opinion is False

    await configure_provider(rt)
    await bot_gw.press(make_callback("lm:so"))
    assert rt.settings.sorting.second_opinion is True
    text = owner_texts(bot_gw)[-1]
    assert "borderline posts" in text.lower() and "sent to OpenAI" in text
    assert "Nothing else, never chat names or identifiers." in text

    await bot_gw.press(make_callback("lm:so"))
    assert rt.settings.sorting.second_opinion is False
    assert owner_texts(bot_gw)[-1].startswith("Second opinion is off")


async def test_monthly_cap_from_the_menu(app: BotApp, rt: Runtime, bot_gw: FakeBotGateway) -> None:
    await configure_provider(rt)
    await bot_gw.press(make_callback("lm:cap"))
    await bot_gw.say(make_bot_message("lots"))
    assert "Send an amount in dollars" in owner_texts(bot_gw)[-1]
    await bot_gw.say(make_bot_message("3"))
    assert rt.settings.llm.monthly_cap_usd == 3.0
    await bot_gw.press(make_callback("lm:cap"))
    await bot_gw.press(make_callback("lm:c0"))
    assert rt.settings.llm.monthly_cap_usd == 0.0
    assert owner_texts(bot_gw)[-1] == "Monthly cap: none."


async def test_stale_buttons_change_nothing(
    app: BotApp, rt: Runtime, bot_gw: FakeBotGateway
) -> None:
    await bot_gw.press(make_callback("lm:md:0"))
    await bot_gw.press(make_callback("lm:pr:99"))
    assert owner_texts(bot_gw) == [plain_text(rt.t("unknown_choice"))] * 2
    assert rt.settings.llm.mode == "none"


# --- the test post ---------------------------------------------------------------------------


async def test_the_test_post_is_the_newest_long_digest_pick_and_never_anything_else(
    rt: Runtime,
) -> None:
    assert (await llm_mod.pick_test_post(rt)).post_id is None
    for n, status in enumerate(
        (PostStatus.unsorted, PostStatus.rejected, PostStatus.duplicate, PostStatus.dropped), 1
    ):
        await stored_post(rt, LONG_TEXT + f" {status}", status=status, hours_ago=1, message_id=n)
    assert (await llm_mod.pick_test_post(rt)).post_id is None  # the bundled sample, not these
    await stored_post(rt, "Short but digested.", status=PostStatus.digested, message_id=12)
    old_digest = await stored_post(
        rt, LONG_TEXT + " digest", status=PostStatus.digest, hours_ago=30, message_id=13
    )
    newer_published = await stored_post(
        rt, LONG_TEXT + " published", status=PostStatus.published, hours_ago=5, message_id=14
    )

    picked = await llm_mod.pick_test_post(rt)

    assert picked.post_id == newer_published.id != old_digest.id
    assert picked.label == "Kun.uz, 2026-10-05"
    assert picked.text.endswith("published")


# --- the catalogue ---------------------------------------------------------------------------


def test_every_llm_key_used_exists_and_nothing_unused_is_shipped() -> None:
    source = inspect.getsource(llm_mod)
    used = set(re.findall(r'"(llm_[a-z_]+|test_sample)"', source))
    with (LOCALES_DIR / "en" / "llm.toml").open("rb") as fh:
        shipped = set(tomllib.load(fh))
    assert used == shipped
    common = {"unknown_choice", "on", "off", "retry"}
    assert common <= Translator(locales_dir=LOCALES_DIR).english_keys()
