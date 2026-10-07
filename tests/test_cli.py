"""cli.py: the ``curator`` commands over a temp home, with the fakes injected (DESIGN §13).

The CLI builds its own runtime per command; ``service.doubles_factory`` hands it the fake
gateways and clock, and ``TG_CURATOR_FAKE_MODELS=1`` the hash embedder, so nothing here
touches Telegram, the network or the model files.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import tomlkit
from click.testing import CliRunner, Result

from tests.fakes import USER_ACCOUNT, FakeBotGateway, FakeClock, FakeLLM, FakeUserGateway
from tg_curator import __version__, control, service
from tg_curator.cli import RUNNING, cli
from tg_curator.clock import local_date
from tg_curator.config import ENV_OVERRIDES, SettingsFile, write_template
from tg_curator.db.store import Store, sqlite_url
from tg_curator.domain import NewPost, PostStatus
from tg_curator.ml import models
from tg_curator.service import Doubles
from tg_curator.telegram.gateway import ChatInfo
from tg_curator.textutil import text_hash

SOURCE = -1_001_000_009_101
OTHER = -1_001_000_009_102


def info(chat_id: int, title: str, username: str | None = None) -> ChatInfo:
    return ChatInfo(
        id=chat_id,
        kind="channel",
        title=title,
        username=username,
        noforwards=False,
        is_creator=False,
        is_admin=False,
        archived=False,
        muted_until=None,
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home that does not exist yet, an environment without Docker overrides, fake models."""
    for var in (*ENV_OVERRIDES, service.WAIT_ENV, "TG_CURATOR_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(models.FAKE_MODELS_ENV, "1")
    monkeypatch.setattr(models, "_embedder", None)
    return tmp_path / "home"


@pytest.fixture
def fakes(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
) -> Doubles:
    doubles = Doubles(user=user_gw, bot=bot_gw, clock=clock, llm=FakeLLM())
    monkeypatch.setattr(service, "doubles_factory", lambda: doubles)
    return doubles


@pytest.fixture
def configured(home: Path) -> Path:
    """The template with the three [telegram] values and an owner filled in."""
    path = home / "settings.toml"
    write_template(path)
    settings_file = SettingsFile(path, env={})
    settings_file.load_sync()

    def fill(doc: Any) -> None:
        doc["telegram"]["api_id"] = 12345
        doc["telegram"]["api_hash"] = "0123456789abcdef"
        doc["telegram"]["bot_token"] = "1:token"

    asyncio.run(settings_file.update(fill))
    return home


def seed(home: Path, clock: FakeClock, work: Callable[[Store], Any]) -> None:
    async def go() -> None:
        store = Store(sqlite_url(home / "curator.db"), clock=clock)
        await store.start()
        try:
            await work(store)
        finally:
            await store.close()

    asyncio.run(go())


def curator(home: Path, *args: str, input: str | None = None) -> Result:  # noqa: A002
    return CliRunner().invoke(cli, ["--home", str(home), *args], input=input)


# --- first start -------------------------------------------------------------------------------


def test_first_run_writes_the_template_and_prints_the_steps(home: Path) -> None:
    result = curator(home, "run")
    assert result.exit_code == 0, result.output
    assert (home / "settings.toml").is_file()
    assert "Next steps" in result.output and "@BotFather" in result.output
    assert (home / "settings.toml").stat().st_mode & 0o777 == 0o600


def test_missing_telegram_values_exit_2_naming_the_keys(home: Path) -> None:
    write_template(home / "settings.toml")
    result = curator(home, "run")
    assert result.exit_code == 2
    for key in ("telegram.api_id", "telegram.api_hash", "telegram.bot_token"):
        assert key in result.output


def test_an_invalid_file_exits_2_with_the_sentence(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "settings.toml").write_text("[digest]\nhour = 99\n")
    result = curator(home, "run")
    assert result.exit_code == 2
    assert "digest.hour" in result.output


def test_other_commands_stop_after_writing_the_template(home: Path, fakes: Doubles) -> None:
    result = curator(home, "stats")
    assert result.exit_code == 1
    assert "Next steps" in result.output


# --- global options ----------------------------------------------------------------------------


def test_service_file_prints_the_shipped_unit(home: Path) -> None:
    shipped = Path(service.__file__).parent / "data" / "tg-curator.service"
    result = CliRunner().invoke(cli, ["--service-file"])
    assert result.exit_code == 0
    assert result.output == shipped.read_text()
    assert "Environment=TG_CURATOR_HOME=/var/lib/tg-curator" in result.output
    assert not home.exists()  # printing the unit touches nothing


def test_version(home: Path) -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_topics_categories_needs_nothing_but_the_package(home: Path) -> None:
    result = curator(home, "topics", "categories")
    assert result.exit_code == 0
    assert "tech" in result.output and "real_estate" in result.output


# --- standalone commands on a seeded store -----------------------------------------------------


def test_preview_prints_every_decision_and_writes_nothing(
    configured: Path, clock: FakeClock, fakes: Doubles
) -> None:
    async def work(store: Store) -> None:
        await store.upsert_chat(info(SOURCE, "Tech news"))
        for mid, text in enumerate(
            ["A new open model beats the benchmark on reasoning tasks", "Football: the derby"], 1
        ):
            new = NewPost(
                chat_id=SOURCE,
                message_id=mid,
                kind="post",
                message_ids=[mid],
                posted_at=clock.now() - timedelta(hours=mid),
                via="live",
                text=text,
                text_hash=text_hash(text),
                urls=[],
            )
            await store.insert_post(new, status=PostStatus.unsorted)

    seed(configured, clock, work)
    result = curator(configured, "preview", "--set", "sorting.confidence=0.9")
    assert result.exit_code == 0, result.output
    assert "preview: last 3 days, 2 decisions" in result.output
    assert "A new open model beats the benchmark" in result.output
    assert "Tech news" in result.output
    assert not [name for name, _ in fakes.bot.calls if name.startswith("send")]  # type: ignore[union-attr]


def test_preview_with_an_unknown_topic_says_what_to_do(configured: Path, fakes: Doubles) -> None:
    result = curator(configured, "preview", "--topic", "nope")
    assert result.exit_code == 1
    assert "nope" in result.output


def test_stats_on_a_seeded_store(configured: Path, clock: FakeClock, fakes: Doubles) -> None:
    async def work(store: Store) -> None:
        await store.upsert_chat(info(SOURCE, "Loud channel"))
        await store.bump_chat_daily(SOURCE, local_date(clock.now(), "UTC"), 300)

    seed(configured, clock, work)
    result = curator(configured, "stats", "--days", "30")
    assert result.exit_code == 0, result.output
    assert "Loud channel" in result.output and "300" in result.output
    assert "stats: 1 chats over the last 30 days" in result.output


def test_topics_syncs_the_example_topics_and_confirms(
    configured: Path, fakes: Doubles, user_gw: FakeUserGateway
) -> None:
    result = curator(configured, "topics")
    assert result.exit_code == 0, result.output
    assert "4 topics, all channels resolved" in result.output
    assert "ml-ai" in result.output and "tracked only" in result.output
    assert user_gw.calls_of("connect") and user_gw.calls_of("disconnect")


def test_chats_lists_the_dialogs_with_identifiers(
    configured: Path, fakes: Doubles, user_gw: FakeUserGateway
) -> None:
    user_gw.add_chat(info(SOURCE, "Kun.uz", username="kunuz"))
    user_gw.add_chat(info(OTHER, "Daryo"))
    result = curator(configured, "chats")
    assert result.exit_code == 0, result.output
    assert str(SOURCE) in result.output and "@kunuz" in result.output
    assert "found 2 chats, 0 are output channels" in result.output


def test_a_command_needing_the_account_says_to_log_in_first(
    configured: Path, fakes: Doubles, user_gw: FakeUserGateway
) -> None:
    user_gw.authorised = False
    result = curator(configured, "chats")
    assert result.exit_code == 1
    assert "curator login" in result.output


def test_digest_send_is_refused_before_go(configured: Path, fakes: Doubles) -> None:
    result = curator(configured, "digest")
    assert result.exit_code == 1
    assert "live" in result.output.lower() or "/go" in result.output


# --- the running service -----------------------------------------------------------------------


def test_commands_go_through_the_socket_while_the_service_runs(
    configured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[str, Mapping[str, Any]]] = []

    async def request(
        path: Path, name: str, args: Mapping[str, Any], on_line: Callable[[str], None]
    ) -> int:
        sent.append((name, dict(args)))
        on_line("from the service")
        return 0

    monkeypatch.setattr(control, "request", request)
    lock = control.ServiceLock(configured)
    assert lock.acquire()
    try:
        result = curator(configured, "preview", "--days", "2", "--set", "sorting.confidence=0.7")
        backfill = curator(configured, "backfill", "--chat", "@kunuz", "--chat", "-1001")
        digest = curator(configured, "digest", "--preview", "ml-ai")
    finally:
        lock.release()
    assert result.exit_code == 0 and "from the service" in result.output
    assert sent == [
        ("preview", {"topic": None, "days": 2, "set": {"sorting.confidence": 0.7}}),
        ("backfill", {"days": 3, "chats": ["@kunuz", "-1001"]}),
        ("digest-preview", {"topic": "ml-ai"}),
    ]
    assert backfill.exit_code == 0 and digest.exit_code == 0


def test_lock_held_and_no_socket_gives_the_running_sentence(configured: Path) -> None:
    lock = control.ServiceLock(configured)
    assert lock.acquire()
    try:
        result = curator(configured, "stats")
        login = curator(configured, "login", "--phone", "+998901234567")
    finally:
        lock.release()
    assert result.exit_code == 1 and RUNNING in result.output
    assert login.exit_code == 1 and RUNNING in login.output


# --- login -------------------------------------------------------------------------------------


def test_login_prompts_in_the_terminal_and_makes_the_account_the_owner(
    configured: Path, fakes: Doubles, user_gw: FakeUserGateway
) -> None:
    user_gw.authorised = False
    user_gw.valid_code = "12345"
    result = curator(configured, "login", input="+998901234567\n11111\n12345\n")
    assert result.exit_code == 0, result.output
    assert "logged in as Test Owner (@test_owner)" in result.output
    assert user_gw.calls_of("send_code") == [{"phone": "+998901234567"}]
    assert [c["code"] for c in user_gw.calls_of("sign_in")] == ["11111", "12345"]
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    assert settings.telegram.owner_id == USER_ACCOUNT.id


def test_login_with_a_password(configured: Path, fakes: Doubles, user_gw: FakeUserGateway) -> None:
    user_gw.authorised = False
    user_gw.password_needed = True
    user_gw.valid_password = "secret"
    result = curator(configured, "login", "--phone", "+998901234567", input="777\nsecret\n")
    assert result.exit_code == 0, result.output
    assert user_gw.calls_of("sign_in_password") == [{"password": "secret"}]
    assert "secret" not in result.output.replace("Two-step verification password", "")


def test_login_when_already_signed_in(configured: Path, fakes: Doubles) -> None:
    result = curator(configured, "login")
    assert result.exit_code == 0
    assert "already logged in as Test Owner" in result.output


# --- llm ---------------------------------------------------------------------------------------


def test_llm_none_is_saved_without_a_test(configured: Path, fakes: Doubles) -> None:
    path = configured / "settings.toml"
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    llm_table: Any = doc["llm"]
    llm_table["mode"] = "provider"
    llm_table["provider"] = "openai"
    llm_table["model"] = "gpt-6-luna"
    llm_table["api_key"] = "sk-SECRET-KEY-123"
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    result = curator(configured, "llm", "--mode", "none")
    assert result.exit_code == 0, result.output
    assert "Nothing leaves the server" in result.output
    settings = SettingsFile(path, env={}).load_sync()
    assert settings.llm.mode == "none"
    assert (settings.llm.provider, settings.llm.model, settings.llm.base_url) == ("", "", "")
    assert settings.llm.api_key == ""
    assert "sk-SECRET-KEY-123" not in path.read_text(encoding="utf-8")


def test_llm_interactive_shows_the_three_choices_first(configured: Path, fakes: Doubles) -> None:
    result = curator(configured, "llm", input="none\n")
    assert result.exit_code == 0, result.output
    out = result.output
    prompt_at = out.index("Language model (none, selfhosted, provider)")
    assert "Nothing leaves the server" in out[:prompt_at]
    assert "a few cents a month" in out[:prompt_at]
    assert "Self-hosted" in out[:prompt_at]
    assert "<b>" not in out


def test_llm_provider_is_tested_before_it_is_saved(
    configured: Path, fakes: Doubles, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tg_curator.llm import factory

    built: list[Any] = []

    def make_llm(settings: Any, store: Any, **_: Any) -> FakeLLM:
        built.append(settings.llm)
        return FakeLLM(summary="A one-line summary of the sample.")

    monkeypatch.setattr(factory, "make_llm", make_llm)
    args = ["--mode", "provider", "--provider", "openai", "--model", "gpt-6-luna", "--key-stdin"]
    result = curator(configured, "llm", *args, input="sk-test-key\n")
    assert result.exit_code == 0, result.output
    assert "Sent to OpenAI" in result.output and "bundled sample" in result.output
    assert "A one-line summary of the sample." in result.output
    assert "sk-test-key" not in result.output
    assert built[0].api_key == "sk-test-key"
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    assert (settings.llm.mode, settings.llm.provider, settings.llm.model) == (
        "provider",
        "openai",
        "gpt-6-luna",
    )


def test_llm_is_not_saved_when_the_test_fails(
    configured: Path, fakes: Doubles, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tg_curator.llm import factory

    monkeypatch.setattr(factory, "make_llm", lambda *a, **k: FakeLLM(summary=None))
    args = ["--mode", "selfhosted", "--base-url", "http://localhost:11434/v1", "--model", "qwen3"]
    result = curator(configured, "llm", *args, "--key-stdin", input="\n")
    assert result.exit_code == 1
    assert "nothing was saved" in result.output
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    assert settings.llm.mode == "none"


# --- topics add / remove -----------------------------------------------------------------------


def test_topics_add_and_remove_with_the_service_stopped(configured: Path, fakes: Doubles) -> None:
    added = curator(configured, "topics", "add", "Crypto & exchanges", "--category", "crypto")
    assert added.exit_code == 0, added.output
    assert "added, channel: none yet (tracked only)" in added.output
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    key = next(t.key for t in settings.topics if t.name == "Crypto & exchanges")

    again = curator(configured, "topics", "add", "crypto & EXCHANGES")
    assert again.exit_code == 1 and "edit the existing topic" in again.output

    removed = curator(configured, "topics", "remove", key)
    assert removed.exit_code == 0, removed.output
    settings = SettingsFile(configured / "settings.toml", env={}).load_sync()
    assert settings.topic(key) is None


def test_topics_add_and_remove_go_through_the_socket_while_the_service_runs(
    configured: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[str, Mapping[str, Any]]] = []

    async def request(
        path: Path, name: str, args: Mapping[str, Any], on_line: Callable[[str], None]
    ) -> int:
        sent.append((name, dict(args)))
        on_line("topic crypto (Crypto) added, channel: -1001")
        return 0

    monkeypatch.setattr(control, "request", request)
    lock = control.ServiceLock(configured)
    assert lock.acquire()
    try:
        added = curator(configured, "topics", "add", "Crypto", "--create-channel")
        removed = curator(configured, "topics", "remove", "crypto")
    finally:
        lock.release()
    assert added.exit_code == 0 and "added, channel" in added.output, added.output
    assert removed.exit_code == 0, removed.output
    assert sent == [
        (
            "topics-add",
            {
                "name": "Crypto",
                "category": None,
                "description": None,
                "example_channel": None,
                "channel": None,
                "create_channel": True,
            },
        ),
        ("topics-remove", {"key": "crypto"}),
    ]
