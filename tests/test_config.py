"""config.py: template, validation sentences, home resolution, env overrides, SettingsFile."""

import logging
import os
import stat
from pathlib import Path

import pytest
import tomlkit

from tg_curator import config
from tg_curator.config import (
    TEMPLATE_PATH,
    Settings,
    SettingsFile,
    ensure_home,
    resolve_home,
    validate_settings,
    write_template,
)
from tg_curator.errors import ConfigError

PROVISIONAL_CATEGORIES = {
    "tech", "science", "finance", "crypto", "politics", "world", "sport", "health", "education",
    "culture", "real_estate", "jobs", "travel", "auto", "society", "disaster", "religion",
    "lifestyle", "humor", "ads",
}  # fmt: skip


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.toml"
    write_template(path)
    return path


@pytest.fixture
def sf(settings_path: Path) -> SettingsFile:
    return SettingsFile(settings_path, env={})


# --- template and defaults -------------------------------------------------------------------


def test_template_loads_with_defaults(sf: SettingsFile) -> None:
    s = sf.load_sync()
    assert s.telegram.api_id == 0 and s.telegram.owner_id == 0
    assert s.general.timezone == "UTC" and s.general.language == "en"
    assert s.publishing.live is False and s.publishing.style == "repost"
    assert s.publishing.min_gap_seconds == 20 and s.publishing.staging_channel == 0
    assert s.sorting.confidence == 0.5 and s.sorting.duplicate_similarity == 0.90
    assert s.sorting.realtime_strength == 3.0 and s.sorting.hold_minutes == 45
    assert s.sorting.dedup_window_days == 3 and s.sorting.neutral_trust == 1.0
    assert s.sorting.link_bonus == 0.25 and s.sorting.length_bonus_chars == 600
    assert s.groups.min_chars == 400 and s.groups.unit_gap_minutes == 5
    assert (s.digest.hour, s.digest.minute, s.digest.items) == (21, 0, 15)
    assert s.digest.window_hours == 26 and s.digest.line_chars == 180
    assert s.review.weekday == "sunday" and s.review.hour == 11
    assert s.review.leave_min_days == 30 and s.review.archive_min_days == 30
    assert s.review.cluster_tightness == 0.60 and s.review.merge_margin == 0.30
    assert s.review.auto_create_topics is False
    assert s.folders.curated_name == "Curated" and s.folders.low_signal_name == "Low signal"
    assert s.llm.mode == "none" and s.llm.monthly_cap_usd == 0.0 and s.llm.timeout_seconds == 30
    assert s.storage.database_url == "" and s.storage.keep_embeddings_days == 30
    assert s.sources == []


def test_template_has_four_live_example_topics(sf: SettingsFile) -> None:
    s = sf.load_sync()
    assert [t.key for t in s.topics] == ["ml-ai", "fintech", "uzbekistan", "football"]
    assert [t.name for t in s.topics] == ["ML & AI", "Fintech", "Узбекистан: новости", "Футбол"]
    assert all(t.channel == 0 for t in s.topics)
    assert s.topic("ml-ai").category == "tech"
    assert s.topic("fintech").category == "finance"
    assert s.topic("football").category == "sport"
    assert s.topic("uzbekistan").category is None
    assert "Новости" in s.topic("uzbekistan").description
    for t in s.topics:
        assert t.category is None or t.category in PROVISIONAL_CATEGORIES


def test_template_mentions_every_key_and_the_tuning_symptoms() -> None:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    doc = tomlkit.parse(text)
    for section, model in Settings.model_fields.items():
        if section in ("topics", "sources"):
            continue
        ann = model.annotation
        for key in ann.model_fields:
            assert key in doc[section], f"{section}.{key} missing from template"
    assert "unsorted" in text and "lower" in text and "raise" in text
    assert "[[sources]]" in text


def test_template_says_the_llm_timeout_only_raises_the_floor() -> None:
    """§17.4: the effective read timeout is max(setting, 60) for providers and max(setting,
    180) self-hosted; the comment must say so or a user lowers it and wonders why nothing
    changes."""
    lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("timeout_seconds"))
    comment = " ".join(lines[start : start + 3])
    assert "max(this, 60)" in comment and "max(this, 180)" in comment
    assert "never lower" in comment


def test_defaults_without_a_file_are_the_same_as_the_template(sf: SettingsFile) -> None:
    from_template = sf.load_sync().model_dump(exclude={"topics"})
    assert validate_settings({}).model_dump(exclude={"topics"}) == from_template


def test_write_template_copies_file_with_0600(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "settings.toml"
    write_template(target)
    assert target.read_text() == TEMPLATE_PATH.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


# --- validation sentences --------------------------------------------------------------------


def _error(data: dict) -> str:
    with pytest.raises(ConfigError) as info:
        validate_settings(data)
    return str(info.value)


def test_bad_timezone_names_key_and_fix() -> None:
    msg = _error({"general": {"timezone": "Mars/Olympus"}})
    assert "general.timezone" in msg and "IANA" in msg


def test_bad_topic_key() -> None:
    msg = _error({"topics": [{"key": "ML AI", "name": "x"}]})
    assert "topics[1].key" in msg and "a-z" in msg
    assert validate_settings({"topics": [{"key": "a1-b_c", "name": "x"}]}).topics[0].key == "a1-b_c"
    assert "topics[1].key" in _error({"topics": [{"key": "-bad", "name": "x"}]})
    assert "topics[1].key" in _error({"topics": [{"key": "a" * 33, "name": "x"}]})


def test_duplicate_topic_keys_and_names() -> None:
    msg = _error({"topics": [{"key": "a", "name": "A"}, {"key": "a", "name": "B"}]})
    assert "appears twice" in msg
    msg = _error({"topics": [{"key": "a", "name": "Same"}, {"key": "b", "name": "same"}]})
    assert "named" in msg


def test_folder_names_limited_to_12_characters() -> None:
    msg = _error({"folders": {"curated_name": "A" * 13}})
    assert "folders.curated_name" in msg and "12" in msg
    assert validate_settings({"folders": {"low_signal_name": "B" * 12}}).folders.low_signal_name


def test_trust_range() -> None:
    assert "sources[1].trust" in _error({"sources": [{"chat": "@x", "trust": 4}]})
    assert "sources[1].trust" in _error({"sources": [{"chat": "@x", "trust": -1}]})
    assert validate_settings({"sources": [{"chat": 123, "trust": 3}]}).sources[0].trust == 3


def test_duplicate_sources() -> None:
    msg = _error({"sources": [{"chat": "@kunuz"}, {"chat": "kunuz", "trust": 2}]})
    assert "listed twice" in msg


def test_weekday_and_time_ranges() -> None:
    assert "review.weekday" in _error({"review": {"weekday": "funday"}})
    assert validate_settings({"review": {"weekday": " Monday "}}).review.weekday == "monday"
    assert validate_settings({"review": {"weekday": "sunday"}}).review.weekday_index == 6
    assert "digest.hour" in _error({"digest": {"hour": 24}})
    assert "digest.minute" in _error({"digest": {"minute": 60}})
    assert "review.hour" in _error({"review": {"hour": -1}})


def test_literals() -> None:
    assert "llm.mode" in _error({"llm": {"mode": "cloud"}})
    assert "publishing.style" in _error({"publishing": {"style": "copy"}})
    msg = _error({"llm": {"mode": "selfhosted"}})
    assert "llm.base_url" in msg
    msg = _error({"llm": {"mode": "provider", "provider": "openai"}})
    assert "llm.model" in msg
    ok = validate_settings(
        {"llm": {"mode": "provider", "provider": "openai", "model": "m", "api_key": "k"}}
    )
    assert ok.llm.mode == "provider"


def test_numeric_ranges_name_the_key() -> None:
    assert "sorting.confidence" in _error({"sorting": {"confidence": 1.5}})
    assert "topics[1].strictness" in _error(
        {"topics": [{"key": "a", "name": "A", "strictness": 2}]}
    )
    assert "storage.database_url" in _error({"storage": {"database_url": "mysql://x"}})


def test_wrong_type_names_key() -> None:
    msg = _error({"digest": {"hour": "nine"}})
    assert "digest.hour" in msg


def test_several_errors_in_one_message() -> None:
    msg = _error({"digest": {"hour": 99}, "general": {"timezone": "Nope"}})
    assert "digest.hour" in msg and "general.timezone" in msg


def test_unknown_keys_warn_not_crash(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[general]\ntimezone = 'UTC'\ncolour = 'blue'\n[spaceship]\nx = 1\n"
        "[[topics]]\nkey = 'a'\nname = 'A'\nkeywords = ['never']\n"
    )
    with caplog.at_level(logging.WARNING, logger="tg_curator.config"):
        s = SettingsFile(path, env={}).load_sync()
    assert s.general.timezone == "UTC"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("general.colour" in m for m in msgs)
    assert any("spaceship" in m for m in msgs)
    assert any("topics[1].keywords" in m for m in msgs)


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[general\ntimezone = 'UTC'\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        SettingsFile(path, env={}).load_sync()


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    sf = SettingsFile(tmp_path / "nope.toml", env={})
    assert not sf.exists()
    with pytest.raises(ConfigError, match="not found"):
        sf.load_sync()
    with pytest.raises(ConfigError, match="not loaded"):
        _ = sf.current


# --- helpers on Settings ---------------------------------------------------------------------


def test_topic_and_source_for() -> None:
    s = validate_settings(
        {
            "topics": [{"key": "a", "name": "A"}],
            "sources": [{"chat": "@KunUz", "trust": 3}, {"chat": -1001, "trust": 0}],
        }
    )
    assert s.topic("a").name == "A"
    assert s.topic("zzz") is None
    assert s.source_for("kunuz").trust == 3
    assert s.source_for("@kunuz").trust == 3
    assert s.source_for(-1001).trust == 0
    assert s.source_for(-1002) is None
    assert s.source_for("other") is None


def test_missing_telegram() -> None:
    assert validate_settings({}).missing_telegram() == [
        "telegram.api_id",
        "telegram.api_hash",
        "telegram.bot_token",
    ]
    full = {"telegram": {"api_id": 1, "api_hash": "h", "bot_token": "t"}}
    assert validate_settings(full).missing_telegram() == []


def test_blank_strings_become_none_for_optional_topic_fields() -> None:
    s = validate_settings(
        {"topics": [{"key": "a", "name": "A", "category": "", "description": " "}]}
    )
    assert s.topics[0].category is None and s.topics[0].description is None


# --- home ------------------------------------------------------------------------------------


def test_resolve_home_precedence(tmp_path: Path) -> None:
    env = {"TG_CURATOR_HOME": str(tmp_path / "env")}
    assert resolve_home(str(tmp_path / "explicit"), env) == (tmp_path / "explicit").resolve()
    assert resolve_home(None, env) == (tmp_path / "env").resolve()
    assert resolve_home(None, {}) == Path("~/.tg-curator").expanduser().resolve()
    assert resolve_home("", env) == (tmp_path / "env").resolve()


def test_ensure_home_creates_0700(tmp_path: Path) -> None:
    home = ensure_home(tmp_path / "deep" / "home")
    assert home.is_dir()
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    ensure_home(home)  # idempotent


def test_ensure_home_chmod_failure_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(self: Path, mode: int) -> None:
        raise PermissionError("read-only mount")

    monkeypatch.setattr(Path, "chmod", boom)
    with caplog.at_level(logging.WARNING, logger="tg_curator.config"):
        ensure_home(tmp_path / "h")
    assert any("0700" in r.getMessage() for r in caplog.records)


# --- environment overrides and renamed keys ---------------------------------------------------


def test_env_values_are_written_into_blank_file_values(settings_path: Path) -> None:
    env = {
        "TG_CURATOR_API_ID": "12345",
        "TG_CURATOR_API_HASH": "abc",
        "TG_CURATOR_BOT_TOKEN": "123:token",
        "TG_CURATOR_DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
    }
    s = SettingsFile(settings_path, env=env).load_sync()
    assert s.telegram.api_id == 12345 and s.telegram.api_hash == "abc"
    assert s.telegram.bot_token == "123:token"
    assert s.storage.database_url.startswith("postgresql+asyncpg://")
    text = settings_path.read_text()
    assert "api_id = 12345" in text and 'bot_token = "123:token"' in text
    assert "# from my.telegram.org" in text  # comments survived the rewrite
    assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600
    # a later load without the environment keeps the written values
    assert SettingsFile(settings_path, env={}).load_sync().telegram.api_id == 12345


def test_env_does_not_override_non_blank_values(settings_path: Path) -> None:
    text = settings_path.read_text().replace("api_id = 0", "api_id = 777")
    settings_path.write_text(text)
    before = settings_path.stat().st_mtime_ns
    s = SettingsFile(settings_path, env={"TG_CURATOR_API_ID": "1"}).load_sync()
    assert s.telegram.api_id == 777
    assert settings_path.stat().st_mtime_ns == before  # nothing rewritten


def test_env_api_id_must_be_numeric(settings_path: Path) -> None:
    with pytest.raises(ConfigError, match="TG_CURATOR_API_ID"):
        SettingsFile(settings_path, env={"TG_CURATOR_API_ID": "abc"}).load_sync()


def test_renamed_keys_are_moved_with_a_notice(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(config.RENAMED_KEYS, "digest.length", "digest.items")
    text = settings_path.read_text().replace("items = 15", "length = 7")
    settings_path.write_text(text)
    with caplog.at_level(logging.INFO, logger="tg_curator.config"):
        s = SettingsFile(settings_path, env={}).load_sync()
    assert s.digest.items == 7
    new_text = settings_path.read_text()
    assert "length = 7" not in new_text and "items = 7" in new_text
    assert any("renamed" in r.getMessage() for r in caplog.records)


# --- SettingsFile writes ----------------------------------------------------------------------


async def test_load_and_update_keep_comments_and_are_atomic(sf: SettingsFile) -> None:
    await sf.load()
    assert sf.loaded
    s = await sf.update(lambda doc: doc["digest"].__setitem__("hour", 20))
    assert s.digest.hour == 20 and sf.current.digest.hour == 20
    text = sf.path.read_text()
    assert "hour = 20" in text and "# local hour the daily digest is sent" in text
    assert stat.S_IMODE(sf.path.stat().st_mode) == 0o600
    assert not [p for p in sf.path.parent.iterdir() if p.name.startswith(".settings.")]


async def test_set_value_creates_tables_and_removes_with_none(sf: SettingsFile) -> None:
    await sf.load()
    s = await sf.set_value("publishing.live", True)
    assert s.publishing.live is True
    s = await sf.set_value("telegram.owner_id", 42)
    assert s.telegram.owner_id == 42
    s = await sf.set_value("llm.api_key", None)
    assert s.llm.api_key == ""
    assert "api_key" not in sf.document()["llm"]


async def test_invalid_update_raises_and_leaves_file_untouched(sf: SettingsFile) -> None:
    await sf.load()
    before = sf.path.read_text()
    with pytest.raises(ConfigError, match="digest.hour"):
        await sf.set_value("digest.hour", 30)
    assert sf.path.read_text() == before
    assert sf.current.digest.hour == 21


async def test_upsert_and_remove_topic(sf: SettingsFile) -> None:
    await sf.load()
    s = await sf.upsert_topic("crypto", name="Crypto", category="crypto", channel="@mycrypto")
    assert s.topic("crypto").name == "Crypto" and s.topic("crypto").channel == "@mycrypto"
    s = await sf.upsert_topic("crypto", channel=-1001234, description=None)
    assert s.topic("crypto").channel == -1001234 and s.topic("crypto").category == "crypto"
    s = await sf.upsert_topic("ml-ai", strictness=0.7)
    assert s.topic("ml-ai").strictness == 0.7 and s.topic("ml-ai").name == "ML & AI"
    assert [t.key for t in s.topics] == ["ml-ai", "fintech", "uzbekistan", "football", "crypto"]
    s = await sf.remove_topic("fintech")
    assert s.topic("fintech") is None and len(s.topics) == 4
    s = await sf.remove_topic("never-there")
    assert len(s.topics) == 4
    with pytest.raises(ConfigError, match="topics"):
        await sf.upsert_topic("dup", name="ML & AI")


async def test_upsert_and_remove_source(sf: SettingsFile) -> None:
    await sf.load()
    s = await sf.upsert_source("@kunuz", 3)
    assert s.source_for("kunuz").trust == 3
    s = await sf.upsert_source("kunuz", 2)  # same chat, updated in place
    assert len(s.sources) == 1 and s.sources[0].trust == 2
    s = await sf.upsert_source(-1001, 0)
    assert len(s.sources) == 2
    with pytest.raises(ConfigError, match="trust"):
        await sf.upsert_source(-1001, 9)
    s = await sf.remove_source("@kunuz")
    assert [src.chat for src in s.sources] == [-1001]
    assert "[[sources]]" in sf.path.read_text()


async def test_on_change_is_called_after_every_update(sf: SettingsFile) -> None:
    seen: list[int] = []

    async def hook(settings: Settings) -> None:
        seen.append(settings.digest.hour)

    sf.on_change = hook
    await sf.load()
    assert seen == []  # a load is not a change
    await sf.set_value("digest.hour", 8)
    await sf.set_value("digest.hour", 9)
    assert seen == [8, 9]

    sync_seen: list[int] = []
    sf.on_change = lambda s: sync_seen.append(s.digest.hour)
    await sf.set_value("digest.hour", 10)
    assert sync_seen == [10]


async def test_reload_picks_up_hand_edits(sf: SettingsFile) -> None:
    await sf.load()
    sf.path.write_text(sf.path.read_text().replace("hour = 21", "hour = 6"))
    s = await sf.load()
    assert s.digest.hour == 6


def test_settings_file_defaults_to_process_environment(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TG_CURATOR_BOT_TOKEN", "1:xyz")
    assert "TG_CURATOR_BOT_TOKEN" in os.environ
    assert SettingsFile(settings_path).load_sync().telegram.bot_token == "1:xyz"
