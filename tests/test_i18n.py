"""i18n.py: English base, language merge, home overrides, missing-key behaviour."""

import logging
from pathlib import Path

import pytest

from tg_curator.i18n import LOCALES_DIR, Translator


def test_english_common_strings_load() -> None:
    t = Translator()
    assert t("yes") == "Yes"
    assert t("cancel") == "Cancel"
    assert "not_owner" in t.keys()
    assert t.has("error_generic")


def test_translator_is_callable_and_formats() -> None:
    t = Translator()
    assert t.t("error_generic", error="boom") == "Something went wrong: boom"
    assert t("error_generic", error="x") == t.t("error_generic", error="x")


def test_missing_key_returns_key_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    t = Translator()
    with caplog.at_level(logging.WARNING, logger="tg_curator.i18n"):
        assert t("no_such_key_xyz") == "no_such_key_xyz"
        assert t("no_such_key_xyz") == "no_such_key_xyz"
    warnings = [r for r in caplog.records if "no_such_key_xyz" in r.getMessage()]
    assert len(warnings) == 1


def test_bad_placeholder_does_not_crash(caplog: pytest.LogCaptureFixture) -> None:
    t = Translator()
    with caplog.at_level(logging.WARNING, logger="tg_curator.i18n"):
        assert t("error_generic") == "Something went wrong: {error}"  # no fmt: text as is
        assert t("error_generic", wrong="x") == "Something went wrong: {error}"
    assert any("could not be formatted" in r.getMessage() for r in caplog.records)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_language_merges_over_english_and_falls_back(tmp_path: Path) -> None:
    locales = tmp_path / "locales"
    _write(locales / "en" / "common.toml", 'yes = "Yes"\nno = "No"\n')
    _write(locales / "en" / "topics.toml", 'added = "Topic {name} added"\n')
    _write(locales / "ru" / "common.toml", 'yes = "Да"\n')
    t = Translator("ru", locales_dir=locales)
    assert t("yes") == "Да"
    assert t("no") == "No"  # silent fallback to English
    assert t("added", name="X") == "Topic X added"
    assert t.keys() == {"yes", "no", "added"}
    assert t.english_keys() == {"yes", "no", "added"}


def test_home_messages_override_everything(tmp_path: Path) -> None:
    locales = tmp_path / "locales"
    _write(locales / "en" / "common.toml", 'yes = "Yes"\n')
    _write(locales / "ru" / "common.toml", 'yes = "Да"\n')
    home = tmp_path / "home"
    _write(home / "messages.toml", 'yes = "Ha"\nextra = "only here"\n')
    t = Translator("ru", home=home, locales_dir=locales)
    assert t("yes") == "Ha"
    assert t("extra") == "only here"


def test_unknown_language_is_plain_english(tmp_path: Path) -> None:
    t = Translator("xx", home=tmp_path)  # no locales/xx, no messages.toml
    assert t("yes") == "Yes"


def test_broken_messages_file_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    home = tmp_path
    _write(home / "messages.toml", "this is = not toml =\n")
    with caplog.at_level(logging.WARNING, logger="tg_curator.i18n"):
        t = Translator(home=home)
    assert t("yes") == "Yes"
    assert any("ignored" in r.getMessage() for r in caplog.records)


def test_non_string_values_are_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    locales = tmp_path / "locales"
    _write(locales / "en" / "common.toml", 'yes = "Yes"\nnumber = 5\n')
    with caplog.at_level(logging.WARNING, logger="tg_curator.i18n"):
        t = Translator(locales_dir=locales)
    assert t.keys() == {"yes"}


def test_shipped_english_catalogue_is_valid_toml() -> None:
    for path in (LOCALES_DIR / "en").glob("*.toml"):
        t = Translator(locales_dir=LOCALES_DIR)
        assert t.keys(), path
