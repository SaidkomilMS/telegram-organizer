"""Audit fixes for the docs and packaging group.

WID-1: Postgres is optional "for people who already run it", on every documented install path.
The image ships the driver, and a missing driver is one clear settings error, not a traceback.

SHIP-7: "a developer should be able to read the whole thing in an afternoon" is backed by a
documented core reading path in CONTRIBUTING.md whose files exist and whose size fits a skim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tg_curator.db.store import Store
from tg_curator.errors import ConfigError, CuratorError

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "tg_curator"


# --- WID-1 -----------------------------------------------------------------------------------


async def test_a_missing_postgres_driver_is_one_clear_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "asyncpg", None)  # import asyncpg -> ModuleNotFoundError
    store = Store("postgresql+asyncpg://u:p@h/db")
    with pytest.raises(CuratorError) as caught:
        await store.start()
    assert isinstance(caught.value, ConfigError)  # exit 2: restarting would not help
    assert str(caught.value) == (
        "database_url points to PostgreSQL but the driver is missing: "
        "pip install 'tg-curator[postgres]'"
    )
    await store.close()


def test_the_docker_image_includes_the_postgres_driver() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'RUN pip install "/build[postgres]"' in text
    assert "asyncpg" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_install_docs_say_the_image_has_the_driver() -> None:
    text = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    section = text.split("## PostgreSQL (optional)", 1)[1].split("\n## ", 1)[0]
    assert "TG_CURATOR_DATABASE_URL" in section
    assert "Docker image already includes the driver" in " ".join(section.split())


# --- SHIP-7 ----------------------------------------------------------------------------------


def _reading_section() -> str:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    return text.split("## Reading the code", 1)[1].split("\n## ", 1)[0]


def test_contributing_promises_the_afternoon_for_the_core_path_only() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    intro = " ".join(text.split("\n## ", 1)[0].split())
    assert "core of tg-curator" in intro and "afternoon" in intro
    assert text.index("## The architecture in ten lines") < text.index("## Reading the code")


def test_the_core_reading_path_names_real_files_and_fits_a_skim() -> None:
    core, _, skip = _reading_section().partition("Safe to skip")
    core_files = re.findall(r"`([\w/]+\.py)`", core)
    assert core_files[:3] == ["domain.py", "contracts.py", "runtime.py"]
    assert core_files[-1] == "service.py"
    for name in core_files:
        assert (PKG / name).is_file(), name
    lines = sum(len((PKG / n).read_text(encoding="utf-8").splitlines()) for n in core_files)
    # The section says "about 7,000 lines"; keep it honest as the core grows.
    assert 5_000 <= lines <= 8_500, lines
    for name in re.findall(r"`([\w/*]+(?:\.py|/\*?))`", skip):
        base = ROOT if name.startswith("tests/") else PKG
        assert (base / name.rstrip("*").rstrip("/")).exists(), name
    skipped = [
        name
        for line in skip.splitlines()
        if line.startswith("- ")
        for name in re.findall(r"`([\w/*]+(?:\.py|/\*?))`", line.split(":", 1)[0])
    ]
    assert "telegram/user_client.py" in skipped and "db/store.py" in skipped
    assert not set(skipped) & set(core_files)
