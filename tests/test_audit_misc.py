"""Spec-gap regressions of the "misc" audit group: first start with an empty settings file,
.gitignore coverage of homes inside the checkout, and the other small behaviours they name."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tg_curator import service
from tg_curator.config import TEMPLATE_PATH, SettingsFile

ROOT = Path(__file__).resolve().parents[1]


# --- SHIP-4: an empty settings file is a first start -----------------------------------------


@pytest.mark.parametrize("content", ["", "# hi\n\n# nothing here yet\n"])
async def test_an_empty_settings_file_prints_the_setup_steps(
    home: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    path = home / "settings.toml"
    path.write_text(content, encoding="utf-8")
    sf = SettingsFile(path, env={})
    assert await service.ready_settings(sf, wait=False) == 0
    out = capsys.readouterr().out
    assert "@BotFather" in out and "my.telegram.org" in out
    assert path.read_text(encoding="utf-8") == TEMPLATE_PATH.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == 0o600


async def test_an_empty_settings_file_with_the_values_in_the_environment_goes_on(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = home / "settings.toml"
    path.write_bytes(b"")
    env = {
        "TG_CURATOR_API_ID": "12345",
        "TG_CURATOR_API_HASH": "abc",
        "TG_CURATOR_BOT_TOKEN": "1:xyz",
    }
    settings = await service.ready_settings(SettingsFile(path, env=env), wait=True)
    assert not isinstance(settings, int)
    assert settings.telegram.api_id == 12345
    assert "Next steps" not in capsys.readouterr().out


async def test_a_broken_settings_file_is_not_mistaken_for_an_empty_one(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = home / "settings.toml"
    path.write_text("[telegram\n", encoding="utf-8")
    assert await service.ready_settings(SettingsFile(path, env={}), wait=False) == 2
    assert path.read_text(encoding="utf-8") == "[telegram\n"


# --- SAFE-10: a home inside the checkout never reaches a commit ------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_gitignore_covers_a_home_inside_the_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(ROOT / ".gitignore", repo / ".gitignore")
    files = [
        "settings.toml",
        "data/settings.toml",
        "data/user.session",
        "home/settings.toml",
        "src/tg_curator/locales/en/settings.toml",
        "src/tg_curator/config.py",
    ]
    for name in files:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked = {line[3:] for line in out.splitlines()}
    assert "src/tg_curator/locales/en/settings.toml" in tracked
    assert "src/tg_curator/config.py" in tracked
    assert not {"settings.toml", "data/settings.toml", "data/user.session"} & tracked
    assert "home/settings.toml" not in tracked


# --- WID-2: the timezone can come from the Docker environment --------------------------------


def test_the_timezone_comes_from_the_environment_until_the_file_names_one(home: Path) -> None:
    from tg_curator.config import write_template

    path = home / "settings.toml"
    write_template(path)
    env = {"TG_CURATOR_TIMEZONE": "Asia/Tashkent"}
    settings = SettingsFile(path, env=env).load_sync()
    assert settings.general.timezone == "Asia/Tashkent"
    assert 'timezone = "Asia/Tashkent"' in path.read_text(encoding="utf-8")
    # once the file names a zone of its own, the environment no longer replaces it
    text = path.read_text(encoding="utf-8").replace(
        'timezone = "Asia/Tashkent"', 'timezone = "Europe/Berlin"'
    )
    path.write_text(text, encoding="utf-8")
    assert SettingsFile(path, env=env).load_sync().general.timezone == "Europe/Berlin"


def test_a_bad_timezone_in_the_environment_gets_the_iana_sentence(home: Path) -> None:
    from tg_curator.config import write_template
    from tg_curator.errors import ConfigError

    path = home / "settings.toml"
    write_template(path)
    with pytest.raises(ConfigError, match="IANA"):
        SettingsFile(path, env={"TG_CURATOR_TIMEZONE": "Mars/Olympus"}).load_sync()
