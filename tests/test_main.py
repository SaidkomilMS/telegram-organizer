"""``python -m tg_curator``: never a traceback, whether or not the CLI module is installed."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_python_m_tg_curator_fails_in_one_sentence_without_the_cli(tmp_path: Path) -> None:
    env = {**os.environ, "TG_CURATOR_HOME": str(tmp_path / "home")}
    result = subprocess.run(
        [sys.executable, "-m", "tg_curator"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert "Traceback" not in result.stderr, result.stderr
    if importlib.util.find_spec("tg_curator.cli") is None:
        assert result.returncode == 1
        assert "tg_curator.cli" in result.stderr and result.stderr.count("\n") == 1
