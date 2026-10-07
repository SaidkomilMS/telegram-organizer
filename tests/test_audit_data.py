"""Audit checks on the shipped settings template (SPEC shipping: "ships fully commented")."""

from __future__ import annotations

import re
import tomllib

from tg_curator.config import PINNED_SECTIONS, TEMPLATE_PATH

_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _lines() -> list[str]:
    return TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()


def _header() -> str:
    """The leading comment block, joined into one line of prose."""
    text = []
    for line in _lines():
        if not line.startswith("#"):
            break
        text.append(line.lstrip("#").strip())
    return " ".join(part for part in text if part)


def test_header_says_bot_edits_apply_at_once() -> None:
    header = _header()
    assert "Edits made in the bot apply at once" in header


def test_header_says_reload_covers_all_but_pinned_sections() -> None:
    header = _header()
    assert "/reload" in header
    # The old wording limited /reload to topics and sources and told users to restart.
    assert "topics and sources can be reloaded" not in header
    assert "Settings take effect on the next start" not in header
    for name in PINNED_SECTIONS:
        assert f"[{name}]" in header, name
    assert "restart" in header


def test_every_key_is_explained() -> None:
    """Each key carries an inline comment or sits under a comment block; [[topics]] keys are
    explained by the topics legend above the first [[topics]] table."""
    lines = _lines()
    legend_start = next(i for i, line in enumerate(lines) if line.startswith("# Topics."))
    legend = "\n".join(lines[legend_start:])
    table = ""
    undocumented = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            table = stripped
            continue
        match = _KEY_LINE.match(line)
        if not match:
            continue
        key = match.group(1)
        if "#" in _STRING.sub("", line):
            continue
        previous = lines[index - 1].strip() if index else ""
        if previous.startswith("#"):
            continue
        if table == "[[topics]]" and re.search(rf"^#\s+{key}\s", legend, re.MULTILINE):
            continue
        undocumented.append(f"{index + 1}: {line}")
    assert not undocumented, undocumented


def test_digest_minute_has_comment() -> None:
    line = next(line for line in _lines() if line.startswith("minute ="))
    assert "#" in line


def test_template_still_parses_with_four_topics() -> None:
    data = tomllib.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert data["digest"]["minute"] == 0
    assert [topic["key"] for topic in data["topics"]] == [
        "ml-ai",
        "fintech",
        "uzbekistan",
        "football",
    ]
