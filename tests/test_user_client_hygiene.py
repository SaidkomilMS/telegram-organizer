"""The account never does what DESIGN §5 forbids: enforced by reading ``user_client.py``.

A grep is deliberately blunt. The forbidden identifiers must not appear anywhere in the
module, not even in a comment, so a future edit cannot smuggle one in behind a docstring;
``send_file`` is allowed exactly once, inside ``copy_media``, where it re-sends media by
reference into an owned chat.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import tg_curator.telegram.user_client as module

SOURCE_PATH = Path(module.__file__)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")

FORBIDDEN = [
    "send_message",
    "send_read_acknowledge",
    "mark_read",
    "ReadHistoryRequest",
    "ReadMentionsRequest",
    "SendReactionRequest",
    "JoinChannelRequest",
    "ImportChatInviteRequest",
    "UpdateProfileRequest",
    "UpdateUsernameRequest",
    "UpdateStatusRequest",
    "delete_dialog",
    "delete_messages",
    "DeleteHistoryRequest",
    "increment=True",
    "catch_up=True",
]


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_forbidden_call_is_absent(needle: str) -> None:
    assert needle not in SOURCE, f"{needle!r} must not appear in user_client.py"


def test_send_file_only_inside_copy_media() -> None:
    occurrences = [m.start() for m in re.finditer(r"\bsend_file\(", SOURCE)]
    assert len(occurrences) == 1
    tree = ast.parse(SOURCE)
    copy_media = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "copy_media"
    )
    segment = ast.get_source_segment(SOURCE, copy_media)
    assert segment is not None and "send_file(" in segment
    lines = SOURCE.splitlines()
    line_no = SOURCE[: occurrences[0]].count("\n") + 1
    assert copy_media.lineno <= line_no <= copy_media.end_lineno, lines[line_no - 1]


def test_views_are_read_without_incrementing_and_nothing_is_replayed() -> None:
    assert "increment=False" in SOURCE
    assert "catch_up=False" in SOURCE
    assert "receive_updates=True" in SOURCE
    assert "flood_sleep_threshold=FLOOD_SLEEP_THRESHOLD" in SOURCE
    assert module.FLOOD_SLEEP_THRESHOLD == 120


def test_only_the_two_client_modules_import_telethon() -> None:
    package = SOURCE_PATH.parent.parent
    offenders = []
    for path in package.rglob("*.py"):
        if path.name in {"user_client.py", "bot_client.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(?:from|import)\s+telethon", text, re.MULTILINE):
            offenders.append(path.relative_to(package).as_posix())
    assert offenders == []


def test_bot_admin_rights_are_post_and_edit_only() -> None:
    rights = re.findall(r"ChatAdminRights\((.*?)\)", SOURCE, re.DOTALL)
    assert rights, "add_bot_admin must build a ChatAdminRights"
    for args in rights:
        granted = {part.split("=")[0].strip() for part in args.split(",") if part.strip()}
        assert granted == {"post_messages", "edit_messages"}
