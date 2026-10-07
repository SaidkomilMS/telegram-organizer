"""Plain single-line logs a user can read in the system journal (DESIGN §1).

Telethon's DEBUG output contains request bodies, hence the WARNING floor for its logger: no
level the user picks for the curator can make a session or a login code appear in the journal.
"""

from __future__ import annotations

import logging
import sys

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger to write ``HH:MM:SS LEVEL logger: message`` lines on stderr."""
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("telethon").setLevel(logging.WARNING)


def mask_phone(phone: str | None) -> str:
    """``+998901234512`` -> ``+9989***12``: enough to recognise the number, not to dial it."""
    if not phone:
        return "***"
    digits = "".join(ch for ch in phone if ch.isdigit())
    prefix = "+" if phone.strip().startswith("+") else ""
    if len(digits) <= 4:
        return prefix + "***"
    return f"{prefix}{digits[:4]}***{digits[-2:]}"
