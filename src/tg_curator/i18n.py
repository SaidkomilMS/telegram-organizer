"""The message catalogue (DESIGN §11.5).

Bot text never lives in code: every module owns ``locales/en/<module>.toml``. A chosen language
is merged over English, ``<home>/messages.toml`` is merged over everything, and a key that is
missing even in English comes back as the key itself with one warning, so a typo in a handler
is visible in the chat and in the log instead of crashing the bot.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).parent / "locales"
DEFAULT_LANGUAGE = "en"


def _load_dir(directory: Path) -> dict[str, str]:
    """Flat ``key = "text"`` pairs from every ``*.toml`` in ``directory`` (missing dir = {})."""
    merged: dict[str, str] = {}
    if not directory.is_dir():
        return merged
    for path in sorted(directory.glob("*.toml")):
        merged.update(_load_file(path))
    return merged


def _load_file(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.warning("messages file %s ignored: %s", path, exc)
        return {}
    flat: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            flat[key] = value
        else:
            log.warning("messages file %s: key %r is not a string, ignored", path, key)
    return flat


class Translator:
    """``t(key, **fmt)`` over the merged catalogue; callable so ``rt.t("key")`` reads well."""

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        home: Path | None = None,
        *,
        locales_dir: Path = LOCALES_DIR,
    ) -> None:
        self.language = language
        self._english = _load_dir(locales_dir / DEFAULT_LANGUAGE)
        self._messages = dict(self._english)
        if language != DEFAULT_LANGUAGE:
            self._messages.update(_load_dir(locales_dir / language))
        if home is not None:
            override = home / "messages.toml"
            if override.is_file():
                self._messages.update(_load_file(override))
        self._warned: set[str] = set()

    def __call__(self, key: str, **fmt: Any) -> str:
        return self.t(key, **fmt)

    def t(self, key: str, **fmt: Any) -> str:
        """The text for ``key`` with ``str.format`` placeholders filled; the key itself when
        it is unknown (warned once per key)."""
        text = self._messages.get(key)
        if text is None:
            if key not in self._warned:
                self._warned.add(key)
                log.warning("missing message key %r (not in locales/en)", key)
            return key
        if not fmt:
            return text
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError) as exc:
            log.warning("message %r could not be formatted: %s", key, exc)
            return text

    def has(self, key: str) -> bool:
        return key in self._messages

    def keys(self) -> set[str]:
        """Every key the catalogue knows; tests use it to assert bot modules use real keys."""
        return set(self._messages)

    def english_keys(self) -> set[str]:
        """Keys present in ``locales/en`` alone (overrides and translations excluded)."""
        return set(self._english)
