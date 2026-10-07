"""The exception hierarchy shared by every module.

Telethon exceptions never leave the gateway implementations (DESIGN §5): they are translated
into the classes below at the boundary, so every other module - and every test using the fakes -
handles exactly one vocabulary of failures.
"""

from __future__ import annotations

from typing import Literal

NotAllowedReason = Literal["creator", "not_a_member", "fresh_session", "other"]
"""Why a subscription or channel write was refused.

``creator``: the account created the chat and cannot leave it.
``not_a_member``: an invite link to a chat the account is not in (never joined).
``fresh_session``: Telegram refuses admin changes from a session younger than a day
(``FRESH_CHANGE_ADMINS_FORBIDDEN``).
``other``: anything else; the message carries the detail.
"""

SessionLostReason = Literal[
    "unregistered", "invalid", "revoked", "expired", "deactivated", "banned", "duplicated", "other"
]
"""Why the account session stopped working (DESIGN §5).

``duplicated`` means the same session file was used from a second process at once
(``AUTH_KEY_DUPLICATED``); the others map one-to-one to Telethon's ``UnauthorizedError``
subclasses.
"""

LoginErrorReason = Literal[
    "bad_code", "expired_code", "bad_password", "bad_phone", "flood", "other"
]
"""Why a login step failed; rendered into a plain sentence by the bind flow and the CLI."""


class CuratorError(Exception):
    """Base class: anything the curator raises on purpose derives from this."""


class ConfigError(CuratorError):
    """A settings problem; the message names the key and says what to fix (DESIGN §4)."""


class TopicExists(CuratorError):
    """A topic with that name already exists; the caller offers to edit it instead."""


class NotOwnedError(CuratorError):
    """A content write targeted a chat that is not registered as owned, or a subscription
    write targeted one that is (DESIGN §1 guard rules)."""


class FloodWait(CuratorError):
    """Telegram asked to wait longer than the gateway sleeps itself (120 s)."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"Telegram asks to wait {seconds} s")
        self.seconds = seconds


class TelegramUnavailable(CuratorError):
    """Telegram could not be reached, or answered with a server-side error (5xx).

    Transient by nature: the caller does nothing differently, it simply tries again on its
    next tick. Kept apart from every other failure so the loop supervisor can log it once per
    outage instead of a traceback per tick, and so no caller mistakes an outage for a verdict
    about the chat (``ChatGone``) or the request (``NotAllowed``). An outage that lasts is
    reported to the owner by the watchdog (DESIGN §11.3, §17.3), not by whoever hit it.
    """


class ForwardsRestricted(CuratorError):
    """The source chat forbids saving content; only text and a link can be kept."""


class MediaUnavailable(CuratorError):
    """The media could not be fetched even after a re-fetch; fall back to text and link."""


class ChatGone(CuratorError):
    """The chat cannot be resolved any more (deleted, kicked, or never accessible)."""


class BotCannotPost(CuratorError):
    """The bot is not an admin with Post Messages in the target channel."""


class FolderLimit(CuratorError):
    """Telegram's folder limit is reached (``DIALOG_FILTERS_TOO_MUCH``)."""


class NotAllowed(CuratorError):
    """An action Telegram or our own rules refuse; ``reason`` is a ``NotAllowedReason``."""

    def __init__(self, reason: NotAllowedReason | str, detail: str = "") -> None:
        super().__init__(detail or f"not allowed: {reason}")
        self.reason = reason


class SessionLost(CuratorError):
    """The account session is no longer valid; ``reason`` is a ``SessionLostReason``."""

    def __init__(self, reason: SessionLostReason | str, detail: str = "") -> None:
        super().__init__(detail or f"session lost: {reason}")
        self.reason = reason


class LoginError(CuratorError):
    """A login step was refused; ``reason`` is a ``LoginErrorReason``."""

    def __init__(self, reason: LoginErrorReason | str, detail: str = "") -> None:
        super().__init__(detail or f"login failed: {reason}")
        self.reason = reason
