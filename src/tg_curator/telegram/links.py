"""Permalinks to Telegram messages (DESIGN §5, §15).

Marked ids are the convention everywhere in the curator: ``-100<bare>`` for channels and
supergroups, a plain negative number for basic groups, which have no message links at all.
"""

from __future__ import annotations

_CHANNEL_MARK = -1_000_000_000_000


def permalink(chat_id: int, username: str | None, message_id: int) -> str | None:
    """``https://t.me/<username>/<id>`` for public chats, ``https://t.me/c/<bare>/<id>`` for
    private channels and supergroups, ``None`` for basic groups (no link exists)."""
    if username:
        return f"https://t.me/{username.lstrip('@')}/{message_id}"
    if chat_id < _CHANNEL_MARK:
        return f"https://t.me/c/{_CHANNEL_MARK - chat_id}/{message_id}"
    if chat_id > 0:
        return f"https://t.me/c/{chat_id}/{message_id}"
    return None
