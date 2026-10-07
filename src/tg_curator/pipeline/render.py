"""Rendering of real-time posts, stubs and digest messages (DESIGN §8, §9.4, §9.5).

Everything a topic channel ever shows is composed here, as Telegram HTML, so the publisher
and the digest only decide *what* goes out and this module decides *how it reads*. Limits are
measured the way Telegram measures them: UTF-16 code units of the plain text after HTML
parsing (``textutil.split_html``).

Every word a channel shows is a catalogue key (``locales/<lang>/render.toml``, overridable in
``messages.toml``), so a translation reaches the topic channels and digests too (SPEC
"Translations of the bot's messages"). The functions take the translator as an optional
``t``; without one they use the English wording of §9.4/§9.5 below, which ``locales/en``
repeats. A sent digest part is found again by the header stored with its body (``digests.body``),
never by re-rendering, so a language change between a send and a reconcile sends nothing twice.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Literal

from tg_curator.domain import Chat, DigestLine, Post, RenderedPost
from tg_curator.telegram.links import permalink
from tg_curator.textutil import html_escape, split_html, utf16_len

CAPTION_LIMIT = 1024
"""Telegram's caption limit (plain text, UTF-16 units) for a single media message."""
TEXT_LIMIT = 4096
"""Telegram's text-message limit."""

COPYABLE_MEDIA = frozenset({"photo", "video", "file", "album"})
"""Media kinds the account can copy into the staging channel; ``other`` is only noted."""

SEP = " · "
SOURCE_LINK_TEXT = "source"
MORE_TEXT = "+{n} more"
DIGEST_LINK_TEXT = "link"
DIGEST_DAILY_TITLE = "Daily digest"
DIGEST_MANUAL_TITLE = "Digest (manual {seq})"
STUB_MOVED = "↪ moved to <b>{topic}</b>"
STUB_MOVED_SHORT = "↪ moved"
STUB_NOT_FOR_ME = "✕ not for me"
STUB_BELONGS = "↪ belongs in <b>{topic}</b>"
"""A moved forward's button message: the forward cannot be edited or deleted, so it stays and
the post is not published again in the right channel (§14.1)."""

_MEDIA_ICONS = {"photo": "📎", "video": "🎬", "file": "📎", "album": "🖼", "other": "📎"}
_MEDIA_NOUNS = {
    "photo": "photo",
    "video": "video",
    "file": "file",
    "album": "album",
    "other": "media",
}
_ATTACHED = "attached"
_AT_SOURCE = "at source"

_TAG_RE = re.compile(r"<[^>]+>")

Translate = Callable[..., str]
"""``rt.t``: a catalogue key plus ``str.format`` placeholders -> the text."""

ENGLISH: dict[str, str] = {
    "render_source": SOURCE_LINK_TEXT,
    "render_more": MORE_TEXT,
    "render_digest_link": DIGEST_LINK_TEXT,
    "render_digest_daily": DIGEST_DAILY_TITLE,
    "render_digest_manual": DIGEST_MANUAL_TITLE,
    "render_stub_moved": STUB_MOVED,
    "render_stub_moved_short": STUB_MOVED_SHORT,
    "render_stub_not_for_me": STUB_NOT_FOR_ME,
    "render_stub_belongs": STUB_BELONGS,
    "render_attached": _ATTACHED,
    "render_at_source": _AT_SOURCE,
    **{f"render_media_{kind}": noun for kind, noun in _MEDIA_NOUNS.items()},
}
"""The English wording, the same as ``locales/en/render.toml`` (a test keeps them equal)."""


def _say(t: Translate | None, key: str, **fmt: Any) -> str:
    if t is None:
        text = ENGLISH[key]
        return text.format(**fmt) if fmt else text
    return str(t(key, **fmt))


def more_reserve(t: Translate | None = None) -> int:
    """Room every real-time post keeps for its longest ``+N`` in the current wording."""
    return plain_units(SEP + _say(t, "render_more", n=999))


def plain_units(markup: str) -> int:
    """UTF-16 length of the text Telegram shows for ``markup`` (tags gone, entities decoded)."""
    return utf16_len(html.unescape(_TAG_RE.sub("", markup)))


MORE_RESERVE = plain_units(SEP + MORE_TEXT.format(n=999))
"""Room every real-time post keeps for its longest ``+N``: the split is computed as if the
header already carried `` · +999 more``, so a later ``+N`` edit always fits the messages that
were sent (§9.4 "+N edits")."""


# --- real-time posts -------------------------------------------------------------------------


def realtime_post(
    post: Post,
    chat: Chat,
    corroboration: int,
    *,
    caption: bool = False,
    media_attached: bool | None = None,
    t: Translate | None = None,
) -> RenderedPost:
    """The real-time post of §9.4: header line, media line, text, split for Telegram.

    ``parts`` are complete message bodies: when the text needs several messages the header
    (and media line) sit on the LAST part, because that is the message that carries the
    ``[Wrong topic]`` button and receives the ``+N`` edits (§9.4 "the message carrying the
    header and the button is always the LAST id"). ``caption=True`` splits for 1024 so the
    publisher can tell whether the whole post fits a single media caption.

    ``media_attached`` says whether the media travels with the post (``attached``) or stays at
    the source (``at source``); the default is what the source allows, the publisher passes
    ``False`` after a failed media copy.
    """
    header = _header(post, chat, corroboration, t)
    media_line = _media_line(post, media_attached, t)
    prefix = header if media_line is None else f"{header}\n{media_line}"
    body = post.html if post.html else html_escape(post.text)
    limit = CAPTION_LIMIT if caption else TEXT_LIMIT
    more = plain_units(SEP + _say(t, "render_more", n=corroboration)) if corroboration > 0 else 0
    parts = _parts(prefix, body, limit, pad=max(0, more_reserve(t) - more))
    return RenderedPost(header=header, media_line=media_line, parts=parts)


def forward_companion(
    post: Post, chat: Chat, corroboration: int, *, t: Translate | None = None
) -> str:
    """The bot's message that follows a forward-style post (§9.4): the header alone — source,
    link to the original and ``+N`` — under which the ``[Wrong topic]`` button sits, since a
    forwarded message can neither be edited nor carry the bot's buttons."""
    return _header(post, chat, corroboration, t)


def _header(post: Post, chat: Chat, corroboration: int, t: Translate | None = None) -> str:
    link = permalink(chat.id, chat.username, post.message_id)
    header = f"<b>{html_escape(chat.title)}</b>"
    if link is not None:
        header += f'{SEP}<a href="{link}">{_say(t, "render_source")}</a>'
    if corroboration > 0:
        header += SEP + _say(t, "render_more", n=corroboration)
    return header


def _media_line(post: Post, media_attached: bool | None, t: Translate | None = None) -> str | None:
    if post.media is None:
        return None
    kind = post.media if post.media in _MEDIA_ICONS else "other"
    if media_attached is None:
        media_attached = kind in COPYABLE_MEDIA and not post.noforwards
    state = _say(t, "render_attached" if media_attached else "render_at_source")
    return f"{_MEDIA_ICONS[kind]} {_say(t, f'render_media_{kind}')} {state}"


def _parts(prefix: str, body: str, limit: int, *, pad: int = 0) -> list[str]:
    """Split ``body`` for ``limit`` and put ``prefix`` (header + media line) on the last part.

    The last part is re-split when the prefix (plus ``pad`` units kept free for a longer
    ``+N``) no longer fits beside it, so every part stays within the limit, the prefix never
    has to be cut, and the split is the same whatever ``+N`` the header carries.
    """
    if not body.strip():
        return [prefix]
    parts = split_html(body, limit)
    if not parts:
        return [prefix]
    prefix_units = plain_units(prefix) + 1 + pad  # the newline between prefix and text
    if plain_units(parts[-1]) + prefix_units > limit:
        parts = parts[:-1] + split_html(parts[-1], max(1, limit - prefix_units))
    parts[-1] = f"{prefix}\n{parts[-1]}"
    return parts


# --- stubs -----------------------------------------------------------------------------------


def stub(
    kind: Literal["moved", "not_for_me", "belongs"],
    topic_name: str | None,
    *,
    t: Translate | None = None,
) -> str:
    """The one-line replacement for a moved or retracted message (§9.4, §14.1).

    ``("moved", None)`` is the short caption put on a media message that went out before its
    text; the text message gets the full stub naming the topic. ``"belongs"`` replaces the
    button message of a forward whose topic was corrected.
    """
    if kind == "not_for_me":
        return _say(t, "render_stub_not_for_me")
    if kind == "belongs":
        return _say(t, "render_stub_belongs", topic=html_escape(topic_name or ""))
    if topic_name is None:
        return _say(t, "render_stub_moved_short")
    return _say(t, "render_stub_moved", topic=html_escape(topic_name))


# --- the daily digest ------------------------------------------------------------------------


def digest_message(
    day: date,
    topic_name: str,
    items: list[DigestLine],
    chats: dict[int, Chat],
    *,
    seq: int = 0,
    posts: Mapping[int, Post] | None = None,
    t: Translate | None = None,
) -> list[str]:
    """The digest of §9.5 as message parts, each with its own unique header.

    Every part starts with ``<b>Daily digest</b> · {date} · {topic}`` (``Digest (manual N)``
    for a manual run) and, when split, ``(i/n)``, so the plain header is unique per (topic,
    day, seq, part) and ``reconcile()`` can find a part it already sent. Items are numbered
    in the given order and never cut in two (``split_html`` keeps ``\\n\\n`` boundaries).

    ``posts`` (``post_id -> Post``) supplies the source chat, permalink and ``+N`` of each
    item; a ``DigestLine`` alone carries none of them, so without ``posts`` an item is just
    its numbered line.
    """
    title = _say(t, "render_digest_daily") if seq == 0 else _say(t, "render_digest_manual", seq=seq)
    header = f"<b>{title}</b>{SEP}{day.isoformat()}{SEP}{html_escape(topic_name)}"
    lines = [
        _digest_item(n, item, chats, posts.get(item.post_id) if posts else None, t)
        for n, item in enumerate(items, 1)
    ]
    if not lines:
        return [header]
    body = "\n\n".join(lines)
    whole = f"{header}\n\n{body}"
    if plain_units(whole) <= TEXT_LIMIT:
        return [whole]
    reserve = plain_units(header) + len(" (99/99)") + 2
    parts = split_html(body, TEXT_LIMIT - reserve)
    n = len(parts)
    return [f"{header} ({i}/{n})\n\n{part}" for i, part in enumerate(parts, 1)]


def _digest_item(
    n: int,
    item: DigestLine,
    chats: dict[int, Chat],
    post: Post | None,
    t: Translate | None = None,
) -> str:
    text = f"{n}. {html_escape(item.line)}"
    if post is None:
        return text
    chat = chats.get(post.chat_id)
    pieces: list[str] = []
    if chat is not None:
        pieces.append(f"<b>{html_escape(chat.title)}</b>")
        link = permalink(chat.id, chat.username, post.message_id)
        if link is not None:
            pieces.append(f'<a href="{link}">{_say(t, "render_digest_link")}</a>')
    if post.corroboration > 0:
        pieces.append(f"+{post.corroboration}")
    if pieces:
        text += " — " + SEP.join(pieces)
    return text
