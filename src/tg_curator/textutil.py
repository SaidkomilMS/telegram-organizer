"""Text helpers on module boundaries (DESIGN §8, §9.2, §9.4).

``normalise``/``text_hash`` decide "same text" for the repeat check, ``url_key`` decides "same
article link", and ``split_html`` cuts a rendered message into the fewest Telegram-sized parts
without ever breaking a tag, an entity or a digest item. Telegram measures its limits in UTF-16
code units of the plain text after HTML parsing, so every length here is measured that way.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit

# --- normalisation and hashing ---------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?",
)
_WS_RE = re.compile(r"\s+")
_DROP_CATEGORIES = ("P", "S", "Cf", "Co", "Cn", "Lm")
# Apostrophe-like marks are spelled many ways (Uzbek "oʻ" / "o'" / "o’"); they vanish without
# splitting the word so every spelling hashes the same.
_APOSTROPHES = frozenset("'`´ʻʼʹ‘’")


def _keep_char(ch: str) -> bool:
    if "︀" <= ch <= "️" or ch == "⃣":
        return False
    return not unicodedata.category(ch).startswith(_DROP_CATEGORIES)


def normalise(text: str) -> str:
    """Casefold, drop links, emoji, symbols and punctuation, collapse whitespace.

    The result is what two reposts of the same text have in common after a channel added its
    footer link, its emoji and its own quotes, so it is what the exact-repeat hash is taken of.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _URL_RE.sub(" ", text)
    text = "".join("" if ch in _APOSTROPHES else ch if _keep_char(ch) else " " for ch in text)
    return _WS_RE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    """sha1 of ``normalise(text)``; ``""`` when nothing is left (never matches anything)."""
    norm = normalise(text)
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# --- links -----------------------------------------------------------------------------------

TELEGRAM_HOSTS = frozenset({"t.me", "telegram.me", "telegram.dog"})
TRACKING_PARAMS = frozenset({"fbclid", "gclid", "yclid", "_ga"})


def _is_tracking(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("utm_") or lowered in TRACKING_PARAMS


def url_key(url: str) -> str | None:
    """The canonical form of an outside link, or ``None`` when it is not one.

    Scheme, ``www.``, fragment, tracking parameters, default ports and the trailing slash are
    removed and the host is lower-cased, so the same article reached through two share buttons
    gets one key. Telegram links and bare domains (a channel's footer) give ``None``: they say
    nothing about which story the post is about.
    """
    raw = url.strip()
    if not raw:
        return None
    if "://" not in raw:
        if raw.lower().startswith("tg:"):
            return None
        raw = "http://" + raw
    try:
        parts = urlsplit(raw)
        hostname = parts.hostname
        port = parts.port  # parsed lazily: a non-numeric or out-of-range port raises here
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    host = (hostname or "").lower().rstrip(".")
    if not host or "." not in host:
        return None
    if host.startswith("www."):
        host = host[4:]
    if host in TELEGRAM_HOSTS:
        return None
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    )
    if not path and not query:
        return None
    key = host + path
    if query:
        key += "?" + urlencode(query)
    return key


def canonical_urls(urls: Sequence[str]) -> list[str]:
    """``url_key`` over a list: outside links only, deduplicated, first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        key = url_key(url)
        if key is not None and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# --- plain text ------------------------------------------------------------------------------

ELLIPSIS = "…"


def first_line(text: str, max_chars: int) -> str:
    """The first non-blank line, whitespace collapsed, cut to ``max_chars`` with an ellipsis.

    This is the digest line when no language model is connected, so it prefers to cut at a
    word boundary when one sits in the last third of the allowed length.
    """
    if max_chars <= 0:
        return ""
    line = ""
    for candidate in text.splitlines():
        candidate = _WS_RE.sub(" ", candidate).strip()
        if candidate:
            line = candidate
            break
    if len(line) <= max_chars:
        return line
    if max_chars == 1:
        return ELLIPSIS
    cut = line[: max_chars - 1]
    if line[len(cut)] != " ":  # the cut lands inside a word: back up to the previous space
        space = cut.rfind(" ")
        if space >= (max_chars - 1) * 2 // 3:
            cut = cut[:space]
    return cut.rstrip() + ELLIPSIS


def html_escape(s: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for Telegram's HTML parse mode (quotes stay readable)."""
    return html.escape(s, quote=False)


def utf16_len(s: str) -> int:
    """Length in UTF-16 code units, the unit Telegram counts limits in."""
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in s)


# --- splitting -------------------------------------------------------------------------------

_TAG_RE = re.compile(r"""<(/?)([a-zA-Z][\w-]*)((?:"[^"]*"|'[^']*'|[^'">])*)>""")
_ENTITY_RE = re.compile(r"&(?:#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z][a-zA-Z0-9]*);")

_PARAGRAPH, _LINE, _SPACE, _ANY = 3, 2, 1, 0


@dataclass(frozen=True, slots=True)
class _Token:
    raw: str
    units: int  # plain-text UTF-16 units this token contributes
    kind: str  # "text" | "entity" | "open" | "close"
    name: str = ""


def _tokenise(markup: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(markup)
    while i < n:
        ch = markup[i]
        if ch == "<":
            m = _TAG_RE.match(markup, i)
            if m:
                closing, name, attrs = m.group(1), m.group(2).lower(), m.group(3)
                if attrs.rstrip().endswith("/"):
                    tokens.append(_Token(m.group(0), 0, "text"))
                else:
                    tokens.append(_Token(m.group(0), 0, "close" if closing else "open", name))
                i = m.end()
                continue
        elif ch == "&":
            m = _ENTITY_RE.match(markup, i)
            if m:
                tokens.append(_Token(m.group(0), utf16_len(html.unescape(m.group(0))), "entity"))
                i = m.end()
                continue
        tokens.append(_Token(ch, utf16_len(ch), "text"))
        i += 1
    return tokens


def _is_ws(tok: _Token) -> bool:
    return tok.kind == "text" and tok.raw.isspace()


def _priority(tokens: list[_Token], i: int) -> int:
    """How good a break *before* ``tokens[i]`` is."""
    prev = tokens[i - 1]
    if prev.kind != "text" or not prev.raw.isspace():
        return _ANY
    if prev.raw != "\n":
        return _SPACE
    j = i - 2
    while j >= 0 and tokens[j].kind in ("open", "close"):
        j -= 1
    if j >= 0 and tokens[j].kind == "text" and tokens[j].raw == "\n":
        return _PARAGRAPH
    return _LINE


def _apply_tag(stack: list[_Token], tok: _Token) -> None:
    if tok.kind == "open":
        stack.append(tok)
    elif tok.kind == "close":
        for k in range(len(stack) - 1, -1, -1):
            if stack[k].name == tok.name:
                del stack[k:]
                break


def _render(part: list[_Token], reopen: list[_Token]) -> tuple[str, list[_Token]]:
    """Join a part, dropping edge whitespace and balancing tags; returns (html, open stack)."""
    start, end = 0, len(part)
    while start < end and _is_ws(part[start]):
        start += 1
    while end > start and _is_ws(part[end - 1]):
        end -= 1
    body = part[start:end]
    stack = list(reopen)
    for tok in body:
        _apply_tag(stack, tok)
    # An open tag right at the end belongs to the next part, not to an empty "<b></b>": it is
    # left out here and handed on with the tags to reopen, so the next part starts with it.
    carried: list[_Token] = []
    while body and stack and body[-1] is stack[-1]:
        body.pop()
        carried.append(stack.pop())
        while body and _is_ws(body[-1]):
            body.pop()
    carry = stack + carried[::-1]
    if not any(t.units for t in body):
        return "", carry  # only whitespace and tags: nothing worth a message
    text = "".join(t.raw for t in reopen) + "".join(t.raw for t in body)
    text += "".join(f"</{t.name}>" for t in reversed(stack))
    return text, carry


def split_html(markup: str, limit: int) -> list[str]:
    """Cut HTML into the fewest parts whose plain text fits ``limit`` UTF-16 units each.

    A cut never lands inside a tag or an entity, never inside a ``"\\n\\n"``-separated item when
    an item boundary is available, and otherwise prefers a line end, then a space. Tags open at
    a cut are closed there and reopened in the next part so every part is valid on its own.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1 UTF-16 unit")
    tokens = _tokenise(markup)
    parts: list[str] = []
    reopen: list[_Token] = []
    start, n = 0, len(tokens)
    while start < n:
        units, end = 0, start
        while end < n and units + tokens[end].units <= limit:
            units += tokens[end].units
            end += 1
        if end == n:
            text, _ = _render(tokens[start:], reopen)
            if text:
                parts.append(text)
            break
        if end == start:
            end = start + 1  # a single oversized token still has to go somewhere
        best, best_prio = end, _ANY - 1
        for i in range(start + 1, end + 1):
            prio = _priority(tokens, i)
            if prio >= best_prio:
                best, best_prio = i, prio
        text, reopen = _render(tokens[start:best], reopen)
        if text:
            parts.append(text)
        start = best
    return parts
