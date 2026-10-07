"""Rendering of real-time posts, stubs and digests (DESIGN §9.4, §9.5)."""

from __future__ import annotations

from datetime import date
from typing import Any

from tests.fakes import START, hrefs, plain_text
from tg_curator.domain import Chat, DigestLine, Post, PostStatus
from tg_curator.pipeline import render
from tg_curator.pipeline.render import (
    CAPTION_LIMIT,
    TEXT_LIMIT,
    digest_message,
    plain_units,
    realtime_post,
    stub,
)
from tg_curator.textutil import text_hash, utf16_len

CHANNEL = -1_001_000_000_001
PUBLIC = -1_001_000_000_002
GROUP = -4_000_000_001  # a basic group: no permalinks


def chat(chat_id: int = CHANNEL, **over: Any) -> Chat:
    values: dict[str, Any] = {
        "id": chat_id,
        "kind": "channel",
        "title": "Tech & <News>",
        "first_seen_at": START,
    }
    values.update(over)
    return Chat(**values)


def post(text: str = "Hello world, a post.", **over: Any) -> Post:
    values: dict[str, Any] = {
        "id": 1,
        "chat_id": CHANNEL,
        "message_id": 42,
        "kind": "post",
        "message_ids": [42],
        "posted_at": START,
        "ingested_at": START,
        "via": "live",
        "text": text,
        "text_hash": text_hash(text),
        "urls": [],
        "status": PostStatus.queued,
    }
    values.update(over)
    return Post(**values)


# --- real-time posts -------------------------------------------------------------------------


def test_header_has_title_link_and_corroboration() -> None:
    rendered = realtime_post(post(), chat(), 3)
    assert rendered.header == (
        '<b>Tech &amp; &lt;News&gt;</b> · <a href="https://t.me/c/1000000001/42">source</a>'
        " · +3 more"
    )
    assert rendered.media_line is None
    assert rendered.parts == [rendered.header + "\nHello world, a post."]


def test_no_plus_n_when_nothing_corroborated() -> None:
    rendered = realtime_post(post(), chat(), 0)
    assert "more" not in rendered.header


def test_public_channel_links_by_username() -> None:
    rendered = realtime_post(post(chat_id=PUBLIC), chat(PUBLIC, username="@kunuz"), 0)
    assert hrefs(rendered.header) == ["https://t.me/kunuz/42"]


def test_basic_group_has_title_without_link() -> None:
    rendered = realtime_post(post(chat_id=GROUP), chat(GROUP, kind="group", title="Chat"), 0)
    assert rendered.header == "<b>Chat</b>"
    assert hrefs(rendered.header) == []


def test_html_text_is_used_verbatim_and_plain_text_is_escaped() -> None:
    html = 'Read <a href="https://example.org/a">this</a> &amp; that'
    rendered = realtime_post(post(html=html), chat(), 0)
    assert rendered.parts[0].endswith("\n" + html)
    rendered = realtime_post(post(text="a < b & c"), chat(), 0)
    assert rendered.parts[0].endswith("\na &lt; b &amp; c")


def test_media_lines_per_kind() -> None:
    for kind, line in [
        ("photo", "📎 photo attached"),
        ("video", "🎬 video attached"),
        ("file", "📎 file attached"),
        ("album", "🖼 album attached"),
    ]:
        rendered = realtime_post(post(media=kind), chat(), 0)
        assert rendered.media_line == line
        assert rendered.parts[0].split("\n")[1] == line


def test_protected_source_says_at_source() -> None:
    rendered = realtime_post(post(media="photo", noforwards=True), chat(), 0)
    assert rendered.media_line == "📎 photo at source"


def test_other_media_is_only_noted() -> None:
    rendered = realtime_post(post(media="other"), chat(), 0)
    assert rendered.media_line == "📎 media at source"


def test_media_attached_flag_overrides_the_default() -> None:
    rendered = realtime_post(post(media="video"), chat(), 0, media_attached=False)
    assert rendered.media_line == "🎬 video at source"
    rendered = realtime_post(post(media="video", noforwards=True), chat(), 0, media_attached=True)
    assert rendered.media_line == "🎬 video attached"


def test_caption_split_at_1024_and_text_at_4096() -> None:
    body = " ".join(f"w{i}" for i in range(400))  # ~2000 units
    assert len(realtime_post(post(text=body), chat(), 0).parts) == 1
    parts = realtime_post(post(text=body), chat(), 0, caption=True).parts
    assert len(parts) > 1
    assert all(plain_units(p) <= CAPTION_LIMIT for p in parts)


def test_header_sits_on_the_last_part_of_a_long_text() -> None:
    body = "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(40))
    rendered = realtime_post(post(text=body, media="photo"), chat(), 2)
    assert len(rendered.parts) > 1
    assert all(plain_units(p) <= TEXT_LIMIT for p in rendered.parts)
    assert rendered.parts[-1].startswith(rendered.header + "\n📎 photo attached\n")
    assert all(not p.startswith("<b>") for p in rendered.parts[:-1])
    joined = plain_text("\n".join(rendered.parts))
    assert "paragraph 0 " in joined and "paragraph 39 " in joined


def test_last_part_is_resplit_when_the_header_does_not_fit() -> None:
    body = "y" * (TEXT_LIMIT - 10)  # fits alone, not with a header
    rendered = realtime_post(post(text=body), chat(), 0)
    assert len(rendered.parts) == 2
    assert all(plain_units(p) <= TEXT_LIMIT for p in rendered.parts)
    assert rendered.parts[-1].startswith(rendered.header)
    assert sum(plain_text(p).count("y") for p in rendered.parts) == TEXT_LIMIT - 10


def test_empty_text_still_renders_the_header() -> None:
    rendered = realtime_post(post(text=""), chat(), 0)
    assert rendered.parts == [rendered.header]


def test_tags_are_not_cut_and_count_nothing() -> None:
    html = "<b>" + "é" * 4000 + "</b> " + "<i>" + "ü" * 500 + "</i>"
    rendered = realtime_post(post(html=html), chat(), 0)
    for part in rendered.parts:
        assert plain_units(part) <= TEXT_LIMIT
        assert part.count("<b>") == part.count("</b>")
        assert part.count("<i>") == part.count("</i>")


def test_plain_units_counts_utf16() -> None:
    assert plain_units("<b>a&amp;b</b>") == 3
    assert plain_units("😀") == 2 == utf16_len("😀")


# --- stubs -----------------------------------------------------------------------------------


def test_stub_texts() -> None:
    assert stub("moved", "ML & AI") == "↪ moved to <b>ML &amp; AI</b>"
    assert stub("moved", None) == "↪ moved"
    assert stub("not_for_me", None) == "✕ not for me"
    assert stub("not_for_me", "ignored") == "✕ not for me"


# --- digests ---------------------------------------------------------------------------------


def _digest_fixture(n: int, *, length: int = 20) -> tuple[list[DigestLine], dict, dict]:
    chats = {
        CHANNEL: chat(),
        PUBLIC: chat(PUBLIC, username="kunuz", title="Kun.uz"),
        GROUP: chat(GROUP, kind="group", title="Group"),
    }
    items, posts = [], {}
    for i in range(1, n + 1):
        cid = [CHANNEL, PUBLIC, GROUP][i % 3]
        posts[i] = post(id=i, chat_id=cid, message_id=100 + i, corroboration=i % 2)
        items.append(DigestLine(position=i, post_id=i, line=f"Line {i} " + "z" * length))
    return items, chats, posts


def test_digest_single_part_has_plain_header_and_numbered_items() -> None:
    items, chats, posts = _digest_fixture(3)
    parts = digest_message(date(2026, 10, 5), "ML & AI", items, chats, posts=posts)
    assert len(parts) == 1
    lines = parts[0].split("\n\n")
    assert lines[0] == "<b>Daily digest</b> · 2026-10-05 · ML &amp; AI"
    # item 1: public channel with +1
    assert lines[1].startswith("1. Line 1 ")
    assert '<b>Kun.uz</b> · <a href="https://t.me/kunuz/101">link</a> · +1' in lines[1]
    # item 2: basic group, no link, no +N
    assert lines[2].endswith("2. Line 2 " + "z" * 20 + " — <b>Group</b>")
    # item 3: private channel
    assert '<a href="https://t.me/c/1000000001/103">link</a> · +1' in lines[3]


def test_digest_manual_header_carries_the_seq() -> None:
    items, chats, posts = _digest_fixture(1)
    parts = digest_message(date(2026, 10, 5), "Fintech", items, chats, seq=2, posts=posts)
    assert parts[0].startswith("<b>Digest (manual 2)</b> · 2026-10-05 · Fintech\n\n")


def test_digest_without_posts_renders_lines_only() -> None:
    items, chats, _ = _digest_fixture(2)
    parts = digest_message(date(2026, 10, 5), "T", items, chats)
    assert parts[0].split("\n\n")[1] == "1. Line 1 " + "z" * 20


def test_digest_lines_are_escaped() -> None:
    items = [DigestLine(position=1, post_id=1, line="a <b> & c")]
    parts = digest_message(date(2026, 10, 5), "T", items, {})
    assert "1. a &lt;b&gt; &amp; c" in parts[0]


def test_empty_digest_is_just_the_header() -> None:
    assert digest_message(date(2026, 10, 5), "T", [], {}) == [
        "<b>Daily digest</b> · 2026-10-05 · T"
    ]


def test_digest_split_has_unique_headers_and_whole_items() -> None:
    items, chats, posts = _digest_fixture(15, length=600)
    parts = digest_message(date(2026, 10, 5), "ML & AI", items, chats, posts=posts)
    assert len(parts) >= 2
    headers = [plain_text(p.split("\n\n")[0]) for p in parts]
    assert len(set(headers)) == len(parts)
    for i, header in enumerate(headers, 1):
        assert header == f"Daily digest · 2026-10-05 · ML & AI ({i}/{len(parts)})"
    assert all(plain_units(p) <= TEXT_LIMIT for p in parts)
    numbers = [line.split(".")[0] for p in parts for line in p.split("\n\n")[1:] if line.strip()]
    assert numbers == [str(i) for i in range(1, 16)]


def test_copyable_media_kinds() -> None:
    assert render.COPYABLE_MEDIA == {"photo", "video", "file", "album"}


# --- translations (SPEC: "Translations of the bot's messages are a settings file") -------------


def test_the_english_wording_is_the_shipped_render_catalogue() -> None:
    import tomllib

    from tg_curator.i18n import LOCALES_DIR

    with (LOCALES_DIR / "en" / "render.toml").open("rb") as fh:
        assert tomllib.load(fh) == render.ENGLISH


def test_a_messages_override_reaches_posts_stubs_and_digests(tmp_path: Any) -> None:
    from tg_curator.i18n import Translator

    (tmp_path / "messages.toml").write_text(
        "\n".join(
            [
                'render_source = "manba"',
                'render_more = "+{n} yana"',
                'render_attached = "ilova"',
                'render_media_photo = "rasm"',
                'render_stub_moved = "↪ ko‘chirildi: <b>{topic}</b>"',
                'render_stub_not_for_me = "✕ menga emas"',
                'render_digest_daily = "Kunlik dayjest"',
                'render_digest_link = "havola"',
            ]
        ),
        encoding="utf-8",
    )
    t = Translator(home=tmp_path)
    p = post(media="photo")
    rendered = render.realtime_post(p, chat(), 2, t=t)
    assert "manba" in rendered.header and "+2 yana" in rendered.header
    assert rendered.media_line == "📎 rasm ilova"
    assert "source" not in rendered.header
    assert render.forward_companion(p, chat(), 1, t=t).endswith("+1 yana")
    assert render.stub("moved", "ML", t=t) == "↪ ko‘chirildi: <b>ML</b>"
    assert render.stub("not_for_me", None, t=t) == "✕ menga emas"
    [part] = render.digest_message(
        date(2026, 10, 5),
        "ML",
        [DigestLine(1, p.id, "A line")],
        {CHANNEL: chat()},
        posts={p.id: p},
        t=t,
    )
    assert part.startswith("<b>Kunlik dayjest</b>") and ">havola</a>" in part
    # without a translator the English wording of §9.4/§9.5 is used
    assert render.stub("not_for_me", None) == "✕ not for me"
    assert render.realtime_post(p, chat(), 2).header.endswith("+2 more")
