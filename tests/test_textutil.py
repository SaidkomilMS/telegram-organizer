"""textutil.py: normalisation, hashing, link keys, first line, HTML escaping and splitting."""

import html
import re

import pytest

from tg_curator.textutil import (
    canonical_urls,
    first_line,
    html_escape,
    normalise,
    split_html,
    text_hash,
    url_key,
    utf16_len,
)

# --- normalise / text_hash -------------------------------------------------------------------


def test_normalise_casefolds_and_collapses_whitespace() -> None:
    assert normalise("  Hello   WORLD\n\tagain ") == "hello world again"


def test_normalise_strips_urls_emoji_and_punctuation() -> None:
    text = "🚀 Breaking: OpenAI released a model! https://example.com/a?x=1 #ai @channel 🔥🔥"
    assert normalise(text) == "breaking openai released a model ai channel"


def test_normalise_russian_and_uzbek() -> None:
    ru = "«Привет, мир!» — сказал он… 😀"
    assert normalise(ru) == "привет мир сказал он"
    uz_a = "Oʻzbekiston Respublikasi: yangi qonun qabul qilindi."
    uz_b = "O'ZBEKISTON RESPUBLIKASI — yangi qonun qabul qilindi"
    assert normalise(uz_a) == normalise(uz_b) == "ozbekiston respublikasi yangi qonun qabul qilindi"


def test_normalise_drops_footer_links_and_domains() -> None:
    a = "Курс доллара вырос на 2%\n\nkun.uz/news/123 | t.me/kunuz | https://kun.uz"
    b = "Курс доллара вырос на 2%"
    assert normalise(a) == normalise(b) == "курс доллара вырос на 2"
    assert normalise("Подробнее: kun.uz/news/123") == "подробнее"


def test_normalise_keeps_digits_and_letters_of_all_scripts() -> None:
    assert normalise("Bitcoin $65,000 → 2026") == "bitcoin 65 000 2026"
    assert normalise("日本語テキスト") == "日本語テキスト"


def test_text_hash_is_stable_and_empty_for_nothing() -> None:
    assert text_hash("Hello, world!") == text_hash("hello world")
    assert len(text_hash("x")) == 40
    assert text_hash("") == ""
    assert text_hash("🔥🔥 ... !!! https://t.me/x") == ""


# --- url_key / canonical_urls ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.example.com/article/", "example.com/article"),
        ("http://Example.COM/Article", "example.com/Article"),
        ("https://example.com/a?utm_source=tg&utm_medium=x", "example.com/a"),
        ("https://example.com/a?b=2&a=1&fbclid=zzz", "example.com/a?a=1&b=2"),
        ("https://example.com/a?gclid=1&yclid=2&_ga=3&UTM_CAMPAIGN=4", "example.com/a"),
        ("https://example.com/a#section", "example.com/a"),
        ("https://example.com:443/a", "example.com/a"),
        ("https://example.com:8080/a", "example.com:8080/a"),
        ("example.com/path", "example.com/path"),
        ("https://kun.uz/news/2026/10/05/sarlavha?id=7", "kun.uz/news/2026/10/05/sarlavha?id=7"),
        ("https://example.com/?page=2", "example.com?page=2"),
    ],
)
def test_url_key_canonical(url: str, expected: str) -> None:
    assert url_key(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://t.me/kunuz/123",
        "http://telegram.me/joinchat/abc",
        "https://telegram.dog/x/1",
        "https://t.me/+AbCdEf",
        "https://example.com",
        "https://example.com/",
        "www.example.com",
        "example.com",
        "https://example.com/?utm_source=x#frag",
        "tg://resolve?domain=x",
        "mailto:someone@example.com",
        "",
        "   ",
        "not a url",
    ],
)
def test_url_key_none_for_telegram_bare_and_invalid(url: str) -> None:
    assert url_key(url) is None


def test_canonical_urls_dedupes_and_keeps_order() -> None:
    urls = [
        "https://t.me/kunuz/1",
        "https://www.example.com/b?utm_source=a",
        "https://example.com/a",
        "http://example.com/b/",
        "https://example.com/a#x",
    ]
    assert canonical_urls(urls) == ["example.com/b", "example.com/a"]
    assert canonical_urls([]) == []


# --- first_line / html_escape ----------------------------------------------------------------


def test_first_line_skips_blank_lines_and_collapses() -> None:
    assert first_line("\n  \n  Первая   строка \nвторая", 100) == "Первая строка"
    assert first_line("", 10) == ""
    assert first_line("\n\n", 10) == ""


def test_first_line_cuts_with_ellipsis_at_word_boundary() -> None:
    line = first_line("Markaziy bank asosiy stavkani 14 foizda saqlab qoldi", 30)
    assert len(line) <= 30
    assert line.endswith("…")
    assert line == "Markaziy bank asosiy stavkani…"
    assert first_line("Markaziy bank asosiy stavkani 14 foizda", 27) == "Markaziy bank asosiy…"


def test_first_line_hard_cuts_long_words() -> None:
    line = first_line("a" * 50, 10)
    assert line == "a" * 9 + "…"
    assert first_line("abc", 0) == ""
    assert first_line("abcdef", 1) == "…"


def test_html_escape() -> None:
    assert html_escape("<b> & \"q\" 'a'") == "&lt;b&gt; &amp; \"q\" 'a'"


def test_utf16_len_counts_surrogate_pairs() -> None:
    assert utf16_len("abc") == 3
    assert utf16_len("😀") == 2
    assert utf16_len("Привет") == 6


# --- split_html ------------------------------------------------------------------------------


def _plain(part: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", part))


def _assert_balanced(part: str) -> None:
    stack: list[str] = []
    for closing, name in re.findall(r"<(/?)([a-zA-Z][\w-]*)[^>]*>", part):
        if closing:
            assert stack and stack[-1] == name.lower(), part
            stack.pop()
        else:
            stack.append(name.lower())
    assert not stack, part


def _assert_valid_split(markup: str, parts: list[str], limit: int) -> None:
    assert parts
    for part in parts:
        plain = _plain(part)
        assert plain.strip(), part  # never a part that is only tags
        assert utf16_len(plain) <= limit, (utf16_len(plain), part)
        _assert_balanced(part)
        assert not re.search(r"&[#\w]+$", plain)  # no entity cut in half
    joined = re.sub(r"\s+", "", "".join(_plain(p) for p in parts))
    assert joined == re.sub(r"\s+", "", _plain(markup))


def test_split_short_text_is_one_part() -> None:
    assert split_html("<b>Hi</b> there", 100) == ["<b>Hi</b> there"]
    assert split_html("", 100) == []
    assert split_html("   ", 100) == []


def test_split_prefers_digest_item_boundaries() -> None:
    items = [
        f'{i}. Item number {i} — <b>Source</b> · <a href="https://t.me/c/1/{i}">link</a>'
        for i in range(1, 7)
    ]
    markup = "<b>Daily digest</b> · ML & AI\n\n" + "\n\n".join(items)
    plain_items = [_plain(i) for i in items]
    limit = utf16_len(plain_items[0]) * 2 + 10
    parts = split_html(markup, limit)
    _assert_valid_split(markup, parts, limit)
    assert len(parts) >= 3
    for part in parts:
        # every item in a part is whole: each line is exactly one rendered item or the header
        for line in _plain(part).split("\n\n"):
            assert line in plain_items or line == "Daily digest · ML & AI", line


def test_split_falls_back_to_lines_then_spaces_then_anything() -> None:
    markup = "line one is here\nline two is here\nline three is here"
    parts = split_html(markup, 20)
    assert parts == ["line one is here", "line two is here", "line three is here"]
    parts = split_html("word " * 10, 12)
    _assert_valid_split("word " * 10, parts, 12)
    assert all(_plain(p).endswith("word") for p in parts)
    parts = split_html("x" * 25, 10)
    assert parts == ["x" * 10, "x" * 10, "x" * 5]


def test_split_never_inside_tag_or_entity_and_reopens_tags() -> None:
    markup = "<b>bold <i>and italic &amp; &lt;escaped&gt; text</i> continues</b> plain tail"
    for limit in range(2, 40):
        parts = split_html(markup, limit)
        _assert_valid_split(markup, parts, limit)
    parts = split_html(markup, 12)
    assert parts[0].startswith("<b>")
    assert parts[1].startswith("<b><i>") or parts[1].startswith("<b>")


def test_split_counts_utf16_units_of_plain_text_not_markup() -> None:
    markup = '<a href="https://example.com/very/long/link/that/does/not/count">😀😀😀</a>'
    assert split_html(markup, 6) == [markup]
    parts = split_html(markup, 4)
    assert len(parts) == 2
    _assert_valid_split(markup, parts, 4)
    parts = split_html("😀" * 5, 3)  # 3 units: one emoji per part, never half a pair
    assert parts == ["😀"] * 5


def test_split_keeps_href_when_reopening_links() -> None:
    markup = '<a href="https://kun.uz/x">Первая часть ссылки и вторая часть</a>'
    parts = split_html(markup, 15)
    assert len(parts) > 1
    assert all(p.startswith('<a href="https://kun.uz/x">') for p in parts)
    _assert_valid_split(markup, parts, 15)


def test_split_long_mixed_input_is_fewest_parts() -> None:
    paragraph = (
        "Markaziy bank asosiy stavkani 14 foizda saqlab qoldi 📈. "
        "Центральный банк сохранил ставку на уровне 14% — <b>Kun.uz</b>. "
        "The central bank kept its key rate at 14% &amp; signalled no change.\n"
    )
    markup = "<b>Header</b>\n\n" + "\n\n".join(paragraph * 3 for _ in range(40))
    limit = 4096
    parts = split_html(markup, limit)
    _assert_valid_split(markup, parts, limit)
    total = utf16_len(_plain(markup))
    assert len(parts) <= total // limit + 2
    assert all(utf16_len(_plain(p)) > limit // 2 for p in parts[:-1])


def test_split_rejects_zero_limit() -> None:
    with pytest.raises(ValueError):
        split_html("x", 0)


def test_split_trims_whitespace_at_cuts_and_keeps_inner_newlines() -> None:
    markup = "<b>Title</b>\nbody line\n\n\n\nnext item"
    parts = split_html(markup, 20)
    assert parts == ["<b>Title</b>\nbody line", "next item"]
