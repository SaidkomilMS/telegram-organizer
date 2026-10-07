"""The similarity model and the cheap text helpers the dedup rule needs (DESIGN §10, §9.2).

Everything a post goes through before it becomes a vector lives here, so the live sorter, the
research scripts and the tests embed exactly the same string: links, mentions, hashtags and
emoji are dropped (a channel's footer must not make two different stories look alike), Uzbek
Cyrillic is transliterated to Latin (the model saw far more Latin Uzbek), and ``"query: "`` is
prepended because that is how ``multilingual-e5`` was trained. ``detect_language`` and
``numbers_in`` are the two lexical signals of the duplicate rule (``research/models.md`` §4.3).

Only this module and ``models.py`` import onnxruntime/tokenizers; the ONNX session is created
once per process by ``models.py`` because the tokenizer alone costs about 320 MB of memory.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from tg_curator.textutil import normalise

# --- input cleaning ---------------------------------------------------------------------------

_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿\U0001f1e6-\U0001f1ff⬀-⯿️‍]+")
_URL_RE = re.compile(r"https?://\S+|t\.me/\S+")
_TAG_RE = re.compile(r"(?<!\w)[@#][\w\d_]+")
_WS_RE = re.compile(r"\s+")

# Uzbek Cyrillic -> Latin (the official 1995 alphabet, with the turned comma for oʻ/gʻ). The
# four letters below exist in Uzbek Cyrillic but not in Russian, so two of them in a text mean
# the text is Uzbek and the whole text is transliterated.
_UZ_LETTERS = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "s", "ч": "ch", "ш": "sh", "ъ": "’",
    "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "o‘", "қ": "q", "ғ": "g‘", "ҳ": "h",
    "щ": "sh", "ы": "i",
}  # fmt: skip
_UZ_ONLY_RE = re.compile("[ўқғҳЎҚҒҲ]")
_UZ_VOWELS = "аеёиоуэюяўы"


def clean(text: str) -> str:
    """Drop URLs, @mentions, #hashtags and emoji and collapse whitespace."""
    cleaned = _URL_RE.sub(" ", text)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = _EMOJI_RE.sub(" ", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def is_uz_cyrillic(text: str) -> bool:
    """True when the text carries at least two Uzbek-only Cyrillic letters."""
    return len(_UZ_ONLY_RE.findall(text)) >= 2


def uz_cyr_to_lat(text: str) -> str:
    """Transliterate Uzbek Cyrillic to Latin, letter by letter, keeping case."""
    out: list[str] = []
    prev = " "
    for ch in text:
        lower = ch.lower()
        replacement = _UZ_LETTERS.get(lower)
        if replacement is None:
            out.append(ch)
        else:
            if lower == "е" and not prev.isalpha():
                replacement = "ye"
            if lower == "ц" and prev.isalpha() and prev.lower() in _UZ_VOWELS:
                replacement = "ts"
            if ch != lower and replacement:
                replacement = replacement[0].upper() + replacement[1:]
            out.append(replacement)
        prev = ch
    return "".join(out)


def prepare(text: str) -> str:
    """The exact string that is embedded: cleaned, and transliterated when it is Uzbek Cyrillic."""
    cleaned = clean(text)
    return uz_cyr_to_lat(cleaned) if is_uz_cyrillic(cleaned) else cleaned


# --- the dedup rule's lexical signals ---------------------------------------------------------

Language = str
"""``"ru"``, ``"uz"``, ``"en"`` or ``"other"`` (§9.2); only decides which threshold applies."""

_CYRILLIC_RE = re.compile("[а-яё]", re.IGNORECASE)
_LATIN_RE = re.compile("[a-z]", re.IGNORECASE)
_UZ_LATIN_RE = re.compile(r"(?i)\b(va|bilan|uchun|ham|bu|bo‘l\w*|qil\w*|so‘m|foiz|yil\w*)\b|[‘’]")
_NUMBER_RE = re.compile(r"\d[\d\s.,:–-]*\d|\d")


def detect_language(text: str) -> Language:
    """Cheap script and stop-word language id, 93 % right on the eval posts.

    It is deliberately crude: it only picks the duplicate threshold (same language 0.90,
    different languages 0.86), and the posts it gets wrong are short, number-heavy ones where
    either threshold gives the same answer.
    """
    cyrillic = len(_CYRILLIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cyrillic == 0 and latin == 0:
        return "other"
    if cyrillic > latin:
        return "uz" if is_uz_cyrillic(text) else "ru"
    return "uz" if len(_UZ_LATIN_RE.findall(text)) >= 3 else "en"


_GROUP_SPACE_RE = re.compile(r"(?<=\d)(?<!\d{4})\s(?=\d{3}(?!\d))")
"""A space (NBSP and narrow NBSP included) between a 1-3 digit run and exactly three digits
groups thousands; any other space separates two numbers (``в 2024 году`` is not glued on)."""
_GROUP_MARK_RE = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")
"""A ``,`` or ``.`` before exactly three digits *may* group thousands (``1,500`` / ``1.500``)."""
_EDGE = ".,:–-"
READING_SEP = "|"
"""Joins the readings of one ambiguous number (``"1.500|1500"``); see ``number_readings``."""


def _piece_readings(piece: str) -> set[str]:
    """The readings of one number written without spaces, ``.`` as the decimal mark.

    Both marks present: the later one is the decimal mark (``1,500.5`` / ``1.500,5``). One mark
    used two or more times, always before three digits: thousands (``1,500,000``). One mark
    once before exactly three digits: ambiguous, so both readings (``1,500`` is 1500 in English
    and 1.5 in Russian). Anything else: the mark is decimal (``3,75``, ``07.10.2026`` as is).
    """
    has_comma, has_dot = "," in piece, "." in piece
    if has_comma and has_dot:
        group = "," if piece.rfind(",") < piece.rfind(".") else "."
        return {piece.replace(group, "").replace(",", ".")}
    if not (has_comma or has_dot):
        return {piece}
    mark = "," if has_comma else "."
    grouping = len(_GROUP_MARK_RE.findall(piece))
    decimal = piece.replace(mark, ".")
    if grouping == piece.count(mark) and grouping >= 2:
        return {piece.replace(mark, "")}
    if grouping == piece.count(mark) == 1:
        return {piece.replace(mark, ""), decimal}
    return {decimal}


def numbers_in(text: str) -> set[str]:
    """The numbers of a post in one locale-free form, for the number veto.

    Two posts that both carry two or more numbers and share none are two different reports
    (rates, scores, exchange rates) however alike the wording is. Translations write the same
    figure differently (``1,500`` in English, ``1 500`` in Russian and Uzbek, ``3,5`` vs
    ``3.5``), so thousands grouping is removed and ``,`` read as the decimal mark; a number
    that may be either (one ``,ddd`` or ``.ddd`` group) is one element holding both readings
    joined by ``READING_SEP``. Count elements for "how many numbers", and compare two sets
    with ``numbers_shared``, never with ``&`` or ``isdisjoint``.
    """
    out: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        for piece in _GROUP_SPACE_RE.sub("", match).split():
            piece = piece.strip(_EDGE)
            if piece:
                out.add(READING_SEP.join(sorted(_piece_readings(piece))))
    return out


def number_readings(numbers: Iterable[str]) -> set[str]:
    """Every reading of every number of a ``numbers_in`` set."""
    return {reading for number in numbers for reading in number.split(READING_SEP)}


def numbers_shared(a: Iterable[str], b: Iterable[str]) -> bool:
    """Whether two ``numbers_in`` sets have a number in common under any reading."""
    return not number_readings(a).isdisjoint(number_readings(b))


# --- embedders --------------------------------------------------------------------------------


def unit_norm(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows (a zero row stays zero) as float32."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class HashEmbedder:
    """The deterministic test double: hashed bag-of-words unit vectors, no download.

    Identical texts give identical vectors, texts that share most words a high cosine and
    unrelated texts a low one, which is all the pipeline rules need to be exercised. Selected
    by ``TG_CURATOR_FAKE_MODELS=1`` (``models.py``).
    """

    id = "hash-embedder-256"
    dim = 256

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._vector(text)
        return out

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in normalise(prepare(text)).split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            n = int.from_bytes(digest, "little")
            v[n % self.dim] += 1.0 if (n >> 8) & 1 else -1.0
        if not v.any():
            v[0] = 1.0
            return v
        return unit_norm(v)


class OnnxEmbedder:
    """``multilingual-e5-small`` through onnxruntime: ``"query: "`` prefix, mean pooling over
    the attention mask, unit-norm float32 output (``research/models.md`` §9.2).

    ``embed`` is blocking and CPU-bound; the sorter runs it in a worker thread. Texts are
    sorted by length so a batch pads as little as possible.
    """

    dim = 384
    prefix = "query: "
    max_tokens = 512
    batch_size = 8

    def __init__(self, model_path: Path, tokenizer_path: Path, *, threads: int, id: str) -> None:  # noqa: A002
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.id = id
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.max_tokens)
        self._tokenizer.no_padding()
        self._pad_id = self._tokenizer.token_to_id("<pad>") or 0
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = [i.name for i in self._session.get_inputs()]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [self.prefix + prepare(t) for t in texts]
        order = np.argsort([len(t) for t in prepared], kind="stable")
        out = np.zeros((len(prepared), self.dim), dtype=np.float32)
        for start in range(0, len(prepared), self.batch_size):
            idx = order[start : start + self.batch_size]
            out[idx] = self._run([prepared[j] for j in idx])
        return unit_norm(out)

    def _run(self, batch: list[str]) -> np.ndarray:
        encodings = [self._tokenizer.encode(t) for t in batch]
        width = max(len(e.ids) for e in encodings)
        ids = np.full((len(encodings), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, enc in enumerate(encodings):
            ids[row, : len(enc.ids)] = enc.ids
            mask[row, : len(enc.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0]
        weights = mask[..., None].astype(np.float32)
        return (hidden * weights).sum(1) / weights.sum(1)
