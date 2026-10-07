"""Audit gaps of the ml group, end to end through the live sorter with the real lexical helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tests.fakes import FakeClassifier
from tests.test_sorter import StubPublisher, cand, fresh
from tg_curator.domain import PostStatus, Topic
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo

CHANNEL = -1_001_000_000_901
ENGLISH = "Floods displaced 1,500 families and damaged 2,300 homes in the river valley."
RUSSIAN = "Наводнение оставило без крова 1 500 семей и разрушило 2 300 домов в долине реки."
DIM = 16


class TranslationEmbedder:
    """The English text on one axis, its Russian translation at cosine 0.88 to it."""

    id = "translation-stub"
    dim = DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            if text.startswith("Floods"):
                out[i, 0] = 1.0
            else:
                out[i, 0] = 0.88
                out[i, 1] = np.sqrt(1.0 - 0.88 * 0.88)
        return out


async def test_sort_2_ru_translation_with_grouped_thousands_corroborates_the_english_root(
    rt: Runtime, make_chat: Callable[..., ChatInfo]
) -> None:
    """SORT-2: "1,500 / 2,300" and "1 500 / 2 300" are the same figures, so the number veto
    must not split an EN/RU translation (cosine 0.88, cross-language threshold 0.86)."""
    rt.embedder = TranslationEmbedder()  # type: ignore[assignment]
    rt.publisher = StubPublisher(rt.store)  # type: ignore[assignment]
    t = await rt.store.upsert_topic(
        Topic(id=0, key="news", name="News", channel_id=CHANNEL, created_at=rt.clock.now())
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.scores = {t.id: 0.9}
    chats = []
    for _ in range(2):
        info = make_chat()
        await rt.store.upsert_chat(info)
        chats.append(info.id)
    sorter = Sorter(rt)

    root = await sorter.submit(cand(chats[0], 1, ENGLISH))
    dup = await sorter.submit(cand(chats[1], 1, RUSSIAN))
    assert root is not None and dup is not None
    assert root.status != PostStatus.duplicate
    assert dup.status == PostStatus.duplicate and dup.dup_kind == "semantic"
    assert dup.duplicate_of == root.id
    assert (await fresh(rt.store, root)).corroboration == 1
