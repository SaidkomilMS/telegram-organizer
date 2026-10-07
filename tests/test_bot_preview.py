"""bot/preview.py: backfill when stale, the summary, and paging through decisions (§11.2)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pytest

from tests.fakes import FakeBotGateway, FakeClassifier, FakeClock, FakeUserGateway
from tests.test_bot_topics import CALLBACK_RE, Driver, button_data, wire
from tg_curator.domain import KV, Topic
from tg_curator.pipeline.backfill import BackfillService
from tg_curator.pipeline.intake import IntakeService
from tg_curator.pipeline.sorter import Sorter
from tg_curator.runtime import Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage

ML_CHANNEL = -1_008_000_000_001
ML_TEXTS = [
    f"Lab {i} released model number {i} with a longer context window and cheaper tokens."
    for i in range(7)
]
SPORT = "Barcelona won the derby after a late goal from the substitute striker on Sunday."
UNSORTED = "A new metro line opened in the capital this morning with twelve stations in total."
EMPTY_TEXT = ""


class Scene:
    def __init__(self, rt: Runtime, drv: Driver, user: FakeUserGateway, ml: Topic, sport: Topic):
        self.rt, self.drv, self.user, self.ml, self.sport = rt, drv, user, ml, sport

    def history_reads(self) -> int:
        return len(self.user.calls_of("history"))


@pytest.fixture
async def scene(
    rt: Runtime,
    clock: FakeClock,
    bot_gw: FakeBotGateway,
    user_gw: FakeUserGateway,
    make_chat: Callable[..., ChatInfo],
    make_message: Callable[..., IncomingMessage],
) -> Scene:
    app = wire(rt)
    await app.start()

    async def no_sleep(_: float) -> None:
        return None

    rt.intake = IntakeService(rt)
    rt.sorter = Sorter(rt)
    rt.backfill = BackfillService(rt, sleep=no_sleep)
    rt.publisher = None

    ml = await rt.store.upsert_topic(
        Topic(id=0, key="ml", name="ML & AI", channel_id=ML_CHANNEL, created_at=clock.now())
    )
    sport = await rt.store.upsert_topic(
        Topic(id=0, key="sport", name="Sport", channel_id=None, created_at=clock.now())
    )
    classifier: FakeClassifier = rt.classifier  # type: ignore[assignment]
    classifier.reload([ml, sport], [])
    for text in ML_TEXTS:
        classifier.text_scores[text] = {ml.id: 0.9, sport.id: 0.1}
    classifier.text_scores[SPORT] = {ml.id: 0.1, sport.id: 0.8}

    a = user_gw.add_chat(make_chat(title="AI Daily", username="aidaily"))
    b = user_gw.add_chat(make_chat(title="Tech Wire"))
    c = user_gw.add_chat(make_chat(title="City News"))
    for chat in (a, b, c):
        await rt.store.upsert_chat(chat)
    hours = iter(range(60, 0, -2))
    for text in ML_TEXTS:
        user_gw.seed(make_message(a, text=text, date=clock.now() - timedelta(hours=next(hours))))
    user_gw.seed(make_message(b, text=ML_TEXTS[0], date=clock.now() - timedelta(hours=1)))  # repeat
    user_gw.seed(make_message(b, text=SPORT, date=clock.now() - timedelta(hours=5)))
    user_gw.seed(make_message(c, text=UNSORTED, date=clock.now() - timedelta(hours=3)))
    user_gw.seed(make_message(c, text=EMPTY_TEXT, date=clock.now() - timedelta(hours=2)))
    return Scene(rt, Driver(rt, bot_gw), user_gw, ml, sport)


async def test_preview_backfills_first_and_sends_the_summary(scene: Scene) -> None:
    drv = scene.drv
    await drv.say("/preview")
    sent = drv.sent()
    progress = sent[0]
    shown = [c["html"] for c in scene.drv.bot.calls_of("edit_text") if c["message_id"] == 1]
    first = scene.drv.bot.calls_of("send_text")[0]["html"]
    assert first == "Reading the last three days from 3 chats… 0/3"
    assert shown[:3] == [
        f"Reading the last three days from 3 chats… {done}/3" for done in (1, 2, 3)
    ]
    assert (
        progress.text == "Read 11 messages from 3 chats (11 new posts stored). Nothing was posted."
    )
    assert scene.history_reads() == 3
    assert await scene.rt.store.kv_get(KV.BACKFILL_LAST_RUN_AT) is not None

    summary = sent[-1].text
    assert "Preview of the last 3 days" in summary and "Nothing was posted" in summary
    assert "ML & AI: 7 posts" in summary
    assert "Sport: 1 posts (no channel yet: tracked only)" in summary
    assert "Repeats: 1" in summary
    assert "Unsorted: 1" in summary
    assert "Ignored: 1" in summary
    ml_block = summary.split("ML & AI: 7 posts")[1].split("Sport:")[0]
    assert ml_block.count("• ") == 3 and "— AI Daily" in ml_block
    assert f"• {UNSORTED[:40]}" in summary and "— City News" in summary
    assert button_data(sent[-1]) == [f"pv:p:{scene.ml.id}:0", f"pv:p:{scene.sport.id}:0"]
    assert scene.drv.bot.sent(ML_CHANNEL) == []  # nothing is ever posted by a preview


async def test_backfill_runs_only_when_stale_or_asked(scene: Scene, clock: FakeClock) -> None:
    await scene.drv.say("/preview")
    assert scene.history_reads() == 3
    clock.advance(timedelta(hours=23))
    await scene.drv.say("/preview")
    assert scene.history_reads() == 3
    await scene.drv.say("/preview refresh")
    assert scene.history_reads() == 6
    clock.advance(timedelta(hours=25))
    await scene.drv.say("/preview")
    assert scene.history_reads() == 9


async def test_topic_decisions_are_paged_with_wrong_topic_buttons(scene: Scene) -> None:
    drv = scene.drv
    await drv.say("/preview")  # stores the backlog
    start = len(drv.sent())
    await drv.say("/preview ml & ai")
    page = drv.sent()[start:]
    items, nav = page[:-1], page[-1]
    assert len(items) == 5
    for msg in items:
        assert len(button_data(msg)) == 1 and button_data(msg)[0].startswith("wt:")
        assert "→ ML &amp; AI" not in msg.text and "→ ML & AI (0.90), digest" in msg.text
    assert items[0].text.startswith("1. ") and "AI Daily" in items[0].text
    assert "decisions 1–5 of 7" in nav.text
    assert button_data(nav) == [f"pv:p:{scene.ml.id}:1"]

    await drv.press(f"pv:p:{scene.ml.id}:1", nav)
    assert nav.buttons is None  # the old navigation goes away
    second = drv.sent()[start + len(page) :]
    assert [m.text.split(".")[0] for m in second[:-1]] == ["6", "7"]
    assert "decisions 6–7 of 7" in second[-1].text
    assert button_data(second[-1]) == [f"pv:p:{scene.ml.id}:0"]
    for msg in drv.sent():
        for data in button_data(msg):
            assert CALLBACK_RE.match(data) and len(data.encode()) <= 64


async def test_a_preview_item_is_corrected_with_the_shared_menu(scene: Scene) -> None:
    drv = scene.drv
    await drv.say("/preview")
    await drv.press(f"pv:p:{scene.ml.id}:0", drv.sent()[-1])
    item = next(m for m in drv.sent() if any(d.startswith("wt:") for d in button_data(m)))
    post_id = int(button_data(item)[0].split(":")[1])
    await drv.press(f"wt:{post_id}", item)
    assert button_data(item) == [
        f"mv:{post_id}:{scene.sport.id}",
        f"mv:{post_id}:0",
        f"mv:{post_id}:x",
    ]
    await drv.press(f"mv:{post_id}:{scene.sport.id}", item)
    post = await scene.rt.store.get_post(post_id)
    assert post is not None and post.topic_id == scene.sport.id and post.corrected
    assert button_data(item) == [f"wt:{post_id}"]


async def test_unknown_topic_and_empty_topic(scene: Scene) -> None:
    drv = scene.drv
    await drv.say("/preview nonsense")
    assert "There is no topic called <b>nonsense</b>" in (drv.last().html or "")
    assert button_data(drv.last()) == [f"pv:p:{scene.ml.id}:0", f"pv:p:{scene.sport.id}:0"]
    await drv.say("/preview sport")  # nothing stored yet: no backfill for a topic page
    assert drv.last().text == "Nothing went to Sport in the last 3 days."
    assert scene.history_reads() == 0


async def test_without_an_account_the_stored_backlog_is_previewed(scene: Scene) -> None:
    scene.rt.user = None
    await scene.drv.say("/preview")
    assert "The account is not bound" in scene.drv.sent()[0].text
    assert "ML & AI: 0 posts" in scene.drv.last().text
