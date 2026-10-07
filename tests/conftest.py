"""Shared fixtures (DESIGN §16): a temp home, the template settings, the fakes, a Runtime.

``store`` is a started ``Store`` on a SQLite file in the temp home, driven by the fake clock
so every timestamp the database writes follows the test's time; ``rt`` wires it with every
fake. ``make_runtime`` builds a Runtime without a database for modules that never touch it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import (
    OWNER_ID,
    START,
    FakeBotGateway,
    FakeClassifier,
    FakeClock,
    FakeEmbedder,
    FakeLLM,
    FakeUserGateway,
    FakeWorld,
)
from tg_curator.config import SettingsFile, ensure_home, write_template
from tg_curator.db.store import Store, sqlite_url
from tg_curator.domain import ChatKind, MediaKind
from tg_curator.i18n import Translator
from tg_curator.runtime import EventBus, Runtime
from tg_curator.telegram.gateway import ChatInfo, IncomingMessage

FIRST_CHAT_ID = -1_001_000_000_001


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return ensure_home(tmp_path / "home")


@pytest.fixture
def settings_file(home: Path) -> SettingsFile:
    """The shipped template written into ``home`` and loaded (owner_id 0, nothing live)."""
    path = home / "settings.toml"
    write_template(path)
    sf = SettingsFile(path, env={})
    sf.load_sync()
    return sf


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


@pytest.fixture
def world(clock: FakeClock) -> FakeWorld:
    return FakeWorld(clock)


@pytest.fixture
def user_gw(world: FakeWorld) -> FakeUserGateway:
    return FakeUserGateway(world)


@pytest.fixture
def bot_gw(world: FakeWorld) -> FakeBotGateway:
    return FakeBotGateway(world)


@pytest.fixture
async def store(home: Path, clock: FakeClock) -> AsyncIterator[Store]:
    """A started ``Store`` on ``home/curator.db`` whose timestamps follow the fake clock."""
    store = Store(sqlite_url(home / "curator.db"), clock=clock)
    await store.start()
    yield store
    await store.close()


@pytest.fixture
async def make_runtime(
    home: Path,
    settings_file: SettingsFile,
    clock: FakeClock,
    user_gw: FakeUserGateway,
    bot_gw: FakeBotGateway,
) -> Callable[..., Any]:
    """``await make_runtime(store=None, owner_id=1001)``: a Runtime wired with every fake.

    The default ``store=None`` is for tests of modules that never touch the database; the
    ``rt`` fixture passes the real started Store.
    """

    async def factory(store: Store | None = None, *, owner_id: int = OWNER_ID) -> Runtime:
        if settings_file.current.telegram.owner_id != owner_id:
            await settings_file.set_value("telegram.owner_id", owner_id)
        rt = Runtime(
            home=home,
            settings_file=settings_file,
            store=store,
            clock=clock,
            events=EventBus(),
            t=Translator(settings_file.current.general.language, home=home),
            embedder=FakeEmbedder(),
            classifier=FakeClassifier(),
            llm=FakeLLM(),
            user=user_gw,
            bot=bot_gw,
        )
        rt.bot_account = await bot_gw.start()
        return rt

    return factory


@pytest.fixture
async def rt(make_runtime: Callable[..., Any], store: Store) -> Runtime:
    """Runtime with all fakes, the started Store, owner 1001 and the bot account set."""
    return await make_runtime(store)


@pytest.fixture
def make_chat() -> Callable[..., ChatInfo]:
    """``make_chat(id=..., kind="channel", title=..., username=None, ...)`` with fresh ids."""
    counter = {"n": 0}

    def factory(
        id: int | None = None,  # noqa: A002 - mirrors the ChatInfo field name
        *,
        kind: ChatKind = "channel",
        title: str | None = None,
        username: str | None = None,
        noforwards: bool = False,
        is_creator: bool = False,
        is_admin: bool = False,
        archived: bool = False,
        muted_until: datetime | None = None,
    ) -> ChatInfo:
        counter["n"] += 1
        n = counter["n"]
        return ChatInfo(
            id=FIRST_CHAT_ID - (n - 1) if id is None else id,
            kind=kind,
            title=f"Source {n}" if title is None else title,
            username=username,
            noforwards=noforwards,
            is_creator=is_creator,
            is_admin=is_admin,
            archived=archived,
            muted_until=muted_until,
        )

    return factory


@pytest.fixture
def make_message(
    clock: FakeClock, make_chat: Callable[..., ChatInfo]
) -> Callable[..., IncomingMessage]:
    """``make_message(chat=..., text=..., ...)`` with per-chat increasing message ids and the
    fake clock's time as the date."""
    next_ids: dict[int, int] = {}

    def factory(
        chat: ChatInfo | None = None,
        *,
        message_id: int | None = None,
        text: str = "A post with enough words to be a post.",
        html: str | None = None,
        date: datetime | None = None,
        sender_id: int | None = None,
        is_outgoing: bool = False,
        is_service: bool = False,
        reply_to_id: int | None = None,
        grouped_id: int | None = None,
        media: MediaKind | None = None,
        urls: tuple[str, ...] = (),
        views: int | None = None,
        forwards: int | None = None,
        fwd_from_chat_id: int | None = None,
        fwd_from_message_id: int | None = None,
        noforwards: bool | None = None,
    ) -> IncomingMessage:
        chat = chat or make_chat()
        if message_id is None:
            message_id = next_ids.get(chat.id, 1)
        next_ids[chat.id] = max(next_ids.get(chat.id, 1), message_id + 1)
        if sender_id is None and chat.kind == "group":
            sender_id = 500
        return IncomingMessage(
            chat=chat,
            message_id=message_id,
            date=date or clock.now(),
            text=text,
            html=html,
            sender_id=sender_id,
            is_outgoing=is_outgoing,
            is_service=is_service,
            reply_to_id=reply_to_id,
            grouped_id=grouped_id,
            media=media,
            urls=urls,
            views=views,
            forwards=forwards,
            fwd_from_chat_id=fwd_from_chat_id,
            fwd_from_message_id=fwd_from_message_id,
            noforwards=chat.noforwards if noforwards is None else noforwards,
        )

    return factory
