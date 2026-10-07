"""folders.py: merging, never empty (deleted when emptied), the 100 cap, staging excluded, the
limit, recreation."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from tests.fakes import OWNER_ID, START, FakeBotGateway, FakeUserGateway
from tg_curator.domain import KV, PROPOSAL_DONE
from tg_curator.runtime import Runtime
from tg_curator.subscriptions.folders import FolderManager
from tg_curator.telegram.gateway import ChatInfo


@pytest.fixture
def folders(rt: Runtime) -> FolderManager:
    rt.folders = FolderManager(rt)
    return rt.folders


async def chat(rt: Runtime, info: ChatInfo, *, role: str = "source", flagged: bool = False) -> int:
    rt.user.add_chat(info)  # type: ignore[union-attr]
    await rt.store.upsert_chat(info, role=role)  # type: ignore[arg-type]
    if flagged:
        await rt.store.set_chat_fields(info.id, in_low_signal=True)
    return info.id


async def flagged_by_curator(rt: Runtime, chat_id: int) -> None:
    """The record the executor leaves behind for a folder move the owner approved."""
    await rt.store.create_proposal(
        "folder", reason="r", review_day=START.date(), chat_id=chat_id, state=PROPOSAL_DONE
    )


# --- creation --------------------------------------------------------------------------------


async def test_creates_both_folders_and_remembers_their_ids(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    out1 = await chat(rt, make_chat(), role="output")
    out2 = await chat(rt, make_chat(), role="output")
    low = await chat(rt, make_chat(), flagged=True)
    await chat(rt, make_chat())  # an ordinary source: in neither folder
    await folders.sync()
    curated = await rt.store.kv_get(KV.FOLDERS_CURATED_ID)
    low_signal = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert user_gw.folders[curated][0] == "Curated"
    assert sorted(user_gw.folders[curated][1]) == sorted([out1, out2])
    assert user_gw.folders[low_signal] == ("Low signal", [low])
    assert {curated, low_signal} <= set(range(2, 256)) and curated != low_signal
    assert {name for name, _ in user_gw.calls} == {"list_folders", "save_folder"}


async def test_never_creates_an_empty_folder(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    await chat(rt, make_chat(), role="output")
    await folders.sync()
    assert [title for title, _ in user_gw.folders.values()] == ["Curated"]
    assert await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None
    low = await chat(rt, make_chat(), flagged=True)
    await folders.sync()  # "Low signal" appears with the first flagged chat
    assert [peers for _, peers in user_gw.folders.values()] == [[-1_001_000_000_001], [low]]


async def test_nothing_is_saved_when_nothing_changed(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    await chat(rt, make_chat(), role="output")
    await chat(rt, make_chat(), flagged=True)
    await folders.sync()
    saves = len(user_gw.calls_of("save_folder"))
    await folders.sync()
    assert len(user_gw.calls_of("save_folder")) == saves == 2
    assert len(user_gw.calls_of("get_folder")) == 2  # read, compared, left alone


async def test_settings_can_switch_a_folder_off(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    await rt.settings_file.set_value("folders.curated", False)
    await rt.settings_file.set_value("folders.low_signal_name", "Noise")
    await chat(rt, make_chat(), role="output")
    low = await chat(rt, make_chat(), flagged=True)
    await folders.sync()
    assert list(user_gw.folders.values()) == [("Noise", [low])]


# --- merging ---------------------------------------------------------------------------------


async def test_merge_keeps_hand_added_chats_and_removes_only_unflagged_ones(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    by_hand = 123456  # a private chat the user dropped into the folder, unknown to the DB
    gone = await chat(rt, make_chat())  # flagged by the curator earlier, undone since
    await flagged_by_curator(rt, gone)
    still = await chat(rt, make_chat(), flagged=True)
    await flagged_by_curator(rt, still)
    new = await chat(rt, make_chat(), flagged=True)
    user_gw.own_folder(5, "Low signal", [by_hand, gone, still])
    await rt.store.kv_set(KV.FOLDERS_LOW_SIGNAL_ID, 5)
    await folders.sync()
    assert user_gw.folders[5] == ("Low signal", [by_hand, still, new])
    [save] = user_gw.calls_of("save_folder")
    assert save["folder_id"] == 5


async def test_curated_keeps_hand_added_chats_and_adds_new_channels(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    by_hand = await chat(rt, make_chat())  # a source the user keeps in Curated by hand
    old = await chat(rt, make_chat(), role="output")
    new = await chat(rt, make_chat(), role="output")
    user_gw.own_folder(3, "Curated", [by_hand, old])
    await rt.store.kv_set(KV.FOLDERS_CURATED_ID, 3)
    await folders.sync()
    assert user_gw.folders[3] == ("Curated", [by_hand, old, new])


async def test_staging_channel_is_kept_out_of_both_folders(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    staging = await chat(rt, make_chat(title="tg-curator media"), role="staging")
    await rt.settings_file.set_value("publishing.staging_channel", staging)
    out = await chat(rt, make_chat(), role="output")
    user_gw.own_folder(4, "Curated", [staging, out])
    await rt.store.kv_set(KV.FOLDERS_CURATED_ID, 4)
    await folders.sync()
    assert user_gw.folders[4] == ("Curated", [out])


async def test_a_folder_deleted_by_hand_is_recreated(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    out = await chat(rt, make_chat(), role="output")
    await rt.store.kv_set(KV.FOLDERS_CURATED_ID, 9)  # remembered, but gone from Telegram
    await folders.sync()
    [save] = user_gw.calls_of("save_folder")
    assert save["folder_id"] is None and save["chat_ids"] == [out]
    new_id = await rt.store.kv_get(KV.FOLDERS_CURATED_ID)
    assert new_id != 9 and user_gw.folders[new_id] == ("Curated", [out])


async def test_low_signal_is_deleted_when_nothing_is_flagged_and_recreated_later(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    """§17.2: Telegram refuses an empty folder, so the curator deletes its own and forgets the
    id; the next flagged chat creates it again."""
    await chat(rt, make_chat(), role="output")
    low = await chat(rt, make_chat(), flagged=True)
    await flagged_by_curator(rt, low)
    await folders.sync()
    first = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert user_gw.folders[first] == ("Low signal", [low])
    await rt.store.set_chat_fields(low, in_low_signal=False)  # the owner undid the move
    await folders.sync()
    assert [c["folder_id"] for c in user_gw.calls_of("delete_folder")] == [first]
    assert first not in user_gw.folders
    assert await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None
    assert [title for title, _ in user_gw.folders.values()] == ["Curated"]  # untouched
    await folders.sync()  # nothing to delete twice
    assert len(user_gw.calls_of("delete_folder")) == 1
    again = await chat(rt, make_chat(), flagged=True)
    await folders.sync()
    second = await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID)
    assert second is not None and user_gw.folders[second] == ("Low signal", [again])


async def test_a_folder_kept_alive_by_a_hand_added_chat_is_not_deleted(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    by_hand = 123456
    gone = await chat(rt, make_chat())
    await flagged_by_curator(rt, gone)
    user_gw.own_folder(5, "Low signal", [by_hand, gone])
    await rt.store.kv_set(KV.FOLDERS_LOW_SIGNAL_ID, 5)
    await folders.sync()
    assert user_gw.folders[5] == ("Low signal", [by_hand])
    assert user_gw.calls_of("delete_folder") == []


async def test_a_stale_id_of_an_empty_folder_is_forgotten(
    rt: Runtime, folders: FolderManager, user_gw: FakeUserGateway
) -> None:
    await rt.store.kv_set(KV.FOLDERS_LOW_SIGNAL_ID, 9)  # deleted by hand, nothing flagged now
    await folders.sync()
    assert await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None
    assert user_gw.calls_of("delete_folder") == [] and user_gw.calls_of("save_folder") == []


async def test_include_peers_is_capped_at_100_with_one_warning(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, caplog: pytest.LogCaptureFixture,
) -> None:  # fmt: skip
    for _ in range(105):
        await chat(rt, make_chat(), role="output")
    with caplog.at_level(logging.WARNING, logger="tg_curator.subscriptions.folders"):
        await folders.sync()
        await chat(rt, make_chat(), role="output")  # over the cap too: the folder is unchanged
        await folders.sync()
    [(_, peers)] = user_gw.folders.values()
    assert len(peers) == 100 and len(user_gw.calls_of("save_folder")) == 1
    warnings = [r for r in caplog.records if "allows 100" in r.getMessage()]
    assert len(warnings) == 1


# --- the limit -------------------------------------------------------------------------------


async def test_folder_limit_disables_the_module_until_reload(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway, bot_gw: FakeBotGateway, caplog: pytest.LogCaptureFixture,
) -> None:  # fmt: skip
    user_gw.folder_limit = 0
    await chat(rt, make_chat(), role="output")
    await chat(rt, make_chat(), flagged=True)
    with caplog.at_level(logging.WARNING, logger="tg_curator.subscriptions.folders"):
        await folders.sync()
        await folders.sync()
    assert await rt.store.kv_get(KV.FOLDERS_DISABLED) == "limit"
    assert len(user_gw.calls_of("save_folder")) == 1  # stopped at the first refusal
    assert len(user_gw.calls_of("list_folders")) == 1  # the second sync returned at once
    assert len([r for r in caplog.records if "folder limit" in r.getMessage()]) == 1
    assert bot_gw.sent(OWNER_ID) == []  # not an owner message (§11.4)
    # /reload clears the key; the next sync tries again
    await rt.store.kv_delete(KV.FOLDERS_DISABLED)
    user_gw.folder_limit = 10
    await folders.sync()
    assert len(user_gw.folders) == 2


async def test_after_a_restart_the_stored_folders_are_still_written(
    rt: Runtime, folders: FolderManager, make_chat: Callable[..., ChatInfo],
    user_gw: FakeUserGateway,
) -> None:  # fmt: skip
    """A fresh gateway owns no folder id: the ids in ``kv folders.*`` that still carry the
    curator's titles are registered before they are saved to or deleted."""
    old = await chat(rt, make_chat(), role="output")
    new = await chat(rt, make_chat(), role="output")
    user_gw.folders[3] = ("Curated", [old])  # created by an earlier process
    user_gw.folders[5] = ("Low signal", [-1_009_999])  # nothing is flagged any more
    await rt.store.kv_set(KV.FOLDERS_CURATED_ID, 3)
    await rt.store.kv_set(KV.FOLDERS_LOW_SIGNAL_ID, 5)
    await flagged_by_curator(rt, -1_009_999)
    await folders.sync()
    assert user_gw.folders[3] == ("Curated", [old, new])
    assert 5 not in user_gw.folders
    assert await rt.store.kv_get(KV.FOLDERS_LOW_SIGNAL_ID) is None
