"""The two folders the curator keeps in the user's Telegram (DESIGN §12).

"Curated" holds the topic channels and "Low signal" the chats the owner agreed to flag. This
is the only module that reads or writes dialog filters, and it only ever writes the two ids it
stored in ``kv folders.*`` (or creates them): a folder the user made is never touched, and the
main list is never reordered. Telegram reuses the id of a deleted folder for the next one the
user creates, so a stored id is the curator's only while the folder behind it still carries
the curator's title (the configured one, or the one the curator last saved, which covers a
rename in the config); otherwise the id is forgotten and the folder left alone. Peers are
merged rather than replaced so a chat the user dropped into one of the folders by hand stays
there; a chat the curator itself once flagged (there is a ``done``/``undone`` folder proposal
for it) and no longer flags is the only kind removed. A chat the user took out of one of the
folders by hand (it was in the list the curator last saved and is missing now) stays out: in
"Low signal" that is the owner's undo, so the flag is cleared and the folder proposal marked
``undone``; in "Curated" the channel is remembered and left out until the user puts it back.
When nothing is left in one of the two folders it is deleted, because Telegram refuses an
empty folder (§17.2). Telegram's folder limit is not something the curator can fix, so hitting
it switches the module off until the owner deletes a folder and sends ``/reload``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import sqlalchemy as sa

from tg_curator.db import schema
from tg_curator.domain import KV, PROPOSAL_DONE, PROPOSAL_UNDONE
from tg_curator.errors import FolderLimit
from tg_curator.subscriptions.review import edit_proposal_message

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime
    from tg_curator.telegram.gateway import UserGateway

log = logging.getLogger(__name__)

FOLDER_PEERS_MAX = 100
"""Telegram's ``include_peers`` limit without Premium (§15)."""
CURATED_HAND_REMOVED = "folders.curated_hand_removed"
"""``kv``: output channels the user took out of "Curated" by hand; left out until put back."""


def _title_key(name: str) -> str:
    """``kv folders.<name>_title``: the title the curator last saved its folder under."""
    return f"folders.{name}_title"


def _peers_key(name: str) -> str:
    """``kv folders.<name>_peers``: the exact peer list the curator last saved (or found)."""
    return f"folders.{name}_peers"


class FolderManager:
    """``sync()``: bring "Curated" and "Low signal" in line with the database."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self._warned_cap: set[str] = set()

    async def sync(self) -> None:
        rt = self._rt
        user = rt.user
        if user is None:
            return
        if await rt.store.kv_get(KV.FOLDERS_DISABLED) is not None:
            return
        settings = rt.settings
        chats = await rt.store.list_chats()
        excluded = {settings.publishing.staging_channel}
        excluded |= {c.id for c in chats if c.role == "staging"}
        existing = dict(await user.list_folders())
        if settings.folders.curated:
            wanted = [c.id for c in chats if c.role == "output" and c.active]
            ok = await self._sync_one(
                user, "curated", KV.FOLDERS_CURATED_ID, settings.folders.curated_name,
                wanted, remove=set(), existing=existing, excluded=excluded,
            )  # fmt: skip
            if not ok:
                return
        if settings.folders.low_signal:
            wanted = [c.id for c in chats if c.role == "source" and c.active and c.in_low_signal]
            remove = await self._formerly_flagged() - set(wanted)
            await self._sync_one(
                user, "low_signal", KV.FOLDERS_LOW_SIGNAL_ID, settings.folders.low_signal_name,
                wanted, remove=remove, existing=existing, excluded=excluded,
            )  # fmt: skip

    async def _sync_one(
        self,
        user: UserGateway,
        name: str,
        kv_key: str,
        title: str,
        wanted: Sequence[int],
        *,
        remove: set[int],
        existing: dict[int, str],
        excluded: set[int],
    ) -> bool:
        """Merge ``wanted`` into one folder; ``False`` when the folder limit stopped us."""
        store = self._rt.store
        folder_id = await self._own_id(user, name, kv_key, title, existing)
        current: list[int] | None = None
        if folder_id is not None:
            current = await user.get_folder(folder_id)
        if current is None:
            if folder_id is not None:
                await self._forget(name, kv_key)
            folder_id, current = None, []  # never created, or deleted by hand: recreate
        else:
            dropped = await self._hand_edits(name, current, wanted)
            wanted = [c for c in wanted if c not in dropped]
        if name == "curated":
            left_out = set(await store.kv_get(CURATED_HAND_REMOVED, []))
            wanted = [c for c in wanted if c not in left_out]
        peers = [c for c in current if c not in remove and c not in excluded]
        peers += [c for c in wanted if c not in peers and c not in excluded]
        if not peers:
            await self._drop(user, title, name, kv_key, folder_id)
            return True
        if len(peers) > FOLDER_PEERS_MAX:
            if name not in self._warned_cap:
                self._warned_cap.add(name)
                log.warning(
                    "folders: %r would hold %d chats; Telegram allows %d, the rest are left out",
                    title, len(peers), FOLDER_PEERS_MAX,
                )  # fmt: skip
            peers = peers[:FOLDER_PEERS_MAX]
        if folder_id is not None and set(peers) == set(current) and existing[folder_id] == title:
            await self._remember(name, title, peers)
            return True
        try:
            new_id = await user.save_folder(folder_id, title, peers)
        except FolderLimit:
            await store.kv_set(KV.FOLDERS_DISABLED, "limit")
            log.warning(
                "folders: Telegram's folder limit is reached; the curator's folders are off "
                "until you delete a folder you do not need and send /reload"
            )
            return False
        if new_id != folder_id:
            await store.kv_set(kv_key, new_id)
        await self._remember(name, title, peers)
        log.info("folders: %r now holds %d chats", title, len(peers))
        return True

    async def _own_id(
        self, user: UserGateway, name: str, kv_key: str, title: str, existing: dict[int, str]
    ) -> int | None:
        """The stored id, but only while the folder behind it is still the curator's.

        Telegram hands the id of a deleted folder to the next one the user creates; such a
        folder carries the user's title, not the curator's. The id is then forgotten (the
        curator's folder was deleted by hand and is recreated with a fresh id) and the user's
        folder is never read, renamed, merged into or deleted.
        """
        store = self._rt.store
        folder_id: int | None = await store.kv_get(kv_key)
        if folder_id is None:
            return None
        live = existing.get(folder_id)
        if live is None:
            return folder_id  # gone from Telegram: get_folder is skipped by the caller below
        if live in {title, await store.kv_get(_title_key(name))}:
            # The gateway trusts only ids it created in this process; a confirmed id from
            # ``kv`` must be registered, or every save/delete after a restart is refused.
            user.register_own_folder(folder_id)
            return folder_id
        log.warning(
            "folders: folder %d is now %r, not the curator's %r; it is left alone and the "
            "curator's folder is created again",
            folder_id, live, title,
        )  # fmt: skip
        await self._forget(name, kv_key)
        return None

    async def _hand_edits(
        self, name: str, current: Sequence[int], wanted: Sequence[int]
    ) -> set[int]:
        """Chats of ``wanted`` the user took out of the folder by hand since the last save.

        Only a chat the curator itself saved into the folder and that is missing now counts;
        a chat it never got in (the 100 cap) does not. In "Low signal" that is the owner's
        undo of the move; in "Curated" the channel is remembered and left out. A channel the
        user puts back into "Curated" by hand is no longer left out.
        """
        store = self._rt.store
        last = set(await store.kv_get(_peers_key(name), []))
        now_in = set(current)
        removed = (last & set(wanted)) - now_in
        if name == "low_signal":
            for chat_id in sorted(removed):
                await self._unflag(chat_id)
            return removed
        left_out = set(await store.kv_get(CURATED_HAND_REMOVED, []))
        updated = (left_out | removed) - now_in
        if updated != left_out:
            if updated:
                await store.kv_set(CURATED_HAND_REMOVED, sorted(updated))
            else:
                await store.kv_delete(CURATED_HAND_REMOVED)
        for chat_id in sorted(removed):
            log.info("folders: channel %d was taken out of Curated by hand; it stays out", chat_id)
        return removed

    async def _unflag(self, chat_id: int) -> None:
        """The owner took a flagged chat out of "Low signal" in Telegram: that is an undo."""
        rt = self._rt
        store = rt.store
        await store.set_chat_fields(chat_id, in_low_signal=False)
        for proposal in await store.proposals_by_state(PROPOSAL_DONE, kind="folder"):
            if proposal.chat_id != chat_id:
                continue
            await store.set_proposal_fields(proposal.id, state=PROPOSAL_UNDONE, result="undone")
            await edit_proposal_message(rt, proposal, rt.t("review_outcome_undone"), None)
        log.info("folders: chat %d was taken out of Low signal by hand; no longer flagged", chat_id)

    async def _remember(self, name: str, title: str, peers: Sequence[int]) -> None:
        """What the folder holds now, as the curator saved (or found) it."""
        store = self._rt.store
        if await store.kv_get(_title_key(name)) != title:
            await store.kv_set(_title_key(name), title)
        if await store.kv_get(_peers_key(name)) != list(peers):
            await store.kv_set(_peers_key(name), list(peers))

    async def _forget(self, name: str, kv_key: str) -> None:
        store = self._rt.store
        for key in (kv_key, _title_key(name), _peers_key(name)):
            if await store.kv_get(key) is not None:
                await store.kv_delete(key)

    async def _drop(
        self, user: UserGateway, title: str, name: str, kv_key: str, folder_id: int | None
    ) -> None:
        """Nothing belongs in the folder any more. Telegram refuses an empty folder, so the
        curator's own folder is deleted and its id forgotten (§17.2); it is created again with
        the next chat that belongs in it. A folder that never existed is simply not created.
        ``folder_id`` is set only for a folder ``_own_id`` confirmed as the curator's."""
        if folder_id is not None:
            await user.delete_folder(folder_id)
            log.info("folders: %r is empty now and was deleted", title)
        await self._forget(name, kv_key)

    async def _formerly_flagged(self) -> set[int]:
        """Chats the curator itself put into "Low signal" at some point (§12 merge rule)."""
        c = schema.proposals.c
        stmt = (
            sa.select(c.chat_id)
            .where(c.kind == "folder")
            .where(c.state.in_([PROPOSAL_DONE, PROPOSAL_UNDONE]))
            .where(c.chat_id.is_not(None))
            .distinct()
        )
        rows = await self._rt.store.execute(stmt)
        return {int(row[0]) for row in rows} if isinstance(rows, list) else set()
