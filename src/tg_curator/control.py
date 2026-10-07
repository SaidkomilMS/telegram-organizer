"""The local control socket: how CLI commands reach a running service (DESIGN §13, §3).

The account session may only be open in one process (two processes on one session make
Telegram terminate both, ``AUTH_KEY_DUPLICATED``), and ``curator run`` holds it for as long as
it runs. So while the service runs, ``curator backfill`` and friends do not open their own
session: they send the command over ``<home>/control.sock`` and print what comes back.

Protocol: the client writes one JSON line ``{"cmd": ..., "args": {...}}``; the service answers
with any number of ``{"line": "..."}`` lines and one final ``{"exit": <code>}``. The command
implementations live here too, so the socket and the CLI's standalone path print the same
lines from the same code.

``service.lock`` is the flock ``curator run`` holds; ``ServiceLock`` takes it and
``service_running`` probes it without keeping it.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import html
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tg_curator.errors import CuratorError, TopicExists

if TYPE_CHECKING:
    from tg_curator.config import SourceSettings
    from tg_curator.domain import TopicSyncResult
    from tg_curator.runtime import Runtime
    from tg_curator.telegram.gateway import UserGateway

log = logging.getLogger(__name__)

SOCKET_NAME = "control.sock"
LOCK_NAME = "service.lock"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

Out = Callable[[str], Awaitable[None]]
"""Where a command writes its output, one line per call."""
Command = Callable[["Runtime", Mapping[str, Any], Out], Awaitable[int]]
"""``async def command(rt, args, out) -> exit code``."""


class ServiceUnreachableError(CuratorError):
    """Nothing answers on the control socket (no service, or it is still starting)."""


def socket_path(home: Path) -> Path:
    return home / SOCKET_NAME


def lock_path(home: Path) -> Path:
    return home / LOCK_NAME


# --- the service lock ------------------------------------------------------------------------


class ServiceLock:
    """The flock on ``<home>/service.lock`` that says "a curator process owns the session".

    ``curator run`` holds it for its whole life; standalone CLI commands hold it while they
    have their own runtime. It is an advisory lock on an open file, so it disappears with the
    process even after a crash — the file itself is never removed (removing lock files races).
    """

    def __init__(self, home: Path) -> None:
        self.path = lock_path(home)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Take the lock without waiting; ``False`` when another process holds it."""
        if self._fd is not None:
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def service_running(home: Path) -> bool:
    """Whether another process holds ``service.lock`` (probed, never kept)."""
    if not home.is_dir():
        return False
    probe = ServiceLock(home)
    if probe.acquire():
        probe.release()
        return False
    return True


# --- the server ------------------------------------------------------------------------------


async def serve(rt: Runtime, path: Path) -> None:
    """Listen on ``path`` until cancelled; the socket file is removed on the way out.

    A leftover file from a crashed run is unlinked first, but only when nothing answers on it
    — the service lock already guarantees that no second service is listening.
    """
    await _remove_stale(path)
    try:
        server = await asyncio.start_unix_server(
            lambda r, w: _handle(rt, r, w), path=os.fspath(path)
        )
    except OSError as exc:
        raise CuratorError(
            f"cannot open the control socket {path}: {exc}; use a shorter --home path"
        ) from exc
    try:
        path.chmod(0o600)
    except OSError as exc:
        log.warning("could not set permissions 0600 on %s: %s", path, exc)
    rt.health["control"] = rt.clock.now()
    log.info("control socket listening on %s", path)
    try:
        await server.serve_forever()
    finally:
        server.close()
        with contextlib.suppress(OSError):
            path.unlink()


async def _remove_stale(path: Path) -> None:
    if not path.exists():
        return
    try:
        _, writer = await asyncio.open_unix_connection(os.fspath(path))
    except OSError:
        path.unlink(missing_ok=True)
        log.info("removed a stale control socket %s", path)
        return
    writer.close()
    raise CuratorError(f"another process answers on {path}; stop it first")


async def _handle(rt: Runtime, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """One connection = one command; the client hanging up mid-way only drops output."""
    connected = True

    async def send(message: dict[str, Any]) -> None:
        nonlocal connected
        if not connected:
            return
        try:
            writer.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        except (ConnectionError, OSError):
            connected = False

    async def out(line: str) -> None:
        await send({"line": line})

    try:
        try:
            request = json.loads(await reader.readline())
            name = str(request["cmd"])
            args = request.get("args") or {}
            if not isinstance(args, dict):
                raise TypeError("args must be an object")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            await out(f"bad request: {exc}")
            code = EXIT_USAGE
        else:
            log.info("control: %s %s", name, json.dumps(args, ensure_ascii=False))
            code = await run_command(rt, name, args, out)
        await send({"exit": code})
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()


async def run_command(rt: Runtime, name: str, args: Mapping[str, Any], out: Out) -> int:
    """Run one command of ``COMMANDS``; a ``CuratorError`` becomes its sentence and exit 1.

    Shared by the socket server and the CLI's standalone path, so a failure reads the same
    whichever way the command ran.
    """
    command = COMMANDS.get(name)
    if command is None:
        await out(f"unknown command {name!r}; known: {', '.join(sorted(COMMANDS))}")
        return EXIT_USAGE
    try:
        return await command(rt, args, out)
    except CuratorError as exc:
        await out(str(exc))
        return EXIT_FAILED
    except Exception as exc:
        log.exception("control: %s failed", name)
        await out(f"{name} failed: {exc}; the service log has the details")
        return EXIT_FAILED


# --- the client ------------------------------------------------------------------------------


async def request(
    path: Path, name: str, args: Mapping[str, Any], on_line: Callable[[str], None]
) -> int:
    """Send one command to the running service, hand every output line to ``on_line`` and
    return its exit code; ``ServiceUnreachableError`` when nothing answers."""
    try:
        reader, writer = await asyncio.open_unix_connection(os.fspath(path))
    except OSError as exc:
        raise ServiceUnreachableError(f"nothing answers on {path}: {exc}") from exc
    try:
        payload = {"cmd": name, "args": dict(args)}
        writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await writer.drain()
        while raw := await reader.readline():
            message = json.loads(raw)
            if "line" in message:
                on_line(str(message["line"]))
            elif "exit" in message:
                return int(message["exit"])
    except (ConnectionError, OSError) as exc:
        raise ServiceUnreachableError(f"the service dropped the connection: {exc}") from exc
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()
    raise ServiceUnreachableError("the service closed the connection without an exit code")


# --- the commands ----------------------------------------------------------------------------


def topics_summary(result: TopicSyncResult) -> list[str]:
    """The confirmation of a topic sync (§13): one line, or the unresolved lines."""
    if not result.unresolved:
        line = f"{result.total} topics, all channels resolved"
        # A topic with channel = 0 is neither resolved nor a problem: say it has no channel
        # yet, as the bot does (``control_topics_tracked``), instead of implying it has one.
        tracked = result.total - result.resolved
        if tracked > 0:
            line += f", {tracked} without a channel yet (tracked only)"
        return [line]
    lines = [f"{result.total} topics, {len(result.unresolved)} with a problem:"]
    lines += [f"  {line}" for line in result.unresolved]
    return lines


async def backfill_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"days": 3, "chats": [ref, ...]}`` — the last days pulled in, one line per chat."""
    days = int(args.get("days", 3))
    refs = list(args.get("chats") or [])
    chat_ids: list[int] | None = None
    if refs:
        user = _user(rt)
        chat_ids = [(await user.resolve_chat(_ref(ref))).id for ref in refs]

    async def progress(done: int, total: int, chat: Any) -> None:
        await out(f"reading {total} chats… {done}/{total} {chat.title}")

    result = await _need(rt.backfill, "backfill").run(days, chat_ids=chat_ids, progress=progress)
    line = (
        f"backfill: {result.chats} chats, {result.messages} messages read, "
        f"{result.submitted} new posts sorted"
    )
    if result.skipped_chats:
        line += f", {len(result.skipped_chats)} chats skipped (see the log)"
    await out(line)
    return EXIT_OK


async def preview_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"days": 3, "topic": key|None, "set": {dotted: value}}`` — every decision, nothing
    posted; ``set`` is the ``--set key=value`` overrides of the CLI."""
    from tg_curator.pipeline.preview import render_text

    # Spec "Preview": the user edits the file and runs the preview again. The file's values
    # go into the replay even when the running service (or the topic and source rows the
    # last sync wrote) still has older ones; ``--set`` wins over both.
    from_file, notes = await file_drift(rt)
    for note in notes:
        await out(note)
    overrides = {**from_file, **dict(args.get("set") or {})}
    report = await _need(rt.preview, "preview").replay(
        days=int(args.get("days", 3)),
        topic_key=args.get("topic") or None,
        overrides=overrides or None,
    )
    topic_names = {t.id: t.name for t in await rt.store.list_topics(active=None)}
    chat_names = {c.id: c.title for c in await rt.store.list_chats()}
    text = render_text(report, topic_names=topic_names, chat_names=chat_names)
    for line in text.splitlines():
        await out(line)
    return EXIT_OK


FILE_ONLY_SECTIONS = ("telegram", "storage", "topics", "sources")
"""Settings fields the preview never takes from the file as dotted overrides: the clients and
the database (pinned until a restart), and the lists, which the engine reads from the topic
and chat rows instead (``file_drift`` handles those)."""


async def file_drift(rt: Runtime) -> tuple[dict[str, Any], list[str]]:
    """The hand edits in the settings file that the replay would otherwise miss.

    Returns preview overrides — every ``section.key`` whose file value differs from the
    running settings (a service runs with what it loaded at start or the last ``/reload``)
    and ``topics.<key>.strictness`` where ``[[topics]]`` differs from the topic row — plus
    ``sources.<chat id>.trust`` where ``[[sources]]`` differs from a known chat's row — plus
    the note lines to print first: what the overrides cover, the changed keys a replay of
    stored posts cannot reflect at all (``apply_overrides`` refuses them, so they are named
    rather than passed on), and the edits a replay cannot show yet (new or removed topics,
    category and description, trust of a chat not seen yet or removed from the file), which
    the next start or ``/reload`` applies. Nothing is written.
    """
    from tg_curator.pipeline.preview import previewable

    try:
        disk = rt.settings_file.peek()
    except CuratorError as exc:
        return {}, [
            f"note: the settings file cannot be read ({exc}); previewing the settings "
            "the curator runs with"
        ]
    running = rt.settings
    overrides: dict[str, Any] = {}
    unshown: list[str] = []
    for name in type(disk).model_fields:
        if name in FILE_ONLY_SECTIONS:
            continue
        mine, theirs = getattr(running, name), getattr(disk, name)
        if mine == theirs:
            continue
        for field in type(theirs).model_fields:
            if getattr(mine, field) != getattr(theirs, field):
                key = f"{name}.{field}"
                if previewable(key):
                    overrides[key] = getattr(theirs, field)
                else:
                    unshown.append(key)

    pending: list[str] = []
    rows = {t.key: t for t in await rt.store.list_topics(active=True)}
    for entry in disk.topics:
        row = rows.pop(entry.key, None)
        if row is None:
            pending.append(f"new topic {entry.key}")
            continue
        if (entry.strictness or None) != row.strictness:
            overrides[f"topics.{entry.key}.strictness"] = entry.strictness
        if entry.category != row.category or entry.description != row.description:
            pending.append(f"category/description of {entry.key}")
    pending += [f"removed topic {key}" for key in rows]
    trust, unknown = await _trust_drift(rt, disk.sources)
    overrides.update(trust)
    pending += unknown

    notes: list[str] = []
    if overrides:
        notes.append(
            f"note: this preview uses the file's {', '.join(sorted(overrides))}; the service "
            "applies them at its next start or on /reload"
        )
    if unshown:
        notes.append(
            f"note: a preview of stored posts cannot show the file's {', '.join(unshown)}; "
            "the service applies them at its next start or on /reload"
        )
    if pending:
        notes.append(
            f"note: not in this preview until the next start or /reload: {', '.join(pending)}"
        )
    return overrides, notes


async def _trust_drift(
    rt: Runtime, sources: Sequence[SourceSettings]
) -> tuple[dict[str, Any], list[str]]:
    """``[[sources]]`` trust the chat rows (what the replay reads) lack.

    A known chat's new trust becomes a ``sources.<chat id>.trust`` preview override; a source
    the curator has not seen yet (no stored posts to replay) and a chat removed from the file
    are named instead.
    """
    chats = await rt.store.list_chats()
    by_id = {c.id: c for c in chats}
    by_name = {c.username.lower(): c for c in chats if c.username}
    listed: set[int] = set()
    overrides: dict[str, Any] = {}
    pending: list[str] = []
    for source in sources:
        if isinstance(source.chat, int):
            chat = by_id.get(source.chat)
        else:
            chat = by_name.get(_username(source.chat))
        if chat is None:
            pending.append(f"trust of source {source.chat}")
            continue
        listed.add(chat.id)
        if chat.trust != float(source.trust):
            overrides[f"sources.{chat.id}.trust"] = source.trust
    pending += [
        f"trust of source {c.username or c.id} (removed from [[sources]])"
        for c in chats
        if c.trust is not None and c.id not in listed
    ]
    return overrides, pending


def _username(ref: str) -> str:
    """``@name``, ``t.me/name`` or ``https://t.me/name`` as the bare lower-case username."""
    name = re.sub(r"^(https?://)?(www\.)?t(elegram)?\.me/", "", ref.strip(), flags=re.I)
    return name.lstrip("@").rstrip("/").lower()


async def digest_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"topic": key|None}`` — the manual digest, sent now (refused when not live)."""
    results = await _need(rt.digest, "digest").send(args.get("topic") or None)
    names = await _topic_names(rt)
    if not results:
        await out("no topic has a channel yet: nothing to send")
    for r in results:
        name = names.get(r.topic_key, r.topic_key)
        if r.skipped_reason:
            await out(f"{name}: {r.skipped_reason}")
        else:
            await out(f"{name}: {r.item_count} posts sent (manual digest {r.seq})")
    return EXIT_OK


async def digest_preview_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"topic": key|None}`` — tonight's digest as it would be sent, nothing changed."""
    drafts = await _need(rt.digest, "digest").preview(args.get("topic") or None)
    names = await _topic_names(rt)
    if not drafts:
        await out("no topic has a channel yet: there is no digest to show")
    for draft in drafts:
        name = names.get(draft.topic_key, draft.topic_key)
        if not draft.items:
            await out(f"{name}: nothing for {draft.day.isoformat()} yet")
            continue
        await out(f"── {name} · {draft.day.isoformat()} · {len(draft.items)} items")
        for part in draft.parts:
            for line in plain(part).splitlines():
                await out(line)
    return EXIT_OK


async def stats_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"days": N|None, "all": bool}`` — per-chat volume, signal, repeats, published."""
    days = args.get("days")
    window = int(days) if days else rt.settings.review.window_days
    rows = await _need(rt.stats, "stats").chat_stats(window, include_left=bool(args.get("all")))
    await out(f"{'chat':<32} {'volume':>6} {'signal':>7} {'repeats':>7} {'published':>9} days  id")
    for s in rows:
        await out(
            f"{_cut(s.title, 32):<32} {s.volume:>6} {s.signal:>7.1%} {s.duplicate_share:>7.1%} "
            f"{s.published:>9} {s.observed_days:>4}  {s.chat_id}"
        )
    await out(f"stats: {len(rows)} chats over the last {window} days")
    for t in await _need(rt.stats, "stats").topic_stats(window):
        name = t.name if t.has_channel else f"{t.name} (no channel yet)"
        await out(f"topic {name}: {t.posts} posts, {t.immediate} would have gone out immediately")
    return EXIT_OK


async def review_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{}`` — build this week's proposals and send them to the bot chat now."""
    review = _need(rt.review, "review")
    proposals = await review.build()
    sent = await review.send()
    await out(f"review: {len(proposals)} proposals, {sent} sent to the bot chat")
    return EXIT_OK


async def chats_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"all": bool}`` — the dialog list synced, then every chat with its identifier."""
    result = await _need(rt.intake, "intake").sync_chats()
    include_left = bool(args.get("all"))
    for chat in await rt.store.list_chats(active=None if include_left else True):
        username = f"@{chat.username}" if chat.username else "-"
        left = "  (left)" if not chat.active else ""
        await out(
            f"{chat.id:>16}  {chat.kind:<7}  {chat.role:<7}  {username:<24} {chat.title}{left}"
        )
    await out(f"found {result.total} chats, {result.outputs} are output channels")
    return EXIT_OK


async def topics_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{}`` — sync the topics from the settings file, list them, confirm or list problems."""
    result = await _need(rt.topics, "topics").sync_from_settings()
    for t in await rt.store.list_topics(active=True):
        channel = str(t.channel_id) if t.channel_id else "no channel (tracked only)"
        category = f"  category {t.category}" if t.category else ""
        await out(f"{t.key:<20} {t.name:<28} {channel}{category}")
    for line in topics_summary(result):
        await out(line)
    return EXIT_OK


async def topics_add_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"name": ..., "category", "description", "example_channel", "channel",
    "create_channel"}`` — a new topic (and its channel), the same steps as /topics add."""
    topics = _need(rt.topics, "topics")
    create_channel = bool(args.get("create_channel"))
    fields = {
        "category": args.get("category") or None,
        "description": args.get("description") or None,
        "example_channel": args.get("example_channel") or None,
        "channel": args.get("channel") or None,
        "create_channel": create_channel,
    }
    try:
        topic = await topics.create(str(args.get("name") or ""), **fields)
    except TopicExists as exc:
        await out(f"{exc}; edit the existing topic in the settings file instead")
        return EXIT_FAILED
    channel = str(topic.channel_id) if topic.channel_id else "none yet (tracked only)"
    await out(f"topic {topic.key} ({topic.name}) added, channel: {channel}")
    wait = topics.channel_wait_minutes()
    if create_channel and topic.channel_id is None and wait:
        await out(f"Telegram asks to wait {wait} min before another channel can be created")
    return EXIT_OK


async def topics_remove_command(rt: Runtime, args: Mapping[str, Any], out: Out) -> int:
    """``{"key": ...}`` — the topic leaves the settings file; its channel and history stay."""
    key = str(args.get("key") or "")
    await _need(rt.topics, "topics").remove(key)
    await out(f"topic {key} removed; its channel and history are kept")
    return EXIT_OK


COMMANDS: dict[str, Command] = {
    "backfill": backfill_command,
    "preview": preview_command,
    "digest": digest_command,
    "digest-preview": digest_preview_command,
    "stats": stats_command,
    "review": review_command,
    "chats": chats_command,
    "topics": topics_command,
    "topics-add": topics_add_command,
    "topics-remove": topics_remove_command,
}
"""What the socket exposes (§13): the commands that need the running service's session."""

NEEDS: dict[str, frozenset[str]] = {
    "backfill": frozenset({"models", "user"}),
    "preview": frozenset({"models"}),
    "digest": frozenset({"user", "bot"}),
    "digest-preview": frozenset({"user"}),
    "stats": frozenset(),
    "review": frozenset({"models", "bot"}),
    "chats": frozenset({"user"}),
    "topics": frozenset({"models", "user", "bot"}),
    "topics-add": frozenset({"models", "user", "bot"}),
    "topics-remove": frozenset({"models"}),
}
"""What a command needs when the CLI runs it standalone (``service.open_runtime``): the local
models only where something is embedded or classified, the clients only where Telegram is
read or written."""


# --- helpers ---------------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def plain(markup: str) -> str:
    """Bot HTML as terminal text: tags dropped, entities decoded."""
    return html.unescape(_TAG_RE.sub("", markup))


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _ref(ref: Any) -> int | str:
    """A ``--chat`` value: numeric ids become ints, anything else is a username or link."""
    text = str(ref).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else text


async def _topic_names(rt: Runtime) -> dict[str, str]:
    return {t.key: t.name for t in await rt.store.list_topics(active=None)}


def _need[T](service: T | None, name: str) -> T:
    if service is None:
        raise CuratorError(f"{name} is not wired in this process")
    return service


def _user(rt: Runtime) -> UserGateway:
    if rt.user is None:
        raise CuratorError("the account is not connected in this process")
    return rt.user
