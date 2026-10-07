"""The bot framework: owner guard, claim, routing and persisted conversations (DESIGN §11.1).

Feature modules register commands, callback prefixes and flows on a ``BotApp``; the app
subscribes to the bot gateway and decides, before any handler runs, whether the sender may be
obeyed at all. Everything that reaches a handler therefore comes from the owner (or, before
an owner exists, from whoever is proving they hold the claim code). The active conversation
is kept in ``kv bot.conversation`` and not in memory so that ``/setup`` and ``/bind`` resume
after a restart; the price is that flow data is written to disk, which is why it must never
hold a login code, a 2FA password or an API key (see ``Flow``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import re
import secrets
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from tg_curator.domain import KV
from tg_curator.errors import (
    ChatGone,
    ConfigError,
    CuratorError,
    FloodWait,
    NotAllowed,
    SessionLost,
    TelegramUnavailable,
)
from tg_curator.telegram.gateway import BotCallback, BotGateway, BotMessage, Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

# The secret a deleted message held, for the "delete it yourself" warning.
_SECRET_NOUNS = {
    "code": "core_secret_code",
    "password": "core_secret_password",
    "token": "core_secret_token",
    "key": "core_secret_key",
}

CALLBACK_DATA_MAX_BYTES = 64
"""Telegram's limit on inline-button payloads; checked in ``ctx.reply`` / ``ctx.edit``."""
CLAIM_MAX_ATTEMPTS = 5
"""Wrong claim codes tolerated before a new one is generated (§11.1)."""
CLAIM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""Unambiguous characters (no 0/O, 1/I) so a code read off a log line is typed right."""
CLAIM_CODE_LENGTH = 8
CALLBACK_ANSWER_DEADLINE = 2.0
"""Seconds a handler has to answer its callback (with its toast) before the core answers it
silently: the tapping client stops waiting after a few seconds, and Telegram takes exactly one
answer per query (§11.1)."""

CommandHandler = Callable[["Ctx", str], Awaitable[None]]
"""``async handler(ctx, args)``; ``args`` is the text after the command, stripped."""
CallbackHandler = Callable[["Ctx", str], Awaitable[None]]
"""``async handler(ctx, data)``; ``data`` is the payload after ``"<prefix>:"``."""

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@\w+)?(?:\s+(.*))?$", re.DOTALL)


# --- conversations ---------------------------------------------------------------------------


@dataclass
class Conversation:
    """What is persisted in ``kv bot.conversation``: the flow, its step and its data."""

    flow: str
    step: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_kv(self) -> dict[str, Any]:
        return {"flow": self.flow, "step": self.step, "data": dict(self.data)}

    @classmethod
    def from_kv(cls, raw: Any) -> Conversation | None:
        if not isinstance(raw, dict) or not raw.get("flow") or not raw.get("step"):
            return None
        data = raw.get("data")
        return cls(flow=str(raw["flow"]), step=str(raw["step"]), data=dict(data or {}))


class Flow:
    """Base class of a multi-step conversation (``app.flow(name, FlowClass)``).

    A step is a method decorated with ``@Flow.step("name")`` taking the owner's text; the
    core instantiates the registered class for every incoming text while the conversation is
    active and calls the method of the current step. ``go()`` moves to another step and
    persists ``data``; ``end()`` closes the conversation. ``start()`` runs once from
    ``ctx.start_flow`` so the flow can send its first prompt.

    ``data`` is written to ``kv bot.conversation`` on every ``go``/``save``, so it must never
    contain a login code, a 2FA password or an API key/token: those are consumed within the
    same handler call that received them (§11.1). Keep ids and small plain values in it.
    """

    first_step: ClassVar[str] = "start"
    secret_steps: ClassVar[frozenset[str]] = frozenset()
    """Steps whose input is a secret (a login code, a 2FA password, an API key/token). Text
    sent at such a step reaches the step verbatim even when it looks like a command, so the
    step can delete it; only a bare command (``/cancel``, ``/status``) still cancels, and its
    message is deleted first (§11.1, §11.2: "deleted immediately", "passed verbatim")."""

    def __init__(self, ctx: Ctx, state: Conversation) -> None:
        self.ctx = ctx
        self.state = state

    @staticmethod
    def step(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Mark a method as the handler of step ``name``."""

        def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn.__flow_step__ = name  # type: ignore[attr-defined]
            return fn

        return mark

    @property
    def data(self) -> dict[str, Any]:
        return self.state.data

    @property
    def current_step(self) -> str:
        return self.state.step

    async def start(self) -> None:
        """Called once right after ``ctx.start_flow``; the default does nothing."""

    async def go(self, step: str, **data: Any) -> None:
        """Move to ``step`` and persist (``data`` is merged into the stored data)."""
        self.state.step = step
        self.state.data.update(data)
        await self.save()

    async def save(self) -> None:
        """Persist the conversation after in-place changes to ``data``."""
        await self.ctx.rt.store.kv_set(KV.BOT_CONVERSATION, self.state.as_kv())

    async def end(self) -> None:
        await self.ctx.end_flow()

    def handler_for(self, step: str) -> Callable[[str], Awaitable[None]] | None:
        """The bound method marked ``@Flow.step(step)``, or ``None``."""
        for name in dir(type(self)):
            attr = getattr(type(self), name, None)
            if callable(attr) and getattr(attr, "__flow_step__", None) == step:
                return getattr(self, name)
        return None


# --- the context handed to handlers ----------------------------------------------------------


class Ctx:
    """What a handler sees: the runtime, the incoming message or callback, and replies.

    Replies always go to the owner's private chat — a callback pressed on a topic-channel
    button is answered privately, because a topic channel only ever receives posts and
    digests (spec "Your topic channels").
    """

    def __init__(
        self,
        app: BotApp,
        *,
        message: BotMessage | None = None,
        callback: BotCallback | None = None,
        chat_id: int,
    ) -> None:
        self.app = app
        self.rt: Runtime = app.rt
        self.message = message
        self.callback = callback
        self.chat_id = chat_id
        self._answered = False
        self._lock: asyncio.Lock | None = None
        self._holds_lock = False

    # --- sending ---

    async def reply(self, key_or_html: str, *, buttons: Buttons | None = None, **fmt: Any) -> int:
        """Send a catalogue message (``key`` + placeholders) or ready HTML to the owner."""
        check_buttons(buttons)
        bot = self._bot()
        return await bot.send_text(self.chat_id, self._text(key_or_html, fmt), buttons=buttons)

    async def edit(
        self,
        key_or_html: str,
        *,
        buttons: Buttons | None = None,
        message_id: int | None = None,
        **fmt: Any,
    ) -> None:
        """Edit the message the callback was pressed on (or ``message_id`` in the owner chat)."""
        check_buttons(buttons)
        bot = self._bot()
        if message_id is None:
            if self.callback is None:
                raise RuntimeError("ctx.edit needs a callback or an explicit message_id")
            chat_id, message_id = self.callback.chat_id, self.callback.message_id
        else:
            chat_id = self.chat_id
        await bot.edit_text(chat_id, message_id, self._text(key_or_html, fmt), buttons=buttons)

    @property
    def answered(self) -> bool:
        """Whether the callback has had its one answer (a callback is answered exactly once)."""
        return self._answered

    async def answer(self, text: str | None = None, alert: bool = False) -> None:
        """Answer the callback: ``text`` is the toast (an alert box with ``alert``).

        Telegram takes exactly one answer per query, so only the first call reaches it. The
        core answers silently when the handler has not answered within
        ``CALLBACK_ANSWER_DEADLINE`` seconds (and after it returns); a text that comes after
        that — or that Telegram refused — is sent to the owner's chat instead, so a refusal
        or an outcome is never lost silently."""
        if self.callback is None:
            return
        if self._answered:
            if text:
                await self._say_plain(text)
            return
        self._answered = True
        try:
            await self._bot().answer_callback(self.callback.query_id, text, alert=alert)
        except CuratorError as exc:
            log.debug("callback answer not delivered: %s", exc)
            if text:
                await self._say_plain(text)

    async def delete_incoming(self, what: str | None = None) -> bool:
        """Delete the owner's own message (the one holding a code, password or key).

        True when it is gone (or there was none). When Telegram refuses, the owner is told to
        delete it by hand: the prompts promise it is deleted, so a silent failure would leave
        a secret in the chat behind that promise (SPEC: "deleted right after they are read").
        ``what`` names the secret in that warning: "code", "password", "token" or "key".
        """
        if self.message is None:
            return True
        try:
            await self._bot().delete_message(self.message.chat_id, self.message.message_id)
        except CuratorError as exc:
            # The log line names the message id only, never its text.
            log.warning("could not delete the owner's message %d: %s", self.message.message_id, exc)
            noun = self.rt.t(_SECRET_NOUNS.get(what or "", "core_secret_message"))
            try:
                await self.reply("core_secret_not_deleted", what=noun)
            except CuratorError as warn_exc:
                log.warning("could not tell the owner to delete it: %s", warn_exc)
            return False
        return True

    # --- the per-chat lock ---

    @contextlib.asynccontextmanager
    async def unlocked(self) -> AsyncIterator[None]:
        """Let the owner's other messages and buttons run while this handler waits on
        something slow (the ``/preview`` backfill reads every chat for minutes): the per-chat
        lock the core holds for this handler is released for the body and taken back after
        it, so ``/pause`` never queues behind a long read."""
        lock = self._lock
        if lock is None or not self._holds_lock:
            yield
            return
        lock.release()
        self._holds_lock = False
        try:
            yield
        finally:
            await lock.acquire()
            self._holds_lock = True

    # --- conversations ---

    async def start_flow(self, name: str, **data: Any) -> Flow:
        """Begin the registered flow ``name`` at its first step and run its ``start()``.

        ``data`` is persisted: never pass a code, a password or a key (see ``Flow``).
        """
        flow_class = self.app.flow_class(name)
        state = Conversation(flow=name, step=flow_class.first_step, data=dict(data))
        await self.rt.store.kv_set(KV.BOT_CONVERSATION, state.as_kv())
        flow = flow_class(self, state)
        await flow.start()
        return flow

    async def end_flow(self) -> None:
        await self.rt.store.kv_delete(KV.BOT_CONVERSATION)

    async def flow(self) -> Flow | None:
        """The active flow bound to this context (for callback handlers that advance it)."""
        state = await self.app.conversation()
        if state is None:
            return None
        flow_class = self.app.flow_classes.get(state.flow)
        if flow_class is None:
            return None
        return flow_class(self, state)

    # --- internals ---

    def _bot(self) -> BotGateway:
        if self.rt.bot is None:
            raise RuntimeError("no bot gateway in the runtime")
        return self.rt.bot

    async def _say_plain(self, text: str) -> None:
        """A toast that could not be a toast, as a plain message in the owner's chat."""
        try:
            await self._bot().send_text(self.chat_id, html_escape(text))
        except CuratorError as exc:
            log.warning("bot: could not tell the owner %r: %s", text, exc)

    def _text(self, key_or_html: str, fmt: dict[str, Any]) -> str:
        if _KEY_RE.match(key_or_html):
            return self.rt.t(key_or_html, **fmt)
        if fmt:
            raise TypeError("placeholders are filled for catalogue keys only, not for HTML")
        return key_or_html


def check_buttons(buttons: Buttons | None) -> None:
    """Refuse a keyboard whose callback payload Telegram would reject (``> 64`` bytes)."""
    for row in buttons or []:
        for button in row:
            if button.data is not None and len(button.data.encode()) > CALLBACK_DATA_MAX_BYTES:
                raise ValueError(
                    f"callback data {button.data!r} is {len(button.data.encode())} bytes; "
                    f"Telegram allows {CALLBACK_DATA_MAX_BYTES}"
                )


# --- the app ---------------------------------------------------------------------------------


class BotApp:
    """Owner guard, claim, routing, conversations and per-chat serialisation (§11.1).

    Construction subscribes to ``rt.bot`` and, when no owner exists yet, generates the claim
    code (``claim_code``) for the service to print; ``await start()`` records its hash in
    ``kv claim.code_hash`` and resets the attempt counter.
    """

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.commands: dict[str, tuple[CommandHandler, str | None]] = {}
        self.callbacks: dict[str, CallbackHandler] = {}
        self.flow_classes: dict[str, type[Flow]] = {}
        self.claim_code: str | None = None
        self._claim_hash: str | None = None
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        if rt.bot is not None:
            rt.bot.on_message(self.handle_message)
            rt.bot.on_callback(self.handle_callback)
        if self.owner_id == 0:
            self._new_claim_code()
        self.command("help", self._help, help_key="core_help_help")

    # --- registration ---

    def command(self, name: str, handler: CommandHandler, *, help_key: str | None = None) -> None:
        """Register ``/name``; ``help_key`` is the catalogue key of its one-line description."""
        self.commands[name.lower()] = (handler, help_key)

    def callback(self, prefix: str, handler: CallbackHandler) -> None:
        """Register the handler of callback payloads ``"<prefix>:<data>"``."""
        self.callbacks[prefix] = handler

    def flow(self, name: str, flow_class: type[Flow]) -> None:
        if not issubclass(flow_class, Flow):
            raise TypeError(f"{flow_class!r} is not a Flow subclass")
        self.flow_classes[name] = flow_class

    def flow_class(self, name: str) -> type[Flow]:
        try:
            return self.flow_classes[name]
        except KeyError:
            raise KeyError(f"flow {name!r} is not registered") from None

    # --- lifecycle ---

    async def start(self) -> None:
        """Persist the claim-code hash (when there is no owner) — call once after wiring."""
        if self._claim_hash is not None:
            await self._store_claim()

    @property
    def owner_id(self) -> int:
        try:
            return self.rt.settings.telegram.owner_id
        except ConfigError:
            return 0

    async def conversation(self) -> Conversation | None:
        return Conversation.from_kv(await self.rt.store.kv_get(KV.BOT_CONVERSATION))

    # --- gateway entry points ---

    async def handle_message(self, msg: BotMessage) -> None:
        """Owner guard, claim, then command or flow dispatch — serialised per chat."""
        if not msg.is_private:
            return
        owner_id = self.owner_id
        if owner_id == 0:
            async with self._locks[msg.chat_id]:
                await self._handle_claim(msg)
            return
        if msg.sender_id != owner_id:
            log.debug("ignored a private message from %d (not the owner)", msg.sender_id)
            return
        ctx = Ctx(self, message=msg, chat_id=msg.chat_id)
        await self._serialised(ctx, msg.chat_id, self._dispatch_message(ctx, msg))

    async def handle_callback(self, cb: BotCallback) -> None:
        """Obey only the owner, route by prefix (serialised), answer exactly once.

        A stranger gets a silent answer at once. For the owner the handler answers with its
        toast; when it has not answered within ``CALLBACK_ANSWER_DEADLINE`` seconds (it waits
        for the chat lock, or its work is slow) the core answers silently, and it always does
        when the handler never answered. A second answer would be refused by Telegram, so a
        silent one up front would swallow every toast (§11.1).
        """
        bot = self.rt.bot
        if bot is None:
            return
        owner_id = self.owner_id
        if owner_id == 0 or cb.sender_id != owner_id:
            await bot.answer_callback(cb.query_id)
            return
        ctx = Ctx(self, callback=cb, chat_id=owner_id)
        deadline = asyncio.create_task(self._answer_at_deadline(ctx))
        try:
            await self._serialised(ctx, cb.chat_id, self._dispatch_callback(ctx, cb))
        finally:
            deadline.cancel()
            if not ctx.answered:
                await ctx.answer()

    async def _answer_at_deadline(self, ctx: Ctx) -> None:
        await asyncio.sleep(CALLBACK_ANSWER_DEADLINE)
        try:
            # Shielded: the handler finishing now cancels this task, not the answer under way.
            await asyncio.shield(ctx.answer())
        except Exception:
            log.exception("bot: the silent callback answer failed")

    async def _serialised(self, ctx: Ctx, chat_id: int, work: Awaitable[None]) -> None:
        """Run one handler under the chat's lock (``ctx.unlocked()`` can lend it out)."""
        lock = self._locks[chat_id]
        ctx._lock = lock
        await lock.acquire()
        ctx._holds_lock = True
        try:
            await self._guarded(ctx, work)
        finally:
            if ctx._holds_lock:
                ctx._holds_lock = False
                lock.release()

    # --- dispatch ---

    async def _dispatch_message(self, ctx: Ctx, msg: BotMessage) -> None:
        parsed = _COMMAND_RE.match(msg.text.strip())
        state = await self.conversation()
        if parsed is not None and state is not None and self._at_secret_step(state):
            name, args = parsed.group(1).lower(), (parsed.group(2) or "").strip()
            if args or (name not in self.commands and name != "cancel"):
                # A password such as "/Secr3t" is not a command: the step reads it verbatim
                # and deletes it (§11.2).
                parsed = None
            else:
                await ctx.delete_incoming()  # whatever it was, it never stays in the chat
        if parsed is not None:
            name, args = parsed.group(1).lower(), (parsed.group(2) or "").strip()
            await ctx.end_flow()  # any command cancels the active flow (§11.1)
            entry = self.commands.get(name)
            if entry is None:
                await ctx.reply("unknown_command")
                return
            log.info("bot: /%s", name)
            await entry[0](ctx, args)
            return
        if state is None:
            await ctx.reply("core_idle")
            return
        flow_class = self.flow_classes.get(state.flow)
        handler = flow_class(ctx, state).handler_for(state.step) if flow_class else None
        if handler is None:
            log.warning("bot: conversation %s/%s has no handler; cancelled", state.flow, state.step)
            await ctx.end_flow()
            await ctx.reply("flow_cancelled")
            return
        await handler(msg.text)

    def _at_secret_step(self, state: Conversation) -> bool:
        flow_class = self.flow_classes.get(state.flow)
        return flow_class is not None and state.step in flow_class.secret_steps

    async def _dispatch_callback(self, ctx: Ctx, cb: BotCallback) -> None:
        prefix, _, data = cb.data.partition(":")
        handler = self.callbacks.get(prefix)
        if handler is None:
            log.warning("bot: no handler for callback prefix %r", prefix)
            return
        await handler(ctx, data)

    async def _guarded(self, ctx: Ctx, work: Awaitable[None]) -> None:
        """Run one handler; a failure is reported to the owner and never stops the bot."""
        try:
            await work
        except asyncio.CancelledError:
            raise
        except CuratorError as exc:
            log.warning("bot: handler failed: %s", exc)
            await self._report(ctx, exc)
        except Exception as exc:
            log.exception("bot: handler crashed")
            await self._report(ctx, exc)

    async def _report(self, ctx: Ctx, exc: Exception) -> None:
        key, fmt = _error_message(exc)
        try:
            await ctx.reply(key, **fmt)
        except CuratorError as inner:
            log.warning("bot: could not report the error to the owner: %s", inner)

    # --- /help ---

    async def _help(self, ctx: Ctx, args: str) -> None:
        lines = []
        for name, (_, help_key) in sorted(self.commands.items()):
            if help_key is not None:
                lines.append(self.rt.t("core_help_line", name=name, text=self.rt.t(help_key)))
            else:
                lines.append(self.rt.t("core_help_line_bare", name=name))
        await ctx.reply("core_help", lines="\n".join(lines))

    # --- claim ---

    async def _handle_claim(self, msg: BotMessage) -> None:
        """Before an owner exists only ``/start <code>`` or the bare code is accepted."""
        ctx = Ctx(self, message=msg, chat_id=msg.chat_id)
        parsed = _COMMAND_RE.match(msg.text.strip())
        if parsed is not None and parsed.group(1).lower() != "start":
            return
        candidate = (parsed.group(2) or "") if parsed is not None else msg.text
        candidate = _normalise_claim(candidate)
        if len(candidate) != CLAIM_CODE_LENGTH:
            # Not even code-shaped: a prompt, not an attempt — a stranger's chatter must not
            # burn the code the owner is about to read off the log.
            await ctx.reply("core_claim_prompt")
            return
        if self._claim_hash is not None and hmac.compare_digest(
            _claim_hash(candidate), self._claim_hash
        ):
            await self._claimed(ctx, msg.sender_id)
            return
        attempts = int(await self.rt.store.kv_get(KV.CLAIM_ATTEMPTS, 0) or 0) + 1
        if attempts >= CLAIM_MAX_ATTEMPTS:
            self._new_claim_code()
            await self._store_claim()
            log.warning("claim: %d wrong attempts; new claim code: %s", attempts, self.claim_code)
            await ctx.reply("core_claim_regenerated")
            return
        await self.rt.store.kv_set(KV.CLAIM_ATTEMPTS, attempts)
        await ctx.reply("core_claim_wrong", left=CLAIM_MAX_ATTEMPTS - attempts)

    async def _claimed(self, ctx: Ctx, owner_id: int) -> None:
        await self.rt.settings_file.set_value("telegram.owner_id", owner_id)
        await self.rt.store.kv_delete(KV.CLAIM_CODE_HASH)
        await self.rt.store.kv_delete(KV.CLAIM_ATTEMPTS)
        self.claim_code = None
        self._claim_hash = None
        log.info("bot claimed by user %d", owner_id)
        await ctx.reply("core_claimed")

    def _new_claim_code(self) -> None:
        code = "".join(secrets.choice(CLAIM_ALPHABET) for _ in range(CLAIM_CODE_LENGTH))
        self.claim_code = f"{code[:4]}-{code[4:]}"
        self._claim_hash = _claim_hash(code)

    async def _store_claim(self) -> None:
        await self.rt.store.kv_set(KV.CLAIM_CODE_HASH, self._claim_hash)
        await self.rt.store.kv_set(KV.CLAIM_ATTEMPTS, 0)


# --- helpers ---------------------------------------------------------------------------------


def _normalise_claim(text: str) -> str:
    """A code as typed (any case, with or without the dash) -> the bare uppercase code."""
    return "".join(ch for ch in text.strip().upper() if ch.isalnum())


def _claim_hash(code: str) -> str:
    return hashlib.sha256(_normalise_claim(code).encode()).hexdigest()


def _error_message(exc: Exception) -> tuple[str, dict[str, Any]]:
    """The catalogue key and placeholders that describe a handler failure to the owner.

    Feature modules render the failures they expect themselves (a channel the bot cannot
    post into names the channel, §11.2); this is the net under everything else.
    """
    if isinstance(exc, FloodWait):
        return "error_flood_wait", {"minutes": max(1, -(-exc.seconds // 60))}
    if isinstance(exc, SessionLost):
        return "error_session_lost", {}
    if isinstance(exc, ChatGone):
        return "error_chat_gone", {}
    if isinstance(exc, NotAllowed) and exc.reason == "not_a_member":
        return "error_not_a_member", {}
    if isinstance(exc, ConfigError):
        return "error_config", {"error": str(exc)}
    if isinstance(exc, TelegramUnavailable):
        return "core_error_unavailable", {}
    if type(exc) is CuratorError:
        # The services refuse with a plain CuratorError whose message is already the
        # catalogue sentence for the owner (digest "not live", a stale review tap, …).
        return "core_error_refused", {"error": str(exc)}
    if isinstance(exc, CuratorError):
        return "error_telegram", {"error": str(exc)}
    return "error_generic", {"error": type(exc).__name__}
