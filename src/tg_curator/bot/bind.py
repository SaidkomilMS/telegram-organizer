"""``/bind``: link (or re-link) the owner's Telegram account from the chat (DESIGN §11.2, §11.3).

The conversation is phone -> code -> optional 2FA password, driven through ``rt.account`` so
the bot and ``curator login`` share one state machine. Three things make it safe:

- **Nothing secret is kept.** The code and the password are read, the owner's message is
  deleted at once, and the value is handed to the account service inside the same handler
  call; the flow data persisted in ``kv bot.conversation`` holds only the step and a callback
  payload. Nothing here logs what the owner typed.
- **The code survives Telegram's own guard.** Telegram revokes a login code that appears as a
  plain message in any chat, so the owner is asked to type it with spaces between the digits
  and every non-digit is stripped before it is used. A code that still comes back rejected was
  most likely sent unmodified, so the reply says so and offers ``[Send a new code]``.
- **A restart costs one step, not a broken login.** ``phone_code_hash`` lives only in the
  gateway's memory, so after a restart the pending code is worthless: the flow notices that
  the account service no longer waits for a code (or password) and goes back to the phone.
  The same happens when an unfinished re-login expired.
- **Binding again never costs the bound account** (spec "every step can be repeated"). On a
  bound account the account service runs the new login beside the working session and swaps
  it in only once it fully succeeded, so intake keeps running meanwhile, and an abandoned
  ``/bind``, a wrong code or a restart changes nothing. Which path a login takes is the
  account service's decision (``begin``), shared with ``curator login``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from tg_curator.bot.core import Ctx, Flow
from tg_curator.errors import LoginError
from tg_curator.telegram.gateway import Button
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.bot.core import BotApp
    from tg_curator.contracts import AccountService
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

FLOW = "bind"
RESEND = "bd:resend"
_REJECTED_CODE = ("bad_code", "expired_code")
_PHONE_RE = re.compile(r"^\+?\d{7,15}$")
_PHONE_SEPARATORS = re.compile(r"[\s\-().]")


def register(app: BotApp) -> None:
    app.command("bind", bind_command, help_key="bind_help_bind")
    app.callback("bd", bind_callback)
    app.flow(FLOW, BindFlow)


async def bind_command(ctx: Ctx, args: str) -> None:
    await begin_bind(ctx)


async def begin_bind(ctx: Ctx, *, then: str | None = None) -> None:
    """Start the bind conversation; ``then`` is the callback payload of a [Continue] button
    offered after a successful login (``/setup`` passes its own, so it gets the owner back)."""
    await ctx.start_flow(FLOW, then=then)


async def logged_in_line(rt: Runtime) -> str | None:
    """The line "logged in as <name> (@user)" when the account is bound now, else ``None``."""
    if rt.account is None:
        return None
    status = await rt.account.status()
    if not status.bound or status.account is None:
        return None
    account = status.account
    username = f" (@{html_escape(account.username)})" if account.username else ""
    return rt.t("bind_logged_in_as", name=html_escape(account.name), username=username)


class BindFlow(Flow):
    """phone -> code -> password; the data is ``{"then": <callback payload> | None}`` only."""

    first_step = "phone"
    secret_steps = frozenset({"code", "password"})

    async def start(self) -> None:
        current = await logged_in_line(self.ctx.rt)
        if current is not None:
            await self.ctx.reply("bind_relink", current=current)
        await self.ctx.reply("bind_ask_phone")

    @Flow.step("phone")
    async def phone(self, text: str) -> None:
        phone = _PHONE_SEPARATORS.sub("", text.strip())
        if not _PHONE_RE.match(phone):
            await self.ctx.reply("bind_phone_invalid")
            return
        if not phone.startswith("+"):
            phone = "+" + phone
        rt = self.ctx.rt
        if rt.account is None or rt.user is None:
            await self.ctx.reply("bind_no_account_client")
            await self.end()
            return
        try:
            await rt.account.begin(phone)
        except LoginError as exc:
            await self.ctx.reply(html_escape(str(exc)))
            return
        await self.go("code")
        await self.ctx.reply("bind_ask_code")

    @Flow.step("code")
    async def code(self, text: str) -> None:
        await self.ctx.delete_incoming("code")  # the code must not stay in the chat (§11.2)
        digits = "".join(ch for ch in text if ch.isdigit())
        account = await self.pending("code")
        if account is None:
            return
        if not digits:
            await self.ctx.reply("bind_code_no_digits")
            return
        try:
            result = await account.code(digits)
        except LoginError as exc:
            if exc.reason in _REJECTED_CODE:
                log.info("bind: the login code was not accepted (%s)", exc.reason)
                await self.ctx.reply("bind_code_rejected", buttons=_resend_buttons(self.ctx.rt))
            else:
                await self.ctx.reply(html_escape(str(exc)))
            return
        if result == "password_needed":
            await self.go("password")
            await self.ctx.reply("bind_ask_password")
            return
        await self._finish()

    @Flow.step("password")
    async def password(self, text: str) -> None:
        await self.ctx.delete_incoming("password")  # must not stay in the chat (§11.2)
        account = await self.pending("password")
        if account is None:
            return
        try:
            await account.password(text)
        except LoginError as exc:
            await self.ctx.reply(html_escape(str(exc)))
            return
        await self._finish()

    async def pending(self, step: str) -> AccountService | None:
        """The account service when it still waits for ``step``; otherwise ``None`` and the
        flow is back at the phone step — the service restarted and the pending code died with
        the process (``phone_code_hash`` is never persisted, §11.2), or an unfinished re-login
        expired."""
        account = self.ctx.rt.account
        if account is not None and (await account.status()).step == step:
            return account
        log.info("bind: no %s is pending any more; asking for the phone again", step)
        await self.go("phone")
        await self.ctx.reply("bind_restarted")
        return None

    async def _finish(self) -> None:
        then = self.data.get("then")
        await self.end()
        line = await logged_in_line(self.ctx.rt) or self.ctx.rt.t("bind_logged_in")
        buttons = [[Button(self.ctx.rt.t("bind_continue"), data=then)]] if then else None
        await self.ctx.reply(self.ctx.rt.t("bind_done", line=line), buttons=buttons)


async def bind_callback(ctx: Ctx, data: str) -> None:
    """``bd:resend`` — [Send a new code]."""
    if f"bd:{data}" != RESEND:
        await ctx.reply("unknown_choice")
        return
    flow = await ctx.flow()
    if not isinstance(flow, BindFlow) or flow.current_step != "code":
        await ctx.reply("unknown_choice")
        return
    account = await flow.pending("code")
    if account is None:
        return
    try:
        await account.resend()
    except LoginError as exc:
        await ctx.reply(html_escape(str(exc)))
        return
    await ctx.reply("bind_code_resent")


def _resend_buttons(rt: Runtime) -> list[list[Button]]:
    return [[Button(rt.t("bind_send_new_code"), data=RESEND)]]
