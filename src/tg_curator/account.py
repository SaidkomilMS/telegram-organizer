"""Binding the account: the login state machine and the session watchdog (DESIGN §11.3).

The service wraps the four login calls of the user gateway so that the bind flow and the CLI
share one state machine and one vocabulary of failures: every ``LoginError`` is re-raised
with the plain catalogue sentence as its message. Nothing secret is kept here — the code and
the password are handed straight to the gateway, ``phone_code_hash`` lives in the gateway's
memory, and the phone number appears in the log only masked.

Binding again must never cost the account it replaces (spec "every step can be repeated",
"/bind sets it up again"). So when the session works, the login runs beside it on the
gateway's second client and is swapped in only once it has fully succeeded; until then
intake and every loop carry on with the old session, and a wrong code, a failed step, an
abandoned or expired login or a restart leaves it bound. Only without a working session (a
first bind, or after a loss) does the login start from an empty one, as it always did.

The watchdog is the part of the service that notices when the account went quiet: no message
ingested for an hour *and* ``ping()`` failing is the signal (a quiet Sunday is not), and it
warns the owner once, re-arming only when intake resumes. A session loss reported by the
gateway puts the service into setup mode — ``setup_mode`` is the flag ``service.py`` reads to
stop the account loops until ``account_bound``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from tg_curator.domain import KV, AccountStatus
from tg_curator.errors import CuratorError, FloodWait, LoginError, SessionLost
from tg_curator.logging_setup import mask_phone
from tg_curator.runtime import EVENT_ACCOUNT_BOUND, EVENT_SESSION_LOST

if TYPE_CHECKING:
    from tg_curator.runtime import Runtime
    from tg_curator.telegram.gateway import Account, UserGateway

log = logging.getLogger(__name__)

WATCHDOG_INTERVAL = timedelta(minutes=5)
"""How often ``watchdog_tick`` runs (§11.3)."""
STALL_AFTER = timedelta(minutes=60)
"""Silence that, together with a failing ``ping()``, counts as "intake stopped" (§14.11)."""
RELOGIN_TTL = timedelta(minutes=30)
"""How long a re-login waits for its code or password. The bind flow cannot tell when the
owner gave up (any command just ends it), and the second client and its file must not stay
open for ever; a code is long expired by then anyway."""

LoginStep = Literal["none", "phone", "code", "password", "ok"]

_ERROR_KEYS: dict[str, str] = {
    "bad_code": "account_error_bad_code",
    "expired_code": "account_error_expired_code",
    "bad_password": "account_error_bad_password",
    "bad_phone": "account_error_bad_phone",
    "flood": "account_error_flood",
    "other": "account_error_other",
}


class AccountService:
    """``rt.account``: status / begin / resend / code / password, plus the watchdog.

    ``setup_mode`` is public on purpose: ``service.py`` sets it when the account is not
    authorised at start and reads it to decide which loops run; a session loss sets it here
    and a successful bind clears it.
    """

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt
        self.setup_mode = False
        self._step: LoginStep = "none"
        self._relogin_since: datetime | None = None  # a re-login runs beside the session
        self._warned = False
        self._quiet_since: datetime | None = None
        if rt.user is not None:
            rt.user.on_session_lost(self._on_session_lost)

    # --- status ---

    async def status(self) -> AccountStatus:
        """``bound`` is what the gateway says now; ``step`` is what the bind flow asks next.
        While a re-login waits for its code the working account is still bound, and says so."""
        user = self._rt.user
        if user is None:
            return AccountStatus(bound=False, account=None, step="none")
        await self._expire_relogin()
        try:
            account = await user.me()
        except CuratorError as exc:
            log.debug("account status: %s", exc)
            account = None
        if self._step in ("code", "password"):
            return AccountStatus(bound=account is not None, account=account, step=self._step)
        if account is not None:
            self._step = "ok"
            return AccountStatus(bound=True, account=account, step="ok")
        if self._step in ("none", "ok"):
            self._step = "phone"
        return AccountStatus(bound=False, account=None, step=self._step)

    # --- the login steps ---

    async def begin(self, phone: str) -> None:
        """Ask Telegram for a code; any login still pending is abandoned first.

        With a working session the login runs beside it (``begin_relogin``) and replaces it
        only when it succeeds. Without one, the session is dropped and the login starts from
        an empty one (§11.2 "always starts fresh"), after the account loops were stopped."""
        user = self._user()
        await self._abandon_relogin()
        self._step = "phone"  # an earlier code died with the login it belonged to
        working = await self._working_account(user)
        log.info("account: requesting a login code for %s", mask_phone(phone))
        try:
            if working is not None:
                self._relogin_since = self._rt.clock.now()
                await user.begin_relogin()
            else:
                await self._enter_setup_mode()
                await user.reset_session()
                await user.connect()
            await user.send_code(phone)
        except BaseException as exc:
            await self._abandon_relogin()
            if isinstance(exc, LoginError | FloodWait):
                raise self._translate(exc) from exc
            raise
        self._step = "code"

    async def resend(self) -> None:
        """[Send a new code] — only while a code is pending."""
        await self._expire_relogin()
        if self._step != "code":
            raise LoginError("other", self._rt.t("account_error_no_code_pending"))
        try:
            await self._user().resend_code()
        except (LoginError, FloodWait) as exc:
            raise self._translate(exc) from exc

    async def code(self, code: str) -> Literal["ok", "password_needed"]:
        """Submit the code (digits only; separators are stripped by the bind flow)."""
        await self._expire_relogin()
        if self._step != "code":
            raise LoginError("other", self._rt.t("account_error_no_code_pending"))
        try:
            result = await self._user().sign_in(code)
        except (LoginError, FloodWait) as exc:
            raise self._translate(exc) from exc
        if result == "password_needed":
            self._step = "password"
            return result
        await self._bound()
        return "ok"

    async def password(self, password: str) -> None:
        """Submit the 2FA password; on success the account is bound."""
        await self._expire_relogin()
        if self._step != "password":
            raise LoginError("other", self._rt.t("account_error_no_password_pending"))
        try:
            await self._user().sign_in_password(password)
        except (LoginError, FloodWait) as exc:
            raise self._translate(exc) from exc
        await self._bound()

    # --- the watchdog ---

    async def watchdog_tick(self) -> None:
        """One check (§11.3, §14.11), run every ``WATCHDOG_INTERVAL`` by the service's
        supervisor: warn once when intake is silent for an hour AND the account does not
        answer; re-arm as soon as a message is ingested again. A re-login left unfinished
        is abandoned here too, since this loop keeps running through one."""
        await self._expire_relogin()
        if self.setup_mode or self._rt.user is None:
            return
        now = self._rt.clock.now()
        if self._quiet_since is None:
            self._quiet_since = now
        warned = bool(await self._rt.store.kv_get(KV.WATCHDOG_WARNED, False))
        last = await self._last_ingested()
        if last is not None and last > self._quiet_since:
            self._quiet_since = last
        if now - self._quiet_since < STALL_AFTER:
            if warned:
                await self._rt.store.kv_delete(KV.WATCHDOG_WARNED)  # intake resumed: re-arm
            return
        if warned:
            return
        try:
            alive = await self._rt.user.ping()
        except SessionLost:
            return  # the gateway's on_session_lost handler takes over
        except CuratorError as exc:
            log.warning("watchdog: ping failed: %s", exc)
            alive = False
        if alive:
            return
        log.warning("watchdog: nothing ingested since %s and the account does not answer", last)
        await self._rt.store.kv_set(KV.WATCHDOG_WARNED, now.isoformat())
        await self._rt.notifier.intake_warning("stalled")

    # --- internals ---

    def _user(self) -> UserGateway:
        if self._rt.user is None:
            raise LoginError("other", self._rt.t("account_error_other", detail="no account client"))
        return self._rt.user

    def _translate(self, exc: LoginError | FloodWait) -> LoginError:
        """The same ``reason`` with the plain catalogue sentence as the message."""
        if isinstance(exc, FloodWait):
            minutes = max(1, -(-exc.seconds // 60))
            return LoginError("flood", self._rt.t("account_error_flood", minutes=minutes))
        key = _ERROR_KEYS.get(exc.reason, "account_error_other")
        return LoginError(exc.reason, self._rt.t(key, detail=str(exc)))

    async def _working_account(self, user: UserGateway) -> Account | None:
        """The account the current session serves, ``None`` when it serves none.

        A question Telegram left unanswered is not "none": dropping a working session because
        Telegram was briefly unreachable is the very loss a re-bind must never cause, so the
        owner is asked to try again instead."""
        try:
            return await user.me()
        except SessionLost:
            return None  # the gateway's loss procedure ran: nothing is left to keep
        except FloodWait as exc:
            raise self._translate(exc) from exc
        except CuratorError as exc:
            log.warning("account: cannot tell whether the session works (%s); nothing changed", exc)
            raise LoginError("other", self._rt.t("account_error_unreachable")) from exc

    async def _enter_setup_mode(self) -> None:
        """The session is about to be dropped for a login from an empty one: stop what uses it.

        Without a working session this is setup mode already, except when the session stopped
        answering "authorised" without the gateway noticing a loss. ``session_lost`` with
        ``reason="rebind"`` then stops the account loops (a loop calling the new, not yet
        signed-in session would be answered "unregistered"); the owner is not warned, and
        ``account_bound`` starts everything again."""
        if self.setup_mode:
            return
        self.setup_mode = True
        log.info("account: dropping the session for a new login; setup mode until it succeeds")
        await self._rt.events.emit(EVENT_SESSION_LOST, reason="rebind")

    async def _expire_relogin(self) -> None:
        """Abandon a re-login nobody finished within ``RELOGIN_TTL``; the working session
        never noticed it."""
        since = self._relogin_since
        if since is None or self._rt.clock.now() - since < RELOGIN_TTL:
            return
        log.info("account: the re-login was not finished in time; abandoned")
        await self._abandon_relogin()
        self._step = "none"  # status() works out "ok" or "phone" from the session again

    async def _abandon_relogin(self) -> None:
        if self._relogin_since is None:
            return
        self._relogin_since = None
        if self._rt.user is not None:
            await self._rt.user.cancel_relogin()

    async def _bound(self) -> None:
        user = self._user()
        self._relogin_since = None  # swapped in by the gateway: nothing left to abandon
        account = await user.me()
        self._step = "ok"
        self.setup_mode = False
        self._quiet_since = self._rt.clock.now()
        await self._rt.store.kv_delete(KV.WATCHDOG_WARNED)
        if account is not None:
            await self._rt.store.kv_set(KV.ACCOUNT_ID, account.id)
            if self._rt.settings.telegram.owner_id == 0:
                await self._rt.settings_file.set_value("telegram.owner_id", account.id)
            log.info(
                "account: signed in as %s (@%s, %s)",
                account.name,
                account.username or "-",
                mask_phone(account.phone),
            )
        await self._rt.events.emit(EVENT_ACCOUNT_BOUND)

    async def _on_session_lost(self, reason: str) -> None:
        log.warning("account: session lost (%s); setup mode until /bind", reason)
        self.setup_mode = True
        if self._relogin_since is None:
            self._step = "phone"
        # else a re-login is under way on its own client: finishing it binds the account again
        await self._rt.notifier.intake_warning("session_lost")
        await self._rt.events.emit(EVENT_SESSION_LOST, reason=reason)

    async def _last_ingested(self) -> datetime | None:
        raw = await self._rt.store.kv_get(KV.INTAKE_LAST_MESSAGE_AT)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            log.warning("watchdog: unreadable intake.last_message_at %r", raw)
            return None
