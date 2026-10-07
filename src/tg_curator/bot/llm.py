"""``/llm``: choose, test and save the language model (DESIGN §11.2 llm.py, §14.18, §15).

Three choices — None, Self-hosted, Provider — each introduced with one line on cost and
privacy. Nothing is written to the settings file until one real post has been summarised by
the chosen model: a wrong key, model name or address therefore never ends up configured. The
test post is the most recent stored post that is (or would be) a digest pick, because that
is exactly the kind of text the model will see anyway; without one, a bundled sample is used
and the reply says so. Chat history is never read for this.

Keys and tokens are read from the owner's message, which is deleted at once, and are kept
only in this process (``LlmWizard.secret``) until the test succeeds and ``[llm]`` is written
in one update. The conversation itself is persisted by the core and so never holds them: a
restart in the middle of the flow simply asks for the key again.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import sqlalchemy as sa
import tomlkit
from tomlkit import TOMLDocument

from tg_curator.bot.core import BotApp, Ctx, Flow
from tg_curator.bot.setup import CONTINUE as CONTINUE_SETUP
from tg_curator.bot.setup import mark_llm_chosen
from tg_curator.clock import local_date
from tg_curator.config import Settings, validate_settings
from tg_curator.db import schema
from tg_curator.domain import PostStatus
from tg_curator.llm import registry
from tg_curator.llm.base import LLMAuthError, LLMError
from tg_curator.llm.factory import make_llm
from tg_curator.llm.providers import normalise_base_url
from tg_curator.telegram.gateway import Button, Buttons
from tg_curator.textutil import html_escape

if TYPE_CHECKING:
    from tg_curator.contracts import LLM
    from tg_curator.runtime import Runtime

log = logging.getLogger(__name__)

PREFIX = "lm"
FLOW = "llm"
TEST_MIN_CHARS = 200
"""A test post shorter than this says little about how the model summarises (§11.2)."""
TEST_STATUSES = (
    PostStatus.digest.value,
    PostStatus.digested.value,
    PostStatus.published.value,
    PostStatus.queued.value,
    PostStatus.held.value,
)
"""Posts that were or would be digest picks, i.e. that reach the model anyway. Nothing else is
ever sent for the test: an unsorted, rejected or group post would leave the server only for
this, which the privacy promise rules out (§14.18); the bundled sample is used instead."""
SLOW_SECONDS = 20.0
"""A self-hosted model slower than this per line makes the nightly digest late (§15)."""
MODEL_BUTTONS = 8
"""How many server-reported models get a button; any other id can be typed."""


# --- the test post ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestPost:
    """What the connection test summarises; ``post_id`` is ``None`` for the bundled sample."""

    __test__ = False  # not a pytest class, despite the name

    post_id: int | None
    text: str
    label: str
    """"<source, date>" for the privacy line, already HTML-escaped."""


async def pick_test_post(rt: Runtime) -> TestPost:
    """The post the test sends: the newest stored post of >= 200 characters that was or would
    be a digest pick; else ``test_sample`` (§11.2, §14.18). Shared with the CLI."""
    p = schema.posts.c
    stmt = (
        sa.select(p.id)
        .where(sa.func.length(p.text) >= TEST_MIN_CHARS)
        .where(p.status.in_(TEST_STATUSES))
        .order_by(p.posted_at.desc(), p.id.desc())
        .limit(1)
    )
    rows = await rt.store.execute(stmt)
    post_id = rows[0][0] if isinstance(rows, list) and rows else None
    return await load_test_post(rt, post_id)


async def load_test_post(rt: Runtime, post_id: int | None) -> TestPost:
    """The test post with this id (the one the privacy line named); the sample when it is
    ``None`` or gone."""
    post = await rt.store.get_post(post_id) if post_id is not None else None
    if post is None:
        return TestPost(None, rt.t("test_sample"), rt.t("llm_sample_label"))
    chat = await rt.store.get_chat(post.chat_id)
    title = chat.title if chat is not None else str(post.chat_id)
    day = local_date(post.posted_at, rt.settings.general.timezone)
    return TestPost(post.id, post.text, html_escape(f"{title}, {day.isoformat()}"))


# --- the wizard ------------------------------------------------------------------------------


class LlmFlow(Flow):
    """The persisted part of ``/llm``: ids, names and addresses only, never a key or token."""

    first_step = "mode"
    secret_steps = frozenset({"key", "token"})
    wizard: ClassVar[LlmWizard]

    @Flow.step("preset")
    async def on_preset(self, text: str) -> None:
        await self.wizard.address(self, text)  # an address typed instead of a preset button

    @Flow.step("address")
    async def on_address(self, text: str) -> None:
        await self.wizard.address(self, text)

    @Flow.step("token")
    async def on_token(self, text: str) -> None:
        await self.wizard.token(self, text)

    @Flow.step("provider")
    async def on_provider(self, text: str) -> None:
        await self.ctx.reply("llm_pick_button")

    @Flow.step("key")
    async def on_key(self, text: str) -> None:
        await self.wizard.key(self, text)

    @Flow.step("model")
    async def on_model(self, text: str) -> None:
        await self.wizard.model(self, text.strip())

    @Flow.step("cap")
    async def on_cap(self, text: str) -> None:
        await self.wizard.cap(self, text)


class LlmWizard:
    """Handlers of ``/llm`` and its ``lm:`` buttons, plus the one in-memory secret.

    ``client`` is handed to every probe and to the tested model (``None`` lets each own its
    own connection); ``timer`` measures the test. Both exist so tests run without a network.
    """

    def __init__(
        self,
        rt: Runtime,
        *,
        client: httpx.AsyncClient | None = None,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rt = rt
        self.client = client
        self.timer = timer
        self.secret: str | None = None

    # --- /llm and the menu ---

    async def command(self, ctx: Ctx, args: str) -> None:
        self.secret = None
        await ctx.reply("llm_menu", buttons=self._menu_buttons(), current=self._current())

    def _current(self) -> str:
        cfg = self.rt.settings.llm
        if cfg.mode == "selfhosted":
            return self.rt.t(
                "llm_current_selfhosted",
                model=html_escape(cfg.model),
                base_url=html_escape(cfg.base_url),
            )
        if cfg.mode == "provider":
            return self.rt.t(
                "llm_current_provider",
                provider=html_escape(_provider_label(cfg.provider)),
                model=html_escape(cfg.model),
                cap=self._cap_text(cfg.monthly_cap_usd),
            )
        return self.rt.t("llm_current_none")

    def _cap_text(self, cap: float) -> str:
        return self.rt.t("llm_cap_value", cap=cap) if cap > 0 else self.rt.t("llm_cap_unset")

    def _menu_buttons(self) -> Buttons:
        t = self.rt.t
        rows = [
            [
                Button(t("llm_choice_none"), f"{PREFIX}:m:0"),
                Button(t("llm_choice_selfhosted"), f"{PREFIX}:m:1"),
                Button(t("llm_choice_provider"), f"{PREFIX}:m:2"),
            ]
        ]
        mode = self.rt.settings.llm.mode
        if mode != "none":
            extra = [self._opinion_button()]
            if mode == "provider":
                extra.insert(0, Button(t("llm_cap_button"), f"{PREFIX}:cap"))
            rows.append(extra)
        return rows

    def _opinion_button(self) -> Button:
        state = self.rt.t("on") if self.rt.settings.sorting.second_opinion else self.rt.t("off")
        return Button(self.rt.t("llm_opinion_button", state=state), f"{PREFIX}:so")

    # --- the buttons ---

    async def callback(self, ctx: Ctx, data: str) -> None:
        action, _, arg = data.partition(":")
        if action == "m":
            await self._mode(ctx, arg)
        elif action == "cap":
            if self.rt.settings.llm.mode != "provider":
                await ctx.reply("llm_cap_provider_only")
                return
            await self._ask_cap(await self._cap_flow(ctx))
        elif action == "c0":
            await self._with_flow(ctx, "cap", lambda flow: self._set_cap(flow, 0.0))
        elif action == "so":
            await self._toggle_opinion(ctx)
        elif action == "p":
            await self._with_flow(ctx, "preset", lambda flow: self._preset(flow, arg))
        elif action == "u":
            await self._with_flow(ctx, "address", self._default_address)
        elif action == "nt":
            await self._with_flow(ctx, "token", self._no_token)
        elif action == "pr":
            await self._with_flow(ctx, "provider", lambda flow: self._provider(flow, arg))
        elif action == "md":
            await self._with_flow(ctx, "model", lambda flow: self._model_button(flow, arg))
        elif action == "t":
            await self._with_flow(ctx, "model", self._test)
        else:
            await ctx.reply("unknown_choice")

    async def _cap_flow(self, ctx: Ctx) -> LlmFlow:
        flow = await ctx.start_flow(FLOW, mode="provider")
        assert isinstance(flow, LlmFlow)
        return flow

    async def _with_flow(self, ctx: Ctx, step: str, action: Callable[[LlmFlow], Any]) -> None:
        """Run ``action`` on the active ``/llm`` flow when it is at ``step``; a stale button
        (another command ran meanwhile, or the step moved on) gets "not available"."""
        flow = await ctx.flow()
        if not isinstance(flow, LlmFlow) or flow.current_step != step:
            await ctx.reply("unknown_choice")
            return
        await action(flow)

    async def _mode(self, ctx: Ctx, arg: str) -> None:
        self.secret = None
        if arg == "0":
            await ctx.end_flow()
            await self._write(
                {"mode": "none", "provider": "", "model": "", "base_url": "", "api_key": ""}
            )
            await self._install(make_llm(self.rt.settings, self.rt.store))
            # "No model" leaves no trace in the settings, so /setup's model step is marked
            # done here when the walkthrough is waiting at it (it was opened from there).
            if await mark_llm_chosen(self.rt):
                buttons = [[Button(self.rt.t("llm_continue_setup"), CONTINUE_SETUP)]]
                await ctx.reply("llm_saved_none", buttons=buttons)
            else:
                await ctx.reply("llm_saved_none")
            return
        post = await pick_test_post(self.rt)
        if arg == "1":
            flow = await ctx.start_flow(FLOW, mode="selfhosted", post_id=post.post_id)
            await flow.go("preset")
            presets = registry.selfhosted_presets()
            buttons = [
                [Button(p.label, f"{PREFIX}:p:{i}") for i, p in enumerate(presets)],
                [Button(self.rt.t("llm_preset_other"), f"{PREFIX}:p:{len(presets)}")],
            ]
            await ctx.reply("llm_selfhosted_intro", buttons=buttons)
        elif arg == "2":
            flow = await ctx.start_flow(FLOW, mode="provider", post_id=post.post_id)
            await flow.go("provider")
            choices = registry.providers()
            buttons = [
                [Button(p.label, f"{PREFIX}:pr:{i}") for i, p in enumerate(choices[j : j + 3], j)]
                for j in range(0, len(choices), 3)
            ]
            await ctx.reply("llm_provider_intro", buttons=buttons)
        else:
            await ctx.reply("unknown_choice")

    # --- self-hosted: preset -> address -> token -> model ---

    async def _preset(self, flow: LlmFlow, arg: str) -> None:
        presets = registry.selfhosted_presets()
        index = _index(arg, len(presets) + 1)
        if index is None:
            await flow.ctx.reply("unknown_choice")
            return
        await flow.go("address", preset=index)
        if index == len(presets):
            await flow.ctx.reply("llm_address_other")
            return
        preset = presets[index]
        button = Button(self.rt.t("llm_address_use", url=preset.base_url), f"{PREFIX}:u")
        await flow.ctx.reply(
            "llm_address_preset",
            buttons=[[button]],
            label=html_escape(preset.label),
            url=html_escape(preset.base_url),
        )

    async def _default_address(self, flow: LlmFlow) -> None:
        presets = registry.selfhosted_presets()
        index = flow.data.get("preset")
        if not isinstance(index, int) or not 0 <= index < len(presets):
            await flow.ctx.reply("llm_address_other")
            return
        await self.address(flow, presets[index].base_url)

    async def address(self, flow: LlmFlow, text: str) -> None:
        try:
            base_url = normalise_base_url(text)
        except LLMError:
            await flow.ctx.reply("llm_address_bad")
            return
        await flow.go("token", base_url=base_url)
        post = await load_test_post(self.rt, flow.data.get("post_id"))
        privacy = self.rt.t(
            "llm_privacy_selfhosted", base_url=html_escape(base_url), test_post=post.label
        )
        no_token = Button(self.rt.t("llm_token_none"), f"{PREFIX}:nt")
        await flow.ctx.reply("llm_token_prompt", buttons=[[no_token]], privacy=privacy)

    async def token(self, flow: LlmFlow, text: str) -> None:
        await flow.ctx.delete_incoming("token")  # the token never stays in the chat
        self.secret = text.strip()
        await flow.go("token", has_token=bool(self.secret))
        await self._server_models(flow)

    async def _no_token(self, flow: LlmFlow) -> None:
        self.secret = ""
        await flow.go("token", has_token=False)
        await self._server_models(flow)

    async def _server_models(self, flow: LlmFlow) -> None:
        """Ask the server which models it serves (the token is sent on this probe, §15)."""
        base_url = str(flow.data.get("base_url", ""))
        try:
            models = await registry.list_models(
                "selfhosted", self.secret or "", base_url=base_url, client=self.client
            )
        except LLMAuthError:
            self.secret = None
            await flow.ctx.reply("llm_token_refused")
            return
        except LLMError as exc:
            await flow.go("address")
            await flow.ctx.reply(
                "llm_server_unreachable",
                base_url=html_escape(base_url),
                error=html_escape(str(exc)),
            )
            return
        await self._ask_model(flow, models[:MODEL_BUTTONS])

    # --- provider: provider -> key -> model ---

    async def _provider(self, flow: LlmFlow, arg: str) -> None:
        choices = registry.providers()
        index = _index(arg, len(choices))
        if index is None:
            await flow.ctx.reply("unknown_choice")
            return
        info = choices[index]
        await flow.go("key", provider=info.key)
        post = await load_test_post(self.rt, flow.data.get("post_id"))
        privacy = self.rt.t(
            "llm_privacy_provider", provider=html_escape(info.label), test_post=post.label
        )
        await flow.ctx.reply("llm_key_prompt", provider=html_escape(info.label), privacy=privacy)

    async def key(self, flow: LlmFlow, text: str) -> None:
        await flow.ctx.delete_incoming("key")  # the key never stays in the chat
        secret = text.strip()
        provider = str(flow.data.get("provider", ""))
        if not secret:
            await flow.ctx.reply("llm_key_empty")
            return
        try:
            await registry.validate_key(provider, secret, client=self.client)
        except LLMAuthError:
            await flow.ctx.reply("llm_key_refused", provider=html_escape(_provider_label(provider)))
            return
        except LLMError as exc:
            await flow.ctx.reply("llm_key_unchecked", error=html_escape(str(exc)))
            return
        self.secret = secret
        models = next((p.models for p in registry.providers() if p.key == provider), [])
        await self._ask_model(flow, list(models))

    # --- the model and the test ---

    async def _ask_model(self, flow: LlmFlow, models: list[str]) -> None:
        await flow.go("model", models=models)
        if not models:
            await flow.ctx.reply("llm_model_type")
            return
        buttons = [[Button(m, f"{PREFIX}:md:{i}")] for i, m in enumerate(models)]
        await flow.ctx.reply("llm_model_prompt", buttons=buttons)

    async def _model_button(self, flow: LlmFlow, arg: str) -> None:
        models = flow.data.get("models") or []
        index = _index(arg, len(models))
        if index is None:
            await flow.ctx.reply("unknown_choice")
            return
        await self.model(flow, str(models[index]))

    async def model(self, flow: LlmFlow, name: str) -> None:
        if not name:
            await flow.ctx.reply("llm_model_type")
            return
        await flow.go("model", model=name)
        await self._test(flow)

    async def _test(self, flow: LlmFlow) -> None:
        """Summarise the test post with the candidate settings; write them only on success."""
        rt = self.rt
        data = flow.data
        provider_mode = data.get("mode") == "provider"
        if self.secret is None and (provider_mode or data.get("has_token")):
            # The service restarted mid-flow: the secret lived only in the old process.
            await flow.go("key" if provider_mode else "token")
            await flow.ctx.reply("llm_secret_again" if provider_mode else "llm_token_again")
            return
        fields = self._fields(data)
        candidate = _with_llm(rt.settings, fields)
        post = await load_test_post(rt, data.get("post_id"))
        await flow.ctx.reply(
            "llm_testing", model=html_escape(fields["model"]), test_post=post.label
        )
        llm = make_llm(
            candidate, rt.store, notifier=rt.notifier, clock=rt.clock, client=self.client
        )
        outcome = await registry.probe(
            llm, post.text, rt.settings.digest.line_chars, timer=self.timer
        )
        line, seconds = outcome.line, outcome.seconds
        if not line:
            await _close(llm)
            retry = Button(rt.t("retry"), f"{PREFIX}:t")
            await flow.ctx.reply(
                "llm_test_failed",
                buttons=[[retry]],
                seconds=f"{seconds:.1f}",
                reason=failure_reason(rt.t, outcome.failure),
            )
            return
        await self._write(fields)
        await self._install(llm)
        self.secret = None
        log.info(
            "llm: %s %s connected after a %.1f s test", fields["mode"], fields["model"], seconds
        )
        notes = []
        if post.post_id is None:
            notes.append(rt.t("llm_test_sample_used"))
        if seconds > SLOW_SECONDS:
            notes.append(rt.t("llm_test_slow", seconds=f"{seconds:.0f}"))
        await flow.ctx.reply(
            "llm_test_ok",
            line=html_escape(line),
            seconds=f"{seconds:.1f}",
            notes="".join(f"\n\n{n}" for n in notes),
        )
        if provider_mode:
            await self._ask_cap(flow)
        else:
            await flow.end()
            await flow.ctx.reply("llm_saved", buttons=[[self._opinion_button()]])

    def _fields(self, data: dict[str, Any]) -> dict[str, str]:
        """The ``[llm]`` keys of the candidate, all of them, for one settings update."""
        if data.get("mode") == "provider":
            return {
                "mode": "provider",
                "provider": str(data.get("provider", "")),
                "model": str(data.get("model", "")),
                "base_url": "",
                "api_key": self.secret or "",
            }
        return {
            "mode": "selfhosted",
            "provider": "",
            "model": str(data.get("model", "")),
            "base_url": str(data.get("base_url", "")),
            "api_key": self.secret or "",
        }

    # --- monthly cap and second opinion ---

    async def _ask_cap(self, flow: LlmFlow) -> None:
        await flow.go("cap")
        no_cap = Button(self.rt.t("llm_cap_none_button"), f"{PREFIX}:c0")
        await flow.ctx.reply(
            "llm_cap_prompt",
            buttons=[[no_cap]],
            current=self._cap_text(self.rt.settings.llm.monthly_cap_usd),
        )

    async def cap(self, flow: LlmFlow, text: str) -> None:
        raw = text.strip().lstrip("$").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            await flow.ctx.reply("llm_cap_bad")
            return
        if value < 0 or not math.isfinite(value):
            await flow.ctx.reply("llm_cap_bad")
            return
        await self._set_cap(flow, value)

    async def _set_cap(self, flow: LlmFlow, value: float) -> None:
        await self.rt.settings_file.set_value("llm.monthly_cap_usd", value)
        await self._install(
            make_llm(
                self.rt.settings,
                self.rt.store,
                notifier=self.rt.notifier,
                clock=self.rt.clock,
                client=self.client,
            )
        )
        await flow.end()
        await flow.ctx.reply(
            "llm_cap_saved", buttons=[[self._opinion_button()]], cap=self._cap_text(value)
        )

    async def _toggle_opinion(self, ctx: Ctx) -> None:
        cfg = self.rt.settings.llm
        if cfg.mode == "none":
            await ctx.reply("llm_opinion_needs_model")
            return
        turn_on = not self.rt.settings.sorting.second_opinion
        await self.rt.settings_file.set_value("sorting.second_opinion", turn_on)
        if not turn_on:
            await ctx.reply("llm_opinion_off")
            return
        if cfg.mode == "provider":
            target = html_escape(_provider_label(cfg.provider))
        else:
            target = self.rt.t("llm_opinion_server", base_url=html_escape(cfg.base_url))
        await ctx.reply("llm_opinion_on", target=target)

    # --- writing ---

    async def _write(self, fields: dict[str, Any]) -> Settings:
        """All ``[llm]`` keys in ONE update: the file is never half-way between two models.
        The update raises ``settings_changed`` through the runtime's hook."""

        def mutate(doc: TOMLDocument) -> None:
            if "llm" not in doc:
                doc["llm"] = tomlkit.table()
            table: Any = doc["llm"]
            for name, value in fields.items():
                table[name] = value

        return await self.rt.settings_file.update(mutate)

    async def _install(self, llm: LLM) -> None:
        """Make the saved model the one the digest, discovery and the sorter use from now on."""
        old, self.rt.llm = self.rt.llm, llm
        if old is not llm:
            await _close(old)


def _with_llm(settings: Settings, fields: dict[str, Any]) -> Settings:
    """The current settings with the candidate ``[llm]`` keys, validated like the file is (a
    ``ConfigError`` sentence names the key)."""
    data = settings.model_dump()
    data["llm"] = {**data["llm"], **fields}
    return validate_settings(data)


def _provider_label(key: str) -> str:
    return next((p.label for p in registry.providers() if p.key == key), key)


def _index(arg: str, size: int) -> int | None:
    return int(arg) if arg.isdigit() and int(arg) < size else None


async def _close(llm: Any) -> None:
    close = getattr(llm, "aclose", None)
    if close is not None:
        await close()


def register(
    app: BotApp,
    *,
    client: httpx.AsyncClient | None = None,
    timer: Callable[[], float] = time.monotonic,
) -> LlmWizard:
    """Wire ``/llm``, the ``lm:`` buttons and the ``llm`` flow into ``app``."""
    wizard = LlmWizard(app.rt, client=client, timer=timer)
    flow_class = type("BoundLlmFlow", (LlmFlow,), {"wizard": wizard})
    app.command("llm", wizard.command, help_key="llm_help_llm")
    app.callback(PREFIX, wizard.callback)
    app.flow(FLOW, flow_class)
    return wizard


FAILURE_KEYS = {
    "auth": "llm_fail_auth",
    "model": "llm_fail_model",
    "timeout": "llm_fail_timeout",
    "unreachable": "llm_fail_unreachable",
    "unavailable": "llm_fail_unavailable",
    "credit": "llm_fail_credit",
    "cap": "llm_fail_cap",
    "request": "llm_fail_request",
    "empty": "llm_fail_empty",
}
"""``ProbeFailure.kind`` -> the sentence that says what to fix (SPEC: every step either works
or says exactly what to fix)."""


def failure_reason(t: Callable[..., str], failure: registry.ProbeFailure | None) -> str:
    """The owner-facing reason for a failed connection test, as bot HTML (values escaped);
    the CLI prints it through ``control.plain``."""
    failure = failure or registry.ProbeFailure("empty")
    return t(
        FAILURE_KEYS.get(failure.kind, "llm_fail_request"),
        provider=html_escape(failure.provider),
        model=html_escape(failure.model),
        base_url=html_escape(failure.base_url),
        detail=html_escape(failure.detail),
        timeout=f"{failure.timeout_seconds:.0f}",
        cap=f"${failure.cap_usd:.2f}",
    )
