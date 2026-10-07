"""The ``curator`` command line (DESIGN §13).

Every command confirms itself in one line and fails with a sentence that says what to do. The
account session may live in one process only (§3), so a command that needs Telegram either
asks the running service over the control socket or, when no service runs, takes
``service.lock`` and builds its own runtime for the duration of the command. ``login`` and
``llm`` are interactive and need the service stopped: the bot offers the same steps while it
runs.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import click
import tomlkit
from tomlkit import TOMLDocument

from tg_curator import __version__, control, service
from tg_curator.config import SETTINGS_FILENAME, Settings, SettingsFile, ensure_home, resolve_home
from tg_curator.domain import KV
from tg_curator.errors import CuratorError, LoginError
from tg_curator.logging_setup import setup_logging

RUNNING = "the service is running; use /bind in the bot or stop the service"
"""§13: the lock is held and the socket does not answer (or the command needs it stopped)."""

LOGIN_ATTEMPTS = 3


def main() -> None:
    """The ``curator`` console script: private files first, then plain logs, then click."""
    os.umask(0o077)  # §3: sessions, the database's -wal/-shm and the socket are created 0600
    setup_logging()
    cli()


def _print_service_file(ctx: click.Context, _: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    unit = resources.files("tg_curator") / "data" / "tg-curator.service"
    click.echo(unit.read_text(encoding="utf-8"), nl=False)
    ctx.exit(0)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--home",
    type=click.Path(file_okay=False),
    default=None,
    help="Data directory (default: $TG_CURATOR_HOME or ~/.tg-curator).",
)
@click.version_option(__version__, prog_name="tg-curator")
@click.option(
    "--service-file",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_service_file,
    help="Print the systemd unit file and exit.",
)
@click.pass_context
def cli(ctx: click.Context, home: str | None) -> None:
    """tg-curator: sorts your Telegram channels into topic channels and a daily digest."""
    ctx.obj = resolve_home(home)


# --- plumbing --------------------------------------------------------------------------------


def _finish(code: int) -> None:
    click.get_current_context().exit(code)


def _settings(home: Path) -> SettingsFile:
    """The loaded settings file, or exit with the same sentences as ``curator run``."""
    ensure_home(home)
    settings_file = SettingsFile(home / SETTINGS_FILENAME)
    outcome = service.check_settings(settings_file)
    if not isinstance(outcome, Settings):
        code, message = outcome
        click.echo(message, err=True)
        _finish(code or 1)  # the template was just written: the command itself did not run
    return settings_file


def _progress(line: str) -> None:
    click.echo(line, err=True)


async def _echo(line: str) -> None:
    click.echo(line)


def _remote(home: Path, name: str, args: Mapping[str, Any]) -> int:
    try:
        return asyncio.run(
            control.request(control.socket_path(home), name, args, lambda line: click.echo(line))
        )
    except control.ServiceUnreachableError:
        click.echo(RUNNING, err=True)
        return 1


def _dispatch(home: Path, name: str, args: Mapping[str, Any]) -> None:
    """Run a socket command in the running service, else standalone under the lock."""
    ensure_home(home)
    if control.service_running(home):
        _finish(_remote(home, name, args))
    lock = control.ServiceLock(home)
    if not lock.acquire():  # the service started between the probe and now
        _finish(_remote(home, name, args))
    try:
        settings_file = _settings(home)
        code = asyncio.run(_guarded(_standalone(home, settings_file, name, args)))
    finally:
        lock.release()
    _finish(code)


async def _standalone(
    home: Path, settings_file: SettingsFile, name: str, args: Mapping[str, Any]
) -> int:
    async with service.open_runtime(
        home, settings_file, needs=control.NEEDS[name], progress=_progress
    ) as rt:
        return await control.run_command(rt, name, args, _echo)


def _stopped(home: Path, work: Callable[[SettingsFile], Awaitable[int]], hint: str) -> None:
    """Run ``work`` with the service stopped: it changes what a running service holds."""
    ensure_home(home)
    lock = control.ServiceLock(home)
    if not lock.acquire():
        click.echo(hint, err=True)
        _finish(1)
    try:
        settings_file = _settings(home)
        code = asyncio.run(_guarded(work(settings_file)))
    finally:
        lock.release()
    _finish(code)


async def _guarded(work: Awaitable[int]) -> int:
    try:
        return await work
    except CuratorError as exc:
        click.echo(str(exc), err=True)
        return 1


async def _ask(prompt: str, **options: Any) -> str:
    """``click.prompt`` off the event loop, so the Telegram connection stays alive while the
    user types."""
    return str(await asyncio.to_thread(click.prompt, prompt, **options))


# --- login -----------------------------------------------------------------------------------


@cli.command()
@click.option("--phone", help="The account's phone number in international format.")
@click.pass_obj
def login(home: Path, phone: str | None) -> None:
    """Sign in the Telegram account (once); the code is typed as Telegram sent it."""
    _stopped(home, lambda sf: _login(home, sf, phone), RUNNING)


async def _login(home: Path, settings_file: SettingsFile, phone: str | None) -> int:
    async with service.open_runtime(home, settings_file, needs=()) as rt:
        user, account = rt.user, rt.account
        assert user is not None and account is not None
        if await user.connect():
            me = await user.me()
            if me is not None:
                await rt.store.kv_set(KV.ACCOUNT_ID, me.id)
                if rt.settings.telegram.owner_id == 0:
                    await settings_file.set_value("telegram.owner_id", me.id)
            click.echo(f"already logged in as {_name(me)}")
            return 0
        phone = phone or await _ask("Phone number (international format, e.g. +998901234567)")
        await account.begin(phone.strip())
        result = await _with_retries(
            lambda: _ask("The code Telegram sent you"),
            lambda code: account.code(code.strip()),
            "bad_code",
        )
        if result is None:
            return 1
        if result == "password_needed":
            done = await _with_retries(
                lambda: _ask("Two-step verification password", hide_input=True),
                account.password,
                "bad_password",
            )
            if done is None:
                return 1
        click.echo(f"logged in as {_name(await user.me())}")
        return 0


async def _with_retries(
    ask: Callable[[], Awaitable[str]], submit: Callable[[str], Awaitable[Any]], retry_on: str
) -> Any:
    """Ask, submit; a wrong code or password may be typed again (``None`` = gave up). The
    answer is handed over as typed: a 2FA password may well end in a space."""
    for attempt in range(1, LOGIN_ATTEMPTS + 1):
        answer = await ask()
        try:
            result = await submit(answer)
        except LoginError as exc:
            click.echo(str(exc), err=True)
            if exc.reason != retry_on or attempt == LOGIN_ATTEMPTS:
                return None
            continue
        return "ok" if result is None else result
    return None


def _name(account: Any) -> str:
    if account is None:
        return "an unknown account"
    return f"{account.name} (@{account.username})" if account.username else account.name


# --- commands that may run in the service ----------------------------------------------------


@cli.command()
@click.option("--all", "include_left", is_flag=True, help="Include chats you left.")
@click.pass_obj
def chats(home: Path, include_left: bool) -> None:
    """List channels and groups with their identifiers, for the settings file."""
    _dispatch(home, "chats", {"all": include_left})


@cli.command()
@click.option("--days", default=3, show_default=True, type=click.IntRange(1, 30))
@click.option("--chat", "refs", multiple=True, help="Only this chat (id, @username or link).")
@click.pass_obj
def backfill(home: Path, days: int, refs: tuple[str, ...]) -> None:
    """Pull in the last few days of posts (safe to repeat)."""
    _dispatch(home, "backfill", {"days": days, "chats": list(refs)})


@cli.command()
@click.option("--topic", default=None, help="Show only the decisions for this topic key.")
@click.option("--days", default=3, show_default=True, type=click.IntRange(1, 30))
@click.option("--set", "assignments", multiple=True, metavar="KEY=VALUE", help="Try a setting.")
@click.pass_obj
def preview(home: Path, topic: str | None, days: int, assignments: tuple[str, ...]) -> None:
    """Every sorting decision on the backlog; nothing is posted."""
    overrides = dict(_assignment(a) for a in assignments)
    _dispatch(home, "preview", {"topic": topic, "days": days, "set": overrides})


def _assignment(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        raise click.BadParameter(f"{raw!r} is not KEY=VALUE, e.g. sorting.confidence=0.6")
    try:
        parsed: Any = tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value.strip()
    return key.strip(), parsed


@cli.command()
@click.option("--preview", "show", is_flag=True, help="Show tonight's digest without sending.")
@click.argument("topic", required=False)
@click.pass_obj
def digest(home: Path, show: bool, topic: str | None) -> None:
    """Send a digest now (all topics or one), or show tonight's with --preview."""
    _dispatch(home, "digest-preview" if show else "digest", {"topic": topic})


@cli.command()
@click.option("--days", default=None, type=click.IntRange(1, 365), help="Window in days.")
@click.option("--all", "include_left", is_flag=True, help="Include chats you left.")
@click.pass_obj
def stats(home: Path, days: int | None, include_left: bool) -> None:
    """Per-chat volume, signal, repeats and what was published."""
    _dispatch(home, "stats", {"days": days, "all": include_left})


@cli.command()
@click.pass_obj
def review(home: Path) -> None:
    """Build this week's proposals and send them to the bot chat now."""
    _dispatch(home, "review", {})


@cli.group(invoke_without_command=True)
@click.pass_context
def topics(ctx: click.Context) -> None:
    """Sync the topics from the settings file and list them."""
    if ctx.invoked_subcommand is None:
        _dispatch(ctx.obj, "topics", {})


@topics.command("categories")
def topics_categories() -> None:
    """The built-in categories a topic can start from."""
    from tg_curator.ml import categories

    for category in categories.all():
        click.echo(f"{category.key:<14} {category.label}")


@topics.command("add")
@click.argument("name")
@click.option("--category", default=None, help="A built-in category key (`topics categories`).")
@click.option("--description", default=None)
@click.option("--example-channel", default=None, help="A channel that is a good example.")
@click.option("--channel", default=None, help="An existing channel (id, @username or link).")
@click.option("--create-channel", is_flag=True, help="Create a private channel for the topic.")
@click.pass_obj
def topics_add(
    home: Path,
    name: str,
    category: str | None,
    description: str | None,
    example_channel: str | None,
    channel: str | None,
    create_channel: bool,
) -> None:
    """Add a topic (and its channel)."""
    if channel and create_channel:
        raise click.UsageError("use either --channel or --create-channel, not both")
    _dispatch(
        home,
        "topics-add",
        {
            "name": name,
            "category": category,
            "description": description,
            "example_channel": example_channel,
            "channel": channel,
            "create_channel": create_channel,
        },
    )


@topics.command("remove")
@click.argument("key")
@click.pass_obj
def topics_remove(home: Path, key: str) -> None:
    """Remove a topic; its channel, posts and statistics are kept."""
    _dispatch(home, "topics-remove", {"key": key})


# --- llm -------------------------------------------------------------------------------------


@cli.command()
@click.option("--mode", type=click.Choice(["none", "selfhosted", "provider"]), default=None)
@click.option("--provider", default=None, help="anthropic, openai, google, mistral, openrouter")
@click.option("--model", default=None)
@click.option("--base-url", default=None, help="Address of a self-hosted OpenAI-compatible server.")
@click.option("--key-stdin", is_flag=True, help="Read the API key or token from stdin.")
@click.pass_obj
def llm(
    home: Path,
    mode: str | None,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    key_stdin: bool,
) -> None:
    """Connect a language model for digest summaries (or none); tested before it is saved."""
    choice = {"mode": mode, "provider": provider, "model": model, "base_url": base_url}
    hint = "the service is running; use /llm in the bot or stop the service"
    _stopped(home, lambda sf: _llm(home, sf, choice, key_stdin), hint)


async def _llm(
    home: Path, settings_file: SettingsFile, choice: dict[str, str | None], key_stdin: bool
) -> int:
    from tg_curator.bot.llm import failure_reason, pick_test_post
    from tg_curator.llm import registry
    from tg_curator.llm.factory import make_llm

    async with service.open_runtime(home, settings_file, needs=()) as rt:
        mode = choice["mode"]
        if mode is None:
            # The same three choices as /llm, each with its line on cost and privacy (SPEC §setup).
            click.echo(control.plain(rt.t("llm_menu", current=_llm_current(rt))))
            mode = await _ask(
                "Language model",
                type=click.Choice(["none", "selfhosted", "provider"]),
                default="none",
            )
        if mode == "none":
            # Same [llm] contents as /llm → None: the provider and its key are blanked.
            cleared = {"mode": "none", "provider": "", "model": "", "base_url": "", "api_key": ""}
            await settings_file.update(lambda doc: _set_llm(doc, cleared))
            click.echo(registry.PRIVACY_NONE)
            click.echo("saved: no language model")
            return 0
        fields = await _llm_fields(mode, choice, key_stdin)
        candidate = _with_llm(rt.settings, fields)
        post = await pick_test_post(rt)
        if mode == "provider":
            label = next(p.label for p in registry.providers() if p.key == fields["provider"])
            privacy = rt.t("llm_privacy_provider", provider=label, test_post=post.label)
        else:
            privacy = rt.t(
                "llm_privacy_selfhosted", base_url=fields["base_url"], test_post=post.label
            )
        click.echo(control.plain(privacy))
        model = make_llm(candidate, rt.store, clock=rt.clock)
        try:
            outcome = await registry.probe(model, post.text, candidate.digest.line_chars)
        finally:
            close = getattr(model, "aclose", None)
            if close is not None:
                await close()
        line, elapsed = outcome.line, outcome.seconds
        if not line:
            # The cause named by the error itself (refused key, unknown model, timeout, ...).
            reason = control.plain(failure_reason(rt.t, outcome.failure))
            click.echo(
                f"the test failed after {elapsed:.1f} s, so nothing was saved. {reason}",
                err=True,
            )
            return 1
        click.echo(f"test summary ({elapsed:.1f} s): {line}")
        await settings_file.update(lambda doc: _set_llm(doc, fields))
        click.echo(f"saved: {fields['mode']} {fields['model']}")
        return 0


def _llm_current(rt: Any) -> str:
    """The current model in the words of the /llm menu, as plain text."""
    from tg_curator.llm import registry

    cfg = rt.settings.llm
    if cfg.mode == "selfhosted":
        return str(rt.t("llm_current_selfhosted", model=cfg.model, base_url=cfg.base_url))
    if cfg.mode == "provider":
        label = next((p.label for p in registry.providers() if p.key == cfg.provider), cfg.provider)
        cap = cfg.monthly_cap_usd
        cap_text = rt.t("llm_cap_value", cap=cap) if cap > 0 else rt.t("llm_cap_unset")
        return str(rt.t("llm_current_provider", provider=label, model=cfg.model, cap=cap_text))
    return str(rt.t("llm_current_none"))


async def _llm_fields(mode: str, choice: dict[str, str | None], key_stdin: bool) -> dict[str, Any]:
    from tg_curator.llm import registry

    if mode == "provider":
        infos = registry.providers()
        keys = [p.key for p in infos]
        provider = choice["provider"] or await _ask("Provider", type=click.Choice(keys))
        if provider not in keys:
            raise CuratorError(f"unknown provider {provider!r}; choose one of {', '.join(keys)}")
        models = next(p.models for p in infos if p.key == provider)
        model = choice["model"] or await _ask(f"Model ({', '.join(models)})", default=models[0])
        key = _stdin_secret() if key_stdin else await _ask("API key", hide_input=True)
        return {"mode": mode, "provider": provider, "model": model, "api_key": key, "base_url": ""}
    presets = registry.selfhosted_presets()
    base_url = choice["base_url"] or await _ask(
        "Server address", default=presets[0].base_url if presets else None
    )
    model = choice["model"] or await _ask("Model name")
    token = (
        _stdin_secret()
        if key_stdin
        else await _ask("Token (empty if none)", default="", hide_input=True, show_default=False)
    )
    return {"mode": mode, "provider": "", "model": model, "api_key": token, "base_url": base_url}


def _stdin_secret() -> str:
    return sys.stdin.readline().strip()


def _with_llm(settings: Settings, fields: Mapping[str, Any]) -> Settings:
    from tg_curator.config import validate_settings

    data = settings.model_dump()
    data["llm"] = {**data["llm"], **fields}
    return validate_settings(data)


def _set_llm(doc: TOMLDocument, fields: Mapping[str, Any]) -> None:
    if "llm" not in doc:
        doc["llm"] = tomlkit.table()
    table: Any = doc["llm"]
    for key, value in fields.items():
        table[key] = value


# --- run -------------------------------------------------------------------------------------


@cli.command()
@click.option("--live", is_flag=True, help="Set publishing.live = true first (= /go).")
@click.pass_obj
def run(home: Path, live: bool) -> None:
    """The service itself: intake, sorting, posting, digests, the bot."""
    _finish(asyncio.run(service.run(home, live=live)))
