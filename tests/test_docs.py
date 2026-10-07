"""The documentation and packaging files agree with the code (DESIGN §13, SPEC "Documentation").

The docs are written for a stranger with a VPS who copies commands verbatim, so a command,
option, bot command, settings key or link that does not exist is a dead end for exactly the
reader they are for. These tests read the Markdown the way that reader does and check every
`curator …` command line against the click commands of ``cli.py``, every bot `/command`
against what the bot modules register, every settings key against ``config.Settings``, and
every relative link against the files and headings it points to. The packaging files
(Dockerfile, compose file, ``.env.example``, CI) are checked for the facts §13 fixes.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import shlex
from collections.abc import Iterator
from pathlib import Path

import click
import pydantic
import pytest

from tg_curator import config

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BOT_DIR = ROOT / "src" / "tg_curator" / "bot"
MARKDOWN = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.md",
    *sorted(DOCS.glob("*.md")),
]
USER_DOCS = ("install", "setup", "tune", "subscriptions", "safety", "faq")

FAQ_QUESTIONS = (
    "Can my account get banned?",
    "Does it send my messages anywhere?",
    "Why is a post unsorted?",
    "Why did a story not go out immediately?",
    "Which language model should I pick, and what does it cost?",
    "Can I run it for two accounts?",
    "Can I copy `user.session` to a second machine?",
    "What happens if I stop it for a week? For months?",
)
"""The questions DESIGN §13 says docs/faq.md answers, as its headings."""

NOT_BOT_COMMANDS = frozenset({"data", "newbot"})
"""Code spans that look like ``/word`` but are not ours: ``/data`` is the container's volume,
``/newbot`` is @BotFather's command."""

SHELL_OPERATORS = frozenset("();<>|&")
PLACEHOLDERS = frozenset({"…", "..."})
"""Prose such as "every `curator …` command" stands for any arguments."""


# --- reading Markdown ------------------------------------------------------------------------

_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_SPAN = re.compile(r"`([^`\n]+)`")


def code_snippets(text: str) -> Iterator[str]:
    """Every line of every fenced block, then every inline code span outside the blocks."""
    for block in _FENCE.findall(text):
        yield from block.replace("\\\n", " ").splitlines()
    yield from _SPAN.findall(_FENCE.sub("", text))


def all_snippets() -> Iterator[tuple[str, str]]:
    for path in MARKDOWN:
        for snippet in code_snippets(path.read_text(encoding="utf-8")):
            yield path.relative_to(ROOT).as_posix(), snippet


def shell_tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:  # an unbalanced quote in prose-like code; plain words are enough then
        return line.split()


def curator_invocations(line: str) -> Iterator[list[str]]:
    """The arguments of each ``curator`` command on a shell line, up to a pipe or redirect."""
    tokens = shell_tokens(line)
    for i, token in enumerate(tokens):
        if token != "curator":
            continue
        args: list[str] = []
        for arg in tokens[i + 1 :]:
            if set(arg) <= SHELL_OPERATORS:
                break
            if arg not in PLACEHOLDERS:
                args.append(arg)
        yield args


def heading_slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def headings(path: Path) -> list[str]:
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,6}) (.+)$", text, re.MULTILINE)]


# --- the code the docs describe --------------------------------------------------------------


def cli_root() -> click.Group:
    """The top click group of ``cli.py`` (the one no other group contains)."""
    if importlib.util.find_spec("tg_curator.cli") is None:
        pytest.skip("cli.py is not built yet")
    cli = importlib.import_module("tg_curator.cli")
    groups = [v for v in vars(cli).values() if isinstance(v, click.Group)]
    nested = {id(c) for g in groups for c in g.commands.values()}
    roots = [g for g in groups if id(g) not in nested]
    assert len(roots) == 1, f"expected one top-level click group in cli.py, found {roots}"
    return roots[0]


def find_option(command: click.Command, name: str) -> click.Option | None:
    for param in command.params:
        if isinstance(param, click.Option) and name in (*param.opts, *param.secondary_opts):
            return param
    return None


def check_invocation(root: click.Group, args: list[str]) -> str | None:
    """Walk ``args`` through the click tree; the first thing that does not exist, or None."""
    command: click.Command = root
    path = "curator"
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("-") and token != "-":
            name = token.split("=", 1)[0]
            if name != "--help":
                option = find_option(command, name)
                if option is None:
                    return f"`{path}` has no option {name}"
                if not option.is_flag and not option.count and "=" not in token:
                    i += 1  # the option's value
        elif isinstance(command, click.Group) and token in command.commands:
            command = command.commands[token]
            path = f"{path} {token}"
        elif not any(isinstance(p, click.Argument) for p in command.params):
            return f"`{path}` has no command or argument {token!r}"
        i += 1
    return None


def walk_commands(group: click.Group, prefix: str = "curator") -> Iterator[str]:
    for name, command in group.commands.items():
        if command.hidden:
            continue
        yield f"{prefix} {name}"
        if isinstance(command, click.Group):
            yield from walk_commands(command, f"{prefix} {name}")


def registered_bot_commands() -> dict[str, Path]:
    """``/name`` -> the bot module whose ``register`` (or the core) adds it."""
    if not (BOT_DIR / "setup.py").exists():
        pytest.skip("the bot feature modules are not built yet")
    found: dict[str, Path] = {}
    for path in sorted(BOT_DIR.glob("*.py")):
        for name in re.findall(r"\.command\(\s*[\"']([a-z_]+)[\"']", path.read_text("utf-8")):
            found[name] = path
    return found


def documented_bot_commands() -> list[tuple[str, str, str, str]]:
    """(file, span, command, first argument word or "") for every ``/command`` code span."""
    found = []
    for where, snippet in all_snippets():
        match = re.fullmatch(r"/([a-z_]+)(?:\s+(\S+).*)?", snippet.strip())
        if match is None or match.group(1) in NOT_BOT_COMMANDS:
            continue
        found.append((where, snippet, match.group(1), match.group(2) or ""))
    return found


def settings_fields() -> list[str]:
    """Every key of the settings file: section keys, then [[topics]] and [[sources]] keys."""
    names: list[str] = []
    for field in config.Settings.model_fields.values():
        model = field.annotation
        args = getattr(model, "__args__", ())
        if args:  # list[TopicSettings] / list[SourceSettings]
            model = args[0]
        if isinstance(model, type) and issubclass(model, pydantic.BaseModel):
            names.extend(model.model_fields)
    return names


# --- CLI -------------------------------------------------------------------------------------


def test_every_curator_command_in_the_docs_exists() -> None:
    root = cli_root()
    problems = []
    seen = 0
    for where, snippet in all_snippets():
        for args in curator_invocations(snippet):
            seen += 1
            problem = check_invocation(root, args)
            if problem is not None:
                problems.append(f"{where}: `{snippet.strip()}`: {problem}")
    assert seen > 20, "the docs should show the commands they describe"
    assert problems == []


def test_every_cli_command_is_documented() -> None:
    root = cli_root()
    documented = {
        " ".join(["curator", *(a for a in args if not a.startswith("-"))])
        for _, snippet in all_snippets()
        for args in curator_invocations(snippet)
    }
    missing = [
        name
        for name in walk_commands(root)
        if not any(d == name or d.startswith(f"{name} ") for d in documented)
    ]
    assert missing == []


def test_the_docs_show_the_global_options() -> None:
    root = cli_root()
    text = "\n".join(s for _, s in all_snippets())
    for option in ("--home", "--version", "--service-file"):
        assert find_option(root, option) is not None, f"cli.py lacks the global {option}"
        assert option in text, f"the docs never show {option}"


def test_the_invocation_checker_rejects_what_does_not_exist() -> None:
    @click.group()
    @click.option("--home")
    def root() -> None: ...

    @root.command()
    @click.option("--days", type=int)
    @click.option("--all", "show_all", is_flag=True)
    def stats(days: int | None, show_all: bool) -> None: ...

    @root.command()
    @click.argument("topic", required=False)
    def digest(topic: str | None) -> None: ...

    assert check_invocation(root, ["--home", "/x", "stats", "--days", "7", "--all"]) is None
    assert check_invocation(root, ["digest", "ml-ai"]) is None
    assert check_invocation(root, ["stat"]) == "`curator` has no command or argument 'stat'"
    assert (
        check_invocation(root, ["stats", "--weeks", "2"]) == "`curator stats` has no option --weeks"
    )
    assert list(curator_invocations("curator --service-file | sudo tee /etc/x")) == [
        ["--service-file"]
    ]
    assert list(curator_invocations("docker compose exec tg-curator curator stats")) == [["stats"]]
    assert list(curator_invocations("curator …")) == [[]]


# --- bot -------------------------------------------------------------------------------------


def test_every_bot_command_in_the_docs_is_registered() -> None:
    registered = registered_bot_commands()
    documented = documented_bot_commands()
    assert len(documented) > 20
    unknown = [
        f"{where}: `{span}`" for where, span, name, _ in documented if name not in registered
    ]
    assert unknown == []


def test_documented_bot_subcommands_are_handled_by_their_module() -> None:
    """``/topics add``, ``/digest preview``…: the word must be one the module looks for."""
    registered = registered_bot_commands()
    missing = []
    for where, span, name, word in documented_bot_commands():
        if not re.fullmatch(r"[a-z]+", word) or name not in registered:
            continue  # a placeholder such as CODE or a topic name, or reported above
        source = registered[name].read_text(encoding="utf-8")
        if not re.search(rf"[\"']{word}[\"']", source):
            missing.append(f"{where}: `{span}`: {registered[name].name} never reads {word!r}")
    assert missing == []


def test_every_registered_bot_command_is_documented() -> None:
    documented = {name for _, _, name, _ in documented_bot_commands()}
    assert sorted(set(registered_bot_commands()) - documented) == []


# --- settings, FAQ, links --------------------------------------------------------------------


def test_every_settings_key_is_documented() -> None:
    text = "".join((DOCS / f"{name}.md").read_text("utf-8") for name in ("tune", "subscriptions"))
    words = set(re.findall(r"[a-z][a-z0-9_]*", " ".join(code_snippets(text))))
    missing = [name for name in settings_fields() if name not in words]
    assert missing == []


def test_the_faq_answers_the_questions_of_the_contract() -> None:
    questions = headings(DOCS / "faq.md")
    for question in FAQ_QUESTIONS:
        assert question in questions
    text = (DOCS / "faq.md").read_text(encoding="utf-8")
    assert "AUTH_KEY_DUPLICATED" in text
    assert "/bind" in text


def test_every_user_doc_exists_and_is_linked_from_the_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in USER_DOCS:
        assert (DOCS / f"{name}.md").is_file()
        assert f"(docs/{name}.md)" in readme


def test_relative_links_point_at_existing_files_and_headings() -> None:
    broken = []
    for path in MARKDOWN:
        text = _FENCE.sub("", path.read_text(encoding="utf-8"))
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target):
                continue  # https:, mailto:
            file_part, _, anchor = target.partition("#")
            linked = (path.parent / file_part).resolve() if file_part else path
            if not linked.exists():
                broken.append(f"{path.name}: {target}")
            elif anchor and anchor not in {heading_slug(h) for h in headings(linked)}:
                broken.append(f"{path.name}: {target} (no such heading)")
    assert broken == []


def test_the_safety_page_states_the_outbound_connections_and_credits() -> None:
    text = (DOCS / "safety.md").read_text(encoding="utf-8")
    for fact in (
        "huggingface.co",
        "about 150 MB",
        "user.session",
        "AUTH_KEY_DUPLICATED",
        "0600",
        "MN-DS",
        "uz-news",
        "lenta.ru",
        "CC BY 4.0",
        "multilingual-e5-small",
    ):
        assert fact in text, fact


# --- packaging -------------------------------------------------------------------------------


def dockerfile_instructions() -> list[tuple[str, str]]:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    lines = [line.strip() for line in text.splitlines()]
    return [
        (line.split(None, 1)[0].upper(), line.split(None, 1)[1] if " " in line else "")
        for line in lines
        if line and not line.startswith("#")
    ]


def test_the_dockerfile_follows_the_contract() -> None:
    instructions = dockerfile_instructions()
    by_kind: dict[str, list[str]] = {}
    for kind, value in instructions:
        by_kind.setdefault(kind, []).append(value)
    assert instructions[0] == ("FROM", "python:3.12-slim")
    env = dict(
        pair.split("=", 1) for value in by_kind["ENV"] for pair in value.split() if "=" in pair
    )
    assert (
        env
        | {
            "HOME": "/data",
            "TG_CURATOR_HOME": "/data",
            "TG_CURATOR_WAIT_FOR_SETTINGS": "1",
            "HF_HOME": "/data/models",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        == env
    )
    runs = " ".join(by_kind["RUN"])
    assert "--uid 1000" in runs and "--gid 1000" in runs
    assert 'pip install "/build[postgres]"' in runs  # TG_CURATOR_DATABASE_URL works in Docker
    assert by_kind["USER"] == ["curator"]
    assert json.loads(by_kind["VOLUME"][0]) == ["/data"]
    assert json.loads(by_kind["CMD"][-1]) == ["curator", "run"]
    copied = " ".join(by_kind["COPY"])
    for needed in ("pyproject.toml", "README.md", "LICENSE", "src /build/src"):
        assert needed in copied  # hatchling reads the readme and licence at build time


def test_the_compose_file_runs_one_service_on_a_named_volume() -> None:
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    (service,) = compose["services"].values()
    assert service["env_file"] == ".env"
    assert service["restart"] == "unless-stopped"
    assert service["build"] == "."
    (mount,) = service["volumes"]
    volume, target = mount.split(":")
    assert target == "/data" and volume in compose["volumes"]


def test_the_env_example_holds_exactly_the_three_values() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    names = [line.split("=", 1)[0] for line in lines if line and not line.startswith("#")]
    assert names == ["TG_CURATOR_API_ID", "TG_CURATOR_API_HASH", "TG_CURATOR_BOT_TOKEN"]
    assert set(names) <= set(config.ENV_OVERRIDES)


def test_ci_lints_and_tests_on_python_312_without_models() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    (job,) = workflow["jobs"].values()
    runs = " ".join(step.get("run", "") for step in job["steps"])
    assert "ruff check" in runs and "ruff format --check" in runs and "pytest" in runs
    assert any(step.get("with", {}).get("python-version") == "3.12" for step in job["steps"])
    env = {**workflow.get("env", {}), **job.get("env", {})}
    for step in job["steps"]:
        env.update(step.get("env", {}))
    assert "TG_CURATOR_TEST_MODELS" not in env  # no model download in CI


def test_the_issue_template_asks_for_text_not_screenshots() -> None:
    text = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.md").read_text(encoding="utf-8")
    assert "`curator preview`" in text and "`curator --version`" in text
    assert "not screenshots" in text
