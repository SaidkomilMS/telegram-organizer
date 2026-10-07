"""Settings: the one commented TOML file, validated into pydantic models (DESIGN §3, §4).

The file is parsed and rewritten with tomlkit so the user's comments survive bot edits; every
validation error becomes a sentence that names the key and says what to fix, because the
settings file is the main thing a non-programmer user touches.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import zoneinfo
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from shutil import copyfile
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from tomlkit import TOMLDocument
from tomlkit.exceptions import TOMLKitError

from tg_curator.errors import ConfigError

log = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "data" / "settings.example.toml"
SETTINGS_FILENAME = "settings.toml"
DEFAULT_HOME = Path("~/.tg-curator")
HOME_ENV = "TG_CURATOR_HOME"

TOPIC_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
FOLDER_NAME_MAX = 12
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Old dotted key -> new dotted key. On load the value is moved and a one-line notice logged,
# so settings files from older versions keep working (spec "Upgrades").
RENAMED_KEYS: dict[str, str] = {}

# Docker hands the three secrets and the database URL through the environment; when the
# file's value is blank they are written into the file so both install paths end the same.
ENV_OVERRIDES: dict[str, str] = {
    "TG_CURATOR_API_ID": "telegram.api_id",
    "TG_CURATOR_API_HASH": "telegram.api_hash",
    "TG_CURATOR_BOT_TOKEN": "telegram.bot_token",
    "TG_CURATOR_DATABASE_URL": "storage.database_url",
    "TG_CURATOR_TIMEZONE": "general.timezone",
}
# Template defaults an environment value may replace as if they were blank: the timezone
# ships as "UTC", which for a Docker user who set TG_CURATOR_TIMEZONE means "not chosen yet".
_ENV_REPLACES_DEFAULT: dict[str, Any] = {"general.timezone": "UTC"}

# --- models ----------------------------------------------------------------------------------

_MODEL_CONFIG = ConfigDict(extra="ignore", validate_assignment=True)

# Plain database URL schemes -> the async driver the store runs on.
_URL_DRIVERS = (
    ("postgresql://", "postgresql+asyncpg://"),
    ("postgres://", "postgresql+asyncpg://"),
    ("sqlite://", "sqlite+aiosqlite://"),
)

# Keys whose value is a secret: a validation error names the key but never echoes the value
# (DESIGN §1 "never log secrets"); a database URL is shown with its password masked.
SECRET_KEYS = frozenset(
    {"telegram.api_hash", "telegram.bot_token", "llm.api_key", "storage.database_url"}
)
_URL_PASSWORD_RE = re.compile(r"^([^:/?#]+://[^:/@?#]*:).*@")

# Sections that keep the values the process started with until a restart (§4, §14.7).
PINNED_SECTIONS = ("telegram", "storage")


class TelegramSettings(BaseModel):
    model_config = _MODEL_CONFIG
    api_id: int = 0
    api_hash: str = ""
    bot_token: str = ""
    owner_id: int = 0


class GeneralSettings(BaseModel):
    model_config = _MODEL_CONFIG
    timezone: str = "UTC"
    language: str = "en"

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
            raise ValueError(
                f'"{value}" is not an IANA timezone name; use one like "Europe/Berlin" '
                'or "Asia/Tashkent"'
            ) from None
        return value

    @field_validator("language")
    @classmethod
    def _valid_language(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z]{2,3}(?:[-_][A-Za-z]{2,4})?", value):
            raise ValueError('must be a language code such as "en" or "ru"')
        return value


class PublishingSettings(BaseModel):
    model_config = _MODEL_CONFIG
    live: bool = False
    style: Literal["repost", "forward"] = "repost"
    min_gap_seconds: int = Field(default=20, ge=1)
    staging_channel: int = 0


class SortingSettings(BaseModel):
    model_config = _MODEL_CONFIG
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    realtime_strength: float = Field(default=3.0, ge=0.0)
    duplicate_similarity: float = Field(default=0.90, ge=0.0, le=1.0)
    duplicate_similarity_cross: float = Field(default=0.86, ge=0.0, le=1.0)
    dedup_window_days: int = Field(default=3, ge=1)
    hold_minutes: int = Field(default=45, ge=0)
    neutral_trust: float = Field(default=1.0, ge=0.0)
    corroboration_weight: float = Field(default=1.0, ge=0.0)
    link_bonus: float = Field(default=0.25, ge=0.0)
    length_bonus: float = Field(default=0.25, ge=0.0)
    length_bonus_chars: int = Field(default=600, ge=1)
    second_opinion: bool = False


class GroupsSettings(BaseModel):
    model_config = _MODEL_CONFIG
    min_chars: int = Field(default=400, ge=0)
    unit_gap_minutes: int = Field(default=5, ge=1)


class DigestSettings(BaseModel):
    model_config = _MODEL_CONFIG
    hour: int = Field(default=21, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    items: int = Field(default=15, ge=1)
    window_hours: int = Field(default=26, ge=1)
    line_chars: int = Field(default=180, ge=20)


class ReviewSettings(BaseModel):
    model_config = _MODEL_CONFIG
    weekday: str = "sunday"
    hour: int = Field(default=11, ge=0, le=23)
    window_days: int = Field(default=30, ge=1)
    max_proposals: int = Field(default=20, ge=1)
    mute_days: int = Field(default=30, ge=1)
    leave_min_days: int = Field(default=30, ge=1)
    leaves_per_day: int = Field(default=3, ge=0)
    leave_interval_minutes: int = Field(default=30, ge=1)
    folder_min_days: int = Field(default=7, ge=1)
    folder_max_signal: float = Field(default=0.05, ge=0.0, le=1.0)
    folder_min_posts: int = Field(default=10, ge=0)
    mute_min_days: int = Field(default=14, ge=1)
    mute_max_signal: float = Field(default=0.03, ge=0.0, le=1.0)
    mute_min_duplicates: float = Field(default=0.6, ge=0.0, le=1.0)
    archive_min_days: int = Field(default=30, ge=1)
    archive_max_signal: float = Field(default=0.02, ge=0.0, le=1.0)
    cluster_min_posts: int = Field(default=30, ge=2)
    cluster_window_days: int = Field(default=7, ge=1)
    cluster_tightness: float = Field(default=0.60, ge=0.0, le=1.0)
    merge_margin: float = Field(default=0.30, ge=0.0, le=1.0)
    auto_create_topics: bool = False

    @field_validator("weekday")
    @classmethod
    def _valid_weekday(cls, value: str) -> str:
        lowered = value.strip().lower()
        if lowered not in WEEKDAYS:
            raise ValueError('must be a weekday name such as "sunday" or "monday"')
        return lowered

    @property
    def weekday_index(self) -> int:
        """0 = Monday … 6 = Sunday, matching ``date.weekday()``."""
        return WEEKDAYS.index(self.weekday)


class FoldersSettings(BaseModel):
    model_config = _MODEL_CONFIG
    curated: bool = True
    low_signal: bool = True
    curated_name: str = "Curated"
    low_signal_name: str = "Low signal"

    @field_validator("curated_name", "low_signal_name")
    @classmethod
    def _folder_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        if len(value) > FOLDER_NAME_MAX:
            raise ValueError(
                f"is {len(value)} characters; Telegram allows at most {FOLDER_NAME_MAX} "
                "for a folder title"
            )
        return value


class LlmSettings(BaseModel):
    model_config = _MODEL_CONFIG
    mode: Literal["none", "selfhosted", "provider"] = "none"
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    monthly_cap_usd: float = Field(default=0.0, ge=0.0)
    timeout_seconds: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def _mode_requirements(self) -> LlmSettings:
        if self.mode == "selfhosted" and not self.base_url:
            raise ValueError(
                "llm.base_url is empty: set the address of your OpenAI-compatible server "
                '(for example "http://localhost:11434/v1") or set llm.mode = "none"'
            )
        if self.mode == "provider":
            for key in ("provider", "model", "api_key"):
                if not getattr(self, key):
                    raise ValueError(
                        f"llm.{key} is empty: run /llm in the bot (or `curator llm`) to "
                        'connect a provider, or set llm.mode = "none"'
                    )
        return self


class StorageSettings(BaseModel):
    model_config = _MODEL_CONFIG
    database_url: str = ""
    keep_embeddings_days: int = Field(default=30, ge=1)
    keep_posts_days: int = Field(default=0, ge=0)

    @field_validator("database_url", mode="before")
    @classmethod
    def _async_driver(cls, value: Any) -> Any:
        """The usual ``postgresql://`` / ``sqlite://`` forms name the async driver the store
        runs on, so they are accepted instead of failing (with the password in the message)."""
        if isinstance(value, str):
            value = value.strip()
            for plain, driver in _URL_DRIVERS:
                if value.lower().startswith(plain):
                    return driver + value[len(plain) :]
        return value

    @field_validator("database_url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        if value and not re.match(r"^(sqlite\+aiosqlite|postgresql\+asyncpg)://", value):
            raise ValueError(
                "must be blank (SQLite in the home directory) or start with "
                '"postgresql+asyncpg://" or "sqlite+aiosqlite://"'
            )
        return value


class TopicSettings(BaseModel):
    model_config = _MODEL_CONFIG
    key: str
    name: str
    channel: int | str = 0
    category: str | None = None
    description: str | None = None
    example_channel: str = ""
    strictness: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("key")
    @classmethod
    def _valid_key(cls, value: str) -> str:
        if not TOPIC_KEY_RE.match(value):
            raise ValueError(
                "must be 1-32 characters of a-z, 0-9, '-' or '_' starting with a letter or "
                'digit, such as "ml-ai"'
            )
        return value

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("category", "description", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("channel", "example_channel", mode="before")
    @classmethod
    def _strip_ref(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class SourceSettings(BaseModel):
    model_config = _MODEL_CONFIG
    chat: int | str
    trust: int = Field(default=1, ge=0, le=3)

    @field_validator("chat", mode="before")
    @classmethod
    def _strip_ref(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("chat")
    @classmethod
    def _non_blank(cls, value: int | str) -> int | str:
        if isinstance(value, str) and not value:
            raise ValueError("must be a numeric chat id, an @username or a t.me link")
        return value


def _same_ref(a: int | str, b: int | str) -> bool:
    """Whether two chat references name the same chat as written (``@kunuz`` == ``kunuz``)."""
    if isinstance(a, int) or isinstance(b, int):
        return a == b
    return a.lstrip("@").lower().rstrip("/") == b.lstrip("@").lower().rstrip("/")


class MlSettings(BaseModel):
    """The local models (DESIGN §10): which ONNX file of the embedder to run and on how many
    threads. The int8 file is the default; the fp32 file is bit-identical on every CPU and is
    the switch for x86 machines without VNNI that miss duplicates."""

    model_config = _MODEL_CONFIG
    embedder_file: str = "onnx/model_qint8_avx512_vnni.onnx"
    threads: int = Field(default=2, ge=1, le=64)


class Settings(BaseModel):
    model_config = _MODEL_CONFIG
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    publishing: PublishingSettings = Field(default_factory=PublishingSettings)
    sorting: SortingSettings = Field(default_factory=SortingSettings)
    groups: GroupsSettings = Field(default_factory=GroupsSettings)
    digest: DigestSettings = Field(default_factory=DigestSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    folders: FoldersSettings = Field(default_factory=FoldersSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ml: MlSettings = Field(default_factory=MlSettings)
    topics: list[TopicSettings] = Field(default_factory=list)
    sources: list[SourceSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_entries(self) -> Settings:
        seen_keys: set[str] = set()
        for topic in self.topics:
            if topic.key in seen_keys:
                raise ValueError(
                    f'topics: key "{topic.key}" appears twice; every [[topics]] key must be unique'
                )
            seen_keys.add(topic.key)
        names = [t.name.casefold() for t in self.topics]
        for name in set(names):
            if names.count(name) > 1:
                raise ValueError(f'topics: two topics are named "{name}"; give each its own name')
        for i, src in enumerate(self.sources):
            for other in self.sources[:i]:
                if _same_ref(src.chat, other.chat):
                    raise ValueError(
                        f'sources: chat "{src.chat}" is listed twice; keep one [[sources]] '
                        "entry per chat"
                    )
        return self

    def topic(self, key: str) -> TopicSettings | None:
        """The ``[[topics]]`` entry with that key, or ``None``."""
        for topic in self.topics:
            if topic.key == key:
                return topic
        return None

    def source_for(self, ref: int | str) -> SourceSettings | None:
        """The ``[[sources]]`` entry whose ``chat`` names ``ref`` (id, @username or link)."""
        for src in self.sources:
            if _same_ref(src.chat, ref):
                return src
        return None

    def missing_telegram(self) -> list[str]:
        """Dotted names of the ``[telegram]`` values still blank (empty = ready to start)."""
        missing = []
        if not self.telegram.api_id:
            missing.append("telegram.api_id")
        if not self.telegram.api_hash:
            missing.append("telegram.api_hash")
        if not self.telegram.bot_token:
            missing.append("telegram.bot_token")
        return missing


# --- home directory --------------------------------------------------------------------------


def resolve_home(explicit: str | None, env: Mapping[str, str] | None = None) -> Path:
    """``--home`` > ``TG_CURATOR_HOME`` > ``~/.tg-curator`` (§3)."""
    environ = os.environ if env is None else env
    raw = explicit or environ.get(HOME_ENV) or str(DEFAULT_HOME)
    return Path(raw).expanduser().resolve()


def ensure_home(home: Path) -> Path:
    """Create ``home`` as 0700; a chmod failure (bind mounts) is a warning, never a stop."""
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        home.chmod(0o700)
    except OSError as exc:
        log.warning("could not set permissions 0700 on %s: %s", home, exc)
    return home


def write_template(path: Path) -> None:
    """Write the commented template to ``path`` (0600). Used on first run and by tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: an interrupted first start must not leave a truncated file behind.
    fd, tmp_name = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        copyfile(TEMPLATE_PATH, tmp)
        _chmod_secret(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _chmod_secret(path)


def _chmod_secret(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError as exc:
        log.warning("could not set permissions 0600 on %s: %s", path, exc)


# --- validation ------------------------------------------------------------------------------


def _dotted(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part + 1}]"
        else:
            out += ("." if out else "") + str(part)
    return out


def _sentence(err: Mapping[str, Any]) -> str:
    loc = tuple(p for p in err["loc"] if not (isinstance(p, str) and p.startswith("function-")))
    key = _dotted(loc)
    msg = err["msg"]
    if msg.startswith("Value error, "):
        msg = msg[len("Value error, ") :]
    if msg.startswith("Assertion failed, "):
        msg = msg[len("Assertion failed, ") :]
    if err["type"] == "missing":
        return f"{key} is missing: add it"
    if not key:
        return msg  # a model-level check: its message names its own subject ("topics: ...")
    if msg.startswith(key.split(".")[0] + ".") or msg.startswith(key.split(".")[0] + ":"):
        return msg  # the validator already named the key
    return f"{key}{_shown(key, err.get('input'))}: {msg}"


def _shown(key: str, value: Any) -> str:
    """`` = <value>`` for an error line; secrets are never echoed (§1)."""
    if isinstance(value, dict | list):
        return ""
    if key in SECRET_KEYS:
        if key == "storage.database_url" and isinstance(value, str):
            return f" = {mask_url(value)!r}"
        return ""
    return f" = {value!r}"


def mask_url(url: str) -> str:
    """``scheme://user:***@host/db``: a URL with its password hidden."""
    return _URL_PASSWORD_RE.sub(r"\1***@", url)


def validate_settings(data: Mapping[str, Any]) -> Settings:
    """Plain data -> ``Settings``; every pydantic error becomes a sentence of a ``ConfigError``."""
    try:
        return Settings.model_validate(dict(data))
    except ValidationError as exc:
        lines = [_sentence(e) for e in exc.errors()]
        raise ConfigError("settings: " + "; ".join(lines)) from None


def _warn_unknown_keys(data: Mapping[str, Any]) -> None:
    def walk(obj: Mapping[str, Any], model: type[BaseModel], prefix: str) -> None:
        for key, value in obj.items():
            if key not in model.model_fields:
                log.warning("settings: unknown key %s%s is ignored", prefix, key)
                continue
            ann = model.model_fields[key].annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel) and isinstance(value, Mapping):
                walk(value, ann, f"{prefix}{key}.")

    walk(data, Settings, "")
    for section, model in (("topics", TopicSettings), ("sources", SourceSettings)):
        for i, entry in enumerate(data.get(section) or []):
            if isinstance(entry, Mapping):
                walk(entry, model, f"{section}[{i + 1}].")


def _parse(text: str, path: Path) -> TOMLDocument:
    try:
        return tomlkit.parse(text)
    except TOMLKitError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}; fix the syntax") from None


def _get_dotted(doc: Mapping[str, Any], dotted: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _set_dotted(doc: TOMLDocument, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = doc
    for part in parts[:-1]:
        if part not in node:
            node[part] = tomlkit.table()
        node = node[part]
    if value is None:
        if parts[-1] in node:
            del node[parts[-1]]
    else:
        node[parts[-1]] = value


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == 0


def _apply_renames(doc: TOMLDocument) -> bool:
    changed = False
    for old, new in RENAMED_KEYS.items():
        value = _get_dotted(doc, old)
        if value is None:
            continue
        if _blank(_get_dotted(doc, new)):
            _set_dotted(doc, new, value)
        _set_dotted(doc, old, None)
        log.info("settings: key %s was renamed to %s; moved", old, new)
        changed = True
    return changed


def _apply_env(doc: TOMLDocument, env: Mapping[str, str]) -> bool:
    changed = False
    for var, dotted in ENV_OVERRIDES.items():
        raw = env.get(var, "").strip()
        current = _get_dotted(doc, dotted)
        replaceable = dotted in _ENV_REPLACES_DEFAULT and current == _ENV_REPLACES_DEFAULT[dotted]
        if not raw or not (_blank(current) or replaceable) or raw == current:
            continue
        value: int | str = raw
        if dotted == "telegram.api_id":
            if not raw.isdigit():
                raise ConfigError(f"{var} = {raw!r} is not a number; api_id is numeric")
            value = int(raw)
        _set_dotted(doc, dotted, value)
        log.info("settings: %s taken from %s and written into settings.toml", dotted, var)
        changed = True
    return changed


# --- the file --------------------------------------------------------------------------------

OnChange = Callable[[Settings], Awaitable[None] | None]


class SettingsFile:
    """The settings file on disk: load, validate, edit with comments kept, save atomically.

    ``load()`` and ``update()`` share one ``asyncio.Lock`` so a ``/reload`` and a bot edit never
    interleave; ``on_change`` is the hook the runtime connects to its event bus so components
    hear about every write (``settings_changed``) without this module importing the runtime.
    """

    def __init__(
        self,
        path: Path,
        *,
        env: Mapping[str, str] | None = None,
        on_change: OnChange | None = None,
    ) -> None:
        self.path = Path(path)
        self._env = env
        self.on_change = on_change
        self._lock = asyncio.Lock()
        self._doc: TOMLDocument | None = None
        self._current: Settings | None = None
        # After a /reload: the running [telegram]/[storage] models, kept in every Settings
        # this file produces until a restart (§4, §14.7). ``None`` until the first reload.
        self._pinned: dict[str, BaseModel] | None = None

    # -- reading --

    @property
    def current(self) -> Settings:
        if self._current is None:
            raise ConfigError(f"settings not loaded yet from {self.path}")
        return self._current

    @property
    def loaded(self) -> bool:
        return self._current is not None

    def exists(self) -> bool:
        return self.path.is_file()

    def load_sync(self) -> Settings:
        """Read, migrate, validate and swap ``.current`` without the lock (for sync starts)."""
        if not self.path.is_file():
            raise ConfigError(
                f"settings file not found: {self.path}; run `curator run` once to create it "
                "from the template"
            )
        doc = _parse(self.path.read_text(encoding="utf-8"), self.path)
        changed = _apply_renames(doc)
        changed = _apply_env(doc, os.environ if self._env is None else self._env) or changed
        data = doc.unwrap()
        _warn_unknown_keys(data)
        settings = validate_settings(data)
        if changed:
            self._write(doc)
        self._doc, self._current = doc, settings
        return settings

    def peek(self) -> Settings:
        """The file's settings as they are on disk now, read and validated like a load, but
        nothing is swapped or written: what the next start or ``/reload`` would apply (the
        preview reads hand edits through this)."""
        if not self.path.is_file():
            raise ConfigError(f"settings file not found: {self.path}")
        doc = _parse(self.path.read_text(encoding="utf-8"), self.path)
        _apply_renames(doc)
        _apply_env(doc, os.environ if self._env is None else self._env)
        return validate_settings(doc.unwrap())

    async def load(self) -> Settings:
        """Re-read the file. On a file already loaded this is ``/reload`` (§4): ``.current``
        keeps the running ``[telegram]`` and ``[storage]`` — the clients and the database were
        built from them — here and in every later ``update()`` until a restart. The returned
        ``Settings`` carries the file's own values, so the caller can tell that a restart is
        needed."""
        async with self._lock:
            running = self._current
            settings = self.load_sync()
            if running is not None:
                if self._pinned is None:
                    self._pinned = {name: getattr(running, name) for name in PINNED_SECTIONS}
                self._current = settings.model_copy(update=self._pinned)
            return settings

    def document(self) -> TOMLDocument:
        """A deep copy of the loaded document (comments included) for inspection."""
        if self._doc is None:
            raise ConfigError(f"settings not loaded yet from {self.path}")
        return tomlkit.parse(tomlkit.dumps(self._doc))

    # -- writing --

    async def update(self, mutator: Callable[[TOMLDocument], None]) -> Settings:
        """Apply ``mutator``, validate, write atomically, swap ``.current``.

        The running settings change by exactly what ``mutator`` does; the hand edits made to
        the file since the last load still wait for ``/reload`` or the next start (§4). The
        file itself is re-read first and the change is applied to what is on disk, so a bot
        or automatic write never throws a pending hand edit away. When that hand edit leaves
        the file invalid, nothing is written and the ``ConfigError`` says what to fix.
        """
        async with self._lock:
            doc = self.document()
            before = {name: _section(doc, name) for name in PINNED_SECTIONS}
            mutator(doc)
            settings = self._keep_pinned(validate_settings(doc.unwrap()), before, doc)
            try:
                on_disk = self._disk_document()
                if on_disk is not None:
                    mutator(on_disk)
                    validate_settings(on_disk.unwrap())
            except ConfigError as exc:
                raise ConfigError(
                    f"{self.path} was edited by hand and is not valid now, so this change was "
                    f"not saved: {exc}. Fix the file (then /reload) and try again"
                ) from None
            self._write(doc if on_disk is None else on_disk)
            self._doc, self._current = doc, settings
        await self._notify(settings)
        return settings

    async def set_value(self, dotted_key: str, value: Any) -> Settings:
        """``set_value("digest.hour", 20)``; ``None`` removes the key."""
        return await self.update(lambda doc: _set_dotted(doc, dotted_key, value))

    async def upsert_topic(self, key: str, **fields: Any) -> Settings:
        """Create or update the ``[[topics]]`` entry ``key``; a ``None`` field is removed."""

        def mutate(doc: TOMLDocument) -> None:
            table = _find_entry(doc, "topics", "key", key)
            if table is None:
                table = tomlkit.table()
                table["key"] = key
                _aot(doc, "topics").append(table)
            _assign(table, fields)

        return await self.update(mutate)

    async def remove_topic(self, key: str) -> Settings:
        return await self.update(lambda doc: _remove_entry(doc, "topics", "key", key))

    async def upsert_source(self, chat: int | str, trust: int) -> Settings:
        """Create or update the ``[[sources]]`` entry for ``chat`` with ``trust`` (0..3)."""

        def mutate(doc: TOMLDocument) -> None:
            table = _find_entry(doc, "sources", "chat", chat)
            if table is None:
                table = tomlkit.table()
                table["chat"] = chat
                _aot(doc, "sources").append(table)
            else:
                table["chat"] = chat
            table["trust"] = trust

        return await self.update(mutate)

    async def remove_source(self, chat: int | str) -> Settings:
        return await self.update(lambda doc: _remove_entry(doc, "sources", "chat", chat))

    # -- internals --

    def _disk_document(self) -> TOMLDocument | None:
        """The file as it is on disk now (renamed keys moved), or ``None`` when it is gone."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        doc = _parse(text, self.path)
        _apply_renames(doc)
        return doc

    def _keep_pinned(
        self, settings: Settings, before: Mapping[str, Mapping[str, Any]], doc: TOMLDocument
    ) -> Settings:
        """After a /reload the running ``[telegram]``/``[storage]`` stay in ``settings``; a key
        of theirs the update itself changed (the owner claim) takes its new value."""
        if self._pinned is None:
            return settings
        for name in PINNED_SECTIONS:
            pinned = self._pinned[name]
            old, new = before[name], _section(doc, name)
            changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
            fresh = getattr(settings, name)
            changed &= set(type(fresh).model_fields)
            if changed:
                pinned = pinned.model_copy(update={k: getattr(fresh, k) for k in changed})
                self._pinned[name] = pinned
            setattr(settings, name, pinned)
        return settings

    def _write(self, doc: TOMLDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".settings.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(tomlkit.dumps(doc))
                fh.flush()
                os.fsync(fh.fileno())
            _chmod_secret(Path(tmp))
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _chmod_secret(self.path)

    async def _notify(self, settings: Settings) -> None:
        if self.on_change is None:
            return
        result = self.on_change(settings)
        if result is not None:
            await result


def _section(doc: TOMLDocument, name: str) -> dict[str, Any]:
    value = doc.unwrap().get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _aot(doc: TOMLDocument, name: str) -> Any:
    if name not in doc:
        doc[name] = tomlkit.aot()
    return doc[name]


def _find_entry(doc: TOMLDocument, section: str, field: str, value: int | str) -> Any | None:
    for table in doc.get(section) or []:
        if field in table and _same_ref(table[field], value):
            return table
    return None


def _remove_entry(doc: TOMLDocument, section: str, field: str, value: int | str) -> None:
    aot = doc.get(section)
    if aot is None:
        return
    for i, table in enumerate(list(aot)):
        if field in table and _same_ref(table[field], value):
            del aot[i]
            return


def _assign(table: Any, fields: Mapping[str, Any]) -> None:
    for name, value in fields.items():
        if value is None:
            if name in table:
                del table[name]
        else:
            table[name] = value
