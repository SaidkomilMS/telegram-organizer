"""The code-level contract between modules (DESIGN §8): one Protocol per cross-module service.

Every service takes the ``Runtime`` in its constructor (``Service(rt)``) and is reached through
``rt.<name>``; the rules in the docstrings below are copied from the design so that an
implementer and a caller read the same sentence. Nothing here is executed.

Module functions (non-Protocol contracts) — plain functions on module boundaries whose
signatures are part of the contract; they live in their owners' modules:

``textutil.py``
    ``normalise(text: str) -> str`` — casefold, collapse whitespace, strip urls/emoji/punctuation
    for hashing.
    ``text_hash(text: str) -> str`` — sha1 of ``normalise(text)``; ``""`` when the normalised
    text is empty.
    ``url_key(url: str) -> str | None`` — canonical outside link: scheme/www/tracking
    params/fragment removed; ``None`` for t.me links and bare domains.
    ``canonical_urls(urls: Sequence[str]) -> list[str]`` — ``url_key`` over a list,
    deduplicated, order kept.
    ``first_line(text: str, max_chars: int) -> str``
    ``html_escape(s: str) -> str``
    ``split_html(html: str, limit: int) -> list[str]`` — fewest parts, never inside a tag or
    entity, never inside a digest item (items are separated by ``"\\n\\n"``); limit measured in
    UTF-16 code units of the plain text after HTML parsing.
``pipeline/render.py``
    ``realtime_post(post: Post, chat: Chat, corroboration: int, *, caption: bool = False)
    -> RenderedPost`` — parts already split for 1024 (``caption=True``) or 4096.
    ``stub(kind: Literal["moved", "not_for_me"], topic_name: str | None) -> str``
    ``digest_message(day: date, topic_name: str, items: list[DigestLine],
    chats: dict[int, Chat], *, seq: int = 0) -> list[str]`` — parts with the unique per-part
    header of §9.5: seq 0 renders "Daily digest", seq >= 1 renders "Digest (manual {seq})".
``pipeline/preview.py``
    ``render_text(report: PreviewReport) -> str`` — one line per decision (§9.6); used by
    ``cli.py preview`` and by the issue template.
``llm/factory.py``, ``llm/registry.py``
    ``make_llm(settings: Settings, store: Store) -> LLM`` — a disabled LLM for ``mode = "none"``.
    ``providers() -> list[ProviderInfo]`` — key, label, models (short sensible list),
    privacy_line.
    ``test_prompt(llm: LLM, text: str, max_chars: int) -> tuple[str | None, float]`` —
    (line, elapsed seconds).
``ml/categories.py``, ``ml/classifier.py``, ``ml/models.py``
    ``all() -> list[Category]`` — built-in categories, stable order.
    ``LocalTopicClassifier(embedder: Embedder, base_model: BaseModel)`` — constructed by
    ``service.py`` / ``cli.py``.
    ``ensure_models(home: Path, progress: Callable[[str], None] | None) -> None``
``pipeline/engine.py``
    ``RecentIndex(window_days: int)`` with
    ``.add(post_id, chat_id, message_id, root_id, text_hash, url_key, fwd_key, embedding)``,
    ``.remove_older_than(ts: datetime)``, ``.match_forward(fwd_key)``,
    ``.match_exact(text_hash)``, ``.match_url(url_keys, embedding)``,
    ``.match_semantic(embedding)``.
    ``DecisionEngine(settings: Settings, classifier: TopicClassifier, index: RecentIndex,
    llm: LLM | None)`` with ``.decide(candidate, embedding, chat, topics, now) -> Decision``
    and ``.strength(post_or_candidate, chat, corroboration: int) -> tuple[float, bool]``
    ((strength, would_realtime), pure).
``clock.py``
    ``scheduled_moment(day: date, hour: int, minute: int, tz: str) -> datetime`` — aware UTC,
    §9.5; used by ``digest.py`` and ``review.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from tg_curator.domain import (
    AccountStatus,
    BackfillResult,
    Candidate,
    Chat,
    ChatStats,
    ChatSyncResult,
    CorrectionResult,
    DigestDraft,
    DigestResult,
    Example,
    MoveResult,
    Post,
    PreviewReport,
    Proposal,
    ProposalKind,
    Topic,
    TopicName,
    TopicOrigin,
    TopicScore,
    TopicSyncResult,
)
from tg_curator.telegram.gateway import Buttons, ChatInfo, IncomingMessage

if TYPE_CHECKING:
    import numpy as np

    from tg_curator.notify import AbsorbedLine
    from tg_curator.subscriptions.stats import TopicStats

ProgressCallback = Callable[[int, int, Chat], Awaitable[None]]
"""``Backfill.run(progress=...)``: ``async callable(done, total, chat)``, called per chat."""


# --- models and messaging --------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """The multilingual similarity model; ``id`` changes whenever stored embeddings must be
    recomputed (``kv ml.embedder_id``, §10)."""

    id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """``(n, dim)`` float32, unit-norm. Blocking: the sorter runs it in a worker thread."""
        ...


@runtime_checkable
class TopicClassifier(Protocol):
    """Base category scores + the per-user layer trained from ``examples`` (§10).

    ``reload()``/``learn()`` build the new user layer fully and then swap it in with one
    attribute assignment; ``predict()`` reads the layer once at entry, so a retrain running
    while the sorter's worker thread predicts never sees a half-swapped matrix.
    """

    def reload(self, topics: Sequence[Topic], examples: Sequence[Example]) -> None:
        """Rebuild the user layer for exactly these (active) topics and examples."""
        ...

    def predict(self, text: str, embedding: np.ndarray) -> list[TopicScore]:
        """Best first, one score in [0, 1] for every active topic."""
        ...

    def learn(self, example: Example) -> None:
        """Add one example; effective immediately (a near-identical post flips at once)."""
        ...

    def category_scores(self, text: str, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Built-in categories ``(key, score)``, best first."""
        ...


@runtime_checkable
class LLM(Protocol):
    """The connected language model (none, self-hosted or provider); ``enabled`` is false for
    ``llm.mode = "none"`` and callers then never ask."""

    enabled: bool

    async def summarise_line(self, text: str, max_chars: int) -> str | None:
        """One sentence in the language of the input text, <= ``max_chars``.

        ``None`` on any failure, timeout, or when ``budget.exhausted(month)`` — callers fall
        back to the first line. ``llm/budget.py`` records every request in ``llm_usage`` and,
        the first time the cap is crossed in a month, calls ``notifier.owner(...)`` once and
        sets ``cap_notified``.
        """
        ...

    async def name_topic(
        self,
        examples: Sequence[str],
        *,
        existing: Sequence[str] = (),
        category_label: str | None = None,
    ) -> TopicName | None:
        """A sharper name and a short description written from the example posts.

        ``existing``: the names of the topics that already exist (the new name must differ);
        ``category_label``: the classifier's closest built-in category, as a hint.
        """
        ...

    async def second_opinion(
        self, text: str, topic: Topic, *, competing: Sequence[Topic] = ()
    ) -> bool | None:
        """Does the post belong to ``topic``? ``None`` on failure (treated as no opinion).

        ``competing``: the runner-up topics the classifier also scored for the post.
        """
        ...


@runtime_checkable
class Notifier(Protocol):
    """Owner notifications (§11.4). ``owner`` is the one primitive; the named helpers are the
    only notifications that exist, so no module can invent a new kind (``review_absorbed`` is
    part of item 2 and ``llm_out_of_credit`` of item 5, §17.5/§17.6)."""

    async def owner(self, html: str, *, buttons: Buttons | None = None) -> int | None:
        """Send to the owner's private chat; serialised so the chat gets at most one message
        per second overall. Returns the message id, or ``None`` when there is no owner or no
        bot yet (a no-op, logged at DEBUG)."""
        ...

    async def digest_line(self, summary: str) -> int | None:
        """(1) the digest line, one per run (``summary`` like ``ML & AI: 15 posts; Fintech: 9``,
        prefixed ``(manual)`` by the caller for a manual send)."""
        ...

    async def proposal(
        self,
        kind: ProposalKind,
        title: str,
        reason: str,
        *,
        details: str | None = None,
        buttons: Buttons | None = None,
    ) -> int | None:
        """(2) one message per proposal on review day, delivered with the review."""
        ...

    async def intake_warning(self, reason: Literal["stalled", "session_lost"]) -> int | None:
        """(3) intake has stopped for over an hour, or the session is lost (§11.3)."""
        ...

    async def cannot_post(self, channel_title: str) -> int | None:
        """(4) the bot cannot post into a topic channel — once per channel until it recovers
        (``BotCannotPost``/``ChatGone`` from the publisher or the digest)."""
        ...

    async def llm_cap_reached(self, month: str, cap: float) -> int | None:
        """(5) the once-a-month "LLM cap reached" line (from ``llm/budget.py``)."""
        ...

    async def llm_out_of_credit(self, provider: str) -> int | None:
        """(5) the provider refused for lack of credit; first lines until the month ends. At
        most once a month together with the cap line (shares ``llm_usage.cap_notified``)."""
        ...

    async def review_absorbed(self, lines: Sequence[AbsorbedLine]) -> int | None:
        """(2) the opening message of a review: one line per topic accepted from a
        ``new_topic`` proposal since the last review, "<topic> absorbed N of M unsorted posts
        since <date>" (§17.6)."""
        ...

    async def topic_created(
        self, name: str, *, has_channel: bool = True, wait_minutes: int | None = None
    ) -> int | None:
        """(6) in auto mode only, "created topic X" (from ``Discovery``). With
        ``has_channel=False`` the message says the private channel is not created yet
        (channel pacing, §8; ``wait_minutes`` when Telegram's wait is known)."""
        ...


# --- pipeline --------------------------------------------------------------------------------


@runtime_checkable
class Intake(Protocol):
    """Telegram messages -> candidates: albums, group conversation units, counting (§9.1)."""

    async def handle_message(self, msg: IncomingMessage) -> None:
        """= ``collect()`` for one message (``via="live"``) + ``sorter.submit`` of what it
        yields."""
        ...

    async def collect(
        self, messages: AsyncIterator[IncomingMessage], *, via: Literal["live", "backfill"]
    ) -> list[Candidate]:
        """Run album grouping and group-unit building over a finite stream WITHOUT
        submitting; stores ``group_messages`` rows (and bumps ``chat_daily`` for the new
        ones) but never inserts posts — the sorter does that on submit (§9.3); used by
        ``Backfill``. Units are closed by ``close_units()``."""
        ...

    async def close_units(
        self, chat_id: int | None, now: datetime, *, force: bool = False
    ) -> list[Candidate]:
        """Close conversation units per §9.1 and return the candidates that pass the floor;
        ``force=True`` closes every open unit (backfill, after each chat)."""
        ...

    async def tick(self) -> None:
        """Flush albums, ``close_units(None, clock.now())`` and submit; purge
        ``group_messages`` older than 3 days."""
        ...

    async def flush_albums(self) -> int:
        """Submit every buffered live album now, settled or not (a clean stop calls it);
        returns how many. What a crash leaves is in ``kv intake.open_albums``."""
        ...

    async def sync_chats(self) -> ChatSyncResult:
        """Dialogs -> ``chats`` (new, left, changed)."""
        ...


@runtime_checkable
class Backfill(Protocol):
    """Pull the last few days of posts, one chat at a time, for previewing (§9.1)."""

    async def run(
        self,
        days: int = 3,
        *,
        chat_ids: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> BackfillResult:
        """``progress``: ``async callable(done: int, total: int, chat: Chat) -> None``, called
        per chat. Candidates are submitted afterwards in global ``posted_at`` order; running
        it twice is harmless; backfilled posts never go out in real time."""
        ...

    @property
    def running(self) -> bool:
        """A run is under way; another ``run`` waits for it (chats are read one at a time)."""
        ...


@runtime_checkable
class Sorter(Protocol):
    """The live sorter: persists decisions, routes, holds and promotes (§9.3)."""

    async def submit(self, c: Candidate) -> Post | None:
        """Full decision, persisted, routed. ``None`` when the candidate already existed."""
        ...

    async def tick(self) -> None:
        """Hold expiry + self-healing, §9.3."""
        ...

    async def resort_unsorted(self, since: datetime) -> int:
        """Re-run questions 2–4 on ``unsorted`` posts (never ``rejected``, never ``corrected``)
        with ``posted_at >= since``, with the current classifier; results are persisted like
        backfill posts (§9.2 last paragraph): ``digest`` when inside the digest window, else
        ``dropped``; ``tracked`` for topics without a channel; never ``held``/``queued``,
        nothing old is published. Returns the number re-sorted."""
        ...


@runtime_checkable
class PreviewService(Protocol):
    """Replay of the backlog with the current (or overridden) settings; nothing written (§9.6)."""

    async def replay(
        self,
        *,
        days: int = 3,
        topic_key: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> PreviewReport:
        """``overrides``: dotted settings keys (``"sorting.confidence"``,
        ``"topics.<key>.strictness"``) applied on a copy of ``rt.settings`` for this replay
        only; nothing is written. Second opinions never cost a request (``llm=None``)."""
        ...


@runtime_checkable
class Publisher(Protocol):
    """The outbox: repost/forward styles, pacing, +N edits, moves (§9.4)."""

    async def enqueue(self, post_id: int) -> bool:
        """Insert the ``publications`` row (``pending``) and set the post ``queued`` in one
        transaction. Idempotent: an existing row in any state -> ``False`` (a ``cancelled``
        row means the post left the real-time path for good)."""
        ...

    async def tick(self) -> None:
        """Send what is due, paced; +N edits. Does nothing unless ``publishing.live`` and not
        paused. A post counts as published only after Telegram returned the message id."""
        ...

    async def reconcile(self) -> None:
        """After a restart: rows stuck in ``sending`` are matched against the channel with
        ``user.find_message`` (found -> ``sent``, else back to ``pending``); then queued posts
        older than ``hold_minutes`` are cancelled into the digest instead of posted late."""
        ...

    async def note_corroboration(self, post_id: int) -> None:
        """Mark the post dirty; ``tick`` edits the +N (no-op if nothing to do)."""
        ...

    async def move(self, post_id: int, new_topic_id: int | None) -> MoveResult:
        """Called ONLY by ``Learning.correct`` (§9.8); stubs/re-publishes per §9.4."""
        ...


@runtime_checkable
class DigestService(Protocol):
    """Ranking, composition, sending and schedule of the daily digest (§9.5)."""

    async def preview(self, topic_key: str | None = None) -> list[DigestDraft]:
        """Compose steps 1–4 without sending or changing any status."""
        ...

    async def send(self, topic_key: str | None = None) -> list[DigestResult]:
        """Manual digest: refused with one sentence when publishing is not live or paused;
        otherwise always inserts a new row with ``seq = COALESCE(MAX(seq), 0) + 1`` and
        ``manual=1`` over the current ``status=digest`` posts; never affects whether the
        scheduled digest runs."""
        ...

    async def tick(self) -> None:
        """Scheduled + missed digests (``due = today`` if past the hour, else ``yesterday``),
        owner line for the topics run in this tick. Not while paused or not live."""
        ...

    async def reconcile(self) -> None:
        """After a restart: rows stuck in ``sending`` — each part not yet in ``message_ids``
        is looked for by its unique header with ``user.find_message`` and sent only if absent
        (§9.5)."""
        ...

    def next_run(self) -> datetime:
        """The next scheduled moment as aware UTC (for ``/status``)."""
        ...


# --- topics and learning ---------------------------------------------------------------------


@runtime_checkable
class TopicsService(Protocol):
    """Create/edit/merge/remove topics, examples, sync with the settings file (§8, §9.7)."""

    async def sync_from_settings(self) -> TopicSyncResult:
        """Mirror ``[[topics]]`` into ``topics`` (new keys -> rows; keys gone from the file ->
        §9.7 deactivation), resolve every topic with ``channel != 0`` through
        ``link_channel()`` (failures are listed in ``TopicSyncResult.unresolved``, never
        raised), and mirror ``[[sources]]``: each ``chat`` is resolved with
        ``user.resolve_chat`` (unresolvable -> unresolved line, skipped), ``chats.trust`` is
        set for listed chats and reset to NULL for chats no longer listed. Called at start
        (§13), on every ``settings_changed`` event and on ``/reload``. Rewrites resolved
        @username/link values to numeric ids. A hand edit that sets ``channel = 0`` on a
        topic that has one is an unresolved line and the stored channel is kept."""
        ...

    async def link_channel(self, key: str, ref: str | int) -> Topic:
        """Used by ``create(channel=...)``, ``update(channel=...)`` and
        ``sync_from_settings``: ``user.resolve_chat(ref)`` (``ChatGone`` -> ``ConfigError``
        "topic channel X cannot be found: check the link or that your account is in it");
        require ``kind == "channel"`` and ``is_creator`` or ``is_admin`` (else ``ConfigError``
        "channel X is not yours: the account must be its creator or an admin");
        ``store.upsert_chat(info, role="output")``; ``user.register_owned(id)``; if not
        ``await bot.can_post(id)``: try ``user.add_bot_admin(id, rt.bot_account.username)``
        and re-check; still false -> ``BotCannotPost`` rendered as "the bot cannot post into
        X: add @<bot> as an admin with Post Messages". Writes ``topics.channel_id`` and the
        settings value (numeric id)."""
        ...

    async def create(
        self,
        name: str,
        *,
        category: str | None = None,
        description: str | None = None,
        channel: str | int | None = None,
        create_channel: bool = False,
        example_channel: str | None = None,
        origin: TopicOrigin = "user",
    ) -> Topic:
        """``key`` = ASCII slug of the name (transliterated, lowercase, "-"), suffixed -2, -3
        on collision; a name that already exists (case-insensitive) raises ``TopicExists`` so
        the caller asks "edit the existing one?" instead of creating a duplicate.

        ``create_channel=True``: channel creation is serialised in ``TopicsService`` — at most
        one ``create_channel`` per 60 s and at most 5 per calendar day (``kv
        topics.create_log``). On ``FloodWait`` (or the daily cap) the topic is saved with
        ``channel = 0``, marked with ``want_channel`` (``tick`` creates and links it once the
        wait is over and tells the owner) and ``channel_wait_minutes()`` tells the caller how
        long. ``category`` must be a built-in key (``ConfigError`` naming close matches). A
        channel already used by another topic, or the staging channel, is refused with a
        ``ConfigError``. A created channel goes through ``link_channel()``
        (admin, owned, can_post). Afterwards ``classifier.reload``,
        ``sorter.resort_unsorted`` and ``examples_changed(reason="topics")``.
        """
        ...

    def channel_wait_minutes(self) -> int | None:
        """Minutes until ``create_channel`` is allowed again; ``None`` = now."""
        ...

    async def create_topic_channel(self, name: str) -> ChatInfo | None:
        """A private channel named ``name`` under the same pacing as ``create(create_channel=
        True)`` (one per 60 s, five per day, ``kv topics.create_log``), adopted as an owned
        output channel; ``None`` when pacing or a ``FloodWait`` says wait
        (``channel_wait_minutes``). The caller links it with ``link_channel`` — the /topics
        [Create channel] button for a topic that already exists (§11.2)."""
        ...

    async def want_channel(self, key: str) -> None:
        """Mark ``key`` as still wanting a created channel (deferred by pacing/FloodWait)."""
        ...

    async def tick(self) -> None:
        """The ``topic_channels`` loop: create one deferred channel when pacing allows, link
        it, and tell the owner; a new FloodWait keeps the mark."""
        ...

    async def update(self, key: str, **changes: Any) -> Topic:
        """Edit name/category/description/channel/example_channel/strictness. ``channel=0``
        raises ``ConfigError`` ("a channel cannot be removed from a topic: remove the topic,
        or /pause to stop posting"); ``description=...`` emits
        ``examples_changed(reason="examples")``."""
        ...

    async def remove(self, key: str) -> None:
        """§9.7 deactivation + remove from settings; the channel is never touched."""
        ...

    async def merge(self, src_key: str, dst_key: str) -> Topic:
        """§9.7 with ``dst``: posts, open publications and examples of ``src`` are
        re-pointed to ``dst``; ``src`` is deactivated."""
        ...

    async def add_examples(self, key: str, texts: Sequence[str]) -> int:
        """Store example posts (``kind='example'``); emits ``examples_changed(reason=
        "examples")``. Returns how many were added."""
        ...

    async def add_example_channel(self, key: str, ref: str) -> int:
        """Read up to 50 recent posts via ``user.history()``; public channels work without
        membership; a private channel the account is not in yields
        ``NotAllowed("not_a_member")``. Returns how many examples were stored."""
        ...

    async def ensure_staging_channel(self) -> int:
        """If ``publishing.staging_channel`` is 0 or ``resolve_chat`` fails (``ChatGone`` ->
        recreate): ``user.create_channel("tg-curator media", about="private staging channel
        used by tg-curator to hand media to the bot; safe to ignore")``,
        ``user.add_bot_admin(id, rt.bot_account.username)`` (post_messages + edit_messages
        only), ``register_owned(id)``, upsert chat with ``role="staging"``, write
        ``publishing.staging_channel``. Returns the id. Callers: ``/go`` (before setting
        ``live=true``), ``service.py`` at start and on ``account_bound`` /
        ``settings_changed`` when live and ``style == "repost"``. The publisher never creates
        it."""
        ...


@runtime_checkable
class Learning(Protocol):
    """Corrections and retraining of the per-user layer (§9.8)."""

    async def correct(self, post_id: int, new_topic_id: int | None) -> CorrectionResult:
        """Does everything, exact steps in §9.8; ``new_topic_id`` ``None`` = "not for me".
        One correction row per post (a new one replaces the previous), ``classifier.learn``
        at once, the status rule of §9.8, ``publisher.move`` when a publications row exists,
        then ``examples_changed(reason="correction")``. Refuses ``ignored``/``duplicate``
        posts with ``moved=False``."""
        ...

    async def retrain(self) -> None:
        """Rebuild the user layer from the DB (also un-learns replaced correction rows)."""
        ...


# --- subscriptions ---------------------------------------------------------------------------


@runtime_checkable
class StatsService(Protocol):
    """Per-chat volume, signal, duplicate share and published counts (§12)."""

    async def chat_stats(
        self, days: int | None = None, *, include_left: bool = False
    ) -> list[ChatStats]:
        """``days`` ``None`` = ``[review].window_days``. Left chats only with
        ``include_left``; staging and output chats never."""
        ...

    async def topic_stats(self, days: int | None = None) -> list[TopicStats]:
        """Per active topic: original posts sorted into it (not rejected) and how many would
        have gone out immediately, over the window ``chat_stats`` uses."""
        ...


@runtime_checkable
class ReviewService(Protocol):
    """The weekly review: proposals, their delivery, decisions and undo (§12)."""

    async def build(self) -> list[Proposal]:
        """Compute + persist ``proposed`` subscription rows (idempotent per ``review_day``;
        one per chat at most, the strongest level whose rule holds; never for ``keep``,
        non-source, left or creator-leave), then call ``rt.discovery.propose()``, which
        persists its own ``new_topic`` / ``merge_topics`` rows for the same ``review_day``;
        returns both."""
        ...

    async def send(self) -> int:
        """One bot message per proposal: the four chat groups (folder, mute, archive, leave;
        ``rv:`` callbacks) then new topics and merges (``ds:`` callbacks), 1 s between
        messages, capped at ``max_proposals``. Returns how many were sent."""
        ...

    async def decide(self, proposal_id: int, decision: str) -> Proposal:
        """``approve`` | ``confirm`` | ``cancel`` | ``skip`` | ``never``. Approve on ``leave``
        first shows a confirmation (``confirming`` -> ``approved`` on ``confirm``)."""
        ...

    async def undo(self, proposal_id: int) -> Proposal:
        """Reverse a done folder/mute/archive action (leave is never undoable)."""
        ...

    async def tick(self) -> None:
        """Weekly schedule (``scheduled_moment`` helper, §9.5) using ``kv review.last_day``;
        missed weeks run at the next start."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Carries out ``approved`` proposals through the account, paced (§12)."""

    async def tick(self) -> None:
        """Run approved actions with pacing (folder/mute/archive 20–90 s apart; leaves at
        most one per ``leave_interval_minutes`` and ``leaves_per_day``), update the proposal
        messages with the outcome. Never acts on an output/staging chat."""
        ...

    async def undo(self, proposal_id: int) -> Proposal:
        """Reverse a ``done`` folder/mute/archive action (``/undo`` or the [Undo] button) and
        edit the proposal message; a leave is refused ("cannot be undone without a new
        invitation"). Called through ``ReviewService.undo``."""
        ...


@runtime_checkable
class FolderManager(Protocol):
    """Keeps "Curated" / "Low signal" in line with the DB; the ONLY caller of
    ``list_folders`` / ``get_folder`` / ``save_folder`` (§12)."""

    async def sync(self) -> None:
        """Merge rather than replace ``include_peers``, never create a folder with zero peers,
        cap at 100, keep the staging channel out, and on ``FolderLimit`` set
        ``kv folders.disabled = "limit"`` and return at once until ``/reload``."""
        ...


@runtime_checkable
class Discovery(Protocol):
    """New-topic and merge proposals from unsorted posts (§12)."""

    async def propose(self) -> list[Proposal]:
        """``new_topic`` and ``merge_topics`` proposals; called ONLY from
        ``ReviewService.build`` (§12). With ``auto_create_topics`` it calls ``accept()``
        itself for at most one topic per day and the owner is told afterwards."""
        ...

    async def accept(self, proposal_id: int, name: str | None = None) -> Topic:
        """Create the topic (``TopicsService.create(create_channel=True,
        origin="discovered")``), store the cluster members as ``examples(kind='cluster')``,
        retrain, ``sorter.resort_unsorted`` (nothing old is published) and emit
        ``examples_changed(reason="topics")``."""
        ...


# --- account ---------------------------------------------------------------------------------


@runtime_checkable
class AccountService(Protocol):
    """Binding the account: the login state machine and its plain messages (§11.3).
    ``begin``/``resend``/``code``/``password`` never persist the code, the password or
    ``phone_code_hash``."""

    async def status(self) -> AccountStatus: ...

    async def begin(self, phone: str) -> None:
        """Send the login code to ``phone``."""
        ...

    async def resend(self) -> None:
        """[Send a new code]."""
        ...

    async def code(self, code: str) -> Literal["ok", "password_needed"]:
        """Submit the code (digits only; the bot strips separators before calling)."""
        ...

    async def password(self, password: str) -> None:
        """Submit the 2FA password. On success: ``kv account.id``, ``owner_id`` if still 0,
        emit ``account_bound``."""
        ...
