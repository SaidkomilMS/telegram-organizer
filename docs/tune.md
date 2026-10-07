# Tune

Everything you can change lives in one commented file, `settings.toml` in the curator's home
(see [Install](install.md#where-things-are)). The bot's `/settings` and `/topics` edit the same
file and keep your comments, so the two never drift apart.

- Changes made **in the bot** apply at once.
- Changes made **by hand** apply on the next start of the service, or right away with `/reload`
  in the bot (everything except `[telegram]` and `[storage]`, which need a restart).
- A file with a mistake is not half-loaded: the service names the key and what to fix.

## Try a change before it applies

The preview replays the last few days of real posts with your settings and prints every
decision. Nothing is posted and nothing is written. Each line starts with the post's number, so
"repeat of post #12" points at the line of the original; "immediate" is where the post ends up
once later repeats from other chats are counted, as the live service would do.

```sh
curator preview                                       # current settings
curator preview --set sorting.confidence=0.4          # what would change at 0.4
curator preview --set sorting.realtime_strength=4 --set topics.ml-ai.strictness=0.6
curator preview --topic ml-ai --days 5
```

`--set key=value` can be repeated; keys are the dotted names below, and a topic's strictness is
`topics.<key>.strictness`; a source's trust is `sources.<chat id>.trust` (0–3). The preview also
reads the file as it is now, so a value you edited by hand is replayed before the service
applies it at its next start or `/reload`; the first lines of the output name those values. Only
the sorting keys, topic strictness, source trust and the digest schedule change a replay of
stored posts; other edits (group floors, the review, posting style, the language model) are
named instead, and `--set` refuses them. When the output reads right, put the value in the file
(or apply it in the bot). In the bot, `/settings` shows **Preview**, **Apply** and **Cancel** for
the sorting keys, topic strictness and source trust: Preview prints the new counts next to the
current ones and only Apply writes the file.

## Topics: the main lever

A topic that keeps missing posts or catching the wrong ones is fixed by **teaching it**, not by
moving thresholds:

- **Correct posts.** The **Wrong topic** button under every real-time post (and in
  `/preview <topic>`) moves the post and teaches the classifier at once. A near-identical post
  flips with it. About seven corrections per topic make it fit your taste.
- **Forward examples.** `/topics`, pick the topic, **Edit**, **Add examples**, then forward posts
  (or paste their text) and tap **Done**. Six examples per topic is a good start for a topic
  that has no built-in category (a region, a company, a niche).
- **Give it an example channel** whose posts are typical for the topic.
- **Give it its own strictness** when one topic needs to be tighter than the others
  (`strictness` in its `[[topics]]` block; 0 means "use `sorting.confidence`").

The `[[topics]]` keys:

| key | meaning |
| --- | --- |
| `key` | stable id (a-z, 0-9, `-`, `_`; at most 32). Never change it after creation. |
| `name` | shown in the channel and in reports |
| `channel` | numeric id, `@username` or t.me link of the private channel; `0` = tracked only, nothing posted. Resolved values are rewritten to the numeric id. A channel can be replaced but not removed: remove the topic instead, or `/pause`. Each topic needs its own channel; the curator's private media channel cannot be one. |
| `category` | optional built-in category key; `curator topics categories` lists them. An unknown key (say `technology` instead of `tech`) is reported by `curator topics` and `/reload` with the closest key |
| `description` | optional, any language |
| `example_channel` | optional `@username`/link of a typical channel |
| `strictness` | optional per-topic confidence threshold (0..1); `0` = use `sorting.confidence` |

Removing a `[[topics]]` block deactivates the topic: its waiting posts become unsorted, its
channel is left alone. Putting the same `key` back revives it (it learns its examples afresh).

## The two sensitivity settings

These two cover most of the feel. Both are in `[sorting]`.

**`confidence`** (default `0.5`): how sure the classifier must be before it sorts a post into a
topic. Stricter means fewer, cleaner posts.

| symptom | move |
| --- | --- |
| good posts stay unsorted | lower it (0.4) |
| posts land in the wrong topic | raise it (0.6) |

**`realtime_strength`** (default `3.0`): how strong a post must be to go out immediately. Stricter
means more goes to the digest.

| symptom | move |
| --- | --- |
| too much goes out in real time | raise it (4.0) |
| important stories only reach the digest | lower it (2.0) |

How strength is counted: a post starts with the trust of its source (1 for an ordinary chat),
gains `link_bonus` (0.25) for an outside link and `length_bonus` (0.25) for being at least
`length_bonus_chars` (600) long, and `corroboration_weight` (1.0) for every **other** chat that
carries the same story. With the defaults, an ordinary source's post goes out immediately once
two other chats have picked it up; a source with trust 3 goes out on its own. A post that is not
strong enough waits `hold_minutes` for others to pick it up, then joins the digest. If it spreads
later the same day, it is still sent in real time, as long as the digest has not gone out yet.

## Trusted sources

List chats you want treated differently:

```toml
[[sources]]
chat = "@kunuz"        # numeric id, @username or t.me link
trust = 3
```

| trust | effect |
| --- | --- |
| 3 | always immediate: every sorted post goes out at once, no waiting, never the digest |
| 2 | needs less: one other chat carrying the story is enough |
| 1 | neutral, the default for every chat not listed |
| 0 | digest only: its posts never go out in real time |

`/settings` has a **Trusted sources** entry to list, add and remove them. `curator chats` prints
every chat with its id.

## Every other setting

### `[sorting]`

| key | default | what it does, and when to move it |
| --- | --- | --- |
| `duplicate_similarity` | `0.90` | how alike two posts must be to count as the same story in other words. Same story posted twice: lower (0.85). Different stories merged: raise (0.95). |
| `duplicate_similarity_cross` | `0.86` | the same for two posts in different languages (a translation scores a little lower). Move it together with the one above. |
| `dedup_window_days` | `3` | a post is a repeat only of something seen in the last N days |
| `hold_minutes` | `45` | how long a sorted post waits for corroboration before joining the digest. Stories reach the digest that should have gone out: raise. |
| `neutral_trust` | `1.0` | the trust of chats not listed under `[[sources]]` |
| `corroboration_weight` | `1.0` | strength per other chat carrying the story |
| `link_bonus` | `0.25` | strength for an outside link |
| `length_bonus`, `length_bonus_chars` | `0.25`, `600` | strength for a long post, and what "long" means |
| `second_opinion` | `false` | ask the connected language model about borderline posts (within 0.10 below the threshold). Costs one request per such post, and sends its text to the model. |

Repeats are caught three more ways that need no tuning: forwards of the same message, the same
text, and the same outside article link.

### `[publishing]`

| key | default | |
| --- | --- | --- |
| `live` | `false` | set by `/go`; until then nothing is posted |
| `style` | `"repost"` | `"repost"`: the bot writes the post and attaches a copy of the media; works for every channel. `"forward"`: your account forwards the original, keeping Telegram's attribution, and the bot adds a short line under it with the source, `+N` and the **Wrong topic** button. A forward cannot be moved: after a correction it stays where it is and only its label changes. Group threads and protected channels are always reposted. |
| `min_gap_seconds` | `20` | minimum spacing between two posts into one channel |
| `staging_channel` | `0` | filled in automatically (the "tg-curator media" channel) |

### `[digest]`

| key | default | |
| --- | --- | --- |
| `hour`, `minute` | `21`, `0` | when the daily digest is sent, in `[general].timezone` |
| `items` | `15` | how many posts a digest lists |
| `window_hours` | `26` | only posts from the last N hours can make the digest; nothing rolls over to the next day |
| `line_chars` | `180` | length of one digest line |

The digest ranks posts by how many other chats carried the story, how many views a post got
compared with what that source usually gets, and the source's trust. A post that did unusually
well for a small channel ranks above one that was ordinary for a big channel.

### `[groups]`

Groups are conversations: a run of messages from one person within a few minutes, plus the
replies to it, is read as one piece of text.

| key | default | |
| --- | --- | --- |
| `min_chars` | `400` | shorter pieces are counted in the statistics but never sorted. A post forwarded from a channel needs only a quarter of it, a shared link a few words of its own. Good group threads are ignored: lower. Chatter gets sorted: raise. |
| `unit_gap_minutes` | `5` | messages by the same person within this window form one piece. Group posts reach a topic channel this long after the conversation goes quiet (at most 30 min after its first message): lower it for faster group posts, raise it to keep threads whole. |

Replies stay with the message they answer for a day, however slowly they come, so a slow
thread is still one post linking to its first message.

### `[general]`

| key | default | |
| --- | --- | --- |
| `timezone` | `"UTC"` | IANA name such as `"Europe/Berlin"` or `"Asia/Tashkent"`; the digest and review times use it |
| `language` | `"en"` | the bot's language (see [CONTRIBUTING](../CONTRIBUTING.md#adding-a-language)) |

To change a few bot messages without translating everything, put the keys you want to reword
into `messages.toml` in the curator's home.

### `[review]` and `[folders]`

The weekly review, its rules and the leave pacing are explained in
[Manage subscriptions](subscriptions.md#the-rules-and-their-settings), with every key.

### Language model

`[llm]` is what `/llm` writes. You rarely edit it by hand.

| key | default | |
| --- | --- | --- |
| `mode` | `"none"` | `"none"`, `"selfhosted"` or `"provider"` |
| `provider` | `""` | `anthropic`, `openai`, `google`, `mistral` or `openrouter` |
| `model` | `""` | model id; any id the provider accepts |
| `base_url` | `""` | the self-hosted server, e.g. `"http://localhost:11434/v1"` |
| `api_key` | `""` | provider key or self-hosted token; kept secret |
| `monthly_cap_usd` | `0.0` | spending cap per month; `0` = none. When reached, digests use first lines until the month ends and the bot says so once. |
| `timeout_seconds` | `30` | per request; can only raise the built-in minimum (60 s for providers, 180 s for self-hosted) |

### `[ml]`

| key | default | |
| --- | --- | --- |
| `embedder_file` | `"onnx/model_qint8_avx512_vnni.onnx"` | the small, fast similarity model. On an older x86 server that misses duplicates, switch to `"onnx/model.onnx"` (full precision, 470 MB download, about twice the memory). |
| `threads` | `2` | CPU threads for the model; 2 suits a 2-core server |

### `[storage]`

| key | default | |
| --- | --- | --- |
| `database_url` | `""` | blank = SQLite file in the home; or `"postgresql+asyncpg://user:pass@host/db"` |
| `keep_embeddings_days` | `30` | similarity data of older posts is purged (the text stays) |
| `keep_posts_days` | `0` | posts older than this are purged; `0` = keep forever |

### `[telegram]`

`api_id`, `api_hash`, `bot_token` (see [Install](install.md)) and `owner_id`, which is filled in
when you claim the bot.
