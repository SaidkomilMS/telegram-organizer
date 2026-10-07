# FAQ

## Can my account get banned?

No tool that uses your account can promise that, but tg-curator is built to look like a person
who reads a lot. It uses Telegram's official API with your own `api_id`, like any Telegram app.
The things that get accounts restricted are joining chats and messaging strangers, both of which
the curator never does (see [Safety](safety.md#what-the-account-never-does)). Almost all of its
traffic is reading new posts as they arrive, which costs nothing. History reads are paced, leaves
are limited to one every 30 minutes and 3 a day, and whenever Telegram asks it to slow down it
waits as long as it is told. The bot, not your account, does all the posting. Two things to
avoid: logging in over and over, and using the same session from two places (below).

## Does it send my messages anywhere?

No. Everything stays on your server and there is no telemetry. Without a language model the
only connections are to Telegram, plus one download of the similarity model from huggingface.co
at the first start (nothing about you is sent). If you connect a provider, it receives the text
of the posts picked for your digest (to write one-line summaries), the example posts behind a
proposed new topic, the one post used to test the connection, and borderline posts only if you
turn on the second opinion. Never chat names or identifiers. With a self-hosted model, the same
text goes only to your own server. Details: [Safety](safety.md#what-leaves-your-server).

## Why is a post unsorted?

Because the classifier was not sure enough that it belongs to any of your topics: its confidence
stayed below `sorting.confidence` (0.5) or the topic's own `strictness`. Unsorted posts never
reach a channel. `curator preview` shows every post with its best topic and confidence, so you
can see how close it came. The usual fixes, best first: correct the post with **Wrong topic**
(in `/preview <topic>`), forward a few more example posts to the topic (`/topics`, the topic,
**Edit**, **Add examples**), give a topic that has only a description a category or an example
channel (a description alone sorts almost nothing), and
only then lower the threshold: `curator preview --set sorting.confidence=0.4` shows the effect
first. Short group messages (under 400 characters, `groups.min_chars`) are not sorted at all;
they only count in the statistics. A run of unsorted posts that look alike may also be a topic
you do not have yet; the weekly review proposes it.

## Why did a story not go out immediately?

Real time is for stories that spread. A post from an ordinary source goes out immediately once
two other chats have carried the same story; until then it waits 45 minutes and, if nobody picks
it up, joins the evening digest. It is still sent in real time if it spreads later that day,
before the digest goes out. A source with trust 3 skips the waiting; trust 2 needs one other
chat ([Tune](tune.md#trusted-sources)). A post also never goes out in real time when publishing
is not live yet (`/go`) or paused (`/pause`), when its topic has no channel, when it was read by
a backfill instead of as it arrived, or when it was a repeat of something already posted (the
original then shows `+N`). If too little goes out, lower `sorting.realtime_strength` from 3 to 2
and check with `curator preview` first.

## Which language model should I pick, and what does it cost?

None is required: without one the digest shows each post's first line and nothing leaves the
server. A model writes one-sentence summaries in the post's own language and names new topics.
Pick the provider you already have an account with and its first (cheapest) model; a
one-sentence summary does not need a big model. Measured cost per topic and month, with the
default 15 digest lines a day (about 450 summaries of ~600-character posts), at list prices of
October 2026:

| provider | model | about per topic and month |
| --- | --- | --- |
| OpenAI | `gpt-6-luna` | $0.03 |
| Mistral | `ministral-8b-2512` | $0.03 |
| Mistral | `mistral-small-latest` | $0.04 |
| Google | `gemini-3.5-flash-lite` | $0.12 to $0.29 |
| Google | `gemini-3.8-flash` | $0.23 to $0.49 (about double from January 2027) |
| Anthropic | `claude-haiku-4-5` | $0.31 |
| Mistral | `mistral-medium-3-5` | $0.46 |
| OpenAI | `gpt-6-sol` | $0.62 |
| Anthropic | `claude-sonnet-5-5` | $0.62 |
| OpenRouter | any of the above | the same model's price, fetched live |
| Self-hosted | anything on Ollama, vLLM, LM Studio, llama.cpp | $0 (your own hardware) |

Naming new topics adds under $0.02 a month; the second opinion, if you turn it on, $0.01 to $0.30.
So three topics on a cheap model cost about ten cents a month. You can set a monthly cap in
`/llm`; when it is reached the digest falls back to first lines until the month ends. A
self-hosted model needs a machine that can run one (a GPU or a strong CPU): the curator's own
2-core server is deliberately not used for that, because a model small enough would cost more time
than it adds in quality.

## Can I run it for two accounts?

Not in one installation: one installation is one account, one set of topics and one bot. Run a
second copy with its own bot and its own home folder. With Docker, copy the project folder (the
compose project then gets its own volume) and use a second `.env`. With systemd, install a second
unit whose `TG_CURATOR_HOME` points elsewhere and whose user is different. Each copy needs about
700 MB of memory, so a 4 GB server holds two comfortably.

## Can I copy `user.session` to a second machine?

No, never. Telegram notices the same login being used from two places at once
(`AUTH_KEY_DUPLICATED`) and terminates it for both, so both copies stop and you have to log in
again. The same happens if two processes on one machine share the file. To move the curator,
stop the old installation, copy `settings.toml` if you like (it holds no session), and run
`/bind` on the new machine: it creates a new session of its own. You can then terminate the old
one under Settings → Devices.

## What happens if I stop it for a week? For months?

**For a week:** nothing is lost, it just has a gap. Posts published while it was down are not
fetched on their own; run `curator backfill --days 7` (or `/preview refresh` for the last three
days) to fill the statistics and the repeat detection. Old posts are never sent in real time, and
only those from the last 26 hours can make a digest. If a digest hour passed during the outage,
that digest is sent at the next start, and a missed weekly review runs then too. Your login
survives, unless you set Telegram to end inactive sessions after one week (Settings → Devices).

**For months:** the same, plus Telegram ends sessions that stay unused longer than the period set
under Settings → Devices, "Automatically terminate old sessions" (usually 6 months). The curator
then starts without the account, the bot tells you, and `/status` says "Account: session lost —
run /bind". Send `/bind` and everything continues with your topics, settings and history intact.
