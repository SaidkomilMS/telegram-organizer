# Set up

Setup is one conversation with your bot, about fifteen minutes. Every step either works or says
exactly what to fix, every step can be repeated without losing anything, and `/setup` picks up
wherever you stopped. The command line can do the same steps, for people who prefer a terminal
or script an install; both write the same settings file.

| step | in the bot | on the command line |
| --- | --- | --- |
| 1. Install and start | | `curator run` |
| 2. Claim the bot | `/start CODE` | `curator login` sets you as the owner |
| 3. Bind the account | `/bind` | `curator login` |
| 4. Create topics | `/topics add` | `curator topics add "ML & AI" --category tech --create-channel` |
| 5. Language model | `/llm` | `curator llm` |
| 6. Preview | `/preview` | `curator backfill` then `curator preview` |
| 7. Go live | `/go` | `curator run --live` |

With Docker, put `docker compose exec tg-curator` in front of each `curator` command; with the
systemd service, use the alias from [Install](install.md#path-2-pip-and-systemd).

## 1. Install and start

Follow [Install](install.md). When the service starts with the three values in place, its log
shows a line with a one-time **claim code**, and lines that confirm each part as it comes up
("4 topics, all channels resolved", "found 142 chats, 3 are output channels" once the account is
bound). Read the log with `docker compose logs -f` or `journalctl -u tg-curator -f`.

## 2. Claim the bot

Open your bot in Telegram and send `/start` followed by the code from the log, for example
`/start K7QM4XPR`. From then on your Telegram account is the bot's owner and everyone else who
writes to it is ignored. Five wrong attempts print a new code to the log.

Then send `/setup`. The bot walks you through steps 3 to 7 and, if you stop halfway, `/setup`
continues from the first step that is not done yet.

## 3. Bind the account

`/bind` asks three things, one message each:

1. **Your phone number** in international format, e.g. `+998901234567`.
2. **The login code** Telegram sends to your other devices. Type it **with a space between the
   digits**, like `1 2 3 4 5`. This matters: Telegram cancels a login code the moment it sees
   it pasted as is into any chat, so a code sent as `12345` will be refused. Spaces, dashes or a
   letter in front (`c12345`) all work; the bot keeps only the digits. If a code is refused
   anyway, tap **Send a new code** and try again with spaces.
3. **Your two-step verification password**, if you set one in Telegram.

The bot deletes the messages holding the code and the password right after reading them and
never stores either. The answer is "logged in as …" with your name. From now on the account stays
bound on the server; nothing asks again unless you end the session yourself (see
[Safety](safety.md#sessions)). `/bind` can be run again any time, for example to switch to
another account: the new login runs beside the current one, which keeps working until the new
sign-in succeeds. If you give up halfway or a code is refused, nothing changes.

On the command line, `curator login` asks the same questions in the terminal. There you type the
code as is (a terminal is not a Telegram chat). `curator login --phone +998901234567` skips the
first question. Stop the service first (`sudo systemctl stop tg-curator`, or
`docker compose stop` and then `docker compose run --rm tg-curator curator login`): the session
may only ever be open in one process. If the account has no owner yet, `curator login` also makes
it the bot's owner.

## 4. Create topics

A topic is a name, the private channel it posts to, and what the classifier has been given
about it. There are no keyword lists.

`/topics add` first shows the four example topics that ship in the settings file ("ML & AI",
"Fintech", "Узбекистан: новости", "Футбол") with **Keep**, **Edit** and **Remove**. Then, for a
new topic:

1. **A name**, e.g. `Crypto & exchanges`. A name that exists already offers to edit that topic.
2. **What the topic is**, any combination of:
   - a **built-in category** from the buttons: Technology & AI, Science & space, Finance &
     economy, Crypto, Politics & government, World & conflict, Sport, Health & medicine,
     Education, Culture & entertainment, Real estate, Jobs & career, Travel & tourism, Auto &
     transport, Society & crime, Weather & disasters, Religion, Lifestyle, Humor & memes, Ads &
     promotions. A topic with a category sorts from the first post;
   - **example posts**: forward a few posts that belong to the topic to the bot;
   - an **example channel**: send the `@username` or link of a channel that is a good example;
     the account reads its recent posts. A private chat you are not in gets "join the chat in
     Telegram first, then send the link again";
   - a **description** in any language. Optional, and weak on its own: a topic that has only a
     description sorts nothing until it gets a category or examples, and the bot says so.
3. **The channel**:
   - **Create a private channel**: the account creates it, makes the bot an admin that may post
     and edit, and links it. Telegram limits how fast an account may create channels; if it asks
     to wait, the topic is saved without a channel and the curator creates it by itself once the
     wait is over, then tells you (`/topics` also offers **Create channel** meanwhile).
   - **Use an existing one**: send its link or `@username`. The account must be its creator or an
     admin, and the channel must not already belong to another topic (each topic has its own). If the bot is not an admin there, the account adds it; if that is not possible the
     reply says "the bot cannot post into X: add @yourbot as an admin with Post Messages".
   - **No channel yet**: the topic is tracked (it shows up in the preview and the statistics)
     but nothing is posted until you give it a channel.

`/topics` lists the topics with buttons to edit (name, description, category, strictness,
channel), merge two topics, remove one, or create a missing channel. Removing a topic never
touches its channel.

Command line:

```sh
curator topics                      # sync the settings file, list topics and channel status
curator topics categories           # the built-in category keys
curator topics add "Crypto & exchanges" --category crypto --create-channel
curator topics add "Tashkent real estate" --example-channel @some_channel --channel @my_channel
curator topics remove crypto-exchanges
```

`curator topics add` takes `--category KEY`, `--description TEXT`, `--example-channel REF`, and
either `--channel REF` (an existing channel) or `--create-channel`. You can also edit the
`[[topics]]` blocks in `settings.toml` by hand; `curator topics` (or `/reload` in the bot) checks
them and says "4 topics, all channels resolved" or exactly which channel could not be used.

### The staging channel

When you go live with the default repost style, the account creates one more private channel,
"tg-curator media". It is how photos and videos are handed to the bot: the account copies the
media there and the bot attaches it to the post in your topic channel. Keep it; nobody else sees
it, and it stays out of the curator's folders and statistics. If you delete it, it is recreated.

## 5. Connect a language model (optional)

`/llm` offers three choices, each with one line about cost and privacy:

- **None** (default): digest lines are the first line of each post. Nothing leaves the server.
- **Self-hosted**: any OpenAI-compatible server: Ollama, vLLM, LM Studio or llama.cpp. Pick a
  preset or type the address, add a token if your server needs one, then pick the model.
- **Provider**: Anthropic, OpenAI, Google, Mistral or OpenRouter. Paste your API key, then
  choose a model from a short list (cheapest first) or type any model id.

Before saving, the bot summarises one real post (the most recent long post that was picked for
a digest, or a bundled sample if there is none yet) and shows the line and how long it took.
Only a working setup is saved. Keys and tokens are deleted from the chat right after they are
read. You can also set a monthly spending cap and turn on the second opinion for borderline posts
(see [Tune](tune.md#language-model)). Which one to pick and what it costs: [FAQ](faq.md).

Command line: `curator llm` asks the same questions. Flags skip them:

```sh
curator llm --mode none
curator llm --mode selfhosted --base-url http://localhost:11434/v1 --model qwen3:8b
echo "$KEY" | curator llm --mode provider --provider openai --model gpt-6-luna --key-stdin
```

`--key-stdin` reads the key or token from standard input, so it never appears in your shell
history or the process list.

## 6. Preview

`/preview` reads the last three days from every chat (one progress message: "reading 142 chats…
37/142"), sorts them with the current settings and reports how many posts went to each topic,
how many were repeats and how many stayed unsorted, with examples. Nothing is posted.

`/preview ML & AI` (any topic name) shows that topic's individual decisions, a page at a time.
Each has a **Wrong topic** button: pick the right topic or "Not for me" and the classifier
learns from it on the spot. `/preview refresh` reads the chats again; otherwise the preview
reuses what was read in the last 24 hours.

Command line:

```sh
curator backfill                    # read the last 3 days of every chat (--days N, --chat REF)
curator preview                     # every decision: topic, confidence, repeat of what, real-time or not
curator preview --topic ml-ai --days 2
```

`curator preview` prints one line per post. It is also what to attach to a bug report.

## 7. Go live

While `timezone` under `[general]` is still `"UTC"`, `/setup` first asks which zone your day is
in: one tap uses the server's zone, [Another zone…] lets you type an IANA name such as
`Asia/Tashkent`, and [Keep UTC] keeps it. The digest and review hours are local to that zone; it
can be changed any time in `/settings` → Timezone.

`/go` checks that the account is bound and that at least one topic has a channel the bot can
post into; if not, it names the missing step. Then it switches publishing on and confirms the
schedule: the digest time in your timezone, the review day, and "send /pause to stop posting
without stopping intake". The first real-time posts appear as soon as a source publishes
something that qualifies; the first digest arrives at the digest hour that evening.

On the command line, `curator run --live` switches publishing on and starts the service (or set
`live = true` under `[publishing]` and restart).

## Day to day

| bot | command line | what it does |
| --- | --- | --- |
| `/status` | | is everything running, last post read, next digest |
| `/pause`, `/resume` | | stop and restart posting; reading and sorting go on |
| `/digest preview` | `curator digest --preview` | tonight's digest, not sent |
| `/digest` | `curator digest` | send a digest now (all topics, or `curator digest ml-ai`) |
| `/stats` | `curator stats` | per-chat signal, repeats and volume |
| `/review` | `curator review` | the subscription review, now |
| `/settings` | edit `settings.toml` | change settings; see [Tune](tune.md) |
| `/reload` | restart the service | re-read the settings file |
| `/help` | `curator --help` | the list of commands |
| | `curator chats` | channels and groups with their ids, for the settings file |

The bot also sends you, unprompted, only these: one line per digest ("ML & AI: 15 posts;
Fintech: 9"), the review proposals once a week, and a warning if reading stops for more than an
hour, the account's session ends, or the bot cannot post into a topic channel.
