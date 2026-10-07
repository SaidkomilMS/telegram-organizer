# tg-curator — Functional Specification

Oct 5, 2026

## What it does

You point tg-curator at your Telegram account, name the topics you care about, and from then on everything your channels and groups publish is sorted for you: the important posts land in a channel per topic within minutes, the rest arrive once a day as a short ranked digest, and the same story never reaches you twice. Over the following weeks it also tells you which subscriptions actually contribute something, proposes what to mute, archive or leave, and does only what you approve with a tap.

It runs on your own server, open source. Sorting is done locally by small models that never leave the server: a multilingual similarity model for duplicates and a topic classifier that ships with a standard set of categories and learns from your corrections, so it works the same for Russian, Uzbek and English channels and costs nothing per post. Written digest summaries and automatic naming of new topics are optional and use a language model you connect: one you host yourself, or a frontier provider you give a key for. Without one, the digest shows the first line of each post.

A typical day with it looks like this. You open your "ML & AI" channel and see the four things that mattered since the morning, each with the source, the original photo if there was one, and a link to the original. At 21:00 the digest arrives with fifteen more items ranked by how widely they spread and how much attention they got, one line each. On Sunday the curator bot sends a review: nine chats posted nothing unique in a month, two are pure reposts of others, and a cluster of forty unsorted posts looks like a topic you don't have yet, so it offers to create "Crypto & exchanges" for you. You tap approve on a few, and it carries that out over the next hour.

## What you need

A Telegram account, a bot, and a server; setup takes about fifteen minutes and needs no programming.

- **Your Telegram account** — the one subscribed to the channels and groups you want curated. It is used for reading, for copying media into your own channels, and for creating those channels when you ask. It is never used to message anyone.
- **A bot** you create with @BotFather in one minute. The bot does all the posting into your topic channels, walks you through setup, and runs the approval dialogue with you, so your own account never looks like a bot.
- **One private channel per topic**, which the setup creates for you (or you create yourself and name in the settings). You can start with two topics and add more later, or let the curator propose them.
- **A small Linux server** — any VPS with 2 CPU cores and 4 GB of memory is enough; no GPU. The first start downloads the two local models (about a gigabyte) once.
- **Telegram API access** for your account, two values from my.telegram.org that every Telegram client app uses. They stay on your server.

No AI provider account is required: without one the digest lists the first line of each post. A key for a frontier provider, or the address of a model you host yourself, turns on written summaries and automatic naming of new topics, and is set up from the bot. No other paid services, no database to install (one file on disk is the default; Postgres is optional for people who already run it).

## Setting up

A new user goes from zero to a running curator in one conversation with the bot; the command line does the same for people who prefer it. Either way, every step either works or says exactly what to fix.

1. **Install** on the server with one command (a Docker Compose file or a `pip install`), fill in the bot token and the two API values, and start the service. On first start it prints a one-time claim code.
2. **Claim the bot.** Open the bot in Telegram, send `/start` and the claim code. From now on this account is the owner; everyone else who writes to the bot is ignored.
3. **Bind the account** by replying to the bot's questions: phone number, the code Telegram sends, the two-factor password if there is one. The bot deletes the messages holding the code and the password right after using them. After this the account stays bound on the server; nothing asks again.
4. **Create the topics** in the same chat: `/topics add`, then a name, and then what the topic is: a category from the built-in list (technology, finance, politics, sport, science…), a few example posts forwarded to the bot, or a channel that is a good example of it. A written description is optional, never required. The bot creates a private channel for the topic, makes itself an admin and links the two. Existing channels can be used instead by sending their link. Four example topics are offered to start from and edit.
5. **Connect a language model** with `/llm`, or skip this. Three choices, each with a line on what it means for cost and privacy: *None* (the default: digest lines are the first line of each post, nothing leaves the server); *Self-hosted* (the address of any OpenAI-compatible server such as Ollama, vLLM, LM Studio or llama.cpp, a token if it needs one, and the model name); *Provider* (pick Anthropic, OpenAI, Google, Mistral or OpenRouter, paste the key, choose a model from a short list of sensible ones). The bot tests the choice by summarising one real post, shows the result and how long it took, and only then saves it. Keys and tokens are deleted from the chat right after they are read.
6. **Preview.** The bot pulls the last three days from every chat and sends a short report: how many posts went to each topic, how many were repeats, how many stayed unsorted, with examples. `/preview <topic>` shows the individual decisions; a wrong one is corrected with a tap and the classifier learns from it on the spot. Nothing is posted during this step.
7. **Go live** with `/go`. The first real-time posts appear as soon as a source publishes something that qualifies; the first digest arrives at the configured hour that evening. The bot confirms the schedule and reminds the user how to pause.

The same seven steps exist as `curator login`, `curator chats`, `curator topics`, `curator llm`, `curator preview` and `curator run` for command-line users and for scripted installs; both paths write the same settings file.

If a step fails the message says what to do: a code that was not accepted, a topic channel that cannot be found, the bot not allowed to post. Every step can be repeated at any time; nothing is lost by running one twice, and `/setup` restarts the walkthrough from wherever it stopped.

## Your topic channels

Each topic channel receives only two kinds of messages: individual posts that were strong enough to go out immediately, and one digest a day. Nothing else ever appears there, so the channel stays readable even when the sources behind it publish hundreds of posts a day.

A real-time post shows the name of the source channel in bold, a `source` link that opens the original, and the full text of the post. If the post had a photo, video or file, that is noted and the link opens it. If other channels carried the same story, the post says so (`+3 more`), which is itself a useful signal: it only came through because it spread. Under each post sits a small **Wrong topic** button: a tap offers the other topics or "not for me", moves the post, and teaches the classifier; only the owner's taps count.

The user chooses between two styles. In **repost** style (default) the bot writes the post as described and attaches the original photo, video or file, copied over by the account, so the topic channel is complete on its own; this works for every channel, including those that forbid forwarding (where only the text and a link are kept). In **forward** style the user's own account forwards the original, so Telegram's own attribution is preserved; channels that forbid forwarding fall back to the repost style automatically.

A story spreading across several sources is posted once: the first copy seen is the one that goes out, and later copies only add to its `+N` count. A post never appears twice in a channel, even if the service restarts mid-way, and a post never appears in more than one topic channel.

A topic that has no channel assigned yet is still tracked in the statistics, so new topics can be tried out before giving them a home.

## The daily digest

Once a day, at an hour the user picks, every topic channel receives one message listing the best of what did not go out in real time: by default the top fifteen posts of the last 26 hours, one line each, newest ranking first.

Each item is the first line of the post or, when a language model is connected, a one-line summary written by it in the language of the original, followed by the source name, a link to the original and, when the story was carried by other chats too, a `+N`. A digest longer than one Telegram message is split into two, never more than needed. Posts that did not make the cut are dropped; they never roll over into the next day, so the digest is always about yesterday.

The ranking favours what spread and what got attention. Just before composing the digest, the curator re-reads the view counts of the candidate posts, compares each to what that source normally gets, and combines that with how many other chats carried the story and how trusted the source is. A post that is unremarkable for a big channel ranks below a post that did unusually well for a small one.

A connected language model sees one post at a time and writes one sentence, once a day in the background, so a provider is billed for a few hundred short requests a month at most. Summaries are deliberately never generated on the server's own CPU: a model small enough to run there costs more in time than it adds in quality, so without a connected model the digest stays extractive. The line length, the number of items and the hour are settings; `curator digest --preview` (or `/digest preview` to the bot) shows tonight's digest without sending it, and `curator digest` sends it right now for all topics or for one.

## How posts are sorted

Every new post goes through four questions, in order, within a couple of minutes of being published: is it a repeat, which topic is it, how strong is it, and should it wait.

**Is it a repeat?** A post is a repeat if its text matches something seen in the last three days, if it points to the same article link, or if it says the same thing in different words (the similarity model catches reworded reposts and translations between Russian and English). A repeat is never published. If it came from a different chat than the original, it counts as corroboration for the original, which is what makes stories that spread rise.

**Which topic?** A local classifier decides, using what each topic was given when it was created (a built-in category, example posts, an example channel, an optional description) and everything the user has corrected since. A post goes to the topic the classifier is surest about, provided it is sure enough; otherwise it stays unsorted and never reaches a channel. There are no keyword lists to maintain: when a post lands in the wrong place, the user corrects it once and the classifier learns from that, on the server, within minutes.

**How strong is it?** A post starts with the trust level of its source (neutral for most, higher for chats the user marks as trusted) and gains a little for carrying an outside link or being long, and a lot for every other chat that posts the same story. With default settings, a post from an ordinary source goes out immediately once two other chats have picked it up; a post from a trusted source goes out immediately on its own.

**Should it wait?** A newly sorted post waits 45 minutes for corroboration before being assigned to the digest. If it becomes strong enough during that time, it goes out right away. If it is assigned to the digest and then spreads later in the day, it is still promoted to real time as long as the digest has not been sent yet.

The user never sees any of this machinery; what they see is that a channel gets a few posts a day that turned out to matter, plus a digest, and that corrections stick. Two settings control the feel: how sure the classifier must be before it sorts a post, and how strong a post must be to go out immediately. Both are set from the bot or in the settings file, and the preview command shows the effect of a change on the last few days of real posts before it applies to anything live.

**For the implementer.** The classifier is a local model, never a keyword list. Start from an open-source multilingual topic classifier (models trained on the IPTC Media Topics taxonomy exist at a few hundred megabytes and cover Russian and English); if none performs well on real posts, train one on top of the similarity embeddings from public news-category datasets. On that base keep a small per-user layer trained from the owner's corrections and examples, retrained on the server whenever new ones arrive, so every installation sorts the way its owner does.

## Channels and groups

Channels are treated as publications: every post is a candidate. Groups are treated as conversations: only substantial messages are candidates, and everything else is counted but never forwarded anywhere.

In a group, messages are first stitched into conversation units: a run of messages from the same person within a few minutes, and the replies that hang off a message, are treated as one piece of text. A unit shorter than about a paragraph (the exact floor is a setting) is ignored for sorting. What passes are the long messages, the articles people forward into the group, announcements, and substantive threads, which are published as one post linking to the first message of the thread. A group that mostly chats will therefore contribute little to the topic channels but still shows up in the statistics with its real volume, which is exactly what makes the later cleanup fair: a loud group with no substance is visible as such.

The user's own messages in groups are ignored. Service messages (joins, pins, title changes) are ignored. Edits to a post are not followed; the version first seen is the version used. Deleted posts are not removed from the topic channels. Albums of several photos count as one post, the one carrying the caption.

Newly joined channels and groups are picked up automatically; leaving one stops its intake but keeps its history and statistics. Nothing needs restarting when subscriptions change.

## Managing your subscriptions

The curator knows, for every chat, how much of what it posts ever reached a topic channel; after a few weeks that becomes a cleanup list, and the bot turns it into proposals the user approves or rejects one tap at a time.

**Statistics.** `curator stats` (or `/stats` to the bot) lists every chat with its volume over the last 30 days, its *signal* (the share of its posts that were sorted into a topic), its *duplicate share* (how often it merely repeated something another chat had already posted), and how many of its posts were actually published. A chat with 300 posts, 2% signal and 70% duplicates is a channel the user is paying attention to for nothing.

**The weekly review.** Every Sunday (day and time are settings) the bot sends one message per proposed action, grouped: chats to move into a "Low signal" folder, chats to mute, chats to archive, chats to leave. Each proposal names the chat and the reason in plain words ("41 posts in 30 days, 0 reached a topic, 85% were repeats of @kunuz") and has three buttons: Approve, Skip, Never ask again. A chat is proposed for leaving only after at least 30 days of observation and only if it contributed nothing unique in that time; before that, the strongest proposal is a folder move or a mute. The rules behind each proposal level are settings with sensible defaults.

**Approval.** Nothing happens until the user taps Approve. Approved actions are carried out by the user's own account, slowly: folder moves and mutes within minutes, leaves no faster than one every half hour and no more than a handful per day, so the account never looks automated. Each proposal message updates itself to show what was done ("Left ✓", "Muted until Nov 4") or why it could not be ("could not leave: admin of this group"). Skipped proposals come back in a later review if the numbers still justify them; "Never ask again" marks the chat as kept.

**Folders.** The curator can keep two folders of its own in the user's Telegram: "Curated" containing the topic channels, and "Low signal" containing the chats it has flagged. It never touches folders the user created, and it never reorders chats in the main list.

**Undo.** Mutes, archives and folder moves are reversible from the bot ("/undo" on the proposal) or by hand in Telegram. Leaving a private group cannot be undone without a new invitation, which is why it is the only action that is never proposed early and always shows a confirmation before executing.

**New topics, found for you.** Unsorted posts are not thrown away; the curator keeps looking for groups of them that resemble each other. When a tight cluster has formed (by default thirty or more posts within a week) the bot proposes a new topic: a name taken from the classifier's built-in categories ("Science & technology", "Real estate"), five example posts so the user can judge it, and, when a language model is connected, a sharper name and a short description written from the examples. A tap creates the topic and its private channel; renaming is one more tap. Everything in the cluster is sorted into the new topic from then on, and the review marks how much of the previously unsorted volume it absorbed. A setting makes this fully automatic (create without asking, notify afterwards) for users who prefer no questions; the default is to ask. The same mechanism points out when two existing topics keep competing for the same posts and suggests merging them.

## Tuning

Everything the user can change is reachable in two ways that stay in sync: through the bot (`/topics`, `/settings`, with buttons and short prompts) and in one commented settings file on the server. Every change can be previewed against the last few days of real posts before it affects a channel.

**Topics** are the main lever. A topic is a name, the channel it posts to, and what the classifier has been given for it: a built-in category, example posts or an example channel, an optional description, and every correction the user has made since. There are no keyword lists. The normal fix for a topic that keeps missing things is to forward it a few more examples or to correct a few posts; a topic can also have its own strictness when one needs to be tighter than the others.

**Trusted sources** are chats the user names with a trust level. A fully trusted source bypasses the waiting and the digest entirely: every sorted post from it goes out immediately. A mildly trusted one needs less corroboration. Trust can also be set below neutral for a chat whose posts should only ever reach the digest.

**Two sensitivity settings** cover the rest: how sure the classifier must be to sort a post (stricter means fewer, cleaner posts), and how strong a post must be to go out immediately (stricter means more goes to the digest). The defaults are tuned for the bundled models; the file says what to move in which direction for each symptom ("good posts unsorted → lower this", "same story posted twice → lower that").

**Smaller settings**, each with a sensible default: the digest hour and length, the waiting time before a post is sent to the digest, the minimum length for group messages, the posting style (repost or forward), the weekly review day, and the leave-pacing limits.

**Language model.** `/llm` (or the same keys in the settings file) connects what writes the summaries and names new topics: none (the default; digests show first lines), self-hosted (address, optional token, model name) or a provider (name, key, model). There is deliberately no built-in model: one small enough for the server's CPU would cost more in time than it adds in quality. For a provider a monthly spending cap can be set; when it is reached the curator falls back to first-line digests until the month ends and says so once. A separate switch lets the connected model give a second opinion on borderline posts, the ones that almost reach a topic's threshold; it is off by default because it costs a request per borderline post and the classifier alone is usually right.

**Preview.** `curator preview` replays the recent backlog with the current settings and prints every decision: the post, its topic, how sure the classifier was, whether it was a repeat and of what, and whether it would have gone out immediately. Nothing is posted. The user edits the file, runs the preview again, and goes live when the output reads right. Settings take effect on the next start of the service; topic descriptions can be reloaded without a restart via `/reload` to the bot.

## Running it day to day

Once started, the service needs no attention: it survives reboots, catches up after outages, and the only things the user hears from it are the digest confirmation and the weekly review.

**Commands**, all under `curator`:

| Command | What the user gets |
| --- | --- |
| `login` | one-time sign-in of the account |
| `chats` | the list of channels and groups with identifiers, for the settings file |
| `backfill` | the last few days of posts pulled in, for previewing |
| `preview` | every sorting decision on the backlog, nothing posted |
| `digest --preview` / `digest` | tonight's digest shown / sent now |
| `stats` | per-chat signal, duplicates and volume |
| `review` | the weekly proposals, sent now |
| `run` | the service itself |

**The bot as a remote control.** The same bot that posts the digests answers the owner in a private chat: `/setup` (the walkthrough, resumable), `/bind` (link or re-link the account), `/topics` (list, add, edit, merge, remove, each with its own channel), `/llm` (choose and test the language model), `/settings` (digest hour, posting style, strictness, review day), `/stats`, `/review`, `/digest`, `/preview`, `/status` (is everything running, when was the last post ingested, when is the next digest), and `/pause` / `/resume` to stop publishing without stopping intake. Anyone other than the owner is ignored.

**Notifications.** The bot sends one line when a digest goes out ("ML & AI: 15 posts; Fintech: 9"), one message per proposal on review day, and a warning if intake stops for more than an hour or the account's session expires. Nothing else.

**Restarts and outages.** Posts are counted as published only after they actually appear, so a crash never causes a repeat and never loses a post that was due. After an outage the service picks up where it left off; posts that arrived while it was down are not fetched retroactively unless the user runs `backfill`, which is safe at any time. If a digest hour passes during an outage, that day's digest is sent at the next start.

**Logs** go to the system journal in plain lines a user can read: what was ingested, what was sorted where, what was posted, what failed and why.

## Safety and your account

The design goal is that running tg-curator is indistinguishable, from Telegram's side, from a person who reads a lot, keeps tidy folders and occasionally leaves a chat.

**What the account never does.** It never sends a message to anyone, never joins anything, never reacts, never marks chats as read, never changes a profile. Its only writes are into things the user owns: forwards and media copies into the user's own channels, creating a topic channel when the user asked for one, edits to the two folders the curator keeps, and the mutes, archives and leaves the user approved. Everything that looks like posting is done by the bot.

**Pacing.** Posting into topic channels is spaced out and stays far under Telegram's limits even on busy news days. History backfill reads chats one at a time with pauses. Approved leaves run no faster than one every half hour, with a small daily cap; if Telegram asks the account to slow down, the curator waits as long as it is told and continues, without the user noticing anything but a delay.

**Nothing irreversible without a tap.** Leaving is only ever proposed, never automatic, and shows a confirmation before it runs. Muting, archiving and folder moves are undoable. The curator never deletes messages anywhere, including in the topic channels.

**Where the data lives.** Everything stays on the user's server: the post archive, the statistics, the login session, the sorting models. No message text, no chat names and no identifiers are sent to any third party, and there is no telemetry. The one exception is chosen explicitly: with a frontier provider selected as the language model, the text of the posts picked for a digest, and of the example posts behind a proposed topic, is sent to that provider to be summarised, and nothing else. With no model connected the only outbound connections are to Telegram; with a self-hosted one, also to that server. The bot says this plainly at the moment of choosing.

**Credentials.** The account session, the settings file with the Telegram keys, and any provider key or self-hosted token are stored readable by the service user only, excluded from version control by default, and the documentation says plainly that whoever holds the session holds the account. When the account is bound or a key is entered through the bot, the login code, the two-factor password and the key are read from the chat and the messages containing them are deleted immediately; the code and the password are never stored. The bot obeys only the owner who claimed it with the one-time code at first start. Logging out from Telegram's devices list revokes the session instantly, and `/bind` sets it up again.

**Content restrictions respected.** Channels that forbid saving content are never forwarded from; the repost style links to the original instead. The topic channels are private to the user unless they choose to share them, and sharing a channel that reposts other people's content is the user's decision, not something the tool encourages.

## Shipping it to your own server

A stranger with a VPS and no Python experience should get from the repository's front page to their first digest in under an hour, and a developer should be able to read the whole thing in an afternoon.

**Install paths.** Two, both documented on the front page: `docker compose up` with a single compose file that holds the service and a volume for the data, and a plain `pip install tg-curator` for people who prefer running it directly, with the system service file included. Both use the same settings file and the same commands, so documentation never forks.

**First-run guidance.** The first start with an empty settings file does not crash; it prints the setup steps and stops. Each later step confirms itself ("logged in as …", "found 142 chats, 3 are output channels", "4 topics, all channels resolved"). The settings file ships fully commented with four example topics in Russian and English, so most users edit instead of write.

**Documentation**, short and task-shaped: Install, Set up, Tune, Manage subscriptions, Safety, FAQ. The FAQ answers the questions a new user predictably has: can my account get banned, does it send my messages anywhere, why is a post unsorted, why did a story not go out immediately, which language model to pick and what it costs, can I run it for two accounts, what happens if I stop it for a week.

**Upgrades.** A new version is a `pip install -U` or an image pull; the database migrates itself on start and settings files from older versions keep working with a one-line notice about any renamed key. Version history on the releases page says what changed for the user, not for the code.

**Language.** The interface, bot replies and documentation are in English; the sorting works for any language the model knows (about a hundred, including Russian and Uzbek). Translations of the bot's messages are a settings file others can contribute.

**Open-source hygiene.** A permissive licence, a contribution guide, an issue template that asks for the preview output rather than screenshots, and a changelog. The example settings, the Docker file and the service file are part of the repository, not of a wiki. No account, key or server of the author is referenced anywhere; nothing phones home.

## Not in the first release

One thing is left out on purpose, with a natural place to be added later.

- **Several accounts** in one installation. One account, one set of topics, one bot; a household or a team runs one copy each.
