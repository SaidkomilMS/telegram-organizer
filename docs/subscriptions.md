# Manage subscriptions

The curator knows, for every chat, how much of what it posts ever reached a topic. After a few
weeks that becomes a cleanup list, and the bot turns it into proposals you approve or reject one
tap at a time. Nothing happens to a chat until you tap **Approve**.

## Statistics

`/stats` in the bot, or `curator stats`, lists every chat over the last 30 days:

- **Volume**: how many messages the chat posted. Service messages (joins, pins, title changes)
  and your own messages in groups are not counted (your own posts in a channel are, and they are
  sorted like any other post); an album counts once. For a group this is every
  message, including the chatter that is never sorted, so a loud group shows its real size.
- **Signal**: the share of that volume that was sorted into one of your topics. Only original
  posts count, not repeats, and not posts you marked "Not for me".
- **Repeats** (duplicate share): how often the chat merely repeated something **another** chat
  had already posted. A chat repeating its own earlier post does not count.
- **Published**: how many of its posts actually reached a topic channel, in real time or in a
  digest.
- **Observed**: for how many days the curator has been watching it (at most 30).

A chat with 300 posts, 2% signal and 70% repeats is one you are paying attention to for nothing.

`curator stats --days 7` changes the window; `curator stats --all` and `/stats all` also list
chats you have left (their history is kept). Your topic channels and the staging channel are
never listed.

## The weekly review

Every Sunday at 11:00 (your timezone; `[review]` `weekday` and `hour`) the bot sends one message
per proposal, grouped in this order: chats to move into the "Low signal" folder, chats to mute,
chats to archive, chats to leave, then new topics and topic merges. Each names the chat and the
reason in plain words, for example:

> 41 posts in 30 days, 0 reached a topic, 85% were repeats of @kunuz

and has three buttons: **Approve**, **Skip**, **Never ask again**. At most 20 proposals are sent
per review (`max_proposals`); the last line says how many are held back for next week.

`/review` in the bot, or `curator review`, runs a review now. If the service was down at review
time, the review runs at the next start.

### What each level means

| proposal | what happens when you approve | undo |
| --- | --- | --- |
| **Low signal folder** | the chat is added to the curator's own "Low signal" folder | yes |
| **Mute** | notifications off for 30 days (`mute_days`) | yes |
| **Archive** | the chat is archived **and** muted (an archived chat that is not muted jumps back out at its next message) | yes, both together |
| **Leave** | the account leaves the chat, after a confirmation | no |

A chat gets at most one proposal per review: the strongest level whose rule holds. Leaving is
proposed only after 30 days of observation and only if **none** of the chat's own posts reached
a topic in that time; before 30 days the strongest proposal is a folder move or a mute. A chat
you created is never proposed for leaving. If you are an admin of the chat, the leave proposal
says so ("you are an admin; leaving drops that").

**Skip** means "not now": the proposal comes back in a later review if the numbers still
justify it. **Never ask again** marks the chat as kept; it is never proposed again.

### Leaving takes a confirmation

Approving a leave first asks "Leave …? This cannot be undone without a new invitation." with
**Yes, leave** and **Cancel**. Leaving a private group cannot be undone without a new invitation,
which is why it is the only action that is never proposed early and always confirmed.

### How fast approved actions run

Approved actions are carried out by your own account, slowly, so it never looks automated:

- folder moves, mutes and archives within minutes, 20 to 90 seconds apart;
- leaves no faster than one every 30 minutes and no more than 3 in any 24 hours
  (`leave_interval_minutes`, `leaves_per_day`).

If Telegram asks the account to slow down, the curator waits as long as it is told and the
proposal message says "Telegram asked to wait, retrying at 14:30".

Each proposal message updates itself to show the outcome: "Moved to “Low signal” ✓",
"Muted until Nov 4 ✓", "Archived and muted ✓", "Left ✓", or why it could not be done ("could not
leave: you created this chat").

### Undo

Mutes, archives and folder moves are reversible: tap **Undo** under the outcome, or reply
`/undo` to the proposal message. You can also undo by hand in Telegram; the curator does not
fight you. A leave cannot be undone.

### The rules and their settings

All in `[review]` of `settings.toml`. A chat with fewer than `folder_min_posts` messages in the
window gets no proposal at all: it has said too little to judge. Chats you marked "Never ask
again", your own channels and chats with an open proposal are skipped too.

| proposal | rule (defaults) | keys |
| --- | --- | --- |
| leave | observed ≥ 30 days and nothing reached a topic | `leave_min_days` |
| archive | observed ≥ 30 days and signal ≤ 2% | `archive_min_days`, `archive_max_signal` |
| mute | observed ≥ 14 days, signal ≤ 3% and repeats ≥ 60% | `mute_min_days`, `mute_max_signal`, `mute_min_duplicates` |
| folder | observed ≥ 7 days and signal ≤ 5% | `folder_min_days`, `folder_max_signal` |

Other `[review]` keys:

| key | default | |
| --- | --- | --- |
| `weekday`, `hour` | `"sunday"`, `11` | when the review is sent |
| `window_days` | `30` | the statistics and proposal window |
| `max_proposals` | `20` | proposals per review |
| `folder_min_posts` | `10` | minimum volume before a chat is judged |
| `mute_days` | `30` | how long an approved mute lasts |
| `leaves_per_day`, `leave_interval_minutes` | `3`, `30` | leave pacing |
| `cluster_min_posts`, `cluster_window_days` | `30`, `7` | new-topic proposals (below) |
| `cluster_tightness` | `0.6` | how alike unsorted posts must be to form a new topic. No topics are ever proposed: lower (0.5). Proposed topics look random: raise (0.7). |
| `merge_margin` | `0.3` | how much two topics must compete for the same posts before a merge is proposed |
| `auto_create_topics` | `false` | create proposed topics without asking (see below) |

## Folders

The curator can keep two Telegram folders of its own:

- **Curated** with your topic channels (`[folders] curated`);
- **Low signal** with the chats it flagged (`[folders] low_signal`). It is created at the first
  approved folder move and removed again when no chat is flagged any more.

It never touches folders you created and never reorders chats in your main list. Chats you add to
its folders by hand stay there. Names are set with `curated_name` and `low_signal_name` (at most
12 characters, Telegram's limit). A folder holds at most 100 chats.

Telegram allows 10 folders per account (30 with Premium). If yours are full, the curator turns
its folders off, logs it once, and `/status` says "Folders: off". Delete a folder you do not
need, then send `/reload`. Turning `low_signal` off also stops folder proposals.

## New topics, found for you

Unsorted posts are not thrown away. The curator keeps looking for groups of them that resemble
each other, and when a tight cluster has formed (30 or more posts within a week) the review
proposes a new topic: a name taken from the closest built-in category (or, with a language model
connected, a sharper name and a one-line description written from the examples) and five example
posts so you can judge it. The buttons:

- **Create** makes the topic and its private channel. The cluster's posts become its examples and
  the recently unsorted posts are sorted again (nothing old is posted). The next review starts
  with a line such as "Crypto & exchanges absorbed 112 of 430 unsorted posts since Oct 4".
- **Rename** asks for a name, then creates it.
- **Dismiss** skips it; it may come back if the cluster keeps growing.

With `auto_create_topics = true` the curator creates at most one such topic a day without asking
and tells you afterwards.

The same mechanism notices two topics that keep competing for the same posts, or that mean nearly
the same, and proposes merging them; **Merge** moves everything of one into the other.
