# Safety

The goal is that running tg-curator looks, from Telegram's side, like a person who reads a lot,
keeps tidy folders and occasionally leaves a chat. This page says exactly what the account does,
what leaves your server, and how to keep the login safe.

## What the account never does

Your account **never** sends a message to anyone, never joins a chat, never reacts, never marks
chats as read, and never changes your profile. The code that talks to Telegram for your account
has no way to do these things, and a test fails the build if one is ever added.

Its only writes are into things you own or chose:

- copying media into your own channels (in repost style, through the private "tg-curator media"
  channel) or forwarding into them (forward style);
- creating a topic channel when you asked for one, and making your bot an admin there with
  exactly two rights: post messages and edit messages;
- editing the two folders the curator keeps ("Curated", "Low signal"), never yours;
- the mutes, archives and leaves **you approved**, one tap each.

Everything that looks like posting is done by the bot, so your account never looks like a bot.
The curator never deletes a message anywhere, including in your topic channels: a post moved to
another topic leaves a one-line stub behind. The only messages it deletes are your own, in your
private chat with the bot: the ones holding a login code, a password or an API key.

The account also never mutes, archives or leaves one of its own topic channels, and never leaves
a chat you created.

## Pacing

- Posting into topic channels is spaced out (at least 20 seconds between two posts into one
  channel, at most 15 writes a minute per channel, 3 seconds between any two posts), far under
  Telegram's limits on the busiest news days.
- Reading history for a preview or backfill goes one chat at a time with pauses.
- Approved leaves run no faster than one every 30 minutes and at most 3 a day.
- When Telegram asks the account to slow down, the curator waits as long as it is told and
  continues. You notice only a delay.

## What leaves your server

Everything stays on your server: the post archive, the statistics, the account's login, the
sorting models. There is no telemetry and nothing phones home. The outbound connections are:

1. **Telegram**, always.
2. **huggingface.co, once, at the first start**: the similarity model (about 150 MB) is downloaded
   from the public model hub with telemetry turned off. Nothing about you is sent. After that the
   curator works offline from the downloaded copy and never contacts huggingface.co again.
3. **The language model you chose**, only if you chose one:
   - *None* (default): nothing else.
   - *Self-hosted*: your own server at the address you gave.
   - *Provider*: that provider's API.

What a language model receives, and nothing more: the text of the posts picked for a digest (one
at a time, to write a one-line summary), the example posts behind a proposed new topic (to name
it), one post when you test the connection in `/llm` (the bot names which), and, only if you turn
on the second opinion, the text of borderline posts. Never chat names, user names or identifiers.
With a provider, its own privacy terms apply to that text; the bot shows this at the moment you
choose. A monthly spending cap can be set.

## Credentials

The curator's home (see [Install](install.md#where-things-are)) holds:

| file | what it is |
| --- | --- |
| `user.session` | your account's login. **Whoever holds this file holds your account**: they can read all your chats and send messages as you. |
| `bot.session` | the bot's login |
| `settings.toml` | your `api_id`, `api_hash`, bot token, and any provider key or self-hosted token |
| `curator.db` | the post archive and statistics |

The home folder is created readable only by the service user (`0700`) and these files only by
that user (`0600`), and they are re-checked at every save. They are listed in `.gitignore` and
`.dockerignore`, so they never end up in a repository or an image. Do not put the home folder in
a backup that other people can read, and do not paste `settings.toml` into an issue.

When you bind the account or enter a key in the bot, the messages holding the login code, the
two-step password and the key are deleted right after they are read. The code and the password
are never stored anywhere, not even in the log. Logs never contain keys, tokens or the session,
and show phone numbers masked (`+9989***12`).

The bot obeys only the owner who claimed it with the one-time code printed at the first start.
Everyone else who writes to it is ignored.

## Sessions

The account's login appears in Telegram under **Settings → Devices** as "tg-curator".

- **To cut the curator off at once**, terminate that session there. The curator notices on its
  next request, stops reading, keeps posting what it can with the bot, and tells you; `/status`
  says "Account: session lost — run /bind". `/bind` sets it up again.
- **Never copy `user.session` to a second machine**, and never run two copies of the curator on
  the same session. Telegram sees the same login used twice at once (`AUTH_KEY_DUPLICATED`) and
  terminates it for both. To move the curator, stop the old one, set up the new one and run
  `/bind` there. Command-line commands are safe while the service runs: they hand the work to
  the running service instead of opening the session a second time.
- **Inactive sessions expire.** Telegram ends sessions that have not been used for a while. The
  period is under Settings → Devices, "Automatically terminate old sessions" (usually 6 months;
  it can be as short as a week). A curator that runs continuously is always active and never
  expires. One that was stopped for longer than that period comes back with "session lost", and
  `/bind` fixes it.

## Content restrictions

Channels that forbid saving or forwarding content are never forwarded from and their media is
never copied: the post keeps its text and a link to the original. Your topic channels are private
unless you choose to share them; sharing a channel that reposts other people's content is your
decision, not something the curator encourages.

## Credits

The similarity model is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT licence), downloaded at the first start.

The topic model shipped inside the package (a small file, no download) was trained on these
public datasets, which we gratefully credit:

- **MN-DS**, a multilingual news dataset with IPTC Media Topics labels, CC BY 4.0,
  [doi:10.5281/zenodo.7394851](https://doi.org/10.5281/zenodo.7394851);
- **uz-news** (Uzbek text classification dataset), CC BY 4.0,
  [doi:10.5281/zenodo.7677431](https://doi.org/10.5281/zenodo.7677431);
- **News Category Dataset** (HuffPost headlines), CC BY 4.0;
- the **lenta.ru** news archive (via the `data-silence/lenta.ru_2-extended` collection; no
  licence stated, credited here).
