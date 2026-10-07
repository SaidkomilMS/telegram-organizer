# tg-curator

You point tg-curator at your Telegram account, name the topics you care about, and from then on
everything your channels and groups publish is sorted for you: the important posts land in a
private channel per topic within minutes, the rest arrive once a day as a short ranked digest,
and the same story never reaches you twice. Over the following weeks it also tells you which
subscriptions actually contribute something, proposes what to mute, archive or leave, and does
only what you approve with a tap.

It runs on your own server and is open source. Sorting is done locally by small models that
never leave the server, so it works the same for Russian, Uzbek and English channels and costs
nothing per post. Written digest summaries are optional and use a language model you connect
(one you host yourself, or a provider you give a key for); without one, the digest shows the
first line of each post.

## What you need

- **Your Telegram account**, the one subscribed to the channels and groups you want curated.
  It only reads, copies media into your own channels and creates those channels when you ask.
  It never messages anyone.
- **A bot** made with [@BotFather](https://t.me/BotFather) (one minute): it does all the
  posting and walks you through setup.
- **Two values from [my.telegram.org](https://my.telegram.org)** ("API development tools"):
  `api_id` and `api_hash`. They stay on your server.
- **A small Linux server**: 2 CPU cores, 4 GB of memory, no GPU. The first start downloads the
  similarity model once (about 150 MB).

No AI provider account, no database to install (one file on disk; PostgreSQL is optional).

## Install

Pick one. Both use the same settings file and the same `curator` commands.

**Docker Compose** (recommended if you have Docker):

```sh
git clone https://github.com/SaidkomilMS/telegram-organizer.git && cd telegram-organizer
cp .env.example .env        # then fill in the three values
docker compose up -d
docker compose logs -f      # shows the one-time claim code
```

**pip and systemd**:

```sh
sudo apt install python3-venv git               # Debian/Ubuntu; needs Python 3.12 or newer
sudo useradd --system --home-dir /var/lib/tg-curator --shell /usr/sbin/nologin tg-curator
sudo python3 -m venv /opt/tg-curator && sudo /opt/tg-curator/bin/pip install git+https://github.com/SaidkomilMS/telegram-organizer.git
sudo ln -s /opt/tg-curator/bin/curator /usr/local/bin/curator
curator --service-file | sudo tee /etc/systemd/system/tg-curator.service
sudo systemctl daemon-reload && sudo systemctl enable --now tg-curator
# the first start writes the settings file and stops: fill in the three values, then
sudo -u tg-curator nano /var/lib/tg-curator/settings.toml   # api_id, api_hash, bot_token
sudo systemctl restart tg-curator
journalctl -u tg-curator -f                     # shows the one-time claim code
```

[docs/install.md](docs/install.md) walks through both paths line by line, including where the
three values go and how to upgrade.

## Set up (about fifteen minutes)

Everything happens in a private chat with your bot. Each step confirms itself or says exactly
what to fix, and `/setup` continues from wherever you stopped.

1. **Install** and start the service (above). It prints a one-time claim code.
2. **Claim the bot**: open it in Telegram and send `/start` followed by the code.
3. **Bind your account** with `/bind`: phone number, the login code (typed with spaces between
   the digits, e.g. `1 2 3 4 5`), and your two-step password if you have one. The bot deletes
   those messages right after reading them.
4. **Create topics** with `/topics add`: a name, then a built-in category, a few forwarded
   example posts or an example channel, then a private channel the bot creates for you.
5. **Connect a language model** with `/llm`, or skip it.
6. **Preview** with `/preview`: the last three days sorted, nothing posted. Fix wrong decisions
   with a tap.
7. **Go live** with `/go`.

Prefer a terminal? `curator login`, `curator topics`, `curator llm`, `curator preview` and
`curator run --live` do the same; see [docs/setup.md](docs/setup.md).

## Documentation

- [Install](docs/install.md): Docker or pip, the service, upgrades
- [Set up](docs/setup.md): the seven steps in the bot and on the command line
- [Tune](docs/tune.md): every setting, and which way to move it for which symptom
- [Manage subscriptions](docs/subscriptions.md): statistics, the weekly review, undo, folders
- [Safety](docs/safety.md): what the account never does, what leaves your server, credentials
- [FAQ](docs/faq.md): bans, privacy, unsorted posts, which language model, costs

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT, see [LICENSE](LICENSE). The bundled topic model was trained on public datasets credited in
[docs/safety.md](docs/safety.md#credits).
