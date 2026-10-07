# Install

Two ways to run tg-curator: Docker Compose, or a plain `pip install` with a systemd service.
Both read the same settings file and accept the same `curator` commands, so the rest of the
documentation applies to either. Choose Docker if the server already has it; choose pip if you
would rather not run Docker.

Either way you need:

- a Linux server with 2 CPU cores and 4 GB of memory (no GPU). The similarity model uses
  about 700 MB of memory; the first start downloads it once from huggingface.co (about 150 MB);
- a bot token from [@BotFather](https://t.me/BotFather): send it `/newbot`, pick a name, copy
  the token it answers with;
- `api_id` and `api_hash` from [my.telegram.org](https://my.telegram.org): log in with your phone
  number, open "API development tools", create an application (any name and short name), and
  copy the two values. Every Telegram client app needs these; they identify the app, not you,
  and they never leave your server.

## Path 1: Docker Compose

You need Docker with the Compose plugin (`docker compose version` prints a version).

```sh
git clone https://github.com/SaidkomilMS/telegram-organizer.git
cd telegram-organizer
cp .env.example .env
nano .env                   # fill in TG_CURATOR_API_ID, TG_CURATOR_API_HASH, TG_CURATOR_BOT_TOKEN
docker compose up -d
docker compose logs -f      # wait for the claim code, then press Ctrl+C
```

What this does:

- `docker-compose.yml` runs one service, `tg-curator`, with a named volume `curator-data`
  mounted at `/data`. Everything the curator keeps lives there: `settings.toml`, the database,
  the account session and the downloaded model. Nothing else on the host is touched.
- On the first start the container writes `/data/settings.toml` from the commented template,
  copies the three values from `.env` into it and goes on to print the claim code.
- From then on **the settings file is the source of truth**: `.env` only fills values that are
  still blank. To correct a value that is already in the file, edit the file (below).
- Optional: `TG_CURATOR_TIMEZONE=Asia/Tashkent` (any IANA name) in `.env` sets `timezone` under
  `[general]`, the zone the digest and review hours are in, while the file still says `"UTC"`.
  You can also pick it later in `/settings` → Timezone, or at the last setup step.
- The container never crash-loops; it prints what is wrong and waits:
  - a value is missing ("Next steps", or "… is empty in /data/settings.toml", then "Waiting
    for /data/settings.toml to change"): put it into `.env` and run `docker compose up -d`
    (a plain `docker compose restart` does not re-read `.env`), or edit the settings file;
  - Telegram refuses a value that is already in the file ("… was refused by Telegram"): edit
    the settings file (below). The container notices the change and starts again by itself.
- The container runs as an unprivileged user (uid 1000), and Docker restarts it after a crash or
  a reboot (`restart: unless-stopped`).

Running commands: every `curator …` command in these docs works inside the container. Prefix
it with `docker compose exec tg-curator`, run from the directory holding the compose file:

```sh
docker compose exec tg-curator curator stats
docker compose exec tg-curator curator preview
```

Editing the settings file: the bot's `/settings` and `/topics` edit it for you and keep your
comments. To edit it by hand, copy it out, edit, copy it back and restart:

```sh
docker compose exec tg-curator cat /data/settings.toml > settings.toml
nano settings.toml
docker compose exec -T tg-curator sh -c 'cat > /data/settings.toml' < settings.toml
docker compose restart
rm settings.toml            # the copy holds your keys
```

A language model running on the same machine (Ollama, LM Studio, …) is reachable from the
container as `http://host.docker.internal:11434/v1` (the compose file maps that name to the
host). Make the server listen on more than `127.0.0.1` for this to work, e.g. `OLLAMA_HOST=0.0.0.0`.

Prefer a folder on the host to a named volume? Replace the volume line with a bind mount and
run the container as your own user, so the files stay readable by you:

```yaml
    user: "${UID}:${GID}"
    volumes:
      - ./data:/data
```

(`export UID GID=$(id -g)` before `docker compose up`, and `mkdir data` first.)

## Path 2: pip and systemd

You need Python 3.12 or newer. Ubuntu 24.04 and Debian 13 have it (`python3 --version`); on
older systems use Docker instead.

```sh
sudo apt install python3-venv git                   # Debian/Ubuntu: the venv module and git
sudo useradd --system --home-dir /var/lib/tg-curator --shell /usr/sbin/nologin tg-curator
sudo python3 -m venv /opt/tg-curator
sudo /opt/tg-curator/bin/pip install git+https://github.com/SaidkomilMS/telegram-organizer.git
sudo ln -s /opt/tg-curator/bin/curator /usr/local/bin/curator
```

This installs the program into its own folder (`/opt/tg-curator`), makes the `curator` command
available, and creates a system user `tg-curator` that owns the data. Then the service, in two
lines:

```sh
curator --service-file | sudo tee /etc/systemd/system/tg-curator.service
sudo systemctl daemon-reload && sudo systemctl enable --now tg-curator
```

`curator --service-file` prints the systemd unit shipped with the package. It runs
`curator run` as `tg-curator`, keeps everything in `/var/lib/tg-curator`, restarts the service
after a failure, and does not restart it when the settings are incomplete (it would only fail
again). The first start writes `/var/lib/tg-curator/settings.toml` from the template and stops.
Fill in the three values (and, while you are there, `timezone` under `[general]`: the IANA zone
such as `"Asia/Tashkent"` that the digest and review hours are in; it is `"UTC"` until you change
it here or in `/settings` → Timezone):

```sh
sudo -u tg-curator nano /var/lib/tg-curator/settings.toml   # api_id, api_hash, bot_token
sudo systemctl restart tg-curator
journalctl -u tg-curator -f                                  # the claim code, then Ctrl+C
```

Running commands: commands must run as the service user with the service's home, so they can
talk to the running service. Add this line to your `~/.bashrc` once:

```sh
alias curator='sudo -u tg-curator curator --home /var/lib/tg-curator'
```

and every `curator …` command in these docs works as written.

The service logs to the system journal in plain lines: what was ingested, sorted where, posted,
what failed and why. `journalctl -u tg-curator --since today` shows today's.

### Without root

`pip install git+https://github.com/SaidkomilMS/telegram-organizer.git` into any virtual environment works as well; the data then lives in
`~/.tg-curator` (or wherever `--home` / `TG_CURATOR_HOME` points). `curator run` starts the
service in the foreground.

## PostgreSQL (optional)

The default database is one SQLite file next to the settings. If you already run PostgreSQL,
install the extra (`pip install "tg-curator[postgres] @ git+https://github.com/SaidkomilMS/telegram-organizer.git"`) and set

```toml
[storage]
database_url = "postgresql+asyncpg://user:password@host/dbname"
```

(or `TG_CURATOR_DATABASE_URL` in `.env`). The Docker image already includes the driver, so with
Docker setting that variable is all it takes. The tables are created on the first start.

## Upgrading

- pip: `sudo /opt/tg-curator/bin/pip install -U git+https://github.com/SaidkomilMS/telegram-organizer.git`, then `sudo systemctl restart tg-curator`.
- Docker: `git pull && docker compose up -d --build` (the image is built from your checkout).

The database migrates itself on start. Settings files from older versions keep working: if a key
was renamed, it is moved to its new name and the log says so in one line
(`settings: key … was renamed to …; moved`). What changed for you is in
[CHANGELOG.md](../CHANGELOG.md).

## Where things are

| file | what it is |
| --- | --- |
| `settings.toml` | the one settings file, fully commented |
| `curator.db` | the database (posts, statistics, decisions) |
| `user.session` | your account's login; whoever holds it holds the account |
| `bot.session` | the bot's login |
| `models/` | the downloaded similarity model |
| `messages.toml` | optional: your own wording or a translation of the bot's messages |

The home is `/data` in Docker, `/var/lib/tg-curator` with the systemd service, and
`~/.tg-curator` otherwise. The folder is readable only by the service user, and the session,
settings and database files only by that user (see [Safety](safety.md)).

Next: [Set up](setup.md).
