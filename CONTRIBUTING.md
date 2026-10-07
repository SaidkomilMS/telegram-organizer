# Contributing

Thank you for helping. The core of tg-curator (the reading path in "Reading the code" below) is
meant to be read by one developer in an afternoon, so the rules are few: plain code over frameworks, small modules, type hints everywhere, docstrings that
say *why*, no dead code and no TODOs left behind.

## Getting started

```sh
git clone https://github.com/SaidkomilMS/telegram-organizer.git && cd telegram-organizer
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,postgres]"
pytest -q                     # the whole suite: no network, no Telegram, no model download
ruff check src tests
ruff format src tests
```

CI runs exactly these on Python 3.12 (see `.github/workflows/ci.yml`). Line length is 100.

To try the service against your own Telegram without downloading the model, set
`TG_CURATOR_FAKE_MODELS=1`: a deterministic stand-in replaces the similarity model (sorting
quality is meaningless then, everything else works). Use a separate home: `curator --home
/tmp/curator-dev run`.

## Layout

```
src/tg_curator/
  cli.py service.py control.py   the `curator` command, the running service, the local socket
  config.py data/                the settings file (template in data/settings.example.toml)
  domain.py contracts.py runtime.py   shared types, service interfaces, the wiring container
  telegram/                      the only code that talks to Telegram (user account, bot)
  pipeline/                      intake, backfill, the sorting engine, publisher, digest, preview
  ml/                            local models: similarity, categories, the per-user classifier
  llm/                           optional language models over plain HTTP
  topics/ subscriptions/         topics and learning; statistics, review, actions, folders
  bot/ locales/                  the bot conversation and every message it sends
  db/                            SQLAlchemy Core schema, migrations, the store
tests/                           one test file per module, plus fakes.py and conftest.py
docs/                            the user documentation
```

## The architecture in ten lines

1. One asyncio process (`service.py`) runs the account client, the bot and every loop.
2. Only `telegram/user_client.py` and `telegram/bot_client.py` import Telethon; everything else
   uses the protocols in `telegram/gateway.py`, which `tests/fakes.py` implements.
3. `pipeline/intake.py` turns new messages into candidates (albums, group conversation units).
4. `pipeline/engine.py` answers four questions per post: repeat? which topic? how strong? wait?
5. `pipeline/sorter.py` persists each decision and routes it; `pipeline/publisher.py` posts.
6. `pipeline/digest.py` ranks and sends the daily digest; posts count as sent only after Telegram
   returned a message id, and restarts reconcile half-done sends.
7. `ml/` holds the local models; `topics/learning.py` turns corrections into examples.
8. `subscriptions/` computes statistics and the weekly proposals and executes approved ones.
9. `bot/core.py` routes the owner's commands and buttons to the feature modules in `bot/`.
10. Every service takes the `Runtime` (`runtime.py`); `contracts.py` documents each interface.

Two rules are structural and tested: the account never sends, joins, reacts, marks as read or
edits a profile (`tests/test_user_client_hygiene.py` greps for the forbidden calls), and only
`ml/` downloads models and only `llm/` talks to language models; nothing else connects anywhere.

## Reading the code

The package is about 23,500 lines, which no one reads properly in an afternoon. The core path
below is about 7,000 lines (comments and docstrings included): enough to follow how a post
becomes a decision and reaches Telegram, read as a skim in an afternoon. Take it in this order:

1. `domain.py` (~470 lines) and `contracts.py` (~640): the types and the service interfaces.
2. `runtime.py` (~180) and `telegram/gateway.py` (~470): how services are wired, and the
   Telegram protocols everything else uses.
3. `config.py` (~810): skim it; look keys up as you meet them.
4. `pipeline/intake.py` (~620), then `pipeline/engine.py` (~530), then `pipeline/sorter.py`
   (~470): how a new message becomes a candidate, a decision and a routed post.
5. `pipeline/publisher.py` (~1,140), then `pipeline/digest.py` (~660): how decisions reach
   Telegram, and why a post counts as sent only after Telegram returned a message id.
6. `service.py` (~1,030): how the loops start, run and stop.

Safe to skip on a first read, and to open when you change them:

- `telegram/user_client.py` and `telegram/bot_client.py`: the Telethon adapters behind the
  gateway protocols (`tests/fakes.py` implements the same protocols).
- `db/store.py`, `db/schema.py` and `db/migrations.py`: the storage behind the store's methods.
- `llm/`: optional language models; off unless configured.
- `ml/`: the local models; treat them as black boxes behind `contracts.py`.
- `bot/*`: one module per command or button flow, routed by `bot/core.py`.
- `topics/`, `subscriptions/`, `pipeline/backfill.py`, `pipeline/preview.py`, `pipeline/render.py`,
  `control.py` and `cli.py`: features and entry points built on the core above.

## Tests

- `pytest -q` runs everything that needs no network. Use the fixtures in `tests/conftest.py`
  (`rt`, `store`, `clock`, `user_gw`, `bot_gw`, …) and the fakes in `tests/fakes.py`;
  `FakeBotGateway.say()` and `.press()` drive the bot as the owner would.
- **The real models**: tests marked `models` load the downloaded similarity model and score it
  against an evaluation set of posts. They are skipped unless you set:

  ```sh
  TG_CURATOR_TEST_MODELS=1 \
  TG_CURATOR_EVAL_DIR=/path/to/eval \
  HF_HOME=/path/to/hf-cache \
  pytest -q -m models
  ```

  `HF_HOME` is where the model is (or will be) downloaded; `TG_CURATOR_EVAL_DIR` holds the
  evaluation files the tests name (`posts.json`, `user_examples.json` and the training caches).
- **PostgreSQL**: the store and migration tests run a second time against PostgreSQL when
  `TG_CURATOR_TEST_PG_URL` points at a throwaway database:

  ```sh
  docker run --rm -d -p 5432:5432 -e POSTGRES_PASSWORD=test --name pg postgres:16
  TG_CURATOR_TEST_PG_URL=postgresql+asyncpg://postgres:test@localhost/postgres pytest -q tests/test_store.py tests/test_migrations.py
  ```

## Adding a language

Every message the bot sends lives in `src/tg_curator/locales/en/*.toml`, one file per module,
flat `key = "text with {placeholders}"`, HTML allowed. To translate:

1. Copy `locales/en` to `locales/<code>` (for example `locales/uz`).
2. Translate the values. Keep the keys and the `{placeholders}` exactly; leave out any key you
   do not translate and the English text is used.
3. Set `language = "<code>"` under `[general]`, restart, and walk through `/help`, `/setup` and
   `/settings` to see your text in place. A broken file is logged and English is used instead.

Users who only want to reword a few messages can put those keys into `messages.toml` in the
curator's home instead. The documentation stays in English.

## Changing behaviour

- Settings: add the key with its default and a comment that says which symptom to fix by moving
  it in which direction (`data/settings.example.toml`, `config.py`). A renamed key goes into
  `config.RENAMED_KEYS` so old files keep working.
- Database: add a migration to `db/migrations.py`; never edit an applied one.
- User-visible text: in the catalogue, never inline.
- Note user-visible changes in `CHANGELOG.md`, written for users.

## Reporting bugs

Use the issue template. It asks for `curator --version` and the relevant lines of
`curator preview` instead of screenshots, because those show what the curator decided and why.

## Credits

The bundled topic model was trained on MN-DS (CC BY 4.0, doi:10.5281/zenodo.7394851), uz-news
(CC BY 4.0, doi:10.5281/zenodo.7677431), the HuffPost News Category Dataset (CC BY 4.0) and the
lenta.ru news archive (no licence stated; credited). The similarity model is
intfloat/multilingual-e5-small (MIT). Keep these credits when you retrain or replace the model.
