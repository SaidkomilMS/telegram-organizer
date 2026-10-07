---
name: Bug report
about: Something did not work, or a post was sorted, ranked or posted wrongly
labels: bug
---

<!--
Please paste text, not screenshots: the text shows what the curator decided and why.
Before pasting, remove anything private (chat names, post texts you do not want public).
Never paste settings.toml, .env, any *.session file, keys, tokens or login codes.
-->

**What happened, and what did you expect?**


**Version**

Output of `curator --version` (with Docker: `docker compose exec tg-curator curator --version`):

```
```

**Install path:** Docker Compose / pip + systemd / other:

**The decisions around the problem**

Output of `curator preview` (add `--topic KEY` or `--days N` to narrow it down), only the lines
for the posts in question. Each line shows the post, its topic and confidence, whether it was a
repeat and of what, and whether it would have gone out immediately.

```
```

**Log lines**

The lines around the time it happened (`journalctl -u tg-curator --since "1 hour ago"` or
`docker compose logs --since 1h tg-curator`):

```
```

**Settings you changed from the defaults** (keys and values only, no secrets):

