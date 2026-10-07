# Changelog

What changed for you, newest first. Upgrading: see [docs/install.md](docs/install.md#upgrading);
the database migrates itself and older settings files keep working.

## 0.1.0 — first release

The first public version. With it you can:

- **Sort everything your channels and groups publish** into a private channel per topic. Strong
  stories, the ones several chats carry or that come from a source you trust, arrive within
  minutes; the rest wait for a daily digest of the top 15, ranked by how widely they spread and
  how much attention they got.
- **Never see the same story twice**: reposts, forwards, reworded copies and translations between
  Russian, Uzbek and English are recognised, and a story that spread shows `+N` instead.
- **Define topics without keyword lists**: pick one of 20 built-in categories, forward a few
  example posts, or name an example channel. Correct a wrong post with one tap and the classifier
  learns it on the spot.
- **Set everything up from the bot** in about fifteen minutes: claim the bot, bind your account,
  create topics and their channels, preview the last three days, and go live. Every step works
  from the command line too.
- **Clean up your subscriptions**: per-chat statistics, a weekly review that proposes folder
  moves, mutes, archives and leaves with the reason in plain words, carried out slowly and only
  after you approve. Mutes, archives and folder moves can be undone.
- **Get new topics proposed** when unsorted posts start to look alike, and merges when two topics
  keep competing for the same posts.
- **Optionally connect a language model** for one-line digest summaries and topic names:
  a self-hosted one (Ollama, vLLM, LM Studio, llama.cpp) or Anthropic, OpenAI, Google, Mistral
  or OpenRouter, with a monthly spending cap.
- **Run it on a small server** (2 cores, 4 GB, no GPU) with Docker Compose or `pip install`
  and a systemd service. All sorting is done locally; nothing is sent anywhere unless you connect
  a language model.
