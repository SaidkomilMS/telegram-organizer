"""Which model files exist, where they live, and the one-time download (DESIGN §10, §13, §14.16).

The embedder is pinned to a repository revision so every installation runs the same weights
and the duplicate thresholds measured in the research stay valid. The download happens once,
with Hugging Face telemetry off; after it the process goes ``HF_HUB_OFFLINE`` so nothing ever
asks huggingface.co again — not even a revision check on later starts. The ONNX session and
the tokenizer (about 700 MB of memory together) are created once per process and shared.
``TG_CURATOR_FAKE_MODELS=1`` swaps in the deterministic ``HashEmbedder`` for tests and
previews on a machine without the models.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from tg_curator.config import Settings
from tg_curator.contracts import Embedder
from tg_curator.ml.base_model import BaseModel, load_for
from tg_curator.ml.classifier import LocalTopicClassifier
from tg_curator.ml.embedder import HashEmbedder, OnnxEmbedder

log = logging.getLogger(__name__)

EMBEDDER_REPO = "intfloat/multilingual-e5-small"
EMBEDDER_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDER_LICENCE = "MIT"
EMBEDDER_ID = f"{EMBEDDER_REPO}@{EMBEDDER_REVISION[:12]}"
"""What ``kv ml.embedder_id`` is compared against; stored embeddings are recomputed when it
changes. The int8 and fp32 files are the same weights (cosine >= 0.99 between their outputs),
so switching ``[ml].embedder_file`` does not invalidate stored embeddings."""
EMBEDDER_INT8_FILE = "onnx/model_qint8_avx512_vnni.onnx"
EMBEDDER_FP32_FILE = "onnx/model.onnx"
EMBEDDER_FILES = ("tokenizer.json", "config.json")
"""Always needed, next to the one ONNX file chosen by ``[ml].embedder_file``."""
TOKENIZER_FILE = "tokenizer.json"
FAKE_MODELS_ENV = "TG_CURATOR_FAKE_MODELS"
FILE_SIZES_MB = {EMBEDDER_INT8_FILE: 118, EMBEDDER_FP32_FILE: 470, "tokenizer.json": 17}

_embedder: Embedder | None = None


def fake_models() -> bool:
    return os.environ.get(FAKE_MODELS_ENV, "").strip() in {"1", "true", "yes"}


def models_dir(home: Path) -> Path:
    """The Hugging Face cache root: ``HF_HOME`` when set (the Docker image sets it to
    ``/data/models``), else ``<home>/models``."""
    override = os.environ.get("HF_HOME")
    return Path(override) if override else home / "models"


def _cache_dir(home: Path) -> Path:
    return models_dir(home) / "hub"


def _needed_files(settings: Settings | None) -> tuple[str, ...]:
    chosen = settings.ml.embedder_file if settings is not None else EMBEDDER_INT8_FILE
    return (*EMBEDDER_FILES, chosen)


def _cached(home: Path, filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(
        EMBEDDER_REPO, filename, cache_dir=_cache_dir(home), revision=EMBEDDER_REVISION
    )
    return Path(hit) if isinstance(hit, str) else None


def ensure_models(
    home: Path,
    progress: Callable[[str], None] | None,
    *,
    settings: Settings | None = None,
) -> None:
    """Make sure the pinned model files are on disk, downloading them once.

    ``progress`` gets one line per file ("downloading tokenizer.json (17 MB)…"). Once every
    file is present the process is switched to ``HF_HUB_OFFLINE=1``. A fake-models process
    never touches the hub.
    """
    if fake_models():
        return
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS; the xet path can stall
    missing = [f for f in _needed_files(settings) if _cached(home, f) is None]
    if missing:
        from huggingface_hub import hf_hub_download

        os.environ.pop("HF_HUB_OFFLINE", None)
        _cache_dir(home).mkdir(parents=True, exist_ok=True)
        for filename in missing:
            size = FILE_SIZES_MB.get(filename)
            note = f" ({size} MB)" if size else ""
            if progress is not None:
                progress(f"downloading {filename}{note} from huggingface.co…")
            log.info("downloading %s %s from huggingface.co", EMBEDDER_REPO, filename)
            hf_hub_download(
                EMBEDDER_REPO,
                filename,
                revision=EMBEDDER_REVISION,
                cache_dir=_cache_dir(home),
            )
    os.environ["HF_HUB_OFFLINE"] = "1"


def make_embedder(home: Path, settings: Settings | None = None) -> Embedder:
    """The process-wide embedder: ``HashEmbedder`` under ``TG_CURATOR_FAKE_MODELS=1``, else
    the ONNX model loaded exactly once (``ensure_models`` must have run)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    if fake_models():
        _embedder = HashEmbedder()
        return _embedder
    threads = settings.ml.threads if settings is not None else 2
    model_file = _needed_files(settings)[-1]
    model_path = _cached(home, model_file)
    tokenizer_path = _cached(home, TOKENIZER_FILE)
    if model_path is None or tokenizer_path is None:
        raise FileNotFoundError(
            f"the model files of {EMBEDDER_REPO} are not in {_cache_dir(home)}: "
            "run the service once with network access so they can be downloaded"
        )
    _embedder = OnnxEmbedder(model_path, tokenizer_path, threads=threads, id=EMBEDDER_ID)
    log.info("loaded %s (%s, %d threads)", EMBEDDER_ID, model_file, threads)
    return _embedder


def make_classifier(embedder: Embedder) -> LocalTopicClassifier:
    """The classifier for ``embedder`` with the matching base model (``load_for``)."""
    return LocalTopicClassifier(embedder, load_for(embedder))


def make_base_model(embedder: Embedder) -> BaseModel:
    return load_for(embedder)
