"""Shared fixtures and test configuration.

FIXTURE SCOPE
-------------
Loading the index and the classifier takes ~2s each. Module-scoped fixtures
keep the full suite under five minutes; function scope would push it past
twenty and nobody would run it.

SKIP POLICY
-----------
Tests that need built artifacts skip rather than fail when those artifacts are
absent, so a fresh clone reports "N skipped" instead of a wall of red. The
artifacts themselves are checked by `test_data.py`, which fails loudly if the
build is broken.
"""
from __future__ import annotations

import pytest

from src.config.settings import settings

# ---------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------
HAS_INDEX = (settings.index_dir / "vectors.npy").exists()
HAS_MODEL = (settings.root / "models" / "intent_classifier.joblib").exists()
HAS_DB = settings.db_path.exists()
HAS_CORPUS = settings.documents_dir.exists() and any(
    settings.documents_dir.rglob("*.pdf"))
HAS_SHOTS = (settings.eval_dir / "screenshots" / "manifest.json").exists()

try:
    import shutil

    import pytesseract  # noqa: F401

    HAS_OCR = shutil.which("tesseract") is not None
except ImportError:
    HAS_OCR = False


needs_index = pytest.mark.skipif(
    not HAS_INDEX, reason="run: python scripts/build_index.py")
needs_model = pytest.mark.skipif(
    not HAS_MODEL, reason="run: python scripts/train_intent_classifier.py")
needs_db = pytest.mark.skipif(
    not HAS_DB, reason="run: python scripts/setup_database.py")
needs_shots = pytest.mark.skipif(
    not HAS_SHOTS, reason="run: python scripts/data_generation/gen_screenshots.py")
needs_ocr = pytest.mark.skipif(
    not HAS_OCR, reason="tesseract-ocr is not installed")


# ---------------------------------------------------------------------
# Expensive shared objects
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def understander():
    if not HAS_MODEL:
        pytest.skip("classifier not trained")
    from src.understanding.pipeline import UnderstandingPipeline

    return UnderstandingPipeline.load()


@pytest.fixture(scope="session")
def retriever():
    if not HAS_INDEX:
        pytest.skip("index not built")
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.retriever import Retriever
    from src.knowledge.vector_store import VectorStore

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    return Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w",
                     top_k=5)


@pytest.fixture(scope="session")
def agent():
    if not (HAS_INDEX and HAS_MODEL and HAS_DB):
        pytest.skip("required artifacts missing")
    from src.agent.loop import SupportAgent

    return SupportAgent()


@pytest.fixture(scope="session")
def pipeline():
    if not HAS_INDEX:
        pytest.skip("index not built")
    from src.rag.generator import build_pipeline

    return build_pipeline()


@pytest.fixture(scope="session")
def corpus():
    if not HAS_CORPUS:
        pytest.skip("corpus missing")
    from src.knowledge.loader import load_corpus

    return load_corpus()


@pytest.fixture(scope="session")
def chunks(corpus):
    from src.knowledge.chunker import build_chunks

    return build_chunks(corpus, strategy="section", max_tokens=200)


# ---------------------------------------------------------------------
# Adversarial and malformed inputs, shared across suites
# ---------------------------------------------------------------------
MALFORMED_TEXT = [
    "",                       # empty
    "   ",                    # whitespace only
    "?",                      # single character
    "a" * 5000,               # very long
    "\n\n\n",                 # newlines only
    "🎧🔥💀" * 50,             # emoji only
    "\x00\x01\x02",           # control characters
    "SELECT * FROM orders;",  # SQL-shaped
    "<script>alert(1)</script>",
    "मेरा ऑर्डर कहाँ है",       # non-Latin script
    "order " * 400,           # extreme repetition
]


@pytest.fixture(params=MALFORMED_TEXT, ids=lambda v: repr(v[:18]))
def malformed_text(request):
    return request.param


@pytest.fixture
def bad_image_bytes():
    return {
        "empty": b"",
        "tiny": b"abc",
        "not_an_image": b"this is plainly not an image file at all",
        "truncated_png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,
        "oversized": b"x" * (11 * 1024 * 1024),
    }
