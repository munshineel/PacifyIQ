"""Embedding backends.

Two implementations behind one interface:

- `tfidf_svd` (default): TF-IDF followed by truncated SVD, i.e. classic Latent
  Semantic Analysis. Pure scikit-learn, no network, no `torch`, deterministic,
  and it fits the deployment budget. It is a genuine baseline, not a stub.
- `groq`: the hosted `nomic-embed-text-v1_5` endpoint. Better semantics, but it
  needs a key and a network round-trip per query.

Why a local default. The corpus is 16,208 words. At that scale an API call per
query buys real but modest quality for a hard dependency on a third party, and
it prevents the evaluation suite from running offline or in CI. The retrieval
evaluation therefore runs on `tfidf_svd`, and the Groq backend is available for
a like-for-like comparison whenever a key is present.

Both return L2-normalised vectors so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.config.settings import settings


class Embedder(ABC):
    """Common interface. `fit` is a no-op for API backends."""

    name: str
    dim: int

    @abstractmethod
    def fit(self, texts: list[str]) -> "Embedder":
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised vectors."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


# =====================================================================
# Local: TF-IDF + SVD (Latent Semantic Analysis)
# =====================================================================

class TfidfSvdEmbedder(Embedder):
    """LSA embeddings. Trained on the corpus itself.

    Character n-grams are included for the same reason they helped the intent
    classifier: error codes like `ERR-DP-0x004` fragment badly under word
    tokenisation, and queries often contain terms the corpus phrases differently.
    """

    name = "tfidf_svd"

    def __init__(self, dim: int = 192, use_char: bool = True, seed: int = 42,
                 max_word_features: int = 4000, max_char_features: int = 6000):
        self.dim = dim
        self.use_char = use_char
        self.seed = seed
        # Uncapped char n-grams over this corpus produce a ~35 MB pickle, which
        # would dominate the deployment budget. Capping costs almost nothing in
        # quality and brings the artifact to a few MB.
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self._vec = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            max_features=self.max_word_features,
            sublinear_tf=True, stop_words="english",
            token_pattern=r"[a-zA-Z0-9#\-]+",
        )
        if self.use_char:
            char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                max_features=self.max_char_features, sublinear_tf=True
            )
            self._vec = FeatureUnion(
                [("word", word), ("char", char)],
                transformer_weights={"word": 1.0, "char": 0.5},
            )
        else:
            self._vec = word

        X = self._vec.fit_transform(texts)
        n_comp = min(self.dim, X.shape[1] - 1, max(len(texts) - 1, 2))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        self._svd.fit(X)
        # float64 components double the artifact size for no measurable gain
        self._svd.components_ = self._svd.components_.astype(np.float32)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before encode()")
        return self._normalize(self._svd.transform(self._vec.transform(texts)).astype(np.float32))

    @property
    def explained_variance(self) -> float:
        return float(self._svd.explained_variance_ratio_.sum()) if self._fitted else 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vec": self._vec, "svd": self._svd, "dim": self.dim,
                 "use_char": self.use_char, "seed": self.seed,
                 "max_word_features": self.max_word_features,
                 "max_char_features": self.max_char_features},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        with open(path, "rb") as f:
            d = pickle.load(f)
        e = cls(dim=d["dim"], use_char=d["use_char"], seed=d["seed"],
                max_word_features=d.get("max_word_features", 8000),
                max_char_features=d.get("max_char_features", 20000))
        e._vec, e._svd, e._fitted = d["vec"], d["svd"], True
        return e


# =====================================================================
# API: Groq hosted embeddings
# =====================================================================

class GroqEmbedder(Embedder):
    """Hosted `nomic-embed-text-v1_5`.

    Responses are cached on disk keyed by text hash, so re-running the
    evaluation suite does not re-bill or re-wait for identical inputs.
    """

    name = "groq"

    def __init__(self, model: str | None = None, cache_dir: Path | None = None,
                 batch_size: int = 64):
        self.model = model or settings.embedding_model
        self.batch_size = batch_size
        self.dim = 768
        self.cache_dir = Path(cache_dir or settings.data_dir / "cache" / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.groq_api_key:
                raise RuntimeError(
                    "PACIFYIQ_GROQ_API_KEY is not set. Use the tfidf_svd backend "
                    "for offline work: build_index.py --backend tfidf_svd"
                )
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def fit(self, texts: list[str]) -> "GroqEmbedder":
        return self  # nothing to fit

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.model}||{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                out[i] = np.array(json.loads(p.read_text())["embedding"], dtype=np.float32)
            else:
                pending.append((i, t))

        if pending:
            client = self._get_client()
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start: start + self.batch_size]
                resp = client.embeddings.create(
                    model=self.model, input=[t for _, t in batch]
                )
                for (idx, text), item in zip(batch, resp.data):
                    vec = np.array(item.embedding, dtype=np.float32)
                    out[idx] = vec
                    self._cache_path(text).write_text(
                        json.dumps({"model": self.model, "embedding": vec.tolist()})
                    )

        arr = np.vstack([v for v in out if v is not None]).astype(np.float32)
        self.dim = arr.shape[1]
        return self._normalize(arr)


# =====================================================================
BACKENDS = {"tfidf_svd": TfidfSvdEmbedder, "groq": GroqEmbedder}


def get_embedder(backend: str = "tfidf_svd", **kwargs) -> Embedder:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {list(BACKENDS)}")
    return BACKENDS[backend](**kwargs)


if __name__ == "__main__":
    from src.knowledge.chunker import build_chunks
    from src.knowledge.loader import load_corpus

    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    texts = [c.text for c in chunks]

    e = get_embedder("tfidf_svd", dim=256).fit(texts)
    V = e.encode(texts)
    print(f"backend            {e.name}")
    print(f"chunks             {len(texts)}")
    print(f"dimension          {e.dim}")
    print(f"explained variance {e.explained_variance:.3f}")
    print(f"matrix             {V.shape}, {V.nbytes / 1024:.1f} KB")
    print(f"norms              min {np.linalg.norm(V, axis=1).min():.4f} "
          f"max {np.linalg.norm(V, axis=1).max():.4f}")

    q = e.encode_one("how long do I have to return an opened laptop")
    sims = V @ q
    print("\ntop 3 for 'how long do I have to return an opened laptop':")
    for i in np.argsort(-sims)[:3]:
        print(f"  {sims[i]:.3f}  {chunks[i].citation:32s} {chunks[i].preview(64)}")
