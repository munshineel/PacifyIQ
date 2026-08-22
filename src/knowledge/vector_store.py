"""Vector store.

Brute-force cosine similarity in NumPy, with an optional FAISS backend that is
verified to return identical results.

WHY BRUTE FORCE IS THE DEFAULT. EDA finding 7c measured the corpus at 16,208
words, which produces ~200 chunks at 200 tokens. Exact search over 200 vectors
takes well under a millisecond. An approximate index (HNSW) trades recall for
speed and has nothing to trade at this scale - it would be complexity added for
its own sake. The FAISS path exists so the comparison can be shown rather than
asserted, and so the store scales if the corpus grows.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.knowledge.chunker import Chunk


@dataclass
class SearchHit:
    """One retrieved chunk with its score."""

    chunk: Chunk
    score: float
    rank: int
    source: str = "dense"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "source": self.source,
            "chunk_id": self.chunk.chunk_id,
            "citation": self.chunk.citation,
            "doc": self.chunk.doc,
            "section": self.chunk.section,
            "text": self.chunk.text,
        }


class VectorStore:
    """Dense index over chunk embeddings."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, backend: str = "numpy"):
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.backend = backend
        self._faiss = None
        if backend == "faiss":
            self._build_faiss()

    # ---------------------------------------------------------------
    def _build_faiss(self) -> None:
        import faiss

        index = faiss.IndexFlatIP(self.vectors.shape[1])   # inner product == cosine
        index.add(self.vectors)                            # vectors are L2-normalised
        self._faiss = index

    # ---------------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[SearchHit]:
        """Cosine similarity search with optional metadata filtering.

        `where` is applied before ranking, so filtering to current-version
        documents cannot be defeated by an archived chunk scoring higher.
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()

        if where is not None:
            keep = [i for i, c in enumerate(self.chunks) if where(c)]
            if not keep:
                return []
            sims = self.vectors[keep] @ q
            order = np.argsort(-sims)[:top_k]
            return [
                SearchHit(self.chunks[keep[i]], float(sims[i]), rank + 1)
                for rank, i in enumerate(order)
            ]

        if self._faiss is not None:
            scores, idx = self._faiss.search(q.reshape(1, -1), min(top_k, len(self.chunks)))
            return [
                SearchHit(self.chunks[int(i)], float(s), rank + 1)
                for rank, (s, i) in enumerate(zip(scores[0], idx[0]))
                if i >= 0
            ]

        sims = self.vectors @ q
        order = np.argsort(-sims)[:top_k]
        return [
            SearchHit(self.chunks[int(i)], float(sims[int(i)]), rank + 1)
            for rank, i in enumerate(order)
        ]

    # ---------------------------------------------------------------
    def score_all(self, query_vec: np.ndarray) -> np.ndarray:
        return self.vectors @ np.asarray(query_vec, dtype=np.float32).ravel()

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        # human-readable copy, for inspection and diffing
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path, backend: str = "numpy") -> "VectorStore":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return cls(chunks, vectors, backend=backend)


def compare_backends(chunks: list[Chunk], vectors: np.ndarray, queries: np.ndarray,
                     top_k: int = 5) -> dict[str, Any]:
    """Verify FAISS returns what brute force returns, and time both."""
    import time

    numpy_store = VectorStore(chunks, vectors, backend="numpy")
    try:
        faiss_store = VectorStore(chunks, vectors, backend="faiss")
    except ImportError:
        return {"faiss_available": False}

    t0 = time.perf_counter()
    np_results = [numpy_store.search(q, top_k) for q in queries]
    t_np = (time.perf_counter() - t0) / len(queries) * 1000

    t0 = time.perf_counter()
    fa_results = [faiss_store.search(q, top_k) for q in queries]
    t_fa = (time.perf_counter() - t0) / len(queries) * 1000

    agree = sum(
        [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]
        for a, b in zip(np_results, fa_results)
    )
    return {
        "faiss_available": True,
        "n_vectors": len(chunks),
        "n_queries": len(queries),
        "numpy_ms_per_query": round(t_np, 3),
        "faiss_ms_per_query": round(t_fa, 3),
        "identical_results_pct": round(100 * agree / len(queries), 1),
    }


if __name__ == "__main__":
    from src.knowledge.chunker import build_chunks
    from src.knowledge.embedder import get_embedder
    from src.knowledge.loader import load_corpus

    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    emb = get_embedder("tfidf_svd").fit([c.text for c in chunks])
    V = emb.encode([c.text for c in chunks])

    qs = emb.encode([
        "how long do I have to return an opened laptop",
        "what does ERR-DP-0x004 mean",
        "how many dead pixels before replacement",
        "when do I get my refund",
        "do you ship to Germany",
    ])
    print("BACKEND COMPARISON")
    for k, v in compare_backends(chunks, V, qs).items():
        print(f"  {k:26s} {v}")

    store = VectorStore(chunks, V)
    print("\nMETADATA FILTERING (archived documents excluded)")
    q = emb.encode_one("what is your return policy")
    for label, flt in [("unfiltered", None), ("current only", lambda c: c.is_current)]:
        hits = store.search(q, top_k=3, where=flt)
        print(f"  {label}:")
        for h in hits:
            flag = "" if h.chunk.is_current else "  <- ARCHIVED"
            print(f"    {h.score:.3f}  {h.chunk.citation:30s}{flag}")
