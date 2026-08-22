"""Lexical retrieval with BM25.

WHY THIS EXISTS. Dense embeddings are weak on exact identifiers. `ERR-DP-0x004`
fragments into several subword pieces with no unified meaning, so its embedding
lands in a fuzzy neighbourhood of other alphanumeric strings. BM25 matches the
literal token and finds it immediately.

EDA finding 7e identified this before any retriever was built; the retrieval
evaluation tags 8 queries as `lexical` specifically to measure it.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from src.knowledge.chunker import Chunk
from src.knowledge.vector_store import SearchHit

# Keep hyphens and 0x prefixes intact so error codes survive tokenisation.
TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.I)

STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "of", "to", "in", "for", "on", "at", "by", "with", "from", "as",
    "that", "this", "it", "its", "i", "you", "my", "your", "we", "our",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, hyphenation preserved.

    Error codes are additionally emitted as their component parts, so a query
    for `ERR-DP-0x004` matches whether the document writes it hyphenated or
    spaced.
    """
    toks = [t.lower() for t in TOKEN.findall(text)]
    out = []
    for t in toks:
        if t in STOP:
            continue
        out.append(t)
        if "-" in t or "_" in t:
            out.extend(p for p in re.split(r"[-_]", t) if p and p not in STOP)
    return out


class BM25Index:
    """Okapi BM25 over the chunk collection."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build()

    def _build(self) -> None:
        from rank_bm25 import BM25Okapi

        self.tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized, k1=self.k1, b=self.b)

    def score_all(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Callable[[Chunk], bool] | None = None,
    ) -> list[SearchHit]:
        scores = self.score_all(query)

        if where is not None:
            mask = np.array([where(c) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)

        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(self.chunks[int(i)], float(scores[int(i)]), rank + 1, source="bm25")
            for rank, i in enumerate(order)
            if np.isfinite(scores[int(i)]) and scores[int(i)] > 0
        ]

    def __len__(self) -> int:
        return len(self.chunks)


if __name__ == "__main__":
    from src.knowledge.chunker import build_chunks
    from src.knowledge.embedder import get_embedder
    from src.knowledge.loader import load_corpus
    from src.knowledge.vector_store import VectorStore

    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    bm = BM25Index(chunks)
    emb = get_embedder("tfidf_svd").fit([c.text for c in chunks])
    store = VectorStore(chunks, emb.encode([c.text for c in chunks]))

    print("DENSE vs BM25 on exact identifiers")
    print("(EDA finding 7e: codes fragment under subword tokenisation)\n")
    for q in ["ERR-DP-0x004", "PAY-402", "THRM-88", "STO-440",
              "how long do I have to return an opened laptop"]:
        print(f"  query: {q}")
        d = store.search(emb.encode_one(q), top_k=1)
        b = bm.search(q, top_k=1)
        print(f"    dense  {d[0].chunk.citation:30s} {d[0].score:.3f}" if d else "    dense  -")
        print(f"    bm25   {b[0].chunk.citation:30s} {b[0].score:.3f}" if b else "    bm25   -")
        print()
