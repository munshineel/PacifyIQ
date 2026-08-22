"""Retrieval.

Four strategies behind one interface, switchable by config so they can be
ablated rather than assumed:

    dense   embedding cosine similarity
    bm25    lexical scoring
    hybrid  reciprocal rank fusion of both
    rrf_w   weighted RRF, dense-leaning

Metadata filtering is applied *before* ranking, so a filter cannot be defeated
by an excluded chunk scoring higher. This matters for two planted defects:

- DEFECT-02: `return_policy_v1_ARCHIVED` is more similar to a naive "what is
  your return policy" query than the current v2, because it is shorter and less
  qualified. Version filtering is the only reliable fix.
- DEFECT-03: EU customers are governed by an addendum that overrides base
  policy. Region filtering surfaces it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.knowledge.bm25 import BM25Index
from src.knowledge.chunker import Chunk
from src.knowledge.embedder import Embedder
from src.knowledge.vector_store import SearchHit, VectorStore

STRATEGIES = ("dense", "bm25", "hybrid", "rrf_w")

# Authority weighting. Failure analysis of the first evaluation run showed that
# `product_faq` chunks crowded out the authoritative policy sections they
# paraphrase: 14 of 24 failures had the correct policy section at rank 6-10,
# displaced by an FAQ restatement of the same fact. The FAQ is written in
# casual language, which matches casually-phrased queries more closely than the
# formal clause that actually governs.
#
# The corpus itself states the precedence (product_faq: "Where it differs from a
# policy document, the policy document governs"), so encoding it is not a hack -
# it is making an existing rule operational.
AUTHORITY = {
    "policy": 1.00,
    "troubleshooting": 0.97,
    "manual": 0.95,
    "faq": 0.88,
    "unknown": 0.90,
}


@dataclass
class RetrievalResult:
    """Everything one query produced, including diagnostics."""

    query: str
    hits: list[SearchHit]
    strategy: str
    n_candidates: int
    top_score: float
    filters: dict[str, Any]
    # Raw component scores, kept because fused RRF scores are compressed into a
    # narrow band (~0.015-0.017) and carry no information about whether the
    # evidence is any good. Abstention needs an interpretable signal, and
    # cosine similarity is one; an RRF score is not.
    max_dense_score: float = 0.0
    max_bm25_score: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def docs(self) -> list[str]:
        return [h.chunk.doc for h in self.hits]

    @property
    def sections(self) -> list[tuple[str, str | None]]:
        return [(h.chunk.doc, h.chunk.section) for h in self.hits]

    def has_conflict(self) -> bool:
        """True when retrieved chunks span both current and archived versions,
        or both a base policy and its regional override.

        This does not decide the answer - it flags that the evidence is not
        unanimous, which Phase 7 turns into a surface-and-escalate behaviour.
        """
        versions = {h.chunk.version for h in self.hits}
        regions = {h.chunk.region for h in self.hits}
        return len(versions) > 1 or len(regions) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "top_score": round(self.top_score, 4),
            "conflict": self.has_conflict(),
            "hits": [h.to_dict() for h in self.hits],
        }


def _rrf(rankings: list[list[int]], weights: list[float], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion.

    Combines rankings rather than scores, which matters because dense cosine
    (0-1) and BM25 (unbounded) are not on comparable scales. Normalising them
    to a common range is fragile; fusing ranks is not.
    """
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return fused


class Retriever:
    """The retrieval component. Independently testable, no LLM involved."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        bm25: BM25Index | None = None,
        strategy: str = "hybrid",
        top_k: int = 5,
        min_score: float = 0.0,
        exclude_archived: bool = True,
        authority_weighting: bool = True,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}, expected {STRATEGIES}")
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Index(store.chunks)
        self.strategy = strategy
        self.top_k = top_k
        self.min_score = min_score
        self.exclude_archived = exclude_archived
        self.authority_weighting = authority_weighting

    # ---------------------------------------------------------------
    def _build_filter(
        self, region: str | None, doc_type: str | None, topic: str | None,
        product: str | None, include_archived: bool,
        docs: list[str] | None = None,
    ) -> tuple[Callable[[Chunk], bool] | None, dict[str, Any]]:
        applied: dict[str, Any] = {}

        def keep(c: Chunk) -> bool:
            if not include_archived and self.exclude_archived and not c.is_current:
                return False
            if docs and c.doc not in docs:
                return False
            if region and c.region not in ("all", region):
                return False
            if doc_type and c.doc_type != doc_type:
                return False
            if topic and c.topic != topic:
                return False
            if product and c.product and product.lower() not in c.product.lower():
                return False
            return True

        if not include_archived and self.exclude_archived:
            applied["exclude_archived"] = True
        for name, val in [("region", region), ("doc_type", doc_type),
                          ("topic", topic), ("product", product)]:
            if val:
                applied[name] = val
        if docs:
            applied["docs"] = list(docs)

        return (keep if applied else None), applied

    # ---------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        region: str | None = None,
        doc_type: str | None = None,
        topic: str | None = None,
        product: str | None = None,
        include_archived: bool = False,
        strategy: str | None = None,
        docs: list[str] | None = None,
    ) -> RetrievalResult:
        k = top_k or self.top_k
        strat = strategy or self.strategy
        where, applied = self._build_filter(
            region, doc_type, topic, product, include_archived, docs
        )

        # Fetch deeper than k when re-ranking, so authority weighting can
        # promote a policy section sitting below the cut rather than only
        # reshuffling what already made it.
        fetch = k * 3 if self.authority_weighting else k

        if strat == "dense":
            hits = self.store.search(self.embedder.encode_one(query), top_k=fetch, where=where)
        elif strat == "bm25":
            hits = self.bm25.search(query, top_k=fetch, where=where)
        else:
            hits = self._fuse(query, fetch, where, weighted=(strat == "rrf_w"))

        hits = [h for h in hits if h.score >= self.min_score]

        if self.authority_weighting and hits:
            for h in hits:
                h.score *= AUTHORITY.get(h.chunk.doc_type, 0.9)
            hits.sort(key=lambda h: -h.score)

        hits = hits[:k]
        for i, h in enumerate(hits, start=1):
            h.rank = i

        n_cand = sum(1 for c in self.store.chunks if where is None or where(c))

        # Always compute the interpretable component scores, whatever strategy
        # was used for ranking.
        dense_all = self.store.score_all(self.embedder.encode_one(query))
        bm25_all = self.bm25.score_all(query)
        if where is not None:
            mask = np.array([where(c) for c in self.store.chunks])
            dense_all = np.where(mask, dense_all, -np.inf)
            bm25_all = np.where(mask, bm25_all, -np.inf)
        max_dense = float(np.max(dense_all)) if np.isfinite(dense_all).any() else 0.0
        max_bm25 = float(np.max(bm25_all)) if np.isfinite(bm25_all).any() else 0.0

        return RetrievalResult(
            query=query,
            hits=hits,
            strategy=strat,
            n_candidates=n_cand,
            top_score=hits[0].score if hits else 0.0,
            filters=applied,
            max_dense_score=max_dense,
            max_bm25_score=max_bm25,
        )

    # ---------------------------------------------------------------
    def _fuse(
        self, query: str, k: int, where: Callable[[Chunk], bool] | None,
        weighted: bool,
    ) -> list[SearchHit]:
        dense_scores = self.store.score_all(self.embedder.encode_one(query))
        bm25_scores = self.bm25.score_all(query)

        if where is not None:
            mask = np.array([where(c) for c in self.store.chunks])
            dense_scores = np.where(mask, dense_scores, -np.inf)
            bm25_scores = np.where(mask, bm25_scores, -np.inf)

        pool = max(k * 4, 20)
        d_rank = [int(i) for i in np.argsort(-dense_scores)[:pool]
                  if np.isfinite(dense_scores[int(i)])]
        b_rank = [int(i) for i in np.argsort(-bm25_scores)[:pool]
                  if np.isfinite(bm25_scores[int(i)]) and bm25_scores[int(i)] > 0]

        weights = [0.7, 0.3] if weighted else [0.5, 0.5]
        fused = _rrf([d_rank, b_rank], weights)

        order = sorted(fused, key=lambda i: -fused[i])[:k]
        out = []
        for rank, idx in enumerate(order, start=1):
            src = ("dense+bm25" if idx in d_rank and idx in b_rank
                   else "dense" if idx in d_rank else "bm25")
            out.append(SearchHit(self.store.chunks[idx], fused[idx], rank, source=src))
        return out

    # ---------------------------------------------------------------
    def explain(self, query: str, top_k: int | None = None, **kw) -> str:
        """Human-readable trace of one retrieval. Used by the evaluation
        notebook so a person can judge whether the evidence answers the query."""
        res = self.retrieve(query, top_k=top_k, **kw)
        lines = [
            f"QUERY     {query}",
            f"strategy  {res.strategy}   candidates {res.n_candidates}"
            f"   filters {res.filters or 'none'}",
        ]
        if res.has_conflict():
            lines.append("⚠ evidence spans multiple versions or regions")
        lines.append("")
        for h in res.hits:
            lines.append(
                f"  [{h.rank}] {h.score:7.4f}  {h.chunk.citation:32s} "
                f"({h.source}, {h.chunk.version})"
            )
            lines.append(f"       {h.chunk.preview(150)}")
        if not res.hits:
            lines.append("  no results above threshold")
        return "\n".join(lines)


if __name__ == "__main__":
    from src.knowledge.chunker import build_chunks
    from src.knowledge.embedder import get_embedder
    from src.knowledge.loader import load_corpus

    chunks = build_chunks(load_corpus(), strategy="section", max_tokens=200)
    emb = get_embedder("tfidf_svd").fit([c.text for c in chunks])
    store = VectorStore(chunks, emb.encode([c.text for c in chunks]))
    r = Retriever(store, emb, strategy="hybrid", top_k=4)

    for q in [
        "how long do I have to return an opened laptop",
        "ERR-DP-0x004",
        "how many dead pixels before you replace the screen",
    ]:
        print(r.explain(q), "\n")

    print("=" * 74)
    print("EU REGIONAL OVERRIDE (DEFECT-03)")
    print("=" * 74)
    q = "how long do I have to return an opened laptop"
    for label, kw in [("no region", {}), ("region=EU", {"region": "EU"})]:
        res = r.retrieve(q, top_k=3, **kw)
        print(f"\n  {label}:")
        for h in res.hits:
            print(f"    {h.chunk.citation:32s} region={h.chunk.region}")
