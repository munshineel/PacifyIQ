"""Context assembly.

Turns retrieved chunks into the evidence block the model sees, within a token
budget, and reports what it noticed on the way: conflicting versions, regional
overrides, weak retrieval.

Ordering matters. Models attend most reliably to the start and end of context,
so the highest-scoring chunk goes first and the second-highest goes last, with
the weaker middle ranks buried where attention is lowest. This is a cheap
mitigation for the "lost in the middle" effect and costs nothing to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.knowledge.retriever import RetrievalResult
from src.llm.client import count_tokens


@dataclass
class AssembledContext:
    """The evidence block plus everything the pipeline learned building it."""

    text: str
    n_chunks: int
    n_tokens: int
    citations: list[str] = field(default_factory=list)
    dropped: int = 0

    top_score: float = 0.0
    score_gap: float = 0.0
    has_version_conflict: bool = False
    has_region_variant: bool = False
    doc_types: set[str] = field(default_factory=set)
    docs: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return self.n_chunks == 0

    @property
    def is_single_source(self) -> bool:
        """One document behind every chunk. Weak evidence for a multi-hop
        question, and worth flagging to the confidence layer."""
        return len(self.docs) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks,
            "n_tokens": self.n_tokens,
            "dropped": self.dropped,
            "top_score": round(self.top_score, 4),
            "score_gap": round(self.score_gap, 4),
            "has_version_conflict": self.has_version_conflict,
            "has_region_variant": self.has_region_variant,
            "docs": sorted(self.docs),
            "doc_types": sorted(self.doc_types),
            "citations": self.citations,
        }


def _interleave(items: list) -> list:
    """Place the strongest evidence at the edges of the context window.

    rank order [1,2,3,4,5] becomes [1,3,5,4,2] - rank 1 first, rank 2 last,
    and the weakest ranks in the middle where attention is lowest.
    """
    if len(items) <= 2:
        return items
    head, tail, flip = [], [], False
    for it in items:
        (tail if flip else head).append(it)
        flip = not flip
    return head + tail[::-1]


def assemble(
    result: RetrievalResult,
    max_tokens: int = 1800,
    max_chunks: int = 5,
    position_aware: bool = True,
) -> AssembledContext:
    """Build the evidence block from a retrieval result."""
    hits = result.hits[:max_chunks]

    if not hits:
        return AssembledContext(text="", n_chunks=0, n_tokens=0)

    scores = [h.score for h in hits]
    top = scores[0]
    gap = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]

    versions = {h.chunk.version for h in hits}
    regions = {h.chunk.region for h in hits}

    ordered = _interleave(hits) if position_aware else hits

    blocks, cites, used, dropped = [], [], 0, 0
    for i, h in enumerate(ordered, start=1):
        citation = h.chunk.citation
        body = " ".join(h.chunk.text.split())
        block = f"[{i}] SOURCE: {citation}\n{body}"
        cost = count_tokens(block)

        if used + cost > max_tokens:
            dropped += 1
            continue

        blocks.append(block)
        cites.append(citation)
        used += cost

    text = "\n\n".join(blocks)
    return AssembledContext(
        text=text,
        n_chunks=len(blocks),
        n_tokens=count_tokens(text),
        citations=cites,
        dropped=dropped,
        top_score=top,
        score_gap=gap,
        has_version_conflict=len(versions) > 1,
        has_region_variant=len(regions) > 1,
        doc_types={h.chunk.doc_type for h in hits},
        docs={h.chunk.doc for h in hits},
    )


def budget_report(system_tokens: int, context_tokens: int, question_tokens: int,
                  max_response: int = 512, window: int = 8192) -> dict[str, Any]:
    """Where the context window is going. Logged per request so the budget can
    be tuned against real usage rather than guessed."""
    used = system_tokens + context_tokens + question_tokens
    return {
        "system": system_tokens,
        "context": context_tokens,
        "question": question_tokens,
        "reserved_response": max_response,
        "total_input": used,
        "headroom": window - used - max_response,
        "utilisation_pct": round(100 * (used + max_response) / window, 1),
    }


if __name__ == "__main__":
    from src.config.settings import settings
    from src.knowledge.bm25 import BM25Index
    from src.knowledge.embedder import TfidfSvdEmbedder
    from src.knowledge.retriever import Retriever
    from src.knowledge.vector_store import VectorStore
    from src.rag.prompts import get_prompt

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    r = Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)

    for q in [
        "how long do I have to return an opened laptop",
        "do you offer student discounts",
    ]:
        res = r.retrieve(q)
        ctx = assemble(res)
        print(f"\nQUERY: {q}")
        for k, v in ctx.to_dict().items():
            print(f"  {k:22s} {v}")

        p = get_prompt("v3")
        print("  budget:", budget_report(
            count_tokens(p.system), ctx.n_tokens, count_tokens(q)
        ))
