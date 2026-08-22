"""Retrieval evaluation.

Independently testable: no LLM, no generation. The question this answers is
narrow and measurable — *does semantic search surface evidence that actually
answers the query?*

Gold labels reference `(doc, section)` pairs, not chunk IDs, because chunk IDs
change with every chunking configuration and would invalidate the entire set on
the first ablation. Sections are the stable join key; `resolve_gold` maps them
onto whatever chunks the current configuration produced.

Metrics
-------
Recall@K      did any gold section appear in the top K
Precision@K   what fraction of the top K were gold
MRR           1 / rank of the first gold hit
nDCG@K        rank-weighted gain, so rank 1 counts more than rank 5
Coverage@K    fraction of *all* gold sections found (matters for multi-hop)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.knowledge.chunker import Chunk
from src.knowledge.retriever import Retriever


# =====================================================================
# Gold resolution
# =====================================================================

def resolve_gold(
    gold_sections: list[dict[str, str]], chunks: list[Chunk]
) -> set[str]:
    """Map (doc, section) gold labels onto chunk IDs in the current index.

    A section usually spans several chunks, so any of them counts as a hit.
    """
    wanted = {(g["doc"], g["section"]) for g in gold_sections}
    return {c.chunk_id for c in chunks if (c.doc, c.section) in wanted}


def gold_section_keys(gold_sections: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(g["doc"], g["section"]) for g in gold_sections}


def retrieved_section_keys(hits) -> list[tuple[str, str | None]]:
    return [(h.chunk.doc, h.chunk.section) for h in hits]


# =====================================================================
# Metrics
# =====================================================================

def recall_at_k(retrieved: list[tuple], gold: set[tuple], k: int) -> float:
    """1.0 if any gold section appears in the top K."""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def coverage_at_k(retrieved: list[tuple], gold: set[tuple], k: int) -> float:
    """Fraction of gold sections found. The metric that matters for multi-hop
    questions, where an answer needs several chunks and finding one is not enough."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: list[tuple], gold: set[tuple], k: int) -> float:
    if k == 0:
        return 0.0
    return len([r for r in retrieved[:k] if r in gold]) / k


def reciprocal_rank(retrieved: list[tuple], gold: set[tuple]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[tuple], gold: set[tuple], k: int) -> float:
    """Binary-relevance nDCG. Rewards placing gold high, not merely present."""
    dcg = sum(
        1.0 / np.log2(i + 1)
        for i, r in enumerate(retrieved[:k], start=1)
        if r in gold
    )
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


# =====================================================================
# Per-query result
# =====================================================================

@dataclass
class QueryEval:
    """One query, fully inspectable."""

    id: str
    query: str
    query_type: str
    difficulty: str
    gold: set[tuple] = field(repr=False, default_factory=set)
    retrieved: list[tuple] = field(repr=False, default_factory=list)
    hits: list = field(repr=False, default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    conflict_flagged: bool = False

    @property
    def answered(self) -> bool:
        """Did retrieval surface evidence that actually answers the question?"""
        return self.metrics.get("recall@5", 0.0) > 0

    @property
    def first_gold_rank(self) -> int | None:
        for i, r in enumerate(self.retrieved, start=1):
            if r in self.gold:
                return i
        return None

    def report(self, max_hits: int = 5) -> str:
        lines = [
            f"\n{'-' * 76}",
            f"[{self.id}] {self.query}",
            f"type={self.query_type}  difficulty={self.difficulty}  "
            f"ANSWERED={'YES' if self.answered else 'NO'}",
            f"gold: {sorted(f'{d}:{s}' for d, s in self.gold)}",
            "",
        ]
        for h in self.hits[:max_hits]:
            key = (h.chunk.doc, h.chunk.section)
            mark = "GOLD" if key in self.gold else "    "
            lines.append(
                f"  {mark} [{h.rank}] {h.score:7.4f}  {h.chunk.citation:30s} "
                f"({h.source})"
            )
            lines.append(f"           {h.chunk.preview(120)}")
        m = self.metrics
        lines.append(
            f"\n  recall@5 {m.get('recall@5', 0):.0f}  "
            f"coverage@5 {m.get('coverage@5', 0):.2f}  "
            f"MRR {m.get('mrr', 0):.3f}  nDCG@5 {m.get('ndcg@5', 0):.3f}"
        )
        return "\n".join(lines)


# =====================================================================
# Runner
# =====================================================================

def load_eval_set(name: str = "retrieval_eval") -> list[dict[str, Any]]:
    path = settings.eval_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def evaluate_query(
    case: dict[str, Any], retriever: Retriever, k: int = 10, **retrieve_kw
) -> QueryEval:
    gold = gold_section_keys(case["gold_sections"])
    result = retriever.retrieve(case["question"], top_k=k, **retrieve_kw)
    retrieved = retrieved_section_keys(result.hits)

    metrics: dict[str, float] = {}
    for kk in (1, 3, 5, 10):
        metrics[f"recall@{kk}"] = recall_at_k(retrieved, gold, kk)
        metrics[f"precision@{kk}"] = precision_at_k(retrieved, gold, kk)
        metrics[f"coverage@{kk}"] = coverage_at_k(retrieved, gold, kk)
        metrics[f"ndcg@{kk}"] = ndcg_at_k(retrieved, gold, kk)
    metrics["mrr"] = reciprocal_rank(retrieved, gold)
    metrics["top_score"] = result.top_score
    metrics["n_gold"] = len(gold)

    return QueryEval(
        id=case["id"],
        query=case["question"],
        query_type=case.get("type", "single"),
        difficulty=case.get("difficulty", "unknown"),
        gold=gold,
        retrieved=retrieved,
        hits=result.hits,
        metrics=metrics,
        conflict_flagged=result.has_conflict(),
    )


def evaluate_all(
    retriever: Retriever, cases: list[dict[str, Any]] | None = None, k: int = 10,
    **retrieve_kw,
) -> list[QueryEval]:
    cases = cases or load_eval_set()
    return [evaluate_query(c, retriever, k=k, **retrieve_kw) for c in cases]


def summarize(results: list[QueryEval]) -> dict[str, float]:
    """Aggregate metrics across all queries."""
    keys = ["recall@1", "recall@3", "recall@5", "recall@10",
            "precision@1", "precision@3", "precision@5",
            "coverage@5", "coverage@10", "ndcg@5", "ndcg@10", "mrr"]
    out = {k: round(float(np.mean([r.metrics.get(k, 0.0) for r in results])), 4) for k in keys}
    out["n_queries"] = len(results)
    out["answered_pct"] = round(100 * np.mean([r.answered for r in results]), 1)
    return out


def breakdown(results: list[QueryEval], by: str = "query_type") -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                by: getattr(r, by),
                "recall@5": r.metrics.get("recall@5", 0.0),
                "coverage@5": r.metrics.get("coverage@5", 0.0),
                "mrr": r.metrics.get("mrr", 0.0),
                "ndcg@5": r.metrics.get("ndcg@5", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    agg = df.groupby(by).agg(["mean", "count"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    keep = [c for c in agg.columns if c.endswith("_mean")] + ["recall@5_count"]
    return agg[keep].rename(columns={"recall@5_count": "n"}).sort_values("recall@5_mean")


def failures(results: list[QueryEval]) -> pd.DataFrame:
    """Queries where no gold section appeared in the top 5."""
    rows = []
    for r in results:
        if r.answered:
            continue
        rows.append(
            {
                "id": r.id,
                "query": r.query,
                "type": r.query_type,
                "difficulty": r.difficulty,
                "gold": "; ".join(sorted(f"{d}:{s}" for d, s in r.gold)),
                "retrieved_top3": "; ".join(
                    f"{d}:{s}" for d, s in r.retrieved[:3]
                ),
                "found_at_rank": r.first_gold_rank,
                "top_score": round(r.metrics.get("top_score", 0.0), 4),
            }
        )
    return pd.DataFrame(rows)


def compare_strategies(
    retriever: Retriever, strategies: tuple[str, ...] = ("dense", "bm25", "hybrid", "rrf_w"),
    cases: list[dict[str, Any]] | None = None, k: int = 10,
) -> pd.DataFrame:
    cases = cases or load_eval_set()
    rows = []
    for s in strategies:
        res = evaluate_all(retriever, cases, k=k, strategy=s)
        summ = summarize(res)
        summ["strategy"] = s
        rows.append(summ)
    cols = ["strategy", "recall@1", "recall@3", "recall@5", "recall@10",
            "precision@5", "coverage@5", "mrr", "ndcg@5", "answered_pct"]
    return pd.DataFrame(rows)[cols].sort_values("recall@5", ascending=False)
