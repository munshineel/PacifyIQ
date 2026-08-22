"""PHASE 6.5 — Understanding-retrieval bridge evaluation.

Measures each routing component independently, on both the retrieval set (does
it find the right evidence) and the abstention set (does it now wrongly answer
questions the corpus cannot answer).

The second measurement matters. Boosting makes retrieval scores go up, and
abstention thresholds on retrieval scores - so a routing change that improves
recall can silently destroy abstention. Both are reported.

    python scripts/evaluate_routing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.knowledge import evaluation as kev  # noqa: E402
from src.knowledge.bm25 import BM25Index  # noqa: E402
from src.knowledge.embedder import TfidfSvdEmbedder  # noqa: E402
from src.knowledge.retriever import Retriever  # noqa: E402
from src.knowledge.vector_store import VectorStore  # noqa: E402
from src.rag.abstention import decide  # noqa: E402
from src.rag.context import assemble  # noqa: E402
from src.rag.routing import RoutedRetriever  # noqa: E402
from src.understanding.pipeline import UnderstandingPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def eval_retrieval(routed: RoutedRetriever | None, base: Retriever, cases, k=10):
    """Recall/MRR/coverage, with or without routing."""
    results = []
    for c in cases:
        gold = kev.gold_section_keys(c["gold_sections"])
        if routed is None:
            res = base.retrieve(c["question"], top_k=k)
        else:
            res, _, _ = routed.retrieve(c["question"], top_k=k)
        retrieved = kev.retrieved_section_keys(res.hits)

        metrics = {}
        for kk in (1, 3, 5, 10):
            metrics[f"recall@{kk}"] = kev.recall_at_k(retrieved, gold, kk)
            metrics[f"coverage@{kk}"] = kev.coverage_at_k(retrieved, gold, kk)
        metrics["mrr"] = kev.reciprocal_rank(retrieved, gold)
        metrics["ndcg@5"] = kev.ndcg_at_k(retrieved, gold, 5)

        results.append(
            kev.QueryEval(
                id=c["id"], query=c["question"],
                query_type=c.get("type", "single"),
                difficulty=c.get("difficulty", "unknown"),
                gold=gold, retrieved=retrieved, hits=res.hits, metrics=metrics,
            )
        )
    return results


def eval_abstention(routed: RoutedRetriever | None, base: Retriever, questions, k=5):
    """Fraction correctly abstained. Boosting inflates scores, so this can
    regress even when retrieval improves."""
    abstained = 0
    for q in questions:
        if routed is None:
            res = base.retrieve(q, top_k=k)
        else:
            res, _, _ = routed.retrieve(q, top_k=k)
        ctx = assemble(res, max_chunks=k)
        if decide(res, ctx).decision.value in ("abstain", "escalate"):
            abstained += 1
    return abstained / len(questions)


def main() -> int:
    pd.set_option("display.width", 200)

    store = VectorStore.load(settings.index_dir)
    emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
    base = Retriever(store, emb, BM25Index(store.chunks), strategy="rrf_w", top_k=5)
    understander = UnderstandingPipeline.load()

    ret_cases = kev.load_eval_set("retrieval_eval")
    unans = [c["question"] for c in json.loads(
        (settings.eval_dir / "unanswerable_eval.json").read_text())["cases"]]
    answerable = [c["question"] for c in ret_cases[:60]]

    print("=" * 80)
    print("PHASE 6.5 — UNDERSTANDING-RETRIEVAL BRIDGE")
    print("=" * 80)
    print(f"  retrieval cases    {len(ret_cases)}")
    print(f"  unanswerable       {len(unans)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. COMPONENT ABLATION")
    print("=" * 80)

    configs = [
        ("baseline (no routing)", None),
        ("+ entities only", dict(use_intent=False, use_meta=False,
                                 use_entities=True, use_enrichment=True)),
        ("+ meta only", dict(use_intent=False, use_meta=True,
                             use_entities=False, use_enrichment=False)),
        ("+ intent only", dict(use_intent=True, use_meta=False,
                               use_entities=False, use_enrichment=False)),
        ("+ all routing", dict(use_intent=True, use_meta=True,
                               use_entities=True, use_enrichment=True)),
    ]

    rows = []
    for label, cfg in configs:
        routed = None if cfg is None else RoutedRetriever(base, understander, **cfg)
        res = eval_retrieval(routed, base, ret_cases)
        s = kev.summarize(res)
        abst = eval_abstention(routed, base, unans)
        false_abst = 1 - eval_abstention(routed, base, answerable)
        rows.append({
            "config": label,
            "recall@1": s["recall@1"], "recall@3": s["recall@3"],
            "recall@5": s["recall@5"], "recall@10": s["recall@10"],
            "coverage@5": s["coverage@5"], "mrr": s["mrr"], "ndcg@5": s["ndcg@5"],
            "abstention": round(abst, 4),
            "false_abst": round(1 - false_abst, 4),
        })
        print(f"  {label:26s} recall@5={s['recall@5']:.3f}  mrr={s['mrr']:.3f}  "
              f"abstention={abst:.3f}")

    df = pd.DataFrame(rows)
    df["balanced_abstention"] = (df["abstention"] * (1 - df["false_abst"])).round(4)
    print("\n" + df.to_string(index=False))
    df.to_csv(RESULTS / "routing_ablation.csv", index=False)

    base_row = df.iloc[0]
    full_row = df.iloc[-1]
    print("\n  DELTA (all routing vs baseline):")
    for m in ["recall@1", "recall@3", "recall@5", "recall@10", "coverage@5",
              "mrr", "ndcg@5", "abstention", "balanced_abstention"]:
        d = full_row[m] - base_row[m]
        print(f"    {m:22s} {base_row[m]:.4f} -> {full_row[m]:.4f}  ({d:+.4f})")

    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. THE QUERIES THIS WAS BUILT TO FIX")
    print("=" * 80)

    routed = RoutedRetriever(base, understander)
    targets = ["R058", "R059", "R064", "R065", "R022", "R038", "R046", "R013",
               "R029", "R030", "R073", "R074", "R075", "R078", "R079"]
    before = {r.id: r for r in eval_retrieval(None, base, ret_cases)}
    after = {r.id: r for r in eval_retrieval(routed, base, ret_cases)}

    rows = []
    for tid in targets:
        if tid not in before:
            continue
        b, a = before[tid], after[tid]
        rows.append({
            "id": tid,
            "query": b.query[:52],
            "before@5": b.metrics["recall@5"],
            "after@5": a.metrics["recall@5"],
            "before_rank": b.first_gold_rank,
            "after_rank": a.first_gold_rank,
            "fixed": bool(a.metrics["recall@5"] and not b.metrics["recall@5"]),
            "broken": bool(b.metrics["recall@5"] and not a.metrics["recall@5"]),
        })
    tdf = pd.DataFrame(rows)
    print(tdf.to_string(index=False))
    tdf.to_csv(RESULTS / "routing_target_queries.csv", index=False)
    print(f"\n  fixed: {tdf['fixed'].sum()}   broken: {tdf['broken'].sum()}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. FULL BEFORE / AFTER")
    print("=" * 80)
    fixed = [i for i in before if after[i].metrics["recall@5"] > before[i].metrics["recall@5"]]
    broken = [i for i in before if after[i].metrics["recall@5"] < before[i].metrics["recall@5"]]
    print(f"  queries fixed by routing   {len(fixed)}")
    print(f"  queries broken by routing  {len(broken)}")
    if broken:
        print("\n  BROKEN (routing made these worse):")
        for i in broken:
            print(f"    {i}  {before[i].query[:58]}")
            print(f"        gold={sorted(f'{d}:{s}' for d, s in before[i].gold)}")
            print(f"        now  ={[f'{d}:{s}' for d, s in after[i].retrieved[:3]]}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. MARGIN THRESHOLD SWEEP")
    print("=" * 80)
    print("  How confident must the classifier be before its intent is trusted?\n")
    rows = []
    for margin in (0.0, 0.10, 0.15, 0.25, 0.40, 1.01):
        r = RoutedRetriever(base, understander, min_margin=margin)
        s = kev.summarize(eval_retrieval(r, base, ret_cases))
        rows.append({"min_margin": margin, "recall@5": s["recall@5"],
                     "mrr": s["mrr"], "coverage@5": s["coverage@5"]})
        print(f"    margin >= {margin:.2f}   recall@5={s['recall@5']:.4f}  "
              f"mrr={s['mrr']:.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "routing_margin_sweep.csv", index=False)

    # -----------------------------------------------------------------
    (RESULTS / "routing_summary.json").write_text(json.dumps({
        "baseline": {k: float(base_row[k]) for k in
                     ["recall@5", "mrr", "coverage@5", "abstention"]},
        "routed": {k: float(full_row[k]) for k in
                   ["recall@5", "mrr", "coverage@5", "abstention"]},
        "queries_fixed": len(fixed),
        "queries_broken": len(broken),
    }, indent=2))

    print("\n" + "=" * 80)
    print("ROUTING EVALUATION COMPLETE")
    print(f"  recall@5    {base_row['recall@5']:.3f} -> {full_row['recall@5']:.3f}"
          f"  ({full_row['recall@5'] - base_row['recall@5']:+.3f})")
    print(f"  MRR         {base_row['mrr']:.3f} -> {full_row['mrr']:.3f}"
          f"  ({full_row['mrr'] - base_row['mrr']:+.3f})")
    print(f"  abstention  {base_row['abstention']:.3f} -> {full_row['abstention']:.3f}"
          f"  ({full_row['abstention'] - base_row['abstention']:+.3f})")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
