"""PHASE 6 — Retrieval evaluation.

Independently testable retrieval: no LLM, no generation. Runs the 120-case
evaluation set against the index and reports whether semantic search surfaces
evidence that actually answers each query.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --ablate     # + chunking sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.knowledge import evaluation as ev  # noqa: E402
from src.knowledge.bm25 import BM25Index  # noqa: E402
from src.knowledge.chunker import build_chunks  # noqa: E402
from src.knowledge.embedder import TfidfSvdEmbedder, get_embedder  # noqa: E402
from src.knowledge.loader import load_corpus  # noqa: E402
from src.knowledge.retriever import Retriever  # noqa: E402
from src.knowledge.vector_store import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def build_retriever(strategy="hybrid", chunk_strategy="section", chunk_size=200,
                    dim=192, top_k=5):
    chunks = build_chunks(load_corpus(), strategy=chunk_strategy,
                          max_tokens=chunk_size, overlap=int(chunk_size * 0.2))
    emb = get_embedder("tfidf_svd", dim=dim).fit([c.text for c in chunks])
    store = VectorStore(chunks, emb.encode([c.text for c in chunks]))
    return Retriever(store, emb, BM25Index(chunks), strategy=strategy, top_k=top_k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true", help="run chunking/top-K sweeps")
    ap.add_argument("--show", type=int, default=6, help="example queries to print")
    args = ap.parse_args()

    pd.set_option("display.width", 190)
    cases = ev.load_eval_set()

    print("=" * 76)
    print(f"RETRIEVAL EVALUATION  ({len(cases)} queries)")
    print("=" * 76)

    # -----------------------------------------------------------------
    print("\nloading index from data/index/ ...")
    if (settings.index_dir / "vectors.npy").exists():
        store = VectorStore.load(settings.index_dir)
        emb = TfidfSvdEmbedder.load(settings.index_dir / "embedder.pkl")
        meta = json.loads((settings.index_dir / "index_metadata.json").read_text())
        print(f"  {len(store)} chunks, {meta['strategy']}_{meta['chunk_size']}, "
              f"dim {meta['embedding_dim']}")
    else:
        print("  no index found - building in memory")
        r0 = build_retriever()
        store, emb = r0.store, r0.embedder

    retriever = Retriever(store, emb, BM25Index(store.chunks), strategy="hybrid", top_k=5)

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("1. STRATEGY COMPARISON")
    print("=" * 76)
    comp = ev.compare_strategies(retriever, cases=cases)
    print(comp.to_string(index=False))
    comp.to_csv(RESULTS / "retrieval_strategy_comparison.csv", index=False)

    best = comp.iloc[0]["strategy"]
    print(f"\n  best strategy by recall@5: {best}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("2. HEADLINE METRICS  (strategy = " + best + ")")
    print("=" * 76)
    results = ev.evaluate_all(retriever, cases, strategy=best)
    summ = ev.summarize(results)
    for k in ["recall@1", "recall@3", "recall@5", "recall@10", "precision@5",
              "coverage@5", "coverage@10", "mrr", "ndcg@5", "answered_pct"]:
        print(f"  {k:14s} {summ[k]}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("3. BREAKDOWN BY QUERY TYPE")
    print("=" * 76)
    bt = ev.breakdown(results, "query_type")
    print(bt.to_string())
    bt.to_csv(RESULTS / "retrieval_by_type.csv")

    print("\nBY DIFFICULTY")
    bd = ev.breakdown(results, "difficulty")
    print(bd.to_string())
    bd.to_csv(RESULTS / "retrieval_by_difficulty.csv")

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("4. FAILURES  (no gold section in top 5)")
    print("=" * 76)
    fails = ev.failures(results)
    print(f"\n  {len(fails)} of {len(results)} queries failed "
          f"({100 * len(fails) / len(results):.1f}%)")
    if len(fails):
        print("\n" + fails.head(15).to_string(index=False))
        fails.to_csv(RESULTS / "retrieval_failures.csv", index=False)

        deeper = fails[fails["found_at_rank"].notna()]
        if len(deeper):
            print(f"\n  {len(deeper)} of those DID retrieve gold, just below rank 5 "
                  f"(ranks: {sorted(deeper['found_at_rank'].astype(int))})")

    # -----------------------------------------------------------------
    print("\n" + "=" * 76)
    print("5. WORKED EXAMPLES")
    print("=" * 76)
    shown = 0
    for r in results:
        if shown >= args.show:
            break
        if r.query_type in ("contradiction", "lexical", "multi", "ambiguous") or not r.answered:
            print(r.report(max_hits=3))
            shown += 1

    # -----------------------------------------------------------------
    if args.ablate:
        print("\n" + "=" * 76)
        print("6. CHUNKING ABLATION")
        print("=" * 76)
        rows = []
        for cstrat in ("section", "fixed"):
            for size in (128, 200, 256, 512):
                rt = build_retriever(strategy=best, chunk_strategy=cstrat,
                                     chunk_size=size)
                res = ev.evaluate_all(rt, cases, strategy=best)
                s = ev.summarize(res)
                s.update({"chunking": cstrat, "size": size,
                          "n_chunks": len(rt.store)})
                rows.append(s)
                print(f"  {cstrat}_{size:<4d} chunks={len(rt.store):4d}  "
                      f"recall@5={s['recall@5']:.3f}  mrr={s['mrr']:.3f}  "
                      f"cov@5={s['coverage@5']:.3f}")
        abl = pd.DataFrame(rows)[
            ["chunking", "size", "n_chunks", "recall@1", "recall@3", "recall@5",
             "coverage@5", "mrr", "ndcg@5"]
        ].sort_values("recall@5", ascending=False)
        print("\n" + abl.to_string(index=False))
        abl.to_csv(RESULTS / "retrieval_chunking_ablation.csv", index=False)

        print("\n  TOP-K SWEEP")
        rows = []
        for k in (1, 3, 5, 10, 20):
            res = ev.evaluate_all(retriever, cases, k=k, strategy=best)
            s = ev.summarize(res)
            rows.append({"top_k": k, "recall@k": s.get(f"recall@{k}", s["recall@10"]),
                         "coverage@5": s["coverage@5"], "mrr": s["mrr"],
                         "precision@5": s["precision@5"]})
        tk = pd.DataFrame(rows)
        print(tk.to_string(index=False))
        tk.to_csv(RESULTS / "retrieval_topk_sweep.csv", index=False)

    # -----------------------------------------------------------------
    (RESULTS / "retrieval_summary.json").write_text(
        json.dumps({"strategy": best, **summ}, indent=2)
    )
    pd.DataFrame(
        [{"id": r.id, "query": r.query, "type": r.query_type,
          "difficulty": r.difficulty, "answered": r.answered,
          "first_gold_rank": r.first_gold_rank, **r.metrics} for r in results]
    ).to_csv(RESULTS / "retrieval_per_query.csv", index=False)

    print("\n" + "=" * 76)
    print(f"RETRIEVAL EVALUATION COMPLETE  |  {best}")
    print(f"  recall@5   {summ['recall@5']:.3f}")
    print(f"  MRR        {summ['mrr']:.3f}")
    print(f"  answered   {summ['answered_pct']}%")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
