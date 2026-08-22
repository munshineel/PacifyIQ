"""PHASE 7 — RAG evaluation.

Runs correctness, faithfulness, citation accuracy and abstention against the
full pipeline. Works entirely offline with the local extractive backend; pass
--backend groq to compare against a hosted model.

    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --backend groq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import available_backends  # noqa: E402
from src.rag import evaluation as ev  # noqa: E402
from src.rag.generator import build_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "groq"])
    ap.add_argument("--prompt", default="v3")
    ap.add_argument("--skip-prompts", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    avail = available_backends()
    print("=" * 78)
    print("RAG EVALUATION")
    print("=" * 78)
    print(f"  backends available: {avail}")
    if args.backend == "groq" and not avail["groq"]:
        print("\n  ERROR: groq selected but PACIFYIQ_GROQ_API_KEY is not set.")
        return 1
    print(f"  using: {args.backend}, prompt {args.prompt}")

    pipe = build_pipeline(backend=args.backend, prompt_version=args.prompt)

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. GENERATION  (25 cases with reference answers)")
    print("=" * 78)
    gen = ev.evaluate_generation(pipe)
    gsum = ev.summarize_generation(gen)
    for k, v in gsum.items():
        print(f"  {k:28s} {v}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. ABSTENTION  (40 questions the corpus cannot answer)")
    print("=" * 78)
    abst = ev.evaluate_abstention(pipe)
    asum = ev.summarize_abstention(abst)
    for k, v in asum.items():
        print(f"  {k:28s} {v}")

    wrong = [r for r in abst if r.answered]
    if wrong:
        print(f"\n  wrongly answered ({len(wrong)}):")
        for r in wrong:
            print(f"    bm25={r.max_bm25:6.2f}  {r.question}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. FALSE ABSTENTION  (the cost side)")
    print("=" * 78)
    fa = ev.evaluate_false_abstention(pipe, n=60)
    print(f"  answerable questions tested   {fa['n_answerable']}")
    print(f"  wrongly refused               {fa['n_refused']}")
    print(f"  false abstention rate         {fa['false_abstention_rate']}")
    if fa["refused"]:
        for r in fa["refused"][:8]:
            print(f"    bm25={r['max_bm25']:6.2f}  {r['question'][:64]}")

    balanced = (
        asum["abstention_rate"] * (1 - fa["false_abstention_rate"])
    )
    print(f"\n  balanced abstention score     {balanced:.4f}")
    print("  (true abstention x (1 - false abstention); a system that refuses")
    print("   everything scores 0 here, which is the point)")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. FAILURES")
    print("=" * 78)
    fails = ev.failure_table(gen)
    print(f"  {len(fails)} of {len(gen)} incorrect\n")
    if len(fails):
        print(fails[["id", "question", "partial", "grounded", "escalated",
                     "misses", "flags"]].to_string(index=False))
        fails.to_csv(RESULTS / "rag_failures.csv", index=False)

    # -----------------------------------------------------------------
    if not args.skip_prompts:
        print("\n" + "=" * 78)
        print("5. PROMPT VERSION COMPARISON")
        print("=" * 78)
        cmp = ev.compare_prompts(
            lambda v: build_pipeline(backend=args.backend, prompt_version=v)
        )
        print(cmp.to_string(index=False))
        cmp.to_csv(RESULTS / "rag_prompt_comparison.csv", index=False)

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("6. WORKED EXAMPLES")
    print("=" * 78)
    for q in [
        "How long do I have to return an opened laptop?",
        "How many dead pixels before you replace the screen?",
        "Do you offer student discounts?",
        "Can you approve my refund right now?",
    ]:
        print("\n" + pipe.explain(q))

    # -----------------------------------------------------------------
    pd.DataFrame([g.to_row() for g in gen]).to_csv(
        RESULTS / "rag_generation_per_case.csv", index=False
    )
    pd.DataFrame([{"id": a.id, "question": a.question, "abstained": a.abstained,
                   "decision": a.decision, "max_bm25": a.max_bm25}
                  for a in abst]).to_csv(
        RESULTS / "rag_abstention_per_case.csv", index=False
    )
    (RESULTS / "rag_summary.json").write_text(json.dumps({
        "backend": args.backend,
        "prompt_version": args.prompt,
        "generation": gsum,
        "abstention": asum,
        "false_abstention": {k: v for k, v in fa.items() if k != "refused"},
        "balanced_abstention_score": round(balanced, 4),
    }, indent=2))

    print("\n" + "=" * 78)
    print(f"RAG EVALUATION COMPLETE  |  backend={args.backend} prompt={args.prompt}")
    print(f"  correctness      {gsum['correctness']:.3f}")
    print(f"  faithfulness     {gsum['faithfulness']:.3f}")
    print(f"  citation acc     {gsum['citation_accuracy']:.3f}")
    print(f"  abstention       {asum['abstention_rate']:.3f}")
    print(f"  false abstention {fa['false_abstention_rate']:.3f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
