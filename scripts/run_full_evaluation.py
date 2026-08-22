"""PHASE 12 — Full evaluation.

Runs every component evaluation and produces one report answering
"how well does PacifyIQ actually work", not "the demo works".

    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --component retrieval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.evaluation.components  # noqa: F401,E402  (registers evaluators)
from src.evaluation.framework import (component_names, headline_table,  # noqa: E402
                                      run_component, summary_table)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ORDER = [
    "intent_classification", "sentiment_classification", "retrieval",
    "rag_quality", "screenshot_understanding", "agent_tools",
    "groundedness", "escalation", "end_to_end",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", help="run one component only")
    ap.add_argument("--no-detail", action="store_true")
    args = ap.parse_args()

    pd.set_option("display.width", 210)
    pd.set_option("display.max_colwidth", 48)

    names = [args.component] if args.component else ORDER
    unknown = [n for n in names if n not in component_names()]
    if unknown:
        print(f"unknown component(s): {unknown}")
        print(f"available: {component_names()}")
        return 1

    print("=" * 100)
    print("PACIFYIQ — FULL EVALUATION")
    print("=" * 100)
    print("  Scoring: deterministic where the answer is a fact, curated where")
    print("  it is a decision. No LLM judge contributes any headline number.")

    results = []
    for name in names:
        print(f"\n  running {name} ...", end=" ", flush=True)
        r = run_component(name)
        results.append(r)
        print(f"{r.runtime_s:.1f}s" + (f"  ERROR: {r.error}" if r.error else ""))

    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("HEADLINE — one row per component")
    print("=" * 100)
    head = headline_table(results)
    print(head.to_string(index=False))
    head.to_csv(RESULTS / "evaluation_headline.csv", index=False)

    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ALL METRICS")
    print("=" * 100)
    full = summary_table(results)
    print(full.to_string(index=False))
    full.to_csv(RESULTS / "evaluation_all_metrics.csv", index=False)

    # -----------------------------------------------------------------
    below = full[full["status"] == "below target"]
    print("\n" + "=" * 100)
    print("METRICS BELOW TARGET")
    print("=" * 100)
    if len(below):
        print(below[["component", "metric", "value", "target"]].to_string(index=False))
    else:
        print("  none")

    # -----------------------------------------------------------------
    if not args.no_detail:
        print("\n" + "=" * 100)
        print("FAILURE ANALYSIS")
        print("=" * 100)
        for r in results:
            if r.failures is None or len(r.failures) == 0:
                continue
            print(f"\n{r.component} — {len(r.failures)} failing case(s)")
            cols = [c for c in r.failures.columns
                    if c not in ("hits", "gold", "retrieved", "metrics")][:7]
            print(r.failures[cols].head(12).to_string(index=False))
            slug = r.component.split(".")[0].strip().replace("/", "_")
            r.failures.to_csv(RESULTS / f"evaluation_failures_{slug}.csv", index=False)

        print("\n" + "=" * 100)
        print("END-TO-END BY CATEGORY")
        print("=" * 100)
        e2e = next((r for r in results if r.component.startswith("10.")), None)
        if e2e and "by_category" in e2e.detail:
            print(e2e.detail["by_category"].to_string())

    # -----------------------------------------------------------------
    payload = {
        "components": [
            {
                "component": r.component, "n_cases": r.n_cases,
                "runtime_s": round(r.runtime_s, 2), "error": r.error,
                "metrics": [m.to_dict() for m in r.metrics],
            }
            for r in results
        ]
    }
    (RESULTS / "evaluation_full.json").write_text(json.dumps(payload, indent=2))

    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    n_metrics = len(full[full["value"].notna()])
    n_targets = len(full[full["status"] != ""])
    n_met = len(full[full["status"] == "meets target"])
    print(f"  components evaluated   {len(results)}")
    print(f"  metrics reported       {n_metrics}")
    print(f"  metrics with a target  {n_targets}")
    print(f"  targets met            {n_met}/{n_targets}")
    print(f"  total cases            {sum(r.n_cases for r in results)}")
    print("\n  Written to reports/results/evaluation_*.{csv,json}")
    print("  Full write-up: reports/evaluation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
