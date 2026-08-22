"""PHASE 3 — EDA runner.

Produces every figure, writes processed outputs for later phases, and prints
the findings that drive Phase 4 decisions.

    python scripts/run_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda import loaders, plots, text_stats  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 170)
    plots.setup()

    print("loading data...")
    tickets = loaders.load_tickets()
    train = loaders.load_intent_train()
    test = loaders.load_intent_test()
    corpus = loaders.load_corpus()

    # ---------------------------------------------------------------
    # Feature engineering for text analysis
    # ---------------------------------------------------------------
    tr_feats = pd.concat(
        [
            train.reset_index(drop=True),
            text_stats.length_features(train["text"]).drop(columns=["text"]),
            text_stats.lexicon_features(train["text"]),
        ],
        axis=1,
    )
    te_feats = pd.concat(
        [
            test.reset_index(drop=True),
            text_stats.length_features(test["text"]).drop(columns=["text"]),
            text_stats.lexicon_features(test["text"]),
        ],
        axis=1,
    )

    overlap = text_stats.class_lexical_overlap(train["text"], train["intent"])

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    print("\ngenerating figures...")
    figs = [
        plots.fig_intent_distribution(tickets),
        plots.fig_sentiment_urgency(tickets),
        plots.fig_class_imbalance(train, test),
        plots.fig_text_characteristics(tr_feats, te_feats),
        plots.fig_class_overlap(overlap),
        plots.fig_temporal(tickets),
        plots.fig_corpus(corpus),
        plots.fig_confidence_leakage(tickets),
    ]
    for f in figs:
        print(f"  {f.name}")

    # ---------------------------------------------------------------
    # Processed outputs for later phases
    # ---------------------------------------------------------------
    print("\nwriting processed outputs...")
    out = loaders.processed_dir()

    tr_feats.to_parquet(out / "train_features.parquet", index=False)
    te_feats.to_parquet(out / "test_features.parquet", index=False)
    overlap.to_csv(out / "class_overlap.csv", index=False)

    intent_profile = (
        tickets.groupby("intent", observed=True)
        .agg(
            tickets=("ticket_id", "size"),
            escalation_pct=("escalated", lambda s: round(100 * s.mean(), 1)),
            negative_pct=("sentiment", lambda s: round(100 * (s == "negative").mean(), 1)),
            high_priority_pct=("priority", lambda s: round(100 * (s == "high").mean(), 1)),
            avg_latency=("latency_seconds", lambda s: round(s.mean(), 2)),
            avg_tokens=("tokens_used", lambda s: round(s.mean())),
        )
        .sort_values("tickets", ascending=False)
    )
    intent_profile["volume_share_pct"] = (
        100 * intent_profile["tickets"] / intent_profile["tickets"].sum()
    ).round(1)
    intent_profile.to_csv(out / "intent_profile.csv")

    distinctive = text_stats.distinctive_terms(train["text"], train["intent"])
    distinctive.to_csv(out / "distinctive_terms.csv", index=False)

    stats = {
        "train_vocabulary": text_stats.vocabulary_stats(train["text"]),
        "test_vocabulary": text_stats.vocabulary_stats(test["text"]),
        "drift": text_stats.vocabulary_drift(train["text"], test["text"]),
        "template_train": text_stats.template_score(train["text"]),
        "template_test": text_stats.template_score(test["text"]),
    }
    for v in stats.values():
        v.pop("top_terms", None)
        v.pop("example_oov", None)
    (out / "text_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    for name in ["train_features.parquet", "test_features.parquet", "class_overlap.csv",
                 "intent_profile.csv", "distinctive_terms.csv", "text_stats.json"]:
        print(f"  {name}")

    # ---------------------------------------------------------------
    # Findings
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("KEY FINDINGS")
    print("=" * 78)

    print("\n[Q1/Q2] Problem types and frequency")
    print(intent_profile[["tickets", "volume_share_pct", "escalation_pct"]].to_string())

    print("\n[Q3/Q4] Negative sentiment and urgency")
    print(
        intent_profile.sort_values("negative_pct", ascending=False)[
            ["negative_pct", "high_priority_pct", "escalation_pct"]
        ].head(6).to_string()
    )

    print("\n[Q5] Class imbalance")
    c = train["intent"].value_counts()
    print(f"  train:  max={c.max()} ({c.idxmax()})  min={c.min()} ({c.idxmin()})  "
          f"ratio={c.max()/c.min():.1f}x")
    ct = test["intent"].value_counts()
    print(f"  test:   max={ct.max()}  min={ct.min()}  ratio={ct.max()/ct.min():.1f}x")

    print("\n[Q6] Language characteristics")
    print(f"  train median length: {tr_feats['n_words'].median():.0f} words   "
          f"test median: {te_feats['n_words'].median():.0f} words   "
          f"({te_feats['n_words'].median()/tr_feats['n_words'].median():.1f}x)")
    print(f"  code-mixed:  train {100*tr_feats['is_codemixed'].mean():.1f}%   "
          f"test {100*te_feats['is_codemixed'].mean():.1f}%")
    print(f"  compound (test only): {100*test['is_compound'].mean():.0f}%")
    d = stats["drift"]
    print(f"  OOV token rate on test: {d['oov_token_rate_pct']}%   "
          f"OOV type rate: {d['oov_type_rate_pct']}%")
    print(f"  template rate: train {stats['template_train']['template_rate_pct']}%   "
          f"test {stats['template_test']['template_rate_pct']}%")

    print("\n[Q7] Design-relevant patterns")
    print("  most confusable intent pairs:")
    for _, r in overlap.head(4).iterrows():
        print(f"    {r['intent_a']:24s} <-> {r['intent_b']:24s} "
              f"J={r['jaccard']:.3f}  [{r['shared_terms']}]")
    print(f"  corpus: {corpus['doc'].nunique()} docs, {len(corpus)} pages, "
          f"median {corpus['n_words'].median():.0f} words/page, "
          f"{corpus['n_words'].sum():,} words total")

    print("\n" + "=" * 78)
    print("EDA complete. See reports/eda_findings.md for the written analysis.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
