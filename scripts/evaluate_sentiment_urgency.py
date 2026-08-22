"""Evaluate the rule-based sentiment and urgency scorers.

Compares against data/eval/sentiment_urgency_eval.json. Read the limitation
note in that file before quoting any number from here.

    python scripts/evaluate_sentiment_urgency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.understanding import evaluation as ev  # noqa: E402
from src.understanding import sentiment as sent  # noqa: E402
from src.understanding.pipeline import UnderstandingPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    pd.set_option("display.width", 170)

    data = json.loads(
        (ROOT / "data" / "eval" / "sentiment_urgency_eval.json").read_text(encoding="utf-8")
    )
    cases = data["cases"]
    print("=" * 74)
    print(f"SENTIMENT & URGENCY EVALUATION  (n={len(cases)})")
    print("=" * 74)
    print(f"\n  LIMITATION: {data['limitation']}\n")

    up = UnderstandingPipeline.load()

    rows = []
    for c in cases:
        u = up.understand(c["text"])
        rows.append(
            {
                "id": c["id"],
                "text": c["text"],
                "true_sentiment": c["sentiment"],
                "pred_sentiment": u.sentiment,
                "sentiment_score": u.sentiment_score,
                "true_urgency": c["urgency"],
                "pred_urgency": u.urgency,
                "urgency_score": u.urgency_score,
                "intent": u.intent,
                "rationale": c["rationale"],
            }
        )
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Sentiment
    # ---------------------------------------------------------------
    labels_s = ["negative", "neutral", "positive"]
    res_s = ev.evaluate(
        "rule-based sentiment", "annotated", df["true_sentiment"], df["pred_sentiment"], labels_s
    )
    print(ev.report_text(res_s))
    print("\nconfusion:")
    print(res_s.confusion.to_string())

    # Does the intent prior actually help, or is the lexicon doing all the work?
    lex_only = [sent.score_sentiment(t, intent=None).label for t in df["text"]]
    res_lex = ev.evaluate(
        "sentiment, lexicon only (no intent prior)", "annotated",
        df["true_sentiment"], lex_only, labels_s
    )
    print(f"\n  ABLATION - intent prior contribution:")
    print(f"    lexicon only              macro-F1 {res_lex.macro_f1:.4f}")
    print(f"    lexicon + intent prior    macro-F1 {res_s.macro_f1:.4f}")
    print(f"    delta                              {res_s.macro_f1 - res_lex.macro_f1:+.4f}")

    # ---------------------------------------------------------------
    # Urgency
    # ---------------------------------------------------------------
    labels_u = ["low", "medium", "high"]
    res_u = ev.evaluate(
        "rule-based urgency", "annotated", df["true_urgency"], df["pred_urgency"], labels_u
    )
    print(ev.report_text(res_u))
    print("\nconfusion:")
    print(res_u.confusion.to_string())

    # Ordinal error matters more than exact match for triage: predicting
    # "medium" when the truth is "high" is a smaller failure than "low".
    order = {"low": 0, "medium": 1, "high": 2}
    dist = (df["pred_urgency"].map(order) - df["true_urgency"].map(order)).abs()
    print("\n  ordinal error (triage view):")
    print(f"    exact match          {(dist == 0).mean():.3f}")
    print(f"    within one level     {(dist <= 1).mean():.3f}")
    print(f"    two levels off       {(dist == 2).sum()} cases")

    # The costly error: a genuinely high-urgency message scored low.
    missed = df[(df["true_urgency"] == "high") & (df["pred_urgency"] == "low")]
    print(f"\n    HIGH scored as LOW:  {len(missed)} "
          f"({100 * len(missed) / max((df['true_urgency'] == 'high').sum(), 1):.0f}% of high cases)")
    for _, r in missed.iterrows():
        print(f"      {r['text'][:64]}")

    # ---------------------------------------------------------------
    # Failures
    # ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("FAILURE CASES")
    print("=" * 74)

    s_fail = df[df["true_sentiment"] != df["pred_sentiment"]]
    print(f"\nsentiment: {len(s_fail)}/{len(df)} wrong")
    for _, r in s_fail.head(10).iterrows():
        print(f"  [{r['true_sentiment']:8s} -> {r['pred_sentiment']:8s}] "
              f"{r['text'][:56]}")

    u_fail = df[df["true_urgency"] != df["pred_urgency"]]
    print(f"\nurgency: {len(u_fail)}/{len(df)} wrong")
    for _, r in u_fail.head(10).iterrows():
        print(f"  [{r['true_urgency']:6s} -> {r['pred_urgency']:6s}] "
              f"{r['text'][:60]}")

    # ---------------------------------------------------------------
    df.to_csv(RESULTS / "sentiment_urgency_predictions.csv", index=False)
    res_s.per_class.to_csv(RESULTS / "per_class_sentiment.csv")
    res_u.per_class.to_csv(RESULTS / "per_class_urgency.csv")

    summary = {
        "n_cases": len(df),
        "sentiment": {
            "macro_f1": round(res_s.macro_f1, 4),
            "weighted_f1": round(res_s.weighted_f1, 4),
            "accuracy": round(res_s.accuracy, 4),
            "lexicon_only_macro_f1": round(res_lex.macro_f1, 4),
            "intent_prior_delta": round(res_s.macro_f1 - res_lex.macro_f1, 4),
        },
        "urgency": {
            "macro_f1": round(res_u.macro_f1, 4),
            "weighted_f1": round(res_u.weighted_f1, 4),
            "accuracy": round(res_u.accuracy, 4),
            "within_one_level": round(float((dist <= 1).mean()), 4),
            "high_scored_low": int(len(missed)),
        },
        "limitation": data["limitation"],
    }
    (RESULTS / "sentiment_urgency_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 74)
    print(f"sentiment macro-F1 {res_s.macro_f1:.4f}   urgency macro-F1 {res_u.macro_f1:.4f}")
    print("Treat both as indicative only - see the limitation note.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
