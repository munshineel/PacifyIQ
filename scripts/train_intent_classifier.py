"""PHASE 4 — Train and compare intent classification baselines.

Runs the full model comparison, the masking ablation, cross-validation, and
error analysis. Saves the winning model and every result table.

    python scripts/train_intent_classifier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda import loaders  # noqa: E402
from src.understanding import evaluation as ev  # noqa: E402
from src.understanding import models as mz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports" / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    pd.set_option("display.width", 190)
    pd.set_option("display.max_columns", 30)

    # =================================================================
    # 1. Data and splits
    # =================================================================
    print("=" * 74)
    print("1. DATA")
    print("=" * 74)

    train_full = loaders.load_intent_train()
    test = loaders.load_intent_test()

    # EDA finding A2: two test rows also appear in training. Remove them.
    leaked = ev.check_leakage(train_full["text"], test["text"])
    print(f"  leaked test rows found: {len(leaked)}")
    for t in sorted(leaked):
        print(f"    removed: {t!r}")
    test = ev.drop_leaked(test, train_full["text"])
    assert not ev.check_leakage(train_full["text"], test["text"]), "leakage remains"

    labels = loaders.INTENT_ORDER

    # Two validation strategies, reported side by side.
    #  - random:  stratified, but templates appear in BOTH halves
    #  - group:   split by template skeleton, so validation phrasings are unseen
    train_r, val_r = ev.make_splits(train_full)
    train_g, val_g = ev.make_group_splits(train_full)

    ov_r = len(
        set(train_r["text"].map(ev.template_skeleton))
        & set(val_r["text"].map(ev.template_skeleton))
    )
    ov_g = len(
        set(train_g["text"].map(ev.template_skeleton))
        & set(val_g["text"].map(ev.template_skeleton))
    )

    print(f"\n  random split   train {len(train_r):5d}  val {len(val_r):4d}  "
          f"shared templates {ov_r:4d}   <- leaks phrasing")
    print(f"  group split    train {len(train_g):5d}  val {len(val_g):4d}  "
          f"shared templates {ov_g:4d}   <- used for selection")
    print(f"  held-out test  {len(test):5d}")

    counts = train_g["intent"].value_counts()
    print(f"\n  classes: {len(labels)}")
    print(f"  train imbalance: {counts.max()}:{counts.min()} = {counts.max()/counts.min():.1f}x")
    print(f"  group-val smallest class: {val_g['intent'].value_counts().min()} examples "
          f"(grouping trades some stratification for honesty)")

    train, val = train_g, val_g
    X_tr, y_tr = train["text"].tolist(), train["intent"].astype(str).tolist()
    X_va, y_va = val["text"].tolist(), val["intent"].astype(str).tolist()
    X_rtr, y_rtr = train_r["text"].tolist(), train_r["intent"].astype(str).tolist()
    X_rva, y_rva = val_r["text"].tolist(), val_r["intent"].astype(str).tolist()
    X_te, y_te = test["text"].tolist(), test["intent"].astype(str).tolist()

    # =================================================================
    # 2. Model comparison on validation
    # =================================================================
    print("\n" + "=" * 74)
    print("2. MODEL COMPARISON (validation)")
    print("=" * 74)

    print(f"  {'model':38s} {'random':>8s} {'group':>8s} {'test':>8s}   fit")
    print("  " + "-" * 70)

    rows, fitted = [], {}
    for label, clf, feats, mask in mz.CANDIDATES + mz.ABLATION:
        # random-split score, to demonstrate why it cannot discriminate
        m_rand = mz.build(clf, features=feats, mask=mask).fit(X_rtr, y_rtr)
        f1_rand = ev.evaluate(label, "val_random", y_rva,
                              m_rand.predict(X_rva), labels).macro_f1

        # group-split model: the one used for selection
        model = mz.build(clf, features=feats, mask=mask)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0

        res_va = ev.evaluate(label, "val_group", y_va, model.predict(X_va), labels)
        res_te = ev.evaluate(label, "test", y_te, model.predict(X_te), labels)

        row = res_va.as_row()
        row["random_split_macro_f1"] = round(f1_rand, 4)
        row["test_macro_f1"] = round(res_te.macro_f1, 4)
        row["gap"] = round(res_va.macro_f1 - res_te.macro_f1, 4)
        row["fit_s"] = round(fit_s, 2)
        row["masked"] = mask
        rows.append(row)
        fitted[label] = (model, res_va, res_te)
        print(f"  {label:38s} {f1_rand:8.4f} {res_va.macro_f1:8.4f} "
              f"{res_te.macro_f1:8.4f}   {fit_s:.2f}s")

    n_tied = sum(1 for r in rows if abs(r["random_split_macro_f1"] - max(
        x["random_split_macro_f1"] for x in rows)) < 1e-9)
    print(f"\n  models tied at the top random-split score: {n_tied}")
    print("  -> a random split cannot select a model on this data")

    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n" + comparison[
        ["model", "random_split_macro_f1", "macro_f1", "weighted_f1",
         "macro_precision", "macro_recall", "test_macro_f1", "gap"]
    ].rename(columns={"macro_f1": "group_macro_f1"}).to_string(index=False))

    # =================================================================
    # 3. Masking ablation
    # =================================================================
    print("\n" + "=" * 74)
    print("3. MASKING ABLATION")
    print("=" * 74)
    print("  EDA finding 7a: order references drove the top class overlap (J=0.25).")
    print("  Does masking them actually help?\n")

    pairs = [
        ("TF-IDF word + LogReg", "TF-IDF word + LogReg (no mask)"),
        ("TF-IDF union + LogReg", "TF-IDF union + LogReg (no mask)"),
        ("TF-IDF union + LinearSVC", "TF-IDF union + LinearSVC (no mask)"),
    ]
    abl_rows = []
    for on, off in pairs:
        if on not in fitted or off not in fitted:
            continue
        _, on_va, on_te = fitted[on]
        _, off_va, off_te = fitted[off]
        abl_rows.append({
            "model": on.replace("TF-IDF ", ""),
            "val_masked": round(on_va.macro_f1, 4),
            "val_unmasked": round(off_va.macro_f1, 4),
            "val_delta": round(on_va.macro_f1 - off_va.macro_f1, 4),
            "test_masked": round(on_te.macro_f1, 4),
            "test_unmasked": round(off_te.macro_f1, 4),
            "test_delta": round(on_te.macro_f1 - off_te.macro_f1, 4),
        })
    ablation = pd.DataFrame(abl_rows)
    ablation.to_csv(RESULTS_DIR / "masking_ablation.csv", index=False)
    print(ablation.to_string(index=False))

    # the specific confusion masking was supposed to fix
    print("\n  order_tracking <-> return_refund_request confusion:")
    for label in ["TF-IDF union + LogReg", "TF-IDF union + LogReg (no mask)"]:
        if label not in fitted:
            continue
        _, r_va, _ = fitted[label]
        cm = r_va.confusion
        a = int(cm.loc["true_order_tracking", "pred_return_refund_request"])
        b = int(cm.loc["true_return_refund_request", "pred_order_tracking"])
        print(f"    {label:38s} {a + b} errors ({a} + {b})")

    # =================================================================
    # 4. Cross-validation for the top models
    # =================================================================
    print("\n" + "=" * 74)
    print("4. CROSS-VALIDATION (5-fold stratified, full training set)")
    print("=" * 74)

    X_all = train_full["text"].tolist()
    y_all = train_full["intent"].astype(str).tolist()

    cv_rows = []
    top_models = comparison[~comparison["model"].str.contains("baseline")].head(4)
    for label in top_models["model"]:
        spec = next((c for c in mz.CANDIDATES + mz.ABLATION if c[0] == label), None)
        if spec is None:
            continue
        _, clf, feats, mask = spec
        cv = ev.cross_validate(mz.build(clf, features=feats, mask=mask), X_all, y_all)
        cv["model"] = label
        cv_rows.append(cv)
        print(f"  {label:38s} {cv['cv_macro_f1_mean']:.4f} "
              f"+/- {cv['cv_macro_f1_std']:.4f} "
              f"[{cv['cv_macro_f1_min']:.3f}, {cv['cv_macro_f1_max']:.3f}]")

    cv_df = pd.DataFrame(cv_rows)[
        ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_macro_f1_min", "cv_macro_f1_max"]
    ]
    cv_df.to_csv(RESULTS_DIR / "cross_validation.csv", index=False)

    # =================================================================
    # 5. Select and refit the winner
    # =================================================================
    print("\n" + "=" * 74)
    print("5. MODEL SELECTION")
    print("=" * 74)

    # A single group split still ties candidates. Break the tie with repeated
    # group splits, averaged. This never touches the test set.
    finalists = comparison[~comparison["model"].str.contains("baseline")].head(5)
    print("  tie-break by repeated group splits (5 seeds), test set untouched:\n")

    rep_rows = []
    for label in finalists["model"]:
        _, clf_i, feats_i, mask_i = next(
            c for c in mz.CANDIDATES + mz.ABLATION if c[0] == label
        )
        r = ev.repeated_group_cv(
            lambda c=clf_i, f=feats_i, m=mask_i: mz.build(c, features=f, mask=m),
            train_full,
        )
        r["model"] = label
        rep_rows.append(r)
        print(f"    {label:38s} {r['repeated_group_f1_mean']:.4f} "
              f"+/- {r['repeated_group_f1_std']:.4f}  "
              f"[{r['repeated_group_f1_min']:.3f}, {r['repeated_group_f1_max']:.3f}]")

    repeated = pd.DataFrame(rep_rows).sort_values(
        "repeated_group_f1_mean", ascending=False
    )
    repeated[["model", "repeated_group_f1_mean", "repeated_group_f1_std",
              "repeated_group_f1_min", "repeated_group_f1_max"]].to_csv(
        RESULTS_DIR / "repeated_group_cv.csv", index=False)

    best_label = repeated.iloc[0]["model"]
    spec = next(c for c in mz.CANDIDATES + mz.ABLATION if c[0] == best_label)
    _, clf, feats, mask = spec
    print(f"\n  SELECTED: {best_label}")
    print("  refitting on the full training set for the final artifact")

    final = mz.build(clf, features=feats, mask=mask)
    final.fit(X_all, y_all)

    res_test = ev.evaluate(best_label, "test (held out)", y_te, final.predict(X_te), labels)
    res_val_only = fitted[best_label][1]

    print(ev.report_text(res_val_only))
    print(ev.report_text(res_test))

    # =================================================================
    # 6. Error analysis
    # =================================================================
    print("\n" + "=" * 74)
    print("6. ERROR ANALYSIS (held-out test)")
    print("=" * 74)

    y_pred_te = final.predict(X_te)

    conf = ev.top_confusions(res_test.confusion, top=10)
    conf.to_csv(RESULTS_DIR / "top_confusions.csv", index=False)
    print("\n  top confusions:")
    print(conf.to_string(index=False))

    fails = ev.failure_cases(
        test, y_te, y_pred_te, extra_cols=["secondary_intent", "note"]
    )
    fails.to_csv(RESULTS_DIR / "failure_cases.csv", index=False)
    print(f"\n  {len(fails)} misclassified of {len(test)} ({100*len(fails)/len(test):.1f}%)")

    # how many "failures" actually predicted the secondary intent?
    comp = ev.compound_accuracy(test, y_pred_te)
    print("\n  compound-message scoring:")
    for k, v in comp.items():
        print(f"    {k:22s} {v}")
    (RESULTS_DIR / "compound_scoring.json").write_text(json.dumps(comp, indent=2))

    # failures grouped by the annotation on each hard case
    if "note" in test.columns:
        tags = {
            "BOUNDARY": "boundary pairs",
            "COMPOUND": "compound messages",
            "DEFECT": "planted defects",
            "IMAGE": "image-dependent",
            "SECURITY": "security-sensitive",
            "HALLUCINATION": "hallucination bait",
        }
        print("\n  error rate by annotated case type:")
        for tag, desc in tags.items():
            mask_t = test["note"].str.contains(tag, na=False)
            if mask_t.sum() == 0:
                continue
            err = (pd.Series(y_pred_te)[mask_t.values] != pd.Series(y_te)[mask_t.values]).mean()
            print(f"    {desc:22s} n={int(mask_t.sum()):3d}  error {100*err:5.1f}%")

    # per-class table for the report
    res_test.per_class.to_csv(RESULTS_DIR / "per_class_test.csv")
    res_val_only.per_class.to_csv(RESULTS_DIR / "per_class_val.csv")
    res_test.confusion.to_csv(RESULTS_DIR / "confusion_test.csv")
    res_val_only.confusion.to_csv(RESULTS_DIR / "confusion_val.csv")

    # =================================================================
    # 7. Persist
    # =================================================================
    print("\n" + "=" * 74)
    print("7. ARTIFACTS")
    print("=" * 74)

    model_path = MODELS_DIR / "intent_classifier.joblib"
    joblib.dump(final, model_path, compress=3)

    metadata = {
        "model": best_label,
        "classifier": clf,
        "features": feats,
        "masking": mask,
        "labels": labels,
        "trained_on": "train.csv (2200 rows, train+val recombined)",
        "n_training_rows": len(X_all),
        "selection_metric": "macro_f1 on group-aware validation split (by template skeleton)",
        "validation_macro_f1": round(res_val_only.macro_f1, 4),
        "test_macro_f1": round(res_test.macro_f1, 4),
        "test_accuracy": round(res_test.accuracy, 4),
        "test_weighted_f1": round(res_test.weighted_f1, 4),
        "cv_macro_f1": cv_df[cv_df["model"] == best_label].to_dict("records"),
        "compound_scoring": comp,
        "leaked_rows_removed": sorted(leaked),
        "sklearn_random_state": mz.RANDOM_STATE,
    }
    (MODELS_DIR / "intent_classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    size_kb = model_path.stat().st_size / 1024
    print(f"  {model_path.name:38s} {size_kb:8.1f} KB")
    print("  intent_classifier_metadata.json")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  reports/results/{f.name}")

    print("\n" + "=" * 74)
    print(f"BASELINE COMPLETE  |  {best_label}")
    print(f"  validation macro-F1 {res_val_only.macro_f1:.4f}")
    print(f"  test macro-F1       {res_test.macro_f1:.4f}   "
          f"(gap {res_val_only.macro_f1 - res_test.macro_f1:+.4f})")
    print(f"  model size          {size_kb:.1f} KB")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
