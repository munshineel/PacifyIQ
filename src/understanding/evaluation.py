"""Evaluation utilities for the understanding layer.

Accuracy is deliberately reported but never used to select a model. With a
10:1 class imbalance a majority-class predictor scores 18.2% accuracy and is
useless; macro-F1 is the selection metric.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

RANDOM_STATE = 42


# =====================================================================
# Splitting
# =====================================================================

def make_splits(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "intent",
    val_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split.

    Stratification matters here: `out_of_scope` has 40 examples, so a random
    split could leave it with 4 in validation and produce meaningless
    per-class numbers.

    WARNING: on templated data this split leaks *phrasing* between train and
    validation, because both halves contain rows generated from the same
    template. Use `make_group_splits` instead for model selection. This
    function is retained to demonstrate the failure mode.
    """
    train, val = train_test_split(
        df,
        test_size=val_size,
        stratify=df[label_col],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def template_skeleton(text: str) -> str:
    """Collapse a message to its generating template.

    Digits, order references and error codes are replaced with placeholders,
    and surface variation (case, doubled spaces, added greetings/suffixes) is
    stripped, so two messages from the same template map to one skeleton.
    """
    t = str(text).lower()
    t = re.sub(r"pac[-\s]?2026[-\s]?\d+", " ", t)
    t = re.sub(r"#?\b\d[\d\-]*\b", " ", t)
    t = re.sub(r"^(hi|hello|hey|sir|please)\s+", "", t)
    t = re.sub(r"\s+(please|asap|urgent|thanks)$", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make_group_splits(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "intent",
    val_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split: no template appears in both halves.

    Why this exists. A random stratified split on template-generated data
    produces a validation set drawn from the *same* templates as training, so
    the model can score near-perfectly by memorising phrasings it has already
    seen. In practice six different models tied at macro-F1 0.9929 on a random
    split — the metric could not discriminate between them, and selecting on
    it picked the worst of the tied models on held-out data.

    Grouping by template skeleton forces validation to contain phrasings the
    model has never seen, which is a much closer proxy for the real test
    distribution and restores the metric's ability to separate models.
    """
    groups = df[text_col].map(template_skeleton)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(splitter.split(df, df[label_col], groups))
    return (
        df.iloc[tr_idx].reset_index(drop=True),
        df.iloc[va_idx].reset_index(drop=True),
    )


def check_leakage(train_texts, test_texts) -> set[str]:
    """Exact-match overlap between two splits, case- and space-insensitive."""
    a = {str(t).lower().strip() for t in train_texts}
    b = {str(t).lower().strip() for t in test_texts}
    return a & b


def drop_leaked(test_df: pd.DataFrame, train_texts, text_col: str = "text") -> pd.DataFrame:
    """Remove test rows that also appear in training.

    EDA finding A2: two rows leaked, both sitting on deliberately ambiguous
    boundaries — exactly the cases the hard test set exists to probe.
    """
    leaked = check_leakage(train_texts, test_df[text_col])
    if not leaked:
        return test_df
    mask = ~test_df[text_col].str.lower().str.strip().isin(leaked)
    return test_df[mask].reset_index(drop=True)


# =====================================================================
# Metrics
# =====================================================================

@dataclass
class EvalResult:
    """Metrics for one model on one split."""

    name: str
    split: str
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    extras: dict[str, Any] = field(repr=False, default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "split": self.split,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "balanced_acc": round(self.balanced_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
        }


def evaluate(
    name: str,
    split: str,
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> EvalResult:
    """Full metric suite for one model/split pair."""
    labels = labels or sorted(set(y_true) | set(y_pred))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=labels
    ).round(4)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )

    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return EvalResult(
        name=name,
        split=split,
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        macro_precision=mp,
        macro_recall=mr,
        per_class=per_class,
        confusion=cm,
    )


def repeated_group_cv(
    build_fn,
    df: pd.DataFrame,
    n_repeats: int = 5,
    text_col: str = "text",
    label_col: str = "intent",
    val_size: float = 0.2,
) -> dict[str, float]:
    """Average macro-F1 over several group-aware splits.

    A single group split still ties models together (two candidates tied at
    0.9940 on one seed). Repeating the split with different seeds and
    averaging gives a selection metric with a variance estimate, which breaks
    ties on evidence rather than on sort order.

    This never touches the held-out test set.
    """
    scores = []
    for seed in range(n_repeats):
        tr, va = make_group_splits(df, text_col, label_col, val_size, seed=seed)
        model = build_fn()
        model.fit(tr[text_col].tolist(), tr[label_col].astype(str).tolist())
        pred = model.predict(va[text_col].tolist())
        scores.append(
            f1_score(va[label_col].astype(str), pred, average="macro", zero_division=0)
        )
    arr = np.array(scores)
    return {
        "repeated_group_f1_mean": round(float(arr.mean()), 4),
        "repeated_group_f1_std": round(float(arr.std()), 4),
        "repeated_group_f1_min": round(float(arr.min()), 4),
        "repeated_group_f1_max": round(float(arr.max()), 4),
        "n_repeats": n_repeats,
    }


def cross_validate(model, X, y, folds: int = 5) -> dict[str, float]:
    """Stratified k-fold macro-F1, reported with its standard deviation.

    A single number without variance is not a result — with 40 examples in the
    smallest class, fold-to-fold variation is substantial.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return {
        "cv_macro_f1_mean": round(float(scores.mean()), 4),
        "cv_macro_f1_std": round(float(scores.std()), 4),
        "cv_macro_f1_min": round(float(scores.min()), 4),
        "cv_macro_f1_max": round(float(scores.max()), 4),
    }


# =====================================================================
# Error analysis
# =====================================================================

def top_confusions(cm: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Most frequent off-diagonal cells, i.e. the systematic mistakes."""
    rows = []
    for i, ti in enumerate(cm.index):
        for j, pj in enumerate(cm.columns):
            if i == j:
                continue
            n = int(cm.iloc[i, j])
            if n > 0:
                rows.append(
                    {
                        "true": ti.replace("true_", ""),
                        "predicted": pj.replace("pred_", ""),
                        "count": n,
                        "pct_of_true": round(100 * n / max(cm.iloc[i].sum(), 1), 1),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def failure_cases(
    df: pd.DataFrame,
    y_true,
    y_pred,
    text_col: str = "text",
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Every misclassified row, with any annotation columns carried through."""
    mask = np.asarray(y_true) != np.asarray(y_pred)
    cols = [text_col] + (extra_cols or [])
    out = df.loc[mask, [c for c in cols if c in df.columns]].copy()
    out["true"] = np.asarray(y_true)[mask]
    out["predicted"] = np.asarray(y_pred)[mask]
    return out.reset_index(drop=True)


def compound_accuracy(
    df: pd.DataFrame, y_pred, primary_col: str = "intent",
    secondary_col: str = "secondary_intent",
) -> dict[str, Any]:
    """Score compound messages fairly.

    42% of the hard test set carries two genuine intents. Strict single-label
    scoring marks a prediction wrong even when it names the second intent
    present in the message. This reports both.
    """
    y_pred = np.asarray(y_pred)
    primary = df[primary_col].to_numpy()
    secondary = df[secondary_col].fillna("").to_numpy()

    strict = y_pred == primary
    lenient = strict | (y_pred == secondary)

    is_compound = df["is_compound"].to_numpy() if "is_compound" in df else secondary != ""

    return {
        "strict_accuracy": round(float(strict.mean()), 4),
        "lenient_accuracy": round(float(lenient.mean()), 4),
        "compound_n": int(is_compound.sum()),
        "compound_strict": round(float(strict[is_compound].mean()), 4)
        if is_compound.any() else None,
        "compound_lenient": round(float(lenient[is_compound].mean()), 4)
        if is_compound.any() else None,
        "simple_strict": round(float(strict[~is_compound].mean()), 4)
        if (~is_compound).any() else None,
    }


def report_text(result: EvalResult) -> str:
    """Formatted classification report for the console or a file."""
    lines = [
        f"\n{'=' * 74}",
        f"{result.name}  [{result.split}, n={result.n}]",
        "=" * 74,
        f"  accuracy          {result.accuracy:.4f}",
        f"  balanced accuracy {result.balanced_accuracy:.4f}",
        f"  MACRO F1          {result.macro_f1:.4f}   <- selection metric",
        f"  weighted F1       {result.weighted_f1:.4f}",
        f"  macro precision   {result.macro_precision:.4f}",
        f"  macro recall      {result.macro_recall:.4f}",
        "",
        result.per_class.to_string(),
    ]
    return "\n".join(lines)


def sklearn_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0, digits=3)
