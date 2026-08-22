"""Unified evaluation framework.

Runs every component evaluation through one interface and produces one report.
The individual `scripts/evaluate_*.py` remain the place to debug a component;
this is the place to answer "how well does the whole thing work".

SCORING PHILOSOPHY
------------------
Three tiers, in descending order of trust:

  DETERMINISTIC   exact match, set membership, arithmetic. Used wherever the
                  answer is a fact: an intent label, a chunk id, an eligibility
                  state, a refund figure. No judgement involved, so no judge.

  CURATED         hand-authored expectations. Used where the correct behaviour
                  is a decision rather than a fact: should this escalate, should
                  this be refused, must this answer contain "14".

  LLM-AS-JUDGE    used ONLY for dimensions that genuinely require reading
                  comprehension - relevance and completeness of free text. It is
                  off by default, requires a key, and is validated against a
                  human-labelled subset before any number it produces is quoted.

The default configuration uses no judge at all. Every headline number in the
report is deterministic or curated.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


class Scoring:
    DETERMINISTIC = "deterministic"
    CURATED = "curated"
    JUDGE = "llm_judge"


@dataclass
class Metric:
    """One measured quantity, with enough context to interpret it."""

    name: str
    value: float
    n: int
    scoring: str = Scoring.DETERMINISTIC
    higher_is_better: bool = True
    baseline: float | None = None       # what a trivial system would score
    target: float | None = None         # what would be good enough
    note: str = ""

    @property
    def lift(self) -> float | None:
        if self.baseline is None:
            return None
        return self.value - self.baseline

    @property
    def meets_target(self) -> bool | None:
        if self.target is None:
            return None
        return self.value >= self.target if self.higher_is_better else self.value <= self.target

    def to_dict(self) -> dict[str, Any]:
        # numpy scalars leak in from pandas aggregations and are not JSON
        # serialisable, which only surfaces at report-writing time.
        def _plain(v):
            if isinstance(v, np.generic):
                return v.item()
            return v

        d = {k: _plain(v) for k, v in asdict(self).items()}
        d["lift"] = _plain(self.lift)
        mt = self.meets_target
        d["meets_target"] = None if mt is None else bool(mt)
        return d


@dataclass
class ComponentResult:
    """Everything one component evaluation produced."""

    component: str
    metrics: list[Metric] = field(default_factory=list)
    n_cases: int = 0
    runtime_s: float = 0.0
    failures: pd.DataFrame | None = field(default=None, repr=False)
    detail: dict[str, Any] = field(default_factory=dict, repr=False)
    error: str | None = None

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def headline(self) -> Metric | None:
        return self.metrics[0] if self.metrics else None

    def to_rows(self) -> list[dict[str, Any]]:
        return [{"component": self.component, **m.to_dict()} for m in self.metrics]


# =====================================================================
# Shared metric helpers
# =====================================================================

def classification_metrics(y_true, y_pred, labels=None) -> dict[str, Any]:
    """Precision, recall, F1, macro-F1 and the confusion matrix.

    Accuracy is reported but never used for selection: with a 10:1 class
    imbalance a majority-class predictor scores 18.2% and is useless.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    labels = labels or sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": mp,
        "macro_recall": mr,
        "per_class": pd.DataFrame(
            {"precision": p, "recall": r, "f1": f, "support": s}, index=labels
        ).round(4),
        "confusion": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        ),
    }


def contains_any(text: str, needles: list[str]) -> bool:
    """Tolerant containment: ignores case and thousands separators.

    Used instead of an LLM judge for factual correctness. Every curated case
    turns on a specific number - 14 days, 5 pixels, Rs 5,000 - and string
    matching checks that exactly, cheaply and without circularity.
    """
    if not needles:
        return True
    hay = " ".join(str(text).lower().split()).replace(",", "")
    return any(str(n).lower().replace(",", "") in hay for n in needles)


def contains_none(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    hay = " ".join(str(text).lower().split()).replace(",", "")
    return not any(str(n).lower().replace(",", "") in hay for n in needles)


def balanced_score(detection: float, false_positive: float) -> float:
    """Detection x (1 - false positives).

    Reported wherever a component can score perfectly by refusing everything.
    A system that always abstains scores 1.0 on abstention and 0.0 here.
    """
    return detection * (1.0 - false_positive)


# =====================================================================
# Registry
# =====================================================================

EvalFn = Callable[[], ComponentResult]
_REGISTRY: dict[str, EvalFn] = {}


def register(name: str):
    def deco(fn: EvalFn) -> EvalFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def run_component(name: str) -> ComponentResult:
    fn = _REGISTRY.get(name)
    if fn is None:
        return ComponentResult(component=name, error=f"no evaluator registered")
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        return ComponentResult(component=name, error=f"{type(e).__name__}: {e}",
                               runtime_s=time.perf_counter() - t0)
    result.runtime_s = time.perf_counter() - t0
    return result


def component_names() -> list[str]:
    return list(_REGISTRY)


def summary_table(results: list[ComponentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.error:
            rows.append({"component": r.component, "metric": "-", "value": None,
                         "n": 0, "scoring": "-", "status": f"ERROR: {r.error}"})
            continue
        for m in r.metrics:
            status = ""
            if m.meets_target is not None:
                status = "meets target" if m.meets_target else "below target"
            rows.append({
                "component": r.component, "metric": m.name,
                "value": round(m.value, 4), "n": m.n, "scoring": m.scoring,
                "baseline": m.baseline, "target": m.target, "status": status,
            })
    return pd.DataFrame(rows)


def headline_table(results: list[ComponentResult]) -> pd.DataFrame:
    """One row per component - the table that goes in the README."""
    rows = []
    for r in results:
        h = r.headline()
        if r.error or h is None:
            rows.append({"component": r.component, "headline metric": "-",
                         "value": "-", "n": 0, "scoring": "-"})
            continue
        rows.append({
            "component": r.component,
            "headline metric": h.name,
            "value": round(h.value, 3),
            "n": h.n,
            "scoring": h.scoring,
            "baseline": h.baseline if h.baseline is not None else "-",
        })
    return pd.DataFrame(rows)
