"""Tests for the evaluation framework (Phase 12).

Guards the reported numbers. A regression in any component should fail here
before it reaches the report.
"""
import pytest

import src.evaluation.components  # noqa: F401  (registers evaluators)
from src.config.settings import settings
from src.evaluation.framework import (Metric, Scoring, balanced_score,
                                      classification_metrics, component_names,
                                      contains_any, contains_none,
                                      headline_table, run_component)

pytestmark = pytest.mark.integration

needs_index = pytest.mark.skipif(
    not (settings.index_dir / "vectors.npy").exists(),
    reason="run scripts/build_index.py first",
)


# =========================================================== framework
def test_all_ten_components_are_registered():
    for name in ["intent_classification", "sentiment_classification",
                 "retrieval", "rag_quality", "screenshot_understanding",
                 "agent_tools", "groundedness", "escalation", "end_to_end"]:
        assert name in component_names()


def test_unknown_component_returns_an_error_not_an_exception():
    r = run_component("does_not_exist")
    assert r.error is not None


def test_metric_reports_lift_against_a_baseline():
    m = Metric("f1", 0.61, 100, baseline=0.09)
    assert m.lift == pytest.approx(0.52)


def test_metric_target_comparison_respects_direction():
    assert Metric("f1", 0.9, 10, target=0.8).meets_target
    assert Metric("errors", 0.0, 10, higher_is_better=False,
                  target=0.0).meets_target
    assert not Metric("errors", 3.0, 10, higher_is_better=False,
                      target=0.0).meets_target


def test_balanced_score_punishes_blanket_refusal():
    """A system that refuses everything scores 1.0 on detection and 0 here."""
    assert balanced_score(1.0, 1.0) == 0.0
    assert balanced_score(1.0, 0.0) == 1.0


def test_containment_ignores_thousands_separators():
    assert contains_any("the fee is Rs 5,000", ["5000"])
    assert contains_any("you have 14 days", ["14"])
    assert contains_none("you have 14 days", ["30"])


def test_classification_metrics_are_complete():
    m = classification_metrics(["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
    for k in ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
              "macro_recall", "per_class", "confusion"]:
        assert k in m
    assert m["confusion"].shape == (2, 2)


# ==================================================== curated dataset
def test_end_to_end_set_is_majority_unanswerable():
    """A test set made only of answerable questions measures fluency, not
    judgement."""
    import json

    data = json.loads((settings.eval_dir / "end_to_end_eval.json").read_text())
    cases = data["cases"]
    resolved = sum(1 for c in cases if c["expected_outcome"] == "resolved")
    assert resolved / len(cases) < 0.55


def test_end_to_end_set_covers_the_required_hard_categories():
    import json

    data = json.loads((settings.eval_dir / "end_to_end_eval.json").read_text())
    categories = {c["category"] for c in data["cases"]}
    categories |= {c["category"] for c in data["image_cases"]}
    for required in ["ambiguous", "missing_info", "conflict", "unanswerable",
                     "hallucination_trap", "injection", "out_of_domain",
                     "screenshot_unreadable", "screenshot_irrelevant"]:
        assert required in categories, f"{required} not represented"


def test_every_outcome_type_is_exercised():
    import json

    data = json.loads((settings.eval_dir / "end_to_end_eval.json").read_text())
    outcomes = {c["expected_outcome"] for c in data["cases"]}
    assert outcomes == {"resolved", "escalated", "needs_information", "refused"}


# ============================================ component result guards
@needs_index
@pytest.mark.parametrize("component,metric,floor", [
    ("intent_classification", "macro_f1", 0.55),
    ("sentiment_classification", "sentiment_macro_f1", 0.70),
    ("retrieval", "recall@5", 0.85),
    ("rag_quality", "faithfulness", 0.95),
    ("screenshot_understanding", "extraction_accuracy", 0.90),
    ("agent_tools", "argument_extraction", 0.95),
    ("groundedness", "detection_rate", 1.0),
    ("escalation", "accuracy", 0.85),
    ("end_to_end", "outcome_accuracy", 0.75),
])
def test_headline_metrics_do_not_regress(component, metric, floor):
    r = run_component(component)
    assert r.error is None, r.error
    m = r.metric(metric)
    assert m is not None, f"{metric} missing from {component}"
    assert m.value >= floor, f"{component}.{metric} fell to {m.value:.4f}"


@needs_index
@pytest.mark.parametrize("component,metric,ceiling", [
    ("rag_quality", "hallucination_rate", 0.05),
    ("rag_quality", "false_abstention_rate", 0.20),
    ("screenshot_understanding", "invented_codes", 0.0),
    ("agent_tools", "tool_execution_errors", 0.0),
    ("groundedness", "false_flag_rate", 0.0),
])
def test_error_metrics_stay_low(component, metric, ceiling):
    r = run_component(component)
    m = r.metric(metric)
    assert m is not None
    assert m.value <= ceiling, f"{component}.{metric} rose to {m.value}"


# ================================================== scoring integrity
@needs_index
def test_no_headline_metric_depends_on_an_llm_judge():
    """Every reported headline is deterministic or curated. A judge is
    available but contributes nothing to the numbers quoted."""
    for name in component_names():
        r = run_component(name)
        if r.error or not r.metrics:
            continue
        assert r.headline().scoring != Scoring.JUDGE, name


@needs_index
def test_agent_does_not_call_every_tool():
    """Guards the claim that tool selection is genuine."""
    from src.agent.tools import REGISTRY

    r = run_component("agent_tools")
    m = r.metric("mean_tools_per_request")
    assert m.value < len(REGISTRY) / 3


@needs_index
def test_vision_measurably_improves_retrieval():
    r = run_component("screenshot_understanding")
    assert r.metric("vision_recall_lift").value > 0.15


@needs_index
def test_headline_table_has_one_row_per_component():
    results = [run_component(n) for n in
               ["retrieval", "escalation", "groundedness"]]
    assert len(headline_table(results)) == 3


# ================================================== documented gaps
@needs_index
def test_unsafe_resolutions_are_tracked():
    """The failure that reaches a customer: answering confidently when the
    system should have refused. Currently non-zero and reported as such."""
    r = run_component("end_to_end")
    m = r.metric("unsafe_resolutions")
    assert m is not None
    assert m.higher_is_better is False
    assert m.value <= 8, f"unsafe resolutions rose to {m.value}"
