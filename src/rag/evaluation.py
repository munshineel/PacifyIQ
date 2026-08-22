"""Generation evaluation.

Four things are measured, and they fail independently:

    correctness      does the answer contain the right fact
    faithfulness     is every claim supported by the retrieved context
    citation accuracy do the cited sources exist and were they provided
    abstention       does it refuse when the corpus has no answer

The last one is the most important and the least commonly reported. A system
that answers everything scores well on the first three and is dangerous.

Correctness is scored by exact-match rules from the evaluation set
(`must_contain` / `must_not_contain`) rather than by an LLM judge. The judge
approach has a circularity problem when the thing being judged is also an LLM,
and every case in this set turns on a specific number - 14 days, 5 pixels,
Rs 57,960 - which string matching checks exactly and cheaply.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.rag.generator import RAGPipeline, RAGResult


def load_cases(name: str) -> list[dict[str, Any]]:
    return json.loads((settings.eval_dir / f"{name}.json").read_text())["cases"]


def _norm(text: str) -> str:
    return re.sub(r"[\s,]+", " ", str(text).lower()).strip()


def contains(answer: str, needle: str) -> bool:
    """Tolerant containment: ignores commas in numbers and case."""
    a = _norm(answer).replace(",", "")
    n = _norm(needle).replace(",", "")
    return n in a


# =====================================================================
# Per-case results
# =====================================================================

@dataclass
class GenerationEval:
    id: str
    question: str
    answer: str

    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    citation_accuracy: float = 0.0
    n_citations: int = 0
    required_citations: list[str] = field(default_factory=list)
    cited_required: int = 0

    is_grounded: bool = True
    hallucination_flags: list[str] = field(default_factory=list)
    escalated: bool = False
    decision: str = ""
    confidence: float = 0.0

    @property
    def correct(self) -> bool:
        """All required facts present, no forbidden ones."""
        return not self.misses and not self.violations

    @property
    def partial(self) -> float:
        if not self.must_contain:
            return 1.0 if not self.violations else 0.0
        return len(self.hits) / len(self.must_contain)

    @property
    def citation_recall(self) -> float:
        if not self.required_citations:
            return 1.0
        return self.cited_required / len(self.required_citations)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question[:60],
            "correct": self.correct,
            "partial": round(self.partial, 3),
            "grounded": self.is_grounded,
            "citation_accuracy": round(self.citation_accuracy, 3),
            "citation_recall": round(self.citation_recall, 3),
            "escalated": self.escalated,
            "decision": self.decision,
            "misses": "; ".join(self.misses),
            "violations": "; ".join(self.violations),
            "flags": ",".join(self.hallucination_flags),
        }


@dataclass
class AbstentionEval:
    id: str
    question: str
    answered: bool
    abstained: bool
    decision: str
    max_bm25: float
    answer: str = ""

    @property
    def correct(self) -> bool:
        return self.abstained


# =====================================================================
# Runners
# =====================================================================

def evaluate_generation(pipeline: RAGPipeline, cases=None) -> list[GenerationEval]:
    cases = cases or load_cases("generation_eval")
    out: list[GenerationEval] = []

    for c in cases:
        r: RAGResult = pipeline.answer(c["question"])
        ans = r.response.answer

        must = c.get("must_contain", []) or []
        must_not = c.get("must_not_contain", []) or []

        hits = [m for m in must if contains(ans, m)]
        misses = [m for m in must if not contains(ans, m)]
        # A skipped abstention is not a content violation; only score
        # forbidden content when the system actually attempted an answer.
        violations = (
            [m for m in must_not if contains(ans, m)] if not r.escalated else []
        )

        req = [
            f"{rc['doc']}:{rc['section']}" for rc in c.get("required_citations", [])
        ]
        cited_keys = set()
        for cit in r.response.citations:
            cited_keys.add(cit.section or "")
        cited_required = sum(
            1 for rc in c.get("required_citations", [])
            if rc["section"] in cited_keys
        )

        out.append(
            GenerationEval(
                id=c["id"],
                question=c["question"],
                answer=ans,
                must_contain=must,
                must_not_contain=must_not,
                hits=hits,
                misses=misses,
                violations=violations,
                citation_accuracy=r.grounding.citation_accuracy if r.grounding else 0.0,
                n_citations=len(r.response.citations),
                required_citations=req,
                cited_required=cited_required,
                is_grounded=r.grounding.is_grounded if r.grounding else True,
                hallucination_flags=(
                    r.grounding.hallucination_flags if r.grounding else []
                ),
                escalated=r.escalated,
                decision=r.trace.decision,
                confidence=r.response.confidence,
            )
        )
    return out


def evaluate_abstention(pipeline: RAGPipeline, cases=None) -> list[AbstentionEval]:
    """Does the system refuse when the corpus genuinely has no answer?"""
    cases = cases or load_cases("unanswerable_eval")
    out = []
    for c in cases:
        r = pipeline.answer(c["question"])
        abstained = r.trace.decision in ("abstain", "escalate") or r.response.is_abstention
        out.append(
            AbstentionEval(
                id=c["id"],
                question=c["question"],
                answered=not abstained,
                abstained=abstained,
                decision=r.trace.decision,
                max_bm25=r.trace.max_bm25,
                answer=r.response.answer[:120],
            )
        )
    return out


def evaluate_false_abstention(pipeline: RAGPipeline, n: int = 60) -> dict[str, Any]:
    """The cost side of abstention: how often does it refuse a question it
    could have answered? A system that abstains on everything scores 100% on
    the abstention set and is useless."""
    cases = load_cases("retrieval_eval")[:n]
    refused = []
    for c in cases:
        r = pipeline.answer(c["question"])
        if r.trace.decision == "abstain":
            refused.append({"id": c["id"], "question": c["question"],
                            "max_bm25": r.trace.max_bm25})
    return {
        "n_answerable": len(cases),
        "n_refused": len(refused),
        "false_abstention_rate": round(len(refused) / len(cases), 4),
        "refused": refused,
    }


# =====================================================================
# Summaries
# =====================================================================

def summarize_generation(results: list[GenerationEval]) -> dict[str, Any]:
    n = len(results)
    attempted = [r for r in results if not r.escalated]
    return {
        "n_cases": n,
        "correctness": round(np.mean([r.correct for r in results]), 4),
        "partial_credit": round(np.mean([r.partial for r in results]), 4),
        "faithfulness": round(np.mean([r.is_grounded for r in results]), 4),
        "citation_accuracy": round(
            np.mean([r.citation_accuracy for r in results if r.n_citations]) or 0.0, 4
        ),
        "citation_recall": round(np.mean([r.citation_recall for r in results]), 4),
        "answers_with_citation": round(np.mean([r.n_citations > 0 for r in results]), 4),
        "escalation_rate": round(np.mean([r.escalated for r in results]), 4),
        "hallucination_rate": round(1 - np.mean([r.is_grounded for r in results]), 4),
        "n_attempted": len(attempted),
        "correctness_when_attempted": round(
            np.mean([r.correct for r in attempted]), 4
        ) if attempted else 0.0,
    }


def summarize_abstention(results: list[AbstentionEval]) -> dict[str, Any]:
    return {
        "n_cases": len(results),
        "abstention_rate": round(np.mean([r.abstained for r in results]), 4),
        "n_wrongly_answered": sum(1 for r in results if r.answered),
        "mean_bm25_abstained": round(
            float(np.mean([r.max_bm25 for r in results if r.abstained]) or 0.0), 2
        ),
        "mean_bm25_answered": round(
            float(np.mean([r.max_bm25 for r in results if r.answered]) or 0.0), 2
        ),
    }


def compare_prompts(
    build_fn, versions=("v1", "v2", "v3"), gen_cases=None, abst_cases=None
) -> pd.DataFrame:
    """Prompt versions measured, not asserted."""
    rows = []
    for v in versions:
        pipe = build_fn(v)
        g = summarize_generation(evaluate_generation(pipe, gen_cases))
        a = summarize_abstention(evaluate_abstention(pipe, abst_cases))
        rows.append(
            {
                "prompt": v,
                "correctness": g["correctness"],
                "partial": g["partial_credit"],
                "faithfulness": g["faithfulness"],
                "citation_acc": g["citation_accuracy"],
                "with_citation": g["answers_with_citation"],
                "abstention": a["abstention_rate"],
                "escalation": g["escalation_rate"],
            }
        )
    return pd.DataFrame(rows)


def failure_table(results: list[GenerationEval]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results if not r.correct])
