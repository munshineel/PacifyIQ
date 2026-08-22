"""Guardrail contract.

A guardrail is a **rule**, not a prompt instruction. Each one is a small pure
function that inspects a specific artefact and returns a verdict. That matters
for three reasons:

1. A rule can be unit-tested. A prompt cannot.
2. A rule holds regardless of what the model decided. A prompt is a request.
3. A rule states *why* it fired, which the escalation package needs.

This module is deliberately separate from `src/agent` and `src/rag`. Guardrails
must be able to veto those layers, so they cannot depend on them.

WHAT THIS IS NOT
----------------
These checks reduce risk. They do not make the system safe. Every rule here is
a pattern or a threshold, and both can be evaded by an input nobody anticipated.
See `reports/safety_report.md` for the measured coverage and the honest gaps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How the pipeline must respond when a rule fires."""

    INFO = "info"           # log it, continue
    CAUTION = "caution"     # continue, but attach a caveat and lower confidence
    ESCALATE = "escalate"   # a human must handle this
    BLOCK = "block"         # refuse; do not generate, do not escalate a payload


# Ordered worst-first for aggregation.
SEVERITY_RANK = {
    Severity.BLOCK: 3, Severity.ESCALATE: 2,
    Severity.CAUTION: 1, Severity.INFO: 0,
}


class Stage(str, Enum):
    INPUT = "input"           # before anything is retrieved or generated
    EVIDENCE = "evidence"     # after retrieval and tools, before generation
    OUTPUT = "output"         # after generation, before the customer sees it
    ACTION = "action"         # before a tool with side effects runs


@dataclass
class Finding:
    """One rule firing."""

    rule: str
    stage: Stage
    severity: Severity
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)
    # What the customer is told. Deliberately vague on security findings: a
    # detailed refusal is a free oracle for anyone probing the system.
    customer_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["severity"] = self.severity.value
        return d

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.rule}: {self.reason}"


@dataclass
class GuardrailVerdict:
    """Aggregate of every rule that fired at one stage."""

    stage: Stage
    findings: list[Finding] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        if not self.findings:
            return Severity.INFO
        return max(self.findings, key=lambda f: SEVERITY_RANK[f.severity]).severity

    @property
    def blocked(self) -> bool:
        return self.severity == Severity.BLOCK

    @property
    def must_escalate(self) -> bool:
        return self.severity in (Severity.BLOCK, Severity.ESCALATE)

    @property
    def rules_fired(self) -> list[str]:
        return [f.rule for f in self.findings]

    @property
    def primary(self) -> Finding | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: SEVERITY_RANK[f.severity])

    def reason(self) -> str:
        p = self.primary
        return p.reason if p else "no findings"

    def customer_message(self) -> str | None:
        p = self.primary
        return p.customer_message if p else None

    def caveats(self) -> list[str]:
        return [f.reason for f in self.findings if f.severity == Severity.CAUTION]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "severity": self.severity.value,
            "blocked": self.blocked,
            "must_escalate": self.must_escalate,
            "rules_fired": self.rules_fired,
            "findings": [f.to_dict() for f in self.findings],
        }

    def merge(self, other: "GuardrailVerdict") -> "GuardrailVerdict":
        return GuardrailVerdict(stage=self.stage,
                                findings=self.findings + other.findings)
