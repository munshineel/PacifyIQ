"""Structured output.

Models return prose by default; the pipeline needs typed fields it can route
on. Three layers of defence, because each fails differently:

1. Ask for JSON in the prompt          — usually works
2. Request provider JSON mode          — guarantees syntax, not schema
3. Parse, repair, validate here        — the only layer that guarantees fields

Even with provider JSON mode, validation is mandatory. Schema-valid output can
still be semantically wrong: a citation pointing at a page that does not exist
is perfectly well-formed JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Common malformations, in the order worth trying.
RE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
RE_OBJECT = re.compile(r"\{.*\}", re.S)
RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
RE_SINGLE_QUOTE_KEY = re.compile(r"'(\w+)'\s*:")
RE_SINGLE_QUOTE_VAL = re.compile(r":\s*'([^']*)'")


@dataclass
class Citation:
    """A pointer into the knowledge base. Validated against the real index."""

    doc_ref: str                 # e.g. POL-RET-002
    page: int | None = None
    section: str | None = None
    raw: str = ""
    verified: bool = False

    def __str__(self) -> str:
        parts = [self.doc_ref]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RE_CITATION = re.compile(
    r"([A-Z]{3}-[A-Z0-9]{2,5}-\d{3})"        # POL-RET-002 / MAN-PB14-001 / MAN-PV27-001
    r"(?:\s*,?\s*p\.?\s*(\d+))?"             # optional page
    r"(?:\s*,?\s*(S\d+(?:\.\d+)?))?",        # optional section
    re.I,
)


def parse_citation(text: str) -> Citation | None:
    m = RE_CITATION.search(str(text))
    if not m:
        return None
    return Citation(
        doc_ref=m.group(1).upper(),
        page=int(m.group(2)) if m.group(2) else None,
        section=m.group(3).upper() if m.group(3) else None,
        raw=str(text).strip(),
    )


@dataclass
class SupportResponse:
    """The typed output the rest of the system routes on."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    needs_escalation: bool = False
    escalation_reason: str | None = None

    # populated by validation, not by the model
    parse_ok: bool = True
    parse_errors: list[str] = field(default_factory=list)
    repaired: bool = False

    @property
    def is_abstention(self) -> bool:
        markers = (
            "don't have", "do not have", "not covered", "no documentation",
            "cannot find", "does not address", "unable to find",
        )
        low = self.answer.lower()
        return any(m in low for m in markers)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        d["is_abstention"] = self.is_abstention
        return d


# =====================================================================
# Parsing
# =====================================================================

def _try_load(text: str) -> dict | None:
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def repair_json(text: str) -> tuple[dict | None, list[str]]:
    """Attempt increasingly aggressive fixes. Returns (parsed, steps_applied)."""
    steps: list[str] = []

    v = _try_load(text)
    if v is not None:
        return v, steps

    m = RE_FENCE.search(text)
    if m:
        steps.append("stripped code fence")
        v = _try_load(m.group(1))
        if v is not None:
            return v, steps
        text = m.group(1)

    m = RE_OBJECT.search(text)
    if m:
        steps.append("extracted object from surrounding prose")
        candidate = m.group(0)
        v = _try_load(candidate)
        if v is not None:
            return v, steps
        text = candidate

    fixed = RE_TRAILING_COMMA.sub(r"\1", text)
    if fixed != text:
        steps.append("removed trailing comma")
        v = _try_load(fixed)
        if v is not None:
            return v, steps
        text = fixed

    fixed = RE_SINGLE_QUOTE_KEY.sub(r'"\1":', text)
    # Values are single-quoted too when a model emits Python dict syntax, so
    # converting keys alone still leaves invalid JSON.
    fixed = RE_SINGLE_QUOTE_VAL.sub(r': "\1"', fixed)
    if fixed != text:
        steps.append("converted single quotes to double")
        v = _try_load(fixed)
        if v is not None:
            return v, steps

    return None, steps + ["unrecoverable"]


def parse_response(text: str) -> SupportResponse:
    """Parse model output into the typed schema, repairing what can be repaired.

    A completely unparseable response is not an error to raise - it is a signal
    to escalate. Raising here would take down the request; escalating hands it
    to a human with context.
    """
    data, steps = repair_json(text)

    if data is None:
        # Prose fallback: keep the text, extract any citations, force escalation.
        cites = [c for c in (parse_citation(x) for x in RE_CITATION.findall(text)) if c]
        return SupportResponse(
            answer=text.strip()[:1500],
            citations=cites,
            confidence=0.0,
            needs_escalation=True,
            escalation_reason="unparseable_model_output",
            parse_ok=False,
            parse_errors=steps,
            repaired=bool(steps),
        )

    errors: list[str] = []

    answer = data.get("answer") or data.get("response") or data.get("text") or ""
    if not isinstance(answer, str):
        answer = str(answer)
        errors.append("answer was not a string")
    if not answer.strip():
        errors.append("empty answer")

    raw_cites = data.get("citations") or data.get("sources") or []
    if isinstance(raw_cites, str):
        raw_cites = [raw_cites]
        errors.append("citations was a string, expected a list")
    citations = []
    for rc in raw_cites:
        if isinstance(rc, dict):
            rc = rc.get("citation") or rc.get("source") or " ".join(map(str, rc.values()))
        c = parse_citation(str(rc))
        if c:
            citations.append(c)
        else:
            errors.append(f"unparseable citation: {str(rc)[:60]}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("confidence was not numeric")
    if not 0.0 <= confidence <= 1.0:
        errors.append(f"confidence {confidence} outside [0,1], clipped")
        confidence = max(0.0, min(1.0, confidence))

    escalate = bool(data.get("needs_escalation", False))
    reason = data.get("escalation_reason")

    return SupportResponse(
        answer=answer.strip(),
        citations=citations,
        confidence=confidence,
        needs_escalation=escalate,
        escalation_reason=str(reason) if reason else None,
        parse_ok=not errors,
        parse_errors=errors,
        repaired=bool(steps),
    )


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_escalation": {"type": "boolean"},
        "escalation_reason": {"type": ["string", "null"]},
    },
    "required": ["answer", "citations", "confidence", "needs_escalation"],
}


if __name__ == "__main__":
    cases = [
        ('{"answer":"14 days","citations":["POL-RET-002, p.1, S2"],'
         '"confidence":0.9,"needs_escalation":false}', "clean"),
        ('```json\n{"answer":"14 days","citations":[],"confidence":0.8,'
         '"needs_escalation":false}\n```', "code fence"),
        ('Sure! Here you go: {"answer":"14 days","citations":["POL-RET-002 p.1 S2"],'
         '"confidence":0.9,"needs_escalation":false} Hope that helps.', "prose wrapper"),
        ('{"answer":"14 days","citations":["POL-RET-002"],"confidence":0.9,}', "trailing comma"),
        ("{'answer':'14 days','citations':[],'confidence':0.5,"
         "'needs_escalation':false}", "single quotes"),
        ('{"answer":"x","citations":[],"confidence":1.7,"needs_escalation":false}',
         "confidence out of range"),
        ("The return window is 14 days per POL-RET-002, p.1, S2.", "prose only"),
        ("complete garbage {{{", "unrecoverable"),
    ]
    for text, label in cases:
        r = parse_response(text)
        print(f"\n{label}")
        print(f"  parse_ok={r.parse_ok} repaired={r.repaired} escalate={r.needs_escalation}")
        print(f"  answer={r.answer[:52]!r}")
        print(f"  citations={[str(c) for c in r.citations]}  confidence={r.confidence}")
        if r.parse_errors:
            print(f"  errors={r.parse_errors}")
