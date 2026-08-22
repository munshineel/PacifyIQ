"""Citation verification and grounding checks.

A citation the model emitted is a claim, not a fact. Two things must be checked:

1. Does the cited source exist, and was it actually in the context?
2. Is the answer's content supported by that context, or did the model add
   something from parametric knowledge?

Check 2 is deliberately lexical rather than model-based. An LLM judge scoring
LLM output has correlated failure modes, and a token-overlap check is cheap,
deterministic, and catches the case that matters most here: fabricated numbers.
Every planted hallucination trap in the corpus is numeric - 144Hz vs 75Hz,
IP68 vs IP53 - and a number in the answer that appears nowhere in the context
is unambiguous evidence of fabrication.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.llm.structured import Citation, SupportResponse

RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
RE_CODE = re.compile(
    r"\b(?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)[-_][\w\-]*\b", re.I
)
RE_WORD = re.compile(r"[a-z][a-z\-]{3,}")

# Numbers so common they carry no evidential weight.
TRIVIAL_NUMBERS = {"0", "1", "2", "100"}


@dataclass
class GroundingReport:
    """Whether an answer is supported by the context it was given."""

    n_citations: int = 0
    n_verified: int = 0
    n_fabricated: int = 0
    fabricated: list[str] = field(default_factory=list)

    numbers_in_answer: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    codes_in_answer: list[str] = field(default_factory=list)
    unsupported_codes: list[str] = field(default_factory=list)

    token_overlap: float = 0.0
    has_citation: bool = False
    is_abstention: bool = False

    @property
    def citation_accuracy(self) -> float:
        return self.n_verified / self.n_citations if self.n_citations else 0.0

    @property
    def is_grounded(self) -> bool:
        """Grounded means: cited, citations real, and no invented figures.

        An abstention is grounded by definition - it asserts nothing, so there
        is nothing to support. Requiring a citation from "I don't have
        documentation covering that" scored correct refusals as hallucinations,
        which inverted the metric exactly where it mattered most.
        """
        if self.is_abstention:
            return not self.unsupported_numbers and not self.unsupported_codes
        return (
            self.has_citation
            and self.n_fabricated == 0
            and not self.unsupported_numbers
            and not self.unsupported_codes
        )

    @property
    def hallucination_flags(self) -> list[str]:
        flags = []
        if self.n_fabricated:
            flags.append(f"fabricated_citation({self.n_fabricated})")
        if self.unsupported_numbers:
            flags.append(f"unsupported_number({','.join(self.unsupported_numbers[:3])})")
        if self.unsupported_codes:
            flags.append(f"unsupported_code({','.join(self.unsupported_codes[:3])})")
        if not self.has_citation and not self.is_abstention:
            flags.append("no_citation")
        return flags

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citation_accuracy"] = round(self.citation_accuracy, 3)
        d["is_grounded"] = self.is_grounded
        d["hallucination_flags"] = self.hallucination_flags
        return d


def verify_citations(
    citations: list[Citation], available: list[str]
) -> tuple[int, list[str]]:
    """Check each citation against the sources actually placed in the context.

    Matching is on document reference and section, ignoring page, because a
    section can span pages and a page-level mismatch is a formatting slip
    rather than a fabrication.
    """
    keys = set()
    for a in available:
        m = re.match(r"([A-Z]{3}-[A-Z0-9]{2,5}-\d{3})(?:,\s*p\.(\d+))?(?:,\s*(S\d+))?", a)
        if m:
            keys.add((m.group(1), m.group(3)))
            keys.add((m.group(1), None))

    verified, fabricated = 0, []
    for c in citations:
        if (c.doc_ref, c.section) in keys or (c.doc_ref, None) in keys:
            c.verified = True
            verified += 1
        else:
            fabricated.append(str(c))
    return verified, fabricated


def check_grounding(
    response: SupportResponse, context_text: str, context_citations: list[str]
) -> GroundingReport:
    """Full grounding check for one answer."""
    rep = GroundingReport(is_abstention=response.is_abstention)
    ctx_low = context_text.lower()

    rep.n_citations = len(response.citations)
    rep.has_citation = rep.n_citations > 0
    rep.n_verified, rep.fabricated = verify_citations(
        response.citations, context_citations
    )
    rep.n_fabricated = len(rep.fabricated)

    # Numbers: the highest-signal hallucination check for this corpus.
    ctx_numbers = {n.replace(",", "") for n in RE_NUMBER.findall(context_text)}
    for n in RE_NUMBER.findall(response.answer):
        clean = n.replace(",", "")
        if clean in TRIVIAL_NUMBERS:
            continue
        rep.numbers_in_answer.append(n)
        if clean not in ctx_numbers:
            rep.unsupported_numbers.append(n)

    ctx_codes = {c.upper() for c in RE_CODE.findall(context_text)}
    for c in RE_CODE.findall(response.answer):
        rep.codes_in_answer.append(c.upper())
        if c.upper() not in ctx_codes:
            rep.unsupported_codes.append(c.upper())

    ans_words = set(RE_WORD.findall(response.answer.lower()))
    ctx_words = set(RE_WORD.findall(ctx_low))
    rep.token_overlap = (
        len(ans_words & ctx_words) / len(ans_words) if ans_words else 0.0
    )
    return rep


def format_citations(citations: list[Citation], titles: dict[str, str] | None = None) -> str:
    """Render citations for display in the UI."""
    if not citations:
        return ""
    titles = titles or {}
    lines = []
    for c in citations:
        title = titles.get(c.doc_ref, "")
        label = f"{title} ({c})" if title else str(c)
        mark = "" if c.verified else "  [unverified]"
        lines.append(f"  - {label}{mark}")
    return "Sources:\n" + "\n".join(lines)


if __name__ == "__main__":
    from src.llm.structured import parse_response

    ctx = (
        "[1] SOURCE: POL-RET-002, p.1, S2\n"
        "Opened consumer electronics may be returned within 14 calendar days of "
        "delivery. Sealed items may be returned within 30 calendar days."
    )
    ctx_cites = ["POL-RET-002, p.1, S2"]

    cases = [
        ('{"answer":"You have 14 calendar days for opened electronics.",'
         '"citations":["POL-RET-002, p.1, S2"],"confidence":0.9,'
         '"needs_escalation":false}', "grounded"),
        ('{"answer":"You have 45 days to return it.",'
         '"citations":["POL-RET-002, p.1, S2"],"confidence":0.9,'
         '"needs_escalation":false}', "fabricated number"),
        ('{"answer":"You have 14 days.","citations":["POL-XYZ-999, p.7, S3"],'
         '"confidence":0.9,"needs_escalation":false}', "fabricated citation"),
        ('{"answer":"You have 14 days.","citations":[],"confidence":0.9,'
         '"needs_escalation":false}', "no citation"),
        ('{"answer":"Error THRM-88 applies.","citations":["POL-RET-002, p.1, S2"],'
         '"confidence":0.9,"needs_escalation":false}', "fabricated code"),
    ]
    for text, label in cases:
        r = parse_response(text)
        g = check_grounding(r, ctx, ctx_cites)
        print(f"\n{label}")
        print(f"  grounded={g.is_grounded}  citation_accuracy={g.citation_accuracy:.2f}"
              f"  overlap={g.token_overlap:.2f}")
        if g.hallucination_flags:
            print(f"  flags={g.hallucination_flags}")
