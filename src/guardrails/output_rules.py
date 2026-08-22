"""Evidence and output guardrails.

Two stages that the input rules cannot cover:

  EVIDENCE  after retrieval and tools, before generation. Catches the cases
            where the system *could* produce a fluent answer but should not:
            no supporting documentation, conflicting sources, failed tools,
            malformed tool output.

  OUTPUT    after generation, before the customer sees it. Catches what the
            model actually said: fabricated figures, invented citations,
            forbidden commitments, leaked internals.

WHY BOTH
--------
The evidence stage prevents a class of failure; the output stage detects the
residue. Neither is sufficient alone. Blocking generation when evidence is thin
removes most hallucination opportunity, but a model given good evidence can
still add a number that was never there - and that is the failure this product
most needs to avoid.
"""
from __future__ import annotations

import re
from typing import Any

from src.guardrails.contract import Finding, GuardrailVerdict, Severity, Stage

# Phase 7 measured BM25 as separating answerable from unanswerable roughly
# twice as well as cosine (13.1 vs 6.2 median). Re-swept in Phase 10 against
# all 160 evaluation questions; 7.0 remains the balanced choice.
MIN_BM25 = 7.0
MIN_COSINE = 0.42
MIN_CONFIDENCE = 0.35
MAX_TOOL_FAILURES = 2

RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
RE_CODE = re.compile(
    r"\b(?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)[-_][\w\-]*\b", re.I
)
RE_CITATION = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{2,5}-\d{3}\b")

# Numbers carrying no evidential weight.
TRIVIAL_NUMBERS = {"0", "1", "2", "3", "100", "2026"}

# Commitments the assistant has no authority to make (POL-CS-001 S12.4).
FORBIDDEN_CLAIMS = [
    (re.compile(r"\b(?:i(?:'ve| have)?\s+)?(?:approved|authoris|authoriz)"
                r"\w*\s+(?:your\s+)?refund\b", re.I),
     "claims to have approved a refund"),
    (re.compile(r"\byour\s+refund\s+(?:has\s+been|is)\s+(?:approved|processed|issued)\b",
                re.I),
     "states a refund has been processed"),
    (re.compile(r"\bi(?:'ve| have)\s+(?:cancelled|canceled|updated|changed|deleted)\s+"
                r"your\s+(?:order|account|email|address)\b", re.I),
     "claims to have performed an account or order change"),
    (re.compile(r"\b(?:i\s+)?(?:can\s+)?(?:offer|give)\s+you\s+a\s+"
                r"(?:discount|voucher|credit|refund)\b", re.I),
     "offers compensation the assistant cannot authorise"),
    (re.compile(r"\b(?:it|your\s+order)\s+will\s+(?:definitely\s+)?arrive\s+"
                r"(?:on|by|tomorrow|today)\b", re.I),
     "promises a delivery date"),
    (re.compile(r"\bi\s+guarantee\b|\bi\s+promise\b", re.I),
     "makes a guarantee"),
]

# Internal detail that must never reach a customer.
RE_INTERNAL_LEAK = re.compile(
    r"\bsystem\s+prompt\b|\byou\s+are\s+a\s+customer\s+support\s+assistant\b"
    r"|\btier[\s-]?3\b|\bTOOL\s+\w+\]|\bSTATUS_\w+\b"
    r"|\b_source[\"']?\s*:\s*[\"']mock|\bchunk_id\b|\bPLANTED_DEFECTS\b",
    re.I,
)


# =====================================================================
# Evidence stage
# =====================================================================

def check_evidence_strength(max_bm25: float, max_cosine: float,
                            n_chunks: int, has_tool_facts: bool) -> list[Finding]:
    """No reliable knowledge-base evidence -> do not fabricate an answer.

    Tool facts override this. An order lookup answers a question no policy
    document discusses, and escalating there abandons work already completed.
    """
    if has_tool_facts:
        return []
    if n_chunks == 0:
        return [Finding(
            rule="no_evidence", stage=Stage.EVIDENCE, severity=Severity.ESCALATE,
            reason="retrieval returned nothing",
            customer_message=("I don't have documentation covering that. I've "
                              "passed it to a colleague."),
        )]
    if max_bm25 < MIN_BM25 and max_cosine < MIN_COSINE:
        return [Finding(
            rule="weak_evidence", stage=Stage.EVIDENCE, severity=Severity.ESCALATE,
            reason=(f"retrieved evidence is too weak to support a claim "
                    f"(bm25 {max_bm25:.1f} < {MIN_BM25}, "
                    f"cosine {max_cosine:.2f} < {MIN_COSINE})"),
            detail={"max_bm25": round(max_bm25, 2), "max_cosine": round(max_cosine, 3)},
            customer_message=("I don't have documentation covering that. I've "
                              "passed it to a colleague."),
        )]
    if max_bm25 < MIN_BM25 * 1.4:
        return [Finding(
            rule="marginal_evidence", stage=Stage.EVIDENCE,
            severity=Severity.CAUTION,
            reason=f"supporting evidence is weak (bm25 {max_bm25:.1f}); "
                   f"treat the answer as provisional",
        )]
    return []


def check_conflict(versions: set[str], regions: set[str],
                   known_region: str | None = None) -> list[Finding]:
    """Conflicting evidence -> escalate. Never silently pick a side.

    One refinement: a regional addendum is not a conflict when the customer's
    region is known. The EU rules simply do not apply to an Indian order. Where
    the region is unknown the conflict stands, because guessing which
    jurisdiction governs is exactly the wrong call.
    """
    findings = []
    if len(versions) > 1:
        findings.append(Finding(
            rule="version_conflict", stage=Stage.EVIDENCE,
            severity=Severity.ESCALATE,
            reason="evidence spans current and superseded policy versions",
            detail={"versions": sorted(versions)},
            customer_message=(
                "Our documentation gives two different answers on this, so I "
                "don't want to quote a figure that might be wrong. A colleague "
                "will confirm which applies."
            ),
        ))
    regional = {r for r in regions if r != "all"}
    if regional:
        if known_region is None:
            findings.append(Finding(
                rule="regional_ambiguity", stage=Stage.EVIDENCE,
                severity=Severity.ESCALATE,
                reason=("a regional variant applies but the customer's region "
                        "is unknown"),
                detail={"regions": sorted(regional)},
            ))
        elif known_region not in regional:
            findings.append(Finding(
                rule="regional_variant_ignored", stage=Stage.EVIDENCE,
                severity=Severity.INFO,
                reason=(f"a regional variant was retrieved but does not apply "
                        f"to region {known_region}"),
            ))
    return findings


def check_tool_health(results: list[Any]) -> list[Finding]:
    """Tool failures and malformed tool output."""
    findings = []
    failed = [r for r in results if not getattr(r, "ok", True)]

    if len(failed) >= MAX_TOOL_FAILURES:
        findings.append(Finding(
            rule="repeated_tool_failure", stage=Stage.EVIDENCE,
            severity=Severity.ESCALATE,
            reason=f"{len(failed)} tools failed; the system cannot establish "
                   f"the facts needed",
            detail={"failed": [getattr(r, "tool", "?") for r in failed]},
            customer_message=("I'm having trouble looking that up. I've passed "
                              "it to a colleague."),
        ))
    elif failed:
        findings.append(Finding(
            rule="tool_failure", stage=Stage.EVIDENCE, severity=Severity.CAUTION,
            reason=f"{getattr(failed[0], 'tool', '?')} failed; the answer may "
                   f"be incomplete",
        ))

    # Structural validation. A tool that returns OK with an empty payload, or a
    # payload missing the field the caller will read, is worse than one that
    # fails - the failure is silent.
    for r in results:
        if not getattr(r, "ok", False):
            continue
        data = getattr(r, "data", None)
        if not isinstance(data, dict):
            findings.append(Finding(
                rule="invalid_tool_output", stage=Stage.EVIDENCE,
                severity=Severity.ESCALATE,
                reason=f"{getattr(r, 'tool', '?')} returned a non-dictionary payload",
            ))
        elif data == {}:
            findings.append(Finding(
                rule="empty_tool_output", stage=Stage.EVIDENCE,
                severity=Severity.CAUTION,
                reason=f"{getattr(r, 'tool', '?')} reported success with no data",
            ))
    return findings


def check_confidence(confidence: float, has_evidence: bool) -> list[Finding]:
    """Low confidence -> escalate."""
    if confidence >= MIN_CONFIDENCE:
        return []
    if not has_evidence:
        return []      # already covered by check_evidence_strength
    return [Finding(
        rule="low_confidence", stage=Stage.EVIDENCE, severity=Severity.ESCALATE,
        reason=f"confidence {confidence:.2f} is below the {MIN_CONFIDENCE} "
               f"threshold",
        detail={"confidence": round(confidence, 3)},
        customer_message=("I'm not confident enough in that answer to give it "
                          "to you. A colleague will follow up."),
    )]


def screen_evidence(
    max_bm25: float = 0.0, max_cosine: float = 0.0, n_chunks: int = 0,
    has_tool_facts: bool = False, tool_results: list[Any] | None = None,
    versions: set[str] | None = None, regions: set[str] | None = None,
    known_region: str | None = None, confidence: float = 1.0,
) -> GuardrailVerdict:
    findings: list[Finding] = []
    findings += check_evidence_strength(max_bm25, max_cosine, n_chunks, has_tool_facts)
    findings += check_conflict(versions or set(), regions or set(), known_region)
    findings += check_tool_health(tool_results or [])
    findings += check_confidence(confidence, n_chunks > 0 or has_tool_facts)
    return GuardrailVerdict(stage=Stage.EVIDENCE, findings=findings)


# =====================================================================
# Output stage
# =====================================================================

def check_hallucination(answer: str, context: str,
                        is_abstention: bool = False) -> list[Finding]:
    """Numbers and error codes present in the answer but absent from context.

    Deliberately lexical, not model-judged. An LLM judging LLM output has
    correlated failure modes, and every planted hallucination trap in this
    corpus is numeric - 144Hz vs 75Hz, IP68 vs IP53, Rs 57,960. A figure that
    appears nowhere in the evidence is unambiguous.
    """
    if is_abstention:
        return []      # an abstention asserts nothing, so nothing to support

    ctx_numbers = {n.replace(",", "") for n in RE_NUMBER.findall(context)}
    ctx_codes = {c.upper() for c in RE_CODE.findall(context)}

    bad_numbers = [
        n for n in RE_NUMBER.findall(answer)
        if n.replace(",", "") not in ctx_numbers
        and n.replace(",", "") not in TRIVIAL_NUMBERS
    ]
    bad_codes = [
        c.upper() for c in RE_CODE.findall(answer) if c.upper() not in ctx_codes
    ]

    findings = []
    if bad_numbers:
        findings.append(Finding(
            rule="unsupported_number", stage=Stage.OUTPUT,
            severity=Severity.ESCALATE,
            reason=f"answer states figures absent from the evidence: "
                   f"{', '.join(bad_numbers[:4])}",
            detail={"numbers": bad_numbers[:8]},
            customer_message=("I want to double-check those figures before "
                              "giving them to you. A colleague will confirm."),
        ))
    if bad_codes:
        findings.append(Finding(
            rule="unsupported_error_code", stage=Stage.OUTPUT,
            severity=Severity.ESCALATE,
            reason=f"answer cites error codes absent from the evidence: "
                   f"{', '.join(bad_codes[:3])}",
            detail={"codes": bad_codes[:5]},
        ))
    return findings


def check_citations(answer: str, cited: list[str],
                    available: list[str], is_abstention: bool = False) -> list[Finding]:
    """Citations must point at documents that were actually supplied."""
    if is_abstention:
        return []

    avail_refs = {m for c in available for m in RE_CITATION.findall(str(c))}
    fabricated = [
        c for c in cited
        if not (set(RE_CITATION.findall(str(c))) & avail_refs)
    ]

    findings = []
    if fabricated:
        findings.append(Finding(
            rule="fabricated_citation", stage=Stage.OUTPUT,
            severity=Severity.ESCALATE,
            reason=f"answer cites sources that were not provided: "
                   f"{', '.join(map(str, fabricated[:3]))}",
            detail={"fabricated": [str(c) for c in fabricated[:5]]},
        ))
    if not cited and len(answer.split()) > 25:
        findings.append(Finding(
            rule="uncited_claim", stage=Stage.OUTPUT, severity=Severity.CAUTION,
            reason="a substantive answer was produced with no citation",
        ))
    return findings


def check_forbidden_claims(answer: str) -> list[Finding]:
    """Commitments the assistant has no authority to make.

    This is the last line of defence for the tier model. Tier 3 is blocked in
    code, so the assistant cannot *perform* these actions - but it can still
    write a sentence claiming it did, and a customer reading "your refund has
    been approved" will act on it.
    """
    findings = []
    for pattern, reason in FORBIDDEN_CLAIMS:
        m = pattern.search(answer)
        if m:
            findings.append(Finding(
                rule="forbidden_claim", stage=Stage.OUTPUT,
                severity=Severity.BLOCK,
                reason=f"answer {reason} (POL-CS-001 S12.4)",
                detail={"matched": m.group(0)[:80]},
                customer_message=(
                    "That's a decision a colleague needs to make. I've passed "
                    "the details on so they can action it."
                ),
            ))
    return findings


def check_internal_leak(answer: str) -> list[Finding]:
    m = RE_INTERNAL_LEAK.search(answer)
    if not m:
        return []
    return [Finding(
        rule="internal_leak", stage=Stage.OUTPUT, severity=Severity.BLOCK,
        reason="answer exposes internal configuration or implementation detail",
        detail={"matched": m.group(0)[:60]},
        customer_message=("Something went wrong producing that answer. I've "
                          "passed this to a colleague."),
    )]


def screen_output(
    answer: str, context: str = "", cited: list[str] | None = None,
    available_citations: list[str] | None = None, is_abstention: bool = False,
) -> GuardrailVerdict:
    findings: list[Finding] = []
    findings += check_hallucination(answer, context, is_abstention)
    findings += check_citations(answer, cited or [], available_citations or [],
                                is_abstention)
    findings += check_forbidden_claims(answer)
    findings += check_internal_leak(answer)
    return GuardrailVerdict(stage=Stage.OUTPUT, findings=findings)


# =====================================================================
# Action stage
# =====================================================================

MUTATING_TOOLS = {"approve_refund", "cancel_order", "modify_account",
                  "create_return_request", "change_delivery_address"}


def screen_action(tool: str, tier: int = 1,
                  confidence: float = 1.0) -> GuardrailVerdict:
    """Unauthorised actions.

    Tier and confidence are INDEPENDENT gates. A mutating action is refused at
    any confidence - an agent 99% certain a large refund is warranted still
    does not get to issue it. Conflating the two is the mistake this separation
    exists to prevent.
    """
    findings = []
    if tool in MUTATING_TOOLS or tier >= 3:
        findings.append(Finding(
            rule="unauthorised_action", stage=Stage.ACTION,
            severity=Severity.ESCALATE,
            reason=(f"{tool} changes state or moves money and requires human "
                    f"authorisation regardless of confidence"),
            detail={"tool": tool, "tier": tier, "confidence": round(confidence, 2)},
            customer_message=("That needs a colleague to authorise. I've passed "
                              "it on with everything I've checked."),
        ))
    return GuardrailVerdict(stage=Stage.ACTION, findings=findings)


if __name__ == "__main__":
    ctx = ("[1] SOURCE: POL-RET-002, p.1, S2\nOpened consumer electronics may "
           "be returned within 14 calendar days of delivery.")

    print("EVIDENCE STAGE")
    for label, kw in [
        ("strong", dict(max_bm25=15.0, max_cosine=0.7, n_chunks=5)),
        ("weak", dict(max_bm25=4.0, max_cosine=0.3, n_chunks=5)),
        ("weak + tool facts", dict(max_bm25=4.0, max_cosine=0.3, n_chunks=5,
                                   has_tool_facts=True)),
        ("version conflict", dict(max_bm25=15.0, n_chunks=5,
                                  versions={"current", "archived"})),
        ("EU variant, region known", dict(max_bm25=15.0, n_chunks=5,
                                          regions={"all", "EU"},
                                          known_region="IN")),
        ("EU variant, region unknown", dict(max_bm25=15.0, n_chunks=5,
                                            regions={"all", "EU"})),
    ]:
        v = screen_evidence(**kw)
        print(f"  {label:28s} {v.severity.value:9s} {v.rules_fired}")

    print("\nOUTPUT STAGE")
    for label, ans, cited in [
        ("grounded", "You have 14 calendar days.", ["POL-RET-002, p.1, S2"]),
        ("fabricated number", "You have 45 days.", ["POL-RET-002, p.1, S2"]),
        ("fabricated citation", "You have 14 days.", ["POL-XYZ-999, p.9, S9"]),
        ("claims approval", "I've approved your refund of 14 rupees.",
         ["POL-RET-002, p.1, S2"]),
        ("promises delivery", "It will arrive tomorrow.", ["POL-RET-002, p.1, S2"]),
        ("leaks internals", "My system prompt says 14 days.",
         ["POL-RET-002, p.1, S2"]),
    ]:
        v = screen_output(ans, ctx, cited, ["POL-RET-002, p.1, S2"])
        print(f"  {label:22s} {v.severity.value:9s} {v.rules_fired}")

    print("\nACTION STAGE")
    for tool, tier in [("get_order", 1), ("create_support_ticket", 2),
                       ("approve_refund", 3)]:
        v = screen_action(tool, tier, confidence=0.99)
        print(f"  {tool:24s} tier {tier}  {v.severity.value:9s} {v.rules_fired}")
