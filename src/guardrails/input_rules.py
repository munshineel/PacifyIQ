"""Input guardrails.

Run before retrieval, tools or generation. Nothing here trusts the model,
because these rules exist precisely for inputs designed to manipulate it.

DESIGN NOTE ON REFUSAL MESSAGES
-------------------------------
Security findings return a deliberately vague customer message. A refusal that
names the rule it tripped is a free oracle: an attacker learns which phrasing
was detected and adjusts. "I can't help with that" is less satisfying and more
defensible.

DESIGN NOTE ON WHAT COUNTS AS AN ATTACK
---------------------------------------
Detecting "ignore previous instructions" is easy and nearly worthless - it
catches the naive case only. The harder and more valuable rules here are the
ones that catch *plausible-sounding* manipulation: a claimed employee ID, a
fabricated system message, a request framed as hypothetical. Those are what a
motivated person actually sends.
"""
from __future__ import annotations

import re
from typing import Any

from src.guardrails.contract import Finding, GuardrailVerdict, Severity, Stage

# =====================================================================
# Patterns
# =====================================================================

# 1. Direct instruction override.
RE_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|discard)\b[^.]{0,40}"
    r"\b(?:previous|prior|earlier|above|all|your|the)\b[^.]{0,30}"
    r"\b(?:instruction|rule|prompt|direction|guideline|constraint|polic)",
    re.I,
)

# 2. Role reassignment.
RE_ROLE_OVERRIDE = re.compile(
    r"\byou\s+are\s+(?:now|a|an)\b[^.]{0,60}"
    r"|\bact\s+as\b[^.]{0,50}"
    r"|\bpretend\s+(?:to\s+be|you)\b"
    r"|\bfrom\s+now\s+on\s+you\b"
    r"|\bno\s+(?:restrictions|rules|limits|guardrails)\b"
    r"|\bdeveloper\s+mode\b|\bDAN\b",
    re.I,
)

# 3. Fabricated authority. The message claims a privileged channel.
RE_FAKE_AUTHORITY = re.compile(
    r"^\s*(?:system|admin|root|developer|assistant)\s*[:>\]]"
    r"|\[\s*(?:system|admin|override|developer)[^\]]*\]"
    r"|<\s*(?:system|admin)[^>]*>"
    r"|\bnew\s+(?:policy|instruction|system\s+prompt)\b"
    r"|\bi\s+am\s+(?:a|an)\s+(?:pacify\s+)?(?:employee|admin|developer|engineer|manager)\b"
    r"|\bemployee\s+id\b|\bstaff\s+(?:code|id)\b"
    r"|\bmy\s+manager\s+(?:at|said)\b",
    re.I,
)

# 4. Prompt or schema extraction.
RE_EXTRACTION = re.compile(
    r"\b(?:print|show|reveal|repeat|output|display|tell\s+me|what\s+(?:is|are))\b"
    r"[^.]{0,40}"
    r"\b(?:system\s+prompt|initial\s+prompt|instruction|configuration|preamble|"
    r"tool\s+schema|function\s+definition|your\s+rules|everything\s+above)\b"
    r"|\brepeat\s+everything\b|\bverbatim\b.{0,30}\b(?:prompt|instruction)\b"
    # "what tools do you have access to" - asks for the tool inventory without
    # using any of the words above.
    r"|\b(?:what|which|list)\b[^.?]{0,30}\b(?:tools?|functions?|apis?|capabilit)"
    r"[^.?]{0,30}\b(?:do\s+you|you\s+have|access|available|schema)\b"
    r"|\blist\s+(?:their|your)\s+schemas?\b",
    re.I,
)

# 5. Data exfiltration - asking about people or records other than one's own.
RE_EXFILTRATION = re.compile(
    r"\b(?:list|show|give|dump|export|how\s+many)\b[^.]{0,40}"
    r"\b(?:all|every|other)\b[^.]{0,30}"
    r"\b(?:customer|order|user|account|email|record|transaction)s?\b"
    r"|\b(?:email|phone|address|name)\s+of\s+(?:the\s+)?(?:customer|user|person)\b"
    r"|\bcustomer\s+who\b|\bsomeone\s+else'?s?\s+(?:order|account)\b"
    r"|\bdatabase\s+schema\b|\bselect\s+\*\s+from\b"
    # Aggregate questions are exfiltration too: "how many customers in X" is a
    # business metric, not a support question, and answering it leaks scale.
    r"|\bhow\s+many\s+(?:customers?|users?|orders?|accounts?)\b"
    r"|\b(?:total|number\s+of)\s+(?:customers?|users?|orders?)\b",
    re.I,
)

# 6. SQL / code injection.
RE_SQL = re.compile(
    r"(?:'|\")\s*;\s*(?:drop|delete|update|insert)\b"
    r"|\bdrop\s+table\b|\bdelete\s+from\b|\bunion\s+select\b"
    r"|\bor\s+1\s*=\s*1\b|--\s*$",
    re.I | re.M,
)

# 7. Indirection - wrapping an instruction in another task.
RE_INDIRECTION = re.compile(
    r"\btranslate\b[^.]{0,60}\b(?:ignore|approve|override|instruction)\b"
    r"|\bsummari[sz]e\s+this\b[^.]{0,80}\b(?:assistant|approved|refund)\b"
    r"|\brepeat\s+after\s+me\b"
    r"|\bwrite\s+(?:a\s+)?(?:story|poem|script)\b[^.]{0,60}"
    r"\b(?:approve|refund|then\s+do\s+it)\b",
    re.I,
)

# 8. Hypothetical framing used to extract prohibited behaviour.
RE_HYPOTHETICAL = re.compile(
    r"\bhypothetical(?:ly)?\b[^.]{0,60}\b(?:approve|refund|override|no\s+restriction)\b"
    r"|\bif\s+you\s+could\b[^.]{0,50}\b(?:approve|refund|ignore)\b"
    r"|\bwhat\s+would\s+you\s+say\s+if\s+you\s+had\s+no\b"
    r"|\bfor\s+(?:testing|debug(?:ging)?)\s+purposes\b",
    re.I,
)

# 9. False premise asserted as established fact.
RE_FALSE_PREMISE = re.compile(
    r"\b(?:pretend|assume|suppose|imagine)\b[^.]{0,40}"
    r"\b(?:policy|window|days?|return|warranty|refund)\b"
    r"|\bsince\s+your\s+policy\s+says\b"
    r"|\byour\s+polic(?:y|ies)\s+(?:say|states?|allows?)\b"
    r"|\b(?:as|since)\s+(?:you|we)\s+(?:agreed|said|confirmed)\b",
    re.I,
)

# 10. Requests for commitments the assistant cannot make.
RE_UNAUTHORISED_ASK = re.compile(
    r"\b(?:give|get)\s+me\s+a\s+(?:discount|voucher|coupon|free)\b"
    r"|\bwaive\s+the\s+(?:fee|charge|restocking)\b"
    r"|\bpromise\s+me\b|\bguarantee\s+(?:it|delivery|that)\b"
    r"|\bconfirm\s+(?:the\s+)?refund\b|\bapprove\s+(?:my|the)\s+refund\b",
    re.I,
)

# 11. Identity-sensitive operations.
RE_IDENTITY_SENSITIVE = re.compile(
    r"\bchange\s+(?:my|the)\s+(?:email|phone|password|address|bank|account)\b"
    r"|\b(?:reset|recover)\s+(?:my\s+)?password\b"
    r"|\bdelete\s+(?:my\s+)?account\b"
    r"|\bsend\s+(?:my\s+)?refund\s+to\s+(?:a\s+)?(?:different|new|another)\b"
    r"|\b(?:different|new|another)\s+(?:bank\s+)?account\s+(?:number|details)\b"
    r"|\bskip\s+the\s+(?:otp|verification)\b|\bwithout\s+(?:the\s+)?otp\b"
    r"|\bverify\s+me\s+with\s+my\s+name\b"
    # Redirecting a refund to a supplied account number, and password resets on
    # a named account, are the two highest-value account-takeover vectors.
    r"|\bchange\s+the\s+(?:refund\s+)?bank\s+account\b"
    r"|\brefund\s+(?:bank\s+)?account\s+(?:for|to|number)\b"
    r"|\breset\s+the\s+password\s+on\b"
    r"|\baccount\s+CUS[-\s]?\d+\b",
    re.I,
)

# 12. PII that should not be in a support message at all.
PII_PATTERNS = {
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # `is` / `was` are included because people write "my otp is 998877" far
    # more often than "otp: 998877".
    "cvv": re.compile(r"\b(?:cvv|cvc|security\s+code)\s*(?:is\s*)?[:=]?\s*\d{3,4}\b",
                      re.I),
    "otp": re.compile(r"\b(?:otp|one[\s-]?time\s+password|code)\s*(?:is\s*)?[:=]?"
                      r"\s*\d{4,8}\b", re.I),
    "password": re.compile(r"\bpassword\s*(?:is\s*)?[:=]?\s*\S{4,}", re.I),
    "aadhaar": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "ifsc_account": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}

# Order references look like card numbers to a naive digit rule.
RE_ORDER_REF = re.compile(r"\bPAC[-\s]?2026[-\s]?\d{4,6}\b", re.I)

# Out-of-domain topics. Kept short and specific: an over-broad scope filter
# refuses legitimate questions, which is its own failure mode.
OUT_OF_DOMAIN = (
    "weather", "football", "cricket match", "election", "stock price",
    "recipe", "write me a poem", "write a python", "write me code",
    "who is the president", "capital of", "translate this to",
    "homework", "medical advice", "legal advice about",
)

# Phrasing that marks a QUESTION ABOUT a process rather than a REQUEST to
# perform it. "How do I delete my account?" is answerable from POL-CS-001 S7;
# "Delete my account" is an account change requiring verified identity.
#
# This is the same distinction the agent draws between an eligibility question
# and an action request (Phase 10). Applying it here keeps the guardrail from
# refusing questions the documentation answers - an over-broad security rule is
# its own failure mode, and one that is much harder to notice than a gap.
RE_PROCESS_QUESTION = re.compile(
    r"^\s*(?:how\s+(?:do|can|would)\s+i|what\s+(?:is|are|happens)|"
    r"can\s+(?:the\s+)?(?:ai|assistant|you|a\s+bot)\b|"
    r"is\s+it\s+possible|do\s+you|does\s+pacify|what's\s+the\s+process|"
    r"who\s+(?:can|do))",
    re.I,
)

# But an imperative or a supplied value turns a question into a request:
# "how do I change my email to x@y.com" is an attempt, not an enquiry.
RE_SUPPLIES_VALUE = re.compile(
    r"\bto\s+\S+@\S+\b|\bto\s+\d{6,}\b|\bnew\s+(?:email|number)\s+is\b",
    re.I,
)


def _is_process_question(text: str) -> bool:
    if RE_SUPPLIES_VALUE.search(text):
        return False
    return bool(RE_PROCESS_QUESTION.match(text.strip()))


GENERIC_SECURITY_MESSAGE = (
    "I can't help with that request. If you have a question about an order, "
    "a policy or a product, I'm happy to help with that."
)


# =====================================================================
# Rules
# =====================================================================

def _mask_orders(text: str) -> str:
    """Order references are business identifiers, not PII, and they trip the
    card-number rule. Remove them before scanning."""
    return RE_ORDER_REF.sub(" ", text)


def check_injection(text: str) -> list[Finding]:
    """Instruction override, role reassignment, fabricated authority."""
    findings = []
    checks = [
        (RE_OVERRIDE, "instruction_override",
         "message attempts to override system instructions"),
        (RE_ROLE_OVERRIDE, "role_override",
         "message attempts to reassign the assistant's role"),
        (RE_FAKE_AUTHORITY, "fabricated_authority",
         "message claims a privileged channel or identity that cannot be verified"),
        (RE_INDIRECTION, "indirect_injection",
         "message wraps an instruction inside another task"),
        (RE_HYPOTHETICAL, "hypothetical_framing",
         "message uses hypothetical framing to request prohibited behaviour"),
    ]
    for pattern, rule, reason in checks:
        m = pattern.search(text)
        if m:
            findings.append(Finding(
                rule=rule, stage=Stage.INPUT, severity=Severity.BLOCK,
                reason=reason,
                detail={"matched": m.group(0)[:80]},
                customer_message=GENERIC_SECURITY_MESSAGE,
            ))
    return findings


def check_extraction(text: str) -> list[Finding]:
    """Attempts to read the system prompt, tool schemas or other customers."""
    findings = []
    for pattern, rule, reason in [
        (RE_EXTRACTION, "prompt_extraction",
         "message attempts to extract system instructions or tool definitions"),
        (RE_EXFILTRATION, "data_exfiltration",
         "message requests data belonging to other customers"),
        (RE_SQL, "sql_injection",
         "message contains SQL-like syntax"),
    ]:
        m = pattern.search(text)
        if m:
            findings.append(Finding(
                rule=rule, stage=Stage.INPUT, severity=Severity.BLOCK,
                reason=reason, detail={"matched": m.group(0)[:80]},
                customer_message=GENERIC_SECURITY_MESSAGE,
            ))
    return findings


def check_unauthorised_request(text: str) -> list[Finding]:
    """Requests for commitments the assistant has no authority to make.

    These escalate rather than block: the customer may genuinely be owed a fee
    waiver, and a human can decide. Refusing outright would be wrong.
    """
    findings = []
    if RE_UNAUTHORISED_ASK.search(text) and not _is_process_question(text):
        findings.append(Finding(
            rule="unauthorised_commitment_requested", stage=Stage.INPUT,
            severity=Severity.ESCALATE,
            reason=("request asks for a discount, waiver, approval or guarantee "
                    "that the assistant cannot authorise (POL-CS-001 S12.4)"),
            customer_message=(
                "That's a decision a colleague needs to make rather than me. "
                "I've passed it on with the details."
            ),
        ))
    m = RE_FALSE_PREMISE.search(text)
    if m:
        # Escalates rather than cautions. A false premise about a return window
        # is not a harmless framing error - if the assistant accepts it, the
        # answer becomes a commitment about money that policy does not support.
        # The customer may also be honestly mistaken, so a human should correct
        # them rather than the assistant refusing outright.
        findings.append(Finding(
            rule="false_premise", stage=Stage.INPUT, severity=Severity.ESCALATE,
            reason=("message asserts a policy the documentation does not "
                    "support; answering from the customer's premise would "
                    "create a commitment policy cannot honour"),
            detail={"matched": m.group(0)[:80]},
            customer_message=(
                "I want to check that against our actual policy rather than "
                "assume it, so I've passed this to a colleague."
            ),
        ))
    return findings


def check_identity_sensitive(text: str) -> list[Finding]:
    """Account changes and refund redirection.

    Escalates on security grounds, not uncertainty. Identity cannot be verified
    in a chat window, and knowing someone's name and order number is not
    verification (POL-CS-001 S6.3).
    """
    m = RE_IDENTITY_SENSITIVE.search(text)
    if not m:
        return []
    if _is_process_question(text):
        return [Finding(
            rule="identity_sensitive_topic", stage=Stage.INPUT,
            severity=Severity.CAUTION,
            reason=("question is about an identity-sensitive process; answer "
                    "from policy but do not act"),
            detail={"matched": m.group(0)[:80]},
        )]
    return [Finding(
        rule="identity_verification_required", stage=Stage.INPUT,
        severity=Severity.ESCALATE,
        reason=("request would alter an account or redirect money; identity "
                "cannot be verified in chat (POL-CS-001 S6)"),
        detail={"matched": m.group(0)[:80]},
        customer_message=(
            "For anything that changes your account or where a refund is sent, "
            "I need to hand you to a colleague who can verify your identity "
            "properly. I've done that."
        ),
    )]


def check_pii(text: str) -> list[Finding]:
    """Detect credentials and payment data the customer should not have sent.

    This is a CAUTION, not a block: the customer has already sent it, and
    refusing them help would compound the mistake. The finding drives redaction
    before logging.
    """
    scanned = _mask_orders(text)
    found = {k: p.search(scanned) for k, p in PII_PATTERNS.items()}
    hits = {k: m.group(0)[:6] + "..." for k, m in found.items() if m}
    if not hits:
        return []
    return [Finding(
        rule="pii_in_message", stage=Stage.INPUT, severity=Severity.CAUTION,
        reason=f"message contains sensitive data ({', '.join(hits)}); "
               f"redact before logging",
        detail={"types": sorted(hits)},
        customer_message=(
            "Please don't share card numbers, passwords or one-time codes in "
            "chat - our staff never ask for them (POL-PAY-001 S6.2)."
        ),
    )]


def check_scope(text: str) -> list[Finding]:
    """Out-of-domain queries. Refuse politely; do not escalate to a human."""
    low = text.lower()
    hit = next((t for t in OUT_OF_DOMAIN if t in low), None)
    if hit is None:
        return []
    return [Finding(
        rule="out_of_domain", stage=Stage.INPUT, severity=Severity.BLOCK,
        reason=f"query is outside Pacify customer support ('{hit}')",
        detail={"topic": hit},
        customer_message=(
            "That's outside what I can help with - I'm here for Pacify orders, "
            "products and policies."
        ),
    )]


def redact(text: str) -> str:
    """Replace sensitive values before a message is written to a log."""
    out = text
    for name, pattern in PII_PATTERNS.items():
        if name == "card_number":
            # protect order references from the digit-run rule
            out = RE_ORDER_REF.sub(lambda m: m.group(0).replace("-", "\x00"), out)
            out = pattern.sub(f"[REDACTED:{name}]", out)
            out = out.replace("\x00", "-")
        else:
            out = pattern.sub(f"[REDACTED:{name}]", out)
    return out


# =====================================================================
# Entry point
# =====================================================================

INPUT_RULES = (
    check_injection,
    check_extraction,
    check_unauthorised_request,
    check_identity_sensitive,
    check_pii,
    check_scope,
)


def screen_input(text: str, image_text: str = "") -> GuardrailVerdict:
    """Run every input rule.

    `image_text` is text extracted from an attached screenshot. It is screened
    with the SAME rules as typed input, because an instruction rendered into a
    PNG is still an instruction once OCR reads it - and it arrives through a
    channel people forget to defend.
    """
    findings: list[Finding] = []
    for rule in INPUT_RULES:
        findings.extend(rule(text))

    if image_text:
        for rule in (check_injection, check_extraction):
            for f in rule(image_text):
                f.rule = f"image_{f.rule}"
                f.reason = f"{f.reason} (text extracted from the attached image)"
                f.detail["source"] = "image"
                findings.append(f)

    return GuardrailVerdict(stage=Stage.INPUT, findings=findings)


if __name__ == "__main__":
    import json

    from src.config.settings import settings

    cases = json.loads(
        (settings.eval_dir / "adversarial_eval.json").read_text()
    )["cases"]
    blocked = 0
    print(f"{'id':8s} {'severity':10s} {'category':26s} rules")
    print("-" * 96)
    for c in cases:
        v = screen_input(c["prompt"])
        if v.must_escalate:
            blocked += 1
        print(f"{c['id']:8s} {v.severity.value:10s} {c['category']:26s} "
              f"{','.join(v.rules_fired) or '-'}")
    print(f"\nblocked or escalated: {blocked}/{len(cases)} "
          f"({100 * blocked / len(cases):.0f}%)")
