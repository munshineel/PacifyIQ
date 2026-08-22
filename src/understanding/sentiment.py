"""Sentiment and urgency scoring.

WHY THESE ARE NOT SUPERVISED MODELS
-----------------------------------
The dataset cannot support supervised text -> sentiment or text -> urgency:

    asset                 text   intent   sentiment   urgency
    train.csv             YES    YES      no          no
    test_hard.csv         YES    YES      no          no
    ticket_history.csv    NO     YES      YES         YES (priority)

Labels and text live in different files. `ticket_history.csv` is aggregate
operational metadata with no message content, so there is nothing to learn a
text classifier from. Fabricating labels with an LLM and then "evaluating"
against them would measure agreement with the labeller, not accuracy.

So these are transparent rule-based scorers built from a domain lexicon, with
an intent-conditional prior taken from the observed ticket history. Every score
is explainable — you can see exactly which terms fired.

They are evaluated in scripts/evaluate_sentiment_urgency.py against a small
hand-annotated set, with the limitations stated plainly.

REPLACEMENT PATH
----------------
Once the system is live (Phase 13+) and real traces accumulate with 👍/👎
feedback and human-agent labels, these become trainable. The interfaces here
are designed so a learned model can drop in without changing callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# =====================================================================
# Lexicons
# =====================================================================
# Weighted rather than binary: "furious" is a stronger signal than "late".
# Domain-specific by necessity - "damaged" and "broken" are strong negatives
# in support and neutral almost everywhere else.

NEGATIVE_TERMS: dict[str, float] = {
    # anger and hostility
    "furious": 3.0, "disgusted": 3.0, "outrageous": 3.0, "pathetic": 3.0,
    "useless": 2.5, "worst": 2.5, "terrible": 2.5, "awful": 2.5,
    "ridiculous": 2.5, "unacceptable": 2.5, "angry": 2.0, "upset": 2.0,
    "disappointed": 2.0, "frustrated": 2.0, "annoyed": 1.5, "rude": 2.0,
    # accusation
    "fraud": 3.0, "scam": 3.0, "theft": 3.0, "cheating": 2.5, "lying": 2.5,
    # escalation threats
    "court": 3.0, "lawyer": 3.0, "legal": 2.0, "chargeback": 3.0,
    "consumer": 2.0, "complaint": 1.5, "manager": 1.5, "supervisor": 1.5,
    # neglect
    "ignored": 2.5, "nobody": 2.0, "never": 1.5, "still": 1.0, "again": 1.5,
    "third": 1.5, "waiting": 1.5, "unresolved": 2.0,
    # product failure
    "broken": 1.5, "damaged": 1.5, "faulty": 1.5, "defective": 1.5,
    "cracked": 1.5, "dead": 1.5, "failed": 1.5, "crashed": 1.5,
    "wrong": 1.0, "late": 1.0, "delayed": 1.0, "stuck": 1.0, "missing": 1.5,
    # romanised hindi
    "ganda": 2.0, "bekar": 2.0, "kharab": 1.5, "pareshan": 2.0,
}

POSITIVE_TERMS: dict[str, float] = {
    "excellent": 2.5, "perfect": 2.5, "love": 2.0, "great": 2.0,
    "fantastic": 2.5, "wonderful": 2.5, "amazing": 2.0, "brilliant": 2.0,
    "happy": 1.5, "pleased": 1.5, "satisfied": 1.5, "impressed": 2.0,
    "helpful": 1.5, "quick": 1.0, "good": 1.0, "nice": 1.0, "fine": 0.5,
    "thanks": 1.0, "thank": 1.0, "appreciate": 1.5, "grateful": 1.5,
    "please": 0.3, "kindly": 0.3, "hope": 0.3,
}

URGENCY_TERMS: dict[str, float] = {
    "urgent": 3.0, "urgently": 3.0, "emergency": 3.0, "critical": 2.5,
    "immediately": 2.5, "asap": 2.5, "right now": 2.5, "today": 1.5,
    "tomorrow": 1.0, "deadline": 2.0, "quickly": 1.5, "soon": 1.0,
    "waiting": 1.0, "still": 0.5, "jaldi": 2.5, "turant": 2.5,
    "abhi": 1.5, "hurry": 2.0,
}

# Explicit negation of sentiment, e.g. "no complaints at all"
NEGATION = re.compile(r"\b(?:not|no|never|n't|dont|don't|isnt|isn't)\s+(\w+)")

# Intent-conditional priors, measured from ticket_history.csv (11,905 rows).
# Not learned from text - these are observed base rates per intent.
INTENT_NEGATIVE_PRIOR: dict[str, float] = {
    "complaint": 0.845,
    "payment_issue": 0.631,
    "return_refund_request": 0.517,
    "warranty_claim": 0.493,
    "technical_support": 0.404,
    "order_tracking": 0.274,
    "shipping_delivery": 0.230,
    "account_management": 0.200,
    "return_policy_question": 0.120,
    "product_information": 0.060,
    "out_of_scope": 0.040,
}

INTENT_HIGH_PRIORITY_PRIOR: dict[str, float] = {
    "complaint": 0.375,
    "payment_issue": 0.266,
    "return_refund_request": 0.249,
    "warranty_claim": 0.264,
    "technical_support": 0.229,
    "order_tracking": 0.203,
    "shipping_delivery": 0.190,
    "account_management": 0.180,
    "return_policy_question": 0.120,
    "product_information": 0.090,
    "out_of_scope": 0.050,
}

TOKEN = re.compile(r"[a-z']+")


# =====================================================================
# Results
# =====================================================================

@dataclass
class SentimentResult:
    label: str                      # positive | neutral | negative
    score: float                    # -1.0 .. +1.0
    lexicon_score: float
    intent_prior: float | None
    matched_negative: list[str] = field(default_factory=list)
    matched_positive: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = [f"{self.label} ({self.score:+.2f})"]
        if self.matched_negative:
            parts.append(f"negative terms: {', '.join(self.matched_negative)}")
        if self.matched_positive:
            parts.append(f"positive terms: {', '.join(self.matched_positive)}")
        if self.negated:
            parts.append(f"negated: {', '.join(self.negated)}")
        if self.intent_prior is not None:
            parts.append(f"intent prior: {self.intent_prior:.2f}")
        return " | ".join(parts)


@dataclass
class UrgencyResult:
    label: str                      # low | medium | high
    score: float                    # 0.0 .. 1.0
    matched_terms: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        active = {k: round(v, 2) for k, v in self.signals.items() if v > 0}
        return f"{self.label} ({self.score:.2f}) | " + str(active)


# =====================================================================
# Scorers
# =====================================================================

def score_sentiment(
    text: str,
    intent: str | None = None,
    prior_weight: float = 0.35,
) -> SentimentResult:
    """Lexicon sentiment, optionally blended with an intent base rate.

    The intent prior is what makes this more than a word list: a terse
    "payment failed" carries no sentiment words but comes from a category that
    is 63% negative in the observed history.
    """
    lower = str(text).lower()
    tokens = TOKEN.findall(lower)
    token_set = set(tokens)

    negated = {m.group(1) for m in NEGATION.finditer(lower)}

    neg_hits = [w for w in tokens if w in NEGATIVE_TERMS and w not in negated]
    pos_hits = [w for w in tokens if w in POSITIVE_TERMS and w not in negated]

    neg_weight = sum(NEGATIVE_TERMS[w] for w in neg_hits)
    pos_weight = sum(POSITIVE_TERMS[w] for w in pos_hits)

    # shouting and repeated punctuation are affect signals independent of words
    raw = str(text)
    letters = [c for c in raw if c.isalpha()]
    if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        neg_weight += 1.5
    neg_weight += 0.5 * min(raw.count("!"), 3)

    total = neg_weight + pos_weight
    lex = 0.0 if total == 0 else (pos_weight - neg_weight) / total

    prior = None
    score = lex
    if intent and intent in INTENT_NEGATIVE_PRIOR:
        prior = INTENT_NEGATIVE_PRIOR[intent]
        # The prior is a NEGATIVE base rate. A 27% negative rate does not mean
        # 73% positive - the remainder is overwhelmingly neutral. So the prior
        # may only push the score downward, never upward. Without this the
        # scorer labels "where is my order 12345" as positive, which is wrong.
        prior_signal = min(0.0, (0.35 - prior) / 0.65)
        w = prior_weight if total > 0 else 0.75
        score = (1 - w) * lex + w * prior_signal

    label = "negative" if score <= -0.20 else "positive" if score >= 0.25 else "neutral"

    return SentimentResult(
        label=label,
        score=round(score, 3),
        lexicon_score=round(lex, 3),
        intent_prior=prior,
        matched_negative=sorted(set(neg_hits)),
        matched_positive=sorted(set(pos_hits)),
        negated=sorted(negated & (token_set & (NEGATIVE_TERMS.keys() | POSITIVE_TERMS.keys()))),
    )


def score_urgency(
    text: str,
    intent: str | None = None,
    sentiment: str | None = None,
) -> UrgencyResult:
    """Combine explicit urgency language, sentiment, and an intent base rate.

    Each contribution is reported separately so a triage decision can be
    audited rather than trusted.
    """
    lower = str(text).lower()
    tokens = TOKEN.findall(lower)

    matched = [w for w in tokens if w in URGENCY_TERMS]
    for phrase in ("right now", "as soon as possible"):
        if phrase in lower:
            matched.append(phrase)

    lexical = min(sum(URGENCY_TERMS.get(w, 2.0) for w in matched) / 5.0, 1.0)

    signals = {
        "lexical": lexical,
        # normalised so each signal is on a 0-1 scale before weighting
        "intent_prior": (INTENT_HIGH_PRIORITY_PRIOR.get(intent, 0.15) / 0.375)
        if intent else 0.0,
        "sentiment": 1.0 if sentiment == "negative" else 0.0,
        "repeat_contact": 1.0 if re.search(
            r"\b(?:third|3rd|again|still|second|2nd)\b", lower) else 0.0,
        "legal_threat": 1.0 if re.search(
            r"\b(?:court|lawyer|legal|chargeback|consumer forum)\b", lower) else 0.0,
    }

    weights = {
        "lexical": 0.30,
        "intent_prior": 0.20,
        "sentiment": 0.25,
        "repeat_contact": 0.15,
        "legal_threat": 0.10,
    }
    score = sum(signals[k] * weights[k] for k in weights)

    # A legal threat is a hard escalation trigger (POL-CS-001 S3.4(d)) and must
    # not be diluted by the weighted blend.
    if signals["legal_threat"] > 0:
        score = max(score, 0.80)

    # Thresholds calibrated to the achievable range: the maximum score without
    # a legal threat is 0.90, and a bare intent prior alone reaches ~0.20.
    label = "high" if score >= 0.55 else "medium" if score >= 0.30 else "low"

    return UrgencyResult(
        label=label,
        score=round(score, 3),
        matched_terms=sorted(set(matched)),
        signals={k: round(v, 3) for k, v in signals.items()},
    )


if __name__ == "__main__":
    cases = [
        ("this is the THIRD time. either refund my money today or i'm doing a chargeback",
         "complaint"),
        ("thanks for the quick reply, one more thing about warranty", "warranty_claim"),
        ("where is my order 12345", "order_tracking"),
        ("payment failed but money was deducted", "payment_issue"),
        ("no complaints at all, just wondering about warranty coverage", "warranty_claim"),
        ("MY LAPTOP IS BROKEN AND NOBODY IS HELPING ME", "technical_support"),
        ("what is your return policy", "return_policy_question"),
    ]
    for text, intent in cases:
        s = score_sentiment(text, intent)
        u = score_urgency(text, intent, s.label)
        print(f"\n{text[:66]}")
        print(f"  sentiment: {s.explain()}")
        print(f"  urgency:   {u.explain()}")
