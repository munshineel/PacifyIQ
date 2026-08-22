"""Rule-based entity extraction.

Deliberately deterministic rather than model-based. Argument extraction is the
most common failure mode in tool-calling agents - the model picks the right
tool and then passes a hallucinated order ID. A regex cannot hallucinate.

Order-ID normalisation is shared with src/db/queries.py so the extracted value
is always in the canonical form the database expects.
"""
from __future__ import annotations

import re

from src.understanding.schema import Entities

RE_ORDER = re.compile(r"\b(?:pac[-\s]?2026[-\s]?(\d{4,6})|#(\d{4,6})|(?<!\d)(\d{5})(?!\d))", re.I)
RE_ERROR = re.compile(
    r"\b((?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)"
    r"(?:[-_][A-Z0-9]+)*[-_]?(?:0x)?[0-9A-F]{2,})\b",
    re.I,
)
RE_AMOUNT = re.compile(r"(?:rs\.?|inr|₹)\s?([\d,]+)|\b([\d,]{4,})\s?(?:rs|inr|rupees)\b", re.I)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})\b",
    re.I,
)

# Product names from data/canonical_facts.md section 8.
PRODUCTS = [
    "pacify probook 14 lite", "pacify probook 16", "pacify probook 14",
    "probook 14 lite", "probook 16", "probook 14", "probook",
    "pacify phone x pro", "pacify phone x", "phone x pro", "phone x",
    "pacify tab 11", "tab 11",
    "pacify vision 32", "pacify vision 27", "vision 32", "vision 27",
    "pacify keylite", "keylite", "pacify soundpods", "soundpods",
    "northwind ultra 15", "northwind", "kestrel note 9", "kestrel",
]


def normalize_order_id(raw: str) -> str:
    """Canonical form: PAC-2026-NNNNN. Mirrors src/db/queries.py."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("2026") and len(digits) > 5:
        digits = digits[4:]
    return f"PAC-2026-{digits}"


def extract(text: str) -> Entities:
    """Pull structured items out of a message."""
    t = str(text)

    orders = []
    for m in RE_ORDER.finditer(t):
        digits = next(g for g in m.groups() if g)
        oid = normalize_order_id(digits)
        if oid not in orders:
            orders.append(oid)

    codes = []
    for m in RE_ERROR.finditer(t):
        c = m.group(1).upper().replace("_", "-")
        if c not in codes:
            codes.append(c)

    amounts = []
    for m in RE_AMOUNT.finditer(t):
        val = next((g for g in m.groups() if g), None)
        if val:
            amounts.append(val.replace(",", ""))

    lower = t.lower()
    products, seen = [], set()
    for p in PRODUCTS:
        if p in lower and not any(p in s for s in seen):
            products.append(p)
            seen.add(p)

    return Entities(
        order_ids=orders,
        error_codes=codes,
        products=products,
        amounts=amounts,
        dates=RE_DATE.findall(t),
        emails=RE_EMAIL.findall(t),
    )


if __name__ == "__main__":
    cases = [
        "where is my order PAC-2026-12345",
        "status of #12347 please",
        "order 12345 kahan hai",
        "getting PAY-402 and also ERR-DP-0x004 on my Vision 27",
        "refund of Rs 64,900 for the ProBook 14 bought on 5th August",
        "my northwind ultra 15 is faulty, email me at user@example.com",
        "hello",
    ]
    for c in cases:
        e = extract(c)
        print(f"\n{c}")
        for k, v in e.to_dict().items() if hasattr(e, "to_dict") else vars(e).items():
            if v:
                print(f"   {k}: {v}")
