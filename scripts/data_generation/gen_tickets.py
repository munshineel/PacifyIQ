"""Synthetic support ticket history for the admin dashboard.

12,000 tickets over 6 months, with DELIBERATELY PLANTED TRENDS so the
emerging-issue detector has something real to find and can be validated.

*** THIS IS SIMULATED DATA. It must be labelled as such in the UI and README. ***
"""
import os, sys
from pathlib import Path
# project-relative: <root>/data
DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")

import csv, random, os, math
from datetime import datetime, timedelta
from collections import Counter

random.seed(11)
OUT = f"{DATA_ROOT}/tickets"
os.makedirs(OUT, exist_ok=True)

END = datetime(2026, 8, 21)
START = END - timedelta(days=182)
N = 12000

BASE_INTENT_W = {
    "order_tracking": 19, "return_policy_question": 14, "return_refund_request": 13,
    "product_information": 12, "shipping_delivery": 11, "technical_support": 10,
    "warranty_claim": 8, "payment_issue": 6, "account_management": 4,
    "complaint": 2, "out_of_scope": 1,
}
SENT_BY_INTENT = {
    "complaint": (0.02, 0.10, 0.88), "payment_issue": (0.05, 0.30, 0.65),
    "return_refund_request": (0.08, 0.42, 0.50), "warranty_claim": (0.07, 0.45, 0.48),
    "technical_support": (0.08, 0.52, 0.40), "order_tracking": (0.12, 0.60, 0.28),
    "shipping_delivery": (0.15, 0.62, 0.23), "return_policy_question": (0.20, 0.68, 0.12),
    "product_information": (0.34, 0.60, 0.06), "account_management": (0.15, 0.65, 0.20),
    "out_of_scope": (0.30, 0.66, 0.04),
}
SUBTOPIC = {
    "order_tracking": ["where is my order", "delayed shipment", "tracking not updating",
                       "delivery attempt failed", "marked delivered not received"],
    "return_policy_question": ["return window", "restocking fee", "opened item eligibility",
                               "return shipping cost", "bulk order returns"],
    "return_refund_request": ["change of mind return", "damaged on arrival",
                              "wrong item received", "refund amount query", "exchange request"],
    "product_information": ["specifications", "stock availability", "model comparison",
                            "compatibility", "extended warranty options"],
    "shipping_delivery": ["shipping cost", "delivery timeline", "address change",
                          "international shipping", "white glove scheduling"],
    "technical_support": ["will not power on", "wifi drops", "display issue",
                          "battery drain", "overheating", "audio fault",
                          "boot failure", "driver problem"],
    "warranty_claim": ["battery health", "dead pixels", "screen fault",
                       "keyboard failure", "third party brand routing", "repair turnaround"],
    "payment_issue": ["payment failed amount debited", "double charge",
                      "refund not received", "emi query", "invoice request"],
    "account_management": ["login failure", "password reset", "email change",
                           "account deletion", "data export"],
    "complaint": ["slow response", "unresolved issue", "agent conduct",
                  "repeated contact", "delivery experience"],
    "out_of_scope": ["general chat", "unrelated question", "bot test"],
}

def day_weight(d):
    """Weekly seasonality + slow overall growth."""
    dow = d.weekday()
    w = 1.0 if dow < 5 else (0.55 if dow == 5 else 0.25)
    growth = 1.0 + 0.25 * ((d - START).days / 182)
    return w * growth

days = [START + timedelta(days=i) for i in range(183)]
dw = [day_weight(d) for d in days]
tot = sum(dw)
per_day = [max(5, int(N * w / tot)) for w in dw]

rows = []
tid = 40000

def pick_intent(d):
    w = dict(BASE_INTENT_W)
    days_ago = (END - d).days

    # --- PLANTED TREND 1: login failure spike, last 7 days, 4.5x ---
    if days_ago <= 7:
        w["account_management"] *= 4.5
    # --- PLANTED TREND 2: display/monitor tech support ramp, last 21 days ---
    if days_ago <= 21:
        w["technical_support"] *= 1.0 + 1.4 * (1 - days_ago / 21)
    # --- PLANTED TREND 3: payment failures, one-off 3-day gateway incident ---
    if 38 <= days_ago <= 41:
        w["payment_issue"] *= 6.0
    # --- PLANTED TREND 4: shipping complaints, festival-season weeks ---
    if 60 <= days_ago <= 95:
        w["shipping_delivery"] *= 1.8
        w["order_tracking"] *= 1.5
        w["complaint"] *= 2.0
    # --- PLANTED TREND 5: returns rise after a v2 policy change 90 days ago ---
    if days_ago <= 90:
        w["return_policy_question"] *= 1.35

    ks = list(w)
    return random.choices(ks, weights=[w[k] for k in ks])[0]

def pick_subtopic(intent, d):
    days_ago = (END - d).days
    subs = SUBTOPIC[intent]
    if intent == "account_management" and days_ago <= 7:
        return "login failure" if random.random() < 0.78 else random.choice(subs)
    if intent == "technical_support" and days_ago <= 21:
        return "display issue" if random.random() < 0.55 else random.choice(subs)
    if intent == "payment_issue" and 38 <= days_ago <= 41:
        return "payment failed amount debited" if random.random() < 0.80 else random.choice(subs)
    return random.choice(subs)

for d, n in zip(days, per_day):
    for _ in range(n):
        intent = pick_intent(d)
        pos, neu, neg = SENT_BY_INTENT[intent]
        sent = random.choices(["positive", "neutral", "negative"], weights=[pos, neu, neg])[0]
        # AI resolution: harder intents escalate more
        base_res = {"order_tracking": 0.93, "return_policy_question": 0.90,
                    "product_information": 0.85, "shipping_delivery": 0.88,
                    "technical_support": 0.72, "return_refund_request": 0.35,
                    "warranty_claim": 0.48, "payment_issue": 0.45,
                    "account_management": 0.22, "complaint": 0.12,
                    "out_of_scope": 0.95}[intent]
        if sent == "negative":
            base_res *= 0.75
        resolved_by = "ai" if random.random() < base_res else "human"
        ts = d + timedelta(hours=random.choices(range(9, 21),
                weights=[6,9,11,12,10,8,7,9,11,10,7,4])[0],
                minutes=random.randrange(60))
        latency = round(random.gauss(1.6, 0.5), 2) if resolved_by == "ai" else round(random.gauss(2.1, 0.7), 2)
        latency = max(0.4, latency)
        tokens = random.randint(900, 4200)
        conf = round(min(0.99, max(0.05, random.gauss(0.82 if resolved_by == "ai" else 0.46, 0.14))), 3)
        rows.append((
            f"TKT-{tid}", ts.strftime("%Y-%m-%d %H:%M:%S"), intent,
            pick_subtopic(intent, d), sent,
            random.choices(["low", "medium", "high"],
                           weights=[0.5, 0.36, 0.14] if sent != "negative" else [0.2, 0.4, 0.4])[0],
            resolved_by,
            "resolved" if random.random() < 0.94 else "open",
            conf, latency, tokens,
            random.choices(["IN", "EU"], weights=[0.9, 0.1])[0],
            random.choices(["chat", "email", "phone"], weights=[0.62, 0.28, 0.10])[0],
            random.choices(["up", "down", ""], weights=[0.30, 0.07, 0.63])[0],
        ))
        tid += 1

random.shuffle(rows)
rows.sort(key=lambda r: r[1])

with open(f"{OUT}/ticket_history.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ticket_id", "created_at", "intent", "subtopic", "sentiment",
                "priority", "resolved_by", "status", "confidence",
                "latency_seconds", "tokens_used", "region", "channel", "feedback"])
    w.writerows(rows)

ai = sum(1 for r in rows if r[6] == "ai")
print(f"ticket_history.csv  {len(rows)} tickets")
print(f"  date range        {rows[0][1][:10]} to {rows[-1][1][:10]}")
print(f"  AI resolved       {ai/len(rows)*100:.1f}%")
print(f"  escalated         {(1-ai/len(rows))*100:.1f}%")
print(f"  negative sentiment {sum(1 for r in rows if r[4]=='negative')/len(rows)*100:.1f}%")

print("\nPLANTED TRENDS (validate the detector against these):")
last7 = [r for r in rows if r[1] >= (END - timedelta(days=7)).strftime("%Y-%m-%d")]
prev = [r for r in rows if (END - timedelta(days=35)).strftime("%Y-%m-%d") <= r[1] < (END - timedelta(days=7)).strftime("%Y-%m-%d")]
def rate(rs, i):
    return sum(1 for r in rs if r[2] == i) / max(1, len(rs)) * 100
print(f"  T1 account_management  last7 {rate(last7,'account_management'):.1f}%  vs prior {rate(prev,'account_management'):.1f}%")
print(f"  T2 technical_support   last7 {rate(last7,'technical_support'):.1f}%  vs prior {rate(prev,'technical_support'):.1f}%")
inc = [r for r in rows if (END-timedelta(days=41)).strftime("%Y-%m-%d") <= r[1][:10] <= (END-timedelta(days=38)).strftime("%Y-%m-%d")]
print(f"  T3 payment_issue       incident window {rate(inc,'payment_issue'):.1f}%  vs baseline {rate(prev,'payment_issue'):.1f}%")

with open(f"{OUT}/PLANTED_TRENDS.md", "w") as f:
    f.write("""# Planted trends in ticket_history.csv

*** ticket_history.csv is SIMULATED data. Label it as such in the UI and README. ***

The emerging-issue detector on the admin dashboard must find these. If it does
not, the detector is wrong. If it finds trends not listed here, they are noise.

| ID | Trend | Window | Magnitude | Dominant subtopic |
|---|---|---|---|---|
| T1 | account_management spike | last 7 days | 4.5x baseline | login failure (78%) |
| T2 | technical_support ramp | last 21 days, rising | up to 2.4x | display issue (55%) |
| T3 | payment_issue incident | 38-41 days ago, 3-day burst | 6x baseline | payment failed amount debited (80%) |
| T4 | shipping/tracking/complaint rise | 60-95 days ago | 1.5-2.0x | festival season |
| T5 | return_policy_question elevated | last 90 days | 1.35x | policy v2 took effect |

T1 is the headline case: a sharp, recent, narrow spike. T2 tests detection of a
gradual ramp rather than a step change. T3 tests detection of a historical
incident that has already resolved. T4 tests seasonality that should NOT be
flagged as an emerging issue. T5 is a slow drift that a naive week-over-week
detector will miss entirely.

A good detector distinguishes T1/T2/T3 (real) from T4 (seasonal) and catches
T5 only with a longer baseline window.
""")
print("\nWrote PLANTED_TRENDS.md")
