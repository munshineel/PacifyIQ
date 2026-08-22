"""Hand-annotated sentiment and urgency labels for evaluation.

*** LIMITATION, STATED UP FRONT ***
These annotations were authored by an LLM, not by a human annotator, and there
is no second annotator so no inter-annotator agreement can be reported. They
are adequate for detecting gross calibration errors in the rule-based scorers
and inadequate for claiming an accuracy figure with confidence.

The dataset provides no alternative: ticket_history.csv carries sentiment and
priority labels but no message text, while the intent CSVs carry text but no
sentiment. See reports/classification_report.md.

Replace with human labels, or with real 👍/👎 feedback from Phase 13 traces,
before treating any number derived from this file as trustworthy.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "eval" / "sentiment_urgency_eval.json"

# (text, sentiment, urgency, rationale)
CASES = [
 # --- clearly negative, high urgency ---
 ("this is the THIRD time. either refund my money today or i'm doing a chargeback and posting this everywhere","negative","high","repeat contact + legal threat + demand"),
 ("i have been waiting 3 weeks for a refund that you said would take 7 days. this is theft. i am contacting consumer court tomorrow.","negative","high","accusation + legal threat"),
 ("Absolutely disgusted. Ordered a 90k laptop, arrived with a cracked screen, and your support has ignored 4 emails.","negative","high","strong anger + neglect + high value"),
 ("your agent hung up on me twice. i want a manager. now.","negative","high","conduct complaint + escalation demand"),
 ("i'm reporting this to consumer court","negative","high","legal threat"),
 ("either fix this today or i'm doing a chargeback","negative","high","ultimatum"),
 ("still no response. still no refund. still nothing. day 19.","negative","high","repeated neglect"),
 ("i've called 4 times and nothing has happened","negative","high","repeat contact"),
 ("this is fraud, you took my money","negative","high","accusation"),
 ("im consulting a lawyer about this","negative","high","legal threat"),

 # --- negative, medium urgency ---
 ("worst service ever","negative","medium","pure affect, no specific action"),
 ("you people are useless","negative","medium","insult without demand"),
 ("bahut ganda service hai","negative","medium","code-mixed complaint"),
 ("no one is helping me, very disappointed","negative","medium","neglect + disappointment"),
 ("worst online shopping experience of my life","negative","medium","strong affect, no urgency"),
 ("card declined but money deducted","negative","medium","money missing"),
 ("i was charged twice for order 12345","negative","medium","billing error"),
 ("refund initiated 10 days ago, nothing in my account","negative","medium","overdue refund"),
 ("my laptop arrived damaged and i want a refund","negative","medium","product failure + action"),
 ("paisa kat gaya but order nahi hua","negative","medium","code-mixed payment failure"),
 ("MY LAPTOP IS BROKEN AND NOBODY IS HELPING ME","negative","medium","shouting + neglect"),
 ("my delivery is 5 days late","negative","medium","delay"),
 ("the free gift wasnt in the box","negative","medium","incomplete order"),
 ("your delivery guy was rude","negative","medium","conduct complaint"),
 ("i was promised a callback yesterday","negative","medium","SLA breach"),
 ("my order was cancelled without my permission","negative","medium","unexpected action"),
 ("the courier marked it delivered but i never got it","negative","medium","missing parcel"),
 ("i think someone hacked my account","negative","high","security incident"),

 # --- negative, low urgency ---
 ("im really disappointed but i understand its policy, can you just tell me the return window","negative","low","disappointment but calm and informational"),
 ("i'm not angry, i just want to know why nobody told me about the restocking fee","negative","low","explicitly de-escalated"),
 ("laptop is 3 years old and the fan is loud","negative","low","minor issue, out of warranty"),
 ("my keyboard has a sticky key","negative","low","minor fault"),
 ("screen has dead pixels after 2 months","negative","low","defect but calm"),

 # --- neutral ---
 ("where is my order 12345","neutral","low","factual query"),
 ("what is your return policy","neutral","low","informational"),
 ("how many days do i have to return","neutral","low","informational"),
 ("do you ship to germany","neutral","low","informational"),
 ("does the ProBook 14 have thunderbolt","neutral","low","product query"),
 ("order kahan hai bro delivery kab hoga","neutral","low","code-mixed factual"),
 ("can i return opened items","neutral","low","policy query"),
 ("how do i factory reset my laptop","neutral","low","how-to"),
 ("what payment methods do you accept","neutral","low","informational"),
 ("is my Phone X still under warranty","neutral","low","status check"),
 ("getting error PAY-402 at checkout","neutral","medium","blocked transaction, no affect"),
 ("monitor goes black randomly, see attached","neutral","low","symptom report"),
 ("can i change my delivery address","neutral","low","request"),
 ("how do i claim warranty","neutral","low","process query"),
 ("what are your support hours","neutral","low","informational"),
 ("i want to return order 12345","neutral","low","transactional, no affect"),
 ("do you have a trade in program","neutral","low","informational"),
 ("12345","neutral","low","bare reference"),
 ("hello","neutral","low","greeting"),
 ("ok","neutral","low","acknowledgement"),
 ("Under EU distance selling rules I have 14 days statutory withdrawal regardless of your store policy. Please confirm.","neutral","low","assertive but not hostile"),
 ("Please confirm you will reimburse within 14 days of receiving the goods as required","neutral","low","formal, rights-aware"),
 ("i need this before diwali, will it make it","neutral","medium","deadline pressure, no affect"),
 ("cancel my order","neutral","low","transactional"),

 # --- positive ---
 ("thanks for the quick reply earlier. One more thing - the refund amount seems short?","positive","low","gratitude then query"),
 ("You guys have been great so far. Just need to sort out this return.","positive","low","explicit praise"),
 ("love the laptop! but the charger stopped working","positive","low","praise plus fault"),
 ("Hi! Hope you're having a good day. Quick one - when's my order arriving? 12345. Thanks so much!","positive","low","warm social wrapper"),
 ("no complaints at all, just wondering about warranty coverage","positive","low","explicitly not a complaint"),
 ("Sorry to bother you, I know you must be busy, but my laptop won't charge","positive","low","apologetic and polite despite fault"),
 ("thanks, that solved it","positive","low","resolution"),
 ("excellent service, very helpful","positive","low","praise"),
]

data = {
    "description": "Hand-annotated sentiment and urgency for evaluating the rule-based scorers.",
    "limitation": (
        "Annotations authored by an LLM, not a human. Single annotator, so no "
        "inter-annotator agreement is available. Adequate for detecting gross "
        "calibration errors, inadequate for a confident accuracy claim."
    ),
    "why_not_supervised": (
        "The dataset provides no text with sentiment labels: ticket_history.csv "
        "has sentiment but no message text, and the intent CSVs have text but no "
        "sentiment. Supervised text->sentiment is therefore not possible."
    ),
    "cases": [
        {"id": f"SU{i:03d}", "text": t, "sentiment": s, "urgency": u, "rationale": r}
        for i, (t, s, u, r) in enumerate(CASES, 1)
    ],
}

OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

from collections import Counter
print(f"wrote {OUT.name}: {len(CASES)} cases")
print("  sentiment:", dict(Counter(c[1] for c in CASES)))
print("  urgency:  ", dict(Counter(c[2] for c in CASES)))
