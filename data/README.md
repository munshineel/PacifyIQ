# PacifyIQ — Data Assets

Everything the PacifyIQ project needs before any application code is written.

**All data here is SYNTHETIC.** Pacify Electronics Pvt. Ltd. is fictional. This
must be stated in the project README and shown in the admin dashboard UI.

---

## Contents

```
pacifyiq_data/
├── canonical_facts.md          ← SOURCE OF TRUTH. Read this first.
├── PLANTED_DEFECTS.md          ← 18 engineered test cases. Read this second.
├── documents/                  ← 13 PDFs, 47 pages, the RAG corpus
│   ├── return_policy_v2.pdf              5 pp
│   ├── return_policy_v1_ARCHIVED.pdf     2 pp   (superseded, on purpose)
│   ├── refund_policy.pdf                 4 pp
│   ├── warranty_policy.pdf               5 pp
│   ├── shipping_policy.pdf               4 pp   (contains DEFECT-01)
│   ├── payment_policy.pdf                3 pp
│   ├── customer_service_policy.pdf       4 pp
│   ├── product_faq.pdf                   6 pp   (contains DEFECT-18)
│   ├── technical_support_faq.pdf         5 pp   (error-code table)
│   ├── eu_regional_addendum.pdf          3 pp
│   └── manuals/  probook14, phonex, vision27   2 pp each
├── _source/                    ← markdown sources for all 13 PDFs
├── db/pacify.db                ← SQLite: 500 customers, 2001 orders, 14 SKUs
├── intents/
│   ├── train.csv                    2,200 rows, 11 classes, imbalanced
│   └── test_hard.csv                  142 rows, hand-authored
├── tickets/
│   ├── ticket_history.csv          11,905 tickets, 6 months
│   └── PLANTED_TRENDS.md           5 trends the detector must find
├── eval/                       ← 295 cases across 7 sets
└── scripts/                    ← regenerate everything, seeded and reproducible
```

---

## Read these two files first

**`canonical_facts.md`** is the single source of truth. Every document, every
database row, every eval answer is consistent with it. If you find a
contradiction that is not in `PLANTED_DEFECTS.md`, it is a bug.

**`PLANTED_DEFECTS.md`** lists all 18 deliberate contradictions, ambiguities and
traps, with the expected system behaviour for each. Without this file you cannot
tell a test case from a defect three weeks from now.

Verify both with `python scripts/check_consistency.py`.

---

## The corpus

47 pages across 13 documents, producing roughly 1,800-2,200 chunks at 512 tokens.
Large enough that retrieval is a real problem, small enough to re-index in under
two minutes during ablations.

The corpus is **deliberately messy**. Six categories of engineered overlap:

| Category | Example | Tests |
|---|---|---|
| Direct contradiction | shipping_policy S11 says 30-day guarantee; return_policy_v2 S2 says 14 days | conflict detection, escalation |
| Version pair | v1 (archived) vs v2 (current) | metadata beats similarity |
| Numeric collision | card refund 5-7 days vs failed-payment reversal 5-7 days | clarification |
| Near-duplicate | same fact in formal / casual / marketing register | Precision@K |
| Multi-hop | refund amount needs 4 chunks across 2 documents | top-K sufficiency |
| Jurisdiction | EU addendum overrides base policy | specific-beats-general |

If the corpus were clean, every retrieval metric would pin near 1.0 and the
evaluation chapter would prove nothing.

Note the **fictional SKUs** (`Pacify ProBook 14`, `Pacify Vision 27`). The model
has zero parametric knowledge of these, so any specification it produces that is
not in the corpus is unambiguously a hallucination. No judgment call in scoring.

Section IDs (`S1`, `S2.3`) are stable and must never be renumbered — eval sets
reference them.

---

## Database

`db/pacify.db`, SQLite, 6 tables. Orders are positioned **against policy
boundaries** so eligibility logic is genuinely exercised.

27 deterministic edge cases with stable IDs, referenced by the agent eval set:

| Order | Case |
|---|---|
| PAC-2026-12345 | day 12 of 14, opened laptop — ELIGIBLE |
| PAC-2026-12346 | day 14 of 14 — eligible on the last day |
| PAC-2026-12347 | day 15 — EXPIRED by one day |
| PAC-2026-12348 | day 22, sealed — still eligible on the 30-day window |
| PAC-2026-12351 | day 8 — the 48h-to-14d grey zone (DEFECT-12) |
| PAC-2026-12352 | no-cost EMI — DEFECT-07 refund arithmetic |
| PAC-2026-12354 | EU customer, opened — DEFECT-03 override |
| PAC-2026-12357 | third-party brand — DEFECT-09 manufacturer routing |
| PAC-2026-12363 | delivery failed after 3 attempts |
| PAC-2026-12368 | bulk order, day 9 — expired on the 7-day window |

Full list: `python scripts/seed_db.py`.

---

## Intent data

`train.csv` — 2,200 rows, 11 classes, **deliberately imbalanced** to match real
queue volumes (order_tracking 18%, out_of_scope 1.8%). This forces **macro-F1**
as the honest metric; accuracy is meaningless here.

`test_hard.csv` — 142 hand-authored cases that are structurally unlike the
training templates: compound (42%), code-mixed, rambling, negated, typo-ridden,
emotionally loaded, and often ambiguous. Includes `secondary_intent` and a `note`
explaining why each case is hard.

**Expect a large macro-F1 gap between the two.** That gap is a finding, not a
failure — it quantifies how far templated training data sits from real customers.

### Two honest caveats

1. **`train.csv` is template-generated by script.** It is a working baseline, not
   a substitute for real data. Consider replacing or augmenting it with the
   **Bitext customer-support dataset** on Hugging Face (27K rows, 27 intents,
   CDLA-Sharing-1.0). It is a genuinely good domain match and carries linguistic
   variation tags. It could not be downloaded here — network access in this
   environment is restricted to PyPI/GitHub — so fetch it yourself:
   `pip install datasets` then load `bitext/Bitext-customer-support-llm-chatbot-training-dataset`.
   Note the share-alike licence before committing derivatives.
   Also consider the CLINC150 out-of-scope split for intent 11.

2. **`test_hard.csv` was authored by Claude, not by a human.** It is
   deliberately harder and messier than the templates, but it is still
   LLM-written. To make the train/test gap finding fully defensible, write or
   rewrite 30-50 of these yourself, or sample real phrasing from the Twitter
   Customer Support corpus. Say in your README which parts are which.

---

## Ticket history

11,905 tickets over 6 months, with weekly seasonality and growth.

**Five planted trends** the emerging-issue detector must find — see
`tickets/PLANTED_TRENDS.md`. T1 (account_management 3.7% → 11.5% in the last 7
days) is the headline case. T4 is seasonal and should **not** be flagged, which
is what separates a good detector from a naive one.

⚠️ This file is simulated. Label it as such in the UI.

---

## Evaluation sets

295 cases across seven files.

| File | Cases | Measures |
|---|---|---|
| `retrieval_eval.json` | 120 | Recall@K, Precision@K, MRR, nDCG |
| `generation_eval.json` | 25 | faithfulness, correctness, citation accuracy |
| `unanswerable_eval.json` | 40 | **abstention rate** |
| `agent_trajectory_eval.json` | 30 | tool selection, argument extraction, recovery |
| `multiturn_eval.json` | 25 | coreference, query rewriting, topic switching |
| `adversarial_eval.json` | 30 | injection, PII, guardrails |
| `vision_eval.json` | 25 | **text-only vs text+vision lift** |

### One design decision worth understanding

Gold labels reference **(document, section)**, not chunk IDs. Chunk IDs change
with every chunking ablation, which would invalidate the entire set on the first
experiment. Resolve section → chunk at eval time by matching section headers.

### Calibration anchor

τ-bench (Sierra) reports that strong function-calling agents succeed on **under
50%** of realistic customer-service tasks, with pass^8 below 25% on retail. If
your trajectory eval reports 95% on the first run, something is wrong with the
eval, not right with the agent. Consider reporting pass^k over repeated trials.

---

## Still to produce

**Vision screenshots.** `vision_eval.json` specifies 25 cases where an error code
exists only in an image, but the images themselves do not exist yet. Generate them
in Phase 9.5 — mock UI screenshots rendered as PNGs (checkout page with PAY-402,
monitor OSD with ERR-DP-0x004, and so on). Rendering them from HTML templates is
straightforward and keeps them reproducible.

---

## Regenerating

All scripts are seeded and reproducible:

```bash
python scripts/build_pdfs.py         # _source/*.md  -> documents/*.pdf
python scripts/seed_db.py            # -> db/pacify.db
python scripts/gen_intents.py        # -> intents/train.csv
python scripts/gen_testset.py        # -> intents/test_hard.csv
python scripts/gen_tickets.py        # -> tickets/ticket_history.csv
python scripts/gen_evalsets.py       # -> eval/*.json
python scripts/check_consistency.py  # verify corpus vs canonical facts
```

Requires `reportlab`, `faker`. Edit a document in `_source/` and rebuild — never
edit the PDFs directly.

---

## SQL layer

`sql/` contains the business-logic views and dashboard analytics. See
`sql/README.md`.

- `01_business_logic_views.sql` — 5 views: return eligibility, warranty status,
  refund waterfall, denormalised order detail, repeat-contact detection
- `02_analytics.sql` — 12 queries powering the admin dashboard

Eligibility and refund arithmetic are computed in SQL, not by the LLM. One
definition shared by the tool layer, the eval harness and the dashboard, and
verified against the 27 deterministic edge-case orders with no model in the loop.

`db/pacify.db` also carries `tickets_raw` (11,905 rows imported from
`tickets/ticket_history.csv`) so the analytics queries run against a single file.
