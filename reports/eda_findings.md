# PacifyIQ — EDA Findings

**Phases 2 and 3.** Dataset audit and exploratory analysis.

Every finding below is attached to a decision that changes what gets built. Figures
are in `reports/figures/`; the notebooks that produced them are in `notebooks/`.

Reproduce with:

```bash
python -m src.eda.audit        # Phase 2
python scripts/run_eda.py      # Phase 3
```

---

## Data inventory

| Asset | Rows | Columns | Notes |
|---|---|---|---|
| `tickets/ticket_history.csv` | 11,905 | 14 | 6 months, aggregate metadata only |
| `intents/train.csv` | 2,200 | 2 | template-generated |
| `intents/test_hard.csv` | 142 | 4 | hand-authored |
| `db/pacify.db :: orders` | 2,001 | 19 | joined to 500 customers, 14 products |
| `documents/*.pdf` | 47 pages | — | 13 documents, 16,208 words |
| `eval/*.json` | 295 cases | — | 7 evaluation sets |

**Structural note.** `ticket_history.csv` has **no free-text column.** It is aggregate
operational metadata — intent, sentiment, latency, tokens — not message content. All
text analysis in this report therefore uses `intents/*.csv`. This constrains what the
dashboard can show: it can report *that* an intent spiked, not *what customers said*.

**No image or attachment column exists in any tabular asset.** Screenshots are
specified in `eval/vision_eval.json` (25 cases) but the PNG files do not exist yet.
Generated in Phase 8.

---

## Part 1 — Audit findings (Phase 2)

### 🔴 A1. Confidence is leaked

| Group | Mean confidence |
|---|---|
| Resolved by AI | 0.815 |
| Escalated to human | 0.461 |

`confidence` was generated **from** `resolved_by` when the history was synthesised.
It separates the two outcomes almost perfectly.

> **Decision.** `confidence` must never be used as a feature to predict escalation.
> Any model using it will score near-perfectly and mean nothing. It is legitimate as a
> *dashboard display* value and as a calibration target in Phase 11, where the question
> is whether a *newly computed* confidence predicts outcomes — but the logged column
> is off-limits as an input.

Figure: `08_confidence_leakage.png`

### 🔴 A2. Train/test leakage — 2 exact overlaps

Two texts appear in both splits with identical labels:

- `"when will i get my money back"` — `payment_issue`
- `"how long does a warranty repair take"` — `warranty_claim`

Small in absolute terms, but both sit on **deliberately ambiguous boundaries**
(DEFECT-08 refund-vs-reversal; warranty turnaround). They are exactly the cases the
hard test set exists to probe.

> **Decision.** Remove both from `test_hard.csv` before Phase 4. Add a leakage
> assertion to the test suite so it cannot recur.

### 🟠 A3. 498 Sunday tickets contradict stated policy

`POL-CS-001 S1.2` states support does not operate on Sundays. The data has 498 Sunday
tickets (4.2%).

Weekday index relative to Friday:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| 0.96 | 0.96 | 0.96 | 0.97 | 1.00 | 0.52 | **0.23** |

> **Decision.** A generator defect — Sunday was down-weighted, not zeroed. Two options:
> regenerate with Sunday excluded, or keep it and document it as "asynchronous channels
> accept messages outside staffed hours." **Recommend keeping and documenting** — real
> support queues do receive out-of-hours contacts, and the pattern gives the trend
> detector a genuine seasonality problem to handle.

### 🟠 A4. Two test classes are too thin to score reliably

`shipping_delivery` and `account_management` have 5 test examples each.

> **Decision.** Report per-class F1 with the sample size alongside it, and do not draw
> conclusions from those two classes. Better: add ~10 hand-written examples to each.

### ✅ A5. Clean where it matters

- `ticket_id` and `order_id` are unique — primary keys hold
- No label conflicts in `train.csv` — every text maps to exactly one intent
- `subtopic` is strictly nested within `intent` — clean hierarchy, no crossover
- No orders delivered before dispatch
- **No PII detected** — no emails, phone numbers, card-like numbers, or IFSC codes in
  any message text
- All timestamps fall inside 09:00–21:00 IST

---

## Part 2 — EDA findings (Phase 3)

### Q1 & Q2 — What problems exist, and which dominate?

| Intent | Tickets | Share | Escalation |
|---|---|---|---|
| order_tracking | 2,231 | 18.7% | 12.7% |
| return_policy_question | 1,906 | 16.0% | 12.6% |
| return_refund_request | 1,420 | 11.9% | **71.1%** |
| shipping_delivery | 1,409 | 11.8% | 17.2% |
| product_information | 1,289 | 10.8% | 15.7% |
| technical_support | 1,244 | 10.4% | 37.1% |
| warranty_claim | 893 | 7.5% | **60.7%** |
| payment_issue | 647 | 5.4% | **62.6%** |
| account_management | 484 | 4.1% | **80.8%** |
| complaint | 264 | 2.2% | **90.5%** |
| out_of_scope | 118 | 1.0% | 7.6% |

**Volume and cost point in opposite directions.** The two largest intents (35% of
volume) escalate at ~13%. `complaint` is 2.2% of volume and escalates at 91%.

> **Decision.** Set deflection targets **per intent**, not globally. A single 65% target
> is meaningless when the achievable ceiling ranges from 9% to 92%. The Phase 13
> dashboard should show deflection against a per-intent target.

Figure: `01_intent_distribution.png`

### Q3 — Which issues generate negative sentiment?

| Intent | Negative % | High priority % |
|---|---|---|
| complaint | 84.5% | 37.5% |
| payment_issue | 63.1% | 26.6% |
| return_refund_request | 51.7% | 24.9% |
| warranty_claim | 49.3% | 26.4% |
| technical_support | 40.4% | 22.9% |
| order_tracking | 27.4% | 20.3% |

Money-adjacent intents carry the negative load. `payment_issue` is only 5.4% of volume
but 63% negative — consistent with the design note that missing money produces the
highest anxiety per ticket.

> **Decision.** Sentiment should drive **tone modulation** for `payment_issue`,
> `return_refund_request` and `complaint`. Confirmed as worth doing.

Figure: `02_sentiment_urgency.png`

### Q4 — Does sentiment predict escalation?

Correlation between negative sentiment and escalation: **r = 0.305**. Real but weak.

Within-intent escalation rates:

| Intent | Negative | Neutral | Positive | Spread |
|---|---|---|---|---|
| order_tracking | 29.6% | 6.3% | 6.7% | **+23pp** |
| shipping_delivery | 34.9% | 13.0% | 8.4% | +27pp |
| technical_support | 48.4% | 29.1% | 32.6% | +16pp |
| complaint | 90.6% | 91.2% | 85.7% | **+5pp** |
| account_management | 87.5% | 79.6% | 75.4% | +12pp |

Sentiment matters most where escalation is otherwise *rare*, and barely at all where
escalation is already the norm.

> **Decision.** Sentiment enters the Phase 11 escalation policy as **one weighted
> signal, not a trigger** — and its weight should be intent-conditional. An angry
> `order_tracking` customer is a meaningful signal; an angry `complaint` customer is
> not, because complaints escalate regardless. This is a more nuanced answer than the
> earlier "test it and see" plan, and it comes from the data.

### Q5 — Class imbalance

- **Train:** 400 (`order_tracking`) : 40 (`out_of_scope`) = **10.0×**
- **Majority-class baseline accuracy: 18.2%**
- **Test:** 19 : 5 = 3.8× — deliberately more uniform

> **Decisions.**
> 1. **Macro-F1 is the headline metric.** Accuracy is meaningless at this imbalance.
> 2. `class_weight="balanced"` in LogisticRegression.
> 3. Stratified splits everywhere.
> 4. Report per-class recall for `out_of_scope` (40 examples) separately — it is
>    load-bearing for the guardrail layer despite being the smallest class.
> 5. Do **not** rebalance the training set. The imbalance mirrors the real queue, and
>    the classifier's priors should reflect it.

Figure: `03_class_imbalance.png`

### Q6 — What does the language look like?

| Metric | Train | Test |
|---|---|---|
| Median words | 6 | 8 |
| Max words | 13 | 68 |
| Median sentences | 1.0 | 1.0 |
| Code-mixed (Hindi-English) | 10.1% | 7.0% |
| Contains order reference | 22.8% | 4.9% |
| Contains error code | 1.8% | 1.4% |
| Negative-lexicon hit | 9.8% | 14.1% |
| **Template rate** | **26.9%** | **0.0%** |

**Vocabulary:** train has 1,031 types over 14,591 tokens (TTR 0.071), 45.7% hapax.
90% of tokens are covered by 308 types; 95% by 444.

**Drift, train → test:**

| | |
|---|---|
| OOV **type** rate | **49.1%** |
| OOV **token** rate | **22.6%** |
| Shared vocabulary | 274 of 538 test types |

> **Decisions.**
> 1. `TfidfVectorizer(max_features≈1500, min_df=2, ngram_range=(1,2))`. The coverage
>    curve shows 444 types reach 95%; 1,500 with bigrams leaves headroom without
>    fitting noise. `min_df=2` drops the 471 hapax terms.
> 2. **Add character n-grams** (`analyzer='char_wb', ngram_range=(3,5)`) in a
>    `FeatureUnion`. With a 22.6% OOV token rate and 12% of training messages carrying
>    injected typos, word-level features alone will be blind on the test set.
> 3. **Expect a large macro-F1 drop** train → test. The 27% → 0% template gap plus the
>    22.6% OOV rate quantifies it in advance. This is the headline Phase 4 finding.

Figure: `04_text_characteristics.png`

**Train message length is nearly constant** (median 6, IQR 5–8, max 13) — a direct
consequence of template generation. Real support messages have a long right tail; the
test set reaches 68 words. Any model relying on length as a feature will not transfer.

### Q7 — Patterns that influence design

#### 7a. Order references contaminate the intent signal

Top lexical overlap: `order_tracking` ↔ `return_refund_request`, **Jaccard = 0.25**,
and the shared terms are entirely order-reference tokens: `12345`, `pac-2026-12345`,
`pac-2026-12350`, `order`, `my`, `for`.

Order-reference rate by intent:

| Intent | % containing an order ref |
|---|---|
| order_tracking | 77.2% |
| return_refund_request | 60.4% |
| payment_issue | 18.5% |

Both high-volume intents mention order IDs, and the specific ID digits are pure noise —
`PAC-2026-12345` carries no more intent signal than `PAC-2026-12350`, but TF-IDF treats
them as distinct features.

> **Decision.** Normalise before vectorising: replace order references with `<ORDER>`,
> error codes with `<ERRCODE>`, and bare digit runs with `<NUM>`. This should measurably
> reduce the `order_tracking` ↔ `return_refund_request` confusion. **Run it as an
> ablation** — with and without masking — and report the delta.

Figure: `05_class_overlap.png`

#### 7b. Weekday seasonality will break naive trend detection

Saturday runs at 52% of a weekday, Sunday at 23%.

> **Decision.** The emerging-issue detector must compare like-for-like. The current
> implementation uses a per-day-normalised 7-day window against a 28-day baseline,
> which handles this because both windows contain whole weeks. **Do not** switch to a
> shorter window without adding day-of-week adjustment.

Figure: `06_temporal_patterns.png`

#### 7c. 🔴 The corpus is much smaller than planned

Measured: **13 documents, 47 pages, 16,208 words ≈ 21,000 tokens.**

| Chunk size | Resulting chunks |
|---|---|
| 256 tokens | ~82 |
| 512 tokens | ~41 |
| 1024 tokens | ~21 |

**My earlier estimate of 1,800–2,200 chunks was wrong by roughly 50×.** This materially
changes three plans:

1. **Retrieval evaluation will be too easy.** With ~41 chunks and 120 eval questions,
   Recall@5 means retrieving 5 of 41 candidates. Metrics will pin near 1.0 and the
   ablations will not separate configurations — exactly the failure the messy corpus
   was designed to prevent.
2. **The ANN phase loses its point.** HNSW versus brute-force is meaningless at 41
   vectors; exact search is instant. The recall-vs-latency tradeoff has no room to
   appear.
3. **Precision@K becomes the more informative metric** than Recall@K at this scale.

> **Decision — three options, in preference order:**
>
> **(a) Chunk smaller, at 128–256 tokens.** Gives 82–164 chunks. Cheap, immediate, and
> arguably more appropriate for a policy corpus where the answer is usually one clause.
> Makes chunk-size ablation meaningful in the 128/256/512 range rather than
> 512/1024/2048.
>
> **(b) Expand the corpus to ~150 pages.** Authoring another ~10 documents (extended
> troubleshooting, more manuals, regional addenda for 3 more EU countries) would reach
> ~1,000 chunks at 256 tokens. Roughly 1.5 sessions. Restores the ANN phase.
>
> **(c) Keep as-is and report honestly** — run the ANN benchmark on synthetic vectors
> to demonstrate the technique, and state plainly that the production corpus is small
> enough for exact search, so FAISS Flat is what ships.
>
> **Recommendation: (a) now, (b) if time allows.** Option (c) is the honest fallback
> and is itself a defensible engineering answer — "we measured, exact search was
> sufficient at our corpus size, so we did not add an approximate index" is a better
> interview answer than an unnecessary HNSW index.

Figure: `07_corpus_structure.png`

#### 7d. Version pair is the most similar document pair

`return_policy_v2` and `return_policy_v1_ARCHIVED` are the highest-cosine pair in the
corpus, by construction.

> **Decision.** Pure semantic similarity cannot separate current from superseded
> policy. Chunks must carry a `version` flag in metadata (Phase 5) and the retriever
> must filter or down-weight archived content (Phase 7). Confirms the planned
> metadata-filtering ablation is necessary, not decorative.

#### 7e. Error codes fragment under subword tokenisation

Codes like `ERR-DP-0x004` and `SYS-0x0000007B` are alphanumeric with hyphens and hex.
They appear in 1–2 documents each, and only 1.8% of training messages contain one.

> **Decision.** Confirms **hybrid retrieval is required, not optional**. BM25 matches
> these exactly; dense embeddings place fragmented codes in a fuzzy neighbourhood of
> other alphanumeric strings. This is also why Phase 8's vision extraction matters —
> extracting a code from a screenshot is only useful if retrieval can then find it.

---

## Decisions summary

| # | Decision | Phase | From |
|---|---|---|---|
| 1 | Never use logged `confidence` as an escalation feature | 11 | A1 |
| 2 | Remove 2 leaked test rows; add a leakage assertion | 4 | A2 |
| 3 | Keep Sunday tickets, document as out-of-hours contacts | 3 | A3 |
| 4 | Macro-F1 headline; `class_weight="balanced"`; stratified splits | 4 | Q5 |
| 5 | Do not rebalance training data — imbalance mirrors the real queue | 4 | Q5 |
| 6 | TF-IDF: `max_features≈1500, min_df=2, ngram=(1,2)` | 4 | Q6 |
| 7 | Add char n-grams (3–5) via FeatureUnion for OOV robustness | 4 | Q6 |
| 8 | Mask order refs / error codes / digits before vectorising; ablate it | 4 | 7a |
| 9 | Per-intent deflection targets, not a single global number | 13 | Q1 |
| 10 | Sentiment as an intent-conditional weighted signal, not a trigger | 11 | Q4 |
| 11 | Sentiment drives tone for payment / refund / complaint | 9 | Q3 |
| 12 | **Chunk at 128–256 tokens; ablate 128/256/512** | 5, 6 | 7c |
| 13 | Consider expanding the corpus to ~150 pages | 5 | 7c |
| 14 | Chunks must carry a `version` flag; retriever must filter | 5, 7 | 7d |
| 15 | Hybrid retrieval confirmed necessary for error codes | 7 | 7e |
| 16 | Trend detector: keep 7d/28d windows; do not shorten without DoW adjustment | 13 | 7b |

---

## Honest limitations

Stated here so they can be repeated in the README and in interviews rather than
discovered by a reviewer.

1. **All data is synthetic.** Pacify Electronics Pvt. Ltd. is fictional. `train.csv` is
   template-generated (26.9% template rate). `test_hard.csv` is hand-authored but by an
   LLM, not by a human — the train/test gap finding would be stronger with 30–50
   human-written examples.
2. **`ticket_history.csv` is simulated** with deliberately planted trends. It is
   labelled as such in the UI and README. It contains no message text, so it cannot
   support text-level analysis.
3. **The corpus is small** (16,208 words). See 7c. Retrieval metrics from this corpus
   will be optimistic relative to a production knowledge base.
4. **No real user interactions exist yet.** Every dashboard number in Phases 1–12 comes
   from simulated history. Real traces only begin accumulating in Phase 13.
5. **Sentiment labels are generated, not annotated.** The intent-conditional escalation
   finding (Q4) rests on synthetic sentiment and should be re-validated against real
   labels before being trusted operationally.

---

## Processed outputs

Written to `data/processed/` for reuse by Phase 4:

| File | Contents |
|---|---|
| `train_features.parquet` | 2,200 rows × 20 engineered text features |
| `test_features.parquet` | 142 rows × 23 features |
| `intent_profile.csv` | Per-intent volume, escalation, sentiment, latency, tokens |
| `class_overlap.csv` | 55 pairwise Jaccard overlaps |
| `distinctive_terms.csv` | Top TF-IDF terms per intent |
| `text_stats.json` | Vocabulary, drift, template statistics |

---

## Next

**Phase 4 — NLP classification baseline.** Every hyperparameter is now justified by a
finding above rather than chosen by convention.
