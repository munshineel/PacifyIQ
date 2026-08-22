# PacifyIQ — Understanding-Retrieval Bridge

**Phase 6.5.** Connect the Phase 4 classifier and entity extractor to the Phase 6
retriever, and measure whether it actually helps.

```bash
python scripts/evaluate_routing.py
```

---

## 1. Why this phase exists

Phases 4 and 6 both worked and neither knew the other existed. Two failure
clusters in the Phase 6/7 reports pointed at the same missing connection:

| Failing query | Gold section | Why retrieval failed |
|---|---|---|
| *"Am I talking to a bot?"* | `customer_service_policy S12` | BM25 score **0.00** — no shared vocabulary |
| *"Can the AI approve my refund?"* | `customer_service_policy S12` | every term points at refund documents |
| *"How do I escalate my issue?"* | `customer_service_policy S3` | falsely abstained |
| *"What does SYS-0x0000007B mean?"* | `technical_support_faq S9` | gold at rank 10 |

The classifier and the entity extractor already had the missing signal. Nothing
was passing it along.

---

## 2. The design constraint, found by measuring first

The obvious design is *"classify the intent, filter retrieval to that topic."*
It does not survive contact with the data.

Running the classifier on the queries retrieval actually fails:

| Query | Predicted intent | Margin | Correct? |
|---|---|---|---|
| *"Am I talking to a bot?"* | `complaint` | 0.103 | ✗ |
| *"What do you need to verify my identity?"* | `shipping_delivery` | 0.035 | ✗ |
| *"What does SYS-0x0000007B mean?"* | `payment_issue` | 0.036 | ✗ |
| *"My laptop will not turn on"* | `warranty_claim` | 0.036 | ✗ |

**All four wrong, all four with a margin under 0.11.** This is not a coincidence:
the queries retrieval finds hard are the same queries the classifier finds hard,
because both struggle with the same thing — unusual phrasing.

> A hard topic filter driven by a classifier this uncertain would **remove the
> correct document more often than it removes noise.** The failure modes are
> correlated, so stacking them compounds rather than cancels.

So the bridge applies three graded signals instead, in order of reliability:

1. **Entity override** — an extracted error code is deterministic and outranks the classifier entirely
2. **Intent boost** — a score multiplier, gated on margin, never a filter
3. **Query enrichment** — entity tokens appended so BM25 matches exactly

Nothing is ever filtered out. A wrong boost costs rank position, not availability.

---

## 3. Candidate injection — the part that actually mattered

The first implementation boosted scores and changed almost nothing. The reason
is structural:

> **Boosting can only reorder what is already in the candidate pool.** For a
> meta-question, the answering document shares almost no vocabulary with the
> query, so it never enters the pool at all — and no amount of re-ranking
> rescues a document that was never retrieved.

The fix is a second, targeted retrieval restricted to the boosted documents,
merged into the pool before re-ranking. This required adding a `docs` filter to
the base retriever.

Before and after, on *"How do I escalate my issue?"*:

```
boost only:   FAQ-TEC-001 S6 · FAQ-PRD-001 S4 · FAQ-PRD-001 S4     (all wrong)
+ injection:  POL-CS-001 S3  · POL-CS-001 S4  · POL-CS-001 S12     (gold at 1)
```

This is the difference between routing that works and routing that only looks
like it works.

---

## 4. Component ablation

120 retrieval queries, 40 unanswerable, 60 answerable for false-abstention.

| Configuration | R@1 | R@3 | R@5 | Cov@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| Baseline (no routing) | 0.542 | 0.742 | 0.875 | 0.793 | 0.664 | 0.726 |
| Entities only | 0.550 | 0.775 | 0.875 | 0.797 | 0.682 | 0.754 |
| **Entities + intent (margin 0.25)** | **0.575** | **0.817** | **0.883** | **0.814** | **0.706** | **0.782** |
| Entities + intent (margin 0.40) | 0.550 | 0.800 | 0.875 | 0.797 | 0.688 | 0.758 |
| Entities + intent + meta | 0.542 | 0.792 | 0.883 | 0.814 | 0.682 | 0.762 |
| **Meta only** | 0.517 | 0.733 | **0.858** | 0.781 | **0.650** | 0.707 |
| Intent only (margin 0.15) | 0.525 | 0.783 | 0.858 | 0.785 | 0.665 | 0.736 |

**Selected: entities + intent at margin 0.25, meta off.** Best on every metric.

### Delta against baseline

| Metric | Before | After | Δ |
|---|---|---|---|
| Recall@1 | 0.542 | 0.575 | **+0.033** |
| Recall@3 | 0.742 | 0.817 | **+0.075** |
| Recall@5 | 0.875 | 0.883 | +0.008 |
| Coverage@5 | 0.793 | 0.814 | +0.021 |
| **MRR** | 0.664 | **0.706** | **+0.042** |
| **nDCG@5** | 0.726 | **0.782** | **+0.056** |
| Abstention | 0.700 | 0.700 | 0.000 |

**The gain is concentrated in ranking, not coverage.** Recall@5 barely moves
(+0.008) while MRR gains +0.042 and recall@3 gains +0.075. Routing does not find
documents retrieval was missing; it **moves the right document up**. Given that
Phase 7 sends 5 chunks to the model and the top of that list gets the most
attention, ranking is the more useful thing to improve.

---

## 5. ⚠️ The meta-question component measurably hurts

The component built specifically to fix *"Am I talking to a bot?"* makes overall
retrieval **worse**: recall@5 0.875 → 0.858, MRR 0.664 → 0.650.

**It fixed 3 target queries and broke 6 others:**

| Broken | Gold | What routing returned instead |
|---|---|---|
| *"What is the difference between an exchange and a replacement?"* | `return_policy_v2 S12` + `warranty_policy S11` | three `product_faq` chunks |
| *"When should I stop troubleshooting and claim warranty?"* | `technical_support_faq S12` | three `warranty_policy` chunks |
| *"Can I get a loaner during repair?"* | `warranty_policy S12` | `customer_service_policy S2` |
| *"What is the inspection fee if my claim is rejected?"* | `warranty_policy S3` | three `return_policy_v2` chunks |

The cause is the trigger being too loose. Any query containing *"escalate"*,
*"verify"*, *"agent"* or *"respond"* injected `customer_service_policy` chunks,
which then displaced correct evidence for questions that merely used those words
in passing.

**`use_meta` defaults to `False`.** The idea is sound — meta-questions genuinely
need routing, and the injection mechanism works when it fires correctly — but on
this corpus the trigger costs more than it returns. The implementation is
retained behind a flag with the measurement recorded, rather than quietly
dropped.

A tighter trigger (requiring a meta marker **and** an absence of product or
policy vocabulary) is the obvious next attempt.

---

## 6. Margin threshold sweep

How confident must the classifier be before its intent is trusted?

| Min margin | Recall@5 | MRR |
|---|---|---|
| 0.00 (always trust) | **0.808** | 0.599 |
| 0.10 | 0.858 | 0.668 |
| 0.15 | 0.867 | 0.662 |
| **0.25** | **0.883** | **0.706** |
| 0.40 | 0.867 | 0.664 |
| 1.01 (never trust) | 0.867 | 0.658 |

**Trusting the classifier unconditionally is worse than ignoring it entirely** —
0.808 versus 0.867. That is the clearest possible statement of the design
constraint from §2, and it is a number rather than an intuition.

The curve is single-peaked at 0.25, which is reassuring: too low and wrong
intents inject noise, too high and the signal is never used.

---

## 7. Target queries

| ID | Query | Before | After | Rank |
|---|---|---|---|---|
| R059 | *"Can the AI approve my refund?"* | ✗ | ✅ | — → **1** |
| R064 | *"What does SYS-0x0000007B mean?"* | ✗ | ✅ | 10 → **2** |
| R022 | *"How long is the warranty on a Pacify laptop?"* | ✗ | ✅ | 7 → **1** |
| R029 | *"My Northwind laptop is faulty..."* | ✅ | ✅ | 3 → 2 |
| R030 | *"How long does a warranty repair take?"* | ✅ | ✅ | 5 → 4 |
| R073 | *"ProBook 14 vs 16?"* | ✗ | ✗ | 9 → 6 |
| R078 | *"I am in Germany..."* | ✅ | ✅ | 4 → 2 |
| R058 | *"Am I talking to a bot?"* | ✗ | ✗ | — → — |

**3 of 15 fixed outright, 8 improved in rank, 0 broken** in the shipped
configuration.

R058 remains unfixed because it needs the meta component, which is off. That is
a deliberate trade: one query against six.

---

## 8. Abstention held

Boosting raises retrieval scores, and abstention thresholds on those scores — so
a routing gain can silently destroy the refusal behaviour. Checked explicitly:

| Metric | Before | After |
|---|---|---|
| Abstention rate | 0.700 | **0.700** |
| Mean BM25 when answering | 10.71 | 13.94 |
| Mean BM25 when abstaining | 5.52 | 5.52 |

Unchanged. Enrichment raises scores for questions the corpus *can* answer
without raising them for questions it cannot — because there are no entities to
extract from *"do you offer student discounts."* There is a test asserting this
does not regress.

### A bug found on the way

Faithfulness briefly read 0.96 after wiring routing in. The cause was not a
hallucination: **the grounding check was scoring correct abstentions as
ungrounded** because *"I don't have documentation covering that"* carries no
citation.

An abstention asserts nothing, so there is nothing to support — it is grounded
by definition. The metric was inverted exactly where it mattered most. Fixed and
regression-tested.

---

## 9. What ships

```python
RoutedRetriever(
    retriever,
    use_entities=True,     # error codes, products -> document boost + enrichment
    use_intent=True,       # topic/doc boost, gated at margin >= 0.25
    use_meta=False,        # measured to hurt; retained behind a flag
    use_enrichment=True,   # entity tokens appended to the query
)
```

Wired into `RAGPipeline` by default, with a `use_routing=False` escape hatch. The
trace now carries `intent`, `intent_margin`, `sentiment`, `urgency` and
`routing_reasons`, so every routing decision is auditable from the dashboard.

---

## 10. Decisions

| # | Decision | Evidence |
|---|---|---|
| 1 | Boost, never filter | Classifier is wrong *and* uncertain on the hard queries |
| 2 | Gate intent at margin ≥ 0.25 | Swept; unconditional trust scores 0.808 vs 0.867 for ignoring it |
| 3 | Entities override intent | Deterministic beats probabilistic; `SYS-0x0000007B` misclassified as `payment_issue` at margin 0.04 |
| 4 | Candidate injection, not just re-ranking | Boosting cannot rescue a document outside the pool |
| 5 | `use_meta=False` | Fixed 3, broke 6 |
| 6 | Region is a filter, not a boost | The EU addendum legally overrides base policy |
| 7 | Abstention is grounded by definition | A refusal asserts nothing |

---

## 11. Honest limitations

1. **The gain is modest** — recall@5 +0.008. The real movement is in ranking
   (MRR +0.042, nDCG +0.056), which matters for generation but is a smaller
   headline than it might appear.
2. **Meta-questions remain unsolved.** The component that addresses them is off
   because its trigger is too loose. R058 still fails.
3. **`INTENT_ROUTING` is hand-authored**, not learned. It encodes an assumption
   about which document answers which intent. On a larger corpus this would need
   to be derived from data.
4. **Adds latency.** Every query now runs classification (~2.5 ms) plus an extra
   retrieval pass for injection. Measured at ~130 ms end to end versus ~50 ms
   without — acceptable, but not free.
5. **The margin threshold is fitted to 120 queries.** With that sample size, 0.25
   versus 0.15 is within noise; the single-peaked curve is more convincing than
   the exact value.

---

## Established

```
question → understand → route → retrieve → assemble → gate → generate
```

Understanding and retrieval are now one pipeline. **MRR 0.664 → 0.706, nDCG
0.726 → 0.782, abstention unchanged at 0.700**, with the component that failed
reported rather than hidden.

**26 Phase 6.5 tests. 211 across the project.**
