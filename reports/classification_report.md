# PacifyIQ — NLP Classification Baseline

**Phase 4.** A transparent, measured baseline before any LLM enters the system.

Reproduce:

```bash
python scripts/train_intent_classifier.py
python scripts/evaluate_sentiment_urgency.py
```

Result tables: `reports/results/`. Model artifact: `models/intent_classifier.joblib`.

---

## 1. Which tasks the data actually supports

The first thing to check was whether the three requested tasks are all learnable
from this dataset. They are not.

| Asset | Has text | intent | sentiment | urgency |
|---|---|---|---|---|
| `intents/train.csv` | ✅ | ✅ | ✗ | ✗ |
| `intents/test_hard.csv` | ✅ | ✅ | ✗ | ✗ |
| `tickets/ticket_history.csv` | ✗ | ✅ | ✅ | ✅ (`priority`) |

**Labels and text live in different files.** `ticket_history.csv` is aggregate
operational metadata with no message content, so there is nothing to fit a text
classifier on. Supervised text → sentiment and text → urgency are **not possible**
with this data.

That produces three different treatments rather than three identical ones:

| Task | Approach | Why |
|---|---|---|
| **Intent** | Full supervised ML | 2,200 labelled messages, 11 classes |
| **Sentiment** | Rule-based lexicon + intent prior | No text-level labels exist |
| **Urgency** | Rule-based, five weighted signals | No text-level labels exist |

Generating sentiment labels with an LLM and then "evaluating" against them would
measure agreement with the labeller, not accuracy. The rule-based scorers are
transparent instead: every score reports exactly which terms fired.

---

## 2. Intent classification

### 2.1 A methodological problem, found and fixed

The first run produced a result that looked excellent and was worthless:

```
TF-IDF word + LogReg        val macro-F1 0.9929
TF-IDF word + LinearSVC     val macro-F1 0.9929
TF-IDF union + LogReg       val macro-F1 0.9929
TF-IDF union + LinearSVC    val macro-F1 0.9929
... 9 models tied at exactly 0.9929
```

**Nine models tied to four decimal places.** The validation metric could not
discriminate at all, and selecting on it picked LinearSVC by sort order — which
turned out to be one of the *weakest* of the tied models on held-out data
(test macro-F1 0.5095 versus 0.6098 for another tied candidate).

**Cause.** EDA measured a 26.9% template rate in the training data. A random
stratified split puts rows generated from the *same template* on both sides, so
the model can score near-perfectly by recognising phrasings it has already seen.
Between the random split's train and validation halves, **180 template skeletons
were shared**.

**Fix — group-aware splitting.** Split by template skeleton so no template appears
in both halves, forcing validation to contain unseen phrasings.

| Split | Shared templates | Score spread across models |
|---|---|---|
| Random stratified | 180 | 0.9311 – 0.9929 (9-way tie at top) |
| **Group by template** | **0** | **0.8457 – 0.9940 (discriminating)** |

A single group split still tied two candidates at 0.9940, so final selection uses
**repeated group splits across five seeds**, averaged, with a standard deviation.
The held-out test set is never consulted for selection.

This is the most transferable lesson in the phase: *a validation metric that
cannot separate models is not a validation metric.*

### 2.2 Splits

| Split | n | Purpose |
|---|---|---|
| Train | 1,767 | fitting |
| Validation (group-aware) | 433 | model selection |
| Test (`test_hard.csv`) | 140 | reported once, at the end |

**Leakage removed.** EDA finding A2 identified two texts present in both
`train.csv` and `test_hard.csv`. Both were removed and an assertion added so it
cannot recur:

- `"when will i get my money back"` — `payment_issue`
- `"how long does a warranty repair take"` — `warranty_claim`

Both sit on deliberately ambiguous boundaries, which is exactly what the hard
test set exists to probe.

Group splitting trades some stratification for honesty: the smallest validation
class holds 2 examples. That is a real cost, disclosed rather than hidden.

### 2.3 Model comparison

Selection metric is **macro-F1**. Accuracy is reported but never used — the classes
are imbalanced 10:1 and a majority-class predictor scores 18.2% accuracy while
being useless.

| Model | Random split | **Group split** | Test | Gap |
|---|---|---|---|---|
| **TF-IDF union(char=1.0) + LinearSVC** | 0.9929 | **0.9940** | **0.6098** | 0.384 |
| TF-IDF union(char=1.5) + LinearSVC | 0.9851 | 0.9940 | 0.5761 | 0.418 |
| TF-IDF union + LinearSVC | 0.9929 | 0.9845 | 0.5726 | 0.412 |
| TF-IDF union(char=1.0) + LogReg | 0.9929 | 0.9712 | 0.5984 | 0.373 |
| TF-IDF word + SGD | 0.9852 | 0.9673 | 0.5442 | 0.423 |
| TF-IDF union + LogReg | 0.9929 | 0.9616 | 0.5863 | 0.375 |
| TF-IDF word + LinearSVC | 0.9929 | 0.9390 | 0.5095 | 0.430 |
| TF-IDF word + LogReg | 0.9929 | 0.9066 | 0.5223 | 0.384 |
| TF-IDF char + LogReg | 0.9802 | 0.8815 | 0.6162 | **0.265** |
| TF-IDF word + ComplementNB | 0.9311 | 0.8457 | 0.4793 | 0.366 |
| Stratified baseline | 0.0997 | 0.0872 | 0.0851 | — |
| Majority baseline | 0.0280 | 0.0171 | 0.0154 | — |

**Tie-break, repeated group splits (5 seeds):**

| Model | mean ± sd | range |
|---|---|---|
| **union(char=1.0) + LinearSVC** | **0.9427 ± 0.0077** | 0.928 – 0.950 |
| union(char=1.5) + LinearSVC | 0.9398 ± 0.0122 | 0.927 – 0.960 |
| union(char=1.0) + LogReg | 0.9285 ± 0.0144 | 0.902 – 0.944 |
| union + LinearSVC (no mask) | 0.9249 ± 0.0095 | 0.910 – 0.937 |
| union + LinearSVC | 0.9205 ± 0.0128 | 0.910 – 0.945 |

The winner also has the **lowest variance**, which matters more than a marginally
higher mean.

### 2.4 Selected baseline

**TF-IDF word(1,2) + char_wb(3,5) union, char weight 1.0 → LinearSVC, class_weight balanced**

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 0.9954 | 0.6357 |
| Balanced accuracy | 0.9912 | 0.6293 |
| **Macro F1** | **0.9940** | **0.6107** |
| Weighted F1 | 0.9953 | 0.6248 |
| Macro precision | 0.9972 | 0.6449 |
| Macro recall | 0.9912 | 0.6293 |

Artifact: 257 KB · fit time 0.25 s · inference 2.45 ms mean.

**Hyperparameters, all traced to an EDA finding:**

| Setting | Value | From |
|---|---|---|
| `max_features` (word) | 1500 | Coverage curve: 444 types = 95% of tokens |
| `min_df` | 2 | 45.7% of vocabulary is hapax |
| `ngram_range` (word) | (1, 2) | — |
| char n-grams | `char_wb` (3,5) | 22.6% OOV token rate on test |
| char weight | 1.0 | Swept 0.6 / 1.0 / 1.5; OOV rate argued for raising it |
| `class_weight` | `balanced` | 10:1 imbalance |
| Resampling | **none** | Imbalance mirrors the real queue; priors should reflect it |

### 2.5 The train → test gap is the headline result

**Validation 0.9940 → test 0.6107. A drop of 0.383.**

This was predicted from EDA before any model was trained:

| Signal | Train | Test |
|---|---|---|
| Template rate | 26.9% | 0.0% |
| OOV token rate | — | 22.6% |
| OOV type rate | — | 49.1% |
| Median length | 6 words | 8 words |
| Max length | 13 words | 68 words |
| Compound messages | ~0% | 42% |

**Character n-grams close about a third of the gap.** The word-only model drops
0.384–0.430; the char-only model drops 0.265; the union lands at 0.384 with a much
higher ceiling. This confirms the EDA-driven decision to add them.

> **The honest reading.** 0.61 macro-F1 is what a bag-of-words model achieves on
> hand-written messages when trained on templates. It is not a weakness of TF-IDF —
> it quantifies how far synthetic training data sits from real customer language.
> A model reporting 0.99 on both splits would mean the test set was too easy.

### 2.6 Per-class performance (test)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| account_management | 0.500 | 1.000 | 0.667 | 5 |
| payment_issue | 0.769 | 0.714 | 0.741 | 14 |
| out_of_scope | 0.900 | 0.600 | 0.720 | 15 |
| warranty_claim | 0.688 | 0.733 | 0.710 | 15 |
| order_tracking | 0.727 | 0.615 | 0.667 | 13 |
| technical_support | 0.615 | 0.615 | 0.615 | 13 |
| complaint | 0.700 | 0.583 | 0.636 | 12 |
| product_information | 0.500 | 0.583 | 0.538 | 12 |
| shipping_delivery | 0.400 | 0.800 | 0.533 | 5 |
| return_policy_question | 0.450 | 0.529 | 0.486 | 17 |
| **return_refund_request** | **0.500** | **0.316** | **0.387** | 19 |

⚠️ `shipping_delivery` and `account_management` have 5 test examples each. Their
F1 has a very wide confidence interval and should not be read as a real estimate.

---

## 3. Failure analysis

**55 of 140 misclassified (39.3%).**

### 3.1 Top confusions

| True | Predicted | n | % of true |
|---|---|---|---|
| return_refund_request | return_policy_question | 8 | 42.1% |
| return_policy_question | return_refund_request | 3 | 17.6% |
| return_policy_question | product_information | 3 | 17.6% |
| order_tracking | shipping_delivery | 2 | 15.4% |
| out_of_scope | account_management | 2 | 13.3% |
| technical_support | return_refund_request | 2 | 15.4% |

### 3.2 The hardest boundary is informational vs transactional

`return_policy_question` ↔ `return_refund_request` accounts for **11 of 55 errors**.
The distinguishing signal is not vocabulary — both discuss returns, windows and
fees. It is whether the customer possesses a specific order and intends to act:

- *"can people return opened laptops"* → informational
- *"can i return this"* → transactional
- *"i am thinking of returning the monitor but wanted to check the fee first"* → ambiguous even to a human

A bag-of-words model has no representation for possession or intent-to-act. This
is a genuine ceiling on the approach, not a tuning failure.

### 3.3 Error rate by annotated case type

Every test case carries a note explaining why it is hard:

| Case type | n | Error rate |
|---|---|---|
| Compound messages | 6 | 66.7% |
| Planted defects | 13 | 61.5% |
| Boundary pairs | 11 | 45.5% |
| Image-dependent | 4 | 25.0% |
| Security-sensitive | 3 | 0.0% |
| Hallucination bait | 5 | 0.0% |

The classifier fails exactly where the test set was designed to be hard, and
succeeds on the categories that are lexically distinctive.

### 3.4 Compound messages need a fairer metric

42% of the test set carries two genuine intents. Strict single-label scoring marks
a prediction wrong even when it correctly identifies the *second* intent present.

| Scoring | Overall | Compound only | Simple only |
|---|---|---|---|
| Strict (primary only) | 0.607 | **0.397** | 0.756 |
| Lenient (primary or secondary) | 0.743 | **0.724** | — |

On compound messages, strict scoring reports 39.7% and lenient 72.4%. **Roughly
half the "failures" on compound messages are the model correctly naming one of the
two intents present.**

> **Design consequence.** Single-label classification cannot represent these
> messages. The classifier emits a primary label plus a full score distribution,
> and the `is_multi_intent` flag fires when the top-two margin is below 0.15. The
> agent handles decomposition; the classifier handles routing and analytics.

### 3.5 Masking ablation — a negative result

EDA finding 7a showed the largest class overlap (`order_tracking` ↔
`return_refund_request`, Jaccard 0.25) was driven **entirely** by order-reference
tokens. Masking them to `<ORDER>` was predicted to help.

**It did not.**

| Model | Group split Δ | Test Δ |
|---|---|---|
| word + LogReg | −0.006 | **−0.037** |
| union + LogReg | −0.004 | **−0.007** |
| union + LinearSVC | 0.000 | **−0.005** |

Masking is neutral-to-slightly-harmful on every model tested. Two reasons:

1. **`min_df=2` already handles it.** Each specific order ID appears once or twice,
   so most were dropped as hapax before masking could matter.
2. **The presence of an order reference is itself informative.** 77% of
   `order_tracking` and 60% of `return_refund_request` messages contain one, versus
   under 20% elsewhere. Masking preserves that signal, but collapsing distinct IDs
   into one high-frequency token makes it *less* discriminative between the two
   intents that both use it heavily.

The confusion masking was meant to fix was already **zero on validation** for the
union models — the hypothesis was addressing a problem the character features had
already solved.

> **The masking is retained but the hypothesis is reported as refuted.** It costs
> ~0.005 macro-F1 and buys robustness against order-ID formats not present in
> training, plus a small privacy benefit. That is a defensible trade, but it is a
> trade — not the improvement the EDA predicted.

---

## 4. Sentiment and urgency

### 4.1 Approach

Rule-based, for the reason in section 1. Sentiment combines a weighted domain
lexicon (140 terms), negation handling, shouting and punctuation signals, and an
**intent-conditional prior** measured from the 11,905-row ticket history.

The prior is what makes it more than a word list: *"payment failed"* contains no
sentiment vocabulary but comes from a category that is 63% negative in the
observed history.

Critically, the prior may only push the score **toward negative, never toward
positive**. A 27% negative base rate does not mean 73% positive — the remainder is
overwhelmingly neutral. Without that constraint the scorer labelled *"where is my
order 12345"* as positive.

Urgency combines five weighted signals: urgency vocabulary, intent base rate,
sentiment, repeat-contact markers, and legal-threat detection. A legal threat is a
hard escalation trigger under `POL-CS-001 S3.4(d)` and bypasses the weighted blend.

### 4.2 Results

Evaluated on 65 hand-annotated messages (`data/eval/sentiment_urgency_eval.json`).

**Sentiment**

| Metric | Value |
|---|---|
| Accuracy | 0.831 |
| **Macro F1** | **0.808** |
| Weighted F1 | 0.832 |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| negative | 0.903 | 0.848 | 0.875 | 33 |
| neutral | 0.769 | 0.833 | 0.800 | 24 |
| positive | 0.750 | 0.750 | 0.750 | 8 |

**Intent-prior ablation:** lexicon alone 0.753 → with prior **0.808**, a gain of
**+0.055**. The prior earns its place.

**Urgency**

| Metric | Value |
|---|---|
| Accuracy | 0.769 |
| **Macro F1** | **0.737** |
| Within one level | **0.954** |
| High scored as Low | 2 of 11 (18%) |

Urgency is ordinal, so distance matters more than exact match: 95.4% of predictions
land within one level, and only 3 cases are two levels off.

### 4.3 Failure modes

**The costly error — high urgency scored low (2 cases):**

- *"Absolutely disgusted. Ordered a 90k laptop, arrived with a cracked screen, and your support has ignored 4 emails."* — strong affect but no urgency vocabulary, no legal threat, no explicit repeat-contact marker
- *"i think someone hacked my account"* — a security incident with entirely calm language

**Both reveal the same weakness:** the scorer keys on *how the customer speaks*
rather than *what happened*. A composed report of a serious problem scores low.

**Sentiment over-firing on informational messages (2 cases):**
*"what payment methods do you accept"* and *"is my Phone X still under warranty"*
were scored negative because their intent carries a high negative base rate. The
prior is too strong when the lexicon is entirely silent.

**Sentiment under-firing on factual complaints (5 cases):** *"the free gift wasnt
in the box"*, *"my keyboard has a sticky key"* — genuine dissatisfaction described
without emotional vocabulary.

### 4.4 ⚠️ Limitation, stated plainly

The 65 annotations were **authored by an LLM, not by a human**, with a single
annotator and therefore no inter-annotator agreement. They are adequate for
catching gross calibration errors — and they did, revealing two bugs during
development — but **inadequate for a confident accuracy claim**.

Do not quote 0.808 or 0.737 as validated performance. Quote them as indicative,
with this caveat attached.

---

## 5. Production decision

### 5.1 The intent classifier ships. Here is why.

| Property | TF-IDF + LinearSVC | LLM call |
|---|---|---|
| Latency | **2.45 ms** | ~300 ms |
| Cost | **0** | per token |
| Determinism | **exact** | varies across runs |
| Size | **257 KB** | n/a |
| Network | **none** | required |
| Test macro-F1 | 0.61 | untested here |

Four reasons it stays in production even though an LLM would likely score higher:

**1. It runs before the LLM and shapes what the LLM sees.** Intent selects the
prompt, the retrieval filter and the tool subset. Using an LLM call to decide what
to send to an LLM call doubles latency and cost for the same routing decision.

**2. The dashboard needs a stable measurement instrument.** Intent distribution and
per-intent deflection are tracked over months. If the classifier is the LLM, then
swapping models silently shifts every trend line, and you cannot tell a real change
in customer behaviour from a change in your own classifier. A frozen 257 KB
artifact is a *fixed instrument*.

**3. It provides an independent escalation signal.** Phase 11 combines four
signals, and they must be genuinely independent. If intent confidence comes from
the same LLM that produced the answer, it is not independent evidence — it is the
same model marking its own work. The classifier margin is the only understanding
signal not produced by the LLM.

**4. It fits the deployment target.** 257 KB, no `torch`, no network. Streamlit
Community Cloud allows ~1 GB total.

### 5.2 What it is *not* used for

- **Not the final routing authority.** The agent decomposes compound messages; the
  classifier's 39.7% strict accuracy on those makes it unsuitable as sole router.
- **Not a confidence source.** LinearSVC has no `predict_proba`; the softmaxed
  margins are *scores*, not calibrated probabilities. Calibration is Phase 11 work
  using multiple signals.
- **Not used where its score is low.** Below the 0.15 multi-intent margin, the flag
  fires and the agent takes over.

### 5.3 What would change the decision

| Trigger | Action |
|---|---|
| Real traces accumulate with human-agent labels | Retrain on real data; the template gap should close substantially |
| Test macro-F1 stays below ~0.55 after retraining | Reconsider — routing errors would be costing more than the latency saves |
| Compound messages exceed ~50% of live traffic | Move to multi-label classification |
| Latency budget relaxes above ~500 ms | Reconsider an LLM classifier for accuracy |

### 5.4 The transformer benchmark

A DistilBERT fine-tune is **deliberately deferred**, not skipped. It needs `torch`
(~800 MB resident, ~2.5 GB on disk) which does not fit the deployment target, so it
can only ever be an offline benchmark.

The honest framing for that benchmark, when it runs: *"DistilBERT scored X, TF-IDF
scored 0.61, and we deployed TF-IDF because the 250 MB model could not fit the
target and the gap did not justify a different deployment."* That is a stronger
answer than "we deployed a transformer" — it shows a deployment constraint being
reasoned about rather than ignored.

---

## 6. Artifacts

| Path | Contents |
|---|---|
| `models/intent_classifier.joblib` | Full pipeline: preprocessing → vectorisers → LinearSVC (257 KB) |
| `models/intent_classifier_metadata.json` | Selection protocol, metrics, removed leaked rows, seed |
| `reports/results/model_comparison.csv` | All 16 models × 3 splits |
| `reports/results/repeated_group_cv.csv` | 5-seed tie-break |
| `reports/results/masking_ablation.csv` | The negative result |
| `reports/results/per_class_{test,val}.csv` | Per-class precision/recall/F1 |
| `reports/results/confusion_{test,val}.csv` | Full confusion matrices |
| `reports/results/top_confusions.csv` | Ranked systematic errors |
| `reports/results/failure_cases.csv` | All 55 misclassified, with annotations |
| `reports/results/compound_scoring.json` | Strict vs lenient |
| `reports/results/sentiment_urgency_*.{csv,json}` | Sentiment/urgency evaluation |

Preprocessing lives **inside** the sklearn Pipeline, so it is persisted with the
model and cannot drift from what was used at training time.

---

## 7. Summary of decisions

| # | Decision | Evidence |
|---|---|---|
| 1 | Group-aware splitting for selection | Random split tied 9 models at 0.9929 |
| 2 | Repeated group splits (5 seeds) as tie-break | One split still tied 2 candidates |
| 3 | Macro-F1 as selection metric | 10:1 imbalance; majority baseline 18.2% accuracy |
| 4 | Add char n-grams | 22.6% OOV; closes ~⅓ of the train→test gap |
| 5 | char weight 1.0, not the 0.6 default | Swept; measurably better and lower variance |
| 6 | Keep masking despite refutation | Costs 0.005, buys format robustness |
| 7 | No resampling | Imbalance mirrors the real queue |
| 8 | Report strict *and* lenient on compound | 42% of test carries two intents |
| 9 | Rule-based sentiment and urgency | Data cannot support supervision |
| 10 | Intent prior only pushes negative | Fixed neutral messages scoring positive |
| 11 | Ship TF-IDF, benchmark transformer offline | 257 KB vs 250 MB against a 1 GB target |

---

## Next

**Phase 5 — Knowledge base.** Chunking at 128–256 tokens per EDA finding 7c, with
page-level provenance and version metadata.
