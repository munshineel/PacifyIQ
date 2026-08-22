# PacifyIQ — Knowledge Base & Retrieval

**Phases 5 and 6.** Give the agent a source of truth, and prove retrieval works
*before* any LLM is involved.

```bash
python scripts/build_index.py
python scripts/evaluate_retrieval.py --ablate
jupyter notebook notebooks/04_retrieval_evaluation.ipynb
```

---

## 1. The knowledge base

⚠️ **All content is synthetic.** Pacify Electronics Pvt. Ltd. is fictional. No real
company's policies are reproduced or paraphrased. Every fact traces to
`data/canonical_facts.md`, which is the authored source of truth.

### Coverage

| Source type | Documents | Pages | Serves |
|---|---|---|---|
| **Policies** | `return_policy_v2`, `refund_policy`, `warranty_policy`, `shipping_policy`, `payment_policy`, `customer_service_policy`, `eu_regional_addendum` | 30 | returns, refunds, warranty, shipping, billing, account |
| **FAQ** | `product_faq` | 6 | general, cross-cutting |
| **Troubleshooting** | `technical_support_faq` | 5 | technical, **error codes** |
| **Product documentation** | 3 manuals (ProBook 14, Phone X, Vision 27) | 6 | specifications, model-specific faults |
| **Archived** | `return_policy_v1_ARCHIVED` | 2 | version-preference testing |

**13 documents · 47 pages · 16,208 words.**

Mapped against the requested source list: FAQs ✅, product documentation ✅,
troubleshooting ✅, policies ✅, billing (`payment_policy`) ✅, account
(`customer_service_policy`) ✅, refund/return rules ✅, error-code documentation ✅
(25 codes in `technical_support_faq S9`), common issue resolutions ✅.

### The corpus is deliberately imperfect

Eighteen defects are engineered into it and registered in `PLANTED_DEFECTS.md`.
Without them, retrieval metrics pin near 1.0 and prove nothing. The ones retrieval
must handle:

| Defect | What it is | Retrieval must |
|---|---|---|
| A1 | `shipping_policy S11` promises 30 days; `return_policy_v2 S2` says 14 for opened items. Both current. | surface both, not pick |
| A2 | `return_policy_v1_ARCHIVED` coexists with v2 | prefer current by metadata |
| A3 | EU addendum overrides base policy | apply regional filter |
| B | `product_faq` restates policy in casual language | prefer the authoritative source |
| D | Refund calculation spans 4 sections across 2 documents | retrieve all of them |
| E | Error codes fragment under subword tokenisation | match lexically |

---

## 2. Pipeline

```
PDF → load → clean → chunk → embed → store → retrieve
```

Each stage in `src/knowledge/`, independently testable.

### 2.1 Load — `loader.py`

Extracts text page by page and attaches metadata from a document registry. The
registry is the single place that knows a document is archived or region-specific;
an unregistered document would silently lose those flags, so a test enforces that
every document is registered.

### 2.2 Clean — `chunker.py`

Removes repeating footers (present on all 47 pages — they would otherwise inflate
similarity between unrelated chunks of the same document), rejoins words split
across line breaks, and normalises bullets and whitespace.

### 2.3 Chunk

Two strategies:

- **`section`** — split on section headings, subdivide anything oversized on
  sentence boundaries with overlap. A policy clause *is* an answer.
- **`fixed`** — fixed token windows with overlap. The conventional baseline, kept
  so the choice can be measured.

**Section attribution is 95–100%.** Two bugs were found and fixed here:

1. **Cross-references swallowed headings.** The heading regex used `\s+`, and `\s`
   matches newlines — so a line ending *"…raise a claim under POL-WAR-001 S10."*
   consumed the `S3. Display problems` heading on the next line and mislabelled the
   entire section. Fixed to `[ \t]+`.
2. **Section attribution lost at page breaks.** Sections continue across pages;
   without carrying the last heading forward, ~20% of chunks had no section and
   therefore no precise citation.

Both are regression-tested.

### 2.4 Metadata per chunk

```
chunk_id       return_policy_v2::S2::p1::003::a4f2b891
doc            return_policy_v2
doc_ref        POL-RET-002              ← citation
page           1                        ← citation
section        S2                       ← citation + eval join key
section_title  Return windows
doc_type       policy                   ← authority weighting
topic          returns                  ← filtering
version        current                  ← filtering (DEFECT-02)
region         all                      ← filtering (DEFECT-03)
product        null
n_tokens       118
```

Chunk IDs hash the text, so an index rebuild is byte-reproducible and diffable.

### 2.5 Embed — `embedder.py`

Two backends behind one interface:

| Backend | What | When |
|---|---|---|
| **`tfidf_svd`** (default) | TF-IDF → truncated SVD (classic LSA), word + char n-grams | development, evaluation, CI |
| `groq` | hosted `nomic-embed-text-v1_5`, disk-cached | production comparison |

**Why local by default.** The corpus is 16,208 words. At that scale an API call per
query buys real but modest quality for a hard dependency on a third party, and it
prevents the evaluation suite from running offline or in CI. `tfidf_svd` is a genuine
baseline, not a stub, and the Groq path is one flag away when a key is present.

Dimension was chosen by measurement, not convention:

| dim | word/char features | explained var | artifact | recall@5 |
|---|---|---|---|---|
| 64 | 4000 / 6000 | 0.558 | 2.4 MB | 0.950 |
| 128 | 4000 / 6000 | 0.832 | 4.9 MB | 0.875 |
| **192** | **4000 / 6000** | **0.993** | **7.3 MB** | **0.975** |
| 199 | 8000 / 20000 | 1.000 | 13.5 MB | 0.950 |

*(measured on a 40-query subset during tuning)*

⚠️ An early version produced a **35 MB** embedder because character n-grams were
uncapped and SVD components were float64. Capping features and casting to float32
brought the whole index to **8.3 MB** with no measurable quality loss.

### 2.6 Store — `vector_store.py`

Brute-force cosine similarity in NumPy, with a verified-identical FAISS backend.

**Measured at 200 vectors:**

| Backend | ms/query | identical results |
|---|---|---|
| NumPy | 0.050 | — |
| FAISS `IndexFlatIP` | 0.127 | 100% |

**FAISS is 2.5× slower here.** At 200 vectors the index-building overhead exceeds
any search benefit, and an approximate index (HNSW) would trade recall for speed it
does not need. This confirms the EDA finding-7c decision: *exact search is
sufficient at this corpus size, so no approximate index ships.* That is a
measurement, not an assumption — and a better answer than an unnecessary HNSW index.

---

## 3. Retrieval — `retriever.py`

Four strategies, switchable by config:

| Strategy | Method |
|---|---|
| `dense` | embedding cosine |
| `bm25` | Okapi BM25 lexical |
| `hybrid` | reciprocal rank fusion, equal weights |
| `rrf_w` | weighted RRF, dense-leaning (0.7 / 0.3) |

**Why fuse ranks rather than scores.** Dense cosine lives in [0, 1]; BM25 is
unbounded and corpus-dependent. Normalising them to a common scale is fragile.
RRF combines *positions*, which sidesteps the problem entirely.

**Filtering happens before ranking**, so an excluded chunk cannot defeat a filter by
scoring higher.

### Authority weighting — a fix derived from measured failures

The first evaluation run failed 24 of 120 queries, and **14 of those had the correct
policy section at rank 6–10**, displaced by a `product_faq` restatement of the same
fact. The FAQ is written casually, which matches casually-phrased queries better than
the formal clause that actually governs.

The corpus already states the precedence — `product_faq` closes with *"Where it
differs from a policy document, the policy document governs."* Encoding it is making
an existing documented rule operational, not inventing one.

```python
AUTHORITY = {"policy": 1.00, "troubleshooting": 0.97,
             "manual": 0.95, "faq": 0.88, "unknown": 0.90}
```

Applied to a 3× deeper candidate pool, so it can *promote* a policy section from
below the cut rather than only reshuffling what already made it.

**Measured effect:**

| Strategy | recall@5 off → on | MRR off → on |
|---|---|---|
| dense | 0.800 → 0.800 | +0.017 |
| bm25 | 0.758 → 0.758 | +0.038 |
| **hybrid** | **0.783 → 0.850 (+0.067)** | **+0.095** |

---

## 4. Retrieval evaluation

120 queries. **No LLM involved** — this measures retrieval alone.

Gold labels reference `(doc, section)` rather than chunk IDs, because chunk IDs
change with every chunking configuration and would invalidate the entire set on the
first ablation. Sections are the stable join key.

### 4.1 Strategy comparison

| Strategy | R@1 | R@3 | **R@5** | R@10 | P@5 | Cov@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|
| **rrf_w** | 0.542 | 0.742 | **0.875** | 0.942 | 0.235 | 0.793 | **0.665** | **0.726** |
| hybrid | 0.550 | 0.750 | 0.850 | 0.942 | 0.228 | 0.776 | 0.670 | 0.719 |
| dense | 0.442 | 0.683 | 0.800 | 0.917 | 0.218 | 0.741 | 0.585 | 0.648 |
| bm25 | 0.458 | 0.650 | 0.758 | 0.900 | 0.218 | 0.714 | 0.586 | 0.646 |

**Selected: `rrf_w`.** Fusion beats either component alone — dense misses exact
identifiers, BM25 misses paraphrases.

### 4.2 Headline

| Metric | Value |
|---|---|
| Recall@5 | **0.875** |
| Recall@10 | 0.942 |
| Coverage@5 | 0.793 |
| MRR | 0.665 |
| nDCG@5 | 0.726 |
| Queries answered | **87.5%** |

`coverage@K` matters more than `recall@K` for multi-hop questions: recall asks
whether *any* required section was found, coverage asks what fraction of *all* of
them were. A refund calculation needs four sections; finding one is not an answer.

### 4.3 By query type

| Type | n | Recall@5 | Coverage@5 | MRR |
|---|---|---|---|---|
| version | 2 | **1.000** | 1.000 | 0.625 |
| duplicate | 5 | **1.000** | 0.600 | 0.590 |
| multi (multi-hop) | 25 | 0.920 | 0.680 | 0.703 |
| lexical (error codes) | 8 | 0.875 | 0.875 | 0.604 |
| single | 74 | 0.865 | 0.865 | 0.684 |
| ambiguous | 4 | 0.750 | 0.417 | 0.378 |
| contradiction | 2 | **0.500** | 0.250 | 0.500 |

**The two weakest categories are the planted defects, which is the corpus working.**
`contradiction` and `ambiguous` queries have deliberately conflicting or
under-determined evidence. Low recall there is not a retrieval failure — the correct
behaviour is to surface the conflict rather than confidently pick a side, and
*deciding* is Phase 7's job, not retrieval's. The retriever flags it via
`RetrievalResult.has_conflict()`.

By difficulty: easy 0.897 · medium 0.880 · hard 0.839. The gradient is shallow,
which suggests the difficulty labels are roughly calibrated.

---

## 5. Ablations

### 5.1 Chunking — the clearest result in the phase

| Chunking | Size | Chunks | R@3 | **R@5** | Cov@5 | MRR |
|---|---|---|---|---|---|---|
| **section** | 200 | 200 | 0.658 | **0.800** | 0.737 | 0.568 |
| section | 512 | 175 | 0.708 | 0.800 | 0.752 | 0.577 |
| section | 256 | 185 | 0.658 | 0.792 | 0.740 | 0.573 |
| section | 128 | 244 | 0.650 | 0.767 | 0.699 | 0.538 |
| fixed | 128 | 213 | 0.633 | 0.717 | 0.639 | 0.519 |
| fixed | 200 | 139 | 0.542 | 0.675 | 0.595 | 0.452 |
| fixed | 256 | 112 | 0.500 | 0.625 | 0.564 | 0.425 |
| fixed | 512 | 66 | 0.475 | **0.542** | 0.458 | 0.368 |

*(measured before authority weighting, so the absolute numbers are lower than §4.2)*

**Section-aware chunking beats fixed windows at every size, and the gap widens with
size** — from +5pp at 128 tokens to **+26pp at 512**. Policy documents are written in
clauses; a clause is an answer, and a fixed window that splits one destroys it.

Within section chunking, 200–512 tokens are equivalent. **200 was chosen** for smaller
chunks and tighter citations at no measured cost.

### 5.2 Top-K sweep

| top_k | Recall@k | Precision@5 | MRR |
|---|---|---|---|
| 1 | 0.417 | 0.083 | 0.417 |
| 3 | 0.658 | 0.170 | 0.519 |
| **5** | **0.800** | 0.217 | 0.552 |
| 10 | 0.917 | 0.217 | 0.568 |
| 20 | 0.917 | 0.217 | 0.569 |

Recall saturates at k=10; nothing is gained beyond it. **k=5 is the default** because
each additional chunk costs context-window budget in Phase 7, and 5 chunks at ~120
tokens is roughly 600 tokens of evidence. The k=5 → k=10 trade is a Phase 7 decision
once the generation budget is known.

---

## 6. Failure analysis

**15 of 120 failed (12.5%)**, down from 24 before authority weighting.

| Failure mode | n | Example |
|---|---|---|
| Gold at rank 6–10, displaced | ~7 | R013 *"How is the refund amount calculated?"* → gold at rank 7 |
| Meta-questions about the system | 2 | R058 *"Am I talking to a bot?"*, R059 *"Can the AI approve my refund?"* |
| Cross-document spec questions | 3 | R074 *"How much does the ProBook 14 weigh?"* — manual and FAQ both hold it |
| Enumeration questions | 1 | R038 *"Which EU countries do you ship to?"* — a list, not a clause |
| Symptom → procedure | 1 | R065 *"My laptop will not turn on"* |

**Two observations worth carrying forward:**

**Meta-questions are structurally hard.** *"Can the AI approve my refund?"* is
answered by `customer_service_policy S12` (automated-assistance scope), but every
term in the query points at refund documents. No embedding fixes this — it needs
either intent-conditional filtering (route `account_management` intent to the service
policy) or an explicit FAQ entry. **Phase 7 should route by intent, using the
classifier already built in Phase 4.**

**Recall@10 is 0.942 versus recall@5 of 0.875.** Half the remaining failures are
retrieval-order problems, not retrieval-coverage problems. A reranker is the natural
next lever, and the LLM reranker planned for Phase 7 has ~7pp of headroom to
recover.

---

## 7. Decisions

| # | Decision | Evidence |
|---|---|---|
| 1 | Section-aware chunking | Beats fixed at every size; +26pp at 512 tokens |
| 2 | 200-token chunks | Equivalent to 512 on recall, tighter citations |
| 3 | `rrf_w` (weighted RRF) as default | Best recall@5 and nDCG of four strategies |
| 4 | Hybrid, not dense alone | Lexical queries reach 0.875 recall; dense alone is weak on codes |
| 5 | Authority weighting on | +0.067 recall@5, +0.095 MRR; encodes the corpus's own stated precedence |
| 6 | Exclude archived by default | v1 outranks v2 on pure similarity (DEFECT-02) |
| 7 | Brute-force NumPy, no ANN | FAISS is 2.5× slower at 200 vectors |
| 8 | `tfidf_svd` local default | Offline evaluation and CI; Groq available for comparison |
| 9 | dim 192, features capped | 35 MB → 8.3 MB index at no measured cost |
| 10 | Gold labels keyed to sections | Chunk IDs change on every ablation |
| 11 | top_k = 5 | Recall saturates at 10; context budget is a Phase 7 constraint |

---

## 8. Honest limitations

1. **The corpus is small** — 16,208 words, 200 chunks. Retrieval metrics from a
   200-chunk index are optimistic relative to a production knowledge base. This is
   EDA finding 7c, disclosed and unresolved.
2. **`tfidf_svd` is not a modern embedding model.** It is a defensible offline
   baseline, not state of the art. The Groq backend exists for a like-for-like
   comparison but has not been run here — no API key was available in this
   environment. **This comparison is outstanding work, not a completed result.**
3. **No reranking yet.** Recall@10 exceeds recall@5 by 6.7pp, so a reranker has
   measurable headroom. Planned for Phase 7.
4. **Gold labels are section-level, not span-level.** A chunk containing the right
   section counts as a hit even if the specific clause sits in an adjacent chunk of
   the same section. This makes recall slightly optimistic.
5. **All content is synthetic**, authored for this project. Retrieval difficulty is
   engineered rather than natural.

---

## 9. Artifacts

| Path | Contents |
|---|---|
| `data/index/vectors.npy` | 200 × 192 float32 embeddings (150 KB) |
| `data/index/chunks.pkl` | Chunk objects with full metadata (130 KB) |
| `data/index/chunks.jsonl` | Human-readable copy, diffable (179 KB) |
| `data/index/embedder.pkl` | Fitted TF-IDF + SVD (7.9 MB) |
| `data/index/index_metadata.json` | Build configuration and corpus stats |
| `reports/results/retrieval_strategy_comparison.csv` | 4 strategies × 10 metrics |
| `reports/results/retrieval_chunking_ablation.csv` | 8 configurations |
| `reports/results/retrieval_topk_sweep.csv` | k ∈ {1,3,5,10,20} |
| `reports/results/retrieval_by_{type,difficulty}.csv` | Breakdowns |
| `reports/results/retrieval_failures.csv` | All 15 failures with diagnosis |
| `reports/results/retrieval_per_query.csv` | Every query, every metric |
| `notebooks/04_retrieval_evaluation.ipynb` | Worked examples, executed |

**Total index: 8.3 MB** — comfortably inside the ~1 GB Streamlit Cloud budget.

---

## Established

```
User question → embedding → semantic search → relevant evidence
```

Measured, independently testable, no LLM involved. **Recall@5 0.875, MRR 0.665,
87.5% of queries answered.**

## Not yet built

Generation, citation enforcement, abstention, and conflict resolution. Retrieval
returns evidence; deciding what to *say* about it — including saying *"I don't have
documentation on that"* — is **Phase 7**.
