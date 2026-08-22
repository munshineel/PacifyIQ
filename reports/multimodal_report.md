# PacifyIQ — Multimodal Screenshot Analysis

**Phase 8.** Customer text + screenshot + context → one enriched request into the
existing pipeline. Built and measured **without an API key**.

```bash
python scripts/data_generation/gen_screenshots.py
python scripts/evaluate_vision.py
```

---

## 1. Headline result

| Metric | Text only | Text + vision | Δ |
|---|---|---|---|
| **Recall@5** | 0.720 | **1.000** | **+0.280** |
| **MRR** | 0.503 | **0.780** | **+0.277** |
| Max BM25 | 6.8 | 38.1 | **+31.3** |

**7 cases rescued, 0 broken.** Error-code extraction **100% (25/25)**. Zero
invented codes across 8 unreadable or irrelevant images.

The BM25 jump is the mechanism: without the image a query like *"laptop won't
charge properly"* has no term the corpus uniquely matches. With it, `BAT-042`
locks onto exactly one section.

---

## 2. This runs offline

Tesseract 5.3.4 is available in the environment, so the vision backend is **real
OCR, not a simulation**. That matters — the ablation above is a measured result,
not a placeholder waiting for a key.

| Backend | What | Requires |
|---|---|---|
| **`local_ocr`** | Tesseract + rule-based interpretation | nothing |
| `groq` | hosted vision model | API key |

OCR has one property a hosted vision model does not: **per-word confidence**.
That makes the visible/inferred/unknown distinction *measurable* rather than a
matter of the model's self-report — which is precisely the failure mode this
phase is designed around.

---

## 3. Architecture

```
Customer text  +  Screenshot  +  Context
                     │
        ┌────────────▼────────────┐
        │  1. VALIDATE            │  type, size, dimensions, corruption
        │  2. PREPROCESS          │  downscale, normalise mode
        │  3. ANALYSE             │  OCR → structured observations
        │     with evidence levels│  visible | inferred | unknown
        │  4. FUSE                │  into ONE enriched query
        └────────────┬────────────┘
                     ▼
        Understanding (intent, sentiment, entities)
                     ▼
        Routing → Retrieval → Abstention gate → Generation
```

### The design point: fusion, not parallel analysis

The image is **not** analysed independently and its result is not presented
separately. It is fused *into the query*, so everything downstream sees one
enriched request:

```
text:      "my payment isn't working"
enriched:  "my payment isn't working PAY-402 payment checkout transaction ..."
evidence:  [IMAGE EVIDENCE] error code PAY-402 (visible) ...
```

The enriched query drives retrieval; the evidence block, with its levels
attached, goes into the prompt. The customer's original wording is what the
trace and the prompt display — a test asserts that.

---

## 4. Visible / inferred / unknown

The requirement that shaped everything else:

> A vision model asked *"what is the error code?"* on a blurred screenshot will
> produce a plausible code. That is **worse than returning nothing** — the code
> flows into retrieval, pulls up a confidently wrong troubleshooting article,
> and the customer is told to reseat a cable for a fault they do not have.

Every field carries an evidence level, and the pipeline treats them differently:

| Level | Meaning | Effect on retrieval |
|---|---|---|
| **VISIBLE** | read directly and clearly | **enters the query** |
| **INFERRED** | deduced, or recovered from a misread | passed to the model, labelled; **never steers retrieval** |
| **UNKNOWN** | could not be determined | reported as absent |

Distribution across the 25 cases: **84% visible, 12% inferred, 4% unknown.**

The inferred cases are real. `ERR-DP-0x004` was OCR'd as `ERR-DP-@x004`. The
repair layer resolves it against the canonical code registry, but because it *is*
a repair it is marked INFERRED — so it reaches the language model as labelled
uncertain evidence and does **not** enter the retrieval query.

### Structured output

```json
{
  "image_type": "payment error dialog",
  "visible_error": "Payment could not be completed",
  "error_code": "PAY-402",
  "ui_context": "a retry action is offered; an order summary is shown",
  "relevant_observations": ["Value visible in image: Rs 64,900"],
  "confidence": 0.95,
  "evidence": {
    "image_type": "visible", "visible_error": "visible",
    "error_code": "visible", "ui_context": "visible"
  }
}
```

---

## 5. Validation

Strict about **format**, lenient about **content**. A blurry screenshot is a
valid upload the vision layer should report as unreadable; it is not a validation
failure. Conflating the two either rejects legitimate uploads or pushes corrupt
files into the model.

| Check | Rule |
|---|---|
| Format | PNG, JPEG, WEBP, GIF. BMP/TIFF rejected — rarely produced by real devices, disproportionately large |
| Size | 200 B – 10 MB |
| Dimensions | 50 px – 8000 px, ≤ 40 MP (decompression-bomb guard) |
| Integrity | `verify()` + full decode; corrupt files never raise |
| Quality | brightness, contrast, **Laplacian variance** — feed confidence, do not gate |

### A measurement bug worth recording

Blur detection first used PIL's `FIND_EDGES` standard deviation. It scored a
**mildly** blurred image at 15.2 and a **severely** blurred one at 14.8 — unable
to distinguish readable from unreadable, which is the only thing the metric
exists to do.

Replaced with a real Laplacian convolution:

| Image | FIND_EDGES | Laplacian variance |
|---|---|---|
| Clear | 38.5 | **1296.5** |
| Mild blur | 15.2 | **3.7** |
| Severe blur | 14.8 | **0.5** |

Three orders of magnitude of separation instead of none. Regression-tested.

---

## 6. Edge cases — all seven categories

| File | Kind | Validation | Code | Useful? | Behaviour |
|---|---|---|---|---|---|
| `valid_clear.png` | valid | ok | PAY-402 | ✅ | read correctly |
| `blurry_mild.png` | blurry | warning | — | ✗ | **reports unknown** |
| `blurry_severe.png` | blurry | warning | — | ✗ | **reports unknown** |
| `too_dark.png` | low quality | warning | PAY-402 | ✅ | **still read — text survived** |
| `tiny_downscaled.png` | low resolution | warning | — | ✗ | reports unknown |
| `irrelevant_product.png` | irrelevant | ok | — | ✗ | text readable, no error info |
| `blank_white.png` | no information | warning | — | ✗ | detected as blank before OCR |
| `noise.png` | no information | ok | — | ✗ | no legible text |
| `oversized_5200x3600.png` | oversized | warning | PAY-402 | ✅ | downscaled, then read |
| `unsupported_format.bmp` | unsupported | **rejected** | — | ✗ | rejected with a clear reason |
| `corrupt_truncated.png` | corrupt | **rejected** | — | ✗ | fails gracefully, no exception |

**Invented codes on the 8 images with no legible code: 0.**

### The distinction that matters

`too_dark.png` returns PAY-402 as VISIBLE, and that is **correct, not a
hallucination**. The image is underexposed but the text survived; OCR read it at
95% confidence and the code genuinely is there.

Being uselessly conservative is also a failure mode. Both directions are tested:
zero invented codes, and 2/2 poor-quality-but-legible images correctly read.

---

## 7. OCR character repair

Real OCR confuses glyphs. Observed on the evaluation set: `0x004` read as
`@x004`.

The repair layer normalises known confusion pairs (`@`↔`0`, `O`↔`0`, `l`↔`1`,
`S`↔`5`, `B`↔`8`) and resolves the result against the 25 canonical Pacify codes.

**It only accepts a repair that maps to exactly one known code.** An ambiguous
garble returns `None` rather than guessing — the same principle the evidence
levels enforce. And a repaired code is marked INFERRED, never VISIBLE.

A casing bug was caught here by test: `repair_code` uppercased before
translating, which turned a lowercase `l` into `L` and lost the `l`→`1`
confusion entirely.

---

## 8. Extraction by UI surface

Ten surfaces rendered, all at 100%:

| Surface | n | Accuracy |
|---|---|---|
| checkout page | 5 | 1.000 |
| diagnostics app | 6 | 1.000 |
| monitor OSD | 3 | 1.000 |
| system tray | 3 | 1.000 |
| stop screen | 2 | 1.000 |
| network dialog | 2 | 1.000 |
| bank redirect · device manager · sound dialog · visible symptom | 1 each | 1.000 |

`visible symptom` is the interesting one: a photograph of a dark monitor with
**no code displayed at all**. Correct behaviour is to report the symptom and
extract *no* code, which is what it does.

### A rendering bug, not a vision bug

`BAT-119` initially failed. The cause was the mock: the notification panel used
orange monospace on a dark grey background, and the code line rendered below the
panel bounds. OCR read every other word at 96% confidence and simply never saw
the code.

The fix belonged in the screenshot generator, **not in a lowered confidence
threshold**. Lowering the threshold would have "fixed" the metric while making
the system more willing to guess — the opposite of what this phase is for.

---

## 9. Cases the image rescued

| ID | Customer text | Code in image | BM25 text → vision |
|---|---|---|---|
| V003 | *"this keeps popping up when I try to pay"* | PAY-402 | 6.1 → **44.3** |
| V005 | *"card not working at checkout"* | PAY-309 | 10.7 → **45.1** |
| V007 | *"laptop won't charge properly"* | BAT-042 | 6.4 → **39.5** |
| V010 | *"computer crashed with this message"* | SYS-0x000000EF | 6.2 → **31.2** |
| V011 | *"laptop gets really hot and slows down"* | THRM-88 | 6.3 → **41.4** |
| V023 | *"charging stopped by itself"* | BAT-007 | 5.1 → **52.5** |
| V024 | *"cannot join my home network"* | WIFI-211 | 4.7 → **22.6** |

Every one sat below the **abstention threshold of BM25 7.0** on text alone. Six
of seven would have been correctly refused as unanswerable — the customer's
words genuinely did not contain enough to answer. The screenshot is what makes
them answerable.

That is the strongest possible statement of why multimodal input matters here,
and it connects directly to Phase 7's finding that BM25 is the abstention signal.

---

## 10. End-to-end behaviour

```
TEXT      my payment isn't working
IMAGE     V003_PAY_402.png
FUSION    image contributed: PAY-402, payment, checkout, transaction
DECISION  caveat
CITED     POL-PAY-001 p.2 S4 · POL-PAY-001 p.3 S10     ← the error-code table
TRACE     bm25=42.8 | cites=2 | grounded=True | 24ms
```

```
TEXT      something is wrong, see photo
IMAGE     blurry_severe.png
FUSION    image present but contributed nothing (no legible text)
DECISION  abstain
TRACE     bm25=6.9 | cites=0 | 0 tokens
```

An unusable image **does not** push an unanswerable question over the abstention
threshold. Tested explicitly, because an attached file creates an expectation
that something was understood.

---

## 11. Trace additions

```
has_image             True
image_contributed     True
image_error_code      PAY-402
image_evidence_level  visible
image_terms           [PAY-402, payment, checkout, transaction]
```

Every image decision is auditable from the dashboard, including *why* an inferred
code was withheld from retrieval.

---

## 12. Honest limitations

1. **The screenshots are synthetic**, rendered by
   `scripts/data_generation/gen_screenshots.py`. Real customer screenshots are
   photographed at angles, cropped badly, compressed by messaging apps, and
   often in languages other than English. **100% extraction on clean renders
   will not transfer.** Treat the ablation *delta* as the finding, not the
   absolute accuracy.
2. **OCR is not a vision model.** It reads text well and understands nothing.
   It cannot describe visible physical damage to a laptop, interpret an
   unfamiliar UI, or answer *"is this the right cable?"* The `groq` backend
   exists for that and **has not been run** — no key was available.
3. **The code registry is closed.** Repair only resolves against 25 known
   Pacify codes. A genuine but unlisted code would be marked INFERRED or
   dropped.
4. **Rule-based surface classification.** `image_type` comes from keyword
   fingerprints, which is brittle against UI redesigns and unseen surfaces.
5. **No image-embedded injection defence yet.** Text extracted from an image is
   treated as data, not instruction — but there is no explicit filter for a
   screenshot containing *"ignore your instructions."* Phase 11 work.

---

## 13. What your key would add

```powershell
python scripts\evaluate_vision.py   # after setting the backend to groq
```

The comparison this report cannot make: **does a hosted vision model beat OCR,
and does it stay honest about legibility?** The specific thing to watch is the
edge-case table — OCR reports UNKNOWN on all 8 unreadable images because word
confidence is measurable. A vision model has no such signal and must be
*instructed* to abstain, which is exactly the failure mode `VISION_PROMPT` and
the blur downgrade guardrail are written to prevent.

If the hosted model invents codes where OCR reported unknown, that is a finding
worth reporting.

---

## 14. Artifacts

| Path | Contents |
|---|---|
| `src/multimodal/validation.py` | validation, quality signals, preprocessing |
| `src/multimodal/vision.py` | OCR + Groq backends, evidence levels, code repair |
| `src/multimodal/fusion.py` | text + image + context → enriched request |
| `scripts/data_generation/gen_screenshots.py` | 25 cases + 11 edge cases |
| `scripts/evaluate_vision.py` | extraction, ablation, edge cases, end-to-end |
| `data/eval/screenshots/` | 25 PNGs + manifest |
| `data/eval/screenshots/edge_cases/` | 11 edge-case files |
| `reports/results/vision_ablation.csv` | per-case text vs vision |
| `reports/results/vision_extraction.csv` | per-case extraction + evidence |
| `reports/results/vision_edge_cases.csv` | all 11 edge cases |

**41 Phase 8 tests, all offline. 252 across the project.**

---

## Established

```
Customer Text + Screenshot + Context
        → Multimodal Understanding (visible | inferred | unknown)
        → Intent + Evidence
        → RAG
        → Grounded answer or escalation
```

**Recall@5 0.720 → 1.000. MRR +0.277. Zero invented codes.** Six of the seven
rescued cases would otherwise have been correctly abstained on — the screenshot
is what makes them answerable at all.
