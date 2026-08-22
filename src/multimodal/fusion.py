"""Multimodal fusion.

Stage 3: combine customer text, image observations and order context into one
piece of evidence that enters the existing understanding → routing → RAG
pipeline.

THE DESIGN POINT
----------------
The image is not analysed independently and the result is not shown to the
customer as a separate answer. It is *fused into the query* so that everything
downstream - intent classification, entity extraction, retrieval, generation -
sees a single enriched request.

Concretely, a customer writing

    "my payment isn't working"

with a screenshot showing PAY-402 becomes

    query text:      "my payment isn't working"
    enriched query:  "my payment isn't working PAY-402 payment gateway timeout"
    evidence block:  [IMAGE EVIDENCE] with evidence levels per field

The enriched query is what retrieval searches. That is the whole mechanism
behind the vision ablation: without the image the query has no specific term to
match, and BM25 has nothing to lock onto.

WHAT THE IMAGE IS NOT ALLOWED TO DO
-----------------------------------
Only a code marked VISIBLE enters the retrieval query. An INFERRED code is
passed to the language model inside the evidence block, clearly labelled, but
never used to steer retrieval - a misread code would pull up a confidently
wrong troubleshooting article, which is worse than no image at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.multimodal.vision import Evidence, ImageAnalysis, analyze_image


@dataclass
class CustomerContext:
    """Optional structured context the customer or session supplies."""

    order_id: str | None = None
    product: str | None = None
    region: str | None = None
    customer_id: str | None = None

    def to_block(self) -> str:
        bits = []
        if self.order_id:
            bits.append(f"Order: {self.order_id}")
        if self.product:
            bits.append(f"Product: {self.product}")
        if self.region:
            bits.append(f"Region: {self.region}")
        return "[CONTEXT] " + " | ".join(bits) if bits else ""


@dataclass
class MultimodalRequest:
    """A support request that may carry text, an image, and context."""

    text: str
    image_path: Path | str | bytes | None = None
    image_filename: str | None = None
    context: CustomerContext = field(default_factory=CustomerContext)


@dataclass
class FusedEvidence:
    """The unified request that enters the pipeline."""

    original_text: str
    enriched_query: str
    evidence_blocks: list[str] = field(default_factory=list)

    image: ImageAnalysis | None = None
    has_image: bool = False
    image_contributed: bool = False
    image_terms: list[str] = field(default_factory=list)

    context: CustomerContext = field(default_factory=CustomerContext)
    fusion_notes: list[str] = field(default_factory=list)

    @property
    def evidence_text(self) -> str:
        return "\n\n".join(b for b in self.evidence_blocks if b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "enriched_query": self.enriched_query,
            "has_image": self.has_image,
            "image_contributed": self.image_contributed,
            "image_terms": self.image_terms,
            "fusion_notes": self.fusion_notes,
            "image": self.image.to_dict() if self.image else None,
            "context": asdict(self.context),
        }

    def summary(self) -> str:
        if not self.has_image:
            return "text only"
        if not self.image_contributed:
            return f"image present but contributed nothing ({self.image.reason})"
        return f"image contributed: {', '.join(self.image_terms)}"


# Terms added alongside a recognised code, so retrieval matches documents that
# describe the fault without naming the code in the same sentence.
CODE_EXPANSIONS = {
    "PAY": ["payment", "checkout", "transaction"],
    "ERR-DP": ["displayport", "monitor", "display"],
    "ERR-HD": ["hdmi", "monitor", "display"],
    "BAT": ["battery", "charging"],
    "WIFI": ["wireless", "network"],
    "SYS": ["boot", "startup", "stop code"],
    "THRM": ["thermal", "overheating", "fan"],
    "DSP": ["display", "panel", "backlight"],
    "AUD": ["audio", "sound"],
    "KEY": ["keyboard"],
    "STO": ["storage", "disk"],
    "MEM": ["memory"],
    "CAM": ["camera"],
}


def _expansion_terms(code: str) -> list[str]:
    for prefix, terms in sorted(CODE_EXPANSIONS.items(), key=lambda x: -len(x[0])):
        if code.upper().startswith(prefix):
            return terms
    return []


def fuse(
    request: MultimodalRequest, vision_backend: str = "local_ocr"
) -> FusedEvidence:
    """Combine text, image and context into one enriched request."""
    fused = FusedEvidence(
        original_text=request.text,
        enriched_query=request.text,
        context=request.context,
    )

    ctx_block = request.context.to_block()
    if ctx_block:
        fused.evidence_blocks.append(ctx_block)

    if request.image_path is None:
        fused.fusion_notes.append("no image attached")
        return fused

    fused.has_image = True
    analysis = analyze_image(
        request.image_path, user_text=request.text,
        backend=vision_backend, filename=request.image_filename,
    )
    fused.image = analysis
    fused.evidence_blocks.append(analysis.to_evidence_block())

    if not analysis.is_useful:
        fused.fusion_notes.append(f"image not usable: {analysis.reason}")
        return fused

    terms: list[str] = []

    # Only a directly-read code steers retrieval. See module docstring.
    if analysis.has_reliable_code:
        terms.append(analysis.error_code)
        terms.extend(_expansion_terms(analysis.error_code))
        fused.fusion_notes.append(
            f"error code {analysis.error_code} read directly from the image "
            f"and added to the retrieval query"
        )
    elif analysis.error_code:
        fused.fusion_notes.append(
            f"error code {analysis.error_code} was INFERRED, not clearly read; "
            f"passed to the model as uncertain evidence but NOT used to steer "
            f"retrieval"
        )

    # An error message is safe to add regardless: it is descriptive prose, so a
    # partial misread degrades matching rather than redirecting it.
    if analysis.visible_error and analysis.evidence.get(
        "visible_error"
    ) == Evidence.VISIBLE:
        words = [
            w for w in analysis.visible_error.split()
            if len(w) > 3 and w.lower() not in {"could", "your", "this", "that",
                                                "with", "from", "have", "been"}
        ]
        terms.extend(words[:6])
        fused.fusion_notes.append("visible error message added to the query")

    if analysis.image_type and analysis.evidence.get(
        "image_type"
    ) == Evidence.VISIBLE and "unknown" not in analysis.image_type:
        head = analysis.image_type.split()[0]
        if len(head) > 3:
            terms.append(head)

    terms = list(dict.fromkeys(t for t in terms if t))
    if terms:
        fused.image_terms = terms
        fused.image_contributed = True
        fused.enriched_query = f"{request.text} {' '.join(terms)}".strip()
    else:
        fused.fusion_notes.append(
            "image analysed but produced no term safe to add to the query"
        )
    return fused


if __name__ == "__main__":
    from src.config.settings import settings

    d = settings.eval_dir / "screenshots"
    cases = [
        ("my payment isn't working", d / "V003_PAY_402.png"),
        ("my monitor keeps going black randomly", d / "V001_ERR_DP_0x004.png"),
        ("something is wrong, see photo", d / "edge_cases" / "blurry_severe.png"),
        ("here is a picture", d / "edge_cases" / "irrelevant_product.png"),
        ("my laptop is slow", None),
    ]
    for text, path in cases:
        f = fuse(MultimodalRequest(text=text, image_path=path))
        print(f"\n{'=' * 74}")
        print(f"TEXT      {text}")
        print(f"IMAGE     {Path(path).name if path else '-'}")
        print(f"FUSION    {f.summary()}")
        print(f"QUERY     {f.enriched_query}")
        for n in f.fusion_notes:
            print(f"  note: {n}")
