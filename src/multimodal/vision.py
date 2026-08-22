"""Vision analysis.

Stage 2 of the multimodal workflow. Extracts structured observations from a
screenshot, with one property treated as non-negotiable:

    THE MODEL MUST DISTINGUISH VISIBLE FROM INFERRED FROM UNKNOWN.

A vision model asked "what is the error code?" on a blurred screenshot will
produce a plausible code. That is far worse than returning nothing: the code
then flows into retrieval, pulls up a confidently wrong troubleshooting
article, and the customer is told to reseat a cable for a fault they do not
have. Every field therefore carries an `Evidence` level, and the pipeline
treats INFERRED and UNKNOWN differently from VISIBLE.

Two backends behind one interface:

    local   Tesseract OCR + rule-based interpretation. Real, offline, and
            honest about legibility because OCR confidence is measurable.
    groq    hosted vision model.

The local backend is not a placeholder. OCR genuinely reads the error codes,
which means the text-only vs text+vision ablation produces real numbers with
no API key.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from src.config.settings import settings
from src.multimodal.validation import (ImageValidation, preprocess, to_base64,
                                       validate_image)


class Evidence(str, Enum):
    """How the system came to believe something about the image."""

    VISIBLE = "visible"      # read directly off the image
    INFERRED = "inferred"    # deduced from layout, colour, partial text
    UNKNOWN = "unknown"      # could not be determined - say so


@dataclass
class Observation:
    """One thing noticed about the image, with its evidence level."""

    field: str
    value: str | None
    evidence: Evidence
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.value
        return d


@dataclass
class ImageAnalysis:
    """Structured output of the vision layer."""

    image_type: str = "unknown"
    visible_error: str | None = None
    error_code: str | None = None
    ui_context: str | None = None
    relevant_observations: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # provenance for every claim above
    evidence: dict[str, Evidence] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)

    extracted_text: str = ""
    ocr_confidence: float = 0.0
    is_useful: bool = False
    reason: str = ""

    backend: str = ""
    latency_ms: float = 0.0
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def has_reliable_code(self) -> bool:
        """Only a directly-read code is safe to feed into retrieval."""
        return (
            self.error_code is not None
            and self.evidence.get("error_code") == Evidence.VISIBLE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_type": self.image_type,
            "visible_error": self.visible_error,
            "error_code": self.error_code,
            "ui_context": self.ui_context,
            "relevant_observations": self.relevant_observations,
            "confidence": round(self.confidence, 3),
            "evidence": {k: v.value for k, v in self.evidence.items()},
            "is_useful": self.is_useful,
            "reason": self.reason,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
        }

    def to_evidence_block(self) -> str:
        """Render for inclusion in the RAG prompt.

        Evidence levels are stated explicitly so the language model cannot
        quietly promote an inference to a fact.
        """
        if not self.is_useful:
            return f"[IMAGE] Attached image provided no usable information ({self.reason})."

        lines = ["[IMAGE EVIDENCE]"]
        lines.append(f"Image type: {self.image_type} "
                     f"({self.evidence.get('image_type', Evidence.UNKNOWN).value})")
        if self.error_code:
            lines.append(f"Error code: {self.error_code} "
                         f"({self.evidence.get('error_code', Evidence.UNKNOWN).value})")
        else:
            lines.append("Error code: not determinable from the image")
        if self.visible_error:
            lines.append(f"Error message: {self.visible_error} "
                         f"({self.evidence.get('visible_error', Evidence.UNKNOWN).value})")
        if self.ui_context:
            lines.append(f"UI context: {self.ui_context} "
                         f"({self.evidence.get('ui_context', Evidence.UNKNOWN).value})")
        for o in self.relevant_observations:
            lines.append(f"- {o}")
        lines.append(
            "Treat items marked 'inferred' as uncertain and items marked "
            "'unknown' as absent. Do not state an inferred value as fact."
        )
        return "\n".join(lines)


# =====================================================================
# Interpretation rules
# =====================================================================

# Deliberately permissive on the trailing segment: OCR confuses characters that
# look alike, and an error code is exactly the kind of short alphanumeric string
# where that happens. Observed on the evaluation set: "0x004" read as "@x004".
RE_CODE = re.compile(
    r"\b((?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)"
    r"(?:[-_][A-Za-z0-9@|!ODSBl]+)+)\b"
)

# Canonical codes, from data/canonical_facts.md S9. Used to repair OCR
# character confusion: a garbled read that resolves to exactly one known code
# is almost certainly that code.
KNOWN_CODES = {
    "PAY-402", "PAY-511", "PAY-207", "PAY-309", "PAY-118", "PAY-604",
    "ERR-DP-0X004", "ERR-DP-0X011", "ERR-HD-0X002",
    "BAT-119", "BAT-042", "BAT-007",
    "WIFI-503", "WIFI-211",
    "SYS-0X0000007B", "SYS-0X000000EF",
    "THRM-88", "THRM-12", "DSP-014", "DSP-051",
    "AUD-330", "KEY-018", "STO-440", "MEM-221", "CAM-090",
}

# Glyph pairs Tesseract routinely confuses in short alphanumeric strings.
OCR_CONFUSIONS = str.maketrans({
    "@": "0", "O": "0", "o": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "G": "6",
})


def repair_code(raw: str) -> tuple[str | None, bool]:
    """Resolve an OCR-garbled code against the known registry.

    Returns (code, was_repaired). Only accepts a repair that maps to exactly
    one known code - an ambiguous repair returns None rather than guessing,
    which is the same principle the evidence levels enforce elsewhere.
    """
    cleaned = raw.replace("_", "-").strip(":,. ")
    candidate = cleaned.upper()
    if candidate in KNOWN_CODES:
        return candidate, False

    # Repair on the ORIGINAL casing: uppercasing first would turn a lowercase
    # "l" into "L" and lose the l->1 confusion entirely.
    parts = cleaned.split("-")
    repaired_parts = [parts[0].upper()] + [
        p.translate(OCR_CONFUSIONS).upper() for p in parts[1:]
    ]
    repaired = "-".join(repaired_parts)
    if repaired in KNOWN_CODES:
        return repaired, True

    # last resort: match on prefix plus digit-only comparison
    def digits(x: str) -> str:
        return re.sub(r"[^0-9]", "", x.translate(OCR_CONFUSIONS))


    matches = [
        k for k in KNOWN_CODES
        if k.split("-")[0] == parts[0] and digits(k) == digits(candidate)
    ]
    if len(matches) == 1:
        return matches[0], True
    return None, False

# Surface fingerprints: (label, required terms, any-of terms)
SURFACE_RULES = [
    ("payment checkout screen", {"checkout"}, {"payment", "order", "total", "card"}),
    ("payment error dialog", set(), {"payment", "transaction", "declined", "gateway"}),
    ("monitor on-screen display", set(), {"displayport", "refresh", "hdmi", "vision"}),
    ("system stop/boot error screen", set(), {"stop code", "restart", "boot device",
                                              "critical_process"}),
    ("diagnostics application", set(), {"diagnostics", "issue detected", "serial",
                                        "threshold"}),
    ("system notification", set(), {"hardware alert", "notification", "battery health",
                                    "charger"}),
    ("network settings dialog", set(), {"network", "wireless", "access point", "wi-fi"}),
    ("sound settings dialog", set(), {"sound", "audio", "driver conflict"}),
    ("device manager", set(), {"device manager", "camera module", "not detected"}),
    ("product photograph", set(), {"probook", "pacify probook"}),
]

ERROR_PHRASES = [
    "could not be completed", "did not respond", "failed", "not supported",
    "not recognised", "not recognized", "critical", "declined", "timeout",
    "out of range", "exceeded", "not responding", "error", "problem",
    "insufficient", "cannot complete", "issue detected", "handshake",
]


def _classify_surface(text: str) -> tuple[str, Evidence]:
    low = text.lower()
    for label, required, anyof in SURFACE_RULES:
        if required and not all(t in low for t in required):
            continue
        if anyof and any(t in low for t in anyof):
            return label, Evidence.VISIBLE
    if len(low.strip()) < 12:
        return "unknown", Evidence.UNKNOWN
    return "unidentified screen or document", Evidence.INFERRED


def _find_error_message(text: str) -> tuple[str | None, Evidence]:
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 12:
            continue
        low = s.lower()
        if any(p in low for p in ERROR_PHRASES):
            if RE_CODE.fullmatch(s.replace("Error code:", "").strip()):
                continue
            return s[:140], Evidence.VISIBLE
    return None, Evidence.UNKNOWN


# =====================================================================
# Backends
# =====================================================================

class VisionBackend(ABC):
    name: str

    @abstractmethod
    def analyze(self, img: Image.Image, validation: ImageValidation,
                user_text: str = "") -> ImageAnalysis:
        ...


class LocalOCRVision(VisionBackend):
    """Tesseract OCR plus rule-based interpretation.

    OCR reports per-word confidence, which makes the visible/inferred/unknown
    distinction measurable rather than a matter of the model's self-report.
    Where OCR confidence is low the field is marked UNKNOWN, which is exactly
    the behaviour a hallucinating vision model fails to produce.
    """

    name = "local_ocr"

    # Below this mean word confidence the reading is not trusted.
    MIN_WORD_CONF = 55.0
    MIN_CODE_CONF = 65.0

    def analyze(self, img: Image.Image, validation: ImageValidation,
                user_text: str = "") -> ImageAnalysis:
        import pytesseract

        t0 = time.perf_counter()
        a = ImageAnalysis(backend=self.name, validation=validation.to_dict())

        # Quality gate: a severely blurred image should not be OCR'd and
        # reported as if it were read.
        # Every exit path must populate all evidence keys, so downstream code
        # can rely on the contract instead of guarding for missing fields.
        a.evidence = {
            f: Evidence.UNKNOWN
            for f in ("image_type", "visible_error", "error_code", "ui_context")
        }

        if validation.is_likely_blank:
            a.reason = "image appears blank or uniform"
            a.image_type = "no usable content"
            a.evidence["image_type"] = Evidence.VISIBLE
            a.confidence = 0.0
            a.latency_ms = (time.perf_counter() - t0) * 1000
            return a

        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
        words, confs = [], []
        for txt, conf in zip(data["text"], data["conf"]):
            txt = (txt or "").strip()
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if txt and c >= 0:
                words.append((txt, c))
                confs.append(c)

        a.extracted_text = pytesseract.image_to_string(img).strip()
        a.ocr_confidence = round(sum(confs) / len(confs) / 100, 3) if confs else 0.0

        if not words or a.ocr_confidence < 0.30:
            a.reason = (
                "no legible text could be extracted"
                if not words else
                f"text extraction confidence too low ({a.ocr_confidence:.2f})"
            )
            a.image_type, a.evidence["image_type"] = (
                ("image with no readable text", Evidence.INFERRED)
                if words else ("unreadable image", Evidence.UNKNOWN)
            )
            a.evidence["error_code"] = Evidence.UNKNOWN
            a.confidence = 0.0
            a.latency_ms = (time.perf_counter() - t0) * 1000
            return a

        text = a.extracted_text

        # ---- image type -------------------------------------------
        a.image_type, a.evidence["image_type"] = _classify_surface(text)

        # ---- error code -------------------------------------------
        # Confidence is taken from the OCR words that make up the match, not
        # from the page average - a clear code on a noisy page is still clear.
        best_code, best_conf, was_repaired = None, 0.0, False
        for m in RE_CODE.finditer(text):
            raw = m.group(1)
            code, repaired = repair_code(raw)
            if code is None:
                continue
            token_confs = [
                c for w, c in words
                if code.split("-")[0].lower() in w.lower()
                or w.strip(":,.").upper().replace("_", "-") == code
                or raw.upper() in w.upper()
            ]
            conf = max(token_confs) if token_confs else a.ocr_confidence * 100
            if conf > best_conf:
                best_code, best_conf, was_repaired = code, conf, repaired

        if best_code and was_repaired:
            a.relevant_observations.append(
                f"Error code {best_code} recovered from an OCR misread; it "
                f"resolves to exactly one known Pacify code."
            )

        if best_code and best_conf >= self.MIN_CODE_CONF:
            a.error_code = best_code
            a.evidence["error_code"] = Evidence.VISIBLE
        elif best_code:
            a.error_code = best_code
            a.evidence["error_code"] = Evidence.INFERRED
            a.relevant_observations.append(
                f"A code resembling {best_code} is present but was read with low "
                f"confidence ({best_conf:.0f}%); it may be misread."
            )
        else:
            a.evidence["error_code"] = Evidence.UNKNOWN

        # ---- error message ----------------------------------------
        a.visible_error, a.evidence["visible_error"] = _find_error_message(text)

        # ---- UI context -------------------------------------------
        ctx = []
        low = text.lower()
        for term, label in [
            ("retry", "a retry action is offered"),
            ("cancel", "a cancel action is offered"),
            ("order summary", "an order summary is shown"),
            ("displayport", "input is DisplayPort"),
            ("hdmi", "input is HDMI"),
            ("serial", "a device serial number is shown"),
            ("reference", "a transaction reference is shown"),
            ("restart", "the device is restarting"),
        ]:
            if term in low:
                ctx.append(label)
        if ctx:
            a.ui_context = "; ".join(ctx[:4])
            a.evidence["ui_context"] = Evidence.VISIBLE
        else:
            a.evidence["ui_context"] = Evidence.UNKNOWN

        # ---- observations -----------------------------------------
        for m in re.finditer(r"\b(Rs\s?[\d,]+|\d{3,4}\s?x\s?\d{3,4}|\d+\s?Hz)\b", text):
            a.relevant_observations.append(f"Value visible in image: {m.group(1)}")
        if validation.warnings:
            a.relevant_observations.extend(
                f"Image quality: {w}" for w in validation.warnings[:2]
            )

        # ---- usefulness and confidence ----------------------------
        a.is_useful = bool(a.error_code or a.visible_error or a.ui_context)
        if not a.is_useful:
            a.reason = "text was readable but contained no error information"

        score = 0.0
        if a.evidence.get("error_code") == Evidence.VISIBLE:
            score += 0.55
        elif a.evidence.get("error_code") == Evidence.INFERRED:
            score += 0.20
        if a.visible_error:
            score += 0.20
        if a.evidence.get("image_type") == Evidence.VISIBLE:
            score += 0.15
        if a.ui_context:
            score += 0.10
        a.confidence = round(min(score * (0.5 + 0.5 * a.ocr_confidence), 0.95), 3)

        a.latency_ms = (time.perf_counter() - t0) * 1000
        return a


VISION_PROMPT = """You are analysing a screenshot a customer attached to a support request.

Return JSON only, with exactly these keys:
{
  "image_type": "<what kind of screen or photo this is>",
  "visible_error": "<the error message text, or null>",
  "error_code": "<the error code, or null>",
  "ui_context": "<what the interface shows, or null>",
  "relevant_observations": ["<other useful details>"],
  "confidence": <0.0-1.0>,
  "evidence": {
     "image_type": "visible|inferred|unknown",
     "visible_error": "visible|inferred|unknown",
     "error_code": "visible|inferred|unknown",
     "ui_context": "visible|inferred|unknown"
  }
}

CRITICAL RULES
1. Mark a field "visible" ONLY if you can read it directly and clearly.
2. Mark it "inferred" if you are deducing it from layout, colour or partial text.
3. Mark it "unknown" and set the value to null if you cannot determine it.
4. If the image is blurred, dark or low-resolution, DO NOT GUESS at text.
   Reporting "unknown" is correct and useful. Inventing a plausible error code
   is a serious failure - it sends the customer to the wrong fix.
5. Never invent an error code that you cannot actually see."""


class GroqVision(VisionBackend):
    """Hosted vision model."""

    name = "groq_vision"

    def __init__(self, model: str | None = None):
        self.model = model or settings.vision_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.groq_api_key:
                raise RuntimeError(
                    "PACIFYIQ_GROQ_API_KEY is not set. Use the local_ocr backend."
                )
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def analyze(self, img: Image.Image, validation: ImageValidation,
                user_text: str = "") -> ImageAnalysis:
        t0 = time.perf_counter()
        a = ImageAnalysis(backend=self.name, validation=validation.to_dict())

        b64 = to_base64(img, fmt="JPEG")
        user = VISION_PROMPT
        if user_text:
            user += f"\n\nThe customer wrote: {user_text!r}"

        r = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=0.0,
            max_tokens=700,
        )
        raw = r.choices[0].message.content or "{}"

        from src.llm.structured import repair_json

        data, _ = repair_json(raw)
        data = data or {}

        a.image_type = data.get("image_type") or "unknown"
        a.visible_error = data.get("visible_error")
        a.error_code = data.get("error_code")
        a.ui_context = data.get("ui_context")
        a.relevant_observations = list(data.get("relevant_observations") or [])
        try:
            a.confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            a.confidence = 0.0

        ev = data.get("evidence") or {}
        for f in ("image_type", "visible_error", "error_code", "ui_context"):
            raw_ev = str(ev.get(f, "unknown")).lower()
            a.evidence[f] = (
                Evidence(raw_ev) if raw_ev in {e.value for e in Evidence}
                else Evidence.UNKNOWN
            )

        # Guardrail against the failure this whole design exists to prevent:
        # a model claiming to have read a code off an image the validator
        # already measured as unreadable.
        if validation.is_likely_blurry and a.evidence.get("error_code") == Evidence.VISIBLE:
            a.evidence["error_code"] = Evidence.INFERRED
            a.relevant_observations.append(
                "Image measured as blurred; a code reported as clearly visible has "
                "been downgraded to inferred."
            )

        a.is_useful = bool(a.error_code or a.visible_error or a.ui_context)
        if not a.is_useful:
            a.reason = "model reported no usable information"
        a.latency_ms = (time.perf_counter() - t0) * 1000
        return a


BACKENDS = {"local_ocr": LocalOCRVision, "groq": GroqVision}


def get_vision(backend: str = "local_ocr", **kw) -> VisionBackend:
    if backend not in BACKENDS:
        raise ValueError(f"unknown vision backend {backend!r}, expected {list(BACKENDS)}")
    return BACKENDS[backend](**kw)


def analyze_image(
    source: Path | str | bytes, user_text: str = "",
    backend: str = "local_ocr", filename: str | None = None,
) -> ImageAnalysis:
    """Full stage 1 + 2: validate, preprocess, analyse."""
    validation, img = validate_image(source, filename)
    if not validation.ok or img is None:
        a = ImageAnalysis(backend=backend, validation=validation.to_dict())
        a.reason = validation.reason
        a.image_type = "rejected"
        a.evidence = {k: Evidence.UNKNOWN for k in
                      ("image_type", "visible_error", "error_code", "ui_context")}
        return a

    img = preprocess(img, validation)
    return get_vision(backend).analyze(img, validation, user_text)


if __name__ == "__main__":
    d = settings.eval_dir / "screenshots"
    for name in ["V003_PAY_402.png", "V001_ERR_DP_0x004.png", "V009_SYS_0x0000007B.png"]:
        a = analyze_image(d / name, "my payment isn't working")
        print(f"\n{name}")
        print(json.dumps(a.to_dict(), indent=2))
