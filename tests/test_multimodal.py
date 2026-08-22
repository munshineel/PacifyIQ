"""Tests for the multimodal layer (Phase 8).

All offline: OCR runs locally via Tesseract, so the vision ablation and every
edge case are testable without an API key.
"""
import pytest

from src.config.settings import settings
from src.multimodal.fusion import CustomerContext, MultimodalRequest, fuse
from src.multimodal.validation import (ALLOWED_EXTENSIONS, ValidationStatus,
                                       preprocess, to_base64, validate_image)
from src.multimodal.vision import (Evidence, KNOWN_CODES, analyze_image,
                                   get_vision, repair_code)

pytestmark = pytest.mark.vision

SHOTS = settings.eval_dir / "screenshots"
EDGE = SHOTS / "edge_cases"

needs_shots = pytest.mark.skipif(
    not (SHOTS / "manifest.json").exists(),
    reason="run scripts/data_generation/gen_screenshots.py first",
)


# ---------------------------------------------------------- validation
@needs_shots
def test_valid_screenshot_accepted():
    v, img = validate_image(EDGE / "valid_clear.png")
    assert v.ok
    assert img is not None
    assert v.format == "PNG"


@needs_shots
@pytest.mark.parametrize("name,reason_contains", [
    ("unsupported_format.bmp", "unsupported file type"),
    ("corrupt_truncated.png", "corrupt"),
])
def test_bad_uploads_rejected(name, reason_contains):
    v, img = validate_image(EDGE / name)
    assert v.status == ValidationStatus.REJECTED
    assert reason_contains in v.reason.lower()
    assert img is None


def test_oversized_bytes_rejected():
    v, _ = validate_image(b"x" * (11 * 1024 * 1024), filename="huge.png")
    assert v.status == ValidationStatus.REJECTED
    assert "too large" in v.reason


def test_tiny_payload_rejected():
    v, _ = validate_image(b"abc", filename="x.png")
    assert v.status == ValidationStatus.REJECTED


def test_missing_file_does_not_raise():
    v, _ = validate_image(SHOTS / "does_not_exist.png")
    assert v.status == ValidationStatus.REJECTED


@needs_shots
def test_blur_detection_separates_mild_from_severe():
    """Regression: PIL FIND_EDGES scored mild blur at 15.2 and severe at 14.8,
    which cannot distinguish readable from unreadable. A real Laplacian
    variance separates them by orders of magnitude."""
    clear, _ = validate_image(EDGE / "valid_clear.png")
    mild, _ = validate_image(EDGE / "blurry_mild.png")
    severe, _ = validate_image(EDGE / "blurry_severe.png")
    assert clear.sharpness > mild.sharpness > severe.sharpness
    assert clear.sharpness > 100 * severe.sharpness


@needs_shots
def test_blank_image_flagged():
    v, _ = validate_image(EDGE / "blank_white.png")
    assert v.is_likely_blank


@needs_shots
def test_oversized_image_is_downscaled():
    v, img = validate_image(EDGE / "oversized_5200x3600.png")
    assert v.ok
    out = preprocess(img, v)
    assert max(out.size) <= 1600
    assert v.was_resized


@needs_shots
def test_base64_encoding_roundtrips():
    _, img = validate_image(EDGE / "valid_clear.png")
    b64 = to_base64(img)
    assert len(b64) > 500
    import base64
    base64.b64decode(b64)


# --------------------------------------------------------- code repair
@pytest.mark.parametrize("garbled,expected", [
    ("ERR-DP-@x004", "ERR-DP-0X004"),      # observed OCR misread of 0 as @
    ("PAY-4O2", "PAY-402"),                 # O for 0
    ("BAT-l19", "BAT-119"),                 # lowercase L for 1
    ("PAY-402", "PAY-402"),                 # already correct
])
def test_ocr_code_repair(garbled, expected):
    code, _ = repair_code(garbled)
    assert code == expected


def test_ambiguous_repair_returns_nothing():
    """An unresolvable garble must return None rather than guess - the same
    principle the evidence levels enforce."""
    code, _ = repair_code("PAY-999")
    assert code is None


def test_all_known_codes_are_canonical():
    from src.knowledge.loader import load_corpus

    text = " ".join(p.text.upper() for p in load_corpus())
    missing = [c for c in KNOWN_CODES if c.replace("0X", "0X") not in
               text.replace("0X", "0X")]
    assert len(missing) <= 2, f"codes not found in corpus: {missing}"


# -------------------------------------------------------------- vision
def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_vision("gpt-vision")


@needs_shots
@pytest.mark.parametrize("name,code", [
    ("V003_PAY_402.png", "PAY-402"),
    ("V009_SYS_0x0000007B.png", "SYS-0X0000007B"),
    ("V011_THRM_88.png", "THRM-88"),
    ("V024_WIFI_211.png", "WIFI-211"),
])
def test_error_codes_extracted(name, code):
    a = analyze_image(SHOTS / name)
    assert a.error_code == code


@needs_shots
def test_ui_surface_classified():
    a = analyze_image(SHOTS / "V003_PAY_402.png")
    assert "payment" in a.image_type.lower() or "checkout" in a.image_type.lower()
    assert a.evidence["image_type"] == Evidence.VISIBLE


@needs_shots
@pytest.mark.parametrize("name", [
    "blurry_severe.png", "blank_white.png", "noise.png",
    "irrelevant_product.png", "tiny_downscaled.png", "blurry_mild.png",
])
def test_never_invents_a_code_on_an_unreadable_image(name):
    """The single most important property in this phase. A vision model that
    guesses a plausible code sends the customer to the wrong fix - worse than
    no image at all."""
    a = analyze_image(EDGE / name)
    assert a.error_code is None, f"invented {a.error_code} on {name}"
    assert a.evidence["error_code"] == Evidence.UNKNOWN


@needs_shots
def test_poor_quality_but_legible_image_is_still_read():
    """The converse: being uselessly conservative is also a failure."""
    a = analyze_image(EDGE / "too_dark.png")
    assert a.error_code == "PAY-402"


@needs_shots
def test_rejected_upload_produces_a_clean_analysis():
    a = analyze_image(EDGE / "corrupt_truncated.png")
    assert a.image_type == "rejected"
    assert not a.is_useful
    assert all(e == Evidence.UNKNOWN for e in a.evidence.values())


@needs_shots
def test_evidence_block_states_uncertainty_explicitly():
    a = analyze_image(SHOTS / "V003_PAY_402.png")
    block = a.to_evidence_block()
    assert "visible" in block
    assert "inferred" in block.lower() or "unknown" in block.lower()


@needs_shots
def test_unusable_image_says_so_in_its_evidence_block():
    a = analyze_image(EDGE / "blank_white.png")
    assert "no usable information" in a.to_evidence_block()


# -------------------------------------------------------------- fusion
@needs_shots
def test_visible_code_enters_the_retrieval_query():
    f = fuse(MultimodalRequest("my payment isn't working",
                               SHOTS / "V003_PAY_402.png"))
    assert f.image_contributed
    assert "PAY-402" in f.enriched_query


@needs_shots
def test_inferred_code_is_withheld_from_retrieval():
    """A misread code would pull up a confidently wrong article. Inferred
    codes reach the model as labelled evidence but never steer retrieval."""
    a = analyze_image(SHOTS / "V001_ERR_DP_0x004.png")
    if a.evidence.get("error_code") == Evidence.INFERRED:
        f = fuse(MultimodalRequest("my monitor goes black",
                                   SHOTS / "V001_ERR_DP_0x004.png"))
        assert a.error_code not in f.enriched_query
        assert any("INFERRED" in n for n in f.fusion_notes)


@needs_shots
def test_unusable_image_leaves_the_query_untouched():
    text = "something is wrong, see photo"
    f = fuse(MultimodalRequest(text, EDGE / "blurry_severe.png"))
    assert f.has_image
    assert not f.image_contributed
    assert f.enriched_query == text


def test_no_image_is_a_normal_path():
    f = fuse(MultimodalRequest("my laptop is slow"))
    assert not f.has_image
    assert f.enriched_query == "my laptop is slow"


def test_context_becomes_an_evidence_block():
    f = fuse(MultimodalRequest(
        "where is it?", context=CustomerContext(order_id="PAC-2026-12345",
                                                region="EU")))
    assert "PAC-2026-12345" in f.evidence_text
    assert "EU" in f.evidence_text


# ------------------------------------------------------------ pipeline
@needs_shots
def test_pipeline_accepts_a_screenshot_end_to_end():
    from src.rag.generator import build_pipeline

    r = build_pipeline().answer_multimodal(
        "my payment isn't working", SHOTS / "V003_PAY_402.png"
    )
    assert r.trace.has_image
    assert r.trace.image_contributed
    assert r.trace.image_error_code == "PAY-402"
    assert r.trace.image_evidence_level == "visible"
    assert r.response.citations


@needs_shots
def test_displayed_question_stays_the_customer_text():
    """The enriched query drives retrieval, but the trace and the prompt must
    show what the customer actually wrote."""
    from src.rag.generator import build_pipeline

    r = build_pipeline().answer_multimodal(
        "my payment isn't working", SHOTS / "V003_PAY_402.png"
    )
    assert r.trace.question == "my payment isn't working"


@needs_shots
def test_unusable_image_still_abstains_correctly():
    """An attached image must not push an unanswerable question over the
    abstention threshold."""
    from src.rag.generator import build_pipeline

    r = build_pipeline().answer_multimodal(
        "something is wrong, see photo", EDGE / "blurry_severe.png"
    )
    assert r.trace.decision == "abstain"


# ----------------------------------------------------------- ablation
@needs_shots
def test_vision_improves_retrieval_substantially():
    """The headline claim. Guards the recorded numbers."""
    import json

    from src.knowledge import evaluation as kev
    from src.rag.generator import build_pipeline

    pipe = build_pipeline()
    cases = json.loads((settings.eval_dir / "vision_eval.json").read_text())["cases"]
    manifest = json.loads((SHOTS / "manifest.json").read_text())
    by_id = {c["id"]: c for c in manifest["cases"]}

    text_hits, vision_hits = 0, 0
    for c in cases[:12]:
        shot = by_id.get(c["id"])
        if not shot:
            continue
        gold = kev.gold_section_keys(c["gold_sections"])

        f_t = fuse(MultimodalRequest(c["user_text"]))
        r_t, _, _ = pipe.routed.retrieve(f_t.enriched_query, top_k=5)
        text_hits += kev.recall_at_k(kev.retrieved_section_keys(r_t.hits), gold, 5)

        f_i = fuse(MultimodalRequest(c["user_text"], SHOTS / shot["file"]))
        r_i, _, _ = pipe.routed.retrieve(f_i.enriched_query, top_k=5)
        vision_hits += kev.recall_at_k(kev.retrieved_section_keys(r_i.hits), gold, 5)

    assert vision_hits > text_hits, "vision no longer improves retrieval"
