"""PHASE 8 — Vision evaluation.

The headline measurement: does the screenshot actually help?

Every case in vision_eval.json is constructed so the error code exists ONLY in
the image, with deliberately vague accompanying text ("my monitor keeps going
black"). Running retrieval with and without the image isolates the contribution.

Also tests the edge cases the workflow must survive: blurry, irrelevant, blank,
oversized, corrupt and unsupported uploads.

    python scripts/evaluate_vision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import settings  # noqa: E402
from src.knowledge import evaluation as kev  # noqa: E402
from src.multimodal.fusion import (CustomerContext, MultimodalRequest,  # noqa: E402
                                   fuse)
from src.multimodal.validation import validate_image  # noqa: E402
from src.multimodal.vision import Evidence, analyze_image  # noqa: E402
from src.rag.generator import build_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SHOTS = settings.eval_dir / "screenshots"


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 46)

    data = json.loads((settings.eval_dir / "vision_eval.json").read_text())
    manifest = json.loads((SHOTS / "manifest.json").read_text())
    by_id = {c["id"]: c for c in manifest["cases"]}

    print("=" * 80)
    print(f"VISION EVALUATION  ({len(data['cases'])} cases)")
    print("=" * 80)
    print("  Every case: the error code appears ONLY in the image.")

    pipe = build_pipeline()

    # =================================================================
    # 1. Extraction accuracy
    # =================================================================
    print("\n" + "=" * 80)
    print("1. ERROR-CODE EXTRACTION")
    print("=" * 80)

    rows = []
    for case in data["cases"]:
        shot = by_id.get(case["id"])
        if not shot:
            continue
        path = SHOTS / shot["file"]
        a = analyze_image(path, user_text=case["user_text"])
        expected = case["code_in_image_only"].upper()
        got = (a.error_code or "").upper()
        ev = a.evidence.get("error_code", Evidence.UNKNOWN).value

        # A "visible symptom" case shows no code at all - the fault is visual
        # (a dark panel). Correct behaviour is to report the symptom and NOT a
        # code, so extracting nothing is the right answer.
        no_code_by_design = case["image_surface"] == "visible symptom"
        correct = (got == "") if no_code_by_design else (got == expected)

        rows.append({
            "id": case["id"],
            "expected": expected,
            "extracted": a.error_code or "-",
            "correct": correct,
            "code_shown": not no_code_by_design,
            "evidence": ev,
            "surface": case["image_surface"],
            "image_type": a.image_type,
            "ocr_conf": round(a.ocr_confidence, 2),
            "confidence": round(a.confidence, 2),
        })

    ext = pd.DataFrame(rows)
    ext.to_csv(RESULTS / "vision_extraction.csv", index=False)

    acc = ext["correct"].mean()
    visible = (ext["evidence"] == "visible").mean()
    shown = ext[ext["code_shown"]]
    print(f"  extraction accuracy        {acc:.3f}  ({ext['correct'].sum()}/{len(ext)})")
    print(f"  on cases showing a code    {shown['correct'].mean():.3f}  "
          f"({shown['correct'].sum()}/{len(shown)})")
    print(f"  marked VISIBLE             {visible:.3f}")
    print(f"  marked INFERRED            {(ext['evidence'] == 'inferred').mean():.3f}")
    print(f"  marked UNKNOWN             {(ext['evidence'] == 'unknown').mean():.3f}")
    print(f"  mean OCR confidence        {ext['ocr_conf'].mean():.3f}")

    print("\n  by UI surface:")
    surf = ext.groupby("surface").agg(
        n=("correct", "size"), accuracy=("correct", "mean")
    ).round(3).sort_values("accuracy")
    print(surf.to_string())

    wrong = ext[~ext["correct"]]
    if len(wrong):
        print(f"\n  failed extractions ({len(wrong)}):")
        print(wrong[["id", "expected", "extracted", "evidence", "surface"]]
              .to_string(index=False))

    # =================================================================
    # 2. THE ABLATION
    # =================================================================
    print("\n" + "=" * 80)
    print("2. RETRIEVAL ABLATION — text-only vs text+vision")
    print("=" * 80)

    rows = []
    for case in data["cases"]:
        shot = by_id.get(case["id"])
        if not shot:
            continue
        gold = kev.gold_section_keys(case["gold_sections"])

        # text only
        f_text = fuse(MultimodalRequest(text=case["user_text"], image_path=None))
        res_t, _, _ = pipe.routed.retrieve(f_text.enriched_query, top_k=5)
        got_t = kev.retrieved_section_keys(res_t.hits)

        # text + vision
        f_img = fuse(MultimodalRequest(
            text=case["user_text"], image_path=SHOTS / shot["file"]
        ))
        res_i, _, _ = pipe.routed.retrieve(f_img.enriched_query, top_k=5)
        got_i = kev.retrieved_section_keys(res_i.hits)

        rows.append({
            "id": case["id"],
            "text": case["user_text"][:38],
            "code": case["code_in_image_only"],
            "recall_text": kev.recall_at_k(got_t, gold, 5),
            "recall_vision": kev.recall_at_k(got_i, gold, 5),
            "mrr_text": round(kev.reciprocal_rank(got_t, gold), 3),
            "mrr_vision": round(kev.reciprocal_rank(got_i, gold), 3),
            "bm25_text": round(res_t.max_bm25_score, 1),
            "bm25_vision": round(res_i.max_bm25_score, 1),
            "contributed": f_img.image_contributed,
        })

    abl = pd.DataFrame(rows)
    abl.to_csv(RESULTS / "vision_ablation.csv", index=False)

    rt, ri = abl["recall_text"].mean(), abl["recall_vision"].mean()
    mt, mi = abl["mrr_text"].mean(), abl["mrr_vision"].mean()
    bt, bi = abl["bm25_text"].mean(), abl["bm25_vision"].mean()

    print(f"\n  {'metric':22s} {'text only':>12s} {'text+vision':>12s} {'delta':>10s}")
    print("  " + "-" * 60)
    print(f"  {'recall@5':22s} {rt:12.3f} {ri:12.3f} {ri - rt:+10.3f}")
    print(f"  {'MRR':22s} {mt:12.3f} {mi:12.3f} {mi - mt:+10.3f}")
    print(f"  {'max BM25':22s} {bt:12.1f} {bi:12.1f} {bi - bt:+10.1f}")
    print(f"\n  image contributed terms in {abl['contributed'].sum()}/{len(abl)} cases")

    fixed = abl[(abl["recall_vision"] > abl["recall_text"])]
    broken = abl[(abl["recall_vision"] < abl["recall_text"])]
    print(f"  fixed by vision   {len(fixed)}")
    print(f"  broken by vision  {len(broken)}")
    if len(fixed):
        print("\n  cases the image rescued:")
        print(fixed[["id", "text", "code", "bm25_text", "bm25_vision"]]
              .to_string(index=False))
    if len(broken):
        print("\n  cases the image damaged:")
        print(broken[["id", "text", "code", "bm25_text", "bm25_vision"]]
              .to_string(index=False))

    # =================================================================
    # 3. Edge cases
    # =================================================================
    print("\n" + "=" * 80)
    print("3. EDGE CASES")
    print("=" * 80)

    rows = []
    for e in manifest["edge_cases"]:
        path = SHOTS / "edge_cases" / e["file"]
        v, _ = validate_image(path)
        a = analyze_image(path, user_text="something is wrong")
        f = fuse(MultimodalRequest(text="something is wrong", image_path=path))
        rows.append({
            "file": e["file"],
            "kind": e["kind"],
            "validation": v.status.value,
            "code": a.error_code or "-",
            "code_evidence": a.evidence.get("error_code", Evidence.UNKNOWN).value,
            "useful": a.is_useful,
            "contributed": f.image_contributed,
            "conf": round(a.confidence, 2),
            "reason": (a.reason or v.reason)[:44],
        })
    edge = pd.DataFrame(rows)
    edge.to_csv(RESULTS / "vision_edge_cases.csv", index=False)
    print(edge.to_string(index=False))

    # The property that matters most: no INVENTED codes.
    #
    # Note the distinction. Reading PAY-402 off an underexposed screenshot that
    # genuinely contains PAY-402 is a correct read, not a hallucination - the
    # image is poor quality but the text survived. What must never happen is a
    # code appearing where none exists, or a different code being reported.
    NO_CODE_PRESENT = {
        "blurry_severe.png", "blank_white.png", "noise.png",
        "irrelevant_product.png", "tiny_downscaled.png", "blurry_mild.png",
        "corrupt_truncated.png", "unsupported_format.bmp",
    }
    no_code = edge[edge["file"].isin(NO_CODE_PRESENT)]
    hallucinated = no_code[no_code["code"] != "-"]

    print(f"\n  images with no legible code: {len(no_code)}")
    print(f"  codes invented on them:      {len(hallucinated)}")
    if len(hallucinated) == 0:
        print("  -> zero invented codes. The system never claims to read what it cannot.")
    else:
        print(hallucinated.to_string(index=False))

    # And the converse: a poor-quality image whose text IS legible should still
    # be read, otherwise the system is uselessly conservative.
    legible_poor = edge[edge["file"].isin(["too_dark.png", "oversized_5200x3600.png"])]
    print(f"  poor-quality but legible images correctly read: "
          f"{(legible_poor['code'] != '-').sum()}/{len(legible_poor)}")

    # =================================================================
    # 4. End-to-end
    # =================================================================
    print("\n" + "=" * 80)
    print("4. END-TO-END EXAMPLES")
    print("=" * 80)

    examples = [
        ("my payment isn't working", SHOTS / "V003_PAY_402.png"),
        ("my monitor keeps going black randomly", SHOTS / "V001_ERR_DP_0x004.png"),
        ("something is wrong, see photo", SHOTS / "edge_cases" / "blurry_severe.png"),
        ("here is a picture of it", SHOTS / "edge_cases" / "irrelevant_product.png"),
    ]
    for text, path in examples:
        r = pipe.answer_multimodal(text, path)
        print(f"\n  TEXT      {text}")
        print(f"  IMAGE     {Path(path).name}")
        print(f"  FUSION    {r.fused.summary()}")
        print(f"  DECISION  {r.trace.decision}")
        print(f"  CITED     {[str(c) for c in r.response.citations][:3]}")
        print(f"  TRACE     {r.trace.summary()}")

    # =================================================================
    (RESULTS / "vision_summary.json").write_text(json.dumps({
        "n_cases": len(ext),
        "extraction_accuracy": round(float(acc), 4),
        "marked_visible": round(float(visible), 4),
        "mean_ocr_confidence": round(float(ext["ocr_conf"].mean()), 4),
        "ablation": {
            "recall_text_only": round(float(rt), 4),
            "recall_text_vision": round(float(ri), 4),
            "recall_delta": round(float(ri - rt), 4),
            "mrr_text_only": round(float(mt), 4),
            "mrr_text_vision": round(float(mi), 4),
            "bm25_text_only": round(float(bt), 2),
            "bm25_text_vision": round(float(bi), 2),
            "fixed": int(len(fixed)),
            "broken": int(len(broken)),
        },
        "edge_cases": {
            "n": len(edge),
            "hallucinated_codes": int(len(hallucinated)),
        },
    }, indent=2))

    print("\n" + "=" * 80)
    print("VISION EVALUATION COMPLETE")
    print(f"  extraction accuracy  {acc:.3f}")
    print(f"  recall@5   {rt:.3f} -> {ri:.3f}  ({ri - rt:+.3f})")
    print(f"  MRR        {mt:.3f} -> {mi:.3f}  ({mi - mt:+.3f})")
    print(f"  invented codes on unreadable images: {len(hallucinated)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
