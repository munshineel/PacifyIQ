"""Generate mock screenshots for the vision evaluation set.

Renders the 25 cases in data/eval/vision_eval.json as PNGs, plus an edge-case
set covering the failure modes the validator and vision layer must handle.

In every case the error code appears ONLY in the image. The accompanying
customer text is deliberately vague ("my monitor keeps going black"), which is
what makes the text-only vs text+vision ablation meaningful: without the image
the retriever has nothing specific to match.

    python scripts/data_generation/gen_screenshots.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "screenshots"
EDGE = OUT / "edge_cases"
OUT.mkdir(parents=True, exist_ok=True)
EDGE.mkdir(parents=True, exist_ok=True)

random.seed(42)

FONT_DIR = Path("/usr/share/fonts/truetype")
SANS = FONT_DIR / "dejavu" / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "dejavu" / "DejaVuSans-Bold.ttf"
MONO = FONT_DIR / "dejavu" / "DejaVuSansMono.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# =====================================================================
# UI surface renderers
# =====================================================================

def checkout_error(code: str, message: str) -> Image.Image:
    """A web checkout page with a payment error banner."""
    W, H = 900, 620
    img = Image.new("RGB", (W, H), "#f4f5f7")
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 64], fill="#1a2b45")
    d.text((28, 22), "Pacify", font=font(SANS_BOLD, 22), fill="white")
    d.text((W - 190, 26), "Secure checkout", font=font(SANS, 14), fill="#a9b8cc")

    d.rectangle([60, 110, W - 60, 300], fill="white", outline="#dde1e6")
    d.rectangle([60, 110, W - 60, 116], fill="#c0392b")
    d.text((88, 140), "Payment could not be completed",
           font=font(SANS_BOLD, 20), fill="#c0392b")
    d.text((88, 178), message, font=font(SANS, 15), fill="#33383d")
    d.text((88, 214), f"Error code: {code}", font=font(MONO, 16), fill="#c0392b")
    d.text((88, 246), "Reference: TXN-8841-2026", font=font(MONO, 12), fill="#7a828a")

    d.rectangle([60, 340, W - 60, 520], fill="white", outline="#dde1e6")
    d.text((88, 366), "Order summary", font=font(SANS_BOLD, 15), fill="#33383d")
    for i, (label, val) in enumerate([
        ("Pacify ProBook 14", "Rs 64,900"), ("Delivery", "Free"),
        ("Total", "Rs 64,900"),
    ]):
        y = 400 + i * 30
        d.text((88, y), label, font=font(SANS, 14), fill="#555c63")
        d.text((W - 200, y), val, font=font(SANS, 14), fill="#33383d")

    d.rectangle([88, 545, 260, 585], fill="#1a2b45")
    d.text((124, 556), "Retry payment", font=font(SANS_BOLD, 14), fill="white")
    return img


def monitor_osd(code: str, message: str) -> Image.Image:
    """A monitor on-screen-display panel over a dark background."""
    W, H = 800, 560
    img = Image.new("RGB", (W, H), "#0a0a0c")
    d = ImageDraw.Draw(img)

    d.rectangle([180, 160, 620, 400], fill="#16181d", outline="#3d4450", width=2)
    d.rectangle([180, 160, 620, 200], fill="#22262e")
    d.text((204, 170), "PACIFY VISION 27", font=font(SANS_BOLD, 16), fill="#dfe3e8")

    d.text((204, 226), message, font=font(SANS, 15), fill="#e8b84b")
    d.text((204, 262), code, font=font(MONO, 20), fill="#e05c4b")

    for i, (k, v) in enumerate([
        ("Input", "DisplayPort"), ("Resolution", "2560 x 1440"),
        ("Refresh", "75 Hz"),
    ]):
        y = 300 + i * 26
        d.text((204, y), k, font=font(SANS, 12), fill="#8b93a0")
        d.text((400, y), v, font=font(MONO, 12), fill="#c8cfd8")

    d.text((204, 372), "Press joystick to dismiss", font=font(SANS, 11), fill="#6b7280")
    return img


def system_notification(code: str, title: str, body: str) -> Image.Image:
    """An OS tray notification."""
    W, H = 760, 420
    img = Image.new("RGB", (W, H), "#2b2f38")
    d = ImageDraw.Draw(img)

    d.rectangle([0, H - 44, W, H], fill="#1f232a")
    d.text((16, H - 32), "14:32   21/08/2026", font=font(SANS, 12), fill="#9aa3b0")

    # Light panel on a dark desktop: a dark-on-dark notification renders with
    # too little contrast for OCR, which is a rendering artefact rather than a
    # realistic screenshot - real notification panels are high contrast.
    d.rectangle([W - 420, 40, W - 24, 220], fill="#f4f6f9", outline="#c3c9d0")
    d.rectangle([W - 420, 40, W - 414, 220], fill="#d98c00")
    d.text((W - 394, 62), title, font=font(SANS_BOLD, 16), fill="#1a2b45")
    d.text((W - 394, 96), body, font=font(SANS, 14), fill="#33383d")
    # Draw the code on a light plate: MONO orange on light grey was too low
    # contrast for reliable OCR, and the fix belongs in the mock, not in a
    # lowered confidence threshold.
    d.rectangle([W - 398, 136, W - 200, 168], fill="#fdf0e0")
    d.text((W - 392, 142), f"Code {code}", font=font(MONO, 16), fill="#8a4500")
    d.text((W - 394, 184), "Pacify Diagnostics", font=font(SANS, 12), fill="#6b7280")
    return img


def stop_screen(code: str, message: str) -> Image.Image:
    """A full-screen boot/stop error."""
    W, H = 900, 560
    img = Image.new("RGB", (W, H), "#1b4f8a")
    d = ImageDraw.Draw(img)
    d.text((80, 120), ":(", font=font(SANS, 72), fill="white")
    d.text((80, 236), "Your device ran into a problem and needs to restart.",
           font=font(SANS, 19), fill="white")
    d.text((80, 300), message, font=font(SANS, 15), fill="#cfe0f2")
    d.text((80, 360), f"Stop code: {code}", font=font(MONO, 17), fill="white")
    d.text((80, 420), "20% complete", font=font(SANS, 14), fill="#cfe0f2")
    return img


def diagnostics_app(code: str, metric: str, value: str, status: str) -> Image.Image:
    """The Pacify Diagnostics desktop app."""
    W, H = 860, 580
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 56], fill="#eef1f5")
    d.text((24, 18), "Pacify Diagnostics", font=font(SANS_BOLD, 17), fill="#1a2b45")
    d.rectangle([0, 56, 200, H], fill="#f7f9fb")
    for i, item in enumerate(["Overview", "Battery", "Storage", "Memory",
                              "Display", "Thermals"]):
        d.text((26, 90 + i * 38), item, font=font(SANS, 14), fill="#4a5560")

    d.rectangle([230, 96, W - 40, 250], fill="#fdf3f2", outline="#e6bcb6")
    d.text((256, 120), "Issue detected", font=font(SANS_BOLD, 17), fill="#c0392b")
    d.text((256, 156), metric, font=font(SANS, 14), fill="#33383d")
    d.text((256, 186), value, font=font(SANS_BOLD, 22), fill="#c0392b")
    d.text((256, 220), f"Code {code}", font=font(MONO, 14), fill="#c0392b")

    d.rectangle([230, 286, W - 40, 420], fill="#f7f9fb", outline="#dde1e6")
    d.text((256, 306), "System", font=font(SANS_BOLD, 14), fill="#33383d")
    for i, (k, v) in enumerate([("Model", "Pacify ProBook 14"),
                                ("Serial", "PB14-2026-88431"),
                                ("Status", status)]):
        y = 340 + i * 26
        d.text((256, y), k, font=font(SANS, 13), fill="#6b7280")
        d.text((440, y), v, font=font(MONO, 13), fill="#33383d")
    return img


def settings_dialog(code: str, title: str, body: str) -> Image.Image:
    """An OS settings dialog, e.g. network or sound."""
    W, H = 720, 480
    img = Image.new("RGB", (W, H), "#f0f2f5")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 48], fill="#ffffff", outline="#dde1e6")
    d.text((20, 15), title, font=font(SANS_BOLD, 15), fill="#1a2b45")

    d.rectangle([40, 90, W - 40, 260], fill="white", outline="#e0a800", width=2)
    d.text((66, 116), "Cannot complete this action",
           font=font(SANS_BOLD, 16), fill="#8a6d00")
    d.text((66, 152), body, font=font(SANS, 14), fill="#4a5560")
    d.text((66, 200), code, font=font(MONO, 15), fill="#8a6d00")

    d.rectangle([W - 220, 300, W - 120, 340], fill="#ffffff", outline="#c3c9d0")
    d.text((W - 200, 312), "Cancel", font=font(SANS, 13), fill="#4a5560")
    d.rectangle([W - 110, 300, W - 40, 340], fill="#1a2b45")
    d.text((W - 96, 312), "Retry", font=font(SANS_BOLD, 13), fill="white")
    return img


def visible_symptom(caption: str) -> Image.Image:
    """A photograph-style shot with no code shown - the symptom is visual."""
    W, H = 780, 520
    img = Image.new("RGB", (W, H), "#101216")
    d = ImageDraw.Draw(img)
    d.rectangle([90, 70, 690, 410], fill="#000000", outline="#2a2f38", width=6)
    for x in range(90, 690, 3):
        shade = int(14 + 10 * random.random())
        d.line([(x, 70), (x, 410)], fill=(shade, shade, shade + 2))
    d.rectangle([90, 70, 690, 130], fill="#050506")
    d.text((120, 448), caption, font=font(SANS, 13), fill="#8b93a0")
    d.text((120, 470), "Pacify Vision 27 - photographed by customer",
           font=font(SANS, 11), fill="#5b616b")
    return img


# =====================================================================
# Case mapping: surface -> renderer
# =====================================================================

RENDERERS = {
    "checkout page": lambda c, m: checkout_error(c, m),
    "monitor OSD": lambda c, m: monitor_osd(c, m),
    "bank redirect": lambda c, m: checkout_error(c, m),
    "system tray": lambda c, m: system_notification(c, "Hardware alert", m),
    "network dialog": lambda c, m: settings_dialog(c, "Network settings", m),
    "stop screen": lambda c, m: stop_screen(c, m),
    "diagnostics app": lambda c, m: diagnostics_app(c, m, "Threshold exceeded", "Attention"),
    "sound dialog": lambda c, m: settings_dialog(c, "Sound settings", m),
    "device manager": lambda c, m: settings_dialog(c, "Device manager", m),
    "visible symptom": lambda c, m: visible_symptom(m),
}

MESSAGES = {
    "PAY-402": "The payment gateway did not respond in time.",
    "PAY-511": "Authentication with your bank was not completed.",
    "PAY-207": "Insufficient funds or credit limit exceeded.",
    "PAY-309": "This card is not enabled for online transactions.",
    "PAY-118": "Transaction declined by your issuing bank.",
    "PAY-604": "Daily transaction limit exceeded.",
    "ERR-DP-0x004": "DisplayPort handshake failed",
    "ERR-DP-0x011": "Refresh rate not supported on this cable",
    "ERR-HD-0x002": "HDMI signal out of range",
    "BAT-119": "Battery health critical",
    "BAT-042": "Charger not recognised",
    "BAT-007": "Charging paused - temperature out of range",
    "WIFI-503": "Wireless driver failed to initialise.",
    "WIFI-211": "Authentication timeout with access point.",
    "SYS-0x0000007B": "INACCESSIBLE_BOOT_DEVICE",
    "SYS-0x000000EF": "CRITICAL_PROCESS_DIED",
    "THRM-88": "Sustained thermal throttling",
    "THRM-12": "Fan not responding",
    "DSP-014": "Screen very dim, backlight appears to have failed",
    "DSP-051": "Panel cable fault detected",
    "AUD-330": "Audio driver conflict detected.",
    "KEY-018": "Keyboard controller not responding",
    "STO-440": "Storage health below threshold",
    "MEM-221": "Memory error detected during test",
    "CAM-090": "Camera module not detected.",
}


def main() -> int:
    eval_path = ROOT / "data" / "eval" / "vision_eval.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    print("=" * 70)
    print("GENERATING VISION EVALUATION SCREENSHOTS")
    print("=" * 70)

    manifest = []
    for case in data["cases"]:
        code = case["code_in_image_only"]
        surface = case["image_surface"]
        renderer = RENDERERS.get(surface, RENDERERS["diagnostics app"])
        message = MESSAGES.get(code, "An error occurred.")

        img = renderer(code, message)
        name = f"{case['id']}_{code.replace('-', '_')}.png"
        img.save(OUT / name)

        manifest.append({
            "id": case["id"], "file": name, "code": code, "surface": surface,
            "user_text": case["user_text"], "size": img.size,
        })
        print(f"  {case['id']}  {code:16s} {surface:18s} {img.size}  {name}")

    # ---- edge cases -------------------------------------------------
    print("\n" + "=" * 70)
    print("EDGE CASES")
    print("=" * 70)
    edge = []

    def save_edge(img, name, kind, note, fmt="PNG"):
        p = EDGE / name
        img.save(p, fmt)
        edge.append({"file": name, "kind": kind, "note": note,
                     "size": img.size, "bytes": p.stat().st_size})
        print(f"  {kind:22s} {name:34s} {img.size}  {p.stat().st_size/1024:.0f} KB")

    base = checkout_error("PAY-402", MESSAGES["PAY-402"])

    save_edge(base, "valid_clear.png", "valid", "clean, code readable")
    save_edge(base.filter(ImageFilter.GaussianBlur(2.2)), "blurry_mild.png",
              "blurry", "mild blur, code may be partially readable")
    save_edge(base.filter(ImageFilter.GaussianBlur(6.0)), "blurry_severe.png",
              "blurry", "severe blur, code unreadable - must report unknown")

    dark = Image.new("RGB", base.size, "black")
    dark.paste(base.point(lambda p: int(p * 0.18)), (0, 0))
    save_edge(dark, "too_dark.png", "low quality", "underexposed, text illegible")

    small = base.resize((160, 110))
    save_edge(small, "tiny_downscaled.png", "low resolution",
              "downscaled beyond legibility")

    # irrelevant: a product photo, no error anywhere
    prod = Image.new("RGB", (700, 500), "#eef1f5")
    dp = ImageDraw.Draw(prod)
    dp.rectangle([150, 130, 550, 370], fill="#c8cfd8", outline="#9aa3b0", width=3)
    dp.text((236, 410), "Pacify ProBook 14", font=font(SANS_BOLD, 20), fill="#33383d")
    save_edge(prod, "irrelevant_product.png", "irrelevant",
              "product photo, no error information")

    blank = Image.new("RGB", (600, 400), "white")
    save_edge(blank, "blank_white.png", "no information", "empty image")

    noise = Image.new("RGB", (500, 400))
    noise.putdata([(random.randint(0, 255),) * 3 for _ in range(500 * 400)])
    save_edge(noise, "noise.png", "no information", "random noise")

    huge = base.resize((5200, 3600))
    save_edge(huge, "oversized_5200x3600.png", "oversized",
              "very large image, must be downscaled before analysis")

    save_edge(base.convert("RGB"), "unsupported_format.bmp", "unsupported format",
              "BMP - should be rejected or converted", fmt="BMP")

    # truncated file
    truncated = EDGE / "corrupt_truncated.png"
    raw = (EDGE / "valid_clear.png").read_bytes()
    truncated.write_bytes(raw[: len(raw) // 3])
    edge.append({"file": "corrupt_truncated.png", "kind": "corrupt",
                 "note": "truncated file, must fail gracefully",
                 "size": None, "bytes": truncated.stat().st_size})
    print(f"  {'corrupt':22s} {'corrupt_truncated.png':34s} truncated")

    # ---- manifest ---------------------------------------------------
    (OUT / "manifest.json").write_text(json.dumps({
        "description": "Mock screenshots for the vision evaluation set. "
                       "Synthetic, rendered by scripts/data_generation/"
                       "gen_screenshots.py. In every case the error code "
                       "appears ONLY in the image, never in the user text.",
        "n_cases": len(manifest),
        "cases": manifest,
        "edge_cases": edge,
    }, indent=2))

    print(f"\n{len(manifest)} evaluation screenshots -> {OUT}")
    print(f"{len(edge)} edge cases -> {EDGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
