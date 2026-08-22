"""Image validation and preprocessing.

Stage 1 of the multimodal workflow. Runs before any model sees the file,
because a vision model is an expensive and unreliable way to discover that an
upload is a truncated BMP.

Validation is deliberately strict about *format* and lenient about *content*.
A blurry screenshot is a valid upload that the vision layer should report as
unreadable; it is not a validation failure. Conflating the two would either
reject legitimate uploads or push a corrupt file into the model.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

# Formats accepted from customers. BMP and TIFF are excluded: they are rarely
# produced by phones or screenshot tools and are disproportionately large.
ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "GIF"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

MAX_BYTES = 10 * 1024 * 1024        # 10 MB
MIN_BYTES = 200                      # smaller than any real screenshot
MAX_DIMENSION = 8000                 # guards against decompression bombs
MIN_DIMENSION = 50
MAX_PIXELS = 40_000_000

# Vision models resize internally; sending more than this wastes tokens and
# adds latency for no gain in legibility.
TARGET_MAX_DIMENSION = 1600


class ValidationStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    WARNING = "warning"          # usable, but quality is poor


@dataclass
class ImageValidation:
    status: ValidationStatus
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    filename: str = ""
    format: str | None = None
    mode: str | None = None
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    megapixels: float = 0.0

    # quality signals, used by the vision layer to calibrate confidence
    mean_brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    is_likely_blurry: bool = False
    is_likely_blank: bool = False

    was_resized: bool = False
    was_converted: bool = False

    @property
    def ok(self) -> bool:
        return self.status != ValidationStatus.REJECTED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def summary(self) -> str:
        if not self.ok:
            return f"REJECTED: {self.reason}"
        bits = [f"{self.format} {self.width}x{self.height}",
                f"{self.size_bytes / 1024:.0f}KB",
                f"sharp={self.sharpness:.1f}"]
        if self.warnings:
            bits.append("warnings=" + ", ".join(self.warnings))
        return " | ".join(bits)


def _quality_signals(img: Image.Image, v: ImageValidation) -> None:
    """Cheap statistics that predict whether text will be legible.

    These feed the vision layer's confidence rather than gating acceptance.
    A model should be told "this image is blurry" so it can say it cannot read
    the code, instead of guessing.
    """
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    v.mean_brightness = round(stat.mean[0], 2)
    v.contrast = round(stat.stddev[0], 2)

    # Variance of the Laplacian: the standard cheap blur measure. A sharp edge
    # produces a large second derivative; blur flattens it.
    #
    # PIL's FIND_EDGES was tried first and rejected: it scored a mildly blurred
    # image at 15.2 and a severely blurred one at 14.8, which cannot separate
    # "readable" from "unreadable". A real Laplacian convolution separates the
    # same pair by an order of magnitude.
    import numpy as np

    arr = np.asarray(gray, dtype=np.float32)
    lap = (
        -4 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1] + arr[2:, 1:-1]
        + arr[1:-1, :-2] + arr[1:-1, 2:]
    )
    v.sharpness = round(float(lap.var()), 2)

    if v.sharpness < 60:
        v.is_likely_blurry = True
        v.warnings.append("low sharpness - text may be unreadable")
    if v.contrast < 12:
        v.is_likely_blank = True
        v.warnings.append("very low contrast - image may be blank or uniform")
    if v.mean_brightness < 35:
        v.warnings.append("very dark - text may be illegible")
    elif v.mean_brightness > 240:
        v.warnings.append("very bright - image may be washed out or blank")
    if max(v.width, v.height) < 400:
        v.warnings.append("low resolution - small text will not be readable")
    if v.sharpness < 15:
        v.warnings.append("severely blurred - do not expect any text to be read")


def validate_image(
    source: Path | str | bytes, filename: str | None = None
) -> tuple[ImageValidation, Image.Image | None]:
    """Validate an upload. Returns (report, decoded image or None).

    Never raises on bad input - a corrupt upload is a normal event in a support
    channel, not an exception.
    """
    v = ImageValidation(status=ValidationStatus.OK)

    # ---- read ------------------------------------------------------
    if isinstance(source, (str, Path)):
        path = Path(source)
        v.filename = filename or path.name
        if not path.exists():
            v.status = ValidationStatus.REJECTED
            v.reason = "file not found"
            return v, None
        data = path.read_bytes()
    else:
        data = source
        v.filename = filename or "upload"

    v.size_bytes = len(data)

    # ---- size ------------------------------------------------------
    if v.size_bytes > MAX_BYTES:
        v.status = ValidationStatus.REJECTED
        v.reason = f"file too large ({v.size_bytes / 1024 / 1024:.1f} MB, max 10 MB)"
        return v, None
    if v.size_bytes < MIN_BYTES:
        v.status = ValidationStatus.REJECTED
        v.reason = f"file too small ({v.size_bytes} bytes) - probably not an image"
        return v, None

    # ---- extension -------------------------------------------------
    ext = Path(v.filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        v.status = ValidationStatus.REJECTED
        v.reason = (f"unsupported file type '{ext}'. "
                    f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        return v, None

    # ---- decode ----------------------------------------------------
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                       # structural check, consumes the file
        img = Image.open(io.BytesIO(data))  # reopen for actual use
        img.load()                          # forces full decode
    except Exception as e:
        v.status = ValidationStatus.REJECTED
        v.reason = f"corrupt or unreadable image ({type(e).__name__})"
        return v, None

    v.format = img.format
    v.mode = img.mode
    v.width, v.height = img.size
    v.megapixels = round(v.width * v.height / 1e6, 2)

    if v.format and v.format.upper() not in ALLOWED_FORMATS:
        v.status = ValidationStatus.REJECTED
        v.reason = (f"unsupported image format '{v.format}'. "
                    f"Accepted: {', '.join(sorted(ALLOWED_FORMATS))}")
        return v, None

    # ---- dimensions ------------------------------------------------
    if max(v.width, v.height) > MAX_DIMENSION:
        v.status = ValidationStatus.REJECTED
        v.reason = f"image too large ({v.width}x{v.height}, max {MAX_DIMENSION}px)"
        return v, None
    if min(v.width, v.height) < MIN_DIMENSION:
        v.status = ValidationStatus.REJECTED
        v.reason = f"image too small ({v.width}x{v.height}, min {MIN_DIMENSION}px)"
        return v, None
    if v.width * v.height > MAX_PIXELS:
        v.status = ValidationStatus.REJECTED
        v.reason = f"too many pixels ({v.megapixels} MP)"
        return v, None

    # ---- normalise -------------------------------------------------
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        v.was_converted = True
    elif img.mode == "L":
        img = img.convert("RGB")
        v.was_converted = True

    _quality_signals(img, v)
    if v.warnings:
        v.status = ValidationStatus.WARNING
    return v, img


def preprocess(img: Image.Image, v: ImageValidation,
               target_max: int = TARGET_MAX_DIMENSION) -> Image.Image:
    """Downscale oversized images before analysis.

    Vision models resize internally anyway, so sending a 5200x3600 screenshot
    costs bandwidth, tokens and latency for identical output.
    """
    if max(img.size) <= target_max:
        return img
    ratio = target_max / max(img.size)
    new = (int(img.width * ratio), int(img.height * ratio))
    v.was_resized = True
    v.warnings.append(f"downscaled from {img.width}x{img.height} to {new[0]}x{new[1]}")
    return img.resize(new, Image.LANCZOS)


def to_base64(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    """Encode for an API call."""
    import base64

    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    from src.config.settings import settings

    edge_dir = settings.eval_dir / "screenshots" / "edge_cases"
    print(f"{'file':34s} {'status':10s} detail")
    print("-" * 100)
    for p in sorted(edge_dir.glob("*")):
        v, img = validate_image(p)
        print(f"{p.name:34s} {v.status.value:10s} {v.summary()}")
