"""Build any missing artifacts at startup.

The index, classifier and database are DERIVED data. Committing them means a
22 MB repository and two sources of truth that can silently disagree - an index
built from an older corpus retrieves stale text with no error anywhere.

Measured rebuild cost: ~29s total, once, on a cold container. Cheaper than
carrying the files, and it guarantees artifacts match the source.

Called by app/Home.py on first run. Safe to run repeatedly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import settings  # noqa: E402

# ORDER MATTERS. seed_db creates tables; setup_database applies views over
# them; build_pdfs renders the corpus the index is built from.
ARTIFACTS = [
    (settings.db_path, "scripts/data_generation/seed_db.py", "database tables"),
    (settings.data_dir / "db" / ".views_applied",
     "scripts/setup_database.py", "database views"),
    (settings.documents_dir / "return_policy_v2.pdf",
     "scripts/data_generation/build_pdfs.py", "knowledge corpus"),
    (settings.root / "models" / "intent_classifier.joblib",
     "scripts/train_intent_classifier.py", "intent classifier"),
    (settings.index_dir / "vectors.npy", "scripts/build_index.py",
     "knowledge index"),
    (settings.eval_dir / "screenshots" / "manifest.json",
     "scripts/data_generation/gen_screenshots.py", "evaluation screenshots"),
]


def missing() -> list[tuple[Path, str, str]]:
    return [a for a in ARTIFACTS if not Path(a[0]).exists()]


def build(progress=None) -> tuple[bool, list[str]]:
    """Build whatever is absent. Returns (ok, log)."""
    log: list[str] = []
    todo = missing()
    if not todo:
        return True, ["all artifacts present"]

    for marker, script, label in todo:
        msg = f"building {label}..."
        log.append(msg)
        if progress:
            progress(msg)

        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT, capture_output=True, text=True, timeout=600)

        # setup_database.py has no single output file, so it writes a marker.
        if str(marker).endswith(".views_applied") and result.returncode == 0:
            Path(marker).parent.mkdir(parents=True, exist_ok=True)
            Path(marker).write_text("views applied")

        if result.returncode != 0 or not Path(marker).exists():
            tail = (result.stderr or result.stdout or "")[-400:]
            log.append(f"FAILED: {label}\n{tail}")
            return False, log
        log.append(f"  {label} ready")

    return True, log


if __name__ == "__main__":
    ok, log = build(progress=lambda m: print(f"  {m}"))
    for line in log:
        print(line)
    raise SystemExit(0 if ok else 1)
