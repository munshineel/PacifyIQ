"""PHASE 2 — Dataset audit runner.

    python scripts/run_audit.py            # print to stdout
    python scripts/run_audit.py --save     # also write reports/audit_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.eda import audit  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 160)
    results = audit.run_all()
    lines = [r.report() for r in results]
    text = "\n".join(lines)
    print(text)

    n_issues = sum(len(r.issues) for r in results)
    summary = (
        f"\n{'=' * 72}\n"
        f"AUDIT COMPLETE: {len(results)} datasets, {n_issues} issues flagged\n"
        f"{'=' * 72}"
    )
    print(summary)

    if "--save" in sys.argv:
        out = Path(__file__).resolve().parents[1] / "reports" / "audit_report.txt"
        out.parent.mkdir(exist_ok=True)
        out.write_text(text + summary, encoding="utf-8")
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
