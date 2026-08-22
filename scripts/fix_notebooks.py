"""Repair notebook bootstrap cells.

The original cells assumed the working directory was notebooks/. VSCode often
sets it to the workspace root instead, which breaks `from src.eda import ...`.

This replaces the bootstrap with one that walks upward from cwd until it finds
the folder containing src/, so it works from either location.

    python scripts/fix_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

NEW_BOOTSTRAP = '''import sys, pathlib

# Locate the project root by walking up until we find src/.
# Works whether the kernel starts in notebooks/ or the workspace root.
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("interpreter: ", sys.executable)
print("project root:", ROOT)

import pandas as pd
from src.eda import loaders, plots, text_stats, audit

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)
plots.setup()'''


def to_source(text: str) -> list[str]:
    """Convert a string to nbformat's line-list representation."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def patch(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # the bootstrap is the cell that inserts into sys.path
        if "sys.path.insert" in src and "from src.eda import" in src:
            cell["source"] = to_source(NEW_BOOTSTRAP)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return changed


def main() -> int:
    if not NOTEBOOKS.is_dir():
        print(f"no notebooks directory at {NOTEBOOKS}")
        return 1

    files = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not files:
        print("no notebooks found")
        return 1

    for f in files:
        status = "patched" if patch(f) else "no bootstrap cell found"
        print(f"  {f.name:36s} {status}")

    print("\nBootstrap cells now locate the project root automatically.")
    print("Re-run the first cell of each notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
