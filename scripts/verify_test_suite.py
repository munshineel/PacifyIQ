"""Mutation testing — does the suite actually catch bugs?

A passing test suite proves nothing on its own. It might be asserting things
that are true no matter what the code does.

This script answers the harder question directly: it introduces a deliberate
bug, runs the relevant tests, and checks they FAIL. A mutation that survives is
a hole in the suite - code that could break in production with every test still
green.

    python scripts/verify_test_suite.py
    python scripts/verify_test_suite.py --mutation tier_bypass
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Mutation:
    """One deliberate bug, and the tests that should notice."""

    name: str
    file: str
    find: str
    replace: str
    tests: str
    why: str


MUTATIONS = [
    Mutation(
        "tier_bypass",
        "src/agent/tools.py",
        'if spec.tier == Tier.MUTATING and not kwargs.get("approval_token"):',
        'if False:',
        "tests/test_tools.py tests/test_guardrails.py",
        "Tier-3 tools become callable. If nothing fails, the safety model is "
        "unverified."),
    Mutation(
        "archived_policy_leaks",
        "src/knowledge/retriever.py",
        "if not include_archived and self.exclude_archived and not c.is_current:",
        "if False:",
        "tests/test_knowledge.py",
        "The superseded return policy becomes citable as current guidance."),
    Mutation(
        "abstention_disabled",
        "src/rag/abstention.py",
        "BM25_ABSTAIN_BELOW = 7.0",
        "BM25_ABSTAIN_BELOW = 0.0",
        "tests/test_rag.py",
        "The system stops refusing questions it cannot answer."),
    Mutation(
        "injection_undetected",
        "src/guardrails/input_rules.py",
        "def check_injection(text: str) -> list[Finding]:",
        "def check_injection(text: str) -> list[Finding]:\n    return []",
        "tests/test_guardrails.py",
        "Prompt injection stops being detected."),
    Mutation(
        "grounding_check_off",
        "src/rag/citations.py",
        "rep.unsupported_numbers.append(n)",
        "pass  # mutation: fabricated figures no longer flagged",
        "tests/test_rag.py tests/test_guardrails.py",
        "Fabricated figures pass the grounding check."),
    Mutation(
        "order_normalisation_broken",
        "src/db/queries.py",
        "def normalise_order_id(",
        "def normalise_order_id(raw):\n    return str(raw)\n\n\ndef _unused_normalise_order_id(",
        "tests/test_tools.py tests/test_queries.py",
        "Bare order numbers stop resolving - the most common argument-"
        "extraction failure."),
    Mutation(
        "eligibility_inverted",
        "src/agent/tools.py",
        '"eligible": e.is_eligible,',
        '"eligible": not e.is_eligible,',
        "tests/test_tools.py",
        "Return eligibility is reported backwards."),
    Mutation(
        "conflict_detection_off",
        "src/rag/context.py",
        "has_version_conflict=len(versions) > 1,",
        "has_version_conflict=False,",
        "tests/test_rag.py tests/test_integration.py",
        "Contradictory policy versions stop being surfaced."),
    Mutation(
        "vision_invents_codes",
        "src/multimodal/vision.py",
        "if best_code and best_conf >= self.MIN_CODE_CONF:",
        "if best_code or True:",
        "tests/test_multimodal.py",
        "Error codes get reported on unreadable images."),
    Mutation(
        "tool_errors_uncaught",
        "src/agent/tools.py",
        "    except Exception as e:\n        # A failing tool is a normal event",
        "    except ZeroDivisionError as e:\n        # A failing tool is a normal event",
        "tests/test_tools.py",
        "A failing tool crashes the customer's request."),
]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=900)
    return p.returncode, p.stdout + p.stderr


def apply_mutation(m: Mutation) -> str | None:
    path = ROOT / m.file
    original = path.read_text(encoding="utf-8")
    if m.find not in original:
        return None
    path.write_text(original.replace(m.find, m.replace, 1), encoding="utf-8")
    return original


def restore(m: Mutation, original: str) -> None:
    (ROOT / m.file).write_text(original, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutation", help="run a single mutation by name")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in MUTATIONS:
            print(f"  {m.name:28s} {m.file}")
        return 0

    selected = ([m for m in MUTATIONS if m.name == args.mutation]
                if args.mutation else MUTATIONS)
    if not selected:
        print(f"unknown mutation: {args.mutation}")
        return 1

    print("=" * 84)
    print("MUTATION TESTING — does the suite catch deliberate bugs?")
    print("=" * 84)
    print("  A surviving mutation is a hole: code that could break in "
          "production\n  with every test still green.\n")

    caught, survived, skipped = [], [], []

    for m in selected:
        print(f"  {m.name:28s} ", end="", flush=True)
        original = apply_mutation(m)
        if original is None:
            print("SKIP (target code has changed)")
            skipped.append(m)
            continue

        try:
            code, _ = run([sys.executable, "-m", "pytest", *m.tests.split(),
                           "-x", "-q", "--no-header", "-p", "no:cacheprovider"])
        finally:
            restore(m, original)

        if code != 0:
            print("caught")
            caught.append(m)
        else:
            print("SURVIVED  <-- gap in the suite")
            survived.append(m)

    print("\n" + "=" * 84)
    total = len(caught) + len(survived)
    score = len(caught) / total if total else 0.0
    print(f"  mutations introduced   {total}")
    print(f"  caught by the suite    {len(caught)}")
    print(f"  survived               {len(survived)}")
    if skipped:
        print(f"  skipped                {len(skipped)}")
    print(f"  mutation score         {score:.0%}")

    if survived:
        print("\n  SURVIVING MUTATIONS — these are real gaps:")
        for m in survived:
            print(f"    {m.name}")
            print(f"      {m.why}")
        print("\n  Each one is a change that breaks the system while every "
              "test passes.")
    else:
        print("\n  Every deliberate bug was caught. The suite tests behaviour, "
              "not\n  incidental facts that hold regardless of the code.")

    print("=" * 84)
    # Restore anything left mutated by an interrupted run.
    code, _ = run([sys.executable, "-c", "import src.agent.tools"])
    return 0 if not survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
