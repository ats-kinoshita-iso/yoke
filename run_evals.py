#!/usr/bin/env python
"""Eval runner — cross-platform alternative to Makefile targets.

Usage:
    uv run python run_evals.py fast      # local models, ~2 min, free
    uv run python run_evals.py full      # API models, ~4 min, ~$0.10
    uv run python run_evals.py compare   # run both, show comparison
"""

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_FILES = [
    "evals/phase1_ingestion_eval.py",
    "evals/phase1_pipeline_eval.py",
    "evals/phase2_retrieval_eval.py",
    "evals/model_comparison.py",
]

RESULTS_DIR = Path("evals/results")
COMMON_FLAGS = ["-v", "-s", "--tb=short", "-p", "no:randomly"]


def _run_pytest(extra_flags: list[str], log_name: str) -> int:
    """Run pytest with the given flags, tee output to a log file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / log_name

    cmd = [sys.executable, "-m", "pytest", *EVAL_FILES, *COMMON_FLAGS, *extra_flags]
    print(f"  $ {' '.join(cmd)}\n")

    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()

    print(f"\n  Log written to {log_path}")
    return proc.returncode


def cmd_fast() -> int:
    """Run evals with local models (--all-local)."""
    print("=" * 60)
    print("  eval-fast: ollama/gemma4:e4b for generation + judging")
    print("=" * 60)
    return _run_pytest(["--all-local"], "eval-fast.log")


def cmd_full() -> int:
    """Run evals with API models."""
    print("=" * 60)
    print("  eval-full: Claude API models")
    print("=" * 60)
    return _run_pytest([], "eval-full.log")


def cmd_compare() -> int:
    """Run both, then print summary comparison."""
    print("=" * 60)
    print("  eval-compare: running both local and API evals")
    print("=" * 60)

    print("\n--- eval-fast (local) ---\n")
    rc_fast = _run_pytest(["--all-local"], "eval-fast.log")

    print("\n--- eval-full (API) ---\n")
    rc_full = _run_pytest([], "eval-full.log")

    # Summary
    print("\n" + "=" * 60)
    print("  Comparison: eval-fast (local) vs eval-full (API)")
    print("=" * 60)

    for label, log_name in [("eval-fast", "eval-fast.log"), ("eval-full", "eval-full.log")]:
        log_path = RESULTS_DIR / log_name
        print(f"\n  --- {label} ---")
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if any(kw in line for kw in ["PASSED", "FAILED", "ERROR", "Avg", "Model Comparison", "average"]):
                    print(f"  {line.strip()}")
        else:
            print("  (no results)")

    print(f"\n  Full logs: {RESULTS_DIR}/eval-fast.log, {RESULTS_DIR}/eval-full.log")
    print(f"  JSON results: {RESULTS_DIR}/*.json")

    return max(rc_fast, rc_full)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yoke eval runner",
        epilog="Examples:\n"
               "  uv run python run_evals.py fast\n"
               "  uv run python run_evals.py full\n"
               "  uv run python run_evals.py compare\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        choices=["fast", "full", "compare"],
        help="fast=local models, full=API models, compare=both",
    )
    args = parser.parse_args()

    commands = {"fast": cmd_fast, "full": cmd_full, "compare": cmd_compare}
    rc = commands[args.target]()
    sys.exit(rc)


if __name__ == "__main__":
    main()
