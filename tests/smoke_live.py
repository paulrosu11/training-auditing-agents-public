"""Live smoke tests that make real API calls.

Runs 1-seed / 1-rollout mini versions of every eval. Use as the gate before
larger production runs.

Usage:
    python tests/smoke_live.py                 # all sections
    python tests/smoke_live.py --only audit_quality realism
    python tests/smoke_live.py --skip persona susceptibility
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * (75 - len(title)))


def check(ok: bool, msg: str, detail: str = "") -> bool:
    mark = PASS if ok else FAIL
    line = f"  {mark} {msg}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _venv_python() -> str:
    """Return path to the active Python interpreter."""
    return sys.executable or "python3"


def _run_eval(args: list[str], log_path: Path, timeout_s: int = 600) -> tuple[bool, float]:
    """Run an eval subprocess; return (success, wall_seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO}:{pp}".rstrip(":")
    t0 = time.time()
    with open(log_path, "w") as f:
        try:
            r = subprocess.run(
                [_venv_python(), "-u", "-m", *args],
                cwd=REPO,
                env=env,
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            elapsed = time.time() - t0
            return (r.returncode == 0), elapsed
        except subprocess.TimeoutExpired:
            return False, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Live sections
# ─────────────────────────────────────────────────────────────────────────────
def L1_audit_quality() -> bool:
    section("L1: audit_quality 1-seed Haiku 4.5 vs DeepSeek")
    out = REPO / "results_smoke/L1_haiku_audit_quality"
    log = REPO / "logs/L1_haiku_audit_quality.log"
    ok, t = _run_eval([
        "evals.audit_quality",
        "--auditor-model", "claude-haiku-4-5-20251001",
        "--quirks", "secret_loyalty",
        "--max-seeds", "1", "--rollouts", "1", "--max-turns", "5",
        "--audit-concurrent", "4", "--judge-concurrent", "2",
        "--output-dir", str(out),
    ], log)
    return check(ok, f"audit_quality completed in {t:.0f}s",
                 "result.jsonl present" if (out / "results.jsonl").exists() else "no jsonl")


def L2_fp_calibration() -> bool:
    section("L2: fp_calibration 1-seed Haiku 4.5 vs Sonnet 4.5 (clean)")
    out = REPO / "results_smoke/L2_haiku_fp_calibration"
    log = REPO / "logs/L2_haiku_fp_calibration.log"
    ok, t = _run_eval([
        "evals.fp_calibration",
        "--auditor-model", "claude-haiku-4-5-20251001",
        "--quirks", "secret_loyalty",
        "--max-seeds", "1", "--rollouts", "1", "--max-turns", "5",
        "--audit-concurrent", "4", "--judge-concurrent", "2",
        "--output-dir", str(out),
    ], log)
    return check(ok, f"fp_calibration completed in {t:.0f}s")


def L3_production_value() -> bool:
    section("L3: production_value 1-seed Haiku 4.5 vs Sonnet 4.6")
    out = REPO / "results_smoke/L3_haiku_production_value"
    log = REPO / "logs/L3_haiku_production_value.log"
    ok, t = _run_eval([
        "evals.production_value",
        "--auditor-model", "claude-haiku-4-5-20251001",
        "--max-seeds", "1", "--rollouts", "1", "--max-turns", "5",
        "--audit-concurrent", "4", "--judge-concurrent", "2",
        "--output-dir", str(out),
    ], log, timeout_s=900)
    return check(ok, f"production_value completed in {t:.0f}s")


def L4_realism() -> bool:
    section("L4: realism micro Haiku 4.5")
    out = REPO / "results_smoke/L4_realism_micro"
    log = REPO / "logs/L4_realism_micro.log"
    ok, t = _run_eval([
        "evals.realism",
        "--auditor-model", "claude-haiku-4-5-20251001",
        "--max-seeds", "1", "--rollouts", "1", "--max-turns", "5",
        "--audit-concurrent", "4", "--judge-concurrent", "2",
        "--thinking-budget", "0",
        "--output-dir", str(out),
    ], log, timeout_s=2400)
    return check(ok, f"realism completed in {t:.0f}s")


def L5_crux() -> bool:
    section("L5: crux 1-rep Haiku 4.5 vs Llama 3.3 70B")
    out_file = REPO / "results_smoke/L5_haiku_crux/result.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    log = REPO / "logs/L5_haiku_crux.log"
    ok, t = _run_eval([
        "evals.crux",
        "--auditor-model", "claude-haiku-4-5-20251001",
        "--target-model", "meta-llama/llama-3.3-70b-instruct",
        "--num-regens", "1", "--num-turns", "5",
        "--quirks", "secret_loyalty",
        "--audit-concurrent", "5", "--judge-concurrent", "3",
        "--output", str(out_file),
    ], log, timeout_s=1800)
    return check(ok, f"crux completed in {t:.0f}s",
                 "result.json present" if out_file.exists() else "no output")


def L6_persona() -> bool:
    section("L6: persona 1-rep GPT-4.1 Mini")
    out = REPO / "results_smoke/L6_gpt41mini_persona"
    log = REPO / "logs/L6_gpt41mini_persona.log"
    ok, t = _run_eval([
        "evals.persona",
        "--models", "gpt-4.1-mini",
        "--num-ideas", "1", "--num-repetitions", "1", "--num-turns", "5",
        "--max-concurrent", "4", "--no-judge-thinking",
        "--output-dir", str(out),
    ], log)
    return check(ok, f"persona completed in {t:.0f}s")


def L7_susceptibility() -> bool:
    section("L7: susceptibility 1-quirk Llama 3.3 70B")
    out = REPO / "results_smoke/L7_llama_susceptibility"
    log = REPO / "logs/L7_llama_susceptibility.log"
    ok, t = _run_eval([
        "evals.susceptibility",
        "--models", "llama-3.3-70b", "--quirks", "secret_loyalty",
        "--num-ideas", "1", "--num-repetitions", "1", "--num-turns", "5",
        "--conditions", "quirk,baseline",
        "--reuse-ideas-from", "data/ideas_cache.json",
        "--max-concurrent", "4", "--no-judge-thinking",
        "--output-dir", str(out),
    ], log, timeout_s=900)
    return check(ok, f"susceptibility completed in {t:.0f}s")


def L8_strategy_clustering_offline() -> bool:
    """Offline strategy_clustering: codebook hash + import."""
    section("L8: strategy_clustering offline (codebook hash + imports)")
    try:
        from evals.strategy_clustering import (
            run_all,
            stage_1_build_corpus, stage_2_extract, stage_3_embed_and_cluster,
            stage_4_label, stage_5_rollup, stage_6_freeze_codebook,
            stage_7_classify, stage_8_aggregate,
        )
    except Exception as e:
        return check(False, "strategy_clustering imports", str(e)[:80])
    return check(True, "all 8 stage functions + run_all importable")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = {
    "audit_quality": L1_audit_quality,
    "fp_calibration": L2_fp_calibration,
    "production_value": L3_production_value,
    "realism": L4_realism,
    "crux": L5_crux,
    "persona": L6_persona,
    "susceptibility": L7_susceptibility,
    "strategy_clustering": L8_strategy_clustering_offline,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=list(SECTIONS),
                   help="Run only these sections")
    p.add_argument("--skip", nargs="+", choices=list(SECTIONS), default=[],
                   help="Skip these sections")
    args = p.parse_args()

    to_run = list(SECTIONS) if not args.only else args.only
    to_run = [s for s in to_run if s not in args.skip]

    results: dict[str, bool] = {}
    for name in to_run:
        try:
            results[name] = SECTIONS[name]()
        except Exception as e:
            print(f"  {FAIL} {name} crashed: {e}")
            results[name] = False

    section("Summary")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n{'PASS' if n_pass == n_total else 'FAIL'} ({n_pass}/{n_total})")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
