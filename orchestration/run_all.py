#!/usr/bin/env python3
"""
Composite eval orchestrator.

Runs the four composite axes (audit_quality, fp_calibration, production_value,
realism) for one or more auditor models. Each axis is launched as an
independent subprocess so the eval modules can manage their own resume / lock
machinery without orchestrator interference.

Per model:

* serial mode (default): subprocess.run each axis in turn
* --parallel: launch all 4 axes concurrently via asyncio.create_subprocess_exec

After all axes for a model finish, ``orchestration.analyze`` is invoked to
materialize ``<model_dir>/summary.json`` + ``<model_dir>/report.md``.

CLI:
    python -m orchestration.run_all \
        --models claude-haiku-4-5-20251001 claude-sonnet-4-5-20251022 \
        --evals audit_quality fp_calibration production_value realism \
        --output-root results --parallel

Each axis is INDEPENDENTLY resumable; the orchestrator only dispatches.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("orchestration.run_all")


# Maps axis name -> dotted import path of its eval driver.
EVAL_SCRIPTS = {
    "audit_quality": "evals.audit_quality",
    "fp_calibration": "evals.fp_calibration",
    "production_value": "evals.production_value",
    "realism": "evals.realism",
}


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_model_name(model: str) -> str:
    """Convert a model id to a filesystem-safe directory name.

    Rules:
      * leading provider prefixes ("anthropic/", "openrouter/", "openai/",
        "deepseek/", "google/", "x-ai/", "meta-llama/") are stripped
      * remaining ``/`` and ``:`` characters become ``_``
      * any character outside ``[A-Za-z0-9._-]`` becomes ``_``
      * runs of ``_`` are collapsed and the result is stripped of leading/
        trailing ``_``

    Examples:
        anthropic/claude-haiku-4-5-20251001 -> claude-haiku-4-5-20251001
        openrouter/x-ai/grok-4.1-fast       -> x-ai_grok-4.1-fast
        meta-llama/Llama-3.3-70B-Instruct   -> Llama-3.3-70B-Instruct
    """
    name = model.strip()
    for prefix in (
        "anthropic/",
        "openrouter/",
        "openai/",
        "deepseek/",
        "google/",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.replace("/", "_").replace(":", "_")
    name = _SAFE_CHARS.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_.")
    return name or "model"


def _build_axis_cmd(
    auditor_model: str,
    axis: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build the subprocess command for one (model, axis) cell."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        EVAL_SCRIPTS[axis],
        "--auditor-model",
        auditor_model,
        "--output-dir",
        str(output_dir),
    ]
    if args.max_seeds is not None:
        cmd += ["--max-seeds", str(args.max_seeds)]
    if args.rollouts is not None:
        cmd += ["--rollouts", str(args.rollouts)]
    if args.max_turns is not None:
        cmd += ["--max-turns", str(args.max_turns)]
    if args.audit_concurrent is not None:
        cmd += ["--audit-concurrent", str(args.audit_concurrent)]
    if args.judge_concurrent is not None:
        cmd += ["--judge-concurrent", str(args.judge_concurrent)]
    return cmd


def _subprocess_env() -> dict:
    """Inherit current env, but ensure REPO_ROOT is on PYTHONPATH so
    ``-m evals.<axis>`` resolves ``core``/``evals`` packages."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


# ---------------------------------------------------------------------------
# Per-axis runners
# ---------------------------------------------------------------------------


def run_axis_serial(
    auditor_model: str,
    axis: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Synchronous subprocess run for one axis. Returns return code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_axis_cmd(auditor_model, axis, output_dir, args)
    logger.info("[%s | %s] launching: %s", auditor_model, axis, " ".join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, env=_subprocess_env(), cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        logger.error(
            "[%s | %s] failed with returncode=%d", auditor_model, axis, proc.returncode
        )
    else:
        logger.info("[%s | %s] complete", auditor_model, axis)
    return proc.returncode


async def run_axis_async(
    auditor_model: str,
    axis: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Async subprocess for parallel execution within one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_axis_cmd(auditor_model, axis, output_dir, args)
    logger.info("[%s | %s] launching (async): %s", auditor_model, axis, " ".join(cmd))
    if args.dry_run:
        return 0
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=_subprocess_env(),
        cwd=str(REPO_ROOT),
    )
    rc = await proc.wait()
    if rc != 0:
        logger.error("[%s | %s] failed with returncode=%d", auditor_model, axis, rc)
    else:
        logger.info("[%s | %s] complete", auditor_model, axis)
    return rc


# ---------------------------------------------------------------------------
# Per-model dispatch
# ---------------------------------------------------------------------------


async def run_model(
    auditor_model: str,
    evals: list[str],
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    """Run all requested axes for one model, then run the analyze step.

    Returns the per-model output dir.
    """
    model_short = sanitize_model_name(auditor_model)
    model_dir = output_root / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] -> %s", auditor_model, model_dir)

    if args.parallel:
        tasks = [
            run_axis_async(auditor_model, axis, model_dir / axis, args)
            for axis in evals
        ]
        rcs = await asyncio.gather(*tasks)
    else:
        rcs = []
        for axis in evals:
            rc = run_axis_serial(auditor_model, axis, model_dir / axis, args)
            rcs.append(rc)

    failed = [a for a, rc in zip(evals, rcs) if rc != 0]
    if failed:
        logger.warning("[%s] axes failed: %s", auditor_model, failed)

    # Run analyze regardless — partial summaries are still useful.
    if not args.dry_run:
        try:
            from orchestration.analyze import analyze_one_model

            analyze_one_model(model_dir)
            logger.info("[%s] wrote summary.json + report.md to %s", auditor_model, model_dir)
        except Exception as e:  # pragma: no cover
            logger.exception("[%s] analyze failed: %s", auditor_model, e)
    return model_dir


async def _amain(args: argparse.Namespace) -> int:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    invalid = [a for a in args.evals if a not in EVAL_SCRIPTS]
    if invalid:
        logger.error("Unknown axis names: %s. Valid: %s", invalid, list(EVAL_SCRIPTS))
        return 2

    overall_rc = 0
    for model in args.models:
        try:
            await run_model(model, args.evals, output_root, args)
        except Exception as e:  # pragma: no cover
            logger.exception("[%s] orchestration failed: %s", model, e)
            overall_rc = 1
    return overall_rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the composite eval suite (4 axes) for one or more auditor models."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="Auditor model IDs (one or more)."
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=list(EVAL_SCRIPTS),
        choices=list(EVAL_SCRIPTS),
        help="Subset of composite axes to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Root output dir. Each model gets <output-root>/<model_short>/.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all axes for one model in parallel via asyncio subprocesses.",
    )
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--audit-concurrent", type=int, default=None)
    parser.add_argument("--judge-concurrent", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, do not execute.",
    )
    args = parser.parse_args()

    rc = asyncio.run(_amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
