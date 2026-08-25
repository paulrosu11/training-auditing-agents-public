#!/usr/bin/env python3
"""
Read a single auditor's per-axis result files and produce ``summary.json`` and
``report.md``.

Per axis input layout (produced by the eval drivers):

    <model_dir>/audit_quality/judged_results.jsonl
    <model_dir>/fp_calibration/judged_results.jsonl
    <model_dir>/production_value/judged_results.jsonl
    <model_dir>/realism/niah_results.json

Outputs:

    <output_dir>/summary.json   # per-axis bucket scores + 4-axis composite
    <output_dir>/report.md      # markdown report (per-axis tables + composite)

CLI:
    python -m orchestration.analyze --baseline-dir results/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from core.aggregate import (
    EVAL_AXES,
    compute_audit_quality_summary,
    compute_composite_score,
    compute_fp_calibration_summary,
    compute_production_value_summary,
    compute_realism_summary,
    generate_summary_report,
    load_results,
)

logger = logging.getLogger("orchestration.analyze")


_AXIS_TO_COMPUTE = {
    "audit_quality": compute_audit_quality_summary,
    "fp_calibration": compute_fp_calibration_summary,
    "production_value": compute_production_value_summary,
}


def _load_axis_summary(model_dir: Path, axis: str) -> dict | None:
    axis_dir = model_dir / axis
    if axis == "realism":
        result_path = axis_dir / "niah_results.json"
        if not result_path.exists():
            logger.warning("[%s] missing %s", axis, result_path)
            return None
        try:
            niah = json.loads(result_path.read_text())
        except json.JSONDecodeError as e:
            logger.error("[%s] JSON decode error in %s: %s", axis, result_path, e)
            return None
        return compute_realism_summary(niah)

    jsonl_path = axis_dir / "judged_results.jsonl"
    if not jsonl_path.exists():
        logger.warning("[%s] missing %s", axis, jsonl_path)
        return None
    rows = load_results(jsonl_path)
    if not rows:
        logger.warning("[%s] empty file %s", axis, jsonl_path)
        return None
    fn = _AXIS_TO_COMPUTE[axis]
    return fn(rows)


def analyze_one_model(model_dir: Path, output_dir: Path | None = None) -> Path:
    """Read a single auditor's eval outputs, compute summaries, write artifacts.

    Returns the directory where ``summary.json`` and ``report.md`` were written.
    """
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_axis: dict[str, dict] = {}
    for axis in EVAL_AXES:
        s = _load_axis_summary(model_dir, axis)
        if s is not None:
            per_axis[axis] = s

    composite = compute_composite_score(per_axis)
    summary = {
        "model_dir": str(model_dir),
        "axes_present": sorted(per_axis.keys()),
        "axes_missing": sorted(set(EVAL_AXES) - set(per_axis.keys())),
        "bucket_scores": {
            a: per_axis[a].get("overall", {}).get("bucket_score") for a in per_axis
        },
        "composite_score": composite,
        "per_axis": per_axis,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    generate_summary_report(per_axis, output_dir / "report.md")

    logger.info(
        "wrote %s and %s (composite=%s)",
        output_dir / "summary.json",
        output_dir / "report.md",
        composite,
    )
    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Aggregate per-axis eval outputs for a single auditor and emit summary.json + report.md."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Per-model output dir (contains audit_quality/, fp_calibration/, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write summary.json + report.md (default: --baseline-dir/analysis/).",
    )
    args = parser.parse_args()

    out = args.output_dir or (args.baseline_dir / "analysis")
    if not args.baseline_dir.exists():
        print(f"Error: {args.baseline_dir} does not exist", file=sys.stderr)
        sys.exit(2)
    analyze_one_model(args.baseline_dir, out)


if __name__ == "__main__":
    main()
