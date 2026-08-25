#!/bin/bash
#SBATCH --job-name=composite_eval
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/composite_%j.out
#SBATCH --error=logs/composite_%j.err

# =============================================================================
# Composite eval (CPU-only; API-bound, no GPU required).
#
# Runs the four composite axes (audit_quality, fp_calibration, production_value,
# realism) in parallel for a single auditor model, then aggregates summaries.
#
# Required env:
#   AUDITOR_MODEL   e.g. claude-haiku-4-5-20251001
#
# Optional env:
#   OUTPUT_ROOT     default: ./results
#   MAX_SEEDS       (cap seeds per quirk for smoke runs)
#   ROLLOUTS        (override per-(quirk,seed) rollouts)
#
# Submit:
#   sbatch --export=ALL,AUDITOR_MODEL=claude-haiku-4-5-20251001 \
#       slurm/composite_eval.sh
# =============================================================================

set -euo pipefail

: "${AUDITOR_MODEL:?Set AUDITOR_MODEL (e.g. claude-haiku-4-5-20251001)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs "$OUTPUT_ROOT"
cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ -n "${MAX_SEEDS:-}" ]]; then
    EXTRA_ARGS+=(--max-seeds "$MAX_SEEDS")
fi
if [[ -n "${ROLLOUTS:-}" ]]; then
    EXTRA_ARGS+=(--rollouts "$ROLLOUTS")
fi

echo "=========================================="
echo "Composite eval"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Auditor:      $AUDITOR_MODEL"
echo "Output root:  $OUTPUT_ROOT"
echo "Started:      $(date)"
echo "=========================================="

python -m orchestration.run_all \
    --models "$AUDITOR_MODEL" \
    --evals audit_quality fp_calibration production_value realism \
    --output-root "$OUTPUT_ROOT" \
    --parallel \
    "${EXTRA_ARGS[@]}"

echo
echo "Composite eval complete: $AUDITOR_MODEL"
echo "Results: $OUTPUT_ROOT"
