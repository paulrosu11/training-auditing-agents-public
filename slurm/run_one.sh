#!/bin/bash
#SBATCH --job-name=eval_one
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_one_%j.out
#SBATCH --error=logs/eval_one_%j.err

# =============================================================================
# Single (model x axis) eval cell.
#
# Required env:
#   AUDITOR_MODEL   e.g. claude-haiku-4-5-20251001
#   EVAL_NAME       one of:
#                     audit_quality | fp_calibration |
#                     production_value | realism |
#                     crux | persona | susceptibility
#   OUTPUT_DIR      directory for this cell's output
#
# Submit:
#   sbatch --export=ALL,AUDITOR_MODEL=...,EVAL_NAME=audit_quality,\
#                  OUTPUT_DIR=results/haiku/audit_quality \
#       slurm/run_one.sh
# =============================================================================

set -euo pipefail

: "${AUDITOR_MODEL:?Set AUDITOR_MODEL}"
: "${EVAL_NAME:?Set EVAL_NAME}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

case "$EVAL_NAME" in
    audit_quality|fp_calibration|production_value|realism|crux|persona|susceptibility) ;;
    *)
        echo "ERROR: unknown EVAL_NAME='$EVAL_NAME'" >&2
        echo "       valid: audit_quality fp_calibration production_value realism crux persona susceptibility" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs "$OUTPUT_DIR"
cd "$REPO_ROOT"

echo "=========================================="
echo "Eval cell"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Auditor:      $AUDITOR_MODEL"
echo "Eval:         $EVAL_NAME"
echo "Output:       $OUTPUT_DIR"
echo "Started:      $(date)"
echo "=========================================="

python -m "evals.${EVAL_NAME}" \
    --auditor-model "$AUDITOR_MODEL" \
    --output-dir "$OUTPUT_DIR"

echo
echo "Eval cell complete: ${AUDITOR_MODEL} / ${EVAL_NAME}"
echo "Output: $OUTPUT_DIR"
