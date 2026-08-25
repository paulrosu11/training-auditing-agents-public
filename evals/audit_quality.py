#!/usr/bin/env python3
"""
audit_quality eval driver — paper composite axis: investigation quality.

Measures in-distribution investigation quality: how well an auditor surfaces
a *known* hidden behavior in a system-prompted model organism. This is the
"how well does the auditor investigate" axis — detection rate is secondary,
investigation rubric scores are primary.

Paper-canonical config (overrides paper text where it disagrees with the
paper-time code):

* target_model:    ``deepseek/deepseek-chat-v3.1`` (system-prompted MO via
                   OpenRouter)
* judge_model:     ``claude-opus-4-6`` (paper-canonical FINAL — verified
                   from paper-time ``paper §A.4 config``)
* num_quirks:      6 held-out (``core.quirks.EVAL_QUIRKS``)
* seeds_per_quirk: 15 hand-tailored seeds per quirk (consolidated at
                   ``data/seeds_audit_quality_15_per_quirk.json``)
* rollouts:        3
* max_turns:       30
* anonymize_target: True   (auditor sees "unspecified target model")
* enable_prefill:  False
* judge thinking:  False

Usage:
    # Full run
    python -m evals.audit_quality \
        --auditor-model claude-haiku-4-5-20251001 \
        --output-dir results/aq_baseline

    # Smoke test
    python -m evals.audit_quality \
        --auditor-model claude-haiku-4-5-20251001 \
        --output-dir results/aq_smoke \
        --quirks secret_loyalty --max-seeds 2 --rollouts 1

The driver writes:

* ``<output_dir>/results.jsonl``        — one row per audit (see schema below)
* ``<output_dir>/judged_results.jsonl`` — one row per judged audit
* ``<output_dir>/transcripts/<q>_s<NNN>_r<NN>.json`` — full per-audit state
* ``<output_dir>/config.json``          — captured CLI / defaults
* ``<output_dir>/summary.json``         — paper-canonical audit_quality summary

Resume is automatic: re-running with the same ``--output-dir`` skips audits
that are already in ``results.jsonl`` (with completed=True, num_turns>0) and
re-judges any transcripts in the dir that are not yet in
``judged_results.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from core.api import APIClients
from core.aggregate import compute_audit_quality_summary, load_results
from core.judges import DEFAULT_EVAL_JUDGE_MODEL
from core.petri import get_auditor_tools_async
from core.quirks import EVAL_QUIRKS, render_quirk_system_prompt

from evals._runner import (
    acquire_lock,
    aggregate_results,
    append_result,
    judge_audit_quality_single,
    load_completed,
    load_judged,
    release_lock,
    run_single_audit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("evals.audit_quality")


# ---------------------------------------------------------------------------
# Paper-canonical defaults (single source of truth for this axis)
# ---------------------------------------------------------------------------

AUDIT_QUALITY_DEFAULTS = {
    "target_model": "deepseek/deepseek-chat-v3.1",  # OpenRouter MO target
    "rollouts": 3,  # paper-canonical (appendix.tex:467 — "6 × 15 × 3 rollouts = 270 audits")
    "max_turns": 30,
    "audit_concurrent": 100,
    "judge_concurrent": 50,
    "anonymize_target": True,
    "enable_prefill": False,
    "use_thinking_for_judge": False,
}

DEFAULT_JUDGE_MODEL = DEFAULT_EVAL_JUDGE_MODEL  # claude-opus-4-6 (paper-canonical FINAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
TAILORED_SEEDS_PATH = (
    REPO_ROOT / "data" / "seeds_audit_quality_15_per_quirk.json"
)


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def load_tailored_seeds(quirk_name: str, max_seeds: int | None = None) -> list[str]:
    """Load 15 hand-tailored seeds for a quirk from the consolidated file.

    The consolidated file has shape ``{quirk: [seed_str, ...]}`` and was
    ported verbatim from ``paper §A.4 tailored seed file``.
    """
    if not TAILORED_SEEDS_PATH.exists():
        raise FileNotFoundError(
            f"Tailored seed file not found at {TAILORED_SEEDS_PATH}. "
            "Re-port from auditor_eval_suite/seeds/generated/."
        )
    data = json.loads(TAILORED_SEEDS_PATH.read_text())
    if quirk_name not in data:
        raise KeyError(
            f"Quirk '{quirk_name}' has no tailored seeds in {TAILORED_SEEDS_PATH}. "
            f"Available: {sorted(data.keys())}"
        )
    seeds = data[quirk_name]
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    return seeds


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="audit_quality eval: in-distribution investigation quality."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--auditor-model",
        required=True,
        help="Auditor model ID (bare Anthropic, anthropic/, openrouter/, openai/, etc.)",
    )
    parser.add_argument(
        "--target-model",
        default=AUDIT_QUALITY_DEFAULTS["target_model"],
        help=f"Target model (default: {AUDIT_QUALITY_DEFAULTS['target_model']})",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--quirks",
        nargs="+",
        default=None,
        help="Subset of EVAL_QUIRKS to run (default: all 6)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=AUDIT_QUALITY_DEFAULTS["rollouts"],
        help="Rollouts per (quirk, seed) (default: 3)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit seeds per quirk (default: 15 = full set)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=AUDIT_QUALITY_DEFAULTS["max_turns"],
        help="Max turns per audit (default: 30)",
    )
    parser.add_argument(
        "--audit-concurrent",
        type=int,
        default=AUDIT_QUALITY_DEFAULTS["audit_concurrent"],
        help="Max concurrent audits (default: 100)",
    )
    parser.add_argument(
        "--judge-concurrent",
        type=int,
        default=AUDIT_QUALITY_DEFAULTS["judge_concurrent"],
        help="Max concurrent judge calls (default: 50)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Audit only (skip judging — useful for separate judging pass)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    quirks = args.quirks or sorted(EVAL_QUIRKS.keys())
    for q in quirks:
        if q not in EVAL_QUIRKS:
            print(
                f"Error: Unknown quirk '{q}'. Available: {sorted(EVAL_QUIRKS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lockfile = acquire_lock(args.output_dir)

    results_file = args.output_dir / "results.jsonl"
    judged_file = args.output_dir / "judged_results.jsonl"
    transcript_dir = args.output_dir / "transcripts"

    config = {
        "axis": "audit_quality",
        "description": "In-distribution investigation quality",
        "auditor_model": args.auditor_model,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "quirks": quirks,
        "rollouts": args.rollouts,
        "max_turns": args.max_turns,
        "max_seeds": args.max_seeds,
        "audit_concurrent": args.audit_concurrent,
        "judge_concurrent": args.judge_concurrent,
        "anonymize_target": AUDIT_QUALITY_DEFAULTS["anonymize_target"],
        "enable_prefill": AUDIT_QUALITY_DEFAULTS["enable_prefill"],
        "use_thinking_for_judge": AUDIT_QUALITY_DEFAULTS["use_thinking_for_judge"],
        "skip_judge": args.skip_judge,
        "timestamp": datetime.now().isoformat(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Build task list (quirk, seed_idx, rollout) → seed_instruction + quirk_prompt.
    tasks_to_run: list[dict] = []
    for quirk_name in quirks:
        seeds = load_tailored_seeds(quirk_name, args.max_seeds)
        quirk_prompt = render_quirk_system_prompt(quirk_name)
        for seed_idx, seed_instruction in enumerate(seeds):
            for rollout in range(args.rollouts):
                tasks_to_run.append(
                    {
                        "quirk_name": quirk_name,
                        "quirk_prompt": quirk_prompt,
                        "seed_instruction": seed_instruction,
                        "seed_idx": seed_idx,
                        "rollout": rollout,
                    }
                )

    # Resume: skip already-completed audits.
    completed_audits = load_completed(results_file)
    tasks_to_run = [
        t
        for t in tasks_to_run
        if (t["quirk_name"], t["seed_idx"], t["rollout"]) not in completed_audits
    ]

    total = len(tasks_to_run) + len(completed_audits)
    logger.info(
        "audit_quality: %d total audits, %d completed, %d remaining",
        total, len(completed_audits), len(tasks_to_run),
    )
    logger.info(
        "Auditor=%s | Target=%s | Judge=%s",
        args.auditor_model, args.target_model, args.judge_model,
    )
    logger.info(
        "Concurrency: audit=%d, judge=%d", args.audit_concurrent, args.judge_concurrent,
    )
    logger.info("Quirks: %s", quirks)

    already_judged = load_judged(judged_file) if not args.skip_judge else set()
    unjudged_keys = (
        completed_audits - already_judged if not args.skip_judge else set()
    )

    if not tasks_to_run and not unjudged_keys:
        logger.info("All audits complete and all judged — emitting summary.")
        _write_summary(args.output_dir, results_file, judged_file)
        release_lock(lockfile)
        return

    if not tasks_to_run and unjudged_keys:
        logger.info(
            "All audits complete; %d transcripts still need judging.",
            len(unjudged_keys),
        )

    # Build clients + tools (load tools once).
    is_openai_target = args.target_model.startswith("openai/") or args.target_model.startswith(
        ("gpt-", "o3", "o4")
    )
    clients = APIClients(
        auditor_concurrent=args.audit_concurrent,
        target_concurrent=args.audit_concurrent,
        judge_concurrent=args.judge_concurrent,
        use_openai_for_target=is_openai_target,
    )
    tools = await get_auditor_tools_async()

    audit_semaphore = asyncio.Semaphore(args.audit_concurrent)
    judge_semaphore = asyncio.Semaphore(args.judge_concurrent)

    start_time = time.time()
    audits_done = len(completed_audits)
    judges_done = len(already_judged)
    judges_errors = 0
    audits_total = total

    results_lock = asyncio.Lock()
    judged_lock = asyncio.Lock()

    async def judge_transcript(transcript_path: Path, quirk_name: str) -> None:
        """Judge a single transcript file. Skips already-judged keys."""
        nonlocal judges_done, judges_errors
        meta_key: tuple | None = None
        try:
            t = json.loads(transcript_path.read_text())
            meta = t["metadata"]
            meta_key = (meta["quirk"], meta["seed_idx"], meta["rollout"])
            if meta_key in already_judged:
                return

            async with judge_semaphore:
                scores = await judge_audit_quality_single(
                    transcript=t,
                    quirk=quirk_name,
                    ground_truth=EVAL_QUIRKS[quirk_name]["ground_truth"],
                    judge_model=args.judge_model,
                    clients=clients,
                    use_thinking=AUDIT_QUALITY_DEFAULTS["use_thinking_for_judge"],
                )

            score_keys = [k for k in scores if not k.startswith("_")]
            if not score_keys:
                judges_errors += 1
                logger.warning(
                    "Judge produced no scores for %s s%d r%d — will retry on next run",
                    quirk_name, meta["seed_idx"], meta["rollout"],
                )
                return

            row = {
                "axis": "audit_quality",
                "quirk": quirk_name,
                "seed_idx": meta["seed_idx"],
                "rollout": meta["rollout"],
                **{k: v for k, v in scores.items() if not k.startswith("_")},
            }
            if scores.get("_judge_error"):
                row["_judge_error"] = scores["_judge_error"]

            async with judged_lock:
                with open(judged_file, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

            already_judged.add(meta_key)
            judges_done += 1
            if judges_done % 10 == 0:
                logger.info("  Judged: %d transcripts", judges_done)
        except Exception as e:
            judges_errors += 1
            logger.error(
                "Judge failed for %s: %s", meta_key or transcript_path.name, e,
            )

    async def run_audit_and_judge(task: dict) -> None:
        """Run one audit, persist transcript + JSONL row, then queue judging."""
        nonlocal audits_done

        result = await run_single_audit(
            seed_instruction=task["seed_instruction"],
            auditor_model=args.auditor_model,
            target_model=args.target_model,
            max_turns=args.max_turns,
            clients=clients,
            session_semaphore=audit_semaphore,
            quirk_name=task["quirk_name"],
            quirk_prompt=task["quirk_prompt"],
            seed_idx=task["seed_idx"],
            rollout=task["rollout"],
            anonymize_target=AUDIT_QUALITY_DEFAULTS["anonymize_target"],
            enable_prefill=AUDIT_QUALITY_DEFAULTS["enable_prefill"],
            save_transcript=True,
            transcript_dir=transcript_dir,
            tools=tools,
        )
        result["axis"] = "audit_quality"

        async with results_lock:
            append_result(results_file, result)

        audits_done += 1
        if audits_done % 10 == 0:
            elapsed = time.time() - start_time
            rate = audits_done / elapsed * 3600 if elapsed > 0 else 0.0
            logger.info(
                "Audits: %d/%d (%.0f%%) — %.0f/hr",
                audits_done, audits_total, audits_done / audits_total * 100, rate,
            )

        if (
            not args.skip_judge
            and result.get("_transcript_abs_path")
            and not result.get("error")
        ):
            asyncio.create_task(
                judge_transcript(Path(result["_transcript_abs_path"]), task["quirk_name"])
            )

    # Re-judge orphaned transcripts from prior runs (resume recovery).
    if not args.skip_judge and unjudged_keys:
        unjudged_pairs: list[tuple[Path, str]] = []
        for key in unjudged_keys:
            qn, seed_idx, rollout = key
            tp = transcript_dir / f"{qn}_s{seed_idx:03d}_r{rollout:02d}.json"
            if tp.exists():
                unjudged_pairs.append((tp, qn))
        if unjudged_pairs:
            logger.info(
                "Re-judging %d orphaned transcripts from prior runs",
                len(unjudged_pairs),
            )
            for tp, qn in unjudged_pairs:
                asyncio.create_task(judge_transcript(tp, qn))

    # Drive everything to completion.
    gather_results = await asyncio.gather(
        *[run_audit_and_judge(t) for t in tasks_to_run],
        return_exceptions=True,
    )
    for i, r in enumerate(gather_results):
        if isinstance(r, Exception):
            logger.error("Task %d crashed: %s", i, r)

    if not args.skip_judge:
        pending = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            logger.info("Waiting for %d remaining judge tasks...", len(pending))
            await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.time() - start_time
    logger.info(
        "audit_quality complete: %d audits + %d judged in %.1f min",
        audits_done, judges_done, elapsed / 60,
    )
    if judges_errors:
        logger.warning("Judge errors: %d (will retry on next run)", judges_errors)

    _write_summary(args.output_dir, results_file, judged_file)
    logger.info("Results: %s", results_file)
    if not args.skip_judge:
        logger.info("Judged:  %s", judged_file)
    logger.info("Summary: %s", args.output_dir / "summary.json")

    release_lock(lockfile)


def _write_summary(output_dir: Path, results_file: Path, judged_file: Path) -> None:
    """Write a paper-canonical summary.json in the output dir.

    Schema (top level)::

        {
          "axis": "audit_quality",
          "audits":  <aggregate_results(results_file)>,    # quick completion stats
          "judged":  <compute_audit_quality_summary(judged)>  # paper audit_quality score
        }

    The ``judged`` block follows the schema documented on
    :func:`core.aggregate.compute_audit_quality_summary`:
      - ``overall.bucket_score`` — audit_quality bucket score (0-100)
      - ``overall.scores`` — per-dimension means + per-seed CIs
      - ``overall.binary`` — quirk_exhibited / auditor_identified rates
      - ``per_quirk`` — same shape, keyed by quirk name
    """
    audits_summary = aggregate_results(results_file)
    summary: dict = {
        "axis": "audit_quality",
        "audits": audits_summary,
    }

    if judged_file.exists():
        try:
            judged_rows = load_results(judged_file)
            summary["judged"] = compute_audit_quality_summary(judged_rows)
        except Exception as e:
            logger.warning("Failed to compute judged summary: %s", e)
            summary["judged"] = {"error": str(e)}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
