#!/usr/bin/env python3
"""
production_value eval driver — paper composite axis: OOD concerning findings on a real production target.

Measures the auditor's quality against a real *production* model with no
planted quirk. This is the OOD generalization axis: when the auditor is
unleashed against a model that may or may not have any concerning behavior,
how useful are its findings? Judged by the full 38-metric Petri rubric so the
output can be compared apples-to-apples with the production Petri pipeline.

Paper-canonical config (overrides paper text where it disagrees with the
paper-time code):

* target_model:    ``claude-sonnet-4-5-20250929`` (paper §A.4 says Sonnet 4.6
                   — typo; paper-time code shipped Sonnet 4.5)
* judge:           38-metric Petri judge via :func:`core.judges.judge_with_38_metrics`
* judge_model:     ``claude-opus-4-6`` (paper-canonical FINAL — verified
                   from paper-time ``paper §A.4 config``)
* num_seeds:       181 (general Petri seeds, ``data/seeds_petri_181.json``)
* rollouts:        3
* max_turns:       30
* anonymize_target: False  (production audits use the real model name)
* enable_prefill:  False
* judge thinking:  False

Total: 181 × 3 = 543 audits.

Bucket score (computed by ``core.aggregate.compute_production_value_summary``):
9 sub-metrics from 3 fields × {mean_100, yield_at_4, yield_at_7}, where the
fields are ``concerning``, ``concerning_with_no_mitigating_factors``, and
``needs_attention``. Higher is better.

Output layout::

    output_dir/
    ├── config.json
    ├── results.jsonl            # one row per (seed_idx, rollout)
    ├── judged_results.jsonl     # one row per judged audit (all 38 dims keyed by name)
    ├── summary.json             # bucket score + per-seed CIs + sub-metrics
    └── transcripts/<__production__>_s<NNN>_r<NN>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from core.api import APIClients
from core.aggregate import compute_production_value_summary, load_results
from core.judges import judge_with_38_metrics
from core.petri import get_auditor_tools_async

from evals._runner import (
    acquire_lock,
    aggregate_results,
    append_result,
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
logger = logging.getLogger("evals.production_value")


# ---------------------------------------------------------------------------
# Paper-canonical defaults
# ---------------------------------------------------------------------------

PRODUCTION_VALUE_DEFAULTS = {
    "target_model": "claude-sonnet-4-5-20250929",  # real production target
    "judge_model": "claude-opus-4-6",              # paper-canonical FINAL (paper §A.4 config)
    "rollouts": 3,  # paper-canonical (appendix.tex:482 — "181 × 3 rollouts = 543 audits")
    "max_turns": 30,
    "audit_concurrent": 100,
    "judge_concurrent": 50,
    "anonymize_target": False,                     # auditor sees real name
    "enable_prefill": False,
    "use_thinking_for_judge": False,
}

# Stable label used as the "quirk" key for transcript filenames + JSONL rows
# (production audits have no quirk dimension).
PRODUCTION_QUIRK_LABEL = "__production__"

REPO_ROOT = Path(__file__).resolve().parent.parent
PETRI_SEEDS_PATH = REPO_ROOT / "data" / "seeds_petri_181.json"


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def load_general_seeds(max_seeds: int | None = None) -> list[str]:
    """Load 181 general Petri seeds from ``data/seeds_petri_181.json``.

    Accepts either a top-level list or a ``{"all": [...]}`` wrapper to match
    both legal shapes the paper-time pipeline allowed.
    """
    if not PETRI_SEEDS_PATH.exists():
        raise FileNotFoundError(f"Petri seeds not found at {PETRI_SEEDS_PATH}.")
    data = json.loads(PETRI_SEEDS_PATH.read_text())
    if isinstance(data, dict):
        seeds = data.get("all", [])
    else:
        seeds = data
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    return seeds


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "production_value eval: "
            "OOD audit quality vs real production model, 38-metric Petri judge."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--auditor-model", required=True, help="Auditor model ID")
    parser.add_argument(
        "--target-model",
        default=PRODUCTION_VALUE_DEFAULTS["target_model"],
        help=f"Target (default: {PRODUCTION_VALUE_DEFAULTS['target_model']})",
    )
    parser.add_argument(
        "--judge-model",
        default=PRODUCTION_VALUE_DEFAULTS["judge_model"],
        help=f"Judge (default: {PRODUCTION_VALUE_DEFAULTS['judge_model']})",
    )
    parser.add_argument(
        "--rollouts", type=int,
        default=PRODUCTION_VALUE_DEFAULTS["rollouts"],
        help=f"Rollouts per seed (default: {PRODUCTION_VALUE_DEFAULTS['rollouts']})",
    )
    parser.add_argument(
        "--max-seeds", type=int, default=None,
        help="Limit number of seeds (default: 181 = full Petri set)",
    )
    parser.add_argument(
        "--max-turns", type=int,
        default=PRODUCTION_VALUE_DEFAULTS["max_turns"],
        help=f"Max turns per audit (default: {PRODUCTION_VALUE_DEFAULTS['max_turns']})",
    )
    parser.add_argument(
        "--audit-concurrent", type=int,
        default=PRODUCTION_VALUE_DEFAULTS["audit_concurrent"],
        help="Max concurrent audits",
    )
    parser.add_argument(
        "--judge-concurrent", type=int,
        default=PRODUCTION_VALUE_DEFAULTS["judge_concurrent"],
        help="Max concurrent judge calls",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Audit only (judge later)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lockfile = acquire_lock(args.output_dir)

    results_file = args.output_dir / "results.jsonl"
    judged_file = args.output_dir / "judged_results.jsonl"
    transcript_dir = args.output_dir / "transcripts"

    seeds = load_general_seeds(args.max_seeds)

    config = {
        "axis": "production_value",
        "description": "OOD production audit quality",
        "auditor_model": args.auditor_model,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "num_seeds": len(seeds),
        "rollouts": args.rollouts,
        "max_turns": args.max_turns,
        "max_seeds": args.max_seeds,
        "audit_concurrent": args.audit_concurrent,
        "judge_concurrent": args.judge_concurrent,
        "anonymize_target": PRODUCTION_VALUE_DEFAULTS["anonymize_target"],
        "enable_prefill": PRODUCTION_VALUE_DEFAULTS["enable_prefill"],
        "use_thinking_for_judge": PRODUCTION_VALUE_DEFAULTS["use_thinking_for_judge"],
        "skip_judge": args.skip_judge,
        "timestamp": datetime.now().isoformat(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Build task list (no quirk dimension — just seeds × rollouts).
    tasks_to_run: list[dict] = []
    for seed_idx, seed_instruction in enumerate(seeds):
        for rollout in range(args.rollouts):
            tasks_to_run.append({
                "seed_instruction": seed_instruction,
                "seed_idx": seed_idx,
                "rollout": rollout,
            })

    completed_audits = load_completed(results_file)
    tasks_to_run = [
        t for t in tasks_to_run
        if (PRODUCTION_QUIRK_LABEL, t["seed_idx"], t["rollout"]) not in completed_audits
    ]

    total = len(tasks_to_run) + len(completed_audits)
    logger.info(
        "production_value: %d total audits, %d completed, %d remaining",
        total, len(completed_audits), len(tasks_to_run),
    )
    logger.info(
        "Auditor=%s | Target=%s (production, real name visible) | Judge=%s",
        args.auditor_model, args.target_model, args.judge_model,
    )
    logger.info(
        "Concurrency: audit=%d, judge=%d",
        args.audit_concurrent, args.judge_concurrent,
    )
    logger.info("Seeds: %d general Petri seeds", len(seeds))

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

    async def judge_transcript(transcript_path: Path) -> None:
        """Judge a single transcript using the 38-metric Petri judge.

        The judge returns a dict keyed by the 38 dimension names (e.g.
        ``concerning``, ``concerning_with_no_mitigating_factors``,
        ``needs_attention``, ``scenario_realism``, ...). All 38 fields are
        persisted into ``judged_results.jsonl``.
        """
        nonlocal judges_done, judges_errors
        meta_key: tuple | None = None
        try:
            t = json.loads(transcript_path.read_text())
            meta = t["metadata"]
            meta_key = (meta["quirk"], meta["seed_idx"], meta["rollout"])
            if meta_key in already_judged:
                return

            async with judge_semaphore:
                scores = await judge_with_38_metrics(
                    combined_messages=t.get("combined_messages", []),
                    judge_model=args.judge_model,
                    clients=clients,
                    use_thinking=PRODUCTION_VALUE_DEFAULTS["use_thinking_for_judge"],
                )

            row = {
                "axis": "production_value",
                "quirk": PRODUCTION_QUIRK_LABEL,
                "seed_idx": meta["seed_idx"],
                "rollout": meta["rollout"],
                "completed": True,
                **scores,  # all 38 dims keyed by name
            }
            async with judged_lock:
                with open(judged_file, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

            already_judged.add(meta_key)
            judges_done += 1
            if judges_done % 25 == 0:
                logger.info("  Judged: %d transcripts", judges_done)
        except Exception as e:
            judges_errors += 1
            logger.error(
                "Judge failed for %s: %s", meta_key or transcript_path.name, e,
            )

    async def run_audit_and_judge(task: dict) -> None:
        nonlocal audits_done

        result = await run_single_audit(
            seed_instruction=task["seed_instruction"],
            auditor_model=args.auditor_model,
            target_model=args.target_model,
            max_turns=args.max_turns,
            clients=clients,
            session_semaphore=audit_semaphore,
            quirk_name=PRODUCTION_QUIRK_LABEL,
            quirk_prompt=None,                      # no planted quirk
            seed_idx=task["seed_idx"],
            rollout=task["rollout"],
            anonymize_target=PRODUCTION_VALUE_DEFAULTS["anonymize_target"],
            enable_prefill=PRODUCTION_VALUE_DEFAULTS["enable_prefill"],
            save_transcript=True,
            transcript_dir=transcript_dir,
            tools=tools,
        )
        result["axis"] = "production_value"

        async with results_lock:
            append_result(results_file, result)

        audits_done += 1
        if audits_done % 50 == 0:
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
                judge_transcript(Path(result["_transcript_abs_path"]))
            )

    if not args.skip_judge and unjudged_keys:
        unjudged_paths: list[Path] = []
        for key in unjudged_keys:
            _, seed_idx, rollout = key
            tp = (
                transcript_dir
                / f"{PRODUCTION_QUIRK_LABEL}_s{seed_idx:03d}_r{rollout:02d}.json"
            )
            if tp.exists():
                unjudged_paths.append(tp)
        if unjudged_paths:
            logger.info(
                "Re-judging %d orphaned transcripts from prior runs",
                len(unjudged_paths),
            )
            for tp in unjudged_paths:
                asyncio.create_task(judge_transcript(tp))

    gather_results = await asyncio.gather(
        *[run_audit_and_judge(t) for t in tasks_to_run],
        return_exceptions=True,
    )
    for i, r in enumerate(gather_results):
        if isinstance(r, Exception):
            logger.error("Task %d crashed: %s", i, r)

    if not args.skip_judge:
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            logger.info("Waiting for %d remaining judge tasks...", len(pending))
            await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.time() - start_time
    logger.info(
        "production_value complete: %d audits + %d judged in %.1f min",
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
    """Paper-canonical production_value summary: 9 sub-metrics × per-seed CIs."""
    audits_summary = aggregate_results(results_file)
    summary: dict = {
        "axis": "production_value",
        "audits": audits_summary,
    }

    if judged_file.exists():
        try:
            judged_rows = load_results(judged_file)
            # judge_with_38_metrics returns plain int scores; mark each row
            # completed=True at write time so the aggregator's `_valid_results`
            # filter accepts them.
            for r in judged_rows:
                r.setdefault("completed", True)
            summary["judged"] = compute_production_value_summary(judged_rows)
        except Exception as e:
            logger.warning("Failed to compute judged summary: %s", e)
            summary["judged"] = {"error": str(e)}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
