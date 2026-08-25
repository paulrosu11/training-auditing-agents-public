"""
Crux (V1+V2 dual-judge) evaluation — paper-canonical.

Loads the 75-crux subset from ``data/cruxes/eval_cruxes_75.json``, replays each
crux ``num_regens`` times with the auditor under test, then judges the
continuation with both the V1 (target-view binary) and V2 (auditor-view binary)
prompts. Both judge calls use ``claude-sonnet-4-20250514`` with extended
thinking **disabled** (paper appendix:601, 1419, 1449); ``enable_prefill`` is
false (paper-canonical default in ``core.audit``).

The aggregation logic mirrors
``proposed_pipeline/lib/evaluation.evaluate_model_on_cruxes`` (lines 1222+):
V1 judge failures are excluded from both numerator and denominator;
``success_rate`` is computed over judge-successful rollouts only. V2 failures
are tracked separately and emitted as ``v2_judge_failures``.

Reuses (no fresh logic):
- ``core.api.APIClients`` — three-semaphore concurrency wrapper.
- ``core.audit.AuditState.load_from_transcript`` and ``run_audit_continuation``.
- ``core.judges.judge_elicitation`` with ``JUDGE_PROMPT_V1`` / ``JUDGE_PROMPT_V2``.
- ``core.petri.get_auditor_tools_async`` — six Petri auditor tools.

CLI:

    python evals/crux.py \
        --auditor-model claude-haiku-4-5-20251001 \
        --output out/crux_haiku_baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.api import APIClients
from core.audit import AuditState, run_audit_continuation
from core.judges import JUDGE_PROMPT_V1, JUDGE_PROMPT_V2, judge_elicitation
from core.petri import get_auditor_tools_async

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGET_MODEL = "meta-llama/llama-3.3-70b-instruct"
PAPER_COMPARABLE_TARGET = "openai/gpt-4.1-mini"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CRUXES_PATH = "data/cruxes/eval_cruxes_75.json"
PAPER_COMPARABLE_CRUXES_PATH = "data/cruxes/eval_cruxes_75_indist.json"
DEFAULT_NUM_REGENS = 10
DEFAULT_NUM_TURNS = 5
DEFAULT_AUDIT_CONCURRENT = 20
DEFAULT_JUDGE_CONCURRENT = 10

# Paper-canonical: judge thinking OFF and prefill OFF
PAPER_USE_THINKING = False
PAPER_ENABLE_PREFILL = False

# Match source eval — retry the judge a few times before giving up.
_JUDGE_MAX_RETRIES = 5
_JUDGE_BACKOFF_DELAYS = (5, 10, 20, 40, 60)


# ---------------------------------------------------------------------------
# Per-rollout result container
# ---------------------------------------------------------------------------

@dataclass
class _RolloutResult:
    quirk: str
    seed_idx: int
    crux_index: int
    regen_idx: int
    success: bool = False
    elicited_v1: Optional[bool] = None  # None = judge failure
    elicited_v2: Optional[bool] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Crux loading
# ---------------------------------------------------------------------------

def _load_cruxes(cruxes_path: Path, quirk_filter: Optional[set[str]] = None) -> list[dict]:
    with open(cruxes_path) as f:
        data = json.load(f)
    cruxes = data.get("cruxes", [])
    if quirk_filter:
        cruxes = [c for c in cruxes if c.get("quirk") in quirk_filter]
    return cruxes


def _resolve_transcript_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


# ---------------------------------------------------------------------------
# Single (crux, regen) rollout
# ---------------------------------------------------------------------------

async def _run_single(
    crux: dict,
    regen_idx: int,
    auditor_model: str,
    target_model: str,
    judge_model: str,
    num_turns: int,
    tools: list[dict],
    clients: APIClients,
) -> _RolloutResult:
    quirk = crux.get("quirk", "")
    seed_idx = int(crux.get("seed_idx", 0))
    crux_index = int(crux.get("crux_index", 0))
    res = _RolloutResult(quirk=quirk, seed_idx=seed_idx, crux_index=crux_index, regen_idx=regen_idx)

    transcript_path = _resolve_transcript_path(crux.get("transcript_path", ""))
    if not transcript_path.exists():
        res.error = f"Transcript not found: {transcript_path}"
        return res

    try:
        with open(transcript_path) as f:
            transcript_data = json.load(f)
    except Exception as e:
        res.error = f"Failed to read transcript: {e}"
        return res

    hidden_system_prompt = crux.get("hidden_system_prompt", "")

    # 1) Load + replay audit
    state = AuditState()
    state.load_from_transcript(
        transcript=transcript_data,
        crux_idx=crux_index,
        target_model=target_model,
        max_turns=15,
    )

    # If the transcript didn't carry the hidden prefix, inject from the crux entry.
    if hidden_system_prompt and not state.hidden_target_system_prefix:
        state.hidden_target_system_prefix = hidden_system_prompt
        if state.target_system_visible:
            state.target_system = hidden_system_prompt + "\n\n" + state.target_system_visible
        else:
            state.target_system = hidden_system_prompt

    try:
        final_state, completed = await run_audit_continuation(
            state,
            auditor_model=auditor_model,
            num_turns=num_turns,
            target_model=target_model,
            tools=tools,
            clients=clients,
            enable_prefill=PAPER_ENABLE_PREFILL,
        )
    except Exception as e:
        res.error = f"Audit error: {e}"
        return res

    if not completed:
        res.error = "Audit did not complete"
        return res

    res.success = True
    judge_hidden = final_state.hidden_target_system_prefix or hidden_system_prompt

    # 2) V1 judge (target-view, combined messages)
    res.elicited_v1 = await _judge_with_retry(
        prompt=JUDGE_PROMPT_V1,
        hidden=judge_hidden,
        messages=final_state.combined_messages,
        judge_model=judge_model,
        clients=clients,
        label="V1",
    )

    # 3) V2 judge (auditor-view, auditor messages — strict)
    res.elicited_v2 = await _judge_with_retry(
        prompt=JUDGE_PROMPT_V2,
        hidden=judge_hidden,
        messages=final_state.auditor_messages,
        judge_model=judge_model,
        clients=clients,
        label="V2",
    )

    return res


async def _judge_with_retry(
    prompt: str,
    hidden: str,
    messages: list[dict],
    judge_model: str,
    clients: APIClients,
    label: str,
) -> Optional[bool]:
    """Call ``judge_elicitation`` up to N times. Returns ``None`` on persistent failure."""
    last_err: Optional[Exception] = None
    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            return await judge_elicitation(
                hidden,
                messages,
                judge_model=judge_model,
                clients=clients,
                use_thinking=PAPER_USE_THINKING,
                judge_prompt=prompt,
            )
        except Exception as e:  # noqa: BLE001 — same pattern as source
            last_err = e
            if attempt < _JUDGE_MAX_RETRIES - 1:
                delay = _JUDGE_BACKOFF_DELAYS[attempt]
                logger.warning(
                    f"{label} judge attempt {attempt + 1}/{_JUDGE_MAX_RETRIES} failed "
                    f"({type(e).__name__}: {e}). Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
    logger.warning(f"{label} judge failed after {_JUDGE_MAX_RETRIES} attempts: {last_err}")
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(rollouts: list[_RolloutResult]) -> dict[str, Any]:
    """Aggregate rollout results into per-crux + overall V1 / V2 stats."""
    by_key: dict[tuple[str, int, int], list[_RolloutResult]] = {}
    for r in rollouts:
        by_key.setdefault((r.quirk, r.seed_idx, r.crux_index), []).append(r)

    per_crux: list[dict[str, Any]] = []
    total_evaluated_v1 = 0
    total_elicited_v1 = 0
    v1_judge_failures = 0
    total_evaluated_v2 = 0
    total_elicited_v2 = 0
    v2_judge_failures = 0

    for (quirk, seed_idx, crux_index), results in by_key.items():
        successful = [r for r in results if r.success]

        # V1 stats — exclude judge failures from both numerator and denominator.
        v1_judged = [r for r in successful if r.elicited_v1 is not None]
        v1_elicited = [r for r in v1_judged if r.elicited_v1]
        v1_fail_in_crux = len(successful) - len(v1_judged)

        sorted_v1 = sorted(v1_judged, key=lambda r: r.regen_idx)
        regen_elicited = [bool(r.elicited_v1) for r in sorted_v1]

        success_rate = (len(v1_elicited) / len(v1_judged)) if v1_judged else 0.0

        per_crux.append({
            "quirk": quirk,
            "seed_idx": seed_idx,
            "crux_index": crux_index,
            "num_evaluated": len(v1_judged),
            "num_elicited": len(v1_elicited),
            "regen_elicited": regen_elicited,
            "success_rate": success_rate,
            "v1_judge_failures": v1_fail_in_crux,
        })

        total_evaluated_v1 += len(v1_judged)
        total_elicited_v1 += len(v1_elicited)
        v1_judge_failures += v1_fail_in_crux

        # V2 stats — same "exclude failures" rule
        for r in successful:
            if r.elicited_v2 is None:
                v2_judge_failures += 1
            else:
                total_evaluated_v2 += 1
                if r.elicited_v2:
                    total_elicited_v2 += 1

    success_rate_v1 = total_elicited_v1 / total_evaluated_v1 if total_evaluated_v1 else 0.0
    success_rate_v2 = total_elicited_v2 / total_evaluated_v2 if total_evaluated_v2 else 0.0

    return {
        "total_evaluated": total_evaluated_v1,
        "total_elicited": total_elicited_v1,
        "success_rate": success_rate_v1,
        "v1_judge_failures": v1_judge_failures,
        "total_evaluated_v2": total_evaluated_v2,
        "total_elicited_v2": total_elicited_v2,
        "success_rate_v2": success_rate_v2,
        "v2_judge_failures": v2_judge_failures,
        "per_crux_results": per_crux,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

async def run_eval(
    auditor_model: str,
    output_path: Path,
    target_model: str = DEFAULT_TARGET_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    num_regens: int = DEFAULT_NUM_REGENS,
    num_turns: int = DEFAULT_NUM_TURNS,
    cruxes_path: Path = REPO_ROOT / DEFAULT_CRUXES_PATH,
    audit_concurrent: int = DEFAULT_AUDIT_CONCURRENT,
    judge_concurrent: int = DEFAULT_JUDGE_CONCURRENT,
    quirk_filter: Optional[set[str]] = None,
) -> dict[str, Any]:
    cruxes = _load_cruxes(cruxes_path, quirk_filter=quirk_filter)
    if not cruxes:
        raise SystemExit(f"No cruxes loaded from {cruxes_path}")

    logger.info(
        f"Crux eval: auditor={auditor_model} target={target_model} "
        f"judge={judge_model} cruxes={len(cruxes)} regens={num_regens} "
        f"turns={num_turns} thinking={PAPER_USE_THINKING}"
    )

    clients = APIClients(
        auditor_concurrent=audit_concurrent,
        target_concurrent=audit_concurrent,
        judge_concurrent=judge_concurrent,
    )

    tools = await get_auditor_tools_async()

    tasks = [
        _run_single(
            crux=crux,
            regen_idx=regen_idx,
            auditor_model=auditor_model,
            target_model=target_model,
            judge_model=judge_model,
            num_turns=num_turns,
            tools=tools,
            clients=clients,
        )
        for crux in cruxes
        for regen_idx in range(num_regens)
    ]
    logger.info(f"Dispatching {len(tasks)} rollouts...")

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    rollouts: list[_RolloutResult] = []
    for r in raw_results:
        if isinstance(r, BaseException):
            logger.error(f"Rollout raised: {r!r}")
            continue
        rollouts.append(r)

    agg = _aggregate(rollouts)

    final = {
        "model": auditor_model,
        "num_cruxes": len(cruxes),
        "num_regens": num_regens,
        **agg,
        "eval_config": {
            "auditor_model": auditor_model,
            "target_model": target_model,
            "judge_model": judge_model,
            "num_turns": num_turns,
            "num_regens": num_regens,
            "thinking": PAPER_USE_THINKING,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(
        f"Wrote {output_path}  V1={final['success_rate']:.3f}  "
        f"V2={final['success_rate_v2']:.3f}  "
        f"(V1 fails={final['v1_judge_failures']}, V2 fails={final['v2_judge_failures']})"
    )
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--auditor-model", required=True, help="Auditor model id (e.g. claude-haiku-4-5-20251001)")
    p.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--num-regens", type=int, default=DEFAULT_NUM_REGENS)
    p.add_argument("--num-turns", type=int, default=DEFAULT_NUM_TURNS)
    p.add_argument("--cruxes", default=str(REPO_ROOT / DEFAULT_CRUXES_PATH))
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--audit-concurrent", type=int, default=DEFAULT_AUDIT_CONCURRENT)
    p.add_argument("--judge-concurrent", type=int, default=DEFAULT_JUDGE_CONCURRENT)
    p.add_argument(
        "--quirks",
        nargs="*",
        default=None,
        help="Restrict to a subset of quirk names",
    )
    p.add_argument(
        "--paper-comparable",
        action="store_true",
        help=(
            "Run with paper-time canonical config: 16-quirk in-distribution "
            "75-crux subset against gpt-4.1-mini target. Reproduces paper "
            "Tab. baseline (~35.1 +/- 1.6 V1 for Haiku 4.5). Default mode "
            "uses the held-out 6-quirk set against Llama 3.3 70B (post-paper "
            "evaluation methodology)."
        ),
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argparser().parse_args()

    if args.paper_comparable:
        # Override defaults to reproduce paper-time baseline.
        # appendix.tex:535-610 documents this configuration.
        if args.target_model == DEFAULT_TARGET_MODEL:
            args.target_model = PAPER_COMPARABLE_TARGET
        default_cruxes_str = str(REPO_ROOT / DEFAULT_CRUXES_PATH)
        if args.cruxes == default_cruxes_str:
            args.cruxes = str(REPO_ROOT / PAPER_COMPARABLE_CRUXES_PATH)
        logging.getLogger(__name__).info(
            "PAPER-COMPARABLE MODE: target=%s cruxes=%s "
            "(reproduces paper Tab. baseline; Haiku 4.5 V1 ~35.1 +/- 1.6)",
            args.target_model, args.cruxes,
        )

    asyncio.run(run_eval(
        auditor_model=args.auditor_model,
        output_path=Path(args.output),
        target_model=args.target_model,
        judge_model=args.judge_model,
        num_regens=args.num_regens,
        num_turns=args.num_turns,
        cruxes_path=Path(args.cruxes),
        audit_concurrent=args.audit_concurrent,
        judge_concurrent=args.judge_concurrent,
        quirk_filter=set(args.quirks) if args.quirks else None,
    ))


if __name__ == "__main__":
    main()
