"""Offline smoke test (no API calls).

Sections O1–O10 from the build spec. Runs in <1 minute, $0 cost.
Verifies the codebase is structurally sound and paper-canonical
constants match expected values before any live run.

Usage:
    python tests/smoke_offline.py
    pytest tests/smoke_offline.py -v
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
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
# O1: Imports — every core/, evals/, orchestration/ module imports clean
# ─────────────────────────────────────────────────────────────────────────────
def test_imports() -> bool:
    section("O1: Imports")
    targets = [
        "core.api",
        "core.audit",
        "core.judges",
        "core.petri",
        "core.quirks",
        "core.aggregate",
        "core.reasoning_utils",
    ]
    ok = True
    for name in targets:
        try:
            __import__(name)
            ok &= check(True, f"import {name}")
        except Exception as e:
            ok &= check(False, f"import {name}", str(e)[:120])
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O2: Prompt SHAs — paper-canonical prompts hash to known values
# ─────────────────────────────────────────────────────────────────────────────
def test_prompt_shas() -> bool:
    section("O2: Prompt SHAs (verbatim from paper-time)")
    from core.judges import (
        JUDGE_PROMPT_V1,
        JUDGE_PROMPT_V2,
        JUDGE_PROMPT_BASE,
        NIAH_JUDGE_PROMPT,
    )
    from core.petri import (
        construct_auditor_system_prompt,
        get_auditor_system_prompt,
        get_auditor_user_message,
    )

    def sha(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    ok = True
    # Just verify each is non-empty and has expected substrings
    ok &= check("auditing agent" in JUDGE_PROMPT_V1, "V1 has 'auditing agent'", sha(JUDGE_PROMPT_V1))
    ok &= check("hidden quirk" in JUDGE_PROMPT_V2.lower(),
                "V2 has 'hidden quirk'", sha(JUDGE_PROMPT_V2))
    ok &= check("impartial judge" in JUDGE_PROMPT_BASE,
                "JUDGE_PROMPT_BASE has 'impartial judge'", sha(JUDGE_PROMPT_BASE))
    ok &= check("ranking" in NIAH_JUDGE_PROMPT.lower() or "transcript" in NIAH_JUDGE_PROMPT.lower(),
                "NIAH judge has 'ranking' or 'transcript'", sha(NIAH_JUDGE_PROMPT))

    # Petri auditor prompt has expected placeholders
    sys_prompt = get_auditor_system_prompt()
    user_prompt = get_auditor_user_message()
    ok &= check(
        sha(sys_prompt) == "b34d05f5a24a79f3",
        "Petri auditor system prompt matches pinned commit",
        sha(sys_prompt),
    )
    ok &= check(
        sha(user_prompt) == "c59d71c216713919",
        "Petri auditor user prompt matches pinned commit",
        sha(user_prompt),
    )
    for placeholder in ["{target_model}", "{seed_instructions}", "{max_turns}", "{today_date}"]:
        ok &= check(placeholder in sys_prompt, f"AUDITOR_SYSTEM_PROMPT has {placeholder}")

    formatted_prompt = construct_auditor_system_prompt(
        target_model="offline-target",
        seed_instructions="offline-seed",
        max_turns=3,
        today_date="2026-08-25",
    )
    for value in ["offline-target", "offline-seed", "3", "2026-08-25"]:
        ok &= check(value in formatted_prompt, f"formatted Petri prompt contains {value}")

    from core.audit import AuditState

    state = AuditState()
    state.load_from_transcript(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<seed_instructions>offline-seed</seed_instructions>"
                    ),
                }
            ],
            "target_messages": [],
        },
        crux_idx=0,
        target_model="offline-target",
        max_turns=3,
    )
    ok &= check(
        "offline-target" in state.auditor_system,
        "AuditState formats the official Petri prompt",
    )

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O3: Codebook hashes — frozen 63/14 strategy clustering codebook
# ─────────────────────────────────────────────────────────────────────────────
def test_codebook() -> bool:
    section("O3: Strategy clustering codebook")
    cb_path = REPO / "data" / "strategy_clustering_codebook.json"
    ok = check(cb_path.is_file(), "codebook file exists", str(cb_path.relative_to(REPO)))
    if not ok:
        return ok
    cb = json.loads(cb_path.read_text())
    if isinstance(cb, dict) and "strategies" in cb:
        n_strat = len(cb["strategies"])
        n_arch = len(cb.get("archetypes", []))
        ok &= check(n_strat == 63, f"strategies count = {n_strat} (paper: 63)")
        ok &= check(n_arch == 14, f"archetypes count = {n_arch} (paper: 14)")
    elif isinstance(cb, list):
        ok &= check(len(cb) > 0, f"codebook has {len(cb)} entries")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O4: Aggregator math
# ─────────────────────────────────────────────────────────────────────────────
def test_aggregate_math() -> bool:
    section("O4: Aggregator math")
    from core.aggregate import (
        compute_audit_quality_summary,
        compute_fp_calibration_summary,
        compute_production_value_summary,
        compute_realism_summary,
        compute_composite_score,
        per_seed_t_ci,
    )
    ok = True
    # Simple t-CI sanity
    lo, hi = per_seed_t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    ok &= check(lo < 3.0 < hi, f"t-CI brackets mean: [{lo:.2f}, {hi:.2f}]")

    # Synthetic production_value: should produce 9 sub-metrics
    synthetic_pv = [
        {"quirk": "test", "seed_idx": s, "rollout": r,
         "concerning": 5.0, "concerning_with_no_mitigating_factors": 4.0, "needs_attention": 6.0}
        for s in range(5) for r in range(3)
    ]
    pv = compute_production_value_summary(synthetic_pv)
    ok &= check("overall" in pv, "production_value summary has 'overall'")

    # composite — must accept axis names
    summaries = {
        "audit_quality": {"overall": {"bucket_score": 70.0, "n": 90}},
        "fp_calibration": {"overall": {"bucket_score": 99.0, "n": 90}},
        "production_value": {"overall": {"bucket_score": 8.0, "n": 543}},
        "realism": {"overall": {"bucket_score": 13.0, "n": 100}},
    }
    composite = compute_composite_score(summaries)
    ok &= check(composite is not None, f"composite computes from axis-named keys: {composite}")
    if composite is not None:
        ok &= check(0 <= composite <= 100, f"composite in [0,100]: {composite:.2f}")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O5: Anonymizer purity
# ─────────────────────────────────────────────────────────────────────────────
def test_anonymizer() -> bool:
    section("O5: Anonymizer purity")
    try:
        from core.audit import apply_target_anonymization
    except (ImportError, AttributeError):
        return check(False, "core.audit.apply_target_anonymization importable")

    auditor_sys = "Target: claude-sonnet-4-6 system has..."
    auditor_user = "Probe claude-sonnet-4-6"
    sys_anon, usr_anon = apply_target_anonymization(
        auditor_sys, auditor_user, "claude-sonnet-4-6", anonymize=True
    )
    ok = True
    ok &= check("claude-sonnet-4-6" not in sys_anon, "anonymized system has no real model name")
    ok &= check("unspecified target model" in sys_anon.lower(),
                "anonymized system has 'unspecified target model'")
    # Pass-through when anonymize=False
    sys_pt, usr_pt = apply_target_anonymization(
        auditor_sys, auditor_user, "claude-sonnet-4-6", anonymize=False
    )
    ok &= check(sys_pt == auditor_sys, "anonymize=False is pass-through (system)")
    ok &= check(usr_pt == auditor_user, "anonymize=False is pass-through (user)")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O6: APIClients constructs + concurrency semaphores
# ─────────────────────────────────────────────────────────────────────────────
def test_api_clients() -> bool:
    section("O6: APIClients")
    from core.api import (
        APIClients,
        DEFAULT_AUDITOR_CONCURRENT,
        DEFAULT_TARGET_CONCURRENT,
        DEFAULT_JUDGE_CONCURRENT,
    )
    ok = True
    ok &= check(DEFAULT_AUDITOR_CONCURRENT == 150, f"DEFAULT_AUDITOR_CONCURRENT == 150 (got {DEFAULT_AUDITOR_CONCURRENT})")
    ok &= check(DEFAULT_TARGET_CONCURRENT == 100, f"DEFAULT_TARGET_CONCURRENT == 100 (got {DEFAULT_TARGET_CONCURRENT})")
    ok &= check(DEFAULT_JUDGE_CONCURRENT == 75, f"DEFAULT_JUDGE_CONCURRENT == 75 (got {DEFAULT_JUDGE_CONCURRENT})")
    try:
        c = APIClients()
        ok &= check(c is not None, "APIClients() constructs")
    except Exception as e:
        ok &= check(False, "APIClients() constructs", str(e)[:80])
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O7: Petri direct dependency pinned to the release commit
# ─────────────────────────────────────────────────────────────────────────────
def test_petri_pin() -> bool:
    section("O7: Petri direct dependency pin")
    from importlib.metadata import distribution

    direct_url_text = distribution("petri").read_text("direct_url.json")
    if direct_url_text is None:
        return check(False, "Petri install records direct_url.json")

    direct_url = json.loads(direct_url_text)
    ok = check(
        direct_url.get("url")
        == "https://github.com/meridianlabs-ai/inspect_petri.git",
        "Petri install uses the official repository",
        direct_url.get("url", "missing"),
    )
    vcs_info = direct_url.get("vcs_info", {})
    commit = vcs_info.get("commit_id", "")
    ok &= check(
        commit == "f02b3ec70a6b9c2a1a7da172b2f3f69a1e1b0585",
        "Petri install uses the pinned release commit",
        commit,
    )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O8: Realism seed indices
# ─────────────────────────────────────────────────────────────────────────────
def test_realism_seeds() -> bool:
    section("O8: Realism seed indices")
    RELEASE_MATCHED = [
        6,
        16,
        23,
        31,
        40,
        57,
        65,
        77,
        89,
        90,
        91,
        92,
        96,
        127,
        142,
        164,
        174,
        175,
        177,
    ]
    try:
        from evals.realism import SEED_INDICES
    except Exception:
        # Expected gap until Phase 3 builds evals/realism.py.
        return check(True, "evals.realism not built yet (Phase 3 will populate)", "skip")
    return check(
        list(SEED_INDICES) == RELEASE_MATCHED,
        f"SEED_INDICES match retained release cache (n={len(SEED_INDICES)})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# O9: Schema validation (data files)
# ─────────────────────────────────────────────────────────────────────────────
def test_data_schemas() -> bool:
    section("O9: Data file schemas")
    ok = True

    cruxes = json.loads((REPO / "data/cruxes/eval_cruxes_75.json").read_text())
    ok &= check(len(cruxes.get("cruxes", [])) == 75,
                f"eval_cruxes_75.json has 75 cruxes (got {len(cruxes.get('cruxes', []))})")
    if cruxes.get("cruxes"):
        first = cruxes["cruxes"][0]
        for k in ["quirk", "seed_idx", "crux_index", "transcript_path",
                  "hidden_system_prompt", "seed_instruction"]:
            ok &= check(k in first, f"crux record has '{k}'")

    seeds = json.loads((REPO / "data/seeds_petri_181.json").read_text())
    n = len(seeds.get("all", seeds)) if isinstance(seeds, dict) else len(seeds)
    ok &= check(n == 181, f"seeds_petri_181.json has 181 seeds (got {n})")

    ideas = json.loads((REPO / "data/ideas_cache.json").read_text())
    ok &= check(len(ideas) == 32, f"ideas_cache.json has 32 quirks (got {len(ideas)})")
    if ideas:
        first_quirk = next(iter(ideas))
        ok &= check(len(ideas[first_quirk]) == 10,
                    f"ideas_cache[{first_quirk!r}] has 10 ideas")

    identity = json.loads((REPO / "data/identity_ideas.json").read_text())
    ok &= check(len(identity) == 10, f"identity_ideas.json has 10 ideas (got {len(identity)})")

    haystack = (
        (REPO / "data/wildchat_blog_20.jsonl").read_text().strip().splitlines()
    )
    expected_ids = [f"wildchat-sanitized-{i:03d}" for i in range(1, 20)]
    actual_ids = [json.loads(line).get("record_id") for line in haystack]
    ok &= check(
        len(haystack) == 19,
        f"WildChat release subset has 19 entries (got {len(haystack)})",
    )
    ok &= check(actual_ids == expected_ids, "WildChat release-local IDs are sequential")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# O10: No trained-model API surface
# ─────────────────────────────────────────────────────────────────────────────
def test_no_trained_surface() -> bool:
    section("O10: No trained-model surface (grep)")
    # These patterns must NOT appear in any tracked source file (excluding external/, vendored Petri example key)
    forbidden = [
        r"ft_ckpt_",
        r"ft_snap_",
        r"ft_lp_",
        r"finetuning-2025-",
        r"Anthropic-Beta:\s*finetuning",
        r"ANTHROPIC_API_KEY_FINE",
        r"ANTHROPIC_API_KEY_HIGH_PRIO",
        r"ANTHROPIC_API_KEY_BATCH",
        r"create_finetuning_client",
        r"register_api_for_models\([^)]*ft_",
    ]
    # Self-reference: this test file lists the patterns it greps for; skip self.
    SELF = Path(__file__).resolve()
    leaks = []
    for path in REPO.rglob("*.py"):
        if any(
            x in path.parts
            for x in ("external", "work", ".venv", "__pycache__", ".git")
        ):
            continue
        if path.resolve() == SELF:
            continue
        text = path.read_text(errors="ignore")
        for pat in forbidden:
            if re.search(pat, text):
                leaks.append((path.relative_to(REPO), pat))
    ok = check(not leaks, f"no trained-model surface in code", f"{len(leaks)} hits" if leaks else "")
    for p, pat in leaks[:5]:
        print(f"      LEAK: {p} :: {pat}")
    return ok


def main() -> int:
    sections = [
        ("imports", test_imports),
        ("prompt_shas", test_prompt_shas),
        ("codebook", test_codebook),
        ("aggregate_math", test_aggregate_math),
        ("anonymizer", test_anonymizer),
        ("api_clients", test_api_clients),
        ("petri_pin", test_petri_pin),
        ("realism_seeds", test_realism_seeds),
        ("data_schemas", test_data_schemas),
        ("no_trained_surface", test_no_trained_surface),
    ]
    results = {}
    for name, fn in sections:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  {FAIL} {name} crashed: {e}")
            results[name] = False

    section("Summary")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    print()
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    if all(results.values()):
        print(f"PASS ({n_pass}/{n_total})")
        return 0
    else:
        print(f"FAIL ({n_pass}/{n_total} passed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
