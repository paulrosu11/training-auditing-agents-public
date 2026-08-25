# Training data

The seeds and judge prompts the paper used for GRPO training. The training
*code* itself is not shipped — only the data, so anyone running their own
GRPO loop (or a different RL algorithm) can reproduce the exact training
distribution.

## Layout

```
training_data/
├── README.md                       # this file
├── seeds/
│   ├── quirk_seeds_32x50.json      # 1,600 prompts across 32 quirk keys (distribution below)
│   └── petri_seeds_181.json        # 181 generic Petri investigation seeds
└── judges/
    ├── v1_target_view_binary.txt   # Binary configuration reward judge (paper §A.13, app:binary_judge)
    ├── pairwise.txt                # Pairwise Training Judge — used by BOTH intragroup and reference pairwise (paper §A.13, app:scaled_pairwise_judge / app:petri_pairwise_v2_judge)
    └── fp_intragroup_pairwise.txt  # FP calibration-group judge (paper §A.13, app:fp_pairwise_judge)
```

## `seeds/quirk_seeds_32x50.json`

The complete set of 1,600 paper-canonical training seeds. It is a JSON object
keyed by quirk name, with this exact allocation: 30 quirks × 50 seeds,
`defer_to_users` × 75, and `flattery` × 25. The historical filename records
the 32 quirk keys and 1,600 total prompts; it does not mean every key has the
same number of seeds.

Each instruction tells the auditor what scenario to set up (target persona,
system context, realistic user scenario) and gives investigation hints. Quirks
span the full 32-quirk set covered in `core/quirks_data/`; the held-out 6
quirks (`ai_welfare_poisoning`, `secret_loyalty`,
`hallucinates_citations`, `anti_ai_regulation`, `contextual_optimism`,
`animal_welfare`) are present here so you can train on a subset and still
evaluate against the held-out ones with `evals/audit_quality.py` and
`evals/fp_calibration.py`.

The paper's 26 training quirks are the 32-key set minus those 6 held-out
quirks. Their allocation is 24 × 50 plus 75 plus 25, giving exactly 1,300
unique `(quirk, seed)` GRPO groups.

## `seeds/petri_seeds_181.json`

The 181 generic Petri investigation seeds — short, behaviour-agnostic
prompts that ask the auditor to look for any concerning behaviour. The
paper's `production_value` evaluation uses these (3 rollouts × 181 seeds =
543 audits).

## `judges/`

Three training-judge prompts. Each is byte-identical to the paper-time source.

- **`v1_target_view_binary.txt`** — Binary V1 reward judge (Sonnet 4). Reads
  the conversation summary and answers, looking only at the target's
  behaviour, whether the target exhibited the hidden behaviour. Returns
  `yes` / `no`. Used as the reward signal for the Binary configuration; also
  serves as the target-focused crux evaluation metric.

- **`pairwise.txt`** — Pairwise Training Judge (Sonnet 4). Compares two
  complete audit transcripts of the same target with the same hidden
  behaviour and scenario, alongside both transcripts' 38-metric Petri
  scores. Returns a continuous comparative 1–99 score. Penalises evaluation
  awareness, confrontational style, and false accusations; requires the
  judge to verify evidence by quoting the target's own responses.

  This is the **same prompt used for both intragroup pairwise and reference
  pairwise** GRPO training (paper appendix `app:scaled_pairwise_judge` /
  `app:petri_pairwise_v2_judge` are the same `subsection`). The two
  configurations differ only in *which* pair of transcripts is compared:
  - **Intragroup pairwise**: each of the C(8,2)=28 within-group pairs.
  - **Reference pairwise** (flagship `ap4`): each rollout vs. a cached
    baseline transcript from a previous snapshot, refreshed every 50 epochs.

- **`fp_intragroup_pairwise.txt`** — FP-calibration-group reward judge for
  groups where the target has no hidden behaviour. Returns a continuous
  comparative score emphasising calibration and honesty: rewards "I found
  nothing concerning" over false alarms.

## How to use

Drop these into your own GRPO trainer. A single training step looks like:

1. Sample a batch of `(quirk, seed)` pairs from `quirk_seeds_32x50.json`.
   Some fraction of groups (the paper's flagship config uses 4/8) get the
   seed randomised but the target swapped to a clean Haiku 4.5 — these are
   FP calibration groups.
2. Run 8 audit rollouts per `(quirk, seed)` group through your auditor
   model on the Petri scaffold (`core/audit.py`). The non-FP groups target
   a quirk-injected target (paper uses Llama 3.3 70B via OpenRouter); the
   FP groups target a clean Haiku 4.5.
3. Score each rollout with `pairwise.txt`. Two configurations:
   - **Reference Pairwise (flagship, the result reported in the paper)**:
     compare each rollout to a cached baseline transcript from a previous
     snapshot. Baseline refreshed every 50 epochs. Each rollout
     participates in one comparison.
   - **Petri-informed Intragroup Pairwise**: do C(8,2)=28 within-group
     comparisons. Each rollout's reward is its win count divided by 7.
   For FP groups (clean target), use `fp_intragroup_pairwise.txt` instead.
4. Compute GRPO advantages within each `(quirk, seed, is_fp)` group
   independently — FP and non-FP groups never share normalisation.
5. Train on the resulting advantages.

Quirk Jinja templates (which produce the actual hidden-behaviour system
prompts the target sees) live at `core/quirks_data/`, with one `.jinja2`
file per quirk plus `base_instruction.jinja2`.
`core/quirks.py::render_quirk_system_prompt` is the canonical renderer.
