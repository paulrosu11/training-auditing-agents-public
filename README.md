# training-auditing-agents

Evaluation code and training data for **"Training Alignment Auditors via
Reinforcement Learning"** (NeurIPS 2026). The repo reproduces the paper's eight
evaluations on untrained frontier baseline models and ships the seed data the
paper used for GRPO training.

## What's in here

**Eight evaluations** (`evals/`):

- `audit_quality` — investigation quality on system-prompted model organisms
- `fp_calibration` — false-positive rate on a clean target
- `production_value` — Petri 38-metric judge on a real production target (Sonnet 4.6)
- `realism` — NIAH pairwise vs. WildChat with Hungarian topic-matching
- `crux` — 75-crux fast eval (use `--paper-comparable` to swap in the in-distribution 16-quirk set + GPT-4.1-mini target the paper uses)
- `persona` — PRISM-4 / Nexus Research persona-adoption across the 14 models in the paper table
- `susceptibility` — differential quirk-exhibition lift (5-stage Sonnet 4.5 / Haiku 4.5 / Opus 4.5 pipeline)
- `strategy_clustering` — 8-stage Clio pipeline producing the frozen 63-strategy / 14-archetype codebook

**Composite** (`orchestration/run_all.py`): unweighted mean of `audit_quality`,
`fp_calibration`, `production_value`, `realism`, each on a 0-100 scale.

**Eval inputs** (`data/`):

- `seeds_audit_quality_15_per_quirk.json` — held-out 6 quirks × 15 tailored seeds used by `audit_quality` and `fp_calibration`.
- `seeds_petri_181.json` — the 181 generic Petri seeds used by `production_value`.
- `cruxes/` — 75-crux subsets (held-out + paper-comparable in-distribution).
- `wildchat_blog_20.jsonl` — the 19 retained WildChat conversations used as the public-release realism haystack (the filename is historical).
- `realism_topic_matching/` — cached Opus 4.6 topic-similarity scores. The retained cache has 3,428 of 3,439 unique pairs; `realism` logs and zero-fills the 11 inherited gaps with 0.0 instead of silently calling the topic judge.
- `strategy_clustering_codebook.json` — the frozen 63-strategy / 14-archetype codebook used by `evals/strategy_clustering` Stage 7.
- `ideas_cache.json`, `identity_ideas.json` — paper-canonical idea pools for `susceptibility` and `persona`.

**Training data** (`training_data/`) — see `training_data/README.md` for the full layout:

- `training_data/seeds/quirk_seeds_32x50.json` — 1,600 paper-canonical
  prompts: 30 quirks × 50 seeds, `defer_to_users` × 75, and `flattery` × 25.
- `training_data/seeds/petri_seeds_181.json` — duplicate of the eval Petri seeds, present for reproducibility of the paper's training mix.
- `training_data/judges/v1_target_view_binary.txt` — Sonnet-4 V1 binary reward judge.
- `training_data/judges/pairwise.txt` — pairwise training judge used by both intragroup and reference pairwise (the paper uses a single prompt for both).
- `training_data/judges/fp_intragroup_pairwise.txt` — FP-calibration-group reward judge.

Quirk Jinja templates live at `core/quirks_data/` (32 templates + `templates/base_instruction.jinja2`).

## Install

Python 3.11 and 3.12 are supported. Clone the repository normally, then choose
either the standard editable install or the frozen `uv` environment:

```bash
git clone https://github.com/paulrosu11/training-auditing-agents-public.git
cd training-auditing-agents-public
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
# Reproducible development and validation environment, including pytest.
uv sync --frozen --extra dev
```

Petri is installed directly from its official repository at commit
`f02b3ec70a6b9c2a1a7da172b2f3f69a1e1b0585`, as pinned in both
`pyproject.toml` and `uv.lock`. No nested repository checkout is required.

## API keys for live runs

The offline smoke test, test suite, data inspection, and release-safety scan do
not require API credentials. Live evaluations do. Copy `.env.example` to
`.env`, add only the keys required by the providers you select, and never
commit that file:

```bash
cp .env.example .env
```

Populate the relevant variables named in `.env.example`; the routing table
below shows which variable each provider uses.

**Provider routing** is keyed off the model-ID prefix:

| Model ID | Provider | Key |
|---|---|---|
| `claude-...` (e.g. `claude-sonnet-4-6`) | Anthropic direct | `ANTHROPIC_API_KEY` |
| `gpt-...`, `openai/...` | OpenAI direct | `OPENAI_API_KEY` |
| `meta-llama/...`, `x-ai/...`, `google/...`, `deepseek/...`, `bytedance-seed/...`, `openrouter/...` | OpenRouter | `OPENROUTER_API_KEY` |

For Anthropic models, always use the bare `claude-...` ID — that hits the
Anthropic API directly with the right tool-use schema, prompt caching, and
extended thinking. (The audit loop probes the response's `provider` field at
runtime to decide whether prefill-as-assistant is supported, so prefill works
correctly across both Anthropic-direct and OpenRouter targets.)

## Smoke

```bash
python tests/smoke_offline.py  # imports, prompt SHAs, codebook hash, seed determinism, schema validation
python tests/smoke_live.py     # real provider requests; 1-seed × 1-rollout caps where supported
```

`smoke_live.py` exercises Anthropic, OpenAI, and OpenRouter paths, so running
the complete script requires explicit credentials for all three providers.
Live commands incur provider costs, and their duration and cost depend on the
selected models, current pricing, retries, and provider limits. Review the
command and current provider terms before opting in.

Realism is many-to-many: every generated audit is judged against all 19
retained WildChat conversations. `evals.realism` logs actual call counts as it
runs. Its call plan is `seeds × rollouts` audits and `audits × 19` NIAH judge
calls; the one-seed, one-rollout live smoke therefore plans 1 audit and 19
judge calls. Passing `--recompute-matching` additionally makes topic-classifier
calls; the default uses the retained cache with 11 inherited gaps zero-filled.

## Run a single eval

```bash
# Investigation quality on system-prompted model organisms
python -m evals.audit_quality \
  --auditor-model claude-sonnet-4-6 \
  --output-dir results/sonnet_46_audit_quality

# Production value (38-metric Petri judge against a clean production target)
python -m evals.production_value \
  --auditor-model claude-haiku-4-5-20251001 \
  --output-dir results/haiku_production_value

# Realism (Hungarian topic-matching uses cached scores by default)
python -m evals.realism \
  --auditor-model claude-opus-4-6 \
  --output-dir results/opus_realism

# Crux fast eval (default = 75 held-out cruxes against Llama 3.3 70B)
python -m evals.crux \
  --auditor-model claude-haiku-4-5-20251001 \
  --output results/haiku_crux.json

# Crux paper-comparable mode (16-quirk in-distribution + gpt-4.1-mini target)
python -m evals.crux \
  --auditor-model claude-haiku-4-5-20251001 \
  --paper-comparable \
  --output results/haiku_crux_paper.json

# Persona adoption across 14 paper-canonical models
# (--models takes a comma-separated list of short-names; default = all 14)
python -m evals.persona \
  --models claude-sonnet-4.6,gpt-5-mini,gemini-3-flash \
  --output-dir results/persona

# Susceptibility lift (quirk vs. baseline) on training targets
python -m evals.susceptibility \
  --models llama-3.3-70b,gemini-3-flash,grok-4.1-fast \
  --output-dir results/susceptibility
```

## Run the full composite

```bash
python -m orchestration.run_all \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6 \
  --evals audit_quality fp_calibration production_value realism \
  --parallel
```

Then aggregate per model:

```bash
python -m orchestration.analyze \
  --baseline-dir results/claude-haiku-4-5-20251001 \
  --output-dir results/claude-haiku-4-5-20251001/analysis
```

## Strategy clustering

The 8-stage Clio pipeline (extract → embed → cluster → label → rollup → freeze
→ classify → aggregate). Stage 7 (forward classification) uses the frozen
`data/strategy_clustering_codebook.json`, so you can classify new transcripts
against the paper's codebook without re-running stages 1-6.

```bash
# Run all 8 stages on your own audit-quality transcripts
python -m evals.strategy_clustering --stage all \
  --corpus-dir results/sonnet_46_audit_quality \
  --out-dir results/clustering

# Forward-classify against the frozen codebook only (skip stages 1-6)
python -m evals.strategy_clustering --stage classify \
  --corpus-dir results/sonnet_46_audit_quality \
  --codebook data/strategy_clustering_codebook.json \
  --out-dir results/clustering
```

## Reproducing paper tables

Run the full frontier matrix:

```bash
python -m orchestration.run_all \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-5-20250929 claude-sonnet-4-6 \
           claude-opus-4-6 google/gemini-3-flash-preview \
           google/gemini-3.1-flash-lite-preview gpt-5-mini \
  --evals audit_quality fp_calibration production_value realism \
  --parallel
```

## Hardware

API-only — no GPU is needed for any of the eight evals shipped here. Runtime is
primarily bounded by Anthropic, OpenAI, and OpenRouter latency and rate limits.
Default concurrency is auditor=150, target=100, judge=75 and is tunable per
evaluation via `--audit-concurrent` and `--judge-concurrent`.

## Release documentation and license

The repository code is MIT licensed; third-party data and dependencies retain
their own terms. Review [DATASET_CARD.md](DATASET_CARD.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [SECURITY.md](SECURITY.md),
and [LICENSE](LICENSE) before use or redistribution.
