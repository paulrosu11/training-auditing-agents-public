"""Evaluation drivers for all paper evaluations.

Four-axis composite (paper Tab. ``frontier_full``):

- :mod:`evals.audit_quality`      — in-distribution investigation quality
- :mod:`evals.fp_calibration`     — false-positive rate on clean targets
- :mod:`evals.production_value`   — OOD concerning findings on a real production target
- :mod:`evals.realism`            — NIAH realism pairwise judging

Standalone evals:

- :mod:`evals.crux`               — 75-crux V1/V2 fast eval (``--paper-comparable`` flag for paper Tab. baseline)
- :mod:`evals.persona`            — PRISM-4 / Nexus Research persona-adoption (paper Tab. ``persona_adoption``)
- :mod:`evals.susceptibility`     — Differential quirk exhibition lift (paper Tab. ``susceptibility``)
- :mod:`evals.strategy_clustering`— 8-stage Clio pipeline (frozen K=63/14 codebook, paper appendix:1078)

Shared infrastructure:

- :mod:`evals._runner`     — resume / lock / JSONL primitives shared by the four composite axes
- :mod:`evals._simple_eval`— shared 5-stage simple-eval lib used by persona/susceptibility
"""
