"""Dependency-light score-cache densification for the realism evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MISSING_SCORE_FILL_VALUE = 0.0


@dataclass(frozen=True)
class ScoreCacheCoverage:
    """Coverage of unique, in-range score pairs in a dense matrix."""

    expected_pairs: int
    present_pairs: int
    missing_pairs: int


def build_dense_score_rows(
    scores: Sequence[Mapping[str, object]],
    n_wc: int,
    n_seeds: int,
) -> tuple[list[list[float]], ScoreCacheCoverage]:
    """Build dense rows, filling every missing score pair with exactly 0.0."""
    if n_wc < 0 or n_seeds < 0:
        raise ValueError("score-matrix dimensions must be non-negative")

    rows = [[MISSING_SCORE_FILL_VALUE for _ in range(n_seeds)] for _ in range(n_wc)]
    present_pairs: set[tuple[int, int]] = set()
    for entry in scores:
        wc = int(entry["wc_idx"])
        seed = int(entry["seed_idx"])
        if 0 <= wc < n_wc and 0 <= seed < n_seeds:
            rows[wc][seed] = float(entry["overall"])
            present_pairs.add((wc, seed))

    expected_pairs = n_wc * n_seeds
    coverage = ScoreCacheCoverage(
        expected_pairs=expected_pairs,
        present_pairs=len(present_pairs),
        missing_pairs=expected_pairs - len(present_pairs),
    )
    return rows, coverage
