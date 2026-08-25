from __future__ import annotations

import json
import re
from pathlib import Path

from tools.release_safety import scan_paths

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAINING_DATA = ROOT / "training_data"
WILDCHAT = DATA / "wildchat_blog_20.jsonl"
TOPIC_MATCHING = DATA / "realism_topic_matching"
DATASET_CARD = ROOT / "DATASET_CARD.md"
THIRD_PARTY_NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LICENSE = ROOT / "LICENSE"
README = ROOT / "README.md"
TRANSCRIPT_ROOTS = (
    DATA / "cruxes" / "transcripts",
    DATA / "cruxes" / "transcripts_indist",
)


def _wildchat_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in WILDCHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_wildchat_has_only_release_local_ids():
    rows = _wildchat_rows()

    assert len(rows) == 19
    assert [row["record_id"] for row in rows] == [
        f"wildchat-sanitized-{index:03d}" for index in range(1, 20)
    ]
    assert all("session_id" not in row for row in rows)


def test_wildchat_release_schema_is_preserved():
    rows = _wildchat_rows()

    assert all(
        set(row) == {"record_id", "source", "char_count", "transcript"} for row in rows
    )
    assert all(isinstance(row["source"], str) for row in rows)
    assert all(isinstance(row["char_count"], int) for row in rows)
    assert all(isinstance(row["transcript"], str) for row in rows)
    assert all(row["char_count"] == len(row["transcript"]) for row in rows)


def test_realism_topic_cache_matches_retained_wildchat_rows():
    rows = _wildchat_rows()
    scores = json.loads(
        (TOPIC_MATCHING / "seed_wildchat_scores_opus.json").read_text(encoding="utf-8")
    )
    assignment = json.loads(
        (TOPIC_MATCHING / "hungarian_assignment_opus.json").read_text(encoding="utf-8")
    )
    expected_indices = set(range(len(rows)))

    assert {entry["wc_idx"] for entry in scores} == expected_indices
    assert {int(index) for index in assignment["assignments"]} == expected_indices
    assert len(assignment["selected_seeds"]) == len(rows)


def test_realism_topic_cache_has_documented_inherited_gaps():
    scores = json.loads(
        (TOPIC_MATCHING / "seed_wildchat_scores_opus.json").read_text(encoding="utf-8")
    )
    pairs = {(entry["wc_idx"], entry["seed_idx"]) for entry in scores}
    dataset_card = DATASET_CARD.read_text(encoding="utf-8").lower()

    assert len(scores) == 3_428
    assert len(pairs) == 3_428
    assert 19 * 181 == 3_439
    assert 3_439 - len(pairs) == 11
    assert "3,428" in dataset_card
    assert "3,439" in dataset_card
    assert "11" in dataset_card
    assert "zero-fill" in dataset_card
    assert "0.0" in dataset_card


def test_realism_cache_zero_fills_missing_pairs():
    from evals.realism_cache import ScoreCacheCoverage, build_dense_score_rows

    scores = [
        {"wc_idx": 0, "seed_idx": 0, "overall": 7.5},
        {"wc_idx": 1, "seed_idx": 1, "overall": 4.25},
    ]

    dense_rows, coverage = build_dense_score_rows(scores, n_wc=2, n_seeds=2)

    assert dense_rows == [[7.5, 0.0], [0.0, 4.25]]
    assert coverage == ScoreCacheCoverage(
        expected_pairs=4,
        present_pairs=2,
        missing_pairs=2,
    )


def test_wildchat_has_no_long_numeric_identifiers():
    identifier_shape = re.compile(r"(?<!\d)\d{10,19}(?!\d)")

    assert all(
        identifier_shape.search(row["transcript"]) is None for row in _wildchat_rows()
    )


def test_wildchat_has_no_cookie_exfiltration_code():
    transcripts = "\n".join(row["transcript"] for row in _wildchat_rows())

    assert "document.cookie" not in transcripts
    assert "XMLHttpRequest" not in transcripts


def test_release_json_and_jsonl_files_parse():
    for root in (DATA, TRAINING_DATA):
        for path in sorted(root.rglob("*.json")):
            json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)


def test_crux_transcript_schema_is_preserved():
    paths = sorted(path for root in TRANSCRIPT_ROOTS for path in root.rglob("*.json"))

    assert len(paths) == 150
    for path in paths:
        transcript = json.loads(path.read_text(encoding="utf-8"))
        assert set(transcript) in (
            {"messages", "metadata", "target_messages"},
            {"events", "messages", "metadata", "target_messages"},
        )
        assert isinstance(transcript["messages"], list)
        assert isinstance(transcript["metadata"], dict)
        assert isinstance(transcript["target_messages"], list)
        if "events" in transcript:
            assert isinstance(transcript["events"], list)


def test_release_data_scanner_is_clean():
    findings = scan_paths([DATA, TRAINING_DATA])

    assert findings == []


def test_dataset_card_documents_release_contract():
    assert DATASET_CARD.exists(), "DATASET_CARD.md is missing"
    content = DATASET_CARD.read_text(encoding="utf-8")
    lower_content = content.lower()

    assert "allenai/WildChat-1M" in content
    assert "ODC-By" in content
    assert "configured model providers" in lower_content
    assert "distressing" in lower_content or "harmful" in lower_content
    assert "takedown" in lower_content


def test_readme_documents_retained_wildchat_count():
    content = README.read_text(encoding="utf-8")

    assert "19 retained WildChat conversations" in content
    assert "20 WildChat conversations" not in content


def test_third_party_notices_attributes_wildchat():
    assert THIRD_PARTY_NOTICES.exists(), "THIRD_PARTY_NOTICES.md is missing"
    content = THIRD_PARTY_NOTICES.read_text(encoding="utf-8")

    assert "allenai/WildChat-1M" in content
    assert "ODC-By" in content


def test_root_mit_license_excludes_third_party_data():
    content = LICENSE.read_text(encoding="utf-8").lower()

    assert "third-party data" in content
    assert "mit grant does not apply" in content
