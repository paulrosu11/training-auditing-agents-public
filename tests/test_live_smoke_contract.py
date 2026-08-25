from pathlib import Path

from evals import realism

ROOT = Path(__file__).resolve().parents[1]


def test_many_to_many_call_estimator():
    assert realism.estimate_realism_calls(20, 5, 20) == (100, 2_000)
    assert realism.estimate_realism_calls(1, 1, 20) == (1, 20)


def test_live_smoke_caps_realism_to_one_seed():
    source = (ROOT / "tests" / "smoke_live.py").read_text()
    l4 = source[source.index("def L4_realism") : source.index("def L5_crux")]

    assert '"--max-seeds", "1"' in l4
