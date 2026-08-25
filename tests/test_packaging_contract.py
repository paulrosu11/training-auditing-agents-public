from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PETRI_DEPENDENCY = (
    "petri @ git+https://github.com/meridianlabs-ai/inspect_petri.git"
    "@f02b3ec70a6b9c2a1a7da172b2f3f69a1e1b0585"
)
PETRI_COMMIT = "f02b3ec70a6b9c2a1a7da172b2f3f69a1e1b0585"


def test_no_git_submodule_contract():
    assert not (ROOT / ".gitmodules").exists()
    assert not (ROOT / "external" / "petri").exists()


def test_petri_is_exact_official_git_dependency():
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert PETRI_DEPENDENCY in project["dependencies"]
    assert not any("file:./external/petri" in dep for dep in project["dependencies"])
    assert project["requires-python"] == ">=3.11,<3.13"


def test_license_describes_the_direct_petri_dependency():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "vendors `external/petri`" not in license_text
    assert "780949d" not in license_text
    assert PETRI_COMMIT in license_text
    assert "Safety Research" in license_text
    assert "THIRD_PARTY_NOTICES.md" in license_text
