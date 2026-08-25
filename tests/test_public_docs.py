from __future__ import annotations

import re
from pathlib import Path

from tools.release_safety import scan_paths

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPOSITORY = "https://github.com/paulrosu11/training-auditing-agents-public"
PETRI_COMMIT = "f02b3ec70a6b9c2a1a7da172b2f3f69a1e1b0585"
FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
ACTION_REFERENCE = re.compile(
    r"^\s+uses:\s+([^@\s]+)@([^\s#]+)",
    re.MULTILINE,
)
EXPECTED_ACTION_SHAS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "astral-sh/setup-uv": "20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
    "gitleaks/gitleaks-action": "e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e",
}
GITLEAKS_VERSION = "8.30.1"
GITLEAKS_LINUX_X64_SHA256 = (
    "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
)
RELEASE_SCAN_PATHS = (
    "README.md",
    "DATASET_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md",
    "LICENSE",
    ".env.example",
    "data",
    "training_data",
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _workflow_step(workflow: str, name: str) -> str:
    match = re.search(
        rf"^      - name: {re.escape(name)}\n.*?(?=^      - name: |\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"workflow step {name!r} is missing"
    return match.group(0)


def test_readme_install_and_links_match_release_contract():
    readme = _read("README.md")

    assert f"git clone {PUBLIC_REPOSITORY}.git" in readme
    assert "--recurse-submodules" not in readme
    assert "git submodule" not in readme.lower()
    assert "pip install -e ." in readme
    assert "uv sync --frozen" in readme
    assert PETRI_COMMIT in readme
    assert "byte-identical" not in readme
    assert "safety-tooling" not in readme
    assert "~$25" not in readme
    for filename in ("DATASET_CARD.md", "THIRD_PARTY_NOTICES.md", "SECURITY.md"):
        assert f"[{filename}]({filename})" in readme


def test_readme_describes_actual_seed_and_live_run_contracts():
    readme = _read("README.md")
    lower_readme = readme.lower()

    assert "30 quirks × 50 seeds" in readme
    assert "`defer_to_users` × 75" in readme
    assert "`flattery` × 25" in readme
    assert "many-to-many" in lower_readme
    assert "actual call counts" in lower_readme
    assert "incur provider costs" in lower_readme


def test_training_data_readme_matches_seed_distribution():
    training_readme = _read("training_data/README.md")

    assert "30 quirks × 50 seeds" in training_readme
    assert "`defer_to_users` × 75" in training_readme
    assert "`flattery` × 25" in training_readme
    assert "32 quirks × 50 seeds" not in training_readme


def test_security_uses_private_reporting_routes():
    security = _read("SECURITY.md")
    lower_security = " ".join(security.lower().split())

    assert f"{PUBLIC_REPOSITORY}/security/advisories/new" in security
    assert "private vulnerability reporting" in lower_security
    assert "takedown" in lower_security
    assert "do not reproduce" in lower_security
    assert "sensitive conversation text" in lower_security
    assert "public issue" in lower_security


def test_github_ownership_and_dependency_updates_are_safe():
    codeowners = _read(".github/CODEOWNERS")
    dependabot = _read(".github/dependabot.yml")

    assert codeowners.strip() == "* @paulrosu11"
    assert dependabot.count('package-ecosystem: "pip"') == 1
    assert dependabot.count('package-ecosystem: "github-actions"') == 1
    assert dependabot.count('interval: "weekly"') == 2


def test_ci_is_read_only_offline_and_reproducible():
    workflow = _read(".github/workflows/ci.yml")

    assert "permissions:\n  contents: read" in workflow
    assert "contents: write" not in workflow
    assert "pull-requests: write" not in workflow
    assert '"3.11"' in workflow
    assert '"3.12"' in workflow
    assert "uv sync --frozen --extra dev" in workflow
    assert "python tests/smoke_offline.py" in workflow
    assert "python -m pytest tests -v" in workflow
    assert "python tools/release_safety.py" in workflow
    assert "gitleaks/gitleaks-action" in workflow
    assert "secrets." not in workflow
    for key_name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        assert key_name not in workflow


def test_public_release_documents_and_data_are_scanner_clean():
    assert scan_paths([ROOT / path for path in RELEASE_SCAN_PATHS]) == []


def test_ci_scans_the_public_release_surface():
    workflow = _read(".github/workflows/ci.yml")
    release_safety_step = _workflow_step(
        workflow, "Scan public release data and documentation"
    )
    expected_command = "uv run --frozen python tools/release_safety.py " + " ".join(
        RELEASE_SCAN_PATHS
    )

    assert expected_command in release_safety_step
    assert "git ls-files" not in release_safety_step


def test_gitleaks_action_uses_only_builtin_read_only_capabilities():
    workflow = _read(".github/workflows/ci.yml")
    action_step = _workflow_step(workflow, "Scan event commit range")

    assert re.search(
        r"^permissions:\n  contents: read\n\njobs:$", workflow, re.MULTILINE
    )
    assert workflow.count("permissions:") == 1
    assert "secrets." not in action_step
    assert "GITHUB_TOKEN: ${{ github.token }}" in action_step
    assert 'GITLEAKS_ENABLE_COMMENTS: "false"' in action_step
    assert 'GITLEAKS_ENABLE_UPLOAD_ARTIFACT: "false"' in action_step


def test_ci_runs_checksum_verified_gitleaks_over_the_complete_current_tree():
    workflow = _read(".github/workflows/ci.yml")
    install_step = _workflow_step(workflow, "Install checksum-verified Gitleaks")
    tree_scan_step = _workflow_step(workflow, "Scan complete current tree")
    archive = f"gitleaks_{GITLEAKS_VERSION}_linux_x64.tar.gz"
    checksums = f"gitleaks_{GITLEAKS_VERSION}_checksums.txt"
    release_url = (
        f"https://github.com/gitleaks/gitleaks/releases/download/v{GITLEAKS_VERSION}"
    )

    assert f'GITLEAKS_VERSION: "{GITLEAKS_VERSION}"' in install_step
    assert f'GITLEAKS_SHA256: "{GITLEAKS_LINUX_X64_SHA256}"' in install_step
    assert release_url in install_step
    assert archive in install_step
    assert checksums in install_step
    assert "curl --fail --location --silent --show-error" in install_step
    assert "sha256sum --check -" in install_step
    assert 'tar -xzf "$RUNNER_TEMP/$GITLEAKS_ARCHIVE"' in install_step
    assert 'echo "$RUNNER_TEMP/gitleaks-bin" >> "$GITHUB_PATH"' in install_step
    assert re.search(
        r"^        run: gitleaks dir --no-banner --redact=100 \.$",
        tree_scan_step,
        re.MULTILINE,
    )
    assert "git ls-files" not in tree_scan_step
    assert "docker" not in tree_scan_step.lower()


def test_ci_pins_every_external_action_to_full_commit_sha():
    workflow = _read(".github/workflows/ci.yml")
    action_references = ACTION_REFERENCE.findall(workflow)
    actual_action_shas: dict[str, set[str]] = {}

    assert action_references
    for action, reference in action_references:
        if action.startswith("./"):
            continue
        actual_action_shas.setdefault(action, set()).add(reference)
        assert FULL_COMMIT_SHA.fullmatch(reference), (
            f"{action} must be pinned to a full commit SHA, got {reference!r}"
        )

    assert actual_action_shas == {
        action: {sha} for action, sha in EXPECTED_ACTION_SHAS.items()
    }
