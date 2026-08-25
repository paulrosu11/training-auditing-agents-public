from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import release_safety
from tools.release_safety import (
    Finding,
    sanitize_text,
    sanitize_value,
    scan_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SAFETY_SCRIPT = REPO_ROOT / "tools" / "release_safety.py"
SANITIZE_RELEASE_DATA_SCRIPT = REPO_ROOT / "tools" / "sanitize_release_data.py"


def _dotted(*parts: str) -> str:
    return ".".join(parts)


def _joined(*parts: str) -> str:
    return "".join(parts)


def _counts(*items: tuple[str, int]) -> dict[str, int]:
    return dict(items)


def test_sanitize_text_replaces_credential_and_contact_shapes():
    aws = "AKIA" + "A" * 16
    stripe = "sk_" + "live_" + "A" * 24
    email = _joined("person", "@", "gmail.com")
    text = f"key={aws} charge={stripe} contact={email}"

    cleaned, counts = sanitize_text(text)

    assert aws not in cleaned
    assert stripe not in cleaned
    assert email not in cleaned
    assert "<AWS_ACCESS_KEY_REDACTED>" in cleaned
    assert "<PAYMENT_TOKEN_REDACTED>" in cleaned
    assert "<EMAIL_REDACTED>" in cleaned
    assert sum(counts.values()) == 3


def test_sanitize_text_replaces_short_live_payment_token_shapes():
    token = "sk_" + "live_" + "A" * 10

    cleaned, counts = sanitize_text(f"charge={token}")

    assert token not in cleaned
    assert cleaned == "charge=<PAYMENT_TOKEN_REDACTED>"
    assert counts == _counts(("payment_" + "token", 1))


def test_sanitize_text_replaces_test_payment_token_shapes():
    token = "sk_" + "test_" + "A" * 10

    cleaned, counts = sanitize_text(f"charge={token}")

    assert token not in cleaned
    assert cleaned == "charge=<PAYMENT_TOKEN_REDACTED>"
    assert counts == _counts(("payment_" + "token", 1))


def test_sanitize_value_recurses_through_lists_and_dicts():
    value = {
        "messages": [{"text": _joined("https://hooks.slack.com/services/", "A/B/C")}]
    }

    cleaned, counts = sanitize_value(value)

    assert cleaned["messages"][0]["text"] == "<SLACK_WEBHOOK_REDACTED>"
    assert counts["slack_webhook"] == 1


def test_sanitize_value_redacts_credential_named_string_fields():
    credential_name = "SERVICE_" + "API_KEY"
    value = "short-example"

    cleaned, counts = sanitize_value({credential_name: value})

    assert cleaned[credential_name] == "<GENERIC_CREDENTIAL_REDACTED>"
    assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_value_requires_an_exact_approved_credential_marker():
    unrecognized_marker = "<UNRECOGNIZED_OPAQUE>"
    generic_marker = "<GENERIC_CREDENTIAL_REDACTED>"
    provider_marker = "<AWS_ACCESS_KEY_REDACTED>"
    value = {
        "SERVICE_API_KEY": unrecognized_marker,
        "DATABASE_PASSWORD": generic_marker,
        "BACKUP_TOKEN": provider_marker,
    }

    cleaned, counts = sanitize_value(value)

    assert counts == _counts(("generic_" + "credential", 1))
    assert cleaned["SERVICE_API_KEY"] == generic_marker
    assert cleaned["DATABASE_PASSWORD"] == generic_marker
    assert cleaned["BACKUP_TOKEN"] == provider_marker


def test_sanitize_value_preserves_non_string_credential_field_shapes():
    value = {
        "SERVICE_PASSWORD": 7,
        "ACCESS_TOKEN": ["part-a", "part-b"],
    }

    cleaned, counts = sanitize_value(value)

    assert cleaned == value
    assert isinstance(cleaned["SERVICE_PASSWORD"], int)
    assert isinstance(cleaned["ACCESS_TOKEN"], list)
    assert counts == {}


def test_sanitize_text_replaces_provider_tokens_and_jwts():
    github = "ghp_" + "A" * 36
    slack = "xoxb-" + "1" * 10 + "-" + "A" * 24
    jwt = "eyJ" + "A" * 12 + "." + "B" * 16 + "." + "C" * 20
    text = f"github={github} slack={slack} bearer={jwt}"

    cleaned, counts = sanitize_text(text)

    assert github not in cleaned
    assert slack not in cleaned
    assert jwt not in cleaned
    assert "<GITHUB_TOKEN_REDACTED>" in cleaned
    assert "<SLACK_TOKEN_REDACTED>" in cleaned
    assert "<JWT_REDACTED>" in cleaned
    assert counts == _counts(
        ("github_" + "token", 1),
        ("slack_" + "token", 1),
        ("jwt", 1),
    )


def test_sanitize_text_replaces_generic_assignments_and_credential_uris():
    generic = "opaque_" + "A" * 24
    generic_name = "api_" + "key"
    uri_password = "password-" + "B" * 16
    uri = "https://" + "release-user:" + uri_password + "@private.example.net/data"
    text = f'{generic_name}="{generic}" endpoint={uri}'

    cleaned, counts = sanitize_text(text)

    assert generic not in cleaned
    assert uri_password not in cleaned
    assert f'{generic_name}="<GENERIC_CREDENTIAL_REDACTED>"' in cleaned
    assert "<CREDENTIAL_URI_REDACTED>" in cleaned
    assert counts == _counts(
        ("credential_" + "uri", 1),
        ("generic_" + "credential", 1),
    )


def test_sanitize_text_replaces_prefixed_environment_credential_names():
    value = "synthetic_" + "A" * 12
    names = ["OPENAI_" + "API_KEY", "DATABASE_" + "PASSWORD"]

    for name in names:
        cleaned, counts = sanitize_text(f"{name}={value}")

        assert cleaned == f"{name}=<GENERIC_CREDENTIAL_REDACTED>"
        assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_replaces_short_generic_values():
    name = "api_" + "key"
    value = "x!"

    cleaned, counts = sanitize_text(f"{name}={value}")

    assert cleaned == f"{name}=<GENERIC_CREDENTIAL_REDACTED>"
    assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_replaces_entire_quoted_and_punctuated_values():
    quoted_name = "service_" + "token"
    quoted_value = "two words !#$%&*?,;[]{}()"
    bare_name = "api_" + "key"
    bare_value = "letters12" + "!$#%&*?()"
    text = f'{quoted_name}="{quoted_value}" {bare_name}={bare_value}'

    cleaned, counts = sanitize_text(text)

    assert cleaned == (
        f'{quoted_name}="<GENERIC_CREDENTIAL_REDACTED>" '
        f"{bare_name}=<GENERIC_CREDENTIAL_REDACTED>"
    )
    assert counts == _counts(("generic_" + "credential", 2))


def test_sanitize_text_replaces_entire_bare_values_with_delimiters():
    name = "SERVICE_" + "API_KEY"
    delimiters = [",", ";", "]", "}"]

    for delimiter in delimiters:
        value = "left" + delimiter + "right"
        cleaned, counts = sanitize_text(f"{name}={value}")

        assert cleaned == f"{name}=<GENERIC_CREDENTIAL_REDACTED>"
        assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_replaces_delimited_bare_values_before_same_line_prose():
    name = "SERVICE_" + "API_KEY"
    prose = "ordinary release prose"
    values = ["left,right", "left;right", "left]right", "left}right"]

    for value in values:
        cleaned, counts = sanitize_text(f"{name}={value} {prose}")

        assert cleaned == f"{name}=<GENERIC_CREDENTIAL_REDACTED> {prose}"
        assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_redacts_assignment_before_following_prose_line():
    name = "SERVICE_" + "PASSWORD"
    value = "x!"
    prose = "ordinary release prose"
    text = f"{name}={value}\n{prose}"

    cleaned, counts = sanitize_text(text)

    assert cleaned == f"{name}=<GENERIC_CREDENTIAL_REDACTED>\n{prose}"
    assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_replaces_public_ip_phone_internal_host_and_user_paths():
    public_ip = _dotted("8", "8", "8", "8")
    phone = _joined("+1 ", "415", "-867-5309")
    internal_host = _joined("database", ".corp")
    user_paths = [
        _joined("/Users/", "alice", "/project/file.json"),
        _joined("/home/", "bob", "/data.jsonl"),
        _joined("/workspace-vast/", "carol", "/run.log"),
    ]
    text = " ".join([public_ip, phone, internal_host, *user_paths])

    cleaned, counts = sanitize_text(text)

    for sensitive in [public_ip, phone, internal_host, *user_paths]:
        assert sensitive not in cleaned
    assert "<IP_ADDRESS_REDACTED>" in cleaned
    assert "<PHONE_REDACTED>" in cleaned
    assert "<INTERNAL_HOSTNAME_REDACTED>" in cleaned
    assert cleaned.count("<USER_PATH_REDACTED>") == 3
    assert counts == {
        "ipv4_address": 1,
        "phone_number": 1,
        "internal_hostname": 1,
        "user_path": 3,
    }


def test_sanitize_text_redacts_only_luhn_valid_cards_in_payment_context():
    valid_card = "4242" * 4
    invalid_card = valid_card[:-1] + "3"
    text = f"card={valid_card} reference={valid_card} payment={invalid_card}"

    cleaned, counts = sanitize_text(text)

    assert cleaned == (
        f"card=<PAYMENT_CARD_REDACTED> reference={valid_card} payment={invalid_card}"
    )
    assert counts == {"payment_card": 1}


def test_sanitize_text_preserves_approved_example_and_reserved_shapes():
    approved = [
        "developer" + "@example.com",
        "person" + "@service.test",
        "nobody" + "@domain.invalid",
        _dotted("192", "0", "2", "10"),
        _dotted("198", "51", "100", "20"),
        _dotted("203", "0", "113", "30"),
        _dotted("10", "0", "0", "1"),
        "202" + "-555-0123",
    ]
    text = " ".join(approved)

    cleaned, counts = sanitize_text(text)

    assert cleaned == text
    assert counts == {}


def test_sanitize_text_only_preserves_conventional_home_user_path():
    approved = [
        _joined("/home/", "user"),
        _joined("/home/", "user", "/test_cases.py"),
        _joined("/home/", "user", "/projects/release/file.py"),
        _joined("/home/", "user", "/.config/tool/settings.json"),
    ]
    sensitive = [
        _joined("/Users/", "user", "/test_cases.py"),
        _joined("/workspace-vast/", "user", "/test_cases.py"),
        _joined("/home/", "User", "/test_cases.py"),
        _joined("/home/", "user", "/../alice/private.txt"),
        _joined("/home/", "user", "/./alice/private.txt"),
        _joined("/home/", "user", "/%2e%2e/alice/private.txt"),
        _joined("/home/", "user", "/%252e%252e/alice/private.txt"),
        _joined("/home/", "user", "/projects%2falice/private.txt"),
        _joined("/home/", "user", "/projects%5c..%5calice/private.txt"),
        _joined(
            "/home/", "user", "/projects", chr(92), "..", chr(92), "alice/private.txt"
        ),
        _joined("/home/", "user", "/\uff0e\uff0e/alice/private.txt"),
        _joined("/home/", "user", "@example/private.txt"),
    ]
    text = " ".join([*approved, *sensitive])

    cleaned, counts = sanitize_text(text)

    assert all(path in cleaned for path in approved)
    assert all(path not in cleaned for path in sensitive)
    assert cleaned.count("<USER_PATH_REDACTED>") == len(sensitive)
    assert counts == {"user_path": len(sensitive)}


def test_sanitize_text_redacts_home_user_root_lookalikes_without_fragments():
    lookalikes = [
        _joined("/", "Home", "/user/private.txt"),
        _joined("/", "HOME", "/user/private.txt"),
        _joined("/users/", "user", "/private.txt"),
        _joined("prefix", "/home/", "user", "/private.txt"),
        _joined(
            "/",
            "\uff48\uff4f\uff4d\uff45",
            "/",
            "\uff55\uff53\uff45\uff52",
            "/private.txt",
        ),
        _joined("/home", "%2f", "user/private.txt"),
        _joined("/home", "%252525252f", "user/private.txt"),
        _joined("C:", chr(92), "Users", chr(92), "alice", chr(92), "private.txt"),
        _joined(
            chr(92),
            chr(92),
            "server",
            chr(92),
            "Users",
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined("prefix", "/Users/", "alice", "/private.txt"),
        _joined("/home/", "/alice/private.txt"),
        _joined("/users/", "/alice/private.txt"),
        _joined("/workspace-vast/", "/alice/private.txt"),
        _joined("file://", "/home/", "alice/private.txt"),
        _joined("file://", "/users/", "alice/private.txt"),
    ]
    ordinary_prose = _joined("The home", "/", "user distinction is generic prose.")
    safe_controls = [
        ordinary_prose,
        _joined("/users", "/"),
        _joined("/v1/users/", "user", "/settings"),
        _joined("/api/home/", "user-guide"),
    ]
    text = "\n".join([*lookalikes, *safe_controls])

    cleaned, counts = sanitize_text(text)

    assert cleaned.splitlines() == [
        *["<USER_PATH_REDACTED>"] * len(lookalikes),
        *safe_controls,
    ]
    assert all(value not in cleaned for value in lookalikes)
    assert counts == {"user_path": len(lookalikes)}


def test_sanitize_text_redacts_normalized_windows_unc_and_file_uri_user_paths():
    sensitive = [
        _joined(
            "c:",
            chr(92),
            "users",
            chr(92),
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined(
            chr(92),
            chr(92),
            "server",
            chr(92),
            "users",
            chr(92),
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined("file:///", "c:/users/", "alice/private.txt"),
    ]
    safe_controls = [
        _joined("/home/", "user", "/public.txt"),
        _joined("/users", "/"),
        _joined("/v1/users/", "user", "/settings"),
        _joined("/api/home/", "user-guide"),
    ]

    cleaned, counts = sanitize_text("\n".join([*sensitive, *safe_controls]))

    assert cleaned.splitlines() == [
        *["<USER_PATH_REDACTED>"] * len(sensitive),
        *safe_controls,
    ]
    assert all(value not in cleaned for value in sensitive)
    assert counts == {"user_path": len(sensitive)}


def test_sanitize_text_only_preserves_exact_builtin_github_token_expression():
    name = _joined("GITHUB_", "TOKEN")
    exact = _joined("${{", " github.token ", "}}")
    approved = [
        f"{name}: {exact}",
        f"{name}:   {exact}   ",
    ]
    near_misses = [
        _joined("prefix", exact),
        _joined(exact, "suffix"),
        _joined("prefix", exact, "suffix"),
        _joined("${{", " github.event.token ", "}}"),
    ]

    for text in approved:
        cleaned, counts = sanitize_text(text)

        assert cleaned == text
        assert counts == {}

    for value in near_misses:
        text = f"{name}: {value}"
        cleaned, counts = sanitize_text(text)

        assert value not in cleaned
        assert "<GENERIC_CREDENTIAL_REDACTED>" in cleaned
        assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_value_only_preserves_exact_builtin_github_token_expression():
    name = _joined("GITHUB_", "TOKEN")
    exact = _joined("${{", " github.token ", "}}")
    prefixed = _joined("prefix", exact)

    cleaned, counts = sanitize_value(
        {name: exact, _joined("BACKUP_", "TOKEN"): prefixed}
    )

    assert cleaned[name] == exact
    assert cleaned["BACKUP_TOKEN"] == "<GENERIC_CREDENTIAL_REDACTED>"
    assert counts == _counts(("generic_" + "credential", 1))


def test_sanitize_text_is_idempotent_for_typed_markers():
    token = "ghp_" + "A" * 36
    cleaned, first_counts = sanitize_text(token)

    cleaned_again, second_counts = sanitize_text(cleaned)

    assert first_counts == _counts(("github_" + "token", 1))
    assert cleaned_again == cleaned
    assert second_counts == {}


def test_sanitize_text_preserves_ordinary_source_code_mappings():
    source = "return {key: sanitize_nested(item) for key, item in nested.items()}"

    cleaned, counts = sanitize_text(source)

    assert cleaned == source
    assert counts == {}


def test_sanitize_text_preserves_incomplete_source_assignment_expressions():
    name = "to" + "ken"
    source = f'{name} = "prefix_" + suffix'

    cleaned, counts = sanitize_text(source)

    assert cleaned == source
    assert counts == {}


def test_scan_paths_returns_typed_deterministic_findings(tmp_path):
    aws = "AKIA" + "A" * 16
    github = "ghp_" + "B" * 36
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.txt"
    second = nested / "b.jsonl"
    token_key = "to" + "ken"
    first.write_text(f"safe\naccess={aws}\n", encoding="utf-8")
    second.write_text(json.dumps({token_key: github}) + "\n", encoding="utf-8")

    findings = scan_paths([tmp_path])

    assert findings == [
        Finding(
            path=str(first),
            kind="aws_access_key",
            line=2,
            fingerprint=hashlib.sha256(aws.encode()).hexdigest(),
        ),
        Finding(
            path=str(second),
            kind="github_token",
            line=1,
            fingerprint=hashlib.sha256(github.encode()).hexdigest(),
        ),
    ]


def test_scan_paths_git_repo_includes_tracked_ignored_names_and_skips_scratch(
    tmp_path,
):
    subprocess.run(
        ["git", "init", "-b", "main", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / ".gitignore").write_text(
        "work/\n.superpowers/\n*.egg-info/\nignored/\n",
        encoding="utf-8",
    )

    tracked_paths = [
        tmp_path / "work" / "tracked.txt",
        tmp_path / ".superpowers" / "tracked.txt",
        tmp_path / "package.egg-info" / "tracked.txt",
    ]
    tracked_secrets = ["AKIA" + letter * 16 for letter in ("W", "S", "M")]
    for path, secret in zip(tracked_paths, tracked_secrets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secret, encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "-f", str(path.relative_to(tmp_path))],
            check=True,
            capture_output=True,
            text=True,
        )

    public_path = tmp_path / "public.txt"
    public_secret = "AKIA" + "P" * 16
    public_path.write_text(public_secret, encoding="utf-8")
    ignored_path = tmp_path / "ignored" / "scratch.txt"
    ignored_path.parent.mkdir()
    ignored_path.write_text("AKIA" + "I" * 16, encoding="utf-8")
    ignored_named_scratch = tmp_path / "work" / "scratch.txt"
    ignored_named_scratch.write_text("AKIA" + "X" * 16, encoding="utf-8")

    expected = [
        Finding(
            path=str(path),
            kind="aws_access_key",
            line=1,
            fingerprint=hashlib.sha256(secret.encode()).hexdigest(),
        )
        for path, secret in [
            *zip(tracked_paths, tracked_secrets),
            (public_path, public_secret),
        ]
    ]

    assert scan_paths([tmp_path]) == sorted(
        expected,
        key=lambda finding: (
            finding.path,
            finding.line,
            finding.kind,
            finding.fingerprint,
        ),
    )


def test_scan_paths_git_repo_reads_absent_tracked_blob_from_clean_sparse_checkout(
    tmp_path,
):
    repo = tmp_path / "sparse-repo"
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Release Test"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "release-test@example.com",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    visible = repo / "visible" / "safe.txt"
    hidden = repo / "hidden" / "tracked.txt"
    visible.parent.mkdir()
    hidden.parent.mkdir()
    visible.write_text("release-safe\n", encoding="utf-8")
    secret = "AKIA" + "Q" * 16
    hidden.write_text(secret + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "visible/safe.txt", "hidden/tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "init", "--cone"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "set", "visible"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not hidden.exists()
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    assert scan_paths([repo]) == [
        Finding(
            path=str(hidden),
            kind="aws_access_key",
            line=1,
            fingerprint=hashlib.sha256(secret.encode()).hexdigest(),
        )
    ]


def test_scan_paths_git_repo_reads_committed_blob_when_skip_worktree_is_safe(
    tmp_path,
):
    repo = tmp_path / "skip-worktree-repo"
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Release Test"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "release-test@example.com",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = repo / "tracked.txt"
    committed_secret = "AKIA" + "K" * 16
    tracked.write_text(committed_secret + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "unsafe fixture"],
        check=True,
        capture_output=True,
        text=True,
    )

    tracked.write_text("release-safe\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--skip-worktree", "tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    assert scan_paths([repo]) == [
        Finding(
            path=str(tracked),
            kind="aws_access_key",
            line=1,
            fingerprint=hashlib.sha256(committed_secret.encode()).hexdigest(),
        )
    ]


def test_scan_paths_git_repo_does_not_follow_tracked_symlink(tmp_path):
    repo = tmp_path / "symlink-repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    outside = tmp_path / "outside.txt"
    outside.write_text("AKIA" + "Y" * 16 + "\n", encoding="utf-8")
    link = repo / "public-link"
    link.symlink_to(Path("..") / outside.name)
    subprocess.run(
        ["git", "add", "public-link"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "symlink fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    assert scan_paths([repo]) == []


def test_scan_paths_git_repo_fails_closed_when_head_blob_is_unreadable(tmp_path):
    repo = tmp_path / "corrupt-object-repo"
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Release Test"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "release-test@example.com",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = repo / "tracked.txt"
    tracked.write_text("release fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
        text=True,
    )
    object_id = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD:tracked.txt"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.unlink()
    (repo / ".git" / "objects" / object_id[:2] / object_id[2:]).unlink()

    with pytest.raises(
        RuntimeError,
        match=r"^failed to read tracked HEAD blob: tracked\.txt$",
    ):
        scan_paths([repo])


def test_scan_paths_non_git_tree_does_not_skip_exportable_basenames(tmp_path):
    paths = [
        tmp_path / "work" / "release.txt",
        tmp_path / ".superpowers" / "release.txt",
        tmp_path / "package.egg-info" / "release.txt",
    ]
    secrets = ["AKIA" + letter * 16 for letter in ("W", "S", "M")]
    for path, secret in zip(paths, secrets):
        path.parent.mkdir(parents=True)
        path.write_text(secret, encoding="utf-8")

    assert scan_paths([tmp_path]) == sorted(
        [
            Finding(
                path=str(path),
                kind="aws_access_key",
                line=1,
                fingerprint=hashlib.sha256(secret.encode()).hexdigest(),
            )
            for path, secret in zip(paths, secrets)
        ],
        key=lambda finding: (
            finding.path,
            finding.line,
            finding.kind,
            finding.fingerprint,
        ),
    )


def test_scan_paths_python_source_distinguishes_references_from_literal_secrets(
    tmp_path,
):
    literal_secrets = {
        1: "default source credential",
        6: "assigned source credential",
        7: "mapping source credential",
        8: "keyword source credential",
        9: "annotated source credential",
        10: "environment source credential",
    }
    escaped_source = chr(92) + "n" + "@app.route"
    path = tmp_path / "client.py"
    path.write_text(
        "\n".join(
            [
                f"def call(api_key={literal_secrets[1]!r}):",
                '    """api_key: Override; defaults to the environment."""',
                '    headers = {"x-api-key": api_key}',
                "    request(api_key=clients.auditor_key)",
                f'    escaped = "{escaped_source}"',
                f"    configured_password = {literal_secrets[6]!r}",
                f'    headers = {{"x-api-key": {literal_secrets[7]!r}}}',
                f"    request(api_key={literal_secrets[8]!r})",
                f"    annotated_password: str = {literal_secrets[9]!r}",
                f'    os.environ["API_KEY"] = {literal_secrets[10]!r}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(secret.encode()).hexdigest(),
        )
        for line, secret in literal_secrets.items()
    ]


def test_scan_paths_python_source_detects_static_string_expressions(tmp_path):
    credential_names = [
        _joined("api_", "key"),
        _joined("service_", "password"),
        _joined("backup_", "token"),
    ]
    secrets = [
        "static f-string secret",
        "static interpolated secret",
        "static concatenated secret",
    ]
    path = tmp_path / "static_credentials.py"
    path.write_text(
        "\n".join(
            [
                f'{credential_names[0]} = f"{secrets[0]}"',
                f"{credential_names[1]} = f\"static {{'interpolated secret'}}\"",
                f'{credential_names[2]} = "static " + "concatenated secret"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(secret.encode()).hexdigest(),
        )
        for line, secret in enumerate(secrets, start=1)
    ]


def test_scan_paths_python_source_evaluates_nested_static_fstring_atoms(tmp_path):
    credential_names = [
        _joined("SERVICE_", "TOKEN"),
        _joined("BACKUP_", "TOKEN"),
        _joined("RUNTIME_", "TOKEN"),
        _joined("NEGATIVE_", "TOKEN"),
    ]
    parts = ["static ", "secret"]
    nested_secret = _joined(*parts)
    numeric_value = "3"
    path = tmp_path / "nested_static_credentials.py"
    path.write_text(
        "\n".join(
            [
                f'{credential_names[0]} = f"{{{parts[0]!r} + {parts[1]!r}}}"',
                f'{credential_names[1]} = f"{{1 + 2}}"',
                f'{credential_names[2]} = f"{{runtime_value + {parts[1]!r}}}"',
                f'{credential_names[3]} = f"{{-1}}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in [(1, nested_secret), (2, numeric_value), (4, "-1")]
    ]


def test_scan_paths_python_source_scans_complete_static_config_literals(tmp_path):
    names = [_joined("SERVICE_", "TOKEN"), _joined("BACKUP_", "TOKEN")]
    values = [
        "embeddedvalue",
        "embedded-value",
        "yamlvalue",
        "quotedvalue",
        "shellvalue",
        "dictvalue",
        "multilinevalue",
        "multiline-value",
    ]
    static_payloads = [
        f"{names[0]}={values[0]}",
        f"{names[0]}={values[1]}",
        f"{names[0]}: {values[2]}",
        f'{names[0]}="{values[3]}"',
        f"export {names[0]}={values[4]}",
        repr({names[0]: values[5]}),
    ]
    multiline_payload = _joined(
        names[0],
        "=",
        values[6],
        "\n",
        names[1],
        ": ",
        values[7],
    )
    path = tmp_path / "static_configuration_literals.py"
    path.write_text(
        "\n".join(
            [
                *[
                    f"configuration_{index} = {payload!r}"
                    for index, payload in enumerate(static_payloads, start=1)
                ],
                f"configuration_multiline = {multiline_payload!r}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in [
            (1, values[0]),
            (2, values[1]),
            (3, values[2]),
            (4, values[3]),
            (5, values[4]),
            (6, values[5]),
            (7, values[6]),
            (8, values[7]),
        ]
    ]


def test_scan_paths_python_source_detects_spaced_ini_assignment(tmp_path):
    name = _joined("SERVICE_", "TOKEN")
    value = "embedded-value"
    ini_payload = _joined("[service]\n", name, " = ", value)
    path = tmp_path / "configuration.py"
    path.write_text(
        "\n".join(
            [
                f"configuration = {ini_payload!r}",
                f"{name} = embedded - value",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=2,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
    ]


def test_scan_paths_python_source_detects_static_format_template_segments(tmp_path):
    names = [_joined("SERVICE_", "TOKEN"), _joined("BACKUP_", "TOKEN")]
    values = ["static-format-only", "static-format-mixed", "static-format-after"]
    path = tmp_path / "static_format_templates.py"
    path.write_text(
        "\n".join(
            [
                f'configuration = "{names[0]}={values[0]}".format()',
                'mixed = """[service]',
                f"{names[0]} = {values[1]}",
                f'{names[1]} = {{}}""".format(runtime_value)',
                'reverse = """[service]',
                f"{names[1]} = {{}}",
                f'{names[0]} = {values[2]}""".format(runtime_value)',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in [(1, values[0]), (3, values[1]), (7, values[2])]
    ]


def test_scan_paths_python_source_detects_static_percent_and_template_segments(
    tmp_path,
):
    names = [_joined("SERVICE_", "TOKEN"), _joined("BACKUP_", "TOKEN")]
    values = [
        "static-percent-only",
        "static-percent-mixed",
        "static-template-only",
        "static-template-mixed",
        "static-percent-after",
        "static-template-after",
    ]
    path = tmp_path / "static_percent_and_template.py"
    path.write_text(
        "\n".join(
            [
                "from string import Template",
                f'percent_static = "{names[0]}={values[0]}" % ()',
                'percent_mixed = """[service]',
                f"{names[0]} = {values[1]}",
                f'{names[1]} = %s""" % runtime_value',
                f'template_static = Template("{names[0]}={values[2]}").substitute()',
                'template_mixed = Template("""[service]',
                f"{names[0]} = {values[3]}",
                f'{names[1]} = $value""").substitute(value=runtime_value)',
                'percent_reverse = """[service]',
                f"{names[1]} = %s",
                f'{names[0]} = {values[4]}""" % runtime_value',
                'template_reverse = Template("""[service]',
                f"{names[1]} = $value",
                f'{names[0]} = {values[5]}""").substitute(value=runtime_value)',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in [
            (2, values[0]),
            (4, values[1]),
            (6, values[2]),
            (8, values[3]),
            (12, values[4]),
            (15, values[5]),
        ]
    ]


def test_percent_template_static_segments_cover_full_grammar_and_line_offsets():
    value = "head%%\nA=%(service)s\nB=%+#08.2Lf\nC=%*.*d\nD=%r\nTAIL"

    assert release_safety._percent_template_static_segments(value) == [
        (0, "head%\nA="),
        (1, "\nB="),
        (2, "\nC="),
        (3, "\nD="),
        (4, "\nTAIL"),
    ]


@pytest.mark.parametrize("conversion", "diouxXeEfFgGcrsa")
def test_percent_template_static_segments_support_every_conversion(conversion):
    assert release_safety._percent_template_static_segments(
        f"left %{conversion} right"
    ) == [(0, "left "), (0, " right")]


def test_percent_template_static_segments_keep_malformed_mapping_static():
    value = "before %(broken\nmiddle %s after"

    assert release_safety._percent_template_static_segments(value) == [
        (0, "before %(broken\nmiddle "),
        (1, " after"),
    ]


def test_percent_template_malformed_mapping_has_bounded_scan_steps():
    value = "%(" * 4096
    parse_with_steps = getattr(
        release_safety,
        "_percent_template_static_segments_with_steps",
        None,
    )

    assert parse_with_steps is not None
    segments, steps = parse_with_steps(value)
    assert segments == [(0, value)]
    assert steps <= len(value) * 3


def test_scan_paths_python_source_ignores_runtime_formatting_placeholders(tmp_path):
    name = _joined("SERVICE_", "TOKEN")
    path = tmp_path / "runtime_formatting.py"
    path.write_text(
        "\n".join(
            [
                "from string import Template",
                f'formatted = "{name}={{}}".format(runtime_value)',
                f'percent = "{name}=%s" % runtime_value',
                f'multiline = """[service]\n{name} = {{}}""".format(runtime_value)',
                f'templated = Template("{name}=$value").substitute(value=runtime_value)',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == []


def test_scan_paths_detects_home_user_root_lookalikes(tmp_path):
    lookalikes = [
        _joined("/", "Home", "/user/private.txt"),
        _joined("/", "HOME", "/user/private.txt"),
        _joined("/users/", "user", "/private.txt"),
        _joined("prefix", "/home/", "user", "/private.txt"),
        _joined(
            "/",
            "\uff48\uff4f\uff4d\uff45",
            "/",
            "\uff55\uff53\uff45\uff52",
            "/private.txt",
        ),
        _joined("/home", "%2f", "user/private.txt"),
        _joined("/home", "%252525252f", "user/private.txt"),
        _joined("C:", chr(92), "Users", chr(92), "alice", chr(92), "private.txt"),
        _joined(
            chr(92),
            chr(92),
            "server",
            chr(92),
            "Users",
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined("prefix", "/Users/", "alice", "/private.txt"),
        _joined("/home/", "/alice/private.txt"),
        _joined("/users/", "/alice/private.txt"),
        _joined("/workspace-vast/", "/alice/private.txt"),
        _joined("file://", "/home/", "alice/private.txt"),
        _joined("file://", "/users/", "alice/private.txt"),
    ]
    approved = _joined("/home/", "user", "/public.txt")
    ordinary_prose = _joined("The home", "/", "user distinction is generic prose.")
    safe_controls = [
        approved,
        ordinary_prose,
        _joined("/users", "/"),
        _joined("/v1/users/", "user", "/settings"),
        _joined("/api/home/", "user-guide"),
    ]
    path = tmp_path / "paths.txt"
    path.write_text(
        "\n".join([*lookalikes, *safe_controls]) + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="user_path",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in enumerate(lookalikes, start=1)
    ]


def test_scan_paths_detects_normalized_windows_unc_and_file_uri_user_paths(tmp_path):
    sensitive = [
        _joined(
            "c:",
            chr(92),
            "users",
            chr(92),
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined(
            chr(92),
            chr(92),
            "server",
            chr(92),
            "users",
            chr(92),
            chr(92),
            "alice",
            chr(92),
            "private.txt",
        ),
        _joined("file:///", "c:/users/", "alice/private.txt"),
    ]
    safe_controls = [
        _joined("/home/", "user", "/public.txt"),
        _joined("/users", "/"),
        _joined("/v1/users/", "user", "/settings"),
        _joined("/api/home/", "user-guide"),
    ]
    path = tmp_path / "normalized_user_paths.txt"
    path.write_text(
        "\n".join([*sensitive, *safe_controls]) + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="user_path",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
        for line, value in enumerate(sensitive, start=1)
    ]


def test_scan_paths_deduplicates_identical_nested_config_findings(tmp_path):
    name = _joined("SERVICE_", "TOKEN")
    value = "duplicate-static-value"
    invalid_payload = _joined(name, "=", value, "\nnot python {")
    path = tmp_path / "nested_configuration.py"
    path.write_text(f"configuration = {invalid_payload!r}\n", encoding="utf-8")

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=1,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
    ]


def test_scan_paths_python_source_ignores_multiline_runtime_config_snippets(
    tmp_path,
):
    names = [_joined("SERVICE_", "TOKEN"), _joined("BACKUP_", "TOKEN")]
    path = tmp_path / "runtime_configuration_snippets.py"
    path.write_text(
        "\n".join(
            [
                f'runtime_f = f"{names[0]}={{runtime_value}}\\n{names[1]}={{other_value}}"',
                f'runtime_add = "{names[0]}=" + runtime_value + "\\n{names[1]}=" + other_value',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == []


def test_scan_paths_python_source_scans_embedded_config_not_docs_or_runtime(
    tmp_path,
):
    name = _joined("SERVICE_", "TOKEN")
    embedded_value = "embedded-configuration-secret"
    embedded_config = f"{name}={embedded_value!r}"
    path = tmp_path / "configuration.py"
    path.write_text(
        "\n".join(
            [
                f'"""{name}=documentation example"""',
                f"configuration = {embedded_config!r}",
                f'runtime_f = f"{name}={{runtime_value}}"',
                f'runtime_add = "{name}=" + runtime_value',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=2,
            fingerprint=hashlib.sha256(embedded_value.encode()).hexdigest(),
        )
    ]


def test_scan_paths_python_source_preserves_embedded_python_references(tmp_path):
    name = _joined("api_", "key")
    snippets = [
        _joined('    headers = {"x-', name, '": ', name, "}"),
        _joined("    request(", name, "=clients.auditor_key)"),
        _joined(
            "    ",
            '"""',
            name,
            ": Override; defaults to the environment.",
            '"""',
        ),
    ]
    path = tmp_path / "embedded_source.py"
    path.write_text(
        "\n".join(
            f"source_{index} = {snippet!r}"
            for index, snippet in enumerate(snippets, start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == []


def test_scan_paths_python_source_preserves_pii_detection_in_semantic_strings(
    tmp_path,
):
    doc_email = _joined("doc-owner", "@", "gmail.com")
    runtime_email = _joined("runtime-owner", "@", "gmail.com")
    path = tmp_path / "contacts.py"
    path.write_text(
        "\n".join(
            [
                f'"""Contact {doc_email}."""',
                f'runtime = f"{{name}} may contact {runtime_email}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="email",
            line=line,
            fingerprint=hashlib.sha256(email.encode()).hexdigest(),
        )
        for line, email in enumerate((doc_email, runtime_email), start=1)
    ]


def test_scan_paths_does_not_report_approved_shapes(tmp_path):
    approved = " ".join(
        [
            "developer" + "@example.com",
            _dotted("192", "0", "2", "10"),
            "202" + "-555-0123",
        ]
    )
    path = tmp_path / "examples.txt"
    path.write_text(approved, encoding="utf-8")

    assert scan_paths([path]) == []


def test_scan_paths_decodes_json_strings_before_matching(tmp_path):
    source = "\n@app.route"
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"source": source}), encoding="utf-8")

    assert sanitize_text(source)[1] == {}
    assert scan_paths([path]) == []


def test_scan_paths_fails_closed_for_unredacted_credential_field_types(tmp_path):
    unrecognized_marker = "<UNRECOGNIZED_OPAQUE>"
    value = {
        "SERVICE_API_KEY": unrecognized_marker,
        "SERVICE_PASSWORD": 7,
        "ACCESS_TOKEN": ["part-a", "part-b"],
        "REDACTED_TOKEN": "<GENERIC_CREDENTIAL_REDACTED>",
    }
    path = tmp_path / "credential-fields.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    findings = scan_paths([path])

    assert len(findings) == 3
    assert {(finding.kind, finding.fingerprint) for finding in findings} == {
        (
            "generic_credential",
            hashlib.sha256(unrecognized_marker.encode()).hexdigest(),
        ),
        ("generic_credential", hashlib.sha256(b"7").hexdigest()),
        (
            "generic_credential",
            hashlib.sha256(b'["part-a","part-b"]').hexdigest(),
        ),
    }
    assert all(finding.path == str(path) for finding in findings)
    assert all(finding.line == 1 for finding in findings)


def test_scan_paths_fingerprints_entire_generic_credential_value(tmp_path):
    name = "DATABASE_" + "PASSWORD"
    value = "tiny value !@#$"
    path = tmp_path / "environment.txt"
    path.write_text(f'{name}="{value}"\n', encoding="utf-8")

    assert scan_paths([path]) == [
        Finding(
            path=str(path),
            kind="generic_credential",
            line=1,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )
    ]


def test_scan_paths_fingerprints_entire_bare_values_with_delimiters(tmp_path):
    name = "SERVICE_" + "API_KEY"
    delimiters = [",", ";", "]", "}"]
    values = ["left" + delimiter + "right" for delimiter in delimiters]
    path = tmp_path / "environment.txt"
    path.write_text(
        "\n".join(f"{name}={value}" for value in values) + "\n",
        encoding="utf-8",
    )

    findings = scan_paths([path])

    assert len(findings) == len(values)
    for line, (value, finding) in enumerate(zip(values, findings), start=1):
        assert finding == Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )


def test_scan_paths_fingerprints_delimited_bare_values_before_same_line_prose(
    tmp_path,
):
    name = "SERVICE_" + "API_KEY"
    prose = "ordinary release prose"
    values = ["left,right", "left;right", "left]right", "left}right"]
    path = tmp_path / "environment.txt"
    path.write_text(
        "\n".join(f"{name}={value} {prose}" for value in values) + "\n",
        encoding="utf-8",
    )

    findings = scan_paths([path])

    assert len(findings) == len(values)
    for line, (value, finding) in enumerate(zip(values, findings), start=1):
        assert finding == Finding(
            path=str(path),
            kind="generic_credential",
            line=line,
            fingerprint=hashlib.sha256(value.encode()).hexdigest(),
        )


def test_release_safety_cli_reports_metadata_only_and_fails_closed(tmp_path):
    secret = "AKIA" + "C" * 16
    path = tmp_path / "unsafe.txt"
    path.write_text(f"value={secret}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RELEASE_SAFETY_SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    assert secret not in result.stdout
    assert json.loads(result.stdout) == {
        "path": str(path),
        "kind": "aws_access_key",
        "line": 1,
        "fingerprint": hashlib.sha256(secret.encode()).hexdigest()[:12],
    }


def test_release_safety_cli_succeeds_silently_for_clean_input(tmp_path):
    path = tmp_path / "safe.txt"
    path.write_text("release-safe example\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RELEASE_SAFETY_SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_safety_cli_rejects_whole_filesystem_release_residue(tmp_path):
    residue_directories = [
        "package.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        ".venv312",
        "venv312",
        "package.dist-info",
        "test-results",
        "test_results",
        "test-results-312",
        "reports",
        "results",
        "results_smoke",
        "logs",
    ]
    residue_files = [
        "module.pyc",
        ".coverage",
        ".coverage.worker-1",
        "coverage.xml",
        "runner.log",
    ]
    sentinel = "private-build-content"
    expected_paths: list[str] = []
    for name in residue_directories:
        path = tmp_path / name
        path.mkdir()
        (path / "payload.txt").write_text(sentinel, encoding="utf-8")
        expected_paths.append(str(path))
    for name in residue_files:
        path = tmp_path / name
        path.write_text(sentinel, encoding="utf-8")
        expected_paths.append(str(path))

    git_internal = tmp_path / ".git" / "objects" / "cache.pyc"
    git_internal.parent.mkdir(parents=True)
    git_internal.write_text(sentinel, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(RELEASE_SAFETY_SCRIPT),
            "--check-release-residue",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    assert sentinel not in result.stdout
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"kind": "release_residue", "path": path} for path in sorted(expected_paths)
    ]


def test_release_safety_cli_release_residue_gate_is_silent_for_clean_tree(
    tmp_path,
):
    (tmp_path / "safe.txt").write_text("release-safe\n", encoding="utf-8")
    for name in ["venv_docs", "contest-results", "package.dist-information"]:
        path = tmp_path / name
        path.mkdir()
        (path / "safe.txt").write_text("release-safe\n", encoding="utf-8")
    (tmp_path / ".coverage-notes").write_text("release-safe\n", encoding="utf-8")
    git_internal = tmp_path / ".git" / "objects" / "cache.pyc"
    git_internal.parent.mkdir(parents=True)
    git_internal.write_text("git-internal-control", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(RELEASE_SAFETY_SCRIPT),
            "--check-release-residue",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_release_safety_cli_suppresses_python_literal_parse_warnings(tmp_path):
    backslash = chr(92)
    path = tmp_path / "pattern.py"
    path.write_text(f'pattern = "{backslash}d"\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RELEASE_SAFETY_SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_sanitize_release_data_cli_rewrites_json_and_jsonl_deterministically(
    tmp_path,
):
    aws = "AKIA" + "D" * 16
    email = _joined("researcher", "@", "gmail.com")
    nested = tmp_path / "nested"
    nested.mkdir()
    json_path = tmp_path / "records.json"
    jsonl_path = nested / "records.jsonl"
    untouched_path = tmp_path / "notes.txt"
    json_path.write_text(
        json.dumps({"z": "safe", "a": aws}, separators=(",", ":")),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        json.dumps({"z": 1, "contact": email}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    untouched_path.write_text(aws, encoding="utf-8")

    first = subprocess.run(
        [
            sys.executable,
            str(SANITIZE_RELEASE_DATA_SCRIPT),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0
    assert first.stderr == ""
    assert aws not in first.stdout
    assert email not in first.stdout
    assert json.loads(first.stdout) == _counts(
        ("aws_access_" + "key", 1),
        ("email", 1),
    )
    assert json_path.read_text(encoding="utf-8") == (
        '{\n  "a": "<AWS_ACCESS_KEY_REDACTED>",\n  "z": "safe"\n}\n'
    )
    assert jsonl_path.read_text(encoding="utf-8") == (
        '{"contact":"<EMAIL_REDACTED>","z":1}\n'
    )
    assert untouched_path.read_text(encoding="utf-8") == aws

    canonical_json = json_path.read_text(encoding="utf-8")
    canonical_jsonl = jsonl_path.read_text(encoding="utf-8")
    second = subprocess.run(
        [
            sys.executable,
            str(SANITIZE_RELEASE_DATA_SCRIPT),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0
    assert second.stderr == ""
    assert json.loads(second.stdout) == {}
    assert json_path.read_text(encoding="utf-8") == canonical_json
    assert jsonl_path.read_text(encoding="utf-8") == canonical_jsonl


def test_sanitize_release_data_cli_skips_nested_git_worktrees(tmp_path):
    aws = "AKIA" + "E" * 16
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps({"credential": aws}), encoding="utf-8")
    nested_repo = tmp_path / "external" / "dependency"
    nested_repo.mkdir(parents=True)
    (nested_repo / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    nested_config = nested_repo / "tsconfig.json"
    original_config = '{\n  "compilerOptions": {}, // JSON-with-comments\n}\n'
    nested_config.write_text(original_config, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SANITIZE_RELEASE_DATA_SCRIPT),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == _counts(("aws_access_" + "key", 1))
    assert aws not in release_path.read_text(encoding="utf-8")
    assert nested_config.read_text(encoding="utf-8") == original_config


def test_sanitize_release_data_cli_replaces_wildchat_source_ids(tmp_path):
    wildchat_path = tmp_path / "data" / "wildchat_blog_20.jsonl"
    wildchat_path.parent.mkdir()
    rows = [
        {
            "session_id": f"source-{index}",
            "source": "WildChat",
            "char_count": len(transcript),
            "transcript": transcript,
        }
        for index, transcript in enumerate(("first transcript", "second transcript"))
    ]
    wildchat_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SANITIZE_RELEASE_DATA_SCRIPT),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {}
    sanitized_rows = [
        json.loads(line)
        for line in wildchat_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in sanitized_rows] == [
        "wildchat-sanitized-001",
        "wildchat-sanitized-002",
    ]
    assert all("session_id" not in row for row in sanitized_rows)
    assert [row["source"] for row in sanitized_rows] == [row["source"] for row in rows]
    assert [row["char_count"] for row in sanitized_rows] == [
        row["char_count"] for row in rows
    ]
    assert [row["transcript"] for row in sanitized_rows] == [
        row["transcript"] for row in rows
    ]
