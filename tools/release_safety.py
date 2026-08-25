"""Detect and sanitize credential and personal-data shapes before release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import ipaddress
import json
import os
import re
import string
import subprocess
import sys
import textwrap
import tokenize
import unicodedata
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Finding:
    path: str
    kind: str
    line: int
    fingerprint: str


@dataclass(frozen=True)
class _ScanFile:
    path: Path
    git_root: Path | None = None
    git_path: str | None = None
    head_blob: str | None = None


@dataclass(frozen=True)
class _SanitizerRule:
    kind: str
    pattern: re.Pattern[str]
    marker: str
    redact_group: str | tuple[str, ...] | None = None
    should_redact: Callable[[str], bool] | None = None


def _redact_email(value: str) -> bool:
    domain = value.rsplit("@", 1)[-1].lower().rstrip(".")
    return not (
        domain in {"example.com", "test", "invalid"}
        or domain.endswith((".example.com", ".test", ".invalid"))
    )


def _redact_ipv4(value: str) -> bool:
    return ipaddress.ip_address(value).is_global


def _redact_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return False
    exchange = digits[3:6]
    subscriber = int(digits[6:])
    return not (exchange == "555" and 100 <= subscriber <= 199)


_PERCENT_ESCAPE_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
_WINDOWS_USER_PATH_PATTERN = re.compile(
    r"^(?:/?[a-z]:|//[^/]+)/users/[^/]+(?:/|$)",
    re.IGNORECASE,
)


def _is_approved_home_user_path(value: str) -> bool:
    if "\\" in value or _PERCENT_ESCAPE_PATTERN.search(value):
        return False

    parts = value.split("/")
    if len(parts) < 3 or parts[:3] != ["", "home", "user"]:
        return False

    descendants = parts[3:]
    if descendants and descendants[-1] == "":
        descendants = descendants[:-1]
    for part in descendants:
        normalized = unicodedata.normalize("NFKC", part)
        if normalized in {"", ".", ".."} or "/" in normalized or "\\" in normalized:
            return False
    return True


def _normalized_user_path_candidate(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    for _ in range(min(len(normalized) + 1, 64)):
        decoded = unicodedata.normalize(
            "NFKC",
            unquote(normalized, errors="replace"),
        )
        if decoded == normalized:
            break
        normalized = decoded
    return normalized.replace("\\", "/")


def _collapse_user_path_separators(value: str) -> str:
    if value.startswith("//"):
        return "//" + re.sub(r"/+", "/", value.lstrip("/"))
    return re.sub(r"/+", "/", value)


def _has_populated_posix_user_root(value: str) -> bool:
    if not value.startswith("/"):
        return False
    parts = [part for part in value.split("/") if part]
    return len(parts) >= 2 and parts[0] in {"home", "users", "workspace-vast"}


def _redact_user_path(value: str) -> bool:
    if _is_approved_home_user_path(value):
        return False

    normalized = _normalized_user_path_candidate(value)
    if _PERCENT_ESCAPE_PATTERN.search(normalized):
        return True

    canonical = _collapse_user_path_separators(normalized)
    folded = canonical.casefold()
    starts_with_user_root = _has_populated_posix_user_root(folded)
    home_user_prefix = re.search(r"/home/user(?:/|$)", folded) is not None
    segments = canonical.split("/")
    qualified_users_prefix = any(
        segment.casefold() == "users"
        and segment != "users"
        and any(segments[index + 1 :])
        for index, segment in enumerate(segments)
    )
    file_user_path = False
    if folded.startswith("file:"):
        try:
            file_path = _collapse_user_path_separators(
                urlsplit(normalized).path
            ).casefold()
        except ValueError:
            return True
        file_user_path = (
            _has_populated_posix_user_root(file_path)
            or _WINDOWS_USER_PATH_PATTERN.match(file_path) is not None
        )
    return (
        starts_with_user_root
        or home_user_prefix
        or qualified_users_prefix
        or file_user_path
        or _WINDOWS_USER_PATH_PATTERN.match(folded) is not None
    )


_BUILTIN_GITHUB_TOKEN_EXPRESSION = re.compile(r"\s*\$\{\{\s*github\.token\s*\}\}\s*")


def _is_builtin_github_token_expression(value: object) -> bool:
    return (
        isinstance(value, str)
        and _BUILTIN_GITHUB_TOKEN_EXPRESSION.fullmatch(value) is not None
    )


def _redact_generic_credential(value: str) -> bool:
    return not _is_builtin_github_token_expression(value)


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_SANITIZER_RULES: tuple[_SanitizerRule, ...] = (
    _SanitizerRule(
        "credential_uri",
        re.compile(
            r"(?i)\b[a-z][a-z0-9+.-]*://"
            r"[^\s/:@]+:[^\s/@]+@[^\s]+"
        ),
        "<CREDENTIAL_URI_REDACTED>",
    ),
    _SanitizerRule(
        "slack_webhook",
        re.compile(
            r"https://hooks\.slack\.com/services/"
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+"
        ),
        "<SLACK_WEBHOOK_REDACTED>",
    ),
    _SanitizerRule(
        "aws_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "<AWS_ACCESS_KEY_REDACTED>",
    ),
    _SanitizerRule(
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            r"gh[pousr]_[A-Za-z0-9]{36,255}|"
            r"github_pat_[A-Za-z0-9_]{22,255}"
            r")(?![A-Za-z0-9_])"
        ),
        "<GITHUB_TOKEN_REDACTED>",
    ),
    _SanitizerRule(
        "payment_token",
        re.compile(
            r"(?<![A-Za-z0-9])(?:sk|rk|pk)_(?:live|test)_"
            r"[A-Za-z0-9]{10,}(?![A-Za-z0-9])"
        ),
        "<PAYMENT_TOKEN_REDACTED>",
    ),
    _SanitizerRule(
        "slack_token",
        re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
        "<SLACK_TOKEN_REDACTED>",
    ),
    _SanitizerRule(
        "jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
            r"(?![A-Za-z0-9_-])"
        ),
        "<JWT_REDACTED>",
    ),
    _SanitizerRule(
        "payment_card",
        re.compile(
            r"(?i)\b(?:credit[\s_-]?card|card(?:[\s_-]?(?:number|no))?|"
            r"payment(?:[\s_-]?card)?|pan)\b\s*(?:=|:)?\s*"
            r"(?P<value>(?:\d[ -]?){12,18}\d)(?!\d)"
        ),
        "<PAYMENT_CARD_REDACTED>",
        redact_group="value",
        should_redact=_luhn_valid,
    ),
    _SanitizerRule(
        "email",
        re.compile(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
            r"(?![A-Za-z0-9-])"
        ),
        "<EMAIL_REDACTED>",
        should_redact=_redact_email,
    ),
    _SanitizerRule(
        "ipv4_address",
        re.compile(
            r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
        ),
        "<IP_ADDRESS_REDACTED>",
        should_redact=_redact_ipv4,
    ),
    _SanitizerRule(
        "phone_number",
        re.compile(
            r"(?<![A-Za-z0-9])(?P<value>"
            r"(?:\+?1[ .-])?\(?[2-9]\d{2}\)?[ .-]"
            r"\d{3}[ .-]\d{4})(?![A-Za-z0-9])"
        ),
        "<PHONE_REDACTED>",
        redact_group="value",
        should_redact=_redact_phone,
    ),
    _SanitizerRule(
        "internal_hostname",
        re.compile(
            r"(?i)(?<![A-Za-z0-9.-])"
            r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
            r"(?:internal|corp|lan|local|intranet)"
            r"(?![A-Za-z0-9.-])"
        ),
        "<INTERNAL_HOSTNAME_REDACTED>",
    ),
    _SanitizerRule(
        "user_path",
        re.compile(
            r"(?<![^\s\"'`<>(){}\[\],;])"
            r"(?=[^\s\"'`<>(){}\[\],;]*[/\\%／＼％])"
            r"[^\s\"'`<>(){}\[\],;]+"
        ),
        "<USER_PATH_REDACTED>",
        should_redact=_redact_user_path,
    ),
    _SanitizerRule(
        "generic_credential",
        re.compile(
            r"(?i)(?P<prefix>(?<![A-Za-z0-9_])"
            r"\"?(?:[A-Za-z_][A-Za-z0-9_-]*)?"
            r"(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|"
            r"password|passwd|pwd|token|credential)\"?"
            r"\s*(?:=|:)\s*)"
            r"(?:"
            r"(?P<github_token_expression_value>"
            r"[^ \t\r\n\"'#]*\$\{\{\s*github\.token\s*\}\}"
            r"[^ \t\r\n\"'#]*)"
            r"(?=[ \t]*(?:\r?\n|$|[,;}\])]|#|"
            r"[A-Za-z_][A-Za-z0-9_-]*\s*(?:=|:)))|"
            r"(?:"
            r"\"(?P<double_quoted_value>(?!<)(?:\\.|[^\"\\\r\n])+)\"|"
            r"'(?P<single_quoted_value>(?!<)(?:\\.|[^'\\\r\n])+)'"
            r")"
            r"(?=[ \t]*(?:\r?\n|$|[,;}\])]|#|"
            r"[A-Za-z_][A-Za-z0-9_-]*\s*(?:=|:)))|"
            r"(?P<bare_value>(?!<)[^\s\"']+)"
            r"(?=(?:[ \t]*(?:\r?\n|$|#)|[ \t]+[A-Za-z0-9_]))"
            r")"
        ),
        "<GENERIC_CREDENTIAL_REDACTED>",
        redact_group=(
            "github_token_expression_value",
            "double_quoted_value",
            "single_quoted_value",
            "bare_value",
        ),
        should_redact=_redact_generic_credential,
    ),
)


_CREDENTIAL_KEY_PATTERN = re.compile(
    r"(?i)^(?:[A-Za-z_][A-Za-z0-9_-]*)?"
    r"(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|"
    r"password|passwd|pwd|token|credential)$"
)
_STATIC_CONFIG_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?P<export>export[ \t]+)?"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)"
    r"(?P<before>[ \t]*)(?P<separator>=|:)(?P<after>[ \t]*)"
    r"(?P<value>\S(?:.*\S)?)\s*$"
)
_INI_SECTION_PATTERN = re.compile(r"^\s*\[[^\]\r\n]+\]\s*(?:[#;].*)?$")
_JSON_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"')
_GENERIC_CREDENTIAL_MARKER = "<GENERIC_CREDENTIAL_REDACTED>"
_APPROVED_TYPED_MARKERS = frozenset(
    {rule.marker for rule in _SANITIZER_RULES} | {_GENERIC_CREDENTIAL_MARKER}
)


_MatchObserver = Callable[[str, str], None]


def _sanitize_text(
    text: str, on_match: _MatchObserver | None = None
) -> tuple[str, dict[str, int]]:
    cleaned = text
    counts: dict[str, int] = {}

    for rule in _SANITIZER_RULES:

        def replacement(match: re.Match[str], *, _rule: _SanitizerRule = rule) -> str:
            redact_group = _rule.redact_group
            if isinstance(redact_group, tuple):
                redact_group = next(
                    group for group in redact_group if match.group(group) is not None
                )
            sensitive = (
                match.group(redact_group)
                if redact_group is not None
                else match.group(0)
            )
            if _rule.should_redact is not None and not _rule.should_redact(sensitive):
                return match.group(0)

            counts[_rule.kind] = counts.get(_rule.kind, 0) + 1
            if on_match is not None:
                on_match(_rule.kind, sensitive)
            if redact_group is None:
                return _rule.marker

            start, end = match.span(redact_group)
            relative_start = start - match.start()
            relative_end = end - match.start()
            return (
                match.group(0)[:relative_start]
                + _rule.marker
                + match.group(0)[relative_end:]
            )

        cleaned = rule.pattern.sub(replacement, cleaned)

    return cleaned, counts


def sanitize_text(text: str) -> tuple[str, dict[str, int]]:
    """Replace credential/PII shapes with stable typed markers."""
    return _sanitize_text(text)


def _is_credential_key(value: str) -> bool:
    return _CREDENTIAL_KEY_PATTERN.fullmatch(value) is not None


def _is_approved_typed_marker(value: object) -> bool:
    return isinstance(value, str) and value in _APPROVED_TYPED_MARKERS


def _is_approved_credential_value(value: object) -> bool:
    return _is_approved_typed_marker(value) or _is_builtin_github_token_expression(
        value
    )


def sanitize_value(value: object) -> tuple[object, dict[str, int]]:
    """Recursively sanitize JSON strings without changing container shape."""
    counts: dict[str, int] = {}

    def merge(nested_counts: dict[str, int]) -> None:
        for kind, count in nested_counts.items():
            counts[kind] = counts.get(kind, 0) + count

    def sanitize_nested(nested: object) -> object:
        if isinstance(nested, str):
            cleaned, nested_counts = sanitize_text(nested)
            merge(nested_counts)
            return cleaned
        if isinstance(nested, list):
            return [sanitize_nested(item) for item in nested]
        if isinstance(nested, dict):
            cleaned: dict[object, object] = {}
            for key, item in nested.items():
                cleaned_item = sanitize_nested(item)
                if (
                    isinstance(key, str)
                    and _is_credential_key(key)
                    and isinstance(cleaned_item, str)
                    and not _is_approved_credential_value(cleaned_item)
                ):
                    counts["generic_credential"] = (
                        counts.get("generic_credential", 0) + 1
                    )
                    cleaned_item = _GENERIC_CREDENTIAL_MARKER
                cleaned[key] = cleaned_item
            return cleaned
        return nested

    return sanitize_nested(value), counts


_NON_GIT_PRUNED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_RELEASE_RESIDUE_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "logs",
        "node_modules",
        "reports",
        "results",
        "results_smoke",
        "venv",
    }
)
_RELEASE_RESIDUE_FILE_NAMES = frozenset(
    {".coverage", ".DS_Store", "coverage.xml", "junit.xml"}
)
_RELEASE_RESIDUE_SUFFIXES = (
    ".dist-info",
    ".egg-info",
    ".log",
    ".pyc",
    ".pyo",
)
_VERSIONED_VENV_DIRECTORY_PATTERN = re.compile(
    r"^\.?venv(?:[-_.]?(?:py)?\d[\w.-]*)$",
    re.IGNORECASE,
)
_TEST_RESULTS_DIRECTORY_PATTERN = re.compile(
    r"^test[-_]results(?:[-_.].*)?$",
    re.IGNORECASE,
)


def _git_worktree_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def _git_pathspec(root: Path, path: Path) -> str:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise FileNotFoundError(path) from error
    return "." if relative == Path(".") else relative.as_posix()


def _git_head_files(root: Path, pathspec: str) -> list[_ScanFile]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", "HEAD"],
        check=False,
        capture_output=True,
    )
    if head.returncode != 0:
        return []

    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            "HEAD",
            "--",
            pathspec,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = os.fsdecode(result.stderr).strip()
        raise RuntimeError(message or "git ls-tree failed")

    files: list[_ScanFile] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", 1)
            raw_mode, raw_type, raw_object = metadata.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("invalid git ls-tree output") from error
        mode = raw_mode.decode("ascii", errors="strict")
        object_type = raw_type.decode("ascii", errors="strict")
        git_path = os.fsdecode(raw_name)
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise RuntimeError(
                f"unsupported tracked HEAD entry: {git_path} ({mode} {object_type})"
            )
        files.append(
            _ScanFile(
                path=root / git_path,
                git_root=root,
                git_path=git_path,
                head_blob=raw_object.decode("ascii", errors="strict"),
            )
        )
    return files


def _git_visible_files(path: Path) -> list[_ScanFile] | None:
    root = _git_worktree_root(path)
    if root is None:
        return None

    pathspec = _git_pathspec(root, path)
    head_files = _git_head_files(root, pathspec)
    head_paths = {file.git_path for file in head_files}
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            pathspec,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = os.fsdecode(result.stderr).strip()
        raise RuntimeError(message or "git ls-files failed")

    files = list(head_files)
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        if name in head_paths:
            continue
        candidate = root / name
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        if candidate.exists() or candidate.is_symlink():
            files.append(_ScanFile(path=candidate))
    return files


def _walk_non_git_files(path: Path) -> list[_ScanFile]:
    files: list[_ScanFile] = []
    pending = [path]
    while pending:
        directory = pending.pop()
        for candidate in sorted(directory.iterdir(), key=lambda item: str(item)):
            if candidate.is_symlink() or not candidate.is_dir():
                files.append(_ScanFile(path=candidate))
            elif candidate.name not in _NON_GIT_PRUNED_DIRECTORY_NAMES:
                pending.append(candidate)
    return files


def _is_release_residue(path: Path) -> bool:
    name = path.name
    is_directory = not path.is_symlink() and path.is_dir()
    return (
        (is_directory and name in _RELEASE_RESIDUE_DIRECTORY_NAMES)
        or (is_directory and _VERSIONED_VENV_DIRECTORY_PATTERN.fullmatch(name))
        or (is_directory and _TEST_RESULTS_DIRECTORY_PATTERN.fullmatch(name))
        or name in _RELEASE_RESIDUE_FILE_NAMES
        or name.startswith(".coverage.")
        or name.endswith(_RELEASE_RESIDUE_SUFFIXES)
    )


def find_release_residue(paths: Sequence[Path]) -> list[Path]:
    """Return build/test residue paths without reading their contents."""
    residue: set[Path] = set()
    pending: list[Path] = []

    for path in paths:
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(path)
        if path.name == ".git":
            continue
        if _is_release_residue(path):
            residue.add(path)
        elif path.is_dir() and not path.is_symlink():
            pending.append(path)

    while pending:
        directory = pending.pop()
        for candidate in sorted(directory.iterdir(), key=lambda item: str(item)):
            if candidate.name == ".git":
                continue
            if _is_release_residue(candidate):
                residue.add(candidate)
            elif candidate.is_dir() and not candidate.is_symlink():
                pending.append(candidate)

    return sorted(residue, key=lambda path: str(path))


def _git_file_source(path: Path) -> _ScanFile:
    root = _git_worktree_root(_nearest_existing_parent(path))
    if root is None:
        return _ScanFile(path=path)
    pathspec = _git_pathspec(root, path)
    for source in _git_head_files(root, pathspec):
        if source.git_path == pathspec:
            return source
    return _ScanFile(path=path)


def _iter_files(paths: Sequence[Path]) -> list[_ScanFile]:
    files: list[_ScanFile] = []

    for path in paths:
        if not path.exists() and not path.is_symlink():
            source = _git_file_source(path)
            if source.head_blob is None:
                raise FileNotFoundError(path)
            files.append(source)
            continue

        if not path.is_symlink() and path.is_dir():
            git_files = _git_visible_files(path)
            files.extend(
                git_files if git_files is not None else _walk_non_git_files(path)
            )
        else:
            files.append(_git_file_source(path))

    unique_files: dict[str, _ScanFile] = {}
    for source in files:
        identity = str(source.path.absolute())
        current = unique_files.get(identity)
        if current is None or (
            current.head_blob is None and source.head_blob is not None
        ):
            unique_files[identity] = source
    return sorted(unique_files.values(), key=lambda source: str(source.path))


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.parent
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(path)
        candidate = parent
    return candidate


def _read_git_head_bytes(source: _ScanFile) -> bytes:
    if source.git_root is None or source.git_path is None or source.head_blob is None:
        raise FileNotFoundError(source.path)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source.git_root),
            "cat-file",
            "blob",
            source.head_blob,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to read tracked HEAD blob: {source.git_path}")
    return result.stdout


def _read_text(source: _ScanFile) -> str | None:
    path = source.path
    if source.head_blob is not None:
        data = _read_git_head_bytes(source)
    elif path.is_symlink():
        data = os.fsencode(path.readlink())
    else:
        data = path.read_bytes()

    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _record_text_findings(
    findings: list[Finding],
    path: Path,
    line_number: int,
    text: str,
    *,
    ignored_kinds: frozenset[str] = frozenset(),
    included_kinds: frozenset[str] | None = None,
) -> None:
    def record(kind: str, sensitive: str) -> None:
        if kind in ignored_kinds or (
            included_kinds is not None and kind not in included_kinds
        ):
            return
        findings.append(
            Finding(
                path=str(path),
                kind=kind,
                line=line_number,
                fingerprint=hashlib.sha256(sensitive.encode("utf-8")).hexdigest(),
            )
        )

    _sanitize_text(text, on_match=record)


def _python_credential_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        return _python_credential_name(node.slice)
    return None


_NON_STATIC = object()
_STATIC_PRIMITIVE_TYPES = (str, int, float, complex, bool, type(None))


def _static_format_atom(node: ast.expr) -> object:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_format_atom(node.left)
        right = _static_format_atom(node.right)
        if left is _NON_STATIC or right is _NON_STATIC:
            return _NON_STATIC
        try:
            value = left + right
        except TypeError:
            return _NON_STATIC
        if isinstance(value, _STATIC_PRIMITIVE_TYPES):
            return value
        return _NON_STATIC
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NON_STATIC
    if isinstance(value, _STATIC_PRIMITIVE_TYPES):
        return value
    return _NON_STATIC


def _static_formatted_value(node: ast.FormattedValue) -> str | None:
    value = _static_format_atom(node.value)
    if value is _NON_STATIC:
        return None

    if node.conversion == ord("s"):
        value = str(value)
    elif node.conversion == ord("r"):
        value = repr(value)
    elif node.conversion == ord("a"):
        value = ascii(value)
    elif node.conversion != -1:
        return None

    format_spec = ""
    if node.format_spec is not None:
        static_spec = _static_string_value(node.format_spec)
        if static_spec is None:
            return None
        format_spec = static_spec
    try:
        return format(value, format_spec)
    except (TypeError, ValueError):
        return None


def _static_string_value(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.FormattedValue):
        return _static_formatted_value(node)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            static_value = _static_string_value(value)
            if static_value is None:
                return None
            parts.append(static_value)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_value(node.left)
        right = _static_string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _is_string_composition(node: ast.AST | None) -> bool:
    return isinstance(node, (ast.JoinedStr, ast.FormattedValue)) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
    )


def _python_docstring_nodes(tree: ast.AST) -> set[ast.expr]:
    docstrings: set[ast.expr] = set()
    owners = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]
    for owner in owners:
        if not owner.body or not isinstance(owner.body[0], ast.Expr):
            continue
        expression = owner.body[0].value
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            docstrings.add(expression)
    return docstrings


def _has_nonstatic_composition_ancestor(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    parent = parents.get(node)
    while _is_string_composition(parent):
        if not isinstance(parent, ast.expr) or _static_string_value(parent) is None:
            return True
        parent = parents.get(parent)
    return False


def _is_template_constructor(node: ast.Call) -> bool:
    function = node.func
    return (isinstance(function, ast.Name) and function.id == "Template") or (
        isinstance(function, ast.Attribute) and function.attr == "Template"
    )


def _runtime_template_style(
    node: ast.expr,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    parent = parents.get(node)
    if (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and parent.attr in {"format", "format_map"}
    ):
        call = parents.get(parent)
        if isinstance(call, ast.Call) and call.func is parent:
            return "format"

    if (
        isinstance(parent, ast.BinOp)
        and isinstance(parent.op, ast.Mod)
        and parent.left is node
    ):
        return "percent"

    if (
        isinstance(parent, ast.Call)
        and _is_template_constructor(parent)
        and parent.args
        and parent.args[0] is node
    ):
        return "template"

    if (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and parent.attr in {"substitute", "safe_substitute"}
    ):
        call = parents.get(parent)
        if isinstance(call, ast.Call) and call.func is parent:
            return "template"
    return None


def _append_static_template_segment(
    segments: list[tuple[int, str]],
    line_offset: int,
    characters: list[str],
) -> None:
    if characters:
        segments.append((line_offset, "".join(characters)))
        characters.clear()


def _format_template_static_segments(value: str) -> list[tuple[int, str]]:
    segments: list[tuple[int, str]] = []
    characters: list[str] = []
    line_offset = 0
    segment_line_offset = 0
    try:
        parsed = string.Formatter().parse(value)
        for literal, field_name, format_spec, _conversion in parsed:
            if not characters:
                segment_line_offset = line_offset
            characters.append(literal)
            line_offset += literal.count("\n")
            if field_name is None:
                continue
            _append_static_template_segment(
                segments,
                segment_line_offset,
                characters,
            )
            line_offset += field_name.count("\n") + format_spec.count("\n")
            segment_line_offset = line_offset
    except ValueError:
        return [(0, value)]
    _append_static_template_segment(segments, segment_line_offset, characters)
    return segments


_PERCENT_FORMAT_FLAGS = frozenset("-+#0 ")
_PERCENT_FORMAT_LENGTH_MODIFIERS = frozenset("hlL")
_PERCENT_FORMAT_CONVERSIONS = frozenset("diouxXeEfFgGcrsa")


def _percent_template_static_segments_with_steps(
    value: str,
) -> tuple[list[tuple[int, str]], int]:
    segments: list[tuple[int, str]] = []
    characters: list[str] = []
    line_offset = 0
    segment_line_offset = 0
    index = 0
    steps = 0

    def append_literal(literal: str) -> None:
        nonlocal line_offset, segment_line_offset
        if not literal:
            return
        if not characters:
            segment_line_offset = line_offset
        characters.append(literal)
        line_offset += literal.count("\n")

    while index < len(value):
        character = value[index]
        steps += 1
        if character != "%":
            append_literal(character)
            index += 1
            continue

        cursor = index + 1
        if cursor < len(value):
            steps += 1
        if cursor < len(value) and value[cursor] == "%":
            append_literal("%")
            index += 2
            continue

        if cursor < len(value):
            steps += 1
        if cursor < len(value) and value[cursor] == "(":
            cursor += 1
            key_start = cursor
            terminator = ""
            while cursor < len(value):
                steps += 1
                terminator = value[cursor]
                if terminator == ")" or terminator in "\r\n":
                    break
                cursor += 1
            if terminator != ")" or cursor == key_start:
                malformed_end = cursor + (terminator == ")")
                append_literal(value[index:malformed_end])
                index = malformed_end
                continue
            cursor += 1

        while cursor < len(value):
            steps += 1
            if value[cursor] not in _PERCENT_FORMAT_FLAGS:
                break
            cursor += 1

        if cursor < len(value):
            steps += 1
            if value[cursor] == "*":
                cursor += 1
            else:
                while cursor < len(value):
                    steps += 1
                    if not value[cursor].isdigit():
                        break
                    cursor += 1

        if cursor < len(value):
            steps += 1
        if cursor < len(value) and value[cursor] == ".":
            cursor += 1
            precision_start = cursor
            if cursor < len(value):
                steps += 1
            if cursor < len(value) and value[cursor] == "*":
                cursor += 1
            else:
                while cursor < len(value):
                    steps += 1
                    if not value[cursor].isdigit():
                        break
                    cursor += 1
                if cursor == precision_start:
                    append_literal(value[index:cursor])
                    index = cursor
                    continue

        if cursor < len(value):
            steps += 1
        if cursor < len(value) and value[cursor] in _PERCENT_FORMAT_LENGTH_MODIFIERS:
            cursor += 1

        if cursor < len(value):
            steps += 1
        if cursor >= len(value) or value[cursor] not in _PERCENT_FORMAT_CONVERSIONS:
            append_literal(value[index:cursor])
            index = cursor
            continue

        field_end = cursor + 1
        _append_static_template_segment(
            segments,
            segment_line_offset,
            characters,
        )
        line_offset += value[index:field_end].count("\n")
        segment_line_offset = line_offset
        index = field_end
    _append_static_template_segment(segments, segment_line_offset, characters)
    return segments, steps


def _percent_template_static_segments(value: str) -> list[tuple[int, str]]:
    segments, _steps = _percent_template_static_segments_with_steps(value)
    return segments


def _string_template_static_segments(value: str) -> list[tuple[int, str]]:
    segments: list[tuple[int, str]] = []
    characters: list[str] = []
    line_offset = 0
    segment_line_offset = 0
    cursor = 0
    for match in string.Template.pattern.finditer(value):
        literal = value[cursor : match.start()]
        if not characters:
            segment_line_offset = line_offset
        characters.append(literal)
        line_offset += literal.count("\n")
        matched = match.group(0)
        if match.group("escaped") is not None:
            characters.append("$")
        elif match.group("named") is not None or match.group("braced") is not None:
            _append_static_template_segment(
                segments,
                segment_line_offset,
                characters,
            )
            segment_line_offset = line_offset
        else:
            characters.append(matched)
        line_offset += matched.count("\n")
        cursor = match.end()
    tail = value[cursor:]
    if not characters:
        segment_line_offset = line_offset
    characters.append(tail)
    _append_static_template_segment(segments, segment_line_offset, characters)
    return segments


def _static_runtime_template_segments(
    value: str,
    style: str,
) -> list[tuple[int, str]]:
    if style == "format":
        return _format_template_static_segments(value)
    if style == "percent":
        return _percent_template_static_segments(value)
    if style == "template":
        return _string_template_static_segments(value)
    raise ValueError(f"unknown runtime template style: {style}")


def _record_python_credential_literal(
    findings: list[Finding],
    path: Path,
    name_node: ast.expr,
    value_node: ast.expr | None,
    *,
    line_offset: int = 0,
) -> None:
    name = _python_credential_name(name_node)
    value = _static_string_value(value_node)
    if name is None or not _is_credential_key(name) or value is None or not value:
        return

    cleaned, counts = sanitize_text(value)
    if counts or _is_approved_credential_value(cleaned):
        return
    findings.append(
        Finding(
            path=str(path),
            kind="generic_credential",
            line=value_node.lineno + line_offset,
            fingerprint=hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
    )


def _record_python_credential_nodes(
    findings: list[Finding],
    path: Path,
    tree: ast.AST,
    *,
    line_offset: int = 0,
) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _record_python_credential_literal(
                    findings,
                    path,
                    target,
                    node.value,
                    line_offset=line_offset,
                )
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            _record_python_credential_literal(
                findings,
                path,
                node.target,
                node.value,
                line_offset=line_offset,
            )
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    _record_python_credential_literal(
                        findings,
                        path,
                        key,
                        value,
                        line_offset=line_offset,
                    )
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is not None:
                    _record_python_credential_literal(
                        findings,
                        path,
                        ast.Name(id=keyword.arg),
                        keyword.value,
                        line_offset=line_offset,
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            positional = [*node.args.posonlyargs, *node.args.args]
            default_args = positional[-len(node.args.defaults) :]
            for argument, default in zip(default_args, node.args.defaults):
                _record_python_credential_literal(
                    findings,
                    path,
                    ast.Name(id=argument.arg),
                    default,
                    line_offset=line_offset,
                )
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    _record_python_credential_literal(
                        findings,
                        path,
                        ast.Name(id=argument.arg),
                        default,
                        line_offset=line_offset,
                    )


def _record_semantic_generic_findings(
    findings: list[Finding],
    path: Path,
    line_number: int,
    text: str,
    *,
    in_ini_section: bool = False,
) -> bool:
    meaningful_lines: list[tuple[int, str]] = []
    config_line_count = 0
    for offset, line in enumerate(text.splitlines(), start=0):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        meaningful_lines.append((offset, line))
        if _INI_SECTION_PATTERN.fullmatch(line) is not None:
            in_ini_section = True
            config_line_count += 1
            continue
        match = _STATIC_CONFIG_ASSIGNMENT_PATTERN.fullmatch(line)
        if match is None or not _is_credential_key(match.group("key")):
            continue
        is_config = (
            match.group("export") is not None
            or match.group("separator") == ":"
            or (not match.group("before") and not match.group("after"))
            or in_ini_section
        )
        if not is_config:
            continue
        config_line_count += 1
        _record_text_findings(
            findings,
            path,
            line_number + offset,
            line,
            included_kinds=frozenset({"generic_credential"}),
        )

    if meaningful_lines and config_line_count == len(meaningful_lines):
        return in_ini_section

    try:
        semantic_tree = _parse_python(textwrap.dedent(text))
    except SyntaxError:
        for offset, line in meaningful_lines:
            _record_text_findings(
                findings,
                path,
                line_number + offset,
                line,
                included_kinds=frozenset({"generic_credential"}),
            )
        return in_ini_section

    _record_python_credential_nodes(
        findings,
        path,
        semantic_tree,
        line_offset=line_number - 1,
    )
    return in_ini_section


def _parse_python(text: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


def _scan_python_text(findings: list[Finding], path: Path, text: str) -> None:
    try:
        tree = _parse_python(text)
    except SyntaxError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            _record_text_findings(findings, path, line_number, line)
        return

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    docstrings = _python_docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.expr):
            static_value = _static_string_value(node)
            parent = parents.get(node)
            if static_value is not None and not (
                _is_string_composition(parent)
                and isinstance(parent, ast.expr)
                and _static_string_value(parent) is not None
            ):
                suppress_generic = (
                    node in docstrings
                    or _has_nonstatic_composition_ancestor(node, parents)
                )
                for offset, line in enumerate(static_value.splitlines(), start=0):
                    _record_text_findings(
                        findings,
                        path,
                        node.lineno + offset,
                        line,
                        ignored_kinds=frozenset({"generic_credential"}),
                    )
                if not suppress_generic:
                    template_style = _runtime_template_style(node, parents)
                    static_segments = (
                        [(0, static_value)]
                        if template_style is None
                        else _static_runtime_template_segments(
                            static_value,
                            template_style,
                        )
                    )
                    in_ini_section = False
                    for segment_offset, segment in static_segments:
                        in_ini_section = _record_semantic_generic_findings(
                            findings,
                            path,
                            node.lineno + segment_offset,
                            segment,
                            in_ini_section=in_ini_section,
                        )

    _record_python_credential_nodes(findings, path, tree)

    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            _record_text_findings(findings, path, token.start[0], token.string)


def _json_records(path: Path, text: str) -> list[tuple[int, object]] | None:
    try:
        if path.suffix.lower() == ".json":
            return [(1, json.loads(text))]
        if path.suffix.lower() == ".jsonl":
            return [
                (line_number, json.loads(line))
                for line_number, line in enumerate(text.splitlines(), start=1)
                if line.strip()
            ]
    except json.JSONDecodeError:
        return None
    return None


def _record_credential_field_findings(
    findings: list[Finding],
    path: Path,
    line_number: int,
    value: object,
) -> None:
    if isinstance(value, list):
        for item in value:
            _record_credential_field_findings(findings, path, line_number, item)
        return
    if not isinstance(value, dict):
        return

    for key, item in value.items():
        if isinstance(key, str) and _is_credential_key(key):
            cleaned = sanitize_text(item)[0] if isinstance(item, str) else item
            if not _is_approved_credential_value(cleaned):
                fingerprint_source = (
                    item
                    if isinstance(item, str)
                    else json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                findings.append(
                    Finding(
                        path=str(path),
                        kind="generic_credential",
                        line=line_number,
                        fingerprint=hashlib.sha256(
                            fingerprint_source.encode("utf-8")
                        ).hexdigest(),
                    )
                )
        _record_credential_field_findings(findings, path, line_number, item)


def _scan_json_text(
    findings: list[Finding],
    path: Path,
    text: str,
    records: list[tuple[int, object]],
) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in _JSON_STRING_PATTERN.finditer(line):
            decoded = json.loads(match.group(0))
            _record_text_findings(findings, path, line_number, decoded)

    for line_number, value in records:
        _record_credential_field_findings(findings, path, line_number, value)


def scan_paths(paths: Sequence[Path]) -> list[Finding]:
    """Return deterministic typed findings for sensitive text in paths."""
    findings: list[Finding] = []

    for source in _iter_files(paths):
        path = source.path
        text = _read_text(source)
        if text is None:
            continue

        if path.suffix.lower() == ".py":
            _scan_python_text(findings, path, text)
            continue

        records = _json_records(path, text)
        if records is not None:
            _scan_json_text(findings, path, text, records)
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            _record_text_findings(findings, path, line_number, line)

    return sorted(
        set(findings),
        key=lambda finding: (
            finding.path,
            finding.line,
            finding.kind,
            finding.fingerprint,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan release paths for credential and personal-data shapes."
    )
    parser.add_argument(
        "--check-release-residue",
        action="store_true",
        help="fail on build/test residue outside .git without reading its contents",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.check_release_residue:
        residue = find_release_residue(args.paths)
        for path in residue:
            print(json.dumps({"kind": "release_residue", "path": str(path)}))
        return 1 if residue else 0

    findings = scan_paths(args.paths)
    for finding in findings:
        print(
            json.dumps(
                {
                    "path": finding.path,
                    "kind": finding.kind,
                    "line": finding.line,
                    "fingerprint": finding.fingerprint[:12],
                },
                sort_keys=True,
            )
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
