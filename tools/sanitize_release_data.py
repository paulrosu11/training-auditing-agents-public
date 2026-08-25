"""Deterministically sanitize JSON and JSONL files for public release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from .release_safety import sanitize_value
else:
    from release_safety import sanitize_value


_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".superpowers",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_WILDCHAT_FILE_NAME = "wildchat_blog_20.jsonl"


def _data_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [root] if root.suffix.lower() in {".json", ".jsonl"} else []

    paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _IGNORED_DIRECTORY_NAMES
            and not (current / name).is_symlink()
            and not (current / name / ".git").exists()
        )
        for file_name in sorted(file_names):
            path = current / file_name
            if path.is_symlink() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            paths.append(path)
    return paths


def _merge_counts(total: dict[str, int], additional: dict[str, int]) -> None:
    for kind, count in additional.items():
        total[kind] = total.get(kind, 0) + count


def _sanitize_json(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cleaned, counts = sanitize_value(value)
    path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts


def _sanitize_jsonl(path: Path) -> dict[str, int]:
    total: dict[str, int] = {}
    output_lines: list[str] = []
    text = path.read_text(encoding="utf-8")
    record_index = 0

    for line in text.splitlines():
        if not line.strip():
            output_lines.append("")
            continue
        value = json.loads(line)
        cleaned, counts = sanitize_value(value)
        record_index += 1
        if path.name == _WILDCHAT_FILE_NAME:
            if not isinstance(cleaned, dict):
                raise TypeError("WildChat rows must be JSON objects")
            cleaned.pop("session_id", None)
            cleaned["record_id"] = f"wildchat-sanitized-{record_index:03d}"
        _merge_counts(total, counts)
        output_lines.append(
            json.dumps(
                cleaned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    output = "\n".join(output_lines)
    if output_lines:
        output += "\n"
    path.write_text(output, encoding="utf-8")
    return total


def sanitize_root(root: Path) -> dict[str, int]:
    """Sanitize JSON/JSONL files below root and return aggregate counts."""
    total: dict[str, int] = {}
    for path in _data_paths(root):
        counts = (
            _sanitize_jsonl(path)
            if path.suffix.lower() == ".jsonl"
            else _sanitize_json(path)
        )
        _merge_counts(total, counts)
    return dict(sorted(total.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize JSON and JSONL data without printing matched values."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    try:
        counts = sanitize_root(args.root)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print(json.dumps({"error": "sanitization_failed"}), file=sys.stderr)
        return 2

    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
