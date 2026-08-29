"""Normalize sealed Task 8 text logs without changing semantic results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3].resolve()
PYTHON_ENV = Path(sys.prefix).resolve()
_TOKEN_SHAPE = re.compile(
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
)
LOGS = (
    "backend-focused-a.txt",
    "backend-focused-b.txt",
    "backend-full-a.txt",
    "backend-full-b.txt",
    "browser-run.txt",
    "frontend-build.txt",
    "frontend-i18n-literals.txt",
    "frontend-test.txt",
    "frontend-typecheck.txt",
    "focused-nodes-a.txt",
    "focused-nodes-b.txt",
    "full-nodes-a.txt",
    "full-nodes-b.txt",
    "mutation-run.txt",
    "offline-authority-run-a.txt",
    "offline-authority-run-b.txt",
    "packet-contracts.txt",
    "vite.txt",
)


def _replace_token_shapes(value: str) -> tuple[str, int]:
    replacements: dict[str, str] = {}
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        token = match.group(0)
        if token not in replacements:
            replacements[token] = f"<TOKEN_SHAPE_{len(replacements) + 1}>"
        return replacements[token]

    return _TOKEN_SHAPE.sub(replace, value), count


def _normalize_file(path: Path) -> dict:
    original = path.read_text(encoding="utf-8")
    repo_replacements = original.count(str(ROOT))
    normalized = original.replace(str(ROOT), "<REPO_ROOT>")
    python_env_replacements = 0
    if PYTHON_ENV != Path(sys.base_prefix).resolve():
        python_env_replacements = normalized.count(str(PYTHON_ENV))
        normalized = normalized.replace(str(PYTHON_ENV), "<PYTHON_ENV>")
    normalized, token_shape_replacements = _replace_token_shapes(normalized)
    newline_count = len(normalized) - len(normalized.rstrip("\n"))
    trailing_blank_lines_removed = max(0, newline_count - 1)
    normalized = normalized.rstrip("\n") + "\n"
    path.write_text(normalized, encoding="utf-8")
    return {
        "repo_root_replacements": repo_replacements,
        "python_env_replacements": python_env_replacements,
        "token_shape_replacements": token_shape_replacements,
        "trailing_blank_lines_removed": trailing_blank_lines_removed,
        "content_changed": normalized != original,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = {name: _normalize_file(PACKET / name) for name in LOGS}
    payload = {
        "schema_version": 1,
        "method": "path_and_ordered_token_shape_placeholders_with_single_terminal_newline",
        "semantic_counts_and_results_preserved": True,
        "placeholders": {
            "checkout": "<REPO_ROOT>",
            "python_environment": "<PYTHON_ENV>",
            "token_shapes": "<TOKEN_SHAPE_N> in first-appearance order per file",
        },
        "files": files,
        "totals": {
            key: sum(int(row[key]) for row in files.values())
            for key in (
                "repo_root_replacements",
                "python_env_replacements",
                "token_shape_replacements",
                "trailing_blank_lines_removed",
            )
        },
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(payload["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
