"""Normalize generated text artifacts without changing semantic content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKET = Path(__file__).resolve().parent
OUTPUT = PACKET / "text-normalization.json"
GENERATED_TEXT = (
    "backend-focused.txt",
    "backend-full-a.txt",
    "backend-full-b.txt",
    "browser-run.txt",
    "frontend-build.txt",
    "frontend-i18n-literals.txt",
    "frontend-test-a.txt",
    "frontend-test-b.txt",
    "frontend-typecheck.txt",
    "full-nodes-a.txt",
    "full-nodes-b.txt",
    "vite.txt",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(path: Path) -> dict[str, object]:
    before = path.read_bytes()
    text = before.decode("utf-8")
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_newlines.split("\n")
    terminal_empty_lines = 0
    while lines and lines[-1] == "":
        lines.pop()
        terminal_empty_lines += 1
    trailing_whitespace_lines = sum(line != line.rstrip(" \t") for line in lines)
    normalized_lines = [line.rstrip(" \t") for line in lines]
    after = ("\n".join(normalized_lines) + "\n").encode("utf-8")
    path.write_bytes(after)
    return {
        "path": str(path.relative_to(PACKET)),
        "before_sha256": sha256(before),
        "after_sha256": sha256(after),
        "byte_count_before": len(before),
        "byte_count_after": len(after),
        "line_count": len(normalized_lines),
        "trailing_whitespace_lines_normalized": trailing_whitespace_lines,
        "terminal_empty_lines_removed": max(0, terminal_empty_lines - 1),
        "semantic_line_order_preserved": True,
    }


def main() -> int:
    rows = [normalize(PACKET / name) for name in GENERATED_TEXT]
    payload = {
        "schema_version": 1,
        "normalization": "strip_trailing_horizontal_whitespace_and_surplus_terminal_blank_lines",
        "semantic_content_rewritten": False,
        "artifacts": rows,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "artifacts": len(rows),
        "trailing_whitespace_lines": sum(
            int(row["trailing_whitespace_lines_normalized"]) for row in rows
        ),
        "terminal_empty_lines_removed": sum(
            int(row["terminal_empty_lines_removed"]) for row in rows
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
