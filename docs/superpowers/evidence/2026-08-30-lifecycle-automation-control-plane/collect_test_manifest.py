"""Collect a canonical pytest node manifest without executing tests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
TOKEN_SHAPE = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}")


def normalize_node(node: str) -> str:
    return TOKEN_SHAPE.sub(
        lambda match: "[TOKEN_SHAPE_SHA256_"
        + hashlib.sha256(match.group(0).encode()).hexdigest()[:16]
        + "]",
        node,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("paths", nargs="+", default=["tests"])
    args = parser.parse_args()
    process = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *args.paths,
        ),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
    nodes = sorted(
        normalize_node(line.strip())
        for line in process.stdout.splitlines()
        if "::" in line and not line.startswith(("=", " "))
    )
    payload = "\n".join(nodes) + "\n"
    Path(args.output).write_text(payload, encoding="utf-8")
    print(
        f"nodes={len(nodes)} sha256={hashlib.sha256(payload.encode()).hexdigest()}"
    )
    return 0 if process.returncode == 0 and nodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
