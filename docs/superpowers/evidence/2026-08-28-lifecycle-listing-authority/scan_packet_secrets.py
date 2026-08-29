"""Fail closed when a packet contains live environment secrets or token shapes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


_SENSITIVE_NAME = re.compile(r"(?:API_KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)")
_TOKEN_PATTERNS = {
    "openai_token_shape": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "github_fine_grained_token_shape": re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    "github_classic_token_shape": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
}


def _scan(packet: Path, output: Path) -> dict:
    environment_values = {
        name: value.encode()
        for name, value in os.environ.items()
        if value and len(value) >= 8 and _SENSITIVE_NAME.search(name)
    }
    findings: list[dict[str, str]] = []
    scanned = 0
    for path in sorted(item for item in packet.rglob("*") if item.is_file()):
        if path.resolve() == output.resolve() or path.name == "SHA256SUMS":
            continue
        scanned += 1
        body = path.read_bytes()
        relative = str(path.relative_to(packet))
        for name, value in environment_values.items():
            if value in body:
                findings.append(
                    {"kind": "environment_secret_value", "name": name, "path": relative}
                )
        for kind, pattern in _TOKEN_PATTERNS.items():
            if pattern.search(body):
                findings.append({"kind": kind, "path": relative})
        if path.suffix in {".json", ".txt", ".log"} and b"environ({" in body:
            findings.append({"kind": "environment_repr", "path": relative})
    return {
        "schema_version": 1,
        "scanned_file_count": scanned,
        "environment_secret_names_checked": sorted(environment_values),
        "finding_count": len(findings),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = _scan(args.packet.resolve(), args.output.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, ValueError) as exc:
        print(f"packet_secret_scan_failed:{type(exc).__name__}", file=sys.stderr)
        return 1
    if payload["finding_count"]:
        print(f"packet_secret_findings:{payload['finding_count']}", file=sys.stderr)
        return 1
    print(json.dumps({"files": payload["scanned_file_count"], "findings": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
