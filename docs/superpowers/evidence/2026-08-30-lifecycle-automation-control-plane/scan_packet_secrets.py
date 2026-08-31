"""Scan packet bytes without ever serializing the compared secret values."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


PACKET = Path(__file__).resolve().parent
OUTPUT_NAMES = {"SHA256SUMS", "secret-scan.json"}
TOKEN_PATTERNS = {
    "openai_like": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
    "github_like": re.compile(
        rb"(?:github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{16,})"
    ),
    "pem_private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def secret_environment() -> dict[str, bytes]:
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {
        name: value.encode()
        for name, value in os.environ.items()
        if value and len(value) >= 8 and any(marker in name.upper() for marker in markers)
    }


def main() -> int:
    environment = secret_environment()
    files = sorted(
        path for path in PACKET.rglob("*")
        if path.is_file() and path.name not in OUTPUT_NAMES
    )
    findings: list[dict[str, str]] = []
    for path in files:
        data = path.read_bytes()
        relative = str(path.relative_to(PACKET))
        for name, value in environment.items():
            if value in data:
                findings.append({
                    "path": relative,
                    "kind": "configured_environment_value",
                    "identifier_sha256": hashlib.sha256(name.encode()).hexdigest(),
                })
        for name, pattern in TOKEN_PATTERNS.items():
            if pattern.search(data):
                findings.append({"path": relative, "kind": name})
    payload = {
        "schema_version": 1,
        "files_scanned": len(files),
        "environment_secret_names_compared": len(environment),
        "secret_values_serialized": 0,
        "findings": findings,
        "passed": not findings,
    }
    (PACKET / "secret-scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"files": len(files), "findings": len(findings)}))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
