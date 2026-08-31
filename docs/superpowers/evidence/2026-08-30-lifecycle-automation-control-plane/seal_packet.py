"""Seal every packet file except the checksum manifest itself."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKET = Path(__file__).resolve().parent
MANIFEST = PACKET / "SHA256SUMS"


def main() -> int:
    files = sorted(
        path for path in PACKET.rglob("*")
        if path.is_file() and path != MANIFEST
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(PACKET)}"
        for path in files
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"sealed={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
