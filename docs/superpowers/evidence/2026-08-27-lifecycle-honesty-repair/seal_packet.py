"""Write deterministic SHA256SUMS for every packet payload except itself."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKET = Path(__file__).resolve().parent
MANIFEST = PACKET / "SHA256SUMS"


def main() -> None:
    files = sorted(
        path
        for path in PACKET.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(PACKET)}"
        for path in files
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
