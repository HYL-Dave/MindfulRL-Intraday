"""Seal the exact Task 8 packet and reject undeclared disk entries."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


PACKET = Path(__file__).resolve().parent
STATIC = {
    "README.md",
    "capture_offline_authority.py",
    "commands.txt",
    "mutation_pytest_probe.py",
    "normalize_packet_logs.py",
    "run_browser_matrix.py",
    "run_mutations.py",
    "run_shadow.py",
    "seal_packet.py",
    "test_packet_contracts.py",
    "verify_old_code.py",
    "write_verification_summary.py",
}
GENERATED = {
    "backend-focused-a.txt",
    "backend-focused-b.txt",
    "backend-full-a.txt",
    "backend-full-b.txt",
    "browser-run.txt",
    "focused-nodes-a.txt",
    "focused-nodes-b.txt",
    "frontend-build.txt",
    "frontend-i18n-literals.txt",
    "frontend-test.txt",
    "frontend-typecheck.txt",
    "mutation-ledger.json",
    "mutation-run.txt",
    "log-normalization.json",
    "offline-authority-run-a.txt",
    "offline-authority-run-b.txt",
    "offline-authority.json",
    "packet-contracts.txt",
    "verification-summary.json",
    "vite.txt",
    "browser/matrix.json",
}
SCREENSHOTS = {
    f"browser/{width}x{height}-{locale}-{scenario}.png"
    for width, height in ((1440, 900), (390, 844))
    for locale in ("en", "zh-Hant")
    for scenario in (
        "active",
        "not-found-monitoring",
        "inactive-history",
        "conflict-attention",
        "otc-continuation",
        "settings-massive-key",
    )
}
EXPECTED = STATIC | GENERATED | SCREENSHOTS


def main() -> int:
    manifest = PACKET / "SHA256SUMS"
    manifest.unlink(missing_ok=True)
    disk = {
        str(path.relative_to(PACKET))
        for path in PACKET.rglob("*")
        if path.is_file()
    }
    if disk != EXPECTED:
        raise SystemExit(
            f"packet_set_mismatch missing={sorted(EXPECTED - disk)} extra={sorted(disk - EXPECTED)}"
        )
    lines = [
        f"{hashlib.sha256((PACKET / name).read_bytes()).hexdigest()}  {name}"
        for name in sorted(EXPECTED)
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    checked = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=PACKET,
        text=True,
        capture_output=True,
        check=True,
    )
    manifest_names = {
        line.split("  ", 1)[1] for line in manifest.read_text(encoding="ascii").splitlines()
    }
    final_disk = {
        str(path.relative_to(PACKET))
        for path in PACKET.rglob("*")
        if path.is_file() and path != manifest
    }
    assert manifest_names == final_disk == EXPECTED
    print(
        f"files={len(EXPECTED)} verified={checked.stdout.count(': OK')} "
        f"packet_digest={hashlib.sha256(manifest.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
