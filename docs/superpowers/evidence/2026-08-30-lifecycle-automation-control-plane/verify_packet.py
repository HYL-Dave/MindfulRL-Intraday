"""Verify cross-artifact truth and, when present, the final packet seal."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
MANIFEST = PACKET / "SHA256SUMS"
BASE_COMMIT = "947a51fca2f078e750bef64cad4817682141ea8f"
PRODUCT_HEAD = "6c20cd557715eab5f0abaafe2b923313ee38ed33"
REQUIRED_REPAIR_MUTATIONS = frozenset(
    {f"M{index}" for index in range(33, 51)}
)


def _load(name: str):
    path = PACKET / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"control_plane_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module:{name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(name: str) -> dict[str, object]:
    return json.loads((PACKET / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest() -> int:
    if not MANIFEST.exists():
        return 0
    entries = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries:
            raise AssertionError(f"duplicate_manifest_path:{relative}")
        entries[relative] = digest
    disk = {
        str(path.relative_to(PACKET))
        for path in PACKET.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if set(entries) != disk:
        raise AssertionError("manifest_disk_set")
    for relative, expected in entries.items():
        if _sha256(PACKET / relative) != expected:
            raise AssertionError(f"manifest_hash:{relative}")
    return len(entries)


def main() -> int:
    writer = _load("write_verification_summary")
    expected_summary = writer.build_summary()
    summary = _json("verification-summary.json")
    if summary != expected_summary:
        raise AssertionError("verification_summary_drift")
    repository = _json("repository-binding.json")
    current_head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tested_head = str(repository["head_commit"])
    if (
        repository["product_head_commit"] != PRODUCT_HEAD
        or repository["base_commit"] != BASE_COMMIT
        or not repository["product_head_is_ancestor"]
        or not repository["post_product_scope_only_packet"]
    ):
        raise AssertionError("repository_authority")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", tested_head, current_head),
        cwd=ROOT,
        check=False,
    ).returncode == 0
    if not ancestor:
        raise AssertionError("tested_head_not_ancestor")
    post_binding_paths = subprocess.run(
        ("git", "diff", "--name-only", f"{tested_head}..{current_head}"),
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if any(
        not path.startswith(
            "docs/superpowers/evidence/2026-08-30-lifecycle-automation-control-plane/"
        )
        for path in post_binding_paths
    ):
        raise AssertionError("post_binding_product_drift")
    if not repository["all_schema_authorities_byte_identical"]:
        raise AssertionError("schema_drift")
    if not repository["all_browser_fixture_authorities_match_tested_head"]:
        raise AssertionError("browser_fixture_drift")
    ledger = _json("mutation-ledger.json")
    mutations = _load("run_mutations")
    if ledger["mutation_count"] != len(mutations.MUTATIONS):
        raise AssertionError("mutation_definition_drift")
    mutation_ids = {row["id"] for row in ledger["mutations"]}
    if not REQUIRED_REPAIR_MUTATIONS.issubset(mutation_ids):
        raise AssertionError("repair_mutation_coverage")
    if not ledger["all_mutations_killed"] or not ledger[
        "all_files_restored_byte_identically"
    ]:
        raise AssertionError("mutation_result")
    scanner = _json("secret-scan.json")
    if not scanner["passed"] or scanner["findings"]:
        raise AssertionError("secret_scan")
    manifest_entries = _verify_manifest()
    print(json.dumps({
        "summary": "valid",
        "mutations": ledger["killed_count"],
        "secret_findings": len(scanner["findings"]),
        "sealed_files": manifest_entries,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
