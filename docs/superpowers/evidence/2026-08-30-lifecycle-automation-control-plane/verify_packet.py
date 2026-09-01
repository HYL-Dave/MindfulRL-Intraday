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
PRODUCT_HEAD = "65c5fa65bb34857e945437a15bf3660d56741232"
REQUIRED_REPAIR_MUTATIONS = frozenset(
    {f"M{index}" for index in range(33, 54)}
)
REQUIRED_SEMANTIC_MUTATIONS = {
    "transition_approval_authority": {
        "id": "M51",
        "owner": "test_due_runner_never_executes_transition_with_unknown_approval_authority",
        "path": "src/service/ticker_identity_scheduler.py",
    },
    "terminal_finalization_not_pending": {
        "id": "M52",
        "owner": "test_finalized_run_never_acquires_terminal_failure_retry_state",
        "path": "src/security_lifecycle_fact_kernel.py",
    },
    "automation_schedule_refresh": {
        "id": "M53",
        "owner": "reloads the complete schedule after each config save",
        "path": "apps/arkscope-web/src/settings/DataStorageSection.tsx",
    },
}
REQUIRED_GUARD_TRIAGE = frozenset({
    "automation_active_incident",
    "automation_legacy_witness",
    "automation_progress_context",
    "automation_progress_initial_stage",
    "automation_progress_missing",
    "automation_query_context",
    "automation_run_evidence_missing",
    "automation_run_not_found",
    "automation_run_not_running",
    "automation_run_not_succeeded",
    "automation_run_reconciliation_lost",
    "automation_status_timestamp",
    "automation_transition_approval_changed",
    "duplicate_case_outcome",
    "ibkr_blocker_context",
    "ibkr_position_symbol",
    "interval_minutes_invalid",
    "retained_source_family_refreshed",
    "terminal_finalization_assessment_missing",
    "terminal_finalization_failure_code",
    "terminal_finalization_not_pending",
    "ticker_identity_scheduler_incident",
    "ticker_identity_transition_status",
    "transition_approval_authority",
})
TRIAGE_CLASSIFICATIONS = frozenset({
    "defensive_redundancy",
    "owned_by_effect",
    "owned_by_reverse_mutation",
})


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


def _verify_readme(summary: dict[str, object]) -> None:
    readme = (PACKET / "README.md").read_text(encoding="utf-8")
    gates = summary["gates"]
    mutations = summary["mutations"]
    browser = summary["browser"]["measured"]
    required = (
        f"- Focused backend owners: `{gates['backend_focused']['passed']} passed`.",
        "- Backend full A/B: each `"
        f"{gates['backend_full_a']['passed']} passed / "
        f"{gates['backend_full_a']['skipped']} skipped / "
        f"{gates['backend_full_a']['warnings']} warnings`;",
        "- Frontend A/B: each `"
        f"{gates['frontend_a']['files']} files / "
        f"{gates['frontend_a']['tests']} passed`.",
        f"- Reverse mutations: `{mutations['killed']}/{mutations['count']}` ",
        f"- Browser: {browser['entries']} EN/zh-Hant desktop/mobile Settings,",
    )
    for fragment in required:
        if fragment not in readme:
            raise AssertionError(f"readme_summary:{fragment}")


def main() -> int:
    writer = _load("write_verification_summary")
    expected_summary = writer.build_summary()
    summary = _json("verification-summary.json")
    if summary != expected_summary:
        raise AssertionError("verification_summary_drift")
    _verify_readme(summary)
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
    mutation_definitions = {
        mutation.mutation_id: mutation for mutation in mutations.MUTATIONS
    }
    ledger_by_id = {row["id"]: row for row in ledger["mutations"]}
    for semantic_name, requirement in REQUIRED_SEMANTIC_MUTATIONS.items():
        mutation = mutation_definitions.get(requirement["id"])
        row = ledger_by_id.get(requirement["id"])
        if (
            mutation is None
            or row is None
            or mutation.path != requirement["path"]
            or tuple(mutation.owner_needles) != (requirement["owner"],)
            or mutation.old == mutation.new
            or not row["killed"]
            or row["product_file"] != requirement["path"]
            or row["owners_observed"] != {requirement["owner"]: True}
        ):
            raise AssertionError(f"semantic_mutation:{semantic_name}")
        if semantic_name != "automation_schedule_refresh" and (
            semantic_name not in mutation.old or semantic_name not in mutation.new
        ):
            raise AssertionError(f"semantic_guard_mutation:{semantic_name}")
    if not ledger["all_mutations_killed"] or not ledger[
        "all_files_restored_byte_identically"
    ]:
        raise AssertionError("mutation_result")
    triage = _json("guard-triage.json")
    entries = triage.get("entries")
    if (
        triage.get("schema_version") != 1
        or triage.get("base_commit") != BASE_COMMIT
        or triage.get("product_head_commit") != PRODUCT_HEAD
        or triage.get("candidate_count") != len(REQUIRED_GUARD_TRIAGE)
        or not isinstance(entries, list)
    ):
        raise AssertionError("guard_triage_authority")
    triage_by_guard = {
        entry.get("guard"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("guard"), str)
    }
    if (
        len(triage_by_guard) != len(entries)
        or set(triage_by_guard) != REQUIRED_GUARD_TRIAGE
    ):
        raise AssertionError("guard_triage_coverage")
    for guard, entry in triage_by_guard.items():
        if (
            entry.get("classification") not in TRIAGE_CLASSIFICATIONS
            or not isinstance(entry.get("evidence"), str)
            or not entry["evidence"].strip()
            or not isinstance(entry.get("reason"), str)
            or not entry["reason"].strip()
        ):
            raise AssertionError(f"guard_triage_entry:{guard}")
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
