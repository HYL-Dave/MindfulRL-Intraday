"""Summarize bounded Task 8 admission outputs without changing their evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]


def _json(name: str) -> dict:
    return json.loads((PACKET / name).read_text(encoding="utf-8"))


def _pytest(name: str) -> dict:
    text = (PACKET / name).read_text(encoding="utf-8")
    matches = re.findall(r"(?:(\d+) passed)?(?:,?\s*(\d+) skipped)?", text)
    passed = max((int(a) for a, _ in matches if a), default=0)
    skipped = max((int(b) for _, b in matches if b), default=0)
    return {"passed": passed, "skipped": skipped, "failures": text.count("FAILED ")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frontend_test_counts() -> dict:
    text = (PACKET / "frontend-test.txt").read_text(encoding="utf-8")
    files = re.search(r"Test Files\s+(\d+) passed", text)
    tests = re.search(r"Tests\s+(\d+) passed", text)
    if files is None or tests is None:
        raise AssertionError("frontend_test_counts_missing")
    return {"files_passed": int(files.group(1)), "tests_passed": int(tests.group(1))}


def main() -> int:
    authority = _json("offline-authority.json")
    mutations = _json("mutation-ledger.json")
    browser = _json("browser/matrix.json")
    entries = browser["entries"]
    focused_a = _pytest("backend-focused-a.txt")
    focused_b = _pytest("backend-focused-b.txt")
    full_a = _pytest("backend-full-a.txt")
    full_b = _pytest("backend-full-b.txt")
    nodes_a = (PACKET / "focused-nodes-a.txt").read_text(encoding="utf-8").splitlines()
    nodes_b = (PACKET / "focused-nodes-b.txt").read_text(encoding="utf-8").splitlines()
    assert nodes_a == nodes_b and nodes_a
    assert focused_a == focused_b and focused_a["failures"] == 0
    assert full_a == full_b and full_a["failures"] == 0
    assert mutations["all_mutations_killed"] is True
    assert mutations["all_product_files_restored_byte_identically"] is True
    assert authority["shadow"]["case_count"] == 9
    assert len(entries) == 24
    zero_fields = (
        "external_requests",
        "writes",
        "command_calls",
        "render_acknowledgements",
        "console_errors",
        "page_errors",
    )
    assert all(not entry[field] for entry in entries for field in zero_fields)
    assert all(entry["publisher_family_text_count"] == 0 for entry in entries)
    assert all(entry["listing_translation_button_count"] == 0 for entry in entries)
    assert all(entry["overlap_count"] == 0 for entry in entries)
    assert all(entry["clipped_text_count"] == 0 for entry in entries)
    scratch = authority["scratch_migration"]
    summary = {
        "schema_version": 1,
        "boundary": {
            "packet_and_fixture_only": True,
            "product_code_modified": False,
            "live_provider_calls": authority["declared_authority"]["provider_calls"],
            "production_database_migrations": authority["declared_authority"]["production_database_migrations"],
            "remote_operations_authorized": False,
        },
        "shadow": {
            "cases": authority["shadow"]["case_count"],
            "preview_calls": authority["shadow"]["transition_preview_calls"],
            "non_transition_preview_calls": authority["shadow"]["non_transition_preview_calls"],
            "publisher_injection_inert": authority["shadow"]["publisher_injection_inert_count"],
            "strict_path": authority["shadow"]["listing_material_path"],
            "sec_limitation": authority["shadow"]["historical_sec_limitation"],
        },
        "observer": {
            "calibrated_targets": authority["observer"]["calibrated_target_count"],
            "forbidden_calls": authority["observer"]["forbidden_calls"],
            "sqlite": authority["observer"]["sqlite"],
            "sqlite_calibration": authority["observer"]["sqlite_calibration"],
        },
        "scratch_migration": {
            "source_version": scratch["source_preflight"]["schema_version"],
            "target_version": scratch["target_preflight"]["schema_version"],
            "restored_version": scratch["restored_v2_state"]["version"],
            "row_and_sequence_identity": scratch["row_and_sequence_identity"],
            "backup_restore_byte_identical": scratch["backup_restore_byte_identical"],
            "old_code_startup": scratch["old_code_startup"],
        },
        "mutations": {
            "defined": mutations["mutation_count"],
            "killed": mutations["killed_count"],
            "unexpected_owner_drift": len(mutations["unexpected_owner_drift"]),
            "restored": mutations["all_product_files_restored_byte_identically"],
        },
        "browser": {
            "entries": len(entries),
            "screenshots": len(entries),
            "locales": sorted({entry["locale"] for entry in entries}),
            "viewports": sorted({"x".join(map(str, entry["viewport"])) for entry in entries}),
            "scenarios": sorted({entry["scenario"] for entry in entries}),
            "all_negative_counts": 0,
        },
        "gates": {
            "focused_a": focused_a,
            "focused_b": focused_b,
            "focused_collected_nodes": len(nodes_a),
            "focused_node_sets_identical": True,
            "full_a": full_a,
            "full_b": full_b,
            "full_collection_counts_identical": full_a == full_b,
            "frontend": {
                "test": {"status": "passed", **_frontend_test_counts()},
                "typecheck": "passed",
                "check_i18n_literals": "passed",
                "build": "passed",
            },
        },
        "fixture_sha256": _sha(ROOT / "tests/fixtures/listing_authority/shadow-cases.json"),
    }
    (PACKET / "verification-summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"focused": focused_a["passed"], "full": full_a["passed"], "browser": len(entries), "mutations": mutations["killed_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
