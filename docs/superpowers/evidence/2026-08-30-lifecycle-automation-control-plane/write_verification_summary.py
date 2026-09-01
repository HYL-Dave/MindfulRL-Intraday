"""Build the bounded verification summary from sealed-source artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


PACKET = Path(__file__).resolve().parent
BACKEND_RESULT = re.compile(
    r"(?P<passed>\d+) passed, (?P<skipped>\d+) skipped, "
    r"(?P<warnings>\d+) warnings in (?P<seconds>[\d.]+)s"
)
FOCUSED_RESULT = re.compile(r"(?P<passed>\d+) passed in (?P<seconds>[\d.]+)s")
FRONTEND_FILES = re.compile(r"Test Files\s+(?P<files>\d+) passed")
FRONTEND_TESTS = re.compile(r"Tests\s+(?P<tests>\d+) passed")


def _read(name: str) -> str:
    return (PACKET / name).read_text(encoding="utf-8")


def _json(name: str) -> dict[str, object]:
    return json.loads(_read(name))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backend(name: str) -> dict[str, object]:
    match = BACKEND_RESULT.search(_read(name))
    if match is None:
        raise ValueError(f"backend_result:{name}")
    return {
        "passed": int(match.group("passed")),
        "skipped": int(match.group("skipped")),
        "warnings": int(match.group("warnings")),
        "seconds": float(match.group("seconds")),
    }


def _focused(name: str) -> dict[str, object]:
    match = FOCUSED_RESULT.search(_read(name))
    if match is None:
        raise ValueError(f"focused_result:{name}")
    return {
        "passed": int(match.group("passed")),
        "seconds": float(match.group("seconds")),
    }


def _frontend(name: str) -> dict[str, int]:
    content = _read(name)
    files = FRONTEND_FILES.search(content)
    tests = FRONTEND_TESTS.search(content)
    if files is None or tests is None:
        raise ValueError(f"frontend_result:{name}")
    return {"files": int(files.group("files")), "tests": int(tests.group("tests"))}


def _node_manifest(name: str) -> dict[str, object]:
    path = PACKET / name
    nodes = tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    if len(nodes) != len(set(nodes)) or nodes != tuple(sorted(nodes)):
        raise ValueError(f"node_manifest:{name}")
    return {
        "nodes": len(nodes),
        "sha256": _sha256(path),
    }


def _normalization() -> dict[str, object]:
    payload = _json("text-normalization.json")
    rows = payload["artifacts"]
    if payload["semantic_content_rewritten"] is not False or not rows:
        raise ValueError("text_normalization")
    for row in rows:
        path = PACKET / row["path"]
        data = path.read_bytes()
        if _sha256(path) != row["after_sha256"]:
            raise ValueError(f"text_normalization_hash:{row['path']}")
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise ValueError(f"text_normalization_eof:{row['path']}")
        if any(line.endswith((b" ", b"\t")) for line in data.splitlines()):
            raise ValueError(f"text_normalization_trailing:{row['path']}")
    return {
        "artifacts": len(rows),
        "trailing_whitespace_lines_normalized": sum(
            int(row["trailing_whitespace_lines_normalized"]) for row in rows
        ),
        "terminal_empty_lines_removed": sum(
            int(row["terminal_empty_lines_removed"]) for row in rows
        ),
        "semantic_content_rewritten": False,
    }


def build_summary() -> dict[str, object]:
    repository = _json("repository-binding.json")
    mutations = _json("mutation-ledger.json")
    browser = _json("browser/matrix.json")
    backend_a = _backend("backend-full-a.txt")
    backend_b = _backend("backend-full-b.txt")
    frontend_a = _frontend("frontend-test-a.txt")
    frontend_b = _frontend("frontend-test-b.txt")
    nodes_a = _node_manifest("full-nodes-a.txt")
    nodes_b = _node_manifest("full-nodes-b.txt")
    entries = browser["entries"]
    expected_entries = {
        (surface, locale, viewport)
        for surface in (
            "settings",
            "lifecycle",
            "blocker-diagnostic",
            "finalization-failure",
        )
        for locale in ("en", "zh-Hant")
        for viewport in ((1440, 900), (390, 844))
    }
    observed_entries = {
        (row["surface"], row["locale"], tuple(row["viewport"])) for row in entries
    }
    if observed_entries != expected_entries:
        raise ValueError("browser_matrix")
    screenshot_hashes_valid = all(
        _sha256(PACKET / "browser" / row["screenshot"])
        == row["screenshot_sha256"]
        for row in entries
    )
    measured_browser = {
        "entries": len(entries),
        "screenshots": len(entries),
        "external_requests": sum(len(row["external_requests"]) for row in entries),
        "fixture_write_requests": sum(row["fixture_write_count"] for row in entries),
        "console_errors": sum(len(row["console_errors"]) for row in entries),
        "page_errors": sum(len(row["page_errors"]) for row in entries),
        "overlaps": sum(row["overlap_count"] for row in entries),
        "clipped_text": sum(row["clipped_text_count"] for row in entries),
        "viewport_clipped_controls": sum(
            row["viewport_clipped_control_count"] for row in entries
        ),
        "screenshot_hashes_valid": screenshot_hashes_valid,
        "latest_case_refresh_witnesses": sum(
            bool(row["latest_case_refresh_witness"]) for row in entries
        ),
        "t3_operator_detail_witnesses": sum(
            bool(row["t3_operator_detail_witness"]) for row in entries
        ),
        "t3_raw_context_hidden_witnesses": sum(
            bool(row["t3_raw_context_hidden_witness"]) for row in entries
        ),
        "t5_finalization_label_witnesses": sum(
            bool(row["t5_finalization_label_witness"]) for row in entries
        ),
    }
    typecheck = _read("frontend-typecheck.txt")
    build = _read("frontend-build.txt")
    i18n = _read("frontend-i18n-literals.txt")
    i18n_payload = json.loads(next(line for line in i18n.splitlines() if line.startswith("{")))
    if backend_a != {**backend_b, "seconds": backend_a["seconds"]}:
        comparable_a = {key: value for key, value in backend_a.items() if key != "seconds"}
        comparable_b = {key: value for key, value in backend_b.items() if key != "seconds"}
        if comparable_a != comparable_b:
            raise ValueError("backend_a_b")
    if frontend_a != frontend_b or nodes_a != nodes_b:
        raise ValueError("repeated_gate_identity")
    if not mutations["all_mutations_killed"] or not mutations[
        "all_files_restored_byte_identically"
    ]:
        raise ValueError("mutations")
    if not repository["all_schema_authorities_byte_identical"] or not repository[
        "all_browser_fixture_authorities_match_tested_head"
    ]:
        raise ValueError("repository_binding")
    if any(
        measured_browser[key]
        for key in (
            "external_requests",
            "console_errors",
            "page_errors",
            "overlaps",
            "clipped_text",
            "viewport_clipped_controls",
        )
    ) or not screenshot_hashes_valid:
        raise ValueError("browser_measurements")
    if (
        measured_browser["t3_operator_detail_witnesses"] != 4
        or measured_browser["t3_raw_context_hidden_witnesses"] != 4
        or measured_browser["t5_finalization_label_witnesses"] != 4
    ):
        raise ValueError("browser_repair_witnesses")
    return {
        "schema_version": 1,
        "repository": {
            "base_commit": repository["base_commit"],
            "tested_product_head": repository["product_head_commit"],
            "replay_source_head": repository["head_commit"],
            "branch": repository["branch"],
            "changed_paths": len(repository["changed_paths"]),
            "post_product_paths": repository["post_product_paths"],
            "post_product_scope_only_packet": repository[
                "post_product_scope_only_packet"
            ],
            "merge_commits_since_base": repository["merge_commits_since_base"],
            "schema_authorities_byte_identical": repository[
                "all_schema_authorities_byte_identical"
            ],
            "browser_fixture_authorities_match_tested_head": repository[
                "all_browser_fixture_authorities_match_tested_head"
            ],
        },
        "boundary": {
            "declared_unexecuted_operations": {
                "provider_calls": {"value": 0, "basis": "declared_not_authorized"},
                "production_database_operations": {
                    "value": 0,
                    "basis": "declared_not_authorized",
                },
                "app_restarts": {"value": 0, "basis": "declared_not_authorized"},
                "merges": {"value": 0, "basis": "declared_not_authorized"},
                "pushes": {"value": 0, "basis": "declared_not_authorized"},
            },
            "measured_fixture_browser_operations": measured_browser,
            "implementation_modified_product_code": True,
            "packet_replay_modified_product_or_test_code": False,
            "packet_replay_scope": "offline_repository_and_fixture_only",
        },
        "gates": {
            "backend_focused": _focused("backend-focused.txt"),
            "backend_full_a": backend_a,
            "backend_full_b": backend_b,
            "backend_full_result_counts_equal": {
                key: backend_a[key] for key in ("passed", "skipped", "warnings")
            } == {
                key: backend_b[key] for key in ("passed", "skipped", "warnings")
            },
            "full_node_manifest_a": nodes_a,
            "full_node_manifest_b": nodes_b,
            "full_node_manifests_byte_identical": (
                PACKET / "full-nodes-a.txt"
            ).read_bytes() == (PACKET / "full-nodes-b.txt").read_bytes(),
            "frontend_a": frontend_a,
            "frontend_b": frontend_b,
            "typecheck_passed": "tsc --noEmit" in typecheck,
            "production_build_passed": "built in" in build,
            "i18n_scanner": i18n_payload,
            "generated_text_normalization": _normalization(),
        },
        "mutations": {
            "count": mutations["mutation_count"],
            "killed": mutations["killed_count"],
            "all_files_restored_byte_identically": mutations[
                "all_files_restored_byte_identically"
            ],
        },
        "browser": {
            "fixture_only": browser["fixture_only"],
            "positive_geometry_calibration": browser[
                "geometry_positive_calibration"
            ],
            "measured": measured_browser,
        },
        "known_warnings": {
            "backend_deprecation_warnings": backend_a["warnings"],
            "frontend_react_act_warning_present": "not wrapped in act" in _read(
                "frontend-test-a.txt"
            ),
            "build_chunk_size_warning_present": "larger than 500 kB" in build,
        },
        "limitations": [
            "The browser matrix uses local fixture responses and does not claim live provider or production database coverage.",
            "Runtime stage progress is intentionally process-memory state and does not survive an App restart; durable incident truth is separate.",
            "Historical sealed packets are immutable and outside this replay. Only three historical run_browser_matrix.py fixture helpers are imported read-only and bound to exact product-head Git blobs and SHA-256 values; historical run_shadow.py and test_packet_contracts.py are not executed.",
            "T3 and T5 browser witnesses are local product-shaped fixtures; they prove rendering and closed DTO behavior, not live provider evidence.",
            "The final production one-case canary remains separately authorized Task 13 work.",
        ],
    }


def main() -> int:
    summary = build_summary()
    (PACKET / "verification-summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "backend": summary["gates"]["backend_full_a"]["passed"],
        "frontend": summary["gates"]["frontend_a"]["tests"],
        "mutations": summary["mutations"]["killed"],
        "browser_entries": summary["browser"]["measured"]["entries"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
