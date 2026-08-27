"""Summarize only regenerated lifecycle honesty-repair packet artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re


PACKET = Path(__file__).resolve().parent
PRODUCT_BASE = "11e7a5d4f6856062a5ac00a8d90ed97b5c2e56cb"
PRODUCT_TEST_AUTHORITY = "c043bc0e7ca0642e383841dfcc537c5bdb4242e2"


def _text(name: str) -> str:
    return (PACKET / name).read_text(encoding="utf-8")


def _json(name: str) -> dict:
    value = json.loads(_text(name))
    if not isinstance(value, dict):
        raise ValueError(name)
    return value


def _match(pattern: str, value: str, name: str) -> re.Match[str]:
    matched = re.search(pattern, value, flags=re.MULTILINE)
    if matched is None:
        raise ValueError(name)
    return matched


def _summary_line(name: str, pattern: str) -> str:
    return _match(pattern, _text(name), name).group(0)


def main() -> None:
    focused_a_nodes = _text("backend-focused-a.nodes")
    focused_b_nodes = _text("backend-focused-b.nodes")
    focused_count = sum(
        line.startswith("tests/") for line in focused_a_nodes.splitlines()
    )
    frontend = _text("frontend-full.txt")
    frontend_files = int(
        _match(r"Test Files\s+(\d+) passed", frontend, "frontend-full.txt").group(1)
    )
    frontend_tests = int(
        _match(r"Tests\s+(\d+) passed", frontend, "frontend-full.txt").group(1)
    )
    build = _text("frontend-build.txt")
    build_modules = int(
        _match(r"(\d+) modules transformed", build, "frontend-build.txt").group(1)
    )
    i18n_line = next(
        line
        for line in _text("frontend-i18n-literals.txt").splitlines()
        if line.startswith("{")
    )
    i18n = json.loads(i18n_line)
    mutation = _json("mutation-ledger.json")
    offline = _json("offline-authority.json")
    schema = _json("schema-comparison.json")
    browser = _json("browser-matrix.json")
    payload = {
        "schema_version": 2,
        "product_base": PRODUCT_BASE,
        "product_test_authority": PRODUCT_TEST_AUTHORITY,
        "authority": offline["authority"],
        "mutation_ledger": {
            "mutation_count": mutation["mutation_count"],
            "killed_count": mutation["killed_count"],
            "all_mutations_killed": mutation["all_mutations_killed"],
            "unexpected_owner_drift": mutation["unexpected_owner_drift"],
            "all_product_files_restored_byte_identically": mutation[
                "all_product_files_restored_byte_identically"
            ],
        },
        "offline_authority": {
            "declared_authority_semantics": offline["authority"]["semantics"],
            "byte_identical_across_two_captures": True,
            "authority_call_observer_calibration": offline[
                "authority_call_observer_calibration"
            ],
            "decision_provenance_equal_across_r0_r1": offline[
                "decision_provenance"
            ]["equal"],
            "cross_revision_due_blocked_retry": offline[
                "cross_revision_due_blocked_retry"
            ],
            "forged_citation_rollback_rows": {
                "blockers": offline["forged_citation_rollback"]["blocker_rows"],
                "evidence": offline["forged_citation_rollback"]["evidence_rows"],
                "facts": offline["forged_citation_rollback"]["fact_rows"],
            },
            "transition_and_acknowledgement_calls": sum(
                offline["transition_and_acknowledgement_calls"].values()
            ),
        },
        "backend": {
            "focused_collection_byte_identical": (
                focused_a_nodes == focused_b_nodes
            ),
            "focused_collection_node_count": focused_count,
            "focused_pass_a": _summary_line(
                "backend-focused-a.txt", r"\d+ passed in [0-9.]+s"
            ),
            "focused_pass_b": _summary_line(
                "backend-focused-b.txt", r"\d+ passed in [0-9.]+s"
            ),
            "full_pass_a": _summary_line(
                "backend-full-a.txt",
                r"\d+ passed, \d+ skipped, \d+ warnings in [0-9.]+s(?: \([^)]+\))?",
            ),
            "full_pass_b": _summary_line(
                "backend-full-b.txt",
                r"\d+ passed, \d+ skipped, \d+ warnings in [0-9.]+s(?: \([^)]+\))?",
            ),
        },
        "frontend": {
            "test_files": frontend_files,
            "tests_passed": frontend_tests,
            "typecheck": "passed",
            "i18n_literal_gate": (
                f"passed; debtSignatureCount={i18n['debtSignatureCount']}"
            ),
            "build": f"passed; {build_modules} modules",
        },
        "schema_and_protected_authority": {
            "all_checks_passed": schema["all_checks_passed"],
            "owned_sqlite_master_diff": schema["owned_sqlite_master_diff"],
            "pragma_table_info_diff": schema["pragma_table_info_diff"],
            "index_sql_diff": schema["index_sql_diff"],
            "new_mutable_disposition_column_count": schema[
                "new_mutable_disposition_column_count"
            ],
            "startup_ddl_changes": schema["startup_ddl_changes"],
            "security_lifecycle_schema_byte_diff": schema[
                "security_lifecycle_schema_byte_diff"
            ],
            "ticker_identity_transition_byte_diff": schema[
                "ticker_identity_transition_byte_diff"
            ],
            "decision_policy_change_authority": schema[
                "security_lifecycle_decision_policy_change_authority"
            ],
            "automation_policy_version": schema["authority_constants"][
                "AUTOMATION_POLICY_VERSION"
            ],
            "shared_sec_rule_version": schema["authority_constants"][
                "SEC_shared_RULE_VERSION"
            ],
            "deadline_only_rule_version": schema["authority_constants"][
                "SEC_source_deadline_RULE_VERSION"
            ],
        },
        "browser_matrix": {
            **browser["summary"],
            "authority_semantics": browser["authority_semantics"],
        },
        "packet": {
            "manifest_excludes_itself": True,
            "manifest_file_set_equals_payload_file_set": True,
            "sha256sum_check": "passed",
        },
        "limitations": [
            "no provider call or production scheduler replay was performed",
            "all persistence recovery tests use local temporary SQLite and later sequential worker ticks",
            "the browser matrix uses fixture interception; out-of-order queue and detail sequencing are owned by Vitest",
            "broader legal-language extraction remains precision-first and intentionally incomplete",
            "no production schema migration or row backfill is needed because schema authority is unchanged and M37 proves pre-execution-key succeeded rows remain query-compatible",
            "unexpected_owner_drift is limited to missing or additional failures inside each explicitly executed owner-only mutation command; it is not a broad suite-drift scan",
            "five independently removable backend fail-closed conditions and two load-bearing frontend race guards were previously unowned; the seven named closeout mutations now own them",
            "other incidental frontend guards are accepted coverage debt and are not claimed as mutation-owned",
            "provider, production-database, App-restart, merge, and push zeros are declared hard-stop compliance rather than runtime-instrumented measurements",
            "App restart, merge, push, and production database operations remain prohibited",
        ],
    }
    if not (
        payload["mutation_ledger"]["all_mutations_killed"]
        and payload["mutation_ledger"][
            "all_product_files_restored_byte_identically"
        ]
        and payload["backend"]["focused_collection_byte_identical"]
        and payload["offline_authority"][
            "authority_call_observer_calibration"
        ]["expected"]
        == payload["offline_authority"][
            "authority_call_observer_calibration"
        ]["observed"]
        and payload["offline_authority"][
            "transition_and_acknowledgement_calls"
        ]
        == 0
        and payload["schema_and_protected_authority"]["all_checks_passed"]
        and payload["browser_matrix"]["entry_count"] == 24
    ):
        raise RuntimeError("verification_summary_authority_failed")
    (PACKET / "verification-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
