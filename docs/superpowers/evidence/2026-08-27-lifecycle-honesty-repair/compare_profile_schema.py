"""Fail closed on schema or protected-authority drift from the product base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MUTABLE_DISPOSITION_COLUMNS = {
    "disposition",
    "queue_bucket",
    "reason_code",
    "disposition_as_of",
}
STARTUP_DDL_FILES = (
    "src/security_lifecycle_schema.py",
    "src/ticker_identity_schema.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    head = json.loads(Path(args.head).read_text(encoding="utf-8"))
    object_equal = base["sqlite_master_owned"] == head["sqlite_master_owned"]
    table_equal = base["pragma_table_info"] == head["pragma_table_info"]
    index_equal = base["index_sql"] == head["index_sql"]
    mutable_columns = sorted(
        {
            column["name"]
            for columns in head["pragma_table_info"].values()
            for column in columns
            if column["name"] in MUTABLE_DISPOSITION_COLUMNS
        }
    )
    base_hashes = base["protected_file_sha256"]
    head_hashes = head["protected_file_sha256"]
    startup_changes = [
        path for path in STARTUP_DDL_FILES if base_hashes[path] != head_hashes[path]
    ]
    schema_equal = (
        base_hashes["src/security_lifecycle_schema.py"]
        == head_hashes["src/security_lifecycle_schema.py"]
    )
    transition_equal = (
        base_hashes["src/ticker_identity_transition.py"]
        == head_hashes["src/ticker_identity_transition.py"]
    )
    policy_file_equal = (
        base_hashes["src/security_lifecycle_decision_policy.py"]
        == head_hashes["src/security_lifecycle_decision_policy.py"]
    )
    constants = head["authority_constants"]
    checks = {
        "owned_sqlite_master_exact": object_equal,
        "pragma_table_info_exact": table_equal,
        "index_sql_exact": index_equal,
        "new_mutable_disposition_column_count_zero": not mutable_columns,
        "startup_ddl_changes_zero": not startup_changes,
        "security_lifecycle_schema_bytes_exact": schema_equal,
        "ticker_identity_transition_bytes_exact": transition_equal,
        "security_lifecycle_decision_policy_bytes_exact": policy_file_equal,
        "ticker_identity_transition_execution_revision_references_zero": (
            head["ticker_identity_transition_execution_revision_references"] == 0
        ),
        "automation_policy_version_v3": (
            constants["AUTOMATION_POLICY_VERSION"]
            == "trusted-lifecycle-automation-v3"
        ),
        "sec_shared_rule_version_3": constants["SEC_shared_RULE_VERSION"] == "3",
        "sec_deadline_only_rule_version_4": (
            constants["SEC_source_deadline_RULE_VERSION"] == "4"
        ),
    }
    report = {
        "schema_version": 1,
        "base_revision": base["revision"],
        "head_revision": head["revision"],
        "owned_sqlite_master_diff": "empty" if object_equal else "changed",
        "pragma_table_info_diff": "empty" if table_equal else "changed",
        "index_sql_diff": "empty" if index_equal else "changed",
        "new_mutable_disposition_columns": mutable_columns,
        "new_mutable_disposition_column_count": len(mutable_columns),
        "startup_ddl_changed_files": startup_changes,
        "startup_ddl_changes": len(startup_changes),
        "security_lifecycle_schema_byte_diff": "empty" if schema_equal else "changed",
        "ticker_identity_transition_byte_diff": "empty" if transition_equal else "changed",
        "security_lifecycle_decision_policy_byte_diff": (
            "empty" if policy_file_equal else "changed"
        ),
        "ticker_identity_transition_execution_revision_references": head[
            "ticker_identity_transition_execution_revision_references"
        ],
        "authority_constants": constants,
        "base_owned_object_count": len(base["sqlite_master_owned"]),
        "head_owned_object_count": len(head["sqlite_master_owned"]),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
