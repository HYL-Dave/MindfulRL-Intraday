"""Capture owned profile schema and protected authority from a repository tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


OWNED_PREFIXES = (
    "security_lifecycle_",
    "ticker_identity_",
    "idx_security_lifecycle_",
    "idx_ticker_identity_",
)
PROTECTED_FILES = (
    "src/security_lifecycle_schema.py",
    "src/ticker_identity_schema.py",
    "src/ticker_identity_transition.py",
    "src/security_lifecycle_decision_policy.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owned(name: str) -> bool:
    return name.startswith(OWNED_PREFIXES)


def _literal_constant(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            raise ValueError(f"non_scalar_constant:{name}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from src.security_lifecycle_schema import create_profile_schema
    from src.ticker_identity_schema import create_ticker_identity_schema

    connection = sqlite3.connect(":memory:")
    try:
        create_profile_schema(connection)
        create_ticker_identity_schema(connection)
        objects = [
            {
                "type": str(row[0]),
                "name": str(row[1]),
                "table": str(row[2]),
                "sql": str(row[3]),
            }
            for row in connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master "
                "WHERE sql IS NOT NULL ORDER BY type,name"
            )
            if _owned(str(row[1]))
        ]
        table_info = {
            item["name"]: [
                {
                    "cid": int(row[0]),
                    "name": str(row[1]),
                    "type": str(row[2]),
                    "notnull": int(row[3]),
                    "default": row[4],
                    "pk": int(row[5]),
                }
                for row in connection.execute(
                    f'PRAGMA table_info("{item["name"]}")'
                )
            ]
            for item in objects
            if item["type"] == "table"
        }
        index_sql = {
            item["name"]: item["sql"] for item in objects if item["type"] == "index"
        }
        transition_source = (repo / "src/ticker_identity_transition.py").read_text(
            encoding="utf-8"
        )
        payload = {
            "schema_version": 1,
            "revision": args.revision,
            "sqlite_master_owned": objects,
            "pragma_table_info": table_info,
            "index_sql": index_sql,
            "protected_file_sha256": {
                path: _sha256(repo / path) for path in PROTECTED_FILES
            },
            "authority_constants": {
                "AUTOMATION_POLICY_VERSION": _literal_constant(
                    repo / "src/security_lifecycle_decision_policy.py",
                    "AUTOMATION_POLICY_VERSION",
                ),
                "SEC_shared_RULE_VERSION": _literal_constant(
                    repo / "src/security_lifecycle_sec_evidence.py", "_RULE_VERSION"
                ),
                "SEC_source_deadline_RULE_VERSION": _literal_constant(
                    repo / "src/security_lifecycle_sec_evidence.py",
                    "_SOURCE_DEADLINE_RULE_VERSION",
                ),
            },
            "ticker_identity_transition_execution_revision_references": (
                transition_source.count("execution_revision")
            ),
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
