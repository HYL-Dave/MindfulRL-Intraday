"""Capture lifecycle-owned profile schema from a clean in-memory database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _owned(name: str) -> bool:
    return name.startswith("security_lifecycle_") or name.startswith(
        "ticker_identity_"
    ) or name.startswith("idx_security_lifecycle_") or name.startswith(
        "idx_ticker_identity_"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
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
        prohibited = {
            column["name"]
            for columns in table_info.values()
            for column in columns
            if column["name"] in {"disposition", "queue_bucket", "reason_code"}
        }
        payload = {
            "repo": str(repo),
            "objects": objects,
            "table_info": table_info,
            "prohibited_projection_columns": sorted(prohibited),
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
