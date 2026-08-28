"""Verify an exact-v2 scratch database with an archived pre-v3 code tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--allowed-root", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    database = Path(args.database).resolve()
    allowed = Path(args.allowed_root).resolve()
    if database != allowed and allowed not in database.parents:
        raise SystemExit("scratch_path_guard")
    if not database.is_file():
        raise SystemExit("scratch_database_missing")

    sys.path.insert(0, str(repo))
    from src.security_lifecycle_schema import verify_profile_connection
    from src.ticker_identity_schema import verify_ticker_identity_connection

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        verify_profile_connection(connection)
        verify_ticker_identity_connection(connection)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    finally:
        connection.close()
    print(json.dumps({"foreign_key_violations": foreign_keys, "integrity": integrity, "old_code_started": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
