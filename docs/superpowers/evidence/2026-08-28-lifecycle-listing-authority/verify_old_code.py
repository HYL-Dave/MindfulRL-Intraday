"""Verify an exact-v2 scratch database with an archived pre-v3 code tree."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse


def _resolve_database_path(database: object) -> Path | None:
    raw = os.fspath(database) if isinstance(database, os.PathLike) else str(database)
    if raw == ":memory:":
        return None
    if raw.startswith("file:"):
        parsed = urlparse(raw)
        if parsed.netloc not in {"", "localhost"}:
            raise AssertionError("sqlite_uri_authority_forbidden")
        raw = unquote(parsed.path)
    return Path(raw).resolve()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _read_only_uri(database: Path) -> str:
    return f"{database.resolve().as_uri()}?mode=ro"


def _uri_mode(database: object) -> str | None:
    raw = os.fspath(database) if isinstance(database, os.PathLike) else str(database)
    if not raw.startswith("file:"):
        return None
    values = parse_qs(urlparse(raw).query).get("mode", [])
    return values[0] if len(values) == 1 else None


@contextmanager
def _sqlite_guard(allowed_root: Path):
    root = allowed_root.resolve()
    real_connect = sqlite3.connect
    state = {
        "file_backed_attempts": 0,
        "blocked_outside_before_access": 0,
        "allowed_inside": 0,
        "allowed_inside_read_only": 0,
        "delegated_connect_calls": 0,
    }

    def guarded_connect(database, *args, **kwargs):
        path = _resolve_database_path(database)
        if path is None:
            raise AssertionError("sqlite_memory_forbidden_in_old_code_probe")
        state["file_backed_attempts"] += 1
        if not _is_within(path, root):
            state["blocked_outside_before_access"] += 1
            raise AssertionError("sqlite_path_outside_scratch")
        state["allowed_inside"] += 1
        if _uri_mode(database) == "ro" and kwargs.get("uri") is True:
            state["allowed_inside_read_only"] += 1
        state["delegated_connect_calls"] += 1
        return real_connect(database, *args, **kwargs)

    with patch.object(sqlite3, "connect", guarded_connect):
        yield state


def _calibrate_outside(root: Path, state: dict) -> dict:
    outside = root.parent / "task8-old-code-outside.db"
    outside.unlink(missing_ok=True)
    link = root / "outside-link.db"
    link.unlink(missing_ok=True)
    link.symlink_to(outside)
    delegated_before = state["delegated_connect_calls"]
    try:
        sqlite3.connect(f"{link.as_uri()}?mode=rwc", uri=True)
    except AssertionError as exc:
        assert str(exc) == "sqlite_path_outside_scratch"
    else:
        raise AssertionError("old_code_sqlite_outside_guard_inactive")
    resolved_through_symlink = _resolve_database_path(
        f"{link.as_uri()}?mode=rwc"
    ) == outside.resolve()
    link.unlink()
    result = {
        "attempts": 1,
        "rejected_before_access": (
            state["delegated_connect_calls"] == delegated_before
        ),
        "target_created": outside.exists(),
        "symlink_uri_resolved_before_containment": resolved_through_symlink,
    }
    outside.unlink(missing_ok=True)
    assert result == {
        "attempts": 1,
        "rejected_before_access": True,
        "target_created": False,
        "symlink_uri_resolved_before_containment": True,
    }
    return result


def _calibrate_inside(root: Path, state: dict) -> dict:
    inside = root / "old-code-inside-calibration.db"
    inside.unlink(missing_ok=True)
    delegated_before = state["delegated_connect_calls"]
    connection = sqlite3.connect(f"{inside.as_uri()}?mode=rwc", uri=True)
    connection.close()
    result = {
        "attempts": 1,
        "delegated_connect_calls": state["delegated_connect_calls"] - delegated_before,
        "file_created": inside.is_file(),
    }
    inside.unlink()
    assert result == {
        "attempts": 1,
        "delegated_connect_calls": 1,
        "file_created": True,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--allowed-root", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    database = Path(args.database).resolve()
    allowed = Path(args.allowed_root).resolve()
    if not _is_within(database, allowed):
        raise SystemExit("scratch_path_guard")
    if not database.is_file():
        raise SystemExit("scratch_database_missing")

    sys.path.insert(0, str(repo))
    from src.security_lifecycle_schema import verify_profile_connection
    from src.ticker_identity_schema import verify_ticker_identity_connection

    with _sqlite_guard(allowed) as guard:
        outside_calibration = _calibrate_outside(allowed, guard)
        inside_calibration = _calibrate_inside(allowed, guard)
        read_only_before = guard["allowed_inside_read_only"]
        connection = sqlite3.connect(_read_only_uri(database), uri=True)
        try:
            verify_profile_connection(connection)
            verify_ticker_identity_connection(connection)
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        finally:
            connection.close()
        actual_read_only_opens = guard["allowed_inside_read_only"] - read_only_before

    assert integrity == "ok"
    assert foreign_keys == 0
    assert actual_read_only_opens == 1
    assert guard == {
        "file_backed_attempts": 3,
        "blocked_outside_before_access": 1,
        "allowed_inside": 2,
        "allowed_inside_read_only": 1,
        "delegated_connect_calls": 2,
    }
    result = {
        "foreign_key_violations": foreign_keys,
        "integrity": integrity,
        "sqlite_guard": {
            "child_owned": True,
            "path_resolution": "file_uri_unquoted_and_symlinks_resolved_before_containment",
            "outside_calibration": outside_calibration,
            "inside_calibration": inside_calibration,
            "actual_restored_database": {
                "contained_after_resolution": _is_within(database, allowed),
                "read_only_opens": actual_read_only_opens,
            },
            "counts": guard,
        },
        "old_code_started": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
