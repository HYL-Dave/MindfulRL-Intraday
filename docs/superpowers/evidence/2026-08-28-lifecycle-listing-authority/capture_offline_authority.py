"""Capture listing authority and exact-v2/v3 migration in scratch storage only."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import unquote, urlparse


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
sys.path.insert(0, str(ROOT))


DECLARED = {
    "provider_calls": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_reads": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_writes": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_preflights": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_backups": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_migrations": {"value": 0, "basis": "declared_not_authorized"},
    "production_database_restores": {"value": 0, "basis": "declared_not_authorized"},
    "app_restarts": {"value": 0, "basis": "declared_not_authorized"},
    "merges": {"value": 0, "basis": "declared_not_authorized"},
    "pushes": {"value": 0, "basis": "declared_not_authorized"},
}

FORBIDDEN = (
    "requests.sessions.Session.request",
    "data_sources.sec_transport.SecTransport.get",
    "src.security_lifecycle_ibkr_evidence.read_ibkr_contract_evidence",
    "data_sources.listing_authority_transport.ListingAuthorityTransport.fetch_nasdaq",
    "data_sources.listing_authority_transport.ListingAuthorityTransport.fetch_massive_ticker",
    "src.security_lifecycle_listing_migration.migrate_listing_authority_schema",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_unavailable:{name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authority_targets():
    import requests
    from data_sources.sec_transport import SecTransport
    from data_sources.listing_authority_transport import ListingAuthorityTransport
    import src.security_lifecycle_ibkr_evidence as ibkr

    return (
        (FORBIDDEN[0], requests.sessions.Session, "request"),
        (FORBIDDEN[1], SecTransport, "get"),
        (FORBIDDEN[2], ibkr, "read_ibkr_contract_evidence"),
        (FORBIDDEN[3], ListingAuthorityTransport, "fetch_nasdaq"),
        (FORBIDDEN[4], ListingAuthorityTransport, "fetch_massive_ticker"),
    )


@contextmanager
def _offline_guard(allowed_root: Path):
    import src.security_lifecycle_listing_migration as migration

    counts = {name: 0 for name, _owner, _attribute in _authority_targets()}
    calibration = {name: 0 for name in FORBIDDEN}
    migration_counts = {"allowed_scratch": 0, "blocked_outside": 0}
    sqlite_counts = {
        "allowed_scratch": 0,
        "allowed_memory": 0,
        "blocked_outside": 0,
    }
    real_migrate = migration.migrate_listing_authority_schema
    real_connect = sqlite3.connect

    def normalize_database(database: object) -> Path | None:
        raw = os.fspath(database) if isinstance(database, os.PathLike) else str(database)
        if raw == ":memory:":
            return None
        if raw.startswith("file:"):
            parsed = urlparse(raw)
            raw = unquote(parsed.path)
        return Path(raw).resolve()

    def guarded_connect(database, *args, **kwargs):
        path = normalize_database(database)
        if path is None:
            sqlite_counts["allowed_memory"] += 1
        elif path != allowed_root and allowed_root not in path.parents:
            sqlite_counts["blocked_outside"] += 1
            raise AssertionError("sqlite_path_outside_scratch")
        else:
            sqlite_counts["allowed_scratch"] += 1
        return real_connect(database, *args, **kwargs)

    def guarded_migrate(path, *args, **kwargs):
        candidate = Path(path).resolve()
        if candidate != allowed_root and allowed_root not in candidate.parents:
            migration_counts["blocked_outside"] += 1
            raise AssertionError("migration_outside_scratch")
        migration_counts["allowed_scratch"] += 1
        return real_migrate(path, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(tempfile, "tempdir", str(allowed_root)))
        stack.enter_context(patch.object(sqlite3, "connect", guarded_connect))
        stack.enter_context(
            patch.object(migration, "migrate_listing_authority_schema", guarded_migrate)
        )
        for name, owner, attribute in _authority_targets():
            def reject(*_args, _name=name, **_kwargs):
                counts[_name] += 1
                raise AssertionError(f"forbidden_authority_call:{_name}")

            stack.enter_context(patch.object(owner, attribute, reject))

        for name, owner, attribute in _authority_targets():
            try:
                getattr(owner, attribute)()
            except AssertionError as exc:
                assert str(exc) == f"forbidden_authority_call:{name}"
                calibration[name] += 1
            else:
                raise AssertionError(f"authority_observer_inactive:{name}")
        for key in counts:
            counts[key] = 0
        try:
            migration.migrate_listing_authority_schema(allowed_root.parent / "outside.db")
        except AssertionError as exc:
            assert str(exc) == "migration_outside_scratch"
        else:
            raise AssertionError("migration_guard_inactive")
        calibration[FORBIDDEN[5]] += 1

        outside_sqlite = allowed_root.parent / "arkscope-listing-outside.db"
        outside_sqlite.unlink(missing_ok=True)
        try:
            sqlite3.connect(outside_sqlite)
        except AssertionError as exc:
            assert str(exc) == "sqlite_path_outside_scratch"
        else:
            raise AssertionError("sqlite_path_guard_inactive")
        assert not outside_sqlite.exists()
        inside_sqlite = allowed_root / "sqlite-guard-calibration.db"
        connection = sqlite3.connect(inside_sqlite)
        connection.close()
        assert inside_sqlite.is_file()
        inside_sqlite.unlink()
        sqlite_calibration = dict(sqlite_counts)
        assert sqlite_calibration == {
            "allowed_scratch": 1,
            "allowed_memory": 0,
            "blocked_outside": 1,
        }
        for key in sqlite_counts:
            sqlite_counts[key] = 0
        yield {
            "forbidden_calls": counts,
            "calibration": calibration,
            "migration": migration_counts,
            "sqlite": sqlite_counts,
            "sqlite_calibration": sqlite_calibration,
        }


def _db_state(path: Path, *, version: str) -> dict:
    from src.security_lifecycle_schema import (
        verify_profile_connection,
        verify_v2_profile_connection,
    )
    from src.ticker_identity_schema import verify_ticker_identity_connection

    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        if version == "v2":
            verify_v2_profile_connection(connection)
        else:
            verify_profile_connection(connection)
        verify_ticker_identity_connection(connection)
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND (name LIKE 'security_lifecycle_%' OR name LIKE 'ticker_identity_%') "
                "ORDER BY name"
            )
        ]
        counts = {table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) for table in tables}
        return {
            "version": version,
            "integrity": str(connection.execute("PRAGMA integrity_check").fetchone()[0]),
            "foreign_key_violations": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
            "table_counts": counts,
        }
    finally:
        connection.close()


def _scratch_migration(root: Path, old_code_root: Path) -> dict:
    import src.security_lifecycle_listing_migration as migration

    helpers = _load_module(
        "listing_migration_helpers",
        ROOT / "tests/test_security_lifecycle_listing_migration.py",
    )
    source = helpers._seeded_v2_profile(root, "source-v2.db")
    before = migration.preflight_listing_authority_migration(source)
    before_state = _db_state(source, version="v2")
    backup_dir = root / "backup"
    backup_dir.mkdir(mode=0o700)
    backup = migration.create_listing_authority_backup(
        source,
        backup_dir / "profile-v2.db",
        approval_sha256=before.approval_sha256,
    )
    changed = migration.migrate_listing_authority_schema(
        source,
        approval_sha256=before.approval_sha256,
        backup_sha256=backup.sha256,
    )
    after = migration.preflight_listing_authority_migration(source)
    after_state = _db_state(source, version="v3")
    restored = root / "restored-v2.db"
    restored_result = migration.restore_listing_authority_backup(
        restored,
        backup.path,
        backup_sha256=backup.sha256,
    )
    restored_state = _db_state(restored, version="v2")
    old = subprocess.run(
        [
            sys.executable,
            str(PACKET / "verify_old_code.py"),
            "--repo",
            str(old_code_root),
            "--database",
            str(restored),
            "--allowed-root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    old_result = json.loads(old.stdout)
    assert old_result["old_code_started"] is True
    assert old_result["integrity"] == "ok"
    assert old_result["foreign_key_violations"] == 0
    old_guard = old_result["sqlite_guard"]
    assert old_guard["child_owned"] is True
    assert old_guard["path_resolution"] == (
        "file_uri_unquoted_and_symlinks_resolved_before_containment"
    )
    assert old_guard["outside_calibration"] == {
        "attempts": 1,
        "rejected_before_access": True,
        "symlink_uri_resolved_before_containment": True,
        "target_created": False,
    }
    assert old_guard["inside_calibration"] == {
        "attempts": 1,
        "delegated_connect_calls": 1,
        "file_created": True,
    }
    assert old_guard["actual_restored_database"] == {
        "contained_after_resolution": True,
        "read_only_opens": 1,
    }
    assert old_guard["counts"] == {
        "allowed_inside": 2,
        "allowed_inside_read_only": 1,
        "blocked_outside_before_access": 1,
        "delegated_connect_calls": 2,
        "file_backed_attempts": 3,
    }
    assert before.owned_rows_sha256 == after.owned_rows_sha256
    assert before.owned_table_row_sha256 == after.owned_table_row_sha256
    assert before.owned_table_rowid_sha256 == after.owned_table_rowid_sha256
    assert before.lifecycle_sequences == after.lifecycle_sequences
    assert restored.read_bytes() == backup.path.read_bytes()
    assert before_state["table_counts"] == after_state["table_counts"] == restored_state["table_counts"]
    return {
        "source_preflight": asdict(before),
        "migration": asdict(changed),
        "target_preflight": asdict(after),
        "restore": {**asdict(restored_result), "path": "scratch/restored-v2.db"},
        "backup": {"sha256": backup.sha256, "mode": oct(backup.path.stat().st_mode & 0o777)},
        "v2_state": before_state,
        "v3_state": after_state,
        "restored_v2_state": restored_state,
        "old_code_startup": old_result,
        "row_and_sequence_identity": True,
        "backup_restore_byte_identical": True,
    }


def capture(old_code_root: Path) -> dict:
    shadow = _load_module("listing_shadow", PACKET / "run_shadow.py")
    with TemporaryDirectory(prefix="arkscope-listing-authority-") as directory:
        scratch = Path(directory).resolve()
        with _offline_guard(scratch) as observed:
            shadow_result = shadow.run()
            migration_result = _scratch_migration(scratch, old_code_root)
            assert all(value == 0 for value in observed["forbidden_calls"].values())
            assert all(value == 1 for value in observed["calibration"].values())
            assert observed["migration"] == {"allowed_scratch": 1, "blocked_outside": 1}
            assert observed["sqlite"]["blocked_outside"] == 0
        return {
            "schema_version": 2,
            "scope": "offline_fixture_and_scratch_only",
            "declared_authority": DECLARED,
            "observer": {
                "method": "calibrated_fail_closed_wrappers_and_sqlite_path_guard",
                "calibrated_target_count": len(observed["calibration"]) + 1,
                **observed,
                "sqlite_path_guard": "all file-backed opens constrained to designated temporary root; in-memory opens have no filesystem path",
            },
            "shadow": shadow_result,
            "scratch_migration": migration_result,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-code-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = capture(Path(args.old_code_root).resolve())
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"calibrated_targets": payload["observer"]["calibrated_target_count"], "shadow_cases": payload["shadow"]["case_count"], "scratch_migrations": payload["observer"]["migration"]["allowed_scratch"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
