from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path


def _load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("eir006_destructive_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load destructive controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_not_held(module, path: Path) -> None:
    records = module._lsof_records([str(path)])
    if records:
        raise AssertionError(f"read-only connection remained open: {records}")


def _create_fixture(module, production_db: Path, fixture_db: Path) -> None:
    target = sqlite3.connect(fixture_db)
    try:
        with module._connect_ro(production_db) as source:
            target.executescript(
                """
                CREATE TABLE financial_cache (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    data TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE fundamentals (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE market_sync_meta (
                    domain TEXT PRIMARY KEY,
                    last_success TEXT,
                    last_error TEXT,
                    rows_added INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE ticker_aliases (
                    alias TEXT PRIMARY KEY,
                    canonical TEXT NOT NULL
                );
                """
            )
            copies = (
                (
                    "financial_cache",
                    "cache_key, source, ticker, data, fetched_at, expires_at",
                    6,
                ),
                ("fundamentals", "id, ticker, snapshot_date, data", 4),
                (
                    "market_sync_meta",
                    "domain, last_success, last_error, rows_added, updated_at",
                    5,
                ),
                ("ticker_aliases", "alias, canonical", 2),
            )
            for table, columns, width in copies:
                rows = list(source.execute(f"SELECT {columns} FROM {table}"))
                placeholders = ",".join("?" for _ in range(width))
                target.executemany(
                    f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                    [tuple(row) for row in rows],
                )
        target.commit()
    finally:
        target.close()


def _probe_file_transport(module, scratch_root: Path) -> None:
    repo = scratch_root / "file-fixture"
    files = {
        "data/prices/15min/AAA_15min_2024_2026.csv": b"15min\n",
        "data/prices/hourly/AAA_hourly_2023.csv": b"hourly\n",
        "data/prices/collection_summary.json": b"{}\n",
    }
    rows = []
    for relative, payload in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        value = path.stat()
        rows.append({"relative_path": relative, "inode": str(value.st_ino)})
    quarantine = scratch_root / "file-quarantine"
    quarantine.mkdir()
    (quarantine / "files").mkdir()
    original_quarantine = module.QUARANTINE_ROOT
    module.QUARANTINE_ROOT = quarantine
    moved: list[str] = []
    try:
        module._move_to_quarantine(repo, rows, moved)
        if moved != list(files):
            raise AssertionError(f"unexpected move ledger: {moved}")
        if (repo / "data" / "prices").exists():
            raise AssertionError("source price root remained after move")
        module._move_back(repo, moved)
        for relative, payload in files.items():
            if (repo / relative).read_bytes() != payload:
                raise AssertionError(f"restored file differs: {relative}")
    finally:
        module.QUARANTINE_ROOT = original_quarantine


def _run(repo_root: Path, scratch_root: Path) -> None:
    if scratch_root.exists():
        raise FileExistsError(f"single-use probe root exists: {scratch_root}")
    scratch_root.mkdir(parents=True)
    packet_root = Path(__file__).resolve(strict=True).parent
    controller_path = packet_root / "destructive_controller.py"
    module = _load_controller(controller_path)
    module._require_authority_identity()
    module._verify_packet(packet_root)
    authorities = module._load_authorities(packet_root)
    fixture_repo = scratch_root / "db-fixture"
    (fixture_repo / "data").mkdir(parents=True)
    fixture_db = fixture_repo / "data" / "market_data.db"
    _create_fixture(module, repo_root / "data" / "market_data.db", fixture_db)

    with module._connect_ro(fixture_db) as connection:
        module._verify_db(connection, authorities)
        snapshot = module._snapshot_bytes(connection, authorities)
    _assert_not_held(module, fixture_db)
    if module._sha256_bytes(snapshot) != module.EXPECTED_SNAPSHOT_SHA256:
        raise AssertionError("fixture snapshot differs from pinned snapshot")
    deleted = module._delete_rows(fixture_repo, authorities)
    if deleted != {"financial_cache": 19, "fundamentals": 130, "market_sync_meta": 1}:
        raise AssertionError(f"unexpected delete counts: {deleted}")
    records = [json.loads(line) for line in snapshot.decode("utf-8").splitlines()]
    restored = module._restore_rows(fixture_repo, records)
    if restored != {"financial_cache": 19, "fundamentals": 130, "market_sync_meta": 1}:
        raise AssertionError(f"unexpected restore counts: {restored}")
    with module._connect_ro(fixture_db) as connection:
        module._verify_db(connection, authorities)
    _assert_not_held(module, fixture_db)
    _probe_file_transport(module, scratch_root)
    result = {
        "authority_id": module.AUTHORITY_ID,
        "controller_sha256": module._sha256_file(controller_path),
        "db_delete_counts": deleted,
        "db_restore_counts": restored,
        "file_transport": "move_and_restore_pass",
        "snapshot_bytes": len(snapshot),
        "snapshot_records": len(records),
        "snapshot_sha256": module._sha256_bytes(snapshot),
    }
    print(json.dumps(result, sort_keys=True))
    shutil.rmtree(scratch_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    arguments = parser.parse_args()
    _run(arguments.repo_root.resolve(strict=True), arguments.scratch_root)


if __name__ == "__main__":
    main()
