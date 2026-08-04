from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
PRODUCT_CUTOVER_TIP = "ce88f72d9f9d710903533505371789d18cce953e"
TASK8_BASE = "657b4aa2c8d67a6e659cba4d0d4c6cd90c8d36f3"
AUTHORITY_ID = "6096b988428a94d053baddd18493eb29077bc627d725a95fd53f75c4755b0dce"
APPROVAL_ENV = "ARKSCOPE_EIR006_DESTRUCTIVE_APPROVED"
EXPECTED_REPO_ROOT = Path("/mnt/md0/PycharmProjects/ArkScope")
PACKET_RELATIVE = Path(
    "docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest"
)
QUARANTINE_ROOT = Path(
    "/mnt/md0/PycharmProjects/.arkscope-eir006-quarantine/"
    + AUTHORITY_ID
)
EXPECTED_SNAPSHOT_SHA256 = "1e3578344dfcac0e445900358265c6606150007a496a71284d87e5ae5821697c"
EXPECTED_SNAPSHOT_RECORDS = 150
EXPECTED_MARKET_DB_IDENTITY = (2304, 127284871)
EXPECTED_PROFILE_DB_IDENTITY = (2304, 127284276)
EXPECTED_CENSUS_OWNER = Path("tests/test_eir006_retired_data_boundaries.py")
EXPECTED_CENSUS_OWNER_SHA256 = (
    "de6e192b7e3a233b26d9a43c5dd8608e0ce26cfad3ef79f3d73e882a3f79fb9c"
)
LSOF_TIMEOUT_SECONDS = 15.0

AUTHORITY_INPUT = {
    "alias_input_sha256": "0a8fbbf845b73bab1740d04ffb77ab1e935884f417c2bece20395187f83d9220",
    "behavior_propagation_sha256": "613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba",
    "cache_classification_sha256": "62e56fc02f5b8a15aaea9f360eee8bd875e10d6a10c7a017c8d568652beef323",
    "consumer_census_sha256": "a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64",
    "file_manifest_sha256": "842c3e08ff8ed9cb11c92033cf67ad5950d357cb8cd1e0662b74683ba554b0fc",
    "fundamentals_manifest_sha256": "6b845506f9fce54ac4dba78ebd96bacc20113a7aefef651b877f62892418c219",
    "old_cache_manifest_sha256": "a4a8d829eb08553a1223f5240de260955fc48a564f8232b943206e0bf88b39bd",
    "product_cutover_tip": PRODUCT_CUTOVER_TIP,
    "schema_version": SCHEMA_VERSION,
    "sync_manifest_sha256": "5b3736ba19e66b2e427b149b143771fb5625eab426e8af7a6317c29461cd15ff",
    "task8_base": TASK8_BASE,
}
AUTHORITY_FILES = {
    "behavior-propagation.tsv": AUTHORITY_INPUT["behavior_propagation_sha256"],
    "cache-classification.tsv": AUTHORITY_INPUT["cache_classification_sha256"],
    "consumer-census.tsv": AUTHORITY_INPUT["consumer_census_sha256"],
    "legacy-fundamentals-rows.tsv": AUTHORITY_INPUT["fundamentals_manifest_sha256"],
    "legacy-price-files.tsv": AUTHORITY_INPUT["file_manifest_sha256"],
    "legacy-sync-rows.tsv": AUTHORITY_INPUT["sync_manifest_sha256"],
    "old-cache-rows.tsv": AUTHORITY_INPUT["old_cache_manifest_sha256"],
    "ticker-aliases.tsv": AUTHORITY_INPUT["alias_input_sha256"],
}
EXPECTED_SCHEDULE_SETTINGS = {
    "schedule.finnhub_news.enabled": "true",
    "schedule.finnhub_news.interval_minutes": "300",
    "schedule.ibkr_news.enabled": "true",
    "schedule.ibkr_prices.enabled": "true",
    "schedule.ibkr_prices.interval_minutes": "720",
    "schedule.polygon_news.enabled": "true",
    "schedule.polygon_news.interval_minutes": "360",
}
OLD_CACHE_KEY = re.compile(r"^metrics_(?P<ticker>[^:\t\r\n]+)_annual_y2$")
CURRENT_SEC_KEY = re.compile(
    r"^fundamentals_analysis:sec_edgar:(?P<ticker>[^:\t\r\n]+):"
    r"(?P<period>annual|quarterly):v1$"
)


class Refusal(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _pretty_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_approval(token: str | None) -> None:
    if token != AUTHORITY_ID or os.environ.get(APPROVAL_ENV) != AUTHORITY_ID:
        raise Refusal(
            "destructive or rollback mode requires both the exact approval token "
            f"and {APPROVAL_ENV}={AUTHORITY_ID}"
        )


def _require_authority_identity() -> None:
    observed = _sha256_bytes(_canonical_json(AUTHORITY_INPUT))
    if observed != AUTHORITY_ID:
        raise Refusal(f"authority identity mismatch: {observed}")


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _verify_repo(repo_root: Path) -> None:
    if repo_root.resolve(strict=True) != EXPECTED_REPO_ROOT.resolve(strict=True):
        raise Refusal(f"unexpected repository root: {repo_root}")
    _run_checked(
        ["git", "merge-base", "--is-ancestor", PRODUCT_CUTOVER_TIP, "HEAD"],
        cwd=repo_root,
    )
    protected = (
        "src",
        "data_sources",
        "apps",
        "tests",
        "config",
        "package.json",
        "package-lock.json",
    )
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            f"{PRODUCT_CUTOVER_TIP}..HEAD",
            "--",
            *protected,
            f":(exclude){EXPECTED_CENSUS_OWNER.as_posix()}",
        ],
        cwd=repo_root,
        check=False,
    )
    if completed.returncode != 0:
        raise Refusal("product/test bytes changed after the reviewed cutover tip")
    census_owner = repo_root / EXPECTED_CENSUS_OWNER
    if (
        not census_owner.is_file()
        or census_owner.is_symlink()
        or _sha256_file(census_owner) != EXPECTED_CENSUS_OWNER_SHA256
    ):
        raise Refusal("reviewed Task 8 census owner identity changed")
    status = _run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *protected],
        cwd=repo_root,
    ).stdout
    if status:
        raise Refusal(f"product/test worktree is not clean: {status.splitlines()}")


def _packet_root(repo_root: Path) -> Path:
    root = (repo_root / PACKET_RELATIVE).resolve(strict=True)
    if root.parent != (repo_root / PACKET_RELATIVE.parent).resolve(strict=True):
        raise Refusal("packet path escaped its reviewed parent")
    return root


def _verify_packet(packet_root: Path) -> None:
    for name, expected in AUTHORITY_FILES.items():
        path = packet_root / name
        if not path.is_file() or path.is_symlink():
            raise Refusal(f"authority file missing or symlinked: {name}")
        observed = _sha256_file(path)
        if observed != expected:
            raise Refusal(f"authority file changed: {name}: {observed}")


def _read_tsv(packet_root: Path, name: str) -> list[dict[str, str]]:
    with (packet_root / name).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not reader.fieldnames:
        raise Refusal(f"authority TSV has no header: {name}")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise Refusal(f"authority TSV has malformed row: {name}")
    return rows


def _load_authorities(packet_root: Path) -> dict[str, list[dict[str, str]]]:
    authorities = {
        "files": _read_tsv(packet_root, "legacy-price-files.tsv"),
        "aliases": _read_tsv(packet_root, "ticker-aliases.tsv"),
        "old_cache": _read_tsv(packet_root, "old-cache-rows.tsv"),
        "cache_classification": _read_tsv(packet_root, "cache-classification.tsv"),
        "fundamentals": _read_tsv(packet_root, "legacy-fundamentals-rows.tsv"),
        "sync": _read_tsv(packet_root, "legacy-sync-rows.tsv"),
    }
    expected_counts = {
        "files": 301,
        "aliases": 3,
        "old_cache": 19,
        "cache_classification": 46,
        "fundamentals": 130,
        "sync": 1,
    }
    observed = {name: len(rows) for name, rows in authorities.items()}
    if observed != expected_counts:
        raise Refusal(f"authority row counts changed: {observed}")
    return authorities


def _connect_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve(strict=True)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
        connection.close()
        raise Refusal(f"query_only did not engage for {path}")
    return connection


def _connect_rw(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path.resolve(strict=True)), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _payload_identity(value: str) -> tuple[str, str]:
    encoded = value.encode("utf-8")
    return str(len(encoded)), _sha256_bytes(encoded)


def _classify_cache_key(cache_key: str) -> str:
    if OLD_CACHE_KEY.fullmatch(cache_key):
        return "old_metrics_annual_y2"
    if CURRENT_SEC_KEY.fullmatch(cache_key):
        return "current_sec_v1"
    return "other_retained_cache"


def _cache_classification_rows(connection: sqlite3.Connection) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    query = """
        SELECT cache_key, source, ticker, data, fetched_at, expires_at
        FROM financial_cache
        ORDER BY cache_key
    """
    for row in connection.execute(query):
        payload_bytes, payload_sha = _payload_identity(str(row["data"]))
        rows.append(
            (
                str(row["cache_key"]),
                _classify_cache_key(str(row["cache_key"])),
                str(row["source"]),
                str(row["ticker"]),
                str(row["fetched_at"]),
                str(row["expires_at"]),
                payload_bytes,
                payload_sha,
            )
        )
    return rows


def _expected_tuples(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [tuple(row[column] for column in columns) for row in rows]


def _verify_aliases(connection: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    observed = [
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            "SELECT alias, canonical FROM ticker_aliases ORDER BY alias"
        )
    ]
    expected = _expected_tuples(rows, ("alias", "canonical"))
    if observed != expected:
        raise Refusal("ticker alias input changed")


def _verify_old_cache(connection: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    observed: list[tuple[str, ...]] = []
    for expected in rows:
        row = connection.execute(
            """
            SELECT cache_key, source, ticker, data, fetched_at, expires_at
            FROM financial_cache
            WHERE cache_key = ?
            """,
            (expected["cache_key"],),
        ).fetchone()
        if row is None:
            raise Refusal(f"old cache row disappeared: {expected['cache_key']}")
        payload_bytes, payload_sha = _payload_identity(str(row["data"]))
        observed.append(
            (
                str(row["cache_key"]),
                str(row["source"]),
                str(row["ticker"]),
                str(row["fetched_at"]),
                str(row["expires_at"]),
                payload_bytes,
                payload_sha,
            )
        )
    columns = (
        "cache_key",
        "source",
        "ticker",
        "fetched_at",
        "expires_at",
        "payload_bytes",
        "payload_sha256",
    )
    if observed != _expected_tuples(rows, columns):
        raise Refusal("old cache row metadata or payload changed")


def _verify_fundamentals(
    connection: sqlite3.Connection, rows: list[dict[str, str]]
) -> None:
    observed: list[tuple[str, ...]] = []
    for expected in rows:
        row = connection.execute(
            "SELECT id, ticker, snapshot_date, data FROM fundamentals WHERE id = ?",
            (int(expected["id"]),),
        ).fetchone()
        if row is None:
            raise Refusal(f"legacy fundamentals row disappeared: {expected['id']}")
        payload_bytes, payload_sha = _payload_identity(str(row["data"]))
        observed.append(
            (
                str(row["id"]),
                str(row["ticker"]),
                str(row["snapshot_date"]),
                payload_bytes,
                payload_sha,
            )
        )
    columns = ("id", "ticker", "snapshot_date", "payload_bytes", "payload_sha256")
    if observed != _expected_tuples(rows, columns):
        raise Refusal("legacy fundamentals row metadata or payload changed")
    all_ids = [str(row[0]) for row in connection.execute("SELECT id FROM fundamentals ORDER BY id")]
    if all_ids != [row["id"] for row in rows]:
        raise Refusal("fundamentals table contains rows outside the exact manifest")


def _verify_sync(connection: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    expected = rows[0]
    row = connection.execute(
        """
        SELECT domain, last_success, last_error, rows_added, updated_at
        FROM market_sync_meta
        WHERE domain = ?
        """,
        (expected["domain"],),
    ).fetchone()
    if row is None:
        raise Refusal("legacy fundamentals sync row disappeared")
    error_text = "" if row["last_error"] is None else str(row["last_error"])
    error_bytes, error_sha = _payload_identity(error_text)
    observed = (
        str(row["domain"]),
        "" if row["last_success"] is None else str(row["last_success"]),
        str(row["rows_added"]),
        str(row["updated_at"]),
        str(int(row["last_error"] is not None)),
        error_bytes,
        error_sha,
    )
    columns = (
        "domain",
        "last_success",
        "rows_added",
        "updated_at",
        "has_error",
        "error_bytes",
        "error_sha256",
    )
    if observed != tuple(expected[column] for column in columns):
        raise Refusal("legacy fundamentals sync row changed")


def _verify_db(
    connection: sqlite3.Connection,
    authorities: dict[str, list[dict[str, str]]],
) -> None:
    _verify_aliases(connection, authorities["aliases"])
    _verify_old_cache(connection, authorities["old_cache"])
    _verify_fundamentals(connection, authorities["fundamentals"])
    _verify_sync(connection, authorities["sync"])
    expected_classification = _expected_tuples(
        authorities["cache_classification"],
        (
            "cache_key",
            "family",
            "source",
            "ticker",
            "fetched_at",
            "expires_at",
            "payload_bytes",
            "payload_sha256",
        ),
    )
    if _cache_classification_rows(connection) != expected_classification:
        raise Refusal("financial cache classification or retained rows changed")


def _verify_schedule_settings(repo_root: Path) -> None:
    profile_path = repo_root / "data" / "profile_state.db"
    with _connect_ro(profile_path) as connection:
        observed = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                """
                SELECT key, value
                FROM profile_settings
                WHERE key >= 'schedule.' AND key < 'schedule/'
                ORDER BY key
                """
            )
        }
    if observed != EXPECTED_SCHEDULE_SETTINGS:
        raise Refusal(f"saved scheduler configuration changed: {observed}")


def _verify_database_identity(repo_root: Path) -> None:
    market = (repo_root / "data" / "market_data.db").stat()
    profile = (repo_root / "data" / "profile_state.db").stat()
    if (market.st_dev, market.st_ino) != EXPECTED_MARKET_DB_IDENTITY:
        raise Refusal("market_data.db device/inode identity changed")
    if (profile.st_dev, profile.st_ino) != EXPECTED_PROFILE_DB_IDENTITY:
        raise Refusal("profile_state.db device/inode identity changed")


def _file_identity(path: Path) -> tuple[int, int, str, int, str]:
    value = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise Refusal(f"manifest path is not a regular non-symlink file: {path}")
    return (
        value.st_ino,
        value.st_size,
        f"{value.st_mode & 0o777:04o}",
        value.st_mtime_ns,
        _sha256_file(path),
    )


def _verify_files(
    repo_root: Path,
    rows: list[dict[str, str]],
    *,
    location: str,
) -> None:
    if location not in {"source", "quarantine"}:
        raise ValueError(location)
    expected_paths = [row["relative_path"] for row in rows]
    if len(expected_paths) != len(set(expected_paths)):
        raise Refusal("file manifest contains duplicate paths")
    if location == "source":
        base = repo_root
        discovered = sorted(
            path.relative_to(repo_root).as_posix()
            for path in (repo_root / "data" / "prices").rglob("*")
            if path.is_file()
        )
        if discovered != expected_paths:
            raise Refusal("source price-file set differs from exact manifest")
    else:
        base = QUARANTINE_ROOT / "files"
        discovered = sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
        if discovered != expected_paths:
            raise Refusal("quarantined price-file set differs from exact manifest")
    for row in rows:
        path = base / row["relative_path"]
        observed = _file_identity(path)
        expected = (
            int(row["inode"]),
            int(row["size"]),
            row["mode"],
            int(row["mtime_ns"]),
            row["sha256"],
        )
        if observed != expected:
            raise Refusal(f"file identity changed: {row['relative_path']}")


def _lsof_records(arguments: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            ["lsof", "-w", "-Fpcfan", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise Refusal(f"lsof holder probe exceeded {LSOF_TIMEOUT_SECONDS}s") from error
    records = [line for line in completed.stdout.splitlines() if line]
    if completed.stderr:
        raise Refusal(f"lsof holder probe reported an error: {completed.stderr.strip()}")
    if completed.returncode == 0 and records:
        return records
    if completed.returncode == 1 and not records:
        return []
    raise Refusal(
        "lsof holder probe returned an ambiguous result: "
        f"exit={completed.returncode}, records={len(records)}"
    )


def _assert_quiesced(repo_root: Path) -> None:
    database = repo_root / "data" / "market_data.db"
    profile = repo_root / "data" / "profile_state.db"
    holders = _lsof_records([str(database), str(profile)])
    if holders:
        raise Refusal(f"database holders remain: {holders}")
    price_holders = _lsof_records(["+D", str(repo_root / "data" / "prices")])
    if price_holders:
        raise Refusal(f"legacy price-file holders remain: {price_holders}")
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden = [
        line.strip()
        for line in processes
        if "python -m src.api" in line
        or "node apps/arkscope-desktop/dev.js" in line
        or "market_data_direct.py" in line
        or "daily_update.py" in line
    ]
    if forbidden:
        raise Refusal(f"writer-owner processes remain: {forbidden}")


def _assert_runtime_active() -> None:
    processes = subprocess.run(
        ["ps", "-eo", "args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not any(line.strip() == "python -m src.api" for line in processes):
        raise Refusal("restarted sidecar process is absent")
    if not any(line.strip() == "node apps/arkscope-desktop/dev.js" for line in processes):
        raise Refusal("restarted desktop owner process is absent")


def _snapshot_records(
    connection: sqlite3.Connection,
    authorities: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for expected in authorities["old_cache"]:
        row = connection.execute(
            """
            SELECT cache_key, source, ticker, data, fetched_at, expires_at
            FROM financial_cache WHERE cache_key = ?
            """,
            (expected["cache_key"],),
        ).fetchone()
        if row is None:
            raise Refusal(f"snapshot cache row absent: {expected['cache_key']}")
        records.append(
            {
                "record_type": "financial_cache",
                "values": {key: row[key] for key in row.keys()},
            }
        )
    for expected in authorities["fundamentals"]:
        row = connection.execute(
            "SELECT id, ticker, snapshot_date, data FROM fundamentals WHERE id = ?",
            (int(expected["id"]),),
        ).fetchone()
        if row is None:
            raise Refusal(f"snapshot fundamentals row absent: {expected['id']}")
        records.append(
            {
                "record_type": "fundamentals",
                "values": {key: row[key] for key in row.keys()},
            }
        )
    for expected in authorities["sync"]:
        row = connection.execute(
            """
            SELECT domain, last_success, last_error, rows_added, updated_at
            FROM market_sync_meta WHERE domain = ?
            """,
            (expected["domain"],),
        ).fetchone()
        if row is None:
            raise Refusal(f"snapshot sync row absent: {expected['domain']}")
        records.append(
            {
                "record_type": "market_sync_meta",
                "values": {key: row[key] for key in row.keys()},
            }
        )
    if len(records) != EXPECTED_SNAPSHOT_RECORDS:
        raise Refusal(f"snapshot record count changed: {len(records)}")
    return records


def _snapshot_bytes(
    connection: sqlite3.Connection,
    authorities: dict[str, list[dict[str, str]]],
) -> bytes:
    return b"".join(_canonical_json(record) for record in _snapshot_records(connection, authorities))


def _write_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise Refusal(f"stale receipt temporary exists: {temporary}")
    _write_exclusive(temporary, _pretty_json(value))
    os.replace(temporary, path)


def _preview_snapshot(repo_root: Path, output: Path) -> None:
    _require_authority_identity()
    _verify_repo(repo_root)
    _verify_database_identity(repo_root)
    packet_root = _packet_root(repo_root)
    _verify_packet(packet_root)
    authorities = _load_authorities(packet_root)
    with _connect_ro(repo_root / "data" / "market_data.db") as connection:
        connection.execute("BEGIN")
        _verify_db(connection, authorities)
        data = _snapshot_bytes(connection, authorities)
        connection.commit()
    if output.exists():
        raise Refusal(f"preview output already exists: {output}")
    _write_exclusive(output, data)
    print(
        json.dumps(
            {
                "bytes": len(data),
                "records": EXPECTED_SNAPSHOT_RECORDS,
                "sha256": _sha256_bytes(data),
            },
            sort_keys=True,
        )
    )


def _preflight(repo_root: Path, *, require_quarantine_absent: bool) -> dict[str, list[dict[str, str]]]:
    _require_authority_identity()
    _verify_repo(repo_root)
    _verify_database_identity(repo_root)
    packet_root = _packet_root(repo_root)
    _verify_packet(packet_root)
    authorities = _load_authorities(packet_root)
    _assert_quiesced(repo_root)
    _verify_schedule_settings(repo_root)
    _verify_files(repo_root, authorities["files"], location="source")
    with _connect_ro(repo_root / "data" / "market_data.db") as connection:
        connection.execute("BEGIN")
        _verify_db(connection, authorities)
        snapshot_sha = _sha256_bytes(_snapshot_bytes(connection, authorities))
        connection.commit()
    if snapshot_sha != EXPECTED_SNAPSHOT_SHA256:
        raise Refusal(f"rollback snapshot identity changed: {snapshot_sha}")
    if require_quarantine_absent and QUARANTINE_ROOT.exists():
        raise Refusal(f"quarantine root already exists: {QUARANTINE_ROOT}")
    return authorities


def _remove_source_directories(repo_root: Path) -> None:
    for relative in ("data/prices/15min", "data/prices/hourly", "data/prices"):
        (repo_root / relative).rmdir()


def _restore_source_directories(repo_root: Path) -> None:
    (repo_root / "data" / "prices" / "15min").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "prices" / "hourly").mkdir(parents=True, exist_ok=True)


def _move_to_quarantine(
    repo_root: Path,
    rows: list[dict[str, str]],
    moved: list[str],
) -> None:
    destination_root = QUARANTINE_ROOT / "files"
    for row in rows:
        relative = row["relative_path"]
        source = repo_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise Refusal(f"quarantine destination exists: {destination}")
        os.replace(source, destination)
        moved.append(relative)
    _remove_source_directories(repo_root)


def _move_back(repo_root: Path, moved: Iterable[str]) -> None:
    _restore_source_directories(repo_root)
    for relative in reversed(list(moved)):
        source = QUARANTINE_ROOT / "files" / relative
        destination = repo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise Refusal(f"rollback destination exists: {destination}")
        os.replace(source, destination)


def _delete_rows(
    repo_root: Path,
    authorities: dict[str, list[dict[str, str]]],
) -> dict[str, int]:
    database_path = repo_root / "data" / "market_data.db"
    connection = _connect_rw(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _verify_db(connection, authorities)
        other_sync_before = list(
            connection.execute(
                """
                SELECT domain, last_success, last_error, rows_added, updated_at
                FROM market_sync_meta WHERE domain <> ? ORDER BY domain
                """,
                (authorities["sync"][0]["domain"],),
            )
        )

        cache_keys = [row["cache_key"] for row in authorities["old_cache"]]
        fundamentals_ids = [int(row["id"]) for row in authorities["fundamentals"]]
        sync_domains = [row["domain"] for row in authorities["sync"]]
        cache_cursor = connection.execute(
            f"DELETE FROM financial_cache WHERE cache_key IN ({','.join('?' for _ in cache_keys)})",
            cache_keys,
        )
        fundamentals_cursor = connection.execute(
            f"DELETE FROM fundamentals WHERE id IN ({','.join('?' for _ in fundamentals_ids)})",
            fundamentals_ids,
        )
        sync_cursor = connection.execute(
            f"DELETE FROM market_sync_meta WHERE domain IN ({','.join('?' for _ in sync_domains)})",
            sync_domains,
        )
        deleted = {
            "financial_cache": cache_cursor.rowcount,
            "fundamentals": fundamentals_cursor.rowcount,
            "market_sync_meta": sync_cursor.rowcount,
        }
        if deleted != {
            "financial_cache": 19,
            "fundamentals": 130,
            "market_sync_meta": 1,
        }:
            raise Refusal(f"delete affected unexpected rows: {deleted}")
        if connection.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0] != 0:
            raise Refusal("legacy fundamentals rows remain after delete")
        for cache_key in cache_keys:
            if connection.execute(
                "SELECT 1 FROM financial_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone():
                raise Refusal(f"old cache key remains after delete: {cache_key}")
        for domain in sync_domains:
            if connection.execute(
                "SELECT 1 FROM market_sync_meta WHERE domain = ?", (domain,)
            ).fetchone():
                raise Refusal(f"legacy sync row remains after delete: {domain}")
        retained_expected = [
            row
            for row in authorities["cache_classification"]
            if row["family"] != "old_metrics_annual_y2"
        ]
        retained_observed = _cache_classification_rows(connection)
        retained_columns = (
            "cache_key",
            "family",
            "source",
            "ticker",
            "fetched_at",
            "expires_at",
            "payload_bytes",
            "payload_sha256",
        )
        if retained_observed != _expected_tuples(retained_expected, retained_columns):
            raise Refusal("retained cache rows changed during delete")
        other_sync_after = list(
            connection.execute(
                """
                SELECT domain, last_success, last_error, rows_added, updated_at
                FROM market_sync_meta WHERE domain <> ? ORDER BY domain
                """,
                (authorities["sync"][0]["domain"],),
            )
        )
        if [tuple(row) for row in other_sync_after] != [tuple(row) for row in other_sync_before]:
            raise Refusal("unrelated sync rows changed during delete")
        connection.commit()
        return deleted
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _create_quarantine_root(repo_root: Path) -> None:
    parent = QUARANTINE_ROOT.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink():
        raise Refusal("quarantine parent may not be a symlink")
    if parent.stat().st_dev != repo_root.stat().st_dev:
        raise Refusal("quarantine parent is not on the repository filesystem")
    QUARANTINE_ROOT.mkdir(mode=0o700)
    (QUARANTINE_ROOT / "db").mkdir(mode=0o700)
    (QUARANTINE_ROOT / "files").mkdir(mode=0o700)


def _execute(repo_root: Path, token: str | None) -> None:
    _require_approval(token)
    authorities = _preflight(repo_root, require_quarantine_absent=True)
    _create_quarantine_root(repo_root)
    moved: list[str] = []
    db_committed = False
    snapshot_path = QUARANTINE_ROOT / "db" / "legacy-rows.jsonl"
    try:
        with _connect_ro(repo_root / "data" / "market_data.db") as connection:
            connection.execute("BEGIN")
            _verify_db(connection, authorities)
            snapshot = _snapshot_bytes(connection, authorities)
            connection.commit()
        if _sha256_bytes(snapshot) != EXPECTED_SNAPSHOT_SHA256:
            raise Refusal("snapshot bytes differ immediately before execution")
        _write_exclusive(snapshot_path, snapshot)
        _move_to_quarantine(repo_root, authorities["files"], moved)
        _assert_quiesced(repo_root)
        deleted = _delete_rows(repo_root, authorities)
        db_committed = True
        receipt = {
            "authority_id": AUTHORITY_ID,
            "completed_at": _utc_now(),
            "deleted_rows": deleted,
            "moved_files": len(moved),
            "snapshot_bytes": len(snapshot),
            "snapshot_records": EXPECTED_SNAPSHOT_RECORDS,
            "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
            "status": "complete",
        }
        _atomic_json(QUARANTINE_ROOT / "execution.json", receipt)
    except BaseException as error:
        restore_error: str | None = None
        if moved and not db_committed:
            try:
                _move_back(repo_root, moved)
            except BaseException as restore_failure:
                restore_error = f"{type(restore_failure).__name__}: {restore_failure}"
        elif db_committed:
            restore_error = "not attempted after committed DB delete; run verify or reviewed rollback"
        failure = {
            "authority_id": AUTHORITY_ID,
            "failed_at": _utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "moved_before_failure": len(moved),
            "restore_error": restore_error,
            "status": "failed",
        }
        _atomic_json(QUARANTINE_ROOT / "failure.json", failure)
        raise


def _read_snapshot(path: Path) -> list[dict[str, object]]:
    data = path.read_bytes()
    if _sha256_bytes(data) != EXPECTED_SNAPSHOT_SHA256:
        raise Refusal("rollback snapshot SHA changed")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    if len(records) != EXPECTED_SNAPSHOT_RECORDS:
        raise Refusal("rollback snapshot record count changed")
    return records


def _restore_rows(repo_root: Path, records: list[dict[str, object]]) -> dict[str, int]:
    database_path = repo_root / "data" / "market_data.db"
    connection = _connect_rw(database_path)
    restored = {"financial_cache": 0, "fundamentals": 0, "market_sync_meta": 0}
    statements = {
        "financial_cache": (
            "cache_key",
            "INSERT INTO financial_cache "
            "(cache_key, source, ticker, data, fetched_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("cache_key", "source", "ticker", "data", "fetched_at", "expires_at"),
        ),
        "fundamentals": (
            "id",
            "INSERT INTO fundamentals (id, ticker, snapshot_date, data) VALUES (?, ?, ?, ?)",
            ("id", "ticker", "snapshot_date", "data"),
        ),
        "market_sync_meta": (
            "domain",
            "INSERT INTO market_sync_meta "
            "(domain, last_success, last_error, rows_added, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("domain", "last_success", "last_error", "rows_added", "updated_at"),
        ),
    }
    try:
        connection.execute("BEGIN IMMEDIATE")
        for record in records:
            record_type = str(record["record_type"])
            values = record["values"]
            if record_type not in statements or not isinstance(values, dict):
                raise Refusal(f"unsupported snapshot record: {record_type}")
            primary_key, insert_sql, columns = statements[record_type]
            table = record_type
            current = connection.execute(
                f"SELECT {', '.join(columns)} FROM {table} WHERE {primary_key} = ?",
                (values[primary_key],),
            ).fetchone()
            expected_tuple = tuple(values[column] for column in columns)
            if current is not None:
                if tuple(current) != expected_tuple:
                    raise Refusal(f"rollback target conflicts: {record_type} {values[primary_key]}")
                continue
            connection.execute(insert_sql, expected_tuple)
            restored[record_type] += 1
        connection.commit()
        return restored
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _rollback(repo_root: Path, token: str | None) -> None:
    _require_approval(token)
    _require_authority_identity()
    _verify_repo(repo_root)
    _verify_database_identity(repo_root)
    _assert_quiesced(repo_root)
    packet_root = _packet_root(repo_root)
    _verify_packet(packet_root)
    authorities = _load_authorities(packet_root)
    snapshot_path = QUARANTINE_ROOT / "db" / "legacy-rows.jsonl"
    records = _read_snapshot(snapshot_path)
    restored = _restore_rows(repo_root, records)
    if (QUARANTINE_ROOT / "files").exists():
        quarantined = [
            row["relative_path"]
            for row in authorities["files"]
            if (QUARANTINE_ROOT / "files" / row["relative_path"]).exists()
        ]
        if quarantined:
            _move_back(repo_root, quarantined)
    _verify_files(repo_root, authorities["files"], location="source")
    with _connect_ro(repo_root / "data" / "market_data.db") as connection:
        connection.execute("BEGIN")
        _verify_db(connection, authorities)
        connection.commit()
    _atomic_json(
        QUARANTINE_ROOT / "rollback.json",
        {
            "authority_id": AUTHORITY_ID,
            "completed_at": _utc_now(),
            "restored_rows": restored,
            "status": "rolled_back",
        },
    )


def _verify_deleted(
    repo_root: Path,
    token: str | None,
    *,
    require_quiesced: bool,
) -> None:
    _require_approval(token)
    _require_authority_identity()
    _verify_repo(repo_root)
    _verify_database_identity(repo_root)
    if require_quiesced:
        _assert_quiesced(repo_root)
    else:
        _assert_runtime_active()
    _verify_schedule_settings(repo_root)
    packet_root = _packet_root(repo_root)
    _verify_packet(packet_root)
    authorities = _load_authorities(packet_root)
    if (repo_root / "data" / "prices").exists():
        raise Refusal("source legacy price root still exists")
    _verify_files(repo_root, authorities["files"], location="quarantine")
    _read_snapshot(QUARANTINE_ROOT / "db" / "legacy-rows.jsonl")
    with _connect_ro(repo_root / "data" / "market_data.db") as connection:
        if connection.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0] != 0:
            raise Refusal("legacy fundamentals rows remain")
        for row in authorities["old_cache"]:
            if connection.execute(
                "SELECT 1 FROM financial_cache WHERE cache_key = ?", (row["cache_key"],)
            ).fetchone():
                raise Refusal(f"old cache key remains: {row['cache_key']}")
        for row in authorities["sync"]:
            if connection.execute(
                "SELECT 1 FROM market_sync_meta WHERE domain = ?", (row["domain"],)
            ).fetchone():
                raise Refusal(f"legacy sync row remains: {row['domain']}")
        _verify_aliases(connection, authorities["aliases"])
        retained_expected = [
            row
            for row in authorities["cache_classification"]
            if row["family"] != "old_metrics_annual_y2"
        ]
        columns = (
            "cache_key",
            "family",
            "source",
            "ticker",
            "fetched_at",
            "expires_at",
            "payload_bytes",
            "payload_sha256",
        )
        if _cache_classification_rows(connection) != _expected_tuples(retained_expected, columns):
            raise Refusal("retained cache rows differ after delete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "preview-snapshot",
            "preflight",
            "execute",
            "verify",
            "post-restart",
            "rollback",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=EXPECTED_REPO_ROOT)
    parser.add_argument("--approval-token")
    parser.add_argument("--snapshot-output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.mode == "preview-snapshot":
            if arguments.snapshot_output is None:
                raise Refusal("preview-snapshot requires --snapshot-output")
            _preview_snapshot(arguments.repo_root, arguments.snapshot_output)
        elif arguments.mode == "preflight":
            _require_approval(arguments.approval_token)
            _preflight(arguments.repo_root, require_quarantine_absent=True)
            print(json.dumps({"authority_id": AUTHORITY_ID, "status": "preflight_pass"}))
        elif arguments.mode == "execute":
            _execute(arguments.repo_root, arguments.approval_token)
        elif arguments.mode == "verify":
            _verify_deleted(
                arguments.repo_root,
                arguments.approval_token,
                require_quiesced=True,
            )
            print(json.dumps({"authority_id": AUTHORITY_ID, "status": "verified_deleted"}))
        elif arguments.mode == "post-restart":
            _verify_deleted(
                arguments.repo_root,
                arguments.approval_token,
                require_quiesced=False,
            )
            print(json.dumps({"authority_id": AUTHORITY_ID, "status": "runtime_restored"}))
        else:
            _rollback(arguments.repo_root, arguments.approval_token)
    except Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(74) from error


if __name__ == "__main__":
    main()
