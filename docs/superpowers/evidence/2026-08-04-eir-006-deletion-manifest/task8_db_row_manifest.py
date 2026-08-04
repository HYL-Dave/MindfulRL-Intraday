from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path


OLD_CACHE_KEY = re.compile(r"^metrics_(?P<ticker>[^:\t\r\n]+)_annual_y2$")
CURRENT_SEC_KEY = re.compile(
    r"^fundamentals_analysis:sec_edgar:(?P<ticker>[^:\t\r\n]+):"
    r"(?P<period>annual|quarterly):v1$"
)
EXPECTED_OLD_CACHE_ROWS = 19
EXPECTED_FUNDAMENTALS_ROWS = 130
EXPECTED_SYNC_ROWS = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return _sha256_bytes(data)


def _safe(value: object) -> str:
    text = "" if value is None else str(value)
    if any(separator in text for separator in ("\t", "\n", "\r")):
        raise ValueError("manifest value contains a record separator")
    return text


def _tsv(columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    lines = ["\t".join(columns)]
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("TSV row width differs from header")
        lines.append("\t".join(_safe(value) for value in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _stat(path: Path) -> dict[str, int | str]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "mode": f"{value.st_mode & 0o777:04o}",
    }


def _payload_identity(value: str) -> tuple[int, str]:
    encoded = value.encode("utf-8")
    return len(encoded), _sha256_bytes(encoded)


def _run(repo_root: Path, output_root: Path) -> None:
    repo_root = repo_root.resolve(strict=True)
    database_path = (repo_root / "data" / "market_data.db").resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"single-use output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    source_path = Path(__file__).resolve(strict=True)
    source_identity = {
        "path": str(source_path),
        "lines": len(source_path.read_bytes().splitlines()),
        "bytes": source_path.stat().st_size,
        "sha256": _sha256_file(source_path),
    }
    database_before = _stat(database_path)
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone() != (1,):
            raise RuntimeError("production SQLite connection is not query-only")
        connection.execute("BEGIN")

        cache_rows = list(
            connection.execute(
                """
                SELECT cache_key, source, ticker, data, fetched_at, expires_at
                FROM financial_cache
                ORDER BY cache_key
                """
            )
        )
        old_cache_rows: list[tuple[object, ...]] = []
        cache_classification: list[tuple[object, ...]] = []
        for cache_key, source, ticker, data, fetched_at, expires_at in cache_rows:
            old_match = OLD_CACHE_KEY.fullmatch(str(cache_key))
            current_match = CURRENT_SEC_KEY.fullmatch(str(cache_key))
            if old_match and current_match:
                raise AssertionError(f"cache key belongs to old and current families: {cache_key}")
            if old_match:
                family = "old_metrics_annual_y2"
            elif current_match:
                family = "current_sec_v1"
            else:
                family = "other_retained_cache"
            payload_bytes, payload_sha = _payload_identity(str(data))
            cache_classification.append(
                (cache_key, family, source, ticker, fetched_at, expires_at, payload_bytes, payload_sha)
            )
            if family == "old_metrics_annual_y2":
                if old_match is None or old_match.group("ticker") != ticker:
                    raise ValueError(f"old cache key/ticker mismatch: {cache_key} / {ticker}")
                old_cache_rows.append(
                    (cache_key, source, ticker, fetched_at, expires_at, payload_bytes, payload_sha)
                )

        fundamentals_rows: list[tuple[object, ...]] = []
        for row_id, ticker, snapshot_date, data in connection.execute(
            "SELECT id, ticker, snapshot_date, data FROM fundamentals ORDER BY id"
        ):
            payload_bytes, payload_sha = _payload_identity(str(data))
            fundamentals_rows.append((row_id, ticker, snapshot_date, payload_bytes, payload_sha))

        sync_rows: list[tuple[object, ...]] = []
        all_sync_rows = list(
            connection.execute(
                """
                SELECT domain, last_success, last_error, rows_added, updated_at
                FROM market_sync_meta
                ORDER BY domain
                """
            )
        )
        for domain, last_success, last_error, rows_added, updated_at in all_sync_rows:
            if domain != "fundamentals":
                continue
            error_text = "" if last_error is None else str(last_error)
            error_bytes, error_sha = _payload_identity(error_text)
            sync_rows.append(
                (
                    domain,
                    last_success,
                    rows_added,
                    updated_at,
                    int(last_error is not None),
                    error_bytes,
                    error_sha,
                )
            )

        if len(old_cache_rows) != EXPECTED_OLD_CACHE_ROWS:
            raise AssertionError(f"old cache count changed: {len(old_cache_rows)}")
        if len(fundamentals_rows) != EXPECTED_FUNDAMENTALS_ROWS:
            raise AssertionError(f"legacy fundamentals count changed: {len(fundamentals_rows)}")
        if len(sync_rows) != EXPECTED_SYNC_ROWS:
            raise AssertionError(f"fundamentals sync count changed: {len(sync_rows)}")
        if any(CURRENT_SEC_KEY.fullmatch(str(row[0])) for row in old_cache_rows):
            raise AssertionError("old cache manifest includes a current SEC cache key")

        logical_snapshot = {
            "financial_cache_rows": len(cache_rows),
            "old_cache_rows": len(old_cache_rows),
            "current_sec_v1_rows": sum(row[1] == "current_sec_v1" for row in cache_classification),
            "other_retained_cache_rows": sum(
                row[1] == "other_retained_cache" for row in cache_classification
            ),
            "fundamentals_rows": len(fundamentals_rows),
            "market_sync_meta_rows": len(all_sync_rows),
            "fundamentals_sync_rows": len(sync_rows),
            "data_version": connection.execute("PRAGMA data_version").fetchone()[0],
            "query_only": connection.execute("PRAGMA query_only").fetchone()[0],
        }
        connection.commit()
    finally:
        connection.close()

    artifacts: dict[str, str] = {}
    artifacts["old-cache-rows.tsv"] = _write(
        output_root / "old-cache-rows.tsv",
        _tsv(
            (
                "cache_key",
                "source",
                "ticker",
                "fetched_at",
                "expires_at",
                "payload_bytes",
                "payload_sha256",
            ),
            old_cache_rows,
        ),
    )
    artifacts["cache-classification.tsv"] = _write(
        output_root / "cache-classification.tsv",
        _tsv(
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
            cache_classification,
        ),
    )
    artifacts["legacy-fundamentals-rows.tsv"] = _write(
        output_root / "legacy-fundamentals-rows.tsv",
        _tsv(
            ("id", "ticker", "snapshot_date", "payload_bytes", "payload_sha256"),
            fundamentals_rows,
        ),
    )
    artifacts["legacy-sync-rows.tsv"] = _write(
        output_root / "legacy-sync-rows.tsv",
        _tsv(
            (
                "domain",
                "last_success",
                "rows_added",
                "updated_at",
                "has_error",
                "error_bytes",
                "error_sha256",
            ),
            sync_rows,
        ),
    )

    repo_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "repo_head": repo_head,
        "source_identity": source_identity,
        "database": {
            "path": str(database_path),
            "stat_before": database_before,
            "stat_after": _stat(database_path),
            "logical_snapshot": logical_snapshot,
        },
        "manifests": {
            "old_cache": {
                "path": "old-cache-rows.tsv",
                "rows": len(old_cache_rows),
                "sha256": artifacts["old-cache-rows.tsv"],
                "primary_key": "cache_key",
            },
            "cache_classification": {
                "path": "cache-classification.tsv",
                "rows": len(cache_classification),
                "sha256": artifacts["cache-classification.tsv"],
            },
            "legacy_fundamentals": {
                "path": "legacy-fundamentals-rows.tsv",
                "rows": len(fundamentals_rows),
                "sha256": artifacts["legacy-fundamentals-rows.tsv"],
                "primary_key": "id",
            },
            "legacy_sync": {
                "path": "legacy-sync-rows.tsv",
                "rows": len(sync_rows),
                "sha256": artifacts["legacy-sync-rows.tsv"],
                "primary_key": "domain",
            },
        },
        "delete_parameter_count": len(old_cache_rows) + len(fundamentals_rows) + len(sync_rows),
        "current_cache_keys_in_delete_manifest": 0,
    }
    result_bytes = _json_bytes(result)
    artifacts["result.json"] = _write(output_root / "result.json", result_bytes)
    sum_lines = [f"{digest}  {name}" for name, digest in sorted(artifacts.items())]
    _write(output_root / "SHA256SUMS", ("\n".join(sum_lines) + "\n").encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    _run(arguments.repo_root, arguments.output_root)


if __name__ == "__main__":
    main()
