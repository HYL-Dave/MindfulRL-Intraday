from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CSV_HEADER = ("datetime", "open", "high", "low", "close", "volume", "ticker")
EXPECTED_15MIN_FILES = 225
EXPECTED_HOURLY_FILES = 75
EXPECTED_SUMMARY_FILES = 1
EXPECTED_TOTAL_FILES = 301
EXPECTED_COUNTS = {
    "physical_rows": 2_547_747,
    "raw_unique_keys": 2_314_293,
    "raw_duplicate_rows": 233_454,
    "raw_conflicting_duplicate_keys": 58,
    "raw_apparent_db_value_diffs": 161,
    "canonical_unique_keys": 2_298_763,
    "canonical_duplicate_rows": 248_984,
    "canonical_conflicting_duplicate_keys": 176,
    "canonical_db_value_diffs": 43,
    "canonical_volume_only_diffs": 23,
    "canonical_ohlc_diffs": 20,
    "canonical_keys_absent_from_db": 0,
    "lc_keys_overlapped_by_hapn": 15_530,
    "lc_hapn_alias_conflicts": 118,
}

_15MIN_NAME = re.compile(r"^(?P<ticker>.+)_15min_\d{4}(?:_\d{4})?\.csv$")
_HOURLY_NAME = re.compile(r"^(?P<ticker>.+)_hourly_\d{4}\.csv$")

Value = tuple[float | None, float | None, float | None, float | None, int | None]
Variants = dict[str, set[Value]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return _sha256_bytes(data)


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_mode)


def _stat_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": f"{stat.st_mode & 0o777:04o}",
    }


def _normalize_timestamp(raw: str) -> str:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    elif re.search(r"[+-]\d{4}$", value):
        value = value[:-2] + ":" + value[-2:]
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp is not absolute: {raw!r}")
    utc = parsed.astimezone(timezone.utc)
    if utc.microsecond:
        raise ValueError(f"timestamp has unsupported fractional seconds: {raw!r}")
    return utc.strftime("%Y-%m-%dT%H:%M:%S+0000")


def _float_or_none(raw: str) -> float | None:
    value = raw.strip()
    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric value: {raw!r}")
    return parsed


def _int_or_none(raw: str) -> int | None:
    value = raw.strip()
    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"non-integral volume: {raw!r}")
    return int(parsed)


def _csv_value(row: dict[str, str]) -> Value:
    return (
        _float_or_none(row["open"]),
        _float_or_none(row["high"]),
        _float_or_none(row["low"]),
        _float_or_none(row["close"]),
        _int_or_none(row["volume"]),
    )


def _db_value(row: tuple[object, ...]) -> Value:
    open_, high, low, close, volume = row
    return (
        None if open_ is None else float(open_),
        None if high is None else float(high),
        None if low is None else float(low),
        None if close is None else float(close),
        None if volume is None else int(volume),
    )


def _resolve_alias(ticker: str, aliases: dict[str, str]) -> str:
    current = ticker
    seen: set[str] = set()
    while current in aliases:
        if current in seen:
            raise ValueError(f"ticker alias cycle at {ticker!r}")
        seen.add(current)
        current = aliases[current]
    return current


def _discover_files(prices_root: Path) -> tuple[list[Path], list[Path], Path]:
    fifteen_root = prices_root / "15min"
    hourly_root = prices_root / "hourly"
    summary = prices_root / "collection_summary.json"
    if not fifteen_root.is_dir() or not hourly_root.is_dir() or not summary.is_file():
        raise ValueError("reviewed price roots are missing")

    fifteen = sorted(
        (path for path in fifteen_root.iterdir() if path.is_file()),
        key=lambda path: path.name.encode("utf-8"),
    )
    hourly = sorted(
        (path for path in hourly_root.iterdir() if path.is_file()),
        key=lambda path: path.name.encode("utf-8"),
    )
    all_files = sorted(
        (path for path in prices_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(prices_root).as_posix().encode("utf-8"),
    )
    expected = {*fifteen, *hourly, summary}
    if set(all_files) != expected:
        extras = sorted(
            path.relative_to(prices_root).as_posix()
            for path in set(all_files) - expected
        )
        raise ValueError(f"unreviewed price files: {extras}")
    if (len(fifteen), len(hourly), len(all_files)) != (
        EXPECTED_15MIN_FILES,
        EXPECTED_HOURLY_FILES,
        EXPECTED_TOTAL_FILES,
    ):
        raise ValueError(
            f"unexpected file counts: 15min={len(fifteen)} "
            f"hourly={len(hourly)} total={len(all_files)}"
        )
    return fifteen, hourly, summary


def _ticker_from_name(path: Path, family: str) -> str:
    matcher = _15MIN_NAME if family == "15min" else _HOURLY_NAME
    match = matcher.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected {family} filename: {path.name}")
    ticker = match.group("ticker")
    if any(separator in ticker for separator in ("\t", "\n", "\r")):
        raise ValueError(f"unsafe ticker in filename: {path.name}")
    return ticker


def _read_csv_file(
    path: Path,
    *,
    repo_root: Path,
    family: str,
    expected_ticker: str,
    retain_values: bool,
) -> tuple[dict[str, object], Variants]:
    before = _stat_identity(path)
    digest = _sha256_file(path)
    variants: Variants = defaultdict(set)
    row_count = 0
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_HEADER:
            raise ValueError(f"unexpected CSV header: {path}")
        for row in reader:
            row_count += 1
            if row["ticker"] != expected_ticker:
                raise ValueError(
                    f"ticker mismatch at {path}:{row_count + 1}: "
                    f"{row['ticker']!r} != {expected_ticker!r}"
                )
            timestamp = _normalize_timestamp(row["datetime"])
            min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)
            if retain_values:
                variants[timestamp].add(_csv_value(row))
    after = _stat_identity(path)
    if before != after:
        raise RuntimeError(f"file changed during read: {path}")
    stat = path.stat()
    record: dict[str, object] = {
        "relative_path": path.relative_to(repo_root).as_posix(),
        "family": family,
        "raw_ticker": expected_ticker,
        "size": stat.st_size,
        "mode": f"{stat.st_mode & 0o777:04o}",
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "row_count": row_count,
        "min_timestamp_utc": min_timestamp or "",
        "max_timestamp_utc": max_timestamp or "",
    }
    return record, dict(variants)


def _read_summary(path: Path, repo_root: Path) -> dict[str, object]:
    before = _stat_identity(path)
    digest = _sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("15min_data", {}).get("total_bars") != 0:
        raise ValueError("legacy collection summary no longer reports zero bars")
    after = _stat_identity(path)
    if before != after:
        raise RuntimeError(f"file changed during read: {path}")
    stat = path.stat()
    return {
        "relative_path": path.relative_to(repo_root).as_posix(),
        "family": "summary",
        "raw_ticker": "-",
        "size": stat.st_size,
        "mode": f"{stat.st_mode & 0o777:04o}",
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
        "row_count": 0,
        "min_timestamp_utc": "-",
        "max_timestamp_utc": "-",
    }


def _db_rows_for_ticker(connection: sqlite3.Connection, ticker: str) -> dict[str, Value]:
    rows: dict[str, Value] = {}
    query = """
        SELECT datetime, open, high, low, close, volume
        FROM prices
        WHERE ticker = ? AND interval = '15min'
        ORDER BY datetime
    """
    for timestamp, open_, high, low, close, volume in connection.execute(query, (ticker,)):
        normalized = _normalize_timestamp(str(timestamp))
        if normalized in rows:
            raise ValueError(f"duplicate SQLite 15min key for {ticker} {normalized}")
        rows[normalized] = _db_value((open_, high, low, close, volume))
    return rows


def _difference_kind(variants: set[Value], database_value: Value | None) -> str | None:
    if database_value is None:
        return "missing_db"
    if database_value in variants:
        return None
    if any(value[:4] == database_value[:4] for value in variants):
        return "volume_only"
    return "ohlc"


def _variant_stats(variants: Variants, physical_rows: int) -> dict[str, int]:
    unique = len(variants)
    return {
        "physical_rows": physical_rows,
        "unique_keys": unique,
        "duplicate_rows": physical_rows - unique,
        "conflicting_duplicate_keys": sum(len(values) > 1 for values in variants.values()),
    }


def _merge_variants(target: Variants, source: Variants) -> None:
    for timestamp, values in source.items():
        target.setdefault(timestamp, set()).update(values)


def _manifest_tsv(records: Iterable[dict[str, object]]) -> bytes:
    columns = (
        "relative_path",
        "family",
        "raw_ticker",
        "size",
        "mode",
        "inode",
        "mtime_ns",
        "sha256",
        "row_count",
        "min_timestamp_utc",
        "max_timestamp_utc",
    )
    lines = ["\t".join(columns)]
    for record in records:
        values = [str(record[column]) for column in columns]
        if any(any(separator in value for separator in ("\t", "\n", "\r")) for value in values):
            raise ValueError("manifest field contains a record separator")
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _diff_tsv(rows: list[tuple[str, ...]], columns: tuple[str, ...]) -> bytes:
    lines = ["\t".join(columns)]
    for row in sorted(rows, key=lambda item: tuple(value.encode("utf-8") for value in item)):
        if any(any(separator in value for separator in ("\t", "\n", "\r")) for value in row):
            raise ValueError("difference field contains a record separator")
        lines.append("\t".join(row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _run(repo_root: Path, output_root: Path) -> None:
    repo_root = repo_root.resolve(strict=True)
    prices_root = (repo_root / "data" / "prices").resolve(strict=True)
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
    fifteen_files, hourly_files, summary_file = _discover_files(prices_root)

    database_before = _stat_record(database_path)
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone() != (1,):
            raise RuntimeError("production SQLite connection is not query-only")
        connection.execute("BEGIN")
        alias_rows = [
            (str(alias), str(canonical))
            for alias, canonical in connection.execute(
                "SELECT alias, canonical FROM ticker_aliases ORDER BY alias"
            )
        ]
        aliases = dict(alias_rows)
        alias_lines = ["alias\tcanonical", *(f"{alias}\t{canonical}" for alias, canonical in alias_rows)]
        alias_bytes = ("\n".join(alias_lines) + "\n").encode("utf-8")
        alias_sha = _write_bytes(output_root / "ticker-aliases.tsv", alias_bytes)

        groups: dict[str, list[Path]] = defaultdict(list)
        for path in fifteen_files:
            groups[_ticker_from_name(path, "15min")].append(path)
        canonical_members: dict[str, list[str]] = defaultdict(list)
        for raw_ticker in sorted(groups, key=lambda value: value.encode("utf-8")):
            canonical_members[_resolve_alias(raw_ticker, aliases)].append(raw_ticker)

        file_records: list[dict[str, object]] = []
        raw_summary = {
            "physical_rows": 0,
            "unique_keys": 0,
            "duplicate_rows": 0,
            "conflicting_duplicate_keys": 0,
            "apparent_db_value_diffs": 0,
            "keys_absent_from_db": 0,
        }
        canonical_summary = {
            "physical_rows": 0,
            "unique_keys": 0,
            "duplicate_rows": 0,
            "conflicting_duplicate_keys": 0,
            "db_value_diffs": 0,
            "volume_only_diffs": 0,
            "ohlc_diffs": 0,
            "keys_absent_from_db": 0,
        }
        raw_diff_rows: list[tuple[str, ...]] = []
        canonical_diff_rows: list[tuple[str, ...]] = []
        multi_member_variants: dict[str, dict[str, Variants]] = defaultdict(dict)
        multi_member_physical: dict[str, dict[str, int]] = defaultdict(dict)

        for raw_ticker in sorted(groups, key=lambda value: value.encode("utf-8")):
            raw_variants: Variants = {}
            physical_rows = 0
            for path in groups[raw_ticker]:
                record, file_variants = _read_csv_file(
                    path,
                    repo_root=repo_root,
                    family="15min",
                    expected_ticker=raw_ticker,
                    retain_values=True,
                )
                file_records.append(record)
                physical_rows += int(record["row_count"])
                _merge_variants(raw_variants, file_variants)

            canonical_ticker = _resolve_alias(raw_ticker, aliases)
            database_rows = _db_rows_for_ticker(connection, canonical_ticker)
            stats = _variant_stats(raw_variants, physical_rows)
            for key in ("physical_rows", "unique_keys", "duplicate_rows", "conflicting_duplicate_keys"):
                raw_summary[key] += stats[key]
            for timestamp, values in raw_variants.items():
                kind = _difference_kind(values, database_rows.get(timestamp))
                if kind is None:
                    continue
                raw_summary["apparent_db_value_diffs"] += 1
                if kind == "missing_db":
                    raw_summary["keys_absent_from_db"] += 1
                raw_diff_rows.append((raw_ticker, canonical_ticker, timestamp, kind))

            if len(canonical_members[canonical_ticker]) == 1:
                for key in ("physical_rows", "unique_keys", "duplicate_rows", "conflicting_duplicate_keys"):
                    canonical_summary[key] += stats[key]
                for timestamp, values in raw_variants.items():
                    kind = _difference_kind(values, database_rows.get(timestamp))
                    if kind is None:
                        continue
                    canonical_summary["db_value_diffs"] += 1
                    if kind == "missing_db":
                        canonical_summary["keys_absent_from_db"] += 1
                    elif kind == "volume_only":
                        canonical_summary["volume_only_diffs"] += 1
                    else:
                        canonical_summary["ohlc_diffs"] += 1
                    canonical_diff_rows.append((canonical_ticker, timestamp, raw_ticker, kind))
            else:
                multi_member_variants[canonical_ticker][raw_ticker] = raw_variants
                multi_member_physical[canonical_ticker][raw_ticker] = physical_rows

        alias_overlap: dict[str, int] = {}
        alias_conflicts: dict[str, int] = {}
        for canonical_ticker in sorted(multi_member_variants, key=lambda value: value.encode("utf-8")):
            member_maps = multi_member_variants[canonical_ticker]
            merged: Variants = {}
            for raw_ticker in sorted(member_maps, key=lambda value: value.encode("utf-8")):
                _merge_variants(merged, member_maps[raw_ticker])
            physical_rows = sum(multi_member_physical[canonical_ticker].values())
            stats = _variant_stats(merged, physical_rows)
            for key in ("physical_rows", "unique_keys", "duplicate_rows", "conflicting_duplicate_keys"):
                canonical_summary[key] += stats[key]
            database_rows = _db_rows_for_ticker(connection, canonical_ticker)
            for timestamp, values in merged.items():
                kind = _difference_kind(values, database_rows.get(timestamp))
                if kind is None:
                    continue
                canonical_summary["db_value_diffs"] += 1
                if kind == "missing_db":
                    canonical_summary["keys_absent_from_db"] += 1
                elif kind == "volume_only":
                    canonical_summary["volume_only_diffs"] += 1
                else:
                    canonical_summary["ohlc_diffs"] += 1
                owners = ",".join(
                    raw_ticker
                    for raw_ticker in sorted(member_maps, key=lambda value: value.encode("utf-8"))
                    if timestamp in member_maps[raw_ticker]
                )
                canonical_diff_rows.append((canonical_ticker, timestamp, owners, kind))

            members = sorted(member_maps, key=lambda value: value.encode("utf-8"))
            for left_index, left in enumerate(members):
                for right in members[left_index + 1 :]:
                    label = f"{left}->{right}"
                    overlap = set(member_maps[left]) & set(member_maps[right])
                    conflicts = sum(
                        len(member_maps[left][timestamp] | member_maps[right][timestamp]) > 1
                        for timestamp in overlap
                    )
                    alias_overlap[label] = len(overlap)
                    alias_conflicts[label] = conflicts

        for path in hourly_files:
            ticker = _ticker_from_name(path, "hourly")
            record, discarded = _read_csv_file(
                path,
                repo_root=repo_root,
                family="hourly",
                expected_ticker=ticker,
                retain_values=False,
            )
            if discarded:
                raise AssertionError("hourly values were unexpectedly retained")
            file_records.append(record)
        file_records.append(_read_summary(summary_file, repo_root))
        file_records.sort(key=lambda record: str(record["relative_path"]).encode("utf-8"))

        manifest_bytes = _manifest_tsv(file_records)
        manifest_sha = _write_bytes(output_root / "legacy-price-files.tsv", manifest_bytes)
        raw_diffs_sha = _write_bytes(
            output_root / "raw-db-differences.tsv",
            _diff_tsv(
                raw_diff_rows,
                ("raw_ticker", "canonical_ticker", "datetime_utc", "difference_kind"),
            ),
        )
        canonical_diffs_sha = _write_bytes(
            output_root / "canonical-db-differences.tsv",
            _diff_tsv(
                canonical_diff_rows,
                ("canonical_ticker", "datetime_utc", "raw_tickers", "difference_kind"),
            ),
        )

        file_counts = {
            "15min_files": sum(record["family"] == "15min" for record in file_records),
            "hourly_files": sum(record["family"] == "hourly" for record in file_records),
            "summary_files": sum(record["family"] == "summary" for record in file_records),
            "total_files": len(file_records),
            "15min_rows": sum(
                int(record["row_count"]) for record in file_records if record["family"] == "15min"
            ),
            "hourly_rows": sum(
                int(record["row_count"]) for record in file_records if record["family"] == "hourly"
            ),
        }
        comparison = {
            "raw": raw_summary,
            "canonical": canonical_summary,
            "alias_overlap": alias_overlap,
            "alias_conflicts": alias_conflicts,
        }
        observed_counts = {
            "physical_rows": raw_summary["physical_rows"],
            "raw_unique_keys": raw_summary["unique_keys"],
            "raw_duplicate_rows": raw_summary["duplicate_rows"],
            "raw_conflicting_duplicate_keys": raw_summary["conflicting_duplicate_keys"],
            "raw_apparent_db_value_diffs": raw_summary["apparent_db_value_diffs"],
            "canonical_unique_keys": canonical_summary["unique_keys"],
            "canonical_duplicate_rows": canonical_summary["duplicate_rows"],
            "canonical_conflicting_duplicate_keys": canonical_summary["conflicting_duplicate_keys"],
            "canonical_db_value_diffs": canonical_summary["db_value_diffs"],
            "canonical_volume_only_diffs": canonical_summary["volume_only_diffs"],
            "canonical_ohlc_diffs": canonical_summary["ohlc_diffs"],
            "canonical_keys_absent_from_db": canonical_summary["keys_absent_from_db"],
            "lc_keys_overlapped_by_hapn": alias_overlap.get("HAPN->LC", alias_overlap.get("LC->HAPN", -1)),
            "lc_hapn_alias_conflicts": alias_conflicts.get("HAPN->LC", alias_conflicts.get("LC->HAPN", -1)),
        }
        if observed_counts != EXPECTED_COUNTS:
            raise AssertionError(
                "decision-relevant comparison changed:\n"
                + json.dumps({"expected": EXPECTED_COUNTS, "observed": observed_counts}, indent=2, sort_keys=True)
            )
        if file_counts != {
            "15min_files": 225,
            "hourly_files": 75,
            "summary_files": 1,
            "total_files": 301,
            "15min_rows": 2_547_747,
            "hourly_rows": 129_575,
        }:
            raise AssertionError(f"legacy file census changed: {file_counts}")

        logical_database_snapshot = {
            "prices_15min_rows": connection.execute(
                "SELECT COUNT(*) FROM prices WHERE interval = '15min'"
            ).fetchone()[0],
            "prices_15min_min_datetime": connection.execute(
                "SELECT MIN(datetime) FROM prices WHERE interval = '15min'"
            ).fetchone()[0],
            "prices_15min_max_datetime": connection.execute(
                "SELECT MAX(datetime) FROM prices WHERE interval = '15min'"
            ).fetchone()[0],
            "prices_15min_tickers": connection.execute(
                "SELECT COUNT(DISTINCT ticker) FROM prices WHERE interval = '15min'"
            ).fetchone()[0],
            "data_version": connection.execute("PRAGMA data_version").fetchone()[0],
            "query_only": connection.execute("PRAGMA query_only").fetchone()[0],
        }
        connection.commit()
    finally:
        connection.close()

    database_after = _stat_record(database_path)
    repo_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "source_identity": source_identity,
        "repo_head": repo_head,
        "file_manifest": {
            "path": "legacy-price-files.tsv",
            "sha256": manifest_sha,
            **file_counts,
        },
        "alias_input": {
            "path": "ticker-aliases.tsv",
            "sha256": alias_sha,
            "rows": len(alias_rows),
        },
        "comparison": comparison,
        "observed_counts": observed_counts,
        "raw_differences": {
            "path": "raw-db-differences.tsv",
            "sha256": raw_diffs_sha,
            "rows": len(raw_diff_rows),
        },
        "canonical_differences": {
            "path": "canonical-db-differences.tsv",
            "sha256": canonical_diffs_sha,
            "rows": len(canonical_diff_rows),
        },
        "database": {
            "path": str(database_path),
            "stat_before": database_before,
            "stat_after": database_after,
            "logical_snapshot": logical_database_snapshot,
        },
    }
    result_bytes = _json_bytes(result)
    result_sha = _write_bytes(output_root / "result.json", result_bytes)
    identities = {
        "result.json": result_sha,
        "legacy-price-files.tsv": manifest_sha,
        "ticker-aliases.tsv": alias_sha,
        "raw-db-differences.tsv": raw_diffs_sha,
        "canonical-db-differences.tsv": canonical_diffs_sha,
    }
    identity_lines = [f"{digest}  {name}" for name, digest in sorted(identities.items())]
    _write_bytes(
        output_root / "SHA256SUMS",
        ("\n".join(identity_lines) + "\n").encode("ascii"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    _run(arguments.repo_root, arguments.output_root)


if __name__ == "__main__":
    main()
