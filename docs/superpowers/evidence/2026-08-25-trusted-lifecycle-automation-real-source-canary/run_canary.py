#!/usr/bin/env python3
"""Run the one authorized real-source lifecycle canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlsplit


PACKET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKET_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from data_sources.sec_transport import SecRequestBudget, SecTransport
from src.security_lifecycle_decision_policy import evaluate_automation_decision
from src.security_lifecycle_sec_evidence import (
    IdentityContext,
    build_identity_context,
    collect_sec_evidence,
)


PRODUCT_AUTHORITY = "7cb479a8058793dc29cbb75bb4ab98b9d6a6f231"
AUTHORIZATION_SCOPE = "four_exact_sec_documents_and_one_ibkr_read_only_shape"
EXPECTED_BRANCH = "trusted-lifecycle-automation-stages3-5"
EXPECTED_OUTPUT_DIR = Path("/tmp/arkscope-lifecycle-real-source-canary-20260825")
EXPECTED_CONFIG_FIELDS = frozenset(
    {
        ("sec_edgar", "user_agent"),
        ("ibkr", "host"),
        ("ibkr", "port"),
        ("ibkr", "client_id"),
    }
)
CASE_ACCESSIONS = {
    "HAPN": "0001409970-26-000087",
    "QBTS": "0001907982-26-000099",
    "CCL": "0001104659-26-057200",
    "BLBD": "0001589526-26-000044",
}
FIRST_DISCOVERY_TICKER = {"HAPN": "LC"}
EXPECTED_PRODUCT_HASHES = {
    "src/security_lifecycle_sec_evidence.py": "764fed401c95904a9e8a36b097fd6c3e692b596156b117093f9b49053edd03e8",
    "src/security_lifecycle_ibkr_evidence.py": "81cdf59fc486a12a2d315422ffc0b481a91eaad6afae04bc27d888701a1b2e11",
    "src/security_lifecycle_decision_policy.py": "38cf449e737775880159605c2c344131664bacccadee79768f0b1940e5635361",
    "src/service/security_lifecycle_automation_scheduler.py": "2647ce6e432098b3467028cff0c677b0809fd4d06ef7e667baa84643872b4deb",
    "data_sources/sec_transport.py": "30de5b2d564b5ab6c364e27bf1f81c7b7009abe8b696be19b140634cbc031e33",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _git_root() -> Path:
    raw = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    root = Path(raw).resolve()
    if root != PROJECT_ROOT:
        raise RuntimeError("canary_project_root_mismatch")
    return root


def _production_root() -> Path:
    raw = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True
    ).strip()
    common = Path(raw)
    if not common.is_absolute():
        common = (_git_root() / common).resolve()
    return common.resolve().parent


def _git_state(root: Path) -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if status:
        raise RuntimeError("worktree_not_clean")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("canary_branch_mismatch")
    return {
        "branch": branch,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "worktree_clean": True,
    }


def _verify_product_bytes(root: Path) -> None:
    for relative, expected in EXPECTED_PRODUCT_HASHES.items():
        actual = _sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"product_byte_mismatch:{relative}")


def _stat(path: Path) -> dict[str, int]:
    current = path.stat()
    return {
        "inode": current.st_ino,
        "size": current.st_size,
        "mtime_ns": current.st_mtime_ns,
    }


def _optional_stat(path: Path) -> dict[str, int] | None:
    try:
        return _stat(path)
    except FileNotFoundError:
        return None


def _read_provider_config(profile: Path) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    before = _stat(profile)
    sidecars_before = {
        suffix: _optional_stat(Path(f"{profile}{suffix}"))
        for suffix in ("-shm", "-wal")
    }
    uri = f"file:{quote(str(profile.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT provider, field, value FROM data_provider_config "
            "WHERE (provider = 'sec_edgar' AND field = 'user_agent') "
            "OR (provider = 'ibkr' AND field IN ('host', 'port', 'client_id'))"
        ).fetchall()
        connection_total_changes = conn.total_changes
    finally:
        conn.close()
    after = _stat(profile)
    sidecars_after = {
        suffix: _optional_stat(Path(f"{profile}{suffix}"))
        for suffix in ("-shm", "-wal")
    }
    values = {
        (str(provider), str(field)): str(value or "").strip()
        for provider, field, value in rows
    }
    if frozenset(values) != EXPECTED_CONFIG_FIELDS or any(not value for value in values.values()):
        raise RuntimeError("provider_config_field_set_mismatch")
    if connection_total_changes != 0:
        raise RuntimeError("production_profile_write_detected")
    if "@" not in values[("sec_edgar", "user_agent")]:
        raise RuntimeError("sec_identity_unconfigured")
    try:
        port = int(values[("ibkr", "port")])
        client_id = int(values[("ibkr", "client_id")])
    except ValueError as exc:
        raise RuntimeError("ibkr_config_invalid") from exc
    if not 1 <= port <= 65535 or not 0 <= client_id <= 19:
        raise RuntimeError("ibkr_config_invalid")
    return values, {
        "field_names": [f"{provider}.{field}" for provider, field in sorted(values)],
        "main_file_before": before,
        "main_file_after": after,
        "main_file_equal": before == after,
        "query_only": True,
        "rows_read": len(rows),
        "sidecars_after": sidecars_after,
        "sidecars_before": sidecars_before,
        "sqlite_total_changes": connection_total_changes,
        "values_persisted": False,
    }


def _load_cases(root: Path) -> dict[str, dict[str, Any]]:
    legacy = json.loads(
        (root / "tests/fixtures/security_lifecycle_legacy_37.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    rows_by_key = {
        (str(row["ticker"]), str(row["source_ref"])): row for row in legacy
    }
    cases: dict[str, dict[str, Any]] = {}
    for ticker, accession in CASE_ACCESSIONS.items():
        row = rows_by_key[(ticker, accession)]
        sibling_kinds = sorted(
            {
                str(candidate["event_type"])
                for candidate in legacy
                if candidate["ticker"] == ticker
                and candidate["source_ref"] == accession
            }
        )
        items = json.loads(str(row["filing_items_json"]))
        parsed = urlsplit(str(row["evidence_url"]))
        primary_document = Path(parsed.path).name
        if (
            parsed.scheme != "https"
            or parsed.hostname != "www.sec.gov"
            or "/Archives/edgar/data/" not in parsed.path
            or not primary_document
        ):
            raise RuntimeError(f"invalid_reviewed_sec_url:{ticker}")
        current_ticker = FIRST_DISCOVERY_TICKER.get(ticker, ticker)
        context = build_identity_context(
            case_id=f"real-source-canary:{ticker}:{accession}",
            observation={
                "ticker": current_ticker,
                "cik": row["cik"],
                "issuer_name": row["issuer_name"],
                "filing_date": row["filing_date"],
                "source_ref": accession,
                "filing_form": row["filing_form"],
                "filing_items": items,
                "event_kinds": sibling_kinds,
            },
            ticker_aliases=(current_ticker,),
        )
        submissions = {
            "cik": str(row["cik"]),
            "filings": {
                "recent": {
                    "form": [row["filing_form"]],
                    "filingDate": [row["filing_date"]],
                    "accessionNumber": [accession],
                    "primaryDocument": [primary_document],
                    "primaryDocDescription": [row["description"]],
                    "items": [",".join(items)],
                    "cik": [row["cik"]],
                    "ticker": [current_ticker],
                }
            },
        }
        cases[ticker] = {
            "context": context,
            "row": row,
            "submissions": submissions,
            "url": str(row["evidence_url"]),
        }
    if len({case["url"] for case in cases.values()}) != 4:
        raise RuntimeError("sec_url_count_mismatch")
    return cases


class _ExactDocumentTransport:
    def __init__(
        self,
        *,
        live: SecTransport,
        context: IdentityContext,
        submissions: Mapping[str, Any],
        exact_url: str,
        capture_path: Path,
    ):
        self.live = live
        self.context = context
        self.submissions = submissions
        self.exact_url = exact_url
        self.capture_path = capture_path
        self.response_body: bytes | None = None
        self.response_status: int | None = None
        self.get_json_calls = 0
        self.document_calls = 0

    def get_json(self, url: str, **kwargs: Any) -> Mapping[str, Any]:
        del kwargs
        expected = f"https://data.sec.gov/submissions/CIK{self.context.cik}.json"
        if url != expected or self.get_json_calls:
            raise RuntimeError("unexpected_sec_submissions_call")
        self.get_json_calls += 1
        return self.submissions

    def get(self, url: str, **kwargs: Any):
        if url != self.exact_url or self.document_calls:
            raise RuntimeError("unexpected_sec_document_call")
        self.document_calls += 1
        response = self.live.get(url, **kwargs)
        self.response_status = response.status_code
        self.response_body = response.body
        self.capture_path.write_bytes(response.body)
        self.capture_path.chmod(0o600)
        return response


def _fact_row(fact: object) -> dict[str, Any]:
    return {
        "cited_text": str(getattr(fact, "cited_text", "")),
        "cited_text_sha256": str(getattr(fact, "cited_text_sha256", "")),
        "evidence_id": str(getattr(fact, "evidence_id", "")),
        "fact_type": str(getattr(fact, "fact_type", "")),
        "rule_id": str(getattr(fact, "rule_id", "")),
        "rule_version": str(getattr(fact, "rule_version", "")),
        "span_end_byte": int(getattr(fact, "span_end_byte", 0)),
        "span_start_byte": int(getattr(fact, "span_start_byte", 0)),
        "value": getattr(fact, "value", None),
    }


def _market_fact_row(fact: object) -> dict[str, Any]:
    return {
        "cited_text_sha256": str(getattr(fact, "cited_text_sha256", "")),
        "evidence_id": str(getattr(fact, "evidence_id", "")),
        "fact_type": str(getattr(fact, "fact_type", "")),
        "rule_id": str(getattr(fact, "extractor_rule_id", "")),
        "rule_version": str(getattr(fact, "extractor_rule_version", "")),
        "span_end_byte": int(getattr(fact, "source_span_end", 0)),
        "span_start_byte": int(getattr(fact, "source_span_start", 0)),
        "value": getattr(fact, "normalized_value", None),
    }


def _decision(
    *, context: IdentityContext, evidence: tuple[object, ...], facts: tuple[object, ...]
) -> dict[str, Any]:
    decision = evaluate_automation_decision(
        case={
            "case_id": context.case_id,
            "ticker": context.current_ticker,
            "cik": context.cik,
            "issuer_name": context.issuer_name,
            "filing_date": context.filing_date,
            "event_kinds": context.event_kinds,
        },
        evidence=evidence,
        facts=facts,
        current_date=datetime.now(timezone.utc).date(),
        active_sources=("manual_lists",),
        transition_preview=lambda _request: {
            "eligible": False,
            "reasons": ["canary_preview_not_authorized"],
        },
    )
    return asdict(decision)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-scope", required=True)
    parser.add_argument("--profile-db", type=Path, required=True)
    parser.add_argument("--shared-lock-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.authorization_scope != AUTHORIZATION_SCOPE:
        raise RuntimeError("authorization_scope_mismatch")
    if args.output_dir.resolve() != EXPECTED_OUTPUT_DIR:
        raise RuntimeError("output_dir_path_mismatch")
    if args.output_dir.exists():
        raise RuntimeError("output_dir_must_be_absent")
    root = _git_root()
    production_root = _production_root()
    expected_profile = (production_root / "data/profile_state.db").resolve()
    expected_lock_dir = (production_root / "data/locks").resolve()
    if args.profile_db.resolve() != expected_profile:
        raise RuntimeError("production_profile_path_mismatch")
    if args.shared_lock_dir.resolve() != expected_lock_dir:
        raise RuntimeError("shared_lock_dir_path_mismatch")
    git_state = _git_state(root)
    _verify_product_bytes(root)
    values, config_report = _read_provider_config(args.profile_db.resolve())
    if not args.shared_lock_dir.is_dir():
        raise RuntimeError("shared_lock_dir_missing")

    args.output_dir.mkdir(mode=0o700, parents=True)
    source_dir = args.output_dir / "sec-source-bytes"
    source_dir.mkdir(mode=0o700)
    os.environ["ARKSCOPE_SEC_USER_AGENT"] = values[("sec_edgar", "user_agent")]
    os.environ["IBKR_HOST"] = values[("ibkr", "host")]
    os.environ["IBKR_PORT"] = values[("ibkr", "port")]
    os.environ["IBKR_CLIENT_ID"] = values[("ibkr", "client_id")]
    os.environ["ARKSCOPE_LOCK_DIR"] = str(args.shared_lock_dir.resolve())

    cases = _load_cases(root)
    budget = SecRequestBudget(
        max_attempts=4,
        max_documents=4,
        max_document_bytes=1_048_576,
        max_total_bytes=4 * 1_048_576,
    )
    live = SecTransport(user_agent=values[("sec_edgar", "user_agent")])
    retrieved_at = _now()
    sec_results: dict[str, Any] = {}
    sec_objects: dict[str, Any] = {}
    shared_transport_diagnostics: dict[str, int]
    try:
        for ticker in ("HAPN", "QBTS", "CCL", "BLBD"):
            case = cases[ticker]
            source_file = f"{ticker}-{CASE_ACCESSIONS[ticker]}.html"
            exact = _ExactDocumentTransport(
                live=live,
                context=case["context"],
                submissions=case["submissions"],
                exact_url=case["url"],
                capture_path=source_dir / source_file,
            )
            try:
                result = collect_sec_evidence(
                    context=case["context"],
                    transport=exact,
                    retrieved_at=retrieved_at,
                    budget=budget,
                )
            except (TypeError, ValueError) as exc:
                body = exact.response_body
                sec_results[ticker] = {
                    "accession": CASE_ACCESSIONS[ticker],
                    "blockers": ["canary_extractor_error"],
                    "conflicts": {},
                    "decision_without_market": None,
                    "diagnostics": budget.diagnostics(),
                    "document_bytes": len(body) if body is not None else 0,
                    "document_calls": exact.document_calls,
                    "document_sha256": (
                        _sha256_bytes(body) if body is not None else None
                    ),
                    "evidence": [],
                    "extractor_error_type": type(exc).__name__,
                    "extractor_status": "failed",
                    "facts": [],
                    "filing_chain_complete": False,
                    "http_status": exact.response_status,
                    "source_file": source_file if body is not None else None,
                    "source_url": case["url"],
                    "submissions_network_calls": 0,
                    "symbol_transitions": [],
                }
                continue
            sec_objects[ticker] = result
            body = exact.response_body
            sec_results[ticker] = {
                "accession": CASE_ACCESSIONS[ticker],
                "blockers": list(result.blockers),
                "conflicts": {key: list(value) for key, value in result.conflicts.items()},
                "decision_without_market": _decision(
                    context=case["context"],
                    evidence=result.evidence,
                    facts=result.facts,
                ),
                "diagnostics": dict(result.diagnostics),
                "document_bytes": len(body) if body is not None else 0,
                "document_calls": exact.document_calls,
                "document_sha256": _sha256_bytes(body) if body is not None else None,
                "evidence": [asdict(row) for row in result.evidence],
                "extractor_error_type": None,
                "extractor_status": "succeeded",
                "facts": [_fact_row(row) for row in result.facts],
                "filing_chain_complete": all(
                    row.source_locator.get("filing_chain_complete") is True
                    for row in result.evidence
                ) if result.evidence else False,
                "http_status": exact.response_status,
                "source_file": source_file if body is not None else None,
                "source_url": case["url"],
                "submissions_network_calls": 0,
                "symbol_transitions": [list(value) for value in result.symbol_transitions],
            }
    finally:
        shared_transport_diagnostics = live.diagnostics(budget)
        live.close()

    from src.service.security_lifecycle_automation_scheduler import _ibkr_evidence

    hapn_context = cases["HAPN"]["context"]
    try:
        ibkr_result, ibkr_facts = _ibkr_evidence(
            hapn_context,
            at=retrieved_at,
            regulator_successors=("HAPN",),
        )
    # This is a one-shot canary: preserve a secret-safe terminal receipt rather
    # than losing the authorized session to an unexpected adapter exception.
    except Exception as exc:
        ibkr_result = None
        ibkr_report = {
            "blockers": ["canary_ibkr_adapter_error"],
            "contract_status": "canary_error",
            "corroboration_family_count": 0,
            "evidence": [],
            "error_type": type(exc).__name__,
            "facts": [],
            "integrated_hapn_decision": None,
            "readonly": True,
            "requests_made": None,
            "source_families": [],
            "symbols_max": ["LC", "HAPN"],
        }
    else:
        hapn_sec = sec_objects.get("HAPN")
        combined_evidence = (
            tuple(hapn_sec.evidence) if hapn_sec is not None else ()
        ) + tuple(ibkr_result.evidence)
        combined_facts = (
            tuple(hapn_sec.facts) if hapn_sec is not None else ()
        ) + tuple(ibkr_facts)
        ibkr_report = {
            "blockers": list(ibkr_result.blockers),
            "contract_status": ibkr_result.contract_status,
            "corroboration_family_count": ibkr_result.corroboration_family_count,
            "evidence": [asdict(row) for row in ibkr_result.evidence],
            "error_type": None,
            "facts": [_market_fact_row(row) for row in ibkr_facts],
            "integrated_hapn_decision": _decision(
                context=hapn_context,
                evidence=combined_evidence,
                facts=combined_facts,
            ),
            "readonly": True,
            "requests_made": ibkr_result.requests_made,
            "source_families": list(ibkr_result.source_families),
            "symbols_max": ["LC", "HAPN"],
        }

    report = {
        "authorization": {
            "scope": AUTHORIZATION_SCOPE,
            "ibkr_sessions_consumed": 1,
            "rerun_authorized": False,
            "sec_acquisitions_consumed": 1,
        },
        "completed_at": _now(),
        "config_read": config_report,
        "git": git_state,
        "ibkr": ibkr_report,
        "operations": {
            "app_cutovers": 0,
            "general_web_search_calls": 0,
            "merges": 0,
            "news_provider_calls": 0,
            "production_database_migrations": 0,
            "production_database_writes": 0,
            "pushes": 0,
            "production_profile_config_queries": 2,
            "production_profile_config_rows": 8,
            "sec_document_attempts": budget.attempt_count,
            "sec_logical_documents": budget.document_count,
            "sec_submissions_network_calls": 0,
        },
        "product_test_authority": PRODUCT_AUTHORITY,
        "retrieved_at": retrieved_at,
        "schema_version": 1,
        "sec": {
            "budget": budget.diagnostics(),
            "cases": sec_results,
            "shared_transport": shared_transport_diagnostics,
        },
    }
    output = args.output_dir / "canary-report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.chmod(0o600)
    print(
        json.dumps(
            {
                "ibkr_contract_status": ibkr_report["contract_status"],
                "ibkr_requests_made": (
                    ibkr_result.requests_made if ibkr_result is not None else None
                ),
                "output": str(output),
                "sec_attempt_count": budget.attempt_count,
                "sec_document_count": budget.document_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
