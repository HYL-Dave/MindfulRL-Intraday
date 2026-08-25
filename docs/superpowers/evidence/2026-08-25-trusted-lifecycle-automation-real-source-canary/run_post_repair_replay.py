"""Replay the consumed SEC/IBKR canary bytes against the repaired extractor."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from typing import Any


PACKET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKET_DIR.parents[3]
CANARY_REPORT = PACKET_DIR / "canary-report.json"
AUTHORIZATION = PACKET_DIR / "authorization.json"
TEST_MODULE = PROJECT_ROOT / "tests/test_security_lifecycle_sec_evidence.py"
PRODUCT_TEST_AUTHORITY = "1ec76167b70cffd3e9bd55c54de7dd2c5fd05c95"
MAX_REPORT_BYTES = 128 * 1024
SOURCE_FILES = {
    "BLBD": (
        "BLBD-0001589526-26-000044.html",
        "14bc650057c6d857e8fee1eb9402aa7645bd9edb82a9bb10eba76d98d913298e",
        43_663,
    ),
    "CCL": (
        "CCL-0001104659-26-057200.html",
        "892bb9f6cf6f9547006d1c6e4514d34afea644651cfced8025f65d8a30ed39a8",
        59_857,
    ),
    "HAPN": (
        "HAPN-0001409970-26-000087.html",
        "48ebd4ef3533760ecbc807f0342dfddf934da5bc686237d742568379c20b732f",
        31_285,
    ),
    "QBTS": (
        "QBTS-0001907982-26-000099.html",
        "bf1046a33ed9a67e555a483d5f640bbe59e24417bce3805bfe64b89fabf5bc43",
        31_068,
    ),
}
EXPECTED_DECISIONS = {
    "BLBD": {
        "action_readiness": "not_applicable",
        "counterparty_name": "Detroit Chassis LLC",
        "decision_issues": ("transaction_terms_partial",),
        "decision_tier": "verified_automatic",
        "destination_venue": None,
        "effective_date": None,
        "outcomes": ("no_tracked_security_change",),
        "rule_id": "lifecycle.no_identity_change",
        "successor_ticker": None,
        "transition_requested": False,
    },
    "CCL": {
        "action_readiness": "not_applicable",
        "counterparty_name": None,
        "decision_issues": ("transaction_terms_not_extracted",),
        "decision_tier": "verified_automatic",
        "destination_venue": None,
        "effective_date": None,
        "outcomes": ("no_tracked_security_change",),
        "rule_id": "lifecycle.no_identity_change",
        "successor_ticker": None,
        "transition_requested": False,
    },
    "HAPN": {
        "action_readiness": "transition_eligible",
        "counterparty_name": None,
        "decision_issues": (),
        "decision_tier": "verified_automatic",
        "destination_venue": "NASDAQ",
        "effective_date": "2026-06-22",
        "outcomes": ("symbol_changed", "venue_transfer"),
        "rule_id": "lifecycle.simple_symbol_continuation",
        "successor_ticker": "HAPN",
        "transition_requested": True,
    },
    "QBTS": {
        "action_readiness": "action_blocked",
        "counterparty_name": None,
        "decision_issues": ("market_corroboration_missing",),
        "decision_tier": "review_suggested",
        "destination_venue": "NASDAQ",
        "effective_date": "2026-07-27",
        "outcomes": ("venue_transfer",),
        "rule_id": "lifecycle.venue_transfer",
        "successor_ticker": "QBTS",
        "transition_requested": False,
    },
}
sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_product_test_authority() -> None:
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            PRODUCT_TEST_AUTHORITY,
            "--",
            "src",
            "data_sources",
            "apps",
            "tests",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _deny_network() -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("network_disabled_for_post_repair_replay")

    socket.socket = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    socket.getaddrinfo = denied  # type: ignore[assignment]


def _source_inventory() -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for ticker, (filename, expected_sha256, expected_bytes) in SOURCE_FILES.items():
        path = PACKET_DIR / "sec-source-bytes" / filename
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        expected = {"bytes": expected_bytes, "sha256": expected_sha256}
        if actual != expected:
            raise AssertionError(
                {"source_bytes_changed": ticker, "expected": expected, "actual": actual}
            )
        inventory[ticker] = {"file": filename, **actual}
    return inventory


def _fact_report(fact: object) -> dict[str, object]:
    return {
        "cited_text_sha256": _field(fact, "cited_text_sha256"),
        "evidence_id": str(_field(fact, "evidence_id")),
        "fact_type": str(_field(fact, "fact_type")),
        "rule_id": str(_field(fact, "extractor_rule_id", _field(fact, "rule_id"))),
        "rule_version": str(
            _field(fact, "extractor_rule_version", _field(fact, "rule_version"))
        ),
        "span_end_byte": _field(fact, "span_end_byte"),
        "span_start_byte": _field(fact, "span_start_byte"),
        "value": _field(fact, "normalized_value", _field(fact, "value")),
    }


def _evidence_report(evidence: object) -> dict[str, object]:
    return {
        "adapter": str(_field(evidence, "adapter")),
        "content_sha256": str(_field(evidence, "content_sha256")),
        "evidence_id": str(_field(evidence, "evidence_id")),
        "kind": str(_field(evidence, "kind")),
        "source_document_sha256": _field(evidence, "source_document_sha256"),
        "source_family": str(_field(evidence, "source_family")),
        "source_url": _field(evidence, "source_url"),
    }


def _case_report(
    *,
    name: str,
    collect_real,
    ibkr_evidence: tuple[dict[str, object], ...],
    ibkr_facts: tuple[dict[str, object], ...],
) -> dict[str, object]:
    from src.security_lifecycle_decision_policy import evaluate_automation_decision

    case, sec = collect_real(name)
    evidence: tuple[object, ...] = tuple(sec.evidence)
    facts: tuple[object, ...] = tuple(sec.facts)
    if name == "HAPN":
        evidence = (*evidence, *ibkr_evidence)
        facts = (*facts, *ibkr_facts)

    preview_calls: list[dict[str, object]] = []

    def preview(request):
        preview_calls.append(dict(request))
        return {
            "eligible": True,
            "block_reasons": (),
            "transition_kind": request["transition_kind"],
        }

    decision = evaluate_automation_decision(
        case={
            "ticker": case["observation"]["ticker"],
            "cik": case["observation"]["cik"],
        },
        evidence=evidence,
        facts=facts,
        current_date="2026-08-25",
        active_sources=("manual_lists",),
        transition_preview=preview,
    )
    expected = EXPECTED_DECISIONS[name]
    actual = {key: getattr(decision, key) for key in expected}
    if actual != expected:
        raise AssertionError(
            {"decision_changed": name, "expected": expected, "actual": actual}
        )

    if name == "HAPN":
        expected_preview = [
            {
                "transition_kind": "symbol_continuation",
                "source_ticker": "LC",
                "successor_ticker": "HAPN",
                "effective_date": "2026-06-22",
                "outcomes": ("symbol_changed", "venue_transfer"),
            }
        ]
        if preview_calls != expected_preview:
            raise AssertionError(
                {"hapn_preview_changed": preview_calls, "expected": expected_preview}
            )
    elif preview_calls:
        raise AssertionError({"unexpected_transition_preview": name, "calls": preview_calls})

    return {
        "source_ticker": case["observation"]["ticker"],
        "source_aliases": list(case["aliases"]),
        "sec_diagnostics": dict(sec.diagnostics),
        "sec_symbol_transitions": [list(item) for item in sec.symbol_transitions],
        "evidence": [
            _evidence_report(item)
            for item in sorted(evidence, key=lambda row: str(_field(row, "evidence_id")))
        ],
        "facts": [
            _fact_report(item)
            for item in sorted(
                facts,
                key=lambda row: (
                    str(_field(row, "fact_type")),
                    json.dumps(
                        _field(row, "normalized_value", _field(row, "value")),
                        sort_keys=True,
                    ),
                    str(_field(row, "evidence_id")),
                ),
            )
        ],
        "decision": dataclasses.asdict(decision),
        "transition_preview_calls": preview_calls,
    }


def build_report() -> dict[str, object]:
    _assert_product_test_authority()
    head = _git("rev-parse", "HEAD")
    source_inventory = _source_inventory()
    canary = json.loads(CANARY_REPORT.read_text(encoding="utf-8"))
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if canary["authorization"]["rerun_authorized"] is not False:
        raise AssertionError("canary_rerun_authority_changed")
    if authorization["authorization_status"] != "CONSUMED":
        raise AssertionError("canary_authorization_not_consumed")
    if canary["ibkr"]["readonly"] is not True:
        raise AssertionError("ibkr_canary_not_readonly")
    if canary["ibkr"]["requests_made"] != 2:
        raise AssertionError("ibkr_request_count_changed")

    test_module = runpy.run_path(str(TEST_MODULE))
    __import__("src.security_lifecycle_decision_policy")
    __import__("src.security_lifecycle_sec_evidence")
    _deny_network()
    collect_real = test_module["_collect_real"]
    ibkr_evidence = tuple(canary["ibkr"]["evidence"])
    ibkr_facts = tuple(canary["ibkr"]["facts"])
    cases = {
        name: _case_report(
            name=name,
            collect_real=collect_real,
            ibkr_evidence=ibkr_evidence,
            ibkr_facts=ibkr_facts,
        )
        for name in sorted(SOURCE_FILES)
    }
    return {
        "schema_version": 1,
        "git_head": head,
        "product_test_authority": PRODUCT_TEST_AUTHORITY,
        "inputs": {
            "authorization_sha256": _sha256(AUTHORIZATION),
            "canary_report_sha256": _sha256(CANARY_REPORT),
            "sec_source_bytes": source_inventory,
            "ibkr_evidence_source": "consumed_read_only_canary_report",
        },
        "execution": {
            "external_network_calls": 0,
            "general_web_search_calls": 0,
            "production_database_operations": 0,
            "provider_calls": 0,
            "replay_source": "captured_public_sec_bytes_and_secret_safe_ibkr_receipt",
            "socket_policy": "denied_after_offline_dependency_import",
        },
        "admission": {
            "extractor_rule_version": "2",
            "hapn_real_a_to_b_transition_eligible": True,
            "qbts_same_symbol_venue_transfer_requires_market_corroboration": True,
            "ccl_no_tracked_security_change": True,
            "blbd_asset_purchase_no_tracked_security_change": True,
            "real_production_transition_apply_exercised": False,
        },
        "cases": cases,
        "limitations": [
            "This is an offline replay of the one consumed canary, not another provider call.",
            "Only HAPN has captured market-infrastructure corroboration in this packet.",
            "No production database, migration, transition apply, reverse, merge, push, or cutover occurred.",
            "One real A-to-B example does not establish broad A-to-B precision.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(
        build_report(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if len(payload) > MAX_REPORT_BYTES:
        raise RuntimeError("post_repair_replay_report_exceeds_limit")
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
