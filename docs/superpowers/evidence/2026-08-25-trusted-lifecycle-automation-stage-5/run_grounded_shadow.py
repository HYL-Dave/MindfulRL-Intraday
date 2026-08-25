"""Produce the bounded Stage 5 four-case shadow report offline."""

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
TEST_MODULE = PROJECT_ROOT / "tests/test_security_lifecycle_grounded_shadow.py"
MANIFEST = PROJECT_ROOT / "tests/fixtures/security_lifecycle_grounded_shadow.json"
SOURCE_SHAPES = PROJECT_ROOT / "tests/fixtures/security_lifecycle_automation_sec.json"
LEGACY_SNAPSHOT = PROJECT_ROOT / "tests/fixtures/security_lifecycle_legacy_37.json"
STAGE4_MANIFEST = (
    PROJECT_ROOT
    / "docs/superpowers/evidence/2026-08-25-trusted-lifecycle-automation-stage-4/SHA256SUMS"
)
MAX_REPORT_BYTES = 64 * 1024
sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _fact_value(fact: object) -> Any:
    value = _field(fact, "normalized_value", None)
    if value is not None:
        return value
    return _field(fact, "value", None)


def _deny_network() -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("network_disabled_for_grounded_shadow")

    socket.socket = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    socket.getaddrinfo = denied  # type: ignore[assignment]


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _case_report(name: str, manifest_case: dict, shadow) -> dict:
    decision, sec, evidence, facts, preview_calls = shadow(name)
    decision_data = dataclasses.asdict(decision)
    return {
        "ticker": name,
        "reviewed_source_pointers": manifest_case["snapshot_rows"],
        "historical_identity_change": manifest_case["historical_identity_change"],
        "extracted_fact_types": sorted({str(_field(fact, "fact_type")) for fact in facts}),
        "extracted_facts": [
            {
                "evidence_id": str(_field(fact, "evidence_id")),
                "fact_type": str(_field(fact, "fact_type")),
                "rule_id": str(
                    _field(fact, "extractor_rule_id", _field(fact, "rule_id"))
                ),
                "value": _fact_value(fact),
            }
            for fact in sorted(
                facts,
                key=lambda item: (
                    str(_field(item, "fact_type")),
                    str(_fact_value(item)),
                    str(_field(item, "evidence_id")),
                ),
            )
        ],
        "evidence": [
            {
                "adapter": str(_field(item, "adapter")),
                "content_sha256": str(_field(item, "content_sha256")),
                "evidence_id": str(_field(item, "evidence_id")),
                "kind": str(_field(item, "kind")),
                "source_family": str(_field(item, "source_family")),
                "source_url": _field(item, "source_url"),
            }
            for item in sorted(evidence, key=lambda row: str(_field(row, "evidence_id")))
        ],
        "sec_diagnostics": dict(sec.diagnostics),
        "decision": {
            key: decision_data[key]
            for key in (
                "decision_tier",
                "action_readiness",
                "relevance",
                "confidence",
                "outcomes",
                "successor_ticker",
                "destination_venue",
                "effective_date",
                "rule_id",
                "rule_version",
                "decision_issues",
                "transition_requested",
            )
        },
        "transition_preview_calls": preview_calls,
    }


def build_report() -> dict:
    test_module = runpy.run_path(str(TEST_MODULE))
    for module in (
        "src.security_lifecycle_decision_policy",
        "src.security_lifecycle_ibkr_evidence",
        "src.security_lifecycle_sec_evidence",
    ):
        __import__(module)
    _deny_network()
    shadow = test_module["_shadow"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "git_head": _git_head(),
        "authority": {
            "grounded_manifest_sha256": _sha256(MANIFEST),
            "legacy_snapshot_sha256": _sha256(LEGACY_SNAPSHOT),
            "source_shape_fixture_sha256": _sha256(SOURCE_SHAPES),
            "stage4_execution_manifest_sha256": _sha256(STAGE4_MANIFEST),
        },
        "execution": {
            "network_calls": 0,
            "production_database_operations": 0,
            "provider_calls": 0,
            "replay_socket_policy": "denied_after_offline_dependency_import",
        },
        "provenance": {
            "case_identity": "reviewed_repository_snapshot",
            "sec_source_text": "synthetic_source_shape_not_captured_provider_bytes",
            "ibkr_contract_snapshot": "synthetic_ibkr_contract_shape",
        },
        "coverage": {
            "historical_a_to_b_cases": 1,
            "historical_a_to_b_ticker": "HAPN",
            "real_production_a_to_b_apply_exercised": False,
            "execution_reverse_authority": "stage_4_scratch_apply_ack_reverse",
        },
        "limitations": [
            "No fresh SEC, IBKR, news, model, or general-search response was obtained.",
            "Synthetic filing prose and IBKR reply shapes are not captured provider bytes.",
            "HAPN is already keyed by its successor ticker, so no real HAPN-to-HAPN transition is requested.",
            "One historical A-to-B example plus synthetic execution does not establish broad A-to-B precision.",
        ],
        "cases": [
            _case_report(name, manifest["cases"][name], shadow)
            for name in sorted(manifest["cases"])
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
        raise RuntimeError("grounded_shadow_report_exceeds_limit")
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
