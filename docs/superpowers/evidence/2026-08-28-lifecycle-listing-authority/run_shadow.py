"""Run the fixture-only listing-authority decision shadow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
FIXTURE = ROOT / "tests/fixtures/listing_authority/shadow-cases.json"
sys.path.insert(0, str(ROOT))


def _load_test_helpers():
    path = ROOT / "tests/test_security_lifecycle_decision_policy.py"
    spec = importlib.util.spec_from_file_location("listing_policy_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("listing_policy_helpers_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_test_helpers()


def _load_kernel_test_helpers():
    path = ROOT / "tests/test_security_lifecycle_fact_kernel.py"
    spec = importlib.util.spec_from_file_location("listing_kernel_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("listing_kernel_helpers_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K = _load_kernel_test_helpers()
LISTING_FIXTURES = ROOT / "tests/fixtures/listing_authority"
AT = "2026-08-28T22:00:00Z"


def _replace(value, old: str, new: str):
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, tuple):
        return tuple(_replace(item, old, new) for item in value)
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    return value


def _fixture(name: str) -> dict:
    if name in {
        "otc_symbol_continuation",
        "terminal_delisting",
        "nasdaq_absence_only",
    }:
        return H._listing_fixture(name)
    if name == "nms_symbol_continuation":
        return _replace(H._listing_fixture(name), "NASDAQ", "NYSE")
    if name == "same_symbol_venue_transfer":
        base = _replace(H._listing_fixture(name), "SAME", "QBTS")
        return _replace(base, "0001409970", "0001907982")
    if name == "historical_hapn":
        base = H._listing_fixture("nms_symbol_continuation")
        base = _replace(base, "OLD", "LC")
        base = _replace(base, "NEW", "HAPN")
        base["case"] = {**base["case"], "ticker": "HAPN"}
        return base
    if name == "historical_ccl":
        base = _replace(H._listing_fixture("completed_acquirer_active"), "KEEP", "CCL")
        base["case"] = {
            **base["case"],
            "ticker": "CCL",
            "issuer_name": "Carnival Corporation & plc",
            "event_kinds": ("acquisition_completed",),
        }
        return base
    if name == "historical_blbd":
        listing, listing_facts = H._active_listing(
            "nasdaq-blbd", "BLBD", cik="0001589526"
        )
        return {
            "case": {
                **H._case(ticker="BLBD", kinds=("merger_agreement",)),
                "cik": "0001589526",
            },
            "evidence": (H._evidence("sec", "regulator"), listing),
            "facts": (
                H._fact("sec", "source_ticker", "BLBD"),
                H._fact("sec", "issuer_cik", "0001589526"),
                H._fact("sec", "security_class", "common_stock"),
                H._fact(
                    "sec",
                    "transaction_structure",
                    {
                        "kind": "asset_acquisition",
                        "terms_status": "complete",
                        "counterparty_name": "Example Assets LLC",
                        "counterparty_ticker": None,
                        "counterparty_cik": None,
                    },
                ),
                H._fact("sec", "tracked_security_effect", "no_identity_change"),
            ) + listing_facts,
        }
    if name == "listing_cik_conflict":
        base = _fixture("nms_symbol_continuation")
        massive, massive_facts = H._active_listing(
            "massive-conflict",
            "NEW",
            adapter="massive_reference",
            market="stocks",
            venue="NASDAQ",
            cik="0000000001",
        )
        return {
            **base,
            "evidence": base["evidence"] + (massive,),
            "facts": base["facts"] + massive_facts,
        }
    raise KeyError(name)


def _read_bound_payload(binding: dict) -> bytes:
    filename = binding.get("filename")
    expected = binding.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise AssertionError("shadow_payload_filename_invalid")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AssertionError("shadow_payload_digest_invalid")
    path = (LISTING_FIXTURES / filename).resolve()
    if path.parent != LISTING_FIXTURES.resolve():
        raise AssertionError("shadow_payload_outside_fixture_root")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise AssertionError(f"shadow_payload_digest_mismatch:{filename}")
    return body


def _nasdaq_records(item: dict) -> tuple[tuple, list[dict]]:
    from src.security_lifecycle_listing_evidence import parse_nasdaq_directories

    bindings = {
        binding["component"]: binding
        for binding in item["listing_payloads"]
        if binding["adapter"] == "nasdaq_symbol_directory"
    }
    if set(bindings) != {"nasdaq_listed", "other_listed"}:
        raise AssertionError(f"shadow_nasdaq_binding_set:{item['id']}")
    nasdaq = _read_bound_payload(bindings["nasdaq_listed"])
    other = _read_bound_payload(bindings["other_listed"])
    snapshot = parse_nasdaq_directories(
        nasdaq_bytes=nasdaq,
        other_bytes=other,
        retrieved_at=AT,
    )
    observed = [
        {
            "filename": binding["filename"],
            "configured_sha256": binding["sha256"],
            "parser_input_sha256": hashlib.sha256(body).hexdigest(),
            "parser_received_exact_repository_bytes": True,
        }
        for binding, body in (
            (bindings["nasdaq_listed"], nasdaq),
            (bindings["other_listed"], other),
        )
    ]
    return snapshot.lookup(item["listing_query"]["ticker"]), observed


def _massive_record(item: dict, binding: dict) -> tuple[object, dict]:
    from src.security_lifecycle_listing_evidence import (
        _massive_source_url,
        parse_massive_ticker,
    )

    ticker = item["listing_query"]["ticker"]
    body = _read_bound_payload(binding)
    record = parse_massive_ticker(
        body,
        ticker,
        expected_active=binding["expected_active"],
        market=binding["market"],
        retrieved_at=AT,
        source_url=_massive_source_url(
            ticker,
            binding["expected_active"],
            binding["market"],
        ),
    )
    return record, {
        "filename": binding["filename"],
        "configured_sha256": binding["sha256"],
        "parser_input_sha256": hashlib.sha256(body).hexdigest(),
        "parser_received_exact_repository_bytes": True,
    }


def _listing_records(item: dict) -> tuple[tuple, str, str, list[dict]]:
    records: tuple = ()
    observed: list[dict] = []
    if any(
        binding["adapter"] == "nasdaq_symbol_directory"
        for binding in item["listing_payloads"]
    ):
        nasdaq_records, nasdaq_observed = _nasdaq_records(item)
        records += tuple(nasdaq_records)
        observed.extend(nasdaq_observed)
    for binding in item["listing_payloads"]:
        if binding["adapter"] != "massive_reference":
            continue
        record, payload_observed = _massive_record(item, binding)
        records += (record,)
        observed.append(payload_observed)
    if not records:
        raise AssertionError(f"shadow_listing_records_missing:{item['id']}")
    query = item["listing_query"]
    return records, query["source_ticker"], query["issuer_cik"], observed


def _persist_listing(item: dict, root: Path) -> tuple[tuple[dict, ...], tuple[dict, ...], dict]:
    from src.security_lifecycle_fact_kernel import SecurityLifecycleFactKernel
    from src.security_lifecycle_investigation import SecurityLifecycleInvestigationStore
    from src.security_lifecycle_listing_evidence import _result
    from src.security_lifecycle_sec_evidence import IdentityContext

    name = item["fixture"]
    records, source_ticker, cik, payloads = _listing_records(item)
    context = IdentityContext(
        case_id=f"offline-{name}",
        current_ticker=source_ticker,
        ticker_aliases=(source_ticker,),
        ibkr_conids=(),
        cik=cik,
        issuer_name="Offline fixture issuer",
        filing_date="2026-08-28",
        accession="offline-accession",
        filing_form="8-K",
        filing_items=("3.01",),
        event_kinds=("listing_status_review",),
        primary_start="2026-08-21",
        primary_end="2026-09-04",
        widened_start="2026-07-29",
        widened_end="2026-09-27",
    )
    result = _result(
        context=context,
        records=tuple(records),
        blockers=(),
        diagnostics={"listing_records": len(records)},
    )
    anchor_evidence = ()
    anchor_facts = ()
    if not result.facts:
        anchor = K._evidence(f"listing-{name}")
        anchor_evidence = (anchor,)
        anchor_facts = (K._fact(anchor),)
    path = root / f"{name}.db"
    connection = sqlite3.connect(path)
    try:
        store = SecurityLifecycleInvestigationStore(
            connection,
            id_factory=lambda prefix, ordinal: f"{prefix}_{ordinal:04d}",
        )
        case_id = store.ensure_case(
            source="sec_edgar",
            source_ref=f"offline-{name}",
            ticker=source_ticker,
            at=AT,
        )
        kernel = SecurityLifecycleFactKernel(store)
        claim = kernel.reserve_run(
            case_id=case_id,
            observation_fingerprint_sha256=hashlib.sha256(name.encode()).hexdigest(),
            policy_version="trusted-lifecycle-automation-v4",
            mode="historical",
            execution_revision="trusted-lifecycle-execution-r1",
            query_context={"case_id": case_id, "cik": cik, "aliases": [source_ticker]},
            diagnostics={"listing_records": len(records)},
            at=AT,
        )
        completed = kernel.complete_run(
            run_id=claim.run_id,
            evidence=result.evidence + anchor_evidence,
            facts=result.facts + anchor_facts,
            blockers=(),
            decision_tier="verified_automatic",
            action_readiness="not_applicable",
            retry_at=None,
            diagnostics={"listing_records": len(records)},
            at=AT,
        )
        assert completed.status == "succeeded"
        evidence = tuple(
            row
            for row in store.list_evidence(case_id)
            if row["source_family"] == "listing_authority"
        )
        listing_evidence_ids = {row["evidence_id"] for row in evidence}
        facts = tuple(
            dict(row)
            for row in connection.execute(
                "SELECT * FROM security_lifecycle_automation_facts "
                "WHERE automation_run_id=? ORDER BY fact_id",
                (claim.run_id,),
            )
            if row["evidence_id"] in listing_evidence_ids
        )
        assert len(evidence) == len(result.evidence)
        assert len(facts) == len(result.facts)
        return evidence, facts, {
            "strict_record_count": len(records),
            "kernel_evidence_count": len(evidence),
            "kernel_fact_count": len(facts),
            "kernel_non_listing_anchor_fact_count": len(anchor_facts),
            "adapters": sorted({record.adapter for record in records}),
            "document_sha256": sorted({record.source_document_sha256 for record in records}),
            "payloads": payloads,
        }
    finally:
        connection.close()


def _publisher_injection():
    evidence = (
        H._evidence("publisher-injected", "publisher"),
        H._evidence("web-injected", "general_web"),
    )
    facts = tuple(
        H._fact(evidence_id, fact_type, value)
        for evidence_id in ("publisher-injected", "web-injected")
        for fact_type, value in (
            ("successor_ticker", "WRONG"),
            ("destination_venue", "WRONG"),
            ("issuer_cik", "0000000001"),
        )
    )
    return evidence, facts


def run() -> dict:
    from src.security_lifecycle_decision_policy import evaluate_automation_decision

    authority = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    total_preview_calls = 0
    with TemporaryDirectory(prefix="arkscope-listing-shadow-") as directory:
      root = Path(directory)
      for item in authority["cases"]:
        fixture = _fixture(item["fixture"])
        listing_evidence, listing_facts, listing_authority = _persist_listing(item, root)
        base_evidence = tuple(
            row for row in fixture["evidence"] if row["source_family"] != "listing_authority"
        )
        base_facts = tuple(
            row
            for row in fixture["facts"]
            if row["evidence_id"]
            in {evidence["evidence_id"] for evidence in base_evidence}
        )
        calls = []

        def preview(request):
            calls.append(dict(request))
            return {
                "eligible": True,
                "block_reasons": (),
                "transition_kind": request["transition_kind"],
            }

        kwargs = {
            "case": fixture["case"],
            "evidence": base_evidence + listing_evidence,
            "facts": base_facts + listing_facts,
            "current_date": date(2026, 8, 28),
            "active_sources": ("manual_lists",),
            "transition_preview": preview,
        }
        baseline = evaluate_automation_decision(**kwargs)
        assert len(calls) == item["preview_calls"], (item["id"], calls)
        total_preview_calls += len(calls)

        injected_evidence, injected_facts = _publisher_injection()
        injected_calls = []

        def injected_preview(request):
            injected_calls.append(dict(request))
            return {
                "eligible": True,
                "block_reasons": (),
                "transition_kind": request["transition_kind"],
            }

        injected = evaluate_automation_decision(
            **{
                **kwargs,
                "evidence": base_evidence + listing_evidence + injected_evidence,
                "facts": base_facts + listing_facts + injected_facts,
                "transition_preview": injected_preview,
            }
        )
        assert injected == baseline
        assert injected_calls == calls
        rows.append(
            {
                "case": item["id"],
                "fixture": item["fixture"],
                "decision": asdict(baseline),
                "preview_calls": len(calls),
                "publisher_injection_inert": True,
                "listing_authority": listing_authority,
                "historical_sec_fact_authority": "frozen_repository_helper_no_provider_bytes",
            }
        )

    by_id = {row["case"]: row for row in rows}
    assert by_id["HAPN"]["decision"]["transition_requested"] is False
    assert by_id["QBTS"]["decision"]["outcomes"] == ("venue_transfer",)
    assert by_id["CCL"]["decision"]["outcomes"] == (
        "no_tracked_security_change",
    )
    assert by_id["BLBD"]["decision"]["outcomes"] == (
        "no_tracked_security_change",
    )
    assert by_id["NMS-A"]["decision"]["action_readiness"] == "transition_eligible"
    assert by_id["OTC-A"]["decision"]["destination_venue"] == "OTC"
    assert by_id["TERM"]["decision"]["outcomes"] == ("listing_ended",)
    assert by_id["MISS"]["decision"]["action_readiness"] == "waiting_market_confirmation"
    assert by_id["CONFLICT"]["decision"]["decision_issues"] == (
        "listing_authority_conflict",
    )
    return {
        "schema_version": 2,
        "fixture_authority": str(FIXTURE.relative_to(ROOT)),
        "policy_version": "trusted-lifecycle-automation-v4",
        "case_count": len(rows),
        "transition_preview_calls": total_preview_calls,
        "non_transition_preview_calls": 0,
        "publisher_injection_inert_count": len(rows),
        "listing_material_path": "repository_fixture_bytes_to_parser_session_to_listing_evidence_builder_to_real_fact_kernel_temporary_sqlite_to_policy",
        "listing_payload_bytes": "exact_repository_bytes_no_substitution_mutation_or_reserialization_before_parser_input",
        "historical_sec_limitation": "Historical SEC facts use frozen repository helpers because this packet contains no provider bytes or calls.",
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run()
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"cases": payload["case_count"], "preview_calls": payload["transition_preview_calls"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
