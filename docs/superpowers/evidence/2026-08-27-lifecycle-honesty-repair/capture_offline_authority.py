"""Capture replay, citation, and projection authority in temporary SQLite only."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


AUTHORITY = {
    "scope": "offline_fixture_and_scratch_only",
    "provider_calls": 0,
    "production_database_reads": 0,
    "production_database_writes": 0,
    "production_database_preflights": 0,
    "production_database_backups": 0,
    "production_database_migrations": 0,
    "production_database_restores": 0,
    "app_restarts": 0,
    "merges": 0,
    "pushes": 0,
}
PRODUCT_TEST_AUTHORITY = "23fe53b72be6b0b0629b596100b41f5ec6a0dcf9"


@contextmanager
def _observe_forbidden_authority_calls():
    from src.security_lifecycle_investigation import (
        SecurityLifecycleInvestigationStore,
    )
    from src import ticker_identity_transition

    counts = {
        "transition_preview": 0,
        "transition_approval": 0,
        "transition_apply": 0,
        "transition_reverse": 0,
        "acknowledgement": 0,
    }
    targets = {
        "transition_preview": (
            (ticker_identity_transition, "build_transition_preview"),
            (ticker_identity_transition, "build_automation_transition_preflight"),
        ),
        "transition_approval": (
            (ticker_identity_transition.TickerIdentityTransitionStore, "approve"),
            (
                ticker_identity_transition.TickerIdentityTransitionStore,
                "approve_automation",
            ),
        ),
        "transition_apply": (
            (ticker_identity_transition.TickerIdentityTransitionStore, "apply"),
        ),
        "transition_reverse": (
            (ticker_identity_transition.TickerIdentityTransitionStore, "reverse"),
        ),
        "acknowledgement": (
            (SecurityLifecycleInvestigationStore, "acknowledge_case"),
            (
                ticker_identity_transition.TickerIdentityTransitionStore,
                "acknowledge_activity",
            ),
        ),
    }

    with ExitStack() as stack:
        for name, boundaries in targets.items():
            for owner, attribute in boundaries:
                def reject(*_args, _name=name, **_kwargs):
                    counts[_name] += 1
                    raise AssertionError(f"forbidden_authority_call:{_name}")

                stack.enter_context(patch.object(owner, attribute, reject))
        yield counts


def _calibrate_authority_call_observer() -> dict:
    from src.security_lifecycle_investigation import (
        SecurityLifecycleInvestigationStore,
    )
    from src import ticker_identity_transition

    probes = {
        "transition_preview": (ticker_identity_transition, "build_transition_preview"),
        "transition_approval": (
            ticker_identity_transition.TickerIdentityTransitionStore,
            "approve",
        ),
        "transition_apply": (
            ticker_identity_transition.TickerIdentityTransitionStore,
            "apply",
        ),
        "transition_reverse": (
            ticker_identity_transition.TickerIdentityTransitionStore,
            "reverse",
        ),
        "acknowledgement": (
            SecurityLifecycleInvestigationStore,
            "acknowledge_case",
        ),
    }
    with _observe_forbidden_authority_calls() as counts:
        for name, (owner, attribute) in probes.items():
            try:
                getattr(owner, attribute)()
            except AssertionError as exc:
                assert str(exc) == f"forbidden_authority_call:{name}"
            else:
                raise AssertionError(f"authority_observer_inactive:{name}")
        observed = dict(counts)
    expected = {name: 1 for name in probes}
    assert observed == expected
    return {
        "method": "fail_closed_boundary_wrappers",
        "expected": expected,
        "observed": observed,
    }


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _scratch(path: Path):
    from src.security_lifecycle_fact_kernel import SecurityLifecycleFactKernel
    from src.security_lifecycle_investigation import SecurityLifecycleInvestigationStore

    connection = sqlite3.connect(path)
    store = SecurityLifecycleInvestigationStore(
        connection,
        id_factory=lambda prefix, ordinal: f"{prefix}_{ordinal:04d}",
    )
    case_id = store.ensure_case(
        source="sec_edgar",
        source_ref="0001409970-26-000131",
        ticker="HAPN",
        at="2026-08-25T01:00:00Z",
    )
    return connection, store, SecurityLifecycleFactKernel(store), case_id


def _reserve(kernel, case_id: str, *, revision: str, at: str, fingerprint: str = "a" * 64):
    return kernel.reserve_run(
        case_id=case_id,
        observation_fingerprint_sha256=fingerprint,
        policy_version="trusted-lifecycle-automation-v3",
        mode="historical",
        execution_revision=revision,
        query_context={"case_id": case_id, "cik": "0001409970", "aliases": ["HAPN", "LC"]},
        diagnostics={"sec_attempts": 0},
        at=at,
    )


def _success_evidence():
    from tests.test_security_lifecycle_fact_kernel import _evidence, _fact

    evidence = _evidence("authority")
    return evidence, _fact(evidence)


def _capture_replay(path: Path) -> dict:
    connection, store, kernel, case_id = _scratch(path)
    try:
        legacy = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r0",
            at="2026-08-25T01:00:00Z",
        )
        legacy_row = store.get_automation_run(legacy.run_id)
        legacy_context = json.loads(legacy_row["query_context_json"])
        del legacy_context["execution_revision"]
        connection.execute(
            "UPDATE security_lifecycle_automation_runs SET query_context_json=? WHERE run_id=?",
            (_canonical(legacy_context), legacy.run_id),
        )
        connection.commit()
        kernel.fail_run(
            run_id=legacy.run_id,
            failure_code="persistence_failed",
            diagnostics={"persist_failures": 1},
            at="2026-08-25T02:00:00Z",
        )
        predecessor_before = store.get_automation_run(legacy.run_id)
        predecessor_before_bytes = _canonical(predecessor_before)

        replay = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r1",
            at="2026-08-26T00:00:00Z",
        )
        predecessor_after = store.get_automation_run(legacy.run_id)
        predecessor_after_bytes = _canonical(predecessor_after)
        evidence, fact = _success_evidence()
        completed = kernel.complete_run(
            run_id=replay.run_id,
            evidence=(evidence,),
            facts=(fact,),
            blockers=(),
            decision_tier="verified_automatic",
            action_readiness="transition_eligible",
            retry_at=None,
            diagnostics={"sec_attempts": 1},
            at="2026-08-26T01:00:00Z",
        )
        r2 = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r2",
            at="2026-08-27T00:00:00Z",
        )
        replay_context = json.loads(
            store.get_automation_run(replay.run_id)["query_context_json"]
        )
        rows = store.list_automation_runs(case_id)
        report = {
            "semantic_policy_version": "trusted-lifecycle-automation-v3",
            "legacy_failed": {
                "run_id": legacy.run_id,
                "status": predecessor_before["status"],
                "execution_revision_present": "execution_revision" in legacy_context,
            },
            "r1_replay": {
                "run_id": replay.run_id,
                "should_execute": replay.should_execute,
                "execution_revision": replay_context["execution_revision"],
                "predecessor_failed_run_id": replay_context["predecessor_failed_run_id"],
                "status": completed.status,
            },
            "immutable_predecessor": {
                "canonical_bytes_before_utf8": predecessor_before_bytes,
                "canonical_bytes_after_utf8": predecessor_after_bytes,
                "before_sha256": hashlib.sha256(predecessor_before_bytes.encode()).hexdigest(),
                "after_sha256": hashlib.sha256(predecessor_after_bytes.encode()).hexdigest(),
                "byte_identical": predecessor_before_bytes == predecessor_after_bytes,
            },
            "r2_after_success": {
                "should_execute": r2.should_execute,
                "selected_run_id": r2.run_id,
                "selected_successful_replay": r2.run_id == replay.run_id,
                "semantic_run_count": len(rows),
            },
        }
        assert report["legacy_failed"]["status"] == "failed"
        assert report["legacy_failed"]["execution_revision_present"] is False
        assert report["r1_replay"]["should_execute"] is True
        assert report["r1_replay"]["predecessor_failed_run_id"] == legacy.run_id
        assert report["immutable_predecessor"]["byte_identical"] is True
        assert report["r1_replay"]["status"] == "succeeded"
        assert report["r2_after_success"] == {
            "should_execute": False,
            "selected_run_id": replay.run_id,
            "selected_successful_replay": True,
            "semantic_run_count": 2,
        }
        return report
    finally:
        connection.close()


def _successful_provenance(path: Path, revision: str) -> dict:
    connection, store, kernel, case_id = _scratch(path)
    try:
        claim = _reserve(
            kernel,
            case_id,
            revision=revision,
            at="2026-08-25T01:00:00Z",
        )
        evidence, fact = _success_evidence()
        result = kernel.complete_run(
            run_id=claim.run_id,
            evidence=(evidence,),
            facts=(fact,),
            blockers=(),
            decision_tier="verified_automatic",
            action_readiness="transition_eligible",
            retry_at=None,
            diagnostics={"sec_attempts": 1},
            at="2026-08-25T02:00:00Z",
        )
        return {
            "execution_revision": revision,
            "run_id": claim.run_id,
            "decision_provenance_sha256": result.decision_provenance_sha256,
            "status": store.get_automation_run(claim.run_id)["status"],
        }
    finally:
        connection.close()


def _cross_revision_due_blocked_retry(path: Path) -> dict:
    from src.security_lifecycle_fact_kernel import AutomationBlocker

    connection, store, kernel, case_id = _scratch(path)
    try:
        initial = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r0",
            at="2026-08-25T00:00:00Z",
        )
        kernel.complete_run(
            run_id=initial.run_id,
            evidence=(),
            facts=(),
            blockers=(
                AutomationBlocker(
                    code="sec_transport_unavailable",
                    retryable=True,
                    context={"attempts": 1},
                ),
            ),
            decision_tier=None,
            action_readiness=None,
            retry_at="2026-08-26T00:00:00Z",
            diagnostics={"sec_attempts": 1},
            at="2026-08-25T01:00:00Z",
        )
        retry = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r1",
            at="2026-08-26T00:00:00Z",
        )
        retry_context = json.loads(
            store.get_automation_run(retry.run_id)["query_context_json"]
        )
        kernel.fail_run(
            run_id=retry.run_id,
            failure_code="persistence_failed",
            diagnostics={"persist_failures": 1},
            at="2026-08-26T01:00:00Z",
        )
        repeated = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r1",
            at="2026-08-27T00:00:00Z",
        )
        report = {
            "initial_execution_revision": retry_context["execution_revision"],
            "latest_attempt_execution_revision": retry_context[
                "latest_attempt_execution_revision"
            ],
            "ordinary_retry_reused_run": retry.run_id == initial.run_id,
            "same_revision_replay": repeated.should_execute,
            "run_count": len(store.list_automation_runs(case_id)),
        }
        assert report == {
            "initial_execution_revision": "trusted-lifecycle-execution-r0",
            "latest_attempt_execution_revision": "trusted-lifecycle-execution-r1",
            "ordinary_retry_reused_run": True,
            "same_revision_replay": False,
            "run_count": 1,
        }
        return report
    finally:
        connection.close()


def _deadline_owner(*, deadline: str, at: str):
    from src.security_lifecycle_sec_evidence import SecSourceDeadline
    from src.service import security_lifecycle_automation_scheduler as scheduler
    from tests.test_security_lifecycle_fact_kernel import _evidence

    month_day = "April 1, 2026" if deadline == "2026-04-01" else "September 1, 2026"
    cited_text = (
        "HAPN merger agreement may be terminated if the merger is not consummated by "
        f"{month_day}."
    )
    evidence = _evidence(f"deadline-{deadline}", excerpt=cited_text)
    cited = cited_text.encode("utf-8")
    source_deadline = SecSourceDeadline(
        date=deadline,
        evidence_id=evidence.evidence_id,
        span_start_byte=0,
        span_end_byte=len(cited),
        cited_text=cited_text,
        cited_text_sha256=hashlib.sha256(cited).hexdigest(),
        rule_id="sec.explicit_transaction_termination_date",
        rule_version="4",
    )
    blocker = scheduler._pending_event_monitoring(
        {"observation": {"kinds": [{"event_type": "merger_agreement", "effective_date": None}]}},
        (),
        source_family_results={
            "regulator": "available",
            "market_infrastructure": "available",
            "publisher": "available",
        },
        source_deadlines=(source_deadline,),
        at=at,
    )
    assert blocker is not None
    return evidence, blocker


def _persist_deadline(path: Path, *, deadline: str, at: str, fingerprint: str) -> tuple[dict, dict]:
    connection, store, kernel, case_id = _scratch(path)
    try:
        claim = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r1",
            at=at,
            fingerprint=fingerprint,
        )
        evidence, blocker = _deadline_owner(deadline=deadline, at=at)
        retry_at = blocker.context.get("next_check_at") if blocker.retryable else None
        result = kernel.complete_run(
            run_id=claim.run_id,
            evidence=(evidence,),
            facts=(),
            blockers=(blocker,),
            decision_tier=None,
            action_readiness=None,
            retry_at=retry_at,
            diagnostics={"sec_attempts": 1},
            at=at,
        )
        stored = store.get_automation_run(claim.run_id)
        context = json.loads(stored["blockers"][0]["context_json"])
        report = {
            "status": result.status,
            "monitoring_reason": context["monitoring_reason"],
            "source_deadline": context["source_deadline"],
            "completed_check_as_of": context.get("as_of"),
            "producer_evidence_id": evidence.evidence_id,
            "persisted_evidence_id": context["source_deadline_evidence_id"],
            "citation_sha256": context["source_deadline_cited_text_sha256"],
            "rule_id": context["source_deadline_rule_id"],
            "rule_version": context["source_deadline_rule_version"],
        }
        return report, stored
    finally:
        connection.close()


def _forged_rollback(path: Path) -> dict:
    from tests.test_security_lifecycle_fact_kernel import _fact

    connection, store, kernel, case_id = _scratch(path)
    try:
        claim = _reserve(
            kernel,
            case_id,
            revision="trusted-lifecycle-execution-r1",
            at="2026-08-27T00:00:00Z",
            fingerprint="c" * 64,
        )
        evidence, blocker = _deadline_owner(
            deadline="2026-04-01",
            at="2026-08-27T00:00:00Z",
        )
        context = dict(blocker.context)
        context["source_deadline_cited_text_sha256"] = "f" * 64
        error = None
        try:
            kernel.complete_run(
                run_id=claim.run_id,
                evidence=(evidence,),
                facts=(_fact(evidence),),
                blockers=(replace(blocker, context=context),),
                decision_tier=None,
                action_readiness=None,
                retry_at=None,
                diagnostics={"sec_attempts": 1},
                at="2026-08-27T00:00:00Z",
            )
        except ValueError as exc:
            error = str(exc)
        counts = {
            "evidence_rows": connection.execute(
                "SELECT COUNT(*) FROM security_lifecycle_evidence WHERE automation_run_id=?",
                (claim.run_id,),
            ).fetchone()[0],
            "fact_rows": connection.execute(
                "SELECT COUNT(*) FROM security_lifecycle_automation_facts WHERE automation_run_id=?",
                (claim.run_id,),
            ).fetchone()[0],
            "blocker_rows": connection.execute(
                "SELECT COUNT(*) FROM security_lifecycle_automation_run_blockers WHERE automation_run_id=?",
                (claim.run_id,),
            ).fetchone()[0],
        }
        report = {
            "error": error,
            "run_status_after_rejection": store.get_automation_run(claim.run_id)["status"],
            **counts,
        }
        assert report == {
            "error": "blocker_citation",
            "run_status_after_rejection": "running",
            "evidence_rows": 0,
            "fact_rows": 0,
            "blocker_rows": 0,
        }
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from src.security_lifecycle_disposition import project_lifecycle_disposition
    from tests.test_security_lifecycle_disposition import _case

    observer_calibration = _calibrate_authority_call_observer()
    with (
        TemporaryDirectory(prefix="arkscope-honesty-authority-") as temp,
        _observe_forbidden_authority_calls() as authority_calls,
    ):
        root = Path(temp)
        replay = _capture_replay(root / "replay.sqlite")
        due_retry = _cross_revision_due_blocked_retry(
            root / "cross-revision-due-retry.sqlite"
        )
        r0 = _successful_provenance(root / "provenance-r0.sqlite", "trusted-lifecycle-execution-r0")
        r1 = _successful_provenance(root / "provenance-r1.sqlite", "trusted-lifecycle-execution-r1")
        pre_deadline, _ = _persist_deadline(
            root / "citation-pre.sqlite",
            deadline="2026-09-01",
            at="2026-08-27T00:00:00Z",
            fingerprint="b" * 64,
        )
        final_deadline, final_run = _persist_deadline(
            root / "citation-final.sqlite",
            deadline="2026-04-01",
            at="2026-08-27T12:00:00Z",
            fingerprint="d" * 64,
        )
        projection = project_lifecycle_disposition(
            _case(automation_runs=(final_run,))
        )
        forged = _forged_rollback(root / "citation-forged.sqlite")

        provenance = {
            "r0": r0,
            "r1": r1,
            "equal": r0["decision_provenance_sha256"] == r1["decision_provenance_sha256"],
        }
        projected = {
            "disposition": projection.disposition,
            "queue_bucket": projection.queue_bucket,
            "reason_code": projection.reason_code,
            "disposition_as_of": projection.disposition_as_of,
            "next_check_at": projection.next_check_at,
        }
        transition_and_acknowledgement_calls = dict(authority_calls)
        payload = {
            "schema_version": 1,
            "product_test_authority": PRODUCT_TEST_AUTHORITY,
            "authority": AUTHORITY,
            "scratch_database_kind": "temporary_sqlite_only",
            "legacy_failed_replay": replay,
            "cross_revision_due_blocked_retry": due_retry,
            "decision_provenance": provenance,
            "producer_to_kernel_citations": {
                "pre_deadline": pre_deadline,
                "final": final_deadline,
            },
            "forged_citation_rollback": forged,
            "final_unconfirmed_projection": projected,
            "authority_call_observer_calibration": observer_calibration,
            "transition_and_acknowledgement_calls": transition_and_acknowledgement_calls,
        }
        assert provenance["equal"] is True
        assert pre_deadline["monitoring_reason"] == "event_completion_not_confirmed"
        assert final_deadline["source_deadline"] == "2026-04-01"
        assert final_deadline["completed_check_as_of"] == "2026-08-27"
        assert projected == {
            "disposition": "not_confirmed_yet",
            "queue_bucket": "history",
            "reason_code": "not_confirmed_as_of",
            "disposition_as_of": "2026-08-27",
            "next_check_at": None,
        }
        assert all(value == 0 for value in transition_and_acknowledgement_calls.values())

    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
