"""Capture v2 replay and stale-transition authority entirely in scratch DBs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _v2_replay(root: Path) -> dict:
    import src.security_lifecycle_automation_worker as worker_module
    import src.security_lifecycle_decision_policy as policy_module
    from tests.test_security_lifecycle_automation_worker import (
        _Harness,
        _bundle,
        _case,
        _store,
    )

    root.mkdir(parents=True, exist_ok=True)
    original_policy = policy_module.AUTOMATION_POLICY_VERSION
    original_worker = worker_module.AUTOMATION_POLICY_VERSION
    case = _case(1)
    harness = _Harness(root, [case])
    harness.bundles[case["case_id"]] = _bundle(case, review_structure="stock")
    try:
        policy_module.AUTOMATION_POLICY_VERSION = "trusted-lifecycle-automation-v2"
        worker_module.AUTOMATION_POLICY_VERSION = "trusted-lifecycle-automation-v2"
        first = harness.worker().run()
        store = _store(harness)
        before_runs = store.list_automation_runs(case["case_id"])
        before_assessments = store.list_assessments(case["case_id"])
        old_run_id = before_runs[0]["run_id"]
        old_assessment_id = before_assessments[0]["assessment_id"]
        old_assessment_before = store.get_assessment(old_assessment_id)
        before = {
            "worker_result": {
                key: value for key, value in first.items() if key != "case_ids"
            },
            "runs": [
                {
                    "policy_version": row["policy_version"],
                    "status": row["status"],
                }
                for row in before_runs
            ],
            "assessments": [
                {
                    "status": row["status"],
                    "run_policy_version": before_runs[0]["policy_version"],
                }
                for row in before_assessments
            ],
        }

        policy_module.AUTOMATION_POLICY_VERSION = "trusted-lifecycle-automation-v3"
        worker_module.AUTOMATION_POLICY_VERSION = "trusted-lifecycle-automation-v3"
        second = harness.worker().run()
        after_runs = store.list_automation_runs(case["case_id"])
        after_assessments = store.list_assessments(case["case_id"])
        old_projection = next(
            row
            for row in store.project_case_state(
                case["case_id"],
                observation_fingerprint_sha256=case[
                    "observation_fingerprint_sha256"
                ],
            )["assessment_history"]
            if row["assessment_id"] == old_assessment_id
        )
        policy_by_run = {
            row["run_id"]: row["policy_version"] for row in after_runs
        }
        old_assessment_after = store.get_assessment(old_assessment_id)
        after = {
            "worker_result": {
                key: value for key, value in second.items() if key != "case_ids"
            },
            "runs": [
                {
                    "policy_version": row["policy_version"],
                    "status": row["status"],
                }
                for row in after_runs
            ],
            "assessments": [
                {
                    "status": row["status"],
                    "run_policy_version": policy_by_run[row["automation_run_id"]],
                }
                for row in after_assessments
            ],
            "old_assessment_projection_stale": old_projection["stale"],
            "old_assessment_storage_preserved_exactly": (
                old_assessment_before == old_assessment_after
            ),
        }
        assert len(before["runs"]) == 1
        assert before["runs"][0]["policy_version"].endswith("v2")
        assert len(after["runs"]) == 2
        assert after["runs"][0]["policy_version"].endswith("v3")
        assert any(row["run_id"] == old_run_id for row in after_runs)
        assert any(
            row["assessment_id"] == old_assessment_id and row["status"] == "draft"
            for row in after_assessments
        )
        assert after["old_assessment_projection_stale"] is True
        assert after["old_assessment_storage_preserved_exactly"] is True
        return {"before": before, "after": after}
    finally:
        policy_module.AUTOMATION_POLICY_VERSION = original_policy
        worker_module.AUTOMATION_POLICY_VERSION = original_worker
        harness.conn.close()


def _stale_transition(root: Path) -> dict:
    from src.ticker_identity_transition import TickerIdentityTransitionStore
    from tests.test_ticker_identity_transition import (
        _build,
        _id_factory,
        _profile_owned_rows,
        _seed_automation_authority,
        _seed_transferable_state,
        _transition_connection,
    )

    root.mkdir(parents=True, exist_ok=True)
    connection = _transition_connection(root)
    try:
        _seed_transferable_state(connection)
        _seed_automation_authority(connection)
        preview = _build(connection, sources=("manual_lists",))
        store = TickerIdentityTransitionStore(
            connection,
            id_factory=_id_factory(),
            clock=lambda: "2026-08-25T13:00:00Z",
        )
        transition = store.approve_automation(
            preview=preview,
            approved_preview_sha256=preview["preview_sha256"],
        )
        connection.execute(
            "UPDATE ticker_identity_transitions SET automation_policy_version=? "
            "WHERE transition_id=?",
            ("trusted-lifecycle-automation-v2", transition["transition_id"]),
        )
        connection.commit()
        before = _profile_owned_rows(connection)
        result = store.apply(
            transition["transition_id"],
            current_preview=preview,
            expected_preview_sha256=preview["preview_sha256"],
            trigger="scheduler",
        )
        after = _profile_owned_rows(connection)
        report = {
            "stored_policy_version": "trusted-lifecycle-automation-v2",
            "current_policy_version": "trusted-lifecycle-automation-v3",
            "apply_status": result["status"],
            "block_reasons": result["block_reasons"],
            "transition_status": result["transition"]["status"],
            "profile_owned_rows_before_sha256": _digest(before),
            "profile_owned_rows_after_sha256": _digest(after),
            "profile_state_drift": before != after,
        }
        assert report["apply_status"] == "blocked"
        assert report["block_reasons"] == ["preview_changed"]
        assert report["transition_status"] == "needs_review"
        assert report["profile_state_drift"] is False
        return report
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="arkscope-disposition-authority-") as temp:
        root = Path(temp)
        payload = {
            "scope": "offline_fixture_and_scratch_only",
            "provider_calls": 0,
            "production_database_reads": 0,
            "production_database_writes": 0,
            "production_database_migrations": 0,
            "production_database_backups": 0,
            "production_database_restores": 0,
            "v2_replay": _v2_replay(root / "replay"),
            "stale_v2_transition_apply": _stale_transition(root / "transition"),
        }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
