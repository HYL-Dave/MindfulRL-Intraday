"""Run the lifecycle control-plane reverse mutations independently."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]
WEB = ROOT / "apps/arkscope-web"
VITEST = ROOT / "node_modules/.bin/vitest"
TOKEN_SHAPE = re.compile(
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
)


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    task: int
    description: str
    path: str
    old: str
    new: str
    command: tuple[str, ...]
    owner_needles: tuple[str, ...]
    cwd: str = "."
    extra_replacements: tuple[tuple[str, str], ...] = ()


def py_test(node: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node)


def web_test(path: str, title: str) -> tuple[str, ...]:
    return (str(VITEST), "run", path, "--reporter=dot", "-t", title)


RUNTIME = "src/service/security_lifecycle_automation_runtime.py"
WORKER = "src/security_lifecycle_automation_worker.py"
KERNEL = "src/security_lifecycle_fact_kernel.py"
SCHEDULER = "src/service/security_lifecycle_automation_scheduler.py"
IBKR = "src/security_lifecycle_ibkr_evidence.py"
SEC = "src/security_lifecycle_sec_evidence.py"
CONFIG = "src/service/security_lifecycle_automation_config.py"
DATA_SCHEDULER = "src/service/data_scheduler.py"
TICKER_SCHEDULER = "src/service/ticker_identity_scheduler.py"
LIFECYCLE_ROUTES = "src/api/routes/security_lifecycle.py"
API = "apps/arkscope-web/src/api.ts"
LIFECYCLE_VIEW = "apps/arkscope-web/src/lifecycle/LifecycleView.tsx"
SETTINGS = "apps/arkscope-web/src/settings/DataStorageSection.tsx"
STYLES = "apps/arkscope-web/src/styles.css"
LISTING_TRANSPORT = "data_sources/listing_authority_transport.py"
NEWS = "src/security_lifecycle_news_evidence.py"


MUTATIONS = (
    Mutation(
        "M01", 1, "replace the exclusive process lock with a shared lock",
        RUNTIME,
        "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)",
        py_test("tests/test_security_lifecycle_automation_runtime.py::test_execution_lock_is_exclusive_and_issues_a_bounded_owner_id"),
        ("test_execution_lock_is_exclusive_and_issues_a_bounded_owner_id",),
    ),
    Mutation(
        "M02", 1, "reject the legitimate conditional IBKR-stage skip",
        RUNTIME,
        '_CONDITIONAL_STAGES = frozenset({"ibkr", "approve"})',
        '_CONDITIONAL_STAGES = frozenset({"approve"})',
        py_test("tests/test_security_lifecycle_automation_runtime.py::test_progress_registry_records_only_explicit_conditional_stage_skips"),
        ("test_progress_registry_records_only_explicit_conditional_stage_skips",),
    ),
    Mutation(
        "M03", 2, "reject a valid human-accepted automation assessment",
        WORKER,
        'in {"automation_policy", "human"}',
        'in {"automation_policy"}',
        py_test("tests/test_security_lifecycle_automation_worker.py::test_human_accepted_automation_assessment_completes_pending_finalization"),
        ("test_human_accepted_automation_assessment_completes_pending_finalization",),
    ),
    Mutation(
        "M04", 2, "bypass the profile-mutation authority at approval",
        WORKER,
        "                and decision.transition_requested\n                and self._may_mutate_profile()\n            )",
        "                and decision.transition_requested\n            )",
        py_test("tests/test_security_lifecycle_automation_worker.py::test_transition_approval_rereads_mutation_authority_at_boundary"),
        ("test_transition_approval_rereads_mutation_authority_at_boundary",),
        extra_replacements=((
            "if approval_stage_started and self._may_mutate_profile():",
            "if approval_stage_started:",
        ),),
    ),
    Mutation(
        "M05", 2, "bypass terminal-finalization retry backoff",
        KERNEL,
        "retry_not_before is None or _instant(timestamp) < _instant(\n                            str(retry_not_before)\n                        )",
        "retry_not_before is None or _instant(timestamp) >= _instant(\n                            str(retry_not_before)\n                        )",
        py_test("tests/test_security_lifecycle_automation_worker.py::test_terminal_finalization_failure_uses_bounded_backoff_without_hot_loop"),
        ("test_terminal_finalization_failure_uses_bounded_backoff_without_hot_loop",),
    ),
    Mutation(
        "M06", 3, "downgrade worker case outcomes to result version one",
        WORKER,
        '            "result_version": 2,\n            "case_ids": [],',
        '            "result_version": 1,\n            "case_ids": [],',
        py_test("tests/test_security_lifecycle_automation_worker.py::test_current_assessment_is_not_reprocessed"),
        ("test_current_assessment_is_not_reprocessed",),
    ),
    Mutation(
        "M07", 3, "accept counter and per-case outcome drift",
        SCHEDULER,
        "            if counts[outcome] != sum(\n                value == outcome for value in case_outcomes.values()\n            ):\n                raise ValueError(\"case_outcomes\")",
        "            if False and counts[outcome] != sum(\n                value == outcome for value in case_outcomes.values()\n            ):\n                raise ValueError(\"case_outcomes\")",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_bounded_v2_result_rejects_counter_and_outcome_drift"),
        ("test_bounded_v2_result_rejects_counter_and_outcome_drift",),
    ),
    Mutation(
        "M08", 3, "treat the unchanged failed run as recovered",
        SCHEDULER,
        "    if latest_run_id == baseline_run_id:\n        return True",
        "    if latest_run_id == baseline_run_id:\n        return False",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_failed_case_skipped_on_next_real_tick_does_not_record_recovery"),
        ("test_failed_case_skipped_on_next_real_tick_does_not_record_recovery",),
    ),
    Mutation(
        "M09", 4, "enable due failed replay without explicit authority",
        KERNEL,
        "        allow_due_failed_retry: bool = False,\n        allow_new_attempt: bool = False,",
        "        allow_due_failed_retry: bool = True,\n        allow_new_attempt: bool = False,",
        py_test("tests/test_security_lifecycle_fact_kernel.py::test_due_failed_retry_requires_explicit_authority_and_preserves_predecessor"),
        ("test_due_failed_retry_requires_explicit_authority_and_preserves_predecessor",),
    ),
    Mutation(
        "M10", 4, "enable attended attempts without explicit authority",
        KERNEL,
        "        allow_due_failed_retry: bool = False,\n        allow_new_attempt: bool = False,",
        "        allow_due_failed_retry: bool = False,\n        allow_new_attempt: bool = True,",
        py_test("tests/test_security_lifecycle_fact_kernel.py::test_attended_new_attempt_preserves_each_terminal_predecessor"),
        ("test_attended_new_attempt_preserves_each_terminal_predecessor",),
    ),
    Mutation(
        "M11", 4, "remove predecessor-cycle detection",
        KERNEL,
        "        if predecessor_run_id is not None and predecessor_run_id in visited:\n"
        "            raise _AttemptChainDataError(\"automation_predecessor_cycle\")",
        "        if False and predecessor_run_id is not None and predecessor_run_id in visited:\n"
        "            raise _AttemptChainDataError(\"automation_predecessor_cycle\")",
        py_test("tests/test_security_lifecycle_fact_kernel.py::test_predecessor_cycle_fails_closed_before_creating_attended_attempt"),
        ("test_predecessor_cycle_fails_closed_before_creating_attended_attempt",),
        extra_replacements=((
            "        if current_id in visited:\n"
            "            raise _AttemptChainDataError(\"automation_predecessor_cycle\")",
            "        if False and current_id in visited:\n"
            "            raise _AttemptChainDataError(\"automation_predecessor_cycle\")",
        ),),
    ),
    Mutation(
        "M12", 5, "truncate an over-budget IBKR identity plan",
        IBKR,
        "    if len(queries) > max_queries:\n"
        "        return _blocked(\n"
        "            \"market_confirmation_missing\",\n"
        "            requests_made=0,\n"
        "            context={\n"
        "                \"code\": \"candidate_budget_exceeded\",\n"
        "                \"candidate_count\": len(queries),\n"
        "                \"query_limit\": max_queries,\n"
        "            },\n"
        "        )",
        "    if len(queries) > max_queries:\n"
        "        queries = queries[:max_queries]",
        py_test("tests/test_security_lifecycle_ibkr_evidence.py::test_budget_overflow_records_candidate_budget_exceeded_context"),
        ("test_budget_overflow_records_candidate_budget_exceeded_context",),
    ),
    Mutation(
        "M13", 5, "place aliases before the current ticker",
        IBKR,
        "        context.current_ticker,\n        *candidate_tickers,\n        *context.ticker_aliases,",
        "        *context.ticker_aliases,\n        context.current_ticker,\n        *candidate_tickers,",
        py_test("tests/test_security_lifecycle_ibkr_evidence.py::test_ibkr_candidate_plan_prioritizes_exact_current_successor_then_aliases"),
        ("test_ibkr_candidate_plan_prioritizes_exact_current_successor_then_aliases",),
    ),
    Mutation(
        "M14", 5, "raise the scheduler alias bound without an owner-visible ambiguity",
        SCHEDULER,
        "_MAX_ALIASES_PER_TICKER = 64",
        "_MAX_ALIASES_PER_TICKER = 1024",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_alias_closure_overflow_is_a_per_case_ibkr_ambiguity"),
        ("test_alias_closure_overflow_is_a_per_case_ibkr_ambiguity",),
    ),
    Mutation(
        "M15", 6, "retain the first duplicate deadline row instead of its latest citation",
        SEC,
        "        if edge in seen_edges:\n            if active is not None and active.date == row.date:\n                active = row\n            continue",
        "        if edge in seen_edges:\n            if active is not None and active.date == row.date:\n                active = active or row\n            continue",
        py_test("tests/test_security_lifecycle_sec_evidence.py::test_repeated_explicit_deadline_extension_is_idempotent_and_selects_latest_row"),
        ("test_repeated_explicit_deadline_extension_is_idempotent_and_selects_latest_row",),
    ),
    Mutation(
        "M16", 6, "weaken forward deadline chronology",
        SEC,
        "if date.fromisoformat(row.date) <= date.fromisoformat(predecessor):",
        "if date.fromisoformat(row.date) == date.fromisoformat(predecessor):",
        py_test("tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_cannot_move_backward_in_time"),
        ("test_deadline_extension_cannot_move_backward_in_time",),
    ),
    Mutation(
        "M17", 7, "reuse SEC material across observation fingerprints",
        SCHEDULER,
        "    if prior_material.observation_fingerprint_sha256 != str(\n        case.get(\"observation_fingerprint_sha256\") or \"\"\n    ):\n        return None",
        "    if False and prior_material.observation_fingerprint_sha256 != str(\n        case.get(\"observation_fingerprint_sha256\") or \"\"\n    ):\n        return None",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_each_closed_sec_reuse_predicate_independently_forces_refresh"),
        ("test_each_closed_sec_reuse_predicate_independently_forces_refresh",),
    ),
    Mutation(
        "M18", 7, "allow a second source-payload automatic retry",
        KERNEL,
        '    "source_payload_invalid": (timedelta(hours=1),),',
        '    "source_payload_invalid": (timedelta(hours=1), timedelta(hours=2)),',
        py_test("tests/test_security_lifecycle_automation_worker.py::test_source_payload_invalid_receives_exactly_one_automatic_retry"),
        ("test_source_payload_invalid_receives_exactly_one_automatic_retry",),
    ),
    Mutation(
        "M19", 8, "enable profile transitions by default",
        CONFIG,
        "    apply_profile_transitions=False,",
        "    apply_profile_transitions=True,",
        py_test("tests/test_security_lifecycle_automation_config.py::test_absent_settings_resolve_to_the_complete_safe_default"),
        ("test_absent_settings_resolve_to_the_complete_safe_default",),
    ),
    Mutation(
        "M20", 8, "grant the transition scheduler unconditional automation authority",
        DATA_SCHEDULER,
        "            allow_automation_approved=transition_mutation_allowed(),",
        "            allow_automation_approved=True,",
        py_test("tests/test_data_scheduler.py::test_malformed_lifecycle_config_disables_analysis_and_mutation"),
        ("test_malformed_lifecycle_config_disables_analysis_and_mutation",),
    ),
    Mutation(
        "M21", 9, "swap the real listing and IBKR stage order",
        RUNTIME,
        'LIFECYCLE_AUTOMATION_STAGE_ORDER: tuple[LifecycleAutomationStage, ...] = (\n    "preparing",\n    "sec",\n    "listing",\n    "ibkr",\n    "evaluate",',
        'LIFECYCLE_AUTOMATION_STAGE_ORDER: tuple[LifecycleAutomationStage, ...] = (\n    "preparing",\n    "sec",\n    "ibkr",\n    "listing",\n    "evaluate",',
        py_test("tests/test_security_lifecycle_automation_runtime.py::test_progress_registry_follows_the_exact_real_stage_order_and_finishes_cleanly"),
        ("test_progress_registry_follows_the_exact_real_stage_order_and_finishes_cleanly",),
    ),
    Mutation(
        "M22", 9, "persist a skipped IBKR stage as completed",
        RUNTIME,
        "                skipped_stages=current.skipped_stages + expected_skips,",
        "                skipped_stages=current.skipped_stages,",
        py_test("tests/test_security_lifecycle_automation_runtime.py::test_progress_registry_records_only_explicit_conditional_stage_skips"),
        ("test_progress_registry_records_only_explicit_conditional_stage_skips",),
    ),
    Mutation(
        "M23", 10, "accept an unknown frontend progress stage",
        API,
        "          : automationEnum(detail.current_stage, AUTOMATION_STAGES),",
        "          : automationString(detail.current_stage) as SecurityLifecycleAutomationStage,",
        web_test("src/LifecycleAutomationApi.test.ts", "rejects an unknown progress stage"),
        ("rejects an unknown progress stage",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M24", 10, "allow a stale automation-status response to commit",
        LIFECYCLE_VIEW,
        "      if (sequence !== automationStatusRequestRef.current) return null;\n      setAutomationStatusSnapshot({ response, sequence });",
        "      if (false && sequence !== automationStatusRequestRef.current) return null;\n      setAutomationStatusSnapshot({ response, sequence });",
        web_test("src/lifecycle/LifecycleView.test.tsx", "keeps automation status bound to the newest request"),
        ("keeps automation status bound to the newest request",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M25", 10, "refresh the original case after an attended run",
        LIFECYCLE_VIEW,
        "    const runSequence = pendingAutomationRun.runSequence;\n    setPendingAutomationRun(null);\n    setBusy((current) => current === \"automation-case\" ? null : current);\n    void (async () => {\n      const currentCaseId = selectedCaseIdRef.current;",
        "    const runSequence = pendingAutomationRun.runSequence;\n    const originalCaseId = pendingAutomationRun.caseId;\n    setPendingAutomationRun(null);\n    setBusy((current) => current === \"automation-case\" ? null : current);\n    void (async () => {\n      const currentCaseId = originalCaseId;",
        web_test("src/lifecycle/LifecycleView.test.tsx", "refreshes only the latest selected case after an attended run completes"),
        ("refreshes only the latest selected case after an attended run completes",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M26", 10, "let stale success outrank an active Settings incident",
        SETTINGS,
        "  const automationState = schedulerIncident || incidentCaseCount > 0\n    ? \"incident\"",
        "  const automationState = false\n    ? \"incident\"",
        web_test("src/settings/DataStorageSection.test.tsx", "prioritizes an active incident over a stale successful result"),
        ("prioritizes an active incident over a stale successful result",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M27", 11, "ignore the per-instance Massive request maximum",
        LISTING_TRANSPORT,
        "if self.massive_request_count >= self.max_massive_requests:",
        "if self.massive_request_count >= MAX_MASSIVE_REQUESTS:",
        py_test("tests/test_listing_authority_transport.py::test_listing_budget_accepts_tighter_per_instance_request_limits"),
        ("test_listing_budget_accepts_tighter_per_instance_request_limits",),
    ),
    Mutation(
        "M28", 11, "expand the canary case limit to production size",
        SCHEDULER,
        "        return cls(\n            case_limit=1,",
        "        return cls(\n            case_limit=2,",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_canary_case_limit_is_enforced_before_execution"),
        ("test_canary_case_limit_is_enforced_before_execution",),
    ),
    Mutation(
        "M29", 11, "use production SEC limits inside a canary run",
        SCHEDULER,
        "        budget = execution_limits.sec_budget()",
        "        budget = SecRequestBudget.lifecycle()",
        py_test("tests/test_security_lifecycle_automation_scheduler.py::test_canary_limits_cross_real_sec_listing_and_ibkr_scheduler_boundaries"),
        ("test_canary_limits_cross_real_sec_listing_and_ibkr_scheduler_boundaries",),
    ),
    Mutation(
        "M30", 11, "reactivate the retired publisher acquisition adapter",
        NEWS,
        'ACQUISITION_ADAPTER_STATUS = "retired"',
        'ACQUISITION_ADAPTER_STATUS = "active"',
        py_test("tests/test_security_lifecycle_news_evidence.py::test_publisher_acquisition_adapter_is_retired_and_has_no_product_caller"),
        ("test_publisher_acquisition_adapter_is_retired_and_has_no_product_caller",),
    ),
    Mutation(
        "M31", 12, "render closed manual evidence commands",
        STYLES,
        ".lifecycle-manual-supplement:not([open]) > .lifecycle-commands {\n  display: none;\n}",
        ".lifecycle-manual-supplement:not([open]) > .lifecycle-commands {\n  display: grid;\n}",
        web_test("src/LifecycleCss.test.ts", "keeps closed manual evidence commands out of layout"),
        ("keeps closed manual evidence commands out of layout",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M32", 12, "render closed evidence bodies",
        STYLES,
        ".lifecycle-evidence-item:not([open]) > .lifecycle-evidence-body {\n  display: none;\n}",
        ".lifecycle-evidence-item:not([open]) > .lifecycle-evidence-body {\n  display: block;\n}",
        web_test("src/LifecycleCss.test.ts", "keeps closed evidence bodies out of layout"),
        ("keeps closed evidence bodies out of layout",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M33", 7, "trust an unversioned legacy recovery as an incident boundary",
        TICKER_SCHEDULER,
        "    for row in rows:\n"
        "        if (\n"
        "            _stored_incident_reconciliation_version(row[\"result\"])\n"
        "            == _INCIDENT_RECONCILIATION_VERSION\n"
        "        ):\n"
        "            return int(row[\"id\"])",
        "    for row in rows:\n"
        "        if True:\n"
        "            return int(row[\"id\"])",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[absent-version]"
        ),
        (
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[absent-version]",
        ),
    ),
    Mutation(
        "M34", 7, "treat a missing failed transition row as settled",
        TICKER_SCHEDULER,
        '        if statuses.get(transition_id) in {None, "approved"}',
        '        if statuses.get(transition_id) == "approved"',
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_missing_failed_transition_row_never_proves_recovery"
        ),
        ("test_missing_failed_transition_row_never_proves_recovery",),
    ),
    Mutation(
        "M35", 7, "drop the incident reconciliation version marker",
        TICKER_SCHEDULER,
        "    if not failed:\n"
        "        stored[\"incident_reconciliation_version\"] = (\n"
        "            _INCIDENT_RECONCILIATION_VERSION\n"
        "        )",
        "    if False and not failed:\n"
        "        stored[\"incident_reconciliation_version\"] = (\n"
        "            _INCIDENT_RECONCILIATION_VERSION\n"
        "        )",
        py_test(
            "tests/test_data_scheduler.py::"
            "test_tick_deduplicates_ticker_transition_failure_and_records_recovery"
        ),
        ("test_tick_deduplicates_ticker_transition_failure_and_records_recovery",),
    ),
    Mutation(
        "M36", 7, "drop unresolved legacy-incident restatement",
        TICKER_SCHEDULER,
        "            if latest_failed_ids.isdisjoint(unresolved_ids):",
        "            if False and latest_failed_ids.isdisjoint(unresolved_ids):",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[absent-version]"
        ),
        (
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[absent-version]",
        ),
    ),
    Mutation(
        "M37", 7, "reconcile only the latest failure witness",
        TICKER_SCHEDULER,
        "    failed_ids = {\n"
        "        str(transition_id)\n"
        "        for incident in incidents\n"
        "        for transition_id in incident[\"failed_transition_ids\"]\n"
        "    }",
        "    failed_ids = {\n"
        "        str(transition_id)\n"
        "        for incident in incidents[-1:]\n"
        "        for transition_id in incident[\"failed_transition_ids\"]\n"
        "    }",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_recovery_tracks_all_unresolved_transition_ids_across_failure_churn"
        ),
        ("test_recovery_tracks_all_unresolved_transition_ids_across_failure_churn",),
    ),
    Mutation(
        "M38", 7, "treat an approved failed transition as settled",
        TICKER_SCHEDULER,
        '        if statuses.get(transition_id) in {None, "approved"}',
        "        if statuses.get(transition_id) is None",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_recovery_tracks_all_unresolved_transition_ids_across_failure_churn"
        ),
        ("test_recovery_tracks_all_unresolved_transition_ids_across_failure_churn",),
    ),
    Mutation(
        "M39", 7, "restore the global recovery-eligible gate",
        TICKER_SCHEDULER,
        '        if status != "succeeded":\n            conn.commit()',
        '        if status != "succeeded" or not bounded["recovery_eligible"]:\n'
        "            conn.commit()",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_scheduler_wide_recovery_is_independent_of_automation_policy_filter"
        ),
        ("test_scheduler_wide_recovery_is_independent_of_automation_policy_filter",),
    ),
    Mutation(
        "M40", 7, "reintroduce the no-table success recovery fallback",
        TICKER_SCHEDULER,
        "        return tuple(sorted(failed_ids))",
        '        return () if result["recovery_eligible"] is True else tuple(sorted(failed_ids))',
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_policy_deferral_and_schema_absence_never_clear_named_failure"
        ),
        ("test_policy_deferral_and_schema_absence_never_clear_named_failure",),
    ),
    Mutation(
        "M41", 7, "trust any integer incident reconciliation version",
        TICKER_SCHEDULER,
        "        if (\n"
        "            _stored_incident_reconciliation_version(row[\"result\"])\n"
        "            == _INCIDENT_RECONCILIATION_VERSION\n"
        "        ):",
        "        if (\n"
        "            _stored_incident_reconciliation_version(row[\"result\"])\n"
        "            is not None\n"
        "        ):",
        py_test(
            "tests/test_ticker_identity_scheduler.py::"
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[unknown-version]"
        ),
        (
            "test_legacy_recovery_is_revalidated_before_becoming_an_incident_boundary"
            "[unknown-version]",
        ),
    ),
    Mutation(
        "M42", 10, "complete an attended run while durable status is still running",
        LIFECYCLE_VIEW,
        '      || automationStatusSnapshot.response.last_status === "running"\n',
        "",
        web_test(
            "src/lifecycle/LifecycleView.test.tsx",
            "keeps an attended run pending while durable status is running without progress",
        ),
        ("keeps an attended run pending while durable status is running without progress",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M43", 10, "ignore durable running status in Settings controls",
        SETTINGS,
        '  const automationRunning = automation?.last_status === "running"\n'
        "    || Boolean(automation?.current_progress.length);",
        "  const automationRunning = Boolean(automation?.current_progress.length);",
        web_test(
            "src/settings/DataStorageSection.test.tsx",
            "keeps write controls disabled while a started run has durable running status",
        ),
        ("keeps write controls disabled while a started run has durable running status",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M44", 2, "collapse a live mutation-authority read error to disabled",
        LIFECYCLE_ROUTES,
        "        except AutomationTransitionMutationAuthorityUnavailable:\n"
        "            raise\n"
        "        except Exception as exc:\n"
        "            raise AutomationTransitionMutationAuthorityUnavailable() from exc",
        "        except Exception:\n"
        "            return False",
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_security_lifecycle_routes.py::"
            "test_manual_run_live_mutation_gate_raises_typed_store_unavailable",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_route_mutation_authority_failure_is_retryable_before_finalization",
        ),
        (
            "test_manual_run_live_mutation_gate_raises_typed_store_unavailable",
            "test_route_mutation_authority_failure_is_retryable_before_finalization",
        ),
    ),
    Mutation(
        "M45", 10, "stop attended-run polling after one status request failure",
        LIFECYCLE_VIEW,
        "    automationStatusPollRevision,\n",
        "",
        web_test(
            "src/lifecycle/LifecycleView.test.tsx",
            "continues polling an attended run after one status request fails",
        ),
        ("continues polling an attended run after one status request fails",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M46", 10, "stop Settings polling after one durable-status request failure",
        SETTINGS,
        "  }, [automationRunning, automation, automationPollRevision, loadAutomation]);",
        "  }, [automationRunning, automation, loadAutomation]);",
        web_test(
            "src/settings/DataStorageSection.test.tsx",
            "continues polling a durable running status after one request fails",
        ),
        ("continues polling a durable running status after one request fails",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M47", 10, "stop Lifecycle polling for durable-only running state",
        LIFECYCLE_VIEW,
        '      && automationStatusSnapshot?.response.last_status !== "running"\n',
        "",
        web_test(
            "src/lifecycle/LifecycleView.test.tsx",
            "polls and exposes a durable run before in-memory progress arrives",
        ),
        ("polls and exposes a durable run before in-memory progress arrives",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M48", 10, "hide the durable-only global running banner",
        LIFECYCLE_VIEW,
        ') : automationStatus?.last_status === "running" ? (',
        ") : false ? (",
        web_test(
            "src/lifecycle/LifecycleView.test.tsx",
            "polls and exposes a durable run before in-memory progress arrives",
        ),
        ("polls and exposes a durable run before in-memory progress arrives",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M49", 10, "enable the case Run button during durable-only running state",
        LIFECYCLE_VIEW,
        '                      || automationStatus?.last_status === "running"\n',
        "",
        web_test(
            "src/lifecycle/LifecycleView.test.tsx",
            "polls and exposes a durable run before in-memory progress arrives",
        ),
        ("polls and exposes a durable run before in-memory progress arrives",),
        cwd="apps/arkscope-web",
    ),
    Mutation(
        "M50", 10, "prefer stale invalid telemetry over durable running state",
        SETTINGS,
        "  const stateKey: SecurityLifecycleAutomationSchedulerStatus | \"absent\" | \"invalid\" = (\n"
        "    currentProgress || automation?.last_status === \"running\"\n"
        "      ? \"running\"\n"
        "      : automation?.telemetry_status === \"invalid\"\n"
        "        ? \"invalid\"\n"
        "        : automation?.last_status ?? \"absent\"\n"
        "  );",
        "  const stateKey: SecurityLifecycleAutomationSchedulerStatus | \"absent\" | \"invalid\" = (\n"
        "    automation?.telemetry_status === \"invalid\"\n"
        "      ? \"invalid\"\n"
        "      : currentProgress || automation?.last_status === \"running\"\n"
        "        ? \"running\"\n"
        "        : automation?.last_status ?? \"absent\"\n"
        "  );",
        web_test(
            "src/settings/DataStorageSection.test.tsx",
            "shows durable running ahead of stale invalid telemetry",
        ),
        ("shows durable running ahead of stale invalid telemetry",),
        cwd="apps/arkscope-web",
    ),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def child_env() -> dict[str, str]:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in blocked)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def sanitize(value: str) -> str:
    return TOKEN_SHAPE.sub("[REDACTED]", value)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"replacement_count:{count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def clear_python_bytecode(path: Path) -> None:
    if path.suffix != ".py":
        return
    cache = path.parent / "__pycache__"
    if not cache.is_dir():
        return
    for candidate in cache.glob(f"{path.stem}.*.pyc"):
        candidate.unlink()


def run_command(mutation: Mutation) -> dict[str, object]:
    process = subprocess.run(
        mutation.command,
        cwd=ROOT / mutation.cwd,
        env=child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    output = sanitize(process.stdout)
    failure_lines = [
        line.strip()
        for line in output.splitlines()
        if line.lstrip().startswith(("FAILED ", "FAIL ", "AssertionError:"))
    ]
    return {
        "exit_code": process.returncode,
        "output_sha256": digest(output.encode("utf-8")),
        "failure_lines": failure_lines,
        "output_tail": output.splitlines()[-20:],
        "_output": output,
    }


def public_result(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "_output"}


def run_mutation(mutation: Mutation) -> dict[str, object]:
    path = ROOT / mutation.path
    original = path.read_bytes()
    baseline: dict[str, object] | None = None
    mutant: dict[str, object] | None = None
    error: str | None = None
    try:
        clear_python_bytecode(path)
        baseline = run_command(mutation)
        if baseline["exit_code"] != 0:
            raise RuntimeError("baseline_failed")
        replace_once(path, mutation.old, mutation.new)
        for old, new in mutation.extra_replacements:
            replace_once(path, old, new)
        clear_python_bytecode(path)
        mutant = run_command(mutation)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    finally:
        path.write_bytes(original)
        clear_python_bytecode(path)

    restored = path.read_bytes() == original
    output = "" if mutant is None else str(mutant["_output"])
    owners_observed = {
        owner: owner in output for owner in mutation.owner_needles
    }
    killed = bool(
        baseline is not None
        and baseline["exit_code"] == 0
        and mutant is not None
        and mutant["exit_code"] != 0
        and all(owners_observed.values())
        and restored
        and error is None
    )
    return {
        "id": mutation.mutation_id,
        "task": mutation.task,
        "description": mutation.description,
        "product_file": mutation.path,
        "command": list(mutation.command),
        "cwd": mutation.cwd,
        "owner_needles": list(mutation.owner_needles),
        "owners_observed": owners_observed,
        "baseline": None if baseline is None else public_result(baseline),
        "mutant": None if mutant is None else public_result(mutant),
        "runner_error": error,
        "killed": killed,
        "restore": {
            "before_sha256": digest(original),
            "after_sha256": digest(path.read_bytes()),
            "byte_identical": restored,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [run_mutation(mutation) for mutation in MUTATIONS]
    payload = {
        "schema_version": 1,
        "mutation_count": len(results),
        "killed_count": sum(bool(row["killed"]) for row in results),
        "all_mutations_killed": all(bool(row["killed"]) for row in results),
        "all_files_restored_byte_identically": all(
            bool(row["restore"]["byte_identical"]) for row in results
        ),
        "credential_environment_removed_from_children": True,
        "mutations": results,
    }
    output = Path(args.output)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    summary = {
        "mutations": payload["mutation_count"],
        "killed": payload["killed_count"],
        "restored": payload["all_files_restored_byte_identically"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if payload["all_mutations_killed"] and payload[
        "all_files_restored_byte_identically"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())
