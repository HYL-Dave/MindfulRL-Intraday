"""Apply each honesty-repair mutation independently and prove its owner kills it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[3]


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    description: str
    path: str
    old: str
    new: str
    owners: tuple[str, ...]
    command: tuple[str, ...]
    cwd: str = "."
    runner: str = "pytest"


KERNEL = "src/security_lifecycle_fact_kernel.py"
WORKER = "src/security_lifecycle_automation_worker.py"
DECISION = "src/security_lifecycle_decision_policy.py"
SEC = "src/security_lifecycle_sec_evidence.py"
SCHEDULER = "src/service/security_lifecycle_automation_scheduler.py"
DISPOSITION = "src/security_lifecycle_disposition.py"
TOOLS = "src/tools/security_lifecycle_tools.py"
VIEW = "apps/arkscope-web/src/lifecycle/LifecycleView.tsx"


MUTATIONS = (
    Mutation(
        "M1",
        "remove deadline citation hash comparison",
        KERNEL,
        "    if hashlib.sha256(cited).hexdigest() != digest:\n"
        "        raise ValueError(error_name)\n",
        "    if False and hashlib.sha256(cited).hexdigest() != digest:\n"
        "        raise ValueError(error_name)\n",
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_mutations_fail_atomically[forged_hash]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_mutations_fail_atomically[forged_hash]",
        ),
    ),
    Mutation(
        "M2",
        "validate deadline citations only for not_confirmed_as_of",
        KERNEL,
        "        has_deadline_fields = bool(\n"
        "            _SOURCE_DEADLINE_CONTEXT_FIELDS.intersection(context)\n"
        "        )\n",
        "        has_deadline_fields = (\n"
        "            context.get(\"monitoring_reason\") == \"not_confirmed_as_of\"\n"
        "            and bool(_SOURCE_DEADLINE_CONTEXT_FIELDS.intersection(context))\n"
        "        )\n",
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_any_deadline_field_triggers_complete_set_before_deadline",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_any_deadline_field_triggers_complete_set_before_deadline",
        ),
    ),
    Mutation(
        "M3",
        "accept a partial deadline provenance set",
        KERNEL,
        "        if not _SOURCE_DEADLINE_CONTEXT_FIELDS.issubset(context):\n"
        "            raise ValueError(\"blocker_citation\")\n",
        "        if not _SOURCE_DEADLINE_CONTEXT_FIELDS.issubset(context):\n"
        "            normalized.append((code, retryable, context_json))\n"
        "            continue\n",
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_mutations_fail_atomically[partial_set]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_blocker_citation_mutations_fail_atomically[partial_set]",
        ),
    ),
    Mutation(
        "M4",
        "bump shared SEC _RULE_VERSION from 3 to 4",
        SEC,
        '_RULE_VERSION = "3"\n',
        '_RULE_VERSION = "4"\n',
        (
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_explicit_outside_date_is_hash_cited_and_conflicts_fail_closed",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_explicit_outside_date_is_hash_cited_and_conflicts_fail_closed",
        ),
    ),
    Mutation(
        "M5",
        "select the from date in a directional extension",
        SEC,
        '        target_date = target_match.group("date")\n',
        "        target_date = next(_ANY_MONTH_DATE.finditer(sentence)).group(\"date\")\n",
        (
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
    ),
    Mutation(
        "M6",
        "accept a negated or historical date sentence",
        SEC,
        "    rf\"\\b(?:outside|termination) date\\s+(?:is|shall be|remains)\\s+\"\n",
        "    rf\"\\b(?:outside|termination) date\\s+(?:is|shall be|remains|of)\\s+\"\n",
        (
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
    ),
    Mutation(
        "M7",
        "backdate final as_of to source_deadline",
        SCHEDULER,
        '        context["as_of"] = today.isoformat()\n',
        '        context["as_of"] = deadline_date.isoformat()\n',
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_pending_event_monitoring_uses_explicit_dates_and_final_source_check",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_pending_event_monitoring_uses_explicit_dates_and_final_source_check",
        ),
    ),
    Mutation(
        "M8",
        "map not_confirmed_as_of back to confirmed_effective",
        DISPOSITION,
        '                    "not_confirmed_yet",\n'
        '                    "history",\n'
        '                    "not_confirmed_as_of",\n',
        '                    "confirmed_effective",\n'
        '                    "history",\n'
        '                    "not_confirmed_as_of",\n',
        (
            "tests/test_security_lifecycle_disposition.py::"
            "test_not_confirmed_as_of_projects_the_actual_completed_check_date",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_disposition.py::"
            "test_not_confirmed_as_of_projects_the_actual_completed_check_date",
        ),
    ),
    Mutation(
        "M9",
        "omit disposition_as_of from the read summary",
        TOOLS,
        '        "disposition_as_of": case["disposition_as_of"],\n',
        "",
        (
            "tests/test_security_lifecycle_tools.py::"
            "test_read_service_exposes_derived_final_check_date_in_list_and_detail",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_tools.py::"
            "test_read_service_exposes_derived_final_check_date_in_list_and_detail",
        ),
    ),
    Mutation(
        "M10",
        "render generic confirmed-complete copy for final unconfirmed History",
        VIEW,
        '  if (reason === "not_confirmed_as_of" && dispositionAsOf) {\n'
        "    return <>{t(($) => $.lifecycle.dispositionReasons.notConfirmedAsOfDated, {\n"
        "      date: dispositionAsOf,\n"
        "    })}</>;\n"
        "  }\n",
        '  if (reason === "not_confirmed_as_of" && dispositionAsOf) {\n'
        '    return <>{lifecycleDispositionLabel("confirmed_effective", locale)}</>;\n'
        "  }\n",
        (
            "apps/arkscope-web/src/lifecycle/LifecycleView.test.tsx::"
            "Lifecycle workflow::renders truthful dated final-History reasons in both "
            "locales without acknowledgement",
        ),
        (
            "npm",
            "test",
            "--",
            "src/lifecycle/LifecycleView.test.tsx",
            "-t",
            "renders truthful dated final-History reasons in both locales without acknowledgement",
            "--reporter=json",
        ),
        cwd="apps/arkscope-web",
        runner="vitest",
    ),
    Mutation(
        "M11",
        "include execution revision in decision provenance",
        KERNEL,
        '        "SELECT case_id,observation_fingerprint_sha256,policy_version,mode "\n'
        '        "FROM security_lifecycle_automation_runs WHERE run_id=?",\n'
        "        (run_id,),\n"
        "    ).fetchone()\n"
        "    if row is None:\n"
        '        raise KeyError("automation_run_not_found")\n'
        "    return _provenance(\n"
        "        case_id=str(row[0]),\n"
        "        observation_fingerprint_sha256=str(row[1]),\n"
        "        policy_version=str(row[2]),\n"
        "        mode=str(row[3]),\n",
        '        "SELECT case_id,observation_fingerprint_sha256,policy_version,mode,"\n'
        '        "query_context_json FROM security_lifecycle_automation_runs WHERE run_id=?",\n'
        "        (run_id,),\n"
        "    ).fetchone()\n"
        "    if row is None:\n"
        '        raise KeyError("automation_run_not_found")\n'
        "    execution_revision = str(\n"
        '        json.loads(str(row[4])).get("execution_revision", "unknown")\n'
        "    )\n"
        "    return _provenance(\n"
        "        case_id=str(row[0]),\n"
        "        observation_fingerprint_sha256=str(row[1]),\n"
        "        policy_version=str(row[2]) + execution_revision,\n"
        "        mode=str(row[3]),\n",
        (
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_execution_revision_does_not_change_decision_or_transition_authority",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_execution_revision_does_not_change_decision_or_transition_authority",
        ),
    ),
    Mutation(
        "M12",
        "replay a failed row at the same execution revision",
        KERNEL,
        '                if row["status"] != "failed" or existing_revision == execution:\n',
        '                if row["status"] != "failed":\n',
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_current_execution_revision_does_not_replay_failed_semantic_run_later",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_current_execution_revision_does_not_replay_failed_semantic_run_later",
        ),
    ),
    Mutation(
        "M13",
        "replay a succeeded semantic run after only execution revision changes",
        KERNEL,
        '                if row["status"] != "failed" or existing_revision == execution:\n',
        '                if (\n'
        '                    row["status"] not in {"failed", "succeeded"}\n'
        '                    or existing_revision == execution\n'
        '                ):\n',
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_successful_replay_prevents_later_revision_fanout",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_successful_replay_prevents_later_revision_fanout",
        ),
    ),
    Mutation(
        "M14",
        "add deadline fields to source_conflict outside the shared validator",
        KERNEL,
        '                            {"fact_types": sorted(conflicts)},\n',
        '                            {\n'
        '                                "fact_types": sorted(conflicts),\n'
        '                                "source_deadline": "2026-04-01",\n'
        '                            },\n',
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_conflicting_current_facts_are_typed_and_never_majority_resolved",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_conflicting_current_facts_are_typed_and_never_majority_resolved",
        ),
    ),
    Mutation(
        "M15",
        "narrow candidate admission to the new deadline grammar",
        SEC,
        "    if _SOURCE_DEADLINE_PHRASE.search(sentence) is not None and (\n"
        "        _ANY_MONTH_DATE.search(sentence) is not None\n"
        "        or _ANY_ISO_DATE.search(sentence) is not None\n"
        "    ):\n"
        "        return True\n",
        "    if any(\n"
        "        pattern.search(sentence) is not None\n"
        "        for pattern in (_TERMINATE_IF_BY, _CURRENT_DEADLINE, _EXTENDED_DEADLINE)\n"
        "    ):\n"
        "        return True\n",
        (
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_evidence_admission_identity_is_unchanged_for_rejected_and_oversized_sentences",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_evidence_admission_identity_is_unchanged_for_rejected_and_oversized_sentences",
        ),
    ),
    Mutation(
        "M16",
        "reject any and/or after a target date",
        SEC,
        '        if _COORDINATE_TARGET.match(sentence[target_match.end("date") :]) is not None:\n',
        '        if re.search(r"\\b(?:and|or)\\b", sentence[target_match.end("date") :], re.IGNORECASE):\n',
        (
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_sec_evidence.py::"
            "test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation",
        ),
    ),
    Mutation(
        "M17",
        "retain every historical market snapshot in current decision quality",
        DECISION,
        "    if len(market) <= 1:\n"
        "        return evidence, facts\n",
        "    if len(market) >= 1:\n"
        "        return evidence, facts\n",
        (
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_market_recheck_uses_latest_snapshot_and_converges_from_frozen_to_fresh",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_market_recheck_uses_latest_snapshot_and_converges_from_frozen_to_fresh",
        ),
    ),
    Mutation(
        "M18",
        "skip decision authority whenever a source conflict blocker is present",
        WORKER,
        '                if not blockers or "source_conflict" in blocker_codes:\n',
        "                if not blockers:\n",
        (
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection[conflict_only]",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection[conflict_plus_pending]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection",
        ),
    ),
    Mutation(
        "M19",
        "skip succeeded runs whose terminal finalization marker is pending",
        KERNEL,
        '                if row["status"] == "succeeded" and _terminal_finalization_pending(\n'
        "                    context\n"
        "                ):\n",
        '                if False and row["status"] == "succeeded" and _terminal_finalization_pending(\n'
        "                    context\n"
        "                ):\n",
        (
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_approval_boundary_recovery_deduplicates_real_transition_and_mutates_no_profile",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_terminal_finalization_recovers_idempotently_after_each_boundary[acceptance]",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_terminal_finalization_recovers_idempotently_after_each_boundary[approval]",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_terminal_finalization_recovers_idempotently_after_each_boundary[assessment]",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_terminal_finalization_recovers_idempotently_after_each_boundary[proposal]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_worker.py",
            "-k",
            "terminal_finalization_recovers_idempotently_after_each_boundary or "
            "approval_boundary_recovery_deduplicates_real_transition_and_mutates_no_profile",
        ),
    ),
    Mutation(
        "M20",
        "project the newest historical run without current observation binding",
        DISPOSITION,
        "    return next((run for run in runs if _artifact_is_current(case, run)), None)\n",
        "    return runs[0] if runs else None\n",
        (
            "tests/test_security_lifecycle_tools.py::"
            "test_new_observation_keeps_old_terminal_run_and_transition_as_activity_only",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_tools.py::"
            "test_new_observation_keeps_old_terminal_run_and_transition_as_activity_only",
        ),
    ),
    Mutation(
        "M21",
        "project a settled transition without current artifact binding",
        DISPOSITION,
        "    if transition is not None and not _artifact_is_current(\n"
        "        case,\n"
        "        transition,\n"
        "        assessment,\n"
        "    ):\n",
        "    if False and transition is not None and not _artifact_is_current(\n"
        "        case,\n"
        "        transition,\n"
        "        assessment,\n"
        "    ):\n",
        (
            "tests/test_security_lifecycle_tools.py::"
            "test_new_observation_keeps_old_terminal_run_and_transition_as_activity_only",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_tools.py::"
            "test_new_observation_keeps_old_terminal_run_and_transition_as_activity_only",
        ),
    ),
    Mutation(
        "M22",
        "use immutable initial execution revision for failed replay eligibility",
        KERNEL,
        "                existing_revision = str(\n"
        "                    context.get(\n"
        "                        _LATEST_ATTEMPT_REVISION_KEY,\n"
        '                        context.get("execution_revision", "unknown"),\n'
        "                    )\n"
        "                )\n",
        "                existing_revision = str(\n"
        '                    context.get("execution_revision", "unknown")\n'
        "                )\n",
        (
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_cross_revision_due_blocked_failure_does_not_replay_same_attempt_revision",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_fact_kernel.py::"
            "test_cross_revision_due_blocked_failure_does_not_replay_same_attempt_revision",
        ),
    ),
    Mutation(
        "M23",
        "omit deadline-only market acquisition eligibility",
        SCHEDULER,
        "            or deadline_due\n",
        "            or False\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_deadline_without_effective_date_caps_schedule_and_triggers_final_market_check",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_deadline_without_effective_date_caps_schedule_and_triggers_final_market_check",
        ),
    ),
    Mutation(
        "M24",
        "allow pre-deadline monitoring to schedule after the deadline",
        SCHEDULER,
        "        next_check = min(next_check, deadline_check)\n",
        "        next_check = max(next_check, deadline_check)\n",
        (
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_deadline_without_effective_date_caps_schedule_and_triggers_final_market_check",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_scheduler.py::"
            "test_deadline_without_effective_date_caps_schedule_and_triggers_final_market_check",
        ),
    ),
    Mutation(
        "M25",
        "allow an older queue success response to commit after a newer response",
        VIEW,
        "      const response = await listSecurityLifecycleCases(requestFilters);\n"
        "      if (requestId !== caseRequestRef.current) return;\n"
        "      setCases(response.cases);\n",
        "      const response = await listSecurityLifecycleCases(requestFilters);\n"
        "      setCases(response.cases);\n",
        (
            "apps/arkscope-web/src/lifecycle/LifecycleView.test.tsx::"
            "Lifecycle workflow::commits only the newest queue response when requests "
            "resolve out of order",
        ),
        (
            "npm",
            "test",
            "--",
            "src/lifecycle/LifecycleView.test.tsx",
            "-t",
            "commits only the newest queue response when requests resolve out of order",
            "--reporter=json",
        ),
        cwd="apps/arkscope-web",
        runner="vitest",
    ),
    Mutation(
        "M26",
        "let approved execute_on mask stale-policy daily revalidation",
        DISPOSITION,
        "                if not _automation_transition_is_stale(transition):\n"
        "                    return transition_execute_at\n",
        "                if True:\n"
        "                    return transition_execute_at\n",
        (
            "tests/test_security_lifecycle_disposition.py::"
            "test_stale_automation_transition_rechecks_before_later_execute_on",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_disposition.py::"
            "test_stale_automation_transition_rechecks_before_later_execute_on",
        ),
    ),
    Mutation(
        "M27",
        "classify a typed IBKR missing receipt as provider unavailability",
        DISPOSITION,
        '    "ibkr_gateway_unavailable": "market_infrastructure",\n'
        '    "ibkr_entitlement_denied": "market_infrastructure",\n',
        '    "ibkr_gateway_unavailable": "market_infrastructure",\n'
        '    "ibkr_contract_missing": "market_infrastructure",\n'
        '    "ibkr_entitlement_denied": "market_infrastructure",\n',
        (
            "tests/test_security_lifecycle_disposition.py::"
            "test_ibkr_missing_is_present_evidence_while_ambiguity_is_conflict",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_disposition.py::"
            "test_ibkr_missing_is_present_evidence_while_ambiguity_is_conflict",
        ),
    ),
    Mutation(
        "M28",
        "allow pending monitoring to mask source-conflict attention",
        DISPOSITION,
        '    elif "source_conflict" in blocker_codes:\n'
        "        disposition, bucket, reason = (\n"
        '            "exception_required",\n'
        '            "attention",\n'
        '            "source_conflict",\n'
        "        )\n",
        '    elif False and "source_conflict" in blocker_codes:\n'
        "        disposition, bucket, reason = (\n"
        '            "exception_required",\n'
        '            "attention",\n'
        '            "source_conflict",\n'
        "        )\n",
        (
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection[conflict_only]",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection[conflict_plus_pending]",
        ),
        (
            "pytest",
            "-q",
            "tests/test_security_lifecycle_automation_worker.py::"
            "test_source_conflict_crosses_worker_kernel_and_attention_projection",
        ),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"mutation_target_count:{path.relative_to(ROOT)}:{count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _pytest_failures(output: str) -> tuple[list[str], int, int]:
    owners = re.findall(r"^FAILED (\S+?)(?:\s+-|$)", output, flags=re.MULTILINE)
    passed = re.search(r"(\d+) passed", output)
    skipped = re.search(r"(\d+) skipped", output)
    return sorted(set(owners)), int(passed.group(1)) if passed else 0, int(skipped.group(1)) if skipped else 0


def _vitest_failures(payload: dict) -> tuple[list[str], int, int]:
    owners: list[str] = []
    passed = 0
    skipped = 0
    for result in payload.get("testResults", []):
        test_path = Path(result["name"]).resolve().relative_to(ROOT)
        for assertion in result.get("assertionResults", []):
            status = assertion.get("status")
            if status == "failed":
                parts = [str(test_path), *assertion.get("ancestorTitles", []), assertion["title"]]
                owners.append("::".join(parts))
            elif status == "passed":
                passed += 1
            elif status in {"pending", "skipped", "todo", "disabled"}:
                skipped += 1
    return sorted(set(owners)), passed, skipped


def _run(mutation: Mutation) -> dict:
    path = ROOT / mutation.path
    original = path.read_bytes()
    before_digest = _sha256(original)
    process: subprocess.CompletedProcess[str] | None = None
    parse_error: str | None = None
    actual_owners: list[str] = []
    passed = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix=f"arkscope-{mutation.mutation_id.lower()}-") as temp:
        try:
            _replace_once(path, mutation.old, mutation.new)
            command = list(mutation.command)
            if mutation.runner == "vitest":
                output_path = Path(temp) / "vitest.json"
                command.append(f"--outputFile={output_path}")
            process = subprocess.run(
                command,
                cwd=ROOT / mutation.cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if mutation.runner == "pytest":
                actual_owners, passed, skipped = _pytest_failures(process.stdout)
            else:
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    actual_owners, passed, skipped = _vitest_failures(payload)
                except (OSError, ValueError, KeyError) as exc:
                    parse_error = f"{type(exc).__name__}:{exc}"
        finally:
            path.write_bytes(original)

    after = path.read_bytes()
    restored = after == original
    expected = sorted(mutation.owners)
    unexpected = sorted(set(actual_owners) - set(expected))
    missing = sorted(set(expected) - set(actual_owners))
    killed = (
        process is not None
        and process.returncode == 1
        and parse_error is None
        and len(actual_owners) == len(expected)
        and not unexpected
        and not missing
    )
    return {
        "id": mutation.mutation_id,
        "mutation": mutation.description,
        "product_files": [mutation.path],
        "command": list(mutation.command),
        "expected_failure_count": len(expected),
        "actual_failure_count": len(actual_owners),
        "expected_owner_node_ids": expected,
        "actual_owner_node_ids": actual_owners,
        "unexpected_owner_node_ids": unexpected,
        "missing_owner_node_ids": missing,
        "passing_test_count": passed,
        "skipped_test_count": skipped,
        "exit_code": None if process is None else process.returncode,
        "parse_error": parse_error,
        "killed": killed,
        "restored_files": [
            {
                "path": mutation.path,
                "before_sha256": before_digest,
                "after_sha256": _sha256(after),
                "byte_identical": restored,
            }
        ],
        "output_tail": [] if process is None else process.stdout.splitlines()[-12:],
    }


def _all_product_paths(mutations: Iterable[Mutation]) -> list[str]:
    return sorted({mutation.path for mutation in mutations})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    initial = {
        path: (ROOT / path).read_bytes() for path in _all_product_paths(MUTATIONS)
    }
    results = [_run(mutation) for mutation in MUTATIONS]
    final_restore = [
        {
            "path": path,
            "sha256": _sha256(initial[path]),
            "byte_identical": (ROOT / path).read_bytes() == initial[path],
        }
        for path in sorted(initial)
    ]
    drift = [
        {
            "mutation_id": result["id"],
            "unexpected_owner_node_ids": result["unexpected_owner_node_ids"],
            "missing_owner_node_ids": result["missing_owner_node_ids"],
        }
        for result in results
        if result["unexpected_owner_node_ids"] or result["missing_owner_node_ids"]
    ]
    payload = {
        "schema_version": 1,
        "mutations": results,
        "mutation_count": len(results),
        "killed_count": sum(result["killed"] for result in results),
        "unexpected_owner_drift": drift,
        "all_mutations_killed": all(result["killed"] for result in results),
        "all_product_files_restored_byte_identically": all(
            item["byte_identical"] for item in final_restore
        ),
        "final_product_file_restore": final_restore,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["all_mutations_killed"] and payload[
        "all_product_files_restored_byte_identically"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())
