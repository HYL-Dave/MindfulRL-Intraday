# Task 4 Report: Predecessor-Linked Retry and Attended Run Authority

Date: 2026-08-30
Implementation: `c059f7c8` (`feat(lifecycle): add attended retry authority`)
Base: `662b47d2`

## Status

DONE. Task 4 is implemented and committed. No provider call, production DB
read/write, App restart, schema migration, merge, or push occurred.

## Result

- `reserve_run` now exposes two independent, default-false authorities:
  `allow_due_failed_retry` and `allow_new_attempt`.
- Scheduled production execution grants only due-failed retry authority.
  Exact-case attended execution grants only new-attempt authority.
- New attempts use `predecessor_run_id`; persisted legacy
  `predecessor_failed_run_id` chains remain readable.
- Retry counts come from a cycle-safe, identity-checked persisted predecessor
  chain with a hard traversal bound. Caller query context is never a retry
  counter authority.
- Failed rows persist only the closed `automatic_retry` object with `class`
  and `retry_not_before` in `query_context_json`. Their schema `retry_at`
  remains NULL.
- The retry matrix is implemented exactly:
  - `persistence_failed`: three retries after 15 minutes, 1 hour, and 6 hours;
  - `source_payload_invalid`: one retry after 1 hour;
  - `internal_error`: one retry after 1 hour;
  - `extractor_failed` and `profile_schema_mismatch`: attended only.
- Pending terminal finalization resumes its existing succeeded run, including
  after automatic finalization retry exhaustion. A completed succeeded run
  receives a predecessor-linked attended attempt.
- Added fire-and-return POST routes for global due work and exact attended
  case work. They call the Task 3 lock-owned run-and-record boundary; a
  persisted running case returns HTTP 409 and a concurrent local dispatch
  returns typed `skipped/already_running`.
- Exact-case targeting happens before worker selection, so `limit=1` cannot
  select a different case. A source-only case is admitted without prematurely
  materializing a profile case row.

## Baseline

Command:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_routes.py
```

Output before Task 4 tests or product edits:

```text
178 passed in 13.87s
```

## RED Evidence

### Reservation authority and predecessor attempts

Command:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'due_failed_retry_requires_explicit_authority or attended_new_attempt_preserves_each_terminal_predecessor or retry_count_comes_from_predecessor_chain_not_caller_context'
```

Initial output:

```text
5 failed, 58 deselected
TypeError: SecurityLifecycleFactKernel.reserve_run() got an unexpected keyword
argument 'allow_due_failed_retry' / 'allow_new_attempt'
```

This RED covered default no-authority behavior, failed/blocked/succeeded
attended attempts, predecessor preservation, and caller-context reset
resistance.

### Retry matrix and reconciliation metadata

Command:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'single_retry_classes or manual_only_failure_classes or reconciled_internal_error or legacy_failed_predecessor or predecessor_cycle or attended_attempt_does_not_change_policy'
```

Initial implementation-check output:

```text
1 failed, 7 passed, 63 deselected
FAILED test_reconciled_internal_error_receives_one_hour_retry_authority
KeyError: 'automatic_retry'
```

The missing producer-side metadata was then added to interrupted-run
reconciliation. A self-review RED subsequently added full persistence retry
exhaustion through all three scheduled attempts.

### Worker authority and exact targeting

Command:

```text
pytest -q tests/test_security_lifecycle_automation_worker.py \
  -k 'worker_automatic_retry_authority_is_opt_in_and_due_only or attended_worker_targets_exact_case_and_creates_new_attempt'
```

Initial output:

```text
2 failed, 46 deselected
TypeError: LifecycleAutomationWorker.__init__() got an unexpected keyword
argument 'allow_due_failed_retry' / 'allow_new_attempt'
```

### Scheduler authority and Task 3 boundary

Command:

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py \
  -k 'scheduled_runner_grants_only_due_failed_retry_authority or recorded_attended_runner_targets_one_case_and_grants_new_attempt_only'
```

Initial output:

```text
2 failed, 53 deselected
```

The failures showed the worker did not receive the authority flags and the
recorded runner did not accept an exact-case target.

### Endpoint family

Command:

```text
pytest -q tests/test_security_lifecycle_routes.py \
  -k 'global_automation_run or case_automation_run or automation_run_returns_typed_skip or app_mounts_the_exact_lifecycle_route_surface'
```

Initial output:

```text
5 failed, 20 deselected
```

The route surface, dispatch function, 409 behavior, and typed local
single-flight response were absent.

Self-review then crossed the real source/profile seam:

```text
pytest -q tests/test_security_lifecycle_routes.py \
  -k 'case_automation_run_materializes_a_source_only_case_in_the_worker'
1 failed, 25 deselected
assert 404 == 200
```

The first route implementation incorrectly required an already-materialized
profile case. The running-row check now uses the verified profile connection
directly after the read service validates the source case.

## GREEN Evidence

### Kernel authority, defaults, chain, and matrix

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'due_failed_retry_requires_explicit_authority or attended_new_attempt_preserves_each_terminal_predecessor or retry_count_comes_from_predecessor_chain_not_caller_context or persistence_failure_allows_exactly_three_automatic_attempts or single_retry_classes or manual_only_failure_classes or reconciled_internal_error or legacy_failed_predecessor or predecessor_cycle or attended_attempt_does_not_change_policy'
..............                                                           [100%]
14 passed, 58 deselected in 0.18s
```

### Worker

```text
pytest -q tests/test_security_lifecycle_automation_worker.py \
  -k 'worker_automatic_retry_authority_is_opt_in_and_due_only or attended_worker_targets_exact_case_and_creates_new_attempt or terminal_finalization_failure_uses_bounded_backoff_without_hot_loop'
...                                                                      [100%]
3 passed, 45 deselected in 0.45s
```

### Scheduler

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py \
  -k 'scheduled_runner_grants_only_due_failed_retry_authority or recorded_attended_runner_targets_one_case_and_grants_new_attempt_only or recorded_runner_persists_result_before_releasing_execution_lock'
...                                                                      [100%]
3 passed, 52 deselected in 0.53s
```

### Endpoints

```text
pytest -q tests/test_security_lifecycle_routes.py \
  -k 'global_automation_run or case_automation_run or automation_run_returns_typed_skip or app_mounts_the_exact_lifecycle_route_surface'
......                                                                   [100%]
6 passed, 20 deselected in 3.05s
```

## Existing No-Auto-Replay Owners

The bodies of these three binding tests were not edited:

- `test_current_policy_retries_a_failed_run_without_deleting_v1_history`
- `test_current_execution_revision_does_not_replay_failed_semantic_run_later`
- `test_cross_revision_due_blocked_failure_does_not_replay_same_attempt_revision`

Exact verification:

```text
pytest -q \
  tests/test_security_lifecycle_fact_kernel.py::test_current_policy_retries_a_failed_run_without_deleting_v1_history \
  tests/test_security_lifecycle_fact_kernel.py::test_current_execution_revision_does_not_replay_failed_semantic_run_later \
  tests/test_security_lifecycle_fact_kernel.py::test_cross_revision_due_blocked_failure_does_not_replay_same_attempt_revision
...                                                                      [100%]
3 passed in 0.13s
```

## Reverse Controls

Each product mutation was applied independently, run against its named owner,
and restored before final verification.

1. Change `allow_due_failed_retry` default from false to true:

   ```text
   FAILED test_due_failed_retry_requires_explicit_authority_and_preserves_predecessor
   assert parked.should_execute is False
   1 failed in 0.16s
   ```

2. Change `allow_new_attempt` default from false to true:

   ```text
   FAILED ...preserves_each_terminal_predecessor[failed]
   FAILED ...preserves_each_terminal_predecessor[blocked]
   FAILED ...preserves_each_terminal_predecessor[succeeded]
   3 failed in 0.20s
   ```

3. Replace predecessor-derived prior-failure count with zero:

   ```text
   FAILED test_retry_count_comes_from_predecessor_chain_not_caller_context
   expected retry_not_before 2026-08-25T09:17:00Z;
   observed 2026-08-25T03:32:00Z
   1 failed in 0.16s
   ```

4. Disable the visited-set cycle check:

   ```text
   FAILED test_predecessor_cycle_fails_closed_before_creating_attended_attempt
   expected automation_predecessor_cycle; observed automation_predecessor_chain_limit
   1 failed in 0.20s
   ```

5. Include retry/predecessor identity in decision provenance:

   ```text
   FAILED test_attended_attempt_does_not_change_policy_or_decision_provenance
   first and attended decision_provenance_sha256 differed
   1 failed in 0.16s
   ```

Post-restore control:

```text
8 passed, 64 deselected in 0.16s
```

## Full Verification

Task files:

```text
201 passed in 14.71s
```

Task plus adjacent runtime, investigation, tools, and listing suites:

```text
319 passed in 18.58s
```

Schema and production scheduler boundary:

```text
113 passed in 4.69s
```

Additional checks:

```text
python -m compileall -q <four modified product modules>  # clean
git diff --check                                         # clean
```

`git diff` is empty for lifecycle schema DDL, decision policy, and the binding
design. Policy version, execution revision constants, decision provenance
shape, lifecycle schema version, and automatic profile-mutation authority are
unchanged.

## Self-Review

- Confirmed each new attended attempt preserves its predecessor row exactly.
- Confirmed automatic retries create isolated rows while due retryable
  `blocked` behavior continues to reuse its existing row.
- Confirmed pending finalization remains a same-run operation.
- Confirmed the scheduler cannot grant both retry authorities through its
  production entry points.
- Confirmed exact-case targeting precedes sorting/limiting.
- Confirmed both POST routes call only
  `run_and_record_security_lifecycle_automation`, never an unrecorded run plus
  a separate recorder call.
- Confirmed route registration needs no router-file change because the
  existing lifecycle router is already mounted.
- No unresolved Task 4 concern or architectural conflict remains.

## Boundaries

All tests used in-memory or temporary databases and monkeypatched transports.
No provider call, production DB read/write, App restart, schema migration,
merge, or push was performed.
