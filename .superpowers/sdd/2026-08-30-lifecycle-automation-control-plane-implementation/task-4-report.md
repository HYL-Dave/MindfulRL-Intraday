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

## Review Round 1 Fixes

Implementation: `5c6f10dc` (`fix(lifecycle): make attended dispatch lock-owned`)

This section supersedes two statements in the original Task 4 result without
rewriting the historical report:

- HTTP dispatch is no longer guarded by a route-local lock. The scheduler now
  acquires the real lifecycle execution flock synchronously, transfers that
  exact entered lease to one background owner, and releases it only after
  startup reconciliation, worker execution, final owner cleanup, and result
  recording.
- A persisted `running` row is no longer sufficient for HTTP 409. Only a live
  shared-flock collision returns `automation_case_running`; after successful
  ownership acquisition, stale running rows are reconciled before exact-case
  attended work begins.

The predecessor-chain validator now requires every row to carry a valid input
evidence digest and the canonical semantic run key derived from that digest.
It compares the complete semantic identity across the chain while continuing
to read a valid legacy `predecessor_failed_run_id` link.

### Review RED Evidence

#### Scheduler-owned dispatch authority

Command before the scheduler dispatch primitive existed:

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py \
  -k 'dispatch_acquires_ownership or dispatch_thread_start_failure'
```

Output:

```text
FF                                                                       [100%]
2 failed, 55 deselected
AttributeError: module ...security_lifecycle_automation_scheduler has no
attribute 'dispatch_and_record_security_lifecycle_automation'
```

These owners require acquisition before `started`, retention through both
reconciliation calls, worker execution and recording, exact owner-ID transfer,
and explicit release if thread startup fails.

#### Live collision and stale-orphan endpoint authority

Command against the route-local-lock and persisted-row-precheck implementation:

```text
pytest -q tests/test_security_lifecycle_routes.py \
  -k 'global_automation_run_dispatches or case_automation_run_dispatches or source_only or real_flock_collision or stale_running_row'
```

Output:

```text
FFFFFF                                                                   [100%]
6 failed, 21 deselected
```

The failures covered the absent scheduler dispatch boundary, global and exact
real-flock collisions, removal of the route-local lock, and a stale persisted
`running` row that had to be reconciled after ownership acquisition instead of
being rejected before it.

#### Complete predecessor semantic identity

Command against the four-field chain identity:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'different_input_evidence_snapshot or noncanonical_semantic_run_key or legacy_predecessor_missing_semantic_metadata or hard_traversal_limit'
```

Output:

```text
FFF.                                                                     [100%]
3 failed, 1 passed, 72 deselected
FAILED ...different_input_evidence_snapshot - Failed: DID NOT RAISE
FAILED ...noncanonical_semantic_run_key - Failed: DID NOT RAISE
FAILED ...legacy_predecessor_missing_semantic_metadata - Failed: DID NOT RAISE
```

The hard traversal guard already existed and passed as a characterization
owner; its reverse control below proves the owner is effective. The malformed
legacy-link owner likewise exercised an existing strict parser and was added
without weakening valid legacy-row readability.

### Review GREEN Evidence

Scheduler ownership and release:

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py \
  -k 'dispatch_acquires_ownership or dispatch_thread_start_failure or recorded_runner_persists_result_before_releasing_execution_lock'
...                                                                      [100%]
3 passed, 54 deselected in 0.52s
```

Live collisions, endpoint dispatch, and stale-orphan recovery:

```text
pytest -q tests/test_security_lifecycle_routes.py \
  -k 'global_automation_run_dispatches or case_automation_run_dispatches or source_only or real_flock_collision or stale_running_row'
......                                                                   [100%]
6 passed, 21 deselected in 2.47s
```

Semantic identity, valid legacy readability, malformed legacy context, cycle,
and traversal limit:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'different_input_evidence_snapshot or noncanonical_semantic_run_key or legacy_predecessor_missing_semantic_metadata or malformed_legacy_predecessor or hard_traversal_limit or legacy_failed_predecessor_field_remains_chain_readable or predecessor_cycle'
.......                                                                  [100%]
7 passed, 70 deselected in 0.18s
```

All directly changed test modules after restoring every mutation:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_routes.py
161 passed in 13.62s
```

### Review Reverse Controls

Each mutation was applied independently and restored before the GREEN and
full-suite runs.

1. Probe and release the transferred lease immediately after acquisition:

   ```text
   FAILED test_dispatch_acquires_ownership_before_return_and_transfers_exact_lease
   assert lock_held is True
   1 failed in 0.45s
   ```

2. Remove explicit lease cleanup when `Thread.start()` raises:

   ```text
   FAILED test_dispatch_thread_start_failure_releases_transferred_ownership
   assert ['lock_acquired'] == ['lock_acquired', 'lock_released']
   1 failed in 0.43s
   ```

3. Disable exact-case mapping of a real flock collision to HTTP 409:

   ```text
   FAILED test_case_automation_run_returns_409_on_real_flock_collision
   assert 200 == 409
   1 failed in 2.14s
   ```

4. Skip startup reconciliation after acquiring the execution lease:

   ```text
   FAILED test_case_automation_run_reconciles_a_stale_running_row_after_lock_acquisition
   assert 0 == 1
   1 failed in 2.19s
   ```

5. Remove evidence digest and canonical semantic key from cross-row identity:

   ```text
   FAILED test_predecessor_chain_rejects_a_different_input_evidence_snapshot
   Failed: DID NOT RAISE <class 'ValueError'>
   1 failed in 0.17s
   ```

6. Disable the per-row canonical semantic-run-key check:

   ```text
   FAILED test_predecessor_chain_rejects_a_noncanonical_semantic_run_key
   Failed: DID NOT RAISE <class 'ValueError'>
   1 failed in 0.17s
   ```

7. Treat a non-string legacy predecessor field as an absent predecessor:

   ```text
   FAILED test_malformed_legacy_predecessor_field_fails_closed
   Failed: DID NOT RAISE <class 'ValueError'>
   1 failed in 0.16s
   ```

8. Disable the predecessor traversal hard bound:

   ```text
   FAILED test_predecessor_chain_enforces_the_hard_traversal_limit
   Failed: DID NOT RAISE <class 'ValueError'>
   1 failed in 0.20s
   ```

The original report's reverse controls for both default-false flags,
caller-context counter-reset resistance, the cycle visited set, and
policy/provenance isolation remain unchanged. Their current positive owners
were rerun together:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  -k 'due_failed_retry_requires_explicit_authority or attended_new_attempt_preserves_each_terminal_predecessor or retry_count_comes_from_predecessor_chain_not_caller_context or predecessor_cycle_fails_closed or attended_attempt_does_not_change_policy_or_decision_provenance'
.......                                                                  [100%]
7 passed, 70 deselected in 0.15s
```

### Review Full Verification

Task 4 focused files:

```text
209 passed in 14.97s
```

Task 4 plus automation runtime, investigation, tools, and listing evidence:

```text
327 passed in 18.86s
```

Lifecycle schema, automation migration, and production scheduler boundary:

```text
113 passed in 5.02s
```

The three binding no-same-revision-auto-replay tests were not edited and were
rerun by exact node ID:

```text
...                                                                      [100%]
3 passed in 0.13s
```

Additional checks:

```text
python -m compileall -q <three modified product modules>  # clean
git diff --check                                          # clean
```

No lifecycle DDL/version, decision policy/version, decision provenance,
automatic profile-mutation authority, provider, production database, App,
merge, or push boundary changed or was exercised in this review fix.
