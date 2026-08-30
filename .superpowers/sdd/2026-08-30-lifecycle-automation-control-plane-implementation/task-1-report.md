# Task 1 Report: Own Execution and Reap Orphaned Running Rows

Date: 2026-08-30
Branch: `lifecycle-automation-control-plane`
Base: `0570a819`
Implementation commit: `7812bd522de98aeaef13d19a0cdc6a2362ee076d`

## Status

Task 1 is complete. Lifecycle automation now requires proven cross-process
execution ownership before it reconciles or reserves work, persists a bounded
reserved execution owner on every running reservation, and terminalizes only
interrupted `status='running'` rows. The due readiness-recheck entry point also
atomically transfers ownership when it changes a succeeded waiting run back to
running.

No schema migration, provider call, production database access, application
start, merge, push, or subagent dispatch occurred.

## Baseline

Command:

```text
pytest -q tests/test_security_lifecycle_automation_worker.py tests/test_security_lifecycle_fact_kernel.py tests/test_security_lifecycle_automation_scheduler.py
```

Result:

```text
123 passed in 4.07s
```

## RED Evidence

### Initial ownership and recovery RED

The product fix was preceded by this focused command:

```text
pytest -q tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_worker.py::test_base_exception_during_evidence_terminalizes_the_owned_running_run \
  tests/test_security_lifecycle_fact_kernel.py::test_reservation_persists_a_reserved_owner_without_changing_run_identity \
  tests/test_security_lifecycle_fact_kernel.py::test_reconciliation_is_owner_scoped_and_only_terminalizes_running_rows \
  tests/test_security_lifecycle_automation_scheduler.py::test_lock_owner_blocks_a_second_connection_then_release_enables_reconciliation \
  tests/test_security_lifecycle_automation_scheduler.py::test_lock_unavailable_never_reconciles_persisted_running_rows
```

Preserved output summary:

```text
FFFFFFF                                                                  [100%]
ModuleNotFoundError: No module named 'src.service.security_lifecycle_automation_runtime'
AssertionError: assert 'running' == 'failed'
TypeError: SecurityLifecycleFactKernel.reserve_run() got an unexpected keyword argument 'execution_owner_id'
7 failed in 0.84s
```

This established the missing strict lock module, the real-worker orphaned-row
failure, and the absent reservation/reconciliation contracts before product
code changed. The same selection passed after the fix:

```text
7 passed in 0.61s
```

### Due readiness-recheck owner RED

The added load-bearing scenario first creates a succeeded
`waiting_effective_date` run under `initial-readiness-owner`, advances the
clock, enters the real due recheck under `due-readiness-owner`, and raises a
`BaseException` from evidence acquisition.

Command:

```text
pytest -q tests/test_security_lifecycle_automation_worker.py::test_base_exception_during_due_readiness_recheck_reaps_the_current_owner
```

The first test draft stopped in setup because a terminal result is counted as
`drafted`, not `accepted`; only that test assertion was corrected. The
subsequent pre-product-fix RED reached the intended failure:

```text
F                                                                        [100%]
>           assert recovered_run["status"] == "failed"
E           AssertionError: assert 'running' == 'failed'
E             - failed
E             + running
1 failed in 0.30s
```

After the product fix, the exact command returned:

```text
1 passed in 0.22s
```

## Files Changed

Production:

- `src/service/security_lifecycle_automation_runtime.py`
- `src/service/security_lifecycle_automation_scheduler.py`
- `src/security_lifecycle_automation_worker.py`
- `src/security_lifecycle_fact_kernel.py`

Focused tests:

- `tests/test_security_lifecycle_automation_runtime.py`
- `tests/test_security_lifecycle_automation_scheduler.py`
- `tests/test_security_lifecycle_automation_worker.py`
- `tests/test_security_lifecycle_fact_kernel.py`

Direct reservation callers updated for the required explicit owner argument:

- `tests/test_security_lifecycle_investigation.py`
- `tests/test_security_lifecycle_listing_evidence.py`
- `tests/test_security_lifecycle_routes.py`
- `tests/test_security_lifecycle_tools.py`

## Design Choices

1. A dedicated non-blocking `flock` owns
   `security_lifecycle_automation.lock` under the existing lock directory.
   The implementation uses the shared path resolver only; it does not reuse
   the existing degraded `FileLock` behavior.
2. Missing `fcntl`, lock-file open failures, and non-contention flock failures
   produce `execution_lock_unavailable`. Contention produces the typed
   `already_running` skipped result. Neither path opens the profile database,
   invokes providers, reserves work, or reconciles rows.
3. Each acquired lease generates a random `slao_` owner bounded to 64 bytes.
   `execution_owner_id` is a required explicit `reserve_run` argument and a
   reserved query-context key; caller-supplied spoofing is rejected.
4. Owner metadata is excluded from semantic run identity and decision
   provenance. New attempts and same-row blocked retries receive the current
   invocation owner.
5. `reserve_readiness_recheck` validates an explicit owner. For due
   `waiting_effective_date` and `waiting_market_confirmation` rows, it updates
   `query_context_json.execution_owner_id` in the same immediate transaction
   that changes the row to `running`. The succeeded
   `waiting_transition_revalidation` path returns before ownership changes.
6. Kernel reconciliation selects and conditionally updates only
   `status='running'`. It validates persisted context/owner shape, supports
   startup recovery of legacy running rows without an owner, and writes
   `failed/internal_error` plus `{"interrupted_execution":1}`.
7. Startup reconciliation runs only after the OS lock is acquired and before
   provider/session construction. Worker and scheduler `finally` scopes both
   cover `BaseException`; current-owner cleanup completes before lock release.
   Runtime progress is not consulted as a lease or recovery authority.
8. Existing succeeded finalization behavior is unchanged. Reconciliation
   tests snapshot a succeeded row and prove byte-for-byte projected equality
   before and after owner-scoped and startup-style reconciliation.

## Reverse-Mutation Ownership

| Guard | Named test owner | Reverse mutation killed |
| --- | --- | --- |
| Dedicated OS lock | `test_execution_lock_is_exclusive_and_issues_a_bounded_owner_id` and `test_lock_owner_blocks_a_second_connection_then_release_enables_reconciliation` | Removing/bypassing flock lets the second invocation enter instead of returning `already_running`. |
| Fail-closed degraded lock | `test_lock_unavailable_never_reconciles_persisted_running_rows` | Treating missing `fcntl` or lock open failure as success reaches DB/provider boundaries and reaps the control row. |
| Persisted owner and identity exclusion | `test_reservation_persists_a_reserved_owner_without_changing_run_identity` | Omitting owner persistence, accepting caller spoofing, or hashing owner into run identity fails assertions. |
| Owner predicate and running-only scope | `test_reconciliation_is_owner_scoped_and_only_terminalizes_running_rows` | Removing the owner predicate reaps another active owner; broadening status mutates the succeeded control row. |
| Startup reconciliation | `test_lock_owner_blocks_a_second_connection_then_release_enables_reconciliation` | Removing post-lock startup reconciliation leaves the original row running after lock release. |
| Real-worker `BaseException` cleanup | `test_base_exception_during_evidence_terminalizes_the_owned_running_run` | Removing the `BaseException`-covering finally leaves the run stranded and the healthy next invocation skips it as current. |
| Due recheck owner handoff | `test_base_exception_during_due_readiness_recheck_reaps_the_current_owner` | Omitting the atomic owner replacement leaves the due recheck running under the stale owner, so current-owner cleanup cannot terminalize it. |

## GREEN Verification

Focused Task 1 suites:

```text
pytest -q tests/test_security_lifecycle_automation_runtime.py tests/test_security_lifecycle_automation_worker.py tests/test_security_lifecycle_fact_kernel.py tests/test_security_lifecycle_automation_scheduler.py
131 passed in 6.18s
```

Direct API/caller suites:

```text
pytest -q tests/test_security_lifecycle_investigation.py tests/test_security_lifecycle_tools.py tests/test_security_lifecycle_listing_evidence.py tests/test_security_lifecycle_routes.py
137 passed in 9.94s
```

Adjacent recheck/finalization selection:

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py::test_blocker_citation_resolves_existing_evidence_owned_by_same_run tests/test_security_lifecycle_fact_kernel.py::test_readiness_recheck_preserves_cited_history_and_recomputes_provenance tests/test_security_lifecycle_automation_worker.py::test_worker_rechecks_pre_effective_terminal_when_effective_date_becomes_due tests/test_security_lifecycle_automation_worker.py::test_base_exception_during_due_readiness_recheck_reaps_the_current_owner tests/test_security_lifecycle_automation_worker.py::test_transition_approval_drift_fails_closed_without_profile_mutation
5 passed in 0.68s
```

Static verification:

```text
python -m compileall -q src/service/security_lifecycle_automation_runtime.py src/service/security_lifecycle_automation_scheduler.py src/security_lifecycle_automation_worker.py src/security_lifecycle_fact_kernel.py
exit 0, no output

git diff --check
exit 0, no output
```

## Commits

- `7812bd522de98aeaef13d19a0cdc6a2362ee076d` -
  `fix(lifecycle): own and recover automation runs`
- This report is committed in a report-only follow-up commit. Its exact hash
  cannot be embedded in its own contents and is returned in the completion
  contract.

## Self-Review

- Re-read the Task 1 brief, binding execution-ownership/reconciliation spec,
  and all load-bearing rulings after implementation.
- Reviewed the complete production diff and every changed direct test fixture.
- Verified every `reserve_run` and `reserve_readiness_recheck` caller supplies
  the explicit owner.
- Verified lock acquisition precedes database reconciliation and provider
  construction, and degraded lock paths cannot reach either boundary.
- Verified reconciliation SQL is both selected and update-guarded by
  `status='running'`; no succeeded finalization code was changed.
- Verified owner cleanup runs before the OS lock context exits and does not use
  runtime stage state.
- Verified the worktree contains no schema, provider, application startup,
  merge, push, or unrelated production changes.

## Concerns

- The lock intentionally requires POSIX `fcntl`; unsupported or degraded
  environments return `execution_lock_unavailable` and perform no work.
- Verification was limited to the Task 1 focused and direct suites required by
  this task. Full backend/frontend admission belongs to later plan tasks and
  was not run here.
