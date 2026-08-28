# Task 5 Implementer Report

## Status

DONE_WITH_CONCERNS

Task 5 implementation is complete and committed. The focused scheduler, worker,
disposition, and adjacent fact-kernel owners pass. The one requested full backend
run exposed four stale fact-kernel fixture failures; the shared fixture was
corrected and all affected tests pass in targeted verification. The full backend
suite was not run a second time because the task explicitly requested one full
run.

## Commit

- Implementation commit: `eebd4d31` (`feat(lifecycle): replace news acquisition with listing checks`)
- Base: `792f04ff`

## RED / GREEN

RED was established before product edits:

- Original focused baseline: `80 passed`.
- Task 5 owner RED: `23 failed, 76 passed`.
- Failures covered explicit session injection, no-news acquisition, IBKR
  diagnostic/blocking siblings, all eight retryability mutations plus the
  non-retryable conflict sibling, active-family projection, listing component
  status, and effective-date reacquisition.
- A separate disposition RED proved that a real pre-effective undetermined draft
  projected as `awaiting_initial_automation` instead of successful Monitoring.

GREEN evidence:

- Final Task 5 scheduler/worker/disposition owners: `100 passed`.
- Affected full-suite fact-kernel cases after fixture correction: `6 passed`.
- Final expanded focused set including fact-kernel owners: `150 passed`.
- Python compilation, Ruff when available, and `git diff --check` passed.

## Blocker Contract

All listing producer strings continue through the shared scheduler `_blockers()`
conversion and the real fact-kernel `complete_run` persistence/readback path.
No listing-only conversion path or producer result shape was added.

Retryability is closed as required:

- `listing_authority_conflict`: non-retryable.
- `listing_directory_unavailable`: retryable.
- `listing_directory_stale`: retryable.
- `listing_directory_schema_mismatch`: retryable.
- `massive_credential_missing`: retryable.
- `massive_access_denied`: retryable.
- `massive_rate_limited`: retryable.
- `massive_reference_unavailable`: retryable.
- `listing_status_unresolved`: retryable.

Optional component failures are omitted before conversion. The owner proves the
same Massive-missing diagnostic is non-blocking for a sufficient NMS-shaped case
and blocking when terminal confirmation requires Massive. IBKR gateway absence,
entitlement absence, and contract missing are persisted only as integer
diagnostics; ambiguity and source conflict remain blockers. Existing
open-position facts still reach v4 policy unchanged.

## Session and Recheck Contracts

- Each scheduler tick constructs one explicit `ListingRequestBudget`, transport,
  and `ListingAuthoritySession`.
- The session is captured by the tick-local worker evidence-loader closure and is
  closed in `finally`.
- There is no `ContextVar`, global tick session, or hidden tick clock state.
- The shared lazy session preserves one Nasdaq directory snapshot and Massive
  memoization across both selected cases.
- Candidate tickers are limited to the tracked ticker and deterministic SEC
  `successor_ticker` facts.
- SEC acquisition is followed by listing lookup; listing diagnostics are merged
  without publisher/news evidence or database access.
- Pre-effective terminal `undetermined` / `waiting_effective_date` completes as a
  successful drafted Monitoring result.
- Its effective date is the explicit next check. At due time, the worker reserves
  the existing semantic run, reacquires evidence, and re-evaluates without policy
  version, execution revision, or manual-reset changes.
- Due draft discovery is restricted to the latest run, preventing an older draft
  from being reopened.

## Source Projection

The active projection is fixed to:

- `regulator`
- `listing_authority`
- `market_infrastructure`
- `manual`

Legacy publisher/general-web evidence remains stored and readable but does not
create active missing/unavailable/conflict status or influence queue selection.
Listing directory and Massive blockers project to the shared
`listing_authority` family; optional omitted blockers cannot poison that family.

## Tests

- `pytest tests/test_security_lifecycle_automation_scheduler.py tests/test_security_lifecycle_automation_worker.py tests/test_security_lifecycle_disposition.py -q`
  - Final: `100 passed in 3.56s`.
- `pytest tests/test_security_lifecycle_fact_kernel.py -k 'producer_deadline_citation or blocker_citation_resolves_existing or blocker_citation_requires_canonical_deadline_dates' -q`
  - Final: `6 passed, 44 deselected`.
- `pytest tests/test_security_lifecycle_automation_scheduler.py tests/test_security_lifecycle_automation_worker.py tests/test_security_lifecycle_disposition.py tests/test_security_lifecycle_fact_kernel.py -q`
  - Final: `150 passed in 3.98s`.
- `pytest -q`
  - Single full run: `4802 passed, 12 skipped, 4 failed in 325.59s`.
  - All four failures came from `_deadline_owner` still declaring publisher
    requiredness. Replacing that stale fixture input with `listing_authority`
    resolved every affected test in the targeted run above.

All provider behavior in the added owners uses fake/injected transports and
sessions. No network or production database was used.

## Files

Product:

- `src/service/security_lifecycle_automation_scheduler.py`
- `src/security_lifecycle_automation_worker.py`
- `src/security_lifecycle_disposition.py`

Direct/adjacent owners:

- `tests/test_security_lifecycle_automation_scheduler.py`
- `tests/test_security_lifecycle_automation_worker.py`
- `tests/test_security_lifecycle_disposition.py`
- `tests/test_security_lifecycle_fact_kernel.py`

Historical `src/security_lifecycle_news_evidence.py`, schema, migrations, and
storage were unchanged.

## Self-review

- Confirmed `AUTOMATION_POLICY_VERSION` remains v4 and
  `AUTOMATION_EXECUTION_REVISION` remains `trusted-lifecycle-execution-r1`.
- Confirmed the scheduler active path has no news module import, news database
  opening, publisher blocker, publisher requiredness, or news diagnostic.
- Confirmed listing strings cross only `_blockers()` and persist with exact
  code/retryability readback for all nine codes.
- Confirmed diagnostic-only IBKR failures are omitted while ambiguity/conflict
  remain blocking.
- Confirmed legacy publisher facts and blockers cannot affect active source
  projection or queue selection.
- Confirmed failed-run replay and terminal finalization owners remain green in
  the expanded focused run.
- Confirmed no schema/migration, app restart, merge, push, or production DB work.

## Concerns

- The full backend suite was run exactly once as requested, but the adjacent
  stale-fixture correction occurred after that run. All four observed failures
  and the full Task 5 focused ownership set pass after the correction; there is
  no post-fix second full-suite result.
