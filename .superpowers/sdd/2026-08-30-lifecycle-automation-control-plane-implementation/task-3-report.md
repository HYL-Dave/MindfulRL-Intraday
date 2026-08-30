# Task 3 Report: Version Per-Case Outcomes and Repair Recovery Witnesses

Date: 2026-08-30
Implementation: `b369308b`

## Result

- Worker summaries now use `result_version=2` with an exact
  `case_outcomes` map, including skipped cases.
- The aggregate validator rejects counter, selected-case, and outcome-map
  disagreement while retaining version-1 result parsing.
- Operational incidents are persisted under the existing
  `scheduler_state` table as a versioned envelope; no lifecycle schema or
  profile schema version changed.
- A failed case remains active when a later real tick merely skips its parked
  run. Recovery requires a newer nonfailed run or successful completion of the
  same run's pending finalization.
- A blocked decision is not classified as an operational failure witness.
- Version-1 mixed batches reconstruct the failed subset from current per-case
  run rows rather than treating every selected case as failed.

## RED

- A real worker sequence first persisted a failed run, then skipped that same
  semantic run on a healthy tick. The old recorder inserted a false recovery.
- A real blocked worker result was previously written as a failure witness.
- Version-2 counter/map drift cases were accepted before the closed map
  contract existed.
- Version-1 compatibility and malformed legacy inputs received explicit
  owners.
- Self-review added a mixed version-1 batch with one failed and one blocked
  run. The first implementation incorrectly kept both as active incidents;
  per-case reconstruction reduced it to the actual failed run.

## Verification

- Task 1 through Task 3 focused suites: `216 passed in 10.27s`.
- Compileall: clean.
- `git diff --check`: clean.

## Reverse Mutations

Each mutation was applied independently and restored:

1. Downgrade worker `result_version` to 1: the real worker owner failed.
2. Treat every partial result, including blocked, as operational failure: the
   blocked-case owner failed.
3. Treat an unchanged failed run as recovered on an empty/skipped tick: the
   failed-then-skipped owner failed.
4. Remove outcome/counter agreement checks: two drift parameters failed.
5. Reject unversioned version-1 blobs: the v1 compatibility owner failed.
6. Remove the scheduler-state existing-connection schema helper: its direct
   idempotency owner failed.
7. Revert v1 mixed-batch reconstruction to all selected IDs: the mixed
   failed/blocked owner failed.

## Boundaries

No schema migration, provider call, production database access, App start,
merge, or push occurred. Existing false recovery history cannot be perfectly
reconstructed when its durable incident state never existed; the new envelope
owns all results recorded after this change.
