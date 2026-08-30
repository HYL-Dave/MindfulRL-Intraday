# Task 3 Report: Version Per-Case Outcomes and Repair Recovery Witnesses

Date: 2026-08-30
Implementation: `b369308b`, review fix `0d8bd8f4`

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

- Initial Task 1 through Task 3 focused suites: `216 passed in 10.27s`.
- Review-fix suites, including the production supervisor boundary:
  `321 passed in 14.37s`.
- Compileall: clean.
- `git diff --check`: clean.

## Review Fix Round

The task reviewer identified four valid gaps and one interpretation that
conflicted with the binding spec.

- Production scheduler callers now use a run-and-record entry point that
  persists incident state before releasing the shared execution lock. The
  original unrecorded runner remains available to focused callers and tests.
- A new failed attempt for an already-active case updates the recovery marker
  without appending another semantically identical failure witness.
- Scheduler-generated empty, unavailable, skipped, and not-installed results
  now emit version 2; readers still accept version 1.
- Version-2 case IDs reject leading or trailing whitespace instead of silently
  normalizing it.
- The proposed rule that only `status=succeeded` may recover a scheduler-level
  incident was rejected. The spec defines recovery as a newer nonfailed
  attempt and separately defines a blocked attempt as completed and
  non-operational. Positive controls cover both failed-case-to-blocked and
  scheduler-failure-to-blocked recovery while the case blocker remains.

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
8. Remove version 2 from current empty producers: four producer owners failed.
9. Remove exact version-2 case-ID validation: the whitespace drift owner
   failed.
10. Compare marker-bearing incidents instead of semantic incident identity:
    the repeated-failure owner failed.
11. Disable recording in the lock-owned runner: the lock-order owner failed.

## Boundaries

No schema migration, provider call, production database access, App start,
merge, or push occurred. Existing false recovery history cannot be perfectly
reconstructed when its durable incident state never existed; the new envelope
owns all results recorded after this change.
