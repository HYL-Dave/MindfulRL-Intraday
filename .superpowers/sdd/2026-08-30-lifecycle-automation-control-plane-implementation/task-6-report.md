# Task 6 Report: Preserve Resolvable SEC Deadline Supersession

Date: 2026-08-30
Base: `05e678f8b19c43af993459d5ec23284675a2b3f2`
Implementation: `0db47994b0fa6a7296afbba45d5aac85d47720b4`

## Status

DONE. Task 6 is implemented, tested, self-reviewed, and committed on
`lifecycle-automation-control-plane`.

No live SEC, listing-authority, or IBKR provider was called. No production DB,
App, schema migration, merge, or push operation occurred. Test persistence used
only pytest-owned temporary SQLite databases.

## Result

- `SecSourceDeadline` now carries transient `kind` and `supersedes_date`
  extraction metadata. Existing constructors remain compatible through
  defaults; every extracted row supplies its actual metadata explicitly.
- Deadline extraction distinguishes `current`, `termination_condition`, and
  `extension`. Explicit `extended from OLD to NEW` text records normalized OLD
  in `supersedes_date`; a bare extension records no predecessor.
- The producer resolves rows in existing filing-chain and sentence order. It
  never sorts or selects by maximum date.
- An explicit extension may establish its stated OLD predecessor by itself. If
  an active predecessor already exists, it must match OLD exactly.
- A bare extension is accepted only when one active predecessor is already
  established by the ordered chain.
- Every accepted extension must move strictly forward in calendar time.
- Multiple coordinate targets, two branch targets from one predecessor,
  contradictory current dates, orphan bare extensions, filing-order reversal,
  and backward extensions emit `sec_evidence_insufficient` with no active
  deadline.
- Successful collapse returns the selected extension row object itself. Its
  evidence ID, byte span, cited-text digest, rule ID, and rule version are not
  reconstructed or changed.
- The real `collect_sec_evidence -> _load_evidence` owner retains NEW, triggers
  the due IBKR path, forwards `ibkr_max_queries=3`, and emits NEW's exact
  citation into the nonretryable `not_confirmed_as_of` blocker.
- The worker readback owner proves `kind` and `supersedes_date` do not become
  persisted blocker fields or automation facts.
- Both existing scheduler `ValueError("source_deadlines")` tripwires remain
  byte-for-byte present at lines 745 and 911. No scheduler product change was
  needed because the consumer's due-date, citation, query-cap, and defensive
  guard behavior was already correct.
- Task 5's `ibkr_max_queries` and `ibkr_identity_blockers` interfaces are
  unchanged.

## RED Evidence

### 1. First RED: real producer-to-scheduler owner

This was the first file change and first test run. No product code had changed.

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
F                                                                        [100%]
E       AssertionError: assert [] == [('slc_deadline_supersession', '2026-08-30T12:00:00Z', (), 3)]
FAILED tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
1 failed in 0.51s
```

The real producer cleared OLD and NEW, so the scheduler had no due date and
made no IBKR call.

### 2. Producer metadata, ordered collapse, and fail-closed REDs

Command: the nine named Task 6 owners in the SEC and scheduler test files.

```text
FFFF.F..F                                                                [100%]
FAILED test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation
  bare extension still emitted ('2026-06-01',)
FAILED test_deadline_rows_carry_transient_extraction_metadata
  AttributeError: 'SecSourceDeadline' object has no attribute 'kind'
FAILED test_ordered_explicit_deadline_extension_selects_new_row_unchanged
  blockers was ('sec_evidence_insufficient',)
FAILED test_deadline_extension_with_two_targets_fails_closed
  sec_evidence_insufficient was absent
FAILED test_bare_deadline_extension_without_predecessor_fails_closed
  the orphan extension remained active
FAILED test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
  ibkr_calls was []
6 failed, 3 passed in 0.57s
```

The three passing controls were the pre-existing broad-clear behavior for
contradictory dates and the two reversed-chronology cases. They were added
before implementation to prevent a later `max()` repair from converting that
accidental fail-closed behavior into false certainty.

### 3. Bare extension with an ordered predecessor

```text
pytest -q tests/test_security_lifecycle_sec_evidence.py::test_ordered_bare_deadline_extension_uses_single_current_predecessor
F                                                                        [100%]
E       AssertionError: assert ('sec_evidence_insufficient',) == ()
FAILED tests/test_security_lifecycle_sec_evidence.py::test_ordered_bare_deadline_extension_uses_single_current_predecessor
1 failed in 0.37s
```

## GREEN Evidence

### Direct Task 6 owners

```text
pytest -q tests/test_security_lifecycle_sec_evidence.py::test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_rows_carry_transient_extraction_metadata \
  tests/test_security_lifecycle_sec_evidence.py::test_ordered_explicit_deadline_extension_selects_new_row_unchanged \
  tests/test_security_lifecycle_sec_evidence.py::test_ordered_bare_deadline_extension_uses_single_current_predecessor \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_with_two_targets_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_contradictory_current_deadlines_fail_closed_without_date_selection \
  tests/test_security_lifecycle_sec_evidence.py::test_bare_deadline_extension_without_predecessor_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_before_its_predecessor_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_cannot_move_backward_in_time \
  tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
..........                                                               [100%]
10 passed in 0.46s
```

The same-predecessor/two-target and transient worker owners were added during
mutation review and pass in the final focused and lifecycle gates below.

### Final Task 6 files

```text
pytest -q tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_automation_worker.py
........................................................................ [ 49%]
........................................................................ [ 99%]
.                                                                        [100%]
145 passed in 8.42s
```

### All lifecycle tests

Every `tests/test_security_lifecycle*.py` node ran in four isolated groups to
avoid relying on the previously recorded monolithic ordering probe.

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
219 passed in 9.05s
```

```text
pytest -q tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
147 passed in 2.44s
```

```text
pytest -q tests/test_security_lifecycle_investigation.py
30 passed in 1.94s
```

```text
pytest -q tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
158 passed in 10.76s
```

Total: `554 passed`, zero failures.

## Reverse Mutations

Every mutation was applied independently and restored before the next run.
Each restored owner was rerun GREEN.

### 1. Restore the old multi-date producer clear

Mutation: `_resolve_source_deadline` returned `None` whenever more than one
distinct row date existed.

```text
FAILED test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
E       AssertionError: assert [] == [('slc_deadline_supersession', '2026-08-30T12:00:00Z', (), 3)]
1 failed in 0.47s
```

This is the required independent proof that the public producer-to-scheduler
owner, not a scheduler stub, catches broken producer collapse.

### 2. Select `max()` date

Mutation: resolver body replaced with
`return max(rows, key=lambda row: row.date, default=None)`.

```text
FFFF                                                                     [100%]
FAILED test_contradictory_current_deadlines_fail_closed_without_date_selection
FAILED test_bare_deadline_extension_without_predecessor_fails_closed
FAILED test_deadline_extension_before_its_predecessor_fails_closed
FAILED test_deadline_extension_cannot_move_backward_in_time
4 failed in 0.36s
```

Each failure exposed an incorrectly active row; this directly proves dates are
not selected by maximum value.

### 3. Retain OLD after a valid extension

Mutation: changed `active = row` to `active = active or row`.

```text
FAILED test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
E       AssertionError: assert '2026-08-28' == '2026-08-30'
1 failed in 0.45s
```

The public seam therefore owns both NEW selection and NEW citation routing.

### 4. Ignore ambiguous coordinate targets

Mutation: removed `deadline_ambiguous` from the producer rejection predicate.

```text
FAILED test_deadline_extension_with_two_targets_fails_closed
E       AssertionError: assert 'sec_evidence_insufficient' in ()
1 failed in 0.32s
```

### 5. Rebind a mismatched explicit predecessor

Mutation: when active date and `supersedes_date` differed, silently replaced
the stated predecessor with the active date instead of rejecting the chain.

```text
FAILED test_deadline_chain_with_two_extension_targets_fails_closed
E       AssertionError: active deadline was 2026-09-01 instead of ()
1 failed in 0.32s
```

### 6. Permit backward extension dates

Mutation: weakened the forward chronology guard from `<=` to `==`.

```text
FAILED test_deadline_extension_cannot_move_backward_in_time
E       AssertionError: active deadline was 2026-08-28 instead of ()
1 failed in 0.32s
```

### 7. Persist transient supersession metadata

Mutation: added `kind` and `supersedes_date` to the scheduler blocker context,
which the worker then persisted.

```text
FAILED test_deadline_supersession_metadata_is_transient_through_worker_persistence
E       AssertionError: assert 'kind' not in {'kind': 'extension', ...}
1 failed in 0.45s
```

## Static Verification

```text
python -m compileall -q src/security_lifecycle_sec_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py \
  src/security_lifecycle_automation_worker.py
# exit 0, no output

git diff --check
# exit 0, no output

test "$(rg -c 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py)" = 2
# exit 0, no output

git show --check --oneline 0db47994
0db47994 fix(lifecycle): preserve SEC deadline supersession
```

## Files

Implementation commit `0db47994` changed exactly:

- `src/security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_automation_scheduler.py`
- `tests/test_security_lifecycle_automation_worker.py`

The scheduler source file named by the implementation plan was inspected but
not changed. Its existing due-IBKR calculation, exact citation forwarding,
`ibkr_max_queries` seam, identity marker handling, and two defensive guards
already implement the required consumer contract. The Task 6 change belongs
to the public SEC producer.

This report is the only additional Task 6 file.

## Self-Review

- Re-read design sections 2.1 and 8.1 and the complete Task 6 brief against the
  committed diff.
- Confirmed the producer uses ordered folding and contains no `max()` date
  selection.
- Confirmed the selected deadline is an original extracted row and all exact
  citation fields cross the public seam.
- Confirmed every specified fail-closed case emits
  `sec_evidence_insufficient` and no active deadline.
- Confirmed both scheduler multi-date checks remain and both direct guard tests
  pass.
- Confirmed `ibkr_max_queries=3` crosses the new public owner and the full Task
  5 identity/query tests pass in the 219-test group.
- Confirmed no schema, migration, durable fact type, worker product code, or
  scheduler product code changed.
- Confirmed committed scope and whitespace with `git show --check`.

No Critical, Important, or Minor issue remained after self-review.

## Concerns

No known implementation concern. Per the task boundary, provider behavior was
validated hermetically rather than through live SEC, listing, or IBKR access.
