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

## Review Round 1 Fix

Fix base: `e3bf04ecfb08e06bc8a86c25352f7ac14a93236c`.

Reviewer verdict read in full before the fix:

- `.superpowers/sdd/2026-08-30-lifecycle-automation-control-plane-implementation/task-6-review.md`

The fix addresses both Important findings and the Minor finding without
changing the scheduler product, worker product, schema, migrations, providers,
production database, or App code.

### Decisions

- Normalize every syntactically matched target and explicit predecessor before
  comparing evidence. Any normalization `ValueError` makes the sentence
  ambiguous evidence, so collection emits `sec_evidence_insufficient` and no
  active deadline.
- Compare normalized target semantics instead of grammar-match counts. Multiple
  matches are equivalent only when they resolve to one target and no more than
  one explicit predecessor. When equivalent matches mix forms, retain an
  explicit extension representative when available so predecessor proof is not
  discarded.
- Replace the fragile immediate coordinate check with a closed sentence rule:
  any date after the first matched target that is not a target or predecessor
  span of an accepted grammar match makes the sentence ambiguous. This rejects
  conditional alternate dates without depending on adjacent `and|or DATE`
  text.
- Fold the ordered filing chain with a set of proven `(OLD, NEW)` edges. A
  duplicate whose target is active replaces the selected row, preserving the
  later row's exact citation identity. A proven older edge recapped after a
  later extension is ignored as historical. An unseen edge whose predecessor
  is not active remains a branch and fails closed.
- Repeated bare extensions to the active target are idempotent and replace the
  selected row. A bare target already reached by a proven older edge may be
  ignored as a historical recap; an unseen non-forward target fails closed.
- No date is selected with `max()`. The resolver returns an original extracted
  row unchanged.

### RED Evidence

All fix-round tests were added before product code changed.

The first producer run was:

```text
pytest -q tests/test_security_lifecycle_sec_evidence.py \
  -k 'repeated_explicit_deadline_extension or repeated_bare_deadline_extension or duplicate_same_target_grammar or deadline_extension_recap_before_next or historical_deadline_edge_recap or conditional_alternate_deadline_target or invalid_syntactic_deadline_dates' \
  tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
FFFFFFF                                                                  [100%]
7 failed, 31 deselected in 0.60s
```

The failures were exact:

- repeated explicit `OLD -> NEW` produced
  `('sec_evidence_insufficient',)`;
- repeated bare `extended to NEW` produced
  `('sec_evidence_insufficient',)`;
- two current/outside grammar matches for the same normalized target produced
  `('sec_evidence_insufficient',)`;
- `OLD -> NEW`, recap `OLD -> NEW`, then `NEW -> NEXT` failed to resolve;
- `OLD -> NEW`, `NEW -> NEXT`, then recap `OLD -> NEW` failed to resolve;
- the conditional alternate target incorrectly selected `2026-08-30`; and
- `February 30, 2026` escaped as `ValueError`.

The `-k` expression intentionally selected the seven new producer owners but
also deselected the scheduler parameter cases, so the required real duplicate
producer-to-scheduler owner was run separately, still before product changes:

```text
pytest -q 'tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation[duplicate-extension]'
F                                                                        [100%]
E       AssertionError: assert [] == [('slc_deadline_supersession',
E         '2026-08-30T12:00:00Z', (), 3)]
1 failed in 0.45s
```

Duplicate SEC extension evidence therefore cleared the producer deadline,
made zero IBKR calls, and could not forward the later filing's exact citation.

The original invalid-date owner contained both invalid target and invalid
predecessor cases in a loop. Its RED stopped on the first escaped target
exception. During self-review it was split into two independent owners; the
invalid-predecessor path is independently proved by the reverse mutation below.

### GREEN Evidence

Direct final owners, including both scheduler parameter cases:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py::test_repeated_explicit_deadline_extension_is_idempotent_and_selects_latest_row \
  tests/test_security_lifecycle_sec_evidence.py::test_repeated_bare_deadline_extension_is_idempotent_and_selects_latest_row \
  tests/test_security_lifecycle_sec_evidence.py::test_duplicate_same_target_grammar_matches_are_idempotent \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_recap_before_next_extension_preserves_chain \
  tests/test_security_lifecycle_sec_evidence.py::test_historical_deadline_edge_recap_after_next_extension_is_ignored \
  tests/test_security_lifecycle_sec_evidence.py::test_conditional_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_invalid_syntactic_deadline_dates_fail_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_invalid_explicit_deadline_predecessor_fails_closed \
  tests/test_security_lifecycle_automation_scheduler.py::test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation
..........                                                               [100%]
10 passed in 0.47s
```

The duplicate scheduler case selects accession
`0000000001-26-000003`, forwards that row's evidence ID and byte-exact citation,
reaches the due IBKR path, and preserves `ibkr_max_queries=3`.

Final Task 6 focused gate:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_automation_worker.py
154 passed in 14.41s
```

Final broader lifecycle groups:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
228 passed in 14.89s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
147 passed in 3.47s
```

```text
pytest -q -p no:cacheprovider tests/test_security_lifecycle_investigation.py
30 passed in 3.90s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
158 passed in 15.30s
```

The four non-overlapping broader groups total `563 passed`, zero failures.

### Reverse Mutations

Every mutation was applied independently, run against public owners, restored,
and rerun GREEN before the next mutation.

#### 1. Reject duplicate same-target grammar

Mutation: restored the reviewed behavior that marks
`len(target_matches) > 1` ambiguous without semantic comparison.

```text
FAILED test_duplicate_same_target_grammar_matches_are_idempotent
E       AssertionError: assert ('sec_evidence_insufficient',) == ()
1 failed in 0.32s
```

Restored owner: `1 passed in 0.28s`.

#### 2. Reject an already-proven explicit edge

Mutation: changed the `edge in seen_edges` branch to `return None`.

```text
FAILED test_repeated_explicit_deadline_extension_is_idempotent_and_selects_latest_row
FAILED test_deadline_extension_recap_before_next_extension_preserves_chain
FAILED test_historical_deadline_edge_recap_after_next_extension_is_ignored
FAILED test_real_sec_deadline_supersession_reaches_due_ibkr_check_with_new_citation[duplicate-extension]
4 failed in 0.50s
```

The real seam again made zero IBKR calls. Restored owners:
`4 passed in 0.40s`.

#### 3. Reject a repeated bare active target

Mutation: changed the bare `row.date == active.date` branch from replacing the
active row to `return None`.

```text
FAILED test_repeated_bare_deadline_extension_is_idempotent_and_selects_latest_row
E       AssertionError: assert ('sec_evidence_insufficient',) == ()
1 failed in 0.32s
```

Restored owner: `1 passed in 0.28s`.

#### 4. Accept an unclaimed conditional date

Mutation: removed the sentence-wide unclaimed-date rejection block.

```text
FAILED test_conditional_alternate_deadline_target_fails_closed
E       AssertionError: active deadline was 2026-08-30 instead of ()
1 failed in 0.32s
```

Restored owner: `1 passed in 0.28s`.

#### 5. Let invalid source dates escape

Mutation: changed `except ValueError` around target/predecessor normalization to
`except RuntimeError`.

The preliminary combined owner failed with the expected escaped `ValueError`
(`1 failed in 0.36s`). After the self-review split, both independent paths
failed in the same mutation run:

```text
FAILED test_invalid_syntactic_deadline_dates_fail_closed
  ValueError: day is out of range for month
FAILED test_invalid_explicit_deadline_predecessor_fails_closed
  ValueError: day is out of range for month
2 failed in 0.42s
```

Restored owners: `2 passed in 0.28s`.

After the behavioral mutation cycle the producer checksum returned exactly to
`faa320207f7a66890c513657b70db67f9b0dc823e756da3fca8a42a371dc3b36`.
A subsequent behavior-neutral readability rewrite of the selected-match branch
produced final checksum
`67663b1805ff27b1d3e90e1689a60b3bbbe895aa498b2170ac31eae59e1463e8`,
after which every final gate above was rerun.

### Static Verification

```text
python -m compileall -q src/security_lifecycle_sec_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py \
  src/security_lifecycle_automation_worker.py
# exit 0, no output

git diff --check
# exit 0, no output

rg -n 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
745:        raise ValueError("source_deadlines")
911:        raise ValueError("source_deadlines")

rg -c 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
2
```

The active resolver contains no `max()` call. The scheduler source and worker
source have no fix-round diff. Task 5's `ibkr_max_queries` and
`ibkr_identity_blockers` interfaces are unchanged.

### Files and Commits

Implementation/test commit:

```text
35e1458b fix(lifecycle): make SEC deadline evidence idempotent
```

It changes exactly:

- `src/security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_automation_scheduler.py`

This report is the only file in the separate report commit. Its own hash is
necessarily recorded in the final task response rather than self-referenced in
the committed file.

### Self-Review

- Re-read the complete reviewer verdict, Task 6 brief, and binding design
  sections 2.1 and 8.1 against the final diff.
- Confirmed equivalent target evidence is idempotent while unseen explicit
  branches, contradictory current dates, orphan bare extensions, backward
  dates, reversed chronology, and conditional alternate targets remain
  fail-closed.
- Confirmed the active result is always an original row and the latest active
  duplicate's evidence ID, span, cited text, digest, rule ID, and rule version
  cross the real producer-to-scheduler seam unchanged.
- Confirmed `kind` and `supersedes_date` remain transient only; no persistence,
  DDL, fact type, schema, or migration changed.
- Confirmed both scheduler `ValueError("source_deadlines")` guards remain
  unchanged and no scheduler product edit was needed.
- Confirmed no provider, production database, App, merge, push, or subagent was
  used.

No Critical, Important, or Minor implementation issue remained after
self-review.

### Concerns

No known product concern. Provider behavior remains hermetically tested as
required by the task boundary. The environment does not have the Black module
installed, so formatting verification used the repository's existing style,
successful compilation, and `git diff --check`.

## Review Round 2 Fix

Fix base: `10ccb9b1fa2190ab375d4b5d79d09522e10d42a2`.

The complete `Round 2 Scoped Re-review` in `task-6-review.md` was read before
editing. The only product change is the remaining Important finding: replace
clause-order date counting with bounded semantic alternate-deadline detection.

### Decisions

- Add closed, sentence-level grammar for three deadline associations:
  a date governed by an `outside date` or `termination date` clause, a direct
  coordinated `or/and DATE`, and a bounded coordinated conditional or extension
  ending in `to/by DATE`.
- Stop deadline-clause scanning at `.` or `;` and cap each bounded scan at 160
  characters. This recognizes the reviewer's pre-target clause without treating
  arbitrary dates elsewhere in the sentence as deadline evidence.
- Keep grammar-matched predecessor and target spans claimed. Normalize only
  deadline-associated unclaimed dates, fail closed if one is invalid, and reject
  only normalized values that differ from the accepted target. Equivalent target
  evidence therefore remains idempotent.
- Remove `first_target_end` and the all-date `_ANY_MONTH_DATE` /
  `_ANY_ISO_DATE` scan from the ambiguity guard. Clause order and unrelated date
  count no longer determine whether an active deadline survives.
- Do not add a second large scheduler fixture for the invalid pre-target case.
  The public producer owner controls rejection, while the existing real
  producer-to-scheduler owner continues to control valid deadline/citation/IBKR
  behavior in the targeted and full gates.

### RED Evidence

Both owners were added before the Round 2 product edit:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py::test_preceding_conditional_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_unrelated_post_target_date_preserves_active_deadline
FF                                                                       [100%]
```

The negative owner exposed false admission:

```text
FAILED test_preceding_conditional_alternate_deadline_target_fails_closed
E       AssertionError: active deadline was 2026-08-30 instead of ()
```

The positive owner exposed unrelated-date over-rejection:

```text
FAILED test_unrelated_post_target_date_preserves_active_deadline
E       AssertionError: assert ('sec_evidence_insufficient',) == ()
2 failed in 0.40s
```

No product code had changed before this run.

### GREEN Evidence

The two direct owners first passed together:

```text
..                                                                       [100%]
2 passed in 0.28s
```

The final targeted control run covered both new owners, the existing suffix
conditional and immediate coordinate cases, explicit from/to claiming,
duplicate grammar and filing rows, recap chains, invalid dates, and both real
scheduler parameter cases:

```text
...............                                                          [100%]
15 passed in 0.48s
```

Final Task 6 focused gate:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_automation_worker.py
156 passed in 14.46s
```

Final broader lifecycle groups:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
230 passed in 14.90s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
147 passed in 3.49s
```

```text
pytest -q -p no:cacheprovider tests/test_security_lifecycle_investigation.py
30 passed in 3.71s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
158 passed in 15.29s
```

The four non-overlapping broader groups total `565 passed`, zero failures.

### Reverse Mutations

Both required mutations were applied independently to the final semantic
implementation, restored, and rerun GREEN.

#### 1. Admit the pre-target alternate

Mutation: removed `_DEADLINE_CLAUSE_DATE` from the bounded association pattern
set.

```text
FAILED test_preceding_conditional_alternate_deadline_target_fails_closed
E       AssertionError: active deadline was 2026-08-30 instead of ()
1 failed in 0.32s
```

Restored owner: `1 passed in 0.28s`.

#### 2. Restore unrelated-date over-rejection

Mutation: added `_ANY_MONTH_DATE` and `_ANY_ISO_DATE` to the association pattern
set, recreating an all-dates guard.

```text
FAILED test_unrelated_post_target_date_preserves_active_deadline
E       AssertionError: assert ('sec_evidence_insufficient',) == ()
1 failed in 0.32s
```

Restored owner: `1 passed in 0.28s`.

The producer checksum returned exactly to
`3814e41097c57ecaa3dccb3e24c6b05079bd3e1436cc3a5d0d54319dbcd996a2`
after the final mutation restoration.

### Static Verification

```text
python -m compileall -q src/security_lifecycle_sec_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py \
  src/security_lifecycle_automation_worker.py
# exit 0, no output

git diff --check
# exit 0, no output

rg -n 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
745:        raise ValueError("source_deadlines")
911:        raise ValueError("source_deadlines")

rg -c 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
2
```

The active resolver still contains no `max()` call. Scheduler and worker source,
scheduler/worker tests, persistence, DDL, facts, migrations, providers, and App
code have no Round 2 diff. Task 5's `ibkr_max_queries` and identity-marker
interfaces are unchanged.

### Files and Commits

Implementation/test commit:

```text
9b7b41f9 fix(lifecycle): classify alternate deadline dates
```

It changes exactly:

- `src/security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_sec_evidence.py`

This report is the only file in the separate report commit. Its hash is recorded
in the final task response rather than self-referenced in the committed file.

### Self-Review

- Re-read the Round 2 verdict and binding Task 6 rulings against the final diff.
- Confirmed the exact pre-target and suffix conditional alternates plus immediate
  `or DATE` fail closed with `sec_evidence_insufficient` and no active deadline.
- Confirmed the unrelated shareholder-meeting date preserves August 30 and its
  original full-sentence citation.
- Confirmed explicit OLD/NEW spans remain claimed, and normalized equivalent
  deadline-associated dates do not become false alternate targets.
- Confirmed duplicate idempotency, branches, chronology, invalid dates, exact
  selected citation identity, the valid real scheduler seam, and
  `ibkr_max_queries=3` remain covered and GREEN.
- Confirmed both scheduler tripwires are unchanged and no scheduler product edit
  was needed.
- Confirmed no provider, production database, App, migration, merge, push, or
  subagent was used.

No Critical, Important, or Minor implementation issue remained after
self-review.

### Concerns

No known product concern. The deadline association grammar is intentionally
closed and bounded; phrasing outside those accepted forms is not interpreted,
which is a deliberate residual coverage limit rather than an open grammar.
Provider behavior remains hermetically tested as required by the task boundary.

## Review Round 3 Fix

Base: `ebb095ec57a29816f44b71251f581fabdac00033`

The Round 3 Important finding was reproduced before product changes. The three
phrase-oriented association regexes admitted complete modal alternatives only
in some word orders and also treated unrelated nearby dates as deadline
targets. This round replaces those regexes with bounded classification of
syntactically complete semantic deadline targets.

### Decisions

- Keep the existing accepted current, termination-condition, and extension
  grammar as the source of accepted evidence rows.
- Classify an additional candidate only when it is either an explicit
  `outside date|termination date` assertion with a closed modality or a
  syntactically complete coordinated deadline continuation.
- Bound explicit assertions to `is|shall be|remains|may be|could be|would be|will
  be DATE` and coordinated continuations to the ruled forms: modal `or|but`,
  bare `or DATE`, `and further extended to DATE`, and bounded conditional
  `or, if|unless|provided that|subject to ..., to|by DATE`.
- Require a coordinated candidate to terminate at the date apart from
  punctuation or closed coordination. A date followed by an unrelated subject
  and predicate is therefore context, not a deadline target.
- Compare normalized semantic values. A distinct candidate target fails closed;
  an equivalent target mention is idempotent.
- Continue claiming both explicit `from OLD` and `to NEW` spans so the
  predecessor is not mistaken for an alternate target.
- Preserve citation selection, ordered collapse, invalid-date handling, the
  public producer-to-scheduler seam, both scheduler multi-date tripwires, and
  Task 5 interfaces without scheduler or persistence changes.

The removed patterns are `_DEADLINE_CLAUSE_DATE`,
`_DIRECT_COORDINATED_DEADLINE_DATE`, and
`_COORDINATED_DEADLINE_ACTION_DATE`. No generic nearby-date pattern was added.

### RED Evidence

The six Round 3 owners were added first and run before changing product code:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py::test_or_may_be_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_but_may_be_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_provided_that_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_coordinated_date_with_unrelated_predicate_preserves_deadline \
  tests/test_security_lifecycle_sec_evidence.py::test_coordinated_by_date_with_unrelated_predicate_preserves_deadline \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_discussion_date_is_not_a_target
FFFFFF                                                                   [100%]
6 failed in 0.46s
```

The `or may be`, `but may be`, and `provided that` cases incorrectly retained
August 30. The unrelated coordinated date, coordinated `by` date, and discussion
date cases incorrectly emitted `sec_evidence_insufficient`. This established
both under-classification and over-classification REDs against the Round 2
product.

### GREEN Evidence

The six new owners passed together after the bounded classifier change:

```text
......                                                                   [100%]
6 passed in 0.29s
```

The targeted control run contained the six new owners; the existing pre-target,
suffix conditional, immediate `or DATE`, and scheduled-for controls; explicit
OLD/NEW claiming; duplicate and recap chains; invalid dates; and the real
producer-to-scheduler parameter owner:

```text
.....................                                                    [100%]
21 passed in 0.50s
```

Final Task 6 focused gate:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_automation_worker.py
162 passed in 14.37s
```

Final broader lifecycle groups:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
236 passed in 14.80s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
147 passed in 3.45s
```

```text
pytest -q -p no:cacheprovider tests/test_security_lifecycle_investigation.py
30 passed in 3.55s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
158 passed in 15.30s
```

The four non-overlapping broader groups total `571 passed`, zero failures.

### Reverse Mutations

All mutations were applied independently to the final implementation, proved by
the named public owners, restored, and rerun GREEN.

#### 1. Disable modal alternate classification

Mutation: changed the coordinated modal branch from `(?:or|but)` to an
unmatched token.

```text
FAILED test_or_may_be_alternate_deadline_target_fails_closed
FAILED test_but_may_be_alternate_deadline_target_fails_closed
2 failed in 0.34s
```

Both mutations incorrectly retained August 30. Restored owners:
`2 passed in 0.28s`.

#### 2. Disable `provided that` conditional classification

Mutation: removed `provided that` from the bounded conditional keyword set.

```text
FAILED test_provided_that_alternate_deadline_target_fails_closed
1 failed in 0.32s
```

The mutation incorrectly retained August 30. Restored owner:
`1 passed in 0.28s`.

#### 3. Reintroduce unrelated-date classification

Mutation: temporarily added `_ANY_MONTH_DATE` and `_ANY_ISO_DATE` to the
candidate-target pattern set.

```text
FAILED test_coordinated_date_with_unrelated_predicate_preserves_deadline
FAILED test_coordinated_by_date_with_unrelated_predicate_preserves_deadline
FAILED test_deadline_discussion_date_is_not_a_target
FAILED test_unrelated_post_target_date_preserves_active_deadline
4 failed in 0.37s
```

The mutation rejected all four unrelated-date positive controls. Restored
owners: `4 passed in 0.28s`.

The producer checksum before mutations and after all restorations was identical:

```text
3c97f1182007e9433352b4a3c3c835813108feeb8045f6cd11e53a97b99f9478
```

### Static Verification

```text
python -m compileall -q src/security_lifecycle_sec_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py \
  src/security_lifecycle_automation_worker.py
# exit 0, no output

git diff --check
# exit 0, no output

rg -n 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
745:        raise ValueError("source_deadlines")
911:        raise ValueError("source_deadlines")

rg -c 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
2
```

The resolver still contains no `max()` date selection. The three replaced
phrase-chasing pattern names are absent. Scheduler and worker product/tests,
persistence, DDL, facts, migrations, providers, production database, and App
code have no Round 3 diff. Task 5's `ibkr_max_queries` and identity-marker
interfaces remain unchanged.

### Files and Commits

Implementation/test commit:

```text
7194d163730079da82fb800dac55c0f52f045d04 fix(lifecycle): classify complete deadline targets
```

It changes exactly:

- `src/security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_sec_evidence.py`

This report is the only file in the separate report commit. Its hash is recorded
in the final task response rather than self-referenced in the committed file.

### Self-Review

- Re-read the Round 3 verdict and checked each full month-date example against
  the final public producer behavior.
- Confirmed modal, conditional, pre-target, suffix, and immediate alternate
  targets fail closed, while unrelated predicates, `by` context, discussion
  context, and scheduled-for context preserve August 30.
- Confirmed candidate continuations must end at their date except for punctuation
  or closed coordination; no all-dates count or clause-order heuristic remains.
- Confirmed explicit OLD and NEW spans remain claimed, and equivalent normalized
  target mentions remain idempotent.
- Confirmed duplicate/recap ordering, branches, chronology, invalid dates, exact
  selected-row citation identity, and the real scheduler seam remain GREEN.
- Confirmed both scheduler tripwires are unchanged and no scheduler product
  change was required.
- Confirmed no provider, production database, App, migration, merge, push, or
  subagent was used.

No Critical, Important, or Minor implementation issue remained after
self-review.

### Concerns

No known product concern. The semantic target grammar is intentionally closed
and bounded; deadline phrasing outside the ruled forms remains unproved evidence
rather than being inferred from proximity. This is the deliberate residual
coverage limit required by the task.

## Review Round 4 Fix

Base: `a3b8ad0338ab89610127184fcf59691a1db20c5a`

The complete `Round 4 Scoped Re-review` in `task-6-review.md` was read before
editing. The remaining two Important findings had one mechanical root cause:
the complete coordinated-target regex assigned different target forms to
separate hard-coded coordinator branches.

### Decisions

- Give `and|or|but` one shared coordinator prefix followed by optional comma
  spacing and exactly one closed target form: modality, direct `to|by`, further
  extension, bounded conditional, or bare date.
- Represent the bare-date alternative with a date lookahead so it participates
  in the same closed alternation without making the other target forms optional.
- Keep `_DEADLINE_TARGET_END` after the one captured target date. A coordinated
  target is therefore classifiable only when its clause terminates at that date;
  trailing shareholder-meeting and mailing predicates remain unrelated context.
- Preserve the accepted bare `or`, `or|but may be`, `and further extended to`,
  and `or if|provided ... to|by` forms through the same grammar rather than
  retaining coordinator-specific branches.
- Leave explicit target assertions, accepted evidence extraction, normalized
  comparison, ordered deadline collapse, citation identity, scheduler/worker
  behavior, and persistence untouched.

### RED Evidence

The two coordinator-specific owners were added before product code changed:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py::test_and_coordinated_deadline_targets_fail_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_but_conditional_deadline_targets_fail_closed
FF                                                                       [100%]
2 failed in 0.41s
```

The `and` owner failed first on terminal bare `and September 1, 2026`; the
producer incorrectly retained the August 30 extension. It also owns the
comma-before-`and` variant and `and may be September 1, 2026`.

The independent `but` owner failed first on `but, if ... to September 1, 2026`;
the producer again retained August 30. It also owns
`but, provided that ... by September 1, 2026`.

### GREEN Evidence

The two new owners passed together after the unified grammar change:

```text
..                                                                       [100%]
2 passed in 0.28s
```

The retained-branch and target-boundary control run included bare `or`,
`or|but may be`, `and further extended to`, `or if|provided`, both new owners,
and the shareholder-meeting and meeting-materials positive controls:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_closed_grammar_emits_only_one_current_target_and_exact_citation \
  tests/test_security_lifecycle_sec_evidence.py::test_deadline_extension_with_two_targets_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_conditional_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_or_may_be_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_but_may_be_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_and_coordinated_deadline_targets_fail_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_but_conditional_deadline_targets_fail_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_provided_that_alternate_deadline_target_fails_closed \
  tests/test_security_lifecycle_sec_evidence.py::test_coordinated_date_with_unrelated_predicate_preserves_deadline \
  tests/test_security_lifecycle_sec_evidence.py::test_coordinated_by_date_with_unrelated_predicate_preserves_deadline
..........                                                               [100%]
10 passed in 0.30s
```

Both scheduler tripwire owners passed directly:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_automation_scheduler.py::test_pending_event_monitoring_rejects_multiple_deadline_dates \
  tests/test_security_lifecycle_automation_scheduler.py::test_acquisition_scheduling_rejects_multiple_deadline_dates_before_market_work
..                                                                       [100%]
2 passed in 0.46s
```

Final Task 6 focused gate:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_automation_worker.py
164 passed in 8.66s
```

Final broader lifecycle groups:

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
238 passed in 9.14s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
147 passed in 2.44s
```

```text
pytest -q -p no:cacheprovider tests/test_security_lifecycle_investigation.py
30 passed in 1.96s
```

```text
pytest -q -p no:cacheprovider \
  tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
158 passed in 10.78s
```

The four non-overlapping broader groups total `573 passed`, zero failures.

### Reverse Mutations

Each coordinator mutation was applied independently to the unified grammar,
run against its dedicated public owner, restored, and rerun GREEN.

#### 1. Remove `and` from the shared coordinator prefix

Mutation: changed `(?:and|or|but)` to `(?:or|but)`.

```text
FAILED test_and_coordinated_deadline_targets_fail_closed
E       AssertionError: The outside date was extended from August 28, 2026
E       to August 30, 2026 and September 1, 2026.
1 failed in 0.32s
```

The mutation incorrectly retained August 30. Restored owner:
`1 passed in 0.28s`.

#### 2. Remove `but` from the shared coordinator prefix

Mutation: changed `(?:and|or|but)` to `(?:and|or)`.

```text
FAILED test_but_conditional_deadline_targets_fail_closed
E       AssertionError: The outside date was extended from August 28, 2026
E       to August 30, 2026 but, if regulatory approval remained outstanding,
E       to September 1, 2026.
1 failed in 0.32s
```

The mutation incorrectly retained August 30. Restored owner:
`1 passed in 0.28s`.

The extractor checksum before mutations and after both restorations was
identical:

```text
001932d8ef58df3d69a18b98126d7dfedd4df1a62c936938a441019c987ec696
```

### Static Verification

```text
python -m compileall -q src/security_lifecycle_sec_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py \
  src/security_lifecycle_automation_worker.py
# exit 0, no output

git diff --check
# exit 0, no output

rg -n 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
745:        raise ValueError("source_deadlines")
911:        raise ValueError("source_deadlines")

rg -c 'raise ValueError\("source_deadlines"\)' \
  src/service/security_lifecycle_automation_scheduler.py
2

git show --check --oneline 65b190f4
65b190f4 fix(lifecycle): unify coordinated deadline targets
```

Scheduler and worker source/tests, persistence, DDL, facts, migrations,
providers, production database, and App code have no Round 4 implementation
diff. Task 5 interfaces and the ordered Task 6 resolver are unchanged.

### Files and Commits

Implementation/test commit:

```text
65b190f45a466a87cc46a28a1e7715493774d8bb fix(lifecycle): unify coordinated deadline targets
```

It changes exactly:

- `src/security_lifecycle_sec_evidence.py`
- `tests/test_security_lifecycle_sec_evidence.py`

This report is the only file in the separate report commit. Its hash is recorded
in the final task response rather than self-referenced in the committed file.

### Self-Review

- Re-read the Round 4 verdict and verified every requested `and` and `but`
  sentence through the public producer.
- Confirmed the target-form alternation contains no coordinator-specific branch;
  `and|or|but` is selected once before all five closed forms.
- Confirmed `_DEADLINE_TARGET_END` remains after the captured date and both
  trailing-predicate positive controls preserve August 30 and exact citation.
- Confirmed retained bare, modality, further-extension, and conditional forms,
  prior duplicate/invalid-date behavior, the real scheduler seam, and both
  scheduler tripwires remain GREEN in the focused and broader gates.
- Confirmed no scheduler, worker, provider, production database, App, migration,
  merge, push, or subagent operation occurred.

No Critical, Important, or Minor implementation issue remained after
self-review.

### Concerns

No known product concern. The grammar remains intentionally closed: unsupported
target wording is not inferred, and coordinated dates followed by unrelated
predicates remain excluded by `_DEADLINE_TARGET_END`.
