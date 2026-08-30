# Task 5 Report: IBKR Identity Planning Without False Absence

Date: 2026-08-30
Implementation: `07e99f16` (`fix(lifecycle): bound IBKR identity planning`)
Base: `f224adce`

## Status

DONE_WITH_CONCERNS. Task 5 is implemented and committed. No live IBKR or
Gateway call, production DB read/write, App operation, schema migration,
merge, or push occurred.

One supplemental monolithic wildcard test ordering probe stalled after
463/539 nodes. It was interrupted rather than reported as green. The exact
same 539 collected lifecycle nodes passed in four isolated groups documented
below. This is a test-order/harness residual, not a Task 5 behavioral failure.

## Root Cause Confirmed

The producer/consumer mismatch had two independent layers:

1. `build_identity_context` admitted the current ticker, a stable alias
   closure, local conIds, and regulator successors. The IBKR adapter then
   raised `ibkr_identity_candidates_exceed_max_queries` when the complete set
   exceeded its default eight-query budget. Six additional aliases plus the
   current ticker and one successor fit exactly; seven additional aliases
   raised before a typed result could be returned.
2. `_load_local_identity_hints` built aliases and conIds for every case as one
   batch. A 65-name closure raised from `_load_cases`, so one case prevented
   every later case from reaching worker selection.

The old adapter could also query multiple local conIds and, when all returned
empty, persist `contract_missing`. That absence was not based on a single
exact identity and therefore was not complete evidence.

## Result

- Candidate planning is now deterministic:
  1. one exact known conId;
  2. current ticker;
  3. regulator successor ticker(s) in supplied stable order;
  4. remaining aliases in canonical stable order.
- Multiple known conIds return `ibkr_contract_ambiguous` before the Gateway
  lock or provider is touched.
- A candidate plan that cannot fit `max_queries` returns the same typed
  ambiguity with `requests_made=0`, no evidence, and no
  `ibkr_contract_missing` receipt.
- A complete plan that fits the active budget and returns no details still
  emits a genuine typed `ibkr_contract_missing` receipt.
- Alias traversal is bounded independently per requested ticker. A closure or
  edge overflow marks only that ticker ambiguous and retains a safe current
  ticker placeholder; later ticker hints are still loaded.
- Portfolio conIds are resolved independently per ticker. More than one
  distinct conId, or a per-ticker row overflow, marks only that ticker
  ambiguous.
- Scheduler case rows carry the internal closed
  `ibkr_identity_blockers=("ibkr_contract_ambiguous",)` marker. When an IBKR
  check is required, that marker becomes a nonretryable typed blocker without
  constructing the IBKR Gateway adapter.
- `_load_evidence` accepts an internal `ibkr_max_queries` value and forwards it
  through `_ibkr_evidence`; the production default remains exactly 8.
- Existing decision-policy filtering remains unchanged: only
  `contract_status == "found"` market evidence enters automatic decision
  material. A real missing receipt now has a grounded-shadow owner for this
  invariant.
- No durable ticker/source ID, lifecycle DDL/version, policy version,
  decision provenance, Task 4 lock/retry behavior, or profile-mutation
  authority changed.

## Baseline

Command before adding Task 5 tests or product code:

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py
```

Output:

```text
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 4.03s
```

## RED Evidence

### Adapter plan, capacity, and conId admission

Command:

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  -k 'candidate_plan or candidate_budget or multiple_known_conids'
```

Initial output:

```text
FFF                                                                      [100%]
FAILED test_ibkr_candidate_plan_prioritizes_exact_current_successor_then_aliases
  At index 2 diff: (0, 'AAA') != (0, 'NEXT')
FAILED test_ibkr_candidate_budget_distinguishes_complete_missing_from_ambiguity
  ValueError: ibkr_identity_candidates_exceed_max_queries
FAILED test_multiple_known_conids_are_ambiguous_before_provider_access
  AssertionError: assert 'missing' == 'ambiguous'
3 failed, 14 deselected in 0.37s
```

This proved three separate breaks: successor priority was below aliases,
budget overflow escaped as an exception, and multiple conIds could produce a
false missing receipt.

### Per-case hint containment, scheduler cap seam, and provider exclusion

Command:

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py \
  -k 'alias_closure_overflow or multiple_local_conids or injected_query_cap or precomputed_ibkr_ambiguity' \
  tests/test_security_lifecycle_grounded_shadow.py \
  -k 'alias_closure_overflow or multiple_local_conids or injected_query_cap or precomputed_ibkr_ambiguity or contract_missing_receipt'
```

Initial output:

```text
FFFF.                                                                    [100%]
FAILED test_alias_closure_overflow_is_a_per_case_ibkr_ambiguity
  ValueError: ticker_aliases_exceed_limit
FAILED test_multiple_local_conids_do_not_poison_a_later_case
  ibkr_conids was (101, 202); no typed identity blocker existed
FAILED test_scheduler_ibkr_seam_forwards_an_injected_query_cap
  TypeError: _ibkr_evidence() got an unexpected keyword argument 'max_queries'
FAILED test_precomputed_ibkr_ambiguity_never_reaches_the_gateway
  Failed: ambiguous identity reached IBKR
4 failed, 1 passed, 62 deselected in 0.84s
```

The one passing control was the pre-existing decision-policy behavior that
excluded a genuine missing receipt. It was subsequently strengthened so a
mutation that admits missing evidence into current decision material fails.

## GREEN Evidence

### Eight direct Task 5 owners

```text
pytest -q \
  tests/test_security_lifecycle_ibkr_evidence.py::test_ibkr_candidate_plan_prioritizes_exact_current_successor_then_aliases \
  tests/test_security_lifecycle_ibkr_evidence.py::test_ibkr_candidate_budget_distinguishes_complete_missing_from_ambiguity \
  tests/test_security_lifecycle_ibkr_evidence.py::test_multiple_known_conids_are_ambiguous_before_provider_access \
  tests/test_security_lifecycle_automation_scheduler.py::test_alias_closure_overflow_is_a_per_case_ibkr_ambiguity \
  tests/test_security_lifecycle_automation_scheduler.py::test_multiple_local_conids_do_not_poison_a_later_case \
  tests/test_security_lifecycle_automation_scheduler.py::test_scheduler_ibkr_seam_forwards_an_injected_query_cap \
  tests/test_security_lifecycle_automation_scheduler.py::test_precomputed_ibkr_ambiguity_never_reaches_the_gateway \
  tests/test_security_lifecycle_grounded_shadow.py::test_contract_missing_receipt_is_excluded_from_automatic_decision_material
........                                                                 [100%]
8 passed in 0.57s
```

The later-case worker continuation owner was added immediately afterward and
is included in the focused and adjacent gates below.

### Task 5 focused files

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py
........................................................................ [ 84%]
.............                                                            [100%]
85 passed in 4.78s
```

### Focused plus decision and worker consumers

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py
........................................................................ [ 39%]
........................................................................ [ 78%]
........................................                                 [100%]
184 passed in 8.29s
```

### Schema, runtime, disposition, and fact-kernel adjacency

```text
pytest -q tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
........................................................................ [ 48%]
........................................................................ [ 97%]
...                                                                      [100%]
147 passed in 2.44s
```

### Remaining lifecycle modules

```text
pytest -q tests/test_security_lifecycle_investigation.py
..............................                                           [100%]
30 passed in 1.97s
```

```text
pytest -q tests/test_security_lifecycle_listing_evidence.py \
  tests/test_security_lifecycle_listing_migration.py \
  tests/test_security_lifecycle_manual_evidence.py \
  tests/test_security_lifecycle_migration.py \
  tests/test_security_lifecycle_news_evidence.py \
  tests/test_security_lifecycle_routes.py \
  tests/test_security_lifecycle_schema.py \
  tests/test_security_lifecycle_sec_evidence.py \
  tests/test_security_lifecycle_tools.py \
  tests/test_security_lifecycle_translation.py
........................................................................ [ 40%]
........................................................................ [ 80%]
..................................                                       [100%]
178 passed in 11.41s
```

These four groups contain all 539 nodes collected by
`tests/test_security_lifecycle*.py`.

## Reverse Controls

Every mutation below was applied independently and restored before final
GREEN verification.

### 1. Put aliases before the regulator successor

Mutation: candidate sequence changed from current/successor/aliases back to
current/aliases/successor.

```text
FAILED test_ibkr_candidate_plan_prioritizes_exact_current_successor_then_aliases
At index 2 diff: (0, 'AAA') != (0, 'NEXT')
1 failed in 0.31s
```

### 2. Truncate an over-budget plan and claim missing

Mutation: `return None` on overflow changed to
`return queries[:max_queries]`.

```text
FAILED test_ibkr_candidate_budget_distinguishes_complete_missing_from_ambiguity
AssertionError: assert 'missing' == 'ambiguous'
1 failed in 0.31s
```

### 3. Allow multiple direct conIds

Mutation: adapter ambiguity threshold changed from `> 1` to `> 99`.

```text
FAILED test_multiple_known_conids_are_ambiguous_before_provider_access
AssertionError: assert 'missing' == 'ambiguous'
1 failed in 0.31s
```

### 4. Restore whole-batch alias overflow failure

Mutation: the per-ticker overflow branch raised
`ticker_aliases_exceed_limit` instead of recording ambiguity.

```text
FAILED test_alias_closure_overflow_is_a_per_case_ibkr_ambiguity
ValueError: ticker_aliases_exceed_limit
1 failed in 0.55s
```

### 5. Allow multiple locally discovered conIds

Mutation: scheduler local conId ambiguity threshold changed from `> 1` to
`> 2`.

```text
FAILED test_multiple_local_conids_do_not_poison_a_later_case
ibkr_conids was (101, 202); ibkr_identity_blockers was ()
1 failed in 0.49s
```

### 6. Bypass the precomputed ambiguity guard

Mutation: the marker branch was disabled so `_ibkr_evidence` was called.

```text
FAILED test_precomputed_ibkr_ambiguity_never_reaches_the_gateway
Failed: ambiguous identity reached IBKR
1 failed in 0.48s
```

### 7. Admit missing receipts into automatic decision material

Mutation: removed the `contract_status == "found"` filter from
`_current_decision_material`.

```text
FAILED test_contract_missing_receipt_is_excluded_from_automatic_decision_material
AssertionError: missing evidence_id was present in current_evidence
1 failed in 0.32s
```

### 8. Ignore the injected per-run query cap

Mutation: `_load_evidence` forwarded the production default 8 instead of the
injected value 3.

```text
FAILED test_deadline_without_effective_date_caps_schedule_and_triggers_final_market_check
At index 0 diff: (..., 8) != (..., 3)
1 failed in 0.46s
```

## Static Verification

```text
python -m compileall -q src/security_lifecycle_ibkr_evidence.py \
  src/service/security_lifecycle_automation_scheduler.py
# exit 0, no output

git diff --check
# exit 0, no output
```

## Supplemental Ordering Probe

The command below was intentionally not counted as a passing gate:

```text
pytest -q tests/test_security_lifecycle*.py
```

It emitted 463 passing dots through `test_security_lifecycle_news_evidence.py`
and then remained silent at the first routes node for more than four minutes.
It was interrupted with exit 130. Running the exact 539 collected nodes in the
four isolated groups above produced 539 passes and no failures. No Task 5
product path creates a thread, process, provider session, or shared test
registry; the residual is therefore recorded for controller-level handling
rather than hidden or repaired outside this task.

## Boundaries

- Files changed by implementation commit: exactly the two named product files
  and three named Task 5 test files.
- No `src/security_lifecycle_decision_policy.py` product change remains; it was
  modified only transiently for reverse control 7 and restored.
- No live provider, Gateway, profile DB, App, merge, push, or migration action
  occurred.
- Work was performed only in the isolated
  `/tmp/arkscope-lifecycle-automation-control-plane` worktree.
