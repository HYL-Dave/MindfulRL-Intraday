# Task 7 Report: Reuse Closed SEC Chains and Classify Operator Blockers

Date: 2026-08-30
Base: `3ccb1674`
Implementation: `57ec0b11`
Self-review fix: `c1438391`

## Status

DONE. Task 7 is implemented, tested, self-reviewed, and committed on
`lifecycle-automation-control-plane`.

No live SEC, listing-authority, Massive, or IBKR provider was called. No
production database, App, schema migration, merge, or push operation occurred.
All persistence tests used pytest-owned temporary SQLite databases.

## Result

- A due blocked reservation now retains the run's existing evidence, facts,
  and blockers while changing the run back to `running`.
- `AutomationPriorMaterial` carries the retained observation fingerprint,
  evidence, facts, and blockers through the kernel, worker, and scheduler
  evidence-loader seam.
- The loader returns retained regulator rows separately from newly acquired
  rows and explicitly identifies the provider families refreshed by the
  attempt.
- Completion validates the retained rows against the transaction's current
  rows, deletes stale refreshed-family facts/evidence, inserts replacements,
  and updates terminal state in one immediate transaction. Acquisition failure
  does not delete prior material; persistence failure rolls all deletes back.
- SEC reuse requires every conjunct independently: unchanged observation
  fingerprint, current date strictly later than `widened_end`, every retained
  regulator locator containing exact boolean
  `filing_chain_complete=true`, valid evidence content/document identities,
  valid fact and deadline citations, and an unambiguous reconstructed active
  deadline.
- Task 6 deadline supersession is reconstructed from retained SEC bytes through
  `_source_deadlines` and `_resolve_source_deadline`; transient `kind` and
  `supersedes_date` fields are not persisted.
- Persisted SEC excerpts are restored to filing/document order before deadline
  reconstruction. Multiple excerpts from one filing without deterministic
  rendered-text positions fail closed to fresh SEC acquisition.
- In-window material, incomplete chains, fingerprint mismatch, malformed
  locators, invalid identities/citations, and unresolved deadline sets all
  reacquire SEC rather than evaluating partial prior facts.
- Listing authority is always refreshed. Conditional IBKR acquisition still
  follows current requiredness, successor identity, ambiguity, and the Task 5
  query cap. Stale listing/IBKR rows are excluded from evaluation and replaced
  atomically on a due blocked retry.
- `massive_credential_missing` is nonretryable and has no `retry_at`. Recovery
  is a saved credential followed by an attended case run, which creates a new
  predecessor-linked attempt.
- `source_payload_invalid` remains distinct and retains exactly one automatic
  retry. Task 4 predecessor/retry semantics, Task 5 IBKR behavior, and Task 6
  exact deadline citations/tripwires remain intact.

## RED Evidence

### 1. Required first RED through the real due-blocked seam

This was the first test change and first Task 7 test run. No product file had
changed.

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py::test_due_listing_retry_preserves_closed_sec_chain_and_refreshes_listing
F                                                                        [100%]
E       AssertionError: assert [(0, 0), (0, 0)] == [(0, 0), (1, 1)]
FAILED tests/test_security_lifecycle_automation_scheduler.py::test_due_listing_retry_preserves_closed_sec_chain_and_refreshes_listing
1 failed in 0.61s
```

The first attempt persisted one SEC evidence row and one fact. The real due
listing reservation deleted both before the worker's loader could inspect
them, so the second loader observation was `(0, 0)` instead of `(1, 1)` and SEC
was reacquired.

After the product change, the same public owner passed:

```text
1 passed in 0.47s
```

### 2. Operator blocker classification RED

The typed worker/kernel owner initially persisted Massive credential absence as
retryable:

```text
FAILED test_scheduler_blocker_strings_persist_through_fact_kernel_readback[massive_credential_missing-False]
E       AssertionError: persisted retryable=True, expected False
1 failed
```

After removing only `massive_credential_missing` from the scheduler retryable
set, the owner passed with `retryable=false` and `retry_at=NULL`.

### 3. Deadline citation and unresolved-set REDs

The retained-citation parameter batch initially passed seven cases and failed
the invalid deadline citation case: the damaged citation was reused instead of
forcing SEC acquisition. Adding exact persisted deadline-citation validation
made all eight cases pass.

The restored unresolved-deadline owner initially failed independently:

```text
FAILED test_unresolved_retained_sec_deadline_forces_fresh_acquisition
E       AssertionError: assert [] == ['2026-06-02T12:00:00Z']
1 failed
```

Reconstructing Task 6 supersession and rejecting an unresolved non-empty date
set made the owner pass.

### 4. Filing-order self-review RED

Self-review found that persisted evidence is read in hash-derived
`evidence_id` order, while Task 6 deadline resolution is filing-chain-order
sensitive. The new public owner used valid retained SEC rows whose hash order
placed the extension before its predecessor:

```text
pytest -q tests/test_security_lifecycle_automation_scheduler.py::test_retained_deadline_reconstruction_uses_filing_order_not_evidence_id
F                                                                        [100%]
E       AssertionError: assert ['2026-06-02T12:00:00Z'] == []
FAILED tests/test_security_lifecycle_automation_scheduler.py::test_retained_deadline_reconstruction_uses_filing_order_not_evidence_id
1 failed
```

The loader now reconstructs deterministic filing/document order. A companion
control proves same-filing excerpts without positions reacquire SEC rather than
guessing order.

## GREEN Evidence

### Baseline before Task 7

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_automation_scheduler.py
194 passed in 8.98s
```

### Final Task 7 files

```text
pytest -q tests/test_security_lifecycle_fact_kernel.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_automation_scheduler.py
214 passed in 10.23s
```

The final focused seam covering unresolved dates, retained filing order,
same-filing ambiguity, the real due-blocked retry, the effective-date Massive
boundary, and attended credential recovery also passed:

```text
6 passed in 0.86s
```

### Broader lifecycle groups

The lifecycle suite ran in four non-overlapping groups:

```text
pytest -q tests/test_security_lifecycle_ibkr_evidence.py \
  tests/test_security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_grounded_shadow.py \
  tests/test_security_lifecycle_decision_policy.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_sec_evidence.py
257 passed in 10.39s
```

```text
pytest -q tests/test_security_lifecycle.py \
  tests/test_security_lifecycle_automation_migration.py \
  tests/test_security_lifecycle_automation_runtime.py \
  tests/test_security_lifecycle_automation_schema.py \
  tests/test_security_lifecycle_disposition.py \
  tests/test_security_lifecycle_fact_kernel.py
148 passed in 2.39s
```

```text
pytest -q tests/test_security_lifecycle_investigation.py
30 passed in 1.99s
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
158 passed in 10.67s
```

Total: `593 passed`, zero failures.

## Reverse Mutations

Every mutation below was applied independently, run against its public owner,
restored, and rerun GREEN before the next mutation.

### Reuse predicates

1. **Observation fingerprint:** bypassed the equality check. The mismatch owner
   failed because SEC calls changed from one to zero. Restored owner:
   `1 passed`.

2. **Strict closed-window boundary:** bypassed `today <= widened_end`. The
   in-window owner failed because SEC calls changed from one to zero. Restored
   owner: `1 passed`.

3. **Complete filing chain:** bypassed the
   `filing_chain_complete is True` requirement. The incomplete-chain owner
   failed because SEC calls changed from one to zero. Restored owner:
   `1 passed`.

4. **Exact locator value:** coerced an invalid locator value to true. The
   malformed-locator owner failed because SEC calls changed from one to zero.
   Restored owner: `1 passed`.

5. **Evidence/fact identities and citations:** disabled both retained-material
   and deadline-citation validators. The invalid content hash, document
   identity, and fact citation owners all reused SEC incorrectly:

   ```text
   3 failed
   ```

   Restored owners: `3 passed`.

6. **Deadline citation:** disabled only the persisted deadline-citation
   validator. The invalid deadline citation owner reused SEC incorrectly:
   `1 failed`. Restored owner: `1 passed`.

7. **Resolved deadline set:** removed the ambiguous/unresolved rejection. The
   restored unresolved-deadline owner made zero SEC calls instead of one:
   `1 failed`. Restored owner: `1 passed in 0.38s`.

8. **Reuse path itself:** forced `_reusable_regulator_material` to return
   `None`. The real due-blocked seam made two SEC calls instead of one:
   `1 failed`. Restored owner: `1 passed`.

9. **Persisted filing order:** bypassed the filing/document ordering helper.
   The extension-before-predecessor owner reacquired SEC:
   `1 failed`. Restored ordering owners: `2 passed in 0.47s`.

### Reservation and atomic replacement

10. **Reservation retention:** restored the old evidence/fact/blocker deletes
    inside due retry reservation. The real seam again observed `(0, 0)` on the
    second attempt:
    `1 failed`. Restored owner: `1 passed`.

11. **Transient Task 6 supersession:** replaced ordered deadline resolution
    with the first extracted row. The positive retained-chain owner selected
    OLD `2026-05-28` instead of NEW `2026-05-30`:
    `1 failed`. Restored owner: `1 passed`.

12. **Explicit refreshed-family contract:** changed reused bundles to return
    `refreshed_source_families=None`. Two contract owners failed because the
    listing/market family tuple was absent:
    `2 failed`. Restored owners: `2 passed`.

13. **Stale-family pruning:** skipped the stale evidence delete in completion.
    The atomic family owner retained stale listing and market rows:
    `1 failed`. Restored owner: `1 passed`.

14. **Rollback:** committed after stale-row deletes and before replacement
    insertion. The injected insert failure then lost the old blocker and
    provider rows:
    `1 failed`. Restored owner: `1 passed`.

### Provider-family refresh controls

15. **Listing refresh:** suppressed listing acquisition when SEC was reused.
    Both provider-family call-count cases observed zero listing calls:
    `2 failed`. Restored owners: `2 passed`.

16. **Conditional IBKR refresh:** removed regulator successor values from the
    IBKR trigger. The required-successor case missed its IBKR call while the
    not-required control still passed:
    `1 failed, 1 passed`. Restored owners: `2 passed`.

17. **No stale provider leakage:** returned all prior evidence as retained
    regulator material. Both provider owners exposed stale listing/market IDs
    during evaluation:
    `2 failed`. Restored owners: `2 passed`.

### Error and retry classification

18. **Massive credential nonretryability:** restored
    `massive_credential_missing` to the retryable set. Two worker owners failed:
    one persisted `retryable=true`, and one produced a non-NULL `retry_at`.
    Restored owners: `2 passed`.

19. **Malformed SEC remains distinct:** remapped `sec_invalid_json` to a
    transport-unavailable code. The real malformed-content owner incorrectly
    reached listing acquisition:
    `1 failed`. Restored owner: `1 passed`.

20. **Exactly one malformed-payload retry:** raised the
    `source_payload_invalid` automatic retry allowance from one to two. The
    exhausted integration owner selected the case instead of recording one
    `skipped_current`:
    `1 failed`. Restored owner: `1 passed`.

## Static Verification

```text
python -m compileall -q \
  src/security_lifecycle_fact_kernel.py \
  src/security_lifecycle_automation_worker.py \
  src/service/security_lifecycle_automation_scheduler.py \
  tests/test_security_lifecycle_fact_kernel.py \
  tests/test_security_lifecycle_automation_worker.py \
  tests/test_security_lifecycle_automation_scheduler.py
# exit 0, no output

git diff --check 3ccb1674..c1438391
# exit 0, no output

git status --short
# exit 0, no output before report creation
```

## Files and Commits

Implementation commit `57ec0b11` changed exactly the six brief-listed files:

- `src/security_lifecycle_fact_kernel.py`
- `src/security_lifecycle_automation_worker.py`
- `src/service/security_lifecycle_automation_scheduler.py`
- `tests/test_security_lifecycle_fact_kernel.py`
- `tests/test_security_lifecycle_automation_worker.py`
- `tests/test_security_lifecycle_automation_scheduler.py`

Self-review fix `c1438391` changed only the scheduler and its test file to
restore retained SEC filing order and add the fail-closed same-filing control.

This report is the only additional Task 7 file.

## Self-Review

- Re-read the complete Task 7 brief and binding design sections 2.6, 7.3, and
  8.3 against the committed diff.
- Confirmed selective SEC reuse is limited to due blocked retries with retained
  blockers and does not alter new-attempt predecessor semantics.
- Confirmed every reuse predicate is conjunctive and independently mutation
  owned.
- Confirmed evaluation receives only validated retained regulator rows plus
  current fresh provider rows.
- Confirmed stale refreshed-family rows are deleted only inside the same
  transaction that inserts replacements and completes the run.
- Confirmed acquisition and injected persistence failures preserve the old
  material.
- Confirmed Task 6 deadline resolution runs over deterministic filing order and
  returns the original selected row with exact citation identity.
- Confirmed listing is always refreshed and IBKR remains current-need,
  identity, ambiguity, and request-cap gated.
- Confirmed a blocked Massive credential run cannot enter the ordinary
  readiness-recheck reservation because that path requires a succeeded run.
- Confirmed no schema, migration, provider, production DB, App, merge, or push
  change occurred.

No Critical, Important, or Minor issue remained after the filing-order fix.

## Scope Boundaries

- No schema or migration was added.
- No provider transport contract or live-provider behavior was changed.
- No policy-version or execution-revision value was used as a retry mechanism.
- No frontend, route, runtime scheduling, production DB, or App code was
  changed.
- No merge or push was performed.

## Concerns

No known implementation concern. Provider behavior and transactional rollback
were validated hermetically rather than through live services or a production
database, as required by the task boundary.
