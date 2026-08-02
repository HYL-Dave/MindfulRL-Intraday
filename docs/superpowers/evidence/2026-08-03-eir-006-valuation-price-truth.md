# EIR-006 Valuation Price Truth Evidence

> **Status:** PLAN REVIEW NEXT - NO PRODUCT OR DATA ACTION AUTHORIZED
>
> **Date:** 2026-08-03
>
> **Design authority:** `124622bc`
>
> **Implementation plan:**
> `docs/superpowers/plans/2026-08-03-eir-006-valuation-price-truth.md`

## 1. Authority And Grounding

### 1.1 Reviewed design

```text
path:
docs/superpowers/specs/2026-08-01-eir-006-valuation-price-truth-design.md
commit: 124622bc
lines: 1168
bytes: 50661
SHA-256:
7ec313d12e87b0e6557172c173d7f9d55468f36551514d463f6faaa0ff62bee4
review verdict: GREEN, zero findings
```

Plan-gate status evolution only:

```text
SHA-256:
a5c155d45ba690fd7e29aeaa37d49eeb44dc224496d51f3733f717afa0686f6c
semantic contract delta from reviewed blob: none
```

### 1.2 Plan-construction boundary

```text
worktree: /tmp/arkscope-eir-006
branch: codex/eir-006-valuation-price-truth
grounding commit: fd6d1b86383df2a98f97b235d9796d4bcaaa7a58
plan base: 124622bc
product edits during plan construction: none
provider requests during plan construction: none
production writes during plan construction: none
physical data movement/deletion during plan construction: none
```

### 1.3 Pinned native assets

| Asset | SHA-256 |
|---|---|
| `/tmp/arkscope_asyncio_wakeup_probe.py` | `10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e` |
| `/tmp/eir002-green-baseline/arkscope_eir002_reporter.py` | `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928` |
| `/tmp/eir002-green-baseline/run_native.sh` | `e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f` |

## 2. Pre-Edit Collection Ledger

### 2.1 Backend

| Stage | Delta | Nodes | SHA-256 | Status |
|---|---:|---:|---|---|
| Base | `+0/-0` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` | reproduced during plan construction |
| Task 1 target | `+10/-0` | 4,563 | `5fdc93f3dc78548048d7269d8088715028a57b1e2c54fe1ac422154d187f3986` | preconstructed |
| Task 2 target | `+19/-1` | 4,571 | `b247d173d3520668a5d475b0ed02f948d117c1097ed5ad86063a2dbf76d07b68` | preconstructed |
| Task 3 target | `+21/-1` | 4,573 | `e0ee195eb90bc9172dae36680b15b3285b3d82013c7c762e1989c955be6ea3b1` | preconstructed |
| Task 4 target | `+27/-1` | 4,579 | `6672d3df26b7c420d3253e4826b7104bfd0e5640ae16a1616ea75dd605b38639` | preconstructed |
| Final target | `+29/-1` | 4,581 | `6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f` | preconstructed |

Focused base:

```text
14 files / 307 nodes
SHA-256:
46f8c9d0cd9e3b525d051e2231d4d48ed2975192886cdeb293ec71662341ae51
```

Focused target:

```text
18 files / 335 nodes
SHA-256:
58230b548925b29035cff401520e0948b01dcaed8da2deed41149bea6b4a5ae1
```

### 2.2 Frontend

| Collection | Files | Nodes | SHA-256 | Status |
|---|---:|---:|---|---|
| Base full | 96 | 1,076 | `de48671aa1d3f70cb87166e3f5b026804e206ac31f8e29fe7e74b38cde9448d5` | reproduced |
| Target full | 97 | 1,077 | `8a88600623ff985d24f20fab85ba8375c574f230a1a964c09667a1184b2ceea2` | preconstructed |
| Base focused | 4 | 45 | `09a2b4ef080a5badab79fb674b5c5c6b85a0eb7c639c2fe9534616eff2b5bb84` | reproduced |
| Target focused | 5 | 46 | `5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127` | preconstructed |

Normalization authority:

```text
vitest list --json
relative_file<TAB>full_test_name
LC_ALL=C sort
```

## 3. Task 0 Grounding

> Not started. Fill only from raw artifacts after the reviewed plan authorizes
> Task 0.

### 3.1 Canonical native base

```text
report path:
report SHA:
collected:
seen:
passed:
skipped:
failed:
errors:
non-passing SHA:
wakeup probe result:
artifact manifest SHA:
boundary restored:
```

### 3.2 Consumer census

```text
discovery stream rows:
discovery stream SHA:
consumer-census rows:
consumer-census SHA:
behavior-propagation rows:
behavior-propagation SHA:
unknown verdicts:
duplicate rows:
reconstruction equality:
```

### 3.3 Protected production boundary

Record metadata/identity only. Do not include secret or data payload contents.

```text
market DB identity:
retired file metadata stream:
retired DB-row counts:
scheduler state read-only witness:
protected source blob stream:
before/after equality:
```

## 4. RED-First Implementation Record

### 4.1 Task 1 - session authority and selector

```text
structural/product RED:
wrong-RED corrections:
GREEN:
stage collection:
market_data_direct existing nodes:
no-create artifact witness:
commit:
```

### 4.2 Task 2 - static/dynamic split

```text
RED:
old key read calls:
static cache forbidden-field witness:
cache-hit price recomputation witness:
base-unit calculation witness:
GREEN:
stage collection:
removed/replacement node identity:
commit:
```

### 4.3 Task 3 - annual analysis and peer absence

```text
RED:
SEC/FD call-order spy:
legacy snapshot spy:
peer absence closed payload:
behavior-propagation owners:
GREEN:
stage collection:
commit:
```

### 4.4 Task 4 - stored SEC projection and retired files

```text
RED:
shared projection multi-consumer witness:
sync price/news preservation witness:
FileBackend no-probe witness:
daily-update no-scan witness:
GREEN:
stage collection:
commit:
```

### 4.5 Task 5 - frontend/current copy/static census

```text
backend RED:
frontend RED:
backend GREEN:
frontend focused GREEN:
frontend collection:
typecheck:
i18n scanner:
post-cutover census rows/SHA:
current-copy file list:
commit:
```

## 5. Mutation Evidence

Every row requires an exact diff artifact, owning result, pre/post product SHA,
and restored GREEN.

| Mutation | Owning node/set | Expected RED | Diff SHA | Result | Restored SHA |
|---|---|---|---|---|---|
| M1 older-day fallback | `test_missing_required_date_does_not_fallback_to_older_bar` | yes | | | |
| M2 26-slot rule | `test_one_row_qualifies_without_slot_completeness` | yes | | | |
| M3 raw UTC date | `test_et_market_date_not_raw_utc_date_owns_selection` | yes | | | |
| M4 old cache key | `test_old_metrics_cache_key_is_ignored` | yes | | | |
| M5 dynamic cache payload | `test_v2_static_cache_excludes_price_and_dynamic_fields` | yes | | | |
| M6 `1e6` unit error | `test_explicit_price_uses_base_unit_shares_without_million_scaling` | yes | | | |
| M7 legacy snapshot override | `test_legacy_ibkr_snapshot_cannot_override_sec_or_price_basis` | yes | | | |
| M8 FileBackend CSV read | FileBackend owning node | yes | | | |
| M9 daily directory scan | daily-update owning node | yes | | | |
| M10 legacy fundamentals projection | stored-SEC owning set | yes | | | |

## 6. Product Verification

### 6.1 Final collections

```text
backend full:
backend focused:
frontend full:
frontend focused:
exact backend delta:
exact frontend delta:
```

### 6.2 Focused and protected gates

```text
backend focused:
price collection truth:
current quote:
Financial Datasets policy/cache/client:
stored-only provider-free:
earnings doubles:
coverage v2:
scheduler price outcomes:
Tranche B protected bytes:
provider counters:
PG attempts:
```

### 6.3 Frontend gates

```text
full Vitest:
typecheck:
build:
i18n scanner:
```

### 6.4 Native canonical tip

```text
report path:
report SHA:
4581 collected == seen:
4509 passed / 72 skipped / 0 failed / 0 errors:
exit 0:
empty non-passing SHA:
artifact manifest/quarantine:
pre/post boundary equality:
```

## 7. Independent Implementation Review

```text
reviewed range:
review verdict:
reviewer collection reconstruction:
reviewer mutation reconstruction:
reviewer native control:
findings/fixes:
exact cleared tip:
```

## 8. Merge And Read-Only Rollout

```text
pre-merge master:
reviewed product tip:
ff-only proof:
merged master:
merged backend/frontend collections:
merged canonical admission:
read-only qualified selector observation:
fixture unavailable observation:
old-cache-ignore fixture witness:
daily-update/SQLite equality:
stored-SEC projection equality:
production writes:
provider calls:
EIR006_PRODUCT_CUTOVER_TIP:
```

Physical old data remains present at this checkpoint. EIR-006 remains
`promoted`; deletion is not authorized by product merge.

## 9. Fresh Deletion Manifest

> Blocked until Section 8 is merged and reviewed. Manifest construction is
> read-only.

```text
manifest commit:
manifest SHA:
comparison implementation SHA:
alias input SHA:
exact file rows/count/SHA:
raw-view summary:
canonical-view summary:
canonical keys absent from SQLite:
exact cache keys/count:
exact fundamentals rows/count:
exact sync rows/count:
consumer census SHA:
writer/process census:
saved scheduler state:
DB identity:
independent manifest review:
bounded exact-source amendment:
separate user approval:
```

## 10. Physical Closeout

> Blocked until the exact manifest and destructive amendment receive
> independent review plus separate user approval.

```text
approval reference:
quiesced writer proof:
file quarantine proof:
DB row snapshot proof:
transaction affected rows:
pre-final verification:
rollback or success disposition:
temporary rollback assets removed:
scheduler state restored:
canonical admission:
read-only production truth:
closeout review:
EIR closure commit:
```

## 11. Honesty Ledger

- Plan review does not authorize product implementation.
- Product implementation review does not authorize merge.
- Product merge does not authorize physical deletion.
- Read-only manifest construction does not authorize physical deletion.
- Physical deletion requires a reviewed exact-source amendment and separate
  user approval of the exact manifest.
- A partial transcript, focused suite, sandbox-incompatible run, or stale data
  count is never promoted to canonical evidence.
- No provider, entitlement, halt, no-trade, or volume cause is inferred from
  `no_qualified_price`.
- Historical documents are not rewritten to make the current tree look
  cleaner.
