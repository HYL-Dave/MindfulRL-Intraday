# EIR-006 Valuation Price Truth Evidence

> **Status:** TASK 0 REVIEW READY - TASK 1 NOT STARTED
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
plan-review clearance tip: e261abc25de5fdc608feea7bbe68fe230cb05789
plan-review verdict: GREEN, zero findings
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
| Base | `+0/-0` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` | Task 0 reproduced |
| Task 1 target | `+10/-0` | 4,563 | `5fdc93f3dc78548048d7269d8088715028a57b1e2c54fe1ac422154d187f3986` | Task 0 preconstructed |
| Task 2 target | `+19/-1` | 4,571 | `b247d173d3520668a5d475b0ed02f948d117c1097ed5ad86063a2dbf76d07b68` | Task 0 preconstructed |
| Task 3 target | `+21/-1` | 4,573 | `e0ee195eb90bc9172dae36680b15b3285b3d82013c7c762e1989c955be6ea3b1` | Task 0 preconstructed |
| Task 4 target | `+27/-1` | 4,579 | `6672d3df26b7c420d3253e4826b7104bfd0e5640ae16a1616ea75dd605b38639` | Task 0 preconstructed |
| Final target | `+29/-1` | 4,581 | `6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f` | Task 0 preconstructed |

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
| Base full | 96 | 1,076 | `ef7f106054745c137ff70fe6ef2043bb7655185379de1f0a6ec3b1b2997b9396` | Task 0 reproduced |
| Target full | 97 | 1,077 | `3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb` | Task 0 preconstructed |
| Base focused | 4 | 45 | `09a2b4ef080a5badab79fb674b5c5c6b85a0eb7c639c2fe9534616eff2b5bb84` | Task 0 reproduced |
| Target focused | 5 | 46 | `5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127` | Task 0 preconstructed |

Normalization authority:

```text
vitest list --json
JSON decode through the pinned plan normalizer
relative_file<TAB>full_test_name
UTF-8 byte-order sort
normalizer: 62 lines / 2,233 bytes
normalizer SHA-256:
955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac
raw JSON text extraction: forbidden
jq @tsv re-escaping: forbidden
```

Plan-review correction: the first plan checkpoint retained an extra JSON/TSV
escape layer for the two `src/sse.test.ts` names containing literal `\n\n`.
Counts and focused identities were unaffected. The full base/target identities
above are rebuilt from decoded runtime names and supersede only those two
earlier full-stream SHAs.

## 3. Task 0 Grounding

Task 0 ran from exact reviewed tip
`e261abc25de5fdc608feea7bbe68fe230cb05789`. Its commands intentionally changed
no product, test, provider, scheduler, production database, or legacy-data
byte. The only tracked changes in this checkpoint are the implementation plan,
this evidence file, and the priority map. Concurrent extension telemetry is
separately attributed in Section 3.3.

### 3.1 Canonical native base

```text
fresh worktree: /tmp/arkscope-eir006-native-base-e261abc2
report path: /tmp/eir002-green-baseline/reports/eir006-task0-native-base-e261abc2.json
report SHA: 2e94c3af06b7ada6596fb4068a5ca237d7c70edc11d806a370f91166a6f0c174
transcript SHA: 968b0ce78d37da12e25044b1954bb3a2e8a6617c2c5d0d0f855726aa1b200b2e
collected: 4553
seen: 4553
passed: 4481
skipped: 72
failed: 0
errors: 0
exit: 0
collected stream SHA: 69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca
non-passing SHA: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
wakeup probe result: {"callback_fired":true,"ready_count":0,"wake_bytes":0}
artifact rows: 575
artifact manifest SHA: aafe16c850dc73a1f36870b2aa5750268016c60dde35ff512add8c04eba203b4
quarantine implementation SHA: af85aa60048ed9a42e6425edb315d5cd22c0b9a839dcff0d79f51bafbb02c14b
boundary restored: yes; tracked diff empty, data/ empty, src/data absent
```

The fresh boundary had no `config/.env`, an intentionally empty `data/`, no
`src/data`, and one pinned ignored `node_modules` symlink. The 575 generated
paths included bytecode/cache files, 15 scratchpad artifacts,
`data/logs/sa_native_host.log`, and
`src/data/cache/risk_free_rate.json`. Every path was inventoried before exact
relative-path quarantine; no pre-existing fresh-worktree file changed.

### 3.2 Consumer census

```text
raw structured submatches: 383
audited exact duplicate occurrences: 8
discovery stream rows: 375
discovery stream SHA: 0c105ac7b247e78d1528efb138f8823fb196c9bb1c92092dba8f52ac194b4b29
consumer-census rows: 375
consumer-census SHA: a8d165a2947f429a6918675fbd1357d8ddbbd11595a4e42b75c6027d8cd2b971
behavior-propagation rows: 4
behavior-propagation SHA: 613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba
unknown verdicts: 0
duplicate rows in admitted streams: 0
empty owners: 0
reconstruction equality: byte-for-byte equal
discovery producer SHA: c3993fc615f491f0fd43a9d69c065203114b527ae68d33720e6b6a50105f4ce3
classifier SHA: 769b0b7a1c96306837fe7db0d4e7d58249822eee0759b340e7b3fab9c6ae5ba6
behavior producer SHA: 0b6409d03193f504de8f4aa46fbcb89aab42cb7ada54e20ad201dd56e1360987
```

Consumer verdict distribution:

| Verdict | Rows |
|---|---:|
| `historical_reference` | 198 |
| `low_level_empty_compatibility` | 7 |
| `retired_current_consumer` | 18 |
| `rewired_current_consumer` | 59 |
| `test_fixture_reference` | 83 |
| `unrelated_lexical_hit` | 10 |

The behavior census proves four live forwarding surfaces: Anthropic bridge,
OpenAI bridge, `/fundamentals`, and the institutional evidence packet. The
OpenAI bridge was absent from the reviewed plan's owner list. Task 0 amends the
plan to include its product file and existing reachability node; node counts and
all preconstructed hashes remain unchanged.

### 3.3 Protected production boundary

Record metadata/identity only. Do not include secret or data payload contents.

```text
market DB identity: inode 127284871 / 3,409,027,072 bytes / mtime_ns 1785696259805305153
market DB SHA: cfb8a0aa9e8df94b3d19543c0f548a22a9aa0bf9a061ca3866744e1f5adbeb9d
retired file metadata stream: 301 paths / 3f353d120b5054dd43406ea85694b5c3f804dc4cd4559398134d307de957b344
retired DB-row counts: old cache 19 / fundamentals 130 / fundamentals sync 1
market totals: financial_cache 46 / sync 3 / prices 2,406,320, all 15-minute
price frontier: 2024-01-02T14:30:00+0000..2026-07-31T19:45:00+0000 / 151 tickers
scheduler state: four sources succeeded, no error/continuation; ibkr_prices enabled / 720 minutes
protected source blob stream: 31 paths / ecb45d1260e9542ddc705d73cc11af2e27bcfa5ca09ac5e16b12f8daac8df6a8
protected path-list SHA: 3778006be667c4aa5d0ee7c687bedd5401fdaa4473eab07b1c4fd08bbab2388d
current FD-enable source slice SHA: bb87764bb49bcab870d4cda5f6bd3094596eea0946ee108755faa3da9217ec53
pre snapshot SHA: ee14800914947c6bc656e740da43ad079a8dba5f9b46f6b2b2a80395a5ea5f35
post snapshot SHA: ec657916cd6e18e503bb8b7714ec80be4b1058552ff2ee0e98ee30f619c7ef3c
market DB/prices before-after equality: byte-for-byte equal
```

All production reads used SQLite URI `mode=ro` plus
`PRAGMA query_only=ON`. The market database, its counts/frontier, the 301-path
legacy-file metadata stream, schedule settings, and scheduler-state rows were
unchanged after native admission.

`data/profile_state.db` did change during the native window: it grew from
45,776,896 bytes (`662aeae4...`) to 45,785,088 bytes (`69c78260...`). Read-only
attribution found exactly three new `job_runs` rows, IDs `18626`, `18627`, and
`18628`, all successful `sa_market_news_refresh` runs with
`trigger_source=extension`; every other time-aware table had zero rows in the
window. The attribution artifact is
`a30c445fe6a088e93e77800887f2f7d91a64cc08afb1c34fc5a00382ad3c1375`
and its query implementation is
`faee8bfaac6b953813228ddf0c32e1a990778fe9bf33184c3cefa1019666483d`.
These external extension writes were preserved. They are not attributed to the
fresh native test run, and this evidence does not claim all production bytes
were static.

### 3.4 Grounding corrections and rejected attempts

- Two frontend-list attempts were rejected before admission: one lacked the
  pinned toolchain from its execution root; one passed a root shape that made
  Vitest resolve the configuration from the wrong directory. The admitted
  third attempt ran from `apps/arkscope-web` and reproduced all four reviewed
  identities through the pinned JSON-decoding normalizer.
- The first detached-worktree creation failed at git-crypt smudge and left no
  registered worktree. The successful retry used the repository's established
  no-op git-crypt filter for this plaintext reviewed tree and rechecked exact
  `e261abc2` before admission.
- The plan's illustrative base and tip commands supplied worktree/output paths
  to a wrapper that accepts one stage name and operates on the current
  directory. Task 0 used the wrapper's actual pinned interface and amends both
  commands before Task 1. The consumed Task 0 stage must not be reused;
  independent review uses a new single-use stage under the same boundary.
- No rejected attempt is used as collection, census, native, or no-write
  evidence.

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
