# Price Collection Partial-Truth Evidence

> **Status: LIVE COMPLETE - MERGED AND NATURAL-CYCLE VERIFIED**
>
> **Historical blocked-run base:** `542776c2e00ae1737d5b424a3b8858b079a63e38`
> **Restart base:** `e6d4b7fac7e91c59e855a7f543caac4f57094d86`
> **Plan-review clearance:** `15933c316a68efd7e503f2778aba68affa2cb4c1`
> **Restart clearance:** `5fecce6536f5d9f4a13903a6c1059e235ba15324`
> **Tiered-contract clearance:** `3863b3be02034b3278f58d7090dcf0bc20445fe3`
> **Runner-design clearance:** `1d08a9f30a87066ea0a2e3b3274a22210cdfa57d`
> **V2 plan-review clearance:** `00d35376511b8bd28c16dd9c40415e0ddbc533ab`
> **V2 blocker packet:** `18ab76f93e3aead749da00a52a9a539ba6a57876`
> **V3 handshake-design clearance:** `6c89d4a1`
> **V3 exact-source plan clearance:** `1a8379e72216e0c109f9498caf64abfa593b299c`
> **Observed:** 2026-07-29 through 2026-07-31 Asia/Taipei

The historical Task 0 attempt stopped under plan Stop Condition 11. No product
file was edited and no partial full-suite output is accepted as an A/B
baseline. The reviewed query-route harness is now merged and this branch is
rebased. Focused review of `7844429a..5fecce65` returned GREEN with zero
findings and authorized a full Task 0 restart. That restart reproduced every
collection and focused gate, then stopped under the same condition at the next
untouched lifespan family, `tests/test_api.py::TestHealth::test_status`.
The later causal diagnosis selected V6 without identifying a code seam and is
merged at `e6d4b7fa`. Focused review cleared the tiered verification contract
at `3863b3be` with zero findings. The authorized Task 0 restart reproduced
every collection and focused gate, but its runtime controller violated the
reviewed termination protocol. The first two partial tiers and the following
launch are therefore `invalid`, not baseline results. Product RED remains
unauthorized. Focused review later cleared exact runner plan `00d35376` and
authorized deterministic-v2 Task 0. That run reproduced every baseline,
probe, and mutation control, then correctly stopped at its first runtime
`invalid`: T3 closed its progress pipe after complete progress and reporter
artifacts but before `Popen.poll()` observed process exit. The current runner
cannot distinguish that final-exit handoff from an early invalid EOF. A
separately reviewed v3 EOF/leader/group-drain amendment, exact-source plan,
and fresh complete deterministic-v3 Task 0 baseline are required before
product RED.
Focused review then cleared the exact v3 plan at `1a8379e7`. The authorized
fresh v3 run fixed the prior EOF/exit invalidation and reproduced every
control, but five of eight tiers remained `unresolved_stall` after their one
pre-registered retry. Three tiers completed naturally; no attempt was
`invalid`. The atomic side is nevertheless incomplete, so those selected
tiers are not a full baseline and product RED remains unauthorized. The next
gate is the already-owned `EIR-005` machine-state observer spec, not another
unchanged retry or another runner amendment.

That ordering was superseded on 2026-07-31 by a direct execution-boundary
A/B. The Codex managed sandbox rejects the selector self-pipe send used by
`asyncio.call_soon_threadsafe()`, while the native boundary delivers it. The
observer campaign was therefore not run. The unchanged v3 protocol completed
all eight base tiers natively on their first attempts; Section 8.11 is the
current Task 0 authority. Product implementation and its first independent
review followed; Sections 3-7 now record the current `7948f68d` review-fix
tip and complete fresh-root native verification.

## 1. Scope And Authorities

- Design authority:
  `docs/superpowers/specs/2026-07-28-price-collection-partial-truth-design.md`.
- Implementation authority:
  `docs/superpowers/plans/2026-07-28-price-collection-partial-truth.md`.
- Clearance branch: `codex/price-collection-truth`.
- `542776c2` remains the historical blocked-run base.
- Reviewed diagnosis closeout `e6d4b7fa` is the restart base and an ancestor
  of the rebased branch. Harness tip `2edf12e1` remains a historical
  prerequisite within that lineage.
- `5fecce6536f5d9f4a13903a6c1059e235ba15324` is the exact focused-reviewed
  restart clearance.
- `3863b3be02034b3278f58d7090dcf0bc20445fe3` is the exact focused-reviewed
  tiered-contract clearance and the Git identity used by the invalid runtime
  attempts documented in Section 8.4.
- `00d35376511b8bd28c16dd9c40415e0ddbc533ab` is the exact focused-reviewed
  deterministic-v2 plan clearance and the Git identity used by the attempts
  documented in Section 8.7.
- `1a8379e72216e0c109f9498caf64abfa593b299c` is the exact focused-reviewed
  deterministic-v3 plan clearance and the Git identity used by the attempts
  documented in Section 8.10.
- `5ff3608a979519b7aee8b68dc9863ca852ac1ce1` is the unchanged docs-only
  product identity admitted by the fresh native-boundary base in Section
  8.11.
- `0075ba9ef49ff3cd4a71d1c6c42d89de7046f7d8` is the merged product base.
- `7948f68dc0d8f5dc4b3268f1fa5e20a15b9dc3af` is the current product tip
  after resolving initial independent-review findings.
- Main-worktree drafts
  `docs/data/IBKR_PACING_AND_ERROR_SEMANTICS.md` and
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` remained untracked and were not
  read as implementation authority, edited, staged, moved, or deleted.

## 2. Canonical Baseline

The four canonical collections reproduced exactly again during the authorized
restart:

| Gate | Reproduced result |
|---|---|
| Backend full collection | `4722`; `fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` |
| Backend focused collection | `151`; `3c07d208ced889497521a779ae46dd88403277c34055c00ba9fd74ada08da428` |
| Backend focused composition | direct `63`, worker `4`, scheduler `84` |
| Frontend full collection | `96` files / `1074` nodes; `e322e7a51e83eedb8b3c7b1fd99e6033f496031968c1a2cb3f59974bfd994f47` |
| Frontend focused collection | `86`; `739385b104c147744e7421f030e3fc628b2d99a981406c9c13aeb25c2a70a479` |
| Frontend focused composition | Settings `36`, resources `14`, display `36` |

Focused behavior also reproduced:

- backend direct/worker/scheduler: `151 passed`;
- frontend focused: `3` files / `86 passed`;
- visible-literal scanner twice: `36/20/0/20` both times;
- tool/no-PG focused gate: `16 passed`, retaining central/OpenAI/Anthropic
  counts `53/54/54`;
- no-PG runtime smoke: `23/23`, `ok=true`, `pg_attempts=[]`.

These collections and focused results reproduced again at tiered-contract
clearance `3863b3be`. They establish Steps 1-4 only. They do not repair or
replace the invalid Step 5 runtime attempts.

They reproduced a third time at deterministic-v2 clearance `00d35376`.
The four node streams, focused behavior, scanners, tool counts, and no-PG
smoke were unchanged. This establishes deterministic-v2 Steps 1-4, but the
incomplete base summary in Section 8.7 is not a baseline.

They reproduced a fourth time at deterministic-v3 clearance `1a8379e7`.
The four node streams, focused behavior, scanners, tool counts, and no-PG
smoke were unchanged. This establishes deterministic-v3 Steps 1-4, but the
incomplete base summary in Section 8.10 is not a baseline.

### 2.1 Isolation correction before grounding

The initial empty-data assertion found an ignored 143,360-byte
`data/profile_state.db` created by the earlier 2026-07-28 baseline attempt.
It was not production data:

```text
isolated inode: 90586961
isolated size: 143360
isolated mtime: 2026-07-28 23:21:11 +0800
isolated SHA-256: fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d
main-worktree production inode: 127284276
main-worktree production size: 43962368
```

The fixture and its WAL/SHM companions were moved reversibly to `/tmp`; none
was deleted. Restart focused tests later recreated the same deterministic
fixture SHA. Before the full-suite attempt, that file was moved to:

```text
path: /tmp/price-truth-restart-task0-profile_state.db
inode: 90597154
size: 143360
SHA-256: fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d
```

`data/` was empty before the restart full-suite attempt and remained empty
after it was stopped.

## 3. RED Evidence

Task 1 added the seven exact direct-collector nodes and evolved the five
existing nodes without renaming them. Before the product edit:

```text
command: python -m pytest -q tests/test_market_data_direct.py
result: 12 failed / 58 passed
```

Eleven failures were the expected missing semantic envelope
(`KeyError: 'status'`). The write-boundary node failed because
`_unresolved_price_target_dates` did not exist. No failure came from provider
access, calendar setup, SQL fixtures, imports, or the lock harness. This is
the intended RED for missing post-write reconciliation and derived status.

Task 2 then added four worker-boundary nodes and evolved the two existing
serialization nodes in place:

```text
command: python -m pytest -q tests/test_prices_runtime.py
result: 6 failed / 2 passed
```

The failures were the intended contract gaps: hard-coded success, failed
collector results exiting zero, coerced malformed counts, absent semantic
fields, and unsanitized non-retryable exception text. Argparse, fixtures, and
provider configuration setup remained valid.

Task 3 added six scheduler nodes and evolved the five existing price worker
seams without renaming them:

```text
command: python -m pytest -q tests/test_data_scheduler.py
result: 6 failed / 84 passed
```

Three outcome failures proved that return-code-zero partial was being
reclassified as succeeded, failed payload audit used the generic worker
message, and the partial-then-success history lacked its first failed audit
projection. Three parser failures proved the new facts were neither preserved
nor validated. The normalized-news audit pin remained green.

Task 4 added the two exact frontend nodes and evolved the resource ledger
without renaming any existing node:

```text
command: npx vitest run \
  src/marketDataDisplay.test.ts \
  src/SettingsProviderConfig.test.ts \
  src/i18n/resources.test.ts
result: 3 failed / 85 passed
```

The display and mounted Settings failures showed the existing generic
`partial` label instead of the bounded LCID fact. The resource inventory
failed at `704` versus `706` Settings leaves and `1783` versus `1785` total
leaves. No failure came from the schedule fixture, DOM mounting, locale
switch, or DTO compilation.

## 4. GREEN Evidence

Task 1 implemented only the locked direct-collector shape: stable reason
codes, one parameterized day-presence query over the original target dates,
post-insert reconciliation under `market_write_lock`, one issue rollup, and
the documented three-value audit projection.

```text
command: python -m pytest -q tests/test_market_data_direct.py
result: 70 passed in 1.28s
tip collection: 70 nodes
tip node SHA-256: 584cdd096455f7d86904d7f208e72c6cc597e4bf7f569c726aac16b199f618cb
base -> tip: 63 -> 70, +7 / -0
```

All five evolved node IDs survived. Provider fetch remains outside
`market_write_lock`; original-target reconciliation runs inside it.

Task 2 added strict status/count/ticker validation, bounded issue identities,
sanitized non-retryable exceptions, and status-derived process exit:

```text
command: python -m pytest -q tests/test_prices_runtime.py tests/test_market_data_direct.py
result: 78 passed in 1.32s
worker collection: 8 nodes
worker node SHA-256: 44138730587258f578358c58068359930abdd818d8da21511fa346795598f374
worker base -> tip: 4 -> 8, +4 / -0
```

The exact retryable lock-busy diagnostic remains available; other raw
exception values and per-ticker provider messages do not cross stdout.

Task 3 now treats the closed child payload as semantic authority, preserves
lock-busy skip classification, stores price partial as durable `partial`
without continuation, and projects only that price outcome to a failed
three-value audit row:

```text
command: python -m pytest -q \
  tests/test_market_data_direct.py \
  tests/test_prices_runtime.py \
  tests/test_data_scheduler.py
result: 168 passed in 4.77s
focused collection: 168 nodes
focused node SHA-256: 9faa90281df39dddccf7bedf3ad2ad7304341560c00dea8ff8b9dd887f5e55a3
focused base -> tip: 151 -> 168, +17 / -0
```

Normalized-news partial continues to write `job_runs.status=succeeded`; the
new protection node proves this slice did not normalize that separate audit
contract.

Task 4 extends only the schedule-result DTO and the pure Settings presenter.
Actionable continuation remains first; a source-exact, status-exact price
branch displays the positive unresolved count and at most 25 non-empty ticker
IDs without creating a Continue action:

```text
command: npx vitest run \
  src/marketDataDisplay.test.ts \
  src/SettingsProviderConfig.test.ts \
  src/i18n/resources.test.ts
result: 88 passed in 2.69s
frontend full collection: 1076 nodes
frontend full node SHA-256: de48671aa1d3f70cb87166e3f5b026804e206ac31f8e29fe7e74b38cde9448d5
frontend focused collection: 88 nodes
frontend focused node SHA-256: b6f01cae4038c5c94f51da05ad920e52b723c387c6f48938f7dce6a13b028e4f
frontend base -> tip: 1074 -> 1076, +2 / -0
```

The existing pre-Slice-5 inventory node now explicitly excludes the two
post-Slice Settings paths while asserting that both paths exist. This keeps
its historical `614/637` claim intact rather than relabeling new resources as
pre-Slice-5 content. The current resource ledger is Settings `706`, Explore
`380`, total `1785` in both locales.

The visible-literal scanner passed twice at `36/20/0/20`. TypeScript
typecheck and the production build both exited zero. The mounted bilingual
node retained the generic failed audit glyph, rendered the price partial
facts, and exposed no Continue control. `DataSourcesSection.tsx` was not
edited.

## 5. Node And Resource Accounting

Tasks 1-3 account for the complete planned backend `+17/-0`: direct
`63 -> 70`, worker `4 -> 8`, and scheduler `84 -> 90`. Task 4 accounts for
the complete frontend `+2/-0`: Settings `36 -> 37`, display `36 -> 37`, and
resources remain `14`. Resource leaves changed only by the two bilingual
price-partial keys: Settings `704 -> 706`, Explore `380`, total
`1783 -> 1785`.

Final collection identities are:

```text
backend full:    4739 / a72bbd36dfad3d36aee2e6630e6024ec9fb4e910bebaf1363d44df8a1aa204dd
backend focused:  168 / 9faa90281df39dddccf7bedf3ad2ad7304341560c00dea8ff8b9dd887f5e55a3
frontend full:   1076 / de48671aa1d3f70cb87166e3f5b026804e206ac31f8e29fe7e74b38cde9448d5
frontend focused:  88 / b6f01cae4038c5c94f51da05ad920e52b723c387c6f48938f7dce6a13b028e4f
```

The exact backend additions are:

```text
tests/test_market_data_direct.py::test_backfill_partial_preserves_healthy_rows_and_marks_unresolved_target
tests/test_market_data_direct.py::test_backfill_failed_when_every_ticker_has_issue
tests/test_market_data_direct.py::test_backfill_resolved_zero_bar_target_stays_succeeded_and_clears_error
tests/test_market_data_direct.py::test_backfill_one_row_low_volume_day_stays_succeeded
tests/test_market_data_direct.py::test_backfill_non_target_rows_do_not_resolve_original_zero_bar_target
tests/test_market_data_direct.py::test_backfill_rechecks_original_target_set_only_once
tests/test_market_data_direct.py::test_backfill_exception_and_unresolved_tickers_share_one_issue_rollup
tests/test_prices_runtime.py::test_prices_worker_prints_sanitized_partial_json_and_exits_zero
tests/test_prices_runtime.py::test_prices_worker_prints_sanitized_failed_result_json_and_exits_nonzero
tests/test_prices_runtime.py::test_prices_worker_rejects_unknown_status_and_malformed_counts
tests/test_prices_runtime.py::test_prices_worker_bounds_sorts_and_sanitizes_ticker_lists
tests/test_data_scheduler.py::test_prices_worker_stdout_parser_preserves_partial_truth_and_bounded_tickers
tests/test_data_scheduler.py::test_prices_worker_stdout_parser_rejects_malformed_partial_payloads
tests/test_data_scheduler.py::test_prices_partial_persists_durable_partial_failed_audit_and_no_continuation
tests/test_data_scheduler.py::test_prices_failed_payload_persists_failed_without_partial
tests/test_data_scheduler.py::test_prices_success_clears_prior_partial_and_preserves_audit_history
tests/test_data_scheduler.py::test_price_partial_projection_does_not_change_normalized_news_audit_status
```

The exact frontend additions are:

```text
src/marketDataDisplay.test.ts	schedulerStateLabel > renders price unresolved count and bounded ticker list without continuation
src/SettingsProviderConfig.test.ts	Settings provider config authority > renders price partial facts without a Continue control in both locales
```

Both full comms are exactly those additions with no removed ID. The twelve
planned in-place backend nodes retained these exact IDs:

```text
tests/test_market_data_direct.py::test_backfill_per_ticker_exception_isolated
tests/test_market_data_direct.py::test_backfill_meta_write_failure_in_error_path_does_not_abort_batch
tests/test_market_data_direct.py::test_backfill_topup_idempotent_on_complete_day
tests/test_market_data_direct.py::test_backfill_ibkr_empty_from_swallowed_request_error_falls_to_polygon
tests/test_market_data_direct.py::test_backfill_fetches_provider_rows_outside_market_write_lock
tests/test_prices_runtime.py::test_prices_worker_prints_sanitized_success_json
tests/test_prices_runtime.py::test_prices_worker_prints_sanitized_error_json
tests/test_data_scheduler.py::test_p0c1_ibkr_prices_runs_prices_worker_subprocess
tests/test_data_scheduler.py::test_p0c_ibkr_prices_no_longer_uses_pg_sync
tests/test_data_scheduler.py::test_price_scope_required
tests/test_data_scheduler.py::test_prices_worker_retryable_lock_busy_is_skip_not_failure
tests/test_data_scheduler.py::test_prices_worker_stdout_parse_preserves_retryable_and_counts
```

The existing resource-count node also retained its ID. No `def test_` or
`it(...)` identity was removed or renamed.

## 6. Mutation Evidence

Each product mutation was applied alone, run only against its named owner,
reversed exactly, and followed by a product-blob identity check:

| Mutation | RED evidence |
|---|---|
| replace the reconciliation call with `unresolved = []` | partial sibling node observed `succeeded` instead of `partial` |
| replace zero-row ticker targets with `[]` | partial sibling node observed `succeeded` instead of `partial` |
| classify day presence as at least 26 stored rows | one-row low-volume node observed `gaps_found=1` instead of `0` |
| pass `error=None` for an unresolved ticker | meta assertion observed advanced `last_success` and cleared `last_error` |
| hard-code worker payload status to `succeeded` | worker partial node observed the wrong closed payload |
| derive scheduler price status from return code zero | scheduler partial node observed `succeeded` instead of `partial` |
| omit `price_partial` from `audit_failed` | scheduler node observed audit `succeeded` instead of `failed` |
| remove the frontend price-label branch | both new frontend nodes observed the generic partial label |

The load-bearing 26-row mutation changed the target classifier itself, not a
helper reached only after an already-empty target set:

```diff
-                    "SELECT DISTINCT substr(datetime, 1, 10) FROM prices "
-                    "WHERE ticker = ? AND interval = ?", (canon, db_interval)).fetchall()
+                    "SELECT substr(datetime, 1, 10) FROM prices "
+                    "WHERE ticker = ? AND interval = ? "
+                    "GROUP BY substr(datetime, 1, 10) HAVING COUNT(*) >= 26",
+                    (canon, db_interval),
+                ).fetchall()
```

That mutation made
`test_backfill_one_row_low_volume_day_stays_succeeded` RED at its
`gaps_found == 0` assertion. Product blobs at the initial `fa0cade9` tip were:

```text
src/market_data_direct.py: f84f125e1cf4f95b1841645136e87fec5f2ad7d9
src/prices_runtime.py: c068acee45f4e798d95643ac169a5e3d446e0c63
src/service/data_scheduler.py: 7f395795538fe8c60e07b7232cf9fdd4ff456f41
apps/arkscope-web/src/marketDataDisplay.ts: 86dc542e0204f7e4688e52a23f36bdc7a6af664c
```

Independent review of `0075ba9e..41ea0ac4` found three product defects and
three test/evidence gaps. Each product defect was reproduced before the fix:

| Review finding | RED evidence |
|---|---|
| a message-less per-ticker `TimeoutError()` became success | the existing exception-isolation node observed `errors={}` instead of `{"BAD": "TimeoutError"}` |
| provider meta treated `error=""` as success | the existing meta node observed `last_success` advance from the planted 2000 value |
| an older observed bar could move the frontier backward | the existing upsert node observed `2026-06-16` instead of preserving `2026-06-18` |
| scheduler accepted contradictory complete ticker facts | the malformed-payload node accepted `error_tickers=["AAPL"]` with unresolved `["LCID"]` |

Fix commit `7948f68dc0d8f5dc4b3268f1fa5e20a15b9dc3af` makes only
`None` mean no error, gives empty exception text a stable class-name reason,
preserves detected target dates through per-ticker exceptions, computes the
maximum fetched bar, prevents provider frontier regression, and requires
unresolved tickers to be a subset of an untruncated error list. The existing
all-issue node now uses two genuinely unresolved tickers. The existing
frontend node now passes 30 tickers; removing the 25-item display bound made
that node alone RED. No node was added, removed, or renamed. Current product
blobs are:

```text
src/market_data_direct.py: 28605175a5ec6dac24642a1dc701dd8ea65e02cc
src/prices_runtime.py: c068acee45f4e798d95643ac169a5e3d446e0c63
src/service/data_scheduler.py: 403a26bb5589827bba824d869562a2eebadb9c5a
apps/arkscope-web/src/marketDataDisplay.ts: 86dc542e0204f7e4688e52a23f36bdc7a6af664c
```

## 7. Protected Boundaries

### 7.1 Initial native tiered tip

The same reviewed native execution boundary and unchanged v3 runner admitted
the initial `fa0cade9` tip. The external wakeup preflight returned:

```json
{"callback_fired": true, "ready_count": 0, "wake_bytes": 0}
```

The tip preflight SHA-256 is
`545b7ef6f85dc713b66de5733c388122e3499f7385ae88adec6062e5bc9d9b35`.
Its immutable tier union was `4739/4739`, byte-equal to the canonical stream,
with loads `591/591/600/590/590/590/590/597`. Every selected attempt was its
first attempt:

| Tier | Outcome | Duration (s) | Non-passing |
|---|---|---:|---:|
| T0 | `complete_pass` | `13.910` | 0 |
| T1 | `complete_nonpassing` | `16.831` | 2 |
| T2 | `complete_nonpassing` | `22.787` | 16 |
| T3 | `complete_pass` | `20.265` | 0 |
| T4 | `complete_pass` | `52.357` | 0 |
| T5 | `complete_nonpassing` | `24.695` | 1 |
| T6 | `complete_nonpassing` | `25.485` | 8 |
| T7 | `complete_pass` | `31.028` | 0 |

```text
tip summary SHA-256: 5cac3072c0151eb863a2fc26b309564a322467c038d64f9f8f7b55014e66f9ca
tip non-passing: 27 / 7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15
base non-passing: 27 / 7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15
new IDs: 0
gone IDs: 0
```

Fresh-process tiers reset process-global, module, fixture, and teardown state
between file groups and are not directly comparable with historical
monolithic runs. Base and tip are comparable because both use this same
tiered protocol and native execution boundary.

The required diagnostic monolithic run naturally executed all `4739` nodes:
`27 failed / 4640 passed / 72 skipped` in `280.00s (0:04:40)`. It remains
diagnostic `invalid` because the unchanged v3 terminal-summary regex does not
accept the optional duration suffix. Its transcript SHA-256 is
`2c55255b5bf4844038f5b9cc672f9385da323ec5107e04602f3e93ab3138f576`;
the report SHA-256 is
`cda36b333db66dc90d1faa8bcfa694448490f697f7cd03c67da26cd08ca48403`.
It does not replace or weaken the complete tiered A/B.

The tip started without `src/data`. T6 created only
`src/data/cache/risk_free_rate.json`; after the diagnostic it was 74 bytes,
inode `92936726`, SHA-256
`66d3d8bdf58de1ae18524c6bfa904f427ee9c2c6d1d7b4c14b5498564d096e74`.
It was moved reversibly to
`/tmp/price-truth-native-tip-src-data-20260731T1605`; `src/data` is absent
again.

### 7.2 Review-fix native tiered tip

The review-fix side did not reuse the initial bank. Its first preflight
(`acf6c2dc...`) refused before pytest because the isolated worktree contained
`data/profile_state.db` (inode `93323623`, 143360 bytes, SHA-256
`fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d`).
That side remains permanently incomplete with
`invalid_attempt="tip-T0-a1:runner_error"` and no selected attempt. The file
was moved reversibly to
`/tmp/price-truth-review-fix-prelaunch-data-20260731T1631/`.

A fresh root at `/tmp/price-truth-tier-v3-review-fix-native` retained the
exact reviewed runner `bb5d2245...`, reporter `09d2bc52...`, builder
`0f0421f8...`, and tier map `3d7adb7e...`. Its wakeup preflight again returned
`true/0/0`; preflight SHA-256 is
`f769ea737eeb6efba44ebf89a28ca2267fcead85a258dcd7658f71a15643d968`.
The root is bound to product commit `7948f68d`, 253 mapped paths, and the
unchanged `4739/a72bbd36...` collection. All selected attempts were first
attempts:

| Tier | Outcome | Duration (s) | Non-passing |
|---|---|---:|---:|
| T0 | `complete_pass` | `13.979` | 0 |
| T1 | `complete_nonpassing` | `17.019` | 2 |
| T2 | `complete_nonpassing` | `21.423` | 16 |
| T3 | `complete_pass` | `20.227` | 0 |
| T4 | `complete_pass` | `52.599` | 0 |
| T5 | `complete_nonpassing` | `24.653` | 1 |
| T6 | `complete_nonpassing` | `24.189` | 8 |
| T7 | `complete_pass` | `30.593` | 0 |

```text
review-fix tip summary SHA-256: caba76e0a21189b7cfb969e324903b803335a15958e5d9d32f815db3c69e37b1
review-fix tip non-passing: 27 / 7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15
base non-passing: 27 / 7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15
new IDs: 0
gone IDs: 0
```

The fresh-root diagnostic naturally executed all `4739` nodes:
`27 failed / 4640 passed / 72 skipped` in `280.09s (0:04:40)`. It remains
diagnostic `invalid` only because the unchanged terminal-summary regex rejects
the duration suffix. No signal was sent and all 9478 progress events arrived.
Transcript SHA-256 is
`feea0ef1137e8585e12b4a456f70eb9acbfa071118e5bff4863d86ca2b8d6a9b`;
report SHA-256 is
`cda36b333db66dc90d1faa8bcfa694448490f697f7cd03c67da26cd08ca48403`.

T6 and the diagnostic created only
`src/data/cache/risk_free_rate.json` (inode `93202653`, 74 bytes, SHA-256
`b232553eeafcb72479141777212aee5eaf5d00e2a1f842defb16d5e5366dfae3`).
The directory was moved reversibly to
`/tmp/price-truth-review-fix-src-data-20260731T1650`; isolated `data/` is
empty and `src/data` is absent again.

### 7.3 Regression and static gates

```text
backend focused: 168 passed
frontend full: 96 files / 1076 passed
frontend typecheck: exit 0
frontend build: exit 0
visible-literal scanner twice: 36/20/0/20
tool/no-PG focused: 16 passed; registry/OpenAI/Anthropic = 53/54/54
no-PG runtime smoke: 23/23; ok=true; pg_attempts=[]
protected schema/coverage/provider/Gateway/catalog tests: 47 passed
```

Diff from plan clearance `15933c316a68efd7e503f2778aba68affa2cb4c1`
is empty for both provider adapters, Coverage, provider health, Gateway lock,
market-data routes, provider config, `sql`, `scripts`, and
`DataSourcesSection.tsx`. The active source catalog remains exactly four
entries: Polygon/Finnhub news at 60 minutes, IBKR news at 120 minutes, and
IBKR prices at 60 minutes. Protected blob/tree identities are:

```text
ibkr_source.py:  b8d65b364fc38fb86ef771cc560dc83819231b07
polygon_source.py: 2816b39ea2773d85e39f12a40bbe554d0f4f8b9a
provider_health.py: 74735caef55173527916e3d961e9598b20aa5310
ibkr_gateway_lock.py: ddae2e5b4cd7a000bd065bd774d069f08c68e235
market_data.py: 77a01ed086bb05397dae2b2d2063ca17732b9655
data_provider_config.py: 1fb6524ba84c225d2e3e8b1bd47975a38113fc9c
provider_config_runtime.py: 588a175da1ff0da8c9e3fe84a3609348c88e17fd
market_coverage/sql/scripts tree: cb538c0e6b5bedd2f4f8d2bae59f1e56895763191d5999fa29aec6177f80f4f2
```

No test or verification action contacted a provider, Gateway, browser, or
production database. Product tip is
`7948f68dc0d8f5dc4b3268f1fa5e20a15b9dc3af`. The two protected main-worktree
drafts remain untracked and byte-identical at `4921194a...` and
`79d4eac9...`.

## 8. Historical Full-Suite Diagnostics And Tier Prototype

### 8.1 Historical pre-harness attempt

The base full suite was launched from empty isolated `data/` with unbuffered
verbose output:

```text
PYTHONUNBUFFERED=1 python -m pytest -vv --tb=short
```

It stopped making progress at:

```text
tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
```

After more than 70 seconds without another node line, the run was interrupted
with exit `130`. The partial transcript is diagnostic only:

```text
path: /tmp/price-truth-base-full.txt
lines/bytes: 58 / 5348 (plus one unterminated active-node line)
SHA-256: 7c4f83d2d3025e8e48b6a177bbdafa75a59b98beff5d8da23cd1453716445f6d
partial failures seen before the hang: 2
```

No normalized non-passing set was derived from this transcript. A bounded
single-node reproduction also stopped at fixture setup:

```text
timeout 20s python -m pytest -vv --tb=short \
  tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
exit: 124
last line: tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
```

No pytest process remained afterward. This is a concrete EIR-002/harness
diagnostic, not evidence against the price-truth product design.

### 8.2 Restart after the harness merge

The authorized restart used the reviewed instrumentation:

```text
PYTHONUNBUFFERED=1 python -m pytest -vv --tb=short \
  -o faulthandler_timeout=120
```

Both converted `test_agents` provider-route nodes passed in full-suite
context. The run then passed
`tests/test_api.py::test_fixed_task_runtime_routes_mount_on_real_app` and
stopped at:

```text
tests/test_api.py::TestHealth::test_status
```

At 120 seconds, faulthandler emitted all-thread stacks. The pytest thread was
waiting in `starlette.testclient.TestClient.__enter__`, reached from the
`tests/test_api.py:41` client fixture through AnyIO's blocking portal. The
portal thread was idle in the asyncio selector. This identifies the blocking
boundary but not the ambient root cause or suspended lifespan coroutine.

The operator sent Ctrl-C only after the dump; the execution session reported
exit `1`. No pytest process remained. The diagnostic transcript was preserved
under a unique name:

```text
path: /tmp/price-truth-restart-blocked-20260729-full.txt
lines/bytes: 204 / 18895
SHA-256: 1e2f8907b3936ccfdd2ace0cfb7f6d2b221752c4dd6c9d16f34288dc74872e1c
```

No normalized non-passing set was derived. This transcript does not establish
that `test_api.py` always stalls, does not weaken the earlier harness result,
and is not an A/B baseline.

### 8.3 Post-diagnosis tier-contract prototype

Diagnosis closeout `e6d4b7fa` fast-forwarded to `master` after focused review.
Merged collection-only verification reproduced:

```text
nodes: 4722
SHA-256: fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0
merge commits in 2edf12e1..e6d4b7fa: none
```

The price branch then rebased from old tip `f7458727` onto `e6d4b7fa`.
The pre-amendment rebased tip was
`813b42c7a8f8067e78cfc4d67602b097bff8cb83`.
Range-diff found the first seven price patches exactly equal; the final two
changed only where the priority map retained later diagnosis entries. The
three price authority files remained byte-identical across that rebase:

```text
spec:     aa275b448e16cc7c4708aecc47786e89f7905f2017173c342d5eb581f0068eb4
plan:     ab17beb904e0469b4a691da5c026bdd121a1bf934b6e4e13c0460c67262c851b
evidence: a6721b8e760d2ff788508739781f529c274004de458441f1b6d1279ef0932983
```

Before changing this authority, a collection-only prototype exercised the
exact Section 2.2 builder and all eight tier path lists. Collection subprocesses
set `ARKSCOPE_DISABLE_SCHEDULER=1` and used distinct temporary
`EDGAR_LOCAL_DATA_DIR` paths; those variables do not change collection
identity, and no runtime behavior result is inferred:

```text
root: /tmp/price-truth-tier-contract-prototype-20260730
builder SHA-256:
0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier-map SHA-256:
3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
mapped files: 253
tier nodes: 591 / 591 / 590 / 590 / 590 / 590 / 590 / 590
concatenated rows: 4722
unique rows: 4722
sorted-union SHA-256:
fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0
remaining `with TestClient(...)` files: 5, all mapped
  T1: tests/test_portfolio_capture_routes.py
  T2: tests/test_signal_factors_p1.py
  T5: tests/test_portfolio_activity_routes.py
  T6: tests/test_api.py, tests/test_events.py
```

`cmp` proved the tier union byte-identical to the canonical collection. This
is a mechanism prototype, not a runtime baseline: no tier behavior run,
non-passing set, Task 0 clearance, or product authorization is inferred.

The canonical stream contains 11 valid node IDs with embedded spaces, proving
that the historical whitespace-token parser is not safe for this protocol.
The stdlib-only scratch reporter at SHA-256
`09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928`
was therefore probed separately:

- a real 11-node `tests/test_env_unquote.py` run exited 0, and reporter
  collected/seen sets both matched the canonical IDs exactly;
- a one-node temporary failing parametrization with ID `id with spaces`
  exited 1 and preserved that full ID in collected, seen, and non-passing
  arrays; and
- the temporary probe is outside the repository and adds no planned node.

A second full collection rebuilt the planned `env -i` runtime boundary with
isolated home, temp, lock, five ArkScope-store, and EDGAR paths. It again
produced exactly `4722` nodes and SHA-256 `fcdb1b7d...`; the isolated worktree
`data/` remained empty. This proves collection identity under the credential-
and database-stripping environment but is still not a runtime behavior
baseline.

The exact `env -i` + `setsid` + reporter command was then exercised as a
contract probe. The real 11-node file completed with `11 passed`, shell and
reporter exit 0, identical collected/seen sets, and an empty non-passing set;
its recorded process had equal PID, PGID, and SID. The temporary
space-containing node completed with exit 1, a terminal summary, and the same
full node ID in collected, seen, and non-passing arrays. Neither probe touched
isolated worktree `data/`, and neither is admitted as a Task 0 baseline.

### 8.4 Tiered Task 0 invalid-runner blocker

Focused review of contract tip `3863b3be` returned GREEN with zero findings
and authorized tiered Task 0. Steps 1-4 then reproduced the exact collections
and focused gates in Section 2. Runtime scratch identity was:

```text
root: /tmp/price-truth-tier-v1
builder SHA-256: 0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
reporter SHA-256: 09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
tier-map SHA-256: 3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
runtime-fingerprint file SHA-256: d754856c368ae05668c6df075410e4e706cd4d1bab2e5b39b7740831db564b96
pip-freeze SHA-256: cf3e80661ab59e43b291b7ed037159aee90248fff7c1d4e38bf79de264b6eec8
```

Collection-only reconstruction again proved all `4722` unique nodes with tier
loads `591/591/590/590/590/590/590/590` and a sorted union byte-identical to
`fcdb1b7d...`. The reporter also preserved all 11 space-containing node IDs.

The runtime attempts are not admissible:

| Tier | Observed partial boundary | Transcript | Classification |
|---|---|---|---|
| T0 | `test_route_rejects_unreviewed_interval_with_typed_422`; dump mtime `09:15:35 +08:00`; process ended `09:36:33` | 661 lines / 67097 bytes; `efe83d6a...` | `invalid` |
| T1 | `test_scheduler_start_stop`; dump mtime `09:41:17 +08:00`; process ended `21:56:59` | 321 lines / 30414 bytes; `7fafeaf5...` | `invalid` |
| T2 | control shell sampled child PID/PGID/SID as `145/1/1` before `setsid` completed; no runtime output | 0 bytes; `e3b0c442...` | `invalid` |

For T0 and T1, faulthandler emitted the required 120-second dump, but the
orchestrator did not enforce the 150-second no-progress termination boundary.
The processes remained alive for materially longer than the reviewed bound.
Both were initially described operationally as stalls, but the closed outcome
table requires runner/protocol failures to be classified as `invalid`.
Neither transcript has a terminal pytest summary or reporter JSON, so no
partial node result is normalized or banked.

T2 exposed a separate control-wrapper race. The reviewed command launches
pytest through `setsid`; the operator added an immediate PID/PGID/SID
assertion outside the reviewed command shape. That assertion sampled the
background process before `setsid` completed, exited the shell, and left no
terminal pytest result. An anchored host-namespace check confirmed that no T2
pytest process survived. The extra assertion was not an authorized contract
change and cannot be repaired by silently reusing the attempt path.

The first invalid attempt required an immediate stop. T1 and T2 were started
only because T0 was incorrectly classified before artifact sealing; they are
retained as control-plane evidence, not as additional authorized tier results.
T3-T7, deferred retries, the diagnostic monolithic run, protected Step 6, and
product RED were not started.

All 46 selected runtime-control artifacts are listed in:

```text
/tmp/price-truth-tier-v1/task0-invalid-manifest.sha256
lines/bytes: 46 / 5407
manifest SHA-256:
1f3f5c6137fbd57da56e8cc7bd1dcf4e55e64fad8fe8896b419aab8882288c98
```

After the stop, isolated worktree `data/` was empty, no matching tier pytest
process remained, and the branch had no tracked or untracked change before
this evidence edit. The main worktree retained only the two protected
untracked drafts, with SHA-256 values `4921194a...` and `79d4eac9...`; neither
was edited, staged, moved, deleted, or cited as authority.

This blocker identifies an execution-control defect, not a price product
defect and not a test verdict. Re-running the same manually supervised wrapper
is not authorized.

### 8.5 Proposed deterministic runner amendment

Independent review of blocker `fa42d44a` returned GREEN with zero findings and
confirmed all three `invalid` classifications. The user selected the
single-module structured-progress design for the replacement runner.

Design Section 13 now requires one SHA-pinned Python file to act as both the
parent controller and a pytest progress plugin using
`pytest_runtest_logstart`/`pytest_runtest_logfinish`. Structured events travel
over one inherited pipe and are persisted to `progress.jsonl`;
the unchanged final reporter remains the sole node-accounting authority.
`Popen(start_new_session=True)` gives the controller stable process-group
ownership, while an in-process monotonic state machine owns the pre-first-node,
active-node, and final-teardown deadlines.

The breach distinction is explicit:

```text
150-second breach + 120-second per-item dump -> unresolved_stall
150-second breach without that dump         -> invalid
```

The runner itself performs `SIGINT`, the complete 10-second grace, optional
`SIGKILL`, atomic attempt recording, and first-invalid refusal. Four mandatory
pre-runtime probes cover natural pass, SIGINT termination, SIGKILL fallback,
and unchanged collect-only identity. The environment allowlist, final
reporter, four outcomes, immutable tier map, banking tuple, one deferred
retry, and base/tip admission contract remain unchanged.

Focused review cleared this design at
`1d08a9f30a87066ea0a2e3b3274a22210cdfa57d` with zero findings. That
clearance authorizes the exact-source implementation plan, not Task 0 runtime
or product RED.

### 8.6 Exact-source plan-construction probes

The plan appendix was built and exercised in scratch before being proposed as
runtime authority:

```text
source root: /tmp/price-truth-runner-plan-final3-20260730
runner lines/bytes: 2140 / 77040
runner SHA-256:
35cda547ac8b1afaba1231d56cb04d703a284cdd81de978397ce7887ac51339e
reporter SHA-256:
09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder SHA-256:
0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
probe summary SHA-256:
47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48
```

The appendix extraction command reproduced the source byte-for-byte at
`/tmp/runner-extracted-final3-check.py`; both files had the runner hash
above and compiled under the pinned interpreter. The dual-role FD check lives
in `pytest_configure`, so module-mode preflight succeeded without
`PRICE_TRUTH_PROGRESS_FD`.

The pristine probe suite completed with every check true:

```text
natural pass:       complete_pass; progress 2; EOF; no signal
SIGINT arm:         unresolved_stall; dump true; SIGINT; killed false
SIGKILL arm:        unresolved_stall; dump true; SIGINT then SIGKILL;
                    descendant process group gone
collection identity: control == plugin == 1 exact node; manifests share
                     SHA-256 85e427423e6a...
FD fail-closed:     missing and garbled values both exit 3 in pytest_configure
```

The short constants were probe-only `2/3/1`; the runner source retains
immutable runtime `120/150/10`.

A final authority check extracted the appendix again, without using the
construction copy, into
`/tmp/price-truth-runner-plan-final4-20260730`. It reproduced the exact
`2140/77040/35cda547...` source identity, compiled, and reran all five probes
in `13.48` seconds. The summary again had SHA-256 `47564c644...`; both
collection manifests and `probe.nodes` had SHA-256 `85e427423...`, and the
two malformed-FD children again exited `3` in `pytest_configure`.

Six load-bearing controls were then exercised in separate scratch roots. Their
mutated source bytes and records remain in those construction roots; the plan
now supplies the exact rerun diffs and requires each Task 0 mutation root to
retain `mutation.diff`:

| Mutation | Reconstructed observation |
|---|---|
| M1 delay parent event handling beyond the already expired window | fast-pass became `invalid`, `deadline_breach_without_dump`; the late event could not revive the window |
| M2 move dump beyond deadline | both sleeps became `invalid`, `deadline_breach_without_dump`, no false stall |
| M3 ignore SIGINT | interruptible arm retained stall truth but recorded `killed=true` and SIGKILL |
| M4 self-alter runner between child launches | the first child record remained; the second child was refused before launch by renewed preflight identity checking |
| M5 missing/garbled FD | both child pytest processes exited `3` in `pytest_configure`; module mode remained usable |
| M6 seed prior invalid record | `run-side` wrote incomplete summary and created no T1-or-later directory |

M5 was repeated from a dedicated pristine root at
`/tmp/price-truth-runner-final3-m5-20260730`; both child pytest arms exited `3`,
the parent suite remained green, and its summary reproduced
`47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48`.

A separate valid eight-node sequencer probe mapped eight distinct safe
repository test files to eight slots, passed the same partition verifier, and
produced a complete side summary with eight selected first attempts at
`/tmp/price-truth-runner-final3-sequence-20260730`. Its summary SHA-256 is
`562fd1e646829bd4babb41de983477222cae1f9e76aa1589eddd04ec89340f39`.
It is not the
4,722-node collection proof or a Task 0 baseline; it proves runner sequencing,
per-attempt admission, aggregation, and atomic completion mechanics.

Two additional controls target state that ordinary one-tier probes cannot
exercise:

- `/tmp/price-truth-retry-probe-artifacts-20260730` began with simultaneous
  T0/T1 stalls and proved that both tiers received their one deferred `a2`
  retry before the side closed incomplete.
- `/tmp/price-truth-bank-probe-artifacts-20260730` first completed all eight
  tiers, then had one banked non-passing artifact altered. Reuse failed closed
  with `banked non-passing artifact changed`, wrote an incomplete summary, and
  named `base-T0-a1` as the invalid attempt.

These are plan-construction observations. PID/timestamp/record hashes are not
promoted to acceptance constants. Section 8.7 records the independently
cleared Task 0 run that extracted the appendix into fresh
`/tmp/price-truth-tier-v2`, regenerated preflight, and repeated the probes and
mutations before the first real tier launch.

### 8.7 Deterministic-v2 Task 0 EOF/exit blocker

Focused review of plan tip `00d35376` returned GREEN with zero findings and
authorized tiered Task 0 v2. Steps 1-4 reproduced every value in Section 2.
The clearance transcript is:

```text
path: /tmp/price-truth-task0-clearance-20260731.txt
SHA-256: e90c0f5313995a9b5659d05444fedf69be5b7e115359a3ebade6c72556db5c38
branch/head: codex/price-collection-truth / 00d35376511b8bd28c16dd9c40415e0ddbc533ab
```

The focused gates created an ignored 143,360-byte
`data/profile_state.db` with SHA-256 `fcfbadad...` in the isolated worktree.
The runner refused before creating a child attempt because `data/` was not
empty. The file was proven SQLite-integrity `ok` and moved reversibly, with
its zero-byte WAL and 32,768-byte SHM companions, to:

```text
/tmp/price-truth-tier-v2/task0-preprobe-generated-data/
```

The integrity read created the WAL/SHM companions after the main file was
inspected, so all three were archived before retrying. `data/` was then empty.
This was a pre-launch setup rejection, not a runtime attempt or `invalid`.

Exact v2 control identities were:

```text
runner:         35cda547ac8b1afaba1231d56cb04d703a284cdd81de978397ce7887ac51339e
reporter:       09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder:        0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier map:       3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
base preflight: 5c8484deeb9188228e4bc730e283d3f67f00a91fd5cf3bcbb32558a1079ac320
probe summary:  47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48
```

The mandatory probe suite returned all five checks true. M1-M4 retained exact
mutation diffs with SHA-256 values
`6a5d13b5.../7f463def.../bc7e8457.../27a064f7...`; each owning assertion
matched. M5 reproduced the same all-true probe summary while both malformed-FD
children failed in `pytest_configure`. M6 seeded one prior invalid record,
returned nonzero, wrote `complete=false` with
`invalid_attempt=base-T0-a1`, and created no T1 attempt. The pristine runner
and fixture hashes were rechecked before runtime.

Runtime then produced:

| Tier | Outcome | Last active boundary | Progress | Duration / cleanup |
|---|---|---|---:|---|
| T0/a1 | `unresolved_stall` | `test_route_rejects_unreviewed_interval_with_typed_422` | 1173 | 172.969s; dump; SIGINT then SIGKILL; group gone |
| T1/a1 | `unresolved_stall` | `test_scheduler_start_stop` | 549 | 158.503s; dump; SIGINT; group gone |
| T2/a1 | `unresolved_stall` | `test_anthropic_calibration_raises_structured_refusal_before_text_extraction` | 769 | 164.864s; dump; SIGINT; group gone |
| T3/a1 | `invalid` / `pipe_eof_while_child_running` | post-last-progress, no active node | 1180 | 20.334s; no dump; SIGINT; group returned 0 and was gone |

The first three classifications are valid unresolved stalls. T3 is the first
invalid and closes the side. Its decisive artifacts show:

```text
expected/collected/seen/non-passing: 590 / 590 / 590 / 0
progress events: 1180 = 2 * 590
report exitstatus: 0
terminal summary: 580 passed, 10 skipped, 9 warnings in 18.86s
active node at EOF: null
deadline phase: post_last_progress
record SHA-256: d89eb6a4ffaefca68d65f9969d2452a08af194ab786f14b7d81ce3f8db860385
transcript SHA-256: 278b9827345f1ead84eca65304b38b1ba856a4627ec3587b014ba26cf1bef831
progress SHA-256: b1110e5ad9747f44e8d1582be4b571931ce7828f53e24f6edeed73eadb1aa39a
report SHA-256: 60190dc3405b48e1eb74199eeacbf2fcf49377f7aac7de96175087221c53c1b8
```

The progress writer closed after all node events and the reporter had already
atomically written a complete result, while the parent still observed
`process.poll() is None`. The reviewed runner therefore applied its closed
EOF rule, marked the attempt invalid, and sent SIGINT. The process group
reported return code `0` and disappeared about 10ms later. These facts do not
permit retroactively calling T3 a natural pass because the runner injected a
signal. They do prove that `EOF while child is running` conflates an
observable final-exit handoff with malformed early EOF.

The atomic side summary has SHA-256 `c7a252a3...`, `complete=false`,
`invalid_attempt=base-T3-a1`, unresolved tiers `[0,1,2]`, and no selected
attempts. T4-T7, all deferred retries, the monolithic diagnostic, protected
Step 6, and product RED were not started.

All pre-manifest non-`__pycache__` files in the v2 root are listed in:

```text
/tmp/price-truth-tier-v2/task0-v2-invalid-manifest.sha256
lines/bytes: 1154 / 163262
manifest SHA-256:
a202dfd4fb160c15ca4a0e517dcc79641dec1ebfd25dba7b1439a4acc63fbb8e
```

After the stop, isolated `data/` was empty and the runner records
`cleanup_complete=true` for every runtime attempt. The branch was clean before
this evidence edit. The two protected main-worktree drafts retained SHA-256
values `4921194a...` and `79d4eac9...`; neither was edited, staged, moved,
deleted, or used as authority.

Restarting the same v2 runner is unauthorized. The next gate is a separately
reviewed amendment to Design Section 13.6 and the exact runner. It must retain
immediate invalidation for partial buffers, unbalanced progress, incomplete
node counts, malformed reporter evidence, or a live group beyond a fixed
bound. It must separately decide a bounded post-EOF exit handshake for the
fully complete shape above, without using transcript text for node accounting.
The amendment needs a deterministic EOF-before-`poll()` probe plus negative
controls proving that early EOF and leaked descendants still fail closed.
No grace duration or exact implementation is authorized by this blocker.

### 8.8 Approved v3 handshake design; focused spec review next

Independent review of blocker packet `18ab76f9` returned GREEN with zero
findings. The review confirmed that v2 bounded three stalls correctly and that
T3 exposed a transport-ordering ambiguity rather than an admissible pass. It
also confirmed that every v2 artifact remains evidence only.

The user then approved the following docs-only design amendment:

- protocol identity advances to `price-truth-tier-v3`, using a fresh
  `/tmp/price-truth-tier-v3`; v1 and v2 roots remain frozen and forbidden;
- `EOF_LEADER_HANDSHAKE_SECONDS` is a complete one-second convergence budget
  after the first clean EOF or natural leader-exit observation;
- `PROCESS_GROUP_DRAIN_SECONDS` is a separate complete one-second budget after
  both EOF and leader exit are observed;
- observing EOF transfers control out of the 150-second no-progress machine,
  so the no-progress and transport clocks cannot race;
- the group-drain stage applies whether EOF or leader exit is observed first;
- successful convergence injects no signal and still passes through every
  unchanged terminal-summary, reporter, manifest, progress-count,
  data-boundary, and Section 12.3 admission check; and
- partial buffers, active/incomplete progress, either transport-stage timeout,
  malformed final evidence, or a group that survives its complete bound
  remain `invalid`.

One pinned scratch probe must force both successful stages. Its
`pytest_sessionfinish` hook starts a same-PGID short-lived descendant without
the progress descriptor, closes that descriptor, and sleeps for 0.5 seconds.
The descendant outlives the leader but drains within stage two. Admission
requires `complete_pass`, no injected signal, and ordered
`leader_exit_after_eof`, `process_group_drain_started`, and `group_drained`
timeline events. M7a zeros only the stage-one bound; M7b zeros only the
stage-two bound. Each mutation must independently turn this same probe
`invalid` in its owning stage.

This section records a reviewed user decision, not implementation evidence.
No v3 runner, appendix, fixture, preflight identity, probe-summary SHA,
mutation artifact, or runtime result exists yet. The current v2 exact-source
plan remains blocked and cannot authorize another run. Focused review of the
amended design is the sole next gate; only after it clears may the exact-source
plan replace the appendix, update all identities and predicted hashes, and
seek separate review.

### 8.9 V3 exact-source plan construction

Focused review of the complete v3 handshake amendment at `6c89d4a1` returned
GREEN with zero findings. This section records plan-construction observations,
not Task 0 runtime admission. The official `/tmp/price-truth-tier-v3` root
does not exist; all work used separately named `*-plan-*` roots. Frozen v1/v2
artifacts were not used as runtime inputs, changed, moved, or deleted.

The final exact runner is `2,413` lines / `89,789` bytes with SHA-256
`bb5d2245071aa48f8f0ad4e28a0966aa26744f213dcec65a69d947a383fd9de9`.
Unchanged identities remain reporter `09d2bc52...` and builder `0f0421f8...`.
New fixture identities are:

```text
probe_eof_handshake/conftest.py       6252ff7bec61796d20cfc0d2b3622ed05b73bf45220c251a9b9747a5f0faa74a
probe_eof_handshake/test_handshake.py 3e2f09ac4d7652b2382e58fb11455e4c9584274b7b6debf03ededa9b1406efa6
probe_eof_handshake.nodes             35a5e9ab7a9d38f9650d368ee1c09836dcdfb6aefd3c758ba4a674c66595b83c
```

RED preceded implementation. With the unchanged v2 transport rule, the new
fixture emitted both required progress events and clean EOF but became
`invalid/pipe_eof_while_child_running`; the parent injected SIGINT. Its RED
record, transcript, progress, and preflight identities are preserved under
`/tmp/price-truth-tier-v3-plan-construction/red-evidence/`.

The pristine v3 suite then closed all six summary checks:

```json
{
  "collection_identity": true,
  "eof_exit_handshake": true,
  "fd_fail_closed": true,
  "pass": true,
  "sigint": true,
  "sigkill": true
}
```

The deterministic summary SHA-256 is
`9f664ea7608385edaf568ae7f35cc94fa5301fea7dd798ea0dd65b14881c1e87`.
After the plan appendix and fixture markers were written, a fresh extraction
into `/tmp/price-truth-tier-v3-plan-review-extract` reproduced all four blobs
byte-for-byte, compiled the runner, and reran the six-check suite to the same
summary SHA. That extraction, not the construction copy, validates the plan's
executable source boundary.
The handshake record was `complete_pass`, injected no signal, and recorded
`leader_exit_after_eof -> process_group_drain_started -> group_drained`.
Observed construction timings were about `0.534s` for EOF-to-leader
convergence and `0.212s` for group drain, inside separate `1s` budgets.
A non-admission reverse-order control also completed naturally and recorded
`child_exit_observed_before_pipe_eof -> pipe_eof_after_leader_exit ->
process_group_drain_started -> group_drained`, proving that group drain is not
specific to the EOF-first order.

All load-bearing controls were reconstructed from final pristine bytes:

| Control | Owning observation |
|---|---|
| M1 delayed ready-event handling | invalid `partial_progress_event_at_eof`; zero admitted progress; transport-terminal priority retained |
| M2 dump after deadline | both sleeping arms invalid without a current-window dump |
| M3 ignore SIGINT | SIGINT arm required ordered SIGKILL; only `sigint` summary check failed |
| M4 mutate runner after first child | fast pass remained; handshake launch was refused by renewed preflight |
| M5 missing/garbled progress FD | both pytest children failed in `pytest_configure`; parent suite remained valid |
| M6 seed prior invalid | side summary closed incomplete before T1 |
| M7a zero stage-one bound | handshake invalid `pipe_eof_while_child_running`; stage two never started |
| M7b zero stage-two bound | stage one succeeded; handshake invalid `pipe_eof_with_live_process_group` after drain timeout |

M1/M2/M3/M4/M7a/M7b exact mutation-diff SHA-256 values are
`1467fd65...`, `6405a626...`, `9232528b...`, `c6372e35...`,
`aa15cc5f...`, and `9732466f...`. M4 separately preserved its pre-execution
mutation diff and post-self-edit runtime drift (`fa1ee4f9...`).

A fresh eight-node sequencer control under protocol v3 selected all eight
first attempts, produced zero non-passing nodes, and completed without retry
or signal. Its construction summary at
`/tmp/price-truth-tier-v3-plan-sequence-v3/base-summary.json` has SHA-256
`7f941eef5ba1b0df4be209bccbc23e07b8c4cb16dff6b101ef18f77ade49e4b1`.
This remains a control-plane observation, not the 4,722-node base.

### 8.10 Deterministic-v3 Task 0 unresolved-tier blocker

Focused review of exact-source plan tip
`1a8379e72216e0c109f9498caf64abfa593b299c` returned GREEN with zero
findings and authorized Task 0 v3. Steps 1-4 reproduced every value in
Section 2. The official root was created fresh at
`/tmp/price-truth-tier-v3`; frozen v1/v2 roots were proven present but were
not read, written, moved, deleted, or imported.

Focused gates created an ignored 143,360-byte `data/profile_state.db` in the
isolated worktree. It had inode `90843007`, SHA-256
`fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d`,
and SQLite integrity `ok`. It was moved reversibly to:

```text
/tmp/price-truth-task0-pre-v3-data-IPrLEYI6/profile_state.db
```

The first pristine probe invocation then refused before creating a child
attempt because the corresponding WAL/SHM files had appeared after the main
file was moved. No process held them. They were preserved in the same
directory with their original identities:

```text
profile_state.db-wal  inode 90843008  size 0      SHA e3b0c442...
profile_state.db-shm  inode 90843009  size 32768  SHA fd4c9fda...
```

This was a pre-launch data-boundary rejection, not a probe result or runtime
`invalid`. The isolated `data/` directory was empty before the successful
probe suite and every runtime launch.

The exact v3 control identities were:

```text
runner:          bb5d2245071aa48f8f0ad4e28a0966aa26744f213dcec65a69d947a383fd9de9
reporter:        09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder:         0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier map:        3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
probe preflight: 7362186a8a726fe1ca626178eb80cf439d2c9981cb6a1247aae20fb3018e9c4a
probe summary:   9f664ea7608385edaf568ae7f35cc94fa5301fea7dd798ea0dd65b14881c1e87
base preflight:  a50142ef8ac029be43250a886af1a7e0af3d120982a7b6c1221d481eb077e451
```

The builder produced the reviewed
`591/591/590/590/590/590/590/590` distribution. All tier collections had
4,722 total and 4,722 unique nodes; their sorted union was byte-equal to the
canonical `fcdb1b7d...` stream. The pristine probe suite returned all six
checks true. M1-M4 and M7a/M7b reproduced exact mutation-diff SHA-256 values
`1467fd65...`, `6405a626...`, `9232528b...`, `c6372e35...`,
`aa15cc5f...`, and `9732466f...`; every owning assertion matched. M5 proved
both malformed progress descriptors fail in `pytest_configure`. M6 seeded a
prior invalid record, wrote an incomplete side with no selected tier, and
created no T1 attempt. The pristine runner and fixtures were re-hashed after
all controls.

The single official `run-side` invocation attempted all initial tiers, banked
three natural completions, then gave each unresolved tier exactly one
ascending retry. Its closed results were:

| Tier | Attempt(s) | Closed outcome | Last active node / natural result | Progress | Duration(s) / cleanup |
|---|---|---|---|---:|---|
| T0 | a1, a2 | `unresolved_stall`, `unresolved_stall` | `test_route_rejects_unreviewed_interval_with_typed_422` both times | 1173, 1173 | 172.935, 173.140; dump; SIGINT then SIGKILL; group gone |
| T1 | a1, a2 | `unresolved_stall`, `unresolved_stall` | `TestRunAgentQuery::test_successful_query` both times | 567, 567 | 158.727, 158.709; dump; SIGINT; group gone |
| T2 | a1, a2 | `unresolved_stall`, `unresolved_stall` | `test_anthropic_calibration_raises_structured_refusal_before_text_extraction` both times | 769, 769 | 164.614, 164.581; dump; SIGINT; group gone |
| T3 | a1 | `complete_pass` | 590 collected / 590 seen / 0 non-passing; 580 passed / 10 skipped | 1180 | 20.504; natural validation; no signal |
| T4 | a1 | `complete_pass` | 590 collected / 590 seen / 0 non-passing; 586 passed / 4 skipped | 1180 | 27.426; natural validation; no signal |
| T5 | a1, a2 | `unresolved_stall`, `unresolved_stall` | `test_get_activity_http_limit_uses_typed_400[0]` both times | 685, 685 | 166.604, 166.685; dump; SIGINT then SIGKILL; group gone |
| T6 | a1, a2 | `unresolved_stall`, `unresolved_stall` | `TestHealth::test_status` both times | 31, 31 | 165.678, 165.657; dump; SIGINT then SIGKILL; group gone |
| T7 | a1 | `complete_nonpassing` | 590 collected / 590 seen / 6 non-passing; 6 failed / 573 passed / 11 skipped | 1180 | 29.135; natural validation; no signal |

All ten stalled attempts had a current-window 120-second dump,
`deadline_phase=active_node`, `invalid_reason=null`, and an empty data boundary
before launch. Each retry stopped at the same node as its initial attempt.
The portal/request shape recurred in T0, T5, and T6. T1 stalled inside a bare
`asyncio.run`; T2 stalled in a bare `asyncio.run` while the separately known
pyrate-limiter thread was also present. These observations add trigger
evidence to `EIR-005`; they do not identify one shared source seam or prove
that the pyrate thread is causal.

The unresolved record/transcript/progress SHA-256 triples are:

```text
T0/a1 e1b65581a30b0814b2b70e0b668d696a91fa6322b5d1e1d834f5770745eec1fd / 2836ff570baccaf9194ac32751515d133ca5fc453bc8700dd86441f1ab16a7d2 / 416b3090574ca2386696be4078f9ebf43124b42bab3189f9a4cd50333c9f6a73
T0/a2 7fe70caf10cf4f70a821851964c91f04e79b7f92af5e122afcb0dbb84dc26dc9 / f3fe3496c53adaad454333d44cc99ae834f28ba3892786c712f49d9302d9f0c0 / 2e8dd364b79dbca6e4e9e343e23ba9938afc00076744c08c6e7f47f3380588c4
T1/a1 e62f913a3c08f40e2ca90069eca0c32f5147c27b2a85ee0419af453c0d5e6819 / 8bde0643ae5e3c259cef25f282f3d4d5cec71a991a489eeb141ebe64bf71d3de / dcecc792b3b8a89c21c53a80efca21104fd1e0e60d31050b64f034df25dd32f1
T1/a2 132a4c6f4bb0e9bcf2deffd2d6be4ab73bb6f06cf3eac3b9a2620c6f13fb2a2d / 789798371d7bf757ab4de15dfcb238cf73923dc97b6b141de82f86893a7ae740 / 90c56f312a2479545705ceaac9ff7a87fecdc6c09c98be7d0e193e479a448c7a
T2/a1 97429de260b8c6b60166d5327de360b2ce59bd8eead28e599bdd1f4e73b3abcd / b42bc24d472f353865dc0e90506f3276bc2be2aeb2424a8c68127b8d5e55aea9 / 671948d1b39fca9e1e35ddf594173d513beecd4f1d8d2f21230e735af41e8723
T2/a2 9f2b6a17f12a721971d223b77320fc527414b65bbdc2a5e4cdc4581b53a30820 / 74255b5696747fe96a72373109111b9245da8e5d328b8e27ccebe8f8c1ae1e60 / 572de51160a27354e7d49a8c9e091fc849e9b458b0528157a2dd576fbc9f4648
T5/a1 d90c8677d50ad37c44d1f62310f1aaf6ed86e8ed476976d2719df4ed82a02651 / 5724800c8015f0f10228243dc9c664192026c0cf91e94fb732da364ab688a442 / 4d1e179312dcbba0835f86ea378113310bd6db5fb33149f92cb0520e70bda4fd
T5/a2 69d7c6a14fef3d84d0b3f98e02b7f3e449f3cfd0f0c282c85ec2d1cc614a34c4 / 5091aff2d53dc0166c5555f8b4f9bf98e5a892690625feba154b54d3812d9ac5 / 69f20f251d57b633a3359750a1cf3df37b7d43ec5f94374b925ec67c030a2470
T6/a1 45f4a4f559f3635a72d2fa4cc8e3a4f4af3da11abcba8a82e53fcb3fe23e3bc5 / 4264ba9fe645fa187acb6dc9e4cafb5fea80913bcde4eb0d93f0c4addbd48cf7 / 8fcbd5f675eecf56e4c2e8e36716e7d888ee3319544d49d96f6d629defda621f
T6/a2 edc110b54581cbe279c72ce52130e50e925415c48312e10d7720a8f2523b2d04 / da7fdafedb97ff69d79ee586c60d8037aba91df90669c83218c456cf20f814ca / af95d651137d7c9a3970ca6f8a6ddfb94c1077df8b6a494d122935e00d025946
```

The selected attempts retain complete reporter-derived artifacts:

| Selected attempt | Record | Transcript | Progress | Report | Non-passing |
|---|---|---|---|---|---|
| `base-T3-a1` | `ffc5cb4d...` | `7bf0ed47...` | `0298b383...` | `60190dc3...` | empty `e3b0c442...` |
| `base-T4-a1` | `35c89093...` | `5103da50...` | `3dd4e6a9...` | `854db0c1...` | empty `e3b0c442...` |
| `base-T7-a1` | `c8d89d54...` | `1eb77733...` | `1bf78fa6...` | `db6e0a03...` | six-node `b6c3c718...` |

Their exact record/transcript/progress/report/non-passing SHA-256 tuples are:

```text
T3/a1 ffc5cb4d8255f6fa6ca8d56f17ff3ef532973fad6d5b6de5e3e15861c27bb51e / 7bf0ed477672b1958820f6409ae02c5f80f456d3c49c4c4f801bb7ba0685a4a5 / 0298b3830f1d11726285da9ea4c5011b6f5e6196d44c35d6801d0af9126e26a5 / 60190dc3405b48e1eb74199eeacbf2fcf49377f7aac7de96175087221c53c1b8 / e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
T4/a1 35c89093445ae418ce03d8f02e3c0b9055a2e8053e6e24ae3a400a06f473c6c3 / 5103da500bff65d07c4c1cec5492d78c5e84784f8b42cdbd0fae80e9eae9e6a1 / 3dd4e6a926fde77f62fcbedbef52e9585134bb1354b4a3e6415cbc8d5b3aa3ca / 854db0c11f2954cf631d09814a53dd10b6567aff9fc5319360b669d6ce0755e9 / e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
T7/a1 c8d89d546aede6e8eb6690613b78ac0bb74c16bf1c25af686d019927a8315fca / 1eb77733401e1ce1c7ad3f95ad612525ea559579c8d11d65781f90f125ee5cdf / 1bf78fa6c7f425a141bc4b049d830d0a04bd6caa06cf5fa6ac869cbc4a206578 / db6e0a03db7accdeac29e09b9d8fcf5b5125cf7cc41842da7cdd35fa92bd9f13 / b6c3c71819ca38a61ea6f969316b847836fa6f4a9e15cdb0ac73d47bc313a58b
```

T7's six nodes are all
`tests/test_chatgpt_oauth_callback_server.py` failures caused by this
sandbox's loopback-bind `PermissionError`. They are a valid selected
non-passing artifact, but cannot become a base union while five tiers remain
unresolved.

The atomic side summary has SHA-256
`a5686da09e1715e1ea81b618826c956b96649bf12075ccf230a387c87782b198`,
`complete=false`, `invalid_attempt=null`, selected attempts
`{3: base-T3-a1, 4: base-T4-a1, 7: base-T7-a1}`, and unresolved tiers
`[0,1,2,5,6]`. The runner process itself returned zero because it emitted a
closed incomplete summary; shell exit alone is not admission. No complete
`base-nonpassing.nodes` was produced.

Every runtime record has `cleanup_complete=true`. Generated worktree files
were moved into the owning attempt's `data-after/` directory, and isolated
`data/` was empty after the side closed. The 3,262 pre-manifest,
non-`__pycache__` files are pinned by:

```text
path: /tmp/price-truth-tier-v3/task0-v3-incomplete-manifest.sha256
lines/bytes: 3262 / 532402
manifest SHA-256:
ff189a4433b571c671ef7e4db82e63c94071d869e4ed48410f2a65c25e622f75
validation: 3262/3262 sha256sum checks OK
```

The monolithic diagnostic, protected Step 6, product RED, provider calls,
production writes, and repair did not run. The branch remained clean before
this evidence edit. Main-worktree draft hashes remained `4921194a...` and
`79d4eac9...`; neither draft was edited, staged, moved, deleted, or used as
authority.

Stop Condition 11 therefore applies at tier granularity. Re-running unchanged
v3 or adding a third retry is unauthorized. The next gate is the separately
reviewed `EIR-005` machine-state observer spec already named in the register:
capture AnyIO wakeup-socket state, selector registrations,
`asyncio.all_tasks`, system load/file descriptors, and SIGINT
receipt/response during the next matching window. This is diagnosis of a
repeated test-runtime blocker, not a new product architecture or a claim that
the price collector itself is defective in these five tests.

### 8.11 Native execution-boundary Task 0 completion

Before the reviewed observer campaign launched, a smaller controlled A/B
identified the missing boundary variable. The exact 942-byte asyncio wakeup
probe at SHA-256 `10647c1e...` produced queued but unfired work in `3/3`
Codex managed-sandbox runs and normal callback delivery in `3/3` native
runs. Direct `socketpair.sendall()` returned `EPERM` only in the managed
sandbox. CPython 3.10.12 source confirms the causal order: enqueue into
`_ready`, attempt the self-pipe send, and swallow `OSError` outside debug
mode. The closeout packet is detailed in the lifespan diagnosis evidence
Section 14.

The original `/tmp/price-truth-tier-v3` remains frozen. Exact reviewed static
inputs were copied byte-for-byte into a fresh, non-mixed root:

```text
/tmp/price-truth-tier-v3-native-boundary
runner:   bb5d2245071aa48f8f0ad4e28a0966aa26744f213dcec65a69d947a383fd9de9
reporter: 09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder:  0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier map: 3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
base preflight: dce06838645768d0e1a2f112fd46d60c4a2f1a88a05c32807c56f8fb0b2a9c1e
```

The pristine probe suite again closed all six checks. Its summary remains
SHA-256 `9f664ea7608385edaf568ae7f35cc94fa5301fea7dd798ea0dd65b14881c1e87`.
The unchanged `run-side` then completed every tier on its first attempt:

| Tier | Outcome | Duration (s) |
|---|---|---:|
| T0 | `complete_pass` | `13.987` |
| T1 | `complete_nonpassing` | `16.947` |
| T2 | `complete_nonpassing` | `20.905` |
| T3 | `complete_pass` | `20.304` |
| T4 | `complete_pass` | `48.183` |
| T5 | `complete_nonpassing` | `24.923` |
| T6 | `complete_nonpassing` | `24.444` |
| T7 | `complete_pass` | `30.100` |

The atomic base summary is:

```text
complete: true
selected attempts: base-T0-a1 through base-T7-a1
unresolved tiers: []
invalid attempt: null
collected union: 4722 / fcdb1b7d...
non-passing: 27 / 7aafce5d...
base-summary SHA-256: f83ce823b09ea8fc16342e99ba164bb4ecedf21458125d17aa038f470b1f7a73
```

All 27 nodes are the existing EIR-002 families; no sandbox loopback-bind
failures remain. The full regular-file evidence manifest at
`/tmp/price-truth-tier-v3-native-boundary-manifest.sha256` contains `6655`
entries and has manifest-file SHA-256 `eb8ed98d...`.

The required monolithic diagnostic also naturally ran all `4722` nodes and
printed `27 failed / 4623 passed / 72 skipped` in `275.56s`, with no signal
or stall. It remains classified `invalid`, not pass, because v3's
diagnostic-only terminal-summary regex accepts `... in 12.34s ...` but not
pytest's longer-duration suffix `... in 275.56s (0:04:35) ...`.
Transcript SHA-256 is `81bd1e56...`; report JSON SHA-256 is `aba2d230...`.
This is a bounded diagnostic-classifier defect, not a missing node or an
admission failure. The tiered base remains the sole admitted side.

The run generated one untracked `src/data/cache/risk_free_rate.json`
(SHA-256 `abf04d73...`). Its birth time `12:55:29 +0800` falls inside T6,
whose `tests/test_rate_curve.py` is the only test caller of
`get_risk_free_rate()`; T7 has no consumer. The monolithic diagnostic later
updated the same cache. The side began without `src/data`, so this is a
generated execution artifact rather than pre-run contamination. After all
processes exited it was moved reversibly to
`/tmp/price-truth-native-task0-src-data-20260731T1410`; the isolated worktree
is clean and `data/` is empty. The tip side must begin with `src/data` absent
and apply the same post-run quarantine.

Focused closeout review cleared `73d5305e..0075ba9e` with zero findings.
`master` and `codex/price-collection-truth` then fast-forwarded to exact
reviewed tip `0075ba9ef49ff3cd4a71d1c6c42d89de7046f7d8`; product RED is
authorized.

The non-blocking review advisory about pytest's optional `(h:mm:ss)` suffix
was explicitly ruled before tip execution. V3 remains unchanged because the
defect is fail-closed and the planned product delta does not add nodes to the
base's slowest tier, T4. If the suffix alone causes a naturally completed tip
tier to be classified `natural_result_validation_failed`, the side must stop
without reinterpretation or a speed-based retry. A separately reviewed v3.1
regex amendment and a complete tip-side rebuild are then required. This
ruling does not promote the invalid monolithic diagnostic above, weaken
reporter/manifest admission, or authorize any other invalid result.

## 9. Review Resolution

Plan F1 was resolved at `9d1e648a`: the mounted frontend node now includes the
existing `Settings provider config authority` describe prefix, matching both
predicted hashes. The 26-slot advisory now requires the exact temporary
mutation diff in this packet when implementation eventually proceeds.

Independent harness implementation review returned GREEN for
`db7f2240..2edf12e1`; `master` then fast-forwarded to exact reviewed tip
`2edf12e1`. Merged verification reproduced full `4722/fcdb1b7d...`, agents
`31/78d7cdbe...`, owned `2/5e1e62ac...`, and `2 passed`. The price branch was
rebased while preserving both reviewed priority-map histories.

The historical blocker at `test_providers_endpoint` is structurally removed.
Focused review of the rebased handoff returned GREEN, and the restarted full
suite proved that converted exposure passes before the untouched
`test_api.py::TestHealth::test_status` family stalls. The instrumentation
therefore did its intended job, but the repository could not produce the
complete same-environment baseline required by the then-current monolithic
plan. Silently excluding a node, accepting partial output, running protected
Step 6, or starting product RED remained prohibited pending a separately
reviewed resolution.

The separately reviewed diagnosis later selected
`V6 ambient_or_machine_state_dominates`; it authorized no source seam and
transferred the unresolved behavior to `EIR-005`. The user then selected a
tiered full-collection protocol instead of waiting for a clean monolithic
window. The contract preserves Stop Condition 11 at tier granularity and
states explicitly that fresh-process tiered results are not directly
comparable with historical monolithic runs. Focused review cleared that
amendment at `3863b3be`; the subsequent runtime controller failed its reviewed
termination and process-identity protocol as recorded in Section 8.4. Focused
review then cleared deterministic control-runner design at `1d08a9f3` and its
exact source plan at `00d35376`. Deterministic-v2 Task 0 then stopped at the
Section 8.7 EOF/exit invalid. Section 8.8 records the approved v3 design;
Section 8.9 records its exact-source plan construction. Focused review cleared
that plan at `1a8379e7`; Section 8.10 records the authorized runtime. V3
resolved the transport invalidation, but five tiers remained unresolved after
their one retry. Those statements describe the historical sandbox-bound
attempt and do not describe the later native execution boundary.

Section 8.11 supersedes that historical blocker. EIR-005 is closed by the
managed-sandbox/native causal A/B; no observer campaign ran. The exact v3
base completed under the native execution boundary. Product implementation
then proceeded, and Sections 3-7 record its RED/GREEN work, initial review,
review-fix commit, and complete fresh-root native tip.

## 10. Integration And Read-Only Release Observation

Independent implementation re-review cleared `0075ba9e..66ef3bbc` with zero
blocking findings. On 2026-07-31, `master` fast-forwarded from `0075ba9e` to
the exact reviewed tip `66ef3bbcd39e2927cf39a23844fc1cb8e2328f00`
without a merge commit.

### 10.1 Merged verification

- Backend focused: `168 passed`.
- Canonical backend collection: `4739`,
  `a72bbd36dfad3d36aee2e6630e6024ec9fb4e910bebaf1363d44df8a1aa204dd`.
- Frontend: `96` files / `1076` tests passed.
- Canonical frontend collection: `1076`,
  `de48671aa1d3f70cb87166e3f5b026804e206ac31f8e29fe7e74b38cde9448d5`.
- Typecheck/build exited zero. The visible-literal scanner returned
  `36/20/0/20` twice.
- The two protected untracked drafts remained byte-identical at
  `4921194a...` and `79d4eac9...`.

The independently reviewed native full-suite result at the exact merged tip
remains `27 failed / 4640 passed / 72 skipped`, with the same exact
`27/7aafce5d...` EIR-002 non-passing set as base and no new failures.

### 10.2 Read-only pre-run facts

The merged desktop first loaded with `ARKSCOPE_DISABLE_SCHEDULER=1`. This
environment-only guard did not change the saved setting:
`schedule.ibkr_prices.enabled=true` and
`schedule.ibkr_prices.interval_minutes=600`.

All observations used SQLite URI `mode=ro` plus `PRAGMA query_only=ON`.
Before the merged natural cycle:

- latest `collect.ibkr_prices` was historical job `18212`, started
  `2026-07-31T00:21:24+00:00`, persisted `succeeded` by the pre-merge runtime;
- LCID price meta reported `last_success=2026-07-31T00:24:18+00:00`,
  `last_bar_datetime=2026-07-29T23:45:00+0000`, `last_error=null`, and
  `rows_added=38`;
- LCID had 60 / 59 / 64 stored 15-minute rows on July 27 / 28 / 29 and no
  July 30 row;
- Coverage v2 reported July 27 complete, while July 30 was
  `indeterminate_tickers`: `149 complete / 0 partial / 1 unknown (LCID)`;
- `market_data.db` was 3,409,027,072 bytes, inode `127284871`, mtime ns
  `1785485926436141726`; `profile_state.db` was 44,650,496 bytes, inode
  `127284276`, mtime ns `1785492173057373242`; and
- both databases returned `integrity_check=ok` and zero rows from
  `foreign_key_check`.

### 10.3 Natural-cycle result

The desktop then restarted normally. No Run control, provider/Gateway probe,
cadence change, retry experiment, or LCID repair occurred. The unchanged
scheduler created job `18329` at `2026-07-31T10:21:39+00:00` and completed it
at `2026-07-31T10:24:36+00:00`:

```json
{
  "status": "succeeded",
  "tickers_scanned": 150,
  "succeeded_ticker_count": 150,
  "gaps_found": 1,
  "rows_added": 62,
  "error_count": 0,
  "unresolved_after_fetch_count": 0,
  "unresolved_after_fetch_tickers": []
}
```

The corresponding `provider_sync_runs` row is `784`, status `succeeded`.
LCID price meta now reports
`last_success=2026-07-31T10:24:36+00:00`,
`last_bar_datetime=2026-07-30T23:45:00+0000`, `last_error=null`, and
`rows_added=62`. LCID has 62 July 30 rows from 08:00Z through 23:45Z.

Coverage v2 then reported both July 27 and July 30 complete. July 30 changed
to `150 complete / 0 partial / 0 unknown`, with no provider errors or
unmatched RTH rows. This is a **Resolved** outcome under the approved contract.
It proves only that the original local zero-row target became present after
the natural fetch; it does not identify the successful upstream path or the
reason for the earlier absence.

After the cycle, `market_data.db` remained 3,409,027,072 bytes at inode
`127284871`; `profile_state.db` was 44,658,688 bytes at inode `127284276`.
Their mtimes advanced to `1785493476277478576` and
`1785493476312478758` ns respectively. Both again returned
`integrity_check=ok` and zero foreign-key violations.

Merged verification and the bounded release observation are complete. This
slice is LIVE. EIR-002 remains next, followed by root `scripts/` retirement;
calendar-aware price scheduling, extended-hours capture, structured provider
outcomes, and the pacing reference rewrite remain separate work.
