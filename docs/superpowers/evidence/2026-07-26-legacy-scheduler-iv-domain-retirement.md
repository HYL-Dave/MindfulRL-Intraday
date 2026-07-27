# Legacy Scheduler Sources And IV Domain Retirement Evidence

> **Status: IMPLEMENTED - INDEPENDENT REVIEW PENDING**
>
> Task 0, both implementation tranches, the migration tool, copied-data proof,
> canonical A/B, and isolated bilingual runtime gates are complete. No
> production archive/apply, provider call, Gateway call, PG call, merge, push,
> or production data deletion has occurred.

## Review And Worktree Boundary

- Behavioral A/B base:
  `7bb7cc29f70ca899a5b598f2322ce181daa17ebe`.
- Plan-review clearance:
  `5f528475420c8de407125bceb32d94050cfa8e14`.
- Task 0 authorization tip:
  `0bdd526112f7975ecf13064a96e2e8672fa16667`.
- Implementation branch: `codex/legacy-scheduler-iv-retirement`.
- Isolated worktree:
  `/home/hyl/.config/superpowers/worktrees/ArkScope/legacy-scheduler-iv-retirement`.
- The worktree uses the reviewed Python environment at
  `/home/hyl/.virtualenvs/llm_app/bin/python` and symlinks the existing npm
  installations associated with the unchanged root lockfile.
- No production data was copied into the worktree. An empty ignored `data/`
  directory exists only because `FileBackend` requires `config/` and `data/`
  directories to identify a project root during tests.

## Task 0 Reproducibility Corrections

Task 0 exposed two docs-only command-shape defects before product edits:

1. Pytest's full collection contains two `scripts/testing/` nodes. The old
   plan recipe filtered only `tests/`, producing `4747` and hash
   `4349171a...` while pytest itself reported `4749`. Including both roots
   reproduces the reviewed `4749` and `e7dc826f...` exactly.
2. The reviewed source counts used source-ID shorthand, but persisted
   `job_runs.job_name` values carry the `collect.` prefix. The Task 0 query and
   plan now name `collect.local_incremental`, `collect.price_backfill`, and
   `collect.iv_history` explicitly.

Neither correction changes product behavior, accounting targets, or an
acceptance constant.

## Canonical Node Baselines

All normalized lists were generated in the isolated worktree with the plan's
reviewed ordering rules.

| Collection | Files | Nodes | SHA-256 |
|---|---:|---:|---|
| Backend full | repository full collection | 4749 | `e7dc826f33c202789f8ad5f43787d1eedd8f288cc55aa4996d0a100761a21b20` |
| Backend focused | 23 | 663 | `415b51d19cac89cb40bb97fb5fdd39296428da3cac8db11b47f98b9770fe07be` |
| Frontend full | 96 | 1072 | `d964d330e9d5935aae3b746dfabe353d4ee5efc70671295127fdaccc8ad243e0` |
| Frontend focused | 9 | 139 | `ccc689302f0a17f89fa2bad12effebcda84819d3bfdbb00a41fde80bcbc0974c` |

Backend node IDs are sorted raw pytest `file::node` values. Frontend node IDs
are sorted repository-relative `file<TAB>name` values. The frontend focused
list is the exact nine-file projection of the canonical full list. During an
exploratory command, Vitest 4 parsed the first file after optional `--json` as
the output path and overwrote `SettingsPostPgExitStorage.test.ts`; the file was
immediately restored exactly from `HEAD` with `apply_patch`, and `git diff`
confirmed zero residual change. The final full/focused hashes above were then
reproduced again from the restored tree.

## Fixed Baselines

- Resource/scanner/foundation suite: `3 files / 43 tests`, all green.
- Resources per locale:
  - Settings: `714`;
  - Explore: `401`;
  - total: `1814`.
- Visible-literal scanner, two consecutive runs:
  `36 candidates / 20 signatures / 0 debt / 20 allowlist`, scope `src/**`.
- Tool count nodes: `3/3` green.
- Central registry: `56`; all three legacy IV-history tools and all three
  retained live/pure option tools are present at base.
- Anthropic bridge: `57`.
- OpenAI bridge: `57`.
- no-PG inventory: `24`.

The first OpenAI bridge-count attempt failed during fixture setup because the
isolated worktree intentionally had no `data/` directory. After creating an
empty ignored directory, the unchanged three-node gate passed. This is an
environment setup observation, not a product test failure.

## Read-Only Production Observation

Observed at `2026-07-26T15:00:46.775243+00:00`. SQLite connections used URI
`mode=ro`. File size and `mtime_ns` were captured before and after the queries;
all source files remained unchanged.

| File | Bytes | `mtime_ns` |
|---|---:|---:|
| `data/profile_state.db` | 43,413,504 | 1785077794255174640 |
| `data/market_data.db` | 3,293,126,656 | 1785075280271015627 |
| `data/options/iv_history/AMD.parquet` | 4,735 | 1772732024459732256 |
| `data/options/iv_history/NVDA.parquet` | 4,701 | 1772732188423298521 |
| `data/options/iv_history/PLTR.parquet` | 4,390 | 1770044155819205290 |
| `data/options/iv_history/PYPL.parquet` | 4,597 | 1772732307750729098 |

Database health:

- profile integrity: `ok`; FK violations: `0`;
- market integrity: `ok`; FK violations: `0`.

Legacy IV payload:

- `iv_history`: `24` rows, `4` tickers (`AMD`, `NVDA`, `PLTR`, `PYPL`);
- dates: `2026-01-30` through `2026-03-06`;
- IDs are non-contiguous, minimum `1`, maximum `248`:
  `1-11, 27, 32, 36, 70, 76, 81, 190, 197, 203, 211, 233, 241, 248`;
- exact schema and `idx_iv_ticker_date` match the reviewed contract;
- Parquet row counts are `9/8/1/6` for `AMD/NVDA/PLTR/PYPL`;
- SQLite and Parquet value multisets are exactly equal;
- canonical value-multiset SHA-256:
  `04e92460bcdc5ff774d6b5cf38bac9252aafa21fadf69a01fdd3a0edf0b8120d`;
- `market_sync_meta` has one `domain='iv'` row with
  `last_success='2026-07-03T00:14:04+00:00'`, `rows_added=0`, and no error.

Historical telemetry and scheduler state:

- `collect.local_incremental`: `1350` `job_runs` rows;
- `collect.price_backfill`: `2` `job_runs` rows;
- `collect.iv_history`: `0` `job_runs` rows;
- `local_incremental` scheduler state is historical `succeeded` with a stored
  result and no error/continuation;
- `price_backfill` scheduler state is the reviewed blank legacy shape;
- `schedule.local_incremental.enabled=false` is the sole matching retained
  profile setting.

These production figures are dated observations. Migration acceptance must
rederive them from a fresh preview; no row count above is a permanent constant.

## Task 0 Result

Task 0 is GREEN after the two documented recipe corrections. No product file,
database, Parquet file, provider, Gateway, PG endpoint, scheduler, or repair
path was changed or invoked. Tranche 1 may begin with RED contracts.

## Tranche 1 Named Checkpoint

The final Tranche-1-owned product commit is the named checkpoint:

```text
TRANCHE_1_TIP=747aa51acc056d98da316153be3edd9c96f90cda
```

This follows the repository's established tranche convention: the final
product commit is the canonical comparison point, and this later evidence
commit records its full 40-character hash. Treating the evidence commit as its
own named hash would require an impossible self-reference.

The Tranche 1 commit sequence is:

```text
9112b21b test: define active scheduler retirement contract
eb3d056a refactor: retire legacy scheduler source identities
747aa51a feat: expose only active schedule controls
```

### RED contracts

Before product edits, the Tranche 1 RED suites collected `165` backend and
`105` frontend nodes. They failed as intended:

- backend: `24 failed / 141 passed`;
- frontend: `8 failed / 97 passed`.

Every failure was an owned retirement or replacement contract. No fixture,
collection, credential, import, or environment failure was accepted as RED.

### Exact node ledger

| Collection | Base | `TRANCHE_1_TIP` | Composition | Tip SHA-256 |
|---|---:|---:|---:|---|
| Backend full | 4749 | 4731 | `+6/-24` | `c914537e3f3458aec4ee2474941898ef90b98ed6459f17b5b70955eca40dd19e` |
| Backend focused | 663 | 645 | `+6/-24` | `067f8cf948168aef4a1bc9a7d35c58eff14c1846d2c69038e34e0dda1db74761` |
| Frontend full | 1072 | 1072 | `+3/-3` | `a6afed7846052078d723bc7d715c5965c4f3c72b04bb77e7fc5f6242d1de16ab` |
| Frontend focused | 139 | 139 | `+3/-3` | `fe9bf6a0e8bcf3675c94387431816a5a44ffebaf257d76f63fe8fb5e82212fa1` |

The two additive NewsWriteMode safety nodes, separate from retirement
replacements, are:

```text
tests/test_data_scheduler.py::test_news_write_mode_classifier_is_exhaustive_for_current_modes
tests/test_data_scheduler.py::test_unknown_news_write_mode_fails_before_provider_adapter_worker_and_telemetry
```

The other four backend additions are exact rename/replacement nodes:

```text
tests/test_daily_update_wrapper.py::test_dry_run_reports_direct_local_collection_without_mirror_controls
tests/test_data_scheduler.py::test_run_source_explicit_tickers_reaches_active_adapter_without_mirror_controls
tests/test_data_scheduler.py::test_schedule_routes_reject_removed_source_ids_before_writes_or_provider_work
tests/test_prices_runtime.py::test_prices_worker_requires_tickers_without_source_selector
```

The 24 backend removals are:

```text
tests/test_daily_update_wrapper.py::test_dry_run_without_sync_db_collect_only
tests/test_daily_update_wrapper.py::test_iv_history_opt_in_only
tests/test_data_scheduler.py::test_blank_price_backfill_history_is_neutral_and_first_run_succeeds
tests/test_data_scheduler.py::test_coverage_derived_price_backfill_is_deliberate_noop
tests/test_data_scheduler.py::test_error_or_result_only_price_backfill_state_fails_closed
tests/test_data_scheduler.py::test_legacy_unproven_gap_manual_continuation_is_rejected_without_worker
tests/test_data_scheduler.py::test_legacy_unproven_gap_scheduler_continuation_is_rejected_without_worker
tests/test_data_scheduler.py::test_local_incremental_has_no_subprocess
tests/test_data_scheduler.py::test_local_incremental_retired_after_p0c
tests/test_data_scheduler.py::test_local_incremental_retirement_does_not_call_local_refresh
tests/test_data_scheduler.py::test_local_refresh_excludes_news_when_pg_exit_audit_cannot_be_read
tests/test_data_scheduler.py::test_post_exit_ibkr_local_refresh_excludes_retired_pg_domains
tests/test_data_scheduler.py::test_price_backfill_does_not_resolve_scope_for_deliberate_noop
tests/test_data_scheduler.py::test_price_backfill_ignores_gateway_lock_but_keeps_source_lock
tests/test_data_scheduler.py::test_price_backfill_source_registered
tests/test_data_scheduler.py::test_run_now_choke_point_guards_scheduled_and_rejects_retired_sources
tests/test_data_scheduler.py::test_run_source_explicit_tickers_and_skip_sync
tests/test_data_scheduler.py::test_run_source_iv_history_retired_before_provider_work
tests/test_data_scheduler.py::test_skip_sync_is_true_collect_only
tests/test_data_scheduler.py::test_skip_sync_message_precedes_legacy_local_news_route
tests/test_data_scheduler.py::test_status_or_continuation_only_price_backfill_state_fails_closed
tests/test_data_scheduler.py::test_status_snapshot_preserves_durable_state_without_planner_metadata
tests/test_data_scheduler.py::test_unknown_tickers_and_provider_errors_never_reach_price_executor
tests/test_prices_runtime.py::test_prices_worker_requires_source_and_tickers
```

Frontend additions and removals are three exact contract renames:

```text
- Settings provider config authority > does_not_render_storage_route_source_badges
+ Settings provider config authority > does_not_render_backend_storage_route_badges_for_active_schedule_rows
- Settings provider config authority > shows_disabled_provider_and_read_only_schedule_states_as_neutral_text
+ Settings provider config authority > renders_disabled_providers_as_neutral_and_all_four_schedule_rows_as_controllable
- Settings backend copy boundary > maps all seven schedule source ids without backend labels
+ Settings backend copy boundary > maps exactly four active schedule source ids without backend labels
```

### Equivalent-environment A/B

The behavioral base was checked out in a temporary detached worktree using the
same Python environment and npm dependency trees as the implementation
worktree. Both sides used an empty local `data/` directory and no production
configuration.

Backend full results:

```text
base: 31 failed / 4644 passed / 74 skipped
tip:  31 failed / 4626 passed / 74 skipped
```

The normalized non-passing node-ID sets are exactly equal at `31/31`, with
SHA-256
`915f43c282f0ce2df34e5f7a5eb39a3dc0e4efa3a8ad59d7994a1c1f4d743a56`
on both sides. Passing nodes changed by exactly `-18`, matching `+6/-24`.
Absolute failure counts are environment observations, not allowlists.

Frontend full results are `96 files / 1072 passed` on both sides. Base and tip
both pass typecheck and production build; the existing chunk-size warning is
unchanged.

### Behavior and mutation evidence

- The source catalog is exactly:
  `['polygon_news', 'finnhub_news', 'ibkr_news', 'ibkr_prices']`.
- All four existing `NewsWriteMode` branches were mutated independently; each
  mutation made the exhaustive classifier node RED.
- Changing the unknown-mode path to a permissive fallback made the
  fail-before-provider/worker/telemetry node RED.
- The prices-worker argv contract parses without `--source`; the active direct
  price primitive itself remains outside this retirement.
- Removed schedule IDs return ordinary `404` before profile writes, DB writes,
  provider readiness, locks, workers, or telemetry.
- Mechanical active-owner search returns no occurrence of
  `price_backfill`, `local_incremental`, `iv_history`, `ScheduleControlMode`,
  `source_control_mode`, `skip_sync`, `sync_flag`, or `_local_refresh` in the
  reviewed scheduler/route/CLI/Settings/API owner set.

### Non-node ledgers

- Resources per locale: Settings `706`, Explore `401`, total `1806`.
- Tool counts remain central/OpenAI/Anthropic `56/57/57`.
- no-PG inventory remains `24`.
- Scanner, two consecutive runs: `36/20/0/20`, global scope `src/**`.
- Scanner SHA-256:
  `c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da`.
- Empty debt manifest SHA-256:
  `d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec`.
- Allowlist SHA-256:
  `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`.

### Read-only production checkpoint observation

The normal desktop/API rooted at the main checkout remained running during
implementation and legitimately advanced live SQLite mtimes after Task 0.
Those live changes are not attributed to this isolated branch. The branch did
not import production paths, invoke scheduler/provider/Gateway/PG work, or
open production data except through the following explicit URI `mode=ro`
checkpoint query.

Immediately before and after that read-only query, all observed sizes and
`mtime_ns` values were identical:

| File | Bytes | `mtime_ns` |
|---|---:|---:|
| `data/profile_state.db` | 43,433,984 | 1785080494457216695 |
| `data/market_data.db` | 3,293,126,656 | 1785080379664695982 |
| `data/options/iv_history/AMD.parquet` | 4,735 | 1772732024459732256 |
| `data/options/iv_history/NVDA.parquet` | 4,701 | 1772732188423298521 |
| `data/options/iv_history/PLTR.parquet` | 4,390 | 1770044155819205290 |
| `data/options/iv_history/PYPL.parquet` | 4,597 | 1772732307750729098 |

Both databases report `integrity_check=ok` and zero FK violations. Relevant
payload remains `24` IV rows, four tickers, non-contiguous IDs with minimum
`1` and maximum `248`; job-run counts remain
`collect.local_incremental=1350`, `collect.price_backfill=2`, and
`collect.iv_history=0`. The one `market_sync_meta` IV row and four Parquet
files remain present. Tranche 1 performs no migration or data deletion.

## Tranche 1 Result

Tranche 1 is GREEN at the named product checkpoint. Its exact focused suites,
full A/B invariants, resources, scanner, active source catalog, and protected
runtime boundaries are frozen for the final two-segment review. Tranche 2 may
now define its RED retirement contracts without modifying Tranche-1-owned
product files.

## Tranche 2 Result

Tranche 2 was implemented RED-first after the named checkpoint. The product
tip before review-ready documentation is `387e4a19`. It removes the old local
IV domain end to end while retaining live option-chain, skew, pure Greeks,
price, fundamentals, financial-cache, and historical telemetry contracts.

Exact backend composition is:

```text
base -> TRANCHE_1_TIP    +6/-24  = -18
TRANCHE_1_TIP -> final  +23/-63  = -40
base -> final           +29/-87  = -58
```

The reviewed plan expected Tranche 2 `+22/-63`. Copied-production proof added
one mutation-sensitive resource-bound node,
`test_logical_database_digest_is_memory_bounded`, making the actual
composition `+23/-63` without changing product behavior.

Frontend composition is exact:

```text
base -> TRANCHE_1_TIP   +3/-3
TRANCHE_1_TIP -> final  +1/-1
base -> final           +4/-4
```

Tranche 2 removes:

- the 24-row SQLite `iv_history` table and four equivalent Parquet stores from
  the active domain contract;
- the history API, frontend DTO/helper, Ticker Detail IV panel, storage-health
  projection, evidence-packet block, DAL/backend readers, and obsolete job
  poller;
- `get_iv_analysis`, `get_iv_history_data`, and `scan_mispricing` from the
  registry and both agent bridges;
- exactly two old-store-coupled scripts,
  `compare_bs_vs_american.py` and `scan_option_mispricing.py`.

The broader `scripts/` retirement is unchanged. Retained
`calculate_greeks`, `get_option_chain`, `get_iv_skew_analysis`,
`src/options_math`, price/fundamental payloads, and all `job_runs` history have
explicit boundary coverage.

Final non-node ledgers are:

```text
resources per locale: Settings 704 / Explore 379 / total 1782
tools: central 53 / OpenAI bridge 54 / Anthropic bridge 54
no-PG smoke: 23
scanner: 36/20/0/20, global scope src/**, twice byte-identical
```

## Migration Tool And Mutation Proof

`scripts/migration/retire_legacy_scheduler_iv.py` implements structured
`preview`, `apply`, and `restore` operations. The final focused migration suite
is `17/17` GREEN. Six independent mutations each turn its owning contract RED:

1. remove preview source-fingerprint validation;
2. permit an incomplete archive;
3. omit archive artifact-hash verification;
4. delete before the resumable phase marker is durable;
5. rewrite on a second apply;
6. restore over a nonempty target.

The original logical database digest accumulated an entire table before
hashing. Against the copied 3.3 GB market database it reached roughly 6 GiB
RSS. The replacement is a deterministic length-framed streaming SHA-256. The
added memory-bound test is RED against the original implementation; the
production-shaped preview completed at approximately 124,912 KiB maximum RSS.

No migration command was run against production paths.

## Copied Production-Shaped Migration Proof

The proof used private copies under
`/tmp/arkscope-legacy-retirement-proof.fZnTtB`; it did not import or write the
live files. Preview derived the targets rather than treating dated counts as
acceptance constants:

```text
IV SQLite rows: 24 across AMD/NVDA/PLTR/PYPL
IV ID bounds: min 1 / max 248, non-contiguous
Parquet rows: 9/8/1/6, exact value multiset match with SQLite
scheduler_state targets: 2
profile setting targets: 1
market_sync_meta IV targets: 1
target job_runs: 1352 (1350 succeeded / 2 failed)
```

The frozen preview SHA-256 is
`1ac1d1ab1216a67cfd31f63f14e49dd72e9cced9743b96ee81941194aded8e30`.
The preserved `job_runs` digest is
`8a120cf820863fd3c325a5ee4d95a9a8177300307fdc01b625e0428d7a297ecc`.

The archive
`backups/legacy_scheduler_iv_retirement_20260726T163737963939Z` has eight
artifacts. Its directory and Parquet directory are mode `0700`; its manifest
is mode `0600`; every artifact hash was verified before deletion. First apply
completed, second and third apply returned `already_applied=true`, and the
second apply left these bytes unchanged:

```text
profile DB  44904de1b54cb3bb1ff2cff2cf58ff815635b8b675b764674bf655054dc72713
market DB   2d24e0de7c398e0421f24a42ebc6f934390900b1a1354f4efe1ea2fe4d82af7e
manifest    e5290995a6a3c2425a732b5708b3c234d83d43957f84197bebaea495bd03517b
```

Restore ran with exact pre-retirement code from the existing detached base
worktree. It restored all target rows/files, SQLite/Parquet equality, both DB
health checks, and all `1352` historical job rows with matching digest. A
temporary attempt to create another detached worktree stopped because the
local git-crypt clean filter had no key; no fallback altered tracked or proof
bytes.

## Canonical Final A/B

Normalized final collections are:

| Gate | Base | Final | Final normalized SHA-256 |
|---|---:|---:|---|
| Backend full | 4749 | 4691 | `ed4b7da05db79204dd847d33d0d9f9bb8f6bbef6c756af48cf218a13f3525acf` |
| Backend focused | 663 | 605 | `4b6e756532ec771c4cef9a43d781b04842621030fc03ed731aaa4bd8dde62121` |
| Frontend full | 96 files / 1072 | 96 files / 1072 | `71e4785f75ace3d65e40a479ce823897ffbcae0bd27ff1855aef1504905e429e` |
| Frontend focused | 9 files / 139 | 9 files / 139 | `9c14c42ec1b211e1e6dd8cb7294adbad20a97bad93271f323f540ad9c240ac21` |

Equivalent-environment backend full observations were base
`4644 passed / 31 failed / 74 skipped` and final
`4592 passed / 27 failed / 72 skipped`. Final introduces zero new non-passing
node IDs; the four disappeared failures are retired old-IV nodes. Absolute
failure counts remain dated environment observations, never allowlists.
Focused final is `560 passed / 26 failed / 19 skipped`; the original
Tranche-1 23-file set rerun at final is
`539 passed / 26 failed / 19 skipped`, proving Tranche 2 did not remove its
surviving contracts.

Additional final gates:

- frontend full and focused GREEN; TypeScript and production build exit 0;
- resource parity/counts, scanner, and foundation suites: `43/43` GREEN;
- scanner hashes remain debt
  `d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec`,
  allowlist
  `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`,
  scopes
  `02e335bebcadfba523d502a7af86a5c184d1ac024230cfec9199dd19b4416c13`,
  scanner
  `c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da`;
- retained option suites: `90 passed / 1 skipped`;
- no-PG: `23/23`, `ok=true`, no PG attempts;
- protected N9, option-math, controller, CSS, desktop, extension, and package
  owners are byte-identical except explicitly authorized files;
- no new `skip`, `only`, or `todo` markers; `git diff --check` is clean.

## Isolated Bilingual Runtime Gate

An isolated API/Vite pair used copied proof data; the normal desktop/API ports
were neither stopped nor reused. Settings Data Sources, Settings Data Storage,
and NVDA Ticker Detail Data were exercised in `zh-Hant` and `en` at 390, 960,
and 1440 px.

Observed contracts:

- exactly four schedule rows: Polygon, Finnhub, IBKR news, and IBKR prices;
- no retired schedule label, control, chip, old-IV heading/content, empty IV
  shell, `/options/NVDA/history` request, or stale IV column;
- price, news, fundamentals, financial cache, and surviving data content stay
  present;
- zero document-level horizontal overflow; narrow schedule tables scroll only
  inside their owning container;
- a 390 px locale switch preserves selected Data/Sync tabs, node identity,
  credential draft, focus, and the four-row schedule without data refetch;
- screenshots were visually inspected and no CSS change was needed.

The only console errors were expected `503` responses for active-universe and
coverage data absent from the copied profile. Isolated ports closed after the
gate; the user's existing app ports remained responsive.

## Production Boundary

The short `mode=ro` checkpoints recorded above prove that the inspection
queries themselves did not alter live bytes. During the much longer copied
migration proof, the already-running desktop legitimately advanced the live
`profile_state.db`; therefore this packet does **not** claim whole-period live
profile byte identity. `market_data.db` and all four Parquet files remained
unchanged in that observation. No branch command invoked production migration
preview/apply/restore, scheduler work, provider work, Gateway work, or PG.

Production archive and deletion remain blocked on independent implementation
GREEN, merge, explicitly stopped writers, a fresh production preview, and a
second explicit user approval.

## Review-Ready Result

Both tranches, the migration tool, copied-data restore proof, final A/B,
technical gates, bilingual runtime matrix, and current-state documentation are
complete. The branch is intentionally **IMPLEMENTED - INDEPENDENT REVIEW
PENDING**. It is not merged or pushed, and nothing in this packet authorizes a
production migration.

After the documentation-only closeout, a final targeted rerun produced:

```text
migration suite                         17 passed
frontend focused                        9 files / 139 passed
resources/scanner/foundation            3 files / 43 passed
visible-literal scanner, two runs       36/20/0/20 both runs
tool-count contract                     3 passed; central count 53
no-PG inventory                         23
```

The Tranche-2 backend data-dependent subset produced its known fixture-bound
shape, `370 passed / 26 failed / 19 skipped`; every failure is in the existing
no-local-market/news-fixture family. This command is a subset of, and must not
be confused with, the canonical 605-node focused A/B above.
