# Legacy Scheduler Sources And IV Domain Retirement Evidence

> **Status: IMPLEMENTATION IN PROGRESS - NO PRODUCTION WRITE**
>
> Task 0 is complete. Product implementation, production archive/apply,
> provider calls, Gateway calls, PG calls, merge, and push have not occurred.

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
