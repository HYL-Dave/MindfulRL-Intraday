# SA Feed Store Truth Evidence

> **Status: IMPLEMENTATION REVIEW GREEN - TEST-CLOSURE RE-REVIEW PENDING - NO PRODUCTION WRITE**

## 1. Authority And Isolation

- `PLAN_REVIEW_CLEARANCE_COMMIT`:
  `a364268c8dcbcc80a38842669a6545198fae8e3d`.
- Branch: `codex/sa-feed-store-truth`.
- Worktree: `/tmp/arkscope-sa-feed-store-truth`.
- The initial direct checkout stopped at the known linked-worktree
  `git-crypt` smudge boundary. The clean retry used `--no-checkout`, copied
  only `.git/git-crypt/keys/default` with mode `0600`, and populated `HEAD`
  with `git read-tree -mu HEAD`. The protected evaluation document is
  `47,608` bytes.
- Existing root and web `node_modules` are linked into the worktree. No
  dependency install or lockfile change occurred.
- No production database, `config/.env`, browser profile, token, user-owned
  data, provider, Gateway, PG service, or external endpoint was read or
  changed. The checkout contains only the repository's tracked default config
  files. A missing worktree-only `data/` directory was created after one
  tool-count fixture could not auto-detect a project root; no production data
  was copied into it. The no-PG smoke then created its own 143,360-byte empty
  profile store, which passed integrity, was identified as an ignored
  worktree artifact, and was deleted before product RED tests.
- The untracked root document
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` was not copied or staged.

## 2. Task 0 Baseline

All normalized collections reproduce the reviewed authority exactly:

| Gate | Result |
|---|---|
| Backend full | `4691`; `ed4b7da05db79204dd847d33d0d9f9bb8f6bbef6c756af48cf218a13f3525acf` |
| Backend focused | `77`; `34a30e6d54c108fadfe4e0425d863c9a6fbfaf1b7f10a93ee82f53d380d3eb2a` |
| Backend focused run | `77 passed` |
| Frontend full | `96` files / `1072` nodes; `71e4785f75ace3d65e40a479ce823897ffbcae0bd27ff1855aef1504905e429e` |
| Frontend focused | `2` files / `25` nodes; `086cce183d540193a966a61148f6e7a9e6c2177a8ebecd49bb71c2c1cfc6d892` |
| Frontend focused run | `25 passed` |
| Scanner run 1 | `36 / 20 / 0 / 20`, scope `src/**` |
| Scanner run 2 | `36 / 20 / 0 / 20`, scope `src/**` |
| Tool gates | three named tests passed; inventories `53/54/54` |
| no-PG | `23/23`, `ok=true`, `pg_attempts=[]` |

The first tool-count attempt produced one fixture-construction error because a
virgin worktree lacked `data/`; tracked default `config/` was already present.
After creating the empty worktree-only data directory, the exact three-node
gate passed. This is recorded as environment setup, not a baseline product
failure.

## 3. Protected Byte Baseline

Tracked-file blob IDs:

```text
src/sa_capture_store.py                         f4eacd5a5746ec96e5f945ecff33cb4c0df1448c
src/api/routes/seeking_alpha.py                 aa75e20a877980257a80f4887a193ab07ec97e63
src/sa/extension_run_protocol.py                fd367b101841f85f2507a71fce932abb31c7eb5b
src/sa/market_news_recovery.py                  1a69bbdf19b3047d0f8d239e171c9ffbae31fedd
src/service/jobs.py                             f58dd01df4240eaf8f0dd3897036b40d180156a1
src/sa_native_host.py                           88712d75076c4d13d832c623c6597c1fab0a1116
apps/arkscope-web/src/styles.css                c2649cbf2a874521721d40f7e1d5d4b392a2d1ba
apps/arkscope-web/src/ui/primitives.css         d92000dc4687eed79b7c7f3319ba9e2a973b0fab
apps/arkscope-web/src/shell/shell.css           80cbacf2e6d986add0d358b447049e961c5f4dce
apps/arkscope-web/src/ui/tokens.ts              ef4f46565f7d11aa9b94bc2bdca9084bde131f2e
apps/arkscope-web/src/ui/tokens.json            144262e61a023e56103a5c3aa1a9bf6eea404436
```

Sorted `git ls-tree -r` family hashes:

```text
extensions/sa_alpha_picks                       40eac710c4e85f6a5773b7ebda7e6eb67b86b0946dbb21a6fc7111ad4851f585
apps/arkscope-web/scripts/i18n                  9757396abe7bcac8ddfc0981f032a3333007e0e425242408d06bb7d18897b33d
sql                                              bcf30e4d419c351082566bfbeb94dcd2f4366bae57dc22d5b2d528e397e9c40f
```

Scanner artifact SHA-256 values remain:

```text
migrated-scopes.json             02e335bebcadfba523d502a7af86a5c184d1ac024230cfec9199dd19b4416c13
visible-literal-allowlist.json   3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212
visible-literal-debt.json        d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec
visible-literal-scanner.mjs      c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da
```

## 4. Grounding Correction

Task 0 found that the reviewed protected list contained four stale nonexistent
paths: `src/sa_native_manifest.py` and root-level `primitives.css`, `shell.css`,
and `tokens.css`. Before product work, the plan was corrected to the current
owners `src/ui/primitives.css`, `src/shell/shell.css`, and
`src/ui/tokens.{ts,json}`; the nonexistent manifest path was removed. This is a
docs-only gate repair with no product, node, resource, or scope change.

## 5. Task 1 - No-Create History Reader

The five named RED nodes failed because `read_job_activity_if_exists` did not
exist, rather than because a fixture touched the real profile path. The
implementation opens only an existing SQLite target using URI `mode=ro`,
checks `sqlite_master`, parameterizes the activity names, and returns only
`none | present | unknown`.

The resulting `tests/test_job_runs.py` collection is `68/68`. Mutation probes
proved both load-bearing properties:

- replacing the primitive with `JobRunsLocalStore(db_path)` created evidence
  and turned the missing-profile no-create node red; and
- adding a succeeded-status or timestamp-style restriction turned the
  relevant-row/history-contract coverage red.

Task commit: `ec5e406e feat: add no-create SA activity history reader`.

## 6. Task 2 - Ordered Missing-Store Classification

One immutable owner now lists the seven reviewed SA activity job names. Seven
separately named tests each seed `running`, `succeeded`, and `failed`, proving
that activity status does not alter prior-activity truth. A separate
completeness node derives the current union from extension operations,
SA-enabled service jobs, and the audited repair owner.

The checkpoint reached `69` job-history plus `25` feed nodes (`94/94`). The
following independent mutations turned reviewed nodes red before restoration:

- removing any one immutable owner name;
- changing a named fixture to an unrelated activity;
- filtering history to succeeded rows;
- constructing the mutating local store in the feed path; and
- moving history/filesystem inspection ahead of `requires_local_sa`.

Task commit: `e5d919fc feat: classify absent SA feed storage honestly`.

## 7. Task 3 - Read-Only Store Capability And Query

The store classifier now owns one direct SQLite URI read-only connection for
open, required-capability inspection, and the existing feed query. The global
SA connector remains byte-identical. Required ordinary tables, both FTS
tables, and required columns are capability-checked; additive schema remains
valid. Directory, broken-symlink, malformed/open-race, missing-table,
missing-column, and post-validation query failures each receive their reviewed
typed reason without path or raw SQLite prose.

The added FastAPI transport node proves every unavailable reason remains an
HTTP `200` typed response. The evolved route node separately preserves
feature-disabled `503` and populated behavior. Job-history plus feed tests are
`107/107`.

Independent mutation probes turned the intended nodes red for:

- routing the missing/race path through the global connector;
- omitting either FTS capability for a request without `q`;
- treating additive columns as incompatible;
- projecting a post-validation failure as valid empty; and
- passing `str(exc)` into the feed response.

Task commit:
`ba6f2852 feat: derive SA feed truth from read-only store capability`.

## 8. Task 4 - Closed Frontend Presentation

Initial mounted RED produced exactly five failing assertions: the exhaustive
reason classifier, neutral first-run copy, unavailable statistics/rows, and
the two resource-count assertions. The final implementation uses a closed
`SAFeedEmptyReason` union plus an exhaustive `switch` and
`unreachableSAFeedReason(value: never)`; no `Set.has()` classifier or CSS was
added.

Adversarial unavailable fixtures carry positive totals, facets, rows, and
pagination potential. Mounted tests prove that all four claim surfaces are
suppressed by `feed.available`, while valid-empty and populated behavior stay
visible. Bilingual copy and Data Sources recovery ownership are exact.

Focused frontend verification is `27/27`. Resource inventories are Explore
`380`, Settings `704`, total `1783`; the Explore News subtree is `44`. Both
locales have exact key parity and no empty leaf.

The four rendering-gate removals independently turned the mounted visibility
node red. Adding a synthetic `future_reason` to the TypeScript union turned
typecheck red at the `never` boundary. All mutations were restored.

Task commit: `841d7241 fix: present unavailable SA feed states truthfully`.

## 9. Canonical Final Accounting

Normalized final collections and hashes:

| Gate | Base | Review-ready tip | Comm |
|---|---|---|---|
| Backend full | `4691`; `ed4b7da05db79204dd847d33d0d9f9bb8f6bbef6c756af48cf218a13f3525acf` | `4722`; `fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` | `+31/-0` |
| Backend focused | `77`; `34a30e6d54c108fadfe4e0425d863c9a6fbfaf1b7f10a93ee82f53d380d3eb2a` | `108`; `b0ec2b6ff11187df092011fbbd576b6f004bc9bf077ce8ee1145ec7b970bb5b0` | `+31/-0` |
| Frontend full | `96/1072`; `71e4785f75ace3d65e40a479ce823897ffbcae0bd27ff1855aef1504905e429e` | `96/1074`; `e322e7a51e83eedb8b3c7b1fd99e6033f496031968c1a2cb3f59974bfd994f47` | `+2/-0` |
| Frontend focused | `25`; `086cce183d540193a966a61148f6e7a9e6c2177a8ebecd49bb71c2c1cfc6d892` | `27`; `ac6bb12b93f3cb27ff84d534d3f3b88153b6bc935a3c5bd449395c751f95b286` | `+2/-0` |

The exact backend additions are:

```text
tests/test_job_runs.py::test_read_job_activity_if_exists_distinguishes_relevant_and_unrelated_rows
tests/test_job_runs.py::test_read_job_activity_if_exists_missing_profile_is_none_and_no_create
tests/test_job_runs.py::test_read_job_activity_if_exists_missing_table_is_none_and_no_mutation
tests/test_job_runs.py::test_read_job_activity_if_exists_unreadable_or_malformed_is_unknown
tests/test_job_runs.py::test_sa_store_activity_job_names_cover_all_current_authorities
tests/test_job_runs.py::test_sa_store_history_contract_has_no_pruning_or_time_cutoff
tests/test_sa_feed.py::test_backend_unavailable_precedes_store_and_history_checks
tests/test_sa_feed.py::test_broken_symlink_sa_store_is_unreadable
tests/test_sa_feed.py::test_directory_sa_store_is_unreadable
tests/test_sa_feed.py::test_extra_feed_schema_remains_compatible
tests/test_sa_feed.py::test_malformed_sa_store_is_unreadable
tests/test_sa_feed.py::test_missing_required_feed_column_is_schema_incompatible
tests/test_sa_feed.py::test_missing_required_feed_table_is_schema_incompatible[sa_articles]
tests/test_sa_feed.py::test_missing_required_feed_table_is_schema_incompatible[sa_articles_fts]
tests/test_sa_feed.py::test_missing_required_feed_table_is_schema_incompatible[sa_market_news]
tests/test_sa_feed.py::test_missing_required_feed_table_is_schema_incompatible[sa_market_news_fts]
tests/test_sa_feed.py::test_missing_required_feed_table_is_schema_incompatible[sa_market_news_tickers]
tests/test_sa_feed.py::test_missing_store_history_extract_sa_comment_signals_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_alpha_picks_refresh_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_extension_manual_fetch_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_market_news_incident_recovery_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_market_news_refresh_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_market_news_repair_is_missing
tests/test_sa_feed.py::test_missing_store_history_sa_market_news_retry_recorded_is_missing
tests/test_sa_feed.py::test_missing_store_with_empty_profile_is_not_created_without_mutation
tests/test_sa_feed.py::test_missing_store_with_unreadable_history_fails_closed_as_missing
tests/test_sa_feed.py::test_missing_store_without_profile_is_not_created_and_creates_nothing
tests/test_sa_feed.py::test_post_validation_query_failure_is_typed_sanitized_and_preserves_request
tests/test_sa_feed.py::test_route_returns_typed_200_for_every_unavailable_store_reason
tests/test_sa_feed.py::test_sa_store_open_failure_is_unreadable_and_sanitized
tests/test_sa_feed.py::test_unexpected_internal_failure_is_typed_sanitized_and_preserves_request
```

The exact frontend additions are:

```text
src/News.test.tsx<TAB>News localization > hides all feed claims and controls for every unavailable SA reason
src/News.test.tsx<TAB>News localization > renders typed SA store availability copy in both locales
```

No backend or frontend node was removed.

## 10. Runtime And Full Verification

The isolated no-create matrix returned `ok=true` for all reviewed states:

```text
no SA + no profile              -> store_not_created, no path created
no SA + empty profile           -> store_not_created, profile preserved
seven activities x three states -> store_missing
malformed/directory profile     -> store_missing, fail closed
directory/broken/malformed SA   -> store_unreadable
missing table/column            -> store_schema_incompatible
compatible additive empty       -> available=true/no_items_in_window
compatible populated            -> one normal projected result
injected post-validation error  -> store_query_failed
```

The reusable SQLite fixture retained exact size, `mtime_ns`, SHA-256,
`schema_version`, table names, integrity, FK result, and relevant row counts.
No absent parent or database appeared.

Focused backend is `108 passed`; focused frontend is `27 passed`; frontend
full is `96 files / 1074 passed`. Typecheck and build exit `0` (the existing
chunk-size warning remains informational). Scanner runs twice at exact
`36/20/0/20`, scope `src/**`.

The sandboxed backend full run stalled symmetrically at the existing FastAPI
`TestClient` portal boundary. A credential-free, network-denied run outside
that sandbox reproduced both sides:

| Tree | Passed | Failed | Skipped |
|---|---:|---:|---:|
| base | 4592 | 27 | 72 |
| tip | 4623 | 27 | 72 |

The normalized non-passing node-ID set is byte-identical: `27/27`, SHA-256
`236251b45d101896f8de6759dd4e30d4a7624dbc821387ca0e1d3bfde0db6670`,
new `0`, gone `0`. Absolute failure totals are environment observations, not
an allowlist. Before the test-only review closure, the exact passing delta was
`+30`; after the closure the exact passing delta and collection delta are both
`+31`.

Tool counts are `53/54/54`; no-PG inventory and runtime smoke are `23/23`,
`ok=true`, `pg_attempts=[]`. The protected paths and directory families in
the plan are byte-identical to clearance. Scanner artifact hashes remain those
in section 3.

## 11. Isolated Browser Evidence

Google Chrome `150.0.7871.128` ran against an isolated Vite instance. A
browser-local closed fetch fixture intercepted every non-Vite request; no
sidecar, production DB, extension, provider, scheduler, or repair action was
reachable.

The matrix covered both `zh-Hant` and `en`, four states
(`not_created/degraded/empty/populated`), and `390/960/1440` CSS pixels: `24`
cases and `25` screenshots including the locale-purity witness. Unavailable
fixtures deliberately contained `total=9`, nonempty facets, two rows, and
pagination potential. Every unavailable case rendered zero statistics, rows,
valid-empty copy, and Load More controls. The neutral/degraded copy and Data
Sources action were exact. Valid empty rendered its zero statistic and empty
copy without a recovery action; populated rendered two rows unchanged. Every
case had zero document/body horizontal overflow.

An in-place initialized-i18next switch from `zh-Hant` to `en` preserved:

```text
SA feed calls       3 -> 3
mode                sa -> sa
query               retained query -> retained query
ticker input        nvda -> nvda
mode DOM identity   preserved
first-row identity  preserved
focus               preserved
document lang       en
```

Browser runtime exceptions, console errors, and HTTP failures are all empty.
The matrix summary SHA-256 is
`3c8f6c881f713386e725f5a5749c5024474f115eed6e96c582a51f1a04f685a3`.

## 12. Review Boundary

Product tip before this docs commit is `841d7241`. No production smoke,
restart, capture, database read/write, merge, or push has occurred. The
separate Alpha Picks availability-alignment follow-up remains open. Independent
implementation review has returned GREEN; the one-node test-only closure in
section 13 is the sole re-review gate before integration.

## 13. Independent Review And Test-Only Closure

Independent implementation review of `664635a5` returned GREEN with zero
required product changes. It independently reproduced all four collection
hashes, exact `+30/-0` and `+2/-0` comm, the nine-row response contract,
seven activity-name mutations, resource/scanner/tool/no-PG gates, and protected
boundaries.

The sole advisory identified a reachable but unpinned outer `get_sa_feed`
catch-all. The inner post-validation query seam already had safe-fallback
coverage; a failure while reading the backend `_sa_db` property reached the
outer fallback, whose old implementation had owned both fixed `days=30` and
raw `error=str(e)` behavior.

One test-only node now drives that exact outer path:

```text
tests/test_sa_feed.py::test_unexpected_internal_failure_is_typed_sanitized_and_preserves_request
```

It requires normalized `days=3650`, `query="private query"`, typed
`store_query_failed`, and no raw marker or `error` field. Two independent RED
probes were observed before restoration:

1. replacing normalized days with `30` failed at `30 != 3650`;
2. adding `result["error"] = str(e)` failed the no-diagnostic assertion.

The restored reviewed product source is byte-identical to `841d7241`; only
`tests/test_sa_feed.py` changes. Focused backend is `108/108`. Final backend
collection accounting is `4691 -> 4722`, exact `+31/-0`, with hashes recorded
in section 9. A credential-free, network-denied full run in the same empty-data
fixture shape as the reviewed tip is `4623 passed / 27 failed / 72 skipped`;
its normalized non-passing set remains exact `27/27`, SHA-256
`236251b45d101896f8de6759dd4e30d4a7624dbc821387ca0e1d3bfde0db6670`,
new `0`, gone `0`. Frontend and every resource/scanner/tool/no-PG/browser
boundary remain unchanged. This one-node test/evidence delta is the only
follow-up re-review scope before integration.
