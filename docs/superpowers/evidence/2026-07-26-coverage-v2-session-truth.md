# Coverage v2 Session-Truth Evidence

> **Status: IMPLEMENTATION REVIEW GREEN - P3 TEST CLOSURE COMPLETE - INTEGRATION PENDING**
>
> This is a review ledger, not a LIVE declaration. The branch is not merged,
> no production database was written, and no provider, Gateway, repair, or
> scheduler action was run.

## Review Boundary

- Plan-review clearance:
  `f6cbcb6e2343c14cd185e0f7e766ce98e77cc8db`.
- Docs-only clearance pointer:
  `40ef09e2bc0bfeee504089f0f5de8a485b296e9e`.
- Task checkpoints:
  - calendar solution and fixtures:
    `9ceaee2aeb2f57a6bbd47ccfaa9d09d8064d04da`;
  - calendar adapter and sole health composer:
    `ce02ba35f814337ed42f78d56b095a69b4f07198`;
  - pure slot classifier:
    `d56439bd5fad5ef345faedb80246e83175d8f248`;
  - read-only observation owner:
    `696a316ce0488d88ea9bf556634eb178ca5f932a`;
  - atomic V2 service/route replacement:
    `2fb052384596275698973b3cf576276b3a2aa3cb`;
  - `COVERAGE_V2_BACKEND_TIP`:
    `742b2e1c47e6835654ad15b70104319df10dbbef`;
  - initial frontend/presentation product tip:
    `0b7197356e537b9d6114f566608611ea77312876`;
  - post-evidence independent-review repair product tip:
    `db410e096747ffd7f834c9231394fe2d041ec53a`;
  - final optional-diagnostic isolation product tip:
    `cb33a1937b22a593a7da69d096d17dccb2d89733`.
  - implementation-review follow-up and P3 test closure: verified in the
    current isolated worktree and intentionally left uncommitted pending
    integration authorization.
- Isolated branch/worktree: `codex/coverage-v2-ground-truth` at
  `/home/hyl/.config/superpowers/worktrees/ArkScope/coverage-v2-ground-truth`.
- No merge or push was performed. The main repository and its `8430` Vite
  process were not modified or stopped.

## Plan-Review Accounting Corrections

The two required plan-review corrections were independently rederived before
product work:

1. `test_route_registered` has the same ID on both sides and therefore evolves
   in place. `test_trading_day_coverage.py` is exactly `+14/-18`, not
   `+15/-19`.
2. The old 25-leaf and new 44-leaf Coverage resource trees share 12 paths.
   The exact resource comm is `+32/-13`, net `+19`, not `+19/-0`.

The old route file also contained stable alias, ordering, storage, error, and
route properties. The implementation replaces the file as a suite while
preserving those properties in named V2 successors; it does not claim that
every removed node was maximum-relative.

## Canonical Test Accounting

### Backend

- Virgin base collection: `4713`; normalized SHA-256
  `a3b91ea6eed808afb7aa7dc860a9f5f8e30de9dd770a9f06245c35d0f04a5d6a`.
- Virgin final-tip collection: `4746`; normalized SHA-256
  `7f1b2515af24dff3f9f00cb8304c4c2a4d63af59ec5567d53ff44c07e62040c4`.
- Exact comm: `+72/-39`; added-list SHA-256
  `5429aecdaaf8e4d80196af2da487eb0f3466dd54ab2c7e26810a94648a147495`;
  removed-list SHA-256
  `cfd051861d946f165490e3c1fc807c491d1d479be86c63da34d54da4557d4157`.
- Additions by owner: scheduler `7`, boundaries `5`, calendar `10`,
  classifier `18`, dependencies `6`, observations `11`, route `15`.
- Focused final tip: `8 files / 227`; `227 passed`; normalized SHA-256
  `4fe4960dbb0b3c1eba08b507740a785c5bce7d68624d5aa4080f4832cac07b16`.
- Explicit semantic set: `18 passed`.
- `test_route_registered` exists unchanged on both sides and is not a removal
  or addition.

The exact 72 additions are:

```text
tests/test_data_scheduler.py::test_coverage_derived_price_backfill_is_deliberate_noop
tests/test_data_scheduler.py::test_legacy_unproven_gap_manual_continuation_is_rejected_without_worker
tests/test_data_scheduler.py::test_legacy_unproven_gap_scheduler_continuation_is_rejected_without_worker
tests/test_data_scheduler.py::test_price_backfill_does_not_resolve_scope_for_deliberate_noop
tests/test_data_scheduler.py::test_price_backfill_ignores_gateway_lock_but_keeps_source_lock
tests/test_data_scheduler.py::test_status_snapshot_preserves_durable_state_without_planner_metadata
tests/test_data_scheduler.py::test_unknown_tickers_and_provider_errors_never_reach_price_executor
tests/test_market_coverage_boundaries.py::test_backend_v2_contract_and_source_contain_no_retired_coverage_fields
tests/test_market_coverage_boundaries.py::test_coverage_enum_consumers_use_exact_exhaustive_matching
tests/test_market_coverage_boundaries.py::test_market_coverage_package_exports_no_write_or_repair_operation
tests/test_market_coverage_boundaries.py::test_market_coverage_package_has_no_provider_gateway_or_pg_runtime_dependency
tests/test_market_coverage_boundaries.py::test_scheduler_has_no_planner_missing_feed_or_unknown_exclusion_path
tests/test_market_coverage_calendar.py::test_adapter_failure_makes_health_unavailable
tests/test_market_coverage_calendar.py::test_calendar_adapter_failure_is_typed_unavailable
tests/test_market_coverage_calendar.py::test_calendar_adapter_returns_closed_without_named_holiday_claim
tests/test_market_coverage_calendar.py::test_calendar_adapter_returns_typed_early_close
tests/test_market_coverage_calendar.py::test_calendar_adapter_returns_typed_regular_session
tests/test_market_coverage_calendar.py::test_calendar_health_is_ok_for_reviewed_dates_and_healthy_horizon
tests/test_market_coverage_calendar.py::test_fixture_review_membership_is_independent_of_forward_horizon
tests/test_market_coverage_calendar.py::test_forward_horizon_uses_calendar_month_boundaries
tests/test_market_coverage_calendar.py::test_low_horizon_is_degraded_without_erasing_reviewed_history
tests/test_market_coverage_calendar.py::test_unreviewed_date_is_degraded_and_unclassifiable
tests/test_market_coverage_classifier.py::test_alias_collision_fills_one_slot_only
tests/test_market_coverage_classifier.py::test_completed_day_count_equations_hold
tests/test_market_coverage_classifier.py::test_early_close_buffer_changes_only_at_1329_1330
tests/test_market_coverage_classifier.py::test_early_close_grid_uses_exact_half_open_slot_starts
tests/test_market_coverage_classifier.py::test_extended_hours_rows_never_fill_rth_slots
tests/test_market_coverage_classifier.py::test_in_window_off_grid_row_is_counted
tests/test_market_coverage_classifier.py::test_off_grid_row_does_not_fill_nearest_slot
tests/test_market_coverage_classifier.py::test_partial_plus_unknown_stays_partial_and_preserves_unknowns
tests/test_market_coverage_classifier.py::test_precedence_all_tickers_complete_is_complete
tests/test_market_coverage_classifier.py::test_precedence_calendar_unavailable_is_unknown
tests/test_market_coverage_classifier.py::test_precedence_complete_observed_cohort_with_unknown_is_indeterminate
tests/test_market_coverage_classifier.py::test_precedence_completed_all_zero_is_unknown
tests/test_market_coverage_classifier.py::test_precedence_observed_partial_ticker_is_partial
tests/test_market_coverage_classifier.py::test_precedence_pre_close_buffer_is_in_progress
tests/test_market_coverage_classifier.py::test_precedence_reviewed_closed_day_is_non_trading
tests/test_market_coverage_classifier.py::test_regular_session_grid_uses_exact_half_open_slot_starts
tests/test_market_coverage_classifier.py::test_single_complete_outlier_does_not_hide_truncation
tests/test_market_coverage_classifier.py::test_uniform_truncation_is_partial
tests/test_market_coverage_dependencies.py::test_exchange_calendar_imports_on_supported_python
tests/test_market_coverage_dependencies.py::test_fixture_release_horizon_covers_twelve_calendar_months
tests/test_market_coverage_dependencies.py::test_reviewed_python_dependency_solution_is_exact
tests/test_market_coverage_dependencies.py::test_xnys_matches_every_reviewed_early_close_fixture
tests/test_market_coverage_dependencies.py::test_xnys_matches_extraordinary_closure_fixture
tests/test_market_coverage_dependencies.py::test_xnys_matches_reviewed_full_session_fixture
tests/test_market_coverage_observations.py::test_missing_market_db_is_typed_unavailable
tests/test_market_coverage_observations.py::test_missing_prices_schema_is_typed_unavailable
tests/test_market_coverage_observations.py::test_optional_provider_diagnostic_corruption_is_quarantined_and_source_preserved
tests/test_market_coverage_observations.py::test_query_only_rejects_accidental_writes
tests/test_market_coverage_observations.py::test_readable_empty_prices_table_is_ok
tests/test_market_coverage_observations.py::test_reader_assigns_rows_by_utc_session_window_not_date_prefix
tests/test_market_coverage_observations.py::test_reader_excludes_extended_hours_rows
tests/test_market_coverage_observations.py::test_reader_is_read_only_and_preserves_database_bytes
tests/test_market_coverage_observations.py::test_reader_maps_aliases_to_canonical_tickers
tests/test_market_coverage_observations.py::test_reader_retains_in_window_off_grid_rows
tests/test_market_coverage_observations.py::test_unreadable_market_db_is_typed_unavailable
tests/test_trading_day_coverage.py::test_calendar_unavailable_returns_unknown_days
tests/test_trading_day_coverage.py::test_early_close_session_uses_derived_fourteen_slot_grid
tests/test_trading_day_coverage.py::test_empty_active_universe_returns_honest_unknown_coverage
tests/test_trading_day_coverage.py::test_low_fixture_horizon_degrades_health_without_erasing_reviewed_days
tests/test_trading_day_coverage.py::test_missing_market_db_is_unavailable_not_empty
tests/test_trading_day_coverage.py::test_provider_errors_remain_separate_diagnostics
tests/test_trading_day_coverage.py::test_readable_empty_market_db_is_ok_with_unknown_days
tests/test_trading_day_coverage.py::test_regular_session_uses_exact_rth_slots_despite_extended_rows
tests/test_trading_day_coverage.py::test_route_coverage_path_is_pure_read_without_provider_scheduler_or_pg
tests/test_trading_day_coverage.py::test_route_preserves_sanitized_active_universe_503
tests/test_trading_day_coverage.py::test_route_rejects_unreviewed_interval_with_typed_422
tests/test_trading_day_coverage.py::test_route_wires_active_universe_and_v2_service
tests/test_trading_day_coverage.py::test_service_dedupes_aliases_and_orders_requested_window
tests/test_trading_day_coverage.py::test_service_emits_exact_v2_contract_without_retired_fields
tests/test_trading_day_coverage.py::test_unreviewed_date_is_unknown_while_reviewed_dates_classify
```

The exact 39 removals are:

```text
tests/test_data_scheduler.py::test_p0c1_price_backfill_runs_prices_worker_with_planned_scope
tests/test_data_scheduler.py::test_price_backfill_empty_scope_fails_loud
tests/test_data_scheduler.py::test_price_backfill_serializes_behind_ibkr_lock
tests/test_data_scheduler.py::test_price_backfill_uses_planner_scope_no_pg_no_mirror
tests/test_data_scheduler.py::test_v13_attended_scheduler_skips_pending_continuation
tests/test_data_scheduler.py::test_v13_gate1_coverage_window_matches_planner_max_days
tests/test_data_scheduler.py::test_v13_gate2_provider_errors_exclude_unresolvable
tests/test_data_scheduler.py::test_v13_no_gaps_is_noop_success
tests/test_data_scheduler.py::test_v13_partial_when_deferred_and_writes_continuation
tests/test_data_scheduler.py::test_v13a_manual_continue_carries_remainder_when_over_budget
tests/test_data_scheduler.py::test_v13a_manual_continue_consumes_saved_deferred_not_fresh_plan
tests/test_data_scheduler.py::test_v14_status_snapshot_exposes_durable_state_and_gap_planned
tests/test_scheduler_planner.py::test_deterministic_selection_order
tests/test_scheduler_planner.py::test_excludes_known_unresolvable_tickers
tests/test_scheduler_planner.py::test_lookback_reaches_oldest_selected_gap
tests/test_scheduler_planner.py::test_max_days_caps_lookback
tests/test_scheduler_planner.py::test_max_tickers_caps_and_defers_rest
tests/test_scheduler_planner.py::test_no_gaps_is_empty_plan
tests/test_scheduler_planner.py::test_non_trading_and_in_progress_days_are_not_gaps
tests/test_scheduler_planner.py::test_pure_no_side_effects
tests/test_scheduler_planner.py::test_selects_tickers_with_missing_complete_days
tests/test_trading_day_coverage.py::test_complete_trading_day_full_missing_counts
tests/test_trading_day_coverage.py::test_complete_when_most_present_well_covered_despite_one_laggard
tests/test_trading_day_coverage.py::test_coverage_status_non_trading_and_in_progress
tests/test_trading_day_coverage.py::test_days_newest_first_and_window
tests/test_trading_day_coverage.py::test_in_progress_today_flagged_incomplete
tests/test_trading_day_coverage.py::test_large_missing_fraction_is_partial_not_complete
tests/test_trading_day_coverage.py::test_outlier_full_does_not_mask_thin_universe
tests/test_trading_day_coverage.py::test_partial_day_lists_thin_ticker
tests/test_trading_day_coverage.py::test_provider_errors_surface_lc
tests/test_trading_day_coverage.py::test_read_only_absent_db_is_honest
tests/test_trading_day_coverage.py::test_route_complete_empty_is_not_unavailable
tests/test_trading_day_coverage.py::test_route_unavailable_returns_sanitized_503
tests/test_trading_day_coverage.py::test_route_wires_universe_and_db
tests/test_trading_day_coverage.py::test_single_gap_in_large_universe_stays_complete
tests/test_trading_day_coverage.py::test_thin_threshold_uses_normalized_interval
tests/test_trading_day_coverage.py::test_uniformly_thin_day_not_read_as_complete
tests/test_trading_day_coverage.py::test_universe_count_dedupes_aliases
tests/test_trading_day_coverage.py::test_weekend_and_holiday_marked_non_trading
```

Initial-product full-suite A/B in the same isolated environment retained the
exact same 72-node baseline failure/error set: base `4567 passed / 65 failed /
7 errors / 74 skipped`, initial tip `4598 passed / 65 failed / 7 errors / 74
skipped`. The failure-node SHA-256 is
`cd32e5ddffadd58cca836be15ea864857cdf0463534a1dda6477cf78aad4152b`
on both sides; new failures `0`, disappeared failures `0`. These existing
failures are not an allowlist. The post-evidence repair adds two focused nodes;
the final replay therefore uses a new matched pair rather than rewriting this
dated A/B checkpoint.

Final-product virgin A/B explicitly changed into each extracted archive before
running pytest. Clearance base produced `4572 passed / 60 failed / 7 errors /
74 skipped`; final product tip `cb33a193` produced `4605 passed / 60 failed /
7 errors / 74 skipped`. The 67 exact non-passing node IDs hash to
`0a2aea1065f9837b7bf55b6eef9971e1474c339b1a0e292bb466b58ccf86da0b` on
both sides; new non-passing IDs `0`, disappeared IDs `0`. The `+33` passing
delta equals the final backend net node delta exactly. These environment-bound
non-passing nodes remain evidence, not an allowlist.

All absolute pass/fail/error totals above are dated environment observations,
not acceptance constants. The load-bearing A/B invariant is equality of the
normalized non-passing node-ID set between equally configured base and tip
runs, plus the exact passing-node delta.

### Frontend

- Virgin base: `95 files / 1063`, all passed; normalized node-list SHA-256
  `a93c02bc28d1924f23f7895338d723e968dcb389a494ff0e0f993e4c092019d4`.
- Virgin tip: `96 files / 1072`, all passed; normalized node-list SHA-256
  `22e2d1bd576ded36fd215ac51c41e1b7811f3e4f580a72b21dfffa88ebdb4ed3`.
- Exact comm: `+11/-2`; all 1,061 surviving IDs remain byte-identical.
- Focused tip: `8 files / 118`, all passed; normalized node-list SHA-256
  `5a4153f289a82b8b7e724e31e2f32ea3ea358b3a6e382bc490e4e03c988202df`.
- The only removed IDs are:
  - `marketDataDisplay.test.ts > distinguishes weekend vs holiday for non_trading`;
  - `marketDataDisplay.test.ts > renders backend coverage_status (UI does not re-derive completeness)`.
- The eleven added IDs are the two mounted Settings contracts, three closed
  V2 contract/exhaustiveness contracts, and six presentation contracts:

```text
SettingsPostPgExitStorage.test.ts > keeps calendar degradation separate from reviewed-day coverage
SettingsPostPgExitStorage.test.ts > keeps unmatched rows and provider issues separate from coverage state
coverageV2Contract.test.ts > accepts the V2 DTO and rejects retired field fixtures
coverageV2Contract.test.ts > exports the exact closed Coverage v2 enum catalogs
coverageV2Contract.test.ts > keeps every frontend coverage enum consumer exhaustive and exact
marketDataDisplay.test.ts > keeps partial and unknown ticker facts independent
marketDataDisplay.test.ts > maps calendar and observation health without parsing diagnostics
marketDataDisplay.test.ts > maps every Coverage v2 day reason in both locales
marketDataDisplay.test.ts > maps every Coverage v2 day status in both locales and reserves positive tone for complete
marketDataDisplay.test.ts > maps non-trading closure reasons without backend prose
marketDataDisplay.test.ts > renders unmatched RTH rows as a separate data-quality warning
```

## Calendar, Classifier, And Read-Only Boundaries

- The installed solution is pinned and mechanically checked:
  `exchange_calendars==4.13.2`, pandas `2.3.1`, NumPy `1.26.4`, with the
  reviewed transitive solution in `requirements.txt`. The exact-solution test
  passes. A repository-wide `python -m pip check` is not clean in this
  environment because of a pre-existing unrelated constraint:

  ```text
  spacy 3.6.1 has requirement typer<0.10.0,>=0.3.0, but you have typer 0.21.1.
  ```

  That global conflict is recorded rather than misrepresented as a passing
  Coverage dependency gate.
- Official fixtures cover a normal session, every reviewed early close, and
  the extraordinary 2025-01-09 closure. Package output matches every fixture.
- The fixture release horizon covers twelve calendar months; runtime health
  reports the reviewed range and forward horizon separately.
- Seven ordered classifier paths have named tests:
  calendar-unavailable `unknown`, `non_trading`, pre-buffer `in_progress`,
  all-zero `unknown`, observed `partial`, `indeterminate_tickers`, and
  `complete`.
- Early-close completion changes at exactly `13:29 -> 13:30 ET` for an actual
  13:00 close plus the existing 30-minute buffer.
- In-window off-grid rows are retained by the reader, counted by the
  classifier, and never fill a nearest slot.
- SQLite reads use URI `mode=ro` plus query-only enforcement. Missing,
  unreadable, schema-invalid, and readable-empty stores remain distinct.
- Optional provider-diagnostic schema, row, and UTF-8 corruption is
  quarantined without changing valid price-observation health; valid
  diagnostic source text, including surrounding whitespace, is retained
  byte-for-byte at the text boundary.
- Static boundaries prove the package has no provider, Gateway, PG, write, or
  repair runtime dependency.

## V2 Contract And Planner Quarantine

- The raw backend and frontend DTOs contain none of the retired fields:
  `max_observed_bar_count`, `full`, `well_covered`, `covered`, `missing`,
  `missing_tickers`, `session_complete`, `thin`, or `complete_like`.
- `market_scope=us_listed_equity_proxy` and `coverage_session=rth` remain two
  separate closed enum axes.
- Unknown tickers and provider issues cannot enter planner candidates or
  exclusions. The scheduler has no V2 `missing_tickers` feed.
- Legacy, malformed, or unreadable continuation state becomes
  `legacy_unproven_gap`; it is never resumed as provider work. A row with no
  status, error, continuation, or result is neutral no-result history.
- The public schedule route exposes no operator action for the deliberate
  read-only no-op path. Its PUT and Run endpoints return typed HTTP 409 before
  provider or write gates, while internal terminal telemetry remains available
  for historical compatibility.
- Four disposable mutations independently proved the boundary nodes are live:
  filtering off-grid reader rows failed the named reader test; sending an
  unknown ticker to the worker failed scheduler isolation; putting a
  provider-error ticker in exclusions failed the same isolation contract; and
  sending a legacy continuation to the worker failed both continuation
  contracts. Clean restoration returned the four-node set to `4 passed`.
- `python -m src.smoke.pg_unreachable_e2e` completed all 24 checks with
  `"ok": true` and `"pg_attempts": []`.

## Resources, Scanner, And Static Gates

- Per locale, Settings resources are `694 -> 714`; total resources are
  `1794 -> 1814`; the Coverage subtree is `25 -> 44`. The implementation
  review follow-up adds one Data Sources leaf after the original exact
  `+32/-13` Coverage migration. Locale key parity drift is zero and empty
  leaves are zero.
- Dynamic-key contracts are `3/3`.
- Scanner ran twice with exact `36/20/0/20`, debt `0`, scope `src/**`.
- Unchanged scanner hashes:
  - migrated scopes
    `02e335bebcadfba523d502a7af86a5c184d1ac024230cfec9199dd19b4416c13`;
  - allowlist
    `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`;
  - debt manifest
    `d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec`;
  - scanner
    `c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da`.
- TypeScript typecheck and Vite build exit zero; only the existing chunk-size
  advisory remains.
- The clearance-to-initial-product range changed exactly 35 authorized paths
  (`20 M / 13 A / 2 D`); unexpected paths were zero. The evidence commit then
  changed six documentation paths, and the review repair modified eight
  already-authorized product/test paths without introducing another path.
  The clearance-to-final-product review packet therefore spans 40 paths
  (`24 M / 14 A / 2 D`), all inside the reviewed maps. The initial
  non-authorized manifest remained byte-identical with recorded SHA-256 prefix
  `6a14b350`.
- Schema/migrations, protected backend families, extensions/Electron, all CSS,
  package manifests/locks, and the existing formatter implementations are
  byte-identical. Only `requirements.txt` changes dependency metadata.
- `git diff --check` exits zero.

## Isolated HTTP Truth Matrix

A route-only FastAPI app exercised the real
`GET /market-data/trading-days` endpoint against fresh or copied SQLite
fixtures. All twelve cases passed; every existing DB digest was identical
before and after, all retired V1 keys were absent from raw JSON, and a socket
spy recorded zero network/provider/Gateway/PG calls.

1. regular 26-slot complete session with extended-hours rows;
2. early-close 14-slot session at both 13:29 and 13:30 ET;
3. uniform truncation -> `partial`;
4. one complete outlier among truncated tickers -> `partial`;
5. complete observed cohort plus unknown ticker -> `indeterminate_tickers`;
6. partial plus unknown ticker -> `partial`, with the unknown cohort separate;
7. one omitted exact slot plus one in-window off-grid row -> 25 observed,
   `partial`, `unmatched_rth_row_count=1`;
8. readable empty DB -> observation health `ok`, day `unknown/no_observations`;
9. missing/unreadable/schema-invalid DB -> their three distinct unavailable
   reason codes;
10. five-month fixture horizon plus a reviewed complete day -> calendar
    `degraded`, day still `complete`;
11. unreviewed date -> `unknown/date_unreviewed`; and
12. provider issue alongside otherwise complete coverage -> complete day plus
    one separate provider diagnostic.

## Browser Matrix And Locale Purity

- A real Chromium Settings surface rendered the worst credible composition in
  both locales at `390`, `760`, `960`, and `1440` CSS pixels: long English
  copy, low calendar horizon, provider issue, partial+unknown, ten expanded
  partial tickers, and unmatched-row warning.
- All eight cases had zero document overflow and zero element-level clipped
  text. At 390, the 640px diagnostic table owns one intentional internal
  horizontal scroll region; the document does not scroll horizontally.
- `indeterminate_tickers` renders with warning color `rgb(184, 134, 11)`, not
  success color `rgb(63, 185, 80)`, in both locales.
- Normal mode contains neither planted unknown ticker IDs nor planted provider
  detail. Developer Mode presents both only in the Coverage section's single
  `Developer diagnostics` disclosure.
- No repair/backfill control exists. Dates, ticker IDs, slot counts, and other
  source values remain unchanged across locale switches.
- Keyboard `Space` collapses and `Enter` expands the real button disclosure.
- An in-place locale switch preserved the `30`-day selection, expansion,
  focused button, and planted identity markers on the lookback, toggle, and
  detail nodes. The request delta was exactly one
  `PUT /profile/settings/ui-locale`; Coverage GET count remained `3 -> 3`.
- Browser console errors: zero.

## Production Read-Only Witness

Observed on 2026-07-26 through a route-only server containing only the real
Coverage GET handler, followed by the real Settings read surface:

- active universe: `150`, digest
  `46f6f080c6699da1cd97adc5ff71eefee063ca96797a150c565c75aa1ec61ce9`;
- calendar health `ok`, reviewed through `2027-12-31`, forward horizon `17`;
- observation health `ok`, provider issues `0`;
- `2026-07-24` is `complete`, expected slots `26`, complete tickers `150`,
  partial `0`, unknown `0`, unmatched RTH rows `0`;
- every other returned trading day from 2026-07-15 through 2026-07-23 is also
  `150/150 complete`; weekend rows remain `non_trading`;
- the browser network list contains only GET requests; no control was invoked.

Before and after the HTTP and Settings reads:

- market DB size `3,293,126,656`, mtime ns
  `1785022441890893935`, integrity `ok`, FK violations `0`;
- profile DB size `43,155,456`, mtime ns `1785022441967894044`, integrity
  `ok`, FK violations `0`;
- the 39,491 relevant price rows retained SHA-256
  `c2b957bffbfa99113f83da24f86ce3c9eb9765b69b8237c915faadeb75291498`;
- the active-universe count and digest remained identical; and
- every size, mtime, integrity, FK, row-count, and digest comparison was true.

These are dated production observations, not acceptance constants and not
fixtures copied into tests.

## Reviewed Deviations And Rejections

- Calendar, observation, alias, scheduler, and frontend reviews produced
  bounded hardening commits before their task checkpoints. Each retained the
  final node/resource totals and was rerun at the task boundary.
- Task 6 review found four unsafe legacy seams and two misleading historical
  IDs. The fixes are recorded in the plan's Task 6 review resolution and
  account for the initial implementation checkpoint's backend `+70/-39`
  composition.
- Task 7 review required stronger compile-time enum-consumer proof,
  locale coverage for `non_trading`, normal-mode unknown-count privacy,
  exhaustive scope/session presentation, and a real accessible disclosure
  button. All were accepted and verified.
- A proposed frontend reclassification/runtime schema layer was rejected:
  same-version local Pydantic output is the DTO authority, and a frontend
  classifier would violate the locked presenter-only boundary.
- No CSS change was needed, so the CSS-deviation protocol was not opened.
- Post-evidence independent review found five bounded defects: empty valid
  universes raised instead of yielding unknown coverage; malformed optional
  provider diagnostics poisoned observation health; diagnostic source text
  was trimmed; translated React keys remounted two nodes; and evidence path
  wording did not name its comparison range. The RED tests failed on those
  exact behaviors. Product tip `db410e09` fixed the original five; same-review
  follow-up `cb33a193` additionally proved and quarantined invalid UTF-8 in the
  optional diagnostic table. Together they retain backend `+2/-0`, an in-place
  mounted frontend assertion, no resource/CSS/schema/dependency change, and
  the same eight already-authorized modified paths.

## Independent Implementation Review Follow-Up

The implementation review returned substantive GREEN and found two bounded
presentation/control defects plus one evidence-accounting issue. The approved
repair is RED-first and does not alter coverage classification:

- `price_backfill` now reports `control_mode=read_only`; stale persisted
  enablement cannot make it due, service and HTTP mutation paths reject it,
  and Settings retains historical run telemetry while rendering no schedule,
  interval, or Run control. Retired sources use `control_mode=retired`.
- A production-shaped pre-V2 row with attempt/update metadata and null status,
  error, continuation, and result remains neutral. Its first deliberate V2
  no-op succeeds immediately. Non-empty legacy continuation/result shapes
  remain `legacy_unproven_gap` and worker-free.
- The bilingual description now states that Coverage v2 retains historical
  run records only and does not schedule, backfill, or write price data. One
  bilingual `Read-only` / `唯讀` label increases Settings resources
  `713 -> 714`, total resources `1813 -> 1814`, and Data Sources
  `162 -> 163` per locale.
- Backend node accounting evolves from `+72/-39` to `+74/-40`: one new
  blank-history contract and one honest rename of the obsolete all-sources Run
  contract. Collection becomes `4747`; focused becomes `228`. Frontend becomes
  `+12/-3` through one renamed mounted node, so collection remains `96/1072`
  and focused remains `8/118`.
- `requirements.txt` now distinguishes the shared NumPy/pandas stack from the
  Coverage-specific calendar dependencies without changing any version.
- Post-repair focused gates are backend `228/228` and frontend `8 files / 118`
  green. Backend full collection is `4747`; this worktree environment observed
  `4612 passed / 61 failed / 74 skipped`, with failures in the already-known
  mounted-data/root-jsdom families. Frontend full is `96 files / 1072`, all
  green. Typecheck/build pass; scanner is deterministic at `36/20/0/20` twice.
- The no-PG smoke passes all 24 checks with `ok=true` and `pg_attempts=[]`.
  `python -m pip check` reproduces only the documented existing
  `spacy 3.6.1` versus `typer 0.21.1` conflict.
- A production `mode=ro` query reproduced the exact blank `price_backfill`
  shape. `profile_state.db` remained `(43180032, 1785050934183146442)` for
  `(size, mtime_ns)` before and after; integrity is `ok`, FK violations are
  zero. This section carries no merge, production-write, or LIVE authority.

## Follow-Up Re-Review Test Closure

Independent follow-up review returned GREEN on the three product/evidence
findings and identified one P3 test-only gap. The product predicate correctly
requires all four outcome fields to be `None`, but its existing tests did not
make each field independently mutation-sensitive.

- Before adding tests, removing `continuation` from the predicate left all four
  relevant existing tests green (`4 passed, 95 deselected`), reproducing the
  review finding rather than assuming it.
- Two named tests now cover `last_status`-only, `continuation`-only,
  `last_error`-only, and `last_result`-only legacy states. Each state must be
  nonblank and project the fixed `legacy_unproven_gap` result without exposing
  its raw continuation, error, or result.
- Four independent mutation probes removed one decisive key at a time. Each
  probe made its owning test fail at the blank-state assertion. The product
  source was restored after every probe; its SHA-256 remains
  `0a783f31ef67392158f991b4584f33a77f184bb817711bb6b994545eebf18039`.
- Backend accounting therefore moves from the reviewed product-fix checkpoint
  `+74/-40` to final `+76/-40`. Full collection is `4749`, with normalized
  node-list SHA-256
  `e7dc826f33c202789f8ad5f43787d1eedd8f288cc55aa4996d0a100761a21b20`.
  The eight-file focused suite is `230/230`, with normalized node-list SHA-256
  `e551d209062867d44bd88289f6e6c8b06d2f0822200ce1320aa8f966863f8330`.
- Frontend, resources, scanner, dependency, product runtime, and production
  evidence are unchanged by this test-only closure. The stable row title stays
  `Price Gap Backfill`; the `Read-only` chip, description, and absent controls
  carry current behavior.

## Cleanup And Review State

- Isolated ports `8467` and `8477` refuse connections after verification.
- The user's main `8430` Vite process remains reachable.
- Browser test pages were closed; no Coverage worktree Vite, uvicorn, Gateway,
  or test process remains.
- The product worktree was clean at each committed product tip before the
  documentation update. The implementation-review follow-up and P3 closure
  remain uncommitted pending integration authorization.
- Independent implementation review is GREEN. Integration remains unperformed
  and requires explicit authorization; this packet does not authorize LIVE
  status, provider work, planner repair, or any production write.
