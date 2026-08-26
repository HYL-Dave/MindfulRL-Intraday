# Lifecycle Automated Disposition Admission Evidence

This packet admits the automated lifecycle disposition implementation entirely
offline. It does not authorize or perform a provider call, production database
read or write, migration, backup, restore, App restart, merge, or push.

## Authority

- Base: `ff795454429d63121f21a53ab14e6f911ce06ab2`
- Product/test authority before this packet:
  `f63a044c3495dcd95db2e996345082eb492baf7d`
- Topology: 12 linear commits and zero merge commits from base to product/test
  authority
- Policy: `trusted-lifecycle-automation-v3`
- Browser app: isolated-worktree Vite frontend at `127.0.0.1:4201`, stopped
  after the matrix
- Browser API: fixture responses intercepted at the page boundary; no
  production backend was started

`offline-authority.json` is generated entirely from temporary SQLite databases.
It records zero provider calls and zero production database operations.

## Mutation Admission

All 12 specified mutations were applied separately, killed by named owner tests,
and restored before the next mutation. `mutation-ledger.json` records the exact
owner sets and restore hashes.

M7 initially survived because the existing no-source-deadline test omitted an
effective-date fact, while the effective-date test also supplied a source
deadline. The new isolated owner proves an effective date never substitutes for
a cited source termination deadline. Under M7 it fails at the production
behavior assertion; under the restored product it passes.

Two old fixtures also needed contract evolution before admission:

- the worker's synthetic IBKR evidence now uses the adapter-v2 canonical payload,
  including `contract_status` and a fresh live `market_data` snapshot; and
- the historical HAPN canary remains unchanged and now correctly proves
  `waiting_market_confirmation`, because that old capture has a real IBKR
  contract snapshot but no fresh quote.

These changes preserve the historical canary rather than inventing live market
data after the fact.

## Automated Gates

Focused backend admission ran twice with identical counts:

```text
127 passed in 9.01s
127 passed in 9.10s
```

Complete backend admission:

```text
4475 passed, 12 skipped, 3 warnings in 238.65s
```

The three warnings are existing `edgar` v6 deprecation notices. Complete
frontend admission:

```text
106 files / 1239 passed
typecheck: passed
visible i18n literal scanner: passed
production build: passed (2193 modules)
```

The existing large-chunk build warning remains non-blocking and unchanged in
kind.

## Schema Authority

`capture_profile_schema.py` creates clean in-memory lifecycle and ticker-identity
profile schemas from the base archive and head worktree. The structured
comparison proves:

```text
sqlite_master owned object diff = empty
PRAGMA table_info diff = empty
columns named disposition / queue_bucket / reason_code = 0
startup DDL changes = 0
```

No existing database file is read by this comparison. Disposition remains a pure
read projection rather than a second mutable workflow state.

## Version-Cutover Authority

`capture_offline_authority.py` ran twice with byte-identical output. It proves:

- one v2 draft and its v2 run remain stored exactly after replay;
- v3 creates a distinct succeeded run and draft;
- the old v2 assessment projects as stale without rewriting its stored row; and
- a v2 automation-approved transition presented to the v3 apply boundary is
  blocked as `preview_changed`, moved to `needs_review`, and leaves all
  profile-owned rows byte-equivalent by canonical SHA-256.

## Browser Matrix

The matrix covers five scenarios in English and Traditional Chinese at
`1440x900` and `390x844`:

- attention with conflicting source facts;
- not-yet-confirmed post-date monitoring with a frozen IBKR quote;
- confirmed monitoring before a future effective date;
- settled history with stale/reopened history retained; and
- the source-missing data-integrity view.

Across the fixtures this covers all four dispositions and all five source-family
states: `confirmed`, `present`, `missing`, `unavailable`, and `conflict`.
All 20 entries produced one nonblank screenshot and reported:

```text
external requests: 0
writes: 0
render acknowledgements: 0
console errors: 0
page errors: 0
visible-control overlaps: 0
clipped visible text: 0
```

Representative desktop/mobile and English/Traditional-Chinese images were
visually inspected. A first pass exposed cross-case fixture prose inherited from
an older HAPN packet; the fixture was corrected to scenario-specific evidence
and the complete 20-entry matrix was rerun.

## Limitations And Live Boundary

This packet does not exercise the current live IBKR market-data shape or a live
v3 scheduler replay. It proves only offline contracts, fixtures, scratch
authority, and browser behavior.

The following remain separate user decisions and are not authorized here:

1. review and fast-forward merge of this implementation branch;
2. bounded read-only live inventory and SEC/IBKR v3 canary;
3. App cutover/restart after canary review; and
4. push.

Before requesting an App restart, capture the separately authorized read-only
live inventory required by the implementation plan. Do not infer current draft
or transition counts from review-time data.

`SHA256SUMS` covers every packet payload except itself. Its own digest is
reported separately after generation and verification.
