# Lifecycle Honesty Repair Final Fix Admission

This self-hashed packet admits product/test authority
`c043bc0e7ca0642e383841dfcc537c5bdb4242e2` entirely offline. It replaces,
and does not extend, the packet previously sealed at `fbfd6738`.

## Authority

- Product base: `11e7a5d4f6856062a5ac00a8d90ed97b5c2e56cb`
- Product/test authority: `c043bc0e7ca0642e383841dfcc537c5bdb4242e2`
- Topology: 20 linear commits and zero merge commits from product base to
  product/test authority
- Policy: `trusted-lifecycle-automation-v3`
- Shared SEC rule version: `3`
- Deadline-only SEC rule version: `4`
- Browser app: transient loopback Vite fixture server at `127.0.0.1:4206`,
  stopped immediately after capture
- Browser API: fixture responses intercepted at the page boundary; no
  production backend was started

`commands.txt` is directly replayable with Bash from any repository directory.
It backgrounds and cleans up the loopback server and runs npm commands through
`--prefix`, so no manual directory or process step is required.

## Repaired Seams

The admitted product and test range owns all final-review findings:

- latest unambiguous market receipt selection while retaining old receipts;
- valid conflict authority and Needs-attention precedence;
- durable terminal decision recovery across assessment, acceptance, proposal,
  and approval boundaries;
- current observation/evidence/provenance projection for terminal artifacts;
- current manual-input evidence binding for automation runs, using the same
  digest producer at write and read boundaries;
- immutable initial and bounded latest-attempt execution revision identity;
- deadline-only final market acquisition and schedule capping;
- newest-only frontend queue and selected-case detail response commits;
- stale-policy daily transition revalidation;
- distinct IBKR missing-receipt and ambiguity source states; and
- typed frontend fixtures plus replayable packet commands.

Terminal recovery stores a bounded decision and provenance in the existing
`query_context_json`. It creates no table, column, index, migration, or startup
DDL. Later sequential worker ticks replay only finalization and do not reacquire
providers. Durable assessment, proposal, and transition records are reused
idempotently; profile mutation remains zero before transition application.

## Mutation Admission

All 47 mutations were applied and restored independently. Every mutation was
killed by its exact expected owner node, and every touched product file was
restored byte-identically. Within each explicitly executed owner-only mutation
command, no expected owner was missing and no additional test failed; the
`unexpected_owner_drift` field is not a broad suite-drift scan. The added mutations
cover latest-market selection, conflict routing and projection, terminal
recovery reservation, run and transition current-artifact binding,
latest-attempt replay eligibility, both deadline-only seams, newest-response
frontend commits, stale approval timing, IBKR source-state semantics,
manual-input digest currentness, terminal-transition binding, fail-closed
market timestamps, explicit-conflict persistence, assessment reuse, and
pre-execution-key row compatibility.

## Scratch Authority

Two temporary-SQLite captures were byte-identical. They prove:

- legacy failed-run replay remains bounded by execution revision;
- decision provenance remains equal across r0 and r1;
- `r0 blocked -> r1 retry fails -> r1 no replay` uses one row;
- initial `execution_revision` remains r0 while
  `latest_attempt_execution_revision` becomes r1;
- valid pre-deadline and final citations cross the producer/kernel boundary;
- a forged citation rolls back with zero evidence, fact, or blocker rows;
- source deadline `2026-04-01` remains distinct from completed check
  `2026-08-27`;
- final projection remains truthful dated History; and
- calibrated fail-closed wrappers exercise all nine exact transition preview,
  approval, apply, reverse, and acknowledgement targets, including the
  import-time `ticker_identity_service` preview alias; all instrumented target
  counts are zero during the scratch authority capture.

The packet's provider, production-database, App-restart, merge, and push zeros
are declared execution boundaries, not runtime-instrumented measurements. The
transition and acknowledgement counts above are separately instrumented and
calibrated.

## Fresh Gates

Focused node collection was performed twice and produced 213 byte-identical
node IDs.

```text
focused A: 229 passed in 12.49s
focused B: 229 passed in 14.53s
full backend A: 4539 passed, 12 skipped, 3 warnings in 248.06s
full backend B: 4539 passed, 12 skipped, 3 warnings in 252.03s
frontend: 106 files / 1246 passed
typecheck: passed
visible i18n literal scanner: passed; debtSignatureCount=0
production build: passed; 2193 modules
```

The three backend warnings are existing EDGAR v6 deprecation notices. The
frontend build retains the existing non-blocking warning for a chunk larger
than 500 kB after minification.

## Schema And Protected Authority

Fresh in-memory schema captures from a local product-base archive and the
product/test authority prove:

```text
owned sqlite_master diff = empty
PRAGMA table_info diff = empty
index SQL diff = empty
new mutable disposition columns = 0
startup DDL changes = 0
security_lifecycle_schema.py byte diff = empty
ticker_identity_transition.py byte diff = empty
ticker_identity_transition.py execution_revision references = 0
AUTOMATION_POLICY_VERSION = trusted-lifecycle-automation-v3
SEC shared _RULE_VERSION = 3
deadline-only rule version = 4
```

`security_lifecycle_decision_policy.py` differs from the product base only as
expected for latest-current-market selection; its policy version is unchanged
and mutation/test authority owns the new behavior. No existing database file
was read. Schema captures instantiate clean in-memory databases only.

## Browser Matrix

The rebuilt offline fixture matrix covers six scenarios in English and
Traditional Chinese at `1440x900` and `390x844`. Across all 24 entries:

```text
external requests = 0
write requests = 0
render acknowledgements = 0
console errors = 0
page errors = 0
visible-control overlaps = 0
clipped visible text = 0
```

Every screenshot is nonblank. The final-unconfirmed row and drawer contain the
exact bilingual dated copy and exclude confirmed-complete language.

## Limitations And Hard Stops

- No provider call or production scheduler replay was performed.
- Persistence recovery was exercised with local temporary SQLite and later
  sequential worker ticks; concurrent multi-worker fault injection is outside
  this packet.
- The browser matrix uses fixture interception; out-of-order queue and detail
  sequencing are owned by focused Vitest interface tests.
- Broader legal-language extraction remains intentionally precision-first.
- No production schema migration or row backfill is needed: schema authority is
  unchanged, and mutation M37 / `test_pre_execution_key_succeeded_row_remains_idempotent`
  proves that a pre-execution-key succeeded row remains query-compatible and is
  reused idempotently.
- Five independently removable backend fail-closed conditions and two
  load-bearing frontend race guards were unowned before this closeout; the seven
  named closeout mutations now own them. Other incidental frontend guards are
  not claimed as mutation-owned and remain accepted coverage debt.
- The provider, production-database, App-restart, merge, and push fields are
  declared hard-stop compliance, not general process instrumentation.
- No production database operation, production App restart, merge, or push was
  performed.

`SHA256SUMS` covers every packet payload except itself. Its own digest is
reported separately in the final fix report.
