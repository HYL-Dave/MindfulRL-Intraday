# Lifecycle Honesty Repair Offline Admission

This self-hashed packet admits the repaired lifecycle seams for independent
review entirely offline. It grants no live authority.

## Authority

- Product base: `11e7a5d4f6856062a5ac00a8d90ed97b5c2e56cb`
- Product/test authority: `b05ea4a2abe2fc2e1fbebd845de624804f46f3d1`
- Topology: 12 linear commits and zero merge commits from product base to
  product/test authority
- Policy: `trusted-lifecycle-automation-v3`
- Shared SEC rule version: `3`
- Deadline-only SEC rule version: `4`
- Browser app: transient isolated-worktree Vite fixture server on
  `127.0.0.1:4206`, stopped after the matrix
- Browser API: fixture responses intercepted at the page boundary; no
  production backend was started

`offline-authority.json` was generated twice with byte-identical output from
temporary SQLite files. It contains the exact zero-live authority block from
the reviewed Task 6 contract.

## Mutation Admission

All 16 specified mutations were applied independently. Every mutation was
killed by its exact expected owner node, unexpected owner drift is empty, and
every touched product file was restored byte-identically before the next
mutation. `mutation-ledger.json` records expected and actual failure counts,
owner node IDs, output tails, and before/after SHA-256 values.

## Scratch Authority

The scratch capture proves:

- one legacy failed row has no execution revision;
- one r1 replay occurs under unchanged semantic policy v3;
- the predecessor's canonical bytes are identical before and after replay;
- the successful replacement prevents r2 fan-out;
- r0 and r1 decision provenance are equal;
- valid pre-deadline and final citations cross the producer/kernel boundary;
- a forged citation rolls back with zero evidence, fact, or blocker rows;
- source deadline `2026-04-01` remains distinct from completed check
  `2026-08-27`;
- final projection is `not_confirmed_yet + history +
  disposition_as_of=2026-08-27`; and
- transition preview, approval, apply, reverse, and acknowledgement calls are
  all zero.

## Fresh Gates

Focused backend collection was performed twice. The persisted node-only files
contain 192 identical node IDs; only pytest's nondeterministic collection-time
summary line was removed before the byte comparison.

```text
focused A: 192 passed in 11.32s
focused B: 192 passed in 11.48s
full backend A: 4510 passed, 12 skipped, 3 warnings in 242.55s
full backend B: 4510 passed, 12 skipped, 3 warnings in 242.93s
frontend: 106 files / 1241 passed
typecheck: passed
visible i18n literal scanner: passed; debtSignatureCount=0
production build: passed; 2193 modules
```

The three backend warnings are the existing `edgar` v6 deprecation notices.
The existing frontend large-chunk build warning remains non-blocking.

## Schema And Protected Authority

Fresh in-memory schema captures from the local product-base archive and repaired
head prove:

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

No existing database file was read. The capture creates clean in-memory schema
instances only.

## Browser Matrix

The offline fixture matrix covers six scenarios, including final-unconfirmed
History, in English and Traditional Chinese at `1440x900` and `390x844`.
Across 24 entries it records:

```text
external requests = 0
write requests = 0
render acknowledgements = 0
console errors = 0
page errors = 0
visible-control overlaps = 0
clipped visible text = 0
```

Every entry has a nonblank screenshot. The final-unconfirmed row and drawer
contain the exact reviewed bilingual dated copy and do not contain the locale's
confirmed-complete label. Representative desktop/mobile images in both locales
were visually inspected. Two packet-runner development attempts failed before
authority completion; the final run deleted the partial screenshot directory
and rebuilt all 24 entries from scratch.

## Limitations And Authorization Boundary

- no provider call or production scheduler replay was performed;
- operator-triggered replay remains unimplemented;
- broader legal-language extraction remains precision-first and intentionally
  incomplete;
- no production migration is needed because schema authority is unchanged; and
- App restart, merge, push, and the Priority Map GREEN entry remain separate
  authorization events.

The original untyped legacy `CASES` fixture minor that omits
`disposition_as_of` remains deferred and is not part of Task 6.

`SHA256SUMS` covers every packet payload except itself. Its own digest is
reported separately in the Task 6 report and closeout.
