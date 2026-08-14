# SA Health Truth and Typed Diagnostics Evidence

> **Status:** TASKS 0-5 COMPLETE; TASK 6 BATCH EXECUTION ACTIVE;
> IMPLEMENTATION NOT YET COMPLETE; NOT MERGED; NOT PUSHED
>
> **Date:** 2026-08-14
>
> **Plan authority:** `73e5e175`
>
> **Product grounding base:** `9c9021afe6e9fe4d27a971f0841d38d213354a94`
>
> **Task 0 packet:** `/tmp/sa-health-diagnostics-task0-73e5e175`
>
> **Task 5 packet:** `/tmp/sa-health-diagnostics-task5-258e387d`

## 1. Process Boundary

Independent plan re-review returned GREEN at `73e5e175`. The user authorized
Tasks 0-6 to run continuously while preserving every task's RED-first
evidence, product/tests and evidence/docs commits, exact staged identities,
and all stop conditions. Task 6 remains the combined implementation-review
gate. Task 7, merge, push, live SA/provider traffic, extension installation,
production writes, schedule/repair actions, and destructive operations remain
unauthorized.

Task 0 changed no product or test byte. The implementation worktree was clean
before and after grounding; `9c9021af..73e5e175` contains only the reviewed
design, plan, and priority-map documents.

## 2. Re-collected Baselines

| Stream | Re-collected identity | Runtime |
|---|---|---|
| Backend full | `4359 / c100ee5de4ad42c490e6048e4b7cf22540e417f579987c676796663608d17afd` | collect-only, zero test bodies |
| Backend SA focused, 11 files | `275 / e6ae1a5a38629f558beff0586a98b5e0ea4f28c6a3a516c1302119b874ce3336` | `275 passed` |
| Frontend full | `101 files / 1172 / d40a30d5e50690f79b644e0b25122da02441eb0cf54ab02793d02269419e23cb` | list-only |
| Frontend SA owners, 4 files | `74 / 7ec82dccd499299ec0a1ebd796740bea7186a804920b89deb9ad898a968bbd01` | `74 passed` |
| Settings regression, 15 files | `246 / c1be07c3d9c7335c4d4172af59cae1234c45c5f6429032f33bbff120280070aa` | `246 passed` |

The Settings runtime authority is the structured Vitest JSON report with
`246 passed / 0 failed / success=true`. Two earlier attempts returned only the
Vitest startup line and no terminal count despite shell exit 0; both are
retained and explicitly rejected rather than treated as passing evidence.

The inherited native baseline remains `4347 passed / 12 skipped / 0 failed`.
Task 0 did not rerun it because the reviewed authority range has zero product
drift; Task 6 owns the final native admission.

## 3. Predicted Identities

All additions were absent from the collected base. Each frontend removal
existed exactly once. Rebuilding from the plan's literal rows produced:

| Stage | Full | Focused |
|---|---|---|
| Backend Task 1 | `4370 / 554dc03c...` | `286 / 9c68d4a2...` |
| Backend Task 2 | `4378 / 03bceb26...` | `294 / 4826f566...` |
| Backend Task 3 | `4386 / c3969f49...` | `302 / 73b47eef...` |
| Backend Task 4 | `4394 / b0285ee3...` | `310 / f9e7c89c...` |
| Frontend Task 5 | `101 files / 1177 / 9530dcd9...` | SA owners `79 / 0d6568f1...`; Settings `249 / a3a5e481...` |

The complete backend addition stream is `35 / 7da0e54b...`; frontend streams
are `+8 / 418a58b6...` and `-3 / f10630ba...`. Full values and complete node
rows are in the packet; abbreviated values here are labels only.

## 4. Ownership And Protected Boundary

- Owned product/test paths: `26` (`20` existing plus `6` planned new paths).
- Byte-protected paths: `17`.
- Protected aggregate: `1c5b539a05e51eef3f52e0cad9efa02063db077cfb7e190f20ccdc8b0580e0ae`.
- Product drift from `9c9021af`: zero paths.
- Pinned reporter, Vitest normalizer, and native wrapper hashes match the plan.

The user-supplied normal-state screenshot remains a dated regression witness:
`927 x 417`, SHA-256
`3e698db56ffe4765c2859e8429b6833deff504a7a08a321f0be51113abb232b7`.
It proves the healthy presentation that must be preserved, not that current
failure diagnostics are sufficient.

## 5. Packet

Task 0 packet `/tmp/sa-health-diagnostics-task0-73e5e175` contains 64 payloads.
`sha256sum -c` passed for every listed payload. `SHA256SUMS` SHA-256 is:

```text
3a787bfee056f60e997ef888071e68575e378fc314a0b4709c166e8afee42b74
```

## 6. Task 1 - API Validation And Durable Projection

Task 1 followed the exact 11-node RED-first ledger. The pre-product run failed
all 11 nodes only at the absent request field, validator, durable projection,
or bounded reader. Full collection was the predicted
`4370/554dc03c78ff70f362fd24df4a4f562510b47597916d42c56251dcb869b85b83`;
the exact 11-row addition stream was
`e257bc36995ed72bfbce39c0886238c04f0de106ea0bd7664f02c45e3b8c99b5`.

Product commit `ed8228cd` adds one closed validator, three server-owned
canonical event documents, recorded/rejected/absent durable projections, and
one read-only allowlisted 20-row reader. Legacy requests retain the exact old
hash document without an `absent` member. Explicit malformed diagnostics retain
the valid terminal outcome and hash only the fixed rejection marker; no raw
rejected value reaches `job_runs`.

The first product-side run found that the existing deduplication scan reused
the local variable name prepared for a new payload. With an older row already
present, a new event could therefore inherit the scanned row's payload. The
implementation separated `existing_payload` from the new payload before
admission. The legacy/new-event regression then passed and directly owns this
failure mode.

Final gates:

- new API/store nodes: `11 passed`;
- existing `tests/test_job_runs.py`: `68 passed`;
- SA focused: `286 passed`, identity
  `9c68d4a2fa4cce3c37b1cfa5365a92dd4de2caf835eb74d9e03b3a9413d70a7c`;
- full collect-only: `4370/554dc03c...`, with RED and GREEN node streams
  byte-identical; and
- all 17 protected paths remained at their Task 0 blobs.

Packet `/tmp/sa-health-diagnostics-task1-45586c03` contains 19 payloads.
Its `SHA256SUMS` SHA-256 is
`19d24d33ce5aa5cbe74f92139d58fe3da395f644feb05765c4dd7d75bb0d27d9`.
No provider, live extension, production profile, network endpoint, or
destructive operation was used.

## 7. Task 2 - Native Persistence Failure Envelope

Task 2 added the exact eight-node native-host ledger. Initial RED was
`7 failed / 1 passed`: seven nodes exposed raw-error, false-result, or bridge
gaps, while the success-shape owner already passed and therefore guarded the
requirement that successful saves gain no failure fields. Full collection was
the predicted `4378/03bceb26c4691823d21d903fb4fb064df4734d69b2c7c8e6ce0ff55509265b18`;
the eight-row addition stream was
`a9df387a09c5e60808d854755fc57473e3e751944bc1e7c3263bb1ceafc820ca`.

Product commit `f53dcfee` gives exactly five active save handlers one closed
failure projector. SQLite busy/locked is retryable `database_busy`;
constraint/integrity/corruption is terminal `database_integrity_failed`; all
other exceptions and false save results are retryable `database_write_failed`.
Responses use fixed text and never interpolate exception values. Python 3.10
does not export all SQLite symbolic result-code names, so the implementation
uses SQLite's stable base codes plus exception-family fallback.

Grounding against the DAL found another failure shape after the first helper
version: several methods return an error dictionary when local storage is not
available. The shared-handler owner was strengthened and failed on this live
path before the final implementation routed returned-error, exception, and
false-result forms through the same projection. Native telemetry now forwards
exactly the optional `extension_diagnostics` sibling and still rejects caller
status/hash/extra fields.

Final gates were `8/8` new nodes and `294/294` SA focused, with focused
identity `4826f566d053acab428e9574ddc64c72acbb57e9a12858cdddffb9a67b27e793`.
Full collection remained `4378/03bceb26...`; protected paths remained exact.
Packet `/tmp/sa-health-diagnostics-task2-3c37dbd6` contains 20 payloads, with
`SHA256SUMS` SHA-256
`6eeea62ec5ee7c4654f1ea41a17cc6c6ee4842f4e7fa76e9b48e915b66808955`.

## 8. Task 3 - Extension Diagnostics And Terminal Conflict

Task 3 added the exact eight-node extension-flow ledger and evolved only the
two authorized dependency/outbox owners. RED and final GREEN collection are
byte-identical at
`4386/c3969f490e2adc485668916784ce0f48d9d974bf11f312b1c635b1ea110b0fc6`;
the exact eight-row addition stream is
`50da1c4b1e9f6be9602f7e77436674eb9cd7d911e6a8068a5862e54f843c8a08`.
Initial RED contained exactly ten failures: the eight absent collector/flow
behaviors plus the two authorized existing owners.

Product commit `259ff1ee` adds one closed, bounded browser-side diagnostic
collector and threads it through Alpha Picks, manual fetch, and Market News.
Every detail failure calls the typed recorder before its count changes; native
transport and local persistence retain distinct stable causes. Telemetry
copies diagnostics into the immutable outbox identity, while server
`event_conflict` terminally removes the queue item as unavailable. Firefox
loads the new literal dependency; the protected Chrome manifest is unchanged.

Pre-commit review found one plan-owned gap: a job function could throw before
an owning flow recorded a phase diagnostic, while the common queue still
created a failed terminal result. The existing successful-save Task 3 node
gained a thrown-job subcase, then `enqueueSaSyncJob` was corrected to record a
fixed `extension_runtime/unknown_failure` phase entry without retaining raw
error text. This changed no node ID or staged identity.

Final gates are exact Task 3 owners `10/10`, SA focused `302/302` with identity
`73b47eef08012db1fcef649cc0c8cdaf989f2cd6bdb13d999b70c504e2986269`,
full collect-only `4386/c3969f49...`, and protected paths `17/17`. Eight
explicit scenario artifacts contain zero URL/body/token/email/path sentinels.
Packet `/tmp/sa-health-diagnostics-task3-536e30e5` contains 35 payloads; its
`SHA256SUMS` SHA-256 is
`83af6697a09d1197ec3f89b79fb5cf7681e04d8841a4d1d0008ba4dd48896d62`.
The rejected broad-glob sentinel scan and relative-path fixture invocation are
retained and excluded from accepted evidence.

No live provider request, extension install/reload, production write,
schedule/repair action, merge, push, or destructive operation occurred.

## 9. Task 4 - Structural Chain State And Bounded Recurrence

Task 4 added the exact eight-node health ledger. RED and final GREEN collection
are byte-identical at
`4394/b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb`;
all eight RED nodes entered their test bodies and failed only because the old
health service lacked the structural reducer, exact degraded outcome, typed
diagnostic projection, or bounded recurrence.

Product commit `2ee63df6` replaces the ambiguous top-level `ok` with one
six-segment `chain_state` authority. Capture and historical repair rows no
longer alter that state. The latest capture preserves exact workload, outcome,
counts, occurrence time, admitted diagnostics, omitted count, and a read-only
20-run recurrence grouped by workload/stage/reason. Legacy rows expose
`diagnostics_status=absent` without inferred cause; malformed stored
projections fail closed to the fixed rejection marker. The
`detail_failures_recorded` producer is gone and degraded captures now remain
warning history under `capture_degraded`.

Final gates are `19/19` health owners, `310/310` SA backend focused, full
collect-only `4394/b0285ee3...`, and protected paths `17/17`. A whole-file
formatter attempt was rejected because these existing files are not a global
Black baseline; all unrelated formatting hunks were removed before commit.
Packet `/tmp/sa-health-diagnostics-task4-c44521d8` contains 20 payloads. Its
`SHA256SUMS` SHA-256 is
`590bdeb881cc0d31f58e0d8b3ed1c162b5f92960e6d12972395e50922a93e161`.

No live provider request, extension install/reload, production write,
schedule/repair action, merge, push, or destructive operation occurred.

## 10. Task 5 - Truthful Frontend Diagnostics

Task 5 applied the exact frontend `+8/-3` semantic ledger. RED collection and
final collection are byte-identical at `101 files / 1177 /
9530dcd91d8a7d684faa5e56f2986fbaeaa22e1d89f67818a12ed5d8ca77d1b1`;
all eight planned additions were absent from the base and all three semantic
renames existed exactly once. Product commit `d9c4361e` replaces the open
`ok` presentation with the closed three-state chain DTO, renders capture
history independently from chain health, and exposes only admitted typed
diagnostic fields. Raw diagnostic `message` and backend `detail` values are
not rendered in either normal or Developer mode.

The exact 44-add/2-retire i18n ledger landed in both locales. Settings count is
`827`, locale total is `1911`, frozen inventory constants remain unchanged,
and the new Traditional Chinese copy uses `擷取`. One pre-existing search
alias still contains `攝入`; Task 5 neither touched nor rendered that alias,
and the browser contract directly proves the term is absent from the DOM.

Final gates:

- SA owner projection `79/0d6568f19a5d572688bbbb303f32c7cff5f86b19ce51bcc5b730b86beb91753d`
  passed `79/79`;
- Settings projection `249/a3a5e481cace86991db6d8ec5da56c2d973d224e1cb1de57f631c210a646a16e`
  passed `249/249`;
- the accepted sequential full run passed `101 files / 1177/1177`;
- typecheck, build, and the visible-literal scanner passed, with scanner debt
  remaining zero; and
- all 17 protected paths remained byte-identical at aggregate
  `1c5b539a05e51eef3f52e0cad9efa02063db077cfb7e190f20ccdc8b0580e0ae`.

The first default-parallel full run is rejected evidence: only the two known,
unmodified five-second timeout owners failed. Their isolated control passed
`26/26`, and the unchanged final bytes then passed the complete sequential
run. The first browser harness run is also rejected because it read an
existing closed `<details>` element without opening it; the final harness
opens the element and changes no repository byte.

Hermetic Chrome at `1322 x 777` and `390 x 844` passed normal and Developer
cases. Healthy, degraded-capture, structurally degraded/interrupted,
historical-repair, typed browser/native/local-persistence, legacy-absence, and
rejected-diagnostic states all rendered with distinct honest copy. Every
network entry was GET; mount, focus, visibility, and idle issued no automatic
SA refresh, while each explicit recheck added exactly one local health GET.
Raw sentinels, `攝入`, overlap, and page overflow were absent. Four screenshots
were inspected at original resolution; Vite, Chrome profiles, and port `8472`
were removed afterward.

Packet `/tmp/sa-health-diagnostics-task5-258e387d` contains 35 payloads. Its
`SHA256SUMS` SHA-256 is
`8934320b46b68fe8db1f200528acb10d6dec924cb3f6a14daa29d5228541d27c`.
No live provider request, extension install/reload, production write,
schedule/repair action, merge, push, or destructive operation occurred.

## 11. Next Gate

Task 6 independently replays M1-M9, runs final backend/frontend/native/static
and browser admission, accounts for generated artifacts, and then stops for
the combined implementation review. Any stop condition still overrides the
batch ruling; Task 7 and merge remain unauthorized.
