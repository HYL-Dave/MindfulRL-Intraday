# Macro Refresh and Scheduler Integration Evidence

> **Status:** TASKS 0-2 COMPLETE; TASK 3 STOPPED ON BOUNDED AMENDMENT REVIEW;
> TASK 6 NOT AUTHORIZED
>
> **Date:** 2026-08-13
>
> **Plan authority:** `f9b69913`
>
> **Product grounding tip:** `bea5890f`
>
> **Task 0 packet:** `/tmp/macro-refresh-scheduler-task0-f9b69913`

## 1. Process Boundary

Focused re-review returned GREEN at `f9b69913`. The user authorized Tasks 0-5
to proceed continuously while preserving the plan's per-task product/tests and
evidence/docs commits, RED/GREEN packets, exact staged identities, and every
hard stop condition. Task 5 remains the combined implementation-review gate.
Task 6, merge, push, and live-provider traffic remain unauthorized.

Task 0 changed no product or test byte. The implementation worktree was clean
before grounding and after artifact cleanup. Product drift from the reviewed
product tip is empty.

## 2. Toolchain And Base Identities

The pinned reporter, normalizer, native wrapper, lockfiles, Node 22.14.0,
Vite 5.4.21, Vitest 4.1.8, and Chrome 150.0.7871.128 were verified by full
version or SHA-256 in the packet.

| Stream | Re-collected identity |
|---|---|
| backend base | `4,341 / 883b1148e8759ea8825ffd8b825db0339dc1e6ca5aefe4c04344b868b3ff1264` |
| backend focused base | `375 / 1c3ce4accfe583b6c059d1c52414376cd1697f24ce8f968afbc844250bd9a1d5` |
| frontend base | `100 files / 1,159 / f19472dd04c73afd979d37f4c083ff8246a007816d58429a2c12295eaadc5e67` |
| frontend focused base | `96 / 9877be8adf0973c2b749f6460156ae86021cc4a91f3d4caab366bd9ce66da46f` |

Focused runtime baselines passed as `375 passed` and `96 passed`. The existing
native control remains admissible because the docs-only authority range has
zero product drift: `4,329 passed / 12 skipped / 0 failed`, report SHA-256
`534cfe6f...`.

## 3. Predicted Stages

Every addition/removal stream was rebuilt from the committed plan text. All
new IDs are absent from the base, every removal ID is present exactly once,
and the resulting identities match the plan:

| Stage | Predicted identity |
|---|---|
| backend Task 1 | `4,349 / 372fe6ab...` |
| backend Task 2 final | `4,359 / c100ee5d...` |
| backend focused | `375 -> 383 -> 393` |
| frontend Task 3 | `101 files / 1,167 / 461b3827...` |
| frontend Task 4 final | `101 files / 1,172 / d40a30d5...` |
| frontend focused | `96 -> 104 -> 109` |

The packet also contains the complete normalized streams and full SHA-256
values; abbreviated values here are labels, not admission authority.

## 4. Ownership And Packet

- Owned product/test paths: `34` (`29` existing plus `5` planned new paths).
- Tracked non-doc paths: `857`.
- Byte-protected paths at Task 0: `828`.
- Task 0 packet: `42` payloads; `sha256sum -c` passed for all entries.
- `SHA256SUMS` SHA-256:
  `2859cfc0f842dcb9063726a33bf43e8fc4d02089bd397fae0ed31c68d1390cef`.

## 5. Task 1 - Shared Execution And Writer Lock

Product/tests commit: `c689de80` (`feat: serialize macro calendar execution`).
Artifact packet: `/tmp/macro-refresh-scheduler-task1-861f2305`, 31 payloads,
manifest SHA-256
`85c8b6faa16badf3d2b92c4853336c021179575af005938e49c2bb107090b0e6`.

The eight planned nodes were added before product code. Collection was exactly
`4,349 / 372fe6ab268fabd0d0bbb66dd26f93fb7f684d63ad19885c3d587bfefa50c340`;
RED was `8 failed`, all caused by the intentionally absent `execution` or
`write_lock` module. No collection/import error in an existing module was
accepted.

The implementation moves all six provider/date/parameter formulas to
`src/macro_calendar/execution.py`, which has no telemetry dependency. Direct
`run_job()` and scheduled `run_source()` each create and finish exactly one
canonical `fetch_*` row. A scheduler-held lease is same-process,
same-thread, active, and single-use; the dispatcher rejects missing authority,
foreign, released, or reused leases rather than reacquiring the non-reentrant
lock.

`src/macro_calendar/write_lock.py` requires both a process-local lock and POSIX
`flock`, rejects symlink/non-regular lock files, fails closed when `fcntl` or
the lock path is unavailable, uses one bounded deadline, and releases the file
descriptor and thread lock on every tested terminal path. A real two-process
test proves mutual exclusion and later reacquisition. The fd witness covers
success, body failure, symlink rejection, and missing-`fcntl` rejection.

GREEN admission:

| Gate | Result |
|---|---|
| eight new owners | `8 passed` |
| backend focused | `383 / de8eb8c4e4549deb7a7b66e9a2dd6e51049106f9a76a8ddd4287c98f4050eeb0`; `383 passed` |
| backend full collection | `4,349 / 372fe6ab268fabd0d0bbb66dd26f93fb7f684d63ad19885c3d587bfefa50c340` |
| protected Task 0 boundary | `828/828`, zero mismatch |

The first protected-tree command was rejected because `git hash-object`
invoked the unavailable git-crypt clean filter in the locked worktree. Its raw
error is retained; the admitted check uses `git hash-object --no-filters` and
compares raw worktree bytes against all 828 Task 0 blobs.

Task 2 may begin RED-first under the batch ruling. Any identity mismatch,
unowned collateral, or other plan stop condition ends the batch immediately.

## 6. Task 2 - Registry, Arbitration, And Automation Truth

Product/tests commit: `792f3d9c` (`feat: schedule macro calendar refreshes`).
Artifact packet: `/tmp/macro-refresh-scheduler-task2-798527b0`, 20 payloads;
`SHA256SUMS` SHA-256
`3537047858f85ea463649b840f9047e55328712dcedc2181fefbc2df621eb2e5`.

The ten planned nodes were added before product edits. Collection was exactly
`4,359 / c100ee5de4ad42c490e6048e4b7cf22540e417f579987c676796663608d17afd`.
The owner run was `9 failed / 9 passed`: the existing interrupted-runtime
reconciliation already satisfied its new contract, while the other nine Task
2 nodes failed only at the planned missing registry, arbitration, snapshot,
or read-projection boundaries. The already-landed eight Task 1 nodes remained
GREEN, so this was admitted as the Task 2 RED state rather than misreported as
ten failures.

The five reviewed sources now use canonical `fetch_*` job identities, default
disabled intervals, exact provider maps, and `macro_calendar.db` presentation
metadata. Market and macro writers have independent one-per-tick arbitration.
A macro writer occupied in this tick, an earlier tick, or another process
defers the next source before `_LAST_ATTEMPT`, result state, or telemetry row
creation. Missing provider configuration is checked only after the selected
source owns the writer lease and one canonical attempt/row exists; it performs
zero provider work and finishes that row failed. Scheduled FRED series always
passes `full_refresh=False` outside telemetry payloads.

`/macro/snapshot` no longer publishes `auto_refresh_enabled`. The replacement
projection opens an existing profile database read-only with `query_only`,
returns default-disabled truth when the file/table is absent, returns unknown
for an invalid/non-file store, and reads the exact five settings when present.
Provider health now derives FRED automation from the two FRED schedule rows,
including enabled-source count; `macro_calendar_enabled` remains only the
existing job/agent capability gate. Finnhub health gained no calendar field.

GREEN admission:

| Gate | Result |
|---|---|
| all macro integration owners | `18 passed` |
| backend focused | `393 / 0dd72ab8e64fa0f8324c441b67ad65a10d886692dc41c0d2d487b403aac6a5d5`; `393 passed` |
| backend full collection | `4,359 / c100ee5de4ad42c490e6048e4b7cf22540e417f579987c676796663608d17afd` |
| PG-unreachable smoke owners | `13 passed` |
| protected Task 0 boundary | `828/828`, zero mismatch |
| unowned product/test paths | `0` |
| live provider requests | `0` |

Task 3 may begin under the batch ruling. Task 6, merge, push, and live-provider
traffic remain unauthorized.

## 7. Task 3 Stop - Attended Macro Busy Visibility

Task 3 reached its exact RED/GREEN collection identity and the focused owners
while product changes were still uncommitted. Admission review then found two
race boundaries.

The frontend manual-run owner first proved that a short macro run can complete
before the first poll observes `running`. The old completion detector required
`running -> terminal`, so that valid success left `macro_status` and
`macro_snapshot` retained. The strengthened existing node failed on both keys;
the bounded correction compares the authoritative terminal revision, including
attempt/result/durable timestamps, and now passes without changing its node ID.

More importantly, an isolated no-provider probe at
`/tmp/macro_manual_busy_probe.py` proved that
`run_source("fred_series", trigger_source="api")` returns
`{status: "deferred", reason: "macro_calendar_busy"}` while leaving
`_LAST_RESULT`, scheduler state, and telemetry empty whenever another macro
writer owns the gate. That behavior is correct for automatic scheduler work
but violates the approved design for explicit manual/API actions: the POST has
already returned `started`, so the UI receives no later busy truth at all.

The amendment keeps automatic `scheduler` deferral byte-for-semantics. For an
attended `api`, `cli`, or `manual` trigger only, the same busy boundary must
create one failed canonical `fetch_*` telemetry row, publish one transient
`skipped` result with stable `macro_calendar_busy` code/reason, perform zero
provider work, and leave `_LAST_ATTEMPT` plus durable scheduler state unchanged.
The existing lock-busy backend owner and existing frontend manual/failure owners
are strengthened in place; collection counts and all pinned stream hashes stay
unchanged.

Per stop condition 2, no further product edit, Task 3 commit, Task 4 work, or
Task 5 admission is authorized until focused review returns GREEN.

Partial review packet:
`/tmp/macro-refresh-scheduler-task3-stop-b010dfb7`, 12 payloads;
`SHA256SUMS` SHA-256
`c27a2fde406d90089f1e4deae9a45ce8c751341dbee2322f51e637f8306bf68c`.
It contains the exact RED/GREEN collection streams, current uncommitted owner
bytes and diff, `104 passed` focused transcript, typecheck/scanner outputs, the
isolated current-behavior probe, and their raw transcripts. No provider request
or production write occurred.
