# Macro Refresh and Scheduler Integration Evidence

> **Status:** TASK 0 COMPLETE; TASKS 1-5 BATCH-AUTHORIZED; TASK 6 NOT
> AUTHORIZED
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
