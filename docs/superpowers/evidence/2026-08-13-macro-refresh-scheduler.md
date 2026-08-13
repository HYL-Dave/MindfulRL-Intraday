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

Task 1 may begin RED-first under the batch ruling. Any identity mismatch,
unowned collateral, or other plan stop condition ends the batch immediately.
