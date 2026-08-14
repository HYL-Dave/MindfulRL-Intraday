# Macro Refresh and Scheduler Integration Evidence

> **Status:** TASKS 0-5 COMPLETE; FULL IMPLEMENTATION REVIEW REQUIRED; TASK 6
> NOT AUTHORIZED
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

## 8. Task 3 - Shared Frontend Controller And Attended Busy

Focused review returned GREEN at `80e8498a`. Product/tests commit `9fea8f49`
extracts one cache-backed schedule hook and table without changing the Data
Sources registry set. The controller rejects stale request sequences, shares
in-flight reads, prevents duplicate mutations, preserves price/news/SEC cache
mappings, and classifies future macro sources from `write_target`. Successful
FRED-series terminal revisions invalidate `macro_status` plus
`macro_snapshot`; the other four macro sources invalidate `macro_status` only.
Failed, skipped, or busy macro outcomes preserve stored-data keys.

The fast-terminal owner proves that a manual run which reaches a newer
successful terminal revision before any observed `running` frame still
invalidates the exact macro keys. It also holds an older GET across the manual
mutation and proves the cache generation discards that stale completion. A
hard mutation ref prevents rapid duplicate POSTs.

The attended writer-busy correction is trigger-scoped. `scheduler` contention
remains a pure defer before attempt/result/row creation. `api`, `cli`, and
`manual` contention each produce one failed canonical `fetch_fred_series` row
and one transient `skipped` result with code/reason `macro_calendar_busy`, with
zero provider calls and no automatic cadence or durable-state mutation.

Commit-time self-review found that provider configuration work no longer
reached the extracted controller's busy guard. The strengthened existing Task
3 node first failed because schedule controls remained enabled, then passed
after `DataScheduleTable` received the external provider-operation busy state.
No node ID or staged identity changed.

GREEN admission:

| Gate | Result |
|---|---|
| backend full collection | `4,359 / c100ee5de4ad42c490e6048e4b7cf22540e417f579987c676796663608d17afd` |
| backend focused | `393 / 0dd72ab8e64fa0f8324c441b67ad65a10d886692dc41c0d2d487b403aac6a5d5`; `393 passed` |
| frontend full collection | `101 files / 1,167 / 461b38278c8125f6995e35a883771789d83df6b641236a02bb44b269e039f9bf` |
| frontend focused | `104 / 8fd324f07dbb75d53ce3629a94922ff4b50b244fea8ef0bd40777d36d82c327a`; `104 passed` |
| frontend static gates | typecheck `0`; scanner `37/20/0/20` |
| protected boundary | `828/828`, zero mismatch |
| unowned product/test paths | `0` |
| live provider requests | `0` |

Final Task 3 packet:
`/tmp/macro-refresh-scheduler-task3-80e8498a`, 23 payloads;
`SHA256SUMS` SHA-256
`3f16fe7a1001327bb64718a570cf847e2debe7f3c9750a4246342469d5c7b615`.
Task 4 proceeds under the batch ruling. Task 6, merge, push, and live-provider
traffic remain unauthorized.

## 9. Task 4 Stop - Snapshot Heading Owner Scope

Task 4 added its reviewed tests first and collected the exact final frontend
identity, `101 files / 1,172 /
d40a30d5e50690f79b644e0b25122da02441eb0cf54ab02793d02269419e23cb`.
The six new Task 4 nodes and the post-PG replacement failed RED for the planned
missing schedule, copy, and CSS behavior. After the bounded product work, the
combined owner run passed `37` nodes and failed one existing Macro node.

`renders English Macro Data status snapshot and table headings` previously read
every `th` in the mounted page. The reviewed schedule table correctly adds five
headings, so the node now receives ten headings while still expecting the five
FRED snapshot headings. This is deterministic collateral, not a product defect
and not permission to weaken the expected FRED headings.

The bounded amendment permits exactly one test-body edit: select
`.settings-fred-table th` and keep the existing five-value expectation, node ID,
and all other bytes in that node unchanged. Counts and normalized collection
streams do not change. Current product/test work remains uncommitted; no live
provider or production data was touched. Tasks 4-5 remain paused until focused
review returns GREEN.

Rejected owner report:
`/tmp/macro-task4-existing-heading-collateral.json`, `14` total / `13` passed /
`1` failed.

Partial packet:
`/tmp/macro-refresh-scheduler-task4-stop-2b5690dc`, five payloads;
`SHA256SUMS` SHA-256
`827db0b86e2e80d7c02338f6f0328f26b347271332633fda8454fc92cfef9dc3`.

## 10. Task 4 Stop - Rendered Source Wrapping Cascade

After the provider-fixture amendment, the final focused suite passed
`109/109`, the separate post-PG owner file passed `14/14`, typecheck and build
exited zero, the scanner returned the same `37/20/0/20` as a detached base
control, and one complete sequential frontend command passed `101 files /
1172/1172`. The required early real-browser gate then failed before Task 4
commit.

At desktop `1322 x 777`, the fixture rendered the exact ten Data Sources rows,
the exact five Macro rows, and the reviewed `30/11/12/12/35` column widths.
The table scroller stayed inside the Settings scroll owner and neither the page
nor main owner overflowed. However, every Macro source cell computed to
`white-space: nowrap`: `.data-table td` has specificity `(0,1,1)` and therefore
outranks the later bare `.settings-schedule-source-cell` selector `(0,1,0)`.
The Finnhub economic and earnings source cells measured `scrollWidth 326/320`
against `clientWidth 294`. `overflow-wrap: anywhere` was present but cannot
wrap while `nowrap` wins.

This is a deterministic rendered product defect and stop condition 19, not a
test preference. The bounded amendment authorizes one selector replacement to
`.settings-schedule-table td.settings-schedule-source-cell`, with all four
existing declarations unchanged, and an in-place strengthening of the already
added wrapping CSS node. No node ID, count, staged identity, or other
product/test edit is authorized. Desktop and mobile computed-style replay must
prove normal wrapping, no source-cell overflow, exact columns, bounded table
scrolling, and zero page/main overflow or cell overlap before Task 4 resumes.

The first browser process was rejected because sandbox network isolation could
not reach local Vite. A later diagnostic used the wrong subsection attribute
and was also rejected; the product DOM uses `data-settings-location`, not
`data-settings-anchor`, for `source_schedules`. Neither run is used as product
evidence. The admitted RED is the recorded computed-style/geometry artifact.
No live provider or production data was contacted or changed.

Partial packet:
`/tmp/macro-refresh-scheduler-task4-stop4-83f881a0`, `12` payloads;
`SHA256SUMS` SHA-256
`f26131aa40557c89e39eb5dcbb8ade4645a5317c8d15d566ea463f29db9c5cf0`.

## 11. Task 4 - Macro Settings Surface And Rendered Layout

Focused review returned GREEN at `6dc7b51c`. Product/tests commit `052c9134`
lands the five Macro schedule rows, one `data_sync` group-scoped schedule
controller, truthful three-state automation copy, local-only status reload,
and the reviewed schedule-table layout. The two visible schedule consumers
share one controller through the initial load and the first 30-second poll;
leaving `data_sync` unmounts that owner.

The final CSS correction changes only the source-cell selector to
`.settings-schedule-table td.settings-schedule-source-cell`. Its four
declarations are byte-identical to the reviewed rule. Fixture-only Chrome
replay at `1322 x 777` and `390 x 844` proves computed
`white-space: normal`, `overflow-wrap: anywhere`, exact
`30/11/12/12/35` columns, wrapped long Finnhub descriptions, no source-cell
overflow, bounded table scrolling, and zero page/main overflow or overlap.
Every browser request was GET; no provider or production endpoint was used.

Final admission:

| Gate | Result |
|---|---|
| frontend collection | `101 files / 1,172 / d40a30d5e50690f79b644e0b25122da02441eb0cf54ab02793d02269419e23cb` |
| frontend focused | `109/109` |
| post-PG owner | `14/14` |
| frontend full | `101 files / 1,172/1,172`, one sequential command |
| static gates | typecheck `0`; build `0`; scanner `37/20/0/20` |
| protected boundary | `826/826`, zero mismatch after excluding the two paths added to the reviewed owned set |
| live traffic | `0`; fixture ledger contains GET only |

The four reviewed stop packets remain linked in the final packet. They record
the heading selector scope, duplicate controller ownership, stale schedule
fixture, and rendered CSS specificity defects without treating any rejected
run as GREEN evidence. Final packet:
`/tmp/macro-refresh-scheduler-task4-6dc7b51c`, `33` payloads;
`SHA256SUMS` SHA-256
`1cf69c36ed1cb6e03777f8b9482df1d5a31e36501d141a78692cf8935b329908`.

Task 5 proceeds under the batch ruling. Task 6, merge, push, and live-provider
traffic remain unauthorized.

## 12. Task 5 Stop - Active Production Writer During Admission

All nine mutations turned their named semantic owners RED and restored the
complete owner bytes before the next mutation. M8 and M9 additionally produced
the required real-Chrome computed-style/geometry failures. Final identities
were exact: backend `4,359 / c100ee5d...`, backend focused `393/393`, frontend
`101 files / 1,172 / d40a30d5...`, and frontend focused `109/109`. Frontend
full passed `1,172/1,172`; typecheck, build, scanner, the scratch DB proof, and
the two-process/descriptor lock probes all passed. The adjusted protected
boundary is `826/826` with zero mismatch.

The first native command was rejected as an input-boundary diagnostic: the
isolated worktree had no ignored `config/.env`, so 17 conditional PG contract
nodes skipped and the result was `4,330 passed / 29 skipped`. A targeted run
with the main-root file visible only through a temporary symlink passed all
`21/21` `test_db_backend.py` nodes. The resulting canonical command then
matched the planned result exactly: `4,359 seen = 4,347 passed / 12 skipped /
0 failed / exit 0`. The symlink was removed immediately; no secret value was
copied into an artifact.

The Task 5 browser matrix exercised desktop `1322 x 777` and mobile `390 x
844`. Both rendered the exact five Macro rows and ten Data Sources rows,
computed the exact `30/11/12/12/35` columns, wrapped long source descriptions,
and kept horizontal overflow inside the table scroller with no page/main
overflow or overlap. Mount, the 30-second idle tick, focus, visibility change,
and local status reload produced zero POSTs. One explicit FRED click produced
exactly one POST, two schedule reads (`running` then `succeeded`), and an honest
`上次成功` terminal surface. The first browser run is rejected evidence because
its harness incorrectly required focus to issue a second GET while the idle
single-flight was still admissibly shared; the corrected harness tests the
approved zero-POST contract without forbidding read coalescing.

Final production equality did not pass. From the `09:06:05` pre-sample to the
post-sample, `macro_calendar.db`, `config/.env`, and
`config/user_profile.local.yaml` were exact. Production `profile_state.db`
kept its inode and mode but changed size, mtime, and SHA. The Desktop app and
sidecar had been running since 2026-08-12. Read-only `job_runs` rows show the
external five-minute `sa_market_news_refresh` sequence from `01:08:44` through
`01:38:44 UTC`, one `collect.ibkr_news` scheduler run, and matching DB mtimes.
The canonical wrapper wrote its own 225,280-byte profile database under the
separate runtime root. This establishes the stop's cause but does not turn an
unequal production manifest into passing evidence.

Task 5 remains incomplete. After focused review, the user must quiesce the
Desktop/sidecar writer (or separately authorize the operator to do so). The
resume gate takes a fresh four-asset pre-manifest, reruns canonical native on
unchanged product bytes, removes the temporary config symlink, and requires a
byte-identical post-manifest. Frontend and fixture-only browser gates do not
rerun. Partial packet:
`/tmp/macro-refresh-scheduler-task5-c286d504`, `140` payloads;
`PARTIAL_SHA256SUMS` SHA-256
`0d8127ec550b857b39a3164b796dc470e7a54696c33e2b17584cdae1efc01f23`.
Task 6, merge, push, and live-provider traffic remain unauthorized.

## 13. Task 5 Resume - Quiesced Production Equality

Focused review returned GREEN at `d7c59b11`. Before the fresh pre-manifest,
Desktop/Electron/sidecar were absent, `profile_state.db` had no file opener,
and three samples across two minutes retained one inode, size, mode, and
nanosecond mtime. The implementation worktree was clean and contained no
`config/.env` entry.

The same canonical wrapper ran with the main-root `config/.env` visible only
through a temporary symlink. It completed `4,359 seen = 4,347 passed / 12
skipped / 0 failed / exit 0`. Reporter JSON SHA-256 is
`7bf1eca48e44c6886767c0502334842c8751015c20f50d2c46d341674752b4de`,
byte-identical to the previously admitted Task 5 report. The symlink was
removed by a guarded `unlink` trap before any post-manifest read. An earlier
operator command containing `rm -f` was rejected by execution policy before
process creation; it created no symlink and ran no test.

Fresh pre and post manifests cover production `macro_calendar.db`,
`profile_state.db`, `config/.env`, and `config/user_profile.local.yaml`. They
are byte-identical and each has SHA-256
`0f1c4a2b9feb6608d2a12faef8185f2aec2511cbab273ebf590061464402d376`.
There was no profile DB opener before or after the command. Stop condition 15
is therefore resolved without waiving equality or rerunning the already
admitted fixture-only frontend/browser gates.

Task 5 is complete and stops for full implementation review. Final packet:
`/tmp/macro-refresh-scheduler-task5-c286d504`, `152` payloads;
`SHA256SUMS` SHA-256
`5b7c1fd1cc6448c70dccc85e30d6368e1c3d443a3be0a4f89e994dad246b45c0`.
Task 6, merge, push, and live-provider traffic remain unauthorized.
