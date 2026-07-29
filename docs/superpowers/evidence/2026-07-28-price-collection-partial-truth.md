# Price Collection Partial-Truth Evidence

> **Status: TASK 0 RESTART BLOCKED - TEST_API LIFESPAN STALL**
>
> **Historical blocked-run base:** `542776c2e00ae1737d5b424a3b8858b079a63e38`
> **Restart base:** `2edf12e11a8ff9299a9b65b900309c8ed218b717`
> **Plan-review clearance:** `15933c316a68efd7e503f2778aba68affa2cb4c1`
> **Restart clearance:** `5fecce6536f5d9f4a13903a6c1059e235ba15324`
> **Observed:** 2026-07-29 Asia/Taipei

The historical Task 0 attempt stopped under plan Stop Condition 11. No product
file was edited and no partial full-suite output is accepted as an A/B
baseline. The reviewed query-route harness is now merged and this branch is
rebased. Focused review of `7844429a..5fecce65` returned GREEN with zero
findings and authorized a full Task 0 restart. That restart reproduced every
collection and focused gate, then stopped under the same condition at the next
untouched lifespan family, `tests/test_api.py::TestHealth::test_status`.
Product RED remains unauthorized.

## 1. Scope And Authorities

- Design authority:
  `docs/superpowers/specs/2026-07-28-price-collection-partial-truth-design.md`.
- Implementation authority:
  `docs/superpowers/plans/2026-07-28-price-collection-partial-truth.md`.
- Clearance branch: `codex/price-collection-truth`.
- `542776c2` remains the historical blocked-run base.
- Reviewed harness tip `2edf12e1` is the restart base and an ancestor of the
  rebased branch.
- `5fecce6536f5d9f4a13903a6c1059e235ba15324` is the exact focused-reviewed
  restart clearance.
- The merged harness changes only `tests/test_agents.py` plus its authority
  documents; the rebased price-truth delta from `2edf12e1` is docs-only.
- Main-worktree drafts
  `docs/data/IBKR_PACING_AND_ERROR_SEMANTICS.md` and
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` remained untracked and were not
  read as implementation authority, edited, staged, moved, or deleted.

## 2. Canonical Baseline

The four canonical collections reproduced exactly again during the authorized
restart:

| Gate | Reproduced result |
|---|---|
| Backend full collection | `4722`; `fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` |
| Backend focused collection | `151`; `3c07d208ced889497521a779ae46dd88403277c34055c00ba9fd74ada08da428` |
| Backend focused composition | direct `63`, worker `4`, scheduler `84` |
| Frontend full collection | `96` files / `1074` nodes; `e322e7a51e83eedb8b3c7b1fd99e6033f496031968c1a2cb3f59974bfd994f47` |
| Frontend focused collection | `86`; `739385b104c147744e7421f030e3fc628b2d99a981406c9c13aeb25c2a70a479` |
| Frontend focused composition | Settings `36`, resources `14`, display `36` |

Focused behavior also reproduced:

- backend direct/worker/scheduler: `151 passed`;
- frontend focused: `3` files / `86 passed`;
- visible-literal scanner twice: `36/20/0/20` both times;
- tool/no-PG focused gate: `16 passed`, retaining central/OpenAI/Anthropic
  counts `53/54/54`;
- no-PG runtime smoke: `23/23`, `ok=true`, `pg_attempts=[]`.

### 2.1 Isolation correction before grounding

The initial empty-data assertion found an ignored 143,360-byte
`data/profile_state.db` created by the earlier 2026-07-28 baseline attempt.
It was not production data:

```text
isolated inode: 90586961
isolated size: 143360
isolated mtime: 2026-07-28 23:21:11 +0800
isolated SHA-256: fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d
main-worktree production inode: 127284276
main-worktree production size: 43962368
```

The fixture and its WAL/SHM companions were moved reversibly to `/tmp`; none
was deleted. Restart focused tests later recreated the same deterministic
fixture SHA. Before the full-suite attempt, that file was moved to:

```text
path: /tmp/price-truth-restart-task0-profile_state.db
inode: 90597154
size: 143360
SHA-256: fcfbadad164a67b48e4e94077ef8ceba15b8126b72403ac869f41a18baf2353d
```

`data/` was empty before the restart full-suite attempt and remained empty
after it was stopped.

## 3. RED Evidence

Not started. Task 0 did not close, so product RED work is unauthorized.

## 4. GREEN Evidence

Not started.

## 5. Node And Resource Accounting

Only the unchanged base collections in Section 2 are established. Planned
`+17/-0` backend, `+2/-0` frontend, and resource deltas have not been applied.

## 6. Mutation Evidence

Not started.

## 7. Protected Boundaries

Task 0 Step 6 was not run after either Stop Condition 11 event. The Git
worktree was clean before this blocked evidence was authored, and no product
or test path was edited.

## 8. Full-Suite A/B

### 8.1 Historical pre-harness attempt

The base full suite was launched from empty isolated `data/` with unbuffered
verbose output:

```text
PYTHONUNBUFFERED=1 python -m pytest -vv --tb=short
```

It stopped making progress at:

```text
tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
```

After more than 70 seconds without another node line, the run was interrupted
with exit `130`. The partial transcript is diagnostic only:

```text
path: /tmp/price-truth-base-full.txt
lines/bytes: 58 / 5348 (plus one unterminated active-node line)
SHA-256: 7c4f83d2d3025e8e48b6a177bbdafa75a59b98beff5d8da23cd1453716445f6d
partial failures seen before the hang: 2
```

No normalized non-passing set was derived from this transcript. A bounded
single-node reproduction also stopped at fixture setup:

```text
timeout 20s python -m pytest -vv --tb=short \
  tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
exit: 124
last line: tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
```

No pytest process remained afterward. This is a concrete EIR-002/harness
diagnostic, not evidence against the price-truth product design.

### 8.2 Restart after the harness merge

The authorized restart used the reviewed instrumentation:

```text
PYTHONUNBUFFERED=1 python -m pytest -vv --tb=short \
  -o faulthandler_timeout=120
```

Both converted `test_agents` provider-route nodes passed in full-suite
context. The run then passed
`tests/test_api.py::test_fixed_task_runtime_routes_mount_on_real_app` and
stopped at:

```text
tests/test_api.py::TestHealth::test_status
```

At 120 seconds, faulthandler emitted all-thread stacks. The pytest thread was
waiting in `starlette.testclient.TestClient.__enter__`, reached from the
`tests/test_api.py:41` client fixture through AnyIO's blocking portal. The
portal thread was idle in the asyncio selector. This identifies the blocking
boundary but not the ambient root cause or suspended lifespan coroutine.

The operator sent Ctrl-C only after the dump; the execution session reported
exit `1`. No pytest process remained. The diagnostic transcript was preserved
under a unique name:

```text
path: /tmp/price-truth-restart-blocked-20260729-full.txt
lines/bytes: 204 / 18895
SHA-256: 1e2f8907b3936ccfdd2ace0cfb7f6d2b221752c4dd6c9d16f34288dc74872e1c
```

No normalized non-passing set was derived. This transcript does not establish
that `test_api.py` always stalls, does not weaken the earlier harness result,
and is not an A/B baseline.

## 9. Review Resolution

Plan F1 was resolved at `9d1e648a`: the mounted frontend node now includes the
existing `Settings provider config authority` describe prefix, matching both
predicted hashes. The 26-slot advisory now requires the exact temporary
mutation diff in this packet when implementation eventually proceeds.

Independent harness implementation review returned GREEN for
`db7f2240..2edf12e1`; `master` then fast-forwarded to exact reviewed tip
`2edf12e1`. Merged verification reproduced full `4722/fcdb1b7d...`, agents
`31/78d7cdbe...`, owned `2/5e1e62ac...`, and `2 passed`. The price branch was
rebased while preserving both reviewed priority-map histories.

The historical blocker at `test_providers_endpoint` is structurally removed.
Focused review of the rebased handoff returned GREEN, and the restarted full
suite proved that converted exposure passes before the untouched
`test_api.py::TestHealth::test_status` family stalls. The instrumentation
therefore did its intended job, but the repository still cannot produce the
complete same-environment baseline required by this plan. Stop Condition 11
remains active. Silently excluding a node, accepting partial output, running
protected Step 6, or starting product RED remains prohibited pending a
separately reviewed resolution.

## 10. Integration And Read-Only Release Observation

The harness prerequisite is merged; price-truth product integration is not
started. Provider calls, production writes, repair, browser work, and release
observation remain unauthorized.
