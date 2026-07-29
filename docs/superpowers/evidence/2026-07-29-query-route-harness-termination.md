# Query Route Harness Lifespan-Exposure Reduction Evidence

> **Status:** REVIEW READY - INDEPENDENT IMPLEMENTATION REVIEW NEXT
>
> **Product base:** `542776c2e00ae1737d5b424a3b8858b079a63e38`
>
> **Plan-review clearance:** `db7f2240399f8de31cab1bd4007ac995d213780b`
>
> **Implementation commit:** `31230232f09b39c966e21645b01637f28aa80e27`
>
> **Observed:** 2026-07-29 Asia/Taipei

## 1. Scope And Authorities

- Design:
  `docs/superpowers/specs/2026-07-29-query-route-harness-termination-design.md`.
- Implementation plan:
  `docs/superpowers/plans/2026-07-29-query-route-harness-termination.md`.
- Isolated branch/worktree:
  `codex/query-harness-termination` at
  `/tmp/arkscope-query-harness-termination`.
- `542776c2` is an ancestor of the clearance commit.
- Before Task 0, only the reviewed design, plan, and priority map differed from
  the product base.
- The main-worktree drafts
  `docs/data/IBKR_PACING_AND_ERROR_SEMANTICS.md` and
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` do not exist in this worktree.
- No product or test file has been edited.

## 2. Canonical Collections

All three normalized collections reproduced exactly:

| Collection | Count | SHA-256 |
|---|---:|---|
| backend full | `4722` | `fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` |
| `tests/test_agents.py` | `31` | `78d7cdbebb60be09616fa13f3a8b85d42373fe46e4ac896d28d7c9900cf48f1e` |
| `TestQueryEndpoint` | `2` | `5e1e62ac3baf8d2d47558d3c43679ab4423cf6eb842e3c68edea447433adf4f7` |

The owned stream contains exactly once each:

```text
tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
tests/test_agents.py::TestQueryEndpoint::test_query_endpoint_bad_provider
```

## 3. Runtime Observations

The pre-change two-node runtime remains intermittent but was non-terminating in
this Task 0 environment. Two attempts at the planned 30-second observation
were themselves closed by the command-orchestration envelope just before the
inner shell could print `timeout`'s status; neither left a pytest process. A
bounded repetition with the same node selection and a 20-second inner timeout
produced exact exit `124` and no test output.

This 20-second bound is an explicit tool-envelope deviation from the plan's
30-second observation, not the structural RED and not an A/B baseline. It is
consistent with the earlier independently recorded timeout and does not make
the incident deterministic across environments.

## 4. Structural RED

Before the test edit, the plan-owned source-boundary command inspected only the
region between the unique API-endpoint and registry banners. It exited `1`
with the exact expected failure:

```text
AssertionError: TestQueryEndpoint still enters the full-app harness via TestClient
```

There was no import, SQLite, profile, provider, or timeout failure. This is the
required structural RED; the existing runtime nodes were not misrepresented as
RED because their old harness was already intermittently capable of passing.

## 5. GREEN And Repetition

The single owned test edit is `+39/-13` in `tests/test_agents.py`. It replaces
the class's full-app fixture with the reviewed minimal FastAPI app, real query
router, async DAL sentinel, deterministic personalization stub,
`ASGITransport`, and synchronous `asyncio.run()` shell. Existing method names
and assertions remain in place.

The strengthened source gate requires the reviewed transport/router/override
shape while forbidding `TestClient`, `create_app`, and `run_in_threadpool` in
the owned region. It exited `0`.

The two owned nodes then returned:

```text
2 passed in 1.92s
```

The bounded repetition completed 20 iterations within the 120-second outer
limit. Every iteration returned `2 passed` in approximately 1.85-1.93 seconds:
`40/40` owned-node executions passed with no stall.

After the implementation commit, the complete source/target/repetition gate
was run again. The source gate exited `0`, the direct result was
`2 passed in 1.87s`, and the second 20-iteration set passed `40/40` at
approximately 1.85-1.99 seconds. Across the two bounded repetition gates,
`80/80` repeated owned-node executions passed.

Post-edit collection results are byte-identical to Section 2:

| Collection | Count | SHA-256 | Base/tip `comm -3` |
|---|---:|---|---|
| backend full | `4722` | `fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` | empty |
| `tests/test_agents.py` | `31` | `78d7cdbebb60be09616fa13f3a8b85d42373fe46e4ac896d28d7c9900cf48f1e` | empty |
| `TestQueryEndpoint` | `2` | `5e1e62ac3baf8d2d47558d3c43679ab4423cf6eb842e3c68edea447433adf4f7` | empty |

## 6. Full-Suite Diagnostics

The base command was:

```text
PYTHONUNBUFFERED=1 python -m pytest -vv --tb=short \
  -o faulthandler_timeout=120
```

It collected all `4722` nodes, then stopped at:

```text
tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint
```

After 120 seconds, pytest emitted the required all-thread dump. The decisive
frames were:

```text
main thread:
  concurrent.futures._base.Future.result
  anyio._backends._asyncio.run_sync_from_thread
  anyio.from_thread.BlockingPortal.start_task_soon
  starlette.testclient.TestClient.__enter__
  tests/test_agents.py:498 in client

AnyIO portal thread:
  selectors.select
  asyncio.base_events._run_once
  asyncio.base_events.run_forever
  anyio.from_thread.run_blocking_portal
```

This proves where the observed run blocked. It does not identify which
ambient state prevented the portal/lifespan handshake from completing and is
not upgraded into a global root-cause claim.

The operator interrupted only after the dump. The session ended and no pytest
process remained. Transcript identity:

```text
path: /tmp/query-harness-base-full.txt
lines: 128
bytes: 12036
SHA-256: bfcda909a11cd774bc0e4fd73bd9cf02b0b4304e32af6687615034e413772ca5
```

Seven setup errors appeared before the active node, but the run was partial;
they are diagnostic observations only. No normalized base non-passing set was
created or accepted.

The isolated `data/` directory contained no files before or after the run, so
no fixture move was needed. No production path was read, moved, or compared by
basename.

The implementation-tip full suite used the same instrumented command. Both
owned nodes passed in the captured transcript:

```text
tests/test_agents.py::TestQueryEndpoint::test_providers_endpoint PASSED
tests/test_agents.py::TestQueryEndpoint::test_query_endpoint_bad_provider PASSED
```

The run then stopped at the next full-app lifespan family:

```text
tests/test_api.py::TestHealth::test_status
```

Its 120-second dump again placed the pytest thread in
`starlette.testclient.TestClient.__enter__`, this time from
`tests/test_api.py:41`, while the AnyIO portal thread remained in asyncio
`select()`. The operator interrupted after the dump. No pytest process or
worktree data file remained.

Tip transcript identity:

```text
path: /tmp/query-harness-tip-full.txt
lines: 204
bytes: 18894
SHA-256: 89f7435d6510d058061110481ba21bc5f377e40dde24610bfee141c901875cce
```

This is the spec-permitted different-node diagnostic stop. It proves the two
owned nodes no longer expose the suite to the first observed lifespan stall;
it also proves that full-suite termination remains globally unresolved. The
tip transcript is partial, so no tip non-passing set or base/tip failure-set
comparison is claimed.

## 7. Protected Boundaries

Final `git diff --quiet` checks returned `0` for all product paths and for
`conftest.py` plus the five adjacent `TestClient` families. The complete
branch delta from reviewed spec tip `3216c1b9` contains only the owned test and
four authority documents. No frontend, data file, provider, Gateway,
scheduler, browser, or production state changed.

## 8. Node Accounting

Accounting is exact `+0/-0`. Both owned IDs survive exactly once, no helper was
collected, and the full/agents/owned normalized streams are byte-identical in
both directions.

## 9. Integration

The implementation is review-ready. Merge, main-worktree changes, price-truth
rebase, and price product RED remain unauthorized pending independent
implementation review and explicit user approval.
