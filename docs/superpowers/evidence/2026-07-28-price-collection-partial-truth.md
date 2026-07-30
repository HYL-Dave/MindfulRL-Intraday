# Price Collection Partial-Truth Evidence

> **Status: TIERED TASK 0 V2 BLOCKED - EOF/EXIT HANDSHAKE INVALID**
>
> **Historical blocked-run base:** `542776c2e00ae1737d5b424a3b8858b079a63e38`
> **Restart base:** `e6d4b7fac7e91c59e855a7f543caac4f57094d86`
> **Plan-review clearance:** `15933c316a68efd7e503f2778aba68affa2cb4c1`
> **Restart clearance:** `5fecce6536f5d9f4a13903a6c1059e235ba15324`
> **Tiered-contract clearance:** `3863b3be02034b3278f58d7090dcf0bc20445fe3`
> **Runner-design clearance:** `1d08a9f30a87066ea0a2e3b3274a22210cdfa57d`
> **V2 plan-review clearance:** `00d35376511b8bd28c16dd9c40415e0ddbc533ab`
> **Observed:** 2026-07-29 through 2026-07-31 Asia/Taipei

The historical Task 0 attempt stopped under plan Stop Condition 11. No product
file was edited and no partial full-suite output is accepted as an A/B
baseline. The reviewed query-route harness is now merged and this branch is
rebased. Focused review of `7844429a..5fecce65` returned GREEN with zero
findings and authorized a full Task 0 restart. That restart reproduced every
collection and focused gate, then stopped under the same condition at the next
untouched lifespan family, `tests/test_api.py::TestHealth::test_status`.
The later causal diagnosis selected V6 without identifying a code seam and is
merged at `e6d4b7fa`. Focused review cleared the tiered verification contract
at `3863b3be` with zero findings. The authorized Task 0 restart reproduced
every collection and focused gate, but its runtime controller violated the
reviewed termination protocol. The first two partial tiers and the following
launch are therefore `invalid`, not baseline results. Product RED remains
unauthorized. Focused review later cleared exact runner plan `00d35376` and
authorized deterministic-v2 Task 0. That run reproduced every baseline,
probe, and mutation control, then correctly stopped at its first runtime
`invalid`: T3 closed its progress pipe after complete progress and reporter
artifacts but before `Popen.poll()` observed process exit. The current runner
cannot distinguish that final-exit handoff from an early invalid EOF. A
separately reviewed EOF-to-process-exit amendment and a fresh complete
deterministic-v2 Task 0 baseline are required before product RED.

## 1. Scope And Authorities

- Design authority:
  `docs/superpowers/specs/2026-07-28-price-collection-partial-truth-design.md`.
- Implementation authority:
  `docs/superpowers/plans/2026-07-28-price-collection-partial-truth.md`.
- Clearance branch: `codex/price-collection-truth`.
- `542776c2` remains the historical blocked-run base.
- Reviewed diagnosis closeout `e6d4b7fa` is the restart base and an ancestor
  of the rebased branch. Harness tip `2edf12e1` remains a historical
  prerequisite within that lineage.
- `5fecce6536f5d9f4a13903a6c1059e235ba15324` is the exact focused-reviewed
  restart clearance.
- `3863b3be02034b3278f58d7090dcf0bc20445fe3` is the exact focused-reviewed
  tiered-contract clearance and the Git identity used by the invalid runtime
  attempts documented in Section 8.4.
- `00d35376511b8bd28c16dd9c40415e0ddbc533ab` is the exact focused-reviewed
  deterministic-v2 plan clearance and the Git identity used by the attempts
  documented in Section 8.7.
- The rebased price-truth delta from `e6d4b7fa` is docs-only.
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

These collections and focused results reproduced again at tiered-contract
clearance `3863b3be`. They establish Steps 1-4 only. They do not repair or
replace the invalid Step 5 runtime attempts.

They reproduced a third time at deterministic-v2 clearance `00d35376`.
The four node streams, focused behavior, scanners, tool counts, and no-PG
smoke were unchanged. This establishes deterministic-v2 Steps 1-4, but the
incomplete base summary in Section 8.7 is not a baseline.

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

Product-behavior mutation work is not started. Section 8.6 records only
scratch control-runner discrimination performed while making its exact-source
plan executable. Those observations authorize no product edit and must be
reproduced from the reviewed appendix before Task 0 runtime.

Deterministic-v2 Task 0 reproduced all mandatory runner probes and six
negative controls before runtime. Their exact identities and outcomes are
recorded in Section 8.7. They validate the reviewed controls but do not
override the later runtime `invalid`.

## 7. Protected Boundaries

Task 0 Step 6 was not run after any Stop Condition 11 event. Deterministic-v2
did not launch T4-T7, deferred retries, the monolithic diagnostic, protected
boundary capture, or product RED after T3 became `invalid`. Isolated `data/`
was empty after runner cleanup. The Git worktree was clean before this blocked
evidence was authored, and no product or test path was edited.

## 8. Historical Full-Suite Diagnostics And Tier Prototype

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

### 8.3 Post-diagnosis tier-contract prototype

Diagnosis closeout `e6d4b7fa` fast-forwarded to `master` after focused review.
Merged collection-only verification reproduced:

```text
nodes: 4722
SHA-256: fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0
merge commits in 2edf12e1..e6d4b7fa: none
```

The price branch then rebased from old tip `f7458727` onto `e6d4b7fa`.
The pre-amendment rebased tip was
`813b42c7a8f8067e78cfc4d67602b097bff8cb83`.
Range-diff found the first seven price patches exactly equal; the final two
changed only where the priority map retained later diagnosis entries. The
three price authority files remained byte-identical across that rebase:

```text
spec:     aa275b448e16cc7c4708aecc47786e89f7905f2017173c342d5eb581f0068eb4
plan:     ab17beb904e0469b4a691da5c026bdd121a1bf934b6e4e13c0460c67262c851b
evidence: a6721b8e760d2ff788508739781f529c274004de458441f1b6d1279ef0932983
```

Before changing this authority, a collection-only prototype exercised the
exact Section 2.2 builder and all eight tier path lists. Collection subprocesses
set `ARKSCOPE_DISABLE_SCHEDULER=1` and used distinct temporary
`EDGAR_LOCAL_DATA_DIR` paths; those variables do not change collection
identity, and no runtime behavior result is inferred:

```text
root: /tmp/price-truth-tier-contract-prototype-20260730
builder SHA-256:
0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier-map SHA-256:
3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
mapped files: 253
tier nodes: 591 / 591 / 590 / 590 / 590 / 590 / 590 / 590
concatenated rows: 4722
unique rows: 4722
sorted-union SHA-256:
fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0
remaining `with TestClient(...)` files: 5, all mapped
  T1: tests/test_portfolio_capture_routes.py
  T2: tests/test_signal_factors_p1.py
  T5: tests/test_portfolio_activity_routes.py
  T6: tests/test_api.py, tests/test_events.py
```

`cmp` proved the tier union byte-identical to the canonical collection. This
is a mechanism prototype, not a runtime baseline: no tier behavior run,
non-passing set, Task 0 clearance, or product authorization is inferred.

The canonical stream contains 11 valid node IDs with embedded spaces, proving
that the historical whitespace-token parser is not safe for this protocol.
The stdlib-only scratch reporter at SHA-256
`09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928`
was therefore probed separately:

- a real 11-node `tests/test_env_unquote.py` run exited 0, and reporter
  collected/seen sets both matched the canonical IDs exactly;
- a one-node temporary failing parametrization with ID `id with spaces`
  exited 1 and preserved that full ID in collected, seen, and non-passing
  arrays; and
- the temporary probe is outside the repository and adds no planned node.

A second full collection rebuilt the planned `env -i` runtime boundary with
isolated home, temp, lock, five ArkScope-store, and EDGAR paths. It again
produced exactly `4722` nodes and SHA-256 `fcdb1b7d...`; the isolated worktree
`data/` remained empty. This proves collection identity under the credential-
and database-stripping environment but is still not a runtime behavior
baseline.

The exact `env -i` + `setsid` + reporter command was then exercised as a
contract probe. The real 11-node file completed with `11 passed`, shell and
reporter exit 0, identical collected/seen sets, and an empty non-passing set;
its recorded process had equal PID, PGID, and SID. The temporary
space-containing node completed with exit 1, a terminal summary, and the same
full node ID in collected, seen, and non-passing arrays. Neither probe touched
isolated worktree `data/`, and neither is admitted as a Task 0 baseline.

### 8.4 Tiered Task 0 invalid-runner blocker

Focused review of contract tip `3863b3be` returned GREEN with zero findings
and authorized tiered Task 0. Steps 1-4 then reproduced the exact collections
and focused gates in Section 2. Runtime scratch identity was:

```text
root: /tmp/price-truth-tier-v1
builder SHA-256: 0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
reporter SHA-256: 09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
tier-map SHA-256: 3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
runtime-fingerprint file SHA-256: d754856c368ae05668c6df075410e4e706cd4d1bab2e5b39b7740831db564b96
pip-freeze SHA-256: cf3e80661ab59e43b291b7ed037159aee90248fff7c1d4e38bf79de264b6eec8
```

Collection-only reconstruction again proved all `4722` unique nodes with tier
loads `591/591/590/590/590/590/590/590` and a sorted union byte-identical to
`fcdb1b7d...`. The reporter also preserved all 11 space-containing node IDs.

The runtime attempts are not admissible:

| Tier | Observed partial boundary | Transcript | Classification |
|---|---|---|---|
| T0 | `test_route_rejects_unreviewed_interval_with_typed_422`; dump mtime `09:15:35 +08:00`; process ended `09:36:33` | 661 lines / 67097 bytes; `efe83d6a...` | `invalid` |
| T1 | `test_scheduler_start_stop`; dump mtime `09:41:17 +08:00`; process ended `21:56:59` | 321 lines / 30414 bytes; `7fafeaf5...` | `invalid` |
| T2 | control shell sampled child PID/PGID/SID as `145/1/1` before `setsid` completed; no runtime output | 0 bytes; `e3b0c442...` | `invalid` |

For T0 and T1, faulthandler emitted the required 120-second dump, but the
orchestrator did not enforce the 150-second no-progress termination boundary.
The processes remained alive for materially longer than the reviewed bound.
Both were initially described operationally as stalls, but the closed outcome
table requires runner/protocol failures to be classified as `invalid`.
Neither transcript has a terminal pytest summary or reporter JSON, so no
partial node result is normalized or banked.

T2 exposed a separate control-wrapper race. The reviewed command launches
pytest through `setsid`; the operator added an immediate PID/PGID/SID
assertion outside the reviewed command shape. That assertion sampled the
background process before `setsid` completed, exited the shell, and left no
terminal pytest result. An anchored host-namespace check confirmed that no T2
pytest process survived. The extra assertion was not an authorized contract
change and cannot be repaired by silently reusing the attempt path.

The first invalid attempt required an immediate stop. T1 and T2 were started
only because T0 was incorrectly classified before artifact sealing; they are
retained as control-plane evidence, not as additional authorized tier results.
T3-T7, deferred retries, the diagnostic monolithic run, protected Step 6, and
product RED were not started.

All 46 selected runtime-control artifacts are listed in:

```text
/tmp/price-truth-tier-v1/task0-invalid-manifest.sha256
lines/bytes: 46 / 5407
manifest SHA-256:
1f3f5c6137fbd57da56e8cc7bd1dcf4e55e64fad8fe8896b419aab8882288c98
```

After the stop, isolated worktree `data/` was empty, no matching tier pytest
process remained, and the branch had no tracked or untracked change before
this evidence edit. The main worktree retained only the two protected
untracked drafts, with SHA-256 values `4921194a...` and `79d4eac9...`; neither
was edited, staged, moved, deleted, or cited as authority.

This blocker identifies an execution-control defect, not a price product
defect and not a test verdict. Re-running the same manually supervised wrapper
is not authorized.

### 8.5 Proposed deterministic runner amendment

Independent review of blocker `fa42d44a` returned GREEN with zero findings and
confirmed all three `invalid` classifications. The user selected the
single-module structured-progress design for the replacement runner.

Design Section 13 now requires one SHA-pinned Python file to act as both the
parent controller and a pytest progress plugin using
`pytest_runtest_logstart`/`pytest_runtest_logfinish`. Structured events travel
over one inherited pipe and are persisted to `progress.jsonl`;
the unchanged final reporter remains the sole node-accounting authority.
`Popen(start_new_session=True)` gives the controller stable process-group
ownership, while an in-process monotonic state machine owns the pre-first-node,
active-node, and final-teardown deadlines.

The breach distinction is explicit:

```text
150-second breach + 120-second per-item dump -> unresolved_stall
150-second breach without that dump         -> invalid
```

The runner itself performs `SIGINT`, the complete 10-second grace, optional
`SIGKILL`, atomic attempt recording, and first-invalid refusal. Four mandatory
pre-runtime probes cover natural pass, SIGINT termination, SIGKILL fallback,
and unchanged collect-only identity. The environment allowlist, final
reporter, four outcomes, immutable tier map, banking tuple, one deferred
retry, and base/tip admission contract remain unchanged.

Focused review cleared this design at
`1d08a9f30a87066ea0a2e3b3274a22210cdfa57d` with zero findings. That
clearance authorizes the exact-source implementation plan, not Task 0 runtime
or product RED.

### 8.6 Exact-source plan-construction probes

The plan appendix was built and exercised in scratch before being proposed as
runtime authority:

```text
source root: /tmp/price-truth-runner-plan-final3-20260730
runner lines/bytes: 2140 / 77040
runner SHA-256:
35cda547ac8b1afaba1231d56cb04d703a284cdd81de978397ce7887ac51339e
reporter SHA-256:
09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder SHA-256:
0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
probe summary SHA-256:
47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48
```

The appendix extraction command reproduced the source byte-for-byte at
`/tmp/runner-extracted-final3-check.py`; both files had the runner hash
above and compiled under the pinned interpreter. The dual-role FD check lives
in `pytest_configure`, so module-mode preflight succeeded without
`PRICE_TRUTH_PROGRESS_FD`.

The pristine probe suite completed with every check true:

```text
natural pass:       complete_pass; progress 2; EOF; no signal
SIGINT arm:         unresolved_stall; dump true; SIGINT; killed false
SIGKILL arm:        unresolved_stall; dump true; SIGINT then SIGKILL;
                    descendant process group gone
collection identity: control == plugin == 1 exact node; manifests share
                     SHA-256 85e427423e6a...
FD fail-closed:     missing and garbled values both exit 3 in pytest_configure
```

The short constants were probe-only `2/3/1`; the runner source retains
immutable runtime `120/150/10`.

A final authority check extracted the appendix again, without using the
construction copy, into
`/tmp/price-truth-runner-plan-final4-20260730`. It reproduced the exact
`2140/77040/35cda547...` source identity, compiled, and reran all five probes
in `13.48` seconds. The summary again had SHA-256 `47564c644...`; both
collection manifests and `probe.nodes` had SHA-256 `85e427423...`, and the
two malformed-FD children again exited `3` in `pytest_configure`.

Six load-bearing controls were then exercised in separate scratch roots. Their
mutated source bytes and records remain in those construction roots; the plan
now supplies the exact rerun diffs and requires each Task 0 mutation root to
retain `mutation.diff`:

| Mutation | Reconstructed observation |
|---|---|
| M1 delay parent event handling beyond the already expired window | fast-pass became `invalid`, `deadline_breach_without_dump`; the late event could not revive the window |
| M2 move dump beyond deadline | both sleeps became `invalid`, `deadline_breach_without_dump`, no false stall |
| M3 ignore SIGINT | interruptible arm retained stall truth but recorded `killed=true` and SIGKILL |
| M4 self-alter runner between child launches | the first child record remained; the second child was refused before launch by renewed preflight identity checking |
| M5 missing/garbled FD | both child pytest processes exited `3` in `pytest_configure`; module mode remained usable |
| M6 seed prior invalid record | `run-side` wrote incomplete summary and created no T1-or-later directory |

M5 was repeated from a dedicated pristine root at
`/tmp/price-truth-runner-final3-m5-20260730`; both child pytest arms exited `3`,
the parent suite remained green, and its summary reproduced
`47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48`.

A separate valid eight-node sequencer probe mapped eight distinct safe
repository test files to eight slots, passed the same partition verifier, and
produced a complete side summary with eight selected first attempts at
`/tmp/price-truth-runner-final3-sequence-20260730`. Its summary SHA-256 is
`562fd1e646829bd4babb41de983477222cae1f9e76aa1589eddd04ec89340f39`.
It is not the
4,722-node collection proof or a Task 0 baseline; it proves runner sequencing,
per-attempt admission, aggregation, and atomic completion mechanics.

Two additional controls target state that ordinary one-tier probes cannot
exercise:

- `/tmp/price-truth-retry-probe-artifacts-20260730` began with simultaneous
  T0/T1 stalls and proved that both tiers received their one deferred `a2`
  retry before the side closed incomplete.
- `/tmp/price-truth-bank-probe-artifacts-20260730` first completed all eight
  tiers, then had one banked non-passing artifact altered. Reuse failed closed
  with `banked non-passing artifact changed`, wrote an incomplete summary, and
  named `base-T0-a1` as the invalid attempt.

These are plan-construction observations. PID/timestamp/record hashes are not
promoted to acceptance constants. Section 8.7 records the independently
cleared Task 0 run that extracted the appendix into fresh
`/tmp/price-truth-tier-v2`, regenerated preflight, and repeated the probes and
mutations before the first real tier launch.

### 8.7 Deterministic-v2 Task 0 EOF/exit blocker

Focused review of plan tip `00d35376` returned GREEN with zero findings and
authorized tiered Task 0 v2. Steps 1-4 reproduced every value in Section 2.
The clearance transcript is:

```text
path: /tmp/price-truth-task0-clearance-20260731.txt
SHA-256: e90c0f5313995a9b5659d05444fedf69be5b7e115359a3ebade6c72556db5c38
branch/head: codex/price-collection-truth / 00d35376511b8bd28c16dd9c40415e0ddbc533ab
```

The focused gates created an ignored 143,360-byte
`data/profile_state.db` with SHA-256 `fcfbadad...` in the isolated worktree.
The runner refused before creating a child attempt because `data/` was not
empty. The file was proven SQLite-integrity `ok` and moved reversibly, with
its zero-byte WAL and 32,768-byte SHM companions, to:

```text
/tmp/price-truth-tier-v2/task0-preprobe-generated-data/
```

The integrity read created the WAL/SHM companions after the main file was
inspected, so all three were archived before retrying. `data/` was then empty.
This was a pre-launch setup rejection, not a runtime attempt or `invalid`.

Exact v2 control identities were:

```text
runner:         35cda547ac8b1afaba1231d56cb04d703a284cdd81de978397ce7887ac51339e
reporter:       09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928
builder:        0f0421f86f46265427914bce7bbede694beaa8d04b5d3b2ea9562d27cd7c8d9c
tier map:       3d7adb7e1db7b92b25b3ae83fe56ec182c6b070802cd45d95e091c673115994a
base preflight: 5c8484deeb9188228e4bc730e283d3f67f00a91fd5cf3bcbb32558a1079ac320
probe summary:  47564c644c95e54007d67e4b08ddaeb35ed8370f858b3f004f00f54ef9e1ad48
```

The mandatory probe suite returned all five checks true. M1-M4 retained exact
mutation diffs with SHA-256 values
`6a5d13b5.../7f463def.../bc7e8457.../27a064f7...`; each owning assertion
matched. M5 reproduced the same all-true probe summary while both malformed-FD
children failed in `pytest_configure`. M6 seeded one prior invalid record,
returned nonzero, wrote `complete=false` with
`invalid_attempt=base-T0-a1`, and created no T1 attempt. The pristine runner
and fixture hashes were rechecked before runtime.

Runtime then produced:

| Tier | Outcome | Last active boundary | Progress | Duration / cleanup |
|---|---|---|---:|---|
| T0/a1 | `unresolved_stall` | `test_route_rejects_unreviewed_interval_with_typed_422` | 1173 | 172.969s; dump; SIGINT then SIGKILL; group gone |
| T1/a1 | `unresolved_stall` | `test_scheduler_start_stop` | 549 | 158.503s; dump; SIGINT; group gone |
| T2/a1 | `unresolved_stall` | `test_anthropic_calibration_raises_structured_refusal_before_text_extraction` | 769 | 164.864s; dump; SIGINT; group gone |
| T3/a1 | `invalid` / `pipe_eof_while_child_running` | post-last-progress, no active node | 1180 | 20.334s; no dump; SIGINT; group returned 0 and was gone |

The first three classifications are valid unresolved stalls. T3 is the first
invalid and closes the side. Its decisive artifacts show:

```text
expected/collected/seen/non-passing: 590 / 590 / 590 / 0
progress events: 1180 = 2 * 590
report exitstatus: 0
terminal summary: 580 passed, 10 skipped, 9 warnings in 18.86s
active node at EOF: null
deadline phase: post_last_progress
record SHA-256: d89eb6a4ffaefca68d65f9969d2452a08af194ab786f14b7d81ce3f8db860385
transcript SHA-256: 278b9827345f1ead84eca65304b38b1ba856a4627ec3587b014ba26cf1bef831
progress SHA-256: b1110e5ad9747f44e8d1582be4b571931ce7828f53e24f6edeed73eadb1aa39a
report SHA-256: 60190dc3405b48e1eb74199eeacbf2fcf49377f7aac7de96175087221c53c1b8
```

The progress writer closed after all node events and the reporter had already
atomically written a complete result, while the parent still observed
`process.poll() is None`. The reviewed runner therefore applied its closed
EOF rule, marked the attempt invalid, and sent SIGINT. The process group
reported return code `0` and disappeared about 10ms later. These facts do not
permit retroactively calling T3 a natural pass because the runner injected a
signal. They do prove that `EOF while child is running` conflates an
observable final-exit handoff with malformed early EOF.

The atomic side summary has SHA-256 `c7a252a3...`, `complete=false`,
`invalid_attempt=base-T3-a1`, unresolved tiers `[0,1,2]`, and no selected
attempts. T4-T7, all deferred retries, the monolithic diagnostic, protected
Step 6, and product RED were not started.

All pre-manifest non-`__pycache__` files in the v2 root are listed in:

```text
/tmp/price-truth-tier-v2/task0-v2-invalid-manifest.sha256
lines/bytes: 1154 / 163262
manifest SHA-256:
a202dfd4fb160c15ca4a0e517dcc79641dec1ebfd25dba7b1439a4acc63fbb8e
```

After the stop, isolated `data/` was empty and the runner records
`cleanup_complete=true` for every runtime attempt. The branch was clean before
this evidence edit. The two protected main-worktree drafts retained SHA-256
values `4921194a...` and `79d4eac9...`; neither was edited, staged, moved,
deleted, or used as authority.

Restarting the same v2 runner is unauthorized. The next gate is a separately
reviewed amendment to Design Section 13.6 and the exact runner. It must retain
immediate invalidation for partial buffers, unbalanced progress, incomplete
node counts, malformed reporter evidence, or a live group beyond a fixed
bound. It must separately decide a bounded post-EOF exit handshake for the
fully complete shape above, without using transcript text for node accounting.
The amendment needs a deterministic EOF-before-`poll()` probe plus negative
controls proving that early EOF and leaked descendants still fail closed.
No grace duration or exact implementation is authorized by this blocker.

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
therefore did its intended job, but the repository could not produce the
complete same-environment baseline required by the then-current monolithic
plan. Silently excluding a node, accepting partial output, running protected
Step 6, or starting product RED remained prohibited pending a separately
reviewed resolution.

The separately reviewed diagnosis later selected
`V6 ambient_or_machine_state_dominates`; it authorized no source seam and
transferred the unresolved behavior to `EIR-005`. The user then selected a
tiered full-collection protocol instead of waiting for a clean monolithic
window. The contract preserves Stop Condition 11 at tier granularity and
states explicitly that fresh-process tiered results are not directly
comparable with historical monolithic runs. Focused review cleared that
amendment at `3863b3be`; the subsequent runtime controller failed its reviewed
termination and process-identity protocol as recorded in Section 8.4. Focused
review then cleared deterministic control-runner design at `1d08a9f3` and its
exact source plan at `00d35376`. Deterministic-v2 Task 0 then stopped at the
Section 8.7 EOF/exit invalid. A reviewed EOF-to-process-exit amendment, not
product implementation or an unchanged retry, is the next gate.

## 10. Integration And Read-Only Release Observation

The harness and diagnosis prerequisites are merged; price-truth product
integration is not started. The tiered contract and deterministic runner
design/plan are reviewed, but deterministic-v2 Task 0 is incomplete and has
no admitted base side. Provider calls, production writes, repair, browser
work, product RED, and release observation remain unauthorized.
