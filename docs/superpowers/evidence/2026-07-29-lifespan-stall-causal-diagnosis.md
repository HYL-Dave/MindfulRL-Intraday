# Full-Suite Lifespan Stall Causal Diagnosis Evidence

> **Status:** DIAGNOSIS REVIEW-READY - INDEPENDENT REVIEW NEXT
>
> **Observed:** 2026-07-29 Asia/Taipei
>
> **Verdict:** `V6 ambient_or_machine_state_dominates`
>
> **Behavior base:** `2edf12e11a8ff9299a9b65b900309c8ed218b717`
>
> **Blocked price tip:** `f7458727b8c7828e9372be29e7698b986e1db757`

## 1. Status And Authorities

This packet records a verdict-only diagnosis. It authorizes no product fix,
test-harness change, dependency change, merge, or restart of the blocked
price-collection line.

| Authority | Identity |
|---|---|
| Approved design | `222636a752d8b64b132ca8d270189ce46c1fe071` |
| Independently reviewed experiment plan | `d3fa6e99542d97823fb6ad4c4ea4009d51b59647` |
| Plan file before status-only closeout | `e162f57831bde18a8fd2ca518b390055c8869e45b313953f9b8f5b625821b2c1` |
| Behavior base | `2edf12e11a8ff9299a9b65b900309c8ed218b717` |
| Frozen price branch | `f7458727b8c7828e9372be29e7698b986e1db757` |
| Task 0 packet | `5388e9b5450441bfa31731fd7d12595f40eebde6bbd92650442eef9b6e72d06b` |

The experiment ran only in
`/tmp/arkscope-lifespan-stall-diagnosis`. The immutable artifact root is
`/tmp/arkscope-lifespan-stall-diagnosis-20260729T143632Z`.

## 2. Environment

Task 0 recorded this package fingerprint before any behavioral trial:

```text
edgartools=5.0.2
httpxthrottlecache=0.3.0
pyrate-limiter=3.9.0
httpx=0.28.1
starlette=0.47.2
anyio=4.9.0
pytest=8.4.1
```

Every trial used a fresh pytest process and a unique temporary root. Scheduler
work was disabled, mutable ArkScope paths were redirected to that root,
`EDGAR_LOCAL_DATA_DIR` was isolated, and the diagnosis worktree's `data/`
directory was empty before and after every trial.

No trial called a provider, Gateway, external HTTP/network endpoint, browser,
scheduler, or production database. The target still exercised its expected
in-process `/status` request through `TestClient`. No operator-intended package
or machine reconfiguration occurred between cells; uncontrolled ambient state
remains part of the verdict.

## 3. Task 0 Identity And Boundaries

The frozen Task 0 packet is:

```text
/tmp/arkscope-lifespan-stall-task0-20260729T143555Z.md
SHA-256 5388e9b5450441bfa31731fd7d12595f40eebde6bbd92650442eef9b6e72d06b
```

It established:

| Control | Result |
|---|---|
| Diagnosis HEAD | exact reviewed plan tip `d3fa6e99...` |
| Behavior-base ancestry | true |
| Product/test/dependency diff from behavior base | empty |
| Backend collection | `4722 / fcdb1b7dc197c35d43684e7dde846ea82dc975ca6bb688162e88c5f312d43ff0` |
| Price worktree | clean, exact `f7458727...` |
| Main tracked state | clean |
| Main untracked state | exact two protected drafts |
| Diagnosis `data/` files or symlinks | `0` |
| Existing pytest/controller process | none |

Phase 1 collection identities also reproduced exactly:

| Cell | Nodes | SHA-256 |
|---|---:|---|
| `A0B0` | 1 | `4e385828ffa504640d8d50a65c2602fc9c6e06530d23127292bdab198ce3d71f` |
| `A0B1` | 2 | `c5743e0d97cedb58531de047f90413f6284e33306d203df08620b4e4c2959cbc` |
| `A1B0` | 7 | `a3af106460307042ae13af1a2b6759c34e25788102cee94b3115346a8afcb484` |
| `A1B1` | 8 | `7d7b4d75fee81eb9305cb3651c7940e7ca87afe3539a198d5ec45eefea547644` |

## 4. Pinned Experiment

The reviewed appendices were extracted without modification:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `scratch/diagnosis_controller.py` | 20,056 | `d069de2236851e89ac6271e24589ca00ab328ef35c338a5cd092be1970ddd200` |
| `scratch/arkscope_edgar_import_probe.py` | 1,235 | `4103a8ed21309b846e1a9ac7bfb249759ce4e3bc5638eb5495e95f6b64e35c17` |
| `scratch/independent_verifier.py` | 8,694 | `645a0c528c3f0693d4bfbdf327d5871ce681f5fbee90f5e3e26884e823d4e40c` |

The deterministic schedules were prepared before results were read:

| Phase | Seed | Blocks prepared | SHA-256 |
|---|---:|---:|---|
| Phase 1 | `20260729` | 20 | `cf2c9fef4c4a546205587ee0c4aa0692208ab538ea4951ec8ef66a1b7df0d419` |
| Phase 2 | `20260730` | 20 | `892ea2a562ff0691ec01738b73f2feec552da7ea7ee6a835263b5713815c5c4d` |

Across each 20-block schedule, every cell occupies every ordinal position
exactly five times. The import-only preflight retained identical collection
identity in E0 and E1. E0 produced a `0 -> 0` leaker-thread transition; E1
produced `0 -> 1` with daemon thread name `PyrateLimiter's Leaker`.

Preflight identity:

```text
preflight.json
SHA-256 a62979033d70d1a2ad79b8c0d305bec02a8f64e83dbf870249161fb78a50a816
```

## 5. Phase 1

Phase 1 varied SEC-test collection (`A`) and execution of the immediately
preceding real-app mount node (`B`). The pre-registered block-10 truncation
condition was met.

| Cell | SEC collected | Mount executed | Matching stalls | Other outcomes | Invalid |
|---|---:|---:|---:|---:|---:|
| `A0B0` | no | no | 10 | 0 | 0 |
| `A0B1` | no | yes | 10 | 0 | 0 |
| `A1B0` | yes | no | 10 | 0 | 0 |
| `A1B1` | yes | yes | 10 | 0 | 0 |

All 40 records had the six required signature flags:

```text
dump=true
future_result=true
portal=true
spawn=true
target=true
testclient=true
```

Durations ranged from `80.03244443098083` to `80.0328846280463` seconds.
Each trial preserved its faulthandler dump and was then closed by the reviewed
outer-controller escalation. There were no boundary violations, replacements,
or worktree data files.

Controller flags:

```text
ubiquitous=true
first_factor_not_necessary=true
mount_not_necessary=true
main_first_factor=false
main_mount_factor=false
interaction=false
early_stop_eligible=true
```

This phase establishes that neither SEC-test collection nor the mount
predecessor was necessary for the observed matching stall in this sample. It
does not identify a mechanism, and factor A is too broad to make an `edgar`
claim.

Phase 1 artifacts:

| Artifact | SHA-256 |
|---|---|
| `summary-phase1-b10.json` | `16c1e179361d9a1827d643ee01faa7e13fb135ad40730d0b49bf0b2e056a55dc` |
| `reconstruction-phase1-b10.json` | `47e7bab32c3b5dbbf77876681ccdb93e4789f59720c9622ee7f05ce57583d4de` |
| `manifest-phase1-b10.sha256` | `79ab780f9d2ae650929d98e870484a77d4f37e05a5831d3087e50b9f5ee6d274` |

## 6. Phase 2

Mandatory Phase 2 replaced SEC-test collection with the identical-byte plugin.
`E0` loaded the plugin without importing `edgar`; `E1` imported `edgar`.
Factor B remained the mount predecessor. The pre-registered block-10
truncation condition was again met.

| Cell | `edgar` imported | Mount executed | Matching stalls | Other outcomes | Invalid |
|---|---:|---:|---:|---:|---:|
| `E0B0` | no | no | 10 | 0 | 0 |
| `E0B1` | no | yes | 10 | 0 | 0 |
| `E1B0` | yes | no | 10 | 0 | 0 |
| `E1B1` | yes | yes | 10 | 0 | 0 |

All 40 snapshots were valid. The 20 E0 records were exactly `0 -> 0`; the 20
E1 records were exactly `0 -> 1`. Every trial nevertheless produced the same
complete matching portal signature. Durations ranged from
`80.0325417839922` to `80.03320529707707` seconds. There were no boundary
violations, replacements, or worktree data files.

Phase 2 produced the same six controller flags as Phase 1, including
`ubiquitous=true`, both factor-not-necessary flags, and no main effect or
interaction.

The plugin import occurs earlier than Phase 1's collection-time import. This is
a timing qualifier, not evidence for `edgar`: matching stalls occurred in
every import-absent and import-present cell. The result establishes that the
`edgar` import and its observed leaker-thread creation were not necessary for
these stalls. It does not prove that either has zero effect in every process
state.

Phase 2 artifacts:

| Artifact | SHA-256 |
|---|---|
| `summary-phase2-b10.json` | `60c55eaca02b233b59651aa2718d3305bfb09e44c1293de91a9fc24faf48f948` |
| `reconstruction-phase2-b10.json` | `d2c2bb02e4c342e7e52ba7a9c2e6e8da7c0d7b34c1f8c4a2444d6ada0721846a` |
| `manifest-phase2-b10.sha256` | `e7aac43cfb49f6ba780e28b468f9a91324875a8099633ec65ceb7f7b73279b7c` |

## 7. Independent Reconstruction

The pinned verifier does not import the controller. It reread every
`record.json` and recomputed scheduled/actual cell order, valid and invalid
attempts, outcomes, signature completeness, thread transitions, early-stop
flags, schedule SHA, and terminal block.

| Check | Controller | Independent verifier |
|---|---|---|
| Phase 1 terminal block | 10 | 10 |
| Phase 1 cell totals | 10 matching stalls in each cell | exact match |
| Phase 1 invalid attempts | 0 | 0 |
| Phase 1 flags | ubiquitous; no main effects or interaction | exact match |
| Phase 2 terminal block | 10 | 10 |
| Phase 2 cell totals | 10 matching stalls in each cell | exact match |
| Phase 2 invalid attempts | 0 | 0 |
| Phase 2 transitions | E0 `0 -> 0` x20; E1 `0 -> 1` x20 | exact match |
| Phase 2 flags | ubiquitous; no main effects or interaction | exact match |

Both verifier files report `ok=true` with an empty `errors` array.

## 8. Verdict And Qualifiers

The first applicable reviewed precedence rule is rule 7:

```text
V6 ambient_or_machine_state_dominates
```

Both matrices were ubiquitous and non-isolating. Matching stalls occurred
with SEC collection absent, mount predecessor absent, and direct `edgar`
import absent. The controlled factors therefore do not isolate the event under
this harness.

The verdict means a condition common to all tested cells and outside the
manipulated factors dominates the observed binary outcome. It does not identify
that condition. In particular, it does not prove:

- that the leaker thread causes or cannot contribute to a stall;
- that `edgar`, SEC collection, or the mount has zero effect in other process
  states;
- that Starlette, AnyIO, pytest, the machine, or ArkScope lifespan is the root
  cause;
- that the full suite is fixed; or
- that a test or product seam should now be changed.

The all-stall result is stronger than a finite no-stall sample for establishing
non-necessity of the tested factors, but it provides no successful cell from
which to estimate a recovery condition.

## 9. Import Sites And Non-Binding Seam Inventory

Grounding was refreshed after the experiment:

1. `tests/test_sec_filings.py:19` imports `SECFilingsClient` at module scope.
   Its file skip does not prevent collection-time import.
2. `tests/test_sec_user_agent.py:44` imports `data_sources.sec_filings` inside
   an active user-agent contract test.
3. `data_sources/sec_filings.py:10` is a docstring example, not an executable
   import site.
4. No active `src/` path imports `data_sources.sec_filings` or constructs
   `SECFilingsClient`.
5. Live DAL-backed SEC facts, agent financials, and insider tools are separate
   from this client.

The design's candidate seams remain inventory only: product lazy import,
moving either test import, supported dependency shutdown, dependency upgrade,
retiring the unconsumed client, or building an app-owned SEC service. The V6
result selects none of them.

A machine-state observer for the AnyIO loop, wakeup socket, selector
registrations, queues, and tasks is a possible separate diagnostic amendment.
It was not run and is not authorized by this packet.

## 10. Product And User Implications

The user-facing concern is not merely a noisy test. The non-terminating gate
blocks a live price-integrity fix, while indiscriminately removing lifespan
tests could hide startup, migration, scheduler, portfolio, or shutdown defects
that matter to the desktop experience.

This experiment narrows the decision:

- Do not treat SEC capability retirement or an `edgar` lazy import as the
  established incident fix.
- Do not convert the remaining real-lifespan families merely to make pytest
  terminate.
- Do not resume price-truth Task 0 from this unreviewed diagnosis packet.
- Preserve the separate product decision about whether parsed 10-K/10-Q
  content belongs in the workbench.

Any later fix proposal must answer the approved four-question gate:

1. **Causal seam:** What did controlled evidence implicate or rule out?
2. **User value:** Does the workbench need parsed filings, structured SEC
   facts/metadata only, or neither?
3. **Coverage integrity:** Which real startup/shutdown behavior remains tested?
4. **Ownership:** If product code remains, who owns the side effect and its
   revalidation trigger?

The next gate is independent reconstruction and review of this diagnosis.
Only after that review may a separate decision authorize more diagnosis or a
fix.

## 11. Deviations And Untested Questions

There were no invalid attempts, replacement attempts, boundary violations, or
trial-protocol deviations. Stopping after 10 complete blocks per phase was the
pre-registered early-truncation path, not a reduced post-hoc budget.

The experiment did not test:

- a bare pyrate-limiter/leaker treatment independent of `edgar`;
- the conditional machine-state observer;
- another package version, dependency upgrade, or supported shutdown hook;
- a complete full-suite run after a fix;
- production app startup, provider, Gateway, network, browser, or database
  behavior;
- whether parsed filing content should remain or be retired;
- more than 10 trials per cell after ubiquitous outcomes made the reviewed
  next verdict deterministic; or
- the mechanism that prevented the portal spawn handshake.

One evidence-only rehash command initially used prototype filenames that do
not exist in the immutable root. It failed without changing an artifact and
was rerun against the pinned `scratch/` and `schedule-*` paths. Both phase
manifests and the final complete manifest subsequently verified clean.

## 12. Artifact Manifest And Read-Only Reproduction

The artifact root contains 345 frozen files plus its manifest:

```text
/tmp/arkscope-lifespan-stall-diagnosis-20260729T143632Z
manifest-complete.sha256 entries: 345
manifest-complete.sha256 SHA-256:
9725dd4ff80bc0546297312d7016abb2d2ec767c35b8722a67f88862e49e5d05
```

The phase manifests contain 140 Phase 1 and 180 Phase 2 artifact entries.
The complete root occupies approximately 4.5 MiB.

Read-only integrity and reconstruction:

```bash
DIAG_ROOT=/tmp/arkscope-lifespan-stall-diagnosis-20260729T143632Z
cd "$DIAG_ROOT"

sha256sum -c --quiet manifest-complete.sha256
sha256sum -c --quiet manifest-phase1-b10.sha256
sha256sum -c --quiet manifest-phase2-b10.sha256

python scratch/independent_verifier.py \
  --root "$DIAG_ROOT" --phase phase1 --through-block 10
python scratch/independent_verifier.py \
  --root "$DIAG_ROOT" --phase phase2 --through-block 10

jq . summary-phase1-b10.json
jq . reconstruction-phase1-b10.json
jq . summary-phase2-b10.json
jq . reconstruction-phase2-b10.json
```

Boundary checks that do not execute product behavior:

```bash
git -C /tmp/arkscope-lifespan-stall-diagnosis \
  diff --quiet 2edf12e11a8ff9299a9b65b900309c8ed218b717 -- \
  src data_sources tests requirements.txt requirements-dev.txt pyproject.toml

git -C /tmp/arkscope-price-collection-truth rev-parse HEAD
git -C /tmp/arkscope-price-collection-truth status --short
find /tmp/arkscope-lifespan-stall-diagnosis/data -type f -o -type l
```

The independent review must reconstruct from raw `record.json` files and not
accept this prose as the primary evidence.
