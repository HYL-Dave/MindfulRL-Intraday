# Settings Navigation and Warm Cache Evidence

> **Status:** TASK 4 COMPLETE; TASK 5 BATCH EXECUTION IN PROGRESS
>
> **Date:** 2026-08-09
>
> **Reviewed plan tip:** `5755ed54ad1f64c06444899f4d4bf458cedd8610`
>
> **Product grounding base:** `3d18e9c0ea54d99fc4824b7919d74a4c3a38502b`
>
> **Task 0 artifact root:**
> `/tmp/settings-navigation-warm-cache-task0-5755ed54`
>
> **Artifact manifest:** `72` payload entries / SHA-256
> `d0da19c1d76153f3ece27281f8948d5a332ce246602a6123c139604923cb19fe`

Task 0 was grounding and evidence only. It changed no product, test, backend,
API, cache, provider, credential, scheduler, profile, or production-data byte.
At Task 0 close, Tasks 1-7 remained unstarted and unauthorized pending review.

## 1. Authority and execution boundary

The isolated worktree was clean on branch
`codex/settings-navigation-warm-cache` at reviewed plan tip `5755ed54`. Its
parent was reviewed design tip `dbd92b86`; its merge base with product base
`3d18e9c0` was exactly `3d18e9c0`. The unlocked main worktree was clean and
remained at that same product base.

The plan-tip diff from product base contained only:

```text
docs/design/PROJECT_PRIORITY_MAP.md
docs/superpowers/plans/2026-08-09-settings-navigation-warm-cache.md
docs/superpowers/specs/2026-08-09-settings-navigation-warm-cache-design.md
```

The independently reviewed design bytes at `dbd92b86` reproduced as 348
lines / 17,627 bytes / SHA-256
`0b7ba0568164fd1d49c75be385e6ccc9f6252fd4239c659d6a4ff484b579ac12`.

Pinned execution assets reproduced as follows:

| Asset | Identity |
|---|---|
| Node | `v22.14.0` |
| Vite | `5.4.21` |
| Vitest | `4.1.8` |
| Python Playwright | `1.58.0` |
| Google Chrome | `150.0.7871.128` |
| `package-lock.json` | `5322cb03099b873066b7572c02db68a427ddc7509fdc9850cf9d8d3948a19f2c` |
| pinned `node_modules/.package-lock.json` | `4dd5182f8111b54dd608344c45513f3b310f39321a3cc9832b60cefc2fa241ff` |
| decoded-list normalizer | `955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac` |
| deterministic pytest reporter | `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928` |

No dependency installation or browser substitution occurred.

## 2. Frontend collection ledger

The pinned normalizer JSON-decoded the full Vitest list and produced the
canonical sorted `relative_file<TAB>full_test_name` stream:

| Stream | Files | Nodes | SHA-256 |
|---|---:|---:|---|
| full base | 97 | 1,084 | `f0e5ecda1371f0559c1cc92af367b7e32daa91663ef2f316ed67e23129ee9637` |
| focused base | 14 | 182 | `1c56ecf00a6d89d2d51191bcbd95946a8dd00c039f26c3c1d3d0bb979878c002` |

The focused stream is the exact fourteen-file projection of the decoded full
stream through `frontend-focused-files.txt`; it is not a prose or raw-JSON
parse.

The plan's four input ledgers reproduced byte-for-byte:

| Ledger | SHA-256 |
|---|---|
| additions TSV | `0437cf44d53244a8baea5151d14ab19bc626a579154807ed6df89d406b464a7b` |
| additions node stream | `e1d32f68d625316cdc658f5c3a6763f394c34da561314d69f9276a6e374d7a14` |
| removal TSV | `551f6620590eb90b8fccdeaf536e5a60f999736dc56e7b8062025b4ea64f38e7` |
| removal node stream | `0042e6192d9d8263ce4ca2767fb8d8da9504b10df17f40d82ff4c7d9980ff9ed` |

Mechanical invariants were `40` unique additions, zero additions already in
base, one removal present exactly once, and fourteen unique focused files.
Applying each reviewed phase produced every predeclared identity:

| Phase | Full nodes / SHA-256 | Focused nodes / SHA-256 |
|---:|---|---|
| 1 | 1,101 / `6f77e16694bc7994ea62a0e51ec13a7ee79fc9f03851da7a55519b04bcbc801f` | 199 / `543ebdffdf922d73045fa42c1e19ae2aba5cf598e8804c359ec0c868ce27fee3` |
| 2 | 1,106 / `10965b1c8e5a51cbf5d38950b0db8410faef1a528e6dc2856e391267019a37bc` | 204 / `e34c217edb518485ebacbbe382a44f47f36e536c9a15f13125add0664910a085` |
| 3 | 1,113 / `eefdbdaa10c83786cdbf9054b76dcf0bae822bafb129aece422958d3e20f0ee8` | 211 / `d74255067e3ca4531c6a2f8590156f20175613c0fdcffd453ab448881ec1bac3` |
| 4 | 1,116 / `09d31fa1bd22d3b0519c8dce2c606d7ec91c41d5b9251437299a2e3a95d74888` | 214 / `d44260d583bda710fabd963c1f9af8730aa92835cd8b5a7177ba5f092f381632` |
| 5 | 1,123 / `9262d7b15a926d7eeb60952e4c351c6c9b944772904fdb82438c62a2a51f6c1c` | 221 / `a2c20d3607e5fd48982b4e1620089a7b59ee7346c23fc2d2709ec2935bdfe16f` |

No staged stream was represented as a passing runtime result.

## 3. Frontend runtime and static gates

The exact fourteen-file focused suite ran in one command:

```text
Test Files  14 passed (14)
Tests       182 passed (182)
Duration    9.66s
```

The single admitted full Task 0 runtime completed without a retry:

```text
Test Files  97 passed (97)
Tests       1084 passed (1084)
Duration    50.32s
```

Additional gates:

| Gate | Result |
|---|---|
| TypeScript typecheck | exit 0 |
| Vite build | exit 0; existing `>500 kB` chunk warning only |
| i18n scanner | `36 / 20 / 0 / 20`; exit 0 |

Runtime transcript identities are recorded in `SHA256SUMS`; the focused and
full transcript SHA-256 values are respectively
`574f6c660aff2dac9eae96b450a06241f7722e72a3491a89763a468bc987f676`
and
`4d01ea34be3e8f4d0d0071292786954378cf4ee9fd216e3510ad2cc9b6e4e105`.

## 4. Protected frontend and backend boundary

All ten protected frontend rows matched. The path-sorted `sha256sum` row
aggregate was exactly
`4eae072b4eae3069b67d5fc0528227b2023500e7582d4063f51a9a288278fef4`.

The complete backend manifest contained 597 tracked paths. Its manifest SHA
was
`59412c6b815fbba9168225bf0761de95124ed5f6a97c91d7d476d2b840aaf319`,
and its diff against product base was empty.

The backend operation was collect-only under an isolated environment and
pinned reporter:

| Field | Result |
|---|---|
| exit status | 0 |
| collected | 4,527 |
| runtime seen | 0 |
| non-passing | 0 |
| collected stream SHA-256 | `4eeb117804ad874c83ffe4c04fd25ecd4de4f460801bfbf95d15c1406f32455d` |
| report SHA-256 | `b2f1539d751614942b45b15d5366383c3d7a1880adcfe3fcedb7b49d1406cc46` |
| transcript SHA-256 | `0e1e5411e4ee48fae348a04b78e69288b949505edc07ce00b3fd08bd71c05c85` |

No backend test body or provider path executed.

## 5. Rejected operator attempts

An initial focused-list command used Vitest's ambiguous
`list --json <paths...>` form. Vitest interpreted the first path as the
optional JSON output path, produced a rejected thirteen-file / 162-node
stream, and temporarily replaced
`apps/arkscope-web/src/AppShell.test.tsx` with JSON output. Git status caught
the mutation immediately. The accidental bytes were retained as
`rejected-appshell-overwrite.json` (SHA-256
`e4c3b953f78f7b3a9dad046cb1fa331e3f4509b7577a25f56db6f4dbe7e8f51b`),
and the tracked file was restored from the unchanged plan tip. Its restored
and current SHA-256 was
`e27f1ee2b365fb8d8e7fc6db8cbfb101208622ed917bc928130a2e0537fd83ce`.

That stream and two empty AppShell-only list attempts are rejected evidence.
The admitted focused identity came solely from filtering the authoritative
decoded full stream through the closed fourteen-file list. The subsequent
focused/full runtimes, protected manifest, and final worktree checks all ran
after byte-exact restoration.

An earlier normalizer invocation also used positional arguments instead of
its required flags and produced no admitted stream. Initial shell probes for
toolchain metadata contained one quoting error and one nonexistent app-local
lockfile path; neither changed repository state, and the accepted toolchain
table was rebuilt from the pinned root lockfiles.

## 6. Generated artifacts and cleanup

Before Task 0, the isolated worktree's ignored state was exactly the pinned
root `node_modules` symlink. The gates generated 31 roots containing 501 files
and 537 total entries: Vite output, Vitest cache, and Python bytecode caches.
Every generated file was recorded by path, size, mode, and SHA-256 before
cleanup. The file projection SHA-256 was
`ba630a81bcc3c683d62edf6e59d87d69fd6b7ac4042cca62d1e2c53d4bd209a0`.

Each generated root was then removed by exact recorded path. Post-cleanup:

- tracked isolated-worktree status was empty;
- ignored isolated-worktree status returned byte-for-byte to only
  `!! node_modules`;
- main-worktree tracked status remained empty; and
- no repo-relative data, profile, cache, or build artifact remained.

The artifact manifest verifies all 72 retained evidence payloads. It excludes
only `SHA256SUMS` itself.

## 7. Task 0 handoff and review

Independent review must reconstruct the base and ten staged streams from raw
artifacts, verify exact `+40/-1`, inspect both runtime transcripts, validate
the protected/backend manifests, and confirm rejected-attempt isolation and
artifact cleanup. Independent review returned GREEN with zero findings at
`8aca8c1a`; the reviewer reconstructed all 72 packet entries, raw and staged
streams, frontend/backend gates, byte restoration, and cleanup.

## 8. Task 1 - cache core

After Task 0 GREEN, the user authorized continuous execution of Tasks 1-5.
This ruling replaces only the per-task independent wait in plan Section 4.
Every task still requires its own RED/GREEN artifacts, exact stage identity,
product/test commit, docs evidence commit, and immediate stop on any plan stop
condition or accounting drift. Task 2's early browser structural check remains
a stop gate. Task 6 mutations/final admission and Task 7 merge remain separate
hard gates.

### 8.1 RED and exact accounting

Task 1 added only the reviewed seventeen IDs in
`src/settings/settingsReadCache.test.ts`. Collection succeeded before product
code and reproduced:

| Stream | Files | Nodes | SHA-256 |
|---|---:|---:|---|
| full stage 1 | 98 | 1,101 | `6f77e16694bc7994ea62a0e51ec13a7ee79fc9f03851da7a55519b04bcbc801f` |
| focused stage 1 | 15 | 199 | `543ebdffdf922d73045fa42c1e19ae2aba5cf598e8804c359ec0c868ce27fee3` |

All seventeen new nodes executed and failed only because the function-level
module import could not find the absent cache owner. There was no collection,
fixture, syntax, timer, or network error. One preceding command used a
duplicated Vitest config/root path and failed before loading tests; it is
recorded as rejected operator evidence and was not admitted as RED.

Two same-node RED refinements then proved that nested `Error`/`Promise` values
were initially retained by ordinary JSON serialization and that a discarded
catalog generation could initially trigger account warmup. The fixes reject
non-data serializer values recursively and derive account keys only from a
current successful catalog outcome. Neither refinement changed node identity.

### 8.2 Implemented contract and GREEN

Product/test commit `e34aaef8` added exactly:

```text
apps/arkscope-web/src/settings/settingsReadCache.ts
apps/arkscope-web/src/settings/settingsReadCache.test.ts
```

The cache is App-injectable pure TypeScript with no React/API/storage/network,
logging, telemetry, or provider DTO parsing. It implements the closed resource
policy, synchronous fresh/stale inspection, one promise per current
generation, invalidation-safe completion, retained stale truth on ordinary
error, `32 / 512 KiB / 4 MiB` LRU bounds, exact credential/source invalidation,
and an injected cancellable allowlisted idle scheduler.

Fresh verification produced:

```text
owner:   17 passed / 17
focused: 199 passed / 199 across 15 files
typecheck: exit 0
full collection: 1101 / 6f77e16694bc7994...
```

Final owner identities:

| Path | SHA-256 |
|---|---|
| `settingsReadCache.ts` | `da18a1fc1dd5c6947aa140c9b74e71ffd2737c82398cb0c7633286ab38843c79` |
| `settingsReadCache.test.ts` | `3847468622e3fe5e1a1a36cd28f1eefa9d8467873cab7e61a60e97f0b8debbd5` |

The Task 1 raw root is
`/tmp/settings-navigation-warm-cache-task1-8aca8c1a`; its `38`-entry manifest
SHA-256 is
`b63d5cdf076383dc26ec0310ea858c9833bb9adeb795bb850393bcdab4b7240e`.
The sole generated Vitest cache file was manifested and removed. Post-product
commit, both tracked worktrees were clean and the isolated ignored state was
again exactly `!! node_modules`. Task 2 is the next batch stage.

## 9. Task 2 stop-and-amend

Task 2 first replaced the one superseded directory node and added the other
five reviewed nodes without touching product code. The decoded collection
matched the predeclared identities exactly:

| Stream | Nodes | SHA-256 |
|---|---:|---|
| full stage 2 | 1,106 | `10965b1c8e5a51cbf5d38950b0db8410faef1a528e6dc2856e391267019a37bc` |
| focused stage 2 | 204 | `e34c217edb518485ebacbbe382a44f47f36e536c9a15f13125add0664910a085` |

The two direct owner files then produced an admissible RED: `30` existing
nodes passed and five intended old-contract assertions failed (complete
directory, two group-top outcomes, and two sticky/shared-offset CSS outcomes).
The busy-rejection protection node remained GREEN, as expected.

An initial implementation made both direct owner files GREEN at `35/35`.
The required complete focused run then stopped at `199 passed / 5 failed`:

- four `SettingsModelRouting.test.ts` cases reached the new group-top effect in
  jsdom, whose `HTMLElement` lacks `scrollTo`; the semantic implementation can
  use the scroll owner's `scrollTop` directly within existing Task 2 scope;
- the existing
  `SettingsPostPgExitStorage.test.ts > uses_normal_user_outcomes_in_the_enabled_settings_directory`
  node still asserted the now-superseded four-link active-group directory.

The latter file was absent from Task 2's owned paths even though the reviewed
design requires all nine links. No node was added, removed, or renamed by this
finding. Task 2 stopped before product/test commit, early browser work, or any
Task 3 action. The uncommitted five-file product/test delta was retained as
`/tmp/settings-navigation-warm-cache-task2-4745b359/task2-paused-product.patch`
(SHA-256
`673846832dc7f607435f36ad955c6a155a610edf0eb08cbd8553253a8598c18c`),
then the worktree was restored to the clean Task 1 tip before this docs-only
amendment.

The bounded amendment adds only `SettingsPostPgExitStorage.test.ts` to Task 2
and authorizes evolving that one existing assertion to the complete registry
order. Stage identities, the `+6/-1` Task 2 delta, and the global `+40/-1`
ledger remain unchanged. The initial stop record requested an extra focused
review, but the user clarified that the already-granted Tasks 1-5 batch
authorization covers this bounded correction. No technical gate was waived:
the expanded focused owner, browser, protected-byte, stage-identity, and
product/docs commit requirements all remained mandatory.

## 10. Task 2 - sticky navigation and complete directory

### 10.1 Implementation and GREEN

Product/test commit `ecf87f0c` changes exactly the six reviewed owners. The
Settings workflow tab row is sticky and non-wrapping inside the existing
`.main` scroll owner; its one shared offset also owns the directory rail and
section anchor margin. The empty directory renders all three registry groups
and all nine sections. Manual group navigation and exact-anchor navigation are
separate post-mount effects: an accepted manual switch restores the Settings
scroll owner to zero and focuses the selected tab without scrolling, while an
exact target mounts its group then focuses the anchor. Dirty-confirm and busy
guards schedule neither effect until navigation is actually accepted.

The existing post-PG-exit directory assertion now checks the complete registry
order without changing its node ID. Fresh verification was:

```text
direct owners: 45 passed / 45
focused:       204 passed / 204 across 15 files
typecheck:     exit 0
full stage:    1106 / 10965b1c8e5a51cb...
focused stage: 204 / e34c217edb518485e...
```

All ten Task 0 protected rows, including generic `ui/Tabs.tsx`, its owner,
both frozen Settings fixtures, registry/copy/i18n owners, and Investor Profile
owners, remained byte-identical. `git diff --check` passed.

### 10.2 Browser evidence

A fail-closed Playwright harness served only typed in-memory GET fixtures and
blocked every unknown or external request. The admitted run used Chrome
`150.0.7871.128` at `1322x777` and `390x844`. Both viewports mechanically
proved:

- three directory groups and nine unique section links;
- one mounted tabpanel and no retained Data Sources owner after leaving it;
- exact macro-anchor focus below the sticky tab row;
- deep Settings scroll `1200` with the row still sticky;
- no page/main horizontal overflow and no clipped tab label; and
- manual switch to Personalization at `scrollTop=0` with selected-tab focus.

The result JSON is SHA-256 `efaab767f65c4101438d780c9d5e1525e6b3051c9fa4aaaccd2ba0f2e4f2dc92`.
The inspected desktop and mobile screenshots are respectively
`b8bf994eb9e6401d040e3a1172c0e1c1f08cd00cf4e6c60de6262aa994c90b31`
and
`3a048e8211d279c1b62fdec783796aa735a569b62f8835381f98d0273d06e12a`.

Two earlier harness attempts are rejected evidence, not product failures. The
first supplied `fundamentals: null` against the current typed status DTO and
therefore crashed the fixture-rendered storage panel. The second passed every
layout assertion but correctly rejected two missing Investor Profile GET
fixtures because console errors are forbidden. The final harness fixed only
its in-memory fixtures; no repository byte changed between those attempts.

### 10.3 Artifacts and handoff

The Task 2 root is
`/tmp/settings-navigation-warm-cache-task2-4745b359`. It contains `17`
payload entries; `SHA256SUMS` has SHA-256
`73f1a4cd5eca233cbe39f76f994aa5ee852d2438151dac324f80072df074ea97`.
Thirty-five Vite/Vitest cache files were individually hashed, then the exact
generated `apps/arkscope-web/node_modules` root was removed. Port `8461` was
closed, ordinary status contained only the six intended owners before commit,
and ignored status returned to the sole pinned root `node_modules` symlink.
Task 3 is the next stage of the already-authorized batch.

## 11. Task 3 - RED and bounded real-ID amendment

The seven exact Task 3 nodes collected without import, fixture, network, or
timer errors. Their first runtime produced `64 passed / 7 failed`; every
failure landed on the intended absent contract: no App-owned cache prop, no
retained catalog render/replacement/generation invalidation, and no scheduled
idle warmup. The decoded stream already reproduced the predeclared full
stage-3 identity `1113/eefdbdaa10c83786...`.

During GREEN wiring, the reviewed account warmup reached a Task 1 fixture
assumption that was not product-real: the test used `local-oauth`, but the
backend's stored credential authority emits `local:<positive-int>`. The cache
validator rejected every colon, so a real active OAuth row could not be warmed.
The bounded amendment adds the existing cache source/test to Task 3 ownership,
accepts only that exact stored-local-ID shape when a colon exists, and evolves
the existing warmup node to `local:7`. It does not relax control-character,
length, unknown-colon, raw-account, secret, or persistence constraints.

After the correction, the seven new owners plus the cache owner produced
`88/88`; the wider App/Settings/protected group produced `121/121`; typecheck
passed. Full and focused decoded identities are exactly
`1113/eefdbdaa10c83786...` and `211/d74255067e3ca453...`. A deliberately wider
intermediate focused run exposed nine old Data/Sync call-count assertions:
warmup now issues the reviewed GETs, while their Task 5 components have not yet
joined the shared cache and therefore duplicate those calls. This is the
predeclared partial-integration state that Task 5 removes; the failures are not
called GREEN and their assertions are not weakened in Task 3.

### 11.1 Product commit and packet

Product/test commit `462bb8af` creates one cache lazily per App root, injects it
through `SettingsView`, renders retained model catalog truth synchronously,
joins visible and idle revalidation, invalidates before every reviewed catalog
mutation refresh, and prevents old completions from repopulating. Idle work is
one cancellable post-paint task and supplies only the reviewed local GET
allowlist; it contains no interval, persistent listener, extension read, POST,
discovery, model/provider execution, schedule mutation, or Investor Profile
request.

Final Task 3 checks were:

```text
exact RED owners:          64 passed / 7 failed (intended contracts only)
new owners + cache owner:  88 passed / 88
wider protected owners:    121 passed / 121
typecheck:                 exit 0
full collection:           1113 / eefdbdaa10c83786...
focused collection:        211 / d74255067e3ca453...
intermediate full focused: 202 passed / 9 failed (rejected, Task 5 boundary)
```

All ten protected rows remained byte-identical. The sole generated Vitest
result file and empty Vite directories were recorded then removed; ignored
state returned to the pinned root `node_modules` only. Raw evidence is under
`/tmp/settings-navigation-warm-cache-task3-d4e4bc4d`: `11` payload entries,
`SHA256SUMS` SHA-256
`77e338a2bc3e161ea120da684871bb63c96a8afdf6afa9245c4ea39affa86ee1`.
Task 4 follows immediately under the user's batch authorization.

## 12. Task 4 bounded cache-handoff ownership amendment

Pre-edit inspection found that the reviewed Task 4 owned-path list named the
Provider implementation and its test but omitted the existing render owner in
`Settings.tsx`. The Provider cannot consume the Task 3 App-owned cache unless
that call site passes the same instance. A Provider-local or module-global
fallback would violate the App-root isolation contract.

The bounded amendment therefore adds `Settings.tsx` only for the prop handoff.
It authorizes no other Settings behavior, no new cache owner, and no test-node
change. Task 4 remains exactly `+3/-0`, with full target
`1116/09d31fa1bd22d3b0519c8dce2c606d7ec91c41d5b9251437299a2e3a95d74888`
and focused target
`214/d44260d583bda710fabd963c1f9af8730aa92835cd8b5a7177ba5f092f381632`.
The user's existing Tasks 1-5 batch authorization remains in force; this is
not an added review wait.

## 13. Task 4 - OAuth account read integration

### 13.1 RED, implementation, and exact identities

The three exact Task 4 nodes were first applied alone to isolated parent
`e7d18fa2`. They collected and ran without import or fixture errors, producing
`15 passed / 3 failed`; the failed set was exactly retained-immediate GET,
retained-on-error, and targeted manual-sync replacement. No existing node
failed.

Product/test commit `bbf81adb` passes the existing App-root cache from
`Settings.tsx` into `ProviderSection`. Cached account views are validated
against the local credential row before display, stale values remain visible
during one shared GET, loader errors preserve last-good truth, and invalid
credential-bound responses never enter visible state. Existing visible,
focus, and manual sync policy remains the sole POST owner. A successful sync
replaces only its exact account key after the existing generation/binding
checks; credential mutation invalidates that key and leaves unrelated account
entries intact.

Verification was:

```text
exact RED:                15 passed / 3 failed
Provider GREEN:           18 passed / 18
cross-handoff regression: 106 passed / 106 across 5 files
typecheck:                exit 0
full collection:          1116 / 09d31fa1bd22d3...
focused collection:       214 / d44260d583bda710...
```

Both normalized collection streams are byte-identical to the preconstructed
Task 4 streams. `git diff --check` passed. No backend, API, credential,
provider request, profile, storage, scheduler, or production-data byte
changed.

### 13.2 Packet and cleanup

Raw evidence is under
`/tmp/settings-navigation-warm-cache-task4-bbf81adb`. It has `10` payload
entries; `SHA256SUMS` has SHA-256
`307db954a1821205d9bfb8e040b4c53482c95fa21fca4e8e6cfebb0593700d05`.
The isolated RED worktree was removed after capture. One generated Vite result
file was inventoried, and the exact app-local `node_modules` cache root was
removed; ignored state returned to the pinned root `node_modules` link only.
Task 5 follows immediately under the user's batch authorization.

## 14. Task 5 bounded cache-handoff ownership amendment

Pre-edit inspection found the same bounded render-owner omission as Task 4:
the reviewed Task 5 list named the four Data/Storage sections and tests but
not `Settings.tsx`, which holds the App-root cache at all four render sites.
The amendment authorizes only those prop handoffs and forbids module-global or
section-owned fallback caches. Task 5 remains exactly `+7/-0`; final full and
focused identities remain `1123/9262d7b1...` and `221/a2c20d36...`. Existing
batch authorization remains in force without an added review wait.
