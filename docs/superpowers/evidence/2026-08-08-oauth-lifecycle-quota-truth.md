# OAuth Lifecycle and Subscription Usage Truth Evidence

> **Status:** COMPLETE; MERGED VERIFICATION GREEN
>
> **Date:** 2026-08-08
>
> **Plan-review tip:** `0753947e049a8ecabeab5220f4d3427eeb256a65`
>
> **Grounding base:** `7257699171a81294b74ff8cde61fb90bb065a2b4`

## 1. Scope and boundary

Task 0 performed collection, one focused offline runtime, and static identity
checks only. At Task 0 close it had changed no product or test file, contacted
no provider, read no real token store/profile database, and made no
production-data or scheduler change. Task 1 began only after independent GREEN
review of that packet.

The branch was clean at reviewed plan tip `0753947e`, and its merge base with
the grounding base was exactly `72576991`. Git used the established no-op
git-crypt filters for this reviewed plaintext worktree.

## 2. Pinned execution assets

| Asset | SHA-256 |
|---|---|
| `/tmp/arkscope_asyncio_wakeup_probe.py` | `10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e` |
| `/tmp/eir002-green-baseline/arkscope_eir002_reporter.py` | `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928` |
| `/tmp/eir002-green-baseline/run_native.sh` | `e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f` |
| `/tmp/eir006_vitest_list_normalizer.py` | `955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac` |
| `package-lock.json` | `5322cb03099b873066b7572c02db68a427ddc7509fdc9850cf9d8d3948a19f2c` |
| pinned `node_modules/.package-lock.json` | `4dd5182f8111b54dd608344c45513f3b310f39321a3cc9832b60cefc2fa241ff` |

Node was exactly `v22.14.0`. The native wrapper's wakeup preflight returned the
required `{"callback_fired": true, "ready_count": 0, "wake_bytes": 0}` for
all three backend invocations.

## 3. Reproduced base identities

### 3.1 Backend

| Collection | Nodes | Node-stream SHA-256 | Reporter SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| full collect-only | 4,581 | `6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f` | `77dc55c95d20dce51948bc425e57d808040868e05d51a94328915b1ff5def15a` | `0e7baf939dc186da6d035b1223260c34801e69678f5c88b57e9543945e4e81c5` |
| focused collect-only | 246 | `57583e93f68a62ef8a2ac82efa70fd3d5374957f0c2156d1f58f337daccfce07` | `c0e27a69a8fb5362ad6fe47380375c74319e9f65607271ef45dea1ce2913bd57` | `a1e1170f83d66dc649a23dd95a389c7f0b31c572cae9bbc5e5ba3b773d025474` |

Collect-only reports had exit status zero, complete collected arrays, zero
runtime-seen nodes by design, and empty non-passing arrays.

The exact eight-file focused runtime produced:

```text
collected: 246
seen: 246
passed: 246
skipped/failed/errors: 0/0/0
exit: 0
runtime report SHA-256: 11c06860a887e202be86147fb771dcf959e8960193f47b70d173c8b81c80c8c5
runtime transcript SHA-256: 694e140e2f30a7133643082e69cbd47a8bff556391c7f191793002dc137042b0
```

The focused file list and its path/blob/SHA/size manifest are respectively
`b400dd594db9d3042a747d769dc94553f0f2ec972cc661d44594b0eb98c64887`
and `bc6cc15603af4ee7562ffaedb185e2fa5955511239f206dd041ad7ba85709671`.

### 3.2 Frontend

Vitest ran from `apps/arkscope-web` against the pinned root toolchain. The
JSON-decoding normalizer, not raw JSON text extraction, produced:

| Collection | Nodes | Node-stream SHA-256 | Raw-list SHA-256 |
|---|---:|---|---|
| full | 1,077 | `3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb` | `16d00b2b335fee28a28b89858edcbd3a2eb4f8807b51e0944e10d38a4fa0c1d9` |
| `ProviderSection` focused | 8 | `08aad36811d22f5d7ca55f16891d17c5d2a2345f9f2ad4802c11db85e223b83a` | `709e2699d8add26711223601aac45dd0eda8e8ccfc11c3b6285d920f6d4a790e` |

The ignored pinned `node_modules` symlink was removed after listing. No repo
artifact remained.

## 4. Provider and secret isolation

The focused runtime used the pinned wrapper's `env -i` allowlist with scratch
HOME, TMPDIR, XDG cache, lock directory, five database overrides, and EDGAR
directory. It inherited no OpenAI/Anthropic key, OAuth token, real profile path,
browser profile, or `CODEX_HOME`.

The eight existing test owners use temporary SQLite/plaintext stores and
injected fakes. Static review found:

- Claude SDK calls are replaced by `_install_fake_query` or local async fakes;
- ChatGPT discovery/execution clients are replaced by scripted clients;
- OAuth route probes and exchanges are injected/monkeypatched;
- the only `urlopen`-shaped test helper is `_fake_http_error`, which raises a
  local constructed `HTTPError`; and
- no `Path.home`, `expanduser`, `~/.codex`, default keyring/token-store
  constructor, direct `requests/httpx` call, or provider client constructor is
  present in the focused test files.

The successful offline runtime is supporting evidence, not a claim that future
new tests are automatically isolated. Tasks 1-7 must retain this boundary.

## 5. Protected paths

Task 0 recorded 37 exact path/blob/SHA/size tuples outside this slice. The path
list SHA is
`9a503591fd04098f5d3a5cb1b15bc81b164c0e9fe7061c4ea8147775a95589ff`;
the tuple-stream SHA is
`bcc7bf54c6e82e39a01c2e98dc9677640605580fa61eba4eba0b3bf39a084e65`.

```text
apps/arkscope-web/src/Settings.tsx
apps/arkscope-web/src/settings/settingsCopy.test.ts
apps/arkscope-web/src/settings/settingsCopy.ts
apps/arkscope-web/src/settings/settingsRegistry.test.ts
apps/arkscope-web/src/settings/settingsRegistry.ts
data_sources/financial_datasets_client.py
docs/design/FINANCIAL_DATASETS_CAPABILITY_SPEND_DECISION.md
scripts/scoring/README.md
scripts/scoring/__init__.py
scripts/scoring/import_news_scores_local.py
scripts/scoring/openai_summary.py
scripts/scoring/score_ibkr_news.py
scripts/scoring/score_risk_anthropic.py
scripts/scoring/score_sentiment_anthropic.py
scripts/scoring/validate_scores.py
src/agents/anthropic_agent/agent.py
src/agents/anthropic_agent/tools.py
src/agents/config.py
src/agents/openai_agent/agent.py
src/agents/openai_agent/tools.py
src/agents/shared/context_manager.py
src/agents/shared/model_catalog.py
src/agents/shared/prompts.py
src/agents/shared/subagent.py
src/fixed_task_runtime_config.py
src/model_capabilities.py
src/model_effective.py
src/model_route_store.py
src/model_routing.py
src/model_task_canary.py
src/news_normalized/schema.py
src/news_normalized/score_import.py
src/news_normalized/scores.py
src/signals/anomaly_detector.py
src/signals/event_chain.py
src/signals/synthesizer.py
src/tools/registry.py
```

This freezes model routing/agent loops/prompts, Financial Datasets policy,
Scripts Tranche B, and broad Settings navigation. It deliberately excludes the
OAuth owners named by plan Section 1.1.

## 6. Task 0 disposition

All four reviewed base identities reproduced exactly, focused runtime is
`246/246` GREEN, the provider/secret boundary stayed hermetic, and protected
identities are banked. The worktree returned clean with no `data`, `src/data`,
or toolchain residue.

Task 0 completed at `01fb7177`. At that point independent review was the sole
next gate and Task 1 remained unauthorized. That dated gate was later cleared;
Sections 7-10 record the authorized Task 1 work.

## 7. Task 1 RED

The eleven exact lifecycle node IDs were added to
`tests/test_oauth_lifecycle_status.py` with all new-product imports inside test
functions. The pre-product run collected all eleven and failed all eleven for
the intended absent contract: ten failed because `src.auth_drivers.oauth_status`
did not exist, and the API-payload node failed because `lifecycle_state` did not
exist. There was no collection, fixture, SQLite, date, or secret-dependent
wrong RED.

## 8. Task 1 implementation

Product/test family `d4adb6e3062a7662c13689002f5faea5e2d59d77` implements:

- the closed lifecycle states `ready`, `refresh_required`,
  `refresh_failed_retryable`, `reauth_required`, and `unverifiable`;
- no-create observation reads, latest-only bounded refresh witnesses, sanitized
  error details, and a no-create singleton accessor;
- OAuth `available` as a projection of lifecycle readiness while preserving the
  existing API-key and environment-derived availability behavior;
- runtime resolution of an explicitly active OAuth credential independently of
  display availability, without credential/provider fallback;
- token-store expiry as ChatGPT OAuth authority, with the compatibility DB
  expiry field no longer written, while user-declared Claude setup-token expiry
  remains DB-owned; and
- explicit token-store injection at the route boundary so pure projections do
  not open a real default token store.

Two implementation corrections kept the reviewed slice narrow. First, Task 1
does not migrate OAuth account labels: account-snapshot ownership belongs to
Task 3. Second, `OAuthObservationStore` contains only Task 1 refresh/lifecycle
state; the account adapter and account snapshot schema were not implemented
early.

## 9. Task 1 verification

| Collection/run | Nodes | Node-stream SHA-256 | Report SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| backend full collect-only | 4,592 | `7f9d48845e7d0a4cde3e5c3e91b944eccc3cbcaf4109c0d11a01ef9a72dbfc54` | `95237174787b859b67f3e82662f6d3aea233ab94d56475c0c0b24c1434cb017e` | not admission-bearing |
| backend focused collect-only | 257 | `b875fd44906f15fff83dee815516f6dc4b99ec1565a2777b0dbd1d88649faeef` | `5042d06d94f7a92078171e13bd4058d31a48e29fca0c4fef4bae810e0f0e5eb8` | not admission-bearing |
| backend focused runtime | 257 | same focused stream | `f744a66c5ce78ca0bf837831016f9e01ec2395a2300a43c49efbd609ecaee507` | `e7590eb35d6b55d4224ecae264373c08c7a0967dad6db53fae3c89612d99c663` |

The focused runtime ended `257 passed in 5.20s`, with `257 collected = 257
seen`, zero non-passing nodes, and exit zero. The related route/inventory
collateral group completed `164 passed`; the eleven owning nodes completed
`11 passed`. `py_compile` and `git diff --check` passed.

All 37 protected paths remained byte-identical to `0753947e`; the protected
path stream remains `9a503591fd04098f5d3a5cb1b15bc81b164c0e9fe7061c4ea8147775a95589ff`.
The new tests use injected token stores, stores under temporary paths, and fake
errors. They contain no provider URL/client, `Path.home`, real profile path, or
default real-token-store construction. No provider call, production-data write,
scheduler change, frontend change, account read, or live synchronization was
performed.

## 10. Task 1 disposition

Task 1 is complete at `d4adb6e3`. The reviewed full/focused identities match
exactly, the lifecycle/store family is GREEN, and protected Tranche B/model
routing/agent/Settings owners are unchanged. At Task 1 close, Task 2 and all
later tasks remained unauthorized. Independent review later cleared that gate;
Sections 11-14 record the authorized Task 2 work.

## 11. Task 2 RED

`tests/test_oauth_cross_process_lock.py` added exactly the four reviewed node
IDs and used spawned Python processes, temporary plaintext stores, filesystem
barriers, and bounded joins. Before the product edit all four failed:

- both processes loaded the same expired record and attempted the one-use
  rotating grant; one succeeded and one received the expected invalid-grant
  family error;
- delete removed the credential and old token while refresh was paused, after
  which the late refresh wrote a new token and reproduced resurrection;
- the bounded timeout node found no timeout/error-code contract on the old
  process-local context manager; and
- the release node found neither the bounded contention result nor a lock file
  whose descriptor lifecycle could be proved.

There was no collection, provider, credential, keyring, home-directory, or
fixture-setup wrong RED. The first two failures directly established the
cross-process mutation defect rather than relying on the absent timeout API.

## 12. Task 2 implementation

Product/test family `f9fd8a1dd0adf8bcd74f4185e9e165817a02a7fd`
changes only `src/auth_drivers/chatgpt_oauth_login.py` and the new four-node test
file. The shared `oauth_credential_lock` now:

- acquires the existing per-credential thread lock and one bounded POSIX
  `flock` before yielding;
- uses `ARKSCOPE_LOCK_DIR/oauth_credentials` when configured, otherwise
  `<profile_state.db parent>/locks/oauth_credentials`;
- names the file only with a SHA-256 credential digest and enforces directory
  mode `0700`, regular non-symlink file mode `0600`, close-on-exec, and
  no-follow open semantics;
- returns typed `ChatGPTOAuthLoginError(error_code="oauth_lock_busy",
  reauth_required=False)` on thread/file timeout, missing `fcntl`, invalid path,
  open/fd, or flock failure; and
- unlocks and closes the fd, then releases the thread lock, on normal return,
  body exception, or acquisition failure. It never continues unlocked.

The existing refresh, in-place re-login completion, and credential-delete
paths already call this one context manager, so the family upgrades all three
without adding another lock owner. Browser wait and authorization-code exchange
remain outside the critical section.

Post-commit source identities are:

| Path | Lines | SHA-256 |
|---|---:|---|
| `src/auth_drivers/chatgpt_oauth_login.py` | 730 | `51a2ce52eb6fee60c62c556e162cd7df776d36a8ef49d1d5e4c59e6f9d5fa07f` |
| `tests/test_oauth_cross_process_lock.py` | 389 | `4efb997a8a4e43381b1c78bd74137b3aa929b91c8f529d04f3e33a26fcc9b555` |

## 13. Task 2 verification

| Collection/run | Nodes | Node-stream SHA-256 | Report SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| backend full collect-only | 4,596 | `b9056110d25f64dc399e176502871a118b091bd4c3a4714933cb348dbc1d7b40` | `4385c884d9aec9d9b8de3cd15ce61fd8cc8304c18a46a691c754698a5836922b` | `ea35b3357aa22e3a3ad1e2ea2bbc29f09946727899388fa543c1565fa41f0761` |
| backend focused collect-only | 261 | `a4b9af293b9c17c9bb93e82c8b99b5a181f06c2ebb9ea023381013998904dca7` | `41d862a6349b4016f4da84dc14998c49cb110e97a12e3242ba6570e5b1a8875d` | `c72060e47234b1087ee1b27df0debf4526e663d657b99d6363fbf365e58b808b` |
| backend focused runtime | 261 | same focused stream | `dc46b448330d021b67927c2c3dc291ef430f935abb1217ca95fb3d2b54ba4e1b` | `e1ccae927488612c21069ec2be7f761a8c34ad2ef6619925f835d43a60753c35` |

The post-commit focused runtime ended `261 passed in 10.45s`, with all 261
nodes seen, zero non-passing nodes, and exit zero. The existing login, manager,
and route owners plus the new lock file also completed `92 passed`; the owning
file alone completed `4 passed`. `py_compile` and `git diff --check` passed.

A separate persisted process proof at `/tmp/arkscope-oauth-task2/process-proof`
contains 34 files with manifest SHA-256
`14eb37ef95af382cfe4b160a158b5de993467173be22876fec120078bf5a13a0`.
Its reporter/transcript SHAs are respectively
`84a58b219834822d9f9004ede90c9a2da46c3842d7d27abb7805dc229f7d0043`
and `7d1edeb663b168ffb8481f919710ad4f079e008508540de7d098106a7bb9ce29`.
The raw results show both refresh workers succeeded with one grant attempt;
delete and refresh both succeeded with final token storage `{}`; the contender
returned `oauth_lock_busy` without entering; and another live process acquired
the lock after both normal and raised-body exits while the prior process
reported zero matching open fds.

All values in that packet are obvious test sentinels. No provider, real token
store, keyring, home directory, profile, production DB, scheduler, frontend, or
account adapter was touched. All 37 protected paths remain byte-identical to
`0753947e`.

## 14. Task 2 disposition

Task 2 is complete at `f9fd8a1d`. The exact staged identities and two-process
behavior match the reviewed plan, lock failures are typed and fail closed, and
the worktree returned clean. Task 3 account adapter/store/API and all later
tasks remained unauthorized until independent Task 2 review. That dated gate
was later cleared; Sections 15-18 record the authorized Task 3 work.

## 15. Task 3 RED

`tests/test_subscription_account_usage.py` added exactly the nine reviewed
adapter/store/route node IDs. An initial fixture transcription used an invalid
f-string and failed during collection; that wrong RED was rejected and the
fixture was corrected before any product edit or accepted RED claim.

The corrected pre-product run collected all nine nodes and failed all nine for
the intended absent Task 3 contract: the bounded Codex adapter, typed account
snapshot DTO/store methods, sync service, and cached/sync routes did not yet
exist. There was no collection, SQLite, fixture-date, network, credential,
keyring, home-directory, or real-profile wrong RED.

All protocol fixtures are local executable children with obvious sentinel
tokens and account identifiers. They assert the exact initialize, external
token login, `account/read`, `account/rateLimits/read`, and
`account/usage/read` sequence and reject any thread/turn operation.

## 16. Task 3 implementation

Product/test family `f1eec320df960e0249eff998838254b6e8a16876`
implements:

- closed typed account, rate-limit, credit, spend-control, and usage DTOs plus
  one latest-only `oauth_account_snapshot` row per credential;
- no-create/query-only cached reads and atomic writes containing only
  allowlisted non-secret fields, `observed_at`, source, and a
  credential-bound account fingerprint;
- a Codex app-server adapter pinned to exact CLI version `0.147.0`, with an
  isolated temporary `CODEX_HOME`, a closed environment allowlist, bounded
  JSONL request IDs, response and output limits, timeout, and process-group
  cleanup on every exit;
- rejection of unexpected server requests and all thread/turn methods, without
  starting a model turn;
- local cached GET and explicit mutating sync POST routes, while the existing
  credential inventory GET remains local-only; and
- per-credential backend single-flight so concurrent sync callers share one
  adapter execution.

The locally generated `0.147.0` protocol schema shows that `account/read`
returns account kind/email/plan but no raw account identifier. Same-account
validation therefore compares token-store account metadata with the ID-token
account claim. Only the resulting credential-bound SHA-256 fingerprint leaves
the adapter; raw account ID, token, email, fixture sentinel, stdout, and stderr
are neither returned nor persisted. This is the actual verified boundary, not
a claim that `account/read` supplies an identifier.

Current sync failure, account mismatch, unsupported protocol/version, malformed
response, and timeout preserve the last good snapshot and its observation
time. Unsupported auth mode is rejected before token loading or child launch.
Missing cached state remains typed unknown; it does not create the profile DB.

Post-commit source identities are:

| Path | Lines | SHA-256 |
|---|---:|---|
| `src/auth_drivers/oauth_status.py` | 568 | `07bff2ce1dd2aa274a407bd04969ec1c73c4c8518f7e2ef89c1b4e69ee0f1eb2` |
| `src/auth_drivers/codex_account_usage.py` | 614 | `4ec1d1989b132b198d4e1367575df360adc9a97a1e51899215e242cfb3302d09` |
| `src/api/dependencies.py` | 378 | `1cf4fb462f04a0735deff56a63916c3db3cde7152ae365fdbff4ac6687bdb4e3` |
| `src/api/routes/config_routes.py` | 1,233 | `ad6d3a718a36c5dc5055f4854f6be51c10b1d7fd066dee47ff935111dc2cc7f6` |
| `tests/test_subscription_account_usage.py` | 632 | `6fd932eb6083e8ec86d95ebc4c5448c90fc0537c57ca1715301567f39c98d305` |

## 17. Task 3 verification

| Collection/run | Nodes | Node-stream SHA-256 | Report SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| backend full collect-only | 4,605 | `3b6cbd5ffbe0decccddb2914d422c650c50c58f72667ccb285f9cf4a74b20c08` | `66d1b717cf096e7a06ff314ad457b9acfb6d7500884c4b1db38a357356de5f54` | `48b2b279096e7a06ff314ad457b9acfb8afd19183536e5461b40c4a1320bf624` |
| backend focused collect-only | 270 | `d9b03cc7320a697abdc4a9049957d390f690d5223e6e5cecd4429a7c34b09338` | `7eabbacaf0507acedbda3a93ecaa038f272f0a7e7a12c4deed7d5ace6cc52e68` | `0afa22f12ad9117da17546d022754964886f2ab9f2d834e9337de18e1585b34d` |
| backend focused runtime | 270 | same focused stream | `8f411c161cbda8cb78e7ccef8e5bbb9be80be1e0b607a3487ece2a0f641f49ff` | `097f44e50e5d6d36fa66bcbec52d44151736b8d25c643f0315ae6721157d000b` |

The committed focused runtime ended `270 passed in 11.60s`, with all 270
nodes seen, zero non-passing nodes, and exit zero. The owning Task 3 file ended
`9 passed`. `py_compile` and `git diff --check` passed.

The nine-node family proves five exhausted-account rate-limit reads leave usage
unchanged; strict account mismatch and unknown protocol preserve the last good
snapshot; exact-version refusal and hung-child timeout leave no child process;
missing cached state is unknown/no-create; inventory, cached read, and explicit
sync are separate HTTP contracts; concurrent sync is single-flight; and
credential listing never refreshes or contacts the provider.

The tests execute only redacted local fixtures under temporary paths. No live
provider request, model turn, real token store, keyring, home directory,
profile DB, production data, scheduler, frontend, or Tranche B owner was
touched. All 37 protected paths remain byte-identical to `0753947e`.

## 18. Task 3 disposition

Task 3 is complete at `f1eec320`. The exact full/focused identities match the
reviewed plan, all nine RED-first account adapter/store/route contracts are
GREEN, bounded child cleanup and credential-bound persistence are proved, and
the worktree returned clean before this docs-only closeout. Task 4 wiring and
all later tasks remain unauthorized until independent Task 3 review.

## 19. Task 4 RED

Task 4 evolved existing owners only; it added, removed, or renamed no node. The
first targeted run produced eight failures, all at the absent reviewed wiring:
refresh did not accept or write the observation/sync seams, the login manager
did not inject those seams, Claude import still duplicated manual expiry into
the token record, and credential deletion did not clear OAuth observations.
There was no collection error, fixture-date error, SQLite error, provider
request, real token-store access, or environment-dependent wrong RED.

The accepted RED assertions required the refresh attempt before the rotating
grant, success only after token save, stable redacted failures, post-commit
account sync, exact re-login/delete invalidation, token-store-only ChatGPT
expiry, DB-only Claude manual expiry, and unchanged API-key deletion behavior.

## 20. Task 4 implementation

Product/test family `610d3471fc3904b5e6e5052d35c4a0c67840bd8b`
implements:

- durable latest-only refresh attempt/success/error witnesses with the closed
  stable-code set and bounded redacted diagnostics;
- token-store save before success witness, so a successful refresh immediately
  changes lifecycle/expiry projection without writing the ChatGPT DB expiry;
- re-login cleanup of discovery plus both credential-bound observation rows
  under the shared lifecycle lock before adopting a replacement token;
- OAuth-only credential deletion that clears the exact refresh/account rows
  before token and metadata removal, while API-key deletion never touches the
  OAuth observation store;
- post-commit login/re-login/refresh account-sync seams whose failure cannot
  roll back valid auth; and
- Claude setup-token import with manual expiry retained only on the credential
  row, never duplicated into the token-store record.

Final self-review found one race after the initial GREEN: an account adapter
could start with the old token, deletion could clear the credential, and the
old adapter result could then recreate its snapshot. The existing
single-flight owner was strengthened without adding a node. Snapshot commit
now reacquires the same credential lifecycle lock, reloads and constant-time
compares the token generation, and returns typed
`credential_changed_during_sync` with no snapshot when it changed. The test
blocks the adapter, completes deletion and exact observation cleanup, then
proves the old result cannot persist or return stale account state.

Post-commit source identities are:

| Path | Lines | SHA-256 |
|---|---:|---|
| `src/api/dependencies.py` | 423 | `bb88908670fba3c108c4aed7cb86ef6465d0cd19695deb2ee7bee281133e12d0` |
| `src/api/routes/config_routes.py` | 1,262 | `e00571e8831e5508a411dc4cd6b824aab6ec78be7e1e67064ce9b861193ef573` |
| `src/auth_drivers/chatgpt_oauth_login.py` | 920 | `157d2fd3e032559af049dcf38afbf5b9d820eb29d5c1da939aa92a478582d774` |
| `src/auth_drivers/chatgpt_oauth_manager.py` | 245 | `dc920a1f7a0b671f410ffd064c5049981d17b4d0e989ca78cf70011e3217b0fb` |
| `src/auth_drivers/oauth_status.py` | 585 | `b955610122dda6dc22d3b3364cdfef007ac4126752b4b7a3f7c4d4120bb2968d` |
| `tests/test_chatgpt_oauth_login.py` | 889 | `dbc4653af98f5f69ea11906e80b49d5e84485290d9884c608dd25f6ad1f8272a` |
| `tests/test_chatgpt_oauth_manager.py` | 473 | `aa4848ec9c2804f44bc26a591ce9042501b0ef700d5dcfaa0d3d4af3f0ecac5b` |
| `tests/test_chatgpt_oauth_routes.py` | 687 | `f94fc6d4c8514067355e6849e964c85e090f871227e5a6b79dbe7d806464f677` |
| `tests/test_oauth_import_route.py` | 177 | `2c00735cbf56545151451dfbebdef0896cdbfce884579c6954de6fc257c5d718` |
| `tests/test_oauth_lifecycle_status.py` | 504 | `556162cb062cef1b1efebbefc9c441676f50fe59d4fdd4757fe477ce81665b6b` |
| `tests/test_subscription_account_usage.py` | 666 | `a6113cacdd15629bca5a5bf859d90de93acd2a7b44cf0fa08b75870f71c93c68` |

## 21. Task 4 verification

| Collection/run | Nodes | Node-stream SHA-256 | Report SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| backend full collect-only | 4,605 | `3b6cbd5ffbe0decccddb2914d422c650c50c58f72667ccb285f9cf4a74b20c08` | `66d1b717cf096e7a06ff314ad457b9acfb6d7500884c4b1db38a357356de5f54` | `e62ae8cae36f033db1fc18edc6fbf561738bb6196bf5398002f22c1c1e76c53d` |
| backend focused collect-only | 270 | `d9b03cc7320a697abdc4a9049957d390f690d5223e6e5cecd4429a7c34b09338` | `7eabbacaf0507acedbda3a93ecaa038f272f0a7e7a12c4deed7d5ace6cc52e68` | `21ba81260e0255d2f051909b46982ea1b11187b37a5090d9afafcf5cc946070b` |
| backend focused runtime | 270 | same focused stream | `8f411c161cbda8cb78e7ccef8e5bbb9be80be1e0b607a3487ece2a0f641f49ff` | `a41bb6c4d56dd44579a692dadf2cb5b5407e87e78284b4f16a093d86135f6002` |

The committed focused runtime saw all 270 nodes, ended `270 passed in 12.34s`,
had zero non-passing nodes, and returned exit zero. The related structured
output, task-canary, discovery-cache, auth-driver, and API-key owners ended
`101 passed`. `py_compile`, `git diff --check`, and all 37 protected tuple
checks passed. Final collection is byte-identical to Task 3, proving Task 4
changed behavior without changing node identity.

All writes in tests used temporary paths. Post-run inspection found both
worktree `data/` and `src/data/` absent; no profile DB, token store, keyring,
provider request, model turn, production data, scheduler, frontend, or Tranche
B owner was touched.

## 22. Task 4 disposition

Task 4 is complete at `610d3471`. Refresh/login/delete integration and exact
account-snapshot invalidation match the reviewed plan, full/focused collection
identities remain exact, and the committed focused runtime is fully GREEN.
Task 5 passive Claude `RateLimitEvent` capture and every later task remain
unstarted and unauthorized until independent Task 4 review.

## 23. Task 5 RED

The one broad ignore node evolved into exactly the three reviewed IDs. Before
product implementation, the exact three-node run ended `2 failed / 1 passed`:
the no-event/no-probe contract was already GREEN, while both event-persistence
owners failed only because `read_account_snapshot(...)` returned `None`. There
was no import, fixture, SDK-constructor, SQLite, provider, or environment
failure. Full/focused collect-only already matched the final reviewed identities
`4607/5180502f...` and `272/6c706f9d...`, proving the required `+3/-1` ledger.

The RED fixtures used real installed SDK dataclasses but a local fake `query()`.
They prescribed a typed five-hour observation, no-create absence, and raw
account/token/email/UUID/session/nested sentinels that must not survive into the
stored model.

## 24. Task 5 implementation

Product/test family `296024b988bdf28940fc7752ec5b99c1576269e0`
implements:

- passive handling of SDK `RateLimitEvent` while ordinary `StreamEvent` and all
  unrelated messages remain non-terminal ignores;
- closed status and window types, bounded utilization, integer reset times, and
  a bounded stable-code overage reason, without reading `RateLimitInfo.raw`,
  event UUID, or session id;
- a credential-bound SHA-256 fingerprint, required provider/auth/observation
  metadata, latest-only profile-state persistence, and an empty typed usage
  summary rather than invented token activity;
- no-create absence: a stream without a rate-limit event performs exactly one
  fake model request and creates neither the DB nor its parent directory; and
- factory injection of the existing local observation store without changing
  provider selection or execution transport. Telemetry validation/write
  failure never fails the user's model request and never replaces absence with
  zero.

Post-commit source identities are:

| Path | Lines | SHA-256 |
|---|---:|---|
| `src/auth_drivers/oauth_status.py` | 600 | `6b8bd7c9b726b4e53a90ff50bb91a5a80807d0cf1c74230ce1ece1d0a278ef1c` |
| `src/auth_drivers/claude_code_sdk_driver.py` | 849 | `cb04433a8db12402b56120aa3432ea7ed0a1615262e29e1c2ca53f7652052ead` |
| `src/auth_drivers/factory.py` | 160 | `c73d08e05092cc8864d7c43ada115842f9475c9eb2433bf3dbe3c8ba4ac10409` |
| `tests/test_claude_code_sdk_driver.py` | 1,025 | `5f1cdbf4902890b4d1fe61573377bf0334c7f51691a2a9c4ceb0259f06f77c2d` |
| `tests/test_auth_factory.py` | 183 | `b27e8faf183e6064dd6e34545b232fca9aa85ffc639611939d17980e606ad407` |

## 25. Task 5 verification

| Collection/run | Nodes | Node-stream SHA-256 | Report SHA-256 | Transcript SHA-256 |
|---|---:|---|---|---|
| backend full collect-only | 4,607 | `5180502f7dfe577ca758db5fb8ebdfe9ca282730a8976adcf65b7ab19c1c2d74` | `cd82864fb24cd088e1a4e863a42cf9084def3ca0c5a87a0baac82b9b420a797e` | `afed8c8eed13b4b633e935503e20fe6bbcc69ffdc0fd56ed0009525e7e4f3402` |
| backend focused collect-only | 272 | `6c706f9d524ba65adc9b143479c0477516a2f2bd16a766a28ef27a46f2a8c4a4` | `7c8c76b21d5250049aaebdaf090529dedac403994d48fe50477dfddb676650f8` | `b3b085f7c3d6102ca96d08217322f44231be9d3d5872a4880fb0e701314bff00` |
| backend focused runtime | 272 | same focused stream | `cfc220a80c01698a681ee3496150edeebb05f20837a790d2a664cc132de14608` | `0afc4b11c217bfa8b2b74a376d1532ed77df0999bbbc5189cb040cef095b1382` |

The committed focused runtime ended `272 passed in 12.75s`, with all 272 nodes
seen, zero non-passing nodes, and exit zero. Its exact file set was:

```text
tests/test_model_credentials_characterization.py
tests/test_chatgpt_oauth_login.py
tests/test_chatgpt_oauth_driver.py
tests/test_chatgpt_oauth_manager.py
tests/test_chatgpt_oauth_routes.py
tests/test_oauth_import_route.py
tests/test_claude_code_sdk_driver.py
tests/test_credential_env_routes.py
tests/test_oauth_lifecycle_status.py
tests/test_oauth_cross_process_lock.py
tests/test_subscription_account_usage.py
```

The exact pytest invocation used those eleven paths in the displayed order,
with `PYTHONPATH=/tmp/eir002-green-baseline:${PYTHONPATH:-}` and
`PRICE_TRUTH_TIER_REPORT=/tmp/arkscope-oauth-task5-post.AyW8Q7/focused-runtime.json`
exported before:

```text
python -m pytest -q -p arkscope_eir002_reporter \
  tests/test_model_credentials_characterization.py \
  tests/test_chatgpt_oauth_login.py \
  tests/test_chatgpt_oauth_driver.py \
  tests/test_chatgpt_oauth_manager.py \
  tests/test_chatgpt_oauth_routes.py \
  tests/test_oauth_import_route.py \
  tests/test_claude_code_sdk_driver.py \
  tests/test_credential_env_routes.py \
  tests/test_oauth_lifecycle_status.py \
  tests/test_oauth_cross_process_lock.py \
  tests/test_subscription_account_usage.py
```

The shared DTO/driver collateral command was exactly:

```text
python -m pytest -q \
  tests/test_claude_code_sdk_driver.py \
  tests/test_auth_factory.py \
  tests/test_subscription_account_usage.py \
  tests/test_oauth_lifecycle_status.py
```

It ended `91 passed`. `py_compile`, `git diff --check`, and all 37 protected
path/blob/SHA/size tuples passed. Tests used only temporary observation DBs and
fake SDK streams. No live provider request, real token store/profile DB,
keyring, production data, scheduler, frontend, or Tranche B owner was touched.

## 26. Task 5 disposition

Task 5 is complete at `296024b9`. Passive Claude quota evidence now updates the
same bounded credential snapshot only when a normal stream already supplies an
event; missing evidence remains unknown and causes no probe. Exact final backend
identities match the reviewed plan. Task 6 Provider Settings UI remains
unstarted and unauthorized until independent Task 5 review.

## 27. Task 6 RED

The seven reviewed frontend nodes were added under the exact describe owner and
the focused collection immediately matched `15/887a712a...`. An initial harness
attempt used a `Date.now()`-based `waitFor` while the cooldown owner installed a
fake system clock; those timeouts were rejected as wrong-RED and the shared test
waiter was corrected to use monotonic `performance.now()` before product code
changed.

The accepted RED ended `8 passed / 7 failed`. All seven new nodes failed only
because lifecycle-specific status, account observations, visible stale sync,
manual cooldown, and exact mutation invalidation did not exist. There was no
import, fixture, browser, provider, secret, token-store, profile-DB, or path
failure.

## 28. Task 6 implementation

Frontend/i18n family `bc1c79d139a39fcb489036b54c7beb71790cba48`
implements:

- closed lifecycle, account-snapshot, rate-limit, usage, cached-read, and
  explicit-sync DTOs in the API client;
- lifecycle-specific credential and provider status, while legacy responses
  without the new field retain the previous `available` display contract;
- direct used/reset/status/overage/source/observation facts, explicitly labeled
  inferred remaining percentage, and `Unknown` for absent evidence;
- cached reads for active OAuth credentials, automatic ChatGPT sync only for a
  visible stale/missing observation, a five-minute TTL, focus recheck, one local
  in-flight request, a ten-second manual cooldown, and no interval poll;
- credential-bound response checks plus generation guards so stale in-flight
  reads cannot repopulate a mutated credential;
- exact invalidation after add/import/login/manual completion/activate/update/
  delete, followed by inventory refresh and, for an active OAuth result, an
  affected-credential cached read; and
- matching English/Traditional Chinese copy plus reviewed i18n inventory and
  lifecycle-aware CSS source-contract evolution.

Post-commit source identities are:

| Path | Lines | SHA-256 |
|---|---:|---|
| `apps/arkscope-web/src/ProviderSection.test.ts` | 763 | `5f0f87deed3cb6b3ec81e271487d9b7104cc557d20a321e303b4a9c71da28cc2` |
| `apps/arkscope-web/src/api.ts` | 2,904 | `d426950f15b560bdbe15ba72a2d8724ef7eb241afa7ba906de960e1774b51017` |
| `apps/arkscope-web/src/i18n/resources.test.ts` | 1,182 | `df3cf136fee12c10cd69acff4558b696d076963ab2f02c1af28882df30be0a03` |
| `apps/arkscope-web/src/i18n/resources/en/settings.ts` | 1,027 | `9844a82c1c3f86de00750600361977de0f75b04ead7778146da548c12839fce1` |
| `apps/arkscope-web/src/i18n/resources/zh-Hant/settings.ts` | 1,026 | `6efe230246784de2717a6106300f82808f25e68d332e156898dcf858e1d8e3d7` |
| `apps/arkscope-web/src/settings/ProviderSection.tsx` | 1,698 | `03f14d41421389d34c13465d6fa0323435bd8dabab79ae673222212d94b46606` |
| `apps/arkscope-web/src/shell/ShellCss.test.ts` | 163 | `4f1bdc7145771db58f64e68806d37752d8904fb0f4e19fe10859078981b09394` |

## 29. Task 6 verification

| Collection/run | Nodes | Node-stream SHA-256 | Transcript SHA-256 |
|---|---:|---|---|
| ProviderSection focused collect-only | 15 | `887a712a206a272d6db3e75c55a1d77ea2bfe032650186458a874c8495fe04bf` | n/a |
| ProviderSection focused runtime | 15 | same focused stream | `a58b51809b67f44b05e66b85cbfc25e9e6f137e27445bc6a128014c26661620a` |
| frontend full collect-only | 1,084 | `f0e5ecda1371f0559c1cc92af367b7e32daa91663ef2f316ed67e23129ee9637` | n/a |
| frontend full runtime | 1,084 | same full stream | `83279825af198f78c807aea3a7d08681c8c79647456183471ae3d1c88533264a` |

The isolated focused run ended `15 passed in 3.32s`. The single admitted full
run ended `97 files / 1,084 passed in 49.65s`. The ProviderSection plus the two
evolved collateral owners ended `40 passed`. Typecheck, production build, and
the visible-literal scanner all returned exit zero; the scanner reported
`36` candidates, `20` signatures, zero debt signatures, and `20` allowlist
entries. Build emitted only the existing large-chunk warning.

The first full-runtime command was accidentally launched twice concurrently
and is not admission evidence. One transcript (`ce3a8f9c...`) exposed the two
deterministic stale contract owners: exact i18n inventory counts had to grow by
the reviewed 27 provider leaves, and `ShellCss.test.ts` still matched the old
literal `cred.available` expression. Its unrelated foundation timeout was
concurrency noise. Both owners passed in isolation, all duplicate Vitest
processes exited, and the later single-process full run above is the only
admitted full result.

The three frozen Settings contracts remained byte-identical:

| Path | SHA-256 |
|---|---|
| `apps/arkscope-web/src/settings/settingsCopy.test.ts` | `00babecf33c522dd32476a49cd1c439d7f85ac5991d5b49aebf24c650d401e00` |
| `apps/arkscope-web/src/settings/settingsRegistry.test.ts` | `b9ad9aef50d464ed7b7e6ecd0a9e4348dafb55eca2337e263e654c93221d7044` |
| `apps/arkscope-web/src/SettingsCss.test.ts` | `fc3e7b831b7deccfcce699172933071bde12eb5d9ddd91b44fa2210c4bbb456d` |

Tests used only mocked `fetch` responses and synthetic credential/account
identities. No live provider, real credential/token store/profile database,
production data, scheduler, backend product, model registry/routing, or
Tranche B owner was touched.

## 30. Task 6 disposition

Task 6 is complete at `bc1c79d1`. Provider Settings now renders the backend
lifecycle and bounded account observations without inventing availability,
remaining quota, or refresh causes, and its automatic sync is visible/stale
only. Exact final frontend identities match the reviewed plan. Task 7 mutation,
native admission, merge, and closeout remain unstarted and unauthorized until
independent Task 6 review.

## 31. Task 7 mutation sensitivity

Independent Task 6 review authorized Task 7. Each final mutation changed only
the named semantic, ran only its owning node, turned that node RED, restored the
exact pre-mutation owner SHA, and then returned GREEN:

| ID | Diff SHA-256 | Observed RED |
|---|---|---|
| M1 | `9a44b12bd636087ab34644664b30720cab2f4a3ff2332c424670ef64c57aecab` | expired OAuth was again projected available |
| M2 | `f934ade6e2c7851d2cd7631711de50eae1901bfc7c6c3417a6d4e34867c49004` | ChatGPT DB expiry changed `refresh_required` to `ready` |
| M3 | `48db0fa0a0458798aaac8c860abdc7641ae8ccce7a3110c6a0b06ab155b0092f` | retryable refresh failure collapsed to `reauth_required` |
| M4 | `def28106c7b4436b331fb9ac229b1164b9d5626b31401ce0691eb5607994a92b` | refreshable active OAuth became runtime-unresolvable |
| M5 | `5f27f0c96ff2a9f933a923a51af9ee5c09f3979bbe723823fcb939dce2910c7d` | two processes no longer consumed the rotating grant once |
| M6 | `5d5eea5b52ae695ed1d4bb2ff7b7948e5c8e28455f5dad0174325ee08c768e79` | account mismatch was admitted instead of failed |
| M7 | `9fa590c633635dedfb804891f690da2d7b273b007aecfd666ad22e0cedcf9c88` | a protocol failure discarded the last-good snapshot |
| M8 | `6161bd9ad4f3e53b82b62f8f1bfcd43f919aba591895ac1a0b30fef8e588f211` | passive Claude rate-limit evidence was ignored |
| M9 | `db666fce6e3d5bb546f05363de6be0a77ce26d17e3e30e53284f9d250afc6aea` | the visible stale sync cadence no longer stayed once-within-TTL |
| M10 | `f8dcc96ec2b1427478765cf30bce2a75d71ef1f868e611d54daa8817dc43e84c` | one mutation invalidated another credential's cache |

M3's first context-free inverse selected a second identical enum occurrence;
the owner SHA check rejected it before M4, and the contextual inverse restored
the exact pristine SHA. Two candidate M9 mutations were also rejected because
the owning node remained GREEN: dead-condition diff `6ab84492...` and hidden-
section diff `0d73e226...`. The admitted M9 directly replaced the TTL semantic
with zero and made the owning cadence assertion RED. These rejected attempts
are retained rather than presented as mutation evidence.

All raw diffs and RED/GREEN transcripts are under
`/tmp/arkscope-oauth-task7-6bba8307/mutations/`. The 72-entry review-packet
checksum manifest is
`/tmp/arkscope-oauth-task7-6bba8307/TASK7_SHA256SUMS`, SHA-256
`71389349d59a7cdd124b2876841cf9200e1505192dcbec050b3c261a7d909f60`.

## 32. Task 7 final identities and gates

All four final collections were rebuilt after exact mutation restoration:

| Collection | Nodes | Node-stream SHA-256 |
|---|---:|---|
| backend full | 4,607 | `5180502f7dfe577ca758db5fb8ebdfe9ca282730a8976adcf65b7ab19c1c2d74` |
| backend focused | 272 | `6c706f9d524ba65adc9b143479c0477516a2f2bd16a766a28ef27a46f2a8c4a4` |
| frontend full | 1,084 | `f0e5ecda1371f0559c1cc92af367b7e32daa91663ef2f316ed67e23129ee9637` |
| frontend ProviderSection | 15 | `887a712a206a272d6db3e75c55a1d77ea2bfe032650186458a874c8495fe04bf` |

The runtime and static gates were:

| Gate | Result | Evidence SHA-256 |
|---|---|---|
| exact backend focused | `272 passed in 12.10s` | report `cfc220a8...`; transcript `951c8cb0...` |
| task-canary/discovery collateral | `168 passed in 3.74s` | `d229229f...` |
| ProviderSection focused | `15 passed in 3.00s` | `29100591...` |
| frontend full | `97 files / 1,084 passed in 48.27s` | `ed179460...` |
| TypeScript typecheck | exit `0` | `f11cc6b0...` |
| production build | exit `0`; existing large-chunk warning only | `48169e56...` |
| visible-literal scanner | `36 / 20 / 0 / 20`, exit `0` | `dd94c589...` |

All nine backend product owners compiled. `git diff --check 0753947e..HEAD`
was clean. New frontend product paths contained no raw token/account-id field
names. The current 37 protected path/blob/SHA/size tuples reproduced Task 0
byte-for-byte at `bcc7bf54c6e82e39a01c2e98dc9677640605580fa61eba4eba0b3bf39a084e65`.
All OAuth-owned mutation/focused/frontend gates used local fakes and touched no
live provider, real token store/profile DB, keyring, production data,
scheduler, model-routing policy, or Tranche B owner.

## 33. Task 7 native canonical admission

A fresh detached worktree at exact tip
`6bba8307a5bb328da2111a4a15b33b761104d70b` used no `config/.env`, an empty
`data/`, absent `src/data`, and only the pinned `node_modules` symlink. The
pinned wrapper and wakeup probe completed the entire suite:

```text
collected: 4607
seen: 4607
passed: 4535
skipped: 72
failed/errors/non-passing: 0/0/0
exit: 0
duration: 307.31s
```

The reporter is `0f450948...`, transcript `df2d3cd0...`, collected stream
`5180502f...`, and empty non-passing stream `e3b0c442...`. The full collected
stream is byte-identical to the final backend authority.

This global canonical run is not described as network-free. It inherits seven
pre-existing `tests/test_yfinance.py` smoke nodes that call public Yahoo data
paths and return booleans rather than assertions; the isolated runtime produced
`tmp/cookies.db` and ticker-timezone cache entries for AAPL/MSFT/GOOGL/AMZN/
META/IBM and treasury symbols. They use no credential or metered account and
are unrelated to the OAuth delta, but their public-network behavior is an
existing test-hygiene fact. The no-live-provider claim above is deliberately
limited to this slice's owned gates.

The first ordinary `git status` in that fresh checkout correctly refused
because the detached worktree had no git-crypt key. It ran no pytest and made
no mutation. Every subsequent status command used the reviewed no-op git-crypt
filters; the admission itself remained unchanged.

The run created exactly 587 files: 564 bytecode files, 15 scratchpad JSONL
files, three JSON files, one native-host log, and four pytest-cache files.
Every exact relative path received inode/size/mode/mtime/SHA evidence and was
moved to `/tmp/arkscope-oauth-task7-6bba8307/native-quarantine/files/`.
The path stream is `4c2bb67e...`; the pre/quarantine 587-row metadata manifests
are byte-identical at `9b814d8c...`. Ordinary status, ignored status, symlinks,
and `data`/`src/data` inventories returned byte-for-byte to their pre-run
states; empty `data/` remained and `src/data` remained absent. The restored
worktree was then removed normally.

## 34. Task 7 disposition

Task 7 verification is complete at `6bba8307`. The OAuth product tip remains
`bc1c79d1`; all later commits through this packet are test/evidence family
commits already covered by the exact gates above. OAuth lifecycle truth,
cross-process refresh serialization, bounded account usage, passive Claude
quota evidence, and Provider Settings rendering are ready for independent
implementation review. Fast-forward merge, fresh exact-master admission, and
docs-only closeout remain explicitly unauthorized until that review returns
GREEN.

## 35. Independent review and fast-forward merge

Independent implementation review returned GREEN for
`6bba8307..02f1e588`. The reviewer independently reproduced all 4,607 backend
nodes, `4535 passed / 72 skipped / 0 failed`, all four final collection
identities, the 72-entry packet checksum manifest, every mutation class, and
the 37 protected tuples. The review explicitly accepted the bounded yfinance
disclosure as pre-existing canonical-suite behavior rather than OAuth behavior.

The main worktree was clean at `7257699171a81294b74ff8cde61fb90bb065a2b4`.
That commit was a strict ancestor of reviewed tip
`02f1e588c6f9d91d4710627de3699a821e0bda6f`; the 16-commit range contained no
merge commit. `git merge --ff-only codex/oauth-lifecycle-quota-truth` advanced
`master` exactly to the reviewed tip. No push occurred; `origin/master`
remained `fd6d1b86383df2a98f97b235d9796d4bcaaa7a58`.

## 36. Fresh exact-master verification

A new detached worktree at exact merged tip `02f1e588` again used no
`config/.env`, an existing empty `data/`, absent `src/data`, and only the pinned
`node_modules` link. The unchanged wrapper, reporter, wakeup probe, package
lock, and Node v22.14.0 identities all matched before execution. The new
single-use stage `oauth-merged-02f1e588-full` completed:

```text
collected: 4607
seen: 4607
passed: 4535
skipped: 72
failed/errors/non-passing: 0/0/0
exit: 0
duration: 302.36s
```

The merged reporter JSON is byte-identical to the branch-side report at
`0f4509485f84c825aa453974a3e9b6598c7057818cf5d6365595dacb0f17bded`.
The timestamp-bearing transcript is `00e1300c66e83127d003bbe301f4795048aadb889333c366c1b80efa748aa2f0`.
The collected stream remains `5180502f...`, byte-identical to the reviewed
target, and the non-passing stream remains empty at `e3b0c442...`.

The run produced exactly 587 repository-relative files. Their exact path stream
is `c5b6851e...`; the pre/quarantine inode/size/mode/mtime/SHA manifests are
byte-identical at `6132f8a3...`. Ordinary status, ignored status, symlink
inventory, empty `data/`, and absent `src/data` all returned byte-for-byte to
their pre-run states before the detached worktree was removed. The 19-entry
merged-evidence checksum manifest is
`/tmp/arkscope-oauth-merged-02f1e588-evidence/SHA256SUMS`, SHA-256
`8da95b35111a7ca321161f041e97c0a693af9f1bbbbd1eb4f0ba94f103a99f4a`.

The merged warning summary independently re-exposed the seven pre-existing
boolean-returning yfinance smoke nodes and six analogous Tiingo nodes. This does
not change OAuth admission: all OAuth-owned gates remain fake/local and
provider-free. The user has separately ruled that the yfinance remnants should
retire; that cleanup is intentionally not mixed into this reviewed OAuth
lineage.

OAuth implementation and merged verification are complete. The canonical
backend baseline is now `4607 collected / 4535 passed / 72 skipped / 0 failed`.
This docs-only checkpoint stops for focused closeout review; no branch deletion,
push, yfinance retirement, or Tranche B rebase is included.
