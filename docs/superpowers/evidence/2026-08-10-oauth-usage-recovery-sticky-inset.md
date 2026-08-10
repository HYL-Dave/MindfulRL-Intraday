# OAuth Usage Recovery and Sticky Inset Evidence

> **Status:** TASK 2 COMPLETE - INDEPENDENT CODEX REVIEW REQUIRED
>
> **Date:** 2026-08-10
>
> **Plan authority:** `953ea7e7` (plan GREEN by Codex review; plan file
> SHA-256 `94d18b33c391a2ad2f737eed4d5bb97bdd800a27866c997b6d888cf284a89975`)
>
> **Product grounding base:** `8cf85597` (= `master`)
>
> **Task 0 artifact root:** `/tmp/oauth-usage-sticky-impl-task0-953ea7e7`
>
> **Implementer:** Fable (design LD 11); reviewer: Codex.

Task 0 changed no product or test byte. Tasks 1-7 remain unstarted and
unauthorized until this re-grounding receives independent GREEN review.

## 1. Boundary and toolchain

- Branch ancestry: six docs-only commits from `8cf85597` to `953ea7e7`
  (`feb8403d`, `70f86bb9`, `1c6774fd`, `0f7c0db7`, `9c22b696`, `953ea7e7`);
  merge-base with master is exactly `8cf85597`; zero non-docs paths in the
  range; worktree clean before and after.
- Product drift since the native-control tip `8ebf7fae`: zero non-docs
  paths. The canonical native control therefore carries:
  `4,253 passed / 29 skipped / 0 failed`, reporter JSON `252535bf...`
  (byte-identical three ways at Tranche B closeout).
- Toolchain pins reproduced by full SHA (packet `toolchain.txt`):
  `package-lock 5322cb03...`, `.package-lock 4dd5182f...`, normalizer
  `955dca59...`, reporter `09d2bc52...`, wrapper `e7c963f1...`, Node
  v22.14.0, Vite 5.4.21, Vitest 4.1.8, anthropic SDK 0.120.2, Playwright
  1.58.0, Chrome 150.0.7871.128.
- Protected boundary: all 17 unconditional blobs plus `api.ts` (bounded to
  the single authorized union line) are identical to `8cf85597` in the
  implementation worktree (packet `protected-blobs.tsv`, 18 rows).

## 2. Re-collected base identities

| Stream | Result |
|---|---|
| backend collect (`pytest --collect-only -q`, sorted) | `4,282 / 281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |
| frontend decoded list (pinned normalizer) | `99 files / 1,124 / da69a2942c03e4794e3384e6125936f9f25c1fafbad7d006b67025f8fd97bc39` |

## 3. Staged identities rebuilt from the committed plan text

The five addition blocks were extracted verbatim from the committed plan
(`awk` over the fenced lists; packet `add_t1.nodes` .. `add_t4.rows`) and
applied to the re-collected bases. Every predicted identity reproduced:

| Stage | Result |
|---|---|
| backend stage 1 | `4,289 / 37bc0a597398404de6247e465e44908ccd265798ba66722242bb8807c1614968` |
| backend stage 2 (final) | `4,303 / 52b862d7bf94f9d4605f8de1b2e92240ea152a41218446c3652b38716af77489` |
| frontend stage 3 | `1,132 / 778d64be3239dbb94df475e2cccde1b61878af3a627a28a677038191ea6a6e9d` |
| frontend stage 4 (final) | `1,134 / 941067a028c7bb6b15c3e3f64012dcf251995804e3f55c9a712cb230d4a4ba64` |
| 21-node backend addition stream | `2b540253de6578a71be09a726a11d29cce396a2e0c29421a7f8a5cfa4b3666bd`; all 21 absent from base |
| backend focused base / s1 / s2 | `61 b0d56cc5...` / `68 a76b86a3...` / `82 1c8c9de1...` |
| frontend focused base / s3 / s4 | `33 fb42f09a...` / `41 efc6accc...` / `43 853c9cef...` |
| Settings 15-file projection base / s4 | `221 a2c20d36...` / `231 e0bb6190...` |

## 4. Focused runtime baselines (one command each)

```text
backend (4-file set, existing 3 files): 61 passed in 3.82s / exit 0
frontend (3-file set):                  33 passed (3 files) / exit 0
```

## 5. Handoff

Independent Codex review of this packet authorizes Task 1 (Codex launcher
repair, `+7` RED-first). The packet root carries every raw stream, list
output, transcript, and `SHA256SUMS`.

## 6. Task 1 - Codex launcher repair (product commit `c04f58d0`)

RED at stage-1 identity `4289/37bc0a59...` (packet `red-collection.nodes`):
all seven new nodes failed on the intended collapsed classification at the
old `codex_account_usage.py:520` seam (`red-run-full.txt`); the nine existing
nodes were untouched.

Implementation per plan §1.1: `_resolve_launcher_and_target()` preserves the
which()/explicit launcher (symlinks intact) and resolves the target for
inspection only; `_isolated_environment(launcher, target, codex_home)` puts
the launcher directory first and the resolved target directory second on the
isolated `PATH`; `_require_shebang_interpreter()` reads a bounded 4,096-byte
first line, accepts only the closed interpreter-name syntax, and returns
typed `interpreter_unavailable` before any spawn; `_verify_version()` now
classifies exactly: non-zero exit -> `adapter_unavailable`, oversized
stdout/stderr -> `protocol_incompatible`, malformed output (not
`codex-cli N.N.N`) -> `protocol_incompatible`, well-formed different version
-> `version_incompatible`. The app-server spawn uses the launcher argv[0]
and the same environment. The shared test fixture additionally records
`argv0`/`PATH` per app-server message and a `version-ran.marker`, giving the
launcher-path and ran-cleanly assertions real witnesses.

GREEN evidence (packet `/tmp/oauth-usage-sticky-impl-task1-953ea7e7`,
manifest `93b660d161a730d26ecab54a9fd6e4762979f78957a577270623ba1174aaf6c7`):

```text
owned file:        16 passed (9 existing + 7 new)
focused (3 files): 68 passed / exit 0
full collection:   4,289 / 37bc0a59... (== stage 1)
protected blobs:   18/18 identical
worktree delta:    exactly the two owned paths
```

Task 2 (Anthropic adapter + dispatch, `+14`) awaits independent Task 1
review.

## 7. Task 2 - Anthropic adapter and dispatch (product commit `ec51bd93`)

RED at stage-2 identity `4303/52b862d7...` (packet `red-collection.nodes`):
the twelve new-file nodes failed on module absence inside test bodies
(collection stayed intact), and both dispatch nodes failed on the missing
`anthropic_adapter` service surface (`red-anthropic.txt`,
`red-dispatch.txt`).

Implementation per plan §1.2: new `src/auth_drivers/anthropic_account_usage.py`
(pinned probe shape `claude-sonnet-5` / `max_tokens=8` / OAuth beta
`oauth-2025-04-20` / identity block / one fixed message; injectable client
factory; default factory `Anthropic(auth_token=..., api_key=None,
timeout=20.0, max_retries=0)`); unified-header parsing with typed validators
(utilization finite `[0,1]` -> `round(x*100, 4)` used-percent, absolute Unix
resets, closed status values, bounded reason/claim patterns,
`fallback-percentage` ignored); closed error vocabulary
(`missing_token`, `sdk_incompatible`, `provider_auth_rejected`,
`provider_access_rejected`, `quota_headers_unavailable`,
`provider_request_rejected`, `timeout`, `transport_error`,
`adapter_unavailable`); 429-with-valid-headers is an observation, 429
without them is `quota_headers_unavailable`. `oauth_status.py`'s source
`Literal` gained exactly `anthropic_oauth_probe`; `OAuthAccountSyncService`
gained `anthropic_adapter` and explicit provider/auth-mode dispatch under the
same lock/single-flight/generation flow; every other pair remains
`unsupported_auth_mode`. The declared section 2.4 backend evolution replaced
the dispatched `claude_code_oauth` unsupported-example with `api_key`
(`subscription-tests.diff`).

GREEN evidence (packet `/tmp/oauth-usage-sticky-impl-task2-c04f58d0`,
manifest `891b4a2e718fc221d13d613cb8b46297fc86336ff3e6dda07f37383e71669b6f`):

```text
adapter file:        12 passed
subscription file:   18 passed (16 + 2 dispatch, incl. the evolved node)
focused (4 files):   82 passed / exit 0
full collection:     4,303 / 52b862d7... (== stage 2)
network guard:       12 passed with socket.connect denied (zero provider use)
protected blobs:     18/18 identical
worktree delta:      exactly the five owned paths
```

One implementation note: raw `utilization * 100` produced
`14.000000000000002` for `0.14`; the adapter rounds the stored used-percent
to four decimals so persisted truth matches header precision.

Task 3 (frontend recovery split, `+8`) awaits independent Task 2 review.

### 7.1 Task 2 review findings F1-F3 fixed (commit `42caff78`)

Codex review found three blocking defects; all confirmed and fixed:

- **F1 (dead seam):** the adapter rejected a missing `observed_at` while the
  service never passes one — every real sync would have been
  `adapter_unavailable` with zero requests, hidden by a hand-rolled fake
  adapter in the dispatch test (the seam-mock lesson again). The adapter now
  self-stamps a UTC receipt time when none is provided, and the dispatch
  node exercises the REAL adapter with a fake raw client through the real
  service call (which passes no `observed_at`; asserted via source
  inspection in the packet's `green-fix` run).
- **F2 (fake success):** empty or partial unified headers on 2xx produced an
  all-unknown `status=available` snapshot, and a 429 with `status=allowed`
  was admitted. Admission is now fail-closed for both paths
  (`_admit_quota_snapshot`): the core (overall status + both windows'
  utilization and reset) must parse, and 429 additionally requires
  `status=rejected`; auxiliary fields (overage status/reason, claim) stay
  None when malformed. The test helper's falsy-`{}` header bug is fixed,
  and the 2xx/429/malformed nodes gained the fail-closed subcases
  (core-malformed -> `quota_headers_unavailable`, never zero).
- **F3 (client leak):** the Anthropic client is now closed in a `finally`
  on every path (success, `sdk_incompatible`, all typed failures); close
  witnesses were added to the request-shape, `sdk_incompatible`, and
  dispatch nodes.

Post-fix verification: adapter+subscription `30 passed`, focused 4-file
`82 passed`, collection unchanged at `4,303/52b862d7...`, network guard
`12 passed` under socket-connect denial, protected blobs `18/18`, updated
packet manifest `639b70bfa1651728799446f7ae0c93b5d4d999d7fd3f1b78a99d66d128ac9ef8`.

### 7.2 Re-review findings fixed (commit `2dde7790`)

- **Receipt time is now receipt time:** the UTC stamp moved from
  pre-request to inside `_observation`, i.e. only after the unified headers
  were received and admitted; the real-adapter dispatch node now asserts the
  stored `observed_at` lies within the sync-call boundary (seconds grain,
  matching `timespec="seconds"`).
- **Admission relaxed to the authority contract:** fail-closed applies only
  to the unified authority (overall `-status` present/valid; 429 must say
  `rejected`). Individual window utilization/reset fields are nullable again
  (absent/malformed -> None, never zero, observation retained), matching
  plan §1.2 and design LD 7. The 2xx node now proves a partial-window
  response records an observation with the missing leg unknown, and the
  malformed node splits window-malformed (nulled) from authority-malformed
  (`quota_headers_unavailable`).

Post-fix: focused 4-file `82 passed`, network guard green, collection
`4,303/52b862d7...` unchanged, protected `18/18`, packet manifest `343c2d37fdd704ab1fcdd1ea8efeffecf2a47037e8725c9abca85c4223642024`.
