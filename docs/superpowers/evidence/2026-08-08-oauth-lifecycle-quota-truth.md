# OAuth Lifecycle and Subscription Usage Truth Evidence

> **Status:** TASK 0 COMPLETE; INDEPENDENT REVIEW REQUIRED BEFORE TASK 1
>
> **Date:** 2026-08-08
>
> **Plan-review tip:** `0753947e049a8ecabeab5220f4d3427eeb256a65`
>
> **Grounding base:** `7257699171a81294b74ff8cde61fb90bb065a2b4`

## 1. Scope and boundary

Task 0 performed collection, one focused offline runtime, and static identity
checks only. It changed no product or test file, contacted no provider, read no
real token store/profile database, and made no production-data or scheduler
change. Task 1 product implementation has not started.

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

Task 0 is complete. Independent review is the sole next gate. Task 1 lifecycle
tests/product code, every later task, merge, and live synchronization remain
unauthorized.
