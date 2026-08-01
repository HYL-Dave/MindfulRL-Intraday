# EIR-002 Green Backend Baseline Evidence

> **Status:** TASK 7 BLOCKED - BOUNDED AMENDMENT REVIEW NEXT
>
> **Date:** 2026-07-31
>
> **Design:** `20d4e7e2`
>
> **Reviewed plan tip:** `dd334506`

## 1. Grounding

- Branch/worktree: `codex/eir-002-green-backend-baseline` at
  `/tmp/arkscope-eir-002`; the worktree was clean before and after runtime
  grounding.
- Plan: `docs/superpowers/plans/2026-07-31-eir-002-green-backend-baseline.md`
  at SHA-256
  `26e67e579802853d9a2fe72f5a6cb877b23087941a5a0d3b1a1498c3f5834dfd`.
- Canonical collection:
  `/tmp/eir002-green-baseline/base-full.nodes` =
  `4739 / a72bbd36dfad3d36aee2e6630e6024ec9fb4e910bebaf1363d44df8a1aa204dd`.
- Focused collection:
  `/tmp/eir002-green-baseline/base-focused.nodes` =
  `132 / 76f8f087a24f2ff2934274cbbd1711d203c9dbe7056ba4bf5d6022b2d1a03f9c`.
- Target identities were constructed before any test edit by subtracting the
  exact nine approved retired node IDs. Canonical target =
  `4730 / c34de9a0fe53e400409d3ec26d75a8c907ee277b121279693ea9f69c8638aabb`;
  focused target =
  `123 / 37386cd24fca323338fcb0fb2bbe5c42b7c471d2e9fde2e7c5f345b5ce631b8f`.
  Both `comm` proofs showed `+0/-9`, and both removed-node streams were
  byte-equal to `/tmp/eir002-green-baseline/retired.nodes`.
- Reporter:
  `/tmp/eir002-green-baseline/arkscope_eir002_reporter.py` =
  `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928`.
- Native wrapper:
  `/tmp/eir002-green-baseline/run_native.sh` =
  `79 lines / 2353 bytes /
  e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f`.
  The source extracted from the reviewed plan was byte-equal to the executed
  file. It pins Node.js to
  `/home/hyl/.nvm/versions/node/v22.14.0/bin/node` and verifies
  `v22.14.0` before entering `env -i`.
- Native wakeup probe:
  `/tmp/arkscope_asyncio_wakeup_probe.py` =
  `942 bytes /
  10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e`.
  It was byte-equal to the frozen EIR-005 closeout artifact and returned
  `{"callback_fired": true, "ready_count": 0, "wake_bytes": 0}` in the
  same native execution boundary.
- Protected current-shape anchors:
  `pytest -q tests/test_sqlite_backend.py
  tests/test_fundamentals_sec_cache.py tests/test_news_scores.py
  tests/test_db_backend.py` returned `94 passed, 18 skipped`. Transcript:
  `/tmp/eir002-green-baseline/protected-anchors.txt`, SHA-256
  `cb4891efc3e6c7e6417cc841488f51d8f0a4a626f193abe6f6b8d970904208c0`.

## 2. Stage Ledger

| Stage | Collection / seen | Passed | Failed | Skipped | Non-passing SHA-256 |
|---|---:|---:|---:|---:|---|
| `base-full-v2` | 4739 / 4739 | 4640 | 27 | 72 | `7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15` |
| `base-focused` | 132 / 132 | 105 | 27 | 0 | `7aafce5d2cba923480cc1fb6221bce4f5a33e0bf61c06cf94227cafefe227f15` |
| `after-retirement-focused` | 123 / 123 | 105 | 18 | 0 | `567ea435111078f45dee4c818e282997e1d562e72cf2ddc5a5101a09527cd225` |
| `after-news-focused` | 123 / 123 | 112 | 11 | 0 | `e6d59e3ec3e24b3d8ef2d68c341af5dcbb8c3bd0f264e71a130a45713ad8203c` |
| `after-prices-focused` | 123 / 123 | 120 | 3 | 0 | `c072d5df09468496bb8fa26ade78cf38e1846be9b1dbe665502db60ae1e69664` |
| `after-fundamentals-focused` | 123 / 123 | 122 | 1 | 0 | `71b6d959c36e1b7d8e9c92b4904a8e68c46c6c4d7992c0bdc939dc1b220798f0` |
| `final-focused` | 123 / 123 | 123 | 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `final-full` | 4730 / 4730 | 4658 | 0 | 72 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The two reporter `nonpassing_node_ids` streams are byte-equal. The accepted
canonical stream is also byte-equal to the frozen EIR-005 native 27-node
stream. The accepted canonical reporter and transcript are:

- `/tmp/eir002-green-baseline/reports/base-full-v2.json`:
  `cda36b333db66dc90d1faa8bcfa694448490f697f7cd03c67da26cd08ca48403`
- `/tmp/eir002-green-baseline/reports/base-full-v2.txt`:
  `f60f0570fd10a4a0dcc34dfafdc5e9233313638e086ff11dcbee0bf43c05e1d6`

The focused reporter and transcript are:

- `/tmp/eir002-green-baseline/reports/base-focused.json`:
  `5994007c18db8a816c90bce8890abcd8e2a12b837b8b5e344303800c7abf7a92`
- `/tmp/eir002-green-baseline/reports/base-focused.txt`:
  `4ee1cbf6e2733030ff1b52afd9c3f3fffb591f289147f1ef38584257d2773bdd`

## 3. Rejected Attempt And Amendments

The first single-use native stage, `base-full`, is preserved as rejected
evidence and is not a ledger row. Its pre-amendment wrapper omitted the
installed Node.js directory from the explicit `env -i` `PATH`.

- It terminated naturally with `4739 collected / 4739 seen / 81 failed`.
- The expected 27-node set was fully present; no historical node disappeared.
- The exact additional set contained 54 nodes across seven
  `tests/test_sa_extension_*` files, and the transcript contained exactly 54
  `FileNotFoundError: [Errno 2] No such file or directory: 'node'` failures.
- Rejected non-passing stream:
  `e8eeb64e04c85d04b0600c6b07d90fa7e766415f8bd9360ffc0cf5ef906d5c04`.
- Unexpected 54-node stream:
  `a0cda3ee23bd2339edf7c751e93bc9cbdedeaaab7382359e0bf8838bb1cdae20`.
- Historical nodes missing from that attempt: empty stream
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Rejected report and transcript remain at
  `/tmp/eir002-green-baseline/reports/base-full.{json,txt}`; runtime remains
  at `/tmp/eir002-green-baseline/runtime/base-full`.

The earlier probe-identity preflight rejection launched no pytest process and
created no runtime/report stage. Docs-only commits `9075865f` and `dd334506`
corrected the probe lineage and then pinned the Node.js toolchain. No rejected
attempt was reinterpreted as an EIR-002 baseline.

## 4. Repository-Relative Artifact Boundary

Task 0 began with an empty Git status, no files under `data/`, and no
`src/data`. It recorded that each canonical full run created only
`src/data/cache/risk_free_rate.json`; each observed cache path was recorded
and moved to a distinct quarantine:

- Rejected run: inode `90595257`, 73 bytes, SHA-256
  `3a1bee06ad385aee524b17951a0878fc243fb49828020c37edb7c1018498edb9`,
  quarantined under
  `/tmp/eir002-green-baseline/task0-invalid-quarantine/src/data/cache/`.
- Accepted `base-full-v2`: inode `90595258`, 73 bytes, SHA-256
  `63914ef1c90ebe829bd1e6dfe71752effe2949b33fb88e3d5242a566f3c42378`,
  quarantined under
  `/tmp/eir002-green-baseline/task0-base-full-v2-quarantine/src/data/cache/`.

**Task 6 erratum:** the Task 0 statement that those were the only generated
files was false. The narrower Task 0 inventory did not capture
`data/logs/sa_native_host.log`. The stricter Task 6 inventory found a
pre-existing 924-byte file at that path whose entries date from the Task 0
native runs. Therefore Task 0's risk-free-cache observations remain valid but
were not exhaustive. This erratum corrects the evidence; it does not
reinterpret or delete the historical run artifacts.

Task 6 captured an empty pre-run Git status plus all 33 existing files under
`data/` and `src/data`, including path, inode, size, modification time, and
SHA-256. `final-full` then created exactly sixteen new files:

- fifteen `data/agent_scratchpad/*.jsonl` files; and
- `src/data/cache/risk_free_rate.json`, inode `90596688`, 73 bytes,
  SHA-256
  `1f548e4774e70354df87545faf3118c4da2f13b858f43dedc40098e3e2b9c700`.

Their exact metadata and hashes are stored in
`/tmp/eir002-green-baseline/final-quarantine/new-artifacts.manifest`
(`b20b7485adc296db898eb002c43b1ccf1e4c11b2604c37b0d1da138bdfbefd33`).
The pre-move and quarantined path streams are byte-equal at
`8a841cf831de7aa08f2878a9560090f724ba4846190fff264e8b5c4e3526db0e`,
and every quarantined SHA matches its source SHA.

The same run appended 462 bytes to the existing
`data/logs/sa_native_host.log`. The append contains only two run-window ping
request/response pairs. Before any restoration:

- the full 1,386-byte post-run file was preserved at
  `/tmp/eir002-green-baseline/final-quarantine/modified/data/logs/sa_native_host.log.post-run`
  with SHA-256
  `4aa0a174bc73daf6b32c207f17ae7d8e89bd481f54000071956e85021d00b9d6`;
- the exact 462-byte tail was preserved beside it as
  `sa_native_host.log.appended-462`, SHA-256
  `ad1b4e563f457865dcc2881db5f8b6ff816b78f7497ff408c9adc78e34663cb0`;
  and
- the first 924 bytes independently reproduced the pre-run SHA-256
  `83d35003ddbc6b6979c83b9513ee4857d0fb83fbae9a113226614c2995281397`.

With explicit approval, the isolated-worktree log was truncated to its exact
924-byte prefix and its pre-run modification time restored. Its inode
`95031007`, size, time, and SHA now match the pre-run manifest exactly.
After moving all sixteen new files by exact path, the restored Git-status
streams are both empty (`e3b0c442...`), the pre/restored 33-path inventories
are byte-equal (`5bdb41a5...`), and the complete pre/restored metadata/hash
manifests are byte-equal (`ef447109...`).

Task 7 has one pre-authorized production safeguard. Before merged-full, it
must snapshot the existing production `data/logs/sa_native_host.log` size,
metadata, and SHA. Restoration is permitted only when the old file is an exact
prefix and every appended line is a merged-run-window test ping. The full
post-run file and append must be archived before truncation. If any production
native host writes during the window, or the difference is not a pure
test-ping append, the file must be preserved and execution must stop rather
than truncate it.

## 5. RED And Mutation Evidence

Task 0 recorded the pre-existing RED set without changing an assertion,
fixture, product file, test node, skip marker, or mutation. Independent review
then authorized Tasks 1-5.

Task 1 removed exactly the nine approved ambient-data nodes from
`tests/test_data_access.py`:

- the post-retirement canonical collection is
  `4730 / c34de9a0fe53e400409d3ec26d75a8c907ee277b121279693ea9f69c8638aabb`;
- the post-retirement focused collection is
  `123 / 37386cd24fca323338fcb0fb2bbe5c42b7c471d2e9fde2e7c5f345b5ce631b8f`;
- both collection streams are byte-equal to their preconstructed targets;
- both `comm` proofs contain exactly the nine approved removals and no
  additions;
- the retained `tests/test_data_access.py` nodes returned `19 passed`; and
- the native five-file checkpoint returned `105 passed / 18 failed /
  123 collected`, with the exact planned non-passing SHA recorded in the
  stage ledger.

The Task 1 reporter and transcript are:

- `/tmp/eir002-green-baseline/reports/after-retirement-focused.json`:
  `2a8655090dd47ac59a21936f7ec6275ed16b2dd6fc6a302d63faf0b6e828253e`
- `/tmp/eir002-green-baseline/reports/after-retirement-focused.txt`:
  `3e6de85aadd29385d183168bd4aedc92cb09665e45fe570083d1df2ab90ca5dd`

Task 2 first strengthened the seven retained news-consumer assertions without
changing their ambient fixtures. All seven remained RED because the old DAL
returned zero articles rather than the fixed two-row dataset. No import,
SQLite, lifespan, or other environment failure occurred. After the reviewed
hermetic backends were added and only those seven nodes were switched:

- `news-green` returned `7 passed`;
- `after-news-focused` returned `112 passed / 11 failed / 123 collected`;
- its non-passing stream matched the planned
  `e6d59e3ec3e24b3d8ef2d68c341af5dcbb8c3bd0f264e71a130a45713ad8203c`;
  and
- no node was added, removed, renamed, or skipped.

The Task 2 reporters and transcripts are:

- `news-red.json` / `news-red.txt`:
  `c757960fbba75095d7e06cbe6e7ba130818aec7b106f27e26a11e800f61b10a5` /
  `2e71e53d67f07b7f57d1d174d3d027a48e4788c0b55a08cff8a2bcd201b30f47`
- `news-green.json` / `news-green.txt`:
  `bab836d1b073d2f75616f1519d3a37c994b22a240fb1090d00aa289a53eeb4fe` /
  `97d7ad469831d364e8e46d5a452411ddd2ad4c2c8494ce578633bec1bcff1d18`
- `after-news-focused.json` / `after-news-focused.txt`:
  `38fe28606db7ec9f017f790b349b03d17e834372cba76a58cf0fde3b8e39f864` /
  `aa301f7b3e81a14d503ddb685102b8126c4f4df1883fe398d03d8989d80d4b53`

Source-breakdown mutation:

```diff
-        "source": "ibkr",
+        "source": "polygon",
```

With that temporary change in the second NVDA row,
`TestNewsTools.test_get_ticker_news` turned RED because the actual breakdown
became `{"polygon": 2}` instead of `{"polygon": 1, "ibkr": 1}`. The file was
restored to its exact pre-mutation SHA
`f68fead8fe41649b1630dc3f067320e24562cb7b6027ebd9a41f28487fc86137`
and the owning node returned `1 passed`. A first context-free restore attempt
hit the sibling source field; the pre-mutation SHA rejected it, and the
contextual restore returned the exact expected bytes before GREEN admission.

Task 3 likewise strengthened the eight retained price/status assertions before
switching fixtures. All eight remained RED against the ambient fixtures,
which returned zero tickers or bars. After switching only those eight nodes
to the already reviewed hermetic seams:

- `prices-green` returned `8 passed`;
- `after-prices-focused` returned `120 passed / 3 failed / 123 collected`;
- the three remaining nodes were exactly the two fundamentals consumers and
  the app-record round-trip; and
- their non-passing stream matched the planned
  `c072d5df09468496bb8fa26ade78cf38e1846be9b1dbe665502db60ae1e69664`.

The Task 3 reporters and transcripts are:

- `prices-red.json` / `prices-red.txt`:
  `f9b463d3139bb35d6943cc54403d72b0a0737ec37c685e5defc089beccb5f91a` /
  `1994210ea89e612136d7c4cd3dd2fc3cdf78118c2df57f1083cee6d2af211990`
- `prices-green.json` / `prices-green.txt`:
  `5b0fa926ed17f519dd74e53b353fbfe9fe755d36d5c6684f4c0ab33e171aece0` /
  `3240d3b51e55aff1c3a42f99da386d17aa12080e5f7883266defd98307495f74`
- `after-prices-focused.json` / `after-prices-focused.txt`:
  `4cbbe017e5ec34c6d839deba0a232e40d0372f0bef13265411365f847c356e6f` /
  `08e492fc9ca447da6df78174e50632627e8a4898f22997292fedc0baeec4df9c`

Price-math mutation:

```diff
-        ("2026-07-30T00:00:00+0000", 105.0, 112.0, 104.0, 110.0, 1200),
+        ("2026-07-30T00:00:00+0000", 105.0, 112.0, 104.0, 100.0, 1200),
```

With that temporary final-close change,
`TestPriceTools.test_get_price_change` turned RED because the actual
`change_pct` became `0.0` rather than `10.0`. The file was restored to its
exact pre-mutation SHA
`80438ed993f9c9cd22509655dcc54ae30b123a4d0f94673d5d8f1348b0c1e480`,
then the owning node returned `1 passed`.

Task 4 strengthened the two fundamentals assertions while keeping the ambient
fixtures and installing spies at the SEC EDGAR and Financial Datasets
fallback seams. Both nodes remained RED because both captured fallbacks were
attempted; no unspied network or credential-dependent result occurred. After
switching only those nodes to the stored IBKR-snapshot seam:

- `fundamentals-green` returned `2 passed`;
- `after-fundamentals-focused` returned
  `122 passed / 1 failed / 123 collected`;
- the only remaining node was
  `tests/test_app_records_store.py::test_report_insert_query_roundtrip`; and
- that one-node stream matched the planned
  `71b6d959c36e1b7d8e9c92b4904a8e68c46c6c4d7992c0bdc939dc1b220798f0`.

The Task 4 reporters and transcripts are:

- `fundamentals-red.json` / `fundamentals-red.txt`:
  `f572890ec1c6752544e05b62cb3ff88cfa3a52a42d4922432ef694a494147a57` /
  `1e8af39255a81da5d86f0401ed8d6cacdaa976c9cf1af8fc843716c31e5998d6`
- `fundamentals-green.json` / `fundamentals-green.txt`:
  `778916978cba9424e9a20ad4440400e216bc7dc8aeb20689701f6767c868a54c` /
  `809ca8ed3c30a0e63ca40cbe4bd4837b9db5383053c61d46ac665d0580149049`
- `after-fundamentals-focused.json` /
  `after-fundamentals-focused.txt`:
  `76cf5255f104eff089d9983726a1a33cd9fe009d436e5497fc61904e42033fa6` /
  `6c306184f576defbbdeb35de7d333a4cac9f270b261de2c3503634363fae8e5a`

Fundamentals mutation:

```diff
-                "market_cap": 1_500_000_000_000.0,
+                "market_cap": None,
```

With that temporary snapshot change,
`TestAnalysisTools.test_get_fundamentals_analysis` turned RED on the exact
market-cap assertion after `provider_calls == []` had passed. The file was
restored to its exact pre-mutation SHA
`45b89dfce6e806c30cbb5a60b2a49460e1b749e46d16e49e5716987ee6fae717`,
then the owning node returned `1 passed`.

Task 5 reproduced the remaining app-record RED: the fixed
`2026-06-20T10:00:00` record fell outside the default moving 30-day window,
so `df.iloc[0]` raised `IndexError`. Only the existing query clock seam was
changed to `store.query_reports(today="2026-06-21")`; the inserted date,
window, and round-trip assertions remained unchanged.

- the owning node returned `1 passed`;
- `tests/test_app_records_store.py` returned `20 passed`;
- `final-focused` returned `123 passed / 0 failed / 123 collected`; and
- its empty non-passing stream matched
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The final focused reporter and transcript are:

- `/tmp/eir002-green-baseline/reports/final-focused.json`:
  `4a688ba8ec085e9cbf0abcee6c3331f7cffbbe639d0a06ec160d4255b0a6bcd8`
- `/tmp/eir002-green-baseline/reports/final-focused.txt`:
  `1ec4da0160810d2fcee78f31a1f23eba19f6b14af4011ce5d33cec53fa248343`

## 6. Protected Boundaries

- No file under `src/`, `data_sources/`, `apps/`, `config/`, `data/`, or
  `scripts/` changed.
- Task 1 changed only `tests/test_data_access.py`, deleting exactly the nine
  approved ambient-data test functions. No retained node was renamed or
  skipped.
- Task 2 changed only `tests/test_api.py`, `tests/test_tools.py`, and
  `tests/test_agents.py`, adding private hermetic helpers and switching the
  seven named news nodes. The module-wide ambient fixtures remain unchanged.
- Task 3 changed assertions and fixture parameters only for the eight named
  price/status nodes in those same test files. No helper data, module-wide
  ambient fixture, node identity, or product file changed.
- Task 4 changed assertions and fixture parameters only for the two named
  fundamentals nodes in `tests/test_api.py` and `tests/test_tools.py`.
  Provider fallbacks remain product-owned and unmodified.
- Task 5 changed one call in `tests/test_app_records_store.py` to use the
  product's existing explicit clock seam. No assertion or product code
  changed.
- Task 6 re-ran
  `git diff --quiet 20d4e7e2 -- src data_sources apps config scripts`
  successfully, and `git diff --name-only 20d4e7e2 -- data` was empty.
  Protected current-shape anchors remained `94 passed / 18 skipped`.
- No provider credential was supplied, and no production database, Gateway,
  scheduler, or product collection operation was used. Optional third-party
  client tests remained inside the blank-credential test environment and are
  not product-data admission evidence.
- The main worktree's untracked
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` was not copied, edited, staged,
  deleted, or cited. It remains untracked at SHA-256
  `79d4eac97d7692684d83f0a067f5987fe434bb76746b98af3e44f1c8ba4bf277`.

## 7. Native Final Admission

Task 6 reproduced both final collections:

- canonical:
  `4730 / c34de9a0fe53e400409d3ec26d75a8c907ee277b121279693ea9f69c8638aabb`;
- focused:
  `123 / 37386cd24fca323338fcb0fb2bbe5c42b7c471d2e9fde2e7c5f345b5ce631b8f`;
- final collection JSON:
  `ab6487af8fd731ae99f0c4dcec0147d9df0528487da24c5c74e5a7f5807822b9`
  canonical and
  `a3c7f6e7d6b4d139d9f068297610b82ea1a2ae8dc844b42c5ddc4cd498ecd5ae`
  focused.

Both node streams are byte-equal to the pre-edit Task 0 targets. The
base-to-final removed stream is byte-equal to `retired.nodes`, the added
stream is empty, all seventeen retained IDs and the app-record ID occur
exactly once, and all nine retired IDs occur zero times.

Immediately before full admission, the pinned 942-byte wakeup probe remained
SHA-256
`10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e`
and returned
`{"callback_fired": true, "ready_count": 0, "wake_bytes": 0}` in the native
execution boundary.

`/tmp/eir002-green-baseline/run_native.sh final-full` then terminated
naturally with:

```text
4658 passed, 72 skipped, 0 failed
4730 collected / 4730 seen
exitstatus 0
```

The reporter's collected and seen streams are byte-equal to each other and
to the final canonical collection. The non-passing stream is empty at
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The final reporter and transcript are:

- `/tmp/eir002-green-baseline/reports/final-full.json`:
  `9babd7b9f24dda99594436dfa98d10c2af86b1b59b3ea5f19313489f4b1c5b9e`
- `/tmp/eir002-green-baseline/reports/final-full.txt`:
  `4df8765c1a414ff03343d8605d3046877b5ac1cf8057780233555a6676d241ff`

Artifact restoration and quarantine then reproduced the exact pre-run status,
path inventory, and complete metadata/hash manifest as recorded in Section 4.
Task 6 admission is therefore complete and review-ready; no merge or EIR
closure has occurred.

## 8. Reviewed Merge And Closeout

Independent implementation review returned GREEN, and `master` fast-forwarded
linearly from `3092fb41` to exact reviewed tip `99bc071e`; the untracked
scripts-retirement decision remained untouched at SHA-256 `79d4eac9...`.
Merged collection reproduced `4730/c34de9a0...`, byte-equal to the target, and
`merged-focused` completed `123 collected / 123 seen / 0 non-passing`.

The first merged window, `2026-08-01T08:07:00+08:00` through
`08:09:39+08:00`, stopped before full admission because a real extension sync
appended non-test actions (`get_market_news_recent_ids`, `save_market_news`,
and `record_extension_job`) and changed `profile_state.db` and
`sa_capture.db`. No production byte was reverted. The 42,402,455-byte
post-focused log and its exact 1,505-byte append are preserved under
`/tmp/eir002-green-baseline/merged-quarantine/blocked-focused/`; their SHA-256
values are `a38d3e37...` and `85319396...`.

After the user paused extension sync, two native-host and three-file
stability samples matched. A fresh resume baseline then ran the unused
`merged-full` stage. It terminated naturally with all 4,730 nodes seen:

```text
4676 passed / 53 skipped / 1 failed
non-passing:
tests/test_db_backend.py::TestFundamentalsDB::test_fundamentals_via_dal
```

Reporter SHA-256 is `95aa3549...`; transcript SHA-256 is `240bcb66...`.
The failure reproduces in 0.45 seconds in the main worktree and skips in the
reviewed worktree. The main root's ignored `config/.env` directly enables the
PostgreSQL integration class; `FundamentalsResult` has never exposed the
asserted `.found` field. Its current typed absence value is
`data_source="none"`.

The failed full run appended only two test pings (486 bytes) to the production
native-host log. The full post-run file and append are archived at SHA-256
`d904740f...` and `7171820a...`; exact-prefix proof reproduced the pre-run
`209c0ed9...`, after which the log's inode, size, nanosecond mtime, and SHA were
restored exactly under the pre-authorized rule. Eighteen new paths were moved
by exact path to
`/tmp/eir002-green-baseline/merged-quarantine/resume-failed-full/new/`;
manifest SHA-256 is `6ba4278e...`. Pre/restored Git-status and path inventories
are byte-equal.

The production `profile_state.db` change was preserved. A read-only query
attributes its run-window update to real scheduler job `18436`,
`collect.ibkr_news`, which finished successfully at
`2026-08-01T00:18:38+00:00`. This independently proves that the production
main root cannot provide a frozen canonical data boundary.

EIR-002 remains `promoted`. No test or product edit is authorized until the
LD 9 / Task 7 bounded amendment receives independent review.
