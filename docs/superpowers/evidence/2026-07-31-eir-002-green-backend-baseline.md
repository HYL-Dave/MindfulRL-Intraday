# EIR-002 Green Backend Baseline Evidence

> **Status:** TASK 0 COMPLETE - INDEPENDENT GROUNDING REVIEW NEXT
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

The pre-run Git status was empty, `data/` contained no files, and `src/data`
did not exist. Each canonical full run created only
`src/data/cache/risk_free_rate.json`; each exact path was recorded and moved
to a distinct quarantine before any later admission run:

- Rejected run: inode `90595257`, 73 bytes, SHA-256
  `3a1bee06ad385aee524b17951a0878fc243fb49828020c37edb7c1018498edb9`,
  quarantined under
  `/tmp/eir002-green-baseline/task0-invalid-quarantine/src/data/cache/`.
- Accepted `base-full-v2`: inode `90595258`, 73 bytes, SHA-256
  `63914ef1c90ebe829bd1e6dfe71752effe2949b33fb88e3d5242a566f3c42378`,
  quarantined under
  `/tmp/eir002-green-baseline/task0-base-full-v2-quarantine/src/data/cache/`.

The now-empty generated directories were removed with exact `rmdir`
operations. The restored status stream was byte-equal to the pre-run stream
before `base-focused`; focused execution created no repo-relative artifact.

## 5. RED And Mutation Evidence

Task 0 records the pre-existing RED set only. No assertion, fixture, product
file, test node, skip marker, or mutation was changed or executed. RED-first
family work remains blocked until independent review clears this packet.

## 6. Protected Boundaries

- No file under `src/`, `data_sources/`, `apps/`, `config/`, `data/`, or
  `scripts/` changed.
- No test file changed.
- No provider credential was supplied, and no production database, Gateway,
  scheduler, or product collection operation was used. Optional third-party
  client tests remained inside the blank-credential test environment and are
  not product-data admission evidence.
- The main worktree's untracked
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` was not copied, edited, staged,
  deleted, or cited.

## 7. Native Final Admission

Not started. Task 0 establishes only the reviewed base and target identities.
Final zero-failure admission belongs to later tasks and must use the same
Node-pinned native wrapper and exact target collection.

## 8. Reviewed Merge And Closeout

Not started. Independent Task 0 review is the next gate.
