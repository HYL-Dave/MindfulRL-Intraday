# Settings Schedule Surface Ownership Evidence

> **Status:** TASK 0 COMPLETE; TASKS 1-3 BATCH EXECUTION ACTIVE
>
> **Reviewed plan tip:** `590ce355ae3f296e666a5dbcabd5b3694d7633a1`
>
> **Product grounding base:** `e2ead4378189d277a40a61f7130730513980794e`

## Task 0 - Re-grounding

The reviewed plan tip has exact merge-base `e2ead437` and differs from that
base only in the design, plan, and priority-map documents. Product and test
bytes are unchanged.

Pre-grounding authority identities:

```text
f87e4b479019857e7d4f8c87a1fdf160916396dfcbd46cb404e249000543ef3b  docs/superpowers/specs/2026-08-14-settings-schedule-surface-ownership-design.md
68616f99c39f033293b800297a2711d38de14653d683f335eec58740d6871a34  docs/superpowers/plans/2026-08-14-settings-schedule-surface-ownership.md
d99bf95f9362a93076d56892fe4fda775999bad2d85814878e6d81812ee9ead9  docs/design/PROJECT_PRIORITY_MAP.md
955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac  pinned Vitest list normalizer
```

Fresh collection and runtime results:

| Stream | Count | SHA-256 / result |
|---|---:|---|
| frontend full | 1,177 | `9530dcd91d8a7d684faa5e56f2986fbaeaa22e1d89f67818a12ed5d8ca77d1b1` |
| six-file focused | 127 | `83b8a5e4941aea52b5818d9fe9073407e69d559338c579d3f25fc930b514a735`; `127 passed` |
| Settings 15-file projection | 249 | `a3a5e481cace86991db6d8ec5da56c2d973d224e1cb1de57f631c210a646a16e` |
| backend collect-only | 4,394 | `b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb` |

The exact two-row removal and addition streams are `9c8ca36c...` and
`d35fbff9...`. Both removals occur once in the base, both additions are absent,
and set subtraction/addition independently produces frontend
`1177/90f56093...`, focused `127/0d6eb258...`, and Settings
`249/ee6618df...`. All eight retained body-evolution IDs exist; their literal
stream is `21264689...`.

All 11 owned file identities and line counts match Section 0.4. All 13
protected files match, and the independently replayed protected aggregate is
`b3b3ac3f2151d34bf3ce0e15a772ebb0a655e46fe63a43eccae142d4208602a1`.

The packet records three rejected operator attempts: a Settings projection
that included an out-of-set i18n test, a Markdown-section extraction that did
not enter the literal fence, and a backend command that referred to a missing
worktree `.venv`. None contributed to admission. The explicit admitted backend
replay, `PYTHONPATH=. pytest --collect-only -q` followed by C-locale sorting,
is byte-identical to the reviewed backend stream.

Packet: `/tmp/schedule-surface-ownership-task0-590ce355`, 53 payloads,
`SHA256SUMS` SHA-256
`0fd92def7afc208e085cb2b58d5e3031c6c32dac477f0483be02de85ec70298c`.

## Process Ruling

After Fable's independent plan review returned GREEN, the user authorized
Tasks 0-3 as one continuous execution batch. Per-task commits, packets,
RED-first evidence, exact identities, and all stop conditions remain binding.
Task 3 is the combined implementation-review gate. Merge, push, live traffic,
provider calls, and production mutation are not authorized.

## Task 1 - Partition and Latest-Job Facts

Product/tests commit: `7de617dd`.

The exact replacement collection is `101 files / 1177 /
90f56093290c70a27369296ec8d8c7de99d084a091134994ae6451bc8e45743b`.
Focused is `127 / 0d6eb25864d2c0a7a5ee76a9d05dacbc45e232106e62461a043b3d7196aac644`,
and the Settings projection is `249 /
ee6618dfc755f5ead79f6012967d3dda2c7a33f37ef87f62c5f98deff0f6c5cf`.
The node delta is exactly the reviewed `-2/+2` ledger.

RED was `2 failed / 125 passed`. The Provider integration exposed ten rows
under Data Sources instead of five, and the focused controller test exposed
that the old table ignored the required scope. No existing-module import,
fixture, timeout, or network failure contributed to RED.

GREEN proves:

- exact `write_target === "macro_calendar.db"` classification;
- disjoint and complete Macro/non-Macro partitions in insertion order;
- future Macro placement and unknown non-Macro fail-visible placement;
- one real shared controller and enabled controls in both partitions;
- Data Sources and Macro each render five current rows in both locales;
- a successful existing provider-health read replaces an older cached Macro
  job timestamp, while a rejected read preserves the admitted timestamp; and
- the disabled-provider neutral contract remains intact.

Final runtime is `127/127`; typecheck exits zero. Structural census records one
mounted `DataScheduleControlsProvider`, one schedule timer, one focus listener,
unchanged `getSchedule` and `getProvidersHealth` caller sets, zero table-level
`jobs` prop, and zero product occurrences of `sourceIds`,
`DataScheduleLocalError`, or `MACRO_SCHEDULE_SOURCE_IDS`. Protected paths remain
`13/13` at aggregate `b3b3ac3f...`.

Packet: `/tmp/schedule-surface-ownership-task1-bf6b74a0`, 40 payloads,
`SHA256SUMS` SHA-256
`966e33572f9dd6dba40caf740d6bd542bd9f092965cb94994253b4ec9cc48e6b`.
