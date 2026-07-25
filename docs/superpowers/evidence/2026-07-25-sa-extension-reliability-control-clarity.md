# SA Extension Reliability and Control-Clarity Evidence

> **Status: IMPLEMENTATION IN PROGRESS - NOT REVIEW READY**
>
> This ledger contains redacted, reproducible engineering evidence only. It
> must not contain licensed article content, raw repair target IDs, URLs,
> credentials, browser-profile data, or production database copies.

## Task 0 - Clearance and Baseline

- Plan-review clearance:
  `7d4f11164861aa0a50cc7771fef7388577f4da0b`.
- Isolated branch/worktree: `codex/sa-extension-reliability` at
  `/mnt/md0/PycharmProjects/ArkScope-sa-extension-reliability`.
- Linked-worktree materialization: known `git-crypt` smudge boundary handled
  with `--no-checkout`, the existing 148-byte key copied to linked Git metadata
  with mode `0600`, and `git read-tree -mu HEAD`. Protected document size:
  `47,608` bytes. Existing root `node_modules` linked without install.
- Backend collection: `4621`; sorted raw pytest node-ID SHA-256:
  `488eeaab65ffad32bd098dbc4b1df0eb3ed3b62feabfe3a62b1a76324d960a17`.
- Backend focused collection/run: `9 files / 238`; `238 passed`; SHA-256:
  `ca36c2cc8616982fa8dd2c2f386743751691de6bd4f9bf52134229d830740de8`.
- Frontend collection: `95 files / 1056`; sorted relative node-list SHA-256:
  `5f9a1624b31a47dc9b786f57fa5de77eca86dde269c68ada3787d7210b05fd13`.
- Frontend focused collection/run: `4 files / 62`; `62 passed`; SHA-256:
  `025e871755c356f0be89089e92d0241d06b335af52ae8a2ca0f66e06b187f643`.
- All fixed-base protected-boundary gates against `c49a2417` pass before
  product edits. `git diff --check` passes and the worktree was clean.

## Pending Evidence

- `PACKAGING_GATE_TIP` and deterministic package artifacts.
- Structured run-protocol fixture parity and telemetry outbox bounds.
- Derived-status consumer audit and native-host privacy gates.
- Repair manifest/state-machine copied-DB proof and separately approved live
  repair, if later authorized.
- Popup accessibility, control disclosure, and browser runtime evidence.
- Final node ledger, normalized hashes, protected boundaries, and canonical
  A/B verification.
