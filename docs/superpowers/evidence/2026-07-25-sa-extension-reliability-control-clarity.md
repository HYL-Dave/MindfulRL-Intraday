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

- Telemetry outbox bounds and atomic sidecar persistence.
- Derived-status consumer audit and native-host privacy gates.
- Repair manifest/state-machine copied-DB proof and separately approved live
  repair, if later authorized.
- Popup accessibility, control disclosure, and browser runtime evidence.
- Final node ledger, normalized hashes, protected boundaries, and canonical
  A/B verification.

## Task 1 - Atomic Firefox Packaging

- RED proof: `11 failed / 3 passed`. All ten new packaging nodes and the
  renamed installer owner failed before the builder/delegation existed; the
  three host-path parameter cases remained green.
- GREEN proof: `14 passed` in the packaging/install-path suite.
- `PACKAGING_GATE_TIP`:
  `a2869f2ce7a2c2e603657593ef9534a438cd02a6`.
- Backend checkpoint collection: `4631`; sorted raw node-list SHA-256:
  `6a005572814a1c539e96b521b0a9cdb2984d47e8ec625f0dc6a696d6da3a635b`.
- Focused checkpoint collection: `248`; sorted raw node-list SHA-256:
  `46fcc1766e2e8301e75b7ce5cdb7089d5d98e81955a916f1f84c0e9c72f2a300`.
- A fresh real Firefox build contains exactly `12` files, including
  `article_identity.js`. Two separate output directories produced identical
  file lists and per-file hashes. Normalized artifact hash-list SHA-256:
  `089a379302cdd6cd7ada109675caae295b38450d88720d7010bab797cd5c8866`.
- Adversarial fixture coverage rejects missing dependencies, dynamic
  `importScripts`, variable/concatenated/computed/spread `executeScript`
  dependencies, remote/traversing/query references, and final-swap failure.
  Failure preserves the previous known-good directory; successful replacement
  drops stale files.

## Task 2 - Structured Run Protocol

- RED proof: the combined protocol/background suite was exactly
  `15 failed / 5 passed`. The twelve pure protocol nodes and three real
  background-adapter nodes failed for absent modules/functions; the five
  existing Alpha Picks nodes remained green.
- Shared fixture: `15` protocol cases plus opaque background-adapter cases.
  The incident-shape case contains exactly `18` retryable details. Python and
  classic-script JS return byte-equivalent JSON values or the same stable
  validation code for every case.
- GREEN proof: protocol `12/12`, Alpha `8/8`, packaging `10/10`. An in-place
  hardening case subsequently exposed unknown legacy Market status as false
  complete (`1 failed`); it now fails closed to `protocol_invalid` with no
  node-accounting change.
- Task 2 product tip:
  `46bce5886b7d466bcc0d0cd3f21d522a4ca41619`.
- Backend checkpoint collection: `4646`; sorted raw node-list SHA-256:
  `8d0a838196e2cd5552844963ce270985bcb169c3ad096df375d48ead7a78c8f1`.
- Focused checkpoint collection/run: `263`; `263 passed`; sorted raw
  node-list SHA-256:
  `00b1d9ed399b898f1821dac7f1039cc6b41359ae578864acb94bbb726f651366`.
- Canonical results derive database status and healthy-anchor eligibility from
  closed phases/items. Raw legacy prose is not admitted to canonical items;
  unknown browser failures become `unknown_failure`; only explicit
  `404`/`410`/removed evidence can be source-unavailable.
- Firefox and Chrome both acquire `extension_run_protocol.js` through the
  dependency graph. Two fresh Firefox builds were byte-identical and each
  contained exactly `13` files. Normalized artifact hash-list SHA-256:
  `f34e12aa153b054735ca133fcb918d682e4d52b9cd6a5481d39aecd5cd719522`.
