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

## Task 3 - Durable Telemetry

- Task 3 product tip:
  `df49d5b5369193c4bd94ef660cd90df0892d956e`.
- The outbox/native/store/protocol suites are `8/8`, `14/14`, `63/63`, and
  `12/12`; the Task 3 core is `105/105` and the canonical focused set is
  `284/284`.
- Backend checkpoint collection is `4667`; full/focused sorted node-list
  SHA-256 values are
  `51548195153e1f9e12a24fa5475d9de33b4afba9e813ae3a5ce5abd1a66ef085`
  and
  `2c3b4e302fff4c190ebbc3bb5a05db2b4ece2186a5c2d0430b41bab25217a717`.
- Two independent Firefox builds contain the same `14` runtime files;
  normalized artifact hash-list SHA-256 is
  `951dc32c892a1d35f64fcb0b0d49eb31752536420f913aa8522a4850444dc7b5`.

## Task 4 - Structured Durable Health

- Task 4 product tip:
  `1e52ed4e391d42ff9421b42308ba14a6fad15ed6`.
- Backend collection is `4677`; the 12-file focused set is `294/294`.
  Full/focused sorted node-list SHA-256 values are
  `ec699e992433f5c2cabe612e5f609fad1ca64ae88915034bb8e56aa2bfcd7de9`
  and
  `d35c5155ed480f3495567ce172bc00b391b6e69d4768526d288d2b738a679a47`.
- Frontend is `95 files / 1063 passed`; the four-file focused set is `69/69`.
  Full/focused sorted relative node-list SHA-256 values are
  `a93c02bc28d1924f23f7895338d723e968dcb389a494ff0e0f993e4c092019d4`
  and
  `5c3859c3c7db7fe90c13f3c46d49610eae64986ef0b89e558246ee0cf13c6cdf`.
- Resources are `694/1794` per locale; scanner remains `36/20/0/20`.

## Task 5 - Durable Market News Repair

- Task 5 product tip:
  `8b20e608765d2ef134a6249a2ebf0bccd400361f`.
- Sixteen domain nodes cover canonical manifests, exact path admission,
  recorded/incident previews, atomic start, resumable progress, reconciliation,
  cancellation, and terminal hash idempotence. Five DAL nodes cover exact-ID
  no-age reads, body readback, inclusive intervals, privacy projection, and
  unavailable-DB fail-closed behavior.
- Recovery is `16/16`; `test_sa_tools.py` is `102/102`; the four-file Task 5
  execution set is `195/195`.
- Backend collection is `4698`; the 13-file checkpoint collection is `315`.
  Full/focused sorted node-list SHA-256 values are
  `d7c35ad0fae96f6f0e4fd0211fdc9a1bdd8e51eeb63d616587f435dabff2f284`
  and
  `3f1b0a608ab2037c403e57eddb53c31dca47e10ef47977d2c10b6e1594375b48`.
- Generic jobs history/status omit frozen target descriptors and expose only
  kind, counts, lifecycle state, and bounded manifest-hash prefix. Fixed
  recovery routes retain the full machine contract.
