# Trusted Lifecycle Automation Real-Source Canary

Status: `CONSUMED_REPAIRED_OFFLINE_ADMISSION_COMPLETE`.

The user authorized one bounded acquisition of four exact SEC primary
documents and one read-only IBKR contract-shape session. The executable
authority is `run_canary.py` plus
`docs/superpowers/plans/2026-08-25-trusted-lifecycle-automation-real-source-canary.md`.

The single authorized run completed at `2026-08-25T10:27:43.271225Z` from
clean commit `f6a7e12987f3614d1be1fb17afd8ecc0f1c9f96f`:

- SEC: four exact documents, four attempts, four HTTP 200 responses, 165,873
  response bytes, zero rate-limit retries, and zero live submissions calls.
- IBKR: one read-only session and two contract-detail requests (`LC`, `HAPN`).
  `LC` had no contract definition; `HAPN` returned one canonical common-stock
  snapshot on Nasdaq.
- The live shapes exposed deterministic SEC extraction gaps. They were
  reproduced RED-first and repaired at product/test authority `1ec76167` by
  advancing the SEC extractor rules from version `1` to `2`.

The repaired extractor was then replayed twice against the exact saved source
bytes and the secret-safe IBKR receipt from the single consumed canary. Both
runs were byte-identical (`8094aeeb...e88dd`):

- HAPN resolves `LC -> HAPN`, `NYSE -> NASDAQ`, effective `2026-06-22`, and
  reaches exactly one eligible transition preview when combined with the
  captured HAPN IBKR contract snapshot.
- QBTS resolves a same-symbol `NYSE -> NASDAQ` transfer effective
  `2026-07-27`, but remains `review_suggested` with the typed blocker
  `market_corroboration_missing`; no symbol transition is requested.
- CCL resolves a completed corporate unification with no tracked-security
  identity change.
- BLBD resolves an asset purchase involving Detroit Chassis LLC with no
  registrant identity change and no invented cash, ratio, or identity date.

Admission is GREEN: four real-source additions, seven focused backend files
(`70P`), collection `4420` twice with byte-identical node streams, full backend
`4408P/12S/0F`, frontend `105 files/1229P`, typecheck, visible-literal scanner,
production build, runtime routes `187/17`, and tools `50/51/51`. The existing
synthetic grounded-shadow replay also remains byte-identical across two runs.

`canary-report.json` remains the immutable secret-safe live receipt.
`post-repair-replay.json` is the offline repaired result; it does not overwrite
or reinterpret the original live output. `sec-source-bytes/` contains the exact
public response bodies used for RED/GREEN work. `network-summary.json` is
derived from the temporary syscall trace; the raw trace was not committed
because it contains the configured IBKR endpoint. Provider failure did not
permit an automatic retry, and no second provider call occurred.

Formal production migration, merge, push, and app cutover remain unauthorized.
The packet proves parser behavior and one real A-to-B decision preview; it does
not claim a production transition apply or broad A-to-B precision.
