# Trusted Lifecycle Automation Real-Source Canary

Status: `CONSUMED_OFFLINE_REPAIR_REQUIRED`.

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
- The live shapes exposed deterministic SEC extraction gaps. No formal
  migration or production automation run is permitted until the offline repair
  and admission replay complete.

`canary-report.json` contains the secret-safe receipt. `sec-source-bytes/`
contains the exact public response bodies used for offline RED/GREEN work.
`network-summary.json` is derived from the temporary syscall trace; the raw
trace was not committed because it contains the configured IBKR endpoint.
Provider failure does not permit an automatic retry. Production migration,
merge, push, and app cutover remain unauthorized.
