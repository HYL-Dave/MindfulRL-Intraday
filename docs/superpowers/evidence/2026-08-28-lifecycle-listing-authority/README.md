# Listing Authority Offline Admission

This packet admits lifecycle listing-authority behavior using repository fixtures,
temporary SQLite databases, calibrated fail-closed observers, independent named
mutations, and a fixture-only browser matrix. `commands.txt` is the unattended
replay entry point. `verification-summary.json` is the bounded result index and
`SHA256SUMS` seals the exact packet set.

The strict shadow path is repository fixture bytes to the production parser/session,
listing evidence builder, real fact kernel in temporary SQLite, and production
policy. Frozen repository helpers provide SEC facts where no SEC provider payload
exists; this packet does not claim real SEC payload coverage.

## Boundaries

- No live SEC, Nasdaq, Massive, or IBKR call was authorized.
- No production database read, write, preflight, backup, migration, or restore was authorized.
- No real production A-to-B execution or production migration was performed.
- No app restart, merge, push, or other remote operation was authorized.
- Declared zero values are declarations, not measurements.
- The browser uses local mocked application data and an owned transient Vite PID only.

## Contents

- `offline-authority.json`: calibrated boundaries, nine-case strict shadow, and v2/v3 scratch migration/restore with old-code startup.
- `mutation-ledger.json`: M01-M20 owner kills and byte-identical restoration.
- `browser/matrix.json`: 24 locale/viewport/scenario entries and negative request/UI assertions.
- `backend-*.txt` and `frontend-*.txt`: the required repeated backend and complete frontend gates.
- `verification-summary.json`: counts, limitations, and cross-output assertions.
- `SHA256SUMS`: exact allowlisted packet manifest; the manifest itself is excluded from its entries.
