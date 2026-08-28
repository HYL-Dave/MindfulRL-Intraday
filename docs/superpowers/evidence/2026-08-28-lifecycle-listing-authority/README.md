# Listing Authority Offline Admission

This packet admits lifecycle listing-authority behavior using repository fixtures,
temporary SQLite databases, calibrated fail-closed observers, independent named
mutations, and a fixture-only browser matrix. `commands.txt` is the unattended
replay entry point. `verification-summary.json` is the bounded result index and
`SHA256SUMS` seals the exact packet set.

The strict shadow path is repository fixture bytes to the production parser/session,
listing evidence builder, real fact kernel in temporary SQLite, and production
policy. Every case binds its exact `shadow-*` Nasdaq/Massive filename and SHA-256
in `shadow-cases.json`; those bytes enter the parser without ticker substitution,
field mutation, or JSON reserialization. Frozen repository helpers provide SEC
facts where no SEC provider payload exists; this packet does not claim real SEC
payload coverage.

## Boundaries

- No live SEC, Nasdaq, Massive, or IBKR call was authorized.
- No production database read, write, preflight, backup, migration, or restore was authorized.
- No real production A-to-B execution or production migration was performed.
- No app restart, merge, push, or other remote operation was authorized.
- Declared zero values are declarations, not measurements.
- The browser uses local mocked application data and an owned transient Vite PID only.
- Browser TERM and OTC History states are synthetic post-apply UI projections. They
  contain product-shaped applied transition, activity, and reversal witnesses, but
  they were not produced by the nine-case shadow execution.
- Browser conflict Attention is an active Nasdaq/active Massive projection carrying
  the SEC and listing CIK shapes even though the compact UI does not render CIK.

## Admission Rules

- Each M01-M20 entry runs its unchanged declared pytest command before mutation,
  admits an exact collected/executed node set with zero failures, then requires
  exit 1, the exact expected failures, and mutation-specific failure signatures.
- Mutation anomaly counts cover only unexpected failures inside each declared
  mutation command. They do not claim observation of tests outside that scope.
- The archived old-code child installs and calibrates its own SQLite guard. URI
  paths are unquoted and symlinks resolved before containment; one outside open is
  rejected before access, one inside calibration opens, and the restored DB opens
  read-only under the same guard before integrity and foreign keys are accepted.
- Focused gates seal identical node manifests. Full gates compare collection counts
  only, per the approved contract; this packet intentionally has no full-suite node
  manifests.
- `normalize_packet_logs.py` replaces checkout and Python-environment absolute paths
  with typed placeholders and reduces terminal blank lines to one newline while
  preserving semantic output and counts. `log-normalization.json` records the step.

## Contents

- `offline-authority.json`: calibrated boundaries, nine-case strict shadow, and v2/v3 scratch migration/restore with old-code startup.
- `mutation-ledger.json`: M01-M20 baseline/mutant commands, exact nodes, signatures,
  scoped failure outcomes, and byte-identical restoration.
- `browser/matrix.json`: 24 locale/viewport/scenario entries, typed forbidden-operation
  declarations, projection provenance, transition witnesses, and negative assertions.
- `backend-*.txt` and `frontend-*.txt`: the required repeated backend and complete frontend gates.
- `verification-summary.json`: counts, limitations, and cross-output assertions.
- `SHA256SUMS`: exact allowlisted packet manifest; the manifest itself is excluded from its entries.
