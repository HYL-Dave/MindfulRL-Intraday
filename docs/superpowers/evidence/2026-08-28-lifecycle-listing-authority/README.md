# Listing Authority Offline Admission

This packet admits lifecycle listing-authority behavior using repository fixtures,
temporary SQLite databases, calibrated fail-closed observers, independent named
mutations, and a fixture-only browser matrix. `commands.txt` is the unattended
replay entry point. `verification-summary.json` is the bounded result index and
`SHA256SUMS` seals the exact packet set. `repository-binding.json` separately
binds the replay to its clean Git head/tree and the exact imported browser
runner blobs; the SHA manifest does not claim to seal files outside this packet.
A controller-development checkpoint without a current `SHA256SUMS` is not an
admitted packet; only the subsequent complete replay may recreate the seal.

The strict shadow path is repository fixture bytes through a fake production
transport and the real `ListingAuthoritySession`, then the real fact kernel in
temporary SQLite and production policy. Every case binds its exact `shadow-*`
Nasdaq/Massive filename and SHA-256 in `shadow-cases.json`; those bytes enter the
session without ticker substitution, field mutation, or JSON reserialization.
Frozen repository helpers provide SEC facts where no SEC provider payload exists;
this packet does not claim real SEC payload coverage.

## Replay Boundaries

- During this packet replay, no live SEC, Nasdaq, Massive, or IBKR call was
  authorized or executed.
- During this packet replay, no production database read, write, preflight,
  backup, migration, or restore was authorized or executed.
- No real production A-to-B execution, App restart, merge, push, or other remote
  operation was authorized or executed.
- Provider, production-database, restart, merge, and push zeros are declared
  unexecuted operations. Browser external-request and write zeros are separately
  measured by the local browser harness.
- The browser uses local mocked application data and an owned transient Vite PID only.
- Browser TERM and OTC History states are synthetic post-apply UI projections. They
  contain product-shaped applied transition, acknowledgement, and both reversal
  witnesses, but they were not produced by the nine-case shadow execution.
- Browser conflict Attention is an active Nasdaq/active Massive projection carrying
  the SEC and listing CIK shapes even though the compact UI does not render CIK.
- Listing evidence is structured status data rather than source prose, so its
  translation-control count is intentionally zero. Regulator source-text evidence
  retains the translation control and original-language source.

The separately authorized live Massive canary is not part of this offline packet.
It is retained only as an unsealed operator observation: four bounded calls using
the credential configured at that time observed AAPL as active and LC as inactive.
No raw provider bytes or reproducible canary artifact were retained, so this packet
does not use that observation as sealed admission evidence.

## Admission Rules

- Each M01-M51 entry runs its unchanged declared pytest command before mutation,
  admits an exact collected/executed node set with zero failures, then requires
  exit 1, the exact expected failures, and mutation-specific failure signatures.
- Mutation anomaly counts cover only unexpected failures inside each declared
  mutation command. They do not claim observation of tests outside that scope.
- The archived old-code child installs and calibrates its own SQLite guard. URI
  paths are unquoted and symlinks resolved before containment; one outside open is
  rejected before access, one inside calibration opens, and the restored DB opens
  read-only under the same guard before integrity and foreign keys are accepted.
- Focused and full backend gates each seal two exact, byte-identical collected-node
  manifests. Passing/skipped counts must agree with the full node identity.
- Browser geometry is scoped to lifecycle activity bands and drawers. Executed
  positive calibrations prove that a same-surface covered control produces an
  overlap failure and vertically clipped text produces a clipping failure; a drawer
  intentionally covering background controls remains excluded from that surface.
- `normalize_packet_logs.py` replaces checkout and Python-environment absolute paths
  and token-shaped fixture parameters with typed placeholders, then reduces terminal
  blank lines to one newline while preserving node order, semantic output, and
  counts. Raw A/B node manifests are compared before this normalization.
  `log-normalization.json` records the step.

## Contents

- `offline-authority.json`: calibrated boundaries, nine-case strict shadow, and v2/v3 scratch migration/restore with old-code startup.
- `mutation-ledger.json`: M01-M51 baseline/mutant commands, exact nodes, signatures,
  scoped failure outcomes, and byte-identical restoration.
- `browser/matrix.json`: 24 locale/viewport/scenario entries, typed forbidden-operation
  declarations, projection provenance, transition/acknowledgement witnesses, expanded
  regulator/listing evidence, and measured translation-control assertions.
- `backend-*.txt` and `frontend-*.txt`: the required repeated backend and complete frontend gates.
- `verification-summary.json`: counts, limitations, and cross-output assertions.
- `repository-binding.json`: clean tested Git head/tree, replay-scope result, and
  exact Git blob/SHA-256 bindings for the three imported browser runners.
- `secret-scan.json`: final packet scan for live environment values, token shapes,
  and serialized environment mappings; matched secret values are never emitted.
- `SHA256SUMS`: exact allowlisted packet manifest; the manifest itself is excluded from its entries.
