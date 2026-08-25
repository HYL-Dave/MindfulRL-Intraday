# Trusted Lifecycle Automation Live Migration Runbook

Status: `PREPARED_OFFLINE_NOT_AUTHORIZED`.

This runbook is an authorization template, not permission to inspect or modify a
live database. Its companion JSON intentionally leaves every live path, digest,
backup identity, restore result, and approval value as `UNAUTHORIZED` or
`NOT_RUN`.

## Exact Authority

- Product/test authority: `7cb479a8058793dc29cbb75bb4ab98b9d6a6f231`.
- Migration API: keyword-only `migrate_automation_profile_schema` in
  `src/security_lifecycle_automation_migration.py`, SHA-256
  `2172c28c4ce3d2fde45d9e2e984d4f39baf12915bfd7142f924761148b499d91`.
- Profile schema authority: `src/security_lifecycle_schema.py`, SHA-256
  `80b42fad0b56814ce64f2550edd3592c3ce47b43398a23d45c043a57c2b5d1b0`.
- Ticker schema authority: `src/ticker_identity_schema.py`, SHA-256
  `008f0f7d7043be3ed73f7c7a4ea60d93156e16e79f54a54e278da13211aed7b7`.
- Old-code rollback authority: `64af5092dd22523c672b8c42e3b84eaba04bec1f`.

Recompute every file hash and require exact equality immediately before any
future live action. A docs-only tip may differ from the product/test authority;
no product, migration, or schema byte may differ.

The repaired offline packet still uses synthetic SEC prose and IBKR contract
shapes. Fetching the four reviewed SEC documents and running one read-only IBKR
shape canary remain separately unauthorized. Neither validation authorizes a
production database operation, migration, merge, push, or cutover.

## Future Authorization Sequence

1. Stop the App and every scheduler writer. Record a positive quiescence
   witness. Merely observing an idle interval is insufficient.
2. Under separate read-only authorization, call
   `preflight_automation_migration(profile_path=<explicit live path>)` twice.
   Require byte-identical reports, V1 schema, integrity `ok`, zero foreign-key
   violations, zero unowned dependencies, zero Tavily runs, zero retired web
   evidence, and exactly four accepted legacy assessments.
3. Materialize the four assessment IDs and source-row digest in a proposed
   approval manifest. For every row, preserve all V1 fields, set
   `acceptance_authority=legacy_migration`, and set all automation provenance
   fields to `NULL`. Obtain explicit user approval of that complete mapping and
   the preflight `approval_sha256`.
4. Under separate backup authorization and while still quiesced, call
   `create_automation_profile_backup` with explicit source and destination
   paths. Record and approve its file SHA-256 and bound source approval digest.
5. Restore that exact backup to an absent scratch path. Verify the V1 profile
   and ticker schema, integrity, foreign keys, and all four assessment rows.
6. Clone the restored scratch state. Boot old product
   `64af5092dd22523c672b8c42e3b84eaba04bec1f` with explicit scratch profile and
   market paths, schedulers disabled, and provider/network access denied. The
   old lifecycle read surface must return all four accepted assessments, and
   instrumentation must prove no production path was opened.
7. Only after separate migration authorization, call
   `migrate_automation_profile_schema` with the explicit live path and approved
   digest. It must acquire `BEGIN IMMEDIATE`, revalidate the same digest under
   lock, preserve unowned state, verify exact V2 schemas, return integrity `ok`,
   and return zero foreign-key violations before commit.
8. Fast-forward only the independently reviewed product tree, run merged-tree
   gates, and start the App only under their separate authorizations.

Any count, schema, row, digest, sidecar, quiescence, restore, old-code boot, or
postflight mismatch is a hard stop. Do not edit an approval manifest in place to
make live data fit an expectation; produce a new preflight and obtain a new
approval.

## Rollback Consequence

The old and new code enforce different exact schema authorities. A code-only
rollback is unsupported. Rollback requires stopping the new App, restoring the
explicitly authorized pre-migration profile backup, then starting old product
`64af5092dd22523c672b8c42e3b84eaba04bec1f`.

Restoring that backup discards every profile write made after cutover. There is
no implicit forward/down-conversion migration. This data-loss consequence must
be presented to and explicitly accepted by the user before live migration.
