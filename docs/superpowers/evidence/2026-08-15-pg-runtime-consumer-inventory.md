# PostgreSQL Runtime Consumer Inventory Evidence

> **Status:** TASK 0 REVIEW GREEN; BATCH A AUTHORIZED; TASK 1 STOPPED FOR
> LOOPBACK-PROBE HARNESS AMENDMENT REVIEW
>
> **Reviewed amendment tip:**
> `da98626d295fe97bddb0a7a3bf478317d18e2f3f`
>
> **Product/design grounding base:**
> `729d8514ac912b447f1892aefd3e897ea8a843b6`

## Task 0 - Exact Docs-Only Baseline

The implementation worktree remained on the reviewed amendment tip. Its merge
base is the product/design grounding commit, and the range changes exactly the
plan, this evidence file, and the priority map. The main tree remained clean at
`a1413ac4244056e819c43ce5bce90969e58c460b`. The design SHA-256 is
`e5218b58472891891acdc56fa054b07a30cc98905d71941890ad15a438bf3935`;
the reviewed amended plan SHA-256 is
`8f1ce8cfac8bf96401b1b0a7d5f1c8881bf53f167e32839c1af667d532328d47`.
No tracked product or test byte changed.

### Rejected setup and operator attempts

The original direct worktree checkout failed at the expected `git-crypt`
smudge boundary. The admitted worktree uses worktree-local no-op filters and
retains ciphertext. The original frontend attempt is also rejected: it linked
the app-local Vite cache instead of the hoisted root toolchain, and `npx`
downloaded unpinned Vitest `4.1.10`. Sanitized npm-log projection recorded 148
HTTP fetch rows. Raw npm logs were hashed and deleted because they contained
machine paths and registry URLs. Focused review approved amendment `da98626d`,
which requires the root toolchain link, exact local Vitest `4.1.8`, and no
package-manager fallback.

The clean rerun had two bounded operator-command rejections. The first summary
queried aggregate reporter fields that do not exist; the admitted summary uses
the actual `seen_node_ids` and `nonpassing_node_ids` arrays. The first package
witness passed inline requirements comments to `packaging.Requirement`; the
corrected witness strips only the space-hash comment suffix before parsing.
Both failures occurred after their preceding collection had completed and
matched its pinned identity. Neither changed a target, product byte, test body,
or runtime boundary. Their bounded explanations are in the final packet.

### Isolated toolchain and runtime

Before every Python or Node witness, the controlling shell unset
`DATABASE_URL` and redirected home, five SQLite paths, token storage, and locks
under the packet runtime root. The worktree `data/` and scratch directories are
empty real directories. The sole main-tree link is root `node_modules`; the
app-local `node_modules`, main `data/`, and private `config/.env` are not linked.
The direct local binary reported `vitest/4.1.8 linux-x64 node-v22.14.0` before
collection. Tool versions were Python `3.10.12`, pytest `8.4.1`, Node
`22.14.0`, and Vitest `4.1.8`. No `npx`, `npm exec`, install, or download
fallback ran during the admitted rerun.

### Exact base identities

The backend command was collect-only with the pinned EIR-002 reporter. It
produced 4,394 globally sorted unique IDs at
`b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb`.
The reporter records 4,394 collected, zero seen, zero nonpassing, and exit 0.

The frontend command used the explicit root binary and equals-sign JSON output
argument. The pinned normalizer produced 1,177 globally sorted unique IDs in
101 files at
`90f56093290c70a27369296ec8d8c7de99d084a091134994ae6451bc8e45743b`.
Both tracked trees remained clean after collection.

### Sanitized environment witnesses

Installed metadata reproduced the plan-author witness exactly:

```json
{"import_providers":{"psycopg":[],"psycopg2":["psycopg2-binary"]},"installed":{"news-please":"1.6.15","psycopg2-binary":"2.9.10"},"observed_reverse_requirements":{"psycopg2-binary":["news-please"]},"repository_direct_requirements":["psycopg[binary]>=3.1"]}
```

The reverse requirement admits only absent markers or markers true under the
default environment with `extra=""`; the witness does not claim installation
history. Private `config/.env` was inspected only for key-name presence and
file metadata. The admitted output is
`{"database_url_key_present":true,"mode":"600","size":6463,"tracked":false}`;
no value, hash, excerpt, or copy entered an artifact.

`git crypt status -e` returned exactly the three reviewed encrypted paths. The
grounding-base and unlocked-main blob IDs are equal for each path:

```text
data_sources/DATA_SOURCES_EVALUATION.md       d07ce126266e3df2f03fed97a94364d6679e8a31
data_sources/IBKR_INVESTOR_DATA_VALUE.md      a1bc22836444f3582b41b74ca7ea3934a8fdd441
data_sources/PAID_SUBSCRIPTION_EVALUATION.md  1622c6010cb9b7323507ca5d4617d594e3a9c0b7
```

Only paths, blob IDs, and equality are retained; no plaintext or ciphertext
bytes were copied into the packet.

### Admission boundary

No runtime test, provider request, remote database connection, production
SQLite open, FastAPI lifespan, scheduler, archive restore, secret value,
encrypted plaintext read into evidence, or product write ran. Packet
`/tmp/pg-runtime-inventory-task0-729d8514` contains 31 checksummed payloads;
`SHA256SUMS` SHA-256 is
`1b53c95c1124877598f3272ae2606a5e9a9de7b26634b928b81217e5ede6df4e`.

The commit carrying this evidence is the immutable `CANDIDATE_SOURCE_TIP` for
Task 1; its resolved commit ID is reported at the review handoff. At that
commit, Task 1 remained unauthorized pending independent Task 0 review; the
subsequent GREEN and batch ruling are recorded below.

## Task 1 - Partial Candidate Extraction and Harness Stop

Independent Task 0 review returned GREEN and byte-compared both base streams
to the reviewer's independently collected copies. The user authorized Batch A
(Tasks 1-2, then classification review) and Batch B (Tasks 3-4, then combined
inventory review), without relaxing per-task commits/packets or any hard stop.

Task 1 scanners ran only against detached exact candidate source
`4c6b8d44ce2e768e95b822b11f618cc40f4bb9f0`. Raw candidate collection and
cross-checking had reached the test-outcome gate; no canonical tracked
`candidates.jsonl` or classification file was created. Two temporary-tool
errors were rejected before admission: zero-node command scripts were first
mistaken for pytest runtime files, and a pytest hook used a non-spec argument
name. The corrected projection preserves those command paths as raw CLI/text
candidates while selecting 64 backend runtime files solely from the 1,553
actual base-stream node IDs.

The complete backend candidate run produced `1,525 passed / 28 skipped`, but
is rejected because its guard recorded one socket attempt. Exact-node replay
proved the sole caller was
`tests/test_data_scheduler.py::test_pg_reachable_probe_is_bounded`, whose
existing contract deliberately probes local `127.0.0.1:9`. The guard raised
before the operating-system call and recorded zero actual network traffic.
The reviewed plan nevertheless had no named exception, so Task 1 stopped
before frontend runtime, candidate union, commit, or Task 2.

The bounded amendment admits only a pre-OS guard interception for that exact
node, operation, loopback target, port, and timeout, recorded as
`socket_guarded_loopback_refusal`. Every other socket guard hit remains a hard
stop. Resume must rerun both the exact node and the entire backend candidate
runtime; the rejected result is not reusable. No product, test, dependency,
configuration, secret, provider, database, lifespan, scheduler, or archive
action occurred.

Partial packet `/tmp/pg-runtime-inventory-task1-4c6b8d44` has 50 checksummed
payloads and manifest SHA-256
`f54d8ef505b7cd19a58886e0a16e75c277770c0f7172686cae2cafa3e38c39d1`.
Its leak audit has no home path, credential-bearing PostgreSQL URI, email, or
non-digest long-hex finding; the one JWT-shaped string is the frozen public
redaction-test node ID already present in the canonical backend base stream.
The implementation and candidate-source trees remain clean.
