# PostgreSQL Runtime Consumer Inventory Evidence

> **Status:** TASK 0 REVIEW GREEN; ZERO-TRACKED-RESIDUE RE-PIN REVIEW GREEN;
> TASK 1 LEDGER SUPERSEDED; TASK 2 STOPPED BEFORE CLASSIFICATION; TASK 1R
> AMENDMENT REVIEW NEXT
>
> **Last independently reviewed execution amendment tip:**
> `a4f45a7e87030f8abc09d0122c14b90e123b3d81`
>
> **Current design amendment tip:**
> `b21b7b280108d6e1eff2562a204c27501bbda075`
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

## User Scope Supersession - Zero Tracked PostgreSQL Residue

The user clarified that the absent PostgreSQL service owns no current ArkScope
capability. The final tracked tree must therefore retain no PostgreSQL-only
code, test, dependency/config surface, comment, docstring, test name, fixture,
Docker/SQL restore support, dump/manifest, archive/history document, or
program-governance narrative. Mixed current files retain only local product
behavior and remove PostgreSQL branches and historical contrast. Git history,
not a second tracked archive workflow, is the repository record.

The design was amended in two docs-only commits:

```text
379227cb94f96c1bf8bd97f64b515de62100f15e  zero tracked residue and five dispositions
74b1aba79b26c81cab6c5616deb587d54e1eddbe  retire PG-named smoke; retain only a positive local-runtime gate if measured
b21b7b280108d6e1eff2562a204c27501bbda075  make the later CLI census independent of retired PG inventory
```

The final amended design SHA-256 is
`a13418721fbe1abd931d13477a204c1b30d50cec5e1785c38460a133422c3391`.
The re-pinned inventory plan SHA-256 is
`fa37140246a4d0e82e8347eb0de4a53db705d8dc5fc8b7d8a43857b0a90ee5f5`.
It removes `retain_archive_asset`, `historical_reference`, and
`archive_history`; every tracked archive/history candidate now maps to exact
no-tail deletion, while a mixed current authority maps to bounded local-only
rewrite. The current PG-named smoke path cannot survive closeout. Any measured
positive startup/scheduler/dynamic-route behavior moves to a positively named
local-runtime gate with no PG-specific comment, fixture, or negative contract.

The inventory remains docs-only. Existing tracked archive/history candidates
must stay byte-identical while being inventoried; no deletion is authorized in
this line. The later reviewed no-tail plan must also append this program's
post-source-tip generated design/plan/evidence/ledger paths to its closeout
delete/modify ledger, because candidate-source freezing intentionally excluded
self-generated authority. An untracked private dump and the three remote
archive tables remain outside this tracked-tree operation and separately
destructive-gated.

The legacy-agent CLI census does not keep this inventory alive. It starts from
post-no-tail master, independently re-scans surviving entrypoints, and may use
only a neutral external-packet seed of surviving `path:symbol` identities.

Task 0 and its immutable candidate-source tip remain valid because both design
amendments are docs-only. Task 1 remains stopped. Focused review must accept
the exact loopback interception and this schema/scope re-pin before Batch A
resumes; no prior rejected runtime or partial candidate result becomes
admitted evidence.

## Task 1 - Complete Candidate Universe

Independent review returned GREEN for `4c6b8d44..a4f45a7e`, including the
exact loopback interception and the zero-tracked-residue re-pin. Batch A then
resumed from the backend runtime gate. No rejected pre-review runtime was
reused.

### Runtime outcomes

The exact loopback node passed in a fresh process with exactly one guarded
`socket.create_connection(("127.0.0.1", 9), timeout=1.0)` refusal, no operating
system socket call, and no unexpected guard hit. A second fresh process ran
the complete 64-file backend candidate set: `1,525 passed / 28 skipped / 0
failed`. Its 1,553 outcome IDs are byte-identical to the backend candidate
projection at SHA-256
`ce487d2ad37d05983925664d4774901b557379df4a7e29244635c72c92a5d5ef`.
The only allowed interception is the reviewed node; all other socket attempt
counts are zero.

The frontend used the pinned root binary `vitest/4.1.8`, with no `npx`, package
manager, install, or download fallback. The ten candidate files ran
sequentially and produced `126 passed / 0 skipped / 0 failed`. Normalized
runtime IDs are byte-identical to the frontend candidate projection at
SHA-256
`7ffa4e8129a50bf7ddac45c06263757cacc23df4dcfddb4678e54fc8fb76c256`.
A packet-only Node preload guard loaded in 11 processes and recorded zero
blocked network attempts; unmocked `fetch`, TCP, or UDP would have failed
before an external request.

### Candidate closure

The dynamic route witness remains 175 rows at
`488231c63e8c9bb0a28a6baf5e972c959c7eeddf9cc5fa10cdffc3330bc95aea`.
Static and dynamic product projections each contain 171 route/method rows,
with 25 mounted router modules and zero unexplained row on either side. The
sanitized package witness remains byte-identical to Task 0. The three
git-crypt paths produced zero PostgreSQL plaintext candidate, after their
base/main blob equality had already been established.

The committed unclassified ledger contains 6,902 unique rows at SHA-256
`dcfa639236a64dbe23dcd537471c0cc050c0fdc378ea6dba63bff7661e546e1d`:

```text
archive_manifest       16
ast                  1200
cli_registry           47
documentation         233
dynamic_route            2
environment_metadata     2
package_manifest          2
test_collection        1679
text_search            3721
```

Seven canonical raw source files have pairwise-zero candidate-ID overlap. A
second process rebuilt the ledger, source summary, and all 21 pairwise rows
byte-for-byte. The raw source counts and SHA-256 values are in
`candidate_union_summary.json`; classification has not yet begun.

### Cleanup, rejected attempts, and boundary

The candidate runtime generated 320 ignored cache/test files in the detached
source worktree and three files in isolated packet runtime roots. Their
path/size/SHA manifests are respectively
`ade195af795d8ba0dddd86298e1caf4cc6eec5916541d4646900042d93163a0d` and
`524400fa30e5b2b32076752889b219eed38478b566287a23150336b1682d5aec`.
They were removed only after manifesting; the candidate source had no tracked
delta, its real `data/` root was empty again, and the detached worktree was
removed.

Four bounded operator attempts are rejected and recorded: a command-policy
rejection before pytest started, an unsupported Vitest option before test
collection, a wrong-column frontend projection rejected by `cmp`, and a
missing output redirection in the first cleanup-manifest loop. None changed a
candidate, source byte, test body, or admitted outcome.

Packet `/tmp/pg-runtime-inventory-task1-4c6b8d44` contains 77 checksummed
payloads; `SHA256SUMS` SHA-256 is
`8fde44271a2cfda261a2c2afe5ba569a8cec80747af7ba480a7945feadcbfc90`.
No product/test/dependency/config byte, provider, remote database, production
SQLite file, FastAPI lifespan, scheduler, tracked archive, private secret,
merge, or push was touched. Task 2 classification now proceeds under Batch A
and remains the required review stop.

## Task 2 Stop - Incomplete PG Vocabulary

Task 2 created only a packet-local path summarizer and no tracked
classification authority. Before assigning any disposition, reachability,
consumer method, CLI class, test role, or documentation status, a direct
source-tip check found that the Task 1 scanner did not emit standalone `PG`
or several identifier forms. The reviewed zero-residue ruling requires those
names to enter the candidate universe; Task 2 owns their eventual
classification.

The first direct witness is
`src/macro_calendar/__init__.py:20` at exact source tip
`4c6b8d44ce2e768e95b822b11f618cc40f4bb9f0`: its docstring names the old `PG`
`MacroCalendarStore`, while the 6,902-row ledger contains no candidate whose
path is `src/macro_calendar/__init__.py`.

A bounded diagnostic scanned immutable Git blobs, excluding the three
git-crypt paths from generic plaintext inspection and skipping five tracked
PNG files as binary. It retained only path, line, byte column, normalized word
or bounded identifier, never surrounding source text. Against the old ledger's
317 distinct paths it found:

```text
standalone ASCII-word PG                  2,645 hits
standalone hits on absent candidate paths  150 hits / 54 paths
PG identifier morphology                    551 hits
identifier hits on absent candidate paths    50 hits / 13 paths
semantic tracked path names                  33 hits / 0 uncovered paths
```

Examples among the missing paths include current product/test surfaces
`src/analyst_consensus.py`, `src/api/routes/macro_calendar.py`,
`src/macro_calendar/__init__.py`, `src/news_providers.py`,
`tests/test_ibkr_gateway_lock.py`, `tests/test_profile_state.py`, and
`apps/arkscope-web/src/SettingsProviderConfig.test.ts`, plus dated documents
that the later no-tail line must delete exactly. Identifier hits also expose
`sourcePg`, `sourcePgFallback`, `SettingsPostPgExitStorage`, and a
`test_*_never_pg` name. Lexically similar but unrelated terms such as Unix
`PGID` are still candidates and require explicit `lexical_non_surface`
adjudication; opaque `package-lock.json` integrity strings are excluded from
the identifier projection rather than misrepresented as source identifiers.

This triggers hard stop 12. Commit `4a03c7ee` and packet
`/tmp/pg-runtime-inventory-task1-4c6b8d44` remain immutable dated evidence, but
their `6,902/dcfa6392...` ledger and candidate runtimes are superseded and
cannot admit Task 2. Task 1R must regenerate the complete source union from the
unchanged source tip, rerun the full expanded backend/frontend candidate
suites, replace the ledger in a new docs-only commit, and receive focused
review before classification resumes. This amendment does not predict the new
counts or hashes.

No product, test, dependency, configuration, provider, database, production
SQLite, FastAPI lifespan, scheduler, secret, encrypted plaintext, archive,
merge, push, no-tail, or CLI-retirement action occurred. The implementation
tree remained clean before this docs-only amendment.

Stop packet
`/tmp/pg-runtime-inventory-task2-stop-pg-vocabulary-4a03c7ee` contains 26
checksummed payloads. Its final manifest SHA-256 is reported at the focused
review handoff rather than self-referenced from this amendment.
