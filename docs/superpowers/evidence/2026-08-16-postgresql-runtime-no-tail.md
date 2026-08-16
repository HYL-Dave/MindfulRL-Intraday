# PostgreSQL Runtime No-Tail Evidence

> **Status:** TASKS 0-3 COMPLETE THROUGH PRODUCT TIP `ac2d3395`; SECTIONS
> 0.7D AND 0.7E CLASS A AMENDMENTS RECORDED; USER-APPROVED A/B CLASSIFICATION ACTIVE;
> BATCH B TASKS 4-5 ACTIVE; TASK 6 REMAINS THE COMBINED REVIEW GATE; TASK 7,
> MERGE, PUSH, LIVE TRAFFIC, AND PRIVATE OR REMOTE MUTATION NOT AUTHORIZED
>
> **Plan/amendment tip:** `9dcbbe4c108f315e16a754bdf5454cef39996857`
>
> **Product base:** `d4677c3d5b8579f95621a62ed056620a083ad1c8`

## 1. Task 0 boundary

Task 0 ran from a clean isolated worktree with only the repository-root
`node_modules` linked. App-local `node_modules`, `config/.env`, main-tree data,
and every private runtime asset were absent. An empty real `data/` directory
and packet-local runtime root were used. `DATABASE_URL` was unset.

The candidate source `4c6b8d44ce2e768e95b822b11f618cc40f4bb9f0`, product base, and plan tip have
zero product, test, or dependency path drift. Main and implementation trees
were clean before evidence edits. No test body, lifespan, scheduler, provider,
database, production asset, or live route was executed.

## 2. Fresh canonical identities

The backend was collected under a packet-local socket guard that rejects
`connect`, `connect_ex`, and `create_connection`:

```text
backend full       4,394  b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb
reporter schema    1
seen               0
nonpassing         0
exit               0
frontend full      1,177 / 101 files
frontend SHA-256   90f56093290c70a27369296ec8d8c7de99d084a091134994ae6451bc8e45743b
```

The guarded backend stream and explicit-exit PTY frontend stream are each
byte-identical to the first fresh collection and the merged inventory bases.
Vitest was invoked only as the pinned root binary at `4.1.8`; no `npx`, package
manager, install, or download fallback was used.

## 3. Reconstructed ledgers

Two packet-local implementations independently rebuilt the plan ledgers and
source projections:

The following values are the immutable Task 0 generation. Section 8 records
the later truthful-replacement re-pin; these dated values are not current
admission identities after the second Task 1 stop.

- path authorities: `161` delete, `174` modify, `1` add, and `22` protected;
- measured DAL contract: `11` assigned surfaces and exactly `36` methods;
- contrast only: all `18` local-rewrite surfaces would incorrectly yield `58`
  methods;
- whole-file retirements: `101` nodes;
- historical retirements: `22` nodes;
- backend replacements: `40` pairs;
- backend additions: `7` nodes;
- frontend replacements: `18` pairs;
- staged backend counts: `4,382`, `4,349`, and `4,278` with every full hash
  matching the plan;
- final frontend: `1,177 / c570a551...`;
- final dynamic routes: `173 / e0d8bf3c...`;
- inventory-focused stages: `1,897 -> 1,885 -> 1,852 -> 1,781`, with Task 4
  retaining `1,781` under its exact final hash; and
- owner projections: Task 1 `444`, Task 2 `147`, Task 4 frontend `54`.

The Python AST reconstruction consumed exactly `1,029` candidates in `57`
paths and produced:

```text
retained node bodies   194  78ea981809b6ef2e2430d6c2b8da49e7790c9bb6c874e76b4799287d32752f76
shared helper scopes    41  0dd3a8c15d575e43294e77fdd9789cd6b01695b362294ff147f9c5328e7f81b8
module scopes           34  75be6faa2a3bcd491cfbe7ed2e6e4b423d4e1f265375c9c965c9b973a569c42a
aggregate              269  540bcb9f6e85bc5091812826b3804cfb8cd1d1829ea0536782261f0e3600e4a4
```

The independent TypeScript projection produced the exact `10` retained-body
and `9` shared-fixture streams. A fresh AST scan also found exactly `12`
direct `psycopg2` importers in the plan's `5 / 3 / 4` task partition.

## 4. Protected and encrypted boundaries

The protected aggregate is
`7e9fa65847e86c9296c541b546ce472d1a7d467b6392a089c116dc02563e5cb6`.
All individual rows match. The three git-crypt paths are tracked with the
`git-crypt` filter and their implementation-tip blob IDs equal unlocked main;
no plaintext was copied into the packet or used as absence evidence.

The tracked inventory `MANIFEST.sha256` passed in full. The packet leak scan
found no connection URI, email, host-home path, bearer value, or real secret.
The only token-shaped strings are two existing synthetic parameter values in
`test_probe_harness` node IDs whose explicit purpose is redaction testing;
they are retained because the canonical collection stream is byte authority.

## 5. Rejected operator evidence

Two frontend list attempts completed at the command wrapper's 30-second yield
boundary without an observable exit status. Their complete JSON was not used
for admission. A PTY run was polled until explicit exit `0`, normalized, and
byte-compared to the canonical stream. An initial leak-audit command with an
invalid shell quote was likewise rejected and replaced by the bounded scan
described above.

## 6. Packet and authorization

Evidence packet: `/tmp/arkscope-pg-no-tail-task0-d4677c3d`

```text
payloads             74
SHA256SUMS SHA-256   96b3d22d975137001f9ea375fe7850b98850e2ddb18a152d36159cc5e2f87a02
```

The user accepted the recommended two-stage batching. Batch A comprises
Tasks 0-1 and then stops for focused implementation review. Batch B comprises
Tasks 2-5 only after that review returns GREEN. Per-task commits and packets,
RED-first execution, every hard stop, Task 6 combined review, and Task 7 merge
authorization remain unchanged. This ruling does not authorize push, live
traffic, production/private/remote mutation, or legacy-agent CLI retirement.

## 7. Task 1 shared-fixture stop

Task 1 established its five RED contracts, implemented the direct local
composition far enough for those five nodes to pass, and recollected the
exact planned `4,382 / c7b9a77a...` stream with zero test bodies seen. The
first native 444-node owner run then reported `438 passed / 1 skipped / 5
failed`.

Four failures are already within the reviewed edit ceiling. The fifth is
`TestNewsTools::test_search_news_by_keyword`: its shared
`_HermeticMarketBackend` lacks the now-direct `query_news_search` method, but
that helper is absent from every Section 0.7a stream. Adding a product
method-presence fallback would violate the local-capability design, so stop
condition 19 was applied. Product and test edits remain uncommitted.

The same run exposed an evidence-boundary failure: the stale SEC-cache fake
raised at the newly direct cache call and the test reached the real SEC
fallback, returning current AAPL data. This unplanned read-only request is
rejected evidence. No further provider-capable runtime command is authorized;
the bounded amendment requires socket-guarded replay after the local cache
fixture is corrected.

One sandboxed FastAPI/AnyIO owner stalled on both unchanged merged-master and
Task 1 bytes, while the Task 1 node passed natively. That transcript is
retained as rejected environment evidence rather than being attributed to
product code.

Partial Task 1 packet:
`/tmp/arkscope-pg-no-tail-task1-d4677c3d`. It contains the exact collection
stream, pre-amendment owner transcript and exit status, uncommitted product
patch, and rejected sandbox stack. Task 1 may resume only after focused review
accepts Section 0.7b; Tasks 2-7 remain unauthorized.

## 8. Task 1 intermediate-runtime and fixture-ceiling stop

Focused review accepted Section 0.7b at `38e447b0`. The resumed run corrected
only its bounded fixtures: all five new contracts passed under the socket
guard, and the exact first-generation owner set finished `443 passed / 1
skipped`. Backend recollection remained the then-pinned 4,382-node stream.

The next required inventory-focused backend partition contained 1,714 nodes
after isolating the one reviewed loopback-refusal node. Its complete pytest
summary was:

```text
1,639 passed / 19 skipped / 45 failed / 11 errors
```

That run is rejected evidence. Its shell wrapper piped through `tee` without
`pipefail`, so a shell status cannot be used as an outcome. The pytest summary
and per-node failures remain authoritative for diagnosis only. No product or
test commit followed it.

The failure set proved that the literal collection ledger and the intermediate
runtime ledger had been incorrectly conflated. Three observed failures were
already inside whole-file owners scheduled for atomic deletion, and all 101
nodes in those six files are obsolete contracts rather than Task 1 runtime
admission owners. Section 0.7c therefore keeps the full 1,885-node Task 1
collection identity but defines an exact 1,784-node survivor runtime set. It
also precomputes the analogous Task 2 survivor set, excluding only the 71
whole-file nodes deleted in Task 3.

The remaining failures split into already-authorized fixture evolution and
exact ceiling gaps: a second hermetic news backend, one freshness body, one SA
digest helper plus five direct-patch bodies, and six false surviving test IDs.
No product method probe or compatibility fallback is authorized. The six IDs
become one-for-one truthful replacements, yielding these mechanically rebuilt
current authorities:

```text
backend replacements      46  f7ac08c4000baddaa9969d7895054ade3024ea224536bdb68286737891cf36ad
Task 1 backend          4,382  ce7c045fab7b4fde2598660e98c5e67964ac0c8871b8d8aca7d3d150c3e90cc8
Task 1 focused          1,885  19ff8f6027ed399b0701fb2840cb3e0658cee860f5de8334f68a6522f826bcca
Task 1 owners             472  cb454b785b7fdfc645a4c5f3765cb8a70dc280ad5f63a76c4dcf0fbd8d246578
Task 1 runtime          1,784  5bc41848aec5327b042c25248f1d6da46cb28c5e8a21faaf7e681d11bc1db0c5
```

The rejected run also created an untracked synthetic `MagicMock/` SQLite path
because a stale fake exposed a mock instead of the current local path. It is a
test artifact, not product state; its receipt is recorded and the directory is
removed before amendment admission. Task 1 remains stopped with all product
edits uncommitted. Tasks 2-7 remain unauthorized pending focused review of
Section 0.7c.

The updated partial packet remains
`/tmp/arkscope-pg-no-tail-task1-d4677c3d`; its manifest is regenerated only
after the amendment diff is final.

## 9. Task 1 local-capability cutover complete

Focused review accepted Section 0.7c at `06e952c4`. The resumed implementation
changed exactly `62` inventory-authorized paths: `61` existing modify paths and
the sole admitted addition, `src/tools/backends/local_capabilities.py`. The
outside-authority path projection is empty. Product commit `693cf7af` creates
the exact 36-callable non-runtime protocol, constructs the local market/SA
composition directly, and cuts every measured retained consumer to its current
local owner. It adds no DSN inference, nominal-type routing, method-presence
probe, compatibility fallback, or provider request.

All final identities and runtimes are admitted:

```text
backend collection       4,382  ce7c045fab7b4fde2598660e98c5e67964ac0c8871b8d8aca7d3d150c3e90cc8
focused collection       1,885  19ff8f6027ed399b0701fb2840cb3e0658cee860f5de8334f68a6522f826bcca
five new contracts           5  passed
Task 1 owners               472  471 passed / 1 skipped / 0 failed
backend survivors         1,614  1,612 passed / 2 skipped / 0 failed
frontend survivors          170  170 passed / 0 failed
recombined survivors      1,784  5bc41848aec5327b042c25248f1d6da46cb28c5e8a21faaf7e681d11bc1db0c5
```

The backend/frontend survivor streams recombine byte-for-byte to the literal
1,784-node authority. The full 1,885-node stream remains collect-only because
its 101 future whole-file retirements are intentionally not intermediate
runtime contracts. Backend collection used the deterministic reporter with
zero test bodies seen and matched the amended stream byte-for-byte.

A packet-local import projection ran app, both agents, CLI, scheduler, native
host, `LocalMarketBackend`, `SACaptureBackend`, `MacroCalendarLocalStore`, and
`JobRunsLocalStore` in ten independent child processes. Every child installed
the socket guard before import; all ten imported successfully and none loaded
`src.tools.backends.db_backend`. Its first invocation is rejected because the
packet directory accidentally became the import root. The admitted harness
requires repository-root cwd explicitly, so the correction changes only the
evidence seam and cannot mask a product import.

All 22 protected paths reproduce aggregate
`7e9fa65847e86c9296c541b546ce472d1a7d467b6392a089c116dc02563e5cb6`.
Exact owner and complete delta pre/post hashes, the product patch, all runtime
transcripts, the import projection, and the survivor recombination are in
`/tmp/arkscope-pg-no-tail-task1-d4677c3d`. Its final manifest contains `66`
payloads and has SHA-256
`af8ef08e68469dc499c2a734102651f8b0e15e69373da5804a9c3e67f59bcb96`;
all entries pass `sha256sum -c`.

No native full-suite claim is made at this intermediate stage. No production,
private, remote, provider, or live-route asset was read or mutated. Tasks 0-1
now stop at the user-approved Batch A combined implementation-review gate;
Tasks 2-7 remain unauthorized until that review returns GREEN.

## 10. Task 2 local-runtime cutover complete

Batch A review returned GREEN and unlocked Tasks 2-5. Product commit
`3e3cb90b` removes the five exact migration/probe paths, unmounts both
migration routes, removes the unavailable news-write mode, and replaces the
network reachability probe with scheduler-state-first plus unconditional local
job-history supplementation. The renamed local-authority test has no alias or
compatibility export.

The admitted identities and runtimes are:

```text
backend collection       4,349  04e93190119d1134903182a61f6ea495d1445ebd5784878196bca2baa49bebc6
focused collection       1,852  1c7f9a06d9518b48355ac952f4e09352862c6628dfaf0c5ff35cd7ae53ad73e0
Task 2 owners               147  147 passed / 0 failed
backend survivors         1,611  1,609 passed / 2 skipped / 0 failed
frontend survivors          170  170 passed / 0 failed
dynamic routes              173  e0d8bf3c01e57bfb5403c68c16aac376be225db56eb638ca44d7eb218acfb37e
```

The route stream came from the real FastAPI lifespan and scheduler under
scratch stores, sealed providers, and a socket guard. It contains neither
migration route and matches the literal target byte-for-byte. Removing the
local-history supplement made the dedicated continuity owner fail because no
last-attempt fact was admitted; restoring the complete source blob returned
the owner to GREEN. All 22 protected paths retain aggregate
`7e9fa65847e86c9296c541b546ce472d1a7d467b6392a089c116dc02563e5cb6`.

Two operator probes are explicitly rejected: a process-name cleanup query
matched its own command line, and an initial staging command named paths that
had already been deleted. The admitted cleanup uses process cwd plus Python
executable type and reports zero worktree Python process. The cached commit
path set was required to equal the 14-path Task 2 delta before commit.

Packet `/tmp/arkscope-pg-no-tail-task2-693cf7af` contains `57` payloads at
manifest SHA-256
`41e885b5860b69d47edd5bcbb45b7c517384762f176586b054d73d9fc85f07db`;
all entries pass `sha256sum -c`. It includes exact last-containing and
`git show` recovery receipts for all 14 rewritten or removed pre-commit
surfaces. No live provider, production database, private configuration,
remote service, merge, push, or legacy-agent CLI action occurred. Batch B now
continues through Tasks 3-5 and then stops at Task 6 for combined review.

## 11. Task 3 live-FRED alias-seam stop

Task 3 removed the exact seven foundation paths and reached its pinned
`4,278 / 80037a1b...` backend collection. The exact runtime-survivor union is
also GREEN: backend `1,609 passed / 2 skipped` and frontend `170 passed`.
The first backend replay exposed two mixed IV-authority tests that still read
the now-deleted backend path. That complete run is rejected collateral
evidence. Replacing only the dead owner row with the current local capability
and SA backend owners made both tests GREEN; the exact 1,611-node backend
replay then reached its admitted result.

The foundation gate, its negative self-tests, requirements closure, isolated
`python -S` import probe, and protected-byte check all pass. A stricter AST
projection initially rejected three arbitrary retired attribute references in
`tests/live/smoke_fred.py`; that result is retained as an intentionally
overbroad diagnostic because Task 4 still owns reviewed current-authority
attribute and prose cleanup. After narrowing the projection to Task 3's
declared import/inheritance/type/alias surface, one real executable alias
remained: the live smoke saves and restores `ing.MacroCalendarStore`, although
the current ingestion module exposes and constructs only
`MacroCalendarLocalStore`.

The inventory already owns the path and all three stale attribute coordinates
under `retain_operator_remove_pg_branch`, but the Task 3 owner list omitted
it. Section 0.7d admits only the exact three-line current-owner rewrite and
preserves all staged identities. Product edits remain uncommitted. The packet
retains the overbroad diagnostic, alias-specific RED, complete runtime
transcripts, structural projections, and pre-amendment product patch. Focused
amendment review is the sole next gate; Tasks 4-7 remain unauthorized.

Focused review returned GREEN for Section 0.7d and independently reproduced
the broken seam, exact three-line correction, unchanged identities, and
`48/48` packet manifest. The user then adopted the Class A/Class B amendment
rule in plan Section 1.1. Section 0.7d meets all four Class A predicates and
was applied without another intermediate wait.

Product commit `ac2d3395` deletes the seven exact foundation paths, removes
the obsolete driver declaration and direct imports, keeps only current local
macro/job-run behavior, and updates the live FRED smoke through its existing
local-store injection seam. The resumed collection is byte-identical to the
pinned Task 3 stream:

```text
backend collection       4,278  80037a1bd0d82270eeef633b0b2640c0a7fd2680b51de906811b85d87755f5e3
focused/runtime union    1,781  19443b6f2665d5f1ec677de6430687e8f5a41d39bfe54dcbaba9d748fb46b2d5
Task 1 owners              472  471 passed / 1 skipped / 0 failed
Task 2 owners              147  147 passed / 0 failed
backend survivors        1,611  1,609 passed / 2 skipped / 0 failed
frontend survivors         170  170 passed / 0 failed
```

The final foundation gate and negative self-tests, narrowed AST projection,
requirements closure, isolated `python -S` import probe, FRED owner suite,
and all 22 protected paths are GREEN. The packet records all rejected command
starts separately; none reached a provider, product runtime, private config,
or production asset. Every one of the 16 rewritten or removed paths has an
exact pre-tip/product-tip row and a successful `git show` recovery digest.
Packet `/tmp/arkscope-pg-no-tail-task3-be16cdd3` contains `81` checksummed
payloads; its `SHA256SUMS` SHA-256 is
`4b991f16f3d7b09b6dad43c17d4c9ae905f3d429568c19836228c64cb572c8c9`.
Batch B continues through Tasks 4-5; Task 6 remains the combined
implementation-review gate.

## 12. Task 4 presenter fallback Class A amendment

Task 4's zero-residue scanner is GREEN over every tracked product path with
the exact temporary-governance exclusion set. Backend and frontend collections
match the pinned final identities, and the two frontend owner files pass
`54/54`. The first sequential frontend full run completed `1,176 passed / 1
failed`; the sole failure was the i18n foundation owner reporting
`["src/marketDataDisplay.ts","presenter_return","normalized"]`.

The rejected transcript is
`/tmp/arkscope-pg-no-tail-task4-1499d827/frontend-full.txt`. It contains no
provider request or production/private access. Direct scanner replay identified
both the raw default return in `newsWriteRouteLabel` and the machine-value
predicate embedded in `newsReadSurfaceLabel`'s return expression. Both are
inside the exact inventory-owned presenter rewrite. Plan Section 0.7e
classifies the bounded localized fallback plus behavior-preserving predicate
relocation as Class A. All node IDs and staged hashes remain unchanged; the
correction proceeds without an intermediate review wait and is replayed at
Task 6.
