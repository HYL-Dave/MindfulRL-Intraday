# PostgreSQL Runtime No-Tail Evidence

> **Status:** TASK 0 COMPLETE; USER-AUTHORIZED BATCH A TASK 1 ACTIVE;
> TASKS 2-7, MERGE, PUSH, LIVE TRAFFIC, AND PRIVATE OR REMOTE MUTATION NOT
> AUTHORIZED
>
> **Plan tip:** `05e159261a409da6451105d1d3bcf7e9a7d62661`
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
