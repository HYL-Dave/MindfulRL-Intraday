# Scripts Retirement Tranche A Evidence

> **Status:** IMPLEMENTATION IN PROGRESS
>
> **Date:** 2026-08-01
>
> **Decision authority:** `d89d433c`
>
> **Reviewed plan tip:** `e128786daf44e75ac95a345bfa0ef59077c2ca62`

## 1. Grounding And Protected Boundaries

- Branch/worktree: `codex/scripts-retirement-decision` at
  `/tmp/arkscope-scripts-retirement`.
- Reviewed plan:
  `docs/superpowers/plans/2026-08-01-scripts-retirement-tranche-a.md`,
  SHA-256
  `f783a8aaba0c9b1d54ef987b947f5373a0868f63fd452c633df0946109240566`.
- The committed authority bytes at `d89d433c` remain
  `0ae53860bb8407d07f7f7aad574530b60488f52c6049964fc9555f563c2bc791`.
  The reviewed worktree status-evolved copy is
  `40749aec44871f26526721b477e49fb458297db5e34529ba553a1c6e2746ad24`.
- The main worktree's protected untracked predecessor was not edited and
  remains
  `79d4eac97d7692684d83f0a067f5987fe434bb76746b98af3e44f1c8ba4bf277`.
- The tracked `scripts/` inventory is exactly
  `34 / 59cf1e8e6cbdbfa0877aafad87ae8fd7107222afba2030fb87066376b1ee66a5`.
- `config/scoring_keys.txt` was checked only through metadata: it exists in
  the main worktree, has mode `0600`, and is ignored by Git. Its contents,
  byte count, and digest were not read. It was not copied into this
  worktree.
- The native boundary reused the EIR-002 assets without modification:
  - wrapper:
    `e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f`;
  - reporter:
    `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928`;
  - wakeup probe:
    `10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e`;
  - Node.js: pinned executable and version `v22.14.0`;
  - main `package-lock.json`:
    `5322cb03099b873066b7572c02db68a427ddc7509fdc9850cf9d8d3948a19f2c`;
  - main `node_modules/.package-lock.json`:
    `4dd5182f8111b54dd608344c45513f3b310f39321a3cc9832b60cefc2fa241ff`.

No product, test, provider, scheduler, production database, browser profile,
or user-configuration path changed during Task 0.

## 2. 159-Node Direct Disposition

The reviewed disposition was materialized before any implementation edit:

| Stream | Rows | SHA-256 |
|---|---:|---|
| Direct disposition | 159 | `15f54aeae019936c660c48eecc6165156ae050604fe4267f780250f98f72728a` |
| Additional domain-core-only removals | 54 | `5259ce298190e13ea6a7a4456dcb54b3d0f3c54c0a65e64a56da975209e175d0` |

Every direct row has exactly one closed disposition:
`remove_migration`, `remove_diagnostic`, `remove_paid_probe`, or
`retain_scoring`. No row is duplicated or unclassified. The 36
score-coupled nodes remain retained through Tranche A. The 54 additional
nodes leave only with their reviewed spent domain cores.

The ledger files remain under
`/tmp/scripts-retirement-tranche-a/ledger/`; they are execution evidence,
not repository inputs.

## 3. Collection Ledger

The physical collection and executable provider-safe admission have different
roles and are intentionally both retained:

| Stage | Delta | Nodes | Ordered stream SHA-256 | Task 0 status |
|---|---:|---:|---|---|
| Physical collect-only base | `+0/-0` | 4,730 | `c34de9a0fe53e400409d3ec26d75a8c907ee277b121279693ea9f69c8638aabb` | reproduced |
| Paid probes retired / provider-safe base | `+0/-2` | 4,728 | `49e4a32b5f536cea97053578f2fba4456ffbbe0c10a4b66540c4f26d2b55329f` | admitted |
| Diagnostics retired | `+0/-5` | 4,725 | `64ce4a619039fa586f065533b900416b1fd3fcbf6d78a99a43c9295a02a83e1d` | reproduced |
| Tranche A final | `+0/-177` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` | preconstructed |

The provider-safe reporter's 4,728-node collected stream is byte-identical to
the preconstructed `stage-paid.nodes` stream. Its collected and seen streams
are also byte-identical. The Task 3 collected stream is byte-identical to
`stage-diagnostic.nodes`. The final-stage identity remains a target, not a
claim that implementation has occurred.

The physical collect-only reporter is
`/tmp/scripts-retirement-tranche-a/task0-collect.json`, SHA-256
`ab6487af8fd731ae99f0c4dcec0147d9df0528487da24c5c74e5a7f5807822b9`.
It recorded 4,730 collected nodes and intentionally executed none.

## 4. Structural RED/GREEN

### 4.1 Task 0 grounding

Task 0 made no implementation edit and therefore did not manufacture a new
product RED. It established the pre-edit safety gates:

- retained scoring authority:
  `36 passed`;
- protected current-shape anchors:
  `94 passed, 18 skipped`;
- provider-safe canonical base:
  `4656 passed, 72 skipped, 0 failed`;
- every admitted collected node was seen exactly once.

The focused commands left ordinary Git status empty, did not create
`comparison_results/`, left the required empty `data/` root empty, and did
not create `src/data`.

Independent review accepted Task 0 at `b7000c2b` and authorized Task 1.

### 4.2 Task 1 manual-smoke ownership

Before the move, `tests/live` did not exist. The reviewed plan wrote
`test ! -e tests/live` while also saying to expect a non-zero result. Those
two statements cannot both hold: the command correctly returned zero for the
missing directory. The intended positive owner assertion,
`test -e tests/live`, returned one and supplied the structural RED. This
command-level correction did not change the reviewed target or scope.

Task 1 then moved, without executing:

| Previous path | Current owner |
|---|---|
| `scripts/live/sdk_driver_smoke.py` | `tests/live/sdk_driver_smoke.py` |
| `scripts/live/sdk_route_smoke.py` | `tests/live/sdk_route_smoke.py` |
| `scripts/p1_2/smoke_fred.py` | `tests/live/smoke_fred.py` |
| `scripts/live/README.md` | `tests/live/README.md` |

The new README states that the files are manual, never default-collected,
never run by automated admission, and can require real credentials, network,
Gateway state, provider entitlement, and spend. It contains the three exact
manual commands. No file in the directory is named `test_*.py`.

Static compilation passed for all three files with bytecode redirected to
`/tmp/scripts-retirement-tranche-a/task1-pycache`. The hermetic FRED test
returned `44 passed`; no live smoke ran.

The first collection-wrapper invocation remained inside the managed sandbox.
Its wakeup probe returned `false/1/0`, so the wrapper rejected the boundary
before creating a stage, report, transcript, or `comparison_results/`.
The authorized native retry then produced:

- report:
  `/tmp/eir002-green-baseline/reports/scripts-a-task1-collect.json`,
  SHA-256
  `ab6487af8fd731ae99f0c4dcec0147d9df0528487da24c5c74e5a7f5807822b9`;
- transcript SHA-256:
  `4ced6c491ae59b5f4d1a944c2502ddcbc645443db0da795eab5dc48313c94cb8`;
- `4730` collected, `0` executed, exit status `0`;
- ordered collection
  `c34de9a0fe53e400409d3ec26d75a8c907ee277b121279693ea9f69c8638aabb`;
- zero node IDs under `tests/live/`; and
- a report byte-identical to the Task 0 physical collect-only report.

Collection imported the legacy Financial Datasets probe and created only the
empty directories `comparison_results/financial_datasets/`. After proving
that they contained no file or symlink, Task 1 removed those exact empty
directories with `rmdir`. No request function or live smoke executed.

### 4.3 Task 2 static extraction and research-utility retirement

Task 2 parsed the two Financial Datasets probe sources without importing or
executing either file. The static artifacts are:

| Artifact | Observation |
|---|---|
| Source call-site stream | `29` attempts; SHA-256 `ba84146cf8e6152d50433baa1b154b63488f9a6b741b450dc920bd637ea59ab0` |
| Sorted unique endpoint stream | `24` paths; SHA-256 `312584b67cc49eccff791832d0d2f54a51c90caf5f26305245d8f5bd1a8eec20` |
| Public OpenAPI snapshot | OpenAPI `3.0.1`, `54` paths; SHA-256 `2f17263f7a960fca93cd7662cf1be583c6ecb68b090313139ed1aca6db702b5a` |

The resulting
`docs/design/FINANCIAL_DATASETS_CAPABILITY_SPEND_DECISION.md` separates:

1. source-literal endpoint and query-key shapes;
2. the rejected Task 0 network observation, which is not cost, entitlement,
   availability, or spend evidence; and
3. official pricing, terms, MCP, documentation-index, and OpenAPI observations
   rechecked on 2026-08-01.

No Financial Datasets provider request or credentialed operation occurred
during Task 2. The document is a static decision input, not a capability
registry or spend authorization. Product registry, enforcement, Settings,
i18n, audit, typed `402`, and dashboard-link behavior remain a separate slice.

The two HuggingFace provenance files moved byte-for-byte:

| Historical owner | SHA-256 before and after |
|---|---|
| `docs/history/news-scoring/SCORING_PROMPTS.md` | `ac367ab76c4e9bc7e576316ebefb1e347f9f44c7b035f52af7b53c7bc5e0e8c0` |
| `docs/history/news-scoring/column_mapping.md` | `94217713d3d6f43e4d718e19178dde2cbb1597da72ec2ba2322745cbd8eba4b2` |

`docs/history/SCRIPTS_RETIREMENT_TRANCHE_A.md` now owns the migration,
diagnostic, dashboard-gap, unusual-options, Financial Datasets, and
news-scoring lineage. It does not promise to rebuild any retired executable.

Exactly these seven Task 2 files were removed:

```text
scripts/analysis/scan_unusual_activity.py
scripts/huggingface/merge_for_release.py
scripts/testing/test_financial_datasets_api.py
scripts/testing/test_financial_datasets_api_retry.py
scripts/visualization/README.md
scripts/visualization/data_loader.py
scripts/visualization/news_dashboard.py
```

At the Task 2 checkpoint, the fixed-ID
`scripts/diagnostics/probe_ibkr_news_bodies.py` remained for Task 3.

Before the post-edit collection, the ignored
`comparison_results/financial_datasets/` directory pair was present again but
contained no file or symlink. Task 2 made no attribution claim about which
external review/collection recreated it. Both exact empty directories were
removed with `rmdir`. After the paid probes were deleted, collect-only did not
recreate `comparison_results/`.

The post-edit collection result is:

| Fact | Observation |
|---|---|
| Collected / executed / exit | `4728 / 0 / 0` |
| Ordered collection | `49e4a32b5f536cea97053578f2fba4456ffbbe0c10a4b66540c4f26d2b55329f` |
| Reporter SHA-256 | `ec52c4678e251bdf99b3dfc75c34e9bdd505a2e77d4f3430948112751e4c7333` |
| Transcript SHA-256 | `05c64d5eed7603ed8fc00408fbf7f1f290f78841112e47d00f6328475e93ceba` |

The collected stream is byte-identical to the preconstructed
`stage-paid.nodes` target. Relative to Task 1, `comm` reports exactly the two
Financial Datasets probe IDs removed and no added ID:

```text
scripts/testing/test_financial_datasets_api.py::test_all_endpoints
scripts/testing/test_financial_datasets_api_retry.py::test_failed_endpoints
```

The repository Markdown-link scan checked 140 local links across 234 tracked
or newly created Markdown files and found zero unresolved targets. The
isolated worktree lacks intentionally ignored documentation/data artifacts, so
resolution used the main worktree only as a read-only fallback for those
pre-existing ignored targets. All new and moved Task 2 links resolve inside
the isolated worktree. `git diff --check` is clean.

### 4.4 Task 3 fixed diagnostic retirement

Before editing, the complete adapter file passed all 20 collected nodes. Static
classification and the reviewed ledger agreed that exactly three nodes
imported or verified the fixed diagnostic executable:

```text
tests/test_news_normalized_ibkr_adapter.py::test_probe_classifies_ibkr_unavailable_without_payload
tests/test_news_normalized_ibkr_adapter.py::test_probe_has_five_reviewed_default_cases
tests/test_news_normalized_ibkr_adapter.py::test_probe_output_never_contains_body_or_exception_payload
```

The structural deletion assertion returned non-zero while
`scripts/diagnostics/probe_ibkr_news_bodies.py` still existed. Task 3 then
deleted that 139-line executable, whose pre-delete SHA-256 was
`665752160adb273bb89d334ed77fd156efb2d2e9290fc90d3923113d16f02187`,
and removed only the three executable-only tests and their imports. No adapter
implementation changed. The remaining adapter file passed all 17 nodes.

The native collect-only result is:

| Fact | Observation |
|---|---|
| Collected / executed / exit | `4725 / 0 / 0` |
| Ordered collection | `64ce4a619039fa586f065533b900416b1fd3fcbf6d78a99a43c9295a02a83e1d` |
| Reporter SHA-256 | `d727494d1fdf4f42bc56d7bc45bf15178cddaa6dd0ff40d18d2544fdb825d595` |
| Transcript SHA-256 | `06dec15e2298343e50d1dae8c7e73625df67ecb354e6c3595192367ed68ef58a` |

The collected stream is byte-identical to the preconstructed
`stage-diagnostic.nodes` target. Relative to the 4,730-node physical base,
`comm` reports exactly the two paid-probe IDs and the three diagnostic IDs
missing, with no added ID. The first comparison command named a nonexistent
`base.nodes` shorthand and therefore produced no comparison result; the
recorded result was recomputed from the actual 4,730-node
`task2/task1.nodes` artifact.

Collection executed no test or provider operation, did not recreate
`comparison_results/`, and left no untracked repository artifact.

The pre-edit 20-node baseline import had created one ignored bytecode artifact
for the diagnostic. Before physical cleanup it was recorded as inode
`93465417`, size `4171`, mtime epoch `1785554210`, and SHA-256
`946f82e960aefcbe2a5fad82ab21a4a8aa1fd5e0c09f8c3e0cd4eb0135612dec`.
The exact `__pycache__` directory was moved, without changing that identity,
to `/tmp/scripts-retirement-tranche-a/quarantine/task3-diagnostic/`; the
resulting empty `scripts/diagnostics/` directory was removed. The worktree
therefore contains no physical diagnostic executable or bytecode residue.

## 5. Native Admission And Artifact Transactions

### 5.1 Rejected unfiltered attempt

The first native command ran the physical 4,730-node collection, including
the two Financial Datasets probe files. It completed, but is immutable
rejected evidence and is not an A/B ledger row:

| Artifact or fact | Observation |
|---|---|
| Reporter | `/tmp/eir002-green-baseline/reports/scripts-a-base-full.json` |
| Reporter SHA-256 | `9babd7b9f24dda99594436dfa98d10c2af86b1b59b3ea5f19313489f4b1c5b9e` |
| Transcript SHA-256 | `c913fea3daef11d4b61335a13e055a7782b241db51d46bec511ae6704fe4db2b` |
| Collection / seen / exit | `4730 / 4730 / 0` |
| Unauthenticated request attempts | 29 |
| Surviving response artifacts | 28; one retry reused an earlier path |
| Surviving HTTP statuses | `2x 200`, `1x 400`, `20x 401`, `4x 404`, `1x 410` |
| Pre/quarantined ignored-artifact manifest | `3e90b5ba1780547bdc8b77c5abf2cda5cf358f1c1fb82fd284f3b148545ca313` |

The native `env -i` boundary supplied no Financial Datasets credential
attributable to the user. These observations do not classify an endpoint as
free or metered, prove entitlement or capability, or establish account spend.
All response artifacts remain quarantined under
`/tmp/scripts-retirement-tranche-a/quarantine/task0-base-full/`.

Historical EIR-002 full-suite runs likewise used blank credentials but did
not exclude these two collected probes. Their reported test outcomes remain
valid; they must not be described as network-free.

### 5.2 Admitted provider-safe attempt

The sole admitted executable base used exactly:

```text
--ignore=scripts/testing/test_financial_datasets_api.py
--ignore=scripts/testing/test_financial_datasets_api_retry.py
```

It produced:

| Fact | Observation |
|---|---|
| Collected / seen | `4728 / 4728` |
| Passed / skipped / failed | `4656 / 72 / 0` |
| Exit status | `0` |
| Collected stream | `49e4a32b5f536cea97053578f2fba4456ffbbe0c10a4b66540c4f26d2b55329f` |
| Reporter SHA-256 | `6763aed1fe282eff53fa32f592d8dbd901b32071d1515b90b1403405a5bdad9d` |
| Transcript SHA-256 | `85d36015d7229ffc79200280d749b651bc152cf548ac8566fe1bbb7177280a07` |

`comparison_results/` was absent before and after the run. Therefore no
admitted Task 0 run imported or executed either Financial Datasets probe.

### 5.3 Generated-artifact transaction

The admitted run created 72 ignored files: 19 under `data/` or `src/data`
and 53 Python bytecode files. The exact evidence is:

| Evidence | Rows | SHA-256 |
|---|---:|---|
| Exact relative paths | 72 | `37d0917f8ba3ecf5af0f2084b19dcb15487edbf545c6f28b468abfa0c588bb11` |
| Path/inode/size/time metadata | 72 | `4322744fe4a9860e0fe28843c50619be41785343e44482e67044bb1444919270` |
| Path-plus-content hashes | 72 | `fee927932f1bf7fd000b7a90819809c2deb92eebd785020ae589c77fc3d63576` |
| Post-run `data`/`src/data` hashes | 19 | `e2cb36b42b5b269eb6ef0aa2246de96490bbf9500b4b15b3fffb944d390bf08d` |
| Post-run path manifest | 28 entries | `10dfad62cdc0280bb2dc3589438a9475b508b8d65fb12cfbcc71fc494ba25332` |

All 72 files were moved by exact relative path to
`/tmp/scripts-retirement-tranche-a/quarantine/task0-provider-safe-base/`.
The quarantined path-plus-hash stream is byte-identical at
`fee927932f1bf7fd000b7a90819809c2deb92eebd785020ae589c77fc3d63576`,
and every original path was absent after the move. Only empty directories
created by the run were removed; the deliberately empty `data/` precondition
and pinned `node_modules` symlink remain.

The first ignored-status comparison was rejected before use because its
inputs were not sorted. The comparison was rerun with both inputs under
`LC_ALL=C sort`; only that corrected 72-path result drove quarantine. No path
was moved or deleted from the invalid comparison.

After quarantine:

- ordinary status is byte-identical to its empty pre-run stream;
- ignored status is byte-identical to its pre-run stream, restored SHA-256
  `72bf9d96d03421a5372b4d6597d1948ebf13883ccec7b6551c55b6abb3178714`;
- `data`/`src/data` inventory is byte-identical to the pre-run empty stream;
- symlink inventory is byte-identical, restored SHA-256
  `3ddb95019fa533ceddd7502fe69d40ee6100388be7042c420270c0808bdf6452`;
- `comparison_results/` remains absent; and
- no pre-existing file was modified or moved.

## 6. Independent Review And Merge

Independent review reconstructed Tasks 0, 1, and 2 and returned GREEN at each
checkpoint. Task 3 is now review-ready. Its review must reconstruct:

1. the pre-edit 20-node adapter baseline;
2. the exact three diagnostic-only IDs and unchanged 17-node adapter contract;
3. deletion of the single fixed diagnostic executable;
4. exact `-5/+0` collection to `4725/64ce4a61...`; and
5. `comparison_results/` absence after collection.

Task 4 onward, merge, production interaction, provider requests, and any
Financial Datasets product-policy implementation remain blocked. The separate
product slice must own endpoint classification, the metered-request toggle,
billing-mode declaration, cache-first behavior, typed `402` handling,
fail-closed unknown endpoints, local usage disclosure, and Settings UI.
