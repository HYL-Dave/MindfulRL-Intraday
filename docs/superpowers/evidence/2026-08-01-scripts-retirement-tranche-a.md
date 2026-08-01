# Scripts Retirement Tranche A Evidence

> **Status:** TRANCHE A MERGED; CLOSEOUT REVIEW REQUIRED
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
| Tranche A final | `+0/-177` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` | reproduced |

The provider-safe reporter's 4,728-node collected stream is byte-identical to
the preconstructed `stage-paid.nodes` stream. Its collected and seen streams
are also byte-identical. The Task 3 collected stream is byte-identical to
`stage-diagnostic.nodes`. The Task 4 collected stream is byte-identical to
both `stage-final.nodes` and the independently prepared
`/tmp/scripts-a-final.nodes`.

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

### 4.5 Task 4 spent migration-gate retirement

The pre-delete consumer census covered all eleven migration CLIs, eight spent
domain cores, and twelve whole test files. Its 31-path manifest and content
manifest are:

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Exact target paths | 31 | `69fee0a654a46a4940c7e9c226a405c1fc4abd5a9ed178c3d43490c3d3b16681` |
| Path-plus-content hashes | 31 | `c9573cffe1743a81649ede23fcac14796b422c04abf12fc93377b6de3f0678cf` |
| Python reference census | 94 lines | `59e4c4f6fd1b54fbefe07c22d3665612319a11b35d51ae40a6077373c7ffd69a` |

No current runtime module imported a target. Before deletion, imports were
confined to target CLIs, target cores, and the twelve removed test files.
`src/api/routes/app_records.py::migration_apply` was a same-name route
function, not a module import. After deletion, the only Python text hit for a
removed namespace was the historical `scripts.migration` wording in the
retained package-marker comment; Task 5 owns final authority and old-path
census wording.

The structural RED proved that `scripts/migration/` still existed and that the
twelve gate-only test files collected exactly 172 nodes. Task 4 then removed:

- eleven migration CLIs;
- eight spent domain migration cores; and
- twelve complete migration-only test files.

The retained `src/news_normalized/score_import.py`,
`tests/test_news_score_import.py`, and
`scripts/scoring/import_news_scores_local.py` remained present.

Before deleting `retire_legacy_scheduler_iv.py`, the 2026-07-26 evidence
reclassified rollback as unsupported and removed its executable restore
runbook. Git history retains the retired executable. The ignored archive and
`RESTORE.txt` are lineage-only pending their separately approved deletion, and
the recorded manifest remains
`30c01ea8fd009a3d47c5ac96ffd4dd9b0282a1adef03faafb91c3dd50dd92fad`.
No ignored archive byte was inspected or modified.

The native collect-only result is:

| Fact | Observation |
|---|---|
| Collected / executed / exit | `4553 / 0 / 0` |
| Ordered collection | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |
| Reporter SHA-256 | `582628ca1246f18ce0ab8a1f1cb0f2e0e0583b7773a928a6d07d77b823f7af2b` |
| Transcript SHA-256 | `a4b3506566c68048a1a72388aa97c4c8da6eaf916d8380262b93c43aa9447562` |

Relative to the 4,730-node physical base, the result removes exactly 177 IDs
and adds none. The actual sorted removed set has SHA-256
`ece749907dc3ed03faf3dcf382727d1ab2a40b7067466834226926db32b1d3a7`.
The preconstructed `remove-final.nodes` artifact contains the same 177 IDs in
a non-sorted construction order; sorting it produces the same SHA and a
byte-identical set. The added stream is empty with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Retained verification produced:

```text
scoring contracts                 36 passed
IBKR normalized adapter           17 passed
legacy-IV retirement boundaries    4 passed
protected SQLite/DAL contracts    94 passed / 18 skipped
python -m compileall              exit 0
```

The pre-delete imports had left four ignored bytecode files under
`scripts/migration/__pycache__`. Their hash stream is
`f30a2d3756693e8adc6c8463b9e7369a9eeeb1232904db2ee809dd3161e5dfd2`;
their inode/size/time metadata stream is
`b9c66a11e60b1013189cace639416cfc688f9a022bf4ee653772d2ec6f92c643`.
The exact directory was moved unchanged to
`/tmp/scripts-retirement-tranche-a/quarantine/task4-migration/`, after which
the empty `scripts/migration/` directory was removed. Collection did not
recreate `comparison_results/` and executed no provider operation.

### 4.6 Task 5 pre-edit old-path census stop

Independent review reconstructed Task 4 at `ff7027d7` and returned GREEN.
Before any Task 5 owner edit, the reviewed raw old-path search at that commit
returned 714 rows across 100 tracked files. The census exposed three current
bindings omitted from the reviewed Task 5 file/staging lists:

- `scripts/__init__.py` still described the removed `scripts.migration`
  namespace as current;
- `tests/test_db_backend.py` still required the nonexistent
  `scripts/migrate_to_supabase.py`; and
- the current workbench spec still presented unimplemented root
  `scripts/profile_export.py`, `scripts/profile_import.py`, and
  `scripts/migrate_pg_to_local.py` command shapes without a supersession.

The census contract also searched generic `scripts/` while allowing only four
old-path classifications. That shape could not honestly classify the retained
Tranche A `scripts/scoring/` owner or lexical non-path matches. Treating either
as historical would hide current truth.

Execution therefore stopped before changing any Task 5 owner. The bounded
docs-only amendment adds the three omitted files and requires two explicit
side streams, `approved_tranche_a_survivor` and `lexical_non_path`, before the
four-class old-path candidate verdict. No Task 5 implementation owner, product
behavior, physical tree, provider, archive, secret, or production-data path
changed. Further implementation is unauthorized until that amendment receives
focused review. The candidate plan SHA-256 is
`8165044f255177b3732363e672f2c3c536875f9365f9f5aa7f03b82a4d4c1601`.

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

Independent review reconstructed Tasks 0 through 3 and returned GREEN at each
checkpoint. Task 4 is now review-ready. Its review must reconstruct:

1. the pre-delete consumer census and exact 31-path target manifest;
2. rollback reclassification before tool removal, with no archive access;
3. deletion of eleven migration CLIs, eight spent cores, and twelve whole
   migration-only test files;
4. exact `-177/+0` collection to `4553/69152591...`;
5. the retained `36`, `17`, and `4` contract groups plus protected
   `94 passed / 18 skipped`; and
6. physical `scripts/migration/` and `comparison_results/` absence.

Task 5 onward, merge, production interaction, provider requests, and any
Financial Datasets product-policy implementation remain blocked. The separate
product slice must own endpoint classification, the metered-request toggle,
billing-mode declaration, cache-first behavior, typed `402` handling,
fail-closed unknown endpoints, local usage disclosure, and Settings UI.

## 7. Task 5 Authority Reconciliation

The preceding block is the historical Task 4 checkpoint. Independent review
accepted the bounded census amendment at `9b1173ac` and authorized Task 5.
No product behavior, provider call, archive byte, production-data path, or
Tranche B score contract changed.

### 7.1 Current authority corrections

Eighteen reviewed owners now agree on the Tranche A end state:

- the top-level layout names only the transitional scoring owner and assigns
  final root removal to Tranche B;
- the package marker no longer claims a removed namespace;
- the PostgreSQL test prerequisite no longer requires a nonexistent importer;
- the workbench capability spec preserves portability and explicit import
  requirements while rejecting its three never-implemented command shapes as
  runnable instructions;
- historical provider, migration, density-analysis, OAuth, and RL references
  are explicitly labeled historical, upstream, or rejected rather than
  presented as local operators; and
- incomplete RL artifacts must be re-exported or retrained with required
  schema metadata; retained telemetry is TensorBoard plus `monitor.csv`.

### 7.2 Physical root and ignored residue

The final tracked path manifest is:

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `task5/tranche-a-scripts.paths` | 9 | `5557a08b373d6d3b2c4e7c7da83739af8e41d94c8bb1e669ace4819f779bf76d` |

It is byte-identical to the reviewed nine-path list. No retired subdirectory,
wrapper, tombstone, symlink, or compatibility module remains. The only
remaining ignored bytecode directories belong to the current root marker and
transitional scoring owner.

Two ignored bytecode files from the retired paid-provider probes were recorded
before exact-path quarantine:

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `task5/retired-pycache.metadata.tsv` | 2 | `daf025edb251d8f2e4ebc4cfe295997712f700148b393615e6fb4c122b370f51` |
| `task5/retired-pycache.sha256` | 2 | `940ae01f1ef23726e09267ab015ef0afd741b12688a6c7d635b201f304154554` |

Hash-only comparison after the move was byte-identical. Only then were the
now-empty retired directories removed.

### 7.3 Closed old-path census

The census used a generated fail-closed classifier, SHA-256
`67573a58cd3974917bc4d7eeaee93fd3bbfda12f6bfc31f97795de945cef341e`.
An unknown owner aborts classification. The exact artifacts are:

| Stream | Rows | SHA-256 |
|---|---:|---|
| Raw discovery superset | 726 | `018fb4ceb225c8222b918639bb1deca3f790ef14faac19e27a9adef724f24fa2` |
| Approved Tranche A survivor | 103 | `800ed7c283ab8802b750dca224710e080561924bd47a16d291b57a506d50fab4` |
| Lexical non-path | 2 | `c0a250f3ba87ae31aabf0479f1e803626a32cc3ca5519c7f219943ce847165c8` |
| Old-path candidates | 621 | `fa7b3fc5c65f84fa32010ec829ab1682670a50c92884768291a23f2cb62a0ce4` |
| Complete partition | 726 | `613c50cbc88bc8412119fe9428e2c97cca50516dc39382171e7155ead3063702` |

Candidate verdicts are closed:

```text
historical_record      558
non_root_owner          35
rejected_old_path       25
upstream_provenance      3
current_runnable         0
```

Removing the partition verdict column reproduces the raw superset
byte-for-byte. The raw stream has no duplicate row, and
`103 + 2 + 621 = 726`; no match was silently omitted.

### 7.4 Verification

Fresh structured collection produced:

```text
4553
69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca
```

Its reporter is byte-identical to Task 4 at
`582628ca1246f18ce0ab8a1f1cb0f2e0e0583b7773a928a6d07d77b823f7af2b`.
Focused verification produced:

```text
retained scoring contracts                 36 passed
RL inference/state and OAuth probe          25 passed / 8 skipped
protected SQLite/DAL contracts              94 passed / 18 skipped
Python compilation                          exit 0
RL shell syntax                             exit 0
```

The six score implementations, daily-update owner, local score importer, Tool
Catalog, and Desktop carry-over authority are byte-identical to the Task 0
base. The scoring README differs only by the reviewed Task 2 provenance-link
move.

The local scoring credential was checked only through metadata: present, mode
`0600`, and ignored by Git. Its contents, byte count, and digest were not read.

Task 5 is review-ready. Task 6 admission, merge, provider interaction,
production interaction, archive deletion, and Tranche B remain blocked.

## 8. Task 6 Canonical Admission

Independent review accepted Task 5 at `c68b8225` and authorized canonical
admission. Task 6 did not change product, test, provider, scheduler, archive,
secret, production-data, or Tranche B score-contract bytes.

### 8.1 Final collection and focused gates

A fresh structured collection produced a reporter byte-identical to the Task 5
report:

| Evidence | Rows or result | SHA-256 |
|---|---:|---|
| Fresh collect reporter | 4,553 collected / 0 executed | `582628ca1246f18ce0ab8a1f1cb0f2e0e0583b7773a928a6d07d77b823f7af2b` |
| Fresh ordered node stream | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |
| Precomputed `stage-final.nodes` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |
| Independent `/tmp/scripts-a-final.nodes` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |

All three node streams are byte-identical. Focused verification produced:

```text
retained scoring contracts                 36 passed
normalized IBKR score adapter              17 passed
protected SQLite/DAL contracts             94 passed / 18 skipped
Python compilation                          exit 0
diff check from 24202182                     exit 0
```

The implementation worktree remained clean after these gates.

### 8.2 Fresh native boundary

The first direct detached-worktree checkout stopped before materialization at
the repository's known linked-worktree `git-crypt` smudge boundary. It left no
registered worktree and no directory. The clean retry used the established
repository procedure: `--no-checkout`, only the existing key copied with mode
`0600` into linked Git metadata, then `git read-tree -mu HEAD`. No key content,
size, or digest was recorded.

The resulting detached worktree was exact `c68b8225`, with:

- no `config/.env` or `config/scoring_keys.txt`;
- an existing empty `data/`;
- no `src/data`, project database, historical dataset, provider credential,
  or production-root symlink;
- only the pinned
  `/mnt/md0/PycharmProjects/ArkScope/node_modules` toolchain symlink; and
- unchanged wrapper, reporter, wakeup probe, Node `v22.14.0`, and both
  lockfile identities.

The pre-run ordinary-status stream was empty. The pre-run ignored stream
contained only `node_modules`; the symlink inventory contained only that
pinned link.

### 8.3 Native final result

The sole native final command was:

```text
/tmp/eir002-green-baseline/run_native.sh scripts-a-final-full
```

It completed naturally in `248.36s`:

| Fact | Observation |
|---|---|
| Collected / seen | `4553 / 4553` |
| Passed / skipped / failed | `4481 / 72 / 0` |
| Errors / exit status | `0 / 0` |
| Collected and seen stream | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |
| Non-passing stream | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Reporter JSON | `2e94c3af06b7ada6596fb4068a5ca237d7c70edc11d806a370f91166a6f0c174` |
| Transcript | `41e1e5a57a67b544657ca82a0b5c92f1bf1aacfb976eaf88711001ff693cd4d9` |

Reporter `collected_node_ids` and `seen_node_ids` are byte-identical to each
other and to the precomputed final target. No partial transcript is used as
admission evidence.

### 8.4 Generated-artifact transaction

The native run created 575 repository-relative files:

```text
552 Python bytecode files
  4 pytest-cache files
 15 agent scratchpad files
  2 hermetic Financial Datasets cache fixtures
  1 native-host test log
  1 risk-free-rate cache
```

The exact transaction evidence is:

| Evidence | Rows | SHA-256 |
|---|---:|---|
| Exact relative paths | 575 | `6c451b866c54971448330685035f303752fec07d03086249ad5532b5a5804ffa` |
| Path/inode/size/mtime/mode/content metadata | 575 | `b649b6cbdf4c83ea70bf2347cf2adcb5a09cad5e08d344ce8ba1cc8eecb76037` |
| Post-run `data` / `src/data` file list | 19 | `7758868c4c9e5dd271aaafdf15c7f34fa71e635b0ccf04b54341f997bcccac6c` |
| Removed generated empty directories | 42 | `9fafa1f6273ed52e91c04aa3a35735eed5aa891413388abb384b694411ff34ef` |

Every file was moved by exact relative path to
`/tmp/scripts-retirement-tranche-a/quarantine/task6-final-full/`. Recomputed
quarantine metadata is byte-identical to the pre-move 575-row stream at
`b649b6cb...`; each original path was absent before generated empty
directories were removed.

Restoration is byte-identical:

| Boundary | Pre / restored SHA-256 |
|---|---|
| Ordinary status | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Ignored path list | `16d30e4462189fb14dd611bdb708c510630c576a1f35b9383e89a4352da36c97` |
| Symlink inventory | `3ddb95019fa533ceddd7502fe69d40ee6100388be7042c420270c0808bdf6452` |
| Empty `data/` file list | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Toolchain hashes | `2ed4198921e0424b1c9e223db5e000828f354f53f558744fe385cd62b8ae7a4d` |

`src/data` is absent again, the empty `data/` precondition and pinned
`node_modules` symlink remain, and no pre-existing file was modified or moved.
The main worktree's protected untracked predecessor remains
`79d4eac97d7692684d83f0a067f5987fe434bb76746b98af3e44f1c8ba4bf277`.

### 8.5 Review gate

Tranche A is implementation-review ready. Independent review must reproduce
the exact `+0/-177` collection, nine-path physical tree, retained gates,
closed old-path census, native reporter, and artifact transaction before
merge. Task 7, final root `scripts/` retirement, Tranche B, provider-policy
implementation, production interaction, archive deletion, and EIR-006 remain
unauthorized.

## 9. Task 7 Reviewed Merge And Merged Verification

Independent implementation review reconstructed the complete
`d89d433c..d6ef3b97` range and returned GREEN. The reviewed implementation
checkpoint is:

```text
SCRIPTS_TRANCHE_A_TIP=d6ef3b9726c00d1ffbbeb70ea11a74aa8ae24678
```

### 9.1 Protected-draft transaction and fast-forward

Before merge, main `master` and `origin/master` were both exact `24202182`.
The sole main-worktree difference was the protected untracked predecessor at
`docs/design/SCRIPTS_RETIREMENT_DECISION.md`.

Its pre-move facts were:

```text
SHA-256: 79d4eac97d7692684d83f0a067f5987fe434bb76746b98af3e44f1c8ba4bf277
inode:   127730122
size:    26932
mtime:   2026-07-27 22:41:00.742118107 +0800
mode:    0664
```

It moved by exact path to
`/tmp/scripts-retirement-tranche-a/quarantine/task7-main-untracked-draft/`.
The quarantined SHA, size, mtime, and mode are identical. The destination inode
is `97523871`; the inode change is recorded rather than hidden because the move
crossed from `/mnt` to `/tmp`. The committed authority became the sole current
file.

`git merge --ff-only codex/scripts-retirement-decision` then advanced
`master` linearly from `24202182` to exact `d6ef3b97`; no merge commit was
created.

### 9.2 Fresh exact-master verification

Merged verification used a new detached exact-`d6ef3b97` worktree with no
`.env`, local scoring secret, project database, historical dataset, provider
credential, or production-root symlink; `data/` was empty and the sole
toolchain link was the pinned root `node_modules`.

Fresh collection remained byte-identical:

| Evidence | Result | SHA-256 |
|---|---:|---|
| Merged collect reporter | 4,553 collected / 0 executed | `582628ca1246f18ce0ab8a1f1cb0f2e0e0583b7773a928a6d07d77b823f7af2b` |
| Merged ordered node stream | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |

Merged focused gates reproduced:

```text
retained scoring contracts                 36 passed
normalized IBKR score adapter              17 passed
protected SQLite/DAL contracts             94 passed / 18 skipped
Python compilation                          exit 0
diff check from 24202182                     exit 0
```

The pre-full gates generated 555 files: 551 Python bytecode files and four
pytest-cache files. Their exact path stream is `e12ad5c6...`; the 555-row
path/inode/size/mtime/mode/content stream and its quarantined reconstruction
are byte-identical at `589b10f9...`. The worktree was restored before native
admission.

### 9.3 Merged native admission

The new single-use native stage was:

```text
/tmp/eir002-green-baseline/run_native.sh scripts-a-merged-full
```

It completed naturally in `270.44s`:

| Fact | Observation |
|---|---|
| Collected / seen | `4553 / 4553` |
| Passed / skipped / failed | `4481 / 72 / 0` |
| Errors / exit status | `0 / 0` |
| Collected and seen stream | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` |
| Non-passing stream | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Reporter JSON | `2e94c3af06b7ada6596fb4068a5ca237d7c70edc11d806a370f91166a6f0c174` |
| Transcript | `442efc5bdfafbabfeb8e981cceb3a100163615bb1882086035881ba4d97e3f68` |

The merged reporter is byte-identical to the pre-merge reporter. No
`comparison_results/` path exists.

### 9.4 Merged artifact transaction

The native run generated the same closed 575-file shape as Task 6:

```text
552 Python bytecode files
  4 pytest-cache files
 15 agent scratchpad files
  2 hermetic Financial Datasets cache fixtures
  1 native-host test log
  1 risk-free-rate cache
```

| Evidence | Rows | SHA-256 |
|---|---:|---|
| Exact relative paths | 575 | `00761d6a9791b09f9084aa5d3637a26015687962776ea6cf6cb1119d42a5bc26` |
| Path/inode/size/mtime/mode/content metadata | 575 | `486e056c8fa4a167f656c94cf0edd79943ba9106edcf584048ebe29a8e94f481` |
| Quarantined metadata reconstruction | 575 | `486e056c8fa4a167f656c94cf0edd79943ba9106edcf584048ebe29a8e94f481` |
| Removed generated empty directories | 42 | `9fafa1f6273ed52e91c04aa3a35735eed5aa891413388abb384b694411ff34ef` |

All 575 files moved by exact relative path to
`/tmp/scripts-retirement-tranche-a/quarantine/task7-merged-full/`.
Pre/restored ordinary status, ignored paths, symlink inventory, empty `data/`,
absent `src/data`, and toolchain hashes are byte-identical:

```text
ordinary status: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
ignored paths:   16d30e4462189fb14dd611bdb708c510630c576a1f35b9383e89a4352da36c97
symlinks:        3ddb95019fa533ceddd7502fe69d40ee6100388be7042c420270c0808bdf6452
data files:      e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
toolchain:       2ed4198921e0424b1c9e223db5e000828f354f53f558744fe385cd62b8ae7a4d
```

No pre-existing file was modified or moved.

### 9.5 Tranche checkpoint

Tranche A is complete at `SCRIPTS_TRANCHE_A_TIP`. Tranche B has not started.
Root `scripts/` intentionally remains only for `scripts/scoring/` and the
package marker. No production score data, local scoring secret, ignored archive
byte, EIR-006 owner, or provider-policy implementation changed. The docs-only
closeout commit requires focused review of this section and the matching
authority/priority-map changes.
