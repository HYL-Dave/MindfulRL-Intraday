# EIR-006 Exact Deletion Manifest

> Status: **V3 READ-ONLY MANIFEST COMPLETE; TASK 9 NOT APPROVED**
>
> Authority ID:
> `9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7`

This packet was built from Task 8 of the reviewed EIR-006 plan. It identifies
the obsolete file and SQLite rows selected by the user's product ruling. It
does not authorize or perform deletion. Task 9 still requires independent
review of this packet and a separate user approval naming the authority ID,
packet identity, and controller identity.

## Decision Facts

- Product cutover authority:
  `ce88f72d9f9d710903533505371789d18cce953e`.
- Task 8 base:
  `25f061b7781cdc9f738a4858aa331dd10a3ef9d2`.
- Exact file authority: 225 15-minute CSVs, 75 hourly CSVs, and one collection
  summary; 301 files total.
- Exact SQLite authority: 19 `metrics_*_annual_y2` cache keys, 130 legacy
  `fundamentals.id` values, and one `market_sync_meta.domain='fundamentals'`
  row; 150 equality parameters total.
- Current cache protection: all 18 SEC v1 rows and nine other retained cache
  rows are outside the delete manifest.
- Current-reader/writer census: zero old-data readers, zero old-data writers,
  zero unknown classifications.
- The current sidecar remains the owner of present SQLite collection. It is not
  an owner of the retired file or legacy-fundamentals paths.

## CSV Admission Result

The pinned implementation and alias input reproduced byte-identically in two
fresh output roots.

| Measure | Raw diagnostic | Canonical deletion authority |
|---|---:|---:|
| physical rows | 2,547,747 | 2,547,747 |
| unique keys | 2,314,293 | 2,298,763 |
| duplicate rows | 233,454 | 248,984 |
| conflicting duplicate keys | 58 | 176 |
| SQLite value differences | 161 | 43 |
| keys absent from SQLite | diagnostic only | 0 |

The 43 canonical differences are 23 volume-only and 20 including OHLC. No
CSV value is imported. `LC -> HAPN` accounts for 15,530 overlapping keys and
118 alias conflicts. The reviewed SQLite authority remains unchanged.

## Exact Authorities

| Artifact | Role | SHA-256 |
|---|---|---|
| `authority-input.json` | approval identity input | `9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7` |
| `legacy-price-files.tsv` | exact file paths and identities | `842c3e08ff8ed9cb11c92033cf67ad5950d357cb8cd1e0662b74683ba554b0fc` |
| `ticker-aliases.tsv` | exact canonicalization input | `0a8fbbf845b73bab1740d04ffb77ab1e935884f417c2bece20395187f83d9220` |
| `old-cache-rows.tsv` | exact old cache keys and payload identities | `a4a8d829eb08553a1223f5240de260955fc48a564f8232b943206e0bf88b39bd` |
| `legacy-fundamentals-rows.tsv` | exact legacy IDs and payload identities | `6b845506f9fce54ac4dba78ebd96bacc20113a7aefef651b877f62892418c219` |
| `legacy-sync-rows.tsv` | exact legacy sync key and metadata | `5b3736ba19e66b2e427b149b143771fb5625eab426e8af7a6317c29461cd15ff` |
| `cache-classification.tsv` | old/current/other cache separation | `62e56fc02f5b8a15aaea9f360eee8bd875e10d6a10c7a017c8d568652beef323` |
| `consumer-census.tsv` | fresh final consumer/writer census | `a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64` |
| `behavior-propagation.tsv` | current detailed-financials callers | `613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba` |

`raw-db-differences.tsv` and `canonical-db-differences.tsv` contain keys and
typed difference classes, not price payloads. The DB manifests retain payload
byte counts and hashes, not payload contents.

## Pinned Producers

| Source | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `task8_price_manifest.py` | 669 | 26,509 | `e4acb819f6a32c05d7d756b1a9e106bba105e1c22d6cd273513d6bf27df2e759` |
| `task8_db_row_manifest.py` | 304 | 11,030 | `27c9dee61a6f7f04ed8fc4226f15c6892478bd63cef94f5df1fcfb2b8ffabc29` |
| `task8_consumer_census.py` | 191 | 7,999 | `eefa4091683d7fd3ff8d91e40b77cc0a2498b56bd576cbf487f355b514679c2f` |
| `destructive_controller.py` | 1,138 | 42,740 | `891edbe1fe0c8005f609fee2ed97403180f3498da53668da6175645c97214d37` |
| `controller_probe.py` | 189 | 7,688 | `e200d63b951fa7b44e8b9e49a3b0b81207a025923c69d827b90b7d8afe2ee981` |

The controller probe used a scratch SQLite fixture populated through a
production read-only connection. It proved 19/130/1 exact deletes, 19/130/1
exact restores, same-filesystem file move/restore, and the rollback snapshot
identity. It also proves that every reviewed read-only connection has no lsof
record after its context exits. Its file fixture now proves the phase-specific
holder sequence: source tree before movement, quarantine tree after movement,
and both existing trees after rollback. An absent post-move source tree is a
typed refusal rather than an `lsof` invocation. It did not invoke the
production mutation path.

A separate holder-probe check exercised the controller's bounded `lsof`
contract against an unheld file, a file held by the probe process, and an
invalid directory. The outcomes were respectively empty, a structured holder
record, and fail-closed refusal. `lsof` is capped at 15 seconds; stderr,
ambiguous exit/output combinations, and every exit other than the documented
holder/no-holder pair are refusals.

The final census producer was rerun against this completed Task 8 worktree,
not only the pre-packet product tree. It requires the exact ten packet files
that contain old-authority search terms, classifies them as self-authority, and
fails if that set changes. The resulting product census remains byte-identical
at 128 rows, with zero current old-data consumers/writers and zero unknowns.
The unchanged two-node owner file now lists those same ten exact paths and is
pinned by the controller as
`de6e192b7e3a233b26d9a43c5dd8608e0ce26cfad3ef79f3d73e882a3f79fb9c`;
every other product/test path remains locked to the product cutover tip.

## Task 9 Mechanics

The exact same-filesystem quarantine root is:

```text
/mnt/md0/PycharmProjects/.arkscope-eir006-quarantine/9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7
```

The rollback snapshot will be written before any move or delete to:

```text
db/legacy-rows.jsonl
```

It is canonical JSON Lines containing the complete 150 rows required for
rollback. Its read-only preview reproduced twice as 875,857 bytes with SHA-256
`1e3578344dfcac0e445900358265c6606150007a496a71284d87e5ae5821697c`.
Those payload bytes are not tracked in this packet.

The only SQL mutations are three parameterized equality deletes:

```sql
DELETE FROM financial_cache WHERE cache_key IN (19 exact parameters);
DELETE FROM fundamentals WHERE id IN (130 exact parameters);
DELETE FROM market_sync_meta WHERE domain IN (1 exact parameter);
```

No `LIKE`, wildcard, path glob, or inferred key is execution authority.
Discovery is accepted only when its exact set equals the reviewed manifests.

## Operational Owner

`operational-state.json` records the dated state. At observation time the
desktop owner was PID 2887595, Electron PID 2887650, and sidecar/scheduler PID
2887713. The price schedule was enabled at 720 minutes. The exact stop owner is
the desktop dev launcher; its reviewed SIGTERM handler terminates Electron,
the sidecar, and the Vite process group.

If any PID, scheduler setting, DB device/inode, manifest row, file identity,
alias input, product byte, or retained cache row changes before Task 9, this
approval packet expires and Task 8 must be rebuilt. Unrelated current
price/news growth is not a reason to copy legacy values, but mutation may not
start until the sidecar is quiesced and all DB/file holders are absent.

## Superseded Task 9 Attempts

The user approved v1 authority
`6096b988428a94d053baddd18493eb29077bc627d725a95fd53f75c4755b0dce`
on 2026-08-07. The first execution moved all 301 files, then failed before any
database delete because the controller's own read-only SQLite handle remained
open after a `with sqlite3.Connection` block. The holder gate rejected the
controller itself with `exit=1, records=5`.

The automatic pre-commit recovery moved all 301 files back with
`restore_error=null`. Reviewed rollback then verified every file and all 150
rows in place, restored `0/0/0` rows, and the exact temporary quarantine was
destroyed. Desktop, sidecar, and the seven saved scheduler settings were restored.
No forward deletion from that authority remains.

V2 changes the read-only helper to a real context manager that closes in
`finally`; the probe first reproduced the old self-holder RED, then proved zero
self-holder records after both fixture reads. Schema version 2 and the new Task
8 base deliberately produce a new authority ID, so the superseded approval
cannot authorize another execution.

The user separately approved V2 authority
`4b1d9083ed054387cd00ae253ab055641fc18e55a7a4e718534fb25a23cf413e`
on 2026-08-07. V2 preflight passed. Execution moved all 301 files and removed
the empty source directories, then its second holder check called `lsof +D`
on the now-absent `data/prices` path. `lsof` returned a usage error and the
controller refused before opening the write transaction. Automatic recovery
moved all 301 files back with `restore_error=null`; all 19/130/1 target rows
were still present. Reviewed rollback restored `0/0/0` rows and emitted
failure/rollback receipt SHAs `923c2a7c...` / `e1a43aea...`. The snapshot kept
its exact `1e357834...` identity until verification, after which the whole V2
temporary root was destroyed. A final native V2 preflight passed, and desktop,
sidecar, database identities, health, and all seven saved schedule settings
were restored. No V2 forward deletion remains.

V3 requires every quiescence call to name one or more existing price trees.
Preflight names the source tree; post-move execution and pre-restart verify name
the quarantine tree; rollback checks every existing source/quarantine tree and
refuses symlinks, duplicates, missing roots, or holders. The V2 source was
replayed against a scratch post-move tree and reproduced the exact `lsof` usage
RED. The V3 probe then passed source, quarantine, and rollback phases. Schema
version 3 and Task 8 base `25f061b7...` create another authority; neither prior
approval may be reused.

## Review Boundary

Before separate approval, the reviewer must reconstruct every tracked SHA,
rerun all three read-only producers, reproduce the V2 post-move RED, rerun
`controller_probe.py`, inspect all four phase-specific quiescence call sites,
and verify the exact stop/start and rollback procedure in the plan amendment.
A partial reconstruction or a matching row count without matching keys is not
approval evidence.
