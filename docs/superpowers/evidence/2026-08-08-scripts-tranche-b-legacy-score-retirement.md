# Scripts Tranche B Legacy Score Retirement Evidence

> **Status:** FULL PLAN AND ONE-TIME REBASE AMENDMENT GREEN; TASK 0 STOPPED ON
> A BOUNDED NO-TAIL OWNER AMENDMENT; IMPLEMENTATION NOT STARTED
> **Date:** 2026-08-08
> **Rebase amendment:** 2026-08-09
> **Current plan base:** `814ef2edd1b6aa66499145e1a9109d05f5fb0d89`
> **Product ruling:** PD 1-PD 8 approved on 2026-08-08
> **Destructive authority:** none; score rows and `scoring_keys.txt` remain blocked

## 1. Approval and scope

The user approved the reviewed `04dd9a67` section 8 bundle and explicitly kept
two later operations outside that approval:

```text
491,808 physical score rows: later independent exact approval required
config/scoring_keys.txt: later independent exact approval required
```

This packet currently records only plan construction. No product/test/runtime
source, production data, local secret, provider, scheduler, or model request has
changed on this branch.

## 2. Plan-gate grounding

### 2.1 Git boundary

```text
reviewed old base: 7257699171a81294b74ff8cde61fb90bb065a2b4
current base:      814ef2edd1b6aa66499145e1a9109d05f5fb0d89
inventory:         098dff56 -> 8c952773
decision:          04dd9a67 -> f34463df
plan:              d1c5b5a5 -> bda76296
no-tail tip:       52354806 -> f9958efb
branch:      codex/scripts-tranche-b-inventory
```

The full plan at `52354806` received independent GREEN review. After all three
predecessor lines merged, the four docs-only commits were rebased once onto
`814ef2ed`; the inventory, decision, plan, and evidence bytes at `f9958efb`
were identical to their reviewed pre-rebase versions. The priority map retained
the same Tranche B decision entry and gained only intervening reviewed history.
Before the rebase and before this amendment, both worktrees were clean.

Original plan source identity committed at `d1c5b5a5` and superseded by the
post-approval no-tail amendment:

```text
674 lines / 36,175 bytes
SHA-256: 09307ddb241e2d0ac19ea61b467e6d07a34eb825663df67381eabdb7bc74562d
Git blob: cda25357d145ca675774b20c852f958ce8455fc8
```

Replacement plan source identity presented for independent review:

```text
715 lines / 39,324 bytes
SHA-256: 70e48d1184f0581033ccbf566f7f33718fac6ac934ae7c2b41d963fcb665f704
Git blob: 383ff74d835828719d6d234e3a0b00336e7d92fa
```

The byte-identical rebased plan at `f9958efb` had that same identity. This
one-time absolute-identity amendment changes it to:

```text
755 lines / 41,529 bytes
SHA-256: 12082ce9b049b0b488cea298ee313b85ba7a884c6f462d5d9bc43a224537c802
Git blob: e1eff4ad70ec44add348b277a67c5f94668241c0
```

### 2.2 Post-approval no-tail amendment

The user clarified that retirement must remove the obsolete architecture, not
preserve it merely because PD 5-PD 6 retain two useful behaviors. Grounding
confirmed that `src/tools/signal_tools.py` is the runtime importer of the old
`src/signals/` package and that the package itself mixes the approved raw
volume/event primitives with scorer prompts, sentiment anomaly, numeric event
impact, sector aggregation, and composite synthesis.

The plan now deletes that package and tool module completely, moves only the
approved pure behavior to `src/news_analytics.py`, and exposes it through
`src/tools/news_event_tools.py`. The old event-chain `TestSignalTools` node
moves from evolved to retired; two honestly named news-event tool nodes replace
the formerly planned one-node volume addition. No compatibility import,
re-export, route, or old test namespace is allowed.

This does not cancel the future Signals roadmap. The Workbench Product Spec and
Priority Map explicitly retain a new evidence-gated Signals product with a
written hypothesis, source-labeled inputs, OOS validation, and kill criteria.
The frozen rows and local scoring secret are only temporarily unchanged because
their destructive disposition needs separate authority. After runtime
disconnection, exact deletion is the default unless a read-only packet proves a
concrete research use worth detailed user review; runtime retention is not an
outcome.

### 2.3 Rebased canonical base inputs

The product tree remains docs-only relative to merged `master`. The
deterministic streams re-collected after OAuth, provider hygiene, and Settings
navigation are:

| Stream | Count | SHA-256 |
|---|---:|---|
| backend full | 4,527 | `4eeb117804ad874c83ffe4c04fd25ecd4de4f460801bfbf95d15c1406f32455d` |
| frontend full | 1,123 | `9262d7b15a926d7eeb60952e4c351c6c9b944772904fdb82438c62a2a51f6c1c` |
| backend focused | 555 | `ea5d897ca3597ef4edca7583db0b363360ceba9e362e516422f901ff8af004dd` |
| frontend focused | 27 | `c77d25d5bf7c868899d555f099fc245b13d35d4052a6a8242ac3f5c1300fb584` |

Task 0 must reproduce these values; this amendment does not substitute set
construction for that executable gate. Backend collect-only at the rebased tip
reported `4527 collected / 0 seen / exit 0`; frontend used the pinned decoded
JSON normalizer.

### 2.4 Node-ledger construction

The plan author independently enumerated whole retired test files, mixed-file
retired IDs, and all new IDs. Streams were combined as sets, sorted by UTF-8
byte order, written with one trailing newline, and checked for uniqueness and
set membership.

| Stream | Count | SHA-256 |
|---|---:|---|
| retired | 138 | `b48b161d573afb37496763c0afe388c2421f06e35eb5cd7de959ba5778c05254` |
| backend additions | 18 | `88ac9e5652c9df79eb42284d6a9c42a2f0f4a60b967badae37524fa127499520` |
| backend RED | 4,545 | `e1fa3f7d54d671c984e9800e38850ccb802f06f83d78aa2114b749bb7414f9da` |
| backend final | 4,407 | `d71b9825e727fd7ccac43b79ebb904144a48a9acc66b75ccae002471822ac8bc` |
| focused RED | 573 | `5e0a5538c4106ca9b9cf0d701ab719d62c3a4056d1e101864ddb09b6beb9fb75` |
| focused final | 435 | `2e5fcb6c22d6a1657e609542138830f2d5fd367a0e353ab30efdfbb8851a7c6a` |

Retirement composition is exact `102 whole-file + 36 mixed-file = 138`.
Addition composition is exact `8 boundary + 1 API + 3 monitor + 1 identity + 1
morning brief + 2 news-event tools + 2 truthful bridge-registry IDs = 18`.

Plan self-review found that two retained bridge tests encoded the old count in
their node names (`tools_count_31`). Changing only their assertions would leave
false test names. The ledger therefore retires those two IDs and adds
`tools_match_registry` replacements. This changes stream identities but not the
final count or native pass/skip arithmetic.

The 25 explicitly retained/evolved base IDs are all present once; their sorted
stream SHA is
`2f0e0dd31390f975eb2b4f20244525a0bf09b0bc112f39f0f4cebfe2db76aa08`.

### 2.5 Projected audit streams

| Projection | Count | SHA-256 |
|---|---:|---|
| storage/writer/root | 4,483 | `a00d996aa45fda19e0a9a473fb2767310b98daf5d2ab8ad1c72754c3b2a080f1` |
| raw DTO/backend | 4,444 | `e71c3b0cfe25931558fe4d01b81fd7a0c653b002c7aa506ace66581b4b0ff458` |
| raw user behavior | 4,444 | `d2dd1b144e3923a60672c23e2f8f8ac9f42c1a42445235b47e83a90d1a51a99e` |
| volume/event/composite | 4,404 | `ebc5563c5d408bf83dd14f643d6a5ffd082f1b02d5159485fdb1e0b2ef00c7a9` |
| final | 4,407 | `d71b9825e727fd7ccac43b79ebb904144a48a9acc66b75ccae002471822ac8bc` |

These are mathematical node-accounting projections. They have not been called
runtime GREEN and do not relax the atomic final gate.

### 2.6 Native target arithmetic

Two retired nodes were run under the blank canonical environment and both were
observed skipped:

```text
tests/test_db_backend.py::TestNewsDB::test_query_news_scored_only
  SKIPPED: DATABASE_URL not configured in config/.env

tests/test_signal_factors_p1.py::TestGetSignalFactorsShape::test_contributions_sum_to_composite_score
  SKIPPED: synthesizer emitted no factors for this fixture

result: 2 skipped in 0.78s
```

No repository-relative artifact remained; the temporary empty `data/` marker
was removed and ordinary status returned clean. Therefore:

```text
passed:  4488 - 136 retired passing + 18 new passing = 4370
skipped:   39 -   2 retired skipped                  =   37
total:   4370 + 37                                  = 4407
```

Task 4 must prove this result; arithmetic is not admission evidence.

### 2.7 Frontend construction

The sole new decoded Vitest node is:

```text
src/legacyScoreRetirement.test.ts<TAB>legacy score retirement boundary > removes score fields from current frontend DTOs and fixtures
```

| Stream | Files / nodes | SHA-256 |
|---|---:|---|
| base | `98 / 1123` | `9262d7b15a926d7eeb60952e4c351c6c9b944772904fdb82438c62a2a51f6c1c` |
| target | `99 / 1124` | `da69a2942c03e4794e3384e6125936f9f25c1fafbad7d006b67025f8fd97bc39` |
| focused base | `3 / 27` | `c77d25d5bf7c868899d555f099fc245b13d35d4052a6a8242ac3f5c1300fb584` |
| focused target | `4 / 28` | `b11cc27b90c570b20aeb728c48d39497e7eff555976f0c00a42d6129b26cf1cd` |

The target was constructed from decoded runtime names, not escaped JSON text.
Task 0 must re-run the pinned normalizer.

### 2.8 One-time absolute-identity re-derivation

The re-derivation root is `/tmp/scripts-tranche-b-rebase-f9958efb`. Its
66-entry `SHA256SUMS` is
`e61626c9d55ffab5bc51d887e0c9c35bb82d33b141a2322f4c312205db49b102`.
It contains the raw backend reporter/transcript, raw frontend Vitest JSON,
frozen ledger inputs, every derived stream/table, the old/current delta streams,
the rebased commit chain, and the old-versus-new range-diff. Its
backend identity table is `6523058c898983dc5507281c02ee22156cc606469ee4408245ce54f014a78695`;
its phase validation table is
`d9f4cc12d8536652979613da42cb3d5bb930ead767046495c203516954974c9b`;
its frontend identity table is
`f5ba905e21cc987b1053e90b27e3d10172e4abd58d97f39a3fce323e46df51f1`.
The backend collect-only reporter is
`b2f1539d751614942b45b15d5366383c3d7a1880adcfe3fcedb7b49d1406cc46`
and records `4527 collected / 0 seen / 0 non-passing / exit 0`.
The two current-base retired-skip owners were rerun through the same native
wrapper and returned `2 skipped / exit 0`; their structured report is
`50a99f651a839d67d65e13d6915ab169d27fe56f0078566835e3d14e8ecad6ca`.

The intervening backend delta from the reviewed pre-handoff base is exact
`-81/+27`; the removed and added streams are respectively
`80a52a1e2844a72a45b02940f143d08d24a578ae79d7c48d6cbd3a028e17a2dc`
and `abcf25ac4d850c9d64fa55932415c268740c00eacb71905cdddc9ac202786f5f`.
Neither stream intersects the corresponding frozen Tranche B retirement or
addition ledger. The frontend delta is exact `-1/+47`; the current full stream
was decoded from Vitest JSON with pinned normalizer `955dca59...`, not text
extraction. All 138 retired backend IDs are present, all 18 additions are
absent, and the frontend addition is absent.

Each phase independently proves every removal present and every addition absent
before mutation:

| Phase | Removal expected/present | Addition expected/pre-existing |
|---|---:|---:|
| 1 | `46/46` | `2/0` |
| 2 | `43/43` | `4/0` |
| 3 | `1/1` | `1/0` |
| 4 | `45/45` | `5/0` |
| 5 | `3/3` | `6/0` |

Stage 5 is byte-identical to the independently constructed final target.
Backend focused identities remain exactly `555 -> 573 -> 435`; frontend
focused identities remain exactly `3/27 -> 4/28`. This is direct evidence that
the handoff changed only absolute full/projection identities, not Tranche B's
relative owner ledger.

The following reviewed pre-handoff identities are retained only as dated
history and are prohibited for current admission:

| Surface | Superseded base | Superseded target |
|---|---|---|
| backend | `4581/6e4994bb...` | `4461/c7cb78b2...` |
| frontend | `97 files / 1077/3f5e9f5b...` | `98 files / 1078/de1e0c3f...` |
| native | `4509 passed / 72 skipped` | `4391 passed / 70 skipped` |

No product/test byte, production row, provider, scheduler, secret, or model
request changed during rebase or re-derivation.
The product-decision amendment changes only its status header and section 9
handoff sequence to record completed predecessor lines; PD 1-PD 8 and their
approval/non-authorization semantics are unchanged.

### 2.9 Protected lineage

The current tracked `training/` tree has 53 paths. Its sorted
`path<TAB>git-blob` stream is:

```text
2284c8989f6104979a11a5111de987f5d6f2974e3d2f74f0cf47ed5b4854e14a
```

Provider-native sentiment and investor-profile risk were found in distinct
owners. They are protected and receive a dedicated new regression node rather
than being removed through token matching.

Self-review found and removed a plan contradiction: the no-tail ruling requires
current `evidence_packet.py` copy to stop naming the deleted `signal_tools`
module, while the original plan protected the entire file byte-for-byte. The
amended contract excludes that file from the byte-identical manifest, permits
only the exact docstring/`_EXCLUSION_NOTE` retirement-copy delta, records
pre/post blobs, and keeps projection/gather logic plus the negative-contract
tests unchanged. This is a bounded current-copy repair, not permission to alter
objective evidence behavior.

### 2.10 Task 0 execution and no-tail stop

Independent review cleared rebase amendment `5be77be2`, authorizing Task 0. The
artifact root is `/tmp/scripts-tranche-b-task0-5be77be2`; its current 637-entry
partial manifest is
`3be60a14192520da2977e4a1604f1c4c251052ff94e3c98719692e99c587822a`.
It is explicitly partial because the owned/protected path manifests remain
blocked on the finding below.

Re-grounding before the stop produced:

| Witness | Result |
|---|---|
| backend full collection | `4527 / 4eeb117804ad874c83ffe4c04fd25ecd4de4f460801bfbf95d15c1406f32455d` |
| backend focused collection | `555 / ea5d897ca3597ef4edca7583db0b363360ceba9e362e516422f901ff8af004dd` |
| frontend full decoded collection | `98 files / 1123 / 9262d7b15a926d7eeb60952e4c351c6c9b944772904fdb82438c62a2a51f6c1c` |
| frontend focused decoded collection | `3 files / 27 / c77d25d5bf7c868899d555f099fc245b13d35d4052a6a8242ac3f5c1300fb584` |
| focused backend runtime | `536 passed / 19 skipped / 0 failed`; report `84bb7106505ccee1d437e0b9e8dfa1c45117b1d628e41f0a7d349e760ed08133` |
| focused frontend runtime | `27 passed`; transcript `df658d97ae9a651c4b4e39e45632b0ef5323e769edcd2d15045da1d737c1d093` |
| native canonical base | `4527 collected = 4527 seen / 4488 passed / 39 skipped / 0 failed`; report `1becf27f6755f4ec7d4b5ffabdf99d2d0db26665ce18dbbb2a1042f75da7c143` |
| native generated artifacts | 568 files plus two empty directories; file manifest `90f7a97e85de76545e9d7453d77c219845371c383df8548d608a566f0b46994a`; pre/final status and data projections equal |

The retired/addition streams, all five phase projections, focused targets, and
frontend target were independently reconstructed from named owners and matched
every section 2 count and full SHA. Four pre-rebase authority files were also
confirmed as identical Git blobs between `52354806` and `f9958efb`.

Two rejected operator/harness attempts are retained rather than rewritten:

- the first focused backend stage omitted the canonical empty `data/` marker and
  returned 10 FileBackend setup-related non-passing nodes; a fresh stage with the
  required marker passed;
- the first frontend focused command ran from repository root, bypassing the app
  Vitest config and its i18n setup; the same three files from the correct
  `apps/arkscope-web` workspace passed 27/27.

Production observation was read-only SQLite URI mode plus
`PRAGMA query_only=ON`. `news_article_scores` still has exactly 491,808 rows and
140,152 distinct article IDs; article timestamps span
`2022-01-01T05:14:26+0000..2026-04-27T00:17:00+0000`, and score operation times
span `2026-06-07T06:03:13.042078+00:00..2026-06-07T06:06:26.870916+00:00`.
The 3,477,532,672-byte DB had inode `127284871` and SHA-256
`123b7f8f00b9d643a0c98244d2e8521e2ef164d0c8439e887d0493d47064abab`;
its inode/size/mtime remained stable during the query. `config/scoring_keys.txt`
was not opened, read, hashed,
printed, copied, or sized; only existence, mode `0600`, inode, and mtime were
recorded.

The subsequent exact owner census stopped on
`src/news_normalized/score_migration.py`. Git proves its only tracked importer is
the already-retired five-node `tests/test_news_score_migration.py`, while the
module imports the planned-deleted `.scores` owner. The reviewed plan omitted
the source module from every disposition. Under the no-tail ruling and stop
condition 15, Task 0 stopped before claiming a closed owned/protected manifest.
This amendment assigns the module to phase-1 deletion without changing any node
ledger. Product edits and Task 1 remain blocked pending focused review.

## 3. Locked implementation sequence

| Task | Status | Gate |
|---|---|---|
| Task 0 re-ground | stopped at owned/protected census | focused no-tail owner amendment review |
| Task 1 RED | blocked | Task 0 GREEN |
| Task 2 atomic cutover | blocked | plan + Task 0 GREEN |
| Task 3 mutation/focused gates | blocked | product target exact |
| Task 4 native admission | blocked | all mutations restored |
| Task 5 independent implementation review/merge | blocked | native GREEN |
| Task 6 merged verification/closeout | blocked | fast-forward merge |
| physical score-row disposition | separately blocked | merged rollout + use analysis + new review/user approval |
| scoring-secret disposition | separately blocked | exact consumer metadata + new review/user approval |

## 4. Plan-gate verification checklist

- [x] PD 1-PD 8 approval recorded without broadening destructive authority.
- [x] Product and data/secret scopes separated.
- [x] Backend base/RED/final and focused streams precomputed.
- [x] Whole-file and mixed-file retirement IDs enumerated exactly.
- [x] Eighteen independent new backend nodes named.
- [x] Legacy Signals namespace removal and honest news/event owners locked.
- [x] Frontend decoded target and focused identities precomputed.
- [x] Two canonical retired skips directly reproduced.
- [x] Five intermediate projections labeled accounting-only.
- [x] Ten product mutations mapped to owning nodes.
- [x] Provider-native/training boundaries byte-protected; EvidencePacket copy
  delta bounded while its negative behavior remains protected.
- [x] Native wakeup/reporter/wrapper/toolchain boundary pinned.
- [x] Independent full-plan review GREEN at `52354806`.
- [x] One-time rebase completed with unchanged relative ledgers and exact new
  full/projection identities.
- [x] Focused rebase-amendment review GREEN at `5be77be2`.
- [ ] Task 0 authorized and executed.

## 5. Honesty boundary

This packet proves that the plan is grounded and internally accounted. It does
not prove that the implementation exists, that the final suite passes, or that
production score rows are deletable. Those claims require their later gates.
