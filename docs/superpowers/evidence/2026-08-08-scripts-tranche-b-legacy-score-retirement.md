# Scripts Tranche B Legacy Score Retirement Evidence

> **Status:** TASKS 0-4 COMPLETE; ATOMIC TIP `8ebf7fae`; NATIVE ADMISSION
> GREEN; INDEPENDENT IMPLEMENTATION REVIEW PENDING; NOT MERGED
> **Date:** 2026-08-08
> **Rebase amendment:** 2026-08-09
> **Current plan base:** `814ef2edd1b6aa66499145e1a9109d05f5fb0d89`
> **Product ruling:** PD 1-PD 8 approved on 2026-08-08
> **Training ruling:** direct Git retirement approved on 2026-08-09; no archive
> branch, tag, copy, disabled scaffold, or compatibility tail
> **Destructive authority:** none; score rows and `scoring_keys.txt` remain blocked

## 1. Approval and scope

The user approved the reviewed `04dd9a67` section 8 bundle and explicitly kept
two later operations outside that approval:

```text
491,808 physical score rows: later independent exact approval required
config/scoring_keys.txt: later independent exact approval required
```

Sections 2.1-2.13 record plan construction and Task 0 before product work.
Section 2.14 records the later protected-gate stop. Section 6 records the
reviewed amendment, atomic product commit, mutation/focused gates, and native
admission. The cutover is committed only on the isolated implementation branch;
it is not merged or pushed. No production data, local secret, provider,
scheduler, or model request changed.

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

The byte-identical rebased plan at `f9958efb` had that same identity. The
one-time absolute-identity amendment at `5be77be2` changed it to:

```text
755 lines / 41,529 bytes
SHA-256: 12082ce9b049b0b488cea298ee313b85ba7a884c6f462d5d9bc43a224537c802
Git blob: e1eff4ad70ec44add348b277a67c5f94668241c0
```

The independently reviewed training-retirement amendment at `92f51f7e` then
produced `859 lines / 47,809 bytes /
d21ab5325656049c4c8de0151f637db031fe025c4a76f767487bdd95ad28ca3d`
(Git blob `f509597e8add91743eb2608dc037720996afb813`). The Git-record commit
at `d991f64b` produced `881 lines / 48,949 bytes /
415f6f83dd0ef045eeed6ed79364f3244b7266b21d9d8a8d79add7c9664e5533`
(Git blob `e9a565faacf5a51dfd7e2d8821a59a559f848f46`).

The current Phase D/PostgreSQL owner-closure plan presented for focused review
is:

```text
983 lines / 55,944 bytes
SHA-256: dedd092297b804a4db5212364a219478c0b45c27023dd6c5b3778eb57509288e
Git blob: 38eab2d9f5ea3ddebeb04253eda58f959dafd87d
```

Its amended product-decision authority is `457 lines / 22,825 bytes /
b490d46b250757c674af89f4f04d8e1030e35de1b442ec4847ee6f89a9b16e01`
(Git blob `7f01edc5cffb06aa4e5b796150fa080c9495c1f3`).

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
| retired | 263 | `93459510fc09e961b0d726527d953ed6fdfd07c584d598ee1de9a60851ca3eda` |
| backend additions | 18 | `88ac9e5652c9df79eb42284d6a9c42a2f0f4a60b967badae37524fa127499520` |
| backend RED | 4,545 | `e1fa3f7d54d671c984e9800e38850ccb802f06f83d78aa2114b749bb7414f9da` |
| backend final | 4,282 | `281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |
| focused RED | 573 | `5e0a5538c4106ca9b9cf0d701ab719d62c3a4056d1e101864ddb09b6beb9fb75` |
| focused final | 421 | `385d0ac7a142ba1cb488a1dccd3d1a7ae8e2065585b59130f4b3bf75120a2739` |

Retirement composition is exact `223 whole-file + 40 mixed-file = 263`.
The added 109-node training-only stream is
`db3cad74da2ec956e252096948d80631297e9d4e8c731fb6706da7b2976941b2`
and has zero intersection with the original 138-node score/signal stream and
the 16-node Phase D stream
`d190b082db0ce6a231b1a93e7de7fef682c0ef3ce48d7c96fd4522e51fa7fee7`.
Addition composition is exact `8 boundary + 1 API + 3 monitor + 1 identity + 1
morning brief + 2 news-event tools + 2 truthful bridge-registry IDs = 18`.

Plan self-review found that two retained bridge tests encoded the old count in
their node names (`tools_count_31`). Changing only their assertions would leave
false test names. The ledger therefore retires those two IDs and adds
`tools_match_registry` replacements. This changes stream identities but not the
final count or native pass/skip arithmetic.

The 33 explicitly retained/evolved base IDs are all present once; their sorted
stream SHA is
`d9cf7a2826d24f72aeb7db840d19bdb979d077e64b23fdf23ccaa79e2e16f67b`.

### 2.5 Projected audit streams

| Projection | Count | SHA-256 |
|---|---:|---|
| training lineage retirement | 4,418 | `284db7fe2fac55bb84ea2bfed4b68a9a566b303b132dda0aaabcfd440978cd56` |
| storage/writer/root | 4,374 | `fcd8775f6255b780c68cb0a943031d49b8f357dd2dcd6da1c8def2af268c19bf` |
| raw DTO/backend | 4,335 | `c6a074a3649b515216402b1b868eb588f57f873474f8b5a15934fcaea48c0d95` |
| raw user behavior | 4,335 | `d17b58f518fb48be84087c6c9169a7738baa478be3d55fac6156449fdc366835` |
| volume/event/composite/Phase D | 4,279 | `7ed812b25be6c29d74d9d3b311d105c218d5eca19b386efa936ae612f291352d` |
| final | 4,282 | `281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |

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
passed:  4488 - 136 score/signal - 101 training - 16 Phase D + 18 new = 4253
skipped:   39 -   2 score/signal -   8 training                      =   29
total:   4253 + 29                                                  = 4282
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

### 2.9 Provider-native protected lineage

Provider-native sentiment and investor-profile risk were found in distinct
owners. They remain protected and receive a dedicated new regression node
rather than being removed through token matching. The former training
protection is superseded by the explicit user ruling in section 2.11.

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

The retired/addition streams, all five then-authorized score/signal phase
projections, focused targets, and frontend target were independently
reconstructed from named owners and matched every then-current section 2 count
and full SHA. Four pre-rebase authority files were also confirmed as identical
Git blobs between `52354806` and `f9958efb`.

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
The amendment assigns the module to phase-1 deletion without changing any node
ledger. Focused review cleared it at `a6e99c02`, revalidated the partial packet,
and allowed Task 0 to resume at manifest construction without rerunning the
unchanged native base gate.

### 2.11 User training-retirement ruling and expanded ledger stop

The next manifest step surfaced an obsolete governance assumption rather than
a hash to normalize away: the plan still protected `training/` under the July
`paused-preserve` ruling. The user explicitly superseded that ruling and chose
direct retirement. Grounding produced these independent facts:

| Witness | Result |
|---|---|
| tracked training tree | 53 paths; current `path<TAB>Git blob` stream `782dd7e42eabaacc814cde180b04f473d6e2433c6dd5e3cc907fa00f96211351` |
| runtime consumers | zero in `src/`, apps, services, schedulers, or data sources |
| external test owners | 8 files / 109 nodes / `db3cad74da2ec956e252096948d80631297e9d4e8c731fb6706da7b2976941b2` |
| focused runtime | `101 passed / 8 skipped / 0 failed` |
| direct deletion | 62 paths / `7c552b4940deeb666cd865656e980f9bba392507e6ed3f9b11b1672269b61c7d` |
| combined retirement at this dated gate | 247 nodes / `149962668e116460f4b88402b1fabb8bb24f0a3409e33d8cade5924dd34ca671` |
| ownerless requirements | 8 package names / 9 lines (`torch` is duplicated); `scipy` remains option-pricing-owned |
| dated final target | 4,298 nodes / `38705c3d431238f5fecb15d3dd4a668cee41912005bfc883d8b4e7275b5efee6` |
| dated native arithmetic | `4,269 passed / 29 skipped / 0 failed` |

The 62 deleted paths are all 53 tracked training files, the eight dedicated
test files, and `tests/live/smoke_yfinance.py`. The same atomic cutover removes
only their proven ownerless package/config/ignore/current-copy tail. Git history
is the sole archive: no preservation branch, tag, copied directory, tarball,
disabled scaffold, or compatibility import is authorized. Future RL, Signals,
or provider-backed options research begins from a new design and current data
contracts.

This is a deliberate product-scope change, not a protected-pin correction. At
this dated gate, `+18/-247 -> 4298` superseded `+18/-138 -> 4407`; section 2.12
subsequently supersedes both. Product bytes remained unchanged, and Task 0 and
Task 1 stayed paused.

The user additionally required the deletion to remain intelligible from Git
history alone. The expanded plan therefore pins subject
`refactor: retire legacy scoring and training lineage` and requires its body to
record the 62-path training scope and then-current 247-node total, retirement
reason, preserved capabilities,
future new-design rule, and the untouched production score-row/secret boundary.
Generic cleanup wording is not admissible.

### 2.12 Task 0 Phase D and PostgreSQL owner-closure stop

Resumed exact owner construction found two omissions that could not be assigned
to a later cleanup without violating the approved no-tail ruling:

1. `src/analysis/pipeline.py` was only one file in an active 18-path Phase D
   package. The surviving factory, service, templates, renderer, API route,
   scheduled job, CLI commands, and enabled config all formed one current
   recommendation surface. Its default strategy chain consumed the retired
   sentiment field and emitted weighted `buy`/`hold`/`sell` output. The future
   on-demand analysis goal does not make this implementation valid scaffolding.
2. `sql/002_add_news_scores.sql` remained an executable score migration, while
   `sql/001_init_schema.sql` still created legacy score columns, a `signals`
   table/index/RLS example, and a sentiment-summary helper.

The corrected disposition deletes all 18 `src/analysis/` paths,
`src/api/routes/analysis.py`, `tests/test_analysis_pipeline.py`, and
`sql/002_add_news_scores.sql`; removes the Phase D route/job/CLI/config/current
copy; evolves generic job/report fixtures rather than deleting those shared
capabilities; and removes only legacy score/signal DDL from
`sql/001_init_schema.sql`.

| Witness | Result |
|---|---|
| Phase D/PostgreSQL direct deletion | 21 sorted paths / `635d5091410cbb953cadb768aa190b23690e035877188b0b58ccf9e160fcdba9` |
| additional retirement | 16 passing nodes / `d190b082db0ce6a231b1a93e7de7fef682c0ef3ce48d7c96fd4522e51fa7fee7` |
| combined retirement | 263 nodes / `93459510fc09e961b0d726527d953ed6fdfd07c584d598ee1de9a60851ca3eda` |
| whole/mixed composition | `223 + 40 = 263` |
| phase 4 projection | 4,279 nodes / `7ed812b25be6c29d74d9d3b311d105c218d5eca19b386efa936ae612f291352d` |
| final target | 4,282 nodes / `281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |
| focused final | 421 nodes / `385d0ac7a142ba1cb488a1dccd3d1a7ae8e2065585b59130f4b3bf75120a2739` |
| retained/evolved owners | 33 nodes / `d9cf7a2826d24f72aeb7db840d19bdb979d077e64b23fdf23ccaa79e2e16f67b` |
| native arithmetic | `4,253 passed / 29 skipped / 0 failed` |

The 16-node stream is exactly all 12 `tests/test_analysis_pipeline.py` nodes,
two `TestAnalysisEndpoint` nodes, one `analysis_watchlist_batch` node, and one
analysis-only summary node. It has zero intersection with the earlier 247-node
stream. Eleven additional unchanged node IDs are explicitly retained/evolved:
two EIR boundary owners, seven generic job persistence/status owners, and two
generic service/job route owners. They move to current authorities/current job
fixtures rather than preserving a retired Phase D name. All 18 additions remain
unchanged and absent from base. Product, test, production data, provider,
scheduler, and secret bytes are unchanged. Task 0
stopped before RED or implementation for independent review of this expanded
owner map.

### 2.13 Task 0 exact manifest completion

Independent focused review cleared the Phase D/PostgreSQL amendment at
`ea4429b0` and authorized Task 0 to resume. The unchanged native baseline
waiver remained valid because `5be77be2..ea4429b0` changed governance documents
only. No product or test byte changed before this manifest pass.

The supplemental packet is
`/tmp/scripts-tranche-b-task0-ea4429b0`. Its 19-entry `SHA256SUMS` has SHA-256
`ff18e224e393790a26bd04b74278912824b63a59d4406afd277f98cd7d4a4b56`;
the deterministic builder is `/tmp/build_tranche_b_task0_manifests.py`, SHA-256
`380fda9fbf6f06658d1c213e7dd0b8d7bfc5ccbaf8ae50368e0a2b3e855b522f`.
It supplements rather than rewrites the earlier 637-entry partial packet
`3be60a14192520da2977e4a1604f1c4c251052ff94e3c98719692e99c587822a`.

| Manifest witness | Result |
|---|---|
| exact owned paths | 196 rows; SHA-256 `c332d72367fc38876bdee57b5d097745a0cb18b00a091de43f5aba929238b282` |
| direct deletion paths | 113 rows; SHA-256 `ddb853a5937d452192e44564d09bc8fa83aa21db5cd3a552f675b16585d096a3` |
| byte-protected paths | 123 rows; SHA-256 `da5c2477eb2848e803ee8088750e41f2ba869935be6fb3e23cee4573e4599c86` |
| behavior-protected pre-blobs | 10 rows; SHA-256 `b360f7d601c1aa3870a9496a40f9ee5ce7ea1848aa3a540fb9df22f74ad97980` |
| Phase D dispositions | 40 rows; SHA-256 `09b63c5031e314a4feab1f415e2e56145d3149301e2921c06d0cae7a79c94ca5` |
| OAuth reviewed handoff | exactly `apps/arkscope-web/src/api.ts` and `PROJECT_PRIORITY_MAP.md`; all other recorded OAuth paths remain byte-protected |
| training family | 53 Git blobs `782dd7e4...`; direct deletion 62 paths `7c552b49...`; nine ownerless requirement lines; zero surviving Python import |
| retained numerical owner | `scipy` remains imported by `src/options_math/option_pricing.py` |
| Phase D direct deletion | 21 paths `635d5091...` |
| exact node witnesses | retired 263 `93459510...`; final 4282 `281cad97...`; focused final 421 `385d0ac7...`; retained/evolved 33 `d9cf7a28...` |

The scoring-key metadata still exactly matches the first Task 0 observation:
existence, mode `0600`, inode, and mtime only. The isolated worktree contains no
gitignored secret. The builder used `lstat` against the main worktree and did
not open, hash, print, copy, size, change, or delete the file. Production score
rows and the market database were not touched. Task 0 therefore closes with
exact owner, deletion, protected, and handoff boundaries; Task 1 may establish
the reviewed 18-node RED without committing a knowingly RED tree.

### 2.14 Task 3 stale shared registry-count owner stop

After the approved `+18/-263` cutover reached exact backend
`4282/281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3`,
focused `421/385d0ac7a142ba1cb488a1dccd3d1a7ae8e2065585b59130f4b3bf75120a2739`,
frontend `1124/da69a2942c03e4794e3384e6125936f9f25c1fafbad7d006b67025f8fd97bc39`,
and frontend focused
`28/b11cc27b90c570b20aeb728c48d39497e7eff555976f0c00a42d6129b26cf1cd`,
the broad protected suite stopped before the atomic commit.

The exact protected command covered 58 files from the Task 0 byte/behavior
manifests and produced `1,141 passed / 6 failed`. All six failures were in
`tests/test_sa_tools.py`; each pinned a pre-cutover aggregate count while its
SA-specific assertions passed. A repository-wide exact-pattern census then
found four more shared count owners outside that protected command. Running the
closed ten-node set produced `10 failed` for the same intended reason:

| Expected before cutover | Current reviewed truth | Owners |
|---:|---:|---:|
| registry/schema `53` | `50` | eight |
| Anthropic bridge `54` | `51` | one |
| news category `10` | `11` | one |

The ten-node sorted stream is
`3c7e2870264e5959a6418701553af6a8870f2adde30b18b9c35e326056b4c305`;
the original 33 retained/evolved nodes plus these ten produce exact 43-node
stream
`7e4f4d2b5290f47c368227223a558043a040304eb6a042af5519e0207a91ed54`.
No node is added, removed, renamed, parametrized, or skipped, so all collection
and native targets remain unchanged.

Task 0's 123-row byte-protected manifest incorrectly protected all of
`tests/test_sa_tools.py` despite the approved registry evolution. The bounded
replacement excludes only that one file and pins the other 122 rows to stream
SHA-256
`c174c7d7b7e9731d4cb04bf00a7b40af1fcaacee5c09f6c77c3f2c585d6f9ca2`.
Only six numeric assertions in `test_sa_tools.py` and one numeric assertion in
each of `test_analyst_tools.py`, `test_memory_tools.py`,
`test_portfolio_tools.py`, and `test_sec_tools.py` may change. The implementation
stopped without an atomic product commit or native target admission pending
independent review of this amendment.

## 3. Locked implementation sequence

| Task | Status | Gate |
|---|---|---|
| Task 0 re-ground | complete; independently GREEN | exact owner/protected/deletion manifests complete |
| Task 1 RED | complete | intended RED artifacts preserved; no RED-only commit |
| Task 2 atomic cutover | complete at `8ebf7fae` | exact target identities reached |
| Task 3 mutation/focused gates | complete | registry amendment `54d442d2` independently GREEN; M1-M11 restored |
| Task 4 native admission | complete | `4282 seen / 4253 passed / 29 skipped / 0 failed` |
| Task 5 independent implementation review/merge | review pending; merge blocked | packet `d5917eb7...` |
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
- [x] Six intermediate projections labeled accounting-only.
- [x] Eleven product mutations mapped to owning witnesses.
- [x] Provider-native boundaries byte-protected; EvidencePacket copy delta
  bounded while its negative behavior remains protected.
- [x] Native wakeup/reporter/wrapper/toolchain boundary pinned.
- [x] Independent full-plan review GREEN at `52354806`.
- [x] One-time rebase completed with unchanged relative ledgers and exact new
  full/projection identities.
- [x] Focused rebase-amendment review GREEN at `5be77be2`.
- [x] Focused no-tail owner amendment review GREEN at `a6e99c02`.
- [x] Expanded direct training-retirement plan independently GREEN at `92f51f7e`.
- [x] Phase D/PostgreSQL owner-closure amendment independently GREEN at `ea4429b0`.
- [x] Task 0 exact owned/protected/deletion manifests independently GREEN at `3d49d139`.
- [x] Shared registry-count amendment independently GREEN at `54d442d2`.
- [x] Atomic product cutover committed at `8ebf7fae` with the exact reviewed subject and body.
- [x] M1-M11, focused/protected, frontend, production-boundary, and native admission gates complete.

## 5. Honesty boundary

This packet proves that the plan is grounded and internally accounted. It does
not authorize merge, physical score-row deletion, or scoring-secret
disposition. The implementation and final suite evidence are recorded below;
the destructive claims still require their later gates.

## 6. Tasks 1-4 implementation and admission

### 6.1 Atomic product cutover

The complete approved cutover is one product commit:

```text
8ebf7fae14bcd1136ae3e9f1c2bfbed05b00da6c
refactor: retire legacy scoring and training lineage
```

The commit changes 193 files (`1,393` insertions, `31,384` deletions). Its
113 direct deletions are byte-for-byte the reviewed
`all-direct-deletion.paths` stream; every changed path belongs to the exact
201-path authorized universe and the unauthorized difference is empty. The
commit body records the 62-path training family, exact `109 + 138 + 16 = 263`
retirement, 18 replacement contracts, preserved raw/provider-native/options
capabilities, future new-design rule, and read-only score-row/metadata-only
secret boundary.

Final collection identities are exact:

| Surface | Result | SHA-256 |
|---|---:|---|
| backend full | 4,282 | `281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |
| backend focused | 421 | `385d0ac7a142ba1cb488a1dccd3d1a7ae8e2065585b59130f4b3bf75120a2739` |
| frontend full | 99 files / 1,124 | `da69a2942c03e4794e3384e6125936f9f25c1fafbad7d006b67025f8fd97bc39` |
| frontend focused | 4 files / 28 | `b11cc27b90c570b20aeb728c48d39497e7eff555976f0c00a42d6129b26cf1cd` |

### 6.2 Focused, protected, and mutation gates

- the ten amended shared-count owners are `10 passed` at registry `50`, bridge
  `51`, and news category `11`;
- the exact backend focused runtime is `404 passed / 17 skipped`;
- the broad protected runtime is `1,222 passed`;
- all 122 byte-protected rows reproduce stream
  `c174c7d7b7e9731d4cb04bf00a7b40af1fcaacee5c09f6c77c3f2c585d6f9ca2`;
- frontend focused is `28/28`; native full Vitest is `1,124/1,124` across 99
  files; typecheck, build, and i18n scanner are GREEN, with scanner result
  `36/20/0/20`; and
- M1-M11 each changed the real reviewed owner, made its owning contract RED,
  and restored the exact pre-mutation SHA. The mutation manifest SHA is
  `5bf15136a76f1e6b504adeba4f912d12a5a290a50bd34904137a8f475ea2a97a`.

A supplementary managed-sandbox full Vitest attempt is rejected evidence: all
19 failures came from the two known subprocess-owning files receiving
`spawnSync ... EPERM`. The immediate native run on identical bytes passed all
1,124 nodes. Earlier split controls also passed the scanner owner `18/18` and
the exact complement `1106/1106`.

### 6.3 Production and native admission boundaries

Read-only SQLite URI mode plus `PRAGMA query_only=ON` still reports exactly
491,808 `news_article_scores` rows and 140,152 distinct article IDs. The
pre/post score projection and metadata-only `config/scoring_keys.txt` witness
are byte-identical; secret content, size, and digest were never read. The DB
identity was stable during each witness query.

Fresh exact-tip native admission used the pinned wakeup probe, wrapper, and
reporter outside the managed sandbox:

```text
4282 collected = 4282 seen
4253 passed / 29 skipped / 0 failed
exit 0
non-passing SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
report SHA-256 252535bf53aa30f93995f2633b39415c28c9bfec94bbea4af5576a9ec320de5b
transcript SHA-256 da604e029b880314fe9c30115f596ba3a5fc67c7f299356e99834b4cd42ded67
```

The run generated 506 ignored files. Every file received relative-path,
inode, size, mode, mtime-ns, and SHA evidence and was renamed on the same
filesystem into packet quarantine. The corrected directory inventory records
35 generated/untracked directory paths and 60 complete generated parent paths.
Ordinary status returned empty; ignored status returned to only the pinned
`node_modules` symlink; `data/` returned to present-and-empty; `src/data`
remained absent. The clean single-use native worktree was then removed through
`git worktree remove`, and its registration is absent.

The first reconciliation attempt is retained as rejected evidence. Its
ignored-file inventory could not see four empty `data/news/...` directories,
so the final empty-`data/` assertion refused the attempt. All 506 files were
restored by exact path with identical metadata before the corrected run.

The raw review packet is
`/tmp/scripts-tranche-b-task4-8ebf7fae`: 668 retained files, all verified by
`SHA256SUMS`; manifest SHA-256 is
`d5917eb703081d80d66b042886c581ba09df7950bc2a7ffb3061a910551b3ce5`.
Merge, production score-row disposition, and scoring-secret disposition remain
blocked pending their separately reviewed gates.
