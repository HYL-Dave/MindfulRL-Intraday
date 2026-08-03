# EIR-006 Valuation Price Truth Evidence

> **Status:** TASK 6 BLOCKED - FRONTEND SEARCH-COMPATIBILITY AMENDMENT REVIEW
>
> **Date:** 2026-08-03
>
> **Design authority:** `124622bc`
>
> **Implementation plan:**
> `docs/superpowers/plans/2026-08-03-eir-006-valuation-price-truth.md`

## 1. Authority And Grounding

### 1.1 Reviewed design

```text
path:
docs/superpowers/specs/2026-08-01-eir-006-valuation-price-truth-design.md
commit: 124622bc
lines: 1168
bytes: 50661
SHA-256:
7ec313d12e87b0e6557172c173d7f9d55468f36551514d463f6faaa0ff62bee4
review verdict: GREEN, zero findings
```

Plan-gate status evolution only:

```text
SHA-256:
a5c155d45ba690fd7e29aeaa37d49eeb44dc224496d51f3733f717afa0686f6c
semantic contract delta from reviewed blob: none
```

### 1.2 Plan-construction boundary

```text
worktree: /tmp/arkscope-eir-006
branch: codex/eir-006-valuation-price-truth
grounding commit: fd6d1b86383df2a98f97b235d9796d4bcaaa7a58
plan base: 124622bc
plan-review clearance tip: e261abc25de5fdc608feea7bbe68fe230cb05789
plan-review verdict: GREEN, zero findings
product edits during plan construction: none
provider requests during plan construction: none
production writes during plan construction: none
physical data movement/deletion during plan construction: none
```

### 1.3 Pinned native assets

| Asset | SHA-256 |
|---|---|
| `/tmp/arkscope_asyncio_wakeup_probe.py` | `10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e` |
| `/tmp/eir002-green-baseline/arkscope_eir002_reporter.py` | `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928` |
| `/tmp/eir002-green-baseline/run_native.sh` | `e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f` |

## 2. Pre-Edit Collection Ledger

### 2.1 Backend

| Stage | Delta | Nodes | SHA-256 | Status |
|---|---:|---:|---|---|
| Base | `+0/-0` | 4,553 | `69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca` | Task 0 reproduced |
| Task 1 target | `+10/-0` | 4,563 | `5fdc93f3dc78548048d7269d8088715028a57b1e2c54fe1ac422154d187f3986` | Task 0 preconstructed |
| Task 2 target | `+19/-1` | 4,571 | `b247d173d3520668a5d475b0ed02f948d117c1097ed5ad86063a2dbf76d07b68` | Task 0 preconstructed |
| Task 3 target | `+21/-1` | 4,573 | `e0ee195eb90bc9172dae36680b15b3285b3d82013c7c762e1989c955be6ea3b1` | Task 0 preconstructed |
| Task 4 target | `+27/-1` | 4,579 | `6672d3df26b7c420d3253e4826b7104bfd0e5640ae16a1616ea75dd605b38639` | Task 0 preconstructed |
| Final target | `+29/-1` | 4,581 | `6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f` | Task 0 preconstructed |

Focused base:

```text
14 files / 307 nodes
SHA-256:
46f8c9d0cd9e3b525d051e2231d4d48ed2975192886cdeb293ec71662341ae51
```

Focused target:

```text
18 files / 335 nodes
SHA-256:
58230b548925b29035cff401520e0948b01dcaed8da2deed41149bea6b4a5ae1
```

### 2.2 Frontend

| Collection | Files | Nodes | SHA-256 | Status |
|---|---:|---:|---|---|
| Base full | 96 | 1,076 | `ef7f106054745c137ff70fe6ef2043bb7655185379de1f0a6ec3b1b2997b9396` | Task 0 reproduced |
| Target full | 97 | 1,077 | `3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb` | Task 0 preconstructed |
| Base focused | 4 | 45 | `09a2b4ef080a5badab79fb674b5c5c6b85a0eb7c639c2fe9534616eff2b5bb84` | Task 0 reproduced |
| Target focused | 5 | 46 | `5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127` | Task 0 preconstructed |

Normalization authority:

```text
vitest list --json
JSON decode through the pinned plan normalizer
relative_file<TAB>full_test_name
UTF-8 byte-order sort
normalizer: 62 lines / 2,233 bytes
normalizer SHA-256:
955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac
raw JSON text extraction: forbidden
jq @tsv re-escaping: forbidden
```

Plan-review correction: the first plan checkpoint retained an extra JSON/TSV
escape layer for the two `src/sse.test.ts` names containing literal `\n\n`.
Counts and focused identities were unaffected. The full base/target identities
above are rebuilt from decoded runtime names and supersede only those two
earlier full-stream SHAs.

## 3. Task 0 Grounding

Task 0 ran from exact reviewed tip
`e261abc25de5fdc608feea7bbe68fe230cb05789`. Its commands intentionally changed
no product, test, provider, scheduler, production database, or legacy-data
byte. The only tracked changes in this checkpoint are the implementation plan,
this evidence file, and the priority map. Concurrent extension telemetry is
separately attributed in Section 3.3.

### 3.1 Canonical native base

```text
fresh worktree: /tmp/arkscope-eir006-native-base-e261abc2
report path: /tmp/eir002-green-baseline/reports/eir006-task0-native-base-e261abc2.json
report SHA: 2e94c3af06b7ada6596fb4068a5ca237d7c70edc11d806a370f91166a6f0c174
transcript SHA: 968b0ce78d37da12e25044b1954bb3a2e8a6617c2c5d0d0f855726aa1b200b2e
collected: 4553
seen: 4553
passed: 4481
skipped: 72
failed: 0
errors: 0
exit: 0
collected stream SHA: 69152591306a8dee5e66e2efeb2f1ec12720c8a1a1ffe36def613f4fe5a676ca
non-passing SHA: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
wakeup probe result: {"callback_fired":true,"ready_count":0,"wake_bytes":0}
artifact rows: 575
artifact manifest SHA: aafe16c850dc73a1f36870b2aa5750268016c60dde35ff512add8c04eba203b4
quarantine implementation SHA: af85aa60048ed9a42e6425edb315d5cd22c0b9a839dcff0d79f51bafbb02c14b
boundary restored: yes; tracked diff empty, data/ empty, src/data absent
```

The fresh boundary had no `config/.env`, an intentionally empty `data/`, no
`src/data`, and one pinned ignored `node_modules` symlink. The 575 generated
paths included bytecode/cache files, 15 scratchpad artifacts,
`data/logs/sa_native_host.log`, and
`src/data/cache/risk_free_rate.json`. Every path was inventoried before exact
relative-path quarantine; no pre-existing fresh-worktree file changed.

### 3.2 Consumer census

```text
raw structured submatches: 383
audited exact duplicate occurrences: 8
discovery stream rows: 375
discovery stream SHA: 0c105ac7b247e78d1528efb138f8823fb196c9bb1c92092dba8f52ac194b4b29
consumer-census rows: 375
consumer-census SHA: a8d165a2947f429a6918675fbd1357d8ddbbd11595a4e42b75c6027d8cd2b971
behavior-propagation rows: 4
behavior-propagation SHA: 613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba
unknown verdicts: 0
duplicate rows in admitted streams: 0
empty owners: 0
reconstruction equality: byte-for-byte equal
discovery producer SHA: c3993fc615f491f0fd43a9d69c065203114b527ae68d33720e6b6a50105f4ce3
classifier SHA: 769b0b7a1c96306837fe7db0d4e7d58249822eee0759b340e7b3fab9c6ae5ba6
behavior producer SHA: 0b6409d03193f504de8f4aa46fbcb89aab42cb7ada54e20ad201dd56e1360987
```

Consumer verdict distribution:

| Verdict | Rows |
|---|---:|
| `historical_reference` | 198 |
| `low_level_empty_compatibility` | 7 |
| `retired_current_consumer` | 18 |
| `rewired_current_consumer` | 59 |
| `test_fixture_reference` | 83 |
| `unrelated_lexical_hit` | 10 |

The behavior census proves four live forwarding surfaces: Anthropic bridge,
OpenAI bridge, `/fundamentals`, and the institutional evidence packet. The
OpenAI bridge was absent from the reviewed plan's owner list. Task 0 amends the
plan to include its product file and existing reachability node; node counts and
all preconstructed hashes remain unchanged.

### 3.3 Protected production boundary

Record metadata/identity only. Do not include secret or data payload contents.

```text
market DB identity: inode 127284871 / 3,409,027,072 bytes / mtime_ns 1785696259805305153
market DB SHA: cfb8a0aa9e8df94b3d19543c0f548a22a9aa0bf9a061ca3866744e1f5adbeb9d
retired file metadata stream: 301 paths / 3f353d120b5054dd43406ea85694b5c3f804dc4cd4559398134d307de957b344
retired DB-row counts: old cache 19 / fundamentals 130 / fundamentals sync 1
market totals: financial_cache 46 / sync 3 / prices 2,406,320, all 15-minute
price frontier: 2024-01-02T14:30:00+0000..2026-07-31T19:45:00+0000 / 151 tickers
scheduler state: four sources succeeded, no error/continuation; ibkr_prices enabled / 720 minutes
protected source blob stream: 31 paths / ecb45d1260e9542ddc705d73cc11af2e27bcfa5ca09ac5e16b12f8daac8df6a8
protected path-list SHA: 3778006be667c4aa5d0ee7c687bedd5401fdaa4473eab07b1c4fd08bbab2388d
current FD-enable source slice SHA: bb87764bb49bcab870d4cda5f6bd3094596eea0946ee108755faa3da9217ec53
pre snapshot SHA: ee14800914947c6bc656e740da43ad079a8dba5f9b46f6b2b2a80395a5ea5f35
post snapshot SHA: ec657916cd6e18e503bb8b7714ec80be4b1058552ff2ee0e98ee30f619c7ef3c
market DB/prices before-after equality: byte-for-byte equal
```

All production reads used SQLite URI `mode=ro` plus
`PRAGMA query_only=ON`. The market database, its counts/frontier, the 301-path
legacy-file metadata stream, schedule settings, and scheduler-state rows were
unchanged after native admission.

`data/profile_state.db` did change during the native window: it grew from
45,776,896 bytes (`662aeae4...`) to 45,785,088 bytes (`69c78260...`). Read-only
attribution found exactly three new `job_runs` rows, IDs `18626`, `18627`, and
`18628`, all successful `sa_market_news_refresh` runs with
`trigger_source=extension`; every other time-aware table had zero rows in the
window. The attribution artifact is
`a30c445fe6a088e93e77800887f2f7d91a64cc08afb1c34fc5a00382ad3c1375`
and its query implementation is
`faee8bfaac6b953813228ddf0c32e1a990778fe9bf33184c3cefa1019666483d`.
These external extension writes were preserved. They are not attributed to the
fresh native test run, and this evidence does not claim all production bytes
were static.

### 3.4 Grounding corrections and rejected attempts

- Two frontend-list attempts were rejected before admission: one lacked the
  pinned toolchain from its execution root; one passed a root shape that made
  Vitest resolve the configuration from the wrong directory. The admitted
  third attempt ran from `apps/arkscope-web` and reproduced all four reviewed
  identities through the pinned JSON-decoding normalizer.
- The first detached-worktree creation failed at git-crypt smudge and left no
  registered worktree. The successful retry used the repository's established
  no-op git-crypt filter for this plaintext reviewed tree and rechecked exact
  `e261abc2` before admission.
- The plan's illustrative base and tip commands supplied worktree/output paths
  to a wrapper that accepts one stage name and operates on the current
  directory. Task 0 used the wrapper's actual pinned interface and amends both
  commands before Task 1. The consumed Task 0 stage must not be reused;
  independent review uses a new single-use stage under the same boundary.
- No rejected attempt is used as collection, census, native, or no-write
  evidence.

## 4. RED-First Implementation Record

### 4.1 Task 1 - session authority and selector

```text
structural/product RED:
  tests/test_valuation_price.py: 10 failed, all at function-level imports
  because src.market_sessions/src.valuation_price did not exist
  TestDetailedFinancialsSchema: 3 failed because valuation_price_basis did
  not exist
wrong-RED corrections: none
intermediate product correction:
  Python 3.10 rejected the real stored +0000 timestamp suffix; the selector
  parser normalized that suffix to +00:00 without changing test expectations
GREEN: 83 passed in 1.40s
  tests/test_valuation_price.py
  tests/test_market_data_direct.py
  tests/test_detailed_financials.py::TestDetailedFinancialsSchema
stage collection:
  RED and GREEN both 4,563 nodes
  5fdc93f3dc78548048d7269d8088715028a57b1e2c54fe1ac422154d187f3986
  both streams byte-identical to the preconstructed Task 1 target
market_data_direct existing nodes:
  70 passed; 70 IDs byte-identical to base
  584cdd096455f7d86904d7f208e72c6cc597e4bf7f569c726aac16b199f618cb
no-create artifact witness:
  each missing nested path, directory, broken symlink, junk DB, missing table,
  and missing-column fixture captured the complete tmp_path tree before and
  after lookup; every pair was byte-identical, the missing parent remained
  absent, and no database, -journal, -wal, or -shm path appeared
compile witness:
  py_compile passed for all four product and both test owners
commit: 33a628b0912cac89ad1afde8ba96d049b7329125
file SHA-256:
  src/market_sessions.py = 4b34b351da0c617637059f113d11890bcabc63b52a1b9a2df7c65b06feb5dc41
  src/market_data_direct.py = 412865670c39feff84d9d2216dfd2a727ecc505be250c8de6d7bd3c0fb3ea735
  src/valuation_price.py = f0dcb02ee0826bda533cd09d2308b24b4739bd6b1204a5084453d3b9f63ac8be
  src/tools/schemas.py = 7249e18bcfa9349086a876d60ca75e8246afc2338b06defdc9790a9bb9a50fbd
  tests/test_valuation_price.py = 7d3408042aef72fca1244443e817e91d932de4a71dbf62bc8cfb5156f813e643
  tests/test_detailed_financials.py = 3caacdc4ae94c46e4163768598223b8aa1b4db998ec3cdd7469228fd0bea3ed3
```

### 4.2 Task 2 - static/dynamic split

```text
RED:
  pytest tests/test_detailed_financials.py \
    tests/test_financial_metrics_calculator.py -q
  -> 10 failed / 10 passed / 1 skipped
  expected causes only: old semantic cache key and legacy snapshot path remained,
  no closed v2 static payload or pure formula existed, cache hits could not
  reselect price, unavailable price did not null all nine dynamic fields, and
  the calculator still reached repository files
RED correction:
  the first helper draft passed " test ", which made ticker normalization a
  competing failure. The helper was changed to "test" before accepting RED;
  no product byte changed, and the accepted 10-failure run exercised only the
  reviewed missing behavior
old key read calls:
  the replacement node observed the legacy metrics_TEST_annual_y2 read before
  implementation; GREEN observes only
  detailed_financials:v2:sec_edgar:TEST:annual:y2
static cache forbidden-field witness:
  one GREEN write has the exact nine top-level v2 keys, source sec_edgar,
  ttl_days 90, and recursively contains no price, timestamp, market_date,
  valuation_price_basis, product dynamic field, or calculator dynamic field;
  each forbidden spelling injected under a nested object is rejected
cache-hit price recomputation witness:
  the same valid static payload served prices 10 and 20 as market caps
  20,000,000 and 40,000,000; the selector ran twice, the SEC calculator was
  forbidden from construction, no cache write occurred, and both earnings
  seams still ran twice
base-unit calculation witness:
  price 10 * 2,000,000 shares = 20,000,000 market cap exactly; dependent
  missing inputs null only their dependent fields, while price=None nulls all
  nine dynamic fields through calculator, convenience, snapshot, and CLI seams
GREEN:
  pytest tests/test_detailed_financials.py \
    tests/test_financial_metrics_calculator.py \
    tests/test_fundamentals_cache.py tests/test_agents.py tests/test_tools.py -q
  -> 81 passed / 1 skipped
GREEN environment note:
  the first affected-suite invocation produced five FileBackend setup errors
  because this isolated worktree lacked the established empty data/ project-root
  marker. After creating only that empty directory, the identical command gave
  81/1/0; the directory was removed afterward and no repository-relative
  artifact remained. This was a harness precondition, not a product correction.
compile witness:
  py_compile passed for all three product and both test owners
stage collection:
  4,571 / b247d173d3520668a5d475b0ed02f948d117c1097ed5ad86063a2dbf76d07b68
  byte-identical to the preconstructed Task 2 target
removed/replacement node identity:
  exactly -1: TestGetDetailedFinancials::test_ibkr_enrichment_overrides
  exactly +9: the six reviewed TestGetDetailedFinancials nodes plus the three
  reviewed test_financial_metrics_calculator.py nodes; no other ID changed
pre/post source SHA-256:
  src/fundamentals/cache.py
    f2797b59537e53c72cd8f68e01b6bccc267069aaa7a341b299615e7a680e8ad6
    -> 63356388650de6f4cc5cdfc9eacb6f0da97b32980bfa740dd49b75f061b3644d
  data_sources/financial_metrics_calculator.py
    29cf804d4b1c58484dc9e5401bc18dc3961c8a32650e37f8c455a7806e1fd588
    -> 5faac9ea76f5ac74d91a69e2d161a9ace4d4dc25e1e654dfb81ee8e78159dcb4
  src/tools/analysis_tools.py
    6fb024266a4476560de49d6e939b99c40b77465b069dcc77bfc239c453081399
    -> 6bcf3701930e3f64cd8758b04af9ca218a1850c37f474823d39a770c2d75380b
  tests/test_detailed_financials.py
    3caacdc4ae94c46e4163768598223b8aa1b4db998ec3cdd7469228fd0bea3ed3
    -> 2d33afa9f2055c4ef2d8972e7317f7df5a063b629e87733bc33c1e6f098d4ae7
  tests/test_financial_metrics_calculator.py (new)
    70fa3e132b9f16a0d55eb8617e1895e7051b69bc4283a97e3c5c41289b838c70
commit: b9efcd33954dd1f73658e2164cfdb00bc155492d
```

### 4.3 Task 3 - annual analysis and peer absence

```text
RED:
  pytest \
    tests/test_fundamentals_sec_cache.py::test_annual_analysis_ignores_legacy_snapshot_and_preserves_sec_fd_order \
    tests/test_peer_comparison.py::TestDataQuality::test_unavailable_valuation_prices_are_counted_named_and_excluded \
    -q
  -> 2 failed
  annual failed only because the contradictory legacy snapshot returned before
  the positive SEC cache; peer failed only because the three reviewed
  valuation-price data-quality fields were absent. No provider or fixture error.
SEC/FD call-order spy:
  positive SEC cache -> no SEC constructor and no FD gate
  SEC miss with positive fixture ->
    sec:init, sec:income, sec:balance, sec:cashflow, build:sec_edgar
    (FD gate absent)
  SEC empty + FD disabled -> fd:disabled only; typed empty result
  SEC empty + FD enabled ->
    fd:enabled, fd:init, fd:income, fd:balance, fd:cashflow,
    build:financial_datasets
legacy snapshot spy:
  a contradictory ibkr result (snapshot_date 2025-09-30, roe 9.99,
  market_cap 999, private marker) was installed on dal.get_fundamentals;
  cache-hit, SEC-positive, FD-disabled, and FD-enabled paths all recorded zero
  legacy calls
peer absence closed payload:
  three static peers remain in peer_count/comparison_matrix; GAMMA keeps its
  static gross margin but has null valuation. PE statistics/rankings use only
  ALPHA/BETA (count/of 2), while data_quality is exactly:
    valuation_price_unavailable_count = 1
    valuation_price_unavailable_tickers = [GAMMA]
    valuation_price_empty_reason_counts = {no_qualified_price: 1}
behavior-propagation owners:
  full reviewed ledger executed Anthropic schema, OpenAI schema, default API,
  stored API modes, direct tool, and both evidence-packet owners against local
  positive SEC cache/provider spies
GREEN:
  full Task 3 propagation command -> 32 passed / 1 skipped
  py_compile passed for the product owner and all five test owners
stage collection:
  4,573 / e0ee195eb90bc9172dae36680b15b3285b3d82013c7c762e1989c955be6ea3b1
  byte-identical to the preconstructed Task 3 target; exactly +2/-0 from Task 2
pre/post source SHA-256:
  src/tools/analysis_tools.py
    6bcf3701930e3f64cd8758b04af9ca218a1850c37f474823d39a770c2d75380b
    -> 00d966a2120822a8ec61be08eb835edcc90ceb3d26a3034e54a6636dd6fd2044
  tests/test_fundamentals_sec_cache.py
    29ec832c5baf21c8a7367790574e2c8fe262b22ab462131efd9633aac4beda3c
    -> a4b18614a6549e4919932c98b42767ca8f329c452ba0ef38ddf590689918147c
  tests/test_peer_comparison.py
    2cafa4fcdd3d128ae279ba5956f669295d4bfcd80d6c7952f2f0a5d30fc89be5
    -> 426fea9292933cd20b4827d665a52a2e4b133deac028d6c01b656b07f6e56c3f
  tests/test_api.py
    6a60ca1bc95c514222c508adac5a40b978704b345a3aa9f0181d4fb11f568bf2
    -> 4075d5ef20345fea6878bfb8f0f43a1db6fa5a9543e6e6abfcf9922a057fbfa2
  tests/test_tools.py
    45b89dfce6e806c30cbb5a60b2a49460e1b749e46d16e49e5716987ee6fae717
    -> 3cde92cd1c6805dacae5739c730a3f018135479f6d86a145aa0a0ba84278f976
  tests/test_evidence_packet.py
    495214bf83f9fa01739b9014f1a9064d455f2bc0663fa9342e145d59eef58148
    -> 4c0a7ff395979ccedc9191f50803111356a207f4854d421c7d34b831b067720a
commit: 56574530a2a2a356c04f0f05de16d7ca64746019
```

### 4.4 Task 4 - stored SEC projection and retired files

```text
RED:
  exact six-node command -> 6 failed
  every failure was the reviewed contract: the legacy fundamentals table still
  projected as stored, projection consumers disagreed, fundamentals sync was
  non-null, daily-update scanned data/prices, and FileBackend probed retired
  price/fundamentals paths. There was no fixture, provider, PG, route-lifespan,
  or collection error.
shared projection multi-consumer witness:
  one market_data.db fixture contains one positive annual SEC v1 cache row plus
  legacy, negative, expired, malformed, quarterly, Financial Datasets,
  old-metrics, v2-detailed-financials, no-snapshot, and ticker-mismatch decoys
  stats -> {row_count: 1, ticker_count: 1, latest_date: 2025-12-31}
  admin coverage -> true; SQLite/local DAL ticker lists -> [AAPL]
  coverage tool -> one row with the same 2025-12-31 bounds
  /status -> fundamentals_tickers 1
  /fundamentals/AAPL?stored=true -> local_cache / 2025-12-31
  PG fallback sentinel was untouched
sync price/news preservation witness:
  both admin and coverage projections return the fixture price/news objects
  byte-for-byte by value while retaining fundamentals: null
FileBackend no-probe witness:
  Path.exists/glob/rglob and pandas read_csv/read_parquet raise on retired roots;
  prices retain the exact six-column empty frame, fundamentals {}, and both
  ticker lists [] without touching any sentinel
daily-update no-scan witness:
  injected SQLite stats map to the unchanged closed keys exists/total_bars/
  latest_date/tickers while every retired repository scan primitive raises
GREEN:
  six RED owners -> 6 passed
  projection owner set -> 113 passed in the native execution boundary
  projection owner set plus existing fundamentals-cache contracts ->
    118 passed in 6.20s
  the first sandbox invocation stopped after one node at the already established
  EIR-005 asyncio wakeup boundary; it was interrupted with exit 130 and no
  partial result was admitted. The identical owner gate completed natively.
  py_compile and git diff --check passed; the retired private price loaders are
  absent and the SEC cache-key grammar remains owned only by fundamentals/cache.py
stage collection:
  4,579 / 6672d3df26b7c420d3253e4826b7104bfd0e5640ae16a1616ea75dd605b38639
  byte-identical to the preconstructed Task 4 target; exactly +6/-0 from Task 3
  reporter JSON SHA:
    10cdb1d17ebe8b33a62a4a1704663900d9362592570a8b72fe9f258327e782bf
pre/post source SHA-256:
  src/fundamentals/cache.py
    63356388650de6f4cc5cdfc9eacb6f0da97b32980bfa740dd49b75f061b3644d
    -> 9a05e95ec7552b74fb79bd38571df53e5ca0d8913344f11d71e46cec49b734ea
  src/market_data_admin.py
    90cfc938f5a3d5db1ecc50c7f1e81fb6d793357a8a276db93c8a51f7161751b5
    -> f4ae17f7e8d277030015b9aedbdbc835ae95758213c234d6486d532735c3ce3c
  src/tools/data_coverage_tools.py
    db663139355f0eb8cdd27fa375293bfaccbb7bb65e9934d277e1e910c325d913
    -> 4db1e45d15c681fa53879db5124f1ae948d2f9872e33a814ffdc99cf95f4d95d
  src/tools/backends/sqlite_backend.py
    b8572be50a0c4d3fb88ca1ed36be75b9bd419cf5c148fcce62afd5ff2e388b35
    -> 328a33c06bc65b545988a2ef1cfd666a264e02aebabf5c38277bdd7518a2c09c
  src/tools/backends/local_market_backend.py
    d96f551f99086dcac83d095fe6b65627f25442cbd79792a9f20146725da483de
    -> 4262cf1d46860b86091e80fe99af9649d473b166a67050f453cb649cc813664b
  src/tools/backends/file_backend.py
    08ba230ba0b6706bb5142c8f6074546e413c849bfb07283eb06f79001458601a
    -> 322c6bc0efe05bb47e73aba935e95938a3dd0b2606c1920f5e3b170d3632338c
  src/daily_update.py
    3bfd28f7c12e37bcca18bf7624f3dfb5da75e7f1e2861b7497d8b1c20604b51e
    -> d159ee2b07c33dbd531639d5c234967567dab0df4153ab880f454f6339bea36a
  tests/test_stored_sec_projection.py (new)
    eeae21c05cfd51fc71b094ebc3b4d5516e98c6cdbc573e14d69fffc8f4cad45d
  tests/test_market_data_admin.py
    5cdff94373821f38a942ee576d46671e6dff90fcd1777f56125a1269d210e16b
    -> e1747b2249ef3994e85d56165aed36f1c4c46edc2741cbff3bcfd290c93cee1c
  tests/test_data_coverage_tools.py
    2faf2a6cca23f7b08f30205e573f1b7fc7a35f90aa70bc02b4d1f892f43964ef
    -> d011cac6ec66425cecefca4cc8e2bb2b00a8c714942805e2acb8f15a39c22671
  tests/test_sqlite_backend.py
    68459f4c16a829a436694d55a07d1a054bcdfc8a035ed6a6e655c45941dd7dc7
    -> b6e34506dd1cc0a57de987488516702694d9d0541e423b2e83a8dd109cc00a1e
  tests/test_daily_update_wrapper.py
    257e391d3f08c74236691c52889725311a0671007601646a5e9b3595b8c7168c
    -> 04dae5f1785661805798360ea60ed1cb5d3456634f6a625a2c1c6bbe1f7d6a50
  tests/test_db_backend_retired_prices.py
    2122ec8d262f9b0fe326928abbabb960e2568d59bedab46edba45831b47ceab9
    -> b231f655ff5526a7b769791170357497f5d8fc8a5ff9aed497be432c010a7a99
commit: 8bd65a722d47d611384f242793bc9ee8df460b74
```

### 4.5 Task 5 - frontend/current copy/static census

```text
backend RED:
  2 failed; both failures were the reviewed stale-current-authority contract
frontend RED:
  46 collected / 37 passed / 9 failed
  every failure was an absent stored-SEC/source-label/current-copy contract;
  there was no import, jsdom, route, network, or collection failure
backend GREEN:
  static-boundary owners -> 2 passed
  owners plus tests/test_agents.py and tests/test_tools.py -> 58 passed
  the first combined invocation lacked the known empty data/ harness marker and
  produced five FileBackend setup errors; it was not admitted. With an empty
  data/ marker the exact gate passed, generated no data bytes, and the marker
  was removed.
frontend focused GREEN:
  5 files / 46 passed
backend collection:
  full -> 4,581 / 6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f
  focused -> 335 / 58230b548925b29035cff401520e0948b01dcaed8da2deed41149bea6b4a5ae1
frontend collection:
  full -> 97 files / 1,077 / 3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb
  focused -> 5 files / 46 / 5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127
typecheck: exit 0
i18n scanner:
  candidateCount 36 / signatureCount 20 / debtSignatureCount 0 /
  allowlistCount 20
post-cutover census:
  128 rows / a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64
  44 historical_reference / 6 low_level_empty_compatibility /
  3 retired_current_consumer / 22 rewired_current_consumer /
  47 test_fixture_reference / 6 unrelated_lexical_hit / 0 unknown
current-copy owners:
  four current data/formula/workbench documents; two training data-prep files;
  registry, Anthropic, OpenAI, analysis-tool, schema, and FileBackend copy;
  Dashboard, Ticker Detail, Settings storage, API source type, both locales,
  and their exact owning tests
implementation commit:
  5229b402c1606c5492c4fddec4b46188b510e348
```

One RED collection command used Vitest's optional `--json [path]` form with a
focused test path as the next argument. Vitest therefore wrote JSON over the
new untracked `Dashboard.test.tsx`. No product or tracked file was affected.
The test was reconstructed from the reviewed contract, the intended
`37 passed / 9 failed` RED was reproduced, and all four final collection
streams above were then rebuilt byte-for-byte. Subsequent commands used the
unambiguous `--json=<output-path>` form; the invalid scratch outputs remain
outside the repository and are not evidence.

Task 5 does not add fundamentals ingestion, scheduling, provider selection, or
Financial Datasets spend policy. It exposes the current stored-SEC projection
and completed-session valuation contract without claiming those separate
capabilities are complete.

## 5. Mutation Evidence

Every row requires an exact diff artifact, owning result, pre/post product SHA,
and restored GREEN.

| Mutation | Owning node/set | Expected RED | Diff SHA | Result | Restored SHA |
|---|---|---|---|---|---|
| M1 older-day fallback | `test_missing_required_date_does_not_fallback_to_older_bar` | yes | `62ca09dc8b7c876efcdeaf1d81c6868086fb759f3277d3693ea69e9416c2d2d2` | RED on stale row becoming available; restored GREEN | `f0dcb02ee0826bda533cd09d2308b24b4739bd6b1204a5084453d3b9f63ac8be` |
| M2 26-slot rule | `test_one_row_qualifies_without_slot_completeness` | yes | `8b008f1b91a9e8e002b7b8d90dc92b2a4af5bfe5778f65625bceb753e8dccb40` | RED on one-row day becoming unavailable; restored GREEN | `f0dcb02ee0826bda533cd09d2308b24b4739bd6b1204a5084453d3b9f63ac8be` |
| M3 raw UTC date | `test_et_market_date_not_raw_utc_date_owns_selection` | yes | `f47dd33f286a5d4716687e711c1a436fb9bf7b82f60ab2b28e2d6cb5971eb0e0` | RED on the 999 UTC-date sentinel; restored GREEN | `f0dcb02ee0826bda533cd09d2308b24b4739bd6b1204a5084453d3b9f63ac8be` |
| M4 old cache key | `test_old_metrics_cache_key_is_ignored` | yes | `fb4faa547588dbb918b2981206b0c78766aebdb0426eea6d4ade574a1b99a7ed` | RED on the extra old-key read; restored GREEN | `a4e8a10db1db889efca4b38edafda57da0602a1d762e5fa9fa0227306c6ac01e` |
| M5 dynamic cache payload | `test_v2_static_cache_excludes_price_and_dynamic_fields` | yes | `4ef0f7a615fe158daf5f131881f0f1b309e16fdd96997c9dbab4c7949913c134` | RED because contaminated payload was rejected and not written; restored GREEN | `a4e8a10db1db889efca4b38edafda57da0602a1d762e5fa9fa0227306c6ac01e` |
| M6 `1e6` unit error | `test_explicit_price_uses_base_unit_shares_without_million_scaling` | yes | `0877880c9473e4d05d24dae0c90d7ea783ca3312fb062ad9b973cf25a0ff4e1c` | RED on million-scaled valuation outputs; restored GREEN | `5faac9ea76f5ac74d91a69e2d161a9ace4d4dc25e1e654dfb81ee8e78159dcb4` |
| M7 legacy snapshot override | `test_legacy_ibkr_snapshot_cannot_override_sec_or_price_basis` | yes | `601ff5e07a27981d198a6d4e4cce142fbd6a57750fcdc2286fa9ddd3b936d10d` | RED on legacy market cap 349866.1 overriding 20000000.0; restored GREEN | `a4e8a10db1db889efca4b38edafda57da0602a1d762e5fa9fa0227306c6ac01e` |
| M8 FileBackend CSV read | FileBackend owning node | yes | `b1b0226866f950d553c8602cc3be668affbb224f7e97bbafc66e5149182451e6` | RED at the unconditional retired-path sentinel; restored GREEN | `c923f08814dedd48f5ec04e052d072c6e30755357aadf8390deeb8b2359ddc2f` |
| M9 daily directory scan | daily-update owning node | yes | `aeed516bd4b26de4fc881912685a044b18581e38511eaaa83627aa8572c454d9` | RED at `data/prices` scan sentinel; restored GREEN | `d159ee2b07c33dbd531639d5c234967567dab0df4153ab880f454f6339bea36a` |
| M10 legacy fundamentals projection | stored-SEC owning set | yes | `59b53cdf5e3e52c2f0a0d62f199c2ccff2aa96ca78426d0e2cf89f85890e9cc1` | both owners RED on legacy coverage/count projection; restored `2 passed` | `9a05e95ec7552b74fb79bd38571df53e5ca0d8913344f11d71e46cec49b734ea` |

### 5.1 Task 6 stop before M8

The first seven cycles used independent mutation directories under
`/tmp/eir006-valuation-price-truth/mutations/`. Before M8, the owning node was
run unmodified as a GREEN precondition. It failed before reaching product code:

```text
tests/test_db_backend_retired_prices.py::
  test_file_backend_prices_and_fundamentals_are_empty_without_path_probes
AttributeError: 'FileBackend' object has no attribute '_prices_dir'
```

Grounded identities at the stop:

```text
test file:
  b231f655ff5526a7b769791170357497f5d8fc8a5ff9aed497be432c010a7a99
FileBackend product:
  c923f08814dedd48f5ec04e052d072c6e30755357aadf8390deeb8b2359ddc2f
all M1-M7 product files restored to HEAD: yes
ordinary worktree status before docs amendment: clean
M8-M10 executed: no
focused/protected/native admission executed: no
```

Task 5 deliberately removed the inert private path attributes; its approved
product behavior remains direct empty compatibility. The stale test fixture,
not FileBackend behavior, caused this wrong-RED. The bounded plan amendment
authorizes only the stronger unconditional post-construction no-probe guards.
No product, node identity, collection, census, provider, production data, or
deletion scope changes at this gate.

### 5.2 Post-review continuation and frontend-runtime stop

Focused review cleared the preceding amendment. The context-scoped no-probe
fixture is now:

```text
commit: 1c45e52e
test SHA:
fc32a4bba59ad921f3fa406dddde4856f66adc4381d282f0ffe6ebbdb0df42f2
backend full identity after fix: 4581 / 6e4994bb...
backend focused identity after fix: 335 / 58230b54...
post-cutover census after fix: 128 / a08e7f68...
```

M8-M10 then completed with the tabled exact artifacts and restored product
SHAs. Native-context verification completed:

```text
18-file backend focused: 333 passed / 2 skipped (335 total)
pinned protected set: 314 nodes / 31072af5426e14d52976dc702d2d5b9e3d8a3e55dd43f5974ae0fa1498d701f2
protected runtime: 313 passed / 1 skipped
```

Immediately before the focused gate, the isolated worktree's `data/` path was
absent. The focused/protected runs created exactly five files under a new
`data/` root: two hermetic Financial Datasets cache fixtures, two zero-byte lock
files, and one test profile database. Before this review handoff, every path
received type/inode/mode/size/mtime/SHA evidence and the entire newly created
root was moved reversibly by exact path to:

```text
quarantine:
  /tmp/eir006-valuation-price-truth/artifacts/task6-pre-amendment-data
manifest:
  /tmp/eir006-valuation-price-truth/artifacts/task6-pre-amendment-data-manifest.tsv
manifest bytes: 1004
manifest SHA-256:
  cc69d01ed2b5ffb7ef8944262c7fc0bd1659ebb2dc94a4d6964fef149fd94596
post-transaction worktree data/: absent
pre-existing file changed: no
```

The initial sandbox focused attempt stopped after 32 nodes with no further
progress and was interrupted with exit 130. It is not admission evidence. The
identical focused command completed in the already established native boundary
and is the result above.

The first full frontend runtime gate then produced exactly one deterministic
failure:

```text
97 files: 96 passed / 1 failed
1077 tests: 1076 passed / 1 failed
owner:
  src/settings/settingsRegistry.test.ts > settings workspace registry >
  keeps bilingual search metadata independent from rendered locale
failed query:
  zh-Hant / data_storage / 查看價格、基本面與交易日資料覆蓋。
standalone owner: 14 passed / 1 failed
JSON artifact: /tmp/eir006-valuation-price-truth/task6-frontend-search-red.json
bytes: 5835
SHA-256:
d77daef893c8aeb211e3bdea67008919d14301ae9145632132816fcffd6e360f
```

Direct code/history inspection proves the failure is a Task 5 search-continuity
regression. The current visible description was intentionally updated, but its
prior pre-I18N-2 full sentence was omitted from zh-Hant `searchAliases` even
though the frozen I18N-2 contract requires that sentence to remain searchable.
Task 5 executed the 46-node focused frontend set and reproduced the 1,077-node
full collection identity; it did not execute full Vitest, so the new runtime
failure does not contradict its recorded gates.

Stop-and-amend is applied before any product correction. The reviewed proposal
adds the exact old sentence only as a hidden search alias and restores the
similarly labeled `settingsCopy.test.ts` baseline row to its actual frozen
pre-I18N-2 values. Current visible copy, English copy, node IDs, collection
identities, backend bytes, and every M1-M10 artifact remain unchanged. No later
Task 6 gate has run.

## 6. Product Verification

### 6.1 Final collections

```text
backend full:
backend focused:
frontend full:
frontend focused:
exact backend delta:
exact frontend delta:
```

### 6.2 Focused and protected gates

```text
backend focused:
price collection truth:
current quote:
Financial Datasets policy/cache/client:
stored-only provider-free:
earnings doubles:
coverage v2:
scheduler price outcomes:
Tranche B protected bytes:
provider counters:
PG attempts:
```

### 6.3 Frontend gates

```text
full Vitest:
typecheck:
build:
i18n scanner:
```

### 6.4 Native canonical tip

```text
report path:
report SHA:
4581 collected == seen:
4509 passed / 72 skipped / 0 failed / 0 errors:
exit 0:
empty non-passing SHA:
artifact manifest/quarantine:
pre/post boundary equality:
```

## 7. Independent Implementation Review

```text
reviewed range:
review verdict:
reviewer collection reconstruction:
reviewer mutation reconstruction:
reviewer native control:
findings/fixes:
exact cleared tip:
```

## 8. Merge And Read-Only Rollout

```text
pre-merge master:
reviewed product tip:
ff-only proof:
merged master:
merged backend/frontend collections:
merged canonical admission:
read-only qualified selector observation:
fixture unavailable observation:
old-cache-ignore fixture witness:
daily-update/SQLite equality:
stored-SEC projection equality:
production writes:
provider calls:
EIR006_PRODUCT_CUTOVER_TIP:
```

Physical old data remains present at this checkpoint. EIR-006 remains
`promoted`; deletion is not authorized by product merge.

## 9. Fresh Deletion Manifest

> Blocked until Section 8 is merged and reviewed. Manifest construction is
> read-only.

```text
manifest commit:
manifest SHA:
comparison implementation SHA:
alias input SHA:
exact file rows/count/SHA:
raw-view summary:
canonical-view summary:
canonical keys absent from SQLite:
exact cache keys/count:
exact fundamentals rows/count:
exact sync rows/count:
consumer census SHA:
writer/process census:
saved scheduler state:
DB identity:
independent manifest review:
bounded exact-source amendment:
separate user approval:
```

## 10. Physical Closeout

> Blocked until the exact manifest and destructive amendment receive
> independent review plus separate user approval.

```text
approval reference:
quiesced writer proof:
file quarantine proof:
DB row snapshot proof:
transaction affected rows:
pre-final verification:
rollback or success disposition:
temporary rollback assets removed:
scheduler state restored:
canonical admission:
read-only production truth:
closeout review:
EIR closure commit:
```

## 11. Honesty Ledger

- Plan review does not authorize product implementation.
- Product implementation review does not authorize merge.
- Product merge does not authorize physical deletion.
- Read-only manifest construction does not authorize physical deletion.
- Physical deletion requires a reviewed exact-source amendment and separate
  user approval of the exact manifest.
- A partial transcript, focused suite, sandbox-incompatible run, or stale data
  count is never promoted to canonical evidence.
- No provider, entitlement, halt, no-trade, or volume cause is inferred from
  `no_qualified_price`.
- Historical documents are not rewritten to make the current tree look
  cleaner.
