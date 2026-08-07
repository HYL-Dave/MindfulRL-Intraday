# EIR-006 Valuation Price Truth Evidence

> **Status:** TASK 9 EXECUTED; CLOSEOUT REVIEW READY
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

### 5.3 Search correction and protected-hash stop

The bounded Settings correction was implemented in commit `b2b05f65`. Before
the product correction, both frozen owners failed on the exact prior zh-Hant
sentence:

```text
artifact:
  /tmp/eir006-valuation-price-truth/task6-frontend-search-dual-red.json
bytes: 10214
SHA-256:
  a33188be0941868a0a9d6bc170780f5649a8987cbb14a9c28716d5a90884f1d7
RED: 23 passed / 2 failed
post-fix owners: 25 passed
```

The old sentence is now a hidden alias only; current visible and English copy
did not revert. Subsequent frontend gates completed:

```text
full Vitest: 97 files / 1077 passed
full identity:
  1077 / 3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb
focused identity:
  46 / 5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127
typecheck: exit 0
build: exit 0
i18n scanner: 36 / 20 / 0 / 20
```

The build repeated only the existing non-blocking `>500 kB` chunk warning
(`914.36 kB`, gzip `271.97 kB`).

The next static protected-byte gate compared all 31 Task 0 rows. Thirty still
matched exactly. The only mismatch was `src/market_data_direct.py`:

```text
Task 0:
  blob 28605175a5ec6dac24642a1dc701dd8ea65e02cc
  SHA-256 8d6ee8ab36e9d7a1185b2aeb66aa5376f5205981bc562b53acefc5b7ec2549f4
  bytes 41111
current / reviewed Task 1 post-extraction:
  blob b516fcf6d55ffe8b2af76091bb1cebafdbdca458
  SHA-256 412865670c39feff84d9d2216dfd2a727ecc505be250c8de6d7bd3c0fb3ea735
  bytes 39517
```

Commit `33a628b0` intentionally moved the completed-session calendar authority
to `src/market_sessions.py`; Task 1 evidence pins the current SHA and proves all
70 market-data-direct IDs remained unchanged and GREEN. This is therefore a
stale Task 0 protection manifest, not later product drift.

The original Task 0 manifest is retained unchanged. A Task 6 adjusted manifest
was constructed by replacing exactly that one row:

```text
path:
  /tmp/eir006-valuation-price-truth/protected/task6-protected-after-task1.tsv
rows: 31
SHA-256:
  5408aabaf89661c429e17c5e68f3db4b4a6dd945a57a1125384f47fb6017e609
diff from Task 0 manifest: exactly src/market_data_direct.py
current tuple verification: 31 / 31
```

Stop-and-amend was applied before consumer-census reconstruction,
production-state comparison, or native canonical admission. No product/test
change is authorized at this stop.

## 6. Product Verification

### 6.1 Final collections

```text
implementation execution range: e261abc2..4d8acead
last product/test commit: b2b05f65
backend full:
  4581 / 6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f
backend focused:
  335 / 58230b548925b29035cff401520e0948b01dcaed8da2deed41149bea6b4a5ae1
frontend full:
  97 files / 1077 / 3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb
frontend focused:
  5 files / 46 / 5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127
exact backend delta: +29 / -1
exact frontend delta: +1 / -0
pre-existing node identity loss beyond the one reviewed removal: none
```

### 6.2 Focused and protected gates

```text
backend focused: 333 passed / 2 skipped (335 total)
protected collection:
  314 / 31072af5426e14d52976dc702d2d5b9e3d8a3e55dd43f5974ae0fa1498d701f2
protected runtime: 313 passed / 1 skipped
price collection truth: included in protected set; GREEN
current quote: included in protected set; GREEN and Task 0 bytes preserved
Financial Datasets policy/cache/client: focused/protected GREEN; bytes preserved
stored-only provider-free: shared projection owners GREEN
earnings doubles: GREEN; no live provider seam
coverage v2: protected GREEN
scheduler price outcomes: protected GREEN; Task 0 bytes preserved
Tranche B protected bytes: Task 0 bytes preserved
adjusted protected source manifest:
  31 / 5408aabaf89661c429e17c5e68f3db4b4a6dd945a57a1125384f47fb6017e609
  31/31 current path/blob/SHA/size tuples verified
provider counters: 0 external requests in EIR-006 owners; all provider seams are doubles
PG attempts: 0 from price/fundamentals projection owners; PG sentinels remained unhit
```

### 6.3 Frontend gates

```text
dual-owner RED artifact:
  23 passed / 2 failed
  a33188be0941868a0a9d6bc170780f5649a8987cbb14a9c28716d5a90884f1d7
post-fix owners: 25 passed
full Vitest: 97 files / 1077 passed
typecheck: exit 0
build: exit 0
build warning: existing >500 kB chunk; 914.36 kB / gzip 271.97 kB
i18n scanner: 36 candidates / 20 signatures / 0 debt / 20 allowlisted
```

### 6.4 Native canonical tip

```text
fresh worktree: /tmp/arkscope-eir006-task6-native-tip-4d8acead
stage: eir006-task6-native-tip-4d8acead
report path:
  /tmp/eir002-green-baseline/reports/eir006-task6-native-tip-4d8acead.json
report SHA:
  14550e6ab8661816d3d30d369bcfb77121ec9a8f4afe54def97b1d29516e8375
transcript SHA:
  1a58178bb249a111cce25637521b53c9b3ba0c367068e9705bff578bb3d57a7a
4581 collected == seen: yes
collected/seen stream SHA:
  6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f
4509 passed / 72 skipped / 0 failed / 0 errors: yes
warnings: 18 existing collection/deprecation/return-value warnings
duration: 253.86 seconds (0:04:13)
exit 0: yes
empty non-passing SHA:
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
wakeup probe: {"callback_fired":true,"ready_count":0,"wake_bytes":0}
generated artifacts:
  581 / a487f4e8a3c3e83c79b1b4632a26842c590173c31625e4b74fa54b4fb07c4965
quarantine:
  /tmp/eir006-valuation-price-truth/artifacts/task6-native-tip-quarantine
pre/post boundary equality:
  tracked diff empty; data/ empty; src/data absent; only pinned node_modules link remains
```

The fresh worktree had no `config/.env`; the unchanged native wrapper used
`env -i`, fixture database overrides, `ARKSCOPE_DISABLE_SCHEDULER=1`, and no
provider credential. The structured reporter independently reproduced the
preconstructed final node stream. The terminal pass/skip total comes from the
complete transcript, not from a partial console sample.

### 6.5 Census and production no-write boundary

```text
post-cutover consumer census:
  128 / a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64
  byte-identical to Task 5
  44 historical / 6 low-level compatibility / 3 retired current /
  22 rewired current / 47 test fixture / 6 unrelated / 0 unknown
behavior propagation:
  4 / 613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba
  byte-identical to Task 0
consumer owner node:
  test_current_runtime_consumer_census_is_closed_and_exact -> 1 passed
```

The first external census replay incorrectly fed the three self-referential
EIR-006 authority files into the dated Task 0 classifier; it stopped on the
current evidence path. A second replay applied that authority exclusion but
still used the dated classifier and stopped on the newer
`tests/test_daily_update_wrapper.py` owner. Neither attempt is admitted. The
admitted stream calls the current closed owner test's structured
`_discover_consumers()` and fail-closed `_verdict()` functions, including its
exact `_EIR006_AUTHORITIES` exclusion, then reproduces the Task 5 stream
byte-for-byte.

Production reads used SQLite URI `mode=ro` plus `PRAGMA query_only=ON`:

```text
market_data.db pre/post SHA:
  7516178876c7f8bbeb69007cf7464bf22362fe280e073157736d58d67258e6e2
market_data.db inode/size:
  127284871 / 3428466688 bytes
retired file metadata pre/post:
  301 paths / 3f353d120b5054dd43406ea85694b5c3f804dc4cd4559398134d307de957b344
old cache / fundamentals / retired sync counts: 19 / 130 / 1
market counts/frontier, schedule settings, scheduler state: byte-identical
normalized snapshot pre/post SHA:
  b42ec9d123d9cb9507c2dd907b95f7fd9d8cc7b538ce10cbb272b6c4c0f8fece
old-data movement/deletion: none
```

`profile_state.db` changed concurrently, from SHA `91995f27...` to
`1bd24a9e...`, while retaining inode and size. Read-only attribution from the
pre-snapshot cutoff found exactly two recent rows, job IDs `18851` and `18852`,
both successful `sa_market_news_refresh` runs with
`trigger_source=extension`; no other time-aware table had a row in the window.
The fresh native wrapper pointed `ARKSCOPE_PROFILE_DB` at its isolated runtime
database. These external extension writes are preserved and are not presented
as EIR-006 activity or rolled back. Attribution artifact SHA:
`93b43f7f69351447d3ab3bce8d99e11b0989ef28faa71fb985aa0e427df64a79`.

## 7. Independent Implementation Review

```text
reviewed range: e261abc2..ce88f72d
review verdict: GREEN, zero findings
reviewer collection reconstruction:
  backend 4581/6e4994bb... and focused 335/58230b54...
  frontend 97 files/1077/3f5e9f5b... and focused 5 files/46/5d64841c...
reviewer mutation reconstruction:
  M1-M10 each killed its owning contract and restored exact product SHA/GREEN
reviewer native control:
  fresh detached exact-tip worktree
  4581 collected == 4581 seen
  4509 passed / 72 skipped / 0 failed / exit 0 in 249.27 seconds
  collected stream == reviewed target == implementation report
findings/fixes: none
exact cleared tip: ce88f72d9f9d710903533505371789d18cce953e
```

## 8. Merge And Read-Only Rollout

```text
pre-merge master: fd6d1b86383df2a98f97b235d9796d4bcaaa7a58
pre/post origin/master: fd6d1b86383df2a98f97b235d9796d4bcaaa7a58 (not pushed)
reviewed product tip: ce88f72d9f9d710903533505371789d18cce953e
ff-only proof: fd6d1b86 is an ancestor; 22 linear commits; zero merge commits
merged master: ce88f72d9f9d710903533505371789d18cce953e
merged backend collections:
  full 4581/6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f
  focused 335/58230b548925b29035cff401520e0948b01dcaed8da2deed41149bea6b4a5ae1
  focused runtime 333 passed / 2 skipped
  protected runtime 313 passed / 1 skipped
merged frontend collections:
  full 97 files/1077/3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb
  focused 5 files/46/5d64841ccdd943eb81f1cea50870115ed60dffe57ff6fc9867179552a4a7f127
merged frontend runtime:
  97 files / 1077 passed; typecheck/build/scanner exit 0
  existing build warning 914.36 kB / gzip 271.97 kB
  scanner 36 candidates / 20 signatures / 0 debt / 20 allowlisted
merged canonical admission:
  report 14550e6ab8661816d3d30d369bcfb77121ec9a8f4afe54def97b1d29516e8375
  transcript 50bb1b960690344e6df9d6e1331cb185277db286524c4fd16d3602a257f2a20b
  4581 collected == seen; 4509 passed / 72 skipped / 0 non-passing; exit 0
  collected/seen 6e4994bb...; non-passing e3b0c442...
merged artifact transactions:
  runtime gates 548/2581297515ee29bbe7bf3215d9373edd9220002a2bede0d583c2142b04147d0a
  native 581/2eefa0f4278d89a8e3f0b4766f4e4fe2339b0385e1b08a4946ecedcb9c207437
  both worktrees restored to empty ordinary status, empty data, absent src/data,
  and only the pinned ignored node_modules symlink
read-only current selector observation:
  NVDA required 2026-08-03 while SQLite frontier was 2026-07-31
  typed no_qualified_price; no fallback to the older close
read-only frontier-aligned witness:
  NVDA available from local_market_db/15min for 2026-07-31
  timestamp 2026-07-31T19:45:00+00:00; observed price is not a test constant
fixture unavailable observation: missing and unreadable stores remained typed/no-create
old-cache-ignore fixture witness: metrics_*_annual_y2 remained ignored with provider doubles
rollout fixture report: 3 passed / dbf7f7c85061d1b8eab3bf9f0d8f72a1ae9508906b56834f70d124bfe665a3bf
daily-update/SQLite equality: exact prices row/ticker/latest mapping matched
stored-SEC projection equality:
  shared projection == local stats == coverage; 6 current tickers; NVDA present
  fundamentals sync is null in admin and coverage projections
physical old rows still present: 19 old cache / 130 fundamentals / 1 retired sync
production writes: zero EIR-006 writes; URI mode=ro and PRAGMA query_only witnesses
provider calls: zero; rollout called only local selectors/status/projections and fixture doubles
EIR006_PRODUCT_CUTOVER_TIP: ce88f72d9f9d710903533505371789d18cce953e
```

The first native artifact enumeration admitted only ignored files and therefore
listed 580 rows (`978465fc...`) while omitting the unignored
`src/data/cache/risk_free_rate.json`. Its non-equal restored boundary rejected
that attempt; an erroneous shell `PASS` line after failed `cmp` commands is not
admission evidence. The corrected `set -euo pipefail` transaction included the
missing path, produced the 581-row manifest above, quarantined every exact
path, and made all pre/restored comparisons pass.

Rollout artifacts are
`3f5e648ed5058950bd798785ac0c2d62c77c69ba8e90e49930e3bde955c4d61c`
for the current production observation,
`dd75ca7c3bcaf4e128844f50d306b409d8dfe76d44f3fc874194113fe9853803`
for the frontier-aligned selector witness, and
`785578c968ee0fae799da92effecccba1f8b17a745e1f1ee63a5f7b18fd253d9`
for the retired-row readback. The production scheduler remained active, so
this section claims only that every EIR-006 connection was read-only; it does
not claim all production files were globally frozen.

Physical old data remains present at this checkpoint. EIR-006 remains
`promoted`; deletion is not authorized by product merge.

## 9. Fresh Deletion Manifest

> Task 8 read-only construction is complete on branch
> `codex/eir-006-deletion-manifest`. Physical mutation remains blocked pending
> independent packet review and separate user approval of the exact authority.

```text
Task 8 base: 657b4aa2c8d67a6e659cba4d0d4c6cd90c8d36f3
product cutover tip: ce88f72d9f9d710903533505371789d18cce953e
packet root: docs/superpowers/evidence/2026-08-04-eir-006-deletion-manifest/
packet SHA256SUMS SHA: af5c090a4da72aa1204c6b8ffc13607b392e49d87acf38098a90f4bd2e24e4b6
authority ID: 6096b988428a94d053baddd18493eb29077bc627d725a95fd53f75c4755b0dce
comparison implementation: 669 lines / 26,509 bytes / e4acb819f6a32c05d7d756b1a9e106bba105e1c22d6cd273513d6bf27df2e759
alias input: 3 rows / 0a8fbbf845b73bab1740d04ffb77ab1e935884f417c2bece20395187f83d9220
exact files: 225 15min + 75 hourly + 1 summary / 842c3e08ff8ed9cb11c92033cf67ad5950d357cb8cd1e0662b74683ba554b0fc
raw view: 2,547,747 physical / 2,314,293 unique / 233,454 duplicate / 58 conflict / 161 DB differences
canonical view: 2,298,763 unique / 248,984 duplicate / 176 conflict / 43 DB differences (23 volume-only + 20 OHLC)
canonical keys absent from SQLite: 0
LC/HAPN overlap/conflict: 15,530 / 118
exact cache keys: 19 / a4a8d829eb08553a1223f5240de260955fc48a564f8232b943206e0bf88b39bd
exact fundamentals rows: 130 / 6b845506f9fce54ac4dba78ebd96bacc20113a7aefef651b877f62892418c219
exact sync rows: 1 / 5b3736ba19e66b2e427b149b143771fb5625eab426e8af7a6317c29461cd15ff
current cache keys selected for deletion: 0
consumer census: 128 / a08e7f683b426c10090f1cb7f6e4f4104f22678147a46639ceaece0bcb088c64
behavior propagation: 4 / 613024acc6568296cb798a2832fb8ca1e67fba05a9857c4e4bd5629755c556ba
controller: 1,110 lines / 41,423 bytes / cd8980e891b4fb8713d008762d7740fd9f91009a37f501d0ee993557bd9933af
rollback snapshot: 150 records / 875,857 bytes / 1e3578344dfcac0e445900358265c6606150007a496a71284d87e5ae5821697c
quarantine root: /mnt/md0/PycharmProjects/.arkscope-eir006-quarantine/6096b988428a94d053baddd18493eb29077bc627d725a95fd53f75c4755b0dce
saved price schedule: enabled=true / interval_minutes=720
observed owners: desktop 4041681 / Electron 4041708 / sidecar+scheduler 4041764
market DB identity: device 2304 / inode 127284871
profile DB identity: device 2304 / inode 127284276
independent manifest review: pending
separate user approval: pending
```

Both file/comparison runs, both DB-row runs, and both census runs produced
byte-identical fresh-root artifacts. The controller's read-only payload
snapshot also reproduced byte-identically. Its scratch probe copied only the
four relevant SQLite tables through a `mode=ro`/`query_only=ON` connection,
then proved exact 19/130/1 delete and restore counts plus file move/restore; it
never invoked the production mutation entry point.

Self-review found that the first controller treated every nonzero `lsof` result
as an empty holder set. The final pinned controller instead bounds each probe at
15 seconds and fail-closes stderr, timeout, ambiguous exit/output pairs, and
unknown exits. A separate scratch probe proved no-holder, active-holder, and
invalid-path behavior before the final packet identity was generated.

The preliminary census producer also failed when rerun against the completed
branch because the new packet truthfully contains retired-path strings. The
final pinned producer requires the exact ten matching packet files as
self-authority before invoking the unchanged product census owner. Two fresh
completed-worktree runs were byte-identical and retained the original
`128/a08e7f68...` census and `4/613024ac...` behavior ledger.

The owning test itself first failed on the new packet README, proving the
classification boundary was visible rather than hidden in the producer. A
bounded test-only amendment added those exact ten files to its authority set;
the existing two nodes then passed `2/2` with no ID change. The final controller
pins that file at `de6e192b...` while every other product/test path remains
cutover-tip exact.

A final read-only data recomputation retained every authority TSV byte-for-byte.
Only allowed live SQLite observations advanced: 15-minute prices grew from
2,406,398 to 2,410,324 rows and the frontier moved from 2026-07-31 to
2026-08-03; DB device/inode/size and all 301 old-file identities, 150 target
rows, aliases, canonical differences, and retained cache identities stayed
unchanged.

The dated operational census found the live sidecar as the only writable
`market_data.db` holder and found no holder of `data/prices/`. The admitted
static census and owner gates were `2 passed`, 113 projection-owner tests, and
six behavior-propagation owner tests. This means the sidecar must be stopped
for Task 9, not that an old-data writer remains.

No archived row payload is tracked. The complete rollback JSONL exists only as
a reproducible Task 9 quarantine artifact; tracked TSVs contain keys,
metadata, payload lengths, and payload hashes.

### 9.1 V2 Rebuild After Fail-Closed Rollback

The v1 manifest received independent review and the user separately approved
authority `6096b988...d0dce` on 2026-08-07. Its first execution stopped after
moving the 301 files and before any DB delete. The controller's own read-only
SQLite handle remained open because `with sqlite3.Connection` does not close
the connection; the post-move lsof gate therefore refused with
`exit=1, records=5`.

The execution receipt recorded `moved_before_failure=301` and
`restore_error=null`. Reviewed rollback then verified all files and 150 target
rows in their original locations, restored `0/0/0` rows, and emitted receipt
SHA `5a27b331c0279b00f5d43ac5d72547a6183bd60fc3d3285658d73105677cc454`.
The exact temporary snapshot retained its reviewed
`1e357834...` identity until the rollback was verified, then the whole v1
quarantine was destroyed. Desktop, sidecar, and all seven schedule settings were
restored; no v1 forward deletion remains.

The v2 controller makes `_connect_ro` a real context manager with an explicit
`finally: connection.close()`. A new probe first reproduced the old self-holder
RED, then proved no lsof record remained after either read context while still
proving exact `19/130/1` scratch delete/restore and file move/restore. All three
Task 8 producers were rerun after rollback: the 301-file manifest, aliases,
raw/canonical differences, 19/130/1 row manifests, 128-row census, and four-row
behavior ledger remain byte-identical.

```text
Task 8 v2 base: 4955e6249f7758b136f4526c02ceecaa726535f4
packet SHA256SUMS SHA: 7c887ae6908b1087003f0d2990adbf5757f672b63ae4525dcbb9461969ef60dd
authority ID: 4b1d9083ed054387cd00ae253ab055641fc18e55a7a4e718534fb25a23cf413e
controller: 1,114 lines / 41,538 bytes / 0ddb451f203a274ec08c5dbba79439971f2cd073e1ec8af2bb27398d974f5d2c
controller probe: 174 lines / 6,951 bytes / 9c3caca64841bbb39236d3d76fbf89f5b244f01f217d64c29a5e07bbee355bd4
quarantine root: /mnt/md0/PycharmProjects/.arkscope-eir006-quarantine/4b1d9083ed054387cd00ae253ab055641fc18e55a7a4e718534fb25a23cf413e
observed owners: desktop 2847946 / Electron 2848002 / sidecar+scheduler 2848089
independent v2 review: GREEN, zero findings
separate v2 user approval: granted with all three exact identities
```

The v1 approval was superseded and did not authorize V2. V2 passed those two
gates before its separately recorded execution below.

### 9.2 V3 Rebuild After The Post-Move Holder-Root Refusal

The user approved V2 authority `4b1d9083...e656c7`, packet SHA
`7c887ae6...60dd`, and controller SHA `0ddb451f...5d2c` on 2026-08-07.
Packet checksums and exact dated PIDs matched, the desktop owner stopped the
reviewed three-process family without SIGKILL, and native preflight passed.

V2 execution wrote the exact 150-record / 875,857-byte / `1e357834...`
snapshot and moved all 301 files. It then refused before opening the DB write
transaction. `_move_to_quarantine()` had removed the now-empty source
`data/prices` directories, but the second `_assert_quiesced()` still called
`lsof +D` on that absent source path. `lsof` returned its usage text on stderr,
which the controller correctly treated as a refusal. The failure receipt says
`moved_before_failure=301` and `restore_error=null`.

Automatic recovery returned all 301 files. A direct read-only check found all
19/130/1 rows still present. Reviewed rollback verified every file and all 150
rows, restored `0/0/0`, and emitted failure/rollback receipt SHAs
`923c2a7c13f014dee2198ce8a591c1290849b14b8c944df8face915019bb61a7` /
`e1a43aead8efc4e3c4a2601ab96300ee0c70649ca2f9efef159c6cb0f1bc4321`.
The snapshot retained its exact SHA until verification, then the exact V2
temporary root was destroyed. A final native V2 preflight passed. The desktop
was restarted normally; `/healthz` returned `ok`, DB identities matched, and
all seven schedule key/value pairs matched the saved authority. No V2 forward
deletion remains.

The V2 source was then replayed against a scratch post-move fixture and
reproduced the same usage-error RED. V3 requires an explicit nonempty tuple of
existing price trees for every quiescence check. Preflight uses the source
tree; execution after movement and pre-restart verify use the quarantine tree;
rollback checks every existing source/quarantine tree and refuses symlinks,
duplicates, missing roots, or holders. The V3 probe passed exact `19/130/1`
scratch delete/restore, source-to-quarantine-to-source file transport, and all
three holder phases while retaining the exact snapshot identity.

All three read-only producers were rerun at V3 base `25f061b7`. Their 301-file,
alias, raw/canonical difference, 19/130/1 row, 46-row classification, 128-row
census, and four-row behavior authority streams are byte-identical to V2.
Only dated result metadata and the current operational owner facts advanced.

```text
Task 8 v3 base: 25f061b7781cdc9f738a4858aa331dd10a3ef9d2
packet SHA256SUMS SHA: 99a813e8311a639af3a45b9d1e6f37b0a97a6e40b620fcb92f42a6e96b18bd22
authority ID: 9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7
controller: 1,138 lines / 42,740 bytes / 891edbe1fe0c8005f609fee2ed97403180f3498da53668da6175645c97214d37
controller probe: 189 lines / 7,688 bytes / e200d63b951fa7b44e8b9e49a3b0b81207a025923c69d827b90b7d8afe2ee981
quarantine root: /mnt/md0/PycharmProjects/.arkscope-eir006-quarantine/9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7
observed owners: desktop 2887595 / Electron 2887650 / sidecar+scheduler 2887713
independent v3 review: GREEN on 25f061b7..27191f05 except one docs-only ordering pin;
  focused GREEN on 27191f05..518ea76b after that pin
separate v3 user approval: received 2026-08-08 by uniquely named reviewed V3
  prefixes; execution expanded and enforced all three complete identities
```

V1 and V2 approvals are superseded. The V3 review and focused re-review both
cleared before the separate user approval and execution recorded below.

## 10. Physical Closeout

> V1 and V2 rolled back without a DB delete. V3 received independent review,
> separate user approval, and completed the exact forward path on 2026-08-08.

```text
approval reference:
  v1 6096b988...d0dce and v2 4b1d9083...e656c7 superseded after fail-closed rollbacks
  v3 user wording named unique reviewed prefixes 9bfb3f2a... / 99a813e8... / 891edbe1...
  controller CLI and environment both used complete authority 9bfb3f2a3e377752d3105c07cf55aceb986ea094314dea8616763046a5e656c7
merged execution tip: 518ea76b94bba67a2c9b6e091fa5ef7bf3a80e5e
packet SHA256SUMS: 99a813e8311a639af3a45b9d1e6f37b0a97a6e40b620fcb92f42a6e96b18bd22; 22/22 OK
controller: 891edbe1fe0c8005f609fee2ed97403180f3498da53668da6175645c97214d37
quiesced writer proof:
  exact desktop/sidecar /proc commands matched dated owners 2887595/2887713
  SIGTERM to desktop owner retired desktop 2887595, Electron 2887650, and sidecar 2887713 within five seconds; no SIGKILL
preflight: authority 9bfb3f2a... / status preflight_pass / exit 0
file quarantine proof:
  execute exit 0 after exact 301-path same-filesystem movement
DB row snapshot proof:
  exact 150-record / 875,857-byte / 1e357834... temporary snapshot existed through post-restart
transaction affected rows: exact 19 old cache / 130 legacy fundamentals / 1 retired sync
pre-final verification: authority 9bfb3f2a... / status verified_deleted / exit 0
runtime restoration:
  desktop 2928518 / Electron 2928573 / sidecar 2928635
  healthz 200 OK at 127.0.0.1:40163
post-restart verification: authority 9bfb3f2a... / status runtime_restored / exit 0
rollback or success disposition: forward success; rollback not invoked
temporary rollback assets removed:
  exact non-symlink authority root removed only after post-restart; root absent; parent has no remaining child
final exact readback:
  225 15-minute files + 75 hourly files + one summary absent
  19 old cache keys + 130 legacy fundamentals IDs + one retired sync key absent
  fundamentals table row count 0
  18 current SEC-v1 + 9 other retained cache rows match every field and payload SHA
  DB device/inodes, ticker aliases, and all seven schedule key/value pairs match
canonical admission:
  product cutover remains 4581 seen / 4509 passed / 72 skipped / 0 non-passing
  Task 9 changed no tracked product/test bytes; the reviewed Task 8 census owner remains 2 passed
read-only production truth:
  GET /market-data/status returned 200, fundamentals projection 6 rows/6 tickers,
  financial_cache 27 rows, fundamentals sync null, and prices authority local
durable rollback archive: none
closeout review: pending
EIR closure commit: pending independent closeout review
```

The user's direct approval repeated the three unique V3 prefixes from the
immediately preceding full-identity review. It did not retype all 64 hex
characters. The executor did not silently substitute a different packet: the
packet check returned `22/22`, and the controller required the complete
authority value independently in both its CLI token and environment variable.

One additional pre-stop check incorrectly assumed Electron stored `electron .`
as two argv entries. It failed before any signal or mutation. Direct `/proc`
inspection showed the reviewed process stored that text as one argv entry; the
plan's exact desktop and sidecar checks both passed, and only those reviewed
checks were used for admission.

The successful `execution.json` was temporary by design and lived inside the
approved quarantine root. It was intentionally destroyed with the rollback
snapshot after successful `post-restart`; no durable receipt SHA is claimed.
The controller's four exit-zero states plus the independent exact-path and
exact-row readback above are the closeout admission evidence.

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
