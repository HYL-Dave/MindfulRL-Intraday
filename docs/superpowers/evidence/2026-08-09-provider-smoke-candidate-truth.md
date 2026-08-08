# Provider Evaluation Hygiene and Tiingo Tail Retirement Evidence

> **Status:** TASK 2 COMPLETE - INDEPENDENT TASK 2 REVIEW REQUIRED;
> TASK 3 IS NOT AUTHORIZED
>
> **Date:** 2026-08-09
> **Task 0 tip before evidence:** `a9c702626aed82e2f08700aefa05ae2495408685`
> **Product grounding base:** `6159fc14956800dc04c4d6c944a2941b9c6c12db`
> **Single-use artifact root:** `/tmp/provider-smoke-hygiene-task0-a9c70262`
> **Artifact manifest:** `44` entries / SHA-256
> `49f5038762337aeefcdf10f5f43baf5528515f1deb6f8a405d0887af9d6b157c`

Task 0 was evidence-only. It did not delete, move, edit, import as a runtime
provider, or execute any January evaluation path, rate-fetch path, Tiingo path,
or manual provider smoke. Product, test, config, data, credentials, scheduler,
and provider state remain unchanged.

## 1. Authority and boundary identities

The isolated worktree was clean on branch `codex/provider-smoke-hygiene` at
`a9c70262`. The unlocked main worktree was clean at `6159fc14`. Commit
`db900ab8` is an ancestor of the Task 0 tip. The only paths changed from the
product grounding base were the reviewed design, plan, and priority map.

The three design identities were reproduced independently:

| Design state | SHA-256 |
|---|---|
| independently reviewed design at `db900ab8` | `729a8e028b06fff3bbbf3533eb7a32cd98aa56bf8c07004eab4b1c6902bdc493` |
| plan-gate clarification at `cff928e5` | `a6aff86e77d697a04ef3e323b6948126afe12cce0562f9219d4e43b7152e0d50` |
| unlocked-census amendment at `a9c70262` | `93008b2fd05542db872486bac8180aec6573c2fbadf5516e37581765f1092593` |

Pinned execution/toolchain assets all matched:

| Asset | SHA-256 / version |
|---|---|
| `/tmp/arkscope_asyncio_wakeup_probe.py` | `10647c1e64c49fc2e082701d7a735e40782620314c125cd103a9a3f9bb37bc2e` |
| deterministic reporter | `09d2bc52c7706b49e5f363fa2c6bcfc93523038f1c805fef08bb98a409301928` |
| native wrapper | `e7c963f1bc97125b70b435fb3c41bf4e59d501f0da561f2ef6d921c12083c84f` |
| `package-lock.json` | `5322cb03099b873066b7572c02db68a427ddc7509fdc9850cf9d8d3948a19f2c` |
| `node_modules/.package-lock.json` | `4dd5182f8111b54dd608344c45513f3b310f39321a3cc9832b60cefc2fa241ff` |
| Node | `v22.14.0` |

The three git-crypt tracked blobs were equal between the isolated tip and the
unlocked main tree before plaintext was read. Their blob IDs are recorded in
`encrypted-blobs.tsv`; none changed during Task 0.

## 2. Collection and retirement ledger

The sole full-suite operation was collect-only:

```bash
/tmp/eir002-green-baseline/run_native.sh \
  provider-hygiene-task0-a9c70262-collect --collect-only
```

The wrapper's native wakeup preflight passed. Reporter JSON SHA-256 is
`cd82864fb24cd088e1a4e863a42cf9084def3ca0c5a87a0baac82b9b420a797e`;
the transcript SHA-256 is
`d8fa338152ff34d17751bca9fc3c4844dcf0789f234de6af6a9b8e7743617aa5`.
The report ended with exit status `0`, `4607` collected IDs, `0` seen IDs, and
`0` non-passing IDs. Therefore no test body or provider path ran.

All streams were reconstructed from the reporter JSON and the exact
`a8970e64` path family, without parsing transcript prose:

| Stream | Count | SHA-256 |
|---|---:|---|
| canonical base | 4607 | `5180502f7dfe577ca758db5fb8ebdfe9ca282730a8976adcf65b7ab19c1c2d74` |
| seen | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| January paths | 15 | `2fff01e35f26d25c22ece520b491110b0066cf6fdccf31506fadab9f34fb30f2` |
| January nodes | 46 | `bb23dd1d6c5415cf5043767bb72c1137dd6d6b897e513984a6c3e590d4645e4a` |
| assertion-invalid Tiingo/yfinance nodes | 13 | `3ead0303136ab8742fae3fa15f916b2febda6088deb4f883c9b025db7d577016` |
| whole-file skipped nodes | 33 | `03f70d140bd4f2990674926904817667ed5f7ef28d10e4afbd71ea36f40aca58` |
| zero-node January paths | 6 | `9c2780536d4c05c9e40f4bbcab1583fe377565aa6663ba07055c2fcdf556f008` |
| rate-curve nodes | 34 | `3d51583972d7d5172fc5cf53be569469bfe192800a07d097ef9f81cd7a32ad21` |
| complete retirement node set | 80 | `a069a5af63bfcb3c6d63ddb4a25ca63bc897f97adde3ef159b83aef7b7be6fb8` |
| projected post-January collection | 4561 | `dd127ce5dd34249a364b6a7965517aac66492b3d044ea8cc21e79a9706e58620` |
| projected final collection | 4527 | `4eeb117804ad874c83ffe4c04fd25ecd4de4f460801bfbf95d15c1406f32455d` |

The complete machine ledger is `collection-ledger.tsv`, SHA-256
`3d7bc7ef5a917193270f94bda2ddb1f9964e9ade67453ad6186da3c0964924f9`.

## 3. Dual reference census

The census used two inputs exactly as reviewed:

1. plaintext, non-encrypted tracked files at the isolated implementation tip;
2. each git-crypt path read from the unlocked main tree only after its tracked
   blob matched the isolated tip and its plaintext was proven not to be the
   git-crypt ciphertext envelope.

All three encrypted paths were enumerated. Two contained Tiingo references:
`data_sources/DATA_SOURCES_EVALUATION.md` and
`data_sources/PAID_SUBSCRIPTION_EVALUATION.md`. The priority map was included in
complete discovery, classified `slice_decision_log`, and only then excluded from
the terminal external-capability projection.

| Census stream | Count | SHA-256 |
|---|---:|---|
| Tiingo discovery paths | 27 | `bb78255d82beddcfa5084159ccfb86d204d89d05e7a4930a90d94837a8c71ba4` |
| Tiingo dispositions | 27 | `7bd0928bd49e040ffff4d7f3653e2610435ebf63c893b8d76169db2e7d30bcf9` |
| Tiingo terminal projection | 13 | `0ba6820c5b4ce2afdc26fdbb379ea6a44eb38f8c33954ed49b9a2a5b65c6c517` |
| evaluation-reference discovery | 13 | `76b31a5c05d6dbe0a2a75af7f2b6d8e61d89bf415a698c8aff1521901b5efde2` |
| evaluation-reference dispositions | 13 | `f9f830dd8941ecb2efb1800cf408b1a477af9f137f1c50223aa763f5d09b602f` |
| evaluation terminal history | 10 | `0625d1220d4f94110ca84c93bfa951fc4e69fa00f7ccf76ee9004184772d160c` |

The expanded command trace is `census-command.trace`, SHA-256
`526914154d5409f3d89a4a71e807e842a7e8a63a78d0044f9e23555e57738d15`.
The compact `census-ledger.tsv` SHA-256 is
`7e5c123d96312f54ff9e85596a72313bb7512c6dcb461416a915919f53556058`.

## 4. Protected-owner manifest

The reviewed 30-row protected manifest was extracted from the plan and compared
to independently measured worktree SHA-256 and byte counts. Every row matched;
the expected and actual streams are byte-identical:

```text
30 rows
4ca66f0b373031fa64c73c537575a8d9fc25bba4fec663522cca59ed766b2fd2
```

No pin was updated to accommodate drift. There was no drift.

## 5. Structural RED and known docs-tip census RED

Structural RED was established before any implementation:

- all fifteen `a8970e64` evaluation paths exist;
- all nine retired option-rate/cache symbols and both rate files exist;
- the Tiingo adapter, diagnostic, export, enum, registry, environment template,
  and profile fallback exist;
- `tests/live/smoke_yfinance.py` is absent; and
- current authorities still contain the reviewed stale Tiingo/tool/training
  claims.

The retained 206-node owner set ran under the native wrapper with only a newly
created empty `data/` marker. It completed as:

```text
206 collected / 206 seen
205 passed / 1 failed
sole non-passing node:
tests/test_eir006_retired_data_boundaries.py::test_current_runtime_consumer_census_is_closed_and_exact
exact assertion:
unclassified or multiply classified consumer:
docs/superpowers/plans/2026-08-09-provider-smoke-candidate-truth.md -> []
```

Reporter SHA-256 is
`ac638402a50302d9754ec52c6e4be9182fb028f4636093dcda804dfe7a394953`;
transcript SHA-256 is
`b8a387ce4e3139a383c6d0c805452a07c954270927a36c8233e110cdb3c49a82`.
This is the exact reviewed structural RED, not a product/runtime regression.
`data/` remained empty and was removed. `src/data` remained absent.

## 6. No-provider command log

Commands admitted by Task 0 were limited to:

1. Git/blob/hash/static source inspection;
2. one deterministic `--collect-only` invocation;
3. one native focused invocation of the twelve retained-provider owner files;
4. static `rg`, `awk`, `jq`, `comm`, and SHA reconstruction; and
5. read-only plaintext inspection of the three verified git-crypt paths.

The focused files use fake/local transports and were the reviewed retained owner
set; none of the fifteen January paths, `tests/test_rate_curve.py`, Tiingo
diagnostic, or future manual smoke was an argument. The wrapper used `env -i` and
received no provider credential. `seen=0` proves collect-only executed no test
body. No comparison directory, cookie, timezone cache, yfinance artifact,
`data/` file, or `src/data` path was created. Both repositories were clean after
the runs.

## 7. Task reviews, native admission, and merge

- Design review: GREEN at `db900ab8`.
- Plan amendment focused review: GREEN at `a9c70262`.
- Task 0 evidence review: GREEN at `8c47e994`.
- Task 1 implementation commit: `5168603e39df77f35e2849177e92954faba944eb`.
- Task 1 evidence review: GREEN at `ec140ae1`.
- Task 2 implementation commit: `7f68f1fdd71225e889eb01e74622420fb288ba64`.
- Task 2 evidence review: pending.
- Tiingo product/config deletion: not started or authorized.
- Final native `4488 passed / 39 skipped / 0 failed` admission: not run; it is a
  post-cutover Task 4 gate and cannot be substituted by Task 0 collect-only.
- Merge: not attempted or authorized.

No partial or projected stream in this document is called a passing runtime.

## 8. Task 1 - complete January-family retirement

Task 1 used the single-use root
`/tmp/provider-smoke-hygiene-task1-8c47e994`. Its recursive `SHA256SUMS`
contains `86` entries and has SHA-256
`28bf3b582ea59d9c32ab37b6e32e6c24b188ff1d6bf46e59771cd7426e0ffc87`.

### 8.1 Exact deletion ledger

The deletion input was reconstructed directly from commit `a8970e64` before any
edit:

```text
15 paths
2fff01e35f26d25c22ece520b491110b0066cf6fdccf31506fadab9f34fb30f2
```

`git rm` removed exactly those paths. No content was moved to `tests/live`, no
skip/xfail replacement was added, and no sixteenth path was deleted. Immediately
after `git rm`, an informational command inspected the unstaged diff and printed
zero because `git rm` had already staged the deletions. That output was rejected;
the staged deletion stream was then reconstructed, compared byte-for-byte with
the pre-edit stream, and admitted only after reproducing the exact count and hash
above.

Implementation commit `5168603e` contains exactly `19` paths: the `15` deletions,
new `tests/live/smoke_yfinance.py`, and the three reviewed owner updates. It is
`+167/-5587` and contains no current provider implementation change.

### 8.2 Manual yfinance contract

The replacement is a newly written operator CLI rather than moved January code:

```text
tests/live/smoke_yfinance.py
157 lines / 5,049 bytes
SHA-256 2bf507bfe657613c344ae71f195f370edad14f887fa884bd965dc58998a396cd
```

AST verification proved:

- no module-level yfinance import and no pytest-shaped function;
- `parse_args` precedes the in-function yfinance import, which precedes
  `yf.download`;
- period and interval values are closed choices;
- required fields are exactly Open, High, Low, Close, and Volume; and
- response validation requires a non-empty table, parseable strictly ordered
  timestamps, one unambiguous value per field, and a finite latest close.

`--help` passed with a sentinel `yfinance.py` that raises on import. Missing
ticker, invalid ticker, and invalid period each returned exit `2` under the same
sentinel, proving those paths do not import yfinance or make a request. Compile
output was redirected outside the repository. No valid ticker execution occurred.

The live README now records the exact invocation, manual/public-network boundary,
and current Yahoo/yfinance terms-and-limits prerequisite. The CLI suppresses
provider stdout/stderr, never prints raw exceptions, cookies, headers, or cache
contents, and returns non-zero for import, request, empty, malformed, or invalid
responses.

### 8.3 Current references and census ownership

`docs/data/IBKR_NEWS_API_LIMITATIONS.md` now points Finnhub and Alpha Vantage
verification at surviving product adapters rather than deleted January scripts.
The EIR-006 census owner:

- removed `tests/test_ibkr_fundamentals.py` from `_TEST_FIXTURES`;
- classified this slice's design, plan, and evidence as historical path-evidence;
- retained both existing node IDs and every other classification; and
- changed from the Task 0 expected `205 passed / 1 failed` RED to `206 passed`.

All 30 protected current provider/product owners remained byte-identical to
`4ca66f0b373031fa64c73c537575a8d9fc25bba4fec663522cca59ed766b2fd2`.

### 8.4 Collection and focused runtime

Post-cutover collect-only report SHA-256 is
`9740a9edf8742016ffa20d4fe5845c8e7867c66032c61e03c0245950d2e068a5`;
transcript SHA-256 is
`a12d61ecc6699a752bd469155101926a0acbdad5cd3717912102331635a09e08`.
It completed with exit `0`, `4561` collected IDs, and `0` seen IDs:

```text
stage collection 4561 / dd127ce5dd34249a364b6a7965517aac66492b3d044ea8cc21e79a9706e58620
base-only IDs   46 / bb23dd1d6c5415cf5043767bb72c1137dd6d6b897e513984a6c3e590d4645e4a
tip-only IDs     0 / e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
tests/live IDs   0 / e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The 46-ID difference is byte-identical to the Task 0 January-node stream.

The same reviewed 206-node retained owner set then ran under the native wrapper
with only an empty `data/` marker:

```text
206 collected / 206 seen / 206 passed / 0 non-passing
```

Reporter SHA-256 is
`b63465735fbe89d4433069afbb7afcbe2dd332558b70022fbfa1f84efd3e67e1`;
transcript SHA-256 is
`ce1c38163dcf6dfbce9f4e71c5afdfa41f7fb458c3fc192eaaa24be6c4491d7f`.
`data/` remained empty and was removed; `src/data` remained absent.

### 8.5 No-provider and unfinished-work boundary

Task 1 executed no January file, rate-curve test, Tiingo diagnostic, or valid
yfinance ticker request. Collection had `seen=0`; focused runtime included only
the reviewed fake/local retained owner set. No credential, comparison output,
cookie, timezone cache, yfinance artifact, repository data file, or production
state was created or changed. Both worktrees were clean after implementation.

Task 2 rate-family deletion, Task 3 Tiingo executable/config cutover, final
`4527` collection, and native `4488 passed / 39 skipped / 0 failed` admission have
not run. Task 1 evidence review is the only next gate.

## 9. Task 2 - unconsumed option-rate family retirement

Task 2 used the single-use root
`/tmp/provider-smoke-hygiene-task2-ec140ae1`. Its recursive `SHA256SUMS`
contains `81` entries and has SHA-256
`e7a6c7c5c40deba873291695d871a410a8bbefff8c2ce2bf9dee42c9201ac447`.

### 9.1 Pre-cutover consumer and node proof

The current post-Task-1 tree was recollected before edits. Reporter SHA-256 is
`9740a9edf8742016ffa20d4fe5845c8e7867c66032c61e03c0245950d2e068a5`;
the new transcript SHA-256 is
`7388ee177ab62c2b260a58de15e9640fcc95874ec13f64449100ecc71bc1c2b6`.
It completed collect-only with `4561` collected, `0` seen, and exit `0`:

```text
stage collection 4561 / dd127ce5dd34249a364b6a7965517aac66492b3d044ea8cc21e79a9706e58620
rate family       34 / 3d51583972d7d5172fc5cf53be569469bfe192800a07d097ef9f81cd7a32ad21
```

The 34-node stream is byte-identical to the independently reviewed Task 0
stream. An uncapped tracked-Python symbol/import census found only these four
paths:

```text
src/options_math/__init__.py
src/options_math/option_pricing.py
src/options_math/rate_curve.py
tests/test_rate_curve.py
```

There was no fifth product/test consumer. In particular,
`src/tools/options_tools.py` retains the caller-supplied `risk_free_rate`
contract and imports only surviving pure pricing functions. The old rate tests
were not executed: six of their nodes could query Yahoo without a provider fake,
so collection plus complete consumer census is the truthful retirement RED.

### 9.2 Exact no-tail cutover

Implementation commit `7f68f1fdd71225e889eb01e74622420fb288ba64`
changes exactly four paths (`+1/-673`):

- deletes `src/options_math/rate_curve.py` and `tests/test_rate_curve.py`;
- removes only the retired rate exports from `src/options_math/__init__.py`; and
- removes `get_risk_free_rate`, its memory/disk cache helpers, Yahoo import path,
  and now-unused `Path`/`Tuple` imports from `option_pricing.py`.

No constant replacement, shim, deprecated export, optional import, or archived
code tail remains. Pure Black-Scholes/Bjerksund-Stensland pricing and explicit
caller-provided rate inputs are unchanged. This retirement also does not reject
future options valuation: a provider-supplied estimate may be designed later
against then-current evidence, but it is a new capability contract rather than
a reason to retain this unconsumed implementation.

### 9.3 Final collection and surviving option owners

Post-cutover collect-only reporter SHA-256 is
`b2f1539d751614942b45b15d5366383c3d7a1880adcfe3fcedb7b49d1406cc46`;
transcript SHA-256 is
`8c1c41ec0e24b1b9f439c52457d32e455aa0ba3739e281351fc4e20d53161eca`.
It completed with exit `0`, `4527` collected IDs, and `0` seen IDs:

```text
final collection 4527 / 4eeb117804ad874c83ffe4c04fd25ecd4de4f460801bfbf95d15c1406f32455d
pre-only IDs       34 / 3d51583972d7d5172fc5cf53be569469bfe192800a07d097ef9f81cd7a32ad21
post-only IDs       0 / e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The final stream is byte-identical to the Task 0 preconstructed authority, and
the pre-only stream is byte-identical to `tests/test_rate_curve.py`'s 34 IDs.

The two surviving option owner files then ran through the native wrapper:

```text
75 collected / 75 seen
74 passed / 1 skipped / 0 non-passing
collection SHA-256 b710ec5c6d50f541fed94994e863a1e2feaf0ec87aeb11c6e3a98eb4e2da099f
```

Reporter SHA-256 is
`250a7167990ce3dbfebbf33237f3a68edd0c32396efc2371d63a1231ba2cbdc4`;
transcript SHA-256 is
`f7ca1d1ad5633ef7181816d1323b98cb0eec5f6e27b4962d11ab45c335a637e3`.

### 9.4 Static and execution boundary

The post-cutover retired-symbol census across tracked Python is the empty
stream. Current options code and the surviving tests contain no retired symbol,
`rate_curve` import,
`risk_free_rate.json` path, or yfinance import. `src/options_math` compiled with
its bytecode redirected to the external artifact root; `git diff --check`
passed. The implementation worktree was clean after commit, with no repository
artifact or data path created.

Task 2 issued no provider request and received no credential. Collect-only had
`seen=0`; the surviving option owners are pure/local tests and the only live
Yahoo path under this slice remains the explicit, non-collected manual smoke
created in Task 1, which was not run. Task 3 Tiingo cutover, final native
`4488 passed / 39 skipped / 0 failed` admission, and merge have not run. Task 2
evidence review is the only next gate.
