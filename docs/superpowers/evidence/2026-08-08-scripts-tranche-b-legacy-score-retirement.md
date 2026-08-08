# Scripts Tranche B Legacy Score Retirement Evidence

> **Status:** PLAN PACKET WRITTEN; IMPLEMENTATION NOT STARTED
> **Date:** 2026-08-08
> **Plan base:** `04dd9a67d75042aa078bedde1e6dbc2a68e7736a`
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
master/base: 7257699171a81294b74ff8cde61fb90bb065a2b4
inventory:   098dff564faea1fc2617e198414ccde6067f23f8
decision:    04dd9a67d75042aa078bedde1e6dbc2a68e7736a
branch:      codex/scripts-tranche-b-inventory
```

`72576991` is the merge base and ancestor of `04dd9a67`. Before this plan-gate
packet the worktree was clean.

Plan source identity presented for review:

```text
674 lines / 36,175 bytes
SHA-256: 09307ddb241e2d0ac19ea61b467e6d07a34eb825663df67381eabdb7bc74562d
Git blob: cda25357d145ca675774b20c852f958ce8455fc8
```

### 2.2 Canonical base inputs

The product tree is docs-only relative to the EIR-006 merged canonical base.
The deterministic streams reused for plan construction are:

| Stream | Count | SHA-256 |
|---|---:|---|
| backend full | 4,581 | `6e4994bb664501cff75cb06dbad18db82ba68cbbe4b2b26c4d480250d7c4699f` |
| frontend full | 1,077 | `3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb` |
| backend focused | 555 | `ea5d897ca3597ef4edca7583db0b363360ceba9e362e516422f901ff8af004dd` |
| frontend focused | 27 | `c77d25d5bf7c868899d555f099fc245b13d35d4052a6a8242ac3f5c1300fb584` |

Task 0 must reproduce these values; this evidence does not substitute prior
results for that executable gate.

### 2.3 Node-ledger construction

The plan author independently enumerated whole retired test files, mixed-file
retired IDs, and all new IDs. Streams were combined as sets, sorted by UTF-8
byte order, written with one trailing newline, and checked for uniqueness and
set membership.

| Stream | Count | SHA-256 |
|---|---:|---|
| retired | 137 | `997fb56be265e28c8f9b990ff2a2315c773d5645af2f7f1c5087a905cc4b2642` |
| backend additions | 17 | `8b309a7af7df5e77cb4ba155fc32b88091305c4b5e7f21f9dd42eddd0b012e61` |
| backend RED | 4,598 | `3378204c781c0433eafbcacca317cfe82613986fc9b3fb716412729a3aa09938` |
| backend final | 4,461 | `463e864ef9ff0ec0b2b231e31836d52ed3a0ec7763ab654de70cdc41fc400d98` |
| focused RED | 572 | `44c969899da478b7a0e7412a32ddc47190205b35419e6031949659489b7bb890` |
| focused final | 435 | `0db45120dc5740c9372f07d6a5a9cec7d525a8d66e667fac6d170c99a827db18` |

Retirement composition is exact `102 whole-file + 35 mixed-file = 137`.
Addition composition is exact `8 boundary + 1 API + 3 monitor + 1 identity + 1
morning brief + 1 volume tool + 2 truthful bridge-registry IDs = 17`.

Plan self-review found that two retained bridge tests encoded the old count in
their node names (`tools_count_31`). Changing only their assertions would leave
false test names. The ledger therefore retires those two IDs and adds
`tools_match_registry` replacements. This changes stream identities but not the
final count or native pass/skip arithmetic.

The 26 explicitly retained/evolved base IDs are all present once; their sorted
stream SHA is
`3215e2d46407119ddc3c104dd8edef935158701cfd8c2d1556a45af4f0d8da7f`.

### 2.4 Projected audit streams

| Projection | Count | SHA-256 |
|---|---:|---|
| storage/writer/root | 4,537 | `e2a744b8fdcb9cadcaa1a9e68f050805faf36b5e7beae1d033d889a71e2f44af` |
| raw DTO/backend | 4,498 | `55b26b2ea092a378f04eb8f64de248e7c74364544ec1ab00eee2c29fb157324c` |
| raw user behavior | 4,498 | `d6a0793368c7cc68b81bb96863028b46db4cbd3dc6200977b7ec8621d5fda2ba` |
| volume/event/composite | 4,458 | `456d0f54a5f354210fb287ebe4425736bff30f0ae0540d504673b63f60eaa76a` |
| final | 4,461 | `463e864ef9ff0ec0b2b231e31836d52ed3a0ec7763ab654de70cdc41fc400d98` |

These are mathematical node-accounting projections. They have not been called
runtime GREEN and do not relax the atomic final gate.

### 2.5 Native target arithmetic

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
passed:  4509 - 135 retired passing + 17 new passing = 4391
skipped:   72 -   2 retired skipped                  =   70
total:   4391 + 70                                  = 4461
```

Task 4 must prove this result; arithmetic is not admission evidence.

### 2.6 Frontend construction

The sole new decoded Vitest node is:

```text
src/legacyScoreRetirement.test.ts<TAB>legacy score retirement boundary > removes score fields from current frontend DTOs and fixtures
```

| Stream | Files / nodes | SHA-256 |
|---|---:|---|
| base | `97 / 1077` | `3f5e9f5bbe88d5ac48015a8c9e9d669dcd649a53a2ac868fc8a98d21f8d7e4eb` |
| target | `98 / 1078` | `de1e0c3fccb1fad3574a5089f76164791895e7c5a70bb4a2ce578b38b30d4192` |
| focused base | `3 / 27` | `c77d25d5bf7c868899d555f099fc245b13d35d4052a6a8242ac3f5c1300fb584` |
| focused target | `4 / 28` | `b11cc27b90c570b20aeb728c48d39497e7eff555976f0c00a42d6129b26cf1cd` |

The target was constructed from decoded runtime names, not escaped JSON text.
Task 0 must re-run the pinned normalizer.

### 2.7 Protected lineage

The current tracked `training/` tree has 53 paths. Its sorted
`path<TAB>git-blob` stream is:

```text
2284c8989f6104979a11a5111de987f5d6f2974e3d2f74f0cf47ed5b4854e14a
```

Provider-native sentiment and investor-profile risk were found in distinct
owners. They are protected and receive a dedicated new regression node rather
than being removed through token matching.

## 3. Locked implementation sequence

| Task | Status | Gate |
|---|---|---|
| Task 0 re-ground | not started | independent evidence review |
| Task 1 RED | blocked | Task 0 GREEN |
| Task 2 atomic cutover | blocked | plan + Task 0 GREEN |
| Task 3 mutation/focused gates | blocked | product target exact |
| Task 4 native admission | blocked | all mutations restored |
| Task 5 independent implementation review/merge | blocked | native GREEN |
| Task 6 merged verification/closeout | blocked | fast-forward merge |
| physical score-row manifest | separately blocked | merged rollout + new review/user approval |
| scoring-secret disposition | separately blocked | exact consumer metadata + new review/user approval |

## 4. Plan-gate verification checklist

- [x] PD 1-PD 8 approval recorded without broadening destructive authority.
- [x] Product and data/secret scopes separated.
- [x] Backend base/RED/final and focused streams precomputed.
- [x] Whole-file and mixed-file retirement IDs enumerated exactly.
- [x] Seventeen independent new backend nodes named.
- [x] Frontend decoded target and focused identities precomputed.
- [x] Two canonical retired skips directly reproduced.
- [x] Five intermediate projections labeled accounting-only.
- [x] Ten product mutations mapped to owning nodes.
- [x] Provider-native/training/EvidencePacket boundaries protected.
- [x] Native wakeup/reporter/wrapper/toolchain boundary pinned.
- [ ] Independent plan review GREEN.
- [ ] Task 0 authorized and executed.

## 5. Honesty boundary

This packet proves that the plan is grounded and internally accounted. It does
not prove that the implementation exists, that the final suite passes, or that
production score rows are deletable. Those claims require their later gates.
