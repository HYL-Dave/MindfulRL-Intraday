# I18N-4/5 Remaining Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:using-git-worktrees` before Task 0,
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute the cleared plan,
> `superpowers:test-driven-development` for every behavior change,
> `superpowers:requesting-code-review` before integration, and
> `superpowers:verification-before-completion` before any passing or complete
> claim. Track every step with the checkbox syntax below.

> **Status: LIVE COMPLETE — INDEPENDENT IMPLEMENTATION REVIEW GREEN;
> FAST-FORWARD MERGED THROUGH `5f35e8b1`**
>
> The combined design received independent full-document GREEN against docs
> tip `78f0d074f99f63a2a832c35352fdd6ba9f76192c`. Independent review has
> cleared this plan; product implementation may begin only from the docs-only
> `PLAN_REVIEW_CLEARANCE_COMMIT`. The behavior A/B base remains
> merged product `93cda66831b7202fd0dfafcc0d1c0604b07e94bd`; later docs commits do
> not replace that anchor. Merge remains blocked on independent implementation
> review GREEN and explicit user approval.

**Goal:** Finish all remaining application-chrome localization in one bounded
two-tranche unit: Tranche A migrates Research and shared model-selection copy;
Tranche B migrates Portfolio, System, and shared residual chrome, retires the
unreachable AppRecords frontend, inventories formatter ownership without
changing it, and closes the visible-literal scanner to zero debt.

**Architecture:** Keep `zh-Hant` and `en` resources static and namespace-typed.
Present semantic IDs at read time through pure presenters that receive the
owning namespace translator. Preserve source, user, generated, identifier, and
measured values byte-for-byte. Freeze a fully green `TRANCHE_A_TIP` before any
Tranche B product edit, then prove canonical `base -> A` and `A -> final`
comparisons separately. The final scanner owns all `src/**`, has an empty debt
manifest, and reports only the unchanged reviewed allowlist.

**Tech stack:** React 18, TypeScript 5.9, i18next 26.3.6 selector API,
react-i18next 17.0.10, Vitest 4/jsdom, the TypeScript-AST visible-literal
scanner, Vite/Electron, pytest for the immutable backend baseline, and
Playwright/CDP against isolated fake-backed frontend data.

---

## Design Authority

1. Combined product/localization authority:
   `docs/superpowers/specs/2026-07-24-i18n-4-5-remaining-surfaces-design.md`.
2. Runtime locale mechanics and selector-last release:
   `docs/superpowers/specs/2026-07-20-app-wide-i18n-decision.md`.
3. Canonical vocabulary: `docs/design/ARKSCOPE_TERMINOLOGY.md`.
4. Research workspace, Portfolio/Holdings, canonical Shell, and P2.8 Slice 5
   authorities continue to own behavior and IA.
5. Product A/B base: `93cda66831b7202fd0dfafcc0d1c0604b07e94bd`.

If this plan conflicts with an authority above, stop and amend the authority
before editing product code.

---

## Independent Spec-Review Resolution

Independent review returned GREEN with zero required changes. Both plan-level
advisories are binding here without changing test, resource, or scanner
accounting:

1. **Tranche A worst composition.** Every A runtime width must mount long
   English chrome, an error banner, active progress, and long planted source
   content in the same Research viewport. Separate happy-path captures do not
   satisfy this gate.
2. **Sidecar copy owner.** `App.tsx` owns the structured sidecar state and
   `Dashboard.tsx` renders it, but all localized sidecar/System outcome copy is
   owned by the `system` namespace. No Shell or Explore copy owner is added.

---

## Independent Plan-Review Resolution

Independent full-plan review returned GREEN with one advisory and no accounting
change:

1. **Dashboard-wide bilingual ownership — incorporated.** The existing planned
   node `renders System sidecar copy from the system namespace in both locales`
   retains its exact ID and is strengthened in place. It must resolve and assert
   all 20 `system` leaves through the mounted Dashboard in both locales,
   including the roughly fourteen non-sidecar System/Health labels as well as
   sidecar loading/error/retry copy. Task 10's runtime matrix remains an
   independent visual gate, not the sole coverage for those leaves.

The initial reviewed frontend ledger was A `+44/-2`, B `+60/-0`. The measured
Research-title overflow deviation below revises A to `+45/-2`; the later
Holdings narrow-width deviation revises B to `+61/-0`. Resource, scanner, and
both focused-suite ledgers remain unchanged.

### Implementation grounding correction

Task 1's RED inventory exposed one plan-only subtree arithmetic error without
changing the reviewed namespace total or product scope. The exact 23 Settings
leaves moved to Common consist of 22 leaves under `settings.models` plus
`settings.providers.openAI.tokenExpired`. Therefore Settings still moves
`702 -> 679`, Common still moves `32 -> 56`, and all node/scanner ledgers remain
unchanged, but the `settings.models` subtree truth is `91 -> 69`, not `91 -> 68`.
No unrelated model leaf may be deleted merely to satisfy the former typo.

### Runtime-reviewed deviation 1: Research title source-content visibility

The Tranche A worst-composition gate found one real stop condition after all
planned Research tests and scanner gates were green. The existing
`.research-conversation-title` rule in `styles.css` combines
`overflow: hidden`, `text-overflow: ellipsis`, and `white-space: nowrap`.
With the reviewed long source title, the EN `1440x900` case measured
`clientWidth=559` and `scrollWidth=606`; both locales also clipped at
`960x768` and `390x844`. Request/replay counts, node identity, draft/focus,
Drawer state, source hashes, horizontal overflow, and overlap were otherwise
green.

This deviation reopens exactly one existing CSS owner and one existing test
file:

1. `styles.css` may change only the `.research-conversation-title` block by
   removing the three truncation declarations and adding intrinsic wrapping
   (`white-space: normal` and `overflow-wrap: anywhere`). No breakpoint,
   width, font-size, or source-value change is allowed.
2. `shell/ShellCss.test.ts` gains exactly one RED-first node named
   `lets Research conversation titles wrap without truncating source content`.
   It must reject hidden/ellipsis/nowrap and require the two wrapping
   declarations.
3. The raw A ledger becomes `+45/-2`; full frontend becomes
   `93 files / 987 nodes`. A focused remains `24/305`; the CSS node is verified
   separately in the now `9/9` Shell CSS suite and in the full run.
4. The same six-case `zh-Hant`/`en` matrix at `1440x900`, `960x768`, and
   `390x844` must rerun with zero clipping while preserving all previously
   green geometry, identity, request, and source-byte checks.
5. The other three CSS owners remain byte-identical. The `styles.css` gate is
   replaced only by the exact reviewed title hunk.

### Runtime-reviewed deviation 2: Holdings fixed-English eyebrow ownership

Task 5's mounted bilingual review exposed one visible PageHeader prop that the
CJK scanner does not inventory: the pre-migration Holdings eyebrow is the
English word `Holdings` in both locales. The locked unnamed-zh byte contract
forbids silently changing that existing zh-Hant chrome to `持倉`, while leaving
the value hardcoded would violate resource ownership.

This deviation adds exactly one direct UI-owned Portfolio leaf,
`holdings.surface.eyebrow`, whose value is byte-identical `Holdings` in both
locales. The ownership fixture records it as a direct claim, distinct from
source-signature and presenter claims. The existing complete-Holdings-chrome
node evolves in place; no test node or scanner count changes.

Consequently the Holdings family is `69`, Portfolio is `374`, and the final
per-locale resource total is `1779`. This is a resource-ledger correction only:
no IA, behavior, formatter, request, CSS, or terminology-authority change is
allowed. Task 5 also installs the already-planned Common `Ticker` placeholder
when its first consumer is wired; Task 7 installs the other four Common leaves.

### Runtime-reviewed deviation 3: JSX entity transfer preserves rendered copy

Task 6 exposed one representation-only transfer edge in the existing Provider
recovery action. JSX source stored the greater-than separators as `&gt;`, which
the JSX parser rendered as visible `>`. An i18next resource string does not
decode HTML entities, so copying the source bytes would incorrectly render the
five literal characters `&gt;`.

The zh-Hant Portfolio resource therefore stores
`前往設定 > Data Sources > IBKR`. This is the exact pre-migration rendered
copy, not a wording correction. Task 6 may change this one resource value
outside its six component/test owners. Resource keys/counts, scanner counts,
runtime behavior, and all other zh-Hant bytes remain unchanged.

### Implementation grounding correction: ownership proof after debt closure

Task 9's RED empty-debt transition exposed one omitted test dependency. The
existing `contains the reviewed remaining-surface namespace inventory in both
locales` node used the debt manifest as a temporary source-list witness for the
Portfolio ownership fixture. Once the reviewed final state empties that
manifest, the old comparison necessarily fails even though the fixture remains
the durable migration audit.

The same node therefore evolves in place with zero accounting change. It now
requires the final debt manifest to be empty, pins the fixture at `372` unique
source claims / `391` occurrences across the exact five Portfolio source
files, and retains its exhaustive claim-to-resource/path checks. Task 9 adds
`src/i18n/resources.test.ts` to its test-only file list. No resource, product,
scanner-policy, or test-node change is authorized by this correction.

### Runtime-reviewed deviation 4: bound the Holdings account filter

Task 10's full Tranche B matrix found one real layout failure at `390x844`.
The account-filter label and `<select>` measured `434px` wide inside a `375px`
content viewport because a long source-owned account label established the
control's intrinsic width. Both locales overflowed identically; all node,
draft, focus, source-byte, request, privacy, and other geometry checks were
green. This is a layout defect, not a reason to shorten or translate source
content.

This deviation reopens exactly `Holdings.tsx`, `styles.css`, and one existing
test file:

1. the local action row and account-filter label receive
   `portfolio-holdings-filter-row` and
   `portfolio-holdings-account-filter`; no source value or control behavior
   changes;
2. `styles.css` gains only class-scoped intrinsic-width rules for that action
   row, label, and `<select>`. The row and label are bounded by
   `min-width: 0` / `max-width: 100%`; the select additionally fills its
   bounded label with `width: 100%`. No breakpoint or fixed width is added;
3. `shell/ShellCss.test.ts` gains exactly one RED-first node named
   `bounds the Holdings account filter against long source labels`, proving the
   classes are wired and all three levels are bounded;
4. B raw accounting becomes `+61/-0`, full frontend becomes
   `94 files / 1048 nodes`, and the final Shell CSS suite becomes `10/10`.
   B focused remains the reviewed `15/232` and the CSS node is verified
   separately plus in the full run;
5. the complete `36`-case bilingual runtime matrix must rerun. The Holdings
   `390x844` select must fit without clipping, and every previously green
   identity, request, privacy, source-byte, and geometry check must remain
   green.

This is the sole Tranche B exception to the frozen A product-owner rule. It
extends the `styles.css` exception by one exact class-scoped hunk; no other A
product owner may change.

---

## Task 0 Evidence

- `PLAN_REVIEW_CLEARANCE_COMMIT`:
  `d05bf833b92e325afd0a4c606d327b6a8c402195`.
- Isolated branch/worktree: `codex/i18n-4-5-remaining-surfaces` at the clearance
  commit. The linked worktree received the existing repository `git-crypt` key
  before `git read-tree -mu HEAD`; the protected evaluation document then read
  as `47,608` bytes.
- Frontend baseline: `90 files / 944 nodes`, all green. Sorted node-list
  SHA-256: `854974cb302ead688e72f8bb04a4cd85746e954a98234f0240d55a9ecdb2be86`.
- Tranche A focused baseline: `21 files / 263 nodes`, all green. Sorted
  focused-list SHA-256:
  `76fb12afdca97cf08b6103d68a66d11eed43d2ec71a7b17026c3b9bf988784c0`.
- Backend baseline: `4621` nodes collected with only the four established
  collection/deprecation warnings.
- Typecheck passed. Production build passed with only the established chunk
  size warning.
- Scanner ran twice with exact `703/656/637/20` and 39 scopes. Both output
  files had SHA-256
  `539fc0b4eec50c08e3cc93ff10696ab7101372187d9500566aebd9002204cde8`.
- Scanner manifests at baseline:
  - debt `bab538b3d57b8fb020a8aa6d1253377e6daec32d3d44a0b44974dac75bb3e3ee`;
  - allowlist `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`;
  - scopes `677345581c6c34b2503ec145fb674001876de7e2bc8f51081d79a9934dbefda8`.
- A direct source scan reconciled manifest `637/668` to current source
  `636/667` with exactly one stale debt entry. Per-file aggregation reproduced
  A `230/242`, reachable B `374/393`, AppRecords `30/30`, and `api.ts` `3/3`.
  The 22 reviewed missed headers/tuple labels were each present; no extra value
  was admitted.
- Product selector wiring remains absent. A read-only production DB query found
  zero `profile_settings.ui_locale` rows.
- Protected base tree/blob identities were captured against `93cda668`,
  including root `src` `f02b086e`, `data_sources` `68108409`, tests
  `e0a987ac`, desktop `1e8a0d6f`, extensions `dacd223b`, and the four CSS
  blobs `b763608a`/`80cbacf2`/`d92000dc`/`4ba943d0`. `git diff --check`
  passed.

## Tranche A Checkpoint Evidence

- `TRANCHE_A_TIP` is
  `34ddf08f5983d523bf1bfb00ce6b06a55a76bce0`; it descends from product base
  `93cda66831b7202fd0dfafcc0d1c0604b07e94bd` and contains no Tranche B
  product edit.
- The reviewed raw node ledger closes at `+45/-2`: A focused is exactly
  `24 files / 305 nodes`, the separate Shell CSS suite is `9/9`, and the full
  frontend is `93 files / 987 nodes`. Typecheck passed; the production build
  passed with only the established chunk-size warning.
- Scanner ran twice with exact `483/448/429/20`, `448` debt occurrences, and
  `48` migrated scopes. The deterministic scanner-output SHA-256 is
  `a87f12d2085c0d5b7e747e52edf4cd3059edf2f0c86b506ebc69891f84a1372d`.
  Manifest SHA-256 values are debt
  `b7e198e12eb24fa6a235836b80480d594b79b3e89bc12f1c796d3120ecb2b6b9`,
  scopes `9816fbfd9d6bada4bcd3cfe8216ba8d3ff79e468baedd37a29ff1efabcac5e45`,
  and unchanged allowlist
  `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`.
- Resource inventories are exact and equal between locales: Common `56`,
  Shell `37`, Settings `679`, Research `207`, Explore `401`.
- The isolated worst-composition gate ran both locales at `1440x900`,
  `960x768`, and `390x844`. After the reviewed title-wrap deviation all six
  cases passed geometry, identity, request-count, focus, source-byte, and
  cleanup checks. The planted source SHA-256 was
  `b50ae8f04b0b8503b1d2ac74251a715e268553b33e0c5a0738966ec02161b960`
  in every case; the pre-hardening evidence SHA-256 was
  `482a7c858c4d9b109be09611a66916eb686cc24b3f83c1c6a8a48e2f876c82e7`.
  Every viewport now also compares all five rendered source fields and their
  aggregate hash against independently planted canonical values, rather than
  only comparing locales with each other. Refreshed evidence SHA-256 is
  `b8948a0916ab1ddb5c818ba0dd9e5f7accb55bf379fbe2be266559f9bb0914b1`.
  Evidence remains under
  `/tmp/arkscope-i18n45-tranche-a-gate-20260724/`; isolated Vite `8454` and
  CDP `9254` were closed after the run.
- The CSS deviation had an exact RED witness: the new Shell CSS node failed
  while the other eight nodes were skipped. The reviewed five-line title hunk
  then made `9/9`, A focused `24/305`, and full `93/987` green; the other three
  CSS owners remained byte-identical. Post-fix screenshots were inspected at
  all three English widths with no clipping, overlap, or horizontal overflow.
- Fresh narrow spec review of the checkpoint manifests, boundary test, and CSS
  hunk returned GREEN with no findings. Earlier per-task spec and quality
  reviews are closed. The subsequent base-to-A review found one locale-derived
  React key, one insufficiently anchored runtime assertion, and mismatched
  documented suite prefixes. The key now uses stable `group.id` with an
  in-place node-identity assertion, the runtime gate is canonical-value
  anchored, and the ledger below names the collected suite exactly. Full
  frontend remained `93/987`; no node was added or removed by these fixes.

---

## Grounded Baseline

Reproduce these values in the cleared worktree before any product edit:

| Surface | Baseline at `93cda668` |
| --- | ---: |
| Frontend tests | `90 files / 944 nodes` |
| Backend collect | `4621` |
| Scanner candidates | `703` |
| Scanner current signatures | `656` |
| Scanner debt signatures | `637` |
| Scanner allowlist entries | `20` |
| Migrated scopes | `39` |
| Debt manifest ceiling | `637 signatures / 668 occurrences` |
| Current source debt | `636 signatures / 667 occurrences` |
| Resources per locale | Common `32`, Shell `37`, Settings `702`, Research `5`, Explore `401` |
| Public locale selector | absent |

The one-signature difference between manifest ceiling and source is the stale
`api.ts` presenter-return record documented by I18N-2 Decision 65. Do not
interpret the debt manifest as current-source truth.

### Baseline scanner partition

| Owner | Signatures | Occurrences |
| --- | ---: | ---: |
| Tranche A Research | `204` | `216` |
| Tranche A shared model copy | `26` | `26` |
| Tranche B reachable Portfolio/System/common | `374` | `393` |
| AppRecords unreachable frontend | `30` | `30` |
| `api.ts` current/stale adjudication | `3` | `3` |
| **Manifest ceiling** | **`637`** | **`668`** |

Scanner hardening must additionally expose exactly 22 current English
table-column labels. Any other discovery is a stop-and-amend.

### Tranche A focused seed suite

Exactly `21 files / 263 nodes`:

```text
src/i18n/visibleLiteralScanner.test.ts                 10
src/i18n/resources.test.ts                             13
src/i18n/foundationBoundaries.test.ts                  10
src/modelPicker.test.ts                                 8
src/modelRoutingUx.test.ts                              7
src/researchErrors.test.tsx                             7
src/researchReducer.test.ts                            72
src/researchSelection.test.ts                          12
src/ResearchHistoryDrawer.test.tsx                     10
src/ResearchWorkspace.test.tsx                         14
src/ResearchPersonalizationContext.test.tsx            12
src/ResearchPendingBubble.test.ts                       1
src/researchRunReplay.test.ts                           3
src/researchModels.test.ts                              7
src/ProviderSection.test.ts                             8
src/ModelRoutingSection.test.ts                        20
src/SettingsModelRouting.test.ts                       14
src/settings/settingsBackendCopy.test.ts               12
src/shell/researchWork.test.tsx                        13
src/ResearchRuntimeSection.test.ts                      5
src/ResearchShellNavigation.test.tsx                    5
                                                       ---
                                                       263
```

### Tranche B focused seed suite at `TRANCHE_A_TIP`

Exactly `14 files / 172 nodes` after A's scanner/resource evolution:

```text
src/i18n/visibleLiteralScanner.test.ts                 14
src/i18n/resources.test.ts                             14
src/i18n/foundationBoundaries.test.ts                  10
src/Holdings.test.tsx                                  26
src/PortfolioActivity.test.tsx                         16
src/PortfolioCapturePanel.test.tsx                     25
src/PortfolioAccountOverview.test.tsx                  10
src/PortfolioRecentActivity.test.tsx                    3
src/AppShell.test.tsx                                  15
src/ui/DataTable.test.tsx                               6
src/ui/overlays.test.tsx                               14
src/MarkdownView.test.ts                                8
src/SettingsPostPgExitStorage.test.ts                   8
src/PortfolioCaptureCss.test.ts                         3
                                                       ---
                                                       172
```

Collection drift is a stop condition. Do not silently revise either suite or
hide a removed assertion behind net growth.

---

## Locked Implementation Decisions

1. One branch, one plan, one independent implementation review, and one merge
   contain two hard product checkpoints. Tranche B product edits may start only
   after a clean, fully green `TRANCHE_A_TIP` has been committed and recorded as
   a full 40-character hash.
2. Independent review compares `93cda668 -> TRANCHE_A_TIP` and
   `TRANCHE_A_TIP -> final` separately. B may not mask A regressions with final
   net counts.
3. After the A checkpoint, B may not modify A-owned product paths. A discovered
   cross-tranche correction requires stop-and-amend and a replacement
   `TRANCHE_A_TIP` with refreshed evidence.
4. Only application chrome localizes. Source content, user input, generated
   prose, evidence, thread titles, ticker/provider/model IDs, raw metric values,
   and persisted semantic IDs remain original.
5. Existing rendered zh-Hant chrome moves byte-for-byte unless this plan names
   a correction. The only visible replacement authorized here is the Holdings
   example placeholder `NVDA -> Ticker`; real ticker values are untouched.
6. Resources use static typed selector keys only. Dynamic key construction,
   lazy loading, Suspense, locale-derived React keys, and runtime source values
   in resources are forbidden.
7. The future locale selector remains absent. `ui_locale` does not enter any
   prompt or trigger translation, refetch, replay, or remount.
8. Locale switching may issue only the controller's locale-preference PUT. It
   preserves mounted node identity, draft/input state, focus, drawer/dialog
   state, scroll position in the practical anchored sense, active work, and
   translated/source results. Data-request counts do not increase.
9. Shared model display copy moves to `common`; Settings duplicate leaves are
   removed. Decision 37 remains stronger, not weaker: `id`, `baseLabel`, and
   `compatibility` stay structural, and production code never reverse-parses a
   decorated model label.
10. `researchErrors.ts` and Portfolio/System outcomes use semantic IDs and
    structured facts. Presenters read no localized or backend English message
    to infer meaning. If a typed discriminator is missing, stop and amend; do
    not parse `.message`.
11. Normal UI never renders arbitrary `Error.message`, `error_detail`, SQL,
    traceback, path, token, or exception text. Developer Mode may render only
    reviewed stable code/status/route facts; unproven detail is omitted.
12. Sidecar structured outcome state is stored by `App.tsx`, rendered by
    `Dashboard.tsx`, and localized exclusively from the `system` namespace.
13. AppRecords frontend retirement is one dedicated commit. It removes only
    the unreachable component and exactly five dead client exports. Backend,
    offline storage, migrations, and archive capabilities remain byte-identical.
14. Scanner hardening is RED-first and recognizes direct `header` values plus
    the reviewed static tuple-column shape. It must not classify machine IDs,
    transport args, reason operands, or reducer comparison operands as copy.
15. Final scanner policy is stronger than CLI success: debt manifest empty,
    migrated scopes exactly `src/**`, allowlist byte-identical, candidates equal
    summed allowlist occurrences, and signatures equal allowlist entries.
16. Formatter work is inventory-only in Task 9: its product and formatter-test
    bytes are unchanged from the Task 8 input. Across product base to final,
    the only formatter-helper delta is the already-reviewed
    `PortfolioActivity.formatNumber` / `formatAmount` / `formatUnknown` change
    that accepts localized unknown fallback copy. Null/non-finite checks,
    `Intl` options, currency append behavior, date/time/percentage/rounding/
    timezone mechanics, and formatter output expectations remain unchanged.
17. Backend, data sources, backend tests, prompts, agents, desktop/native code,
    extensions, package manifests/lockfiles, and all four CSS owners are
    byte-identical by default.
18. A measured bilingual overflow may reopen one existing CSS owner only via a
    reviewed stop-and-amend with a named RED geometry node, exact hunk, and
    rerun boundary matrix. Copy may wrap; it may not be truncated or shortened
    to hide overflow.

---

## Exact Resource Ledger

### Tranche A

Research has `204` debt signatures, of which two completion-state operands are
scanner false positives. Add exactly `202` context-specific Research leaves to
the existing five, producing `Research 207`. The one-leaf-per-true-signature
rule is deliberate even where two existing zh-Hant literals have identical
bytes: separate UI contexts keep independent English grammar and ownership
instead of being coupled by accidental Chinese equality.

Common adds exactly 24 shared model leaves:

| Family | Leaves |
| --- | ---: |
| picker groups | `4` |
| reason labels | `9` |
| auth modes | `4` |
| thinking modes | `5` |
| compatibility grammatical contexts | `2` |
| **Common additions** | **`24`** |

The two compatibility contexts are a punctuation-free decorated-label suffix
and a complete Settings notice. A presenter must not splice punctuation from
the old decorated label or reverse-parse it.

Settings removes exactly 23 now-duplicate model leaves: four groups, nine
reasons, four auth modes, five thinking modes, and one legacy-mode label.
Settings-only explanatory prose stays in Settings.

The pre-Slice-5 Settings-origin inventory remains historically `612`, but its
physical ownership becomes `589` leaves in Settings plus the exact 23 moved
model leaves in Common. The Settings `models` subtree falls from `91` to `69`;
the twenty-third moved leaf is the Provider reauthentication label named in the
implementation grounding correction above.
The replacement resource node asserts all three values; no test name may keep
claiming that all 612 leaves still reside in Settings.

Tranche A resource target per locale:

| Namespace | Base | A delta | A target |
| --- | ---: | ---: | ---: |
| Common | `32` | `+24` | `56` |
| Shell | `37` | `0` | `37` |
| Settings | `702` | `-23` | `679` |
| Research | `5` | `+202` | `207` |
| Explore | `401` | `0` | `401` |
| **Total** | **`1177`** | **`+203`** | **`1380`** |

### Tranche B and final

Common adds exactly five leaves: DataTable action heading, row-action ARIA
template, ConfirmDialog default cancel, Markdown blocked-image fallback, and
generic `Ticker` placeholder.

Portfolio adds exactly 374 leaves after runtime-reviewed deviation 2:

| Family | Derivation | Leaves |
| --- | --- | ---: |
| Holdings | `68 debt - NVDA placeholder + count plural pair + fixed-English eyebrow` | `69` |
| Activity | `143 debt - 2 broker_day_gap operands + one plural` | `142` |
| Capture | `67 debt + one review-change plural` | `68` |
| Account Overview | existing semantic chrome | `36` |
| Recent Activity | `36 debt + one field-count plural` | `37` |
| scanner-missed table labels | exact inventory below | `22` |
| **Portfolio** |  | **`374`** |

Capture's executions/fees summary uses one neutral two-count sentence, not a
four-way plural cross product. System adds exactly 20 leaves, one per Dashboard
debt signature. Sidecar error copy is part of those System leaves.

Final resource target per locale:

| Namespace | Final leaves |
| --- | ---: |
| Common | `61` |
| Shell | `37` |
| Settings | `679` |
| Research | `207` |
| Explore | `401` |
| Portfolio | `374` |
| System | `20` |
| **Total** | **`1779`** |

Both new namespaces are statically imported by `resources.ts`. `zh-Hant` and
`en` key sets are identical and all leaves are non-empty.

---

## Scanner-Hardening Inventory

The RED scanner fixtures must expose exactly these 22 values that current
policy misses. They become resource values in B; until then the A checkpoint
debt manifest owns them.

| # | Owner | Current static label |
| ---: | --- | --- |
| 1 | Holdings | `Account` |
| 2 | Holdings | `Symbol` |
| 3 | Holdings | `Asset` |
| 4 | Holdings | `Qty` |
| 5 | Holdings | `Currency` |
| 6 | Holdings | `Avg Cost` |
| 7 | Holdings | `Market Value` |
| 8 | Holdings | `Notes` |
| 9 | Holdings | `Status` |
| 10 | Capture | `Avg Cost` |
| 11 | Capture | `Market Value` |
| 12 | Capture | `Unrealized P&L` |
| 13 | Account Overview | `Capture Run` |
| 14 | Account Overview | `Base Currency` |
| 15 | account-value tuple | `Net Liquidation` |
| 16 | account-value tuple | `Total Cash` |
| 17 | account-value tuple | `Settled Cash` |
| 18 | account-value tuple | `Gross Position Value` |
| 19 | account-value tuple | `Buying Power` |
| 20 | account-value tuple | `Available Funds` |
| 21 | account-value tuple | `Initial Margin` |
| 22 | account-value tuple | `Maintenance Margin` |

The nested Holdings `Unrealized P&L` is already current scanner debt and is not
one of the 22 additions. Professional English labels above may remain the same
bytes in `zh-Hant`; moving them into resources is still required.

Reviewed false-positive operands that policy must classify without allowlist
growth:

- two `sendCalibrationMessage` transport arguments;
- four model-picker reason-code comparison operands;
- two `broker_day_gap` comparison operands; and
- Research `running`/`complete` completion-state operands.

---

## Exact Test-Node Ledger

### Base to `TRANCHE_A_TIP`: raw `+45/-2`

| Test file | Add | Remove | Contract |
| --- | ---: | ---: | --- |
| `i18n/visibleLiteralScanner.test.ts` | `4` | `0` | header, tuple, model operand, Research state fixtures |
| `i18n/resources.test.ts` | `3` | `2` | generic inventory + shared owner + historical-origin inventory |
| `modelRoutingUx.test.ts` | `1` | `0` | all shared semantic IDs resolve in both locales |
| `i18n/researchPresentation.test.ts` | `10` | `0` | new pure presenter contract |
| `ResearchWorkspace.test.tsx` | `8` | `0` | bilingual mounted workspace and state preservation |
| `ResearchHistoryDrawer.test.tsx` | `6` | `0` | filters, mutations, in-flight and source title behavior |
| `ResearchEvidenceDrawer.test.tsx` | `7` | `0` | new Evidence mounted suite |
| `ResearchRunProgress.test.tsx` | `5` | `0` | new progress mounted suite |
| `shell/ShellCss.test.ts` | `1` | `0` | reviewed title-wrapping runtime deviation |
| **A** | **`45`** | **`2`** | net `+43` |

The two removed nodes are exactly:

```text
contains exactly 702 Settings 32 Common 5 Research and 401 Explore leaves per locale
contains exactly 612 pre-Slice-5 Settings leaves per locale
```

They are replaced by generic reviewed namespace and Settings-origin inventory
nodes. The latter requires current pre-Slice-5 Settings leaves `589`, model
subtree `69`, and the exact 23 leaves moved to Common to reconcile to the
historical `612`; it does not leave a stale count in a test ID. No semantic
coverage disappears. Full frontend closes at `93 files / 987 nodes`; A focused
closes at `24 files / 305 nodes`, with the separate Shell CSS suite at `9/9`.

Required new A node IDs:

```text
visible literal scanner > detects direct DataTable header properties including ASCII-only copy
visible literal scanner > detects tuple-backed static column labels without treating tuple IDs as copy
visible literal scanner > ignores reviewed model reason operands while retaining visible reason presenters
visible literal scanner > ignores Research completion-state operands while retaining visible status labels
bundled i18n resources > contains the reviewed remaining-surface namespace inventory in both locales
bundled i18n resources > moves shared model chrome to one Common owner without Settings duplicates
bundled i18n resources > preserves the reviewed pre-Slice-5 Settings-origin inventory across the Common move
Models terminology > resolves every shared model group reason auth mode thinking mode and compatibility context in both locales
research presentation > maps selection provenance and provider quota chrome in both locales
research presentation > maps run and history statuses without translating stable IDs
research presentation > maps Evidence token and timing labels without changing values
research presentation > maps empty response disconnect and progress outcomes from semantic IDs
research presentation > maps suggested prompts in both locales before they become drafts
research presentation > keeps Provider model effort run and error identifiers original
research presentation > preserves unknown stable values instead of collapsing them
research presentation > uses only static Research resource selectors
research presentation > renders no raw resource key for every closed presenter branch
research presentation > keeps source work and generated content outside the presenter
Research workspace contracts > renders the complete English Research workspace around original source content
Research workspace contracts > switches locale without remounting the workspace or resetting thread draft focus or drawers
Research workspace contracts > localizes unselected suggestions and freezes the selected draft across locale changes
Research workspace contracts > preserves transcript tool model effort and generated answer bytes
Research workspace contracts > renders late stream outcomes in the current locale without replaying the request
Research workspace contracts > uses structured thread not-found facts instead of parsing Error.message
Research workspace contracts > keeps active progress and error chrome localized in one mounted workspace
Research workspace contracts > keeps model selection metadata and no decorated-label reverse parsing
responsive application shell CSS > lets Research conversation titles wrap without truncating source content
Research history drawer > localizes filters statuses and actions in both locales
Research history drawer > renders structured 404 and 409 outcomes without parsing messages
Research history drawer > preserves search draft focus and selected thread across locale changes
Research history drawer > renders an in-flight rename result in the current locale
Research history drawer > preserves source thread titles exactly
Research history drawer > omits arbitrary diagnostics in normal mode
Research Evidence drawer > localizes headings token statistics and timing labels in both locales
Research Evidence drawer > preserves source trace evidence and context bytes
Research Evidence drawer > preserves disclosure scroll and focus across locale changes
Research Evidence drawer > retains the existing Developer diagnostic boundary
Research Evidence drawer > keeps unknown stable identifiers distinguishable
Research Evidence drawer > renders partial Evidence without claiming completeness
Research Evidence drawer > updates shared model and personalization labels reactively
Research run progress > maps every bounded run status in both locales
Research run progress > preserves exact progress and token values
Research run progress > renders semantic failure facts without raw detail
Research run progress > preserves node identity while locale changes
Research run progress > keeps the completion destination contract unchanged
```

### `TRANCHE_A_TIP` to final: raw `+61/-0`

| Test file | Add | Remove | Contract |
| --- | ---: | ---: | --- |
| `i18n/visibleLiteralScanner.test.ts` | `4` | `0` | B operands + durable zero-debt closure |
| `i18n/portfolioPresentation.test.ts` | `12` | `0` | new pure Portfolio presenter |
| `Holdings.test.tsx` | `8` | `0` | mounted bilingual holdings behavior |
| `PortfolioActivity.test.tsx` | `7` | `0` | activity/filter/detail behavior |
| `PortfolioCapturePanel.test.tsx` | `7` | `0` | semantic outcomes and polling races |
| `PortfolioAccountOverview.test.tsx` | `5` | `0` | account values/table headers |
| `PortfolioRecentActivity.test.tsx` | `4` | `0` | recent activity states |
| `AppShell.test.tsx` | `5` | `0` | System sidecar state and locale switching |
| `ui/DataTable.test.tsx` | `3` | `0` | heading/ARIA/source preservation |
| `ui/overlays.test.tsx` | `2` | `0` | default/caller-owned cancel behavior |
| `MarkdownView.test.ts` | `2` | `0` | fallback localization/source preservation |
| `i18n/foundationBoundaries.test.ts` | `1` | `0` | global scope and exact final arithmetic |
| `shell/ShellCss.test.ts` | `1` | `0` | reviewed narrow Holdings filter deviation |
| **B** | **`61`** | **`0`** | net `+61` |

Full final frontend is `94 files / 1048 nodes`; B focused is
`15 files / 232 nodes`, with the Shell CSS suite separately `10/10`.
Base-to-final raw accounting is `+106/-2`, net `+104`.

Required B node IDs:

```text
visible literal scanner > ignores calibration transport operands while retaining visible calibration copy
visible literal scanner > ignores broker day-gap comparison operands while retaining activity labels
visible literal scanner > requires empty debt global src scope and exact allowlist arithmetic
visible literal scanner > keeps real presenter returns visible after machine-operand narrowing
portfolio presentation > maps every Portfolio operation outcome in both locales
portfolio presentation > maps validation and empty states without parsing backend text
portfolio presentation > exposes only reviewed safe ApiError fields in Developer Mode
portfolio presentation > omits arbitrary error details in normal mode
portfolio presentation > maps activity field IDs to local labels
portfolio presentation > keeps unknown stable IDs visible and distinguishable
portfolio presentation > selects reviewed one and other count copy
portfolio presentation > renders late outcomes in the active locale
portfolio presentation > preserves source user and measured values
portfolio presentation > uses only static Portfolio resource selectors
portfolio presentation > covers both locales for every closed operation branch
portfolio presentation > never returns a raw resource key
Holdings > renders complete holdings chrome in both locales
Holdings > replaces only the example ticker placeholder and preserves real symbols
Holdings > localizes table headers without changing sorting or row identity
Holdings > localizes edit close and validation outcomes by operation
Holdings > preserves open editor draft and focus across locale changes
Holdings > renders an in-flight mutation outcome in the active locale
Holdings > preserves archived filters selection and scroll position
Holdings > omits raw mutation details in normal mode
Portfolio activity > renders filters statuses and expanded rows in both locales
Portfolio activity > maps activity field IDs without printing schema names
Portfolio activity > preserves execution commission and source values
Portfolio activity > localizes count grammar without changing pagination
Portfolio activity > preserves expansion filters focus and scroll across locale changes
Portfolio activity > renders late load outcomes in the active locale
Portfolio activity > omits raw details in normal mode
Portfolio capture > renders schedule run and review chrome in both locales
Portfolio capture > preserves a poll issue published while settings save is pending without raw detail
Portfolio capture > retries initial status failure on idle cadence with semantic copy
Portfolio capture > announces a terminal start outcome only once without raw detail
Portfolio capture > keeps terminal and settings race ordering unchanged
Portfolio capture > preserves dirty controls and focus across locale changes
Portfolio capture > renders late polling outcomes in the active locale
Portfolio account overview > renders account and value headers in both locales
Portfolio account overview > preserves account currency and measured values
Portfolio account overview > localizes loading empty partial and error states
Portfolio account overview > preserves selected account across locale changes
Portfolio account overview > omits raw diagnostics in normal mode
Portfolio recent activity > renders recent activity chrome in both locales
Portfolio recent activity > localizes count grammar without changing rows
Portfolio recent activity > preserves source values and identifiers
Portfolio recent activity > preserves state across locale changes
App shell > stores sidecar failures as structured System outcomes without raw Error.message
App shell > renders System sidecar copy from the system namespace in both locales
App shell > shows only reviewed sidecar facts in Developer Mode
App shell > preserves the active view focus and status state across locale changes
App shell > issues only the locale preference PUT while System copy changes
DataTable > localizes the action heading without changing columns or cells
DataTable > localizes the row action accessible name with source values intact
DataTable > reacts to locale changes without remounting rows
overlays > localizes the built-in ConfirmDialog cancel label
overlays > preserves caller-owned labels focus and keyboard behavior across locale changes
MarkdownView > localizes blocked-image fallback chrome
MarkdownView > preserves Markdown source and rendered source text across locale changes
i18n foundation boundaries > closes remaining localization with global src scope and exact empty-debt arithmetic
```

All pre-existing tests not named for removal retain their node IDs. Assertions
may evolve only where this plan explicitly changes the contract.

---

## File Map

### Tranche A product and resources

**Modify**

- `apps/arkscope-web/src/Research.tsx`
- `apps/arkscope-web/src/ResearchHistoryDrawer.tsx`
- `apps/arkscope-web/src/ResearchEvidenceDrawer.tsx`
- `apps/arkscope-web/src/ResearchRunProgress.tsx`
- `apps/arkscope-web/src/researchErrors.ts`
- `apps/arkscope-web/src/researchSelection.ts`
- `apps/arkscope-web/src/researchReducer.ts`
- `apps/arkscope-web/src/modelRoutingUx.ts`
- `apps/arkscope-web/src/modelPicker.ts`
- `apps/arkscope-web/src/settings/ModelRoutingSection.tsx`
- `apps/arkscope-web/src/settings/ProviderSection.tsx`
- `apps/arkscope-web/src/settings/settingsBackendCopy.ts`
- `apps/arkscope-web/src/styles.css`, only the exact runtime-reviewed
  deviations 1 and 4 hunks
- `apps/arkscope-web/src/i18n/resources.ts`
- `apps/arkscope-web/src/i18n/resources/en/common.ts`
- `apps/arkscope-web/src/i18n/resources/en/settings.ts`
- `apps/arkscope-web/src/i18n/resources/en/research.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/common.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/settings.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/research.ts`

**Create**

- `apps/arkscope-web/src/i18n/researchPresentation.ts`
- `apps/arkscope-web/src/i18n/researchPresentation.test.ts`
- `apps/arkscope-web/src/ResearchEvidenceDrawer.test.tsx`
- `apps/arkscope-web/src/ResearchRunProgress.test.tsx`
- `apps/arkscope-web/scripts/i18n/fixtures/table-headers.tsx.txt`
- `apps/arkscope-web/scripts/i18n/fixtures/tuple-columns.tsx.txt`
- `apps/arkscope-web/scripts/i18n/fixtures/machine-operands.tsx.txt`

**Modify tests/tooling**

- `apps/arkscope-web/src/i18n/visibleLiteralScanner.test.ts`
- `apps/arkscope-web/src/i18n/resources.test.ts`
- `apps/arkscope-web/src/modelRoutingUx.test.ts`
- `apps/arkscope-web/src/modelPicker.test.ts`
- `apps/arkscope-web/src/shell/ShellCss.test.ts`, only the one deviation node
- every existing A focused test only as its owning contract requires
- `apps/arkscope-web/scripts/i18n/visible-literal-scanner.mjs`
- `apps/arkscope-web/scripts/i18n/visible-literal-debt.json`
- `apps/arkscope-web/scripts/i18n/migrated-scopes.json`

### Tranche B product and resources

**Modify**

- `apps/arkscope-web/src/Holdings.tsx`
- `apps/arkscope-web/src/PortfolioActivity.tsx`
- `apps/arkscope-web/src/PortfolioCapturePanel.tsx`
- `apps/arkscope-web/src/PortfolioAccountOverview.tsx`
- `apps/arkscope-web/src/PortfolioRecentActivity.tsx`
- `apps/arkscope-web/src/Dashboard.tsx`
- `apps/arkscope-web/src/App.tsx`, only the three reviewed seams
- `apps/arkscope-web/src/ui/DataTable.tsx`
- `apps/arkscope-web/src/ui/ConfirmDialog.tsx`
- `apps/arkscope-web/src/MarkdownView.tsx`
- `apps/arkscope-web/src/i18n/resources.ts`
- `apps/arkscope-web/src/i18n/resources/en/common.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/common.ts`

**Create**

- `apps/arkscope-web/src/i18n/portfolioPresentation.ts`
- `apps/arkscope-web/src/i18n/portfolioPresentation.test.ts`
- `apps/arkscope-web/src/i18n/systemPresentation.ts`
- `apps/arkscope-web/src/i18n/resources/en/portfolio.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/portfolio.ts`
- `apps/arkscope-web/src/i18n/resources/en/system.ts`
- `apps/arkscope-web/src/i18n/resources/zh-Hant/system.ts`
- `apps/arkscope-web/scripts/i18n/fixtures/portfolio-machine-operands.tsx.txt`
- `docs/design/I18N_FORMATTER_INVENTORY.md`

**Dedicated deletion commit**

- Delete `apps/arkscope-web/src/settings/legacy/AppRecordsSection.tsx`.
- Modify `apps/arkscope-web/src/api.ts` only to remove:
  `AppRecordsTablePreview`, `AppRecordsMigrationPreview`,
  `AppRecordsMigrationResult`, `previewAppRecordsMigration`, and
  `applyAppRecordsMigration`.

**Modify tests/tooling**

- all B focused tests named above
- `apps/arkscope-web/src/i18n/visibleLiteralScanner.test.ts`
- `apps/arkscope-web/src/i18n/foundationBoundaries.test.ts`
- `apps/arkscope-web/scripts/i18n/visible-literal-scanner.mjs`
- `apps/arkscope-web/scripts/i18n/visible-literal-debt.json`
- `apps/arkscope-web/scripts/i18n/migrated-scopes.json`

No other product path is owned. Discovering another reachable remaining
surface is a stop-and-amend.

---

## Protected Boundaries

Compare all protected paths to product base `93cda668`, not the docs tip.

1. Root `src/`, `data_sources/`, and `tests/` are byte-identical.
2. Research prompts, agent/provider backend code, DTOs, and schemas are
   byte-identical.
3. `apps/arkscope-desktop/` and `extensions/` are byte-identical.
4. Root and app `package.json`, `package-lock.json`, and desktop package files
   are byte-identical.
5. `shell/shell.css`, `ui/primitives.css`, and `settings/settings.css` are
   byte-identical. Runtime-reviewed deviations 1 and 4 replace the `styles.css`
   byte gate only with the exact `.research-conversation-title` wrapping hunk
   and the class-scoped Holdings account-filter bounds; no other `styles.css`
   byte may change.
6. `api.ts` differs only by the exact five AppRecords export removals.
7. Task 9 commit `b6ea67b6` has zero formatter product or formatter-test diff
   from its Task 8 input `6a076db3`. From product base to final, the only
   formatter-helper exception is in `PortfolioActivity.tsx`: the three named
   helpers accept caller-localized unknown copy, preserve the same
   null/non-finite predicates and `Intl.NumberFormat` options, and preserve the
   same source-currency append behavior. No other formatter helper or formatter
   output expectation may differ.
8. Tranche B may not change any frozen Tranche A product owner.
9. The public selector remains absent and production `ui_locale` remains
   untouched by verification.

---

## Task 0: Plan Clearance, Worktree, and Baseline Reproduction

**Files:** docs only until the plan is independently cleared.

- [x] **Step 1: Incorporate independent plan-review findings without product edits**

  Update this plan's review-resolution section, exact ledgers, and status. A
  finding that changes product scope, scanner arithmetic, resource counts, or
  test-node accounting requires re-review rather than an informal note.

- [x] **Step 2: Record the clearance commit**

  Commit the docs-only cleared plan and record its full hash as
  `PLAN_REVIEW_CLEARANCE_COMMIT`. Product implementation starts from that docs
  commit, while behavioral A/B remains anchored to `93cda668`.

- [x] **Step 3: Create the isolated implementation worktree**

  Use `superpowers:using-git-worktrees` and create branch
  `codex/i18n-4-5-remaining-surfaces` from the clearance commit. Do not edit
  the main worktree. Mount the same existing root `node_modules` into any
  virgin archive used for comparison; do not run dependency installation that
  changes lockfiles.

- [x] **Step 4: Reproduce exact baselines from a clean tree**

  Run:

  ```bash
  cd apps/arkscope-web
  npx vitest list
  npm test
  npm run typecheck
  npm run build
  npm run check:i18n-literals
  npm run check:i18n-literals
  cd ../..
  pytest --collect-only -q
  git diff --check
  ```

  Require frontend `90/944`, backend collect `4621`, scanner
  `703/656/637/20` twice with byte-identical output, 39 scopes, and the exact
  A focused `21/263`. Record SHA-256 of scanner output, debt, allowlist, scopes,
  and sorted frontend node list. Build may emit only the existing chunk-size
  warning.

- [x] **Step 5: Reconcile source and debt inventories**

  Prove the ceiling partition `230+374+30+3=637`, source `636/667`, A
  `230/242`, and the exact 22 scanner-missed values. Any drift stops work.

- [x] **Step 6: Capture protected-byte baselines**

  Hash or archive the exact protected groups above. Confirm the product
  selector is absent and a read-only production query still finds no
  `profile_settings.ui_locale`. Do not write production data.

- [x] **Step 7: Commit Task 0 evidence docs only**

  ```bash
  git add docs/superpowers/plans/2026-07-24-i18n-4-5-remaining-surfaces.md
  git commit -m "docs: ground remaining i18n implementation"
  ```

---

## Task 1: RED Scanner Hardening and Shared Model/Resource Authority

**Files:** scanner fixtures/tool/tests/manifests, resources, model owner,
mechanical Settings consumers, and their tests.

- [ ] **Step 1: Add scanner RED fixtures first**

  Add the four A scanner nodes. Require direct `header` and reviewed static
  tuple columns to be detected; require model reason and Research completion
  operands to be ignored while visible presenter returns remain detected.
  Run only `visibleLiteralScanner.test.ts`; all four new nodes must fail against
  the old scanner for the reviewed reasons.

- [ ] **Step 2: Implement the narrow AST coverage**

  Add `header` to visible property handling and recognize only the audited
  tuple-column form. Do not generalize all tuple strings. Add explicit
  classification for the six A machine operands. Require exactly 22 new
  current candidates and no other discovery.

- [ ] **Step 3: Put the 22 B labels into interim debt**

  Add the exact 22 signatures to the A checkpoint debt ceiling while removing
  all 230 A-owned signatures as they migrate. Do not touch the allowlist.

- [ ] **Step 4: Add resource and model-owner RED assertions**

  Replace the old exact-count node with the generic inventory node and add the
  shared-owner node. Strengthen `modelRoutingUx.test.ts` to require every group,
  reason, auth mode, thinking mode, and compatibility context in both locales.
  Strengthen `modelPicker.test.ts` in place so ID/baseLabel/compatibility remain
  structural and decorated-label parsing is impossible.

- [ ] **Step 5: Move model copy to Common**

  Add exactly 24 Common leaves, remove exactly 23 Settings duplicates, and
  evolve `modelRoutingUx.ts`/`modelPicker.ts` to return semantic structure plus
  namespace-typed display. Settings and Research pass a reactive Common
  translator. Preserve current zh-Hant and I18N-2 English bytes exactly.

- [ ] **Step 6: Add Research resource skeleton**

  Add the 202 reviewed Research leaves in both locales. No source value,
  generated text, dynamic identifier, or formatter output belongs in the
  resources. Require A resource counts `56/37/679/207/401`.

- [ ] **Step 7: Run focused gates and commit**

  Run scanner, resource, model-picker, model-routing, Settings routing/provider,
  and typecheck tests. Require no allowlist change and no backend/CSS/package
  diff.

  ```bash
  git add apps/arkscope-web/src/i18n apps/arkscope-web/src/modelPicker.ts \
    apps/arkscope-web/src/modelPicker.test.ts \
    apps/arkscope-web/src/modelRoutingUx.ts \
    apps/arkscope-web/src/modelRoutingUx.test.ts \
    apps/arkscope-web/src/settings apps/arkscope-web/scripts/i18n
  git commit -m "feat: establish remaining i18n authorities"
  ```

---

## Task 2: RED Research Semantic Presentation

**Files:** `i18n/researchPresentation.ts`, `researchErrors.ts`,
`researchSelection.ts`, `researchReducer.ts`, and corresponding tests.

- [ ] **Step 1: Add the ten pure-presenter RED nodes**

  Cover both locales, all closed semantic branches, unknown stable IDs, no raw
  resource keys, no source/generated content, and unchanged identifiers and
  metric values. Presenter functions receive a Research-typed translator; no
  singleton lookup is allowed.

- [ ] **Step 2: Evolve Research error tests in place**

  Keep existing error-node IDs. Change expected title/detail/action output to
  translator-backed values, preserve navigation targets and `preservePartial`,
  and prove no `.message` parsing. Developer detail retains only its already
  reviewed sanitized boundary; this unit does not broaden it.

- [ ] **Step 3: Implement semantic presenters**

  Move status, selection, quota, provenance, suggested-prompt, empty-response,
  disconnect, and progress chrome behind exhaustive typed switches. Keep
  Research work, transcript content, tool payloads, and generated answer text
  outside the presenter.

- [ ] **Step 4: Store semantic outcomes, not localized text**

  Catch/reducer paths keep code/operation/structured facts. Late asynchronous
  completion is rendered in the then-current locale. Do not change reducer
  sequencing, replay, run IDs, persistence, or Provider behavior.

- [ ] **Step 5: Verify and commit**

  Run the new presenter suite plus existing `researchErrors`, reducer,
  selection, replay, model, and shell-work tests. Typecheck must prove namespace
  translators cannot be crossed.

  ```bash
  git add apps/arkscope-web/src/i18n/researchPresentation.ts \
    apps/arkscope-web/src/i18n/researchPresentation.test.ts \
    apps/arkscope-web/src/researchErrors.ts \
    apps/arkscope-web/src/researchErrors.test.tsx \
    apps/arkscope-web/src/researchSelection.ts \
    apps/arkscope-web/src/researchSelection.test.ts \
    apps/arkscope-web/src/researchReducer.ts \
    apps/arkscope-web/src/researchReducer.test.ts
  git commit -m "feat: localize Research semantic outcomes"
  ```

---

## Task 3: RED Research Surfaces and Freeze `TRANCHE_A_TIP`

**Files:** four Research surfaces, mounted tests, A scanner/resource evidence,
and this plan ledger.

- [x] **Step 1: Add all mounted Research RED nodes**

  Add the exact eight Workspace, six History, seven Evidence, and five Progress
  node IDs in the ledger. First prove English chrome is not wired while source
  text, drafts, and generated output remain unchanged.

- [x] **Step 2: Wire reactive namespace subscriptions**

  Each surface uses reactive `research` and, where needed, `common` hooks and
  passes namespace-typed translators to pure presenters. Do not read the
  i18next singleton inside a presenter. Every memoized display value includes
  translator/locale dependencies.

- [x] **Step 3: Preserve live state across locale changes**

  Prove no locale-keyed remount, no data refetch, and no run replay. Seed a
  draft and node marker, focus a control, open History/Evidence, set a practical
  scroll anchor, and keep active progress. Change locale through the controller
  test seam and require the same nodes/state/focus while chrome changes.

- [x] **Step 4: Preserve source and generated bytes**

  Plant thread titles, evidence, tool payloads, prompt draft, and generated
  answer text in both languages and mixed text. Require exact byte preservation
  before/after locale switching. Suggested prompts localize only before user
  selection; once copied into the draft, they become user content and freeze.

- [x] **Step 5: Run A focused/full/static gates**

  Require A focused `24/305`, Shell CSS `9/9`, full frontend `93/987`, typecheck/build, resource
  target `56/37/679/207/401`, and scanner exactly:

  ```text
  candidates             483
  current signatures     448
  manifest debt          429
  allowlist               20
  manifest occurrences   448
  migrated scopes         48
  ```

  Run scanner twice and hash output/debt/allowlist/scopes. The allowlist must be
  byte-identical to base.

- [x] **Step 6: Run the A bilingual worst-composition matrix**

  In an isolated fake-backed app, run `zh-Hant` and `en` at `1440x900`,
  `960x768`, and `390x844`. At every width mount one Research composition that
  simultaneously contains:

  - the longest reviewed English workspace/history/evidence chrome;
  - an error banner;
  - active run progress;
  - long planted source/evidence/generated content; and
  - an open History or Evidence drawer where the viewport permits it.

  Also exercise locale switching during draft and in-flight work. Record
  viewport, bar/panel dimensions, request counts, focused node, source hashes,
  and screenshot path. Require zero document overflow, clipping, overlap,
  truncation, replay, or source mutation. Separate happy-path screenshots do
  not satisfy this advisory.

- [x] **Step 7: Commit the A product checkpoint**

  Commit all A product/tests/manifests. Then record the resulting full hash as
  `TRANCHE_A_TIP`; it must descend from `93cda668`.

  ```bash
  git add apps/arkscope-web/src apps/arkscope-web/scripts/i18n
  git commit -m "feat: complete I18N-4 Research tranche"
  ```

- [x] **Step 8: Record A evidence without changing A product bytes**

  Add the full hash and all A evidence to this plan, commit docs only, and run
  `git diff TRANCHE_A_TIP --` on every A product owner to prove the evidence
  commit changed no product byte.

  ```bash
  git add docs/superpowers/plans/2026-07-24-i18n-4-5-remaining-surfaces.md
  git commit -m "docs: record I18N-4 tranche checkpoint"
  ```

From this point onward, any B edit to an A product owner is a stop condition.

---

## Task 4: RED Portfolio Resources and Semantic Presenter

**Files:** Portfolio resources, `i18n/portfolioPresentation.ts`, tests, and B
scanner fixture/test evolution.

- [ ] **Step 1: Add B scanner RED nodes**

  Add the four named nodes for calibration transport operands,
  `broker_day_gap`, final empty-debt arithmetic, and retained visible presenter
  detection. Do not weaken the A fixtures or alter A checkpoint evidence.

- [ ] **Step 2: Add the twelve presenter RED nodes**

  Cover every closed Portfolio operation/state, both locales, list/count
  grammar, unknown stable IDs, schema-field display mapping, late outcomes,
  normal privacy, bounded Developer facts, source preservation, static keys,
  and zero raw key output.

- [ ] **Step 3: Add Portfolio resources**

  Create both `portfolio.ts` files with exactly 374 non-empty leaves and
  identical key sets. Preserve all unnamed zh-Hant bytes. Add the five Common
  leaves only when their owning primitive/placeholder is wired in Tasks 5-7.

- [ ] **Step 4: Implement the pure presenter**

  Use exhaustive switches over semantic operation/state/field IDs. Read
  structured `ApiError` fields, never `.message`. Preserve source/user values,
  stable IDs, and numbers. Unknown IDs remain distinguishable without exposing
  schema field names as normal labels.

- [ ] **Step 5: Verify and commit**

  Run presenter, scanner, resource, and typecheck tests. Resources may be
  temporarily unused but must remain fully typed and static.

  ```bash
  git add apps/arkscope-web/src/i18n apps/arkscope-web/scripts/i18n
  git commit -m "feat: establish Portfolio localization boundary"
  ```

---

## Task 5: RED Holdings and Portfolio Activity

**Files:** `Holdings.tsx`, `PortfolioActivity.tsx`, and their tests.

- [ ] **Step 1: Add the eight Holdings and seven Activity RED nodes**

  Exercise complete bilingual chrome, table headers, operations, filters,
  expanded rows, count grammar, in-flight outcomes, locale-state preservation,
  source values, and normal-mode privacy.

- [ ] **Step 2: Migrate Holdings chrome**

  Subscribe reactively to `portfolio` and `common`. Replace only the static
  example placeholder `NVDA` with Common `Ticker`; never rewrite real symbols.
  Install that one Common leaf here; Task 7 installs the remaining four Common
  residual leaves. Preserve the fixed-English `Holdings` eyebrow through the
  direct resource owner from runtime-reviewed deviation 2.
  Preserve sort/filter/row keys, editor ownership, archived behavior, and all
  formatter output.

- [ ] **Step 3: Migrate Activity chrome**

  Present field/state/operation IDs through the Portfolio owner. Keep
  execution, commission, currency, run, timestamp, and source values original.
  Classify both `broker_day_gap` machine operands without changing their logic.

- [ ] **Step 4: Prove locale-switch continuity**

  Seed an open Holdings editor and expanded Activity row with filters and a
  scroll anchor. Switch locale and require same node identity, draft, focus,
  expansion, filters, and no data request delta.

- [ ] **Step 5: Verify and commit**

  Run both mounted suites, presenter/resources/scanner, typecheck, and the
  protected formatter expectations. No CSS change is allowed here.

  ```bash
  git add apps/arkscope-web/src/Holdings.tsx \
    apps/arkscope-web/src/Holdings.test.tsx \
    apps/arkscope-web/src/PortfolioActivity.tsx \
    apps/arkscope-web/src/PortfolioActivity.test.tsx
  git commit -m "feat: localize Holdings and Portfolio activity"
  ```

---

## Task 6: RED Capture, Account Overview, and Recent Activity

**Files:** three Portfolio surfaces and their tests.

- [ ] **Step 1: Add the seven, five, and four RED nodes**

  Cover bilingual chrome, scanner-missed headers, count grammar, dirty state,
  active polling, race ordering, in-flight locale changes, source values, and
  normal-mode privacy.

- [ ] **Step 2: Evolve existing raw-detail nodes in place**

  Keep these existing node IDs while replacing raw-detail expectations with
  semantic localized copy and explicit absence of the planted detail:

  ```text
  preserves_a_poll_issue_published_while_settings_save_is_pending
  retries_initial_status_failure_on_the_idle_cadence
  announces_a_terminal_start_detail_only_once
  shows_next_due_and_recent_runs_without_raw_account_id
  ```

  The old strings `transient poll failure`, `sidecar warming up`, and
  `IBKR provider configuration is incomplete` must not appear in normal mode.
  Existing poll/save/start race sequencing remains unchanged.

- [ ] **Step 3: Migrate Capture**

  Store semantic outcomes and facts, render through the Portfolio presenter,
  and keep polling cadence, dirty controls, current-run ordering, terminal
  announcement dedupe, and Provider behavior intact.

- [ ] **Step 4: Migrate Account and Recent Activity**

  Wire the exact direct/tuple header inventory to resources. Preserve account,
  currency, measured values, row order, and time/number formatter output.

- [ ] **Step 5: Verify and commit**

  Run all three suites plus Portfolio presenter/resources/scanner/typecheck.
  Require no formatter or CSS diff.

  ```bash
  git add apps/arkscope-web/src/PortfolioCapturePanel.tsx \
    apps/arkscope-web/src/PortfolioCapturePanel.test.tsx \
    apps/arkscope-web/src/PortfolioAccountOverview.tsx \
    apps/arkscope-web/src/PortfolioAccountOverview.test.tsx \
    apps/arkscope-web/src/PortfolioRecentActivity.tsx \
    apps/arkscope-web/src/PortfolioRecentActivity.test.tsx
  git commit -m "feat: localize Portfolio capture and account views"
  ```

---

## Task 7: RED System and Common Residual Chrome

**Files:** `App.tsx`, `Dashboard.tsx`, common primitives, System/Common
resources, `i18n/systemPresentation.ts`, and owning tests.

- [ ] **Step 1: Add System and common RED nodes**

  Add five AppShell, three DataTable, two overlay, and two Markdown nodes. Plant
  a hostile sidecar `Error.message` and prove App state contains none of it,
  normal UI exposes none, and Developer Mode shows only reviewed safe facts.
  The existing planned bilingual System node must mount Dashboard in both
  locales and assert all 20 `system` leaves, not only the six-ish sidecar
  states. Its ID stays unchanged and no node is added.

- [ ] **Step 2: Add System resources with explicit ownership**

  Create exactly 20 leaves per locale under `system`. Sidecar loading/error/
  retry/diagnostic chrome lives here even though `App.tsx` stores state and
  `Dashboard.tsx` renders it. Do not put it in Shell, Explore, or Common.

- [ ] **Step 3: Replace raw sidecar state**

  Replace `{kind: "error", message: string}` with a structured safe outcome
  containing only reviewed status/code/route facts. Unknown errors become a
  generic semantic failure. Never retain raw `.message` in React state.

- [ ] **Step 4: Migrate Dashboard**

  Subscribe to `system`, present structured outcomes, and preserve health
  polling, retry, navigation, Developer Mode ownership, and all measured values.
  `System / Health` remains the canonical mixed label.

- [ ] **Step 5: Migrate remaining common primitive defaults**

  Add the remaining four Common leaves and wire DataTable, ConfirmDialog, and
  Markdown. The Holdings `Ticker` placeholder leaf was installed with its first
  consumer in Task 5. Caller-supplied labels remain caller-owned; Markdown
  source and DataTable cells never enter resources.

- [ ] **Step 6: Prove locale-switch request boundary**

  Locale change may issue the preference PUT only. Require same active view,
  node identity, focus, status state, dialog state, table rows, and zero data
  refetch. Late structured outcomes render in the active locale.

- [ ] **Step 7: Verify and commit**

  Run AppShell, common primitive, resource, scanner, typecheck, and no-PG smoke.
  Require System `20`, Common `61`, and no App raw-message storage.

  ```bash
  git add apps/arkscope-web/src/App.tsx \
    apps/arkscope-web/src/Dashboard.tsx \
    apps/arkscope-web/src/AppShell.test.tsx \
    apps/arkscope-web/src/ui \
    apps/arkscope-web/src/MarkdownView.tsx \
    apps/arkscope-web/src/MarkdownView.test.ts \
    apps/arkscope-web/src/i18n
  git commit -m "feat: localize System and residual common chrome"
  ```

---

## Task 8: Retire AppRecords Frontend in One Commit

**Files:** delete the legacy component; modify `api.ts` only for five symbols;
close the named docs backlog.

- [ ] **Step 1: Prove zero production consumers before deletion**

  Use `rg` for `AppRecordsSection` and each of the five exports. Require the
  component to have no production consumer and every export consumer to be
  inside that component only. A real consumer is a stop condition.

- [ ] **Step 2: Delete exactly the reviewed frontend surface**

  Remove `AppRecordsSection.tsx`, its 30 debt signatures, and only the five
  named `api.ts` symbols. Do not touch routes, backend storage, migration code,
  archive tables, or offline tools.

- [ ] **Step 3: Prove the exact API exception**

  Diff `api.ts` against `93cda668`; require exactly three interface and two
  wrapper removals, no other hunk. Run a zero-reference grep for all six names.

- [ ] **Step 4: Close the old frontend backlog and commit alone**

  Update only the standing backlog statement that says App Records panel full
  removal is deferred. Commit product deletion and that narrow status sync in
  one dedicated commit:

  ```bash
  git add apps/arkscope-web/src/api.ts \
    apps/arkscope-web/src/settings/legacy/AppRecordsSection.tsx \
    docs/design/PROJECT_PRIORITY_MAP.md
  git commit -m "refactor: retire AppRecords frontend"
  ```

- [ ] **Step 5: Re-run frontend and backend byte gates**

  Frontend compiles/tests without the exports. Backend App Records paths remain
  byte-identical to base. No test node is removed by this commit.

---

## Task 9: Close Scanner Debt and Record Formatter Inventory

**Files:** scanner tool/tests/manifests, foundation and resource boundaries,
formatter inventory doc, and this plan evidence ledger.

- [ ] **Step 1: Adjudicate the three API records**

  Two current transport/presenter literals must either migrate or be correctly
  reclassified by the reviewed scanner semantics; the one stale Decision 65
  row is removed. Do not change remaining API transport behavior and do not add
  an allowlist entry.

- [ ] **Step 2: Empty the debt manifest**

  Remove every migrated signature and set migrated scopes exactly to
  `["src/**"]`. Require the scanner's durable assertions, not only CLI success:

  ```text
  candidateCount       36
  signatureCount       20
  debtSignatureCount    0
  allowlistCount        20
  candidateCount == sum(allowlist occurrence counts)
  signatureCount == allowlist entry count
  ```

  The allowlist file remains byte-identical to base. Run twice and require
  byte-identical output.

- [ ] **Step 3: Add the global foundation pin**

  Evolve the existing migrated-path contract in place and add the named final
  arithmetic node. Require `src/**` and reject any narrower or extra scope.

- [ ] **Step 4: Write the formatter inventory without changing formatters**

  Create `docs/design/I18N_FORMATTER_INVENTORY.md` with columns for owner,
  current behavior, locale dependency, consumers, and future decision owner.
  At minimum inventory:

  - `timeDisplay.formatSystemTimestamp` and `formatMarketTimestamp`;
  - Research History's local `Intl.DateTimeFormat`;
  - Evidence timestamps and numeric `toLocaleString`;
  - Holdings `formatNum`/`formatMaybe`;
  - Activity `formatNumber`/`formatAmount`/`formatUnknown`;
  - Capture `formatLocalTime`/`formatReviewMetric`;
  - Account Overview amount formatting;
  - Recent Activity number formatting; and
  - Dashboard `toLocaleTimeString`.

  Record current outputs only. Task 9 must have zero formatter product or
  formatter-test diff against the Task 8 input. The base-to-final review must
  admit only the already-reviewed `PortfolioActivity` unknown-fallback
  localization described in Locked Decision 16; another formatter-helper
  change or any formatter output-expectation change is a stop condition.

- [ ] **Step 5: Verify final resource and scanner arithmetic**

  Require seven namespaces and exact per-locale counts
  `61/37/679/207/401/374/20`, total `1779`, equal key sets, no empty leaves,
  no dynamic keys, scanner `36/20/0/20`, and one global scope.

- [ ] **Step 6: Commit closure**

  ```bash
  git add apps/arkscope-web/scripts/i18n \
    apps/arkscope-web/src/i18n/foundationBoundaries.test.ts \
    apps/arkscope-web/src/i18n/resources.test.ts \
    apps/arkscope-web/src/i18n/visibleLiteralScanner.test.ts \
    docs/design/I18N_FORMATTER_INVENTORY.md
  git commit -m "test: close application localization debt"
  ```

---

## Task 10: Canonical, Runtime, and Review-Ready Evidence

**Files:** this plan evidence ledger only unless a reviewed stop-and-amend is
required. Freeze product before requesting review.

- [x] **Step 1: Prove two exact frontend node comparisons**

  In virgin archives with the same hoisted `node_modules`, list and sort test
  nodes for base, `TRANCHE_A_TIP`, and final product tip. Require:

  ```text
  base -> A       +45/-2    90/944 -> 93/987
  A -> final      +61/-0    93/987 -> 94/1048
  base -> final  +106/-2    net +104
  ```

  The only removals are the two named resource-count nodes. Hash each
  full/add/remove list. Any other removal/rename is a stop condition.

- [x] **Step 2: Run exact focused suites**

  Require A focused `24/305` at `TRANCHE_A_TIP` and B focused `15/232` at
  final. Also run final A focused unchanged against final to prove B did not
  regress frozen A behavior.

- [x] **Step 3: Run canonical frontend gates**

  ```bash
  cd apps/arkscope-web
  npm test
  npm run typecheck
  npm run build
  npm run check:i18n-literals
  npm run check:i18n-literals
  cd ../..
  python -m src.smoke.pg_unreachable_e2e
  git diff --check
  ```

  Require final `94/1048`, scanner `36/20/0/20`, `src/**`, no-PG
  `ok:true`/`pg_attempts:[]`, and only the existing build warning.

- [x] **Step 4: Prove protected bytes and exact exceptions**

  Compare all protected groups to `93cda668`. Require all default groups
  byte-identical, `api.ts` exactly the five-symbol deletion, and no B diff in
  frozen A product owners. Compare Task 9 `b6ea67b6` to Task 8 input
  `6a076db3` and require zero formatter product/test diff. For base-to-final,
  inspect formatter helper bodies and expectations: permit only
  `PortfolioActivity.formatNumber(value, unknown)`,
  `formatAmount(value, currency, unknown)`, and
  `formatUnknown(value, unknownLabel)` replacing fixed `未知` fallback copy
  with the caller-supplied localized value. Their finite checks,
  `Intl.NumberFormat(undefined, { maximumFractionDigits: 4 })`, currency-code
  append, boolean/JSON handling, and tested numeric/date output remain
  unchanged; any other formatter-specific delta is a stop condition. Hash the
  scanner tool at A and final so final narrowing cannot erase A evidence.

- [x] **Step 5: Run Tranche B bilingual responsive matrix**

  Use an isolated fake-backed sidecar/Vite/CDP environment, temporary DBs,
  scheduler disabled, and no production profile or paid Provider. Run Holdings,
  Portfolio Activity/Capture/Account/Recent, and Dashboard/System in both
  locales at `1440x900`, `1024x768`, `961x768`, `960x768`, `959x768`, and
  `390x844`.

  Every surface must show its worst credible composition: longest English
  chrome, full table/filter density, error banner, open menu/dialog or expanded
  row, and dirty/active state where applicable. Require zero document overflow,
  clipping, overlap, truncation, or source mutation. If any surface fails at
  any width, stop for a reviewed CSS deviation and rerun the complete matrix;
  a 960 failure additionally requires that surface's `959/960/961` three-run
  evidence after repair.

- [x] **Step 6: Exercise locale-switch preservation and privacy**

  Across A and B prove same node identity, draft/input/filter/expansion/dialog/
  drawer state, focus, practical scroll anchor, active polling/run state, and
  no data request delta except locale PUT. Late outcomes use the active locale.
  Plant token, traceback, sqlite, path, IP, HTML, and long exception text;
  normal mode shows none and Developer Mode shows only reviewed structured
  facts.

- [x] **Step 7: Exercise AppRecords absence and selector absence**

  Prove no reachable UI/import/API client symbol remains, backend/offline
  capability is untouched, and no locale selector/placeholder appears.
  Production `ui_locale` remains absent before and after normal zh-Hant smoke.

- [x] **Step 8: Inspect screenshots and clean isolated artifacts**

  Record locale, viewport, surface, state, dimensions, request counts, focused
  node, source hashes, and screenshot paths. Inspect every image. Stop all
  processes, prove ports refuse connection, and remove temporary profiles, DBs,
  source-bearing screenshots, and harness files.

- [x] **Step 9: Freeze product and commit evidence docs only**

  Record `TRANCHE_A_TIP`, final product tip, all hashes/counts, runtime results,
  and deviations in this plan. Commit docs only after product freeze:

  ```bash
  git add docs/superpowers/plans/2026-07-24-i18n-4-5-remaining-surfaces.md
  git commit -m "docs: record remaining i18n verification"
  ```

  No product change after the frozen tip inherits independent review.

### Task 10 review-ready evidence

- Final product/test/tooling tip is
  `20666d33f440fb06ec3e85ce2367674db3685ef9`. The preceding docs-only
  deviation authority is `dd6a37fc`; no product byte changed after the final
  runtime and canonical gates below.
- Test lists were normalized as
  `relative-test-path<TAB>full-node-name<LF>`. Full-list SHA-256 values are:
  base `d3cbae7499f3a72ee490e0e20278463d39ba354deb9d8fa151b79e3bdd0841a7`
  (`944`), A
  `79da014adc02c094605ecb6b069e0333be87ebc3b2c54bca86f421f7ee8f9b3a`
  (`987`), and final
  `b26eee9d7c22acc3e34d0e7922737f426dcf793cf4e4108bd92c752d24f78775`
  (`1048`). Base-to-A is exactly `+45/-2` with add hash
  `97bfbe88584ed369d986134784d4180d079f74ec9c39c5f29a75b5cc208c88dd`
  and remove hash
  `a7d1d1be8dcd2f8ab04e7ead892a6b9c4f3ea3930e52cd9e03c8ec19497ada8b`.
  A-to-final is exactly `+61/-0`, add hash
  `1b2fe1dff62dd2e30c9e681d029cb806052585d2505872c36a59bba9c6502ea4`,
  and the empty-list hash
  `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`.
  Base-to-final is `+106/-2`, add hash
  `496e3546ba73212401e24dac1813ab807f3484fb3607e16820da3047fc679f13`.
  The only removals remain the two reviewed resource-count nodes.
- Final focused and canonical suites are green: frozen A `24/305`, B
  `15/232`, Shell CSS `10/10`, and full frontend `94/1048`. Typecheck is
  clean. Production build is clean except for the established chunk-size
  warning. No-PG smoke reports `ok:true` and `pg_attempts:[]`.
- The final scanner ran twice with byte-identical
  `36/20/0/20`, scope `src/**`; canonical result SHA-256 is
  `f49e0a7bd73b68fc9846324f2ec60d881ea248b549b092b606f52395e616b85d`.
  Final manifest hashes are debt
  `d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec`,
  unchanged allowlist
  `3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212`,
  and scopes
  `02e335bebcadfba523d502a7af86a5c184d1ac024230cfec9199dd19b4416c13`.
  The scanner-tool SHA-256 evolves from A
  `130414b042edebc53558d48f51837485115c2278bfb2eaed8f2ebeb669925ed8`
  to final
  `c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da`;
  the exact diff is only B's reviewed calibration-transport and
  `broker_day_gap` machine-operand classifiers.
- Protected root backend/data/test, desktop, extension, package, Shell,
  primitive, and Settings CSS groups are byte-identical to product base.
  Frozen A product owners are byte-identical from `TRANCHE_A_TIP` to final
  except deviation 4's exact `styles.css` Holdings hunk. Base-to-final
  `styles.css` has only deviations 1 and 4; `api.ts` has only the reviewed
  three-interface/two-wrapper AppRecords deletion. All six frontend symbols
  have zero references. Task 9 versus its Task 8 input changes only scanner
  manifests/boundary tests and the formatter inventory; the only
  base-to-final formatter helper exception remains the reviewed localized
  unknown fallback in `PortfolioActivity`.
- Deviation 4 has a true RED witness: the new named Shell CSS node failed while
  the other nine passed. The first label-only contract then passed statically
  but the full runtime matrix correctly remained red, exposing the action-row
  intrinsic-width owner. The reviewed row/label/select contract produced
  `10/10` and reduced the `390x844` Holdings control from `434px` to the
  available viewport width without truncating source labels or adding a
  breakpoint/fixed width. Independent narrow deviation review returned GREEN
  with zero finding.
- Final isolated runtime evidence identifies the exact final product tip and
  records `36` cases, `72` screenshots, and `1188/1188` boolean checks across
  six surfaces, both locales, and all six viewports. It reports zero finding,
  unchanged production-profile metadata, unchanged temporary Portfolio DB
  bytes, local-only network traffic, no paid Provider request, and closed
  sidecar/Vite/CDP ports. Evidence JSON SHA-256 is
  `81525c6223b70739c5977371afc7bcbe5b6081ac64c17f34bfba7c4635341eab`;
  the sorted screenshot ledger SHA-256 is
  `386cbb10cf959c96970523fcca61a644d4478ed1aa54aeed93323a8f2fb51255`.
  Six surface contact sheets plus an enlarged `390x844` sheet were inspected;
  no clipping, overlap, truncation, incoherent reflow, or source mutation was
  observed.
- Harness failures are retained as diagnostics, not product findings: the
  first run could not resolve Vite from `/tmp`; two Activity runs exposed
  missing React mount waits; and the first complete run used transient
  error/detail nodes as reading-position anchors. Those harness defects were
  corrected without touching product. The complete run then isolated the real
  Holdings overflow and, after the reviewed fix, passed in full.
- `profile_settings.ui_locale` remains absent by immutable read-only query
  before and after the normal zh-Hant/runtime work. AppRecords UI and public
  locale selector remain absent. `git diff --check` is clean.

### Merge closeout

- Independent implementation review returned GREEN with zero findings after
  a virgin two-stage comparison from base
  `93cda66831b7202fd0dfafcc0d1c0604b07e94bd` through
  `TRANCHE_A_TIP=34ddf08f5983d523bf1bfb00ce6b06a55a76bce0` to product
  `20666d33f440fb06ec3e85ce2367674db3685ef9`. The reviewed branch/evidence
  tip `5f35e8b15517ebc130df569cc90e3fe0abdd5aef` then fast-forwarded to
  `master` with explicit user approval.
- Fresh merged-tree verification passed B focused `15/232`, Shell CSS
  `10/10`, full frontend `94/1048`, typecheck, production build, resource
  contracts, scanner `36/20/0/20` twice with global `src/**` scope, no-PG
  `ok:true` with `pg_attempts:[]`, protected-byte gates, the exact five-symbol
  `api.ts` exception, AppRecords zero references, and `git diff --check`.
  The 24-file A set collects `310` nodes at final because B adds five reviewed
  scanner/foundation nodes to shared files; all `305` frozen A-owned nodes
  remain present and green.
- The normal desktop restarted from merged `master` in zh-Hant. Research,
  Holdings, Portfolio Capture/sync records, and System / Health rendered
  cleanly while source/generated content remained original. Immutable
  read-only queries before and after the smoke found zero persisted
  `profile_settings.ui_locale` rows. The public selector remains absent.
- During that closeout smoke, a coordinate-driven harness click accidentally
  invoked the existing review-mode apply action for capture run `307` at
  `2026-07-24T15:35:06.321530+00:00`. Read-only impact analysis found exactly
  ten canonical IBKR rows matching the valid latest run, with no add, remove,
  close, quantity, average-cost, currency, identity, metadata, note, or manual
  adjustment change. All `49` complete observations from runs `258` through
  `307` carry the same ten-position set and the same non-market fields; the
  operation only refreshed mark-to-market value, unrealized P&L, and sync
  timestamps. No speculative rollback was attempted because the exact prior
  manually accepted run is not persisted and run `307` is a valid complete
  observation. Future production smoke must use semantic locators and must not
  drive mutation controls with raw coordinates.
- I18N-6 Release is now the sole next i18n unit. It owns the formatter
  decisions recorded in `I18N_FORMATTER_INVENTORY.md`, the final app-wide
  audit and bilingual matrix, controller-backed locale selector, and Design
  Kit release synchronization.

---

## Stop Conditions

Stop and return to design/plan review if any of the following occurs:

1. baseline debt does not reconcile to `230+374+30+3`, source is not
   `636/667`, or scanner hardening discovers other than the exact 22 labels;
2. test collection differs from base `90/944`, A `93/987`, final `94/1048`,
   raw `+45/-2` then `+61/-0`, the separate A/final Shell CSS `9/9 -> 10/10`,
   or either focused ledger;
3. resource counts cannot close at A `56/37/679/207/401` and final
   `61/37/679/207/401/374/20`;
4. A scanner cannot close at `483/448/429/20` with 448 debt occurrences and 48
   scopes, or final cannot close at `36/20/0/20` with `src/**`;
5. a new allowlist entry appears necessary;
6. Tranche B needs to edit a frozen A product owner outside the exact
   runtime-reviewed deviation 4 `styles.css` hunk;
7. a reachable remaining surface lies outside the file map;
8. semantic presentation requires parsing English `.message` or a missing
   backend discriminator;
9. backend, schema, DTO, prompt, Provider, data-source, desktop, extension,
   package, or dependency changes appear necessary;
10. model routing, eligibility, metadata, or decorated-label parsing behavior
    changes;
11. source/user/generated content or dynamic identifiers must enter resources;
12. locale switching requires refetch, replay, remount, or state reset;
13. AppRecords has a real production consumer or backend/offline capability
    would be removed;
14. Task 9 changes formatter product/test bytes relative to its Task 8 input,
    base-to-final contains a formatter-helper delta beyond the exact
    `PortfolioActivity` localized-unknown exception, or a numeric/date/
    rounding/timezone/currency output expectation changes;
15. unreviewed CSS is needed or copy must be truncated/shortened to fit;
16. `App.tsx` requires a change beyond structured sidecar state, reviewed
    capability wiring, or locale reactivity for that path;
17. `api.ts` differs beyond the exact five dead AppRecords exports; or
18. selector, SA extension, `/sa/feed`, Settings sanitizer, calibration refusal,
    or another deferred backlog item becomes entangled.

---

## Post-Review Integration and Closeout

Independent implementation review must return GREEN before integration. After
explicit user approval:

1. restore any protected main-worktree draft exactly;
2. fast-forward merge `codex/i18n-4-5-remaining-surfaces` only;
3. rerun merged-tree A focused `24/305`, B focused `15/232`, Shell CSS `10/10`,
   full `94/1048`,
   typecheck, build, scanner `36/20/0/20`, no-PG, resource counts, protected
   bytes, exact `api.ts` exception, and `git diff --check`;
4. restart the normal desktop in zh-Hant and smoke Research, Holdings,
   Portfolio Capture, and System without writing `profile_settings.ui_locale`;
5. flip design/plan/app-wide decision/map status to LIVE COMPLETE and record
   base/A/product/evidence/merge hashes;
6. close the absorbed standalone I18N-5 and AppRecords frontend backlog;
7. clean the feature worktree/branch after merged verification; and
8. set I18N-6 Release as the sole NEXT i18n unit while keeping all unrelated
   standing backlog entries visible.

The public locale selector remains absent until I18N-6 independently passes its
coverage and release gates.
