# ArkScope I18N-4/5 Remaining Surfaces Design

> **Status: LIVE COMPLETE — INDEPENDENT IMPLEMENTATION REVIEW GREEN;
> FAST-FORWARD MERGED THROUGH `5f35e8b1`**
>
> Written against merged `master` at `93cda66831b7` on 2026-07-24. This
> document is the bounded product, localization, retirement, and verification
> authority for the combined I18N-4/5 remaining-surfaces unit. It replaces two
> complete process cycles with one reviewed unit containing two mechanical
> tranches. This design alone did not authorize product implementation; the
> independently reviewed plan now does. The implementation passed its named
> Tranche A checkpoint, final independent review, merged-tree verification,
> and normal zh-Hant desktop smoke. Neither document authorizes the public
> locale selector, formatter behavior changes, backend changes, or deferred SA
> work; those remaining locale-release decisions belong to I18N-6.
> Independent full-document review returned GREEN with zero required changes.
> Its two plan-level advisories are incorporated in the implementation plan:
> Tranche A must exercise the worst credible Research composition in one
> mounted viewport, and sidecar structured-outcome copy belongs to the
> `system` namespace. The RED-first plan is
> [`2026-07-24-i18n-4-5-remaining-surfaces.md`](../plans/2026-07-24-i18n-4-5-remaining-surfaces.md).
> Independent plan review returned GREEN; the mounted Dashboard bilingual
> assertion was strengthened in place with zero accounting change.

## 1. Purpose and Authority

I18N-0 established locale authority, synchronous bootstrap, typed resources,
and the visible-literal scanner. I18N-1 through I18N-3 migrated Shell, common
Shell UI, Settings, and Explore. P2.8 Slice 5 shipped the redesigned Investor
Profile bilingual from birth.

The remaining literal debt is comparable in size to the already reviewed
I18N-2 Settings migration, but splitting it into separate Research and
Portfolio/System process cycles now adds more review ceremony than risk
reduction. This design therefore combines the remaining work into one branch,
one implementation plan, one independent implementation review, and one
merge/closeout while preserving a hard, reviewable midpoint:

- **Tranche A — Research and shared model-selection copy**; then
- **Tranche B — Portfolio, System, common residuals, AppRecords retirement,
  and scanner closure**.

Authority order:

1. [`ARKSCOPE_TERMINOLOGY.md`](../../design/ARKSCOPE_TERMINOLOGY.md) owns
   canonical English and Traditional Chinese terms.
2. [`2026-07-20-app-wide-i18n-decision.md`](2026-07-20-app-wide-i18n-decision.md)
   owns locale mechanics, source-content boundaries, and selector-last release.
3. The Research workspace, Portfolio/Holdings, and canonical Shell designs
   continue to own domain behavior and information architecture.
4. This document owns the combined tranche boundary, remaining-surface copy
   ownership, AppRecords frontend retirement, scanner end state, and evidence
   required before I18N-6.

### 1.1 Grounded baseline

At `93cda66831b7`:

- frontend is `90 files / 944 tests`;
- the scanner reports `703` candidate occurrences, `656` current signatures,
  `637` debt signatures, `20` allowlist entries, and `39` migrated scopes;
- the manifest ceiling of `637` debt signatures / `668` occurrences accounts
  exactly as `230 + 374 + 30 + 3`, while current source has `636/667` because
  one `api.ts` debt entry is stale;
- Tranche A's `230` signatures occupy `242` occurrences (`204/216` Research
  plus `26/26` shared model copy); signatures, not occurrences, are the tranche
  accounting authority;
- `AppRecordsSection.tsx` has no production consumer;
- its five frontend API client exports are consumed only by that unreachable
  component, while backend/offline App Records machinery remains separate;
- `modelRoutingUx.ts` and `modelPicker.ts` are shared by Settings and Research;
- `App.tsx` still stores a raw sidecar `Error.message` for System diagnostics;
  and
- the public locale selector remains absent and production `ui_locale` remains
  unset unless a reviewed development/test path writes it.

The current scanner is a debt ratchet, not yet a complete copy inventory. Its
visible-property set omits table `header`, and it does not trace tuple-backed
column specifications. A direct audit found 22 additional English table labels
in Holdings, Capture, and Account Overview. They are owned Tranche B copy even
though they are absent from the `637` manifest ceiling. Scanner hardening and
their migration are part of this unit; final coverage may not be inferred from
the old manifest alone.

Exact manifest-debt ownership:

| Tranche | File or family | Signatures |
| --- | --- | ---: |
| A | `Research.tsx` | 68 |
| A | `ResearchHistoryDrawer.tsx` | 57 |
| A | `ResearchEvidenceDrawer.tsx` | 42 |
| A | `researchErrors.ts` | 20 |
| A | `ResearchRunProgress.tsx` | 12 |
| A | `researchSelection.ts` | 4 |
| A | `researchReducer.ts` | 1 |
| A | `modelRoutingUx.ts` + `modelPicker.ts` | 26 |
|  | **Tranche A subtotal** | **230** |
| B | Portfolio Activity/Capture/Overview/Recent/Holdings | 350 |
| B | `Dashboard.tsx` | 20 |
| B | `DataTable.tsx` + `MarkdownView.tsx` + `ConfirmDialog.tsx` | 4 |
| B | owned manifest debt subtotal (translated chrome plus scanner adjudication) | 374 |
| B | retired `AppRecordsSection.tsx` | 30 |
| B | `api.ts` scanner adjudication | 3 |
|  | **Tranche B subtotal** | **407** |
|  | **Total** | **637** |

Tranche B's `374` owned manifest signatures occupy `393` occurrences before
AppRecords and `api.ts`; not every owned signature is visible copy. The scanner
also misclassifies machine-state comparison operands in Research, model
selection, and Portfolio. All such operands require narrow scanner-classification
RED fixtures, not translation or allowlist expansion. The 22 newly detected
table labels are additional product copy and receive their own exact plan
ledger rather than being retroactively hidden inside `374`.

### 1.2 Explicit supersessions

This document supersedes only the following forward-looking clauses; the
historical documents remain unchanged as provenance:

- I18N-2 and Slice 4 protection/deferral of the unreachable AppRecords panel;
- I18N-2/Slice 5 protection of shared model picker display copy, while
  preserving their Decision 37 semantic metadata contract;
- Slice 5's handoff of remaining Evidence Drawer chrome to I18N-4;
- I18N-3's handoff of Research, App sidecar raw-error cleanup, and formatter
  audit to separate I18N-4/I18N-5 units; and
- the app-wide decision's former standalone I18N-5 sequence row.

No completed behavior or historical evidence is retroactively rewritten.

## 2. Goals and Non-Goals

### 2.1 Goals

This unit must:

1. migrate all remaining reachable application-owned chrome to static typed
   `zh-Hant` and `en` resources;
2. preserve user, source, provider, broker, and generated content unchanged;
3. converge shared model-selection labels to one reactive resource owner;
4. replace locale-dependent strings stored in UI state with semantic outcomes;
5. retire unreachable AppRecords frontend code without deleting backend or
   offline archive/migration capabilities;
6. remove all scanner debt and replace the path list with a global `src/**`
   migrated-scope contract;
7. preserve current workflows, drafts, focus, scrolling, polling, and request
   behavior across locale changes; and
8. leave separately reviewable base-to-A and A-to-final evidence.

### 2.2 Non-goals

This unit does not:

- expose or placeholder-render the public locale selector;
- pass `ui_locale` into prompts or auto-translate Research output;
- alter Research, Portfolio, Holdings, or Dashboard information architecture;
- change backend routes, DTOs, prompt construction, database schema, or stored
  semantic IDs;
- change date, time, timezone, number, currency, percentage, or market-session
  formatter behavior;
- remove backend/offline App Records storage or migration tools;
- align Settings diagnostics with the stricter Explore boundary;
- repair SA extension packaging, Backfill, degraded observability, or
  `/sa/feed` empty-versus-bad semantics; or
- absorb I18N-6 selector/release work.

## 3. Mechanical Tranche Contract

### 3.1 One unit, two hard checkpoints

Implementation uses one worktree and one reviewed plan. Tranche A must reach a
fully green checkpoint before Tranche B product edits begin. The evidence
ledger records the full commit hash under the exact name `TRANCHE_A_TIP`.

At `TRANCHE_A_TIP`, evidence must include:

- full and focused frontend results;
- exact added, removed, renamed, and evolved test-node accounting;
- exact resource leaf counts per namespace and locale;
- after the reviewed table-header scanner hardening, scanner exactly `483`
  candidate occurrences, `448` current signatures, `429` manifest-debt
  signatures, `20` allowlist entries, `448` manifest-debt occurrences, and `48`
  migrated scopes (the current 39 plus the nine Tranche A owned source files);
- typecheck, production build, selector-absence, and protected-byte gates; and
- a Tranche A bilingual Research smoke.

The A checkpoint includes the known 22 newly detected B-side table labels and
still includes the one stale `api.ts` manifest row; current live debt at that
point is therefore one signature/occurrence lower than the manifest ceiling.
Any additional scanner discovery is a stop-and-amend, not a number silently
absorbed into final accounting.

The ledger records the full 40-character commit hash, proves ancestry from
`93cda66831b7`, and hashes the scanner output/manifests used for the checkpoint.
All commands run from the clean committed tree. Tranche A helper code must live
in an already migrated scope such as `src/i18n/**` or an owned file; introducing
an additional production path is a stop-and-amend because it changes the exact
48-scope checkpoint.

Independent review runs two canonical comparisons:

1. `93cda66831b7 -> TRANCHE_A_TIP`; and
2. `TRANCHE_A_TIP -> final product tip`.

Tranche B may not hide a Tranche A regression in a net final count. After the
checkpoint, Tranche B must not edit Tranche A-owned product files. A discovered
cross-tranche correction requires stop-and-amend review and a replacement
`TRANCHE_A_TIP` with refreshed evidence.

### 3.2 Commit ownership

The implementation plan must preserve reviewable commits for:

1. Tranche A resources/presenters and Research wiring;
2. the completed Tranche A checkpoint;
3. Portfolio/System/common residual migration;
4. AppRecords frontend retirement as its own commit;
5. scanner debt/scope closure; and
6. final evidence/docs.

Commit boundaries do not replace RED-first testing. Every behavior change is
first demonstrated by its owning test.

## 4. Tranche A: Research Boundaries

### 4.1 Translate chrome, preserve work and evidence

Research application chrome localizes, including workspace controls, thread
history commands, run progress, selection reasons, closed status/error labels,
Evidence Drawer headings, and accessible names.

The following remain byte-for-byte source/user/generated content and never
pass through `t()` merely because locale changes:

- composer drafts and sent user prompts;
- thread titles and transcript turns;
- streamed/final model answers;
- tool names, tool inputs/outputs, result previews, evidence text, citations,
  context snapshots, and source rationale;
- Provider/model names, effort IDs, error codes, run IDs, and stable enum IDs;
  and
- exact current or per-run personalization context snapshots.

Existing zh-Hant chrome moves byte-for-byte by default. Every copy correction
must be named in the implementation plan; terminology authority wins when the
legacy literal conflicts with a canonical term.

### 4.2 Product suggestions freeze after selection

Unselected starter/suggested prompts are application-owned affordances and may
localize. Selecting one copies its current-locale text into the composer. From
that point it is draft/work content: later locale changes must not rewrite it,
replace it, or resend a translated variant.

No existing prompt, transcript, or generated answer is automatically
translated. Existing explicit user-triggered translation behavior elsewhere
is unaffected.

### 4.3 Semantic outcomes, not translated state

Reducers and async handlers store stable IDs or structured facts, not rendered
sentences. In particular, the current reducer-owned `連線中斷` state becomes a
semantic outcome rendered at read time. Late Provider/network results arriving
after a locale change render in the current locale without reissuing work.

Research error presentation evolves the existing typed `researchErrors`
owner. It must not infer semantics from English `Error.message` text. Existing
Developer Mode diagnostic and redaction behavior is preserved, not broadened
into a new general sanitizer in this unit.

`ResearchHistoryDrawer` mutation conflict/not-found handling and Research
thread-not-found handling use structured status/code fields. If a required
typed discriminator does not exist, implementation stops for a reviewed
backend amendment; it may not add an English substring heuristic.

## 5. Shared Model-Selection Copy

### 5.1 Reopened protected boundary

`modelRoutingUx.ts` and `modelPicker.ts` were protected in Slice 5. This design
explicitly reopens them because their visible labels serve both Settings and
Research. All other Slice 5 immutable boundaries remain closed.

One common resource/presenter owner supplies shared model group, auth-mode,
compatibility, effort/thinking, and selection-reason chrome. Settings and
Research consume it through reactive namespace-typed translation hooks. No
singleton-only read may leave mounted consumers stale after an in-place locale
change.

### 5.2 Decision 37 evolves, never weakens

Existing `id`, `baseLabel`, and `compatibility` metadata remain explicit and
covered. Production code must not recover semantic state by parsing a decorated
label such as `base label · compatibility note`.

- model IDs, Provider names, and provider-supplied model labels remain source
  identifiers;
- application-owned group/reason/compatibility labels localize;
- a display label is composed from stable metadata at render time; and
- existing ordering, compatibility, and no-reverse-parsing tests evolve in
  place rather than losing coverage.

This owner change applies mechanically to Settings, Research,
`ResearchEvidenceDrawer`, and existing model picker consumers. It does not
migrate unrelated Research chrome early or change model eligibility/routing.

## 6. Tranche B: Portfolio, System, and Common Residuals

### 6.1 Portfolio and Holdings

Localize page, table, filter, form, dialog, disclosure, action, loading, empty,
partial, and error chrome. Preserve unchanged:

- account, ticker, broker, order, execution, source, and stable status IDs;
- user notes, annotation text, imported content, and legitimate domain/source
  reason values (not transport or persisted exception detail);
- positions, lots, trades, prices, P&L, counts, and all numeric values; and
- existing sorting, filtering, polling, mutation, reconciliation, and
  persistence semantics.

Closed enums map through exhaustive typed presenters. Unknown stable IDs remain
distinguishable and visible according to the owning domain's existing fallback;
normal UI never exposes schema field names as labels. Portfolio activity
`change.field` values map through a closed local display owner rather than
printing API field names.

Portfolio errors store operation IDs and structured facts. Normal Mode renders
localized actionable copy. Raw `Error.message`, `error_detail`, traceback,
path, SQL, or arbitrary exception text is not rendered in normal UI.
Developer Mode may show only already-reviewed safe structured fields such as a
stable code/status/route; unproven detail is omitted rather than passed through.

### 6.2 System and `App.tsx`

`Dashboard.tsx` localizes System/Developer chrome while preserving technical
identifiers and measured values. `System / Health` remains the canonical mixed
label.

`App.tsx` is in scope only for:

1. replacing the sidecar raw `e.message` state with a structured safe outcome;
2. passing existing `developerMode` or navigation capability where a migrated
   owned surface requires it; and
3. locale-reactive wiring needed by those exact paths.

No Shell navigation, layout, provider lifecycle, startup, or sidecar polling
behavior may change. Any wider `App.tsx` diff is a stop condition.
The current Dashboard already gates the raw sidecar detail behind Developer
Mode; this work is structured-state hardening, not a claim that normal mode
currently leaks it.

### 6.3 Shared residual chrome

- `DataTable` localizes its action-column heading and row action accessible
  name while preserving cell/source values.
- `ConfirmDialog` localizes only its built-in default cancel label; caller
  supplied labels remain caller-owned.
- `MarkdownView` localizes blocked-image fallback chrome while preserving the
  Markdown source and rendered source text exactly.

These common primitives may subscribe directly to the common namespace. Their
portal, focus, keyboard, ARIA, and caller-override behavior remains unchanged.

## 7. AppRecords Frontend Retirement

`AppRecordsSection.tsx` is unreachable and has no production consumer. It must
not be translated merely to reduce a scanner count.

One dedicated implementation commit removes:

- the legacy Settings component;
- its five dead frontend `api.ts` exports: three interfaces and two
  preview/apply wrappers;
- its exact 30 debt signatures; and
- the standing "App Records panel full removal deferred" frontend backlog.

The same commit includes a zero-reference grep gate for the component and each
removed client symbol. The otherwise protected `api.ts` boundary is opened only
for those named dead symbols in that commit.

Backend routes, local stores, archive data, migration scripts, and tests for
offline/backend App Records capabilities remain byte-identical. Their later
retirement, if any, requires a separate owner and evidence.

## 8. Locale-Switch and Formatter Contracts

### 8.1 Pure display change

An in-place locale change may issue only the locale controller's reviewed
profile-settings request. It must not refetch Research, Portfolio, Holdings, or
System data and must not remount owned subtrees with a locale key.

The following survive with node identity where applicable:

- active Research thread, composer draft, streamed/completed answer, selected
  model/effort, history/evidence Drawer, scroll anchor, focus, and in-flight run;
- active Holdings view, filters, expanded rows, sync/manual-adjustment drafts,
  annotation editor, confirm dialogs, busy/poll state, scroll, and focus;
- Dashboard state and Developer Mode; and
- explicit translated/generated content already obtained by the user.

Memoized display values include locale/translator dependencies. In-flight
results store semantic outcomes and render using the locale current when they
arrive.

### 8.2 Formatter audit only

This unit creates `docs/design/I18N_FORMATTER_INVENTORY.md` listing each
remaining visible date/time/number/currency/percentage formatter, its owner,
current behavior, locale dependency, and future decision owner.

No formatter implementation, locale argument, output, rounding, timezone,
sign convention, or test expectation changes here. Any formatter behavior diff
is a stop condition and belongs to I18N-6 or a separately reviewed unit.

## 9. Resource and Scanner End State

### 9.1 Resource ownership

The implementation plan must ground exact leaf counts before edits. Expected
ownership is:

- extend `research` for Research-only chrome;
- add `portfolio` for Holdings/Portfolio chrome;
- add `system` for Dashboard/System chrome; and
- extend `common` for shared model-selection and primitive chrome.

AppRecords receives no resources. Both locales ship together. Recursive tests
require exact key-path parity, non-empty leaves, static selectors, and no raw
key rendering. Existing zh-Hant chrome moves byte-for-byte except changes named
in the plan copy ledger.

### 9.2 Debt zero and global scope

Final literal-policy output is locked to:

- `candidateCount = 36`;
- `signatureCount = 20`;
- `debtSignatureCount = 0`;
- `allowlistCount = 20`; and
- migrated scopes exactly `['src/**']`.

The 36 remaining candidate occurrences are the 20 already reviewed stable/source
allowlist signatures. No new allowlist entry is authorized by this design.

Scanner adjudication covers all known pure false-positive signatures:

- two transport arguments inside `sendCalibrationMessage`;
- four model-picker reason-code comparison/return operands;
- two `broker_day_gap` comparison operands; and
- `running` / `complete` completion-state comparison operands.

The mixed `未知` expression in Portfolio remains owned by its genuinely visible
rendered sentence; the machine-state sentinel itself is not translated.

Scanner coverage also gains RED fixtures and implementation for direct
`header` properties and tuple-backed static column labels. The known result is
22 additional Tranche B English labels. The implementation plan must list each
before/after value and prove the scanner sees them before migration.

`NVDA` in the manual-Holdings ticker placeholder is an example hint, not stored
source data. It must not be translated or added to the allowlist. Replace the
example with a common-resource-owned generic `Ticker` placeholder in both
locales; real ticker values remain untouched.

The three `api.ts` debt records close as follows:

1. remove the stale manifest record;
2. add RED scanner fixtures for transport arguments and the machine-state
   operands above; and
3. narrow classification without weakening detection of actual visible
   presenter returns.

Production `api.ts` transport literals remain unchanged except the separately
named dead AppRecords wrappers in §7. The existing foundation-boundary test
evolves in place from the 39-path list to the global `src/**` contract.
The durable final assertion also requires an empty debt manifest, exact current
allowlist contents/counts, `candidateCount` equal to summed allowlist
occurrences, and `signatureCount` equal to allowlist entries; scanner CLI
success alone is insufficient because stale debt is otherwise tolerated.

## 10. File and Ownership Boundary

### 10.1 Tranche A owned product paths

- `apps/arkscope-web/src/Research.tsx`
- `apps/arkscope-web/src/ResearchHistoryDrawer.tsx`
- `apps/arkscope-web/src/ResearchEvidenceDrawer.tsx`
- `apps/arkscope-web/src/ResearchRunProgress.tsx`
- `apps/arkscope-web/src/researchErrors.ts`
- `apps/arkscope-web/src/researchSelection.ts`
- `apps/arkscope-web/src/researchReducer.ts`
- `apps/arkscope-web/src/modelRoutingUx.ts`
- `apps/arkscope-web/src/modelPicker.ts`
- directly corresponding resource, presenter, and test files

Settings files may receive only the mechanical shared-owner import/wiring and
tests required by §5. They do not reopen Settings IA or copy migration.

Before the A checkpoint, scanner tooling/tests may change only for the reviewed
`header`/tuple coverage and A-owned machine-state false positives. Tranche B may
later evolve the same tooling for B false positives, stale debt removal, and
global-scope closure; both versions are hashed in their respective evidence, so
the final scanner cannot erase what the A checkpoint observed.

### 10.2 Tranche B owned product paths

- `PortfolioActivity.tsx`
- `PortfolioCapturePanel.tsx`
- `PortfolioAccountOverview.tsx`
- `PortfolioRecentActivity.tsx`
- `Holdings.tsx`
- `Dashboard.tsx`
- `App.tsx`, limited by §6.2
- `ui/DataTable.tsx`, `ui/ConfirmDialog.tsx`, and `MarkdownView.tsx`
- `settings/legacy/AppRecordsSection.tsx`, deletion only
- `api.ts`, limited to §7 dead symbols and §9 scanner evidence
- directly corresponding resource, presenter, and test files
- scanner manifests/tool/tests and the formatter inventory

Backend `src/`, `data_sources/`, backend tests, desktop/native code,
extensions, package manifests/lockfiles, and CSS are byte-identical by default.
A measured bilingual overflow may reopen one existing CSS owner only after a
reviewed stop-and-amend with RED geometry evidence and exact diff accounting.

## 11. Verification and Accounting

### 11.1 Plan ledger

The implementation plan must declare before product edits:

- exact baseline and target test files/nodes;
- explicit additions, removals, renames, and in-place evolutions;
- every zh-Hant wording change as exact before/after copy;
- a direct AST/source inventory of English fallback/static-column copy in every
  owned file, including values the pre-unit scanner did not classify;
- per-namespace resource leaves for both locales;
- scanner four-count and scope checkpoints at base, A, and final;
- protected-byte commands and exact exceptions; and
- exact focused suites for each tranche.

Net counts may not conceal removed assertions. The AppRecords removals, scanner
fixture additions, shared model-owner evolution, and any stale test-name
correction are named individually.

### 11.2 Required static and unit evidence

At minimum:

- resource key parity/non-empty/static-key gates;
- scanner hostile fixtures and exact terminal counts;
- selector absence;
- Research source/draft/output byte preservation;
- suggested-prompt localization followed by draft freeze;
- structured Research 404/409/disconnect handling without message parsing;
- Decision 37 metadata/no-reverse-parsing coverage;
- Portfolio operation-specific semantic outcome rendering;
- normal-mode raw-detail absence and bounded Developer diagnostics;
- in-place evolution of existing Capture tests that currently expect raw
  `error_detail`, so the old leak assertion is removed rather than masked;
- App sidecar state contains no raw `Error.message`;
- AppRecords zero-reference and backend-byte gates;
- common primitive focus/ARIA/caller override preservation; and
- locale-switch node identity, draft, focus, scroll, in-flight, and zero-data-
  request contracts.

### 11.3 Runtime matrix

Tranche A runs Research in both locales at `1440`, `960`, and `390` widths,
covering workspace, history, evidence, active progress, error, long source
content, and locale switching during draft/in-flight work.

Tranche B runs Holdings/Portfolio and Dashboard/System in both locales at all
six canonical widths: `1440`, `1024`, `961`, `960`, `959`, and `390`. It covers
worst-case table/filter density, long errors, open menus/dialogs, Developer Mode
on/off, dirty Capture state, active polling, expanded Activity rows, and locale
switching. Common primitives receive focused runtime coverage through their
owning surfaces.

Every width must have zero horizontal document overflow and complete labels.
Research uses its three owned widths; Portfolio/System always includes the
`959/960/961` triple because Holdings behavior changes at that boundary. Full
text may wrap; it may not be truncated or shrunk to hide overflow.

## 12. Sequence, Stops, and Acceptance

### 12.1 Sequence

1. Independent written review approves this design.
2. One RED-first implementation plan grounds exact tests/resources/copy and is
   independently reviewed.
3. Task 0 records the plan-clearance commit and creates the isolated worktree.
4. Scanner header/tuple RED coverage lands, Tranche A implements, and the exact
   `TRANCHE_A_TIP` evidence is recorded.
5. Tranche B implements, including the dedicated AppRecords retirement commit.
6. Independent review runs base-to-A and A-to-final evidence separately.
7. Explicit user approval precedes fast-forward merge.
8. Merged-tree verification and normal zh-Hant smoke precede LIVE closeout.
9. I18N-6 remains the sole next i18n unit after closeout.

### 12.2 Stop conditions

Stop and amend before proceeding if:

- the pre-hardening manifest ceiling does not reconcile exactly to
  `230 + 374 + 30 + 3`, or the header/tuple hardening does not expose the 22
  separately inventoried labels;
- a reachable remaining surface lies outside the file/accounting map;
- Tranche B needs to edit a frozen Tranche A product owner;
- a backend discriminator is required to avoid English-message parsing;
- AppRecords frontend symbols have a real production consumer;
- any backend/offline App Records capability would be removed;
- model routing/eligibility behavior changes;
- a formatter output or formatter test expectation changes;
- a new allowlist entry appears necessary;
- final scanner output differs from `36/20/0/20` or scope is not `src/**`;
- product code needs a new dependency, backend/schema change, or public selector;
- unreviewed CSS is needed; or
- canonical test accounting drifts from the reviewed plan.

### 12.3 Deferred work

This design leaves visible and separate:

- I18N-6 selector, release audit, full bilingual visual matrix, and Design Kit
  locale sync;
- formatter policy/implementation after this unit's inventory;
- Settings diagnostic sanitizer alignment;
- SA extension packaging integrity, News Backfill, degraded observability, and
  old-article policy;
- `/sa/feed` empty-versus-bad semantics;
- calibration Anthropic refusal normalization and fallback-model review; and
- any backend/offline App Records retirement.

### 12.4 Acceptance

The unit is complete only when:

- both tranche comparisons independently pass;
- all reachable Research, Portfolio, Holdings, Dashboard/System, and common
  residual chrome renders correctly in both locales;
- user/source/generated content and stable identifiers remain unchanged;
- locale switching is display-only and preserves work/state/focus;
- AppRecords frontend code is gone while backend/offline capability is intact;
- formatter behavior is unchanged and its inventory exists;
- scanner output is exactly `36/20/0/20`, debt is zero, and scope is `src/**`;
- the selector remains absent; and
- merged-tree gates and normal zh-Hant desktop smoke are green.
