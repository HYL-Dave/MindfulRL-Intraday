# ArkScope I18N-6 Release Design

> **Status: INDEPENDENT PLAN REVIEW GREEN — IMPLEMENTATION CLEARED**
>
> Written against merged `master` at `cf7050ed` on 2026-07-25. This document
> is the product and verification authority for the final app-wide i18n
> release unit. Independent written review returned GREEN. It does not
> authorize implementation until a separately reviewed RED-first implementation
> plan is cleared. The public locale selector remains absent until that release
> gate passes.

## 1. Purpose and Authority

I18N-0 established the profile-backed locale authority, synchronous cached
bootstrap, typed static resources, and the visible-literal scanner. I18N-1
through I18N-3 migrated Shell, Settings, and Explore. P2.8 Slice 5 shipped the
redesigned Investor Profile bilingual from birth. The combined I18N-4/5 unit
then migrated Research, Portfolio, System, and the remaining shared chrome,
retired the unreachable AppRecords frontend, and closed the scanner debt at
zero.

I18N-6 is deliberately smaller. It releases the already-complete bilingual
application by:

1. adding exactly one public locale selector in the Settings PageHeader;
2. proving the existing locale controller from first paint through durable
   write, rollback, reload, and in-place switching;
3. running the final all-surface bilingual and responsive matrix;
4. converting the formatter inventory from an open decision into a frozen V1
   boundary; and
5. synchronizing release documentation and, after product release, the
   external Design Kit locale state.

Authority order:

1. [`ARKSCOPE_TERMINOLOGY.md`](../../design/ARKSCOPE_TERMINOLOGY.md) owns
   canonical terminology and the recorded locale-autonym exception.
2. [`2026-07-20-app-wide-i18n-decision.md`](2026-07-20-app-wide-i18n-decision.md)
   owns locale authority, cache/write semantics, source-content boundaries,
   and selector-last sequencing.
3. [`I18N_FORMATTER_INVENTORY.md`](../../design/I18N_FORMATTER_INVENTORY.md)
   owns the observed formatter list and, after this decision, its frozen V1
   interpretation.
4. Existing Shell, Settings, Research, Explore, Portfolio, and System designs
   continue to own their information architecture and domain behavior.
5. This document owns the selector implementation boundary, production
   first-write discipline, release matrix, and I18N-6 acceptance evidence.

### 1.1 Grounded baseline

At `cf7050ed`:

- the frontend is `94 files / 1048 tests` and is green;
- the literal scanner reports exactly `36` candidate occurrences, `20`
  current signatures, `0` debt signatures, and `20` allowlist entries, with
  global scope `src/**`;
- each locale has `1779` non-empty resource leaves: Common `61`, Shell `37`,
  Settings `679`, Research `207`, Explore `401`, Portfolio `374`, and System
  `20`;
- `settings.locale` contains only the already-shipped `writeFailed` leaf;
- `SUPPORTED_UI_LOCALES` is exactly `['zh-Hant', 'en']`, with `zh-Hant` as the
  default and fallback;
- `LocaleProvider` exposes `locale`, `busy`, `errorCode`, and `setLocale`;
- `createUiLocaleController` already implements optimistic application,
  overlap prevention, sequence ordering, authority validation, rollback, and
  write-through cache updates only after a successful PUT;
- `main.tsx` already makes controller-driven locale application update both
  i18next and `<html lang>`;
- `PageHeader` already provides an `actions` slot whose existing primitive CSS
  wraps without a new selector-specific layout primitive;
- Settings currently renders no PageHeader action and no locale selector; and
- the latest production read-only observation finds no
  `profile_settings.ui_locale` row. That is an observation before release, not
  a permanent acceptance constant after the user intentionally uses the new
  selector.

### 1.2 Why the selector remains a bounded change

The durable DB setting, GET/PUT API, bootstrap cache, controller, React
provider, resources, and global locale subscriptions already exist. I18N-6
does not introduce another locale authority or another navigation model. Its
only production UI addition is a Settings-local control connected to the
existing controller.

The low product risk does not justify weak verification. The selector is the
first public writer of a preference that earlier i18n units deliberately kept
absent, and it is the release point at which every reachable surface becomes
user-switchable. The final gates therefore concentrate on authority purity,
state preservation, and layout rather than re-testing every domain rule in
two languages.

## 2. Scope

### 2.1 In scope

- a Settings-local `LocaleSelector` component;
- exactly one mount in the Settings `PageHeader.actions` slot;
- two new Settings resource leaves per locale: selector label and self-name;
- localized write-failure rendering using the existing semantic error code;
- exact selector presence/absence and registry-boundary tests;
- test-harness wiring needed because direct `SettingsView` mounts must now
  provide the real locale context;
- isolated end-to-end locale writes, reloads, rollback, and retry evidence;
- all-surface bilingual responsive and state-preservation evidence;
- formatter inventory closure and formatter byte gates;
- merged-tree and production read-only closeout; and
- post-release Design Kit locale-control and bilingual-state synchronization.

### 2.2 Out of scope

- backend, schema, route, DTO, or API-client changes;
- locale-controller, provider, bootstrap, cache, or supported-locale changes;
- OS, browser, Electron, timezone, or geolocation locale detection;
- Simplified Chinese or any third locale;
- live cross-window locale push, polling, SSE, or broadcast synchronization;
- a fourth Settings tab, registry section, topbar selector, or new shared UI
  primitive;
- dynamic resource loading, Suspense, ICU, or new dependencies;
- translation or rewriting of source, user, or generated content;
- automatic card translation caused by `ui_locale`;
- formatter output, precision, grouping, sign, currency, timezone, or date
  changes;
- SA extension packaging/backfill/observability work, `/sa/feed` empty-versus-
  bad semantics, Settings diagnostic-sanitizer alignment, or calibration
  refusal normalization; and
- any unrelated visual or behavioral cleanup discovered during the matrix.

## 3. Locked Decisions

1. V1 supports exactly `zh-Hant` and `en`; `zh-Hant` remains default and
   fallback. `Simplified Chinese` appears only in a negative absence guard and
   is not a supported option.
2. The selector is a Settings-local component, not a shared primitive.
3. It mounts exactly once in the Settings PageHeader action slot and nowhere
   else.
4. It is not a Settings registry entry, workflow group, fourth tab, or Shell
   topbar control.
5. The visible control is a compact labeled native `<select>`.
6. The label follows the active locale. The options are autonyms and remain
   `繁體中文` and `English` regardless of the active locale.
7. Autonyms live in locale resources under the same semantic key and are read
   through fixed-locale translators. Product TypeScript/TSX contains no
   hard-coded autonym literal.
8. The selector calls only `useUiLocale().setLocale()`. Direct
   `i18next.changeLanguage()` is forbidden in the selector owner.
9. The existing locale controller remains the single writer and owns
   optimistic application, `<html lang>`, durable PUT, cache update, rollback,
   retry, and overlap prevention.
10. A write failure is rendered after rollback, so its localized error copy
    uses the restored locale. Raw authority or exception detail never renders.
11. Locale switching is display-only. It does not navigate, invoke a Settings
    workflow guard, refetch page data, remount a surface, or mutate source and
    generated content.
12. Production closeout does not create the first `ui_locale` value. The full
    writer chain is proved against isolated state; the user's first production
    selection remains intentional.
13. `ui_locale` controls interface copy only. Existing formatter behavior is
    frozen per owner and does not become locale-aware in I18N-6.
14. No CSS change is planned. A measured bilingual overflow requires an
    explicit reviewed deviation with RED evidence and an owner-specific
    responsive gate.
15. Design Kit synchronization follows product release and records both
    autonym selector states and bilingual Settings states. It does not
    pre-authorize product code.

## 4. Selector Contract

### 4.1 Component owner

The product owner is:

```text
apps/arkscope-web/src/settings/LocaleSelector.tsx
```

It uses:

- `useTranslation('settings')` for active-locale copy and access to the
  initialized i18n instance;
- `useUiLocale()` for `locale`, `busy`, `errorCode`, and `setLocale`; and
- `isUiLocale()` to reject any unexpected DOM value before calling the
  controller.

It must not import the API authority, cache adapter, DB concepts, or a global
controller. It must not provide a context-free fallback. Rendering outside a
`LocaleProvider` remains a programming error, as it is today for every
`useUiLocale` consumer.

### 4.2 Resource shape

The Settings locale subtree becomes:

```text
settings.locale.label
settings.locale.selfName
settings.locale.writeFailed
```

Values are:

| Key | `zh-Hant` | `en` |
| --- | --- | --- |
| `label` | `介面語言` | `Interface language` |
| `selfName` | `繁體中文` | `English` |
| `writeFailed` | existing reviewed value | existing reviewed value |

The Chinese option is rendered with
`i18n.getFixedT('zh-Hant', 'settings')`; the English option uses
`i18n.getFixedT('en', 'settings')`. Both read the same `locale.selfName` key.
Changing active locale therefore changes the field label and error copy but
never changes either option's self-name.

The resource inventory changes only as follows:

| Namespace | Base | Delta | Target |
| --- | ---: | ---: | ---: |
| Settings | 679 | +2 | 681 |
| All namespaces per locale | 1779 | +2 | 1781 |

All other namespace counts stay fixed.

### 4.3 DOM and interaction

The control has one programmatically associated localized label and one
controlled `<select value={locale}>` containing exactly two options in the
stable supported-locale order: `zh-Hant`, then `en`.

The interaction sequence is the existing controller sequence:

1. an already-busy controller rejects re-entry before applying anything;
2. a valid new selection is applied optimistically to i18next and
   `<html lang>`;
3. controller state becomes `busy: true`, disabling the select;
4. the validated PUT is sent;
5. success commits the locale, clears any prior error, and then updates the
   cache; or
6. failure restores the committed locale, leaves cache unchanged, and exposes
   only `errorCode: 'write_failed'`.

A duplicate event while busy cannot issue a second PUT. An invalid value
cannot reach `setLocale`.

### 4.4 Error presentation

`write_failed` renders the existing localized `settings.locale.writeFailed`
copy adjacent to the selector. It contains no raw exception, status body,
path, or backend detail. Because controller rollback precedes the final error
state, a failed `zh-Hant -> en` write renders the Traditional Chinese error;
the reverse path renders the English error.

A later successful selection clears the prior error. Error state is not
persistent and does not create another preference key.

### 4.5 Placement

`Settings.tsx` changes only its PageHeader composition and import surface:

```tsx
<PageHeader
  title={...}
  actions={<LocaleSelector />}
/>
```

The action is above the existing three workflow tabs. It does not move route
save/import/export controls, create a global action band, or alter Settings
anchors and search.

## 5. Locale Authority and Switching Purity

### 5.1 Existing authority remains unchanged

The following layers keep their current roles:

| Layer | Role |
| --- | --- |
| `profile_settings.ui_locale` | durable profile authority |
| `arkscope.ui.locale.v1` | best-effort first-paint cache only |
| `bootstrapUiLocale` | synchronous cache/default first paint |
| `createUiLocaleController` | reconcile and write sequencing |
| `LocaleProvider` | reactive React subscription |
| Settings selector | user command surface only |

The cache is write-through, never an independent authority. It is updated only
after a valid stored DB response. No selector path writes cache first and
repairs the DB later.

### 5.2 Cold start and reconciliation

- Valid cached `zh-Hant` or `en` applies synchronously before React render.
- Missing, corrupt, unsupported, or inaccessible cache falls back to
  `zh-Hant`.
- Authority GET may correct the bootstrap locale once and then updates cache.
- GET failure is display-level fail-open: the bootstrap locale remains usable.
- Reconciliation remains coalesced under StrictMode.
- If a user write operation has already started, reconciliation yields to that
  operation rather than overwriting it.
- V1 convergence occurs at startup and on Settings changes only. There is no
  live multi-window push.

### 5.3 Request purity

Changing locale permits exactly one network request: the locale authority PUT.
It must not cause any surface GET, polling restart, search request, model
catalog request, feed reload, profile reload, or navigation request.

The gate counts requests and requires:

```text
count == 1
method == PUT
pathname == /profile/settings/ui-locale
```

Query strings, unrelated background requests, or a second locale write fail
the isolated selector gate.

### 5.4 State preservation

The locale change rerenders copy in place. It must preserve:

- active top-level view and Settings workflow tab;
- Settings directory query, exact target, and Drawer state;
- focused control and a stable node-identity marker;
- dirty controlled and uncontrolled drafts;
- disclosure, menu, modal, and Evidence/History Drawer state;
- scroll reading position without requiring pixel-perfect equality;
- in-flight operations and late structured outcomes;
- generated AI card text and manually requested translation output;
- source/user content, ticker/model/provider IDs, and persisted semantic IDs.

No product subtree may use `key={locale}` or an equivalent locale-derived
remount key. Memoized visible copy includes `t`/locale in its dependency path.

### 5.5 Production first-write discipline

Before merge and after merged desktop restart, production checks are read-only:

1. query whether `profile_settings.ui_locale` exists;
2. navigate normally to Settings;
3. locate the selector by semantic role and localized label;
4. verify its visible value and two options; and
5. do not change it.

The full `absent -> en -> zh-Hant` writer lifecycle runs only against an
isolated profile DB and isolated browser storage. Production mutation controls
must not be exercised through coordinate-based smoke. This explicitly carries
forward the run-307 closeout lesson.

After release, a real user selection is expected to create the durable value.
The pre-release absent state must not be preserved by adding a hidden second
gate that rejects `en`.

## 6. Formatter Boundary

### 6.1 V1 decision

`ui_locale` controls localized interface messages only. Number, date, time,
currency, percentage, precision, sign, grouping, timezone, source slicing,
and fallback behavior remain exactly as recorded in
`I18N_FORMATTER_INVENTORY.md`.

This explicitly accepts the current heterogeneous behavior:

- some owners use browser locale;
- some use fixed `en-US`;
- some use locale-neutral digits and punctuation;
- some preserve source substrings; and
- localized unit/counter copy may surround an unchanged raw value.

A `zh-Hant` interface running under an English browser locale may therefore
retain English-style date/number formatting. That is accepted V1 behavior,
not a release finding.

### 6.2 Copy/value boundary

i18next owns message grammar, plural pairs, counters, and unit suffix copy. It
does not take over the raw formatting function. For example, a Research
elapsed value produced by `toFixed(1)` remains byte-identical while its seconds
suffix follows active interface copy.

### 6.3 Future changes

Binding any formatter to `ui_locale` requires a separate owner-specific unit.
That unit must name before/after output, precision/grouping/timezone effects,
tests, and visual consequences for each owner. I18N-6 does not pre-approve such
a migration.

The inventory header and all 20 owner rows replace their open I18N-6 pointer
with this frozen V1 ruling. Product formatter functions and their behavior
tests remain byte-identical.

## 7. Mechanical Contracts

### 7.1 Resource and scanner closure

I18N-6 must retain:

- exact `zh-Hant == en` resource-key parity;
- all resource leaves as non-empty strings;
- no dynamic translation keys;
- no raw key rendered in either locale;
- scanner output exactly `36/20/0/20` with scope `src/**`;
- zero debt-manifest entries; and
- the exact 20-entry reviewed stable-identifier/source-content allowlist.

Autonyms in resource modules do not enter scanner source. No autonym or
selector-label allowlist entry is permitted.

### 7.2 Existing selector guards

Four existing nodes must evolve explicitly because the release intentionally
reverses their old absence expectation:

1. `foundationBoundaries.test.ts` —
   `keeps the public locale selector absent after Settings migration`;
2. `SettingsWorkspace.test.tsx` —
   `renders no locale selector or raw planted diagnostic in Settings PageHeader`;
3. `SettingsWorkspace.test.tsx` —
   `omits_legacy_model_header_runtime_band_and_global_route_actions`; and
4. `SettingsModelRouting.test.ts` —
   `owns_save_in_models_and_import_export_in_a_closed_advanced_disclosure`.

The implementation plan must name each in-place evolution or rename and
account any test-ID `+N/-N` explicitly. Net-positive selector tests may not
hide them.

The first node becomes an owner whitelist:

- selector identifiers may appear only in `LocaleSelector.tsx` and the
  `Settings.tsx` mount;
- App, Shell, topbar, `main.tsx`, `LocaleProvider.tsx`, and every other surface
  remain selector-free;
- `LocaleSelector.tsx` is included in the direct-`changeLanguage()` ban; and
- the three Settings groups remain unchanged.

The existing node
`renders no language selector autonym or planned locale affordance` remains
unchanged. Its `productionFiles('src/settings')` glob automatically includes
the new owner and continues to prove product source contains no autonym
literal. Existing registry nodes that prove three non-empty groups, complete
anchor assignment, and AppRecords exclusion also remain unchanged and provide
the no-fake-locale-section guard.

A fifth existing node is a guaranteed in-place resource-accounting evolution:
`resources.test.ts` —
`contains the reviewed remaining-surface namespace inventory in both locales`.
It keeps its test ID and changes only the reviewed Settings and total counts
from `679/1779` to `681/1781` per locale. This evolution has zero test-node
`+N/-N` impact and must appear explicitly beside the four selector/action
guard evolutions in the implementation-plan ledger.

### 7.3 Test context is mandatory

Six existing suites directly mount `SettingsView` without `LocaleProvider`:

- `SettingsModelRouting.test.ts`;
- `SettingsProviderConfig.test.ts`;
- `SettingsPostPgExitStorage.test.ts`;
- `SettingsInvestorProfileIntegration.test.tsx`;
- `SettingsNewsStorage.test.ts`; and
- `SettingsWorkspace.test.tsx`.

Their harnesses must render through a deterministic test locale provider and
real controller contract. A shared test-only wrapper is allowed and preferred
to six hand-written context mocks. Existing domain test IDs and assertions are
otherwise unchanged except for the four selector-related nodes listed above.

Production code must not add an optional locale context, default no-op writer,
or environment-specific selector fallback merely to keep old test harnesses
green.

### 7.4 Immutable boundaries

The implementation plan defines constructive byte gates for:

- backend source and tests;
- DB/schema/migration files;
- frontend `api.ts`;
- `main.tsx`;
- `i18n/localeController.ts`, `LocaleProvider.tsx`, `bootstrap.ts`, and
  `locale.ts`;
- `settingsRegistry.ts` and Settings preference/guard owners;
- Shell, navigation, desktop, and extensions;
- package manifests and lockfiles;
- every CSS file; and
- every formatter product and behavior-test owner named by the inventory.

Resource modules, `Settings.tsx`, the new selector, focused test infrastructure,
the named tests, and release documentation are the only expected change
families.

### 7.5 Backend-origin and source-content closure

The final release reruns the durable backend-copy gates. No reachable normal-
mode path may render raw `detail`, `warning`, `last_error`, exception text, or
assembled generic API error text without its reviewed classification.

Locale switching must not alter prompts, provider payloads, generated card
content, manually translated content, article text, Research transcript text,
proposal rationale, tickers, model/provider names, or stored semantic IDs.

## 8. Required Verification

### 8.1 Unit and mounted selector evidence

The RED-first plan must include named nodes proving at least:

1. active-locale label copy and fixed-locale option autonyms in both locales;
2. exactly two allowlisted option values in stable order;
3. a valid selection delegates to `setLocale` and an invalid value does not;
4. busy state disables the select and duplicate interaction emits one write;
5. failure rolls back option and copy, shows the restored-locale error, and
   exposes no raw detail;
6. a later successful retry clears the error;
7. Settings PageHeader has exactly one selector while registry/topbar/other
   surfaces have none;
8. `LocaleSelector.tsx` contains no direct `changeLanguage()`;
9. existing Settings tests remain behaviorally green through the real locale
   context; and
10. resource counts are exactly Settings `681` and total `1781` per locale.

The implementation plan, not this design, locks the final exact node ledger
after collecting the RED tests. Baseline nodes may not be silently removed or
renamed.

### 8.2 Isolated authority lifecycle

With an isolated profile DB and browser storage:

1. start with no DB value and no cache; first paint is `zh-Hant`;
2. select `English`;
3. observe optimistic English copy, `lang='en'`, disabled select, and exactly
   one locale PUT;
4. resolve success and verify DB/cache both store `en`;
5. reload and prove the synchronous first paint is already English and
   authority reconciliation requires no visible correction;
6. select `繁體中文`, persist `zh-Hant`, reload, and prove the inverse chain;
7. repeat with a failed PUT and prove rollback, unchanged cache, restored-
   locale error copy, and no raw detail; then
8. restore the API, retry successfully, and prove error state clears.

StrictMode instrumentation proves one startup authority GET. A user write that
starts before reconciliation completes wins and is not overwritten by the
stale GET.

### 8.3 Canonical all-surface matrix

Both locales run at all six canonical viewports:

- `1440x900`;
- `1024x768`;
- `961x768`;
- `960x768`;
- `959x768`; and
- `390x844`.

The matrix covers:

- Home load/error/empty and dense market state;
- Watchlist filters, archived state, and mutation feedback;
- Universe filters, long alerts, and recovery navigation;
- News market and Seeking Alpha modes;
- Ticker Detail and AI Card, including generated and manually translated
  content retention;
- Research workspace, History, Evidence, active progress, and error state;
- Holdings, Capture, Account Overview, and Recent Activity;
- System / Health status and sidecar state; and
- all three Settings tabs, directory/Drawer, dirty guard, Investor Profile,
  long provider/model controls, and the PageHeader selector.

Each surface mounts at least one worst credible composition with long English
chrome, relevant source content, an error/partial banner where supported, and
dense controls or facets. Gates check:

- no clipping, overlap, or unintended document-level horizontal overflow;
- complete labels without truncation or font shrinking;
- valid focus names and keyboard behavior;
- no raw resource key or unclassified raw diagnostic;
- exactly one selector when Settings is mounted; and
- a usable 390px PageHeader action arrangement in both locales.

### 8.4 Targeted `760px` boundary trio

The canonical six widths own the Shell `960px` breakpoint. Existing
`styles.css` also has `max-width: 760px` behavior. Rather than multiplying
every surface into a nine-width matrix, I18N-6 runs `759/760/761` against the
actual selector families owned by that media boundary:

- Watchlist, News, and Universe for `.surface-head` wrapping;
- Settings Model Routing for `.settings-grid`, with a mounted credential
  summary and its nested `.model-credential-summary .btn-ghost` action;
- Settings Data Sources for `.provider-config-field`; and
- the longest mounted English `.ui-inline-alert` composition, exercising the
  generic `.main`/alert stacking rule.

The plan records the exact fixture chosen for the longest alert. The source
census also records `.page-head` and `.page-head-actions` as dead CSS with no
reachable production consumer; they therefore receive no runtime fixture. If
source inspection finds another reachable owner of a behavior-changing 760px
rule, or either dead selector gains a reachable consumer, the owner is added
before the plan is cleared. Static `max-width: 760px` values that are not
media-query behavior do not create a breakpoint gate.

### 8.5 Locale-switch purity matrix

Each surface family receives at least one in-place switch in a deep state.
The gate records:

- same route and active surface;
- stable node marker and preserved focused control;
- draft/query/menu/Drawer/disclosure state retained;
- scroll not reset to zero and the anchored item still visible;
- source/generated/manually translated text byte-preserved;
- late in-flight outcome rendered with current-locale chrome; and
- request counter showing only the locale PUT and zero data requests.

### 8.6 Formatter runtime gate

A deterministic timestamp and numeric fixture is rendered before and after an
in-place locale switch. The raw formatted value must be byte-identical. A
representative message with a localized suffix/counter must change only its
copy portion. Tests set browser/timezone conditions explicitly; they do not
depend on the reviewer's host locale.

### 8.7 Canonical commands

The plan must provide exact commands for:

- focused selector/foundation/Settings/formatter tests;
- full frontend tests;
- TypeScript checking and production build;
- deterministic scanner runs and hash comparison;
- resource parity/count audit;
- backend/no-PG equivalence;
- protected-byte and CSS gates;
- both-locale browser matrix and request counters; and
- `git diff --check` plus worktree hygiene.

## 9. Documentation and Design Kit

### 9.1 Documentation

The release updates:

- this design and its later RED-first implementation plan;
- the app-wide i18n decision status and fixed-sequence table;
- `I18N_FORMATTER_INVENTORY.md` header and all future-owner cells;
- `PROJECT_PRIORITY_MAP.md`; and
- closeout evidence with merge/product hashes and production read-only result.

`ARKSCOPE_TERMINOLOGY.md` is already
`ADOPTED AND LIVE, 2026-07-24`; I18N-6 does not churn that truthful header.

### 9.2 External Design Kit

After the product release gate passes, the Design Kit receives:

- the compact PageHeader locale selector specimen;
- both active-locale labels with fixed autonym options;
- Settings PageHeader at desktop and 390px;
- representative bilingual Settings states; and
- the note that formatter output is independent of `ui_locale` in V1.

If the external design scope is unavailable, use `/design-login`, not the
general `/login`. Authentication failure does not invalidate reviewed product
code, but closeout must say the external sync is pending rather than claiming
I18N-6 fully synchronized.

## 10. File Boundary

### 10.1 Expected product changes

| File | Change |
| --- | --- |
| `apps/arkscope-web/src/settings/LocaleSelector.tsx` | add the Settings-local selector |
| `apps/arkscope-web/src/Settings.tsx` | mount the selector in `PageHeader.actions` |
| `apps/arkscope-web/src/i18n/resources/zh-Hant/settings.ts` | add `label` and `selfName` |
| `apps/arkscope-web/src/i18n/resources/en/settings.ts` | add `label` and `selfName` |

No other product file is expected.

### 10.2 Expected test changes

- new focused `LocaleSelector` test file;
- one shared test-only locale wrapper if the plan chooses that shape;
- the six direct-Settings mounted suites listed in §7.3;
- `foundationBoundaries.test.ts`;
- `resources.test.ts`; and
- any existing formatter boundary test strengthened without changing product
  formatter expectations.

The plan must reconcile this map against actual collection before Task 1.

### 10.3 Expected documentation changes

- this spec;
- app-wide i18n decision;
- formatter inventory;
- priority map; and
- the later implementation plan/evidence ledger.

## 11. Sequence and Stop Conditions

### 11.1 Sequence

1. Independent written review approves this design.
2. One RED-first implementation plan locks exact node/resource/file accounting
   and is independently reviewed.
3. Task 0 records the plan-clearance commit and creates an isolated worktree.
4. Selector/resource RED tests and mandatory test-context wiring land before
   production mounting.
5. Product implementation remains bounded to the four product files in §10.1.
6. Static, unit, isolated-authority, formatter, responsive, and full-surface
   gates complete before independent implementation review.
7. Explicit user approval precedes fast-forward merge.
8. Merged-tree verification completes before the normal desktop restart.
9. Production smoke is semantic-locator and read-only; it does not write the
   first locale value.
10. Product status turns LIVE, then the external Design Kit companion sync is
    performed and recorded.

### 11.2 Stop conditions

Stop and amend before continuing if:

- a third locale or browser/OS detector becomes necessary;
- the selector needs a backend, schema, API-client, controller, provider,
  bootstrap, cache, or `main.tsx` change;
- a production fallback is proposed for missing `LocaleProvider` context;
- selector code needs direct `changeLanguage()` or a cache write;
- more than one public selector or a registry/topbar owner appears;
- Settings resources do not close at `681` or total resources at `1781` per
  locale;
- scanner output differs from `36/20/0/20`, debt is nonzero, or the allowlist
  changes;
- any source/generated content, prompt, or manual translation changes;
- any formatter output or formatter behavior test changes;
- a CSS change is needed without measured evidence and reviewed deviation;
- the 760px owner census differs from §8.4;
- locale switching triggers a data request, navigation, remount, or state loss;
- isolated reload does not converge cache and DB authority;
- a production smoke would write `ui_locale` or invoke another mutation; or
- exact test/file accounting drifts from the reviewed plan.

## 12. Deferred Work

I18N-6 leaves visible and separate:

- any future owner-specific formatter localization;
- live cross-window locale synchronization;
- additional locales;
- SA extension packaging integrity, News Backfill, degraded observability, and
  old-article policy;
- `/sa/feed` empty-versus-bad semantics;
- Settings diagnostic-sanitizer alignment with the stricter Explore boundary;
- calibration Anthropic refusal normalization and fallback-model review; and
- any new app-level preference group beyond the single locale selector.

## 13. Acceptance

I18N-6 is complete only when:

1. the independently reviewed implementation plan and implementation both
   return GREEN;
2. exact test accounting names every addition, rename/evolution, and removal;
3. resources close at Settings `681`, total `1781` per locale, with parity and
   no empty leaves;
4. scanner remains exactly `36/20/0/20`, scope `src/**`, debt zero;
5. Settings exposes exactly one controller-backed autonym selector and no
   other surface exposes one;
6. both isolated write/reload cycles and failure/retry recovery pass;
7. locale switching preserves state/content and issues only the locale PUT;
8. formatter product behavior is byte-identical and its inventory is frozen;
9. canonical six-width all-surface gates and targeted 760px trios pass in both
   locales without clipping or unintended overflow;
10. backend/no-PG, typecheck, build, byte, and merged-tree gates pass;
11. production desktop smoke verifies the selector without writing the absent
    locale authority; and
12. release docs are truthful and the post-release Design Kit sync is either
    recorded complete or explicitly marked externally pending.
