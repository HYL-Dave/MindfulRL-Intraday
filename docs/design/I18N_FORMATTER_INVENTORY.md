# I18N Formatter Inventory

> Status: V1 UI-LOCALE BOUNDARY FROZEN BY REVIEWED I18N-6 DESIGN
> (2026-07-25; IMPLEMENTATION PLAN WRITTEN — INDEPENDENT PLAN REVIEW PENDING)
>
> This document records formatter ownership after the I18N-4/5 surface
> migration. I18N-6 has now ruled that `ui_locale` controls interface copy
> only. It does not authorize output, rounding, timezone, sign, locale, or
> fallback changes. Any future binding to `ui_locale` requires a separately
> reviewed owner-specific formatter unit with named before/after output.

## Interpretation

- `ui_locale` means ArkScope's profile-backed `zh-Hant` / `en` preference.
- `browser locale` means the locale selected by `Intl` or `toLocale*` when the
  code passes `undefined` or no locale. It does not necessarily change when
  `ui_locale` changes in place.
- Source values, identifiers, currencies, and timestamps remain unchanged by
  the completed copy migration. Existing fallback policy also remains intact;
  the reviewed `PortfolioActivity` exception supplies the same unknown state
  from localized caller copy instead of a fixed `未知` literal.
- Existing browser-locale dependencies remain accepted V1 behavior. A
  `zh-Hant` interface under an English browser locale may therefore retain
  English-style number/date formatting; this is not an I18N-6 release defect.
- Localized message grammar, counters, and unit suffixes may follow
  `ui_locale`, while the raw value supplied by the formatter remains under its
  current owner. For example, a `toFixed(1)` elapsed value stays unchanged even
  when its localized seconds suffix changes.

## Inventory

| Owner | Current behavior | Locale dependency | Consumers | Future decision owner |
| --- | --- | --- | --- | --- |
| `timeDisplay.dateParts`, `formatSystemTimestamp`, `formatMarketTimestamp` | Normalizes compact ISO offsets; renders `MM-DD HH:mm` with 24-hour fields. System order is local timezone then ET; market order is ET then local timezone. Missing values render `—`; invalid values pass through unchanged. | Fixed `en-US` formatter; host-resolved local timezone plus fixed `America/New_York`. Independent of `ui_locale`. | Research Evidence, Portfolio Activity, Recent Activity, Account Overview, and Settings storage/provider status. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `ui/BoundedProgress.formatElapsed` | Floors milliseconds to whole seconds and clamps negatives to zero. Values below one minute render `Ns`; longer values render `Nm SSs`. | Locale-neutral digits and fixed unit letters; surrounding labels are localized. | Research progress and Shell background-work progress. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `SourceRunProgress` percentage | Displays `max(0, done) / total`; the bar value is additionally capped at `total`, and the rounded percentage is clamped to `0..100`. Missing, non-finite, or non-positive totals remain indeterminate. | Locale-neutral digits and punctuation; independent of `ui_locale`. | Settings source-run progress. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `App` sidecar `lastOk` | Captures the successful check time with `new Date().toLocaleTimeString()`. | Browser locale and local timezone at capture time; the stored display string does not reformat on an in-place `ui_locale` change. | Shell topbar sidecar status. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `Dashboard` server time | Converts the status timestamp with `toLocaleTimeString()`. | Browser locale and local timezone, not explicit `ui_locale`. | System / Health status tiles. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `ResearchHistoryDrawer.formatLocalTime` | Valid dates render year, two-digit month/day, and two-digit hour/minute/second; invalid source text passes through. | Browser locale and local timezone via `Intl.DateTimeFormat(undefined, ...)`. | Research thread created/updated metadata. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| Research workspace and Evidence elapsed time | Uses `toFixed(1)` seconds and appends a localized seconds suffix. | Locale-neutral decimal point; only the suffix follows `ui_locale`. | Research transcript metadata and Evidence timing. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| Research Evidence token statistics | Uses `number.toLocaleString()` for token-count rows. | Browser locale; not explicitly tied to `ui_locale`. | Evidence Drawer token details. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `Holdings.formatNum` / `formatMaybe` | `formatNum` passes numeric inputs to `Intl.NumberFormat` with at most four fraction digits. `formatMaybe` renders null and non-finite optional values as an empty string before delegating. | Browser locale through `Intl.NumberFormat(undefined, ...)`. | Holdings quantity, average cost, market value, and unrealized P&L columns. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `PortfolioActivity.formatNumber` / `formatAmount` / `formatUnknown` | Finite numbers use at most four fraction digits. Amounts append the source currency code rather than applying currency style. Unknown values use localized caller copy; booleans remain literal `true` / `false`; objects use `JSON.stringify` or the unknown fallback. | Browser locale for numbers; source currency/boolean/JSON values are not localized. | Portfolio activity summaries, execution/commission detail, and field-change detail. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `PortfolioCapturePanel.formatLocalTime` | Missing values render `-`; valid dates use `Date.toLocaleString()`; invalid values pass through. | Browser locale and local timezone; not explicit `ui_locale`. | Capture run timestamps and next-due status. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `PortfolioCapturePanel.formatReviewMetric` | Runtime number values use at most four fraction digits; missing/non-number values render `-`; changed updates render `before → after`. There is no separate finite-value guard before `Intl.NumberFormat`. | Browser locale for numbers; punctuation is fixed. | Capture review table quantities and financial metrics. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `PortfolioAccountOverview.formatAmount` | Missing/non-finite values render `—`. With a currency, uses `Intl` currency style and at most two fraction digits; an invalid currency falls back to a localized number plus the source currency code. Without a currency, uses at most two fraction digits. | Browser locale through `Intl.NumberFormat(undefined, ...)`; source currency controls currency formatting. | Account totals, per-currency values, positions, and P&L. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `PortfolioRecentActivity.formatNumber` | Null values render `—`; other numeric inputs use fixed `en-US` grouping with at most six fraction digits, without a separate finite-value guard. Compact timestamps retain only the first half of shared market/system timestamp output. | Fixed `en-US` for numbers; shared timestamp rules above. Independent of `ui_locale`. | Recent portfolio activity summaries. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| Explore numeric helpers (`Home`, `Watchlist`, `Universe`, `TickerDetail`) | General values use browser grouping with at most two fraction digits. Percentages use `toFixed(2)`, preserve the existing positive-sign rule, and append `%`. Missing values render `—`. Home dates use abbreviated month/day and two-digit hour/minute; invalid dates pass through. | Browser locale for general values and Home dates; locale-neutral decimal point for percentages. Not explicitly tied to `ui_locale`. | Home market summary, Watchlist, Universe, and Ticker Detail. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `News` count formatting | Feed totals, facets, comments, and pagination counts use `toLocaleString()`. | Browser locale; not explicit `ui_locale`. | Market and Seeking Alpha news views. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `settings/DataSourcesSection.formatCount` / `shortDate` | Finite counts use fixed `en-US` grouping; missing values render `—`. FRED snapshot dates use the first ten source characters with no parsing; missing dates render `—`. Body-backlog timestamps use `formatSystemTimestamp`. | Fixed `en-US` for counts; source-text slicing for FRED dates; shared timestamp rules for backlog status. Independent of `ui_locale`. | Data-source FRED summary and body-backlog status. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| Settings storage count/value formatting (`DataStorageSection`, `MacroStorageSection`, `NewsStorageSection`) | Row counts, observation counts, values, and row deltas use `toLocaleString()`; missing values retain each surface's existing `—` behavior. Timestamps use `formatSystemTimestamp`. | Browser locale for counts/values; shared timestamp rules above. | Data Storage, Macro Data, and News Storage settings sections. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| Settings model/provider timestamps | Discovery, model-test, and credential-expiry timestamps use `formatSystemTimestamp`. | Shared timestamp rules above. | Model Routing and Provider settings sections. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |
| `credentialDisplay.isoToDateInput` / `dateInputToIso` | Valid stored dates render UTC `YYYY-MM-DD`; missing or invalid stored values render an empty input. A non-empty date input appends `T00:00:00+00:00`; an empty input stays empty. | Fixed UTC ISO conversion, independent of browser locale and `ui_locale`. | Provider credential expiry editor. | V1 frozen; any ui_locale binding requires a separately reviewed owner-specific unit. |

## Frozen Boundary

Task 9 commit `b6ea67b6eabccdc593d91e04928331634f3647b1` has zero
formatter product or formatter-test diff from its Task 8 input
`6a076db3a9af3f6d3513d95c25efeea76154de06`.

Across product base `93cda66831b7202fd0dfafcc0d1c0604b07e94bd` to final,
the only verified formatter-helper change is the already-reviewed
`PortfolioActivity.formatNumber` / `formatAmount` / `formatUnknown` semantic
fallback localization. The helpers now accept caller-supplied unknown copy
instead of returning or comparing against fixed `未知`; `formatAmount` performs
the same null/non-finite guard directly before formatting. No other inventory
formatter has an equivalent fallback-copy change.

Numeric/date mechanics and their output expectations remain unchanged. This
includes finite checks, `Intl` options, fraction limits, rounding, timezone and
market/system timestamp behavior, source-currency append behavior, and
boolean/JSON handling. The boundary is therefore the exact exception above,
not a whole-formatter-code byte-identity claim.

## Future Change Rule

Formatter localization is not an open I18N-6 task. A future proposal must be
bounded by owner and record exact before/after examples, precision/grouping,
timezone/day-boundary effects, updated tests, and responsive consequences.
No owner inherits `ui_locale` merely because interface copy is bilingual.
