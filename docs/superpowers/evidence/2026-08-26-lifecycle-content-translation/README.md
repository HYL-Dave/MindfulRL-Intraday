# Lifecycle Content Translation Admission Evidence

This packet admits the shared Content Translation route and the lifecycle
evidence presentation entirely offline. It does not authorize or perform a
provider call, production database read or write, migration, App restart,
merge, or push.

## Authority

- Product/test authority before this docs packet:
  `3a841dce45283b4713fd1e391fe45a3a13bcd223`
- Browser app: the isolated worktree Vite frontend at `127.0.0.1:4198`
- API authority: fixture responses intercepted in the browser; the production
  backend was not started
- Provider calls: `0`
- Production database operations: `0`
- External browser requests: `0`

The browser failure fixtures return synthetic non-2xx `Response` objects from
the page's `fetch` boundary. They still exercise the production
`fetch -> ApiError metadata parser -> LifecycleView` path while avoiding
Chromium's expected resource-error console noise. Successful translation uses
the intercepted loopback API route. The cached scenario begins with a stored
translation and issues no translation mutation.

## Automated Admission

Focused backend admission ran twice with isolated pytest base directories:

```text
103 passed in 7.25s
103 passed in 7.21s
```

Full backend admission:

```text
4434 passed, 12 skipped, 3 warnings in 248.46s
```

The three warnings are existing `edgar` v6 deprecation notices. Frontend
admission:

```text
105 files / 1233 passed
typecheck: passed
visible i18n literal scanner: passed
production build: passed (2193 modules)
```

The i18n resource ledger changed from `699` to `710` Explore leaves, from
`312` to `323` lifecycle leaves, and from `2198` to `2209` total namespace
leaves. This accounts for the eleven new reviewed translation strings.

## Browser Matrix

`run_browser_matrix.py` executed these seven scenarios in English and
Traditional Chinese at `1440x900` and `390x844`:

- successful adjacent translation
- stored cached translation
- retryable timeout
- authentication rejection
- quota exhaustion
- invalid model output
- evidence-content conflict

All `28` entries produced a distinct screenshot and passed dimension,
nonblank-pixel, viewport overflow, drawer bounds, visible-control overlap, and
text overflow checks. Every entry retained the original source excerpt. All
eight success/cache locale-and-viewport entries displayed the translated
excerpt. The matrix observed:

```text
external requests: 0
unexpected mutations: 0
console errors: 0
page errors: 0
provider calls: 0
production backend starts: 0
production database operations: 0
```

Representative desktop/mobile, English/Traditional-Chinese, success/failure
screenshots were visually inspected. Error responses expose only reviewed copy
and bounded provider/model/harness identity; the fixture diagnostic sentinel
does not render.

## Limitation

No live translation provider was called. This packet proves routing contracts,
safe error classification and metadata handling, cache presentation, bilingual
copy, responsive layout, and browser behavior. It does not prove that any live
credential, provider model, or subscription harness can complete a translation.

`SHA256SUMS` covers every packet payload except itself. Its digest is reported
separately after generation.
