# Calibration Anthropic Refusal Evidence

> **Status:** LIVE COMPLETE - MERGED 2026-07-25
> **Date:** 2026-07-25

## Boundary And Commits

- Plan-review clearance:
  `d19d964218dcc84bdd8aa908e27b577c8f079fdd`.
- Response-seam product commit:
  `5fa47c5b63c42bfa5cb88a5631b84bd2e43de028`.
- Route/durable-state product tip:
  `1a634ced487fb997dfe8fbd606a4bdd453996ad2`.
- Review-ready isolated branch/worktree: `codex/calibration-anthropic-refusal` at
  `/home/hyl/.config/superpowers/worktrees/ArkScope/calibration-anthropic-refusal`.
- Before independent review, no merge, push, production database access,
  Provider call, prompt change, frontend change, schema change, retry,
  fallback, or effort-policy change was performed.

## RED-First Proof

### Response seam

The new node
`test_anthropic_calibration_raises_structured_refusal_before_text_extraction`
uses an Anthropic-shaped response with `stop_reason="refusal"` and deliberately
valid JSON content. Before the product branch, the exact node failed only with:

```text
Failed: DID NOT RAISE <class 'src.anthropic_refusal.AnthropicRefusalError'>
```

Credential resolution and text parsing both succeeded, so the RED was the
missing stop-reason branch rather than an adjacent setup failure.

After the branch, the direct node plus the three existing synthesis/translation
refusal nodes and the stream refusal node passed: `5 passed`.

### Route and durable state

The new node
`test_calibration_refusal_records_model_refusal_instead_of_generic_failure`
injects `AnthropicRefusalError` with planted private model/category/detail
values. Before the typed catch, it failed because the route returned and stored:

```text
code: calibration_responder_failed
message: Calibration responder failed. Retry this turn.
diagnostic: Provider call failed.
```

After the catch, the new node and three existing generic/retry/privacy route
contracts passed: `4 passed`. The responder is called once, the turn remains
retryable, and none of the planted values appears in the turn or HTTP payload.

## Exact Node Ledger

Raw pytest collection is:

| Set | Clearance | Product tip | Delta |
|---|---:|---:|---:|
| Full backend | 4711 | 4713 | +2 |
| Calibration focused | 46 | 48 | +2 |

Sorted raw pytest node-ID SHA-256 values:

| Set | Clearance | Product tip |
|---|---|---|
| Full | `e6b41ef56cea3cfbfb2a67bda1b4ea2e96e968cbde161e30be297145f946e9a1` | `a3b91ea6eed808afb7aa7dc860a9f5f8e30de9dd770a9f06245c35d0f04a5d6a` |
| Focused | `0f877bbe6709e217030cd57927ef8914ed390fa85eedb4d8cbf5b16506ba342b` | `f27158a2f11a5a6b9cf4c57de55a2071a737a1bd8aa57fa8a8c0a4f244a535a2` |

Additions are exactly:

1. `tests/test_investor_profile_calibration.py::test_anthropic_calibration_raises_structured_refusal_before_text_extraction`
2. `tests/test_investor_profile_calibration_routes.py::test_calibration_refusal_records_model_refusal_instead_of_generic_failure`

There are no removed or renamed nodes. The focused behavior run is
`48 passed`.

## Equal-Environment Virgin A/B

Both sides were clean `git archive` trees of the clearance commit and product
tip. Both mounted the same repository root and web `node_modules`; neither had
an untracked `.env`, and both had the same tracked `data/` and `config/`
presence.

The first full runs exposed the repository's registered `test_agents` fixture
classification instability:

| Side | Passed | Failed | Errors | Skipped | Warnings |
|---|---:|---:|---:|---:|---:|
| Clearance raw | 4606 | 31 | 0 | 74 | 18 |
| Product raw | 4602 | 30 | 7 | 74 | 18 |

Every differing node was in `tests/test_agents.py`. Fresh archive copies then
ran that file alone on both sides in the same order. Both produced exactly
`23 passed / 1 failed / 7 errors`; their normalized node-and-status SHA-256 was
identical:
`107dddc7744f7abe6dae5db3dd5ec354951ac0e2415569d875bb980f1d967502`.

Replacing only that unstable family's full-run classifications with the
fresh same-file observation yields:

| Side | Passed | Failed | Errors | Skipped |
|---|---:|---:|---:|---:|
| Clearance canonical | 4600 | 30 | 7 | 74 |
| Product canonical | 4602 | 30 | 7 | 74 |

The passed delta is the exact `+2` collection delta. Common-node status drift
is empty, removals are empty, and the 74-node skip set is byte-identical with
SHA-256
`9fe883c3c11b406d97045d229cd512ec50afaa4bef25f09dd7d7d1678e9a4159`.
Both terminal warning summaries contain the same 18 warning sites: the one
`TestResult` collection warning, three Edgar deprecations, six Tiingo
return-value warnings, and eight yfinance return/future warnings. Absolute
failure/error counts remain environment observations, not allowlisted values.

## Boundary Gates

- `python -m src.smoke.pg_unreachable_e2e`: `ok: true`, `pg_attempts: []`.
- `git diff --check` from clearance to product tip: clean.
- Protected paths are byte-identical: shared refusal helper; synthesis;
  Anthropic agent loop; calibration store/policy/schema; all web app files;
  extensions; package manifests and lockfile.
- Authorized product paths are exactly:
  `src/investor_profile_calibration_agent.py` and
  `src/api/routes/investor_profile_calibration.py`.
- Authorized tests are exactly the two corresponding calibration test files.
- Product diff shape is one helper import plus one pre-extraction refusal
  branch, and one typed import plus one fixed diagnostic helper and typed catch.

## Deferred UI Debt

The backend now records and returns `model_refusal`, but
`InvestorProfilePanel.tsx` still maps every turn failure to the single generic
localized title. EIR-004 records that bounded UX debt for the next Investor
Profile-owned UI slice; this implementation does not authorize frontend work.

## Independent Review And Merge Closeout

Independent implementation review returned GREEN with zero findings. `master`
fast-forwarded through this evidence tip,
`8e73dba45127adf5ef8bbfdbceda45c775d7a295`.

The merged tree then reproduced focused `48/48`, backend collection `4713`,
and no-PG `ok:true` with `pg_attempts:[]`. Merge and closeout did not access a
production database, call a Provider, or change prompts, frontend code, or
schema. EIR-004 remains open. Coverage v2 follows only after a fresh inventory
resolves the authority for per-day expected session length; this closeout does
not preselect an authority or promote observed bar counts into constants.
