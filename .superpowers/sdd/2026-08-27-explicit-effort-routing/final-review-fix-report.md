# Final review remediation report

## Scope

One bounded RED-first remediation round for requirements A-G, starting from
`1c8fbd613cbbe5cf7e65bbda673cd41ad1bab25b` on
`lifecycle-ux-honesty`.

## RED evidence

The following commands were run after adding focused owners and before changing
product code:

```text
pytest -q tests/test_model_routing.py -k 'shared_task_route_admission or catalog_exposes_canonical_current_and_retired_model_policy or explicit_profile_values_equal_to_builtins or preserves_unsupported_profile_effort'
```

Result: `10 failed, 71 deselected`. Missing shared admission/lifecycle facts,
incorrect default source attribution, and unsupported effort normalization were
all observed.

```text
pytest -q tests/test_card_synthesis.py -k 'fixed_task_seams'
```

Result: `18 failed, 6 passed, 31 deselected`. Every invalid route reached a
monkeypatched provider seam; current and unknown-custom positive controls passed.

```text
pytest -q tests/test_research_routes.py -k 'query_sync_rejects_invalid or query_sync_dispatches_current or query_sync_unknown_provider or query_stream_rejects_before_response or query_stream_dispatches_discovered or query_stream_unknown_provider_is_400 or internal_research_stream or research_runs_reject_invalid or research_runs_admit_custom'
```

Result: `13 failed, 7 passed, 46 deselected`. Legacy sync/stream admission,
pre-response validation, explicit effort forwarding, unknown-provider timing,
and the internal stream guard failed; existing run-route controls passed.

```text
pytest -q tests/test_model_effective.py -k 'v2_effort_options_are_model_specific or v2_entry_schema_and_grouping'
```

Result: `1 failed, 1 passed, 21 deselected`. Unknown capability facts serialized
`effort_options: []` instead of omitting the field.

```text
npm test --workspace apps/arkscope-web -- src/researchModels.test.ts src/researchSelection.test.ts
```

Result: `7 failed, 36 passed` across `2` files. Shared lifecycle matching was
absent and provider identity was not enforced.

```text
npm test --workspace apps/arkscope-web -- src/ResearchWorkspace.test.tsx src/SettingsModelRouting.test.ts src/i18n/researchPresentation.test.ts
```

Result: `5 failed, 65 passed` across `3` files. Retired recovery, stale delete
state, and literal historical effort presentation failed.

## GREEN evidence

Focused owners after implementation:

```text
pytest -q tests/test_model_routing.py -k 'shared_task_route_admission or catalog_exposes_canonical_current_and_retired_model_policy or explicit_profile_values_equal_to_builtins or preserves_unsupported_profile_effort'
```

Result: `10 passed, 71 deselected`.

```text
pytest -q tests/test_card_synthesis.py -k 'fixed_task_seams'
```

Result: `24 passed, 31 deselected`. The later explicit-override expansion also
passed as `30 passed, 27 deselected` with:

```text
pytest -q tests/test_card_synthesis.py -k 'fixed_task_seams or provider_override_uses_explicit_task_effort'
```

```text
pytest -q tests/test_research_routes.py -k 'query_sync_rejects_invalid or query_sync_dispatches_current or query_sync_unknown_provider or query_stream_rejects_before_response or query_stream_dispatches_discovered or query_stream_unknown_provider_is_400 or internal_research_stream or research_runs_reject_invalid or research_runs_admit_custom'
```

Result: `20 passed, 46 deselected`. A subsequent current/custom stream expansion
is included in the final backend suite below.

```text
pytest -q tests/test_model_effective.py -k 'v2_effort_options_are_model_specific or v2_entry_schema_and_grouping'
```

Result: `2 passed, 21 deselected`.

```text
npm test --workspace apps/arkscope-web -- src/researchModels.test.ts src/researchSelection.test.ts
```

Result: `2` files passed, `43` tests passed.

```text
npm test --workspace apps/arkscope-web -- src/ResearchWorkspace.test.tsx src/SettingsModelRouting.test.ts src/i18n/researchPresentation.test.ts
```

Result: `3` files passed, `70` tests passed.

Final affected-suite verification:

```text
pytest -q tests/test_model_routing.py tests/test_model_effective.py tests/test_card_synthesis.py tests/test_research_routes.py
```

Result: `228 passed in 6.35s`.

```text
npm test --workspace apps/arkscope-web -- src/researchModels.test.ts src/researchSelection.test.ts src/ResearchWorkspace.test.tsx src/SettingsModelRouting.test.ts src/ModelRoutingSection.test.ts src/modelRoutingUx.test.ts src/i18n/researchPresentation.test.ts
```

Result: `7` files passed, `146` tests passed.

```text
npm run typecheck --workspace apps/arkscope-web
```

Result: `tsc --noEmit` passed.

## Limitations

None identified within this bounded remediation scope. No provider, network,
production database, App, browser, merge, or push operation was performed.
