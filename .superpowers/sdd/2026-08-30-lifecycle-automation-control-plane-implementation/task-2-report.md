# Task 2 Report: Complete Human-Accepted Finalization Truthfully

Date: 2026-08-30
Implementation: `3a9ce638`

## Result

- Human acceptance of an automation-authored assessment is now a valid,
  idempotent terminal state. Its authority remains `human`.
- Finalization failure remains on the succeeded run; `fail_run` stays
  forbidden once a current assessment exists.
- The shared closed metadata shape is
  `terminal_finalization_failure={code,failed_at,attempt_count,retry_not_before}`.
- Automatic retry intervals are 15 minutes, 1 hour, and 6 hours. A fourth
  recorded failure requires an attended Run again.
- Pending finalization projects to Attention and exposes only the four safe
  metadata fields; raw query context and exception text remain hidden.

## RED

The initial five-test selection returned `5 failed`:

- human-accepted recovery returned `accepted=0`;
- no finalization failure metadata existed;
- accepted state projected to History;
- provider-neutral output omitted safe failure state;
- the `fail_run` owner initially exposed a fixture citation error, which was
  corrected to use the persisted evidence ID and then passed against baseline.

The same selection returned `5 passed` after implementation.

## Verification

- Task 1 + Task 2 focused suites: `198 passed in 9.35s`.
- Direct investigation/routes/listing suites: `121 passed in 8.99s`.
- Compileall: clean.
- `git diff --check`: clean.

## Reverse Mutations

Each mutation was applied independently and restored:

1. Remove `human` from accepted authorities: named human-finalization test failed.
2. Bypass the retry-not-before gate: bounded-backoff test failed.
3. Bypass finalization Attention priority: disposition test failed.
4. Accept an unknown metadata code: closed-validator test failed.
5. Omit safe metadata from provider-neutral output: tools test failed.

## Boundaries

No schema migration, provider call, production database access, App start,
merge, or push occurred. Failures before any assessment exists remain outside
this metadata path, as required by the design.
