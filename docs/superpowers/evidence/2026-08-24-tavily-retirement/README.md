# Tavily Retirement Admission Evidence

Generated at `2026-08-25T00:43:11+08:00` from branch
`security-lifecycle-automation` in the isolated worktree
`/tmp/arkscope-lifecycle-automation`.

## Result

Stage 1 is mechanically admitted for focused review.

- Base product: `64af5092dd22523c672b8c42e3b84eaba04bec1f`
- Admission tip before this evidence-only closeout:
  `4f6959e2058bc546df12635a336ed1b7f31e5600`
- Backend collection: `4324 -> 4294`
- Backend A: `4282 passed / 12 skipped / 3 existing warnings`
- Backend B: `4282 passed / 12 skipped / 3 existing warnings`
- Frontend: `104 files / 1220 tests`
- Runtime: `184` routes, `50` registry tools, `51` tools in each bridge
- General web tools: only `web_browse`
- Tavily key names in tracked current-product paths: zero
- Provider calls, live preflight, live migration, merge, and push: none

The remaining current-product Tavily literals are closed to the dormant schema
value, the explicit read-only retirement preflight, and two UI assertions that
the retired label is absent. Historical design/decision records were not
rewritten.

## Rejected Run

The first full backend attempt is not admitted. It ended with
`4251 passed / 12 skipped / 31 failed`: 30 failures came from the isolated
worktree lacking a root `node_modules` resolution path for `jsdom`; one failure
proved that `p1_4_l0_overflow.json` had changed its tool name to `web_browse`
without changing the old search arguments/result shape. A gitignored root
dependency symlink fixed the environment. The fixture was corrected to the
real `web_browse(url, extract_links, max_chars)` contract, after which the four
affected SA extension files passed `40/40` and replay/observability passed
`95/95`. Only the later independent A/B runs are admitted.

## State Boundary

All test commands ran from the isolated worktree. Its resolved data root is
`/tmp/arkscope-lifecycle-automation/data`; the two admitted pytest roots were
`/tmp/arkscope-tavily-retirement-a-final` and
`/tmp/arkscope-tavily-retirement-b-final`. No command received a production
database path. The running App was not stopped or inspected, and its production
databases were not read or hashed for this offline gate.

See the adjacent structured summary, node-set diff, runtime inventory, command
ledger, and commit chain.
