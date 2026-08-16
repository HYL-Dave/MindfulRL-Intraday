# Legacy-Agent CLI and Entrypoint Census Evidence

> **Status:** TASK 0 COMPLETE; INDEPENDENT TASK 0 REVIEW NEXT; TASKS 1-4
> NOT AUTHORIZED; NOT MERGED; NOT PUSHED
>
> **Product/candidate source:**
> `241ccdba6dc7c2cf1b162dd254ada88f25b6a9b0`
>
> **Reviewed design:** `8321207cf5f15baa0b70f9394ed6d3ae30135206`
>
> **Reviewed implementation plan:**
> `692f46cef02dc5102fde19d427315cb55c750147`

## Task 0 - Exact docs-only baseline

The implementation branch starts at the reviewed plan tip. Its exact merge
base is product source `241ccdba`; the intervening range has zero merge commits
and changes only the reviewed design, implementation plan, and priority map.
Local `master` remains `241ccdba`, `origin/master` remains `e2ead437`, both
trees were clean before evidence edits, and no product or test byte changed.

The isolated packet is
`/tmp/legacy-agent-cli-census-task0-241ccdba`. It contains 84 manifested
payloads. `SHA256SUMS` has SHA-256
`f73b0a71dfca8ed6248ab1867e5a5315008957fc56f4ae8380fdffeb7178eb48`;
every entry verifies.

## Collection identities

Backend discovery used the pinned EIR-002 reporter, a fail-closed socket guard,
and a packet-local `MetaPathFinder` that rejects `src.agents.__main__` before
execution. It collected exactly 4,278 nodes at
`ecafdab7a1cee8d6f64dd6763f017d2ef15dd414b80065950f949d8b471a09ce`,
with `seen=0`, `nonpassing=0`, exit zero, zero socket attempts, and zero blocked
module attempts. A synthetic decoy module proves the import blocker fires
before module code can write its execution marker; no harmless replacement was
preloaded into `sys.modules`.

Frontend discovery used the root-hoisted Vitest 4.1.8 binary directly and the
pinned normalizer. It produced 1,177 nodes across 101 files at
`c570a551b64ed95155c02f83499e78eb3409f2cba66ea9d46862dffad0ea239b`.
Two raw lists differed only in discovery order; their normalized streams are
byte-identical. The admitted run has a separate exit-zero receipt. The four
CLI/Discord owner files project to 209 nodes at
`a57b4414d626d15ba37c21326e40c78dce70b76e55e212d8a72c674e1cedab0a`;
the Discord owner file projects to 92 at
`30ce40ca6db5e2bd6d139351b2b06a8c0643529189255f2f6cb25386fb8012d0`.

## Literal grounding and protection

Two independent source-blob rebuilds reproduce every pinned stream. Explicit
plan code blocks compare byte-for-byte, and the 52-row Markdown command stream
also has an independent POSIX-ERE reconstruction equal to the Python result.

```text
main guards, non-test       16  2a3fe77398bf40d399781c1fdd25cc0a27a383b75a55ea54d51eda54647b42c6
main guards, all tracked    19  e6d94b2fa319b7fab587bf372af9d6b3a8b6b3a02465680e65a9cf0de4e41b7c
parser paths                10  9dc2361e00fac587f9a6b7af6de3a32c80204d52d89268193000e1c646218df8
shell paths                  4  c7bf1f1a5751addec5f12e046563202bcd58d00d1d2c2af54a5871e8e9b4924a
package manifests            3  1726c5e1f94cfca45b519de6998d3487a8fb118605ef984ba5f8effa664e0d30
executable-mode paths         2  c51acae446809281a36386c05cfb83e94249f4613ea60f3e578628c5e603831e
package __main__ paths        2  720781cd2667caa075a8c6c4c6268584af5892a84ee832e699ae284a9ffc075a
shebang paths                15  3f53bfa0d770fb38e8dea167fe345f1f4b69d78b6434a15a3429289d8c45bf63
npm scripts                  12  2d60d0890af322f01a077f050c4215020619770a066403b2095a748db48057fb
manifest targets             13  efcb8df26aadea16204d7fcd140e2b06afeabd4db5c017daed089c27ce05130b
README command tokens         8  f5ad84174df725b22c94ba6b5d7f30d20d78714169193d1b8982fe2edd4642ef
Markdown commands            52  9a36d3138ec8ee6dea90ab60539ad323d639aab0ffe3ef93296d6519e8e4ba8a
Markdown command paths       15  37906c5b4e37b9d4b0d0dfdf73912df4817a540a016d21890117edd3cf7c68f9
required launch edges         5  e8474c73b3d7f867ede01a14ad938f0ccfec4719fb96b081e6313e5676feeb88
capability floor             28  d799dbab042593f149a851fdd3ddc17783dcd90559f8eecbc4429d05b0b2d2f7
protected base rows         942  76665c8a39b514e1896613edeb416c0fded416944c9cfc44c89f9fa1750a0eea
```

All eight grounded source files match their reviewed line counts and SHA-256
values. Static AST inspection confirms the unconditional `main()` call at
`src/agents/__main__.py:17`, import-time `_load_env()` at
`src/agents/cli.py:72`, zero importers of the dangerous module, and zero
non-test constructors or `start_bot()` callers for `MindfulDiscordBot`.

The three git-crypt blob IDs match between source and the unlocked main tree.
Only normalized command tokens, path, line, and per-file command count enter
the packet; no surrounding encrypted plaintext is retained. The encrypted
paid-subscription document contributes exactly two command observations, while
the other two encrypted documents contribute zero.

## Isolation and rejected artifacts

The worktree has no `config/.env`, no app-local `node_modules`, and no link to
main `data/`. Its real `data/` directory is empty; HOME, XDG, runtime, profile,
token-store, and lock paths are packet-local and contain no produced runtime
files. No CLI, Discord bot, provider, collector, scheduler, browser, desktop,
sidecar, package script, or test body ran.

The leak audit reports zero email, private-home, PostgreSQL URI, JWT, or secret
key matches outside the exact synthetic redaction-test node IDs. Those fixed
fixtures occur only at their pinned `1 JWT + 2 key-shape` count in each of four
canonical backend artifacts; any other location or count remains a failure.

Five rejected tool attempts remain manifested: one frontend wrapper without an
observable exit code, one Markdown heading locator that ignored a line wrap,
one mistyped Discord projection hash inside the packet tool, one `splitlines`
versus `wc -l` interpretation error for files without final newlines, and the
initial over-broad leak classification of synthetic redaction node IDs. None
contributes to admission; each correction was followed by a complete rerun.

## Product-direction context and next gate

The user clarified that future value lies in preserving model-callable
capabilities through a defined interface; MCP and HTTP API are both admissible
delivery mechanisms, and whether they converge is a separate architecture
decision. This is decision context for Task 2 recommendations, not a Task 0
classification or retirement authorization. CLI wrappers, Discord, skills,
operator commands, and shared capabilities remain unclassified.

Task 0 now stops for independent review. Task 1 candidate extraction, product
or test edits, retirement, Track B/skill/Discord policy, merge, push, live
commands, provider traffic, secret handling, and destructive operations remain
unauthorized.
