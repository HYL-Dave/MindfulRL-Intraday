# Legacy-Agent CLI and Entrypoint Census Evidence

> **Status:** TASKS 0-1 REVIEWED GREEN; TASK 2 STOPPED BEFORE AUTHORITY
> GENERATION FOR NORMALIZED-AUTHORITY AMENDMENT REVIEW; TASKS 3-4 NOT
> AUTHORIZED; NOT MERGED; NOT PUSHED
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

Task 0 passed independent review. Task 1 was then authorized as the next
docs-only census stage.

## Task 1 - Raw candidate universe

The accepted packet is
`/tmp/legacy-agent-cli-census-task1-dd210d65`. It contains 69 manifested
payloads. `SHA256SUMS` has SHA-256
`a98d9f4b7be8d6595a5b4ccb4d7168c2d12e62d466056e594ab8d52434be6786`;
every entry verifies.

Two independent static extractor processes read the clean detached source
worktree at `241ccdba` and produced byte-identical output files. The canonical
raw stream contains 295 candidates across 14 source families and 90 paths,
SHA-256
`ed539ba7bef49350986a296cd2c134ecb75933e6f0344236c23e0507a401ed48`.
No product module was imported or executed.

```text
browser_manifest                    20  b7d6afe635847d7bd5bf74f983b905ae92aeb6fd832009463826d62c28e20587
desktop_manifest                     1  3e4c999b7b7ea5cf12cfa480acf9ab13f92c69892ad53224d2754e73eb7fe790
documented_command                  52  659c181484c65936a8f4e524b4c4255b7ea8dc444746df49614db50e75eecbfe
file_executable_mode                 2  144aed1cbaf76dbfb6ced8efc2f80c1acd3eef263f36dde8cf1c60bfffb26928
file_shebang                        15  965d33af79afef76596e7280cdfd854763e9b43d32081e877df76d161eb05cdd
generated_native_manifest_contract   6  f3f818b639b2a988ce4fc9520c86423fd42203a5a642c2bfd2c6358d7933c40e
javascript_subprocess_target         3  058234a115b5295fd22870eb9a4755090b55c7f18f3dadf12a5440782eebeaae
npm_script                          12  e4029dceaf0e016556575b2351d706d6880bdc9774b6041f48e80d2949ce5e69
python_main_guard                   19  49a5aa8aa7ad852430ce8380fea0df0827605d390af26fd08b075fa0da9b3852
python_module_wrapper                2  2d84510018a37e8c033c984244e3d101139166c68c2b6e9e21550a659e2149c1
python_parser                       10  ce5560bff4fc7ab9bba53ad027777180a850eebf2e9ccde827f9b6fd628f791f
python_subprocess_target            14  fa7a1a5c3b4b9d5b0b4c35f59571c67ce3cdde107140f7ec1fdfc0f94c3a84b8
shell_path                           4  2ba21767a5bddf77e020510804df814747b118b21b78d424ef679a019a418118
test_consumer                      135  6057505bc4d81e1623373c9fb56be91a51325d5cf6e3976fe162e9ad867f0d8e
```

The 135 test observations cover 32 files and join to 450 exact canonical test
IDs, SHA-256
`cf1c16316f0152ece67b38fc23d55e01c30044cc27bdb53fb9a886d3c7baf295`.
Helper- and fixture-scoped observations follow same-file static reachability;
they are not assigned to every test in a file. A discriminating synthetic
contract reduced the CLI overflow helper's import from 39 false file-wide joins
to its four actual callers. Task 2 still owns final exact capability test
ownership; these raw joins are grounded search inputs, not dispositions.

All 13 reconstructable literal/floor streams plus the capability and protected
baselines pass the independent validator. The candidate universe contains 47
explicit overlap groups. Seven unresolved dynamic launches remain present at
`dc5ceadba15ee28e47961705fe1909642e9fc984318da0612c149cf940e8049b`;
Task 2 must trace or stop on each rather than treating it as absent.

## Extractor verification and safety

Thirteen synthetic extractor contracts pass. Twelve branch mutations are each
killed by their named owner. The additional contract distinguishes a helper's
real callers from unrelated tests in the same file. Both full scanner runs,
their family summaries, joins, overlaps, dynamics, source coverage, encrypted
minimization records, and validation reports compare byte-for-byte.

Six pre-admission generations were rejected for broad aliases, namespace
prefix matching, missing global/local command resolution, an unresolved
starred tail, file-wide helper joins, or copying a private absolute shebang.
Only bounded count/hash receipts enter the admitted packet. Raw rejected output
was quarantined outside it because it contained the private source shebang. The
accepted row records only `python3` and a `private_absolute_redacted` marker.

The final leak audit reports zero private path, email, provider credential,
database URI, or non-fixture token matches. The one JWT-shaped value is only
the already pinned synthetic redaction-test node ID in the canonical backend
collection. The source, implementation parent, and main worktrees were clean;
`master` remained `241ccdba`, origin remained `e2ead437`, the 942 protected
rows stayed byte-identical, no product/test byte changed, and no census, CLI,
Discord, provider, or app process remained.

## Task 2 stop - normalized authority was underspecified

Independent Task 1 review returned GREEN and authorized Task 2. Before writing
tracked census authority, the first classification generator run failed loud on
an unmapped path-form test target. Corrected diagnostic runs then exposed a more
fundamental plan defect: `capabilities.tsv`, `consumers.tsv`, `tests.tsv`, and
`current_invocations.tsv` contain detail fields absent from the fixed
`entrypoints.jsonl` schema. They therefore cannot be regenerated solely from
that JSON ledger as plan Section 0.5 and validator item 20 originally required.

Execution stopped before any tracked census authority or product byte was
created or changed. The bounded amendment makes those four files normalized co-authorities
generated with the entrypoint ledger from exact source/candidate inputs,
retains `recommendations.tsv` as the pure ledger projection, and requires
bidirectional foreign-key closure plus two byte-identical full generations.
Task 3 still independently reconstructs every file. Candidate universe
`295/14/90`, Task 1 joins, schemas, source base, product/test protection, and all
retirement gates remain unchanged.

Two rejected diagnostic attempts are not admission evidence: a shell cleanup
command was rejected before execution by the command-safety boundary, and the
first generator run rejected an unmapped path-form test target rather than
guessing a fallback. Packet-local draft outputs are untracked and will be
recreated from scratch only after amendment review.

## Next gate

Focused review of the normalized-authority amendment is required before Task 2
resumes. Canonical tracked inventory, recommendations, CLI/Discord/skill/operator
disposition, product or test edits, retirement, merge, push, live commands,
provider traffic, secret handling, and destructive operations remain
unauthorized. The MCP/HTTP model-callable-interface direction remains
recommendation context only.
