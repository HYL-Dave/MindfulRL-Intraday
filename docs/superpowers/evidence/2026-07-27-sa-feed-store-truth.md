# SA Feed Store Truth Evidence

> **Status: IMPLEMENTATION IN PROGRESS - NO PRODUCTION WRITE - INDEPENDENT REVIEW PENDING**

## 1. Authority And Isolation

- `PLAN_REVIEW_CLEARANCE_COMMIT`:
  `a364268c8dcbcc80a38842669a6545198fae8e3d`.
- Branch: `codex/sa-feed-store-truth`.
- Worktree: `/tmp/arkscope-sa-feed-store-truth`.
- The initial direct checkout stopped at the known linked-worktree
  `git-crypt` smudge boundary. The clean retry used `--no-checkout`, copied
  only `.git/git-crypt/keys/default` with mode `0600`, and populated `HEAD`
  with `git read-tree -mu HEAD`. The protected evaluation document is
  `47,608` bytes.
- Existing root and web `node_modules` are linked into the worktree. No
  dependency install or lockfile change occurred.
- No production database, `config/.env`, browser profile, token, user-owned
  data, provider, Gateway, PG service, or external endpoint was read or
  changed. The checkout contains only the repository's tracked default config
  files. A missing worktree-only `data/` directory was created after one
  tool-count fixture could not auto-detect a project root; no production data
  was copied into it. The no-PG smoke then created its own 143,360-byte empty
  profile store, which passed integrity, was identified as an ignored
  worktree artifact, and was deleted before product RED tests.
- The untracked root document
  `docs/design/SCRIPTS_RETIREMENT_DECISION.md` was not copied or staged.

## 2. Task 0 Baseline

All normalized collections reproduce the reviewed authority exactly:

| Gate | Result |
|---|---|
| Backend full | `4691`; `ed4b7da05db79204dd847d33d0d9f9bb8f6bbef6c756af48cf218a13f3525acf` |
| Backend focused | `77`; `34a30e6d54c108fadfe4e0425d863c9a6fbfaf1b7f10a93ee82f53d380d3eb2a` |
| Backend focused run | `77 passed` |
| Frontend full | `96` files / `1072` nodes; `71e4785f75ace3d65e40a479ce823897ffbcae0bd27ff1855aef1504905e429e` |
| Frontend focused | `2` files / `25` nodes; `086cce183d540193a966a61148f6e7a9e6c2177a8ebecd49bb71c2c1cfc6d892` |
| Frontend focused run | `25 passed` |
| Scanner run 1 | `36 / 20 / 0 / 20`, scope `src/**` |
| Scanner run 2 | `36 / 20 / 0 / 20`, scope `src/**` |
| Tool gates | three named tests passed; inventories `53/54/54` |
| no-PG | `23/23`, `ok=true`, `pg_attempts=[]` |

The first tool-count attempt produced one fixture-construction error because a
virgin worktree lacked `data/`; tracked default `config/` was already present.
After creating the empty worktree-only data directory, the exact three-node
gate passed. This is recorded as environment setup, not a baseline product
failure.

## 3. Protected Byte Baseline

Tracked-file blob IDs:

```text
src/sa_capture_store.py                         f4eacd5a5746ec96e5f945ecff33cb4c0df1448c
src/api/routes/seeking_alpha.py                 aa75e20a877980257a80f4887a193ab07ec97e63
src/sa/extension_run_protocol.py                fd367b101841f85f2507a71fce932abb31c7eb5b
src/sa/market_news_recovery.py                  1a69bbdf19b3047d0f8d239e171c9ffbae31fedd
src/service/jobs.py                             f58dd01df4240eaf8f0dd3897036b40d180156a1
src/sa_native_host.py                           88712d75076c4d13d832c623c6597c1fab0a1116
apps/arkscope-web/src/styles.css                c2649cbf2a874521721d40f7e1d5d4b392a2d1ba
apps/arkscope-web/src/ui/primitives.css         d92000dc4687eed79b7c7f3319ba9e2a973b0fab
apps/arkscope-web/src/shell/shell.css           80cbacf2e6d986add0d358b447049e961c5f4dce
apps/arkscope-web/src/ui/tokens.ts              ef4f46565f7d11aa9b94bc2bdca9084bde131f2e
apps/arkscope-web/src/ui/tokens.json            144262e61a023e56103a5c3aa1a9bf6eea404436
```

Sorted `git ls-tree -r` family hashes:

```text
extensions/sa_alpha_picks                       40eac710c4e85f6a5773b7ebda7e6eb67b86b0946dbb21a6fc7111ad4851f585
apps/arkscope-web/scripts/i18n                  9757396abe7bcac8ddfc0981f032a3333007e0e425242408d06bb7d18897b33d
sql                                              bcf30e4d419c351082566bfbeb94dcd2f4366bae57dc22d5b2d528e397e9c40f
```

Scanner artifact SHA-256 values remain:

```text
migrated-scopes.json             02e335bebcadfba523d502a7af86a5c184d1ac024230cfec9199dd19b4416c13
visible-literal-allowlist.json   3b397a21ab7f8a1cd37819ae55d892e26f1946dc3c791aebf28d2eba2577c212
visible-literal-debt.json        d6eaaf3e70bd344e8c3bd2d89dcc9818081e2735db9191d31dd5757246868cec
visible-literal-scanner.mjs      c22c7e784c6f1c25587a980ca7b441658f58632a004d117985e765cad70fb8da
```

## 4. Grounding Correction

Task 0 found that the reviewed protected list contained four stale nonexistent
paths: `src/sa_native_manifest.py` and root-level `primitives.css`, `shell.css`,
and `tokens.css`. Before product work, the plan was corrected to the current
owners `src/ui/primitives.css`, `src/shell/shell.css`, and
`src/ui/tokens.{ts,json}`; the nonexistent manifest path was removed. This is a
docs-only gate repair with no product, node, resource, or scope change.
