# OAuth Usage Recovery and Sticky Inset Evidence

> **Status:** TASK 0 COMPLETE - INDEPENDENT CODEX REVIEW REQUIRED
>
> **Date:** 2026-08-10
>
> **Plan authority:** `953ea7e7` (plan GREEN by Codex review; plan file
> SHA-256 `94d18b33c391a2ad2f737eed4d5bb97bdd800a27866c997b6d888cf284a89975`)
>
> **Product grounding base:** `8cf85597` (= `master`)
>
> **Task 0 artifact root:** `/tmp/oauth-usage-sticky-impl-task0-953ea7e7`
>
> **Implementer:** Fable (design LD 11); reviewer: Codex.

Task 0 changed no product or test byte. Tasks 1-7 remain unstarted and
unauthorized until this re-grounding receives independent GREEN review.

## 1. Boundary and toolchain

- Branch ancestry: six docs-only commits from `8cf85597` to `953ea7e7`
  (`feb8403d`, `70f86bb9`, `1c6774fd`, `0f7c0db7`, `9c22b696`, `953ea7e7`);
  merge-base with master is exactly `8cf85597`; zero non-docs paths in the
  range; worktree clean before and after.
- Product drift since the native-control tip `8ebf7fae`: zero non-docs
  paths. The canonical native control therefore carries:
  `4,253 passed / 29 skipped / 0 failed`, reporter JSON `252535bf...`
  (byte-identical three ways at Tranche B closeout).
- Toolchain pins reproduced by full SHA (packet `toolchain.txt`):
  `package-lock 5322cb03...`, `.package-lock 4dd5182f...`, normalizer
  `955dca59...`, reporter `09d2bc52...`, wrapper `e7c963f1...`, Node
  v22.14.0, Vite 5.4.21, Vitest 4.1.8, anthropic SDK 0.120.2, Playwright
  1.58.0, Chrome 150.0.7871.128.
- Protected boundary: all 17 unconditional blobs plus `api.ts` (bounded to
  the single authorized union line) are identical to `8cf85597` in the
  implementation worktree (packet `protected-blobs.tsv`, 18 rows).

## 2. Re-collected base identities

| Stream | Result |
|---|---|
| backend collect (`pytest --collect-only -q`, sorted) | `4,282 / 281cad976a2df29224f41d7442f39ee6deb5b78165fb9efe3945bee6d520abe3` |
| frontend decoded list (pinned normalizer) | `99 files / 1,124 / da69a2942c03e4794e3384e6125936f9f25c1fafbad7d006b67025f8fd97bc39` |

## 3. Staged identities rebuilt from the committed plan text

The five addition blocks were extracted verbatim from the committed plan
(`awk` over the fenced lists; packet `add_t1.nodes` .. `add_t4.rows`) and
applied to the re-collected bases. Every predicted identity reproduced:

| Stage | Result |
|---|---|
| backend stage 1 | `4,289 / 37bc0a597398404de6247e465e44908ccd265798ba66722242bb8807c1614968` |
| backend stage 2 (final) | `4,303 / 52b862d7bf94f9d4605f8de1b2e92240ea152a41218446c3652b38716af77489` |
| frontend stage 3 | `1,132 / 778d64be3239dbb94df475e2cccde1b61878af3a627a28a677038191ea6a6e9d` |
| frontend stage 4 (final) | `1,134 / 941067a028c7bb6b15c3e3f64012dcf251995804e3f55c9a712cb230d4a4ba64` |
| 21-node backend addition stream | `2b540253de6578a71be09a726a11d29cce396a2e0c29421a7f8a5cfa4b3666bd`; all 21 absent from base |
| backend focused base / s1 / s2 | `61 b0d56cc5...` / `68 a76b86a3...` / `82 1c8c9de1...` |
| frontend focused base / s3 / s4 | `33 fb42f09a...` / `41 efc6accc...` / `43 853c9cef...` |
| Settings 15-file projection base / s4 | `221 a2c20d36...` / `231 e0bb6190...` |

## 4. Focused runtime baselines (one command each)

```text
backend (4-file set, existing 3 files): 61 passed in 3.82s / exit 0
frontend (3-file set):                  33 passed (3 files) / exit 0
```

## 5. Handoff

Independent Codex review of this packet authorizes Task 1 (Codex launcher
repair, `+7` RED-first). The packet root carries every raw stream, list
output, transcript, and `SHA256SUMS`.
