# SA Extension Reliability and Control-Clarity Evidence

> **Status: IMPLEMENTATION REVIEW-READY - INDEPENDENT REVIEW NEXT**
>
> This ledger contains redacted, reproducible engineering evidence only. It
> must not contain licensed article content, raw repair target IDs, URLs,
> credentials, browser-profile data, or production database copies.

## Task 0 - Clearance and Baseline

- Plan-review clearance:
  `7d4f11164861aa0a50cc7771fef7388577f4da0b`.
- Isolated branch/worktree: `codex/sa-extension-reliability` at
  `/mnt/md0/PycharmProjects/ArkScope-sa-extension-reliability`.
- Linked-worktree materialization: known `git-crypt` smudge boundary handled
  with `--no-checkout`, the existing 148-byte key copied to linked Git metadata
  with mode `0600`, and `git read-tree -mu HEAD`. Protected document size:
  `47,608` bytes. Existing root `node_modules` linked without install.
- Backend collection: `4621`; sorted raw pytest node-ID SHA-256:
  `488eeaab65ffad32bd098dbc4b1df0eb3ed3b62feabfe3a62b1a76324d960a17`.
- Backend focused collection/run: `9 files / 238`; `238 passed`; SHA-256:
  `ca36c2cc8616982fa8dd2c2f386743751691de6bd4f9bf52134229d830740de8`.
- Frontend collection: `95 files / 1056`; sorted relative node-list SHA-256:
  `5f9a1624b31a47dc9b786f57fa5de77eca86dde269c68ada3787d7210b05fd13`.
- Frontend focused collection/run: `4 files / 62`; `62 passed`; SHA-256:
  `025e871755c356f0be89089e92d0241d06b335af52ae8a2ca0f66e06b187f643`.
- All fixed-base protected-boundary gates against `c49a2417` pass before
  product edits. `git diff --check` passes and the worktree was clean.

## Review Boundary

- Product base: `c49a2417`.
- Packaging checkpoint:
  `a2869f2ce7a2c2e603657593ef9534a438cd02a6`.
- Review-ready product tip:
  `c85f4bedb2597765357f3c64e59a632f19ce104a`.
- This branch is not merged and was not installed into either production
  browser. No historical repair or production telemetry write was executed.

## Task 1 - Atomic Firefox Packaging

- RED proof: `11 failed / 3 passed`. All ten new packaging nodes and the
  renamed installer owner failed before the builder/delegation existed; the
  three host-path parameter cases remained green.
- GREEN proof: `14 passed` in the packaging/install-path suite.
- `PACKAGING_GATE_TIP`:
  `a2869f2ce7a2c2e603657593ef9534a438cd02a6`.
- Backend checkpoint collection: `4631`; sorted raw node-list SHA-256:
  `6a005572814a1c539e96b521b0a9cdb2984d47e8ec625f0dc6a696d6da3a635b`.
- Focused checkpoint collection: `248`; sorted raw node-list SHA-256:
  `46fcc1766e2e8301e75b7ce5cdb7089d5d98e81955a916f1f84c0e9c72f2a300`.
- A fresh real Firefox build contains exactly `12` files, including
  `article_identity.js`. Two separate output directories produced identical
  file lists and per-file hashes. Normalized artifact hash-list SHA-256:
  `089a379302cdd6cd7ada109675caae295b38450d88720d7010bab797cd5c8866`.
- Adversarial fixture coverage rejects missing dependencies, dynamic
  `importScripts`, variable/concatenated/computed/spread `executeScript`
  dependencies, remote/traversing/query references, and final-swap failure.
  Failure preserves the previous known-good directory; successful replacement
  drops stale files.

## Task 2 - Structured Run Protocol

- RED proof: the combined protocol/background suite was exactly
  `15 failed / 5 passed`. The twelve pure protocol nodes and three real
  background-adapter nodes failed for absent modules/functions; the five
  existing Alpha Picks nodes remained green.
- Shared fixture: `15` protocol cases plus opaque background-adapter cases.
  The incident-shape case contains exactly `18` retryable details. Python and
  classic-script JS return byte-equivalent JSON values or the same stable
  validation code for every case.
- GREEN proof: protocol `12/12`, Alpha `8/8`, packaging `10/10`. An in-place
  hardening case subsequently exposed unknown legacy Market status as false
  complete (`1 failed`); it now fails closed to `protocol_invalid` with no
  node-accounting change.
- Task 2 product tip:
  `46bce5886b7d466bcc0d0cd3f21d522a4ca41619`.
- Backend checkpoint collection: `4646`; sorted raw node-list SHA-256:
  `8d0a838196e2cd5552844963ce270985bcb169c3ad096df375d48ead7a78c8f1`.
- Focused checkpoint collection/run: `263`; `263 passed`; sorted raw
  node-list SHA-256:
  `00b1d9ed399b898f1821dac7f1039cc6b41359ae578864acb94bbb726f651366`.
- Canonical results derive database status and healthy-anchor eligibility from
  closed phases/items. Raw legacy prose is not admitted to canonical items;
  unknown browser failures become `unknown_failure`; only explicit
  `404`/`410`/removed evidence can be source-unavailable.
- Firefox and Chrome both acquire `extension_run_protocol.js` through the
  dependency graph. Two fresh Firefox builds were byte-identical and each
  contained exactly `13` files. Normalized artifact hash-list SHA-256:
  `f34e12aa153b054735ca133fcb918d682e4d52b9cd6a5481d39aecd5cd719522`.

## Task 3 - Durable Telemetry

- Task 3 product tip:
  `df49d5b5369193c4bd94ef660cd90df0892d956e`.
- The outbox/native/store/protocol suites are `8/8`, `14/14`, `63/63`, and
  `12/12`; the Task 3 core is `105/105` and the canonical focused set is
  `284/284`.
- Backend checkpoint collection is `4667`; full/focused sorted node-list
  SHA-256 values are
  `51548195153e1f9e12a24fa5475d9de33b4afba9e813ae3a5ce5abd1a66ef085`
  and
  `2c3b4e302fff4c190ebbc3bb5a05db2b4ece2186a5c2d0430b41bab25217a717`.
- Two independent Firefox builds contain the same `14` runtime files;
  normalized artifact hash-list SHA-256 is
  `951dc32c892a1d35f64fcb0b0d49eb31752536420f913aa8522a4850444dc7b5`.

## Task 4 - Structured Durable Health

- Task 4 product tip:
  `1e52ed4e391d42ff9421b42308ba14a6fad15ed6`.
- Backend collection is `4677`; the 12-file focused set is `294/294`.
  Full/focused sorted node-list SHA-256 values are
  `ec699e992433f5c2cabe612e5f609fad1ca64ae88915034bb8e56aa2bfcd7de9`
  and
  `d35c5155ed480f3495567ce172bc00b391b6e69d4768526d288d2b738a679a47`.
- Frontend is `95 files / 1063 passed`; the four-file focused set is `69/69`.
  Full/focused sorted relative node-list SHA-256 values are
  `a93c02bc28d1924f23f7895338d723e968dcb389a494ff0e0f993e4c092019d4`
  and
  `5c3859c3c7db7fe90c13f3c46d49610eae64986ef0b89e558246ee0cf13c6cdf`.
- Resources are `694/1794` per locale; scanner remains `36/20/0/20`.

## Task 5 - Durable Market News Repair

- Task 5 product tip:
  `8b20e608765d2ef134a6249a2ebf0bccd400361f`.
- Sixteen domain nodes cover canonical manifests, exact path admission,
  recorded/incident previews, atomic start, resumable progress, reconciliation,
  cancellation, and terminal hash idempotence. Five DAL nodes cover exact-ID
  no-age reads, body readback, inclusive intervals, privacy projection, and
  unavailable-DB fail-closed behavior.
- Recovery is `16/16`; `test_sa_tools.py` is `102/102`; the four-file Task 5
  execution set is `195/195`.
- Backend collection is `4698`; the 13-file checkpoint collection is `315`.
  Full/focused sorted node-list SHA-256 values are
  `d7c35ad0fae96f6f0e4fd0211fdc9a1bdd8e51eeb63d616587f435dabff2f284`
  and
  `3f1b0a608ab2037c403e57eddb53c31dca47e10ef47977d2c10b6e1594375b48`.
- Generic jobs history/status omit frozen target descriptors and expose only
  kind, counts, lifecycle state, and bounded manifest-hash prefix. Fixed
  recovery routes retain the full machine contract.

## Task 6 - Honest Popup and Recovery Runtime

- Task 6 product tip:
  `679b7a65a97fc48f8b8d1f9551ad95c597b424db`.
- The mounted/runtime checkpoint is `39/39`. The popup has exactly five
  permanent controls grouped `3 + 2`: `Quick Update`, `Full Article Scan`,
  `Deep Repair Scan`, `Sync Latest News`, and `Catch Up News (24h)`.
- Every permanent control has one `aria-describedby` prose owner. The same
  description appears on hover/focus and in the five-row disclosure table;
  there is no `title`, `window.confirm`, or separate help page.
- The table exposes the actual Alpha list/enrichment/comment bounds, configured
  Full/Deep pending-row limits (`10` / `50`), Market News `3/18` routine
  limits, and the 24-hour catch-up boundary. It states a non-guarantee for
  every action.
- Recorded-ID retry is contextual. Incident recovery is normally under
  Advanced, promotes itself for an actual gap beyond 24 hours, repeats the
  exact interval and discovery bound before confirmation, and resumes by
  frozen run ID plus manifest hash after popup close/reopen.
- A fresh Firefox artifact contains exactly `15` files. Two builds had the
  same file list and byte hashes; normalized artifact-manifest SHA-256:
  `fbf549aea6a74126caffd57be49861d1eafb999d212f3f0050702db794fe6ce5`.
- Shared protocol corpus SHA-256:
  `92e3fd012dcf67c708951f18d708f8930e25988785a516987fd0f43eb8d37635`.
  JS and Python fixture replay remains byte-equivalent.

## Task 7 - Canonical Accounting and Copied-DB Proof

### Exact test ledger

- Backend: `4621 -> 4710`, exact `+91/-2`; focused `14 files / 327`.
  Final full/focused normalized node-list SHA-256 values:
  `50a65708b78b172c7267dd72bde05e375026fcf930f47660eec0d9f1a675f68a`
  and
  `e5a1902f9dd759693930a6016613496632b62a93fdf34c1192cc661808dad942`.
- Backend removals are exactly:
  `test_firefox_installer_copies_every_popup_script_dependency` and
  `test_record_extension_job_rejects_invalid_status`.
- Frontend: `95 files / 1056 -> 95 files / 1063`, exact `+8/-1`;
  focused `4 files / 69`. Final full/focused normalized SHA-256 values:
  `a93c02bc28d1924f23f7895338d723e968dcb389a494ff0e0f993e4c092019d4`
  and
  `5c3859c3c7db7fe90c13f3c46d49610eae64986ef0b89e558246ee0cf13c6cdf`.
- The frontend removal is exactly
  `displaySAExtensionSegments > maps every known extension segment in both locales and preserves unknown ids`.
- Virgin full backend A/B is `4510 -> 4599` passed with both sides retaining
  `30 failed / 74 skipped / 18 warnings / 7 errors`; the normalized
  failure/error ID sets are identical. Frontend is `1063/1063`; typecheck,
  build, and the i18n scanner (`36/20/0/20`, `src/**`) pass.
- A fresh final-tip archive, with the same repository-root `node_modules`
  mounted as the canonical A/B, independently reproduced the head summary:
  `30 failed / 4599 passed / 74 skipped / 18 warnings / 7 errors`. Running an
  archive without that declared dependency predictably adds jsdom harness
  failures and is not canonical evidence.
- The direct PG-unreachable smoke exits `0` with `ok: true` and
  `pg_attempts: []` across all 24 checks.

### Protocol, outbox, and consumer audit

- The telemetry outbox is bounded at `100` records, seven days,
  `131072` bytes per record, and `4194304` total bytes. Count, age,
  per-record, total-byte, conflict, and storage-failure loss remain visible;
  duplicate flush reuses the same client event ID.
- One native/store transaction owns event dedupe and completed-row creation.
  A complete local capture may be audit-pending, but cannot become a durable
  healthy anchor until persistence succeeds.
- `/jobs/status` and `/jobs/history` deliberately expose a degraded extension
  run as database `failed`. Market News health consumes the structured result
  and advances its anchor only for `healthy_anchor_eligible=true`. Scheduler
  backoff state has no extension-push coupling, so the honest status change
  does not spend scheduler retries.

### Packaging and protected boundaries

- Two empty-directory builds were byte-identical. Removing
  `article_identity.js` in a disposable source tree made the builder exit `2`
  and preserved the existing output byte-for-byte.
- All fixed-base protected gates pass. The only authorized source-side hunks
  are the conditional removed marker, reviewed Market News recovery reads,
  exact structured health rendering, and the reviewed extension/runtime
  owners.
- The formatter inventory resolves to the following exact 50 paths. Columns
  are `path`, base SHA-256, and tip SHA-256. Forty-eight rows are identical;
  the two differing files are owned health product/test files. The
  `DataSourcesSection` `shortDate`/`formatCount` block itself is byte-identical
  (`23f155a9e42e3bfdd5ed6ab3f1e78c0e2c9c1154227fc9a371d5d00052b4e6df`).

```text
apps/arkscope-web/src/timeDisplay.ts 8a372dd36b996ca5c8e0af73301afcd07b12189a66a57b427c1ab3212d07f091 8a372dd36b996ca5c8e0af73301afcd07b12189a66a57b427c1ab3212d07f091
apps/arkscope-web/src/ui/BoundedProgress.tsx 3266a1aa047f0641763650dd26641e7fce025f9b2b45924008ccd711a38a0eb9 3266a1aa047f0641763650dd26641e7fce025f9b2b45924008ccd711a38a0eb9
apps/arkscope-web/src/SourceRunProgress.tsx 67511b135268d28122fcf4d78c231c2f6a05c7307985306c8d8ec55ef8967752 67511b135268d28122fcf4d78c231c2f6a05c7307985306c8d8ec55ef8967752
apps/arkscope-web/src/App.tsx 13b6bf02f75df4f02ccdcad648b1b4338f318b3d9b7e920c2e4ef3699a23f572 13b6bf02f75df4f02ccdcad648b1b4338f318b3d9b7e920c2e4ef3699a23f572
apps/arkscope-web/src/Dashboard.tsx a8583cdd20acd38481e3b315bc3c79f13faf1a7c8e4e95a767896955860d4111 a8583cdd20acd38481e3b315bc3c79f13faf1a7c8e4e95a767896955860d4111
apps/arkscope-web/src/ResearchHistoryDrawer.tsx c492c5f0471996f65c88034e17005336cdaca4a70a539cc4632da8b27f5dea4e c492c5f0471996f65c88034e17005336cdaca4a70a539cc4632da8b27f5dea4e
apps/arkscope-web/src/Research.tsx fe172e26a260c3e49245e8ae19d092862af5765d585dfb2dc5ad3742369ff06b fe172e26a260c3e49245e8ae19d092862af5765d585dfb2dc5ad3742369ff06b
apps/arkscope-web/src/ResearchEvidenceDrawer.tsx b4a20ec23817594295183dd0782586ce1455762663383ec915ac2177f5d722cc b4a20ec23817594295183dd0782586ce1455762663383ec915ac2177f5d722cc
apps/arkscope-web/src/Holdings.tsx 08388f3b4c5ec6c9015c6cf289326e487dc73bc912174e24c6e6bc359a8fefb5 08388f3b4c5ec6c9015c6cf289326e487dc73bc912174e24c6e6bc359a8fefb5
apps/arkscope-web/src/PortfolioActivity.tsx 1bc19631ef4545411cf3bcc00a9f39d71cfb8fc195a748d7ff21f98571ca0ac9 1bc19631ef4545411cf3bcc00a9f39d71cfb8fc195a748d7ff21f98571ca0ac9
apps/arkscope-web/src/PortfolioCapturePanel.tsx 963dd55e6bc5fb0488085122ee60d9fc2a3a36049774cc703975c10b485f0cd4 963dd55e6bc5fb0488085122ee60d9fc2a3a36049774cc703975c10b485f0cd4
apps/arkscope-web/src/PortfolioAccountOverview.tsx 9b2130586fa15384f700b0274fdc2178e5b1c4a5dec9f3b6a657655be5d65bf9 9b2130586fa15384f700b0274fdc2178e5b1c4a5dec9f3b6a657655be5d65bf9
apps/arkscope-web/src/PortfolioRecentActivity.tsx f042962d3baeb2c411b78222ed33d4253f41fda77bef940dd3ea4299b2a748ad f042962d3baeb2c411b78222ed33d4253f41fda77bef940dd3ea4299b2a748ad
apps/arkscope-web/src/Home.tsx 896af651c5857400dc31977f8d8d19f57311894f1e345b621638e7b22d9d24f7 896af651c5857400dc31977f8d8d19f57311894f1e345b621638e7b22d9d24f7
apps/arkscope-web/src/Watchlist.tsx 4d2834421a49f55f5c95e9f17ef905e4fb73cb228664d3bf5daf58b82026ede4 4d2834421a49f55f5c95e9f17ef905e4fb73cb228664d3bf5daf58b82026ede4
apps/arkscope-web/src/Universe.tsx 871905efa921b724ac1fdbf099c2f8af6c65a78a926557b7826857e10fd710ca 871905efa921b724ac1fdbf099c2f8af6c65a78a926557b7826857e10fd710ca
apps/arkscope-web/src/TickerDetail.tsx 1390e318e926111df8ec163a77c2528f9af3c19da0338a8993c1042ff7bc16c0 1390e318e926111df8ec163a77c2528f9af3c19da0338a8993c1042ff7bc16c0
apps/arkscope-web/src/News.tsx 3946721d75f7751c813bb63884472152cf07db9b599d5a22fc0d4fa21048b202 3946721d75f7751c813bb63884472152cf07db9b599d5a22fc0d4fa21048b202
apps/arkscope-web/src/settings/DataSourcesSection.tsx ab18573236e9c6e1bffc75395712b2ff8d00d091dcb61fed8f55a50514102363 b641673221b7a5ccc09f8cb1df5082d83c533b55298b7c94bc65f4cd11b0e306
apps/arkscope-web/src/settings/DataStorageSection.tsx f84961300e183d2bb3d618f8208b4b424d4c61406fc9524a0826e24dbe6f7d54 f84961300e183d2bb3d618f8208b4b424d4c61406fc9524a0826e24dbe6f7d54
apps/arkscope-web/src/settings/MacroStorageSection.tsx 86f15683b0a6d2db7bdeac1f3d506b5cc59627add57878866c74986a56f7f49b 86f15683b0a6d2db7bdeac1f3d506b5cc59627add57878866c74986a56f7f49b
apps/arkscope-web/src/settings/NewsStorageSection.tsx 84bc6af7132f8934f0a3315f9f6dc2279f25cb54e4bca4ef919cfb850345a87e 84bc6af7132f8934f0a3315f9f6dc2279f25cb54e4bca4ef919cfb850345a87e
apps/arkscope-web/src/settings/ModelRoutingSection.tsx 8bdea37a01c06a9e5754cda8fa20b5cec2882b083f9896b2ef24877d501d99c1 8bdea37a01c06a9e5754cda8fa20b5cec2882b083f9896b2ef24877d501d99c1
apps/arkscope-web/src/settings/ProviderSection.tsx 88d46727c8836d73e97d410f417e5aaf9b39e3bf90cb8d0587062f512fdb0391 88d46727c8836d73e97d410f417e5aaf9b39e3bf90cb8d0587062f512fdb0391
apps/arkscope-web/src/credentialDisplay.ts f5a9f84b50697b21ad59b2fb24e70db2ebe9f16e34b1e40fa61b118801042645 f5a9f84b50697b21ad59b2fb24e70db2ebe9f16e34b1e40fa61b118801042645
apps/arkscope-web/src/timeDisplay.test.ts 4bad526a5d5385d3241d4b51dc80c3eef90a78bc816a6e818c50cc5d4b3f62ab 4bad526a5d5385d3241d4b51dc80c3eef90a78bc816a6e818c50cc5d4b3f62ab
apps/arkscope-web/src/ui/BoundedProgress.test.tsx b44edda08ae4a6b82f5d1550dab9803c4cc26de5adc335e256aa1a7bcf9031a1 b44edda08ae4a6b82f5d1550dab9803c4cc26de5adc335e256aa1a7bcf9031a1
apps/arkscope-web/src/SourceRunProgress.test.tsx 5a8fae23d3ca1105485c19943a539ba9b15696cba48946f0d4d4fd09423e7e02 5a8fae23d3ca1105485c19943a539ba9b15696cba48946f0d4d4fd09423e7e02
apps/arkscope-web/src/AppShell.test.tsx e27f1ee2b365fb8d8e7fc6db8cbfb101208622ed917bc928130a2e0537fd83ce e27f1ee2b365fb8d8e7fc6db8cbfb101208622ed917bc928130a2e0537fd83ce
apps/arkscope-web/src/ResearchHistoryDrawer.test.tsx 4409b865f556958d1c703620adfaf9288247e1fb079de6f2b067968c983fdc58 4409b865f556958d1c703620adfaf9288247e1fb079de6f2b067968c983fdc58
apps/arkscope-web/src/ResearchWorkspace.test.tsx bc3e26f6290775952d2c1ea45ea70ffc635312c61184f55181e5e5895143972d bc3e26f6290775952d2c1ea45ea70ffc635312c61184f55181e5e5895143972d
apps/arkscope-web/src/ResearchEvidenceDrawer.test.tsx 889041f104711904b3926f5a691dcc5f156643631a755d2304aaab2caad49012 889041f104711904b3926f5a691dcc5f156643631a755d2304aaab2caad49012
apps/arkscope-web/src/ResearchRunProgress.test.tsx 78eb1a911ebf59d0ca1b9052f894a58d4dbb1d8d10ee166f590c4a1006f64c35 78eb1a911ebf59d0ca1b9052f894a58d4dbb1d8d10ee166f590c4a1006f64c35
apps/arkscope-web/src/Holdings.test.tsx 2823c90c23ac0845db3c55f430bd7369c33f0617982f2f7ed4cdd5b62933bf60 2823c90c23ac0845db3c55f430bd7369c33f0617982f2f7ed4cdd5b62933bf60
apps/arkscope-web/src/PortfolioActivity.test.tsx 85f9e7aca2e83677cab61721d3e56c1ab9a1740c181a418d39634c73d8da47fa 85f9e7aca2e83677cab61721d3e56c1ab9a1740c181a418d39634c73d8da47fa
apps/arkscope-web/src/PortfolioCapturePanel.test.tsx 2a8fbcd0f0b138fbf83af5f9c7ad419275944eeb10fcb6a9810cc65d46718eb2 2a8fbcd0f0b138fbf83af5f9c7ad419275944eeb10fcb6a9810cc65d46718eb2
apps/arkscope-web/src/PortfolioAccountOverview.test.tsx 20946c5b48ff4c4c7e66b9c361f38034f6c24e79d2242a52ba5d89c831344782 20946c5b48ff4c4c7e66b9c361f38034f6c24e79d2242a52ba5d89c831344782
apps/arkscope-web/src/PortfolioRecentActivity.test.tsx df7d28b8d8b3178b375ea2771e882dfd80d390427ea74c448c0349c118a5ed44 df7d28b8d8b3178b375ea2771e882dfd80d390427ea74c448c0349c118a5ed44
apps/arkscope-web/src/Home.test.tsx 5b9a8ce4ec48a9fd7167f5946719b6f8e957526639e5af8d87e72f4629302d83 5b9a8ce4ec48a9fd7167f5946719b6f8e957526639e5af8d87e72f4629302d83
apps/arkscope-web/src/Watchlist.test.tsx b11f2cc6761925e1f11b87aa50adb0714a77c63ee902bd6b4add0e84745f7c8e b11f2cc6761925e1f11b87aa50adb0714a77c63ee902bd6b4add0e84745f7c8e
apps/arkscope-web/src/Universe.test.tsx 8b35c45f16f08565626bda027e2097fd164bf44c643b7165a1797e103c6793df 8b35c45f16f08565626bda027e2097fd164bf44c643b7165a1797e103c6793df
apps/arkscope-web/src/TickerDetail.test.tsx 1b438a0ce4cc64f60e404fefa77e8b43b8bf253df5b7d4bc2f3c6b5e1e9e7454 1b438a0ce4cc64f60e404fefa77e8b43b8bf253df5b7d4bc2f3c6b5e1e9e7454
apps/arkscope-web/src/News.test.tsx 8aacedf3a34c4a7b214f2cb758ae2d4e0b49991cb88dc20f843648d7242753f4 8aacedf3a34c4a7b214f2cb758ae2d4e0b49991cb88dc20f843648d7242753f4
apps/arkscope-web/src/SettingsProviderConfig.test.ts 5f489d141e1cf92286692b48368fc40663d3f4398eb069025779b294a666c82c c05f58e1429ade8c0379fcb843debc32d538c7cc19640e5c1e6026858f40539b
apps/arkscope-web/src/SettingsPostPgExitStorage.test.ts 40bf8c3a25fc2fa54df05825b2fd5cade5f5679e4503c7b27f70fd1e2c9b90e6 40bf8c3a25fc2fa54df05825b2fd5cade5f5679e4503c7b27f70fd1e2c9b90e6
apps/arkscope-web/src/settings/MacroStorageSection.test.tsx 4b86fda29923122d650d738f23b435440f794be4dbd29b2e441e4df8a1d93a83 4b86fda29923122d650d738f23b435440f794be4dbd29b2e441e4df8a1d93a83
apps/arkscope-web/src/SettingsNewsStorage.test.ts 18c2bf75ab1f203f473851c41a5b9a37fc0f536db5c58dbc504eceb4670f2bf3 18c2bf75ab1f203f473851c41a5b9a37fc0f536db5c58dbc504eceb4670f2bf3
apps/arkscope-web/src/SettingsModelRouting.test.ts 02c2fd6a17ce2fe6092bb235d3916f1e983014aa4c4cf9d3dd2462095ee87378 02c2fd6a17ce2fe6092bb235d3916f1e983014aa4c4cf9d3dd2462095ee87378
apps/arkscope-web/src/ProviderSection.test.ts bc43ba462107149a236860c2fac505dbd0a8c076ce3d0523f10629434e591a0f bc43ba462107149a236860c2fac505dbd0a8c076ce3d0523f10629434e591a0f
apps/arkscope-web/src/credentialDisplay.test.ts 18ef4d9b5d338b97a1d35cfcb0fa5c7dfde6b76c2c5bfbd22d153b1df2090fa7 18ef4d9b5d338b97a1d35cfcb0fa5c7dfde6b76c2c5bfbd22d153b1df2090fa7
```

### Copied-DB proofs and reviewed deviation

- Four online backups were created as two `0600` profile/capture pairs.
  Immediate pre/post source size, mtime, and integrity checks were unchanged.
  The running production app legitimately changed the profile DB later during
  the long browser matrix, so no long-duration byte-equality claim is made.
- Pair A read-only previews were repeatable and byte-identical. The default
  recorded preview had no executable work; incident discovery reported `1093`
  known targets, unknown missing-metadata count, a trailing 168-hour bound,
  and no verified pre-protocol anchor.
- Pair B proved atomic duplicate start, checkpoint idempotence,
  interruption/resume, body-readback reclassification, retryable prevention
  of complete, terminal hash stability, second-finalize idempotence, SQLite
  integrity `ok`, and FK count `0`. Synthetic result SHA-256:
  `be8223ca5e5168045870cc26582668354c382323f5760114853bf698f0a0866d`.
- The fresh historical preview found `25` source runs, `450` failure records,
  `118` unique targets, `88` bodies already present, and `30` still missing.
  The missing-body published interval is
  `2026-07-18T21:29:00+00:00` through
  `2026-07-19T16:18:00+00:00`. Manifest SHA-256:
  `efe8848ea2d257108023ce9c083ca5229cc7ba245f5724d73ed910753a372159`;
  salted redacted target-set SHA-256:
  `dcb883262a4634c22ad43b5c77e583f6a5a23dc6aef40966f637d5b8c96f4c60`.
  No raw target ID, title, body, or full URL is committed.
- That historical gate exposed a real pagination defect: explicit source-run
  repair looked only at the latest 200 rows, so later `not_due` rows could hide
  the requested incident runs. An in-place adversarial test inserted 201 later
  rows and failed `source_run_not_found`; product tip
  `c85f4bedb2597765357f3c64e59a632f19ce104a` adds chunked exact-ID reads.
  The repaired preview reproduced the counts/hashes above. This reviewed
  deviation changes no node ID or accounting.

## Task 8 - Isolated Runtime and Accessibility

### Browser/native lifecycle

- Chrome-for-Testing loaded the real source extension and a temporary native
  manifest pointed at the copied-DB sidecar. A complete quick run persisted as
  `succeeded`; an injected one-detail failure persisted as `failed` with one
  retryable item and did not advance the healthy anchor. With the sidecar
  down, the capture remained usable and one audit event stayed pending. On
  restore/popup reopen, that same event persisted exactly once, the queue
  emptied, and a second reopen added zero rows.
- Chrome also ran the real popup with a synthetic service-worker native
  boundary: five controls, `3 + 2` groups, five disclosure rows,
  `Retry Recorded Failures (1)`, `Resume Active Repair`, promoted 30-hour
  Advanced state, zero CJK/title attributes, zero document overflow, and zero
  clipped text.
- Firefox loaded the freshly built exact XPI. The Snap native-messaging portal
  was explicitly disabled for the gate; otherwise it resolves the global user
  host rather than the isolated test host. A synthetic native host under a
  disposable Snap home returned only fixture data. Firefox verified the same
  five controls/groups/table, one focus-visible description owner per button,
  Retry/resume, 30-hour Advanced preview, confirmation focus, Escape focus
  return, keyboard tab order, zero CJK/title attributes, and zero document or
  element-level clipping. Real sidecar/native integration authority therefore
  comes from Chrome; Firefox authority is exact-artifact compatibility plus
  isolated UI/native-protocol behavior.

### Recovery and localized health matrices

- The synthetic recovery/protocol/outbox execution set is fresh `48/48`.
  It covers exact-ID no-age retry, a 30-hour interval, the 168-hour cap,
  unknown metadata before discovery, zero-known-ID discovery, reached-start
  and unresolved evidence, close/reopen resume, shared mutex, four-state
  reason validation, and every outbox bound.
- The real Settings composition rendered six extension states (`complete`,
  `degraded`, unknown code, active repair, retryable repair, telemetry unseen)
  at `390/760/960/1440` in `zh-Hant` and `en`: `48/48` normal-mode cases.
  Every case had zero raw planted detail, zero document overflow, and zero
  health-panel leaf-text clipping.
- Two additional Developer Mode cases exposed only the validated stable code
  after opening the diagnostics disclosure. Unsafe path-shaped code and both
  planted raw details were absent from the DOM.

### Cleanup and remaining authorization

- Isolated sidecar/Vite ports `8467/8477` refuse connections; production Vite
  `8430` remained running. Browser profiles, temporary native manifests,
  Snap test home, XPI, screenshots, private preview files, and all copied
  licensed databases were deleted. No `arkscope-sa-ext-*` or copied-DB A/B
  path remains in `/tmp`.
- This evidence authorizes independent implementation review only. Merge,
  production browser installation, production repair execution, and any
  historical-manifest approval remain separate gates.
