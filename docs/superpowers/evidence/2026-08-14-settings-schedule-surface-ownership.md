# Settings Schedule Surface Ownership Evidence

> **Status:** TASKS 0-3 COMPLETE; INDEPENDENT IMPLEMENTATION REVIEW NEXT;
> NOT MERGED
>
> **Reviewed plan tip:** `590ce355ae3f296e666a5dbcabd5b3694d7633a1`
>
> **Product grounding base:** `e2ead4378189d277a40a61f7130730513980794e`

## Task 0 - Re-grounding

The reviewed plan tip has exact merge-base `e2ead437` and differs from that
base only in the design, plan, and priority-map documents. Product and test
bytes are unchanged.

Pre-grounding authority identities:

```text
f87e4b479019857e7d4f8c87a1fdf160916396dfcbd46cb404e249000543ef3b  docs/superpowers/specs/2026-08-14-settings-schedule-surface-ownership-design.md
68616f99c39f033293b800297a2711d38de14653d683f335eec58740d6871a34  docs/superpowers/plans/2026-08-14-settings-schedule-surface-ownership.md
d99bf95f9362a93076d56892fe4fda775999bad2d85814878e6d81812ee9ead9  docs/design/PROJECT_PRIORITY_MAP.md
955dca592d243505ced622a84e88a35160a3fa787ffb954f38a6a43e1a72fcac  pinned Vitest list normalizer
```

Fresh collection and runtime results:

| Stream | Count | SHA-256 / result |
|---|---:|---|
| frontend full | 1,177 | `9530dcd91d8a7d684faa5e56f2986fbaeaa22e1d89f67818a12ed5d8ca77d1b1` |
| six-file focused | 127 | `83b8a5e4941aea52b5818d9fe9073407e69d559338c579d3f25fc930b514a735`; `127 passed` |
| Settings 15-file projection | 249 | `a3a5e481cace86991db6d8ec5da56c2d973d224e1cb1de57f631c210a646a16e` |
| backend collect-only | 4,394 | `b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb` |

The exact two-row removal and addition streams are `9c8ca36c...` and
`d35fbff9...`. Both removals occur once in the base, both additions are absent,
and set subtraction/addition independently produces frontend
`1177/90f56093...`, focused `127/0d6eb258...`, and Settings
`249/ee6618df...`. All eight retained body-evolution IDs exist; their literal
stream is `21264689...`.

All 11 owned file identities and line counts match Section 0.4. All 13
protected files match, and the independently replayed protected aggregate is
`b3b3ac3f2151d34bf3ce0e15a772ebb0a655e46fe63a43eccae142d4208602a1`.

The packet records three rejected operator attempts: a Settings projection
that included an out-of-set i18n test, a Markdown-section extraction that did
not enter the literal fence, and a backend command that referred to a missing
worktree `.venv`. None contributed to admission. The explicit admitted backend
replay, `PYTHONPATH=. pytest --collect-only -q` followed by C-locale sorting,
is byte-identical to the reviewed backend stream.

Packet: `/tmp/schedule-surface-ownership-task0-590ce355`, 53 payloads,
`SHA256SUMS` SHA-256
`0fd92def7afc208e085cb2b58d5e3031c6c32dac477f0483be02de85ec70298c`.

## Process Ruling

After Fable's independent plan review returned GREEN, the user authorized
Tasks 0-3 as one continuous execution batch. Per-task commits, packets,
RED-first evidence, exact identities, and all stop conditions remain binding.
Task 3 is the combined implementation-review gate. Merge, push, live traffic,
provider calls, and production mutation are not authorized.

## Task 1 - Partition and Latest-Job Facts

Product/tests commit: `7de617dd`.

The exact replacement collection is `101 files / 1177 /
90f56093290c70a27369296ec8d8c7de99d084a091134994ae6451bc8e45743b`.
Focused is `127 / 0d6eb25864d2c0a7a5ee76a9d05dacbc45e232106e62461a043b3d7196aac644`,
and the Settings projection is `249 /
ee6618dfc755f5ead79f6012967d3dda2c7a33f37ef87f62c5f98deff0f6c5cf`.
The node delta is exactly the reviewed `-2/+2` ledger.

RED was `2 failed / 125 passed`. The Provider integration exposed ten rows
under Data Sources instead of five, and the focused controller test exposed
that the old table ignored the required scope. No existing-module import,
fixture, timeout, or network failure contributed to RED.

GREEN proves:

- exact `write_target === "macro_calendar.db"` classification;
- disjoint and complete Macro/non-Macro partitions in insertion order;
- future Macro placement and unknown non-Macro fail-visible placement;
- one real shared controller and enabled controls in both partitions;
- Data Sources and Macro each render five current rows in both locales;
- a successful existing provider-health read replaces an older cached Macro
  job timestamp, while a rejected read preserves the admitted timestamp; and
- the disabled-provider neutral contract remains intact.

Final runtime is `127/127`; typecheck exits zero. Structural census records one
mounted `DataScheduleControlsProvider`, one schedule timer, one focus listener,
unchanged `getSchedule` and `getProvidersHealth` caller sets, zero table-level
`jobs` prop, and zero product occurrences of `sourceIds`,
`DataScheduleLocalError`, or `MACRO_SCHEDULE_SOURCE_IDS`. Protected paths remain
`13/13` at aggregate `b3b3ac3f...`.

Packet: `/tmp/schedule-surface-ownership-task1-bf6b74a0`, 40 payloads,
`SHA256SUMS` SHA-256
`966e33572f9dd6dba40caf740d6bd542bd9f092965cb94994253b4ec9cc48e6b`.

## Task 2 - Stop at Macro Title Consumer

The eight reviewed retained owners went RED exactly as planned: `8 failed /
119 passed`, with failures confined to the owner-specific title and resource
expectations. The collection remained `1177/90f56093...`.

After applying only the two locale resource changes, five owners became GREEN
and three remained RED. `MacroStorageSection.tsx` still read
`$.dataSources.schedule.title`, so the new `macroStorage.schedule.title` path
was not consumed and Macro still displayed the Data Sources-specific title.
This deterministic product wiring was absent from the Task 2 file list.

No Macro product edit has been made. Current Task 2 test/resource changes are
uncommitted and preserved with the RED and partial-GREEN transcripts in
`/tmp/schedule-surface-ownership-task2-7a4185e4`. The bounded plan amendment
authorizes one existing heading-key replacement and no other Task 2 hunk in
that file. Execution remains stopped pending focused review.

## Task 2 - Stop at Scanner Prediction

Focused review approved the heading-consumer amendment at `926c128f`. The
exact one-line `MacroStorageSection.tsx` key replacement then produced
`127/127` focused, `249/249` Settings, and one sequential `1177/1177` full
runtime. Typecheck and build exited zero. Those changes remain uncommitted.

The i18n scanner returned `37/20/0/20`, not the planned `36/20/0/20`.
Execution stopped before committing or beginning Task 3. A fresh scanner run
at product base `e2ead437` and at the current Task 2 worktree returned the same
37 candidates. Their line-number-independent
`{file, kind, literal, signature}` projections are byte-identical at SHA-256
`5dc46070ca18211027f0bf626955b79ca3409331d09e2562fda4250c62612ac3`.
No candidate was added or removed by this line.

The plan had inferred that retiring `DataScheduleLocalError` and its local
missing-source state would remove one visible-literal candidate. The scanner
shows that neither typed structure was a candidate at the base. The bounded
amendment corrects only the Task 2 and Task 3 expected scanner result to
`37/20/0/20`; scanner code, allowlist, product, tests, resources, identities,
and protected bytes remain outside this amendment.

## Task 2 - Owner-Specific Copy Complete

Focused review returned GREEN for scanner amendment `adc50d5b`. Product/tests
commit `9dc2ebf4` atomically lands the eight reviewed Task 2 paths: the exact
bilingual Data Sources and Macro schedule titles, one new Macro resource path,
the existing Macro heading's one-line consumer-key replacement, and only the
eight authorized existing owner-body updates. The heading file changes by
exactly `1+/1-`; it has no second Task 2 hunk.

The admitted final gates are focused `127/127`, Settings `249/249`, sequential
full `1177/1177`, typecheck exit zero, build exit zero, and scanner
`37/20/0/20`. Collection identities remain byte-identical to Task 1:
frontend `1177/90f56093...`, focused `127/0d6eb258...`, and Settings
`249/ee6618df...`. Task 3 mutations and final admission are now active under
the existing batch ruling; merge, push, live traffic, provider calls, and
production mutation remain unauthorized.

## Task 3 - Mutations and Final Admission Complete

M1-M7 each changed a live ownership contract, made its declared owner RED,
and restored the complete owner byte-for-byte before the next mutation. M6's
first table-local request mutation was rejected because the named unit owner
stayed GREEN. The admitted M6 mutation entered the real shared hook and made
both the unit request-count owner and a hermetic browser ledger RED with extra
`/schedule` GETs. M7 independently proved the bilingual Data Sources and Macro
headings cannot share a resource key. Every admitted mutation has
`mutated != pre` and `restored == pre` evidence.

Final collection identities are byte-identical to the reviewed targets:

- frontend: `101 files / 1177 /
  90f56093290c70a27369296ec8d8c7de99d084a091134994ae6451bc8e45743b`;
- focused six-file projection: `127 /
  0d6eb25864d2c0a7a5ee76a9d05dacbc45e232106e62461a043b3d7196aac644`;
- Settings fifteen-file projection: `249 /
  ee6618dfc755f5ead79f6012967d3dda2c7a33f37ef87f62c5f98deff0f6c5cf`;
  and
- unchanged backend collect-only identity: `4394 /
  b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb`.

Admitted runtime is focused `127/127`, Settings `249/249`, and one sequential
full command `1177/1177`. Typecheck and build exit zero; the scanner returns
the corrected `37/20/0/20`. The thirteen protected paths reproduce aggregate
`b3b3ac3f2151d34bf3ce0e15a772ebb0a655e46fe63a43eccae142d4208602a1`.
There is no Python delta. Retired ownership censuses find zero `sourceIds`,
`DataScheduleLocalError`, `MACRO_SCHEDULE_SOURCE_IDS`, optional/global scope,
or table-level `jobs` prop. Structural projection finds one mounted schedule
provider, one controller/cache loader/timer/focus listener, two consumers, and
one existing provider-health loader.

The final fixture-only browser matrix passed at `1322x777` and `390x844` in
both zh-TW and en. The five non-Macro and five Macro rows are disjoint,
complete, and ordered; provider-health latest-job truth is retained; headings
are owner-specific; mount, idle, focus, visibility, local reload, and locale
reload remain GET-only; each viewport's explicit Run click is its sole POST.
There are no external requests, errors, cell overlaps, or page-level
horizontal overflow. Existing table-local horizontal scrolling remains
available on narrow viewports. Four screenshots were inspected at original
resolution.

Six browser-harness iterations were rejected before admission (incorrect
heading level, imprecise Macro heading selection, unscoped reload button,
case-sensitive English aria label, decorated Run button name, and failure to
open the mobile directory drawer). They changed no product byte and remain in
the packet as rejected evidence. Vite, Chrome profiles, port `8474`, and all
temporary processes were removed.

Packet: `/tmp/schedule-surface-ownership-task3-f7105482`, 108 payloads,
`SHA256SUMS` SHA-256
`eba91721d540927c2e21fde43e7341e34fc3e675082e1a95cdd22fc690b5d836`;
all entries verify. Tasks 0-3 are complete. Independent implementation review
is the only next gate; Task 4, merge, push, live/provider traffic, and
production mutation remain unauthorized.
