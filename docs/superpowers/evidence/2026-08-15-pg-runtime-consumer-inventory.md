# PostgreSQL Runtime Consumer Inventory Evidence

> **Status:** TASK 0 STOPPED; BOUNDED TOOLCHAIN AMENDMENT REVIEW NEXT
>
> **Reviewed plan tip:**
> `a1413ac4244056e819c43ce5bce90969e58c460b`
>
> **Product/design grounding base:**
> `729d8514ac912b447f1892aefd3e897ea8a843b6`

## Task 0 - Stop At Unpinned Frontend Toolchain Fallback

The isolated implementation branch started at the exact reviewed plan tip.
Its merge base is the product/design grounding commit, and the range changes
only the reviewed plan and priority-map documents. The design SHA-256 remains
`e5218b58472891891acdc56fa054b07a30cc98905d71941890ad15a438bf3935`.
The first direct worktree checkout was rejected at the expected `git-crypt`
smudge boundary. The admitted checkout uses worktree-local no-op clean/smudge
filters, keeps all three encrypted paths as ciphertext, and leaves both tracked
trees clean.

The backend collect-only command completed before the stop and produced exact
identity `4,394/b0285ee3a3d124c4bbe380ad0dea022ef09fa46b52b6a14a0375c5f2459a62fb`.
The reporter records `seen=0`, `nonpassing=0`, and `exitstatus=0`. This result is
diagnostic pre-stop evidence only and must be rerun after amendment review; it
does not complete any Task 0 gate.

The worktree incorrectly linked the app-local `node_modules` directory. That
directory contains Vite cache data but no Vitest binary because this repository
hoists workspace dependencies to root `node_modules`. `npx vitest --version`
then downloaded `vitest 4.1.10` instead of using the pinned local `4.1.8`.
Sanitized npm-log inspection found 148 `http fetch GET` rows, including cache
misses. The later frontend list command failed with unresolved repository
dependencies. This frontend run is rejected, and the observed package-registry
traffic triggers hard stop condition 7.

Raw npm logs are deleted after hashing because they contain forbidden machine
paths and registry URLs. The partial packet retains only a bounded incident
summary, rejected frontend transcript, helper copies, exact pre-stop backend
artifacts, and a partial manifest. No provider, remote database, product DB,
FastAPI lifespan, scheduler, test body, tracked product byte, private config
value, or encrypted plaintext was touched. The only remote traffic was the
rejected package-manager fallback described above.

Resume requires focused amendment review GREEN. Then the root `node_modules`
toolchain link must resolve exact `vitest/4.1.8`; `npx`, `npm exec`, installs,
and package-manager fallback are forbidden; the scratch home starts empty; and
all Task 0 commands, including backend collect-only, run again before any
artifact is admitted. Task 1 remains unauthorized.

Partial packet: `/tmp/pg-runtime-inventory-task0-729d8514`, 21 payloads;
`PARTIAL_SHA256SUMS` SHA-256
`6573d0caba5f9863341728ceaefb1a5b8cc5974d8fb1b7ac8bf790fbc6e6ce35`.
