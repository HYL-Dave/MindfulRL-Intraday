# Retired local assets and legacy score research export

> **Status:** EXECUTED 2026-08-10
> **Product base:** `8cf85597d866c6d9cd0b75c75a24f86d73ca65a1`
> **Tracked prevention commit:** `b4b12f84b99d895e6aab433fb62255b910d9c77c`
> **Authority:** explicit user approval to delete the retired local assets,
> preserve the frozen score dataset in a private research database, validate
> it, and remove the score table from the product database.

## 1. Regeneration prevention

The Finnhub and Polygon collector CLIs retained cwd-relative `FileHandler`
instances after their root logs had otherwise become ownerless. Commit
`b4b12f84` removes those two file handlers while preserving console logging,
provider collection, and scheduler behavior. The existing import-safety owner
now also asserts that CLI setup contains exactly one `logging.StreamHandler`
and no `logging.FileHandler`.

Verification:

- `tests/test_collector_adapters.py`: `12 passed`;
- a real temporary-cwd invocation of both `_setup_cli_logging()` functions
  created zero files;
- `src/collectors/` contains no `collect_finnhub_news.log`,
  `collect_polygon_news.log`, or `logging.FileHandler` reference.

This tracked change remains on the isolated
`codex/retired-assets-research-cleanup` branch until the active OAuth repair
line closes. That preserves the OAuth line's reviewed product base. It must be
rebased and merged before the two CLI changes are considered live on `master`.

## 2. Exact local deletion

The following ignored local paths were removed by exact path, without archive
or compatibility copies:

```text
scripts/
training/
api_keys_tier1.txt
api_keys_tier5.txt
config/scoring_keys.txt
collect_finnhub_news.log
collect_ibkr_news.log
collect_polygon_news.log
finrl_news_pipeline.log
```

The two directories contained only stale bytecode and old training logs; no
tracked path existed below either root. No listed path was a symlink or held by
a running process. Secret contents and digests were not read. Local deletion
does not itself revoke a key at its provider; provider-side revocation remains
independent if any retired value is still valid.

All nine paths remained absent after the desktop app and Python sidecar were
restarted.

## 3. Private research database

The frozen score dataset and its associated normalized news records were
exported outside the repository:

```text
/mnt/md0/ArkScope-private-research/legacy-news-scores-2026-06-07.sqlite3
```

Identity and boundary:

- parent directory mode: `0700`;
- database mode: `0600`;
- database size: `609,476,608` bytes;
- database SHA-256:
  `284f45ca7ee9ea939fe456ab2c0bcd197690fad7e2e8c60207d866ecaefaccd5`;
- external manifest:
  `legacy-news-scores-2026-06-07.manifest.json`, mode `0600`, SHA-256
  `71e7c564d2b008ddc7c65d7d69d4bf76ee29ef3831bafa12942cf97f6d58067e`;
- `PRAGMA integrity_check = ok` and `PRAGMA foreign_key_check` returned zero
  rows.

The export used one SQLite read snapshot. Each source projection and destination
table was streamed in primary-key order through the same canonical JSON row
encoder. Counts and SHA-256 digests matched exactly:

| Table | Rows | Row-stream SHA-256 |
|---|---:|---|
| `news_articles` | 140,152 | `cbe6a58ba3f12a59380d69704e1459e7cb966a7e61cf7d5419a60bafd79c2858` |
| `news_article_bodies` | 140,152 | `2da725e0734530aa22b2b08c2396532e12010b64730c59d770c2a10b6cac45d5` |
| `news_article_body_variants` | 455 | `337ea441a950be0ee52963dae4d0bfe0a63358bbe121041f7bc69a662d173eaf` |
| `news_article_titles` | 140,155 | `902f56c2b42845072f87db443498984f5b1ffbb94c1b23f1402b549ebbc407ee` |
| `news_article_keys` | 280,347 | `b1819bb74b5703bb92c47970c58b53b6a1855d1a80e4c4ba8d070a19278fdb88` |
| `news_article_tickers` | 372,708 | `ccc66b7f2978f6e3ef8bf5995f430793d039ad8530e75daf58bb13ee67d7455d` |
| `news_article_scores` | 491,808 | `e5c498740ef6fedf7f9aa0052f06e9792f9e3adc7678af9a482cf1d5d13e5a71` |

All 491,808 scores belong to 140,152 articles and every score had a normalized
parent. The score window is the frozen 2026-06-07 batch. The database embeds a
`PRIVATE RESEARCH ONLY` redistribution warning. Polygon, IBKR, and Finnhub
content has not been cleared for redistribution; any open-source release needs
a new provider-license and dataset-content review.

## 4. Product-table removal

The desktop process tree was stopped before mutation. The private research DB
identity, score count, and integrity were rechecked, then
`data/market_data.db` was opened with `BEGIN IMMEDIATE` and the exact
`news_article_scores` table was dropped. The transaction asserted the expected
table plus four index objects, unchanged non-score schema, unchanged retained
row counts, and zero foreign-key violations before commit.

`VACUUM` then physically rewrote the product database so retired rows did not
remain merely as freelist pages. Results at the quiescent admission boundary:

- size: `3,477,532,672 -> 3,195,076,608` bytes;
- reclaimed: `282,456,064` bytes;
- inode and mode remained `127284871` / `0644`;
- `news_article_scores` table: absent;
- schema references to `news_article_scores`: zero;
- freelist pages: zero;
- `PRAGMA integrity_check = ok`;
- `PRAGMA foreign_key_check`: zero rows.

Retained row counts inside the transaction were unchanged:

| Table | Rows |
|---|---:|
| `news` | 453,379 |
| `news_articles` | 334,055 |
| `news_article_bodies` | 334,055 |
| `news_article_body_variants` | 735 |
| `news_article_titles` | 334,082 |
| `news_article_keys` | 786,463 |
| `news_article_tickers` | 703,954 |
| `prices` | 2,425,998 |
| `financial_cache` | 43 |
| `market_sync_meta` | 2 |

Focused retirement/schema tests returned `30 passed`. The desktop, Vite,
Electron, and Python sidecar restarted successfully; `/healthz` returned HTTP
200 with `status=ok`. Live news ingestion then resumed and added new normalized
articles, while the retired score table remained absent.

## 5. Explicit boundary

- The research database is not an ArkScope runtime input and is outside Git.
- Existing `data/market_data.db.bak-*` historical backup files were not touched
  by this exact authorization. They may contain older score-table copies and
  need their own deletion decision if physical eradication across backups is
  desired.
- No provider request, score recomputation, credential read, push, or permanent
  public archive occurred.
