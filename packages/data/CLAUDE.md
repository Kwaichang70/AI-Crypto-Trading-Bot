IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Data Package
Data acquisition, caching, and feature engineering.

### Components
- **MarketDataService** — OHLCV candle fetching and caching
  - Timeframes: 1m, 5m, 1h (configurable per strategy, default 5m)
  - Timestamp/timezone normalization
  - Rate limiting + retries with exponential backoff
  - Local caching with PostgreSQL cache index
- **Feature Pipeline** — Technical indicator computation
  - RSI, MACD, ATR, returns, volatility
  - Extensible for ML feature engineering
- **ModelStrategy Placeholder** — ML-ready inference interface
  - `predict(features) -> signal/confidence`
  - Training offline (separate from trading runtime)
  - Model versioning (file + metadata in DB)

### Data Sources
- Exchange APIs via CCXT abstraction
- Store raw + derived data in PostgreSQL

### WP2.3: `bar_alignment.py` (pure multi-symbol alignment leaf)
- Stdlib only: no logging, env, wall clock, I/O or metrics, and no `data.*` / `trading.*` /
  `common.*` import. Bars are duck-typed through `.timestamp` and `.volume`; the timeframe is passed
  in as integer seconds. Everything persisted or hashed is an int, str or bool (never a float).
  AST-pinned in `tests/unit/test_wp23_alignment.py` (exact import whitelist, banned names, no float
  literal / `float` / `/` / `**`; `leaf_problems` has a mutated-source self-test).
- Import direction: `trading -> data` (and `apps -> data`), never `data -> trading`. The engine,
  runner, optimizer and `apps/api` import `data.bar_alignment` directly (it is not re-exported from
  `data/__init__`). Nothing under `packages/data` may import `trading` or `apps`; no package-wide
  test pins that yet (WP23-P2-ARCH-10, carry-forward).
- The tolerance numbers live ONLY in its `_TOL_*` constants block (U-01 (a)); `BAR_ALIGNMENT_VERSION
  = "wp23-1"`; `BarAlignmentError` has a closed set of 8 codes. Grid anchor: the epoch for every tf
  that divides a day; for 1w the phase of the first bar of the first non-empty symbol. `gap_runs` are
  structural (absent slots only), while `max_gap_bars` merges absent and zero-volume slots (QA-04).
- See `packages/trading/CLAUDE.md` (WP2.3) for how the engine, runner and API use it.

### WP2.3 P0: `fetch_ohlcv_range` pager rules (`services/ccxt_market_data.py`)
- Never stop on a short page or on an empty page. Non-empty page: `cursor = last_ts + tf`; empty
  page: `cursor += page_limit * tf` (a window without trades is skipped). A page that adds no new bar
  (only duplicates or only pre-start bars) logs `fetch_ohlcv_range_no_progress` and ends the fetch.
- Stop at `stop = min(end, floor_tf(now) + tf)`. `now` comes only from the module seam `_now_ms()`
  (tests patch `data.services.ccxt_market_data._now_ms`); `start >= stop` makes zero requests.
- Guarantee: no bar outside `[start, stop)` is returned. The still-forming bar (open time
  `floor_tf(now)`) is inside that interval; trimming it is the caller's job (the API's
  `requested_window`, SY-23-14).
- Page cap: `2 * ceil((stop - start) / (eff * tf)) + 2` with `eff = max(page_limit // 4,
  min(page_limit, longest page so far))` (`page_limit` until a non-empty page), recomputed on every
  non-empty page. On reaching it the pager logs `fetch_ohlcv_range_page_cap` and returns what it has
  gathered (the alignment tail check then rejects a truncated tail). Do not take `eff` from the first
  page only (that made the cap vacuous, WP23-P0-R2-SEC-01) or from ccxt metadata (it overstates).
- `page_limit` must be >= 1 (`ValueError`) and <= the exchange's own maximum per request, or an
  empty-page jump skips windows. Coverage limit of the floor: a count-based exchange is read in full
  only if its maximum is at least about `page_limit / 8` (38 at the default 300); below that the cap
  truncates with a warning (WP23-P0-R2-SEC-03).
- A Retry-After value is capped at `_BACKOFF_CAP` (60 s). Pager logs carry symbol, timeframe, ISO
  timestamps and integers only, never exception text (CF-36); `fetch_ohlcv_range_complete` adds `pages` and `empty_pages`.
- The pager has no deadline: under the live trailing-gap `wait_for` a timeout discards every page
  already gathered (WP23-P0-RISK-03 / SEC-01, carry-forward).
