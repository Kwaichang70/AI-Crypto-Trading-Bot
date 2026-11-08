IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## FastAPI Backend
Python 3.11+ REST API server using FastAPI framework.

### Key Endpoints (from KickOff.md)
- `POST /runs/start` — Start backtest/paper/live run
- `POST /runs/{run_id}/stop` — Stop a run
- `GET /runs/{run_id}/status` — Get run status
- `GET /runs/{run_id}/trades` — Get trades for a run
- `GET /runs/{run_id}/orders` — Get orders for a run
- `GET /runs/{run_id}/metrics` — Get PnL, drawdown, exposure metrics
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics (optional)

### Technical Requirements
- Pydantic models with strict validation
- SQLAlchemy for database access
- Structured JSON logging
- Async where beneficial
- Type hints throughout

### WP2.4 (point-in-time MTF context: API marker, flags, leaderboard)
- Phase A only. The API passes NO signal source to `BacktestRunner` / `ParameterOptimizer`, so every API backtest and optimize
  run uses `UNAVAILABLE` (consumed fields `None`, status `unavailable`; `not_used` for strategies that consume nothing). The
  store, migration 021, backfill CLI and API PIT loading (P5-P7) are not built; Alembic head is still 020.
- `cost_profiles.engine_semantics(mode)` adds `mtf_context = MTF_CONTEXT_VERSION` ("wp24-1") for backtest and optimize only
  (paper/live get none; `promote_to_live` re-stamps mode `live` and pops the `mtf_context` block). `timestamp_clock`,
  `bar_alignment` and `metrics` are not bumped. The run-specific block `config["mtf_context"]` (backtest) and
  `run_metadata["mtf_context"]` (optimize) is written by `runs.py` / `optimize.py` from `runner.last_mtf_context` /
  `optimizer.last_mtf_context` (`.to_metadata()`); it is server-written only, never a request field.
- `cost_profiles.mtf_context_response_fields(config, run_mode, strategy_name=None)` returns `{"legacy_mtf_context": bool,
  "mtf_context_status": str | None}` (wire names `legacyMtfContext`, `mtfContextStatus`). The one rule lives in
  `schemas.is_legacy_mtf_context`: True iff mode is backtest/optimize AND the marker is not a `str` in
  `KNOWN_MTF_CONTEXT_VERSIONS` AND the strategy is in `LEGACY_MTF_CONSUMER_STRATEGIES` (`rsi_mean_reversion`,
  `dca_rsi_hybrid`, `model_strategy`, `ensemble`) or `LEGACY_MTF_CONSUMER_CLASS_NAMES` (optimize rows store
  `strategy_cls.__name__`), or the name is missing / not a usable `str` (fail-closed). The consumer sets are a frozen
  historic fact (who read C10 at `bbe7142`) and do NOT follow `consumes_mtf_signals`; a test keeps them in step with the
  registry. `mtf_context_status` is echoed only if it is in `KNOWN_MTF_CONTEXT_STATUSES`
  (`not_used`/`unavailable`/`partial`/`pit`), else `None`; it is `None` for paper/live. The function never raises and never logs.
  Wired on run list/detail (`run_persistence.py`) and optimize POST / GET-one / GET-list.
- `RunResponse.config` echoes the stored JSONB verbatim, so a tampered `config.mtf_context` is visible there. Bind any UI
  badge to `mtfContextStatus` / `legacyMtfContext`, never to `config.mtf_context.*` (CF-P3-01).
- Leaderboard: `leaderboard_eligible` = trades/confidence AND cost AND metrics AND `is_leaderboard_mtf_eligible` (False iff
  `legacy_mtf_context`). `/portfolio/summary` counts the excluded runs in `legacy_mtf_context_runs_excluded`, counted last
  (cost, then metrics, then mtf). It reads only the marker and the strategy name, never `status`, `source`, `reason` or
  `fields`: `partial` / `unavailable` runs stay eligible but are not WP2.8 promotion evidence for consumer strategies.
- Import discipline: the `apps/` files that import from `trading.mtf_signals` are pinned per name in
  `tests/unit/test_wp24_ast_pins.py` (`_PINNED_IMPORT_NAMES`; `schemas.py` may import only the two version constants). Any new
  importer needs a pin entry reviewed by security; `LIVE_SIGNALS` / `LiveSignals` are never allowed in `apps/`.
- Exact-dict `engine_semantics` pins (five files, plus a whole-`run_metadata` pin) include `"mtf_context": "wp24-1"`; the paper
  pin does not.
- Open: P7 must keep everything written into `config["mtf_context"]` inside closed values (no exception text, DSN, URL, table
  name; CF-P3-08), extend the dynamic-import scan (CF-P3-10), and keep the marker server-written (CF-P3-11). CF-24-19 is the UI
  work. P1-P3 ship in one release.
- See `packages/trading/CLAUDE.md` (WP2.4), `reports/vp2-wp2.4/synthesis-spec.md` and `Documentation/WP2.4-register.md`.
