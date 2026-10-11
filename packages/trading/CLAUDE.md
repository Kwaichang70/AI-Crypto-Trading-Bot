IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Trading Package
Core trading engine containing the heart of the system.

### Components
- **StrategyEngine** — Pluggable strategy interface: `on_bar(data) -> signals`
  - Signals: BUY/SELL/HOLD, target position, confidence (float)
  - Parameter schema + validation per strategy
- **ExecutionEngine** — Order placement and fill simulation
  - Paper mode: simulated fills, partial fills, latency, slippage, fees
  - Live mode: real order placement via CCXT (limit/market), idempotency keys
  - Order state machine: NEW -> PARTIAL -> FILLED/CANCELED/REJECTED
- **RiskManager** — Pre-trade checks and position management
  - Exposure limits, drawdown checks, daily loss limits, max position size
  - Position sizing (fixed fractional of equity)
  - Stop-loss / take-profit / trailing stop (configurable)
  - Kill-switch capability
- **Baseline Strategies** — MA Crossover, RSI Mean Reversion, Breakout (Donchian/ATR)

### Design Principles
- Spot-only for MVP (max leverage = 1)
- Deterministic backtests via seed control
- No silent failures — all errors must be logged and handled
- Fee/slippage model must be configurable (taker %, maker %, slippage bps)


### WP1.3a (exit-config validation, no-pyramiding held gate)
- `exit_config.py` is the ONE pure validator for bracket/trailing config and
  `allow_pyramiding` -- no engine/DB imports. `create_run`/`promote_to_live`/
  `resume_run` (API), `StrategyEngine.__init__`, `BacktestRunner.__init__`
  and `ParameterOptimizer.__init__` all call it; none of them may
  independently re-implement bracket/trailing parsing.
- `None`/`""`/exact `0` mean "unset" for the four bracket pct/multiplier
  fields and `trailing_stop_pct` -- EXCEPT `bracket_atr_period`, which is
  never "off" (0 is a hard 422).
- `BaseStrategy.requires_exit_manager` defaults `True` (fail-closed); every
  registry strategy declares it explicitly in its own class body, test-
  enforced. `default_allow_pyramiding` defaults `False`; only
  `DCARSIHybridStrategy`/`GridTradingStrategy` set it `True`.
- The engine's held gate (`StrategyEngine._entry_blocked_by_held_position`)
  only touches `SignalDirection.BUY`; it runs inside the step-5 per-signal
  loop, after the kill-switch/protective filter. Live pyramiding is banned
  at the API layer AND in `run_orchestrator.run_live_engine` (defence in
  depth) -- never inside `StrategyEngine` itself, so library-mechanics
  tests can still exercise two same-symbol BUYs via an explicit,
  commented `allow_pyramiding=True`.
- See `reports/vp2-wp1.3a/synthesis-spec.md` for the full bounds table
  (E1-E11, W1-W8) and the fix map.

### WP1.5a (daily-PnL & peak semantics)
- `PortfolioAccounting` is the ONLY owner of the daily-loss state
  (`_nav_day_start`, `_worst_nav_pnl` latch, `_peak_equity`, source). Only
  `update_market_prices` and `_apply_fill` (via `_record_nav`) write the peak
  or the latch; `risk_snapshot()` / `get_daily_pnl()` / `get_summary()` are
  pure (test-pinned, WP15-T-19). Engines and risk managers are stateless
  readers of `RiskSnapshot`; never read `get_summary()` string keys.
- Day roll (SYN-01 / Option B): the roll sets `_nav_day_start` provisionally to
  the pre-roll NAV and arms `_day_start_pending`; the first NAV observation of
  the day (`_record_nav`) raises the start if higher, once, before folding into
  the latch. A loss on the bar spanning 00:00 counts toward the new day, its
  gain does not create a cushion; a high outlier on the first mark can latch
  the day (fail-closed). `nav_day_start` / `daily_loss_basis` may step up once
  per day: read them from `risk_snapshot()` on every evaluation, never cache.
- The risk-day clock is `PortfolioAccounting._now()`: wall clock UTC in
  paper/live (injectable `clock=`), the pushed bar CLOSE time in backtests
  (`advance_clock`, called once per bar before any mark). Never add a
  `datetime.now()` to the risk-day path (AST-pinned, WP15-T-18).
- Gate semantics: `daily_pnl <= -(daily_loss_basis * max_daily_loss_pct)`
  blocks (inclusive); basis `<= 0` blocks; `daily_loss_basis=None` keeps the
  legacy `current_equity` basis for direct callers. Both gates stay BUY-only
  via `BaseRiskManager._ENTRY_ONLY_RULES`.
- LIVE drawdown limit is 0.20 via `live_risk_parameters()` (R-02), applied only
  at `run_orchestrator.run_live_engine`; `RiskParameters()` stays 0.30. WP1.9
  must replace it with config-driven per-mode defaults and keep live at 0.20.
- Paper takes sizing + gate inputs from the portfolio through
  `BaseExecutionEngine.attach_risk_state_source` (separate from live's
  `attach_position_source`); the engine-local ledger still supplies
  `open_positions` / held quantity (WP1.10 revisits it).
- WP1.5b (resume seed wiring): `run_persistence.load_resume_snapshot(db, run,
  *, now=None)` fills `ResumeSnapshot.{day_start,day_min,day_first}_equity_hint`,
  `daily_hints_day`, `run_started_at`; both orchestrator `from_fills` sites pass
  `daily_seed=resume.daily_seed` (`None` only for a snapshot that never read
  hints). No coercion anywhere: `_seed_daily_state` type-validates every seed
  field first (`_invalid_seed_field`) and degrades an invalid seed or a
  seeding exception to `unknown` / `reason="seed_invalid"` (BUYs blocked until
  00:00 UTC, exits unaffected); a resume is never aborted by seed data.
- See `reports/vp2-wp1.5/synthesis-spec.md`.

### WP1.6 (circuit breaker wiring, C2/C3/C23)
- `CircuitBreaker.evaluate(RiskSnapshot) -> CircuitBreakerDecision` is the ONLY
  production path (once per processed bar in `StrategyEngine._process_bar`, after
  the marks, before the strategies). The legacy methods (`check`,
  `check_graduated`, `check_daily_limit_reset`, `get_position_size_multiplier`,
  `trip`, `reset`) are unreachable from production (AST-pinned, WP16-T-25);
  removal is CF-29. Never read `get_summary()` keys for a decision.
- Build a breaker ONLY via `build_run_circuit_breaker(run_id, risk_manager.params)`
  (orchestrator paper + live, the integration harness, the `BacktestRunner`
  opt-in). Never construct `CircuitBreaker` / `CircuitBreakerConfig` directly in
  `apps/` or `backtest.py`.
- Every effect is BUY-only (C23): one filter site, `_drop_entry_signals`, first
  matching reason wins (`kill_switch`, `protective_mode`,
  `circuit_breaker_eval_failed`, `circuit_breaker_halt|daily_limit`). The breaker
  NEVER sets `_stop_event` or `_auto_stop_reason` (D4) and has no wall-clock
  read (day = `snapshot.day`, time = `snapshot.as_of`).
- REDUCE rides on `Signal.entry_size_multiplier` (engine-owned, `(0, 1]`, BUY
  only, strategies must not set it) and is applied to the FINAL quantity in
  `BaseExecutionEngine._resolve_order_quantity`; `target_position` is never
  mutated. The SELL branch never reads it.
- HALT auto-recovers (U-1): exit below 0.9 x limit for 3 bars, then REDUCE; REDUCE
  exits below 0.4 x limit for 3 bars; DAILY_LIMIT clears on the `snapshot.day`
  roll. `evaluate` is transactional (state commits only after every computation
  succeeded). Bands are `Decimal` derived from the run's `RiskParameters`.
- Backtests: opt-in (`BacktestRunner(enable_circuit_breaker=True)`), default off (U-2).
- Restart watch (CF-35, user decision): every new breaker enters HALT at
  `halt_exit` (0.9 x limit) while `_evaluations < N` and REDUCE at `reduce_exit`
  (0.4 x limit) while `_evaluations < 2N`; streaks start at 0; nothing is
  persisted. Result: never laxer than the WP1.5 gate; never laxer than an
  uninterrupted breaker from the 2N-th evaluation on; before that at most 2N-1
  laxer evaluations, only at dd < 0.9 x limit (`test_wp16_restart_watch.py` T-46,
  T-48). Do not shorten either window, make the REDUCE window equal to the HALT
  window, or add a resume-only flag: each re-opens WP16-SEC-206. Never state a
  restart bound without an exhaustive probe over every restart point.
- Telegram: `TelegramNotifier.send_message` and the CB alert path log the exception
  TYPE only (`error_type`), never `str(exc)` / `repr(exc)` / `exc_info`: aiohttp
  exception text embeds the request URL, which contains the bot token
  (WP16-SEC-06; `tests/unit/test_wp16_telegram_log_hygiene.py`). Alert text is sent
  with `parse_mode="HTML"`: never put a raw `<`, `>` or `&` in it.
- See `reports/vp2-wp1.6/synthesis-spec.md`.

### WP1.3b-a (exit anchoring in all modes, trailing semantics, R-04)
- Bracket levels are FROZEN PRICES per lot (`BracketAnchor`), created by
  `StrategyEngine._on_fill_routed` (called from `_route_exit_fills` and
  `_check_resting_orders`; never raises) when a BUY takes a symbol from flat to
  open. The ATR is read once (`atr_for_bars`, W = 6*period+1 bars) and never again;
  `BracketExitManager.check(atr_value=...)` only matters for a lot WITHOUT an anchor
  (lazy `late` anchor). Adds from other orders keep the first entry's levels (U-6);
  further fills of the OPENING order re-blend E (`portfolio.blend_entry_price`), ATR0 stays.
- Anchor bar = pure function of persisted data: `anchor_at = min(order.created_at,
  first opening fill executed_at)`, rule bar = last bar with `ts + tf <= anchor_at`
  (`select_anchor_bar` / `anchor_bar_timestamp`). PAPER/LIVE need exactly W contiguous bars
  ending there; otherwise the anchor is `entry_provisional` (interim levels, finalised on a LATER
  `_process_bar` from the window or up to 3 bounded `fetch_ohlcv` attempts, else `entry_unverified`)
  or `entry_deferred` (no ATR estimate at all yet). BACKTEST uses the in-memory signal bar
  (`_order_signal_bar_ts`) and reads no wall clock (AST-pinned, WP13B-T-31).
- Resume: the orchestrator calls `engine.prime_resume_exit_state(resume.fills, resume.orders)`
  BEFORE `start()` at the paper and live resume sites; `_restore_exit_anchors` (end of
  `_warmup_bar_windows`) recomputes anchor + trailing mark with the SAME functions
  (`lot_entry_from_opening_fills`). Failure never drops protection:
  `resume_fallback_current_atr` (interim, non-final until the attempts are spent; error log
  `bracket_exit.resume_fallback_final` + `bracket_anchor_source_total{source,final}`) or
  `entry_deferred`. The trailing gap (bars missing between the anchor bar and the warmup window,
  up to 1000) is fetched through the paginated `fetch_ohlcv_range` (at most 4 pages of 300);
  a gap that cannot be fetched, or is longer, logs `trailing_stop.seed_partial`.
- Trailing: HWM_0 = gross opening VWAP; ratchets only at bar closes with `bar_ts >
  anchor_bar_ts`; KEPT on a trigger and on a quantity change (no re-seed from the current price).
- R-04: an exit that was not submitted (`[]`, every order terminal+unfilled, or
  `process_signal` raised) clears the manager's pending guard at once
  (`StrategyEngine._submit_exit_signal`, `engine.exit_not_submitted`). Accepted/unknown submits
  keep the guard + TTL; the live in-flight SELL reservation still prevents a double sell.
- `entry.stop_inside_entry_cost` is a log only (U-4): the entry is never dropped.
- Anchoring changes ATR-bracket, trailing and add-to-position backtest/paper results (D-13b-04):
  fixed brackets without trailing/adds are byte-identical. Runs before this change are "legacy
  exit semantics" (`engine.exit_semantics version="wp13b-1"` is logged at start).
- The market-data service retries `fetch_ohlcv` internally; the 1.3b-a fetches only bound the
  total time (`asyncio.wait_for`, 10 s).
- ATR period capped at 49 (W = 6*period+1 <= 295 <= ccxt Coinbase `maxLimit` 300; one fetch always
  suffices; `POLICY_ATR_PERIOD_MAX`, uniform in every mode, pinned by an invariant test). Runs persisted
  with a larger period fail the WP1.3a validator on resume (422 / protective salvage / boot orphan);
  pre-deploy check DS-1: `python scripts/wp13a_exit_config_precheck.py --json`.
- Pending anchors (WP1.3b-a fix delta, PAPER/LIVE): an anchor that cannot be verified at once is armed
  with the INTERIM ATR = `min(ATR over the window ending at the rule bar, ATR of the newest window)`
  (`StrategyEngine._interim_atr`; set once via `BracketExitManager.arm_interim`, never re-armed, never
  looser than the per-bar ATR HEAD used). `_finalize_pending_anchors` (step 4c) then retries the EXACT
  window (always targeting the rule bar, like a resume) at most 3 times, at most one fetch per pass
  engine-wide, and finalises to `entry` / `resume_recomputed`, or keeps the interim levels as
  `entry_unverified` / `resume_fallback_current_atr` (error logs, `bracket_exit.anchor_finalized` with
  `delta_bps`). A resume fallback is NOT final until those attempts are spent. A held lot with no
  protective level logs `bracket_exit.lot_unprotected` (error, bars 1, 11, ...) and counts
  `lot_unprotected_bars_total`. A flat symbol is skipped (no fetch).
- Resume restore: the bracket half and the trailing half have their own `try` (`engine.exit_restore_failed`
  with `part`); a trailing failure seeds HEAD's rule (`max(gross VWAP, closes >= opened_at)`) and logs
  `trailing_stop.seed_partial` (`reason="restore_failed"`); an empty warmup window logs it with
  `reason="no_warmup_bars"`.
- See `reports/vp2-wp1.3b/synthesis-spec.md`.

### WP1.3b-b (live intra-bar exit check)
- LIVE only. A sibling task (`StrategyEngine._intrabar_exit_loop`, started by `run_live_loop` after
  the warmup + anchor restore and stopped in its `finally`) ticks every `exit_check_interval_s`
  (first tick after `min(60, I)` s). The bar loop body is unchanged, so bar-poll timing is the same
  with the feature on or off (`test_t37_*`).
- Config: `Settings.live_exit_check_interval_s` (env `LIVE_EXIT_CHECK_INTERVAL_S`, default 300, `0`
  = off, else 60..3600) -> `exit_config.resolve_run_exit_check` -> engine config
  `exit_check_interval_s` (logged as `runs.intrabar_exit_checks`). OFF for paper/backtest, smoke
  runs (forced), an interval not below the timeframe, and runs without a bracket stop-loss /
  trailing stop. The engine re-validates and raises `ValueError` (non-LIVE, out of range, >= poll
  interval, no `supports_ticker_quotes`). Warning W8 (`bar_close_exit_checks_only`, now also for 1h)
  is shown only when the check is NOT effective.
- A tick: held symbols with a protective level and no unchanged pending exit (none held = ZERO
  exchange calls) -> reading 1 (`BaseMarketDataService.fetch_ticker_quote`: BID, uncached, NOT
  retried, 10 s bound) -> symbols with `bid1 <= level` -> 10 s confirm wait (stop-interruptible,
  lock NOT held) -> reading 2 -> `p = max(bid1, bid2)` must still be `<= level` -> `_cycle_lock`
  (`wait_for(acquire, 5 s)`; busy: abandon, ONE fast retry after 30 s) ->
  `bracket.check(intrabar=True)` (stop-loss only, TP is bar-close only), else
  `trailing.check(intrabar=True)` -> `_submit_exit_signal` (so the WP1.11a cap and the WP1.4b
  idempotent submit apply). Never fires on a missing, invalid (non-positive, non-finite, crossed
  book) or unconfirmed reading, and never uses `last`.
- READ-ONLY (CF-04 / CF-30, AST-pinned by `test_t44_*`): a tick never calls `update_market_prices`,
  never evaluates the circuit breaker, ticks the cooldown, calls a strategy, counts a bar, ages a
  pending TTL, raises the trailing peak, opens a lot or emits a BUY. A real fill is applied at the
  EXCHANGE's own price/fee; only the mark argument is the last CLOSED bar close.
- Kill switch, protective mode, HALT / DAILY_LIMIT and `reconcile_required` never stop an intra-bar
  exit (entries only). `flatten()` holds the lock for its run: the tick abandons and retries.
- Failures: a failed reading never changes pending / levels / HWM; 3 consecutive failed ticks for a
  held symbol log `exit_check.degraded` (error, repeated every 15 min; `exit_check.recovered`
  after); a 429 / DDoS answer abandons the tick and doubles the effective interval up to `max(I, 900
  s)` (below the timeframe), restored after one clean tick. The bar-close SL always keeps working.
- CF-13b-22: `_submit_exit_signal` keeps a per-symbol not-submitted streak (warning 1..2, error at 3
  and every 10th after, `exit_not_submitted_total{trigger,reason}`); a SELL already in flight
  (`LiveExecutionEngine.pending_sell_quantity > 0`) is `inflight_exit_pending`, info, outside the
  streak.
- Log hygiene (CF-36) for all 1.3b-b code: `error_type` and numbers only, never `str(exc)`, a URL or
  a raw ticker dict (`test_t68_*`). The pre-existing `str(exc)` sites remain on CF-36.
- See `reports/vp2-wp1.3b/synthesis-spec.md` section 5.3 and
  `docs/runbooks/circuit-breaker-halt.md`.
