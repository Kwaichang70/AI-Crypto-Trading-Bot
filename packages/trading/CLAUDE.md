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
