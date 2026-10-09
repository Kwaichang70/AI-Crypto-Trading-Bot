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
- See `reports/vp2-wp1.5/synthesis-spec.md`.
