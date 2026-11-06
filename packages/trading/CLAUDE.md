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
  (orchestrator paper + live, the integration harness, the `BacktestRunner`,
  default on since WP2.1c). Never construct `CircuitBreaker` / `CircuitBreakerConfig` directly in
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
- Backtests: ON by default since WP2.1c (SY-21-13, closes CF-26): `BacktestRunner` and
  `ParameterOptimizer` default `enable_circuit_breaker=BACKTEST_CIRCUIT_BREAKER_DEFAULT`
  (`True`, `trading.safety`); the API backtest, the optimizer, the ML gate and the research
  scripts all use it (scripts: `--no-circuit-breaker`). There is NO API toggle. An explicit
  `enable_circuit_breaker=False` is the pre-2.1c path and stays byte-identical to the 2.1a
  goldens (pin it in any test calibrated to the no-breaker numbers). The value is persisted as
  `engine_semantics.backtest_circuit_breaker` on backtest runs and optimization `run_metadata`
  (absent on paper/live; `false` on 2.1a/2.1b rows, never rewritten, never read as legacy).
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
  up to 1000) is fetched through the paginated `fetch_ohlcv_range`: 4 pages of 300 when dense;
  with holes (since WP2.3 P0 a short page never ends the fetch) up to about 2·ceil(gap/page)+2
  requests, all inside the same 40 s `wait_for` bound (WP23-P0-RISK-04). A timeout discards
  every page already gathered, so on a slow exchange the seed can be looser than a partial
  fetch would give (WP23-P0-RISK-03 / SEC-01, carry-forward: pager deadline). A gap that cannot
  be fetched, or is longer, logs `trailing_stop.seed_partial`.
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
  a raw ticker dict (`test_t68_*`). The pre-existing sites are fixed in cf36b (`ccxt_retry.*`,
  `live.ticker_fetch_failed`, `live.invalid_ticker_price`, the eight `live.*` I/O sites,
  `engine.bracket_exit_error` / `trailing_stop_error`, `engine.warmup_*` / `poll_*`): they log
  `**exc_log_fields(exc)`, and the traceback lives only in a DEBUG `<event>.trace` event
  (`tests/unit/test_cf36_sites.py`, `tests/integration/test_cf36_sites_realio.py`). New code in this
  package follows the same pattern; `translate_ccxt_error` never returns exception text.
  Layer 1 only by design (ARB-CF36-06): `live.fill_parse_failed`, `live.reconciliation_transition_failed`
  and the `.exception` catch-alls (`engine.start_failed`, `runs.live_engine_error`).
- See `reports/vp2-wp1.3b/synthesis-spec.md` section 5.3 and
  `docs/runbooks/circuit-breaker-halt.md`.

### WP2.1a (one cost model: `trading/costs.py`)
- `CostProfile` (frozen, Decimal-only, bps) is the ONLY cost source. `RiskParameters` has NO fee or
  slippage field (do not add one: C25 was two cost sources). `costs.py` is a leaf: stdlib only,
  reads no environment (AST-pinned, `tests/unit/test_wp21_costs.py`).
- Registry `COST_PROFILES` is append-only; every fingerprint is pinned by a test. The default is the
  code constant `DEFAULT_COST_PROFILE_ID` (one reviewed line; never env: a `*COST_PROFILE*` env var
  is ignored and logged by name only as `cost_profile.env_ignored`). Docs name the default by id,
  not by number.
- Wiring: `BacktestRunner` / `ParameterOptimizer` / `PaperExecutionEngine` / `StrategyEngine` take
  `cost_profile` (`None` -> the current default). `run_paper_engine` / `run_live_engine` take it as a
  REQUIRED keyword. `LiveExecutionEngine(cost_profile=...)` is an ESTIMATE: the live ledger always
  books the exchange's real fees; the profile only feeds the log-only `cost_profile.tier_mismatch`
  (start) and `cost_profile.fill_fee_drift` (per real fill, 5 bps) checks.
- The run's profile is frozen in `RunORM.config["cost_profile"]` (+ `engine_semantics`) at create;
  promote overwrites it with the default; auto-retry copies it; resume / boot / salvage resolve it
  WITHOUT writing (`resolve_run_cost_profile`; absent or invalid -> a legacy profile, never raises).
- Exit-floor basis: the API validates at `policy_cost_model(run_profile)` = the stricter of the
  default and the run's own profile; the engines and a resume use the run's own profile (API pass
  implies engine pass).
- Bracket levels sit on the all-in (fee-inclusive) basis (WP1.3b), so a higher taker moves every stop
  closer to the market (X-15). Goldens calibrated on the old costs pin `legacy_backtest_v0`
  explicitly; per-profile goldens are pinned per profile id with analytic per-fill checks.
- See `reports/vp2-wp2.1/synthesis-spec.md`.

### WP2.2 (bar-time clock: `trading/clock.py`)
- A backtest stamps every fill, order, trade and equity point with BAR CLOSE time
  (`close_i = max(ts) + timeframe`, the same value as the WP1.5 risk clock); the seed equity point is
  `t0 = max(first bar open)`. Paper and live keep the wall clock (`datetime.now(tz=UTC)` through
  `trading.clock._wall_now`). `runs.*` audit columns, logs, uuid4 ids and monotonic telemetry stay wall.
- `clock.py` is a leaf (stdlib + structlog). `BacktestRunner.run` creates ONE `BarClock(t0)` and hands it
  to `PortfolioAccounting(clock=bar_clock.now)` and `StrategyEngine(clock=bar_clock)`. The execution
  engine gets it ONLY through `StrategyEngine.__init__ -> attach_clock` (BACKTEST only; never use
  `PaperExecutionEngine(clock=)` in the runner). `run_backtest` computes `close_i` once and passes the
  SAME object to `bar_clock.advance` and `portfolio.advance_clock`.
  A direct `StrategyEngine` built without a clock gets a run-scoped fallback `BarClock`, installed INSIDE the
  `run_backtest` `with` (attach to the execution engine first, then the engine's own clock) so a rejected attach
  leaves nothing behind. On exit the scope undoes only what it installed: onto the execution engine captured at
  entry, and only `if clock is bar_clock` (never a re-read or swapped engine). So one engine can run several
  backtests; a concurrent second `run_backtest` on the same engine raises `RuntimeError` (type name only).
- Guards (identity, never `isinstance(x, WallClock)` / `type(x) is`): BACKTEST `StrategyEngine` accepts
  `None` or a `BarClock`; PAPER and LIVE accept `None` or `WALL_CLOCK`; `LiveExecutionEngine.attach_clock`
  accepts only `WALL_CLOCK`; neither `BaseExecutionEngine` nor `LiveExecutionEngine` has a `clock=`
  constructor parameter. Messages carry type names only. Fail-closed tripwires (`RuntimeError`):
  `StrategyEngine.start` (not BACKTEST), `run_live_loop`, `LiveExecutionEngine.on_start`,
  `run_paper_engine` / `run_live_engine`.
- Ambient `ContextVar`: `use_clock` is entered ONLY in `StrategyEngine.run_backtest` (after an explicit
  `isinstance(.., BarClock)` raise, never an `assert`), `utc_now` is used ONLY as a pydantic
  `default_factory`, and every persisted field is still stamped explicitly. `ambient_clock()` is read only
  at the tripwire sites, as an operand of `is` / `is not`. Do not wrap a `yield` in `use_clock`, spawn
  tasks/threads inside `run_backtest`, or adopt Python 3.14 (`thread_inherit_context`) without
  re-reviewing this (a test fails under that flag).
- Test seam for paper/live time (S18 / WP4.8): monkeypatch `trading.clock._wall_now` and nothing else.
  `trading.portfolio._utc_now` (the WP1.5 seam, kept by name so its monkeypatches still work) delegates to it
  at call time, so the portfolio's seed / curve / position stamps follow the same patch. The `datetime.now`
  literal in its docstring is load-bearing for `test_wp15a_contracts::test_wp15_t18` until that test is
  retargeted at delegation (CF); do not reword it.
  Never patch `WALL_CLOCK` or `WallClock.now`, never inject a fake clock into a paper/live engine
  (the guards reject it by design), and wrap a `BarClock` instead of patching an instance (`__slots__`).
- Same-bar events share a stamp; `(timestamp, bar_index)` totally orders equity points. At most one
  `Fill` per `Order` (pinned): reopen the sequence-key decision (SY-22-06) if partial fills are added.
- Lots are dropped on a full-flatten SELL (`close_lot` in both managers), so an equal bar-time `lot_key`
  on a same-bar reopen can never revive the closed lot.
- Known limit (WP2.4): the MTF context (C10) still reads wall time in API backtests. (The
  multi-symbol index pairing, where a lagging symbol carried the leading symbol's close, is gone
  since WP2.3; see below.)
- See `reports/vp2-wp2.2/synthesis-spec.md` and `reports/vp2-wp2.2/synthesis-r2-guards.md`.

### WP2.3 (multi-symbol bar alignment: `data/bar_alignment.py`)
- BACKTEST only; paper/live are untouched (CF-23-01..03). Policy `union_present_mask`: multi-symbol
  `run_backtest` walks the sorted union of observed bar timestamps (`build_timeline`, at the old
  `min(len)` site). Per step only PRESENT symbols get `on_bar`, a mark, a bracket/trailing check, a
  fill and a history step; an absent symbol keeps its last mark and, if held, still counts as
  exposed. No synthetic bar, ever (no ffill).
- Single symbol (`len(self._symbols) == 1`, a pinned literal): index walk, no `build_timeline`, no
  validation, byte-identical to before. Multi-symbol full-row guard: `build_timeline` raises
  `bar_alignment_off_grid` (phase mismatch, duplicate or non-increasing bar) or
  `bar_alignment_not_trimmed` (first or last step not a full row) before any state change, so
  `t0 = max(first opens) == timeline[0]`. The WP2.2 clock lines are textually unchanged and
  `test_wp22_ast_pins.py` is unedited; the clock never reads `Step.ts`.
- `BacktestRunner.run` order: `_validate_bars` -> `align_bars(window=requested_window)` ->
  telemetry -> `check_alignment` -> warmup check on `len(timeline)` -> rebind `bars_by_symbol` to
  the aligned dict -> the unchanged `t0` line -> `run_id` + engine. A rejection happens before
  `run_id` and engine construction. Everything after the rebind sees the execution window: terminal
  MTM, start/end/duration, `total_bars = len(timeline) - W`, exposure denominators, the OOS range
  (`oos_start_index` is an ordinal into the timeline). Enforced for more than one distinct symbol
  or whenever `requested_window` is given; a single symbol without a window (goldens, scripts, ML
  gate) is report-only.
- Tolerances are code constants (the `_TOL_*` block in the leaf, U-01 (a)); never env, never an
  API field (`alignment_tolerance=` on the runner/optimizer is for tests and scripts). Abort when
  missing > 1% of the grid (warn > 0.1%; zero-volume counts as missing); `max_gap_bars` > 4 (warn >= 2),
  counted over absent OR zero-volume slots (QA-04); head > 0; tail > 1; any off-grid bar; per
  walk-forward test fold missing > max(1, floor(2% * n)) or a run > 4.
- `BarAlignmentError(ValueError)`: a closed set of 8 codes (`bar_alignment_` + `off_grid`,
  `no_common_row`, `head_coverage`, `tail_coverage`, `missing_bars`, `gap_too_long`,
  `fold_tolerance`, `not_trimmed`), fixed message `bar alignment check failed (<code>)`; precedence
  is code order, then `self._symbols` order. Log from `.code` / `.violations`, never `str(exc)`. The
  API maps it to 422 `{"code": <code>, "error_type": "BarAlignmentError"}`
  (`bar_alignment_http_error`), before `except ValueError` in runs and around `optimizer.run`.
- Optimizer: `ParameterOptimizer.run` aligns and checks ONCE (`_align_once`), including every
  walk-forward test fold (`fold_check`), before any combination. Inner runners get
  `record_alignment=False` (they trim and build the timeline: no tolerance check, no telemetry; only the
  structural `off_grid` / `no_common_row` check still runs on an empty multi-symbol timeline), and
  `except BarAlignmentError: raise` precedes every per-combination `except Exception`.
- Walk-forward: `WalkForwardValidator.split` splits by TIME: fold boundaries are ordinals on the
  union timeline, snapped to full rows at both edges (rolling `train_start` too), each symbol sliced
  by `bisect`. Unequal bar counts are legal; a window without a full row raises `ValueError`. A single
  symbol keeps the old index slices.
- Duplicate symbols: `StrategyEngine.__init__` raises `ValueError("duplicate symbols are not
  allowed")` in every mode. The API rejects them at the request edge (run create and optimize, 422)
  and fails closed with `duplicate_symbols_in_config` at resume, paper boot recovery and promote
  (CF-23-20). Never dedupe silently.
- BACKTEST drops a signal whose symbol is absent at the step (`_drop_absent_symbol_signals`, log
  `engine.signal_dropped_symbol_absent`), before the entry filter and the signal count; paper/live
  keep it (test-pinned).
- Report: `runner.last_alignment` / `optimizer.last_alignment` (reset at every `run`);
  `BacktestResult` has NO alignment field. Persisted as `config["bar_alignment"] =
  report.to_metadata()` (ints, strings, bools) on backtest config and optimize `run_metadata`.
  Compare runs by `presence_digest` (sha256 over sorted `symbol|ts_ms` lines of the bars handed to
  the engine), never by `fingerprint`: the optimizer computes `holes_*` with warmup 0, so its
  fingerprint differs on identical data. The digest covers the execution window, so it depends on
  the whole basket and the window, not on one symbol's data alone (QA-08).
- Marker `engine_semantics.bar_alignment = "wp23-1"` on backtest and optimize only;
  `timestamp_clock` stays `"wp22-1"`. `legacy_bar_alignment` (run responses,
  `bar_alignment_response_fields`) is True iff backtest AND the symbol count is not exactly one AND
  the marker is not in `KNOWN_BAR_ALIGNMENTS`; it never raises. Single-symbol legacy runs are not
  flagged.
- Telemetry: `bar_alignment_runs_total{outcome}`, `bar_alignment_missing_bars_total{symbol,timeframe}`,
  `bar_alignment_rejected_total{reason}` via `common.metrics` directly (no `run_id` label); logs
  `backtest.bar_alignment` / `_gaps` / `_rejected` carry numbers, symbols, ISO times and codes only.
- Window: the API passes `requested_window = (start, min(end, floor_tf(now)))` and reads `now` once,
  BEFORE the fetch (`runs._window_now`); runner and engine never read the wall clock. 1w caveat:
  `floor_tf` is epoch-anchored (Thursday 00:00 UTC), so from Thursday to Sunday the forming weekly
  bar is not trimmed (WP23-P4-CR-03, carry-forward).
- API request bounds (WP23-P0-SEC-02, backtest create and optimize only; 422 `{"code": ...}` before any
  idempotency claim or fetch, never an echo of the request): more than `MAX_BACKTEST_SYMBOLS` (20) symbols
  `too_many_symbols`; a start before `MIN_BACKTEST_START` (2009-01-03) or an overflowing UTC conversion of
  it `start_too_early`; `ceil((end - start) / tf) * symbols` above `MAX_BACKTEST_BARS` (500 000) or an
  overflowing end `range_too_large` (`check_backtest_request_bounds`, `api/services/cost_profiles.py`).
  Paper/live carry no symbol cap (WP23-P4-SEC-06, carry-forward).
- See `reports/vp2-wp2.3/synthesis-spec.md` and `Documentation/WP2.3-register.md`.

### WP2.5 (metric sampling: `trading/equity_sampling.py`, `trading/metrics.py`)
- A backtest computes its return metrics from ONE equity value per bar, not from every curve point.
  Before WP2.5 `n` counted warmup points, one point per bar and one extra point per fill (C13). The
  sample is built AFTER the run from `result.equity_curve`; `strategy_engine.py`, `portfolio.py`, `engines/`
  and `test_wp22_ast_pins.py` are untouched since 01d393d, and the curve, fills, trades, `EquitySnapshot`
  rows and every golden digest are byte-identical.
- `trading.equity_sampling` is a stdlib-only leaf (AST-pinned, `tests/unit/test_wp25_ast_pins.py`): no
  `trading.*`/`common.*`/`data.*` import, no logging, env or wall clock. Points are duck-typed
  (`.timestamp`, `.equity`). API: `sample_equity_per_bar(points, *, start_at, timeframe_s, mode=...) ->
  BarSample`, `last_per_stamp`, `simple_returns`, `EquitySamplingError(ValueError)` with the closed codes
  `equity_sampling_unordered` and `equity_sampling_no_base_point` (message `equity sampling failed (<code>)`).
- Semantics: collapse to the last point per stamp (a decreasing stamp raises); keep stamps `>= start_at`
  (none left = empty sample, n 0); the first kept stamp MUST equal `start_at` (the base point), else
  `equity_sampling_no_base_point`. All kept stamps on `start_at + k*tf` = mode `calendar_asof`: slots
  `k = 0..K` each take the equity of the last stamp `<=` the slot, and `carried_slots` /
  `carried_stamps` list the slots `k >= 1` without an own point (a multi-symbol all-absent slot carries
  the previous value). Otherwise mode `present_steps` (distinct stamps, `carried_slots = 0`).
  `BarSample` (frozen) has `stamps`, `equity` (`n_returns + 1` values, or none when empty), `n_returns`,
  `returns()` and `from_stamp(at)` (base = as-of value at `at`; in calendar mode `at` must be a grid stamp;
  before the first stamp it fails closed; after the last it gives an empty sample). Grid stamps are bar
  CLOSE times; an open-labelled benchmark series shifts +1 timeframe before a join.
- `BacktestRunner.run`: `_step_close(timeline, k, t0, tf_s)` is `t0` for `k == 0`, else `timeline[k-1].ts + tf`.
  `trading_start_at = _step_close(timeline, warmup_bars, ...)` (= `curve[W].timestamp`, also for W = 0);
  `trading_end_at = equity_curve[-1].timestamp`, or `None` only when the sample is empty (the run
  stopped before the base point). Production code NEVER passes `mode=` to the sampler (AST-pinned,
  `**kwargs` counts); the automatic off-grid fallback to `present_steps` (single-symbol library runs only)
  logs `backtest.metric_sampling_fallback` with `metric_sampling` and `n_returns` only.
- `trading.metrics.compute_return_metrics(returns, periods_per_year) -> ReturnMetrics(n, sharpe, sortino, psr,
  skew, kurtosis)` is the ONLY place where Sharpe, Sortino, PSR and n are derived (the full run and the OOS
  window both call it, so SR and n cannot come from different samplings; AST-pinned: no inline moments in
  `backtest.py`). `return_moments` = population skewness and NON-excess kurtosis, `(0.0, 3.0)` below 4
  observations. PSR uses `n = len(returns)`; `_PSR_MIN_OBSERVATIONS = 30` now counts BARS, so fewer than
  30 post-warmup returns give `psr = None` and `confidence_flag = None`. DSR is unchanged (no n input); the
  real DSR is WP2.7.
- `compute_sharpe` and `return_moments` treat `std == 0` or `std <= _ZERO_STD_REL_TOL * |mean|`
  (`_ZERO_STD_REL_TOL = 1e-9`) as zero std (Sharpe 0.0). `compute_cagr(initial, final, days: float)` keeps
  the HEAD arithmetic `ratio ** (365.25 / days) - 1`, returns `inf` on `OverflowError` and never raises.
- `periods_per_year(tf)` (`metrics.py`) is strict: an unknown timeframe raises `ValueError("unknown timeframe
  for annualisation")`. There is no fallback in the runner (AST-pinned: `backtest.py` does not use
  `TIMEFRAME_PERIODS_PER_YEAR`), and the factor is resolved once, before the simulation. The table equals
  `365.25 * 86400 / tf_seconds` for every `TimeFrame`. The legacy path of `scripts/backfill_metrics_v2.py`
  still has its own `.get(tf, 8_766.0)` on purpose (rows without a marker must reproduce).
- What reads the sample: Sharpe, Sortino, PSR, skew/kurtosis, `n_observations`, `confidence_flag`,
  `max_drawdown_duration_bars` (`_max_dd_duration(sample.equity)`, in bars), `cagr` and `oos_sharpe` /
  `oos_n_returns`. `cagr = compute_cagr(sample.equity[0], final_equity, span_days)` with
  `span_days = max(1.0, (trading_end_at - trading_start_at) / 86400)` (an empty sample gives 0.0);
  `calmar_ratio = compute_calmar(cagr, max_dd)`. `max_drawdown_pct` STAYS over all curve points (so it
  agrees with every point's `drawdown_pct` and with the portfolio router). `start_date`, `end_date` and
  `duration_days` are unchanged (bar-open based).
- OOS is cut BY TIME: `oos_start_at = _step_close(timeline, oos_start_index, ...)`, then
  `compute_return_metrics(sample.from_stamp(oos_start_at).returns(), P)`; `oos_sharpe` and `oos_n_returns`
  come from that one result (gap-free: `oos_n_returns == N - oos_start_index`). The engine's
  `oos_equity_curve_offset` stays in the summary, but the runner reads it only for the `is None` check and
  the info log (AST-pinned).
- 7 new `BacktestResult` fields, all defaulted: `metrics_version: str = ""` (the runner writes
  `METRICS_VERSION`), `trading_start_at`, `trading_end_at` (`datetime | None`), `periods_per_year`
  (`float | None`), `metric_sampling` (`"calendar_asof" | "present_steps" | None`), `carried_slots: int = 0`,
  `oos_start_at` (`datetime | None`). In calendar mode `n_observations == total_bars + carried_slots`.
- `METRICS_VERSION = "wp25-1"` and `KNOWN_METRICS_VERSIONS` live in `trading.metrics` (not in the leaf,
  which `metrics.py` cannot be: it imports pydantic and structlog). Any later change to a persisted
  curve-derived number bumps `METRICS_VERSION`; add the new value to `KNOWN_METRICS_VERSIONS` together with it
  (the `scripts/backfill_metrics_v2.py` guard skips only KNOWN markers).
- All-points returns (`compute_returns_from_equity`) may be called only at the allow-list
  `scripts/wp21_rebaseline.py::_pre_wp25_sharpe` and `scripts/backfill_metrics_v2.py::_recompute_psr_from_snapshots`
  (AST-pinned); never in `backtest.py`, `optimizer.py`, `walk_forward.py` or `apps/`. Future metrics
  (ulcer, omega, CVaR, benchmark joins) must take `BarSample` returns.
- CF-26 freeze (UD-4 (a)): `scripts/wp21_rebaseline.py` computes its `sharpe_pre_wp25` and `cagr` keys
  with `_pre_wp25_sharpe` and `_pre_wp25_cagr` (the 117236f formulas: all points, bar-open `duration_days`),
  so a CF-26 run on any later commit still reproduces the baseline PR #4 waits for.
  `tests/integration/test_wp22_bar_clock.py::TestT16RebaselineUnaffected` (the 117236f digest) and
  `test_wp21c_rebaseline.py` are unedited; feeding the new Sharpe or CAGR into the record makes T-16 fail.
  Do not relabel the record to wp25-1.
- API (`apps/api`): `engine_semantics.metrics = METRICS_VERSION` is stamped on backtest and optimize only
  (`cost_profiles.engine_semantics`; `promote_to_live` re-stamps mode `live`, so it cannot leak);
  `timestamp_clock` and `bar_alignment` keep their own versions. `metrics_response_fields(config, run_mode)`
  returns `{"legacy_metrics": bool}`: True iff the mode is backtest or optimize AND the marker is not a `str`
  in `KNOWN_METRICS_VERSIONS` (missing, unknown, non-string, a malformed `engine_semantics`, `config=None`).
  There is NO single-symbol exemption (unlike `legacy_bar_alignment`), paper/live/unknown modes are never
  flagged, and it neither raises nor logs. It is wired on run list and detail and on optimize POST,
  GET-one and GET-list (wire name `legacyMetrics`).
- Response fields on `BacktestMetricsResponse`: `trading_start_at`, `trading_end_at`, `periods_per_year`,
  `metric_sampling`, `carried_slots`. NOT `oos_start_at` (engine-level; WP2.7) and NOT `metrics_version`.
  Leaderboard: `leaderboard_eligible` adds `is_leaderboard_metrics_eligible` (known marker), and
  `/portfolio/summary` counts the runs it excludes in `legacy_metrics_runs_excluded`. ML: `train_model` writes
  `extra["walk_forward"]["metrics"]`; the activation gate warns (does not reject) when that label is not
  known. Optimize persists and returns non-finite entry metrics as `None` (`_scrub_non_finite`; ranking
  unchanged). Old rows are labelled, never recomputed. Paper/live are unchanged and carry no marker.
- Carry-forwards: the "known marker" rule exists in four places (`schemas.py`, `cost_profiles.py`,
  `backfill_metrics_v2.py`, `model_activation_gate.py`); keep them in step with `KNOWN_METRICS_VERSIONS`.
  CAGR/Calmar over short windows are extrapolations, and the optimizer's `cagr`/`calmar` ranking now depends on
  each trial's span (WP2.7).
- See `reports/vp2-wp2.5/synthesis-spec.md` and `Documentation/WP2.5-register.md`.
