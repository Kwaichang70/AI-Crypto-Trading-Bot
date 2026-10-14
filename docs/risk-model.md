# AI Crypto Trading Bot -- Risk Model

## Overview

The risk management system enforces capital preservation through three
mechanisms: a 6-stage pre-trade check pipeline, fixed-fractional position
sizing, and emergency stop controls (circuit breaker + kill switch). All
risk checks are synchronous with zero I/O in the hot path.

MVP scope: spot-only trading, no leverage, no short positions.

## Risk Parameters

All configurable limits with their defaults. Percentages are expressed as
decimal fractions (0.01 = 1%).

```python
@dataclass(frozen=True, slots=True)
class RiskParameters:
    # Position limits
    max_open_positions: int = 3
    max_position_size_pct: float = 0.10      # max 10% of equity per position

    # Trade-level risk
    per_trade_risk_pct: float = 0.01         # risk 1% of equity per trade
    max_order_size_quote: Decimal = Decimal("10000")  # hard cap per order

    # Run-level circuit breakers
    max_daily_loss_pct: float = 0.08         # block BUYs at a day's NAV loss >= 8% (WP1.5)
    max_drawdown_pct: float = 0.30           # block BUYs at drawdown >= 30% (paper/backtest);
                                             # LIVE runs use 0.20 via live_risk_parameters() (R-02)

    # Fees and slippage are NOT risk limits (WP2.1): they live in
    # trading.costs.CostProfile, one per run -- see "Fee Model" below.

    # Cooldown
    cooldown_after_loss_streak: int = 3      # bars to pause after streak
    loss_streak_count: int = 3               # losses that trigger cooldown
```

### Validation Rules

- `per_trade_risk_pct` must be in (0, 0.05] -- hard cap at 5% risk per trade
- `max_drawdown_pct` must be in (0, 0.50] -- hard cap at 50% drawdown
- `max_open_positions` must be >= 1

These guards prevent misconfiguration from causing catastrophic capital loss.

## Pre-Trade Check Pipeline

The DefaultRiskManager runs six checks in order on every proposed order.
All violations are collected before a verdict is issued. If any violation
is blocking, the order is rejected. Warnings are logged but do not block.

```
Order proposed by ExecutionEngine
         |
         v
+--------------------+
| 1. Kill Switch     |  Is the kill switch active?
|    BLOCKING        |  If yes: reject immediately
+--------+-----------+
         |
         v
+--------------------+
| 2. Loss Cooldown   |  Are we in post-loss cooldown?
|    BLOCKING        |  If cooldown_bars_remaining > 0: reject
+--------+-----------+
         |
         v
+--------------------+
| 3. Max Positions   |  Count non-flat open positions
|    BLOCKING        |  If count >= max_open_positions: reject
+--------+-----------+
         |
         v
+--------------------+
| 4. Daily Loss      |  gate daily_pnl vs daily_loss_basis * max_daily_loss_pct
|    BLOCKING        |  If daily_pnl <= -threshold: reject (BUY only)
+--------+-----------+
         |
         v
+--------------------+
| 5. Max Drawdown    |  (peak_equity - current_equity) / peak_equity
|    BLOCKING        |  If drawdown >= max_drawdown_pct: reject
+--------+-----------+
         |
         v
+--------------------+
| 6. Order Size      |  Notional cap + concentration cap
|    MAY ADJUST      |  quantity may be reduced to fit limits
+--------+-----------+
         |
         v
   RiskCheckResult
   (approved, adjusted_quantity, rejection_reasons, warnings)
```

### Check Details

**Stage 1 -- Kill Switch:** A one-way latch. Once triggered, all orders are
blocked until an operator explicitly calls `reset_kill_switch()`. The
StrategyEngine also checks this at the bar level in LIVE mode.

**Stage 2 -- Loss Cooldown:** After `loss_streak_count` consecutive losing
trades, trading pauses for `cooldown_after_loss_streak` bars. The cooldown
counter decrements on every `tick_cooldown()` call. When it reaches zero,
the consecutive loss counter resets.

**Stage 3 -- Max Positions:** Counts all non-flat open positions. If the
count equals or exceeds `max_open_positions`, new entries are blocked. Closing
trades (SELL on existing positions) are always allowed.

**Stage 4 -- Daily Loss (WP1.5):** The engines feed the gate the
mark-to-market NAV change since the UTC day start, taken as a running minimum
(`RiskSnapshot.gate_daily_pnl(nav) = min(worst_nav_pnl, nav - nav_day_start)`):
once the loss reaches the limit, BUYs stay blocked for the rest of the UTC
day even if NAV recovers (the latch). The threshold is
`daily_loss_basis * max_daily_loss_pct` with the basis
`daily_loss_basis = max(0, min(nav_day_start, initial_cash))`, fixed from the
first NAV observation of the UTC day: a gain at that first observation can
raise it once (never above `initial_cash`); after that it does not move, so a
profit later in the day never raises it (no compounding). The boundary is
inclusive: `daily_pnl <= -threshold` blocks. A basis `<= 0` blocks
(fail closed). The check is BUY-only: a reducing SELL (strategy exit,
stop-loss, take-profit, trailing stop, flatten) is downgraded to a warning.
Direct callers that pass no `daily_loss_basis` keep the legacy basis
`current_equity`.

**Stage 5 -- Max Drawdown:** Computes `(peak_equity - current_equity) / peak_equity`
from the portfolio's marked NAV and running peak (WP1.5: paper and backtest no
longer use the paper engine's own fill-price ledger). If the drawdown fraction
equals or exceeds `max_drawdown_pct`, new BUY orders are blocked. The limit is
0.30 for paper and backtest and 0.20 for LIVE runs (R-02, set at the single
live risk-manager construction site via `live_risk_parameters()`). Exposure-
reducing SELLs are never blocked by this gate.

**Stage 6 -- Order Size:** Validates the requested order quantity against
two caps: the absolute notional cap (`max_order_size_quote`) and the
concentration cap (`max_position_size_pct * current_equity`). If the order
exceeds either cap, the quantity is adjusted downward. This is the only check
that modifies the order rather than rejecting it outright.

### RiskCheckResult

```python
class RiskCheckResult(BaseModel):
    approved: bool                    # False = do not submit
    adjusted_quantity: Decimal        # may be < requested quantity
    rejection_reasons: list[str]      # human-readable failure reasons
    warnings: list[str]              # non-blocking concerns
    checked_at: datetime              # UTC timestamp
```

## Position Sizing

Fixed-fractional sizing with three caps. The RiskManager computes the
position size in base asset given the current equity and trade parameters.

### Formula

```
risk_amount = equity * per_trade_risk_pct * confidence

if stop_loss_price is provided:
    stop_distance = abs(entry_price - stop_loss_price) / entry_price
else:
    stop_distance = 0.01  (default 1%)

raw_size_quote = risk_amount / stop_distance
raw_size_base  = raw_size_quote / entry_price
```

### Three Caps

The raw position size is then capped by the minimum of:

1. **Per-trade risk cap:** `equity * per_trade_risk_pct / stop_distance`
2. **Concentration cap:** `equity * max_position_size_pct / entry_price`
3. **Absolute cap:** `max_order_size_quote / entry_price`

```
final_size = min(
    raw_size_base,
    equity * max_position_size_pct / entry_price,
    max_order_size_quote / entry_price,
)
```

The result is rounded down to 8 decimal places (standard crypto precision).

### Confidence Scaling

The `confidence` parameter from the strategy Signal scales the position size
linearly:

- `confidence = 1.0` -- full position size (bounded by caps)
- `confidence = 0.5` -- half the position size
- `confidence = 0.0` -- zero position size (order will not be placed)

This allows strategies to express conviction gradients without binary
all-or-nothing positions.

## Circuit Breaker

The CircuitBreaker (WP1.6) is a graduated, **auto-recovering entry guard**. It
is evaluated once per processed bar from `PortfolioAccounting.risk_snapshot()`
(mark-to-market NAV, running peak, UTC-day start) and uses the run's own
`RiskParameters` (`build_run_circuit_breaker(run_id, params)`), so it applies
the same limits as the WP1.5 gates. It never reads `get_summary()` keys, never
reads the wall clock (the day is `snapshot.day`), never stops the run and
never writes portfolio state. Nothing is persisted and nothing is
operator-resettable: every level is re-derived from data.

**Every effect is BUY-only** (C23). Strategy SELLs, bracket stop-loss /
take-profit, trailing stops, resting orders and `flatten` are never blocked or
scaled. A breaker failure (snapshot or evaluation exception, non-finite
input) drops that bar's BUYs only (fail-closed for entries; exits run).

### Levels and bands

`L` = the run's `max_drawdown_pct` (live 0.20, paper 0.30), `D` = its
`max_daily_loss_pct` (0.08). All values are exact `Decimal`.

| Level | Enters (1 bar) | Leaves | Effect |
|-------|----------------|--------|--------|
| REDUCE | `dd >= 0.5 L` | `dd < 0.4 L` for 3 consecutive bars | BUY quantity x 0.5 (final quantity, after the risk ceiling) |
| DAILY_LIMIT | `gate_daily_pnl <= -(daily_loss_basis * D)`, basis `<= 0`, or unknown day state | first bar of the next UTC day (`snapshot.day`), no debounce | BUYs dropped |
| HALT | `dd >= L` | `dd < 0.9 L` for 3 consecutive bars, then steps down to REDUCE (never straight to OK) | BUYs dropped; run NOT stopped |

| Run | HALT entry / exit | REDUCE entry / exit |
|-----|-------------------|---------------------|
| LIVE | 20% / 18% | 10% / 8% |
| PAPER | 30% / 27% | 15% / 12% |

Escalation is immediate; de-escalation needs N = 3 consecutive evaluations
(processed bars) and an in-band reading resets the count (hysteresis). The
most severe active condition wins (`HALT > DAILY_LIMIT > REDUCE > OK`), and
`active_conditions` lists all of them. The raw HALT and DAILY_LIMIT predicates
are identical to the WP1.5 gates (`>=` / `<=` inclusive), so on any snapshot
where a gate would block, the breaker has `entries_blocked=True`; its
recovery is stricter than the gate (in `[0.9 L, L)` the breaker still blocks).
`pre_trade_check` remains the per-order enforcement of record.

A flat run whose NAV is at or beyond the limit has a constant NAV and never
recovers (neither does the gate); start a new run.

### Restart / resume

A fresh breaker is built on every start and resume; nothing is persisted, so
it cannot know the level an uninterrupted breaker would hold. It therefore
runs a **restart watch**: during its first N evaluations it enters HALT at
`dd >= 0.9 L` (instead of `L`), and during its first 2N evaluations it enters
REDUCE at `dd >= 0.4 L` (instead of `0.5 L`). Streaks start at 0; a level
entered during the watch is then held by the ordinary hysteresis. N and 2N
are exactly how long an uninterrupted breaker that was in HALT can still hold
HALT and then REDUCE. Measured exhaustively (WP1.6 final synthesis, round 2)
and pinned by WP16-T-46:

- never laxer than the WP1.5 gate;
- from the 2N-th evaluation on, never laxer than an uninterrupted breaker;
- before that, at most 2N-1 = 5 laxer evaluations, only at `dd < 0.9 L`
  (at full size only at `dd < 0.4 L`);
- stricter than an uninterrupted breaker for as long as the drawdown stays in
  `[0.9 L, L)` (HALT instead of REDUCE) or `[0.4 L, 0.5 L)` (REDUCE instead
  of OK) after a restart there (CF-35, user decision).

The watch applies to every new instance, including a brand-new run, whose
drawdown starts at 0.

### Backtests

Off by default (`BacktestRunner(enable_circuit_breaker=False)`); WP2.1 turns
it on by default with a re-baseline.

### Observability

Gauge `circuit_breaker_state` (OK=0, REDUCE=1, DAILY_LIMIT=2, HALT=3; zeroed
on engine stop); structured events `circuit_breaker.transition`,
`circuit_breaker.halt_entered` (critical, once per entry),
`circuit_breaker.halt_cleared`, `engine.circuit_breaker_entries_blocked`,
`engine.circuit_breaker_eval_failed`; Telegram on HALT entered / cleared;
`GET /api/v1/runs/{id}/circuit-breaker` (typed state). The reset endpoint is
retired (410 Gone). See `docs/runbooks/circuit-breaker-halt.md`.

### Legacy API

`check()`, `check_graduated()`, `check_daily_limit_reset()`,
`get_position_size_multiplier()`, `trip()` and `reset()` (and the
`max_consecutive_losses` config field) are legacy: unreachable from production
code (AST-pinned), kept only for the existing unit tests until CF-29. The
risk manager's loss-streak cooldown is the only consecutive-loss mechanism.

## Live Trading Gate

The LiveTradingGate is a three-layer activation gate checked before any live
order is submitted. All three layers must pass.

### Layer 1: Environment

The `ENABLE_LIVE_TRADING` environment variable must be explicitly set to `true`.
This is `false` by default in both the application settings and the Docker
Compose configuration. This prevents accidental live trading from
misconfigured deployments.

### Layer 2: API Keys

Both `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET` must be configured and
non-empty. These are stored as Pydantic `SecretStr` fields, which means they
are never serialised to logs, JSON responses, or error messages.

### Layer 3: Confirmation Token

A runtime confirmation token must be provided at run creation time and must
match the `LIVE_TRADING_CONFIRM_TOKEN` stored in the environment. This prevents
programmatic scripts from accidentally creating live runs without operator
awareness.

### Gate Check Result

```python
gate = LiveTradingGate()
result = gate.check_gate(settings, confirm_token="my-secret-token")

if result.passed:
    # All three layers satisfied -- live trading allowed
    ...
else:
    # result.failures contains human-readable reasons
    # result.layer_results shows per-layer pass/fail
    for failure in result.failures:
        print(f"Gate failed: {failure}")
```

### Convenience API

```python
# Raises LiveTradingGateError if any layer fails
gate.require_gate(settings, confirm_token="my-secret-token")
```

## Drawdown Management

### Peak Equity Tracking

The PortfolioAccounting service tracks peak equity continuously. It updates
on every fill and on every bar (via `update_market_prices`). Peak equity is
never decreased -- it only ratchets upward.

```
drawdown = (peak_equity - current_equity) / peak_equity
```

### Max Drawdown Calculation

For backtest results, max drawdown is computed by iterating the full equity
curve and finding the worst peak-to-trough decline. Max drawdown duration
is the longest span (in bars) from a new peak to recovery to that peak.

### Daily PnL, Day Roll and the Daily-Loss Gate (WP1.5)

`PortfolioAccounting` is the single owner of the daily state; engines and the
risk manager only read `risk_snapshot()` (pure).

- **Realised figure (display, API/UI):** `get_daily_pnl()` is realised-only
  (FIX-07) and reads 0 on a new UTC day even before the first fill.
- **Gate figure:** the mark-to-market NAV change since the UTC day start, as a
  running minimum (latch), over the fixed day-start basis (see Stage 4).
- **UTC day roll:** forward-only, at 00:00:00 UTC. A roll happens in
  `update_market_prices`, on a fill, or via `advance_clock`; it sets
  `nav_day_start` provisionally to the NAV at the previous marks, arms a
  one-shot re-base, and clears the realised accumulator and the latch. The
  first NAV observation of the new day (mark or fill) raises `nav_day_start`
  to that NAV if it is higher, before it is folded into the latch, so
  `nav_day_start = max(NAV at the last pre-roll observation, NAV at the first
  observation of the day)`. A low outlier on the last pre-roll mark therefore
  never depresses the next day's start. Asymmetry (RK-05): a loss on the bar
  spanning 00:00 counts toward the new day, its gain does not create a
  cushion. Residual: a high outlier on the first mark of a day becomes the
  day start and can latch that day (fail-closed, BUY-only, until 00:00 UTC).
  The peak never resets.
- **Risk-day clock:** wall clock (UTC) in paper and live. In a backtest
  `StrategyEngine.run_backtest` pushes the bar CLOSE time (bar timestamp +
  timeframe) once per bar, warmup included, so the wall clock is never read
  and a backtest is deterministic. An hourly bar opening 23:00 on day D
  belongs to D+1; every 1d bar is its own day.
- **Resume:** `PortfolioAccounting.from_fills(daily_seed=...)` seeds the day
  state from persisted equity-snapshot hints (`ResumeDailySeed`). The day
  start is unknown, and BUYs stay blocked until the next 00:00 UTC, only when
  a seed exists but has neither a day-start hint nor a run that started today,
  when the seed is dated after the resume's UTC day (`seed_from_the_future`),
  or when any seed field fails the strict type check or seeding raises
  (`seed_invalid`, see below).
  Without a seed (legacy callers, `replay_estimate`) the gate *value* is never
  weaker than HEAD's realised-today value, but the threshold uses the fixed
  day-start basis, so losses in (pct/(1+pct)*S, pct*S] that HEAD's shrinking
  `current_equity` basis blocked now pass, by design. The path can also be
  stricter than HEAD: mark drift since the original fills counts toward the
  resume day (until 00:00 UTC). A `day_first_equity` hint raises a snapshot
  day start to `max(day_start_equity, day_first_equity)`.
- **Resume seed source (WP1.5b):** `run_persistence.load_resume_snapshot`
  reads three hints with two parameterised statements served by
  `ix_equity_snapshots_run_id_timestamp` over `equity_snapshots`, with `T` =
  00:00 UTC of the resume's UTC day:
  `day_start_equity_hint` (latest row with `timestamp < T`, ties by the higher
  `bar_index`), and `day_min_equity_hint` / `day_first_equity_hint` (minimum
  and earliest row with `timestamp >= T`, ties by the lower `bar_index`; one
  statement, so one is set if and only if the other is), plus
  `daily_hints_day` and `run_started_at`. `ResumeSnapshot.daily_seed` is
  `None` only if the hints were never read. Both orchestrator resume paths
  (paper boot recovery and live/protective resume) pass it to `from_fills`.
  The rebuild's own equity point (positions valued at their fill prices) is
  never persisted, so the hints only ever see real marks (WP15B-CR-01).
- **Seed validation, no coercion (WP1.5b):** every seed field is checked
  strictly before any comparison: `Decimal` that is finite and `>= 0` for the
  three equities, a `date` that is not a `datetime` for `as_of_day`, `None` or
  an aware `datetime` for `run_started_at`. A `float`, `int`, `str`, `bool`,
  NaN, negative value or naive datetime is **rejected, never repaired** (the
  column is `Numeric(20, 8)` and yields `Decimal`, so a wrong type is an
  upstream defect). The result is `day_start_source = "unknown"` (BUYs blocked
  until 00:00 UTC, exits unaffected) with `resume.daily_state_unknown`
  (`reason="seed_invalid"`, field and type name only, never the value). Seed
  construction and seeding never abort a resume, a protective resume included.
- **Sizing:** paper and backtest BUYs size on `min(NAV, initial_capital)`,
  exactly like live (D-WP15-B): no compounding.

## Fee Model

### One cost model for every mode (WP2.1)

Fees and slippage are described by ONE frozen, Decimal-only `CostProfile`
(`packages/trading/costs.py`) per run. `RiskParameters` carries **no** cost
field, so backtest, paper and live cannot disagree about costs.

- **Default:** `DEFAULT_COST_PROFILE_ID` in `packages/trading/costs.py`. It is a
  reviewed code constant. No environment variable selects it: a
  `DEFAULT_COST_PROFILE` (or any `*COST_PROFILE*`) variable is ignored and the
  API logs one `cost_profile.env_ignored` warning with the variable *name* only.
  Switching the default, for instance after the operator's fee-tier check
  (deploy runbook, G-LIVE L-21), is a one-line reviewed change.
- **Registry (append-only; the numbers of an existing id never change):**

| id | maker bps | taker bps | slippage bps | class |
|---|---|---|---|---|
| `coinbase_advanced_1` | 60 | 120 | 10 | live-equivalent |
| `coinbase_advanced_d6` | 40 | 60 | 10 | live-equivalent |
| `coinbase_stress_t80` / `coinbase_stress_t100` | 40 | 80 / 100 | 10 | stress (sensitivity only) |
| `frictionless_diagnostic` | 0 | 0 | 0 | diagnostic (gross edge) |
| `legacy_backtest_v0` | 10 | 15 | 5 | legacy (reproduces pre-WP2.1 backtests) |
| `legacy_runtime_v0` | 40 | 60 | 5 | legacy (reproduces pre-WP2.1 paper runs) |

- **Per mode:** a backtest or optimization run accepts any registry id
  (`costProfile` in the request); a paper run only live-equivalent ids; a live
  run accepts none. The profile is frozen in `RunORM.config["cost_profile"]`
  (next to `config["engine_semantics"]`) when the run is created. Promote to
  live overwrites it with the default; auto-retry copies it; resume and boot
  recovery only read it. A run without the key is a *legacy-cost* run (it
  resolves to a legacy profile and is hidden from the default leaderboard).
- **Live:** the ledger always uses the exchange's real fees. The profile is an
  estimate for the exit-cost floor, the stop-cost log and two log-only drift
  warnings: `cost_profile.tier_mismatch` (market taker vs profile, at start)
  and `cost_profile.fill_fee_drift` (a real fill's fee more than 5 bps off).

### Fee Calculation

Fees are calculated per fill in the execution engine from the profile
(`fee_rate = bps / 10_000`, exact Decimal):

```
MARKET orders: fee = quantity * fill_price * profile.fee_rate("taker")
LIMIT orders:  fee = quantity * fill_price * profile.fee_rate("maker")
```

### Slippage Model

Slippage is applied to MARKET orders only:

```
BUY:  fill_price = last_price * (1 + profile.slippage_bps / 10000)
SELL: fill_price = last_price * (1 - profile.slippage_bps / 10000)
```

This models the spread between the quoted price and the actual execution price.

### Exit-cost floor and the all-in bracket basis

The WP1.3a entry-cost floor (E6/E7) is `c_side = taker fee + slippage` of the
profile (a fixed stop-loss or take-profit at or below it is rejected). The
API validates at the stricter of the default profile and the run's own profile;
the engines and a resume use the run's own profile, so no existing run becomes
un-resumable.

Bracket levels sit on the *all-in* (fee-inclusive) entry basis (WP1.3b), so a
higher taker fee pulls every stop closer to the market: a stop at fraction `d`
below the basis is only about `d - c_side` below the entry close. This is a
cost mechanism, not a bug; a re-baseline therefore includes earlier stop-outs,
not only fee drag.

### All-In Cost Basis

The PortfolioAccounting service uses all-in cost basis for position tracking.
Entry fees are included in the average entry price:

```
all_in_entry_price = (fill_price * quantity + fee) / quantity
```

This ensures that PnL calculations accurately reflect the true cost of entering
a position. Exit fees are subtracted from the realised PnL at close time.

### Fee Impact on Metrics

All performance metrics (CAGR, Sharpe, Sortino, profit factor, etc.) are
computed net of fees. The `total_fees_paid` field on BacktestResult provides
the cumulative fee burden for analysis.
