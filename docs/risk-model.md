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

    # Fee / slippage model
    taker_fee_pct: float = 0.001             # 0.10% taker fee
    maker_fee_pct: float = 0.0005            # 0.05% maker fee
    slippage_bps: int = 5                    # 5 basis points slippage

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

The CircuitBreaker is an emergency stop mechanism independent of the
RiskManager's pre-trade checks. It evaluates risk conditions on every bar
tick or before every order submission.

### Configuration

```python
class CircuitBreakerConfig:
    max_daily_loss_pct: float = 0.05      # 5% daily loss limit
    max_drawdown_pct: float = 0.15        # 15% drawdown limit
    max_consecutive_losses: int = 5       # 5 consecutive losses
```

### Triggers

The circuit breaker trips when any of these conditions is met:

| Trigger              | Condition                                          |
|----------------------|----------------------------------------------------|
| Daily loss           | `abs(daily_pnl) / equity >= max_daily_loss_pct`    |
| Drawdown             | `drawdown >= max_drawdown_pct`                     |
| Consecutive losses   | `consecutive_losses >= max_consecutive_losses`     |
| Manual trip          | Operator calls `breaker.trip(reason)`              |

### Behavior

- Once tripped, the breaker stays open -- trading is halted
- Logs at CRITICAL severity with `alert="TRADING_HALTED"`
- `is_tripped` returns True until manual reset
- `reset()` clears the trip state but preserves `trip_count` for audit
- Trip count is never reset within a session
- Idempotent: if already tripped, subsequent checks return True without
  re-evaluating conditions (first trip reason wins)

### Manual Controls

```python
breaker = CircuitBreaker(config=CircuitBreakerConfig(), run_id="run-123")

# Automatic check (called on every bar)
if breaker.check(equity=9500, daily_pnl=-600, drawdown=0.12, consecutive_losses=3):
    # Trading halted -- do not submit orders
    ...

# Manual trip
breaker.trip("Operator initiated emergency stop")

# Reset after review (requires explicit operator action)
breaker.reset()

# Inspect state
state = breaker.state  # CircuitBreakerState(is_tripped, trip_reason, tripped_at, trip_count)
```

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
  or when the seed is dated after the resume's UTC day
  (`seed_from_the_future`).
  Without a seed (legacy callers, `replay_estimate`) the gate *value* is never
  weaker than HEAD's realised-today value, but the threshold uses the fixed
  day-start basis, so losses in (pct/(1+pct)*S, pct*S] that HEAD's shrinking
  `current_equity` basis blocked now pass, by design. The path can also be
  stricter than HEAD: mark drift since the original fills counts toward the
  resume day (until 00:00 UTC). A `day_first_equity` hint raises a snapshot
  day start to `max(day_start_equity, day_first_equity)`. (The database
  wiring of the seed is WP1.5b.)
- **Sizing:** paper and backtest BUYs size on `min(NAV, initial_capital)`,
  exactly like live (D-WP15-B): no compounding.

## Fee Model

### Fee Parameters

| Parameter       | Default | Description                            |
|-----------------|---------|----------------------------------------|
| `taker_fee_pct` | 0.001   | Taker fee as fraction (0.1%)           |
| `maker_fee_pct` | 0.0005  | Maker fee as fraction (0.05%)          |
| `slippage_bps`  | 5       | Slippage in basis points (0.05%)       |

### Fee Calculation

Fees are calculated per fill in the execution engine:

```
MARKET orders: fee = quantity * fill_price * taker_fee_pct
LIMIT orders:  fee = quantity * fill_price * maker_fee_pct
```

### Slippage Model

Slippage is applied to MARKET orders only:

```
BUY:  fill_price = last_price * (1 + slippage_bps / 10000)
SELL: fill_price = last_price * (1 - slippage_bps / 10000)
```

This models the spread between the quoted price and the actual execution price.

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
