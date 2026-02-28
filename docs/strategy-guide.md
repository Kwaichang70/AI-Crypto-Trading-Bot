# AI Crypto Trading Bot -- Strategy Development Guide

## Strategy Interface

Every trading strategy inherits from `BaseStrategy` and implements one required
method: `on_bar`. The strategy receives historical OHLCV bars and returns zero
or more `Signal` objects indicating desired trading actions.

```python
from collections.abc import Sequence
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.models import Signal
from common.models import OHLCVBar

class MyStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="my_strategy",
        version="1.0.0",
        description="Example strategy",
        author="Your Name",
        tags=["trend"],
    )

    @property
    def min_bars_required(self) -> int:
        return 50  # minimum history needed for indicators

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        # bars is sorted oldest-first, most recent last
        # NEVER access beyond bars[-1] -- that would be look-ahead bias
        ...
        return []
```

### Required Implementation

| Method/Property       | Required | Description                                        |
|-----------------------|----------|----------------------------------------------------|
| `on_bar(bars)`        | Yes      | Process bar history, return Signals                |
| `min_bars_required`   | Yes      | Minimum bars before meaningful signals             |
| `metadata`            | Recommended | Class-level StrategyMetadata for discovery      |
| `parameter_schema()`  | Recommended | JSON Schema for API parameter validation        |
| `_validate_params()`  | Optional | Pydantic-based parameter validation                |
| `on_start(run_id)`    | Optional | Initialise state, load models, warm up buffers     |
| `on_stop()`           | Optional | Flush buffers, persist state, release resources    |

### Strategy Lifecycle

```
1. __init__(strategy_id, params)    Constructor -- validate params
2. on_start(run_id)                  Once before first bar
3. on_bar(bars) x N                  Called on every completed candle
4. on_stop()                         Once after last bar or on shutdown
```

### Constructor

The constructor receives a unique `strategy_id` and a `params` dictionary.
The `strategy_id` is used as the originating identifier on all emitted Signals.

```python
def __init__(self, strategy_id: str, params: dict[str, Any] | None = None):
    super().__init__(strategy_id, params)
    # Access validated params via self._params
    self._fast_period = self._params.get("fast_period", 10)
    self._slow_period = self._params.get("slow_period", 30)
```

## Signal Model

A Signal represents a trading intention emitted by a strategy. Signals are
consumed by the ExecutionEngine, which converts them into Orders after risk
checks.

```python
from decimal import Decimal
from common.types import SignalDirection
from trading.models import Signal

signal = Signal(
    strategy_id=self._strategy_id,
    symbol="BTC/USDT",
    direction=SignalDirection.BUY,       # BUY, SELL, or HOLD
    target_position=Decimal("1000"),     # notional size in quote currency
    confidence=0.85,                      # float in [0, 1]
    metadata={                            # arbitrary context
        "fast_ma": 45200.50,
        "slow_ma": 44800.30,
        "reason": "golden_cross",
    },
)
```

### Signal Fields

| Field             | Type            | Description                                   |
|-------------------|-----------------|-----------------------------------------------|
| `strategy_id`     | str             | Must match your strategy's ID                 |
| `symbol`          | str             | Trading pair (e.g. "BTC/USDT")                |
| `direction`       | SignalDirection  | BUY, SELL, or HOLD                            |
| `target_position` | Decimal         | Desired notional size in quote currency (>=0)  |
| `confidence`      | float           | Confidence score in [0, 1], scales position size |
| `generated_at`    | datetime        | Auto-set to current UTC time                  |
| `metadata`        | dict            | Strategy-specific context (indicator values)   |

### Signal Semantics

- `direction=BUY` with `target_position > 0`: open or increase a long position
- `direction=SELL` with `target_position=0`: close the position entirely
- `direction=HOLD`: no action (can be omitted -- returning an empty list is equivalent)
- `confidence` scales the final position size calculated by the RiskManager
- An empty return from `on_bar` is treated as HOLD for all symbols

## Built-in Strategies

Three baseline strategies are planned for the MVP. Each demonstrates a
different signal generation pattern.

### MA Crossover (Trend Following)

Generates BUY signals when a fast moving average crosses above a slow moving
average, and SELL signals on the inverse crossover.

Parameters:

| Parameter      | Type | Default | Description                  |
|----------------|------|---------|------------------------------|
| `fast_period`  | int  | 10      | Fast MA lookback period      |
| `slow_period`  | int  | 30      | Slow MA lookback period      |
| `ma_type`      | str  | "sma"   | Moving average type (sma/ema)|

Minimum bars required: `slow_period + 1`

### RSI Mean Reversion

Generates BUY signals when RSI drops below an oversold threshold and SELL
signals when RSI rises above an overbought threshold.

Parameters:

| Parameter        | Type  | Default | Description                |
|------------------|-------|---------|----------------------------|
| `rsi_period`     | int   | 14      | RSI lookback period        |
| `oversold`       | float | 30.0    | Buy threshold              |
| `overbought`     | float | 70.0    | Sell threshold             |

Minimum bars required: `rsi_period + 1`

### Breakout

Generates BUY signals when price breaks above the highest high of the lookback
period, and SELL signals when price breaks below the lowest low.

Parameters:

| Parameter        | Type | Default | Description                    |
|------------------|------|---------|--------------------------------|
| `lookback`       | int  | 20      | Donchian channel lookback      |
| `atr_period`     | int  | 14      | ATR period for stop placement  |
| `atr_multiplier` | float| 2.0     | Stop distance in ATR multiples |

Minimum bars required: `max(lookback, atr_period) + 1`

## ML Strategy Pattern

The `ModelStrategy` base class extends `BaseStrategy` with hooks for
feature extraction and model inference. This pattern separates the ML
pipeline from trading logic.

```python
class ModelStrategy(BaseStrategy):
    """Base for ML-driven strategies."""

    def on_start(self, run_id: str) -> None:
        super().on_start(run_id)
        self._model = self._load_model()

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        if len(bars) < self.min_bars_required:
            return []

        features = self._extract_features(bars)
        prediction = self._model.predict(features)
        confidence = self._model.predict_proba(features)

        return self._prediction_to_signals(
            prediction, confidence, bars[-1]
        )

    def _load_model(self):
        """Load trained model weights. Override in subclass."""
        raise NotImplementedError

    def _extract_features(self, bars: Sequence[OHLCVBar]) -> Any:
        """Extract feature vector from bar history. Override in subclass."""
        raise NotImplementedError

    def _prediction_to_signals(self, prediction, confidence, current_bar):
        """Convert model output to Signal objects. Override in subclass."""
        raise NotImplementedError
```

### Feature Vector Schema Convention

ML strategies should document their feature schema in the `metadata` field
of emitted signals:

```python
metadata={
    "features": {
        "rsi_14": 42.5,
        "ma_ratio": 1.02,
        "volume_zscore": 0.8,
        "atr_pct": 0.015,
    },
    "model_version": "v2.1.0",
    "prediction_raw": 0.72,
}
```

## Parameter Validation

Use Pydantic models for parameter validation. Override `_validate_params()`
and `parameter_schema()` in your strategy.

```python
from pydantic import BaseModel, Field

class MACrossParams(BaseModel):
    fast_period: int = Field(default=10, ge=2, le=200)
    slow_period: int = Field(default=30, ge=5, le=500)
    ma_type: str = Field(default="sma", pattern="^(sma|ema)$")

class MACrossoverStrategy(BaseStrategy):

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        validated = MACrossParams(**params)
        if validated.fast_period >= validated.slow_period:
            raise ValueError(
                f"fast_period ({validated.fast_period}) must be < "
                f"slow_period ({validated.slow_period})"
            )
        return validated.model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return MACrossParams.model_json_schema()
```

The API uses `parameter_schema()` to validate strategy configuration payloads
before instantiation. The frontend's dynamic form builder also reads this
schema via `GET /api/v1/strategies/{name}/schema`.

## Testing Strategies

### Unit Test Structure

Test each strategy in isolation by constructing synthetic bars and asserting
on the returned signals.

```python
import pytest
from decimal import Decimal
from datetime import datetime, UTC
from common.models import OHLCVBar
from common.types import TimeFrame, SignalDirection

def make_bar(close: float, index: int = 0) -> OHLCVBar:
    """Helper to create a bar with a given close price."""
    return OHLCVBar(
        symbol="BTC/USDT",
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=datetime(2025, 1, 1, index, tzinfo=UTC),
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=Decimal("100"),
    )

def test_ma_crossover_buy_signal():
    strategy = MACrossoverStrategy(
        "test_ma",
        {"fast_period": 3, "slow_period": 5},
    )
    strategy.on_start("test-run")

    # Build a series where fast MA crosses above slow MA
    prices = [100, 99, 98, 97, 96, 97, 99, 102, 105, 108]
    bars = [make_bar(p, i) for i, p in enumerate(prices)]

    signals = strategy.on_bar(bars)

    assert len(signals) >= 1
    assert signals[0].direction == SignalDirection.BUY
    assert signals[0].strategy_id == "test_ma"
    assert signals[0].symbol == "BTC/USDT"
    assert 0 <= signals[0].confidence <= 1
```

### Test Principles

1. Test with both trending and ranging market data
2. Verify that `on_bar` returns empty list when `len(bars) < min_bars_required`
3. Test edge cases: all bars same price, single bar, gap bars
4. Assert signal fields: strategy_id, symbol, direction, confidence range
5. Verify parameter validation rejects invalid inputs
6. Test `on_start` / `on_stop` lifecycle hooks if they have side effects

## Backtesting

### Running a Backtest

Use `BacktestRunner` to run a deterministic backtest with your strategy.

```python
import asyncio
from decimal import Decimal
from common.types import TimeFrame
from trading.backtest import BacktestRunner

async def run_backtest():
    runner = BacktestRunner(
        strategies=[
            MACrossoverStrategy("ma_cross", {"fast": 10, "slow": 30}),
        ],
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        slippage_bps=5,          # 0.05% slippage
        maker_fee_bps=5,         # 0.05% maker fee
        taker_fee_bps=10,        # 0.10% taker fee
        seed=42,                  # deterministic
    )

    result = await runner.run(bars_by_symbol)

    print(f"Total Return: {result.total_return_pct:.2%}")
    print(f"CAGR:         {result.cagr:.2%}")
    print(f"Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"Sortino:      {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2%}")
    print(f"Win Rate:     {result.win_rate:.2%}")
    print(f"Profit Factor:{result.profit_factor:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Total Fees:   {result.total_fees_paid}")

asyncio.run(run_backtest())
```

### BacktestResult Metrics

| Metric                      | Type    | Description                                |
|-----------------------------|---------|--------------------------------------------|
| `total_return_pct`          | float   | (final - initial) / initial                |
| `cagr`                      | float   | Compound Annual Growth Rate                |
| `max_drawdown_pct`          | float   | Worst peak-to-trough decline               |
| `max_drawdown_duration_bars`| int     | Longest recovery period in bars            |
| `sharpe_ratio`              | float   | Annualised, risk-free rate = 0             |
| `sortino_ratio`             | float   | Annualised, downside deviation only        |
| `calmar_ratio`              | float   | CAGR / max_drawdown                        |
| `total_trades`              | int     | Completed round-trip trades                |
| `win_rate`                  | float   | Fraction of winning trades                 |
| `profit_factor`             | float   | Gross profit / gross loss                  |
| `average_trade_pnl`         | Decimal | Average PnL per trade                      |
| `exposure_pct`              | float   | Fraction of bars with open positions       |
| `total_fees_paid`           | Decimal | Cumulative fees in quote currency          |
| `equity_curve`              | list    | Per-bar equity + drawdown points           |
| `trades`                    | list    | Full trade log (TradeResult objects)        |

### Deterministic Backtesting

Backtests are deterministic when:

1. A `seed` is provided (sets `random.seed(seed)` before execution)
2. The same bar data is used
3. No external state dependencies

The BacktestRunner validates input data before running:

- All configured symbols must have bar data
- Bars must be sorted by timestamp ascending (no look-ahead bias)
- No duplicate timestamps within a symbol
- Minimum bars must exceed warmup period

## Best Practices

### No Look-Ahead Bias

The `bars` parameter in `on_bar` is a growing window. The current bar is always
`bars[-1]`. Never access data beyond the end of the sequence. The system
enforces this structurally: in backtest mode, `bars[0:i+1]` is passed on step i.

```python
# CORRECT: use only historical data
current_close = bars[-1].close
past_closes = [b.close for b in bars[-20:]]

# WRONG: this would not compile at runtime, but the pattern to avoid
# is any form of indexing that assumes knowledge of future bars
```

### Decimal Arithmetic

All price and quantity calculations must use `Decimal`, not `float`. This
prevents IEEE-754 rounding errors that accumulate over thousands of trades.

```python
from decimal import Decimal

# CORRECT
sma = sum(b.close for b in bars[-10:]) / Decimal("10")

# WRONG -- introduces floating-point precision loss
sma = sum(float(b.close) for b in bars[-10:]) / 10.0
```

### Warm-Up Handling

Set `min_bars_required` to the minimum history your indicators need.
The StrategyEngine skips `on_bar` calls during warmup. Your strategy should
still guard against insufficient data:

```python
@property
def min_bars_required(self) -> int:
    return self._slow_period + 1

def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
    if len(bars) < self.min_bars_required:
        return []
    # ... compute indicators
```

### Signal Confidence

Use the `confidence` field to express the strength of a signal. The
RiskManager scales position size proportionally:

- `confidence=1.0` -- full position size (max allowed by risk rules)
- `confidence=0.5` -- half position size
- `confidence=0.0` -- effectively no position (but signal is still logged)

Confidence should be derived from your strategy's conviction metrics:
RSI distance from threshold, MA divergence magnitude, model probability, etc.

### Strategy Statefulness

Strategies are stateless with respect to orders and positions. They do not
know whether their signals were executed, rejected, or partially filled.
This separation keeps strategies pure signal generators and simplifies
testing.

If your strategy needs internal state (e.g. a trailing stop tracker or
accumulator), use instance variables initialised in `on_start`:

```python
def on_start(self, run_id: str) -> None:
    super().on_start(run_id)
    self._last_signal_bar = -1
    self._entry_price = None
```

### Metadata for Debugging

Always include relevant indicator values in the signal `metadata` dict.
This data is persisted in the `signals` table and enables post-hoc analysis
of why specific signals were generated.

```python
Signal(
    strategy_id=self._strategy_id,
    symbol=symbol,
    direction=SignalDirection.BUY,
    target_position=Decimal("500"),
    confidence=0.82,
    metadata={
        "rsi": float(rsi_value),
        "sma_fast": float(sma_fast),
        "sma_slow": float(sma_slow),
        "bar_index": len(bars) - 1,
    },
)
```
