"""
packages/trading/strategies
----------------------------
Concrete strategy implementations that extend ``BaseStrategy``.

Each strategy is a self-contained module with its own parameter validation,
indicator calculations, and signal generation logic.

Available strategies
~~~~~~~~~~~~~~~~~~~~
- **MACrossoverStrategy** -- Dual SMA crossover (trend-following)
- **RSIMeanReversionStrategy** -- RSI overbought/oversold mean-reversion
- **BreakoutStrategy** -- Donchian channel breakout with ATR scaling
- **ModelStrategy** -- ML model-based strategy (Sprint 2 placeholder)
- **DCARSIHybridStrategy** -- DCA + RSI hybrid (systematic accumulation)
- **GridTradingStrategy** -- Grid trading (buy low / sell high at fixed levels)

Usage::

    from trading.strategies import MACrossoverStrategy

    strat = MACrossoverStrategy(
        strategy_id="ma-cross-btc-1h",
        params={"fast_period": 10, "slow_period": 50},
    )
"""

from trading.strategies.breakout import BreakoutStrategy
from trading.strategies.dca_rsi_hybrid import DCARSIHybridStrategy
from trading.strategies.grid_trading import GridTradingStrategy
from trading.strategies.ma_crossover import MACrossoverStrategy
from trading.strategies.model_strategy import ModelStrategy
from trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy

__all__ = [
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BreakoutStrategy",
    "ModelStrategy",
    "DCARSIHybridStrategy",
    "GridTradingStrategy",
]
