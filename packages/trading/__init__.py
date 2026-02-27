"""
packages/trading
-----------------
Trading engine core: strategies, execution, risk, portfolio, and orchestration.
"""

from trading.execution import BaseExecutionEngine
from trading.models import Fill, Order, Position, Signal, TradeResult, RiskCheckResult
from trading.portfolio import PortfolioAccounting
from trading.risk import BaseRiskManager, RiskParameters
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import EngineState, StrategyEngine

__all__ = [
    # Orchestrator
    "StrategyEngine",
    "EngineState",
    # Strategy
    "BaseStrategy",
    "StrategyMetadata",
    # Execution
    "BaseExecutionEngine",
    # Risk
    "BaseRiskManager",
    "RiskParameters",
    # Portfolio
    "PortfolioAccounting",
    # Models
    "Signal",
    "Order",
    "Fill",
    "Position",
    "TradeResult",
    "RiskCheckResult",
]
