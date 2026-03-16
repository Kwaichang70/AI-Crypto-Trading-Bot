"""
packages/trading
-----------------
Trading engine core: strategies, execution, risk, portfolio, backtesting,
and orchestration.
"""

from trading.adaptive_optimizer import (
    AdaptiveOptimizer,
    OptimizerState,
    ParameterAdjustment,
    ParameterChange,
    RollbackDecision,
    SAFEGUARDS,
)
from trading.backtest import BacktestRunner
from trading.execution import BaseExecutionEngine
from trading.optimizer import OptimizationEntry, OptimizationResult, ParameterOptimizer
from trading.metrics import (
    BacktestResult,
    EquityCurvePoint,
    TradeStatistics,
    compute_cagr,
    compute_calmar,
    compute_exposure,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_profit_factor,
    compute_returns_from_equity,
    compute_sharpe,
    compute_sortino,
    compute_trade_statistics,
)
from trading.models import Fill, Order, Position, Signal, SkippedTrade, TradeResult, RiskCheckResult
from trading.performance_analyzer import (
    IndicatorAnalysis,
    IndicatorStats,
    PairAnalysis,
    PairStats,
    ParameterAnalysis,
    PerformanceAnalyzer,
    PerformanceReport,
    RSIBucketStats,
    RegimeAnalysis,
    RegimeStats,
)
from trading.portfolio import PortfolioAccounting
from trading.reporting import (
    AlertEvent,
    AlertLevel,
    AlertType,
    DailyReport,
    ReportingService,
    WeeklyReport,
)
from trading.risk import BaseRiskManager, RiskParameters
from trading.safety import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResponse,
    GateCheckResult,
    LiveTradingGate,
    LiveTradingGateError,
)
from trading.trade_journal import (
    ExitReasonDetector,
    TradeExcursionTracker,
    TradeSkipLogger,
)
from trading.strategy import BaseStrategy, StrategyMetadata
from trading.strategy_engine import EngineState, StrategyEngine

__all__ = [
    # Orchestrator
    "StrategyEngine",
    "EngineState",
    # Adaptive Optimizer (Sprint 34)
    "AdaptiveOptimizer",
    "ParameterAdjustment",
    "ParameterChange",
    "RollbackDecision",
    "OptimizerState",
    "SAFEGUARDS",
    # Optimization
    "ParameterOptimizer",
    "OptimizationResult",
    "OptimizationEntry",
    # Backtesting
    "BacktestRunner",
    "BacktestResult",
    "EquityCurvePoint",
    "TradeStatistics",
    # Metric functions
    "compute_cagr",
    "compute_sharpe",
    "compute_sortino",
    "compute_calmar",
    "compute_profit_factor",
    "compute_max_drawdown",
    "compute_max_drawdown_duration",
    "compute_exposure",
    "compute_returns_from_equity",
    "compute_trade_statistics",
    # Strategy
    "BaseStrategy",
    "StrategyMetadata",
    # Execution
    "BaseExecutionEngine",
    # Risk
    "BaseRiskManager",
    "RiskParameters",
    # Safety
    "LiveTradingGate",
    "LiveTradingGateError",
    "GateCheckResult",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerResponse",
    # Trade Journal (Sprint 32)
    "TradeExcursionTracker",
    "ExitReasonDetector",
    "TradeSkipLogger",
    # Performance Analyzer (Sprint 33)
    "PerformanceAnalyzer",
    "PerformanceReport",
    "RegimeAnalysis",
    "RegimeStats",
    "IndicatorAnalysis",
    "IndicatorStats",
    "ParameterAnalysis",
    "RSIBucketStats",
    "PairAnalysis",
    "PairStats",
    # Reporting (Sprint 35)
    "ReportingService",
    "AlertEvent",
    "AlertLevel",
    "AlertType",
    "DailyReport",
    "WeeklyReport",
    # Portfolio
    "PortfolioAccounting",
    # Models
    "Signal",
    "Order",
    "Fill",
    "Position",
    "TradeResult",
    "SkippedTrade",
    "RiskCheckResult",
]
