"""
packages/trading/metrics.py
-----------------------------
Backtest result model and standalone performance metric computations.

Every metric function in this module is a pure function -- it takes numeric
inputs and returns a scalar result.  This makes each formula independently
testable without standing up backtest infrastructure.

Financial formulae
------------------
- CAGR: ``(final / initial) ^ (365 / days) - 1``
- Sharpe (annualised): ``mean(returns) / std(returns) * sqrt(periods_per_year)``
- Sortino: ``mean(returns) / downside_deviation * sqrt(periods_per_year)``
- Calmar: ``CAGR / |max_drawdown|``
- Profit Factor: ``sum(winning_pnl) / |sum(losing_pnl)|``
- Exposure: ``bars_in_market / total_bars``
- Max Drawdown Duration: longest contiguous peak-to-recovery span in bars

All monetary values use ``Decimal`` for precision.  Ratios and percentages
use ``float`` because they are derived statistical quantities where Decimal
precision offers no practical benefit and numpy interop is simpler.
"""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import structlog
from pydantic import BaseModel, Field

from common.types import TimeFrame
from trading.models import TradeResult

__all__ = [
    "BacktestResult",
    "EquityCurvePoint",
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
    "TIMEFRAME_PERIODS_PER_YEAR",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Annualisation factors
# ---------------------------------------------------------------------------
# Crypto markets trade 24/7/365 -- no weekends off.

TIMEFRAME_PERIODS_PER_YEAR: dict[TimeFrame, float] = {
    TimeFrame.ONE_MINUTE: 365.25 * 24 * 60,          # 525_960
    TimeFrame.THREE_MINUTES: 365.25 * 24 * 20,       # 175_320
    TimeFrame.FIVE_MINUTES: 365.25 * 24 * 12,        # 105_192
    TimeFrame.FIFTEEN_MINUTES: 365.25 * 24 * 4,      #  35_064
    TimeFrame.THIRTY_MINUTES: 365.25 * 24 * 2,       #  17_532
    TimeFrame.ONE_HOUR: 365.25 * 24,                  #   8_766
    TimeFrame.FOUR_HOURS: 365.25 * 6,                 #   2_191.5
    TimeFrame.ONE_DAY: 365.25,                        #     365.25
    TimeFrame.ONE_WEEK: 365.25 / 7,                   #      52.18
}


# ---------------------------------------------------------------------------
# Equity curve point
# ---------------------------------------------------------------------------

class EquityCurvePoint(BaseModel):
    """
    A single point on the equity curve, recording portfolio value at a
    specific bar timestamp.

    Attributes
    ----------
    timestamp : datetime
        UTC timestamp of the bar.
    equity : Decimal
        Total portfolio equity (cash + unrealised position value) at
        this point.
    drawdown_pct : float
        Current drawdown from peak equity as a decimal fraction
        (e.g. 0.05 = 5% drawdown).
    """

    model_config = {"frozen": True}

    timestamp: datetime
    equity: Decimal
    drawdown_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Current drawdown from peak as decimal fraction",
    )


# ---------------------------------------------------------------------------
# Trade statistics helper
# ---------------------------------------------------------------------------

class TradeStatistics(BaseModel):
    """Aggregated statistics from a list of completed trades."""

    model_config = {"frozen": True}

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_trade_pnl: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """
    Comprehensive backtest results with all performance metrics.

    This model is the primary output of ``BacktestRunner.run()``.
    It contains run metadata, return metrics, risk metrics, trade
    statistics, the full equity curve, and the trade log.

    All percentage fields are expressed as decimal fractions
    (e.g. 0.10 = 10%).
    """

    # Run metadata
    run_id: str = Field(description="Unique identifier for this backtest run")
    strategy_ids: list[str] = Field(description="Strategy IDs used in this run")
    symbols: list[str] = Field(description="Trading pairs backtested")
    timeframe: TimeFrame = Field(description="Candle timeframe")
    start_date: datetime = Field(description="First bar timestamp (UTC)")
    end_date: datetime = Field(description="Last bar timestamp (UTC)")
    duration_days: int = Field(
        ge=0,
        description="Calendar days between first and last bar",
    )

    # Capital
    initial_capital: Decimal = Field(description="Starting cash in quote currency")
    final_equity: Decimal = Field(description="Ending equity in quote currency")

    # Returns
    total_return_pct: float = Field(
        description="(final_equity - initial_capital) / initial_capital",
    )
    cagr: float = Field(description="Compound Annual Growth Rate")

    # Risk metrics
    max_drawdown_pct: float = Field(
        ge=0.0,
        description="Maximum peak-to-trough decline as decimal fraction",
    )
    max_drawdown_duration_bars: int = Field(
        ge=0,
        description="Longest peak-to-recovery span in bars",
    )
    sharpe_ratio: float = Field(
        description="Annualised Sharpe ratio (risk-free rate = 0)",
    )
    sortino_ratio: float = Field(
        description="Annualised Sortino ratio (downside deviation only)",
    )
    calmar_ratio: float = Field(
        description="CAGR / max_drawdown",
    )

    # Trade statistics
    total_trades: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    profit_factor: float = Field(
        ge=0.0,
        description="Gross profit / gross loss; inf if no losses",
    )
    average_trade_pnl: Decimal
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal

    # Exposure
    total_bars: int = Field(ge=0)
    bars_in_market: int = Field(ge=0)
    exposure_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of bars with an open position",
    )

    # Equity curve
    equity_curve: list[EquityCurvePoint] = Field(
        default_factory=list,
        description="Per-bar equity curve for charting",
    )

    # Trade log
    trades: list[TradeResult] = Field(
        default_factory=list,
        description="All completed round-trip trades",
    )

    # Fee summary
    total_fees_paid: Decimal = Field(
        default=Decimal("0"),
        description="Total fees paid in quote currency",
    )


# ===================================================================
# Standalone metric functions
# ===================================================================


def compute_cagr(
    initial: Decimal,
    final: Decimal,
    days: int,
) -> float:
    """
    Compute Compound Annual Growth Rate.

    Formula
    -------
    CAGR = (final / initial) ^ (365.25 / days) - 1

    Parameters
    ----------
    initial : Decimal
        Starting portfolio value. Must be > 0.
    final : Decimal
        Ending portfolio value.
    days : int
        Number of calendar days in the evaluation period.

    Returns
    -------
    float
        CAGR as a decimal fraction.  Returns 0.0 when inputs are
        degenerate (zero capital, zero days, negative final value).

    Examples
    --------
    >>> compute_cagr(Decimal("10000"), Decimal("11000"), 365)
    0.1  # approximately 10% annual return
    """
    if initial <= Decimal("0") or days <= 0:
        return 0.0

    ratio = float(final / initial)
    if ratio <= 0:
        # Cannot take a fractional power of a non-positive number.
        return -1.0

    exponent = 365.25 / days
    return float(ratio ** exponent) - 1.0


def compute_sharpe(
    returns: Sequence[float],
    periods_per_year: float,
) -> float:
    """
    Compute annualised Sharpe ratio with zero risk-free rate.

    Formula
    -------
    Sharpe = mean(R) / std(R) * sqrt(periods_per_year)

    where R is the series of per-period returns.

    Parameters
    ----------
    returns : Sequence[float]
        Per-period percentage returns as decimal fractions.
    periods_per_year : float
        Number of return observations per year (for annualisation).

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns 0.0 if fewer than 2
        observations or if standard deviation is zero.

    Notes
    -----
    Uses sample standard deviation (N-1 denominator) for an unbiased
    estimate.  This is consistent with industry practice for
    out-of-sample Sharpe estimation.
    """
    n = len(returns)
    if n < 2 or periods_per_year <= 0:
        return 0.0

    mean_r = sum(returns) / n

    # Sample variance (Bessel's correction: N-1)
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance)

    if std_r == 0.0:
        return 0.0

    return (mean_r / std_r) * math.sqrt(periods_per_year)


def compute_sortino(
    returns: Sequence[float],
    periods_per_year: float,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualised Sortino ratio.

    Formula
    -------
    Sortino = (mean(R) - target) / downside_deviation * sqrt(periods_per_year)

    Downside deviation uses only returns below the target (default 0).

    Parameters
    ----------
    returns : Sequence[float]
        Per-period percentage returns as decimal fractions.
    periods_per_year : float
        Number of return observations per year.
    target_return : float
        Minimum acceptable return per period.  Default 0.0.

    Returns
    -------
    float
        Annualised Sortino ratio.  Returns 0.0 if fewer than 2
        observations or if downside deviation is zero.

    Notes
    -----
    Downside deviation is computed using **all** observations in the
    denominator (not just the negative ones), consistent with the
    original Sortino & Price (1994) definition.  Specifically::

        DD = sqrt( sum(min(r - target, 0)^2) / (N - 1) )

    This avoids inflating the ratio by excluding non-negative periods
    from the count.
    """
    n = len(returns)
    if n < 2 or periods_per_year <= 0:
        return 0.0

    mean_r = sum(returns) / n

    # Downside deviation: only negative excess returns contribute
    downside_sq = sum(
        min(r - target_return, 0.0) ** 2 for r in returns
    )
    # Use N-1 for sample statistic consistency
    downside_dev = math.sqrt(downside_sq / (n - 1))

    if downside_dev == 0.0:
        # No downside deviation — all returns met or exceeded target.
        # Return +inf to signal "unboundedly good Sortino", consistent with
        # how compute_profit_factor handles zero gross loss.  Return 0.0 only
        # in the degenerate case where mean return is also below target
        # (mathematically impossible given downside_dev == 0, but guard here
        # for numerical safety).
        return float("inf") if mean_r >= target_return else 0.0

    return ((mean_r - target_return) / downside_dev) * math.sqrt(periods_per_year)


def compute_calmar(
    cagr: float,
    max_drawdown: float,
) -> float:
    """
    Compute Calmar ratio.

    Formula
    -------
    Calmar = CAGR / |max_drawdown|

    Parameters
    ----------
    cagr : float
        Compound Annual Growth Rate as a decimal fraction.
    max_drawdown : float
        Maximum drawdown as a positive decimal fraction (e.g. 0.15 = 15%).

    Returns
    -------
    float
        Calmar ratio.  Returns 0.0 if max_drawdown is zero or near-zero.
    """
    if abs(max_drawdown) < 1e-12:
        return 0.0
    return cagr / abs(max_drawdown)


def compute_profit_factor(
    trades: Sequence[TradeResult],
) -> float:
    """
    Compute profit factor: gross profit / gross loss.

    Parameters
    ----------
    trades : Sequence[TradeResult]
        Completed round-trip trades.

    Returns
    -------
    float
        Profit factor >= 0.  Returns 0.0 if no trades.
        Returns ``float('inf')`` if there are winning trades but zero
        gross loss (no losing trades).

    Notes
    -----
    A profit factor > 1.0 indicates a profitable system on a gross
    basis (before considering the return of capital).  Values above
    2.0 are generally considered strong.
    """
    if not trades:
        return 0.0

    gross_profit = Decimal("0")
    gross_loss = Decimal("0")

    for trade in trades:
        if trade.realised_pnl > Decimal("0"):
            gross_profit += trade.realised_pnl
        elif trade.realised_pnl < Decimal("0"):
            gross_loss += abs(trade.realised_pnl)

    if gross_loss == Decimal("0"):
        if gross_profit > Decimal("0"):
            return float("inf")
        return 0.0

    return float(gross_profit / gross_loss)


def compute_max_drawdown(
    equity_curve: Sequence[EquityCurvePoint],
) -> float:
    """
    Compute maximum peak-to-trough drawdown from an equity curve.

    Parameters
    ----------
    equity_curve : Sequence[EquityCurvePoint]
        Equity curve points ordered by timestamp ascending.

    Returns
    -------
    float
        Maximum drawdown as a positive decimal fraction.
        Returns 0.0 if the curve has fewer than 2 points.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = Decimal("0")
    max_dd = Decimal("0")

    for point in equity_curve:
        if point.equity > peak:
            peak = point.equity
        if peak > Decimal("0"):
            dd = (peak - point.equity) / peak
            if dd > max_dd:
                max_dd = dd

    return float(max_dd)


def compute_max_drawdown_duration(
    equity_curve: Sequence[EquityCurvePoint],
) -> int:
    """
    Compute the longest drawdown duration in bars.

    A drawdown period starts when equity drops below the current peak
    and ends when equity recovers to or exceeds that peak.  The
    duration is measured in number of bars (equity curve points).

    Parameters
    ----------
    equity_curve : Sequence[EquityCurvePoint]
        Equity curve points ordered by timestamp ascending.

    Returns
    -------
    int
        Longest peak-to-recovery span in bars.
        Returns 0 if there are fewer than 2 points or no drawdown
        occurred.
    """
    if len(equity_curve) < 2:
        return 0

    peak = Decimal("0")
    current_dd_duration = 0
    max_dd_duration = 0

    for point in equity_curve:
        if point.equity >= peak:
            # New peak or recovery to previous peak
            peak = point.equity
            if current_dd_duration > max_dd_duration:
                max_dd_duration = current_dd_duration
            current_dd_duration = 0
        else:
            # In drawdown
            current_dd_duration += 1

    # Handle case where the run ends in a drawdown
    if current_dd_duration > max_dd_duration:
        max_dd_duration = current_dd_duration

    return max_dd_duration


def compute_exposure(
    bars_in_market: int,
    total_bars: int,
) -> float:
    """
    Compute percentage of time spent in market.

    Parameters
    ----------
    bars_in_market : int
        Number of bars where at least one position was open.
    total_bars : int
        Total number of bars processed (excluding warm-up).

    Returns
    -------
    float
        Exposure as a decimal fraction in [0, 1].
        Returns 0.0 if total_bars is zero.
    """
    if total_bars <= 0:
        return 0.0
    return min(1.0, bars_in_market / total_bars)


def compute_returns_from_equity(
    equity_curve: Sequence[EquityCurvePoint],
) -> list[float]:
    """
    Compute per-period simple returns from an equity curve.

    Formula
    -------
    R_t = (E_t - E_{t-1}) / E_{t-1}

    Parameters
    ----------
    equity_curve : Sequence[EquityCurvePoint]
        Equity curve points ordered by timestamp ascending.

    Returns
    -------
    list[float]
        Per-period returns as decimal fractions.  Length is
        ``len(equity_curve) - 1``.  Returns empty list if
        fewer than 2 points.

    Notes
    -----
    Returns are computed using simple (arithmetic) returns, not
    log returns.  For the sub-daily periods typical of crypto
    backtests, the difference is negligible.
    """
    if len(equity_curve) < 2:
        return []

    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev_eq = equity_curve[i - 1].equity
        curr_eq = equity_curve[i].equity
        if prev_eq > Decimal("0"):
            ret = float((curr_eq - prev_eq) / prev_eq)
        else:
            ret = 0.0
        returns.append(ret)

    return returns


def compute_trade_statistics(
    trades: Sequence[TradeResult],
) -> TradeStatistics:
    """
    Compute aggregated trade statistics from completed trades.

    Parameters
    ----------
    trades : Sequence[TradeResult]
        Completed round-trip trades.

    Returns
    -------
    TradeStatistics
        Aggregated statistics including win rate, profit factor,
        average PnL, and extremes.
    """
    if not trades:
        return TradeStatistics()

    total = len(trades)
    winners = [t for t in trades if t.realised_pnl > Decimal("0")]
    losers = [t for t in trades if t.realised_pnl < Decimal("0")]

    winning_count = len(winners)
    losing_count = len(losers)

    gross_profit = sum((t.realised_pnl for t in winners), Decimal("0"))
    gross_loss = sum((abs(t.realised_pnl) for t in losers), Decimal("0"))

    total_pnl = sum((t.realised_pnl for t in trades), Decimal("0"))
    average_pnl = total_pnl / total

    average_win = (
        gross_profit / winning_count if winning_count > 0 else Decimal("0")
    )
    average_loss = (
        -(gross_loss / losing_count) if losing_count > 0 else Decimal("0")
    )

    largest_win = max(
        (t.realised_pnl for t in winners), default=Decimal("0")
    )
    largest_loss = min(
        (t.realised_pnl for t in losers), default=Decimal("0")
    )

    win_rate = winning_count / total if total > 0 else 0.0
    profit_factor = compute_profit_factor(trades)

    return TradeStatistics(
        total_trades=total,
        winning_trades=winning_count,
        losing_trades=losing_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_trade_pnl=average_pnl,
        average_win=average_win,
        average_loss=average_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )
