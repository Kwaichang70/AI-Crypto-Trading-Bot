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
    "is_profit_factor_infinite",
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
    profit_factor: float | None = None
    profit_factor_is_infinite: bool = False
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
    profit_factor: float | None = Field(
        default=None,
        description=(
            "Gross profit / gross loss. None when no trades (indeterminate) or "
            "when all trades are winners (infinite — see profit_factor_is_infinite). "
            "0.0 when all trades are losers. Positive float in the normal mixed case."
        ),
    )
    profit_factor_is_infinite: bool = Field(
        default=False,
        description="True iff there are winning trades but zero losing trades",
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
) -> float | None:
    """
    Compute profit factor: gross profit / gross loss.

    Parameters
    ----------
    trades : Sequence[TradeResult]
        Completed round-trip trades.

    Returns
    -------
    float | None
        ``None`` if no trades (indeterminate — check :func:`is_profit_factor_infinite`
        to distinguish from the infinite case).
        ``None`` if gross_profit > 0 and gross_loss == 0 (infinite — use the
        sibling helper :func:`is_profit_factor_infinite` to detect this state).
        ``0.0`` if all trades are losers (no winners, well-defined sentinel).
        Positive ``float`` in the normal mixed case.

    Notes
    -----
    A profit factor > 1.0 indicates a profitable system on a gross basis
    (before considering the return of capital).  Values above 2.0 are
    generally considered strong.  The three-state encoding (None/0.0/float)
    is wire-safe: JSON ``null`` signals indeterminate or infinite; the sibling
    boolean ``is_profit_factor_infinite`` disambiguates.
    """
    if not trades:
        return None

    gross_profit = Decimal("0")
    gross_loss = Decimal("0")

    for trade in trades:
        if trade.realised_pnl > Decimal("0"):
            gross_profit += trade.realised_pnl
        elif trade.realised_pnl < Decimal("0"):
            gross_loss += abs(trade.realised_pnl)

    if gross_loss == Decimal("0"):
        if gross_profit > Decimal("0"):
            return None
        return 0.0

    return float(gross_profit / gross_loss)


def is_profit_factor_infinite(trades: Sequence[TradeResult]) -> bool:
    """Return True iff gross_profit > 0 AND gross_loss == 0.

    This is the wire-safe companion to :func:`compute_profit_factor`.  When
    ``compute_profit_factor`` returns ``None``, callers must check this helper
    to distinguish the "infinite" state (winners exist, no losers) from the
    "indeterminate" state (empty trade list).

    Parameters
    ----------
    trades : Sequence[TradeResult]
        Completed round-trip trades.

    Returns
    -------
    bool
        ``True`` iff the trade list is non-empty, gross_profit > 0, and
        gross_loss == 0.  ``False`` for empty lists, all-loser lists, and
        normal mixed lists.
    """
    if not trades:
        return False
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    for trade in trades:
        if trade.realised_pnl > Decimal("0"):
            gross_profit += trade.realised_pnl
        elif trade.realised_pnl < Decimal("0"):
            gross_loss += abs(trade.realised_pnl)
    return gross_profit > Decimal("0") and gross_loss == Decimal("0")


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
    pf_is_infinite = is_profit_factor_infinite(trades)

    return TradeStatistics(
        total_trades=total,
        winning_trades=winning_count,
        losing_trades=losing_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        profit_factor_is_infinite=pf_is_infinite,
        average_trade_pnl=average_pnl,
        average_win=average_win,
        average_loss=average_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )


# ===================================================================
# Sprint 44 QT-005: extended drawdown / tail-risk / regime metrics
# ===================================================================


def compute_ulcer_index(equity_curve: Sequence[EquityCurvePoint]) -> float:
    """Ulcer Index — RMS of percentage drawdowns from running peak.

    Penalises long, slow drawdowns far more than max-DD does.  Two
    strategies with the same max-DD but different drawdown durations
    will have meaningfully different Ulcer scores — useful for catching
    "slow bleed" strategies that look fine by max-DD alone.

    Formula::

        peak_t        = max(E_0 .. E_t)
        dd_t (%)      = (peak_t - E_t) / peak_t * 100      (>= 0)
        Ulcer Index   = sqrt( mean( dd_t^2 ) )

    Returns ``0.0`` for empty or single-point curves (no drawdown to
    measure).  Always non-negative; lower is better.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = float(equity_curve[0].equity)
    squared_dd = 0.0
    n = 0
    for point in equity_curve:
        eq = float(point.equity)
        if eq > peak:
            peak = eq
        if peak > 0.0:
            dd_pct = (peak - eq) / peak * 100.0
        else:
            dd_pct = 0.0
        squared_dd += dd_pct * dd_pct
        n += 1

    if n == 0:
        return 0.0
    import math as _m

    return _m.sqrt(squared_dd / n)


def compute_omega_ratio(
    returns: Sequence[float],
    threshold: float = 0.0,
) -> float:
    """Omega ratio at threshold ``θ``: ``sum(gains above θ) / sum(losses below θ)``.

    Captures the entire return distribution rather than collapsing it
    to a mean+variance like Sharpe / Sortino.  Omega(0) > 1 means more
    upside than downside relative to zero; higher is better.

    Returns ``inf`` when there are no returns below threshold and at
    least one above (pure win-streak — informational, not actionable).
    Returns ``0.0`` when there are no returns at all OR no returns above
    threshold (the strategy never beat the benchmark).
    """
    if not returns:
        return 0.0

    gains = 0.0
    losses = 0.0
    for r in returns:
        diff = r - threshold
        if diff > 0:
            gains += diff
        elif diff < 0:
            losses += -diff
    if gains <= 0.0:
        return 0.0
    if losses <= 0.0:
        return float("inf")
    return gains / losses


def compute_cvar(
    returns: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """Conditional Value-at-Risk (Expected Shortfall) at ``confidence`` level.

    The mean return across the WORST ``(1 - confidence)`` fraction of
    observations.  At 95% confidence, CVaR is the average return on the
    worst 5% of bars — a much sharper tail-risk measure than max-DD or
    standard deviation because it directly captures left-tail mass.

    Returns ``0.0`` for empty inputs.  Result is signed — negative
    indicates an expected loss in the tail (the typical case).

    Parameters
    ----------
    returns:
        Per-period returns (e.g. from :func:`compute_returns_from_equity`).
    confidence:
        VaR confidence level; must be in ``(0, 1)``.  Default ``0.95``
        meaning we look at the worst 5% of bars.
    """
    if not returns:
        return 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    sorted_returns = sorted(returns)
    tail_fraction = 1.0 - confidence
    # Number of observations in the tail — at least 1 so single-return
    # inputs still yield a defined CVaR.
    tail_n = max(1, int(len(sorted_returns) * tail_fraction))
    tail = sorted_returns[:tail_n]
    return sum(tail) / len(tail)


def compute_exposure_adjusted_sharpe(
    sharpe: float,
    exposure_pct: float,
) -> float:
    """Sprint 44 QT-006 — divide Sharpe by ``sqrt(exposure)``.

    Selective strategies that spend only 30 % of bars in market are
    structurally penalised by the standard time-weighted Sharpe because
    the idle bars contribute zero-variance noise that inflates the
    sample count and deflates the ratio.  Dividing by ``sqrt(exposure)``
    re-normalises so an always-in baseline (exposure=1.0) is unchanged
    while a 25 %-exposure strategy gets its Sharpe doubled.

    Returns the raw Sharpe when exposure is ``<= 0`` (degenerate input,
    no trades) so callers cannot accidentally divide by zero.
    """
    if exposure_pct <= 0.0:
        return sharpe
    import math as _m

    return sharpe / _m.sqrt(exposure_pct)


# ===================================================================
# Sprint 44 QT-012: buy-and-hold benchmark comparison
# ===================================================================


def compute_buy_and_hold_return(
    benchmark_prices: Sequence[float],
) -> float:
    """Total return of a buy-and-hold benchmark over the same window.

    Computed as ``(end_price - start_price) / start_price`` so it can be
    compared directly against ``BacktestResult.total_return_pct``.
    Returns ``0.0`` for empty / single-price inputs.
    """
    if len(benchmark_prices) < 2:
        return 0.0
    start = benchmark_prices[0]
    end = benchmark_prices[-1]
    if start == 0.0:
        return 0.0
    return (end - start) / start


def compute_alpha_beta(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> tuple[float, float]:
    """Ordinary least-squares regression of strategy returns on benchmark.

    Returns ``(alpha, beta)`` where:

    * ``beta``  = ``Cov(R_s, R_b) / Var(R_b)`` — sensitivity to benchmark.
    * ``alpha`` = ``mean(R_s) - beta * mean(R_b)`` — return component
      that is NOT explained by benchmark exposure.

    Both series must be the same length; mismatched input returns
    ``(0.0, 0.0)``.  ``Var(R_b) = 0`` returns ``(mean(R_s), 0.0)`` —
    a flat benchmark yields zero beta by construction.
    """
    n = len(strategy_returns)
    if n == 0 or n != len(benchmark_returns):
        return 0.0, 0.0

    mean_s = sum(strategy_returns) / n
    mean_b = sum(benchmark_returns) / n

    cov = sum(
        (s - mean_s) * (b - mean_b)
        for s, b in zip(strategy_returns, benchmark_returns, strict=False)
    ) / n
    var_b = sum((b - mean_b) ** 2 for b in benchmark_returns) / n

    if var_b == 0.0:
        return mean_s, 0.0

    beta = cov / var_b
    alpha = mean_s - beta * mean_b
    return alpha, beta


def compute_information_ratio(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> float:
    """Information ratio = mean(active_return) / stdev(active_return).

    Where ``active_return_t = R_strategy_t - R_benchmark_t``.  Measures
    consistency of out-performance per unit of tracking-error risk.
    Returns ``0.0`` on mismatched lengths or zero tracking error.
    """
    n = len(strategy_returns)
    if n == 0 or n != len(benchmark_returns):
        return 0.0

    active = [
        s - b for s, b in zip(strategy_returns, benchmark_returns, strict=False)
    ]
    mean_active = sum(active) / n
    if n < 2:
        return 0.0
    var_active = sum((a - mean_active) ** 2 for a in active) / (n - 1)
    if var_active == 0.0:
        return 0.0
    import math as _m

    return mean_active / _m.sqrt(var_active)
