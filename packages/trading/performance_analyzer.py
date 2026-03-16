"""
packages/trading/performance_analyzer.py
-----------------------------------------
Post-trade performance analysis engine (Sprint 33).

Consumes ``TradeResult`` and ``SkippedTrade`` records from the trade journal
and produces a structured ``PerformanceReport``.  The report is designed as a
read-only input contract for the Sprint 34 ``AdaptiveOptimizer``.

No external statistical libraries (scipy, numpy, pandas) are used.  All
computations are implemented with Python stdlib arithmetic.

Usage example
-------------
::

    from trading.performance_analyzer import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer(min_trades=30)
    report = analyzer.analyze(trades=completed_trades, skipped=skipped_trades)

    if report.is_actionable:
        optimizer.consume(report)
    else:
        for warning in report.warnings:
            log.warning(warning)

Thread-safety
-------------
NOT thread-safe.  Designed for single-threaded use within the StrategyEngine
or a dedicated analysis task.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from trading.models import SkippedTrade, TradeResult

__all__ = [
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
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Indicator keys that are parameter echoes, not signal indicators.
# Excluded from correlation / IC analysis.
_EXCLUDED_INDICATOR_KEYS: frozenset[str] = frozenset(
    {"rsi_period", "oversold", "overbought", "close"}
)

# RSI "strong signal" thresholds for false-signal heuristic
_RSI_STRONG_OVERSOLD: float = 25.0
_RSI_STRONG_OVERBOUGHT: float = 75.0
_FGI_EXTREME_FEAR_MAX: float = 24.0
_FGI_EXTREME_GREED_MIN: float = 76.0

# Regime concentration warning threshold
_REGIME_CONCENTRATION_PCT: float = 0.80


# ---------------------------------------------------------------------------
# Private statistical helper functions
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    """Return arithmetic mean, or 0.0 if the list is empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: list[float], ddof: int = 1) -> float:
    """
    Return sample standard deviation (ddof=1 by default), or 0.0 if
    ``len(values) <= ddof``.
    """
    n = len(values)
    if n <= ddof:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - ddof)
    return math.sqrt(variance)


def _per_trade_sharpe(return_pcts: list[float]) -> float:
    """
    Per-trade Sharpe ratio: mean(returns) / std(returns).

    Returns 0.0 when std == 0 or fewer than 2 trades are available.
    No annualisation factor is applied -- unit of analysis is per-trade.
    """
    if len(return_pcts) < 2:
        return 0.0
    std = _safe_std(return_pcts, ddof=1)
    if std == 0.0:
        return 0.0
    return _safe_mean(return_pcts) / std


def _rank_data(values: list[float]) -> list[float]:
    """
    Assign 1-based average ranks to values with tie-breaking.

    Example: [10, 20, 20, 30] -> [1.0, 2.5, 2.5, 4.0]
    """
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        # Average rank for positions i..j-1 (1-based)
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """
    Pearson product-moment correlation coefficient.

    Returns 0.0 on degenerate input (fewer than 3 samples, zero variance,
    unequal lengths).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    product = var_x * var_y
    if product <= 0.0:  # CR-001: catches zero AND negative float noise
        return 0.0
    denom = math.sqrt(product)
    return cov / denom


def _spearman(x: list[float], y: list[float]) -> float:
    """
    Spearman rank correlation coefficient.

    Computed via rank-transform followed by Pearson correlation on the ranks.
    Returns 0.0 on degenerate input (fewer than 3 samples, unequal lengths).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    return _pearson(_rank_data(x), _rank_data(y))


def _sample_factor(n_trades: int, min_required: int) -> float:
    """
    Confidence sample factor: 0.0 below min_required, linear to 1.0 at
    2 * min_required.  Hard floor: 0.0 if n_trades < min_required.
    """
    if n_trades < min_required:
        return 0.0
    return min(1.0, n_trades / (2.0 * min_required))


# ---------------------------------------------------------------------------
# Report data classes -- all frozen Pydantic models
# ---------------------------------------------------------------------------


class RegimeStats(BaseModel):
    """Per-regime performance statistics."""

    model_config = {"frozen": True}

    regime: str
    trade_count: int
    win_count: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    skipped_would_profit: int
    skipped_correctly: int
    skipped_unknown: int


class RegimeAnalysis(BaseModel):
    """Consolidated regime performance breakdown."""

    model_config = {"frozen": True}

    by_regime: list[RegimeStats]
    best_regime: str | None
    worst_regime: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class IndicatorStats(BaseModel):
    """Per-indicator statistical analysis."""

    model_config = {"frozen": True}

    indicator_name: str
    sample_count: int
    correlation_with_pnl: float
    information_coefficient: float
    false_signal_count: int
    false_signal_rate: float


class IndicatorAnalysis(BaseModel):
    """Consolidated indicator contribution analysis."""

    model_config = {"frozen": True}

    by_indicator: list[IndicatorStats]
    most_predictive: str | None
    highest_false_signal_rate: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class RSIBucketStats(BaseModel):
    """Performance statistics for trades within an RSI value bucket."""

    model_config = {"frozen": True}

    bucket_label: str
    bucket_low: float
    bucket_high: float
    trade_count: int
    win_rate: float
    avg_pnl_pct: float
    sharpe: float


class ParameterAnalysis(BaseModel):
    """Strategy parameter effectiveness analysis."""

    model_config = {"frozen": True}

    rsi_buckets: list[RSIBucketStats]
    stop_loss_hit_rate: float
    take_profit_hit_rate: float
    trailing_stop_hit_rate: float
    signal_exit_rate: float
    avg_mfe_winners: float
    avg_mae_losers: float
    avg_mfe_all: float
    avg_mae_all: float
    mfe_beyond_tp_count: int
    mfe_beyond_tp_rate: float
    confidence: float = Field(ge=0.0, le=1.0)


class PairStats(BaseModel):
    """Per-symbol performance breakdown."""

    model_config = {"frozen": True}

    symbol: str
    trade_count: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    sharpe: float
    best_regime: str | None
    worst_regime: str | None
    skip_count: int


class PairAnalysis(BaseModel):
    """Consolidated pair performance analysis."""

    model_config = {"frozen": True}

    by_symbol: list[PairStats]
    best_symbol: str | None
    worst_symbol: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class PerformanceReport(BaseModel):
    """Consolidated analysis report from all four analysis modules."""

    model_config = {"frozen": True}

    generated_at: datetime
    analysis_window_start: datetime | None
    analysis_window_end: datetime | None
    total_trades: int
    total_skipped: int
    overall_win_rate: float
    overall_avg_pnl_pct: float
    overall_confidence: float = Field(ge=0.0, le=1.0)
    is_actionable: bool = Field(
        description=(
            "True only if overall_confidence >= PerformanceAnalyzer.CONFIDENCE_THRESHOLD"
        )
    )
    regime: RegimeAnalysis
    indicators: IndicatorAnalysis
    parameters: ParameterAnalysis
    pairs: PairAnalysis
    warnings: list[str]
    safeguards_applied: dict[str, int | float]


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------


class PerformanceAnalyzer:
    """
    Analyzes trade journal data to identify parameter optimisation opportunities.

    Runs after every N trades or on a time interval.  Produces a structured
    ``PerformanceReport`` that the AdaptiveOptimizer (Sprint 34) will consume
    to adjust strategy parameters.

    Thread-safety
    -------------
    NOT thread-safe.  Designed for single-threaded use within the
    StrategyEngine or a dedicated analysis task.
    """

    # Minimum sample thresholds for statistical validity
    MIN_TRADES_FOR_ANALYSIS: int = 30
    MIN_TRADES_FOR_REGIME: int = 20  # CR-002: risk spec requires 20
    MIN_TRADES_FOR_INDICATOR: int = 10
    MIN_TRADES_FOR_SHARPE: int = 2
    MIN_TRADES_FOR_PAIR_RANKING: int = 5

    # RSI bucket configuration
    RSI_BUCKET_WIDTH: float = 5.0
    RSI_BUCKET_MIN: float = 0.0
    RSI_BUCKET_MAX: float = 100.0

    # False signal confidence threshold (used with explicit "confidence" key)
    FALSE_SIGNAL_CONFIDENCE_THRESHOLD: float = 0.5

    # Overall actionability threshold
    CONFIDENCE_THRESHOLD: float = 0.65

    def __init__(self, min_trades: int = 30) -> None:
        """
        Parameters
        ----------
        min_trades:
            Minimum number of trades required before analysis produces
            actionable results.  Below this threshold the report is still
            generated but ``is_actionable`` is False and warnings are emitted.
        """
        self._min_trades = min_trades
        self._log = structlog.get_logger(__name__).bind(
            component="performance_analyzer"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        trades: list[TradeResult],
        skipped: list[SkippedTrade],
    ) -> PerformanceReport:
        """
        Run all four analysis modules and return a consolidated report.

        Parameters
        ----------
        trades:
            Completed round-trip trades to analyse.
        skipped:
            Trades that were evaluated but not taken.

        Returns
        -------
        PerformanceReport
            Frozen Pydantic model with all analysis results.

        Edge cases
        ----------
        - Empty trades list: report with total_trades=0, empty sub-analyses,
          warning appended.
        - Empty skipped list: regime skipped_* fields are all 0.
        - Trades with None signal_context: excluded from indicator analysis.
        - Trades with None regime_at_entry: excluded from regime analysis.
        """
        warnings: list[str] = []
        n = len(trades)

        if n < self._min_trades:
            warnings.append(
                f"Analysis based on {n} trades "
                f"(minimum {self._min_trades} recommended for statistical significance)"
            )

        # Overall metrics
        overall_win_rate = 0.0
        overall_avg_pnl_pct = 0.0
        if n > 0:
            win_count = sum(1 for t in trades if t.realised_pnl > Decimal(0))
            overall_win_rate = win_count / n
            overall_avg_pnl_pct = _safe_mean([t.return_pct for t in trades])

        # Analysis window timestamps
        analysis_window_start: datetime | None = None
        analysis_window_end: datetime | None = None
        if trades:
            analysis_window_start = min(t.entry_at for t in trades)
            analysis_window_end = max(t.exit_at for t in trades)

        # Run all four sub-analyses
        regime_result = self._regime_analysis(trades, skipped, warnings)
        indicator_result = self._indicator_analysis(trades, warnings)
        parameter_result = self._parameter_analysis(trades, warnings)
        pair_result = self._pair_analysis(trades, skipped, warnings)

        # Regime concentration warning
        if n > 0:
            regime_counts: dict[str, int] = {}
            for t in trades:
                if t.regime_at_entry is not None:
                    regime_counts[t.regime_at_entry] = (
                        regime_counts.get(t.regime_at_entry, 0) + 1
                    )
            if regime_counts:
                dominant_regime = max(regime_counts, key=lambda k: regime_counts[k])
                dominant_pct = regime_counts[dominant_regime] / n
                if dominant_pct >= _REGIME_CONCENTRATION_PCT:
                    warnings.append(
                        f"Regime concentration: {dominant_pct:.0%} of trades in "
                        f"{dominant_regime}. Recommendations may not generalise to "
                        f"other market conditions."
                    )

        # Overall confidence is the minimum of all section confidences
        section_confidences = [
            regime_result.confidence,
            indicator_result.confidence,
            parameter_result.confidence,
            pair_result.confidence,
        ]
        overall_confidence = min(section_confidences) if section_confidences else 0.0
        # Hard floor: no confidence if below the minimum trade threshold
        if n < self._min_trades:
            overall_confidence = min(
                overall_confidence,
                _sample_factor(n, self._min_trades),
            )

        is_actionable = overall_confidence >= self.CONFIDENCE_THRESHOLD

        safeguards_applied: dict[str, int | float] = {
            "MIN_TRADES_FOR_ANALYSIS": self._min_trades,
            "MIN_TRADES_FOR_REGIME": self.MIN_TRADES_FOR_REGIME,
            "MIN_TRADES_FOR_INDICATOR": self.MIN_TRADES_FOR_INDICATOR,
            "MIN_TRADES_FOR_SHARPE": self.MIN_TRADES_FOR_SHARPE,
            "MIN_TRADES_FOR_PAIR_RANKING": self.MIN_TRADES_FOR_PAIR_RANKING,
            "CONFIDENCE_THRESHOLD": self.CONFIDENCE_THRESHOLD,
            "FALSE_SIGNAL_CONFIDENCE_THRESHOLD": self.FALSE_SIGNAL_CONFIDENCE_THRESHOLD,
            "REGIME_CONCENTRATION_PCT": _REGIME_CONCENTRATION_PCT,
        }

        self._log.info(
            "performance_analyzer.report_generated",
            total_trades=n,
            total_skipped=len(skipped),
            overall_confidence=round(overall_confidence, 4),
            is_actionable=is_actionable,
            warning_count=len(warnings),
        )

        return PerformanceReport(
            generated_at=datetime.now(tz=UTC),
            analysis_window_start=analysis_window_start,
            analysis_window_end=analysis_window_end,
            total_trades=n,
            total_skipped=len(skipped),
            overall_win_rate=overall_win_rate,
            overall_avg_pnl_pct=overall_avg_pnl_pct,
            overall_confidence=overall_confidence,
            is_actionable=is_actionable,
            regime=regime_result,
            indicators=indicator_result,
            parameters=parameter_result,
            pairs=pair_result,
            warnings=warnings,
            safeguards_applied=safeguards_applied,
        )

    # ------------------------------------------------------------------
    # Private analysis methods
    # ------------------------------------------------------------------

    def _regime_analysis(
        self,
        trades: list[TradeResult],
        skipped: list[SkippedTrade],
        warnings: list[str],
    ) -> RegimeAnalysis:
        """Group trades and skipped entries by regime label and compute stats."""
        # Group trades by regime
        regime_trades: dict[str, list[TradeResult]] = {}
        for t in trades:
            if t.regime_at_entry is not None:
                trade_bucket = regime_trades.setdefault(t.regime_at_entry, [])
                trade_bucket.append(t)

        # Group skipped by regime -- use a distinct variable name to avoid
        # mypy type-narrowing conflict with trade_bucket above.
        regime_skipped: dict[str, list[SkippedTrade]] = {}
        for s in skipped:
            if s.regime_at_skip is not None:
                skip_bucket = regime_skipped.setdefault(s.regime_at_skip, [])
                skip_bucket.append(s)

        # Build per-regime stats
        all_regimes = set(regime_trades.keys()) | set(regime_skipped.keys())
        stats: list[RegimeStats] = []
        qualifying: list[RegimeStats] = []  # trade_count >= MIN_TRADES_FOR_REGIME

        for regime in sorted(all_regimes):
            rtrades = regime_trades.get(regime, [])
            rskipped = regime_skipped.get(regime, [])
            tc = len(rtrades)
            win_count = sum(1 for t in rtrades if t.realised_pnl > Decimal(0))
            win_rate = win_count / tc if tc > 0 else 0.0
            return_pcts = [t.return_pct for t in rtrades]
            avg_pnl = _safe_mean(return_pcts)
            total_pnl = sum(return_pcts)

            skipped_would_profit = sum(
                1
                for s in rskipped
                if s.hypothetical_outcome_pct is not None
                and s.hypothetical_outcome_pct > 0
            )
            skipped_correctly = sum(
                1
                for s in rskipped
                if s.hypothetical_outcome_pct is not None
                and s.hypothetical_outcome_pct <= 0
            )
            skipped_unknown = sum(
                1 for s in rskipped if s.hypothetical_outcome_pct is None
            )

            rs = RegimeStats(
                regime=regime,
                trade_count=tc,
                win_count=win_count,
                win_rate=win_rate,
                avg_pnl_pct=avg_pnl,
                total_pnl_pct=total_pnl,
                skipped_would_profit=skipped_would_profit,
                skipped_correctly=skipped_correctly,
                skipped_unknown=skipped_unknown,
            )
            stats.append(rs)

            if tc < self.MIN_TRADES_FOR_REGIME:
                warnings.append(
                    f"Regime {regime}: only {tc} trades "
                    f"(minimum {self.MIN_TRADES_FOR_REGIME} for reliable win rate)"
                )
            else:
                qualifying.append(rs)

        best_regime: str | None = None
        worst_regime: str | None = None

        if qualifying:
            best_regime = max(qualifying, key=lambda r: r.win_rate).regime
            worst_regime = min(qualifying, key=lambda r: r.win_rate).regime
        elif all_regimes:
            warnings.append(
                "No regime has minimum "
                f"{self.MIN_TRADES_FOR_REGIME} trades for best/worst classification"
            )

        # Confidence based on the total number of labelled trades
        labelled_count = sum(len(v) for v in regime_trades.values())
        confidence = _sample_factor(labelled_count, self.MIN_TRADES_FOR_ANALYSIS)

        return RegimeAnalysis(
            by_regime=stats,
            best_regime=best_regime,
            worst_regime=worst_regime,
            confidence=confidence,
        )

    def _indicator_analysis(
        self,
        trades: list[TradeResult],
        warnings: list[str],
    ) -> IndicatorAnalysis:
        """
        Per-indicator Pearson / Spearman IC analysis.

        Numeric values in signal_context are extracted per indicator key.
        Parameter echo keys (rsi_period, oversold, overbought, close) are
        excluded.  Non-finite values are filtered out.
        """
        # Collect all indicator keys
        all_keys: set[str] = set()
        for t in trades:
            if t.signal_context is not None:
                for k, v in t.signal_context.items():
                    if k not in _EXCLUDED_INDICATOR_KEYS and isinstance(v, (int, float)):
                        all_keys.add(k)

        indicator_stats: list[IndicatorStats] = []

        for key in sorted(all_keys):
            indicator_values: list[float] = []
            return_pcts: list[float] = []
            false_signal_count = 0

            for t in trades:
                ctx = t.signal_context
                if ctx is None:
                    continue
                raw = ctx.get(key)
                if not isinstance(raw, (int, float)):
                    continue
                val = float(raw)
                if not math.isfinite(val):
                    continue
                pnl = t.return_pct
                if not math.isfinite(pnl):
                    continue
                indicator_values.append(val)
                return_pcts.append(pnl)

                # False signal detection
                if pnl < 0:
                    if key == "rsi":
                        if val < _RSI_STRONG_OVERSOLD or val > _RSI_STRONG_OVERBOUGHT:
                            false_signal_count += 1
                    elif key == "fear_greed_index":
                        if (
                            val <= _FGI_EXTREME_FEAR_MAX
                            or val >= _FGI_EXTREME_GREED_MIN
                        ):
                            false_signal_count += 1
                    else:
                        conf_raw = ctx.get("confidence")
                        if isinstance(conf_raw, (int, float)) and math.isfinite(
                            float(conf_raw)
                        ):
                            if float(conf_raw) > self.FALSE_SIGNAL_CONFIDENCE_THRESHOLD:
                                false_signal_count += 1

            sample_count = len(indicator_values)
            if sample_count < 3:
                corr = 0.0
                ic = 0.0
                warnings.append(
                    f"Indicator {key}: only {sample_count} samples "
                    f"(minimum 3 for any correlation)"
                )
            else:
                corr = _pearson(indicator_values, return_pcts)
                ic = _spearman(indicator_values, return_pcts)

            if sample_count < self.MIN_TRADES_FOR_INDICATOR:
                warnings.append(
                    f"Indicator {key}: only {sample_count} samples "
                    f"(minimum {self.MIN_TRADES_FOR_INDICATOR} for reliable IC)"
                )

            false_signal_rate = (
                false_signal_count / sample_count if sample_count > 0 else 0.0
            )

            indicator_stats.append(
                IndicatorStats(
                    indicator_name=key,
                    sample_count=sample_count,
                    correlation_with_pnl=corr,
                    information_coefficient=ic,
                    false_signal_count=false_signal_count,
                    false_signal_rate=false_signal_rate,
                )
            )

        # Identify most predictive and highest false signal rate
        # Only consider indicators with sufficient samples
        qualifying_indicators = [
            s
            for s in indicator_stats
            if s.sample_count >= self.MIN_TRADES_FOR_INDICATOR
        ]
        most_predictive: str | None = None
        highest_false_signal_rate: str | None = None

        if qualifying_indicators:
            most_predictive = max(
                qualifying_indicators,
                key=lambda s: abs(s.information_coefficient),
            ).indicator_name
            highest_false_signal_rate = max(
                qualifying_indicators,
                key=lambda s: s.false_signal_rate,
            ).indicator_name

        # Section confidence
        total_with_context = sum(
            1 for t in trades if t.signal_context is not None
        )
        confidence = _sample_factor(total_with_context, self.MIN_TRADES_FOR_ANALYSIS)

        return IndicatorAnalysis(
            by_indicator=indicator_stats,
            most_predictive=most_predictive,
            highest_false_signal_rate=highest_false_signal_rate,
            confidence=confidence,
        )

    def _parameter_analysis(
        self,
        trades: list[TradeResult],
        warnings: list[str],
    ) -> ParameterAnalysis:
        """
        RSI bucketing, exit reason rates, MFE/MAE analysis, and MFE beyond TP.
        """
        # --- RSI bucketing ---
        rsi_bucket_map: dict[float, list[TradeResult]] = {}
        for t in trades:
            ctx = t.signal_context
            if ctx is None:
                continue
            raw = ctx.get("rsi")
            if not isinstance(raw, (int, float)):
                continue
            val = float(raw)
            if (
                not math.isfinite(val)
                or val < self.RSI_BUCKET_MIN
                or val > self.RSI_BUCKET_MAX
            ):
                continue
            # Clamp RSI == 100.0 into the last bucket [95, 100)
            bucket_low = min(
                math.floor(val / self.RSI_BUCKET_WIDTH) * self.RSI_BUCKET_WIDTH,
                self.RSI_BUCKET_MAX - self.RSI_BUCKET_WIDTH,
            )
            rsi_bucket_map.setdefault(bucket_low, []).append(t)

        rsi_buckets: list[RSIBucketStats] = []
        for bucket_low in sorted(rsi_bucket_map.keys()):
            bucket_high = bucket_low + self.RSI_BUCKET_WIDTH
            bucket_trades = rsi_bucket_map[bucket_low]
            tc = len(bucket_trades)
            return_pcts = [t.return_pct for t in bucket_trades]
            win_rate = (
                sum(1 for p in return_pcts if p > 0) / tc if tc > 0 else 0.0
            )
            avg_pnl = _safe_mean(return_pcts)
            sharpe = 0.0
            if tc < self.MIN_TRADES_FOR_SHARPE:
                warnings.append(
                    f"RSI bucket {bucket_low:.0f}-{bucket_high:.0f}: "
                    f"only {tc} trades (Sharpe unreliable)"
                )
            else:
                sharpe = _per_trade_sharpe(return_pcts)
            label = f"{bucket_low:.0f}-{bucket_high:.0f}"
            rsi_buckets.append(
                RSIBucketStats(
                    bucket_label=label,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    trade_count=tc,
                    win_rate=win_rate,
                    avg_pnl_pct=avg_pnl,
                    sharpe=sharpe,
                )
            )

        # --- Exit reason rates ---
        trades_with_exit = [t for t in trades if t.exit_reason is not None]
        total_with_reason = len(trades_with_exit)
        stop_loss_hit_rate: float
        take_profit_hit_rate: float
        trailing_stop_hit_rate: float
        signal_exit_rate: float
        if total_with_reason == 0:
            stop_loss_hit_rate = 0.0
            take_profit_hit_rate = 0.0
            trailing_stop_hit_rate = 0.0
            signal_exit_rate = 0.0
            if trades:
                warnings.append(
                    "No exit reason data available -- parameter analysis limited"
                )
        else:
            stop_loss_hit_rate = (
                sum(1 for t in trades_with_exit if t.exit_reason == "stop_loss")
                / total_with_reason
            )
            take_profit_hit_rate = (
                sum(1 for t in trades_with_exit if t.exit_reason == "take_profit")
                / total_with_reason
            )
            trailing_stop_hit_rate = (
                sum(1 for t in trades_with_exit if t.exit_reason == "trailing_stop")
                / total_with_reason
            )
            signal_exit_rate = (
                sum(1 for t in trades_with_exit if t.exit_reason == "signal_exit")
                / total_with_reason
            )

        # --- MFE/MAE analysis ---
        winners = [t for t in trades if t.realised_pnl > Decimal(0)]
        losers = [t for t in trades if t.realised_pnl < Decimal(0)]  # CR-003: break-even is not a loss

        winners_mfe: list[float] = [
            t.mfe_pct for t in winners if t.mfe_pct is not None
        ]
        losers_mae: list[float] = [
            t.mae_pct for t in losers if t.mae_pct is not None
        ]
        all_mfe: list[float] = [t.mfe_pct for t in trades if t.mfe_pct is not None]
        all_mae: list[float] = [t.mae_pct for t in trades if t.mae_pct is not None]

        avg_mfe_winners = _safe_mean(winners_mfe)
        avg_mae_losers = _safe_mean(losers_mae)
        avg_mfe_all = _safe_mean(all_mfe)
        avg_mae_all = _safe_mean(all_mae)

        if not all_mfe and not all_mae and trades:
            warnings.append(
                "No MAE/MFE data available -- excursion analysis skipped"
            )

        # --- MFE beyond take-profit threshold ---
        # Infer TP threshold from the average return of TP-exit trades.
        # Fall back to 2% if no TP exits exist.
        tp_trades = [t for t in trades_with_exit if t.exit_reason == "take_profit"]
        tp_threshold: float
        if tp_trades:
            tp_threshold = _safe_mean([t.return_pct for t in tp_trades])
        else:
            tp_threshold = 0.02

        mfe_beyond_tp_count = sum(1 for v in all_mfe if v > tp_threshold)
        mfe_beyond_tp_rate = (
            mfe_beyond_tp_count / len(all_mfe) if all_mfe else 0.0
        )

        # Section confidence
        confidence = _sample_factor(len(trades), self.MIN_TRADES_FOR_ANALYSIS)

        return ParameterAnalysis(
            rsi_buckets=rsi_buckets,
            stop_loss_hit_rate=stop_loss_hit_rate,
            take_profit_hit_rate=take_profit_hit_rate,
            trailing_stop_hit_rate=trailing_stop_hit_rate,
            signal_exit_rate=signal_exit_rate,
            avg_mfe_winners=avg_mfe_winners,
            avg_mae_losers=avg_mae_losers,
            avg_mfe_all=avg_mfe_all,
            avg_mae_all=avg_mae_all,
            mfe_beyond_tp_count=mfe_beyond_tp_count,
            mfe_beyond_tp_rate=mfe_beyond_tp_rate,
            confidence=confidence,
        )

    def _pair_analysis(
        self,
        trades: list[TradeResult],
        skipped: list[SkippedTrade],
        warnings: list[str],
    ) -> PairAnalysis:
        """Group trades by symbol and compute per-pair statistics."""
        # Group trades by symbol
        symbol_trades: dict[str, list[TradeResult]] = {}
        for t in trades:
            symbol_trades.setdefault(t.symbol, []).append(t)

        # Group skipped by symbol (count only)
        symbol_skipped: dict[str, int] = {}
        for s in skipped:
            symbol_skipped[s.symbol] = symbol_skipped.get(s.symbol, 0) + 1

        all_symbols = set(symbol_trades.keys()) | set(symbol_skipped.keys())
        pair_stats: list[PairStats] = []

        for symbol in sorted(all_symbols):
            strades = symbol_trades.get(symbol, [])
            tc = len(strades)
            skip_count = symbol_skipped.get(symbol, 0)

            win_rate = 0.0
            avg_pnl_pct = 0.0
            total_pnl_pct = 0.0
            sharpe = 0.0

            if tc > 0:
                return_pcts = [t.return_pct for t in strades]
                win_count = sum(1 for p in return_pcts if p > 0)
                win_rate = win_count / tc
                avg_pnl_pct = _safe_mean(return_pcts)
                total_pnl_pct = sum(return_pcts)
                sharpe = _per_trade_sharpe(return_pcts)

            # Per-symbol best/worst regime (min 3 trades per sub-group)
            sub_regime: dict[str, list[float]] = {}
            for t in strades:
                if t.regime_at_entry is not None:
                    sub_regime.setdefault(t.regime_at_entry, []).append(t.return_pct)

            qualifying_sub = {
                r: pcts for r, pcts in sub_regime.items() if len(pcts) >= 3
            }
            best_regime: str | None = None
            worst_regime: str | None = None
            if qualifying_sub:
                sub_win_rates = {
                    r: sum(1 for p in pcts if p > 0) / len(pcts)
                    for r, pcts in qualifying_sub.items()
                }
                best_regime = max(sub_win_rates, key=lambda k: sub_win_rates[k])
                worst_regime = min(sub_win_rates, key=lambda k: sub_win_rates[k])

            pair_stats.append(
                PairStats(
                    symbol=symbol,
                    trade_count=tc,
                    win_rate=win_rate,
                    avg_pnl_pct=avg_pnl_pct,
                    total_pnl_pct=total_pnl_pct,
                    sharpe=sharpe,
                    best_regime=best_regime,
                    worst_regime=worst_regime,
                    skip_count=skip_count,
                )
            )

        # Sort descending by total_pnl_pct
        pair_stats.sort(key=lambda p: p.total_pnl_pct, reverse=True)

        qualifying_pairs = [
            p for p in pair_stats if p.trade_count >= self.MIN_TRADES_FOR_PAIR_RANKING
        ]
        best_symbol: str | None = None
        worst_symbol: str | None = None

        if qualifying_pairs:
            best_symbol = qualifying_pairs[0].symbol
            worst_symbol = qualifying_pairs[-1].symbol
        elif all_symbols:
            warnings.append(
                f"No pair has minimum {self.MIN_TRADES_FOR_PAIR_RANKING} trades "
                f"for ranking"
            )

        # Section confidence
        confidence = _sample_factor(len(trades), self.MIN_TRADES_FOR_ANALYSIS)

        return PairAnalysis(
            by_symbol=pair_stats,
            best_symbol=best_symbol,
            worst_symbol=worst_symbol,
            confidence=confidence,
        )
