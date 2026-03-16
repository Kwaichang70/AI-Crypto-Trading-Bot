"""
tests/unit/test_performance_analyzer.py
-----------------------------------------
Unit tests for PerformanceAnalyzer and its sub-modules (Sprint 33).

Modules under test
------------------
packages/trading/performance_analyzer.py

Test coverage
-------------
TestStatisticalHelpers (8 tests)
- _pearson perfect positive / negative / near-zero / empty
- _rank_data with ties
- _spearman on monotonic input
- _safe_mean empty
- _per_trade_sharpe with zero std

TestRegimeAnalysis (6 tests)
- single regime all winners
- mixed regimes per-regime win rates
- skipped trade evaluation
- no trades returns empty
- None regime grouped as UNKNOWN (excluded from by_regime, not grouped)
- confidence scales with trade count

TestIndicatorAnalysis (5 tests)
- RSI vs PnL negative correlation
- no signal_context returns empty
- false signal detection
- single indicator produces one IndicatorStats entry
- fewer than MIN_TRADES_FOR_INDICATOR → IC = 0.0

TestParameterAnalysis (6 tests)
- RSI value bucketing
- exit reason rates
- MFE winners average
- MAE losers average
- no RSI key in context → empty rsi_buckets
- RSI=100 clamped to last bucket (95-100)

TestPairAnalysis (5 tests)
- single pair produces one PairStats entry
- multiple pairs produce correct per-pair stats
- best_regime per pair uses highest win-rate regime
- empty trades → empty by_symbol
- total_pnl_pct is sum of individual return_pcts

TestPerformanceReport (5 tests)
- full analyze with 50 trades + 10 skipped populates all sections
- below min_trades emits insufficient data warning
- overall_confidence is min of section confidences
- empty input report with zeros and warnings
- generated_at is within last minute
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from common.types import OrderSide
from trading.models import SkippedTrade, TradeResult
from trading.performance_analyzer import (
    PerformanceAnalyzer,
    _pearson,
    _per_trade_sharpe,
    _rank_data,
    _safe_mean,
    _spearman,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_BASE_ENTRY_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_BASE_EXIT_AT = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)


def _make_trade(
    symbol: str = "BTC/USD",
    entry_price: float = 100.0,
    exit_price: float = 105.0,
    quantity: float = 1.0,
    regime_at_entry: str | None = "NEUTRAL",
    exit_reason: str | None = "signal_exit",
    mae_pct: float | None = -0.02,
    mfe_pct: float | None = 0.05,
    signal_context: dict | None = None,
    strategy_id: str = "rsi",
    run_id: str = "run-1",
) -> TradeResult:
    """Construct a TradeResult with sensible defaults."""
    ep = Decimal(str(entry_price))
    xp = Decimal(str(exit_price))
    qty = Decimal(str(quantity))
    gross = (xp - ep) * qty
    fees = Decimal("0.001") * ep * qty
    realised = gross - fees
    return TradeResult(
        run_id=run_id,
        symbol=symbol,
        side=OrderSide.BUY,
        entry_price=ep,
        exit_price=xp,
        quantity=qty,
        realised_pnl=realised,
        total_fees=fees,
        entry_at=_BASE_ENTRY_AT,
        exit_at=_BASE_EXIT_AT,
        strategy_id=strategy_id,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        exit_reason=exit_reason,
        regime_at_entry=regime_at_entry,
        signal_context=signal_context,
    )


def _make_skipped(
    symbol: str = "BTC/USD",
    skip_reason: str = "regime_risk_off",
    regime_at_skip: str | None = "RISK_OFF",
    hypothetical_outcome_pct: float | None = None,
    hypothetical_entry_price: float = 100.0,
    run_id: str = "run-1",
) -> SkippedTrade:
    """Construct a SkippedTrade with sensible defaults."""
    return SkippedTrade(
        run_id=run_id,
        symbol=symbol,
        skip_reason=skip_reason,
        regime_at_skip=regime_at_skip,
        hypothetical_outcome_pct=hypothetical_outcome_pct,
        hypothetical_entry_price=Decimal(str(hypothetical_entry_price)),
    )


# ---------------------------------------------------------------------------
# TestStatisticalHelpers
# ---------------------------------------------------------------------------


class TestStatisticalHelpers:
    """Tests for module-level statistical helper functions."""

    def test_pearson_perfect_positive(self) -> None:
        result = _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert math.isclose(result, 1.0, abs_tol=1e-9)

    def test_pearson_perfect_negative(self) -> None:
        result = _pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
        assert math.isclose(result, -1.0, abs_tol=1e-9)

    def test_pearson_no_correlation(self) -> None:
        # Alternating pattern — low absolute correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 4.0, 1.0, 5.0]
        result = _pearson(x, y)
        assert abs(result) < 0.8  # not strongly correlated

    def test_pearson_empty(self) -> None:
        assert _pearson([], []) == 0.0

    def test_rank_with_ties(self) -> None:
        # [10, 20, 20, 30] → [1.0, 2.5, 2.5, 4.0]
        result = _rank_data([10.0, 20.0, 20.0, 30.0])
        assert result == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_monotonic(self) -> None:
        # Perfectly monotonic increasing → spearman = 1.0
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _spearman(x, y)
        assert math.isclose(result, 1.0, abs_tol=1e-9)

    def test_safe_mean_empty(self) -> None:
        assert _safe_mean([]) == 0.0

    def test_safe_sharpe_zero_std(self) -> None:
        # All same values → std = 0 → Sharpe = 0
        result = _per_trade_sharpe([1.0, 1.0, 1.0, 1.0])
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestRegimeAnalysis
# ---------------------------------------------------------------------------


class TestRegimeAnalysis:
    """Tests for the _regime_analysis sub-module via PerformanceAnalyzer."""

    def test_single_regime_all_winners(self) -> None:
        # 10 profitable trades in NEUTRAL → win_rate = 1.0
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [
            _make_trade(regime_at_entry="NEUTRAL", entry_price=100, exit_price=110)
            for _ in range(10)
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        assert len(report.regime.by_regime) == 1
        stats = report.regime.by_regime[0]
        assert stats.regime == "NEUTRAL"
        assert stats.win_rate == 1.0
        assert stats.trade_count == 10

    def test_mixed_regimes(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        # 6 FEAR trades: all winners (exit > entry)
        fear_trades = [
            _make_trade(regime_at_entry="FEAR", entry_price=100, exit_price=110)
            for _ in range(6)
        ]
        # 6 GREED trades: all losers (exit < entry)
        greed_trades = [
            _make_trade(regime_at_entry="GREED", entry_price=100, exit_price=90)
            for _ in range(6)
        ]
        report = analyzer.analyze(trades=fear_trades + greed_trades, skipped=[])
        by_regime = {s.regime: s for s in report.regime.by_regime}
        assert "FEAR" in by_regime
        assert "GREED" in by_regime
        assert by_regime["FEAR"].win_rate == 1.0
        assert by_regime["GREED"].win_rate == 0.0

    def test_skipped_trade_evaluation(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        skipped = [
            # 3 that would have profited
            _make_skipped(regime_at_skip="RISK_OFF", hypothetical_outcome_pct=0.03),
            _make_skipped(regime_at_skip="RISK_OFF", hypothetical_outcome_pct=0.01),
            _make_skipped(regime_at_skip="RISK_OFF", hypothetical_outcome_pct=0.05),
            # 2 correctly skipped (would have lost)
            _make_skipped(regime_at_skip="RISK_OFF", hypothetical_outcome_pct=-0.02),
            _make_skipped(regime_at_skip="RISK_OFF", hypothetical_outcome_pct=0.0),
        ]
        report = analyzer.analyze(trades=[], skipped=skipped)
        by_regime = {s.regime: s for s in report.regime.by_regime}
        assert "RISK_OFF" in by_regime
        stats = by_regime["RISK_OFF"]
        assert stats.skipped_would_profit == 3
        assert stats.skipped_correctly == 2
        assert stats.skipped_unknown == 0

    def test_no_trades_returns_empty(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        report = analyzer.analyze(trades=[], skipped=[])
        assert report.regime.by_regime == []
        assert report.regime.best_regime is None
        assert report.regime.worst_regime is None

    def test_regime_none_excluded_from_by_regime(self) -> None:
        # Trades with regime_at_entry=None should NOT appear in by_regime
        analyzer = PerformanceAnalyzer(min_trades=3)
        trades = [
            _make_trade(regime_at_entry=None, entry_price=100, exit_price=110)
            for _ in range(5)
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        # None regimes are excluded from grouping (not grouped as "UNKNOWN")
        assert report.regime.by_regime == []

    def test_confidence_scales_with_count(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        # 5 trades → well below min → low confidence
        few_trades = [
            _make_trade(regime_at_entry="NEUTRAL") for _ in range(5)
        ]
        report_few = analyzer.analyze(trades=few_trades, skipped=[])
        # 30 trades → at min threshold → higher confidence
        many_trades = [
            _make_trade(regime_at_entry="NEUTRAL") for _ in range(30)
        ]
        report_many = analyzer.analyze(trades=many_trades, skipped=[])
        assert report_few.regime.confidence < report_many.regime.confidence


# ---------------------------------------------------------------------------
# TestIndicatorAnalysis
# ---------------------------------------------------------------------------


class TestIndicatorAnalysis:
    """Tests for the _indicator_analysis sub-module."""

    def test_rsi_correlation_with_pnl_negative(self) -> None:
        # High RSI → negative PnL (mean reversion: high RSI = overbought = sell)
        # RSI: 70, 75, 80, 72, 78  PnL: losing
        # RSI: 20, 25, 15, 22, 18  PnL: winning
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = []
        high_rsi_values = [70.0, 75.0, 80.0, 72.0, 78.0]
        low_rsi_values = [20.0, 25.0, 15.0, 22.0, 18.0]
        for rsi in high_rsi_values:
            trades.append(_make_trade(
                entry_price=100, exit_price=95,  # loser
                signal_context={"rsi": rsi},
            ))
        for rsi in low_rsi_values:
            trades.append(_make_trade(
                entry_price=100, exit_price=108,  # winner
                signal_context={"rsi": rsi},
            ))
        report = analyzer.analyze(trades=trades, skipped=[])
        rsi_stats = next(
            (s for s in report.indicators.by_indicator if s.indicator_name == "rsi"),
            None,
        )
        assert rsi_stats is not None
        # Higher RSI correlates with worse PnL → negative correlation
        assert rsi_stats.correlation_with_pnl < 0

    def test_no_signal_context_returns_empty(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [_make_trade(signal_context=None) for _ in range(10)]
        report = analyzer.analyze(trades=trades, skipped=[])
        assert report.indicators.by_indicator == []

    def test_false_signal_detection(self) -> None:
        # Trades with high confidence but negative PnL trigger false signal counting
        # The generic path: conf > FALSE_SIGNAL_CONFIDENCE_THRESHOLD and pnl < 0
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = []
        for _ in range(5):
            trades.append(_make_trade(
                entry_price=100,
                exit_price=95,  # loser
                signal_context={"my_signal": 0.9, "confidence": 0.8},
            ))
        # Add a few winners to meet sample threshold
        for _ in range(6):
            trades.append(_make_trade(
                entry_price=100,
                exit_price=110,
                signal_context={"my_signal": 0.2, "confidence": 0.3},
            ))
        report = analyzer.analyze(trades=trades, skipped=[])
        signal_stats = next(
            (s for s in report.indicators.by_indicator if s.indicator_name == "my_signal"),
            None,
        )
        assert signal_stats is not None
        assert signal_stats.false_signal_count > 0

    def test_single_indicator_one_entry(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [
            _make_trade(signal_context={"rsi": float(i * 5 + 20)})
            for i in range(12)
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        rsi_entries = [
            s for s in report.indicators.by_indicator if s.indicator_name == "rsi"
        ]
        assert len(rsi_entries) == 1

    def test_min_trades_guard_ic_zero(self) -> None:
        # Fewer than MIN_TRADES_FOR_INDICATOR (10) trades → IC uses 0.0 if <3 samples
        # With exactly 2 samples, _pearson / _spearman return 0.0 (n < 3 guard)
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [
            _make_trade(signal_context={"rsi": 30.0}),
            _make_trade(signal_context={"rsi": 70.0}),
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        rsi_stats = next(
            (s for s in report.indicators.by_indicator if s.indicator_name == "rsi"),
            None,
        )
        assert rsi_stats is not None
        assert rsi_stats.information_coefficient == 0.0


# ---------------------------------------------------------------------------
# TestParameterAnalysis
# ---------------------------------------------------------------------------


class TestParameterAnalysis:
    """Tests for the _parameter_analysis sub-module."""

    def test_rsi_bucketing(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        # RSI values spanning multiple 5-wide buckets
        rsi_values = [25.0, 28.0, 35.0, 72.0]
        trades = [
            _make_trade(signal_context={"rsi": v}) for v in rsi_values
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        bucket_labels = {b.bucket_label for b in report.parameters.rsi_buckets}
        assert "25-30" in bucket_labels
        assert "35-40" in bucket_labels
        assert "70-75" in bucket_labels

    def test_exit_reason_rates(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = (
            [_make_trade(exit_reason="stop_loss") for _ in range(3)]
            + [_make_trade(exit_reason="take_profit") for _ in range(2)]
            + [_make_trade(exit_reason="signal_exit") for _ in range(5)]
        )
        report = analyzer.analyze(trades=trades, skipped=[])
        pa = report.parameters
        assert math.isclose(pa.stop_loss_hit_rate, 3 / 10, abs_tol=1e-9)
        assert math.isclose(pa.take_profit_hit_rate, 2 / 10, abs_tol=1e-9)
        assert math.isclose(pa.signal_exit_rate, 5 / 10, abs_tol=1e-9)

    def test_mfe_winners_avg(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        # Winners with known MFE values
        trades = [
            _make_trade(entry_price=100, exit_price=110, mfe_pct=0.10),
            _make_trade(entry_price=100, exit_price=110, mfe_pct=0.20),
            _make_trade(entry_price=100, exit_price=110, mfe_pct=0.30),
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        expected_avg = (0.10 + 0.20 + 0.30) / 3
        assert math.isclose(report.parameters.avg_mfe_winners, expected_avg, abs_tol=1e-9)

    def test_mae_losers_avg(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        # Losers with known MAE values (negative)
        trades = [
            _make_trade(entry_price=100, exit_price=90, mae_pct=-0.10),
            _make_trade(entry_price=100, exit_price=90, mae_pct=-0.20),
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        expected_avg = (-0.10 + -0.20) / 2
        assert math.isclose(report.parameters.avg_mae_losers, expected_avg, abs_tol=1e-9)

    def test_no_rsi_in_context_empty_buckets(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [
            _make_trade(signal_context={"macd": 1.5}) for _ in range(5)
        ]
        report = analyzer.analyze(trades=trades, skipped=[])
        assert report.parameters.rsi_buckets == []

    def test_rsi_100_clamped_to_last_bucket(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [_make_trade(signal_context={"rsi": 100.0})]
        report = analyzer.analyze(trades=trades, skipped=[])
        assert len(report.parameters.rsi_buckets) == 1
        bucket = report.parameters.rsi_buckets[0]
        # Clamped to [95, 100) bucket
        assert bucket.bucket_low == 95.0
        assert bucket.bucket_high == 100.0


# ---------------------------------------------------------------------------
# TestPairAnalysis
# ---------------------------------------------------------------------------


class TestPairAnalysis:
    """Tests for the _pair_analysis sub-module."""

    def test_single_pair(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [_make_trade(symbol="BTC/USD") for _ in range(10)]
        report = analyzer.analyze(trades=trades, skipped=[])
        assert len(report.pairs.by_symbol) == 1
        assert report.pairs.by_symbol[0].symbol == "BTC/USD"
        assert report.pairs.by_symbol[0].trade_count == 10

    def test_multiple_pairs(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        btc_trades = [
            _make_trade(symbol="BTC/USD", entry_price=100, exit_price=110)
            for _ in range(6)
        ]
        eth_trades = [
            _make_trade(symbol="ETH/USD", entry_price=50, exit_price=45)
            for _ in range(6)
        ]
        report = analyzer.analyze(trades=btc_trades + eth_trades, skipped=[])
        symbols = {p.symbol: p for p in report.pairs.by_symbol}
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols
        assert symbols["BTC/USD"].win_rate == 1.0
        assert symbols["ETH/USD"].win_rate == 0.0

    def test_best_regime_per_pair(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        # BTC: 4 trades in NEUTRAL (winners) + 3 in FEAR (losers)
        trades = (
            [
                _make_trade(
                    symbol="BTC/USD",
                    entry_price=100,
                    exit_price=110,
                    regime_at_entry="NEUTRAL",
                )
                for _ in range(4)
            ]
            + [
                _make_trade(
                    symbol="BTC/USD",
                    entry_price=100,
                    exit_price=90,
                    regime_at_entry="FEAR",
                )
                for _ in range(3)
            ]
        )
        report = analyzer.analyze(trades=trades, skipped=[])
        btc = next(p for p in report.pairs.by_symbol if p.symbol == "BTC/USD")
        assert btc.best_regime == "NEUTRAL"
        assert btc.worst_regime == "FEAR"

    def test_empty_trades_empty_by_symbol(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        report = analyzer.analyze(trades=[], skipped=[])
        assert report.pairs.by_symbol == []
        assert report.pairs.best_symbol is None

    def test_pair_total_pnl_is_sum_of_returns(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=5)
        trades = [
            _make_trade(symbol="BTC/USD", entry_price=100, exit_price=105)
            for _ in range(5)
        ]
        expected_total = sum(t.return_pct for t in trades)
        report = analyzer.analyze(trades=trades, skipped=[])
        btc = next(p for p in report.pairs.by_symbol if p.symbol == "BTC/USD")
        assert math.isclose(btc.total_pnl_pct, expected_total, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# TestPerformanceReport
# ---------------------------------------------------------------------------


class TestPerformanceReport:
    """End-to-end tests for the full PerformanceAnalyzer.analyze() output."""

    def _make_diverse_trades(self, n: int = 50) -> list[TradeResult]:
        """Generate n trades across two symbols and two regimes."""
        trades = []
        for i in range(n):
            symbol = "BTC/USD" if i % 2 == 0 else "ETH/USD"
            regime = "NEUTRAL" if i % 3 != 0 else "FEAR"
            # Alternate winners and losers
            exit_price = 110.0 if i % 3 != 2 else 90.0
            rsi_val = 30.0 + (i % 40)
            trades.append(
                _make_trade(
                    symbol=symbol,
                    entry_price=100.0,
                    exit_price=exit_price,
                    regime_at_entry=regime,
                    exit_reason="signal_exit" if i % 4 != 0 else "stop_loss",
                    mae_pct=-0.02,
                    mfe_pct=0.05,
                    signal_context={"rsi": rsi_val},
                )
            )
        return trades

    def test_full_analyze_populates_all_sections(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        trades = self._make_diverse_trades(50)
        skipped = [
            _make_skipped(hypothetical_outcome_pct=0.02 if i % 2 == 0 else -0.01)
            for i in range(10)
        ]
        report = analyzer.analyze(trades=trades, skipped=skipped)

        assert report.total_trades == 50
        assert report.total_skipped == 10
        assert len(report.regime.by_regime) > 0
        assert len(report.parameters.rsi_buckets) > 0
        assert len(report.pairs.by_symbol) > 0

    def test_below_min_trades_emits_warning(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        trades = [_make_trade() for _ in range(5)]
        report = analyzer.analyze(trades=trades, skipped=[])
        # At least one warning about insufficient data
        insufficient_warnings = [
            w for w in report.warnings if "minimum" in w.lower() or "insufficient" in w.lower()
        ]
        assert len(insufficient_warnings) > 0

    def test_overall_confidence_is_min_of_sections(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        trades = self._make_diverse_trades(50)
        report = analyzer.analyze(trades=trades, skipped=[])
        section_confidences = [
            report.regime.confidence,
            report.indicators.confidence,
            report.parameters.confidence,
            report.pairs.confidence,
        ]
        assert math.isclose(
            report.overall_confidence, min(section_confidences), abs_tol=1e-9
        )

    def test_empty_input_report_with_zeros(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        report = analyzer.analyze(trades=[], skipped=[])
        assert report.total_trades == 0
        assert report.total_skipped == 0
        assert report.overall_win_rate == 0.0
        assert report.overall_avg_pnl_pct == 0.0
        assert report.is_actionable is False
        assert len(report.warnings) > 0

    def test_generated_at_is_recent(self) -> None:
        analyzer = PerformanceAnalyzer(min_trades=30)
        before = datetime.now(tz=UTC)
        report = analyzer.analyze(trades=[], skipped=[])
        after = datetime.now(tz=UTC)
        assert before <= report.generated_at <= after + timedelta(seconds=1)
