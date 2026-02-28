"""
tests/unit/test_backtest.py
----------------------------
Unit tests for the BacktestRunner module.

Module under test
-----------------
    packages/trading/backtest.py -- BacktestRunner high-level API.

Coverage groups
---------------
1.  TestBacktestRunnerInit       -- Constructor validation (5 tests)
2.  TestBacktestBarValidation    -- _validate_bars() error paths (4 tests)
3.  TestBacktestRunNoSignals     -- End-to-end run with a no-signal strategy (5 tests)
4.  TestBacktestDeterminism      -- Seed-controlled reproducibility (2 tests)
5.  TestBacktestMultiSymbol      -- Multi-symbol run (1 test)
6.  TestBacktestMetrics          -- Metrics sanity for zero-trade runs (3 tests)

Design notes
------------
- All tests use the ``make_bars(n, seed=42)`` factory from ``tests/conftest.py``
  for deterministic OHLCV data.
- The ``_AlwaysHoldStrategy`` stub never emits signals, isolating infrastructure
  behaviour from trading logic.
- No internal components are mocked.  BacktestRunner constructs PaperExecutionEngine,
  PortfolioAccounting, DefaultRiskManager, and StrategyEngine internally; tests
  verify the system end-to-end.
- ``asyncio_mode = "auto"`` is configured in ``pyproject.toml``, so
  ``@pytest.mark.asyncio`` is included for explicitness but not strictly required.
- Warmup: BacktestRunner computes warmup as max(strategy.min_bars_required * 2, 50).
  _AlwaysHoldStrategy.min_bars_required == 0, so warmup == 50.
  Tests requiring a successful run must supply > 50 bars (200 bars is used throughout).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import BacktestResult
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

# Import the shared bar factory from the root conftest.
# pytest resolves conftest fixtures automatically, but ``make_bars`` is a
# plain function (not a fixture), so we import it directly.
# pytest adds ``tests/`` to sys.path via testpaths in pyproject.toml,
# so conftest is importable without the package prefix.
from conftest import make_bars


# ---------------------------------------------------------------------------
# Test strategy stubs
# ---------------------------------------------------------------------------


class _AlwaysHoldStrategy(BaseStrategy):
    """
    Minimal strategy that never generates signals.

    Used for pure infrastructure tests where we care about the backtest
    plumbing (metrics, equity curve, field population) rather than trading
    behaviour.
    """

    metadata = StrategyMetadata(
        name="always_hold",
        version="1.0.0",
        description="Test strategy that always holds",
    )

    def on_bar(self, bars: Sequence[OHLCVBar]) -> list[Signal]:
        return []

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Module-level bar factory helpers
# ---------------------------------------------------------------------------


def _make_btc_bars(n: int = 200, seed: int = 42) -> list[OHLCVBar]:
    """Deterministic BTC/USDT bars using the shared conftest factory."""
    return make_bars(n, seed=seed, symbol="BTC/USDT")


def _make_eth_bars(n: int = 200, seed: int = 99) -> list[OHLCVBar]:
    """Deterministic ETH/USDT bars (different seed for independence)."""
    return make_bars(n, seed=seed, symbol="ETH/USDT")


def _make_runner(
    *,
    strategies: list[BaseStrategy] | None = None,
    symbols: list[str] | None = None,
    initial_capital: Decimal = Decimal("10000"),
    seed: int | None = 42,
    slippage_bps: int = 5,
    maker_fee_bps: int = 10,
    taker_fee_bps: int = 15,
) -> BacktestRunner:
    """
    Convenience factory for BacktestRunner with sane test defaults.

    By default creates a runner with a single _AlwaysHoldStrategy against
    BTC/USDT on the ONE_HOUR timeframe, seeded at 42.
    """
    if strategies is None:
        strategies = [_AlwaysHoldStrategy(strategy_id="test-hold")]
    if symbols is None:
        symbols = ["BTC/USDT"]
    return BacktestRunner(
        strategies=strategies,
        symbols=symbols,
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=initial_capital,
        slippage_bps=slippage_bps,
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        seed=seed,
    )


# ===========================================================================
# 1. Constructor validation
# ===========================================================================


class TestBacktestRunnerInit:
    """BacktestRunner.__init__ must reject degenerate inputs immediately."""

    def test_empty_strategies_raises_value_error(self) -> None:
        """Providing an empty strategies list must raise ValueError."""
        with pytest.raises(ValueError, match="strategy"):
            BacktestRunner(
                strategies=[],
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
            )

    def test_empty_symbols_raises_value_error(self) -> None:
        """Providing an empty symbols list must raise ValueError."""
        with pytest.raises(ValueError, match="symbol"):
            BacktestRunner(
                strategies=[_AlwaysHoldStrategy(strategy_id="s1")],
                symbols=[],
                timeframe=TimeFrame.ONE_HOUR,
            )

    def test_zero_initial_capital_raises_value_error(self) -> None:
        """initial_capital == 0 must raise ValueError."""
        with pytest.raises(ValueError, match="initial_capital"):
            BacktestRunner(
                strategies=[_AlwaysHoldStrategy(strategy_id="s1")],
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=Decimal("0"),
            )

    def test_negative_initial_capital_raises_value_error(self) -> None:
        """Negative initial_capital must raise ValueError."""
        with pytest.raises(ValueError, match="initial_capital"):
            BacktestRunner(
                strategies=[_AlwaysHoldStrategy(strategy_id="s1")],
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=Decimal("-500"),
            )

    def test_valid_construction_with_defaults(self) -> None:
        """Valid construction with default parameters must succeed."""
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="s1")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
        )
        assert runner is not None


# ===========================================================================
# 2. Bar validation (_validate_bars)
# ===========================================================================


class TestBacktestBarValidation:
    """
    _validate_bars() is called inside run() before any computation.
    All invalid bar inputs must raise ValueError before the engine starts.
    """

    @pytest.mark.asyncio
    async def test_missing_symbol_in_bars_raises_value_error(self) -> None:
        """
        Runner configured for BTC/USDT but bars_by_symbol contains only
        ETH/USDT must raise ValueError mentioning the missing symbol.
        """
        runner = _make_runner(symbols=["BTC/USDT"])
        # Wrong key: BTC/USDT absent, only ETH/USDT present
        bars = {"ETH/USDT": _make_btc_bars()}

        with pytest.raises(ValueError, match="BTC/USDT"):
            await runner.run(bars)

    @pytest.mark.asyncio
    async def test_empty_bar_list_raises_value_error(self) -> None:
        """
        Passing an empty list for a configured symbol must raise ValueError.

        The check must fire even though the key is present in the dict.
        """
        runner = _make_runner(symbols=["BTC/USDT"])
        bars: dict[str, list[OHLCVBar]] = {"BTC/USDT": []}

        with pytest.raises(ValueError, match="Empty bar list"):
            await runner.run(bars)

    @pytest.mark.asyncio
    async def test_insufficient_bars_for_warmup_raises_value_error(self) -> None:
        """
        Providing exactly ``warmup_bars`` bars (== 50 for _AlwaysHoldStrategy)
        must raise ValueError.  Validation requires strictly more than
        ``warmup_bars`` bars to allow at least one active trading bar.

        Providing 50 bars against a warmup of 50 fails because
        ``min_bars <= self._warmup_bars`` (50 <= 50) triggers the check.
        """
        runner = _make_runner()
        # Exactly warmup_bars bars — not enough (warmup == 50).
        bars = {"BTC/USDT": _make_btc_bars(n=50)}

        with pytest.raises(ValueError, match="[Ii]nsufficient bars"):
            await runner.run(bars)

    @pytest.mark.asyncio
    async def test_valid_bars_pass_validation(self) -> None:
        """
        200 well-formed, ascending-timestamp bars must pass all validation
        checks without raising.  The run itself should complete successfully.
        """
        runner = _make_runner()
        bars = {"BTC/USDT": _make_btc_bars(n=200)}
        # Should not raise
        result = await runner.run(bars)
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_reverse_sorted_bars_raises_value_error(self) -> None:
        """Bars with reverse-sorted timestamps must raise ValueError (look-ahead bias guard)."""
        runner = _make_runner(symbols=["BTC/USDT"])
        bars = list(reversed(_make_btc_bars(n=200)))
        with pytest.raises(ValueError, match=r"(?i)sort"):
            await runner.run({"BTC/USDT": bars})

    @pytest.mark.asyncio
    async def test_duplicate_timestamps_raise_value_error(self) -> None:
        """Bars with duplicate timestamps must raise ValueError."""
        runner = _make_runner(symbols=["BTC/USDT"])
        bars = _make_btc_bars(n=200)
        # Create a duplicate by copying timestamp from bar[0] to bar[1]
        dup_bar = OHLCVBar(
            symbol=bars[1].symbol,
            timeframe=bars[1].timeframe,
            timestamp=bars[0].timestamp,
            open=bars[1].open,
            high=bars[1].high,
            low=bars[1].low,
            close=bars[1].close,
            volume=bars[1].volume,
        )
        bars[1] = dup_bar
        with pytest.raises(ValueError, match=r"(?i)duplicate"):
            await runner.run({"BTC/USDT": bars})


# ===========================================================================
# 3. Full run — no-signal strategy
# ===========================================================================


class TestBacktestRunNoSignals:
    """
    End-to-end run using _AlwaysHoldStrategy (emits zero signals).

    These tests verify that the backtest infrastructure works correctly in
    the degenerate case where no trading occurs: equity should be unchanged,
    all trade-count fields should be zero, and the result object should be
    fully populated.
    """

    @pytest.fixture
    async def no_signal_result(self) -> BacktestResult:
        """
        Cached run result for _AlwaysHoldStrategy over 200 BTC/USDT bars.

        Re-used across multiple tests in this class to avoid repeated
        async engine starts.
        """
        runner = _make_runner(seed=42)
        bars = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}
        return await runner.run(bars)

    @pytest.mark.asyncio
    async def test_no_signal_strategy_produces_zero_trades(
        self, no_signal_result: BacktestResult
    ) -> None:
        """A strategy that never emits signals must produce zero completed trades."""
        assert no_signal_result.total_trades == 0

    @pytest.mark.asyncio
    async def test_no_signal_strategy_equity_unchanged(
        self, no_signal_result: BacktestResult
    ) -> None:
        """
        With no trades executed, final equity must equal initial capital.

        The paper engine charges no fees when no orders are placed, so
        equity must be precisely equal to the starting capital.
        """
        assert no_signal_result.final_equity == no_signal_result.initial_capital

    @pytest.mark.asyncio
    async def test_result_has_all_expected_fields(
        self, no_signal_result: BacktestResult
    ) -> None:
        """
        All ~30 declared fields on BacktestResult must be present and
        have their expected types.

        This test guards against accidental removal of result fields
        or type mismatches introduced during refactoring.
        """
        r = no_signal_result

        # Metadata fields
        assert isinstance(r.run_id, str) and r.run_id.startswith("bt-")
        assert isinstance(r.strategy_ids, list) and len(r.strategy_ids) == 1
        assert isinstance(r.symbols, list) and r.symbols == ["BTC/USDT"]
        assert isinstance(r.timeframe, TimeFrame)
        assert isinstance(r.start_date, datetime)
        assert isinstance(r.end_date, datetime)
        assert isinstance(r.duration_days, int) and r.duration_days >= 0

        # Capital
        assert isinstance(r.initial_capital, Decimal)
        assert isinstance(r.final_equity, Decimal)

        # Returns
        assert isinstance(r.total_return_pct, float)
        assert isinstance(r.cagr, float)

        # Risk metrics
        assert isinstance(r.max_drawdown_pct, float) and r.max_drawdown_pct >= 0.0
        assert isinstance(r.max_drawdown_duration_bars, int)
        assert r.max_drawdown_duration_bars >= 0
        assert isinstance(r.sharpe_ratio, float)
        assert isinstance(r.sortino_ratio, float)
        assert isinstance(r.calmar_ratio, float)

        # Trade statistics
        assert isinstance(r.total_trades, int) and r.total_trades >= 0
        assert isinstance(r.winning_trades, int) and r.winning_trades >= 0
        assert isinstance(r.losing_trades, int) and r.losing_trades >= 0
        assert isinstance(r.win_rate, float)
        assert isinstance(r.profit_factor, float)
        assert isinstance(r.average_trade_pnl, Decimal)
        assert isinstance(r.average_win, Decimal)
        assert isinstance(r.average_loss, Decimal)
        assert isinstance(r.largest_win, Decimal)
        assert isinstance(r.largest_loss, Decimal)

        # Exposure
        assert isinstance(r.total_bars, int) and r.total_bars >= 0
        assert isinstance(r.bars_in_market, int) and r.bars_in_market >= 0
        assert isinstance(r.exposure_pct, float)
        assert 0.0 <= r.exposure_pct <= 1.0

        # Curve and log
        assert isinstance(r.equity_curve, list)
        assert isinstance(r.trades, list)
        assert isinstance(r.total_fees_paid, Decimal)

    @pytest.mark.asyncio
    async def test_equity_curve_length_matches_bar_count(
        self, no_signal_result: BacktestResult
    ) -> None:
        """
        The equity curve must be non-empty for a 200-bar run and its length
        must not exceed the number of bars supplied.

        PortfolioAccounting records equity on every ``update_market_prices()``
        call, which happens on every bar including warmup.  The resulting
        raw curve is stored; the BacktestRunner converts it to EquityCurvePoints.
        """
        r = no_signal_result
        # Non-empty after a 200-bar run
        assert len(r.equity_curve) > 0
        # Cannot exceed total bars supplied
        assert len(r.equity_curve) <= 200

    @pytest.mark.asyncio
    async def test_initial_capital_matches_input(
        self, no_signal_result: BacktestResult
    ) -> None:
        """result.initial_capital must exactly equal the value passed to the runner."""
        assert no_signal_result.initial_capital == Decimal("10000")


# ===========================================================================
# 4. Determinism
# ===========================================================================


class TestBacktestDeterminism:
    """
    BacktestRunner must produce byte-identical results when given the same
    seed and input data.  Two runs with different seeds are allowed to
    produce different results.
    """

    @pytest.mark.asyncio
    async def test_same_seed_produces_identical_results(self) -> None:
        """
        Two independent runners with the same seed and identical bars must
        produce identical BacktestResult values across every scalar field
        and every equity curve point.

        Equality is checked field-by-field to produce a precise failure
        message if any single field diverges.
        """
        bars = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}

        runner_a = _make_runner(seed=42)
        result_a = await runner_a.run({"BTC/USDT": list(bars["BTC/USDT"])})

        runner_b = _make_runner(seed=42)
        result_b = await runner_b.run({"BTC/USDT": list(bars["BTC/USDT"])})

        # Scalar metrics must be identical
        assert result_a.final_equity == result_b.final_equity
        assert result_a.total_return_pct == result_b.total_return_pct
        assert result_a.cagr == result_b.cagr
        assert result_a.max_drawdown_pct == result_b.max_drawdown_pct
        assert result_a.sharpe_ratio == result_b.sharpe_ratio
        assert result_a.total_trades == result_b.total_trades
        assert result_a.total_fees_paid == result_b.total_fees_paid
        assert result_a.total_bars == result_b.total_bars

        # Equity curve must have the same length
        assert len(result_a.equity_curve) == len(result_b.equity_curve)

        # Per-point equity values must match exactly
        for idx, (pt_a, pt_b) in enumerate(
            zip(result_a.equity_curve, result_b.equity_curve)
        ):
            assert pt_a.equity == pt_b.equity, (
                f"Equity curve diverged at index {idx}: "
                f"{pt_a.equity} != {pt_b.equity}"
            )

    @pytest.mark.asyncio
    async def test_different_seeds_can_produce_different_results(self) -> None:
        """
        Two runs with different seeds (and different bar series) are allowed
        to produce different results.

        For a no-signal strategy there is no stochastic trading behaviour,
        so equity may coincidentally be identical.  We therefore assert only
        that both runs complete without error and that both metrics are finite.
        """
        bars_42 = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}
        bars_99 = {"BTC/USDT": _make_btc_bars(n=200, seed=99)}

        runner_42 = _make_runner(seed=42)
        result_42 = await runner_42.run(bars_42)

        runner_99 = _make_runner(seed=99)
        result_99 = await runner_99.run(bars_99)

        # Both runs complete successfully
        assert isinstance(result_42, BacktestResult)
        assert isinstance(result_99, BacktestResult)

        # Key metrics must be finite (no NaN or inf for a no-trade run)
        assert math.isfinite(result_42.total_return_pct)
        assert math.isfinite(result_99.total_return_pct)


# ===========================================================================
# 5. Multi-symbol
# ===========================================================================


class TestBacktestMultiSymbol:
    """BacktestRunner must handle multiple symbols without error."""

    @pytest.mark.asyncio
    async def test_multi_symbol_run_succeeds(self) -> None:
        """
        A run with two symbols (BTC/USDT and ETH/USDT), each with 200 bars,
        must complete and return a BacktestResult listing both symbols.

        With a no-signal strategy, no trades are placed and equity must
        remain exactly equal to initial_capital.
        """
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="hold-multi")],
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=42,
        )
        bars = {
            "BTC/USDT": _make_btc_bars(n=200, seed=42),
            "ETH/USDT": _make_eth_bars(n=200, seed=99),
        }
        result = await runner.run(bars)

        assert isinstance(result, BacktestResult)
        assert "BTC/USDT" in result.symbols
        assert "ETH/USDT" in result.symbols
        assert result.total_trades == 0
        # No fees with no trades: equity must equal initial capital exactly
        assert result.final_equity == Decimal("10000")


# ===========================================================================
# 6. Metrics sanity — zero-trade runs
# ===========================================================================


class TestBacktestMetrics:
    """
    Verifies that individual metrics are correctly computed and internally
    consistent for a no-trade run.

    These tests act as regression guards: if the metrics formulas change
    in a way that breaks the zero-trade invariants, they will fail.
    """

    @pytest.fixture
    async def zero_trade_result(self) -> BacktestResult:
        """
        Shared fixture: single-symbol, 200-bar, no-signal run.

        Re-used across all metrics tests to avoid repeated setup cost.
        """
        runner = _make_runner(seed=42)
        bars = {"BTC/USDT": _make_btc_bars(n=200, seed=42)}
        return await runner.run(bars)

    @pytest.mark.asyncio
    async def test_no_trades_zero_win_rate(
        self, zero_trade_result: BacktestResult
    ) -> None:
        """
        With zero completed trades, win_rate must be 0.0.

        compute_trade_statistics() returns a default TradeStatistics with
        win_rate=0.0 when the trade list is empty.
        """
        assert zero_trade_result.total_trades == 0
        assert zero_trade_result.win_rate == 0.0

    @pytest.mark.asyncio
    async def test_no_trades_zero_max_drawdown(
        self, zero_trade_result: BacktestResult
    ) -> None:
        """
        With no trades and flat equity, the maximum drawdown must be zero.

        When the paper engine holds cash throughout, equity equals
        initial_capital on every bar.  The drawdown of a flat curve is 0.0.
        """
        assert zero_trade_result.max_drawdown_pct == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_cagr_sign_matches_total_return_sign(
        self, zero_trade_result: BacktestResult
    ) -> None:
        """
        CAGR and total_return_pct must have the same sign.

        Both metrics measure overall profitability: if one is positive the
        other must be positive, and vice versa.  For a no-trade run both
        are 0.0 (equity unchanged from initial_capital).
        """
        r = zero_trade_result

        # Both must be finite floats (no NaN/inf in a stable run)
        assert math.isfinite(r.total_return_pct)
        assert math.isfinite(r.cagr)

        if r.total_return_pct > 0.0:
            assert r.cagr > 0.0, (
                f"total_return_pct={r.total_return_pct:.6f} is positive "
                f"but cagr={r.cagr:.6f} is not"
            )
        elif r.total_return_pct < 0.0:
            assert r.cagr < 0.0, (
                f"total_return_pct={r.total_return_pct:.6f} is negative "
                f"but cagr={r.cagr:.6f} is not"
            )
        else:
            # Zero return: CAGR must also be zero
            assert r.cagr == pytest.approx(0.0, abs=1e-9)
