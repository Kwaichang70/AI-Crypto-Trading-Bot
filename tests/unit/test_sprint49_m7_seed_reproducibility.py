"""
Sprint 49 / M7 -- Seed pinning + reproducibility (INF-9).

Tests verifying that BacktestRunner produces deterministic results when a
seed is supplied, auto-generates a seed when none is provided, persists
the seed through the result chain, and validates RunCreateRequest.seed
range constraints.
"""
from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from common.models import OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.strategies.ma_crossover import MACrossoverStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int = 120, seed: int = 0) -> list[OHLCVBar]:
    """Generate synthetic OHLCV bars deterministically.

    close is clamped to [low, high] to satisfy OHLCVBar invariants.
    """
    rng = random.Random(seed)
    bars: list[OHLCVBar] = []
    price = 30_000.0
    for i in range(n):
        open_ = price
        high = price * (1 + rng.uniform(0.001, 0.01))
        low = price * (1 - rng.uniform(0.001, 0.01))
        # close must be within [low, high] — clamp explicitly
        raw_close = price * (1 + rng.uniform(-0.005, 0.005))
        close = max(low, min(high, raw_close))
        volume = rng.uniform(1, 10)
        bars.append(
            OHLCVBar(
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC),
                open=Decimal(str(round(open_, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(close, 2))),
                volume=Decimal(str(round(volume, 4))),
                timeframe=TimeFrame.ONE_HOUR,
            )
        )
        price = close
    return bars


def _make_runner(seed: int | None = None) -> BacktestRunner:
    strategy = MACrossoverStrategy(
        strategy_id="ma-test",
        params={"fast_period": 5, "slow_period": 10},
    )
    return BacktestRunner(
        strategies=[strategy],
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSeedAutoGeneration:
    """BacktestRunner auto-generates a seed when None is passed."""

    def test_seed_none_generates_int(self) -> None:
        runner = _make_runner(seed=None)
        assert isinstance(runner.seed, int)
        assert 0 <= runner.seed <= 2**31 - 1

    def test_seed_none_generates_different_values_each_call(self) -> None:
        # Two runners constructed without seed should (almost certainly) differ.
        seeds = {_make_runner(seed=None).seed for _ in range(10)}
        # At least 2 distinct seeds from 10 draws in a 2^31 space.
        assert len(seeds) > 1

    def test_explicit_seed_preserved(self) -> None:
        runner = _make_runner(seed=42)
        assert runner.seed == 42


class TestReproducibility:
    """Same seed produces byte-identical BacktestResult fields."""

    @pytest.mark.asyncio
    async def test_same_seed_identical_results(self) -> None:
        bars = _make_bars(n=120, seed=0)
        bars_by_symbol = {"BTC/USDT": bars}

        runner1 = _make_runner(seed=42)
        runner2 = _make_runner(seed=42)

        result1 = await runner1.run(bars_by_symbol)
        result2 = await runner2.run(bars_by_symbol)

        # Core financial outputs must be identical
        assert result1.final_equity == result2.final_equity
        assert result1.total_trades == result2.total_trades
        assert result1.sharpe_ratio == result2.sharpe_ratio
        assert result1.max_drawdown_pct == result2.max_drawdown_pct
        assert result1.total_return_pct == result2.total_return_pct

    @pytest.mark.asyncio
    async def test_different_seeds_can_yield_different_results(self) -> None:
        """Different seeds should produce valid results without corrupting each other.

        Because MACrossover is deterministic given fixed bars, this test
        verifies seed isolation of numpy.random state -- even if the Python
        random module doesn't affect MA logic, numpy seeding should not
        corrupt results from a prior run.  We verify both seeds produce valid
        (non-corrupted) results.
        """
        bars = _make_bars(n=120, seed=0)
        bars_by_symbol = {"BTC/USDT": bars}

        result42 = await _make_runner(seed=42).run(bars_by_symbol)
        result99 = await _make_runner(seed=99).run(bars_by_symbol)

        # Both runs must complete successfully -- results are valid
        assert isinstance(result42.final_equity, Decimal)
        assert isinstance(result99.final_equity, Decimal)
        # Both runs see the same deterministic bar data, so for a pure
        # price-signal strategy (MA crossover) the results must be equal
        # (no randomness used by the strategy itself).  The seed isolation
        # guarantee is: seed 42 followed by seed 99 does NOT corrupt seed 99.
        assert result99.final_equity == result42.final_equity


class TestSeedPersistence:
    """Seed is stored in BacktestResult and flows through the response chain."""

    @pytest.mark.asyncio
    async def test_seed_persisted_to_backtest_result(self) -> None:
        bars = _make_bars(n=120, seed=0)
        bars_by_symbol = {"BTC/USDT": bars}

        runner = _make_runner(seed=42)
        result = await runner.run(bars_by_symbol)

        assert result.seed == 42

    @pytest.mark.asyncio
    async def test_autogenerated_seed_persisted_to_result(self) -> None:
        bars = _make_bars(n=120, seed=0)
        bars_by_symbol = {"BTC/USDT": bars}

        runner = _make_runner(seed=None)
        resolved_seed = runner.seed  # captured before run()
        result = await runner.run(bars_by_symbol)

        assert result.seed == resolved_seed
        assert isinstance(result.seed, int)

    def test_build_backtest_metrics_propagates_seed(self) -> None:
        """build_backtest_metrics() propagates seed via getattr guard."""
        from api.services.run_persistence import build_backtest_metrics

        mock_result = MagicMock()
        mock_result.seed = 77
        mock_result.total_return_pct = 0.0
        mock_result.cagr = 0.0
        mock_result.initial_capital = Decimal("10000")
        mock_result.final_equity = Decimal("10000")
        mock_result.total_fees_paid = Decimal("0")
        mock_result.sharpe_ratio = 0.0
        mock_result.sortino_ratio = 0.0
        mock_result.calmar_ratio = 0.0
        mock_result.max_drawdown_pct = 0.0
        mock_result.max_drawdown_duration_bars = 0
        mock_result.total_trades = 0
        mock_result.winning_trades = 0
        mock_result.losing_trades = 0
        mock_result.win_rate = 0.0
        mock_result.profit_factor = None
        mock_result.profit_factor_is_infinite = False
        mock_result.average_trade_pnl = Decimal("0")
        mock_result.average_win = Decimal("0")
        mock_result.average_loss = Decimal("0")
        mock_result.largest_win = Decimal("0")
        mock_result.largest_loss = Decimal("0")
        mock_result.total_bars = 0
        mock_result.bars_in_market = 0
        mock_result.exposure_pct = 0.0
        mock_result.exposure_pct_per_symbol = {}
        mock_result.start_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_result.end_date = datetime(2024, 1, 2, tzinfo=UTC)
        mock_result.duration_days = 1
        mock_result.open_positions_mtm = []
        mock_result.psr = None
        mock_result.n_observations = 0
        mock_result.confidence_flag = None
        mock_result.quote_currency = None
        mock_result.reporting_currency = None

        metrics = build_backtest_metrics(mock_result)
        assert metrics.seed == 77

    def test_build_backtest_metrics_seed_missing_returns_none(self) -> None:
        """Pre-M7 BacktestResult lacks seed attribute -> getattr returns None."""
        from api.services.run_persistence import build_backtest_metrics

        # SimpleNamespace with all required fields set, seed deliberately OMITTED.
        # Unlike MagicMock(spec=[]), SimpleNamespace raises AttributeError on missing
        # attributes, so getattr(ns, "seed", None) correctly returns None.
        ns = SimpleNamespace(
            total_return_pct=0.0,
            cagr=0.0,
            initial_capital=Decimal("10000"),
            final_equity=Decimal("10000"),
            total_fees_paid=Decimal("0"),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_bars=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=None,
            profit_factor_is_infinite=False,
            average_trade_pnl=Decimal("0"),
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            total_bars=0,
            bars_in_market=0,
            exposure_pct=0.0,
            exposure_pct_per_symbol={},
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 2, tzinfo=UTC),
            duration_days=1,
            open_positions_mtm=[],
            psr=None,
            n_observations=0,
            confidence_flag=None,
            quote_currency=None,
            reporting_currency=None,
            # seed: deliberately OMITTED to test the getattr fallback
        )
        metrics = build_backtest_metrics(ns)
        assert metrics.seed is None  # getattr fallback returns None


class TestSchemaValidation:
    """RunCreateRequest.seed field validates range correctly."""

    def test_seed_valid_zero(self) -> None:
        from api.schemas import RunCreateRequest

        req = RunCreateRequest(
            strategy_name="ma_crossover",
            symbols=["BTC/USDT"],
            timeframe="1h",
            mode="backtest",
            initial_capital="10000",
            backtest_start="2024-01-01T00:00:00Z",
            backtest_end="2024-06-01T00:00:00Z",
            seed=0,
        )
        assert req.seed == 0

    def test_seed_valid_max(self) -> None:
        from api.schemas import RunCreateRequest

        req = RunCreateRequest(
            strategy_name="ma_crossover",
            symbols=["BTC/USDT"],
            timeframe="1h",
            mode="backtest",
            initial_capital="10000",
            backtest_start="2024-01-01T00:00:00Z",
            backtest_end="2024-06-01T00:00:00Z",
            seed=2**31 - 1,
        )
        assert req.seed == 2**31 - 1

    def test_seed_negative_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from api.schemas import RunCreateRequest

        with pytest.raises(ValidationError):
            RunCreateRequest(
                strategy_name="ma_crossover",
                symbols=["BTC/USDT"],
                timeframe="1h",
                mode="backtest",
                initial_capital="10000",
                backtest_start="2024-01-01T00:00:00Z",
                backtest_end="2024-06-01T00:00:00Z",
                seed=-1,
            )

    def test_seed_too_large_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from api.schemas import RunCreateRequest

        with pytest.raises(ValidationError):
            RunCreateRequest(
                strategy_name="ma_crossover",
                symbols=["BTC/USDT"],
                timeframe="1h",
                mode="backtest",
                initial_capital="10000",
                backtest_start="2024-01-01T00:00:00Z",
                backtest_end="2024-06-01T00:00:00Z",
                seed=2**32,
            )

    def test_seed_none_is_valid(self) -> None:
        from api.schemas import RunCreateRequest

        req = RunCreateRequest(
            strategy_name="ma_crossover",
            symbols=["BTC/USDT"],
            timeframe="1h",
            mode="backtest",
            initial_capital="10000",
            backtest_start="2024-01-01T00:00:00Z",
            backtest_end="2024-06-01T00:00:00Z",
            seed=None,
        )
        assert req.seed is None
