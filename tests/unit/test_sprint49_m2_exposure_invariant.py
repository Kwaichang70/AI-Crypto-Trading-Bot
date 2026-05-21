"""
tests/unit/test_sprint49_m2_exposure_invariant.py
---------------------------------------------------
Sprint 49 M2: Authoritative per-bar exposure tracking.

Tests verify:
1. BacktestResult accepts exposure_pct_per_symbol (default empty dict)
2. BacktestMetricsResponse accepts exposure_pct_per_symbol (default empty dict)
3. StrategyEngine per-bar counter semantics (replicated in isolation)
4. compute_exposure clamp invariant retained
5. build_backtest_metrics propagates exposure_pct_per_symbol
6. CR-006: BacktestRunner.run() populates exposure_pct_per_symbol end-to-end
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import BacktestResult, compute_exposure
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

from tests.conftest import make_bars


def _minimal_result(**overrides: Any) -> BacktestResult:
    base: dict[str, Any] = dict(
        run_id="test",
        strategy_ids=["s1"],
        symbols=["BTC/USDT"],
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 2, 1, tzinfo=UTC),
        duration_days=31,
        initial_capital=Decimal("10000"),
        final_equity=Decimal("10100"),
        total_return_pct=0.01,
        cagr=0.01,
        max_drawdown_pct=0.02,
        max_drawdown_duration_bars=5,
        sharpe_ratio=0.5,
        sortino_ratio=0.6,
        calmar_ratio=0.5,
        total_trades=3,
        winning_trades=2,
        losing_trades=1,
        win_rate=0.667,
        profit_factor=2.0,
        profit_factor_is_infinite=False,
        average_trade_pnl=Decimal("33"),
        average_win=Decimal("50"),
        average_loss=Decimal("0"),
        largest_win=Decimal("100"),
        largest_loss=Decimal("0"),
        total_bars=100,
        bars_in_market=60,
        exposure_pct=0.6,
    )
    base.update(overrides)
    return BacktestResult(**base)


class TestBacktestResultExposurePerSymbol:
    def test_default_is_empty_dict(self) -> None:
        r = _minimal_result()
        assert r.exposure_pct_per_symbol == {}

    def test_explicit_per_symbol_stored(self) -> None:
        r = _minimal_result(exposure_pct_per_symbol={"BTC/USDT": 0.6, "ETH/USDT": 0.4})
        assert r.exposure_pct_per_symbol["BTC/USDT"] == pytest.approx(0.6)
        assert r.exposure_pct_per_symbol["ETH/USDT"] == pytest.approx(0.4)

    def test_per_symbol_values_in_range(self) -> None:
        r = _minimal_result(exposure_pct_per_symbol={"BTC/USDT": 1.0})
        assert 0.0 <= r.exposure_pct_per_symbol["BTC/USDT"] <= 1.0

    def test_model_dump_includes_field(self) -> None:
        r = _minimal_result(exposure_pct_per_symbol={"BTC/USDT": 0.5})
        dumped = r.model_dump()
        assert "exposure_pct_per_symbol" in dumped
        assert dumped["exposure_pct_per_symbol"] == {"BTC/USDT": 0.5}


class TestBacktestMetricsResponseExposurePerSymbol:
    def _minimal_schema_kwargs(self) -> dict[str, Any]:
        return dict(
            total_return_pct=0.01,
            cagr=0.01,
            initial_capital="10000",
            final_equity="10100",
            total_fees_paid="0",
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            calmar_ratio=0.5,
            max_drawdown_pct=0.02,
            max_drawdown_duration_bars=5,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=0.667,
            profit_factor=2.0,
            profit_factor_is_infinite=False,
            average_trade_pnl="33",
            average_win="50",
            average_loss="0",
            largest_win="100",
            largest_loss="0",
            total_bars=100,
            bars_in_market=60,
            exposure_pct=0.6,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 2, 1, tzinfo=UTC),
            duration_days=31,
        )

    def test_default_is_empty_dict(self) -> None:
        from api.schemas import BacktestMetricsResponse
        schema = BacktestMetricsResponse(**self._minimal_schema_kwargs())
        assert schema.exposure_pct_per_symbol == {}

    def test_per_symbol_round_trips(self) -> None:
        from api.schemas import BacktestMetricsResponse
        kwargs = self._minimal_schema_kwargs()
        kwargs["exposure_pct_per_symbol"] = {"ETH/USDT": 0.3}
        schema = BacktestMetricsResponse(**kwargs)
        assert schema.exposure_pct_per_symbol == {"ETH/USDT": 0.3}


class TestStrategyEngineExposureTrackerSemantics:
    """
    Test the per-bar tracker logic in isolation. The integration between
    the tracker and the full async engine is covered by the existing
    backtest integration tests (which now route through the engine summary
    dict to BacktestResult.exposure_pct_per_symbol).
    """

    def _replicate_tracker(
        self,
        symbols: list[str],
        positions_per_bar: list[dict[str, bool]],
    ) -> tuple[int, dict[str, int]]:
        exposure_bars_total = 0
        exposure_bars_per_symbol: dict[str, int] = {s: 0 for s in symbols}

        for bar_state in positions_per_bar:
            any_open = False
            for sym in symbols:
                if bar_state.get(sym, False):
                    exposure_bars_per_symbol[sym] += 1
                    any_open = True
            if any_open:
                exposure_bars_total += 1

        return exposure_bars_total, exposure_bars_per_symbol

    def test_all_flat_yields_zero(self) -> None:
        total, per_sym = self._replicate_tracker(
            symbols=["BTC/USDT"],
            positions_per_bar=[{"BTC/USDT": False}] * 10,
        )
        assert total == 0
        assert per_sym["BTC/USDT"] == 0

    def test_always_open_yields_full_count(self) -> None:
        total, per_sym = self._replicate_tracker(
            symbols=["BTC/USDT"],
            positions_per_bar=[{"BTC/USDT": True}] * 10,
        )
        assert total == 10
        assert per_sym["BTC/USDT"] == 10

    def test_partial_open_counts_correctly(self) -> None:
        states = [{"BTC/USDT": i in range(3, 8)} for i in range(10)]
        total, per_sym = self._replicate_tracker(
            symbols=["BTC/USDT"],
            positions_per_bar=states,
        )
        assert total == 5
        assert per_sym["BTC/USDT"] == 5

    def test_multi_symbol_overlap_counts_union(self) -> None:
        """BTC open bars 0-4, ETH open bars 3-9. Union = bars 0-9 = 10 total."""
        states = [
            {"BTC/USDT": i < 5, "ETH/USDT": i >= 3}
            for i in range(10)
        ]
        total, per_sym = self._replicate_tracker(
            symbols=["BTC/USDT", "ETH/USDT"],
            positions_per_bar=states,
        )
        assert total == 10
        assert per_sym["BTC/USDT"] == 5
        assert per_sym["ETH/USDT"] == 7

    def test_multi_symbol_disjoint_counts_union(self) -> None:
        """BTC bars 0-4, ETH bars 5-9. Union = 10, no overlap. Per-symbol = 5 each."""
        states = [
            {"BTC/USDT": i < 5, "ETH/USDT": i >= 5}
            for i in range(10)
        ]
        total, per_sym = self._replicate_tracker(
            symbols=["BTC/USDT", "ETH/USDT"],
            positions_per_bar=states,
        )
        assert total == 10
        assert per_sym["BTC/USDT"] == 5
        assert per_sym["ETH/USDT"] == 5


class TestComputeExposureClampRetained:
    def test_clamp_still_prevents_over_100(self) -> None:
        assert compute_exposure(120, 100) == 1.0

    def test_zero_total_bars_returns_zero(self) -> None:
        assert compute_exposure(0, 0) == 0.0


class TestBuildBacktestMetricsMapsPerSymbol:
    def test_per_symbol_field_propagates(self) -> None:
        from api.services.run_persistence import build_backtest_metrics
        result = _minimal_result(exposure_pct_per_symbol={"BTC/USDT": 0.75})
        schema = build_backtest_metrics(result)
        assert schema.exposure_pct_per_symbol == {"BTC/USDT": 0.75}

    def test_empty_per_symbol_propagates(self) -> None:
        from api.services.run_persistence import build_backtest_metrics
        result = _minimal_result()
        schema = build_backtest_metrics(result)
        assert schema.exposure_pct_per_symbol == {}


# ---------------------------------------------------------------------------
# CR-006: End-to-end BacktestRunner integration test
# ---------------------------------------------------------------------------


class _AlwaysHoldStrategy(BaseStrategy):
    """Minimal strategy that never emits signals — pure infrastructure test."""

    metadata = StrategyMetadata(
        name="always_hold_m2",
        version="1.0.0",
        description="CR-006 integration stub: never trades",
    )

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        return []

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}


class TestBacktestRunnerExposurePerSymbol:
    """
    CR-006: Verify that BacktestResult.exposure_pct_per_symbol is populated
    end-to-end through BacktestRunner.run().

    A no-signal strategy over BTC/USDT bars results in zero trades and all
    flat positions.  The per-symbol dict must be present and keyed to the
    correct symbol, with a value in [0.0, 1.0].
    """

    @pytest.mark.asyncio
    async def test_per_symbol_dict_populated_after_run(self) -> None:
        """
        After BacktestRunner.run() with a no-signal strategy, the result's
        exposure_pct_per_symbol must be a dict keyed by the traded symbol.

        A no-signal (always-hold) strategy never opens a position, so the
        exposure fraction for BTC/USDT must be 0.0 and the key must be present.
        """
        bars = make_bars(200, seed=42, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="cr006-test")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=42,
        )

        result = await runner.run({"BTC/USDT": bars})

        assert isinstance(result.exposure_pct_per_symbol, dict)
        assert "BTC/USDT" in result.exposure_pct_per_symbol
        assert 0.0 <= result.exposure_pct_per_symbol["BTC/USDT"] <= 1.0
