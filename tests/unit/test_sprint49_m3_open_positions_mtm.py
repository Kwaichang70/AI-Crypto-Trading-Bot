"""
tests/unit/test_sprint49_m3_open_positions_mtm.py
---------------------------------------------------
Sprint 49 M3: Closed/open split -- OpenPositionMTM + BacktestResult wiring.

Tests verify:
1. OpenPositionMTM model construction, immutability, round-trip, negative PnL (3 tests)
2. BacktestResult.open_positions_mtm field -- default empty list + explicit populate (1 test)
3. BacktestRunner end-to-end: buy-never-sell strategy leaves open position in result (1 test)
4. build_backtest_metrics propagates open_positions_mtm to BacktestMetricsResponse (1 test)

Total: 6 tests across 4 classes.

n_closed_trades was dropped from BacktestResult + BacktestMetricsResponse per
arch-critic Q3 (single-source-of-truth violation). RunORM.n_closed_trades SQL column
is populated by persist_backtest_results() from result.total_trades directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import BacktestResult, OpenPositionMTM
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

from tests.conftest import make_bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_result(**overrides: Any) -> BacktestResult:
    """Construct a minimal valid BacktestResult for schema-level tests."""
    base: dict[str, Any] = dict(
        run_id="test-m3",
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
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=0.5,
        profit_factor=1.5,
        profit_factor_is_infinite=False,
        average_trade_pnl=Decimal("25"),
        average_win=Decimal("50"),
        average_loss=Decimal("0"),
        largest_win=Decimal("50"),
        largest_loss=Decimal("0"),
        total_bars=100,
        bars_in_market=40,
        exposure_pct=0.4,
    )
    base.update(overrides)
    return BacktestResult(**base)


def _make_open_mtm(
    *,
    symbol: str = "BTC/USDT",
    quantity: str = "0.01",
    entry_price: str = "50000",
    last_price: str = "51000",
    unrealised_pnl: str = "10",
) -> OpenPositionMTM:
    return OpenPositionMTM(
        symbol=symbol,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        last_price=Decimal(last_price),
        unrealised_pnl=Decimal(unrealised_pnl),
        opened_at=datetime(2024, 1, 15, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Strategy stubs
# ---------------------------------------------------------------------------

class _BuyNeverSellStrategy(BaseStrategy):
    """
    Buys on the first post-warmup bar, never sells.
    After the run ends, BTC/USDT is still open -- verifying
    open_positions_mtm is populated with a non-zero MTM record.
    """

    metadata = StrategyMetadata(
        name="Buy Never Sell M3",
        description="Always holds an open position",
        version="1.0.0",
    )

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {}

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        # Emit BUY on every bar (strategy_id is set via __init__ and stored on instance)
        return [
            Signal(
                strategy_id=self._strategy_id,
                symbol=bars[-1].symbol,
                direction=SignalDirection.BUY,
                target_position=Decimal("0.01"),
                confidence=1.0,
            )
        ]


# ---------------------------------------------------------------------------
# Test class 1: OpenPositionMTM model
# ---------------------------------------------------------------------------

class TestOpenPositionMTMModel:
    """Validates the OpenPositionMTM Pydantic model schema and constraints."""

    def test_construction_and_fields(self) -> None:
        """Model constructs correctly; all 6 fields are accessible."""
        mtm = _make_open_mtm()
        assert mtm.symbol == "BTC/USDT"
        assert mtm.quantity == Decimal("0.01")
        assert mtm.entry_price == Decimal("50000")
        assert mtm.last_price == Decimal("51000")
        assert mtm.unrealised_pnl == Decimal("10")
        assert isinstance(mtm.opened_at, datetime)

    def test_frozen_prevents_mutation(self) -> None:
        """OpenPositionMTM is frozen -- attribute assignment must raise.

        Pydantic v2 frozen models raise ValidationError on assignment attempt.
        """
        mtm = _make_open_mtm()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            mtm.last_price = Decimal("52000")  # type: ignore[misc]

    def test_negative_unrealised_pnl_for_loss(self) -> None:
        """Positions with last_price < entry_price yield negative unrealised_pnl."""
        mtm = _make_open_mtm(entry_price="50000", last_price="48000", unrealised_pnl="-20")
        assert mtm.unrealised_pnl < Decimal("0")


# ---------------------------------------------------------------------------
# Test class 2: BacktestResult.open_positions_mtm field
# ---------------------------------------------------------------------------

class TestBacktestResultOpenPositionsMTM:
    """BacktestResult accepts the new open_positions_mtm field."""

    def test_defaults_to_empty_list(self) -> None:
        """open_positions_mtm is optional; defaults to empty list."""
        r = _minimal_result()
        assert r.open_positions_mtm == []

    def test_explicit_assignment_stored(self) -> None:
        """Explicitly passing open_positions_mtm stores the list correctly."""
        mtm = _make_open_mtm()
        r = _minimal_result(open_positions_mtm=[mtm])
        assert len(r.open_positions_mtm) == 1
        assert r.open_positions_mtm[0].symbol == "BTC/USDT"
        assert r.open_positions_mtm[0].unrealised_pnl == Decimal("10")


# ---------------------------------------------------------------------------
# Test class 3: BacktestRunner end-to-end integration
# ---------------------------------------------------------------------------

class TestBacktestRunnerOpenPositionsMTM:
    """
    Verifies BacktestRunner.run() populates open_positions_mtm correctly.

    Uses a real BacktestRunner with a synthetic strategy -- no mocks.
    """

    @pytest.mark.asyncio
    async def test_buy_never_sell_yields_open_position(self) -> None:
        """
        A strategy that buys and never sells leaves a position open at
        end-of-backtest.  open_positions_mtm must contain exactly 1 record
        with quantity > 0 and a last_price equal to the terminal bar's close.
        """
        bars = make_bars(200, seed=42)
        runner = BacktestRunner(
            strategies=[_BuyNeverSellStrategy("buy_never_sell", {})],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("100000"),
        )
        result = await runner.run({"BTC/USDT": bars})
        # At least 1 open position (the BUY was never closed)
        assert len(result.open_positions_mtm) >= 1
        open_pos = result.open_positions_mtm[0]
        assert open_pos.symbol == "BTC/USDT"
        assert open_pos.quantity > Decimal("0")
        # last_price must equal the terminal bar close (Decimal converted)
        terminal_close = Decimal(str(bars[-1].close))
        assert open_pos.last_price == terminal_close
        # unrealised_pnl = (last_price - entry_price) * quantity
        expected_unrealised = (open_pos.last_price - open_pos.entry_price) * open_pos.quantity
        assert open_pos.unrealised_pnl == expected_unrealised


# ---------------------------------------------------------------------------
# Test class 4: build_backtest_metrics propagation
# ---------------------------------------------------------------------------

class TestBuildBacktestMetricsPropagation:
    """open_positions_mtm round-trips through build_backtest_metrics."""

    def test_open_positions_mtm_propagated(self) -> None:
        """
        build_backtest_metrics serialises open_positions_mtm into
        BacktestMetricsResponse.open_positions_mtm with Decimal-to-str conversion.
        """
        from api.services.run_persistence import build_backtest_metrics

        mtm = _make_open_mtm(
            symbol="ETH/USDT",
            quantity="0.5",
            entry_price="3000",
            last_price="3100",
            unrealised_pnl="50",
        )
        r = _minimal_result(open_positions_mtm=[mtm])
        response = build_backtest_metrics(r)

        assert len(response.open_positions_mtm) == 1
        resp_pos = response.open_positions_mtm[0]
        assert resp_pos.symbol == "ETH/USDT"
        # Serialiser converts Decimal -> str; verify numeric value preserved
        assert Decimal(resp_pos.quantity) == Decimal("0.5")
        assert Decimal(resp_pos.entry_price) == Decimal("3000")
        assert Decimal(resp_pos.last_price) == Decimal("3100")
        assert Decimal(resp_pos.unrealised_pnl) == Decimal("50")
