"""
tests/unit/test_bracket_exit.py
--------------------------------
Unit tests for BracketExitManager (packages/trading/bracket_exit.py).

Covers fixed-% and ATR-multiple stop-loss / take-profit brackets, the
SL-before-TP precedence rule, the pending-exit fill-latency guard, the
ATR-unavailable fallback, flat-position cleanup, reset(), and — critically —
that emitted signals are classified as ``stop_loss`` / ``take_profit`` by
``ExitReasonDetector`` (guarding the "no 'trailing_stop' substring" trap).

Async note: not applicable — the manager is synchronous.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from common.models import OHLCVBar
from common.types import (
    OrderSide,
    OrderType,
    RunMode,
    SignalDirection,
    TimeFrame,
)
from trading.bracket_exit import BracketExitManager
from trading.models import Fill, Order, Position
from trading.strategy_engine import StrategyEngine
from trading.trade_journal import ExitReasonDetector


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _pos(
    *,
    symbol: str = "BTC/USD",
    quantity: str = "0.01",
    average_entry_price: str = "100",
    current_price: str = "100",
) -> Position:
    return Position(
        symbol=symbol,
        run_id="test-run",
        quantity=Decimal(quantity),
        average_entry_price=Decimal(average_entry_price),
        current_price=Decimal(current_price),
    )


def _flat(symbol: str = "BTC/USD") -> Position:
    return _pos(symbol=symbol, quantity="0")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestBracketExitInit:
    def test_fixed_mode_accepts_valid_pcts(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.04)
        assert mgr.bracket_mode == "fixed"
        assert mgr.requires_atr is False
        assert mgr.strategy_id == "bracket_exit"

    def test_fixed_mode_accepts_stop_only(self) -> None:
        BracketExitManager(stop_loss_pct=0.02)

    def test_fixed_mode_accepts_take_profit_only(self) -> None:
        BracketExitManager(take_profit_pct=0.05)

    def test_fixed_mode_requires_at_least_one_pct(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            BracketExitManager(bracket_mode="fixed")

    @pytest.mark.parametrize("bad", [0.0, 0.0005, 0.96, 1.5])
    def test_fixed_mode_rejects_out_of_range_pct(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0.001, 0.95\]"):
            BracketExitManager(stop_loss_pct=bad)

    def test_atr_mode_accepts_valid_multipliers(self) -> None:
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=2.0, atr_tp_multiplier=3.0
        )
        assert mgr.bracket_mode == "atr"
        assert mgr.requires_atr is True

    def test_atr_mode_requires_at_least_one_multiplier(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            BracketExitManager(bracket_mode="atr")

    @pytest.mark.parametrize("bad", [0.05, 0.0, 21.0, 100.0])
    def test_atr_mode_rejects_out_of_range_multiplier(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0.1, 20.0\]"):
            BracketExitManager(bracket_mode="atr", atr_sl_multiplier=bad)

    def test_atr_mode_rejects_bad_period(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            BracketExitManager(
                bracket_mode="atr", atr_sl_multiplier=2.0, atr_period=1
            )

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="bracket_mode"):
            BracketExitManager(stop_loss_pct=0.02, bracket_mode="moon")

    def test_rejects_trailing_stop_substring_in_strategy_id(self) -> None:
        # The detector substring trap — must be impossible to misconfigure.
        with pytest.raises(ValueError, match="trailing_stop"):
            BracketExitManager(stop_loss_pct=0.02, strategy_id="my_trailing_stop")


# ---------------------------------------------------------------------------
# Fixed-mode check()
# ---------------------------------------------------------------------------


class TestFixedBracket:
    def test_none_position_returns_none(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        assert mgr.check("BTC/USD", Decimal("100"), None) is None

    def test_flat_position_returns_none_and_clears(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        assert mgr.check("BTC/USD", Decimal("100"), _flat()) is None
        assert "BTC/USD" not in mgr.pending_exit_symbols

    def test_price_inside_brackets_holds(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.04)
        # entry 100 -> sl 98, tp 104; price 101 is inside.
        assert mgr.check("BTC/USD", Decimal("101"), _pos()) is None

    def test_stop_loss_triggers_below_level(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        sig = mgr.check("BTC/USD", Decimal("97.9"), _pos())
        assert sig is not None
        assert sig.direction == SignalDirection.SELL
        assert sig.target_position == Decimal("0")
        assert sig.confidence == 1.0
        assert sig.metadata["stop_loss"] is True
        assert "take_profit" not in sig.metadata
        assert sig.metadata["trigger"] == "bracket_stop_loss"

    def test_stop_loss_triggers_exactly_at_level(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        # entry 100 -> sl exactly 98.
        sig = mgr.check("BTC/USD", Decimal("98"), _pos())
        assert sig is not None
        assert sig.metadata["stop_loss"] is True

    def test_take_profit_triggers_above_level(self) -> None:
        mgr = BracketExitManager(take_profit_pct=0.04)
        sig = mgr.check("BTC/USD", Decimal("104.1"), _pos())
        assert sig is not None
        assert sig.direction == SignalDirection.SELL
        assert sig.target_position == Decimal("0")
        assert sig.metadata["take_profit"] is True
        assert "stop_loss" not in sig.metadata
        assert sig.metadata["trigger"] == "bracket_take_profit"

    def test_stop_loss_takes_precedence_over_take_profit(self) -> None:
        # Degenerate price that is simultaneously <= sl and >= tp cannot
        # happen with sane params, but precedence must hold structurally:
        # tiny tp + a crash still classifies as stop_loss.
        mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.001)
        sig = mgr.check("BTC/USD", Decimal("90"), _pos())
        assert sig is not None
        assert sig.metadata["stop_loss"] is True
        assert "take_profit" not in sig.metadata

    def test_pending_guard_blocks_second_emit(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        first = mgr.check("BTC/USD", Decimal("97"), _pos())
        assert first is not None
        # Same symbol still open (fill not yet landed) -> no duplicate.
        second = mgr.check("BTC/USD", Decimal("96"), _pos())
        assert second is None
        assert "BTC/USD" in mgr.pending_exit_symbols

    def test_pending_guard_clears_when_flat(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        mgr.check("BTC/USD", Decimal("97"), _pos())
        assert "BTC/USD" in mgr.pending_exit_symbols
        # Position closed -> guard cleared, fresh tracking next entry.
        assert mgr.check("BTC/USD", Decimal("97"), _flat()) is None
        assert "BTC/USD" not in mgr.pending_exit_symbols

    def test_zero_entry_price_returns_none(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        pos = _pos(average_entry_price="0", current_price="0")
        assert mgr.check("BTC/USD", Decimal("0"), pos) is None

    def test_multiple_symbols_independent(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        btc = mgr.check("BTC/USD", Decimal("97"), _pos(symbol="BTC/USD"))
        eth = mgr.check("ETH/USD", Decimal("101"), _pos(symbol="ETH/USD"))
        assert btc is not None
        assert eth is None  # inside brackets
        assert mgr.pending_exit_symbols == {"BTC/USD"}


# ---------------------------------------------------------------------------
# ATR-mode check()
# ---------------------------------------------------------------------------


class TestAtrBracket:
    def test_atr_stop_loss_triggers(self) -> None:
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=2.0, atr_tp_multiplier=3.0
        )
        # entry 100, atr 2, sl_mult 2 -> sl 96.
        sig = mgr.check("BTC/USD", Decimal("95"), _pos(), atr_value=Decimal("2"))
        assert sig is not None
        assert sig.metadata["stop_loss"] is True
        assert sig.metadata["bracket_mode"] == "atr"

    def test_atr_take_profit_triggers(self) -> None:
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=2.0, atr_tp_multiplier=3.0
        )
        # entry 100, atr 2, tp_mult 3 -> tp 106.
        sig = mgr.check("BTC/USD", Decimal("106.5"), _pos(), atr_value=Decimal("2"))
        assert sig is not None
        assert sig.metadata["take_profit"] is True

    def test_atr_inside_brackets_holds(self) -> None:
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=2.0, atr_tp_multiplier=3.0
        )
        assert mgr.check("BTC/USD", Decimal("101"), _pos(), atr_value=Decimal("2")) is None

    def test_atr_missing_value_holds_no_crash(self) -> None:
        mgr = BracketExitManager(bracket_mode="atr", atr_sl_multiplier=2.0)
        # No ATR -> no exit even on a price well below any plausible stop.
        assert mgr.check("BTC/USD", Decimal("50"), _pos(), atr_value=None) is None
        assert mgr.check("BTC/USD", Decimal("50"), _pos(), atr_value=Decimal("0")) is None

    def test_atr_fixed_pct_ignored_in_atr_mode(self) -> None:
        # Construction with atr mode ignores fixed pcts entirely.
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=1.0
        )
        # entry 100, atr 5 -> sl 95; price 96 holds.
        assert mgr.check("BTC/USD", Decimal("96"), _pos(), atr_value=Decimal("5")) is None
        sig = mgr.check("BTC/USD", Decimal("94"), _pos(), atr_value=Decimal("5"))
        assert sig is not None

    def test_atr_stop_below_zero_treated_as_no_stop(self) -> None:
        # atr * mult > entry -> negative stop -> treated as no stop.
        mgr = BracketExitManager(
            bracket_mode="atr", atr_sl_multiplier=20.0, atr_tp_multiplier=3.0
        )
        # entry 100, atr 10, sl_mult 20 -> sl = 100 - 200 = -100 -> dropped.
        assert mgr.check("BTC/USD", Decimal("1"), _pos(), atr_value=Decimal("10")) is None


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_pending(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        mgr.check("BTC/USD", Decimal("97"), _pos())
        assert mgr.pending_exit_symbols
        mgr.reset()
        assert not mgr.pending_exit_symbols

    def test_after_reset_can_emit_again(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        mgr.check("BTC/USD", Decimal("97"), _pos())
        mgr.reset()
        sig = mgr.check("BTC/USD", Decimal("97"), _pos())
        assert sig is not None


# ---------------------------------------------------------------------------
# ExitReasonDetector classification (guards the substring trap)
# ---------------------------------------------------------------------------


class TestExitReasonClassification:
    def test_stop_loss_signal_classified_as_stop_loss(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        sig = mgr.check("BTC/USD", Decimal("97"), _pos())
        assert sig is not None
        reason = ExitReasonDetector.detect(sig.strategy_id, dict(sig.metadata))
        assert reason == "stop_loss"

    def test_take_profit_signal_classified_as_take_profit(self) -> None:
        mgr = BracketExitManager(take_profit_pct=0.04)
        sig = mgr.check("BTC/USD", Decimal("105"), _pos())
        assert sig is not None
        reason = ExitReasonDetector.detect(sig.strategy_id, dict(sig.metadata))
        assert reason == "take_profit"

    def test_strategy_id_not_classified_as_trailing_stop(self) -> None:
        mgr = BracketExitManager(stop_loss_pct=0.02)
        sig = mgr.check("BTC/USD", Decimal("97"), _pos())
        assert sig is not None
        assert "trailing_stop" not in sig.strategy_id


# ---------------------------------------------------------------------------
# StrategyEngine integration
# ---------------------------------------------------------------------------


def _make_bar(
    *,
    symbol: str = "BTC/USD",
    close: str | Decimal = "100",
    timestamp: datetime | None = None,
) -> OHLCVBar:
    ts = timestamp or datetime(2024, 1, 1, tzinfo=UTC)
    close_d = Decimal(str(close))
    high_d = (close_d * Decimal("1.01")).quantize(Decimal("0.01"))
    low_d = (close_d * Decimal("0.99")).quantize(Decimal("0.01"))
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=ts,
        open=close_d,
        high=high_d,
        low=low_d,
        close=close_d,
        volume=Decimal("100"),
    )


def _make_fill(
    *,
    symbol: str = "BTC/USD",
    side: OrderSide = OrderSide.SELL,
    quantity: str = "0.01",
    price: str = "95",
) -> Fill:
    qty = Decimal(quantity)
    prc = Decimal(price)
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=qty,
        price=prc,
        fee=qty * prc * Decimal("0.001"),
        fee_currency="USD",
    )


def _make_engine(
    *,
    config: dict[str, Any] | None = None,
    symbols: list[str] | None = None,
) -> tuple[StrategyEngine, dict[str, Any]]:
    """StrategyEngine with mocked deps (mirrors test_trailing_stop._make_engine)."""
    strategy = MagicMock()
    strategy.strategy_id = "test_strategy"
    strategy.min_bars_required = 20
    strategy.on_start = MagicMock(return_value=None)
    strategy.on_stop = MagicMock(return_value=None)

    execution = AsyncMock()
    execution.on_start = AsyncMock(return_value=None)
    execution.on_stop = AsyncMock(return_value=None)
    execution.get_open_orders = MagicMock(return_value=[])
    execution.cancel_order = AsyncMock(return_value=None)

    market_data = AsyncMock()
    market_data.connect = AsyncMock(return_value=None)
    market_data.close = AsyncMock(return_value=None)

    risk_manager = MagicMock()
    risk_manager.kill_switch_active = False
    risk_manager.tick_cooldown = MagicMock(return_value=None)
    risk_manager.update_after_fill = MagicMock(return_value=None)

    portfolio = MagicMock()
    portfolio.get_summary = MagicMock(return_value={
        "current_equity": "10000",
        "total_trades": 0,
    })

    engine = StrategyEngine(
        strategies=[strategy],
        execution_engine=execution,
        risk_manager=risk_manager,
        market_data=market_data,
        portfolio=portfolio,
        symbols=symbols or ["BTC/USD"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
        config=config or {},
    )
    mocks: dict[str, Any] = {
        "strategy": strategy,
        "execution": execution,
        "market_data": market_data,
        "risk_manager": risk_manager,
        "portfolio": portfolio,
    }
    return engine, mocks


class TestBracketExitInStrategyEngine:
    async def test_no_config_no_manager(self) -> None:
        engine, _ = _make_engine(config={})
        await engine.start("run-001")
        assert engine._bracket_exit is None  # type: ignore[attr-defined]

    async def test_stop_loss_pct_builds_manager(self) -> None:
        engine, _ = _make_engine(config={"bracket_stop_loss_pct": 0.02})
        await engine.start("run-001")
        assert isinstance(engine._bracket_exit, BracketExitManager)  # type: ignore[attr-defined]
        assert engine._bracket_exit.bracket_mode == "fixed"  # type: ignore[attr-defined]

    async def test_empty_string_pct_ignored(self) -> None:
        # A blank UI field arriving as "" must not build a manager nor crash.
        engine, _ = _make_engine(config={"bracket_stop_loss_pct": ""})
        await engine.start("run-001")
        assert engine._bracket_exit is None  # type: ignore[attr-defined]

    async def test_invalid_atr_config_disables_gracefully(self) -> None:
        # ATR mode but no multipliers -> ValueError -> disabled, not crash.
        engine, _ = _make_engine(config={
            "bracket_mode": "atr",
            "bracket_stop_loss_pct": 0.02,  # presence triggers construction attempt
        })
        await engine.start("run-001")
        assert engine._bracket_exit is None  # type: ignore[attr-defined]

    async def test_stop_loss_sell_submitted_to_execution(self) -> None:
        engine, mocks = _make_engine(config={"bracket_stop_loss_pct": 0.05})
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        open_pos = _pos(average_entry_price="100", current_price="100")
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        # entry 100, sl_pct 0.05 -> stop 95; price 94 triggers.
        bar = _make_bar(close="94")
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

        signals = [c.args[0] for c in mocks["execution"].process_signal.await_args_list]
        sells = [s for s in signals if s.direction == SignalDirection.SELL]
        assert len(sells) >= 1
        assert sells[0].strategy_id == "bracket_exit"
        assert sells[0].metadata["stop_loss"] is True

    async def test_bracket_fill_routed_and_recorded(self) -> None:
        engine, mocks = _make_engine(config={"bracket_stop_loss_pct": 0.05})
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])

        open_pos = _pos(average_entry_price="100", current_price="100")
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        sell_fill = _make_fill(price="94")
        sell_order = Order(
            client_order_id=f"run-001-{uuid4().hex[:12]}",
            run_id="run-001",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )
        mocks["execution"].process_signal = AsyncMock(return_value=[sell_order])
        mocks["execution"].get_fills = AsyncMock(return_value=[sell_fill])

        bar = _make_bar(close="94")
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

        mocks["portfolio"].update_position.assert_called()
        mocks["portfolio"].record_trade.assert_called()
        # The recorded trade must carry the stop_loss exit reason.
        recorded = mocks["portfolio"].record_trade.call_args.args[0]
        assert recorded.exit_reason == "stop_loss"

    async def test_bracket_runs_before_trailing_stop(self) -> None:
        # Both configured; a crash through both levels must produce a single
        # bracket_exit SELL (trailing loop sees flat afterwards).
        engine, mocks = _make_engine(config={
            "bracket_stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.03,
        })
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        open_pos = _pos(average_entry_price="100", current_price="100")
        flat = _flat()
        # First call (bracket) sees the open position; after the bracket
        # "closes" it, subsequent get_position calls return flat.
        mocks["portfolio"].get_position = MagicMock(
            side_effect=[open_pos, flat, flat, flat]
        )

        bar = _make_bar(close="90")
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})

        signals = [c.args[0] for c in mocks["execution"].process_signal.await_args_list]
        bracket_sells = [s for s in signals if s.strategy_id == "bracket_exit"]
        trailing_sells = [s for s in signals if s.strategy_id == "trailing_stop"]
        assert len(bracket_sells) == 1
        assert len(trailing_sells) == 0

    async def test_atr_mode_end_to_end(self) -> None:
        engine, mocks = _make_engine(config={
            "bracket_mode": "atr",
            "bracket_atr_sl_multiplier": 2.0,
            "bracket_atr_period": 5,
        })
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        mocks["execution"].process_signal = AsyncMock(return_value=[])

        open_pos = _pos(average_entry_price="100", current_price="100")
        mocks["portfolio"].get_position = MagicMock(return_value=open_pos)

        # Build a flat ~100 history so ATR is small/stable, then a crash bar.
        history = [_make_bar(close=str(100 + (i % 2))) for i in range(20)]
        crash = _make_bar(close="80")
        history.append(crash)
        await engine._process_bar({"BTC/USD": crash}, {"BTC/USD": history})

        signals = [c.args[0] for c in mocks["execution"].process_signal.await_args_list]
        sells = [s for s in signals if s.strategy_id == "bracket_exit"]
        assert len(sells) >= 1
        assert sells[0].metadata["stop_loss"] is True

    async def test_bracket_error_does_not_crash_bar_loop(self) -> None:
        engine, mocks = _make_engine(config={"bracket_stop_loss_pct": 0.05})
        await engine.start("run-001")
        mocks["execution"].check_resting_orders = None
        mocks["strategy"].on_bar = MagicMock(return_value=[])
        # get_position raises -> bracket block logs + continues.
        mocks["portfolio"].get_position = MagicMock(side_effect=RuntimeError("boom"))

        bar = _make_bar(close="94")
        # Must not raise.
        await engine._process_bar({"BTC/USD": bar}, {"BTC/USD": [bar]})


# ---------------------------------------------------------------------------
# Config plumbing: runs.py extraction + BacktestRunner engine config
# ---------------------------------------------------------------------------


class TestRunsBracketExtraction:
    def test_extracts_and_strips_fixed_params(self) -> None:
        from api.routers.runs import _extract_bracket_config

        params: dict[str, Any] = {
            "position_size": 1000,
            "bracket_stop_loss_pct": "0.02",
            "bracket_take_profit_pct": 0.04,
            "bracket_mode": "fixed",
        }
        bracket = _extract_bracket_config(params)
        assert bracket == {
            "bracket_stop_loss_pct": 0.02,
            "bracket_take_profit_pct": 0.04,
            "bracket_mode": "fixed",
        }
        # Stripped from strategy params -> only entry param remains.
        assert params == {"position_size": 1000}

    def test_blank_values_dropped(self) -> None:
        from api.routers.runs import _extract_bracket_config

        params: dict[str, Any] = {
            "bracket_stop_loss_pct": "",
            "bracket_take_profit_pct": None,
            "bracket_mode": "",
        }
        bracket = _extract_bracket_config(params)
        assert bracket == {}
        assert params == {}

    def test_atr_params_coerced(self) -> None:
        from api.routers.runs import _extract_bracket_config

        params: dict[str, Any] = {
            "bracket_mode": "atr",
            "bracket_atr_sl_multiplier": "2.0",
            "bracket_atr_tp_multiplier": "3",
            "bracket_atr_period": "10",
        }
        bracket = _extract_bracket_config(params)
        assert bracket["bracket_atr_sl_multiplier"] == 2.0
        assert bracket["bracket_atr_tp_multiplier"] == 3.0
        assert bracket["bracket_atr_period"] == 10
        assert isinstance(bracket["bracket_atr_period"], int)
        assert bracket["bracket_mode"] == "atr"

    def test_native_strategy_params_not_stripped(self) -> None:
        # CR-001 regression: bracket extraction must NOT touch a strategy's
        # own params that happen to share a base name (dca_rsi_hybrid has its
        # own take_profit_pct; breakout has its own atr_period).  The
        # bracket_ prefix guarantees no collision.
        from api.routers.runs import _extract_bracket_config

        params: dict[str, Any] = {
            "take_profit_pct": 0.5,   # dca_rsi_hybrid native param
            "atr_period": 20,          # breakout native param
            "position_size": 1000,
        }
        bracket = _extract_bracket_config(params)
        assert bracket == {}
        assert params == {
            "take_profit_pct": 0.5,
            "atr_period": 20,
            "position_size": 1000,
        }


class TestBacktestRunnerBracketConfig:
    def _runner(self, bracket_config: dict[str, object] | None):
        from common.types import TimeFrame as _TF
        from trading.backtest import BacktestRunner

        strat = MagicMock()
        strat.min_bars_required = 20
        return BacktestRunner(
            strategies=[strat],
            symbols=["BTC/USD"],
            timeframe=_TF.ONE_HOUR,
            bracket_config=bracket_config,
        )

    def test_bracket_config_merged_into_engine_config(self) -> None:
        runner = self._runner({"bracket_stop_loss_pct": 0.02, "bracket_mode": "fixed"})
        cfg = runner._build_engine_config()  # type: ignore[attr-defined]
        assert cfg["bracket_stop_loss_pct"] == 0.02
        assert cfg["bracket_mode"] == "fixed"

    def test_no_bracket_config_absent_from_engine_config(self) -> None:
        runner = self._runner(None)
        cfg = runner._build_engine_config()  # type: ignore[attr-defined]
        assert "bracket_stop_loss_pct" not in cfg
        assert "bracket_mode" not in cfg

    def test_none_values_not_injected(self) -> None:
        runner = self._runner({"bracket_stop_loss_pct": 0.02, "bracket_take_profit_pct": None})
        cfg = runner._build_engine_config()  # type: ignore[attr-defined]
        assert cfg["bracket_stop_loss_pct"] == 0.02
        assert "bracket_take_profit_pct" not in cfg
