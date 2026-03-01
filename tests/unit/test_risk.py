"""
tests/unit/test_risk.py
------------------------
Unit tests for the risk management layer.

Modules under test
------------------
- packages/trading/risk.py       — BaseRiskManager, RiskParameters, helpers
- packages/trading/risk_manager.py — DefaultRiskManager implementation

Test coverage  (44 tests total)
---------------------------------
- Kill switch blocks ALL orders
- Cooldown period blocks orders until expiry
- FIX-12 regression: consecutive_losses resets when cooldown expires
- Max open positions enforcement
- Daily loss limit
- Drawdown limit
- Order size / notional cap (warning, not block)
- Notional cap reduced to zero → blocking rejection
- Concentration cap (blocking when fully saturated)
- Concentration cap (partial room → quantity reduced, not rejected)
- Position sizing formula: risk_amount / (entry * distance)
- Position sizing caps: max_order_size_quote, max_position_size_pct
- Position sizing: tight stop-loss distance floored at 0.1%
- Position sizing: negative equity guard returns zero
- Position sizing: concentration cap dominates when tighter than risk formula
- update_after_fill: win resets streak; loss increments streak; streak triggers cooldown
- pre_trade_check: price unknown blocks MARKET order
- pre_trade_check: market_price used when order.price is None
- pre_trade_check: price resolved from existing position.current_price
- pre_trade_check: multiple simultaneous violations all collected
- RiskParameters validation: per_trade_risk_pct and max_drawdown_pct bounds
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType
from trading.models import Order, Position
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides) -> RiskParameters:
    """Build RiskParameters with sensible test defaults, allowing overrides."""
    defaults = {
        "max_open_positions": 3,
        "max_position_size_pct": 0.10,
        "per_trade_risk_pct": 0.01,
        "max_order_size_quote": Decimal("10000"),
        "max_daily_loss_pct": 0.05,
        "max_drawdown_pct": 0.15,
        "taker_fee_pct": 0.001,
        "maker_fee_pct": 0.0005,
        "slippage_bps": 5,
        "cooldown_after_loss_streak": 3,
        "loss_streak_count": 3,
    }
    defaults.update(overrides)
    return RiskParameters(**defaults)


def _make_manager(run_id: str = "test-run", **param_overrides) -> DefaultRiskManager:
    """Build a DefaultRiskManager with test parameters."""
    return DefaultRiskManager(run_id=run_id, params=_make_params(**param_overrides))


def _make_limit_order(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.01"),
    price: Decimal = Decimal("50000"),
    run_id: str = "test-run",
) -> Order:
    """Build a LIMIT order suitable for pre_trade_check."""
    return Order(
        client_order_id=f"{run_id}-{uuid4().hex[:12]}",
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
    )


def _make_market_order(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.01"),
    run_id: str = "test-run",
) -> Order:
    """Build a MARKET order (price=None) suitable for pre_trade_check."""
    return Order(
        client_order_id=f"{run_id}-{uuid4().hex[:12]}",
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=None,
    )


def _make_position(
    *,
    symbol: str = "BTC/USDT",
    quantity: Decimal = Decimal("0.1"),
    current_price: Decimal = Decimal("50000"),
    run_id: str = "test-run",
) -> Position:
    """Build an open position snapshot."""
    return Position(
        symbol=symbol,
        run_id=run_id,
        quantity=quantity,
        average_entry_price=current_price,
        current_price=current_price,
    )


# ===========================================================================
# Kill switch
# ===========================================================================


class TestKillSwitch:
    """Tests for kill-switch activation and reset."""

    def test_kill_switch_blocks_all_orders(self) -> None:
        """After triggering the kill switch, pre_trade_check rejects everything."""
        manager = _make_manager()
        manager.trigger_kill_switch("test halt")
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False
        assert any("kill switch" in r.lower() for r in result.rejection_reasons)

    def test_kill_switch_active_property(self) -> None:
        """kill_switch_active property reflects current state."""
        manager = _make_manager()
        assert manager.kill_switch_active is False
        manager.trigger_kill_switch("reason")
        assert manager.kill_switch_active is True

    def test_kill_switch_reset_allows_orders(self) -> None:
        """After resetting the kill switch, orders are approved again."""
        manager = _make_manager()
        manager.trigger_kill_switch("temporary halt")
        manager.reset_kill_switch()
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True

    def test_kill_switch_is_not_auto_cleared(self) -> None:
        """Kill switch stays active until explicitly reset."""
        manager = _make_manager()
        manager.trigger_kill_switch("persistent halt")
        order = _make_limit_order()
        # Simulate multiple ticks — kill switch should persist
        for _ in range(5):
            manager.tick_cooldown()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False


# ===========================================================================
# Cooldown
# ===========================================================================


class TestCooldown:
    """Tests for cooldown period management."""

    def test_cooldown_blocks_orders_during_period(self) -> None:
        """Orders are rejected while cooldown_bars_remaining > 0."""
        manager = _make_manager(cooldown_after_loss_streak=3, loss_streak_count=1)
        # Trigger cooldown via update_after_fill
        manager.update_after_fill(Decimal("-100"), is_loss=True)
        assert manager.in_cooldown is True
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False
        assert any("cooldown" in r.lower() for r in result.rejection_reasons)

    def test_cooldown_expires_after_correct_number_of_ticks(self) -> None:
        """Cooldown expires exactly after cooldown_after_loss_streak ticks."""
        cooldown_bars = 3
        manager = _make_manager(
            cooldown_after_loss_streak=cooldown_bars,
            loss_streak_count=1,
        )
        manager.update_after_fill(Decimal("-100"), is_loss=True)
        assert manager.in_cooldown is True
        for _ in range(cooldown_bars):
            manager.tick_cooldown()
        assert manager.in_cooldown is False

    def test_cooldown_orders_approved_after_expiry(self) -> None:
        """Orders are approved once the cooldown has fully expired."""
        cooldown_bars = 2
        manager = _make_manager(
            cooldown_after_loss_streak=cooldown_bars,
            loss_streak_count=1,
        )
        manager.update_after_fill(Decimal("-50"), is_loss=True)
        for _ in range(cooldown_bars):
            manager.tick_cooldown()
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True

    def test_fix_12_cooldown_expiry_resets_consecutive_losses(self) -> None:
        """
        FIX-12 regression: when cooldown expires, consecutive_losses must
        reset to 0 so that a single subsequent loss does NOT immediately
        re-trigger another cooldown.

        Before FIX-12, consecutive_losses was not reset on expiry,
        causing the next single loss to instantly re-trigger cooldown
        (because the streak count already stood at loss_streak_count).
        """
        cooldown_bars = 2
        streak_count = 2
        manager = _make_manager(
            cooldown_after_loss_streak=cooldown_bars,
            loss_streak_count=streak_count,
        )

        # Trigger cooldown by reaching streak threshold
        for _ in range(streak_count):
            manager.update_after_fill(Decimal("-50"), is_loss=True)
        assert manager.in_cooldown is True

        # Let cooldown expire
        for _ in range(cooldown_bars):
            manager.tick_cooldown()
        assert manager.in_cooldown is False

        # ONE loss after cooldown expiry should NOT immediately re-trigger cooldown
        manager.update_after_fill(Decimal("-30"), is_loss=True)
        assert manager.in_cooldown is False, (
            "FIX-12 regression: single loss after cooldown expiry "
            "must not re-trigger cooldown immediately"
        )

    def test_tick_cooldown_without_active_cooldown_is_noop(self) -> None:
        """Calling tick_cooldown when not in cooldown is safe (no-op)."""
        manager = _make_manager()
        manager.tick_cooldown()  # should not raise or change state
        assert manager.in_cooldown is False


# ===========================================================================
# Max open positions
# ===========================================================================


class TestMaxPositions:
    """Tests for the max open positions check."""

    def test_at_max_positions_blocks_new_order(self) -> None:
        """When max_open_positions is reached, orders are rejected."""
        manager = _make_manager(max_open_positions=2)
        open_positions = [
            _make_position(symbol="BTC/USDT"),
            _make_position(symbol="ETH/USDT"),
        ]
        order = _make_limit_order(symbol="SOL/USDT")
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=open_positions,
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False
        assert any("max open positions" in r.lower() for r in result.rejection_reasons)

    def test_below_max_positions_allows_order(self) -> None:
        """Orders are allowed when below max_open_positions."""
        manager = _make_manager(max_open_positions=3)
        open_positions = [_make_position(symbol="BTC/USDT")]
        order = _make_limit_order(symbol="ETH/USDT")
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=open_positions,
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True

    def test_flat_positions_not_counted(self) -> None:
        """Positions with quantity=0 (flat) do not count toward the limit."""
        manager = _make_manager(max_open_positions=1)
        flat_position = _make_position(symbol="BTC/USDT", quantity=Decimal("0"))
        order = _make_limit_order(symbol="ETH/USDT")
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[flat_position],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True


# ===========================================================================
# Daily loss limit
# ===========================================================================


class TestDailyLossLimit:
    """Tests for the daily loss circuit breaker."""

    def test_daily_loss_at_limit_blocks_order(self) -> None:
        """
        Daily PnL at or below -(max_daily_loss_pct * equity) blocks orders.
        With equity=10_000 and limit=5%, threshold=-500.
        """
        manager = _make_manager(max_daily_loss_pct=0.05)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("-501"),  # exceeds -500 threshold
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False
        assert any("daily loss" in r.lower() for r in result.rejection_reasons)

    def test_daily_loss_below_limit_allows_order(self) -> None:
        """Daily loss within limit does not block orders."""
        manager = _make_manager(max_daily_loss_pct=0.05)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("-400"),  # within 500 threshold
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True

    def test_positive_daily_pnl_always_allows_order(self) -> None:
        """Positive daily PnL never triggers the daily loss check."""
        manager = _make_manager(max_daily_loss_pct=0.05)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("500"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True


# ===========================================================================
# Drawdown limit
# ===========================================================================


class TestDrawdownLimit:
    """Tests for the maximum drawdown circuit breaker."""

    def test_drawdown_at_limit_blocks_order(self) -> None:
        """
        Drawdown >= max_drawdown_pct blocks orders.
        Peak=10_000, current=8_400 → drawdown=16% >= 15%.
        """
        manager = _make_manager(max_drawdown_pct=0.15)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("8400"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),  # 16% drawdown
        )
        assert result.approved is False
        assert any("drawdown" in r.lower() for r in result.rejection_reasons)

    def test_drawdown_below_limit_allows_order(self) -> None:
        """Drawdown below threshold does not block orders."""
        manager = _make_manager(max_drawdown_pct=0.15)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("9000"),  # 10% drawdown < 15%
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
        )
        assert result.approved is True

    def test_zero_peak_equity_skips_drawdown_check(self) -> None:
        """A peak_equity of 0 means the check cannot run — no violation."""
        manager = _make_manager(max_drawdown_pct=0.15)
        order = _make_limit_order()
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("0"),  # no peak — check should be skipped
        )
        assert result.approved is True


# ===========================================================================
# Order size cap
# ===========================================================================


class TestOrderSizeCap:
    """Tests for the notional order-size and concentration caps."""

    def test_order_exceeding_notional_cap_is_reduced(self) -> None:
        """
        An order with notional value above max_order_size_quote is
        approved but with adjusted_quantity reduced to fit the cap.
        """
        manager = _make_manager(max_order_size_quote=Decimal("500"))
        # 0.02 BTC at 50000 USDT = 1000 USDT > 500 USDT cap
        order = _make_limit_order(
            quantity=Decimal("0.02"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),  # large enough to avoid other checks
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
        )
        # Should be approved (reduced, not rejected)
        assert result.approved is True
        # adjusted_quantity should be <= 500/50000 = 0.01
        assert result.adjusted_quantity <= Decimal("0.01")

    def test_order_within_notional_cap_unchanged(self) -> None:
        """An order within the notional cap is approved unchanged."""
        manager = _make_manager(max_order_size_quote=Decimal("10000"))
        # 0.01 BTC at 50000 = 500 USDT << 10000 cap
        order = _make_limit_order(
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
        )
        assert result.approved is True
        assert result.adjusted_quantity == Decimal("0.01")


# ===========================================================================
# Position sizing
# ===========================================================================


class TestPositionSizing:
    """Tests for calculate_position_size."""

    def test_basic_sizing_with_stop_loss(self) -> None:
        """
        With concentration cap disabled (max_position_size_pct=1.0) and a
        large order cap, the formula dominates:

        risk_amount = 10000 * 0.01 * 1.0 = 100 USDT
        distance = |50000 - 49000| / 50000 = 2%
        size = 100 / (50000 * 0.02) = 0.1 BTC
        """
        manager = _make_manager(
            per_trade_risk_pct=0.01,
            max_order_size_quote=Decimal("100000"),   # non-binding cap
            max_position_size_pct=1.0,               # non-binding concentration cap
        )
        size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            confidence=1.0,
        )
        expected = Decimal("100") / (Decimal("50000") * Decimal("0.02"))
        assert abs(size - expected) < Decimal("0.00000001")

    def test_sizing_without_stop_loss_uses_default_distance(self) -> None:
        """
        When stop_loss_price is None, default distance = 1%.
        size = (10000 * 0.01) / (50000 * 0.01) = 0.2 BTC
        Subject to caps.
        """
        manager = _make_manager(
            per_trade_risk_pct=0.01,
            max_order_size_quote=Decimal("100000"),
            max_position_size_pct=1.0,  # set high to avoid cap interference
        )
        size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        expected = Decimal("100") / (Decimal("50000") * Decimal("0.01"))
        assert abs(size - expected) < Decimal("0.0001")

    def test_confidence_scales_position_size(self) -> None:
        """Half confidence produces half the position size."""
        manager = _make_manager(
            per_trade_risk_pct=0.01,
            max_order_size_quote=Decimal("100000"),
            max_position_size_pct=1.0,
        )
        full_size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        half_size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=0.5,
        )
        assert abs(full_size / 2 - half_size) < Decimal("0.00001")

    def test_zero_equity_returns_zero(self) -> None:
        """Zero equity guard returns Decimal(0)."""
        manager = _make_manager()
        size = manager.calculate_position_size(
            equity=Decimal("0"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        assert size == Decimal("0")

    def test_zero_entry_price_returns_zero(self) -> None:
        """Zero entry price guard returns Decimal(0)."""
        manager = _make_manager()
        size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("0"),
            stop_loss_price=None,
            confidence=1.0,
        )
        assert size == Decimal("0")

    def test_size_capped_by_max_order_size(self) -> None:
        """Position size is capped by max_order_size_quote / entry_price."""
        manager = _make_manager(
            per_trade_risk_pct=0.05,
            max_order_size_quote=Decimal("100"),  # very tight cap
            max_position_size_pct=1.0,
        )
        size = manager.calculate_position_size(
            equity=Decimal("1000000"),  # large equity → large uncapped size
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        # Cap = 100/50000 = 0.002
        assert size <= Decimal("0.002")

    def test_result_has_8_decimal_precision(self) -> None:
        """Position size is rounded down to 8 decimal places."""
        manager = _make_manager()
        size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            confidence=1.0,
        )
        # Ensure no more than 8 decimal places
        assert size == size.quantize(Decimal("0.00000001"))


# ===========================================================================
# update_after_fill
# ===========================================================================


class TestUpdateAfterFill:
    """Tests for consecutive-loss tracking and cooldown activation."""

    def test_win_resets_consecutive_losses(self) -> None:
        """A winning trade resets the consecutive loss counter."""
        manager = _make_manager(loss_streak_count=3)
        manager.update_after_fill(Decimal("-50"), is_loss=True)
        manager.update_after_fill(Decimal("-50"), is_loss=True)
        # Now 2 consecutive losses
        manager.update_after_fill(Decimal("100"), is_loss=False)
        # Win should reset counter — next 2 losses should NOT trigger cooldown
        manager.update_after_fill(Decimal("-50"), is_loss=True)
        manager.update_after_fill(Decimal("-50"), is_loss=True)
        assert manager.in_cooldown is False

    def test_loss_streak_triggers_cooldown(self) -> None:
        """Reaching loss_streak_count consecutive losses activates cooldown."""
        manager = _make_manager(loss_streak_count=3, cooldown_after_loss_streak=5)
        for _ in range(3):
            manager.update_after_fill(Decimal("-50"), is_loss=True)
        assert manager.in_cooldown is True

    def test_below_streak_threshold_no_cooldown(self) -> None:
        """Streak below threshold does not activate cooldown."""
        manager = _make_manager(loss_streak_count=3, cooldown_after_loss_streak=5)
        for _ in range(2):  # only 2 losses, threshold is 3
            manager.update_after_fill(Decimal("-50"), is_loss=True)
        assert manager.in_cooldown is False


# ===========================================================================
# Position sizing — edge cases (new coverage)
# ===========================================================================


class TestPositionSizingEdgeCases:
    """Edge-case tests for calculate_position_size covering previously uncovered paths."""

    def test_sizing_tight_stop_loss_floored_at_minimum_distance(self) -> None:
        """
        Stop-loss distance < 0.1% is floored at 0.001 to prevent inflated
        position sizes (line 190-191 of risk_manager.py).

        With entry=50000 and stop=49999.99 the raw distance is
        0.0000002, far below the 0.001 floor.  The result must equal
        the size produced when the stop is placed at exactly 0.1% away
        (entry=50000, stop=49950), which uses the same 0.001 distance.
        """
        manager = _make_manager(
            per_trade_risk_pct=0.01,
            max_order_size_quote=Decimal("1000000"),
            max_position_size_pct=1.0,
        )
        # Stop so close that raw distance = 0.0000002 → floored to 0.001
        tight = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49999.99"),
            confidence=1.0,
        )
        # Stop exactly at 0.1% distance → distance = 0.001 (no floor needed)
        normal = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49950"),
            confidence=1.0,
        )
        # Both paths use 0.001 distance, so the results must be identical
        assert tight == normal

    def test_sizing_negative_equity_returns_zero(self) -> None:
        """
        Negative equity triggers the guard at line 177 of risk_manager.py
        (``if equity <= Decimal(0)``) and returns Decimal(0).
        """
        manager = _make_manager()
        size = manager.calculate_position_size(
            equity=Decimal("-1000"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        assert size == Decimal("0")

    def test_sizing_capped_by_concentration_limit(self) -> None:
        """
        When max_position_size_pct * equity / entry_price is the binding
        constraint, the returned size must not exceed that ceiling
        (lines 202-207 of risk_manager.py).

        With equity=10000, entry=50000, and max_position_size_pct=0.01:
            concentration_cap = 0.01 * 10000 / 50000 = 0.002 BTC
        The uncapped risk-formula result (0.2 BTC) is far above this.
        """
        manager = _make_manager(
            per_trade_risk_pct=0.01,
            max_order_size_quote=Decimal("1000000"),  # non-binding
            max_position_size_pct=0.01,               # very tight: 1% of equity
        )
        size = manager.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            confidence=1.0,
        )
        # concentration cap = 0.01 * 10000 / 50000 = 0.002
        assert size <= Decimal("0.002")
        assert size > Decimal("0")


# ===========================================================================
# pre_trade_check — MARKET order price resolution (new coverage)
# ===========================================================================


class TestPreTradeCheckMarketOrderPriceResolution:
    """
    Tests for the _resolve_effective_price fallback chain used when a MARKET
    order has no limit price (lines 286-297, 413-421 of risk_manager.py).
    """

    def test_price_unknown_blocks_market_order(self) -> None:
        """
        When no effective price can be resolved (order.price is None, no
        matching open position, and market_price is None), the order must
        be blocked with a price-related rejection reason (lines 286-297).
        """
        manager = _make_manager()
        order = _make_market_order(
            symbol="BTC/USDT",
            quantity=Decimal("0.01"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("10000"),
            market_price=None,
        )
        assert result.approved is False
        assert any(
            "price" in r.lower() for r in result.rejection_reasons
        ), f"Expected price-related rejection, got: {result.rejection_reasons}"

    def test_market_price_used_when_order_has_no_price(self) -> None:
        """
        When order.price is None and no open position exists for the symbol,
        the caller-supplied market_price is used (line 420-421 of risk_manager.py).
        A small, well-within-cap order must be approved.
        """
        manager = _make_manager(
            max_order_size_quote=Decimal("100000"),
            max_position_size_pct=1.0,
            max_open_positions=5,
        )
        order = _make_market_order(
            symbol="BTC/USDT",
            quantity=Decimal("0.001"),  # 0.001 * 50000 = 50 USDT — well within caps
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
            market_price=Decimal("50000"),
        )
        assert result.approved is True

    def test_price_resolved_from_existing_position_current_price(self) -> None:
        """
        When order.price is None but an open position exists for the same
        symbol with a valid current_price, that price is used instead of
        market_price (line 416-418 of risk_manager.py).
        A small order must be approved.
        """
        manager = _make_manager(
            max_order_size_quote=Decimal("100000"),
            max_position_size_pct=1.0,
            max_open_positions=5,
        )
        # Existing open position provides the price reference
        existing = _make_position(symbol="BTC/USDT", current_price=Decimal("50000"))
        order = _make_market_order(
            symbol="BTC/USDT",
            quantity=Decimal("0.001"),  # 50 USDT notional
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
            market_price=None,  # no market_price; should fall back to position price
        )
        assert result.approved is True


# ===========================================================================
# Concentration cap — additional edge cases (new coverage)
# ===========================================================================


class TestConcentrationCapEdgeCases:
    """
    Edge cases for the per-symbol concentration cap inside _check_order_size
    (lines 334-394 of risk_manager.py).
    """

    def test_existing_position_at_ceiling_blocks_additional_order(self) -> None:
        """
        When the existing position already equals the concentration ceiling,
        remaining_value == 0, so the new order is blocked (lines 349-363).

        Setup:
            equity = 100 000 USDT
            max_position_size_pct = 10%  → cap = 10 000 USDT
            existing: 0.2 BTC * 50 000 = 10 000 USDT  (exactly at cap)
        Any additional quantity has zero room and must be rejected.
        """
        manager = _make_manager(
            max_position_size_pct=0.10,
            max_order_size_quote=Decimal("100000"),
            max_open_positions=5,
        )
        existing = _make_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.2"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="BTC/USDT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
        )
        assert result.approved is False
        assert any(
            "concentration" in r.lower() for r in result.rejection_reasons
        ), f"Expected concentration rejection, got: {result.rejection_reasons}"

    def test_partial_concentration_room_reduces_quantity(self) -> None:
        """
        When some room remains under the concentration cap, quantity is
        reduced to fit (lines 365-394 of risk_manager.py).  The order is
        approved (warning, not block) with a smaller adjusted_quantity.

        Setup:
            equity = 100 000 USDT
            max_position_size_pct = 10%  → cap = 10 000 USDT
            existing: 0.16 BTC * 50 000 = 8 000 USDT  (room = 2 000)
            order:    0.1  BTC * 50 000 = 5 000 USDT  (exceeds room)
        Expected: approved=True, adjusted_quantity <= 0.04 BTC (2000/50000)
        """
        manager = _make_manager(
            max_position_size_pct=0.10,
            max_order_size_quote=Decimal("100000"),
            max_open_positions=5,
        )
        existing = _make_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.16"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
        )
        assert result.approved is True
        # Room = 2000 USDT → max quantity = 2000 / 50000 = 0.04 BTC
        assert result.adjusted_quantity <= Decimal("0.04")
        assert result.adjusted_quantity > Decimal("0")


# ===========================================================================
# Notional cap → zero (new coverage)
# ===========================================================================


class TestNotionalCapToZero:
    """Tests for the blocking path when notional cap forces quantity to zero."""

    def test_notional_cap_shrinks_quantity_to_zero_blocks_order(self) -> None:
        """
        When max_order_size_quote is so tight that quantizing the capped
        quantity to 8 decimal places yields 0, the order is blocked with a
        descriptive rejection reason (lines 307-319 of risk_manager.py).

        max_order_size_quote = 0.000000001 USDT (1 nano-USDT)
        capped_qty = 1e-9 / 50000 = 2e-14
        2e-14 rounds DOWN to 0.00000000 at 8 d.p. → blocking.

        Note: 0.001 USDT would give 0.00000002 BTC (2E-8) which is a valid
        8-decimal quantity and only generates a warning, not a block.
        """
        manager = _make_manager(
            max_order_size_quote=Decimal("0.000000001"),  # 1 nano-USDT: rounds to zero at 8 d.p.
            max_open_positions=5,
        )
        order = _make_limit_order(
            quantity=Decimal("1"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("100000"),
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=Decimal("100000"),
        )
        assert result.approved is False
        assert any(
            "max_order_size" in r.lower()
            or "notional" in r.lower()
            or "cannot be reduced" in r.lower()
            for r in result.rejection_reasons
        ), f"Expected notional-cap blocking reason, got: {result.rejection_reasons}"


# ===========================================================================
# Multiple simultaneous violations (new coverage)
# ===========================================================================


class TestMultipleSimultaneousViolations:
    """
    Verify that pre_trade_check collects ALL blocking violations rather than
    short-circuiting after the first one (lines 83-114 of risk_manager.py).
    """

    def test_multiple_violations_all_collected(self) -> None:
        """
        When kill switch, cooldown, max_positions, and daily_loss all fire
        simultaneously, every violation must appear in rejection_reasons.

        Setup:
            - kill switch triggered manually
            - cooldown active (loss_streak_count=1 → one loss triggers it)
            - max_open_positions=1 with one existing position
            - daily_pnl=-1000 vs threshold of -(0.05 * 10000) = -500
        Expected: approved=False, at least 4 rejection reasons.
        """
        manager = _make_manager(
            max_open_positions=1,
            max_daily_loss_pct=0.05,
            cooldown_after_loss_streak=5,
            loss_streak_count=1,
        )
        # Activate kill switch
        manager.trigger_kill_switch("test halt")
        # Activate cooldown via one loss (loss_streak_count=1 means threshold is 1)
        manager.update_after_fill(Decimal("-100"), is_loss=True)
        assert manager.in_cooldown is True

        order = _make_limit_order()
        existing = _make_position(symbol="ETH/USDT")  # fills the single slot

        result = manager.pre_trade_check(
            order=order,
            current_equity=Decimal("10000"),
            open_positions=[existing],
            daily_pnl=Decimal("-1000"),  # exceeds -500 threshold
            peak_equity=Decimal("10000"),
        )
        assert result.approved is False
        # All four checks must have fired: kill_switch + cooldown + max_positions + daily_loss
        assert len(result.rejection_reasons) >= 4, (
            f"Expected >= 4 rejection reasons, got {len(result.rejection_reasons)}: "
            f"{result.rejection_reasons}"
        )


# ===========================================================================
# RiskParameters validation (new coverage)
# ===========================================================================


class TestRiskParametersValidation:
    """
    Tests for the __post_init__ validator in RiskParameters
    (lines 69-81 of risk.py).
    """

    def test_per_trade_risk_pct_above_maximum_raises(self) -> None:
        """
        per_trade_risk_pct > 0.05 (5%) is unsafe and must raise ValueError.
        The error message must identify the field name.
        """
        with pytest.raises(ValueError, match="per_trade_risk_pct"):
            RiskParameters(per_trade_risk_pct=0.10)

    def test_per_trade_risk_pct_zero_raises(self) -> None:
        """
        per_trade_risk_pct == 0 is not allowed (must be strictly positive).
        """
        with pytest.raises(ValueError, match="per_trade_risk_pct"):
            RiskParameters(per_trade_risk_pct=0.0)

    def test_max_drawdown_pct_above_maximum_raises(self) -> None:
        """
        max_drawdown_pct > 0.50 (50%) is unsafe and must raise ValueError.
        The error message must identify the field name.
        """
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            RiskParameters(max_drawdown_pct=0.60)

    def test_valid_boundary_values_do_not_raise(self) -> None:
        """
        Boundary values at exactly the allowed maximums must not raise.
        per_trade_risk_pct=0.05 and max_drawdown_pct=0.50 are both in range.
        """
        params = RiskParameters(per_trade_risk_pct=0.05, max_drawdown_pct=0.50)
        assert params.per_trade_risk_pct == 0.05
        assert params.max_drawdown_pct == 0.50
