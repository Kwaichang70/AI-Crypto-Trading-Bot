"""
tests/unit/test_risk.py
------------------------
Unit tests for the risk management layer.

Modules under test
------------------
- packages/trading/risk.py       — BaseRiskManager, RiskParameters, helpers
- packages/trading/risk_manager.py — DefaultRiskManager implementation

Test coverage  (48 tests total)
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
- SELL orders bypass concentration cap; BUY orders still blocked (bug-fix regression)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide, OrderStatus, OrderType
from trading.models import Order, Position
from trading.risk import RiskParameters, symbol_cluster
from trading.risk_manager import DefaultRiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides) -> RiskParameters:
    """Build RiskParameters with sensible test defaults, allowing overrides."""
    defaults = {
        "max_open_positions": 3,
        "max_position_size_pct": 0.10,
        "max_portfolio_exposure_pct": 1.0,  # disabled in unit tests unless explicitly overridden
        "per_trade_risk_pct": 0.01,
        "max_order_size_quote": Decimal("10000"),
        "max_daily_loss_pct": 0.05,
        "max_drawdown_pct": 0.15,
        "taker_fee_pct": 0.001,
        "maker_fee_pct": 0.0005,
        "slippage_bps": 5,
        "cooldown_after_loss_streak": 3,
        "loss_streak_count": 3,
        # QT-003 (Sprint 46): disable cluster cap by default so tests that
        # don't explicitly configure it are not constrained by the new gate.
        # Mirrors max_portfolio_exposure_pct=1.0 above.
        "max_cluster_exposure_pct": 1.0,
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
        """Orders are allowed when below max_open_positions.

        QT-003 (Sprint 46): position quantity reduced from 0.1 BTC to
        0.001 BTC so the existing exposure (50 USDT) stays well below the
        new cluster cap (40 % of equity); this test asserts
        max_positions semantics, not exposure semantics.
        """
        manager = _make_manager(max_open_positions=3)
        open_positions = [
            _make_position(symbol="BTC/USDT", quantity=Decimal("0.001"))
        ]
        order = _make_limit_order(symbol="ETH/USDT", quantity=Decimal("0.001"))
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


# ===========================================================================
# SELL order concentration bypass (bug-fix regression tests)
# ===========================================================================


class TestSellOrderConcentrationBypass:
    """
    Regression tests for the fix that adds an ``order.side == OrderSide.BUY``
    guard on the concentration cap check (section b of _check_order_size).

    Before the fix, SELL orders were incorrectly blocked by the concentration
    cap when the existing position was at or above the cap.  This trapped
    capital: a trader holding a fully-concentrated position could not exit.

    Test matrix
    -----------
    1. SELL when position is at exactly the concentration ceiling → APPROVED.
    2. SELL when position has grown above the ceiling (price increase) → APPROVED.
    3. BUY when position is at the ceiling → still BLOCKED (regression guard).
    4. SELL with quantity so large it exceeds the absolute notional cap
       (section a) → quantity is capped / order blocked per notional rules,
       confirming section a still applies to SELL orders.
    """

    # Shared risk configuration for all four tests in this class.
    # equity=100 000, cap=10% → concentration ceiling = 10 000 USDT.
    _EQUITY = Decimal("100000")
    _PEAK = Decimal("100000")
    _CAP_PCT = 0.10   # 10% → max position value = 10 000 USDT

    def _make_concentrated_manager(self) -> DefaultRiskManager:
        """Return a manager configured with a 10% concentration cap."""
        return _make_manager(
            max_position_size_pct=self._CAP_PCT,
            max_order_size_quote=Decimal("100000"),  # non-binding for tests 1-3
            max_open_positions=5,
        )

    def _make_position_at_cap(self) -> Position:
        """
        Position whose notional value equals the concentration ceiling exactly.

        0.2 BTC * 50 000 USDT = 10 000 USDT = 10% of 100 000 equity.
        """
        return _make_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.2"),
            current_price=Decimal("50000"),
        )

    def test_sell_bypasses_concentration_cap(self) -> None:
        """
        When an existing BTC position is at exactly the concentration ceiling
        (10 000 USDT), a SELL order for 0.01 BTC must be APPROVED.

        Without the ``order.side == OrderSide.BUY`` guard, the concentration
        check would compute:
            existing_value = 10 000
            proposed_value = 10 000 + (0.01 * 50 000) = 10 500  > cap
        and block the order.  With the fix, SELL orders skip section b entirely.

        Setup:
            equity          = 100 000 USDT
            concentration   = 10% → cap = 10 000 USDT
            existing pos    = 0.2 BTC * 50 000 = 10 000 USDT  (at cap)
            order           = SELL 0.01 BTC @ 50 000
        Expected: approved=True
        """
        manager = self._make_concentrated_manager()
        existing = self._make_position_at_cap()
        order = _make_limit_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"SELL order must bypass concentration cap; "
            f"rejection_reasons={result.rejection_reasons}"
        )

    def test_sell_beyond_concentration_cap_still_approved(self) -> None:
        """
        When a position has grown above the concentration ceiling due to
        unrealised gains (price increased from 50 000 to 60 000), a SELL
        order must still be APPROVED.

        The position's notional value is now 0.2 * 60 000 = 12 000 USDT,
        which is 20% above the 10 000 USDT cap.  Without the fix, ANY
        further buy would be blocked — but more critically the broken code
        also blocked sells because it treated the SELL qty as additive to
        existing_value rather than checking order.side.

        Setup:
            equity          = 100 000 USDT
            concentration   = 10% → cap = 10 000 USDT
            existing pos    = 0.2 BTC * 60 000 = 12 000 USDT  (above cap)
            order           = SELL 0.01 BTC @ 60 000
        Expected: approved=True
        """
        manager = self._make_concentrated_manager()
        # Price has risen: position now exceeds cap
        existing = _make_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.2"),
            current_price=Decimal("60000"),
        )
        order = _make_limit_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            price=Decimal("60000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"SELL order must bypass concentration cap even when position "
            f"exceeds cap; rejection_reasons={result.rejection_reasons}"
        )

    def test_buy_still_blocked_by_concentration_cap(self) -> None:
        """
        Regression guard: BUY orders must still be blocked by the
        concentration cap when the existing position is at the ceiling.

        This test ensures the ``order.side == OrderSide.BUY`` guard did NOT
        accidentally disable concentration enforcement for BUY orders.

        Setup:
            equity          = 100 000 USDT
            concentration   = 10% → cap = 10 000 USDT
            existing pos    = 0.2 BTC * 50 000 = 10 000 USDT  (at cap)
            order           = BUY 0.01 BTC @ 50 000
        Expected: approved=False, rejection contains "concentration"
        """
        manager = self._make_concentrated_manager()
        existing = self._make_position_at_cap()
        order = _make_limit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is False, (
            "BUY order must still be blocked by the concentration cap"
        )
        assert any(
            "concentration" in r.lower() for r in result.rejection_reasons
        ), (
            f"Expected concentration rejection for BUY, "
            f"got: {result.rejection_reasons}"
        )

    def test_sell_still_checked_by_notional_cap(self) -> None:
        """
        SELL orders bypass the concentration cap (section b) but are still
        subject to the absolute notional cap (section a).

        With max_order_size_quote = 0.000000001 USDT (1 nano-USDT) any
        meaningful SELL quantity has its notional value quantized down to
        zero at 8 decimal places, which triggers the blocking path in
        section a (lines 307-323 of risk_manager.py).

        Setup:
            equity              = 100 000 USDT
            max_order_size_quote = 0.000000001 USDT  (1 nano-USDT)
            order               = SELL 0.5 BTC @ 50 000  (25 000 USDT notional)
        Expected: approved=False, rejection contains notional-cap language.

        This confirms section a still runs for SELL orders, preserving the
        absolute order-size safety net regardless of direction.
        """
        manager = _make_manager(
            max_position_size_pct=self._CAP_PCT,
            max_order_size_quote=Decimal("0.000000001"),  # 1 nano-USDT: forces qty to 0
            max_open_positions=5,
        )
        existing = self._make_position_at_cap()
        order = _make_limit_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is False, (
            "SELL order must be blocked by the notional cap when "
            "max_order_size_quote forces quantity to zero"
        )
        assert any(
            "max_order_size" in r.lower()
            or "notional" in r.lower()
            or "cannot be reduced" in r.lower()
            for r in result.rejection_reasons
        ), (
            f"Expected notional-cap blocking reason for SELL, "
            f"got: {result.rejection_reasons}"
        )


# ===================================================================
# TestATRPositionSizing — Sprint 42 QT-002
# ===================================================================


class TestATRPositionSizing:
    """QT-002: ATR-scaled position sizing.  ``sizing_mode="atr"`` derives
    the stop distance from ATR * multiplier so quieter symbols get
    larger positions and more volatile ones get smaller.  Default
    behaviour (``sizing_mode="fixed"``) is preserved bit-for-bit."""

    def test_fixed_mode_unchanged_when_no_atr(self) -> None:
        """Existing callers (atr_value omitted) get identical sizing."""
        params = RiskParameters(per_trade_risk_pct=0.02)
        rm = DefaultRiskManager(run_id="r-fixed", params=params)

        qty = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("98"),  # 2% stop distance
            confidence=1.0,
        )
        # risk_amount = 10000 * 0.02 * 1.0 = 200
        # distance = 0.02; size = 200 / (100 * 0.02) = 100 base units
        # max_concentration_cap = 0.15 * 10000 / 100 = 15
        # final = min(100, 15) = 15
        assert qty == Decimal("15.00000000")

    def test_atr_mode_scales_distance_by_volatility(self) -> None:
        """Higher ATR → wider stop → SMALLER position size."""
        params = RiskParameters(
            per_trade_risk_pct=0.02,
            sizing_mode="atr",
            atr_risk_multiplier=Decimal("1.5"),
            max_position_size_pct=0.99,   # disable concentration cap
            max_order_size_quote=Decimal("1000000"),
        )
        rm = DefaultRiskManager(run_id="r-atr", params=params)

        # ATR = $3 on $100 entry → distance = 1.5 * 3 / 100 = 0.045 (4.5%)
        # risk_amount = 10000 * 0.02 * 1.0 = 200
        # size = 200 / (100 * 0.045) = 44.44... base units
        qty_high_vol = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=None,
            confidence=1.0,
            atr_value=Decimal("3"),
        )

        # ATR = $1 → distance = 0.015 → size = 200 / 1.5 = 133.33...
        qty_low_vol = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=None,
            confidence=1.0,
            atr_value=Decimal("1"),
        )

        assert qty_low_vol > qty_high_vol, (
            "Lower volatility (smaller ATR) MUST yield larger position size"
        )

    def test_atr_mode_falls_back_to_fixed_when_atr_missing(self) -> None:
        """sizing_mode=atr without atr_value uses the legacy fixed path."""
        params = RiskParameters(
            sizing_mode="atr",
            atr_risk_multiplier=Decimal("1.5"),
        )
        rm = DefaultRiskManager(run_id="r-atr-fallback", params=params)

        qty_fallback = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("98"),
            confidence=1.0,
            atr_value=None,
        )
        # Same as the fixed-mode test above — fallback to stop_loss_price.
        assert qty_fallback == Decimal("15.00000000")

    def test_atr_mode_clamps_zero_atr(self) -> None:
        """ATR=0 (degenerate, e.g. warmup period) must not blow up sizing."""
        params = RiskParameters(
            sizing_mode="atr",
            atr_risk_multiplier=Decimal("1.5"),
        )
        rm = DefaultRiskManager(run_id="r-atr-zero", params=params)
        qty = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("98"),
            confidence=1.0,
            atr_value=Decimal("0"),
        )
        # ATR=0 → falls back to stop_loss_price path
        assert qty == Decimal("15.00000000")

    def test_atr_distance_floor_protects_against_near_zero_atr(self) -> None:
        """Tiny ATR (illiquid symbol) is clamped to the 0.1% floor so
        position size cannot be inflated to absurd levels."""
        params = RiskParameters(
            sizing_mode="atr",
            atr_risk_multiplier=Decimal("1.5"),
            max_position_size_pct=0.99,
            max_order_size_quote=Decimal("1000000"),
        )
        rm = DefaultRiskManager(run_id="r-atr-floor", params=params)
        # ATR=0.001 on $100 → raw distance = 1.5 * 0.001 / 100 = 1.5e-5
        # → clamped to 0.001 (0.1%)
        qty = rm.calculate_position_size(
            equity=Decimal("10000"),
            entry_price=Decimal("100"),
            stop_loss_price=None,
            confidence=1.0,
            atr_value=Decimal("0.001"),
        )
        # With distance clamped to 0.001:
        # size_from_risk = 200 / (100 * 0.001) = 2000
        # max_concentration_cap = 0.99 * 10000 / 100 = 99
        # final = min(2000, 99) = 99 — concentration cap saves the day
        # even when the ATR floor itself is permissive.
        assert qty == Decimal("99.00000000")

    def test_invalid_sizing_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="sizing_mode must be"):
            RiskParameters(sizing_mode="kelly")

    def test_invalid_atr_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="atr_risk_multiplier must be > 0"):
            RiskParameters(atr_risk_multiplier=Decimal("0"))

    def test_unreasonable_atr_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="unreasonably large"):
            RiskParameters(atr_risk_multiplier=Decimal("10"))


# ===================================================================
# QT-003 (Sprint 46) — Correlation-aware cluster exposure cap
# ===================================================================


class TestQT003ClusterExposureCap:
    """QT-003: cluster-aware exposure cap.  A nominally "diversified"
    portfolio of four crypto majors at rho>0.8 is effectively one beta
    bet, so the cluster cap bites BEFORE the total portfolio cap when
    one cluster fills up.  These tests cover:

      1. symbol_cluster() lookup (direct unit test)
      2. Same-cluster prospective exposure blocks the order
      3. Same-cluster order exactly at cap is approved (strict >)
      4. Cross-cluster order is approved (cluster cap doesn't bite)
      5. SELL orders bypass the cluster check (matches total-cap behaviour)
      6. Unknown base asset lands in the 'other' cluster
      7. Order with no resolvable price skips the cluster check
      8. Same-symbol current_price fallback for MARKET orders
      9. RiskParameters validation: cluster cap range, cluster <= total cap
    """

    # Shared baseline: 10 000 equity, 60% total cap, 40% cluster cap,
    # concentration + position-count caps disabled so the cluster
    # check is the ONLY blocking gate.
    _EQUITY = Decimal("10000")
    _PEAK = Decimal("10000")

    def _make_cluster_manager(self) -> DefaultRiskManager:
        return _make_manager(
            max_portfolio_exposure_pct=0.60,
            max_cluster_exposure_pct=0.40,   # cluster cap = 4000 USDT
            max_position_size_pct=0.99,      # disable concentration cap
            max_order_size_quote=Decimal("1000000"),
            max_open_positions=20,
        )

    # ----- 1. symbol_cluster() lookup --------------------------------

    def test_symbol_cluster_btc_is_crypto_majors(self) -> None:
        assert symbol_cluster("BTC/USDT") == "crypto_majors"
        assert symbol_cluster("ETH/EUR") == "crypto_majors"
        assert symbol_cluster("SOL/USD") == "crypto_majors"

    def test_symbol_cluster_stablecoin(self) -> None:
        assert symbol_cluster("USDT/USD") == "stablecoin"
        assert symbol_cluster("USDC/EUR") == "stablecoin"

    def test_symbol_cluster_unknown_falls_back_to_other(self) -> None:
        assert symbol_cluster("SHIB/USDT") == "other"
        assert symbol_cluster("PEPE/USD") == "other"

    def test_symbol_cluster_is_case_insensitive(self) -> None:
        assert symbol_cluster("btc/usdt") == "crypto_majors"
        assert symbol_cluster("eth/eur") == "crypto_majors"

    def test_symbol_cluster_without_slash(self) -> None:
        """Defensive fallback for bare asset names (no slash) that CCXT would
        never normally produce — not part of the documented public contract."""
        assert symbol_cluster("BTC") == "crypto_majors"
        assert symbol_cluster("UNKNOWN") == "other"

    # ----- 2. Same-cluster prospective exposure blocks ----------------

    def test_same_cluster_buy_blocked_when_prospective_exceeds_cap(self) -> None:
        """BTC position at 2500 USDT + new ETH buy at 2000 USDT = 4500 USDT
        crypto_majors exposure > 4000 cap -> BLOCKED."""
        manager = self._make_cluster_manager()
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="ETH/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("0.04"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is False
        assert any(
            "cluster" in r.lower() and "crypto_majors" in r
            for r in result.rejection_reasons
        ), (
            f"Expected cluster rejection mentioning crypto_majors, "
            f"got: {result.rejection_reasons}"
        )

    def test_same_cluster_buy_allowed_when_under_cap(self) -> None:
        """BTC position at 2500 USDT + new ETH buy at 1000 USDT = 3500 USDT
        crypto_majors exposure < 4000 cap -> APPROVED."""
        manager = self._make_cluster_manager()
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="ETH/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("0.02"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"crypto_majors at 3500 < 4000 cap should pass; "
            f"rejection_reasons={result.rejection_reasons}"
        )

    # ----- 3. Boundary: prospective EXACTLY at cap (CR-004) -----------

    def test_same_cluster_buy_allowed_when_prospective_exactly_at_cap(self) -> None:
        """Cluster check uses strict ``>`` so prospective == cap is APPROVED.
        Contrast with the total-portfolio check which uses ``>=`` (blocks at
        equality).  Documents the asymmetry and guards against an accidental
        change to ``>=`` going undetected."""
        manager = self._make_cluster_manager()
        # 0.05 BTC @ 50000 = 2500 existing.  Order: 0.03 ETH @ 50000 = 1500.
        # Prospective = 2500 + 1500 = 4000 == cluster_cap (4000) -> APPROVED.
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="ETH/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("0.03"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"Prospective == cluster_cap must be APPROVED (strict >); "
            f"rejection_reasons={result.rejection_reasons}"
        )

    # ----- 4. Cross-cluster order is approved -------------------------

    def test_cross_cluster_order_not_constrained(self) -> None:
        """BTC at 2500 USDT (crypto_majors) + USDC buy at 1500 USDT
        (stablecoin) -> each cluster well under cap, APPROVED."""
        manager = self._make_cluster_manager()
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="USDC/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("1500"),
            price=Decimal("1"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"Cross-cluster buy must pass; rejection_reasons={result.rejection_reasons}"
        )

    # ----- 5. SELL bypasses cluster check -----------------------------

    def test_sell_bypasses_cluster_cap_even_when_cluster_full(self) -> None:
        """When crypto_majors cluster is over cap, a SELL must still be
        approved (mirrors the SELL bypass on the total cap)."""
        manager = self._make_cluster_manager()
        # Cluster already saturated: 0.1 BTC @ 50000 = 5000 > 4000 cap.
        existing = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.1"),
            current_price=Decimal("50000"),
        )
        order = _make_limit_order(
            symbol="BTC/EUR",
            side=OrderSide.SELL,
            quantity=Decimal("0.05"),
            price=Decimal("50000"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is True, (
            f"SELL must bypass cluster cap; rejection_reasons={result.rejection_reasons}"
        )

    # ----- 6. Unknown base asset -> 'other' cluster -------------------

    def test_unknown_base_asset_uses_other_cluster_and_still_capped(self) -> None:
        """SHIB -> 'other' cluster.  A 5000-USDT SHIB buy with no other
        positions still trips the cluster cap (5000 > 4000)."""
        manager = self._make_cluster_manager()
        order = _make_limit_order(
            symbol="SHIB/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("50"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
        )
        assert result.approved is False
        assert any(
            "cluster" in r.lower() and "'other'" in r
            for r in result.rejection_reasons
        ), (
            f"Expected cluster rejection mentioning 'other', "
            f"got: {result.rejection_reasons}"
        )

    # ----- 7. Order with no resolvable price skips cluster check ------

    def test_market_order_with_no_price_skips_cluster_check(self) -> None:
        """When an order has no price AND no existing same-symbol position
        provides a current_price, the cluster check skips this order rather
        than guessing notional from quantity alone.  The downstream
        concentration + notional caps still apply.

        Here: MARKET BUY for ETH (no existing ETH position) -> cluster check
        is skipped; the order is approved on the cluster axis.  ``market_price``
        IS passed so the order_size / concentration check has a price to use."""
        manager = self._make_cluster_manager()
        # Existing BTC position at 2500 -- does NOT serve as fallback for ETH.
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        order = _make_market_order(
            symbol="ETH/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
            market_price=Decimal("50000"),
        )
        assert result.approved is True, (
            f"Cluster check must be skipped when order price + same-symbol "
            f"current_price are both unavailable; "
            f"rejection_reasons={result.rejection_reasons}"
        )

    # ----- 8. Same-symbol current_price fallback ----------------------

    def test_market_order_uses_same_symbol_current_price_fallback(self) -> None:
        """When order.price is None but an existing position in the SAME
        symbol has a current_price, that price is used to value the
        prospective notional for the cluster check."""
        manager = self._make_cluster_manager()
        # Existing BTC at 2500.  Same symbol -> its current_price is the
        # fallback used to value a price-less MARKET BUY of BTC.
        existing_btc = _make_position(
            symbol="BTC/EUR",
            quantity=Decimal("0.05"),
            current_price=Decimal("50000"),
        )
        # MARKET BUY 0.04 BTC -> notional via fallback = 0.04 * 50000 = 2000.
        # Prospective crypto_majors = 2500 + 2000 = 4500 > 4000 -> BLOCKED.
        order = _make_market_order(
            symbol="BTC/EUR",
            side=OrderSide.BUY,
            quantity=Decimal("0.04"),
        )
        result = manager.pre_trade_check(
            order=order,
            current_equity=self._EQUITY,
            open_positions=[existing_btc],
            daily_pnl=Decimal("0"),
            peak_equity=self._PEAK,
            market_price=Decimal("50000"),
        )
        assert result.approved is False
        assert any(
            "cluster" in r.lower() and "crypto_majors" in r
            for r in result.rejection_reasons
        ), (
            f"Expected cluster rejection via same-symbol price fallback, "
            f"got: {result.rejection_reasons}"
        )

    # ----- 9. RiskParameters validation -------------------------------

    def test_cluster_cap_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cluster_exposure_pct"):
            RiskParameters(max_cluster_exposure_pct=0.0)

    def test_cluster_cap_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cluster_exposure_pct"):
            RiskParameters(max_cluster_exposure_pct=1.5)

    def test_cluster_cap_above_total_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <= max_portfolio_exposure_pct"):
            RiskParameters(
                max_portfolio_exposure_pct=0.40,
                max_cluster_exposure_pct=0.50,
            )

    def test_cluster_cap_equal_to_total_cap_accepted(self) -> None:
        """Boundary value: cluster == total cap is allowed (cluster cap
        effectively bites at the same point as the total cap)."""
        params = RiskParameters(
            max_portfolio_exposure_pct=0.50,
            max_cluster_exposure_pct=0.50,
        )
        assert params.max_cluster_exposure_pct == 0.50
