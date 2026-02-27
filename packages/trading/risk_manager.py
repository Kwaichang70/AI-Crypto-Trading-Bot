"""
packages/trading/risk_manager.py
---------------------------------
Concrete implementation of BaseRiskManager.

DefaultRiskManager runs every pre-trade check helper in a fixed order,
collects ALL violations (blocking + warning), and returns a single
RiskCheckResult.  Position sizing uses fixed-fractional Kelly-aware
logic capped by order-size and concentration limits.

This module is intentionally *synchronous* -- zero I/O in the hot path.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import Sequence

import structlog

from trading.models import Order, Position, RiskCheckResult
from trading.risk import BaseRiskManager, RiskParameters, RiskViolation

__all__ = ["DefaultRiskManager"]

logger = structlog.get_logger(__name__)

# Sentinel used when no stop-loss price is provided.
_DEFAULT_STOP_DISTANCE_PCT = Decimal("0.01")  # 1%

# Precision for base-asset quantities returned by position sizing.
_QTY_PRECISION = Decimal("0.00000001")  # 8 decimal places


class DefaultRiskManager(BaseRiskManager):
    """
    Production risk manager with full pre-trade gating, fixed-fractional
    position sizing, and loss-streak cooldown management.

    Every public method is synchronous and performs no I/O.

    Parameters
    ----------
    run_id:
        Identifies the trading run this manager is scoped to.
    params:
        Immutable risk configuration.  See ``RiskParameters`` for defaults
        and validation rules.
    """

    def __init__(self, run_id: str, params: RiskParameters | None = None) -> None:
        super().__init__(run_id=run_id, params=params or RiskParameters())

    # ------------------------------------------------------------------
    # pre_trade_check
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        order: Order,
        current_equity: Decimal,
        open_positions: list[Position],
        daily_pnl: Decimal,
        peak_equity: Decimal,
        market_price: Decimal | None = None,
    ) -> RiskCheckResult:
        """
        Evaluate **all** risk rules against a proposed order.

        Check order:
            1. kill_switch
            2. cooldown
            3. max_positions
            4. daily_loss
            5. drawdown
            6. order_size (notional cap + concentration cap)

        All violations are collected before a verdict is issued.  If any
        violation is *blocking*, the order is rejected.  If only *warnings*
        exist, the order is approved -- but ``adjusted_quantity`` may be
        reduced when the position-size concentration cap applies.
        """
        violations: list[RiskViolation] = []

        # 1. Kill switch
        v = self._check_kill_switch()
        if v is not None:
            violations.append(v)

        # 2. Cooldown
        v = self._check_cooldown()
        if v is not None:
            violations.append(v)

        # 3. Max open positions
        v = self._check_max_positions(open_positions)
        if v is not None:
            violations.append(v)

        # 4. Daily loss
        v = self._check_daily_loss(daily_pnl, current_equity)
        if v is not None:
            violations.append(v)

        # 5. Drawdown
        v = self._check_drawdown(current_equity, peak_equity)
        if v is not None:
            violations.append(v)

        # 6. Order size / concentration
        order_violations, adjusted_qty = self._check_order_size(
            order, current_equity, open_positions, market_price=market_price,
        )
        violations.extend(order_violations)

        # ----- Partition into blocking / warning -----
        blocking = [v for v in violations if v.blocking]
        warnings = [v for v in violations if not v.blocking]

        blocking_msgs = [v.message for v in blocking]
        warning_msgs = [v.message for v in warnings]

        if blocking:
            result = RiskCheckResult(
                approved=False,
                adjusted_quantity=Decimal(0),
                rejection_reasons=blocking_msgs,
                warnings=warning_msgs,
            )
            self._log.debug(
                "risk.pre_trade_check.rejected",
                order_id=str(order.order_id),
                symbol=order.symbol,
                rejection_reasons=blocking_msgs,
                warnings=warning_msgs,
            )
            return result

        # Approved (possibly with adjusted quantity)
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=adjusted_qty,
            rejection_reasons=[],
            warnings=warning_msgs,
        )
        self._log.debug(
            "risk.pre_trade_check.approved",
            order_id=str(order.order_id),
            symbol=order.symbol,
            adjusted_quantity=str(adjusted_qty),
            warnings=warning_msgs,
        )
        return result

    # ------------------------------------------------------------------
    # calculate_position_size
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None,
        confidence: float,
    ) -> Decimal:
        """
        Fixed-fractional position sizing scaled by strategy confidence.

        Steps:
            1. risk_amount = equity * per_trade_risk_pct * confidence
            2. distance  = |entry - stop| / entry  (default 1% if no stop)
            3. size_base = risk_amount / (entry_price * distance)
            4. Cap at max_order_size_quote / entry_price
            5. Cap at max_position_size_pct * equity / entry_price
            6. Return min of all values, rounded DOWN to 8 decimal places.
        """
        if equity <= Decimal(0) or entry_price <= Decimal(0):
            return Decimal(0)

        confidence_d = Decimal(str(max(0.0, min(1.0, confidence))))

        risk_pct = Decimal(str(self._params.per_trade_risk_pct))
        risk_amount = equity * risk_pct * confidence_d

        # Determine stop-loss distance
        if stop_loss_price is not None and stop_loss_price > Decimal(0):
            distance = abs(entry_price - stop_loss_price) / entry_price
            # Guard against unrealistically tight stops that would inflate
            # position size: floor at 0.1% distance.
            if distance < Decimal("0.001"):
                distance = Decimal("0.001")
        else:
            distance = _DEFAULT_STOP_DISTANCE_PCT

        # Core sizing
        size_from_risk = risk_amount / (entry_price * distance)

        # Cap 1: absolute order size
        max_order_cap = self._params.max_order_size_quote / entry_price

        # Cap 2: concentration cap
        max_concentration_cap = (
            Decimal(str(self._params.max_position_size_pct)) * equity / entry_price
        )

        # Take the smallest
        final_size = min(size_from_risk, max_order_cap, max_concentration_cap)

        # Never negative
        if final_size <= Decimal(0):
            return Decimal(0)

        return final_size.quantize(_QTY_PRECISION, rounding=ROUND_DOWN)

    # ------------------------------------------------------------------
    # update_after_fill
    # ------------------------------------------------------------------

    def update_after_fill(
        self,
        realised_pnl: Decimal,
        *,
        is_loss: bool,
    ) -> None:
        """
        Update loss-streak tracking and trigger cooldown if threshold is hit.
        """
        if is_loss:
            self._consecutive_losses += 1
            self._log.info(
                "risk.fill_update.loss",
                consecutive_losses=self._consecutive_losses,
                realised_pnl=str(realised_pnl),
            )
        else:
            self._consecutive_losses = 0
            self._log.info(
                "risk.fill_update.win",
                realised_pnl=str(realised_pnl),
            )

        if self._consecutive_losses >= self._params.loss_streak_count:
            self._cooldown_bars_remaining = self._params.cooldown_after_loss_streak
            self._log.warning(
                "risk.cooldown_activated",
                consecutive_losses=self._consecutive_losses,
                cooldown_bars=self._cooldown_bars_remaining,
                loss_streak_threshold=self._params.loss_streak_count,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_order_size(
        self,
        order: Order,
        current_equity: Decimal,
        open_positions: list[Position],
        market_price: Decimal | None = None,
    ) -> tuple[list[RiskViolation], Decimal]:
        """
        Validate the proposed order against notional-size and concentration
        limits.  Returns a list of violations and the (possibly reduced)
        adjusted quantity.

        Two sub-checks:
            a) Absolute notional cap: order value vs ``max_order_size_quote``
            b) Concentration cap: resulting position value vs
               ``max_position_size_pct * equity``

        If neither cap is breached the adjusted quantity equals the original
        order quantity.  If a cap IS breached the quantity is reduced to the
        cap boundary; if the reduction makes it zero the violation is
        blocking.  A non-zero reduction emits a *warning* rather than a
        block, because the trade can still proceed at a smaller size.
        """
        violations: list[RiskViolation] = []
        adjusted_qty = order.quantity

        # Determine the effective price for notional calculations.
        # For LIMIT orders use the limit price; for MARKET orders use
        # the latest current_price from an existing position for the same
        # symbol, or fall back to the order price field.
        effective_price = self._resolve_effective_price(order, open_positions, market_price)
        if effective_price is None or effective_price <= Decimal(0):
            violations.append(
                RiskViolation(
                    rule="order_size_price_unknown",
                    message=(
                        f"Cannot determine effective price for {order.symbol} "
                        f"order; market price unavailable and order has no limit price."
                    ),
                    blocking=True,
                )
            )
            return violations, Decimal(0)

        # (a) Absolute notional cap ------------------------------------------
        order_notional = adjusted_qty * effective_price
        max_notional = self._params.max_order_size_quote

        if order_notional > max_notional:
            capped_qty_a = (max_notional / effective_price).quantize(
                _QTY_PRECISION, rounding=ROUND_DOWN,
            )
            if capped_qty_a <= Decimal(0):
                violations.append(
                    RiskViolation(
                        rule="max_order_size",
                        message=(
                            f"Order notional {order_notional:.2f} exceeds cap "
                            f"{max_notional:.2f} and cannot be reduced to a "
                            f"valid quantity."
                        ),
                        blocking=True,
                    )
                )
                return violations, Decimal(0)
            violations.append(
                RiskViolation(
                    rule="max_order_size",
                    message=(
                        f"Order notional {order_notional:.2f} exceeds cap "
                        f"{max_notional:.2f}; quantity reduced from "
                        f"{adjusted_qty} to {capped_qty_a}."
                    ),
                    blocking=False,  # warning: we reduce, not reject
                )
            )
            adjusted_qty = capped_qty_a

        # (b) Concentration cap -----------------------------------------------
        if current_equity > Decimal(0):
            # Find existing position for this symbol (if any)
            existing_value = Decimal(0)
            for pos in open_positions:
                if pos.symbol == order.symbol and not pos.is_flat:
                    existing_value += pos.notional_value

            proposed_value = existing_value + (adjusted_qty * effective_price)
            max_position_value = (
                Decimal(str(self._params.max_position_size_pct)) * current_equity
            )

            if proposed_value > max_position_value:
                # How much room is left?
                remaining_value = max_position_value - existing_value
                if remaining_value <= Decimal(0):
                    violations.append(
                        RiskViolation(
                            rule="max_position_concentration",
                            message=(
                                f"Position concentration for {order.symbol}: "
                                f"existing {existing_value:.2f} already at or "
                                f"above cap {max_position_value:.2f} "
                                f"({self._params.max_position_size_pct:.1%} "
                                f"of {current_equity:.2f} equity)."
                            ),
                            blocking=True,
                        )
                    )
                    return violations, Decimal(0)

                capped_qty_b = (remaining_value / effective_price).quantize(
                    _QTY_PRECISION, rounding=ROUND_DOWN,
                )
                if capped_qty_b <= Decimal(0):
                    violations.append(
                        RiskViolation(
                            rule="max_position_concentration",
                            message=(
                                f"Position concentration cap for {order.symbol} "
                                f"leaves insufficient room for any order."
                            ),
                            blocking=True,
                        )
                    )
                    return violations, Decimal(0)

                if capped_qty_b < adjusted_qty:
                    violations.append(
                        RiskViolation(
                            rule="max_position_concentration",
                            message=(
                                f"Position concentration for {order.symbol}: "
                                f"proposed {proposed_value:.2f} exceeds cap "
                                f"{max_position_value:.2f}; quantity reduced "
                                f"from {adjusted_qty} to {capped_qty_b}."
                            ),
                            blocking=False,
                        )
                    )
                    adjusted_qty = capped_qty_b

        return violations, adjusted_qty

    @staticmethod
    def _resolve_effective_price(
        order: Order,
        open_positions: Sequence[Position],
        market_price: Decimal | None = None,
    ) -> Decimal | None:
        """
        Determine the best available price for notional calculations.

        Priority:
            1. order.price  (non-None for LIMIT orders)
            2. current_price from an existing open position for the same symbol
            3. market_price passed by the caller (from ticker / last bar)
            4. None  (caller must handle the missing-price scenario)
        """
        if order.price is not None and order.price > Decimal(0):
            return order.price

        for pos in open_positions:
            if pos.symbol == order.symbol and pos.current_price > Decimal(0):
                return pos.current_price

        if market_price is not None and market_price > Decimal(0):
            return market_price

        return None
