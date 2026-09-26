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

import math
from decimal import ROUND_DOWN, Decimal
from typing import Sequence

import structlog

from common.types import OrderSide
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
            5b. portfolio_exposure
            6. order_size (notional cap + concentration cap)

        All violations are collected before a verdict is issued.  If any
        violation is *blocking*, the order is rejected.  If only *warnings*
        exist, the order is approved -- but ``adjusted_quantity`` may be
        reduced when the position-size concentration cap applies.

        Side-aware exits (WP1.2)
        -------------------------
        A SELL that :meth:`BaseRiskManager.is_exposure_reducing` classifies
        as exposure-reducing (``0 < quantity <= held``) bypasses gates 1-5
        above: each becomes a non-blocking warning via
        :meth:`BaseRiskManager._bypass` instead of rejecting the order, and
        step 6 (order-size / concentration) is skipped entirely -- an exit
        needs no notional cap (C24). A SELL that exceeds ``held`` (with
        ``held > 0``) is clamped to ``min(quantity, held)`` -- a warning
        (``sell_clamped_to_held``), not a block -- and is then treated as
        reducing too. A SELL with ``held == 0`` is always blocked by the
        new ``sell_without_position`` rule and runs through every gate
        unchanged, exactly like a BUY. A SELL with ``quantity <= 0`` is
        always blocked by ``invalid_quantity`` (WP12-S-01) -- never
        clamped, since that would raise rather than lower the quantity;
        this only matters for a caller that bypasses ``Order``'s own
        ``gt=0`` validation (e.g. via ``model_copy``). Portfolio / cluster
        exposure (5b) stays BUY-only, unchanged.
        """
        violations: list[RiskViolation] = []

        held = self._held_for(order.symbol, open_positions)
        reducing = self.is_exposure_reducing(order, open_positions)
        adjusted_qty = order.quantity

        if order.side == OrderSide.SELL and not reducing:
            if order.quantity <= Decimal(0):
                # WP12-S-01: a non-reducing SELL with quantity <= 0 must be
                # rejected outright. The clamp below only ever LOWERS a
                # quantity; treating quantity<=0 as "exceeds held" would
                # instead RAISE it to `held`, breaking the "adjusted never
                # exceeds requested" contract for a caller that bypasses
                # Order's own `gt=0` validation (e.g. via `model_copy`).
                violations.append(
                    RiskViolation(
                        rule="invalid_quantity",
                        message=(
                            f"invalid_quantity: SELL quantity {order.quantity} "
                            f"for {order.symbol} is <= 0."
                        ),
                        blocking=True,
                    )
                )
            elif held > Decimal(0):
                # S3: SELL exceeds held. Clamp instead of reject -- a block
                # here would reopen C6 through an edge case neither engine
                # produces today. Treat the clamped order as reducing so it
                # also bypasses gates 1-5 and the order-size check below.
                # WP12-S-01: `min(...)` is defence in depth -- `quantity`
                # is already known to be > held in this branch (`quantity
                # > 0` and `not reducing`), so this is currently equivalent
                # to `held`, but it can never raise the quantity even if
                # that invariant is ever broken upstream.
                adjusted_qty = min(order.quantity, held)
                reducing = True
                violations.append(
                    RiskViolation(
                        rule="sell_clamped_to_held",
                        message=(
                            f"sell_clamped_to_held: SELL quantity {order.quantity} "
                            f"for {order.symbol} exceeds held {held}; clamped to "
                            f"{adjusted_qty}."
                        ),
                        blocking=False,
                    )
                )
                self._log.warning(
                    "risk.sell_clamped_to_held",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    requested_qty=str(order.quantity),
                    held=str(held),
                    adjusted_qty=str(adjusted_qty),
                )
            else:
                # S4: defence in depth -- a SELL with nothing held is always
                # rejected. Not reducing, so every gate below still applies
                # to it (including order-size), exactly like a BUY.
                violations.append(
                    RiskViolation(
                        rule="sell_without_position",
                        message=(
                            f"sell_without_position: SELL rejected for "
                            f"{order.symbol}: no held quantity (held=0)."
                        ),
                        blocking=True,
                    )
                )

        bypassed_rules: list[str] = []

        def _gate(gate_violation: RiskViolation | None) -> None:
            if gate_violation is None:
                return
            if reducing and gate_violation.rule in self._ENTRY_ONLY_RULES:
                bypassed_rules.append(gate_violation.rule)
                violations.append(self._bypass(gate_violation))
            else:
                violations.append(gate_violation)

        # 1. Kill switch
        _gate(self._check_kill_switch())

        # 1b. Protective mode (WP1.8 S-10): risk-layer backstop alongside
        # the strategy-layer _drop_entry_signals filter -- BUY entries
        # stay blocked even if a bug ever bypasses the strategy filter.
        _gate(self._check_protective_mode(order))

        # 2. Cooldown
        _gate(self._check_cooldown())

        # 3. Max open positions
        _gate(self._check_max_positions(open_positions))

        # 4. Daily loss
        _gate(self._check_daily_loss(daily_pnl, current_equity))

        # 5. Drawdown
        _gate(self._check_drawdown(current_equity, peak_equity))

        # 5b. Portfolio exposure (already BUY-only; unchanged by WP1.2)
        v = self._check_portfolio_exposure(order, current_equity, open_positions)
        if v is not None:
            violations.append(v)

        # 6. Order size / concentration. C24: a reducing SELL (including a
        # clamped one) needs no notional cap and skips this step entirely;
        # adjusted_qty was already set above (order.quantity, or `held` if
        # clamped).
        if not reducing:
            order_violations, sized_qty = self._check_order_size(
                order,
                current_equity,
                open_positions,
                market_price=market_price,
            )
            violations.extend(order_violations)
            adjusted_qty = sized_qty

        if bypassed_rules:
            self._log.warning(
                "risk.exposure_reducing_bypass",
                order_id=str(order.order_id),
                symbol=order.symbol,
                side=order.side.value,
                requested_qty=str(order.quantity),
                held=str(held),
                adjusted_qty=str(adjusted_qty),
                bypassed_rules=bypassed_rules,
            )

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
                side=order.side.value,
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
            side=order.side.value,
            adjusted_quantity=str(adjusted_qty),
            warnings=warning_msgs,
        )
        return result

    # ------------------------------------------------------------------
    # calculate_position_size
    # ------------------------------------------------------------------

    def _compute_stop_distance(
        self,
        *,
        entry_price: Decimal,
        stop_loss_price: Decimal | None,
        atr_value: Decimal | None,
    ) -> Decimal:
        """QT-002: derive the fractional stop distance for sizing.

        ATR mode (``sizing_mode == "atr"``) uses ``atr * multiplier /
        entry_price`` when a positive ``atr_value`` is available; the
        floor of 0.1 % protects against degenerate near-zero ATR readings
        (early-bar warmup, illiquid pairs) that would otherwise blow up
        position size.  Fixed mode preserves the legacy behaviour exactly.
        """
        if (
            self._params.sizing_mode == "atr"
            and atr_value is not None
            and atr_value > Decimal(0)
        ):
            distance = (atr_value * self._params.atr_risk_multiplier) / entry_price
            if distance < Decimal("0.001"):
                self._log.warning(
                    "risk.atr_distance_below_floor",
                    raw_distance=str(distance),
                    msg="ATR-derived distance < 0.1%; clamping to floor.",
                )
                distance = Decimal("0.001")
            return distance

        if self._params.sizing_mode == "atr" and atr_value is None:
            self._log.warning(
                "risk.atr_value_missing",
                msg="sizing_mode='atr' but no atr_value provided; "
                    "falling back to fixed-distance path.",
            )

        if stop_loss_price is not None and stop_loss_price > Decimal(0):
            distance = abs(entry_price - stop_loss_price) / entry_price
            # Guard against unrealistically tight stops that would inflate
            # position size: floor at 0.1% distance.
            if distance < Decimal("0.001"):
                distance = Decimal("0.001")
            return distance

        return _DEFAULT_STOP_DISTANCE_PCT

    def calculate_position_size(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None,
        confidence: float,
        atr_value: Decimal | None = None,
    ) -> Decimal:
        """
        Position sizing scaled by strategy confidence.

        Two modes (see :class:`RiskParameters`):

        * ``sizing_mode="fixed"`` (default) — fixed-fractional risk distance
          from ``stop_loss_price`` (or 1 % default when missing).
        * ``sizing_mode="atr"`` (QT-002) — distance = ``atr_value *
          atr_risk_multiplier`` / ``entry_price``; falls back to fixed-mode
          with a warning when ``atr_value`` is missing.

        Steps (both modes):
            1. risk_amount = equity * per_trade_risk_pct * confidence
            2. distance    = mode-dependent (see above)
            3. size_base   = risk_amount / (entry_price * distance)
            4. Cap at max_order_size_quote / entry_price
            5. Cap at max_position_size_pct * equity / entry_price
            6. Return min of all values, rounded DOWN to 8 decimal places.
        """
        if equity <= Decimal(0) or entry_price <= Decimal(0):
            return Decimal(0)

        if not math.isfinite(confidence):
            self._log.warning(
                "risk.invalid_confidence",
                confidence=confidence,
                msg="Confidence is NaN or Inf; returning zero position size.",
            )
            return Decimal(0)

        confidence_d = Decimal(str(max(0.0, min(1.0, confidence))))

        risk_pct = Decimal(str(self._params.per_trade_risk_pct))
        risk_amount = equity * risk_pct * confidence_d

        # Determine stop-loss distance (mode-dependent — QT-002)
        distance = self._compute_stop_distance(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            atr_value=atr_value,
        )

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
               (applies to all orders).
            b) Concentration cap: resulting position value vs
               ``max_position_size_pct * equity`` (BUY orders only — SELL
               orders reduce position size and therefore bypass this check
               to avoid trapping capital).

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

        # (b) Concentration cap -- only applies to BUY orders.
        # SELL orders reduce the position, so they can never exceed
        # concentration limits.  Blocking SELLs would trap capital.
        if order.side == OrderSide.BUY and current_equity > Decimal(0):
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
