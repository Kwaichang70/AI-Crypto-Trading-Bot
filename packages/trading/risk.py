"""
packages/trading/risk.py
------------------------
RiskManager skeleton — pre-trade check interface and position sizing.

The RiskManager is a synchronous component because pre-trade checks must
complete within microseconds in the hot path. Any I/O (reading from DB for
current equity) must be cached and refreshed asynchronously in the background.

Design principles
-----------------
- All checks are additive: multiple rules can fail simultaneously.
  All failures are collected and returned in ``RiskCheckResult.rejection_reasons``.
- The kill-switch is an emergency override that immediately blocks all orders.
- Position sizing uses fixed-fractional Kelly criterion by default.
- MVP: spot-only, max_leverage = 1 (no short positions).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

from common.types import OrderSide
from trading.models import Order, Position, RiskCheckResult

__all__ = [
    "BaseRiskManager",
    "RiskParameters",
    "RiskViolation",
]

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskParameters:
    """
    Immutable risk configuration for a single trading run.

    All monetary values are in quote currency (e.g. USDT).
    All percentage values are expressed as fractions (0.01 = 1%).
    """

    # Position limits
    max_open_positions: int = 3
    max_position_size_pct: float = 0.10   # max 10% of equity per position
    max_portfolio_exposure_pct: float = 0.30  # max 30% of equity in open positions

    # Trade-level risk
    per_trade_risk_pct: float = 0.01      # risk 1% of equity per trade
    max_order_size_quote: Decimal = Decimal("10000")  # hard cap per order

    # Run-level circuit breakers
    max_daily_loss_pct: float = 0.05      # halt if daily loss >= 5% of start equity
    max_drawdown_pct: float = 0.15        # halt if drawdown >= 15% of peak equity

    # Fee / slippage model (Coinbase Advanced Trade lowest tier)
    taker_fee_pct: float = 0.006          # 0.60% taker fee
    maker_fee_pct: float = 0.004          # 0.40% maker fee
    slippage_bps: int = 5                 # 5 basis points slippage

    # Cooldown
    cooldown_after_loss_streak: int = 3   # bars to pause after N consecutive losses
    loss_streak_count: int = 3            # number of losses that triggers cooldown

    def __post_init__(self) -> None:
        if not (0 < self.per_trade_risk_pct <= 0.05):
            raise ValueError(
                f"per_trade_risk_pct {self.per_trade_risk_pct} out of safe range (0, 0.05]"
            )
        if not (0 < self.max_drawdown_pct <= 0.50):
            raise ValueError(
                f"max_drawdown_pct {self.max_drawdown_pct} out of safe range (0, 0.50]"
            )
        if self.max_open_positions < 1:
            raise ValueError(
                f"max_open_positions must be >= 1, got {self.max_open_positions}"
            )
        if not (0 < self.max_portfolio_exposure_pct <= 1.0):
            raise ValueError(
                f"max_portfolio_exposure_pct {self.max_portfolio_exposure_pct} "
                f"out of safe range (0, 1.0]"
            )


@dataclass(slots=True)
class RiskViolation:
    """A single rule violation identified during a pre-trade check."""

    rule: str
    message: str
    blocking: bool = True   # False = warning only; True = blocks order


class BaseRiskManager(abc.ABC):
    """
    Abstract risk manager.

    Usage pattern
    -------------
    1. Inject into ExecutionEngine at construction time.
    2. Call ``pre_trade_check(order, portfolio_state)`` before any order
       submission. If ``result.approved`` is False, discard the order.
    3. Call ``update_portfolio_state(...)`` after each fill to keep
       internal equity/drawdown accumulators current.
    4. Call ``trigger_kill_switch(reason)`` to halt all trading immediately.
    5. Call ``reset_kill_switch()`` to resume (requires manual operator action).

    Parameters
    ----------
    run_id:
        The run this risk manager is scoped to.
    params:
        Immutable risk parameters for this run.
    """

    def __init__(self, run_id: str, params: RiskParameters) -> None:
        self._run_id = run_id
        self._params = params
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str | None = None
        self._consecutive_losses: int = 0
        self._cooldown_bars_remaining: int = 0
        self._log = structlog.get_logger(__name__).bind(run_id=run_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def params(self) -> RiskParameters:
        return self._params

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def trigger_kill_switch(self, reason: str) -> None:
        """
        Immediately halt all new order submissions.

        This is a one-way latch — only ``reset_kill_switch`` can clear it.
        The reason is logged at CRITICAL severity.

        Parameters
        ----------
        reason:
            Human-readable explanation for the halt.
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._log.critical(
            "risk.kill_switch_triggered",
            reason=reason,
            alert="TRADING_HALTED",
        )

    def reset_kill_switch(self) -> None:
        """
        Clear the kill switch and resume normal operation.

        This MUST be called explicitly by an operator — it is never
        cleared automatically.
        """
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._log.warning("risk.kill_switch_reset")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
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
        Evaluate all risk rules against a proposed order.

        This method MUST be fast (no I/O). All required state must be
        passed in as arguments; the implementation reads only ``self._params``
        and the provided portfolio snapshot.

        Parameters
        ----------
        order:
            The proposed order awaiting approval.
        current_equity:
            Total portfolio equity in quote currency (cash + open position value).
        open_positions:
            List of all currently open positions for this run.
        daily_pnl:
            Net PnL since the start of the current trading day in quote currency.
        peak_equity:
            Highest equity reached since run start, used for drawdown calculation.
        market_price:
            Current market price for the order's symbol. Required for MARKET
            orders where order.price is None and no existing position provides
            a reference price. Defaults to None for backward compatibility.

        Returns
        -------
        RiskCheckResult:
            ``approved=True`` if all blocking rules pass.
            ``adjusted_quantity`` may be smaller than ``order.quantity``
            if the position-size cap was applied.
        """
        ...

    @abc.abstractmethod
    def calculate_position_size(
        self,
        equity: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal | None,
        confidence: float,
    ) -> Decimal:
        """
        Compute the order size in base asset using fixed-fractional sizing.

        If ``stop_loss_price`` is None, falls back to using
        ``params.per_trade_risk_pct`` of equity as the risk amount and
        assumes a default stop-loss distance of 1%.

        Parameters
        ----------
        equity:
            Current portfolio equity in quote currency.
        entry_price:
            Expected entry price for the trade.
        stop_loss_price:
            Price at which the stop-loss would trigger.
            If None, a default distance is used.
        confidence:
            Strategy confidence scalar in [0, 1]. Scales the position size
            proportionally.

        Returns
        -------
        Decimal:
            Position size in base asset, rounded to exchange-appropriate
            precision.
        """
        ...

    @abc.abstractmethod
    def update_after_fill(
        self,
        realised_pnl: Decimal,
        *,
        is_loss: bool,
    ) -> None:
        """
        Update internal accumulators after a trade is closed.

        Increments or resets ``_consecutive_losses`` and manages
        ``_cooldown_bars_remaining``.

        Parameters
        ----------
        realised_pnl:
            Net realised PnL of the closed trade.
        is_loss:
            True if the trade was a losing trade.
        """
        ...

    def tick_cooldown(self) -> None:
        """
        Advance the cooldown counter by one bar.

        Call on every bar tick regardless of whether an order is being
        considered. Cooldown expires when ``_cooldown_bars_remaining``
        reaches 0, at which point ``_consecutive_losses`` is reset so
        that the next single loss does not immediately re-trigger cooldown.
        """
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1
            if self._cooldown_bars_remaining == 0:
                self._consecutive_losses = 0
                self._log.info("risk.cooldown_expired")

    @property
    def in_cooldown(self) -> bool:
        """True when the engine is in a post-loss cooldown period."""
        return self._cooldown_bars_remaining > 0

    # ------------------------------------------------------------------
    # Shared helper — used by concrete implementations
    # ------------------------------------------------------------------

    def _check_kill_switch(self) -> RiskViolation | None:
        if self._kill_switch_active:
            return RiskViolation(
                rule="kill_switch",
                message=f"Kill switch active: {self._kill_switch_reason}",
                blocking=True,
            )
        return None

    def _check_cooldown(self) -> RiskViolation | None:
        if self.in_cooldown:
            return RiskViolation(
                rule="loss_cooldown",
                message=(
                    f"In cooldown for {self._cooldown_bars_remaining} more bars "
                    f"after {self._consecutive_losses} consecutive losses"
                ),
                blocking=True,
            )
        return None

    def _check_max_positions(
        self,
        open_positions: list[Position],
    ) -> RiskViolation | None:
        non_flat = [p for p in open_positions if not p.is_flat]
        if len(non_flat) >= self._params.max_open_positions:
            return RiskViolation(
                rule="max_open_positions",
                message=(
                    f"Max open positions reached: "
                    f"{len(non_flat)}/{self._params.max_open_positions}"
                ),
                blocking=True,
            )
        return None

    def _check_daily_loss(
        self,
        daily_pnl: Decimal,
        current_equity: Decimal,
    ) -> RiskViolation | None:
        threshold = current_equity * Decimal(str(self._params.max_daily_loss_pct))
        if daily_pnl < -threshold:
            return RiskViolation(
                rule="max_daily_loss",
                message=(
                    f"Daily loss {daily_pnl:.2f} exceeds threshold "
                    f"-{threshold:.2f} ({self._params.max_daily_loss_pct:.1%})"
                ),
                blocking=True,
            )
        return None

    def _check_drawdown(
        self,
        current_equity: Decimal,
        peak_equity: Decimal,
    ) -> RiskViolation | None:
        if peak_equity <= Decimal(0):
            return None
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= Decimal(str(self._params.max_drawdown_pct)):
            return RiskViolation(
                rule="max_drawdown",
                message=(
                    f"Drawdown {drawdown:.1%} exceeds limit "
                    f"{self._params.max_drawdown_pct:.1%}"
                ),
                blocking=True,
            )
        return None

    def _check_portfolio_exposure(
        self,
        order: Order,
        current_equity: Decimal,
        open_positions: list[Position],
    ) -> RiskViolation | None:
        """Block BUY orders if total portfolio exposure exceeds cap."""
        if order.side != OrderSide.BUY or current_equity <= Decimal(0):
            return None
        total_exposure = sum(
            (p.notional_value for p in open_positions if not p.is_flat),
            Decimal(0),
        )
        cap = Decimal(str(self._params.max_portfolio_exposure_pct)) * current_equity
        if total_exposure >= cap:
            return RiskViolation(
                rule="max_portfolio_exposure",
                message=(
                    f"Total portfolio exposure {total_exposure:.2f} "
                    f"already at or above cap {cap:.2f} "
                    f"({self._params.max_portfolio_exposure_pct:.0%} of equity)"
                ),
                blocking=True,
            )
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"run_id={self._run_id!r}, "
            f"kill_switch={self._kill_switch_active})"
        )
