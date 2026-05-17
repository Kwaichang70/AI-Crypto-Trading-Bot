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
    "symbol_cluster",
]


# ---------------------------------------------------------------------------
# QT-003 (Sprint 46): Asset-cluster mapping for correlation-aware exposure
# ---------------------------------------------------------------------------
#
# Static base-asset → cluster lookup.  Major crypto assets are grouped under
# ``crypto_majors`` because rolling 30-day return correlation between BTC,
# ETH, SOL, XRP, ADA, DOGE, AVAX et al. routinely exceeds 0.8 in stress
# periods — the "diversified portfolio" of four such names has an
# effective N closer to 1.5 than to 4.
#
# Anything not in this map falls into the catch-all ``other`` cluster so
# the cluster cap can still apply (defensive default).  A future iteration
# could derive clusters dynamically from a rolling correlation matrix.
_BASE_ASSET_CLUSTERS: dict[str, str] = {
    # Bitcoin + correlated majors
    "BTC": "crypto_majors",
    "ETH": "crypto_majors",
    "SOL": "crypto_majors",
    "XRP": "crypto_majors",
    "ADA": "crypto_majors",
    "DOGE": "crypto_majors",
    "AVAX": "crypto_majors",
    "MATIC": "crypto_majors",
    "DOT": "crypto_majors",
    "LINK": "crypto_majors",
    "BNB": "crypto_majors",
    "LTC": "crypto_majors",
    "BCH": "crypto_majors",
    # Stablecoin reserve assets — kept separate so stablecoin-only
    # strategies do not collide with the majors cluster cap.
    "USDT": "stablecoin",
    "USDC": "stablecoin",
    "DAI": "stablecoin",
}


def symbol_cluster(symbol: str) -> str:
    """Return the cluster name for a CCXT-style symbol like ``BTC/EUR``.

    Resolves by base asset (the part before the first slash).  Unknown
    base assets land in the ``other`` cluster so the cluster-exposure
    cap still applies — defensive default avoids accidental
    cluster-bypass for newly-listed tokens that have not been
    classified yet.
    """
    base = symbol.split("/", 1)[0].strip().upper() if "/" in symbol else symbol.upper()
    return _BASE_ASSET_CLUSTERS.get(base, "other")

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RiskParameters:
    """
    Immutable risk configuration for a single trading run.

    All monetary values are in quote currency (e.g. USDT).
    All percentage values are expressed as fractions (0.01 = 1%).
    """

    # Position limits
    max_open_positions: int = 10
    max_position_size_pct: float = 0.15   # max 15% of equity per position
    max_portfolio_exposure_pct: float = 0.60  # max 60% of equity in open positions

    # Trade-level risk
    per_trade_risk_pct: float = 0.02      # risk 2% of equity per trade
    max_order_size_quote: Decimal = Decimal("10000")  # hard cap per order

    # Run-level circuit breakers
    max_daily_loss_pct: float = 0.08      # halt if daily loss >= 8% of start equity
    max_drawdown_pct: float = 0.30        # halt if drawdown >= 30% of peak equity

    # Fee / slippage model (Coinbase Advanced Trade lowest tier)
    taker_fee_pct: float = 0.006          # 0.60% taker fee
    maker_fee_pct: float = 0.004          # 0.40% maker fee
    slippage_bps: int = 5                 # 5 basis points slippage

    # Cooldown
    cooldown_after_loss_streak: int = 3   # bars to pause after N consecutive losses
    loss_streak_count: int = 3            # number of losses that triggers cooldown

    # QT-002 (Sprint 42): ATR-scaled position sizing
    #
    # ``sizing_mode="fixed"`` (default) preserves the legacy fixed-fractional
    # behaviour: risk distance is taken from ``stop_loss_price`` or falls back
    # to a hard-coded 1 % default.  ``sizing_mode="atr"`` derives the stop
    # distance from ``atr_value * atr_risk_multiplier``, scaling position size
    # inversely with volatility — quieter symbols get bigger positions, more
    # volatile ones get smaller, so each trade puts roughly the same dollar
    # amount of equity at risk.  Callers in ATR mode MUST supply an
    # ``atr_value`` to ``calculate_position_size``; otherwise the manager
    # falls back to the fixed-distance path with a warning.
    sizing_mode: str = "fixed"
    atr_risk_multiplier: Decimal = Decimal("1.5")

    # QT-003 (Sprint 46): correlation-aware cluster exposure cap.
    #
    # In addition to the flat ``max_portfolio_exposure_pct`` cap, every
    # asset cluster (see ``symbol_cluster``) gets its own ceiling.  Four
    # majors at ρ>0.8 are effectively one bet; capping the whole cluster
    # at 40 % prevents an apparent "diversified" portfolio from being
    # 60 %-exposed to the same beta.  Set to 1.0 to disable the cluster
    # gate while keeping the legacy total cap.
    max_cluster_exposure_pct: float = 0.40

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
        # QT-002: validate sizing-mode + ATR multiplier
        if self.sizing_mode not in ("fixed", "atr"):
            raise ValueError(
                f"sizing_mode must be 'fixed' or 'atr', got {self.sizing_mode!r}"
            )
        if self.atr_risk_multiplier <= Decimal("0"):
            raise ValueError(
                f"atr_risk_multiplier must be > 0, got {self.atr_risk_multiplier}"
            )
        if self.atr_risk_multiplier > Decimal("5"):
            # 5x ATR stop is already very wide; anything larger is almost
            # certainly a misconfiguration.
            raise ValueError(
                f"atr_risk_multiplier {self.atr_risk_multiplier} is unreasonably "
                f"large (>5); review configuration."
            )
        # QT-003: validate cluster cap
        if not (0 < self.max_cluster_exposure_pct <= 1.0):
            raise ValueError(
                f"max_cluster_exposure_pct {self.max_cluster_exposure_pct} "
                f"out of safe range (0, 1.0]"
            )
        if self.max_cluster_exposure_pct > self.max_portfolio_exposure_pct:
            # Cluster cap must be <= total cap; otherwise the cluster
            # check can never bite before the total cap does.
            raise ValueError(
                f"max_cluster_exposure_pct ({self.max_cluster_exposure_pct}) "
                f"must be <= max_portfolio_exposure_pct "
                f"({self.max_portfolio_exposure_pct})"
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

    @property
    def consecutive_losses(self) -> int:
        """Current count of consecutive losing trades."""
        return self._consecutive_losses

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
        atr_value: Decimal | None = None,
    ) -> Decimal:
        """
        Compute the order size in base asset using fixed-fractional sizing.

        Behaviour depends on ``params.sizing_mode``:

        * ``"fixed"`` (default) — risk distance from ``stop_loss_price`` (or
          a 1 % default when no stop is supplied).  Preserves the legacy
          per-trade-risk-percent fixed-fractional behaviour.
        * ``"atr"`` (QT-002, Sprint 42) — distance = ``atr_value *
          params.atr_risk_multiplier`` divided by ``entry_price``.  Equivalent
          to a target-vol sizing scheme — quieter symbols get bigger
          positions, more volatile ones get smaller.  When ``atr_value`` is
          missing the manager falls back to the fixed path with a warning.

        Parameters
        ----------
        equity:
            Current portfolio equity in quote currency.
        entry_price:
            Expected entry price for the trade.
        stop_loss_price:
            Price at which the stop-loss would trigger.
            If None and ``sizing_mode == "fixed"``, a default distance is used.
        confidence:
            Strategy confidence scalar in [0, 1]. Scales the position size
            proportionally.
        atr_value:
            Optional ATR (Average True Range) value in PRICE units (same
            currency / scale as ``entry_price``).  Required when
            ``params.sizing_mode == "atr"``; ignored otherwise.

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
        """Block BUY orders when total OR cluster exposure exceeds cap.

        QT-003 (Sprint 46): a portfolio nominally diversified across four
        major crypto names is in reality one beta bet because the names
        co-move at ρ>0.8.  The cluster cap therefore also bites BEFORE
        the total cap when one cluster (e.g. crypto_majors) approaches
        ``max_cluster_exposure_pct`` of equity.  The order's intended
        cluster is computed against the prospective exposure (= current
        cluster exposure + this order's notional) so the gate stops
        cluster-stacking ahead of fill.
        """
        if order.side != OrderSide.BUY or current_equity <= Decimal(0):
            return None

        open_only = [p for p in open_positions if not p.is_flat]
        total_exposure = sum((p.notional_value for p in open_only), Decimal(0))
        total_cap = (
            Decimal(str(self._params.max_portfolio_exposure_pct)) * current_equity
        )
        if total_exposure >= total_cap:
            return RiskViolation(
                rule="max_portfolio_exposure",
                message=(
                    f"Total portfolio exposure {total_exposure:.2f} "
                    f"already at or above cap {total_cap:.2f} "
                    f"({self._params.max_portfolio_exposure_pct:.0%} of equity)"
                ),
                blocking=True,
            )

        # QT-003 cluster check.  Build per-cluster running totals and
        # add the prospective order notional to its target cluster.  The
        # order's notional uses ``order.quantity * price`` where price is
        # ``order.price`` for LIMIT or the current market price for
        # MARKET — we use ``order.price`` when set, falling back to the
        # most-recent ``current_price`` of any existing position in the
        # same symbol.  When neither is available the order's quantity
        # alone is non-informative for notional, so the cluster check
        # skips it (the regular max_position_size + concentration caps
        # still apply downstream).
        cluster_exposure: dict[str, Decimal] = {}
        for p in open_only:
            cluster_exposure.setdefault(symbol_cluster(p.symbol), Decimal(0))
            cluster_exposure[symbol_cluster(p.symbol)] += p.notional_value

        target_cluster = symbol_cluster(order.symbol)
        order_price: Decimal | None = order.price
        if order_price is None:
            for p in open_only:
                if p.symbol == order.symbol and p.current_price > Decimal(0):
                    order_price = p.current_price
                    break
        if order_price is not None and order_price > Decimal(0):
            prospective = (
                cluster_exposure.get(target_cluster, Decimal(0))
                + order.quantity * order_price
            )
            cluster_cap = (
                Decimal(str(self._params.max_cluster_exposure_pct)) * current_equity
            )
            if prospective > cluster_cap:
                return RiskViolation(
                    rule="max_cluster_exposure",
                    message=(
                        f"Cluster '{target_cluster}' prospective exposure "
                        f"{prospective:.2f} would exceed cap {cluster_cap:.2f} "
                        f"({self._params.max_cluster_exposure_pct:.0%} of equity); "
                        f"current cluster exposure is "
                        f"{cluster_exposure.get(target_cluster, Decimal(0)):.2f}."
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
