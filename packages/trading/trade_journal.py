"""
packages/trading/trade_journal.py
----------------------------------
Trade journaling utilities for adaptive learning (Sprint 32).

Components
----------
TradeExcursionTracker
    Tracks Maximum Adverse Excursion (MAE) and Maximum Favorable
    Excursion (MFE) for open positions on a per-bar basis. Values are
    expressed as a percentage of the entry price.

ExitReasonDetector
    Stateless utility that classifies the reason a position was closed
    (trailing_stop, signal_exit, etc.) from the strategy_id and optional
    signal metadata.

TradeSkipLogger
    Records trades that were evaluated but not taken (e.g. risk-blocked)
    for subsequent adaptive learning analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog

from trading.models import SkippedTrade

__all__ = [
    "TradeExcursionTracker",
    "ExitReasonDetector",
    "TradeSkipLogger",
]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Private state container for a single open position's excursion tracking
# ---------------------------------------------------------------------------

@dataclass
class _ExcursionState:
    """Per-symbol state for MAE/MFE tracking while a position is open."""

    entry_price: Decimal
    side: str                          # "long" (spot-only MVP)
    mfe_pct: float = 0.0               # Maximum Favorable Excursion, as fraction of entry
    mae_pct: float = 0.0               # Maximum Adverse Excursion, as fraction of entry (positive)
    regime_at_entry: str | None = None
    signal_context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# TradeExcursionTracker
# ---------------------------------------------------------------------------

class TradeExcursionTracker:
    """
    Bar-by-bar MAE/MFE tracker for open positions.

    Call ``on_position_open`` when a position is entered, ``on_bar`` on
    every subsequent bar, and ``on_position_close`` when the position is
    exited to retrieve the final excursion statistics.

    All excursion values are expressed as a fraction of the entry price
    (e.g. 0.02 = 2% excursion).  Positive values for both MAE and MFE
    make downstream comparisons straightforward.

    Thread-safety
    -------------
    This class is NOT thread-safe.  It is designed for single-threaded
    use within the StrategyEngine bar loop.
    """

    def __init__(self) -> None:
        self._positions: dict[str, _ExcursionState] = {}
        self._log = structlog.get_logger(__name__).bind(
            component="trade_excursion_tracker"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_position_open(
        self,
        symbol: str,
        entry_price: Decimal,
        side: str = "long",
        regime_at_entry: str | None = None,
        signal_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a new open position for excursion tracking.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. "BTC/USDT".
        entry_price:
            Fill price at position open.
        side:
            "long" for a long position (only supported mode in spot MVP).
        regime_at_entry:
            Optional regime label at open time (e.g. "RISK_ON").
        signal_context:
            Optional snapshot of indicator values at entry.
        """
        if entry_price <= Decimal("0"):
            self._log.warning(
                "excursion_tracker.invalid_entry_price",
                symbol=symbol,
                entry_price=str(entry_price),
            )
            return

        self._positions[symbol] = _ExcursionState(
            entry_price=entry_price,
            side=side,
            regime_at_entry=regime_at_entry,
            signal_context=signal_context,
        )
        self._log.debug(
            "excursion_tracker.position_opened",
            symbol=symbol,
            entry_price=str(entry_price),
        )

    def on_bar(
        self,
        symbol: str,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> None:
        """
        Update MAE/MFE for an open position given the current bar's OHLC.

        For a long position:
        - MFE is updated if the bar's high exceeds the best seen so far.
        - MAE is updated if the bar's low falls further from entry than any
          prior bar's low.

        Parameters
        ----------
        symbol:
            Trading pair to update.
        high:
            Bar high price.
        low:
            Bar low price.
        close:
            Bar close price (not used for excursion, reserved for future use).
        """
        state = self._positions.get(symbol)
        if state is None:
            return

        entry = state.entry_price
        if entry <= Decimal("0"):
            return

        # Long position excursion:
        # MFE = how far price moved in our favour (up)
        favorable_pct = float((high - entry) / entry)
        # MAE = how far price moved against us (down), expressed as positive
        adverse_pct = float((entry - low) / entry)

        if favorable_pct > state.mfe_pct:
            state.mfe_pct = favorable_pct

        if adverse_pct > state.mae_pct:
            state.mae_pct = adverse_pct

    def on_position_close(
        self,
        symbol: str,
    ) -> tuple[float, float, str | None, dict[str, Any] | None] | None:
        """
        Retrieve and remove excursion state for a closed position.

        Parameters
        ----------
        symbol:
            Trading pair being closed.

        Returns
        -------
        tuple[mae_pct, mfe_pct, regime_at_entry, signal_context] or None
            Returns None if no tracking state exists for this symbol.
        """
        state = self._positions.pop(symbol, None)
        if state is None:
            self._log.debug(
                "excursion_tracker.close_no_state",
                symbol=symbol,
            )
            return None

        self._log.debug(
            "excursion_tracker.position_closed",
            symbol=symbol,
            mae_pct=state.mae_pct,
            mfe_pct=state.mfe_pct,
        )

        return (state.mae_pct, state.mfe_pct, state.regime_at_entry, state.signal_context)

    def clear(self) -> None:
        """Remove all tracked positions (called on engine stop)."""
        self._positions.clear()
        self._log.debug("excursion_tracker.cleared")

    @property
    def tracked_symbols(self) -> list[str]:
        """Return list of symbols currently being tracked."""
        return list(self._positions.keys())


# ---------------------------------------------------------------------------
# ExitReasonDetector
# ---------------------------------------------------------------------------

class ExitReasonDetector:
    """
    Stateless classifier for trade exit reasons.

    Derives the exit reason from the strategy_id that generated the
    closing signal and optional signal metadata keys.

    Valid reasons
    -------------
    - ``trailing_stop``  -- closed by the trailing stop manager
    - ``signal_exit``    -- closed by a normal strategy SELL signal
    - ``stop_loss``      -- closed by a hard stop-loss rule
    - ``take_profit``    -- closed by a take-profit rule
    - ``regime_change``  -- closed due to regime shift
    - ``manual``         -- manually closed via API
    """

    VALID_REASONS: tuple[str, ...] = (
        "trailing_stop",
        "signal_exit",
        "stop_loss",
        "take_profit",
        "regime_change",
        "manual",
    )

    @staticmethod
    def detect(
        strategy_id: str,
        signal_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Classify the exit reason from strategy context.

        Priority order:
        1. If strategy_id contains "trailing_stop" -> ``trailing_stop``
        2. If signal_metadata has key "exit_reason" -> use that value
        3. If signal_metadata has key "stop_loss" or "take_profit" -> map to reason
        4. Default -> ``signal_exit``

        Parameters
        ----------
        strategy_id:
            The strategy_id from the closing signal.
        signal_metadata:
            Optional metadata dict from the closing signal.

        Returns
        -------
        str
            One of the VALID_REASONS values.
        """
        # Trailing stop manager sets strategy_id="trailing_stop"
        if "trailing_stop" in strategy_id:
            return "trailing_stop"

        if signal_metadata:
            # Explicit exit_reason in metadata takes precedence
            explicit = signal_metadata.get("exit_reason")
            if isinstance(explicit, str) and explicit in ExitReasonDetector.VALID_REASONS:
                return explicit

            # Infer from metadata keys
            if signal_metadata.get("stop_loss"):
                return "stop_loss"
            if signal_metadata.get("take_profit"):
                return "take_profit"
            if signal_metadata.get("regime_change"):
                return "regime_change"

        return "signal_exit"


# ---------------------------------------------------------------------------
# TradeSkipLogger
# ---------------------------------------------------------------------------

class TradeSkipLogger:
    """
    Records trades that were evaluated but not taken.

    Skipped trades are logged for adaptive learning analysis  -- they allow
    the system to compare hypothetical outcomes against actual risk blocks.

    Usage
    -----
    ::

        logger = TradeSkipLogger(run_id="run-123")
        logger.log_skip(
            symbol="BTC/USDT",
            skip_reason="max_position_size",
            signal_context={"rsi": 28.4, "confidence": 0.8},
            hypothetical_entry_price=Decimal("42000"),
        )
        all_skips = logger.get_all_skipped()
    """

    def __init__(self, run_id: str = "") -> None:
        self._run_id = run_id
        self._skipped: list[SkippedTrade] = []
        self._log = structlog.get_logger(__name__).bind(
            component="trade_skip_logger",
            run_id=run_id,
        )

    def set_run_id(self, run_id: str) -> None:
        """Update the run_id (called when run_id is available after init)."""
        self._run_id = run_id
        self._log = self._log.bind(run_id=run_id)

    def log_skip(
        self,
        symbol: str,
        skip_reason: str,
        signal_context: dict[str, Any] | None = None,
        hypothetical_entry_price: Decimal | None = None,
        regime_at_skip: str | None = None,
    ) -> SkippedTrade:
        """
        Record a skipped trade event.

        Parameters
        ----------
        symbol:
            Trading pair that was evaluated.
        skip_reason:
            Human-readable reason the trade was not taken (e.g. risk rule name).
        signal_context:
            Indicator snapshot at the time the signal was evaluated.
        hypothetical_entry_price:
            The price at which the trade would have been entered.
        regime_at_skip:
            Market regime label at skip time.

        Returns
        -------
        SkippedTrade
            The recorded skip event.
        """
        skip = SkippedTrade(
            run_id=self._run_id,
            symbol=symbol,
            skip_reason=skip_reason,
            signal_context=signal_context,
            hypothetical_entry_price=hypothetical_entry_price,
            regime_at_skip=regime_at_skip,
        )
        self._skipped.append(skip)

        self._log.debug(
            "trade_skip_logger.skip_recorded",
            symbol=symbol,
            skip_reason=skip_reason,
            skip_id=str(skip.skip_id),
        )

        return skip

    def get_all_skipped(self) -> list[SkippedTrade]:
        """Return a snapshot of all recorded skip events."""
        return list(self._skipped)

    @property
    def skip_count(self) -> int:
        """Total number of skipped trade events recorded."""
        return len(self._skipped)

    def get_skip_summary(self) -> dict[str, int]:
        """
        Aggregate skip counts by reason.

        Returns
        -------
        dict[str, int]
            Mapping of skip_reason -> count.
        """
        summary: dict[str, int] = {}
        for skip in self._skipped:
            summary[skip.skip_reason] = summary.get(skip.skip_reason, 0) + 1
        return summary

    def clear(self) -> None:
        """Clear all recorded skip events."""
        self._skipped.clear()
        self._log.debug("trade_skip_logger.cleared")
