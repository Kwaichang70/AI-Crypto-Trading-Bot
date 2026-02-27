"""
packages/common/types.py
------------------------
Canonical enumerations shared across all packages.
Import from here — never redefine in individual modules.
"""

from __future__ import annotations

from enum import StrEnum, auto

__all__ = [
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "SignalDirection",
    "RunMode",
    "TimeFrame",
    "AssetClass",
    "LogLevel",
]


class OrderSide(StrEnum):
    """Direction of a trade order."""

    BUY = auto()
    SELL = auto()


class OrderType(StrEnum):
    """Execution type for an order. MVP supports MARKET and LIMIT only."""

    MARKET = auto()
    LIMIT = auto()
    # Post-MVP
    STOP_LIMIT = auto()
    STOP_MARKET = auto()


class OrderStatus(StrEnum):
    """
    Order state machine.

    Allowed transitions:
        NEW -> PENDING_SUBMIT -> OPEN -> PARTIAL -> FILLED
        NEW -> PENDING_SUBMIT -> OPEN -> CANCELED
        NEW -> PENDING_SUBMIT -> REJECTED
        * -> EXPIRED  (exchange TTL)
    """

    NEW = auto()            # Created locally, not yet submitted
    PENDING_SUBMIT = auto() # Submitted to execution engine, awaiting ACK
    OPEN = auto()           # Acknowledged by exchange, resting in order book
    PARTIAL = auto()        # Partially filled, still open
    FILLED = auto()         # Fully filled
    CANCELED = auto()       # Canceled by user or system
    REJECTED = auto()       # Rejected by exchange (insufficient funds, etc.)
    EXPIRED = auto()        # Expired by exchange TTL


class SignalDirection(StrEnum):
    """Trading signal produced by a strategy on each bar."""

    BUY = auto()
    SELL = auto()
    HOLD = auto()


class RunMode(StrEnum):
    """Execution mode for a trading run."""

    BACKTEST = auto()      # Historical simulation — no orders placed
    PAPER = auto()         # Simulated fills with real market data
    LIVE = auto()          # Real order placement — requires safety gates


class TimeFrame(StrEnum):
    """
    OHLCV candlestick timeframe identifiers.

    Values match CCXT exchange timeframe strings directly.
    """

    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class AssetClass(StrEnum):
    """Asset class — MVP is SPOT only."""

    SPOT = auto()
    # Post-MVP placeholders (not implemented)
    FUTURES = auto()
    OPTIONS = auto()


class LogLevel(StrEnum):
    """Structured log severity levels matching Python's logging module."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
