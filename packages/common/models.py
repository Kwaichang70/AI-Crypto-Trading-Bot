"""
packages/common/models.py
--------------------------
Shared domain value objects used across multiple packages.
These are pure data transfer objects with no business logic dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from common.types import TimeFrame

__all__ = [
    "MultiTimeframeContext",
    "OHLCVBar",
]


class OHLCVBar(BaseModel):
    """
    A single OHLCV candlestick bar.

    Timestamps are always UTC. ``volume`` is denominated in the base asset.
    """

    model_config = {"frozen": True}

    symbol: str = Field(description="Trading pair, e.g. 'BTC/USDT'")
    timeframe: TimeFrame = Field(description="Candle duration")
    timestamp: datetime = Field(description="Bar open time in UTC")
    open: Decimal = Field(gt=Decimal(0), description="Open price")
    high: Decimal = Field(gt=Decimal(0), description="High price")
    low: Decimal = Field(gt=Decimal(0), description="Low price")
    close: Decimal = Field(gt=Decimal(0), description="Close price")
    volume: Decimal = Field(ge=Decimal(0), description="Volume in base asset")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v.astimezone(UTC)
        raise ValueError(f"Expected datetime, got {type(v)}")

    @model_validator(mode="after")
    def validate_ohlcv_consistency(self) -> OHLCVBar:
        if self.low > self.high:
            raise ValueError("low must be <= high")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must be within [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must be within [low, high]")
        return self


@dataclass(frozen=True)
class MultiTimeframeContext:
    """
    Read-only higher-timeframe bar context passed to strategies.

    Strategies use this to access bars from higher timeframes (e.g., 4h, 1d)
    while executing on a lower primary timeframe (e.g., 1h). This is purely
    informational — no orders are placed on higher timeframes.
    """

    htf_bars: dict[str, dict[str, list[OHLCVBar]]] = field(default_factory=dict)
    """
    Higher-timeframe bars keyed by timeframe string, then by symbol.
    E.g. {"4h": {"BTC/USD": [bar1, bar2, ...]}, "1d": {"BTC/USD": [bar1, ...]}}

    Bars are filtered to prevent look-ahead bias: only bars whose entire
    period has completed before the current primary bar are included.
    """
