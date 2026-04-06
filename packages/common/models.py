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

    External market signal fields (all optional, default None)
    ----------------------------------------------------------
    fear_greed_index:
        Alternative.me Crypto Fear & Greed Index value in [0, 100].
        Injected by StrategyEngine._build_mtf_context() (Sprint 32).
    btc_dominance:
        Bitcoin market cap dominance percentage in [0, 100] from CoinGecko.
        High (>55%) = BTC-driven risk-off for altcoins.
        Low (<45%) = alt-season / diversified liquidity.
    market_cap_change_24h:
        Total crypto market cap 24h % change from CoinGecko.
        Positive = expanding market, negative = contracting.
    total_volume_change_24h:
        Total crypto trading volume 24h % change from CoinGecko.
    fed_funds_rate:
        Effective Federal Funds Rate (percent) from FRED.
        Higher rates historically compress risk-asset valuations.
    yield_curve_spread:
        10-Year minus 2-Year Treasury yield spread (percent) from FRED.
        Negative = inverted curve (recession signal, risk-off).
    whale_net_flow:
        Net USD flow of large on-chain transactions in the last hour from
        Whale Alert.  Positive = inflow to exchanges (sell pressure).
        Negative = outflow from exchanges (accumulation).
    """

    htf_bars: dict[str, dict[str, list[OHLCVBar]]] = field(default_factory=dict)
    """
    Higher-timeframe bars keyed by timeframe string, then by symbol.
    E.g. {"4h": {"BTC/USD": [bar1, bar2, ...]}, "1d": {"BTC/USD": [bar1, ...]}}

    Bars are filtered to prevent look-ahead bias: only bars whose entire
    period has completed before the current primary bar are included.
    """

    fear_greed_index: int | None = field(default=None)

    # CoinGecko market structure signals
    btc_dominance: float | None = field(default=None)
    market_cap_change_24h: float | None = field(default=None)
    total_volume_change_24h: float | None = field(default=None)

    # FRED macro-economic signals
    fed_funds_rate: float | None = field(default=None)
    yield_curve_spread: float | None = field(default=None)

    # Whale Alert on-chain flow signals
    whale_net_flow: float | None = field(default=None)
