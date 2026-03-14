"""
tests/conftest.py
-----------------
Shared pytest fixtures for the AI Crypto Trading Bot test suite.

All fixtures produce deterministic data with explicit seeds and fixed
timestamps so tests are reproducible across runs.

Usage
-----
Import fixtures by declaring them as function parameters in your test.
Pytest resolves them automatically from this module.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from common.models import OHLCVBar
from common.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    TimeFrame,
)
from trading.models import Fill, Order, Signal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_SYMBOL = "BTC/USDT"
_RUN_ID = "test-run-001"


# ---------------------------------------------------------------------------
# OHLCVBar factories
# ---------------------------------------------------------------------------


def make_bar(
    *,
    symbol: str = _SYMBOL,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    timestamp: datetime | None = None,
    open_: Decimal | float | str = "50000",
    high: Decimal | float | str = "50500",
    low: Decimal | float | str = "49500",
    close: Decimal | float | str = "50200",
    volume: Decimal | float | str = "10",
) -> OHLCVBar:
    """
    Factory function to create a single OHLCVBar with sane defaults.

    The OHLCV values default to a realistic BTC/USDT bar.  All price
    parameters accept ``Decimal``, ``float``, or ``str`` for convenience.
    """
    ts = timestamp or _FIXED_TIMESTAMP
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
    )


@pytest.fixture
def sample_bar() -> OHLCVBar:
    """A single OHLCVBar with default BTC/USDT values."""
    return make_bar()


@pytest.fixture
def sample_bars_rising() -> list[OHLCVBar]:
    """
    50 bars with steadily rising close prices from 50_000 to 54_900.

    Price increases by 100 per bar.  Used for golden-cross and upside
    breakout scenarios.
    """
    bars: list[OHLCVBar] = []
    base_price = Decimal("50000")
    step = Decimal("100")
    for i in range(50):
        close = base_price + step * i
        high = close + Decimal("200")
        low = close - Decimal("200")
        open_ = close - Decimal("50")
        ts = datetime(2024, 1, 1, i, 0, 0, tzinfo=UTC)
        bars.append(
            make_bar(
                timestamp=ts,
                open_=open_,
                high=high,
                low=low,
                close=close,
            )
        )
    return bars


@pytest.fixture
def sample_bars_falling() -> list[OHLCVBar]:
    """
    50 bars with steadily falling close prices from 54_900 to 50_000.

    Price decreases by 100 per bar.  Used for death-cross and downside
    breakout scenarios.
    """
    bars: list[OHLCVBar] = []
    base_price = Decimal("54900")
    step = Decimal("100")
    for i in range(50):
        close = base_price - step * i
        high = close + Decimal("200")
        low = close - Decimal("200")
        open_ = close + Decimal("50")
        ts = datetime(2024, 1, 1, i, 0, 0, tzinfo=UTC)
        bars.append(
            make_bar(
                timestamp=ts,
                open_=open_,
                high=high,
                low=low,
                close=close,
            )
        )
    return bars


def make_bars(
    n: int,
    *,
    start_price: float = 50000.0,
    seed: int = 42,
    symbol: str = _SYMBOL,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
) -> list[OHLCVBar]:
    """
    Factory to create *n* bars with pseudo-random price walk.

    Uses a seeded random generator for determinism.  The resulting bars
    have OHLCVs that are internally consistent (low <= open/close <= high).

    Parameters
    ----------
    n:
        Number of bars to create.
    start_price:
        Starting close price.
    seed:
        Random seed — always provide an explicit value for determinism.
    symbol:
        Trading pair identifier.
    timeframe:
        Candle duration.
    """
    rng = random.Random(seed)  # noqa: S311 — deliberate seeded PRNG for tests
    bars: list[OHLCVBar] = []
    price = start_price

    for i in range(n):
        change_pct = rng.uniform(-0.02, 0.02)
        close = max(1.0, price * (1 + change_pct))
        high = close * rng.uniform(1.000, 1.005)
        low = close * rng.uniform(0.995, 1.000)
        open_ = rng.uniform(low, high)
        volume = rng.uniform(5.0, 50.0)

        ts = datetime(2024, 1, 1, tzinfo=UTC).replace(
            hour=i % 24,
            day=(i // 24) + 1,
        )

        bars.append(
            make_bar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open_=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(volume, 4),
            )
        )
        price = close

    return bars


# ---------------------------------------------------------------------------
# Signal fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_signal() -> Signal:
    """A BUY signal with default values and full confidence."""
    return Signal(
        strategy_id="test-strategy",
        symbol=_SYMBOL,
        direction=SignalDirection.BUY,
        target_position=Decimal("1000"),
        confidence=0.8,
        generated_at=_FIXED_TIMESTAMP,
        metadata={"indicator": "sma", "value": 50200},
    )


# ---------------------------------------------------------------------------
# Order fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_order() -> Order:
    """A LIMIT BUY order in NEW status with a price set."""
    return Order(
        client_order_id=f"{_RUN_ID}-aabbcc112233",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
        status=OrderStatus.NEW,
        created_at=_FIXED_TIMESTAMP,
        updated_at=_FIXED_TIMESTAMP,
    )


@pytest.fixture
def sample_market_order() -> Order:
    """A MARKET SELL order in NEW status (no price field)."""
    return Order(
        client_order_id=f"{_RUN_ID}-ddeeff445566",
        run_id=_RUN_ID,
        symbol=_SYMBOL,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
        price=None,
        status=OrderStatus.NEW,
        created_at=_FIXED_TIMESTAMP,
        updated_at=_FIXED_TIMESTAMP,
    )


# ---------------------------------------------------------------------------
# Fill fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_buy_fill() -> Fill:
    """A BUY fill for 0.01 BTC at 50_000 USDT with 0.6% taker fee."""
    qty = Decimal("0.01")
    price = Decimal("50000")
    fee = qty * price * Decimal("0.006")
    return Fill(
        order_id=uuid4(),
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        quantity=qty,
        price=price,
        fee=fee,
        fee_currency="USDT",
        is_maker=False,
        executed_at=_FIXED_TIMESTAMP,
    )


@pytest.fixture
def sample_sell_fill() -> Fill:
    """A SELL fill for 0.01 BTC at 51_000 USDT with 0.6% taker fee."""
    qty = Decimal("0.01")
    price = Decimal("51000")
    fee = qty * price * Decimal("0.006")
    return Fill(
        order_id=uuid4(),
        symbol=_SYMBOL,
        side=OrderSide.SELL,
        quantity=qty,
        price=price,
        fee=fee,
        fee_currency="USDT",
        is_maker=False,
        executed_at=_FIXED_TIMESTAMP,
    )
