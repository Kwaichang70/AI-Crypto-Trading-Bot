"""
tests/integration/fakes/bar_paths.py
---------------------------------------
Deterministic OHLCV price paths for the WP1.5a BacktestRunner scenarios.

``scripted_path`` builds ``warmup`` flat bars that END right before
``first_open`` followed by one bar per entry of ``closes``; bar ``j`` of the
scenario opens at ``first_open + j * timeframe``.  ``BacktestRunner`` skips
strategy calls for the first ``warmup`` bars (it needs at least 50), so the
scenario's ``on_bar`` call index ``j`` is exactly ``closes[j]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from common.models import OHLCVBar
from common.types import TimeFrame

__all__ = ["TIMEFRAME_DELTA", "scripted_path"]

TIMEFRAME_DELTA: dict[TimeFrame, timedelta] = {
    TimeFrame.ONE_HOUR: timedelta(hours=1),
    TimeFrame.ONE_DAY: timedelta(days=1),
}


def scripted_path(
    closes: Sequence[Decimal | str | int],
    *,
    first_open: datetime,
    symbol: str,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    warmup: int = 50,
    warmup_price: Decimal | str | int | None = None,
) -> list[OHLCVBar]:
    """Bars for ``symbol``: ``warmup`` flat bars, then one bar per close."""
    step = TIMEFRAME_DELTA[timeframe]
    prices = [Decimal(str(c)) for c in closes]
    flat = Decimal(str(warmup_price)) if warmup_price is not None else prices[0]

    bars: list[OHLCVBar] = []
    previous = flat
    start = first_open - warmup * step
    for i in range(warmup + len(prices)):
        close = flat if i < warmup else prices[i - warmup]
        bars.append(
            OHLCVBar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=start + i * step,
                open=previous,
                high=max(previous, close),
                low=min(previous, close),
                close=close,
                volume=Decimal("10"),
            )
        )
        previous = close
    return bars
