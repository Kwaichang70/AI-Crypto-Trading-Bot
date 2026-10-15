"""
tests/unit/_wp15_support.py
----------------------------
Shared helpers for the WP1.5a unit tests (daily-PnL & peak semantics).

Not collected by pytest (leading underscore).  Everything here is
deterministic: injected clocks, fixed UTC dates, no wall-clock reads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from common.types import OrderSide
from trading.models import Fill
from trading.portfolio import PortfolioAccounting

SYMBOL = "BTC/USDT"
INITIAL = Decimal("10000")

#: Day D of the scenarios (UTC).  D+1 is ``D + timedelta(days=1)``.
D = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
D1 = D + timedelta(days=1)
D2 = D + timedelta(days=2)


class Clock:
    """A mutable injected risk clock."""

    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


def make_fill(
    *,
    side: OrderSide = OrderSide.BUY,
    quantity: str = "10",
    price: str = "1000",
    fee: str = "0",
    symbol: str = SYMBOL,
    executed_at: datetime | None = None,
) -> Fill:
    kwargs: dict[str, object] = {}
    if executed_at is not None:
        kwargs["executed_at"] = executed_at
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency="USDT",
        **kwargs,  # type: ignore[arg-type]
    )


def portfolio_with_position(
    clock: Clock,
    *,
    initial: Decimal = INITIAL,
    quantity: str = "10",
    price: str = "1000",
) -> PortfolioAccounting:
    """A portfolio holding ``quantity`` BTC bought at ``price`` (no fee).

    With the defaults NAV == 10 000 == initial cash (cash 0 + 10 x 1000).
    """
    p = PortfolioAccounting("wp15", initial, clock=clock)
    p.update_position(
        make_fill(quantity=quantity, price=price), current_price=Decimal(price)
    )
    return p


def mark(p: PortfolioAccounting, price: str) -> None:
    p.update_market_prices({SYMBOL: Decimal(price)})
