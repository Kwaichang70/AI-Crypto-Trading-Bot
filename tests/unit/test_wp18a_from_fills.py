"""
tests/unit/test_wp18a_from_fills.py
-------------------------------------
Unit tests for ``PortfolioAccounting.from_fills`` (WP1.8a S6/A-06, the
orphan-resume portfolio replay).

Mandatory test list (synthesis spec §6, 1.8a):
- replay equals incremental application
- two partial BUYs give the fee-inclusive VWAP
- the position resets when flat
- a partial SELL
- yesterday's losses are excluded
- the peak honours the hint and is never lowered
- input order does not change the result
- no callback fires
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from common.types import OrderSide
from trading.models import Fill
from trading.portfolio import PortfolioAccounting

RUN_ID = "wp18a-from-fills-test"


def _fill(
    *,
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    quantity: str = "0.01",
    price: str = "50000",
    fee: str = "0.5",
    fee_currency: str = "USDT",
    executed_at: datetime,
) -> Fill:
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency=fee_currency,
        executed_at=executed_at,
    )


def _t(hour: int, *, day_offset: int = 0) -> datetime:
    base = datetime(2026, 9, 20, tzinfo=UTC) + timedelta(days=day_offset)
    return base.replace(hour=hour)


class TestReplayMatchesIncremental:
    def test_replay_equals_incremental_application(self) -> None:
        fills = [
            _fill(quantity="0.01", price="50000", fee="0.5", executed_at=_t(9)),
            _fill(quantity="0.02", price="51000", fee="1.0", executed_at=_t(10)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.01",
                price="52000",
                fee="0.5",
                executed_at=_t(11),
            ),
        ]

        incremental = PortfolioAccounting(run_id=RUN_ID, initial_cash=Decimal("10000"))
        for fill in fills:
            incremental.update_position(fill, current_price=fill.price)

        replayed = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills, now=_t(12)
        )

        assert replayed.cash == incremental.cash
        assert replayed.total_realised_pnl == incremental.total_realised_pnl
        assert replayed.total_fees_paid == incremental.total_fees_paid
        pos_a = incremental.get_position("BTC/USDT")
        pos_b = replayed.get_position("BTC/USDT")
        assert pos_a is not None and pos_b is not None
        assert pos_a.quantity == pos_b.quantity
        assert pos_a.average_entry_price == pos_b.average_entry_price
        assert pos_a.realised_pnl == pos_b.realised_pnl


class TestVwapAndPartialSell:
    def test_two_partial_buys_give_fee_inclusive_vwap(self) -> None:
        fills = [
            _fill(quantity="0.01", price="50000", fee="0.5", executed_at=_t(9)),
            _fill(quantity="0.01", price="52000", fee="0.5", executed_at=_t(10)),
        ]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills
        )
        position = portfolio.get_position("BTC/USDT")
        assert position is not None
        expected_avg = (
            (Decimal("50000") * Decimal("0.01") + Decimal("0.5"))
            + (Decimal("52000") * Decimal("0.01") + Decimal("0.5"))
        ) / Decimal("0.02")
        assert position.average_entry_price == expected_avg.quantize(Decimal("0.00000001"))
        assert position.quantity == Decimal("0.02")

    def test_partial_sell_reduces_quantity_and_realises_pnl(self) -> None:
        fills = [
            _fill(quantity="0.02", price="50000", fee="1.0", executed_at=_t(9)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.01",
                price="55000",
                fee="0.5",
                executed_at=_t(10),
            ),
        ]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills
        )
        position = portfolio.get_position("BTC/USDT")
        assert position is not None
        assert position.quantity == Decimal("0.01")
        assert portfolio.total_realised_pnl != Decimal("0")

    def test_position_resets_when_flat(self) -> None:
        fills = [
            _fill(quantity="0.02", price="50000", fee="1.0", executed_at=_t(9)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.02",
                price="55000",
                fee="1.0",
                executed_at=_t(10),
            ),
        ]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills
        )
        position = portfolio.get_position("BTC/USDT")
        assert position is not None
        assert position.is_flat
        assert position.quantity == Decimal("0")


class TestDailyPnlBoundary:
    def test_yesterdays_losses_are_excluded(self) -> None:
        fills = [
            _fill(quantity="0.02", price="50000", fee="1.0", executed_at=_t(9, day_offset=-1)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.02",
                price="40000",  # a loss, but it happened "yesterday"
                fee="1.0",
                executed_at=_t(10, day_offset=-1),
            ),
        ]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID,
            initial_cash=Decimal("10000"),
            fills=fills,
            now=_t(9, day_offset=0),
        )
        assert portfolio.get_daily_pnl() == Decimal("0")
        # the realised loss itself is still tracked in the cumulative total
        assert portfolio.total_realised_pnl < Decimal("0")


class TestPeakHint:
    def test_peak_honours_the_hint_and_is_never_lowered(self) -> None:
        fills = [_fill(quantity="0.01", price="50000", fee="0.5", executed_at=_t(9))]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID,
            initial_cash=Decimal("10000"),
            fills=fills,
            peak_equity_hint=Decimal("99999"),
        )
        assert portfolio.get_peak_equity() == Decimal("99999")

    def test_peak_hint_below_computed_peak_does_not_lower_it(self) -> None:
        fills = [
            _fill(quantity="0.1", price="50000", fee="0.5", executed_at=_t(9)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.1",
                price="60000",
                fee="0.5",
                executed_at=_t(10),
            ),
        ]
        portfolio = PortfolioAccounting.from_fills(
            run_id=RUN_ID,
            initial_cash=Decimal("10000"),
            fills=fills,
            peak_equity_hint=Decimal("1"),
        )
        assert portfolio.get_peak_equity() >= Decimal("10000")


class TestOrderIndependenceAndNoCallback:
    def test_input_order_does_not_change_the_result(self) -> None:
        fills = [
            _fill(quantity="0.01", price="50000", fee="0.5", executed_at=_t(9)),
            _fill(quantity="0.02", price="51000", fee="1.0", executed_at=_t(10)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.01",
                price="52000",
                fee="0.5",
                executed_at=_t(11),
            ),
        ]
        forward = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills, now=_t(12)
        )
        reversed_ = PortfolioAccounting.from_fills(
            run_id=RUN_ID, initial_cash=Decimal("10000"), fills=list(reversed(fills)), now=_t(12)
        )
        assert forward.cash == reversed_.cash
        assert forward.total_realised_pnl == reversed_.total_realised_pnl
        pos_f = forward.get_position("BTC/USDT")
        pos_r = reversed_.get_position("BTC/USDT")
        assert pos_f is not None and pos_r is not None
        assert pos_f.quantity == pos_r.quantity
        assert pos_f.average_entry_price == pos_r.average_entry_price

    def test_no_callback_fires_during_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record_calls: list[object] = []
        original_record_trade = PortfolioAccounting.record_trade

        def _spy_record_trade(self: PortfolioAccounting, trade: object) -> None:
            record_calls.append(trade)
            original_record_trade(self, trade)  # type: ignore[arg-type]

        monkeypatch.setattr(PortfolioAccounting, "record_trade", _spy_record_trade)

        fills = [
            _fill(quantity="0.01", price="50000", fee="0.5", executed_at=_t(9)),
            _fill(
                side=OrderSide.SELL,
                quantity="0.01",
                price="55000",
                fee="0.5",
                executed_at=_t(10),
            ),
        ]
        PortfolioAccounting.from_fills(run_id=RUN_ID, initial_cash=Decimal("10000"), fills=fills)
        assert record_calls == [], (
            "from_fills must never call record_trade (and therefore never "
            "fire on_trade_recorded) -- the DB already holds the completed "
            "TradeResult rows; the replay only rebuilds cash/position state"
        )
