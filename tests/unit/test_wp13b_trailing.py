"""
tests/unit/test_wp13b_trailing.py
-----------------------------------
WP1.3b-a trailing-stop semantics (SY-13b-05, AC-13B-06) -- WP13B-T-17..T-19.

    T-17  HWM_0 is the gross opening VWAP (the fill), not the signal close   (red)
    T-18  the same-bar check after the entry never ratchets with the anchor
          bar's close; the next bar's close does                             (red)
    T-19  the mark survives a trigger; a quantity change while pending does
          not re-seed it to the current price                                (red)

The new manager API is reached through the instance so the file imports at the
pre-1.3b commit (the red-before-green run then fails test by test).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from common.types import TimeFrame
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.models import Position
from trading.trailing_stop import TrailingStopManager

SYM = "BTC/USD"
T0 = datetime(2026, 3, 1, tzinfo=UTC)
LOT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _pos(qty: str = "1", entry: str = "100") -> Position:
    return Position(
        symbol=SYM,
        run_id="r",
        quantity=Decimal(qty),
        average_entry_price=Decimal(entry),
        current_price=Decimal(entry),
        opened_at=LOT,
        updated_at=LOT,
    )


def _ts(i: int) -> datetime:
    return T0 + timedelta(hours=i)


async def test_t17_hwm0_is_the_gross_fill_not_the_signal_close() -> None:
    """Paper/backtest BUYs fill above the signal close (slippage).  HEAD seeded the
    peak from the signal bar's close; WP1.3b-a seeds the gross fill price."""
    strategy = ScriptedSequenceStrategy("t17", {"script": {0: "buy"}, "target_notional": "1000"})
    runner = BacktestRunner(
        strategies=[strategy],
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        slippage_bps=50,
        seed=7,
        trailing_stop_pct=0.10,
    )
    engine, *_ = runner._build_engine("t17")
    await engine.start("t17")
    bars = scripted_path([100] * 3, first_open=T0 + timedelta(hours=60), symbol=SYM, warmup=50)
    await engine.run_backtest({SYM: bars[:51]})  # the BUY bar only
    fills = engine._execution_engine.get_all_fills()
    assert len(fills) == 1
    fill_price = fills[0].price
    assert fill_price > Decimal("100")  # slippage worked against the taker
    mgr = engine._trailing_stop
    assert mgr is not None
    assert mgr.peak_prices[SYM] == fill_price  # HEAD: Decimal("100")
    position = engine.portfolio.get_position(SYM)
    level = mgr.protective_level(SYM, position)
    assert level == fill_price * (Decimal("1") - Decimal("0.10"))


def test_t18_same_bar_check_after_entry_does_not_ratchet_with_the_anchor_bar_close() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    pos = _pos()
    # fill at 99 while the signal bar (ts 5) closed at 100
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("99"), anchor_bar_ts=_ts(5))
    assert mgr.check(SYM, Decimal("100"), pos, bar_ts=_ts(5)) is None
    assert mgr.peak_prices[SYM] == Decimal("99")  # HEAD: 100 (a pre-entry price)
    # the next bar's close does ratchet
    assert mgr.check(SYM, Decimal("101"), pos, bar_ts=_ts(6)) is None
    assert mgr.peak_prices[SYM] == Decimal("101")
    # a bar_ts of None (legacy direct caller) keeps ratcheting
    assert mgr.check(SYM, Decimal("102"), pos) is None
    assert mgr.peak_prices[SYM] == Decimal("102")


def test_t18b_further_opening_fill_only_raises_the_mark() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("99"), anchor_bar_ts=_ts(5))
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("101"), anchor_bar_ts=None)  # same lot
    assert mgr.peak_prices[SYM] == Decimal("101")
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("95"), anchor_bar_ts=None)  # never lowers
    assert mgr.peak_prices[SYM] == Decimal("101")
    # a NEW lot replaces the mark
    mgr.open_lot(SYM, lot_key=LOT + timedelta(days=1), hwm0=Decimal("50"), anchor_bar_ts=_ts(9))
    assert mgr.peak_prices[SYM] == Decimal("50")


def test_t19_mark_survives_a_trigger_and_a_quantity_change_does_not_reseed() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10, pending_stop_ttl=5)
    full = _pos("1.0")
    mgr.check(SYM, Decimal("100"), full)
    sig = mgr.check(SYM, Decimal("85"), full)
    assert sig is not None
    assert mgr.peak_prices[SYM] == Decimal("100")  # HEAD: popped on trigger

    # partial fill -> residual; the guard clears but the mark is NOT re-seeded from 85
    residual = _pos("0.4")
    again = mgr.check(SYM, Decimal("85"), residual)
    assert again is not None  # same mark -> still below the level -> re-protected now
    assert again.metadata["peak_price"] == "100"
    assert mgr.peak_prices[SYM] == Decimal("100")

    # flat finally drops it
    assert mgr.check(SYM, Decimal("85"), _pos("0")) is None
    assert SYM not in mgr.peak_prices


def test_t19b_protective_level_is_pure_and_none_without_a_mark() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    assert mgr.protective_level(SYM, _pos()) is None
    mgr.check(SYM, Decimal("100"), _pos())
    before = dict(mgr.peak_prices)
    assert mgr.protective_level(SYM, _pos()) == Decimal("90")
    assert mgr.protective_level(SYM, _pos("0")) is None
    assert mgr.peak_prices == before


def test_t19c_resume_seed_never_lowers_and_survives_open_lot() -> None:
    mgr = TrailingStopManager(trailing_stop_pct=0.10)
    mgr.open_lot(SYM, lot_key=LOT, hwm0=Decimal("120"), anchor_bar_ts=_ts(3))
    mgr.seed_peak(SYM, Decimal("110"))
    assert mgr.peak_prices[SYM] == Decimal("120")
    mgr.seed_peak(SYM, Decimal("130"))
    assert mgr.peak_prices[SYM] == Decimal("130")
