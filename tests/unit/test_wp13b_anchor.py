"""
tests/unit/test_wp13b_anchor.py
---------------------------------
WP1.3b-a (ATR-bracket anchoring, all modes) -- tests WP13B-T-01..T-16 and T-31
of ``reports/vp2-wp1.3b/synthesis-spec.md`` section 7.

The new API is reached through the module (``bx.<name>``) rather than by
``from ... import`` so this file imports at the pre-1.3b commit too: the
red-before-green run then fails test by test (AttributeError for the new API,
AssertionError for the behaviour tests) instead of failing at collection.

    T-01  fixed levels equal HEAD's formula and never move
    T-02  ATR levels ignore every later ATR value               (behaviour, red)
    T-03  crash-bar pin: anchored SL 94 fires, recomputed 92.29 would not (red)
    T-04  a held lot keeps its levels when the current ATR is None        (red)
    T-05  flat drops the anchor; lot_key mismatch -> lazy re-anchor
    T-06  partials of the opening order re-blend E, ATR0 frozen
    T-07  an add (other order) keeps the levels and logs it
    T-08  partial SELL: levels unchanged, residual protected
    T-09  provisional anchor finalised exactly once
    T-10  deferred anchor (ATR None at entry) finalised once, then frozen
    T-11  ATR stop <= 0 means no stop
    T-12  select_anchor_bar table
    T-13  lot_entry_from_fills table
    T-14  stop_inside_entry_cost is logged, never dropped (U-4)
    T-15  blend_entry_price refactor leaves PortfolioAccounting unchanged
    T-16  projected_levels == anchor levels (WP1.10 hook)
    T-31  AST: the backtest anchor path reads no clock
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from structlog.testing import capture_logs

import trading.bracket_exit as bx
import trading.strategy_engine as se
from common.models import OHLCVBar
from common.types import OrderSide, OrderType, RunMode, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.bracket_exit import BracketExitManager
from trading.models import Fill, Order, Position
from trading.portfolio import PortfolioAccounting

SYM = "BTC/USD"
T0 = datetime(2026, 3, 1, tzinfo=UTC)
LOT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
TF_S = 3600


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _pos(
    entry: str = "100", qty: str = "1", opened_at: datetime = LOT, symbol: str = SYM
) -> Position:
    return Position(
        symbol=symbol,
        run_id="r",
        quantity=Decimal(qty),
        average_entry_price=Decimal(entry),
        current_price=Decimal(entry),
        opened_at=opened_at,
        updated_at=opened_at,
    )


def _bar(
    i: int, close: str = "100", *, high: str | None = None, low: str | None = None
) -> OHLCVBar:
    c = Decimal(close)
    return OHLCVBar(
        symbol=SYM,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=T0 + timedelta(hours=i),
        open=c,
        high=Decimal(high) if high is not None else c + 1,
        low=Decimal(low) if low is not None else c - 1,
        close=c,
        volume=Decimal("10"),
    )


def _fill(
    order_id: UUID,
    *,
    side: OrderSide = OrderSide.BUY,
    qty: str = "1",
    price: str = "100",
    fee: str = "0",
    at: datetime = LOT,
    symbol: str = SYM,
) -> Fill:
    return Fill(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency="USD",
        executed_at=at,
    )


def _order(order_id: UUID, *, created_at: datetime, side: OrderSide = OrderSide.BUY) -> Order:
    return Order(
        order_id=order_id,
        client_order_id=f"r-{order_id.hex[:12]}",
        run_id="r",
        symbol=SYM,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        created_at=created_at,
    )


def _atr_mgr(sl: float | None = 2.0, tp: float | None = 3.0) -> BracketExitManager:
    return BracketExitManager(bracket_mode="atr", atr_sl_multiplier=sl, atr_tp_multiplier=tp)


def _open(
    mgr: BracketExitManager,
    *,
    entry: str = "100",
    atr: str | None = None,
    order_id: UUID | None = None,
    final: bool = True,
    source: Any = "entry",
    lot_key: datetime = LOT,
) -> Any:
    return mgr.open_lot(
        SYM,
        lot_key=lot_key,
        opening_order_id=order_id,
        anchor_at=None,
        anchor_bar_ts=None,
        entry_basis=Decimal(entry),
        basis_qty=Decimal("1"),
        atr=Decimal(atr) if atr is not None else None,
        source=source,
        final=final,
    )


# ---------------------------------------------------------------------------
# T-01 .. T-04  levels are frozen prices
# ---------------------------------------------------------------------------


def test_t01_fixed_levels_equal_head_formula_and_never_move() -> None:
    mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.04)
    anchor = _open(mgr, entry="100")
    assert anchor.stop_price == Decimal("100") * (Decimal("1") - Decimal("0.02"))
    assert anchor.take_profit_price == Decimal("100") * (Decimal("1") + Decimal("0.04"))
    pos = _pos("100")
    for i in range(50):  # prices inside [98, 104]
        assert mgr.check(SYM, Decimal("98.5") + Decimal(i) * Decimal("0.1"), pos) is None
    assert mgr.get_anchor(SYM) == anchor


def test_t02_atr_levels_ignore_every_later_atr_value() -> None:
    """Red at HEAD: HEAD recomputes the level from the ATR passed on every bar."""
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    pos = _pos("100")
    # first check freezes ATR0 = 2 -> stop 96, tp 106
    assert mgr.check(SYM, Decimal("100"), pos, atr_value=Decimal("2")) is None
    # ATR doubles over 20 bars: a recomputed stop would slide to 92
    for k in range(20):
        atr = Decimal("2") + Decimal(k) / Decimal("10")
        assert mgr.check(SYM, Decimal("99"), pos, atr_value=atr) is None
    sig = mgr.check(SYM, Decimal("95"), pos, atr_value=Decimal("5"))
    assert sig is not None and sig.metadata["stop_loss"] is True
    assert Decimal(sig.metadata["stop_price"]) == Decimal("96")
    assert Decimal(sig.metadata["atr_anchor"]) == Decimal("2")

    # ATR halves: a recomputed stop would tighten to 98 and fire at 97
    mgr2 = _atr_mgr(sl=2.0, tp=3.0)
    assert mgr2.check(SYM, Decimal("100"), pos, atr_value=Decimal("2")) is None
    assert mgr2.check(SYM, Decimal("97"), pos, atr_value=Decimal("1")) is None


def test_t03_crash_bar_pin_anchored_stop_fires_where_recomputed_would_not() -> None:
    """E=100, ATR0=4, k=1.5 -> SL 94.  Crash bar: prev close 98, low 78, close 92.8.
    The bar's own TR is 20, so the Wilder ATR becomes (13*4+20)/14 = 5.142857 and a
    RECOMPUTED stop would be 100 - 1.5*5.142857 = 92.29 -- below the 92.8 close."""
    mgr = _atr_mgr(sl=1.5, tp=None)
    pos = _pos("100")
    assert mgr.check(SYM, Decimal("98"), pos, atr_value=Decimal("4")) is None  # freezes ATR0
    atr_with_crash = (Decimal(13) * Decimal(4) + Decimal(20)) / Decimal(14)
    recomputed_stop = Decimal("100") - Decimal("1.5") * atr_with_crash
    assert recomputed_stop.quantize(Decimal("0.01")) == Decimal("92.29")
    assert Decimal("92.8") > recomputed_stop  # negative pin: the recomputed stop would hold
    sig = mgr.check(SYM, Decimal("92.8"), pos, atr_value=atr_with_crash)
    assert sig is not None
    assert Decimal(sig.metadata["stop_price"]) == Decimal("94")


def test_t04_held_lot_keeps_its_levels_when_current_atr_is_none() -> None:
    """FM9: HEAD returns no exit at all when the current ATR is None."""
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    pos = _pos("100")
    assert mgr.check(SYM, Decimal("100"), pos, atr_value=Decimal("2")) is None
    sig = mgr.check(SYM, Decimal("90"), pos, atr_value=None)
    assert sig is not None and sig.metadata["stop_loss"] is True


# ---------------------------------------------------------------------------
# T-05 .. T-11  lifecycle
# ---------------------------------------------------------------------------


def test_t05_flat_drops_anchor_and_lot_key_mismatch_reanchors() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry="100", atr="2")
    assert mgr.get_anchor(SYM).stop_price == Decimal("96")  # type: ignore[union-attr]

    # a different lot (opened_at differs) must never be judged by the stale anchor
    other = _pos("200", opened_at=LOT + timedelta(days=1))
    with capture_logs() as logs:
        assert mgr.check(SYM, Decimal("190"), other, atr_value=Decimal("10")) is None
    assert any(e["event"] == "bracket_exit.anchor_lot_mismatch" for e in logs)
    fresh = mgr.get_anchor(SYM)
    assert fresh is not None and fresh.source == "late"
    assert fresh.stop_price == Decimal("180")  # 200 - 2*10, NOT 96

    # flat drops the anchor; a reopen anchors anew
    flat = _pos("200", qty="0", opened_at=LOT + timedelta(days=1))
    assert mgr.check(SYM, Decimal("190"), flat) is None
    assert mgr.get_anchor(SYM) is None
    _open(mgr, entry="50", atr="1", lot_key=LOT + timedelta(days=2))
    assert mgr.get_anchor(SYM).stop_price == Decimal("48")  # type: ignore[union-attr]


def test_t06_opening_order_partials_reblend_entry_atr_frozen() -> None:
    oid = uuid4()
    f1 = _fill(oid, qty="0.6", price="100", fee="0.3", at=LOT)
    f2 = _fill(oid, qty="0.4", price="104", fee="0.2", at=LOT + timedelta(seconds=1))
    lot1 = bx.lot_entry_from_opening_fills(SYM, None, [f1])
    lot2 = bx.lot_entry_from_opening_fills(SYM, None, [f1, f2])
    assert lot1 is not None and lot2 is not None
    # E per blend_entry_price: (0.6*100+0.3 + 0.4*104+0.2) / 1.0 = 102.1
    assert lot2.entry_basis == Decimal("102.1")

    mgr = _atr_mgr(sl=2.0, tp=3.0)
    _open(mgr, entry=str(lot1.entry_basis), atr="2", order_id=oid)
    updated = mgr.add_opening_fill(
        SYM, order_id=oid, entry_basis=lot2.entry_basis, basis_qty=lot2.basis_qty
    )
    assert updated is not None
    assert updated.atr == Decimal("2")  # frozen
    assert updated.stop_price == Decimal("102.1") - Decimal("4")
    assert updated.take_profit_price == Decimal("102.1") + Decimal("6")


def test_t07_add_from_another_order_keeps_levels_and_logs() -> None:
    oid = uuid4()
    mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.04)
    before = _open(mgr, entry="100", order_id=oid)
    with capture_logs() as logs:
        below = mgr.add_opening_fill(
            SYM, order_id=uuid4(), entry_basis=Decimal("100"), basis_qty=Decimal("1")
        )
        above = mgr.add_opening_fill(
            SYM, order_id=uuid4(), entry_basis=Decimal("100"), basis_qty=Decimal("1")
        )
    assert below == before and above == before
    assert [e["event"] for e in logs].count("bracket_exit.anchor_kept_on_add") == 2


def test_t08_partial_sell_keeps_levels_and_residual_is_protected() -> None:
    mgr = BracketExitManager(stop_loss_pct=0.02)
    before = _open(mgr, entry="100")
    residual = _pos("100", qty="0.4")
    assert mgr.check(SYM, Decimal("99"), residual) is None
    assert mgr.get_anchor(SYM) == before
    sig = mgr.check(SYM, Decimal("97"), residual)
    assert sig is not None and sig.metadata["stop_loss"] is True


def test_t09_provisional_anchor_is_finalised_exactly_once() -> None:
    mgr = _atr_mgr(sl=2.0, tp=3.0)
    prov = _open(mgr, entry="100", atr="2", final=False, source="entry_provisional")
    assert prov.final is False and prov.stop_price == Decimal("96")  # protected at once
    first = mgr.finalize_anchor(
        SYM, atr=Decimal("3"), anchor_bar_ts=T0, source="entry"
    )
    assert first is not None and first.final and first.source == "entry"
    assert first.stop_price == Decimal("94")
    again = mgr.finalize_anchor(SYM, atr=Decimal("9"), anchor_bar_ts=T0, source="entry")
    assert again is None
    assert mgr.get_anchor(SYM) == first


def test_t11_atr_stop_at_or_below_zero_means_no_stop() -> None:
    mgr = _atr_mgr(sl=20.0, tp=3.0)
    anchor = _open(mgr, entry="100", atr="10")  # 100 - 200 < 0
    assert anchor.stop_price is None and anchor.take_profit_price == Decimal("130")
    assert mgr.check(SYM, Decimal("1"), _pos("100")) is None


# ---------------------------------------------------------------------------
# T-12  select_anchor_bar
# ---------------------------------------------------------------------------


def test_t12_select_anchor_bar_table() -> None:
    bars = [_bar(i) for i in range(5)]  # open times T0+0h..T0+4h; closes T0+1h..T0+5h
    close_of = lambda i: T0 + timedelta(hours=i + 1)  # noqa: E731
    # inclusive close
    assert bx.select_anchor_bar(bars, close_of(2), TF_S) == 2
    # one microsecond before the close -> the previous bar
    assert bx.select_anchor_bar(bars, close_of(2) - timedelta(microseconds=1), TF_S) == 1
    # well after the last close -> the newest bar
    assert bx.select_anchor_bar(bars, T0 + timedelta(days=9), TF_S) == 4
    # before the first close -> None; empty -> None
    assert bx.select_anchor_bar(bars, T0 + timedelta(minutes=59), TF_S) is None
    assert bx.select_anchor_bar([], T0, TF_S) is None
    # gaps: bars 0,1,3 -> an instant inside the missing bar 2 picks bar 1
    gapped = [bars[0], bars[1], bars[3]]
    assert bx.select_anchor_bar(gapped, close_of(2), TF_S) == 1
    assert bx.select_anchor_bar(gapped, close_of(3), TF_S) == 2
    # weekly bars
    week = 7 * 86400
    wbars = [
        OHLCVBar(
            symbol=SYM,
            timeframe=TimeFrame.ONE_WEEK,
            timestamp=T0 + timedelta(weeks=i),
            open=Decimal(1),
            high=Decimal(2),
            low=Decimal(1),
            close=Decimal(1),
            volume=Decimal(1),
        )
        for i in range(3)
    ]
    assert bx.select_anchor_bar(wbars, T0 + timedelta(weeks=2), week) == 1
    assert bx.select_anchor_bar(wbars, T0 + timedelta(weeks=2, seconds=-1), week) == 0


def test_t12b_anchor_bar_timestamp_is_arithmetic_on_the_bar_grid() -> None:
    bars = [_bar(i) for i in range(10, 15)]
    # newer than the window (a boundary passed): the next open
    assert bx.anchor_bar_timestamp(bars, T0 + timedelta(hours=16, seconds=2), TF_S) == (
        T0 + timedelta(hours=15)
    )
    # older than the window (a resumed lot)
    assert bx.anchor_bar_timestamp(bars, T0 + timedelta(hours=4, seconds=2), TF_S) == (
        T0 + timedelta(hours=3)
    )
    # inside the window
    assert bx.anchor_bar_timestamp(bars, T0 + timedelta(hours=13), TF_S) == T0 + timedelta(hours=12)
    assert bx.anchor_bar_timestamp([], T0, TF_S) is None


def test_t12c_window_is_contiguous() -> None:
    assert bx.window_is_contiguous([_bar(i) for i in range(5)], TF_S)
    assert not bx.window_is_contiguous([_bar(0), _bar(1), _bar(3)], TF_S)
    assert bx.window_is_contiguous([], TF_S) and bx.window_is_contiguous([_bar(0)], TF_S)


# ---------------------------------------------------------------------------
# T-13  lot_entry_from_fills
# ---------------------------------------------------------------------------


def _portfolio_from(fills: list[Fill]) -> PortfolioAccounting:
    return PortfolioAccounting.from_fills("r", Decimal("100000"), fills)


def test_t13_lot_entry_from_fills_table() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    t = LOT

    # single fill
    f = _fill(a, qty="2", price="100", fee="1", at=t)
    lot = bx.lot_entry_from_fills([f], [], SYM)
    assert lot is not None and lot.opening_order_id == a
    assert lot.entry_basis == _portfolio_from([f]).get_position(SYM).average_entry_price  # type: ignore[union-attr]
    assert lot.entry_gross_vwap == Decimal("100") and lot.basis_qty == Decimal("2")
    assert lot.anchor_at == t  # no order known -> the fill's executed_at

    # two partials of the opening order == the portfolio's blend
    p1 = _fill(a, qty="1", price="100", fee="0.5", at=t)
    p2 = _fill(a, qty="1", price="102", fee="0.5", at=t + timedelta(seconds=1))
    lot = bx.lot_entry_from_fills([p2, p1], [], SYM)  # input order is irrelevant
    assert lot is not None
    assert lot.entry_basis == _portfolio_from([p1, p2]).get_position(SYM).average_entry_price  # type: ignore[union-attr]
    assert lot.entry_gross_vwap == Decimal("101") and lot.basis_qty == Decimal("2")

    # an add from another order is ignored (SY-13b-03)
    add = _fill(b, qty="1", price="80", fee="0", at=t + timedelta(minutes=5))
    lot = bx.lot_entry_from_fills([p1, p2, add], [], SYM)
    assert lot is not None and lot.opening_order_id == a and lot.basis_qty == Decimal("2")
    assert lot.entry_basis == _portfolio_from([p1, p2]).get_position(SYM).average_entry_price  # type: ignore[union-attr]

    # close, then reopen with another order: the second lot
    sell = _fill(a, side=OrderSide.SELL, qty="2", price="110", at=t + timedelta(hours=1))
    reopen = _fill(c, qty="3", price="90", fee="0", at=t + timedelta(hours=2))
    lot = bx.lot_entry_from_fills([p1, p2, sell, reopen], [], SYM)
    assert lot is not None and lot.opening_order_id == c and lot.entry_basis == Decimal("90")

    # fully closed -> None
    assert bx.lot_entry_from_fills([p1, p2, sell], [], SYM) is None
    assert bx.lot_entry_from_fills([], [], SYM) is None

    # a partial SELL does not end the lot
    part = _fill(a, side=OrderSide.SELL, qty="0.5", price="110", at=t + timedelta(hours=1))
    lot = bx.lot_entry_from_fills([p1, p2, part], [], SYM)
    assert lot is not None and lot.opening_order_id == a and lot.basis_qty == Decimal("2")

    # same-timestamp BUY and SELL: the BUY replays first (replay_sort_key)
    s_buy = _fill(a, qty="1", price="100", at=t)
    s_sell = _fill(b, side=OrderSide.SELL, qty="1", price="101", at=t)
    assert bx.lot_entry_from_fills([s_sell, s_buy], [], SYM) is None

    # other symbols are ignored
    other = _fill(a, qty="1", price="5", at=t, symbol="ETH/USD")
    assert bx.lot_entry_from_fills([other], [], SYM) is None


def test_t13b_anchor_at_is_the_earlier_of_order_created_and_first_fill() -> None:
    oid = uuid4()
    fill = _fill(oid, at=LOT)
    early = _order(oid, created_at=LOT - timedelta(seconds=2))
    late_import = _order(oid, created_at=LOT + timedelta(days=3))  # WP1.8b import time
    lot = bx.lot_entry_from_fills([fill], [early], SYM)
    assert lot is not None and lot.anchor_at == LOT - timedelta(seconds=2)
    lot = bx.lot_entry_from_fills([fill], [late_import], SYM)
    assert lot is not None and lot.anchor_at == LOT  # created_at later than the fill
    # naive datetimes are treated as UTC
    naive = _order(oid, created_at=LOT.replace(tzinfo=None) - timedelta(seconds=1))
    lot = bx.lot_entry_from_fills([fill], [naive], SYM)
    assert lot is not None and lot.anchor_at == LOT - timedelta(seconds=1)


# ---------------------------------------------------------------------------
# T-15  blend_entry_price
# ---------------------------------------------------------------------------


def test_t15_blend_entry_price_matches_portfolio_accounting_goldens() -> None:
    from trading.portfolio import blend_entry_price

    oid = uuid4()
    f1 = _fill(oid, qty="0.5", price="100", fee="0.3", at=LOT)
    f2 = _fill(oid, qty="0.5", price="110", fee="0.33", at=LOT + timedelta(seconds=1))
    # hand-computed goldens: (50 + 0.3)/0.5 = 100.6 ; (50.3 + 55.33)/1.0 = 105.63
    first = blend_entry_price(Decimal(0), Decimal(0), f1)
    assert first == Decimal("100.6")
    second = blend_entry_price(Decimal("0.5"), first, f2)
    assert second == Decimal("105.63")
    pf = PortfolioAccounting(run_id="r", initial_cash=Decimal("10000"))
    pf.update_position(f1, Decimal("100"))
    assert pf.get_position(SYM).average_entry_price == first  # type: ignore[union-attr]
    pf.update_position(f2, Decimal("110"))
    assert pf.get_position(SYM).average_entry_price == second  # type: ignore[union-attr]
    # rounding: 8 dp, ROUND_HALF_UP
    odd = _fill(oid, qty="3", price="10", fee="0.00000001", at=LOT)
    assert blend_entry_price(Decimal(0), Decimal(0), odd) == Decimal("10.00000000")
    odd2 = _fill(oid, qty="0.00000003", price="1", fee="0", at=LOT)
    assert blend_entry_price(Decimal(0), Decimal(0), odd2) == Decimal("1.00000000")


# ---------------------------------------------------------------------------
# engine-level (backtest mode, real components)
# ---------------------------------------------------------------------------


def _bt_runner(
    script: dict[int, str], *, bracket: dict[str, Any] | None, trailing: float | None = None,
    allow_pyramiding: bool = False, targets: dict[int, str] | None = None,
) -> BacktestRunner:
    strategy = ScriptedSequenceStrategy(
        "wp13b", {"script": script, "target_notional": "1000", "targets": targets or {}}
    )
    return BacktestRunner(
        strategies=[strategy],
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        bracket_config=bracket,
        trailing_stop_pct=trailing,
        allow_pyramiding=allow_pyramiding,
    )


def _wavy_bars(n: int, *, amp: int = 2, last: str | None = None) -> list[OHLCVBar]:
    closes = [100 + (i % 2) * amp for i in range(n)]
    if last is not None:
        closes[-1] = int(last)
    return scripted_path(
        closes, first_open=T0 + timedelta(hours=n), symbol=SYM, warmup=0, warmup_price=100
    )


async def _engine(runner: BacktestRunner) -> se.StrategyEngine:
    engine, _pf, _rm, _ex = runner._build_engine("wp13b")
    await engine.start("wp13b")
    return engine


async def test_t10_deferred_anchor_waits_for_a_computable_atr_then_freezes() -> None:
    """The BUY comes at bar 50 (backtest warmup), trailing is configured so the entry is
    allowed.  WP1.3b-a FX-01 caps the ATR period at 49, so the 51 bars of history at the
    entry would already carry an ATR; the first 55 bars are therefore FLAT (ATR = 0 ->
    not computable).  The anchor starts `entry_deferred` (no bracket levels), is finalised
    once at the first bar whose window has an ATR (the first volatile bar), and is frozen
    afterwards."""
    # Wilder smoothing from a flat start is slow (ATR ~ 2/49 at the first volatile bar),
    # so the widest allowed multiplier keeps the wavy closes from stopping the lot out.
    bracket = {
        "bracket_mode": "atr",
        "bracket_atr_sl_multiplier": 20.0,
        "bracket_atr_period": 49,
    }
    runner = _bt_runner({0: "buy"}, bracket=bracket, trailing=0.20)
    engine = await _engine(runner)
    closes = [100] * 55 + [100 + ((i + 1) % 2) * 2 for i in range(75)]
    bars = scripted_path(
        closes, first_open=T0 + timedelta(hours=130), symbol=SYM, warmup=0, warmup_price=100
    )
    bracket_mgr = engine._bracket_exit
    assert bracket_mgr is not None
    seen: list[tuple[int, str, bool]] = []

    orig = bracket_mgr.finalize_anchor

    def spy(symbol: str, **kw: Any) -> Any:
        res = orig(symbol, **kw)
        seen.append((engine.bar_count, kw["source"], res is not None))
        return res

    bracket_mgr.finalize_anchor = spy  # type: ignore[method-assign]
    await engine.run_backtest({SYM: bars})
    anchor = bracket_mgr.get_anchor(SYM)
    assert anchor is not None and anchor.final and anchor.atr is not None
    assert anchor.source == "entry_deferred"
    assert len([s for s in seen if s[2]]) == 1  # exactly one effective finalisation
    # frozen at the finalising bar's window, and later bars never moved it
    idx = next(i for i, b in enumerate(bars) if b.timestamp == anchor.anchor_bar_ts)
    assert anchor.atr == se.atr_for_bars(bars[: idx + 1], 49)
    assert idx < len(bars) - 1


async def test_t14_stop_inside_entry_cost_is_logged_and_never_drops_the_entry() -> None:
    bracket = {"bracket_mode": "atr", "bracket_atr_sl_multiplier": 0.1, "bracket_atr_period": 14}
    runner = _bt_runner({0: "buy"}, bracket=bracket)
    engine = await _engine(runner)
    # tiny range -> ATR ~0.02 -> d = 0.1*0.02/100 = 0.00002 << c_side
    bars = scripted_path(
        [100] * 60, first_open=T0 + timedelta(hours=80), symbol=SYM, warmup=40, warmup_price=100
    )
    bars = [
        b.model_copy(update={"high": b.close + Decimal("0.02"), "low": b.close - Decimal("0.02")})
        for b in bars
    ]
    with capture_logs() as logs:
        await engine.run_backtest({SYM: bars})
    events = [e["event"] for e in logs]
    assert "entry.stop_inside_entry_cost" in events
    entry_log = next(e for e in logs if e["event"] == "entry.stop_inside_entry_cost")
    assert {"d", "c_side", "c_rt", "symbol"} <= set(entry_log)
    # the entry was NOT dropped (U-4: log only): the BUY filled
    fills = engine._execution_engine.get_all_fills()
    assert [f.side for f in fills][:1] == [OrderSide.BUY]


def test_t14b_cost_log_levels() -> None:
    runner = _bt_runner({}, bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05})
    engine, *_ = runner._build_engine("x")
    c_side = Decimal(str(engine._exit_cost_model.c_side))
    c_rt = Decimal(str(engine._exit_cost_model.c_rt))

    def anchor(stop: Decimal) -> Any:
        return bx.BracketAnchor(
            symbol=SYM, lot_key=LOT, opening_order_id=None, anchor_at=None, anchor_bar_ts=None,
            entry_basis=Decimal("100"), basis_qty=Decimal("1"), atr=None, stop_price=stop,
            take_profit_price=None, source="entry", final=True,
        )

    with capture_logs() as logs:
        engine._log_stop_cost(anchor(Decimal("100") * (1 - c_side)))  # d == c_side
        engine._log_stop_cost(anchor(Decimal("100") * (1 - (c_side + c_rt) / 2)))
        engine._log_stop_cost(anchor(Decimal("100") * (1 - c_rt * 2)))  # clearly outside
    assert [e["event"] for e in logs] == [
        "entry.stop_inside_entry_cost",
        "entry.stop_inside_round_trip_cost",
    ]


async def test_t16_projected_levels_equal_anchor_levels_when_rule_bar_is_processed_bar() -> None:
    bracket = {"bracket_mode": "atr", "bracket_atr_sl_multiplier": 1.5,
               "bracket_atr_tp_multiplier": 3.0, "bracket_atr_period": 14}
    runner = _bt_runner({0: "buy"}, bracket=bracket)
    engine = await _engine(runner)
    bars = _wavy_bars(70, amp=3)
    # stop right after the entry bar (call index 0 == bar index 50) so the anchor is fresh
    await engine.run_backtest({SYM: bars[:51]})
    mgr = engine._bracket_exit
    assert mgr is not None
    anchor = mgr.get_anchor(SYM)
    assert anchor is not None and anchor.source == "entry"
    processed_window = bars[:51]
    expected = mgr.projected_levels(anchor.entry_basis, se.atr_for_bars(processed_window, 14))
    assert (anchor.stop_price, anchor.take_profit_price) == expected
    assert anchor.anchor_bar_ts == processed_window[-1].timestamp


# ---------------------------------------------------------------------------
# engine hook (``_on_fill_routed``): opening-order partials, adds, late fills
# ---------------------------------------------------------------------------


def _apply(engine: se.StrategyEngine, order: Order | None, fill: Fill) -> None:
    """What ``_route_exit_fills`` does for one fill: portfolio first, then the hook."""
    pre = engine.portfolio.get_position(fill.symbol)
    engine.portfolio.update_position(fill, fill.price)
    engine._on_fill_routed(order, fill, pre)


async def test_t06b_hook_reblends_a_second_fill_of_the_opening_order_atr_frozen() -> None:
    bracket = {"bracket_mode": "atr", "bracket_atr_sl_multiplier": 2.0,
               "bracket_atr_tp_multiplier": 3.0, "bracket_atr_period": 14}
    engine = await _engine(_bt_runner({}, bracket=bracket))
    bars = _wavy_bars(60, amp=2)
    engine._current_history = {SYM: bars}
    oid = uuid4()
    order = _order(oid, created_at=LOT)
    f1 = _fill(oid, qty="0.6", price="100", fee="0.3", at=LOT)
    f2 = _fill(oid, qty="0.4", price="104", fee="0.2", at=LOT + timedelta(seconds=1))
    _apply(engine, order, f1)
    mgr = engine._bracket_exit
    assert mgr is not None
    first = mgr.get_anchor(SYM)
    assert first is not None and first.atr is not None
    _apply(engine, order, f2)
    second = mgr.get_anchor(SYM)
    assert second is not None
    assert second.atr == first.atr  # ATR0 frozen
    assert second.entry_basis == Decimal("102.1")
    assert second.entry_basis == engine.portfolio.get_position(SYM).average_entry_price  # type: ignore[union-attr]
    assert second.stop_price == Decimal("102.1") - 2 * first.atr


async def test_t07b_hook_keeps_the_levels_on_an_add_from_another_order() -> None:
    engine = await _engine(
        _bt_runner({}, bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05,
                                "bracket_take_profit_pct": 0.10}, allow_pyramiding=True)
    )
    engine._current_history = {SYM: _wavy_bars(60)}
    first_oid = uuid4()
    _apply(engine, _order(first_oid, created_at=LOT), _fill(first_oid, price="100", at=LOT))
    mgr = engine._bracket_exit
    assert mgr is not None
    before = mgr.get_anchor(SYM)
    assert before is not None
    for price in ("90", "120"):  # an add below, then above, the first entry
        oid = uuid4()
        with capture_logs() as logs:
            add_fill = _fill(oid, price=price, at=LOT + timedelta(minutes=1))
            _apply(engine, _order(oid, created_at=LOT), add_fill)
        assert mgr.get_anchor(SYM) == before
        assert any(e["event"] == "bracket_exit.anchor_kept_on_add" for e in logs)
    # the position average moved, the anchored levels did not
    assert engine.portfolio.get_position(SYM).average_entry_price != before.entry_basis  # type: ignore[union-attr]


async def test_t27b_backtest_late_fill_uses_the_recorded_in_memory_signal_bar() -> None:
    bracket = {"bracket_mode": "atr", "bracket_atr_sl_multiplier": 2.0, "bracket_atr_period": 14}
    engine = await _engine(_bt_runner({}, bracket=bracket))
    bars = _wavy_bars(70, amp=2)
    signal_idx = 55
    oid = uuid4()
    engine._order_signal_bar_ts[oid] = bars[signal_idx].timestamp  # recorded at order creation
    # the fill is routed 8 bars later (a resting order): the history now ends at bar 63
    engine._current_history = {SYM: bars[:64]}
    _apply(engine, _order(oid, created_at=LOT), _fill(oid, price="100", at=LOT))
    anchor = engine._bracket_exit.get_anchor(SYM)  # type: ignore[union-attr]
    assert anchor is not None and anchor.source == "entry"
    assert anchor.anchor_bar_ts == bars[signal_idx].timestamp
    assert anchor.atr == se.atr_for_bars(bars[: signal_idx + 1], 14)


async def test_t05b_the_hook_never_raises_and_a_sell_ends_the_lot() -> None:
    fixed = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    engine = await _engine(_bt_runner({}, bracket=fixed))
    engine._current_history = {SYM: _wavy_bars(60)}
    oid = uuid4()
    buy = _fill(oid, qty="2", price="100", at=LOT)
    _apply(engine, _order(oid, created_at=LOT), buy)
    assert SYM in engine._lots
    sell = _fill(uuid4(), side=OrderSide.SELL, qty="2", price="101", at=LOT + timedelta(hours=1))
    _apply(engine, None, sell)
    assert SYM not in engine._lots
    # a broken portfolio must not break fill routing
    engine._portfolio = None  # type: ignore[assignment]
    with capture_logs() as logs:
        engine._on_fill_routed(None, buy, None)
    assert any(e["event"] == "engine.fill_hook_error" for e in logs)


def test_r04_exit_not_submitted_table() -> None:
    from common.types import OrderStatus

    def order(status: OrderStatus, filled: str = "0") -> Order:
        base = _order(uuid4(), created_at=LOT, side=OrderSide.SELL)
        return base.model_copy(update={"status": status, "filled_quantity": Decimal(filled)})

    ns = se.StrategyEngine._exit_not_submitted
    assert ns([]) is True
    assert ns([order(OrderStatus.REJECTED)]) is True
    assert ns([order(OrderStatus.CANCELED), order(OrderStatus.EXPIRED)]) is True
    # accepted / in flight / filled / partially filled-then-cancelled all count as submitted
    assert ns([order(OrderStatus.OPEN)]) is False
    assert ns([order(OrderStatus.PENDING_SUBMIT)]) is False
    assert ns([order(OrderStatus.PARTIAL, "0.5")]) is False
    assert ns([order(OrderStatus.FILLED, "1")]) is False
    assert ns([order(OrderStatus.CANCELED, "0.4")]) is False
    assert ns([order(OrderStatus.REJECTED), order(OrderStatus.OPEN)]) is False


async def test_cf13a15_atr_period_is_normalised_upstream_not_defaulted_in_the_engine() -> None:
    engine = await _engine(
        _bt_runner({}, bracket={"bracket_mode": "atr", "bracket_atr_sl_multiplier": 2.0})
    )
    assert engine._bracket_exit is not None and engine._bracket_exit.atr_period == 14
    tree = ast.parse(textwrap.dedent(inspect.getsource(se.StrategyEngine.__init__)))
    defaults = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.BoolOp)
        and isinstance(n.op, ast.Or)
        and any(isinstance(v, ast.Constant) and v.value == 14 for v in n.values)
    ]
    assert defaults == []  # no `x or 14` fallback in the engine (CF-13a-15)


async def test_stop_clears_anchors_and_lot_state() -> None:
    fixed = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    engine = await _engine(_bt_runner({}, bracket=fixed))
    engine._current_history = {SYM: _wavy_bars(60)}
    oid = uuid4()
    _apply(engine, _order(oid, created_at=LOT), _fill(oid, price="100", at=LOT))
    assert engine._bracket_exit.get_anchor(SYM) is not None  # type: ignore[union-attr]
    await engine.stop()
    assert engine._bracket_exit.get_anchor(SYM) is None  # type: ignore[union-attr]
    assert engine._lots == {} and engine._order_signal_bar_ts == {}


# ---------------------------------------------------------------------------
# T-31  AST: the backtest anchor path reads no clock
# ---------------------------------------------------------------------------

_CLOCK_ATTRS = {"now", "utcnow", "today", "time", "monotonic", "perf_counter", "time_ns"}


def _clock_calls(fn: Any) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _CLOCK_ATTRS:
                found.append(ast.unparse(node.func))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _CLOCK_ATTRS:
                found.append(node.func.id)
    return found


@pytest.mark.parametrize(
    "name",
    ["_anchor_for_entry", "_backtest_signal_bar_ts", "_record_signal_bar", "_finalize_one_anchor"],
)
def test_t31_backtest_anchor_path_has_no_wall_clock(name: str) -> None:
    fn = getattr(se.StrategyEngine, name)  # (lazy: this file imports at the pre-1.3b commit)
    assert _clock_calls(fn) == []
    assert RunMode.BACKTEST  # the backtest branch is the one this pins


async def test_t31b_signal_bar_is_recorded_in_memory_only() -> None:
    """Behaviour (CR-07): a BUY order records the open time of the bar being processed
    (``history[-1]``) in memory; a SELL, and a symbol without history, record nothing."""
    fixed = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
    engine = await _engine(_bt_runner({}, bracket=fixed))
    bars = _wavy_bars(60)
    buy, sell = uuid4(), uuid4()
    orders = [
        _order(buy, created_at=LOT),
        _order(sell, created_at=LOT, side=OrderSide.SELL),
    ]
    engine._record_signal_bar(orders, SYM, {SYM: bars})
    assert engine._order_signal_bar_ts == {buy: bars[-1].timestamp}
    engine._order_signal_bar_ts.clear()
    engine._record_signal_bar(orders, SYM, {})
    assert engine._order_signal_bar_ts == {}
