"""
tests/unit/test_wp22_lot_close.py
---------------------------------
WP2.2 P2a (SY-22-07, WP22-T-11): ``close_lot`` on a full-flatten SELL.

    AT-10        lot HWM 110, then a same-bar SELL + BUY at 106: the new lot's
                 peak is 106 (stop 102.82) and the next bar at 105 does not exit
    equivalence  ``close_lot`` leaves the state ``check()`` leaves on a flat
                 position, for both managers
    stale anchor a skipped open hook can no longer reuse the closed lot's anchor
    wiring       only a FULL-flatten SELL reaches ``close_lot`` (engine hook)

A same-bar collision needs bar-time stamps (P2b).  Until then the key collision
is forced by passing the SAME ``opened_at`` / ``lot_key`` to both lots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from common.types import OrderSide, TimeFrame
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading.backtest import BacktestRunner
from trading.bracket_exit import BracketExitManager
from trading.models import Fill, Position
from trading.trailing_stop import TrailingStopManager

SYM = "BTC/USD"
OTHER = "ETH/USD"
T0 = datetime(2026, 3, 1, tzinfo=UTC)
COLLIDING_KEY = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)  # same value for lot A and lot B
OPENING_ORDER = UUID("00000000-0000-0000-0000-0000000000a1")


def _pos(
    qty: str = "1", entry: str = "100", *, symbol: str = SYM, opened_at: datetime = COLLIDING_KEY
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


def _flat(symbol: str = SYM) -> Position:
    return _pos("0", symbol=symbol)


def _ts(i: int) -> datetime:
    return T0 + timedelta(hours=i)


def _open_trailing(
    mgr: TrailingStopManager, hwm0: str, *, symbol: str = SYM, anchor: datetime | None = None
) -> None:
    mgr.open_lot(
        symbol,
        lot_key=COLLIDING_KEY,
        hwm0=Decimal(hwm0),
        anchor_bar_ts=anchor if anchor is not None else _ts(0),
    )


def _open_bracket(
    mgr: BracketExitManager, entry: str, *, symbol: str = SYM, source: Any = "entry"
) -> None:
    mgr.open_lot(
        symbol,
        lot_key=COLLIDING_KEY,
        opening_order_id=OPENING_ORDER,
        anchor_at=_ts(0),
        anchor_bar_ts=_ts(0),
        entry_basis=Decimal(entry),
        basis_qty=Decimal(1),
        atr=None,
        source=source,
        final=True,
    )


def _trailing_state(mgr: TrailingStopManager, symbol: str) -> tuple[object, ...]:
    return (
        mgr._peak_prices.get(symbol),
        mgr._lots.get(symbol),
        symbol in mgr._pending_stop_symbols,
        mgr._pending_qty.get(symbol),
        mgr._pending_age.get(symbol),
    )


def _bracket_state(mgr: BracketExitManager, symbol: str) -> tuple[object, ...]:
    return (
        mgr._anchors.get(symbol),
        symbol in mgr._pending_exit_symbols,
        mgr._pending_qty.get(symbol),
        mgr._pending_age.get(symbol),
        symbol in mgr._atr_warned_symbols,
        symbol in mgr._late_warned,
    )


# ---------------------------------------------------------------------------
# AT-10
# ---------------------------------------------------------------------------


def _ratchet_to_110(mgr: TrailingStopManager) -> None:
    for i, px in enumerate(["104", "108", "110"], start=1):
        assert mgr.check(SYM, Decimal(px), _pos(), bar_ts=_ts(i)) is None
    assert mgr.peak_prices[SYM] == Decimal("110")


def test_at10_same_bar_sell_buy_new_lot_peak_is_106_stop_102_82() -> None:
    mgr = TrailingStopManager(0.03)
    _open_trailing(mgr, "100")
    _ratchet_to_110(mgr)

    # same bar (4): the SELL flattens lot A (hook -> close_lot), the BUY opens
    # lot B with an EQUAL lot_key and gross VWAP 106
    mgr.close_lot(SYM)
    _open_trailing(mgr, "106", anchor=_ts(4))

    assert mgr.peak_prices[SYM] == Decimal("106")
    assert mgr.protective_level(SYM, _pos(entry="106")) == Decimal("102.82")
    # next bar at 105: above 102.82, so no exit (HEAD kept 110 -> stop 106.70 -> exit)
    assert mgr.check(SYM, Decimal("105"), _pos(entry="106"), bar_ts=_ts(5)) is None
    assert mgr.peak_prices[SYM] == Decimal("106")


def test_at10_without_close_lot_the_collision_keeps_the_old_mark() -> None:
    """Control: documents the defect P2a fixes (equal key = 'further fill')."""
    mgr = TrailingStopManager(0.03)
    _open_trailing(mgr, "100")
    _ratchet_to_110(mgr)
    _open_trailing(mgr, "106", anchor=_ts(4))  # no close_lot
    assert mgr.peak_prices[SYM] == Decimal("110")
    assert mgr.protective_level(SYM, _pos(entry="106")) == Decimal("106.70")
    assert mgr.check(SYM, Decimal("105"), _pos(entry="106"), bar_ts=_ts(5)) is not None


# ---------------------------------------------------------------------------
# equivalence with the check() flat branch
# ---------------------------------------------------------------------------


def _trailing_with_pending(symbol: str = SYM) -> TrailingStopManager:
    mgr = TrailingStopManager(0.03)
    _open_trailing(mgr, "100", symbol=symbol)
    assert mgr.check(symbol, Decimal("110"), _pos(symbol=symbol), bar_ts=_ts(1)) is None
    # trigger -> pending guard, qty and age populated
    assert mgr.check(symbol, Decimal("100"), _pos(symbol=symbol), bar_ts=_ts(2)) is not None
    assert symbol in mgr._pending_stop_symbols
    return mgr


def test_trailing_close_lot_equals_check_on_flat_position() -> None:
    via_check = _trailing_with_pending()
    via_close = _trailing_with_pending()
    assert (
        _trailing_state(via_check, SYM)
        == _trailing_state(via_close, SYM)
        != (
            None,
            None,
            False,
            None,
            None,
        )
    )

    assert via_check.check(SYM, Decimal("100"), _flat(), bar_ts=_ts(3)) is None
    via_close.close_lot(SYM)

    assert _trailing_state(via_close, SYM) == _trailing_state(via_check, SYM)
    assert _trailing_state(via_close, SYM) == (None, None, False, None, None)


def test_trailing_close_lot_is_per_symbol_and_idempotent() -> None:
    mgr = _trailing_with_pending()
    _open_trailing(mgr, "50", symbol=OTHER)
    before_other = _trailing_state(mgr, OTHER)
    mgr.close_lot(SYM)
    mgr.close_lot(SYM)
    assert _trailing_state(mgr, SYM) == (None, None, False, None, None)
    assert _trailing_state(mgr, OTHER) == before_other
    mgr.close_lot("NEVER/SEEN")  # unknown symbol: no error


def _bracket_with_state(symbol: str = SYM) -> BracketExitManager:
    mgr = BracketExitManager(stop_loss_pct=0.02, take_profit_pct=0.04)
    _open_bracket(mgr, "100", symbol=symbol)
    # stop breach -> pending guard populated
    assert mgr.check(symbol, Decimal("97"), _pos(symbol=symbol)) is not None
    assert symbol in mgr._pending_exit_symbols
    mgr._atr_warned_symbols.add(symbol)
    mgr._late_warned.add(symbol)
    return mgr


def test_bracket_close_lot_equals_check_on_flat_position() -> None:
    via_check = _bracket_with_state()
    via_close = _bracket_with_state()
    assert _bracket_state(via_check, SYM) == _bracket_state(via_close, SYM)
    assert _bracket_state(via_close, SYM)[0] is not None

    assert via_check.check(SYM, Decimal("97"), _flat()) is None
    via_close.close_lot(SYM)

    assert _bracket_state(via_close, SYM) == _bracket_state(via_check, SYM)
    assert _bracket_state(via_close, SYM) == (None, False, None, None, False, False)


def test_bracket_close_lot_is_per_symbol_and_idempotent() -> None:
    mgr = _bracket_with_state()
    _open_bracket(mgr, "50", symbol=OTHER)
    before_other = _bracket_state(mgr, OTHER)
    mgr.close_lot(SYM)
    mgr.close_lot(SYM)
    assert _bracket_state(mgr, SYM) == (None, False, None, None, False, False)
    assert _bracket_state(mgr, OTHER) == before_other
    mgr.close_lot("NEVER/SEEN")


# ---------------------------------------------------------------------------
# stale bracket anchor (amendment to AD-09)
# ---------------------------------------------------------------------------


def test_skipped_open_hook_cannot_reuse_the_closed_lots_anchor() -> None:
    # lot A: entry 100, stop 98 (2%), frozen at open
    mgr = BracketExitManager(stop_loss_pct=0.02)
    _open_bracket(mgr, "100")
    assert mgr.protective_level(SYM, _pos(entry="100")) == Decimal("98.00")

    # lot B reopens with an EQUAL key at 200 and its open hook is skipped.
    lot_b = _pos(entry="200")
    stale = BracketExitManager(stop_loss_pct=0.02)
    _open_bracket(stale, "100")
    # control (no close_lot): A's anchor is accepted for B (stop 98, far below 200)
    assert stale.protective_level(SYM, lot_b) == Decimal("98.00")
    assert stale.check(SYM, Decimal("150"), lot_b) is None  # B unprotected at 150

    # with close_lot on flatten: the anchor is gone, B gets its own late anchor
    mgr.close_lot(SYM)
    assert mgr.get_anchor(SYM) is None
    assert mgr.check(SYM, Decimal("250"), lot_b) is None  # creates the late anchor
    anchor = mgr.get_anchor(SYM)
    assert anchor is not None
    assert anchor.source == "late"
    assert anchor.entry_basis == Decimal("200")
    assert mgr.protective_level(SYM, lot_b) == Decimal("196.00")


# ---------------------------------------------------------------------------
# engine wiring: only the full-flatten SELL reaches close_lot
# ---------------------------------------------------------------------------


def _engine() -> Any:
    runner = BacktestRunner(
        strategies=[ScriptedSequenceStrategy("p2a", {"script": {}, "target_notional": "1000"})],
        symbols=[SYM],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital=Decimal("10000"),
        seed=7,
        trailing_stop_pct=0.03,
        bracket_config={
            "bracket_mode": "fixed",
            "bracket_stop_loss_pct": 0.02,
            "bracket_take_profit_pct": 0.04,
        },
    )
    engine, *_ = runner._build_engine("p2a")
    assert engine._trailing_stop is not None
    assert engine._bracket_exit is not None
    return engine


def _seed_both(engine: Any) -> None:
    _open_trailing(engine._trailing_stop, "100")
    _open_bracket(engine._bracket_exit, "100")


def _sell(qty: str) -> Fill:
    return Fill(
        order_id=uuid4(),
        symbol=SYM,
        side=OrderSide.SELL,
        quantity=Decimal(qty),
        price=Decimal("105"),
        fee=Decimal(0),
        fee_currency="USD",
    )


def test_full_flatten_sell_closes_both_lots() -> None:
    engine = _engine()
    _seed_both(engine)
    engine._on_fill_routed(None, _sell("1"), _pos(qty="1"))
    assert _trailing_state(engine._trailing_stop, SYM) == (None, None, False, None, None)
    assert _bracket_state(engine._bracket_exit, SYM) == (None, False, None, None, False, False)


def test_unknown_pre_fill_position_counts_as_full_flatten() -> None:
    engine = _engine()
    _seed_both(engine)
    engine._on_fill_routed(None, _sell("1"), None)
    assert SYM not in engine._trailing_stop._lots
    assert engine._bracket_exit.get_anchor(SYM) is None


def test_partial_sell_keeps_both_lots() -> None:
    engine = _engine()
    _seed_both(engine)
    engine._on_fill_routed(None, _sell("0.4"), _pos(qty="1"))
    assert engine._trailing_stop.peak_prices[SYM] == Decimal("100")
    assert SYM in engine._trailing_stop._lots
    assert engine._bracket_exit.get_anchor(SYM) is not None
