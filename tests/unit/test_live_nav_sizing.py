"""
tests/unit/test_live_nav_sizing.py
------------------------------------
WP1.4 (Verbeterplan v2 §4 row 1.4, D5) -- unit tests for live NAV / sizing
basis: run equity comes exclusively from the attached ``PortfolioAccounting``
(never the exchange balance), the sizing basis is
``max(0, min(NAV, initial_capital))``, and BUYs are capped by run cash and a
fresh exchange free-quote balance.

Module under test
------------------
    packages/trading/execution.py -- ``EquitySnapshot``, ``_sizing_basis``,
    ``_cap_buy_quantity`` (``BaseExecutionEngine``)
    packages/trading/engines/live.py -- ``LiveExecutionEngine._equity_snapshot``,
    ``_quote_asset``, ``_taker_buffer``, ``_inflight_buy_orders``,
    ``process_signal``, ``on_start``

Test IDs mirror ``reports/vp2-wp1.4/synthesis-spec.md``'s mandatory table
(U-01..U-14) so a reviewer can cross-reference directly; each test's
docstring repeats its ID and the invariant (I-1..I-8) it proves.

Design notes
------------
- Mirrors ``tests/unit/test_live_position_ledger.py``'s style: a real
  ``PortfolioAccounting`` (there is no reason to mock the single source of
  truth) + a real or mocked ``BaseRiskManager`` + a ``MagicMock`` CCXT
  exchange. U-01 (per the spec) uses the REAL ``DefaultRiskManager``;
  every other scenario uses a lenient mock risk manager so the test
  isolates the D5 NAV/sizing/affordability logic from the (separately
  tested) risk-gating logic itself.
- Positions are injected directly into ``portfolio._position_snapshots``
  (and ``portfolio._cash``/``_peak_equity`` where a specific starting
  state is needed), exactly as ``test_live_position_ledger.py`` does --
  these are unit tests of the engine+portfolio contract, not of fill
  processing itself.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.engines.live import LiveExecutionEngine
from trading.execution import EquitySnapshot
from trading.models import Fill, Order, Position, RiskCheckResult, Signal
from trading.portfolio import PortfolioAccounting
from trading.risk_manager import DefaultRiskManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_ID = "nav-sizing-test-run"
_QUOTE = "EUR"

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _market(base: str, quote: str = _QUOTE, *, taker: Decimal | None = None) -> dict[str, Any]:
    market: dict[str, Any] = {
        "base": base,
        "quote": quote,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
    }
    if taker is not None:
        market["taker"] = float(taker)
    return market


def _make_mock_exchange(
    *,
    markets: dict[str, Any],
    fetch_balance_response: dict[str, Any] | None = None,
    fetch_ticker_response: dict[str, Any] | None = None,
) -> MagicMock:
    exchange = MagicMock()
    exchange.id = "mock-exchange"
    exchange.markets = markets
    exchange.load_markets = AsyncMock(return_value=markets)
    exchange.fetch_balance = AsyncMock(
        return_value=fetch_balance_response
        or {"total": {_QUOTE: 100000.0}, "free": {_QUOTE: 100000.0}}
    )
    exchange.fetch_ticker = AsyncMock(
        return_value=fetch_ticker_response or {"last": "100"}
    )
    exchange.create_order = AsyncMock(
        return_value={
            "id": "exch-001", "status": "open", "filled": "0",
            "average": None, "price": "100",
        }
    )
    exchange.fetch_order = AsyncMock(
        return_value={
            "id": "exch-001", "status": "closed", "filled": None,
            "average": None, "price": "100",
        }
    )
    exchange.cancel_order = AsyncMock(return_value={"id": "exch-001", "status": "canceled"})
    exchange.close = AsyncMock(return_value=None)
    exchange.has = {"fetchOrderTrades": False, "fetchMyTrades": True}
    exchange.fetch_order_trades = AsyncMock(return_value=[])
    exchange.fetch_my_trades = AsyncMock(return_value=[])
    return exchange


def _make_risk_manager_mock(
    *, approved: bool = True, position_size: Decimal = Decimal("999999")
) -> MagicMock:
    """A lenient risk manager: ``calculate_position_size`` never binds
    (unless the caller shrinks ``position_size``) and ``pre_trade_check``
    passes ``order.quantity`` straight through as ``adjusted_quantity`` --
    isolates the D5 NAV/sizing/affordability logic under test from the
    separately-tested risk-gating logic itself."""
    mock = MagicMock()

    def _pre_trade_check(*, order: Order, **_: Any) -> RiskCheckResult:
        return RiskCheckResult(
            approved=approved,
            adjusted_quantity=order.quantity if approved else Decimal("0"),
            rejection_reasons=[] if approved else ["test rejection"],
            warnings=[],
        )

    mock.pre_trade_check.side_effect = _pre_trade_check
    mock.calculate_position_size.return_value = position_size
    return mock


def _make_engine(
    *,
    initial_cash: Decimal = Decimal("1000"),
    symbols: list[str],
    markets: dict[str, Any],
    risk_manager: Any = None,
    fetch_balance_response: dict[str, Any] | None = None,
    fetch_ticker_response: dict[str, Any] | None = None,
) -> tuple[LiveExecutionEngine, PortfolioAccounting, MagicMock]:
    """Build a real ``LiveExecutionEngine`` + a real ``PortfolioAccounting``
    attached as its ``LivePositionSource``, exactly as ``StrategyEngine``
    does -- mirrors ``test_live_position_ledger.py``'s
    ``_make_engine_with_source``."""
    rm = risk_manager if risk_manager is not None else _make_risk_manager_mock()
    ex = _make_mock_exchange(
        markets=markets,
        fetch_balance_response=fetch_balance_response,
        fetch_ticker_response=fetch_ticker_response,
    )
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=initial_cash)
    engine.attach_position_source(portfolio, symbols=symbols)
    return engine, portfolio, ex


def _make_signal(
    *, symbol: str, direction: SignalDirection, target: Decimal, confidence: float = 1.0
) -> Signal:
    return Signal(
        strategy_id="nav-sizing-test-strategy",
        symbol=symbol,
        direction=direction,
        target_position=target,
        confidence=confidence,
    )


def _open_position(symbol: str, quantity: Decimal, price: Decimal) -> Position:
    return Position(
        symbol=symbol, run_id=_RUN_ID, quantity=quantity,
        average_entry_price=price, current_price=price,
    )


def _make_order(
    *, symbol: str, side: OrderSide, quantity: Decimal = Decimal("1")
) -> Order:
    return Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=symbol, side=side,
        order_type=OrderType.MARKET, quantity=quantity,
    )


# ===========================================================================
# U-01: two 15% positions, flat price -> drawdown ~ 0, BUY approved by the
# REAL DefaultRiskManager (per the spec, this is the one case that must use
# the real risk manager, not a mock).
# ===========================================================================


@pytest.mark.asyncio
async def test_u01_two_15pct_positions_flat_price_zero_drawdown_buy_approved() -> None:
    """U-01: two 15% positions at a flat price -> drawdown ~ 0, and a BUY
    on a third symbol is approved by the real DefaultRiskManager."""
    symbols = ["BTC/EUR", "ETH/EUR", "SOL/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    risk_manager = DefaultRiskManager(run_id=_RUN_ID)
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        risk_manager=risk_manager,
        fetch_ticker_response={"last": "100"},
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(
        "BTC/EUR", Decimal("0.003"), Decimal("50000")
    )  # 150 EUR, 15%
    portfolio._position_snapshots["ETH/EUR"] = _open_position(
        "ETH/EUR", Decimal("0.05"), Decimal("3000")
    )  # 150 EUR, 15%
    portfolio._cash = Decimal("700")  # 1000 - 150 - 150

    assert portfolio.current_equity == Decimal("1000")
    assert portfolio.get_peak_equity() == Decimal("1000")

    orders = await engine.process_signal(
        _make_signal(symbol="SOL/EUR", direction=SignalDirection.BUY, target=Decimal("50"))
    )

    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY


# ===========================================================================
# U-02: external base and quote balances are excluded from NAV (I-1).
# ===========================================================================


def test_u02_external_base_and_quote_excluded_from_nav() -> None:
    """U-02: NAV never reads the exchange balance -- large external
    EUR/BTC holdings on the account are excluded entirely (I-1)."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, _portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={
            "total": {"EUR": 999999.0, "BTC": 50.0},
            "free": {"EUR": 999999.0, "BTC": 50.0},
        },
    )

    snapshot = engine._equity_snapshot("BTC/EUR", Decimal("100"))

    assert snapshot is not None
    assert snapshot.nav == Decimal("1000")
    ex.fetch_balance.assert_not_called()


# ===========================================================================
# U-03: basis is 1000 at NAV 1200 and 800 at NAV 800; at NAV 120 / peak 120
# the gate's drawdown is 0 even though the sizing basis is capped (I-2/I-3).
# ===========================================================================


def test_u03_profit_caps_basis_but_gate_sees_full_nav_zero_drawdown() -> None:
    """U-03: basis is 1000 at NAV 1200, 800 at NAV 800; at NAV 120 / peak
    120 the drawdown gate sees 0%, even though the sizing basis is capped
    to the smaller initial_capital (100)."""
    assert LiveExecutionEngine._sizing_basis(Decimal("1200"), Decimal("1000")) == Decimal("1000")
    assert LiveExecutionEngine._sizing_basis(Decimal("800"), Decimal("1000")) == Decimal("800")

    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ = _make_engine(
        initial_cash=Decimal("100"), symbols=symbols, markets=markets,
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(
        "BTC/EUR", Decimal("1"), Decimal("20")
    )
    portfolio._cash = Decimal("100")
    portfolio._peak_equity = Decimal("120")

    snapshot = engine._equity_snapshot("BTC/EUR", Decimal("20"))

    assert snapshot is not None
    assert snapshot.nav == Decimal("120")
    assert snapshot.peak == Decimal("120")
    assert snapshot.sizing_basis == Decimal("100")
    drawdown = (snapshot.peak - snapshot.nav) / snapshot.peak
    assert drawdown == Decimal("0")


# ===========================================================================
# U-04: BUY affordability caps from run cash and from a lower free-quote
# balance (with a warning logged) (I-4).
# ===========================================================================


def test_u04_cap_buy_quantity_helper_table() -> None:
    """U-04: _cap_buy_quantity caps from run cash and from free quote."""
    # available=80 binds: max qty = 80 / (10 * 1) = 8
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("100"), Decimal("10"), Decimal("80"), Decimal("0")
    ) == Decimal("8")
    # already under the cap: untouched
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("1"), Decimal("10"), Decimal("80"), Decimal("0")
    ) == Decimal("1")


@pytest.mark.asyncio
async def test_u04_cap_from_free_quote_below_run_cash_warns() -> None:
    """U-04: run cash 80, free quote 30 -> BUY capped to
    30 / (price * (1 + buf)), with live.run_cash_exceeds_exchange_free
    logged as a warning."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={"total": {"EUR": 30.0}, "free": {"EUR": 30.0}},
        fetch_ticker_response={"last": "10"},
    )
    portfolio._cash = Decimal("80")

    with capture_logs() as cap:
        orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("1000"))
        )

    assert len(orders) == 1
    notional = orders[0].quantity * Decimal("10")
    buf = engine._taker_buffer("BTC/EUR") + engine._buy_cap_slippage_pct  # 0.01 + 0.005
    assert notional * (Decimal("1") + buf) <= Decimal("30") + Decimal("0.01")
    assert any(e.get("event") == "live.run_cash_exceeds_exchange_free" for e in cap)


# ===========================================================================
# Security round 2 (reports/vp2-wp1.4/security-report.md)
# ===========================================================================


@pytest.mark.asyncio
async def test_s01_coinbase_submit_reuses_recorded_price_not_second_ticker() -> None:
    """WP14-S-01: submit_order's Coinbase market-BUY path reuses the SAME
    price process_signal's affordability cap sized against -- a rising
    ticker between the two fetches must not let the order spend more."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={"total": {"EUR": 1000.0}, "free": {"EUR": 1000.0}},
    )
    ex.id = "coinbase"
    # If submit_order fetched a SECOND ticker (the bug), it would see 999
    # instead of the 100 process_signal's cap actually used.
    ex.fetch_ticker = AsyncMock(side_effect=[{"last": "100"}, {"last": "999"}])
    portfolio._cash = Decimal("1000")

    orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("2000"))
    )

    assert len(orders) == 1
    assert ex.fetch_ticker.call_count == 1, "submit_order must not fetch a second ticker"
    call = ex.create_order.call_args
    submitted_price = Decimal(call.args[4])
    submitted_amount = Decimal(call.args[3])
    assert submitted_price == Decimal("100")
    buf = engine._taker_buffer("BTC/EUR") + engine._buy_cap_slippage_pct
    notional = submitted_amount * submitted_price * (Decimal("1") + buf)
    assert notional <= Decimal("1000") + Decimal("0.01")


@pytest.mark.asyncio
async def test_s10_buy_sizing_price_hint_never_leaks_on_non_coinbase_exchange() -> None:
    """WP14-S-10: _buy_sizing_price must not leak on a non-Coinbase
    exchange. submit_order now pops the hint unconditionally, before the
    Coinbase-specific branch, so every BUY -- on any exchange -- removes
    its own entry regardless of whether that branch ever runs."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={"total": {"EUR": 1000.0}, "free": {"EUR": 1000.0}},
    )
    assert ex.id != "coinbase"  # the mock's default id
    portfolio._cash = Decimal("1000")

    for _ in range(5):
        orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("10"))
        )
        assert len(orders) == 1

    assert engine._buy_sizing_price == {}


@pytest.mark.asyncio
async def test_s02_cap_prices_at_ask_when_higher_than_last() -> None:
    """WP14-S-02: the affordability cap prices at max(last, ask) -- when
    the book's ask is above last, the cap uses ask (the tighter, safer
    bound), not last alone."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        risk_manager=_make_risk_manager_mock(position_size=Decimal("999999")),
        fetch_balance_response={"total": {"EUR": 1000.0}, "free": {"EUR": 1000.0}},
        fetch_ticker_response={"last": "100", "ask": "110"},
    )
    portfolio._cash = Decimal("1000")

    orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100000"))
    )

    assert len(orders) == 1
    buf = engine._taker_buffer("BTC/EUR") + engine._buy_cap_slippage_pct
    max_qty_at_ask = Decimal("1000") / (Decimal("110") * (Decimal("1") + buf))
    assert orders[0].quantity <= max_qty_at_ask + Decimal("0.00000001")
    # Sanity: capping at plain `last` (100) would have allowed a
    # meaningfully larger quantity -- prove ask (110) is what actually bound it.
    max_qty_at_last = Decimal("1000") / (Decimal("100") * (Decimal("1") + buf))
    assert orders[0].quantity < max_qty_at_last


def test_s03_taker_buffer_rejects_negative_and_non_finite() -> None:
    """WP14-S-03: a negative or non-finite (NaN/Inf) taker fee from the
    exchange falls back to the 1% default."""
    markets: dict[str, Any] = {
        "A/EUR": _market("A", taker=Decimal("-0.5")),
        "B/EUR": _market("B"),
    }
    markets["B/EUR"]["taker"] = float("nan")
    engine, _, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=["A/EUR", "B/EUR"], markets=markets,
    )

    assert engine._taker_buffer("A/EUR") == Decimal("0.01")
    assert engine._taker_buffer("B/EUR") == Decimal("0.01")

    markets["A/EUR"]["taker"] = float("inf")
    assert engine._taker_buffer("A/EUR") == Decimal("0.01")


@pytest.mark.asyncio
async def test_s03_non_finite_free_quote_blocks_buy() -> None:
    """WP14-S-03: a NaN/Inf free-quote balance is treated exactly like a
    missing one -- blocked, never a raised InvalidOperation comparison."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={"total": {"EUR": "NaN"}, "free": {"EUR": "NaN"}},
    )
    portfolio._cash = Decimal("1000")

    with capture_logs() as cap:
        orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert orders == []
    assert any(e.get("event") == "live.buy_blocked_balance_unavailable" for e in cap)


# ===========================================================================
# U-05: a balance fault, an unparseable value, or a missing quote key all
# block the BUY; SELL is always unaffected (I-4/I-5).
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fetch_balance_response",
    [
        pytest.param(None, id="fetch_raises"),
        pytest.param({"total": {"EUR": "oops"}, "free": {"EUR": "oops"}}, id="unparseable"),
        pytest.param({"total": {}, "free": {}}, id="missing_quote_key"),
    ],
)
async def test_u05_balance_fault_blocks_buy(
    fetch_balance_response: dict[str, Any] | None,
) -> None:
    """U-05: a balance fetch failure, an unparseable value, or a missing
    quote key all block the BUY with live.buy_blocked_balance_unavailable."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, _portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response=fetch_balance_response or {"total": {}, "free": {}},
    )
    if fetch_balance_response is None:
        ex.fetch_balance.side_effect = Exception("exchange unreachable")

    with capture_logs() as cap:
        orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )

    assert orders == []
    assert any(e.get("event") == "live.buy_blocked_balance_unavailable" for e in cap)


@pytest.mark.asyncio
async def test_u05_balance_fault_does_not_block_sell() -> None:
    """U-05: the same balance fault that blocks a BUY never blocks a SELL."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(
        "BTC/EUR", Decimal("1"), Decimal("100")
    )
    ex.fetch_balance.side_effect = Exception("exchange unreachable")

    orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.SELL, target=Decimal("0"))
    )

    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL


# ===========================================================================
# U-06: a held non-signal position marked <= 0 makes NAV unavailable; BUY
# is blocked, SELL is unaffected (R-08).
# ===========================================================================


@pytest.mark.asyncio
async def test_u06_held_non_signal_position_marked_zero_blocks_buy() -> None:
    """U-06: a held non-signal position marked <= 0 makes NAV unavailable
    -- BUY is blocked with live.nav_unavailable; SELL is unaffected."""
    symbols = ["BTC/EUR", "ETH/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    engine, portfolio, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(
        "BTC/EUR", Decimal("0.01"), Decimal("0")
    )
    portfolio._cash = Decimal("1000")

    with capture_logs() as cap:
        buy_orders = await engine.process_signal(
            _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert buy_orders == []
    assert any(e.get("event") == "live.nav_unavailable" for e in cap)

    sell_orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.SELL, target=Decimal("0"))
    )
    assert len(sell_orders) == 1


# ===========================================================================
# U-07: an unrouted BUY blocks a BUY on another symbol (run-wide); SELL is
# unaffected; a REJECTED order never blocks (S-02/A-04).
# ===========================================================================


@pytest.mark.asyncio
async def test_u07_inflight_buy_blocks_other_symbol_sell_unaffected() -> None:
    """U-07: an unrouted (OPEN) BUY blocks a BUY on a different symbol
    run-wide; a SELL is unaffected."""
    symbols = ["BTC/EUR", "ETH/EUR", "SOL/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._cash = Decimal("1000")

    stuck = await engine.submit_order(_make_order(symbol="BTC/EUR", side=OrderSide.BUY))
    assert stuck.status == OrderStatus.OPEN

    with capture_logs() as cap:
        buy_orders = await engine.process_signal(
            _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert buy_orders == []
    assert any(e.get("event") == "live.buy_blocked_inflight_buy" for e in cap)

    portfolio._position_snapshots["SOL/EUR"] = _open_position(
        "SOL/EUR", Decimal("1"), Decimal("10")
    )
    sell_orders = await engine.process_signal(
        _make_signal(symbol="SOL/EUR", direction=SignalDirection.SELL, target=Decimal("0"))
    )
    assert len(sell_orders) == 1


@pytest.mark.asyncio
async def test_u07_rejected_order_does_not_block_buy() -> None:
    """U-07: a REJECTED order is neither in-flight nor unrouted-settled --
    it never blocks a subsequent BUY."""
    symbols = ["BTC/EUR", "ETH/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    engine, portfolio, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._cash = Decimal("1000")
    rejected = _make_order(symbol="BTC/EUR", side=OrderSide.BUY)
    engine._orders[rejected.order_id] = rejected.model_copy(
        update={"status": OrderStatus.REJECTED}
    )

    assert engine._inflight_buy_orders() is None

    orders = await engine.process_signal(
        _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
    )
    assert len(orders) == 1


# ===========================================================================
# Security round 2 (reports/vp2-wp1.4/security-report.md)
# ===========================================================================


@pytest.mark.asyncio
async def test_s04_stale_inflight_block_alert_fires_once() -> None:
    """WP14-S-04: once a blocking BUY has been stuck for longer than
    _buy_inflight_stale_after_s (default 15 min), live.buy_inflight_block_stale
    logs at error level exactly once -- not on every subsequent blocked
    attempt, and not before the threshold."""
    symbols = ["BTC/EUR", "ETH/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    engine, portfolio, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._cash = Decimal("1000")
    stuck = await engine.submit_order(_make_order(symbol="BTC/EUR", side=OrderSide.BUY))
    assert stuck.status == OrderStatus.OPEN

    # First blocked attempt: too soon for the stale alert.
    with capture_logs() as cap1:
        await engine.process_signal(
            _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert not any(e.get("event") == "live.buy_inflight_block_stale" for e in cap1)

    # Simulate 16 minutes having passed since the block was first observed.
    engine._inflight_block_started_at[stuck.order_id] = (
        datetime.now(tz=UTC) - timedelta(seconds=960)
    )

    with capture_logs() as cap2:
        await engine.process_signal(
            _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    stale_events = [e for e in cap2 if e.get("event") == "live.buy_inflight_block_stale"]
    assert len(stale_events) == 1
    assert stale_events[0].get("log_level") == "error"
    assert stale_events[0].get("order_id") == str(stuck.order_id)

    # A further blocked attempt does not re-log the stale alert.
    with capture_logs() as cap3:
        await engine.process_signal(
            _make_signal(symbol="ETH/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert not any(e.get("event") == "live.buy_inflight_block_stale" for e in cap3)


@pytest.mark.asyncio
async def test_s05_pending_submit_no_exchange_id_flags_reconcile_sell_still_passes() -> None:
    """WP14-S-05 (Probe PF): a PENDING_SUBMIT BUY with no exchange id
    (create_order raised before the exchange ever acknowledged it) keeps
    blocking new BUYs AND flags reconcile_required[symbol] =
    'buy_order_state_unknown'; a SELL still goes out unaffected."""
    symbols = ["BTC/EUR", "ETH/EUR"]
    markets = {s: _market(s.split("/")[0]) for s in symbols}
    engine, portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._cash = Decimal("1000")
    portfolio._position_snapshots["ETH/EUR"] = _open_position(
        "ETH/EUR", Decimal("1"), Decimal("10")
    )

    ex.create_order.side_effect = ValueError("exchange rejected the order shape")
    with pytest.raises(ValueError):
        await engine.submit_order(_make_order(symbol="BTC/EUR", side=OrderSide.BUY))

    stuck = [o for o in engine._orders.values() if o.status == OrderStatus.PENDING_SUBMIT]
    assert len(stuck) == 1
    assert stuck[0].order_id not in engine._exchange_order_map

    with capture_logs() as cap:
        buy_orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
        )
    assert buy_orders == []
    assert any(e.get("event") == "live.buy_blocked_inflight_buy" for e in cap)
    assert engine.reconcile_required.get("BTC/EUR") == "buy_order_state_unknown"

    ex.create_order.side_effect = None  # the SELL's own submit reaches the exchange normally
    sell_orders = await engine.process_signal(
        _make_signal(symbol="ETH/EUR", direction=SignalDirection.SELL, target=Decimal("0"))
    )
    assert len(sell_orders) == 1


# ===========================================================================
# U-08: from_fills(peak_equity_hint=1500) with NAV back at 1000 trips the
# drawdown gate for a BUY; a protective SELL still passes (I-3).
# ===========================================================================


@pytest.mark.asyncio
async def test_u08_resumed_peak_hint_rejects_buy_sell_still_passes() -> None:
    """U-08: from_fills(peak_equity_hint=1500) with NAV 1000 -> BUY
    rejected on drawdown; SELL passes."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    # 8 BTC @ 100 EUR leaves 200 EUR of run cash (cash=0 would make the
    # BUY affordability cap itself zero the quantity before it ever
    # reaches the drawdown gate this test is about).
    fill = Fill(
        order_id=uuid4(), symbol="BTC/EUR", side=OrderSide.BUY,
        quantity=Decimal("8"), price=Decimal("100"), fee=Decimal("0"),
        fee_currency="EUR",
    )
    portfolio = PortfolioAccounting.from_fills(
        run_id=_RUN_ID, initial_cash=Decimal("1000"), fills=[fill],
        peak_equity_hint=Decimal("1500"),
    )
    assert portfolio.current_equity == Decimal("1000")
    assert portfolio.get_peak_equity() == Decimal("1500")

    risk_manager = DefaultRiskManager(run_id=_RUN_ID)
    ex = _make_mock_exchange(markets=markets, fetch_ticker_response={"last": "100"})
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=risk_manager, exchange=ex, enable_live_trading=True,
    )
    engine.attach_position_source(portfolio, symbols=symbols)

    with capture_logs() as cap:
        buy_orders = await engine.process_signal(
            _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("50"))
        )
    assert buy_orders == []
    assert any(e.get("event") == "live.signal_rejected" for e in cap)

    sell_orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.SELL, target=Decimal("0"))
    )
    assert len(sell_orders) == 1


# ===========================================================================
# U-09: computing a snapshot never mutates the portfolio's curve or peak
# (I-6).
# ===========================================================================


def test_u09_equity_snapshot_never_mutates_portfolio() -> None:
    """U-09: computing an EquitySnapshot never mutates the portfolio's
    equity curve or peak (I-6)."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._position_snapshots["BTC/EUR"] = _open_position(
        "BTC/EUR", Decimal("1"), Decimal("50")
    )
    portfolio._cash = Decimal("950")
    curve_before = list(portfolio.get_equity_curve())
    peak_before = portfolio.get_peak_equity()

    snapshot = engine._equity_snapshot("BTC/EUR", Decimal("200"))

    assert snapshot is not None
    assert snapshot.nav == Decimal("1150")  # (950 + 50) + 1*(200-50)
    assert portfolio.get_equity_curve() == curve_before
    assert portfolio.get_peak_equity() == peak_before


# ===========================================================================
# U-10: with a source attached, on_start() seeds no peak -- spies prove
# _fetch_equity/_fetch_peak_equity are never called (S-03/A-05).
# ===========================================================================


@pytest.mark.asyncio
async def test_u10_on_start_with_source_never_calls_legacy_equity_helpers() -> None:
    """U-10: with a source attached, on_start() seeds no peak -- spies
    show _fetch_equity/_fetch_peak_equity are never called."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, _portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )

    fetch_equity_spy = AsyncMock(wraps=engine._fetch_equity)
    fetch_peak_spy = AsyncMock(wraps=engine._fetch_peak_equity)
    engine._fetch_equity = fetch_equity_spy  # type: ignore[method-assign]
    engine._fetch_peak_equity = fetch_peak_spy  # type: ignore[method-assign]

    await engine.on_start()

    fetch_equity_spy.assert_not_called()
    fetch_peak_spy.assert_not_called()
    assert engine._peak_equity == Decimal("0")


# ===========================================================================
# U-11: with no LivePositionSource attached, the legacy (D8) path is
# unchanged -- equity comes from _fetch_equity(), the exchange balance.
# ===========================================================================


@pytest.mark.asyncio
async def test_u11_no_source_legacy_path_unchanged() -> None:
    """U-11: with no LivePositionSource attached, process_signal's BUY
    path is the legacy (D8) one -- equity from _fetch_equity(), no
    quote-mismatch/in-flight/affordability checks apply."""
    markets = {"BTC/EUR": _market("BTC")}
    ex = _make_mock_exchange(
        markets=markets,
        fetch_balance_response={"total": {"EUR": 5000.0}, "free": {"EUR": 5000.0}},
        fetch_ticker_response={"last": "100"},
    )
    risk_manager = _make_risk_manager_mock(position_size=Decimal("2"))
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=risk_manager, exchange=ex, enable_live_trading=True,
    )
    # Deliberately no attach_position_source() call (D8).

    orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
    )

    assert len(orders) == 1
    ex.fetch_balance.assert_called()  # the legacy equity path hits the exchange


# ===========================================================================
# U-12: a run whose symbols don't share one quote currency blocks BUYs at
# on_start(); SELLs are unaffected (S-04/R-09).
# ===========================================================================


@pytest.mark.asyncio
async def test_u12_mixed_quote_blocks_buy_sell_unaffected() -> None:
    """U-12: mixed or missing quote currencies block BUYs run-wide at
    on_start(); SELLs are unaffected."""
    symbols = ["BTC/EUR", "ETH/USD"]
    markets = {"BTC/EUR": _market("BTC", "EUR"), "ETH/USD": _market("ETH", "USD")}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    portfolio._position_snapshots["ETH/USD"] = _open_position(
        "ETH/USD", Decimal("1"), Decimal("50")
    )

    with capture_logs() as cap:
        await engine.on_start()
    assert any(e.get("event") == "live.quote_mismatch" for e in cap)
    assert engine._run_buy_block == "quote_mismatch"

    buy_orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("100"))
    )
    assert buy_orders == []

    sell_orders = await engine.process_signal(
        _make_signal(symbol="ETH/USD", direction=SignalDirection.SELL, target=Decimal("0"))
    )
    assert len(sell_orders) == 1


# ===========================================================================
# U-13: the A-07 initial-capital warning never blocks; a balance fault
# during on_start()'s own capital check never raises.
# ===========================================================================


@pytest.mark.asyncio
async def test_u13_initial_capital_warning_never_blocks() -> None:
    """U-13: live.initial_capital_exceeds_free_quote is a warning only --
    it never sets _run_buy_block and never blocks a subsequent BUY."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, portfolio, _ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
        fetch_balance_response={"total": {"EUR": 10.0}, "free": {"EUR": 10.0}},
        fetch_ticker_response={"last": "10"},
    )

    with capture_logs() as cap:
        await engine.on_start()  # must not raise despite free << initial_capital
    assert any(e.get("event") == "live.initial_capital_exceeds_free_quote" for e in cap)
    assert engine._run_buy_block is None

    portfolio._cash = Decimal("50")
    orders = await engine.process_signal(
        _make_signal(symbol="BTC/EUR", direction=SignalDirection.BUY, target=Decimal("20"))
    )
    assert len(orders) == 1


@pytest.mark.asyncio
async def test_u13_balance_fault_during_on_start_never_raises() -> None:
    """U-13: a fetch_balance failure during on_start()'s capital-warning
    check is swallowed by _fetch_balance_cached -- on_start() never
    raises."""
    symbols = ["BTC/EUR"]
    markets = {"BTC/EUR": _market("BTC")}
    engine, _portfolio, ex = _make_engine(
        initial_cash=Decimal("1000"), symbols=symbols, markets=markets,
    )
    ex.fetch_balance.side_effect = Exception("exchange unreachable")

    await engine.on_start()  # must not raise


# ===========================================================================
# U-14: helper table tests, including NAV <= 0.
# ===========================================================================


def test_u14_sizing_basis_table_including_nav_le_zero() -> None:
    """U-14: _sizing_basis table, including NAV <= 0 (floored at 0)."""
    assert LiveExecutionEngine._sizing_basis(Decimal("-50"), Decimal("1000")) == Decimal("0")
    assert LiveExecutionEngine._sizing_basis(Decimal("0"), Decimal("1000")) == Decimal("0")
    assert LiveExecutionEngine._sizing_basis(Decimal("500"), Decimal("1000")) == Decimal("500")
    assert LiveExecutionEngine._sizing_basis(Decimal("1500"), Decimal("1000")) == Decimal("1000")


def test_u14_cap_buy_quantity_table_including_non_positive_inputs() -> None:
    """U-14: _cap_buy_quantity table, including non-positive qty/price/available."""
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("10"), Decimal("100"), Decimal("500"), Decimal("0.01")
    ) == Decimal("500") / (Decimal("100") * Decimal("1.01"))
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("1"), Decimal("100"), Decimal("500"), Decimal("0.01")
    ) == Decimal("1")
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("10"), Decimal("100"), Decimal("0"), Decimal("0.01")
    ) == Decimal("0")
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("0"), Decimal("100"), Decimal("500"), Decimal("0.01")
    ) == Decimal("0")
    assert LiveExecutionEngine._cap_buy_quantity(
        Decimal("10"), Decimal("0"), Decimal("500"), Decimal("0.01")
    ) == Decimal("0")


def test_u14_quote_asset_and_taker_buffer_helpers() -> None:
    """U-14: _quote_asset/_taker_buffer table tests."""
    markets = {
        "BTC/EUR": _market("BTC", "EUR", taker=Decimal("0.0025")),
        "ETH/EUR": _market("ETH", "EUR"),  # no explicit taker -> default 1%
    }
    engine, _, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=["BTC/EUR", "ETH/EUR"], markets=markets,
    )

    assert engine._quote_asset("BTC/EUR") == "EUR"
    assert engine._quote_asset("UNKNOWN/XYZ") is None
    assert engine._taker_buffer("BTC/EUR") == Decimal("0.0025")
    assert engine._taker_buffer("ETH/EUR") == Decimal("0.01")
    assert engine._taker_buffer("UNKNOWN/XYZ") == Decimal("0.01")


def test_u14_inflight_buy_orders_helper_table() -> None:
    """U-14: _inflight_buy_orders returns the blocking Order for
    OPEN/PARTIAL/PENDING_SUBMIT and unrouted-settled BUYs; None when
    empty, REJECTED, or fully routed (security round 2: returns the
    order itself, not a bool -- WP14-S-04)."""
    engine, _, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=["BTC/EUR"], markets={"BTC/EUR": _market("BTC")},
    )
    assert engine._inflight_buy_orders() is None

    order = _make_order(symbol="BTC/EUR", side=OrderSide.BUY)
    open_order = order.model_copy(update={"status": OrderStatus.OPEN})
    engine._orders[open_order.order_id] = open_order
    # Give it an exchange id so the S-05 reconcile-flag branch (tested
    # separately below) does not also fire here.
    engine._exchange_order_map[open_order.order_id] = "exch-open-1"
    blocking = engine._inflight_buy_orders()
    assert blocking is not None
    assert blocking.order_id == open_order.order_id

    engine._orders.clear()
    filled = order.model_copy(
        update={"status": OrderStatus.FILLED, "filled_quantity": Decimal("1")}
    )
    engine._orders[filled.order_id] = filled
    assert engine._inflight_buy_orders() is not None  # not yet routed

    engine._routed_gross_qty[filled.order_id] = Decimal("1")
    assert engine._inflight_buy_orders() is None  # fully routed now

    engine._orders.clear()
    rejected = order.model_copy(update={"status": OrderStatus.REJECTED})
    engine._orders[rejected.order_id] = rejected
    assert engine._inflight_buy_orders() is None


def test_u14_inflight_buy_orders_dust_residual_does_not_block() -> None:
    """WP14-S-04 (Probe PE): a settled BUY whose filled_quantity exceeds
    the routed amount by less than one amount step (a rounding residual,
    e.g. a CCXT float round-trip reporting filled = amount + 1e-9) does
    NOT block -- only a genuine unrouted amount above the dust tolerance
    does."""
    engine, _, _ = _make_engine(
        initial_cash=Decimal("1000"), symbols=["BTC/EUR"],
        markets={"BTC/EUR": _market("BTC")},  # amount precision: 8dp -> step 1e-8
    )
    order = _make_order(symbol="BTC/EUR", side=OrderSide.BUY, quantity=Decimal("1"))
    settled = order.model_copy(
        update={"status": OrderStatus.FILLED, "filled_quantity": Decimal("1.000000001")}
    )
    engine._orders[settled.order_id] = settled
    engine._routed_gross_qty[settled.order_id] = Decimal("1")

    assert engine._inflight_buy_orders() is None

    # A residual ABOVE the tolerance still blocks.
    settled2 = order.model_copy(
        update={"status": OrderStatus.FILLED, "filled_quantity": Decimal("1.01")}
    )
    engine._orders.clear()
    engine._orders[settled2.order_id] = settled2
    engine._routed_gross_qty[settled2.order_id] = Decimal("1")
    assert engine._inflight_buy_orders() is not None


def test_u14_equity_snapshot_type_is_frozen_dataclass() -> None:
    """U-14: EquitySnapshot is the frozen dataclass the spec mandates."""
    snap = EquitySnapshot(
        nav=Decimal("1"), peak=Decimal("1"), cash=Decimal("1"), sizing_basis=Decimal("1")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.nav = Decimal("2")  # type: ignore[misc]
