"""
tests/unit/test_wp111a_sell_cap.py
------------------------------------
WP1.11a (external-coin-safe SELL cap) mandatory tests T1-T15.

Module under test
-----------------
    packages/trading/engines/live.py -- LiveExecutionEngine (D1-D11)

Two fixture styles are used, matching the rest of the ``test_live_*``
suite:

- ``_make_engine_with_source``: a real ``PortfolioAccounting`` (as the
  ``LivePositionSource``) + a ``MagicMock`` exchange, mirroring
  ``test_live_position_ledger.py``'s fixture of the same name. Used for
  every test where the exact ccxt balance-response *semantics* (v2 vs v3)
  don't matter -- only the shape of the response does.
- ``FakeCCXTExchange`` (``tests/integration/fakes/fake_ccxt_exchange.py``):
  a real, order-book-aware fake. Used for T1 (CE1/P9, needs real v2
  ``free == total`` semantics) and T5 (needs a real ``lock_balance`` hold
  that doesn't touch our own tracked quantity).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import ccxt.async_support as ccxt_async
import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from trading.engines.live import _INFLIGHT_SELL_MAX_AGE_S, LiveExecutionEngine
from trading.models import Order, Position, RiskCheckResult, Signal
from trading.portfolio import PortfolioAccounting

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_BASE = "BTC"
_QUOTE = "USDT"
_RUN_ID = "wp111a-test-run"
_STRATEGY_ID = "wp111a-test-strategy"
_LAST_PRICE = Decimal("50000")

_DEFAULT_MARKET: dict[str, Any] = {
    "base": _BASE,
    "quote": _QUOTE,
    "precision": {"amount": 8, "price": 2},
}


# ---------------------------------------------------------------------------
# Factory helpers (MagicMock exchange + real PortfolioAccounting)
# ---------------------------------------------------------------------------


def _make_mock_exchange(
    *,
    markets: dict[str, Any] | None = None,
    fetch_balance_response: dict[str, Any] | None = None,
) -> MagicMock:
    exchange = MagicMock()
    exchange.id = "mock-exchange"
    exchange.markets = markets if markets is not None else {_SYMBOL: dict(_DEFAULT_MARKET)}
    exchange.create_order = AsyncMock(
        return_value={
            "id": "exch-001", "status": "open", "filled": "0",
            "average": None, "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_order = AsyncMock(
        return_value={
            "id": "exch-001", "status": "closed", "filled": "0",
            "average": str(_LAST_PRICE), "price": str(_LAST_PRICE),
        }
    )
    exchange.fetch_ticker = AsyncMock(return_value={"last": str(_LAST_PRICE)})
    exchange.fetch_balance = AsyncMock(
        return_value=fetch_balance_response
        or {"total": {_BASE: 0.0}, "free": {_BASE: 0.0}}
    )
    exchange.load_markets = AsyncMock(return_value=exchange.markets)
    exchange.cancel_order = AsyncMock(return_value={"id": "exch-001", "status": "canceled"})
    exchange.close = AsyncMock(return_value=None)
    exchange.has = {"fetchOrderTrades": True, "fetchMyTrades": True}
    exchange.fetch_order_trades = AsyncMock(return_value=[])
    exchange.fetch_my_trades = AsyncMock(return_value=[])
    exchange.fetch_orders = AsyncMock(return_value=[])
    return exchange


def _make_risk_manager_mock(*, approved: bool = True) -> MagicMock:
    mock = MagicMock()

    def _pre_trade_check(*, order: Order, **_: Any) -> RiskCheckResult:
        return RiskCheckResult(
            approved=approved,
            adjusted_quantity=order.quantity if approved else Decimal("0"),
            rejection_reasons=[] if approved else ["test rejection"],
            warnings=[],
        )

    mock.pre_trade_check.side_effect = _pre_trade_check
    mock.calculate_position_size.return_value = Decimal("999")
    return mock


def _make_engine_with_source(
    *,
    markets: dict[str, Any] | None = None,
    fetch_balance_response: dict[str, Any] | None = None,
    exchange: MagicMock | None = None,
    initial_cash: Decimal = Decimal("100000"),
) -> tuple[LiveExecutionEngine, PortfolioAccounting, MagicMock, MagicMock]:
    """Build an engine with a *real* PortfolioAccounting attached as its
    LivePositionSource, exactly as StrategyEngine.__init__ does. Pass an
    already-built ``exchange`` to share one exchange across two engines
    (T6)."""
    rm = _make_risk_manager_mock()
    ex = exchange if exchange is not None else _make_mock_exchange(
        markets=markets, fetch_balance_response=fetch_balance_response,
    )
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=initial_cash)
    engine.attach_position_source(portfolio, symbols=[_SYMBOL])
    return engine, portfolio, rm, ex


def _make_signal(
    *,
    direction: SignalDirection = SignalDirection.SELL,
    target_position: Decimal = Decimal("0"),
    confidence: float = 1.0,
) -> Signal:
    return Signal(
        strategy_id=_STRATEGY_ID, symbol=_SYMBOL, direction=direction,
        target_position=target_position, confidence=confidence,
    )


def _open_position(quantity: Decimal) -> Position:
    return Position(
        symbol=_SYMBOL, run_id=_RUN_ID, quantity=quantity,
        average_entry_price=_LAST_PRICE, current_price=_LAST_PRICE,
    )


def _stale_open_sell_order(quantity: Decimal) -> Order:
    """An OPEN SELL, already older than ``_INFLIGHT_SELL_MAX_AGE_S`` --
    the exact shape ``_pending_sell_quantity``'s D2/D3 stale branch (and
    ``_stale_open_sell``'s BUY-block check) matches on."""
    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=quantity,
    )
    return order.model_copy(update={
        "status": OrderStatus.OPEN,
        "updated_at": datetime.now(UTC) - timedelta(seconds=_INFLIGHT_SELL_MAX_AGE_S + 1),
    })


# ===========================================================================
# T1 (P9/CE1, v2 fake): second SELL at +301s returns [].
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t1_p9_ce1_v2_fake_stale_sell_blocks_second_sell() -> None:
    """T1 (P9/CE1): on a v2-semantics exchange (``free == total``, real
    ccxt Coinbase's default ``fetchBalance`` option), a stale OPEN SELL
    that has most likely already filled (every live SELL is a MARKET
    order) must not let a second SELL sell the external coin it left
    behind. Proves D1's pure-ledger cap needs no ``free - external`` term
    even under the exact semantics that broke the withdrawn security
    formula (WP111a-S-01's CE1).
    """
    ex = FakeCCXTExchange(exchange_id="coinbase", balance_mode="v2")
    ex.register_market(_SYMBOL, base=_BASE, quote=_QUOTE, amount_precision=8)
    ex.seed_flat_bars(_SYMBOL, count=3, price=_LAST_PRICE)
    await ex.load_markets()
    ex.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    ex.set_balance(_BASE, Decimal("1.01"))  # 0.01 own + 1.0 external

    rm = _make_risk_manager_mock()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=Decimal("100000"))
    engine.attach_position_source(portfolio, symbols=[_SYMBOL])
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.01"))

    # The fake fills the SELL instantly on its own book (balance mutated),
    # but fetch_order fails from the very first post-submit reconcile
    # onward -- exactly a persistently-unreconcilable order (Coinbase
    # async processing that never resolves, P9/CE1).
    async def _always_fails(*args: Any, **kwargs: Any) -> Any:
        raise ccxt_async.OrderNotFound("no such order")

    ex.fetch_order = _always_fails  # type: ignore[assignment]

    first_orders = await engine.process_signal(_make_signal())
    assert len(first_orders) == 1
    first_order_id = first_orders[0].order_id
    assert engine._orders[first_order_id].status == OrderStatus.OPEN
    # Age it past the staleness threshold.
    engine._orders[first_order_id] = engine._orders[first_order_id].model_copy(
        update={"updated_at": datetime.now(UTC) - timedelta(seconds=_INFLIGHT_SELL_MAX_AGE_S + 1)}
    )

    with capture_logs() as cap:
        second_orders = await engine.process_signal(_make_signal())

    assert second_orders == [], "must never sell the external coin the stale SELL left behind"
    assert any(e.get("event") == "live.sell_reserve_stale" for e in cap)
    assert engine.reconcile_required.get(_SYMBOL) == "sell_order_state_unknown"


# ===========================================================================
# T2 (CE2): stale OPEN SELL that has filled, plus external coins, gives 0.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t2_ce2_stale_filled_sell_with_external_gives_zero() -> None:
    """T2 (CE2): the resting SELL has actually filled on the exchange
    (B=0 own left there, only 1.0 external remains) but the local copy is
    still OPEN. The ledger reservation alone (no balance read needed)
    must still give cap 0."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 1.0}, "free": {_BASE: 1.0}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    stale = _stale_open_sell_order(Decimal("0.02"))
    engine._orders[stale.order_id] = stale
    engine._exchange_order_map[stale.order_id] = "exch-stale"
    ex.fetch_order.side_effect = Exception("OrderNotFound: no such order")

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0")
    assert sell_cap.own_avail == Decimal("0")

    orders = await engine.process_signal(_make_signal())
    assert orders == []


# ===========================================================================
# T3 (CE3): partial fill, then cancel -- reconciled, or qty-routed stays
# reserved.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t3_cancel_reconciles_the_true_fill() -> None:
    """T3a: cancel_order's post-ack reconcile (D5(b)) picks up a fill the
    exchange confirms happened AFTER the last local PARTIAL update --
    _cancel_unconfirmed clears and the reservation shrinks to match the
    now-known-true filled_quantity."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.OPEN})
    order = order.model_copy(update={
        "status": OrderStatus.PARTIAL, "filled_quantity": Decimal("0.01"),
    })
    engine._orders[order.order_id] = order
    engine._exchange_order_map[order.order_id] = "exch-partial"

    ex.fetch_order.return_value = {
        "id": "exch-partial", "status": "closed", "filled": "0.02",
        "average": str(_LAST_PRICE), "price": str(_LAST_PRICE),
    }
    canceled = await engine.cancel_order(order.order_id)

    assert canceled.status == OrderStatus.CANCELED
    assert order.order_id not in engine._cancel_unconfirmed
    assert engine._orders[order.order_id].filled_quantity == Decimal("0.02")
    # Nothing has been routed yet -- the full known fill still reserves.
    assert engine._pending_sell_quantity(_SYMBOL) == Decimal("0.02")


@pytest.mark.asyncio
async def test_wp111a_t3_unconfirmed_cancel_reserves_full_remaining_quantity() -> None:
    """T3b: if the post-cancel reconcile itself fails, the order stays in
    _cancel_unconfirmed and _pending_sell_quantity reserves the FULL
    remaining quantity (D1) -- never just the last known filled_quantity,
    since the cancel may have raced a complete fill with no evidence
    either way yet."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={
        "status": OrderStatus.PARTIAL, "filled_quantity": Decimal("0.01"),
    })
    engine._orders[order.order_id] = order
    engine._exchange_order_map[order.order_id] = "exch-partial2"
    ex.fetch_order.side_effect = Exception("network blip")

    canceled = await engine.cancel_order(order.order_id)

    assert canceled.status == OrderStatus.CANCELED
    assert order.order_id in engine._cancel_unconfirmed
    # D1: qty(0.02) - routed(0), NOT filled_quantity(0.01) - routed(0).
    assert engine._pending_sell_quantity(_SYMBOL) == Decimal("0.02")


# ===========================================================================
# T4: normal exit returns the full quantity; external coins with no
# orders give exactly own.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t4_normal_exit_returns_full_quantity() -> None:
    """T4a: no external coins, no open orders -- a full close sells
    exactly own."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    orders = await engine.process_signal(_make_signal())
    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.02")


@pytest.mark.asyncio
async def test_wp111a_t4_external_coins_no_orders_gives_exactly_own() -> None:
    """T4b: external coins exist but no open orders on this symbol at
    all -- the exit is exactly own, never touching the external part."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 1.02}, "free": {_BASE: 1.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    orders = await engine.process_signal(_make_signal())
    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.02")


# ===========================================================================
# T5 (v3 fake): the user's manual SELL does not wrongly block the exit.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t5_v3_manual_sell_does_not_block_exit() -> None:
    """T5: a v3-semantics exchange (``free = available``, ``used = hold``)
    with the user's own manual SELL resting on PART of the external
    stash -- ``free`` still comfortably covers ``own_avail``, so the cap
    is exactly ``own_avail`` (D1), never wrongly reduced."""
    ex = FakeCCXTExchange(exchange_id="coinbase", balance_mode="v3")
    ex.register_market(_SYMBOL, base=_BASE, quote=_QUOTE, amount_precision=8)
    ex.seed_flat_bars(_SYMBOL, count=3, price=_LAST_PRICE)
    await ex.load_markets()
    ex.set_now_ms(int(datetime.now(tz=UTC).timestamp() * 1000))
    ex.set_balance(_BASE, Decimal("1.02"))  # 0.02 own + 1.0 external
    ex.lock_balance(_BASE, Decimal("0.5"))  # the user's manual SELL, well inside the external stash

    rm = _make_risk_manager_mock()
    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
    )
    portfolio = PortfolioAccounting(run_id=_RUN_ID, initial_cash=Decimal("100000"))
    engine.attach_position_source(portfolio, symbols=[_SYMBOL])
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    orders = await engine.process_signal(_make_signal())
    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.02")


# ===========================================================================
# T6: two engines sharing one balance -- neither sells beyond its own
# ledger.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t6_two_engines_share_one_balance() -> None:
    """T6: engine A owns 0.02, engine B owns 0.03, the shared exchange
    reports total=free=0.05 (exactly the sum, no other external coins).
    Each engine's cap must come from its OWN ledger, never the other's,
    and neither may exceed its own tracked quantity even though the
    shared free balance (0.05) would allow it."""
    shared_ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.05}, "free": {_BASE: 0.05}},
    )
    engine_a, portfolio_a, _, _ = _make_engine_with_source(exchange=shared_ex)
    portfolio_a._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    engine_b, portfolio_b, _, _ = _make_engine_with_source(exchange=shared_ex)
    portfolio_b._position_snapshots[_SYMBOL] = _open_position(Decimal("0.03"))

    orders_a = await engine_a.process_signal(_make_signal())
    orders_b = await engine_b.process_signal(_make_signal())

    assert len(orders_a) == 1 and orders_a[0].quantity == Decimal("0.02")
    assert len(orders_b) == 1 and orders_b[0].quantity == Decimal("0.03")


# ===========================================================================
# T7: a deposit or a withdrawal between bars -- cap never exceeds
# own_avail*.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t7_deposit_and_withdrawal_never_inflate_cap() -> None:
    """T7: a deposit (external funds arrive) never raises the cap above
    own_avail*; a withdrawal of OUR OWN coins shrinks the cap to free and
    raises the (non-hard-block) I8 flag, but the cap is still bounded by
    own_avail* on the way back up."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    cap1 = await engine._held_quantity(_SYMBOL)
    assert cap1.cap == Decimal("0.02")

    # A deposit: 1 BTC of external funds arrives between bars.
    ex.fetch_balance.return_value = {"total": {_BASE: 1.02}, "free": {_BASE: 1.02}}
    cap2 = await engine._held_quantity(_SYMBOL)
    assert cap2.cap == Decimal("0.02"), "a deposit must never inflate the cap above own_avail*"

    # A withdrawal of OUR OWN coins: only 0.01 of our 0.02 remains.
    ex.fetch_balance.return_value = {"total": {_BASE: 0.01}, "free": {_BASE: 0.01}}
    cap3 = await engine._held_quantity(_SYMBOL)
    assert cap3.cap == Decimal("0.01")
    assert cap3.cap <= cap3.own_avail
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"


# ===========================================================================
# T8: NaN, Inf, negative, None, or a missing base coin -- unavailable,
# with no exception.
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balance",
    [
        {"total": {_BASE: float("nan")}, "free": {_BASE: 0.02}},
        {"total": {_BASE: 0.02}, "free": {_BASE: float("inf")}},
        {"total": {_BASE: 0.02}, "free": {_BASE: -1.0}},
        {"total": {_BASE: 0.02}, "free": {}},
        {"total": {}, "free": {}},
    ],
    ids=["total-nan", "free-inf", "free-negative", "free-missing", "base-missing"],
)
async def test_wp111a_t8_malformed_balance_fields_are_unavailable_no_exception(
    balance: dict[str, Any],
) -> None:
    """T8: every malformed/missing balance field is treated as
    unavailable (D8) -- with a clean ledger, that means D6's
    floor_prec(own_avail) fallback, never an exception and never a
    silently-wrong cap."""
    ex = _make_mock_exchange(fetch_balance_response=balance)
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0.02")
    assert sell_cap.cap_reason == "balance_unavailable"


# ===========================================================================
# T9: U-111a matrix -- clean ledger gives floor; each doubt source gives
# 0 plus critical; doubt survives a balance_unavailable overwrite.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t9_clean_ledger_fetch_failure_gives_floored_own_avail() -> None:
    """T9 (clean ledger branch): a fetch failure with NO doubt source
    falls back to floor_prec(own_avail), logged at critical only on the
    doubt path (not this one)."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0.02")
    assert sell_cap.cap_reason == "balance_unavailable"


@pytest.mark.asyncio
async def test_wp111a_t9_unknown_sell_submit_doubt_blocks_fetch_failure() -> None:
    """T9 (doubt source 1/5): an unresolved unknown SELL submit."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )
    engine._orders[order.order_id] = order  # PENDING_SUBMIT, no exchange id
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from trading.engines.live import _UnknownSubmit
    engine._unknown_submits[order.order_id] = _UnknownSubmit(
        symbol=_SYMBOL, side=OrderSide.SELL, submit_at=_dt.now(tz=_UTC),
    )

    with capture_logs() as cap:
        sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0")
    assert sell_cap.cap_reason == "ledger_doubt"
    assert any(
        e.get("event") == "live.sell_blocked_ledger_doubt" and e.get("log_level") == "critical"
        for e in cap
    )


@pytest.mark.asyncio
async def test_wp111a_t9_stale_open_sell_doubt_blocks_fetch_failure() -> None:
    """T9 (doubt source 2/5): a stale OPEN/PARTIAL SELL."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    stale = _stale_open_sell_order(Decimal("0.01"))
    engine._orders[stale.order_id] = stale
    engine._exchange_order_map[stale.order_id] = "exch-stale"
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0")
    assert sell_cap.cap_reason == "ledger_doubt"


@pytest.mark.asyncio
async def test_wp111a_t9_contradicted_reserve_doubt_blocks_fetch_failure() -> None:
    """T9 (doubt source 3/5): a positive contradicted-SELL reserve."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    engine._contradicted_sell_reserve[_SYMBOL] = Decimal("0.005")
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0")
    assert sell_cap.cap_reason == "ledger_doubt"


@pytest.mark.asyncio
async def test_wp111a_t9_cancel_unconfirmed_doubt_blocks_fetch_failure() -> None:
    """T9 (doubt source 4/5): an order still awaiting cancel confirmation."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.01"),
    )
    order = order.model_copy(update={"status": OrderStatus.CANCELED})
    engine._orders[order.order_id] = order
    engine._exchange_order_map[order.order_id] = "exch-cancelled"
    engine._cancel_unconfirmed.add(order.order_id)
    ex.fetch_balance.side_effect = Exception("exchange unavailable")

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0")
    assert sell_cap.cap_reason == "ledger_doubt"


@pytest.mark.asyncio
async def test_wp111a_t9_i8_doubt_blocks_fetch_failure_and_survives_overwrite() -> None:
    """T9 (doubt source 5/5, and the "survives overwrite" clause): a
    sticky I8 mismatch keeps failing SELLs closed even after
    ``reconcile_required[symbol]`` gets overwritten to
    ``"balance_unavailable"`` by some unrelated sync_positions call --
    ``_ledger_doubt`` is derived from ``_i8_doubt`` directly, never from
    ``reconcile_required``."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))
    engine._i8_doubt.add(_SYMBOL)
    # Simulate the single-reason map losing the I8 reason to a later,
    # unrelated balance_unavailable flag (the exact race D6's background
    # note warns about).
    engine._flag_reconcile(_SYMBOL, "balance_unavailable")
    assert engine.reconcile_required.get(_SYMBOL) == "balance_unavailable"

    ex.fetch_balance.side_effect = Exception("exchange unavailable")
    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap == Decimal("0"), "the sticky I8 doubt must survive the reason overwrite"
    assert sell_cap.cap_reason == "ledger_doubt"


# ===========================================================================
# T10: I8 gives min(own_avail*, free), a critical log, and the sticky
# doubt -- never a hard block (U1).
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t10_i8_mismatch_is_not_a_hard_block() -> None:
    """T10: own_avail (0.03) exceeds the fresh exchange total (0.02) --
    NOT a hard block (U1): the cap is still min(own_avail, free), the
    mismatch is logged critical, and the symbol becomes sticky-doubtful."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.03"))

    with capture_logs() as cap:
        sell_cap = await engine._held_quantity(_SYMBOL)

    assert sell_cap.cap == Decimal("0.02"), "I8 must not zero the cap (U1, not a hard block)"
    assert sell_cap.cap_reason == "i8"
    assert _SYMBOL in engine._i8_doubt
    assert any(
        e.get("event") == "live.sell_i8_mismatch" and e.get("log_level") == "critical"
        for e in cap
    )
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"


# ===========================================================================
# T11: live.sell_capped carries exactly the documented fields, no balance
# dict, no `info`.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t11_sell_capped_log_fields() -> None:
    """T11: whenever the cap constrains the sold quantity, ``live.sell_capped``
    carries symbol/strategy_id/requested/own/reserved/own_avail/cap/
    cap_reason/balance_fresh/free -- and nothing balance-dict-shaped."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.01}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    with capture_logs() as cap:
        orders = await engine.process_signal(_make_signal())

    assert len(orders) == 1
    assert orders[0].quantity == Decimal("0.01")
    events = [e for e in cap if e.get("event") == "live.sell_capped"]
    assert len(events) == 1
    event = events[0]
    for key in (
        "symbol", "strategy_id", "requested", "own", "reserved",
        "own_avail", "cap", "cap_reason", "balance_fresh", "free",
    ):
        assert key in event, f"missing {key!r}"
    assert event["cap_reason"] == "free"
    assert "balance" not in event
    assert "info" not in event


# ===========================================================================
# T12: unknown submit plus a cancel with no exchange id -- reserve kept.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t12_unknown_submit_plus_no_id_cancel_keeps_reserve() -> None:
    """T12: an unknown SELL submit (no exchange id yet) reserves its full
    quantity; cancelling it (D5(a)) must NOT release that reserve -- the
    order is left exactly as PENDING_SUBMIT and the exchange is never
    called."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.PENDING_SUBMIT})
    engine._orders[order.order_id] = order  # PENDING_SUBMIT, no exchange id

    assert engine._pending_sell_quantity(_SYMBOL) == Decimal("0.02")

    canceled = await engine.cancel_order(order.order_id)

    assert canceled.status == OrderStatus.PENDING_SUBMIT
    ex.cancel_order.assert_not_called()
    assert engine._pending_sell_quantity(_SYMBOL) == Decimal("0.02"), (
        "the no-id cancel must not release the reserve"
    )

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.own_avail == Decimal("0")
    assert sell_cap.cap == Decimal("0")


# ===========================================================================
# T13: the G2 cid lookup releases a stale order.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t13_g2_cid_lookup_releases_stale_order() -> None:
    """T13: fetch_order keeps failing for a stale OPEN SELL, but a cid
    lookup (``fetch_orders``) finds the order closed with a finite fill --
    ``check_resting_orders``'s G2 fallback (D4) applies it and the order
    transitions to FILLED."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    stale = _stale_open_sell_order(Decimal("0.02"))
    engine._orders[stale.order_id] = stale
    engine._exchange_order_map[stale.order_id] = "exch-stale-g2"
    ex.fetch_order.side_effect = Exception("OrderNotFound: no such order")
    ex.fetch_orders.return_value = [
        {
            "id": "exch-stale-g2",
            "clientOrderId": stale.client_order_id,
            "symbol": _SYMBOL,
            "side": "sell",
            "status": "closed",
            "filled": "0.02",
            "average": str(_LAST_PRICE),
            "price": str(_LAST_PRICE),
        }
    ]

    candidates = await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)

    assert engine._orders[stale.order_id].status == OrderStatus.FILLED
    assert engine._orders[stale.order_id].filled_quantity == Decimal("0.02")
    assert stale.order_id in {o.order_id for o in candidates}


# ===========================================================================
# T14: `filled` None on a closed/terminal status is not settled.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t14_filled_none_on_closed_is_not_settled() -> None:
    """T14 (D9, A3 gap): a ``filled=None`` on a TERMINAL exchange status
    is a reconcile FAILURE -- no mutation, no ``updated_at`` refresh, so a
    stale/garbage response can never make a SELL's reservation look
    settled when it isn't."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.OPEN})
    before_updated_at = order.updated_at

    with capture_logs() as cap:
        result, confirmed = engine._apply_exchange_order_state(
            order,
            {"id": "exch-x", "status": "closed", "filled": None, "average": None, "price": "50000"},
        )

    assert confirmed is False
    assert result.status == OrderStatus.OPEN
    assert result.filled_quantity == Decimal("0")
    assert result.updated_at == before_updated_at
    assert any(e.get("event") == "live.reconcile_filled_invalid" for e in cap)


# ===========================================================================
# T15: lock -- two concurrent SELLs never sell more than own_avail*.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_t15_concurrent_sells_never_exceed_own_avail() -> None:
    """T15 (D10): two SELL signals fired concurrently for the same symbol
    must never together sell more than own_avail* -- the per-symbol lock
    (held from the cap computation through submit_order returning)
    serialises them."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    results = await asyncio.gather(
        engine.process_signal(_make_signal()),
        engine.process_signal(_make_signal()),
    )
    total_sold = sum(
        (order.quantity for orders in results for order in orders), Decimal("0")
    )
    assert total_sold <= Decimal("0.02")
    # Exactly one of the two must have gone through in full; the other
    # must have seen own_avail already reserved to 0.
    non_empty = [orders for orders in results if orders]
    assert len(non_empty) == 1
    assert non_empty[0][0].quantity == Decimal("0.02")


# ===========================================================================
# Round 2 (critic + security review): WP111a-C-01 / WP111a-S2-01 -- the
# per-symbol SELL lock must be released on ANY exception between the
# acquire and submit_order returning, not just on the happy-path returns.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_lock_released_on_pre_trade_check_exception() -> None:
    """WP111a-C-01/S2-01 (probe 1): risk_manager.pre_trade_check raises
    once -- the per-symbol lock must still be released, and the very next
    SELL for the same symbol must proceed (not hang forever)."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, rm, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    rm.pre_trade_check.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await engine.process_signal(_make_signal())

    assert not engine._sell_locks[_SYMBOL].locked(), (
        "the lock leaked: every future SELL for this symbol -- including "
        "stop-loss/take-profit exits -- would now hang forever"
    )

    # Restore a normal risk manager and confirm the next SELL for the
    # SAME symbol actually proceeds, bounded in time (a leaked lock would
    # hang here instead of raising or returning).
    def _pre_trade_check(*, order: Order, **_: Any) -> RiskCheckResult:
        return RiskCheckResult(
            approved=True, adjusted_quantity=order.quantity,
            rejection_reasons=[], warnings=[],
        )

    rm.pre_trade_check.side_effect = _pre_trade_check
    orders = await asyncio.wait_for(engine.process_signal(_make_signal()), timeout=3.0)
    assert len(orders) == 1


@pytest.mark.asyncio
async def test_wp111a_round2_lock_released_on_malformed_market_limit() -> None:
    """WP111a-C-01/S2-01 (probe 2): a market whose ``limits.amount.min``
    is a non-numeric string makes ``float(min_amount)`` raise
    ``ValueError`` -- the lock must still be released."""
    markets = {
        _SYMBOL: {
            "base": _BASE, "quote": _QUOTE,
            "precision": {"amount": 8, "price": 2},
            "limits": {"amount": {"min": "n/a"}, "cost": {"min": None}},
        }
    }
    ex = _make_mock_exchange(
        markets=markets,
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex, markets=markets)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    with pytest.raises(ValueError):
        await engine.process_signal(_make_signal())

    assert not engine._sell_locks[_SYMBOL].locked()

    # Fix the market and confirm the next SELL proceeds, bounded in time.
    ex.markets[_SYMBOL]["limits"]["amount"]["min"] = 0.0001
    orders = await asyncio.wait_for(engine.process_signal(_make_signal()), timeout=3.0)
    assert len(orders) == 1


@pytest.mark.asyncio
async def test_wp111a_round2_lock_released_on_ticker_fetch_cancelled() -> None:
    """WP111a-C-01/S2-01 (probe 3): a ``CancelledError`` raised while
    awaiting ``fetch_ticker`` (e.g. the task is cancelled mid-flight) must
    still release the lock -- ``CancelledError`` is a ``BaseException``,
    not caught by the inner ``except Exception``, so it only reaches the
    lock's release via the outer ``finally``."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.02}, "free": {_BASE: 0.02}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    ex.fetch_ticker.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await engine.process_signal(_make_signal())

    assert not engine._sell_locks[_SYMBOL].locked()


# ===========================================================================
# Round 2: WP111a-S2-02 -- the create_order/adoption path (shared code,
# _apply_create_response) had no D9-equivalent guard on `filled`.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_s202_create_response_nan_filled_stays_open() -> None:
    """S2-02: an instant "closed" reply with ``filled: "NaN"`` must not be
    trusted -- stay OPEN (full quantity still reserved via the OPEN
    branch, which doesn't depend on ``filled_quantity`` at all), log
    ``live.reconcile_filled_invalid`` at critical, and never raise."""
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.PENDING_SUBMIT})
    engine._orders[order.order_id] = order

    with capture_logs() as cap:
        result = engine._apply_create_response(
            order,
            {
                "id": "exch-nan", "status": "closed", "filled": "NaN",
                "average": None, "price": str(_LAST_PRICE),
            },
        )

    assert result.status == OrderStatus.OPEN
    assert engine._pending_sell_quantity(_SYMBOL) == Decimal("0.02")
    assert any(
        e.get("event") == "live.reconcile_filled_invalid" and e.get("log_level") == "critical"
        for e in cap
    )


# ===========================================================================
# Round 2: WP111a-S2-03 -- an I8 mismatch's BUY block must survive a later
# reconcile_required overwrite AND self-clear (_i8_doubt is sticky, J4).
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_s203_i8_doubt_blocks_buy_after_overwrite_and_recovery() -> None:
    """S2-03: I8 sets ``_i8_doubt``; a later ``balance_unavailable``
    overwrite (and its own self-clear on recovery) must not un-block
    BUYs -- ``_i8_doubt`` is checked directly, independent of whatever
    ``reconcile_required`` currently says."""
    ex = _make_mock_exchange(
        fetch_balance_response={"total": {_BASE: 0.01}, "free": {_BASE: 0.01}},
    )
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    sell_cap = await engine._held_quantity(_SYMBOL)
    assert sell_cap.cap_reason == "i8"
    assert _SYMBOL in engine._i8_doubt

    # Simulate the exact race the report describes: some other flow
    # overwrites the single-reason map, then clears it again.
    engine._flag_reconcile(_SYMBOL, "balance_unavailable")
    engine._maybe_clear_balance_unavailable(_SYMBOL)
    assert engine.reconcile_required.get(_SYMBOL) is None

    with capture_logs() as cap:
        orders = await engine.process_signal(
            _make_signal(direction=SignalDirection.BUY, target_position=Decimal("100"))
        )

    assert orders == []
    assert any(e.get("event") == "live.buy_blocked_i8_doubt" for e in cap)


# ===========================================================================
# Round 2 (recommended): WP111a-S2-04 -- a stale _cancel_unconfirmed SELL
# gets D3's critical alert + run-wide BUY block too.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_s204_stale_cancel_unconfirmed_blocks_buys() -> None:
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={
        "status": OrderStatus.CANCELED,
        "updated_at": datetime.now(UTC) - timedelta(seconds=_INFLIGHT_SELL_MAX_AGE_S + 1),
    })
    engine._orders[order.order_id] = order
    engine._exchange_order_map[order.order_id] = "exch-cancel-stale"
    engine._cancel_unconfirmed.add(order.order_id)

    with capture_logs() as cap:
        pending = engine._pending_sell_quantity(_SYMBOL)
    assert pending == Decimal("0.02")
    assert any(
        e.get("event") == "live.sell_reserve_stale" and e.get("log_level") == "critical"
        for e in cap
    )

    buy_orders = await engine.process_signal(
        _make_signal(direction=SignalDirection.BUY, target_position=Decimal("100"))
    )
    assert buy_orders == []


# ===========================================================================
# Round 2 (recommended): WP111a-S2-05 -- G2 cid match with a disagreeing
# exchange id must not be adopted.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_s205_g2_cid_match_wrong_exchange_id_not_adopted() -> None:
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    stale = _stale_open_sell_order(Decimal("0.02"))
    engine._orders[stale.order_id] = stale
    engine._exchange_order_map[stale.order_id] = "exch-mine"
    ex.fetch_order.side_effect = Exception("OrderNotFound: no such order")
    ex.fetch_orders.return_value = [
        {
            "id": "exch-foreign", "clientOrderId": stale.client_order_id,
            "symbol": _SYMBOL, "side": "sell", "status": "closed",
            "filled": "0.02", "average": str(_LAST_PRICE), "price": str(_LAST_PRICE),
        }
    ]

    with capture_logs() as cap:
        await engine.check_resting_orders(_SYMBOL, _LAST_PRICE)

    assert engine._orders[stale.order_id].status == OrderStatus.OPEN
    assert any(e.get("event") == "live.submit_lookup_cid_mismatch" for e in cap)
    assert engine.reconcile_required.get(_SYMBOL) == "submit_lookup_mismatch"


# ===========================================================================
# Round 2 (recommended): WP111a-S2-06 -- a reply with status None must
# not refresh updated_at.
# ===========================================================================


def test_wp111a_round2_s206_status_none_does_not_refresh_updated_at() -> None:
    ex = _make_mock_exchange()
    engine, _, _, _ = _make_engine_with_source(exchange=ex)

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.OPEN})
    before_updated_at = order.updated_at

    result, confirmed = engine._apply_exchange_order_state(
        order,
        {"id": "exch-x", "status": None, "filled": "0.01", "average": None, "price": "50000"},
    )

    assert confirmed is False
    assert result.status == OrderStatus.OPEN
    assert result.updated_at == before_updated_at


# ===========================================================================
# Round 2 (recommended): WP111a-S2-08 / WP111a-C-04 -- on_stop must use
# the transitioned order object, and only SELLs join
# _cancel_unconfirmed.
# ===========================================================================


@pytest.mark.asyncio
async def test_wp111a_round2_s208_on_stop_uses_transitioned_order() -> None:
    ex = _make_mock_exchange()
    engine, portfolio, _, _ = _make_engine_with_source(exchange=ex)
    portfolio._position_snapshots[_SYMBOL] = _open_position(Decimal("0.02"))

    order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.SELL,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    order = order.model_copy(update={"status": OrderStatus.OPEN})
    engine._orders[order.order_id] = order
    engine._exchange_order_map[order.order_id] = "exch-onstop"
    # The cancel ack succeeds, but the post-cancel reconcile's fetch_order
    # still reports "open" -- pre-fix, passing the STALE (pre-transition)
    # order into that reconcile let this overwrite self._orders back to
    # OPEN.
    ex.fetch_order.return_value = {
        "id": "exch-onstop", "status": "open", "filled": "0",
        "average": None, "price": "50000",
    }

    await engine.on_stop()

    assert engine._orders[order.order_id].status == OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_wp111a_round2_c04_on_stop_only_adds_sell_to_cancel_unconfirmed() -> None:
    ex = _make_mock_exchange()
    engine, _, _, _ = _make_engine_with_source(exchange=ex)

    buy_order = Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID, symbol=_SYMBOL, side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=Decimal("0.02"),
    )
    buy_order = buy_order.model_copy(update={"status": OrderStatus.OPEN})
    engine._orders[buy_order.order_id] = buy_order
    engine._exchange_order_map[buy_order.order_id] = "exch-buy-onstop"

    await engine.on_stop()

    assert buy_order.order_id not in engine._cancel_unconfirmed
    assert engine._orders[buy_order.order_id].status == OrderStatus.CANCELED
