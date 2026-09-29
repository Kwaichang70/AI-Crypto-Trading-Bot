"""
tests/unit/test_wp14b_submit_classification.py
--------------------------------------------------
WP1.4b (idempotent-submit spec) unit tests:

- T3: every class in D3's allowlist, parametrised -- "not placed" classes
  reject immediately with no lookup; "ambiguous" classes (including the
  D3 edge cases: InvalidNonce, OrderNotFound, DuplicateOrderId, an
  exact-class ExchangeError, BadResponse, a bare ValueError, and a success
  response with no id) resolve via the cid lookup. W1 (at most one
  create_order call, ever) holds throughout.
- T13: an exact cid match is required -- a symbol or side mismatch on an
  otherwise-matching cid counts as "failed", never adopted.
- T14: a flag with a different (non-submit-unknown) reason is never
  overwritten or cleared by this WP's own bookkeeping.
- T15: every submit-path log event is bounded (``error`` truncated to 200
  chars) and never carries the raw response, request params or headers.

Module under test: packages/trading/engines/live.py
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import ccxt.async_support as ccxt_async
import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, OrderStatus, OrderType
from trading.engines.live import LiveExecutionEngine
from trading.models import Order, RiskCheckResult

_SYMBOL = "BTC/USDT"
_RUN_ID = "wp14b-classify-run"


def _make_engine(*, exchange_id: str = "mock-exchange") -> tuple[LiveExecutionEngine, MagicMock]:
    ex = MagicMock()
    ex.id = exchange_id
    ex.markets = {_SYMBOL: {}}
    ex.create_order = AsyncMock(
        return_value={"id": "exch-001", "status": "open", "filled": "0", "average": None}
    )
    ex.fetch_orders = AsyncMock(return_value=[])
    ex.close = AsyncMock(return_value=None)

    rm = MagicMock()
    rm.pre_trade_check.return_value = RiskCheckResult(
        approved=True, adjusted_quantity=Decimal("0.1"), rejection_reasons=[], warnings=[],
    )
    rm.calculate_position_size.return_value = Decimal("0.1")

    engine = LiveExecutionEngine(
        run_id=_RUN_ID, risk_manager=rm, exchange=ex, enable_live_trading=True,
    )
    return engine, ex


def _make_order(*, side: OrderSide = OrderSide.BUY, symbol: str = _SYMBOL) -> Order:
    return Order(
        client_order_id=f"{_RUN_ID}-{uuid4().hex[:12]}",
        run_id=_RUN_ID,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
    )


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------

_NOT_PLACED: list[Exception] = [
    ccxt_async.InsufficientFunds("insufficient funds"),
    ccxt_async.InvalidOrder("bad order shape"),
    ccxt_async.BadRequest("bad request"),
    ccxt_async.BadSymbol("bad symbol"),
    ccxt_async.AuthenticationError("bad api key"),
    ccxt_async.PermissionDenied("permission denied"),
    ccxt_async.ArgumentsRequired("missing argument"),
    ccxt_async.NotSupported("not supported"),
    ccxt_async.OperationRejected("operation rejected"),
]

_AMBIGUOUS: list[Exception] = [
    ccxt_async.InvalidNonce("invalid nonce"),
    ccxt_async.OrderNotFound("order not found"),
    ccxt_async.DuplicateOrderId("duplicate client order id"),
    ccxt_async.ExchangeError("internal_server_error"),  # exact class, not a subclass
    ccxt_async.BadResponse("bad response body"),
    ccxt_async.NullResponse("null response body"),
    ccxt_async.RequestTimeout("request timed out"),
    ccxt_async.RateLimitExceeded("rate limited"),
    ccxt_async.DDoSProtection("ddos protection"),
    ccxt_async.ExchangeNotAvailable("503"),
    ccxt_async.OnMaintenance("on maintenance"),
    ValueError("a non-ccxt parse failure"),
]


@pytest.mark.parametrize("exc", _NOT_PLACED, ids=[type(e).__name__ for e in _NOT_PLACED])
@pytest.mark.asyncio
async def test_t3_not_placed_classes_reject_with_no_lookup(exc: Exception) -> None:
    engine, ex = _make_engine()
    ex.create_order.side_effect = exc
    order = _make_order()

    result = await engine.submit_order(order)

    assert result.status == OrderStatus.REJECTED
    ex.fetch_orders.assert_not_called()
    ex.create_order.assert_awaited_once()  # W1


@pytest.mark.parametrize("exc", _AMBIGUOUS, ids=[type(e).__name__ for e in _AMBIGUOUS])
@pytest.mark.asyncio
async def test_t3_ambiguous_classes_resolve_via_lookup(
    exc: Exception, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    ex.create_order.side_effect = exc
    order = _make_order()

    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    assert result.exchange_order_id is None
    ex.fetch_orders.assert_awaited()  # the lookup path was taken
    ex.create_order.assert_awaited_once()  # W1: still only one call, ever


@pytest.mark.asyncio
async def test_t3_success_with_no_id_is_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    """D3: a success response with no usable ``id`` (Coinbase's
    ``parse_order({})`` shape) is exactly as ambiguous as any exception --
    never silently stored as exchange_order_id "None" (W6)."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    ex.create_order.return_value = {"id": None, "status": "open", "filled": None}
    order = _make_order()

    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    assert result.exchange_order_id is None
    ex.fetch_orders.assert_awaited()
    ex.create_order.assert_awaited_once()


# ---------------------------------------------------------------------------
# T13: exact cid match required -- symbol/side mismatch counts as failed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t13_symbol_mismatch_on_cid_match_is_failed_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    order = _make_order()

    # A cid match with the WRONG symbol -- must never be adopted.
    ex.create_order.side_effect = ccxt_async.ExchangeError("ambiguous")
    ex.fetch_orders.return_value = [
        {
            "id": "exch-mismatch-1",
            "clientOrderId": order.client_order_id,
            "symbol": "ETH/USDT",
            "side": "buy",
            "status": "closed",
            "filled": "0.1",
            "average": "50000",
        }
    ]

    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    assert result.exchange_order_id is None
    assert engine.reconcile_required.get(_SYMBOL) == "submit_lookup_mismatch"


@pytest.mark.asyncio
async def test_t13_side_mismatch_on_cid_match_is_failed_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    order = _make_order(side=OrderSide.BUY)

    ex.create_order.side_effect = ccxt_async.ExchangeError("ambiguous")
    ex.fetch_orders.return_value = [
        {
            "id": "exch-mismatch-2",
            "clientOrderId": order.client_order_id,
            "symbol": _SYMBOL,
            "side": "sell",  # wrong side
            "status": "closed",
            "filled": "0.1",
            "average": "50000",
        }
    ]

    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    assert result.exchange_order_id is None
    assert engine.reconcile_required.get(_SYMBOL) == "submit_lookup_mismatch"


# ---------------------------------------------------------------------------
# T14: a different reason already flagged is never overwritten or cleared.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t14_existing_different_flag_never_overwritten_or_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    engine._flag_reconcile(_SYMBOL, "own_exceeds_exchange_total")

    ex.create_order.side_effect = ccxt_async.ExchangeError("ambiguous")
    order = _make_order()
    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    # D11: buy_submit_unknown must NOT have overwritten the pre-existing,
    # unrelated reason.
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"

    # Resolve the unknown submit (adopt it) -- the pre-existing flag must
    # STILL not be cleared by this WP's self-clearing bookkeeping.
    ex.fetch_orders.return_value = [
        {
            "id": "exch-found-1",
            "clientOrderId": order.client_order_id,
            "symbol": _SYMBOL,
            "side": "buy",
            "status": "closed",
            "filled": "0.1",
            "average": "50000",
        }
    ]
    await engine._resolve_unknown_submits(_SYMBOL)

    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"


@pytest.mark.asyncio
async def test_t14_reverse_case_i8_overwrites_then_adoption_i8_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T14, reverse order (WP1.4b round 2, R-06): an unknown SELL sets
    ``sell_submit_unknown`` FIRST; an I8 mismatch then overwrites it
    (``_flag_reconcile`` always overwrites, I4); adoption must NOT restore
    or clear the I8 reason -- ``_maybe_clear_submit_unknown`` only ever
    clears when the CURRENT reason is still exactly one of the two
    self-clearing ones.

    (The companion R-01 case -- a ``balance_unavailable`` outage-and-
    recovery cycle re-setting the flag -- is covered by
    ``test_r01_balance_unavailable_outage_re_flags_submit_unknown`` in
    ``tests/integration/test_wp14b_idempotent_submit.py``, which needs the
    full ``process_signal``/``_held_quantity`` path this module's bare
    mock exchange does not exercise.)
    """
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()

    ex.create_order.side_effect = ccxt_async.ExchangeError("ambiguous")
    order = _make_order(side=OrderSide.SELL)
    result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT
    assert engine.reconcile_required.get(_SYMBOL) == "sell_submit_unknown"

    # I8 overwrites it (a real mismatch takes priority over this WP's own
    # self-clearing bookkeeping).
    engine._flag_reconcile(_SYMBOL, "own_exceeds_exchange_total")
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"

    # Adoption resolves the unknown submit -- I8's reason must remain.
    ex.fetch_orders.return_value = [
        {
            "id": "exch-found-2",
            "clientOrderId": order.client_order_id,
            "symbol": _SYMBOL,
            "side": "sell",
            "status": "closed",
            "filled": "0.1",
            "average": "50000",
        }
    ]
    await engine._resolve_unknown_submits(_SYMBOL)

    assert order.order_id not in engine._unknown_submits
    assert engine.reconcile_required.get(_SYMBOL) == "own_exceeds_exchange_total"


# ---------------------------------------------------------------------------
# T15: logs are bounded and never carry the response/params/headers.
# ---------------------------------------------------------------------------

_FORBIDDEN_LOG_KEYS = {"response", "params", "ccxt_response", "headers", "last_http_response"}


@pytest.mark.asyncio
async def test_t15_submit_logs_are_bounded_and_response_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    engine, ex = _make_engine()
    huge_message = "X" * 5000
    ex.create_order.side_effect = ccxt_async.ExchangeError(huge_message)
    ex.fetch_orders.return_value = []
    order = _make_order()

    with capture_logs() as cap:
        result = await engine.submit_order(order)

    assert result.status == OrderStatus.PENDING_SUBMIT

    submit_events = [
        e for e in cap
        if str(e.get("event", "")).startswith("live.order_sub")
        or str(e.get("event", "")).startswith("live.submit_lookup")
    ]
    assert submit_events, "expected at least one submit-path log event"
    for event in submit_events:
        for key in _FORBIDDEN_LOG_KEYS:
            assert key not in event, f"{event.get('event')} must never log {key!r}"
        error_field = event.get("error")
        if error_field is not None:
            assert len(str(error_field)) <= 200
