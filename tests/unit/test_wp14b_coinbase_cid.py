"""
tests/unit/test_wp14b_coinbase_cid.py
----------------------------------------
WP1.4b (idempotent-submit spec) T2: the REAL ``ccxt.async_support.coinbase``
class, with only its low-level HTTP transport (``Exchange.fetch``, BELOW
``fetch2``) stubbed -- no network, no mocked ccxt request-building/retry
internals -- proves:

- D1/W2: the request body Coinbase actually receives carries our full
  ``client_order_id`` (cid), not the ``ccxt-<uuid>`` ccxt would otherwise
  mint on its own.
- WP1.4b round 2 (S-06): even with ``exchange.options["maxRetriesOnFailure"]
  = 3`` set (ccxt's own retry-on-failure knob, read by ``fetch2``), exactly
  ONE POST reaches the wire -- our per-call ``params["maxRetriesOnFailure"]
  = 0`` always wins (ccxt's ``handle_option_and_params`` checks ``params``
  before ``options``) and is stripped by ``fetch2`` before signing, so it
  never appears in the signed body either.
- WP1.4b round 3 (S-R2-01): the real Coinbase ``fetch_orders`` defaults
  ``limit`` to 100 and returns the OLDEST 100 orders in the (paginated,
  cursor-terminated) window -- with more than 100 orders in-window, our own
  (newest) order would be silently dropped unless every call site passes
  ``limit=None``. This is proved end-to-end against the real ccxt pagination
  / filtering code (``fetch_paginated_call_cursor``, ``filter_by_since_limit``,
  ``filter_by_limit``), not a re-implementation of it.

Stubbing at ``v3PrivatePostBrokerageOrders`` (the generated implicit-API
method) instead -- as this test did before round 2 -- would COMPLETELY
BYPASS ``fetch2``'s retry loop (that generated method's only job is to call
``self.request(...) -> self.fetch2(...)``), making S-06 unobservable. Only
a ``fetch``-level stub lets ``create_order`` -> the real Coinbase
``create_order`` -> the generated method -> ``request`` -> ``fetch2``
(retry loop) -> ``fetch`` (this stub) run for real. Likewise, the S-R2-01
test below stubs the GET side at ``fetch`` so the real ``fetch_orders`` ->
``fetch_paginated_call_cursor`` -> ``parse_orders``/``parse_order`` ->
``filter_by_since_limit`` chain runs for real.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import ccxt.async_support as ccxt_async
import pytest

from common.types import OrderSide, OrderStatus, OrderType
from trading.engines.live import LiveExecutionEngine
from trading.models import Order, RiskCheckResult

_SYMBOL = "BTC/USD"
_MARKET: dict[str, Any] = {
    "id": "BTC-USD",
    "symbol": _SYMBOL,
    "base": "BTC",
    "quote": "USD",
    "type": "spot",
    "spot": True,
    "precision": {"amount": 8, "price": 2},
    "limits": {"amount": {"min": 0.0001}, "cost": {"min": 1}},
}

# WP1.4b round 3 (S-R2-01): mirrors ``_lookup_by_cid``'s own margin
# constant (``_LOOKUP_SINCE_MARGIN_MS``) -- kept as a literal here (rather
# than imported) so the test fails loudly if that constant ever changes
# without this test being revisited.
_LOOKUP_SINCE_MARGIN_MS = 3_600_000


def _make_real_coinbase(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
    """A real ``ccxt.async_support.coinbase`` instance with ONLY
    ``Exchange.fetch`` (the actual HTTP transport, called by ``fetch2``
    after signing/retry handling) stubbed. Returns ``(exchange, calls)``
    where ``calls`` records every ``(method, url, body)`` the stub saw, in
    order.
    """
    exchange = ccxt_async.coinbase({
        "enableRateLimit": False,
        # Real credentials are required for ccxt's own ``sign()`` step
        # (called once per ``fetch2`` invocation, before the retry loop) --
        # never used for an actual network call, since ``fetch`` itself is
        # stubbed below.
        "apiKey": "wp14b-test-key",
        "secret": "wp14b-test-secret",
    })
    exchange.markets = {_SYMBOL: dict(_MARKET)}
    exchange.options["brokerId"] = "ccxt"

    calls: list[dict[str, Any]] = []

    async def _fake_fetch(
        url: str, method: str = "GET", headers: Any = None, body: Any = None,
    ) -> dict[str, Any]:
        calls.append({"method": method, "url": url, "body": body})
        request = json.loads(body) if body else {}
        cid = request.get("client_order_id")
        return {
            "success": True,
            "order_id": "exch-real-99",
            "success_response": {
                "order_id": "exch-real-99",
                "product_id": _MARKET["id"],
                "side": request.get("side", "BUY"),
                "client_order_id": cid,
            },
            "order_configuration": {},
        }

    monkeypatch.setattr(exchange, "fetch", _fake_fetch, raising=False)
    return exchange, calls


def _make_risk_manager_mock() -> MagicMock:
    mock = MagicMock()
    mock.pre_trade_check.return_value = RiskCheckResult(
        approved=True, adjusted_quantity=Decimal("0.01"),
        rejection_reasons=[], warnings=[],
    )
    mock.calculate_position_size.return_value = Decimal("0.01")
    return mock


@pytest.mark.asyncio
async def test_t2_real_coinbase_class_receives_our_cid_and_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2: the outgoing request body's ``client_order_id`` equals our own
    49-character ``{run_id}-<12 hex>`` cid, proved against the REAL ccxt
    Coinbase adapter code (metaprogrammed request building, ``omit``,
    ``extend``, ``fetch2``'s retry loop, ``parse_order`` -- none of it
    mocked), with only the network transport stubbed.

    S-06: with ``exchange.options["maxRetriesOnFailure"] = 3`` (a knob
    nothing in this codebase sets, but an operator or a future ccxt config
    change could), exactly ONE POST must still reach the wire.
    """
    exchange, calls = _make_real_coinbase(monkeypatch)
    exchange.options["maxRetriesOnFailure"] = 3

    run_id = str(uuid4())  # 36 chars -> cid = 36 + 1 + 12 = 49 chars
    cid = f"{run_id}-{uuid4().hex[:12]}"
    assert len(cid) == 49

    engine = LiveExecutionEngine(
        run_id=run_id,
        risk_manager=_make_risk_manager_mock(),
        exchange=exchange,
        enable_live_trading=True,
    )

    order = Order(
        client_order_id=cid,
        run_id=run_id,
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
    )
    # Mirrors process_signal's WP14-S-01 sizing-price hint: a MARKET order
    # carries no price of its own (the Order model forbids it), but
    # Coinbase needs one to size a market BUY -- submit_order's Coinbase
    # branch reuses this hint instead of fetching a (stubbed-out-of-scope
    # here) ticker.
    engine._buy_sizing_price[order.order_id] = Decimal("50000")

    result = await engine.submit_order(order)

    post_calls = [c for c in calls if c["method"] == "POST"]
    # S-06: exactly 1 POST despite exchange.options["maxRetriesOnFailure"] == 3.
    assert len(post_calls) == 1
    body = json.loads(post_calls[0]["body"])

    # The real Coinbase adapter's own default ("ccxt-" + uuid) must have
    # been overridden by OUR cid -- not merely present alongside it.
    assert body["client_order_id"] == cid
    assert not body["client_order_id"].startswith("ccxt-")
    # S-06: fetch2 strips the retry-control key before signing -- it must
    # never appear in the signed body.
    assert "maxRetriesOnFailure" not in body

    # And the exchange's response round-trips our cid back out via the
    # real ``parse_order`` (unmodified ccxt code).
    assert result.exchange_order_id == "exch-real-99"

    await exchange.close()


@pytest.mark.asyncio
async def test_s06_retry_option_set_would_otherwise_cause_multiple_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-06 (negative control): WITHOUT our per-call override, a failing
    ``fetch`` under ``exchange.options["maxRetriesOnFailure"] = 3`` really
    would retry inside ccxt's own ``fetch2`` -- proving the fix in the
    positive test above is actually load-bearing, not just untriggered."""
    exchange = ccxt_async.coinbase({
        "enableRateLimit": False,
        "apiKey": "wp14b-test-key",
        "secret": "wp14b-test-secret",
    })
    exchange.markets = {_SYMBOL: dict(_MARKET)}
    exchange.options["brokerId"] = "ccxt"
    exchange.options["maxRetriesOnFailure"] = 3

    calls: list[dict[str, Any]] = []

    async def _always_fails(
        url: str, method: str = "GET", headers: Any = None, body: Any = None,
    ) -> dict[str, Any]:
        calls.append({"method": method})
        raise ccxt_async.ExchangeNotAvailable("simulated outage")

    monkeypatch.setattr(exchange, "fetch", _always_fails, raising=False)

    with pytest.raises(ccxt_async.ExchangeNotAvailable):
        # No params["maxRetriesOnFailure"] override here -- the raw ccxt
        # call, not through our engine.
        await exchange.create_order(_SYMBOL, "market", "buy", "0.01", "50000")

    # 1 initial attempt + 3 retries = 4 POSTs.
    assert len(calls) == 4

    await exchange.close()


def _iso(ms: int) -> str:
    """A ``created_time`` string ccxt's ``parse8601`` accepts, matching the
    real Coinbase response shape (``...Z`` suffix, microsecond precision)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _make_raw_order(
    *, exchange_id: str, ms: int, cid: str, status: str = "FILLED",
) -> dict[str, Any]:
    """One entry in a real Coinbase ``fetchOrders`` (``.../orders/historical/batch``)
    response envelope -- see ``coinbase.py``'s ``parse_order`` docstring for
    the exact shape this mirrors."""
    return {
        "order_id": exchange_id,
        "product_id": _MARKET["id"],
        "side": "BUY",
        "client_order_id": cid,
        "status": status,
        "order_configuration": {"market_market_ioc": {"base_size": "0.01"}},
        "created_time": _iso(ms),
        "filled_size": "0.01",
        "average_filled_price": "50000",
        "total_fees": "0.5",
    }


def _make_real_coinbase_with_orders(
    monkeypatch: pytest.MonkeyPatch, orders_response: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Same as ``_make_real_coinbase`` but the stubbed ``fetch`` answers a
    GET (``fetch_orders``) with a real Coinbase ``fetchOrders`` envelope
    carrying ``orders_response`` verbatim (``{"orders": ..., "sequence":
    "0", "has_next": False, "cursor": ""}}`` -- the empty ``"cursor"`` is
    what terminates ccxt's cursor-pagination loop after exactly one HTTP
    call), and a POST (``create_order``) with a ``RequestTimeout`` --
    simulating "accepted, then the response was lost" (D3's classic
    ambiguous case), forcing ``submit_order`` onto the D5 inline cid-lookup
    path (and, if that also times out inline, D8's per-bar resolver).
    """
    exchange = ccxt_async.coinbase({
        "enableRateLimit": False,
        "apiKey": "wp14b-test-key",
        "secret": "wp14b-test-secret",
    })
    exchange.markets = {_SYMBOL: dict(_MARKET)}
    exchange.options["brokerId"] = "ccxt"

    calls: list[dict[str, Any]] = []

    async def _fake_fetch(
        url: str, method: str = "GET", headers: Any = None, body: Any = None,
    ) -> dict[str, Any]:
        calls.append({"method": method, "url": url, "body": body})
        if method == "POST":
            raise ccxt_async.RequestTimeout("simulated: accepted, response lost")
        return {
            "orders": orders_response,
            "sequence": "0",
            "has_next": False,
            "cursor": "",
        }

    monkeypatch.setattr(exchange, "fetch", _fake_fetch, raising=False)
    return exchange, calls


@pytest.mark.asyncio
async def test_s_r2_01_live_engine_adopts_newest_order_beyond_ccxt_limit_100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.4b round 3 (S-R2-01): the real ccxt Coinbase ``fetch_orders``
    defaults ``limit`` to 100 and its paginated path ends with
    ``filter_by_since_limit`` -> ``filter_by_limit(..., fromStart=True)``,
    which keeps the OLDEST ``limit`` entries of an ascending-sorted result.
    With 150 orders inside the 1h lookup window and OUR order the single
    NEWEST of them, the buggy default would silently drop it -- exactly
    the "indistinguishable from never placed" failure mode security
    flagged. ``_lookup_by_cid`` now passes ``limit=None``, so the real
    (unmodified) ccxt filtering keeps every in-window entry and our order
    is found and adopted.

    This exercises the LIVE ENGINE side of the fix end-to-end: a real
    ``create_order`` POST that times out (D3 ambiguous), followed by the
    inline D5 ``_lookup_by_cid`` resolution -- both against the real ccxt
    Coinbase adapter, only ``Exchange.fetch`` stubbed.
    """
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    run_id = str(uuid4())
    cid = f"{run_id}-{uuid4().hex[:12]}"

    submit_at = datetime.now(tz=UTC)
    submit_ms = int(submit_at.timestamp() * 1000)
    since_ms = submit_ms - _LOOKUP_SINCE_MARGIN_MS

    # 150 filler orders, oldest-first, all safely inside the lookup
    # window's first few minutes -- under the buggy limit=100 default
    # these are exactly the 100 that WOULD be returned, burying our
    # (newest) order below the cutoff.
    fillers = [
        _make_raw_order(
            exchange_id=f"filler-exch-{i:04d}",
            ms=since_ms + 1_000 + i * 1_000,
            cid=f"{run_id}-other-{i:04d}",
        )
        for i in range(150)
    ]
    # Our own order: the single NEWEST entry in the window.
    target = _make_raw_order(
        exchange_id="target-exch-id", ms=submit_ms, cid=cid, status="OPEN",
    )

    exchange, calls = _make_real_coinbase_with_orders(
        monkeypatch, [*fillers, target],
    )

    engine = LiveExecutionEngine(
        run_id=run_id,
        risk_manager=_make_risk_manager_mock(),
        exchange=exchange,
        enable_live_trading=True,
    )

    order = Order(
        client_order_id=cid,
        run_id=run_id,
        symbol=_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.01"),
    )
    engine._buy_sizing_price[order.order_id] = Decimal("50000")

    result = await engine.submit_order(order)

    # Adopted -- not left as an unresolved "unknown submit".
    assert result.status in (OrderStatus.OPEN, OrderStatus.FILLED)
    assert result.exchange_order_id == "target-exch-id"
    assert order.order_id not in engine._unknown_submits

    post_calls = [c for c in calls if c["method"] == "POST"]
    # Exactly one POST reached the wire -- the timeout is never retried by
    # re-submitting; only the read-side cid lookup resolves it.
    assert len(post_calls) == 1

    await exchange.close()
