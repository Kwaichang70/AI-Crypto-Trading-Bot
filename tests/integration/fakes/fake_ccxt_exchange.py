"""
tests/integration/fakes/fake_ccxt_exchange.py
-----------------------------------------------
In-memory fake of the CCXT async exchange surface used by
``trading.engines.live.LiveExecutionEngine`` and
``data.services.ccxt_market_data.CCXTMarketDataService``.

WP1.0 (Verbeterplan v2, Documentation/Verbeterplan-v2-2026-09.md §4 Fase 1)
------------------------------------------------------------------------
This fake exists so ``tests/integration/test_live_protective_paths.py`` can
drive the *real* ``LiveExecutionEngine`` + ``StrategyEngine`` + real
``CCXTMarketDataService`` against a deterministic, in-memory exchange
instead of a real one — without ever injecting ``engine._positions``
directly (that would hide bug C1 rather than prove it).

Scope: implements exactly the methods the production live path calls today
(grepped from ``packages/trading/engines/live.py``,
``packages/data/services/ccxt_market_data.py`` and
``apps/api/services/run_orchestrator.py::run_live_engine``):

    load_markets, markets, timeframes, has, id,
    fetch_balance, fetch_ticker, fetch_ohlcv,
    create_order, fetch_order, cancel_order, fetch_my_trades, close

Design notes
------------
- One ``FakeCCXTExchange`` instance is shared between the
  ``LiveExecutionEngine``'s own exchange handle and the
  ``CCXTMarketDataService``'s internal exchange handle (both are the *same*
  CCXT class in production — ``ccxt.async_support.<exchange_id>`` — so the
  test harness monkeypatches that class attribute to a factory that always
  returns this one instance; see ``tests/integration/fakes/live_harness.py``).
- Market orders always fill *immediately* and *fully* at the current
  scripted price (``push_bar``'s ``close``), with a configurable taker fee
  deducted in quote currency by default. No slippage — determinism over
  realism; realism is achieved instead by mirroring Coinbase's documented
  async settlement quirk (see below).
- To mirror the real Coinbase behaviour that ``LiveExecutionEngine`` already
  codes defensively around (``live.py`` docstring: "Coinbase processes
  market orders asynchronously"), ``create_order`` returns
  ``status="open", filled=None`` on the initial response and only reports
  ``status="closed"`` once ``fetch_order`` is polled (i.e. via
  ``LiveExecutionEngine._reconcile_order`` after its
  ``await asyncio.sleep(2)``). The harness neutralises that sleep via
  ``monkeypatch``; see the test module's ``_fast_sleep`` fixture.
- ``has["fetchOrderTrades"] = False`` mirrors real Coinbase capabilities, so
  ``LiveExecutionEngine.get_fills`` exercises its ``fetch_my_trades``
  fallback path exactly like production does.
- No randomness, no wall-clock reads anywhere in this module — every
  timestamp is derived from a synthetic bar cursor advanced only by
  ``push_bar`` / ``seed_flat_bars``.

WP1.1 (Verbeterplan v2 §4 row 1.1, R-23) extensions
----------------------------------------------------
Added for the live position-ledger fix's integration coverage, all
test-side configuration (not part of the real CCXT surface):

- ``set_fee_currency(symbol, "base")`` — charge the taker fee for that
  symbol's fills in the base asset instead of quote (D6/A-05's
  fee-normalisation path).
- ``lock_balance(currency, amount)`` — simulate funds locked in another
  open order: ``fetch_balance``'s ``free`` becomes ``total - locked``,
  proving the SELL cap uses ``free`` (I1/D9), not ``total``.
- ``queue_balance_error(exc)`` — the next ``fetch_balance`` call raises
  ``exc`` (fault injection for the "balance unavailable" paths, I1/I4).
- ``queue_partial_fill(symbol, first_fraction)`` — the next order for
  ``symbol`` reports a partial fill on its *first* ``fetch_order`` poll
  (``first_fraction`` of the requested amount) and completes fully on the
  second poll, exercising ``LiveExecutionEngine``'s partial-fill /
  idempotent-routing path (I9) against a real reconcile cycle.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import ccxt

__all__ = ["FakeCCXTExchange"]

# Candle duration in milliseconds per TimeFrame value. Mirrors
# ``data.services.ccxt_market_data._TIMEFRAME_DURATION_MS`` (kept as a
# separate literal here deliberately -- the fake must not import
# production internals, only its own scripted notion of "time").
_TIMEFRAME_DURATION_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}

# Fixed synthetic epoch for the first seeded bar (2026-01-01T00:00:00Z).
# Arbitrary but deterministic -- no wall-clock dependency anywhere.
_EPOCH_START_MS = 1_767_225_600_000

_QTY_PRECISION = Decimal("0.00000001")


class FakeCCXTExchange:
    """
    In-memory stand-in for a ``ccxt.async_support`` exchange instance.

    Parameters
    ----------
    exchange_id:
        Reported via ``.id``. Defaults to ``"coinbase"`` to exercise the
        Coinbase-specific branches in ``LiveExecutionEngine`` (market-buy
        price lookup, ``fetchOrderTrades`` fallback).
    taker_fee_pct:
        Fraction (e.g. ``Decimal("0.006")`` = 0.60%) deducted from every
        fill, matching the production Coinbase Advanced taker tier used
        elsewhere in the codebase (``trading.risk.RiskParameters.taker_fee_pct``).
    """

    def __init__(
        self,
        *,
        exchange_id: str = "coinbase",
        taker_fee_pct: Decimal = Decimal("0.006"),
    ) -> None:
        self.id = exchange_id
        self.has: dict[str, bool] = {"fetchOrderTrades": False}
        self.timeframes: dict[str, str] = dict.fromkeys(_TIMEFRAME_DURATION_MS, "")
        self.markets: dict[str, dict[str, Any]] = {}

        self._taker_fee_pct = taker_fee_pct
        self._market_defs: dict[str, dict[str, Any]] = {}
        self._balances: dict[str, Decimal] = {}
        # WP1.1 (R-23): currency -> amount locked in some *other* open
        # order, subtracted from `free` but not from `total`.
        self._locked: dict[str, Decimal] = {}
        # WP1.1 (R-23): symbol -> fee currency override ("base" or "quote";
        # default is "quote", matching the pre-WP1.1 unconditional behaviour).
        self._fee_currency_override: dict[str, str] = {}
        # WP1.4 (security round 2, WP14-S-02 probe PA): symbol -> a
        # fraction by which a market BUY's actual fill price exceeds the
        # ticker's `last` at fill time -- simulates slippage the fake's
        # otherwise-deterministic fill model doesn't produce on its own.
        self._fill_slippage_pct: dict[str, Decimal] = {}
        # symbol -> list of [ts_ms, open, high, low, close, volume]
        self._closed_bars: dict[str, list[list[Any]]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._trades: list[dict[str, Any]] = []
        self._order_seq = 0
        self._queued_errors: dict[str, Exception] = {}
        # WP1.1 (R-23): one-shot fault injection for the next fetch_balance().
        self._queued_balance_error: Exception | None = None
        # WP1.4: how many more fetch_balance() calls should still raise
        # ``_queued_balance_error`` -- defaults to 1 (queue_balance_error's
        # original WP1.1 single-shot contract); a caller that needs a
        # fault to survive ccxt_retry's automatic retries passes
        # ``times=`` to keep failing across every attempt.
        self._queued_balance_error_remaining: int = 0
        # WP1.1 (R-23): symbol -> fraction of the next order's amount to
        # report as filled on the *first* fetch_order poll only.
        self._queued_partial_fills: dict[str, Decimal] = {}
        # WP1.8b: one-shot fault injection for scan_and_import's three
        # exchange calls (fetch_orders / cancel_order / trade fetch).
        self._queued_fetch_orders_errors: dict[str, Exception] = {}
        self._queued_cancel_errors: dict[str, Exception] = {}
        self._queued_trades_errors: dict[str, Exception] = {}
        self._closed = False

        # Test-visible audit log of every create_order call, in call order.
        # Each entry: {"id", "symbol", "side", "amount" (Decimal), "price" (Decimal)}.
        self.order_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Test-side configuration API (NOT part of the CCXT surface)
    # ------------------------------------------------------------------

    def register_market(
        self,
        symbol: str,
        *,
        base: str,
        quote: str,
        min_amount: str = "0.0001",
        min_cost: str | None = "1",
        amount_precision: int = 8,
        price_precision: int = 2,
    ) -> None:
        """Register a tradeable pair, mirroring a CCXT ``market`` dict shape."""
        self._market_defs[symbol] = {
            "id": symbol.replace("/", "-"),
            "symbol": symbol,
            "base": base,
            "quote": quote,
            "active": True,
            "spot": True,
            "precision": {"amount": amount_precision, "price": price_precision},
            "limits": {
                "amount": {"min": float(min_amount), "max": None},
                "cost": {
                    "min": float(min_cost) if min_cost is not None else None,
                    "max": None,
                },
            },
        }

    def set_balance(self, currency: str, amount: Decimal) -> None:
        """Set the total balance for ``currency``.

        ``free`` mirrors ``total`` unless :meth:`lock_balance` has locked
        some of it away (R-23).
        """
        self._balances[currency] = amount

    def lock_balance(self, currency: str, amount: Decimal) -> None:
        """WP1.1 (R-23): simulate ``amount`` of ``currency`` locked in some
        *other* open order (e.g. a resting limit order).

        ``fetch_balance``'s ``free[currency]`` becomes
        ``total[currency] - amount`` (floored at 0); ``total`` is
        unaffected. Exercises the I1/D9 invariant that the SELL cap uses
        ``free``, never ``total``.
        """
        self._locked[currency] = amount

    def balance_of(self, currency: str) -> Decimal:
        """Return the current total balance for ``currency`` (test-side accessor)."""
        return self._balances.get(currency, Decimal("0"))

    def set_fill_slippage_pct(self, symbol: str, pct: Decimal) -> None:
        """WP1.4 (security round 2, probe PA): a market BUY for ``symbol``
        fills at ``last_price * (1 + pct)`` instead of exactly
        ``last_price`` -- proves the engine's affordability-cap slippage
        margin (``buy_cap_slippage_pct``) keeps run cash non-negative even
        when the exchange fills worse than the ticker quoted. BUY side
        only (a real book's slippage always works against the taker).
        """
        self._fill_slippage_pct[symbol] = pct

    def set_fee_currency(self, symbol: str, currency: str) -> None:
        """WP1.1 (R-23): charge the taker fee for ``symbol``'s fills in
        ``currency`` instead of the market's quote currency.

        ``currency`` is normally ``"base"`` or ``"quote"`` (case-sensitive
        market currency codes also work directly, e.g. ``"BTC"``).
        """
        self._fee_currency_override[symbol] = currency

    def queue_balance_error(self, exc: Exception, *, times: int = 1) -> None:
        """WP1.1 (R-23): make the *next* ``fetch_balance`` call raise ``exc``.

        WP1.4: ``times`` (default 1, the original contract) lets a caller
        make ``exc`` raise on each of the next ``times`` calls -- needed
        when the caller goes through ``ccxt_retry`` (which transparently
        retries a single transient failure), so the fault must survive
        every retry attempt to actually reach the engine's own error
        handling.
        """
        self._queued_balance_error = exc
        self._queued_balance_error_remaining = times

    def queue_partial_fill(self, symbol: str, first_fraction: Decimal) -> None:
        """WP1.1 (R-23): the next order created for ``symbol`` reports only
        ``first_fraction`` of its amount as filled on the first
        ``fetch_order`` poll, then completes fully on the second poll.
        """
        self._queued_partial_fills[symbol] = first_fraction

    @property
    def taker_fee_pct(self) -> Decimal:
        """The taker fee fraction applied to every fill (test-side accessor)."""
        return self._taker_fee_pct

    def seed_flat_bars(
        self,
        symbol: str,
        *,
        count: int,
        price: Decimal,
        timeframe: str = "1h",
    ) -> None:
        """Seed ``count`` flat (open == high == low == close) historical bars.

        Used to satisfy ``StrategyEngine._warmup_bar_windows()``'s
        ``fetch_ohlcv(limit=max(warmup_bars, 100))`` call before the
        scripted test bars begin.
        """
        tf_ms = _TIMEFRAME_DURATION_MS[timeframe]
        bars = self._closed_bars.setdefault(symbol, [])
        start = _EPOCH_START_MS if not bars else bars[-1][0] + tf_ms
        price_f = float(price)
        for i in range(count):
            bars.append([start + i * tf_ms, price_f, price_f, price_f, price_f, 1.0])

    def push_bar(self, symbol: str, close: Decimal, *, timeframe: str = "1h") -> None:
        """Advance ``symbol``'s price path by one closed candle.

        The new candle's open is the previous candle's close (a simple
        continuous walk); high/low are the min/max of open and close since
        every bracket/trailing check in production reads only ``bar.close``
        (see ``strategy_engine.py`` sections 5a/5b) — wicks are irrelevant
        to every scenario in this harness.
        """
        tf_ms = _TIMEFRAME_DURATION_MS[timeframe]
        bars = self._closed_bars.setdefault(symbol, [])
        if bars:
            ts = bars[-1][0] + tf_ms
            open_ = bars[-1][4]
        else:
            ts = _EPOCH_START_MS
            open_ = float(close)
        close_f = float(close)
        high = max(open_, close_f)
        low = min(open_, close_f)
        bars.append([ts, open_, high, low, close_f, 1.0])

    def queue_order_error(self, symbol: str, exc: Exception) -> None:
        """Make the *next* ``create_order`` call for ``symbol`` raise ``exc``.

        Supports future WPs that need to exercise min-notional / precision
        error handling (``ccxt.InvalidOrder`` and friends) without changing
        this fixture. Unused by the WP1.0 scenario table itself.
        """
        self._queued_errors[symbol] = exc

    # ------------------------------------------------------------------
    # WP1.8b: exchange-scan (scan_and_import) test-side extensions
    # ------------------------------------------------------------------

    def seed_exchange_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: str,
        amount: Decimal,
        price: Decimal,
        status: str = "open",
        filled: Decimal | None = None,
        timestamp_ms: int | None = None,
        trades: list[dict[str, Any]] | None = None,
    ) -> str:
        """Directly seed an exchange-side order that predates this
        ``FakeCCXTExchange`` session (WP1.8b) -- simulates an order placed
        by a now-dead engine instance before a crash, for
        ``scan_and_import`` tests.

        Bypasses ``create_order`` entirely: no balance mutation, no
        automatic trade generation. The caller supplies ``trades``
        explicitly for a filled/partially-filled order. Marked internally
        as "seeded/resting" so, unlike every order created through
        ``create_order`` (which this fake always settles synchronously --
        see the module docstring), :meth:`cancel_order` can genuinely
        cancel it and :meth:`fetch_order` reports its OWN current status
        (mutated in place by a later cancel), not an unconditional
        "closed".

        Returns
        -------
        str
            The synthetic exchange order id (``fetch_orders``/
            ``cancel_order``/``fetch_order`` all key off this).
        """
        self._order_seq += 1
        exchange_order_id = f"fake-seeded-{self._order_seq}"
        ts_ms = timestamp_ms if timestamp_ms is not None else self._now_ms(symbol)
        filled_amount = filled if filled is not None else (
            amount if status == "closed" else Decimal("0")
        )
        order_record: dict[str, Any] = {
            "id": exchange_order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": float(amount),
            "price": float(price),
            "average_fill_price": float(price),
            "status": status,
            "filled": float(filled_amount),
            "average": float(price) if filled_amount > Decimal("0") else None,
            "timestamp": ts_ms,
            "_partial_fraction": None,
            "_partial_polled_once": True,
            "_seeded_resting": True,
        }
        self._orders[exchange_order_id] = order_record
        for trade in trades or []:
            t = dict(trade)
            t.setdefault("order", exchange_order_id)
            t.setdefault("symbol", symbol)
            t.setdefault("side", side)
            t.setdefault("timestamp", ts_ms)
            t.setdefault("takerOrMaker", "taker")
            self._trades.append(t)
        return exchange_order_id

    def queue_fetch_orders_error(self, symbol: str, exc: Exception) -> None:
        """WP1.8b: make the *next* ``fetch_orders(symbol=...)`` call raise ``exc``."""
        self._queued_fetch_orders_errors[symbol] = exc

    def queue_cancel_error(self, exchange_order_id: str, exc: Exception) -> None:
        """WP1.8b: make the *next* ``cancel_order(exchange_order_id, ...)``
        call raise ``exc`` (distinct from the built-in "already filled"/
        "already terminal" races below)."""
        self._queued_cancel_errors[exchange_order_id] = exc

    def queue_trades_error(self, symbol: str, exc: Exception) -> None:
        """WP1.8b: make the *next* ``fetch_my_trades(symbol=...)`` (the
        fallback this fake always uses, ``has['fetchOrderTrades']`` is
        False) call raise ``exc``."""
        self._queued_trades_errors[symbol] = exc

    # ------------------------------------------------------------------
    # CCXT surface -- lifecycle
    # ------------------------------------------------------------------

    async def load_markets(self, reload: bool = False) -> dict[str, Any]:
        self.markets = dict(self._market_defs)
        return self.markets

    async def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    # CCXT surface -- account
    # ------------------------------------------------------------------

    async def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._queued_balance_error is not None:
            exc = self._queued_balance_error
            self._queued_balance_error_remaining -= 1
            if self._queued_balance_error_remaining <= 0:
                self._queued_balance_error = None
            raise exc

        total = {currency: float(amount) for currency, amount in self._balances.items()}
        free: dict[str, float] = {}
        used: dict[str, float] = {}
        for currency, amount in self._balances.items():
            locked = self._locked.get(currency, Decimal("0"))
            free_amount = amount - locked
            if free_amount < Decimal("0"):
                free_amount = Decimal("0")
            free[currency] = float(free_amount)
            used[currency] = float(locked)
        return {"total": total, "free": free, "used": used}

    # ------------------------------------------------------------------
    # CCXT surface -- market data
    # ------------------------------------------------------------------

    async def fetch_ticker(
        self, symbol: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"symbol": symbol, "last": float(self._last_price(symbol))}

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        bars = self._closed_bars.get(symbol, [])
        if not bars:
            return []

        n = limit if limit is not None else len(bars)

        if n >= len(bars):
            # WP10-C-05 fix: the caller wants (at least) the full seeded
            # history -- expose it unpadded so no real bar is silently
            # dropped. This is the ``_warmup_bar_windows()`` call pattern
            # (limit == the seeded count); the synthetic "still forming"
            # duplicate below is only meaningful for a small recent window.
            return [list(row) for row in (bars[-n:] if n > 0 else [])]

        # Pad with an exact duplicate of the last closed bar so that
        # CCXTMarketDataService.get_latest_bar()'s ``bars[-2]`` convention
        # ("the last candle returned might still be forming, use the one
        # before it") always resolves to the most recently pushed *closed*
        # bar -- see the module docstring and the producer report §"Bar
        # padding convention" for the full derivation. Only reached when
        # ``n < len(bars)``, i.e. there is at least one real bar to spare.
        exposed = [*bars, bars[-1]]
        tail = exposed[-n:] if n > 0 else []
        return [list(row) for row in tail]

    # ------------------------------------------------------------------
    # CCXT surface -- trading
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: str | float | Decimal,
        price: str | float | Decimal | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ccxt.ExchangeNotAvailable(f"{self.id} is closed")

        queued = self._queued_errors.pop(symbol, None)
        if queued is not None:
            raise queued

        market = self.markets.get(symbol) or self._market_defs.get(symbol)
        if market is None:
            raise ccxt.BadSymbol(f"{symbol} not listed on fake exchange {self.id!r}")

        qty = Decimal(str(amount)).quantize(_QTY_PRECISION)
        fill_price = self._last_price(symbol)
        if side == "buy":
            slippage = self._fill_slippage_pct.get(symbol)
            if slippage is not None:
                fill_price = (fill_price * (Decimal("1") + slippage)).quantize(
                    Decimal("0.00000001")
                )
        notional = qty * fill_price

        limits = market.get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min")
        min_cost = (limits.get("cost") or {}).get("min")
        if min_amount is not None and qty < Decimal(str(min_amount)):
            raise ccxt.InvalidOrder(
                f"{symbol} order amount {qty} below exchange minimum {min_amount}"
            )
        if min_cost is not None and notional < Decimal(str(min_cost)):
            raise ccxt.InvalidOrder(
                f"{symbol} order cost {notional} below exchange minimum {min_cost}"
            )

        base, quote = market["base"], market["quote"]
        if side == "sell":
            held = self._balances.get(base, Decimal("0"))
            if qty > held:
                raise ccxt.InsufficientFunds(
                    f"insufficient {base} balance: have {held}, need {qty}"
                )

        # WP1.1 (R-23): fee currency defaults to quote (pre-WP1.1 behaviour)
        # unless overridden per-symbol via set_fee_currency("base").
        fee_currency_choice = self._fee_currency_override.get(symbol, "quote")
        fee_currency = base if fee_currency_choice == "base" else quote
        if fee_currency == base:
            fee_cost = (qty * self._taker_fee_pct).quantize(_QTY_PRECISION)
        else:
            fee_cost = (notional * self._taker_fee_pct).quantize(_QTY_PRECISION)

        self._apply_fill(
            base=base,
            quote=quote,
            side=side,
            qty=qty,
            price=fill_price,
            fee=fee_cost,
            fee_currency=fee_currency,
        )

        self._order_seq += 1
        exchange_order_id = f"fake-order-{self._order_seq}"
        ts_ms = self._now_ms(symbol)

        # WP1.1 (R-23): a queued partial-fill fraction for this symbol
        # attaches to this order only; consumed (one-shot) here.
        partial_fraction = self._queued_partial_fills.pop(symbol, None)

        # WP1.8b: the real LiveExecutionEngine.submit_order passes the
        # client_order_id through params["clientOrderId"]; ccxt normalises
        # it onto the order dict's top-level "clientOrderId" key.
        client_order_id = (params or {}).get("clientOrderId")

        # Mirrors Coinbase: the initial response is still "open"/unfilled;
        # the caller (LiveExecutionEngine) reconciles via fetch_order after
        # its post-submit wait. fetch_order below always reports "closed"
        # unless a partial-fill fraction was queued for this order.
        order_record: dict[str, Any] = {
            "id": exchange_order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side,
            "type": type,
            "amount": float(qty),
            "price": float(price) if price is not None else float(fill_price),
            "average_fill_price": float(fill_price),
            "status": "open",
            "filled": None,
            "average": None,
            "timestamp": ts_ms,
            "_partial_fraction": (str(partial_fraction) if partial_fraction is not None else None),
            "_partial_polled_once": False,
        }
        self._orders[exchange_order_id] = order_record

        self._trades.append(
            {
                "order": exchange_order_id,
                "symbol": symbol,
                "side": side,
                "amount": float(qty),
                "price": float(fill_price),
                "fee": {"cost": float(fee_cost), "currency": fee_currency},
                "takerOrMaker": "taker",
                "timestamp": ts_ms,
            }
        )

        self.order_log.append(
            {
                "id": exchange_order_id,
                "symbol": symbol,
                "side": side,
                "amount": qty,
                "price": fill_price,
            }
        )

        return dict(order_record)

    async def fetch_order(
        self,
        id: str,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        order = self._orders.get(id)
        if order is None:
            raise ccxt.OrderNotFound(str(id))

        # WP1.1 (R-23): a partial-fill fraction was queued for this order --
        # report it as still-open/partially-filled on the FIRST poll only,
        # then fall through to the normal fully-closed response afterwards.
        partial_fraction_raw = order.get("_partial_fraction")
        if partial_fraction_raw is not None and not order.get("_partial_polled_once"):
            order["_partial_polled_once"] = True
            partial_fraction = Decimal(partial_fraction_raw)
            partial_amount = (Decimal(str(order["amount"])) * partial_fraction).quantize(
                _QTY_PRECISION
            )
            settled = dict(order)
            settled["status"] = "open"
            settled["filled"] = float(partial_amount)
            settled["average"] = order["average_fill_price"]
            return settled

        if order.get("_seeded_resting", False):
            # WP1.8b: a seeded exchange-side order reports its OWN current
            # status/filled (mutated in place by cancel_order below), not
            # the unconditional "closed" every create_order-originated
            # order short-circuits to below -- this fake never advances a
            # seeded order to "closed" on its own; only a test / the scan
            # cancel path changes its status.
            return dict(order)

        settled = dict(order)
        settled["status"] = "closed"
        settled["filled"] = order["amount"]
        settled["average"] = order["average_fill_price"]
        return settled

    async def cancel_order(
        self,
        id: str,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        queued = self._queued_cancel_errors.pop(id, None)
        if queued is not None:
            raise queued

        order = self._orders.get(id)
        if order is None:
            raise ccxt.OrderNotFound(str(id))

        if not order.get("_seeded_resting", False):
            # Every order created via create_order settles synchronously in
            # this fake, so by the time anything calls cancel_order the
            # exchange-side order is already filled -- exercise the same
            # race ccxt reports for a real already-filled order.
            raise ccxt.InvalidOrder(f"order {id} already filled, cannot cancel")

        if order["status"] in ("closed", "canceled", "expired", "rejected"):
            # WP1.8b (P-01): the scan/test raced an already-cancelled or
            # already-filled seeded order -- mirrors a real exchange's
            # InvalidOrder/OrderNotFound for a second cancel attempt.
            raise ccxt.InvalidOrder(f"order {id} already {order['status']}, cannot cancel")

        order["status"] = "canceled"
        return dict(order)

    async def fetch_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """WP1.8b: every order (created via create_order OR seeded via
        seed_exchange_order) for ``symbol``, since ``since`` (ms epoch)."""
        if symbol is not None:
            queued = self._queued_fetch_orders_errors.pop(symbol, None)
            if queued is not None:
                raise queued
        results: list[dict[str, Any]] = []
        for order in self._orders.values():
            if symbol is not None and order["symbol"] != symbol:
                continue
            if since is not None and order["timestamp"] < since:
                continue
            results.append(dict(order))
        return results

    async def fetch_my_trades(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if symbol is not None:
            queued = self._queued_trades_errors.pop(symbol, None)
            if queued is not None:
                raise queued
        if symbol is None:
            return list(self._trades)
        return [t for t in self._trades if t["symbol"] == symbol]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_price(self, symbol: str) -> Decimal:
        bars = self._closed_bars.get(symbol)
        if not bars:
            raise ccxt.BadSymbol(f"no price history seeded for {symbol!r} yet")
        return Decimal(str(bars[-1][4]))

    def _now_ms(self, symbol: str) -> int:
        bars = self._closed_bars.get(symbol)
        return bars[-1][0] if bars else _EPOCH_START_MS

    def _apply_fill(
        self,
        *,
        base: str,
        quote: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        fee: Decimal,
        fee_currency: str,
    ) -> None:
        if side == "buy":
            cost = qty * price
            if fee_currency == base:
                self._balances[quote] = self._balances.get(quote, Decimal("0")) - cost
                self._balances[base] = self._balances.get(base, Decimal("0")) + qty - fee
            else:
                self._balances[quote] = self._balances.get(quote, Decimal("0")) - cost - fee
                self._balances[base] = self._balances.get(base, Decimal("0")) + qty
        else:
            proceeds = qty * price
            if fee_currency == base:
                self._balances[quote] = self._balances.get(quote, Decimal("0")) + proceeds
                self._balances[base] = self._balances.get(base, Decimal("0")) - qty - fee
            else:
                self._balances[quote] = self._balances.get(quote, Decimal("0")) + proceeds - fee
                self._balances[base] = self._balances.get(base, Decimal("0")) - qty
