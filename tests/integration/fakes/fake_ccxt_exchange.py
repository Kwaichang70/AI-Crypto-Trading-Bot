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
  deducted in quote currency. No partial fills, no slippage — determinism
  over realism; realism is achieved instead by mirroring Coinbase's
  documented async settlement quirk (see below).
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
        # symbol -> list of [ts_ms, open, high, low, close, volume]
        self._closed_bars: dict[str, list[list[Any]]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._trades: list[dict[str, Any]] = []
        self._order_seq = 0
        self._queued_errors: dict[str, Exception] = {}
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
        """Set the total (== free; the fake never locks funds) balance for ``currency``."""
        self._balances[currency] = amount

    def balance_of(self, currency: str) -> Decimal:
        """Return the current balance for ``currency`` (test-side accessor)."""
        return self._balances.get(currency, Decimal("0"))

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
        total = {currency: float(amount) for currency, amount in self._balances.items()}
        return {"total": total, "free": dict(total), "used": dict.fromkeys(total, 0.0)}

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
        # Pad with an exact duplicate of the last closed bar so that
        # CCXTMarketDataService.get_latest_bar()'s ``bars[-2]`` convention
        # ("the last candle returned might still be forming, use the one
        # before it") always resolves to the most recently pushed *closed*
        # bar -- see the module docstring and the producer report §"Bar
        # padding convention" for the full derivation.
        exposed = [*bars, bars[-1]]
        n = limit if limit is not None else len(exposed)
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

        fee_cost = (notional * self._taker_fee_pct).quantize(_QTY_PRECISION)
        self._apply_fill(base=base, quote=quote, side=side, qty=qty, price=fill_price, fee=fee_cost)

        self._order_seq += 1
        exchange_order_id = f"fake-order-{self._order_seq}"
        ts_ms = self._now_ms(symbol)

        # Mirrors Coinbase: the initial response is still "open"/unfilled;
        # the caller (LiveExecutionEngine) reconciles via fetch_order after
        # its post-submit wait. fetch_order below always reports "closed".
        order_record = {
            "id": exchange_order_id,
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
        }
        self._orders[exchange_order_id] = order_record

        self._trades.append(
            {
                "order": exchange_order_id,
                "symbol": symbol,
                "side": side,
                "amount": float(qty),
                "price": float(fill_price),
                "fee": {"cost": float(fee_cost), "currency": quote},
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
        order = self._orders.get(id)
        if order is None:
            raise ccxt.OrderNotFound(str(id))
        # Every order in this fake settles synchronously inside create_order,
        # so by the time anything calls cancel_order the exchange-side order
        # is already filled -- exercise the same race ccxt reports for a
        # real already-filled order.
        raise ccxt.InvalidOrder(f"order {id} already filled, cannot cancel")

    async def fetch_my_trades(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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
    ) -> None:
        if side == "buy":
            cost = qty * price
            self._balances[quote] = self._balances.get(quote, Decimal("0")) - cost - fee
            self._balances[base] = self._balances.get(base, Decimal("0")) + qty
        else:
            proceeds = qty * price
            self._balances[quote] = self._balances.get(quote, Decimal("0")) + proceeds - fee
            self._balances[base] = self._balances.get(base, Decimal("0")) - qty
