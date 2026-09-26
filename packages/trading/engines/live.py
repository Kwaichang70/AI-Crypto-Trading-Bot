"""
packages/trading/engines/live.py
----------------------------------
Live execution engine using CCXT for real exchange order placement.

Safety contract
---------------
- ``enable_live_trading`` must be explicitly True at construction time.
  Any attempt to submit an order when the gate is False raises RuntimeError.
- All orders pass through the RiskManager pre-trade check before submission.
- Exchange API calls are wrapped with error handling and logging.
- The kill-switch on the RiskManager can halt all trading at any time.

Error recovery (Sprint 37)
--------------------------
CCXT API calls use ``ccxt_retry`` for exponential backoff with jitter.
Balance fetching uses a 10-second TTL cache to reduce rate-limit pressure.
Cancel operations handle the "already filled" race condition gracefully.
The order state machine, signal processing, and risk-check flow are fully
functional.

Live position ledger (Verbeterplan v2 WP1.1, C1/C22)
------------------------------------------------------
``PortfolioAccounting`` is the single source of truth for *how much of a
symbol the bot itself owns* (Option B, ``reports/vp2-wp1.1/arch-design.md``
WP11-A-01). ``StrategyEngine`` attaches it via :meth:`attach_position_source`
right after construction (duck-typed; a no-op for the paper engine, which
has no such method). ``self._positions`` becomes a read-through *cache* of
the attached source, refreshed on every ``process_signal`` call and by
``sync_positions()`` -- it is never written to except from that refresh path
once a source is attached.

With no source attached (most unit tests that never call ``on_start()``),
the legacy behaviour is preserved: callers may inject
``engine._positions[...]`` directly (8 such unit-test injection sites
survive this WP unchanged) and BUY/SELL/daily-PnL all read/write that dict
exactly as before. **Live mode itself fails closed**: ``on_start()`` raises
``RuntimeError`` if live trading is enabled and no position source is
attached (WP1.1 round 2, S-09) -- the legacy path is for tests that never
reach ``on_start()``, not for a real live run.

``reconcile_required`` (I4) is a per-symbol flag that blocks BUYs only --
SELLs (strategy, bracket, trailing) are always allowed, capped at
``floor_to_amount_precision(min(own_avail, free))`` (I1/D15) -- a flagged
symbol must never leave a protective exit unable to fire. Only the
``balance_unavailable`` reason clears itself, on the next successful sync;
every other reason waits for an operator or WP1.8's fill-history rebuild.

Round 2 hardening (security review WP11-S-01..09)
----------------------------------------------------
- **S-01**: ``get_fills``'s "no exchange mapping" early exit and every
  non-synthesising branch of ``_synthesize_unrouted_fill`` now return
  ``[]``, never the order's cached fill history -- returning history there
  caused a PARTIAL/CANCELED order's already-routed fill to be handed back
  to the caller again, double-counting ``own`` quantity.
- **S-02**: a new ``_routed_gross_qty`` per-order-id ledger tracks the raw
  (pre-fee-normalisation) trade amount for every trade marked routed
  (including trades whose *net* quantity is <= 0 after a base-currency fee)
  plus any synthesised quantity. ``check_resting_orders`` and
  ``_synthesize_unrouted_fill`` compare against this gross figure, not
  against ``sum(Fill.quantity)`` (net) -- a base-currency fee no longer
  makes an order look permanently under-routed.
- **S-03**: ``_held_quantity`` now also subtracts this-symbol SELL
  quantity still in flight (submitted but not yet routed) from ``own``
  before capping/flagging, so a second SELL signal on the same bar/next
  bar cannot re-sell the same (already-in-flight) quantity out of an
  external holding.
- **S-04**: ``get_fills`` parses a whole batch of trades into a local
  buffer and only commits to ``_routed_trade_keys``/``_fills``/
  ``_routed_gross_qty`` after the batch finishes, so one trade's parse
  error can no longer silently drop fills already parsed earlier in the
  same call; a parse error flags ``fill_parse_failed`` and leaves that one
  trade unrouted (retried next call), same treatment as I3.
- **S-05**: ``precisionMode`` is read from the *exchange* object (where
  real ccxt places it) with the market dict as an override, not the other
  way around.
- **S-07**: balance values are parsed with ``_safe_decimal``; an
  unparseable balance is treated as unavailable, never as zero.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Protocol, overload
from uuid import UUID, uuid4

import ccxt.async_support as ccxt_async
import structlog

from common.types import OrderSide, OrderStatus, OrderType, SignalDirection
from trading.execution import BaseExecutionEngine, EquitySnapshot
from trading.ccxt_errors import translate_ccxt_error
from trading.ccxt_retry import ccxt_retry
from trading.models import Fill, Order, Position, Signal
from trading.risk import BaseRiskManager

__all__ = ["LiveExecutionEngine", "LivePositionSource", "_parse_ccxt_trades"]

logger = structlog.get_logger(__name__)

_PRICE_PRECISION = Decimal("0.00000001")
_QTY_PRECISION = Decimal("0.00000001")

# Terminal order states whose ``filled_quantity`` may still have unrouted
# fills (a synthesised partial that later gains a second real trade, or a
# CANCELED order that carries a partial fill). Used by check_resting_orders
# (D3/S-01) to decide which terminal orders still need a get_fills() poll.
_TERMINAL_ORDER_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)

# Order states that still have SELL quantity potentially in flight for the
# WP11-S-03 "own_avail" computation.
_IN_FLIGHT_ORDER_STATUSES = frozenset(
    {OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN, OrderStatus.PARTIAL}
)
_SETTLED_ORDER_STATUSES = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED}
)

# WP11-S-R2-02: an OPEN/PARTIAL SELL whose state hasn't been confirmed by a
# successful reconcile in longer than this is no longer trusted enough to
# keep reserving its quantity against future SELLs (see
# LiveExecutionEngine._pending_sell_quantity).
_INFLIGHT_SELL_MAX_AGE_S = 300


@overload
def _safe_decimal(value: Any, default: Decimal = ...) -> Decimal: ...
@overload
def _safe_decimal(value: Any, default: None = ...) -> Decimal | None: ...
def _safe_decimal(value: Any, default: Decimal | None = Decimal("0")) -> Decimal | None:
    """Convert a CCXT response value to Decimal, handling None and non-numeric.

    Coinbase returns None for fields like 'filled' and 'average' on market
    orders that are still processing asynchronously.

    WP1.1 round 2 (S-07): overloaded so ``default=None`` callers (balance
    parsing, which must distinguish "genuinely zero" from "unparseable/
    unavailable") get an ``Decimal | None`` return type under mypy, while
    every pre-existing ``Decimal``-defaulted call site keeps its original
    ``Decimal`` return type unchanged.
    """
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _is_integral(value: int | float) -> bool:
    """True if ``value`` represents a whole number (CCXT DECIMAL_PLACES
    precision is reported as an int number of decimals; TICK_SIZE precision
    is reported as the (usually fractional) step itself)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return False


# ---------------------------------------------------------------------------
# WP1.8b: module-level trade-parsing helpers, extracted from
# LiveExecutionEngine.get_fills's per-trade loop (and the two smaller
# helpers it used, ``_extract_fee_from_ccxt``/``_normalize_fee``) so
# ``apps.api.services.run_recovery.scan_and_import``'s exchange-scan
# import path normalises a batch of raw CCXT trade dicts IDENTICALLY to
# the live-engine reconcile path -- same idempotency key, same
# fee-currency normalisation (D6), same fail-soft per-trade behaviour
# (WP11-S-04). Pure functions: no ``self``, no I/O, no logging side
# effects beyond the ``log``/``on_skip`` callbacks the caller supplies.
# The three ``LiveExecutionEngine`` instance methods of (almost) the same
# name are now thin wrappers that resolve ``self._base_asset``/
# ``self._quote_currency``/``self._log`` and delegate here -- kept as
# methods (not removed) because existing unit tests call
# ``engine._extract_fee_from_ccxt(...)`` directly.
# ---------------------------------------------------------------------------


def _trade_key(trade: dict[str, Any]) -> tuple[Any, ...]:
    """I9: idempotency key for a CCXT trade record -- ``(id, trade id)``
    when the exchange provides a trade id, else
    ``(synthetic, timestamp, amount, price)``. See
    ``LiveExecutionEngine._trade_key``'s original docstring (WP11-S-08)
    for the documented same-order-without-then-with-id limitation this
    key inherits.
    """
    trade_id = trade.get("id")
    if trade_id:
        return ("id", trade_id)
    return ("synthetic", trade.get("timestamp"), trade.get("amount"), trade.get("price"))


def _extract_fee_from_ccxt_trade(
    ccxt_trade: dict[str, Any],
    *,
    symbol: str | None,
    quote_currency: str,
    log: Any = None,  # noqa: ANN401
) -> tuple[Decimal, str]:
    """Extract ``(fee_amount, fee_currency)`` from a single CCXT trade dict.

    ``quote_currency`` is the caller-resolved market quote (D6 default
    when the trade itself carries no fee currency); ``symbol``/``log`` are
    used only for the negative-fee (maker rebate) warning's structured
    fields and are optional so a caller with no logger (e.g. a one-shot
    import script) can pass ``log=None``.
    """
    fee_info = ccxt_trade.get("fee") or {}
    fee_cost = _safe_decimal(fee_info.get("cost"), default=Decimal("0"))
    if fee_cost < Decimal("0"):
        # WP11-S-R2-03: a negative fee (a maker rebate) would fail
        # Fill.fee's ge=0 constraint later -- clamp it to 0.
        if log is not None:
            log.warning("live.fee_rebate_ignored", symbol=symbol)
        fee_cost = Decimal("0")
    fee_currency = fee_info.get("currency")
    if not fee_currency:
        fee_currency = quote_currency
    return fee_cost, str(fee_currency)


def _normalize_trade_fee(
    *,
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    fee_amount: Decimal,
    fee_currency: str,
    base_asset: str | None,
    quote_currency: str,
    log: Any = None,  # noqa: ANN401
) -> tuple[Decimal, Decimal, str]:
    """WP11-A-05 (D6): normalise a CCXT trade's fee into quote currency.

    Identical logic to the pre-WP1.8b ``LiveExecutionEngine._normalize_fee``
    method, with ``base_asset``/``quote_currency`` passed in explicitly
    instead of resolved via ``self._base_asset``/``self._quote_currency``
    (a caller with no live exchange market cache -- e.g. WP1.8b's
    ``scan_and_import``, which loads markets on its OWN throwaway exchange
    handle -- can still call this).
    """
    if fee_amount <= Decimal("0") or base_asset is None or fee_currency != base_asset:
        if (
            fee_amount > Decimal("0")
            and fee_currency not in (base_asset, quote_currency)
            and log is not None
        ):
            log.warning(
                "live.fee_currency_unconverted",
                symbol=symbol,
                fee_currency=fee_currency,
            )
        return quantity, fee_amount, fee_currency

    fee_in_quote = (fee_amount * price).quantize(_QTY_PRECISION, rounding=ROUND_HALF_UP)
    net_quantity = quantity - fee_amount if side == OrderSide.BUY else quantity
    if log is not None:
        log.info(
            "live.fee_normalized_from_base",
            symbol=symbol,
            side=side.value,
            fee_base=str(fee_amount),
            fee_quote=str(fee_in_quote),
        )
    return net_quantity, fee_in_quote, quote_currency


def _parse_ccxt_trades(
    order: Order,
    trades: Sequence[dict[str, Any]],
    *,
    already_routed: Collection[tuple[Any, ...]] = (),
    base_asset: str | None,
    quote_currency: str,
    log: Any = None,  # noqa: ANN401
    on_skip: Callable[[str], None] | None = None,
) -> tuple[list[Fill], Decimal, set[tuple[Any, ...]]]:
    """Shared trade-parse helper (WP1.8b S2/S3).

    Turns a batch of raw CCXT trade dicts belonging to ``order`` into
    ``Fill`` objects, applying the exact same idempotency-key dedup
    (:func:`_trade_key`), fee-currency normalisation
    (:func:`_normalize_trade_fee`/D6) and fail-soft per-trade handling
    (WP11-S-04: one bad trade record flags/skips only itself; every fill
    parsed earlier in the same batch is still returned) that
    ``LiveExecutionEngine.get_fills`` used inline before WP1.8b. Used by
    both ``get_fills`` (the live reconcile path) and
    ``apps.api.services.run_recovery.scan_and_import`` (the WP1.8b
    exchange-scan import path) so a resumed run's imported fills
    normalise identically to one that was routed live.

    Parameters
    ----------
    order:
        The parent order every trade in ``trades`` belongs to.
    trades:
        Raw CCXT trade dicts (``fetch_order_trades``/``fetch_my_trades``).
    already_routed:
        Idempotency keys already turned into a ``Fill`` by a prior call --
        skipped again here. Empty (the default) for a one-shot caller.
    base_asset, quote_currency:
        The order's market base/quote currency codes, resolved by the
        caller (this function has no exchange handle of its own).
    log:
        Optional bound structlog logger for warnings (never raises).
    on_skip:
        Optional callback invoked with a reason code
        (``"fill_parse_failed"`` or ``"fill_price_invalid"``) for every
        trade this call could not turn into a ``Fill``. ``get_fills``
        uses this to call its own ``_flag_reconcile``.

    Returns
    -------
    tuple[list[Fill], Decimal, set[tuple[Any, ...]]]
        ``(new_fills, gross_quantity_delta, newly_routed_keys)`` --
        ``gross_quantity_delta`` is the RAW (pre fee-normalisation) trade
        amount summed across every still-new trade in this batch (matches
        ``order.filled_quantity``'s unit, S-02); ``newly_routed_keys`` is
        the set of idempotency keys this call consumed.
    """
    pending_new_keys: set[tuple[Any, ...]] = set()
    pending_fills: list[Fill] = []
    pending_gross = Decimal("0")

    for trade in trades:
        key = _trade_key(trade)
        if key in already_routed or key in pending_new_keys:
            continue

        price = _safe_decimal(trade.get("price"), default=None)
        if price is None:
            if on_skip is not None:
                on_skip("fill_parse_failed")
            if log is not None:
                log.error(
                    "live.fill_parse_failed",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    field="price",
                )
            continue
        if price <= Decimal("0"):
            # I3: a fill whose price is non-positive is a legitimate
            # invalid-price business case (not a parse error) -- not
            # routed, left pending for an operator/WP1.8 to resolve.
            if on_skip is not None:
                on_skip("fill_price_invalid")
            if log is not None:
                log.error(
                    "live.invalid_fill_price",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                )
            continue

        raw_quantity = _safe_decimal(trade.get("amount"), default=None)
        if raw_quantity is None:
            if on_skip is not None:
                on_skip("fill_parse_failed")
            if log is not None:
                log.error(
                    "live.fill_parse_failed",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    field="amount",
                )
            continue

        fill: Fill | None = None
        try:
            fee_amount, fee_currency = _extract_fee_from_ccxt_trade(
                trade, symbol=order.symbol, quote_currency=quote_currency, log=log
            )
            timestamp_raw = trade.get("timestamp")
            if timestamp_raw is None:
                timestamp_ms: float = order.updated_at.timestamp() * 1000
            else:
                timestamp_ms = float(timestamp_raw)
            executed_at = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

            quantity, fee_amount, fee_currency = _normalize_trade_fee(
                symbol=order.symbol,
                side=order.side,
                quantity=raw_quantity,
                price=price,
                fee_amount=fee_amount,
                fee_currency=fee_currency,
                base_asset=base_asset,
                quote_currency=quote_currency,
                log=log,
            )

            if quantity > Decimal("0"):
                fill = Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=quantity,
                    price=price,
                    fee=fee_amount,
                    fee_currency=fee_currency,
                    is_maker=trade.get("takerOrMaker") == "maker",
                    executed_at=executed_at,
                )
        except Exception as exc:
            if on_skip is not None:
                on_skip("fill_parse_failed")
            if log is not None:
                log.error(
                    "live.fill_parse_failed",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    error=str(exc),
                )
            continue

        # This trade is now accounted for either way (Fill produced or
        # net-zero skip above) -- mark it routed and count its GROSS
        # amount (S-02) so it is never reprocessed.
        pending_new_keys.add(key)
        pending_gross += raw_quantity

        if fill is not None:
            pending_fills.append(fill)

    return pending_fills, pending_gross, pending_new_keys


class LivePositionSource(Protocol):
    """Duck-typed interface the live engine reads its own held quantity,
    open positions, daily PnL, run NAV/cash and peak from (WP11-A-02,
    extended WP1.4/WP14-A-01).

    ``PortfolioAccounting`` already implements this surface exactly;
    nothing new needs to be built there -- ``StrategyEngine`` just attaches
    it via :meth:`LiveExecutionEngine.attach_position_source`.
    """

    def get_position(self, symbol: str) -> Position | None: ...

    def get_open_positions(self) -> list[Position]: ...

    def get_daily_pnl(self) -> Decimal: ...

    # WP1.4 (D5/I-1): run NAV, cash and peak all come from here -- never
    # the exchange balance. ``PortfolioAccounting`` implements all four.
    @property
    def cash(self) -> Decimal: ...

    @property
    def initial_cash(self) -> Decimal: ...

    @property
    def current_equity(self) -> Decimal: ...

    def get_peak_equity(self) -> Decimal: ...


class LiveExecutionEngine(BaseExecutionEngine):
    """
    Live execution engine that places real orders via CCXT.

    This engine is the production-path for real capital deployment.
    It requires an explicit enable gate and a CCXT async exchange instance.

    Parameters
    ----------
    run_id:
        Unique identifier for the trading run.
    risk_manager:
        Injected risk manager for pre-trade checks and position sizing.
    exchange:
        A CCXT async exchange instance (e.g. ccxt.pro.binance()).
        Must already be configured with API credentials.
    enable_live_trading:
        Explicit gate. If False (default), any call to submit_order
        raises RuntimeError. This is the outermost safety gate.
    """

    def __init__(
        self,
        run_id: str,
        risk_manager: BaseRiskManager,
        exchange: Any,  # ccxt.async_support.Exchange — typed as Any to avoid hard import
        *,
        enable_live_trading: bool = False,
        config: dict[str, Any] | None = None,
        buy_cap_slippage_pct: Decimal = Decimal("0.005"),
        buy_inflight_stale_after_s: float = 900.0,
    ) -> None:
        super().__init__(run_id=run_id, config=config)
        self._risk_manager = risk_manager
        self._exchange = exchange
        self._enable_live_trading = enable_live_trading

        # WP1.4 (S-02, security round 2): additional margin folded into the
        # BUY affordability cap's buf, on top of the market's own taker fee
        # -- covers a fill above last/ask and a misreported taker fee
        # (amends I-4: cap price is max(last, ask), buf = taker + this).
        self._buy_cap_slippage_pct = buy_cap_slippage_pct
        # WP1.4 (S-04, security round 2): how long a BUY may keep blocking
        # every other BUY before an operator-facing stale alert fires
        # (once, at error level). Does not expire the block itself (S-02:
        # "no age expiry" still holds) -- it only makes a stuck block
        # impossible to miss in the logs.
        self._buy_inflight_stale_after_s = buy_inflight_stale_after_s

        # Fill registry: order_id -> list of Fill objects
        self._fills: dict[UUID, list[Fill]] = {}

        # Position tracking: symbol -> Position. Read/write cache when no
        # LivePositionSource is attached (legacy, D8); read-through cache of
        # the attached source otherwise (WP11-A-02).
        self._positions: dict[str, Position] = {}

        # WP1.1: the attached position source (PortfolioAccounting in
        # production) and the run's own symbols, set by attach_position_source.
        self._position_source: LivePositionSource | None = None
        self._run_symbols: tuple[str, ...] = ()

        # WP1.1 (I4): per-symbol reconcile flag -> reason code. Blocks BUYs
        # only; never blocks a SELL/exit.
        self._reconcile_required: dict[str, str] = {}

        # WP1.4 (S-04): run-wide BUY block reason, set once at on_start
        # (e.g. "quote_mismatch") and never self-healed mid-run (unlike
        # _reconcile_required's per-symbol, sometimes-self-clearing flags).
        self._run_buy_block: str | None = None

        # WP1.4 (S-01, security round 2): order_id -> the price
        # process_signal's affordability cap sized the BUY against.
        # submit_order's Coinbase market-BUY path pops this instead of
        # fetching a SECOND, later ticker -- a rising price between the
        # two fetches let a BUY spend more than the cap allowed.
        self._buy_sizing_price: dict[UUID, Decimal] = {}

        # WP1.4 (S-04, security round 2): order_id -> the moment this
        # engine first observed the order blocking new BUYs, and the set
        # of order_ids already alerted on (so the stale alert fires
        # exactly once per stuck order).
        self._inflight_block_started_at: dict[UUID, datetime] = {}
        self._inflight_block_stale_alerted: set[UUID] = set()

        # WP1.1 (I9): idempotent fill routing. order_id -> set of trade keys
        # already turned into a routed Fill (or explicitly skipped as a
        # zero-net trade -- see _routed_gross_qty).
        self._routed_trade_keys: dict[UUID, set[tuple[Any, ...]]] = {}
        # WP1.1 round 2 (S-02/S-03): order_id -> cumulative GROSS trade
        # quantity accounted for so far (raw exchange amount, before any
        # fee-currency normalisation -- the same unit as
        # ``Order.filled_quantity``). Includes synthesised quantities.
        # Used for "is this order fully routed yet" (check_resting_orders,
        # _synthesize_unrouted_fill) and for "how much of this SELL is
        # still in flight" (_pending_sell_quantity) -- both need the GROSS
        # figure; ``sum(Fill.quantity)`` (net of fees) is the wrong
        # denominator and was the root cause of S-02.
        self._routed_gross_qty: dict[UUID, Decimal] = {}
        # WP1.1 (R-12/§3 scope): orders for which a synthetic fill already
        # accounts for the full filled_quantity -- later trade records for
        # these orders are permanently ignored so they cannot double-count.
        self._synthesized_orders: set[UUID] = set()

        # Peak equity tracker for drawdown calculation
        self._peak_equity: Decimal = Decimal("0")

        # Balance cache: (result, monotonic timestamp) — 10s TTL
        self._balance_cache: dict[str, Any] | None = None
        self._balance_cache_time: float = 0.0

        # Map internal order_id -> exchange order ID for reconciliation
        self._exchange_order_map: dict[UUID, str] = {}

        # Reverse map: exchange order ID -> internal order_id
        self._reverse_order_map: dict[str, UUID] = {}

        self._log = self._log.bind(
            engine="live",
            exchange=getattr(exchange, "id", "unknown"),
            live_trading_enabled=enable_live_trading,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def risk_manager(self) -> BaseRiskManager:
        return self._risk_manager

    @property
    def exchange(self) -> Any:
        """Return the underlying CCXT exchange instance."""
        return self._exchange

    @property
    def is_live_enabled(self) -> bool:
        return self._enable_live_trading

    @property
    def positions(self) -> dict[str, Position]:
        """Return a copy of current positions."""
        return dict(self._positions)

    @property
    def reconcile_required(self) -> Mapping[str, str]:
        """Per-symbol reconcile-required reason codes (I4). Blocks BUYs
        only; SELLs are never blocked. Empty when nothing is flagged."""
        return dict(self._reconcile_required)

    # ------------------------------------------------------------------
    # WP1.1: position source wiring (WP11-A-02)
    # ------------------------------------------------------------------

    def attach_position_source(
        self, source: LivePositionSource, *, symbols: Sequence[str]
    ) -> None:
        """Attach the run's ``PortfolioAccounting`` as the position source.

        Called by ``StrategyEngine`` right after construction (duck-typed;
        paper engines have no such method so this is a no-op there). Once
        attached, ``own`` quantity for every held-quantity/mismatch/daily-PnL
        computation comes from ``source`` -- ``self._positions`` becomes a
        read-through cache refreshed from it (D1/D8).
        """
        self._position_source = source
        self._run_symbols = tuple(symbols)
        self._log.info(
            "live.position_source_attached",
            symbols=list(self._run_symbols),
        )

    def _own_quantity(self, symbol: str) -> Decimal:
        """Own quantity for ``symbol`` (I2): from the attached source, or
        the legacy ``_positions`` cache when none is attached (D8)."""
        if self._position_source is not None:
            position = self._position_source.get_position(symbol)
            return position.quantity if position is not None else Decimal("0")
        position = self._positions.get(symbol)
        return position.quantity if position is not None else Decimal("0")

    def _sync_positions_cache(self, symbol: str) -> None:
        """Refresh the legacy ``_positions[symbol]`` cache from the attached
        source so external readers of ``.positions`` see current data, even
        though the source is the single source of truth (D1)."""
        if self._position_source is None:
            return
        position = self._position_source.get_position(symbol)
        if position is not None:
            self._positions[symbol] = position
        else:
            self._positions.pop(symbol, None)

    def _flag_reconcile(self, symbol: str, reason: str) -> None:
        """Set ``reconcile_required[symbol] = reason`` and log at error
        level (I4). Idempotent to call repeatedly; only ``sync_positions``'s
        ``balance_unavailable`` clearing is special-cased -- every other
        reason persists until an operator or WP1.8 clears it."""
        self._reconcile_required[symbol] = reason
        self._log.error("live.reconcile_required", symbol=symbol, reason=reason)

    def _maybe_clear_balance_unavailable(self, symbol: str) -> None:
        """I4: ``balance_unavailable`` is the one reason that clears itself,
        on the next successful balance fetch. Every other reason is left
        untouched here."""
        if self._reconcile_required.get(symbol) == "balance_unavailable":
            del self._reconcile_required[symbol]
            self._log.info(
                "live.reconcile_cleared", symbol=symbol, reason="balance_unavailable"
            )

    # ------------------------------------------------------------------
    # WP1.1: precision helpers (D5)
    # ------------------------------------------------------------------

    def _amount_step(self, symbol: str) -> Decimal:
        """Return the market's minimum amount increment (D5).

        DECIMAL_PLACES markets report an int number of decimals in
        ``market["precision"]["amount"]``; TICK_SIZE markets
        (``precisionMode == 4``) report the step itself, which is generally
        not a whole number. Falls back to 8 decimals when the market or its
        precision is unknown.

        WP1.1 round 2 (S-05): real ccxt (4.5.40) reports ``precisionMode``
        on the *exchange* object (a market-wide, not per-market, setting for
        the exchanges this system targets) -- the market dict is consulted
        first only so a test or an unusual exchange can override it
        per-market, but the exchange-level value is the realistic default.
        """
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(symbol) or {}
        precision = (market.get("precision") or {}).get("amount")
        if precision is None:
            return _QTY_PRECISION
        exchange_precision_mode = getattr(self._exchange, "precisionMode", None)
        precision_mode = market.get("precisionMode", exchange_precision_mode)
        if precision_mode == 4 or not _is_integral(precision):
            step = Decimal(str(precision))
            return step if step > Decimal("0") else _QTY_PRECISION
        return Decimal(10) ** (-int(precision))

    def _floor_to_amount_precision(self, symbol: str, qty: Decimal) -> Decimal:
        """D5: floor ``qty`` down to the market's amount precision.

        One helper for both SELL and BUY (pure Decimal arithmetic, no
        float round-trip): SELL relies on the floor to never oversell;
        BUY relies on it to never submit a size finer than the exchange
        accepts.
        """
        if qty <= Decimal("0"):
            return Decimal("0")
        step = self._amount_step(symbol)
        if step <= Decimal("0"):
            return qty
        return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step

    def _amount_tolerance(self, symbol: str) -> Decimal:
        """I8: dust tolerance for the own-vs-exchange mismatch check -- one
        amount step, or 1e-8 when the step is unknown."""
        return self._amount_step(symbol)

    def _base_asset(self, symbol: str) -> str | None:
        """``market["base"]`` for ``symbol``, or None if the market/base is
        unknown. Pure lookup -- callers decide whether a missing market
        matters and flag ``reconcile_required`` themselves (I6)."""
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(symbol)
        if market is None:
            return None
        base = market.get("base")
        return str(base) if base else None

    def _quote_currency(self, symbol: str) -> str:
        """``market["quote"]`` for ``symbol``, defaulting to ``"USD"`` when
        the market or its quote is unknown (used only for fee-currency
        defaulting, never for sizing)."""
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(symbol) or {}
        quote = market.get("quote")
        return str(quote) if quote else "USD"

    def _quote_asset(self, symbol: str) -> str | None:
        """WP1.4 (S-04/R-09): ``market["quote"]`` for ``symbol``, or
        ``None`` if the market or its quote is unknown. Unlike
        ``_quote_currency`` (which defaults to ``"USD"`` for fee-currency
        bookkeeping only), a missing quote here must block a BUY -- it is
        never silently defaulted."""
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(symbol)
        if market is None:
            return None
        quote = market.get("quote")
        return str(quote) if quote else None

    def _taker_buffer(self, symbol: str) -> Decimal:
        """WP1.4 (S-08): the market's own taker fee, used as the BUY
        affordability cap's safety margin -- deliberately NOT
        ``risk_manager.params`` (the ledger tests mock that, and the fee
        is a property of the market, not the risk config). Defaults to 1%
        when the market or its taker fee is unknown/unparseable.

        Security round 2 (S-03): also falls back to 1% when the exchange
        reports a taker fee that is negative or non-finite (NaN/Inf) --
        a negative fee would double the allowed quantity (only the risk
        manager's own size cap saved it from oversizing), and a non-finite
        one propagates into a ``ZeroDivisionError`` or a silently-wrong
        comparison downstream.
        """
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(symbol) or {}
        buf = _safe_decimal(market.get("taker"), default=Decimal("0.01"))
        if not buf.is_finite() or buf < Decimal("0"):
            return Decimal("0.01")
        return buf

    # ------------------------------------------------------------------
    # WP1.1: held-quantity / SELL cap (WP11-A-03, D9, I1; round 2 S-03/S-07)
    # ------------------------------------------------------------------

    def _pending_sell_quantity(self, symbol: str) -> Decimal:
        """WP11-S-03: total SELL quantity for ``symbol`` still in flight --
        submitted to the exchange but not yet fully routed into the
        position source.

        Sums, over every SELL order this engine has ever submitted for
        ``symbol``:

        - still in flight (``PENDING_SUBMIT``/``OPEN``/``PARTIAL``):
          ``order.quantity - routed_gross`` (the un-filled remainder is
          also "pending" in the sense that it could still fill and needs
          to be reserved against);
        - settled (``FILLED``/``CANCELED``/``EXPIRED``):
          ``max(order.filled_quantity - routed_gross, 0)`` (filled but not
          yet routed into the portfolio).

        Both use ``_routed_gross_qty`` (gross, matching ``filled_quantity``'s
        unit), never ``sum(Fill.quantity)`` (net of fees, S-02).
        """
        pending = Decimal("0")
        now = datetime.now(tz=UTC)
        for order in self._orders.values():
            if order.symbol != symbol or order.side != OrderSide.SELL:
                continue

            if (
                order.status == OrderStatus.PENDING_SUBMIT
                and order.order_id not in self._exchange_order_map
            ):
                # WP11-S-R2-01 (regression fix): create_order raised before
                # the exchange ever acknowledged this order -- it has no
                # exchange id and will never resolve on its own. Reserving
                # its full quantity would block every future SELL forever;
                # flag it for an operator instead of reserving it.
                self._flag_reconcile(symbol, "sell_order_state_unknown")
                continue

            routed_gross = self._routed_gross_qty.get(order.order_id, Decimal("0"))

            if order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
                age_s = (now - order.updated_at).total_seconds()
                if age_s > _INFLIGHT_SELL_MAX_AGE_S:
                    # WP11-S-R2-02 (regression fix): this order's state can
                    # no longer be trusted -- updated_at only advances on a
                    # successful reconcile, so a persistently-failing
                    # fetch_order (e.g. OrderNotFound) means it may never
                    # reconcile again. Reserving it forever would block
                    # every future SELL; flag it instead.
                    self._flag_reconcile(symbol, "sell_order_state_unknown")
                    continue
                pending += max(order.quantity - routed_gross, Decimal("0"))
            elif order.status == OrderStatus.PENDING_SUBMIT:
                # PENDING_SUBMIT with a known exchange id is not reachable
                # via submit_order's own flow today (the exchange id is
                # only recorded after create_order succeeds, by which point
                # the order has already moved past PENDING_SUBMIT) -- kept
                # for defensiveness/symmetry with the check above.
                pending += max(order.quantity - routed_gross, Decimal("0"))
            elif order.status in _SETTLED_ORDER_STATUSES:
                pending += max(order.filled_quantity - routed_gross, Decimal("0"))
        return pending

    def _inflight_buy_orders(self) -> Order | None:
        """WP1.4 (S-02/A-04, hardened security round 2): the first BUY
        order this engine has ever submitted, anywhere in the run (not
        just this symbol), that is still in flight
        (``PENDING_SUBMIT``/``OPEN``/``PARTIAL``) or settled with more
        than a dust residual unrouted
        (``filled_quantity - routed_gross > _amount_tolerance(symbol)``),
        or ``None`` if nothing is blocking. ``REJECTED`` never blocks (it
        is neither in-flight nor in ``_SETTLED_ORDER_STATUSES``).

        Security round 2 (WP14-S-04): the settled branch used a bare
        ``>`` against the *exact* routed amount -- a sub-satoshi rounding
        residual the exchange reports on ``filled`` (e.g. ``amount +
        1e-9``) then blocked every future BUY forever, with nothing in
        ``reconcile_required`` and no order id in the log. Tolerating up
        to one amount step (the same dust tolerance the I8 mismatch check
        already uses) fixes the false positive while still catching a
        genuine unrouted fill. The caller (``process_signal``) is
        responsible for the one-time stale alert
        (``live.buy_inflight_block_stale``) and for logging
        ``blocking_order_id``/``blocking_status``/``unrouted`` on
        ``live.buy_blocked_inflight_buy``.

        Security round 2 (WP14-S-05): a PENDING_SUBMIT BUY with no
        exchange id (``create_order`` raised before the exchange ever
        acknowledged it -- its state is genuinely unknown, it may have
        been accepted anyway) additionally flags
        ``reconcile_required[symbol] = "buy_order_state_unknown"``,
        mirroring the SELL side's WP11-S-R2-01 handling -- unlike the
        SELL path (which stops *reserving* it but still sells), a BUY has
        no "reserve, don't block" fallback, so it keeps blocking.

        Unlike ``_pending_sell_quantity``, there is no age expiry (R3): a
        stuck BUY blocks every new BUY, run-wide, until it genuinely
        settles and routes or an operator intervenes -- the R-03
        alternative (reduce available run cash by the in-flight estimate
        instead of blocking outright) was rejected in favour of this
        simpler, fail-closed rule (one engine per run, so a run-wide block
        cannot starve an unrelated run).
        """
        for order in self._orders.values():
            if order.side != OrderSide.BUY:
                continue
            if order.status in (
                OrderStatus.PENDING_SUBMIT, OrderStatus.OPEN, OrderStatus.PARTIAL,
            ):
                if (
                    order.status == OrderStatus.PENDING_SUBMIT
                    and order.order_id not in self._exchange_order_map
                ):
                    self._flag_reconcile(order.symbol, "buy_order_state_unknown")
                return order
            if order.status in _SETTLED_ORDER_STATUSES:
                routed_gross = self._routed_gross_qty.get(order.order_id, Decimal("0"))
                unrouted = order.filled_quantity - routed_gross
                if unrouted > self._amount_tolerance(order.symbol):
                    return order
        return None

    def _maybe_log_inflight_block_stale(self, order: Order) -> None:
        """WP1.4 (S-04, security round 2): once ``order`` has been
        blocking new BUYs for longer than ``_buy_inflight_stale_after_s``
        (default 15 minutes), log ``live.buy_inflight_block_stale`` at
        error level -- exactly once per order -- so a genuinely stuck BUY
        is impossible to miss in the logs.

        This is an ALERT only: per spec S-02, there is still no age
        expiry -- the block itself never lifts on its own.
        """
        now = datetime.now(tz=UTC)
        started_at = self._inflight_block_started_at.setdefault(order.order_id, now)
        if order.order_id in self._inflight_block_stale_alerted:
            return
        age_s = (now - started_at).total_seconds()
        if age_s > self._buy_inflight_stale_after_s:
            self._inflight_block_stale_alerted.add(order.order_id)
            self._log.error(
                "live.buy_inflight_block_stale",
                order_id=str(order.order_id),
                symbol=order.symbol,
                status=order.status.value,
                age_seconds=age_s,
            )

    async def _held_quantity(self, symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        """Return ``(own, own_avail, capped)`` for a SELL against ``symbol``.

        ``own`` is read from the attached position source (I2).
        ``own_avail`` (WP11-S-03) subtracts this-symbol SELL quantity still
        in flight from ``own`` -- a second SELL signal before the first is
        routed must never re-sell the same quantity out of an external
        holding. ``capped`` floors SELL exposure to ``min(own_avail, free)``
        using a **fresh** balance fetch (D15) -- both ``own_avail`` and
        ``capped`` fall back to being uncapped (never capped down further)
        when the balance is unavailable or the market's base asset is
        unknown, per I1's "never block the exit".

        Also runs the I8 mismatch check (``own_avail`` vs exchange
        ``total``, D9/S-03) so it fires on every SELL, not only at startup
        sync, without going stale while a SELL is still in flight.
        """
        own = self._own_quantity(symbol)
        if own <= Decimal("0"):
            return own, own, own

        pending_sell = self._pending_sell_quantity(symbol)
        own_avail = max(own - pending_sell, Decimal("0"))
        if own_avail <= Decimal("0"):
            return own, own_avail, Decimal("0")

        base = self._base_asset(symbol)
        balance = await self._fetch_balance_cached(fresh=True)
        if balance is None or base is None:
            return own, own_avail, own_avail

        tol = self._amount_tolerance(symbol)
        # WP11-S-07: unparseable balance values are treated as unavailable
        # (None), never coerced to zero.
        total = _safe_decimal((balance.get("total") or {}).get(base), default=None)
        if total is not None and own_avail > total + tol:
            self._flag_reconcile(symbol, "own_exceeds_exchange_total")

        free = _safe_decimal((balance.get("free") or {}).get(base), default=None)
        if free is None:
            return own, own_avail, own_avail

        capped = min(own_avail, free)
        return own, own_avail, max(capped, Decimal("0"))

    # ------------------------------------------------------------------
    # Safety gate
    # ------------------------------------------------------------------

    def _enforce_live_gate(self) -> None:
        """
        Check the live-trading safety gate.

        Raises
        ------
        RuntimeError
            If live trading is not enabled.
        """
        if not self._enable_live_trading:
            raise RuntimeError(
                "Live trading is not enabled. "
                "Set enable_live_trading=True and provide valid API credentials "
                "to place real orders. This is a safety gate to prevent "
                "accidental capital deployment."
            )

    async def _fetch_balance_cached(self, fresh: bool = False) -> dict[str, Any] | None:
        """
        Fetch account balance with a 10-second TTL cache.

        Parameters
        ----------
        fresh:
            WP1.1 (WP11-A-08): bypass the TTL and force a real fetch. Used
            by the SELL cap (D15) and ``sync_positions`` so a same-bar BUY
            followed by a bracket SELL never reads a stale (pre-fill)
            balance.

        Returns None on failure (CR-002: never re-raises).
        """
        now = time.monotonic()
        if (
            not fresh
            and self._balance_cache is not None
            and (now - self._balance_cache_time) < 10.0
        ):
            return dict(self._balance_cache)

        try:
            balance = await ccxt_retry(
                self._exchange.fetch_balance,
                max_retries=2, base_delay=1.0, operation="fetch_balance",
            )
            self._balance_cache = balance
            self._balance_cache_time = now
            return dict(balance)
        except Exception as exc:
            self._log.warning(
                "live.balance_fetch_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            return None

    def _invalidate_balance_cache(self) -> None:
        """WP11-A-08: drop the cached balance so the next read is a real
        fetch. Called after ``create_order`` succeeds, after a filled-qty
        change in ``_reconcile_order``, and after ``cancel_order`` succeeds
        -- every point where the exchange balance actually changed."""
        self._balance_cache = None
        self._balance_cache_time = 0.0

    # ------------------------------------------------------------------
    # CCXT response mapping
    # ------------------------------------------------------------------

    def _map_ccxt_order_status(self, ccxt_status: str) -> OrderStatus:
        """
        Map a CCXT order status string to our internal OrderStatus enum.

        CCXT statuses: 'open', 'closed', 'canceled', 'expired', 'rejected'

        Parameters
        ----------
        ccxt_status:
            Status string from the CCXT order response.

        Returns
        -------
        OrderStatus:
            Mapped internal status.
        """
        mapping: dict[str, OrderStatus] = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(ccxt_status, OrderStatus.OPEN)

    def _extract_fee_from_ccxt(
        self,
        ccxt_trade: dict[str, Any],
        symbol: str | None = None,
    ) -> tuple[Decimal, str]:
        """
        Extract fee amount and currency from a CCXT trade dict.

        Parameters
        ----------
        ccxt_trade:
            A single trade dict from exchange.fetch_order_trades().
        symbol:
            WP1.1 (D6): when the trade carries no fee currency, default to
            this symbol's market quote currency rather than a hardcoded
            "USDT" -- a fee is a quote-currency amount economically, and
            defaulting to the wrong quote silently mixes units. ``None``
            (the pre-WP1.1 call signature) preserves the old "USDT" default
            for callers that have no symbol context.

        Returns
        -------
        tuple[Decimal, str]:
            (fee_amount, fee_currency)
        """
        quote = self._quote_currency(symbol) if symbol is not None else "USDT"
        # WP1.8b: delegates to the module-level, self-free helper so
        # ``apps.api.services.run_recovery.scan_and_import`` shares this
        # exact fee-parsing logic (see the docstring above _parse_ccxt_trades).
        return _extract_fee_from_ccxt_trade(
            ccxt_trade, symbol=symbol, quote_currency=quote, log=self._log
        )

    def _normalize_fee(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        fee_amount: Decimal,
        fee_currency: str,
    ) -> tuple[Decimal, Decimal, str]:
        """WP11-A-05 (D6): normalise a CCXT trade's fee into quote currency.

        A fee charged in the base asset mixes units with a base-asset
        ``Fill.quantity`` if left as-is. For a BUY, the fee is subtracted
        from the received quantity (net fill) and re-expressed in quote
        currency (``fee * price``) so the cost basis stays exact. For a
        SELL, only the fee's currency is converted -- the sold quantity is
        left unchanged; the SELL cap (``min(own_avail, free)``, I1) already
        absorbs any base-asset dust. Any other fee currency (including the
        quote itself) is left unchanged.

        Note (S-02): the returned ``quantity`` is the *net* fill quantity
        used for ``Fill.quantity`` / cost-basis purposes only. Callers must
        track the order's *gross* routed quantity (``_routed_gross_qty``)
        separately using the pre-normalisation raw trade amount, since
        ``order.filled_quantity`` from the exchange is always gross.
        """
        # WP1.8b: delegates to the module-level, self-free helper (see
        # _parse_ccxt_trades' docstring) -- identical behaviour, kept as a
        # method because existing unit tests call it directly.
        return _normalize_trade_fee(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee_amount=fee_amount,
            fee_currency=fee_currency,
            base_asset=self._base_asset(symbol),
            quote_currency=self._quote_currency(symbol),
            log=self._log,
        )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the live exchange via CCXT.

        The full flow:
        1. Enforce the live-trading safety gate.
        2. Transition NEW -> PENDING_SUBMIT.
        3. Call exchange.create_order() via CCXT.
        4. On success: transition to OPEN and record exchange_order_id.
        5. On failure: transition to REJECTED with error details.

        Parameters
        ----------
        order:
            A fully validated Order with status=NEW.

        Returns
        -------
        Order:
            Updated order reflecting the submission outcome.

        Raises
        ------
        RuntimeError:
            If live trading is not enabled.
        """
        self._enforce_live_gate()

        # Register the order
        self._orders[order.order_id] = order

        # NEW -> PENDING_SUBMIT
        order = self._transition(order, OrderStatus.PENDING_SUBMIT)

        try:
            # WP14-S-10 (security round 2 follow-up): pop the sizing-price
            # hint unconditionally, before anything else in this block can
            # raise. Previously this only happened inside the Coinbase
            # branch below, so a hint recorded for a BUY on any OTHER
            # exchange (or one that fails before reaching that branch)
            # was never removed and leaked in _buy_sizing_price for the
            # life of the run. Harmless (order ids are unique, so a stale
            # hint can never be reused for a different order), but worth
            # cleaning up rather than deferring to the WP14-S-06 rewrite.
            sizing_price_hint = self._buy_sizing_price.pop(order.order_id, None)

            # Build CCXT order parameters
            ccxt_order_type = order.order_type.value  # 'market' or 'limit'
            ccxt_side = order.side.value  # 'buy' or 'sell'
            price_param = str(order.price) if order.price is not None else None

            # Coinbase requires a price for market BUY orders on spot markets
            # to calculate total cost (amount * price). Fetch last price if needed.
            if (
                price_param is None
                and ccxt_order_type == "market"
                and ccxt_side == "buy"
                and self._exchange.id == "coinbase"
            ):
                # WP14-S-01 (security round 2): reuse the SAME price
                # process_signal's affordability cap already sized this
                # BUY against, instead of fetching a second, later ticker
                # -- a rising price between the two fetches let the order
                # spend more than run cash allowed (amount * this second
                # price > the cap that was actually enforced).
                if sizing_price_hint is not None:
                    price_param = str(sizing_price_hint)
                else:
                    try:
                        ticker = await ccxt_retry(
                            self._exchange.fetch_ticker, order.symbol,
                            max_retries=1, base_delay=0.5, operation=f"fetch_ticker_for_buy({order.symbol})",
                        )
                        price_param = str(ticker.get("last", "0"))
                    except Exception:
                        self._log.warning(
                            "live.market_buy_price_fallback_failed",
                            symbol=order.symbol,
                        )

            params: dict[str, Any] = {}
            if order.client_order_id:
                params["clientOrderId"] = order.client_order_id

            ccxt_response = await ccxt_retry(
                self._exchange.create_order,
                order.symbol,
                ccxt_order_type,
                ccxt_side,
                str(order.quantity),
                price_param,
                max_retries=2, base_delay=1.0, operation=f"create_order({order.symbol})",
                params=params,
            )

            # WP11-A-08: the exchange balance just changed (funds locked or
            # spent); drop the cache so the next read (e.g. a same-bar
            # bracket SELL's cap) is not stale.
            self._invalidate_balance_cache()

            # Extract exchange order ID
            exchange_order_id = str(ccxt_response.get("id", ""))

            # Record the mapping
            self._exchange_order_map[order.order_id] = exchange_order_id
            self._reverse_order_map[exchange_order_id] = order.order_id

            # Update order with exchange info (do NOT store yet -- wait for fill data)
            order = order.model_copy(update={
                "exchange_order_id": exchange_order_id,
                "updated_at": datetime.now(tz=UTC),
            })

            # Determine initial status from exchange response
            ccxt_status = ccxt_response.get("status", "open")
            mapped_status = self._map_ccxt_order_status(ccxt_status)

            # PENDING_SUBMIT -> OPEN (or directly to FILLED for instant fills)
            #
            # Coinbase returns None for "filled" and "average" on market orders
            # that are still processing. Guard with _safe_decimal() to prevent
            # decimal.InvalidOperation on None/non-numeric values.
            if mapped_status == OrderStatus.FILLED:
                order = self._transition(order, OrderStatus.OPEN)

                # Extract fill data from response
                filled_qty = _safe_decimal(
                    ccxt_response.get("filled"), order.quantity
                )
                avg_price = _safe_decimal(
                    ccxt_response.get("average")
                ) or _safe_decimal(ccxt_response.get("price"), Decimal("0"))

                # Apply fill data and store atomically before terminal transition
                order = order.model_copy(update={
                    "filled_quantity": filled_qty,
                    "average_fill_price": avg_price,
                    "updated_at": datetime.now(tz=UTC),
                })
                self._orders[order.order_id] = order

                order = self._transition(order, OrderStatus.FILLED)
            else:
                order = self._transition(order, mapped_status)

                # If partially filled on creation
                filled_qty = _safe_decimal(ccxt_response.get("filled"), Decimal("0"))
                if filled_qty > Decimal("0"):
                    avg_price = _safe_decimal(
                        ccxt_response.get("average"), Decimal("0")
                    )
                    order = order.model_copy(update={
                        "filled_quantity": filled_qty,
                        "average_fill_price": avg_price if avg_price > 0 else None,
                        "updated_at": datetime.now(tz=UTC),
                    })
                    self._orders[order.order_id] = order

            self._log.info(
                "live.order_submitted",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                symbol=order.symbol,
                side=order.side.value,
                type=order.order_type.value,
                quantity=str(order.quantity),
                status=order.status.value,
            )

        except ccxt_async.NetworkError as exc:
            # Transient network error — order rejected (submit already uses ccxt_retry)
            self._log.error(
                "live.order_submission_network_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.AuthenticationError as exc:
            self._log.error(
                "live.order_submission_auth_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.InsufficientFunds as exc:
            self._log.error(
                "live.order_submission_insufficient_funds",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except ccxt_async.ExchangeError as exc:
            self._log.error(
                "live.order_submission_exchange_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = self._transition(order, OrderStatus.REJECTED)
        except Exception as exc:
            # Unexpected error -- do NOT silently absorb. Log and re-raise.
            self._log.critical(
                "live.order_submission_unexpected_error",
                order_id=str(order.order_id),
                symbol=order.symbol,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        return order

    async def cancel_order(self, order_id: UUID) -> Order:
        """
        Cancel an open or partially-filled order on the exchange.

        Parameters
        ----------
        order_id:
            Internal UUID of the order to cancel.

        Returns
        -------
        Order:
            Updated order with status CANCELED (or FILLED if race occurred).

        Raises
        ------
        RuntimeError:
            If live trading is not enabled.
        KeyError:
            If the order_id is not found.
        """
        self._enforce_live_gate()

        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        exchange_order_id = self._exchange_order_map.get(order_id)

        try:
            if exchange_order_id:
                await ccxt_retry(
                    self._exchange.cancel_order,
                    exchange_order_id,
                    order.symbol,
                    max_retries=2, base_delay=1.0,
                    operation=f"cancel_order({order.symbol})",
                )

            # CR-003: transition inside success branch only
            order = self._transition(order, OrderStatus.CANCELED)
            # WP11-A-08: a cancel releases any locked funds back to "free".
            self._invalidate_balance_cache()

            self._log.info(
                "live.order_canceled",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                symbol=order.symbol,
            )

        except (ccxt_async.OrderNotFound, ccxt_async.InvalidOrder):
            # Race condition: order was already filled on exchange
            self._log.info(
                "live.cancel_already_terminal",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
            )
            order = await self._reconcile_order(order)

        except Exception as exc:
            self._log.warning(
                "live.cancel_failed",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            order = await self._reconcile_order(order)

        return order

    async def get_order(self, order_id: UUID) -> Order:
        """
        Fetch the current state of an order, reconciling with the exchange.

        For orders with an exchange_order_id, this calls exchange.fetch_order()
        and reconciles the local state with the exchange state.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        Order:
            The most up-to-date order record.

        Raises
        ------
        KeyError:
            If the order_id is not found.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        # If the order is in a terminal state, no need to reconcile
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        if order.status in terminal:
            return order

        return await self._reconcile_order(order)

    async def process_signal(self, signal: Signal) -> list[Order]:
        """
        Convert a trading Signal into zero or more submitted Orders.

        Same risk-gated flow as the paper engine, but using real
        exchange submission.

        WP1.1 (C1/C22): SELL sizing and the "do we hold this?" guard now
        read from the attached ``LivePositionSource`` (D1) instead of the
        never-populated legacy ``_positions`` dict; a symbol flagged
        ``reconcile_required`` blocks BUYs only (I4/D2) -- SELLs are always
        allowed, capped at ``min(own_avail, free)`` (I1/D15/S-03).

        WP1.4 (D5): with a source attached, a BUY additionally passes a
        quote-mismatch check, an in-flight-BUY check, a NAV snapshot
        (``_equity_snapshot``, blocking on ``live.nav_unavailable``) and a
        cash/free-quote affordability cap before the existing min-amount/
        min-cost/precision checks -- sizes at ``min(NAV, initial_capital)``
        (I-2), never the exchange balance (I-1). The risk gate itself
        always sees the raw NAV, not the sizing basis (I-3). SELLs and the
        no-source (D8) path are unaffected (I-5).

        Security round 2 (amends I-4): the affordability cap prices at
        ``max(last, ask)`` (WP14-S-02, a bare ``last`` under-caps whenever
        a market BUY actually fills above it) and ``buf = taker +
        buy_cap_slippage_pct`` (a configurable margin beyond the taker fee
        alone). The same price is handed to ``submit_order`` via
        ``_buy_sizing_price`` so its Coinbase market-BUY path sizes the
        order against the SAME number instead of a second, later ticker
        (WP14-S-01). ``_inflight_buy_orders`` now returns the blocking
        ``Order`` (not just a bool) so ``live.buy_blocked_inflight_buy``
        can log which order and by how much (WP14-S-04), tolerates a dust
        rounding residual instead of blocking forever on it (WP14-S-04),
        and flags ``reconcile_required`` for a PENDING_SUBMIT BUY with no
        exchange id, mirroring the SELL side (WP14-S-05).

        Parameters
        ----------
        signal:
            The Signal emitted by a strategy's on_bar call.

        Returns
        -------
        list[Order]:
            Orders created and submitted (may be empty).
        """
        self._enforce_live_gate()

        if signal.direction == SignalDirection.HOLD:
            return []

        side = (
            OrderSide.BUY
            if signal.direction == SignalDirection.BUY
            else OrderSide.SELL
        )

        # Keep the legacy `_positions` cache fresh from the source (if any)
        # before either branch below reads it (D1).
        self._sync_positions_cache(signal.symbol)

        held_quantity: Decimal | None = None
        if side == OrderSide.BUY:
            # I4/D2: a flagged symbol blocks BUYs only; SELLs are never
            # blocked by reconcile_required.
            if signal.symbol in self._reconcile_required:
                self._log.warning(
                    "live.buy_blocked_reconcile_required",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    reason=self._reconcile_required[signal.symbol],
                )
                return []
            if self._position_source is not None:
                # WP1.4 (S-04): a run whose symbols don't share one quote
                # currency never sizes/caps a BUY off the right balance --
                # blocked once at on_start, never healed mid-run.
                if self._run_buy_block is not None:
                    self._log.warning(
                        "live.buy_blocked_quote_mismatch",
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                        reason=self._run_buy_block,
                    )
                    return []
                # WP1.4 (S-02/A-04): a BUY still in flight anywhere in the
                # run, or filled but not yet routed, blocks every new BUY
                # -- otherwise a late fill lets the next BUY spend the
                # same run cash twice (R-03's alternative, a reduced
                # run_cash_avail, was rejected in favour of this simpler
                # rule). No age expiry (R3): a stuck order blocks until it
                # genuinely routes or an operator intervenes.
                blocking_order = self._inflight_buy_orders()
                if blocking_order is not None:
                    # Security round 2 (WP14-S-04): a one-time, error-level
                    # alert if this has been blocking for too long, plus
                    # the blocking order's id/status/unrouted amount on
                    # every occurrence -- both were missing before.
                    self._maybe_log_inflight_block_stale(blocking_order)
                    blocking_routed = self._routed_gross_qty.get(
                        blocking_order.order_id, Decimal("0")
                    )
                    self._log.warning(
                        "live.buy_blocked_inflight_buy",
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                        blocking_order_id=str(blocking_order.order_id),
                        blocking_status=blocking_order.status.value,
                        unrouted=str(blocking_order.filled_quantity - blocking_routed),
                    )
                    return []
        else:
            own, own_avail, capped = await self._held_quantity(signal.symbol)
            if own <= Decimal("0"):
                self._log.info(
                    "live.sell_no_position",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                )
                return []
            if own_avail <= Decimal("0"):
                # WP11-S-03: this symbol's own quantity is fully accounted
                # for by SELL orders already in flight -- do NOT sell the
                # same quantity again (e.g. out of an external holding).
                # This does not block protection: the exit is already
                # submitted and will route once its fill is picked up.
                self._log.warning(
                    "live.sell_inflight_pending",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    own=str(own),
                    reserved=str(own - own_avail),
                )
                return []
            if capped <= Decimal("0"):
                self._log.warning(
                    "live.sell_capped_to_zero",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    own=str(own),
                )
                return []
            held_quantity = capped

        # Fetch current ticker for position sizing (uses ccxt_retry)
        try:
            ticker = await ccxt_retry(
                self._exchange.fetch_ticker, signal.symbol,
                max_retries=2, base_delay=1.0,
                operation=f"fetch_ticker({signal.symbol})",
            )
            last_price = Decimal(str(ticker.get("last", "0")))
            if last_price <= Decimal("0"):
                self._log.error(
                    "live.invalid_ticker_price",
                    symbol=signal.symbol,
                    ticker=ticker,
                )
                return []
        except Exception as exc:
            self._log.error(
                "live.ticker_fetch_failed",
                symbol=signal.symbol,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            return []

        # SYN-S51: target_position is the authoritative sizing input; the
        # risk manager ceiling only de-sizes. equity is fetched from the
        # same source the legacy calculate_position_size call used; for SELL
        # we pass the held quantity so the resolver can full-close / cap.
        #
        # WP1.4 (D5/S-01): with a source attached, NAV/peak/basis come from
        # the portfolio, never the exchange balance (I-1). BUY sizes at
        # ``min(NAV, initial_capital)`` (I-2); SELL and the risk gate (I-3)
        # both use the source's raw ``current_equity`` so a run in profit
        # never shows a false drawdown. With no source, the legacy
        # exchange-balance path (D8) is unchanged.
        nav: Decimal
        peak_equity: Decimal | None
        if self._position_source is not None:
            source = self._position_source
            if side == OrderSide.BUY:
                snapshot = self._equity_snapshot(signal.symbol, last_price)
                if snapshot is None:
                    self._log.warning(
                        "live.nav_unavailable",
                        strategy_id=signal.strategy_id,
                        symbol=signal.symbol,
                    )
                    return []
                equity = snapshot.sizing_basis
                nav = snapshot.nav
                peak_equity = snapshot.peak
            else:
                nav = source.current_equity
                equity = nav
                peak_equity = max(source.get_peak_equity(), nav)
        else:
            equity = await self._fetch_equity()
            nav = equity
            peak_equity = None  # fetched later, exactly where D8 always fetched it

        quantity = self._resolve_order_quantity(
            signal=signal,
            side=side,
            last_price=last_price,
            equity=equity,
            held_quantity=held_quantity,
        )

        if quantity <= Decimal("0"):
            self._log.warning(
                "live.zero_quantity",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                equity=str(equity),
                last_price=str(last_price),
                confidence=signal.confidence,
            )
            return []

        # WP1.4 (I-4/I-5): BUY affordability cap -- run cash and a fresh
        # exchange free-quote balance, never the sizing basis alone (D5
        # would otherwise let a run with a shrunk account balance still
        # size off its own book). Applied before the min-amount/min-cost
        # checks and the precision floor (spec BUY sequence step 7).
        # SELLs are never touched by this guard (I-5).
        #
        # Security round 2 (amends I-4): the cap price is
        # ``max(last, ask)``, not ``last`` alone, and ``buf`` folds in a
        # slippage margin on top of the taker fee -- ``last`` (or a
        # misreported taker fee) alone let a fill push run cash negative
        # (WP14-S-02). ``buy_cap_price`` also becomes the price
        # ``submit_order``'s Coinbase path sizes the order against
        # (WP14-S-01) instead of fetching a second, later ticker.
        buy_cap_price: Decimal | None = None
        if side == OrderSide.BUY and self._position_source is not None:
            source = self._position_source
            quote = self._quote_asset(signal.symbol)
            balance = await self._fetch_balance_cached(fresh=True)
            free_quote = (
                _safe_decimal((balance.get("free") or {}).get(quote), default=None)
                if balance is not None and quote is not None
                else None
            )
            # WP14-S-03 (security round 2): a non-finite (NaN/Inf) free
            # balance is exactly as unusable as a missing one -- a NaN
            # comparison below would otherwise raise InvalidOperation.
            if free_quote is None or not free_quote.is_finite():
                self._log.warning(
                    "live.buy_blocked_balance_unavailable",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                )
                return []
            cash = max(source.cash, Decimal("0"))
            if free_quote < cash:
                self._log.warning(
                    "live.run_cash_exceeds_exchange_free",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    run_cash=str(cash),
                    free_quote=str(free_quote),
                )
            ask_price = _safe_decimal(ticker.get("ask"), default=None)
            if ask_price is None or not ask_price.is_finite() or ask_price <= Decimal("0"):
                ask_price = last_price
            buy_cap_price = max(last_price, ask_price)
            buf = self._taker_buffer(signal.symbol) + self._buy_cap_slippage_pct
            quantity = self._cap_buy_quantity(
                quantity, buy_cap_price, min(cash, free_quote), buf
            )

        # ------------------------------------------------------------------
        # Exchange minimum order size validation
        # Checked after all quantity adjustments (position cap, etc.) so we
        # evaluate the final quantity that would actually be submitted.
        # markets is populated by load_markets() in on_start(); if markets is
        # not yet loaded (e.g. in disabled-gate mode) the dict will be empty
        # and the guard is skipped gracefully.
        # ------------------------------------------------------------------
        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        market = markets.get(signal.symbol)
        if market:
            limits: dict[str, Any] = market.get("limits") or {}
            amount_limits: dict[str, Any] = limits.get("amount") or {}
            cost_limits: dict[str, Any] = limits.get("cost") or {}

            min_amount = amount_limits.get("min")
            min_cost = cost_limits.get("min")

            notional = quantity * last_price

            if min_amount is not None and float(quantity) < float(min_amount):
                self._log.warning(
                    "live.below_min_amount",
                    symbol=signal.symbol,
                    quantity=str(quantity),
                    min_amount=str(min_amount),
                    msg=(
                        f"Order quantity {quantity} below exchange minimum "
                        f"{min_amount} for {signal.symbol}"
                    ),
                )
                return []

            if min_cost is not None and float(notional) < float(min_cost):
                self._log.warning(
                    "live.below_min_cost",
                    symbol=signal.symbol,
                    notional=str(notional),
                    min_cost=str(min_cost),
                    msg=(
                        f"Order notional {notional} below exchange minimum cost "
                        f"{min_cost} for {signal.symbol}"
                    ),
                )
                return []

        # WP1.1 (D5): floor to the market's own amount precision (step size
        # or decimal places) instead of a fixed 8dp — one helper for both
        # sides. A quantity that floors to zero (sub-precision dust) is
        # treated exactly like the zero-quantity guard above.
        floored_quantity = self._floor_to_amount_precision(signal.symbol, quantity)
        if floored_quantity <= Decimal("0"):
            self._log.warning(
                "live.zero_quantity_after_precision_floor",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                quantity=str(quantity),
            )
            return []

        # Build the proposed order
        client_order_id = f"{self._run_id}-{uuid4().hex[:12]}"
        proposed_order = Order(
            client_order_id=client_order_id,
            run_id=self._run_id,
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=floored_quantity,
        )

        # Pre-trade risk check. WP1.1 (WP11-A-03): read open positions from
        # the attached source when present (D1); fall back to the legacy
        # `_positions` cache otherwise (D8).
        if self._position_source is not None:
            open_positions = self._position_source.get_open_positions()
        else:
            open_positions = [
                p for p in self._positions.values() if not p.is_flat
            ]
        daily_pnl = self._calculate_daily_pnl()
        if peak_equity is None:
            # D8 (no source): fetched here, exactly where this call has
            # always lived -- never moved earlier, so a return above this
            # point (zero quantity, below-minimum, precision floor) still
            # never triggers a peak fetch, unchanged from before WP1.4.
            peak_equity = await self._fetch_peak_equity()

        risk_result = self._risk_manager.pre_trade_check(
            order=proposed_order,
            current_equity=nav,
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            peak_equity=peak_equity,
            market_price=last_price,
        )

        if not risk_result.approved:
            self._log.warning(
                "live.signal_rejected",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                reasons=risk_result.rejection_reasons,
            )
            return []

        # Apply adjusted quantity from risk check (only when approved;
        # on rejection adjusted_quantity is 0 and we already returned above)
        if risk_result.approved and risk_result.adjusted_quantity < proposed_order.quantity:
            proposed_order = proposed_order.model_copy(update={
                "quantity": risk_result.adjusted_quantity,
            })

        if risk_result.warnings:
            self._log.warning(
                "live.risk_warnings",
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                warnings=risk_result.warnings,
            )

        # WP14-S-01 (security round 2): record the price the affordability
        # cap sized this BUY against so submit_order's Coinbase path can
        # reuse it instead of fetching a second, later ticker.
        if side == OrderSide.BUY and buy_cap_price is not None:
            self._buy_sizing_price[proposed_order.order_id] = buy_cap_price

        # Submit
        submitted_order = await self.submit_order(proposed_order)

        # Coinbase processes market orders asynchronously — the initial response
        # often returns status="open" with filled=None.  Wait briefly and
        # reconcile so the order transitions to FILLED and position tracking
        # picks up the fill.
        if submitted_order.status == OrderStatus.OPEN:
            await asyncio.sleep(2)
            submitted_order = await self._reconcile_order(submitted_order)

        return [submitted_order]

    def _trade_key(self, trade: dict[str, Any]) -> tuple[Any, ...]:
        """I9: idempotency key for a CCXT trade record -- ``(id, trade id)``
        when the exchange provides a trade id, else
        ``(synthetic, timestamp, amount, price)``.

        WP11-S-08 (documented limitation, no behaviour change this WP): if
        the *same* trade is first observed without an id (fallback key) and
        later re-observed *with* an id (e.g. an exchange backfilling trade
        ids asynchronously, or a fallback-then-primary API switch), it will
        be treated as two different trades and routed twice. This is a
        known, accepted limitation for exchanges that reliably omit trade
        ids on ``fetch_my_trades`` (Coinbase, the production target) --
        such exchanges never later "gain" an id for the same trade through
        this code path. A future WP should pin the fallback key to only
        exchanges known to omit trade ids (rather than switching per-call
        on whatever a given response happens to contain) if this system
        ever adds an exchange that is inconsistent within a single order's
        lifetime.
        """
        # WP1.8b: delegates to the module-level, self-free helper (see
        # _parse_ccxt_trades' docstring).
        return _trade_key(trade)

    def _synthesize_unrouted_fill(self, order: Order) -> list[Fill]:
        """§3 scope / R-12: trades are empty or the fetch failed while the
        order is FILLED -- synthesise exactly one fill for the unrouted
        quantity at ``average_fill_price`` with fee 0, and permanently
        ignore any later trade records for this order so a real trade that
        eventually appears cannot double-count it.

        WP1.1 round 2 (S-01/S-02, CRITICAL): every branch that does **not**
        synthesise a new fill returns ``[]`` -- never the order's cached
        fill history. Returning ``cached`` here (the round-1 defect) meant
        a PARTIAL order, or a CANCELED order carrying a partial fill, whose
        trade fetch failed or came back empty had its already-routed fill
        handed back to the caller *again* on every subsequent poll, with no
        bound (P1/P2). ``unrouted`` is computed from ``_routed_gross_qty``
        (gross, S-02), not ``sum(Fill.quantity)`` (net of fees) -- a
        base-currency fee no longer makes this look under-routed forever
        (P3).
        """
        if order.status != OrderStatus.FILLED:
            return []

        already_routed_gross = self._routed_gross_qty.get(order.order_id, Decimal("0"))
        unrouted = order.filled_quantity - already_routed_gross
        if unrouted <= Decimal("0"):
            return []

        price = order.average_fill_price
        if price is None or price <= Decimal("0"):
            self._flag_reconcile(order.symbol, "fill_price_invalid")
            self._log.error(
                "live.invalid_fill_price",
                order_id=str(order.order_id),
                symbol=order.symbol,
            )
            return []

        synthetic = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=unrouted,
            price=price,
            fee=Decimal("0"),
            fee_currency=self._quote_currency(order.symbol),
            is_maker=False,
            executed_at=order.updated_at,
        )
        self._fills.setdefault(order.order_id, []).append(synthetic)
        self._routed_gross_qty[order.order_id] = already_routed_gross + unrouted
        self._synthesized_orders.add(order.order_id)
        self._log.warning(
            "live.fill_synthesized",
            order_id=str(order.order_id),
            symbol=order.symbol,
            quantity=str(unrouted),
            price=str(price),
        )
        return [synthetic]

    async def get_fills(self, order_id: UUID) -> list[Fill]:
        """
        Return newly-routed fills for an order by querying the exchange.

        WP1.1 (I9): idempotent per ``(id, trade id)`` (or ``(synthetic,
        timestamp, amount, price)`` when the exchange omits a trade id,
        S-08) -- a trade already turned into a Fill is never returned
        again, so callers such as ``StrategyEngine._check_resting_orders``
        (D3, late fills) can call this repeatedly without double-routing
        into ``PortfolioAccounting``. The full historical set stays in
        ``self._fills`` so :meth:`get_all_fills` is unaffected.

        WP1.1 round 2 hardening:

        - **S-01**: the "no exchange mapping yet" early exit returns ``[]``,
          never the order's cached fill history (which would otherwise be
          re-delivered to the caller and double-routed).
        - **S-04**: this batch of trades is parsed into a local buffer and
          only committed to ``_routed_trade_keys``/``_fills``/
          ``_routed_gross_qty`` after every trade in the batch has been
          handled without raising -- one trade's parse error can no longer
          silently drop fills already parsed earlier in the same call. A
          trade's timestamp falls back to the order's own ``updated_at``
          (in ms) when missing; any other per-trade parse failure flags
          ``fill_parse_failed`` and leaves that one trade unrouted (retried
          on the next call), the same treatment I3 gives an invalid price.

        Parameters
        ----------
        order_id:
            Internal UUID of the order.

        Returns
        -------
        list[Fill]:
            Newly-routed fills sorted by executed_at ascending (empty if
            nothing new since the last call).
        """
        order = self._orders.get(order_id)
        exchange_order_id = self._exchange_order_map.get(order_id)

        if exchange_order_id is None or order is None:
            # WP11-S-01: never return cached history here -- there is
            # nothing new to route, and re-delivering old fills would
            # double-count them in the caller.
            return []

        if order_id in self._synthesized_orders:
            return []

        already_routed = self._routed_trade_keys.get(order_id, set())

        try:
            # Use exchange.has to pick the best fills-fetching method.
            # Binance supports fetchOrderTrades; Coinbase does not.
            if self._exchange.has.get("fetchOrderTrades"):
                ccxt_trades = await self._exchange.fetch_order_trades(
                    id=exchange_order_id,
                    symbol=order.symbol,
                )
            else:
                # Fallback: fetch all recent trades and filter by order ID.
                all_trades = await self._exchange.fetch_my_trades(
                    symbol=order.symbol,
                )
                ccxt_trades = [
                    t for t in all_trades
                    if t.get("order") == exchange_order_id
                ]
        except Exception as exc:
            self._log.warning(
                "live.fetch_fills_failed",
                order_id=str(order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            return self._synthesize_unrouted_fill(order)

        if not ccxt_trades:
            return self._synthesize_unrouted_fill(order)

        # WP1.8b: the per-trade parse loop now lives in the module-level,
        # self-free _parse_ccxt_trades (shared with
        # apps.api.services.run_recovery.scan_and_import) -- this call is
        # byte-for-byte the same batch-then-commit contract WP11-S-04
        # documented (one bad trade record can never lose fills already
        # parsed earlier in this same call; nothing is committed to shared
        # state until every still-new trade in the batch has been handled).
        pending_fills, pending_gross, pending_new_keys = _parse_ccxt_trades(
            order,
            ccxt_trades,
            already_routed=already_routed,
            base_asset=self._base_asset(order.symbol),
            quote_currency=self._quote_currency(order.symbol),
            log=self._log,
            on_skip=lambda reason: self._flag_reconcile(order.symbol, reason),
        )

        # Commit atomically (S-04).
        if pending_new_keys:
            self._routed_trade_keys.setdefault(order_id, set()).update(pending_new_keys)
        if pending_gross > Decimal("0"):
            self._routed_gross_qty[order_id] = (
                self._routed_gross_qty.get(order_id, Decimal("0")) + pending_gross
            )
        for fill in pending_fills:
            self._fills.setdefault(order_id, []).append(fill)

        return sorted(pending_fills, key=lambda f: f.executed_at)

    def get_all_fills(self) -> list[Fill]:
        """Return all fills across all orders, sorted by executed_at.

        Note: returns only locally cached fills -- the full history
        accumulated across every :meth:`get_fills` call (including
        synthesised fills), not just the most recent call's newly-routed
        ones. If some orders' fills were never fetched from the exchange,
        they will not be included.
        """
        all_fills: list[Fill] = []
        for fills in self._fills.values():
            all_fills.extend(fills)
        return sorted(all_fills, key=lambda f: f.executed_at)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _reconcile_order(self, order: Order) -> Order:
        """
        Reconcile local order state with the exchange.

        Fetches the current order state from the exchange and updates
        the local record to match, respecting state-machine transitions.

        Parameters
        ----------
        order:
            The local order to reconcile.

        Returns
        -------
        Order:
            Updated order reflecting the exchange state.
        """
        exchange_order_id = self._exchange_order_map.get(order.order_id)
        if exchange_order_id is None:
            return order

        previous_filled_qty = order.filled_quantity

        try:
            ccxt_order = await ccxt_retry(
                self._exchange.fetch_order,
                exchange_order_id,
                order.symbol,
                max_retries=2, base_delay=1.0,
                operation=f"reconcile_order({order.symbol})",
            )

            # Update fill data (guard against None from Coinbase)
            filled_qty = _safe_decimal(ccxt_order.get("filled"), Decimal("0"))
            avg_price = (
                _safe_decimal(ccxt_order.get("average"))
                or _safe_decimal(ccxt_order.get("price"))
                or None
            )

            order = order.model_copy(update={
                "filled_quantity": filled_qty,
                "average_fill_price": avg_price,
                "updated_at": datetime.now(tz=UTC),
            })
            self._orders[order.order_id] = order

            if filled_qty != previous_filled_qty:
                # WP11-A-08: the exchange balance changed with the fill.
                self._invalidate_balance_cache()

            # Determine target status
            ccxt_status = ccxt_order.get("status", "open")
            target_status = self._map_ccxt_order_status(ccxt_status)

            # Handle partial fills
            if (
                target_status == OrderStatus.OPEN
                and filled_qty > Decimal("0")
                and filled_qty < order.quantity
            ):
                target_status = OrderStatus.PARTIAL

            # Only transition if the status actually changed
            if target_status != order.status:
                try:
                    order = self._transition(order, target_status)
                except Exception as exc:
                    self._log.warning(
                        "live.reconciliation_transition_failed",
                        order_id=str(order.order_id),
                        from_status=order.status.value,
                        to_status=target_status.value,
                        error=str(exc),
                    )

            self._log.debug(
                "live.order_reconciled",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                status=order.status.value,
                filled_quantity=str(order.filled_quantity),
            )

        except Exception as exc:
            self._log.warning(
                "live.reconciliation_failed",
                order_id=str(order.order_id),
                exchange_order_id=exchange_order_id,
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )

        return order

    async def reconcile_open_orders(self) -> int:
        """Reconcile all orders still in OPEN or PARTIAL state with the exchange.

        Coinbase processes market orders asynchronously, so orders may stay
        in OPEN state after submission.  This method polls the exchange for
        the current state of each open order and updates the local record.

        Returns the number of orders that transitioned to a terminal state.
        """
        open_orders = self.get_open_orders()
        if not open_orders:
            return 0

        transitioned = 0
        for order in open_orders:
            old_status = order.status
            order = await self._reconcile_order(order)
            if order.status != old_status:
                transitioned += 1
                self._log.info(
                    "live.order_reconciled_terminal",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    old_status=old_status.value,
                    new_status=order.status.value,
                    filled_quantity=str(order.filled_quantity),
                )

        if transitioned > 0:
            self._log.info(
                "live.reconcile_open_orders_completed",
                total_open=len(open_orders),
                transitioned=transitioned,
            )
        return transitioned

    # ------------------------------------------------------------------
    # Resting-order / late-fill check (D3, S-01)
    # ------------------------------------------------------------------

    async def check_resting_orders(self, symbol: str, price: Decimal) -> list[Order]:
        """WP1.1 (D3): re-poll orders that may carry an unrouted fill.

        Called every bar by ``StrategyEngine._check_resting_orders`` (the
        same duck-typed hook the paper engine uses for limit orders), which
        then calls :meth:`get_fills` on each returned order to actually
        route any new fill into ``PortfolioAccounting`` -- this method never
        calls ``get_fills`` itself, so it cannot short-circuit that
        idempotent routing (I9).

        Two candidate classes for ``symbol``:
        - OPEN/PARTIAL orders: re-reconciled against the exchange so a fill
          that arrived after ``process_signal``'s own wait is picked up on
          the very next bar (a late fill, WP11-A-R1).
        - Terminal orders (FILLED/CANCELED/REJECTED/EXPIRED) whose routed
          **gross** fill quantity (``_routed_gross_qty``, S-02 -- not
          ``sum(Fill.quantity)``, which is net of fees) is still below
          ``filled_quantity`` -- e.g. a synthesised partial fill that later
          gains a second real trade record, or a CANCELED order that
          carries a partial fill.
        """
        candidates: list[Order] = []
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            if order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
                order = await self._reconcile_order(order)
                candidates.append(order)
            elif order.status in _TERMINAL_ORDER_STATUSES and order.filled_quantity > Decimal("0"):
                routed_gross = self._routed_gross_qty.get(order.order_id, Decimal("0"))
                if routed_gross < order.filled_quantity:
                    candidates.append(order)
        return candidates

    # ------------------------------------------------------------------
    # Equity helpers (exchange-aware; legacy no-source only, WP1.4 S-03)
    # ------------------------------------------------------------------

    async def _fetch_equity(self) -> Decimal:
        """
        Fetch current account equity from the exchange.

        Falls back to local position tracking if exchange call fails.

        WP1.4 (S-03): legacy no-source (D8) path ONLY. With a
        ``LivePositionSource`` attached, NAV comes exclusively from
        :meth:`_equity_snapshot` / the source's own ``current_equity`` --
        this method is never called in that case. WP4.1 removes it once
        the no-source path itself is retired.

        Returns
        -------
        Decimal:
            Total account equity in quote currency.
        """
        balance = await self._fetch_balance_cached()
        if balance is not None:
            total = balance.get("total", {})
            for quote in ("EUR", "USDT", "BUSD", "USD", "USDC"):
                if quote in total and total[quote] is not None and float(total[quote]) > 0:
                    return Decimal(str(total[quote]))
            self._log.warning("live.equity_fallback_to_local")

        # Fallback: use peak_equity (last known good value) or position tracking
        if self._peak_equity > Decimal("0"):
            self._log.debug("live.equity_fallback_to_peak", peak=str(self._peak_equity))
            return self._peak_equity

        self._log.warning(
            "live.equity_fallback_to_positions",
            msg="Using position-only equity -- cash balance unknown.",
        )
        total_value = Decimal("0")
        for pos in self._positions.values():
            if not pos.is_flat:
                total_value += pos.notional_value
        return total_value

    async def _fetch_peak_equity(self) -> Decimal:
        """
        Return the highest equity observed during this run.

        Uses the INTERNAL peak tracker (seeded at startup from current
        exchange balance). This ensures drawdown is calculated relative
        to THIS run's starting equity, not a historical account high
        from before this run started.

        WP1.4 (S-03): legacy no-source (D8) path ONLY -- same caveat as
        ``_fetch_equity``. With a source attached, the portfolio is the
        sole peak owner (A-05): this engine keeps no peak of its own, so
        WP1.8's ``from_fills(peak_equity_hint=...)`` resume seed is never
        overwritten by a stale exchange balance. WP4.1 removes this
        method once the no-source path itself is retired.

        Returns
        -------
        Decimal:
            Peak equity (highest equity seen since engine start).
        """
        current = await self._fetch_equity()
        # If peak is 0 (not yet seeded), initialize to current equity
        if self._peak_equity <= Decimal("0"):
            self._peak_equity = current
        elif current > self._peak_equity:
            self._peak_equity = current
        return self._peak_equity

    def _equity_snapshot(self, symbol: str, last: Decimal) -> EquitySnapshot | None:
        """WP1.4 (S-01/A-01/A-02, D5): a read-only NAV/peak/basis snapshot
        for a BUY on ``symbol``, computed entirely from the attached
        ``LivePositionSource`` -- never the exchange balance (I-1).

        ``nav`` is the source's own ``current_equity`` adjusted for
        ``symbol`` alone, substituting the fresh ticker ``last`` for that
        one position's (possibly one-bar-stale) mark; every other held
        position's mark is exactly what ``current_equity`` already used,
        refreshed at the start of the current bar (``strategy_engine.py``
        marks before any signal fires).

        Returns ``None`` ("NAV unavailable") when:

        - any OTHER held run position's mark is <= 0 (a stale/invalid
          mark makes the source's ``current_equity`` itself untrustworthy
          -- R-08), or
        - the resulting NAV is <= 0.

        The caller blocks the BUY with ``live.nav_unavailable`` in either
        case. Never mutates the source (I-6): only reads ``current_equity``,
        ``get_position``, ``get_open_positions``, ``cash``, ``initial_cash``
        and ``get_peak_equity()``.
        """
        source = self._position_source
        assert source is not None, "_equity_snapshot requires an attached source"

        for pos in source.get_open_positions():
            if pos.symbol != symbol and pos.current_price <= Decimal("0"):
                return None

        own_qty = self._own_quantity(symbol)
        position = source.get_position(symbol)
        adjustment = Decimal("0")
        if position is not None and own_qty > Decimal("0"):
            adjustment = own_qty * (last - position.current_price)

        nav = source.current_equity + adjustment
        if nav <= Decimal("0"):
            return None

        peak = max(source.get_peak_equity(), nav)
        return EquitySnapshot(
            nav=nav,
            peak=peak,
            cash=source.cash,
            sizing_basis=self._sizing_basis(nav, source.initial_cash),
        )

    def _calculate_daily_pnl(self) -> Decimal:
        """
        Calculate daily PnL.

        WP1.1 (WP11-A-04): delegates to the attached position source's
        ``get_daily_pnl()`` when present -- otherwise the local
        ``_positions`` cache's ``realised_pnl`` sum is always 0 (nothing
        ever wrote it), which would make the daily-loss gate permanently
        inert. Falls back to the legacy local-tracking sum when no source
        is attached (D8).

        Returns
        -------
        Decimal:
            Net daily PnL in quote currency.
        """
        if self._position_source is not None:
            return self._position_source.get_daily_pnl()

        total_pnl = Decimal("0")
        for pos in self._positions.values():
            total_pnl += pos.realised_pnl
        return total_pnl

    # ------------------------------------------------------------------
    # Position sync
    # ------------------------------------------------------------------

    async def sync_positions(self) -> dict[str, Position]:
        """
        Synchronize the run's own symbols against a fresh exchange balance.

        WP1.1 (WP11-A-06, rewritten for C1/C22): this NEVER creates a
        Position from the exchange balance (I2) -- own quantity comes only
        from the attached position source's own fills. It only:

        - clears a stale ``balance_unavailable`` flag on success (I4);
        - flags ``reconcile_required`` when own quantity exceeds the
          exchange's reported total beyond the dust tolerance (I8), or when
          a run symbol's market is missing (I6);
        - logs (but never flags) when the exchange holds more than the
          bot's own tracked quantity -- external holdings are left alone
          (I5), and refreshes the legacy ``_positions`` cache from the
          source (D1).

        Should be called on startup (``on_start``, WP11-A-07) and may be
        called periodically; without a position source attached this is a
        no-op over the (possibly empty) legacy ``_positions`` cache (D8).

        Returns
        -------
        dict[str, Position]:
            A copy of the (unchanged) local position cache, keyed by symbol.
        """
        if not self._run_symbols:
            self._log.info(
                "live.sync_positions_completed", active_positions=0, total_tracked=0
            )
            return dict(self._positions)

        balance = await self._fetch_balance_cached(fresh=True)
        if balance is None:
            for symbol in self._run_symbols:
                self._flag_reconcile(symbol, "balance_unavailable")
            self._log.warning(
                "live.sync_positions_balance_unavailable",
                symbols=list(self._run_symbols),
            )
            return dict(self._positions)

        markets: dict[str, Any] = getattr(self._exchange, "markets", {}) or {}
        total_balances: dict[str, Any] = balance.get("total", {}) or {}
        active = 0

        for symbol in self._run_symbols:
            self._maybe_clear_balance_unavailable(symbol)
            self._sync_positions_cache(symbol)

            own = self._own_quantity(symbol)
            if own > Decimal("0"):
                active += 1

            market = markets.get(symbol)
            base = market.get("base") if market else None
            if market is None or not base:
                self._flag_reconcile(symbol, "market_missing")
                continue

            # WP11-S-07: an unparseable balance entry is treated as
            # unavailable, never coerced to zero (which would look like an
            # "own > total" mismatch or a fabricated external holding).
            exch_total = _safe_decimal(total_balances.get(base), default=None)
            if exch_total is None:
                if own > Decimal("0"):
                    self._flag_reconcile(symbol, "own_exceeds_exchange_total")
                continue

            tol = self._amount_tolerance(symbol)
            if own > exch_total + tol:
                self._flag_reconcile(symbol, "own_exceeds_exchange_total")
            elif exch_total > own + tol:
                self._log.info(
                    "live.external_holdings_ignored",
                    symbol=symbol,
                    own=str(own),
                    exchange_total=str(exch_total),
                )

        self._log.info(
            "live.sync_positions_completed",
            active_positions=active,
            total_tracked=len(self._run_symbols),
        )
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """
        Initialize the live trading engine.

        Validates exchange connectivity and loads initial state.

        WP1.1 (WP11-A-07/D8, hardened round 2 S-09): runs ``sync_positions()``
        last, after markets are loaded.

        WP1.4 (S-04/S-05): once markets are loaded, checks that every run
        symbol shares one quote currency (blocking BUYs run-wide,
        ``live.quote_mismatch``, if not) and warns (never blocks)
        ``live.initial_capital_exceeds_free_quote`` when the account's
        free quote balance is below ``initial_capital``. No peak is
        seeded here any more -- the portfolio is the sole peak owner
        (A-05); see ``_fetch_equity``'s docstring.

        **Fails closed (S-09):** if live trading is enabled and no
        ``LivePositionSource`` is attached, logs
        ``live.position_source_missing`` at critical level and raises
        ``RuntimeError`` -- a real live run must never fall back to the
        legacy (and never fill-populated) ``_positions`` cache; that
        fallback exists purely so unit tests that never call ``on_start()``
        can inject ``_positions`` directly (D8/S03).

        Raises
        ------
        RuntimeError
            If live trading is enabled and no position source is attached,
            or if ``load_markets`` fails.
        """
        await super().on_start()

        if not self._enable_live_trading:
            self._log.warning(
                "live.engine_started_disabled",
                msg="Live trading gate is OFF. Orders will be rejected.",
            )
            return

        if self._position_source is None:
            self._log.critical(
                "live.position_source_missing",
                msg=(
                    "No LivePositionSource attached; refusing to start live "
                    "trading (WP1.1 S-09 fail-closed). Call "
                    "attach_position_source() before on_start()."
                ),
            )
            raise RuntimeError(
                "LiveExecutionEngine.on_start(): live trading is enabled but "
                "no LivePositionSource is attached. This is a fail-closed "
                "safety gate (Verbeterplan v2 WP1.1, finding WP11-S-09) -- "
                "call attach_position_source() (StrategyEngine does this "
                "automatically) before starting a live run."
            )

        try:
            # Load markets for symbol validation and minimum order size checks
            await ccxt_retry(
                self._exchange.load_markets,
                max_retries=3, base_delay=2.0, operation="load_markets",
            )
            # WP1.4 (S-03/A-05): no peak seed here -- the portfolio is the
            # sole peak owner once a source is attached (guaranteed at this
            # point: the S-09 gate above already raised otherwise), so
            # WP1.8's from_fills(peak_equity_hint=...) resume seed is never
            # stomped by a stale exchange balance. See _fetch_equity's and
            # _fetch_peak_equity's docstrings for the no-source path this
            # replaces.
            if self._position_source is not None:
                # WP1.4 (S-04/R-09): every run symbol must share one quote
                # currency -- the BUY affordability cap (I-4) reads a
                # single free[quote] balance, so a mixed-quote run would
                # cap against the wrong currency. Blocks BUYs for the rest
                # of the run; never self-heals (unlike reconcile_required).
                quotes: set[str] = set()
                missing_quote = False
                for run_symbol in self._run_symbols:
                    q = self._quote_asset(run_symbol)
                    if q is None:
                        missing_quote = True
                    else:
                        quotes.add(q)
                if missing_quote or len(quotes) > 1:
                    self._run_buy_block = "quote_mismatch"
                    self._log.warning(
                        "live.quote_mismatch",
                        symbols=list(self._run_symbols),
                        quotes=sorted(quotes),
                    )
                else:
                    # WP1.4 (S-05/A-07): informational only -- never raises
                    # or blocks (_fetch_balance_cached already fails soft).
                    quote = next(iter(quotes), None)
                    free_quote: Decimal | None = None
                    if quote is not None:
                        balance = await self._fetch_balance_cached(fresh=True)
                        if balance is not None:
                            free_quote = _safe_decimal(
                                (balance.get("free") or {}).get(quote), default=None
                            )
                    initial_cash = self._position_source.initial_cash
                    if free_quote is None or free_quote < initial_cash:
                        self._log.warning(
                            "live.initial_capital_exceeds_free_quote",
                            initial_cash=str(initial_cash),
                            free_quote=str(free_quote) if free_quote is not None else None,
                            quote=quote,
                        )
            self._log.info(
                "live.engine_started",
                exchange=self._exchange.id,
                markets_loaded=len(self._exchange.markets),
            )
        except Exception as exc:
            self._log.error(
                "live.engine_start_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )
            raise

        await self.sync_positions()

    async def on_stop(self) -> None:
        """
        Shut down the live trading engine.

        Cancels open orders and closes the exchange connection.
        """
        # Cancel all open orders directly via exchange (bypass live gate for shutdown)
        open_orders = self.get_open_orders()
        cancel_count = 0
        for order in open_orders:
            try:
                exchange_order_id = self._exchange_order_map.get(order.order_id)
                if exchange_order_id:
                    await ccxt_retry(
                        self._exchange.cancel_order,
                        exchange_order_id,
                        order.symbol,
                        max_retries=1, base_delay=0.5,
                        operation=f"shutdown_cancel({order.symbol})",
                    )
                self._transition(order, OrderStatus.CANCELED)
                cancel_count += 1
            except (ccxt_async.OrderNotFound, ccxt_async.InvalidOrder):
                self._log.debug(
                    "live.cancel_already_terminal_on_stop",
                    order_id=str(order.order_id),
                )
            except Exception:
                self._log.warning(
                    "live.cancel_failed_on_stop",
                    order_id=str(order.order_id),
                )

        # Close exchange connection
        try:
            # Graceful exchange connection shutdown
            if hasattr(self._exchange, "close"):
                await self._exchange.close()
        except Exception as exc:
            self._log.warning(
                "live.exchange_close_failed",
                error=str(exc),
                user_message=translate_ccxt_error(exc),
            )

        self._log.info(
            "live.engine_stopped",
            total_orders=len(self._orders),
            total_fills=sum(len(f) for f in self._fills.values()),
            canceled_on_stop=cancel_count,
        )

        await super().on_stop()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LiveExecutionEngine("
            f"run_id={self._run_id!r}, "
            f"exchange={getattr(self._exchange, 'id', 'unknown')!r}, "
            f"live_enabled={self._enable_live_trading}, "
            f"orders={len(self._orders)})"
        )
