"""
apps/api/services/run_recovery.py
-----------------------------------
Live-resume preparation, the real exchange scan-and-import (WP1.8b,
Verbeterplan v2 synthesis spec S2/S3), and the orphan-holding alert
repeater (WP1.8a synthesis spec §3-§8).

Public API
----------
- :class:`ImportReport` -- the result of a :func:`scan_and_import` pass:
  counts of cancelled/imported orders and imported fills.
- :func:`scan_and_import` -- **WP1.8b real implementation.** For each of
  the run's configured symbols, fetches every exchange order placed since
  the run started (``fetch_orders(since=started_at, paginate)``), keeps
  only the ones whose ``clientOrderId`` carries this run's exact
  ``f"{run_id}-{12 hex chars}"`` shape (WP18b-S-05: a prefix-only match is
  treated as foreign, never cancelled/imported), cancels every one still
  open and polls it to a terminal ccxt status (30s timeout, P-01: an
  already-cancelled/filled order is handled gracefully), then reconciles
  each matched order against the DB: an order with zero persisted fills
  and a non-zero exchange fill is imported whole (order row + every parsed
  fill, one DB transaction per order -- O3, dedup by ``client_order_id``,
  never adopted); an order that already has persisted fills is only ever
  verified for consistency, never re-imported (idempotent re-run). Any
  scan/cancel/trade-fetch failure, a DB error mid-import, or a persisted
  order missing from the scan entirely, raises ``ResumeRejected`` --
  fail-closed, never overridable (S11/U3).
- :func:`prepare_live_resume` -- loads the persisted fill/order history,
  validates it (O10/R-06), invokes :func:`scan_and_import`, then reloads
  and re-validates before returning the ``ResumeSnapshot`` the live engine
  replays from. Raises ``ResumeRejected`` on any failure.
- :func:`orphan_holding_repeater` -- background task (registered on
  ``AppContainer.background_tasks``) that repeats a critical alert every
  15 minutes for every live run sitting ``orphaned`` with a non-zero last
  position snapshot (S8, minimum alerting), and logs critical once per
  cycle for every live run stuck ``resuming`` for more than 10 minutes
  (WP18b-S-04 -- a resume whose request was cancelled mid-flight, in a
  process that keeps running rather than rebooting).

WP1.8a -> WP1.8b -> WP1.8b round 2
-----------------------------------
1.8a shipped :func:`scan_and_import` as a fail-closed stub. 1.8b round 1
replaced it with the real implementation described above. Round 2
(WP18b-S-01/S-02/S-04/S-05, security re-audit) closes an ABA race on the
``resuming`` status: every per-order DB write now happens inside a
**fenced** short transaction -- ``SELECT ... FROM runs WHERE id=:id AND
status='resuming' AND config->>'resume_attempt_id'=:fence FOR SHARE`` --
so a resume attempt that the kill switch (or a second resume) has already
superseded can no longer import anything, even though its own in-flight
``scan_and_import`` call has no way to know that without re-checking. The
fence lives in the JSONB ``config`` column, not ``runs.updated_at`` -- a
``BEFORE UPDATE`` trigger (migration 001) unconditionally overwrites
``updated_at`` with the database's own ``now()`` on every UPDATE, so an
application-supplied value there can never be compared back for equality.
``_reject`` (in ``apps.api.routers.runs``) now rolls back before
reverting, so a half finished per-order write can never be silently
committed, and a DB error mid-import is caught and converted to a 409
(``order_import_failed``) instead of a raw 500.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import FillORM, OrderORM, RunORM
from api.services.audit_log import record_audit_event
from api.services.run_persistence import load_resume_snapshot
from common.types import OrderSide, OrderStatus, OrderType
from trading.engines.live import _parse_ccxt_trades
from trading.models import Order
from trading.recovery import ResumeRejected, ResumeSnapshot, check_fill_integrity

__all__ = [
    "CANCEL_POLL_INTERVAL_SECONDS",
    "CANCEL_POLL_TIMEOUT_SECONDS",
    "ORPHAN_REPEAT_INTERVAL_SECONDS",
    "RESUMING_STUCK_ALERT_SECONDS",
    "ImportReport",
    "orphan_holding_repeater",
    "prepare_live_resume",
    "scan_and_import",
]

logger = structlog.get_logger(__name__)

#: S8 minimum alerting cadence -- repeat the critical alert every 15
#: minutes while a live run stays orphaned with an unprotected position.
ORPHAN_REPEAT_INTERVAL_SECONDS: float = 900.0

#: WP18b-S-04: a live run sitting 'resuming' longer than this is stuck
#: (its resume request was cancelled/crashed without reverting) --
#: alert every repeater cycle once this threshold is crossed.
RESUMING_STUCK_ALERT_SECONDS: float = 600.0

#: S2/S3: cancel-then-poll-to-terminal timeout for an open prefixed order
#: found during the exchange scan. Module-level (not a function
#: parameter) so a test can monkeypatch it down for a fast, deterministic
#: timeout test without threading a kwarg through
#: ``prepare_live_resume``'s public signature.
CANCEL_POLL_TIMEOUT_SECONDS: float = 30.0
CANCEL_POLL_INTERVAL_SECONDS: float = 0.5

#: WP18b-S-06: both the primary and supplementary exchange scans widen
#: their ``since`` anchor by this margin to absorb clock skew between this
#: process and the exchange.
_SCAN_CLOCK_SKEW_MARGIN = timedelta(minutes=5)

#: Same fallback tolerance convention as ``trading.recovery`` (one
#: satoshi-scale unit when the market's own amount step is unavailable).
_DEFAULT_TOLERANCE = Decimal("0.00000001")

_TERMINAL_CCXT_STATUSES = frozenset({"closed", "canceled", "expired", "rejected"})

_ORDER_TYPE_FROM_CCXT: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
}
_ORDER_STATUS_FROM_CCXT: dict[str, OrderStatus] = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}

#: WP18b-S-05: the engine only ever mints ``f"{run_id}-{uuid4().hex[:12]}"``
#: -- a manually-crafted order using our prefix but NOT this exact shape
#: (e.g. ``f"{run_id}-SPOOFED-by-manual-api-call"``) must be treated as
#: foreign, never cancelled or imported (D15 defence in depth).
_CLIENT_ORDER_ID_SUFFIX_RE = r"[0-9a-f]{12}"


def _client_order_id_pattern(prefix: str) -> re.Pattern[str]:
    return re.compile(re.escape(prefix) + _CLIENT_ORDER_ID_SUFFIX_RE)


def _log_exc(log: Any, event: str, exc: Exception, **fields: Any) -> None:  # noqa: ANN401
    """WP18b-S-07: log the exception TYPE plus a truncated message, never
    the raw ``str(exc)`` unbounded -- a signed ccxt GET request's URL can
    appear in some adapters' error messages (never the key/secret itself,
    but still needlessly verbose/potentially sensitive query data)."""
    log.error(event, error_type=type(exc).__name__, error=str(exc)[:200], **fields)


@dataclass(frozen=True)
class ImportReport:
    """Result of a :func:`scan_and_import` pass (WP1.8b)."""

    orders_cancelled: int = 0
    orders_imported: int = 0
    fills_imported: int = 0
    foreign_orders_seen: int = 0


def _decimal_or_zero(value: Any) -> Decimal:  # noqa: ANN401
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


async def _fetch_prefixed_orders(
    exchange: Any,  # noqa: ANN401
    symbol: str,
    *,
    since_ms: int,
    prefix: str,
    log: Any,  # noqa: ANN401
) -> tuple[dict[str, dict[str, Any]], int]:
    """Fetch every order for ``symbol`` since ``since_ms`` (S3) and split it
    into (matched-by-exact-shape dict keyed by ``clientOrderId``, count of
    foreign orders left untouched -- P-04/spec item 1).

    WP18b-S-05: "matched" requires the FULL ``f"{prefix}{12 hex chars}"``
    shape, not merely ``startswith(prefix)`` -- a prefix-only match (e.g. a
    manually crafted ``clientOrderId``) is foreign too and is counted (with
    a warning logged, count only -- never the spoofed id itself) alongside
    every other foreign order.
    """
    try:
        raw_orders = await exchange.fetch_orders(symbol, since=since_ms, params={"paginate": True})
    except Exception as exc:
        _log_exc(log, "recovery.scan_fetch_orders_failed", exc, symbol=symbol)
        raise ResumeRejected("exchange_scan_failed") from exc

    pattern = _client_order_id_pattern(prefix)
    matched: dict[str, dict[str, Any]] = {}
    foreign_count = 0
    spoofed_prefix_count = 0
    for raw in raw_orders or []:
        client_order_id = raw.get("clientOrderId")
        cid = str(client_order_id) if client_order_id else ""
        if cid and pattern.fullmatch(cid):
            matched[cid] = raw
            continue
        foreign_count += 1
        if cid.startswith(prefix):
            spoofed_prefix_count += 1

    if spoofed_prefix_count:
        log.warning(
            "recovery.scan_prefix_spoof_suspected",
            symbol=symbol,
            count=spoofed_prefix_count,
        )
    return matched, foreign_count


async def _cancel_and_await_terminal(
    exchange: Any,  # noqa: ANN401
    raw_order: dict[str, Any],
    symbol: str,
    log: Any,  # noqa: ANN401
) -> tuple[dict[str, Any], bool]:
    """S2: cancel an open prefixed order then poll it to a terminal ccxt
    status (30s timeout). Returns ``(final_raw_order, did_cancel)``.

    WP18b-C-03: ANY exception from ``cancel_order`` (not just
    ``ccxt.OrderNotFound``/``InvalidOrder``, P-01) falls through to the
    poll below -- the poll's own ``fetch_order`` is the actual source of
    truth for whether the order reached a terminal state, so failing
    before even trying it would reject a resume for a cancel race that, on
    some exchanges' ccxt adapters, raises a different exception class than
    the two special-cased here. This stays fail-closed: the poll's own
    timeout/fetch failure below still raises.
    """
    exchange_order_id = raw_order.get("id")
    did_cancel = False
    if raw_order.get("status") not in _TERMINAL_CCXT_STATUSES:
        try:
            await exchange.cancel_order(exchange_order_id, symbol)
            did_cancel = True
        except Exception as exc:
            log.info(
                "recovery.scan_cancel_exception_polling_anyway",
                exchange_order_id=exchange_order_id,
                symbol=symbol,
                error_type=type(exc).__name__,
            )

    deadline = time.monotonic() + CANCEL_POLL_TIMEOUT_SECONDS
    fetched = raw_order
    while fetched.get("status") not in _TERMINAL_CCXT_STATUSES:
        if time.monotonic() >= deadline:
            log.error(
                "recovery.scan_cancel_poll_timeout",
                exchange_order_id=exchange_order_id,
                symbol=symbol,
            )
            raise ResumeRejected("order_cancel_timeout")
        await asyncio.sleep(CANCEL_POLL_INTERVAL_SECONDS)
        try:
            fetched = await exchange.fetch_order(exchange_order_id, symbol)
        except Exception as exc:
            _log_exc(
                log,
                "recovery.scan_cancel_poll_failed",
                exc,
                exchange_order_id=exchange_order_id,
                symbol=symbol,
            )
            raise ResumeRejected("order_cancel_failed") from exc
    return fetched, did_cancel


async def _fetch_order_trades(
    exchange: Any,  # noqa: ANN401
    *,
    exchange_order_id: str,
    symbol: str,
    log: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Fetch every trade for one order, via the same
    ``has['fetchOrderTrades']`` fallback ``LiveExecutionEngine.get_fills``
    uses (S3) -- a resumed run's imported fills are fetched identically to
    a live reconcile."""
    try:
        if exchange.has.get("fetchOrderTrades"):
            trades = await exchange.fetch_order_trades(id=exchange_order_id, symbol=symbol)
        else:
            all_trades = await exchange.fetch_my_trades(symbol=symbol)
            trades = [t for t in all_trades if t.get("order") == exchange_order_id]
    except Exception as exc:
        _log_exc(
            log,
            "recovery.scan_trade_fetch_failed",
            exc,
            exchange_order_id=exchange_order_id,
            symbol=symbol,
        )
        raise ResumeRejected("trade_fetch_failed") from exc
    return list(trades)


def _build_pseudo_order(
    *,
    order_id: Any,  # noqa: ANN401 -- UUID, kept loose to avoid importing uuid.UUID twice
    client_order_id: str,
    run_id: str,
    raw_order: dict[str, Any],
) -> Order:
    """Build a validated (pure) ``Order`` from a raw ccxt order dict.

    ``order_id`` here is a throwaway placeholder (WP18b-S-01) -- the REAL
    persisted id is decided later, inside the fenced per-order transaction,
    from a fresh re-read of the DB. This function is only used to
    normalise the raw ccxt fields (symbol/side/quantity/status/timestamps)
    through the same validated ``Order`` model the rest of the codebase
    uses; nothing here is written to the DB directly.

    Raising ``ResumeRejected("exchange_scan_failed")`` on a
    ``ValidationError``/``ValueError`` is deliberate -- a malformed
    exchange response (bad enum, ``filled > amount``, ...) must fail this
    resume closed exactly like every other scan failure (U3), not crash
    with a raw exception type.
    """
    raw_type = str(raw_order.get("type") or "market").lower()
    order_type = _ORDER_TYPE_FROM_CCXT.get(raw_type, OrderType.MARKET)
    raw_price = raw_order.get("price")
    price = (
        Decimal(str(raw_price))
        if (order_type == OrderType.LIMIT and raw_price is not None)
        else None
    )

    raw_ts = raw_order.get("timestamp")
    created_at = datetime.fromtimestamp(raw_ts / 1000, tz=UTC) if raw_ts else datetime.now(tz=UTC)

    try:
        return Order(
            order_id=order_id,
            client_order_id=client_order_id,
            run_id=run_id,
            symbol=str(raw_order.get("symbol")),
            side=OrderSide(str(raw_order.get("side") or "").lower()),
            order_type=order_type,
            quantity=_decimal_or_zero(raw_order.get("amount")) or Decimal("0.00000001"),
            price=price,
            status=_ORDER_STATUS_FROM_CCXT.get(
                str(raw_order.get("status") or "").lower(), OrderStatus.OPEN
            ),
            filled_quantity=_decimal_or_zero(raw_order.get("filled")),
            average_fill_price=(
                Decimal(str(raw_order["average"])) if raw_order.get("average") is not None else None
            ),
            exchange_order_id=str(raw_order.get("id")) if raw_order.get("id") is not None else None,
            created_at=created_at,
            updated_at=created_at,
        )
    except Exception as exc:
        raise ResumeRejected("exchange_scan_failed") from exc


async def scan_and_import(
    db: AsyncSession,
    run: RunORM,
    exchange: Any,  # noqa: ANN401
    *,
    fence: str,
) -> ImportReport:
    """
    Scan the exchange for orders placed under ``run``'s ``clientOrderId``
    prefix and import anything missing from the DB (WP1.8b S2/S3).

    See the module docstring for the full algorithm. Every DB write here
    happens in its OWN short, FENCED transaction (one ``db.commit()`` per
    scanned order, plus a final one for the summary audit row) -- this
    function never holds the run row's lock for longer than one order's
    worth of pure-DB work (no network I/O between the fence check and that
    transaction's commit/rollback, WP18b-S-01).

    Parameters
    ----------
    fence:
        The ``resume_attempt_id`` (a UUID string) the caller's own
        ``orphaned -> resuming`` transaction (a) generated and embedded in
        ``run.config`` before committing. Every per-order transaction
        re-checks ``status='resuming' AND config->>'resume_attempt_id' =
        fence`` before writing anything -- if a concurrent kill-switch,
        stop, or a second resume attempt has since moved/re-cycled the
        row, this fence check fails and the whole scan aborts with
        ``resume_state_lost`` instead of importing under a superseded
        attempt's authority (the ABA race WP18b-S-01 closes).

        NOTE: the fence is a JSONB ``config`` field, NOT ``runs.updated_at``
        -- a ``BEFORE UPDATE`` trigger (migration 001,
        ``trigger_set_updated_at()``) unconditionally overwrites
        ``updated_at`` with the database's own ``now()`` on every UPDATE,
        so an application-supplied ``updated_at`` value can never be
        compared back for equality after a round trip. ``config`` has no
        such trigger.

    Raises
    ------
    ResumeRejected
        On any scan/cancel/trade-fetch failure, a persisted order missing
        from the scan (``"exchange_scan_incomplete"``), an under/over
        -counted persisted fill history
        (``"fill_history_partial"``/``"fill_history_corrupt"``), a lost
        fence (``"resume_state_lost"``), or a DB error mid-import
        (``"order_import_failed"``). Never overridable (S11/U3).
    """
    log = logger.bind(component="scan_and_import", run_id=str(run.id))
    symbols = [str(s) for s in ((run.config or {}).get("symbols") or [])]
    prefix = f"{run.id}-"
    since_ms = int((run.started_at - _SCAN_CLOCK_SKEW_MARGIN).timestamp() * 1000)

    try:
        await exchange.load_markets()
    except Exception as exc:
        _log_exc(log, "recovery.scan_load_markets_failed", exc)
        raise ResumeRejected("exchange_scan_failed") from exc

    # Persisted state, read ONCE before the exchange scan -- used only for
    # the upfront S3 completeness check and to anchor the WP18b-S-06
    # supplementary fetch below. The per-order AUTHORITATIVE state is
    # always re-read fresh inside each order's fenced transaction further
    # down (never trusted from this snapshot).
    order_rows_result = await db.execute(select(OrderORM).where(OrderORM.run_id == run.id))
    persisted_orders = list(order_rows_result.scalars().all())
    persisted_client_order_ids = {o.client_order_id for o in persisted_orders}

    # WP18b-S-06: a supplementary fetch anchored on the latest persisted
    # order's own timestamp, to guard against ccxt's `paginate` cap
    # truncating the newest orders out of the primary (run-start-anchored)
    # fetch on a long-lived run.
    since_2_ms: int | None = None
    if persisted_orders:
        latest_created_at = max(o.created_at for o in persisted_orders)
        since_2_ms = int((latest_created_at - _SCAN_CLOCK_SKEW_MARGIN).timestamp() * 1000)

    scanned_by_client_id: dict[str, dict[str, Any]] = {}
    foreign_count = 0
    for symbol in symbols:
        matched, foreign = await _fetch_prefixed_orders(
            exchange, symbol, since_ms=since_ms, prefix=prefix, log=log
        )
        foreign_count += foreign
        scanned_by_client_id.update(matched)

        if since_2_ms is not None and since_2_ms > since_ms:
            matched_2, foreign_2 = await _fetch_prefixed_orders(
                exchange, symbol, since_ms=since_2_ms, prefix=prefix, log=log
            )
            foreign_count += foreign_2
            scanned_by_client_id.update(matched_2)

    if foreign_count:
        log.info("recovery.scan_foreign_orders_untouched", count=foreign_count)

    # S3: a persisted order missing from the scan entirely -> fail closed
    # BEFORE acting on anything else in this pass.
    for client_order_id in persisted_client_order_ids:
        if client_order_id not in scanned_by_client_id:
            log.error("recovery.scan_persisted_order_missing", client_order_id=client_order_id)
            raise ResumeRejected("exchange_scan_incomplete")

    orders_cancelled = 0
    orders_imported = 0
    fills_imported = 0

    for client_order_id, raw_order in scanned_by_client_id.items():
        symbol = str(raw_order.get("symbol"))
        final_raw_order, did_cancel = await _cancel_and_await_terminal(
            exchange, raw_order, symbol, log
        )
        if did_cancel:
            orders_cancelled += 1

        exchange_order_id = str(final_raw_order.get("id"))
        trades = await _fetch_order_trades(
            exchange, exchange_order_id=exchange_order_id, symbol=symbol, log=log
        )

        pseudo_order = _build_pseudo_order(
            order_id=uuid4(),
            client_order_id=client_order_id,
            run_id=str(run.id),
            raw_order=final_raw_order,
        )

        market = (getattr(exchange, "markets", None) or {}).get(symbol) or {}
        base_asset = market.get("base")
        quote_currency = market.get("quote") or "USD"

        new_fills, gross_delta, _keys = _parse_ccxt_trades(
            pseudo_order,
            trades,
            base_asset=base_asset,
            quote_currency=quote_currency,
            log=log,
        )

        gross_filled = pseudo_order.filled_quantity
        tolerance = _DEFAULT_TOLERANCE

        # WP18b-S-01/S-02: from here to the matching commit/rollback below,
        # NO network I/O happens -- only the fence check, a fresh re-read
        # of this order's persisted state, and (at most) one order row plus
        # its fills. The whole block is one short, fenced transaction.
        try:
            fence_row = await db.execute(
                select(RunORM.id)
                .where(
                    RunORM.id == run.id,
                    RunORM.status == "resuming",
                    RunORM.config["resume_attempt_id"].astext == fence,
                )
                .with_for_update(read=True)
            )
            if fence_row.scalar_one_or_none() is None:
                raise ResumeRejected("resume_state_lost")

            fresh_order_result = await db.execute(
                select(OrderORM).where(
                    OrderORM.run_id == run.id, OrderORM.client_order_id == client_order_id
                )
            )
            fresh_persisted_order = fresh_order_result.scalar_one_or_none()
            fresh_persisted_qty = Decimal("0")
            if fresh_persisted_order is not None:
                fresh_fill_result = await db.execute(
                    select(func.sum(FillORM.quantity)).where(
                        FillORM.order_id == fresh_persisted_order.id
                    )
                )
                fresh_persisted_qty = fresh_fill_result.scalar() or Decimal("0")

            if fresh_persisted_qty == Decimal("0"):
                if gross_filled <= Decimal("0"):
                    # Nothing filled, nothing persisted -- nothing to do.
                    await db.commit()
                    continue
                if abs(gross_delta - gross_filled) > tolerance:
                    # The trade fetch did not fully account for this
                    # order's own reported gross fill -- do not import a
                    # short count.
                    log.error(
                        "recovery.scan_import_gross_mismatch",
                        client_order_id=client_order_id,
                        gross_delta=str(gross_delta),
                        gross_filled=str(gross_filled),
                    )
                    raise ResumeRejected("exchange_scan_incomplete")

                if fresh_persisted_order is None:
                    order_row = OrderORM(
                        id=uuid4(),
                        client_order_id=client_order_id,
                        run_id=run.id,
                        symbol=pseudo_order.symbol,
                        side=pseudo_order.side.value,
                        order_type=pseudo_order.order_type.value,
                        quantity=pseudo_order.quantity,
                        price=pseudo_order.price,
                        status=pseudo_order.status.value,
                        filled_quantity=pseudo_order.filled_quantity,
                        average_fill_price=pseudo_order.average_fill_price,
                        exchange_order_id=pseudo_order.exchange_order_id,
                        created_at=pseudo_order.created_at,
                        updated_at=pseudo_order.updated_at,
                    )
                    db.add(order_row)
                    await db.flush()
                else:
                    order_row = fresh_persisted_order
                    order_row.status = pseudo_order.status.value
                    order_row.filled_quantity = pseudo_order.filled_quantity
                    order_row.average_fill_price = pseudo_order.average_fill_price
                    order_row.updated_at = pseudo_order.updated_at

                fill_rows = [
                    FillORM(
                        id=f.fill_id,
                        order_id=order_row.id,
                        symbol=f.symbol,
                        side=f.side.value,
                        quantity=f.quantity,
                        price=f.price,
                        fee=f.fee,
                        fee_currency=f.fee_currency,
                        is_maker=f.is_maker,
                        executed_at=f.executed_at,
                        expected_price=f.expected_price,
                        slippage_bps_realized=f.slippage_bps_realized,
                    )
                    for f in new_fills
                ]
                db.add_all(fill_rows)
                await db.commit()

                orders_imported += 1
                fills_imported += len(fill_rows)
                log.warning(
                    "recovery.scan_order_imported",
                    client_order_id=client_order_id,
                    symbol=symbol,
                    fills=len(fill_rows),
                )
            else:
                e_value = sum((f.quantity for f in new_fills), Decimal("0"))
                if fresh_persisted_qty < e_value - tolerance:
                    raise ResumeRejected("fill_history_partial")
                if fresh_persisted_qty > gross_filled + tolerance:
                    raise ResumeRejected("fill_history_corrupt")
                await db.commit()
        except ResumeRejected:
            await db.rollback()
            raise
        except SQLAlchemyError as exc:
            await db.rollback()
            _log_exc(
                log,
                "recovery.scan_order_import_db_error",
                exc,
                client_order_id=client_order_id,
            )
            raise ResumeRejected("order_import_failed") from exc

    report = ImportReport(
        orders_cancelled=orders_cancelled,
        orders_imported=orders_imported,
        fills_imported=fills_imported,
        foreign_orders_seen=foreign_count,
    )

    if orders_cancelled or orders_imported:
        try:
            await record_audit_event(
                db,
                event_type="resume_orders_imported",
                resource_type="run",
                resource_id=str(run.id),
                request=None,
                payload={
                    "orders_cancelled": orders_cancelled,
                    "orders_imported": orders_imported,
                    "fills_imported": fills_imported,
                    "foreign_orders_seen": foreign_count,
                },
            )
            await db.commit()
        except Exception:
            log.exception("recovery.scan_audit_write_failed")

    return report


async def prepare_live_resume(
    db: AsyncSession,
    run: RunORM,
    exchange: Any = None,  # noqa: ANN401
    *,
    fence: str,
) -> ResumeSnapshot:
    """
    Validate and reconcile a live run's exchange state, then return the
    ``ResumeSnapshot`` the resume endpoint replays into a fresh engine
    stack.

    Steps (WP18-R-05..07, S2/S3, S11):
    1. Load the persisted fill/order history and validate it (O10/R-06) --
       corrupt or partial history fails closed before any exchange call.
    2. Call :func:`scan_and_import` -- cancels every open prefixed order,
       polls each to terminal, and imports any order with zero persisted
       fills (WP1.8b), fenced against the caller's own resume attempt
       (WP18b-S-01).
    3. Reload and re-validate -- an import can only ever add rows, so a
       second, unconditional integrity check is cheap insurance.

    Parameters
    ----------
    fence:
        Forwarded verbatim to :func:`scan_and_import` -- see its docstring.

    Raises
    ------
    ResumeRejected
        On any validation failure or exchange-scan failure. Never
        overridable (S11) -- the only caller-visible outcome is a 409 with
        the ``reason`` code, the resume endpoint's own audit row, and a
        status rollback from ``resuming`` back to ``orphaned``.
    """
    symbols = set((run.config or {}).get("symbols") or [])

    snapshot = await load_resume_snapshot(db, run)
    check_fill_integrity(snapshot.fills, snapshot.orders, symbols=symbols)

    await scan_and_import(db, run, exchange, fence=fence)

    snapshot = await load_resume_snapshot(db, run)
    check_fill_integrity(snapshot.fills, snapshot.orders, symbols=symbols)
    return snapshot


async def orphan_holding_repeater(
    interval_seconds: float = ORPHAN_REPEAT_INTERVAL_SECONDS,
) -> None:
    """
    Repeat a critical alert every ``interval_seconds`` while a live run
    sits ``orphaned`` with a non-zero last position snapshot (S8, minimum
    alerting: structured logs + audit rows already fire once when the run
    is first orphaned -- see ``run_orchestrator``/``routers.runs`` -- this
    loop is the *ongoing* reminder for as long as nobody has resumed it).

    WP18b-S-04: also logs critical, every cycle, for every live run sitting
    ``resuming`` for longer than :data:`RESUMING_STUCK_ALERT_SECONDS` -- a
    resume whose HTTP request was cancelled (client disconnect, worker
    cancellation) mid-scan reverts via a shielded fenced rollback in
    ``resume_run``'s own ``except asyncio.CancelledError`` handler, but a
    process that keeps running (rather than rebooting through
    ``recover_orphaned_runs``) would otherwise have no alerting path at all
    for a row stuck at ``resuming``.

    Runs as a named ``AppContainer.background_tasks`` entry so
    ``container.shutdown()`` cancels it on API shutdown along with every
    other long-lived task (LS-004).  Telegram delivery and a Grafana panel
    are explicitly out of scope here (S8 gap) -- this is the structured-log
    floor, always on, that a log-shipping/alerting pipeline can page from.
    """
    from api.db.models import PositionSnapshotORM
    from api.db.session import get_session_factory

    log = logger.bind(component="orphan_repeater")
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                factory = get_session_factory()
                async with factory() as db:
                    now = datetime.now(tz=UTC)
                    result = await db.execute(
                        select(RunORM).where(
                            RunORM.status.in_(["orphaned", "resuming"]),
                            RunORM.run_mode == "live",
                        )
                    )
                    for run in result.scalars().all():
                        if run.status == "resuming":
                            age_seconds = (now - run.updated_at).total_seconds()
                            if age_seconds > RESUMING_STUCK_ALERT_SECONDS:
                                log.critical(
                                    "recovery.resume_stuck",
                                    run_id=str(run.id),
                                    age_seconds=age_seconds,
                                )
                            continue

                        snap_result = await db.execute(
                            select(PositionSnapshotORM).where(PositionSnapshotORM.run_id == run.id)
                        )
                        held_symbols = [
                            snap.symbol for snap in snap_result.scalars().all() if snap.quantity > 0
                        ]
                        if held_symbols:
                            log.critical(
                                "recovery.orphan_holding_unprotected",
                                run_id=str(run.id),
                                symbols=held_symbols,
                            )
            except Exception:
                log.warning("recovery.orphan_repeater_cycle_failed", exc_info=True)
    except asyncio.CancelledError:
        log.info("recovery.orphan_repeater_stopped")
