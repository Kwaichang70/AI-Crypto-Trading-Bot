"""WP2.6 P6a (SY-26-10/11/16/82..88): the history-cache wiring for backtest and optimize.

``runs._fetch_bars_for_backtest`` delegates here only when the caller passes ``now`` and
``OHLCV_CACHE_MODE`` is not ``off``; mode ``off`` never imports this module. Paper and live never
touch it (AST-pinned in ``tests/unit/test_wp26_api.py``): the live tail stays on
``fetch_ohlcv_range``.

* ``_Upstream`` adapts the helper's ``CCXTMarketDataService`` (built there, credentials
  unchanged, CF-26-20) to ``RangeUpstream`` in integer ms.
* ``fetch_history_bars`` fetches the series sequentially after one ``connect`` (SY-26-84), builds
  the engine bars from the canonical candles and the persisted ``ohlcv_data`` block (ints, strs,
  bools and nullable hashes only; server-written, never a request field).
* Every OHLCV error is mapped here, by type and ``.code`` only, to an ``HTTPException`` with
  ``detail={"code": ...}`` (SY-26-83), so none can reach ``create_run``'s ``except ValueError``.
  The pre-existing ``DataNotAvailableError`` (400) and ``MarketDataError`` (502) pass through to
  the helper's own ``except`` chain unchanged.
* The store runs on its own sessions (``_store_session_factory``), never the request session.

Evidence rule (DoD 6, CF-26-19, CF-26-66, QR-26P5-R2-03; applied in WP2.8): a run is
reproducible evidence only with ``mode = "require"``, ``store = "ok"``, every series
``fetch_complete`` true and ``volatile_tail_bars == 0``, and a re-run that reproduces
``data_hash``. A block with ``rounded_values > 0`` is not bit-comparable with a mode ``off``
run of the same request (SY-26-02 rounding): never compare runs across modes (CF-26-33).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Literal, Protocol

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.db.session import get_session_factory
from api.services.cost_profiles import MAX_BACKTEST_BARS, _as_utc
from api.services.ohlcv_store import SqlOhlcvStore
from common.models import OHLCVBar
from common.types import TimeFrame
from data import ohlcv_cache as oc
from data import ohlcv_integrity as leaf
from trading.strategy_engine import _TIMEFRAME_SECONDS

if TYPE_CHECKING:
    from data.services.ccxt_market_data import CCXTMarketDataService

__all__ = ["HistoryBars", "fetch_history_bars"]

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MS: Final = timedelta(milliseconds=1)
_INTERNAL: Final = (leaf.CODE_BAD_TYPE, leaf.CODE_BAD_DIGEST, leaf.CODE_NOT_CANONICAL)  # bugs
_Mode = Literal["read_through", "require"]
_Series = list[dict[str, object]]
# (data_hash, store, store_reason, rounded_values, divergent_bars) of the block
_Facts = tuple[str | None, str, str, int, int]


class _Log(oc.LogSink, Protocol):
    def info(self, event: str, /, **fields: object) -> None: ...


class HistoryBars(dict[str, list[OHLCVBar]]):
    """Bars by symbol plus the ``ohlcv_data`` block the routers persist (SY-26-16)."""

    ohlcv_data: dict[str, object] | None = None


def _store_session_factory() -> async_sessionmaker[AsyncSession]:
    """The store's own sessions (``hide_parameters=True`` engine); tests patch this seam."""
    return get_session_factory()


def _ms(moment: datetime) -> int:
    """Integer ms since the epoch; never ``timestamp() * 1000`` (CF-26-73)."""
    return (_as_utc(moment) - _EPOCH) // _MS


class _Upstream:
    """``RangeUpstream`` over ``CCXTMarketDataService`` for ONE timeframe, in integer ms."""

    def __init__(self, service: CCXTMarketDataService, timeframe: TimeFrame) -> None:
        self._svc, self._tf = service, timeframe

    async def connect(self) -> None:
        await self._svc.connect()

    def check_series(self, symbol: str, timeframe_s: int) -> None:
        self._svc.check_series(symbol, self._tf)

    async def fetch_range(
        self, symbol: str, timeframe_s: int, start_ms: int, end_ms: int
    ) -> oc.UpstreamRange:
        start, end = (_EPOCH + timedelta(milliseconds=ms) for ms in (start_ms, end_ms))
        got = await self._svc.fetch_ohlcv_range_ex(symbol, self._tf, start, end)
        rows = [
            oc.RawRow(_ms(b.timestamp), (b.open, b.high, b.low, b.close, b.volume))
            for b in got.bars
        ]
        return oc.UpstreamRange(rows, got.complete)


def _rejected(
    exc: oc.OhlcvRequireError | oc.OhlcvStoreError | leaf.OhlcvDataError,
    log: _Log,
) -> HTTPException:
    """Closed status and code for an OHLCV error: type and ``.code`` only, never the text."""
    if isinstance(exc, oc.OhlcvRequireError):
        unavailable = exc.code == "ohlcv_store_unavailable"
        http = status.HTTP_503_SERVICE_UNAVAILABLE if unavailable else 422
        code: str = exc.code
    elif type(exc) is leaf.OhlcvDataError and exc.code not in _INTERNAL:
        http, code = status.HTTP_422_UNPROCESSABLE_ENTITY, exc.code
    else:  # invariant, store error (cannot escape the source) or an internal code: a bug
        http, code = status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_ohlcv_invariant"
    log.warning("runs.backtest_ohlcv_rejected", code=exc.code, error_type=type(exc).__name__)
    return HTTPException(status_code=http, detail={"code": code})


def _line(
    symbol: str,
    bars: tuple[int, int, int],
    span: tuple[int, int],
    digest: str | None,
    fetched: tuple[str, int, bool, bool],
) -> dict[str, object]:
    """One series line; ``bars`` = (n, first_ms, last_ms) as delivered to the runner.

    Only ``data_hash``, ``n_bars``, ``from_ms``/``to_ms``, ``volatile_tail_bars`` and
    ``fetch_complete`` describe the data; ``source``, ``bypass_disjoint`` and the block counters
    describe this run's fetches (CF-26-72), so never compare runs by them.
    """
    return {
        "symbol": symbol,
        "n_bars": bars[0],
        "first_ms": bars[1],
        "last_ms": bars[2],
        "from_ms": span[0],
        "to_ms": max(span),
        "digest": digest,
        "source": fetched[0],
        "volatile_tail_bars": fetched[1],
        "fetch_complete": fetched[2],
        "bypass_disjoint": fetched[3],
    }


async def _cached(
    up: _Upstream,
    mode: _Mode,
    exchange: str,
    order: list[str],
    plan: tuple[TimeFrame, tuple[int, int], int],
    log: _Log,
) -> tuple[dict[str, list[OHLCVBar]], _Series, _Facts]:
    timeframe, window, now_ms = plan
    tf_s = _TIMEFRAME_SECONDS[timeframe]
    source = oc.HistoricalBarSource(
        up,
        SqlOhlcvStore(_store_session_factory()),
        exchange=exchange,
        now_ms=now_ms,
        mode=mode,
        log=log,
        offload=asyncio.to_thread,
        cap_bars=MAX_BACKTEST_BARS,
    )
    await source.connect()
    got = await source.get_ranges(order, tf_s, *window)
    try:
        data_hash = leaf.run_digest(r.summary for r in got)
    except leaf.OhlcvDataError as exc:  # the summaries come from the leaf: an invariant broke
        raise oc.OhlcvInvariantError(exc.code) from None
    ok = [r.store == "ok" for r in got]
    store = "ok" if all(ok) else ("partial" if any(ok) else "unavailable")
    reason = next((r.store_reason for r in got if r.store != "ok"), "none")
    bars = {
        s: [
            OHLCVBar(
                symbol=s,
                timeframe=timeframe,
                timestamp=_EPOCH + timedelta(milliseconds=c.open_ms),
                open=c.open_,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
            for c in r.bars
        ]
        for s, r in zip(order, got, strict=True)
    }
    lines = [
        _line(
            s,
            (len(r.bars), r.bars[0].open_ms, r.bars[-1].open_ms) if r.bars else (0, 0, 0),
            (r.from_ms, r.to_ms),
            r.summary.digest,
            (r.source, r.volatile_tail_bars, r.fetch_complete, r.bypass_disjoint),
        )
        for s, r in zip(order, got, strict=True)
    ]
    rounded, divergent = sum(r.rounded_values for r in got), sum(r.divergent_bars for r in got)
    return bars, lines, (data_hash, store, reason, rounded, divergent)


async def _bypass(
    service: CCXTMarketDataService,
    mode: _Mode,
    order: list[str],
    timeframe: TimeFrame,
    window: tuple[datetime, datetime],
) -> tuple[dict[str, list[OHLCVBar]], _Series, _Facts]:
    """A timeframe the cache does not hold (1w, SY-26-65): today's bars and a block without
    hashes. ``n_bars``/``first_ms``/``last_ms`` count the bars delivered to the runner, before
    its trim of the forming bar (QR-26P5-R2-02)."""
    await service.connect()
    for symbol in order:  # validation first: an unknown symbol keeps today's 400
        service.check_series(symbol, timeframe)
    if mode == "require":
        raise oc.OhlcvRequireError("ohlcv_timeframe_not_cached")
    fetches = (service.fetch_ohlcv_range_ex(s, timeframe, *window) for s in order)
    got = await asyncio.gather(*fetches)
    span = (_ms(window[0]), _ms(window[1]))
    lines = [
        _line(
            s,
            (len(g.bars), _ms(g.bars[0].timestamp), _ms(g.bars[-1].timestamp))
            if g.bars
            else (0, 0, 0),
            span,
            None,
            ("exchange", 0, g.complete, False),
        )
        for s, g in zip(order, got, strict=True)
    ]
    return (
        {s: g.bars for s, g in zip(order, got, strict=True)},
        lines,
        (None, "bypass", "timeframe_not_cached", 0, 0),
    )


async def fetch_history_bars(
    service: CCXTMarketDataService,
    mode: _Mode,
    exchange: str,
    symbols: list[str],
    timeframe: TimeFrame,
    window: tuple[datetime, datetime],
    now: datetime,
    log: _Log,
) -> HistoryBars:
    """Bars for ``symbols`` through the cache, plus the block; OHLCV errors become HTTP errors."""
    tf_s, order, span = (
        _TIMEFRAME_SECONDS[timeframe],
        sorted(set(symbols)),
        (_ms(window[0]), _ms(window[1])),
    )
    try:
        if span[1] <= span[0]:  # a sub-ms window: no bar can open in it (SEC-26P6A-01)
            raise leaf.OhlcvDataError(leaf.CODE_OUT_OF_RANGE)
        if tf_s in leaf.CACHEABLE_TIMEFRAMES_S:
            plan = (timeframe, span, _ms(now))
            bars, lines, facts = await _cached(
                _Upstream(service, timeframe), mode, exchange, order, plan, log
            )
        else:
            bars, lines, facts = await _bypass(service, mode, order, timeframe, window)
    except (oc.OhlcvRequireError, oc.OhlcvStoreError, leaf.OhlcvDataError) as exc:
        raise _rejected(exc, log) from None
    data_hash, store, reason, rounded, divergent = facts
    out = HistoryBars((s, bars[s]) for s in symbols)
    out.ohlcv_data = {
        "version": leaf.OHLCV_DATA_VERSION,
        "data_hash": data_hash,
        "mode": mode,
        "store": store,
        "store_reason": reason,
        "exchange": exchange,
        "timeframe_s": tf_s,
        "settle_bars": leaf.SETTLE_BARS,
        "rounded_values": rounded,
        "divergent_bars": divergent,
        "requested_from_ms": span[0],
        "requested_to_ms": span[1],
        "series": lines,
    }
    log.info(
        "runs.backtest_ohlcv_data", mode=mode, store=store, store_reason=reason, data_hash=data_hash
    )
    return out
