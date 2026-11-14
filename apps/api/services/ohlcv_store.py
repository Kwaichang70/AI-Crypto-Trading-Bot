"""
apps/api/services/ohlcv_store.py
--------------------------------
WP2.6 P4 (SY-26-06/07/09): the Postgres adapter of ``data.ohlcv_cache.OhlcvStore`` over
``ohlcv_bars`` / ``ohlcv_coverage`` (migration 022). ``ohlcv_history`` wires it (P6a).

* Own short transactions on sessions from ``session_factory``; never the request session.
  ``extend`` pins ``READ COMMITTED`` as its first statement (a REPEATABLE READ server or role
  default would fail concurrent writers of one series with 40001 on the coverage-row lock);
  ``read`` pins ``REPEATABLE READ, READ ONLY`` so the coverage and its bars are one snapshot.
* ``text()`` statements and bound parameters only. Identifiers are case-sensitive series keys
  and never reach SQL text: every ``text()`` argument is one string literal, and the module
  has no f-string or ``%`` (both AST-pinned); the timeouts are constant literals too.
* No UPDATE or DELETE on ``ohlcv_bars``; only the coverage row is updated (SY-26-09).
* Every bar passes ``prepare_extend`` before any I/O, so a CHECK or range SQLSTATE (23514,
  22003, 22001) aborting a whole batch is not expected; if it happens it maps to ``error``.
* Rows read back go through ``canonicalise_candle`` only; a failure is ``stored_row_invalid``.
  The raw rows are fetched inside the transaction; the conversion runs after it, in slices of at
  most ``_SLICE`` rows, on a worker thread above ``INLINE_MAX_ROWS`` rows (P6b, SY-26-53), as does
  the ``prepare_extend`` check of a large batch. No session or connection is held across a hop.
  A bad stored row is reported after every page is fetched, so a later-page SQL failure takes
  precedence (fail-open either way).
* ``fetched_at`` / ``verified_at`` are passed explicitly (never the ``now()`` default).
* Logs: ``exc_log_fields`` and numbers only. Never a DSN, SQL, parameter or value. The store
  logs a failure itself; callers must not log it a second time. The engine behind
  ``session_factory`` must use ``hide_parameters=True`` (as ``api.db.session`` does).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, TypeVar

import structlog
from sqlalchemy import DateTime, Numeric, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import BindParameter

from common.logging import exc_log_fields
from common.metrics import metrics
from data.ohlcv_cache import (
    INLINE_MAX_ROWS,
    Coverage,
    ExtendResult,
    OhlcvStoreError,
    SeriesKey,
    StoreCode,
    StoredRead,
    prepare_extend,
)
from data.ohlcv_integrity import (
    CACHEABLE_TIMEFRAMES_S,
    Candle,
    OhlcvDataError,
    canonicalise_candle,
    tf_ms,
)

__all__ = ["DIVERGENT_METRIC", "SqlOhlcvStore", "store_error_code"]

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")

DIVERGENT_METRIC: Final = "ohlcv_cache_divergent_bars_total"  # SY-26-09: the one metric name
_INGEST_VERSION: Final = 1
_PAGE: Final = 50_000  # keyset page of a read (DB-26-19)
_SLICE: Final = 10_000  # rows per thread hop when canonicalising or validating off the loop
_READ_TIMEOUT: Final = text("SET LOCAL statement_timeout = '30s'")
_WRITE_TIMEOUT: Final = text("SET LOCAL statement_timeout = '15s'")
_LOCK_TIMEOUT: Final = text("SET LOCAL lock_timeout = '5s'")
_WRITE_ISOLATION: Final = text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")  # DB-26P4-01
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MS: Final = timedelta(milliseconds=1)
_TF_LABEL: Final = dict(
    zip(CACHEABLE_TIMEFRAMES_S, ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"), strict=True)
)
_SQLSTATE_CODE: Final[dict[str, StoreCode]] = {
    "42P01": "missing_table",
    "42501": "denied",
    "57014": "timeout",  # statement_timeout
    "55P03": "timeout",  # lock_timeout
}

# Literal statements only (no concatenation, no interpolation): every value is a bound parameter.
_COVERAGE_SQL: Final = text(
    "SELECT covered_from, covered_to FROM ohlcv_coverage "
    "WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe"
)
_LOCK_SQL: Final = text(
    "SELECT covered_from, covered_to FROM ohlcv_coverage "
    "WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe FOR UPDATE"
)
_ANCHOR_SQL: Final = text(
    "INSERT INTO ohlcv_coverage (exchange, symbol, timeframe, covered_from, covered_to) "
    "VALUES (:exchange, :symbol, :timeframe, :anchor, :anchor) "
    "ON CONFLICT (exchange, symbol, timeframe) DO NOTHING"
)
# Zero rows = the span does not touch the coverage: the store raises coverage_disjoint. The DB
# would accept a shrinking UPDATE, so this text is pinned by a test (CF-26-44).
_UPDATE_SQL: Final = text(
    "UPDATE ohlcv_coverage SET covered_from = LEAST(covered_from, :a), "
    "covered_to = GREATEST(covered_to, :b), verified_at = :verified_at "
    "WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe "
    "AND :a <= covered_to AND :b >= covered_from "
    "RETURNING covered_from, covered_to"
)
_BARS_SQL: Final = text(
    "SELECT open_time, open, high, low, close, volume FROM ohlcv_bars "
    "WHERE exchange = :exchange AND symbol = :symbol AND timeframe = :timeframe "
    "AND open_time >= :lo AND open_time < :hi ORDER BY open_time LIMIT :page"
)


def _array_params() -> tuple[BindParameter[Any], ...]:
    return (
        bindparam("open_time", type_=ARRAY(DateTime(timezone=True))),
        *(bindparam(name, type_=ARRAY(Numeric())) for name in ("o", "h", "l", "c", "v")),
    )


# ON CONFLICT names exactly the primary key columns; in-batch duplicates are removed by
# prepare_extend, so rows_in_batch - inserted counts rows that were already stored.
_INSERT_SQL: Final = text(
    "INSERT INTO ohlcv_bars (exchange, symbol, timeframe, open_time, open, high, low, close, "
    "volume, fetched_at, ingest_version) "
    "SELECT CAST(:exchange AS VARCHAR(32)), CAST(:symbol AS VARCHAR(32)), "
    "CAST(:timeframe AS VARCHAR(4)), t.open_time, t.o, t.h, t.l, t.c, t.v, "
    "CAST(:fetched_at AS TIMESTAMPTZ), CAST(:ingest_version AS SMALLINT) "
    "FROM unnest(CAST(:open_time AS TIMESTAMPTZ[]), CAST(:o AS NUMERIC[]), "
    "CAST(:h AS NUMERIC[]), CAST(:l AS NUMERIC[]), CAST(:c AS NUMERIC[]), "
    "CAST(:v AS NUMERIC[])) AS t(open_time, o, h, l, c, v) ORDER BY t.open_time "
    "ON CONFLICT (exchange, symbol, timeframe, open_time) DO NOTHING"
).bindparams(*_array_params())
# Counts batch bars whose stored values differ; bars this transaction just inserted are equal.
_DIVERGENCE_SQL: Final = text(
    "SELECT count(*) FROM ohlcv_bars b JOIN "
    "unnest(CAST(:open_time AS TIMESTAMPTZ[]), CAST(:o AS NUMERIC[]), "
    "CAST(:h AS NUMERIC[]), CAST(:l AS NUMERIC[]), CAST(:c AS NUMERIC[]), "
    "CAST(:v AS NUMERIC[])) AS t(open_time, o, h, l, c, v) ON b.open_time = t.open_time "
    "WHERE b.exchange = :exchange AND b.symbol = :symbol AND b.timeframe = :timeframe "
    "AND (b.open IS DISTINCT FROM t.o OR b.high IS DISTINCT FROM t.h "
    "OR b.low IS DISTINCT FROM t.l OR b.close IS DISTINCT FROM t.c "
    "OR b.volume IS DISTINCT FROM t.v)"
).bindparams(*_array_params())


def _dt(ms: int) -> datetime:
    return _EPOCH + ms * _MS  # exact integer arithmetic, no float


def _ms(stamp: datetime) -> int:
    return (stamp - _EPOCH) // _MS


def _series(key: SeriesKey) -> dict[str, Any]:
    return {"exchange": key.exchange, "symbol": key.symbol, "timeframe": _TF_LABEL[key.timeframe_s]}


def _sqlstate(exc: BaseException) -> str | None:
    for err in (exc, getattr(exc, "orig", None), exc.__cause__):
        state = getattr(err, "sqlstate", None)
        if isinstance(state, str):
            return state
    return None


def store_error_code(exc: BaseException) -> StoreCode:
    """Closed store code for a raw error, by SQLSTATE and type only (never the message)."""
    state = _sqlstate(exc)
    if state in _SQLSTATE_CODE:
        return _SQLSTATE_CODE[state]
    if isinstance(exc, TimeoutError):
        return "timeout"
    if (
        isinstance(exc, OSError)
        or (state is not None and (state.startswith(("08", "57P"))))
        or (isinstance(exc, DBAPIError) and exc.connection_invalidated)
    ):
        return "unavailable"
    return "error"  # includes 23514 / 22003 / 22001: data errors are never retried (CF-26-40)


def _arrays(batch: Sequence[Candle]) -> dict[str, Any]:
    return {
        "open_time": [_dt(c.open_ms) for c in batch],
        "o": [c.open_ for c in batch],
        "h": [c.high for c in batch],
        "l": [c.low for c in batch],
        "c": [c.close for c in batch],
        "v": [c.volume for c in batch],
    }


async def _off_loop(rows: int, fn: Callable[..., _T], *args: object) -> _T:
    # inline up to INLINE_MAX_ROWS rows, else on a worker thread (SY-26-53)
    return fn(*args) if rows <= INLINE_MAX_ROWS else await asyncio.to_thread(fn, *args)


def _canon_rows(rows: Sequence[Any], tf_s: int) -> list[Candle]:
    try:
        return [canonicalise_candle(_ms(r[0]), *r[1:6], timeframe_s=tf_s)[0] for r in rows]
    except OhlcvDataError:
        raise OhlcvStoreError("stored_row_invalid") from None


async def _canon(raw: list[Any], tf_s: int) -> tuple[Candle, ...]:
    # the stored rows as candles, in order, in slices of at most _SLICE rows per thread hop
    out: list[Candle] = []
    total = len(raw)
    while raw:
        part = raw[:_SLICE]
        del raw[:_SLICE]  # free converted rows as we go: peak is raw + candles, not both in full
        out += await _off_loop(total, _canon_rows, part, tf_s)
    return tuple(out)


class SqlOhlcvStore:
    """Postgres ``OhlcvStore``. Raises ``OhlcvStoreError`` (closed code) or ``OhlcvDataError``."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def _wrap(self, op: str, exc: Exception) -> Exception:
        if isinstance(exc, OhlcvStoreError | OhlcvDataError):
            return exc  # already closed-vocabulary
        code = store_error_code(exc)
        logger.warning("ohlcv_store_failed", op=op, code=code, **exc_log_fields(exc))
        return OhlcvStoreError(code)

    async def read(self, key: SeriesKey, start_ms: int, end_ms: int) -> StoredRead:
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                )
                await session.execute(_READ_TIMEOUT)
                row = (await session.execute(_COVERAGE_SQL, _series(key))).first()
                if row is None:
                    return StoredRead(None, ())
                cover = Coverage(_ms(row[0]), _ms(row[1]))
                lo, hi = max(start_ms, cover.from_ms), min(end_ms, cover.to_ms)
                raw = await self._pages(session, key, lo, hi)
            return StoredRead(cover, await _canon(raw, key.timeframe_s))  # after the transaction
        except Exception as exc:
            raise self._wrap("read", exc) from None

    async def read_bars(self, key: SeriesKey, start_ms: int, end_ms: int) -> tuple[Candle, ...]:
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                await session.execute(_READ_TIMEOUT)
                raw = await self._pages(session, key, start_ms, end_ms)
            return await _canon(raw, key.timeframe_s)  # after the transaction
        except Exception as exc:
            raise self._wrap("read_bars", exc) from None

    async def _pages(self, session: AsyncSession, key: SeriesKey, lo: int, hi: int) -> list[Any]:
        # the raw rows of [lo, hi): no CPU work inside the transaction (SY-26-53)
        out: list[Any] = []
        step = tf_ms(key.timeframe_s)
        while lo < hi:
            params = {**_series(key), "lo": _dt(lo), "hi": _dt(hi), "page": _PAGE}
            rows = (await session.execute(_BARS_SQL, params)).all()
            out.extend(rows)
            if len(rows) < _PAGE:
                break
            lo = _ms(out[-1][0]) + step
        return out

    async def extend(
        self, key: SeriesKey, span: Coverage, bars: Sequence[Candle], *, fetched_at_ms: int
    ) -> ExtendResult:
        # data errors surface before any I/O
        batch = await _off_loop(len(bars), prepare_extend, key, span, bars, fetched_at_ms)
        try:
            result = await self._extend(key, span, batch, _dt(fetched_at_ms))
        except Exception as exc:
            raise self._wrap("extend", exc) from None
        if result.divergent:
            logger.warning(
                "ohlcv_cache_divergent_bars",
                exchange=key.exchange,
                symbol=key.symbol,
                timeframe_s=key.timeframe_s,
                divergent=result.divergent,
                conflicts=result.conflicts,
            )
            metrics.increment(DIVERGENT_METRIC, result.divergent)
        return result

    async def _extend(
        self, key: SeriesKey, span: Coverage, batch: tuple[Candle, ...], fetched: datetime
    ) -> ExtendResult:
        ident = _series(key)
        async with self._session_factory() as session, session.begin():
            await session.execute(_WRITE_ISOLATION)  # first: before any snapshot is taken
            await session.execute(_LOCK_TIMEOUT)
            await session.execute(_WRITE_TIMEOUT)
            await session.execute(_ANCHOR_SQL, {**ident, "anchor": _dt(span.from_ms)})
            await session.execute(_LOCK_SQL, ident)  # the per-series serialisation point
            inserted = 0
            if batch:
                write = {**ident, **_arrays(batch), "fetched_at": fetched}
                write["ingest_version"] = _INGEST_VERSION
                inserted = int((await session.execute(_INSERT_SQL, write)).rowcount or 0)  # type: ignore[attr-defined]
            bounds = {"a": _dt(span.from_ms), "b": _dt(span.to_ms), "verified_at": fetched}
            row = (await session.execute(_UPDATE_SQL, {**ident, **bounds})).first()
            if row is None:  # rolls back the anchor and the bars too
                raise OhlcvStoreError("coverage_disjoint")
            conflicts = len(batch) - inserted
            divergent = 0
            if conflicts:
                found = await session.execute(_DIVERGENCE_SQL, {**ident, **_arrays(batch)})
                divergent = int(found.scalar_one())
            return ExtendResult(inserted, conflicts, divergent, Coverage(_ms(row[0]), _ms(row[1])))
