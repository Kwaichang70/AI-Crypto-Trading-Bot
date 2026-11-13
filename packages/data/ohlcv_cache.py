"""OHLCV cache store contract (WP2.6 P4/P5, SY-26-05..11/33): Protocol, dataclasses, fake, source.

``OhlcvStore`` is the seam between the history source (P5) and the Postgres adapter
(``apps/api/services/ohlcv_store.py``). ``Candle`` from ``data.ohlcv_integrity`` is THE canonical
bar type (SY-26-33): there is no second bar class, and a store hands back only
``canonicalise_candle`` output. Coverage is ONE contiguous half-open interval per series that
only grows by touching spans (SY-26-05). Bars are immutable once written: the first write wins,
a differing re-write is counted as divergent and never applied (SY-26-09).

``prepare_extend`` is the single validate-before-write step both stores run, so a bad bar is
refused before any I/O (CF-26-40) and the fake cannot drift from the SQL adapter. Imports only
``data.ohlcv_integrity``; never ``apps`` or ``trading``. No clock: every time is an argument.
Not re-exported from ``data/__init__``.

``HistoricalBarSource`` (P5) is the read-through composition for history callers: closed-bar and
settle clamp, one contiguous coverage interval per series, fetch of the missing spans outside any
store call, ``extend`` in chunks of at most ``MAX_EXTEND_BARS``, keep-first re-read after a
conflict, cold = warm by one canonicaliser. Everything impure is injected: the upstream (integer
ms and seconds, structural, so this module imports no exchange, ``datetime`` or ``Decimal``
type), ``now_ms`` (the router's value, also ``fetched_at_ms``), the logger and an ``offload``
for CPU-heavy leaf calls. It logs only ``stored_row_invalid`` and ``coverage_disjoint`` (the
store logs every other code), never exception text, and never walks ``__context__``. A leaf
error on bars it has already validated (from a store call or from ``summarise_series``) is
re-raised as ``OhlcvInvariantError``, so the router can tell a bug (500) from bad exchange data.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, TypeVar, get_args

from data.ohlcv_integrity import (
    CACHEABLE_TIMEFRAMES_S,
    Candle,
    OhlcvDataError,
    SeriesSummary,
    canonicalise_candle,
    ceil_tf,
    check_closed,
    check_identifiers,
    closed_end_ms,
    is_on_grid,
    is_storable,
    series_digest,
    settle_end_ms,
    summarise_series,
    tf_ms,
)

__all__ = [
    "INLINE_MAX_ROWS",
    "MAX_EXTEND_BARS",
    "REQUEST_CAP_BARS",
    "STORE_CODES",
    "Coverage",
    "ExtendResult",
    "HistoricalBarSource",
    "HistoryResult",
    "InMemoryOhlcvStore",
    "LogSink",
    "OhlcvInvariantError",
    "OhlcvRequireError",
    "OhlcvStore",
    "OhlcvStoreError",
    "RangeUpstream",
    "RawRow",
    "SeriesKey",
    "StoreCode",
    "StoredRead",
    "UpstreamRange",
    "prepare_extend",
]

StoreCode = Literal[
    "missing_table",
    "unavailable",
    "timeout",
    "denied",
    "coverage_disjoint",
    "stored_row_invalid",
    "error",
]
STORE_CODES: Final[tuple[str, ...]] = get_args(StoreCode)
MAX_EXTEND_BARS: Final = 10_000  # SY-26-07 step 5: one transaction, one chunk
REQUEST_CAP_BARS: Final = 500_000  # DB-26-06; P6 passes cost_profiles.MAX_BACKTEST_BARS (SY-26-27)
INLINE_MAX_ROWS: Final = 8_000  # SY-26-33: below this the leaf calls run inline, above they offload
_OWN_LOGGED: Final = ("stored_row_invalid", "coverage_disjoint")  # the store logs the other five
_T = TypeVar("_T")
_REASONS: Final = ("missing_table", "unavailable", "timeout", "denied")  # block store_reason


class OhlcvStoreError(Exception):
    """Store failure with a closed ``code``. The message is fixed text, never data (CF-36)."""

    def __init__(self, code: StoreCode) -> None:
        if code not in STORE_CODES:
            raise ValueError("unknown ohlcv store code")
        super().__init__(f"ohlcv store failed ({code})")
        self.code: StoreCode = code

    def __reduce__(self) -> tuple[type[OhlcvStoreError], tuple[str]]:
        return (type(self), (self.code,))


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """One series. Identifiers are validated on construction and are case-sensitive."""

    exchange: str
    symbol: str
    timeframe_s: int

    def __post_init__(self) -> None:
        check_identifiers(self.exchange, self.symbol, self.timeframe_s)  # bad_identifier


@dataclass(frozen=True, slots=True)
class Coverage:
    """Half-open ``[from_ms, to_ms)`` the exchange was fully asked for (grid-aligned by callers)."""

    from_ms: int
    to_ms: int

    def __post_init__(self) -> None:
        ok = type(self.from_ms) is int and type(self.to_ms) is int
        if not ok or not 0 <= self.from_ms <= self.to_ms:
            raise OhlcvDataError("bad_type")  # not a span


@dataclass(frozen=True, slots=True)
class StoredRead:
    """``bars`` are the stored candles in ``[start, end)`` inside ``coverage``, ascending."""

    coverage: Coverage | None
    bars: tuple[Candle, ...]


@dataclass(frozen=True, slots=True)
class ExtendResult:
    """``conflicts`` = batch bars already stored; ``divergent`` = those whose values differ."""

    inserted: int
    conflicts: int
    divergent: int
    coverage: Coverage


class OhlcvStore(Protocol):
    async def read(self, key: SeriesKey, start_ms: int, end_ms: int) -> StoredRead:
        """One snapshot: the coverage and its bars in ``[start, end)``. May raise."""
        ...

    async def extend(
        self, key: SeriesKey, span: Coverage, bars: Sequence[Candle], *, fetched_at_ms: int
    ) -> ExtendResult:
        """One transaction: store ``bars`` (first write wins) and grow coverage by ``span``.

        ``span`` must touch the coverage (or there is none yet); else ``coverage_disjoint``.
        """

    async def read_bars(self, key: SeriesKey, start_ms: int, end_ms: int) -> tuple[Candle, ...]:
        """Stored bars in ``[start, end)``, coverage ignored (the conflict re-read)."""
        ...


def prepare_extend(
    key: SeriesKey, span: Coverage, bars: Sequence[Candle], fetched_at_ms: int
) -> tuple[Candle, ...]:
    """Validate an ``extend`` call and return the bars to write (ascending, deduplicated).

    Raises ``OhlcvDataError`` before any I/O: ``off_grid`` (span edge), ``forming_or_future``
    (span past the settle boundary of ``fetched_at_ms``, SY-26-05c), ``out_of_range`` (a bar
    outside the span, or more than ``MAX_EXTEND_BARS``), and whatever ``series_digest`` finds
    (non-canonical, off-grid or invalid bars). Bars sharing an ``open_ms`` keep the first: an
    in-batch duplicate is not a divergence (CF-26-43).
    """
    tf = key.timeframe_s
    if not (is_on_grid(span.from_ms, tf) and is_on_grid(span.to_ms, tf)):
        raise OhlcvDataError("off_grid")
    if span.to_ms > settle_end_ms(fetched_at_ms, tf):
        raise OhlcvDataError("forming_or_future")
    first: dict[int, Candle] = {}
    for candle in bars:
        if type(candle.open_ms) is not int:
            raise OhlcvDataError("bad_type")
        first.setdefault(candle.open_ms, candle)
    batch = tuple(first[ms] for ms in sorted(first))
    if len(batch) > MAX_EXTEND_BARS:
        raise OhlcvDataError("out_of_range")
    if batch and (batch[0].open_ms < span.from_ms or batch[-1].open_ms >= span.to_ms):
        raise OhlcvDataError("out_of_range")
    series_digest(key.exchange, key.symbol, tf, batch)  # canonical, on-grid, valid, or it raises
    return batch


class InMemoryOhlcvStore:
    """Fake with the semantics of the SQL store (no database, no clock)."""

    def __init__(self) -> None:
        self._coverage: dict[SeriesKey, Coverage] = {}
        self._bars: dict[SeriesKey, dict[int, Candle]] = {}
        self.fail_with: BaseException | None = None  # test hook: raise on the next call

    def _slice(self, key: SeriesKey, lo: int, hi: int) -> tuple[Candle, ...]:
        stored = self._bars.get(key, {})
        return tuple(stored[ms] for ms in sorted(stored) if lo <= ms < hi)

    async def read(self, key: SeriesKey, start_ms: int, end_ms: int) -> StoredRead:
        if self.fail_with is not None:
            raise self.fail_with
        cover = self._coverage.get(key)
        if cover is None:
            return StoredRead(None, ())
        return StoredRead(
            cover, self._slice(key, max(start_ms, cover.from_ms), min(end_ms, cover.to_ms))
        )

    async def read_bars(self, key: SeriesKey, start_ms: int, end_ms: int) -> tuple[Candle, ...]:
        if self.fail_with is not None:
            raise self.fail_with
        return self._slice(key, start_ms, end_ms)

    async def extend(
        self, key: SeriesKey, span: Coverage, bars: Sequence[Candle], *, fetched_at_ms: int
    ) -> ExtendResult:
        if self.fail_with is not None:
            raise self.fail_with
        batch = prepare_extend(key, span, bars, fetched_at_ms)
        current = self._coverage.get(key) or Coverage(span.from_ms, span.from_ms)
        if span.from_ms > current.to_ms or span.to_ms < current.from_ms:
            raise OhlcvStoreError("coverage_disjoint")
        stored = self._bars.setdefault(key, {})
        inserted = divergent = 0
        for candle in batch:
            old = stored.get(candle.open_ms)
            if old is None:
                stored[candle.open_ms] = candle
                inserted += 1
            elif old != candle:
                divergent += 1
        merged = Coverage(min(current.from_ms, span.from_ms), max(current.to_ms, span.to_ms))
        self._coverage[key] = merged
        return ExtendResult(inserted, len(batch) - inserted, divergent, merged)


# ---------------------------------------------------------------------------
# HistoricalBarSource (WP2.6 P5): the read-through composition, against the Protocols
# ---------------------------------------------------------------------------
class OhlcvRequireError(Exception):
    """``require`` cannot be honoured: the router maps ``code`` to the 503 / 422 detail code."""

    def __init__(
        self, code: Literal["ohlcv_store_unavailable", "ohlcv_timeframe_not_cached"]
    ) -> None:
        super().__init__(f"ohlcv require failed ({code})")
        self.code = code


class OhlcvInvariantError(OhlcvDataError):
    """A leaf check failed on bars the source had already validated: a bug (CF-26-56, 500)."""


@dataclass(frozen=True, slots=True)
class RawRow:
    """One exchange row: integer ms and open, high, low, close, volume (Decimal, str or int)."""

    open_ms: int
    values: tuple[Any, Any, Any, Any, Any]


@dataclass(frozen=True, slots=True)
class UpstreamRange:
    """The rows of one ``[start, end)`` fetch; ``complete`` is false if a safety limit ended it."""

    rows: Sequence[RawRow]
    complete: bool


class RangeUpstream(Protocol):
    async def connect(self) -> None: ...

    # Raise what today's fetch raises for an unknown symbol or timeframe (the 400 mapping).
    def check_series(self, symbol: str, timeframe_s: int) -> None: ...

    async def fetch_range(
        self, symbol: str, timeframe_s: int, start_ms: int, end_ms: int
    ) -> UpstreamRange: ...


class LogSink(Protocol):
    def warning(self, event: str, /, **fields: object) -> None: ...


@dataclass(frozen=True, slots=True)
class HistoryResult:
    """One series: canonical ``bars`` and the SY-26-10 facts. ``summary`` feeds ``run_digest``."""

    bars: tuple[Candle, ...]
    summary: SeriesSummary
    from_ms: int  # the clamped request [from_ms, to_ms), on the grid
    to_ms: int
    source: str  # "cache", "exchange" or "mixed"
    store: str  # "ok" or "unavailable"
    store_reason: str  # "none" or a block store_reason code
    volatile_tail_bars: int
    rounded_values: int
    divergent_bars: int
    bypass_disjoint: bool
    fetch_complete: bool  # false if a safety limit (page cap, no progress) cut a fetch short


@dataclass(frozen=True, slots=True)
class _Req:
    key: SeriesKey
    a: int  # first slot
    e: int  # end of the request: closed-bar clamp, on the grid
    settle: int  # nothing at or after it is stored
    step: int
    s_end: int  # min(e, settle): the end of the storable part


@dataclass(slots=True)
class _Acc:
    reason: str = "none"
    fetched: bool = False
    bypass: bool = False
    rounded: int = 0
    divergent: int = 0
    tail: int = 0
    incomplete: bool = False


def _canon(rows: Sequence[RawRow], tf_s: int, now_ms: int) -> tuple[list[Candle], int]:
    out: list[Candle] = []
    rounded = 0
    for r in rows:
        candle, n = canonicalise_candle(r.open_ms, *r.values, timeframe_s=tf_s)
        check_closed(candle.open_ms, now_ms, tf_s)
        out.append(candle)
        rounded += n
    return out, rounded


class HistoricalBarSource:
    """One request's source (the cold budget is per instance). ``mode`` is ``read_through``
    (fail-open) or ``require`` (fail-closed); the router builds no source for mode ``off``."""

    def __init__(
        self,
        upstream: RangeUpstream,
        store: OhlcvStore,
        *,
        exchange: str,
        now_ms: int,
        mode: Literal["read_through", "require"] = "read_through",
        log: LogSink | None = None,
        offload: Callable[..., Awaitable[Any]] | None = None,
        cap_bars: int = REQUEST_CAP_BARS,
    ) -> None:
        if mode not in ("read_through", "require"):  # no "off" source; a typo must not fail open
            raise ValueError("unknown ohlcv source mode")
        self._upstream, self._store, self._exchange = upstream, store, exchange
        self._now_ms, self._mode, self._log = now_ms, mode, log
        self._offload, self._cap, self._cold_used = offload, cap_bars, 0
        self._connected = False

    async def connect(self) -> None:
        """Once per request, before any concurrent ``get_range`` (today's single connect)."""
        await self._upstream.connect()
        self._connected = True

    async def get_ranges(
        self, symbols: Sequence[str], timeframe_s: int, start_ms: int, end_ms: int
    ) -> list[HistoryResult]:
        # Sequential, so the cold budget is spent in the order given (deterministic).
        return [await self.get_range(s, timeframe_s, start_ms, end_ms) for s in symbols]

    async def get_range(
        self, symbol: str, timeframe_s: int, start_ms: int, end_ms: int
    ) -> HistoryResult:
        """Canonical bars with ``start <= open < end`` that are closed at ``now_ms``."""
        if type(timeframe_s) is int and timeframe_s not in CACHEABLE_TIMEFRAMES_S:
            raise OhlcvRequireError("ohlcv_timeframe_not_cached")  # 1w: the router decides
        key = SeriesKey(self._exchange, symbol, timeframe_s)
        if not self._connected:
            await self.connect()
        self._upstream.check_series(symbol, timeframe_s)
        if end_ms <= start_ms:
            raise ValueError("end must be strictly after start")
        end = ceil_tf(closed_end_ms(end_ms, self._now_ms, timeframe_s), timeframe_s)
        settle, step = settle_end_ms(self._now_ms, timeframe_s), tf_ms(timeframe_s)
        q = _Req(key, ceil_tf(start_ms, timeframe_s), end, settle, step, min(end, settle))
        if q.a < 0 or (q.e - q.a) // step > self._cap:  # CF-26-52 cap; no pre-epoch window
            raise OhlcvDataError("out_of_range")
        served: dict[int, Candle] = {}
        acc = _Acc()
        cov: Coverage | None = None
        if q.s_end > q.a:
            try:
                got = await self._call("read", key, self._store.read(key, q.a, q.s_end))
                cov = got.coverage
                served.update((c.open_ms, c) for c in got.bars)
            except OhlcvStoreError as exc:
                acc.reason = self._failed(exc.code)
        cached = bool(served)
        spans = [(q.a, q.e, False)] if q.e > q.a else []
        extend = q.s_end > q.a and acc.reason == "none"
        if extend:
            spans, gap = self._spans(q, cov)
            need = sum((hi - lo) // step for lo, hi, _ in spans)
            late = cov is not None and cov.from_ms > q.settle  # written under a later now_ms
            if late or (gap and (gap > q.s_end - q.a or self._cold_used + need > self._cap)):
                if self._mode == "require":  # SY-26-05(a), SY-26-27: the gap is not fetched
                    raise OhlcvRequireError("ohlcv_store_unavailable")
                spans, extend, acc.bypass = [(q.a, q.e, False)], False, True
                need = (q.e - q.a) // step
            self._cold_used += need
        for lo, hi, left in spans:
            await self._fill(q, lo, hi, left, extend, served, acc, cov)
        bars = tuple(served[ms] for ms in sorted(served) if q.a <= ms < q.e)
        try:
            summary = await self._run(
                len(bars), summarise_series, self._exchange, symbol, timeframe_s, bars
            )
        except OhlcvDataError as exc:  # the served bars are canonical: an invariant broke
            raise OhlcvInvariantError(exc.code) from None
        source = "cache" if not acc.fetched else ("mixed" if cached else "exchange")
        return HistoryResult(
            bars,
            summary,
            q.a,
            q.e,
            source,
            "ok" if acc.reason == "none" else "unavailable",
            acc.reason,
            acc.tail,
            acc.rounded,
            acc.divergent,
            acc.bypass,
            not acc.incomplete,
        )

    def _spans(self, q: _Req, cov: Coverage | None) -> tuple[list[tuple[int, int, bool]], int]:
        """Spans to fetch ``(lo, hi, left)`` and the in-between gap that keeps coverage contiguous.

        Each edge extension re-fetches one stored bar (SY-26-05(d)). The right span runs to the
        request end, so the volatile tail comes from the same fetch (SY-26-07 step 6).
        """
        cf, ct = (cov.from_ms, cov.to_ms) if cov else (q.a, q.a)
        spans: list[tuple[int, int, bool]] = []
        if q.a < cf:
            probe = cf + q.step if cf + q.step <= min(ct, q.settle) else cf
            spans.append((q.a, probe, True))
        right = q.s_end
        if ct < q.s_end:
            right = ct - q.step if cov is not None and ct - q.step >= cf else ct
        if q.e > right:
            spans.append((right, q.e, False))
        gap = max(cf - q.s_end, q.a - ct, 0) if cov is not None else 0
        return spans, gap

    async def _fill(
        self,
        q: _Req,
        lo: int,
        hi: int,
        left: bool,
        extend: bool,
        served: dict[int, Candle],
        acc: _Acc,
        cov: Coverage | None,
    ) -> None:
        """Fetch ``[lo, hi)``, serve it, and (when complete with a new bar) extend coverage."""
        tf_s, key = q.key.timeframe_s, q.key
        got = await self._upstream.fetch_range(key.symbol, tf_s, lo, hi)
        acc.fetched = True
        acc.incomplete |= not got.complete
        if extend and not got.complete and self._mode == "require":  # not storable: fail closed
            raise OhlcvRequireError("ohlcv_store_unavailable")
        first: dict[int, RawRow] = {}  # explicit: sorted, in-window, first duplicate wins
        for row in got.rows:
            if type(row.open_ms) is not int:
                raise OhlcvDataError("bad_type")
            if lo <= row.open_ms < hi:
                first.setdefault(row.open_ms, row)
        rows = [first[ms] for ms in sorted(first)]
        body: list[Candle] = []
        for i in range(0, len(rows), MAX_EXTEND_BARS):
            part, n = await self._run(
                len(rows), _canon, rows[i : i + MAX_EXTEND_BARS], tf_s, self._now_ms
            )
            acc.rounded += n
            for c in part:
                served.setdefault(c.open_ms, c)
                if is_storable(c.open_ms, self._now_ms, tf_s):
                    body.append(c)
                else:
                    acc.tail += 1
        top = min(hi, q.settle)
        # SY-26-05(b): a bar inside ``cov`` (the overlap probe) is not a bar returned
        fresh = any(cov is None or not cov.from_ms <= c.open_ms < cov.to_ms for c in body)
        if extend and acc.reason == "none" and got.complete and fresh and top > lo:
            await self._extend_chunks(q, lo, top, body, left, served, acc)

    async def _extend_chunks(
        self,
        q: _Req,
        lo: int,
        top: int,
        body: list[Candle],
        left: bool,
        served: dict[int, Candle],
        acc: _Acc,
    ) -> None:
        size = MAX_EXTEND_BARS * q.step
        groups: dict[int, list[Candle]] = {}
        for c in body:
            groups.setdefault((c.open_ms - lo) // size, []).append(c)
        order = list(enumerate(range(lo, top, size)))  # ascending; left spans run top-down
        try:
            for i, x in reversed(order) if left else order:
                y = min(x + size, top)
                res = await self._extend_one(q.key, x, y, groups.get(i, []), q.step)
                acc.divergent += res.divergent
                if res.conflicts:  # keep-first: the stored values win (SY-26-07 step 7)
                    stored = await self._call(
                        "read_bars", q.key, self._store.read_bars(q.key, x, y)
                    )
                    served.update((c.open_ms, c) for c in stored)
        except OhlcvStoreError as exc:
            acc.reason = self._failed(exc.code)

    async def _extend_one(
        self, key: SeriesKey, x: int, y: int, group: list[Candle], step: int
    ) -> ExtendResult:
        """``extend``; on ``coverage_disjoint`` or a lock timeout re-read and retry once."""
        span = Coverage(x, y)
        try:
            return await self._call(
                "extend", key, self._store.extend(key, span, group, fetched_at_ms=self._now_ms)
            )
        except OhlcvStoreError as exc:
            first = exc
        if first.code not in ("coverage_disjoint", "timeout"):
            raise first
        cover = (await self._call("read", key, self._store.read(key, x, x + step))).coverage
        if cover is not None and (x > cover.to_ms or y < cover.from_ms):
            raise first  # still not touching: the retry cannot succeed
        return await self._call(
            "extend", key, self._store.extend(key, span, group, fetched_at_ms=self._now_ms)
        )

    async def _call(self, op: str, key: SeriesKey, call: Awaitable[_T]) -> _T:
        try:
            return await call
        except OhlcvStoreError as exc:
            if self._log is not None and exc.code in _OWN_LOGGED:
                self._log.warning(
                    "ohlcv_store_failed",
                    op=op,
                    code=exc.code,
                    exchange=key.exchange,
                    symbol=key.symbol,
                    timeframe_s=key.timeframe_s,
                )
            raise
        except OhlcvDataError as exc:  # the source validated every row: an invariant broke
            raise OhlcvInvariantError(exc.code) from None

    def _failed(self, code: str) -> str:
        """The block ``store_reason`` for a store failure (fail-open), or ``require``'s refusal."""
        if self._mode == "require":
            raise OhlcvRequireError("ohlcv_store_unavailable") from None
        return code if code in _REASONS else "error"

    async def _run(self, n: int, fn: Callable[..., _T], *args: object) -> _T:
        """Run a leaf call inline below ``INLINE_MAX_ROWS``, else through the injected offload."""
        if self._offload is None or n <= INLINE_MAX_ROWS:
            return fn(*args)
        result: _T = await self._offload(fn, *args)
        return result
