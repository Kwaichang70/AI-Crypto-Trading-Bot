"""OHLCV cache store contract (WP2.6 P4, SY-26-06/07/09/33): Protocol, dataclasses, fake.

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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, get_args

from data.ohlcv_integrity import (
    Candle,
    OhlcvDataError,
    check_identifiers,
    is_on_grid,
    series_digest,
    settle_end_ms,
)

__all__ = [
    "MAX_EXTEND_BARS",
    "STORE_CODES",
    "Coverage",
    "ExtendResult",
    "InMemoryOhlcvStore",
    "OhlcvStore",
    "OhlcvStoreError",
    "SeriesKey",
    "StoreCode",
    "StoredRead",
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
