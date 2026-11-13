"""WP2.6 P5 (SY-26-05/07/08/11, SY-26-27): ``HistoricalBarSource`` against the Protocols.

A scriptable fake upstream (integer ms, no exchange) and a recording ``InMemoryOhlcvStore``.
No network, no Postgres, no wall clock: ``now_ms`` is injected.  One slot = one grid step;
slot ``i`` opens at ``T0 + i * step``.  With ``NOW`` inside slot 1000, ``F = floor(now)`` is the
open of slot 1000, nothing at or after ``SETTLE = F - 1 step`` is stored, and slot 999 (closed,
inside the settle margin) is the volatile tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

import data.services.ccxt_market_data as cmd
from common.types import TimeFrame
from data.ohlcv_cache import (
    INLINE_MAX_ROWS,
    MAX_EXTEND_BARS,
    Coverage,
    ExtendResult,
    HistoricalBarSource,
    InMemoryOhlcvStore,
    OhlcvInvariantError,
    OhlcvRequireError,
    OhlcvStoreError,
    RawRow,
    SeriesKey,
    StoredRead,
    UpstreamRange,
    _canon,
)
from data.ohlcv_integrity import (
    Candle,
    OhlcvDataError,
    canonicalise_candle,
    check_closed,
    closed_end_ms,
    is_storable,
    run_digest,
    summarise_series,
)

T0, H = 1_704_067_200_000, 3_600_000
F = T0 + 1000 * H
NOW = F + 123_456
SYM = "BTC/EUR"
END = T0 + 5000 * H  # far in the future: clamped to F


def vals(i: int) -> tuple[str, str, str, str, str]:
    return (str(100 + i % 50), str(103 + i % 50), str(98 + i % 50), str(101 + i % 50), "7")


class Up:
    """Scriptable upstream: rows for the slots in ``have`` inside each requested window."""

    def __init__(self, n: int = 1000, *, step: int = H, complete: bool = True) -> None:
        self.have, self.step, self.complete = set(range(n)), step, complete
        self.calls: list[tuple[int, int]] = []
        self.connects, self.checked = 0, 0
        self.revise: dict[int, tuple[str, ...]] = {}
        self.dup, self.junk, self.bad_check = False, False, None

    async def connect(self) -> None:
        self.connects += 1

    def check_series(self, symbol: str, timeframe_s: int) -> None:
        self.checked += 1
        if self.bad_check:
            raise self.bad_check

    async def fetch_range(self, symbol: str, tf_s: int, lo: int, hi: int) -> UpstreamRange:
        self.calls.append((lo, hi))
        rows = []
        for i in sorted(self.have):
            ms = T0 + i * self.step
            if lo <= ms < hi:
                rows.append(RawRow(ms, self.revise.get(i, vals(i))))  # type: ignore[arg-type]
        if self.dup:
            rows = rows + rows[::-1]  # unsorted, every row twice
        if self.junk:  # rows outside the window must not leak
            rows += [RawRow(lo - self.step, vals(1)), RawRow(hi, vals(2))]
        ok = self.complete[len(self.calls) - 1] if type(self.complete) is list else self.complete
        return UpstreamRange(rows, ok)


class Rec(InMemoryOhlcvStore):
    """In-memory store that records calls and replays scripted failures per op."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[tuple[Any, ...]] = []
        self.fail: dict[str, list[BaseException]] = {}
        self.race: dict[int, Candle] | None = None  # a concurrent writer's bars, written first
        self.race_cover: Coverage | None = None  # a writer that moved the coverage away

    def _maybe(self, op: str) -> None:
        if self.fail.get(op):
            raise self.fail[op].pop(0)

    async def read(self, key: SeriesKey, start_ms: int, end_ms: int) -> StoredRead:
        self.ops.append(("read", start_ms, end_ms))
        self._maybe("read")
        return await super().read(key, start_ms, end_ms)

    async def read_bars(self, key: SeriesKey, start_ms: int, end_ms: int) -> tuple[Candle, ...]:
        self.ops.append(("read_bars", start_ms, end_ms))
        self._maybe("read_bars")
        return await super().read_bars(key, start_ms, end_ms)

    async def extend(
        self, key: SeriesKey, span: Coverage, bars: Sequence[Candle], *, fetched_at_ms: int
    ) -> ExtendResult:
        self.ops.append(("extend", span.from_ms, span.to_ms, len(bars), fetched_at_ms))
        if self.race_cover:
            self._coverage[key], self.race_cover = self.race_cover, None
        if self.race:
            self._bars.setdefault(key, {}).update(self.race)
            self.race = None
        self._maybe("extend")
        return await super().extend(key, span, bars, fetched_at_ms=fetched_at_ms)

    def of(self, op: str) -> list[tuple[Any, ...]]:
        return [o for o in self.ops if o[0] == op]


class Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, /, **fields: object) -> None:
        self.events.append((event, fields))


def source(up: Up, store: Rec, *, log: Log | None = None, **kw: Any) -> HistoricalBarSource:
    return HistoricalBarSource(up, store, exchange="coinbase", now_ms=NOW, log=log, **kw)


async def get(src: HistoricalBarSource, start: int = 0, end: int = END, tf_s: int = 3600) -> Any:
    return await src.get_range(SYM, tf_s, T0 + start * H, end)


def cover(store: Rec, tf_s: int = 3600) -> Coverage | None:
    return store._coverage.get(SeriesKey("coinbase", SYM, tf_s))


# ---------------------------------------------------------------------------
# cold = warm, scope, settle
# ---------------------------------------------------------------------------
async def test_cold_fetches_once_stores_up_to_settle_and_keeps_the_tail_volatile() -> None:
    up, store = Up(), Rec()
    res = await get(source(up, store))
    assert up.calls == [(T0, F)] and up.connects == 1 and up.checked == 1
    assert [c.open_ms for c in res.bars] == [T0 + i * H for i in range(1000)]
    assert (res.from_ms, res.to_ms, res.volatile_tail_bars) == (T0, F, 1)
    assert (res.source, res.store, res.store_reason) == ("exchange", "ok", "none")
    assert cover(store) == Coverage(T0, F - H)  # SETTLE_BARS = 1: slot 999 is not stored
    assert len(store._bars[SeriesKey("coinbase", SYM, 3600)]) == 999
    assert store.of("extend") == [("extend", T0, F - H, 999, NOW)]  # fetched_at = the router's now
    assert store.ops[0] == ("read", T0, F - H)  # the first read is the storable part (CR T-2e)


async def test_warm_equals_cold_bit_for_bit_and_fetches_only_the_tail() -> None:
    cold_up, store = Up(), Rec()
    cold_up.revise[3] = ("100", "103", "98", "101", "1.0000000000005")  # rounds: counted once
    cold = await get(source(cold_up, store))
    warm_up = Up()
    warm_up.revise = cold_up.revise  # the exchange still says the same
    warm = await get(source(warm_up, store))
    assert warm_up.calls == [(F - H, F)]  # the tail only: the last closed bar is still served
    assert warm.bars == cold.bars and warm.summary == cold.summary
    assert [c.volume.as_tuple() for c in warm.bars] == [c.volume.as_tuple() for c in cold.bars]
    assert cold.bars[3].volume == Decimal("1.000000000001")
    assert (cold.rounded_values, warm.rounded_values) == (1, 0)
    assert (warm.source, warm.volatile_tail_bars) == ("mixed", 1)
    assert run_digest([cold.summary]) == run_digest([warm.summary])


async def test_a_covered_closed_request_makes_no_exchange_call_but_still_validates() -> None:
    store = Rec()
    await get(source(Up(), store))
    up = Up()
    res = await get(source(up, store), 10, T0 + 50 * H)
    assert up.calls == [] and up.checked == 1 and (res.source, len(res.bars)) == ("cache", 40)


async def test_end_and_start_are_open_time_bounds_on_the_grid() -> None:
    half = H // 2
    res = await get(source(Up(), Rec()), 0, T0 + 10 * H + half)  # slot 10 opens before the end
    assert [c.open_ms for c in res.bars] == [T0 + i * H for i in range(11)]
    res = await source(Up(), Rec()).get_range(SYM, 3600, T0 + half, T0 + 5 * H)  # next slot
    assert (res.from_ms, res.bars[0].open_ms) == (T0 + H, T0 + H)
    res = await get(source(Up(), Rec()), 1500)  # starts after F: nothing to serve
    assert res.bars == () and res.summary == summarise_series("coinbase", SYM, 3600, ())


# ---------------------------------------------------------------------------
# extension, probe, divergence, chunking
# ---------------------------------------------------------------------------
async def test_right_and_left_extension_refetch_one_stored_bar_as_a_probe() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up()
    await get(source(up, store), 100, T0 + 300 * H)  # right: probe slot 199
    await get(source(up, store), 50, T0 + 300 * H)  # left: probe slot 100
    assert up.calls == [(T0 + 199 * H, T0 + 300 * H), (T0 + 50 * H, T0 + 101 * H)]
    assert cover(store) == Coverage(T0 + 50 * H, T0 + 300 * H)
    assert [o[1:4] for o in store.of("extend")] == [
        (T0 + 100 * H, T0 + 200 * H, 100),
        (T0 + 199 * H, T0 + 300 * H, 101),
        (T0 + 50 * H, T0 + 101 * H, 51),
    ]


async def test_a_revised_probe_bar_is_counted_and_the_stored_value_wins() -> None:
    store = Rec()
    first = await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up()
    up.revise[199] = ("1", "2", "1", "2", "9")  # the exchange now disagrees about slot 199
    res = await get(source(up, store), 100, T0 + 300 * H)
    assert res.divergent_bars == 1
    assert res.bars[99] == first.bars[99] and res.bars[99].close != Decimal(2)
    assert store.of("read_bars") == [("read_bars", T0 + 199 * H, T0 + 300 * H)]
    assert next(o for o in store.of("extend") if o[1] == T0 + 199 * H)[4] == NOW


async def test_a_concurrent_writer_wins_the_conflict_and_its_value_is_served() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    raced = canonicalise_candle(T0 + 250 * H, "1", "2", "1", "2", "1", timeframe_s=3600)[0]
    store.race = {raced.open_ms: raced}
    res = await get(source(Up(), store), 100, T0 + 300 * H)
    assert res.bars[150] == raced and res.divergent_bars == 1 and store.of("read_bars")


async def test_rounding_and_divergence_add_up_over_both_edge_spans() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    up, odd = Up(), ("1", "2", "1", "2", "1.0000000000005")  # revised, and over the scale
    up.revise = {100: odd, 199: odd}
    res = await get(source(up, store), 50, T0 + 300 * H)
    assert (res.divergent_bars, res.rounded_values) == (2, 2)


async def test_after_a_failed_extension_no_further_span_is_written() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    store.fail["extend"] = [OhlcvStoreError("unavailable")] * 3
    res = await get(source(Up(), store), 50, T0 + 300 * H)  # left and right both missing
    assert (
        len(store.of("extend")) == 2 and res.store_reason == "unavailable"
    )  # the first + its failed left


async def test_a_re_read_that_is_still_disjoint_is_not_retried() -> None:
    store, log = Rec(), Log()
    store.race_cover = Coverage(T0 + 500 * H, T0 + 600 * H)  # a writer moved the coverage away
    res = await get(source(Up(), store, log=log), 0, T0 + 200 * H)
    assert (res.store_reason, len(store.of("extend")), len(log.events)) == ("error", 1, 1)


async def test_a_request_inside_the_volatile_tail_never_touches_the_store() -> None:
    up, store = Up(), Rec()
    res = await get(source(up, store), 999)  # slot 999: closed but not yet settled
    assert store.ops == [] and up.calls == [(T0 + 999 * H, F)] and len(res.bars) == 1


def test_canon_refuses_a_row_that_is_not_closed() -> None:
    row = RawRow(T0, vals(0))  # type: ignore[arg-type]
    assert _canon([row], 3600, T0 + H)[1] == 0
    with pytest.raises(OhlcvDataError) as err:
        _canon([row], 3600, T0 + H - 1)
    assert err.value.code == "forming_or_future"


async def test_an_incomplete_fetch_never_overrides_a_stored_bar() -> None:
    store = Rec()
    first = await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up(complete=False)
    up.revise[199] = ("1", "2", "1", "2", "9")
    res = await get(source(up, store), 100, T0 + 300 * H)
    assert res.bars[99] == first.bars[99] and len(store.of("extend")) == 1


async def test_extensions_are_chunked_at_the_maximum_ascending_right_descending_left() -> None:
    assert MAX_EXTEND_BARS == 10_000  # SY-26-07 step 5 (P4 O-4, mutant C01; critic T-3)
    n = 2 * MAX_EXTEND_BARS + 500
    store, up = Rec(), Up(60_000, step=60_000)
    await get(source(up, store), 0, T0 + n * 60_000, 60)  # cold: three chunks ascending
    spans = [(o[1] - T0, o[2] - T0, o[3]) for o in store.of("extend")]
    step = MAX_EXTEND_BARS * 60_000
    assert spans == [(0, step, 10_000), (step, 2 * step, 10_000), (2 * step, n * 60_000, 500)]
    assert cover(store, 60) == Coverage(T0, T0 + n * 60_000)
    left = Rec()
    await source(up, left).get_range(SYM, 60, T0 + 30_000 * 60_000, T0 + 40_000 * 60_000)
    left.ops.clear()
    await source(up, left).get_range(SYM, 60, T0, T0 + 40_000 * 60_000)
    starts = [o[1] for o in left.of("extend")]
    assert starts == sorted(starts, reverse=True) and len(starts) == 4  # nearest coverage first
    assert all(
        o[2] - o[1] <= MAX_EXTEND_BARS * 60_000 and o[3] <= MAX_EXTEND_BARS
        for o in left.of("extend")
    )


# ---------------------------------------------------------------------------
# disjoint requests and the combined cold bound (SY-26-05(a), SY-26-27)
# ---------------------------------------------------------------------------
async def test_a_small_gap_is_fetched_to_keep_coverage_contiguous() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up()
    res = await get(source(up, store), 230, T0 + 300 * H)  # gap 30 <= own span 70
    assert up.calls == [(T0 + 199 * H, T0 + 300 * H)] and not res.bypass_disjoint
    assert cover(store) == Coverage(T0 + 100 * H, T0 + 300 * H) and len(res.bars) == 70


async def test_a_gap_longer_than_the_request_is_served_bypass_disjoint_and_not_stored() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    before, up = list(store.ops), Up()
    res = await get(source(up, store), 600, T0 + 610 * H)  # gap 400 > own span 10
    assert up.calls == [(T0 + 600 * H, T0 + 610 * H)] and res.bypass_disjoint
    assert (res.source, res.store, len(res.bars)) == ("exchange", "ok", 10)
    assert [o for o in store.ops if o not in before and o[0] == "extend"] == []
    assert cover(store) == Coverage(T0 + 100 * H, T0 + 200 * H)


@pytest.mark.parametrize("order", [("A/EUR", "B/EUR"), ("B/EUR", "A/EUR")])
async def test_the_combined_cold_bound_is_spent_in_the_given_order(order: tuple[str, str]) -> None:
    store = Rec()
    for sym in order:
        await HistoricalBarSource(Up(), store, exchange="coinbase", now_ms=NOW).get_range(
            sym, 3600, T0 + 100 * H, T0 + 200 * H
        )
    src = HistoricalBarSource(Up(), store, exchange="coinbase", now_ms=NOW, cap_bars=150)
    got = await src.get_ranges(order, 3600, T0 + 230 * H, T0 + 300 * H)  # each needs 101 bars
    assert [r.bypass_disjoint for r in got] == [False, True]  # the second one exceeds the 150


async def test_require_refuses_what_it_cannot_store() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    with pytest.raises(OhlcvRequireError) as err:
        await get(source(Up(), store, mode="require"), 600, T0 + 610 * H)
    assert err.value.code == "ohlcv_store_unavailable"


# ---------------------------------------------------------------------------
# clean finish only (SY-26-05(b))
# ---------------------------------------------------------------------------
async def test_require_refuses_an_incomplete_fetch_it_cannot_store() -> None:
    store = Rec()
    with pytest.raises(OhlcvRequireError):
        await get(source(Up(complete=False), store, mode="require"))
    assert store.of("extend") == []


async def test_incomplete_or_empty_fetches_never_advance_coverage() -> None:
    store = Rec()
    res = await get(source(Up(complete=False), store))
    assert len(res.bars) == 1000 and cover(store) is None and store.of("extend") == []
    assert res.fetch_complete is False and (await get(source(Up(), Rec()))).fetch_complete
    empty = Up(0)
    assert (await get(source(empty, store))).bars == () and cover(store) is None
    assert store.of("extend") == []


async def test_an_edge_span_with_only_the_probe_bar_never_advances_coverage() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up()
    up.have = {100, 199}  # outage: only the two stored edge bars come back
    res = await get(source(up, store), 50, T0 + 300 * H)
    assert cover(store) == Coverage(T0 + 100 * H, T0 + 200 * H) and store.of("extend")[1:] == []
    assert len(res.bars) == 100 and res.store == "ok"
    one = await get(source(Up(201), store), 100, T0 + 300 * H)  # one new bar (slot 200) is enough
    assert cover(store) == Coverage(T0 + 100 * H, T0 + 300 * H) and len(one.bars) == 101


@pytest.mark.parametrize("first", [999, 1000, 1001])  # settle, F and F + 1 slot at NOW
async def test_coverage_written_under_a_later_now_is_bypassed(first: int) -> None:
    store, later = Rec(), NOW + 3 * H
    await HistoricalBarSource(Up(1200), store, exchange="coinbase", now_ms=later).get_range(
        SYM, 3600, T0 + first * H, T0 + 1002 * H
    )
    res = await get(source(Up(1200), store), 100, T0 + 990 * H)  # the exchange has the forming bar
    assert (len(res.bars), res.store, res.bypass_disjoint) == (890, "ok", first > 999)
    assert cover(store) == Coverage(T0 + (first if first > 999 else 100) * H, T0 + 1002 * H)


async def test_an_empty_middle_chunk_still_advances_coverage_when_the_span_has_bars() -> None:
    up = Up(0, step=60_000)
    up.have = {0, 25_000}  # chunks: [0,10k) one bar, [10k,20k) none, [20k,..) one bar
    store = Rec()
    await get(source(up, store), 0, T0 + 30_000 * 60_000, 60)
    assert [o[3] for o in store.of("extend")] == [1, 0, 1] and cover(store, 60) == Coverage(
        T0, T0 + 30_000 * 60_000
    )


# ---------------------------------------------------------------------------
# failures, logging, require
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("code", "reason", "logged"),
    [
        ("missing_table", "missing_table", False),
        ("unavailable", "unavailable", False),
        ("timeout", "timeout", False),
        ("denied", "denied", False),
        ("error", "error", False),
        ("coverage_disjoint", "error", True),
        ("stored_row_invalid", "error", True),
    ],
)
@pytest.mark.parametrize("op", ["read", "extend"])
async def test_fail_open_serves_the_same_bars_and_logs_only_two_codes(
    code: Any, reason: str, logged: bool, op: str
) -> None:
    store, log = Rec(), Log()
    store.fail[op] = [OhlcvStoreError(code)] * 3
    res = await get(source(Up(), store, log=log))
    assert res.summary == (await get(source(Up(), Rec()))).summary
    assert (res.store, res.store_reason) == ("unavailable", reason)
    expected = [
        (
            "ohlcv_store_failed",
            {"op": op, "code": code, "exchange": "coinbase", "symbol": SYM, "timeframe_s": 3600},
        )
    ]
    retried = op == "extend" and code in ("coverage_disjoint", "timeout")
    assert log.events == (expected * (2 if retried else 1) if logged else [])


async def test_a_context_exception_is_never_logged_or_walked() -> None:
    store, log = Rec(), Log()
    try:
        try:
            raise RuntimeError("postgresql://u:SECRET@h/db")
        except RuntimeError:
            raise OhlcvStoreError("stored_row_invalid") from None
    except OhlcvStoreError as err:
        store.fail["read"] = [err]
    res = await get(source(Up(), store, log=log))
    assert res.store_reason == "error" and "SECRET" not in repr(log.events) + repr(res)


async def test_require_turns_every_store_failure_into_one_closed_error() -> None:
    for op in ("read", "extend"):
        store = Rec()
        store.fail[op] = [OhlcvStoreError("timeout")] * 3
        with pytest.raises(OhlcvRequireError) as err:
            await get(source(Up(), store, mode="require"))
        assert err.value.code == "ohlcv_store_unavailable" and err.value.__suppress_context__
        assert str(err.value) == "ohlcv require failed (ohlcv_store_unavailable)"


@pytest.mark.parametrize("mode", ["read_through", "require"])
async def test_one_week_is_not_cached_and_the_router_decides(mode: Any) -> None:
    up = Up()
    with pytest.raises(OhlcvRequireError) as err:
        await source(up, Rec(), mode=mode).get_range(SYM, 604_800, T0, END)
    assert err.value.code == "ohlcv_timeframe_not_cached" and up.connects == 0


@pytest.mark.parametrize("code", ["coverage_disjoint", "timeout"])
async def test_extend_is_retried_once_after_a_re_read(code: Any) -> None:
    store = Rec()
    store.fail["extend"] = [OhlcvStoreError(code)]
    res = await get(source(Up(), store))
    assert res.store == "ok" and cover(store) == Coverage(T0, F - H)
    assert [o[0] for o in store.ops] == ["read", "extend", "read", "extend"]
    assert store.ops[2][1:] == (T0, T0 + H)


async def test_a_second_failure_is_a_store_failure_with_the_bars_still_served() -> None:
    store, log = Rec(), Log()
    store.fail["extend"] = [OhlcvStoreError("coverage_disjoint")] * 2
    res = await get(source(Up(), store, log=log))
    assert (res.store, res.store_reason, len(res.bars)) == ("unavailable", "error", 1000)
    assert len(log.events) == 2 and len(store.of("extend")) == 2


async def test_a_programming_error_in_the_store_is_not_swallowed() -> None:
    store = Rec()
    store.fail["read"] = [RuntimeError("bug")]
    with pytest.raises(RuntimeError):
        await get(source(Up(), store))


@pytest.mark.parametrize("mode", ["read_through", "require"])
async def test_a_leaf_error_on_validated_bars_is_an_invariant_error(
    mode: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Rec()
    store.fail["extend"] = [OhlcvDataError("out_of_range")]  # the rows were valid: a store bug
    with pytest.raises(OhlcvInvariantError) as err:
        await get(source(Up(), store, mode=mode))
    assert err.value.code == "out_of_range" and err.value.__suppress_context__
    assert err.value.__cause__ is None and err.value.args == OhlcvDataError("out_of_range").args

    def broken(*args: object) -> None:
        raise OhlcvDataError("not_increasing")

    monkeypatch.setattr("data.ohlcv_cache.summarise_series", broken)
    with pytest.raises(OhlcvInvariantError) as err:
        await get(source(Up(), Rec(), mode=mode))
    assert err.value.code == "not_increasing" and err.value.__suppress_context__
    assert err.value.__cause__ is None and err.value.args == OhlcvDataError("not_increasing").args


# ---------------------------------------------------------------------------
# bounds, duplicates, invalid rows, identifiers
# ---------------------------------------------------------------------------
async def test_every_store_window_is_bounded_to_the_request_cap() -> None:
    store = Rec()
    await get(source(Up(), store, cap_bars=1000))
    with pytest.raises(
        OhlcvDataError
    ) as err:  # CF-26-52: one bar over the cap: before any store call
        await get(source(Up(), (s2 := Rec()), cap_bars=999))
    assert err.value.code == "out_of_range" and s2.ops == []
    with pytest.raises(OhlcvDataError) as neg:  # SEC-26P5-09: a pre-epoch start, before any I/O
        await source(Up(), (s3 := Rec())).get_range(SYM, 3600, -5 * H, T0 + 5 * H)
    assert neg.value.code == "out_of_range" and s3.ops == []
    up = Up(0, step=60_000)
    up.have = set(range(0, 30_000, 3))
    store = Rec()
    await get(source(up, store, cap_bars=30_000), 0, T0 + 30_000 * 60_000, 60)
    await get(source(up, store, cap_bars=30_000), 0, T0 + 30_000 * 60_000, 60)
    windows = [o for o in store.ops if o[0] in ("read", "read_bars")]
    assert windows and all((o[2] - o[1]) // 60_000 <= 30_000 for o in windows)
    assert all((o[2] - o[1]) // 60_000 <= MAX_EXTEND_BARS for o in store.of("read_bars"))


async def test_duplicate_unsorted_and_out_of_window_rows_are_handled_explicitly() -> None:
    clean = await get(source(Up(), Rec()))
    up = Up()
    up.dup = up.junk = True
    messy = await get(source(up, store := Rec()))
    assert messy.bars == clean.bars and messy.summary == clean.summary
    assert store.of("extend")[0][3] == 999  # no in-batch duplicates reach the store


async def test_the_first_of_two_rows_with_one_timestamp_wins() -> None:
    class Twice(Up):
        async def fetch_range(self, symbol: str, tf_s: int, lo: int, hi: int) -> UpstreamRange:
            got = await super().fetch_range(symbol, tf_s, lo, hi)
            return UpstreamRange([*got.rows, RawRow(T0, ("1", "2", "1", "2", "1"))], True)

    res = await get(source(Twice(5), Rec()), 0, T0 + 5 * H)
    assert res.bars[0].close == Decimal(101)


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (RawRow(T0, ("100", "99", "98", "101", "1")), "bad_ohlc"),
        (RawRow(T0, ("100", "103", "98", "101", "-1")), "bad_volume"),
        (RawRow(T0, (100.5, "103", "98", "101", "1")), "bad_type"),
        (RawRow(T0, ("100", "103", "98", "101", "NaN")), "not_finite"),
        (RawRow(T0, ("1E20", "1E20", "1E20", "1E20", "1")), "out_of_range"),
        (RawRow(T0 + 1, ("100", "103", "98", "101", "1")), "off_grid"),
        (RawRow(1.5 * T0, ("100", "103", "98", "101", "1")), "bad_type"),  # type: ignore[arg-type]
        (RawRow(True, ("100", "103", "98", "101", "1")), "bad_type"),  # type: ignore[arg-type]
    ],
)
async def test_an_invalid_exchange_row_is_refused_and_nothing_is_stored(
    row: RawRow, code: str
) -> None:
    class One(Up):
        async def fetch_range(self, symbol: str, tf_s: int, lo: int, hi: int) -> UpstreamRange:
            return UpstreamRange([row], True)

    store = Rec()
    with pytest.raises(OhlcvDataError) as err:
        await get(source(One(), store), 0, T0 + 5 * H)
    assert (err.value.code, type(err.value)) == (code, OhlcvDataError) and store.of("extend") == []


async def test_identifiers_are_case_sensitive_and_validated_before_any_call() -> None:
    up, store = Up(), Rec()
    await source(up, store).get_range("btc/eur", 3600, T0, T0 + 5 * H)
    assert SeriesKey("coinbase", "btc/eur", 3600) in store._coverage and not store._bars.get(
        SeriesKey("coinbase", SYM, 3600)
    )
    with pytest.raises(OhlcvDataError) as err:
        await source(up, store).get_range("BTC|X/EUR", 3600, T0, END)
    assert err.value.code == "bad_identifier"


async def test_upstream_errors_propagate_and_a_failed_connect_is_retried() -> None:
    class Boom(Up):
        async def connect(self) -> None:
            self.connects += 1
            if self.connects == 1:
                raise ConnectionError("down")

    up, store = Boom(), Rec()
    src = source(up, store)
    with pytest.raises(ConnectionError):
        await get(src)
    assert (await get(src)).store == "ok" and up.connects == 2
    up.bad_check = LookupError("unknown symbol")
    with pytest.raises(LookupError):
        await get(source(up, Rec()))
    up.bad_check = None
    with pytest.raises(ValueError):
        await src.get_range(SYM, 3600, T0 + H, T0 + H)


async def test_connect_runs_once_for_all_series_of_a_request() -> None:
    up = Up()
    got = await source(up, Rec()).get_ranges(["A/EUR", "B/EUR", "C/EUR"], 3600, T0, T0 + 9 * H)
    assert up.connects == 1 and up.checked == 3 and [len(r.bars) for r in got] == [9, 9, 9]


# ---------------------------------------------------------------------------
# offload seam (SY-26-33 / SY-26-53) and the leaf guard vectors (CF-26-61)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n", "seen"),
    [
        (INLINE_MAX_ROWS, []),  # the threshold itself runs inline
        (MAX_EXTEND_BARS + 1, [("_canon", 10_000), ("_canon", 1), ("summarise_series", 10_001)]),
    ],
)
async def test_large_series_are_offloaded_in_slices_small_ones_run_inline(
    n: int, seen: list[tuple[str, int]]
) -> None:
    got: list[tuple[str, int]] = []

    async def offload(fn: Any, *args: Any) -> Any:
        got.append((fn.__name__, len(args[0]) if fn.__name__ == "_canon" else len(args[-1])))
        return fn(*args)

    window = (0, T0 + n * 60_000, 60)
    res = await get(source(Up(n, step=60_000), Rec(), offload=offload), *window)
    assert got == seen
    assert res == await get(source(Up(n, step=60_000), Rec()), *window)


@pytest.mark.parametrize("bad", [1.5, True, "1", None, 2.0])
def test_the_callers_leaf_helpers_refuse_non_int_times(bad: Any) -> None:
    for call in (
        lambda: closed_end_ms(bad, NOW, 3600),
        lambda: closed_end_ms(END, bad, 3600),
        lambda: is_storable(bad, NOW, 3600),
        lambda: is_storable(T0, bad, 3600),
        lambda: check_closed(bad, NOW, 3600),
        lambda: check_closed(T0, bad, 3600),
    ):
        with pytest.raises(OhlcvDataError) as err:
            call()
        assert err.value.code == "bad_type"


# ---------------------------------------------------------------------------
# end to end: the REAL pager (fetch_ohlcv_range_ex) over a fake ccxt exchange
# ---------------------------------------------------------------------------
class _Cx:
    """Minimal ccxt stand-in: ``n`` hourly bars from T0, at most ``cap`` per request."""

    def __init__(self, n: int = 1000, cap: int = 300) -> None:
        self.n, self.cap = n, cap
        self.timeframes = {"1h": "1 hour"}
        self.load_markets = AsyncMock(return_value={SYM: {}})
        self.fetch_ohlcv = AsyncMock(side_effect=self._fetch)

    async def _fetch(self, symbol: str, timeframe: str, since: int, limit: int) -> Any:
        first = -(-(since - T0) // H)
        idx = [i for i in range(first, first + min(limit, self.cap)) if 0 <= i < self.n]
        return [[T0 + i * H, 100.5 + i % 9, 120.0, 90.25, 101.1 + i % 9, 3.5] for i in idx]


class Adapter:
    """The few lines P6 owns: integer ms in, the service's ``RangeFetch`` out as rows."""

    def __init__(self, svc: cmd.CCXTMarketDataService) -> None:
        self.svc = svc

    async def connect(self) -> None:
        await self.svc.connect()

    def check_series(self, symbol: str, timeframe_s: int) -> None:
        self.svc.check_series(symbol, TimeFrame.ONE_HOUR)

    async def fetch_range(self, symbol: str, tf_s: int, lo: int, hi: int) -> UpstreamRange:
        when = [datetime.fromtimestamp(ms / 1000, UTC) for ms in (lo, hi)]
        got = await self.svc.fetch_ohlcv_range_ex(symbol, TimeFrame.ONE_HOUR, *when)
        rows = [
            RawRow(int(b.timestamp.timestamp() * 1000), (b.open, b.high, b.low, b.close, b.volume))
            for b in got.bars
        ]
        return UpstreamRange(rows, got.complete)


async def _real(cx: _Cx, monkeypatch: pytest.MonkeyPatch) -> Adapter:
    monkeypatch.setattr(cmd, "_now_ms", lambda: NOW)
    monkeypatch.setattr(cmd, "_MIN_REQUEST_INTERVAL", 0.0)
    svc = cmd.CCXTMarketDataService("binance", cache_ttl_seconds=0)
    svc._exchange = cx  # type: ignore[assignment]
    return Adapter(svc)


async def test_real_pager_cold_equals_warm_and_a_page_cap_never_advances_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, cx = Rec(), _Cx()
    cold = await get(source(await _real(cx, monkeypatch), store))
    assert cx.fetch_ohlcv.call_count == 4 and cover(store) == Coverage(T0, F - H)
    cx.fetch_ohlcv.reset_mock()
    warm = await get(source(await _real(cx, monkeypatch), store))
    assert cx.fetch_ohlcv.call_count == 1 and warm.bars == cold.bars  # the tail page only
    assert cold.bars[0].open_ == Decimal("100.5") and cold.bars[0].volume == Decimal("3.5")
    capped, fresh = _Cx(cap=1), Rec()  # one bar per request: the pager hits its page cap
    res = await get(source(await _real(capped, monkeypatch), fresh), 0, T0 + 2000 * H)
    assert len(res.bars) < 1000 and cover(fresh) is None and fresh.of("extend") == []


# ---------------------------------------------------------------------------
# review r1 (CR-26P5-02/03/06, SEC-26P5-08): gap, bound, mode and constant edges
# ---------------------------------------------------------------------------
async def test_a_request_left_of_coverage_obeys_the_gap_bound_and_serves_only_the_request() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    up = Up()
    res = await get(source(up, store), 60, T0 + 90 * H)  # gap 10 <= own span 30
    assert up.calls == [(T0 + 60 * H, T0 + 101 * H)] and not res.bypass_disjoint
    assert [c.open_ms for c in res.bars] == [T0 + i * H for i in range(60, 90)]  # no probe bar
    assert cover(store) == Coverage(T0 + 60 * H, T0 + 200 * H)
    far = await get(source(up := Up(), store), 10, T0 + 20 * H)  # gap 40 > own span 10
    assert far.bypass_disjoint and up.calls == [(T0 + 10 * H, T0 + 20 * H)]
    assert cover(store) == Coverage(T0 + 60 * H, T0 + 200 * H)


async def test_the_gap_bound_and_the_combined_bound_are_inclusive() -> None:
    for cap, bypass in ((101, False), (100, True)):  # a gap of 50 == own span 50 needs 101 bars
        store = Rec()
        await get(source(Up(), store), 100, T0 + 200 * H)
        res = await get(source(Up(), store, cap_bars=cap), 250, T0 + 300 * H)
        assert res.bypass_disjoint is bypass


async def test_a_bypass_spends_only_the_bars_it_fetches() -> None:
    store = Rec()
    await get(source(Up(), store), 100, T0 + 200 * H)
    src = source(Up(), store, cap_bars=150)
    first = await get(src, 600, T0 + 610 * H)  # gap 400 > own span 10: bypass, spends 10
    second = await get(src, 230, T0 + 300 * H)  # needs 101: 10 + 101 <= 150
    assert (first.bypass_disjoint, second.bypass_disjoint) == (True, False)


async def test_require_serves_an_incomplete_fetch_that_has_nothing_to_store() -> None:
    up, store = Up(complete=False), Rec()
    res = await get(source(up, store, mode="require"), 999)  # slot 999 is the volatile tail
    assert len(res.bars) == 1 and store.ops == []


async def test_a_right_extension_is_chunked_from_its_probe_not_from_the_request_start() -> None:
    up, store = Up(25_000, step=60_000), Rec()
    await get(source(up, store), 0, T0 + 100 * 60_000, 60)
    await get(source(up, store), 0, T0 + 25_000 * 60_000, 60)
    spans = [((o[1] - T0) // 60_000, (o[2] - T0) // 60_000) for o in store.of("extend")]
    assert spans == [(0, 100), (99, 10_099), (10_099, 20_099), (20_099, 25_000)]


async def test_a_left_probe_never_reaches_past_the_settle_of_an_earlier_now() -> None:
    store = Rec()
    await get(source(Up(), store), 400, T0 + 600 * H)
    up = Up()
    early = HistoricalBarSource(up, store, exchange="coinbase", now_ms=T0 + 401 * H + 5)
    await early.get_range(SYM, 3600, T0 + 390 * H, T0 + 600 * H)
    assert up.calls[0] == (T0 + 390 * H, T0 + 400 * H)  # then the tail span


@pytest.mark.parametrize("mode", ["off", "REQUIRE", "requre", "", None])
def test_an_unknown_mode_is_refused_so_require_cannot_fail_open_by_a_typo(mode: Any) -> None:
    with pytest.raises(ValueError):
        source(Up(), Rec(), mode=mode)


async def test_one_incomplete_span_marks_the_result_and_the_epoch_is_a_valid_start() -> None:
    store, up = Rec(), Up(complete=[False, True])  # type: ignore[arg-type]  # left cut, right not
    await get(source(Up(), store), 100, T0 + 200 * H)
    assert (await get(source(up, store), 50, T0 + 300 * H)).fetch_complete is False  # N07
    assert (await source(Up(0), Rec()).get_range(SYM, 3600, 0, 10 * H)).from_ms == 0  # N05
