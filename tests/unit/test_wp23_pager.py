"""
tests/unit/test_wp23_pager.py
-----------------------------
WP2.3 commit P0 (SY-23-12): ``CCXTMarketDataService.fetch_ohlcv_range`` pagination.

HEAD stopped at the first short or empty page, which silently truncated windowed
exchanges (Coinbase omits buckets without trades: 299 of 800 / 299 of 2000 bars).
The fix: ``cursor = last_ts + tf`` after a non-empty page, ``cursor += page_limit * tf``
after an empty one, never stop on a short page, stop at
``min(end, floor_tf(now) + tf)`` (``_now_ms`` seam), bounded page cap with a warning,
no-progress guard kept.

Covers WP23-T-01 (pager) and WP23-T-02 (live trailing-gap path, ``_resume_trailing_peak``).
Hermetic: fake exchanges only, no network, the clock is patched through ``_now_ms``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt.async_support as ccxt_async
import pytest
import structlog
from structlog.testing import capture_logs

from common.models import OHLCVBar
from common.types import TimeFrame
from data.services import ccxt_market_data as cmd
from data.services.ccxt_market_data import CCXTMarketDataService
from trading import strategy_engine as se

_SYMBOL = "BTC/USDT"
_TF_MS = 3_600_000
_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_T0_MS = int(_T0.timestamp() * 1000)
#: Fixed "now" (2025-06-01T00:30Z) so that no test depends on the wall clock.
_NOW = datetime(2025, 6, 1, 0, 30, tzinfo=UTC)
_NOW_MS = int(_NOW.timestamp() * 1000)


def _row(ts_ms: int, close: float = 102.0) -> list[int | float]:
    return [ts_ms, close, close + 1.0, close - 1.0, close, 1_000.0]


def _slot(i: int) -> int:
    return _T0_MS + i * _TF_MS


class _FakeExchange:
    """Fake CCXT exchange over a set of existing slot indexes.

    ``mode="window"``  mimics ccxt Coinbase (coinbase.py:3640-3666): the answer is every
    existing bucket in ``[since, since + min(limit, max_limit) * tf)``; a window without
    trades is an EMPTY page although later windows hold data.
    ``mode="count"``   mimics Binance: the first ``min(limit, max_limit)`` existing
    buckets with ``ts >= since``; an empty page means no data at or after ``since``.
    """

    def __init__(
        self,
        slots: set[int],
        *,
        mode: str = "window",
        max_limit: int = 300,
        closes: dict[int, float] | None = None,
        containing: bool = False,
    ) -> None:
        self.containing = containing
        self.slots = slots
        self.mode = mode
        self.max_limit = max_limit
        self.closes = closes or {}
        self.sinces: list[int] = []

        self.load_markets = AsyncMock(return_value={_SYMBOL: {}})
        self.close = AsyncMock()
        self.set_sandbox_mode = MagicMock()
        self.timeframes = {"1h": "1 hour"}
        self.fetch_ohlcv = AsyncMock(side_effect=self._fetch)

    async def _fetch(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[int | float]]:
        self.sinces.append(since)
        n = min(limit, self.max_limit)
        if self.containing:  # the bucket CONTAINING ``since`` (unaligned start)
            first = (since - _T0_MS) // _TF_MS
        else:
            first = -(-(since - _T0_MS) // _TF_MS)  # first slot index with ts >= since
        if self.mode == "window":
            wanted = [i for i in range(first, first + n) if i in self.slots]
        else:
            wanted = sorted(i for i in self.slots if i >= first)[:n]
        return [_row(_slot(i), self.closes.get(i, 102.0)) for i in wanted]

    @property
    def calls(self) -> int:
        return int(self.fetch_ohlcv.call_count)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmd, "_now_ms", lambda: _NOW_MS)
    monkeypatch.setattr(cmd, "_MIN_REQUEST_INTERVAL", 0.0)


def _stuttering_exchange(first: int) -> _FakeExchange:
    """Misbehaving exchange: ``first`` bars on the first page, then ONE bar per page."""
    ex = _FakeExchange(set(range(10_000)), mode="count")

    async def fetch(symbol: str, timeframe: str, since: int, limit: int) -> list[list[int | float]]:
        ex.sinces.append(since)
        idx = (since - _T0_MS) // _TF_MS
        n = first if len(ex.sinces) == 1 else 1
        return [_row(_slot(idx + k)) for k in range(n)]

    ex.fetch_ohlcv = AsyncMock(side_effect=fetch)
    return ex


def _sized_exchange(sizes: list[int]) -> _FakeExchange:
    """Misbehaving exchange: call n returns ``sizes[n]`` contiguous bars (1 after the list)."""
    ex = _FakeExchange(set(), mode="count")

    async def fetch(symbol: str, timeframe: str, since: int, limit: int) -> list[list[int | float]]:
        n = sizes[len(ex.sinces)] if len(ex.sinces) < len(sizes) else 1
        ex.sinces.append(since)
        idx = (since - _T0_MS) // _TF_MS
        return [_row(_slot(idx + k)) for k in range(n)]

    ex.fetch_ohlcv = AsyncMock(side_effect=fetch)
    return ex


async def _service(ex: _FakeExchange, page_limit: int = 300) -> CCXTMarketDataService:
    with patch("data.services.ccxt_market_data.ccxt_async") as patched:
        patched.binance = MagicMock(return_value=ex)
        for name in (
            "AuthenticationError",
            "NetworkError",
            "ExchangeError",
            "RateLimitExceeded",
            "BadSymbol",
            "BadRequest",
            "ExchangeNotAvailable",
        ):
            setattr(patched, name, getattr(ccxt_async, name))
        svc = CCXTMarketDataService(
            exchange_id="binance", cache_ttl_seconds=0, page_limit=page_limit
        )
    await svc.connect()
    return svc


async def _range(
    ex: _FakeExchange, n_slots: int, page_limit: int = 300, end_slot: int | None = None
) -> list[OHLCVBar]:
    svc = await _service(ex, page_limit)
    end = _T0 + timedelta(hours=n_slots if end_slot is None else end_slot)
    return await svc.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, _T0, end)


# ---------------------------------------------------------------------------
# WP23-T-01  windowed (Coinbase-like) exchange
# ---------------------------------------------------------------------------


class TestWindowedExchange:
    async def test_t01_2000_requested_one_missing_returns_1999(self) -> None:
        ex = _FakeExchange(set(range(2000)) - {150})
        bars = await _range(ex, 2000)
        assert len(bars) == 1999  # HEAD: 149 (first window short)

    async def test_t01_2000_requested_150_missing_returns_all_existing(self) -> None:
        missing = set(range(7, 2000, 13))  # 154 scattered holes, one or more per window
        assert len(missing) >= 150
        ex = _FakeExchange(set(range(2000)) - missing)
        bars = await _range(ex, 2000)
        assert len(bars) == 2000 - len(missing)  # HEAD: 299
        assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)

    async def test_t01_800_requested_one_missing_returns_799(self) -> None:
        ex = _FakeExchange(set(range(800)) - {250})
        bars = await _range(ex, 800)
        assert len(bars) == 799  # HEAD: 250
        assert ex.calls == 3  # cursor 0, 299... windows tile the range, no extra call

    async def test_t01_empty_300_slot_window_mid_range_is_skipped(self) -> None:
        # slots 300..599 never traded: the 2nd window is an empty page.
        slots = set(range(0, 300)) | set(range(600, 900))
        ex = _FakeExchange(slots)
        bars = await _range(ex, 900)
        assert len(bars) == 600  # HEAD: 300 (stopped at the empty page)
        assert ex.calls == 3

    async def test_t01_empty_first_window_does_not_drop_later_data(self) -> None:
        ex = _FakeExchange(set(range(300, 700)))  # start before the listing
        bars = await _range(ex, 700)
        assert len(bars) == 400  # HEAD: 0
        assert bars[0].timestamp == _T0 + timedelta(hours=300)

    async def test_t01_cursor_uses_last_ts_plus_tf_not_page_limit(self) -> None:
        """G2: the exchange maximum (100) is below page_limit (300): no bar may be skipped."""
        ex = _FakeExchange(set(range(450)), mode="count", max_limit=100)
        bars = await _range(ex, 450)
        assert len(bars) == 450
        assert ex.sinces[:5] == [_slot(0), _slot(100), _slot(200), _slot(300), _slot(400)]

    async def test_t01_window_exchange_max_below_page_limit_with_holes(self) -> None:
        ex = _FakeExchange(set(range(450)) - {50, 51, 52}, mode="window", max_limit=100)
        bars = await _range(ex, 450)
        assert len(bars) == 447


# ---------------------------------------------------------------------------
# WP23-T-01  count-based exchange: HEAD-identical bars, bounded extra calls
# ---------------------------------------------------------------------------


class TestCountBasedExchange:
    async def test_t01_count_based_data_to_end_same_bars_same_calls(self) -> None:
        ex = _FakeExchange(set(range(650)), mode="count")
        bars = await _range(ex, 650)
        assert [b.timestamp for b in bars] == [_T0 + timedelta(hours=i) for i in range(650)]
        assert ex.calls == 3  # HEAD: 300 + 300 + 50(short) = 3; cursor reaches end: no extra

    async def test_t01_count_based_end_beyond_data_at_most_one_extra_call(self) -> None:
        ex = _FakeExchange(set(range(650)), mode="count")
        bars = await _range(ex, 650, end_slot=900)  # end is < page_limit*tf past the data
        assert len(bars) == 650
        assert ex.calls <= 3 + 1  # HEAD 3 calls

    async def test_t01_count_based_data_far_before_end_is_bounded_by_cap(self) -> None:
        ex = _FakeExchange(set(range(10)), mode="count")
        bars = await _range(ex, 3000)
        assert len(bars) == 10
        # empty windows after the data: 1 data page + ceil(2990 / 300) empty ones, below the cap
        assert ex.calls <= 2 * -(-3000 // 300) + 2
        assert ex.calls == 1 + 10  # exact: cursor 10, 310, ..., 3010

    async def test_t01_page_limit_exact_multiple_no_phantom_call(self) -> None:
        ex = _FakeExchange(set(range(600)), mode="count")
        bars = await _range(ex, 600)
        assert len(bars) == 600
        assert ex.calls == 2


# ---------------------------------------------------------------------------
# WP23-T-01  clock seam: future ``end``
# ---------------------------------------------------------------------------


class TestFutureEnd:
    async def test_t01_future_end_stops_at_floor_now_plus_tf(self) -> None:
        # now = 2025-06-01T00:30Z -> forming bar 00:00, stop = 01:00.
        floor_idx = (_NOW_MS - _T0_MS) // _TF_MS
        ex = _FakeExchange(set(range(floor_idx + 1)), mode="window")
        svc = await _service(ex)
        end = _NOW + timedelta(days=400)
        bars = await svc.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, _T0, end)
        assert len(bars) == floor_idx + 1  # includes the forming bar, as HEAD did
        stop_ms = _slot(floor_idx + 1)
        assert max(ex.sinces) < stop_ms
        assert ex.calls == -(-(floor_idx + 1) // 300)

    async def test_t01_start_at_or_after_stop_makes_no_request(self) -> None:
        ex = _FakeExchange(set())
        svc = await _service(ex)
        start = _NOW + timedelta(hours=5)
        bars = await svc.fetch_ohlcv_range(
            _SYMBOL, TimeFrame.ONE_HOUR, start, start + timedelta(hours=10)
        )
        assert bars == []
        assert ex.calls == 0

    async def test_t01_now_is_read_only_through_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []

        def fake_now() -> int:
            seen.append(1)
            return _NOW_MS

        monkeypatch.setattr(cmd, "_now_ms", fake_now)
        ex = _FakeExchange(set(range(5)))
        await _range(ex, 5)
        assert seen == [1]  # once per call, up front

    def test_t01_seam_returns_int_milliseconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.undo()  # the autouse patch would hide the real function
        value = cmd._now_ms()
        assert isinstance(value, int)
        assert value > 1_700_000_000_000


# ---------------------------------------------------------------------------
# guards: end boundary, no-progress, page cap, log hygiene
# ---------------------------------------------------------------------------


class TestGuards:
    async def test_end_boundary_excludes_bars_at_or_after_end(self) -> None:
        ex = _FakeExchange(set(range(100)), mode="count")
        bars = await _range(ex, 10)
        assert len(bars) == 10
        assert ex.calls == 1

    async def test_no_progress_guard_is_kept(self) -> None:
        """A page whose bars are all duplicates ends the fetch (misbehaving exchange)."""
        ex = _FakeExchange(set())
        ex.fetch_ohlcv = AsyncMock(side_effect=lambda **kw: [_row(_slot(i)) for i in range(5)])
        with capture_logs() as logs:
            bars = await _range(ex, 100, page_limit=5)
        assert len(bars) == 5
        assert ex.calls == 2
        assert [e for e in logs if e["event"] == "fetch_ohlcv_range_no_progress"]

    async def test_page_cap_bounds_requests_and_warns(self) -> None:
        """A first page of 5 then one bar per page: the cap (2*ceil(100/5)+2 = 42) ends it."""
        ex = _stuttering_exchange(first=5)
        with capture_logs() as logs:
            bars = await _range(ex, 100, page_limit=5)
        assert ex.calls == 42
        assert len(bars) == 5 + 41  # what was gathered is returned
        caps = [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]
        assert len(caps) == 1
        assert caps[0]["log_level"] == "warning"
        assert caps[0]["pages"] == 42
        assert caps[0]["max_pages"] == 42

    async def test_cr01_small_exchange_maximum_is_not_cut_by_the_cap(self) -> None:
        """900 slots, count-based exchange maximum 100, page_limit 300: all 900, no warning."""
        ex = _FakeExchange(set(range(900)), mode="count", max_limit=100)
        with capture_logs() as logs:
            bars = await _range(ex, 900)
        assert len(bars) == 900  # page_limit-based cap (8) returned 800
        assert ex.calls == 9
        assert not [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]

    async def test_cr01_cap_stays_bounded_with_a_one_bar_first_page(self) -> None:
        """eff >= 1 keeps the cap at most 2*ceil(span/tf)+2 requests, whatever the first page."""
        ex = _stuttering_exchange(first=1)
        with capture_logs() as logs:
            await _range(ex, 100, page_limit=5)
        assert ex.calls <= 2 * 100 + 2
        assert not [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]

    async def test_r2_sec01_one_bar_pages_are_capped_near_the_page_limit_cap(self) -> None:
        """1 bar per page over 3000 slots: eff is floored at 300 // 4 = 75, so the cap is
        2*ceil(3000/75)+2 = 82 requests (v3: 3000 requests, one per slot)."""
        ex = _sized_exchange([1])
        with capture_logs() as logs:
            bars = await _range(ex, 3000)
        assert ex.calls == 82
        assert len(bars) == 82
        caps = [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]
        assert len(caps) == 1
        assert caps[0]["max_pages"] == 82

    async def test_r2_sec01_a_full_page_later_tightens_the_cap(self) -> None:
        """A sparse first page (1 bar), then a full page: eff follows the longest page (300),
        so the cap drops to the page_limit cap 2*ceil(3000/300)+2 = 22."""
        ex = _sized_exchange([1, 300])
        with capture_logs() as logs:
            bars = await _range(ex, 3000)
        assert ex.calls == 22
        assert len(bars) == 1 + 300 + 20
        caps = [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]
        assert [c["max_pages"] for c in caps] == [22]

    async def test_r2_sec01_floor_still_reads_an_exchange_maximum_of_40(self) -> None:
        """Count-based exchange maximum 40 (< floor 75): cap 2*ceil(1200/75)+2 = 34 >= the 30
        requests needed, so all 1200 bars come back without a warning."""
        ex = _FakeExchange(set(range(1200)), mode="count", max_limit=40)
        with capture_logs() as logs:
            bars = await _range(ex, 1200)
        assert len(bars) == 1200
        assert ex.calls == 30
        assert not [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]

    async def test_page_cap_not_hit_on_a_normal_fetch(self) -> None:
        ex = _FakeExchange(set(range(2000)) - {150})
        with capture_logs() as logs:
            await _range(ex, 2000)
        assert not [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]

    async def test_complete_log_carries_pages_and_empty_pages(self) -> None:
        ex = _FakeExchange(set(range(0, 300)) | set(range(600, 900)))
        with capture_logs() as logs:
            await _range(ex, 900)
        done = [e for e in logs if e["event"] == "fetch_ohlcv_range_complete"]
        assert len(done) == 1
        assert done[0]["pages"] == 3
        assert done[0]["empty_pages"] == 1
        assert done[0]["bars_fetched"] == 600

    async def test_cf36_new_events_log_numbers_only(self) -> None:
        ex = _stuttering_exchange(first=5)
        with capture_logs() as logs:
            await _range(ex, 100, page_limit=5)
        assert [e for e in logs if e["event"] == "fetch_ohlcv_range_page_cap"]
        for ev in logs:
            if ev["event"] in {"fetch_ohlcv_range_page_cap", "fetch_ohlcv_range_complete"}:
                for key, val in ev.items():
                    if key in {"event", "log_level", "symbol", "timeframe", "exchange"}:
                        continue
                    assert isinstance(val, int), (ev["event"], key)

    async def test_exchange_error_text_never_reaches_a_log(self) -> None:
        ex = _FakeExchange(set(range(10)))
        ex.fetch_ohlcv = AsyncMock(side_effect=ccxt_async.BadSymbol("secret-url-text"))
        svc = await _service(ex)
        with capture_logs() as logs, pytest.raises(Exception):  # noqa: B017
            await svc.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, _T0, _T0 + timedelta(hours=5))
        # The traceback lives only in the DEBUG ``<event>.trace`` event (CF-36 layer 2);
        # every other event, including this method's own, carries no exception text.
        assert "secret-url-text" not in repr(
            [e for e in logs if not str(e["event"]).endswith(".trace")]
        )


# ---------------------------------------------------------------------------
# WP23-T-02  live trailing-gap path (StrategyEngine._resume_trailing_peak)
# ---------------------------------------------------------------------------


def _bar(i: int, close: float) -> OHLCVBar:
    return OHLCVBar(
        symbol=_SYMBOL,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=_T0 + timedelta(hours=i),
        open=Decimal(str(close)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 1)),
        close=Decimal(str(close)),
        volume=Decimal("1000"),
    )


async def _peak(
    svc: Any, gap_start: int, warm_start: int, *, entry: str = "50"
) -> tuple[Decimal, list[dict[str, Any]], AsyncMock]:
    """Run the real ``_resume_trailing_peak`` against ``svc`` for a gap [gap_start, warm_start)."""
    wrapped = AsyncMock(side_effect=svc.fetch_ohlcv_range)
    fake_self = SimpleNamespace(
        _log=structlog.get_logger("p0-test"),
        _tf_s=3600,
        _timeframe=TimeFrame.ONE_HOUR,
        _market_data=SimpleNamespace(fetch_ohlcv_range=wrapped),
        _metric=MagicMock(),
    )
    rule_ts = _T0 + timedelta(hours=gap_start - 1)
    warm = [_bar(warm_start, 100.0), _bar(warm_start + 1, 100.0)]
    with capture_logs() as logs:
        peak = await se.StrategyEngine._resume_trailing_peak(
            fake_self,  # type: ignore[arg-type]
            _SYMBOL,
            warm,
            rule_ts,
            SimpleNamespace(entry_gross_vwap=Decimal(entry)),  # type: ignore[arg-type]
        )
    return peak, logs, wrapped


class TestLiveTrailingGapPath:
    async def test_t02_gapped_range_peak_not_below_head(self) -> None:
        # Gap = slots 1..600, slot 100 never traded; the highest close sits at slot 500.
        closes = {i: 100.0 + i * 0.01 for i in range(1, 601)}
        ex = _FakeExchange(set(range(1, 601)) - {100}, closes=closes)
        svc = await _service(ex)
        peak, logs, _ = await _peak(svc, gap_start=1, warm_start=601)
        head_peak = max(
            Decimal(str(closes[i])) for i in range(1, 301) if i != 100
        )  # HEAD: first page
        assert peak >= head_peak
        assert peak == Decimal(str(closes[600]))  # the full gap was recovered
        # one hole remains, so len(gap) != missing: seed_partial keeps firing (CF-23-16)
        assert [e for e in logs if e["event"] == "trailing_stop.seed_partial"]

    async def test_t02_dense_1000_bar_gap_uses_at_most_4_pages(self) -> None:
        closes = {i: 100.0 + i * 0.001 for i in range(1, 1001)}
        ex = _FakeExchange(set(range(1, 1001)), closes=closes)
        svc = await _service(ex)
        peak, logs, wrapped = await _peak(svc, gap_start=1, warm_start=1001)
        assert wrapped.call_count == 1
        assert ex.calls <= se._TRAILING_GAP_MAX_PAGES
        assert ex.calls == 4
        assert peak == Decimal(str(closes[1000]))
        assert not [e for e in logs if e["event"] == "trailing_stop.seed_partial"]

    async def test_t02_timeout_path_still_logs_seed_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def never(**_: Any) -> list[OHLCVBar]:
            await asyncio.sleep(3600)
            return []

        monkeypatch.setattr(se, "_EXIT_FETCH_TIMEOUT_S", 0.01)
        fake_self = SimpleNamespace(
            _log=structlog.get_logger("p0-test"),
            _tf_s=3600,
            _timeframe=TimeFrame.ONE_HOUR,
            _market_data=SimpleNamespace(fetch_ohlcv_range=never),
            _metric=MagicMock(),
        )
        warm = [_bar(11, 100.0)]
        with capture_logs() as logs:
            peak = await se.StrategyEngine._resume_trailing_peak(
                fake_self,  # type: ignore[arg-type]
                _SYMBOL,
                warm,
                _T0,
                SimpleNamespace(entry_gross_vwap=Decimal("50")),  # type: ignore[arg-type]
            )
        assert peak == Decimal("100.0")
        assert [e for e in logs if e["event"] == "trailing_stop.seed_partial"]
        assert [e for e in logs if e["event"] == "engine.exit_gap_fetch_failed"]


# ---------------------------------------------------------------------------
# SEC-03: the pager never returns a bar outside [start, stop)
# ---------------------------------------------------------------------------


def _scripted(pages: list[list[int]]) -> _FakeExchange:
    """Exchange that answers call n with the slot indexes ``pages[n]`` (then empty pages)."""
    ex = _FakeExchange(set())
    queue = list(pages)

    async def fetch(symbol: str, timeframe: str, since: int, limit: int) -> list[list[int | float]]:
        ex.sinces.append(since)
        return [_row(_slot(i)) for i in (queue.pop(0) if queue else [])]

    ex.fetch_ohlcv = AsyncMock(side_effect=fetch)
    return ex


class TestWindowBounds:
    async def test_sec03_pre_start_bar_is_dropped(self) -> None:
        ex = _scripted([[-1, 0, 1]])
        bars = await _range(ex, 2)
        assert [b.timestamp for b in bars] == [_T0, _T0 + timedelta(hours=1)]

    async def test_sec03_page_of_only_pre_start_bars_fails_closed(self) -> None:
        ex = _scripted([[-2, -1], [0, 1, 2]])
        with capture_logs() as logs:
            bars = await _range(ex, 3)
        assert bars == []
        assert ex.calls == 1  # no-progress guard ends the fetch
        assert [e for e in logs if e["event"] == "fetch_ohlcv_range_no_progress"]

    async def test_sec03_bar_between_stop_and_end_is_dropped(self) -> None:
        floor_idx = (_NOW_MS - _T0_MS) // _TF_MS  # the forming bar
        ex = _scripted([[floor_idx - 1, floor_idx, floor_idx + 1, floor_idx + 2]])
        svc = await _service(ex)
        end = _NOW + timedelta(days=30)  # end > now
        bars = await svc.fetch_ohlcv_range(
            _SYMBOL, TimeFrame.ONE_HOUR, _T0 + timedelta(hours=floor_idx - 1), end
        )
        assert [b.timestamp for b in bars] == [
            _T0 + timedelta(hours=floor_idx - 1),
            _T0 + timedelta(hours=floor_idx),
        ]
        assert ex.calls == 1

    async def test_sec03_unaligned_start_drops_the_containing_bucket(self) -> None:
        ex = _FakeExchange(set(range(10)), mode="window", containing=True)
        svc = await _service(ex)
        start = _T0 + timedelta(minutes=30)
        bars = await svc.fetch_ohlcv_range(
            _SYMBOL, TimeFrame.ONE_HOUR, start, _T0 + timedelta(hours=5)
        )
        assert [b.timestamp for b in bars] == [_T0 + timedelta(hours=h) for h in (1, 2, 3, 4)]
        assert all(b.timestamp >= start for b in bars)


# ---------------------------------------------------------------------------
# SEC-05 / SEC-06: Retry-After cap, page_limit validation
# ---------------------------------------------------------------------------


class TestHardening:
    async def test_sec05_retry_after_is_capped_at_backoff_cap(self) -> None:
        ex = _FakeExchange(set(range(3)))
        rows = [_row(_slot(i)) for i in range(3)]
        ex.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.RateLimitExceeded("retry-after: 86400"), rows]
        )
        svc = await _service(ex)
        with patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock) as sl:
            await svc.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, _T0, _T0 + timedelta(hours=3))
        waits = [c.args[0] for c in sl.call_args_list]
        assert waits  # the retry did sleep (patched, not real)
        assert max(waits) <= cmd._BACKOFF_CAP
        assert max(waits) == cmd._BACKOFF_CAP

    async def test_sec05_small_retry_after_is_unchanged(self) -> None:
        ex = _FakeExchange(set(range(3)))
        rows = [_row(_slot(i)) for i in range(3)]
        ex.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.RateLimitExceeded("retry-after: 7"), rows]
        )
        svc = await _service(ex)
        with patch("data.services.ccxt_market_data.asyncio.sleep", new_callable=AsyncMock) as sl:
            await svc.fetch_ohlcv_range(_SYMBOL, TimeFrame.ONE_HOUR, _T0, _T0 + timedelta(hours=3))
        assert max(c.args[0] for c in sl.call_args_list) == pytest.approx(7.0)

    @pytest.mark.parametrize("bad", [0, -1, -300])
    def test_sec06_page_limit_below_one_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match=r"^page_limit must be >= 1$"):
            CCXTMarketDataService(exchange_id="binance", page_limit=bad)
