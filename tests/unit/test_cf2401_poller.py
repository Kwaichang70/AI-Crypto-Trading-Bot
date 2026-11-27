"""
tests/unit/test_cf2401_poller.py
---------------------------------
CF-24-01 P5 (ContextPoller): T-06 (P5 clause), T-25 (poller), T-26, T-27, T-30, T-31, T-32, the
G-C 48 h fake-clock soak and the carry-forwards CF-2401-32/33/34/35/36/41/45/49/50/52/53/54/59/
60/62/64/65/66/69 of reports/vp2-cf2401.

No network: a discrete-event ``FakeClock`` drives the poller and every client is a scripted fake
(sockets are blocked). The one loopback class talks to a raw asyncio server on 127.0.0.1.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import heapq
import itertools
import random
import re
import socket
import sys
import tracemalloc
import types
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

import api.services.context_poller as cp
from api.container import AppContainer
from api.routers import health as health_router
from api.routers import signals as signals_router
from api.services.context_poller import ContextPoller, derive_status, parse_retry_after
from common.metrics import metrics
from data import live_context_policy as pol
from data.live_context import LIVE_BOOK, LIVE_CONSUMERS, LiveContextBook, t10y2y_observations
from data.macro_data import MacroSnapshot
from data.market_signals import CoinGeckoSnapshot
from data.point_in_time import SERIES_RULES, Observation, PitSeries, to_ms
from data.provider_http import Err, Ok
from data.sentiment import FearGreedClient, FearGreedSnapshot
from data.whale_tracker import WhaleFlowSnapshot
from tests._cf36_logging import SENTINEL, ConfigureFn, disable_layer1

_T = TypeVar("_T")
H = 3_600_000
MIN = 60_000
START = to_ms(datetime(2024, 3, 11, 12, 0, tzinfo=UTC))  # a Monday, on a bucket boundary
FGI, DOM, FRED, WHALE = (
    "alternative_me:fgi",
    "coingecko:btc_dominance",
    "fred:T10Y2Y",
    "whale:net_flow_1h",
)
_SRC = (Path(__file__).resolve().parents[2] / "apps/api/services/context_poller.py").read_text()


def dt(ms: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)


# ---------------------------------------------------------------------------
# Harness: discrete-event clock, scripted clients, spies
# ---------------------------------------------------------------------------
class FakeClock:
    """Wall = START + advanced seconds + ``offset_ms``; sleepers wake in time order."""

    def __init__(self, wall_ms: int = START) -> None:
        self.wall0, self.t, self.offset_ms, self._seq = wall_ms, 0.0, 0, 0
        self._timers: list[tuple[float, int, asyncio.Future[None]]] = []
        self.sleeps: list[tuple[str, float]] = []  # (task name, seconds) of poller sleeps
        self.on_sleep: Callable[[], None] | None = None

    def wall_ms(self) -> int:
        return self.wall0 + int(self.t * 1000) + self.offset_ms

    def monotonic(self) -> float:
        return 10_000.0 + self.t

    def _timer(self, seconds: float) -> asyncio.Future[None]:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._timers, (self.t + max(0.0, seconds), self._seq, fut))
        return fut

    async def sleep(self, seconds: float) -> None:
        name = asyncio.current_task().get_name()  # type: ignore[union-attr]
        self.sleeps.append((name, seconds))
        if self.on_sleep:
            self.on_sleep()
        await self._timer(seconds)

    async def latency(self, seconds: float) -> None:  # a fetch taking time; not a poller sleep
        await self._timer(seconds)

    async def bounded(self, awaitable: Awaitable[_T], seconds: float) -> _T:
        task = asyncio.ensure_future(awaitable)
        timer = self._timer(seconds)
        try:
            await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            task.cancel()
            raise
        if task.done():
            timer.cancel()
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise TimeoutError

    async def advance(self, seconds: float) -> None:
        end = self.t + seconds
        await self._settle()
        while self._timers and self._timers[0][0] <= end:
            when, _, fut = heapq.heappop(self._timers)
            self.t = max(self.t, when)
            if not fut.done():
                fut.set_result(None)
            await self._settle()
        self.t = end
        await self._settle()

    @staticmethod
    async def _settle() -> None:
        for _ in range(60):
            await asyncio.sleep(0)


class AutoClock(FakeClock):
    """For tests that drive ``_cycle`` directly: a sleep just advances the time (no timeouts)."""

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(("auto", seconds))
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)

    async def latency(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)

    async def bounded(self, awaitable: Awaitable[_T], seconds: float) -> _T:
        return await awaitable


class Probe:
    def __init__(self) -> None:
        self.active = self.max_active = 0


class FakeClient:
    """``script`` items: a result, a ``BaseException`` to raise, or ``callable(kwargs)``."""

    def __init__(
        self,
        clock: FakeClock,
        script: list[Any],
        *,
        latency: float = 0.0,
        probe: Probe | None = None,
    ) -> None:
        self.clock, self.script, self.latency = clock, list(script), latency
        self.probe = probe or Probe()
        self.calls: list[tuple[float, dict[str, Any]]] = []
        self.adopted: list[tuple[Any, float]] = []
        self.active = self.max_active = 0

    async def fetch_validated(self, **kw: Any) -> Any:
        self.calls.append((self.clock.t, kw))
        self.active += 1
        self.probe.active += 1
        self.max_active = max(self.max_active, self.active)
        self.probe.max_active = max(self.probe.max_active, self.probe.active)
        try:
            if self.latency:
                await self.clock.latency(self.latency)
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
            if isinstance(item, BaseException):
                raise item
            return item(kw) if callable(item) else item
        finally:
            self.active -= 1
            self.probe.active -= 1

    def adopt_validated(self, snapshot: Any, *, now_mono: float) -> None:
        self.adopted.append((snapshot, now_mono))


def rnd(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 -- seeded jitter, not a secret


class ConstRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class SpySink:
    def __init__(self) -> None:
        self.offered: list[tuple[str, Observation]] = []

    def offer(self, series_id: str, observation: Observation) -> None:
        self.offered.append((series_id, observation))

    async def flush(self, timeout_s: float) -> None:
        return None


def fgi_ok(*, newest_day: int = START - START % (24 * H), n: int = 30) -> Ok[Any]:
    return Ok(
        [
            FearGreedSnapshot(
                value=40 + i % 10, classification="Fear", timestamp=dt(newest_day - i * 24 * H)
            )
            for i in range(n)
        ]
    )


def dom_ok(value: float = 52.5) -> Ok[Any]:
    return Ok(
        CoinGeckoSnapshot(
            btc_dominance=value,
            market_cap_change_24h=1.0,
            total_volume_change_24h=2.0,
            timestamp=dt(START),
        )
    )


def whale_ok(value: float = 1000.0) -> Ok[Any]:
    return Ok(WhaleFlowSnapshot(net_flow=value, large_tx_count=3, timestamp=dt(START)))


def fred_ok(rows: list[tuple[date, str]] | None = None) -> Ok[Any]:
    rows = rows or [(date(2024, 3, 4 + i), f"0.{40 + i}") for i in range(5)]
    pairs = tuple((d, Decimal(v)) for d, v in rows)
    return Ok(
        MacroSnapshot(
            yield_curve_spread=float(pairs[-1][1]),
            timestamp=dt(START),
            yield_curve_observed_on=pairs[-1][0],
            yield_curve_rows=pairs,
        )
    )


_LOOPBACK_TESTS = ("test_a_real_aiohttp", "test_a_gzip_bomb")  # the only tests that may use sockets


@pytest.fixture(autouse=True)
def _isolated(request: pytest.FixtureRequest) -> Iterator[None]:
    metrics.reset()

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("socket use in a poller unit test")

    patch = pytest.MonkeyPatch()  # not the shared fixture: it is undone here, nowhere else
    if not request.node.name.startswith(_LOOPBACK_TESTS):
        patch.setattr(socket.socket, "connect", boom)
    yield
    patch.undo()
    metrics.reset()


def counter(name: str, **labels: str) -> int:
    key = name + (
        "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else ""
    )
    return int(metrics.get_all()["counters"].get(key, 0))


def gauge(name: str, **labels: str) -> float | None:
    key = name + (
        "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else ""
    )
    return metrics.get_all()["gauges"].get(key)


def make(clock: FakeClock, rng: Any = None, **clients: Any) -> ContextPoller:
    return ContextPoller(clock=clock, rng=rng or ConstRng(0.0), **clients)


async def run(poller: ContextPoller, clock: FakeClock, seconds: float) -> None:
    poller.start()
    try:
        await clock.advance(seconds)
    finally:
        await poller.stop()


def starts_of(sid: str, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the monotonic start of every settled cycle of ``sid``."""
    seen: list[float] = []
    original = ContextPoller._settle

    def spy(self: ContextPoller, p: Any, result: Any, fetch_ms: int, start: float) -> float:
        if p.series_id == sid:
            seen.append(start - 10_000.0)
        return original(self, p, result, fetch_ms, start)

    monkeypatch.setattr(ContextPoller, "_settle", spy)
    return seen


# ---------------------------------------------------------------------------
# T-26: cadence, jitter, buckets, budgets, single-flight, supervisor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("sid", "interval", "seconds"), [(FGI, 3_600, 12 * 3_600), (FRED, 21_600, 4 * 86_400)]
)
async def test_unbucketed_series_run_on_the_table_interval_with_ten_percent_jitter(
    sid: str, interval: int, seconds: int
) -> None:
    clock = FakeClock()
    client = FakeClient(clock, [fgi_ok() if sid == FGI else fred_ok()])
    poller = make(clock, rnd(7), **{"fgi" if sid == FGI else "fred": client})
    await run(poller, clock, seconds)
    times = [t for t, _ in client.calls]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert 0 <= times[0] <= 30 and len(gaps) >= 3  # the boot round is staggered 0-30 s
    assert all(0.9 * interval - 1e-6 <= g <= 1.1 * interval + 1e-6 for g in gaps)
    assert len({round(g, 3) for g in gaps}) > 1  # and it is actually jittered


@pytest.mark.parametrize(
    ("key", "ok", "sid"), [("coingecko", dom_ok, DOM), ("whale", whale_ok, WHALE)]
)
async def test_bucketed_series_poll_once_per_bucket_at_bucket_start_plus_jitter(
    key: str, ok: Callable[[], Ok[Any]], sid: str
) -> None:
    clock = FakeClock()
    client = FakeClient(clock, [ok()])
    poller = make(clock, rnd(3), **{key: client})
    await run(poller, clock, 6 * 3_600)
    walls = [START + int(t * 1000) for t, _ in client.calls]
    assert 0 <= client.calls[0][0] <= 30  # boot round
    later = walls[1:]
    assert len(later) == 6 or len(later) == 5  # one per later bucket, none twice
    assert all(w % H <= 300_000 for w in later)
    assert len({w // H for w in walls}) == len(walls)  # no second attempt after a validated one


@pytest.mark.parametrize("failure", [Err("http_5xx"), Err("http_429")])
async def test_the_g_c_soak_48h_always_failing_keeps_budgets_gaps_and_memory(
    failure: Err, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G-C (48 h fake clock, sockets blocked) + CF-2401-52/53 + RK-P3-04, jitter at +10 %."""
    clock = FakeClock()
    clients = {k: FakeClient(clock, [failure]) for k in ("fgi", "coingecko", "fred", "whale")}
    sids = {"fgi": FGI, "coingecko": DOM, "fred": FRED, "whale": WHALE}
    starts = {s: starts_of(s, monkeypatch) for s in sids.values()}
    poller = make(clock, ConstRng(1.0), **clients)  # +10 % jitter: the worst case for cap/4
    tracemalloc.start()
    try:
        await run(poller, clock, 48 * 3_600)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    for key, sid in sids.items():
        cadence = pol.CADENCE[sid]
        reqs = [t for t, _ in clients[key].calls for _ in range(cadence.requests_per_cycle)]
        for i, t in enumerate(reqs):  # rolling hour, retries included
            assert sum(1 for u in reqs if t - 3_600 < u <= t) <= cadence.budget_per_hour, (sid, i)
        gaps = [b - a for a, b in zip(starts[sid], starts[sid][1:], strict=False)]
        assert gaps and max(gaps) <= pol.max_gap_s(sid) + 1e-6, sid  # cap/4 between cycle starts
        assert min(gaps) >= pol.FLOOR_S - 1e-6
        assert counter("mtf_context_poll_total", series=sid, outcome=failure.outcome) > 0
    whale_calls = [t for t, _ in clients["whale"].calls]
    assert max(b - a for a, b in itertools.pairwise(whale_calls)) <= 50 * 60
    assert counter("mtf_context_poll_total", series=WHALE, outcome="budget") > 0  # RK-P3-04
    assert len(clock.sleeps) < 12_000  # no busy loop (2880 ticks + a few hundred cycles each)
    assert peak < 8 * 1024 * 1024 and not poller._bg
    assert all(len(p.stamps_ms) <= p.cadence.budget_per_hour for p in poller._providers)
    assert LIVE_BOOK.snapshot() == {}  # a dead provider never wrote a row


async def test_the_g_c_soak_48h_with_healthy_providers_stays_within_the_nominal_calls() -> None:
    """SY-01-08 / SY-01-22d nominal calls per day (+10 % jitter, +1 boot round), 48 h."""
    clock = FakeClock()
    clients = {
        k: FakeClient(clock, [ok])
        for k, ok in (
            ("fgi", fgi_ok()),
            ("coingecko", dom_ok()),
            ("fred", fred_ok()),
            ("whale", whale_ok()),
        )
    }
    await run(make(clock, rnd(11), **clients), clock, 48 * 3_600)
    calls = {k: len(c.calls) for k, c in clients.items()}
    limits = {"fgi": 55, "coingecko": 49, "fred": 11, "whale": 49}  # 24/day, 24/day, 4/day, 24/day
    assert all(calls[k] <= limits[k] for k in calls) and all(calls.values()), calls
    assert counter("mtf_context_poll_total", series=DOM, outcome="http_5xx") == 0


async def test_one_attempt_at_a_time_per_provider_and_globally() -> None:
    clock, probe = FakeClock(), Probe()
    clients = {
        k: FakeClient(clock, [ok], latency=4.0, probe=probe)
        for k, ok in (
            ("fgi", fgi_ok()),
            ("coingecko", dom_ok()),
            ("fred", fred_ok()),
            ("whale", whale_ok()),
        )
    }
    await run(make(clock, **clients), clock, 600)
    assert probe.max_active == 1 and all(c.max_active == 1 for c in clients.values())
    assert all(c.calls for c in clients.values())


async def test_a_second_start_is_a_noop_and_stop_is_idempotent() -> None:
    clock = FakeClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))
    first = poller.start()
    names = {t.get_name() for t in asyncio.all_tasks()}
    assert poller.start() is first and {t.get_name() for t in asyncio.all_tasks()} == names
    await poller.stop()
    await poller.stop()
    assert not poller.running


async def test_the_supervisor_ticks_every_heartbeat_tick_and_every_sleep_respects_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beats: list[int] = []
    original = LiveContextBook.heartbeat

    def spy(self: LiveContextBook, now_ms: int) -> bool:
        beats.append(now_ms)
        return original(self, now_ms)

    monkeypatch.setattr(LiveContextBook, "heartbeat", spy)
    clock = FakeClock()
    poller = make(clock)  # nothing is due: no provider at all
    await run(poller, clock, 600)
    assert len(beats) == 11  # t = 0, 60, ..., 600
    assert {s for n, s in clock.sleeps if n == "context_poller"} == {float(pol.HEARTBEAT_TICK_S)}
    assert pol.HEARTBEAT_TICK_S >= pol.FLOOR_S
    clock2, starts = FakeClock(), starts_of(FGI, monkeypatch)
    p2 = make(clock2, rnd(5), fgi=FakeClient(clock2, [Err("http_5xx")]))
    await run(p2, clock2, 6 * 3_600)  # in-cycle backoffs are 1-3 s by design; cycles are >= floor
    assert len(starts) >= 3 and min(b - a for a, b in itertools.pairwise(starts)) >= pol.FLOOR_S


async def test_a_dead_provider_task_is_restarted_at_most_three_times_an_hour() -> None:
    clock = FakeClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))
    poller.start()
    task_of = poller._providers[0]
    for _ in range(5):
        assert task_of.task is not None
        task_of.task.cancel()
        await clock.advance(61)  # one supervisor tick notices the dead task
    await poller.stop()
    assert counter("mtf_context_task_restarts_total", series=FGI) == 3


async def test_a_contended_bucket_start_counts_the_semaphore_wait_against_the_cycle() -> None:
    """N-6/A-4: both snapshot series are due in one bucket while the global semaphore is held."""
    clock = FakeClock()
    cg, wh = FakeClient(clock, [dom_ok()]), FakeClient(clock, [whale_ok()])
    poller = make(clock, coingecko=cg, whale=wh)
    await poller._sem.acquire()  # held by someone else for the whole 30 s cycle
    poller.start()
    await clock.advance(31)
    poller._sem.release()
    await poller.stop()
    assert cg.calls == [] and wh.calls == []  # no request was sent
    for sid in (DOM, WHALE):
        assert counter("mtf_context_poll_total", series=sid, outcome="timeout") == 1
    assert all(len(p.stamps_ms) == 0 for p in poller._providers)  # A-4: no budget spent
    assert all(p.failed_cycles == 1 for p in poller._providers)  # a failed attempt for G-E


# ---------------------------------------------------------------------------
# T-27: retries, Retry-After, backoff, auth_failed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("outcome", ["timeout", "connect_error", "http_5xx", "http_429"])
async def test_retry_classes_get_two_retries(outcome: str) -> None:
    clock = AutoClock()
    client = FakeClient(clock, [Err(outcome)])  # type: ignore[arg-type]
    poller = make(clock, fgi=client)
    await poller._cycle(poller._providers[0])
    assert (
        len(client.calls) == 3
        and counter("mtf_context_poll_total", series=FGI, outcome=outcome) == 1
    )


@pytest.mark.parametrize(
    "outcome", ["http_3xx", "http_4xx", "oversize", "auth_failed", "invalid_payload"]
)
async def test_other_failures_are_not_retried(outcome: str) -> None:
    clock = AutoClock()
    client = FakeClient(clock, [Err(outcome)])  # type: ignore[arg-type]
    poller = make(clock, fgi=client)
    await poller._cycle(poller._providers[0])
    assert len(client.calls) == 1


async def test_every_attempt_of_a_cycle_reuses_the_cycle_fetch_ms() -> None:
    """CF-2401-50: retries never re-derive the clock; the row is stamped with the first value."""
    clock = AutoClock()
    client = FakeClient(clock, [Err("http_5xx"), Err("http_5xx"), dom_ok()])
    poller = make(clock, coingecko=client)
    await poller._cycle(poller._providers[0])
    stamps = [kw["now_ms"] for _, kw in client.calls]
    assert len(stamps) == 3 and len(set(stamps)) == 1
    assert clock.wall_ms() > stamps[0]  # the wall clock did move during the backoffs
    assert LIVE_BOOK.snapshot()[DOM].observations[0].available_ms == stamps[0]


@pytest.mark.parametrize(
    ("header", "in_cycle_retry", "next_gap"),
    [("5", True, None), ("30", False, 60.0), ("120", False, 120.0), ("999999", False, 3_600.0)],
)
async def test_retry_after_seconds(
    header: str, in_cycle_retry: bool, next_gap: float | None
) -> None:
    clock = AutoClock()
    client = FakeClient(clock, [Err("http_429", retry_after=header), fgi_ok()])
    poller = make(clock, fgi=client)
    delay = await poller._cycle(poller._providers[0])
    if in_cycle_retry:
        assert len(client.calls) == 2 and client.calls[1][0] - client.calls[0][0] >= 5
    else:
        assert len(client.calls) == 1 and delay == next_gap  # never earlier than max(floor, RA)


async def test_an_http_date_retry_after_is_parsed_and_garbage_falls_back_to_backoff() -> None:
    clock = AutoClock()
    when = dt(START + 200_000).strftime("%a, %d %b %Y %H:%M:%S GMT")
    client = FakeClient(clock, [Err("http_429", retry_after=when)])
    poller = make(clock, fgi=client)
    assert await poller._cycle(poller._providers[0]) == 200.0
    plain = FakeClient(clock, [Err("http_429")])
    p2 = make(clock, fgi=plain)
    await p2._cycle(p2._providers[0])
    assert len(plain.calls) == 3  # no usable header: the normal backoff schedule
    now = START
    assert parse_retry_after("12", now) == 12 and parse_retry_after(None, now) is None
    assert parse_retry_after("Mon, 11 Mar 2024 12:00:00 GMT", now) == 0
    assert parse_retry_after("Mon, 11 Mar 2024 12:01:00 GMT", now) == 60
    for bad in (
        "Tue, 11 Mar 2024 12:00:00 GMT",
        "Mon, 30 Feb 2024 12:00:00 GMT",
        "Mon, 11 Xyz 2024 12:00:00 GMT",
        "soon",
        "1e3",
    ):
        assert parse_retry_after(bad, now) is None, bad


async def test_the_semaphore_and_single_flight_are_free_during_every_sleep() -> None:
    clock = FakeClock()
    client = FakeClient(clock, [Err("http_5xx")])
    poller = make(clock, rnd(2), fgi=client)
    seen: list[bool] = []
    clock.on_sleep = lambda: seen.append(poller._sem.locked())
    await run(poller, clock, 3 * 3_600)
    assert seen and not any(seen)


async def test_auth_failed_waits_six_hours_and_logs_one_error_per_transition() -> None:
    clock = FakeClock()
    client = FakeClient(clock, [Err("auth_failed")])
    poller = make(clock, fred=client)
    with capture_logs() as logs:
        await run(poller, clock, 13 * 3_600)
    assert len(client.calls) == 3  # boot, +6 h, +12 h; never retried in a cycle
    assert [r["log_level"] for r in logs if r["event"] == "context_poller.poll_failed"] == ["error"]
    assert (
        poller.health(LIVE_BOOK.freshness(clock.wall_ms()), clock.wall_ms()).series[2].status
        == "auth_failed"
    )


async def test_an_exhausted_budget_is_a_budget_outcome_without_a_request() -> None:
    clock = AutoClock()
    client = FakeClient(clock, [Err("http_5xx")])
    poller = make(clock, whale=client)
    p = poller._providers[0]
    for _ in range(4):
        await poller._cycle(p)
    assert (
        len(client.calls) == 8
        and counter("mtf_context_poll_total", series=WHALE, outcome="budget") == 2
    )


# ---------------------------------------------------------------------------
# CF-2401-35 / -34: classification, labels
# ---------------------------------------------------------------------------
async def test_dns_failures_are_egress_blocked_and_parser_errors_invalid_not_retried() -> None:
    clock = AutoClock()
    dns = FakeClient(clock, [Err("connect_error", error_type="ClientConnectorDNSError")])
    p1 = make(clock, fgi=dns)
    await p1._cycle(p1._providers[0])
    assert len(dns.calls) == 1 and p1._providers[0].last_outcome == "egress_blocked"
    parser = FakeClient(clock, [Err("connect_error", error_type="ClientResponseError")])
    p2 = make(clock, fgi=parser)
    await p2._cycle(p2._providers[0])
    assert len(parser.calls) == 1 and p2._providers[0].last_outcome == "invalid_payload"
    plain = FakeClient(clock, [Err("connect_error", error_type="ClientConnectorError")])
    p3 = make(clock, fgi=plain)
    await p3._cycle(p3._providers[0])
    assert len(plain.calls) == 3  # an ordinary connect error stays retryable


async def test_rejection_labels_are_distinct_closed_codes() -> None:
    clock = AutoClock()
    for reason in ("bad_content_type", "bad_encoding", "bad_json"):
        poller = make(clock, fgi=FakeClient(clock, [Err("invalid_payload", reason)]))  # type: ignore[arg-type]
        await poller._cycle(poller._providers[0])
    redirect = make(clock, fgi=FakeClient(clock, [Err("http_3xx")]))
    await redirect._cycle(redirect._providers[0])
    for reason in ("bad_content_type", "bad_encoding", "bad_json"):
        assert counter("mtf_context_value_rejected_total", series=FGI, reason=reason) == 1
    assert counter("mtf_context_poll_total", series=FGI, outcome="http_3xx") == 1
    assert counter("mtf_context_poll_total", series=FGI, outcome="invalid_payload") == 3
    health = redirect.health(LIVE_BOOK.freshness(clock.wall_ms()), clock.wall_ms())
    assert health.series[0].last_outcome == "http_3xx"  # distinct from bad_encoding in health too


async def test_only_parsed_retry_after_seconds_are_logged() -> None:
    clock = AutoClock()
    leaky = "Mon, 11 Mar 2024 12:03:20 GMT"
    poller = make(clock, fgi=FakeClient(clock, [Err("http_429", retry_after=leaky)]))
    with capture_logs() as logs:
        await poller._cycle(poller._providers[0])
    row = next(r for r in logs if r["event"] == "context_poller.poll_failed")
    assert row["retry_after_s"] == 200 and leaky not in repr(logs)


# ---------------------------------------------------------------------------
# T-06 (P5 clause), ER3-03, CF-2401-45/50/59: the order of one validated Ok
# ---------------------------------------------------------------------------
@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    log: list[tuple[str, Any]] = []
    for name in ("heartbeat", "publish", "freshness"):
        original = getattr(LiveContextBook, name)

        def spy(self: LiveContextBook, *a: Any, _n: str = name, _o: Any = original) -> Any:
            out = _o(self, *a)
            log.append((_n, (a, out)))
            return out

        monkeypatch.setattr(LiveContextBook, name, spy)
    return log


async def test_one_fetch_ms_and_the_order_convert_heartbeat_publish_adopt_freshness_offer(
    events: list[tuple[str, Any]],
) -> None:
    clock, sink = AutoClock(), SpySink()
    client = FakeClient(clock, [dom_ok()], latency=3.0)  # the wall clock moves during the fetch
    stale_stamp = CoinGeckoSnapshot(
        btc_dominance=52.5,
        market_cap_change_24h=1,
        total_volume_change_24h=1,
        timestamp=dt(START + 5 * H),
    )  # a snapshot time that is NOT fetch_ms
    client.script = [Ok(stale_stamp)]
    poller = make(clock, coingecko=client, sink=sink)
    original_offer = sink.offer
    sink.offer = lambda s, o: (events.append(("offer", s)), original_offer(s, o))[1]  # type: ignore[method-assign]
    client.adopt_validated = lambda snap, *, now_mono: events.append(("adopt", now_mono))  # type: ignore[method-assign]
    await poller._cycle(poller._providers[0])
    names = [n for n, _ in events]
    assert names == ["heartbeat", "publish", "adopt", "freshness", "offer"]
    fetch_ms = client.calls[0][1]["now_ms"]
    assert clock.wall_ms() == fetch_ms + 3_000  # so wall_ms != fetch_ms at the heartbeat/gate
    assert fetch_ms == START and events[0][1][0] == (fetch_ms,) and events[1][1][0][2] == fetch_ms
    obs = sink.offered[0][1]
    assert obs.available_ms == fetch_ms and obs.observed_ms == START - START % H  # CF-2401-50
    assert events[3][1][0] == (fetch_ms,)  # the gate is freshness(fetch_ms), after the call


async def test_a_converter_or_publish_exception_leaves_caches_unchanged_and_offers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, sink = AutoClock(), SpySink()
    nan = CoinGeckoSnapshot.model_construct(
        btc_dominance=float("nan"),
        market_cap_change_24h=0.0,
        total_volume_change_24h=0.0,
        timestamp=dt(START),
    )
    client = FakeClient(clock, [Ok(nan)])
    poller = make(clock, coingecko=client, sink=sink)
    with capture_logs() as logs:
        await poller._cycle(poller._providers[0])
    assert client.adopted == [] and sink.offered == [] and LIVE_BOOK.snapshot() == {}
    assert counter("mtf_context_poll_total", series=DOM, outcome="invalid_payload") == 1
    assert counter("mtf_context_value_rejected_total", series=DOM, reason="internal_error") == 1
    row = next(r for r in logs if r["event"] == "context_poller.apply_failed")
    assert row["error_type"] == "PitError" and "NaN" not in repr(row)
    ok_client = FakeClient(clock, [dom_ok()])
    p2 = make(clock, coingecko=ok_client, sink=sink)

    def broken(*_a: Any, **_k: Any) -> int:
        raise RuntimeError("publish blew up")

    monkeypatch.setattr(LiveContextBook, "publish", broken)
    await p2._cycle(p2._providers[0])
    assert ok_client.adopted == [] and sink.offered == []


@pytest.mark.parametrize("recorded", [DOM, WHALE])
async def test_recorder_offer_vectors(recorded: str) -> None:
    clock, sink = AutoClock(), SpySink()
    key, ok = ("coingecko", dom_ok) if recorded == DOM else ("whale", whale_ok)
    client = FakeClient(clock, [ok(10.0), ok(10.0), ok(20.0)])
    poller = make(clock, **{key: client, "sink": sink})
    p = poller._providers[0]
    await poller._cycle(p)
    assert len(sink.offered) == 1  # a first sighting is offered once
    held = LIVE_BOOK.snapshot()[recorded]
    await poller._cycle(p)  # an identical re-publish (same fetch_ms and value): publish -> 0
    assert LIVE_BOOK.snapshot()[recorded] is held
    await clock.advance(120)
    await poller._cycle(p)  # same bucket, different value: publish -> 0
    assert len(sink.offered) == 1 and len(LIVE_BOOK.snapshot()[recorded].observations) == 1


async def test_a_sighting_published_while_regressed_is_held_but_never_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ER3-03: the gate is freshness(fetch_ms).regressed after the call, not a cached heartbeat."""
    clock, sink = AutoClock(), SpySink()
    client = FakeClient(clock, [whale_ok(5.0)])
    poller = make(clock, whale=client, sink=sink)
    LIVE_BOOK.heartbeat(START + 2 * H)  # the mark is two hours ahead of the wall clock
    monkeypatch.setattr(LiveContextBook, "heartbeat", lambda self, now_ms: False)  # a stale False
    await poller._cycle(poller._providers[0])
    assert LIVE_BOOK.freshness(START).regressed is True
    assert sink.offered == [] and len(LIVE_BOOK.snapshot()[WHALE].observations) == 1


async def test_fgi_and_t10y2y_are_never_offered_and_both_publish() -> None:
    clock, sink = AutoClock(), SpySink()
    fgi, fred = FakeClient(clock, [fgi_ok()]), FakeClient(clock, [fred_ok()])
    poller = make(clock, fgi=fgi, fred=fred, sink=sink)
    for p in poller._providers:
        await poller._cycle(p)
    assert sink.offered == [] and {FGI, FRED} <= set(LIVE_BOOK.snapshot())
    assert fgi.adopted and fred.adopted


async def test_the_first_sighting_of_each_fgi_day_is_logged_once_with_closed_fields() -> None:
    """SY-01-06: ``mtf_context.fgi_first_seen`` is the CF-24-16 evidence; INFO, numbers only."""
    clock = AutoClock()
    day = START - START % (24 * H)
    client = FakeClient(clock, [fgi_ok(newest_day=day), fgi_ok(newest_day=day),
                                fgi_ok(newest_day=day + 24 * H)])  # fmt: skip
    poller = make(clock, fgi=client)
    with capture_logs() as logs:
        for _ in range(2):
            await poller._cycle(poller._providers[0])
        clock.offset_ms = 24 * H  # a day later the provider has published the next day
        await poller._cycle(poller._providers[0])
    seen = [r for r in logs if r["event"] == "mtf_context.fgi_first_seen"]
    assert [r["observed_day"] for r in seen] == ["2024-03-11", "2024-03-12"]
    assert all(r["log_level"] == "info" and set(r) <= {"event", "log_level", "component",
               "observed_day", "first_seen"} for r in seen)  # fmt: skip


async def _crr1_scenario(cls: type[ContextPoller]) -> tuple[int, Decimal | None, list[Any]]:
    """CF-2401-59: a forward-stamped row, the clock corrected by 30 min, no tick in between."""
    LIVE_BOOK.reset()
    bucket = START
    clock, sink = AutoClock(bucket - 50 * MIN), SpySink()  # 11:10: previous bucket's sighting
    client = FakeClient(clock, [whale_ok(3.0)])
    poller = cls(clock=clock, rng=ConstRng(0.0), whale=client, sink=sink)
    p = poller._providers[0]
    await poller._cycle(p)  # bucket 11:00 holds 3.0
    clock.offset_ms = 90 * MIN  # the faulted clock reads 12:40
    client.script = [whale_ok(5.0)]
    await poller._cycle(p)  # bucket 12:00 gets the forward-stamped 5.0 (stamp 12:40)
    clock.offset_ms = 20 * MIN - (clock.wall_ms() - clock.offset_ms - bucket)  # corrected: 12:10
    client.script = [whale_ok(7.0)]
    sink.offered.clear()
    await poller._cycle(p)  # the real sighting
    poller._tick()  # the next supervisor tick
    clock.offset_ms += 26 * MIN  # 12:36: the guard lifts (12:36 + 5 min >= mark 12:40)
    now = clock.wall_ms()
    served = LIVE_BOOK.view(now)[WHALE].asof(now, max_age_ms=SERIES_RULES[WHALE].max_age_ms)
    return len(sink.offered), served, list(sink.offered)


_NO_BEAT = re.sub(r" +self\._beat\(fetch_ms\).*\n", "", _SRC)  # mutant 1: heartbeat removed
_PUBLISH = "            applied = LIVE_BOOK.publish(p.series_id, rows, fetch_ms)\n"
_MOVED_BEAT = _NO_BEAT.replace(
    _PUBLISH, _PUBLISH + "            self._beat(fetch_ms)\n"
)  # mutant 2


async def test_the_pre_publish_heartbeat_closes_r3_res_1_and_both_mutants_reopen_it() -> None:
    offered, served, _ = await _crr1_scenario(ContextPoller)
    assert (offered, served) == (0, Decimal("7.00000000"))  # applied, not offered, served later
    no_beat, moved = _NO_BEAT, _MOVED_BEAT
    assert no_beat != _SRC and moved != no_beat
    for source in (no_beat, moved):
        module = types.ModuleType("cp_mutant")
        sys.modules["cp_mutant"] = module  # dataclasses resolve string annotations through it
        try:
            exec(compile(source, "cp_mutant", "exec"), module.__dict__)  # noqa: S102 -- own source
            _, mutant_served, _ = await _crr1_scenario(module.ContextPoller)
        finally:
            del sys.modules["cp_mutant"]
        assert mutant_served == Decimal("3.00000000")  # the bucket ended empty; the old one serves


def apply_order_problems(source: str) -> list[str]:
    """In ``_apply``: convert, then ``_beat(fetch_ms)``, then publish, with no await in between."""
    fn = next(n for n in ast.walk(ast.parse(source)) if getattr(n, "name", "") == "_apply")
    calls = [(n.lineno, ast.unparse(n)) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    order = {
        k: next((ln for ln, s in calls if s.startswith(k)), -1)
        for k in ("p.convert(", "self._beat(fetch_ms)", "LIVE_BOOK.publish(")
    }
    found = [] if -1 not in order.values() else ["missing"]
    if (
        not found
        and not order["p.convert("] < order["self._beat(fetch_ms)"] < order["LIVE_BOOK.publish("]
    ):
        found.append("order")
    if any(isinstance(n, ast.Await) for n in ast.walk(fn)) or isinstance(fn, ast.AsyncFunctionDef):
        found.append("await")
    return found


def test_apply_is_synchronous_and_ordered_and_the_pin_catches_the_mutants() -> None:
    assert apply_order_problems(_SRC) == []
    assert apply_order_problems(_SRC.replace("self._beat(fetch_ms)", "pass")) != []
    assert apply_order_problems(_SRC.replace("    def _apply(", "    async def _apply(")) != []
    assert apply_order_problems(_MOVED_BEAT) != []


# ---------------------------------------------------------------------------
# T-25 (poller part): the clock guard
# ---------------------------------------------------------------------------
async def test_a_regressed_heartbeat_counts_once_per_call_and_warns_once_per_transition() -> None:
    clock = AutoClock()
    poller = make(clock)
    LIVE_BOOK.heartbeat(START + H)  # the mark is an hour ahead
    with capture_logs() as logs:
        for _ in range(3):
            assert poller._beat(clock.wall_ms()) is True
        clock.offset_ms = 2 * H
        assert poller._beat(clock.wall_ms()) is False
        assert poller._beat(clock.wall_ms()) is False
    assert counter("mtf_context_clock_regressed_total") == 3
    events = [r["event"] for r in logs]
    assert events.count("mtf_context.clock_regressed") == 1
    assert events.count("mtf_context.clock_recovered") == 1


async def test_the_regressed_reads_gauge_equals_the_book_and_counts_every_engine_read() -> None:
    clock = FakeClock()
    poller = make(clock)
    LIVE_BOOK.heartbeat(START + H)
    for _ in range(4):
        LIVE_BOOK.view(START)  # an engine read, consumer or not (CF-2401-69)
    poller._tick()
    assert LIVE_BOOK.regressed_reads == 4 and gauge("mtf_context_regressed_reads") == 4.0
    description = health_router.ContextPollerHealth.model_fields["regressed"].description or ""
    assert "every engine read (consumer and non-consumer)" in description


# ---------------------------------------------------------------------------
# T-31 / CF-2401-62/64: status, metrics, health
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"configured": False}, "not_configured"),
        ({"fresh": True}, "fresh"),
        ({"fresh": True, "regressed": True}, "stale"),  # A-2
        ({"rows": 3}, "stale"),
        ({}, "unavailable"),
        ({"last_outcome": "invalid_payload"}, "invalid"),
        ({"last_outcome": "egress_blocked", "rows": 2}, "egress_blocked"),
        ({"last_outcome": "auth_failed", "fresh": True}, "auth_failed"),
        (
            {"last_outcome": "auth_failed", "regressed": True},
            "auth_failed",
        ),  # A-2: others unchanged
        ({"last_outcome": "egress_blocked", "regressed": True}, "egress_blocked"),
        ({"last_outcome": "invalid_payload", "regressed": True}, "invalid"),
        ({"configured": False, "regressed": True}, "not_configured"),
    ],
)
def test_status_mapping_and_the_a2_rule(kwargs: dict[str, Any], expected: str) -> None:
    base = {"configured": True, "rows": 0, "fresh": False, "regressed": False, "last_outcome": None}
    assert derive_status(**{**base, **kwargs}) == expected
    assert expected in cp.STATUSES


async def test_health_block_uses_freshness_only_and_reports_a_regressed_fresh_series_as_stale() -> (
    None
):
    clock = AutoClock()
    poller = make(clock, coingecko=FakeClient(clock, [dom_ok()]))
    container = AppContainer.for_testing()
    container.background_tasks.context_poller = poller
    app = FastAPI()
    app.state.container = container
    app.include_router(health_router.router, prefix="/api/v1")
    web = TestClient(app)
    await poller._cycle(poller._providers[0])
    body = web.get("/api/v1/health/background").json()["context_poller"]
    rows = {r["series"]: r for r in body["series"]}
    assert rows[DOM]["status"] == "fresh" and rows[FGI]["status"] == "not_configured"
    assert set(rows) == set(pol.CADENCE) and body["regressed"] is False
    LIVE_BOOK.heartbeat(START + 2 * H)  # regressed from now on
    before = (LIVE_BOOK._high_water_ms, LIVE_BOOK.regressed_reads)
    body = web.get("/api/v1/health/background").json()["context_poller"]
    assert (
        body["regressed"] is True
        and {r["series"]: r["status"] for r in body["series"]}[DOM] == "stale"
    )
    assert (
        LIVE_BOOK._high_water_ms,
        LIVE_BOOK.regressed_reads,
    ) == before  # a probe mutates nothing
    poller._tick()  # RK-P5-03: the gauge follows A-2 as well
    assert gauge("mtf_context_fresh", series=DOM) == 0.0
    assert (
        web.get("/api/v1/health/background").json()["history_cache_warmer"]["configured"] is False
    )


async def test_a_dead_or_stalled_poller_is_visible_on_health_and_as_a_gauge() -> None:
    """CF-2401-64: no heartbeat in 3 ticks degrades health; a done supervisor zeroes the gauge."""
    clock = FakeClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))
    poller.start()
    await clock.advance(30)
    book = LIVE_BOOK.freshness(clock.wall_ms())
    assert (
        poller.health(book, clock.wall_ms()).alive is True
        and gauge("mtf_context_poller_alive") == 1.0
    )
    clock.t += pol.POLLER_DEAD_AFTER_S + 1  # the loop is stalled: running, but no tick
    stalled = poller.health(book, clock.wall_ms())
    assert stalled.running is True and stalled.alive is False
    assert gauge("mtf_context_poller_heartbeat_timestamp_seconds") == pytest.approx(
        START / 1000 + 30, abs=61
    )
    task = poller._task
    assert task is not None
    task.cancel()
    await clock.advance(0)
    await poller.stop()
    assert poller.health(book, clock.wall_ms()).running is False


async def test_metrics_labels_are_code_constants_and_the_gauges_are_exported() -> None:
    clock = FakeClock()
    poller = make(clock, coingecko=FakeClient(clock, [dom_ok()]))
    token = LIVE_CONSUMERS.started("paper")
    await run(poller, clock, 300)
    snap = metrics.get_all()
    keys = [
        k for kind in ("counters", "gauges") for k in snap[kind] if k.startswith("mtf_context_")
    ]
    allowed = {
        "series": set(pol.CADENCE),
        "outcome": set(cp.OUTCOMES),
        "mode": {"paper", "live"},
        "reason": set(cp.REASONS) | {"unknown"},
    }
    for key in keys:
        if "{" in key:
            for pair in key.split("{")[1].rstrip("}").split(","):
                name, value = pair.split("=")
                assert value.strip('"') in allowed[name], key
    assert gauge("mtf_context_series_configured", series=DOM) == 1.0
    assert gauge("mtf_context_series_configured", series=WHALE) == 0.0
    assert gauge("mtf_context_consumer_runs_active", mode="paper") == 1.0
    assert gauge("mtf_context_consumer_runs_active", mode="live") == 0.0
    assert (
        gauge("mtf_context_fresh", series=DOM) == 1.0
        and gauge("mtf_context_age_seconds", series=DOM) is not None
    )
    LIVE_CONSUMERS.stopped(token)


async def test_one_warning_per_status_change_and_a_reminder_at_most_every_six_hours() -> None:
    clock = FakeClock()
    listener_calls: list[tuple[str, str, str]] = []

    class Listener(cp.NoopStatusListener):
        async def on_status_change(self, series: str, old: str, new: str) -> None:
            listener_calls.append((series, old, new))

    poller = make(clock, coingecko=FakeClient(clock, [Err("http_5xx")]), listener=Listener())
    poller.start()
    with capture_logs() as logs:
        await clock.advance(30)
        poller._providers[0].last_outcome = "ok"
        LIVE_BOOK.publish(
            DOM, (cp.snapshot_observation(DOM, 50.0, clock.wall_ms()),), clock.wall_ms()
        )
        await clock.advance(60)  # unavailable -> fresh
        await clock.advance(7 * 3_600)  # fresh -> stale after the 6 h cap
        await clock.advance(13 * 3_600)  # a standing bad status: reminders
    await poller.stop()
    changed = [
        (r["series"], r["old"], r["new"])
        for r in logs
        if r["event"] == "mtf_context.status_changed"
    ]
    assert (DOM, "unavailable", "fresh") in changed and (DOM, "fresh", "stale") in changed
    assert set(listener_calls) == set(changed)
    reminders = [
        r for r in logs if r["event"] == "mtf_context.status_reminder" and r["series"] == DOM
    ]
    assert 2 <= len(reminders) <= 4  # 13 h of staleness: spaced >= 6 h apart


async def test_a_listener_failure_never_reaches_the_schedule_and_logs_error_type_only() -> None:
    clock = FakeClock()

    class Boom(cp.NoopStatusListener):
        async def on_poller_dead(self) -> None:
            raise RuntimeError(f"{SENTINEL}-secret")

    poller = make(clock, listener=Boom())
    with capture_logs() as logs:
        poller._notify(lambda: poller._listener.on_poller_dead())
        await clock._settle()
    row = next(r for r in logs if r["event"] == "context_poller.listener_failed")
    assert row["error_type"] == "RuntimeError" and SENTINEL not in repr(logs)


async def test_a_supervisor_death_calls_the_listener_once_and_a_shutdown_cancel_does_not() -> None:
    clock, dead = FakeClock(), []

    class Listener(cp.NoopStatusListener):
        async def on_poller_dead(self) -> None:
            dead.append(1)

    poller = make(clock, listener=Listener())
    task = poller.start()
    task.add_done_callback(poller.on_supervisor_done)
    task.cancel()  # any cancellation that is not the lifespan shutdown is a death (A-3)
    await clock._settle()
    assert dead == [1] and gauge("mtf_context_poller_alive") == 0.0
    clock2, dead2 = FakeClock(), []

    class Listener2(cp.NoopStatusListener):
        async def on_poller_dead(self) -> None:
            dead2.append(1)

    quiet = make(clock2, listener=Listener2())
    t2 = quiet.start()
    t2.add_done_callback(quiet.on_supervisor_done)
    await quiet.stop()
    await clock2._settle()
    assert dead2 == []


async def test_flush_sink_is_bounded_and_counts_a_timeout() -> None:
    clock = FakeClock()

    class Slow(SpySink):
        async def flush(self, timeout_s: float) -> None:
            await clock.latency(60)

    poller = make(clock, sink=Slow())
    flushing = asyncio.ensure_future(poller.flush_sink(5.0))
    await clock.advance(6)
    await asyncio.wait_for(flushing, 2.0)  # fails, never hangs
    assert counter("mtf_context_record_flush_timeout_total") == 1


# ---------------------------------------------------------------------------
# CF-2401-49/65, T-32: the plural T10Y2Y delivery and the display caches
# ---------------------------------------------------------------------------
async def test_fred_cold_start_serves_at_once_and_a_gap_longer_than_a_day_skips_no_date() -> None:
    clock = AutoClock()
    rows = [(date(2024, 3, 4 + i), f"0.{40 + i}") for i in range(5)]
    poller = make(clock, fred=FakeClient(clock, [fred_ok(rows)]))
    await poller._cycle(poller._providers[0])
    now = START  # Monday 2024-03-11 12:00
    assert LIVE_BOOK.view(now)[FRED].asof(now, max_age_ms=SERIES_RULES[FRED].max_age_ms) == Decimal(
        "0.42"
    )
    assert len(LIVE_BOOK.snapshot()[FRED].observations) == 5  # every date, not just the newest
    clock.offset_ms = 7 * 24 * H  # an outage of five business days: next poll Mon 03-18 12:00
    later = [(date(2024, 3, 11 + i), f"0.{50 + i}") for i in range(5)]  # none seen before
    poller2 = make(clock, fred=FakeClient(clock, [fred_ok(later)]))
    await poller2._cycle(poller2._providers[0])
    pts = [
        (to_ms(datetime(d.year, d.month, d.day, tzinfo=UTC)), Decimal(v)) for d, v in rows + later
    ]
    assert {o.observed_ms for o in LIVE_BOOK.snapshot()[FRED].observations} == {t for t, _ in pts}
    oracle, cap = PitSeries(FRED, t10y2y_observations(pts, clock.wall_ms())), 7 * 24 * H
    for t in range(clock.wall_ms(), clock.wall_ms() + 4 * 24 * H, H):  # live == the full series
        assert LIVE_BOOK.view(t)[FRED].asof(t, max_age_ms=cap) == oracle.asof(t, max_age_ms=cap)


async def test_a_page_with_one_unusable_row_is_filtered_upstream_and_never_half_published() -> None:
    clock = AutoClock()
    bad = MacroSnapshot.model_construct(
        yield_curve_spread=0.4,
        timestamp=dt(START),
        yield_curve_rows=((date(2024, 3, 4), Decimal("0.4")), (date(2024, 3, 5), None)),
    )
    poller = make(clock, fred=FakeClient(clock, [Ok(bad)]))
    await poller._cycle(poller._providers[0])
    assert LIVE_BOOK.snapshot() == {} and poller._providers[0].last_outcome == "invalid_payload"
    empty = make(clock, fred=FakeClient(clock, [Ok(MacroSnapshot(timestamp=dt(START)))]))
    await empty._cycle(empty._providers[0])
    assert LIVE_BOOK.snapshot() == {}  # an empty page is invalid, never a silent success


async def test_a_fred_page_with_one_future_dated_row_is_rejected_whole_against_fetch_ms() -> None:
    """ER3-06/RK-P3-06 at the P5 boundary; CF-2401-50: never re-derived from snapshot.timestamp."""
    clock = AutoClock()  # fetch_ms = START = Mon 2024-03-11 12:00 UTC
    rows = (
        (date(2024, 3, 7), Decimal("0.41")),
        (date(2024, 3, 8), Decimal("0.42")),
        (date(2024, 3, 12), Decimal("0.43")),  # 03-12 00:00 > fetch_ms + 5 min
    )
    snap = MacroSnapshot(
        yield_curve_spread=0.43,
        timestamp=dt(START + 48 * H),  # a snapshot time that is NOT fetch_ms
        yield_curve_observed_on=date(2024, 3, 12),
        yield_curve_rows=rows,
    )
    client = FakeClient(clock, [Ok(snap)])
    poller = make(clock, fred=client)
    await poller._cycle(poller._providers[0])
    assert LIVE_BOOK.snapshot() == {} and client.adopted == []
    assert poller._providers[0].last_outcome == "invalid_payload"


async def test_after_a_validated_poll_signals_current_shows_the_validated_value() -> None:
    clock = AutoClock()
    fgi = FearGreedClient()
    list30 = fgi_ok().value
    fake = FakeClient(clock, [Ok(list30)])
    fake.adopt_validated = lambda snap, *, now_mono: fgi.adopt_validated(snap, now_mono=now_mono)  # type: ignore[method-assign]
    poller = make(clock, fgi=fake)
    container = AppContainer.for_testing(fgi_client=fgi)
    app = FastAPI()
    app.state.container = container
    app.include_router(signals_router.router, prefix="/api/v1")
    web = TestClient(app)
    assert web.get("/api/v1/signals/current").json()["fearGreedIndex"] is None
    await poller._cycle(poller._providers[0])
    body = web.get("/api/v1/signals/current").json()
    assert body["fearGreedIndex"] == max(list30, key=lambda s: s.timestamp).value
    assert await fgi.get_history(limit=30) == sorted(
        list30, key=lambda s: s.timestamp, reverse=True
    )
    before = fgi._latest_cache
    fake.script = [
        Ok(
            [
                FearGreedSnapshot.model_construct(
                    value=float("nan"), classification="x", timestamp=dt(START)
                )
            ]
        )
    ]
    await poller._cycle(poller._providers[0])
    assert fgi._latest_cache is before  # a rejected publish leaves the display caches as they were
    assert (
        web.get("/api/v1/signals/current").json() == body
        and "NaN" not in web.get("/api/v1/signals/current").text
    )


# ---------------------------------------------------------------------------
# T-30: poller AST pins (imports, log forms, never-served names), each with a self-test
# ---------------------------------------------------------------------------
_ALLOWED_ROOTS = {"data", "common", "structlog"}
_STDLIB = {
    "__future__", "asyncio", "collections", "dataclasses", "datetime", "random", "re", "time",
    "typing",
}  # fmt: skip  # SEC-P5-01: an explicit list, so no os / socket / urllib / subprocess
_LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}
_TEXT_CALLS = {"str", "repr", "format", "vars"}


def _exc_names(tree: ast.AST) -> set[str]:
    """Handler names, ``Exception``-annotated parameters, targets of ``x = ....exception()``."""
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and "Exception" in ast.unparse(n.annotation or ast.Constant("")):
            names.add(n.arg)
        if isinstance(n, ast.Assign) and ".exception()" in ast.unparse(n.value):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return names


def exc_text_problems(tree: ast.AST, events: tuple[str, ...] = ("",)) -> list[str]:
    """T-SEC-06 / CF-36: no exception text or object reaches a log call (SEC-P5-01)."""
    excs, found = _exc_names(tree), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in _TEXT_CALLS:
            found += ["text(exc)" for a in n.args if isinstance(a, ast.Name) and a.id in excs]
        if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) in excs:
            found += [] if n.attr == "__traceback__" else [f"exc.{n.attr}"]
        if not (isinstance(n, ast.Call) and getattr(n.func, "attr", "") in _LOG_METHODS and n.args):
            continue  # task.exception() has no positional event
        if not str(getattr(n.args[0], "value", "")).startswith(events):
            continue
        found += ["log.exception"] if getattr(n.func, "attr", "") == "exception" else []
        found += ["exc-arg" for a in n.args[1:] if isinstance(a, ast.Name) and a.id in excs]
        for k in n.keywords:
            found += ["exc_info"] if k.arg == "exc_info" else []
            if k.arg and isinstance(k.value, ast.Name) and k.value.id in excs:
                found.append(f"exc-object:{k.arg}")
    return found


_BANNED_TOKENS = ("market_cap_change_24h", "total_volume_change_24h", "fed_funds_rate", "fedfunds")
_LIVE_GATE_ENV = {"ENABLE_LIVE_TRADING", "LIVE_TRADING_CONFIRM_TOKEN"}
_BANNED_NAMES = {
    "get_settings",
    "Settings",
    "enable_live_trading",
    "kill_switch",
    "run_orchestrator",
    "safety",
    "mtf_signals",
    "circuit_breaker",
    "environ",
    "getenv",
}


def poller_problems(source: str) -> list[str]:
    tree = ast.parse(source)
    found: list[str] = []
    handlers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for mod in mods:
                root = mod.split(".")[0]
                if node.__class__ is ast.ImportFrom and node.level:
                    found.append(f"relative:{mod}")
                elif root not in _ALLOWED_ROOTS and root not in _STDLIB:
                    found.append(f"import:{mod}")
        if isinstance(node, ast.ExceptHandler) and node.name:
            handlers.add(node.name)
        if isinstance(node, ast.keyword) and node.arg == "exc_info":
            found.append("exc_info")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"str", "repr", "format"}:
            if any(isinstance(a, ast.Name) and a.id in handlers for a in node.args):
                found.append("str(exc)")
        if (
            isinstance(node, ast.FormattedValue)
            and isinstance(node.value, ast.Name)
            and node.value.id in handlers
        ):
            found.append("fstring(exc)")
        names = [getattr(node, "id", None), getattr(node, "attr", None), getattr(node, "arg", None)]
        texts = [*names, node.value if isinstance(node, ast.Constant) else None]
        found += [
            f"never-served:{t}"
            for x in texts
            if isinstance(x, str)
            for t in _BANNED_TOKENS
            if t in x.lower()
        ]
        found += [f"banned:{x}" for x in names if x in _BANNED_NAMES]
        if isinstance(node, ast.Constant) and node.value in _LIVE_GATE_ENV:
            found.append(f"live-gate:{node.value!r}")
    return found + exc_text_problems(tree)


def test_the_poller_imports_only_data_common_stdlib_and_logs_no_exception_text() -> None:
    assert poller_problems(_SRC) == []


@pytest.mark.parametrize(
    "line",
    [
        "from trading.safety import LIVE_CONTEXT_CONSUMER_ALLOWLIST\n",
        "from trading.mtf_signals import LIVE_SIGNALS\n",
        "from api.services import kill_switch\n",
        "from api.services import run_orchestrator\n",
        "from api.config import get_settings\n",
        "import aiohttp\n",
        "def f():\n    try:\n        pass\n    except Exception as exc:\n        return str(exc)\n",
        "def f(g):\n    try:\n        g()\n    except Exception as exc:\n        print(f'{exc}')\n",
        "def f(log):\n    log.warning('x', exc_info=True)\n",
        "X = 'market_cap_change_24h'\n",
        "def f(snap):\n    return snap.total_volume_change_24h\n",
        "def f(snap):\n    return snap.fed_funds_rate\n",
        "Y = 'FEDFUNDS'\n",
        "import os\n",  # SEC-P5-01: the stdlib is an explicit list
        "import urllib.request\n",
        "X = 'ENABLE_LIVE_TRADING'\n",
        "def f(log, task):\n    failure = task.exception()\n    log.error('x', e=repr(failure))\n",
        "def f(log, exc: Exception):\n    log.warning('x', error=exc)\n",
        "def f(log, exc: Exception):\n    log.warning('x', detail=exc.args)\n",
        "def f(log):\n    try:\n        pass\n    except Exception:\n        log.exception('x')\n",
    ],
)
def test_the_poller_pin_catches_mutations(line: str) -> None:
    assert poller_problems(_SRC.replace("\nlogger = ", "\n" + line + "\nlogger = ", 1)) != []


def test_health_py_references_no_consumer_registry_and_only_freshness_of_the_book() -> None:
    text = (Path(__file__).resolve().parents[2] / "apps/api/routers/health.py").read_text()
    assert "LIVE_CONSUMERS" not in text and "regressed_reads" not in text
    assert (
        "LIVE_BOOK.freshness(" in text
        and "LIVE_BOOK.view" not in text
        and "LIVE_BOOK.heartbeat" not in text
    )


def test_no_other_conftest_defines_the_isolation_fixture() -> None:
    """CF-2401-63 (gate grep): only tests/conftest.py defines it."""
    root = Path(__file__).resolve().parents[2]
    hits = [
        p
        for p in root.rglob("conftest.py")
        if ".venv" not in p.parts and "_cf2401_live_context_isolated" in p.read_text()
    ]
    assert [p.relative_to(root).as_posix() for p in hits] == ["tests/conftest.py"]


# ---------------------------------------------------------------------------
# CF-2401-36: real aiohttp on loopback, sentinel key, gzip bomb (T-30 harness)
# ---------------------------------------------------------------------------
KEY = f"{SENTINEL}KEY0123456789"


class _Loopback:
    """A raw asyncio HTTP/1.1 server on 127.0.0.1 answering every request from ``responses``."""

    def __init__(self, responses: list[bytes]) -> None:
        self.responses, self.served = responses, 0
        self.server: asyncio.Server | None = None

    async def __aenter__(self) -> _Loopback:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    @property
    def base(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/x"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        payload = self.responses[min(self.served, len(self.responses) - 1)]
        self.served += 1
        view = memoryview(payload)
        with contextlib.suppress(ConnectionError):
            for i in range(0, len(view), 16_384):  # streamed: the server holds no second copy
                writer.write(view[i : i + 16_384])
                await writer.drain()
        writer.close()


def _http(status: str, headers: str = "", body: bytes = b"") -> bytes:
    head = f"HTTP/1.1 {status}\r\n{headers}Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    return head.encode() + body


class _FastClock(cp.SystemClock):
    async def sleep(self, seconds: float) -> None:  # backoff waits are instant; I/O stays real
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    "response",
    [
        _http(
            "429 Too Many Requests",
            f"Retry-After: 3\r\nX-Echo: {KEY}\r\n",
            b'{"e":"' + KEY.encode() + b'"}',
        ),
        _http("500 Internal Server Error", "", KEY.encode()),
        _http("401 Unauthorized", "", KEY.encode()),
        _http("302 Found", f"Location: http://127.0.0.1:9/?api_key={KEY}\r\n"),
        b"HTTP/9.9 xyz\r\n\r\n" + KEY.encode(),  # a parser-level error
        _http(
            "200 OK",
            "Content-Type: application/json\r\nContent-Encoding: gzip\r\n",
            b"\x1f\x8b" + b"\0" * 4096,
        ),
        _http(
            "200 OK", "Content-Type: application/json\r\n", b'{"result":"' + KEY.encode() + b'"}'
        ),
    ],
    ids=["429", "500", "401", "redirect", "parser", "gzip", "shape"],
)
async def test_a_real_aiohttp_failure_never_leaks_the_key_into_logs_metrics_or_health(
    response: bytes, monkeypatch: pytest.MonkeyPatch, production_logging: ConfigureFn
) -> None:
    import data.whale_tracker as whale_mod
    from data.whale_tracker import WhaleAlertClient

    cap = production_logging(json_output=True, log_level="DEBUG")
    disable_layer1()  # prove the sites on their own, not the redactor
    async with _Loopback([response]) as srv:
        monkeypatch.setattr(whale_mod, "_ENDPOINT", srv.base)
        client = WhaleAlertClient(api_key=KEY)
        poller = ContextPoller(whale=client, clock=_FastClock(), rng=ConstRng(0.0))
        try:
            await poller._cycle(poller._providers[0])
            now = poller.now_ms()
            snapshot = repr(poller.health(LIVE_BOOK.freshness(now), now))
        finally:
            await client.close()
    assert KEY not in cap.text and KEY not in snapshot and KEY not in repr(metrics.get_all())
    assert "127.0.0.1" not in cap.text  # no URL or host text either
    assert "context_poller.poll_failed" in cap.text  # non-vacuous (SEC-P5-02)
    assert poller._providers[0].last_outcome != "ok" and LIVE_BOOK.snapshot() == {}


class _Silent(_Loopback):
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        await asyncio.sleep(1.0)  # never answers before the client gives up
        writer.close()


async def test_a_real_aiohttp_timeout_never_leaks_the_key(  # T-30 / T-SEC-05 timeout vector
    monkeypatch: pytest.MonkeyPatch, production_logging: ConfigureFn
) -> None:
    import aiohttp

    import data.provider_http as http_mod
    import data.whale_tracker as whale_mod
    from data.whale_tracker import WhaleAlertClient

    cap = production_logging(json_output=True, log_level="DEBUG")
    disable_layer1()
    monkeypatch.setattr(http_mod, "_TIMEOUT", aiohttp.ClientTimeout(total=0.2))
    async with _Silent([b""]) as srv:
        monkeypatch.setattr(whale_mod, "_ENDPOINT", srv.base)
        client = WhaleAlertClient(api_key=KEY)
        poller = ContextPoller(whale=client, clock=_FastClock(), rng=ConstRng(0.0))
        try:
            await poller._cycle(poller._providers[0])
        finally:
            await client.close()
    assert poller._providers[0].last_outcome == "timeout" and "poll_failed" in cap.text
    assert KEY not in cap.text and "127.0.0.1" not in cap.text
    assert KEY not in repr(metrics.get_all())


async def test_a_gzip_bomb_is_refused_on_the_header_with_a_bounded_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data.whale_tracker as whale_mod
    from data.whale_tracker import WhaleAlertClient

    bomb = _http(
        "200 OK", "Content-Type: application/json\r\nContent-Encoding: gzip\r\n", b"\0" * (6 << 20)
    )
    async with _Loopback([bomb]) as srv:
        monkeypatch.setattr(whale_mod, "_ENDPOINT", srv.base)
        client = WhaleAlertClient(api_key=KEY)
        poller = ContextPoller(whale=client, clock=_FastClock(), rng=ConstRng(0.0))
        tracemalloc.start()
        try:
            await poller._cycle(poller._providers[0])
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
            await client.close()
    assert peak < 1 << 20 and poller._providers[0].last_reason == "bad_encoding"


# ---------------------------------------------------------------------------
# CF-2401-62 (exact int now_ms) and CF-2401-55 (the pure policy helpers)
# ---------------------------------------------------------------------------
async def test_the_poller_and_health_pass_an_exact_int_to_the_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds: dict[str, list[bool]] = {"heartbeat": [], "freshness": [], "publish": []}
    for name in kinds:
        original = getattr(LiveContextBook, name)

        def spy(self: LiveContextBook, *args: Any, _n: str = name, _o: Any = original) -> Any:
            kinds[_n].append(type(args[-1]) is int)
            return _o(self, *args)

        monkeypatch.setattr(LiveContextBook, name, spy)
    clock = AutoClock()
    poller = make(clock, coingecko=FakeClient(clock, [dom_ok()]))
    poller._tick()
    await poller._cycle(poller._providers[0])
    container = AppContainer.for_testing()
    container.background_tasks.context_poller = poller
    app = FastAPI()
    app.state.container = container
    app.include_router(health_router.router, prefix="/api/v1")
    TestClient(app).get("/api/v1/health/background")
    assert all(v and all(v) for v in kinds.values()), kinds


async def test_a_clock_that_returns_a_float_never_reaches_the_book() -> None:
    class FloatClock(AutoClock):
        def wall_ms(self) -> int:
            return float(super().wall_ms())  # type: ignore[return-value]

    clock = FloatClock()
    poller = make(clock)
    with capture_logs() as logs:
        poller._tick()
    assert LIVE_BOOK._high_water_ms == 0 and LIVE_BOOK.snapshot() == {}
    assert [r["error_type"] for r in logs if r["event"] == "context_poller.tick_failed"] == [
        "PitError"
    ]


@pytest.mark.parametrize("sid", list(pol.CADENCE))
def test_next_interval_only_doubles_and_stays_between_the_floor_and_cap_over_four(
    sid: str,
) -> None:
    """CF-2401-55: the ``FLOOR_S`` clamp in ``next_interval_s`` is dead under the cadence pin."""
    values = [pol.next_interval_s(sid, n) for n in range(80)]
    assert values == sorted(values) and values[0] == pol.CADENCE[sid].interval_s
    assert all(pol.FLOOR_S <= v <= pol.max_gap_s(sid) for v in values)


def test_policy_helpers_are_pure_and_reject_a_bad_draw() -> None:
    assert pol.jittered_s(100.0, 0.0) == pytest.approx(90.0)
    assert pol.jittered_s(100.0, 1.0) == pytest.approx(110.0)
    assert pol.jittered_s(100.0, 0.5) == pytest.approx(100.0)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="uniform draw"):
            pol.jittered_s(1.0, bad)
    for sid in pol.CADENCE:  # jitter at +10 % never lifts a gap past cap/4 (CF-2401-52)
        assert pol.clamped_gap_s(sid, float(pol.max_gap_s(sid)), 1.0) == pol.max_gap_s(sid)
        assert pol.clamped_gap_s(sid, 1.0, 0.0) == pol.FLOOR_S
    assert pol.bucket_start_ms(START + 59 * MIN) == START
    assert pol.first_attempt_ms(START, 0.0) == START
    assert pol.first_attempt_ms(START, 1.0) == START + 300_000
    stamps = [START - H, START - H + 1, START]  # the rolling hour is open at its old end
    assert pol.requests_in_window(stamps, START) == 2
    assert pol.budget_allows(stamps, START, 4, 2) and not pol.budget_allows(stamps, START, 3, 2)


async def test_fred_is_budgeted_at_two_attempt_bounds_and_the_others_at_one() -> None:
    """CF-2401-32: two sequential FRED requests get 2 x ATTEMPT_BOUND_S; cancellation propagates."""
    clock = FakeClock()
    slow = {
        "fred": FakeClient(clock, [fred_ok()], latency=15.0),  # > 10 s, < 20 s: still a success
        "coingecko": FakeClient(clock, [dom_ok()], latency=12.0),  # > 10 s: cut at the bound
    }
    poller = make(clock, **slow)
    poller.start()
    await clock.advance(40)
    assert counter("mtf_context_poll_total", series=FRED, outcome="ok") == 1
    assert counter("mtf_context_poll_total", series=DOM, outcome="timeout") >= 1
    assert FRED in LIVE_BOOK.snapshot() and DOM not in LIVE_BOOK.snapshot()
    cancelled = make(clock, fgi=FakeClient(clock, [asyncio.CancelledError()]))
    cancelled.start()
    await clock.advance(5)
    task = cancelled._providers[0].task
    assert task is not None and task.cancelled()  # a cancelled fetch ends the task, never swallowed
    await cancelled.stop()
    await poller.stop()


class _Stop(BaseException):
    """Ends a driven provider loop; not an ``Exception``, so the loop cannot swallow it."""


class _CountingClock(AutoClock):
    async def sleep(self, seconds: float) -> None:
        if len(self.sleeps) >= 4:
            raise _Stop
        await super().sleep(seconds)


async def test_a_dead_supervisor_is_not_alive_at_once_and_zeroes_the_fresh_gauges() -> None:
    clock = FakeClock()
    poller = make(clock, coingecko=FakeClient(clock, [dom_ok(), Err("http_5xx")]))
    task = poller.start()
    task.add_done_callback(poller.on_supervisor_done)
    await clock.advance(120)
    assert gauge("mtf_context_fresh", series=DOM) == 1.0
    task.cancel()
    await clock.advance(0)
    now = clock.wall_ms()
    assert poller.health(LIVE_BOOK.freshness(now), now).alive is False
    assert all(gauge("mtf_context_fresh", series=s) == 0.0 for s in pol.CADENCE)
    await poller.stop()


async def test_a_raising_cycle_still_waits_the_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _CountingClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))

    def boom(*_a: Any, **_k: Any) -> float:
        raise RuntimeError("bookkeeping")

    monkeypatch.setattr(ContextPoller, "_settle", boom)
    with pytest.raises(_Stop):
        await poller._provider_loop(poller._providers[0], 0.0)
    waits = [s for _, s in clock.sleeps[1:]]
    assert len(waits) == 3 and min(waits) >= pol.FLOOR_S
    assert poller._providers[0].failed_cycles == 4


async def test_fred_charges_two_requests_per_attempt_under_a_short_retry_after() -> None:
    clock = FakeClock()
    client = FakeClient(clock, [Err("http_429", retry_after="60")])
    await run(make(clock, fred=client), clock, 2 * 3_600)
    times = [t for t, _ in client.calls]
    assert len(times) >= 3
    for t in times:
        window = sum(1 for u in times if t - 3_600 < u <= t)
        assert 2 * window <= pol.CADENCE[FRED].budget_per_hour, t


async def test_a_tick_that_fails_in_the_heartbeat_is_not_counted_as_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    poller = make(clock)
    poller.start()
    await clock.advance(1)

    def broken(self: LiveContextBook, now_ms: int) -> bool:
        raise RuntimeError("book")

    monkeypatch.setattr(LiveContextBook, "heartbeat", broken)
    await clock.advance(pol.POLLER_DEAD_AFTER_S + 61)
    now = clock.wall_ms()
    health = poller.health(LIVE_BOOK.freshness(now), now)
    assert health.running is True and health.alive is False
    await poller.stop()


async def test_a_restarted_provider_task_waits_the_floor_before_its_first_request() -> None:
    clock = FakeClock()
    client = FakeClient(clock, [fgi_ok()])
    poller = make(clock, fgi=client)
    poller.start()
    await clock.advance(5)  # the boot round is done
    first = len(client.calls)
    task = poller._providers[0].task
    assert task is not None
    task.cancel()
    await clock.advance(90)  # the tick at t = 60 restarts it
    assert counter("mtf_context_task_restarts_total", series=FGI) == 1
    assert len(client.calls) == first  # the restarted task is still inside its floor wait
    await poller.stop()


async def test_a_backward_wall_step_during_a_bucketed_fetch_never_parks_the_series() -> None:
    clock = AutoClock()

    def stepped(_kw: Any) -> Ok[Any]:
        clock.offset_ms -= 24 * H  # the wall clock steps back a day while the request is in flight
        return dom_ok()

    poller = make(clock, coingecko=FakeClient(clock, [stepped]))
    delay = await poller._cycle(poller._providers[0])
    assert pol.FLOOR_S <= delay <= pol.BUCKET_MS / 1_000 + pol.BUCKET_JITTER_MAX_S


# ---------------------------------------------------------------------------
# CR-P5 r1 additions: the production clock, the `alive` field over HTTP, the cycle deadline, the
# rolling restart hour, boot/bucket jitter and listener cleanup (FRED cost: RK-P5-03).
# ---------------------------------------------------------------------------
def test_the_poller_outcomes_are_the_provider_outcomes_plus_three() -> None:
    """CR-P5-10: a new provider outcome must be added here too, or /health would 500."""
    from data.provider_http import OUTCOMES as HTTP

    assert set(cp.OUTCOMES) == set(HTTP) | {"ok", "unchanged", "budget"}


async def test_the_system_clock_is_a_real_bounded_clock() -> None:
    """CR-P5-01: every other test runs on a fake clock; this one pins the production clock."""
    import time

    clock = cp.SystemClock()
    assert type(clock.wall_ms()) is int and abs(clock.wall_ms() - time.time() * 1000) < 5_000
    first = clock.monotonic()
    assert clock.monotonic() >= first

    async def never() -> None:
        await asyncio.Event().wait()

    async def fine() -> int:
        return 7

    async def boom() -> None:
        raise ValueError("inner")

    began = time.monotonic()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(3):  # a guard so that a broken `bounded` fails, not hangs
            await clock.bounded(never(), 0.05)
    assert 0.04 <= time.monotonic() - began < 1.0
    assert await clock.bounded(fine(), 1.0) == 7
    with pytest.raises(ValueError, match="inner"):
        await clock.bounded(boom(), 1.0)
    task = asyncio.ensure_future(clock.bounded(never(), 5.0))
    await asyncio.sleep(0)
    task.cancel()  # an external cancel is a cancel, never a TimeoutError
    with pytest.raises(asyncio.CancelledError):
        await task
    began = time.monotonic()
    await clock.sleep(0.05)
    assert time.monotonic() - began >= 0.04


async def test_the_health_endpoint_keeps_a_ticking_poller_alive_and_flags_a_stalled_one() -> None:
    """CR-P5-02 / CF-2401-64: `alive` over HTTP; each tick refreshes it, a stall clears it."""
    clock = FakeClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))
    container = AppContainer.for_testing()
    container.background_tasks.context_poller = poller
    app = FastAPI()
    app.state.container = container
    app.include_router(health_router.router, prefix="/api/v1")
    web = TestClient(app)
    poller.start()
    await clock.advance(10 * pol.HEARTBEAT_TICK_S)  # well past POLLER_DEAD_AFTER_S
    body = web.get("/api/v1/health/background").json()["context_poller"]
    assert body["running"] is True and body["alive"] is True
    assert body["heartbeat_age_seconds"] <= pol.HEARTBEAT_TICK_S
    clock.t += pol.POLLER_DEAD_AFTER_S + 1  # the loop stalls: no tick arrives
    body = web.get("/api/v1/health/background").json()["context_poller"]
    assert body["running"] is True and body["alive"] is False
    await poller.stop()


async def test_an_attempt_that_starts_late_in_the_cycle_is_cut_at_the_cycle_deadline() -> None:
    """CR-P5-03 / P5-PR-11: the 30 s bound holds by construction (attempt <= remaining)."""
    clock = FakeClock()
    client = FakeClient(clock, [whale_ok()], latency=1e9)  # never answers
    poller = make(clock, whale=client)
    await poller._sem.acquire()  # another provider holds the global semaphore
    settled: list[tuple[float, str]] = []
    clock.on_sleep = lambda: settled.append((clock.t, clock.sleeps[-1][0]))
    poller.start()
    await clock.advance(25)
    poller._sem.release()
    await clock.advance(30)
    await poller.stop()
    assert [round(c[0]) for c in client.calls] == [25]
    provider_sleeps = [round(t) for t, name in settled if name.startswith("context_poller:")]
    assert provider_sleeps == [0, 30]  # the cycle ended at its 30 s deadline, not at 25 + 10


async def test_the_restart_budget_is_a_rolling_hour() -> None:
    """CR-P5-04: three restarts per hour; the fourth waits for the hour to pass, then happens."""
    clock = FakeClock()
    poller = make(clock, fgi=FakeClient(clock, [fgi_ok()]))
    poller.start()
    provider = poller._providers[0]
    for _ in range(4):
        assert provider.task is not None
        provider.task.cancel()
        await clock.advance(61)
    assert counter("mtf_context_task_restarts_total", series=FGI) == 3
    await clock.advance(3_600)  # the first restart is now an hour old
    assert counter("mtf_context_task_restarts_total", series=FGI) == 4
    await poller.stop()


async def test_the_boot_round_and_the_bucket_start_are_actually_jittered() -> None:
    """CR-P5-06: the U(0, 30 s) boot stagger and the U(0, 300 s) bucket jitter are spread, not 0."""
    clock = FakeClock()
    clients = {
        k: FakeClient(clock, [ok])
        for k, ok in (
            ("fgi", fgi_ok()),
            ("coingecko", dom_ok()),
            ("fred", fred_ok()),
            ("whale", whale_ok()),
        )
    }
    await run(make(clock, rnd(21), **clients), clock, 6 * 3_600)
    firsts = [c.calls[0][0] for c in clients.values()]
    assert max(firsts) <= 30 and len({round(t, 3) for t in firsts}) == 4  # staggered
    offsets = {int(t * 1000 + START) % H for t, _ in clients["whale"].calls[1:]}
    assert len(offsets) > 1 and max(offsets) <= 300_000  # jittered inside the bucket's first 5 min


async def test_stop_cancels_a_pending_listener_call() -> None:
    """CR-P5-07: `stop()` also ends fire-and-forget listener tasks."""
    clock = FakeClock()

    class Hang(cp.NoopStatusListener):
        async def on_poller_dead(self) -> None:
            await asyncio.Event().wait()

    poller = make(clock, listener=Hang())
    poller.start()
    poller._notify(lambda: poller._listener.on_poller_dead())
    await clock._settle()
    pending = list(poller._bg)
    assert pending and not pending[0].done()
    await poller.stop()
    assert pending[0].cancelled() and not poller._bg
