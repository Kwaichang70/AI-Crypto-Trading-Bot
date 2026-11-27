"""
apps/api/services/context_poller.py
------------------------------------
CF-24-01 P5: the one supervised poller that feeds ``LIVE_BOOK`` (SY-01-03/08/15, SY-01-21a/b/c/j,
SY-01-22d, ER3-03, CF-2401-45..69).

One task per configured provider plus a supervisor tick (heartbeat, restart <= 3/h, metrics,
status changes). One cycle uses ONE integer ``fetch_ms`` (CF-2401-50). Order per validated
``Ok``: convert, ``heartbeat`` (CF-2401-59), ``publish``; then ``adopt_validated`` and, for a
recorded series only, the recorder offer iff ``publish`` returned 1 and
``freshness(fetch_ms).regressed`` is False (ER3-03). The semaphore is never held across a sleep,
and its wait counts against the 30 s cycle (N-6). ``mtf_context_regressed_reads`` counts ENGINE
reads, consumer and non-consumer alike (CF-2401-69). Logs carry ``error_type`` metadata and
closed codes only. No ``trading.*`` import (T-30); the Telegram listener (P6) and the recorder
(P7) plug into ``StatusListener`` / ``RecorderSink``, no-ops here.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol, TypeVar, get_args

import structlog

import data.live_context_policy as policy
from common.logging import exc_log_fields
from common.metrics import metrics
from data.live_context import (
    LIVE_BOOK,
    LIVE_CONSUMERS,
    BookFreshness,
    fgi_observations,
    snapshot_observation,
    t10y2y_observations,
)
from data.point_in_time import Observation, to_ms
from data.provider_http import Err, ErrReason, Ok

logger = structlog.get_logger(__name__)

OUTCOMES: Final = (  # closed ``poll_total{outcome}`` vocabulary (SY-01-18)
    "ok", "unchanged", "http_3xx", "http_4xx", "http_429", "http_5xx", "timeout", "connect_error",
    "egress_blocked", "invalid_payload", "oversize", "auth_failed", "budget",
)  # fmt: skip
STATUSES: Final = (
    "fresh", "stale", "invalid", "unavailable", "not_configured", "egress_blocked", "auth_failed",
)  # fmt: skip
REASONS: Final[frozenset[str]] = frozenset(get_args(ErrReason))
_RETRYABLE: Final = frozenset({"timeout", "connect_error", "http_5xx", "http_429"})
_SUCCESS: Final = frozenset({"ok", "unchanged"})
# CF-2401-35: classes are decided here because provider_http only maps proxy errors itself.
_EGRESS_ERROR_TYPES: Final = frozenset({"ClientConnectorDNSError"})
_PARSER_ERROR_TYPES: Final = frozenset({"ClientResponseError", "ContentTypeError", "InvalidURL"})
_DAYS: Final = "Mon Tue Wed Thu Fri Sat Sun".split()
_MONTHS: Final = {
    m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)
}
_RA_SECONDS: Final = re.compile(r"[0-9]{1,6}")
_RA_DATE: Final = re.compile(
    r"([A-Z][a-z]{2}), ([0-9]{2}) ([A-Z][a-z]{2}) ([0-9]{4}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) GMT"
)
_MAX_RESTARTS_PER_HOUR: Final = 3
_FGI: Final = "alternative_me:fgi"
_DOMINANCE: Final = "coingecko:btc_dominance"
_T10Y2Y: Final = "fred:T10Y2Y"
_WHALE: Final = "whale:net_flow_1h"
_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Seams: clock, status listener (P6), recorder sink (P7)
# ---------------------------------------------------------------------------
class PollerClock(Protocol):
    def wall_ms(self) -> int: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...
    async def bounded(self, awaitable: Awaitable[_T], seconds: float) -> _T:
        """Await ``awaitable`` for at most ``seconds``; ``TimeoutError`` otherwise."""
        ...


class SystemClock:
    def wall_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def bounded(self, awaitable: Awaitable[_T], seconds: float) -> _T:
        async with asyncio.timeout(seconds):
            return await awaitable


class StatusListener(Protocol):
    """SY-01-21j. Fixed strings only downstream; exceptions are caught by the poller."""

    async def on_status_change(self, series: str, old: str, new: str) -> None: ...
    async def on_poller_dead(self) -> None: ...
    async def on_recorder_failing(self) -> None: ...


class NoopStatusListener:
    async def on_status_change(self, series: str, old: str, new: str) -> None:
        return None

    async def on_poller_dead(self) -> None:
        return None

    async def on_recorder_failing(self) -> None:
        return None


class RecorderSink(Protocol):
    """P7 hook. ``offer`` is synchronous and must never block; the poller never awaits it."""

    def offer(self, series_id: str, observation: Observation) -> None: ...
    async def flush(self, timeout_s: float) -> None: ...


class NoopRecorderSink:
    def offer(self, series_id: str, observation: Observation) -> None:
        return None

    async def flush(self, timeout_s: float) -> None:
        return None


# ---------------------------------------------------------------------------
# Value types and pure helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SeriesHealth:
    series_id: str
    status: str  # closed ``Status`` vocabulary
    last_outcome: str | None  # closed ``Outcome`` vocabulary
    last_reason: str | None  # closed ``ErrReason`` vocabulary
    rows: int
    age_seconds: float | None
    failed_cycles: int
    task_alive: bool


@dataclass(frozen=True, slots=True)
class PollerHealth:
    configured: bool
    running: bool
    alive: bool  # running and a supervisor heartbeat within ``POLLER_DEAD_AFTER_S``
    heartbeat_age_seconds: float | None
    regressed: bool
    series: tuple[SeriesHealth, ...]


def derive_status(
    *,
    configured: bool,
    rows: int,
    fresh: bool,
    regressed: bool,
    last_outcome: str | None,
) -> str:
    """Closed health status of one series (SY-01-18). A-2: while the clock is regressed a
    would-be ``fresh`` is shown as ``stale``; every other status is unchanged."""
    if not configured:
        return "not_configured"
    if last_outcome == "auth_failed":
        return "auth_failed"
    if last_outcome == "egress_blocked":
        return "egress_blocked"
    if fresh:
        return "stale" if regressed else "fresh"
    if last_outcome == "invalid_payload":
        return "invalid"
    return "stale" if rows else "unavailable"


def parse_retry_after(value: str | None, wall_ms: int) -> int | None:
    """Seconds from a header already shape-checked by ``provider_http``; ``None`` if unusable.

    delta-seconds are used as is; an HTTP-date is parsed for real and measured from ``wall_ms``.
    A date in the past is 0. Anything else (and any unparseable date) is ``None``.
    """
    if value is None:
        return None
    if _RA_SECONDS.fullmatch(value):
        return int(value)
    match = _RA_DATE.fullmatch(value)
    if match is None or match[3] not in _MONTHS:
        return None
    try:
        parts = (int(match[4]), _MONTHS[match[3]], int(match[2]))
        when = datetime(*parts, int(match[5]), int(match[6]), int(match[7]), tzinfo=UTC)
    except ValueError:
        return None
    if _DAYS[when.weekday()] != match[1]:
        return None
    return max(0, min((to_ms(when) - wall_ms) // 1_000, 999_999))


def _label(reason: str | None) -> str:
    return reason if reason in REASONS else "unknown"


@dataclass(frozen=True, slots=True)
class _Result:
    outcome: str
    reason: str | None = None
    retry_after_s: int | None = None

    @property
    def retryable(self) -> bool:
        return self.outcome in _RETRYABLE


@dataclass(eq=False)
class _Provider:
    series_id: str
    client: Any
    convert: Callable[[Any, int], tuple[Observation, ...]]
    recorded: bool  # offered to the recorder (dominance, whale; SY-01-07)
    takes_now_ms: bool  # the fetch takes the cycle's fetch_ms (CF-2401-50)
    cadence: policy.Cadence
    failed_cycles: int = 0
    stamps_ms: deque[int] = field(default_factory=deque)  # monotonic ms of each request sent
    restarts: deque[float] = field(default_factory=deque)
    last_outcome: str | None = None
    last_reason: str | None = None
    task: asyncio.Task[None] | None = None
    inflight: bool = False

    def fetch(self, fetch_ms: int) -> Awaitable[Ok[Any] | Err]:
        if self.takes_now_ms:
            return self.client.fetch_validated(now_ms=fetch_ms)  # type: ignore[no-any-return]
        return self.client.fetch_validated()  # type: ignore[no-any-return]

    def adopt(self, snapshot: Any, now_mono: float) -> None:  # noqa: ANN401
        self.client.adopt_validated(snapshot, now_mono=now_mono)


def _day_ms(day: date) -> int:
    return to_ms(datetime(day.year, day.month, day.day, tzinfo=UTC))


def _fgi_rows(snaps: Any, fetch_ms: int) -> tuple[Observation, ...]:  # noqa: ANN401
    return fgi_observations([(to_ms(s.timestamp), s.value) for s in snaps], fetch_ms)


def _dominance_rows(snap: Any, fetch_ms: int) -> tuple[Observation, ...]:  # noqa: ANN401
    return (snapshot_observation(_DOMINANCE, snap.btc_dominance, fetch_ms),)


def _whale_rows(snap: Any, fetch_ms: int) -> tuple[Observation, ...]:  # noqa: ANN401
    return (snapshot_observation(_WHALE, snap.net_flow, fetch_ms),)


def _fred_rows(snap: Any, fetch_ms: int) -> tuple[Observation, ...]:  # noqa: ANN401
    """CF-2401-49/65: the page arrives pre-filtered (released, in-band); empty is invalid."""
    points = [(_day_ms(day), value) for day, value in snap.yield_curve_rows]
    if not points:
        raise ValueError("empty_page")
    return t10y2y_observations(points, fetch_ms)


#: (client argument, series, converter, recorded, fetch takes now_ms)
_KINDS: Final = (
    ("fgi", _FGI, _fgi_rows, False, False),
    ("coingecko", _DOMINANCE, _dominance_rows, True, True),
    ("fred", _T10Y2Y, _fred_rows, False, False),
    ("whale", _WHALE, _whale_rows, True, True),
)


# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------
class ContextPoller:
    def __init__(
        self,
        *,
        fgi: Any = None,  # noqa: ANN401
        coingecko: Any = None,  # noqa: ANN401
        fred: Any = None,  # noqa: ANN401
        whale: Any = None,  # noqa: ANN401
        listener: StatusListener | None = None,
        sink: RecorderSink | None = None,
        clock: PollerClock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        clients = {"fgi": fgi, "coingecko": coingecko, "fred": fred, "whale": whale}
        self._providers = [
            _Provider(series, client, convert, recorded, takes_ms, policy.CADENCE[series])
            for arg, series, convert, recorded, takes_ms in _KINDS
            if (client := clients[arg]) is not None  # built without session= (CF-2401-33)
        ]
        self._listener: StatusListener = listener or NoopStatusListener()
        self._sink: RecorderSink = sink or NoopRecorderSink()
        self._timebase: PollerClock = clock or SystemClock()
        self._rng = rng or random.Random()  # noqa: S311 -- jitter, not a secret
        self._sem = asyncio.Semaphore(1)  # global: one HTTP attempt at a time
        self._task: asyncio.Task[None] | None = None
        self._bg: set[asyncio.Task[None]] = set()
        self._stopping = False
        self._regressed = False
        self._last_tick_mono: float | None = None
        self._status: dict[str, tuple[str, float]] = {}  # series -> (status, last log mono)
        self._fgi_days: set[int] = set()
        self._log = logger.bind(component="context_poller")

    # -- lifecycle ---------------------------------------------------------
    def now_ms(self) -> int:
        """The poller clock's wall time as an exact ``int`` (health reads the book at this ms)."""
        return self._timebase.wall_ms()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task[None]:
        """Start the supervisor and the provider tasks. A second call returns the same task."""
        if self._task is not None:
            return self._task
        loop = asyncio.get_running_loop()
        self._last_tick_mono = self._timebase.monotonic()
        self._task = loop.create_task(self._supervise(), name="context_poller")
        for p in self._providers:
            self._spawn(p, self._rng.random() * policy.STARTUP_STAGGER_S)
        return self._task

    async def stop(self) -> None:
        """Cancel the supervisor, the provider tasks and pending listener calls. Idempotent."""
        self._stopping = True
        tasks = [t for t in (self._task, *(p.task for p in self._providers), *self._bg) if t]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        metrics.gauge("mtf_context_poller_alive", 0.0)

    async def flush_sink(self, timeout_s: float = 5.0) -> None:
        """Shutdown step after ``stop``: flush the recorder for at most ``timeout_s`` (SY-01-15)."""
        try:
            await self._timebase.bounded(self._sink.flush(timeout_s), timeout_s)
        except TimeoutError:
            metrics.increment("mtf_context_record_flush_timeout_total")
        except Exception as exc:
            self._log.warning("context_poller.flush_failed", **exc_log_fields(exc))

    def on_supervisor_done(self, task: asyncio.Task[None]) -> None:
        """N-5/A-3: done-callback of the supervisor task. A shutdown cancel is not a death."""
        metrics.gauge("mtf_context_poller_alive", 0.0)
        for sid in policy.CADENCE:  # RK-P5-01: nothing refreshes these now; never leave "fresh"
            metrics.gauge("mtf_context_fresh", 0.0, labels={"series": sid})
        if self._stopping:
            return
        failure = None if task.cancelled() else task.exception()
        fields = exc_log_fields(failure) if failure is not None else {}
        self._log.error("context_poller.dead", cancelled=task.cancelled(), **fields)
        self._notify(lambda: self._listener.on_poller_dead())

    def _spawn(self, p: _Provider, delay_s: float) -> None:
        p.task = asyncio.get_running_loop().create_task(
            self._provider_loop(p, delay_s), name=f"context_poller:{p.series_id}"
        )

    def _notify(self, call: Callable[[], Awaitable[None]]) -> None:
        """Run a listener call in the background; its failure never reaches the schedule."""

        async def guarded() -> None:
            try:
                await call()
            except Exception as exc:
                self._log.warning("context_poller.listener_failed", **exc_log_fields(exc))

        try:
            task = asyncio.get_running_loop().create_task(guarded())
        except RuntimeError:  # no running loop (interpreter shutdown)
            return
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # -- supervisor --------------------------------------------------------
    async def _supervise(self) -> None:
        while True:
            self._tick()
            await self._timebase.sleep(float(max(policy.HEARTBEAT_TICK_S, policy.FLOOR_S)))

    def _warn_exc(self, event: str, series: str, exc: Exception) -> None:
        self._log.warning(event, series=series, **exc_log_fields(exc))

    def _tick(self) -> None:
        try:
            now_ms = self._timebase.wall_ms()
            self._beat(now_ms)
            self._last_tick_mono = self._timebase.monotonic()
            self._ensure_tasks()
            self._export(now_ms)
        except Exception as exc:  # the supervisor never dies of its own bookkeeping
            self._log.warning("context_poller.tick_failed", **exc_log_fields(exc))

    def _beat(self, now_ms: int) -> bool:
        """``LIVE_BOOK.heartbeat``; counts and logs the regressed/recovered transitions."""
        regressed = LIVE_BOOK.heartbeat(now_ms)
        if regressed:
            metrics.increment("mtf_context_clock_regressed_total")
            if not self._regressed:
                self._log.warning("mtf_context.clock_regressed")
        elif self._regressed:
            self._log.info("mtf_context.clock_recovered")
        self._regressed = regressed
        return regressed

    def _ensure_tasks(self) -> None:
        """Restart a dead provider task, at most ``_MAX_RESTARTS_PER_HOUR`` per hour."""
        mono = self._timebase.monotonic()
        for p in self._providers:
            if p.task is None or not p.task.done() or self._stopping:
                continue
            failure = None if p.task.cancelled() else p.task.exception()
            fields = exc_log_fields(failure) if failure is not None else {}
            while p.restarts and mono - p.restarts[0] >= 3_600:
                p.restarts.popleft()
            if len(p.restarts) >= _MAX_RESTARTS_PER_HOUR:
                continue  # left dead: the series ages to stale and the health block shows it
            p.restarts.append(mono)
            metrics.increment("mtf_context_task_restarts_total", labels={"series": p.series_id})
            self._log.warning("context_poller.task_restarted", series=p.series_id, **fields)
            self._spawn(p, float(policy.FLOOR_S))

    def _series(self, book: BookFreshness) -> Iterator[tuple[str, _Provider | None, Any, str]]:
        """(series, provider, freshness row, status) for the four series, configured or not."""
        configured = {p.series_id: p for p in self._providers}
        for sid in policy.CADENCE:
            p, row = configured.get(sid), book.series.get(sid)
            status = derive_status(
                configured=p is not None,
                rows=row.rows if row else 0,
                fresh=bool(row and row.fresh),
                regressed=book.regressed,
                last_outcome=p.last_outcome if p else None,
            )
            yield sid, p, row, status

    def _export(self, now_ms: int) -> None:
        """Gauges and status-change detection from ``freshness`` (never mutates the book)."""
        book = LIVE_BOOK.freshness(now_ms)
        mono = self._timebase.monotonic()
        metrics.gauge("mtf_context_poller_alive", 1.0)
        metrics.gauge("mtf_context_poller_heartbeat_timestamp_seconds", now_ms / 1_000)
        metrics.gauge("mtf_context_regressed_reads", float(LIVE_BOOK.regressed_reads))
        for mode, count in LIVE_CONSUMERS.counts().items():
            metrics.gauge("mtf_context_consumer_runs_active", float(count), labels={"mode": mode})
        for sid, p, row, status in self._series(book):
            labels = {"series": sid}
            metrics.gauge("mtf_context_series_configured", float(p is not None), labels=labels)
            metrics.gauge("mtf_context_fresh", float(status == "fresh"), labels=labels)
            if row is not None and row.age_ms is not None:
                metrics.gauge("mtf_context_age_seconds", row.age_ms / 1_000, labels=labels)
            self._note_status(sid, status, mono)

    def _note_status(self, sid: str, status: str, mono: float) -> None:
        """One WARNING per status change, a reminder at most every 6 h while it stays bad."""
        previous = self._status.get(sid)
        if previous is None:
            self._status[sid] = (status, mono)
            return
        old, logged = previous
        if status != old:
            self._status[sid] = (status, mono)
            self._log.warning("mtf_context.status_changed", series=sid, old=old, new=status)
            self._notify(lambda: self._listener.on_status_change(sid, old, status))
        elif status not in ("fresh", "not_configured") and (
            mono - logged >= policy.STATUS_REMINDER_S
        ):
            self._status[sid] = (status, mono)
            self._log.warning("mtf_context.status_reminder", series=sid, status=status)

    def health(self, book: BookFreshness, now_ms: int) -> PollerHealth:
        """Closed-vocabulary snapshot for ``/health/background``; touches no book state."""
        age = (
            None
            if self._last_tick_mono is None
            else self._timebase.monotonic() - self._last_tick_mono
        )
        series = tuple(
            SeriesHealth(
                sid,
                status,
                p.last_outcome if p else None,
                _label(p.last_reason) if p and p.last_reason else None,
                row.rows if row else 0,
                None if row is None or row.age_ms is None else row.age_ms / 1_000,
                p.failed_cycles if p else 0,
                bool(p and p.task is not None and not p.task.done()),
            )
            for sid, p, row, status in self._series(book)
        )
        return PollerHealth(
            configured=bool(self._providers),
            running=self.running,
            alive=self.running and age is not None and age <= policy.POLLER_DEAD_AFTER_S,
            heartbeat_age_seconds=age,
            regressed=book.regressed,
            series=series,
        )

    # -- provider loop -----------------------------------------------------
    async def _provider_loop(self, p: _Provider, delay_s: float) -> None:
        while True:
            await self._timebase.sleep(max(0.0, delay_s))
            if p.inflight:  # single-flight per provider
                delay_s = float(policy.FLOOR_S)
                continue
            p.inflight = True
            try:
                delay_s = await self._cycle(p)
            except Exception as exc:
                p.failed_cycles += 1
                p.last_outcome, p.last_reason = "invalid_payload", "internal_error"
                self._warn_exc("context_poller.cycle_failed", p.series_id, exc)
                delay_s = float(policy.next_interval_s(p.series_id, p.failed_cycles))
            finally:
                p.inflight = False

    async def _cycle(self, p: _Provider) -> float:
        """One cycle; returns the delay in seconds to the next cycle start."""
        clock = self._timebase
        fetch_ms = clock.wall_ms()  # the ONE fetch_ms of this cycle (CF-2401-50)
        start = clock.monotonic()
        deadline = start + policy.CYCLE_BUDGET_S
        bound = float(policy.ATTEMPT_BOUND_S * p.cadence.requests_per_cycle)  # CF-2401-32
        result = await self._attempt(p, fetch_ms, deadline, bound)
        for retry in range(policy.MAX_RETRIES):
            if not result.retryable:
                break
            wait = policy.backoff_s(retry, self._rng.random())
            if result.retry_after_s is not None:
                wait = max(wait, float(result.retry_after_s))
            if not policy.wait_fits_cycle(wait, deadline - clock.monotonic(), bound):
                break
            await clock.sleep(wait)  # the semaphore is not held here
            result = await self._attempt(p, fetch_ms, deadline, bound)
        return self._settle(p, result, fetch_ms, start)

    async def _attempt(self, p: _Provider, fetch_ms: int, deadline: float, bound: float) -> _Result:
        clock = self._timebase
        cost = p.cadence.requests_per_cycle
        now_ms = int(clock.monotonic() * 1_000)
        while p.stamps_ms and p.stamps_ms[0] <= now_ms - policy.BUDGET_WINDOW_MS:
            p.stamps_ms.popleft()
        if not policy.budget_allows(p.stamps_ms, now_ms, p.cadence.budget_per_hour, cost):
            return _Result("budget")  # no request sent; counts as an attempt (RK-P3-04)
        try:  # N-6: the semaphore wait counts against the cycle
            await clock.bounded(self._sem.acquire(), max(0.0, deadline - clock.monotonic()))
        except TimeoutError:
            return _Result("timeout")  # A-4: no request sent, no budget spent
        try:
            remaining = deadline - clock.monotonic()
            if remaining <= 0:
                return _Result("timeout")
            now_ms = int(clock.monotonic() * 1_000)
            p.stamps_ms.extend([now_ms] * cost)
            try:
                got = await clock.bounded(p.fetch(fetch_ms), min(bound, remaining))
            except TimeoutError:
                return _Result("timeout")
            except Exception as exc:
                self._warn_exc("context_poller.fetch_failed", p.series_id, exc)
                return _Result("invalid_payload", "internal_error")
        finally:
            self._sem.release()
        if isinstance(got, Err):
            return self._classify(got)
        return self._apply(p, got.value, fetch_ms)

    def _classify(self, err: Err) -> _Result:
        outcome, reason = err.outcome, err.reason
        if outcome == "connect_error" and err.error_type in _EGRESS_ERROR_TYPES:
            outcome = "egress_blocked"
        elif outcome == "connect_error" and err.error_type in _PARSER_ERROR_TYPES:
            outcome, reason = "invalid_payload", "bad_status"
        header = err.retry_after if outcome in ("http_429", "http_5xx") else None
        return _Result(outcome, reason, parse_retry_after(header, self._timebase.wall_ms()))

    def _apply(self, p: _Provider, snapshot: Any, fetch_ms: int) -> _Result:  # noqa: ANN401
        """ER3-03 order. Synchronous from the heartbeat to the offer: no ``await`` in between."""
        try:
            rows = p.convert(snapshot, fetch_ms)
            self._beat(fetch_ms)  # CF-2401-59: same fetch_ms, immediately before publish
            applied = LIVE_BOOK.publish(p.series_id, rows, fetch_ms)
        except Exception as exc:  # SEC-P3-08: any converter/publish error is invalid_payload
            self._warn_exc("context_poller.apply_failed", p.series_id, exc)
            return _Result("invalid_payload", "internal_error")
        try:
            p.adopt(snapshot, self._timebase.monotonic())
        except Exception as exc:  # the book is already right; the display caches keep the old value
            self._warn_exc("context_poller.adopt_failed", p.series_id, exc)
        try:
            if p.recorded and applied == 1 and not LIVE_BOOK.freshness(fetch_ms).regressed:
                self._sink.offer(p.series_id, rows[-1])
            if p.series_id == _FGI and applied:
                self._note_fgi(max(o.observed_ms for o in rows), fetch_ms)
        except Exception as exc:  # a recorder failure never reaches the schedule
            self._warn_exc("context_poller.offer_failed", p.series_id, exc)
        return _Result("ok" if applied else "unchanged")

    def _note_fgi(self, newest_ms: int, fetch_ms: int) -> None:
        """SY-01-06: log the first sighting of each FGI day (CF-24-16 evidence)."""
        if newest_ms in self._fgi_days:
            return
        self._fgi_days = {*sorted(self._fgi_days)[-63:], newest_ms}
        self._log.info(
            "mtf_context.fgi_first_seen",
            observed_day=datetime.fromtimestamp(newest_ms / 1_000, UTC).date().isoformat(),
            first_seen=datetime.fromtimestamp(fetch_ms / 1_000, UTC).isoformat(),
        )

    def _settle(self, p: _Provider, result: _Result, fetch_ms: int, start: float) -> float:
        """Bookkeeping and the delay to the next cycle start (SY-01-08/21c/22d, CF-2401-52/53)."""
        clock = self._timebase
        outcome, sid = result.outcome, p.series_id
        metrics.increment("mtf_context_poll_total", labels={"series": sid, "outcome": outcome})
        if outcome == "invalid_payload":
            labels = {"series": sid, "reason": _label(result.reason)}
            metrics.increment("mtf_context_value_rejected_total", labels=labels)
        previous, p.last_outcome, p.last_reason = p.last_outcome, outcome, result.reason
        success = outcome in _SUCCESS
        p.failed_cycles = 0 if success else p.failed_cycles + 1
        if success:
            self._log.debug("context_poller.polled", series=sid, outcome=outcome)
        elif outcome != previous:
            level = self._log.error if outcome == "auth_failed" else self._log.warning
            fields = {"series": sid, "outcome": outcome, "reason": _label(result.reason)}
            level(
                "context_poller.poll_failed",
                failed_cycles=p.failed_cycles,
                retry_after_s=result.retry_after_s,
                **fields,
            )
        end, unit = clock.monotonic(), self._rng.random()
        if outcome == "auth_failed":
            return float(policy.AUTH_FAILED_WAIT_S)
        if success and p.cadence.bucketed:  # no second attempt in this bucket (SY-01-22d)
            nxt = policy.first_attempt_ms(policy.bucket_start_ms(fetch_ms) + policy.BUCKET_MS, unit)
            most = policy.BUCKET_MS / 1_000 + policy.BUCKET_JITTER_MAX_S  # RK-P5-02: wall steps
            return max(float(policy.FLOOR_S), min((nxt - clock.wall_ms()) / 1_000, most))
        if result.retry_after_s is not None:  # header-imposed: may exceed cap/4, never below floor
            return float(policy.retry_after_next_s(result.retry_after_s))
        base = p.cadence.interval_s if success else policy.next_interval_s(sid, p.failed_cycles)
        gap = policy.clamped_gap_s(sid, float(base), unit)
        return max(start + gap - end, float(policy.FLOOR_S))  # gap measured between cycle starts
