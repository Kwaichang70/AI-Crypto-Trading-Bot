"""Context recorder and boot seed (CF-24-01 P7, SY-01-07, ER3-03, CF-2401-46/99/109).

``ContextRecorder`` is the poller's ``RecorderSink``: ``offer`` never blocks or raises, and one
writer task appends validated rows through ``ContextStore`` (no SQL here). A write failure is a
counter, an ``error_type`` log and ``on_recorder_failing``, never an error for the poller.
``seed`` replays the stored window into ``LIVE_BOOK`` at boot with each row's recorded stamp.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

import structlog

from api.services.context_poller import NoopStatusListener, StatusListener
from api.services.context_store import (
    ContextRow,
    ContextStore,
    load_window_safe,
    quantize_store_value,
)
from common.logging import exc_log_fields
from common.metrics import _label_key, metrics
from data.live_context import LIVE_BOOK
from data.point_in_time import BACK_7D_MS, SERIES_RULES, Observation
from data.provider_bands import DOMINANCE_MAX, DOMINANCE_MIN_EXCLUSIVE, WHALE_ABS_MAX, in_band

log = structlog.get_logger(__name__)

# SY-01-07 allowlist with its value band (low, high, low is open); FGI and T10Y2Y are not recorded.
_BANDS: Final[Mapping[str, tuple[Decimal, Decimal, bool]]] = {
    "coingecko:btc_dominance": (DOMINANCE_MIN_EXCLUSIVE, DOMINANCE_MAX, True),
    "whale:net_flow_1h": (-WHALE_ABS_MAX, WHALE_ABS_MAX, False),
}
SERIES: Final = tuple(_BANDS)
FAILURES: Final = "mtf_context_recorder_write_failures_total"  # exported as trading_<name>
DROPPED: Final = "mtf_context_record_dropped_total"
QUEUE_MAX: Final = 256  # drop-oldest when full
BATCH_MAX: Final = 64
DAILY_BUDGET: Final = 48  # rows per series and UTC day: a guard against a bucket bug
RECHECK_S: Final = 3_600.0  # missing table: try again after an hour
SEED_TIMEOUT_S: Final = 3.0
_DAY_MS: Final = 86_400_000
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_UNDEFINED_TABLE: Final = "42P01"


def _dt(ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=ms)  # exact: no float, so the store's ms check holds


def _row(series_id: str, o: Observation) -> ContextRow:
    """The validated store row of an offered observation; raises on any bad shape."""
    low, high, low_open = _BANDS[series_id]
    value = quantize_store_value(o.value)  # NaN, inf, bool and |v| >= 1e16 raise
    in_band(value, low, high, low_open=low_open)
    return ContextRow(series_id, _dt(o.observed_ms), _dt(o.available_ms), value, o.basis)


def _seedable(series_id: str, o: Observation, now_ms: int) -> bool:
    """SEC-P7-01: a stored row is re-banded like a provider row; recorded, already available."""
    try:
        _row(series_id, o)
    except Exception:
        return False
    return o.basis == "recorded" and o.available_ms <= now_ms


def _missing_table(err: Exception) -> bool:
    """SQLSTATE 42P01 on the error, its driver error or its cause (never the message text)."""
    return _UNDEFINED_TABLE in {
        getattr(e, "sqlstate", None) for e in (err, getattr(err, "orig", None), err.__cause__)
    }


class ContextRecorder:
    """``RecorderSink`` for ``ContextPoller`` plus the boot seed."""

    def __init__(
        self,
        store: ContextStore,
        listener: StatusListener | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store, self._mono = store, monotonic
        self._listener: StatusListener = listener or NoopStatusListener()
        self._queue: asyncio.Queue[ContextRow] = asyncio.Queue(QUEUE_MAX)
        self._task: asyncio.Task[None] | None = None
        self._held: set[tuple[str, int]] = set()  # (series, bucket) stored, seeded or queued
        self._daily: Counter[tuple[str, int]] = Counter()
        self._until: float | None = None  # not None: table missing; re-check after this time

    def start(self) -> None:
        """Start the writer and create the failure counters at 0 for the allowlist (DO-P6-01)."""
        for sid in SERIES:
            with metrics._lock:  # the collector has no public zero-init; increment needs >= 1
                metrics._counters.setdefault(_label_key(FAILURES, {"series": sid}), 0)
        if self._task is None:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run(), name="context_recorder")

    async def seed(self, now_ms: int) -> None:
        """Replay the stored window into the book. Never raises; a failed load leaves it empty.

        PIT: only rows with ``available <= now`` go in (``publish`` raises on a future stamp,
        AR4-07b), with their stored ``available_at``, one per bucket: the earliest (CF-2401-46).
        """
        window_start = now_ms - BACK_7D_MS - max(SERIES_RULES[s].max_age_ms for s in SERIES)
        try:
            async with asyncio.timeout(SEED_TIMEOUT_S):
                window = await load_window_safe(self._store, SERIES, _dt(window_start), _dt(now_ms))
        except TimeoutError:
            log.warning("context_recorder.seed_timeout")
            return
        for sid, series in window.series.items():
            self._held.update((sid, o.observed_ms) for o in series.observations)
            first: dict[int, Observation] = {}
            for o in sorted(series.observations, key=lambda o: o.available_ms):
                if _seedable(sid, o, now_ms):
                    first.setdefault(o.observed_ms, o)
            try:
                LIVE_BOOK.publish(sid, first.values(), now_ms)
            except Exception as exc:  # fail-closed: this series stays empty
                log.warning("context_recorder.seed_failed", series=sid, **exc_log_fields(exc))
        log.info("context_recorder.seeded", rows=window.row_count, reason=window.reason)

    # -- RecorderSink ------------------------------------------------------
    def offer(self, series_id: str, observation: Observation) -> None:
        """Never blocks, never raises. Full queue: the oldest row is dropped and counted."""
        if series_id not in _BANDS:
            return
        try:
            row = _row(series_id, observation)
        except Exception:  # out of band, NaN, wrong basis, available < observed ...
            return self._count(DROPPED, series_id)
        bucket = (series_id, observation.observed_ms)
        day = (series_id, observation.available_ms // _DAY_MS)
        if bucket in self._held:  # CF-2401-46: one row per bucket, whatever the value
            return
        if self._daily[day] >= DAILY_BUDGET:
            return self._count(DROPPED, series_id)
        self._held.add(bucket)
        self._daily[day] += 1
        if self._queue.full():
            self._count(DROPPED, self._queue.get_nowait().series_id)
        self._queue.put_nowait(row)

    async def flush(self, timeout_s: float) -> None:
        """Shutdown (SY-01-15): stop the writer, write the queue in ``timeout_s``, count the rest"""
        try:
            async with asyncio.timeout(timeout_s):  # CR-P7-01: the writer cancel too
                if self._task is not None:
                    self._task.cancel()
                    await asyncio.gather(self._task, return_exceptions=True)
                    self._task = None
                while not self._queue.empty():
                    await self._write(self._drain())
        finally:
            self._drop(self._drain(QUEUE_MAX))  # RK-P7-02: every row left is counted

    # -- writer ------------------------------------------------------------
    def _off(self) -> bool:
        return self._until is not None and self._mono() < self._until

    def _count(self, name: str, series_id: str) -> None:
        metrics.increment(name, labels={"series": series_id})  # allowlisted series only

    def _drain(self, limit: int = BATCH_MAX) -> list[ContextRow]:
        return [self._queue.get_nowait() for _ in range(min(limit, self._queue.qsize()))]

    def _drop(self, rows: list[ContextRow]) -> None:
        for r in rows:
            self._count(DROPPED, r.series_id)

    async def _run(self) -> None:
        while True:
            batch = [await self._queue.get(), *self._drain(BATCH_MAX - 1)]
            try:
                await self._write(batch)
            except Exception as exc:  # the writer outlives any bookkeeping bug
                log.error("context_recorder.writer_error", **exc_log_fields(exc))

    async def _write(self, batch: list[ContextRow]) -> None:
        if self._off():  # table missing: rows are dropped until the hourly re-check
            return self._drop(batch)
        try:
            await self._store.write_observations(batch)
        except asyncio.CancelledError:
            self._drop(batch)
            raise
        except Exception as exc:
            self._drop(batch)
            if _missing_table(exc):  # not a failure: disabled, one INFO line, hourly re-check
                if self._until is None:
                    log.info("context_recorder.disabled", reason="table_missing")
                self._until = self._mono() + RECHECK_S
                return
            for sid in {r.series_id for r in batch}:
                self._count(FAILURES, sid)
            log.warning("context_recorder.write_failed", rows=len(batch), **exc_log_fields(exc))
            try:
                await self._listener.on_recorder_failing()
            except Exception as lexc:
                log.warning("context_recorder.listener_failed", **exc_log_fields(lexc))
        else:
            if self._until is not None:
                self._until = None
                log.info("context_recorder.enabled")
