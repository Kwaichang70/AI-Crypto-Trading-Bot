"""CF-97-03 P7a: one research slot per API process (refuse-only, synchronous).

Backtest create, optimize, ML training and retraining take the slot; nothing waits for it, and no
live, paper, emergency, stop, health or read path ever imports this module (AST-pinned).  A lease
is released by ``release()`` (request jobs) or by the done-callback of the bound future (thread
jobs: a cancelled request must not free the slot while its thread still runs).  State is plain
synchronous data: no asyncio primitive exists at import time.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from typing import Any, Final, Literal, NoReturn, Protocol

import structlog
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry, Pool, PoolProxiedConnection

from common.logging import exc_log_fields
from common.metrics import metrics

Kind = Literal["backtest", "optimize", "ml_train", "ml_retrain"]
HOLD_LONG_S: Final = 1800.0
RETRY_AFTER_S: Final = 30
_log = structlog.get_logger("admission")
_IN_RESEARCH: contextvars.ContextVar[bool] = contextvars.ContextVar("research_job", default=False)
_held: Lease | None = None
_pool_ready = False


class Releasable(Protocol):
    def release(self) -> None: ...


class AdmissionRefused(HTTPException):
    """429 ``{"code": "admission_busy"}`` with a constant ``Retry-After``; nothing else leaks."""

    def __init__(self) -> None:
        super().__init__(429, {"code": "admission_busy"}, {"Retry-After": str(RETRY_AFTER_S)})


class Lease:
    """Idempotent handle on the slot; ``bind_thread`` hands release over to a future."""

    def __init__(self, kind: Kind, task: asyncio.Task[Any] | None) -> None:
        self.kind, self.task, self.t0, self.open = kind, task, time.monotonic(), True
        self.token: contextvars.Token[bool] | None = None
        self.thread: asyncio.Future[Any] | None = None
        loop = asyncio.get_running_loop()
        self.timer = loop.call_later(HOLD_LONG_S, self._long) if task else None

    def _long(self) -> None:
        _log.warning("admission.hold_long", kind=self.kind, seconds=HOLD_LONG_S)

    def bind_thread(self, future: asyncio.Future[Any]) -> None:
        self.thread = future
        future.add_done_callback(self._thread_done)

    def _thread_done(self, future: asyncio.Future[Any]) -> None:
        if not future.cancelled() and (exc := future.exception()) is not None:
            _log.warning("admission.thread_job_failed", kind=self.kind, **exc_log_fields(exc))
        self._free()

    def release(self) -> None:
        if self.token is not None:  # CF-97-26: reset in the acquiring context, never a callback's
            with contextlib.suppress(ValueError):
                _IN_RESEARCH.reset(self.token)
            self.token = None
        if self.thread is None:
            self._free()

    def _free(self) -> None:
        global _held
        if not self.open:
            return
        self.open = False
        if self.timer is not None:
            self.timer.cancel()
        if _held is self:
            _held = None
            metrics.gauge("admission_slot_held", 0.0)
            metrics.observe(
                "admission_hold_seconds", time.monotonic() - self.t0, {"kind": self.kind}
            )
        else:
            _log.error("admission_release_unmatched", kind=self.kind)


class _NoLease:
    """The lease of a create that takes no slot (paper, live): releasing it is a no-op."""

    def release(self) -> None:
        return None


NO_LEASE: Final = _NoLease()


def _leaked() -> bool:
    held = _held
    if held is None or not held.open:
        return False
    done = (
        held.thread.done() if held.thread is not None else (held.task is None or held.task.done())
    )
    if done:
        _log.error("admission.slot_leak_recovered", kind=held.kind)
        held._free()
    return done


def _install_pool_timing() -> None:
    global _pool_ready
    _pool_ready = True
    try:
        from api.config import get_settings

        settings = get_settings()
        if settings.db_pool_size + settings.db_max_overflow < 9:
            _log.warning(
                "admission.pool_total_low", total=settings.db_pool_size + settings.db_max_overflow
            )

        @event.listens_for(Pool, "checkout")
        def _out(
            dbapi: DBAPIConnection, record: ConnectionPoolEntry, proxy: PoolProxiedConnection
        ) -> None:
            with contextlib.suppress(Exception):
                if _IN_RESEARCH.get() and _held is not None:
                    record.info["research_t0"] = time.monotonic()

        @event.listens_for(Pool, "checkin")
        def _in(dbapi: DBAPIConnection, record: ConnectionPoolEntry) -> None:
            with contextlib.suppress(Exception):
                if (t0 := record.info.pop("research_t0", None)) is not None:
                    metrics.observe("admission_checkout_seconds", time.monotonic() - t0)

    except Exception:  # never block an admission for the sake of a metric
        _log.error("admission.pool_timing_unavailable")


def try_acquire(kind: Kind) -> Lease:
    """Take the slot or raise ``AdmissionRefused``; synchronous (no await between test and set)."""
    global _held
    if _held is not None and _held.open and not _leaked():
        _refuse(kind)
    if _held is not None:  # a closed lease left in the slot: inconsistent, fail closed
        _log.error("admission.state_inconsistent")
        _refuse(kind)
    if not _pool_ready:
        _install_pool_timing()
    metrics.gauge("admission_slot_held", 1.0)  # CF-97-31: before the slot, so a failure leaks none
    lease = Lease(kind, asyncio.current_task())
    lease.token = _IN_RESEARCH.set(True)
    _held = lease
    return lease


def _refuse(kind: Kind) -> NoReturn:
    metrics.increment("admission_refused_total", labels={"code": "admission_busy", "kind": kind})
    _log.warning("admission.refused", kind=kind)
    raise AdmissionRefused


def _reset_for_tests() -> None:
    global _held
    if _held is not None and _held.timer is not None:
        _held.timer.cancel()
    _held = None
