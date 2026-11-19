"""CF-97-03 P7a (T-a, T-d, T-e, T-f, T-g, T-k, T-l): the research admission slot.

Deterministic: loop futures and threading events only; no wall-time assertion, no Postgres.  The
slot is process-global state, so every test starts and ends with ``_reset_for_tests``.  T-k reuses
the helpers (``Log``, ``Rec``, ``Up``, ``get``, ``source``) of ``test_wp26_bar_source.py``, an
append-only file whose names are stable.
"""

from __future__ import annotations

import asyncio
import threading
import types
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.requests import Request
from structlog.testing import capture_logs

import api.routers.ml as ml_module
import api.routers.runs as runs_module
from api.config import get_settings
from api.services import admission, ohlcv_store
from api.services.retraining import RetrainingService
from common.types import TimeFrame
from data.market_data import MarketDataError
from data.ohlcv_cache import OhlcvRequireError, OhlcvStoreError, SeriesKey
from tests.unit.test_wp26_bar_source import Log, Rec, Up, get, source

_LEAK = "CF9703-SECRET-TEXT"


@pytest.fixture(autouse=True)
def _slot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    get_settings.cache_clear()
    admission._reset_for_tests()
    yield
    admission._reset_for_tests()
    get_settings.cache_clear()


@pytest.fixture
def m(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    rec = MagicMock()
    monkeypatch.setattr(admission, "metrics", rec)
    return rec


def _events(logs: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == name]


async def _until(cond: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):  # a hang guard only: the condition, not time, ends the wait
        while not cond():
            await asyncio.sleep(0.001)


async def _until_free() -> None:
    await _until(lambda: admission._held is None)


# --------------------------------------------------------------------------------------------
# T-a: slot semantics
# --------------------------------------------------------------------------------------------
async def test_a_one_is_admitted_and_the_second_gets_the_exact_closed_refusal(m: MagicMock) -> None:
    first = admission.try_acquire("backtest")
    with pytest.raises(admission.AdmissionRefused) as refused:
        admission.try_acquire("optimize")
    exc = refused.value
    assert (exc.status_code, exc.detail) == (429, {"code": "admission_busy"})
    assert exc.headers == {"Retry-After": "30"}
    m.increment.assert_called_once_with(
        "admission_refused_total", labels={"code": "admission_busy", "kind": "optimize"}
    )
    assert admission._held is first
    first.release()
    assert admission._held is None
    m.gauge.assert_any_call("admission_slot_held", 1.0)
    m.gauge.assert_called_with("admission_slot_held", 0.0)
    assert m.observe.call_args.args[0] == "admission_hold_seconds"
    assert m.observe.call_args.args[2] == {"kind": "backtest"}
    admission.try_acquire("ml_train").release()  # free again: the next one is admitted


async def test_a_double_release_is_a_no_op_and_an_unmatched_one_logs_a_closed_code(
    m: MagicMock,
) -> None:
    lease = admission.try_acquire("backtest")
    lease.release()
    lease.release()
    assert m.observe.call_count == 1
    second = admission.try_acquire("optimize")
    lease.release()  # the stale handle never frees the new holder
    assert admission._held is second
    admission._reset_for_tests()  # the slot is cleared under a live handle
    with capture_logs() as logs:
        second.release()  # must not raise
    (event,) = _events(logs, "admission_release_unmatched")
    assert event["log_level"] == "error" and event["kind"] == "optimize"
    assert admission._held is None
    admission.NO_LEASE.release()  # the lease of a create that takes no slot


async def test_a_an_inconsistent_slot_fails_closed_with_a_closed_log() -> None:
    lease = admission.try_acquire("backtest")
    lease.open = False  # a closed lease left in the slot
    with capture_logs() as logs, pytest.raises(admission.AdmissionRefused):
        admission.try_acquire("optimize")
    assert _events(logs, "admission.state_inconsistent")


# --------------------------------------------------------------------------------------------
# T-d: the thread lease (ML, retraining) and leak recovery
# --------------------------------------------------------------------------------------------
async def test_d_a_thread_lease_is_freed_by_its_future_and_not_by_release() -> None:
    lease = admission.try_acquire("ml_train")
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    lease.bind_thread(fut)
    lease.release()  # the request ended; the thread did not
    assert admission._held is lease
    fut.set_result(1)
    await asyncio.sleep(0)
    assert admission._held is None


async def test_d_a_cancelled_bound_future_frees_the_slot() -> None:
    lease = admission.try_acquire("ml_train")
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    lease.bind_thread(fut)
    fut.cancel()
    await asyncio.sleep(0)
    assert admission._held is None


async def _lease_from_a_finished_task(bind: asyncio.Future[Any] | None) -> admission.Lease:
    async def grab() -> admission.Lease:
        lease = admission.try_acquire("ml_train")
        if bind is not None:
            lease.bind_thread(bind)
        return lease

    return await asyncio.create_task(grab())  # the holder task is done when this returns


async def test_d_leak_recovery_for_a_task_lease_and_for_a_thread_lease() -> None:
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    leaked = await _lease_from_a_finished_task(None)  # (i) a done task lease is recovered
    with capture_logs() as logs:
        admission.try_acquire("backtest").release()
    assert _events(logs, "admission.slot_leak_recovered") and not leaked.open
    held = await _lease_from_a_finished_task(fut)  # (ii) request done, thread future not
    with capture_logs() as logs, pytest.raises(admission.AdmissionRefused):
        admission.try_acquire("optimize")
    assert not _events(logs, "admission.slot_leak_recovered") and held.open
    fut.set_result(1)  # (iii) recovered the moment the future is done, before its callback runs
    with capture_logs() as logs:
        fresh = admission.try_acquire("optimize")
    assert _events(logs, "admission.slot_leak_recovered")
    await asyncio.sleep(0)  # the old callback runs now and must leave the new holder alone
    assert admission._held is fresh and fresh.open


async def test_d_a_thread_exception_after_the_request_is_gone_leaves_no_text_in_the_logs() -> None:
    lease = admission.try_acquire("ml_train")
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    lease.bind_thread(fut)
    with capture_logs() as logs:
        fut.set_exception(RuntimeError(_LEAK))
        await asyncio.sleep(0)
    (event,) = _events(logs, "admission.thread_job_failed")
    assert event["error_type"] == "RuntimeError" and _LEAK not in repr(logs)
    assert admission._held is None


async def _train(db: Any) -> Any:
    return await ml_module.train_model(
        symbol="BTC/USDT", exchange="binance", timeframe="1h", bars=200, n_estimators=10,
        horizon=5, threshold=0.01, num_wf_folds=2, db=db,
    )  # fmt: skip


async def test_d_a_cancelled_ml_request_keeps_the_slot_until_its_thread_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, started = threading.Event(), threading.Event()

    def fake_thread(**_: Any) -> Any:
        started.set()
        gate.wait(10)
        raise RuntimeError(_LEAK)  # raised after the request is gone

    monkeypatch.setattr(ml_module, "_train_model_with_wf_sync", fake_thread)
    with capture_logs() as logs:
        request = asyncio.create_task(_train(AsyncMock()))
        await _until(started.is_set)
        assert admission._held is not None and admission._held.kind == "ml_train"
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert admission._held is not None and admission._held.open  # the thread still runs
        with pytest.raises(admission.AdmissionRefused):
            admission.try_acquire("backtest")
        gate.set()
        await _until_free()
    assert _events(logs, "admission.thread_job_failed") and _LEAK not in repr(logs)


async def test_d_an_ml_request_that_ends_normally_frees_the_slot_with_its_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_thread(**_: Any) -> Any:
        raise ValueError("bad window")

    monkeypatch.setattr(ml_module, "_train_model_with_wf_sync", fake_thread)
    with pytest.raises(HTTPException) as err:
        await _train(AsyncMock())
    assert err.value.status_code == 400 and admission._held is None


# --------------------------------------------------------------------------------------------
# T-e: retraining (scheduler skip, manual 429, slot held until the job future is done)
# --------------------------------------------------------------------------------------------
def _service() -> RetrainingService:
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one=MagicMock(return_value=99)))
    ctx = MagicMock()
    ctx.__aenter__.return_value = session
    ctx.__aexit__.return_value = False
    return RetrainingService(db_session_factory=lambda: ctx, min_trades_for_retrain=50)


async def test_e_the_scheduler_skips_when_busy_and_never_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, ran = _service(), []

    async def job(**kw: Any) -> None:
        ran.append(kw)

    monkeypatch.setattr(svc, "_do_retrain", job)
    busy = admission.try_acquire("backtest")  # never released in this test: a wait would hang
    with capture_logs() as logs:
        await asyncio.wait_for(svc._check_and_retrain("BTC/USDT", "1h"), 5)
    (event,) = _events(logs, "admission.skipped_busy")
    assert event["symbol"] == "BTC/USDT" and event["timeframe"] == "1h" and not ran
    assert admission._held is busy


async def test_e_the_scheduled_job_holds_the_slot_until_its_future_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, gate, jobs = _service(), asyncio.Event(), []

    async def job(**_: Any) -> None:
        jobs.append(asyncio.current_task())
        await gate.wait()

    monkeypatch.setattr(svc, "_do_retrain", job)
    loop_task = asyncio.create_task(svc._check_and_retrain("BTC/USDT", "1h"))
    await _until(lambda: bool(jobs))
    assert admission._held is not None and admission._held.kind == "ml_retrain"
    loop_task.cancel()  # the polling loop is cancelled; the job is not
    with pytest.raises(asyncio.CancelledError):
        await loop_task
    assert admission._held is not None and not jobs[0].done()
    gate.set()
    await _until_free()
    assert jobs[0].done()


async def test_e_the_manual_route_is_429_while_busy_and_holds_the_slot_for_its_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, calls = asyncio.Event(), []

    class Service:
        async def manual_retrain(self, symbol: str, timeframe: str) -> None:
            calls.append(symbol)
            await gate.wait()

    monkeypatch.setattr(ml_module, "_retraining_service", Service())
    app = types.SimpleNamespace(state=types.SimpleNamespace())
    request = Request({"type": "http", "app": app})
    busy = admission.try_acquire("optimize")
    with pytest.raises(admission.AdmissionRefused) as refused:
        await ml_module.manual_retrain("BTC/USDT", request, "1h")
    assert refused.value.status_code == 429 and not calls
    busy.release()
    assert (await ml_module.manual_retrain("BTC/USDT", request, "1h"))["status"] == "accepted"
    assert admission._held is not None and admission._held.kind == "ml_retrain"
    await _until(lambda: bool(calls))
    assert admission._held is not None  # the task is still running
    gate.set()
    await _until_free()


# --------------------------------------------------------------------------------------------
# T-f / T-g: hold_long and the pool-total warning
# --------------------------------------------------------------------------------------------
async def test_f_hold_long_is_armed_at_1800_s_and_warns_once_with_closed_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, armed = asyncio.get_running_loop(), []
    real_call_later = loop.call_later

    def spy(delay: float, callback: Callable[[], None]) -> asyncio.TimerHandle:
        armed.append((delay, callback))
        return real_call_later(delay, callback)

    monkeypatch.setattr(loop, "call_later", spy)
    lease = admission.try_acquire("optimize")
    assert armed == [(1800.0, lease._long)] and lease.timer is not None
    timer = lease.timer
    with capture_logs() as logs:
        lease._long()  # what the loop runs when the timer fires
    (event,) = _events(logs, "admission.hold_long")
    assert event == {
        "event": "admission.hold_long",
        "log_level": "warning",
        "kind": "optimize",
        "seconds": 1800.0,
    }
    lease.release()
    assert timer.cancelled()


@pytest.mark.parametrize(
    ("pool", "overflow", "warns"), [(2, 3, True), (10, 20, False), (5, 4, False)]
)
async def test_g_the_pool_total_warning(
    monkeypatch: pytest.MonkeyPatch, pool: int, overflow: int, warns: bool
) -> None:
    settings = types.SimpleNamespace(db_pool_size=pool, db_max_overflow=overflow)
    monkeypatch.setattr("api.config.get_settings", lambda: settings)
    monkeypatch.setattr(admission, "event", MagicMock(listens_for=lambda *_: lambda fn: fn))
    with capture_logs() as logs:
        admission._install_pool_timing()
    found = _events(logs, "admission.pool_total_low")
    assert bool(found) is warns and (not warns or found[0]["total"] == pool + overflow)


def _listeners(monkeypatch: pytest.MonkeyPatch) -> dict[str, Callable[..., None]]:
    """Install the pool timing with a recording ``event`` and return the two listeners."""
    got: dict[str, Callable[..., None]] = {}

    def listens_for(_target: Any, name: str) -> Callable[[Any], Any]:
        def register(fn: Callable[..., None]) -> Callable[..., None]:
            got[name] = fn
            return fn

        return register

    settings = types.SimpleNamespace(db_pool_size=10, db_max_overflow=20)
    monkeypatch.setattr("api.config.get_settings", lambda: settings)
    monkeypatch.setattr(admission, "event", MagicMock(listens_for=listens_for))
    admission._install_pool_timing()
    return got


async def test_r2_a_checkout_counts_as_research_only_while_a_slot_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _listeners(monkeypatch)["checkout"]
    admission._IN_RESEARCH.set(True)  # a stale flag in this context, and no holder
    stale = types.SimpleNamespace(info={})
    checkout(None, stale, None)
    assert stale.info == {}
    lease = admission.try_acquire("ml_retrain")
    held = types.SimpleNamespace(info={})
    checkout(None, held, None)
    assert "research_t0" in held.info
    lease.release()


async def test_r2_the_pool_listeners_never_raise_into_a_checkout_or_checkin(
    m: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = _listeners(monkeypatch)
    m.observe.side_effect = RuntimeError(_LEAK)
    record = types.SimpleNamespace(info={"research_t0": 0.0})
    got["checkin"](None, record)  # the histogram raises inside: swallowed
    assert m.observe.called and record.info == {}

    class Broken:
        @property
        def info(self) -> dict[str, float]:
            raise RuntimeError(_LEAK)

    admission.try_acquire("backtest")
    got["checkout"](None, Broken(), None)
    got["checkin"](None, Broken())


async def test_r2_a_failing_pool_timing_setup_never_blocks_an_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> None:
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(admission, "_pool_ready", False)
    monkeypatch.setattr("api.config.get_settings", boom)
    monkeypatch.setattr(admission, "event", MagicMock())
    with capture_logs() as logs:
        lease = admission.try_acquire("backtest")
    assert admission._held is lease and admission._pool_ready  # not retried on every admission
    (event,) = _events(logs, "admission.pool_timing_unavailable")
    assert event == {"event": "admission.pool_timing_unavailable", "log_level": "error"}
    assert _LEAK not in repr(logs)


# --------------------------------------------------------------------------------------------
# T-k: CF-26-86 (a pool timeout is store code "timeout", counted; the modes keep their behaviour)
# --------------------------------------------------------------------------------------------
async def test_k_a_pool_timeout_is_code_timeout_counted_and_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = MagicMock()
    monkeypatch.setattr(ohlcv_store, "metrics", counter)
    factory = MagicMock(side_effect=PoolTimeoutError(_LEAK))
    store = ohlcv_store.SqlOhlcvStore(factory)
    with capture_logs() as logs, pytest.raises(OhlcvStoreError) as err:
        await store.read(SeriesKey("coinbase", "BTC/USDT", 3600), 0, 1)
    assert (
        err.value.code == "timeout"
        and ohlcv_store.store_error_code(PoolTimeoutError("x")) == "timeout"
    )
    counter.increment.assert_called_once_with("ohlcv_store_pool_timeout_total")
    (event,) = _events(logs, "ohlcv_store_failed")
    assert event["code"] == "timeout" and _LEAK not in repr(logs)
    plain = ohlcv_store.SqlOhlcvStore(MagicMock(side_effect=TimeoutError(_LEAK)))
    counter.reset_mock()
    with pytest.raises(OhlcvStoreError):
        await plain.read(SeriesKey("coinbase", "BTC/USDT", 3600), 0, 1)
    counter.increment.assert_not_called()  # only the pool's own timeout is counted
    wrapped = store._wrap("read", PoolTimeoutError(_LEAK))
    fail_open, strict = Rec(), Rec()
    fail_open.fail["read"] = strict.fail["read"] = [wrapped] * 3
    res = await get(source(Up(), fail_open, log=Log()))
    assert (res.store, res.store_reason) == ("unavailable", "timeout")  # read_through: still served
    with pytest.raises(OhlcvRequireError) as req:
        await get(source(Up(), strict, mode="require"))
    assert req.value.code == "ohlcv_store_unavailable"  # require keeps its closed error


# --------------------------------------------------------------------------------------------
# T-l: CF-26-21 (the fetch-latency histogram observes on every exit of the fetch)
# --------------------------------------------------------------------------------------------
class _Service:
    failure: BaseException | None = None

    def __init__(self, **_: Any) -> None: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def fetch_ohlcv_range(self, **_: Any) -> list[Any]:
        if self.failure is not None:
            raise self.failure
        return []


async def _fetch(now: Any) -> Any:
    start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
    return await runs_module._fetch_bars_for_backtest(
        ["BTC/USDT"], TimeFrame.ONE_HOUR, start, end, MagicMock(), now=now
    )


async def test_l_the_fetch_latency_histogram_observes_on_success_and_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = MagicMock()
    monkeypatch.setattr(runs_module, "metrics", rec)
    monkeypatch.setattr("data.services.ccxt_market_data.CCXTMarketDataService", _Service)
    await _fetch(None)
    name, seconds, labels = rec.observe.call_args.args
    assert (name, labels) == ("history_fetch_seconds", {"mode": "off"}) and seconds >= 0
    monkeypatch.setattr(_Service, "failure", MarketDataError("down"))
    with pytest.raises(HTTPException) as err:
        await _fetch(None)
    assert err.value.status_code == 502 and rec.observe.call_count == 2
    monkeypatch.setenv("OHLCV_CACHE_MODE", "read_through")  # the cache path labels its own mode
    get_settings.cache_clear()
    monkeypatch.setattr("api.services.ohlcv_history.fetch_history_bars", AsyncMock(return_value={}))
    await _fetch(datetime(2024, 1, 3, tzinfo=UTC))
    assert rec.observe.call_args.args[2] == {"mode": "read_through"}
