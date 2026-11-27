"""CF-97-03 P7a (T-b, T-c, T-h, T-i): the slot on the real routes, with mocks and no Postgres.

``create_run`` and ``run_optimization`` are called directly (the ``test_cf9702_cancel_mid_hop.py``
scaffolding: a mock session and the in-memory idempotency store); T-c, T-h and T-i also go
through the app over HTTP.  Wherever a job ends, the slot must be free afterwards.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from starlette.requests import Request
from structlog.testing import capture_logs

import api.routers.ml as ml_module
import api.routers.optimize as optimize_module
import api.routers.runs as runs_module
from api.config import get_settings
from api.main import create_app
from api.routers.optimize import OptimizeRequest
from api.schemas import RunCreateRequest
from api.services import admission
from api.services import kill_switch as _kill_switch
from api.services import run_persistence as rp
from data.bar_alignment import BAR_ALIGNMENT_CODES, BarAlignmentError
from tests._cf2697_fingerprint import make_bars
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore
from trading import backtest as bt

_N, _CHUNK, _LEAK = 300, 40, "CF9703-SECRET-TEXT"
_WINDOW = {"backtestStart": "2024-01-01T00:00:00Z", "backtestEnd": "2024-01-01T05:00:00Z"}
_BODY = {
    "strategyName": "ma_crossover", "strategyParams": {"fast_period": 5, "slow_period": 20},
    "symbols": ["BTC/USDT"], "timeframe": "1m", "mode": "backtest", "initialCapital": "10000",
    **_WINDOW,
}  # fmt: skip
_OPT = {
    "strategyName": "ma_crossover", "paramGrid": {"fast_period": [5, 6], "slow_period": [20]},
    "symbols": ["BTC/USDT"], "timeframe": "1m", "initialCapital": "10000", "topN": 5,
    "maxCombinations": 10, **_WINDOW,
}  # fmt: skip
_BARS = {"BTC/USDT": make_bars("BTC/USDT", _N, 7)}
_ALIGN = BarAlignmentError(BAR_ALIGNMENT_CODES[0])
_CLOSED = {"event", "log_level", "kind", "request_id", "timestamp"}


@pytest.fixture(autouse=True)
def _slot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    get_settings.cache_clear()
    admission._reset_for_tests()
    yield
    admission._reset_for_tests()
    get_settings.cache_clear()


class _Job:
    """One create_run or run_optimization call against a mock session; every hook is optional."""

    def __init__(self, mp: pytest.MonkeyPatch, kind: str, fetch: Any = None) -> None:
        self.kind, self.ctr, self.on = kind, {"s2": 0, "persist": 0, "flush": 0, "commit": 0}, {}
        self.db, self.store = AsyncMock(), InMemoryIdempotencyStore(stale_after_seconds=240.0)
        self.db.add, self.db.add_all = MagicMock(), MagicMock()
        self.db.flush = AsyncMock(side_effect=partial(self._tick, "flush"))
        self.db.commit = AsyncMock(side_effect=partial(self._tick, "commit"))
        self.store.fail = AsyncMock(wraps=self.store.fail)
        self.on: dict[str, Callable[[int], Any]] = {}
        self.fetch = fetch or AsyncMock(return_value=_BARS)
        mp.setattr(rp, "_PERSIST_CHUNK_ROWS", _CHUNK)
        mp.setattr(runs_module, "_fetch_bars_for_backtest", self.fetch)
        mp.setattr(optimize_module, "_fetch_bars_for_backtest", self.fetch)
        mp.setattr(bt, "_yield_to_loop", partial(self._tick, "s2"))
        mp.setattr(rp, "_yield_to_loop", partial(self._tick, "persist"))

    async def _tick(self, key: str) -> None:
        self.ctr[key] += 1
        if key in self.on:
            await self.on[key](self.ctr[key])

    async def go(self, key: str | None = None) -> Any:
        _kill_switch.reset_state_for_tests()
        if self.kind == "optimize":
            body = OptimizeRequest.model_validate(_OPT)
            return await optimize_module.run_optimization(body=body, db=self.db)
        return await runs_module.create_run(
            body=RunCreateRequest.model_validate(_BODY), db=self.db,
            request=Request({"type": "http", "method": "POST", "path": "/", "headers": []}),
            response=Response(), store=self.store, idempotency_key=key or str(uuid.uuid4()),
        )  # fmt: skip


def _raising(at: int, exc: BaseException) -> Callable[[int], Any]:
    async def hook(n: int) -> None:
        if n == at:
            raise exc

    return hook


async def _outcome(job: _Job) -> BaseException | None:
    try:
        await job.go()
    except BaseException as exc:
        return exc
    return None


@pytest.fixture(params=["create", "optimize"])
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --------------------------------------------------------------------------------------------
# T-b: the slot is released on every exit path, for create and for optimize
# --------------------------------------------------------------------------------------------
async def test_b_the_slot_is_held_mid_job_and_free_after_success(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    job, seen = _Job(monkeypatch, kind), []

    async def probe(n: int) -> None:
        if n == 1:
            seen.append(admission._held.kind if admission._held else None)
            with pytest.raises(admission.AdmissionRefused):  # a second research job is refused
                admission.try_acquire("backtest")

    job.on["s2"] = probe
    assert await _outcome(job) is None
    assert seen == ["backtest" if kind == "create" else "optimize"]
    assert admission._held is None
    assert job.ctr["s2"] >= 4 and job.ctr["flush"] >= 1 and job.ctr["commit"] >= 1


_FAILURES: list[tuple[str, BaseException]] = [
    ("http", HTTPException(502, "exchange")), ("alignment", _ALIGN),
    ("value", ValueError(_LEAK)), ("generic", RuntimeError(_LEAK)),
    ("cancel", asyncio.CancelledError()),
]  # fmt: skip


@pytest.mark.parametrize(("name", "exc"), _FAILURES, ids=[f[0] for f in _FAILURES])
async def test_b_the_slot_is_free_after_a_failure_at_the_fetch_and_at_the_runner(
    monkeypatch: pytest.MonkeyPatch, kind: str, name: str, exc: BaseException
) -> None:
    for where in ("fetch", "runner"):
        job = _Job(monkeypatch, kind, AsyncMock(side_effect=exc) if where == "fetch" else None)
        if where == "runner":
            job.on["s2"] = _raising(2, exc)
        caught = await _outcome(job)
        assert admission._held is None, (name, where)  # whatever the outcome, the slot is free
        assert where == "runner" or caught is not None  # a failing fetch always ends the job
        if kind == "create" and caught is not None:  # the claim is failed once, never completed
            assert job.store.fail.await_count == 1 and job.db.commit.await_count == 0


async def test_b_a_cancel_at_every_suspension_point_frees_the_slot(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    control = _Job(monkeypatch, kind)
    assert await _outcome(control) is None
    steps = ("s2", "flush", "commit") if kind == "optimize" else ("s2", "persist", "flush")
    cancelled = 0
    for key in steps:
        for k in range(1, control.ctr[key] + 1):
            job = _Job(monkeypatch, kind)
            job.on[key] = _raising(k, asyncio.CancelledError())
            caught = await _outcome(job)
            assert isinstance(caught, asyncio.CancelledError), (key, k)
            assert admission._held is None, (key, k)
            cancelled += 1
    job = _Job(monkeypatch, kind, AsyncMock(side_effect=asyncio.CancelledError()))
    assert isinstance(await _outcome(job), asyncio.CancelledError) and admission._held is None
    assert cancelled >= 10  # the loop did run over the suspension points


async def test_b_a_real_task_cancel_while_holding_the_slot(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    job, reached = _Job(monkeypatch, kind), asyncio.Event()

    async def park(n: int) -> None:
        if n == 2:
            reached.set()
            await asyncio.sleep(3600)  # parked until the cancel (a drain cancel by uvicorn)

    job.on["s2"] = park
    task = asyncio.create_task(job.go())
    await reached.wait()
    assert admission._held is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert admission._held is None
    admission.try_acquire("backtest").release()  # the next job is admitted


# --------------------------------------------------------------------------------------------
# T-a (replay) and T-c: refusal leftovers, over HTTP and on the mock route
# --------------------------------------------------------------------------------------------
async def test_a_a_replay_takes_no_slot_and_is_never_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, key = _Job(monkeypatch, "create"), str(uuid.uuid4())
    await job.go(key)
    row = job.store.row(uuid.UUID(key))
    assert row is not None and row.status == "completed"
    job.store.mark_run_exists(row.run_id)  # type: ignore[arg-type]
    job.db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=MagicMock()))
    )
    monkeypatch.setattr(runs_module, "_run_orm_to_detail_response", lambda _run: "REPLAYED")
    busy = admission.try_acquire("optimize")
    assert await job.go(key) == "REPLAYED"
    assert admission._held is busy and busy.open  # nothing taken, nothing freed, no refusal


def _post(client: TestClient, key: str) -> Any:
    return client.post("/api/v1/runs", json=_BODY, headers={"Idempotency-Key": key})


async def test_c_a_refusal_leaves_nothing_and_a_same_key_retry_reclaims(
    monkeypatch: pytest.MonkeyPatch,
    client_dev_with_db: TestClient,
    mock_db_session: AsyncMock,
    idempotency_store: InMemoryIdempotencyStore,
) -> None:
    fetch = AsyncMock(return_value=_BARS)
    monkeypatch.setattr(runs_module, "_fetch_bars_for_backtest", fetch)
    key = str(uuid.uuid4())

    holder = admission.try_acquire("optimize")  # held by this (running) test task
    resp = _post(client_dev_with_db, key)
    assert resp.status_code == 429 and resp.json() == {"detail": {"code": "admission_busy"}}
    assert resp.headers["Retry-After"] == "30"
    row = idempotency_store.row(uuid.UUID(key))
    assert row is not None and row.status == "failed" and row.run_id is None  # claim failed
    assert mock_db_session.add.call_count == 0 and mock_db_session.add_all.call_count == 0
    assert mock_db_session.flush.await_count == 0 and mock_db_session.commit.await_count == 0
    assert fetch.await_count == 0  # no exchange call, no history row
    holder._free()
    retry = _post(client_dev_with_db, key)
    assert retry.status_code == 201 and fetch.await_count == 1
    row = idempotency_store.row(uuid.UUID(key))
    assert row is not None and row.status == "completed" and admission._held is None


# --------------------------------------------------------------------------------------------
# T-h: rejected requests (401, rate-limited 429) never touch admission state
# --------------------------------------------------------------------------------------------
def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = admission.try_acquire

    def spy(k: Any) -> Any:
        calls.append(k)
        return real(k)

    monkeypatch.setattr(admission, "try_acquire", spy)
    return calls


def test_h_an_unauthenticated_request_never_reaches_the_slot(
    monkeypatch: pytest.MonkeyPatch, client_prod: TestClient
) -> None:
    calls = _spy(monkeypatch)
    for url, body in (("/api/v1/runs", _BODY), ("/api/v1/optimize", _OPT)):
        assert client_prod.post(url, json=body).status_code == 401, url
    assert client_prod.post("/api/v1/ml/train?symbol=BTC/USDT").status_code == 401
    assert client_prod.post("/api/v1/ml/retrain/BTC/USDT").status_code == 401
    assert calls == [] and admission._held is None


def test_h_a_rate_limited_request_never_reaches_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_WRITE", "2/minute")
    get_settings.cache_clear()
    calls = _spy(monkeypatch)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        codes = [client.post("/api/v1/optimize", json={}).status_code for _ in range(3)]
    assert codes == [422, 422, 429] and calls == [] and admission._held is None


def test_h_the_spy_does_see_an_admitted_request(
    monkeypatch: pytest.MonkeyPatch, client_dev: TestClient
) -> None:
    calls = _spy(monkeypatch)

    def fake_thread(**_: Any) -> Any:
        raise ValueError("bad window")

    monkeypatch.setattr(ml_module, "_train_model_with_wf_sync", fake_thread)
    assert client_dev.post("/api/v1/ml/train?symbol=BTC/USDT&bars=200").status_code == 400
    assert calls == ["ml_train"] and admission._held is None


# --------------------------------------------------------------------------------------------
# T-i: disclosure (the refusal is the closed dict; nothing of the holder, in the body or logs)
# --------------------------------------------------------------------------------------------
async def test_i_a_refusal_discloses_nothing_of_the_holder(
    monkeypatch: pytest.MonkeyPatch, client_dev: TestClient
) -> None:
    holder = admission.try_acquire("backtest")
    holder.kind = _LEAK  # type: ignore[assignment]  # sentinel holder data
    requests = (
        ("/api/v1/optimize", _OPT), ("/api/v1/ml/train?symbol=BTC/USDT", None),
        ("/api/v1/ml/retrain/BTC/USDT", None),
    )  # fmt: skip
    monkeypatch.setattr(ml_module, "_retraining_service", MagicMock())
    with capture_logs() as logs:
        for url, body in requests:
            resp = client_dev.post(url, json=body) if body else client_dev.post(url)
            assert resp.status_code == 429, url
            assert resp.json() == {"detail": {"code": "admission_busy"}}, url
            assert resp.headers["Retry-After"] == "30" and _LEAK not in resp.text
    assert not [e for e in logs if e["event"] == "admission.refused" and set(e) - _CLOSED]
    assert _LEAK not in repr(logs)
    holder._free()


def test_i_the_refusal_class_carries_only_the_closed_body() -> None:
    exc = admission.AdmissionRefused()
    assert isinstance(exc, HTTPException) and exc.detail == {"code": "admission_busy"}
    assert exc.headers == {"Retry-After": "30"}
