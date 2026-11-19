"""CF-97-02 (T-9702-03): a cancel or an error inside the sliced post-loop hop leaves nothing behind.

``create_run`` is called directly (the ``test_wp21_cost_api.py`` scaffolding: a mock session and the
in-memory idempotency store, no Postgres).  A ``CancelledError`` raised at every yield of
``BacktestRunner.run`` (the S2 yields) and of ``persist_backtest_results``, and at every flush after
the run row's, must end in the ``except BaseException`` branch: rollback, claim failed, no commit,
nothing committed.  An ordinary error at a flush takes HEAD's 500 path.  One case delivers a
real ``task.cancel()``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import api.routers.runs as runs_module
from api.config import get_settings
from api.schemas import RunCreateRequest
from api.services import kill_switch as _kill_switch
from api.services import run_persistence as rp
from tests._cf2697_fingerprint import make_bars
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore
from trading import backtest as bt
from trading.backtest import BacktestRunner

_INIT = BacktestRunner.__init__
_N, _CHUNK, _SECRET = 300, 40, "CF9702-SECRET-TEXT"
_BODY = {
    "strategyName": "ma_crossover", "strategyParams": {"fast_period": 5, "slow_period": 20},
    "symbols": ["BTC/USDT"], "timeframe": "1m", "mode": "backtest", "initialCapital": "10000",
    "backtestStart": "2024-01-01T00:00:00Z", "backtestEnd": "2024-01-01T05:00:00Z",
}  # fmt: skip


class _Run:
    """One create_run call against a mock session; every hook is optional."""

    def __init__(self, mp: pytest.MonkeyPatch) -> None:
        self.mp, self.ctr = mp, {"s2": 0, "persist": 0, "flush": 0}
        self.db, self.store = AsyncMock(), InMemoryIdempotencyStore(stale_after_seconds=240.0)
        self.db.add, self.db.add_all = MagicMock(), MagicMock()
        self.db.flush = AsyncMock(side_effect=partial(self._tick, "flush"))
        self.store.fail = AsyncMock(wraps=self.store.fail)
        self.runner: list[BacktestRunner] = []
        self.on: dict[str, Callable[[int], Any]] = {}
        mp.setattr(rp, "_PERSIST_CHUNK_ROWS", _CHUNK)
        mp.setattr(
            runs_module,
            "_fetch_bars_for_backtest",
            AsyncMock(return_value={"BTC/USDT": make_bars("BTC/USDT", _N, 7)}),
        )

        def init(runner: BacktestRunner, *a: Any, **kw: Any) -> None:
            self.runner.append(runner)
            _INIT(runner, *a, **kw)

        mp.setattr(BacktestRunner, "__init__", init)
        mp.setattr(bt, "_yield_to_loop", partial(self._tick, "s2"))
        mp.setattr(rp, "_yield_to_loop", partial(self._tick, "persist"))

    async def _tick(self, key: str) -> None:
        self.ctr[key] += 1
        if key in self.on:
            await self.on[key](self.ctr[key])

    async def go(self) -> Any:
        _kill_switch.reset_state_for_tests()
        return await runs_module.create_run(
            body=RunCreateRequest.model_validate(_BODY), db=self.db,
            request=Request({"type": "http", "method": "POST", "path": "/", "headers": []}),
            response=Response(), store=self.store, idempotency_key=str(uuid.uuid4()),
        )  # fmt: skip

    def left_nothing(self) -> list[str]:
        db, fail = self.db, self.store.fail
        checks = {"a commit": db.commit.await_count > 0, "no rollback": not db.rollback.await_count,
                  "claim not failed once": fail.await_count != 1}  # fmt: skip
        return [name for name, bad in checks.items() if bad]


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    get_settings.cache_clear()


async def _counts(mp: pytest.MonkeyPatch) -> dict[str, int]:
    control = _Run(mp)
    resp = await control.go()
    assert resp.status == "stopped" and control.db.commit.await_count == 1
    assert not control.store.fail.await_count and control.runner[0].last_portfolio is not None
    assert control.ctr["s2"] == 4 and control.ctr["persist"] > 4 and control.ctr["flush"] > 6
    return control.ctr


def _raising(at: int, exc: BaseException) -> Callable[[int], Any]:
    async def hook(n: int) -> None:
        if n == at:
            raise exc

    return hook


async def _cancel_at(mp: pytest.MonkeyPatch, key: str, k: int) -> _Run:
    run = _Run(mp)
    run.on[key] = _raising(k, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await run.go()
    return run


async def test_a_cancel_at_every_yield_and_flush_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctr = await _counts(monkeypatch)
    for key in ("s2", "persist"):
        for k in range(1, ctr[key] + 1):
            run = await _cancel_at(monkeypatch, key, k)
            assert run.left_nothing() == [], (key, k)
            assert (run.runner[0].last_portfolio is None) == (key == "s2"), (key, k)
    for k in range(2, ctr["flush"] + 1):  # flush 1 is the run row, before the backtest
        assert (await _cancel_at(monkeypatch, "flush", k)).left_nothing() == [], k


async def test_an_error_at_any_flush_takes_the_500_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ctr = await _counts(monkeypatch)
    for k in range(2, ctr["flush"] + 1):
        run = _Run(monkeypatch)
        run.on["flush"] = _raising(k, RuntimeError(_SECRET))
        with pytest.raises(HTTPException) as caught:
            await run.go()
        assert caught.value.status_code == 500 and _SECRET not in str(caught.value.detail)
        assert run.left_nothing() == [], k


async def test_a_real_task_cancel_inside_a_chunk_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(monkeypatch)

    async def cancel_self(n: int) -> None:
        if n == 3:
            asyncio.current_task().cancel()  # type: ignore[union-attr]
            await asyncio.sleep(0)

    run.on["persist"] = cancel_self
    task = asyncio.create_task(run.go())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.left_nothing() == [] and run.ctr["persist"] == 3
