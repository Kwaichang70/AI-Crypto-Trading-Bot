"""CF-97-03 P7b1: the egress budget on the routes (F2, D5, C2, auto-retry, never-gated, live.*).

Mock sessions and the in-memory claim store, no Postgres.  ``test_smoke_roundtrip_live.py`` is the
source of the live-engine helpers (``_make_exchange``, ``_strategy``, ``_build_and_warm``).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from starlette.requests import Request

import api.routers.ml as ml_module
import api.routers.runs as runs_module
from api.config import get_settings
from api.schemas import RunCreateRequest
from api.services import (
    admission,
)
from api.services import (
    egress_budget as eb,
)
from api.services import (
    kill_switch as _kill_switch,
)
from api.services import (
    run_orchestrator as orch,
)
from common.types import TimeFrame
from data.market_data import RateLimitError
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.idempotency_store import InMemoryIdempotencyStore
from tests.integration.fakes.live_harness import patch_exchange_factory, step_bar
from tests.integration.test_smoke_roundtrip_live import (
    PRICE,
    SYMBOL,
    TIMEFRAME_STR,
    _build_and_warm,
    _make_exchange,
    _strategy,
)

_START, _END = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 5, tzinfo=UTC)
_ADMIN = "a1b2c3d4" * 4
_GOLDEN_502 = (
    "Exchange error fetching historical data: Rate limit exceeded for 'BTC/USDT' after 3 retries"
)


@pytest.fixture(autouse=True)
def _budget(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for k, v in (
        ("EXCHANGE_ID", "kraken"),
        ("REQUIRE_API_AUTH", "false"),
        ("RATE_LIMIT_ENABLED", "false"),
        ("PROMETHEUS_ENABLED", "false"),
        ("ADMIN_API_KEY", _ADMIN),
    ):
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    get_settings.cache_clear()
    eb._reset_for_tests()
    admission._reset_for_tests()
    yield
    eb._reset_for_tests()
    admission._reset_for_tests()
    get_settings.cache_clear()


class _Service:
    """Stands in for CCXTMarketDataService in ``_fetch_bars_for_backtest``; counts calls."""

    calls = 0
    failure: BaseException | None = None

    def __init__(self, **_: Any) -> None: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def fetch_ohlcv_range(self, **_: Any) -> list[Any]:
        _Service.calls += 1
        if _Service.failure is not None:
            raise _Service.failure
        return []


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> type[_Service]:
    _Service.calls, _Service.failure = 0, None
    monkeypatch.setattr("data.services.ccxt_market_data.CCXTMarketDataService", _Service)
    return _Service


async def _fetch(now: datetime | None = None) -> Any:
    return await runs_module._fetch_bars_for_backtest(
        ["BTC/USDT"], TimeFrame.ONE_MINUTE, _START, _END, MagicMock(), now=now
    )


# --------------------------------------------------------------------------------------------
# pin 6 (F2) and pin 9 (D5)
# --------------------------------------------------------------------------------------------
async def test_a_cooldown_refuses_even_a_range_the_cache_could_serve(
    monkeypatch: pytest.MonkeyPatch, service: type[_Service]
) -> None:
    monkeypatch.setenv("OHLCV_CACHE_MODE", "read_through")
    get_settings.cache_clear()
    cached = AsyncMock(return_value={})
    monkeypatch.setattr("api.services.ohlcv_history.fetch_history_bars", cached)
    await _fetch(datetime(2026, 1, 3, tzinfo=UTC))  # no cooldown: the cache path runs
    assert cached.await_count == 1
    eb.suspend_history("kraken")
    with pytest.raises(HTTPException) as busy:
        await _fetch(datetime(2026, 1, 3, tzinfo=UTC))
    assert (busy.value.status_code, busy.value.detail, busy.value.headers) == (
        429,
        {"code": "history_fetch_busy"},
        {"Retry-After": "60"},
    )
    assert (
        cached.await_count == 1 and service.calls == 0
    )  # 0 exchange calls, the cache not consulted


async def test_a_history_429_keeps_its_502_and_then_starts_the_cooldown(
    service: type[_Service],
) -> None:
    service.failure = RateLimitError(
        "Rate limit exceeded for 'BTC/USDT' after 3 retries", retry_after=45
    )
    with pytest.raises(HTTPException) as first:
        await _fetch()
    assert (first.value.status_code, first.value.detail) == (
        502,
        _GOLDEN_502,
    )  # byte-identical to a7c6b9e
    assert abs(eb.history_suspended_until["kraken"] - time.monotonic() - 45.0) < 1.0
    with pytest.raises(HTTPException) as next_one:
        await _fetch()
    assert next_one.value.status_code == 429 and service.calls == 1


async def test_a_real_service_ddos_gives_502_then_429_with_no_further_exchange_call(
    production_logging: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ccxt.async_support as cxa

    calls: list[int] = []

    class Ban(FakeCCXTExchange):
        async def fetch_ohlcv(self, *a: Any, **k: Any) -> Any:
            calls.append(1)
            raise cxa.DDoSProtection("418 banned")

    fake = Ban(exchange_id="kraken")
    fake.register_market("BTC/USDT", base="BTC", quote="USDT")
    monkeypatch.setattr(cxa, "kraken", lambda cfg=None: fake)
    production_logging()
    eb.install_log_processor()

    async def instant(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)
    with pytest.raises(HTTPException) as first:
        await _fetch()
    assert first.value.status_code == 502 and len(calls) == 4  # the frozen retries
    with pytest.raises(HTTPException) as later:
        await _fetch()
    assert (
        later.value.status_code == 429
        and later.value.detail == {"code": "history_fetch_busy"}
        and len(calls) == 4
    )
    assert eb._LIVE_RATE_LIMITED_UNTIL["kraken"] - time.monotonic() > 590.0


async def test_the_ml_route_maps_an_exhausted_history_budget_to_the_closed_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_thread(**_: Any) -> Any:
        raise eb.HistoryBudgetExhausted

    monkeypatch.setattr(ml_module, "_train_model_with_wf_sync", fake_thread)
    with pytest.raises(HTTPException) as err:
        await ml_module.train_model(
            symbol="BTC/USDT",
            exchange="kraken",
            timeframe="1h",
            bars=200,
            n_estimators=10,
            horizon=5,
            threshold=0.01,
            num_wf_folds=2,
            db=AsyncMock(),
        )
    assert (err.value.status_code, err.value.detail) == (
        429,
        {"code": "history_fetch_busy"},
    ) and admission._held is None


# --------------------------------------------------------------------------------------------
# pin 4 (C2): the promote with no stored timeframe, and over the cap
# --------------------------------------------------------------------------------------------
async def _promote(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> tuple[Any, InMemoryIdempotencyStore, uuid.UUID]:
    for k, v in (
        ("ENABLE_LIVE_TRADING", "true"),
        ("EXCHANGE_API_KEY", "k" * 12),
        ("EXCHANGE_API_SECRET", "s" * 12),
        ("LIVE_TRADING_CONFIRM_TOKEN", "t" * 12),
    ):
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    _kill_switch.reset_state_for_tests()
    monkeypatch.setattr(
        "api.services.promotion_gate.evaluate_paper_run_eligibility",
        AsyncMock(return_value=MagicMock(eligible=True)),
    )
    source = MagicMock(id=uuid.uuid4(), run_mode="paper", status="stopped", config=config)
    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=source))
    )
    db.add, db.add_all = MagicMock(), MagicMock()
    store, key = InMemoryIdempotencyStore(stale_after_seconds=240.0), uuid.uuid4()
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(HTTPException) as err:
        await runs_module.promote_to_live(
            run_id=source.id,
            db=db,
            request=request,
            response=Response(),
            store=store,
            x_live_confirm_token="t" * 12,
            idempotency_key=str(key),
        )
    return err.value, store, key


_CONFIG = {
    "strategy_name": "momentum_breakout",
    "strategy_params": {"lookback": 20, "trend_sma_period": 20},
    "bracket_config": {
        "bracket_mode": "fixed",
        "bracket_stop_loss_pct": 0.02,
        "bracket_take_profit_pct": 0.04,
    },
    "symbols": ["BTC/USDT"],
    "initial_capital": "10000",
}


async def test_a_promote_with_no_stored_timeframe_is_refused_and_registers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err, store, key = await _promote(monkeypatch, dict(_CONFIG))
    assert (err.status_code, err.detail) == (409, {"code": "egress_budget_exceeded"}), err.detail
    row = store.row(key)
    assert (
        row is not None and row.status == "failed" and eb._weights == {} and eb._reserved == {}
    )  # claim failed


async def test_a_promote_over_the_cap_is_refused_with_the_claim_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eb._weights["other"] = ("kraken", 19.0)
    err, store, key = await _promote(
        monkeypatch, {**_CONFIG, "timeframe": "1m", "symbols": ["A/USDT", "B/USDT"]}
    )
    assert (err.status_code, err.detail) == (409, {"code": "egress_budget_exceeded"}), err.detail
    assert store.row(key).status == "failed" and eb._reserved == {}  # type: ignore[union-attr]


# --------------------------------------------------------------------------------------------
# pin 2(f): the auto-retry spawns even when the registration fails
# --------------------------------------------------------------------------------------------
async def test_the_auto_retry_still_spawns_when_the_registration_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.db import session as db_session
    from trading.costs import default_cost_profile

    crashed = MagicMock(status="error", entries_latch_reason=None, id=uuid.uuid4(), config={})
    db = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=crashed))
    )
    db.add = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__.return_value, ctx.__aexit__.return_value = db, False
    monkeypatch.setattr(db_session, "get_session_factory", lambda: lambda: ctx)
    ran: list[str] = []

    async def fake_engine(**kw: Any) -> None:
        ran.append(kw["run_id_str"])

    monkeypatch.setattr(orch, "run_paper_engine", fake_engine)
    monkeypatch.setattr(eb, "register", MagicMock(side_effect=RuntimeError("SECRET-TEXT")))
    _kill_switch.reset_state_for_tests()

    async def instant(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)
    before = set(orch._RUN_TASKS)
    with structlog.testing.capture_logs() as logs:
        await orch._auto_retry_paper_run(
            crashed_run_id=str(crashed.id), auto_retry_attempt=0, strategy_cls=object,
            strategy_name="s", strategy_params={}, symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_MINUTE, initial_capital="1", trailing_stop_pct=None,
            bracket_config=None, enable_adaptive_learning=False, auto_apply_learning=False,
            cost_profile=default_cost_profile(),
        )  # fmt: skip
    (new,) = set(orch._RUN_TASKS) - before
    await orch._RUN_TASKS.pop(new)
    assert ran == [new] and [e["event"] for e in logs if e["event"].startswith("egress.")] == [
        "egress.registry_unavailable"
    ]
    assert "SECRET-TEXT" not in repr(logs)


# --------------------------------------------------------------------------------------------
# pin 11 (deterministic M-B3) and pin 7(c)
# --------------------------------------------------------------------------------------------
async def test_the_protection_paths_answer_with_the_budget_exhausted_a_cooldown_and_the_slot_held(
    client_dev_with_db: TestClient,
) -> None:
    _kill_switch.reset_state_for_tests()
    eb._weights["fill"] = ("kraken", 20.0)
    eb.suspend_history("kraken", 600)
    eb._LIVE_RATE_LIMITED_UNTIL["*"] = time.monotonic() + 600
    eb._next_slot["kraken"] = time.monotonic() + 600
    admission.try_acquire("backtest")  # the P7a slot, held by this running test task
    admin = {"X-Admin-Key": _ADMIN}
    calls = [
        ("GET", "/health", {}), ("GET", "/api/v1/health/background", {}),
        ("GET", "/api/v1/emergency/kill-switch", {"headers": admin}),
        ("POST", "/api/v1/emergency/kill-switch", {"headers": admin, "json": {"flatten": False}}),
        ("POST", "/api/v1/emergency/kill-switch/clear",
         {"headers": admin, "json": {"reason": "p7b check"}}),
        ("DELETE", f"/api/v1/runs/{uuid.uuid4()}", {"headers": admin}),
        ("POST", f"/api/v1/runs/{uuid.uuid4()}/resume",
         {"headers": {**admin, "Idempotency-Key": str(uuid.uuid4())}}),
    ]  # fmt: skip
    for method, url, kw in calls:
        t = time.perf_counter()
        resp = client_dev_with_db.request(method, url, **kw)
        assert time.perf_counter() - t < 1.0 and resp.status_code not in (409, 429), (
            url,
            resp.status_code,
        )
    assert eb._weights == {"fill": ("kraken", 20.0)} and admission._held is not None


def _chains() -> list[list[Any]]:
    out = [structlog.get_config()["processors"]]
    out += [
        h.formatter.foreign_pre_chain
        for h in logging.getLogger().handlers
        if hasattr(h.formatter, "foreign_pre_chain")
    ]
    return out


def test_a_real_lifespan_startup_installs_the_processor_once_in_both_chains(
    production_logging: Any,
) -> None:
    from api.main import create_app

    production_logging()
    assert all(c.count(eb.live_429_processor) == 0 for c in _chains())  # absent before startup
    with TestClient(create_app(), raise_server_exceptions=False):
        chains = _chains()
        assert len(chains) >= 2 and all(c.count(eb.live_429_processor) == 1 for c in chains)
        eb.install_log_processor()  # a second install after startup must not duplicate it
        assert all(c.count(eb.live_429_processor) == 1 for c in _chains())
        logging.getLogger("bridge").warning("rate_limit_hit")  # a stdlib record reaches the bridge
    assert "*" in eb._LIVE_RATE_LIMITED_UNTIL


# --------------------------------------------------------------------------------------------
# pin 16: a live engine's order failure reaches the processor (the live.* path)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("exc_name", ["RateLimitExceeded", "DDoSProtection"])
async def test_a_live_order_failure_with_a_rate_limit_error_suspends_history(
    production_logging: Any, monkeypatch: pytest.MonkeyPatch, exc_name: str
) -> None:
    import ccxt.async_support as cxa

    async def instant(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", instant)
    exchange = _make_exchange(free_eur=Decimal("70"))
    patch_exchange_factory(monkeypatch, exchange)
    cap = production_logging()
    eb.install_log_processor()
    stack = await _build_and_warm(
        exchange, _strategy(hold_bars=1, exit_retry_bars=4), run_id="p7b-live-429"
    )
    exchange.queue_order_error(SYMBOL, getattr(cxa, exc_name)("429 SECRET-TEXT"))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # the entry order hits the fault
    line = cap.lines_with("live.order_submission")
    assert line and exc_name in line[0] and "SECRET-TEXT" not in cap.text
    assert (
        eb._LIVE_RATE_LIMITED_UNTIL["coinbase"] - time.monotonic() > 50.0
    )  # the harness exchange id
    await stack.engine.stop()


# --------------------------------------------------------------------------------------------
# CR-P7B-02: the create route's cap, for paper and for live
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["paper", "live"])
async def test_create_run_refuses_over_the_cap_and_reserves_then_consumes_under_it(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    for k, v in (
        ("ENABLE_LIVE_TRADING", "true"),
        ("EXCHANGE_API_KEY", "k" * 12),
        ("EXCHANGE_API_SECRET", "s" * 12),
        ("LIVE_TRADING_CONFIRM_TOKEN", "t" * 12),
    ):
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    _kill_switch.reset_state_for_tests()
    monkeypatch.setattr(runs_module, "_reject_live_context_consumer", MagicMock())
    monkeypatch.setattr("api.services.audit_log.record_audit_event", AsyncMock())
    stop, started = asyncio.Event(), []

    async def engine(**kw: Any) -> None:
        started.append(kw["run_id_str"])
        await stop.wait()

    monkeypatch.setattr(runs_module, "_run_paper_engine", engine)
    monkeypatch.setattr(runs_module, "_run_live_engine", engine)

    def body() -> RunCreateRequest:  # the route strips the bracket keys, so one body per call
        return RunCreateRequest.model_validate(
            {
                "strategyName": "momentum_breakout",
                "strategyParams": {
                    "lookback": 20,
                    "trend_sma_period": 20,
                    **_CONFIG["bracket_config"],
                },
                "symbols": ["A/USDT", "B/USDT"],
                "timeframe": "1m",
                "mode": mode,
                "initialCapital": "10000",
            }
        )

    db, store = AsyncMock(), InMemoryIdempotencyStore(stale_after_seconds=240.0)
    db.add, db.add_all = MagicMock(), MagicMock()
    key = uuid.uuid4()

    async def create() -> Any:
        return await runs_module.create_run(
            body=body(), db=db,
            request=Request({"type": "http", "method": "POST", "path": "/", "headers": []}),
            response=Response(), store=store, x_live_confirm_token="t" * 12,
            idempotency_key=str(key),
        )  # fmt: skip

    eb._weights["other"] = ("kraken", 19.0)  # 19 + 2 > 20
    with pytest.raises(HTTPException) as err:
        await create()
    assert (err.value.status_code, err.value.detail) == (409, {"code": "egress_budget_exceeded"})
    row = store.row(key)
    assert row is not None and row.status == "failed" and eb._reserved == {}
    assert set(eb._weights) == {"other"} and started == []
    eb._weights["other"] = ("kraken", 18.0)  # 18 + 2 <= 20
    key = uuid.uuid4()
    run = await create()
    assert eb._reserved == {} and eb._weights[str(run.id)] == ("kraken", 2.0)  # consumed
    stop.set()
    await asyncio.gather(*runs_module._RUN_TASKS.values(), return_exceptions=True)
    await asyncio.sleep(0)
    runs_module._RUN_TASKS.clear()
    assert started == [str(run.id)] and set(eb._weights) == {"other"}
