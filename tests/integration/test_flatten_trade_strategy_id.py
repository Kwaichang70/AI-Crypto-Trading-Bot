"""
tests/integration/test_flatten_trade_strategy_id.py
--------------------------------------------------
CF-DOC-02 -- pins what ``GET /runs/{id}/trades`` reports for a round trip
that a ``smoke_roundtrip`` LIVE run opened and an operator flatten closed.

Traced behaviour (no engine change; this test only pins it):

* ``StrategyEngine.flatten`` emits ``Signal(strategy_id="operator_flatten",
  metadata={"exit_reason": "flatten", ...})`` (strategy_engine.py, flatten).
* ``_route_exit_fills`` -> ``_record_trade_if_closed(strategy_id=
  signal.strategy_id)`` stamps the trade with the CLOSING signal's
  ``strategy_id``. The opening signal's id is not stored on the trade.
* ``ExitReasonDetector.detect`` does not know ``"flatten"``, so the stored
  ``exit_reason`` is ``signal_exit``.
* ``TradeResponse`` has no ``exit_reason`` field: the API does not expose
  it. The stored value is pinned at the ORM layer instead.

Chain driven end to end, hermetic (no PostgreSQL): real ``StrategyEngine`` +
``LiveExecutionEngine`` + ``PortfolioAccounting`` (``live_harness``) ->
``stop_run(flatten=True)`` / ``emergency_stop_run(flatten=True)`` ->
``persist_paper_results`` (real ``TradeORM`` construction, capturing fake
session) -> ``list_trades`` -> ``TradeListResponse``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from api.db.models import RunORM, TradeORM
from api.routers.portfolio import list_trades
from api.routers.runs import emergency_stop_run, stop_run
from api.schemas import TradeListResponse
from api.services.run_persistence import persist_paper_results
from common.types import TimeFrame
from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange
from tests.integration.fakes.live_harness import (
    LiveStack,
    build_live_stack,
    patch_exchange_factory,
    start_and_warmup,
    step_bar,
)
from trading.strategies.smoke_roundtrip import SmokeRoundtripStrategy

SYMBOL = "XRP/EUR"
BASE = "XRP"
QUOTE = "EUR"
TIMEFRAME = TimeFrame.FIVE_MINUTES
TIMEFRAME_STR = "5m"
PRICE = Decimal("0.50")
CAPITAL = Decimal("65")
OPENING_STRATEGY_ID = "smoke_roundtrip-cfdoc02"


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(delay: float = 0, result: object = None) -> object:
        return result

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


def _make_exchange() -> FakeCCXTExchange:
    ex = FakeCCXTExchange(taker_fee_pct=Decimal("0.012"))
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE, min_cost="1", min_amount="0.01")
    ex.set_balance(QUOTE, Decimal("70"))
    ex.seed_flat_bars(SYMBOL, count=100, price=PRICE, timeframe=TIMEFRAME_STR)
    return ex


async def _holding_stack(
    monkeypatch: pytest.MonkeyPatch, run_id: uuid.UUID, *, hold_bars: int
) -> tuple[LiveStack, FakeCCXTExchange]:
    """Drive a live smoke run to HOLDING (entry BUY filled, exit not yet due)."""
    exchange = _make_exchange()
    patch_exchange_factory(monkeypatch, exchange)
    strategy = SmokeRoundtripStrategy(
        strategy_id=OPENING_STRATEGY_ID,
        params={"notional_quote": 9.0, "hold_bars": hold_bars, "exit_retry_bars": 4},
    )
    stack = await build_live_stack(
        exchange=exchange,
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=CAPITAL,
        run_id=str(run_id),
        engine_config={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
    )
    await start_and_warmup(stack, str(run_id))
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # entry BUY
    position = stack.portfolio.get_position(SYMBOL)
    assert position is not None and not position.is_flat, "precondition: HOLDING"
    assert stack.portfolio.get_trade_history() == [], "precondition: no closed trade yet"
    return stack, exchange


def _run_row(run_id: uuid.UUID) -> MagicMock:
    run = MagicMock(spec=RunORM)
    run.id = run_id
    run.run_mode = "live"
    run.status = "running"
    run.config = {"strategy_name": "smoke_roundtrip"}
    run.entries_latch_reason = None
    run.entries_latched_at = None
    run.started_at = datetime.now(tz=UTC)
    run.stopped_at = None
    run.created_at = datetime.now(tz=UTC)
    run.updated_at = datetime.now(tz=UTC)
    run.n_closed_trades = None
    run.metrics_v2_backfilled = False
    run.recovered_from_run_id = None
    run.promoted_from_run_id = None
    return run


def _request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


def _db_returning(run: MagicMock) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = run
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    return db


async def _persist_and_list(
    stack: LiveStack, run_id: uuid.UUID
) -> tuple[list[TradeORM], TradeListResponse]:
    """persist_paper_results (real TradeORM build) -> list_trades (real mapper)."""
    captured: list[Any] = []
    session = AsyncMock()
    session.add_all = MagicMock(side_effect=lambda rows: captured.extend(rows))
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    @asynccontextmanager
    async def _session_cm() -> Any:
        yield session

    with patch("api.db.session.get_session_factory", return_value=_session_cm):
        await persist_paper_results(
            run_id_str=str(run_id),
            portfolio=stack.portfolio,
            execution_engine=stack.execution,
            log=MagicMock(),
        )
    trade_orms = [r for r in captured if isinstance(r, TradeORM)]

    run_result = MagicMock()
    run_result.scalar_one_or_none.return_value = _run_row(run_id)
    count_result = MagicMock()
    count_result.scalar_one.return_value = len(trade_orms)
    page_result = MagicMock()
    page_result.scalars.return_value.all.return_value = trade_orms
    list_db = AsyncMock()
    list_db.execute = AsyncMock(side_effect=[run_result, count_result, page_result])
    response = await list_trades(run_id, list_db)
    return trade_orms, response


def _assert_flatten_trade(
    trade_orms: list[TradeORM],
    response: TradeListResponse,
    cap: Sequence[Mapping[str, Any]],
) -> None:
    assert len(trade_orms) == 1
    # API-visible value: the CLOSING (flatten) signal's id -- NOT the opening
    # signal's strategy id, despite the schema wording before CF-DOC-02.
    assert response.total == 1
    item = response.items[0]
    assert item.strategy_id == "operator_flatten"
    assert item.strategy_id != OPENING_STRATEGY_ID
    assert item.symbol == SYMBOL
    assert item.side == "buy"  # opening side

    dumped = item.model_dump(by_alias=True)
    assert dumped["strategyId"] == "operator_flatten"
    # The API does not expose exit_reason at all.
    assert "exitReason" not in dumped and "exit_reason" not in dumped

    # Stored exit reason: "flatten" is not a valid stored reason, so the
    # detector falls through to signal_exit (matches the runbook's note).
    assert trade_orms[0].strategy_id == "operator_flatten"
    assert trade_orms[0].exit_reason == "signal_exit"
    recorded = [e for e in cap if e.get("event") == "engine.trade_recorded"]
    assert len(recorded) == 1 and recorded[0]["exit_reason"] == "signal_exit"


async def test_stop_run_flatten_trade_strategy_id_is_operator_flatten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid.uuid4()
    stack, exchange = await _holding_stack(monkeypatch, run_id, hold_bars=6)
    run = _run_row(run_id)
    db = _db_returning(run)
    task = MagicMock()
    task.done.return_value = False

    with capture_logs() as cap:
        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run_id): stack.engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run_id): task}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            response = await stop_run(run_id, db, _request(), flatten=True)

    assert response.flatten is not None and response.flatten.outcome == "flattened"
    assert run.status == "stopped"
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]

    trade_orms, listing = await _persist_and_list(stack, run_id)
    _assert_flatten_trade(trade_orms, listing, cap)
    await stack.engine.stop()


async def test_emergency_stop_flatten_trade_strategy_id_is_operator_flatten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid.uuid4()
    stack, exchange = await _holding_stack(monkeypatch, run_id, hold_bars=6)
    run = _run_row(run_id)
    db = _db_returning(run)
    task = MagicMock()
    task.done.return_value = False

    with capture_logs() as cap:
        with (
            patch("api.routers.runs._RUN_ENGINES", {str(run_id): stack.engine}),
            patch("api.routers.runs._RUN_TASKS", {str(run_id): task}),
            patch("api.services.audit_log.record_audit_event", new=AsyncMock()),
        ):
            await emergency_stop_run(run_id, _request(), db, reason="cf-doc-02", flatten=True)

    assert run.status == "stopped"
    assert [o["side"] for o in exchange.order_log] == ["buy", "sell"]

    trade_orms, listing = await _persist_and_list(stack, run_id)
    _assert_flatten_trade(trade_orms, listing, cap)
    await stack.engine.stop()


async def test_strategy_own_exit_keeps_strategy_id_for_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: when the strategy itself closes, the trade carries the
    strategy's id (which is also the opening id) -- so 'strategy_id' is the
    closing signal's id in general, and only coincides with the opening one
    when the same strategy closes."""
    run_id = uuid.uuid4()
    stack, _exchange = await _holding_stack(monkeypatch, run_id, hold_bars=1)
    await step_bar(stack, SYMBOL, PRICE, timeframe=TIMEFRAME_STR)  # strategy SELL

    trade_orms, listing = await _persist_and_list(stack, run_id)
    assert len(trade_orms) == 1
    assert listing.items[0].strategy_id == OPENING_STRATEGY_ID
    assert trade_orms[0].exit_reason == "signal_exit"
    await stack.engine.stop()
