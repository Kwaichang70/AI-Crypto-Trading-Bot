"""
tests/migrations/test_wp18b_scan_and_import.py
--------------------------------------------------
Real-Postgres tests for WP1.8b's ``run_recovery.scan_and_import`` (S2/S3):
the real order-level ``clientOrderId``-prefix exchange scan, cancel-then-
import, and its integrity branches.

MUST run against a real PostgreSQL instance (not SQLite/mocked) -- the
DB-side dedup ("re-running a resume never imports twice") and the
persisted-order-vs-exchange-scan comparisons are exactly the kind of
thing a fake session can accidentally make trivially true. Reads
``MIGRATION_TEST_DATABASE_URL`` and SKIPS when unset, same convention as
``test_017_orphaned_status.py`` / ``test_wp18a_resume_races.py``.

Setup performed OUTSIDE this test (documented in the producer report):
    pg_ctlcluster 16 main start
    createuser wp18b_test --pwprompt
    createdb wp18b_migration_test -O wp18b_test
    export MIGRATION_TEST_DATABASE_URL=postgresql+asyncpg://wp18b_test:...@localhost:5432/wp18b_migration_test

Mandatory scenarios (synthesis spec §6, 1.8b):
- an unpersisted BUY is imported, then the SL fires, with the same run_id
  (``TestUnpersistedBuyImportedThenStopLossFires``)
- an open order is cancelled and its partial fill imported
  (``TestOpenOrderCancelledPartialFillImported``)
- a foreign order is left untouched (``TestForeignOrderUntouched``)
- scan / cancel / trade-fetch failure -> 409 (``TestScanCancelTradeFetchFailures``)
- the partial, corrupt and incomplete branches of S3 (``TestS3IntegrityBranches``)
- a re-run of the resume never imports twice (``TestReRunNeverImportsTwice``)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

_MIGRATION_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _MIGRATION_URL,
    reason=(
        "MIGRATION_TEST_DATABASE_URL not set -- these tests need a real "
        "Postgres instance and are skipped in environments without one. "
        "See the module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

SYMBOL = "BTC/USD"
BASE = "BTC"
QUOTE = "USD"
TIMEFRAME_STR = "1h"
# Well before FakeCCXTExchange's synthetic bar epoch (2026-01-01T00:00:00Z,
# _EPOCH_START_MS = 1_767_225_600_000) so `since=started_at` never
# excludes an order the fake generates during a test.
_STARTED_AT = datetime(2020, 1, 1, tzinfo=UTC)


def _alembic_config(database_url: str) -> Any:
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url
    from api.config import get_settings

    get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "infra" / "alembic" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO_ROOT / "infra" / "alembic"))
    return cfg


def _asyncpg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


async def _seed_run(
    dsn: str,
    run_id: uuid.UUID,
    *,
    symbols: list[str] | None = None,
    status: str = "resuming",
) -> str:
    """Seed a run row and return the ``resume_attempt_id`` fence embedded
    in its config (WP18b-S-01) -- pass this same value as
    ``scan_and_import``/``prepare_live_resume``'s ``fence=`` kwarg."""
    import asyncpg

    fence = str(uuid.uuid4())
    conn = await asyncpg.connect(dsn)
    try:
        config = {
            "strategy_name": "grid_trading",
            "symbols": symbols or [SYMBOL],
            "timeframe": TIMEFRAME_STR,
            "initial_capital": "10000",
            "strategy_params": {},
            "resume_attempt_id": fence,
        }
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', $2, $3::jsonb, $4, now(), now())
            """,
            run_id,
            status,
            json.dumps(config),
            _STARTED_AT,
        )
        return fence
    finally:
        await conn.close()


@pytest.fixture()
def pg_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the app's settings at the scratch DB and run every migration."""
    assert _MIGRATION_URL is not None  # guarded by pytestmark skipif
    monkeypatch.setenv("DATABASE_URL", _MIGRATION_URL)
    from api.config import get_settings

    get_settings.cache_clear()

    import api.db.session as session_module

    session_module._engine = None
    session_module._session_factory = None

    from alembic import command

    cfg = _alembic_config(_MIGRATION_URL)
    command.upgrade(cfg, "head")

    return _MIGRATION_URL


@pytest.fixture()
async def db_session(pg_env: str) -> Any:
    from api.db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.rollback()


def _make_exchange() -> Any:
    from tests.integration.fakes.fake_ccxt_exchange import FakeCCXTExchange

    ex = FakeCCXTExchange(exchange_id="coinbase")
    ex.register_market(SYMBOL, base=BASE, quote=QUOTE)
    ex.seed_flat_bars(SYMBOL, count=5, price=Decimal("50000"), timeframe=TIMEFRAME_STR)
    return ex


async def _load_run(db: Any, run_id: uuid.UUID) -> Any:
    from sqlalchemy import select

    from api.db.models import RunORM

    result = await db.execute(select(RunORM).where(RunORM.id == run_id))
    run = result.scalar_one()
    return run


async def _persisted_orders_and_fills(db: Any, run_id: uuid.UUID) -> tuple[list[Any], list[Any]]:
    from sqlalchemy import select

    from api.db.models import FillORM, OrderORM

    order_result = await db.execute(select(OrderORM).where(OrderORM.run_id == run_id))
    order_rows = list(order_result.scalars().all())
    order_ids = [o.id for o in order_rows]
    fill_rows: list[Any] = []
    if order_ids:
        fill_result = await db.execute(select(FillORM).where(FillORM.order_id.in_(order_ids)))
        fill_rows = list(fill_result.scalars().all())
    return order_rows, fill_rows


# ---------------------------------------------------------------------------
# An unpersisted BUY is imported, then the SL fires, with the same run_id.
# ---------------------------------------------------------------------------


class TestUnpersistedBuyImportedThenStopLossFires:
    async def test_scan_import_then_stop_loss_fires(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from tests.integration.fakes.live_harness import (
            build_live_stack,
            build_resumed_live_stack,
            patch_exchange_factory,
            start_and_warmup,
            step_bar,
        )
        from tests.integration.fakes.scripted_strategy import ScriptedSignalStrategy
        from trading.risk import RiskParameters

        async def _instant_sleep(delay: float = 0, result: object = None) -> object:
            return result

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        run_id = uuid.uuid4()
        run_id_str = str(run_id)
        exchange = _make_exchange()
        exchange.set_balance(QUOTE, Decimal("1000"))
        patch_exchange_factory(monkeypatch, exchange)

        from common.types import TimeFrame

        # 1. A "before the crash" engine places a real BUY on the exchange
        #    -- never persisted anywhere (no DB write in this harness).
        strategy_before = ScriptedSignalStrategy(
            "buy-before-crash", {"direction": "buy", "call_index": 0, "target_notional": "100"}
        )
        stack_before = await build_live_stack(
            exchange=exchange,
            strategy=strategy_before,
            symbol=SYMBOL,
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("1000"),
            run_id=run_id_str,
            engine_config={"bracket_stop_loss_pct": 0.05},
            risk_params=RiskParameters(),
        )
        await start_and_warmup(stack_before, run_id_str)
        await step_bar(stack_before, SYMBOL, Decimal("50000"), timeframe=TIMEFRAME_STR)

        buy_orders = [o for o in exchange.order_log if o["side"] == "buy"]
        assert len(buy_orders) == 1
        bought_qty = buy_orders[0]["amount"]
        assert exchange.balance_of(BASE) == bought_qty
        # "crash" -- stack_before is abandoned, never persisted, never stopped.

        # 2. Seed the DB row the resume path would already have (status
        #    doesn't matter to scan_and_import itself -- it only reads
        #    run.config/run.started_at/run.id).
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        from api.services.run_recovery import prepare_live_resume

        snapshot = await prepare_live_resume(db_session, run, exchange, fence=fence)
        assert len(snapshot.fills) == 1
        assert snapshot.fills[0].quantity == bought_qty

        order_rows, fill_rows = await _persisted_orders_and_fills(db_session, run_id)
        assert len(order_rows) == 1
        assert len(fill_rows) == 1

        # 3. Resume a SECOND engine stack under the SAME run_id, portfolio
        #    rebuilt from the IMPORTED fills (not from stack_before's
        #    in-memory history) -- proves the DB import round-trip feeds
        #    a real resumed engine correctly.
        strategy_after = ScriptedSignalStrategy(
            "buy-after-resume-noop", {"direction": "buy", "call_index": 999}
        )
        stack_after = await build_resumed_live_stack(
            exchange=exchange,
            strategy=strategy_after,
            symbol=SYMBOL,
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("1000"),
            run_id=run_id_str,
            fills=snapshot.fills,
            engine_config={"bracket_stop_loss_pct": 0.05},
        )
        await start_and_warmup(stack_after, run_id_str)

        position = stack_after.portfolio.get_position(SYMBOL)
        assert position is not None and not position.is_flat

        breach_price = Decimal("50000") * Decimal("0.80")
        await step_bar(stack_after, SYMBOL, breach_price, timeframe=TIMEFRAME_STR)

        sell_orders = [o for o in exchange.order_log if o["side"] == "sell"]
        assert sell_orders, "expected the stop-loss to fire against the imported entry"
        assert sell_orders[-1]["amount"] == bought_qty

        await stack_after.engine.stop()


# ---------------------------------------------------------------------------
# An open order is cancelled and its partial fill imported.
# ---------------------------------------------------------------------------


class TestOpenOrderCancelledPartialFillImported:
    async def test_open_order_cancelled_and_partial_fill_imported(self, db_session: Any) -> None:
        from api.services.run_recovery import ImportReport, scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        client_order_id = f"{run_id}-{uuid.uuid4().hex[:12]}"
        exchange_order_id = exchange.seed_exchange_order(
            client_order_id=client_order_id,
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.02"),
            price=Decimal("50000"),
            status="open",
            filled=Decimal("0.008"),
            trades=[
                {
                    "id": "t-open-1",
                    "price": 50000.0,
                    "amount": 0.008,
                    "fee": {"cost": 0.24, "currency": QUOTE},
                    "takerOrMaker": "taker",
                }
            ],
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert isinstance(report, ImportReport)
        assert report.orders_cancelled == 1
        assert report.orders_imported == 1
        assert report.fills_imported == 1

        order_rows, fill_rows = await _persisted_orders_and_fills(db_session, run_id)
        assert len(order_rows) == 1
        assert order_rows[0].status == "canceled"
        assert order_rows[0].filled_quantity == Decimal("0.008")
        assert len(fill_rows) == 1
        assert fill_rows[0].quantity == Decimal("0.008")

        # The exchange-side order really was cancelled (not left open).
        final = await exchange.fetch_order(exchange_order_id, SYMBOL)
        assert final["status"] == "canceled"


# ---------------------------------------------------------------------------
# A foreign order is left untouched.
# ---------------------------------------------------------------------------


class TestForeignOrderUntouched:
    async def test_foreign_order_untouched(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        foreign_id = exchange.seed_exchange_order(
            client_order_id="some-other-run-deadbeef",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="open",
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert report.foreign_orders_seen == 1
        assert report.orders_cancelled == 0
        assert report.orders_imported == 0

        untouched = await exchange.fetch_order(foreign_id, SYMBOL)
        assert untouched["status"] == "open"

        order_rows, _fill_rows = await _persisted_orders_and_fills(db_session, run_id)
        assert order_rows == []


# ---------------------------------------------------------------------------
# scan / cancel / trade-fetch failure -> 409 (ResumeRejected).
# ---------------------------------------------------------------------------


class TestScanCancelTradeFetchFailures:
    async def test_fetch_orders_failure_rejected(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange.queue_fetch_orders_error(SYMBOL, RuntimeError("boom"))

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "exchange_scan_failed"

    async def test_cancel_exception_falls_through_to_poll_then_succeeds(
        self, db_session: Any
    ) -> None:
        """WP18b-C-03: a cancel_order exception (any type, not just
        OrderNotFound/InvalidOrder) no longer fails the resume immediately
        -- it falls through to the terminal-state poll, which is the real
        source of truth. Here the order is ALREADY terminal (closed) by
        the time the poll's first fetch_order call runs, so the resume
        succeeds despite the cancel-side exception."""
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange_order_id = exchange.seed_exchange_order(
            client_order_id=f"{run_id}-{uuid.uuid4().hex[:12]}",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="open",
        )
        exchange.queue_cancel_error(exchange_order_id, RuntimeError("cancel boom"))

        # The order is ALREADY closed on the exchange by the time cancel
        # was attempted (a legitimate race) -- the poll's own fetch_order
        # is what must discover this, not the cancel exception.
        exchange._orders[exchange_order_id]["status"] = "closed"
        exchange._orders[exchange_order_id]["filled"] = 0.01
        exchange._orders[exchange_order_id]["average"] = 50000.0
        exchange._trades.append(
            {
                "id": "cancel-race-trade",
                "order": exchange_order_id,
                "symbol": SYMBOL,
                "side": "buy",
                "amount": 0.01,
                "price": 50000.0,
                "fee": {"cost": 0.3, "currency": QUOTE},
                "takerOrMaker": "taker",
                "timestamp": exchange._now_ms(SYMBOL),
            }
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert report.orders_imported == 1
        assert report.fills_imported == 1

    async def test_cancel_exception_then_poll_failure_rejected(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The poll's OWN fetch_order failure (not the cancel exception
        itself) is what still fails closed with 'order_cancel_failed'."""
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange_order_id = exchange.seed_exchange_order(
            client_order_id=f"{run_id}-{uuid.uuid4().hex[:12]}",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="open",
        )
        exchange.queue_cancel_error(exchange_order_id, RuntimeError("cancel boom"))

        async def _raising_fetch_order(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("fetch_order boom")

        monkeypatch.setattr(exchange, "fetch_order", _raising_fetch_order)

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "order_cancel_failed"

    async def test_trade_fetch_failure_rejected(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=f"{run_id}-{uuid.uuid4().hex[:12]}",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="closed",
            filled=Decimal("0.01"),
            trades=[
                {
                    "id": "t1",
                    "price": 50000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.3, "currency": QUOTE},
                }
            ],
        )
        exchange.queue_trades_error(SYMBOL, RuntimeError("trades boom"))

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "trade_fetch_failed"


    async def test_cancel_poll_timeout_rejected(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled order that never reaches a terminal ccxt status
        within the poll window fails closed too (distinct from an
        outright cancel_order exception)."""
        import api.services.run_recovery as run_recovery_module
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        monkeypatch.setattr(run_recovery_module, "CANCEL_POLL_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(run_recovery_module, "CANCEL_POLL_INTERVAL_SECONDS", 0.01)

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=f"{run_id}-{uuid.uuid4().hex[:12]}",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="open",
        )

        # A cancel that "succeeds" per the exchange's own bookkeeping but
        # never actually flips the order's status (a real-world exchange
        # inconsistency) -- the poll loop must time out instead of
        # spinning forever.
        async def _flaky_cancel(
            order_id: str, symbol: str | None = None, params: Any = None
        ) -> Any:
            return {"id": order_id, "status": "open"}

        monkeypatch.setattr(exchange, "cancel_order", _flaky_cancel)

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "order_cancel_timeout"


# ---------------------------------------------------------------------------
# The partial, corrupt and incomplete branches of S3.
# ---------------------------------------------------------------------------


class TestS3IntegrityBranches:
    async def _seed_persisted_order(
        self, db_session: Any, run_id: uuid.UUID, client_order_id: str, *, filled_quantity: Decimal
    ) -> Any:
        from api.db.models import OrderORM

        order_row = OrderORM(
            id=uuid.uuid4(),
            client_order_id=client_order_id,
            run_id=run_id,
            symbol=SYMBOL,
            side="buy",
            order_type="market",
            quantity=Decimal("0.02"),
            price=None,
            status="filled",
            filled_quantity=filled_quantity,
            average_fill_price=Decimal("50000"),
            exchange_order_id=None,
            created_at=_STARTED_AT,
            updated_at=_STARTED_AT,
        )
        db_session.add(order_row)
        await db_session.flush()
        return order_row

    async def _seed_persisted_fill(
        self, db_session: Any, order_row: Any, *, quantity: Decimal
    ) -> None:
        from api.db.models import FillORM

        db_session.add(
            FillORM(
                id=uuid.uuid4(),
                order_id=order_row.id,
                symbol=SYMBOL,
                side="buy",
                quantity=quantity,
                price=Decimal("50000"),
                fee=Decimal("0.3"),
                fee_currency=QUOTE,
                is_maker=False,
                executed_at=_STARTED_AT,
            )
        )
        await db_session.flush()

    async def test_fill_history_partial_rejected(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        client_order_id = f"{run_id}-{uuid.uuid4().hex[:12]}"
        order_row = await self._seed_persisted_order(
            db_session, run_id, client_order_id, filled_quantity=Decimal("0.01")
        )
        # Only HALF of the order's own filled_quantity was ever persisted.
        await self._seed_persisted_fill(db_session, order_row, quantity=Decimal("0.005"))

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=client_order_id,
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.02"),
            price=Decimal("50000"),
            status="closed",
            filled=Decimal("0.01"),
            trades=[
                {
                    "id": "pt1",
                    "price": 50000.0,
                    "amount": 0.005,
                    "fee": {"cost": 0.15, "currency": QUOTE},
                },
                {
                    "id": "pt2",
                    "price": 50000.0,
                    "amount": 0.005,
                    "fee": {"cost": 0.15, "currency": QUOTE},
                },
            ],
        )

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "fill_history_partial"

    async def test_fill_history_corrupt_rejected(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        client_order_id = f"{run_id}-{uuid.uuid4().hex[:12]}"
        # Persisted MORE than the exchange itself ever reported as filled.
        order_row = await self._seed_persisted_order(
            db_session, run_id, client_order_id, filled_quantity=Decimal("0.01")
        )
        await self._seed_persisted_fill(db_session, order_row, quantity=Decimal("0.05"))

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=client_order_id,
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.02"),
            price=Decimal("50000"),
            status="closed",
            filled=Decimal("0.01"),
            trades=[
                {
                    "id": "ct1",
                    "price": 50000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.3, "currency": QUOTE},
                }
            ],
        )

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "fill_history_corrupt"

    async def test_exchange_scan_incomplete_rejected(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        client_order_id = f"{run_id}-{uuid.uuid4().hex[:12]}"
        await self._seed_persisted_order(
            db_session, run_id, client_order_id, filled_quantity=Decimal("0.01")
        )

        # The exchange scan finds NOTHING for this run -- the persisted
        # order has vanished from its history entirely.
        exchange = _make_exchange()

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.reason == "exchange_scan_incomplete"


# ---------------------------------------------------------------------------
# A re-run of the resume never imports twice.
# ---------------------------------------------------------------------------


class TestReRunNeverImportsTwice:
    async def test_rerun_does_not_duplicate_import(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=f"{run_id}-{uuid.uuid4().hex[:12]}",
            symbol=SYMBOL,
            side="buy",
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            status="closed",
            filled=Decimal("0.01"),
            trades=[
                {
                    "id": "rt1",
                    "price": 50000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.3, "currency": QUOTE},
                }
            ],
        )

        first_report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert first_report.orders_imported == 1
        assert first_report.fills_imported == 1

        order_rows_1, fill_rows_1 = await _persisted_orders_and_fills(db_session, run_id)
        assert len(order_rows_1) == 1
        assert len(fill_rows_1) == 1

        second_report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert second_report.orders_imported == 0
        assert second_report.fills_imported == 0

        order_rows_2, fill_rows_2 = await _persisted_orders_and_fills(db_session, run_id)
        assert len(order_rows_2) == 1
        assert len(fill_rows_2) == 1
        assert fill_rows_2[0].id == fill_rows_1[0].id
