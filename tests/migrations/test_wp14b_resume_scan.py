"""
tests/migrations/test_wp14b_resume_scan.py
-----------------------------------------------
WP1.4b (idempotent-submit spec) T12: real-Postgres tests for the D13
resume-scan fixes in ``apps.api.services.run_recovery.scan_and_import``.

MUST run against a real PostgreSQL instance -- same convention as
``test_wp18b_scan_and_import.py`` (reads ``MIGRATION_TEST_DATABASE_URL``,
skips when unset).

T12's four cases:
1. a never-placed REJECTED row (no exchange id, missing from the scan) --
   skipped, resume proceeds.
2. an old PENDING_SUBMIT row with no exchange id, missing from the scan,
   older than the settle window -- marked REJECTED (``never_placed``)
   inside the resume's own fenced transaction.
3. a young PENDING_SUBMIT row with no exchange id, missing from the scan
   -- 409 ``unknown_submit_settling`` (still might resolve).
4. an order actually placed on the exchange but never persisted (crash
   between send and persist) -- imported (D12: the exchange is the
   write-ahead log).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
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
        "See test_wp18b_scan_and_import.py's module docstring for scratch-DB setup."
    ),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

SYMBOL = "BTC/USD"
BASE = "BTC"
QUOTE = "USD"
TIMEFRAME_STR = "1h"
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
    started_at: datetime | None = None,
) -> str:
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
        effective_started_at = started_at if started_at is not None else _STARTED_AT
        await conn.execute(
            """
            INSERT INTO runs (id, run_mode, status, config, started_at, created_at, updated_at)
            VALUES ($1, 'live', $2, $3::jsonb, $4, now(), now())
            """,
            run_id, status, json.dumps(config), effective_started_at,
        )
        return fence
    finally:
        await conn.close()


@pytest.fixture()
def pg_env(monkeypatch: pytest.MonkeyPatch) -> str:
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
    # WP1.4b round 2 (S-02a): the targeted lookup's since/until window is
    # computed from a REAL wall-clock created_at -- align the fake's
    # default order-timestamp clock to real time too (its synthetic
    # bar-time epoch, 2026-01-01, would otherwise fall outside a tight
    # +/-1h window built around "now").
    ex.set_now_ms(int(datetime.now(UTC).timestamp() * 1000))
    return ex


async def _load_run(db: Any, run_id: uuid.UUID) -> Any:
    from sqlalchemy import select

    from api.db.models import RunORM

    result = await db.execute(select(RunORM).where(RunORM.id == run_id))
    return result.scalar_one()


async def _seed_unresolved_order(
    db: Any,
    run_id: uuid.UUID,
    client_order_id: str,
    *,
    status: str,
    created_at: datetime,
) -> Any:
    """A persisted row shaped exactly like an unresolved WP1.4b submit:
    no exchange id, no fills."""
    from api.db.models import OrderORM

    order_row = OrderORM(
        id=uuid.uuid4(),
        client_order_id=client_order_id,
        run_id=run_id,
        symbol=SYMBOL,
        side="buy",
        order_type="market",
        quantity=Decimal("0.01"),
        price=None,
        status=status,
        filled_quantity=Decimal("0"),
        average_fill_price=None,
        exchange_order_id=None,
        created_at=created_at,
        updated_at=created_at,
    )
    db.add(order_row)
    await db.flush()
    return order_row


async def _fetch_order_row(db: Any, order_id: uuid.UUID) -> Any:
    from sqlalchemy import select

    from api.db.models import OrderORM

    result = await db.execute(select(OrderORM).where(OrderORM.id == order_id))
    return result.scalar_one()


# ---------------------------------------------------------------------------
# Case 1: a never-placed REJECTED row is skipped, resume proceeds.
# ---------------------------------------------------------------------------


class TestNeverPlacedRejectedRowSkipped:
    async def test_rejected_row_with_no_exchange_id_is_skipped(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        await _seed_unresolved_order(
            db_session, run_id, cid, status="rejected", created_at=_STARTED_AT,
        )

        exchange = _make_exchange()
        report = await scan_and_import(db_session, run, exchange, fence=fence)

        assert report.orders_imported == 0
        assert report.orders_cancelled == 0


# ---------------------------------------------------------------------------
# Case 2: an old pending/canceled row with no id -> marked REJECTED.
# ---------------------------------------------------------------------------


class TestOldUnresolvedRowMarkedNeverPlaced:
    @pytest.mark.parametrize("seed_status", ["pending_submit", "canceled"])
    async def test_old_row_marked_rejected(self, db_session: Any, seed_status: str) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        # WP1.4b round 2 (S-02c): the resume-specific settle window is 900s
        # (_UNKNOWN_SUBMIT_SETTLE_RESUME_S), not the live engine's 120s.
        old_created_at = datetime.now(UTC) - timedelta(seconds=950)
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status=seed_status, created_at=old_created_at,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        report = await scan_and_import(db_session, run, exchange, fence=fence)

        assert report.orders_imported == 0
        assert report.orders_cancelled == 0

        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "rejected"
        assert refreshed.exchange_order_id is None


# ---------------------------------------------------------------------------
# Case 3: a young pending row -> 409 unknown_submit_settling.
# ---------------------------------------------------------------------------


class TestYoungUnresolvedRowSettling:
    async def test_young_row_returns_settling_rejection(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        recent_created_at = datetime.now(UTC) - timedelta(seconds=5)
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status="pending_submit", created_at=recent_created_at,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.args[0] == "unknown_submit_settling"

        # Fail-closed, unresolved -- the row is left exactly as it was.
        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "pending_submit"
        assert refreshed.exchange_order_id is None


# ---------------------------------------------------------------------------
# Case 4: an order placed but never persisted -> imported, exchange_order_id
# backfilled.
# ---------------------------------------------------------------------------


class TestUnpersistedOrderImportedAndBackfilled:
    async def test_unpersisted_order_imported(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        exchange.seed_exchange_order(
            client_order_id=cid, symbol=SYMBOL, side="buy",
            amount=Decimal("0.01"), price=Decimal("50000"),
            status="closed", filled=Decimal("0.01"),
            trades=[{
                "id": "t-unpersisted-1", "price": 50000.0, "amount": 0.01,
                "fee": {"cost": 0.3, "currency": QUOTE}, "takerOrMaker": "taker",
            }],
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert report.orders_imported == 1
        assert report.fills_imported == 1

        from sqlalchemy import select

        from api.db.models import OrderORM

        result = await db_session.execute(
            select(OrderORM).where(OrderORM.run_id == run_id, OrderORM.client_order_id == cid)
        )
        row = result.scalar_one()
        assert row.status == "filled"
        assert row.exchange_order_id is not None

    async def test_exchange_order_id_backfilled_on_update_branch(self, db_session: Any) -> None:
        """D13: when a persisted (unresolved) row IS matched by the scan,
        the update branch backfills its exchange_order_id (previously left
        NULL forever even once matched)."""
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status="pending_submit", created_at=_STARTED_AT,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=cid, symbol=SYMBOL, side="buy",
            amount=Decimal("0.01"), price=Decimal("50000"),
            status="closed", filled=Decimal("0.01"),
            trades=[{
                "id": "t-backfill-1", "price": 50000.0, "amount": 0.01,
                "fee": {"cost": 0.3, "currency": QUOTE}, "takerOrMaker": "taker",
            }],
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)
        assert report.orders_imported == 1

        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "filled"
        assert refreshed.exchange_order_id is not None


# ---------------------------------------------------------------------------
# WP1.4b round 2 (S-02): P12a (listing lag), P12b (pagination truncation),
# P12c (clock skew wider than the primary scan's margin) -- all three must
# now be rescued by the S-02(a) targeted lookup instead of wrongly
# concluding never_placed (P12a/c) or silently completing on a truncated
# page (P12b).
# ---------------------------------------------------------------------------


class TestP12aListingLagRescuedByTargetedLookup:
    async def test_listing_lag_found_by_targeted_lookup(self, db_session: Any) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        old_created_at = datetime.now(UTC) - timedelta(seconds=950)
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status="pending_submit", created_at=old_created_at,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        exchange.seed_exchange_order(
            client_order_id=cid, symbol=SYMBOL, side="buy",
            amount=Decimal("0.01"), price=Decimal("50000"),
            status="closed", filled=Decimal("0.01"),
            trades=[{
                "id": "t-p12a-1", "price": 50000.0, "amount": 0.01,
                "fee": {"cost": 0.3, "currency": QUOTE}, "takerOrMaker": "taker",
            }],
        )
        # Hide the order from BOTH the primary and supplementary scan
        # calls (2 fetch_orders calls for this one symbol/persisted-order
        # combination) -- the S-02(a) targeted lookup (the 3rd call) must
        # still find it.
        exchange.hide_from_listing(SYMBOL, calls=2)

        report = await scan_and_import(db_session, run, exchange, fence=fence)

        assert report.orders_imported == 1
        assert report.fills_imported == 1

        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "filled"
        assert refreshed.exchange_order_id is not None


class TestP12bPaginationTruncationRejected:
    async def test_truncated_page_raises_exchange_scan_truncated(
        self, db_session: Any,
    ) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        exchange = _make_exchange()
        # Shrink the pagination cap so a handful of seeded orders is enough
        # to simulate a truncated page (S-02b) -- Coinbase's real defaults
        # (10 x 1000 = 10,000) would need an unrealistically large fixture.
        exchange.options["paginationCalls"] = 2
        exchange.options["maxEntriesPerRequest"] = 10  # cap == 20
        for i in range(20):
            exchange.seed_exchange_order(
                client_order_id=f"manual-foreign-order-{i}",
                symbol=SYMBOL, side="buy",
                amount=Decimal("0.01"), price=Decimal("50000"),
                status="closed", filled=Decimal("0.01"),
            )

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.args[0] == "exchange_scan_truncated"


class TestP12cClockSkewRescuedByTargetedLookup:
    async def test_six_minute_clock_skew_found_by_targeted_lookup(
        self, db_session: Any,
    ) -> None:
        from api.services.run_recovery import scan_and_import

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        now = datetime.now(UTC)
        # A RECENT run.started_at (not the fixed 2020 _STARTED_AT every
        # other case in this module uses) -- otherwise the primary scan's
        # since-anchor is so far in the past that no realistic clock skew
        # could ever push an order before it.
        fence = await _seed_run(dsn, run_id, started_at=now - timedelta(seconds=950))
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        order_created_at = now - timedelta(seconds=950)
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status="pending_submit", created_at=order_created_at,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        # The exchange's clock reports this order 6 minutes EARLIER than
        # our own persisted created_at -- outside the primary/supplementary
        # scan's 5-minute margin, but comfortably inside the targeted
        # lookup's 1-hour margin.
        skewed_timestamp_ms = int((order_created_at - timedelta(minutes=6)).timestamp() * 1000)
        exchange.seed_exchange_order(
            client_order_id=cid, symbol=SYMBOL, side="buy",
            amount=Decimal("0.01"), price=Decimal("50000"),
            status="closed", filled=Decimal("0.01"),
            timestamp_ms=skewed_timestamp_ms,
            trades=[{
                "id": "t-p12c-1", "price": 50000.0, "amount": 0.01,
                "fee": {"cost": 0.3, "currency": QUOTE}, "takerOrMaker": "taker",
                "timestamp": skewed_timestamp_ms,
            }],
        )

        report = await scan_and_import(db_session, run, exchange, fence=fence)

        assert report.orders_imported == 1
        assert report.fills_imported == 1

        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "filled"
        assert refreshed.exchange_order_id is not None


# ---------------------------------------------------------------------------
# WP1.4b round 3 (S-R2-03): a FAILED targeted cid lookup must fail closed
# (ResumeRejected("exchange_scan_failed")) instead of silently returning
# None -- which the caller previously treated as definitive "not found"
# evidence, wrongly concluding never_placed on nothing more than a
# transient error.
# ---------------------------------------------------------------------------


class TestP12dTargetedLookupFailureFailsClosed:
    async def test_targeted_lookup_exception_rejects_instead_of_never_placed(
        self, db_session: Any,
    ) -> None:
        from api.services.run_recovery import scan_and_import
        from trading.recovery import ResumeRejected

        run_id = uuid.uuid4()
        dsn = _asyncpg_dsn(_MIGRATION_URL)  # type: ignore[arg-type]
        fence = await _seed_run(dsn, run_id)
        run = await _load_run(db_session, run_id)

        cid = f"{run_id}-{uuid.uuid4().hex[:12]}"
        old_created_at = datetime.now(UTC) - timedelta(seconds=950)
        order_row = await _seed_unresolved_order(
            db_session, run_id, cid, status="pending_submit", created_at=old_created_at,
        )
        order_id = order_row.id
        await db_session.commit()

        exchange = _make_exchange()
        # No matching order exists anywhere -- the primary scan (and its
        # WP18b-S-06 supplementary call, if any) genuinely finds nothing,
        # exactly like a real never_placed case. Only the S-02(a) targeted
        # lookup -- the NEXT fetch_orders(symbol=...) call after those --
        # is made to fail transiently.
        # Mirrors TestP12aListingLagRescuedByTargetedLookup's own comment:
        # 2 fetch_orders calls happen for this one symbol/persisted-order
        # combination (primary scan + the WP18b-S-06 supplementary scan,
        # since this order's created_at is recent enough that since_2_ms >
        # since_ms) before the S-02(a) targeted lookup -- the 3rd call.
        exchange.queue_fetch_orders_error(
            SYMBOL, RuntimeError("transient network error"), times=1, after=2,
        )

        with pytest.raises(ResumeRejected) as exc_info:
            await scan_and_import(db_session, run, exchange, fence=fence)
        assert exc_info.value.args[0] == "exchange_scan_failed"

        # Fail-closed, unresolved -- the row must NOT have been wrongly
        # marked never_placed on nothing more than a transient error.
        refreshed = await _fetch_order_row(db_session, order_id)
        assert refreshed.status == "pending_submit"
        assert refreshed.exchange_order_id is None
