"""
apps/api/services/run_recovery.py
-----------------------------------
Live-resume preparation and the orphan-holding alert repeater (WP1.8a,
Verbeterplan v2 synthesis spec §3-§8).

Public API
----------
- :class:`ImportReport` -- the (currently empty) result of a WP1.8b
  exchange scan; the 1.8a stub never returns one.
- :func:`scan_and_import` -- **1.8a fail-closed stub.**  Always raises
  ``ResumeRejected("exchange_scan_not_implemented")``.  WP1.8b replaces
  this with the real order-level prefix scan / cancel-then-import (S2/S3).
  Tests inject a replacement via
  ``monkeypatch.setattr(run_recovery, "scan_and_import", fake)`` to
  exercise the rest of the resume pipeline (the module-level name is
  looked up fresh on every call, so a monkeypatch on this module's global
  takes effect immediately -- see :func:`prepare_live_resume`).
- :func:`prepare_live_resume` -- loads the persisted fill/order history,
  validates it (O10/R-06), invokes :func:`scan_and_import`, then reloads
  and re-validates before returning the ``ResumeSnapshot`` the live engine
  replays from.  Raises ``ResumeRejected`` on any failure (S11: never
  overridable).
- :func:`orphan_holding_repeater` -- background task (registered on
  ``AppContainer.background_tasks``) that repeats a critical alert every
  15 minutes for every live run sitting ``orphaned`` with a non-zero last
  position snapshot (S8, minimum alerting).

Why the stub is still strictly safer than HEAD
-----------------------------------------------
At HEAD, an orphaned live run auto-restarts with an *empty* portfolio and
no exchange reconciliation at all (C7/A-01). In 1.8a, every live resume
attempt in production is rejected with 409 until WP1.8b lands -- fail
closed, not fail open.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import RunORM
from api.services.run_persistence import load_resume_snapshot
from trading.recovery import ResumeRejected, ResumeSnapshot, check_fill_integrity

__all__ = [
    "ORPHAN_REPEAT_INTERVAL_SECONDS",
    "ImportReport",
    "orphan_holding_repeater",
    "prepare_live_resume",
    "scan_and_import",
]

logger = structlog.get_logger(__name__)

#: S8 minimum alerting cadence -- repeat the critical alert every 15
#: minutes while a live run stays orphaned with an unprotected position.
ORPHAN_REPEAT_INTERVAL_SECONDS: float = 900.0


@dataclass(frozen=True)
class ImportReport:
    """Result of a (WP1.8b) exchange scan-and-import pass.

    Empty/zero in 1.8a -- ``scan_and_import`` never returns one in
    production; only an injected test replacement constructs it.
    """

    orders_cancelled: int = 0
    orders_imported: int = 0
    fills_imported: int = 0


async def scan_and_import(
    db: AsyncSession,
    run: RunORM,
    exchange: Any,  # noqa: ANN401
) -> ImportReport:
    """
    Scan the exchange for orders/trades placed under ``run``'s
    ``clientOrderId`` prefix and import anything missing from the DB
    (WP1.8b S2/S3: cancel every open prefixed order, poll to terminal,
    then order-level dedup against the DB by ``client_order_id``).

    **1.8a stub.** Not implemented -- always fails closed.  Every
    production live resume is rejected with 409
    ``exchange_scan_not_implemented`` until WP1.8b lands.  This is a
    deliberate, safer regression from HEAD's silent auto-restart-with-
    empty-portfolio behaviour (C7), not an oversight.

    Tests exercise the rest of the resume pipeline by monkeypatching this
    module's ``scan_and_import`` global before calling
    :func:`prepare_live_resume` (which looks the name up fresh on every
    call, so the patch takes effect without needing a hook or a flag).
    """
    raise ResumeRejected("exchange_scan_not_implemented")


async def prepare_live_resume(
    db: AsyncSession,
    run: RunORM,
    exchange: Any = None,  # noqa: ANN401
) -> ResumeSnapshot:
    """
    Validate and (in 1.8b) reconcile a live run's exchange state, then
    return the ``ResumeSnapshot`` the resume endpoint replays into a fresh
    engine stack.

    Steps (WP18-R-05..07, S2/S3, S11):
    1. Load the persisted fill/order history and validate it (O10/R-06) --
       corrupt or partial history fails closed before any exchange call.
    2. Call :func:`scan_and_import` (the 1.8a stub always raises here).
    3. Reload and re-validate -- an import (1.8b) can only ever add rows,
       so a second, unconditional integrity check is cheap insurance.

    Raises
    ------
    ResumeRejected
        On any validation failure or exchange-scan failure.  Never
        overridable (S11) -- the only caller-visible outcome is a 409 with
        the ``reason`` code, the resume endpoint's own audit row, and a
        status rollback from ``resuming`` back to ``orphaned``.
    """
    symbols = set((run.config or {}).get("symbols") or [])

    snapshot = await load_resume_snapshot(db, run)
    check_fill_integrity(snapshot.fills, snapshot.orders, symbols=symbols)

    await scan_and_import(db, run, exchange)

    snapshot = await load_resume_snapshot(db, run)
    check_fill_integrity(snapshot.fills, snapshot.orders, symbols=symbols)
    return snapshot


async def orphan_holding_repeater(
    interval_seconds: float = ORPHAN_REPEAT_INTERVAL_SECONDS,
) -> None:
    """
    Repeat a critical alert every ``interval_seconds`` while a live run
    sits ``orphaned`` with a non-zero last position snapshot (S8, minimum
    alerting: structured logs + audit rows already fire once when the run
    is first orphaned -- see ``run_orchestrator``/``routers.runs`` -- this
    loop is the *ongoing* reminder for as long as nobody has resumed it).

    Runs as a named ``AppContainer.background_tasks`` entry so
    ``container.shutdown()`` cancels it on API shutdown along with every
    other long-lived task (LS-004).  Telegram delivery and a Grafana panel
    are explicitly out of scope here (S8 gap) -- this is the structured-log
    floor, always on, that a log-shipping/alerting pipeline can page from.
    """
    from api.db.models import PositionSnapshotORM
    from api.db.session import get_session_factory

    log = logger.bind(component="orphan_repeater")
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                factory = get_session_factory()
                async with factory() as db:
                    result = await db.execute(
                        select(RunORM).where(
                            RunORM.status == "orphaned",
                            RunORM.run_mode == "live",
                        )
                    )
                    for run in result.scalars().all():
                        snap_result = await db.execute(
                            select(PositionSnapshotORM).where(PositionSnapshotORM.run_id == run.id)
                        )
                        held_symbols = [
                            snap.symbol for snap in snap_result.scalars().all() if snap.quantity > 0
                        ]
                        if held_symbols:
                            log.critical(
                                "recovery.orphan_holding_unprotected",
                                run_id=str(run.id),
                                symbols=held_symbols,
                            )
            except Exception:
                log.warning("recovery.orphan_repeater_cycle_failed", exc_info=True)
    except asyncio.CancelledError:
        log.info("recovery.orphan_repeater_stopped")
