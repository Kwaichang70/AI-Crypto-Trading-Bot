"""
apps/api/services/retention.py
------------------------------
WP2.2 P0 (SY-22-10, QD-03): data-retention policy for ``equity_snapshots``.

Policy (``cutoff = now - retention_days``):

- ``paper`` / ``live`` runs: delete a snapshot when
  ``equity_snapshots.timestamp < cutoff`` (unchanged from the pre-WP2.2
  inline prune in ``main.py``).
- ``backtest`` runs: delete ALL of a run's snapshots, and only when
  ``runs.created_at < cutoff``. A backtest's snapshot timestamps are
  *simulated* market time (often years old); pruning on them would delete
  the equity curve of a run created today. ``equity_snapshots`` has no
  insertion-time column, so run creation time is the retention clock.
- any other ``run_mode``: NEVER deleted (fail-safe). Every statement matches
  one explicit allow-listed mode; there is no "everything else" branch.

SQL shape: one batched DELETE per mode (``paper``, ``live``, ``backtest``),
one transaction per batch (precedent:
``api.services.idempotency.prune_expired_idempotency_keys``), so a large
backlog never holds one long-lived lock or builds one huge WAL burst, and
the per-mode counts can be logged separately.

Victim selection is driven per run through an index, never by a scan of
``equity_snapshots`` (WP22-P0V2-03; measured with ``EXPLAIN (ANALYZE)`` on a
2.2 M row table, see the producer report)::

    DELETE FROM equity_snapshots WHERE id = ANY (ARRAY(
      SELECT v.id FROM runs JOIN LATERAL (
        SELECT es.id FROM equity_snapshots es
        WHERE es.run_id = runs.id [AND es.timestamp < :cutoff]
        ORDER BY es.timestamp | es.bar_index LIMIT :batch) v ON true
      WHERE runs.run_mode = :mode [AND runs.created_at < :cutoff]
      LIMIT :batch))

- paper/live: ordered ``Index Scan`` on ``ix_equity_snapshots_run_id_timestamp``
  (run_id, timestamp), stopped early by the LIMIT.
- backtest: ordered ``Index Scan`` on ``uq_equity_snapshots_run_bar``
  (run_id, bar_index), stopped early by the LIMIT.
- ``equity_snapshots_pkey``: the outer ``id = ANY (ARRAY(...))``. A plain
  ``id IN (subquery)`` lets the planner flip to ``Hash Semi Join`` + ``Seq Scan``
  over the whole table once VACUUM has shrunk ``reltuples`` (measured 196-260 ms
  per batch at ~1 M rows); the planner cannot know the length of an
  ``ARRAY(subquery)``, so at production table sizes (measured from ~300 k rows)
  the planner probes the primary key. On a small table it may still choose a
  ``Seq Scan``; the victim set is the same either way. The lock order is
  deterministic per plan: the ``ScalarArrayOp`` index scan sorts and
  de-duplicates the array, so concurrent pruners on the pkey plan lock rows in
  ascending ``id`` order (a ``Seq Scan`` plan locks in physical order, and
  pruners with the same statistics choose the same plan).
- ``runs`` is tiny relative to ``equity_snapshots``; its mode / created_at
  filter is a scan of ``runs`` only.

No new index / migration is needed. Left alone, the planner would pick a
Bitmap Heap Scan (+ top-N sort) per run, which rebuilds the whole run's
bitmap on every batch and cannot skip dead index entries left by earlier
batches / days (no LP_DEAD hints), so each batch pins
``SET LOCAL enable_bitmapscan = off`` (transaction-scoped; same ``SET LOCAL``
precedent as ``api.services.idempotency``). ``enable_seqscan = off`` is NOT
used: its inflated cost estimate trips JIT compilation on every tiny batch.

Logging (CF-36): one ``equity_prune.completed`` event on success, or one
``equity_prune.partial`` event (totals so far) if a batch raises. The counts in
``partial`` are a lower bound ("at least" this many rows were removed): a batch
is added only after ``commit()`` returns, so a failure or cancellation that
lands after the server committed but before the client saw the reply leaves
that batch uncounted (off by at most one batch, ``DEFAULT_PRUNE_BATCH`` rows).
Nothing deletes extra rows. Only integers, mode names and an ISO timestamp are
logged; never exception text, run ids, or the DSN.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Final

import structlog
from sqlalchemy import BigInteger, Delete, Select, any_, delete, func, select, text, true
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.db.models import EquitySnapshotORM, RunORM

__all__ = [
    "DEFAULT_PRUNE_BATCH",
    "build_backtest_prune_stmt",
    "build_timestamp_prune_stmt",
    "prune_equity_snapshots",
]

logger = structlog.get_logger(__name__)

#: Rows deleted per transaction.
DEFAULT_PRUNE_BATCH: Final[int] = 5000

#: Modes pruned by snapshot timestamp (positive allow-list).
_TIMESTAMP_MODES: Final[tuple[str, ...]] = ("paper", "live")
#: Mode pruned by run creation time (positive allow-list).
_BACKTEST_MODE: Final[str] = "backtest"

#: Keeps victim selection on an ordered index scan (see module docstring).
#: Transaction-scoped: reverts on commit / rollback.
_PIN_ORDERED_INDEX_SCAN: Final = text("SET LOCAL enable_bitmapscan = off")


def _delete_ids_from(victims: Select[tuple[int]]) -> Delete:
    """``DELETE ... WHERE id = ANY (ARRAY(<victims>))`` (see module docstring)."""
    ids = func.array(victims.scalar_subquery(), type_=ARRAY(BigInteger))
    return delete(EquitySnapshotORM).where(EquitySnapshotORM.id == any_(ids))


def build_timestamp_prune_stmt(cutoff: datetime, batch: int, mode: str) -> Delete:
    """paper/live: up to ``batch`` snapshots of ``mode`` runs with ``timestamp < cutoff``.

    ``mode`` must be in the paper/live allow-list; anything else raises
    ``ValueError`` before a statement is built.
    """
    if mode not in _TIMESTAMP_MODES:
        raise ValueError(f"mode must be one of {_TIMESTAMP_MODES}")
    per_run = (
        select(EquitySnapshotORM.id)
        .where(
            EquitySnapshotORM.run_id == RunORM.id,
            EquitySnapshotORM.timestamp < cutoff,
        )
        .order_by(EquitySnapshotORM.timestamp)
        .correlate(RunORM)
        .limit(batch)
        .lateral("v")
    )
    victims = (
        select(per_run.c.id)
        .select_from(RunORM)
        .join(per_run, true())
        .where(RunORM.run_mode == mode)
        .limit(batch)
    )
    return _delete_ids_from(victims)


def build_backtest_prune_stmt(cutoff: datetime, batch: int) -> Delete:
    """backtest: up to ``batch`` snapshots of runs with ``created_at < cutoff``.

    Deliberately has NO predicate on ``equity_snapshots.timestamp``.
    """
    per_run = (
        select(EquitySnapshotORM.id)
        .where(EquitySnapshotORM.run_id == RunORM.id)
        .order_by(EquitySnapshotORM.bar_index)
        .correlate(RunORM)
        .limit(batch)
        .lateral("v")
    )
    victims = (
        select(per_run.c.id)
        .select_from(RunORM)
        .join(per_run, true())
        .where(
            RunORM.run_mode == _BACKTEST_MODE,
            RunORM.created_at < cutoff,
        )
        .limit(batch)
    )
    return _delete_ids_from(victims)


async def _run_batched(
    session_factory: async_sessionmaker[AsyncSession],
    cutoff: datetime,
    batch: int,
    build: Callable[[datetime, int], Delete],
    counts: dict[str, int],
    key: str,
) -> None:
    """Delete in batches until a short batch; add each committed batch to ``counts[key]``.

    ``counts`` is updated after every commit so a later failure can still
    report what was removed (``equity_prune.partial``). The figure is a lower
    bound: a batch committed server-side whose reply was lost is not counted
    (off by at most one batch).
    """
    while True:
        async with session_factory() as session:
            await session.execute(_PIN_ORDERED_INDEX_SCAN)
            result = await session.execute(build(cutoff, batch))
            # DBAPI rowcount may be None or -1 ("unknown"): never go negative.
            deleted = max(result.rowcount or 0, 0)  # type: ignore[attr-defined]
            await session.commit()
        counts[key] += deleted
        if deleted < batch:
            return


async def prune_equity_snapshots(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    retention_days: int,
    now: datetime | None = None,
    batch: int = DEFAULT_PRUNE_BATCH,
) -> int:
    """Apply the SY-22-10 equity-snapshot retention policy.

    Returns the total number of ``equity_snapshots`` rows deleted.
    ``now`` (tz-aware) is injectable for tests; defaults to current UTC.

    Emits ``equity_prune.completed`` (per-mode counts) when anything was
    deleted. If a batch raises, emits ``equity_prune.partial`` with the
    totals committed so far (at least that many rows were removed; off by at
    most one batch) and re-raises.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now - timedelta(days=retention_days)

    counts: dict[str, int] = {"paper": 0, "live": 0, _BACKTEST_MODE: 0}
    stage = ""
    try:
        for mode in _TIMESTAMP_MODES:
            stage = mode
            await _run_batched(
                session_factory,
                cutoff,
                batch,
                partial(build_timestamp_prune_stmt, mode=mode),
                counts,
                mode,
            )
        stage = _BACKTEST_MODE
        await _run_batched(
            session_factory, cutoff, batch, build_backtest_prune_stmt, counts, _BACKTEST_MODE
        )
    except BaseException:  # incl. CancelledError: report, then re-raise untouched
        logger.warning(
            "equity_prune.partial",
            failed_in=stage,
            deleted=sum(counts.values()),
            deleted_paper=counts["paper"],
            deleted_live=counts["live"],
            deleted_backtest=counts[_BACKTEST_MODE],
            cutoff=str(cutoff),
            retention_days=retention_days,
        )
        raise

    total = sum(counts.values())
    if total:
        logger.info(
            "equity_prune.completed",
            deleted=total,
            deleted_paper=counts["paper"],
            deleted_live=counts["live"],
            deleted_backtest=counts[_BACKTEST_MODE],
            cutoff=str(cutoff),
            retention_days=retention_days,
        )
    return total
