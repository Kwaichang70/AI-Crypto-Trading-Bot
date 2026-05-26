#!/usr/bin/env python3
"""
scripts/backfill_metrics_v2.py
-------------------------------
One-shot operational backfill for Sprint 49 metric fields on legacy backtest runs.

Targets rows where ``metrics_v2_backfilled = false AND run_mode = 'backtest'`` and
writes back:
  - ``n_closed_trades``          (RunORM top-level SQL column)
  - ``profit_factor``            (JSONB config.backtest_metrics -- M1 None/inf semantics)
  - ``profit_factor_is_infinite``(JSONB config.backtest_metrics -- M1 flag)
  - ``psr``                      (JSONB config.backtest_metrics -- recomputed from equity)
  - ``confidence_flag``          (JSONB config.backtest_metrics -- M4)
  - ``quote_currency``           (JSONB config.backtest_metrics -- M6)

What this script CANNOT recompute from stored data (documented skip):
  - ``exposure_pct_per_symbol``  -- needs per-bar position tracking (not stored)
  - ``open_positions_mtm``       -- needs terminal-bar prices (not stored)
  - ``seed``                     -- legacy runs never logged seed

Usage
-----
  # Dry-run: prints per-run diff for first 10 rows then aggregate stats
  python scripts/backfill_metrics_v2.py

  # Write changes in batches of 100 (idempotent -- safe to re-run)
  python scripts/backfill_metrics_v2.py --apply

  # Target a single run (for verification before bulk apply)
  python scripts/backfill_metrics_v2.py --run-id <UUID> --apply

  # Override default batch size (default 100)
  python scripts/backfill_metrics_v2.py --apply --batch-size 50

  # Limit how many runs are processed in this execution
  python scripts/backfill_metrics_v2.py --apply --limit 20

Exit codes
----------
  0 -- success (dry-run or apply both succeed)
  1 -- fatal error (DB connection failure, import error)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Ensure workspace packages are importable when run directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg_dir in (_REPO_ROOT / "packages", _REPO_ROOT / "apps"):
    if str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

import structlog
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from api.db import RunORM, EquitySnapshotORM, get_session_factory
from trading.backtest import _detect_quote_currency  # noqa: PLC2701
from trading.metrics import (
    _derive_confidence_flag,  # noqa: PLC2701 -- private helper, cross-module per M4 pattern
    compute_psr,
    compute_sharpe,
    compute_returns_from_equity,
    EquityCurvePoint,
    TIMEFRAME_PERIODS_PER_YEAR,
)
from common.types import TimeFrame

logger = structlog.get_logger("backfill_metrics_v2")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PREVIEW_ROWS = 10  # Number of per-run diff rows to print in dry-run mode


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill_metrics_v2",
        description=(
            "Backfill Sprint 49 metric fields on legacy backtest runs. "
            "Default mode is dry-run; use --apply to write."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write changes to DB. Without this flag the script is a no-op dry-run.",
    )
    p.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Target a single run UUID (useful for spot checks before bulk apply).",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=100,
        help="Number of rows to process per DB transaction in apply mode.",
    )
    p.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Maximum number of runs to process (None = all).",
    )
    return p


# ---------------------------------------------------------------------------
# Per-run computation helpers
# ---------------------------------------------------------------------------

def _recompute_profit_factor(
    raw_metrics: dict[str, Any],
) -> tuple[float | None, bool]:
    """Return (profit_factor, profit_factor_is_infinite) from stored JSONB.

    Logic mirrors M1 canonical semantics:
      - 0 trades            -> (None, False)
      - winners>0, losers=0 -> (None, True)   [infinite]
      - winners=0, losers>0 -> (0.0,  False)  [all losers]
      - mixed               -> (gross_profit/gross_loss, False)
    """
    winning = int(raw_metrics.get("winning_trades") or 0)
    losing = int(raw_metrics.get("losing_trades") or 0)
    total = int(raw_metrics.get("total_trades") or 0)

    if total == 0:
        return None, False

    if winning > 0 and losing == 0:
        return None, True

    if winning == 0 and losing > 0:
        return 0.0, False

    # Mixed case: prefer stored value if already valid, else re-derive.
    existing_pf = raw_metrics.get("profit_factor")
    if existing_pf is not None and isinstance(existing_pf, float | int) and existing_pf >= 0:
        return float(existing_pf), False

    # Fallback: derive from gross_profit / gross_loss if stored
    gp_raw = raw_metrics.get("gross_profit")
    gl_raw = raw_metrics.get("gross_loss")
    if gp_raw is not None and gl_raw is not None:
        try:
            gp = Decimal(str(gp_raw))
            gl = abs(Decimal(str(gl_raw)))
            if gl > Decimal("0"):
                return float(gp / gl), False
        except Exception:  # noqa: BLE001
            pass

    # Cannot derive; leave unchanged
    return existing_pf if isinstance(existing_pf, float | int) else None, False


def _recompute_psr_from_snapshots(
    snapshots: list[EquitySnapshotORM],
    timeframe_str: str | None,
) -> float | None:
    """Recompute PSR from stored equity snapshots.

    Returns None when:
      - Fewer than 30 equity points (PSR asymptotic validity threshold).
      - Timeframe is unknown / missing.
      - Sharpe ratio computation yields 0.0 with no variance.
    """
    if len(snapshots) < 31:  # need at least 31 points for 30 returns
        return None

    # Build minimal EquityCurvePoint sequence (sorted by bar_index)
    sorted_snaps = sorted(snapshots, key=lambda s: s.bar_index)
    curve_points = [
        EquityCurvePoint(
            timestamp=s.timestamp,
            equity=s.equity,
            drawdown_pct=float(s.drawdown_pct),
        )
        for s in sorted_snaps
    ]

    returns = compute_returns_from_equity(curve_points)
    n = len(returns)
    if n < 30:
        return None

    # Resolve annualisation factor
    tf_enum: TimeFrame | None = None
    if timeframe_str:
        try:
            tf_enum = TimeFrame(timeframe_str)
        except ValueError:
            pass

    periods_per_year = TIMEFRAME_PERIODS_PER_YEAR.get(tf_enum or TimeFrame.ONE_HOUR, 8_766.0)

    # Per-period (de-annualised) Sharpe for PSR formula
    annual_sharpe = compute_sharpe(returns, periods_per_year)
    if annual_sharpe == 0.0:
        return None

    per_period_sharpe = annual_sharpe / math.sqrt(periods_per_year)

    # Higher moments for PSR
    mean_r = sum(returns) / n
    try:
        skew_r = statistics.mean(
            [(r - mean_r) ** 3 for r in returns]
        ) / (statistics.stdev(returns) ** 3)
        kurt_r = statistics.mean(
            [(r - mean_r) ** 4 for r in returns]
        ) / (statistics.stdev(returns) ** 4)
    except statistics.StatisticsError:
        skew_r = 0.0
        kurt_r = 3.0

    return cast(
        "float | None",
        compute_psr(
            per_period_sharpe,
            n_observations=n,
            skew=skew_r,
            kurtosis=kurt_r,
        ),
    )


# ---------------------------------------------------------------------------
# Per-run backfill logic
# ---------------------------------------------------------------------------

class _RunDiff:
    """Container for computed deltas for one run."""

    __slots__ = (
        "run_id",
        "n_closed_trades",
        "profit_factor",
        "profit_factor_is_infinite",
        "psr",
        "confidence_flag",
        "quote_currency",
        "skip_reason",
    )

    def __init__(self, run_id: uuid.UUID) -> None:
        self.run_id = run_id
        self.n_closed_trades: int | None = None
        self.profit_factor: float | None = None
        self.profit_factor_is_infinite: bool = False
        self.psr: float | None = None
        self.confidence_flag: str | None = None
        self.quote_currency: str | None = None
        self.skip_reason: str | None = None

    @property
    def is_skip(self) -> bool:
        return self.skip_reason is not None

    def __repr__(self) -> str:
        return (
            f"RunDiff(run_id={self.run_id!s:.8}, "
            f"n_trades={self.n_closed_trades}, "
            f"pf={self.profit_factor}, "
            f"pf_inf={self.profit_factor_is_infinite}, "
            f"psr={self.psr}, "
            f"flag={self.confidence_flag}, "
            f"quote={self.quote_currency})"
        )


def _compute_diff(
    run: RunORM,
    snapshots: list[EquitySnapshotORM],
) -> _RunDiff:
    """Compute the backfill diff for a single run.

    Never raises — all failures are encoded as skip_reason.
    """
    diff = _RunDiff(run.id)

    config = run.config or {}
    raw_metrics: Any = config.get("backtest_metrics")

    if not isinstance(raw_metrics, dict):
        diff.skip_reason = "no backtest_metrics in config JSONB"
        return diff

    # --- n_closed_trades ---
    # Legacy pre-Sprint-49 total_trades was always closed-only (no synthetic-close
    # mechanism existed before M3). Safe to treat as n_closed_trades.
    total_trades_raw = raw_metrics.get("total_trades")
    try:
        diff.n_closed_trades = int(total_trades_raw) if total_trades_raw is not None else 0
    except (TypeError, ValueError):
        diff.n_closed_trades = 0

    # --- profit_factor + profit_factor_is_infinite (M1 semantics) ---
    if "winning_trades" not in raw_metrics and "losing_trades" not in raw_metrics:
        logger.warning(
            "backfill.missing_pf_fields",
            run_id=str(run.id),
            message="winning_trades/losing_trades absent; leaving profit_factor unchanged",
        )
        # Leave as existing value; do not override
        existing_pf = raw_metrics.get("profit_factor")
        diff.profit_factor = float(existing_pf) if isinstance(existing_pf, float | int) else None
        diff.profit_factor_is_infinite = bool(raw_metrics.get("profit_factor_is_infinite", False))
    else:
        diff.profit_factor, diff.profit_factor_is_infinite = _recompute_profit_factor(raw_metrics)

    # --- PSR ---
    diff.psr = _recompute_psr_from_snapshots(
        snapshots,
        timeframe_str=config.get("timeframe"),
    )

    # --- confidence_flag (M4) ---
    diff.confidence_flag = _derive_confidence_flag(
        diff.psr,
        diff.n_closed_trades or 0,
    )

    # --- quote_currency (M6) ---
    symbols_raw = config.get("symbols")
    if isinstance(symbols_raw, list) and symbols_raw:
        valid_symbols = [s for s in symbols_raw if isinstance(s, str) and s]
        if valid_symbols:
            diff.quote_currency = _detect_quote_currency(valid_symbols)
        else:
            logger.warning(
                "backfill.malformed_symbols",
                run_id=str(run.id),
                symbols=symbols_raw,
            )
    else:
        logger.warning(
            "backfill.missing_symbols",
            run_id=str(run.id),
        )

    return diff


def _apply_diff_to_orm(run: RunORM, diff: _RunDiff) -> None:
    """Mutate run in-place to apply the computed diff.

    Also sets metrics_v2_backfilled = True so the row is not reprocessed.
    """
    from sqlalchemy.orm.attributes import flag_modified  # local import keeps top-level clean

    # Top-level SQL column
    run.n_closed_trades = diff.n_closed_trades

    # Mutate JSONB in-place (must flag_modified for SQLAlchemy change detection)
    config = dict(run.config or {})
    raw_metrics: dict[str, Any] = dict(config.get("backtest_metrics") or {})

    raw_metrics["profit_factor"] = diff.profit_factor
    raw_metrics["profit_factor_is_infinite"] = diff.profit_factor_is_infinite
    raw_metrics["psr"] = diff.psr
    raw_metrics["confidence_flag"] = diff.confidence_flag
    raw_metrics["quote_currency"] = diff.quote_currency
    # Fields this backfill CANNOT reconstruct -- mark as empty to signal "legacy"
    raw_metrics.setdefault("exposure_pct_per_symbol", {})
    raw_metrics.setdefault("open_positions_mtm", [])

    config["backtest_metrics"] = raw_metrics
    run.config = config
    try:
        flag_modified(run, "config")
    except AttributeError:
        # run is not a SQLAlchemy-tracked instance (e.g. SimpleNamespace in tests)
        # — attribute write above is already visible; no tracking needed.
        pass

    # Idempotency sentinel
    run.metrics_v2_backfilled = True


# ---------------------------------------------------------------------------
# Aggregate stats collector
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self) -> None:
        self.total_found: int = 0
        self.total_skipped: int = 0
        self.total_updated: int = 0
        self.n_trades_updated: int = 0
        self.pf_updated: int = 0
        self.psr_computed: int = 0
        self.quote_detected: int = 0
        self.batch_errors: int = 0

    def record(self, diff: _RunDiff) -> None:
        if diff.is_skip:
            self.total_skipped += 1
            return
        self.total_updated += 1
        if diff.n_closed_trades is not None:
            self.n_trades_updated += 1
        if diff.profit_factor is not None or diff.profit_factor_is_infinite:
            self.pf_updated += 1
        if diff.psr is not None:
            self.psr_computed += 1
        if diff.quote_currency is not None:
            self.quote_detected += 1

    def print_summary(self, apply: bool) -> None:
        action = "updated" if apply else "would_update"
        log = logger.bind(apply=apply)
        log.info(
            "backfill.summary_line",
            runs_found=self.total_found,
            runs_skipped=self.total_skipped,
            runs_processable=self.total_updated,
            **{f"{action}_n_closed_trades": self.n_trades_updated},
            **{f"{action}_profit_factor": self.pf_updated},
            **{f"{action}_psr": self.psr_computed},
            **{f"{action}_quote_currency": self.quote_detected},
            batch_errors=self.batch_errors,
        )


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------

async def _run(
    *,
    apply: bool,
    run_id_filter: uuid.UUID | None,
    batch_size: int,
    limit: int | None,
) -> int:
    """Execute the backfill.

    Returns
    -------
    int
        Exit code (0 = success, 1 = fatal error).
    """
    stats = _Stats()
    session_factory = get_session_factory()

    log = logger.bind(apply=apply, batch_size=batch_size, limit=limit)
    log.info("backfill.start")

    # ------------------------------------------------------------------
    # Phase 1: Load all candidate run IDs
    # ------------------------------------------------------------------
    async with session_factory() as session:
        stmt = (
            select(RunORM.id)
            .where(
                RunORM.run_mode == "backtest",
                RunORM.metrics_v2_backfilled.is_(False),
            )
            .order_by(RunORM.created_at.asc())
        )
        if run_id_filter is not None:
            stmt = stmt.where(RunORM.id == run_id_filter)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        candidate_ids: list[uuid.UUID] = list(result.scalars().all())

    stats.total_found = len(candidate_ids)
    log.info("backfill.candidates_found", count=stats.total_found)

    if not candidate_ids:
        log.info("backfill.no_candidates", message="No runs need backfilling (all up to date or none found).")
        stats.print_summary(apply)
        return 0

    # ------------------------------------------------------------------
    # Phase 2: Process in batches
    # ------------------------------------------------------------------
    diffs: list[_RunDiff] = []
    previewed = 0

    for batch_start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[batch_start : batch_start + batch_size]

        async with session_factory() as session:
            # Load run + equity snapshots in one round-trip via selectinload
            stmt2 = (
                select(RunORM)
                .where(RunORM.id.in_(batch_ids))
                .options(selectinload(RunORM.equity_snapshots))
                .order_by(RunORM.created_at.asc())
            )
            result2 = await session.execute(stmt2)
            runs: list[RunORM] = list(result2.scalars().all())

            batch_diffs: list[_RunDiff] = []
            for run in runs:
                diff = _compute_diff(run, run.equity_snapshots)
                batch_diffs.append(diff)
                stats.record(diff)

                # Dry-run preview
                if previewed < _PREVIEW_ROWS:
                    if diff.is_skip:
                        log.info(
                            "backfill.skip",
                            run_id=str(run.id)[:8],
                            reason=diff.skip_reason,
                        )
                    else:
                        existing_n = run.n_closed_trades
                        existing_pf = (run.config or {}).get(
                            "backtest_metrics", {}
                        ).get("profit_factor")
                        log.info(
                            "backfill.diff",
                            run_id=str(run.id)[:8],
                            n_trades_before=existing_n,
                            n_trades_after=diff.n_closed_trades,
                            pf_before=existing_pf,
                            pf_after=diff.profit_factor,
                            psr=diff.psr,
                            confidence_flag=diff.confidence_flag,
                            quote_currency=diff.quote_currency,
                        )
                    previewed += 1
                elif previewed == _PREVIEW_ROWS:
                    log.info("backfill.preview_truncated", showing_first=_PREVIEW_ROWS)
                    previewed += 1

            diffs.extend(batch_diffs)

            if apply:
                # Apply mutations to ORM objects already in the session identity map —
                # no new equity snapshot query needed (write phase only mutates config + n_closed_trades).
                try:
                    stmt3 = (
                        select(RunORM)
                        .where(RunORM.id.in_(batch_ids))
                    )
                    result3 = await session.execute(stmt3)
                    write_runs = {r.id: r for r in result3.scalars().all()}

                    for diff in batch_diffs:
                        if diff.is_skip:
                            # Even on skip: mark backfilled so we don't retry
                            wr = write_runs.get(diff.run_id)
                            if wr is not None:
                                wr.metrics_v2_backfilled = True
                            continue
                        wr = write_runs.get(diff.run_id)
                        if wr is not None:
                            _apply_diff_to_orm(wr, diff)

                    await session.commit()
                    log.info(
                        "backfill.batch_committed",
                        batch_start=batch_start,
                        batch_size=len(batch_ids),
                    )

                except Exception:  # noqa: BLE001
                    await session.rollback()
                    stats.batch_errors += 1
                    log.error(
                        "backfill.batch_rolled_back",
                        batch_start=batch_start,
                        exc_info=True,
                    )
                    # Continue with next batch rather than aborting everything

    stats.print_summary(apply)

    if not apply:
        log.info("backfill.dry_run_complete", hint="Re-run with --apply to write changes.")

    return 0 if stats.batch_errors == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    run_id_filter: uuid.UUID | None = None
    if args.run_id is not None:
        try:
            run_id_filter = uuid.UUID(args.run_id)
        except ValueError:
            print(f"ERROR: --run-id '{args.run_id}' is not a valid UUID.", file=sys.stderr)
            sys.exit(1)

    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1.", file=sys.stderr)
        sys.exit(1)

    try:
        exit_code = asyncio.run(
            _run(
                apply=args.apply,
                run_id_filter=run_id_filter,
                batch_size=args.batch_size,
                limit=args.limit,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("backfill.fatal_error", exc_info=True)
        print(f"FATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
