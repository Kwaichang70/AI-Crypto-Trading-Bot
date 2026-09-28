#!/usr/bin/env python3
"""
scripts/wp13a_exit_config_precheck.py
--------------------------------------
WP1.3a (SY-13a-21) -- pre-deploy precheck.

Deploying the WP1.3a patch restarts the API, which orphans every running
live run (the boot-recovery path never validates or crashes -- orphan
only). The operator then resumes each one, and THAT resume runs the new
``trading.exit_config`` validator for the first time. This script lets the
operator find out BEFORE deploying, by reading every non-terminal paper and
live run's persisted config and running the exact same validator, so a
config that would fail at resume time is known in advance and can be fixed,
or a protective-resume-then-flatten plan can be prepared (AC7).

Read-only. Makes NO writes: no ``UPDATE``, no ``INSERT``, no state mutation
of any kind. Safe to run against production at any time, repeatedly.

Usage
-----
    python scripts/wp13a_exit_config_precheck.py
    python scripts/wp13a_exit_config_precheck.py --run-id <UUID>
    python scripts/wp13a_exit_config_precheck.py --json

Exit codes
----------
    0 -- every checked run's exit config is valid (E1-E11), no live_
         pyramiding_forbidden hits.
    1 -- at least one run is invalid; see the printed report.
    2 -- fatal error (DB connection failure, import error).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure workspace packages are importable when run directly (mirrors every
# other script in this directory -- see e.g. backfill_metrics_v2.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg_dir in (_REPO_ROOT / "packages", _REPO_ROOT / "apps"):
    if str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

# Non-terminal statuses this precheck cares about -- 'stopped'/'error'/
# 'archived' runs no longer trade and are irrelevant here.
_NON_TERMINAL_STATUSES = ("running", "orphaned", "resuming")


def _check_one_run(
    run_id: uuid.UUID, run_mode: str, status_value: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Run the shared validator against one run's persisted config.

    Returns a plain dict (JSON-safe) describing the outcome -- never
    raises; every failure mode is captured in the returned dict.
    """
    from common.types import RunMode
    from trading.exit_config import ExitConfigError, validate_run_exit_config
    from trading.strategy_availability import get_availability, is_mode_allowed

    entry: dict[str, Any] = {
        "run_id": str(run_id),
        "mode": run_mode,
        "status": status_value,
        "strategy_name": None,
        "valid": None,
        "codes": [],
        "warnings": [],
        "note": None,
    }

    strategy_name_raw = str(config.get("strategy_name", ""))
    strategy_name = strategy_name_raw.lower().replace("-", "_")
    entry["strategy_name"] = strategy_name_raw

    from api.routers.runs import _get_strategy_registry

    strategy_cls = _get_strategy_registry().get(strategy_name)
    if strategy_cls is None:
        entry["valid"] = False
        entry["codes"] = ["unknown_strategy"]
        entry["note"] = "strategy is no longer registered"
        return entry

    try:
        run_mode_enum = RunMode(run_mode)
    except ValueError:
        run_mode_enum = RunMode.PAPER if run_mode == "paper" else RunMode.LIVE

    if not is_mode_allowed(strategy_name, run_mode_enum):
        availability = get_availability(strategy_name)
        entry["valid"] = False
        entry["codes"] = ["strategy_mode_not_allowed"]
        entry["note"] = f"status={availability.status.value}"
        return entry

    strategy_params: dict[str, Any] = config.get("strategy_params") or {}
    bracket_config: dict[str, Any] = config.get("bracket_config") or {}
    try:
        verdict = validate_run_exit_config(
            strategy_cls,
            bracket=bracket_config,
            trailing_stop_pct=strategy_params.get("trailing_stop_pct"),
            mode=run_mode_enum,
            allow_pyramiding=config.get("allow_pyramiding"),
            timeframe=str(config.get("timeframe", "1h")),
        )
    except ExitConfigError as exc:
        entry["valid"] = False
        entry["codes"] = [exc.code]
        if exc.issues:
            entry["note"] = "; ".join(f"{i.field}:{i.reason}" for i in exc.issues)
        elif exc.hint:
            entry["note"] = exc.hint
        return entry

    entry["valid"] = True
    entry["warnings"] = [w.code for w in verdict.warnings]
    return entry


async def _run(run_id_filter: uuid.UUID | None) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from api.db.models import RunORM
    from api.db.session import get_session_factory

    factory = get_session_factory()
    results: list[dict[str, Any]] = []
    async with factory() as session:
        # WP13a-S-05 (security round 2): enforce read-only at the
        # PostgreSQL level, not just "the code only ever issues a
        # SELECT" -- ``postgresql_readonly`` is pushed down to a real
        # ``SET TRANSACTION READ ONLY`` on asyncpg/psycopg alike, so even
        # a future bug in this script (or a strategy_cls property with a
        # side effect) could never write.  Any attempted write raises
        # ``InFailedSqlTransactionError``/``ReadOnlySqlTransactionError``
        # from Postgres itself.
        await session.connection(execution_options={"postgresql_readonly": True})
        stmt = select(RunORM).where(
            RunORM.status.in_(_NON_TERMINAL_STATUSES),
            RunORM.run_mode.in_(["paper", "live"]),
        )
        if run_id_filter is not None:
            stmt = stmt.where(RunORM.id == run_id_filter)
        rows = await session.execute(stmt)
        runs = list(rows.scalars().all())

    for run in runs:
        config = dict(run.config or {})
        entry = _check_one_run(run.id, run.run_mode, run.status, config)
        results.append(entry)

    return results


def _print_report(results: list[dict[str, Any]], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        if not results:
            print("wp13a-precheck: no non-terminal paper/live runs found.")
        for entry in results:
            status_tag = "OK" if entry["valid"] else "INVALID"
            line = (
                f"[{status_tag}] run_id={entry['run_id']} mode={entry['mode']} "
                f"status={entry['status']} strategy={entry['strategy_name']!r}"
            )
            if entry["codes"]:
                line += f" codes={entry['codes']}"
            if entry["warnings"]:
                line += f" warnings={entry['warnings']}"
            if entry["note"]:
                line += f" note={entry['note']!r}"
            print(line)

    total = len(results)
    invalid = [e for e in results if not e["valid"]]
    print(
        f"wp13a-precheck: {total} run(s) checked, {len(invalid)} invalid. "
        "No writes were made."
    )
    return 1 if invalid else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Check only this run UUID (default: every non-terminal paper/live run).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report.",
    )
    args = parser.parse_args()

    run_id_filter: uuid.UUID | None = None
    if args.run_id is not None:
        try:
            run_id_filter = uuid.UUID(args.run_id)
        except ValueError:
            print(f"wp13a-precheck: invalid --run-id {args.run_id!r}", file=sys.stderr)
            return 2

    try:
        results = asyncio.run(_run(run_id_filter))
    except Exception as exc:
        print(f"wp13a-precheck: fatal error: {exc}", file=sys.stderr)
        return 2

    return _print_report(results, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
