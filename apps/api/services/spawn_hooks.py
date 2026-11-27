"""CF-97-03 P7b: the one shared seam at every paper/live engine spawn site.

Called as ONE synchronous statement directly before ``_RUN_TASKS[...] = task``.  Never raises
(fail-open: a closed ERROR code).  In P7b it only registers the run's egress weight; P7d extends
it with ``LIVE_HOSTED`` membership (live spawns), so the spawn sites are not touched twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import structlog

from api.services import egress_budget

_log = structlog.get_logger("spawn_hooks")


def on_run_spawned(
    run_id: object, mode: str, symbols: Sequence[str], timeframe: object, task: asyncio.Task[None]
) -> None:
    try:
        rid = str(run_id)
        egress_budget.register(rid, mode, list(symbols), timeframe, task)
        task.add_done_callback(lambda _task: on_run_ended(rid, task))
    except Exception:
        _log.error("egress.registry_unavailable")


def on_run_ended(run_id: str, task: asyncio.Task[None] | None = None) -> None:
    try:
        egress_budget.unregister(run_id, task)
    except Exception:
        _log.error("egress.registry_unavailable")
