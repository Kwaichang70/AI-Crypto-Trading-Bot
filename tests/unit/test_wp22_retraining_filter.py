"""WP2.2 P4 -- retraining queries only read paper/live trades (SY-22-11).

Backtest trades become aligned in-sample labels once WP2.2 stamps bar time,
so both retraining queries must join ``runs`` and allow-list ``run_mode``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services.retraining import TRAINING_RUN_MODES, RetrainingService


def _service() -> tuple[RetrainingService, list[Any]]:
    captured: list[Any] = []
    result = MagicMock()
    result.scalar_one = MagicMock(return_value=0)
    result.all = MagicMock(return_value=[])

    async def _execute(stmt: Any) -> Any:
        captured.append(stmt)
        return result

    session = AsyncMock()
    session.execute = _execute
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    svc = RetrainingService(db_session_factory=MagicMock(return_value=ctx))
    return svc, captured


def _sql(stmt: Any) -> str:
    return " ".join(str(stmt.compile()).split())


async def _count_sql() -> str:
    svc, cap = _service()
    await svc._check_and_retrain("BTC/USDT", "1h")
    return _sql(cap[0])


async def _fetch_sql() -> str:
    svc, cap = _service()
    await svc._fetch_trade_dicts("BTC/USDT")
    return _sql(cap[0])


def test_allow_list_is_paper_live_only() -> None:
    assert set(TRAINING_RUN_MODES) == {"paper", "live"}


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", [_count_sql, _fetch_sql])
async def test_query_joins_runs_and_allow_lists_run_mode(builder: Any) -> None:
    sql = await builder()
    assert "JOIN runs ON runs.id = trades.run_id" in sql
    assert "runs.run_mode IN (" in sql
    assert "NOT IN" not in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", [_count_sql, _fetch_sql])
async def test_bound_modes_include_paper_live_exclude_backtest_optimize(
    builder: Any,
) -> None:
    svc, cap = _service()
    if builder is _count_sql:
        await svc._check_and_retrain("BTC/USDT", "1h")
    else:
        await svc._fetch_trade_dicts("BTC/USDT")
    params = cap[0].compile().params
    bound: set[str] = set()
    for v in params.values():
        if isinstance(v, (list, tuple)):
            bound |= {x for x in v if isinstance(x, str)}
    assert bound == {"paper", "live"}
    assert not bound & {"backtest", "optimize"}
