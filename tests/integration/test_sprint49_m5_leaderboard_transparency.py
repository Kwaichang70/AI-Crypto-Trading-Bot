"""
tests/integration/test_sprint49_m5_leaderboard_transparency.py
---------------------------------------------------------------
Integration tests for Sprint 49 M5: leaderboard transparency.

Covers
------
1.  list_runs -- 422 on invalid sort_by (JSONB field not permitted)
2.  list_runs -- 422 on invalid sort_order
3.  list_runs -- HTTP 200 when sort_by=n_closed_trades (valid column)
4.  list_runs -- HTTP 200 and filtering when min_closed_trades supplied
5.  get_aggregate_portfolio -- ineligible runs excluded from best_run_return_pct
6.  get_aggregate_portfolio -- eligible_runs_count populated correctly

Uses client_dev_with_db + mock_db_session fixtures from conftest.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
_UUID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")

_BASE_CONFIG = {
    "strategy_name": "ma_crossover",
    "strategy_params": {},
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "mode": "backtest",
    "initial_capital": "10000.00",
}


def _make_run_ns(
    *,
    n_closed_trades: int | None = None,
    config: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Minimal RunORM-like namespace for Pydantic from_attributes deserialization."""
    return SimpleNamespace(
        id=_UUID,
        run_mode="backtest",
        status="stopped",
        config=config or _BASE_CONFIG,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_closed_trades,
    )


def _wire_list_runs_mock(mock_db_session: AsyncMock, runs: list[Any]) -> None:
    """Wire mock_db_session for a list_runs call (count then page)."""
    scalar_result = MagicMock()
    scalar_result.scalar_one.return_value = len(runs)
    scalars_result = MagicMock()
    scalars_result.scalars.return_value.all.return_value = runs
    mock_db_session.execute.side_effect = [scalar_result, scalars_result]


# ---------------------------------------------------------------------------
# Tests: list_runs sort parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestListRunsSortValidation:
    """sort_by and sort_order validation on GET /api/v1/runs."""

    def test_invalid_sort_by_returns_422(
        self, client_dev_with_db: Any, mock_db_session: AsyncMock
    ) -> None:
        """sort_by=psr (JSONB field) must return HTTP 422 immediately."""
        resp = client_dev_with_db.get("/api/v1/runs?sort_by=psr")
        assert resp.status_code == 422
        body = resp.json()
        assert "sort_by" in body["detail"].lower()

    def test_invalid_sort_order_returns_422(
        self, client_dev_with_db: Any, mock_db_session: AsyncMock
    ) -> None:
        """sort_order=random must return HTTP 422."""
        resp = client_dev_with_db.get("/api/v1/runs?sort_order=random")
        assert resp.status_code == 422

    def test_sort_by_n_closed_trades_accepted(
        self, client_dev_with_db: Any, mock_db_session: AsyncMock
    ) -> None:
        """sort_by=n_closed_trades must return HTTP 200."""
        run_ns = _make_run_ns(n_closed_trades=25)
        _wire_list_runs_mock(mock_db_session, [run_ns])

        resp = client_dev_with_db.get("/api/v1/runs?sort_by=n_closed_trades")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1

    def test_min_closed_trades_filter_accepted(
        self, client_dev_with_db: Any, mock_db_session: AsyncMock
    ) -> None:
        """min_closed_trades=10 must return HTTP 200 with filtered results."""
        run_ns = _make_run_ns(n_closed_trades=15)
        _wire_list_runs_mock(mock_db_session, [run_ns])

        resp = client_dev_with_db.get("/api/v1/runs?min_closed_trades=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"][0]["nClosedTrades"] == 15
