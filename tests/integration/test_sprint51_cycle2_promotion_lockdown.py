"""
tests/integration/test_sprint51_cycle2_promotion_lockdown.py
-------------------------------------------------------------
Sprint 51 Cycle 2 — strategy-availability lockdown on the paper->live
promotion endpoint.

Endpoint under test
-------------------
POST /api/v1/runs/{run_id}/promote-to-live

The promote_to_live handler applies an OUTER availability guard (IMPL-S51C2-103)
that fires BEFORE the per-run evidence gate (evaluate_paper_run_eligibility):
a DEMOTED strategy must never reach live via the promotion path even if its
paper-run evidence would otherwise satisfy the gate.

Because the availability guard fires immediately after the source-run fetch
and before the eligibility DB query, the mock only needs to return the source
run from a single execute() call.  The strategy_name is read from the source
run's immutable config snapshot.

Test strategy
-------------
Hermetic — uses the shared client_dev_with_db / mock_db_session fixtures
(no real PostgreSQL).  SimpleNamespace mimics the RunORM source paper run.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


_FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_DEMOTED_STRATEGIES = ["ma_crossover", "breakout", "model_strategy"]


def _make_scalar_one_or_none_result(value: object) -> MagicMock:
    """execute() result exposing .scalar_one_or_none() -> value."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _make_paper_source_run(strategy_name: str) -> SimpleNamespace:
    """A stopped paper run whose config snapshot carries ``strategy_name``."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        run_mode="paper",
        status="stopped",
        config={
            "strategy_name": strategy_name,
            "strategy_params": {},
            "symbols": ["BTC/USDT"],
            "timeframe": "1h",
            "mode": "paper",
            "initial_capital": "10000.00",
        },
        started_at=_FIXED_NOW,
        stopped_at=_FIXED_NOW,
        created_at=_FIXED_NOW,
        updated_at=_FIXED_NOW,
    )


@pytest.mark.integration
class TestPromotionStrategyAvailabilityLockdown:
    """promote-to-live must reject a demoted strategy's paper run with 422."""

    @pytest.mark.parametrize("strategy", _DEMOTED_STRATEGIES)
    def test_demoted_paper_run_promotion_rejected_422(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
        strategy: str,
    ) -> None:
        """TEST-S51C2-300: promoting a demoted strategy's paper run returns 422.

        The 422 must fire before the evidence gate (no eligibility DB query is
        reached), and its detail must reference live availability + the demoted
        status.
        """
        source = _make_paper_source_run(strategy)

        # Single execute() call returns the source run; the availability guard
        # fires immediately afterwards (before the eligibility COUNT query).
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(source)

        resp = client_dev_with_db.post(
            f"/api/v1/runs/{source.id}/promote-to-live"
        )

        assert resp.status_code == 422, (
            f"Promotion of demoted {strategy} must be 422, got "
            f"{resp.status_code}: {resp.text}"
        )
        detail = resp.json()["detail"]
        assert isinstance(detail, str), "detail must be a plain string"
        assert "live" in detail.lower()
        assert "demoted" in detail.lower()

    def test_unknown_source_run_returns_404_not_422(
        self,
        client_dev_with_db: TestClient,
        mock_db_session: AsyncMock,
    ) -> None:
        """TEST-S51C2-301: a missing source run returns 404 (404 precedes the lockdown 422)."""
        mock_db_session.execute.return_value = _make_scalar_one_or_none_result(None)

        missing = uuid.uuid4()
        resp = client_dev_with_db.post(f"/api/v1/runs/{missing}/promote-to-live")

        assert resp.status_code == 404, (
            f"Missing source run must be 404, got {resp.status_code}: {resp.text}"
        )
