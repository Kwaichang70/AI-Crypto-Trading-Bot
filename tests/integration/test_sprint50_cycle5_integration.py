"""
tests/integration/test_sprint50_cycle5_integration.py
------------------------------------------------------
Integration tests for Sprint 50 Cycle 5: Promotion Gates.

These tests exercise the service + ORM boundary via mocked DB sessions
following the Sprint 10 SimpleNamespace pattern (no real PostgreSQL required).

Coverage groups (2 tests)
---------------------------------------------------------------------------
TestPromotionGateServiceIntegration (1):
  test_evaluate_paper_run_eligibility_full_path
    -- Eligible paper run (mode=paper, status=stopped, 60 trades, 8 days)
    -- Verifies eligible=True, trade_count=60, runtime_days approx 8, reasons=[]

TestModelActivationGateServiceIntegration (1):
  test_check_oos_eligibility_with_worst_fold_floor
    -- Model with OOS skill score=0.6, worst_fold=-0.1, floor=0.0
    -- Verifies blocked by worst-fold floor (not median gate)
    -- Verifies reason="worst_fold_below_floor"
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ===========================================================================
# TestPromotionGateServiceIntegration
# ===========================================================================


@pytest.mark.integration
class TestPromotionGateServiceIntegration:
    """Integration test for the full promotion gate service path."""

    @pytest.mark.asyncio
    async def test_evaluate_paper_run_eligibility_full_path(self) -> None:
        """Eligible paper run returns fully-populated PromotionEligibility(eligible=True)."""
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        started = datetime(2026, 1, 1, tzinfo=UTC)
        stopped = started + timedelta(days=8)

        run_orm = SimpleNamespace(
            id=uuid.uuid4(),
            run_mode="paper",
            status="stopped",
            started_at=started,
            stopped_at=stopped,
            config={"strategy_name": "ma_crossover"},
        )

        # Mock DB session returning COUNT=60
        scalar_result = MagicMock()
        scalar_result.scalar_one.return_value = 60
        db = AsyncMock()
        db.execute = AsyncMock(return_value=scalar_result)

        result = await evaluate_paper_run_eligibility(
            db=db,
            run_orm=run_orm,
            min_trades=50,
            min_runtime_days=7.0,
        )

        assert result.eligible is True
        assert result.trade_count == 60
        assert result.runtime_days == pytest.approx(8.0, abs=0.01)
        assert result.reasons == []
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_paper_run_ineligible_trade_count(self) -> None:
        """Ineligible due to insufficient trade count returns correct failure reasons."""
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        started = datetime(2026, 1, 1, tzinfo=UTC)
        stopped = started + timedelta(days=10)

        run_orm = SimpleNamespace(
            id=uuid.uuid4(),
            run_mode="paper",
            status="stopped",
            started_at=started,
            stopped_at=stopped,
            config={},
        )

        scalar_result = MagicMock()
        scalar_result.scalar_one.return_value = 5  # below 50
        db = AsyncMock()
        db.execute = AsyncMock(return_value=scalar_result)

        result = await evaluate_paper_run_eligibility(
            db=db,
            run_orm=run_orm,
            min_trades=50,
            min_runtime_days=7.0,
        )

        assert result.eligible is False
        assert "trade_count_below_min" in result.reasons
        assert "runtime_below_min" not in result.reasons


# ===========================================================================
# TestModelActivationGateServiceIntegration
# ===========================================================================


@pytest.mark.integration
class TestModelActivationGateServiceIntegration:
    """Integration test for the OOS gate service with real dataclass interaction."""

    def test_check_oos_eligibility_with_worst_fold_floor(self) -> None:
        """Model blocked by worst-fold floor even though median passes."""
        from api.services.model_activation_gate import check_oos_eligibility

        model_version = SimpleNamespace(
            id=uuid.uuid4(),
            symbol="ETH/USD",
            timeframe="4h",
            walk_forward_oos_skill_score=Decimal("0.60"),  # above 0.5 median threshold
            extra={
                "walk_forward": {
                    "status": "ok",
                    "metric_type": "directional_zscore_proxy",
                    "oos_skill_median_deflated": 0.60,
                    "oos_skill_worst": -0.10,  # below 0.0 floor
                    "oos_skill_median_raw": 0.75,
                    "fold_trade_counts": [50, 45, 55, 40, 60],
                }
            },
        )

        result = check_oos_eligibility(
            model_version,
            min_oos_skill_score=0.5,
            min_worst_fold_skill_score=0.0,
        )

        assert result.eligible is False
        assert result.reason == "worst_fold_below_floor"
        assert result.oos_skill_score == pytest.approx(0.60)
        assert result.worst_fold_skill_score == pytest.approx(-0.10)

    def test_check_oos_eligibility_passes_all_gates(self) -> None:
        """Model passing all gates returns eligible=True with no reason."""
        from api.services.model_activation_gate import check_oos_eligibility

        model_version = SimpleNamespace(
            id=uuid.uuid4(),
            symbol="BTC/USD",
            timeframe="1h",
            walk_forward_oos_skill_score=Decimal("0.70"),
            extra={
                "walk_forward": {
                    "status": "ok",
                    "metric_type": "directional_zscore_proxy",
                    "oos_skill_median_deflated": 0.70,
                    "oos_skill_worst": 0.20,
                    "fold_trade_counts": [50, 45, 55, 40, 60],
                }
            },
        )

        result = check_oos_eligibility(
            model_version,
            min_oos_skill_score=0.5,
            min_worst_fold_skill_score=0.0,
            min_trades_per_fold=20,
        )

        assert result.eligible is True
        assert result.reason == ""
        assert result.warning == ""
        assert result.detail == ""
