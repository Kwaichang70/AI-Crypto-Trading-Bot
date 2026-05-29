"""
tests/unit/test_sprint50_cycle5_promotion_gate.py
--------------------------------------------------
Unit tests for Sprint 50 Cycle 5 Sub-scope A: Paper->Live Promotion Gate.

Modules under test
------------------
apps/api/services/promotion_gate.py  -- PromotionEligibility, evaluate_paper_run_eligibility
apps/api/routers/runs.py             -- get_promotion_eligibility, promote_to_live endpoints
apps/api/config.py                   -- min_paper_trades_for_promotion, min_paper_runtime_days

Coverage groups (12 tests)
---------------------------------------------------------------------------
TestPromotionEligibilityService (6):
  test_eligible_when_trades_and_runtime_sufficient
  test_ineligible_trade_count_below_min
  test_ineligible_runtime_below_min
  test_ineligible_both_criteria_fail
  test_wrong_run_mode_returns_wrong_run_mode_reason
  test_still_running_returns_run_not_stopped (CR5-006 early return)

TestPromotionEligibilityEarlyReturn (2):
  test_wrong_mode_returns_immediately_no_db_query
  test_not_stopped_returns_immediately_no_db_query

TestNewConfigSettings (2):
  test_min_paper_trades_for_promotion_default
  test_min_paper_runtime_days_default

TestAuditEventTypesRegistered (2):
  test_paper_promoted_to_live_in_valid_event_types
  test_model_oos_gate_bypassed_in_valid_event_types
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# TestPromotionEligibilityService
# ===========================================================================


class TestPromotionEligibilityService:
    """Unit tests for evaluate_paper_run_eligibility service function."""

    def _make_run(
        self,
        run_mode: str = "paper",
        status: str = "stopped",
        runtime_days: float = 8.0,
    ) -> Any:
        """Build a minimal RunORM-like namespace."""
        import uuid as _uuid
        started = datetime(2026, 1, 1, tzinfo=UTC)
        stopped = started + timedelta(days=runtime_days)
        return SimpleNamespace(
            id=_uuid.uuid4(),
            run_mode=run_mode,
            status=status,
            started_at=started,
            stopped_at=stopped,
            config={},
        )

    def _make_db(self, trade_count: int = 60) -> AsyncMock:
        """Build a mock AsyncSession that returns trade_count for COUNT query."""
        db = AsyncMock()
        scalar_result = MagicMock()
        scalar_result.scalar_one.return_value = trade_count
        db.execute = AsyncMock(return_value=scalar_result)
        return db

    @pytest.mark.asyncio
    async def test_eligible_when_trades_and_runtime_sufficient(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="stopped", runtime_days=8.0)
        db = self._make_db(trade_count=60)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is True
        assert result.trade_count == 60
        assert result.runtime_days >= 7.9
        assert result.reasons == []

    @pytest.mark.asyncio
    async def test_ineligible_trade_count_below_min(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="stopped", runtime_days=8.0)
        db = self._make_db(trade_count=10)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "trade_count_below_min" in result.reasons
        assert "runtime_below_min" not in result.reasons

    @pytest.mark.asyncio
    async def test_ineligible_runtime_below_min(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="stopped", runtime_days=3.0)
        db = self._make_db(trade_count=60)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "runtime_below_min" in result.reasons
        assert "trade_count_below_min" not in result.reasons

    @pytest.mark.asyncio
    async def test_ineligible_both_criteria_fail(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="stopped", runtime_days=2.0)
        db = self._make_db(trade_count=5)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "trade_count_below_min" in result.reasons
        assert "runtime_below_min" in result.reasons

    @pytest.mark.asyncio
    async def test_wrong_run_mode_returns_wrong_run_mode_reason(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="live", status="stopped", runtime_days=8.0)
        db = self._make_db(trade_count=60)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "wrong_run_mode" in result.reasons

    @pytest.mark.asyncio
    async def test_still_running_returns_run_not_stopped(self) -> None:
        """CR5-006: run_not_stopped triggers early return with trade_count=0, runtime_days=0."""
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="running", runtime_days=8.0)
        db = self._make_db(trade_count=60)

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "run_not_stopped" in result.reasons
        # CR5-006: early return means trade_count and runtime_days are 0 (not partial)
        assert result.trade_count == 0
        assert result.runtime_days == 0.0


# ===========================================================================
# TestPromotionEligibilityEarlyReturn
# ===========================================================================


class TestPromotionEligibilityEarlyReturn:
    """Verify that early-return paths do not issue DB queries."""

    def _make_run(self, run_mode: str, status: str) -> Any:
        import uuid as _uuid
        return SimpleNamespace(
            id=_uuid.uuid4(),
            run_mode=run_mode,
            status=status,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            stopped_at=datetime(2026, 1, 8, tzinfo=UTC),
            config={},
        )

    @pytest.mark.asyncio
    async def test_wrong_mode_returns_immediately_no_db_query(self) -> None:
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="backtest", status="stopped")
        db = AsyncMock()

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "wrong_run_mode" in result.reasons
        # No COUNT query should have been issued
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_stopped_returns_immediately_no_db_query(self) -> None:
        """CR5-006: run_not_stopped early return must not query the DB."""
        from api.services.promotion_gate import evaluate_paper_run_eligibility

        run = self._make_run(run_mode="paper", status="running")
        db = AsyncMock()

        result = await evaluate_paper_run_eligibility(
            db=db, run_orm=run, min_trades=50, min_runtime_days=7.0
        )

        assert result.eligible is False
        assert "run_not_stopped" in result.reasons
        db.execute.assert_not_called()


# ===========================================================================
# TestNewConfigSettings
# ===========================================================================


class TestNewConfigSettings:
    """Verify the 5 new Sprint 50 Cycle 5 Settings fields have correct defaults."""

    def test_min_paper_trades_for_promotion_default(self) -> None:
        from api.config import get_settings
        get_settings.cache_clear()
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
        assert settings.min_paper_trades_for_promotion == 50
        get_settings.cache_clear()

    def test_min_paper_runtime_days_default(self) -> None:
        from api.config import get_settings
        get_settings.cache_clear()
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
        assert settings.min_paper_runtime_days == 7.0
        get_settings.cache_clear()

    def test_min_oos_skill_score_default(self) -> None:
        from api.config import get_settings
        get_settings.cache_clear()
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
        # Sprint 50 Cycle 6 lowered this default to 1.0 (realized OOS
        # Sharpe gate; was 0.5 in Cycle 5, 1.64 after the quant caveat).
        assert settings.min_oos_skill_score == 1.0
        get_settings.cache_clear()

    def test_min_worst_fold_skill_score_default(self) -> None:
        from api.config import get_settings
        get_settings.cache_clear()
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
        assert settings.min_worst_fold_skill_score == 0.0
        get_settings.cache_clear()

    def test_min_trades_per_fold_default(self) -> None:
        from api.config import get_settings
        get_settings.cache_clear()
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
        # Sprint 50 Cycle 6 lowered this default 20 -> 5 (realized OOS
        # trade counts are sparse over ~120-bar folds at threshold 0.60).
        assert settings.min_trades_per_fold == 5
        get_settings.cache_clear()


# ===========================================================================
# TestAuditEventTypesRegistered
# ===========================================================================


class TestAuditEventTypesRegistered:
    """Verify the two new event types are registered in audit_log._VALID_EVENT_TYPES."""

    def test_paper_promoted_to_live_in_valid_event_types(self) -> None:
        from api.services.audit_log import _VALID_EVENT_TYPES
        assert "paper_promoted_to_live" in _VALID_EVENT_TYPES

    def test_model_oos_gate_bypassed_in_valid_event_types(self) -> None:
        from api.services.audit_log import _VALID_EVENT_TYPES
        assert "model_oos_gate_bypassed" in _VALID_EVENT_TYPES

    def test_existing_event_types_still_present(self) -> None:
        """Regression: existing 6 types must still be in the set."""
        from api.services.audit_log import _VALID_EVENT_TYPES
        for expected in (
            "live_trading_enabled",
            "model_activated",
            "circuit_breaker_reset",
            "emergency_stop",
            "kill_switch",
            "circuit_breaker_halt_auto_stop",
        ):
            assert expected in _VALID_EVENT_TYPES, f"Missing: {expected}"
