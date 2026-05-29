"""
tests/unit/test_sprint50_cycle5_model_gate.py
----------------------------------------------
Unit tests for Sprint 50 Cycle 5 Sub-scope B: Walk-Forward OOS Gate.

Modules under test
------------------
apps/api/services/model_activation_gate.py  -- OOSEligibility, check_oos_eligibility

Coverage groups (v1 + v2 additions = 16 tests)
---------------------------------------------------------------------------
TestCheckOOSEligibility (3):
  test_eligible_when_oos_skill_score_above_threshold
  test_ineligible_when_oos_skill_score_below_threshold  (reason="oos_skill_below_min")
  test_passes_with_warning_when_oos_skill_score_is_null

TestMedianAggregation (3):
  test_worst_fold_below_floor_blocks_activation
  test_worst_fold_at_floor_passes
  test_eligible_with_worst_fold_above_floor

TestDeflatedSkillScoreGate (2):
  test_gate_compares_against_deflated_not_raw
  test_reason_is_oos_skill_below_min_not_threshold

TestInsufficientOOSSamples (4):
  test_insufficient_samples_reason_is_distinct_from_oos_below_min
  test_insufficient_samples_message_in_detail_not_warning
  test_sufficient_samples_does_not_trigger_insufficient_reason
  test_insufficient_samples_checked_before_skill_comparison

TestOOSEligibilityDataclass (2):
  test_detail_field_exists_and_empty_by_default
  test_warning_reserved_for_pass_with_warning_cases

TestModelVersionOrmConstraintInModels (2):
  test_promoted_from_run_id_in_run_orm
  test_walk_forward_oos_skill_score_in_model_version_orm
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


# ===========================================================================
# Helpers
# ===========================================================================


def _make_model_version(
    oos_skill_score: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Build a minimal ModelVersionORM-like namespace."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        symbol="BTC/USD",
        timeframe="1h",
        walk_forward_oos_skill_score=Decimal(str(oos_skill_score)) if oos_skill_score is not None else None,
        extra=extra,
    )


# ===========================================================================
# TestCheckOOSEligibility
# ===========================================================================


class TestCheckOOSEligibility:
    """Basic gate pass/fail/warning logic."""

    def test_eligible_when_oos_skill_score_above_threshold(self) -> None:
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(oos_skill_score=0.8)
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.eligible is True
        assert result.oos_skill_score == pytest_approx(0.8)
        assert result.warning == ""
        assert result.reason == ""

    def test_ineligible_when_oos_skill_score_below_threshold(self) -> None:
        """reason must be 'oos_skill_below_min' (directional z-score proxy, not Sharpe)."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(oos_skill_score=0.3)
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.eligible is False
        assert result.reason == "oos_skill_below_min"
        assert result.oos_skill_score == pytest_approx(0.3)

    def test_passes_with_warning_when_oos_skill_score_is_null(self) -> None:
        """Pre-Cycle-5 model (NULL OOS): pass with warning, not block."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(oos_skill_score=None)
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.eligible is True
        assert result.oos_skill_score is None
        assert "walk-forward OOS gate" in result.warning


# ===========================================================================
# TestMedianAggregation
# ===========================================================================


class TestMedianAggregation:
    """Tests for worst-fold floor gate (QT5-001)."""

    def test_worst_fold_below_floor_blocks_activation(self) -> None:
        """Model passes median gate but worst fold is catastrophic."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.55,  # deflated median > 0.5 threshold
            extra={"walk_forward": {"oos_skill_worst": -0.2, "status": "ok"}},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=0.5, min_worst_fold_skill_score=0.0
        )

        assert result.eligible is False
        assert result.reason == "worst_fold_below_floor"
        assert result.worst_fold_skill_score == pytest_approx(-0.2)

    def test_worst_fold_at_floor_passes(self) -> None:
        """Worst fold exactly at floor (0.0 >= 0.0) must pass."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.55,
            extra={"walk_forward": {"oos_skill_worst": 0.0, "status": "ok"}},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=0.5, min_worst_fold_skill_score=0.0
        )

        assert result.eligible is True

    def test_eligible_with_worst_fold_above_floor(self) -> None:
        """Normal case: median and worst fold both pass."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.6,
            extra={"walk_forward": {"oos_skill_worst": 0.3, "status": "ok"}},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=0.5, min_worst_fold_skill_score=0.0
        )

        assert result.eligible is True
        assert result.reason == ""


# ===========================================================================
# TestDeflatedSkillScoreGate
# ===========================================================================


class TestDeflatedSkillScoreGate:
    """Verify the gate reads the DEFLATED MEDIAN stored in the typed column."""

    def test_gate_compares_against_deflated_not_raw(self) -> None:
        """If raw median=0.6 but deflated=0.42 < threshold=0.5, must fail."""
        from api.services.model_activation_gate import check_oos_eligibility

        # The typed column stores 0.42 (deflated); raw is diagnostic only in JSONB
        mv = _make_model_version(
            oos_skill_score=0.42,
            extra={"walk_forward": {
                "oos_skill_median_raw": 0.6,
                "oos_skill_median_deflated": 0.42,
                "oos_skill_worst": 0.2,
                "status": "ok",
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.eligible is False
        assert result.reason == "oos_skill_below_min"
        assert result.oos_skill_score == pytest_approx(0.42)

    def test_reason_is_oos_skill_below_min_not_threshold(self) -> None:
        """The failure reason string must be 'oos_skill_below_min' (not a sharpe name)."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(oos_skill_score=0.1)
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.reason == "oos_skill_below_min"
        # Old names must NOT appear
        assert result.reason != "oos_sharpe_below_min"
        assert result.reason != "oos_sharpe_below_threshold"


# ===========================================================================
# TestInsufficientOOSSamples
# ===========================================================================


class TestInsufficientOOSSamples:
    """QT5-003: insufficient_oos_samples is semantically distinct from skill failure."""

    def test_insufficient_samples_reason_is_distinct_from_oos_below_min(self) -> None:
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.5,  # above threshold -- but samples insufficient
            extra={"walk_forward": {
                "status": "insufficient_samples",
                "folds_below_threshold": [2, 4],
                "fold_trade_counts": [45, 32, 8, 60, 12],
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5, min_trades_per_fold=20)

        assert result.eligible is False
        assert result.reason == "insufficient_oos_samples"
        # Must NOT say skill is bad -- it means we can't measure skill
        assert result.reason != "oos_skill_below_min"

    def test_insufficient_samples_message_in_detail_not_warning(self) -> None:
        """CR5v2-002: ineligible paths use 'detail' field, not 'warning'."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.5,
            extra={"walk_forward": {
                "status": "insufficient_samples",
                "folds_below_threshold": [2, 4],
                "fold_trade_counts": [45, 32, 8, 60, 12],
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5, min_trades_per_fold=20)

        # detail field should have human-readable message
        assert result.eligible is False
        # warning must be empty for ineligible results
        assert result.warning == ""
        # detail must be non-empty with fold info
        assert result.detail != ""
        assert "8" in result.detail  # actual_min_fold_trades

    def test_sufficient_samples_does_not_trigger_insufficient_reason(self) -> None:
        """When all fold counts >= min_trades_per_fold, reason != insufficient_oos_samples."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.6,
            extra={"walk_forward": {
                "status": "ok",
                "fold_trade_counts": [45, 32, 50, 60, 28],
                "oos_skill_worst": 0.2,
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5, min_trades_per_fold=20)

        assert result.reason != "insufficient_oos_samples"

    def test_insufficient_samples_checked_before_skill_comparison(self) -> None:
        """Insufficient-samples check must fire even when OOS skill score would pass."""
        from api.services.model_activation_gate import check_oos_eligibility

        # OOS skill score is well above threshold but samples flag is set
        mv = _make_model_version(
            oos_skill_score=2.0,
            extra={"walk_forward": {
                "status": "insufficient_samples",
                "folds_below_threshold": [0],
                "fold_trade_counts": [3, 50, 50, 50, 50],
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5, min_trades_per_fold=20)

        assert result.eligible is False
        assert result.reason == "insufficient_oos_samples"


# ===========================================================================
# TestOOSEligibilityDataclass
# ===========================================================================


class TestOOSEligibilityDataclass:
    """CR5v2-002: OOSEligibility dataclass structure validation."""

    def test_detail_field_exists_and_empty_by_default(self) -> None:
        """OOSEligibility must have a 'detail' field defaulting to ''."""
        from api.services.model_activation_gate import OOSEligibility

        result = OOSEligibility(eligible=True, oos_skill_score=0.6, threshold=0.5)
        assert hasattr(result, "detail")
        assert result.detail == ""

    def test_warning_reserved_for_pass_with_warning_cases(self) -> None:
        """For eligible=True (null OOS), warning is non-empty; detail stays empty."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(oos_skill_score=None)
        result = check_oos_eligibility(mv, min_oos_skill_score=0.5)

        assert result.eligible is True
        assert result.warning != ""
        assert result.detail == ""

    def test_backward_compat_oos_sharpe_property(self) -> None:
        """OOSEligibility.oos_sharpe property must alias oos_skill_score."""
        from api.services.model_activation_gate import OOSEligibility

        result = OOSEligibility(eligible=True, oos_skill_score=0.7, threshold=0.5)
        assert result.oos_sharpe == 0.7  # backward-compat property

    def test_backward_compat_worst_fold_sharpe_property(self) -> None:
        """OOSEligibility.worst_fold_sharpe property must alias worst_fold_skill_score."""
        from api.services.model_activation_gate import OOSEligibility

        result = OOSEligibility(
            eligible=True, oos_skill_score=0.7, threshold=0.5,
            worst_fold_skill_score=0.3,
        )
        assert result.worst_fold_sharpe == 0.3  # backward-compat property


# ===========================================================================
# TestModelVersionOrmConstraintInModels
# ===========================================================================


class TestModelVersionOrmConstraintInModels:
    """Verify the ORM columns added to models.py exist at import time."""

    def test_promoted_from_run_id_in_run_orm(self) -> None:
        """RunORM.promoted_from_run_id column must be declared."""
        from api.db.models import RunORM
        assert hasattr(RunORM, "promoted_from_run_id")

    def test_walk_forward_oos_skill_score_in_model_version_orm(self) -> None:
        """ModelVersionORM.walk_forward_oos_skill_score column must be declared."""
        from api.db.models import ModelVersionORM
        assert hasattr(ModelVersionORM, "walk_forward_oos_skill_score")

    def test_walk_forward_oos_sharpe_column_removed(self) -> None:
        """ModelVersionORM.walk_forward_oos_sharpe must NOT exist (renamed)."""
        from api.db.models import ModelVersionORM
        assert not hasattr(ModelVersionORM, "walk_forward_oos_sharpe")

    def test_audit_event_orm_check_constraint_includes_new_types(self) -> None:
        """AuditEventORM.__table_args__ CheckConstraint must include promotion types."""
        from sqlalchemy import CheckConstraint
        from api.db.models import AuditEventORM

        # Find the event_type check constraint
        check_str = ""
        for arg in AuditEventORM.__table_args__:
            if isinstance(arg, CheckConstraint):
                check_str = str(arg.sqltext)
                break

        assert "paper_promoted_to_live" in check_str
        assert "model_oos_gate_bypassed" in check_str


# ===========================================================================
# pytest_approx shorthand
# ===========================================================================

import pytest

def pytest_approx(value: float, rel: float = 1e-6) -> Any:
    return pytest.approx(value, rel=rel)
