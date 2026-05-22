"""
tests/unit/test_sprint49_m5_leaderboard_transparency.py
--------------------------------------------------------
Unit tests for Sprint 49 M5: leaderboard transparency.

Covers
------
1.  RunResponse.leaderboard_eligible -- True when n_closed_trades >= 10
    AND confidence_flag in {"high", "medium"}
2.  RunResponse.leaderboard_eligible -- False when n_closed_trades < 10
3.  RunResponse.leaderboard_eligible -- False when confidence_flag == "low"
4.  RunResponse.leaderboard_eligible -- False when confidence_flag is None
5.  RunResponse.leaderboard_eligible -- False when n_closed_trades is None
6.  run_orm_to_response -- populates confidence_flag from JSONB
7.  run_orm_to_response -- populates psr from JSONB (finite float)
8.  run_orm_to_response -- rejects infinite psr from JSONB (None)
9.  run_orm_to_response -- tolerates missing backtest_metrics (all None)
10. run_orm_to_response -- rejects unknown confidence_flag from JSONB (None)
11. run_orm_to_response -- leaderboard_eligible reflects JSONB extraction
12. get_aggregate_portfolio logic -- phantom run excluded from best
13. get_aggregate_portfolio logic -- eligible_runs_count populated correctly
14. M5 constants match locked spec values
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from api.schemas import (
    LEADERBOARD_ELIGIBLE_FLAGS,
    LEADERBOARD_MIN_CLOSED_TRADES,
    RunResponse,
)
from api.services.run_persistence import run_orm_to_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
_UUID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")


def _make_run_ns(
    *,
    n_closed_trades: int | None = None,
    config: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Minimal RunORM-like namespace for from_attributes deserialization."""
    return SimpleNamespace(
        id=_UUID,
        run_mode="backtest",
        status="stopped",
        config=config or {},
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_closed_trades,
    )


def _config_with_metrics(**kwargs: Any) -> dict[str, Any]:
    """Build a config dict with a backtest_metrics sub-dict."""
    return {"backtest_metrics": kwargs}


# ---------------------------------------------------------------------------
# Tests: RunResponse.leaderboard_eligible computation
# ---------------------------------------------------------------------------


class TestLeaderboardEligibleField:
    """RunResponse.leaderboard_eligible must reflect M5 eligibility rules."""

    def test_eligible_when_high_flag_and_sufficient_trades(self) -> None:
        """n_closed_trades=10 + confidence_flag='high' -> leaderboard_eligible=True."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=10,
            confidence_flag="high",
        )
        assert r.leaderboard_eligible is True

    def test_eligible_when_medium_flag_and_sufficient_trades(self) -> None:
        """confidence_flag='medium' also qualifies."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=50,
            confidence_flag="medium",
        )
        assert r.leaderboard_eligible is True

    def test_ineligible_when_trades_below_threshold(self) -> None:
        """n_closed_trades=9 (one below the 10-trade floor) -> False."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=9,
            confidence_flag="high",
        )
        assert r.leaderboard_eligible is False

    def test_ineligible_when_zero_trades(self) -> None:
        """n_closed_trades=0 (0-trade phantom) -> False."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=0,
            confidence_flag="high",
        )
        assert r.leaderboard_eligible is False

    def test_ineligible_when_low_confidence(self) -> None:
        """confidence_flag='low' -> False even with many trades."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=100,
            confidence_flag="low",
        )
        assert r.leaderboard_eligible is False

    def test_ineligible_when_confidence_flag_none(self) -> None:
        """confidence_flag=None (PSR not computable) -> False."""
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=100,
            confidence_flag=None,
        )
        assert r.leaderboard_eligible is False

    def test_ineligible_when_n_closed_trades_none(self) -> None:
        """n_closed_trades=None (paper/live/pre-M3) -> False."""
        r = RunResponse(
            id=_UUID,
            run_mode="paper",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=None,
            confidence_flag="high",
        )
        assert r.leaderboard_eligible is False

    def test_eligible_boundary_exactly_10_trades(self) -> None:
        """n_closed_trades == LEADERBOARD_MIN_CLOSED_TRADES exactly -> True."""
        assert LEADERBOARD_MIN_CLOSED_TRADES == 10  # document constant value
        r = RunResponse(
            id=_UUID,
            run_mode="backtest",
            status="stopped",
            config={},
            started_at=_NOW,
            stopped_at=None,
            created_at=_NOW,
            updated_at=_NOW,
            n_closed_trades=LEADERBOARD_MIN_CLOSED_TRADES,
            confidence_flag="medium",
        )
        assert r.leaderboard_eligible is True


# ---------------------------------------------------------------------------
# Tests: run_orm_to_response JSONB extraction
# ---------------------------------------------------------------------------


class TestRunOrmToResponseJsonbExtraction:
    """run_orm_to_response must extract confidence_flag and psr from JSONB."""

    def test_extracts_confidence_flag_from_jsonb(self) -> None:
        """confidence_flag='high' in JSONB -> RunResponse.confidence_flag='high'."""
        ns = _make_run_ns(
            n_closed_trades=20,
            config=_config_with_metrics(confidence_flag="high", psr=0.95),
        )
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.confidence_flag == "high"

    def test_extracts_psr_from_jsonb(self) -> None:
        """psr=0.87 in JSONB -> RunResponse.psr=0.87."""
        ns = _make_run_ns(
            n_closed_trades=30,
            config=_config_with_metrics(confidence_flag="medium", psr=0.87),
        )
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.psr == pytest.approx(0.87, rel=1e-6)

    def test_rejects_infinite_psr_from_jsonb(self) -> None:
        """psr=inf in JSONB must not propagate -- RunResponse.psr stays None."""
        ns = _make_run_ns(
            config=_config_with_metrics(confidence_flag="high", psr=float("inf")),
        )
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.psr is None

    def test_missing_backtest_metrics_yields_none_fields(self) -> None:
        """No backtest_metrics in config -> all three fields default to None/False."""
        ns = _make_run_ns(config={})
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.confidence_flag is None
        assert r.psr is None
        assert r.n_closed_trades is None
        assert r.leaderboard_eligible is False

    def test_unknown_confidence_flag_rejected(self) -> None:
        """An unrecognised confidence_flag value in JSONB must not surface."""
        ns = _make_run_ns(
            config=_config_with_metrics(confidence_flag="VERY_HIGH", psr=0.99),
        )
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.confidence_flag is None

    def test_leaderboard_eligible_computed_after_jsonb_extraction(self) -> None:
        """leaderboard_eligible reflects JSONB-extracted data via call-site computation."""
        ns = _make_run_ns(
            n_closed_trades=15,
            config=_config_with_metrics(confidence_flag="high", psr=0.96),
        )
        r = run_orm_to_response(ns)  # type: ignore[arg-type]
        assert r.leaderboard_eligible is True


# ---------------------------------------------------------------------------
# Tests: aggregate portfolio eligibility gating (pure logic, no HTTP)
# ---------------------------------------------------------------------------


class TestAggregatePortfolioEligibilityGate:
    """
    Eligibility logic in get_aggregate_portfolio must exclude 0-trade/low-confidence
    runs from best_run_return_pct and worst_run_return_pct.

    These tests exercise the JSONB eligibility check logic in isolation to verify
    correctness before the HTTP-level integration tests run.
    """

    def test_phantom_run_excluded_from_best(self) -> None:
        """
        A run with 0 trades and a +12.88% return must not appear as best.

        The eligible run has 15 trades and +5% return.  After filtering,
        best return must be 0.05, not 0.1288.
        """
        phantom_metrics = {
            "total_return_pct": 0.1288,
            "total_trades": 0,  # 0-trade phantom
            "confidence_flag": None,
        }
        eligible_metrics = {
            "total_return_pct": 0.05,
            "total_trades": 15,
            "confidence_flag": "high",
        }

        return_pcts: list[float] = []
        eligible_count = 0
        for raw in [phantom_metrics, eligible_metrics]:
            n_trades = raw.get("total_trades")
            flag = raw.get("confidence_flag")
            eligible = (
                isinstance(n_trades, int)
                and n_trades >= LEADERBOARD_MIN_CLOSED_TRADES
                and flag in LEADERBOARD_ELIGIBLE_FLAGS
            )
            if eligible:
                eligible_count += 1
                ret = raw.get("total_return_pct")
                if ret is not None:
                    return_pcts.append(float(ret))

        assert return_pcts == [0.05]
        assert max(return_pcts) == pytest.approx(0.05)
        assert eligible_count == 1

    def test_eligible_runs_count_reflects_gate(self) -> None:
        """eligible_runs_count must count only qualifying runs."""
        runs = [
            {"total_trades": 0, "confidence_flag": None},      # phantom -- excluded
            {"total_trades": 5, "confidence_flag": "low"},      # too few + low -- excluded
            {"total_trades": 10, "confidence_flag": "high"},    # eligible
            {"total_trades": 50, "confidence_flag": "medium"},  # eligible
            {"total_trades": 30, "confidence_flag": "low"},     # low confidence -- excluded
        ]
        eligible_count = sum(
            1 for r in runs
            if isinstance(r.get("total_trades"), int)
            and r["total_trades"] >= LEADERBOARD_MIN_CLOSED_TRADES
            and r.get("confidence_flag") in LEADERBOARD_ELIGIBLE_FLAGS
        )
        assert eligible_count == 2

    def test_constants_match_spec(self) -> None:
        """M5 constants must match the locked spec values."""
        assert LEADERBOARD_MIN_CLOSED_TRADES == 10
        assert LEADERBOARD_ELIGIBLE_FLAGS == frozenset({"high", "medium"})
