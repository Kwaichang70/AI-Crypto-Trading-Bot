"""
tests/unit/test_wp21b_optimize_cost.py
----------------------------------------
WP2.1b (python-backend half) unit coverage: the cost model of an OPTIMIZATION run and the
server-computed ``liveEquivalentCosts`` flag.

    WP21-T-34 (unit part)
        * ``OptimizationRunORM.run_metadata``: additive, nullable JSONB, no DB comment (it must
          match migration 020 exactly)
        * ``cost_response_fields`` over the ``run_metadata`` shape: a current run is not legacy;
          NULL / a missing key / a malformed value is legacy and NEVER raises (AC-05)
    SY-21-10
        * ``liveEquivalentCosts`` is the cost half of ``leaderboardEligible`` (same helper), is
          independent of the trade-count / confidence thresholds, and ``leaderboardEligible``
          implies it
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects.postgresql import JSONB

from api.db.models import OptimizationRunORM
from api.schemas import leaderboard_eligible
from api.services.cost_profiles import cost_response_fields, engine_semantics, stamp_cost_config
from api.services.run_persistence import run_orm_to_detail_response, run_orm_to_response
from trading.costs import (
    COST_PROFILES,
    DEFAULT_COST_PROFILE,
    get_cost_profile,
    is_leaderboard_cost_eligible,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FIELDS = {
    "cost_profile",
    "legacy_costs",
    "cost_profile_invalid",
    "legacy_exit_semantics",
    "live_equivalent_costs",
}


def _stamped(profile_id: str | None = None) -> dict[str, Any]:
    """What POST /optimize writes to ``optimization_runs.run_metadata``."""
    metadata: dict[str, Any] = {}
    profile = get_cost_profile(profile_id) if profile_id else DEFAULT_COST_PROFILE
    stamp_cost_config(metadata, profile, mode="optimize")
    return metadata


# ---------------------------------------------------------------------------
# The ORM column
# ---------------------------------------------------------------------------


class TestRunMetadataColumn:
    def test_additive_nullable_jsonb_without_a_db_comment(self) -> None:
        column = OptimizationRunORM.__table__.c.run_metadata
        assert isinstance(column.type, JSONB)
        assert column.nullable is True
        # Migration 020 sets no DB comment, no default and no server default: the model
        # must not either, or autogenerate would report drift.
        assert column.comment is None
        assert column.default is None
        assert column.server_default is None
        assert column.primary_key is False
        assert not column.foreign_keys

    def test_a_row_built_without_the_column_reads_none(self) -> None:
        row = OptimizationRunORM(id=uuid.uuid4())
        assert row.run_metadata is None


# ---------------------------------------------------------------------------
# cost_response_fields over the run_metadata shape (T-34, AC-05)
# ---------------------------------------------------------------------------


class TestCostFieldsOfAnOptimizationRun:
    def test_what_the_post_writes_is_the_spec_shape(self) -> None:
        md = _stamped()
        assert set(md) == {"cost_profile", "engine_semantics"}
        assert md["cost_profile"] == DEFAULT_COST_PROFILE.to_metadata()
        sem = md["engine_semantics"]
        assert sem == engine_semantics("optimize")
        # SY-21-11: present on backtest / optimize metadata; true from WP2.1c (false in 2.1a/2.1b)
        assert sem["backtest_circuit_breaker"] is True
        assert {"schema", "cost_model", "exit_semantics", "daily_risk"} <= set(sem)

    def test_a_current_run_is_not_legacy(self) -> None:
        fields = cost_response_fields(_stamped(), "optimize")
        assert set(fields) == _FIELDS
        assert fields["cost_profile"].id == "coinbase_advanced_1"
        assert fields["cost_profile"].source == "registry"
        assert fields["cost_profile"].fee_source == "model"
        assert fields["cost_profile"].fingerprint == DEFAULT_COST_PROFILE.fingerprint()
        assert fields["legacy_costs"] is False
        assert fields["cost_profile_invalid"] is False
        assert fields["legacy_exit_semantics"] is False
        assert fields["live_equivalent_costs"] is True

    def test_a_registry_legacy_id_is_legacy_but_not_invalid(self) -> None:
        fields = cost_response_fields(_stamped("legacy_backtest_v0"), "optimize")
        assert fields["legacy_costs"] is True
        assert fields["cost_profile_invalid"] is False
        assert fields["live_equivalent_costs"] is False
        assert fields["cost_profile"].id == "legacy_backtest_v0"
        assert fields["cost_profile"].source == "registry"  # an explicit choice, not inferred

    @pytest.mark.parametrize(
        "metadata",
        [
            pytest.param(None, id="null"),
            pytest.param({}, id="empty-object"),
            pytest.param({"engine_semantics": {"schema": 1}}, id="key-missing"),
            pytest.param({"cost_profile": None}, id="key-null"),
            pytest.param({"cost_profile": None, "engine_semantics": None}, id="both-null"),
        ],
    )
    def test_null_or_missing_means_legacy_and_is_not_flagged_invalid(self, metadata: Any) -> None:
        fields = cost_response_fields(metadata, "optimize")
        assert fields["legacy_costs"] is True
        assert fields["cost_profile_invalid"] is False  # nothing was persisted to be malformed
        assert fields["live_equivalent_costs"] is False
        profile = fields["cost_profile"]
        assert (profile.id, profile.source, profile.legacy) == (
            "legacy_backtest_v0",
            "legacy_inferred",
            True,
        )
        assert profile.round_trip_bps == "40"

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param("junk", id="string"),
            pytest.param(7, id="int"),
            pytest.param(True, id="bool"),
            pytest.param([], id="empty-list"),
            pytest.param([1, 2], id="list"),
            pytest.param({}, id="empty-object"),
            pytest.param({"junk": 1}, id="unknown-key"),
            pytest.param({"id": "coinbase_advanced_1"}, id="id-only"),
        ],
    )
    def test_a_malformed_profile_is_legacy_and_invalid_and_never_raises(self, bad: Any) -> None:
        fields = cost_response_fields({"cost_profile": bad}, "optimize")
        assert fields["legacy_costs"] is True
        assert fields["cost_profile_invalid"] is True
        assert fields["live_equivalent_costs"] is False
        assert fields["cost_profile"].source == "legacy_inferred"

    def test_tampered_numbers_with_a_registry_id_are_invalid(self) -> None:
        md = _stamped()
        md["cost_profile"]["taker_fee_bps"] = "1"  # the fingerprint no longer matches
        fields = cost_response_fields(md, "optimize")
        assert fields["cost_profile_invalid"] is True and fields["legacy_costs"] is True
        assert fields["live_equivalent_costs"] is False

    @pytest.mark.parametrize("not_a_mapping", ["text", 5, 1.5, True, [], [{"cost_profile": 1}]])
    def test_a_non_object_run_metadata_is_legacy_never_an_error(self, not_a_mapping: Any) -> None:
        fields = cost_response_fields(not_a_mapping, "optimize")
        assert fields["legacy_costs"] is True
        assert fields["cost_profile"].id == "legacy_backtest_v0"
        assert fields["legacy_exit_semantics"] is True
        assert fields["live_equivalent_costs"] is False

    @pytest.mark.parametrize("semantics", ["x", 3, [], None])
    def test_a_bad_engine_semantics_only_flags_the_exit_semantics(self, semantics: Any) -> None:
        md = _stamped()
        md["engine_semantics"] = semantics
        fields = cost_response_fields(md, "optimize")
        assert fields["legacy_exit_semantics"] is True
        assert fields["legacy_costs"] is False and fields["live_equivalent_costs"] is True

    def test_optimize_resolves_like_a_backtest(self) -> None:
        assert cost_response_fields(None, "optimize")["cost_profile"].id == "legacy_backtest_v0"
        assert cost_response_fields(None, "backtest")["cost_profile"].id == "legacy_backtest_v0"


# ---------------------------------------------------------------------------
# liveEquivalentCosts (SY-21-10)
# ---------------------------------------------------------------------------

_EXPECTED_LIVE_EQUIVALENT = {
    "coinbase_advanced_1": True,
    "coinbase_advanced_d6": True,
    "coinbase_stress_t80": False,
    "coinbase_stress_t100": False,
    "frictionless_diagnostic": False,
    "legacy_backtest_v0": False,
    "legacy_runtime_v0": False,
}


def _run(config: dict[str, Any], *, n_trades: int | None, flag: str | None) -> Any:
    if flag is not None:
        config = {**config, "backtest_metrics": {"confidence_flag": flag, "psr": 0.9}}
    return SimpleNamespace(
        id=uuid.UUID("bbbbbbbb-0000-0000-0000-000000000021"),
        run_mode="backtest",
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_trades,
    )


class TestLiveEquivalentCosts:
    def test_the_expectation_table_covers_the_whole_registry(self) -> None:
        # a newly appended registry id must get a conscious expectation here
        assert set(COST_PROFILES) == set(_EXPECTED_LIVE_EQUIVALENT)

    @pytest.mark.parametrize(("profile_id", "expected"), sorted(_EXPECTED_LIVE_EQUIVALENT.items()))
    def test_per_registry_profile(self, profile_id: str, expected: bool) -> None:
        cfg = {"cost_profile": get_cost_profile(profile_id).to_metadata()}
        assert cost_response_fields(cfg, "backtest")["live_equivalent_costs"] is expected
        assert is_leaderboard_cost_eligible(cfg) is expected  # the SAME helper

    def test_an_override_with_live_equivalent_numbers_is_not_live_equivalent(self) -> None:
        cfg = {
            "cost_profile": {
                **DEFAULT_COST_PROFILE.to_metadata(),
                "source": "override",
                "id": "custom",
            }
        }
        assert cost_response_fields(cfg, "backtest")["live_equivalent_costs"] is False

    @pytest.mark.parametrize("cfg", [None, {}, {"cost_profile": "junk"}, {"cost_profile": {}}])
    def test_absent_and_invalid_are_not_live_equivalent(self, cfg: Any) -> None:
        assert cost_response_fields(cfg, "backtest")["live_equivalent_costs"] is False

    @pytest.mark.parametrize("profile_id", sorted(_EXPECTED_LIVE_EQUIVALENT))
    @pytest.mark.parametrize(
        ("n_trades", "flag"),
        [(None, None), (3, "high"), (50, "low"), (50, "high"), (50, "medium")],
    )
    def test_leaderboard_eligible_implies_live_equivalent_costs(
        self, profile_id: str, n_trades: int | None, flag: str | None
    ) -> None:
        cfg = {"cost_profile": get_cost_profile(profile_id).to_metadata()}
        resp = run_orm_to_response(_run(cfg, n_trades=n_trades, flag=flag))
        assert resp.leaderboard_eligible is leaderboard_eligible(n_trades, flag, cfg)
        if resp.leaderboard_eligible:
            assert resp.live_equivalent_costs is True
        assert resp.live_equivalent_costs is _EXPECTED_LIVE_EQUIVALENT[profile_id]

    def test_it_is_independent_of_the_trade_count_and_confidence(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata()}
        thin = run_orm_to_response(_run(cfg, n_trades=3, flag="high"))
        assert thin.leaderboard_eligible is False  # < 10 closed trades
        assert thin.live_equivalent_costs is True  # but the COST model qualifies
        none = run_orm_to_response(_run(cfg, n_trades=None, flag=None))  # no-metrics path
        assert none.leaderboard_eligible is False and none.live_equivalent_costs is True

    def test_the_detail_response_and_the_wire_name(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata()}
        detail = run_orm_to_detail_response(_run(cfg, n_trades=None, flag=None))
        assert detail.live_equivalent_costs is True
        body = run_orm_to_response(_run(cfg, n_trades=50, flag="high")).model_dump(
            by_alias=True, mode="json"
        )
        assert body["liveEquivalentCosts"] is True and body["leaderboardEligible"] is True
        legacy = run_orm_to_response(_run({}, n_trades=50, flag="high")).model_dump(
            by_alias=True, mode="json"
        )
        assert legacy["liveEquivalentCosts"] is False and legacy["legacyCosts"] is True
