"""WP2.5 / P3: API and persistence of the metric-integrity contract (hermetic, no Postgres).

Binding spec: ``reports/vp2-wp2.5/synthesis-spec.md`` (SY-25-02/03/12/13/15/16/19/20/22/26,
user decisions UD-1 and UD-3).  Test ids are ``WP25-T-26`` .. ``WP25-T-34`` (the backfill half
of T-31 lives in ``test_wp25_backfill_guard.py``).

    T-26  marker ``engine_semantics.metrics`` (backtest + optimize only), exact-key diff
    T-27  ``legacy_metrics`` truth table and wiring (list, detail, optimize POST/GET-one/GET-list)
    T-28  response fields (SY-25-12), stub-safe ``build_backtest_metrics``, tampered JSONB
    T-29  leaderboard: metrics half + ``legacy_metrics_runs_excluded`` (UD-1)
    T-30  ML ``walk_forward.metrics`` label and the pre-WP2.5 activation warning (UD-3)
    T-32  paper / live unchanged (exact dicts, source pins, promotion gate pin)
    T-33  optimize: non-finite entry metrics are persisted and returned as ``None``
    T-34  alembic head 020, ORM metadata unchanged, log hygiene
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

import api.routers.optimize as optimize_module
import api.routers.runs as runs_module
from api.db.models import OptimizationRunORM, RunORM
from api.routers.optimize import OptimizationRunSummary, OptimizeResponse
from api.schemas import (
    AggregatePortfolioResponse,
    BacktestMetricsResponse,
    RunDetailResponse,
    RunResponse,
    is_leaderboard_metrics_eligible,
    leaderboard_eligible,
)
from api.services import cost_profiles
from api.services.cost_profiles import (
    KNOWN_METRICS_VERSIONS,
    cost_response_fields,
    engine_semantics,
    metrics_response_fields,
    stamp_cost_config,
)
from api.services.model_activation_gate import check_oos_eligibility
from api.services.run_persistence import (
    build_backtest_metrics,
    run_orm_to_detail_response,
    run_orm_to_response,
)
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _C,
    _OPT_BARS,
    _OPT_BODY,
    _clean_run_tasks,
    _hdr,
    _paper_source,
    _post_backtest,
    _promote,
    app,  # fixture: requested by name in ``c`` below
)
from tests.unit.test_sprint50_cycle6_oos_backtest import _minimal_result
from tests.unit.test_wp21h_ml_profile import _result as _ml_result
from tests.unit.test_wp21h_ml_profile import _train
from trading import metrics as metrics_module
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile
from trading.metrics import METRICS_VERSION

_REPO = Path(__file__).resolve().parents[2]
_NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
_MARKER = "wp25-1"
_SENTINEL = "WP25SENTINEL-MARKER-VALUE"


@pytest.fixture()
def c(app: Any) -> Generator[_C, None, None]:  # noqa: F811
    client = _C(app)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: ``engine_semantics("backtest")`` as of WP2.3 (before WP2.5): the pinned dict WITHOUT ``metrics``.
_PRE_WP25_BACKTEST: dict[str, Any] = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
    "backtest_circuit_breaker": True,
    "timestamp_clock": "wp22-1",
    "bar_alignment": "wp23-1",
}


def _run_row(config: Any, *, mode: str = "backtest", n_trades: int | None = None) -> Any:
    return SimpleNamespace(
        id=uuid.UUID("aaaaaaaa-0000-0000-0000-000000000025"),
        run_mode=mode,
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_trades,
    )


def _sem(**sem: Any) -> dict[str, Any]:
    return {"engine_semantics": {"schema": 1, **sem}}


def _stub_without_wp25_attrs() -> Any:
    """A result-like object that predates the five WP2.5 fields (getattr guard)."""
    data = _minimal_result().model_dump()
    for name in (
        "trading_start_at",
        "trading_end_at",
        "periods_per_year",
        "metric_sampling",
        "carried_slots",
        "oos_start_at",
        "metrics_version",
    ):
        data.pop(name)
    return SimpleNamespace(**data)


def _stored_metrics(**overrides: Any) -> dict[str, Any]:
    """What ``persist_backtest_results`` stores: ``build_backtest_metrics`` -> JSON dump."""
    result = _minimal_result(
        trading_start_at=datetime(2024, 1, 3, 2, 0, tzinfo=UTC),
        trading_end_at=datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
        periods_per_year=8766.0,
        metric_sampling="calendar_asof",
        carried_slots=3,
        metrics_version=_MARKER,
    )
    stored = build_backtest_metrics(result).model_dump(mode="json")
    stored.update(overrides)
    return stored


# ===========================================================================
# T-26  the marker
# ===========================================================================


class TestT26Marker:
    def test_the_version_constants(self) -> None:
        assert METRICS_VERSION == _MARKER
        assert metrics_module.KNOWN_METRICS_VERSIONS == frozenset({_MARKER})
        # one owner: the API re-exports the formula module's set, it does not copy it
        assert cost_profiles.KNOWN_METRICS_VERSIONS is metrics_module.KNOWN_METRICS_VERSIONS
        assert "KNOWN_METRICS_VERSIONS" in cost_profiles.__all__
        assert "metrics_response_fields" in cost_profiles.__all__

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_present_on_backtest_and_optimize(self, mode: str) -> None:
        assert engine_semantics(mode)["metrics"] == _MARKER

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_absent_on_paper_and_live(self, mode: str) -> None:
        assert "metrics" not in engine_semantics(mode)

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_the_dict_differs_from_the_wp23_dict_by_exactly_the_one_key(self, mode: str) -> None:
        sem = engine_semantics(mode)
        assert set(sem) - set(_PRE_WP25_BACKTEST) == {"metrics"}
        assert {k: v for k, v in sem.items() if k != "metrics"} == _PRE_WP25_BACKTEST
        # neither earlier marker is bumped (SY-25-03)
        assert sem["timestamp_clock"] == "wp22-1"
        assert sem["bar_alignment"] == "wp23-1"

    @pytest.mark.parametrize(
        ("mode", "present"),
        [("backtest", True), ("optimize", True), ("paper", False), ("live", False)],
    )
    def test_stamp_cost_config_overwrites_an_inherited_value(
        self, mode: str, present: bool
    ) -> None:
        config: dict[str, Any] = {"engine_semantics": {"metrics": "inherited"}}
        stamp_cost_config(config, DEFAULT_COST_PROFILE, mode=mode)
        assert ("metrics" in config["engine_semantics"]) is present
        if present:
            assert config["engine_semantics"]["metrics"] == _MARKER

    def test_a_promoted_live_config_has_no_metrics_key(self, c: _C) -> None:
        source = _paper_source(
            uuid.uuid4(),
            engine_semantics={"schema": 1, "metrics": _MARKER, "bar_alignment": "wp23-1"},
        )
        resp, _ = _promote(c, source)
        assert resp.status_code == 201, resp.text
        assert "metrics" not in c.created_run().config["engine_semantics"]

    def test_a_created_backtest_persists_the_marker_with_the_numbers(self, c: _C) -> None:
        """SY-25-26: the marker and the per-bar numbers arrive in the same run."""
        resp = _post_backtest(c)
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert cfg["engine_semantics"]["metrics"] == _MARKER
        stored = cfg["backtest_metrics"]
        assert stored["metric_sampling"] == "calendar_asof"
        assert stored["trading_start_at"] is not None
        assert stored["periods_per_year"] == pytest.approx(8766.0)
        body = resp.json()
        assert body["legacyMetrics"] is False
        assert body["backtestMetrics"]["metricSampling"] == "calendar_asof"

    def test_the_marker_is_only_in_the_one_stamp(self) -> None:
        """No second writer: ``"metrics"`` is assigned in ``engine_semantics`` and nowhere else."""
        src = (_REPO / "apps/api/services/cost_profiles.py").read_text()
        assert src.count('semantics["metrics"] = METRICS_VERSION') == 1


# ===========================================================================
# T-27  legacy_metrics
# ===========================================================================


class TestT27LegacyFlag:
    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_a_known_marker_is_not_legacy(self, mode: str) -> None:
        assert metrics_response_fields(_sem(metrics=_MARKER), mode) == {"legacy_metrics": False}

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            "not-a-mapping",
            {"engine_semantics": None},
            {"engine_semantics": "wp25-1"},
            {"engine_semantics": ["metrics"]},
            {"engine_semantics": {}},
            _sem(),  # the key missing (a WP2.3 row)
            _sem(metrics=None),
            _sem(metrics="wp24-9"),  # unknown version
            _sem(metrics="WP25-1"),
            _sem(metrics=""),
            _sem(metrics=1),
            _sem(metrics=True),
            _sem(metrics=["wp25-1"]),  # unhashable: ``in frozenset`` would raise
            _sem(metrics={"v": "wp25-1"}),
        ],
        ids=lambda cfg: repr(cfg)[:40],
    )
    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_unmarked_or_malformed_is_legacy_and_never_raises(self, config: Any, mode: str) -> None:
        assert metrics_response_fields(config, mode) == {"legacy_metrics": True}

    @pytest.mark.parametrize("mode", ["paper", "live", "unknown", ""])
    @pytest.mark.parametrize(
        "config", [None, {}, _sem(), _sem(metrics="junk"), _sem(metrics=_MARKER)]
    )
    def test_paper_live_and_unknown_modes_are_never_legacy(self, mode: str, config: Any) -> None:
        """No mapping of an unknown mode to backtest (contrast ``cost_response_fields``)."""
        assert metrics_response_fields(config, mode) == {"legacy_metrics": False}

    def test_a_single_symbol_unmarked_backtest_is_flagged(self) -> None:
        """No single-symbol exemption: the defect hit every run (contrast bar_alignment)."""
        cfg = {"symbols": ["BTC/USD"], **_sem(bar_alignment="wp23-1")}
        assert metrics_response_fields(cfg, "backtest") == {"legacy_metrics": True}
        assert cost_profiles.bar_alignment_response_fields(cfg, "backtest") == {
            "legacy_bar_alignment": False
        }

    def test_only_the_one_key_is_returned_and_cost_fields_do_not_carry_it(self) -> None:
        assert set(metrics_response_fields({}, "backtest")) == {"legacy_metrics"}
        assert "legacy_metrics" not in cost_response_fields(_sem(), "backtest")
        assert "legacy_metrics" not in cost_response_fields(_sem(), "optimize")

    def test_the_known_set_is_what_decides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cost_profiles, "KNOWN_METRICS_VERSIONS", frozenset({_MARKER, "wp26-1"}))
        assert (
            metrics_response_fields(_sem(metrics="wp26-1"), "backtest")["legacy_metrics"] is False
        )

    def test_it_never_logs(self) -> None:
        with capture_logs() as logs:
            for cfg in (
                None,
                _sem(metrics=_SENTINEL),
                _sem(metrics=_MARKER),
                {"engine_semantics": 3},
            ):
                metrics_response_fields(cfg, "backtest")
                metrics_response_fields(cfg, "optimize")
        assert logs == []

    # --- wiring -----------------------------------------------------------

    def test_the_field_defaults_to_false_and_has_the_camel_alias(self) -> None:
        field = RunResponse.model_fields["legacy_metrics"]
        assert field.default is False
        assert "legacy_metrics" in RunDetailResponse.model_fields  # inherited
        for model in (OptimizeResponse, OptimizationRunSummary):
            assert model.model_fields["legacy_metrics"].default is False
        dumped = run_orm_to_response(_run_row({})).model_dump(by_alias=True, mode="json")
        assert dumped["legacyMetrics"] is True

    def test_list_and_detail_set_the_flag(self) -> None:
        marked = _run_row({**_sem(metrics=_MARKER)})
        legacy = _run_row({**_sem()})
        assert run_orm_to_response(marked).legacy_metrics is False
        assert run_orm_to_detail_response(marked).legacy_metrics is False
        assert run_orm_to_response(legacy).legacy_metrics is True
        assert run_orm_to_detail_response(legacy).legacy_metrics is True
        # the early-return path (no backtest_metrics at all) carries it too
        assert "backtest_metrics" not in legacy.config

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_paper_and_live_runs_are_not_flagged(self, mode: str) -> None:
        assert run_orm_to_response(_run_row({}, mode=mode)).legacy_metrics is False
        assert run_orm_to_detail_response(_run_row({}, mode=mode)).legacy_metrics is False

    def test_it_is_independent_of_the_other_legacy_flags(self) -> None:
        # a wp23 row: timestamps and alignment are current, metrics are not
        row = _run_row(
            {
                "symbols": ["A/USD", "B/USD"],
                **_sem(timestamp_clock="wp22-1", bar_alignment="wp23-1"),
            }
        )
        resp = run_orm_to_response(row)
        assert (resp.legacy_timestamps, resp.legacy_bar_alignment, resp.legacy_metrics) == (
            False,
            False,
            True,
        )
        # and a row with the marker only: the other two flag, this one does not
        row2 = _run_row({"symbols": ["A/USD", "B/USD"], **_sem(metrics=_MARKER)})
        resp2 = run_orm_to_response(row2)
        assert (resp2.legacy_timestamps, resp2.legacy_bar_alignment, resp2.legacy_metrics) == (
            True,
            True,
            False,
        )

    # --- optimize ---------------------------------------------------------

    def test_optimize_post_persists_the_marker_and_returns_the_flag(self, c: _C) -> None:
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest", new=AsyncMock(return_value=_OPT_BARS)
        ):
            resp = c.client.post("/api/v1/optimize", json=_OPT_BODY)
        assert resp.status_code == 200, resp.text
        orm = next(
            call.args[0]
            for call in c.db.add.call_args_list
            if isinstance(call.args[0], OptimizationRunORM)
        )
        assert orm.run_metadata["engine_semantics"]["metrics"] == _MARKER
        assert resp.json()["legacyMetrics"] is False

    @staticmethod
    def _opt_row(run_metadata: Any) -> Any:
        return SimpleNamespace(
            id=uuid.uuid4(),
            strategy_name="ma_crossover",
            symbols=["BTC/USD"],
            timeframe="1h",
            rank_by="sharpe_ratio",
            total_combinations=1,
            completed_combinations=1,
            failed_combinations=0,
            elapsed_seconds=1.5,
            created_at=_NOW,
            run_metadata=run_metadata,
            entries=[SimpleNamespace(rank=1, params={"a": 1}, metrics={"sharpe_ratio": 1.0})],
        )

    @pytest.mark.parametrize(
        ("run_metadata", "expected"),
        [
            ({**_sem(metrics=_MARKER)}, False),
            ({**_sem()}, True),
            (None, True),  # NULL run_metadata (pre-migration-020 row)
            ({"engine_semantics": "junk"}, True),
            ({"engine_semantics": {"metrics": _SENTINEL}}, True),
        ],
        ids=["marked", "unmarked", "null", "malformed", "unknown-version"],
    )
    async def test_optimize_get_one_and_list_carry_the_flag(
        self, run_metadata: Any, expected: bool
    ) -> None:
        row = self._opt_row(run_metadata)
        one = AsyncMock()
        one_result = MagicMock()
        one_result.scalar_one_or_none.return_value = row
        one.execute = AsyncMock(return_value=one_result)
        detail = await optimize_module.get_optimization_run(row.id, one)
        assert detail.legacy_metrics is expected

        many = AsyncMock()
        many_result = MagicMock()
        many_result.scalars.return_value.all.return_value = [row]
        many.execute = AsyncMock(return_value=many_result)
        listing = await optimize_module.list_optimization_runs(many)
        assert [r.legacy_metrics for r in listing] == [expected]
        assert listing[0].model_dump(by_alias=True)["legacyMetrics"] is expected


# ===========================================================================
# T-28  response fields
# ===========================================================================


class TestT28ResponseFields:
    def test_the_five_fields_round_trip_through_jsonb(self) -> None:
        stored = _stored_metrics()
        # JSONB holds ISO text; json round trip proves the stored shape is plain JSON
        stored = json.loads(json.dumps(stored))
        assert isinstance(stored["trading_start_at"], str)
        detail = run_orm_to_detail_response(
            _run_row({**_sem(metrics=_MARKER), "backtest_metrics": stored})
        )
        bm = detail.backtest_metrics
        assert bm is not None
        assert bm.trading_start_at == datetime(2024, 1, 3, 2, 0, tzinfo=UTC)
        assert bm.trading_start_at is not None and bm.trading_start_at.utcoffset() == timedelta(0)
        assert bm.trading_end_at == datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        assert bm.periods_per_year == 8766.0
        assert bm.metric_sampling == "calendar_asof"
        assert bm.carried_slots == 3
        wire = bm.model_dump(by_alias=True, mode="json")
        for key in (
            "tradingStartAt",
            "tradingEndAt",
            "periodsPerYear",
            "metricSampling",
            "carriedSlots",
        ):
            assert key in wire

    def test_oos_start_at_and_the_metrics_version_are_not_exposed(self) -> None:
        wire = BacktestMetricsResponse.model_fields
        assert "oos_start_at" not in wire
        assert "metrics_version" not in wire
        assert "oosStartAt" not in build_backtest_metrics(_minimal_result()).model_dump(
            by_alias=True
        )

    def test_a_legacy_stored_dict_validates_with_numbers_verbatim(self) -> None:
        stored = _stored_metrics()
        for key in (
            "trading_start_at",
            "trading_end_at",
            "periods_per_year",
            "metric_sampling",
            "carried_slots",
        ):
            stored.pop(key)
        stored.update(
            sharpe_ratio=1.2345, sortino_ratio=2.5, psr=0.97, n_observations=999, cagr=0.5
        )
        detail = run_orm_to_detail_response(_run_row({"backtest_metrics": stored}))
        bm = detail.backtest_metrics
        assert bm is not None
        assert (bm.sharpe_ratio, bm.sortino_ratio, bm.psr, bm.n_observations, bm.cagr) == (
            1.2345,
            2.5,
            0.97,
            999,
            0.5,
        )
        assert bm.trading_start_at is None and bm.trading_end_at is None
        assert bm.periods_per_year is None and bm.metric_sampling is None and bm.carried_slots == 0
        assert detail.legacy_metrics is True  # labelled, never recomputed

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("trading_start_at", "<script>alert(1)</script>"),
            ("trading_end_at", _SENTINEL),
            ("metric_sampling", _SENTINEL),
            ("carried_slots", -1),
            ("periods_per_year", "NaN-text"),
        ],
    )
    def test_a_tampered_value_is_a_parse_warning_and_the_run_is_still_served(
        self, key: str, value: Any
    ) -> None:
        stored = _stored_metrics()
        stored[key] = value
        row = _run_row({**_sem(metrics=_MARKER), "backtest_metrics": stored})
        with capture_logs() as logs:
            detail = run_orm_to_detail_response(row)
        assert detail.backtest_metrics is None  # not echoed
        assert detail.legacy_metrics is False
        events = [e for e in logs if e["event"] == "runs.backtest_metrics_parse_error"]
        assert len(events) == 1
        assert _SENTINEL not in repr(events) and "script" not in repr(events)
        # list conversion is unaffected by the bad detail payload
        assert run_orm_to_response(row).legacy_metrics is False

    def test_the_parse_warning_carries_no_exception_text(self) -> None:
        """Pydantic's error text embeds the offending input; only the type name is logged."""
        stored = _stored_metrics(trading_start_at=_SENTINEL)
        with capture_logs() as logs:
            run_orm_to_detail_response(_run_row({"backtest_metrics": stored}))
        (event,) = [e for e in logs if e["event"] == "runs.backtest_metrics_parse_error"]
        assert "exc_info" not in event
        assert event["error_type"] == "ValidationError"

    def test_a_stub_result_without_the_attributes_does_not_raise(self) -> None:
        built = build_backtest_metrics(_stub_without_wp25_attrs())
        assert built.trading_start_at is None and built.trading_end_at is None
        assert built.periods_per_year is None and built.metric_sampling is None
        assert built.carried_slots == 0

    def test_a_mock_result_degrades_to_none_instead_of_raising(self) -> None:
        """A ``MagicMock`` result answers every ``getattr``: the typed readers fail closed."""
        stub = MagicMock(spec=[])  # no attributes at all -> AttributeError -> defaults
        for name, value in _stub_without_wp25_attrs().__dict__.items():
            setattr(stub, name, value)
        stub.trading_start_at = MagicMock()
        stub.periods_per_year = MagicMock()
        stub.metric_sampling = MagicMock()
        stub.carried_slots = MagicMock()
        built = build_backtest_metrics(stub)
        assert built.trading_start_at is None and built.periods_per_year is None
        assert built.metric_sampling is None and built.carried_slots == 0

    @pytest.mark.parametrize("bad", [-1, -(10**9)])
    def test_a_negative_carried_slots_degrades_to_zero(self, bad: int) -> None:
        """Hardening: the ``value < 0`` branch of ``_carried_slots_attr`` (``ge=0`` would raise)."""
        stub = _minimal_result().model_copy(update={"carried_slots": bad})
        assert build_backtest_metrics(stub).carried_slots == 0

    def test_an_int_too_large_for_a_float_is_none_not_an_overflow(self) -> None:
        """SEC-12: ``math.isfinite(10**400)`` raises OverflowError; the reader returns None."""
        built = build_backtest_metrics(
            _minimal_result().model_copy(update={"periods_per_year": 10**400})
        )
        assert built.periods_per_year is None

    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, True, "8766", None])
    def test_periods_per_year_is_finite_or_none(self, bad: Any) -> None:
        built = build_backtest_metrics(
            _minimal_result().model_copy(update={"periods_per_year": bad})
        )
        assert built.periods_per_year is None
        json.dumps(built.model_dump(mode="json"), allow_nan=False)

    def test_a_finite_dump_is_json_valid_even_for_n_below_two(self) -> None:
        result = _minimal_result(sharpe_ratio=0.0, n_observations=1, periods_per_year=365.25)
        dumped = build_backtest_metrics(result).model_dump(mode="json")
        json.dumps(dumped, allow_nan=False)
        assert dumped["periods_per_year"] == 365.25

    def test_the_descriptions_no_longer_claim_the_point_based_n(self) -> None:
        for name in ("n_observations", "total_bars", "psr"):
            description = BacktestMetricsResponse.model_fields[name].description or ""
            assert "equity curve length - 1" not in description
        assert "per-bar" in (
            BacktestMetricsResponse.model_fields["n_observations"].description or ""
        )
        src = (_REPO / "apps/api/schemas.py").read_text()
        assert "equity curve length - 1" not in src


# ===========================================================================
# T-29  leaderboard
# ===========================================================================


def _eligible_cfg(*, marked: bool, profile: Any = DEFAULT_COST_PROFILE) -> dict[str, Any]:
    cfg: dict[str, Any] = {"cost_profile": profile.to_metadata()}
    if marked:
        cfg["engine_semantics"] = {"schema": 1, "metrics": _MARKER}
    return cfg


class TestT29Leaderboard:
    def test_unmarked_is_not_eligible_and_marked_is(self) -> None:
        assert leaderboard_eligible(10, "high", _eligible_cfg(marked=False)) is False
        assert leaderboard_eligible(10, "high", _eligible_cfg(marked=True)) is True

    def test_each_half_still_decides(self) -> None:
        good = _eligible_cfg(marked=True)
        assert leaderboard_eligible(9, "high", good) is False  # trades
        assert leaderboard_eligible(50, "low", good) is False  # confidence
        stress = _eligible_cfg(marked=True, profile=get_cost_profile("coinbase_stress_t80"))
        assert leaderboard_eligible(50, "high", stress) is False  # cost
        assert leaderboard_eligible(50, "high", None) is False

    @pytest.mark.parametrize(
        "marker", [None, "", "wp24-9", 1, ["wp25-1"], {"v": 1}, True], ids=repr
    )
    def test_a_malformed_marker_fails_closed(self, marker: Any) -> None:
        cfg = _eligible_cfg(marked=True)
        cfg["engine_semantics"]["metrics"] = marker
        assert is_leaderboard_metrics_eligible(cfg) is False
        assert leaderboard_eligible(50, "high", cfg) is False

    @pytest.mark.parametrize(
        "config", [None, {}, {"engine_semantics": None}, {"engine_semantics": []}]
    )
    def test_the_metrics_half_never_raises(self, config: Any) -> None:
        assert is_leaderboard_metrics_eligible(config) is False

    def test_run_response_uses_the_same_rule(self) -> None:
        for marked, expected in ((True, True), (False, False)):
            cfg = {
                **_eligible_cfg(marked=marked),
                "backtest_metrics": {"confidence_flag": "high", "psr": 0.97},
            }
            assert run_orm_to_response(_run_row(cfg, n_trades=40)).leaderboard_eligible is expected

    async def test_summary_counts_the_excluded_runs_and_keeps_the_cost_counter(self) -> None:
        from api.routers.portfolio import get_aggregate_portfolio

        def _metrics(ret: float, trades: int = 40) -> dict[str, Any]:
            return {
                "backtest_metrics": {
                    "total_trades": trades,
                    "confidence_flag": "high",
                    "total_return_pct": ret,
                }
            }

        d6 = get_cost_profile("coinbase_advanced_d6")
        configs = [
            # eligible (marked, live-equivalent cost)
            {"initial_capital": "1000", **_eligible_cfg(marked=True), **_metrics(0.10)},
            {"initial_capital": "1000", **_eligible_cfg(marked=True, profile=d6), **_metrics(0.20)},
            # pass trades + confidence + cost, lack the marker -> legacy_metrics_runs_excluded
            {"initial_capital": "1000", **_eligible_cfg(marked=False), **_metrics(9.0)},
            {"initial_capital": "1000", **_eligible_cfg(marked=False, profile=d6), **_metrics(8.0)},
            {
                "initial_capital": "1000",
                **_eligible_cfg(marked=False),
                "engine_semantics": {"metrics": _SENTINEL},
                **_metrics(7.0),
            },
            # cost exclusion comes FIRST, marked or not (existing semantics unmoved)
            {
                "initial_capital": "1000",
                **_eligible_cfg(marked=True, profile=get_cost_profile("coinbase_stress_t80")),
                **_metrics(6.0),
            },
            {
                "initial_capital": "1000",
                **_eligible_cfg(marked=False, profile=get_cost_profile("frictionless_diagnostic")),
                **_metrics(5.0),
            },
            {"initial_capital": "1000", **_metrics(4.0)},  # legacy: no profile, no marker
            # a 4th cost-excluded config: the two counters must differ (3 vs 4), or a
            # swapped counter branch would survive
            {
                "initial_capital": "1000",
                **_eligible_cfg(marked=True, profile=get_cost_profile("coinbase_stress_t80")),
                **_metrics(2.0),
            },
            # thresholds only: counted nowhere
            {"initial_capital": "1000", **_eligible_cfg(marked=True), **_metrics(3.0, trades=3)},
        ]
        db = AsyncMock()
        status_result = MagicMock()
        status_result.all.return_value = []
        trade_result = MagicMock()
        trade_result.one.return_value = SimpleNamespace(total=0, wins=0, losses=0, pnl=0, fees=0)
        cfg_result = MagicMock()
        cfg_result.all.return_value = [(cfg,) for cfg in configs]
        db.execute = AsyncMock(side_effect=[status_result, trade_result, cfg_result])
        resp = await get_aggregate_portfolio(db=db)
        assert resp.eligible_runs_count == 2
        assert resp.best_run_return_pct == pytest.approx(0.20)
        assert resp.worst_run_return_pct == pytest.approx(0.10)
        assert resp.legacy_metrics_runs_excluded == 3
        assert resp.legacy_cost_runs_excluded == 4
        wire = resp.model_dump(by_alias=True)
        assert wire["legacyMetricsRunsExcluded"] == 3 and wire["legacyCostRunsExcluded"] == 4

    def test_the_counter_defaults_to_zero(self) -> None:
        assert AggregatePortfolioResponse.model_fields["legacy_metrics_runs_excluded"].default == 0

    def test_both_call_sites_still_use_the_one_helper(self) -> None:
        from api.routers import portfolio as portfolio_router

        src = inspect.getsource(portfolio_router.get_aggregate_portfolio)
        assert "leaderboard_eligible(" in src
        assert "LEADERBOARD_MIN_CLOSED_TRADES" not in src
        assert "KNOWN_METRICS_VERSIONS" not in src  # the marker rule lives in schemas only


# ===========================================================================
# T-30  ML gate
# ===========================================================================

_WF_OK = {
    "metric_type": "realized_oos_sharpe",
    "schema_version": 3,
    "status": "ok",
    "oos_skill_worst": 1.5,
}


def _model(score: float | None, wf: dict[str, Any] | None) -> Any:
    return SimpleNamespace(
        id=uuid.uuid4(),
        symbol="BTC/USD",
        timeframe="1h",
        walk_forward_oos_skill_score=Decimal(str(score)) if score is not None else None,
        extra=None if wf is None else {"walk_forward": wf},
    )


class TestT30MlGate:
    async def test_a_new_model_block_carries_the_label_and_keeps_its_schema(self) -> None:
        from tests.unit.test_wp21h_ml_profile import _ml

        ml = _ml()
        version, _ = await _train(
            ml, ml._TrainedModel(result=_ml_result(), cost_profile=DEFAULT_COST_PROFILE)
        )
        wf = version.extra["walk_forward"]
        assert wf["metrics"] == _MARKER
        assert wf["schema_version"] == 3
        assert wf["metric_type"] == "realized_oos_sharpe"
        assert wf["oos_measurement"] == "oos_window"

    def test_the_label_name_matches_the_marker_key_in_source(self) -> None:
        from api.routers import ml as ml_mod

        src = inspect.getsource(ml_mod.train_model)
        assert '"metrics": METRICS_VERSION' in src
        assert '"schema_version": 3' in src

    def test_an_unmarked_model_is_gated_normally_and_passes_with_the_fixed_warning(self) -> None:
        result = check_oos_eligibility(_model(1.5, dict(_WF_OK)), min_oos_skill_score=1.0)
        assert result.eligible is True and result.reason == "" and result.detail == ""
        assert result.warning == (
            "walk-forward Sharpe predates WP2.5 (metrics wp25-1); retrain via POST /ml/train"
        )

    def test_an_unmarked_model_below_the_threshold_is_still_rejected_without_a_warning(
        self,
    ) -> None:
        """UD-3: thresholds are unchanged and nothing passes because of the label."""
        result = check_oos_eligibility(_model(0.99, dict(_WF_OK)), min_oos_skill_score=1.0)
        assert result.eligible is False and result.reason == "oos_skill_below_min"
        assert result.warning == ""

    def test_the_threshold_default_is_still_one(self) -> None:
        from api.config import get_settings

        assert get_settings().min_oos_skill_score == pytest.approx(1.0)

    def test_a_marked_model_has_no_warning(self) -> None:
        result = check_oos_eligibility(
            _model(1.5, {**_WF_OK, "metrics": _MARKER}), min_oos_skill_score=1.0
        )
        assert result.eligible is True and result.warning == ""

    @pytest.mark.parametrize("label", [None, "", "wp24-9", 1, ["wp25-1"], {"v": 1}, True], ids=repr)
    def test_a_malformed_label_warns_like_a_missing_one(self, label: Any) -> None:
        result = check_oos_eligibility(
            _model(1.5, {**_WF_OK, "metrics": label}), min_oos_skill_score=1.0
        )
        assert result.eligible is True and "predates WP2.5" in result.warning

    def test_the_warning_never_echoes_the_stored_label(self) -> None:
        result = check_oos_eligibility(
            _model(1.5, {**_WF_OK, "metrics": _SENTINEL}), min_oos_skill_score=1.0
        )
        assert _SENTINEL not in result.warning

    def test_the_other_passthroughs_are_unchanged(self) -> None:
        legacy_proxy = check_oos_eligibility(
            _model(0.2, {"metric_type": "directional_zscore_proxy"}), min_oos_skill_score=1.0
        )
        assert legacy_proxy.eligible is True and "z-score proxy" in legacy_proxy.warning
        null_score = check_oos_eligibility(_model(None, None), min_oos_skill_score=1.0)
        assert (
            null_score.eligible is True and "before the walk-forward OOS gate" in null_score.warning
        )

    def test_the_gate_logs_no_label_value(self) -> None:
        with capture_logs() as logs:
            check_oos_eligibility(
                _model(1.5, {**_WF_OK, "metrics": _SENTINEL}), min_oos_skill_score=1.0
            )
        assert _SENTINEL not in repr(logs)

    def test_the_none_to_zero_fold_path_is_unchanged(self) -> None:
        from api.routers import ml as ml_mod

        src = inspect.getsource(ml_mod._train_model_with_wf_sync)
        compact = " ".join(src.split())
        assert "result.oos_sharpe if result.oos_sharpe is not None else 0.0" in compact

    async def test_a_fold_window_has_one_return_per_bar_gap_free(self) -> None:
        """The gate's fold Sharpe is per-bar: ``oos_n_returns == len(test window)``."""
        from tests.unit.test_wp23_runner import _gen, _run_ma

        bars = _gen(600, 20251)
        for oos_start in (450, 500, 580):
            _, res = await _run_ma(bars, symbols=("AAA/EUR",), oos=oos_start)
            assert res.oos_n_returns == 600 - oos_start
            assert res.metrics_version == _MARKER


# ===========================================================================
# T-32  paper / live unchanged
# ===========================================================================

_PAPER_LIVE_SEMANTICS = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
}


def _calls_and_strings(func: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def _function(path: str, name: str) -> ast.AST:
    tree = ast.parse((_REPO / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


class TestT32PaperLiveUnchanged:
    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_the_exact_paper_and_live_dicts(self, mode: str) -> None:
        assert engine_semantics(mode) == _PAPER_LIVE_SEMANTICS
        config: dict[str, Any] = {}
        stamp_cost_config(config, DEFAULT_COST_PROFILE, mode=mode)
        assert config["engine_semantics"] == _PAPER_LIVE_SEMANTICS

    def test_persist_paper_results_does_not_touch_the_wp25_surface(self) -> None:
        names = _calls_and_strings(
            _function("apps/api/services/run_persistence.py", "persist_paper_results")
        )
        forbidden = {
            "metrics_response_fields",
            "build_backtest_metrics",
            "METRICS_VERSION",
            "KNOWN_METRICS_VERSIONS",
            "legacy_metrics",
            "trading_start_at",
            "periods_per_year",
            "metric_sampling",
            "backtest_metrics",
        }
        assert names & forbidden == set()

    def test_the_orchestrator_has_no_wp25_reference(self) -> None:
        for path in (
            "apps/api/services/run_orchestrator.py",
            "apps/api/services/promotion_gate.py",
        ):
            text = (_REPO / path).read_text()
            for token in (
                "METRICS_VERSION",
                "KNOWN_METRICS_VERSIONS",
                "legacy_metrics",
                "metrics_response_fields",
            ):
                assert token not in text, (path, token)

    def test_the_promotion_gate_never_reads_a_metric(self) -> None:
        """A future ``require Sharpe >= x`` gate must come with a version check (BD-07)."""
        gate = ast.parse((_REPO / "apps/api/services/promotion_gate.py").read_text())
        gate_names = _calls_and_strings(gate)
        for token in ("backtest_metrics", "sharpe_ratio", "sharpe", "psr", "confidence_flag"):
            assert token not in gate_names, token
        eligibility = _calls_and_strings(
            _function("apps/api/routers/runs.py", "get_promotion_eligibility")
        )
        for token in ("backtest_metrics", "sharpe_ratio", "psr", "confidence_flag"):
            assert token not in eligibility, token
        promote = _function("apps/api/routers/runs.py", "promote_to_live")
        names = _calls_and_strings(promote)
        for token in ("sharpe_ratio", "psr", "confidence_flag", "metrics_response_fields"):
            assert token not in names, token
        # `backtest_metrics` appears once, in the tuple of keys the promotion DROPS
        drops = [
            n
            for n in ast.walk(promote)
            if isinstance(n, ast.Constant) and n.value == "backtest_metrics"
        ]
        assert len(drops) == 1


# ===========================================================================
# T-33  optimize: non-finite scrub
# ===========================================================================


def _poisoned_run(original: Any) -> Any:
    async def _run(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        # mutate the in-memory entries the way a +inf CAGR / Sortino would arrive
        result.entries[0].metrics["cagr"] = math.inf
        result.entries[0].metrics["calmar_ratio"] = math.inf
        result.entries[0].metrics["sortino_ratio"] = -math.inf
        result.entries[0].metrics["probabilistic_sharpe"] = math.nan
        return result

    return _run


class TestT33OptimizeScrub:
    def _post(self, c: _C) -> tuple[Any, Any]:
        from trading.optimizer import ParameterOptimizer

        seen: dict[str, Any] = {}
        original = ParameterOptimizer.run

        async def _spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = await _poisoned_run(original)(self, *args, **kwargs)
            seen["entries"] = result.entries
            return result

        with (
            patch.object(ParameterOptimizer, "run", _spy),
            patch(
                "api.routers.optimize._fetch_bars_for_backtest",
                new=AsyncMock(return_value=_OPT_BARS),
            ),
        ):
            resp = c.client.post("/api/v1/optimize", json=_OPT_BODY)
        return resp, seen

    def test_non_finite_entry_metrics_are_persisted_and_returned_as_none(self, c: _C) -> None:
        resp, seen = self._post(c)
        assert resp.status_code == 200, resp.text
        first = resp.json()["entries"][0]["metrics"]
        for key in ("cagr", "calmar_ratio", "sortino_ratio", "probabilistic_sharpe"):
            assert key in first and first[key] is None
        persisted = [
            call.args[0].metrics
            for call in c.db.add.call_args_list
            if type(call.args[0]).__name__ == "OptimizationEntryORM"
        ]
        assert len(persisted) == len(seen["entries"])
        json.dumps(persisted, allow_nan=False)  # what JSONB would see
        assert persisted[0]["cagr"] is None and persisted[0]["calmar_ratio"] is None
        # the in-memory entries (and so the ranking) are untouched
        assert seen["entries"][0].metrics["cagr"] == math.inf

    def test_the_post_response_entries_are_scrubbed_at_python_level(self, c: _C) -> None:
        """CR-02: JSON mode maps inf to null anyway, so pin the model objects themselves."""
        real = optimize_module.OptimizeEntryResponse
        built: list[Any] = []

        def _spy(**kwargs: Any) -> Any:
            entry = real(**kwargs)
            built.append(entry)
            return entry

        with patch.object(optimize_module, "OptimizeEntryResponse", _spy):
            resp, seen = self._post(c)
        assert resp.status_code == 200, resp.text
        assert seen["entries"][0].metrics["cagr"] == math.inf  # the source is poisoned
        assert built and built[0].metrics["cagr"] is None
        for entry in built:
            json.dumps(entry.metrics, allow_nan=False)

    def test_the_post_response_wires_the_metrics_flag(self, c: _C) -> None:
        """CR-03: a run stamped WITHOUT the marker must come back ``legacyMetrics: true``."""
        real_stamp = optimize_module.stamp_cost_config

        def _stamp_without_marker(config: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            real_stamp(config, *args, **kwargs)
            config["engine_semantics"].pop("metrics", None)

        with patch.object(optimize_module, "stamp_cost_config", _stamp_without_marker):
            resp, _ = self._post(c)
        assert resp.status_code == 200, resp.text
        assert resp.json()["legacyMetrics"] is True

    def test_finite_values_are_byte_identical(self, c: _C) -> None:
        resp, seen = self._post(c)
        assert resp.status_code == 200, resp.text
        persisted = [
            call.args[0].metrics
            for call in c.db.add.call_args_list
            if type(call.args[0]).__name__ == "OptimizationEntryORM"
        ]
        for entry, stored in zip(seen["entries"], persisted, strict=True):
            for key, value in entry.metrics.items():
                if isinstance(value, float) and not math.isfinite(value):
                    assert stored[key] is None
                else:
                    assert stored[key] == value
                    assert type(stored[key]) is type(value)
            assert list(stored) == list(entry.metrics)  # no key dropped or reordered

    def test_the_helper_leaves_none_and_ints_alone(self) -> None:
        scrub = optimize_module._scrub_non_finite
        src = {"a": 1.5, "b": None, "c": 3, "d": math.inf, "e": -math.inf, "f": math.nan, "g": 0.0}
        out = scrub(src)
        assert out == {"a": 1.5, "b": None, "c": 3, "d": None, "e": None, "f": None, "g": 0.0}
        assert src["d"] == math.inf  # the input is not mutated
        assert type(out["c"]) is int

    async def test_get_one_scrubs_what_it_reads(self) -> None:
        row = TestT27LegacyFlag._opt_row({**_sem(metrics=_MARKER)})
        row.entries = [
            SimpleNamespace(
                rank=1, params={}, metrics={"sharpe_ratio": 1.0, "cagr": math.inf, "x": None}
            )
        ]
        db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        db.execute = AsyncMock(return_value=result)
        detail = await optimize_module.get_optimization_run(row.id, db)
        assert detail.entries[0].metrics == {"sharpe_ratio": 1.0, "cagr": None, "x": None}
        json.dumps(detail.model_dump(mode="json"), allow_nan=False)

    def test_the_scrub_is_confined_to_the_persistence_and_response_edge(self) -> None:
        src = (_REPO / "packages/trading/optimizer.py").read_text()
        assert "_scrub_non_finite" not in src


# ===========================================================================
# T-34  migrations and log hygiene
# ===========================================================================


class TestT34HeadAndHygiene:
    def test_alembic_head_is_still_020(self) -> None:
        from alembic.script import ScriptDirectory

        heads = ScriptDirectory(str(_REPO / "infra/alembic")).get_heads()
        assert heads == ["020"]

    def test_the_orm_metadata_gained_no_wp25_column(self) -> None:
        for table in (RunORM.__table__, OptimizationRunORM.__table__):
            for column in table.columns:
                assert "metrics_version" not in column.name
                assert "legacy_metrics" not in column.name
        assert "run_metadata" in OptimizationRunORM.__table__.columns
        assert "config" in RunORM.__table__.columns

    def test_no_wp25_name_in_the_db_package(self) -> None:
        for path in (_REPO / "apps/api/db").rglob("*.py"):
            assert "METRICS_VERSION" not in path.read_text(), path

    def test_the_new_code_paths_log_no_marker_value_or_payload(self, c: _C) -> None:
        """Create backtest + optimize POST: no log line carries the marker or a metric payload."""
        with capture_logs() as logs:
            resp = _post_backtest(c)
            with patch(
                "api.routers.optimize._fetch_bars_for_backtest",
                new=AsyncMock(return_value=_OPT_BARS),
            ):
                opt = c.client.post("/api/v1/optimize", json=_OPT_BODY)
        assert resp.status_code == 201 and opt.status_code == 200
        dumped = repr(logs)
        assert _MARKER not in dumped
        assert "legacy_metrics" not in dumped
        assert "engine_semantics" not in dumped

    def test_the_flag_and_eligibility_helpers_never_log(self) -> None:
        with capture_logs() as logs:
            for cfg in (None, _sem(metrics=_SENTINEL), _eligible_cfg(marked=True)):
                metrics_response_fields(cfg, "backtest")
                is_leaderboard_metrics_eligible(cfg)
                leaderboard_eligible(50, "high", cfg)
        assert logs == []

    def test_the_new_source_has_no_logging_call_with_a_marker_or_exception_text(self) -> None:
        for path in (
            "apps/api/services/cost_profiles.py",
            "apps/api/services/model_activation_gate.py",
            "apps/api/routers/optimize.py",
        ):
            tree = ast.parse((_REPO / path).read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {"debug", "info", "warning", "error", "critical", "exception"}
                ):
                    continue
                text = ast.unparse(node)
                for token in (
                    "metrics_version",
                    "METRICS_VERSION",
                    "wf_metrics",
                    "marker",
                    "legacy_metrics",
                    "str(exc)",
                ):
                    assert token not in text, (path, token, text)

    def test_the_runs_module_is_the_only_caller_pattern_for_the_flag(self) -> None:
        """The flag is wired where the other three are, and nowhere a client could set it."""
        assert "legacy_metrics" not in inspect.getsource(runs_module.create_run)
        for model in (RunResponse, OptimizeResponse, OptimizationRunSummary):
            assert model.model_fields["legacy_metrics"].default is False
        assert (
            "legacy_metrics"
            not in __import__(
                "api.schemas", fromlist=["RunCreateRequest"]
            ).RunCreateRequest.model_fields
        )
        assert "legacy_metrics" not in optimize_module.OptimizeRequest.model_fields


def test_known_versions_is_the_frozenset_of_the_one_constant() -> None:
    assert KNOWN_METRICS_VERSIONS == frozenset({METRICS_VERSION})
