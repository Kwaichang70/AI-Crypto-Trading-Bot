"""
tests/unit/test_wp22_api_semantics.py
--------------------------------------
WP2.2 P3: the API half of the bar-time clock (hermetic: no Postgres, no network).

    WP22-T-13  marker ``engine_semantics.timestamp_clock`` and the ``legacy_timestamps`` rule
               (SY-22-08, SY-22-09); also the C4 leg of T-15 (the WP2.5 legacy marker)
    WP22-T-14  diagnostics ``lastUpdated`` tie-break by ``bar_index`` (SY-22-16)
    WP22-T-02  the deferred assertion: ``position_snapshots.snapshot_at`` of a backtest is the
               last equity point's stamp, i.e. the final bar close (SY-22-15, P2B-03 / CR-03)
"""

from __future__ import annotations

import ast
import inspect
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from api.db.models import EquitySnapshotORM, PositionSnapshotORM
from api.routers import runs as runs_router
from api.routers.optimize import OptimizeEntryResponse, OptimizeResponse
from api.schemas import RunDetailResponse, RunResponse
from api.services import cost_profiles
from api.services.cost_profiles import (
    KNOWN_TIMESTAMP_CLOCKS,
    cost_response_fields,
    engine_semantics,
    stamp_cost_config,
    timestamp_response_fields,
)
from api.services.run_persistence import (
    EmptyEquityCurveError,
    _final_bar_close,
    run_orm_to_detail_response,
    run_orm_to_response,
)
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _C,
    _clean_run_tasks,
    _post_backtest,
    app,  # fixture: requested by name in ``http`` below
)
from tests.integration.test_wp22_bar_clock import _by_type, _dod_run, _persist
from trading.backtest import BacktestRunner
from trading.clock import TIMESTAMP_CLOCK_VERSION
from trading.costs import DEFAULT_COST_PROFILE

_NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)

#: ``engine_semantics("backtest")`` at c5add9b+p2a1+p2b (before P3): the two pinned dicts.
_PRE_P3_BACKTEST = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
    "backtest_circuit_breaker": True,
}


def _run(config: Any, *, mode: str = "backtest") -> Any:
    return SimpleNamespace(
        id=uuid.UUID("aaaaaaaa-0000-0000-0000-000000000022"),
        run_mode=mode,
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=None,
    )


def _cfg(clock: Any = "wp22-1", *, present: bool = True) -> dict[str, Any]:
    sem: dict[str, Any] = {"schema": 1, "cost_model": "wp21-1"}
    if present:
        sem["timestamp_clock"] = clock
    return {"engine_semantics": sem}


# ---------------------------------------------------------------------------
# T-13  marker
# ---------------------------------------------------------------------------


class TestT13Marker:
    def test_the_marker_value_is_the_clock_module_version(self) -> None:
        assert TIMESTAMP_CLOCK_VERSION == "wp22-1"
        assert KNOWN_TIMESTAMP_CLOCKS == frozenset({"wp22-1"})

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_present_on_backtest_and_optimize(self, mode: str) -> None:
        assert engine_semantics(mode)["timestamp_clock"] == "wp22-1"

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_absent_on_paper_and_live(self, mode: str) -> None:
        assert "timestamp_clock" not in engine_semantics(mode)

    def test_the_pinned_backtest_dict_differs_by_exactly_the_one_key(self) -> None:
        assert engine_semantics("backtest") == {**_PRE_P3_BACKTEST, "timestamp_clock": "wp22-1"}
        assert engine_semantics("optimize") == {**_PRE_P3_BACKTEST, "timestamp_clock": "wp22-1"}
        assert engine_semantics("paper") == {
            k: v for k, v in _PRE_P3_BACKTEST.items() if k != "backtest_circuit_breaker"
        }

    def test_the_marker_is_independent_of_the_breaker_flag(self) -> None:
        sem = engine_semantics("backtest", backtest_circuit_breaker=False)
        assert sem["backtest_circuit_breaker"] is False
        assert sem["timestamp_clock"] == "wp22-1"

    @pytest.mark.parametrize(
        ("mode", "present"),
        [("backtest", True), ("optimize", True), ("paper", False), ("live", False)],
    )
    def test_stamp_cost_config_writes_it_only_for_backtest_and_optimize(
        self, mode: str, present: bool
    ) -> None:
        config: dict[str, Any] = {"engine_semantics": {"timestamp_clock": "inherited"}}
        stamp_cost_config(config, DEFAULT_COST_PROFILE, mode=mode)
        assert ("timestamp_clock" in config["engine_semantics"]) is present
        if present:
            assert config["engine_semantics"]["timestamp_clock"] == "wp22-1"


class TestT13LegacyFlag:
    def test_a_freshly_stamped_backtest_is_not_legacy(self) -> None:
        config: dict[str, Any] = {}
        stamp_cost_config(config, DEFAULT_COST_PROFILE, mode="backtest")
        assert timestamp_response_fields(config, "backtest") == {"legacy_timestamps": False}

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            "not-a-mapping",
            {"engine_semantics": None},
            {"engine_semantics": "wp22-1"},
            {"engine_semantics": ["timestamp_clock"]},
            _cfg(present=False),  # the key missing (a WP2.1 row)
            _cfg("wp22-0"),  # unknown version
            _cfg("WP22-1"),
            _cfg(""),
            _cfg(None),
            _cfg(True),
            _cfg(1),
            _cfg(["wp22-1"]),  # unhashable: `in frozenset` would raise
            _cfg({"v": "wp22-1"}),
        ],
        ids=lambda c: repr(c)[:40],
    )
    def test_a_backtest_without_a_known_marker_is_legacy_and_never_raises(
        self, config: Any
    ) -> None:
        assert timestamp_response_fields(config, "backtest") == {"legacy_timestamps": True}

    @pytest.mark.parametrize("mode", ["paper", "live"])
    @pytest.mark.parametrize("config", [None, {}, _cfg(present=False), _cfg("junk"), _cfg()])
    def test_paper_and_live_are_never_legacy(self, mode: str, config: Any) -> None:
        assert timestamp_response_fields(config, mode) == {"legacy_timestamps": False}

    def test_only_the_one_key_is_returned(self) -> None:
        assert set(timestamp_response_fields({}, "backtest")) == {"legacy_timestamps"}

    def test_cost_response_fields_does_not_carry_it(self) -> None:
        """It also feeds the optimize responses, where the flag has no meaning."""
        assert "legacy_timestamps" not in cost_response_fields(_cfg(), "backtest")
        assert "legacy_timestamps" not in cost_response_fields(_cfg(), "optimize")
        for model in (OptimizeEntryResponse, OptimizeResponse):
            assert "legacy_timestamps" not in model.model_fields

    def test_the_known_set_is_what_decides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cost_profiles, "KNOWN_TIMESTAMP_CLOCKS", frozenset({"wp22-1", "wp23-1"})
        )
        assert timestamp_response_fields(_cfg("wp23-1"), "backtest")["legacy_timestamps"] is False


class TestT13ResponseWiring:
    def test_the_field_is_on_run_responses_only(self) -> None:
        assert "legacy_timestamps" in RunResponse.model_fields
        assert "legacy_timestamps" in RunDetailResponse.model_fields
        assert RunResponse.model_fields["legacy_timestamps"].default is False

    @pytest.mark.parametrize("convert", [run_orm_to_response, run_orm_to_detail_response])
    def test_legacy_and_current_backtests_and_paper(self, convert: Any) -> None:
        assert convert(_run(_cfg(present=False))).legacy_timestamps is True
        assert convert(_run(_cfg())).legacy_timestamps is False
        assert convert(_run(_cfg(present=False), mode="paper")).legacy_timestamps is False
        assert convert(_run({}, mode="live")).legacy_timestamps is False

    def test_the_flag_survives_the_metrics_branch_of_run_orm_to_response(self) -> None:
        config = {
            **_cfg(present=False),
            "backtest_metrics": {"confidence_flag": "high", "psr": 0.9},
        }
        resp = run_orm_to_response(_run(config))
        assert resp.confidence_flag == "high"
        assert resp.legacy_timestamps is True

    def test_it_serialises_camel_case(self) -> None:
        body = run_orm_to_response(_run(_cfg(present=False))).model_dump(by_alias=True)
        assert body["legacyTimestamps"] is True
        body = run_orm_to_response(_run(_cfg())).model_dump(by_alias=True)
        assert body["legacyTimestamps"] is False

    def test_a_tampered_value_never_500s_a_get(self) -> None:
        resp = run_orm_to_response(_run(_cfg(["x"])))
        assert resp.legacy_timestamps is True


# ---------------------------------------------------------------------------
# T-14  lastUpdated tie-break
# ---------------------------------------------------------------------------


class _CapturingDb:
    """Serves ``get_diagnostics``: the run, the equity query (captured), two counts."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, stmt: Any) -> Any:
        self.statements.append(stmt)
        result = MagicMock()
        if len(self.statements) == 1:
            result.scalar_one_or_none.return_value = SimpleNamespace(
                status="stopped", run_mode="backtest", config={"symbols": ["BTC/USDT"]}
            )
        elif len(self.statements) == 2:
            result.scalar_one_or_none.return_value = None
        else:
            result.scalar.return_value = 0
        return result


async def _capture_equity_statement(run_id: uuid.UUID | None = None) -> Any:
    db = _CapturingDb()
    await runs_router.get_diagnostics(run_id or uuid.uuid4(), db)  # type: ignore[arg-type]
    return db.statements[1]


class TestT14LastUpdatedTieBreak:
    async def test_the_query_orders_by_timestamp_desc_then_bar_index_desc(self) -> None:
        stmt = await _capture_equity_statement()
        sql = str(stmt.compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]
        order_by = sql.split("ORDER BY", 1)[1].split("LIMIT", 1)[0]
        cols = [c.strip() for c in order_by.split(",")]
        assert cols == ["equity_snapshots.timestamp DESC", "equity_snapshots.bar_index DESC"]
        assert "LIMIT" in sql

    async def test_among_rows_with_the_max_timestamp_the_highest_bar_index_wins(self) -> None:
        """Execute the endpoint's own statement against an in-memory table."""
        run_id = uuid.uuid4()
        stmt = await _capture_equity_statement(run_id)
        t = datetime(2026, 3, 10, 12, 0)
        engine = create_engine("sqlite://")
        EquitySnapshotORM.__table__.create(engine)  # type: ignore[attr-defined]

        def row(bar_index: int, ts: datetime, equity: str) -> EquitySnapshotORM:
            return EquitySnapshotORM(
                id=100 + bar_index,  # sqlite does not autoincrement a BigInteger PK
                run_id=run_id,
                equity=Decimal(equity),
                cash=Decimal("0"),
                unrealised_pnl=Decimal("0"),
                realised_pnl=Decimal("0"),
                drawdown_pct=Decimal("0"),
                bar_index=bar_index,
                timestamp=ts,
            )

        with Session(engine) as session:
            # Inserted out of order on purpose; bar 3/4 share the max stamp (mark + fill).
            session.add_all(
                [
                    row(4, t, "1004"),
                    row(0, t - timedelta(hours=2), "1000"),
                    row(3, t, "1003"),
                    row(2, t - timedelta(hours=1), "1002"),
                    row(1, t - timedelta(hours=1), "1001"),
                ]
            )
            session.commit()
            picked = session.execute(stmt).scalar_one()
        assert picked.bar_index == 4
        assert picked.equity == Decimal("1004")

    async def test_the_endpoint_returns_the_stamp_of_the_picked_row(self) -> None:
        db = AsyncMock()
        run = SimpleNamespace(status="stopped", run_mode="backtest", config={})
        snap = SimpleNamespace(equity=Decimal("5"), drawdown_pct=Decimal("0"), timestamp=_NOW)
        results = []
        for kind in ("run", "snap", "count", "count"):
            r = MagicMock()
            r.scalar_one_or_none.return_value = run if kind == "run" else snap
            r.scalar.return_value = 0
            results.append(r)
        db.execute.side_effect = results
        body = await runs_router.get_diagnostics(uuid.uuid4(), db)
        assert body["lastUpdated"] == _NOW.isoformat()


# ---------------------------------------------------------------------------
# T-02 (deferred from P2b)  snapshot_at == the final bar close
# ---------------------------------------------------------------------------


class TestT02SnapshotAt:
    async def test_open_positions_with_an_empty_curve_raise_and_never_read_the_wall(self) -> None:
        run = await _dod_run()
        broken = SimpleNamespace(**{**vars(run.result), "equity_curve": []})
        run.result = broken
        with pytest.raises(EmptyEquityCurveError, match="no equity curve"):
            await _persist(run)

    async def test_no_open_positions_and_an_empty_curve_do_not_raise(self) -> None:
        run = await _dod_run()
        run.result = SimpleNamespace(**{**vars(run.result), "equity_curve": []})
        assert run.runner.last_portfolio is not None
        run.runner.last_portfolio._position_snapshots.clear()
        rec, _ = await _persist(run)
        assert _by_type(rec, PositionSnapshotORM) == []

    async def test_position_snapshot_at_is_the_last_equity_stamp_and_the_last_close(self) -> None:
        run = await _dod_run()
        rec, _ = await _persist(run)
        snaps = _by_type(rec, PositionSnapshotORM)
        assert snaps, "the T-02 scenario ends with one open lot: the test would be vacuous"
        last_close = run.closes[-1]
        assert run.curve[-1].timestamp == last_close
        assert {s["snapshot_at"] for s in snaps} == {last_close}
        equity = _by_type(rec, EquitySnapshotORM)
        assert max(e["timestamp"] for e in equity) == last_close
        assert all(s["snapshot_at"] < datetime(2026, 4, 1, tzinfo=UTC) for s in snaps)

    async def test_the_paper_path_still_stamps_wall_time(self) -> None:
        src = (
            Path(inspect.getsourcefile(runs_router) or "").parents[1]
            / "services"
            / "run_persistence.py"
        )
        tree = ast.parse(src.read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "persist_paper_results"
        )
        text = ast.unparse(fn)
        assert "snapshot_at=now" in text and "datetime.now" in text


# ---------------------------------------------------------------------------
# WP22-P3-R2-CR-01  the empty-curve invariant breach is a 500, not a 400
# ---------------------------------------------------------------------------

_GENERIC_500 = "Backtest execution failed. See server logs for details."


def _open_position() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USD",
        quantity=Decimal("1"),
        average_entry_price=Decimal("100"),
        current_price=Decimal("101"),
        unrealised_pnl=Decimal("1"),
        realised_pnl=Decimal("0"),
        total_fees_paid=Decimal("0"),
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def http(request: pytest.FixtureRequest) -> Generator[_C, None, None]:
    """The hermetic create-run harness of ``test_wp21_cost_api`` (mocked DB, no network)."""
    client = _C(request.getfixturevalue("app"))
    yield client
    client.close()


class TestP3R2Cr01EmptyCurveIs500:
    def test_it_is_a_runtime_error_and_not_a_value_error(self) -> None:
        assert issubclass(EmptyEquityCurveError, RuntimeError)
        assert not issubclass(EmptyEquityCurveError, ValueError)

    def test_the_helper_raises_the_dedicated_error(self) -> None:
        with pytest.raises(EmptyEquityCurveError, match="no equity curve"):
            _final_bar_close([])

    def test_the_endpoint_maps_it_to_500_with_a_generic_detail_and_a_rollback(
        self, http: _C
    ) -> None:
        """Real ``persist_backtest_results`` + real dispatch: open lot, emptied curve."""
        original_run = BacktestRunner.run

        async def _run_then_break(self: BacktestRunner, *a: Any, **kw: Any) -> Any:
            result = await original_run(self, *a, **kw)
            assert self.last_portfolio is not None
            self.last_portfolio._position_snapshots["BTC/USD"] = _open_position()
            return SimpleNamespace(**{**vars(result), "equity_curve": []})

        with patch.object(BacktestRunner, "run", _run_then_break):
            resp = _post_backtest(http)
        assert resp.status_code == 500, resp.text
        assert resp.json()["detail"] == _GENERIC_500
        assert "equity curve" not in resp.text and "data quality" not in resp.text
        http.db.rollback.assert_awaited()  # the run row and everything added are discarded
        http.db.commit.assert_not_awaited()
        assert http.created_run().status == "error"

    def test_control_a_value_error_from_the_same_site_is_still_a_400(self, http: _C) -> None:
        """Proves the test above discriminates: ``ValueError`` takes the 400 branch."""
        with patch(
            "api.services.run_persistence._final_bar_close",
            side_effect=ValueError("boom"),
        ):
            original_run = BacktestRunner.run

            async def _run_with_lot(self: BacktestRunner, *a: Any, **kw: Any) -> Any:
                result = await original_run(self, *a, **kw)
                assert self.last_portfolio is not None
                self.last_portfolio._position_snapshots["BTC/USD"] = _open_position()
                return result

            with patch.object(BacktestRunner, "run", _run_with_lot):
                resp = _post_backtest(http)
        assert resp.status_code == 400, resp.text
        http.db.rollback.assert_awaited()
