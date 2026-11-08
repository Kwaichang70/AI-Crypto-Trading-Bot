"""WP2.3 / P4: API wiring of the bar-alignment contract (hermetic: no Postgres, no network).

Binding spec: ``reports/vp2-wp2.3/synthesis-spec.md`` (SY-23-07/08/14/16, I-8/I-11, WP23-T-21,
T-22), the P3 critic hand-offs (CR-05, CF-23-20), P0 security SEC-02 and QA N2.  Test ids are
``WP23-P4-NN``.

    P4-01..05  marker ``engine_semantics.bar_alignment`` (backtest + optimize only), no
               ``timestamp_clock`` bump, the exact-key diff of the pinned dicts
    P4-06..09  ``legacy_bar_alignment`` truth table, never raises, response wiring
    P4-10..17  422 mapping of ``BarAlignmentError`` (runs + optimize), handler order, CF-36
    P4-18..24  ``requested_window``: clamp, injected ``now``, passed to runner and optimizer
    P4-25..29  persisted ``config["bar_alignment"]``; ``presence_digest`` (QA N2); total_bars
    P4-30..37  SEC-02 bounds (symbols, start, range) at, below and above the limit
    P4-38..46  CF-23-20 duplicate symbols: request edge, resume, recovery, promote
    P4-47     the ML gate stays windowless and byte-identical
    P4-49..52 v2 folds: `now` read once before the fetch (SEC-01), datetime overflow (SEC-05)
"""

from __future__ import annotations

import ast
import json
import uuid
from collections.abc import Generator, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from structlog.testing import capture_logs

import api.routers.optimize as optimize_module
import api.routers.runs as runs_module
from api.db.models import OptimizationRunORM
from api.schemas import BacktestMetricsResponse, RunCreateRequest, RunDetailResponse, RunResponse
from api.services import cost_profiles
from api.services.cost_profiles import (
    KNOWN_BAR_ALIGNMENTS,
    KNOWN_TIMESTAMP_CLOCKS,
    MAX_BACKTEST_BARS,
    MAX_BACKTEST_SYMBOLS,
    MIN_BACKTEST_START,
    bar_alignment_http_error,
    bar_alignment_response_fields,
    check_backtest_request_bounds,
    engine_semantics,
    requested_bar_window,
)
from api.services.run_orchestrator import _RUN_TASKS
from api.services.run_persistence import (
    DUPLICATE_SYMBOLS_REASON,
    has_duplicate_symbols,
    run_orm_to_detail_response,
    run_orm_to_response,
)
from common.types import TimeFrame
from data.bar_alignment import BAR_ALIGNMENT_CODES, BAR_ALIGNMENT_VERSION, BarAlignmentError
from tests._cf36_logging import ConfigureFn
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.conftest import make_bars
from tests.integration import test_wp18a_resume_endpoint as resume_mod
from tests.integration.test_wp18a_resume_endpoint import resume_app  # noqa: F401  (fixture)
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _BACKTEST,
    _C,
    _LIVE,
    _OPT_BODY,
    _PAPER,
    CONFIRM_TOKEN,
    _clean_run_tasks,
    _hdr,
    _paper_source,
    _promote,
    app,  # fixture: requested by name in ``c`` below
)
from trading.backtest import BacktestRunner
from trading.optimizer import ParameterOptimizer

_REPO = Path(__file__).resolve().parents[2]
_BTC, _ETH = "BTC/USD", "ETH/USD"
_NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)

#: ``engine_semantics("backtest")`` at WP2.2 (before P4): the pinned dict WITHOUT the new key.
_PRE_P4_BACKTEST: dict[str, Any] = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
    "backtest_circuit_breaker": True,
    "timestamp_clock": "wp22-1",
}


@pytest.fixture()
def c(app: Any) -> Generator[_C, None, None]:  # noqa: F811
    client = _C(app)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bars(symbol: str, n: int = 120, drop: Iterable[int] = ()) -> list[Any]:
    """``n`` hourly bars from 2024-01-01T00:00Z (``make_bars``), minus the ``drop`` indices."""
    gone = set(drop)
    return [b for i, b in enumerate(make_bars(n, symbol=symbol)) if i not in gone]


def _post_bt(c: _C, bars: dict[str, list[Any]], **overrides: Any) -> Any:
    body = {**_BACKTEST, **overrides}
    with patch("api.routers.runs._fetch_bars_for_backtest", return_value=bars):
        return c.client.post("/api/v1/runs", json=body, headers=_hdr())


def _post_opt(c: _C, bars: dict[str, list[Any]], **overrides: Any) -> Any:
    with patch("api.routers.optimize._fetch_bars_for_backtest", new=AsyncMock(return_value=bars)):
        return c.client.post("/api/v1/optimize", json={**_OPT_BODY, **overrides})


def _single(n: int = 120, drop: Iterable[int] = ()) -> dict[str, list[Any]]:
    return {_BTC: _bars(_BTC, n, drop)}


def _opt_run(c: _C) -> Any:
    for call in c.db.add.call_args_list:
        if isinstance(call.args[0], OptimizationRunORM):
            return call.args[0]
    raise AssertionError("no OptimizationRunORM was added")


def _floats(obj: object) -> list[float]:
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, dict):
        return [f for v in obj.values() for f in _floats(v)]
    if isinstance(obj, (list, tuple)):
        return [f for v in obj for f in _floats(v)]
    return []


def _cfg(**sem: Any) -> dict[str, Any]:
    return {"symbols": [_BTC, _ETH], "engine_semantics": {"schema": 1, **sem}}


def _run_row(config: Any, *, mode: str = "backtest") -> Any:
    return SimpleNamespace(
        id=uuid.UUID("aaaaaaaa-0000-0000-0000-000000000023"),
        run_mode=mode,
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=None,
    )


# ---------------------------------------------------------------------------
# P4-01..05  the marker (SY-23-16, T-22)
# ---------------------------------------------------------------------------


class TestMarker:
    def test_p4_01_the_marker_is_the_leaf_version(self) -> None:
        assert BAR_ALIGNMENT_VERSION == "wp23-1"
        assert KNOWN_BAR_ALIGNMENTS == frozenset({"wp23-1"})

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_p4_02_present_on_backtest_and_optimize(self, mode: str) -> None:
        assert engine_semantics(mode)["bar_alignment"] == "wp23-1"

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_p4_03_absent_on_paper_and_live(self, mode: str) -> None:
        assert "bar_alignment" not in engine_semantics(mode)

    def test_p4_04_timestamp_clock_is_not_bumped_and_the_dict_gains_exactly_one_key(self) -> None:
        assert engine_semantics("backtest")["timestamp_clock"] == "wp22-1"
        assert KNOWN_TIMESTAMP_CLOCKS == frozenset({"wp22-1"})
        for mode in ("backtest", "optimize"):
            sem = engine_semantics(mode)
            # WP2.5 (SY-25-03): ``metrics`` is the second added key; WP2.4 (SY-24-07): the third.
            assert sem == {
                **_PRE_P4_BACKTEST,
                "bar_alignment": "wp23-1",
                "metrics": "wp25-1",
                "mtf_context": "wp24-1",
            }
            assert set(sem) - set(_PRE_P4_BACKTEST) == {"bar_alignment", "metrics", "mtf_context"}

    def test_p4_05_the_stale_comment_is_gone(self) -> None:
        src = (_REPO / "apps/api/services/cost_profiles.py").read_text()
        assert "WP2.3 bumps the version" not in src
        assert "does NOT bump" in src

    def test_p4_05b_http_marker_per_mode(self, c: _C) -> None:
        resp = _post_bt(c, _single())
        assert resp.status_code == 201, resp.text
        assert c.created_run().config["engine_semantics"]["bar_alignment"] == "wp23-1"
        assert c.created_run().config["engine_semantics"]["timestamp_clock"] == "wp22-1"
        c.db.add.reset_mock()
        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp = c.client.post("/api/v1/runs", json=_PAPER, headers=_hdr())
        assert resp.status_code == 201, resp.text
        assert "bar_alignment" not in c.created_run().config["engine_semantics"]
        assert "bar_alignment" not in c.created_run().config
        c.db.add.reset_mock()
        with patch("api.routers.runs._run_live_engine", new=AsyncMock()):
            resp = c.client.post(
                "/api/v1/runs",
                json=_LIVE,
                headers=_hdr(**{"X-Live-Confirm-Token": CONFIRM_TOKEN}),
            )
        assert resp.status_code == 201, resp.text
        assert "bar_alignment" not in c.created_run().config["engine_semantics"]
        assert "bar_alignment" not in c.created_run().config


# ---------------------------------------------------------------------------
# P4-06..09  legacy_bar_alignment (SY-23-16; the SY-22-09 pattern)
# ---------------------------------------------------------------------------

_TWO = [_BTC, _ETH]
_OK = {"bar_alignment": "wp23-1"}
_LEGACY_ROWS: list[tuple[str, Any, str, bool]] = [
    ("multi_no_semantics", {"symbols": _TWO}, "backtest", True),
    ("multi_empty_semantics", {"symbols": _TWO, "engine_semantics": {}}, "backtest", True),
    ("multi_known_marker", _cfg(**_OK), "backtest", False),
    ("multi_unknown_marker", _cfg(bar_alignment="wp99-1"), "backtest", True),
    ("multi_empty_string_marker", _cfg(bar_alignment=""), "backtest", True),
    ("multi_none_marker", _cfg(bar_alignment=None), "backtest", True),
    ("multi_int_marker", _cfg(bar_alignment=1), "backtest", True),
    ("multi_list_marker", _cfg(bar_alignment=["wp23-1"]), "backtest", True),
    ("multi_dict_marker", _cfg(bar_alignment={"v": "wp23-1"}), "backtest", True),
    ("multi_pre_wp23_clock_only", _cfg(timestamp_clock="wp22-1"), "backtest", True),
    ("multi_semantics_is_list", {"symbols": _TWO, "engine_semantics": ["x"]}, "backtest", True),
    ("multi_semantics_is_str", {"symbols": _TWO, "engine_semantics": "x"}, "backtest", True),
    ("multi_semantics_is_none", {"symbols": _TWO, "engine_semantics": None}, "backtest", True),
    ("three_known_marker", {**_cfg(**_OK), "symbols": [_BTC, _ETH, "SOL/USD"]}, "backtest", False),
    ("single_no_semantics", {"symbols": [_BTC]}, "backtest", False),
    ("single_unknown_marker", {**_cfg(bar_alignment="x"), "symbols": [_BTC]}, "backtest", False),
    ("single_known_marker", {**_cfg(**_OK), "symbols": [_BTC]}, "backtest", False),
    ("symbols_missing_known_marker", {"engine_semantics": _OK}, "backtest", False),
    ("symbols_missing_no_marker", {"engine_semantics": {}}, "backtest", True),
    ("symbols_none_no_marker", {"symbols": None}, "backtest", True),
    ("symbols_empty_no_marker", {"symbols": []}, "backtest", True),
    ("symbols_is_str_no_marker", {"symbols": _BTC}, "backtest", True),
    ("symbols_is_dict_no_marker", {"symbols": {_BTC: 1}}, "backtest", True),
    ("symbols_non_str_item_no_marker", {"symbols": [1]}, "backtest", True),
    ("symbols_unhashable_item_no_marker", {"symbols": [["a"]]}, "backtest", True),
    ("symbols_none_item_no_marker", {"symbols": [None]}, "backtest", True),
    ("symbols_empty_known_marker", {"symbols": [], "engine_semantics": _OK}, "backtest", False),
    ("symbols_is_str_known_marker", {"symbols": _BTC, "engine_semantics": _OK}, "backtest", False),
    (
        "symbols_non_str_item_known_marker",
        {"symbols": [1], "engine_semantics": _OK},
        "backtest",
        False,
    ),
    ("single_non_str_item_unknown_marker", {"symbols": [1]}, "backtest", True),
    (
        "duplicate_symbols_known",
        {"symbols": [_BTC, _BTC], "engine_semantics": _OK},
        "backtest",
        False,
    ),
    ("config_none", None, "backtest", True),
    ("config_is_int", 5, "backtest", True),
    ("config_is_str", "x", "backtest", True),
    ("config_is_list", [1, 2], "backtest", True),
    ("paper_multi_no_marker", {"symbols": _TWO}, "paper", False),
    ("live_multi_no_marker", {"symbols": _TWO}, "live", False),
    ("paper_config_none", None, "paper", False),
    ("optimize_mode_is_never_flagged", {"symbols": _TWO}, "optimize", False),
    ("unknown_mode_is_never_flagged", {"symbols": _TWO}, "weird", False),
]


class TestLegacyFlag:
    @pytest.mark.parametrize(
        ("config", "mode", "expected"),
        [pytest.param(cfg, mode, exp, id=name) for name, cfg, mode, exp in _LEGACY_ROWS],
    )
    def test_p4_06_truth_table_and_never_raises(
        self, config: Any, mode: str, expected: bool
    ) -> None:
        assert bar_alignment_response_fields(config, mode) == {"legacy_bar_alignment": expected}

    def test_p4_07_the_flag_is_computed_by_a_helper_separate_from_the_timestamp_one(self) -> None:
        # the WP2.2 flag keeps its meaning: a WP2.2 multi-symbol backtest is NOT a legacy
        # timestamp run, but IS a legacy bar-alignment run
        cfg = _cfg(timestamp_clock="wp22-1")
        assert cost_profiles.timestamp_response_fields(cfg, "backtest") == {
            "legacy_timestamps": False
        }
        assert bar_alignment_response_fields(cfg, "backtest") == {"legacy_bar_alignment": True}

    def test_p4_08_both_response_converters_carry_the_flag(self) -> None:
        legacy, fresh = _run_row({"symbols": _TWO}), _run_row(_cfg(**_OK))
        for convert in (run_orm_to_response, run_orm_to_detail_response):
            assert convert(legacy).legacy_bar_alignment is True
            assert convert(fresh).legacy_bar_alignment is False
            assert convert(_run_row({"symbols": _TWO}, mode="paper")).legacy_bar_alignment is False
        assert isinstance(run_orm_to_response(fresh), RunResponse)
        assert isinstance(run_orm_to_detail_response(fresh), RunDetailResponse)
        assert RunResponse.model_fields["legacy_bar_alignment"].default is False

    def test_p4_09_http_create_reports_the_flag_false_for_a_new_multi_symbol_run(
        self, c: _C
    ) -> None:
        bars = {_BTC: _bars(_BTC), _ETH: _bars(_ETH)}
        resp = _post_bt(c, bars, symbols=_TWO)
        assert resp.status_code == 201, resp.text
        assert resp.json()["legacyBarAlignment"] is False
        meta = c.created_run().config["bar_alignment"]
        assert [s["symbol"] for s in meta["symbols"]] == _TWO


# ---------------------------------------------------------------------------
# P4-10..17  BarAlignmentError -> 422 (SY-23-08, CR-05, I-8)
# ---------------------------------------------------------------------------


def _handler_names(node: ast.Try) -> list[str]:
    return [ast.unparse(h.type) if h.type is not None else "<bare>" for h in node.handlers]


def _function(path: str, name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse((_REPO / path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name)


class TestRejectionMapping:
    @pytest.mark.parametrize("code", BAR_ALIGNMENT_CODES)
    def test_p4_10_the_helper_maps_every_closed_code(self, code: str) -> None:
        err = bar_alignment_http_error(BarAlignmentError(code))
        assert err.status_code == 422
        detail: object = err.detail
        assert detail == {"code": code, "error_type": "BarAlignmentError"}

    def test_p4_11_a_foreign_code_degrades_to_a_fixed_one(self) -> None:
        exc = BarAlignmentError(BAR_ALIGNMENT_CODES[0])
        exc.code = "free text: BTC/USD 17"
        detail: object = bar_alignment_http_error(exc).detail
        assert detail == {
            "code": "bar_alignment_failed",
            "error_type": "BarAlignmentError",
        }

    def test_p4_12_the_runs_branch_precedes_except_value_error(self) -> None:
        fn = _function("apps/api/routers/runs.py", "create_run")
        tries = [
            t
            for t in ast.walk(fn)
            if isinstance(t, ast.Try) and "BarAlignmentError" in _handler_names(t)
        ]
        assert len(tries) == 1
        names = _handler_names(tries[0])
        assert "ValueError" in names
        assert names.index("BarAlignmentError") < names.index("ValueError")
        assert names.index("HTTPException") < names.index("BarAlignmentError")

    def test_p4_13_runs_422_run_row_error_in_flight_and_nothing_echoed(self, c: _C) -> None:
        """The row's ``error`` status is in-flight only: the outer rollback discards it."""
        # the window is 120 slots, the exchange returned 100: tail_coverage (the E3 shape)
        resp = _post_bt(c, _single(100))
        assert resp.status_code == 422, resp.text
        assert resp.json() == {
            "detail": {"code": "bar_alignment_tail_coverage", "error_type": "BarAlignmentError"}
        }
        for leaked in ("bar alignment check failed", _BTC, "Insufficient", "data quality"):
            assert leaked not in resp.text
        run = c.created_run()
        assert run.status == "error" and run.stopped_at is not None
        c.db.rollback.assert_awaited()
        c.db.commit.assert_not_awaited()  # nothing durable: the log line is the record

    def test_p4_14_multi_symbol_rejection_uses_the_first_code_in_precedence_order(
        self, c: _C
    ) -> None:
        bars = {_BTC: _bars(_BTC), _ETH: _bars(_ETH, drop=range(50, 60))}
        resp = _post_bt(c, bars, symbols=_TWO)
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["code"] == "bar_alignment_missing_bars"

    def test_p4_15_a_generic_value_error_is_still_the_400_branch(self, c: _C) -> None:
        with patch.object(BacktestRunner, "run", side_effect=ValueError("boom")):
            resp = _post_bt(c, _single())
        assert resp.status_code == 400, resp.text

    def test_p4_16_the_log_has_code_and_type_only_and_no_exception_text(self, c: _C) -> None:
        with capture_logs() as logs:
            resp = _post_bt(c, _single(100))
        assert resp.status_code == 422
        (event,) = [e for e in logs if e["event"] == "runs.backtest_bar_alignment_rejected"]
        assert event["code"] == "bar_alignment_tail_coverage"
        assert event["error_type"] == "BarAlignmentError"
        assert "error" not in event and "exc_info" not in event
        blob = json.dumps(logs, default=str)
        assert "bar alignment check failed" not in blob

    def test_p4_16b_production_chain_prints_no_exception_text(
        self, c: _C, production_logging: ConfigureFn
    ) -> None:
        cap = production_logging(json_output=True)
        resp = _post_bt(c, _single(100))
        assert resp.status_code == 422
        assert cap.lines_with("runs.backtest_bar_alignment_rejected"), "non-vacuous"
        assert "bar alignment check failed" not in cap.text
        assert "Traceback" not in cap.text

    def test_p4_17_optimize_422_before_persistence_and_nothing_echoed(self, c: _C) -> None:
        bars = {_BTC: _bars(_BTC, 300)}
        with capture_logs() as logs:
            resp = _post_opt(c, bars, backtestEnd="2024-01-14T12:00:00Z")
        assert resp.status_code == 422, resp.text
        assert resp.json() == {
            "detail": {"code": "bar_alignment_tail_coverage", "error_type": "BarAlignmentError"}
        }
        assert "bar alignment check failed" not in resp.text and _BTC not in resp.text
        c.db.commit.assert_not_awaited()
        assert not [x for x in c.db.add.call_args_list if isinstance(x.args[0], OptimizationRunORM)]
        (event,) = [e for e in logs if e["event"] == "optimize.bar_alignment_rejected"]
        assert event["code"] == "bar_alignment_tail_coverage"
        assert "error" not in event and "bar alignment check failed" not in json.dumps(
            logs, default=str
        )
        assert not [e for e in logs if e["event"] == "optimizer.combination_failed"]

    def test_p4_17b_the_optimize_call_sits_inside_the_bar_alignment_try(self) -> None:
        fn = _function("apps/api/routers/optimize.py", "run_optimization")
        guarded = [
            t
            for t in ast.walk(fn)
            if isinstance(t, ast.Try)
            and "BarAlignmentError" in _handler_names(t)
            and "optimizer.run" in ast.unparse(ast.Module(body=t.body, type_ignores=[]))
        ]
        assert len(guarded) == 1


# ---------------------------------------------------------------------------
# P4-18..24  requested_window (SY-23-14, CR-05)
# ---------------------------------------------------------------------------

_T = TimeFrame.ONE_HOUR


class TestRequestedWindow:
    def test_p4_18_future_end_is_clamped_to_the_floor_of_now(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2030, 1, 1, tzinfo=UTC)
        now = datetime(2026, 3, 10, 12, 34, 56, tzinfo=UTC)
        assert requested_bar_window(start, end, _T, now) == (
            start,
            datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        )

    def test_p4_19_a_past_end_is_untouched_and_now_on_a_boundary_is_its_own_floor(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 6, 1, tzinfo=UTC)
        assert requested_bar_window(start, end, _T, _NOW) == (start, end)
        on_boundary = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        assert requested_bar_window(start, datetime(2031, 1, 1, tzinfo=UTC), _T, on_boundary) == (
            start,
            on_boundary,
        )

    @pytest.mark.parametrize(
        ("tf", "floor"),
        [
            (TimeFrame.ONE_MINUTE, datetime(2026, 3, 10, 12, 34, tzinfo=UTC)),
            (TimeFrame.FIFTEEN_MINUTES, datetime(2026, 3, 10, 12, 30, tzinfo=UTC)),
            (TimeFrame.FOUR_HOURS, datetime(2026, 3, 10, 12, 0, tzinfo=UTC)),
            (TimeFrame.ONE_DAY, datetime(2026, 3, 10, tzinfo=UTC)),
        ],
    )
    def test_p4_20_floor_is_epoch_anchored_per_timeframe(
        self, tf: TimeFrame, floor: datetime
    ) -> None:
        now = datetime(2026, 3, 10, 12, 34, 56, tzinfo=UTC)
        _, end = requested_bar_window(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2031, 1, 1, tzinfo=UTC), tf, now
        )
        assert end == floor

    def test_p4_21_naive_datetimes_are_read_as_utc_and_the_result_is_aware(self) -> None:
        start, end = requested_bar_window(
            datetime(2024, 1, 1), datetime(2031, 1, 1), _T, datetime(2026, 3, 10, 12, 30)
        )
        assert start == datetime(2024, 1, 1, tzinfo=UTC)
        assert end == datetime(2026, 3, 10, 12, 0, tzinfo=UTC)

    def test_p4_22_the_helper_module_reads_no_clock(self) -> None:
        tree = ast.parse((_REPO / "apps/api/services/cost_profiles.py").read_text())
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not {
            c for c in calls if c.endswith((".now", ".utcnow", "time.time", "time.time_ns"))
        }

    def test_p4_23_the_runner_gets_the_clamped_window_from_the_injected_now(self, c: _C) -> None:
        """The forming bar is trimmed and the tail check passes (WP23-T-21)."""
        seen: dict[str, Any] = {}
        original = BacktestRunner.run

        async def _spy(self: BacktestRunner, *a: Any, **kw: Any) -> Any:
            seen.update(kw)
            return await original(self, *a, **kw)

        now = datetime(2024, 1, 4, 12, 30, tzinfo=UTC)
        with (
            patch("api.routers.runs._window_now", return_value=now),
            patch.object(BacktestRunner, "run", _spy),
        ):
            resp = _post_bt(c, _single())  # 120 bars up to 2024-01-06, request end 2024-01-06
        assert resp.status_code == 201, resp.text
        assert seen["requested_window"] == (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 4, 12, 0, tzinfo=UTC),
        )
        meta = c.created_run().config["bar_alignment"]
        assert meta["window"] == {
            "start": "2024-01-01T00:00:00+00:00",
            "end": "2024-01-04T12:00:00+00:00",
            "requested": True,
            "slots": 84,
        }
        (sym,) = meta["symbols"]
        assert sym["trimmed_outside"] == 36 and sym["tail"] == 0 and sym["bars_out"] == 84

    def test_p4_24_without_a_clamp_a_past_window_is_passed_through(self, c: _C) -> None:
        seen: dict[str, Any] = {}
        original = BacktestRunner.run

        async def _spy(self: BacktestRunner, *a: Any, **kw: Any) -> Any:
            seen.update(kw)
            return await original(self, *a, **kw)

        with patch.object(BacktestRunner, "run", _spy):
            resp = _post_bt(c, _single())
        assert resp.status_code == 201, resp.text
        assert seen["requested_window"] == (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 6, tzinfo=UTC),
        )
        assert c.created_run().config["bar_alignment"]["symbols"][0]["trimmed_outside"] == 0

    def test_p4_24b_the_optimizer_gets_the_same_window_and_now_is_the_runs_seam(
        self, c: _C
    ) -> None:
        seen: dict[str, Any] = {}
        original = ParameterOptimizer.run

        async def _spy(self: ParameterOptimizer, *a: Any, **kw: Any) -> Any:
            seen.update(kw)
            return await original(self, *a, **kw)

        now = datetime(2024, 1, 10, 6, 15, tzinfo=UTC)
        with (
            patch("api.routers.runs._window_now", return_value=now),
            patch.object(ParameterOptimizer, "run", _spy),
        ):
            resp = _post_opt(c, {_BTC: _bars(_BTC, 300)})  # request end 2024-01-13T12:00
        assert resp.status_code == 200, resp.text
        assert seen["requested_window"] == (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 10, 6, 0, tzinfo=UTC),
        )
        assert vars(optimize_module)["_runs_router"] is runs_module


# ---------------------------------------------------------------------------
# P4-25..29  persisted metadata, QA N2, total_bars
# ---------------------------------------------------------------------------


class TestPersistedMetadata:
    def test_p4_25_backtest_config_holds_the_report_with_no_floats(self, c: _C) -> None:
        resp = _post_bt(c, _single(drop=[10]))
        assert resp.status_code == 201, resp.text
        meta = c.created_run().config["bar_alignment"]
        assert meta["version"] == "wp23-1" and meta["policy"] == "union_present_mask"
        assert isinstance(meta["presence_digest"], str) and len(meta["presence_digest"]) == 64
        assert meta["symbols"][0]["missing"] == 1
        assert _floats(meta) == []
        json.dumps(meta)  # JSONB-safe

    def test_p4_26_optimize_run_metadata_holds_the_report_with_no_floats(self, c: _C) -> None:
        resp = _post_opt(c, {_BTC: _bars(_BTC, 300)})
        assert resp.status_code == 200, resp.text
        meta = _opt_run(c).run_metadata["bar_alignment"]
        assert meta["version"] == "wp23-1" and _floats(meta) == []
        assert _opt_run(c).run_metadata["engine_semantics"]["bar_alignment"] == "wp23-1"
        assert _opt_run(c).run_metadata["engine_semantics"]["timestamp_clock"] == "wp22-1"

    def test_p4_27_runs_are_compared_by_presence_digest_not_fingerprint(self, c: _C) -> None:
        """QA N2: the optimizer aligns with warm-up 0, so only ``fingerprint`` moves."""
        bars = {_BTC: _bars(_BTC, 120, drop=[10])}
        assert _post_bt(c, bars).status_code == 201
        runner_meta = c.created_run().config["bar_alignment"]
        resp = _post_opt(c, bars, backtestEnd="2024-01-06T00:00:00Z")
        assert resp.status_code == 200, resp.text
        opt_meta = _opt_run(c).run_metadata["bar_alignment"]
        assert runner_meta["presence_digest"] == opt_meta["presence_digest"]
        assert runner_meta["fingerprint"] != opt_meta["fingerprint"]
        assert runner_meta["symbols"][0]["holes_in_warmup"] == 1
        assert opt_meta["symbols"][0]["holes_in_warmup"] == 0
        assert opt_meta["symbols"][0]["holes_in_trading"] == 1

    def test_p4_28_the_n2_note_is_documented_where_the_report_is_persisted(self) -> None:
        src = (_REPO / "apps/api/services/run_persistence.py").read_text()
        assert "presence_digest" in src and "NOT by" in src

    def test_p4_29_total_bars_means_timeline_steps_minus_warmup_and_is_documented(
        self, c: _C
    ) -> None:
        resp = _post_bt(c, _single(drop=[100]))  # one hole late in the run
        assert resp.status_code == 201, resp.text
        steps = c.created_run().config["bar_alignment"]["steps"]
        assert steps == 119
        desc = BacktestMetricsResponse.model_fields["total_bars"].description or ""
        assert "minus the warm-up" in desc
        total = c.created_run().config["backtest_metrics"]["total_bars"]
        assert total < steps  # steps - W, W > 0


# ---------------------------------------------------------------------------
# P4-30..37  SEC-02 bounds
# ---------------------------------------------------------------------------

_START = datetime(2024, 1, 1, tzinfo=UTC)


def _bounds(*, end: datetime, n: int = 1, tf: TimeFrame = _T, start: datetime = _START) -> None:
    check_backtest_request_bounds(start=start, end=end, timeframe=tf, n_symbols=n)


def _code(excinfo: pytest.ExceptionInfo[HTTPException]) -> Any:
    assert excinfo.value.status_code == 422
    return excinfo.value.detail


class TestBounds:
    def test_p4_30_the_constants(self) -> None:
        assert (MAX_BACKTEST_SYMBOLS, MAX_BACKTEST_BARS) == (20, 500_000)
        assert MIN_BACKTEST_START == datetime(2009, 1, 3, tzinfo=UTC)

    def test_p4_31_symbol_count_boundary(self) -> None:
        _bounds(end=_START + timedelta(hours=10), n=20)
        with pytest.raises(HTTPException) as ei:
            _bounds(end=_START + timedelta(hours=10), n=21)
        assert _code(ei) == {"code": "too_many_symbols"}

    def test_p4_32_start_boundary(self) -> None:
        _bounds(start=MIN_BACKTEST_START, end=MIN_BACKTEST_START + timedelta(hours=10))
        early = MIN_BACKTEST_START - timedelta(seconds=1)
        with pytest.raises(HTTPException) as ei:
            _bounds(start=early, end=early + timedelta(hours=10))
        assert _code(ei) == {"code": "start_too_early"}
        # a naive datetime is read as UTC
        with pytest.raises(HTTPException) as ei:
            _bounds(start=datetime(2008, 12, 31), end=datetime(2009, 1, 1))
        assert _code(ei) == {"code": "start_too_early"}

    def test_p4_33_bar_budget_boundary_one_symbol(self) -> None:
        _bounds(end=_START + timedelta(hours=MAX_BACKTEST_BARS))  # exactly the budget
        with pytest.raises(HTTPException) as ei:
            _bounds(end=_START + timedelta(hours=MAX_BACKTEST_BARS + 1))
        assert _code(ei) == {"code": "range_too_large"}

    def test_p4_34_the_bar_budget_multiplies_by_the_symbol_count_and_rounds_up(self) -> None:
        _bounds(end=_START + timedelta(hours=100_000), n=5)  # 500 000 exactly
        with pytest.raises(HTTPException) as ei:
            _bounds(end=_START + timedelta(hours=100_000, minutes=1), n=5)  # ceil -> 100 001
        assert _code(ei) == {"code": "range_too_large"}

    def test_p4_35_the_budget_is_per_timeframe(self) -> None:
        one_year = _START + timedelta(days=366)
        _bounds(end=one_year, tf=TimeFrame.FIVE_MINUTES)  # 105 408 bars
        with pytest.raises(HTTPException) as ei:
            _bounds(end=one_year, tf=TimeFrame.ONE_MINUTE)  # 527 040 bars
        assert _code(ei) == {"code": "range_too_large"}

    def test_p4_36_http_backtest_rejects_before_any_fetch(self, c: _C) -> None:
        def post(**overrides: Any) -> Any:
            body = {**_BACKTEST, **overrides}
            with patch(
                "api.routers.runs._fetch_bars_for_backtest",
                side_effect=AssertionError("bar fetch must not be reached"),
            ):
                resp = c.client.post("/api/v1/runs", json=body, headers=_hdr())
            # CR-01: the bounds are checked BEFORE ``store.claim`` -- no idempotency row,
            # not even one a failure path would later release.
            assert c.store._rows == {}
            return resp

        many = [f"S{i}/USD" for i in range(21)]
        resp = post(symbols=many)
        assert resp.status_code == 422 and resp.json()["detail"] == {"code": "too_many_symbols"}
        resp = post(backtestStart="2008-12-31T00:00:00Z", backtestEnd="2009-01-02T00:00:00Z")
        assert resp.status_code == 422 and resp.json()["detail"] == {"code": "start_too_early"}
        resp = post(timeframe="1m", backtestEnd="2025-01-01T00:00:00Z")
        assert resp.status_code == 422 and resp.json()["detail"] == {"code": "range_too_large"}
        c.db.add.assert_not_called()  # nothing was created

    def test_p4_37_http_optimize_rejects_before_any_fetch(self, c: _C) -> None:
        def post(**overrides: Any) -> Any:
            with patch(
                "api.routers.optimize._fetch_bars_for_backtest",
                new=AsyncMock(side_effect=AssertionError("bar fetch must not be reached")),
            ):
                return c.client.post("/api/v1/optimize", json={**_OPT_BODY, **overrides})

        many = [f"S{i}/USD" for i in range(21)]
        assert post(symbols=many).json()["detail"] == {"code": "too_many_symbols"}
        assert post(backtestStart="2008-12-31T00:00:00Z").json()["detail"] == {
            "code": "start_too_early"
        }
        resp = post(timeframe="1m", backtestEnd="2025-01-01T00:00:00Z")
        assert resp.status_code == 422 and resp.json()["detail"] == {"code": "range_too_large"}

    def test_p4_37b_paper_and_live_are_not_subject_to_the_history_bounds(self, c: _C) -> None:
        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp = c.client.post(
                "/api/v1/runs",
                json={**_PAPER, "symbols": [f"S{i}/USDT" for i in range(21)]},
                headers=_hdr(),
            )
        assert resp.status_code == 201, resp.text


# ---------------------------------------------------------------------------
# P4-38..46  CF-23-20 duplicate symbols
# ---------------------------------------------------------------------------


class TestDuplicateSymbols:
    def test_p4_38_the_run_request_rejects_a_duplicate_and_keeps_the_order(self) -> None:
        base: dict[str, Any] = {
            "strategy_name": "ma_crossover",
            "timeframe": "1h",
            "mode": "paper",
            "initial_capital": "10000",
        }
        ok = RunCreateRequest(symbols=[_ETH, _BTC, "SOL/USD"], **base)
        assert ok.symbols == [_ETH, _BTC, "SOL/USD"]  # order preserved, nothing deduped
        with pytest.raises(ValidationError) as ei:
            RunCreateRequest(symbols=[_BTC, _ETH, _BTC], **base)
        assert "symbols must be unique" in str(ei.value)
        assert _ETH not in str(ei.value).split("input_value")[0]

    @pytest.mark.parametrize("mode", ["backtest", "paper"])
    def test_p4_39_http_create_duplicate_is_422_in_every_mode(self, c: _C, mode: str) -> None:
        body = {**(_BACKTEST if mode == "backtest" else _PAPER), "symbols": [_BTC, _BTC]}
        with patch(
            "api.routers.runs._fetch_bars_for_backtest",
            side_effect=AssertionError("bar fetch must not be reached"),
        ):
            resp = c.client.post("/api/v1/runs", json=body, headers=_hdr())
        assert resp.status_code == 422, resp.text
        c.db.add.assert_not_called()

    def test_p4_40_http_optimize_duplicate_is_422(self, c: _C) -> None:
        resp = _post_opt(c, {_BTC: _bars(_BTC, 300)}, symbols=[_BTC, _BTC])
        assert resp.status_code == 422, resp.text
        assert "symbols must be unique" in resp.text
        c.db.commit.assert_not_awaited()

    def test_p4_41_the_helper_table(self) -> None:
        assert has_duplicate_symbols([_BTC, _BTC]) is True
        assert has_duplicate_symbols([_BTC, _ETH, _BTC]) is True
        assert has_duplicate_symbols([_BTC, _ETH]) is False
        assert has_duplicate_symbols([_BTC, "btc/usd"]) is False  # exact match, like the engine
        # SEC-08: a non-string element is skipped, never a reason to stop scanning
        assert has_duplicate_symbols([_BTC, 1, _BTC]) is True
        assert has_duplicate_symbols([1, _BTC, None, _BTC]) is True
        assert has_duplicate_symbols([_BTC, 1]) is False
        for malformed in (None, "BTC/USD", [], [1, 1], [["a"], ["a"]], {_BTC: 1}, 5):
            assert has_duplicate_symbols(malformed) is False  # left to the existing shape checks

    def test_p4_42_live_resume_of_a_duplicate_config_fails_closed_with_a_distinct_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        monkeypatch.setattr(runs_module, "_STRATEGY_REGISTRY", {"grid_trading": MagicMock()})
        run_row = resume_mod._make_run_row(status="orphaned")
        run_row.config["symbols"] = [_BTC, _BTC]
        session = resume_mod._DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
        client = resume_mod._client(resume_app, session)
        spy = AsyncMock()
        with patch("api.routers.runs._run_live_engine", spy):
            resp = client.post(
                f"/api/v1/runs/{resume_mod.RUN_ID}/resume",
                headers={
                    "X-Live-Confirm-Token": resume_mod.CONFIRM_TOKEN,
                    "X-Admin-Key": resume_mod.ADMIN_KEY,
                },
            )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": DUPLICATE_SYMBOLS_REASON}
        spy.assert_not_called()
        assert run_row.status == "orphaned"  # untouched: before the CAS
        assert run_row.config["symbols"] == [_BTC, _BTC]  # nothing deduped
        session.commit.assert_not_awaited()

    def test_p4_43_control_a_clean_config_still_resumes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        monkeypatch.setattr(runs_module, "_STRATEGY_REGISTRY", {"grid_trading": MagicMock()})
        run_row = resume_mod._make_run_row(status="orphaned")
        session = resume_mod._DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
        client = resume_mod._client(resume_app, session)

        async def _scan(db: Any, run: Any, exchange: Any, *, fence: Any = None) -> Any:
            from api.services.run_recovery import ImportReport

            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        monkeypatch.setattr(run_recovery_module, "scan_and_import", _scan)
        with patch("api.routers.runs._run_live_engine", AsyncMock()):
            resp = client.post(
                f"/api/v1/runs/{resume_mod.RUN_ID}/resume",
                headers={
                    "X-Live-Confirm-Token": resume_mod.CONFIRM_TOKEN,
                    "X-Admin-Key": resume_mod.ADMIN_KEY,
                },
            )
        _RUN_TASKS.clear()
        assert resp.status_code == 200, resp.text

    async def test_p4_44_paper_boot_recovery_marks_a_duplicate_config_errored(self) -> None:
        from tests.unit import test_sprint24_run_recovery as rec

        config = {
            "strategy_name": "grid_trading",
            "symbols": [_BTC, _BTC],
            "timeframe": "1h",
            "initial_capital": "10000",
            "strategy_params": {},
            "mode": "paper",
        }
        orphan = rec._make_orphan(run_mode="paper", status="running", config=config)
        select_session = rec._make_session_for_select([orphan])
        write_session = rec._make_session_for_write(orphan)
        factory = rec._build_factory(select_session, write_session, write_session)
        spy, mark = AsyncMock(), AsyncMock()
        with (
            patch("api.db.session.get_session_factory", return_value=factory),
            rec._patched_registry("grid_trading"),
            patch(
                "api.routers.runs._load_resume_snapshot",
                AsyncMock(return_value=rec._EMPTY_SNAPSHOT),
            ),
            patch("api.routers.runs._run_paper_engine", spy),
            patch("api.routers.runs._mark_run_error", mark),
        ):
            recovered = await runs_module.recover_orphaned_runs()
        _RUN_TASKS.clear()
        assert recovered == 0
        spy.assert_not_called()
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.kwargs["reason"] == DUPLICATE_SYMBOLS_REASON
        assert orphan.config["symbols"] == [_BTC, _BTC]  # nothing deduped

    def test_p4_45_promote_of_a_duplicate_paper_config_is_a_422_before_the_claim(
        self, c: _C
    ) -> None:
        source = _paper_source(uuid.uuid4(), symbols=[_BTC, _BTC])
        resp, live_spy = _promote(c, source)
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": DUPLICATE_SYMBOLS_REASON}
        live_spy.assert_not_called()
        c.db.add.assert_not_called()

    def test_p4_46_the_reason_is_distinct_from_every_existing_resume_reason(self) -> None:
        assert DUPLICATE_SYMBOLS_REASON == "duplicate_symbols_in_config"
        src = (_REPO / "apps/api/routers/runs.py").read_text()
        assert src.count('reason="empty_symbols"') == 1
        assert "reason=DUPLICATE_SYMBOLS_REASON" in src


# ---------------------------------------------------------------------------
# P4-47..48  the ML gate stays windowless
# ---------------------------------------------------------------------------


class TestMlGateUntouched:
    def test_p4_47_ml_router_never_passes_a_requested_window(self) -> None:
        src = (_REPO / "apps/api/routers/ml.py").read_text()
        assert "requested_window" not in src
        assert "bar_alignment" not in src
        assert "runner.run({symbol: bt_series}, oos_start_index=oos_start_index)" in " ".join(
            src.split()
        )


# ---------------------------------------------------------------------------
# P4-49..52  v2 folds
# ---------------------------------------------------------------------------


class TestV2Folds:
    def test_p4_49_runs_reads_now_once_and_before_the_fetch(self, c: _C) -> None:
        """SEC-01: one shared call-order log; ``now`` after the fetch admits the forming bar."""
        order: list[str] = []
        now = datetime(2024, 1, 4, 12, 30, tzinfo=UTC)

        def _now() -> datetime:
            order.append("now")
            return now

        async def _fetch(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
            order.append("fetch")
            return _single()

        with (
            patch("api.routers.runs._window_now", side_effect=_now) as now_mock,
            patch("api.routers.runs._fetch_bars_for_backtest", side_effect=_fetch),
        ):
            resp = c.client.post("/api/v1/runs", json=_BACKTEST, headers=_hdr())
        assert resp.status_code == 201, resp.text
        assert order == ["now", "fetch"]
        assert now_mock.call_count == 1
        # the value read before the fetch is the one that clamped the window
        assert c.created_run().config["bar_alignment"]["window"]["end"] == (
            "2024-01-04T12:00:00+00:00"
        )

    def test_p4_50_optimize_reads_now_once_and_before_the_fetch(self, c: _C) -> None:
        order: list[str] = []
        now = datetime(2024, 1, 10, 6, 15, tzinfo=UTC)

        def _now() -> datetime:
            order.append("now")
            return now

        async def _fetch(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
            order.append("fetch")
            return {_BTC: _bars(_BTC, 300)}

        with (
            patch("api.routers.runs._window_now", side_effect=_now) as now_mock,
            patch("api.routers.optimize._fetch_bars_for_backtest", new=_fetch),
        ):
            resp = c.client.post("/api/v1/optimize", json=_OPT_BODY)
        assert resp.status_code == 200, resp.text
        assert order == ["now", "fetch"]
        assert now_mock.call_count == 1
        window = _opt_run(c).run_metadata["bar_alignment"]["window"]
        assert window["end"] == "2024-01-10T06:00:00+00:00"

    @pytest.mark.parametrize(
        ("start", "end", "code"),
        [
            ("0001-01-01T00:00:00+14:00", "2024-01-06T00:00:00Z", "start_too_early"),
            ("2024-01-01T00:00:00Z", "9999-12-31T23:00:00-12:00", "range_too_large"),
        ],
    )
    def test_p4_51_datetime_overflow_is_a_fixed_422_not_a_500(
        self, c: _C, start: str, end: str, code: str
    ) -> None:
        """SEC-05: ``astimezone(UTC)`` overflows at both ends of the calendar."""
        with pytest.raises(OverflowError):
            datetime.fromisoformat(start if code == "start_too_early" else end).astimezone(UTC)
        with pytest.raises(HTTPException) as ei:
            _bounds(start=datetime.fromisoformat(start), end=datetime.fromisoformat(end))
        assert _code(ei) == {"code": code}
        with patch(
            "api.routers.runs._fetch_bars_for_backtest",
            side_effect=AssertionError("bar fetch must not be reached"),
        ):
            resp = c.client.post(
                "/api/v1/runs",
                json={**_BACKTEST, "backtestStart": start, "backtestEnd": end},
                headers=_hdr(),
            )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": code}
        assert c.store._rows == {}
        with patch(
            "api.routers.optimize._fetch_bars_for_backtest",
            new=AsyncMock(side_effect=AssertionError("bar fetch must not be reached")),
        ):
            resp = c.client.post(
                "/api/v1/optimize",
                json={**_OPT_BODY, "backtestStart": start, "backtestEnd": end},
            )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == {"code": code}
