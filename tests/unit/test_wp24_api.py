"""WP2.4 / P3: API marker, legacy flag and leaderboard half (hermetic: no Postgres, no network).

Binding spec: ``reports/vp2-wp2.4/synthesis-spec.md`` (SY-24-07, SY-24-08, SY-24-11, WP24-T-27 ..
T-32), UD-3 default (a), and the P2 carry-forwards CF-P2-01 (``reason`` is never an eligibility
input), CF-P2-02 (flags and the leaderboard key off the marker and ``status`` only) and CF-P2-03
(``ingest_cutoff`` is always null in Phase A).  Test ids are ``WP24-P3-NN``.

    T-27  marker ``engine_semantics.mtf_context`` on backtest + optimize only; the pinned dicts
          differ by exactly that key; promote never carries it; no client input reaches it
    T-28  ``config["mtf_context"]`` persisted on the backtest config and optimize ``run_metadata``
    T-29  ``legacy_mtf_context`` truth table, wire names, never raises, frozen consumer set
    T-30  leaderboard half + ``legacy_mtf_context_runs_excluded`` + ``mtfContextStatus`` round trip
    T-31  API path == direct ``BacktestRunner`` for rsi_mean_reversion and dca_rsi_hybrid
    T-32  log hygiene of the new flag / create / optimize paths
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import math
import random
import uuid
from collections.abc import Generator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

import api.routers.optimize as optimize_module
import api.routers.portfolio as portfolio_module
import api.routers.runs as runs_module
from api import schemas as schemas_module
from api.db.models import OptimizationRunORM
from api.schemas import (
    AggregatePortfolioResponse,
    RunCreateRequest,
    RunDetailResponse,
    RunResponse,
    is_leaderboard_mtf_eligible,
    leaderboard_eligible,
)
from api.services import cost_profiles
from api.services.cost_profiles import (
    KNOWN_MTF_CONTEXT_STATUSES,
    KNOWN_MTF_CONTEXT_VERSIONS,
    LEGACY_MTF_CONSUMER_STRATEGIES,
    engine_semantics,
    mtf_context_response_fields,
    stamp_cost_config,
)
from api.services.run_persistence import run_orm_to_detail_response, run_orm_to_response
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.bar_paths import scripted_path
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _BACKTEST,
    _C,
    _OPT_BODY,
    _PAPER,
    _clean_run_tasks,
    _hdr,
    _paper_source,
    _promote,
    app,  # fixture: requested by name in ``c`` below
)
from tests.integration.test_wp24_c10 import _install_raising_clients, _uninstall_clients
from trading import mtf_signals
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile
from trading.mtf_signals import MTF_CONTEXT_VERSION
from trading.safety import BACKTEST_CIRCUIT_BREAKER_DEFAULT

_REPO = Path(__file__).resolve().parents[2]
_NOW = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
_MARKER = "wp24-1"
_SENTINEL = "WP24SENTINEL-TAMPERED-VALUE"
_SYM = "BTC/USD"

#: ``engine_semantics("backtest")`` at WP2.5 (before this commit): the dict WITHOUT the new key.
_PRE_P3_BACKTEST: dict[str, Any] = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
    "backtest_circuit_breaker": True,
    "timestamp_clock": "wp22-1",
    "bar_alignment": "wp23-1",
    "metrics": "wp25-1",
}
_PAPER_LIVE_DICT: dict[str, Any] = {
    "schema": 1,
    "cost_model": "wp21-1",
    "exit_semantics": "wp13b-1",
    "daily_risk": "wp15-1",
}


@pytest.fixture()
def c(app: Any) -> Generator[_C, None, None]:  # noqa: F811
    client = _C(app)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_row(config: Any, *, mode: str = "backtest", n_trades: int | None = None) -> Any:
    return SimpleNamespace(
        id=uuid.UUID("aaaaaaaa-0000-0000-0000-000000000024"),
        run_mode=mode,
        status="stopped",
        config=config,
        started_at=_NOW,
        stopped_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
        n_closed_trades=n_trades,
    )


def _cfg(
    *,
    strategy: object = "rsi_mean_reversion",
    marker: object = _MARKER,
    status: object = "unavailable",
    with_marker: bool = True,
    with_strategy: bool = True,
    profile: Any = DEFAULT_COST_PROFILE,
) -> dict[str, Any]:
    """A fully leaderboard-eligible backtest config unless a knob says otherwise."""
    sem: dict[str, Any] = {"schema": 1, "metrics": "wp25-1"}
    if with_marker:
        sem["mtf_context"] = marker
    cfg: dict[str, Any] = {
        "cost_profile": profile.to_metadata(),
        "engine_semantics": sem,
        "mtf_context": {"version": _MARKER, "source": "unavailable", "status": status},
    }
    if with_strategy:
        cfg["strategy_name"] = strategy
    return cfg


def _flags(config: Any, mode: str = "backtest", name: Any = None) -> dict[str, Any]:
    return mtf_context_response_fields(config, mode, name)


def _bars(n: int = 300, seed: int = 5) -> dict[str, list[Any]]:
    rnd = random.Random(seed)  # noqa: S311 -- seeded price path, not security
    closes = [round(100 + 12 * math.sin(i / 5) + rnd.uniform(-1, 1), 2) for i in range(n)]
    first_open = datetime(2024, 1, 3, 2, 0, tzinfo=UTC)
    return {_SYM: scripted_path(closes, first_open=first_open, symbol=_SYM, warmup=50)}


def _window(bars: Mapping[str, list[Any]]) -> dict[str, str]:
    """The request window that exactly covers ``bars`` (first open .. last open + 1h)."""
    series = bars[_SYM]
    end = series[-1].timestamp + timedelta(hours=1)
    return {
        "backtestStart": series[0].timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backtestEnd": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _post_bt(c: _C, bars: Mapping[str, list[Any]], **overrides: Any) -> Any:
    body = {**_BACKTEST, **_window(bars), **overrides}
    with patch("api.routers.runs._fetch_bars_for_backtest", return_value=dict(bars)):
        return c.client.post("/api/v1/runs", json=body, headers=_hdr())


def _post_opt(c: _C, bars: Mapping[str, list[Any]], **overrides: Any) -> Any:
    with patch(
        "api.routers.optimize._fetch_bars_for_backtest", new=AsyncMock(return_value=dict(bars))
    ):
        body = {**_OPT_BODY, **_window(bars), **overrides}
        return c.client.post("/api/v1/optimize", json=body)


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


# ===========================================================================
# T-27  marker
# ===========================================================================


class TestT27Marker:
    def test_p3_01_the_marker_is_the_engine_version(self) -> None:
        assert MTF_CONTEXT_VERSION == _MARKER
        assert KNOWN_MTF_CONTEXT_VERSIONS == frozenset({_MARKER})
        assert cost_profiles.KNOWN_MTF_CONTEXT_VERSIONS is mtf_signals.KNOWN_MTF_CONTEXT_VERSIONS

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    def test_p3_02_the_pinned_dicts_differ_by_exactly_the_one_key(self, mode: str) -> None:
        sem = engine_semantics(mode)
        assert sem == {**_PRE_P3_BACKTEST, "mtf_context": _MARKER}
        assert set(sem) - set(_PRE_P3_BACKTEST) == {"mtf_context"}
        # every other marker is unchanged (nothing is bumped)
        for key, value in _PRE_P3_BACKTEST.items():
            assert sem[key] == value

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_p3_03_absent_on_paper_and_live_and_their_pin_is_unchanged(self, mode: str) -> None:
        assert engine_semantics(mode) == _PAPER_LIVE_DICT
        assert "mtf_context" not in engine_semantics(mode)

    def test_p3_04_stamp_cost_config_writes_the_marker_only_in_the_two_modes(self) -> None:
        for mode, present in (("backtest", True), ("optimize", True), ("paper", False)):
            cfg: dict[str, Any] = {}
            stamp_cost_config(cfg, DEFAULT_COST_PROFILE, mode=mode)
            assert ("mtf_context" in cfg["engine_semantics"]) is present
            assert "mtf_context" not in cfg  # the block is the router's, never the stamp's

    def test_p3_05_http_backtest_is_marked_and_paper_is_not(self, c: _C) -> None:
        resp = _post_bt(c, _bars())
        assert resp.status_code == 201, resp.text
        assert c.created_run().config["engine_semantics"]["mtf_context"] == _MARKER
        assert resp.json()["legacyMtfContext"] is False

    def test_p3_06_http_paper_create_carries_neither_marker_nor_block(self, c: _C) -> None:
        with patch("api.routers.runs._run_paper_engine", new=AsyncMock()):
            resp = c.client.post("/api/v1/runs", json=_PAPER, headers=_hdr())
        assert resp.status_code in (200, 201), resp.text
        cfg = c.created_run().config
        assert "mtf_context" not in cfg["engine_semantics"]
        assert "mtf_context" not in cfg
        assert resp.json()["legacyMtfContext"] is False
        assert resp.json()["mtfContextStatus"] is None

    def test_p3_07_promote_never_carries_the_marker_or_the_block(self, c: _C) -> None:
        # A (hand-tampered) paper source that DOES carry both: the live re-stamp and the pop
        # remove them.
        source = _paper_source(
            uuid.uuid4(),
            engine_semantics={**engine_semantics("backtest")},
            mtf_context={"version": _MARKER, "source": "pit", "status": "pit"},
        )
        resp, _spy = _promote(c, source)
        assert resp.status_code == 201, resp.text
        cfg = c.created_run().config
        assert "mtf_context" not in cfg["engine_semantics"]
        assert "mtf_context" not in cfg
        assert "backtest_circuit_breaker" not in cfg["engine_semantics"]
        assert source.config["mtf_context"]["status"] == "pit"  # the SOURCE row is untouched

    def test_p3_08_no_request_model_has_an_mtf_input(self) -> None:
        for model in (RunCreateRequest, optimize_module.OptimizeRequest):
            names = set(model.model_fields) | {
                f.alias for f in model.model_fields.values() if f.alias
            }
            assert not {n for n in names if "mtf" in n.lower() or "semantics" in n.lower()}

    @pytest.mark.parametrize(
        "extra",
        [
            {"mtfContext": {"status": "pit", "source": "pit"}},
            {"mtf_context": {"status": "pit"}},
            {"engineSemantics": {"mtf_context": "wp24-1"}},
            {"strategyParams": {"mtf_context": {"status": "pit"}}},
        ],
        ids=["camel-block", "snake-block", "semantics", "in-strategy-params"],
    )
    def test_p3_09_a_client_cannot_spoof_the_block_or_the_marker_on_create(
        self, c: _C, extra: dict[str, Any]
    ) -> None:
        bars = _bars()
        body = {**_BACKTEST, **_window(bars)}
        body["strategyParams"] = {**_BACKTEST["strategyParams"], **extra.pop("strategyParams", {})}
        body.update(extra)
        with patch("api.routers.runs._fetch_bars_for_backtest", return_value=bars):
            resp = c.client.post("/api/v1/runs", json=body, headers=_hdr())
        assert resp.status_code in (201, 422), resp.text
        if resp.status_code == 201:
            cfg = c.created_run().config
            # server-built, whatever the client sent
            assert cfg["mtf_context"]["status"] == "not_used"
            assert cfg["mtf_context"]["source"] == "unavailable"
            assert cfg["engine_semantics"]["mtf_context"] == _MARKER
            # a spoof inside strategy_params is inert: it stays nested there, never top-level
            assert cfg["strategy_params"].get("mtf_context") in (None, {"status": "pit"})
        else:
            assert not any(type(x.args[0]).__name__ == "RunORM" for x in c.db.add.call_args_list)

    def test_p3_10_a_client_cannot_spoof_the_block_on_optimize(self, c: _C) -> None:
        resp = _post_opt(
            c,
            _bars(500),
            mtfContext={"status": "pit", "source": "pit"},
            engineSemantics={"mtf_context": "wp24-1"},
        )
        assert resp.status_code in (200, 422), resp.text
        if resp.status_code == 200:
            meta = _opt_run(c).run_metadata
            assert meta["mtf_context"]["status"] == "not_used"
            assert meta["mtf_context"]["source"] == "unavailable"

    def test_p3_11_only_the_engine_report_is_written_and_resume_never_writes_it(self) -> None:
        runs_src = (_REPO / "apps/api/routers/runs.py").read_text()
        opt_src = (_REPO / "apps/api/routers/optimize.py").read_text()
        assert runs_src.count('config_snapshot["mtf_context"] =') == 1
        assert 'config_snapshot["mtf_context"] = runner.last_mtf_context.to_metadata()' in runs_src
        assert opt_src.count('run_metadata["mtf_context"] =') == 1
        assert 'run_metadata["mtf_context"] = optimizer.last_mtf_context.to_metadata()' in opt_src
        # the write is in the backtest branch of create_run: not in resume, recovery or promote
        tree = ast.parse(runs_src)
        writers = [
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef))
            and 'config_snapshot["mtf_context"] =' in (ast.get_source_segment(runs_src, fn) or "")
            and not any(
                isinstance(inner, (ast.AsyncFunctionDef, ast.FunctionDef)) and inner is not fn
                for inner in ast.walk(fn)
                if 'config_snapshot["mtf_context"] ='
                in (ast.get_source_segment(runs_src, inner) or "")
            )
        ]
        assert writers == ["create_run"]


# ===========================================================================
# T-28  persisted block
# ===========================================================================

_RSI_BODY: dict[str, Any] = {"strategyName": "rsi_mean_reversion", "strategyParams": {}}


class TestT28PersistedBlock:
    def test_p3_12_non_consumer_backtest_persists_not_used_unavailable(self, c: _C) -> None:
        resp = _post_bt(c, _bars())
        assert resp.status_code == 201, resp.text
        block = c.created_run().config["mtf_context"]
        assert block["version"] == _MARKER
        assert block["source"] == "unavailable"
        assert block["status"] == "not_used"
        assert block["series"] == {}
        assert block["series_digest"] is None
        assert block["ingest_cutoff"] is None  # CF-P2-03: always null in Phase A
        assert _floats(block) == []
        json.dumps(block)  # JSONB-safe
        body = resp.json()
        assert body["legacyMtfContext"] is False and body["mtfContextStatus"] == "not_used"

    def test_p3_13_consumer_backtest_persists_unavailable_with_clients_installed(
        self, c: _C, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        touched = _install_raising_clients(monkeypatch)  # the API's boot-warmed clients
        resp = _post_bt(c, _bars(), **_RSI_BODY)
        assert resp.status_code == 201, resp.text
        block = c.created_run().config["mtf_context"]
        assert (block["source"], block["status"]) == ("unavailable", "unavailable")
        assert set(block["fields"].values()) <= {"no_source", "not_served"}
        assert touched == []
        body = resp.json()
        assert body["legacyMtfContext"] is False and body["mtfContextStatus"] == "unavailable"

    def test_p3_14_optimize_run_metadata_holds_the_block(self, c: _C) -> None:
        resp = _post_opt(c, _bars(500))
        assert resp.status_code == 200, resp.text
        meta = _opt_run(c).run_metadata
        assert meta["engine_semantics"]["mtf_context"] == _MARKER
        block = meta["mtf_context"]
        assert (block["version"], block["source"], block["status"]) == (
            _MARKER,
            "unavailable",
            "not_used",
        )
        assert block["ingest_cutoff"] is None and _floats(block) == []
        json.dumps(block)
        body = resp.json()
        assert body["legacyMtfContext"] is False and body["mtfContextStatus"] == "not_used"

    def test_p3_15_optimize_consumer_reads_unavailable(self, c: _C) -> None:
        resp = _post_opt(
            c,
            _bars(500),
            strategyName="rsi_mean_reversion",
            paramGrid={"rsi_period": [10, 14]},
        )
        assert resp.status_code == 200, resp.text
        assert _opt_run(c).run_metadata["mtf_context"]["status"] == "unavailable"
        assert resp.json()["mtfContextStatus"] == "unavailable"

    def test_p3_16_one_report_per_run_not_per_combination(self, c: _C) -> None:
        resp = _post_opt(c, _bars(500))
        assert resp.status_code == 200, resp.text
        assert list(_opt_run(c).run_metadata).count("mtf_context") == 1


# ===========================================================================
# T-29  legacy flag truth table
# ===========================================================================

#: (case id, config, mode, strategy-name argument, expected legacy flag)
_CONSUMER = "rsi_mean_reversion"
_TABLE: list[tuple[str, Any, str, Any, bool]] = [
    ("known-consumer", _cfg(), "backtest", None, False),
    ("known-non-consumer", _cfg(strategy="ma_crossover"), "backtest", None, False),
    ("missing-consumer", _cfg(with_marker=False), "backtest", None, True),
    (
        "missing-non-consumer",
        _cfg(strategy="ma_crossover", with_marker=False),
        "backtest",
        None,
        False,
    ),
    ("none-consumer", _cfg(marker=None), "backtest", None, True),
    ("none-non-consumer", _cfg(strategy="breakout", marker=None), "backtest", None, False),
    ("unknown-consumer", _cfg(marker="wp24-2"), "backtest", None, True),
    ("unknown-non-consumer", _cfg(strategy="breakout", marker="wp99"), "backtest", None, False),
    ("non-str-int-consumer", _cfg(marker=1), "backtest", None, True),
    ("non-str-list-consumer", _cfg(marker=["wp24-1"]), "backtest", None, True),
    ("non-str-dict-consumer", _cfg(marker={"wp24-1": 1}), "backtest", None, True),
    ("bool-consumer", _cfg(marker=True), "backtest", None, True),
    ("sem-str", {"engine_semantics": "junk", "strategy_name": _CONSUMER}, "backtest", None, True),
    ("sem-list", {"engine_semantics": [], "strategy_name": _CONSUMER}, "backtest", None, True),
    ("sem-none", {"engine_semantics": None, "strategy_name": _CONSUMER}, "backtest", None, True),
    ("sem-absent", {"strategy_name": _CONSUMER}, "backtest", None, True),
    (
        "sem-non-consumer",
        {"engine_semantics": "junk", "strategy_name": "grid_trading"},
        "backtest",
        None,
        False,
    ),
    ("config-none", None, "backtest", None, True),  # no name at all: fail-closed
    ("config-none-name-arg-consumer", None, "backtest", "dca_rsi_hybrid", True),
    ("config-none-name-arg-non-consumer", None, "backtest", "ma_crossover", False),
    ("config-list", [], "backtest", None, True),
    ("config-empty", {}, "backtest", None, True),
    ("name-missing", _cfg(with_marker=False, with_strategy=False), "backtest", None, True),
    ("name-none", _cfg(with_marker=False, strategy=None), "backtest", None, True),
    ("name-int", _cfg(with_marker=False, strategy=7), "backtest", None, True),
    ("name-list", _cfg(with_marker=False, strategy=["breakout"]), "backtest", None, True),
    ("name-dict", _cfg(with_marker=False, strategy={"a": 1}), "backtest", None, True),
    ("name-arg-non-str", _cfg(with_marker=False, strategy="breakout"), "backtest", 5, True),
    (
        "name-arg-wins",
        _cfg(with_marker=False, strategy="breakout"),
        "backtest",
        "model_strategy",
        True,
    ),
    ("name-arg-wins-2", _cfg(with_marker=False, strategy=_CONSUMER), "backtest", "breakout", False),
    ("ensemble", _cfg(with_marker=False, strategy="ensemble"), "backtest", None, True),
    ("model_strategy", _cfg(with_marker=False, strategy="model_strategy"), "backtest", None, True),
    ("dca", _cfg(with_marker=False, strategy="dca_rsi_hybrid"), "backtest", None, True),
    (
        "class-name-rsi",
        _cfg(with_marker=False, strategy="RSIMeanReversionStrategy"),
        "optimize",
        None,
        True,
    ),
    (
        "class-name-ma",
        _cfg(with_marker=False, strategy="MACrossoverStrategy"),
        "optimize",
        None,
        False,
    ),
    (
        "unnormalised-key",
        _cfg(with_marker=False, strategy="RSI-Mean-Reversion"),
        "backtest",
        None,
        True,
    ),
    ("optimize-unmarked-consumer", _cfg(with_marker=False), "optimize", None, True),
    ("optimize-marked-consumer", _cfg(), "optimize", None, False),
    (
        "optimize-unmarked-non-consumer",
        _cfg(strategy="breakout", with_marker=False),
        "optimize",
        None,
        False,
    ),
    ("paper-unmarked-consumer", _cfg(with_marker=False), "paper", None, False),
    ("live-unmarked-consumer", _cfg(with_marker=False), "live", None, False),
    ("paper-no-name", {}, "paper", None, False),
    ("live-config-none", None, "live", None, False),
    ("unknown-mode", _cfg(with_marker=False), "weird", None, False),
    ("empty-mode", _cfg(with_marker=False), "", None, False),
    # E3 / SY-24-08: an empty or whitespace-only name is missing (fail-closed); never normalised
    ("empty-name-backtest", _cfg(with_marker=False, strategy=""), "backtest", None, True),
    ("blank-name-optimize", _cfg(with_marker=False, strategy=" "), "optimize", None, True),
    (
        "blank-name-arg-wins",
        _cfg(with_marker=False, strategy="ma_crossover"),
        "optimize",
        "\t\n",
        True,
    ),
    ("paper-empty-name", _cfg(with_marker=False, strategy=""), "paper", None, False),
]


class TestT29LegacyFlag:
    @pytest.mark.parametrize(
        ("config", "mode", "name", "expected"),
        [pytest.param(*row[1:], id=row[0]) for row in _TABLE],
    )
    def test_p3_17_truth_table(self, config: Any, mode: str, name: Any, expected: bool) -> None:
        fields = _flags(config, mode, name)
        assert fields["legacy_mtf_context"] is expected
        assert set(fields) == {"legacy_mtf_context", "mtf_context_status"}

    def test_p3_18_the_table_covers_both_outcomes_per_mode(self) -> None:
        seen = {(row[2], row[4]) for row in _TABLE}
        assert {
            ("backtest", True),
            ("backtest", False),
            ("optimize", True),
            ("optimize", False),
        } <= seen
        assert {("paper", False), ("live", False)} <= seen

    def test_p3_19_a_marked_run_is_never_flagged_whatever_the_name(self) -> None:
        for name in (None, 5, ["x"], "ensemble", "model_strategy", "", "anything"):
            cfg = _cfg(strategy=name)
            assert _flags(cfg)["legacy_mtf_context"] is False
            assert _flags(cfg, "optimize")["legacy_mtf_context"] is False

    def test_p3_20_an_unknown_non_empty_name_is_not_flagged(self) -> None:
        assert (
            _flags(_cfg(with_marker=False, strategy="some_new_strategy"))["legacy_mtf_context"]
            is False
        )

    def test_p3_21_hostile_objects_never_raise_and_never_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _Boom:
            def __eq__(self, other: object) -> bool:
                raise AssertionError("compared")

            __hash__ = None  # type: ignore[assignment]

            def __str__(self) -> str:
                raise AssertionError("stringified")

            __repr__ = __str__

        hostile: list[Any] = [
            None,
            0,
            "x",
            [],
            [[]],
            {"engine_semantics": _Boom(), "strategy_name": _Boom(), "mtf_context": _Boom()},
            {"engine_semantics": {"mtf_context": _Boom()}, "strategy_name": _Boom()},
            {"engine_semantics": {"mtf_context": {}}, "strategy_name": {}, "mtf_context": []},
            {"engine_semantics": {"mtf_context": []}, "mtf_context": {"status": []}},
            {"mtf_context": {"status": {}}},
            {"mtf_context": {"status": _Boom()}},
            {"mtf_context": "pit"},
            {"mtf_context": {"status": ["pit"]}},
        ]
        caplog.set_level(logging.DEBUG)
        with capture_logs() as events:
            for cfg in hostile:
                for mode in ("backtest", "optimize", "paper", "live", "", "x"):
                    out = _flags(cfg, mode)
                    assert isinstance(out["legacy_mtf_context"], bool)
                    assert out["mtf_context_status"] is None
                assert is_leaderboard_mtf_eligible(cfg) in (True, False)
            assert _flags(cfg, "backtest", _Boom())["legacy_mtf_context"] is True  # type: ignore[arg-type]
        assert events == []
        assert caplog.records == []

    def test_p3_22_the_frozen_consumer_set_is_the_registry_derivation_plus_ensemble(self) -> None:
        registry = runs_module._get_strategy_registry()
        derived = {key for key, cls in registry.items() if cls.consumes_mtf_signals}
        assert LEGACY_MTF_CONSUMER_STRATEGIES == frozenset(derived | {"ensemble"})
        assert LEGACY_MTF_CONSUMER_STRATEGIES == frozenset(
            {"rsi_mean_reversion", "dca_rsi_hybrid", "model_strategy", "ensemble"}
        )
        assert isinstance(LEGACY_MTF_CONSUMER_STRATEGIES, frozenset)
        from trading.strategies.ensemble import EnsembleStrategy

        class_names = {registry[key].__name__ for key in derived} | {EnsembleStrategy.__name__}
        assert schemas_module.LEGACY_MTF_CONSUMER_CLASS_NAMES == frozenset(class_names)
        assert "ensemble" not in registry  # why it is named explicitly

    def test_p3_23_the_status_set_equals_the_engines_constants(self) -> None:
        engine = {
            getattr(mtf_signals, name)
            for name in (
                "_STATUS_NOT_USED",
                "_STATUS_UNAVAILABLE",
                "_STATUS_PARTIAL",
                "_STATUS_PIT",
            )
        }
        assert KNOWN_MTF_CONTEXT_STATUSES == frozenset(engine)
        assert cost_profiles.KNOWN_MTF_CONTEXT_STATUSES is schemas_module.KNOWN_MTF_CONTEXT_STATUSES

    def test_p3_24_run_list_and_detail_carry_both_fields_and_the_wire_names(self) -> None:
        legacy = _cfg(with_marker=False, status="partial")
        for build in (run_orm_to_response, run_orm_to_detail_response):
            resp = build(_run_row(legacy))
            assert resp.legacy_mtf_context is True and resp.mtf_context_status == "partial"
            wire = resp.model_dump(by_alias=True, mode="json")
            assert wire["legacyMtfContext"] is True and wire["mtfContextStatus"] == "partial"
            ok = build(_run_row(_cfg(status="pit")))
            assert (ok.legacy_mtf_context, ok.mtf_context_status) == (False, "pit")
        assert issubclass(RunDetailResponse, RunResponse)

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_p3_25_paper_and_live_rows_are_never_flagged_and_carry_no_status(
        self, mode: str
    ) -> None:
        cfg = _cfg(with_marker=False, status="pit")
        for build in (run_orm_to_response, run_orm_to_detail_response):
            resp = build(_run_row(cfg, mode=mode))
            assert (resp.legacy_mtf_context, resp.mtf_context_status) == (False, None)

    @staticmethod
    def _opt_row(run_metadata: Any, name: str = "RSIMeanReversionStrategy") -> Any:
        return SimpleNamespace(
            id=uuid.uuid4(),
            strategy_name=name,
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
        ("run_metadata", "name", "legacy", "status"),
        [
            (
                {"engine_semantics": {"mtf_context": _MARKER}, "mtf_context": {"status": "pit"}},
                "RSIMeanReversionStrategy",
                False,
                "pit",
            ),
            ({"engine_semantics": {}}, "RSIMeanReversionStrategy", True, None),
            ({"engine_semantics": {}}, "rsi_mean_reversion", True, None),
            ({"engine_semantics": {}}, "MACrossoverStrategy", False, None),
            (None, "DCARSIHybridStrategy", True, None),
            ({"engine_semantics": "junk"}, "ModelStrategy", True, None),
            ({"engine_semantics": {"mtf_context": _SENTINEL}}, "EnsembleStrategy", True, None),
            ({"mtf_context": {"status": _SENTINEL}}, "BreakoutStrategy", False, None),
        ],
        ids=[
            "marked",
            "unmarked-rsi",
            "unmarked-key",
            "non-consumer",
            "null",
            "junk",
            "unknown",
            "tamper",
        ],
    )
    async def test_p3_26_optimize_get_one_and_list_carry_the_fields(
        self, run_metadata: Any, name: str, legacy: bool, status: str | None
    ) -> None:
        row = self._opt_row(run_metadata, name)
        one = AsyncMock()
        one_result = MagicMock()
        one_result.scalar_one_or_none.return_value = row
        one.execute = AsyncMock(return_value=one_result)
        detail = await optimize_module.get_optimization_run(row.id, one)
        assert (detail.legacy_mtf_context, detail.mtf_context_status) == (legacy, status)

        many = AsyncMock()
        many_result = MagicMock()
        many_result.scalars.return_value.all.return_value = [row]
        many.execute = AsyncMock(return_value=many_result)
        listing = await optimize_module.list_optimization_runs(many)
        assert [(r.legacy_mtf_context, r.mtf_context_status) for r in listing] == [(legacy, status)]
        wire = listing[0].model_dump(by_alias=True, mode="json")
        assert wire["legacyMtfContext"] is legacy and wire["mtfContextStatus"] == status
        assert _SENTINEL not in json.dumps(detail.model_dump(by_alias=True, mode="json"))

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, False),
            ({"strategyName": _CONSUMER, "paramGrid": {"rsi_period": [10, 14]}}, True),
        ],
        ids=["non-consumer", "consumer"],
    )
    def test_p3_44_optimize_post_passes_the_stored_class_name_to_the_flag(
        self,
        c: _C,
        monkeypatch: pytest.MonkeyPatch,
        overrides: dict[str, Any],
        expected: bool,
    ) -> None:
        # CR-02 / M24.  A fresh run is marked, so the name is never consulted; unmark it.  The
        # POST must then feed ``orm_run.strategy_name`` (the class name) to the one rule.  With
        # the name dropped the helper falls back to ``run_metadata["strategy_name"]`` (never
        # written), fails closed and flags the NON-consumer row True: that row kills M24.  The
        # consumer row guards against a constant non-consumer name.
        real = optimize_module.stamp_cost_config

        def unmarked(config: Any, *args: Any, **kwargs: Any) -> Any:
            out = real(config, *args, **kwargs)
            del config["engine_semantics"]["mtf_context"]
            return out

        monkeypatch.setattr(optimize_module, "stamp_cost_config", unmarked)
        resp = _post_opt(c, _bars(500), **overrides)
        assert resp.status_code == 200, resp.text
        orm_run = _opt_run(c)
        assert "mtf_context" not in orm_run.run_metadata["engine_semantics"]
        assert orm_run.strategy_name.endswith("Strategy")  # the class name, not the registry key
        assert resp.json()["legacyMtfContext"] is expected


# ===========================================================================
# T-30  leaderboard
# ===========================================================================


def _metrics(ret: float, trades: int = 40) -> dict[str, Any]:
    return {
        "backtest_metrics": {
            "total_trades": trades,
            "confidence_flag": "high",
            "total_return_pct": ret,
        }
    }


class TestT30Leaderboard:
    def test_p3_27_an_unmarked_consumer_is_ineligible_a_marked_one_is(self) -> None:
        assert leaderboard_eligible(50, "high", _cfg(with_marker=False)) is False
        assert leaderboard_eligible(50, "high", _cfg()) is True

    def test_p3_28_an_unmarked_non_consumer_and_the_other_halves_are_unaffected(self) -> None:
        assert leaderboard_eligible(50, "high", _cfg(strategy="ma_crossover", with_marker=False))
        assert leaderboard_eligible(9, "high", _cfg()) is False  # trades
        assert leaderboard_eligible(50, "low", _cfg()) is False  # confidence
        stress = _cfg(profile=get_cost_profile("coinbase_stress_t80"))
        assert leaderboard_eligible(50, "high", stress) is False  # cost
        no_metrics = _cfg()
        del no_metrics["engine_semantics"]["metrics"]
        assert leaderboard_eligible(50, "high", no_metrics) is False  # metrics

    def test_p3_29_a_missing_or_non_str_strategy_name_is_ineligible_when_unmarked(self) -> None:
        for name in (None, 3, ["rsi_mean_reversion"]):
            assert leaderboard_eligible(50, "high", _cfg(with_marker=False, strategy=name)) is False
        assert (
            leaderboard_eligible(50, "high", _cfg(with_marker=False, with_strategy=False)) is False
        )

    def test_p3_45_an_empty_or_blank_name_is_ineligible_when_unmarked(self) -> None:
        for name in ("", " ", "\t\n"):
            cfg = _cfg(with_marker=False, strategy=name)
            assert is_leaderboard_mtf_eligible(cfg) is False
            assert leaderboard_eligible(50, "high", cfg) is False

    def test_p3_30_the_mtf_half_is_the_negation_of_the_flag(self) -> None:
        for _id, config, _mode, name, _expected in _TABLE:
            if name is not None or config is None:
                continue
            flag = _flags(config, "backtest")["legacy_mtf_context"]
            assert is_leaderboard_mtf_eligible(config) is (not flag)

    @pytest.mark.parametrize("status", ["not_used", "unavailable", "partial", "pit"])
    def test_p3_31_every_status_of_a_marked_consumer_stays_eligible(self, status: str) -> None:
        # CF-P2-02 / UD-3(a): status labels, never excludes.
        cfg = _cfg(status=status)
        assert leaderboard_eligible(50, "high", cfg) is True
        resp = run_orm_to_response(
            _run_row({**cfg, "backtest_metrics": {"confidence_flag": "high"}}, n_trades=50)
        )
        assert resp.leaderboard_eligible is True and resp.mtf_context_status == status

    def test_p3_32_reason_source_fields_and_ingest_cutoff_never_decide(self) -> None:
        # CF-P2-01 / CF-P2-03: bounded free text and an unvalidated cutoff are not inputs.
        base = _cfg()
        for tamper in (
            {"reason": "legacy"},
            {"reason": "no_signal_source"},
            {"reason": _SENTINEL},
            {"reason": ["x"]},
            {"source": "pit"},
            {"source": 5},
            {"fields": {"fear_greed_index": "pit"}},
            {"fields": "junk"},
            {"ingest_cutoff": "not-an-iso-string"},
            {"ingest_cutoff": 12},
            {"series_digest": "zz"},
        ):
            cfg = {**base, "mtf_context": {**base["mtf_context"], **tamper}}
            assert leaderboard_eligible(50, "high", cfg) is True
            assert _flags(cfg)["legacy_mtf_context"] is False
            assert _flags(cfg)["mtf_context_status"] == "unavailable"
            unmarked = {**_cfg(with_marker=False), "mtf_context": cfg["mtf_context"]}
            assert leaderboard_eligible(50, "high", unmarked) is False

    @pytest.mark.parametrize(
        "tampered",
        [_SENTINEL, "PIT", "pit ", "", 5, None, True, ["pit"], {"pit": 1}, 1.5],
        ids=["sentinel", "case", "space", "empty", "int", "none", "bool", "list", "dict", "float"],
    )
    def test_p3_33_a_tampered_status_gives_none_and_is_not_echoed(self, tampered: Any) -> None:
        cfg = _cfg(status=tampered)
        assert _flags(cfg)["mtf_context_status"] is None
        resp = run_orm_to_detail_response(_run_row(cfg))
        assert resp.mtf_context_status is None
        dumped = resp.model_dump(by_alias=True, mode="json")
        assert dumped["mtfContextStatus"] is None
        # The raw ``config`` JSONB is echoed by the existing response contract (like every
        # other stored key); the DERIVED fields must not carry the tampered text.
        derived = {k: v for k, v in dumped.items() if k != "config"}
        assert _SENTINEL not in json.dumps(derived, default=str)

    def test_p3_34_a_malformed_block_gives_none(self) -> None:
        for block in (None, "pit", ["pit"], 5, {}, {"status": None}):
            assert _flags({**_cfg(), "mtf_context": block})["mtf_context_status"] is None
        cfg = _cfg()
        del cfg["mtf_context"]
        assert _flags(cfg)["mtf_context_status"] is None

    async def test_p3_35_summary_counts_in_order_cost_metrics_mtf(self) -> None:
        d6 = get_cost_profile("coinbase_advanced_d6")
        stress = get_cost_profile("coinbase_stress_t80")

        def _no_metrics_marker(cfg: dict[str, Any]) -> dict[str, Any]:
            del cfg["engine_semantics"]["metrics"]
            return cfg

        configs = [
            # eligible: marked consumers (any status) and unmarked non-consumers
            {"initial_capital": "1000", **_cfg(status="pit"), **_metrics(0.10)},
            {"initial_capital": "1000", **_cfg(status="partial", profile=d6), **_metrics(0.20)},
            {"initial_capital": "1000", **_cfg(status="unavailable"), **_metrics(0.05)},
            {
                "initial_capital": "1000",
                **_cfg(strategy="ma_crossover", with_marker=False, status="not_used"),
                **_metrics(0.07),
            },
            # mtf exclusion: pass trades + confidence + cost + metrics, consumer w/o marker
            {"initial_capital": "1000", **_cfg(with_marker=False), **_metrics(9.0)},
            {
                "initial_capital": "1000",
                **_cfg(with_marker=False, strategy="ensemble"),
                **_metrics(8.0),
            },
            {
                "initial_capital": "1000",
                **_cfg(with_marker=False, with_strategy=False),
                **_metrics(7.0),
            },
            {"initial_capital": "1000", **_cfg(with_marker=False, strategy=4), **_metrics(6.5)},
            # metrics exclusion FIRST (also unmarked consumer): counted as metrics only
            {
                "initial_capital": "1000",
                **_no_metrics_marker(_cfg(with_marker=False)),
                **_metrics(6.0),
            },
            {"initial_capital": "1000", **_no_metrics_marker(_cfg()), **_metrics(5.5)},
            # cost exclusion FIRST (also unmarked consumer): counted as cost only
            {"initial_capital": "1000", **_cfg(with_marker=False, profile=stress), **_metrics(5.0)},
            {"initial_capital": "1000", **_cfg(profile=stress), **_metrics(4.0)},
            {"initial_capital": "1000", **_metrics(3.5)},  # nothing: cost first
            # thresholds only: counted nowhere
            {"initial_capital": "1000", **_cfg(with_marker=False), **_metrics(3.0, trades=3)},
        ]
        db = AsyncMock()
        status_result = MagicMock()
        status_result.all.return_value = []
        trade_result = MagicMock()
        trade_result.one.return_value = SimpleNamespace(total=0, wins=0, losses=0, pnl=0, fees=0)
        cfg_result = MagicMock()
        cfg_result.all.return_value = [(cfg,) for cfg in configs]
        db.execute = AsyncMock(side_effect=[status_result, trade_result, cfg_result])
        resp = await portfolio_module.get_aggregate_portfolio(db=db)
        assert resp.eligible_runs_count == 4
        assert resp.best_run_return_pct == pytest.approx(0.20)
        assert resp.worst_run_return_pct == pytest.approx(0.05)
        assert resp.legacy_mtf_context_runs_excluded == 4
        assert resp.legacy_metrics_runs_excluded == 2
        assert resp.legacy_cost_runs_excluded == 3
        wire = resp.model_dump(by_alias=True)
        assert wire["legacyMtfContextRunsExcluded"] == 4
        assert wire["legacyMetricsRunsExcluded"] == 2 and wire["legacyCostRunsExcluded"] == 3

    def test_p3_36_the_counter_defaults_to_zero_and_the_helper_is_the_only_rule(self) -> None:
        assert (
            AggregatePortfolioResponse.model_fields["legacy_mtf_context_runs_excluded"].default == 0
        )
        src = inspect.getsource(portfolio_module.get_aggregate_portfolio)
        assert "leaderboard_eligible(" in src
        assert "LEGACY_MTF_CONSUMER" not in src and "KNOWN_MTF_CONTEXT" not in src

    def test_p3_37_the_round_trip_through_jsonb_text(self) -> None:
        cfg = json.loads(json.dumps(_cfg(status="partial")))
        assert _flags(cfg) == {"legacy_mtf_context": False, "mtf_context_status": "partial"}


# ===========================================================================
# T-31  API vs script parity
# ===========================================================================


def _result_key(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("total_trades", "total_return_pct", "max_drawdown_pct", "win_rate", "sharpe_ratio")
    return {k: metrics.get(k) for k in keys}


class TestT31ApiScriptParity:
    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    async def test_p3_38_api_path_with_clients_installed_equals_a_direct_runner(
        self, name: str, c: _C, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bars = _bars(300)
        registry = runs_module._get_strategy_registry()
        # the script: no clients at all
        _uninstall_clients(monkeypatch)
        runner = BacktestRunner(
            strategies=[registry[name](strategy_id=f"{name}-script", params={})],
            symbols=[_SYM],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000.00"),
            seed=11,
            cost_profile=DEFAULT_COST_PROFILE,
            enable_circuit_breaker=BACKTEST_CIRCUIT_BREAKER_DEFAULT,
        )
        direct = await runner.run(dict(bars))

        # the API: the boot-warmed (here: raising) clients are installed
        touched = _install_raising_clients(monkeypatch)
        resp = _post_bt(
            c,
            bars,
            strategyName=name,
            strategyParams={},
            seed=11,
        )
        assert resp.status_code == 201, resp.text
        assert touched == []
        cfg = c.created_run().config
        assert cfg["mtf_context"]["status"] == "unavailable"
        assert cfg["seed"] == 11
        stored = cfg["backtest_metrics"]
        script = direct.model_dump(mode="json")
        assert _result_key(stored) == _result_key(script)
        assert stored["total_trades"] > 0  # the consumer really traded; not a vacuous match
        # the runner's own report is what the API stored
        assert runner.last_mtf_context is not None
        assert runner.last_mtf_context.to_metadata() == cfg["mtf_context"]


# ===========================================================================
# T-32  log hygiene
# ===========================================================================


def _fn_names(fn: Any) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(inspect.getsource(fn))) if isinstance(n, ast.Name)}


class TestT32LogHygiene:
    @pytest.mark.parametrize(
        "fn",
        [
            schemas_module.is_legacy_mtf_context,
            schemas_module.stored_mtf_context_status,
            schemas_module.is_leaderboard_mtf_eligible,
            schemas_module._is_mtf_consumer_name,
            mtf_context_response_fields,
        ],
        ids=lambda f: f.__name__,
    )
    def test_p3_39_the_flag_helpers_have_no_logger_and_no_exception_handling(self, fn: Any) -> None:
        tree = ast.parse(inspect.getsource(fn))
        assert not _fn_names(fn) & {"log", "logger", "logging", "structlog", "print", "exc"}
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert not calls & {"print", "str", "repr", "format"}  # no text is ever built
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Try, ast.Raise, ast.JoinedStr))]

    def test_p3_40_the_flag_paths_log_nothing_even_for_a_tampered_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = _cfg(with_marker=False, status=_SENTINEL)
        cfg["mtf_context"]["reason"] = _SENTINEL
        caplog.set_level(logging.DEBUG)
        with capture_logs() as events:
            for build in (run_orm_to_response, run_orm_to_detail_response):
                build(_run_row(cfg))
            _flags(cfg)
            leaderboard_eligible(50, "high", cfg)
        assert events == [] and caplog.records == []

    def test_p3_41_create_and_optimize_log_no_marker_payload_or_exception_text(
        self, c: _C, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        with capture_logs() as events:
            assert (
                _post_bt(
                    c, _bars(), strategyName="rsi_mean_reversion", strategyParams={}
                ).status_code
                == 201
            )
            assert _post_opt(c, _bars(500)).status_code == 200
        text = json.dumps(events, default=str) + "\n".join(r.getMessage() for r in caplog.records)
        for forbidden in ("series_digest", "ingest_cutoff", "insufficient_coverage", "no_source"):
            assert forbidden not in text
        assert not any("mtf_context" in str(e.get("event", "")) for e in events)
        assert not any("mtf_context" in e for e in events)

    def test_p3_42_the_new_router_lines_do_not_log(self) -> None:
        for rel, needle in (
            ("apps/api/routers/runs.py", "last_mtf_context"),
            ("apps/api/routers/optimize.py", "last_mtf_context"),
        ):
            lines = (_REPO / rel).read_text().splitlines()
            for i, line in enumerate(lines):
                if needle in line and "mtf_context" in line and "=" in line:
                    window = "\n".join(lines[max(0, i - 1) : i + 3])
                    assert "log." not in window and "logger." not in window

    def test_p3_43_alembic_has_no_new_revision(self) -> None:
        versions = sorted((_REPO / "infra/alembic/versions").glob("0*.py"))
        assert versions[-1].name.startswith("020_")
        assert not list((_REPO / "infra/alembic/versions").glob("021_*"))
