"""CF-24-01 P0 (SY-01-05): the live gate for strategies that consume external MTF context.

Acceptance tests T-01 and T-02 of ``reports/vp2-cf2401/synthesis-spec.md``.

T-01  Live create / promote / resume of rsi, dca, model_strategy and an ensemble with any
      consumer -> 422 ``live_context_consumer_blocked``; momentum and the other empty-set
      strategies pass; an unknown name is refused; ``run_live_engine`` refuses before any
      exchange I/O; paper and backtest are unaffected; the allowlist is pinned empty.
T-02  The existing live gates keep their order and codes; the new 422 carries the code only
      (no echo); ``"mtf_signals" not in run_orchestrator.py`` still holds.
"""

from __future__ import annotations

import ast
import re
import uuid
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routers.runs as runs_module
import trading.safety as safety_module
from api.config import get_settings
from api.schemas import RunCreateRequest
from api.services import kill_switch as _kill_switch
from tests.integration import test_wp18a_resume_endpoint as resume_mod
from tests.integration.test_wp18a_resume_endpoint import resume_app  # noqa: F401  (fixture)
from tests.integration.test_wp21_cost_api import (  # noqa: F401  (fixtures + HTTP harness)
    _BACKTEST,
    _C,
    _LIVE,
    _PAPER,
    CONFIRM_TOKEN,
    _clean_run_tasks,
    _hdr,
    _paper_source,
    _promote,
    app,  # fixture: requested by name in ``c`` below
)
from trading.safety import (
    LIVE_CONTEXT_CONSUMER_ALLOWLIST,
    LIVE_CONTEXT_CONSUMER_BLOCKED_CODE,
    live_context_consumer_refusal,
)
from trading.strategies import (
    DCARSIHybridStrategy,
    GridTradingStrategy,
    MomentumBreakoutStrategy,
    RSIMeanReversionStrategy,
)
from trading.strategies.ensemble import EnsembleStrategy

_REPO = Path(__file__).resolve().parents[2]
_CODE = "live_context_consumer_blocked"

#: Consumers at this commit (non-empty ``consumes_mtf_signals``), registry keys.
_CONSUMERS = ("rsi_mean_reversion", "dca_rsi_hybrid", "model_strategy")
#: Registry strategies whose consumed-field set is empty: they must still pass.
_NON_CONSUMERS = (
    "momentum_breakout",
    "ma_crossover",
    "breakout",
    "grid_trading",
    "sl_tp_reversion",
    "smoke_roundtrip",
)
_SAFETY_IMPORTS_AT_BD93C27: set[tuple[str, ...]] = {
    ("from", "__future__", "annotations"),
    ("from", "collections.abc", "Callable"),
    ("from", "dataclasses", "dataclass"),
    ("from", "datetime", "UTC"),
    ("from", "datetime", "date"),
    ("from", "datetime", "datetime"),
    ("from", "decimal", "Decimal"),
    ("from", "enum", "StrEnum"),
    ("from", "pydantic", "BaseModel"),
    ("from", "pydantic", "Field"),
    ("from", "pydantic", "SecretStr"),
    ("from", "pydantic", "model_validator"),
    ("from", "trading.portfolio", "RiskSnapshot"),
    ("from", "trading.risk", "RiskParameters"),
    ("from", "typing", "Any"),
    ("from", "typing", "Final"),
    ("import", "hmac"),
    ("import", "structlog"),
}
_BRACKET = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.05,
    "bracket_take_profit_pct": 0.1,
}


@pytest.fixture()
def c(app: Any):  # type: ignore[no-untyped-def]  # noqa: F811
    client = _C(app)
    yield client
    client.close()


def _live_body(strategy: str, **overrides: Any) -> dict[str, Any]:
    return {**_LIVE, "strategyName": strategy, "strategyParams": dict(_BRACKET), **overrides}


def _post_live(c: _C, body: dict[str, Any], *, token: str | None = CONFIRM_TOKEN) -> Any:
    extra = {"X-Live-Confirm-Token": token} if token is not None else {}
    spy = AsyncMock()
    with patch("api.routers.runs._run_live_engine", new=spy):
        resp = c.client.post("/api/v1/runs", json=body, headers=_hdr(**extra))
    resp.spy = spy  # type: ignore[attr-defined]
    return resp


# ---------------------------------------------------------------------------
# T-01  the pure helper
# ---------------------------------------------------------------------------


def _registry() -> dict[str, Any]:
    return dict(runs_module._get_strategy_registry())


class TestT01Helper:
    def test_the_allowlist_is_pinned_empty_and_immutable(self) -> None:
        assert LIVE_CONTEXT_CONSUMER_ALLOWLIST == frozenset()
        assert type(LIVE_CONTEXT_CONSUMER_ALLOWLIST) is frozenset
        assert LIVE_CONTEXT_CONSUMER_BLOCKED_CODE == _CODE

    def test_the_fixture_lists_match_the_registry_declarations(self) -> None:
        reg = _registry()
        assert set(_CONSUMERS) | set(_NON_CONSUMERS) == set(reg)
        for name in _CONSUMERS:
            assert reg[name].consumes_mtf_signals, name
        for name in _NON_CONSUMERS:
            assert reg[name].consumes_mtf_signals == frozenset(), name

    @pytest.mark.parametrize("name", _CONSUMERS)
    def test_a_consumer_class_is_refused(self, name: str) -> None:
        assert live_context_consumer_refusal(name, _registry()[name]) == _CODE

    @pytest.mark.parametrize("name", _NON_CONSUMERS)
    def test_every_empty_set_class_passes(self, name: str) -> None:
        assert live_context_consumer_refusal(name, _registry()[name]) is None

    def test_momentum_breakout_passes_as_class_and_as_instance(self) -> None:
        assert MomentumBreakoutStrategy.consumes_mtf_signals == frozenset()
        assert live_context_consumer_refusal("momentum_breakout", MomentumBreakoutStrategy) is None
        instance = MomentumBreakoutStrategy(strategy_id="m-1")
        assert live_context_consumer_refusal("momentum_breakout", instance) is None

    def test_a_consumer_instance_is_refused(self) -> None:
        instance = RSIMeanReversionStrategy(strategy_id="rsi-1")
        assert live_context_consumer_refusal("rsi_mean_reversion", instance) == _CODE

    @pytest.mark.parametrize(
        "strategy",
        [None, object(), object, MagicMock(), MagicMock(spec=[]), "rsi_mean_reversion", 5, [1]],
    )
    def test_an_unknown_or_unreadable_strategy_is_refused_fail_closed(
        self, strategy: object
    ) -> None:
        # a failed lookup (None), a class without the declaration, a test double: all refused
        assert live_context_consumer_refusal("momentum_breakout", strategy) == _CODE

    def test_a_class_without_the_declaration_is_refused(self) -> None:
        class _Bare:
            pass

        assert live_context_consumer_refusal("bare", _Bare) == _CODE
        assert live_context_consumer_refusal("bare", _Bare()) == _CODE

    @pytest.mark.parametrize(
        "bad",
        [
            ["fear_greed_index"],
            {"fear_greed_index"},
            ("fear_greed_index",),
            (),
            set(),
            [],
            None,
            "x",
            0,
        ],
    )
    def test_a_non_frozenset_declaration_is_refused(self, bad: object) -> None:
        class _Cls:
            consumes_mtf_signals: ClassVar[object] = bad

        class _Inst:
            def mtf_signal_fields(self) -> object:
                return bad

        assert live_context_consumer_refusal("momentum_breakout", _Cls) == _CODE
        assert live_context_consumer_refusal("momentum_breakout", _Inst()) == _CODE

    def test_any_failure_while_reading_the_set_is_refused_without_leaking(self) -> None:
        class _Boom:
            def mtf_signal_fields(self) -> frozenset[str]:
                raise RuntimeError("sentinel-secret-text")

        assert live_context_consumer_refusal("momentum_breakout", _Boom()) == _CODE

    def test_the_result_is_the_code_or_none_only(self) -> None:
        for strategy in (RSIMeanReversionStrategy, MomentumBreakoutStrategy, None):
            out = live_context_consumer_refusal("x", strategy)
            assert out is None or out == _CODE

    def test_an_ensemble_with_a_consumer_sub_is_refused(self) -> None:
        rsi = RSIMeanReversionStrategy(strategy_id="rsi-1")
        mom = MomentumBreakoutStrategy(strategy_id="mom-1")
        ens = EnsembleStrategy("ens-1", [mom, rsi])
        assert ens.mtf_signal_fields()
        assert live_context_consumer_refusal("ensemble", ens) == _CODE

    def test_an_ensemble_with_a_dca_sub_is_refused(self) -> None:
        ens = EnsembleStrategy(
            "ens-2",
            [MomentumBreakoutStrategy(strategy_id="m"), DCARSIHybridStrategy(strategy_id="d")],
        )
        assert live_context_consumer_refusal("ensemble", ens) == _CODE

    def test_an_ensemble_of_empty_set_subs_passes_by_the_union(self) -> None:
        ens = EnsembleStrategy(
            "ens-3",
            [MomentumBreakoutStrategy(strategy_id="a"), GridTradingStrategy(strategy_id="b")],
        )
        assert ens.mtf_signal_fields() == frozenset()
        assert live_context_consumer_refusal("ensemble", ens) is None

    def test_an_ensemble_class_is_refused_by_the_base_default(self) -> None:
        assert live_context_consumer_refusal("ensemble", EnsembleStrategy) == _CODE

    def test_the_allowlist_lifts_one_strategy_and_only_that_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            safety_module, "LIVE_CONTEXT_CONSUMER_ALLOWLIST", frozenset({"rsi_mean_reversion"})
        )
        reg = _registry()
        assert (
            live_context_consumer_refusal("rsi_mean_reversion", reg["rsi_mean_reversion"]) is None
        )
        assert live_context_consumer_refusal("dca_rsi_hybrid", reg["dca_rsi_hybrid"]) == _CODE
        assert live_context_consumer_refusal("model_strategy", reg["model_strategy"]) == _CODE
        # the allowlist never admits an unreadable strategy, whatever its name
        assert live_context_consumer_refusal("rsi_mean_reversion", None) == _CODE
        assert live_context_consumer_refusal("rsi_mean_reversion", MagicMock()) == _CODE

    def test_a_non_string_name_never_matches_the_allowlist(self) -> None:
        assert live_context_consumer_refusal(None, RSIMeanReversionStrategy) == _CODE  # type: ignore[arg-type]

    def test_the_helper_is_pure_and_imports_nothing_new_into_safety(self) -> None:
        import ast

        src = (_REPO / "packages/trading/safety.py").read_text()
        tree = ast.parse(src)
        imported: set[tuple[str, ...]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {("import", a.name) for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {("from", node.module or "", a.name) for a in node.names}
        # exactly the import set of bd93c27: nothing new (no registry, strategies or apps)
        assert imported == _SAFETY_IMPORTS_AT_BD93C27
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "live_context_consumer_refusal"
        )
        assert not [n for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))]
        called = {
            (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
        }
        assert not called & {"open", "print", "warning", "error", "info", "exception", "getenv"}
        assert "str(exc)" not in (ast.get_source_segment(src, fn) or "")


# ---------------------------------------------------------------------------
# T-01  API: create / promote / resume
# ---------------------------------------------------------------------------


class TestT01Create:
    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    def test_live_create_of_a_consumer_is_422_with_the_code_only(self, c: _C, name: str) -> None:
        resp = _post_live(c, _live_body(name))
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        resp.spy.assert_not_called()
        c.db.add.assert_not_called()

    def test_live_create_of_model_strategy_is_refused_before_nothing_runs(self, c: _C) -> None:
        # model_strategy is already demoted for live (availability 422); it must never be 201.
        resp = _post_live(c, _live_body("model_strategy"))
        assert resp.status_code == 422, resp.text
        resp.spy.assert_not_called()
        c.db.add.assert_not_called()

    def test_model_strategy_is_refused_by_this_gate_even_if_availability_allowed_it(
        self, c: _C, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runs_module, "is_mode_allowed", lambda *_a, **_k: True)
        resp = _post_live(c, _live_body("model_strategy", strategyParams=dict(_BRACKET)))
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        resp.spy.assert_not_called()

    def test_the_response_never_echoes_input(self, c: _C) -> None:
        resp = _post_live(c, _live_body("RSI-Mean-Reversion", initialCapital="12345.67"))
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        low = resp.text.lower()
        for needle in ("rsi", "12345", "mean", CONFIRM_TOKEN.lower(), "btc"):
            assert needle not in low

    @pytest.mark.parametrize("name", ["momentum_breakout", "grid_trading"])
    def test_an_empty_set_strategy_still_creates_live(self, c: _C, name: str) -> None:
        params: dict[str, Any] = (
            {**_BACKTEST["strategyParams"]} if name == "momentum_breakout" else {}
        )
        resp = _post_live(c, {**_LIVE, "strategyName": name, "strategyParams": params})
        assert resp.status_code == 201, resp.text
        resp.spy.assert_called_once()

    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    def test_paper_create_of_a_consumer_is_unaffected(self, c: _C, name: str) -> None:
        spy = AsyncMock()
        body = {**_PAPER, "strategyName": name, "strategyParams": dict(_BRACKET)}
        with patch("api.routers.runs._run_paper_engine", new=spy):
            resp = c.client.post("/api/v1/runs", json=body, headers=_hdr())
        assert resp.status_code == 201, resp.text
        spy.assert_called_once()

    def test_backtest_create_of_a_consumer_is_unaffected(self, c: _C) -> None:
        from tests.conftest import make_bars

        bars = {"BTC/USD": make_bars(120, symbol="BTC/USD")}
        body = {
            **_BACKTEST,
            "strategyName": "rsi_mean_reversion",
            "strategyParams": dict(_BRACKET),
        }
        with patch("api.routers.runs._fetch_bars_for_backtest", return_value=bars):
            resp = c.client.post("/api/v1/runs", json=body, headers=_hdr())
        assert resp.status_code == 201, resp.text


class TestT01Promote:
    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    def test_promote_of_a_consumer_paper_run_is_422_with_the_code_only(
        self, c: _C, name: str
    ) -> None:
        source = _paper_source(uuid.uuid4(), strategy_name=name)
        resp, live_spy = _promote(c, source)
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        live_spy.assert_not_called()
        c.db.add.assert_not_called()
        assert source.config["strategy_name"] == name  # the source run is untouched

    def test_promote_of_an_unregistered_strategy_keeps_its_existing_400(self, c: _C) -> None:
        source = _paper_source(uuid.uuid4(), strategy_name="nope")
        resp, live_spy = _promote(c, source)
        assert resp.status_code in (400, 422), resp.text
        assert resp.json().get("detail") != {"code": _CODE}
        live_spy.assert_not_called()

    def test_kill_switch_409_wins_over_the_new_422_on_promote(self, c: _C) -> None:
        # RK-P0-04
        source = _paper_source(uuid.uuid4(), strategy_name="rsi_mean_reversion")
        _kill_switch.latch_in_memory("cf2401-test")
        try:
            resp, live_spy = _promote(c, source)
        finally:
            _kill_switch.reset_state_for_tests()
        assert resp.status_code == 409, resp.text
        assert resp.json() == {"detail": {"code": "kill_switch_active"}}
        live_spy.assert_not_called()

    def test_promote_of_an_empty_set_strategy_still_works(self, c: _C) -> None:
        resp, live_spy = _promote(c, _paper_source(uuid.uuid4()))  # grid_trading
        assert resp.status_code == 201, resp.text
        live_spy.assert_called_once()


class TestT01Resume:
    @staticmethod
    def _resume(
        monkeypatch: pytest.MonkeyPatch,
        app_: Any,
        name: str,
        mode: str = "normal",
        *,
        kill_switch_count: int = 0,
        **config_overrides: Any,
    ) -> tuple[Any, Any, Any, AsyncMock]:
        real = _registry()[name]
        monkeypatch.setattr(runs_module, "_STRATEGY_REGISTRY", {name: real})
        run_row = resume_mod._make_run_row(status="orphaned")
        run_row.config["strategy_name"] = name
        run_row.config["bracket_config"] = dict(_BRACKET)  # real classes need an exit config
        run_row.config["allow_pyramiding"] = False
        run_row.config.update(config_overrides)
        session = resume_mod._DispatchSession(
            run_row=run_row,
            runs_update_rowcounts=[1, 1],
            kill_switch_count=kill_switch_count,
        )
        client = resume_mod._client(app_, session)

        async def _scan(db: Any, run: Any, exchange: Any, *, fence: Any = None) -> Any:
            from api.services.run_recovery import ImportReport

            return ImportReport()

        import api.services.run_recovery as run_recovery_module

        monkeypatch.setattr(run_recovery_module, "scan_and_import", _scan)
        spy = AsyncMock()
        with patch("api.routers.runs._run_live_engine", spy):
            resp = client.post(
                f"/api/v1/runs/{resume_mod.RUN_ID}/resume?mode={mode}",
                headers={
                    "X-Live-Confirm-Token": resume_mod.CONFIRM_TOKEN,
                    "X-Admin-Key": resume_mod.ADMIN_KEY,
                },
            )
        runs_module._RUN_TASKS.clear()
        return resp, run_row, session, spy

    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    def test_resume_of_a_consumer_is_422_with_the_code_only_before_the_cas(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
        name: str,
    ) -> None:
        resp, run_row, session, spy = self._resume(monkeypatch, resume_app, name)
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        spy.assert_not_called()
        assert run_row.status == "orphaned"  # untouched: before the CAS
        session.commit.assert_not_awaited()

    @pytest.mark.parametrize("name", ["rsi_mean_reversion", "dca_rsi_hybrid"])
    def test_a_protective_resume_of_a_consumer_is_not_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
        name: str,
    ) -> None:
        # G-11: protective drops every entry; refusing it would strand the exits.
        resp, _row, _session, spy = self._resume(monkeypatch, resume_app, name, "protective")
        assert resp.status_code == 200, resp.text
        assert _CODE not in resp.text
        spy.assert_called_once()

    @pytest.mark.parametrize("name", ["grid_trading", "momentum_breakout"])
    def test_resume_of_an_empty_set_strategy_still_resumes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
        name: str,
    ) -> None:
        resp, _row, _session, _spy = self._resume(monkeypatch, resume_app, name)
        assert resp.status_code == 200, resp.text

    def test_resume_pyramiding_ban_keeps_its_code_and_wins_for_a_consumer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        # T-02 / E-1: every existing pre-CAS resume code wins over the new check.
        resp, run_row, session, spy = self._resume(
            monkeypatch, resume_app, "rsi_mean_reversion", allow_pyramiding=True
        )
        assert resp.status_code == 422, resp.text
        assert "live_pyramiding_forbidden" in resp.text
        assert _CODE not in resp.text
        spy.assert_not_called()
        assert run_row.status == "orphaned"
        session.commit.assert_not_awaited()

    def test_resume_invalid_exit_config_keeps_its_code_for_a_consumer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        resp, run_row, _session, spy = self._resume(
            monkeypatch,
            resume_app,
            "rsi_mean_reversion",
            bracket_config={**_BRACKET, "bracket_stop_loss_pct": 5.0},
        )
        assert resp.status_code == 422, resp.text
        assert "invalid_exit_config" in resp.text
        assert _CODE not in resp.text
        spy.assert_not_called()
        assert run_row.status == "orphaned"

    def test_a_post_start_kill_switch_is_masked_by_the_pre_cas_422(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        # RK-P0-03: S4 ``kill_switch_after_start`` runs AFTER the CAS, and the consumer
        # 422 runs BEFORE it, so the 422 masks S4.  Both outcomes are a refusal; the run
        # stays orphaned and nothing is committed.
        resp, run_row, session, spy = self._resume(
            monkeypatch, resume_app, "rsi_mean_reversion", kill_switch_count=1
        )
        assert resp.status_code == 422, resp.text
        assert resp.json() == {"detail": {"code": _CODE}}
        spy.assert_not_called()
        assert run_row.status == "orphaned"
        session.commit.assert_not_awaited()

    def test_kill_switch_409_wins_over_the_new_422_on_normal_resume(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resume_app: Any,  # noqa: F811
    ) -> None:
        # RK-P0-04
        _kill_switch.latch_in_memory("cf2401-test")
        try:
            resp, _row, _session, spy = self._resume(monkeypatch, resume_app, "rsi_mean_reversion")
        finally:
            _kill_switch.reset_state_for_tests()
        assert resp.status_code == 409, resp.text
        assert resp.json() == {"detail": {"code": "kill_switch_active"}}
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# T-01  defence in depth in run_live_engine
# ---------------------------------------------------------------------------


def _session_factory() -> tuple[MagicMock, MagicMock]:
    mock_run = MagicMock()
    mock_run.status = "running"
    result = MagicMock()
    result.scalar_one_or_none.return_value = mock_run
    db_mock = MagicMock()
    db_mock.execute = AsyncMock(return_value=result)
    db_mock.commit = AsyncMock()
    db_mock.rollback = AsyncMock()
    ctx = MagicMock()

    async def _aenter(self: Any = None) -> MagicMock:
        return db_mock

    async def _aexit(self: Any = None, *args: Any) -> bool:
        return False

    ctx.__aenter__ = _aenter
    ctx.__aexit__ = _aexit
    return MagicMock(return_value=MagicMock(return_value=ctx)), mock_run


async def _call_live_engine(strategy_cls: Any, name: str, **extra: Any) -> None:
    from api.services.run_orchestrator import run_live_engine
    from common.types import TimeFrame
    from trading.costs import DEFAULT_COST_PROFILE

    await run_live_engine(
        run_id_str=str(uuid.uuid4()),
        strategy_cls=strategy_cls,
        strategy_name=name,
        strategy_params={},
        symbols=["BTC/USD"],
        timeframe=TimeFrame.ONE_HOUR,
        initial_capital="10000",
        cost_profile=DEFAULT_COST_PROFILE,
        **extra,
    )


class TestT01Orchestrator:
    @pytest.mark.parametrize(
        ("cls", "name"),
        [
            (RSIMeanReversionStrategy, "rsi_mean_reversion"),
            (DCARSIHybridStrategy, "dca_rsi_hybrid"),
            (object, "momentum_breakout"),  # no declaration: fail-closed
            (None, "momentum_breakout"),  # failed lookup: fail-closed
            (MagicMock(), "momentum_breakout"),  # unreadable declaration: fail-closed
        ],
    )
    async def test_run_live_engine_refuses_before_any_exchange_io(
        self, cls: Any, name: str
    ) -> None:
        sf, mock_run = _session_factory()
        settings_spy = MagicMock(side_effect=AssertionError("settings must not be reached"))
        exchange_spy = MagicMock(side_effect=AssertionError("exchange must not be built"))
        with (
            patch("api.config.get_settings", settings_spy),
            patch("api.db.session.get_session_factory", sf),
            patch("api.services.run_orchestrator.build_live_ccxt_exchange", exchange_spy),
        ):
            await _call_live_engine(cls, name)
        settings_spy.assert_not_called()
        exchange_spy.assert_not_called()
        assert mock_run.status == "error"

    async def test_the_raised_error_carries_the_code_only(self) -> None:
        import structlog

        sf, _run = _session_factory()
        with (
            structlog.testing.capture_logs() as logs,
            patch("api.config.get_settings", MagicMock(side_effect=AssertionError)),
            patch("api.db.session.get_session_factory", sf),
        ):
            await _call_live_engine(RSIMeanReversionStrategy, "rsi_mean_reversion")
        err = [e for e in logs if e.get("event") == "runs.live_engine_error"]
        assert len(err) == 1
        src = (_REPO / "apps/api/services/run_orchestrator.py").read_text()
        assert "raise RuntimeError(_refusal)" in src  # the helper's closed code, nothing else

    async def test_an_empty_set_strategy_passes_the_guard(self) -> None:
        sf, mock_run = _session_factory()
        reached = MagicMock(side_effect=RuntimeError("stop-after-guard"))
        with (
            patch("api.config.get_settings", reached),
            patch("api.db.session.get_session_factory", sf),
        ):
            await _call_live_engine(MomentumBreakoutStrategy, "momentum_breakout")
        reached.assert_called_once()  # went past the guard to the settings read
        assert mock_run.status == "error"  # ... which this test made fail on purpose

    async def test_a_protective_resume_of_a_consumer_is_exempt_in_the_orchestrator(self) -> None:
        sf, mock_run = _session_factory()
        reached = MagicMock(side_effect=RuntimeError("stop-after-guard"))
        with (
            patch("api.config.get_settings", reached),
            patch("api.db.session.get_session_factory", sf),
        ):
            await _call_live_engine(
                RSIMeanReversionStrategy, "rsi_mean_reversion", protective_mode=True
            )
        reached.assert_called_once()  # the guard was skipped: settings were reached
        assert mock_run.status == "error"  # ... which this test made fail on purpose

    async def test_the_raised_message_is_exactly_the_code(self) -> None:
        import sys

        sf, _run = _session_factory()
        seen: list[BaseException | None] = []

        async def _capture(**_kw: Any) -> None:
            seen.append(sys.exc_info()[1])  # the exception being handled by the except clause

        with (
            patch("api.config.get_settings", MagicMock(side_effect=AssertionError)),
            patch("api.db.session.get_session_factory", sf),
            patch("api.services.run_orchestrator._cancel_engine_side_tasks", _capture),
        ):
            await _call_live_engine(RSIMeanReversionStrategy, "rsi_mean_reversion")
        first = seen[0]
        assert type(first) is RuntimeError
        assert str(first) == _CODE
        assert first.args == (_CODE,)

    def test_the_orchestrator_guard_is_inside_the_not_protective_block(self) -> None:
        src = (_REPO / "apps/api/services/run_orchestrator.py").read_text()
        body = src[src.index("async def run_live_engine(") :]
        block_start = body.index("        if not protective_mode:\n            validate_smoke_run(")
        block_end = body.index("        settings = get_settings()")
        block = body[block_start:block_end]
        assert (
            "            _refusal = live_context_consumer_refusal(strategy_name, strategy_cls)"
            in block
        )
        assert block.index("validate_smoke_run(") < block.index("live_context_consumer_refusal(")
        assert (
            "            if _refusal is not None:\n                raise RuntimeError(_refusal)"
            in block
        )

    def test_the_guard_sits_after_the_existing_checks_and_before_any_io(self) -> None:
        src = (_REPO / "apps/api/services/run_orchestrator.py").read_text()
        start = src.index("async def run_live_engine(")
        body = src[start:]
        order = [
            body.index("live_pyramiding_forbidden"),
            body.index('mode="live",\n                symbols=symbols'),
            body.index("live_context_consumer_refusal(strategy_name, strategy_cls)"),
            body.index("settings = get_settings()"),
            body.index("build_live_ccxt_exchange(settings)"),
        ]
        assert order == sorted(order)
        assert src.count("live_context_consumer_refusal(") == 1  # one guard call only


# ---------------------------------------------------------------------------
# T-02  existing gates keep order and codes; no echo; the orchestrator token pin
# ---------------------------------------------------------------------------


class TestT02ExistingGates:
    def test_layer_1_disabled_live_trading_stays_403_for_a_consumer(
        self, c: _C, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
        get_settings.cache_clear()
        resp = _post_live(c, _live_body("rsi_mean_reversion"))
        assert resp.status_code == 403, resp.text
        assert _CODE not in resp.text

    def test_layer_3_wrong_confirm_token_stays_403_for_a_consumer(self, c: _C) -> None:
        resp = _post_live(c, _live_body("rsi_mean_reversion"), token="wrong-token")  # noqa: S106
        assert resp.status_code == 403, resp.text
        assert _CODE not in resp.text

    def test_missing_confirm_token_stays_403_for_a_consumer(self, c: _C) -> None:
        resp = _post_live(c, _live_body("rsi_mean_reversion"), token=None)
        assert resp.status_code == 403, resp.text
        assert _CODE not in resp.text

    def test_kill_switch_409_still_wins_over_the_new_422(self, c: _C) -> None:
        _kill_switch.latch_in_memory("cf2401-test")
        try:
            resp = _post_live(c, _live_body("rsi_mean_reversion"))
        finally:
            _kill_switch.reset_state_for_tests()
        assert resp.status_code == 409, resp.text
        assert resp.json() == {"detail": {"code": "kill_switch_active"}}

    def test_pyramiding_ban_keeps_its_code_and_wins_for_a_consumer(self, c: _C) -> None:
        resp = _post_live(c, _live_body("dca_rsi_hybrid", allowPyramiding=True))
        assert resp.status_code == 422, resp.text
        assert "live_pyramiding_forbidden" in resp.text
        assert _CODE not in resp.text

    def test_duplicate_symbols_keep_their_422_for_a_consumer(self, c: _C) -> None:
        resp = _post_live(c, _live_body("rsi_mean_reversion", symbols=["BTC/USDT", "BTC/USDT"]))
        assert resp.status_code == 422, resp.text
        assert _CODE not in resp.text
        c.db.add.assert_not_called()

    def test_the_create_handler_order_is_unchanged_with_the_new_check_last(self) -> None:
        src = (_REPO / "apps/api/routers/runs.py").read_text()
        create = src[src.index("async def create_run(") : src.index("async def list_runs(")]
        marks = [
            "validate_run_exit_config(",
            "runs.live_trading_gate_failed",
            "runs.create_blocked_kill_switch_active",
            "runs.concurrency_cap_hit",
            "_reject_live_context_consumer(strategy_name, strategy_cls, log)",
            "await store.claim(",
        ]
        positions = [create.index(m) for m in marks]
        assert positions == sorted(positions), positions

    def test_the_resume_and_promote_order_is_unchanged_with_the_new_check_last(self) -> None:
        src = (_REPO / "apps/api/routers/runs.py").read_text()
        promote = src[src.index("async def promote_to_live(") : src.index("async def resume_run(")]
        marks = [
            "runs.promote_duplicate_symbols",
            "runs.promotion_live_gate_failed",
            "runs.promote_blocked_kill_switch_active",
            "_reject_live_context_consumer(promotion_strategy_name",
            "await store.claim(",
        ]
        positions = [promote.index(m) for m in marks]
        assert positions == sorted(positions), positions
        resume = src[src.index("async def resume_run(") :]
        marks = [
            "runs.resume_live_gate_failed",
            "runs.resume_duplicate_symbols",
            "runs.resume_exit_config_invalid",
            "Stop an existing run before resuming.",
            "_reject_live_context_consumer(strategy_name, strategy_cls, log)",
            "# Transaction (a): compare-and-set orphaned -> resuming",
        ]
        positions = [resume.index(m) for m in marks]
        assert positions == sorted(positions), positions

    def test_the_new_422_adds_no_request_input(self) -> None:
        fields = set(RunCreateRequest.model_fields)
        assert not {f for f in fields if re.search(r"context|consumer|allowlist", f, re.I)}
        src = (_REPO / "apps/api/routers/runs.py").read_text()
        helper = src[
            src.index("def _reject_live_context_consumer(") : src.index(
                "def _get_strategy_registry("
            )
        ]
        assert 'detail={"code": code}' in helper
        assert "live_context_consumer_refusal(strategy_name, strategy_cls)" in helper
        code = helper.split('"""')[2]  # the function body after its docstring
        assert "body" not in code and "request" not in code.lower()
        # the log line carries no field at all
        assert 'log.warning("runs.live_context_consumer_blocked")' in helper

    def test_the_orchestrator_token_pin_still_holds(self) -> None:
        src = (_REPO / "apps/api/services/run_orchestrator.py").read_text()
        assert "mtf_signals" not in src
        assert "consumes_mtf_signals" not in src
        assert "mtf_signal_fields" not in src
        for name in ("live_context_consumer_refusal", "LIVE_CONTEXT_CONSUMER_BLOCKED_CODE"):
            assert "mtf_signals" not in name
        assert "mtf_signals" not in _CODE

    def test_the_existing_wp24_orchestrator_pin_is_untouched(self) -> None:
        pin = (_REPO / "tests/unit/test_wp24_engine.py").read_text()
        assert '"mtf_signals" not in src' in pin

    def test_an_unknown_strategy_class_is_422_code_only_at_the_router_helper(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as ei:
            runs_module._reject_live_context_consumer("whatever", None, MagicMock())
        assert ei.value.status_code == 422
        assert ei.value.detail == {"code": _CODE}


# ---------------------------------------------------------------------------
# E-2  static pins: who passes ``protective_mode`` and who builds the live engine
# ---------------------------------------------------------------------------


class TestE2ProtectiveModeSource:
    _ENGINE_NAMES: ClassVar[set[str]] = {"run_live_engine", "_run_live_engine"}

    def test_protective_mode_is_passed_only_by_resume_run(self) -> None:
        # Whole-module walk (CR2-01): module-level statements, class bodies, methods and
        # nested functions are all seen; each hit is attributed to its innermost
        # enclosing function, or "<module>".
        tree = ast.parse((_REPO / "apps/api/routers/runs.py").read_text())
        found: list[tuple[str, str, str]] = []

        def visit(node: ast.AST, owner: str) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                owner = node.name
            if isinstance(node, ast.Call):
                found.extend(
                    (owner, ast.unparse(node.func), ast.unparse(kw.value))
                    for kw in node.keywords
                    if kw.arg == "protective_mode"
                )
            for child in ast.iter_child_nodes(node):
                visit(child, owner)

        visit(tree, "<module>")
        assert found == [("resume_run", "_run_live_engine", "mode == 'protective'")]

    def test_no_other_module_imports_or_calls_the_live_engine(self) -> None:
        # AST, not a substring check: apps/api/main.py and
        # apps/api/services/run_persistence.py mention the name in a comment and a
        # docstring.
        names = {"run_live_engine", "_run_live_engine"}
        skip = {
            _REPO / "apps/api/routers/runs.py",
            _REPO / "apps/api/services/run_orchestrator.py",
        }
        offenders: list[str] = []
        for root in ("apps", "packages", "scripts"):
            for path in sorted((_REPO / root).rglob("*.py")):
                if path in skip or "tests" in path.relative_to(_REPO).parts:
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom) and any(
                        a.name in names for a in node.names
                    ):
                        offenders.append(f"{path.relative_to(_REPO)}:{node.lineno} import")
                    elif isinstance(node, ast.Call):
                        fn = node.func
                        called = None
                        if isinstance(fn, ast.Name):
                            called = fn.id
                        elif isinstance(fn, ast.Attribute):
                            called = fn.attr
                        if called in names:
                            offenders.append(f"{path.relative_to(_REPO)}:{node.lineno} call")
        assert offenders == []

    def test_no_star_unpack_into_the_live_engine_call(self) -> None:
        # SEC-P0-10 / CR2-01: no ``**`` into the engine call, and no stored reference
        # (``_eng = _run_live_engine`` would hide a later ``_eng(**kwargs)``): every
        # engine name in runs.py must be the callee of a Call.
        tree = ast.parse((_REPO / "apps/api/routers/runs.py").read_text())
        callees = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)}
        refs = [
            node
            for node in ast.walk(tree)
            if (isinstance(node, ast.Name) and node.id in self._ENGINE_NAMES)
            or (isinstance(node, ast.Attribute) and node.attr in self._ENGINE_NAMES)
        ]
        stored = [node.lineno for node in refs if node not in callees]
        bad = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and node.func in refs
            and any(kw.arg is None for kw in node.keywords)
        ]
        assert stored == []
        assert bad == []

    def test_no_other_module_references_the_live_engine(self) -> None:
        # Any reference, not only a call: catches partial(), getattr(..., "name"),
        # a stored reference and a star import.  No 'tests' directory exclusion.
        skip = {
            _REPO / "apps/api/routers/runs.py",
            _REPO / "apps/api/services/run_orchestrator.py",
        }
        offenders: list[str] = []
        for root in ("apps", "packages", "scripts"):
            for path in sorted((_REPO / root).rglob("*.py")):
                if path in skip:
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if (
                        (isinstance(node, ast.Name) and node.id in self._ENGINE_NAMES)
                        or (isinstance(node, ast.Attribute) and node.attr in self._ENGINE_NAMES)
                        or (isinstance(node, ast.Constant) and node.value in self._ENGINE_NAMES)
                        or (isinstance(node, ast.alias) and node.name in self._ENGINE_NAMES)
                        or (
                            isinstance(node, ast.ImportFrom)
                            and (node.module or "").endswith("run_orchestrator")
                            and any(a.name == "*" for a in node.names)
                        )
                    ):
                        offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        assert offenders == []

    def test_the_orchestrator_never_calls_its_own_live_engine(self) -> None:
        tree = ast.parse((_REPO / "apps/api/services/run_orchestrator.py").read_text())
        calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name | ast.Attribute)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
            in self._ENGINE_NAMES
        ]
        assert calls == []
