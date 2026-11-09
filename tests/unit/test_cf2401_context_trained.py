"""CF-24-01 P4b -- ``context_trained()`` and the two engine filter reasons (T-18, T-19 (P4b part);
SY-01-16, SY-01-21g, ER2-06 N-4, SY-01-21l-N1 ``bar_ts_ms``, CF-2401-73/74/75).
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
from structlog import get_context
from structlog.testing import capture_logs

import api.routers.runs as runs_module
import trading.clock as clock_mod
from common.models import MultiTimeframeContext
from common.types import RunMode
from data.live_context import LIVE_BOOK, fgi_observations
from data.ml_features import feature_names_for_schema
from data.point_in_time import to_ms
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit._wp16_support import (
    HALT_ONLY,
    HELD,
    OTHER,
    ScriptedSignals,
    bar,
    build_engine,
    buy,
    sell,
    skipped_reasons,
)  # fmt: skip
from trading.safety import build_run_circuit_breaker
from trading.strategies.ensemble import EnsembleStrategy
from trading.strategies.model_strategy import _CONTEXT_FEATURES, ModelStrategy
from trading.strategy_engine import StrategyEngine

UNVERIFIED, MISSING = "mtf_context_model_unverified", "mtf_context_model_input_missing"
_ERR_KEYS = {"event", "log_level", "strategy_id", "error_type"}  # N-4 / SY-01-20: error_type ONLY
_ENGINE = Path(inspect.getfile(StrategyEngine))


class _Stub:
    """A picklable stand-in artefact: importances over ``n`` columns, context share ``ctx``."""

    def __init__(self, n: int, ctx: float, *, bad: Any = None) -> None:
        self.n_features_in_ = n
        names = feature_names_for_schema(2 if n == 14 else 1)
        self.feature_importances_ = np.array(
            [ctx if x in _CONTEXT_FEATURES else 0.1 for x in names]
        )
        if bad is not None:
            self.feature_importances_ = bad


def _model(tmp: Path, name: str, **kw: Any) -> str:
    path = tmp / f"{name}.joblib"
    joblib.dump(_Stub(**kw), path)
    return str(path)


def _started(tmp: Path, path: str) -> ModelStrategy:
    s = ModelStrategy("m", {"model_path": path, "model_dir": str(tmp)})
    s.on_start("r")
    return s


def _swap(tmp: Path, s: ModelStrategy, version: str, path: str) -> None:
    (tmp / "btc_usdt_active.json").write_text(
        json.dumps({"version_id": version, "model_path": path})
    )
    s._check_model_version("BTC/USDT")


# --- T-18: the interface, per strategy ---


class TestT18Interface:
    def test_base_and_the_eight_registry_strategies_are_false(self) -> None:
        registry = runs_module._get_strategy_registry()
        assert set(registry) == {
            "momentum_breakout", "rsi_mean_reversion", "dca_rsi_hybrid", "ma_crossover", "breakout",
            "grid_trading", "sl_tp_reversion", "smoke_roundtrip", "model_strategy",
        }  # fmt: skip
        for name, cls in registry.items():
            if name != "model_strategy":
                assert cls(strategy_id=name, params={}).context_trained() is False, name

    def test_the_context_columns_are_the_four_v2_extras(self) -> None:
        assert _CONTEXT_FEATURES == {
            "htf_trend",
            "fgi_level_norm",
            "fgi_delta_7d",
            "btc_dom_delta_7d",
        }

    @pytest.mark.parametrize(
        ("n", "ctx", "expected"),
        [(14, 0.05, True), (14, 1e-9, True), (14, 0.0, False), (10, 0.0, False)],
    )
    def test_importance_sum_over_the_context_columns(
        self, tmp_path: Path, n: int, ctx: float, expected: bool
    ) -> None:
        assert _started(tmp_path, _model(tmp_path, "a", n=n, ctx=ctx)).context_trained() is expected

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros(3),
            np.full(14, np.nan),
            np.full(14, -0.1),
            np.array(["x"] * 14),
            None,
            np.array([np.inf, *[0.0] * 13]),
        ],
        ids=["length", "nan", "negative", "text", "absent", "inf-outside-context"],
    )
    def test_missing_or_unresolvable_importances_are_context_trained(
        self, tmp_path: Path, bad: Any
    ) -> None:
        stub = _Stub(14, 0.0, bad=bad)
        if bad is None:
            del stub.feature_importances_
        joblib.dump(stub, tmp_path / "b.joblib")
        with capture_logs() as logs:
            assert _started(tmp_path, str(tmp_path / "b.joblib")).context_trained() is True
        assert all(set(e) <= {"event", "log_level", "error_type", "strategy_id"} for e in logs
                   if e["event"] == "model_strategy.importances_failed")  # fmt: skip

    def test_no_artefact_is_fail_closed_and_on_stop_resets(self, tmp_path: Path) -> None:
        assert ModelStrategy("m", {}).context_trained() is True
        s = _started(tmp_path, _model(tmp_path, "a", n=14, ctx=0.0))
        assert s.context_trained() is False
        s.on_stop()
        assert s.context_trained() is True

    def test_the_value_is_recomputed_on_a_hot_swap_both_ways(self, tmp_path: Path) -> None:
        inert, trained = _model(tmp_path, "i", n=14, ctx=0.0), _model(tmp_path, "t", n=14, ctx=0.2)
        s = _started(tmp_path, inert)
        assert s.context_trained() is False
        _swap(tmp_path, s, "v1", trained)
        assert s.context_trained() is True
        _swap(tmp_path, s, "v2", inert)
        assert s.context_trained() is False

    def test_a_failed_hot_swap_keeps_the_loaded_artefacts_value(self, tmp_path: Path) -> None:
        s = _started(tmp_path, _model(tmp_path, "i", n=14, ctx=0.0))
        _swap(tmp_path, s, "v1", str(tmp_path / "missing.joblib"))
        assert s.context_trained() is False

    def test_a_half_finished_hot_swap_is_fail_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = _started(tmp_path, _model(tmp_path, "i", n=14, ctx=0.0))

        def boom(*_: Any, **__: Any) -> int:
            raise RuntimeError("x")

        monkeypatch.setattr(s, "_detect_schema_version", boom)
        _swap(tmp_path, s, "v1", _model(tmp_path, "j", n=14, ctx=0.0))
        assert s.context_trained() is True  # the new artefact was never classified

    def test_the_ensemble_is_the_union_over_its_subs(self, tmp_path: Path) -> None:
        rule = runs_module._get_strategy_registry()["rsi_mean_reversion"]("r", params={})
        model = _started(tmp_path, _model(tmp_path, "i", n=14, ctx=0.0))
        ens = EnsembleStrategy("e", [rule, model])
        assert ens.context_trained() is False
        _swap(tmp_path, model, "v1", _model(tmp_path, "t", n=14, ctx=0.2))
        assert ens.context_trained() is True
        assert EnsembleStrategy("e2", [rule]).context_trained() is False


# --- T-18 / T-19: the engine filter tail ---


class _Probe(ScriptedSignals):
    consumes_mtf_signals = frozenset({"fear_greed_index"})

    def __init__(self, trained: Any) -> None:
        super().__init__("wp16", {"signals": lambda _i: [buy(OTHER), sell(HELD)]})
        self.trained, self.reads = trained, 0

    def context_trained(self) -> Any:
        self.reads += 1
        if isinstance(self.trained, BaseException):
            raise self.trained
        return self.trained


async def _run(
    trained: Any, mode: RunMode = RunMode.PAPER, *, served: bool = False, **kw: Any
) -> tuple[StrategyEngine, Any, _Probe]:
    probe = _Probe(trained)
    engine, ex, _ = build_engine(cb=kw.pop("cb", None), signals=lambda _i: [], mode=mode, **kw)
    engine._strategies = [probe]
    if served:  # a later stage where the consumed field is served
        ctx = MultiTimeframeContext(fear_greed_index=50)
        engine._build_mtf_context = lambda *_a, **_k: ctx  # type: ignore[method-assign]
    await bar(engine, "1000")
    return engine, ex, probe


def _sides(ex: Any) -> list[str]:
    return sorted(s.direction.value for s in ex.signals)


class TestT18Engine:
    @pytest.mark.parametrize(
        ("mode", "served", "reason"),
        [(RunMode.LIVE, False, UNVERIFIED), (RunMode.LIVE, True, UNVERIFIED),
         (RunMode.PAPER, False, MISSING)],
    )  # fmt: skip
    async def test_a_trained_models_buy_is_dropped_and_the_exit_runs(
        self, mode: RunMode, served: bool, reason: str
    ) -> None:
        with capture_logs() as logs:
            engine, ex, _ = await _run(True, mode, served=served)
        assert _sides(ex) == ["sell"] and skipped_reasons(engine) == [reason]
        assert any(e["event"] == "engine.context_model_entries_blocked" for e in logs)
        assert not engine._stop_event.is_set()

    async def test_paper_blocks_only_while_a_consumed_field_is_none(self) -> None:
        engine, ex, _ = await _run(True, RunMode.PAPER, served=True)
        assert _sides(ex) == ["buy", "sell"] and skipped_reasons(engine) == []

    @pytest.mark.parametrize(
        "served",
        [{"fear_greed_index": 50}, {"btc_dominance_7d_ago": 52.5}, {},
         {"fear_greed_index": 50, "btc_dominance_7d_ago": 52.5}],
    )  # fmt: skip
    async def test_paper_blocks_while_any_of_two_consumed_fields_is_none(
        self, served: dict[str, Any]
    ) -> None:
        probe = _Probe(True)
        probe.consumes_mtf_signals = frozenset({"fear_greed_index", "btc_dominance_7d_ago"})
        engine, ex, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.PAPER)
        engine._strategies = [probe]
        ctx = MultiTimeframeContext(**served)
        engine._build_mtf_context = lambda *_a, **_k: ctx  # type: ignore[method-assign]
        await bar(engine, "1000")
        blocked = len(served) < 2
        assert skipped_reasons(engine) == ([MISSING] if blocked else [])
        assert _sides(ex) == (["sell"] if blocked else ["buy", "sell"])

    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE, RunMode.BACKTEST])
    async def test_an_inert_model_is_never_blocked(self, mode: RunMode) -> None:
        engine, ex, _ = await _run(False, mode)
        assert _sides(ex) == ["buy", "sell"] and skipped_reasons(engine) == []

    async def test_a_backtest_is_never_filtered(self) -> None:
        _, ex, probe = await _run(True, RunMode.BACKTEST)
        assert _sides(ex) == ["buy", "sell"] and probe.reads == 0

    @pytest.mark.parametrize("value", [MagicMock(), 1, "yes", [1], None], ids=repr)
    @pytest.mark.parametrize("mode", [RunMode.PAPER, RunMode.LIVE])
    async def test_only_the_identity_true_blocks(self, mode: RunMode, value: Any) -> None:
        _, ex, probe = await _run(value, mode)
        assert _sides(ex) == ["buy", "sell"] and probe.reads == 1

    @pytest.mark.parametrize(
        ("mode", "served", "reason"),
        [(RunMode.LIVE, False, UNVERIFIED), (RunMode.PAPER, False, MISSING),
         (RunMode.PAPER, True, None)],
    )  # fmt: skip
    async def test_n4_a_raising_context_trained_is_treated_as_true(
        self, mode: RunMode, served: bool, reason: str | None
    ) -> None:
        with capture_logs() as logs:
            engine, ex, _ = await _run(RuntimeError("SECRET-TEXT"), mode, served=served)
        assert skipped_reasons(engine) == ([reason] if reason else [])
        assert _sides(ex) == (["sell"] if reason else ["buy", "sell"])  # exits unaffected
        (err,) = [e for e in logs if e["event"] == "engine.context_trained_failed"]
        assert err["error_type"] == "RuntimeError" and "SECRET" not in str(logs)
        assert err["strategy_id"] == "wp16"
        assert set(err) - set(get_context(engine._log)) == _ERR_KEYS

    async def test_the_value_is_read_at_filter_time_every_bar(self) -> None:
        engine, ex, probe = await _run(False, RunMode.LIVE)
        assert _sides(ex) == ["buy", "sell"]
        probe.trained = True  # e.g. a hot-swap between bars
        await bar(engine, "1000")
        assert _sides(ex)[-2:] == ["sell", "sell"] and skipped_reasons(engine) == [UNVERIFIED]
        probe.trained = False
        await bar(engine, "1000")
        assert skipped_reasons(engine) == [UNVERIFIED] and probe.reads == 3

    @pytest.mark.parametrize(
        ("trained", "mode", "reason", "logged"),
        [(True, RunMode.PAPER, MISSING, True), (True, RunMode.LIVE, UNVERIFIED, False),
         (False, RunMode.PAPER, None, False)],
    )  # fmt: skip
    @pytest.mark.parametrize("fields", [None, [1], RuntimeError("SECRET-TEXT")], ids=repr)
    async def test_a_malformed_field_set_blocks_the_buy_and_never_skips_the_exits(
        self, trained: bool, mode: RunMode, reason: str | None, logged: bool, fields: Any
    ) -> None:
        def malformed() -> Any:
            if isinstance(fields, BaseException):
                raise fields
            return fields

        probe = _Probe(trained)
        probe.mtf_signal_fields = malformed  # type: ignore[method-assign]
        engine, ex, _ = build_engine(cb=None, signals=lambda _i: [], mode=mode)
        engine._strategies = [probe]
        with capture_logs() as logs:
            await bar(engine, "1000")  # nothing escapes _process_bar
        assert skipped_reasons(engine) == ([reason] if reason else [])
        assert _sides(ex) == (["sell"] if reason else ["buy", "sell"])
        errs = [e for e in logs if e["event"] == "engine.context_fields_failed"]
        bound = set(get_context(engine._log))
        assert [set(e) - bound for e in errs] == ([_ERR_KEYS] if logged else [])
        assert "SECRET" not in str(logs)


class TestT19Order:
    async def test_every_earlier_reason_wins_and_one_record_per_buy(self) -> None:
        halt = build_run_circuit_breaker("p4b", HALT_ONLY)
        failing = MagicMock()
        failing.evaluate.side_effect = RuntimeError("x")
        cases: list[tuple[dict[str, Any], str]] = [
            ({}, "kill_switch"), ({"protective_mode": True}, "protective_mode"),
            ({"cb": failing}, "circuit_breaker_eval_failed"),
            ({"cb": halt}, "circuit_breaker_halt"),
        ]  # fmt: skip
        for kw, expected in cases:
            probe = _Probe(True)
            engine, ex, _ = build_engine(
                cb=kw.pop("cb", None), signals=lambda _i: [], mode=RunMode.LIVE, **kw
            )
            engine._strategies = [probe]
            if expected == "kill_switch":
                engine._risk_manager.trigger_kill_switch("t")
            await bar(engine, "800" if expected == "circuit_breaker_halt" else "1000")
            assert skipped_reasons(engine) == [expected], expected
            assert _sides(ex) == ["sell"] and probe.reads == 0  # never even asked

    async def test_two_buys_two_records_and_sells_pass(self) -> None:
        probe = _Probe(True)
        probe._fn = lambda _i: [buy(OTHER), buy(HELD), sell(HELD)]
        engine, ex, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.PAPER)
        engine._strategies = [probe]
        await bar(engine, "1000")
        assert skipped_reasons(engine) == [MISSING, MISSING] and _sides(ex) == ["sell"]
        assert probe.reads == 1  # one read per strategy per bar
        assert engine._total_signals == 1  # counted after the filter, like the earlier reasons

    async def test_a_strategy_without_a_buy_is_never_asked(self) -> None:
        seller, buyer = _Probe(RuntimeError("x")), _Probe(False)
        seller._strategy_id, buyer._strategy_id = "s", "b"
        seller._fn = lambda _i: [sell(HELD).model_copy(update={"strategy_id": "s"})]
        buyer._fn = lambda _i: [buy(OTHER).model_copy(update={"strategy_id": "b"})]
        engine, ex, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.LIVE)
        engine._strategies = [seller, buyer]
        with capture_logs() as logs:
            await bar(engine, "1000")
        assert seller.reads == 0 and buyer.reads == 1 and _sides(ex) == ["buy", "sell"]
        assert not [e for e in logs if e["event"] == "engine.context_trained_failed"]

    async def test_another_strategys_buy_is_untouched(self) -> None:
        trained, inert = _Probe(True), _Probe(False)
        trained._strategy_id, inert._strategy_id = "a", "b"
        trained._fn = lambda _i: [buy(OTHER).model_copy(update={"strategy_id": "a"})]
        inert._fn = lambda _i: [buy(OTHER).model_copy(update={"strategy_id": "b"})]
        engine, ex, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.LIVE)
        engine._strategies = [trained, inert]
        await bar(engine, "1000")
        assert [s.strategy_id for s in ex.signals] == ["b"] and skipped_reasons(engine) == [
            UNVERIFIED
        ]


# --- CF-2401-73: the PC-8 pin (SY-01-01 does not remove the need: the filter is new code) ---

_LOG_ARGS = "strategy_id=strategy.strategy_id,\n                    error_type=type(exc).__name__,"
_ASSIGN_CALL = "        bar_signals = self._drop_context_model_buys(bar_signals, current_bars)\n"


def context_gate_problems(src: str) -> list[str]:
    """The two reasons live ONLY in ``_drop_context_model_buys``, which is the LAST entry filter,
    BUY-only, fail-closed (N-4), never filters a backtest and never touches the stop state."""
    tree = ast.parse(src)
    fns = {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    gate, proc = fns.get("_drop_context_model_buys"), fns.get("_process_bar")
    if gate is None or proc is None:
        return ["gate or _process_bar missing"]
    out: list[str] = []
    span = range(gate.lineno, (gate.end_lineno or 0) + 1)
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Constant)
            and n.value in (UNVERIFIED, MISSING)
            and n.lineno not in span
        ):
            out.append(f"reason outside the gate: line {n.lineno}")
    calls = [
        (i, s) for i, s in enumerate(proc.body)
        if "_drop_context_model_buys" in ast.unparse(s) or "_drop_entry_signals" in ast.unparse(s)
    ]  # fmt: skip
    kinds = [("ctx" if "_drop_context_model_buys" in ast.unparse(s) else "entry") for _, s in calls]
    if kinds != ["entry", "ctx"] or not isinstance(calls[-1][1], ast.Assign):
        out.append(f"not an unconditional statement after the entry filter: {kinds}")
    g = ast.unparse(gate)
    unparsed = {
        "buy-only": "s.direction == SignalDirection.BUY and s.strategy_id in blocked" in g,
        "identity": "strategy.context_trained() is True" in g,
        "backtest": "if self._run_mode == RunMode.BACKTEST:\n        return signals" in g,
        "live-first": f"'{UNVERIFIED}' if self._run_mode == RunMode.LIVE else '{MISSING}'" in g,
        "fail-closed": "except Exception as exc:\n            trained = True" in g,
        "fields-fail-closed": "except Exception as exc:\n            block = True" in g,
        "error_type": "error_type=type(exc).__name__" in g and "exc_info" not in g,
        "no-stop": not re.search(r"_stop_event|_auto_stop_reason|evaluate", g),
    }  # fmt: skip
    out += [f"gate: {k}" for k, ok in unparsed.items() if not ok]
    return out


class TestPc8Pin:
    def test_the_real_engine_is_clean(self) -> None:
        assert context_gate_problems(_ENGINE.read_text()) == []

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (f'"{UNVERIFIED}"', '"x"'),  # swapped reason (live <-> paper)
            ("strategy.context_trained() is True", "strategy.context_trained()"),
            ("except Exception as exc:\n                trained = True",
             "except Exception as exc:\n                trained = False"),
            ("                block = True\n", "                block = False\n"),
            ("        if self._run_mode == RunMode.BACKTEST:\n            return signals\n", ""),
            (_LOG_ARGS, _LOG_ARGS + " exc_info=True,"),
            ("s.direction == SignalDirection.BUY and s.strategy_id", "s.strategy_id"),
            (_ASSIGN_CALL, "        if _entry_skip_reason is None:\n    " + _ASSIGN_CALL),
            (_ASSIGN_CALL, ""),
        ],
    )  # fmt: skip
    def test_each_mutation_is_caught(self, old: str, new: str) -> None:
        src = _ENGINE.read_text()
        assert src.count(old) >= 1
        assert context_gate_problems(src.replace(old, new, 1)) != []

    def test_the_call_moved_before_the_entry_filter_is_caught(self) -> None:
        src = _ENGINE.read_text()
        anchor = "        _entry_skip_reason: str | None = None\n"
        assert (
            context_gate_problems(
                src.replace(_ASSIGN_CALL, "").replace(anchor, _ASSIGN_CALL + anchor)
            )
            != []
        )

    def test_the_swapped_reasons_are_caught_exactly(self) -> None:
        src = (
            _ENGINE.read_text()
            .replace(UNVERIFIED, "@")
            .replace(MISSING, UNVERIFIED)
            .replace("@", MISSING)
        )
        assert context_gate_problems(src) == ["gate: live-first"]


# --- CF-2401-74: ``bar_ts_ms`` (SY-01-21l-N1) ---


def _shadow(ts: datetime) -> dict[str, Any]:
    engine, _, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.PAPER)
    with capture_logs() as logs:
        engine._build_mtf_context(ts)
    (event,) = [e for e in logs if e["event"] == "engine.mtf_context_shadow"]
    return event


class TestBarKey:
    @pytest.mark.parametrize("micro", [0, 1_000, 7_000, 123_000, 456_789, 999_999])
    def test_it_is_the_exact_integer_epoch_ms_of_the_bar(self, micro: int) -> None:
        ts = datetime(2026, 3, 10, 12, 0, 1, micro, tzinfo=UTC)
        value = _shadow(ts)["bar_ts_ms"]
        assert type(value) is int
        assert value == (to_ms(ts.replace(microsecond=0)) + micro // 1000)

    def test_a_naive_timestamp_is_utc_and_it_is_never_an_age_or_the_books(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ts = datetime(2020, 1, 2, 3, 4, 5, 6_000)
        wall = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        monkeypatch.setattr(clock_mod, "_wall_now", lambda: wall)
        base = _shadow(ts)
        assert base["bar_ts_ms"] == to_ms(ts.replace(tzinfo=UTC, microsecond=0)) + 6
        now = to_ms(wall)
        day = now - now % 86_400_000
        LIVE_BOOK.publish(
            "alternative_me:fgi", fgi_observations([(day - 86_400_000, 30)], now), now
        )
        assert _shadow(ts)["bar_ts_ms"] == base["bar_ts_ms"]  # the book changed, the key did not
        assert _shadow(ts + timedelta(hours=1))["bar_ts_ms"] - base["bar_ts_ms"] == 3_600_000

    def test_the_expression_is_a_pure_integer_floor_division(self) -> None:
        fn = next(
            n for n in ast.walk(ast.parse(_ENGINE.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_build_mtf_context"
        )  # fmt: skip
        kws = [k for c in ast.walk(fn) if isinstance(c, ast.Call) for k in c.keywords]
        (key,) = [k for k in kws if k.arg == "bar_ts_ms"]
        (calc,) = [
            a
            for a in ast.walk(fn)
            if isinstance(a, ast.Assign) and ast.unparse(a.targets[0]) == "bar_ms"
        ]
        assert ast.unparse(key.value) == "bar_ms" and isinstance(calc.value, ast.BinOp)
        assert isinstance(calc.value.op, ast.FloorDiv)
        assert not re.search(r"timestamp|float|round|(?<!/)/(?!/)|\*", ast.unparse(calc.value))


# --- CF-2401-75: ``_live_fields`` comes from ``self._strategies`` ---


def test_an_iterator_of_strategies_is_not_emptied_before_the_fields_are_read() -> None:
    engine, _, _ = build_engine(cb=None, signals=lambda _i: [], mode=RunMode.PAPER)
    again = StrategyEngine(
        strategies=iter([_Probe(False)]),  # type: ignore[arg-type]
        execution_engine=engine._execution_engine, risk_manager=engine._risk_manager,
        market_data=engine._market_data, portfolio=engine._portfolio, symbols=[HELD],
        timeframe=engine._timeframe, run_mode=RunMode.PAPER,
    )  # fmt: skip
    assert len(again._strategies) == 1 and again._live_fields == frozenset({"fear_greed_index"})
    assert "verbatim provider reads" not in _ENGINE.read_text()
