"""
tests/unit/test_wp21h_ml_profile.py
-------------------------------------
WP2.1-H, CF-21a-02 (+ CR-07): the ML walk-forward gate records the cost profile it TRAINED with.

``_train_model_with_wf_sync`` reads the platform default once, runs every fold backtest
under that object and hands it back next to the result dict as ``_TrainedModel(result,
cost_profile)``; ``train_model`` records that object in ``ModelVersionORM.extra["cost_profile"]``
instead of making a second ``default_cost_profile()`` call (which could differ if the default
moved in between).  The profile is a required field of the return type, so a result without it
cannot be built, and it never enters the HTTP response.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.test_sprint50_cycle6_oos_backtest import TestTrainWfAggregation
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE, CostProfile, get_cost_profile

_D6 = get_cost_profile("coinbase_advanced_d6")
_T80 = get_cost_profile("coinbase_stress_t80")
_FOLDS: dict[str, Any] = {
    "fold_sharpes": [1.2, 0.8, 1.6],
    "fold_trades": [10, 12, 8],
    "fold_dds": [0.1, 0.1, 0.1],
}


def _ml() -> Any:
    import importlib

    return importlib.import_module("api.routers.ml")


async def _train(ml: Any, trained: Any) -> tuple[Any, dict[str, Any]]:
    """Run ``train_model`` against a stubbed sync trainer; return (the ORM row, the response)."""
    db = AsyncMock()
    db.add = MagicMock()
    with patch.object(ml, "_train_model_with_wf_sync", return_value=trained):
        response = await ml.train_model(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe="1h",
            bars=400,
            n_estimators=10,
            horizon=1,
            threshold=0.01,
            num_wf_folds=3,
            db=db,
        )
    version = next(
        c.args[0] for c in db.add.call_args_list if type(c.args[0]).__name__ == "ModelVersionORM"
    )
    return version, response


def _result() -> dict[str, Any]:
    return {
        "status": "completed",
        "model_path": "models/x.joblib",
        "bars_fetched": 400,
        "metrics": {"accuracy": 0.5},
        "walk_forward_status": "ok",
        "walk_forward_oos_skill_score": 1.0,
    }


def _spy_runner_profiles() -> tuple[list[Any], Any]:
    seen: list[Any] = []
    original_init = BacktestRunner.__init__

    def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
        seen.append(kw.get("cost_profile"))
        original_init(self, *args, **kw)

    return seen, patch.object(BacktestRunner, "__init__", _spy)


class TestCf21a02TheGateRecordsTheProfileItTrainedWith:
    def test_the_sync_function_returns_the_profile_every_fold_backtest_ran_under(self) -> None:
        seen, spy = _spy_runner_profiles()
        with spy:
            out = TestTrainWfAggregation()._run_trained(**_FOLDS)
        assert isinstance(out, _ml()._TrainedModel)
        assert isinstance(out.cost_profile, CostProfile)
        assert len(seen) == 3 and all(p is out.cost_profile for p in seen)
        assert out.cost_profile is DEFAULT_COST_PROFILE

    def test_the_result_dict_does_not_carry_the_profile(self) -> None:
        out = TestTrainWfAggregation()._run_trained(**_FOLDS)
        assert "cost_profile" not in out.result
        assert not any(isinstance(v, CostProfile) for v in out.result.values())

    async def test_train_model_records_the_returned_profile_not_a_fresh_default(self) -> None:
        ml = _ml()
        # the returned profile deliberately differs from the platform default
        version, _ = await _train(ml, ml._TrainedModel(result=_result(), cost_profile=_D6))
        assert version.extra["cost_profile"] == _D6.to_metadata()
        assert version.extra["cost_profile"] != DEFAULT_COST_PROFILE.to_metadata()

    async def test_the_http_response_is_the_result_dict_unchanged(self) -> None:
        ml = _ml()
        result = _result()
        expected = dict(result)
        _, response = await _train(ml, ml._TrainedModel(result=result, cost_profile=_D6))
        assert response == expected
        assert not any(isinstance(v, CostProfile) for v in response.values())

    async def test_trained_and_recorded_profiles_are_equal_even_if_the_default_moves(
        self,
    ) -> None:
        """The pin: the profile the fold backtests ran under == the profile recorded.

        ``default_cost_profile`` is made to MOVE between its first and second read
        (D6 first, then T80).  With a second read in ``train_model`` (the 2.1a code) the
        row would say T80 while the folds ran under D6.
        """
        ml = _ml()
        reads = iter([_D6, _T80])
        calls = 0

        def _moving_default() -> CostProfile:
            nonlocal calls
            calls += 1
            return next(reads)

        seen, spy = _spy_runner_profiles()
        with patch.object(ml, "default_cost_profile", _moving_default), spy:
            # in a worker thread, like production: the fold runner calls ``asyncio.run``,
            # which cannot run inside this test's event loop
            out = await asyncio.to_thread(TestTrainWfAggregation()._run_trained, **_FOLDS)
            version, _ = await _train(ml, out)

        assert calls == 1, "the default is read exactly once per training run"
        assert out.result["walk_forward_fold_sharpes"] == _FOLDS["fold_sharpes"]  # folds RAN
        assert len(seen) == 3 and all(p == _D6 for p in seen)
        assert version.extra["cost_profile"] == seen[0].to_metadata() == _D6.to_metadata()


class TestCr07AMissingProfileIsImpossible:
    """The profile is not an in-band dict key that a caller could fail to find."""

    def test_a_result_without_the_profile_cannot_be_constructed(self) -> None:
        with pytest.raises(TypeError):
            _ml()._TrainedModel(result=_result())  # type: ignore[call-arg]

    def test_the_two_fields_are_the_result_and_the_profile(self) -> None:
        assert _ml()._TrainedModel._fields == ("result", "cost_profile")

    def test_every_return_of_the_sync_function_builds_a_trained_model(self) -> None:
        """The only way out of ``_train_model_with_wf_sync`` is a ``_TrainedModel(...)``:
        no bare dict, no other path that would leave the caller without the profile."""
        tree = ast.parse(inspect.getsource(_ml()._train_model_with_wf_sync))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        returns: list[ast.Return] = []

        def _walk(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                    continue  # a nested function's return is not ours
                if isinstance(child, ast.Return):
                    returns.append(child)
                _walk(child)

        _walk(func)
        assert returns, "the sync function has no return?"
        for ret in returns:
            call = ret.value
            assert isinstance(call, ast.Call), f"line {ret.lineno}"
            assert isinstance(call.func, ast.Name) and call.func.id == "_TrainedModel"
            assert {k.arg for k in call.keywords} == {"result", "cost_profile"}
