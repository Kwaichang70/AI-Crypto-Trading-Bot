"""
tests/unit/test_sprint50_cycle6_oos_backtest.py
------------------------------------------------
Sprint 50 Cycle 6 -- Real per-fold OOS backtest Sharpe gate.

Modules under test
------------------
packages/trading/metrics.py                 -- BacktestResult.oos_sharpe / oos_n_returns
packages/trading/backtest.py                -- BacktestRunner.run(..., oos_start_index=)
packages/trading/strategy_engine.py         -- run_backtest(..., oos_start_index=) boundary capture
apps/api/services/model_activation_gate.py  -- check_oos_eligibility (Cycle 6 routing + DD floor)
apps/api/config.py                          -- min_oos_skill_score / max_fold_drawdown / min_trades_per_fold
apps/api/routers/ml.py                      -- _train_model_with_wf_sync realized-OOS aggregation + JSONB

Unique test IDs: TEST-C6-001 .. TEST-C6-0xx (see docstrings).

Test matrix mapping (producer report sprint50-cycle6-oos-backtest-producer.md §9)
--------------------------------------------------------------------------------
TC#1  additivity / default None                -> TestOOSFieldsAdditive
TC#2  run() with oos_start_index populates      -> TestOOSWindowSlicing (golden value, no-fill)
TC#3  OOS excludes warmup/in-sample             -> TestOOSWindowSlicing.test_oos_excludes_insample_region
TC#4  out-of-range oos_start_index -> None+warn -> TestOOSOutOfRange
TC#5  equity-point ordinal correctness w/ fills -> TestOOSUnderInSampleFills
TC#6  _train_model_with_wf_sync aggregation     -> TestTrainWfAggregation (ccxt + heavy parts mocked)
TC#7  JSONB schema v3 + metric_type             -> TestTrainWfJsonbContract
TC#8  gate realized-Sharpe threshold            -> TestGateRealizedSharpeThreshold
TC#9  gate legacy z-score passthrough           -> TestGateLegacyZScorePassthrough
TC#10 gate max-drawdown floor (B1)              -> TestGateMaxDrawdownFloor
config defaults                                 -> TestConfigDefaults
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.backtest import BacktestRunner
from trading.metrics import (
    BacktestResult,
    compute_returns_from_equity,
    compute_sharpe,
)
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

from tests.conftest import make_bars

# Periods-per-year for 1h timeframe (metrics.py TIMEFRAME_PERIODS_PER_YEAR).
_PPY_1H = 365.25 * 24


# ===========================================================================
# Strategy stubs
# ===========================================================================


class _AlwaysHoldStrategy(BaseStrategy):
    """Never trades. Equity curve is then 1:1 with bars (1 seed + 1/bar),
    which lets us recompute the OOS slice deterministically by bar ordinal."""

    metadata = StrategyMetadata(
        name="c6_always_hold",
        version="1.0.0",
        description="never trades",
    )

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        return []


class _BuyAtBarStrategy(BaseStrategy):
    """Emits a single BUY exactly at a chosen post-warmup bar index, then holds.

    Used to force an IN-SAMPLE fill so the equity curve is NOT 1:1 with bars;
    proves the OOS boundary uses a live len() capture rather than naive bar
    arithmetic.
    """

    metadata = StrategyMetadata(
        name="c6_buy_at_bar",
        version="1.0.0",
        description="one in-sample BUY then hold",
    )

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._bar_count = 0
        # Fire on the Nth on_bar invocation (1-based), defaulting to bar 1.
        self._fire_on = int((params or {}).get("fire_on", 1))
        self._fired = False

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        self._bar_count += 1
        if self._bar_count == self._fire_on and not self._fired:
            self._fired = True
            return [
                Signal(
                    strategy_id=self._strategy_id,
                    symbol=bars[-1].symbol,
                    direction=SignalDirection.BUY,
                    target_position=Decimal("0.01"),
                    confidence=1.0,
                )
            ]
        return []


# ===========================================================================
# TestOOSFieldsAdditive  (TC#1)
# ===========================================================================


class TestOOSFieldsAdditive:
    """TC#1 -- additivity: new fields default None; existing behaviour unchanged."""

    def test_backtest_result_oos_fields_default_none(self) -> None:
        """TEST-C6-001: BacktestResult constructed without OOS kwargs -> None."""
        r = _minimal_result()
        assert r.oos_sharpe is None
        assert r.oos_n_returns is None

    @pytest.mark.asyncio
    async def test_run_without_oos_leaves_fields_none(self) -> None:
        """TEST-C6-002: run() with no oos_start_index leaves OOS fields None."""
        bars = make_bars(200, seed=42, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-add")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=42,
        )
        result = await runner.run({"BTC/USDT": bars})
        assert result.oos_sharpe is None
        assert result.oos_n_returns is None

    @pytest.mark.asyncio
    async def test_no_oos_metrics_identical_to_baseline(self) -> None:
        """TEST-C6-003: two no-OOS runs (seeded) produce identical full-sample metrics."""
        bars = make_bars(200, seed=42, symbol="BTC/USDT")

        def _runner() -> BacktestRunner:
            return BacktestRunner(
                strategies=[_AlwaysHoldStrategy(strategy_id="c6-base")],
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                initial_capital=Decimal("10000"),
                seed=42,
            )

        r1 = await _runner().run({"BTC/USDT": bars})
        r2 = await _runner().run({"BTC/USDT": bars})
        assert r1.sharpe_ratio == r2.sharpe_ratio
        assert r1.total_trades == r2.total_trades
        assert r1.final_equity == r2.final_equity
        # OOS fields stay None in the baseline (no regression).
        assert r1.oos_sharpe is None and r2.oos_sharpe is None


# ===========================================================================
# TestOOSWindowSlicing  (TC#2, TC#3)  -- the LOAD-BEARING golden value test
# ===========================================================================


class TestOOSWindowSlicing:
    """TC#2/#3 -- OOS Sharpe reflects ONLY the OOS window and excludes warmup."""

    @pytest.mark.asyncio
    async def test_oos_sharpe_golden_value_no_fill(self) -> None:
        """TEST-C6-010: with a no-fill strategy the equity curve is 1:1 with bars,
        so the OOS boundary is exactly oos_start_index. Recompute the OOS Sharpe
        independently from result.equity_curve[oos_start_index:] and assert match.

        This is the load-bearing proof that warmup bars are excluded: the OOS
        Sharpe is built ONLY from the equity points at/after oos_start_index.
        """
        n = 300
        warmup = 50  # min_bars_required=0 -> max(0*2, 50)=50
        oos_start = 200
        bars = make_bars(n, seed=7, symbol="BTC/USDT")

        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-golden")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            initial_capital=Decimal("10000"),
            seed=7,
        )
        result = await runner.run({"BTC/USDT": bars}, oos_start_index=oos_start)

        assert result.oos_sharpe is not None
        assert result.oos_n_returns is not None
        assert result.oos_n_returns >= 0

        # No-fill curve: 1 seed point + 1 per bar -> point[i+1] == bar i.
        # Boundary captured BEFORE bar oos_start appends -> len == oos_start + 1.
        # slice_start = boundary - 1 = oos_start.
        curve = result.equity_curve
        assert len(curve) == n + 1  # confirms the 1:1 assumption holds
        expected_slice = curve[oos_start:]
        expected_returns = compute_returns_from_equity(expected_slice)
        expected_sharpe = compute_sharpe(expected_returns, _PPY_1H)

        assert result.oos_n_returns == len(expected_returns)
        assert result.oos_sharpe == pytest.approx(expected_sharpe, rel=1e-9, abs=1e-9)

    @pytest.mark.asyncio
    async def test_oos_excludes_warmup_returns(self) -> None:
        """TEST-C6-011: OOS return count == (n - oos_start), i.e. warmup +
        in-sample returns are EXCLUDED from the OOS window (no-fill case)."""
        n = 300
        oos_start = 220
        bars = make_bars(n, seed=11, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-excl")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=11,
        )
        result = await runner.run({"BTC/USDT": bars}, oos_start_index=oos_start)
        # curve[oos_start:] has (n + 1 - oos_start) points -> (n - oos_start) returns.
        assert result.oos_n_returns == n - oos_start

    @pytest.mark.asyncio
    async def test_oos_sharpe_differs_from_full_sample(self) -> None:
        """TEST-C6-012: a trading strategy yields an OOS Sharpe distinct from the
        full-sample Sharpe (OOS window is a strict, different subset)."""
        n = 300
        oos_start = 200
        bars = make_bars(n, seed=3, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_BuyAtBarStrategy(strategy_id="c6-diff", params={"fire_on": 5})],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=3,
        )
        result = await runner.run({"BTC/USDT": bars}, oos_start_index=oos_start)
        assert result.oos_sharpe is not None
        # Full-sample and OOS Sharpe should not be byte-identical for a real curve.
        assert result.oos_sharpe != result.sharpe_ratio


# ===========================================================================
# TestOOSOutOfRange  (TC#4)
# ===========================================================================


class TestOOSOutOfRange:
    """TC#4 -- out-of-range oos_start_index leaves fields None + logs a warning."""

    @pytest.mark.asyncio
    async def test_oos_index_below_warmup_is_none(self) -> None:
        """TEST-C6-020: oos_start_index < warmup_bars (50) -> OOS fields None, warn."""
        bars = make_bars(200, seed=42, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-below")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=42,
        )
        with patch.object(runner, "_log", MagicMock()) as mlog:
            result = await runner.run({"BTC/USDT": bars}, oos_start_index=10)
        assert result.oos_sharpe is None
        assert result.oos_n_returns is None
        _assert_warned(mlog, "backtest.oos_start_index_out_of_range")

    @pytest.mark.asyncio
    async def test_oos_index_at_or_above_num_bars_is_none(self) -> None:
        """TEST-C6-021: oos_start_index >= num_bars -> OOS fields None, warn."""
        n = 200
        bars = make_bars(n, seed=42, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-above")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=42,
        )
        with patch.object(runner, "_log", MagicMock()) as mlog:
            result = await runner.run({"BTC/USDT": bars}, oos_start_index=n)
        assert result.oos_sharpe is None
        assert result.oos_n_returns is None
        _assert_warned(mlog, "backtest.oos_start_index_out_of_range")

    @pytest.mark.asyncio
    async def test_out_of_range_does_not_crash(self) -> None:
        """TEST-C6-022: wildly out-of-range index does not raise."""
        bars = make_bars(200, seed=42, symbol="BTC/USDT")
        runner = BacktestRunner(
            strategies=[_AlwaysHoldStrategy(strategy_id="c6-crash")],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=42,
        )
        result = await runner.run({"BTC/USDT": bars}, oos_start_index=99999)
        assert result.oos_sharpe is None  # no exception


# ===========================================================================
# TestOOSUnderInSampleFills  (TC#5)
# ===========================================================================


class TestOOSUnderInSampleFills:
    """TC#5 -- in-sample fills add extra equity points; the OOS boundary must
    use a LIVE len() capture, not naive bar arithmetic."""

    @pytest.mark.asyncio
    async def test_oos_boundary_robust_to_insample_fill(self) -> None:
        """TEST-C6-030: a strategy that fills once in-sample (fire_on=5, bar ~54)
        adds an extra equity point. The naive bar-ordinal slice
        curve[oos_start:] would then be OFF BY ONE relative to the correct
        live-len() boundary. Prove the implementation does NOT use the naive
        slice by showing its OOS Sharpe matches the boundary-shifted slice and
        NOT the naive one.
        """
        n = 300
        oos_start = 200
        bars = make_bars(n, seed=5, symbol="BTC/USDT")
        # One in-sample BUY on the 5th on_bar call (bar index ~ warmup+4 = 54),
        # which is well inside the in-sample region (< oos_start=200).
        runner = BacktestRunner(
            strategies=[_BuyAtBarStrategy(strategy_id="c6-fill", params={"fire_on": 5})],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=5,
        )
        result = await runner.run({"BTC/USDT": bars}, oos_start_index=oos_start)
        assert result.oos_sharpe is not None

        curve = result.equity_curve
        # Exactly ONE in-sample fill => curve length = (n + 1 seed) + 1 fill point.
        assert len(curve) == n + 2, (
            "expected exactly one extra equity point from the single in-sample fill"
        )

        # The fill happened in-sample, so the correct OOS boundary is shifted by
        # the +1 in-sample fill point relative to the naive bar-ordinal slice.
        # Correct boundary (live len at bar oos_start) = oos_start + 1 (seed) + 1 (fill)
        #   -> slice_start = boundary - 1 = oos_start + 1.
        correct_slice = curve[oos_start + 1:]
        correct_returns = compute_returns_from_equity(correct_slice)
        correct_sharpe = compute_sharpe(correct_returns, _PPY_1H)

        # The naive (WRONG) slice that ignores the in-sample fill point.
        naive_slice = curve[oos_start:]
        naive_returns = compute_returns_from_equity(naive_slice)
        naive_sharpe = compute_sharpe(naive_returns, _PPY_1H)

        assert result.oos_n_returns == len(correct_returns)
        assert result.oos_sharpe == pytest.approx(correct_sharpe, rel=1e-9, abs=1e-12)
        # And confirm the naive slice would genuinely differ (otherwise the test
        # is not discriminating). The fill is in-sample so the slices differ by
        # one return observation.
        assert len(naive_returns) == len(correct_returns) + 1


# ===========================================================================
# TestConfigDefaults  (config TC)
# ===========================================================================


class TestConfigDefaults:
    """Config TC -- Cycle 6 defaults on a fresh (cache-cleared) Settings."""

    def test_cycle6_config_defaults(self) -> None:
        """TEST-C6-040: min_oos_skill_score==1.0, max_fold_drawdown==0.25,
        min_trades_per_fold==5 on a fresh Settings()."""
        from api.config import get_settings

        get_settings.cache_clear()
        try:
            s = get_settings()
            assert s.min_oos_skill_score == pytest.approx(1.0)
            assert s.max_fold_drawdown == pytest.approx(0.25)
            assert s.min_trades_per_fold == 5
        finally:
            get_settings.cache_clear()

    def test_max_fold_drawdown_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TEST-C6-041: MAX_FOLD_DRAWDOWN env var overrides the default."""
        from api.config import Settings, get_settings

        monkeypatch.setenv("MAX_FOLD_DRAWDOWN", "0.40")
        get_settings.cache_clear()
        try:
            s = Settings()
            assert s.max_fold_drawdown == pytest.approx(0.40)
        finally:
            get_settings.cache_clear()


# ===========================================================================
# Gate tests (TC#8, TC#9, TC#10)
# ===========================================================================


def _make_model_version(
    oos_skill_score: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """ModelVersionORM-like namespace (mirrors cycle-5 gate test helper)."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        symbol="BTC/USD",
        timeframe="1h",
        walk_forward_oos_skill_score=(
            Decimal(str(oos_skill_score)) if oos_skill_score is not None else None
        ),
        extra=extra,
    )


class TestGateRealizedSharpeThreshold:
    """TC#8 -- realized-Sharpe median gate vs min_oos_skill_score=1.0."""

    def test_median_below_threshold_rejected(self) -> None:
        """TEST-C6-050: realized row, median 0.8 < 1.0 -> oos_skill_below_min."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.8,
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "schema_version": 3,
                "status": "ok",
                "oos_skill_worst": 0.5,
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=1.0)
        assert result.eligible is False
        assert result.reason == "oos_skill_below_min"
        assert result.oos_skill_score == pytest.approx(0.8)

    def test_median_at_or_above_threshold_eligible(self) -> None:
        """TEST-C6-051: realized row, median 1.5 >= 1.0 -> eligible."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=1.5,
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "schema_version": 3,
                "status": "ok",
                "oos_skill_worst": 0.7,
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=1.0)
        assert result.eligible is True
        assert result.reason == ""

    def test_worst_fold_floor_rejects_negative(self) -> None:
        """TEST-C6-052: TC#8 worst-fold floor -- worst Sharpe < 0.0 -> reject."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=1.5,  # median passes
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "status": "ok",
                "oos_skill_worst": -0.3,  # worst fold catastrophic
            }},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=1.0, min_worst_fold_skill_score=0.0
        )
        assert result.eligible is False
        assert result.reason == "worst_fold_below_floor"
        assert result.worst_fold_skill_score == pytest.approx(-0.3)


class TestGateLegacyZScorePassthrough:
    """TC#9 -- legacy directional_zscore_proxy rows pass with a warning."""

    def test_legacy_zscore_below_one_passes_with_warning(self) -> None:
        """TEST-C6-060: metric_type=directional_zscore_proxy, score 0.2 < 1.0 ->
        eligible=True, reason='', warning mentions retrain."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=0.2,
            extra={"walk_forward": {
                "metric_type": "directional_zscore_proxy",
                "schema_version": 2,
                "status": "ok",
            }},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=1.0)
        assert result.eligible is True
        assert result.reason == ""
        assert result.warning != ""
        assert "retrain" in result.warning.lower()
        # The legacy score is still surfaced for diagnostics.
        assert result.oos_skill_score == pytest.approx(0.2)

    def test_legacy_passthrough_fires_before_skill_gate(self) -> None:
        """TEST-C6-061: legacy routing precedes the skill comparison even when the
        score would otherwise fail the realized-Sharpe gate."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=-5.0,  # would fail any realized gate
            extra={"walk_forward": {"metric_type": "directional_zscore_proxy"}},
        )
        result = check_oos_eligibility(mv, min_oos_skill_score=1.0)
        assert result.eligible is True
        assert result.reason == ""


class TestGateMaxDrawdownFloor:
    """TC#10 -- per-fold max-drawdown floor (B1)."""

    def test_drawdown_above_floor_rejected(self) -> None:
        """TEST-C6-070: worst_fold_drawdown 0.30 > 0.25 (Sharpe ok) ->
        fold_drawdown_exceeds_floor + worst_fold_drawdown_observed==0.30."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=1.5,
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "status": "ok",
                "oos_skill_worst": 0.5,
                "worst_fold_drawdown": 0.30,
            }},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=1.0, max_fold_drawdown=0.25
        )
        assert result.eligible is False
        assert result.reason == "fold_drawdown_exceeds_floor"
        assert result.worst_fold_drawdown_observed == pytest.approx(0.30)

    def test_drawdown_at_floor_not_rejected(self) -> None:
        """TEST-C6-071: worst_fold_drawdown 0.25 == floor (not >) -> not rejected
        on drawdown grounds (eligible)."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=1.5,
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "status": "ok",
                "oos_skill_worst": 0.5,
                "worst_fold_drawdown": 0.25,
            }},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=1.0, max_fold_drawdown=0.25
        )
        assert result.eligible is True
        assert result.reason == ""

    def test_drawdown_below_floor_eligible(self) -> None:
        """TEST-C6-072: worst_fold_drawdown 0.20 <= 0.25 -> eligible, observed set."""
        from api.services.model_activation_gate import check_oos_eligibility

        mv = _make_model_version(
            oos_skill_score=1.2,
            extra={"walk_forward": {
                "metric_type": "realized_oos_sharpe",
                "status": "ok",
                "oos_skill_worst": 0.4,
                "worst_fold_drawdown": 0.20,
            }},
        )
        result = check_oos_eligibility(
            mv, min_oos_skill_score=1.0, max_fold_drawdown=0.25
        )
        assert result.eligible is True
        assert result.worst_fold_drawdown_observed == pytest.approx(0.20)


# ===========================================================================
# TestTrainWfAggregation  (TC#6) -- ccxt + heavy parts mocked for determinism
# ===========================================================================


def _make_fold(index: int, train_n: int, test_n: int, symbol: str) -> Any:
    """A WalkForwardFold-like namespace with train_bars/test_bars dicts."""
    base = make_bars(train_n + test_n, seed=100 + index, symbol=symbol)
    return SimpleNamespace(
        index=index,
        train_bars={symbol: base[:train_n]},
        test_bars={symbol: base[train_n:]},
    )


class TestTrainWfAggregation:
    """TC#6 -- _train_model_with_wf_sync fold aggregation + result-dict contract.

    The heavy machinery (ccxt fetch, ModelTrainer, ModelStrategy, BacktestRunner,
    WalkForwardValidator.split) is mocked so the test is unit-fast and fully
    deterministic. We feed controlled per-fold BacktestResult objects (varying
    oos_sharpe / total_trades / max_drawdown_pct) and assert the function's REAL
    median/worst/max-drawdown aggregation + status logic + result-dict keys.
    """

    def _run(
        self,
        *,
        fold_sharpes: list[float],
        fold_trades: list[int],
        fold_dds: list[float],
        min_trades_per_fold: int = 5,
        num_folds: int = 3,
        symbol: str = "BTC/USD",
    ) -> dict[str, Any]:
        from apps.api.routers import ml as ml_mod  # noqa: F401
        import importlib

        ml = importlib.import_module("api.routers.ml")

        # Synthetic OHLCV ccxt rows (timestamps in ms).
        raw_rows = [
            [1_700_000_000_000 + i * 3_600_000, 100.0, 101.0, 99.0, 100.5, 10.0]
            for i in range(400)
        ]
        fake_exchange = MagicMock()
        fake_exchange.fetch_ohlcv.return_value = raw_rows
        fake_exchange.close.return_value = None
        fake_ccxt = MagicMock()
        fake_ccxt.binance = MagicMock(return_value=fake_exchange)

        # Per-fold BacktestResult objects returned by the mocked runner.run().
        results = [
            _bt_result(oos_sharpe=s, total_trades=t, max_dd=d)
            for s, t, d in zip(fold_sharpes, fold_trades, fold_dds, strict=True)
        ]
        result_iter = iter(results)

        folds = [_make_fold(i, train_n=260, test_n=90, symbol=symbol) for i in range(num_folds)]

        # Mock the full-model ModelTrainer + fold ModelTrainer.
        fake_trainer = MagicMock()
        fake_trainer.prepare_dataset.return_value = ([[0.0] * 10], [0])
        fake_trainer.train.return_value = {"train_samples": 100, "test_samples": 20}
        fake_trainer.save_model.return_value = "models/fake.joblib"

        fake_validator = MagicMock()
        fake_validator.split.return_value = folds

        # ModelStrategy is constructed before runner.run; return a REAL
        # lightweight strategy so BacktestRunner.__init__ (which reads
        # strategy.min_bars_required to size warmup) receives a numeric value.
        # runner.run is patched below, so on_start/on_bar never execute.
        def _fake_model_strategy(*a: Any, **kw: Any) -> Any:
            return _AlwaysHoldStrategy(strategy_id=str(kw.get("strategy_id", "wf")))

        async def _fake_run(self_runner: Any, *a: Any, **kw: Any) -> Any:
            return next(result_iter)

        with patch.dict("sys.modules", {"ccxt": fake_ccxt}), \
             patch("data.ml_training.ModelTrainer", return_value=fake_trainer), \
             patch("trading.walk_forward.WalkForwardValidator", return_value=fake_validator), \
             patch("trading.strategies.model_strategy.ModelStrategy", side_effect=_fake_model_strategy), \
             patch("trading.backtest.BacktestRunner.run", _fake_run):
            return ml._train_model_with_wf_sync(
                symbol=symbol,
                exchange="binance",
                timeframe="1h",
                bars=400,
                n_estimators=5,
                horizon=1,
                threshold=0.001,
                num_wf_folds=num_folds,
                min_trades_per_fold=min_trades_per_fold,
            )

    def test_aggregation_keys_and_median_worst(self) -> None:
        """TEST-C6-080: result dict has all Cycle-6 keys; median/worst/worst-DD
        computed correctly; no DSR/accuracy-proxy keys present."""
        out = self._run(
            fold_sharpes=[1.2, 0.8, 1.6],
            fold_trades=[10, 12, 8],
            fold_dds=[0.10, 0.18, 0.12],
        )
        # Median of [1.2, 0.8, 1.6] = 1.2 ; worst = 0.8 ; worst DD = 0.18
        assert out["walk_forward_oos_skill_score"] == pytest.approx(1.2)
        assert out["walk_forward_oos_sharpe_median"] == pytest.approx(1.2)
        assert out["walk_forward_oos_sharpe_worst"] == pytest.approx(0.8)
        assert out["walk_forward_worst_fold_drawdown"] == pytest.approx(0.18)
        assert out["walk_forward_fold_sharpes"] == pytest.approx([1.2, 0.8, 1.6])
        assert out["walk_forward_fold_trade_counts"] == [10, 12, 8]
        assert out["walk_forward_status"] == "ok"
        assert out["walk_forward_folds_below_threshold"] == []
        # No deflated-sharpe / accuracy-proxy keys leak through.
        assert "walk_forward_oos_skill_score_raw" not in out
        assert "walk_forward_fold_skill_scores" not in out
        assert "walk_forward_oos_skill_score_deflated" not in out

    def test_insufficient_samples_status(self) -> None:
        """TEST-C6-081: a fold with trades < min_trades_per_fold marks status
        'insufficient_samples' and lists folds_below_threshold."""
        out = self._run(
            fold_sharpes=[1.5, 1.4, 1.6],
            fold_trades=[10, 2, 9],  # fold index 1 below floor of 5
            fold_dds=[0.10, 0.05, 0.08],
            min_trades_per_fold=5,
        )
        assert out["walk_forward_status"] == "insufficient_samples"
        assert out["walk_forward_folds_below_threshold"] == [1]

    def test_all_folds_sufficient_status_ok(self) -> None:
        """TEST-C6-082: all folds >= floor -> status 'ok', empty below-threshold."""
        out = self._run(
            fold_sharpes=[1.1, 1.2, 1.3],
            fold_trades=[6, 7, 8],
            fold_dds=[0.10, 0.11, 0.12],
            min_trades_per_fold=5,
        )
        assert out["walk_forward_status"] == "ok"
        assert out["walk_forward_folds_below_threshold"] == []

    def test_no_dsr_call_in_source(self) -> None:
        """TEST-C6-083: deflated_sharpe_ratio is NOT called by the fold loop
        (DSR removed in Cycle 6) -- verified via source introspection."""
        import inspect

        from api.routers import ml as ml_mod

        src = inspect.getsource(ml_mod._train_model_with_wf_sync)
        # Only allowed mention is in prose noting its removal; no call syntax.
        import ast

        tree = ast.parse(src)
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        if (
            func.body
            and isinstance(func.body[0], ast.Expr)
            and isinstance(func.body[0].value, ast.Constant)
        ):
            func.body = func.body[1:]  # drop docstring so prose is excluded
        code_only = ast.unparse(func)

        assert "deflated_sharpe_ratio" not in code_only
        assert "_np" not in code_only  # numpy only used by the removed proxy
        compact = code_only.replace(" ", "")
        assert "(2*acc" not in compact
        assert "2*accuracy" not in compact


# ===========================================================================
# TestTrainWfJsonbContract  (TC#7)
# ===========================================================================


class TestTrainWfJsonbContract:
    """TC#7 -- the JSONB block assembled by train_model from the result dict.

    The persistence block in train_model is inline dict-assembly; we replicate
    the EXACT transform here against a result dict produced by the real
    aggregation (TC#6 surface) to assert the schema_version=3 / metric_type /
    oos_measurement contract and the gate-read key mapping. The result-dict keys
    consumed here are precisely the ones asserted present in TC#6, tying the two
    together.
    """

    def test_jsonb_schema_v3_and_gate_keys(self) -> None:
        """TEST-C6-090: assembled JSONB has schema_version==3,
        metric_type=='realized_oos_sharpe', oos_measurement=='oos_window', and the
        four gate-read keys mapped from the result dict; typed column ==
        median."""
        # A result dict shaped exactly like _train_model_with_wf_sync returns.
        result = {
            "walk_forward_oos_skill_score": 1.2,
            "walk_forward_oos_sharpe_median": 1.2,
            "walk_forward_oos_sharpe_worst": 0.8,
            "walk_forward_fold_sharpes": [1.2, 0.8, 1.6],
            "walk_forward_fold_trade_counts": [10, 12, 8],
            "walk_forward_worst_fold_drawdown": 0.18,
            "walk_forward_status": "ok",
            "walk_forward_folds_below_threshold": [],
        }
        num_wf_folds = 3

        # EXACT replica of the train_model persistence transform (ml.py L178-197).
        existing_extra: dict[str, Any] = {}
        existing_extra["walk_forward"] = {
            "schema_version": 3,
            "metric_type": "realized_oos_sharpe",
            "oos_measurement": "oos_window",
            "status": result.get("walk_forward_status", "ok"),
            "num_folds": num_wf_folds,
            "oos_skill_worst": result.get("walk_forward_oos_sharpe_worst"),
            "fold_trade_counts": result.get("walk_forward_fold_trade_counts", []),
            "folds_below_threshold": result.get("walk_forward_folds_below_threshold", []),
            "oos_sharpe_median": result.get("walk_forward_oos_sharpe_median"),
            "fold_sharpes": result.get("walk_forward_fold_sharpes", []),
            "worst_fold_drawdown": result.get("walk_forward_worst_fold_drawdown"),
        }
        column_value = result.get("walk_forward_oos_skill_score")

        wf = existing_extra["walk_forward"]
        assert wf["schema_version"] == 3
        assert wf["metric_type"] == "realized_oos_sharpe"
        assert wf["oos_measurement"] == "oos_window"
        # Gate-read keys present with the SAME names the gate consumes.
        assert "status" in wf
        assert "oos_skill_worst" in wf
        assert "fold_trade_counts" in wf
        assert "folds_below_threshold" in wf
        assert wf["oos_skill_worst"] == pytest.approx(0.8)
        # Typed column stores the median.
        assert column_value == pytest.approx(wf["oos_sharpe_median"])

    def test_jsonb_block_present_in_source_with_v3(self) -> None:
        """TEST-C6-091: train_model source contains the schema_version=3 +
        realized_oos_sharpe constants (guards against silent regression of the
        replica above)."""
        import inspect

        from api.routers import ml as ml_mod

        src = inspect.getsource(ml_mod.train_model)
        assert '"schema_version": 3' in src
        assert '"metric_type": "realized_oos_sharpe"' in src
        assert '"oos_measurement": "oos_window"' in src
        # Old Cycle-5 proxy markers must be gone from the persistence block.
        assert '"directional_zscore_proxy"' not in src


# ===========================================================================
# Helpers
# ===========================================================================


def _assert_warned(mock_log: Any, event: str) -> None:
    """Assert a structlog .warning(event, ...) was emitted with the given event."""
    calls = [c for c in mock_log.warning.call_args_list if c.args and c.args[0] == event]
    assert calls, f"expected a warning(event={event!r}); got {mock_log.warning.call_args_list}"


def _bt_result(*, oos_sharpe: float, total_trades: int, max_dd: float) -> BacktestResult:
    """A BacktestResult carrying the fields the WF fold loop reads."""
    return _minimal_result(
        oos_sharpe=oos_sharpe,
        oos_n_returns=90,
        total_trades=total_trades,
        max_drawdown_pct=max_dd,
    )


def _minimal_result(**overrides: Any) -> BacktestResult:
    base: dict[str, Any] = dict(
        run_id="c6-test",
        strategy_ids=["s1"],
        symbols=["BTC/USD"],
        timeframe="1h",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 2, 1, tzinfo=UTC),
        duration_days=31,
        initial_capital=Decimal("10000"),
        final_equity=Decimal("10100"),
        total_return_pct=0.01,
        cagr=0.01,
        max_drawdown_pct=0.02,
        max_drawdown_duration_bars=5,
        sharpe_ratio=0.5,
        sortino_ratio=0.6,
        calmar_ratio=0.5,
        total_trades=3,
        winning_trades=2,
        losing_trades=1,
        win_rate=0.667,
        profit_factor=2.0,
        profit_factor_is_infinite=False,
        average_trade_pnl=Decimal("33"),
        average_win=Decimal("50"),
        average_loss=Decimal("0"),
        largest_win=Decimal("100"),
        largest_loss=Decimal("0"),
        total_bars=100,
        bars_in_market=60,
        exposure_pct=0.6,
    )
    base.update(overrides)
    return BacktestResult(**base)
