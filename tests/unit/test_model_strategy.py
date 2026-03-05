"""
tests/unit/test_model_strategy.py
-----------------------------------
Unit tests for the ML Model-Based Trading Strategy.

Module under test
-----------------
packages/trading/strategies/model_strategy.py

Test coverage
-------------
- _safe_log(): positive value, zero guard, negative guard
- _sma_float(): exact period, insufficient data, tail-only window
- _wilder_rsi(): insufficient data fallback, all-up/all-down directional,
  neutral alternating, avg_loss==0 returns 100.0
- ModelStrategy._build_feature_vector(): 10-element output, log_return_1
  correctness, RSI normalisation, SMA ratio sign, volume ratio fallback,
  _rolling_std warmup guard (insufficient log returns)
- ModelStrategy lifecycle: on_start with empty/missing model path, on_stop
  clears state, warmup guard, no-model guard, BUY/SELL/HOLD signal routing,
  below-threshold filter, absent predict_proba uses fixed confidence,
  prediction exception resilience, min_bars_required equals feature_window

Design notes
------------
- All tests are synchronous (strategy methods are not async).
- asyncio_mode = "auto" in pyproject.toml; no @pytest.mark.asyncio needed.
- Bar factory uses datetime objects with TimeFrame -- required by OHLCVBar.
- pytest.approx is used throughout for float comparisons.
- No shared fixtures; each test is fully self-contained.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from common.models import OHLCVBar
from common.types import SignalDirection, TimeFrame
from data.ml_features import _safe_log, _sma_float, _wilder_rsi
from trading.strategies.model_strategy import ModelStrategy


# ---------------------------------------------------------------------------
# Bar factory helpers (local, no conftest dependency)
# ---------------------------------------------------------------------------


def _bar(
    close: float = 100.0,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: float = 1000.0,
    symbol: str = "BTC/USDT",
) -> OHLCVBar:
    """
    Construct a minimal OHLCVBar suitable for model_strategy tests.

    ``high`` defaults to close * 1.01 and ``low`` to close * 0.99 so that
    the OHLCV consistency constraint (low <= open/close <= high) is satisfied
    without callers specifying every field.  ``open_`` defaults to ``close``
    (flat candle).
    """
    c = Decimal(str(close))
    h = Decimal(str(high)) if high is not None else (c * Decimal("1.01"))
    l = Decimal(str(low)) if low is not None else (c * Decimal("0.99"))  # noqa: E741
    o = Decimal(str(open_)) if open_ is not None else c
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=Decimal(str(volume)),
    )


def _bars_n(n: int, base_close: float = 100.0, step: float = 0.0) -> list[OHLCVBar]:
    """
    Generate *n* bars with incrementing (or flat) close prices.

    Successive close prices are ``base_close + i * step`` for i in 0..n-1.
    All bars share the same timestamp; that is acceptable for unit tests
    because model_strategy never inspects timestamps.
    """
    return [_bar(close=base_close + i * step) for i in range(n)]


def _bars_alternating(n: int, low_close: float = 99.0, high_close: float = 101.0) -> list[OHLCVBar]:
    """
    Generate *n* bars alternating between two close prices.

    Used to produce a near-neutral RSI (equal up/down moves).
    """
    return [
        _bar(close=high_close if i % 2 == 0 else low_close)
        for i in range(n)
    ]


def _make_strategy(**params: object) -> ModelStrategy:
    """
    Construct a ModelStrategy with default params, optionally overriding any field.

    Default: feature_window=100, prediction_threshold=0.60, position_size=1000.0,
    model_path="" (no model -- placeholder mode).
    """
    return ModelStrategy(strategy_id="test_ml", params=dict(params))


def _inject_mock_model(
    strategy: ModelStrategy,
    *,
    predict_label: int = 2,
    predict_proba: list[float] | None = None,
) -> MagicMock:
    """
    Inject a MagicMock model into a strategy instance and set _model_loaded=True.

    Parameters
    ----------
    strategy:
        The ModelStrategy to mutate.
    predict_label:
        Class label the mock model.predict returns (0=SELL, 1=HOLD, 2=BUY).
    predict_proba:
        3-element probability list.  If None, defaults to [0.1, 0.1, 0.8].

    Returns
    -------
    MagicMock
        The injected mock model, for caller inspection.
    """
    probas = predict_proba if predict_proba is not None else [0.1, 0.1, 0.8]
    mock_model = MagicMock()
    mock_model.predict.return_value = [predict_label]
    mock_model.predict_proba.return_value = [probas]
    strategy._model = mock_model
    strategy._model_loaded = True
    return mock_model


# ===========================================================================
# _safe_log
# ===========================================================================


class TestSafeLog:
    """Unit tests for the _safe_log module-level helper."""

    def test_positive_value_returns_log(self) -> None:
        """
        _safe_log of a positive Decimal must equal math.log of the same float.

        This verifies the happy path: a valid positive price yields the correct
        natural logarithm without any guard clamping.
        """
        result = _safe_log(Decimal("100"))
        assert result == pytest.approx(math.log(100.0))

    def test_zero_returns_zero(self) -> None:
        """
        _safe_log of zero must return 0.0 instead of raising a math error.

        log(0) is -infinity; the guard clamps it to 0.0 so that feature
        vectors remain finite even on zero-price bars.
        """
        result = _safe_log(Decimal("0"))
        assert result == 0.0

    def test_negative_returns_zero(self) -> None:
        """
        _safe_log of a negative Decimal must return 0.0.

        Negative prices are impossible in spot markets but may appear in
        synthetic test data; the guard prevents a ValueError from math.log.
        """
        result = _safe_log(Decimal("-50"))
        assert result == 0.0

    def test_small_positive_value(self) -> None:
        """
        _safe_log of a very small positive value must return a large negative
        number (not 0.0), because the value is positive.

        This distinguishes the guard boundary: only zero and negative values
        are clamped; any strictly positive value passes through.
        """
        result = _safe_log(Decimal("0.000001"))
        assert result == pytest.approx(math.log(1e-6))
        assert result < 0.0


# ===========================================================================
# _sma_float
# ===========================================================================


class TestSmaFloat:
    """Unit tests for the _sma_float module-level helper."""

    def test_exact_period_returns_average(self) -> None:
        """
        _sma_float over exactly period values must return their arithmetic mean.

        With [10.0, 20.0, 30.0] and period=3 the mean is 20.0.
        """
        result = _sma_float([10.0, 20.0, 30.0], 3)
        assert result == pytest.approx(20.0)

    def test_fewer_than_period_returns_zero(self) -> None:
        """
        _sma_float returns 0.0 when len(values) < period.

        This is the warm-up guard: SMA cannot be computed with insufficient
        history, so 0.0 is used as a safe sentinel that does not distort
        the feature vector.
        """
        result = _sma_float([10.0, 20.0], 3)
        assert result == 0.0

    def test_more_than_period_uses_tail_only(self) -> None:
        """
        _sma_float must average only the last *period* values when len > period.

        With [10.0, 20.0, 30.0, 40.0] and period=3, only [20.0, 30.0, 40.0]
        are used: mean = 30.0.
        """
        result = _sma_float([10.0, 20.0, 30.0, 40.0], 3)
        assert result == pytest.approx(30.0)

    def test_single_value_period_one_returns_that_value(self) -> None:
        """
        _sma_float with period=1 must return the last element of the sequence.
        """
        result = _sma_float([5.0, 10.0, 99.0], 1)
        assert result == pytest.approx(99.0)

    def test_empty_sequence_returns_zero(self) -> None:
        """
        _sma_float with an empty sequence must return 0.0 without raising.
        """
        result = _sma_float([], 3)
        assert result == 0.0


# ===========================================================================
# _wilder_rsi
# ===========================================================================


class TestWilderRsi:
    """Unit tests for the _wilder_rsi module-level helper."""

    def test_insufficient_data_returns_50(self) -> None:
        """
        _wilder_rsi returns 50.0 (neutral) when fewer than period+1 values
        are provided.

        With only 5 values and the default period=14, there is not enough
        history to seed the Wilder averages.
        """
        result = _wilder_rsi([100.0, 101.0, 102.0, 103.0, 104.0], period=14)
        assert result == pytest.approx(50.0)

    def test_all_up_bars_returns_near_100(self) -> None:
        """
        Monotonically increasing prices produce an RSI close to 100.

        With 20 strictly ascending closes and period=14, avg_loss approaches
        zero (all deltas are positive), driving RSI toward 100.
        """
        closes = [100.0 + i for i in range(20)]
        result = _wilder_rsi(closes, period=14)
        assert result > 90.0

    def test_all_down_bars_returns_near_zero(self) -> None:
        """
        Monotonically decreasing prices produce an RSI close to 0.

        With 20 strictly descending closes, avg_gain approaches zero,
        driving RSI toward 0.
        """
        closes = [200.0 - i for i in range(20)]
        result = _wilder_rsi(closes, period=14)
        assert result < 10.0

    def test_neutral_returns_near_50(self) -> None:
        """
        Alternating up/down bars of equal magnitude produce RSI near 50.

        With symmetric gains and losses, the Wilder averages converge to the
        same value, yielding RS=1 and RSI=50.
        """
        closes = [100.0 if i % 2 == 0 else 101.0 for i in range(30)]
        result = _wilder_rsi(closes, period=14)
        # Allow a wide tolerance since Wilder smoothing takes time to settle
        assert 35.0 < result < 65.0

    def test_avg_loss_zero_returns_100(self) -> None:
        """
        When avg_loss is exactly zero (period+1 values, all gains), the
        function must return 100.0 via the early-return guard instead of
        dividing by zero.
        """
        # period=14 requires 15 values; all gains means avg_loss = 0.0
        closes = [100.0 + i for i in range(15)]
        result = _wilder_rsi(closes, period=14)
        assert result == pytest.approx(100.0)

    def test_custom_period(self) -> None:
        """
        _wilder_rsi respects a custom period argument.

        With period=2 and 4 values (>= period+1=3), computation must proceed
        and return a value in the valid [0, 100] RSI range.
        """
        closes = [100.0, 102.0, 101.0, 103.0]
        result = _wilder_rsi(closes, period=2)
        assert 0.0 <= result <= 100.0


# ===========================================================================
# ModelStrategy._build_feature_vector
# ===========================================================================


class TestBuildFeatureVector:
    """
    Unit tests for ModelStrategy._build_feature_vector().

    Each test instantiates a fresh strategy, calls on_start(), and invokes
    _build_feature_vector directly to verify specific properties of the
    10-element output without going through the full on_bar prediction path.
    """

    def test_returns_ten_features(self) -> None:
        """
        _build_feature_vector must return a list of exactly 10 elements for
        any input with at least 100 bars.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(110)
        result = strategy._build_feature_vector(bars)
        assert len(result) == 10

    def test_log_return_1_correct(self) -> None:
        """
        Feature[0] (log_return_1) must equal log(close[-1]) - log(close[-2]).

        Using bars with a known price step, the 1-bar log return is computable
        analytically and can be cross-checked against the feature vector.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=100.0, step=1.0)
        # close[-1] = 199.0, close[-2] = 198.0
        expected = math.log(199.0) - math.log(198.0)
        result = strategy._build_feature_vector(bars)
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_log_return_5_correct(self) -> None:
        """
        Feature[1] (log_return_5) must equal log(close[-1]) - log(close[-6]).
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=100.0, step=1.0)
        # close[-1] = 199.0, close[-6] = 194.0
        expected = math.log(199.0) - math.log(194.0)
        result = strategy._build_feature_vector(bars)
        assert result[1] == pytest.approx(expected, rel=1e-6)

    def test_log_return_10_correct(self) -> None:
        """
        Feature[2] (log_return_10) must equal log(close[-1]) - log(close[-11]).
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=100.0, step=1.0)
        # close[-1] = 199.0, close[-11] = 189.0
        expected = math.log(199.0) - math.log(189.0)
        result = strategy._build_feature_vector(bars)
        assert result[2] == pytest.approx(expected, rel=1e-6)

    def test_rsi_14_normalised_to_0_1(self) -> None:
        """
        Feature[5] (rsi_14) must lie in [0.0, 1.0].

        The production code divides the raw RSI by 100.0 for normalisation.
        This test verifies the constraint holds for typical market-like inputs.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_alternating(100)
        result = strategy._build_feature_vector(bars)
        assert 0.0 <= result[5] <= 1.0

    def test_sma_ratios_positive(self) -> None:
        """
        Features[6] and [7] (SMA ratios) must be positive for positive close
        prices.

        With flat prices, SMA(10)/SMA(50) and SMA(20)/SMA(100) both equal 1.0
        exactly.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=100.0)
        result = strategy._build_feature_vector(bars)
        assert result[6] > 0.0
        assert result[7] > 0.0

    def test_sma_ratio_flat_prices_equals_one(self) -> None:
        """
        For perfectly flat close prices, SMA(10)/SMA(50) and SMA(20)/SMA(100)
        must both equal 1.0 because numerator and denominator are identical.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=50000.0)
        result = strategy._build_feature_vector(bars)
        assert result[6] == pytest.approx(1.0)
        assert result[7] == pytest.approx(1.0)

    def test_volume_ratio_fallback_when_zero(self) -> None:
        """
        Feature[8] (volume_ratio_10) must return 1.0 when all bar volumes are
        zero, activating the fallback guard (sma_vol_10 == 0.0 => 1.0).

        Zero-volume bars can occur in synthetic test data; the guard prevents
        division by zero.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        # volume=0 is allowed by OHLCVBar (volume ge=0)
        zero_vol_bars = [
            OHLCVBar(
                symbol="BTC/USDT",
                timeframe=TimeFrame.ONE_HOUR,
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("0"),
            )
            for _ in range(100)
        ]
        result = strategy._build_feature_vector(zero_vol_bars)
        assert result[8] == pytest.approx(1.0)

    def test_high_low_range_non_negative(self) -> None:
        """
        Feature[9] (high_low_range) must be non-negative for any valid bar.

        high >= low is guaranteed by OHLCVBar's model validator, so the
        difference divided by a positive close is always >= 0.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100)
        result = strategy._build_feature_vector(bars)
        assert result[9] >= 0.0

    def test_high_low_range_formula(self) -> None:
        """
        Feature[9] must equal (high - low) / close of the last bar.

        With a known bar where high=102, low=98, close=100 the expected
        value is (102 - 98) / 100 = 0.04.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        # 99 flat bars, then one bar with explicit high/low
        flat_bars = _bars_n(99)
        last_bar = _bar(close=100.0, high=102.0, low=98.0, open_=100.0)
        bars = flat_bars + [last_bar]
        result = strategy._build_feature_vector(bars)
        assert result[9] == pytest.approx(0.04)

    def test_volatility_features_non_negative(self) -> None:
        """
        Features[3] and [4] (volatility_10, volatility_20) must be non-negative
        as they are computed as sqrt of variance.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        bars = _bars_n(100, base_close=100.0, step=0.5)
        result = strategy._build_feature_vector(bars)
        assert result[3] >= 0.0
        assert result[4] >= 0.0

    def test_volatility_warmup_guard_returns_zero(self) -> None:
        """
        When there are fewer log returns than the rolling volatility window,
        the internal _rolling_std guard must return 0.0 for that feature.

        With 10 bars there are 9 log returns, which is less than the
        volatility_10 window of 10 and far less than volatility_20 window
        of 20.  Both features[3] and features[4] must be 0.0.

        _build_feature_vector is called directly (bypassing the on_bar
        warmup guard) to exercise the _rolling_std branch with minimal data.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        # 10 bars produce 9 log returns -- both 10 and 20 bar windows exceed this
        bars = _bars_n(10, base_close=100.0, step=1.0)
        result = strategy._build_feature_vector(bars)
        # volatility_10: 9 log returns < 10 window -> 0.0
        assert result[3] == pytest.approx(0.0)
        # volatility_20: 9 log returns < 20 window -> 0.0
        assert result[4] == pytest.approx(0.0)


# ===========================================================================
# ModelStrategy lifecycle
# ===========================================================================


class TestModelStrategyLifecycle:
    """
    Unit tests for ModelStrategy lifecycle hooks: on_start, on_bar, on_stop.

    Tests in this class exercise the full signal-generation path by injecting
    mock model objects directly into the strategy instance after on_start()
    is called without a model_path (placeholder mode).
    """

    # ------------------------------------------------------------------ #
    # on_start -- model loading
    # ------------------------------------------------------------------ #

    def test_on_start_empty_model_path_stays_unloaded(self) -> None:
        """
        on_start() with the default empty model_path must NOT set
        _model_loaded=True.

        The strategy stays in placeholder mode: _model is None and
        _model_loaded is False.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        assert strategy._model_loaded is False
        assert strategy._model is None

    def test_on_start_sets_run_id(self) -> None:
        """
        on_start() must propagate the run_id to the BaseStrategy._run_id
        attribute regardless of whether a model is loaded.
        """
        strategy = _make_strategy()
        strategy.on_start("run-abc")
        assert strategy.run_id == "run-abc"

    def test_on_start_missing_file_stays_unloaded(self) -> None:
        """
        on_start() with a model_path pointing to a non-existent file must
        not raise and must leave _model_loaded=False.

        The strategy logs a warning and continues in placeholder mode.
        """
        strategy = _make_strategy(model_path="/nonexistent/model.pkl")
        strategy.on_start("run1")
        assert strategy._model_loaded is False
        assert strategy._model is None

    def test_on_start_whitespace_only_path_stripped_and_treated_as_empty(self) -> None:
        """
        A model_path consisting solely of whitespace must be stripped by the
        Pydantic validator and treated as an empty string, leaving the
        strategy unloaded.
        """
        strategy = _make_strategy(model_path="   ")
        strategy.on_start("run1")
        assert strategy._model_loaded is False

    def test_on_start_joblib_import_error_stays_unloaded(self) -> None:
        """
        If joblib is not installed, on_start() must catch the ImportError,
        log an error, and remain in placeholder mode (_model_loaded=False).

        The test uses pathlib.Path.exists to return True so that the file
        check passes, then patches joblib.load to raise ImportError.
        """
        strategy = _make_strategy(model_path="/fake/model.pkl")
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("joblib.load", side_effect=ImportError("no joblib")),
        ):
            strategy.on_start("run1")
        assert strategy._model_loaded is False

    def test_on_start_load_exception_stays_unloaded(self) -> None:
        """
        If joblib.load raises a non-ImportError exception (e.g. file is
        corrupt), on_start() must catch it and leave _model_loaded=False.
        """
        strategy = _make_strategy(model_path="/fake/model.pkl")
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("joblib.load", side_effect=OSError("corrupt file")),
        ):
            strategy.on_start("run1")
        assert strategy._model_loaded is False

    def test_on_start_successful_load_sets_model_loaded(self) -> None:
        """
        When joblib.load succeeds, on_start() must set _model_loaded=True
        and assign the loaded object to _model.
        """
        fake_model = MagicMock()
        strategy = _make_strategy(model_path="/valid/model.pkl")
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("joblib.load", return_value=fake_model),
        ):
            strategy.on_start("run1")
        assert strategy._model_loaded is True
        assert strategy._model is fake_model

    # ------------------------------------------------------------------ #
    # on_stop -- teardown
    # ------------------------------------------------------------------ #

    def test_on_stop_clears_model(self) -> None:
        """
        on_stop() must set both _model and _model_loaded to their reset
        states (None and False respectively), releasing the model reference.
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        # Manually inject a model as if loading succeeded
        strategy._model = MagicMock()
        strategy._model_loaded = True

        strategy.on_stop()

        assert strategy._model is None
        assert strategy._model_loaded is False

    def test_on_stop_idempotent_when_no_model(self) -> None:
        """
        on_stop() must not raise even when called without a model being
        loaded (i.e. when _model is None and _model_loaded is False).
        """
        strategy = _make_strategy()
        strategy.on_start("run1")
        # Should not raise
        strategy.on_stop()
        assert strategy._model is None
        assert strategy._model_loaded is False

    # ------------------------------------------------------------------ #
    # min_bars_required
    # ------------------------------------------------------------------ #

    def test_min_bars_required_equals_feature_window_default(self) -> None:
        """
        min_bars_required must equal the feature_window parameter value.
        The default feature_window is 100.
        """
        strategy = _make_strategy()
        assert strategy.min_bars_required == 100

    def test_min_bars_required_reflects_custom_feature_window(self) -> None:
        """
        min_bars_required must reflect a custom feature_window passed at
        construction time.
        """
        strategy = _make_strategy(feature_window=200)
        assert strategy.min_bars_required == 200

    # ------------------------------------------------------------------ #
    # on_bar -- warmup guard
    # ------------------------------------------------------------------ #

    def test_on_bar_warmup_returns_empty(self) -> None:
        """
        on_bar() must return [] when the number of bars is less than
        feature_window, regardless of whether a model is loaded.

        This is the warm-up guard that prevents feature extraction before
        sufficient history is available.
        """
        strategy = _make_strategy(feature_window=100)
        strategy.on_start("run1")
        _inject_mock_model(strategy)
        bars = _bars_n(50)  # below feature_window of 100
        assert strategy.on_bar(bars) == []

    def test_on_bar_exactly_at_warmup_boundary_returns_signal(self) -> None:
        """
        on_bar() must NOT skip when bars == feature_window (boundary case).

        The production code is ``if len(bars) < feature_window``, so providing
        exactly feature_window bars must pass the guard and proceed to prediction.
        """
        strategy = _make_strategy(feature_window=100, prediction_threshold=0.5)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.1, 0.1, 0.8])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        assert len(signals) == 1

    # ------------------------------------------------------------------ #
    # on_bar -- no model guard
    # ------------------------------------------------------------------ #

    def test_on_bar_no_model_returns_empty(self) -> None:
        """
        on_bar() must return [] when _model_loaded is False, even if enough
        bars are provided.

        This covers the placeholder mode where no model file is configured.
        """
        strategy = _make_strategy(feature_window=100)
        strategy.on_start("run1")
        # _model_loaded remains False (no model_path configured)
        bars = _bars_n(110)
        assert strategy.on_bar(bars) == []

    # ------------------------------------------------------------------ #
    # on_bar -- BUY signal
    # ------------------------------------------------------------------ #

    def test_on_bar_mock_model_buy_above_threshold(self) -> None:
        """
        When the mock model predicts label=2 (BUY) with probability 0.8, which
        exceeds the default threshold of 0.6, on_bar() must return exactly one
        Signal with direction=BUY.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.1, 0.1, 0.8])
        bars = _bars_n(100, base_close=100.0, step=0.5)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].strategy_id == "test_ml"

    def test_on_bar_buy_signal_has_correct_target_position(self) -> None:
        """
        The emitted BUY signal must have target_position equal to the
        position_size parameter (converted to Decimal).
        """
        strategy = _make_strategy(position_size=2500.0, prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.1, 0.1, 0.8])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        assert signals[0].target_position == Decimal("2500.0")

    def test_on_bar_buy_signal_confidence_clamped(self) -> None:
        """
        Signal confidence must be in [0.0, 1.0] and equal to the BUY class
        probability from predict_proba, rounded to 4 decimal places.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.05, 0.05, 0.90])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        assert 0.0 <= signals[0].confidence <= 1.0
        assert signals[0].confidence == pytest.approx(0.9, rel=1e-3)

    # ------------------------------------------------------------------ #
    # on_bar -- SELL signal
    # ------------------------------------------------------------------ #

    def test_on_bar_mock_model_sell_above_threshold(self) -> None:
        """
        When the mock model predicts label=0 (SELL) with probability 0.8,
        on_bar() must return exactly one Signal with direction=SELL.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=0, predict_proba=[0.8, 0.1, 0.1])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.SELL

    def test_on_bar_sell_signal_symbol_matches_last_bar(self) -> None:
        """
        The SELL signal's symbol must match the symbol on the last bar in
        the bars sequence.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=0, predict_proba=[0.8, 0.1, 0.1])
        bars = [_bar(symbol="ETH/USDT") for _ in range(100)]
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        assert signals[0].symbol == "ETH/USDT"

    # ------------------------------------------------------------------ #
    # on_bar -- HOLD
    # ------------------------------------------------------------------ #

    def test_on_bar_mock_model_hold_returns_empty(self) -> None:
        """
        When the mock model predicts label=1 (HOLD), on_bar() must return []
        regardless of the confidence value.

        The HOLD direction is filtered before the threshold check.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=1, predict_proba=[0.05, 0.90, 0.05])
        bars = _bars_n(100)
        assert strategy.on_bar(bars) == []

    # ------------------------------------------------------------------ #
    # on_bar -- threshold filter
    # ------------------------------------------------------------------ #

    def test_on_bar_below_threshold_returns_empty(self) -> None:
        """
        When the predicted BUY probability (0.4) is below the configured
        threshold (0.6), on_bar() must return [] rather than emitting a signal.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.3, 0.3, 0.4])
        bars = _bars_n(100)
        assert strategy.on_bar(bars) == []

    def test_on_bar_at_threshold_boundary_emits_signal(self) -> None:
        """
        When confidence equals the threshold exactly, on_bar() must NOT filter
        the signal.

        The production code is ``if confidence < prediction_threshold``, so
        confidence == threshold must pass through and emit a signal.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.2, 0.2, 0.6])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY

    def test_on_bar_threshold_zero_always_emits_on_buy(self) -> None:
        """
        With prediction_threshold=0.0, any BUY prediction (even very low
        confidence) must emit a signal.
        """
        strategy = _make_strategy(prediction_threshold=0.0)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.33, 0.33, 0.34])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY

    def test_on_bar_threshold_one_never_emits(self) -> None:
        """
        With prediction_threshold=1.0, no prediction can ever meet the bar
        (probabilities are always < 1.0 in practice), so on_bar() must
        return [] for any real-valued confidence.
        """
        strategy = _make_strategy(prediction_threshold=1.0)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.0, 0.0, 0.999])
        bars = _bars_n(100)
        assert strategy.on_bar(bars) == []

    # ------------------------------------------------------------------ #
    # on_bar -- absent predict_proba
    # ------------------------------------------------------------------ #

    def test_on_bar_no_predict_proba_uses_fixed_confidence(self) -> None:
        """
        When the injected model has no predict_proba attribute, on_bar() must
        fall back to a fixed confidence of 0.5.

        With the default threshold of 0.6, a BUY prediction with confidence
        0.5 must NOT cross the threshold, so on_bar() returns [].
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")

        # Build a model that lacks predict_proba
        mock_model = MagicMock(spec=["predict"])  # spec excludes predict_proba
        mock_model.predict.return_value = [2]  # BUY
        strategy._model = mock_model
        strategy._model_loaded = True

        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        # confidence 0.5 < threshold 0.6 -- signal suppressed
        assert signals == []

    def test_on_bar_no_predict_proba_below_threshold_with_lower_threshold(self) -> None:
        """
        When there is no predict_proba and threshold is set below 0.5, the
        fixed confidence of 0.5 must be sufficient to pass the threshold and
        emit a BUY signal.
        """
        strategy = _make_strategy(prediction_threshold=0.40)
        strategy.on_start("run1")

        mock_model = MagicMock(spec=["predict"])
        mock_model.predict.return_value = [2]  # BUY
        strategy._model = mock_model
        strategy._model_loaded = True

        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        # confidence 0.5 >= threshold 0.4 -- signal emitted
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].confidence == pytest.approx(0.5, rel=1e-3)

    # ------------------------------------------------------------------ #
    # on_bar -- exception resilience
    # ------------------------------------------------------------------ #

    def test_on_bar_prediction_exception_returns_empty(self) -> None:
        """
        When model.predict raises a RuntimeError, on_bar() must catch it,
        log an error, and return [] rather than propagating the exception.

        This guards the bar loop from being crashed by a defective model.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model exploded")
        strategy._model = mock_model
        strategy._model_loaded = True

        bars = _bars_n(100)
        # Must not raise
        signals = strategy.on_bar(bars)
        assert signals == []

    def test_on_bar_predict_proba_exception_returns_empty(self) -> None:
        """
        When model.predict_proba raises, on_bar() must handle it gracefully
        and return [] without propagating the exception.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")

        mock_model = MagicMock()
        mock_model.predict.return_value = [2]
        mock_model.predict_proba.side_effect = RuntimeError("proba failed")
        strategy._model = mock_model
        strategy._model_loaded = True

        bars = _bars_n(100)
        signals = strategy.on_bar(bars)
        assert signals == []

    # ------------------------------------------------------------------ #
    # on_bar -- signal metadata
    # ------------------------------------------------------------------ #

    def test_on_bar_signal_metadata_contains_feature_keys(self) -> None:
        """
        The emitted Signal's metadata dict must contain the 'features' key
        with all 10 named feature entries.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.1, 0.1, 0.8])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        meta = signals[0].metadata
        assert "features" in meta
        features = meta["features"]
        expected_keys = {
            "log_return_1", "log_return_5", "log_return_10",
            "volatility_10", "volatility_20", "rsi_14",
            "sma_ratio_10_50", "sma_ratio_20_100",
            "volume_ratio_10", "high_low_range",
        }
        assert set(features.keys()) == expected_keys

    def test_on_bar_signal_metadata_contains_prediction_info(self) -> None:
        """
        Signal metadata must contain model_type, prediction_label,
        prediction_confidence, and prediction_threshold keys.
        """
        strategy = _make_strategy(prediction_threshold=0.60)
        strategy.on_start("run1")
        _inject_mock_model(strategy, predict_label=2, predict_proba=[0.1, 0.1, 0.8])
        bars = _bars_n(100)
        signals = strategy.on_bar(bars)

        assert len(signals) == 1
        meta = signals[0].metadata
        assert "model_type" in meta
        assert "prediction_label" in meta
        assert meta["prediction_label"] == 2
        assert "prediction_confidence" in meta
        assert "prediction_threshold" in meta
        assert meta["prediction_threshold"] == pytest.approx(0.60)

    # ------------------------------------------------------------------ #
    # Parameter validation
    # ------------------------------------------------------------------ #

    def test_invalid_feature_window_below_minimum_raises(self) -> None:
        """
        feature_window < 100 must raise a ValidationError at construction time.

        The Pydantic schema declares ge=100 for feature_window to ensure
        enough bars for SMA(100) computation.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_strategy(feature_window=50)

    def test_invalid_feature_window_above_maximum_raises(self) -> None:
        """
        feature_window > 5000 must raise a ValidationError at construction time.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_strategy(feature_window=10000)

    def test_invalid_prediction_threshold_above_one_raises(self) -> None:
        """
        prediction_threshold > 1.0 must raise a ValidationError.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_strategy(prediction_threshold=1.5)

    def test_invalid_position_size_zero_raises(self) -> None:
        """
        position_size must be > 0.0 (gt constraint). A value of 0.0 must
        raise a ValidationError.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_strategy(position_size=0.0)

    def test_valid_boundary_feature_window_100(self) -> None:
        """
        feature_window=100 (the minimum allowed) must not raise.
        """
        strategy = _make_strategy(feature_window=100)
        assert strategy.min_bars_required == 100

    def test_valid_boundary_prediction_threshold_zero(self) -> None:
        """
        prediction_threshold=0.0 (the minimum allowed) must not raise.
        """
        strategy = _make_strategy(prediction_threshold=0.0)
        assert strategy._params["prediction_threshold"] == pytest.approx(0.0)

    # ------------------------------------------------------------------ #
    # parameter_schema
    # ------------------------------------------------------------------ #

    def test_parameter_schema_returns_dict(self) -> None:
        """
        parameter_schema() must return a non-empty dict (JSON Schema object).
        """
        schema = ModelStrategy.parameter_schema()
        assert isinstance(schema, dict)
        assert len(schema) > 0

    def test_parameter_schema_contains_model_path(self) -> None:
        """
        The returned JSON Schema must describe the model_path property.
        """
        schema = ModelStrategy.parameter_schema()
        properties = schema.get("properties", {})
        assert "model_path" in properties

    def test_parameter_schema_contains_feature_window(self) -> None:
        """
        The returned JSON Schema must describe the feature_window property.
        """
        schema = ModelStrategy.parameter_schema()
        properties = schema.get("properties", {})
        assert "feature_window" in properties
