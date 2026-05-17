"""
tests/unit/test_walk_forward.py
--------------------------------
Unit tests for :mod:`trading.walk_forward` (Sprint 41 QT-004).

Covers
------
1. WalkForwardValidator constructor validation.
2. split() boundary / degenerate / happy paths for both expanding and
   rolling modes.
3. deflated_sharpe_ratio() haircut semantics in both the simple and full
   formulations.
4. _inv_norm_cdf() bounds + spot-value accuracy.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from common.models import OHLCVBar
from trading.walk_forward import (
    WalkForwardFold,
    WalkForwardValidator,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from trading.walk_forward import _inv_norm_cdf  # noqa: PLC2701 (unit test)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int, symbol: str = "BTC/USDT") -> list[OHLCVBar]:
    """Deterministic toy OHLCV series — values don't matter for splitting."""
    from datetime import UTC, datetime, timedelta

    base_ts = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        OHLCVBar(
            symbol=symbol,
            timeframe="1h",
            timestamp=base_ts + timedelta(hours=i),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1"),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# WalkForwardValidator — constructor
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorInit:
    def test_rejects_single_fold(self) -> None:
        with pytest.raises(ValueError, match="num_folds must be >= 2"):
            WalkForwardValidator(num_folds=1)

    def test_rejects_train_fraction_zero(self) -> None:
        with pytest.raises(ValueError, match="train_fraction must be in"):
            WalkForwardValidator(num_folds=4, train_fraction=0.0)

    def test_rejects_train_fraction_one(self) -> None:
        with pytest.raises(ValueError, match="train_fraction must be in"):
            WalkForwardValidator(num_folds=4, train_fraction=1.0)

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            WalkForwardValidator(num_folds=4, mode="diagonal")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WalkForwardValidator — split()
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorSplit:
    def test_rejects_empty_input(self) -> None:
        v = WalkForwardValidator()
        with pytest.raises(ValueError, match="at least one symbol"):
            v.split({})

    def test_rejects_inconsistent_bar_counts(self) -> None:
        v = WalkForwardValidator()
        with pytest.raises(ValueError, match="same number of bars"):
            v.split({"A": _make_bars(100, "A"), "B": _make_bars(50, "B")})

    def test_rejects_too_few_bars(self) -> None:
        v = WalkForwardValidator(num_folds=10)
        # num_folds=10 requires >= 20 bars
        with pytest.raises(ValueError, match="Need at least"):
            v.split({"BTC/USDT": _make_bars(15)})

    def test_expanding_mode_fold_boundaries(self) -> None:
        """Expanding mode: train window starts at 0, grows each fold."""
        v = WalkForwardValidator(num_folds=4, train_fraction=0.7, mode="expanding")
        folds = v.split({"BTC/USDT": _make_bars(100)})

        assert len(folds) == 4

        # fold 0: train[0:70], test[70:77]
        assert len(folds[0].train_bars["BTC/USDT"]) == 70
        assert len(folds[0].test_bars["BTC/USDT"]) == 7

        # All folds: train_start == 0 in expanding mode
        assert folds[0].train_bars["BTC/USDT"][0].timestamp == _make_bars(1)[0].timestamp
        assert folds[3].train_bars["BTC/USDT"][0].timestamp == _make_bars(1)[0].timestamp

        # Train window grows across folds
        train_lengths = [len(f.train_bars["BTC/USDT"]) for f in folds]
        assert train_lengths == sorted(train_lengths), "train must grow monotonically"

        # Last fold consumes all trailing bars
        total_consumed = (
            len(folds[3].train_bars["BTC/USDT"])
            + len(folds[3].test_bars["BTC/USDT"])
        )
        assert total_consumed == 100

    def test_rolling_mode_fold_boundaries(self) -> None:
        """Rolling mode: train window is fixed size, translates forward."""
        v = WalkForwardValidator(num_folds=4, train_fraction=0.7, mode="rolling")
        folds = v.split({"BTC/USDT": _make_bars(100)})

        # All folds have the same training window length (fixed-size)
        train_lengths = [len(f.train_bars["BTC/USDT"]) for f in folds]
        assert len(set(train_lengths)) == 1, (
            f"rolling mode must keep window fixed, got {train_lengths}"
        )

    def test_fold_indices_are_sequential(self) -> None:
        v = WalkForwardValidator(num_folds=3)
        folds = v.split({"BTC/USDT": _make_bars(50)})
        assert [f.index for f in folds] == [0, 1, 2]

    def test_fold_returns_frozen_dataclass(self) -> None:
        v = WalkForwardValidator()
        folds = v.split({"BTC/USDT": _make_bars(100)})
        with pytest.raises((AttributeError, Exception)):
            folds[0].index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


class TestExpectedMaxSharpe:
    def test_zero_stddev_yields_zero_haircut(self) -> None:
        assert expected_max_sharpe(sharpe_stddev=0.0, num_trials=100) == 0.0

    def test_single_trial_yields_zero_haircut(self) -> None:
        assert expected_max_sharpe(sharpe_stddev=1.0, num_trials=1) == 0.0

    def test_simple_formula_matches_analytic(self) -> None:
        """Simple variant: σ · sqrt(2·ln(N))."""
        result = expected_max_sharpe(
            sharpe_stddev=0.3, num_trials=100, use_full_formula=False
        )
        expected = 0.3 * math.sqrt(2.0 * math.log(100))
        assert result == pytest.approx(expected)

    def test_full_formula_in_same_order_as_simple(self) -> None:
        """Both variants are positive and within 25% of each other — the
        full Bailey/López de Prado formulation adds the γ-weighted tail
        term so it can differ materially from the simple sqrt(2·ln N)
        approximation, but must still give the same order of magnitude."""
        simple = expected_max_sharpe(
            sharpe_stddev=0.3, num_trials=100, use_full_formula=False
        )
        full = expected_max_sharpe(
            sharpe_stddev=0.3, num_trials=100, use_full_formula=True
        )
        assert simple > 0
        assert full > 0
        assert abs(simple - full) / simple < 0.25

    def test_haircut_grows_with_num_trials(self) -> None:
        """More trials → bigger expected maximum → bigger haircut."""
        h10 = expected_max_sharpe(sharpe_stddev=0.3, num_trials=10)
        h100 = expected_max_sharpe(sharpe_stddev=0.3, num_trials=100)
        h1000 = expected_max_sharpe(sharpe_stddev=0.3, num_trials=1000)
        assert h10 < h100 < h1000

    def test_rejects_negative_stddev(self) -> None:
        with pytest.raises(ValueError, match="sharpe_stddev must be >= 0"):
            expected_max_sharpe(sharpe_stddev=-0.1, num_trials=10)

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="num_trials must be >= 1"):
            expected_max_sharpe(sharpe_stddev=0.3, num_trials=0)


class TestDeflatedSharpeRatio:
    def test_deflates_by_expected_maximum(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            sharpe_stddev=0.3,
            num_trials=100,
            use_full_formula=False,
        )
        haircut = 0.3 * math.sqrt(2.0 * math.log(100))
        assert dsr == pytest.approx(2.0 - haircut)

    def test_no_deflation_for_single_trial(self) -> None:
        dsr = deflated_sharpe_ratio(observed_sharpe=1.5, sharpe_stddev=0.3, num_trials=1)
        assert dsr == pytest.approx(1.5)

    def test_deflation_can_push_sharpe_negative(self) -> None:
        """A modest observed Sharpe with many trials is likely noise."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            sharpe_stddev=0.3,
            num_trials=500,
            use_full_formula=False,
        )
        assert dsr < 0, "500-trial grid should deflate a 0.5 Sharpe below zero"


# ---------------------------------------------------------------------------
# Inverse standard normal CDF (Acklam approximation)
# ---------------------------------------------------------------------------


class TestInvNormCdf:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _inv_norm_cdf(0.0)
        with pytest.raises(ValueError):
            _inv_norm_cdf(1.0)
        with pytest.raises(ValueError):
            _inv_norm_cdf(-0.1)

    def test_median_is_zero(self) -> None:
        assert _inv_norm_cdf(0.5) == pytest.approx(0.0, abs=1e-8)

    def test_known_quantiles(self) -> None:
        """Spot-check a few standard normal quantiles."""
        # P(Z <= 1.0) ≈ 0.8413
        assert _inv_norm_cdf(0.8413) == pytest.approx(1.0, abs=1e-3)
        # P(Z <= 2.0) ≈ 0.9772
        assert _inv_norm_cdf(0.9772) == pytest.approx(2.0, abs=1e-3)
        # P(Z <= -1.96) ≈ 0.025
        assert _inv_norm_cdf(0.025) == pytest.approx(-1.96, abs=1e-3)

    def test_symmetric_around_median(self) -> None:
        """Φ⁻¹(1-p) = -Φ⁻¹(p)."""
        assert _inv_norm_cdf(0.95) == pytest.approx(-_inv_norm_cdf(0.05), rel=1e-4)
