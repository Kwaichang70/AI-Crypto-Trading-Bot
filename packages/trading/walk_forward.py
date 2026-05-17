"""
packages/trading/walk_forward.py
---------------------------------
Walk-forward validation + Deflated Sharpe Ratio (Sprint 41 QT-004).

Context
-------
Sprint 29 shipped :class:`ParameterOptimizer` — a grid search that
evaluates every combination on the full sample and ranks by in-sample
Sharpe.  A textbook overfit setup: the winner is almost guaranteed to be
the combination that best fits the noise rather than the signal.

This module adds two pieces that address QT-004 from the Sprint 40
refine review:

1. :class:`WalkForwardValidator` slices the full OHLCV history into K
   (train_window, test_window) folds.  The optimizer can then evaluate
   each parameter combination on every test fold, aggregate the
   out-of-sample Sharpe, and rank by OOS performance instead of the
   trivially-overfit in-sample metric.

2. :func:`deflated_sharpe_ratio` applies the haircut derived in
   Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, and
   López de Prado (2018), *Advances in Financial Machine Learning*
   ch. 7.  Running N independent backtests inflates the expected
   maximum Sharpe by roughly ``σ_SR · √(2·ln N)``; the deflated
   variant subtracts that expectation so the reported headline
   number reflects the signal that survives the multiple-testing bias.

MVP scope
---------
The walk-forward splitter supports *expanding* (growing training
window) and *rolling* (fixed-size window) schemes with configurable
fold count.  The DSR uses the haircut approximation, not the full
Probabilistic Sharpe Ratio — adequate for gating optimizer outputs
before live deployment; a future sprint can tighten the statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from common.models import OHLCVBar

__all__ = [
    "WalkForwardFold",
    "WalkForwardValidator",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
]


# ---------------------------------------------------------------------------
# Fold dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test split.

    Attributes
    ----------
    index:
        Zero-based fold index in the sequence of folds produced by the
        validator.
    train_bars:
        Mapping of ``symbol -> bars`` to be used for strategy warmup
        and any in-sample fitting logic.  Strictly earlier in time than
        ``test_bars``.
    test_bars:
        Mapping of ``symbol -> bars`` the backtest must evaluate.  This
        is the out-of-sample window — metrics derived from these bars
        are the ones reported for ranking.
    """

    index: int
    train_bars: dict[str, list[OHLCVBar]]
    test_bars: dict[str, list[OHLCVBar]]


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Split OHLCV history into train/test folds for walk-forward evaluation.

    Both *expanding* and *rolling* schemes are supported.  The validator is
    symbol-agnostic: it slices every symbol's bar list by the same index
    boundaries so each fold covers the same calendar window for every
    symbol (implicitly assuming all symbols share the same timeframe and
    bar timestamps, which is how :class:`BacktestRunner` already consumes
    the data).

    Parameters
    ----------
    num_folds:
        How many folds to emit.  Must be ``>= 2`` — a single fold would
        degenerate back to the in-sample case.
    train_fraction:
        Fraction of the bars allocated to the training window in each
        fold.  Must satisfy ``0 < train_fraction < 1``.  Only meaningful
        in rolling mode; expanding mode uses a shared anchor.
    mode:
        ``"expanding"`` (default) — training window grows from the
        anchor on every step; test window slides forward.  Typical for
        strategy evaluation because it mimics how a live system would
        accumulate history.
        ``"rolling"`` — training window size is fixed; both windows
        translate forward.  Useful when the series has structural
        breaks that make very-old data unrepresentative.
    """

    def __init__(
        self,
        num_folds: int = 4,
        train_fraction: float = 0.7,
        mode: Literal["expanding", "rolling"] = "expanding",
    ) -> None:
        if num_folds < 2:
            raise ValueError(f"num_folds must be >= 2, got {num_folds}")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must be in (0, 1), got {train_fraction}"
            )
        if mode not in ("expanding", "rolling"):
            raise ValueError(
                f"mode must be 'expanding' or 'rolling', got {mode!r}"
            )

        self._num_folds = num_folds
        self._train_fraction = train_fraction
        self._mode = mode

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @property
    def mode(self) -> Literal["expanding", "rolling"]:
        return self._mode

    def split(
        self, bars_by_symbol: dict[str, list[OHLCVBar]]
    ) -> list[WalkForwardFold]:
        """Produce ``num_folds`` folds from the input bar series.

        Raises
        ------
        ValueError
            When the input is empty, when symbols have inconsistent bar
            counts, or when the derived fold sizes would be degenerate
            (train or test window empty for any fold).
        """
        if not bars_by_symbol:
            raise ValueError("bars_by_symbol must contain at least one symbol")

        bar_counts = {sym: len(bars) for sym, bars in bars_by_symbol.items()}
        unique_counts = set(bar_counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                f"All symbols must have the same number of bars; got {bar_counts}"
            )
        total_bars = next(iter(unique_counts))
        if total_bars < self._num_folds * 2:
            raise ValueError(
                f"Need at least {self._num_folds * 2} bars for "
                f"{self._num_folds} folds; got {total_bars}"
            )

        # Divide the tail of the series into num_folds equal test windows.
        # The head (or an expanding window anchored at it) feeds training.
        #
        # Expanding example with 100 bars and num_folds=4, train_fraction=0.7:
        #   fold 0: train=[  0 .. 70), test=[70 .. 77)
        #   fold 1: train=[  0 .. 77), test=[77 .. 85)   (test windows
        #   fold 2: train=[  0 .. 85), test=[85 .. 92)    sized so the
        #   fold 3: train=[  0 .. 92), test=[92 ..100)    last one hits end)
        #
        # Rolling example, same shapes:
        #   fold 0: train=[ 0..70),  test=[70..77)
        #   fold 1: train=[ 7..77),  test=[77..85)
        #   fold 2: train=[15..85),  test=[85..92)
        #   fold 3: train=[22..92),  test=[92..100)

        initial_train = int(total_bars * self._train_fraction)
        remaining = total_bars - initial_train
        if remaining < self._num_folds:
            raise ValueError(
                "train_fraction leaves too little tail for the requested "
                f"num_folds — initial_train={initial_train}, "
                f"total={total_bars}, folds={self._num_folds}"
            )

        test_size = remaining // self._num_folds
        if test_size == 0:
            raise ValueError(
                "Derived test window would be empty — increase bar count "
                "or reduce num_folds"
            )

        folds: list[WalkForwardFold] = []
        for i in range(self._num_folds):
            test_start = initial_train + i * test_size
            # Last fold consumes everything up to the end so rounding
            # leftovers do not get silently discarded.
            test_end = (
                total_bars
                if i == self._num_folds - 1
                else test_start + test_size
            )

            if self._mode == "expanding":
                train_start = 0
            else:  # rolling
                train_start = test_start - initial_train

            train_end = test_start

            train_slice = {
                sym: bars[train_start:train_end]
                for sym, bars in bars_by_symbol.items()
            }
            test_slice = {
                sym: bars[test_start:test_end]
                for sym, bars in bars_by_symbol.items()
            }

            # Defence-in-depth: the slicing math above guarantees non-empty
            # windows, but assert so a future refactor cannot silently emit
            # a degenerate fold.
            assert all(len(v) > 0 for v in train_slice.values()), (
                f"fold {i} train slice empty for some symbol"
            )
            assert all(len(v) > 0 for v in test_slice.values()), (
                f"fold {i} test slice empty for some symbol"
            )

            folds.append(
                WalkForwardFold(
                    index=i,
                    train_bars=train_slice,
                    test_bars=test_slice,
                )
            )

        return folds


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def expected_max_sharpe(
    sharpe_stddev: float,
    num_trials: int,
    *,
    use_full_formula: bool = True,
) -> float:
    """Expected maximum Sharpe across ``num_trials`` independent backtests.

    Haircut approximation used to deflate the observed best Sharpe so the
    reported number reflects the signal rather than the multiple-testing
    bias.  Two variants:

    Simple (``use_full_formula=False``):
        ``σ_SR · √(2·ln N)`` — corresponds to the expected maximum of N
        iid standard normals.  Good enough for small-to-medium N and
        fast to compute.

    Full (default, ``use_full_formula=True``):
        ``σ_SR · ((1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)))``, where
        γ ≈ 0.5772 is the Euler–Mascheroni constant.  Tighter at
        moderate N; follows the Bailey & López de Prado (2014)
        formulation.

    Parameters
    ----------
    sharpe_stddev:
        Standard deviation of the Sharpe ratios observed across trials.
        Must be ``>= 0``; when ``0`` the function returns ``0``.
    num_trials:
        Number of trials / parameter combinations evaluated.  Must be
        ``>= 1``.  When ``1`` the haircut is ``0`` (no multiple-testing
        adjustment required).

    Returns
    -------
    float
        The additive haircut.  Subtract from the observed maximum
        Sharpe to obtain the Deflated Sharpe Ratio.
    """
    if sharpe_stddev < 0:
        raise ValueError(f"sharpe_stddev must be >= 0, got {sharpe_stddev}")
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")
    if num_trials == 1 or sharpe_stddev == 0.0:
        return 0.0

    if not use_full_formula:
        return sharpe_stddev * math.sqrt(2.0 * math.log(num_trials))

    # Full formula uses Φ⁻¹ (inverse standard normal CDF).  We avoid the
    # SciPy dependency by approximating via a rational series (Acklam).
    gamma = 0.5772156649015329  # Euler–Mascheroni constant
    term_1 = _inv_norm_cdf(1.0 - 1.0 / num_trials)
    term_2 = _inv_norm_cdf(1.0 - 1.0 / (num_trials * math.e))
    return sharpe_stddev * ((1.0 - gamma) * term_1 + gamma * term_2)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_stddev: float,
    num_trials: int,
    *,
    use_full_formula: bool = True,
) -> float:
    """Deflate ``observed_sharpe`` by the expected maximum under the null.

    See :func:`expected_max_sharpe` for the haircut derivation.  This is
    the one-call entry point used by the optimizer to report the
    bias-adjusted Sharpe alongside the raw value.
    """
    haircut = expected_max_sharpe(
        sharpe_stddev, num_trials, use_full_formula=use_full_formula
    )
    return observed_sharpe - haircut


# ---------------------------------------------------------------------------
# Inverse standard normal CDF (Acklam rational approximation)
# ---------------------------------------------------------------------------
# Source: P. J. Acklam, "An algorithm for computing the inverse normal
# cumulative distribution function".  Max absolute error ~1.15e-9, which
# is well below the precision we need for a Sharpe haircut.


def _inv_norm_cdf(p: float) -> float:
    """Inverse of the standard normal CDF for ``0 < p < 1``.

    Raises ``ValueError`` for ``p`` outside the open unit interval so
    callers cannot silently receive ``±inf`` when upstream math produces
    a degenerate probability.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients (Acklam)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
