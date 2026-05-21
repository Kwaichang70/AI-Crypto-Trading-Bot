"""
packages/trading/strategies/ensemble.py
----------------------------------------
QT-008 (Sprint 46) -- Strategy ensemble layer.

Wraps multiple BaseStrategy instances and combines their per-bar signals
into a single output signal per symbol.  Three combination methods are
provided:

  - ``equal_weight``  -- weighted average with uniform weights.
  - ``inverse_vol``   -- weighted by 1 / rolling-std of each sub-strategy's
                        recent signal magnitudes.  Less volatile (more
                        consistent) sub-strategies receive higher weight.
                        Subs that have NEVER emitted a non-zero magnitude
                        within the lookback receive ZERO weight (the
                        opposite of what a naive 1/eps clamp would do).
  - ``majority_vote`` -- discrete vote; output direction is the most-voted
                        non-HOLD direction iff its vote share >= min_agreement,
                        else no signal is emitted.

Combined signal contract:
  - ``direction``: BUY, SELL.  HOLD is never emitted as a Signal; the
    method returns None and on_bar drops it from the output list.
  - ``target_position``: always non-negative; direction is carried by
    the ``direction`` field (matches the Signal model invariant
    ``ge=Decimal(0)``).
  - ``confidence``: contribution-weighted average of sub-strategy
    confidences, clipped to [0, 1].

Backward compat: HTF timeframes are the UNION of all sub-strategies'.
``min_bars_required`` is the MAX of all sub-strategies' so the engine
warms up enough history for every wrapped strategy.

MVP scope (Sprint 46 QT-008): class + unit tests only.  Registry +
API/UI integration deferred to Sprint 47.

Multi-symbol fairness (S47-4 Sprint 47):
the inverse-vol magnitude history is keyed PER SYMBOL.  A sub trading
4 symbols therefore has the same per-symbol weight as a single-symbol
sub of equivalent consistency, fixing the QT-008 CR-005 limitation.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar

import structlog

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["EnsembleStrategy"]

logger = structlog.get_logger(__name__)


_VALID_METHODS: frozenset[str] = frozenset(
    {"equal_weight", "inverse_vol", "majority_vote"}
)
_DEFAULT_VOL_LOOKBACK = 20
_MIN_VOL_LOOKBACK = 3
_MAX_VOL_LOOKBACK = 200


class EnsembleStrategy(BaseStrategy):
    """Combine multiple sub-strategies into a single output stream.

    Parameters (constructor kwargs)
    -------------------------------
    sub_strategies:
        Pre-instantiated ``BaseStrategy`` instances.  Must be non-empty.
    combination_method:
        One of ``"equal_weight"``, ``"inverse_vol"``, ``"majority_vote"``.
        Default ``"equal_weight"``.
    vol_lookback:
        Number of recent signal magnitudes to use for inverse-vol
        weighting (default 20; clamped to [3, 200]).
    min_agreement:
        For ``majority_vote``: minimum vote share (in [0.5, 1.0]) for the
        winning direction.  Default 0.6.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Ensemble Strategy",
        version="0.1.0",
        description=(
            "Combines multiple sub-strategies using equal-weight, "
            "inverse-volatility, or majority-vote rules."
        ),
        author="trading-engine-architect",
        tags=["ensemble", "meta", "voting"],
    )

    def __init__(
        self,
        strategy_id: str,
        sub_strategies: list[BaseStrategy],
        *,
        combination_method: str = "equal_weight",
        vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
        min_agreement: float = 0.6,
    ) -> None:
        if not sub_strategies:
            raise ValueError("sub_strategies must contain at least one BaseStrategy")
        if combination_method not in _VALID_METHODS:
            raise ValueError(
                f"combination_method must be one of {sorted(_VALID_METHODS)}, "
                f"got {combination_method!r}"
            )
        if not (_MIN_VOL_LOOKBACK <= vol_lookback <= _MAX_VOL_LOOKBACK):
            raise ValueError(
                f"vol_lookback must be in [{_MIN_VOL_LOOKBACK}, "
                f"{_MAX_VOL_LOOKBACK}], got {vol_lookback}"
            )
        if not (0.5 <= min_agreement <= 1.0):
            raise ValueError(
                f"min_agreement must be in [0.5, 1.0], got {min_agreement}"
            )

        super().__init__(
            strategy_id=strategy_id,
            params={
                "combination_method": combination_method,
                "vol_lookback": vol_lookback,
                "min_agreement": min_agreement,
                "sub_strategy_ids": [s.strategy_id for s in sub_strategies],
            },
        )
        self._sub_strategies = list(sub_strategies)
        self._combination_method = combination_method
        self._vol_lookback = vol_lookback
        self._min_agreement = min_agreement
        # S47-4 (Sprint 47): symbol-keyed magnitude history.  Each symbol
        # tracks one deque per sub-strategy of the most recent
        # ``|target_position|`` values for THAT symbol -- 0.0 when the sub
        # did not emit for the symbol on a bar where some other sub did.
        # A sub that never trades a given symbol therefore accumulates an
        # all-zero history for it and is excluded (weight 0) from that
        # symbol's combination, matching the CR-002 silent-sub semantic
        # without falsely penalising multi-symbol subs.
        self._magnitude_history: dict[str, list[deque[float]]] = {}

    def _ensure_history_for_symbol(self, symbol: str) -> None:
        """Lazily allocate the per-sub deque list for a newly-seen symbol."""
        if symbol not in self._magnitude_history:
            self._magnitude_history[symbol] = [
                deque(maxlen=self._vol_lookback) for _ in self._sub_strategies
            ]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def min_bars_required(self) -> int:
        """Max over sub-strategies so the engine warms up enough history."""
        return max((s.min_bars_required for s in self._sub_strategies), default=0)

    @property
    def htf_timeframes(self) -> list[str]:
        """Union of all sub-strategies' HTF timeframe declarations."""
        seen: list[str] = []
        for sub in self._sub_strategies:
            for tf in sub.htf_timeframes:
                if tf not in seen:
                    seen.append(tf)
        return seen

    @property
    def sub_strategies(self) -> tuple[BaseStrategy, ...]:
        """Read-only view of the wrapped sub-strategies."""
        return tuple(self._sub_strategies)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_start(self, run_id: str) -> None:
        # Fail-fast: if any sub-strategy raises during on_start, the
        # exception propagates immediately.  No rollback of already-started
        # subs is performed; callers must treat on_start failure as
        # non-recoverable and not call on_bar (CR-006 documented contract).
        super().on_start(run_id)
        for sub in self._sub_strategies:
            sub.on_start(run_id)

    def on_stop(self) -> None:
        for sub in self._sub_strategies:
            try:
                sub.on_stop()
            except Exception:
                self._log.warning(
                    "ensemble_strategy.sub_on_stop_failed",
                    sub_id=sub.strategy_id,
                )
        for deques in self._magnitude_history.values():
            for d in deques:
                d.clear()
        self._magnitude_history.clear()
        super().on_stop()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        """Collect per-symbol signals from each sub-strategy, then combine."""
        per_sub_signals: list[list[Signal]] = []
        for i, sub in enumerate(self._sub_strategies):
            try:
                sigs = sub.on_bar(bars, mtf_context=mtf_context)
            except Exception:
                self._log.warning(
                    "ensemble_strategy.sub_on_bar_failed",
                    sub_id=sub.strategy_id,
                    sub_index=i,
                )
                sigs = []
            per_sub_signals.append(list(sigs))

        # Determine the set of symbols any sub emitted for this bar.
        symbols: set[str] = set()
        for sigs in per_sub_signals:
            for s in sigs:
                symbols.add(s.symbol)
        # CR-001 (S47-4): early-return BEFORE the history-update loop so
        # silent bars (no sub emitted anything) skip both the loop and
        # the combination phase, leaving history untouched.
        if not symbols:
            return []

        # S47-4: update magnitude history per (symbol, sub).  For every
        # symbol any sub emitted for, record each sub's |target_position|
        # for THAT symbol (0.0 when the sub did not emit for it).
        for symbol in symbols:
            self._ensure_history_for_symbol(symbol)
            for i, sigs in enumerate(per_sub_signals):
                sig = self._first_signal_for_symbol(sigs, symbol)
                mag = float(abs(sig.target_position)) if sig is not None else 0.0
                self._magnitude_history[symbol][i].append(mag)

        combined: list[Signal] = []
        for symbol in sorted(symbols):
            sig = self._combine_per_symbol(symbol, per_sub_signals)
            if sig is not None and sig.direction is not SignalDirection.HOLD:
                combined.append(sig)
        return combined

    # ------------------------------------------------------------------
    # Combination helpers
    # ------------------------------------------------------------------

    def _combine_per_symbol(
        self,
        symbol: str,
        per_sub_signals: list[list[Signal]],
    ) -> Signal | None:
        """Combine all sub-strategy signals for one symbol."""
        sub_signals_for_symbol: list[Signal | None] = [
            self._first_signal_for_symbol(sigs, symbol) for sigs in per_sub_signals
        ]
        weights = self._compute_weights(symbol)

        if self._combination_method == "majority_vote":
            return self._combine_majority(symbol, sub_signals_for_symbol, weights)
        return self._combine_weighted(symbol, sub_signals_for_symbol, weights)

    @staticmethod
    def _first_signal_for_symbol(
        signals: list[Signal], symbol: str,
    ) -> Signal | None:
        """Return the first signal for the given symbol, or None."""
        for s in signals:
            if s.symbol == symbol:
                return s
        return None

    def _compute_weights(self, symbol: str) -> list[float]:
        """Compute per-sub weights for combination of the given symbol.

        Both ``inverse_vol`` and ``majority_vote`` combination methods
        receive PER-SYMBOL weights since S47-4 (Sprint 47).

        - Equal-weight: 1/N for every sub regardless of symbol.
        - Inverse-vol / majority-vote base (per-symbol since S47-4):
            * Symbol not yet seen (no history): uniform weight 1.0/N.
            * Warmup (< _MIN_VOL_LOOKBACK history samples for this symbol):
              uniform weight 1.0 per sub.
            * Sub silent for this symbol (all history magnitudes == 0):
              weight 0.0 (CR-002 -- silent sub does NOT receive 1/eps).
            * Otherwise: 1 / max(pstdev, 1e-9).
          All weights are normalised to sum to 1.0.  If the total collapses
          to zero (e.g. ALL subs silent for this symbol) the function falls
          back to uniform weights to avoid a divide-by-zero downstream.
        """
        n = len(self._sub_strategies)
        if n == 0:
            return []

        if self._combination_method == "equal_weight":
            return [1.0 / n] * n

        # ``None`` return from .get() means the symbol has never been seen
        # by on_bar -- the uniform-weight fallback is the deliberate
        # warmup semantic (CR-004 informational).
        histories = self._magnitude_history.get(symbol)
        if histories is None:
            return [1.0 / n] * n

        weights: list[float] = []
        for hist in histories:
            if len(hist) < _MIN_VOL_LOOKBACK:
                weights.append(1.0)
                continue
            hist_list = list(hist)
            if all(v == 0.0 for v in hist_list):
                weights.append(0.0)
                continue
            try:
                vol = statistics.pstdev(hist_list)
            except statistics.StatisticsError:
                vol = 0.0
            weights.append(1.0 / max(vol, 1e-9))

        total = sum(weights)
        if total <= 0.0 or math.isnan(total):
            return [1.0 / n] * n
        return [w / total for w in weights]

    def _combine_weighted(
        self,
        symbol: str,
        sub_signals: list[Signal | None],
        weights: list[float],
    ) -> Signal | None:
        """Inverse-vol / equal-weight combination.

        Combined direction:
            signed_sum = sum(weight * sign(direction) * |target_position|)
            > 0 -> BUY,  < 0 -> SELL,  == 0 (or no contributors) -> None.

        Combined target_position:
            contribution-weighted average of |target_position| over
            sub-strategies that emitted a non-HOLD signal.  Always >= 0
            (matches the Signal ``ge=Decimal(0)`` constraint -- direction
            is carried by the ``direction`` field, not by sign).

        Combined confidence:
            contribution-weighted average of sub-strategy confidences.
        """
        signed_sum = 0.0
        abs_sum = 0.0
        conf_sum = 0.0
        contrib_weight = 0.0

        for sig, w in zip(sub_signals, weights):
            if sig is None or sig.direction is SignalDirection.HOLD:
                continue
            sign = 1.0 if sig.direction is SignalDirection.BUY else -1.0
            mag = float(abs(sig.target_position))
            signed_sum += w * sign * mag
            abs_sum += w * mag
            conf_sum += w * sig.confidence
            contrib_weight += w

        if contrib_weight <= 0.0 or abs_sum <= 0.0:
            return None

        if signed_sum > 0.0:
            direction = SignalDirection.BUY
        elif signed_sum < 0.0:
            direction = SignalDirection.SELL
        else:
            return None

        target_magnitude = abs_sum / contrib_weight
        confidence = max(0.0, min(1.0, conf_sum / contrib_weight))

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=direction,
            target_position=Decimal(str(target_magnitude)),
            confidence=confidence,
        )

    def _combine_majority(
        self,
        symbol: str,
        sub_signals: list[Signal | None],
        weights: list[float],
    ) -> Signal | None:
        """Weighted majority vote on direction.

        Tie-breaking: BUY wins when ``buy_w == sell_w``.  At the minimum
        allowed ``min_agreement=0.5`` this directional bias activates; for
        any ``min_agreement > 0.5`` a perfectly tied vote never reaches
        the share threshold.
        """
        buy_w = 0.0
        sell_w = 0.0
        hold_w = 0.0
        for sig, w in zip(sub_signals, weights):
            if sig is None or sig.direction is SignalDirection.HOLD:
                hold_w += w
            elif sig.direction is SignalDirection.BUY:
                buy_w += w
            else:
                sell_w += w

        total = buy_w + sell_w + hold_w
        if total <= 0.0:
            return None

        if buy_w >= sell_w:
            winner_w = buy_w
            winner = SignalDirection.BUY
        else:
            winner_w = sell_w
            winner = SignalDirection.SELL

        share = winner_w / total
        if share < self._min_agreement:
            return None

        abs_sum = 0.0
        conf_sum = 0.0
        contrib_weight = 0.0
        for sig, w in zip(sub_signals, weights):
            if sig is None or sig.direction is not winner:
                continue
            mag = float(abs(sig.target_position))
            abs_sum += w * mag
            conf_sum += w * sig.confidence
            contrib_weight += w

        if contrib_weight <= 0.0 or abs_sum <= 0.0:
            return None

        target_magnitude = abs_sum / contrib_weight
        confidence = max(0.0, min(1.0, conf_sum / contrib_weight))

        return Signal(
            strategy_id=self._strategy_id,
            symbol=symbol,
            direction=winner,
            target_position=Decimal(str(target_magnitude)),
            confidence=confidence,
        )
