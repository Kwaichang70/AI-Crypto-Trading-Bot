"""
tests/unit/test_ensemble_strategy.py
-------------------------------------
QT-008 (Sprint 46) -- EnsembleStrategy MVP tests.
"""
from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection, TimeFrame
from trading.models import Signal
from trading.strategies.ensemble import EnsembleStrategy
from trading.strategy import BaseStrategy, StrategyMetadata


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


class _StubStrategy(BaseStrategy):
    """A scripted strategy that emits a fixed signal (or none) per on_bar."""

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Stub", version="0.0", description="test", author="t",
    )

    def __init__(
        self,
        strategy_id: str,
        signals_per_bar: Signal | list[Signal] | None = None,
        *,
        min_bars: int = 1,
        htf: list[str] | None = None,
        raise_on_bar: bool = False,
        raise_on_stop: bool = False,
    ) -> None:
        super().__init__(strategy_id=strategy_id, params={})
        self._signal_template = signals_per_bar
        self._min_bars = min_bars
        self._htf = htf or []
        self._raise_on_bar = raise_on_bar
        self._raise_on_stop = raise_on_stop
        self.on_start_called = False
        self.on_stop_called = False

    @property
    def min_bars_required(self) -> int:
        return self._min_bars

    @property
    def htf_timeframes(self) -> list[str]:
        return list(self._htf)

    def on_start(self, run_id: str) -> None:
        super().on_start(run_id)
        self.on_start_called = True

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        if self._raise_on_bar:
            raise RuntimeError("stub.on_bar boom")
        if self._signal_template is None:
            return []
        if isinstance(self._signal_template, list):
            return list(self._signal_template)
        return [self._signal_template]

    def on_stop(self) -> None:
        if self._raise_on_stop:
            raise RuntimeError("stub.on_stop boom")
        self.on_stop_called = True
        super().on_stop()


def _bar(symbol: str = "BTC/USDT") -> OHLCVBar:
    return OHLCVBar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_HOUR,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1000"),
    )


def _signal(
    symbol: str,
    direction: SignalDirection,
    target: float,
    confidence: float = 0.7,
    strategy_id: str = "stub",
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        target_position=Decimal(str(target)),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestEnsembleConstructorValidation:

    def test_empty_sub_strategies_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EnsembleStrategy(strategy_id="e", sub_strategies=[])

    def test_unknown_combination_method_raises(self) -> None:
        with pytest.raises(ValueError, match="combination_method"):
            EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                combination_method="nope",
            )

    def test_vol_lookback_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="vol_lookback"):
            EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                vol_lookback=1,
            )

    def test_vol_lookback_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match="vol_lookback"):
            EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                vol_lookback=10_000,
            )

    def test_vol_lookback_boundary_values_accepted(self) -> None:
        for vlb in (3, 200):
            e = EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                vol_lookback=vlb,
            )
            assert e._vol_lookback == vlb

    def test_min_agreement_below_0_5_raises(self) -> None:
        with pytest.raises(ValueError, match="min_agreement"):
            EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                min_agreement=0.4,
            )

    def test_min_agreement_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="min_agreement"):
            EnsembleStrategy(
                strategy_id="e",
                sub_strategies=[_StubStrategy("a")],
                min_agreement=1.1,
            )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestEnsembleProperties:

    def test_min_bars_required_is_max_over_subs(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", min_bars=50),
                _StubStrategy("b", min_bars=100),
                _StubStrategy("c", min_bars=20),
            ],
        )
        assert e.min_bars_required == 100

    def test_htf_timeframes_is_union(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", htf=["4h"]),
                _StubStrategy("b", htf=["1d", "4h"]),
                _StubStrategy("c", htf=[]),
            ],
        )
        assert e.htf_timeframes == ["4h", "1d"]

    def test_sub_strategies_property_returns_tuple(self) -> None:
        subs = [_StubStrategy("a"), _StubStrategy("b")]
        e = EnsembleStrategy(strategy_id="e", sub_strategies=subs)
        assert isinstance(e.sub_strategies, tuple)
        assert len(e.sub_strategies) == 2


# ---------------------------------------------------------------------------
# Equal-weight combination
# ---------------------------------------------------------------------------


class TestEqualWeightCombination:

    def test_all_buy_emits_buy_with_avg_target(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.8)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.BUY, 200, 0.6)),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        sig = out[0]
        assert sig.direction is SignalDirection.BUY
        assert float(sig.target_position) == pytest.approx(150.0, abs=1e-9)
        assert sig.confidence == pytest.approx(0.7, abs=1e-9)
        assert sig.strategy_id == "e"

    def test_all_sell_emits_sell_with_positive_target_position(self) -> None:
        """CR-001 regression: SELL combined signal must have non-negative
        ``target_position`` (direction is carried by ``direction`` field)."""
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.SELL, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 200, 0.5)),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.SELL
        assert float(out[0].target_position) == pytest.approx(150.0, abs=1e-9)
        assert out[0].target_position >= Decimal(0)   # invariant

    def test_mixed_directions_buy_dominant(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 50, 0.5)),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        # Signed sum = +25 -> BUY.  |target| avg = 75.
        assert len(out) == 1
        assert out[0].direction is SignalDirection.BUY
        assert float(out[0].target_position) == pytest.approx(75.0, abs=1e-9)

    def test_mixed_directions_sell_dominant(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 50, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 100, 0.5)),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        # Signed sum = -25 -> SELL.  |target| avg = 75.
        assert len(out) == 1
        assert out[0].direction is SignalDirection.SELL
        assert float(out[0].target_position) == pytest.approx(75.0, abs=1e-9)

    def test_canceling_signals_emit_nothing(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 100, 0.5)),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        assert out == []

    def test_only_one_sub_emits_passes_through(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", None),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.BUY


# ---------------------------------------------------------------------------
# Inverse-vol combination
# ---------------------------------------------------------------------------


class TestInverseVolCombination:

    def test_warmup_uses_uniform_weights(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.BUY, 200, 0.5)),
            ],
            combination_method="inverse_vol",
            vol_lookback=5,
        )
        out = e.on_bar([_bar()])
        assert float(out[0].target_position) == pytest.approx(150.0, abs=1e-9)

    def test_consistent_sub_has_lower_vol_after_warmup(self) -> None:
        sub_a = _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5))
        sub_b = _StubStrategy("b", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5))
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[sub_a, sub_b],
            combination_method="inverse_vol",
            vol_lookback=5,
        )
        for i in range(5):
            sub_b._signal_template = _signal(
                "BTC/USDT", SignalDirection.BUY, 100 if i % 2 == 0 else 500, 0.5,
            )
            e.on_bar([_bar()])
        std_a = statistics.pstdev(e._magnitude_history[0])
        std_b = statistics.pstdev(e._magnitude_history[1])
        assert std_b > std_a

    def test_silent_sub_gets_zero_weight(self) -> None:
        """CR-002 remediation: a constantly-silent sub-strategy must NOT
        dominate inverse-vol weighting via 1/eps."""
        sub_a = _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5))
        sub_b = _StubStrategy("b", None)   # always silent
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[sub_a, sub_b],
            combination_method="inverse_vol",
            vol_lookback=3,
        )
        # Drive 4 bars: sub A always emits BUY 100; sub B never emits.
        for _ in range(4):
            e.on_bar([_bar()])
        # Internal weight inspection.
        weights = e._compute_weights()
        # Sub B should have weight 0; sub A should hold all the weight.
        assert weights[1] == 0.0
        assert weights[0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Majority vote combination
# ---------------------------------------------------------------------------


class TestMajorityVoteCombination:

    def test_two_of_three_buy_meets_default_60pct(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.BUY, 200, 0.5)),
                _StubStrategy("c", _signal("BTC/USDT", SignalDirection.SELL, 50, 0.5)),
            ],
            combination_method="majority_vote",
            min_agreement=0.6,
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.BUY
        assert float(out[0].target_position) == pytest.approx(150.0, abs=1e-9)

    def test_below_min_agreement_emits_no_signal(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 100, 0.5)),
                _StubStrategy("c", None),
            ],
            combination_method="majority_vote",
            min_agreement=0.6,
        )
        out = e.on_bar([_bar()])
        assert out == []

    def test_two_of_three_sell_emits_sell(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.SELL, 100, 0.5)),
                _StubStrategy("b", _signal("BTC/USDT", SignalDirection.SELL, 200, 0.5)),
                _StubStrategy("c", _signal("BTC/USDT", SignalDirection.BUY, 50, 0.5)),
            ],
            combination_method="majority_vote",
            min_agreement=0.6,
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.SELL
        assert out[0].target_position >= Decimal(0)
        assert float(out[0].target_position) == pytest.approx(150.0, abs=1e-9)

    def test_single_voter_with_50pct_threshold_emits(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy("a", _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5)),
                _StubStrategy("b", None),
            ],
            combination_method="majority_vote",
            min_agreement=0.5,
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.BUY


# ---------------------------------------------------------------------------
# Lifecycle + robustness
# ---------------------------------------------------------------------------


class TestEnsembleLifecycle:

    def test_on_start_propagates(self) -> None:
        subs = [_StubStrategy("a"), _StubStrategy("b")]
        e = EnsembleStrategy(strategy_id="e", sub_strategies=subs)
        e.on_start(run_id="r")
        assert all(s.on_start_called for s in subs)

    def test_on_stop_propagates(self) -> None:
        subs = [_StubStrategy("a"), _StubStrategy("b")]
        e = EnsembleStrategy(strategy_id="e", sub_strategies=subs)
        e.on_stop()
        assert all(s.on_stop_called for s in subs)

    def test_on_stop_swallows_sub_exception(self) -> None:
        subs = [
            _StubStrategy("a"),
            _StubStrategy("b", raise_on_stop=True),
            _StubStrategy("c"),
        ]
        e = EnsembleStrategy(strategy_id="e", sub_strategies=subs)
        e.on_stop()
        assert subs[0].on_stop_called is True
        assert subs[1].on_stop_called is False
        assert subs[2].on_stop_called is True

    def test_sub_on_bar_exception_treated_as_no_signal(self) -> None:
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[
                _StubStrategy(
                    "a",
                    _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5),
                ),
                _StubStrategy("b", raise_on_bar=True),
            ],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar()])
        assert len(out) == 1
        assert out[0].direction is SignalDirection.BUY
        assert float(out[0].target_position) == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Multi-symbol
# ---------------------------------------------------------------------------


class TestMultiSymbolHandling:

    def test_emits_one_signal_per_symbol(self) -> None:
        sigs = [
            _signal("BTC/USDT", SignalDirection.BUY, 100, 0.5),
            _signal("ETH/USDT", SignalDirection.SELL, 50, 0.5),
        ]
        e = EnsembleStrategy(
            strategy_id="e",
            sub_strategies=[_StubStrategy("a", sigs)],
            combination_method="equal_weight",
        )
        out = e.on_bar([_bar("BTC/USDT")])
        symbols = sorted(s.symbol for s in out)
        assert symbols == ["BTC/USDT", "ETH/USDT"]
