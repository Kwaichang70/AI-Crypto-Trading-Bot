"""
tests/integration/fakes/scripted_strategy.py
-----------------------------------------------
Minimal deterministic strategy for the WP1.0 live protective-path harness.

Emits exactly one BUY or SELL signal on a caller-chosen ``on_bar`` call
index (0-based, counted from the first post-warmup bar) and HOLDs on every
other call. This gives the test full, explicit control over *when* the
engine's own signal path opens or attempts to close a position, so every
scenario's price path (stop-loss / take-profit / trailing-stop breach) can
be scripted deterministically on top of it via
``fake_ccxt_exchange.push_bar``.

This is intentionally not one of the eight production strategies under
``packages/trading/strategies/`` -- none of them expose a "buy on command"
hook, and reusing one would make the scenario price paths depend on
indicator warm-up math instead of the exact bar the test wants to trigger
on. Momentum_breakout's *bracket/trailing-stop engine config* (the thing
this WP actually needs to be faithful to) is reused as-is via
``StrategyEngine``'s ``bracket_*`` / ``trailing_stop_pct`` config keys in
the test module -- only the entry-signal strategy itself is a stand-in.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["ScriptedSignalStrategy"]


class ScriptedSignalStrategy(BaseStrategy):
    """Emit one scripted BUY or SELL signal, HOLD otherwise.

    Params
    ------
    direction:
        ``"buy"`` or ``"sell"``.
    call_index:
        0-based index (into the sequence of post-warmup ``on_bar`` calls)
        on which to emit the signal.
    target_notional:
        ``Signal.target_position`` value, as a string (e.g. ``"100"``).
        Ignored (fixed at ``"0"``, i.e. full-close) when ``direction`` is
        ``"sell"``, matching how the real bracket/trailing managers emit
        their exit signals.
    """

    metadata = StrategyMetadata(
        name="scripted_signal_test_strategy",
        description="WP1.0 harness-only strategy: one scripted BUY/SELL, HOLD otherwise.",
        tags=["test-only"],
    )

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._direction = SignalDirection(self._params.get("direction", "buy"))
        self._call_index = int(self._params.get("call_index", 0))
        self._target_notional = Decimal(str(self._params.get("target_notional", "100")))
        self._call_count = -1
        self._fired = False

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        self._call_count += 1
        if self._fired or not bars or self._call_count != self._call_index:
            return []

        self._fired = True
        symbol = bars[-1].symbol
        target = Decimal("0") if self._direction == SignalDirection.SELL else self._target_notional
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=self._direction,
                target_position=target,
                confidence=1.0,
            )
        ]
