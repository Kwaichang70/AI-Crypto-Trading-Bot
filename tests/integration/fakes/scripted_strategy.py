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
from typing import Any, Literal

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["ScriptedSequenceStrategy", "ScriptedSignalStrategy"]


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
    # WP1.3a (SY-13a-06): test-only double, no bracket/trailing configured.
    requires_exit_manager = False

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


class ScriptedSequenceStrategy(BaseStrategy):
    """Emit a scripted BUY/SELL on several ``on_bar`` call indices (WP1.5a).

    ``ScriptedSignalStrategy`` can fire only once per instance; the daily-PnL
    scenarios need a sequence (BUY, a rejected BUY later the same UTC day, a
    BUY on the next day, an exit ...).

    Params
    ------
    script:
        ``dict[int, Literal["buy", "sell"]]`` -- 0-based index into the
        sequence of post-warmup ``on_bar`` calls -> direction.  Every other
        call HOLDs.  (JSON-ish string keys are accepted and coerced.)
    target_notional:
        ``Signal.target_position`` for BUY entries, as a string.  A SELL
        always carries ``target_position = 0`` (full close).
    targets:
        Optional ``dict[int, str]`` overriding ``target_notional`` for the
        BUY at a given call index (e.g. a small probe BUY after a big one).
    """

    metadata = StrategyMetadata(
        name="scripted_sequence_test_strategy",
        description="WP1.5a harness-only strategy: a scripted BUY/SELL sequence.",
        tags=["test-only"],
    )
    # WP1.3a (SY-13a-06): test-only double; the tests configure brackets
    # explicitly where they want an engine-level exit.
    requires_exit_manager = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        raw = self._params.get("script") or {}
        self._script: dict[int, Literal["buy", "sell"]] = {int(k): v for k, v in raw.items()}
        self._target_notional = Decimal(str(self._params.get("target_notional", "100")))
        self._targets: dict[int, Decimal] = {
            int(k): Decimal(str(v)) for k, v in (self._params.get("targets") or {}).items()
        }
        self._call_count = -1

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
        action = self._script.get(self._call_count)
        if action is None or not bars:
            return []
        is_sell = action == "sell"
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=bars[-1].symbol,
                direction=SignalDirection.SELL if is_sell else SignalDirection.BUY,
                target_position=(
                    Decimal("0")
                    if is_sell
                    else self._targets.get(self._call_count, self._target_notional)
                ),
                confidence=1.0,
            )
        ]
