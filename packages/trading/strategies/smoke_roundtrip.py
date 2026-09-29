"""
packages/trading/strategies/smoke_roundtrip.py
------------------------------------------------
SmokeRoundtripStrategy -- WP-SMOKE (VP2 herstart-protocol step 2, D2).

A deliberately trivial, deterministic mechanics-test strategy: BUY on the
first bar it ever sees for a symbol, hold for ``hold_bars`` further bars,
then SELL ``target_position=0`` on every bar of a bounded ``exit_retry_bars``
window. No indicator, no wall clock, no randomness -- backtests are exactly
reproducible and paper/live runs are driven purely by an in-memory call
counter per symbol.

This strategy carries **no trading edge**. It exists only to exercise the
BUY -> fill -> SELL -> fill -> flat round trip mechanics against a real
exchange (Coinbase) with real money, at minimal notional (~EUR 9), per
``reports/vp2-smoke/synthesis-spec.md`` (WP-SMOKE, D-SMK-1..4).

Binding spec: ``reports/vp2-smoke/synthesis-spec.md`` section 6.

Invariants (spec section 6)
----------------------------
- At most one BUY per instance: ``_entry_emitted`` (per symbol) never
  resets, even after the exit window closes (state DONE is terminal).
- No wall clock, no randomness -- deterministic in backtest; the BUY
  lands on the very first ``on_bar`` call this instance ever receives for
  a symbol (in backtest, that is the first post-warmup bar).
- A fresh instance (e.g. after a protective resume, which constructs a new
  strategy object) restarts at IDLE and replays IDLE -> HOLD -> EXIT. Its
  BUY is dropped by the engine's protective-mode entry filter
  (``run_orchestrator``/``StrategyEngine``, unchanged); its SELLs sell the
  rebuilt position from the ledger, or no-op if already flat. This is the
  intended salvage path (C-2 of the synthesis spec) -- this strategy makes
  no attempt to detect "did my BUY actually fill" on its own; it has no
  ledger access, by design (BaseStrategy strategies are stateless w.r.t.
  order placement).

The mandatory fixed stop-loss (``bracket_stop_loss_pct``, engine-level,
popped by the API layer's ``_extract_bracket_config`` before this
strategy's own parameter schema is validated) is the net beneath this
bounded SELL window -- see ``requires_exit_manager = True`` below.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import SignalDirection
from trading.models import Signal
from trading.strategy import BaseStrategy, StrategyMetadata

__all__ = ["SmokeRoundtripStrategy"]


class _SmokeRoundtripParams(BaseModel):
    """Pydantic parameter schema for the smoke round-trip strategy.

    ``extra="forbid"`` (SY-13a-02 style strictness) so the JSON schema
    emits ``additionalProperties: false`` and any unrecognised key is a
    hard rejection at construction time (third validation layer, behind
    the API's own ``smoke_guard.validate_smoke_run`` G-6 and
    ``_validate_params_against_schema``).

    Every numeric bound uses ``ge``/``le`` only, never ``gt``/``lt`` --
    Pydantic v2 renders ``gt``/``lt`` as JSON Schema ``exclusiveMinimum``/
    ``exclusiveMaximum``, which the API's schema validator does not
    understand (SMK-A-14 / synthesis spec section 6).

    Engine-level bracket keys (``bracket_stop_loss_pct`` etc.) are
    deliberately NOT declared here: ``_extract_bracket_config`` (API layer)
    pops them out of ``strategy_params`` before this model ever sees them,
    exactly like every other strategy in this codebase.
    """

    model_config = ConfigDict(extra="forbid")

    notional_quote: float = Field(
        default=9.00,
        ge=5.00,
        le=9.50,
        description="BUY target_position in quote currency (EUR for the live smoke run).",
    )
    hold_bars: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Processed bars after the entry bar before the first SELL attempt.",
    )
    exit_retry_bars: int = Field(
        default=4,
        ge=1,
        le=5,
        description="Number of consecutive bars on which SELL target_position=0 is emitted.",
    )


class _SymbolState:
    """Per-symbol, in-memory-only state (no wall clock, no randomness).

    ``n`` counts ``on_bar`` calls this instance has processed for the
    symbol (0-based). ``entry_bar_n`` is the value of ``n`` at the bar the
    single BUY was emitted. ``entry_emitted`` is the latch that never
    resets. ``window_closed_logged`` guards the single terminal WARNING.
    """

    __slots__ = ("entry_bar_n", "entry_emitted", "n", "window_closed_logged")

    def __init__(self) -> None:
        self.n: int = -1
        self.entry_bar_n: int | None = None
        self.entry_emitted: bool = False
        self.window_closed_logged: bool = False


class SmokeRoundtripStrategy(BaseStrategy):
    """
    Diagnostic mechanics-test strategy: one BUY, hold, then a bounded SELL
    window. See the module docstring and
    ``reports/vp2-smoke/synthesis-spec.md`` section 6 for the full
    state-machine table and rationale.
    """

    metadata: ClassVar[StrategyMetadata] = StrategyMetadata(
        name="Smoke round-trip (diagnostic)",
        version="1.0.0",
        description=(
            "Mechanics test, no edge: BUY on the first bar, hold, SELL in a "
            "bounded window. Verifies the full order round trip against a "
            "real exchange at minimal notional (WP-SMOKE, VP2 D2)."
        ),
        author="python-backend-specialist",
        tags=["diagnostic", "smoke-test", "mechanics", "no-edge"],
    )

    # A-2 (synthesis spec section 3): explicit declarations, test-enforced
    # (ST-27 in test_wp13a_exit_config.py).
    requires_exit_manager: ClassVar[bool] = True
    default_allow_pyramiding: ClassVar[bool] = False

    def __init__(self, strategy_id: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(strategy_id, params)
        self._state: dict[str, _SymbolState] = {}

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return _SmokeRoundtripParams(**params).model_dump()

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return _SmokeRoundtripParams.model_json_schema()

    @property
    def min_bars_required(self) -> int:
        # No indicator warm-up at all -- the strategy acts on the very
        # first bar it is ever shown.
        return 1

    def on_bar(
        self,
        bars: Sequence[OHLCVBar],
        *,
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        if not bars:
            return []

        symbol = bars[-1].symbol
        state = self._state.setdefault(symbol, _SymbolState())
        state.n += 1
        n = state.n

        # IDLE -> HOLD: the very first on_bar call this instance has ever
        # processed for this symbol emits exactly one BUY. The latch
        # (`entry_emitted`) never resets, even across a long-running
        # instance that keeps receiving bars after DONE (SMK-T-01).
        if not state.entry_emitted:
            state.entry_emitted = True
            state.entry_bar_n = n
            notional_quote = Decimal(str(self._params["notional_quote"]))
            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.BUY,
                target_position=notional_quote,
                confidence=1.0,
                metadata={"trigger": "smoke_roundtrip", "phase": "entry"},
            )
            self._log.info(
                "smoke_roundtrip.entry",
                symbol=symbol,
                notional_quote=str(notional_quote),
                run_id=self._run_id,
            )
            return [signal]

        # Every call after the entry bar is measured relative to it.
        assert state.entry_bar_n is not None  # set unconditionally above
        offset = n - state.entry_bar_n
        hold_bars: int = self._params["hold_bars"]
        exit_retry_bars: int = self._params["exit_retry_bars"]

        # HOLD: nothing emitted while the position is meant to stay open.
        if offset < hold_bars:
            return []

        # EXIT: a bounded SELL target_position=0 window. Covers a
        # rejected first SELL, a late-routed BUY fill, and an ambiguous
        # submit (A-6 / C-2 of the synthesis spec) -- re-emitting a
        # target_position=0 SELL is always safe (sell_no_position /
        # sell_inflight_pending / sell_capped_to_zero at the engine layer).
        if offset < hold_bars + exit_retry_bars:
            attempt = offset - hold_bars + 1
            signal = Signal(
                strategy_id=self._strategy_id,
                symbol=symbol,
                direction=SignalDirection.SELL,
                target_position=Decimal("0"),
                confidence=1.0,
                metadata={
                    "trigger": "smoke_roundtrip",
                    "phase": "exit",
                    "exit_reason": "signal_exit",
                    "attempt": attempt,
                },
            )
            self._log.info(
                "smoke_roundtrip.exit_attempt",
                symbol=symbol,
                attempt=attempt,
                run_id=self._run_id,
            )
            return [signal]

        # DONE: terminal. The strategy has no ledger access, so it cannot
        # know whether it actually sold -- this is a WARNING, not
        # CRITICAL, so a successful run never raises an alarm (deliberate
        # deviation from the risk design's original CRITICAL proposal;
        # flatness is verified by the operator, synthesis spec section 4
        # C-2 / section 10 P8). Logged once per symbol per instance.
        if not state.window_closed_logged:
            state.window_closed_logged = True
            self._log.warning(
                "smoke_roundtrip.exit_window_closed",
                symbol=symbol,
                run_id=self._run_id,
            )
        return []
