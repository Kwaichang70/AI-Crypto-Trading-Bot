"""
packages/trading/strategy_engine.py
-------------------------------------
StrategyEngine -- central orchestrator that ties together all trading core
components for a complete trading run.

Responsibilities
----------------
1. **Run lifecycle**: ``start(run_id)`` -> bar loop -> ``stop()``
2. **Bar-by-bar processing**: fetch candles -> feed to strategies -> collect
   signals -> execute via engine -> route fills to portfolio & risk
3. **Component wiring**: holds references to all core components and mediates
   their interactions
4. **Multi-strategy support**: runs multiple strategies in parallel, each
   producing independent signals
5. **Fill routing**: routes fills from the execution engine to
   PortfolioAccounting and RiskManager

Run modes
---------
- BACKTEST: walk through pre-fetched historical bars deterministically
- PAPER: poll market data service for new bars on interval
- LIVE: same polling loop as PAPER but with real order placement

Safety invariants
-----------------
- In every mode, the risk manager kill-switch blocks new *entries* only
  (D3, WP1.2): BUY signals are dropped after the strategy loop via
  ``_drop_entry_signals``; strategy SELLs, bracket exits (5a), trailing
  stops (5b) and resting-order fills (6) all keep running while it is
  active. The side-aware ``DefaultRiskManager`` gate is what actually
  enforces this; the engine-level filter only avoids a pointless
  downstream attempt for BUYs and keeps the skip audit populated.
- The circuit breaker (WP1.6) is evaluated once per processed bar from a typed
  ``RiskSnapshot``.  HALT / DAILY_LIMIT drop BUYs only (C23, same
  ``_drop_entry_signals`` site as the kill switch); REDUCE halves the final
  BUY quantity via ``Signal.entry_size_multiplier``.  Exits are never blocked
  and the run is NEVER stopped by the breaker (D4); a breaker failure drops
  that bar's BUYs (fail-closed, entries only).
- Strategy exceptions are caught and logged without crashing the bar loop
- The bar loop never crashes -- all errors are logged and processing continues
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum, auto
from typing import Any, Literal
from uuid import UUID

import pandas as pd
import structlog

from common.models import MultiTimeframeContext, OHLCVBar
from common.types import OrderSide, OrderStatus, RunMode, SignalDirection, TimeFrame
from data.indicators import atr as _atr_series
from data.market_data import BaseMarketDataService, MarketDataError
from trading.bracket_exit import (
    AnchorSource,
    BracketAnchor,
    BracketExitManager,
    LotEntry,
    anchor_bar_timestamp,
    lot_entry_from_opening_fills,
    open_lot_fills,
    window_is_contiguous,
)
from trading.execution import BaseExecutionEngine
from trading.exit_config import (
    ExitCostModel,
    parse_exit_config,
    require_exit_manager,
    resolve_allow_pyramiding,
)
from trading.models import Fill, Order, Position, Signal, TradeResult
from trading.portfolio import PortfolioAccounting
from trading.risk import BaseRiskManager
from trading.safety import CircuitBreaker, CircuitBreakerDecision, CircuitBreakerResponse
from trading.strategy import BaseStrategy
from trading.trade_journal import ExitReasonDetector, TradeExcursionTracker, TradeSkipLogger
from trading.trailing_stop import TrailingStopManager

__all__ = [
    "EngineState",
    "FlattenPreconditionError",
    "FlattenResult",
    "FlattenSymbolResult",
    "StrategyEngine",
    "atr_for_bars",
]

logger = structlog.get_logger(__name__)


# Numeric encoding of the graduated circuit-breaker response for the
# ``circuit_breaker_state`` gauge (Grafana alerts fire on value >= 3 = HALT).
_CB_STATE_VALUES: dict[CircuitBreakerResponse, int] = {
    CircuitBreakerResponse.OK: 0,
    CircuitBreakerResponse.REDUCE: 1,
    CircuitBreakerResponse.DAILY_LIMIT: 2,
    CircuitBreakerResponse.HALT: 3,
}

# CR-005: ATR window for bracket exits — bars older than this many periods
# have < e^-6 Wilder weight, so the bounded window is numerically equivalent
# to full history while keeping per-bar ATR cost O(period) instead of O(n).
_ATR_WINDOW_MULTIPLIER = 6

#: WP1.3b-a: per-call timeout (seconds) for the bounded exit-state OHLCV
#: fetches (anchor finalisation, resume recompute, trailing gap fill).  No retry.
_EXIT_FETCH_TIMEOUT_S: float = 10.0
#: WP1.3b-a (FX-01): ccxt's Coinbase ``fetch_ohlcv`` clamps ``limit`` to 300 bars.
#: Documentation plus the invariant ``POLICY_ATR_PERIOD_MAX * 6 + 1 <= 300``
#: (pinned by a test); there is no runtime clamp -- the single WP1.3a validator
#: refuses an ATR period whose window ``W = 6*period+1`` would not fit one call.
_EXCHANGE_OHLCV_MAX_LIMIT: int = 300
#: WP1.3b-a (FX-03): a pending (provisional / deferred / fallback) anchor gets at
#: most this many exact-window fetch attempts, at most one per bar pass.
_ANCHOR_FETCH_MAX_ATTEMPTS: int = 3
#: WP1.3b-a (FX-03, SEC-06): engine-wide budget of finalisation fetches per
#: ``_finalize_pending_anchors`` call (i.e. per bar pass, under the bar lock).
_ANCHOR_FETCHES_PER_PASS: int = 1
#: WP1.3b-a (FX-06): ``bracket_exit.lot_unprotected`` is logged on the first bar
#: of an unprotected streak and then every this-many bars.
_UNPROTECTED_LOG_EVERY: int = 10
#: WP1.3b-a: at most this many missing bars are fetched to rebuild the trailing
#: high-water mark after a resume (beyond that: ``trailing_stop.seed_partial``).
_TRAILING_GAP_FETCH_LIMIT: int = 1000
#: WP1.3b-a (SY-13b-04): an order in one of these states with nothing filled
#: means the exit was NOT submitted (R-04); any other state counts as accepted.
_TERMINAL_UNFILLED_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED}
)


def atr_for_bars(bars: Sequence[OHLCVBar], period: int) -> Decimal | None:
    """
    Latest ATR (price units) of ``bars``; the module-level extraction of
    ``StrategyEngine._compute_atr_for_symbol``'s body (WP1.3b-a, unchanged
    return value).

    Returns None when there are too few bars for the configured ATR period
    (Wilder warm-up) or the ATR is not positive.  CR-005: only the last
    ``period * 6 + 1`` bars are used -- Wilder smoothing decays by
    ``(1 - 1/period)`` per bar, so older bars contribute < e^-6 (~0.25%).
    """
    if not bars or len(bars) <= period:
        return None
    window = period * _ATR_WINDOW_MULTIPLIER + 1
    if len(bars) > window:
        bars = bars[-window:]
    highs = pd.Series([float(b.high) for b in bars])
    lows = pd.Series([float(b.low) for b in bars])
    closes = pd.Series([float(b.close) for b in bars])
    atr_series = _atr_series(highs, lows, closes, period=period)
    last = atr_series.iloc[-1]
    if pd.isna(last) or last <= 0:
        return None
    return Decimal(str(last))


def _opt_float(value: Any) -> float | None:
    """Coerce a config value to float, mapping None / "" to None.

    Used when building optional engine managers from the run config, where
    a blank UI field may arrive as an empty string rather than absent.
    """
    if value is None or value == "":
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
#: Multiplier applied to each strategy's ``min_bars_required`` when deriving
#: the auto-warmup window.  Doubling gives strategies enough history to fill
#: rolling indicators (RSI/ATR/etc.) before signals start flowing.
_DEFAULT_WARMUP_MULTIPLIER: int = 2
#: Floor for the auto-warmup window even when strategies request less.
_MIN_WARMUP_BARS: int = 50

# Sprint 50 Cycle 4: HALT auto-stop reason marker. Public, immutable. The
# orchestrator reads engine.auto_stop_reason and compares to this value
# before writing a 'circuit_breaker_halt_auto_stop' audit row.
# WP1.6 (D4): NOTHING sets the reason any more -- HALT blocks entries only
# and never stops the run.  Kept (with the orchestrator audit branches) as an
# unreachable legacy path until CF-29.
_HALT_AUTO_STOP_REASON: str = "circuit_breaker_halt"


# ---------------------------------------------------------------------------
# Engine state enum
# ---------------------------------------------------------------------------

class EngineState(StrEnum):
    """Lifecycle state of the StrategyEngine."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Timeframe -> seconds mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_SECONDS: dict[TimeFrame, int] = {
    TimeFrame.ONE_MINUTE: 60,
    TimeFrame.THREE_MINUTES: 180,
    TimeFrame.FIVE_MINUTES: 300,
    TimeFrame.FIFTEEN_MINUTES: 900,
    TimeFrame.THIRTY_MINUTES: 1800,
    TimeFrame.ONE_HOUR: 3600,
    TimeFrame.FOUR_HOURS: 14400,
    TimeFrame.ONE_DAY: 86400,
    TimeFrame.ONE_WEEK: 604800,
}


# ---------------------------------------------------------------------------
# WP1.7a: flatten (SY-05/SY-12, arch-design WP17-A-02, risk-design WP17-R-05..09)
# ---------------------------------------------------------------------------

#: A remaining quantity at or below this tolerance after a SELL counts as
#: fully flattened ("dust") -- the same rounding tolerance the harness uses
#: (tests/integration/test_live_protective_paths.py::_QTY_TOLERANCE) for a
#: CCXT float-JSON round trip, and comfortably below any real exchange's
#: minimum order size.
_FLATTEN_DUST_TOLERANCE = Decimal("0.00000001")
#: Wait between re-poll/resend attempts when a SELL attempt produced no
#: order at all (e.g. transiently in-flight-elsewhere) -- bounded by the
#: caller's own deadline either way.
_FLATTEN_POLL_SECONDS = 1.0
#: WP17-A-02: "re-sends the SELL at most twice" -- three total attempts.
_FLATTEN_MAX_ATTEMPTS = 3


class FlattenPreconditionError(RuntimeError):
    """Raised by :meth:`StrategyEngine.flatten` when the kill switch is not
    active (I7) -- flatten never runs while entries are still open."""


@dataclass(slots=True)
class FlattenSymbolResult:
    """Per-symbol outcome of one :meth:`StrategyEngine.flatten` call (SY-12)."""

    symbol: str
    #: no_position | flat | dust | partial | in_flight | failed
    status: str
    #: None | ledger_doubt | inflight_other | submit_unknown | rejected |
    #: timeout_open | live_gate_closed | error
    cause: str | None
    held_before: Decimal
    sold_qty: Decimal
    remaining_qty: Decimal
    order_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class FlattenResult:
    """Run-level outcome of one :meth:`StrategyEngine.flatten` call (SY-12)."""

    run_id: str
    #: noop | flattened | partial | failed
    outcome: str
    complete: bool
    symbols: list[FlattenSymbolResult] = field(default_factory=list)
    #: Always True at this layer -- the engine itself never persists
    #: anything.  Callers (stop_run/emergency_stop_run/kill_switch) that
    #: persist a per-run or global latch overwrite this field with whether
    #: THEIR OWN write succeeded before returning the result to the API
    #: caller.
    latch_persisted: bool = True


def build_synthetic_flatten_result(
    engine: StrategyEngine,
    *,
    reason: str,
    cause: str,
    error: str | None = None,
) -> FlattenResult:
    """WP1.7a round 2 (S-05/S-07): a best-effort ``FlattenResult`` for
    when ``engine.flatten()`` itself raised, or hung past even the
    caller's own outer safety-net timeout (S-06 covers the in-process
    ``_cycle_lock`` timeout case *inside* ``flatten()`` itself; this
    covers a caller-side ``wait_for``/``except`` wrapper catching
    something ``flatten()``'s own defences did not).

    Never returns a complete result -- callers (``stop_run``,
    ``emergency_stop_run``, the kill-switch flatten pass) persist
    ``flatten_incomplete`` and audit critically exactly as they would
    for any other incomplete flatten.

    Parameters
    ----------
    reason:
        The same ``reason`` the failed/timed-out ``flatten()`` call was
        given (echoed for log/audit context only -- not stored on the
        result itself).
    cause:
        ``"timeout_open"`` for a caller-side timeout, ``"error"`` for an
        exception -- controls whether each held symbol is reported
        ``"in_flight"`` (timeout: the SELL may still be in flight
        somewhere) or ``"failed"`` (exception: nothing is known to still
        be in flight).
    """
    symbol_results: list[FlattenSymbolResult] = []
    for symbol in engine.symbols:
        try:
            position = engine.portfolio.get_position(symbol)
            held = (
                position.quantity
                if position is not None and not position.is_flat
                else Decimal("0")
            )
        except Exception:
            held = Decimal("0")

        if held <= Decimal("0"):
            symbol_results.append(
                FlattenSymbolResult(
                    symbol=symbol,
                    status="no_position",
                    cause=None,
                    held_before=Decimal("0"),
                    sold_qty=Decimal("0"),
                    remaining_qty=Decimal("0"),
                )
            )
            continue

        status = "in_flight" if cause == "timeout_open" else "failed"
        symbol_results.append(
            FlattenSymbolResult(
                symbol=symbol,
                status=status,
                cause=cause,
                held_before=held,
                sold_qty=Decimal("0"),
                remaining_qty=held,
                error=error,
            )
        )

    non_flat = [r for r in symbol_results if r.status != "no_position"]
    # WP17a-S-R2-04 (round 3): a caller-side timeout means the SELL
    # may still be in flight -- reporting it as "failed" (a hard,
    # known failure) is wrong and could suppress a retry a caller
    # otherwise gates on outcome. "partial" (never "complete", still
    # never true -- see ``complete=False`` below) matches every other
    # in_flight/lock_timeout path's outcome derivation.
    if not non_flat:
        outcome = "noop"
    elif cause == "timeout_open":
        outcome = "partial"
    else:
        outcome = "failed"
    return FlattenResult(
        run_id=engine.run_id or "",
        outcome=outcome,
        complete=False,
        symbols=symbol_results,
    )


@dataclass
class _EngineLot:
    """Engine-side facts of one open lot (WP1.3b-a): the opening order, its BUY
    fills so far, and the :class:`LotEntry` derived from them.  Kept so a later
    fill of the opening order can re-blend the entry basis without a re-read."""

    opening_order_id: UUID | None
    order: Order | None
    fills: list[Fill]
    entry: LotEntry


class StrategyEngine:
    """
    Central orchestrator for a trading run.

    Wires together strategies, execution engine, risk manager, market data
    service, and portfolio accounting into a coherent bar-by-bar processing
    loop.  Supports BACKTEST, PAPER, and LIVE run modes through a unified
    async interface.

    Parameters
    ----------
    strategies :
        One or more strategy instances. Each produces independent signals
        on every bar.
    execution_engine :
        The execution engine (paper or live) that processes signals into
        orders and fills.
    risk_manager :
        Pre-trade risk gating and position sizing.
    market_data :
        Market data service for fetching OHLCV candles (used in PAPER/LIVE
        modes).
    portfolio :
        Portfolio accounting for equity tracking, PnL, and drawdown.
    symbols :
        List of trading pairs to monitor, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    timeframe :
        Candle timeframe for the run.
    run_mode :
        BACKTEST, PAPER, or LIVE.
    config :
        Optional configuration overrides. Recognised keys:
        - ``warmup_bars`` (int): number of bars required before strategies
          receive their first ``on_bar`` call. Default 50.
        - ``max_bars_history`` (int): maximum rolling window size for
          live/paper mode. Default 500.
        - ``poll_interval_seconds`` (float | None): override the default
          polling interval derived from ``timeframe``. Default None.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        execution_engine: BaseExecutionEngine,
        risk_manager: BaseRiskManager,
        market_data: BaseMarketDataService,
        portfolio: PortfolioAccounting,
        symbols: list[str],
        timeframe: TimeFrame,
        run_mode: RunMode,
        config: dict[str, Any] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        protective_mode: bool = False,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        if not symbols:
            raise ValueError("At least one symbol is required")

        self._strategies = list(strategies)
        self._circuit_breaker = circuit_breaker
        self._execution_engine = execution_engine
        self._risk_manager = risk_manager
        self._market_data = market_data
        self._portfolio = portfolio
        self._symbols = list(symbols)
        self._timeframe = timeframe
        self._run_mode = run_mode
        self._config: dict[str, Any] = config or {}
        # WP1.8a (S10/O9): a live run resumed with mode=protective drops
        # every BUY signal (via _drop_entry_signals below) so a rebuilt
        # position can only shrink -- exits (strategy SELL, bracket,
        # trailing) keep running exactly as under the kill switch (D3).
        # Never set outside a resume; always False for a fresh run.
        self._protective_mode = protective_mode

        # WP1.1 (Verbeterplan v2, C1/C22): attach the portfolio as the
        # execution engine's live position source. Duck-typed -- only
        # LiveExecutionEngine implements attach_position_source; paper/
        # backtest engines have no such method, so this is a no-op there.
        _attach_position_source = getattr(
            self._execution_engine, "attach_position_source", None
        )
        if callable(_attach_position_source):
            _attach_position_source(self._portfolio, symbols=self._symbols)

        # WP1.5 (C-10): attach the portfolio as the engine's risk-state source
        # in EVERY mode (duck-typed, so a test double without the method is
        # fine).  Paper/backtest read NAV / peak / day-start basis / latch
        # from it; live reads the same portfolio through its position source.
        # Deliberately a separate hook from attach_position_source (that one
        # means "the ledger the held quantity comes from", C1/C22).
        _attach_risk_state_source = getattr(
            self._execution_engine, "attach_risk_state_source", None
        )
        if callable(_attach_risk_state_source):
            _attach_risk_state_source(self._portfolio)

        # Derived configuration
        config_warmup = self._config.get("warmup_bars")
        if config_warmup is not None:
            self._warmup_bars = int(config_warmup)
        else:
            max_min_bars = max(
                (s.min_bars_required for s in self._strategies),
                default=0,
            )
            self._warmup_bars = max(
                max_min_bars * _DEFAULT_WARMUP_MULTIPLIER, _MIN_WARMUP_BARS
            )
        self._max_bars_history: int = int(
            self._config.get("max_bars_history", 500)
        )
        poll_override = self._config.get("poll_interval_seconds")
        self._poll_interval: float = (
            float(poll_override)
            if poll_override is not None
            else float(_TIMEFRAME_SECONDS.get(timeframe, 60))
        )

        # Run state
        self._state = EngineState.IDLE
        self._run_id: str | None = None
        self._bar_count: int = 0
        self._total_signals: int = 0
        self._total_orders: int = 0
        self._total_fills: int = 0
        # M2 (Sprint 49): authoritative per-bar exposure counters for backtest mode.
        # Populated by run_backtest() and read by BacktestRunner.run() via the return dict.
        # Reset to zero at the start of each run_backtest() call.
        self._exposure_bars_total: int = 0
        self._exposure_bars_per_symbol: dict[str, int] = {}
        self._stop_event: asyncio.Event = asyncio.Event()
        # WP1.7a (SY-04, I6): serialises flatten() against the regular
        # per-bar pipeline. ``_poll_and_process`` holds it around
        # ``_process_bar`` (which itself calls ``_check_resting_orders``
        # and every ``get_fills``); ``flatten`` holds it for its whole
        # run. Never taken in backtest mode (single-threaded, no flatten).
        self._cycle_lock: asyncio.Lock = asyncio.Lock()

        # Sprint 50 Cycle 4: auto-stop reason.  WP1.6 (D4): never assigned any
        # more (pinned by a contract test) -- always None.  Removal: CF-29.
        self._auto_stop_reason: str | None = None

        # Adaptive learning - excursion and skip tracking (Sprint 32)
        self._excursion_tracker = TradeExcursionTracker()
        self._skip_logger = TradeSkipLogger()
        self._last_mtf_context: MultiTimeframeContext | None = None

        # Rolling bar windows for paper/live mode: symbol -> list[OHLCVBar]
        self._bar_windows: dict[str, list[OHLCVBar]] = {
            s: [] for s in self._symbols
        }

        # WP1.3a (SY-13a-01/06/22): a bad exit config is now a hard
        # construction failure everywhere (create/promote/resume already
        # validated it -- this is defence in depth for any caller that
        # builds a StrategyEngine directly, e.g. BacktestRunner/tests).  The
        # previous "warn and disable" behaviour (``engine.trailing_stop_
        # disabled`` / ``engine.bracket_exit_disabled``) is gone: a run must
        # never silently trade with no exit (C15).
        #
        # A protective resume MAY waive ``require_exit_manager`` (never the
        # parse itself) via ``exit_config_waived=True``, which is accepted
        # only together with ``protective_mode=True`` (SY-13a-16) -- entries
        # are already fully blocked in protective mode (``_drop_entry_
        # signals``), so a missing exit cannot let a new position open
        # unprotected; it only lets the operator reach ``flatten=true``.
        exit_config_waived = bool(self._config.get("exit_config_waived", False))
        if exit_config_waived and not protective_mode:
            raise ValueError(
                "exit_config_waived=True is only accepted with protective_mode=True"
            )

        _bracket_raw = {
            k: self._config.get(k)
            for k in (
                "bracket_mode",
                "bracket_stop_loss_pct",
                "bracket_take_profit_pct",
                "bracket_atr_sl_multiplier",
                "bracket_atr_tp_multiplier",
                "bracket_atr_period",
            )
            if k in self._config
        }
        _exit_cfg = parse_exit_config(
            bracket=_bracket_raw,
            trailing_stop_pct=self._config.get("trailing_stop_pct"),
        )
        if not exit_config_waived:
            require_exit_manager(self._strategies, _exit_cfg)

        self._trailing_stop: TrailingStopManager | None = None
        if _exit_cfg.trailing_stop_pct is not None:
            self._trailing_stop = TrailingStopManager(
                trailing_stop_pct=_exit_cfg.trailing_stop_pct,
                strategy_id="trailing_stop",
            )

        # Bracket exit manager (optional, fixed/ATR stop-loss + take-profit).
        # Anchored to position entry price; complements the trailing stop.
        # Config keys carry a ``bracket_`` prefix to avoid collision with any
        # strategy's own parameter names.
        self._bracket_exit: BracketExitManager | None = None
        if _exit_cfg.bracket:
            _raw_atr_period: Any = _exit_cfg.bracket.get("bracket_atr_period")
            self._bracket_exit = BracketExitManager(
                stop_loss_pct=_opt_float(_exit_cfg.bracket.get("bracket_stop_loss_pct")),
                take_profit_pct=_opt_float(_exit_cfg.bracket.get("bracket_take_profit_pct")),
                bracket_mode=str(_exit_cfg.bracket.get("bracket_mode") or "fixed"),
                atr_sl_multiplier=_opt_float(_exit_cfg.bracket.get("bracket_atr_sl_multiplier")),
                atr_tp_multiplier=_opt_float(_exit_cfg.bracket.get("bracket_atr_tp_multiplier")),
                # CF-13a-15: ``parse_exit_config`` always normalises the period
                # (default 14) when a bracket is configured -- no ``or 14``.
                atr_period=int(_raw_atr_period),
                strategy_id="bracket_exit",
            )

        # WP1.3b-a: lot / anchor bookkeeping (all in memory; rebuilt on resume).
        self._tf_s: int = _TIMEFRAME_SECONDS.get(timeframe, 60)
        # Same default cost model the exit-config validator uses (SY-13a-01).
        self._exit_cost_model: ExitCostModel = ExitCostModel.from_risk_parameters()
        # symbol -> facts of the open lot (opening order, its fills, LotEntry).
        self._lots: dict[str, _EngineLot] = {}
        # BACKTEST only: opening-order id -> open time of the bar that was being
        # processed when the order was created (the signal bar).  In memory only;
        # backtest timestamps are wall clock, so the persisted-data rule
        # (``anchor_at``) cannot be used there.
        self._order_signal_bar_ts: dict[UUID, datetime] = {}
        # The history of the bar currently being processed (entry hook input).
        self._current_history: dict[str, list[OHLCVBar]] | None = None
        # prime_resume_exit_state() output, consumed by _restore_exit_anchors().
        self._primed_lots: dict[str, _EngineLot] = {}
        # symbol -> ``_bar_count`` when its anchor was created/restored; a
        # provisional/deferred anchor is finalised on a LATER pass only.
        self._anchor_created_bar: dict[str, int] = {}
        # symbol -> exact-window fetch attempts spent on its pending anchor
        # (WP1.3b-a FX-03: at most ``_ANCHOR_FETCH_MAX_ATTEMPTS``, <= 1 per pass;
        # the restore fetch of a resumed lot counts as attempt 1).
        self._anchor_fetch_attempts: dict[str, int] = {}
        # symbol -> where the pending anchor came from: the entry hook ("entry")
        # or a resume restore ("resume"); decides its final ``source`` tag.
        self._anchor_origin: dict[str, Literal["entry", "resume"]] = {}
        # Remaining finalisation fetches in the current ``_finalize_pending_anchors``
        # call (engine-wide, ``_ANCHOR_FETCHES_PER_PASS``; SEC-06).
        self._anchor_fetch_budget: int = _ANCHOR_FETCHES_PER_PASS
        # symbol -> error type of the last failed exact-window fetch (log only).
        self._last_exit_fetch_error: dict[str, str] = {}
        # symbol -> consecutive bars a held lot had no protective level (FX-06).
        self._unprotected_bars: dict[str, int] = {}

        # WP1.3a (SY-13a-08): resolve allow_pyramiding from the (already
        # type-checked) config, defaulting per-strategy when absent.  A
        # non-bool value is a construction error (E1-style, but this is the
        # engine layer -- the API layer never lets a non-bool through).
        _raw_pyramiding = self._config.get("allow_pyramiding")
        if _raw_pyramiding is not None and not isinstance(_raw_pyramiding, bool):
            raise ValueError(
                f"allow_pyramiding must be a bool, got {_raw_pyramiding!r}"
            )
        self._allow_pyramiding: bool = resolve_allow_pyramiding(
            _raw_pyramiding, self._strategies
        )

        # Higher-timeframe bar data (optional, for multi-TF strategies)
        self._htf_bars: dict[str, dict[str, list[OHLCVBar]]] | None = None

        self._log = structlog.get_logger(__name__).bind(
            component="strategy_engine",
            run_mode=run_mode.value,
            timeframe=timeframe.value,
            symbols=self._symbols,
            strategy_count=len(self._strategies),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        """Current lifecycle state of the engine."""
        return self._state

    @property
    def run_id(self) -> str | None:
        """Run identifier, set after ``start()`` is called."""
        return self._run_id

    @property
    def run_mode(self) -> RunMode:
        """The run mode (BACKTEST, PAPER, or LIVE)."""
        return self._run_mode

    @property
    def bar_count(self) -> int:
        """Number of bars processed so far."""
        return self._bar_count

    @property
    def portfolio(self) -> PortfolioAccounting:
        """Direct access to the portfolio accounting instance."""
        return self._portfolio

    @property
    def symbols(self) -> list[str]:
        """WP1.7a: the run's configured trading pairs -- used by the
        stop-run 422 ``flatten_decision_required`` response to report
        every currently-held symbol without reaching into a private
        attribute."""
        return list(self._symbols)

    @property
    def risk_manager(self) -> BaseRiskManager:
        """WP1.7a: read-only access to the engine's risk manager -- used
        by callers (``apply_latch``, the stop/emergency-stop/kill-switch
        routers) that need to trigger or inspect the kill-switch latch
        without reaching into a private attribute."""
        return self._risk_manager

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        """The circuit breaker instance, or None if not configured."""
        return self._circuit_breaker

    @property
    def auto_stop_reason(self) -> str | None:
        """Reason string when the engine initiated its own shutdown, else None.

        WP1.6 (D4): no producer remains -- a circuit-breaker HALT blocks
        entries only and never stops the run, so this is always ``None``.  The
        orchestrator ``finally`` branches that read it are unreachable legacy
        (removal: CF-29).  Callers MUST NOT write to this attribute.
        """
        return self._auto_stop_reason

    # ------------------------------------------------------------------
    # Lifecycle: start
    # ------------------------------------------------------------------

    async def start(self, run_id: str) -> None:
        """
        Initialise all components and prepare for bar processing.

        This method transitions the engine from IDLE to RUNNING. It calls
        ``on_start`` on each strategy and the execution engine, and
        connects to the market data service for PAPER/LIVE modes.

        Parameters
        ----------
        run_id :
            Unique identifier for this trading run.

        Raises
        ------
        RuntimeError
            If the engine is not in IDLE state.
        """
        if self._state != EngineState.IDLE:
            raise RuntimeError(
                f"Cannot start engine in state {self._state.value}; "
                f"expected IDLE"
            )

        self._state = EngineState.STARTING
        self._run_id = run_id
        self._stop_event.clear()  # Reset for this run (supports restart)
        self._log = self._log.bind(run_id=run_id)
        self._skip_logger.set_run_id(run_id)

        self._log.info("engine.starting")

        try:
            # Start execution engine
            await self._execution_engine.on_start()

            # Connect market data service (paper/live only)
            if self._run_mode in (RunMode.PAPER, RunMode.LIVE):
                await self._market_data.connect()

            # Start strategies
            for strategy in self._strategies:
                try:
                    strategy.on_start(run_id)
                except Exception:
                    self._log.exception(
                        "engine.strategy_start_failed",
                        strategy_id=strategy.strategy_id,
                    )
                    raise

            self._state = EngineState.RUNNING

            # Warn if strategies declare htf_timeframes in paper/live mode.
            # HTF data is only auto-provided in backtest mode via run_backtest;
            # paper/live would otherwise silently skip MTF analysis.
            if self._run_mode in (RunMode.PAPER, RunMode.LIVE):
                for strategy in self._strategies:
                    if strategy.htf_timeframes:
                        self._log.warning(
                            "engine.htf_not_available_in_live_mode",
                            strategy_id=strategy.strategy_id,
                            htf_timeframes=strategy.htf_timeframes,
                            msg="Strategy declares htf_timeframes but HTF data "
                                "is not auto-fetched in paper/live mode. "
                                "mtf_context will be None.",
                        )

            self._log.info(
                "engine.started",
                strategies=[s.strategy_id for s in self._strategies],
            )
            # WP1.3b-a (SY-13b-16, D-13b-04): runs started from this commit on
            # use anchored exit semantics; older runs are "legacy" (their
            # baselines are re-run in WP2.8).
            self._log.info("engine.exit_semantics", version="wp13b-1")

        except Exception:
            self._state = EngineState.ERROR
            self._log.exception("engine.start_failed")
            raise

    # ------------------------------------------------------------------
    # Lifecycle: stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """
        Gracefully shut down the engine.

        Stops all strategies, cancels open orders via the execution engine,
        closes the market data connection, and logs the final portfolio
        summary.
        """
        if self._state not in (EngineState.RUNNING, EngineState.ERROR):
            self._log.warning(
                "engine.stop_invalid_state",
                state=self._state.value,
            )
            return

        self._state = EngineState.STOPPING
        self._stop_event.set()
        self._log.info("engine.stopping")

        # Stop strategies
        for strategy in self._strategies:
            try:
                strategy.on_stop()
            except Exception:
                self._log.exception(
                    "engine.strategy_stop_failed",
                    strategy_id=strategy.strategy_id,
                )

        # Reset trailing stop tracking state
        if self._trailing_stop is not None:
            self._trailing_stop.reset()

        # Reset bracket exit tracking state (clears the anchors too)
        if self._bracket_exit is not None:
            self._bracket_exit.reset()
        self._lots.clear()
        self._order_signal_bar_ts.clear()
        self._primed_lots.clear()
        self._anchor_created_bar.clear()
        self._anchor_fetch_attempts.clear()
        self._anchor_origin.clear()
        self._last_exit_fetch_error.clear()
        self._unprotected_bars.clear()
        self._current_history = None

        # Clear adaptive-learning trackers so a subsequent start() begins with
        # a clean MAE/MFE baseline and empty skip-trade log.
        self._excursion_tracker.clear()
        skip_summary = self._skip_logger.get_skip_summary()
        if skip_summary:
            self._log.info(
                "engine.skip_summary",
                total_skips=self._skip_logger.skip_count,
                by_reason=skip_summary,
            )
        self._skip_logger.clear()
        self._last_mtf_context = None

        # Clear HTF bar data
        self._htf_bars = None

        # Cancel open orders
        open_orders = self._execution_engine.get_open_orders()
        for order in open_orders:
            try:
                await self._execution_engine.cancel_order(order.order_id)
                self._log.info(
                    "engine.order_canceled_on_stop",
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                )
            except Exception:
                self._log.exception(
                    "engine.cancel_failed_on_stop",
                    order_id=str(order.order_id),
                )

        # Stop execution engine
        await self._execution_engine.on_stop()

        # Close market data connection (paper/live only)
        if self._run_mode in (RunMode.PAPER, RunMode.LIVE):
            try:
                await self._market_data.close()
            except Exception:
                self._log.exception("engine.market_data_close_failed")

        # Log final summary
        summary = self._portfolio.get_summary()
        self._log.info(
            "engine.stopped",
            bars_processed=self._bar_count,
            total_signals=self._total_signals,
            total_orders=self._total_orders,
            total_fills=self._total_fills,
            portfolio_summary=summary,
        )

        # WP1.6 (K-20): zero the breaker gauge so a stopped run's last value
        # (e.g. HALT = 3) does not keep a ``max(...)`` Grafana alert firing.
        if self._circuit_breaker is not None:
            try:
                from common.metrics import metrics as _mc

                _mc.gauge(
                    "circuit_breaker_state",
                    0.0,
                    labels={"run_id": self._run_id or "unknown"},
                )
            except Exception:
                self._log.debug("engine.circuit_breaker_gauge_reset_failed", exc_info=True)

        self._state = EngineState.STOPPED

    # ------------------------------------------------------------------
    # Backtest entry point
    # ------------------------------------------------------------------

    async def run_backtest(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
        htf_bars: dict[str, dict[str, list[OHLCVBar]]] | None = None,
        *,
        oos_start_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Walk through historical bars deterministically.

        Each bar step feeds a growing window of history to every strategy
        (preventing look-ahead bias). The async interface is maintained for
        uniformity with live mode, but no real I/O occurs.

        Parameters
        ----------
        bars_by_symbol :
            Pre-fetched OHLCV bars keyed by symbol. Each list must be
            sorted by timestamp ascending.
        htf_bars :
            Optional higher-timeframe bar data keyed by timeframe string,
            then by symbol. Passed to strategies via MultiTimeframeContext.
        oos_start_index :
            Sprint 50 Cycle 6 (IMPL-C6-002): bar ordinal (0-based into the
            shortest symbol series) marking the first out-of-sample bar. When
            set, the engine records the live equity-curve length at the exact
            moment processing crosses into that bar and returns it in the
            summary dict under "oos_equity_curve_offset". This lets
            BacktestRunner.run() slice OOS-only per-period returns CORRECTLY
            even though the equity curve is NOT 1:1 with bars (every fill
            appends an extra point). None = no OOS measurement (default; zero
            behaviour change).

        Returns
        -------
        dict[str, Any]
            Portfolio summary at the end of the backtest.

        Raises
        ------
        RuntimeError
            If run mode is not BACKTEST.
        ValueError
            If no bars are provided for any configured symbol.
        """
        if self._run_mode != RunMode.BACKTEST:
            raise RuntimeError(
                f"run_backtest() requires RunMode.BACKTEST, "
                f"got {self._run_mode.value}"
            )

        if self._state != EngineState.RUNNING:
            raise RuntimeError(
                f"run_backtest() requires engine state RUNNING "
                f"(call start() first), got {self._state.value}"
            )

        # Validate that bars exist for all symbols
        for symbol in self._symbols:
            if symbol not in bars_by_symbol or not bars_by_symbol[symbol]:
                raise ValueError(
                    f"No bars provided for symbol {symbol}"
                )

        # Determine the number of bars to process (shortest series)
        num_bars = min(len(bars_by_symbol[s]) for s in self._symbols)

        # Store HTF bars for multi-timeframe context building
        self._htf_bars = htf_bars

        # Reset exposure counters for this backtest run
        self._exposure_bars_total = 0
        self._exposure_bars_per_symbol = dict.fromkeys(self._symbols, 0)

        # Sprint 50 Cycle 6 (IMPL-C6-002): equity-curve length captured at the
        # moment processing crosses into the first OOS bar. Stays None when
        # oos_start_index is None OR when num_bars never reaches the boundary.
        # The equity curve is NOT 1:1 with bars (every fill appends a point), so
        # we read the LIVE len() at the boundary rather than computing an offset.
        oos_equity_curve_offset: int | None = None

        self._log.info(
            "engine.backtest_starting",
            total_bars=num_bars,
            warmup_bars=self._warmup_bars,
        )

        # Walk forward bar by bar
        for bar_idx in range(num_bars):
            if self._stop_event.is_set():
                self._log.info(
                    "engine.backtest_stopped_early",
                    bar_index=bar_idx,
                )
                break

            # Sprint 50 Cycle 6 (IMPL-C6-002): capture the equity-curve length
            # the instant we reach the first OOS bar, BEFORE this bar appends any
            # equity points. Everything appended from here on is the OOS window.
            # Guard `is None` so a re-entry (impossible in a single loop, but
            # defensive) cannot overwrite the boundary.
            if (
                oos_start_index is not None
                and oos_equity_curve_offset is None
                and bar_idx >= oos_start_index
            ):
                oos_equity_curve_offset = len(self._portfolio.get_equity_curve())

            # Build the current bar snapshot and the growing history
            current_bars: dict[str, OHLCVBar] = {}
            history_by_symbol: dict[str, list[OHLCVBar]] = {}

            for symbol in self._symbols:
                bar = bars_by_symbol[symbol][bar_idx]
                current_bars[symbol] = bar
                # Growing window: all bars up to and including current
                history_by_symbol[symbol] = bars_by_symbol[symbol][
                    : bar_idx + 1
                ]

            # WP1.5 (§5.2): push the BAR CLOSE time (latest bar timestamp +
            # one timeframe) to the portfolio's risk-day clock -- every bar,
            # warmup included, BEFORE any mark or fill of this bar.  Close
            # time matches paper/live, which only ever see a bar after its
            # close; the wall clock is never read on the risk path, so a
            # backtest is deterministic.  Forward-only: an earlier time (pre-
            # WP2.3 multi-symbol misalignment) is a no-op.  Duck-typed so a
            # portfolio test double without the method does not crash.
            _advance_clock = getattr(self._portfolio, "advance_clock", None)
            if callable(_advance_clock):
                _advance_clock(
                    max(b.timestamp for b in current_bars.values())
                    + timedelta(seconds=_TIMEFRAME_SECONDS[self._timeframe])
                )

            # Update last prices on the paper execution engine
            self._update_engine_prices(current_bars)

            # Skip strategy calls during warmup, but still update prices
            if bar_idx < self._warmup_bars:
                # Update market prices in portfolio during warmup
                prices = {
                    s: bar.close for s, bar in current_bars.items()
                }
                self._portfolio.update_market_prices(prices)
                continue

            # Process this bar
            await self._process_bar(current_bars, history_by_symbol)

            # M2 (Sprint 49): after _process_bar has applied fills, query live
            # portfolio positions to count this bar authoritatively.
            # Using get_position() per symbol is O(n_symbols) dict lookups — cheap
            # compared to the O(bars × trades) post-hoc estimate it replaces.
            _any_open = False
            for _sym in self._symbols:
                _pos = self._portfolio.get_position(_sym)
                if _pos is not None and not _pos.is_flat:
                    self._exposure_bars_per_symbol[_sym] += 1
                    _any_open = True
            if _any_open:
                self._exposure_bars_total += 1

        self._log.info(
            "engine.backtest_complete",
            bars_processed=self._bar_count,
            total_signals=self._total_signals,
            total_orders=self._total_orders,
        )

        summary: dict[str, Any] = dict(self._portfolio.get_summary())
        # Include M2 exposure counters so BacktestRunner.run() can use them
        # directly instead of the post-hoc _estimate_bars_in_market heuristic.
        summary["exposure_bars_total"] = self._exposure_bars_total
        summary["exposure_bars_per_symbol"] = dict(self._exposure_bars_per_symbol)
        # Sprint 50 Cycle 6 (IMPL-C6-002): OOS equity-curve boundary offset.
        # None when oos_start_index was not supplied OR the loop never reached it.
        summary["oos_equity_curve_offset"] = oos_equity_curve_offset
        return summary

    # ------------------------------------------------------------------
    # Paper / Live loop entry point
    # ------------------------------------------------------------------

    async def run_live_loop(self) -> None:
        """
        Poll for new bars on each timeframe interval and process them.

        Runs until ``stop()`` is called or the stop event is set. Each
        iteration fetches the latest bar for every symbol, appends it to
        the rolling window, and processes it through the strategy pipeline.

        Raises
        ------
        RuntimeError
            If run mode is not PAPER or LIVE.
        """
        if self._run_mode not in (RunMode.PAPER, RunMode.LIVE):
            raise RuntimeError(
                f"run_live_loop() requires RunMode.PAPER or LIVE, "
                f"got {self._run_mode.value}"
            )

        if self._state != EngineState.RUNNING:
            raise RuntimeError(
                f"run_live_loop() requires engine state RUNNING "
                f"(call start() first), got {self._state.value}"
            )

        self._log.info(
            "engine.live_loop_starting",
            poll_interval_seconds=self._poll_interval,
            warmup_bars=self._warmup_bars,
        )

        # Initial warmup: fetch recent bars for each symbol
        await self._warmup_bar_windows()

        # Main polling loop
        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                await self._poll_and_process()
            except Exception:
                self._log.exception("engine.live_loop_iteration_error")

            # Sleep until the next candle interval
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self._poll_interval - elapsed)

            if sleep_time > 0:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=sleep_time,
                    )
                except TimeoutError:
                    # Normal: timeout means the stop event was not set
                    pass

        self._log.info("engine.live_loop_exited")

    # ------------------------------------------------------------------
    # Core bar processing
    # ------------------------------------------------------------------

    def _compute_atr_for_symbol(
        self,
        symbol: str,
        history_by_symbol: dict[str, list[OHLCVBar]],
    ) -> Decimal | None:
        """
        Compute the latest ATR (price units) for ``symbol`` from its history.

        Returns None when there are too few bars for the configured ATR
        period (Wilder warm-up) — the bracket manager treats None as
        "hold, no ATR bracket this bar".  Used only in ATR bracket mode, and
        since WP1.3b-a only for the entry-protectable gate and the lazy
        ``late`` anchor: anchored lots never re-read the ATR (their levels
        are frozen prices).
        """
        if self._bracket_exit is None:
            return None
        return atr_for_bars(
            history_by_symbol.get(symbol) or [], self._bracket_exit.atr_period
        )

    async def _process_bar(
        self,
        current_bars: dict[str, OHLCVBar],
        history_by_symbol: dict[str, list[OHLCVBar]],
    ) -> None:
        """
        Process a single bar across all strategies and symbols.

        This is the inner loop that drives the entire trading pipeline
        on each new candle.

        Parameters
        ----------
        current_bars :
            The latest bar for each symbol.
        history_by_symbol :
            Full bar history per symbol up to and including the current bar.
        """
        bar_start = time.monotonic()
        # WP1.3b-a: the entry hook (``_on_fill_routed``) anchors against the
        # history of the bar being processed.
        self._current_history = history_by_symbol

        # Get a representative timestamp for logging
        first_bar = next(iter(current_bars.values()))
        bar_timestamp = first_bar.timestamp

        # 1. Update market prices in portfolio
        prices = {s: bar.close for s, bar in current_bars.items()}
        self._portfolio.update_market_prices(prices)

        # Update MAE/MFE excursion tracker for every open position.
        for sym, b in current_bars.items():
            self._excursion_tracker.on_bar(
                symbol=sym, high=b.high, low=b.low, close=b.close
            )

        # 2. Tick risk manager cooldown
        self._risk_manager.tick_cooldown()

        # 3. Kill switch blocks new entries in every mode; exits keep
        # running (D3, WP1.2). BUY signals are dropped after the strategy
        # loop below via ``_drop_entry_signals``; SELLs, brackets (5a),
        # trailing stops (5b) and resting orders (6) are unaffected. `is
        # True` guards against a MagicMock risk manager in tests -- a mock
        # attribute is truthy even when a test never sets it explicitly.
        _kill_switch_engaged = self._risk_manager.kill_switch_active is True

        # Count only bars that reach strategy processing
        self._bar_count += 1

        # 3b. WP1.6 circuit breaker: ONE evaluation per processed bar, from the
        # typed ``RiskSnapshot`` (never ``get_summary()`` keys, C3), after the
        # marks above and before the strategies.  It is also evaluated under
        # the kill switch / protective mode so its state and alerts stay
        # truthful.  A failure is fail-closed for ENTRIES only (P-7): that
        # bar's BUYs are dropped below, exits run, breaker state is untouched
        # (``evaluate`` is transactional).  It NEVER stops the run (D4).
        _cb_decision: CircuitBreakerDecision | None = None
        _cb_failed = False
        if self._circuit_breaker is not None:
            try:
                _cb_decision = self._circuit_breaker.evaluate(self._portfolio.risk_snapshot())
            except Exception:
                _cb_failed = True
                self._log.error(
                    "engine.circuit_breaker_eval_failed",
                    bar_timestamp=str(bar_timestamp),
                    exc_info=True,
                )

        # 4. For each strategy: call on_bar and process resulting signals
        bar_signals: list[Signal] = []
        bar_orders: int = 0
        bar_fills: int = 0

        # Build multi-timeframe context (look-ahead bias filtered)
        mtf_context = self._build_mtf_context(bar_timestamp)
        # C3: Store for use in trade recording / skip logging (Sprint 32)
        self._last_mtf_context = mtf_context

        for strategy in self._strategies:
            try:
                signals = self._call_strategy_on_bar(
                    strategy, history_by_symbol, mtf_context=mtf_context
                )
                bar_signals.extend(signals)
            except Exception:
                self._log.exception(
                    "engine.strategy_on_bar_error",
                    strategy_id=strategy.strategy_id,
                    bar_index=self._bar_count,
                )
                continue

        # 4b. WP1.2 (S6) + WP1.6 (C23): the SINGLE BUY-only entry filter.  The
        # first matching reason wins, so a dropped BUY gets exactly one skip
        # record and one block reason (K-10); the risk gate is never asked
        # about it.  SELLs pass through unchanged -- exits keep running under
        # the kill switch, protective mode, HALT and DAILY_LIMIT.
        _entry_skip_reason: str | None = None
        _entry_log_event = "engine.kill_switch_entries_blocked"
        if _kill_switch_engaged:
            _entry_skip_reason = "kill_switch"
        elif self._protective_mode:
            _entry_skip_reason = "protective_mode"
        elif _cb_failed:
            _entry_skip_reason = "circuit_breaker_eval_failed"
            _entry_log_event = "engine.circuit_breaker_entries_blocked"
        elif _cb_decision is not None and _cb_decision.entries_blocked:
            _entry_skip_reason = f"circuit_breaker_{_cb_decision.level.value}"
            _entry_log_event = "engine.circuit_breaker_entries_blocked"
        if _entry_skip_reason is not None:
            bar_signals = self._drop_entry_signals(
                bar_signals,
                current_bars,
                skip_reason=_entry_skip_reason,
                log_event=_entry_log_event,
            )

        # REDUCE (WP1.6): the multiplier rides on each BUY ``Signal`` and is
        # applied to the FINAL resolved quantity in the execution engine.
        _entry_mult = (
            _cb_decision.entry_size_multiplier
            if _cb_decision is not None and not _cb_decision.entries_blocked
            else Decimal(1)
        )

        self._total_signals += len(bar_signals)

        # 4c. WP1.3b-a: advance the non-final bracket anchors (PAPER/LIVE: interim
        # levels, bounded exact-window retries, then final) created on an EARLIER
        # pass.  Two things keep this pass's own anchors out: the PLACEMENT (before
        # the signal loop and ``_check_resting_orders``, so this pass's entry hook
        # has not run yet) and the ``_anchor_created_bar`` guard in
        # ``_finalize_pending_anchors`` (defence in depth for an anchor created at
        # the current ``_bar_count``; a restore at warmup carries count 0 and is
        # advanced on the first pass).  Always before step 5a.
        try:
            await self._finalize_pending_anchors(history_by_symbol)
        except Exception as exc:
            # error_type only (CF-36 / AC-13B-18): this path does exchange I/O
            self._log.error("engine.anchor_finalize_error", error_type=type(exc).__name__)

        # 5. Process each signal through the execution engine
        for signal in bar_signals:
            try:
                # WP1.6: REDUCE -- BUY only; ``target_position`` is never mutated.
                if signal.direction == SignalDirection.BUY and _entry_mult < 1:
                    signal = signal.model_copy(
                        update={
                            "entry_size_multiplier": min(
                                signal.entry_size_multiplier, _entry_mult
                            )
                        }
                    )

                # WP1.3a (SY-13a-10): per-signal, fail-closed "no pyramiding"
                # gate.  BUY-only; SELLs are never touched (R-I7/I5).  Runs
                # after the kill-switch/protective/CB filter and the
                # REDUCE multiplier above, before ``process_signal`` so a
                # second same-bar BUY sees the first one's fill (paper/
                # backtest: synchronous MARKET fills) or in-flight state
                # (live: the existing WP1.4 run-wide block, SY-13a-11).
                if signal.direction == SignalDirection.BUY and not self._allow_pyramiding:
                    try:
                        _held = self._entry_blocked_by_held_position(
                            signal.symbol, current_bars.get(signal.symbol)
                        )
                    except Exception:
                        self._log.error(
                            "engine.entry_held_state_unknown",
                            symbol=signal.symbol,
                            strategy_id=signal.strategy_id,
                            exc_info=True,
                        )
                        continue
                    if _held:
                        self._log_held_skip(signal)
                        continue

                # WP1.3a (SY-13a-14, partial): a BUY that has an ATR bracket
                # configured but no computable ATR yet, and no trailing stop
                # to fall back on, would open with no protectable downside
                # level this bar.  This IS logged as a missed entry
                # (TradeSkipLogger) -- unlike the held-skip above, it is not
                # a suppressed add-on, it is a genuinely dropped entry.
                if signal.direction == SignalDirection.BUY and not self._entry_protectable(
                    signal.symbol, history_by_symbol
                ):
                    bar_ref = current_bars.get(signal.symbol)
                    self._skip_logger.log_skip(
                        symbol=signal.symbol,
                        skip_reason="exit_level_unavailable",
                        hypothetical_entry_price=bar_ref.close if bar_ref is not None else None,
                        signal_context=dict(signal.metadata) if signal.metadata else None,
                    )
                    self._log.warning(
                        "engine.entry_skipped_exit_level_unavailable",
                        symbol=signal.symbol,
                        strategy_id=signal.strategy_id,
                    )
                    continue

                orders = await self._execution_engine.process_signal(signal)
                bar_orders += len(orders)
                if (
                    self._run_mode == RunMode.BACKTEST
                    and signal.direction == SignalDirection.BUY
                ):
                    self._record_signal_bar(orders, signal.symbol, history_by_symbol)

                # Route fills to portfolio and risk manager (WP1.7a:
                # extracted into _route_exit_fills, shared with 5a/5b and
                # flatten()).
                fill_count, _ = await self._route_exit_fills(
                    orders, current_bars, signal
                )
                bar_fills += fill_count

            except Exception:
                self._log.exception(
                    "engine.signal_processing_error",
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                )
                # Log skipped trade on execution error so the post-run audit
                # captures decisions that never reached the execution engine.
                if signal.direction == SignalDirection.BUY:
                    bar_ref = current_bars.get(signal.symbol)
                    self._skip_logger.log_skip(
                        symbol=signal.symbol,
                        skip_reason="execution_error",
                        hypothetical_entry_price=bar_ref.close if bar_ref else None,
                        signal_context=dict(signal.metadata) if signal.metadata else None,
                    )
                continue

        # 5a. Check fixed/ATR brackets (stop-loss + take-profit) for open
        # positions.  Runs BEFORE the trailing stop so a hard SL is the
        # first line of defence; the trailing loop re-fetches the position
        # and sees it flat if a bracket already closed it (no double exit).
        # Bracket exits are SELL-only (full close), so no on_position_open
        # bookkeeping is needed here — the excursion tracker state was set
        # when the original entry filled in the strategy-signal loop above.
        if self._bracket_exit is not None:
            needs_atr = self._bracket_exit.requires_atr
            for symbol, bar in current_bars.items():
                try:
                    position = self._portfolio.get_position(symbol)
                    # WP1.3b-a: an anchored lot never re-reads the ATR (its
                    # levels are frozen prices).  The ATR is computed only for
                    # a lot WITHOUT a valid anchor (lazy ``late`` path; a stale
                    # anchor of another lot counts as none, the manager drops it).
                    _anchor = self._bracket_exit.get_anchor(symbol)
                    atr_value = (
                        self._compute_atr_for_symbol(symbol, history_by_symbol)
                        if needs_atr
                        and (
                            _anchor is None
                            or (position is not None and _anchor.lot_key != position.opened_at)
                        )
                        else None
                    )
                    exit_signal = self._bracket_exit.check(
                        symbol=symbol,
                        current_price=bar.close,
                        position=position,
                        atr_value=atr_value,
                    )
                    if exit_signal is not None:
                        self._unprotected_bars.pop(symbol, None)
                        n_orders, n_fills = await self._submit_exit_signal(
                            self._bracket_exit, exit_signal, current_bars
                        )
                        bar_orders += n_orders
                        bar_fills += n_fills
                    else:
                        self._track_lot_protection(symbol, position)
                except Exception:
                    self._log.exception(
                        "engine.bracket_exit_error",
                        symbol=symbol,
                    )

        # 5b. Check trailing stops for open positions
        if self._trailing_stop is not None:
            for symbol, bar in current_bars.items():
                try:
                    position = self._portfolio.get_position(symbol)
                    stop_signal = self._trailing_stop.check(
                        symbol=symbol,
                        current_price=bar.close,
                        position=position,
                        bar_ts=bar.timestamp,
                    )
                    if stop_signal is not None:
                        n_orders, n_fills = await self._submit_exit_signal(
                            self._trailing_stop, stop_signal, current_bars
                        )
                        bar_orders += n_orders
                        bar_fills += n_fills
                except Exception:
                    self._log.exception(
                        "engine.trailing_stop_error",
                        symbol=symbol,
                    )

        self._total_orders += bar_orders
        self._total_fills += bar_fills

        # 6. Check resting orders (paper engine limit order support)
        await self._check_resting_orders(current_bars)

        # 7. Log bar summary
        bar_elapsed_ms = (time.monotonic() - bar_start) * 1000
        self._log.debug(
            "engine.bar_processed",
            bar_index=self._bar_count,
            bar_timestamp=str(bar_timestamp),
            signals=len(bar_signals),
            orders=bar_orders,
            fills=bar_fills,
            elapsed_ms=round(bar_elapsed_ms, 2),
        )

        # 8. Update Prometheus-compatible metrics
        # TO-008 (Sprint 44): tag every counter / gauge / histogram with the
        # ``run_id`` label so Grafana panels can isolate throughput, signal
        # quality, and drawdown per-run instead of aggregating across all
        # concurrent strategies.  When run_id is None (start() not yet
        # called) we use 'unknown' as a stable placeholder.
        try:
            from common.metrics import metrics as _mc
            _run_label = {"run_id": self._run_id or "unknown"}
            _mc.increment("bars_processed_total", labels=_run_label)
            if len(bar_signals) > 0:
                _mc.increment(
                    "signals_generated_total", len(bar_signals), labels=_run_label
                )
            if bar_orders > 0:
                _mc.increment(
                    "orders_submitted_total", bar_orders, labels=_run_label
                )
            if bar_fills > 0:
                _mc.increment(
                    "fills_executed_total", bar_fills, labels=_run_label
                )
            _summary = self._portfolio.get_summary()
            _mc.gauge(
                "portfolio_equity",
                float(_summary["current_equity"]),
                labels=_run_label,
            )
            _mc.gauge(
                "portfolio_drawdown_pct",
                float(_summary["drawdown_pct"]),
                labels=_run_label,
            )
            _mc.gauge(
                "active_positions",
                float(_summary["open_positions"]),
                labels=_run_label,
            )
            # Safety-state gauges so Grafana can alert on HALT / kill-switch
            # without log scraping: OK=0, REDUCE=1, DAILY_LIMIT=2, HALT=3.
            # WP1.6: the breaker's own level; unchanged when this bar's
            # evaluation failed (``evaluate`` is transactional).
            _mc.gauge(
                "circuit_breaker_state",
                float(
                    _CB_STATE_VALUES.get(self._circuit_breaker.current_response, 0)
                    if self._circuit_breaker is not None
                    else 0
                ),
                labels=_run_label,
            )
            _mc.gauge(
                "kill_switch_active",
                1.0 if self._risk_manager.kill_switch_active else 0.0,
                labels=_run_label,
            )
            _mc.observe(
                "bar_processing_duration_seconds",
                bar_elapsed_ms / 1000.0,
                labels=_run_label,
            )
        except Exception:
            # Metrics must never crash bar processing; surface at debug level
            # so operational regressions are still observable in verbose logs.
            self._log.debug("engine.metrics_update_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Entry-signal filtering (WP1.2 / S6)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # WP1.3a: no-pyramiding held gate + entry-protectability gate
    # ------------------------------------------------------------------

    def _entry_dust_threshold(self, symbol: str, last: Decimal | None) -> Decimal:
        """Duck-typed: only ``LiveExecutionEngine`` implements
        ``entry_dust_threshold`` (SY-13a-12).  Paper/backtest have no
        exchange minimums, so the default is ``Decimal(0)`` (same as
        ``is_flat``).  Looked up on the *type*, not the instance, so a bare
        ``MagicMock()`` execution engine in a unit test (which auto-creates
        any instance attribute on access) is correctly treated as "no
        accessor" -- only a real ``LiveExecutionEngine`` (or a test double
        built with ``spec=LiveExecutionEngine``) has this on its class."""
        accessor = getattr(type(self._execution_engine), "entry_dust_threshold", None)
        if not callable(accessor):
            return Decimal("0")
        bound_accessor: Any = accessor.__get__(self._execution_engine)
        result = bound_accessor(symbol, last)
        return result if isinstance(result, Decimal) else Decimal(str(result))

    def _entry_blocked_by_held_position(
        self, symbol: str, last_bar: OHLCVBar | None
    ) -> bool:
        """True iff a BUY for ``symbol`` must be dropped because the ledger
        already shows more than dust (SY-13a-10/12, R-I2).

        Deliberately does NOT catch exceptions -- the caller wraps this in
        its own try/except and drops the BUY on ANY exception, fail-closed
        (R-I6/SY-13a-10), logging ``engine.entry_held_state_unknown``.
        """
        position = self._portfolio.get_position(symbol)
        if position is None:
            return False
        if not isinstance(position, Position):
            # WP13a-S-04 (security round 2): a non-None value that is NOT a
            # real Position is a ledger-state anomaly (e.g. a broken/mocked
            # portfolio), not evidence of "flat" -- fail closed (R-I6) by
            # treating it as held, exactly like the caller's own
            # try/except treats any exception from this method.
            return True
        if position.is_flat:
            return False
        last_price = Decimal(str(last_bar.close)) if last_bar is not None else None
        threshold = self._entry_dust_threshold(symbol, last_price)
        return position.quantity > threshold

    def _entry_protectable(
        self, symbol: str, history_by_symbol: dict[str, list[OHLCVBar]]
    ) -> bool:
        """True unless an ATR bracket is configured and ATR cannot be
        computed yet for ``symbol``, with no trailing stop to fall back on
        (SY-13a-14, partial -- the ``stop_inside_entry_cost`` drop decision
        is deferred to WP1.3b/CF-13a-4)."""
        if self._bracket_exit is None or not self._bracket_exit.requires_atr:
            return True
        if self._trailing_stop is not None:
            return True
        atr_value = self._compute_atr_for_symbol(symbol, history_by_symbol)
        return atr_value is not None

    def _log_held_skip(self, signal: Signal) -> None:
        """Log a suppressed pyramiding add-on.  Deliberately NOT written to
        ``TradeSkipLogger`` (SY-13a-10): a suppressed add-on is not a missed
        entry, and a level-triggered strategy (rsi_mean_reversion, dca)
        would otherwise log one skip per bar and pollute adaptive-learning
        skip analysis.  Info in paper/live, debug in backtest (lower noise
        for the hot backtest loop)."""
        log_fn = (
            self._log.debug
            if self._run_mode == RunMode.BACKTEST
            else self._log.info
        )
        log_fn(
            "engine.entry_skipped_position_held",
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
        )

    def _drop_entry_signals(
        self,
        signals: list[Signal],
        current_bars: dict[str, OHLCVBar],
        *,
        skip_reason: str,
        log_event: str = "engine.kill_switch_entries_blocked",
    ) -> list[Signal]:
        """
        Drop BUY signals only, logging a skip for each one.

        Called while entries are blocked: the kill switch (D3), protective
        mode, and (WP1.6, C23) the circuit breaker's HALT / DAILY_LIMIT /
        evaluation failure. SELL signals pass through unchanged -- exits
        must keep running even while entries are blocked. The side-aware
        ``DefaultRiskManager`` gate is what actually enforces the block;
        this filter only avoids a pointless downstream order attempt for
        BUYs and keeps the skip audit populated.  ``log_event`` names the
        once-per-bar warning (the kill-switch name stays the default).
        """
        kept: list[Signal] = []
        dropped_buy_count = 0
        for signal in signals:
            if signal.direction != SignalDirection.BUY:
                kept.append(signal)
                continue
            dropped_buy_count += 1
            if self._skip_logger is not None:
                bar_ref = current_bars.get(signal.symbol)
                self._skip_logger.log_skip(
                    symbol=signal.symbol,
                    skip_reason=skip_reason,
                    hypothetical_entry_price=bar_ref.close if bar_ref is not None else None,
                    signal_context=dict(signal.metadata) if signal.metadata else None,
                )
        if dropped_buy_count:
            _first_bar = next(iter(current_bars.values()), None)
            self._log.warning(
                log_event,
                bar_timestamp=str(_first_bar.timestamp) if _first_bar is not None else None,
                dropped_buy_count=dropped_buy_count,
                skip_reason=skip_reason,
            )
        return kept

    # ------------------------------------------------------------------
    # Strategy invocation
    # ------------------------------------------------------------------

    def _call_strategy_on_bar(
        self,
        strategy: BaseStrategy,
        history_by_symbol: dict[str, list[OHLCVBar]],
        mtf_context: MultiTimeframeContext | None = None,
    ) -> list[Signal]:
        """
        Call a strategy's ``on_bar`` for each symbol and collect signals.

        Parameters
        ----------
        strategy :
            The strategy to invoke.
        history_by_symbol :
            Bar history for each symbol.
        mtf_context :
            Optional higher-timeframe context for strategies that declared
            htf_timeframes. None if no HTF data is available.

        Returns
        -------
        list[Signal]
            Signals produced by this strategy across all symbols.
        """
        all_signals: list[Signal] = []

        for symbol in self._symbols:
            bars = history_by_symbol.get(symbol, [])
            if not bars:
                continue

            signals = strategy.on_bar(bars, mtf_context=mtf_context)

            if signals:
                all_signals.extend(signals)

        return all_signals

    def _build_mtf_context(
        self,
        current_timestamp: datetime,
    ) -> MultiTimeframeContext | None:
        """
        Build a MultiTimeframeContext filtered to prevent look-ahead bias.

        Only includes HTF bars whose full period has completed before
        the current primary bar timestamp.  Also injects all available
        external market signal values from module-level singleton clients
        (FGI, CoinGecko, FRED, Whale Alert).  All signal reads are
        best-effort: any exception is swallowed to ensure bar processing
        never crashes due to a failed external signal fetch.
        """
        # Read cached FGI value (best-effort; never crash bar processing
        # if the client is unavailable or has not populated its cache yet).
        fgi_value: int | None = None
        fgi_value_7d_ago: int | None = None
        try:
            from data.sentiment import get_global_client as _get_fgi_client
            _fgi_client = _get_fgi_client()
            if _fgi_client is not None:
                fgi_value = _fgi_client.cached_value
                # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
                # Returns None when history has not been populated yet by the
                # background refresh task; v2 feature builder treats None as
                # "no observed change".
                fgi_value_7d_ago = _fgi_client.value_at_offset_from_cache(
                    days_ago=7,
                )
        except Exception:
            pass  # FGI is best-effort; never crash bar processing

        # CoinGecko market structure signals (best-effort)
        btc_dominance: float | None = None
        btc_dominance_7d_ago: float | None = None
        market_cap_change_24h: float | None = None
        total_volume_change_24h: float | None = None
        try:
            from data.market_signals import get_global_client as _get_cg_client
            _cg_client = _get_cg_client()
            if _cg_client is not None:
                _cg_snap = _cg_client.cached_value
                if _cg_snap is not None:
                    btc_dominance = _cg_snap.btc_dominance
                    market_cap_change_24h = _cg_snap.market_cap_change_24h
                    total_volume_change_24h = _cg_snap.total_volume_change_24h
                # QT-007 v2 feature pipeline: cache-only sync read (no I/O).
                btc_dominance_7d_ago = (
                    _cg_client.btc_dominance_at_offset_from_cache(days_ago=7)
                )
        except Exception:
            pass  # CoinGecko is best-effort; never crash bar processing

        # FRED macro-economic signals (best-effort)
        fed_funds_rate: float | None = None
        yield_curve_spread: float | None = None
        try:
            from data.macro_data import get_global_client as _get_fred_client
            _fred_client = _get_fred_client()
            if _fred_client is not None:
                _fred_snap = _fred_client.cached_value
                if _fred_snap is not None:
                    fed_funds_rate = _fred_snap.fed_funds_rate
                    yield_curve_spread = _fred_snap.yield_curve_spread
        except Exception:
            pass  # FRED is best-effort; never crash bar processing

        # Whale Alert on-chain flow signals (best-effort)
        whale_net_flow: float | None = None
        try:
            from data.whale_tracker import get_global_client as _get_whale_client
            _whale_client = _get_whale_client()
            if _whale_client is not None:
                _whale_snap = _whale_client.cached_value
                if _whale_snap is not None:
                    whale_net_flow = _whale_snap.net_flow
        except Exception:
            pass  # Whale Alert is best-effort; never crash bar processing

        # Build kwargs for MultiTimeframeContext — only include non-None signal
        # fields so the frozen dataclass default values are used for all absent
        # signals, preserving backward compatibility with existing tests.
        ctx_kwargs: dict[str, Any] = {}
        if fgi_value is not None:
            ctx_kwargs["fear_greed_index"] = fgi_value
        if btc_dominance is not None:
            ctx_kwargs["btc_dominance"] = btc_dominance
        if market_cap_change_24h is not None:
            ctx_kwargs["market_cap_change_24h"] = market_cap_change_24h
        if total_volume_change_24h is not None:
            ctx_kwargs["total_volume_change_24h"] = total_volume_change_24h
        if fed_funds_rate is not None:
            ctx_kwargs["fed_funds_rate"] = fed_funds_rate
        if yield_curve_spread is not None:
            ctx_kwargs["yield_curve_spread"] = yield_curve_spread
        if whale_net_flow is not None:
            ctx_kwargs["whale_net_flow"] = whale_net_flow
        if fgi_value_7d_ago is not None:
            ctx_kwargs["fear_greed_index_7d_ago"] = fgi_value_7d_ago
        if btc_dominance_7d_ago is not None:
            ctx_kwargs["btc_dominance_7d_ago"] = btc_dominance_7d_ago

        # Always return a context carrying the signal fields (fgi/btc_dom/...)
        # even when no HTF bars exist — strategies may condition on sentiment
        # or macro signals without needing multi-timeframe OHLCV.
        if self._htf_bars is None:
            if ctx_kwargs:
                return MultiTimeframeContext(**ctx_kwargs)
            return None

        filtered: dict[str, dict[str, list[OHLCVBar]]] = {}
        for tf_str, bars_by_sym in self._htf_bars.items():
            try:
                tf_key = TimeFrame(tf_str)
            except ValueError:
                self._log.warning(
                    "engine.unknown_htf_timeframe",
                    timeframe=tf_str,
                    msg="Unrecognised HTF timeframe  -- excluding all bars as safety default.",
                )
                # Exclude all bars for unknown timeframes (safe default)
                filtered[tf_str] = {sym: [] for sym in bars_by_sym}
                continue
            tf_duration = _TIMEFRAME_SECONDS.get(tf_key, 0)
            filtered_sym: dict[str, list[OHLCVBar]] = {}
            for symbol, bars in bars_by_sym.items():
                # Only include bars whose full period ended before current_timestamp
                # A bar opened at T with duration D is complete at T + D
                filtered_sym[symbol] = [
                    b for b in bars
                    if b.timestamp.timestamp() + tf_duration <= current_timestamp.timestamp()
                ]
            filtered[tf_str] = filtered_sym

        return MultiTimeframeContext(htf_bars=filtered, **ctx_kwargs)

    # ------------------------------------------------------------------
    # Regime classification (Sprint 32 CR-001)
    # ------------------------------------------------------------------

    @staticmethod
    def _fgi_to_regime(fgi: int | None) -> str | None:
        """Classify a Fear & Greed Index value into a regime label."""
        if fgi is None:
            return None
        if fgi <= 24:
            return "EXTREME_FEAR"
        elif fgi <= 44:
            return "FEAR"
        elif fgi <= 55:
            return "NEUTRAL"
        elif fgi <= 75:
            return "GREED"
        else:
            return "EXTREME_GREED"

    # ------------------------------------------------------------------
    # Fill routing
    # ------------------------------------------------------------------

    async def _route_exit_fills(
        self,
        orders: list[Order],
        current_bars: dict[str, OHLCVBar] | None,
        signal: Signal,
    ) -> tuple[int, Decimal]:
        """Route every fill for ``orders`` to the portfolio, the excursion
        tracker, trade recording and the risk manager (WP1.7a, extracted
        unchanged from the pre-WP1.7a inline blocks 5/5a/5b so every
        caller -- the main signal loop, bracket exits, trailing stops and
        :meth:`flatten` -- shares byte-for-byte identical fill-routing
        semantics, arch-design WP17-A-02).

        MUST be called with ``_cycle_lock`` held (I6) whenever
        ``current_bars`` reflects a live/paper bar in progress --
        ``_poll_and_process`` holds it around the whole ``_process_bar``
        call, and ``flatten`` holds it for its whole run, so two callers
        can never both read the same not-yet-routed
        ``LiveExecutionEngine._routed_trade_keys`` state for one order.

        Parameters
        ----------
        orders:
            Orders just returned by ``process_signal`` (or a bracket/
            trailing-stop/flatten exit signal's own call to it).
        current_bars:
            The current per-symbol bar snapshot, or ``None`` when no bar
            context is available (``flatten`` outside a poll cycle) -- a
            fill priced against a symbol missing from this dict falls
            back to the fill's own execution price rather than being
            skipped (unlike the pre-WP1.7a main-loop guard, which only
            ever fired for pre-existing multi-symbol bugs no test
            exercises).

        Returns
        -------
        tuple[int, Decimal]:
            ``(fill_count, total_filled_qty)`` across every routed fill.
        """
        fill_count = 0
        total_qty = Decimal("0")
        for order in orders:
            fills = await self._execution_engine.get_fills(order.order_id)
            fill_count += len(fills)

            for fill in fills:
                symbol_bar = (current_bars or {}).get(fill.symbol)
                current_price = symbol_bar.close if symbol_bar is not None else fill.price

                # Capture position BEFORE fill for trade recording
                pre_fill_pos = self._portfolio.get_position(fill.symbol)

                self._portfolio.update_position(fill, current_price)
                # WP1.3b-a: lot / anchor lifecycle (never raises).
                self._on_fill_routed(order, fill, pre_fill_pos)

                # C4: Start excursion tracking when a new BUY fill opens a position (Sprint 32)
                if fill.side.value == "buy" or str(fill.side) in ("buy", "BUY"):
                    post_fill_pos = self._portfolio.get_position(fill.symbol)
                    if post_fill_pos is not None and not post_fill_pos.is_flat:
                        fgi_val: int | None = None
                        if self._last_mtf_context is not None:
                            fgi_val = self._last_mtf_context.fear_greed_index
                        self._excursion_tracker.on_position_open(
                            symbol=fill.symbol,
                            entry_price=fill.price,
                            side="long",
                            regime_at_entry=self._fgi_to_regime(fgi_val),
                            signal_context=dict(signal.metadata) if signal.metadata else None,
                        )

                # Record trade if this fill closed/reduced a position
                self._record_trade_if_closed(
                    fill=fill,
                    pre_fill_position=pre_fill_pos,
                    strategy_id=signal.strategy_id,
                    signal_metadata=dict(signal.metadata) if signal.metadata else None,
                )

                # Determine if this fill closed a position (for risk
                # manager loss tracking). A SELL fill on a position
                # that is now flat indicates a completed trade.
                self._route_fill_to_risk_manager(fill, current_price)

                total_qty += fill.quantity

        return fill_count, total_qty

    def _route_fill_to_risk_manager(
        self,
        fill: Fill,
        current_price: Decimal,
    ) -> None:
        """
        Route fill information to the risk manager for loss-streak tracking.

        This method examines the portfolio's position snapshot to determine
        whether the fill resulted in a closed (or partially closed) position
        and whether the trade was profitable.

        Parameters
        ----------
        fill :
            The fill event.
        current_price :
            Current market price for the fill's symbol.
        """
        # Only SELL fills can close positions (spot-only MVP)
        if fill.side != OrderSide.SELL:
            return

        # Check the position state after the fill has been applied to
        # portfolio. If the position is flat, a round trip was completed.
        position = self._portfolio.get_position(fill.symbol)
        if position is not None and position.is_flat:
            realised_pnl = position.realised_pnl
            is_loss = realised_pnl < Decimal("0")
            try:
                self._risk_manager.update_after_fill(
                    realised_pnl=realised_pnl,
                    is_loss=is_loss,
                )
            except Exception:
                self._log.exception(
                    "engine.risk_update_after_fill_error",
                    symbol=fill.symbol,
                )

    def _record_trade_if_closed(
        self,
        fill: Fill,
        pre_fill_position: Position | None,
        strategy_id: str,
        signal_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Detect round-trip completion and record a TradeResult.

        Called after update_position() has applied the fill. Compares the
        pre-fill position state with the post-fill state to determine
        whether a position was fully or partially closed.

        Sprint 32: Enriches TradeResult with MAE/MFE excursion data,
        exit reason classification, regime at entry, and signal context.

        Parameters
        ----------
        fill :
            The fill event that was just applied.
        pre_fill_position :
            The position snapshot captured BEFORE update_position() was called.
            None if no position existed for this symbol.
        strategy_id :
            The strategy that generated the signal leading to this fill.
        signal_metadata :
            Optional metadata dict from the closing signal (Sprint 32).
        """
        # Only SELL fills can close long positions (spot-only MVP)
        if fill.side != OrderSide.SELL:
            return
        # No pre-existing position to close
        if pre_fill_position is None or pre_fill_position.is_flat:
            return

        closed_qty = min(fill.quantity, pre_fill_position.quantity)
        if closed_qty <= Decimal("0"):
            return

        # PnL for the closed portion
        pnl = (fill.price - pre_fill_position.average_entry_price) * closed_qty - fill.fee

        # Total fees for this trade: exit fill fee only.
        # Entry fees are already embedded in average_entry_price (all-in cost
        # basis), so adding them again would double-count.
        total_fees = fill.fee

        if self._run_id is None:
            self._log.error("engine.trade_record_no_run_id", symbol=fill.symbol)
            return

        now = datetime.now(tz=UTC)

        # Enrich the trade record with excursion (MAE/MFE) and exit
        # classification so post-run analytics can slice by exit reason.
        excursion_data = self._excursion_tracker.on_position_close(fill.symbol)
        mae_pct: float | None = None
        mfe_pct: float | None = None
        regime_at_entry: str | None = None
        entry_signal_context: dict[str, Any] | None = None

        if excursion_data is not None:
            mae_pct, mfe_pct, regime_at_entry, entry_signal_context = excursion_data

        exit_reason = ExitReasonDetector.detect(
            strategy_id=strategy_id,
            signal_metadata=signal_metadata,
        )

        try:
            trade = TradeResult(
                run_id=self._run_id,
                symbol=fill.symbol,
                side=OrderSide.BUY,  # Opening side for long position (spot-only)
                # entry_price is the all-in cost basis (includes entry fees),
                # not the raw execution price. Matches portfolio VWAP calculation.
                entry_price=pre_fill_position.average_entry_price,
                exit_price=fill.price,
                quantity=closed_qty,
                realised_pnl=pnl,
                total_fees=total_fees,
                entry_at=pre_fill_position.opened_at,
                exit_at=now,
                strategy_id=strategy_id,
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
                exit_reason=exit_reason,
                regime_at_entry=regime_at_entry,
                signal_context=entry_signal_context,
            )
            self._portfolio.record_trade(trade)
            self._log.info(
                "engine.trade_recorded",
                trade_id=str(trade.trade_id),
                symbol=trade.symbol,
                pnl=str(trade.realised_pnl),
                quantity=str(trade.quantity),
                exit_reason=exit_reason,
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
            )
        except Exception:
            self._log.exception(
                "engine.trade_record_error",
                symbol=fill.symbol,
            )

    # ------------------------------------------------------------------
    # WP1.3b-a: exit anchors (entry anchoring, R-04, resume recompute)
    # ------------------------------------------------------------------

    @staticmethod
    def _exit_not_submitted(orders: Sequence[Order]) -> bool:
        """True when an exit signal produced NO accepted order (R-04, SY-13b-04).

        That is an empty list, or every returned order terminal
        (``REJECTED``/``CANCELED``/``EXPIRED``) with nothing filled.  An order
        in any other state (``PENDING_SUBMIT``/``OPEN``/``PARTIAL``/``FILLED``/
        unknown) counts as accepted and keeps the manager's pending guard.
        """
        if not orders:
            return True
        return all(
            o.status in _TERMINAL_UNFILLED_STATUSES and o.filled_quantity == Decimal(0)
            for o in orders
        )

    async def _submit_exit_signal(
        self,
        manager: BracketExitManager | TrailingStopManager,
        exit_signal: Signal,
        current_bars: dict[str, OHLCVBar],
    ) -> tuple[int, int]:
        """Submit a bracket/trailing exit and route its fills; R-04 safe.

        Returns ``(orders, fills)``.  When nothing was submitted (empty order
        list, every order terminal and unfilled, or ``process_signal`` raised
        before returning) the manager's pending guard is cleared at once and
        ``engine.exit_not_submitted`` is logged, so the next evaluation retries
        instead of waiting out the 3-bar TTL.  An accepted order keeps the
        guard.  A double sell stays impossible: the live engine's in-flight
        reservation runs before any exchange call.
        """
        symbol = exit_signal.symbol
        trigger = exit_signal.metadata.get("trigger")
        try:
            orders = await self._execution_engine.process_signal(exit_signal)
        except Exception:
            manager.clear_pending(symbol)
            self._log.error(
                "engine.exit_not_submitted",
                symbol=symbol,
                trigger=trigger,
                reason="process_signal_raised",
            )
            raise
        if self._exit_not_submitted(orders):
            manager.clear_pending(symbol)
            self._log.warning(
                "engine.exit_not_submitted",
                symbol=symbol,
                trigger=trigger,
                reason="no_orders" if not orders else "orders_terminal_unfilled",
            )
        fill_count, _ = await self._route_exit_fills(orders, current_bars, exit_signal)
        return len(orders), fill_count

    def _track_lot_protection(self, symbol: str, position: Position | None) -> None:
        """FX-06 (SEC-01 C-1.3): make a held lot WITHOUT any protective level loud.

        Called at step 5a after ``check`` returned no exit.  A held lot is
        protected when the bracket anchor has a stop OR the trailing mark exists.
        Otherwise the streak grows, ``lot_unprotected_bars_total`` is incremented
        every bar and ``bracket_exit.lot_unprotected`` is logged at error level on
        the first bar of the streak and every 10th bar after.  The streak resets
        when the lot is protected or flat.  PAPER/LIVE only; a bracket without a
        stop-loss leg is never reported (WP13BA-CR-201).
        """
        if (
            position is None
            or position.is_flat
            or self._bracket_exit is None
            or self._run_mode == RunMode.BACKTEST
            or not self._bracket_exit.has_stop_loss
        ):
            self._unprotected_bars.pop(symbol, None)
            return
        protected = self._bracket_exit.protective_level(symbol, position) is not None or (
            self._trailing_stop is not None
            and self._trailing_stop.protective_level(symbol, position) is not None
        )
        if protected:
            self._unprotected_bars.pop(symbol, None)
            return
        streak = self._unprotected_bars.get(symbol, 0) + 1
        self._unprotected_bars[symbol] = streak
        self._metric("lot_unprotected_bars_total")
        if streak % _UNPROTECTED_LOG_EVERY == 1:
            self._log.error(
                "bracket_exit.lot_unprotected",
                symbol=symbol,
                bars_unprotected=streak,
            )

    def _record_signal_bar(
        self,
        orders: Sequence[Order],
        symbol: str,
        history_by_symbol: dict[str, list[OHLCVBar]],
    ) -> None:
        """BACKTEST: remember which bar was being processed when a BUY order
        was created (in memory only; no clock is read)."""
        history = history_by_symbol.get(symbol)
        if not history:
            return
        for order in orders:
            if order.side == OrderSide.BUY:
                self._order_signal_bar_ts[order.order_id] = history[-1].timestamp

    def _entry_history(self, symbol: str) -> list[OHLCVBar]:
        """History the entry hook anchors against: the bar being processed,
        else the rolling live window."""
        current = self._current_history
        if current is not None and symbol in current:
            return current[symbol]
        return self._bar_windows.get(symbol, [])

    def _metric(self, name: str, **labels: str) -> None:
        """Increment a run-labelled counter; metrics never raise."""
        try:
            from common.metrics import metrics as _mc

            _mc.increment(
                name, labels={"run_id": self._run_id or "unknown", **labels}
            )
        except Exception:
            self._log.debug("engine.metrics_update_failed", exc_info=True)

    def _atr_window_size(self) -> int:
        """``W = 6 * period + 1`` bars: the bounded Wilder window (CR-005)."""
        period = self._bracket_exit.atr_period if self._bracket_exit is not None else 14
        return period * _ATR_WINDOW_MULTIPLIER + 1

    def _window_ending_at(
        self, bars: Sequence[OHLCVBar], end_ts: datetime | None
    ) -> list[OHLCVBar] | None:
        """Exactly ``W`` contiguous bars ending at ``end_ts`` (inclusive), or None."""
        if end_ts is None:
            return None
        n = self._atr_window_size()
        for i in range(len(bars) - 1, -1, -1):
            if bars[i].timestamp == end_ts:
                if i + 1 < n:
                    return None
                window = list(bars[i + 1 - n : i + 1])
                return window if window_is_contiguous(window, self._tf_s) else None
            if bars[i].timestamp < end_ts:
                return None
        return None

    async def _fetch_exact_window(
        self, symbol: str, end_ts: datetime
    ) -> list[OHLCVBar] | None:
        """One bounded ``fetch_ohlcv`` (10 s) for the ``W`` bars ending at ``end_ts``.

        ``None`` on any failure or when the answer is not exactly ``W``
        contiguous bars ending there.  The whole post-processing is inside the
        guarded block (WP13BA-SEC-03): a non-list answer or a non-``OHLCVBar``
        element is ``MalformedWindow``.  Logs ``error_type`` only.
        """
        n = self._atr_window_size()
        since = end_ts - timedelta(seconds=self._tf_s * (n - 1))
        try:
            bars = await asyncio.wait_for(
                self._market_data.fetch_ohlcv(
                    symbol=symbol, timeframe=self._timeframe, since=since, limit=n
                ),
                timeout=_EXIT_FETCH_TIMEOUT_S,
            )
            if not isinstance(bars, list) or not all(isinstance(b, OHLCVBar) for b in bars):
                self._exit_fetch_failed(symbol, "MalformedWindow")
                return None
            window = [b for b in bars if b.timestamp <= end_ts][-n:]
            if (
                len(window) != n
                or window[-1].timestamp != end_ts
                or not window_is_contiguous(window, self._tf_s)
            ):
                self._exit_fetch_failed(
                    symbol, "IncompleteWindow", got=len(window), expected=n
                )
                return None
        except Exception as exc:
            self._exit_fetch_failed(symbol, type(exc).__name__)
            return None
        self._last_exit_fetch_error.pop(symbol, None)
        return window

    def _exit_fetch_failed(self, symbol: str, error_type: str, **extra: int) -> None:
        """Log one failed exact-window fetch (``error_type`` only, never text)."""
        self._last_exit_fetch_error[symbol] = error_type
        self._log.error(
            "engine.exit_window_fetch_failed",
            symbol=symbol,
            error_type=error_type,
            **extra,
        )

    @staticmethod
    def _interim_atr(
        history: Sequence[OHLCVBar], rule_ts: datetime | None, period: int
    ) -> Decimal | None:
        """The INTERIM ATR of a pending anchor: ``min`` of the available estimates.

        * ``ATR_best(R)``: Wilder ATR over the (up to) ``W = 6p+1`` bars ending at
          the last bar with ``ts <= R`` (``rule_ts``);
        * ``ATR_newest``: ``atr_for_bars(history[-W:])`` -- HEAD's per-bar ATR.

        Each term is ``None`` when it has <= ``period`` bars.  The minimum makes
        the interim stop never looser than HEAD would have set on that bar
        (``I <= ATR_newest``, I-2); it is an ESTIMATE of the exact ``ATR(R)``, not a
        bound (I-3).  ``None`` when neither term is computable.  Pure; no clock.
        """
        if not history:
            return None
        window_n = period * _ATR_WINDOW_MULTIPLIER + 1
        estimates: list[Decimal] = []
        newest = atr_for_bars(history[-window_n:], period)
        if newest is not None:
            estimates.append(newest)
        if rule_ts is not None:
            for i in range(len(history) - 1, -1, -1):
                if history[i].timestamp <= rule_ts:
                    best = atr_for_bars(history[max(0, i + 1 - window_n) : i + 1], period)
                    if best is not None:
                        estimates.append(best)
                    break
        return min(estimates) if estimates else None

    def _clear_anchor_pending(self, symbol: str) -> None:
        """Drop the engine-side pending-anchor state of ``symbol`` (finalised / flat)."""
        self._anchor_fetch_attempts.pop(symbol, None)
        self._anchor_origin.pop(symbol, None)
        self._last_exit_fetch_error.pop(symbol, None)

    def _log_stop_cost(self, anchor: BracketAnchor) -> None:
        """U-4: log (never drop) a stop that sits inside the entry/round-trip cost."""
        if anchor.stop_price is None or anchor.entry_basis <= Decimal(0):
            return
        d = (anchor.entry_basis - anchor.stop_price) / anchor.entry_basis
        c_side = Decimal(str(self._exit_cost_model.c_side))
        c_rt = Decimal(str(self._exit_cost_model.c_rt))
        if d <= c_side:
            self._log.warning(
                "entry.stop_inside_entry_cost",
                symbol=anchor.symbol,
                d=str(d),
                c_side=str(c_side),
                c_rt=str(c_rt),
            )
        elif d <= c_rt:
            self._log.info(
                "entry.stop_inside_round_trip_cost",
                symbol=anchor.symbol,
                d=str(d),
                c_side=str(c_side),
                c_rt=str(c_rt),
            )

    def _on_fill_routed(
        self, order: Order | None, fill: Fill, pre_fill_pos: Position | None
    ) -> None:
        """Lot / anchor lifecycle hook, called after a fill reached the portfolio.

        NEVER raises: an anchoring fault must not break fill routing (the lazy
        ``late`` anchor then protects the lot at the next bar close).  For a BUY
        that takes the symbol from flat to open it opens the lot (bracket anchor
        + trailing high-water mark); a further fill of the opening order
        re-blends the entry basis; a fill of any other order is an add and keeps
        the levels (U-6).  A SELL that leaves the symbol flat ends the lot.
        """
        try:
            self._on_fill_routed_impl(order, fill, pre_fill_pos)
        except Exception:
            self._log.exception("engine.fill_hook_error", symbol=fill.symbol)

    def _on_fill_routed_impl(
        self, order: Order | None, fill: Fill, pre_fill_pos: Position | None
    ) -> None:
        if self._bracket_exit is None and self._trailing_stop is None:
            return
        symbol = fill.symbol
        if fill.side == OrderSide.SELL:
            # Closed in full -> the lot ends (derived from the pre-fill
            # position so no extra portfolio read is needed on exits).
            if pre_fill_pos is None or fill.quantity >= pre_fill_pos.quantity:
                closed = self._lots.pop(symbol, None)
                if closed is not None and closed.opening_order_id is not None:
                    # CR-10: the signal-bar entry of a closed lot is never read again
                    self._order_signal_bar_ts.pop(closed.opening_order_id, None)
                self._clear_anchor_pending(symbol)
            return
        if fill.side != OrderSide.BUY:
            return
        position = self._portfolio.get_position(symbol)
        if position is None or position.is_flat:
            return

        was_flat = pre_fill_pos is None or pre_fill_pos.is_flat
        lot = self._lots.get(symbol)
        if was_flat:
            entry = lot_entry_from_opening_fills(symbol, order, [fill])
            if entry is None:
                return
            self._lots[symbol] = _EngineLot(
                opening_order_id=fill.order_id, order=order, fills=[fill], entry=entry
            )
            self._open_exit_lot(symbol, entry, position)
            return
        if lot is None:
            # Open position the engine never saw open (direct use, adoption):
            # the lazy ``late`` anchor handles it at the next bar close.
            return
        if fill.order_id == lot.opening_order_id:
            lot.fills.append(fill)
            entry = lot_entry_from_opening_fills(symbol, lot.order, lot.fills)
            if entry is None:
                return
            lot.entry = entry
            if self._bracket_exit is not None:
                self._bracket_exit.add_opening_fill(
                    symbol,
                    order_id=fill.order_id,
                    entry_basis=entry.entry_basis,
                    basis_qty=entry.basis_qty,
                )
            if self._trailing_stop is not None:
                self._trailing_stop.open_lot(
                    symbol,
                    lot_key=position.opened_at,
                    hwm0=entry.entry_gross_vwap,
                    anchor_bar_ts=None,
                )
            return
        # An add from another order: the first entry's levels stay (U-6).
        if self._bracket_exit is not None:
            self._bracket_exit.add_opening_fill(
                symbol,
                order_id=fill.order_id,
                entry_basis=lot.entry.entry_basis,
                basis_qty=lot.entry.basis_qty,
            )

    def _open_exit_lot(self, symbol: str, entry: LotEntry, position: Position) -> None:
        """Flat -> open: create the bracket anchor and the trailing mark."""
        history = self._entry_history(symbol)
        self._anchor_created_bar[symbol] = self._bar_count
        self._clear_anchor_pending(symbol)
        self._unprotected_bars.pop(symbol, None)
        anchor_bar_ts: datetime | None
        if self._bracket_exit is not None:
            anchor = self._anchor_for_entry(symbol, entry, history)
            anchor_bar_ts = anchor.anchor_bar_ts
            if not anchor.final:
                self._anchor_fetch_attempts[symbol] = 0
                self._anchor_origin[symbol] = "entry"
            self._log_stop_cost(anchor)
            self._metric("bracket_anchor_source_total", source=anchor.source)
        elif self._run_mode == RunMode.BACKTEST:
            anchor_bar_ts = self._backtest_signal_bar_ts(entry, history)
        else:
            anchor_bar_ts = anchor_bar_timestamp(history, entry.anchor_at, self._tf_s)
        if self._trailing_stop is not None:
            self._trailing_stop.open_lot(
                symbol,
                lot_key=position.opened_at,
                hwm0=entry.entry_gross_vwap,
                anchor_bar_ts=anchor_bar_ts,
            )

    def _backtest_signal_bar_ts(
        self, lot: LotEntry, history: Sequence[OHLCVBar]
    ) -> datetime | None:
        """BACKTEST anchor bar: the bar being processed when the opening order
        was created (in memory; falls back to the processed bar).  No clock."""
        ts = (
            self._order_signal_bar_ts.get(lot.opening_order_id)
            if lot.opening_order_id is not None
            else None
        )
        if ts is None and history:
            ts = history[-1].timestamp
        return ts

    def _anchor_for_entry(
        self, symbol: str, lot: LotEntry, history: Sequence[OHLCVBar]
    ) -> BracketAnchor:
        """Create the bracket anchor of a new lot (SY-13b-01/02).

        * BACKTEST: the in-memory signal bar and today's ATR window (it may be
          shorter than ``W`` early in a run).  Reads no wall clock.
        * PAPER/LIVE: the rule bar of ``lot.anchor_at`` and exactly ``W``
          contiguous bars ending at it.  When that window is not (yet)
          available the anchor is ``entry_provisional`` (levels from the best
          available window, finalised once later) or ``entry_deferred`` (no ATR
          at all).
        """
        bracket = self._bracket_exit
        if bracket is None:
            raise RuntimeError("_anchor_for_entry requires a bracket manager")
        position = self._portfolio.get_position(symbol)
        lot_key = position.opened_at if position is not None else lot.anchor_at
        period = bracket.atr_period
        window_n = self._atr_window_size()
        atr: Decimal | None = None
        source: AnchorSource = "entry"
        final = True
        anchor_bar_ts: datetime | None
        if self._run_mode == RunMode.BACKTEST:
            anchor_bar_ts = self._backtest_signal_bar_ts(lot, history)
            if bracket.requires_atr:
                end = len(history)
                for i in range(len(history) - 1, -1, -1):
                    if history[i].timestamp == anchor_bar_ts:
                        end = i + 1
                        break
                atr = atr_for_bars(history[max(0, end - window_n) : end], period)
                if atr is None:
                    source, final = "entry_deferred", False
        else:
            anchor_bar_ts = anchor_bar_timestamp(history, lot.anchor_at, self._tf_s)
            if bracket.requires_atr:
                window = self._window_ending_at(history, anchor_bar_ts)
                atr = atr_for_bars(window, period) if window is not None else None
                if atr is None:
                    # FX-02: the interim ATR is the MIN of the rule-bar window and
                    # the newest window (never looser than HEAD's per-bar ATR).
                    atr = self._interim_atr(history, anchor_bar_ts, period)
                    source = "entry_provisional" if atr is not None else "entry_deferred"
                    final = False
        return bracket.open_lot(
            symbol,
            lot_key=lot_key,
            opening_order_id=lot.opening_order_id,
            anchor_at=lot.anchor_at,
            anchor_bar_ts=anchor_bar_ts,
            entry_basis=lot.entry_basis,
            basis_qty=lot.basis_qty,
            atr=atr,
            source=source,
            final=final,
        )

    async def _finalize_pending_anchors(
        self, history_by_symbol: dict[str, list[OHLCVBar]]
    ) -> None:
        """Advance every non-final bracket anchor one step (PAPER/LIVE state machine).

        Runs once per bar pass (step 4c, under the bar lock) for each held symbol
        whose anchor is ``final=False`` and was created on an EARLIER pass.  The
        per-anchor procedure is :meth:`_finalize_one_anchor`.  Engine-wide I/O
        budget: at most ``_ANCHOR_FETCHES_PER_PASS`` exact-window fetch per call
        (a symbol skipped for budget spends no attempt), so a pass is never held
        for more than one ``_EXIT_FETCH_TIMEOUT_S`` by finalisation (SEC-06).
        """
        bracket = self._bracket_exit
        if bracket is None or not bracket.requires_atr:
            return
        period = bracket.atr_period
        self._anchor_fetch_budget = _ANCHOR_FETCHES_PER_PASS
        for symbol in self._symbols:
            anchor = bracket.get_anchor(symbol)
            if anchor is None or anchor.final:
                continue
            # CR-02: a closed lot has nothing left to verify -- no fetch, no log.
            position = self._portfolio.get_position(symbol)
            if position is None or position.is_flat:
                self._clear_anchor_pending(symbol)
                continue
            if self._anchor_created_bar.get(symbol, -1) >= self._bar_count:
                continue
            history = history_by_symbol.get(symbol) or []
            if not history:
                continue
            try:
                changed = await self._finalize_one_anchor(symbol, anchor, history, period)
            except Exception as exc:
                self._log.error(
                    "engine.anchor_finalize_error",
                    symbol=symbol,
                    error_type=type(exc).__name__,
                )
                continue
            if changed is not None:
                self._log_stop_cost(changed)
                self._metric("bracket_anchor_source_total", source=changed.source)

    async def _finalize_one_anchor(
        self,
        symbol: str,
        anchor: BracketAnchor,
        history: list[OHLCVBar],
        period: int,
    ) -> BracketAnchor | None:
        """One step for one pending anchor; returns the anchor it changed, if any.

        BACKTEST: only a deferred anchor finalises, on the newest window (unchanged).

        PAPER/LIVE (synthesis ``final-synthesis-1.3b-a.md`` section 4 steps 1-7;
        ``R`` = the rule bar):

        1. ``R`` = stored ``anchor_bar_ts`` else computed from ``anchor_at``
           (lazily, after a resume with an empty warmup window); none -> skip.
        2. ``exact`` = the ``W`` contiguous bars ending at ``R`` from the live
           window (no I/O).
        3. Absent, attempts < 3 and the pass budget unspent -> one bounded fetch
           (an attempt).
        4. ``atr(exact)`` computable -> FINALISE (source ``entry`` or
           ``resume_recomputed`` by origin), anchor bar ``R``.
        5. Else, an anchor without levels -> arm the INTERIM ATR once
           (``entry_provisional`` / ``resume_fallback_current_atr``).
        6. Verification over (3 attempts, or an exact window without an ATR) and
           the anchor is armed -> FINALISE UNVERIFIED, levels unchanged (error log).
        7. Otherwise stay pending; no I/O beyond step 3.
        """
        bracket = self._bracket_exit
        if bracket is None:
            return None
        newest_ts = history[-1].timestamp
        if self._run_mode == RunMode.BACKTEST:
            if anchor.source != "entry_deferred":
                return None
            atr = atr_for_bars(history, period)
            if atr is None:
                return None
            return bracket.finalize_anchor(
                symbol, atr=atr, anchor_bar_ts=newest_ts, source="entry_deferred"
            )

        origin = self._anchor_origin.get(symbol, "entry")
        rule_ts = anchor.anchor_bar_ts
        if rule_ts is None and anchor.anchor_at is not None:
            rule_ts = anchor_bar_timestamp(history, anchor.anchor_at, self._tf_s)
        if rule_ts is None:
            return None  # step 1: nothing to anchor against yet

        exact = self._window_ending_at(history, rule_ts)  # step 2
        attempts = self._anchor_fetch_attempts.get(symbol, 0)
        if (
            exact is None
            and attempts < _ANCHOR_FETCH_MAX_ATTEMPTS
            and self._anchor_fetch_budget > 0
        ):
            attempts += 1  # step 3
            self._anchor_fetch_attempts[symbol] = attempts
            self._anchor_fetch_budget -= 1
            exact = await self._fetch_exact_window(symbol, rule_ts)
        exact_atr = atr_for_bars(exact, period) if exact is not None else None

        if exact_atr is not None:  # step 4
            done = bracket.finalize_anchor(
                symbol,
                atr=exact_atr,
                anchor_bar_ts=rule_ts,
                source="entry" if origin == "entry" else "resume_recomputed",
            )
            self._clear_anchor_pending(symbol)
            return done

        changed: BracketAnchor | None = None
        armed = anchor.atr is not None
        if not armed:  # step 5
            interim = self._interim_atr(history, rule_ts, period)
            if interim is not None:
                interim_source: AnchorSource = (
                    "entry_provisional" if origin == "entry" else "resume_fallback_current_atr"
                )
                changed = bracket.arm_interim(
                    symbol, atr=interim, anchor_bar_ts=rule_ts, source=interim_source
                )
                if changed is not None:
                    armed = True
                    self._log.warning(
                        "bracket_exit.interim_armed",
                        symbol=symbol,
                        source=interim_source,
                        attempts=attempts,
                        error_type=self._last_exit_fetch_error.get(symbol),
                    )

        verification_over = attempts >= _ANCHOR_FETCH_MAX_ATTEMPTS or exact is not None
        if verification_over and armed:  # step 6
            if origin == "entry":
                self._log.error(
                    "bracket_exit.entry_unverified",
                    symbol=symbol,
                    source="entry_unverified",
                    attempts=attempts,
                    error_type=self._last_exit_fetch_error.get(symbol),
                    note="could not verify the entry ATR; keeping the interim levels",
                )
                final_source: AnchorSource = "entry_unverified"
            else:
                self._log.error(
                    "bracket_exit.resume_fallback_final",
                    symbol=symbol,
                    source="resume_fallback_current_atr",
                    attempts=attempts,
                    error_type=self._last_exit_fetch_error.get(symbol),
                    note="entry ATR could not be recomputed; keeping the interim levels",
                )
                final_source = "resume_fallback_current_atr"
            done = bracket.finalize_anchor(
                symbol, atr=None, anchor_bar_ts=None, source=final_source
            )
            self._clear_anchor_pending(symbol)
            return done if done is not None else changed
        return changed  # step 7

    # ------------------------------------------------------------------
    # WP1.3b-a: resume (prime + restore)
    # ------------------------------------------------------------------

    def prime_resume_exit_state(
        self, fills: Sequence[Fill], orders: Sequence[Order]
    ) -> None:
        """Remember each open lot's entry facts from the persisted history.

        Pure and synchronous; called by the orchestrator BEFORE ``start()`` at
        the paper boot-resume and the live ``POST /resume`` sites.  The result
        is consumed by :meth:`_restore_exit_anchors` at the end of the warmup.
        Never raises (a symbol that fails is simply not primed).
        """
        self._primed_lots.clear()
        for symbol in self._symbols:
            try:
                opening_fills = open_lot_fills(fills, symbol)
                if not opening_fills:
                    continue
                order = next(
                    (o for o in orders if o.order_id == opening_fills[0].order_id), None
                )
                entry = lot_entry_from_opening_fills(symbol, order, opening_fills)
                if entry is None:
                    continue
                self._primed_lots[symbol] = _EngineLot(
                    opening_order_id=entry.opening_order_id,
                    order=order,
                    fills=list(opening_fills),
                    entry=entry,
                )
            except Exception:
                self._log.error("engine.resume_prime_failed", symbol=symbol, exc_info=True)

    async def _restore_exit_anchors(self) -> None:
        """Rebuild the bracket anchors and trailing marks of every open position.

        Replaces the pre-WP1.3b trailing-only seed block.  Runs at the end of
        ``_warmup_bar_windows``, before the first bar evaluation.  Protection is
        never dropped: a failed recompute falls back to the current ATR or to a
        deferred anchor (trailing stop only until finalised).
        """
        if self._bracket_exit is None and self._trailing_stop is None:
            self._primed_lots.clear()
            return
        for symbol in self._symbols:
            try:
                position = self._portfolio.get_position(symbol)
                if position is None or position.is_flat:
                    continue
                await self._restore_symbol_exit_state(symbol, position)
            except Exception as exc:
                self._log.error(
                    "engine.exit_restore_failed",
                    symbol=symbol,
                    error_type=type(exc).__name__,
                )
        self._primed_lots.clear()

    async def _restore_symbol_exit_state(self, symbol: str, position: Position) -> None:
        lot = self._primed_lots.get(symbol)
        if lot is None:
            # Not primed (legacy caller): the position's own numbers.
            entry = LotEntry(
                symbol=symbol,
                opening_order_id=None,
                anchor_at=position.opened_at,
                entry_basis=position.average_entry_price,
                entry_gross_vwap=position.average_entry_price,
                basis_qty=position.quantity,
            )
            lot = _EngineLot(opening_order_id=None, order=None, fills=[], entry=entry)
        self._lots[symbol] = lot
        entry = lot.entry
        bars = self._bar_windows.get(symbol, [])
        rule_ts = anchor_bar_timestamp(bars, entry.anchor_at, self._tf_s)
        self._anchor_created_bar[symbol] = self._bar_count
        self._clear_anchor_pending(symbol)
        self._unprotected_bars.pop(symbol, None)

        # SEC-03: the bracket half and the trailing half are independent
        # protective mechanisms -- a fault in one never skips the other.
        bracket = self._bracket_exit
        if bracket is not None:
            try:
                await self._restore_bracket_anchor(symbol, position, lot, bars, rule_ts)
            except Exception as exc:
                # The lazy ``late`` anchor then protects the lot at the next bar.
                self._log.error(
                    "engine.exit_restore_failed",
                    symbol=symbol,
                    part="bracket",
                    error_type=type(exc).__name__,
                )

        trailing = self._trailing_stop
        if trailing is not None:
            try:
                seed = await self._resume_trailing_peak(symbol, bars, rule_ts, entry)
                trailing.open_lot(
                    symbol,
                    lot_key=position.opened_at,
                    hwm0=seed,
                    anchor_bar_ts=rule_ts,
                )
                trailing.seed_peak(symbol, seed)
            except Exception as exc:
                self._log.error(
                    "engine.exit_restore_failed",
                    symbol=symbol,
                    part="trailing",
                    error_type=type(exc).__name__,
                )
                self._seed_trailing_after_failure(symbol, position, entry, bars, rule_ts)

    async def _restore_bracket_anchor(
        self,
        symbol: str,
        position: Position,
        lot: _EngineLot,
        bars: Sequence[OHLCVBar],
        rule_ts: datetime | None,
    ) -> None:
        """Install the bracket anchor of a resumed lot (the bracket half of the restore)."""
        bracket = self._bracket_exit
        if bracket is None:
            return
        entry = lot.entry
        atr: Decimal | None = None
        source: AnchorSource = "resume_recomputed"
        final = True
        if bracket.requires_atr:
            atr, source, final = await self._resume_atr(
                symbol, bars, rule_ts, bracket.atr_period
            )
        stop, tp = bracket.projected_levels(entry.entry_basis, atr)
        anchor = BracketAnchor(
            symbol=symbol,
            lot_key=position.opened_at,
            opening_order_id=entry.opening_order_id,
            anchor_at=entry.anchor_at,
            anchor_bar_ts=rule_ts,
            entry_basis=entry.entry_basis,
            basis_qty=entry.basis_qty,
            atr=atr,
            stop_price=stop,
            take_profit_price=tp,
            source=source,
            final=final,
        )
        bracket.restore_anchor(anchor)
        self._log_stop_cost(anchor)
        self._metric("bracket_anchor_source_total", source=source)
        if final:
            self._clear_anchor_pending(symbol)
        else:
            self._anchor_origin[symbol] = "resume"
            if anchor.atr is not None:
                self._log.warning(
                    "bracket_exit.interim_armed",
                    symbol=symbol,
                    source=source,
                    attempts=self._anchor_fetch_attempts.get(symbol, 0),
                    error_type=self._last_exit_fetch_error.get(symbol),
                )

    def _seed_trailing_after_failure(
        self,
        symbol: str,
        position: Position,
        entry: LotEntry,
        bars: Sequence[OHLCVBar],
        rule_ts: datetime | None,
    ) -> None:
        """HEAD's WP1.8a seed rule when the trailing restore raised (SEC-03, R04).

        ``max(entry gross VWAP, max close of the warmup bars with ts >= the
        position's opened_at)``.  Never the whole window (a pre-entry close could
        put the trail above the market).  If this raises too, the lazy seed stays.
        """
        trailing = self._trailing_stop
        if trailing is None:
            return
        try:
            seed = max(
                [
                    entry.entry_gross_vwap,
                    *(b.close for b in bars if b.timestamp >= position.opened_at),
                ]
            )
            trailing.open_lot(
                symbol,
                lot_key=position.opened_at,
                hwm0=seed,
                anchor_bar_ts=rule_ts,
            )
            trailing.seed_peak(symbol, seed)
            self._log.error(
                "trailing_stop.seed_partial",
                symbol=symbol,
                reason="restore_failed",
                note="seeded from the closes available in the warmup window",
            )
            self._metric("trailing_seed_partial_total")
        except Exception as exc:
            self._log.error(
                "engine.exit_restore_failed",
                symbol=symbol,
                part="trailing_fallback",
                error_type=type(exc).__name__,
            )

    async def _resume_atr(
        self,
        symbol: str,
        bars: Sequence[OHLCVBar],
        rule_ts: datetime | None,
        period: int,
    ) -> tuple[Decimal | None, AnchorSource, bool]:
        """ATR0 of a resumed lot: the ``W`` bars ending at the rule bar, from the
        warmup window or one bounded fetch (attempt 1 of ``_ANCHOR_FETCH_MAX_ATTEMPTS``).

        Not recomputable -> the INTERIM ATR (``min`` of the available estimates,
        :meth:`_interim_atr`) as a NON-final ``resume_fallback_current_atr`` anchor:
        :meth:`_finalize_one_anchor` retries the exact window on later passes and
        either finalises it (``resume_recomputed``) or settles on the interim
        levels.  No ATR at all -> ``entry_deferred`` (no levels yet).
        """
        window = self._window_ending_at(bars, rule_ts)
        if window is None and rule_ts is not None:
            self._anchor_fetch_attempts[symbol] = 1
            window = await self._fetch_exact_window(symbol, rule_ts)
        if window is not None:
            atr = atr_for_bars(window, period)
            if atr is not None:
                return atr, "resume_recomputed", True
        interim = self._interim_atr(bars, rule_ts, period)
        if interim is not None:
            return interim, "resume_fallback_current_atr", False
        self._log.error(
            "bracket_exit.resume_anchor_deferred",
            symbol=symbol,
            note="ATR not computable; trailing stop only until the anchor is finalised",
        )
        return None, "entry_deferred", False

    async def _resume_trailing_peak(
        self,
        symbol: str,
        bars: Sequence[OHLCVBar],
        rule_ts: datetime | None,
        entry: LotEntry,
    ) -> Decimal:
        """``max(HWM_0, max close of bars after the anchor bar)`` for a resume.

        When the warmup window starts after ``anchor bar + 1`` the gap is
        fetched once (``since = anchor_bar_ts + tf``, at most 1000 bars, 10 s);
        on failure the available closes are used and ``trailing_stop.seed_partial``
        is logged.
        """
        peak = entry.entry_gross_vwap
        if rule_ts is None:
            # FX-08: no warmup bars -> HWM_0 only; say so (looser than live).
            self._log.warning(
                "trailing_stop.seed_partial",
                symbol=symbol,
                reason="no_warmup_bars",
                note="no warmup bars; seeded from the entry fill only",
            )
            self._metric("trailing_seed_partial_total")
            return peak
        tf = timedelta(seconds=self._tf_s)
        first_wanted = rule_ts + tf
        closes: list[Decimal] = [b.close for b in bars if b.timestamp > rule_ts]
        if bars and bars[0].timestamp > first_wanted:
            missing = (bars[0].timestamp - first_wanted) // tf
            gap: list[OHLCVBar] | None = None
            if missing <= _TRAILING_GAP_FETCH_LIMIT:
                try:
                    fetched = await asyncio.wait_for(
                        self._market_data.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=self._timeframe,
                            since=first_wanted,
                            limit=missing,
                        ),
                        timeout=_EXIT_FETCH_TIMEOUT_S,
                    )
                    gap = [
                        b for b in fetched if first_wanted <= b.timestamp < bars[0].timestamp
                    ]
                except Exception as exc:
                    self._log.error(
                        "engine.exit_gap_fetch_failed",
                        symbol=symbol,
                        error_type=type(exc).__name__,
                    )
            if gap is not None and len(gap) == missing:
                closes.extend(b.close for b in gap)
            else:
                if gap:
                    closes.extend(b.close for b in gap)
                self._log.warning(
                    "trailing_stop.seed_partial",
                    symbol=symbol,
                    missing_bars=missing,
                    note="seeded from the closes available",
                )
                self._metric("trailing_seed_partial_total")
        return max([peak, *closes])

    # ------------------------------------------------------------------
    # Resting order check (paper engine)
    # ------------------------------------------------------------------

    async def _check_resting_orders(
        self,
        current_bars: dict[str, OHLCVBar],
    ) -> None:
        """
        Check and fill resting limit orders against current bar prices.

        Only applicable to PaperExecutionEngine which exposes
        ``check_resting_orders(symbol, price)``.

        Parameters
        ----------
        current_bars :
            Latest bar for each symbol.
        """
        check_fn = getattr(
            self._execution_engine, "check_resting_orders", None
        )
        if check_fn is None:
            return

        for symbol, bar in current_bars.items():
            try:
                filled_orders = await check_fn(symbol, bar.close)
                if filled_orders:
                    for order in filled_orders:
                        fills = await self._execution_engine.get_fills(
                            order.order_id
                        )
                        for fill in fills:
                            pre_fill_pos = self._portfolio.get_position(
                                fill.symbol
                            )
                            self._portfolio.update_position(
                                fill, bar.close
                            )
                            # WP1.3b-a: lot / anchor lifecycle (never raises).
                            self._on_fill_routed(order, fill, pre_fill_pos)
                            # TODO: track Order ->strategy_id mapping for
                            # correct multi-strategy attribution on resting fills.
                            self._record_trade_if_closed(
                                fill=fill,
                                pre_fill_position=pre_fill_pos,
                                strategy_id=self._strategies[0].strategy_id,
                                signal_metadata=None,  # Resting order: no signal metadata
                            )
                            self._route_fill_to_risk_manager(
                                fill, bar.close
                            )
                        self._total_fills += len(fills)
                    self._total_orders += len(filled_orders)
            except Exception:
                self._log.exception(
                    "engine.resting_order_check_error",
                    symbol=symbol,
                )

    # ------------------------------------------------------------------
    # WP1.7a: flatten -- sell every held symbol through the same capped
    # process_signal path a strategy SELL uses (D15/J1/I1). Called by
    # stop_run (flatten=true), emergency_stop_run (flatten=true) and the
    # global kill switch's own optional flatten pass.
    # ------------------------------------------------------------------

    async def flatten(self, reason: str, timeout_s: float = 30.0) -> FlattenResult:
        """Sell down every symbol this run holds, to flat or dust.

        Precondition (I7): the kill switch MUST already be active (the
        caller is responsible for latching it -- e.g. via
        ``risk_manager.trigger_kill_switch("stop_in_progress")`` -- BEFORE
        calling this) -- flatten never runs while entries are still open,
        so a fresh BUY can never re-open a position this call is in the
        middle of closing. Raises :class:`FlattenPreconditionError`
        otherwise.

        Holds ``_cycle_lock`` for the ENTIRE call (I6/SY-04): waiting for
        the lock counts against ``timeout_s``, exactly like the
        arch-design spec requires, so a flatten queued behind an
        in-progress bar still respects its own deadline rather than
        blocking indefinitely.

        Every SELL goes through ``process_signal`` with
        ``strategy_id="operator_flatten"`` and ``target_position=0`` (a
        full-close signal) -- the WP1.11a external-coin-safe cap, the
        WP1.4b idempotent submit and the WP1.2 kill-switch bypass all
        apply unchanged (I1): flatten can never sell more than
        ``min(own, balance)``, and it never touches a coin the run never
        bought.

        Parameters
        ----------
        reason:
            Human-readable cause, echoed into the ``run_flatten`` audit
            row and every per-symbol SELL signal's metadata.
        timeout_s:
            Overall wall-clock budget across every symbol (default 30s,
            matching the UI's own flatten-aware timeout, AC8).

        Returns
        -------
        FlattenResult:
            ``outcome`` is ``"noop"`` when every symbol was already flat,
            ``"flattened"`` when every symbol reached ``flat``/``dust``,
            ``"partial"`` when at least one symbol sold something but did
            not fully clear, and ``"failed"`` when nothing was reduced at
            all. ``complete`` is the boolean the caller should gate a
            normal stop on (I8: never cancel the task while incomplete).
        """
        if self._risk_manager.kill_switch_active is not True:
            raise FlattenPreconditionError(
                "flatten() requires the kill switch to already be active "
                f"(I7); run_id={self._run_id!r}, reason={reason!r}"
            )

        deadline = time.monotonic() + timeout_s

        # WP1.7a round 2 (S-06): the deadline covers ACQUIRING the lock,
        # not just the per-symbol work after it -- a hung _process_bar
        # (or a hung exchange call inside it) must never let flatten()
        # block past its own advertised timeout_s. If the lock can't be
        # acquired in time, nothing below has touched the execution
        # engine at all, so every held symbol is reported "in_flight"
        # with cause "lock_timeout" (never "failed": the bot did not
        # even attempt a SELL, so this is not a hard failure).
        remaining = max(0.0, deadline - time.monotonic())
        try:
            await asyncio.wait_for(self._cycle_lock.acquire(), timeout=remaining)
        except TimeoutError:
            symbol_results = [self._lock_timeout_result(symbol) for symbol in self._symbols]
            return self._finalize_flatten_result(reason, symbol_results)

        try:
            symbol_results = [
                await self._flatten_symbol(symbol, reason, deadline)
                for symbol in self._symbols
            ]
        finally:
            self._cycle_lock.release()

        return self._finalize_flatten_result(reason, symbol_results)

    def _lock_timeout_result(self, symbol: str) -> FlattenSymbolResult:
        """WP1.7a round 2 (S-06): best-effort per-symbol result when
        ``_cycle_lock`` could not be acquired before the deadline."""
        position = self._portfolio.get_position(symbol)
        held = position.quantity if position is not None and not position.is_flat else Decimal("0")
        if held <= Decimal("0"):
            return FlattenSymbolResult(
                symbol=symbol,
                status="no_position",
                cause=None,
                held_before=Decimal("0"),
                sold_qty=Decimal("0"),
                remaining_qty=Decimal("0"),
            )
        return FlattenSymbolResult(
            symbol=symbol,
            status="in_flight",
            cause="lock_timeout",
            held_before=held,
            sold_qty=Decimal("0"),
            remaining_qty=held,
        )

    def _finalize_flatten_result(
        self, reason: str, symbol_results: list[FlattenSymbolResult]
    ) -> FlattenResult:
        """Compute outcome/complete from per-symbol results and log
        (extracted so both the normal path and the lock-timeout early
        return in :meth:`flatten` share identical outcome semantics)."""
        non_flat_results = [r for r in symbol_results if r.status != "no_position"]
        if not non_flat_results:
            outcome = "noop"
            complete = True
        else:
            complete = all(r.status in ("flat", "dust") for r in non_flat_results)
            if complete:
                outcome = "flattened"
            else:
                # "failed" is reserved for a run where NOTHING recognisably
                # safe happened anywhere -- every other incomplete case
                # (partial fills, an order still in flight, a ledger-doubt
                # block) is "partial": the operator can retry or
                # investigate, but it is not a hard failure (risk-design
                # WP17-R-08/R-17).
                outcome = (
                    "failed"
                    if all(r.status == "failed" for r in non_flat_results)
                    else "partial"
                )

        result = FlattenResult(
            run_id=self._run_id or "",
            outcome=outcome,
            complete=complete,
            symbols=symbol_results,
        )

        log_fields = {
            "reason": reason,
            "outcome": outcome,
            "symbols": [
                {
                    "symbol": r.symbol,
                    "status": r.status,
                    "cause": r.cause,
                    "held_before": str(r.held_before),
                    "sold_qty": str(r.sold_qty),
                    "remaining_qty": str(r.remaining_qty),
                }
                for r in symbol_results
            ],
        }
        if not complete:
            self._log.critical("flatten.incomplete", **log_fields)
        else:
            self._log.warning("flatten.complete", **log_fields)
        return result

    async def _flatten_symbol(
        self, symbol: str, reason: str, deadline: float
    ) -> FlattenSymbolResult:
        """Flatten a single symbol -- see :meth:`flatten` for the contract.

        Re-reads the live position from the portfolio before every
        attempt (never a cached quantity) so a fill routed by a
        concurrently-completing order, or a partial fill from this
        method's own previous attempt, is always reflected.
        """
        position = self._portfolio.get_position(symbol)
        held_before = (
            position.quantity if position is not None and not position.is_flat else Decimal("0")
        )
        if held_before <= Decimal("0"):
            return FlattenSymbolResult(
                symbol=symbol,
                status="no_position",
                cause=None,
                held_before=Decimal("0"),
                sold_qty=Decimal("0"),
                remaining_qty=Decimal("0"),
            )

        # Best-effort snapshot of the last known bar per symbol, used only
        # for fill pricing (P&L bookkeeping) -- flatten can run outside a
        # poll cycle, so there is no guaranteed "current" bar the way
        # _process_bar always has one.
        current_bars = {s: w[-1] for s, w in self._bar_windows.items() if w}

        order_ids: list[str] = []
        sold_qty = Decimal("0")
        cause: str | None = None
        error: str | None = None
        attempts = 0

        # WP1.7a round 2 (S-15): a public, narrow accessor -- never the
        # general-purpose reconcile_required dict, which an unrelated
        # reason (e.g. a stale BUY-side flag) could also set. Engines
        # that expose no such concept (paper/backtest) never report
        # ledger doubt.
        ledger_doubt_fn = getattr(self._execution_engine, "ledger_doubt", None)

        def _ledger_doubt_now() -> bool:
            if not callable(ledger_doubt_fn):
                return False
            try:
                return bool(ledger_doubt_fn(symbol))
            except Exception:
                return False

        while True:
            position = self._portfolio.get_position(symbol)
            remaining = (
                position.quantity
                if position is not None and not position.is_flat
                else Decimal("0")
            )
            if remaining <= _FLATTEN_DUST_TOLERANCE:
                status = "flat" if remaining <= Decimal("0") else "dust"
                return FlattenSymbolResult(
                    symbol=symbol,
                    status=status,
                    # S-15: a complete (flat/dust) result carries no
                    # cause -- any cause value leftover from an earlier,
                    # since-resolved attempt this loop made is stale and
                    # must not be reported as if it were still true.
                    cause=None,
                    held_before=held_before,
                    sold_qty=sold_qty,
                    remaining_qty=max(remaining, Decimal("0")),
                    order_ids=order_ids,
                    error=error,
                )

            ledger_doubt = _ledger_doubt_now()

            if time.monotonic() >= deadline or attempts >= _FLATTEN_MAX_ATTEMPTS:
                # S-15: an order-status-derived cause from THIS flatten's
                # own most recent attempt (rejected/submit_unknown/
                # timeout_open) is more specific and more actionable than
                # the engine-level ledger-doubt check, so it always wins
                # when we have one; ledger_doubt/timeout_open are only
                # ever a fallback for "we have no specific order to
                # point at".
                if cause is None:
                    cause = "ledger_doubt" if ledger_doubt else "timeout_open"
                # sold_qty > 0 always wins as "partial" (real progress was
                # made, whatever blocked the rest). A ledger doubt with
                # NOTHING sold is still "partial", never "failed" -- the
                # bot correctly refused to sell against a balance it is
                # not sure of (risk-design WP17-R-17); "in_flight" covers
                # an order that genuinely still exists somewhere
                # (unconfirmed submit, or reserved by another SELL);
                # "failed" is reserved for a rejected order or a raised
                # exception with nothing sold.
                if sold_qty > Decimal("0") or cause == "ledger_doubt":
                    status = "partial"
                elif cause in ("timeout_open", "submit_unknown", "inflight_other"):
                    status = "in_flight"
                else:
                    status = "failed"
                return FlattenSymbolResult(
                    symbol=symbol,
                    status=status,
                    cause=cause,
                    held_before=held_before,
                    sold_qty=sold_qty,
                    remaining_qty=remaining,
                    order_ids=order_ids,
                    error=error,
                )

            attempts += 1
            signal = Signal(
                strategy_id="operator_flatten",
                symbol=symbol,
                direction=SignalDirection.SELL,
                target_position=Decimal("0"),
                confidence=1.0,
                metadata={"exit_reason": "flatten", "flatten_reason": reason},
            )
            try:
                orders = await self._execution_engine.process_signal(signal)
            except Exception as exc:  # flatten must never raise
                self._log.exception(
                    "engine.flatten_signal_error", symbol=symbol, reason=reason
                )
                error = str(exc)
                cause = "error"
                orders = []

            if orders:
                order_ids.extend(str(o.order_id) for o in orders)
                _, filled_qty = await self._route_exit_fills(orders, current_bars, signal)
                sold_qty += filled_qty
                # S-15: derive cause from THIS attempt's order status
                # first -- overwrite unconditionally (the most recent
                # attempt is always the most authoritative signal),
                # never leave a prior attempt's now-stale cause in place.
                attempt_cause: str | None = None
                for o in orders:
                    status_value = o.status.value
                    if status_value == "rejected":
                        attempt_cause = "rejected"
                    elif status_value == "pending_submit":
                        attempt_cause = attempt_cause or "submit_unknown"
                    elif status_value in ("open", "partial"):
                        attempt_cause = attempt_cause or "timeout_open"
                if attempt_cause is not None:
                    cause = attempt_cause
                continue

            # No order was created this attempt: either genuinely blocked
            # (ledger doubt, another SELL already in flight for this
            # symbol) or transiently rejected upstream. Wait briefly and
            # re-read the position rather than busy-looping.
            cause = "ledger_doubt" if ledger_doubt else "inflight_other"

            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                continue
            await asyncio.sleep(min(_FLATTEN_POLL_SECONDS, remaining_time))

    # ------------------------------------------------------------------
    # Engine price updates (paper engine)
    # ------------------------------------------------------------------

    def _update_engine_prices(
        self,
        current_bars: dict[str, OHLCVBar],
    ) -> None:
        """
        Update last-known prices on the execution engine.

        PaperExecutionEngine requires ``set_last_price()`` to be called
        before signal processing. This method handles the dispatch.

        Parameters
        ----------
        current_bars :
            Latest bar for each symbol.
        """
        set_price_fn = getattr(
            self._execution_engine, "set_last_price", None
        )
        if set_price_fn is None:
            return

        for symbol, bar in current_bars.items():
            set_price_fn(symbol, bar.close)

    # ------------------------------------------------------------------
    # Paper/Live helpers
    # ------------------------------------------------------------------

    async def _warmup_bar_windows(self) -> None:
        """
        Fetch initial bar history for each symbol to satisfy strategy
        warm-up requirements.

        Uses ``market_data.fetch_ohlcv()`` with a limit equal to the
        configured warmup size.
        """
        fetch_limit = max(self._warmup_bars, 100)

        for symbol in self._symbols:
            try:
                bars = await self._market_data.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self._timeframe,
                    limit=fetch_limit,
                )
                self._bar_windows[symbol] = bars

                # Update engine prices with the most recent bar
                if bars:
                    set_price_fn = getattr(
                        self._execution_engine, "set_last_price", None
                    )
                    if set_price_fn is not None:
                        set_price_fn(symbol, bars[-1].close)

                self._log.info(
                    "engine.warmup_loaded",
                    symbol=symbol,
                    bars_loaded=len(bars),
                )
            except MarketDataError:
                self._log.exception(
                    "engine.warmup_fetch_failed",
                    symbol=symbol,
                )
            except Exception:
                self._log.exception(
                    "engine.warmup_unexpected_error",
                    symbol=symbol,
                )

        # WP1.3b-a (replaces the WP1.8a A-09 trailing-only seed block): after a
        # resume, recompute the bracket anchor and the trailing high-water mark
        # of every open position from the persisted history primed by
        # ``prime_resume_exit_state`` -- before the first bar evaluation.  A
        # fresh run has no open position, so this is a no-op there.  It never
        # aborts the warmup.
        try:
            await self._restore_exit_anchors()
        except Exception as exc:
            self._log.error("engine.exit_restore_failed", error_type=type(exc).__name__)

    async def _poll_and_process(self) -> None:
        """
        Fetch the latest bar for each symbol, update the rolling window,
        and process the bar through the strategy pipeline.

        Deduplicates bars by checking whether the latest fetched bar's
        timestamp matches the last bar in the window.
        """
        current_bars: dict[str, OHLCVBar] = {}
        new_bar_found = False

        for symbol in self._symbols:
            try:
                latest_bar = await self._market_data.get_latest_bar(
                    symbol=symbol,
                    timeframe=self._timeframe,
                )

                window = self._bar_windows[symbol]

                # Deduplicate: only add if timestamp is newer
                if window and latest_bar.timestamp <= window[-1].timestamp:
                    # No new bar yet for this symbol
                    current_bars[symbol] = window[-1]
                    continue

                # Append new bar and trim to max window size
                window.append(latest_bar)
                if len(window) > self._max_bars_history:
                    self._bar_windows[symbol] = window[
                        -self._max_bars_history :
                    ]

                current_bars[symbol] = latest_bar
                new_bar_found = True

            except MarketDataError:
                self._log.exception(
                    "engine.poll_fetch_failed",
                    symbol=symbol,
                )
                # Use last known bar if available
                window = self._bar_windows.get(symbol, [])
                if window:
                    current_bars[symbol] = window[-1]
            except Exception:
                self._log.exception(
                    "engine.poll_unexpected_error",
                    symbol=symbol,
                )
                window = self._bar_windows.get(symbol, [])
                if window:
                    current_bars[symbol] = window[-1]

        # Only process if we got bars for all symbols and at least one is new
        if len(current_bars) < len(self._symbols):
            self._log.warning(
                "engine.poll_incomplete",
                symbols_received=len(current_bars),
                symbols_expected=len(self._symbols),
            )
            return

        if not new_bar_found:
            return

        # Update engine prices
        self._update_engine_prices(current_bars)

        # Build history windows for strategy calls
        history_by_symbol: dict[str, list[OHLCVBar]] = {
            s: list(self._bar_windows[s]) for s in self._symbols
        }

        # Process the bar. WP1.7a (I6): held for the WHOLE call (not just
        # the get_fills sub-steps) so a concurrent flatten() can never
        # interleave with a bar still mid-flight -- e.g. both reading the
        # same not-yet-reserved own_avail, or two callers racing
        # LiveExecutionEngine.get_fills' pre-await already_routed read
        # (arch-design WP17-A-02).
        async with self._cycle_lock:
            await self._process_bar(current_bars, history_by_symbol)

    # ------------------------------------------------------------------
    # Summary / status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """
        Return the current engine status as a serialisable dictionary.

        Includes engine state, run metrics, and portfolio summary.

        Returns
        -------
        dict[str, Any]
            Engine status snapshot.
        """
        result: dict[str, Any] = {
            "state": self._state.value,
            "run_id": self._run_id,
            "run_mode": self._run_mode.value,
            "timeframe": self._timeframe.value,
            "symbols": self._symbols,
            "strategies": [s.strategy_id for s in self._strategies],
            "bar_count": self._bar_count,
            "total_signals": self._total_signals,
            "total_orders": self._total_orders,
            "total_fills": self._total_fills,
        }

        if self._state in (EngineState.RUNNING, EngineState.STOPPED):
            result["portfolio_summary"] = self._portfolio.get_summary()

        return result

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StrategyEngine("
            f"state={self._state.value!r}, "
            f"run_id={self._run_id!r}, "
            f"mode={self._run_mode.value!r}, "
            f"strategies={len(self._strategies)}, "
            f"symbols={self._symbols}, "
            f"bars={self._bar_count})"
        )
