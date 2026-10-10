"""
packages/trading/safety.py
---------------------------
Safety gates and circuit breaker for live trading protection.

This module implements two independent safety mechanisms:

1. **LiveTradingGate**  -- Three-layer activation gate that must be fully
   satisfied before any live order is submitted.  Checked by the
   ``LiveExecutionEngine`` and by the API run-creation endpoint.

2. **CircuitBreaker**  -- Graduated entry guard evaluated once per processed
   bar from a typed :class:`~trading.portfolio.RiskSnapshot` (WP1.6, C2/C3/C23).

   WP1.6 semantics (``CircuitBreaker.evaluate``):
   - REDUCE: drawdown >= 0.5 x the run limit  -> BUY quantity x 0.5
   - DAILY_LIMIT: the WP1.5 daily-loss predicate  -> BUYs dropped until the
     UTC day rolls (``snapshot.day``, never the wall clock)
   - HALT: drawdown >= the run limit  -> BUYs dropped.  HALT does NOT stop the
     run and does NOT latch: it auto-recovers (user decision U-1) after N
     consecutive evaluations below 0.9 x the limit, stepping down to REDUCE.
   - Every effect is BUY-only (C23): exits are never blocked.

   The pre-WP1.6 API (``check``, ``check_graduated``,
   ``check_daily_limit_reset``, ``get_position_size_multiplier``, ``trip``,
   ``reset``) is LEGACY -- unreachable from production code, kept only for the
   existing unit tests (removal: CF-29).

Both classes are stateless with respect to persistence  -- they evaluate
conditions from injected state at call time.  The CircuitBreaker holds
in-memory hysteresis counters; nothing is persisted (P-8): every level is
re-derived from data after a restart (restart watch, see ``CircuitBreaker``).

Design principles
-----------------
- Fail-closed: any ambiguity defaults to blocking orders (entries only).
- No operator state: HALT / REDUCE / DAILY_LIMIT are data-derived.
- No I/O and no wall-clock read in the evaluation path.
- Full audit trail via structlog at WARNING/CRITICAL severity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

import hmac
import structlog
from pydantic import BaseModel, Field, model_validator
from pydantic import SecretStr

from trading.portfolio import RiskSnapshot
from trading.risk import RiskParameters

__all__ = [
    "LiveTradingGate",
    "LiveTradingGateError",
    "GateCheckResult",
    "GateLayer",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerResponse",
    "CircuitBreakerState",
    "CircuitBreakerDecision",
    "CircuitBreakerInputs",
    "CircuitBreakerThresholds",
    "CircuitBreakerTransition",
    "DEESCALATION_EVALUATIONS",
    "HALT_EXIT_RATIO",
    "REDUCE_ENTRY_RATIO",
    "REDUCE_EXIT_RATIO",
    "REDUCE_MULTIPLIER",
    "build_run_circuit_breaker",
]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Gate check result models
# ---------------------------------------------------------------------------


class GateLayer(BaseModel):
    """Result of evaluating a single gate layer."""

    model_config = {"frozen": True}

    name: str = Field(description="Layer identifier")
    passed: bool = Field(description="True if the layer is satisfied")
    detail: str = Field(default="", description="Human-readable explanation")


class GateCheckResult(BaseModel):
    """
    Aggregate result of evaluating all three live-trading gate layers.

    ``passed`` is True only when every layer is satisfied.
    ``failures`` lists the human-readable reasons for each failed layer.
    """

    model_config = {"frozen": True}

    passed: bool = Field(description="True only if all layers pass")
    layer_results: dict[str, bool] = Field(
        description="Per-layer pass/fail mapping: layer_name -> passed"
    )
    layers: list[GateLayer] = Field(
        default_factory=list,
        description="Detailed per-layer results",
    )
    failures: list[str] = Field(
        default_factory=list,
        description="Human-readable failure reasons",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp of the gate check",
    )


# ---------------------------------------------------------------------------
# Three-layer live trading gate
# ---------------------------------------------------------------------------


class LiveTradingGate:
    """
    Three-layer safety gate for live trading activation.

    All three layers must be satisfied before any live order is submitted:

    1. **Environment**  -- ``enable_live_trading`` must be explicitly True
       in the application settings (``Settings.enable_live_trading``).
    2. **API Keys**  -- Valid exchange API key and secret must be configured
       (non-empty ``Settings.exchange_api_key`` and ``exchange_api_secret``).
    3. **Confirmation**  -- A runtime confirmation token must be provided
       and must match the stored ``live_trading_confirm_token``.

    This gate is checked by the ``LiveExecutionEngine`` on startup and
    by the API ``/runs`` endpoint before allowing a run in LIVE mode.

    Usage
    -----
    ::

        gate = LiveTradingGate()
        result = gate.check_gate(settings)
        if not result.passed:
            raise RuntimeError(f"Gate check failed: {result.failures}")

    Or use the shortcut that raises on failure::

        gate.require_gate(settings)  # raises LiveTradingGateError
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger(__name__).bind(component="live_trading_gate")

    # ------------------------------------------------------------------ #
    # Layer checks
    # ------------------------------------------------------------------ #

    def _check_environment_layer(self, settings: Any) -> GateLayer:
        """
        Layer 1: Verify the ``enable_live_trading`` flag is True.

        Parameters
        ----------
        settings:
            Application Settings object (``apps.api.config.Settings``).

        Returns
        -------
        GateLayer
        """
        enabled = getattr(settings, "enable_live_trading", False)
        if enabled:
            return GateLayer(
                name="environment",
                passed=True,
                detail="ENABLE_LIVE_TRADING is True",
            )
        return GateLayer(
            name="environment",
            passed=False,
            detail="ENABLE_LIVE_TRADING is not set to True. "
                   "Set enable_live_trading=true in environment or .env file.",
        )

    def _check_api_keys_layer(self, settings: Any) -> GateLayer:
        """
        Layer 2: Verify that exchange API key and secret are configured.

        Both ``exchange_api_key`` and ``exchange_api_secret`` must be
        non-empty.  SecretStr values are unwrapped for the emptiness check
        but never logged.

        Parameters
        ----------
        settings:
            Application Settings object.

        Returns
        -------
        GateLayer
        """
        api_key_raw = getattr(settings, "exchange_api_key", None)
        api_secret_raw = getattr(settings, "exchange_api_secret", None)

        # Unwrap SecretStr if needed
        api_key: str | None = (
            api_key_raw.get_secret_value()
            if isinstance(api_key_raw, SecretStr)
            else (api_key_raw if isinstance(api_key_raw, str) else None)
        )
        api_secret: str | None = (
            api_secret_raw.get_secret_value()
            if isinstance(api_secret_raw, SecretStr)
            else (api_secret_raw if isinstance(api_secret_raw, str) else None)
        )

        key_ok = bool(api_key and api_key.strip())
        secret_ok = bool(api_secret and api_secret.strip())

        if key_ok and secret_ok:
            return GateLayer(
                name="api_keys",
                passed=True,
                detail="Exchange API key and secret are configured",
            )

        missing: list[str] = []
        if not key_ok:
            missing.append("exchange_api_key")
        if not secret_ok:
            missing.append("exchange_api_secret")

        return GateLayer(
            name="api_keys",
            passed=False,
            detail=f"Missing or empty exchange credentials: {', '.join(missing)}. "
                   f"Configure EXCHANGE_API_KEY and EXCHANGE_API_SECRET.",
        )

    def _check_confirmation_layer(
        self,
        settings: Any,
        confirm_token: str | None = None,
    ) -> GateLayer:
        """
        Layer 3: Verify the runtime confirmation token.

        The ``live_trading_confirm_token`` in settings acts as a stored
        secret.  The caller must provide a matching ``confirm_token`` at
        runtime (e.g. via API request header or CLI flag).

        If no token is configured in settings, this layer requires that a
        non-empty token is provided (belt-and-suspenders).

        Parameters
        ----------
        settings:
            Application Settings object.
        confirm_token:
            Token provided by the operator at runtime.

        Returns
        -------
        GateLayer
        """
        stored_raw = getattr(settings, "live_trading_confirm_token", None)
        stored: str | None = (
            stored_raw.get_secret_value()
            if isinstance(stored_raw, SecretStr)
            else (stored_raw if isinstance(stored_raw, str) else None)
        )

        # No token configured in settings and none provided
        if not stored and not confirm_token:
            return GateLayer(
                name="confirmation",
                passed=False,
                detail="No confirmation token configured or provided. "
                       "Set LIVE_TRADING_CONFIRM_TOKEN in environment and "
                       "provide the matching token at runtime.",
            )

        # Token configured but none provided at runtime
        if stored and not confirm_token:
            return GateLayer(
                name="confirmation",
                passed=False,
                detail="Confirmation token configured but not provided at runtime. "
                       "Pass the confirm_token when creating a LIVE run.",
            )

        # Token provided but none configured  -- require exact match impossible
        if not stored and confirm_token:
            return GateLayer(
                name="confirmation",
                passed=False,
                detail="No confirmation token configured in settings. "
                       "Set LIVE_TRADING_CONFIRM_TOKEN to enable token verification.",
            )

        # Both present  -- constant-time comparison to prevent timing attacks.
        # Guards above guarantee both are non-None; assert to enforce the
        # invariant and fail loudly rather than silently matching empty strings.
        assert stored is not None, "stored token must be non-None at this point"
        assert confirm_token is not None, "confirm_token must be non-None at this point"
        if hmac.compare_digest(stored, confirm_token):
            return GateLayer(
                name="confirmation",
                passed=True,
                detail="Confirmation token verified",
            )

        return GateLayer(
            name="confirmation",
            passed=False,
            detail="Confirmation token does not match. "
                   "Verify the token and try again.",
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def check_gate(
        self,
        settings: Any,
        confirm_token: str | None = None,
    ) -> GateCheckResult:
        """
        Evaluate all three gate layers and return the aggregate result.

        Parameters
        ----------
        settings:
            Application Settings object (``apps.api.config.Settings``).
        confirm_token:
            Runtime confirmation token provided by the operator.

        Returns
        -------
        GateCheckResult
            ``passed=True`` only if all three layers are satisfied.
        """
        layers = [
            self._check_environment_layer(settings),
            self._check_api_keys_layer(settings),
            self._check_confirmation_layer(settings, confirm_token),
        ]

        failures = [layer.detail for layer in layers if not layer.passed]
        all_passed = len(failures) == 0

        result = GateCheckResult(
            passed=all_passed,
            layer_results={layer.name: layer.passed for layer in layers},
            layers=layers,
            failures=failures,
        )

        if all_passed:
            self._log.info("live_trading_gate.passed")
        else:
            self._log.warning(
                "live_trading_gate.failed",
                failures=failures,
                layer_results=result.layer_results,
            )

        return result

    def require_gate(
        self,
        settings: Any,
        confirm_token: str | None = None,
    ) -> None:
        """
        Evaluate all gate layers and raise if any layer fails.

        This is a convenience wrapper around ``check_gate()`` for use in
        code paths where a failed gate should be a hard error.

        Parameters
        ----------
        settings:
            Application Settings object.
        confirm_token:
            Runtime confirmation token.

        Raises
        ------
        LiveTradingGateError
            If any gate layer is not satisfied.
        """
        result = self.check_gate(settings, confirm_token)
        if not result.passed:
            raise LiveTradingGateError(
                f"Live trading gate check failed: {'; '.join(result.failures)}",
                gate_result=result,
            )


class LiveTradingGateError(RuntimeError):
    """Raised when the live trading gate check fails."""

    def __init__(self, message: str, gate_result: GateCheckResult) -> None:
        super().__init__(message)
        self.gate_result = gate_result


# ---------------------------------------------------------------------------
# Graduated circuit breaker response enum (Sprint 32)
# ---------------------------------------------------------------------------


class CircuitBreakerResponse(StrEnum):
    """
    Graduated response level from the circuit breaker.

    OK
        Normal operation  -- no restrictions.
    REDUCE
        Drawdown between reduce_drawdown_pct and max_drawdown_pct  -- scale
        down BUY position sizes by reduce_position_multiplier.
    DAILY_LIMIT
        Daily loss limit reached  -- no new BUY entries until next trading day.
        Trailing stops and SELL signals continue to fire.
    HALT
        Hard halt  -- max drawdown or consecutive losses breached.
        All new signals are blocked (equivalent to check() returning True).
    """

    OK = "ok"
    REDUCE = "reduce"
    DAILY_LIMIT = "daily_limit"
    HALT = "halt"


# ---------------------------------------------------------------------------
# Circuit breaker configuration
# ---------------------------------------------------------------------------


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for the circuit breaker thresholds.

    All percentage values are expressed as fractions (0.05 = 5%).

    Sprint 32 additions
    -------------------
    reduce_drawdown_pct:
        Drawdown level at which position sizes are reduced (soft warning).
        Must be less than max_drawdown_pct.
    reduce_position_multiplier:
        Fraction of normal position size to use when in REDUCE state.
    daily_loss_resume_next_day:
        If True, the DAILY_LIMIT state automatically clears at midnight UTC.
    """

    model_config = {"frozen": True}

    max_daily_loss_pct: float = Field(
        default=0.05,
        gt=0.0,
        le=0.50,
        description="Maximum daily loss as fraction of equity before tripping (0.05 = 5%)",
    )
    max_drawdown_pct: float = Field(
        default=0.15,
        gt=0.0,
        le=0.50,
        description="Maximum drawdown from peak equity before tripping (0.15 = 15%)",
    )
    max_consecutive_losses: int = Field(
        default=5,
        ge=2,
        le=50,
        description="Number of consecutive losing trades before tripping",
    )
    reduce_drawdown_pct: float = Field(
        default=0.10,
        gt=0.0,
        le=0.50,
        description="Drawdown threshold for graduated REDUCE response (must be < max_drawdown_pct)",
    )
    reduce_position_multiplier: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Position size multiplier applied when in REDUCE state (0.5 = 50% of normal)",
    )
    daily_loss_resume_next_day: bool = Field(
        default=True,
        description="If True, DAILY_LIMIT state clears automatically at midnight UTC",
    )

    @model_validator(mode="after")
    def validate_drawdown_ordering(self) -> CircuitBreakerConfig:
        """Ensure reduce_drawdown_pct < max_drawdown_pct."""
        if self.reduce_drawdown_pct >= self.max_drawdown_pct:
            raise ValueError(
                f"reduce_drawdown_pct ({self.reduce_drawdown_pct}) must be "
                f"less than max_drawdown_pct ({self.max_drawdown_pct})"
            )
        return self


# ---------------------------------------------------------------------------
# WP1.6: typed thresholds, inputs, transitions and decisions
# ---------------------------------------------------------------------------

#: WP1.6 §5.2 band constants.  They become per-run parameters in WP1.9 (CF-28).
HALT_EXIT_RATIO: Final[Decimal] = Decimal("0.9")
REDUCE_ENTRY_RATIO: Final[Decimal] = Decimal("0.5")
REDUCE_EXIT_RATIO: Final[Decimal] = Decimal("0.4")
REDUCE_MULTIPLIER: Final[Decimal] = Decimal("0.5")
#: Consecutive evaluations (processed bars) below the exit band before the
#: HALT -> REDUCE and REDUCE -> OK de-escalations (U-3, K-02).
DEESCALATION_EVALUATIONS: Final[int] = 3

_LIMIT_CEILING: Final[Decimal] = Decimal("0.50")
_ZERO: Final[Decimal] = Decimal(0)
_ONE: Final[Decimal] = Decimal(1)

#: ``RiskSnapshot`` Decimal fields validated (finite) before ANY state change.
_SNAPSHOT_DECIMAL_FIELDS: Final[tuple[str, ...]] = (
    "nav",
    "peak_equity",
    "drawdown_pct",
    "nav_day_start",
    "daily_loss_basis",
    "worst_nav_pnl",
)


def _require_finite_decimal(name: str, value: object) -> Decimal:
    """Return ``value`` if it is a finite ``Decimal``; raise ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class CircuitBreakerThresholds:
    """Validated, exact-``Decimal`` bands of one run's circuit breaker (§5.2).

    ``halt_entry`` is the run's ``max_drawdown_pct`` (the WP1.5 gate limit);
    every other band is derived from it in ``Decimal`` so no float drift can
    appear (SYN-02: ``0.2 * 0.9 == 0.18000000000000002`` in float).
    """

    max_daily_loss_pct: Decimal
    halt_entry: Decimal
    halt_exit: Decimal
    reduce_entry: Decimal
    reduce_exit: Decimal
    reduce_multiplier: Decimal
    deescalation_evaluations: int

    def __post_init__(self) -> None:
        for name in (
            "max_daily_loss_pct",
            "halt_entry",
            "halt_exit",
            "reduce_entry",
            "reduce_exit",
            "reduce_multiplier",
        ):
            _require_finite_decimal(name, getattr(self, name))
        if not (
            _ZERO < self.reduce_exit < self.reduce_entry < self.halt_exit < self.halt_entry
            and self.halt_entry <= _LIMIT_CEILING
        ):
            raise ValueError(
                "thresholds must satisfy 0 < reduce_exit < reduce_entry < halt_exit "
                f"< halt_entry <= 0.50, got reduce_exit={self.reduce_exit} "
                f"reduce_entry={self.reduce_entry} halt_exit={self.halt_exit} "
                f"halt_entry={self.halt_entry}"
            )
        if not (_ZERO < self.max_daily_loss_pct <= _LIMIT_CEILING):
            raise ValueError(
                f"max_daily_loss_pct must be in (0, 0.50], got {self.max_daily_loss_pct}"
            )
        if not (_ZERO < self.reduce_multiplier <= _ONE):
            raise ValueError(
                f"reduce_multiplier must be in (0, 1], got {self.reduce_multiplier}"
            )
        n = self.deescalation_evaluations
        if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= 50):
            raise ValueError(f"deescalation_evaluations must be an int in [1, 50], got {n!r}")

    @classmethod
    def from_risk_parameters(cls, params: RiskParameters) -> CircuitBreakerThresholds:
        """Derive the bands from the run's own ``RiskParameters`` (P-2).

        ``Decimal(str(float))`` is exactly what the WP1.5 gate does
        (``risk.py`` ``_check_drawdown`` / ``_check_daily_loss``), so the
        CB limit and the gate limit are the same ``Decimal``.
        """
        limit = Decimal(str(params.max_drawdown_pct))
        return cls(
            max_daily_loss_pct=Decimal(str(params.max_daily_loss_pct)),
            halt_entry=limit,
            halt_exit=limit * HALT_EXIT_RATIO,
            reduce_entry=limit * REDUCE_ENTRY_RATIO,
            reduce_exit=limit * REDUCE_EXIT_RATIO,
            reduce_multiplier=REDUCE_MULTIPLIER,
            deescalation_evaluations=DEESCALATION_EVALUATIONS,
        )

    def as_str_dict(self) -> dict[str, str]:
        """The seven threshold values as strings (API / log payload)."""
        return {
            "max_daily_loss_pct": str(self.max_daily_loss_pct),
            "halt_entry": str(self.halt_entry),
            "halt_exit": str(self.halt_exit),
            "reduce_entry": str(self.reduce_entry),
            "reduce_exit": str(self.reduce_exit),
            "reduce_multiplier": str(self.reduce_multiplier),
            "deescalation_evaluations": str(self.deescalation_evaluations),
        }


@dataclass(frozen=True, slots=True)
class CircuitBreakerInputs:
    """Run-level figures one evaluation used.  Never cached across evaluations (E-6)."""

    as_of: datetime
    day: date
    nav: Decimal
    peak_equity: Decimal
    drawdown_pct: Decimal
    gate_daily_pnl: Decimal
    daily_loss_basis: Decimal
    daily_threshold: Decimal
    day_start_source: str
    state_unknown: bool
    open_position_count: int

    def as_str_dict(self) -> dict[str, str]:
        """Allow-listed string view (API / log payload)."""
        return {
            "as_of": self.as_of.isoformat(),
            "day": self.day.isoformat(),
            "nav": str(self.nav),
            "peak_equity": str(self.peak_equity),
            "drawdown_pct": str(self.drawdown_pct),
            "gate_daily_pnl": str(self.gate_daily_pnl),
            "daily_loss_basis": str(self.daily_loss_basis),
            "daily_threshold": str(self.daily_threshold),
            "day_start_source": self.day_start_source,
            "state_unknown": str(self.state_unknown).lower(),
            "open_position_count": str(self.open_position_count),
        }


@dataclass(frozen=True, slots=True)
class CircuitBreakerTransition:
    """One change of the effective level (emitted once per change)."""

    run_id: str
    from_level: CircuitBreakerResponse
    to_level: CircuitBreakerResponse
    at: datetime  # ``snapshot.as_of`` (the risk clock), never the wall clock
    reason: str
    drawdown_pct: Decimal
    thresholds: CircuitBreakerThresholds


@dataclass(frozen=True, slots=True)
class CircuitBreakerDecision:
    """Result of one :meth:`CircuitBreaker.evaluate` call."""

    level: CircuitBreakerResponse
    active_conditions: tuple[CircuitBreakerResponse, ...]
    entries_blocked: bool
    entry_size_multiplier: Decimal  # 1 (OK) / reduce_multiplier (REDUCE) / 0 (blocked)
    reason: str | None
    inputs: CircuitBreakerInputs
    transition: CircuitBreakerTransition | None


# ---------------------------------------------------------------------------
# Circuit breaker state snapshot
# ---------------------------------------------------------------------------


class CircuitBreakerState(BaseModel):
    """Snapshot of the circuit breaker's current state for API exposure.

    The first five fields are the legacy contract (``is_tripped`` is
    ``manual trip or level == HALT``; ``graduated_response`` is the effective
    level; ``trip_reason`` / ``tripped_at`` / ``trip_count`` keep their legacy
    manual-trip meaning).  Every WP1.6 field has a default.
    """

    model_config = {"frozen": True}

    is_tripped: bool = Field(description="True if the breaker is currently open")
    trip_reason: str | None = Field(
        default=None,
        description="Reason the breaker was tripped (None if not tripped)",
    )
    tripped_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the breaker was tripped",
    )
    trip_count: int = Field(
        default=0,
        description="Total number of times the breaker has tripped in this session",
    )
    graduated_response: str = Field(
        default=CircuitBreakerResponse.OK,
        description="Current graduated response level (Sprint 32)",
    )
    # ---- WP1.6 -------------------------------------------------------- #
    level: str = Field(default="ok", description="Effective level: ok/reduce/daily_limit/halt")
    active_conditions: list[str] = Field(
        default_factory=list, description="Active conditions, most severe first"
    )
    entries_blocked: bool = Field(default=False, description="True while BUYs are dropped")
    entry_size_multiplier: str = Field(default="1", description="BUY size multiplier (Decimal)")
    halt_since: datetime | None = Field(default=None, description="Risk-clock start of the HALT")
    last_halt_reason: str | None = Field(default=None, description="Reason of the last HALT")
    halt_count: int = Field(default=0, description="HALT entries in this process")
    halt_exit_progress: int = Field(
        default=0, description="Consecutive evaluations below halt_exit (k of N)"
    )
    reduce_exit_progress: int = Field(
        default=0, description="Consecutive evaluations below reduce_exit (k of N)"
    )
    restart_watch: str = Field(
        default="none",
        description="Restart watch (CF-35): halt / reduce (exit band is the entry band) / none",
    )
    daily_limit_day: date | None = Field(default=None, description="Risk day of the DAILY_LIMIT")
    evaluations: int = Field(default=0, description="Evaluations performed")
    last_evaluated_at: datetime | None = Field(
        default=None, description="Risk-clock time of the last evaluation"
    )
    thresholds: dict[str, str] = Field(default_factory=dict, description="Bands, as strings")
    inputs: dict[str, str] | None = Field(
        default=None, description="Last evaluation inputs (allow-list), as strings"
    )


# ---------------------------------------------------------------------------
# Circuit breaker implementation
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Graduated, auto-recovering entry guard (WP1.6).

    **Production path:** :meth:`evaluate` -- called once per processed bar by
    ``StrategyEngine._process_bar`` with ``PortfolioAccounting.risk_snapshot()``.
    Escalation is immediate; HALT and REDUCE de-escalate after N consecutive
    evaluations below an exit band (HALT always steps down to REDUCE first);
    DAILY_LIMIT clears when ``snapshot.day`` advances.  All effects are
    BUY-only and nothing is persisted or operator-resettable.  Build instances
    only through :func:`build_run_circuit_breaker`.

    **Restart watch (WP1.6 CF-35, user decision "stricter"):** an instance
    cannot know the level an uninterrupted breaker would hold, so during its
    first N evaluations it enters HALT at ``halt_exit`` (0.9 L) instead of
    ``halt_entry``, and during its first 2N evaluations it enters REDUCE at
    ``reduce_exit`` (0.4 L) instead of ``reduce_entry``.  That is exactly how
    long an uninterrupted breaker that was in HALT could still be in HALT
    (N evaluations below 0.9 L) and then in REDUCE (N more below 0.4 L).
    Streaks start at 0; a level entered during the watch is then held by the
    ordinary hysteresis.  Result (WP16-T-46): from the 2N-th evaluation on, a
    new instance is never laxer than an uninterrupted one; before that only
    at dd < 0.9 L, for at most 2N-1 evaluations.  It applies to every new
    instance (resume or not): the breaker does not know which it is.

    **Legacy path (not used by production since WP1.6; removal CF-29):**
    ``check``, ``check_graduated``, ``check_daily_limit_reset``,
    ``get_position_size_multiplier``, ``trip`` and ``reset``.  They read the
    wall clock and/or stay sticky until reset; they are kept so the existing
    unit tests stay green.

    Parameters
    ----------
    config : CircuitBreakerConfig
        Legacy threshold configuration (mirror of the run's risk parameters
        when built by :func:`build_run_circuit_breaker`).
    run_id : str
        Run identifier for structured logging.
    thresholds : CircuitBreakerThresholds | None
        WP1.6 bands.  ``None`` makes :meth:`evaluate` raise ``RuntimeError``.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        run_id: str = "",
        *,
        thresholds: CircuitBreakerThresholds | None = None,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._run_id = run_id
        self._tripped: bool = False
        self._trip_reason: str | None = None
        self._tripped_at: datetime | None = None
        self._trip_count: int = 0
        self._current_response: CircuitBreakerResponse = CircuitBreakerResponse.OK
        self._daily_limit_date: date | None = None
        self._log = structlog.get_logger(__name__).bind(
            component="circuit_breaker",
            run_id=run_id,
        )
        # ---- WP1.6 machine state (in-memory only; P-8) ---------------- #
        self._thresholds: CircuitBreakerThresholds | None = thresholds
        self._halt: bool = False
        self._halt_exit_streak: int = 0
        self._reduce: bool = False
        self._reduce_exit_streak: int = 0
        self._daily_limit_day: date | None = None
        self._evaluations: int = 0
        self._halt_since: datetime | None = None
        self._halt_reason: str | None = None
        self._halt_count: int = 0
        self._last_inputs: CircuitBreakerInputs | None = None
        self._last_evaluated_at: datetime | None = None
        self._listeners: list[Callable[[CircuitBreakerTransition], None]] = []

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_tripped(self) -> bool:
        """True when the circuit breaker is open and trading is halted."""
        return self._tripped

    @property
    def config(self) -> CircuitBreakerConfig:
        """Return the breaker configuration."""
        return self._config

    @property
    def current_response(self) -> CircuitBreakerResponse:
        """Current graduated response level."""
        return self._current_response

    @property
    def thresholds(self) -> CircuitBreakerThresholds | None:
        """The WP1.6 bands, or ``None`` when the breaker was built without them."""
        return self._thresholds

    def _active_conditions(self) -> tuple[CircuitBreakerResponse, ...]:
        """Active conditions, most severe first (halt, daily_limit, reduce)."""
        conditions: list[CircuitBreakerResponse] = []
        if self._halt or self._tripped:
            conditions.append(CircuitBreakerResponse.HALT)
        if self._daily_limit_day is not None:
            conditions.append(CircuitBreakerResponse.DAILY_LIMIT)
        if self._reduce:
            conditions.append(CircuitBreakerResponse.REDUCE)
        return tuple(conditions)

    def _multiplier_for(self, level: CircuitBreakerResponse) -> Decimal:
        """BUY size multiplier of ``level``: 1 / reduce_multiplier / 0 (blocked)."""
        if level == CircuitBreakerResponse.OK:
            return _ONE
        if level == CircuitBreakerResponse.REDUCE:
            if self._thresholds is not None:
                return self._thresholds.reduce_multiplier
            return Decimal(str(self._config.reduce_position_multiplier))
        return _ZERO

    @property
    def state(self) -> CircuitBreakerState:
        """Return a snapshot of the current breaker state (read-only, atomic)."""
        level = self._current_response
        t = self._thresholds
        return CircuitBreakerState(
            is_tripped=self._tripped or level == CircuitBreakerResponse.HALT,
            trip_reason=self._trip_reason,
            tripped_at=self._tripped_at,
            trip_count=self._trip_count,
            graduated_response=level,
            level=level.value,
            active_conditions=[c.value for c in self._active_conditions()],
            entries_blocked=level
            in (CircuitBreakerResponse.HALT, CircuitBreakerResponse.DAILY_LIMIT),
            entry_size_multiplier=str(self._multiplier_for(level)),
            halt_since=self._halt_since,
            last_halt_reason=self._halt_reason,
            halt_count=self._halt_count,
            halt_exit_progress=self._halt_exit_streak,
            reduce_exit_progress=self._reduce_exit_streak,
            restart_watch=self._restart_watch(),
            daily_limit_day=self._daily_limit_day,
            evaluations=self._evaluations,
            last_evaluated_at=self._last_evaluated_at,
            thresholds=t.as_str_dict() if t is not None else {},
            inputs=self._last_inputs.as_str_dict() if self._last_inputs is not None else None,
        )

    def _restart_watch(self) -> str:
        """``halt`` / ``reduce`` while the exit bands act as entry bands, else ``none``."""
        t = self._thresholds
        if t is None:
            return "none"
        if self._evaluations < t.deescalation_evaluations:
            return "halt"
        if self._evaluations < 2 * t.deescalation_evaluations:
            return "reduce"
        return "none"

    def add_transition_listener(
        self, listener: Callable[[CircuitBreakerTransition], None]
    ) -> None:
        """Register a callback invoked once per level change (never raises into evaluate)."""
        self._listeners.append(listener)

    # ------------------------------------------------------------------ #
    # WP1.6: the production evaluation (RiskSnapshot -> decision)
    # ------------------------------------------------------------------ #

    def evaluate(self, snapshot: RiskSnapshot) -> CircuitBreakerDecision:
        """Evaluate one processed bar from a typed :class:`RiskSnapshot` (§5.2).

        Pure with respect to the portfolio (P-5): the only input is
        ``snapshot``; the basis / day start are read fresh every call (E-6).
        There is no wall-clock read (P-6): the day comes from
        ``snapshot.day`` and times from ``snapshot.as_of``.

        Transactional (SYN-06): inputs are validated first and the machine
        state is committed only after every computation succeeded, so a raise
        leaves the breaker exactly as it was.

        Raises
        ------
        TypeError
            ``snapshot`` is not a :class:`RiskSnapshot`.
        RuntimeError
            The breaker was built without thresholds.
        ValueError
            A snapshot Decimal field is not finite.
        """
        if not isinstance(snapshot, RiskSnapshot):
            raise TypeError(
                f"evaluate() requires a RiskSnapshot, got {type(snapshot).__name__}"
            )
        t = self._thresholds
        if t is None:
            raise RuntimeError(
                "circuit breaker thresholds not configured; use build_run_circuit_breaker"
            )
        for field_name in _SNAPSHOT_DECIMAL_FIELDS:
            _require_finite_decimal(field_name, getattr(snapshot, field_name))

        dd = snapshot.drawdown_pct
        basis = snapshot.daily_loss_basis
        daily_threshold = basis * t.max_daily_loss_pct  # E-6: recomputed every call
        gate_pnl = snapshot.gate_daily_pnl(snapshot.nav)
        # Identical to the WP1.5 gate (risk.py _check_daily_loss, inclusive).
        raw_daily = snapshot.state_unknown or basis <= _ZERO or gate_pnl <= -daily_threshold
        n = t.deescalation_evaluations
        # Restart watch (CF-35): the exit band is the entry band for the first
        # N (HALT) / 2N (REDUCE) evaluations of every instance.
        halt_watch = self._evaluations < n
        reduce_watch = self._evaluations < 2 * n

        # Work on locals; commit at the end.
        prev_halt, prev_reduce, prev_dl = self._halt, self._reduce, self._daily_limit_day
        halt, h_streak = self._halt, self._halt_exit_streak
        reduce, r_streak = self._reduce, self._reduce_exit_streak
        dl_day = self._daily_limit_day

        # REDUCE machine (entry 0.5L, exit 0.4L for N consecutive evaluations).
        if dd >= t.reduce_entry or (reduce_watch and dd >= t.reduce_exit):
            reduce, r_streak = True, 0
        elif reduce:
            if dd < t.reduce_exit:
                r_streak += 1
                if r_streak >= n:
                    reduce, r_streak = False, 0
            else:
                r_streak = 0
        # HALT machine (entry L, exit 0.9L for N consecutive evaluations).
        if dd >= t.halt_entry or (halt_watch and dd >= t.halt_exit):
            halt, h_streak = True, 0
        elif halt:
            if dd < t.halt_exit:
                h_streak += 1
                if h_streak >= n:
                    # One-level step-down: HALT never clears straight to OK.
                    halt, h_streak = False, 0
                    reduce, r_streak = True, 0
            else:
                h_streak = 0
        if halt:
            reduce, r_streak = True, 0  # invariant: HALT implies REDUCE
        # DAILY machine: latches for the risk day, clears on the day roll only.
        if raw_daily:
            dl_day = snapshot.day  # SYN-01: always the current day
        elif dl_day is not None and snapshot.day > dl_day:
            dl_day = None

        if halt or self._tripped:
            level = CircuitBreakerResponse.HALT
        elif dl_day is not None:
            level = CircuitBreakerResponse.DAILY_LIMIT
        elif reduce:
            level = CircuitBreakerResponse.REDUCE
        else:
            level = CircuitBreakerResponse.OK

        inputs = CircuitBreakerInputs(
            as_of=snapshot.as_of,
            day=snapshot.day,
            nav=snapshot.nav,
            peak_equity=snapshot.peak_equity,
            drawdown_pct=dd,
            gate_daily_pnl=gate_pnl,
            daily_loss_basis=basis,
            daily_threshold=daily_threshold,
            day_start_source=snapshot.day_start_source,
            state_unknown=snapshot.state_unknown,
            open_position_count=snapshot.open_position_count,
        )
        if level == CircuitBreakerResponse.HALT:
            if halt:
                if dd >= t.halt_entry:
                    reason: str | None = f"drawdown {dd} >= limit {t.halt_entry}"
                elif not prev_halt:  # entered below L: only the restart watch does that
                    reason = f"drawdown {dd} >= halt exit band {t.halt_exit} (restart watch)"
                else:
                    reason = f"drawdown {dd} >= halt exit band {t.halt_exit} (recovery pending)"
            else:
                reason = f"manual trip: {self._trip_reason}"
        elif level == CircuitBreakerResponse.DAILY_LIMIT:
            reason = (
                "daily loss state unknown or basis unavailable"
                if snapshot.state_unknown or basis <= _ZERO
                else f"daily pnl {gate_pnl} <= -{daily_threshold}"
            )
        elif level == CircuitBreakerResponse.REDUCE:
            reason = (
                f"reduce active: drawdown {dd} (entry >= {t.reduce_entry}, "
                f"exit < {t.reduce_exit} for {n} evaluations)"
            )
        else:
            reason = None

        # ---- commit ------------------------------------------------- #
        prev_level = self._current_response
        self._halt, self._halt_exit_streak = halt, h_streak
        self._reduce, self._reduce_exit_streak = reduce, r_streak
        self._daily_limit_day = dl_day
        self._current_response = level
        self._evaluations += 1
        self._last_inputs = inputs
        self._last_evaluated_at = snapshot.as_of
        if halt and not prev_halt:
            self._halt_since = snapshot.as_of
            self._halt_reason = reason
            self._halt_count += 1
        elif prev_halt and not halt:
            self._halt_since = None  # _halt_reason kept as last_halt_reason

        transition: CircuitBreakerTransition | None = None
        if level != prev_level:
            transition = CircuitBreakerTransition(
                run_id=self._run_id,
                from_level=prev_level,
                to_level=level,
                at=snapshot.as_of,
                reason=reason if reason is not None else "recovered",
                drawdown_pct=dd,
                thresholds=t,
            )
        decision = CircuitBreakerDecision(
            level=level,
            active_conditions=self._active_conditions(),
            entries_blocked=level
            in (CircuitBreakerResponse.HALT, CircuitBreakerResponse.DAILY_LIMIT),
            entry_size_multiplier=self._multiplier_for(level),
            reason=reason,
            inputs=inputs,
            transition=transition,
        )

        # ---- emit (logs, then listeners; nothing here may raise) ----- #
        self._emit_events(
            decision,
            prev_halt=prev_halt,
            prev_reduce=prev_reduce,
            prev_daily_limit_day=prev_dl,
        )
        return decision

    def _emit_events(
        self,
        decision: CircuitBreakerDecision,
        *,
        prev_halt: bool,
        prev_reduce: bool,
        prev_daily_limit_day: date | None,
    ) -> None:
        """Structured log events and listener fan-out for one evaluation (§5.8)."""
        t = self._thresholds
        if t is None:  # unreachable: evaluate() checked
            return
        inp = decision.inputs
        if self._halt and not prev_halt:
            self._log.critical(
                "circuit_breaker.halt_entered",
                alert="ENTRIES_BLOCKED",
                drawdown_pct=str(inp.drawdown_pct),
                halt_entry=str(t.halt_entry),
                halt_exit=str(t.halt_exit),
                nav=str(inp.nav),
                peak_equity=str(inp.peak_equity),
                open_position_count=inp.open_position_count,
            )
        elif prev_halt and not self._halt:
            self._log.warning(
                "circuit_breaker.halt_cleared",
                drawdown_pct=str(inp.drawdown_pct),
                halt_exit=str(t.halt_exit),
                deescalation_evaluations=t.deescalation_evaluations,
                next_level="reduce",
            )
        if self._daily_limit_day is not None and prev_daily_limit_day is None:
            self._log.warning(
                "circuit_breaker.daily_limit_entered",
                day=inp.day.isoformat(),
                gate_daily_pnl=str(inp.gate_daily_pnl),
                threshold=str(inp.daily_threshold),
                day_start_source=inp.day_start_source,
            )
        elif self._daily_limit_day is None and prev_daily_limit_day is not None:
            self._log.info(
                "circuit_breaker.daily_limit_cleared",
                day=inp.day.isoformat(),
                gate_daily_pnl=str(inp.gate_daily_pnl),
                threshold=str(inp.daily_threshold),
                day_start_source=inp.day_start_source,
            )
        if self._reduce and not prev_reduce:
            self._log.warning(
                "circuit_breaker.reduce_entered",
                drawdown_pct=str(inp.drawdown_pct),
                band=str(t.reduce_entry),
            )
        elif prev_reduce and not self._reduce:
            self._log.info(
                "circuit_breaker.reduce_cleared",
                drawdown_pct=str(inp.drawdown_pct),
                band=str(t.reduce_exit),
            )
        transition = decision.transition
        if transition is None:
            return
        self._log.warning(
            "circuit_breaker.transition",
            from_level=transition.from_level.value,
            to_level=transition.to_level.value,
            reason=transition.reason,
            inputs=inp.as_str_dict(),
            thresholds=t.as_str_dict(),
        )
        for listener in list(self._listeners):
            try:
                listener(transition)
            except Exception:
                self._log.error(
                    "circuit_breaker.listener_failed",
                    listener=repr(listener),
                    exc_info=True,
                )

    # ------------------------------------------------------------------ #
    # Legacy API (pre-WP1.6) -- NOT used by production; removal CF-29
    # ------------------------------------------------------------------ #

    def check(
        self,
        equity: float,
        daily_pnl: float,
        drawdown: float,
        consecutive_losses: int,
    ) -> bool:
        """
        Evaluate risk conditions and trip the breaker if any threshold
        is exceeded.

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        This method is idempotent: if the breaker is already tripped, it
        returns True immediately without re-evaluating conditions.

        IMPORTANT: This method is unchanged from pre-Sprint 32. The graduated
        response system is available via ``check_graduated()``.

        Parameters
        ----------
        equity : float
            Current portfolio equity in quote currency.
        daily_pnl : float
            Net PnL since the start of the current trading day.
        drawdown : float
            Current drawdown as a fraction (0.10 = 10% below peak).
        consecutive_losses : int
            Number of consecutive losing trades.

        Returns
        -------
        bool
            True if trading should be halted (breaker is tripped).
        """
        if self._tripped:
            return True

        # Check max daily loss
        if equity > 0.0:
            daily_loss_pct = abs(min(0.0, daily_pnl)) / equity
            if daily_loss_pct >= self._config.max_daily_loss_pct:
                self._trip_internal(
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit "
                    f"{self._config.max_daily_loss_pct:.2%}"
                )
                return True

        # Check max drawdown
        if drawdown >= self._config.max_drawdown_pct:
            self._trip_internal(
                f"Drawdown {drawdown:.2%} exceeds limit "
                f"{self._config.max_drawdown_pct:.2%}"
            )
            return True

        # Check consecutive losses
        if consecutive_losses >= self._config.max_consecutive_losses:
            self._trip_internal(
                f"Consecutive losses ({consecutive_losses}) reached limit "
                f"({self._config.max_consecutive_losses})"
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    # Graduated response (Sprint 32)
    # ------------------------------------------------------------------ #

    def check_graduated(
        self,
        equity: float,
        daily_pnl: float,
        drawdown: float,
    ) -> CircuitBreakerResponse:
        """
        Evaluate graduated risk conditions without tripping the hard breaker.

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        This method evaluates soft thresholds and returns an action-level
        response without permanently tripping the breaker. The hard ``check()``
        method remains unchanged and handles the full-halt case.

        Daily limit reset:
            If ``daily_loss_resume_next_day=True`` (default), the DAILY_LIMIT
            state is automatically cleared when the UTC date changes.

        Parameters
        ----------
        equity : float
            Current portfolio equity in quote currency.
        daily_pnl : float
            Net PnL since the start of the current trading day.
        drawdown : float
            Current drawdown as a fraction (0.10 = 10% below peak).

        Returns
        -------
        CircuitBreakerResponse
            The current graduated response level.
        """
        # Check if daily limit should be reset (new trading day)
        if self._config.daily_loss_resume_next_day:
            self.check_daily_limit_reset()

        # If hard-tripped, always return HALT
        if self._tripped:
            self._current_response = CircuitBreakerResponse.HALT
            return self._current_response

        # Check DAILY_LIMIT  -- already in daily limit state
        if self._current_response == CircuitBreakerResponse.DAILY_LIMIT:
            return self._current_response

        # Check if daily loss limit is newly reached
        if equity > 0.0:
            daily_loss_pct = abs(min(0.0, daily_pnl)) / equity
            if daily_loss_pct >= self._config.max_daily_loss_pct:
                self._current_response = CircuitBreakerResponse.DAILY_LIMIT
                self._daily_limit_date = datetime.now(tz=UTC).date()
                self._log.warning(
                    "circuit_breaker.daily_limit_reached",
                    daily_loss_pct=f"{daily_loss_pct:.2%}",
                    limit=f"{self._config.max_daily_loss_pct:.2%}",
                )
                return self._current_response

        # Check REDUCE threshold (drawdown between soft and hard thresholds)
        if (
            drawdown >= self._config.reduce_drawdown_pct
            and drawdown < self._config.max_drawdown_pct
        ):
            if self._current_response != CircuitBreakerResponse.REDUCE:
                self._log.warning(
                    "circuit_breaker.reduce_response_activated",
                    drawdown=f"{drawdown:.2%}",
                    reduce_threshold=f"{self._config.reduce_drawdown_pct:.2%}",
                    multiplier=self._config.reduce_position_multiplier,
                )
            self._current_response = CircuitBreakerResponse.REDUCE
            return self._current_response

        # All clear
        self._current_response = CircuitBreakerResponse.OK
        return self._current_response

    def get_position_size_multiplier(self) -> float:
        """
        Return the position size multiplier for the current response level.

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        Returns
        -------
        float
            1.0 for OK, reduce_position_multiplier for REDUCE, 0.0 for HALT/DAILY_LIMIT.
        """
        if self._current_response == CircuitBreakerResponse.OK:
            return 1.0
        elif self._current_response == CircuitBreakerResponse.REDUCE:
            return self._config.reduce_position_multiplier
        else:
            return 0.0

    def check_daily_limit_reset(self) -> bool:
        """
        Check if the DAILY_LIMIT state should be cleared (new UTC day).

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        Returns
        -------
        bool
            True if the daily limit was reset.
        """
        if (
            self._current_response == CircuitBreakerResponse.DAILY_LIMIT
            and self._daily_limit_date is not None
        ):
            today = datetime.now(tz=UTC).date()
            if today > self._daily_limit_date:
                self._current_response = CircuitBreakerResponse.OK
                self._daily_limit_date = None
                self._log.info(
                    "circuit_breaker.daily_limit_reset",
                    new_date=str(today),
                )
                return True
        return False

    # ------------------------------------------------------------------ #
    # Manual controls
    # ------------------------------------------------------------------ #

    def trip(self, reason: str) -> None:
        """
        Manually trip the circuit breaker.

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        Use this for operator-initiated emergency stops or when external
        systems detect anomalous conditions.

        Parameters
        ----------
        reason : str
            Human-readable reason for the trip.
        """
        self._trip_internal(reason)

    def reset(self) -> None:
        """
        Reset the circuit breaker after manual review.

        Legacy -- not used by production since WP1.6 (wall clock / sticky);
        removal CF-29.

        Does not touch the WP1.6 machine state (halt / reduce / daily-limit
        counters); those are data-derived and re-evaluated every bar.

        Clears the tripped state, graduated response, and daily limit state.
        Allows trading to resume. The trip count is preserved for audit purposes.
        """
        if not self._tripped:
            self._log.debug("circuit_breaker.reset_while_not_tripped")
            return

        self._log.warning(
            "circuit_breaker.reset",
            previous_reason=self._trip_reason,
            tripped_at=str(self._tripped_at) if self._tripped_at else None,
            trip_count=self._trip_count,
        )

        self._tripped = False
        self._trip_reason = None
        self._tripped_at = None
        self._current_response = CircuitBreakerResponse.OK
        self._daily_limit_date = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _trip_internal(self, reason: str) -> None:
        """
        Internal trip mechanism with logging.

        If already tripped, logs a debug message and returns without
        updating the reason (first trip wins).
        """
        if self._tripped:
            self._log.debug(
                "circuit_breaker.already_tripped",
                existing_reason=self._trip_reason,
                new_reason=reason,
            )
            return

        self._tripped = True
        self._trip_reason = reason
        self._tripped_at = datetime.now(tz=UTC)
        self._trip_count += 1
        self._current_response = CircuitBreakerResponse.HALT

        self._log.critical(
            "circuit_breaker.tripped",
            reason=reason,
            trip_count=self._trip_count,
            alert="TRADING_HALTED",
        )

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker("
            f"tripped={self._tripped}, "
            f"reason={self._trip_reason!r}, "
            f"trip_count={self._trip_count}, "
            f"response={self._current_response})"
        )


def build_run_circuit_breaker(run_id: str, params: RiskParameters) -> CircuitBreaker:
    """THE one construction helper for a run's circuit breaker (WP1.6, P-2).

    Used by the orchestrator (paper + live), the integration harness and the
    ``BacktestRunner`` opt-in, so production and tests share one set of
    thresholds derived from the run's own :class:`RiskParameters`
    (live 0.20 / paper 0.30 drawdown, 0.08 daily loss).

    Raises
    ------
    ValueError
        The derived thresholds are invalid (run start fails loudly, RK16-14).
    """
    thresholds = CircuitBreakerThresholds.from_risk_parameters(params)
    config = CircuitBreakerConfig(
        max_daily_loss_pct=params.max_daily_loss_pct,
        max_drawdown_pct=params.max_drawdown_pct,
        reduce_drawdown_pct=float(thresholds.reduce_entry),
        reduce_position_multiplier=float(thresholds.reduce_multiplier),
    )
    return CircuitBreaker(config=config, run_id=run_id, thresholds=thresholds)
