"""
packages/trading/safety.py
---------------------------
Safety gates and circuit breaker for live trading protection.

This module implements two independent safety mechanisms:

1. **LiveTradingGate** — Three-layer activation gate that must be fully
   satisfied before any live order is submitted.  Checked by the
   ``LiveExecutionEngine`` and by the API run-creation endpoint.

2. **CircuitBreaker** — Emergency stop mechanism that halts all trading
   when configurable risk thresholds are breached.  Once tripped, the
   breaker stays open until manually reset by an operator.

Both classes are stateless with respect to persistence — they evaluate
conditions from injected state at call time.  The CircuitBreaker holds
in-memory trip state that survives across bar ticks within a single process
but must be persisted externally for crash recovery.

Design principles
-----------------
- Fail-closed: any ambiguity defaults to blocking orders.
- Explicit reset: breakers never auto-reset.
- No I/O in the hot path: all inputs are passed as arguments.
- Full audit trail via structlog at WARNING/CRITICAL severity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import hmac
import structlog
from pydantic import BaseModel, Field

__all__ = [
    "LiveTradingGate",
    "GateCheckResult",
    "GateLayer",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
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

    1. **Environment** — ``enable_live_trading`` must be explicitly True
       in the application settings (``Settings.enable_live_trading``).
    2. **API Keys** — Valid exchange API key and secret must be configured
       (non-empty ``Settings.exchange_api_key`` and ``exchange_api_secret``).
    3. **Confirmation** — A runtime confirmation token must be provided
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
        api_key = (
            api_key_raw.get_secret_value()
            if hasattr(api_key_raw, "get_secret_value")
            else api_key_raw
        )
        api_secret = (
            api_secret_raw.get_secret_value()
            if hasattr(api_secret_raw, "get_secret_value")
            else api_secret_raw
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
        stored = (
            stored_raw.get_secret_value()
            if hasattr(stored_raw, "get_secret_value")
            else stored_raw
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

        # Token provided but none configured — require exact match impossible
        if not stored and confirm_token:
            return GateLayer(
                name="confirmation",
                passed=False,
                detail="No confirmation token configured in settings. "
                       "Set LIVE_TRADING_CONFIRM_TOKEN to enable token verification.",
            )

        # Both present — constant-time comparison to prevent timing attacks
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
# Circuit breaker configuration
# ---------------------------------------------------------------------------


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for the circuit breaker thresholds.

    All percentage values are expressed as fractions (0.05 = 5%).
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


# ---------------------------------------------------------------------------
# Circuit breaker state snapshot
# ---------------------------------------------------------------------------


class CircuitBreakerState(BaseModel):
    """Snapshot of the circuit breaker's current state for API exposure."""

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


# ---------------------------------------------------------------------------
# Circuit breaker implementation
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Emergency stop mechanism that halts all trading when triggered.

    The circuit breaker evaluates risk conditions on every bar tick or
    before every order submission.  When any threshold is breached, the
    breaker trips and remains open until an operator explicitly resets it.

    Triggers
    --------
    - Max daily loss exceeded
    - Max drawdown from peak exceeded
    - Rapid consecutive losses (configurable threshold)
    - Manual trip via API or kill switch

    Once tripped, the circuit breaker stays open until ``reset()`` is called.
    This is a deliberate design choice: automatic recovery from risk events
    is dangerous and requires human review.

    Parameters
    ----------
    config : CircuitBreakerConfig
        Threshold configuration.
    run_id : str
        Run identifier for structured logging.

    Usage
    -----
    ::

        breaker = CircuitBreaker(config=CircuitBreakerConfig(), run_id="run-123")

        # Check before every order
        if breaker.check(equity=9500, daily_pnl=-600, drawdown=0.12, consecutive_losses=3):
            # Trading halted — do not submit orders
            ...

        # Manual trip
        breaker.trip("Operator initiated emergency stop")

        # Reset after review
        breaker.reset()
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        run_id: str = "",
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._run_id = run_id
        self._tripped: bool = False
        self._trip_reason: str | None = None
        self._tripped_at: datetime | None = None
        self._trip_count: int = 0
        self._log = structlog.get_logger(__name__).bind(
            component="circuit_breaker",
            run_id=run_id,
        )

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
    def state(self) -> CircuitBreakerState:
        """Return a snapshot of the current breaker state."""
        return CircuitBreakerState(
            is_tripped=self._tripped,
            trip_reason=self._trip_reason,
            tripped_at=self._tripped_at,
            trip_count=self._trip_count,
        )

    # ------------------------------------------------------------------ #
    # Core check
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

        This method is idempotent: if the breaker is already tripped, it
        returns True immediately without re-evaluating conditions.

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
    # Manual controls
    # ------------------------------------------------------------------ #

    def trip(self, reason: str) -> None:
        """
        Manually trip the circuit breaker.

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

        Clears the tripped state and allows trading to resume.
        The trip count is preserved for audit purposes.
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
            f"trip_count={self._trip_count})"
        )
