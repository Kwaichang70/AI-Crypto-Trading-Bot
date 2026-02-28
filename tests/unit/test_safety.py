"""
tests/unit/test_safety.py
--------------------------
Unit tests for the safety gates and circuit breaker module.

Modules under test
------------------
- packages/trading/safety.py  — LiveTradingGate, CircuitBreaker, and
  their associated data models (GateCheckResult, GateLayer,
  CircuitBreakerConfig, CircuitBreakerState, LiveTradingGateError).

Test coverage
-------------
CircuitBreaker — normal operation
- Fresh breaker starts in non-tripped state
- check() returns False when all metrics are within thresholds
- check() trips and returns True when daily loss exceeds threshold
- check() trips and returns True when drawdown exceeds threshold
- check() trips and returns True when consecutive losses reach threshold
- Already-tripped breaker returns True immediately (idempotent)
- Manual trip() sets tripped state and stores the supplied reason
- First-trip-wins: a second trip() call does not overwrite the reason
- reset() clears tripped state, allowing trading to resume
- reset() preserves the cumulative trip_count for audit purposes
- state property returns a fully populated CircuitBreakerState snapshot

CircuitBreaker — edge cases
- Zero equity skips the daily-loss percentage check (no ZeroDivisionError)
- Drawdown exactly at threshold triggers the breaker (>= boundary)
- Drawdown just below threshold does not trigger the breaker
- Positive daily_pnl never triggers the daily-loss check
- Multiple trip/reset cycles correctly increment trip_count
- reset() on a fresh (non-tripped) breaker is a safe no-op
- Custom config thresholds are respected over defaults

LiveTradingGate — layer checks
- All three layers pass when settings are correct and token matches
- Gate fails when enable_live_trading is False (Layer 1)
- Gate fails when exchange_api_key is empty (Layer 2)
- Gate fails when exchange_api_secret is empty (Layer 2)
- Gate fails when no confirmation token is provided (Layer 3)
- Gate fails when a wrong confirmation token is provided (Layer 3)
- require_gate() raises LiveTradingGateError on any layer failure
- require_gate() does not raise when all three layers pass
- GateCheckResult.failures contains only the details of failed layers
- GateCheckResult.layer_results maps each layer name to its bool result
"""

from __future__ import annotations

import types
from datetime import UTC, datetime

import pytest

from trading.safety import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    GateCheckResult,
    GateLayer,
    LiveTradingGate,
    LiveTradingGateError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_breaker(
    *,
    max_daily_loss_pct: float = 0.05,
    max_drawdown_pct: float = 0.15,
    max_consecutive_losses: int = 5,
    run_id: str = "test-run-001",
) -> CircuitBreaker:
    """
    Build a CircuitBreaker with configurable thresholds.

    Defaults match CircuitBreakerConfig defaults so tests that do not
    care about configuration can use them without arguments.
    """
    config = CircuitBreakerConfig(
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
    )
    return CircuitBreaker(config=config, run_id=run_id)


def _make_gate_settings(
    *,
    enable_live: bool = True,
    api_key: str = "key123",
    api_secret: str = "secret456",
    confirm_token: str = "tok",
) -> types.SimpleNamespace:
    """
    Build a fake application Settings object as SimpleNamespace.

    All four fields are those consulted by LiveTradingGate's three layer
    checks.  Plain strings are used (no SecretStr wrapping) to keep
    tests simple and dependency-free.
    """
    return types.SimpleNamespace(
        enable_live_trading=enable_live,
        exchange_api_key=api_key,
        exchange_api_secret=api_secret,
        live_trading_confirm_token=confirm_token,
    )


# ===========================================================================
# CircuitBreaker — normal operation
# ===========================================================================


class TestCircuitBreakerBasic:
    """Tests for core CircuitBreaker behaviour under normal operating conditions."""

    def test_fresh_breaker_is_not_tripped(self) -> None:
        """A newly constructed CircuitBreaker must have is_tripped == False."""
        breaker = _make_breaker()
        assert breaker.is_tripped is False

    def test_check_returns_false_when_within_all_thresholds(self) -> None:
        """
        check() returns False when equity, daily_pnl, drawdown, and
        consecutive_losses are all comfortably within their thresholds.
        """
        breaker = _make_breaker(
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            max_consecutive_losses=5,
        )
        # daily_loss_pct = abs(min(0, -200)) / 10000 = 2% < 5%
        # drawdown = 0.10 < 0.15
        # consecutive_losses = 3 < 5
        result = breaker.check(
            equity=10_000,
            daily_pnl=-200.0,
            drawdown=0.10,
            consecutive_losses=3,
        )
        assert result is False
        assert breaker.is_tripped is False

    def test_check_trips_on_daily_loss_exceeding_threshold(self) -> None:
        """
        check() trips the breaker and returns True when the daily loss
        percentage meets or exceeds max_daily_loss_pct.

        With equity=10_000 and daily_pnl=-600:
            daily_loss_pct = abs(-600) / 10_000 = 0.06 >= 0.05 (threshold).
        """
        breaker = _make_breaker(max_daily_loss_pct=0.05)
        result = breaker.check(
            equity=10_000,
            daily_pnl=-600.0,
            drawdown=0.05,
            consecutive_losses=1,
        )
        assert result is True
        assert breaker.is_tripped is True

    def test_check_trips_on_drawdown_exceeding_threshold(self) -> None:
        """
        check() trips the breaker and returns True when drawdown meets or
        exceeds max_drawdown_pct.

        drawdown=0.16 >= 0.15 triggers the breaker.
        """
        breaker = _make_breaker(max_drawdown_pct=0.15)
        result = breaker.check(
            equity=10_000,
            daily_pnl=-100.0,
            drawdown=0.16,
            consecutive_losses=1,
        )
        assert result is True
        assert breaker.is_tripped is True

    def test_check_trips_on_consecutive_losses_exceeding_threshold(self) -> None:
        """
        check() trips the breaker and returns True when consecutive_losses
        meets or exceeds max_consecutive_losses.

        With max_consecutive_losses=5, passing consecutive_losses=5 trips it.
        """
        breaker = _make_breaker(max_consecutive_losses=5)
        result = breaker.check(
            equity=10_000,
            daily_pnl=-100.0,
            drawdown=0.05,
            consecutive_losses=5,
        )
        assert result is True
        assert breaker.is_tripped is True

    def test_check_returns_true_immediately_when_already_tripped(self) -> None:
        """
        If the breaker is already tripped, check() returns True immediately
        regardless of the supplied metric values — it is idempotent.
        """
        breaker = _make_breaker()
        breaker.trip("pre-trip for test")
        assert breaker.is_tripped is True

        # All metrics are safe — but the breaker is already open.
        result = breaker.check(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.0,
            consecutive_losses=0,
        )
        assert result is True

    def test_manual_trip_sets_tripped_state_and_reason(self) -> None:
        """
        trip(reason) opens the breaker and stores the supplied reason string.
        """
        breaker = _make_breaker()
        reason = "Operator initiated emergency stop"
        breaker.trip(reason)
        assert breaker.is_tripped is True
        assert breaker.state.trip_reason == reason

    def test_trip_is_first_trip_wins_reason_not_overwritten(self) -> None:
        """
        When trip() is called on an already-tripped breaker the existing
        reason must not be overwritten (first-trip-wins semantics).
        """
        breaker = _make_breaker()
        first_reason = "first trip"
        second_reason = "second trip attempt"

        breaker.trip(first_reason)
        breaker.trip(second_reason)

        assert breaker.state.trip_reason == first_reason

    def test_reset_clears_tripped_state(self) -> None:
        """
        After reset() is called, is_tripped must return False and
        trip_reason must be None, allowing trading to resume.
        """
        breaker = _make_breaker()
        breaker.trip("test trip")
        assert breaker.is_tripped is True

        breaker.reset()

        assert breaker.is_tripped is False
        assert breaker.state.trip_reason is None

    def test_reset_preserves_trip_count(self) -> None:
        """
        reset() clears the tripped state but must NOT reset trip_count.
        trip_count provides an audit trail of how many times the breaker
        has been opened during the session.
        """
        breaker = _make_breaker()
        breaker.trip("first trip")
        breaker.reset()

        state = breaker.state
        assert state.trip_count == 1
        assert state.is_tripped is False

    def test_state_property_returns_circuit_breaker_state(self) -> None:
        """
        state returns a CircuitBreakerState instance with all fields
        correctly populated after a trip.
        """
        breaker = _make_breaker()
        before = datetime.now(tz=UTC)
        breaker.trip("drawdown breach")
        after = datetime.now(tz=UTC)

        state = breaker.state

        assert isinstance(state, CircuitBreakerState)
        assert state.is_tripped is True
        assert state.trip_reason == "drawdown breach"
        assert state.trip_count == 1
        assert state.tripped_at is not None
        assert before <= state.tripped_at <= after

    def test_repr_contains_key_state_fields(self) -> None:
        """__repr__ must include tripped state, reason, and trip_count."""
        breaker = _make_breaker()
        repr_str = repr(breaker)
        assert "CircuitBreaker" in repr_str
        assert "tripped=False" in repr_str
        assert "trip_count=0" in repr_str


# ===========================================================================
# CircuitBreaker — edge cases
# ===========================================================================


class TestCircuitBreakerEdgeCases:
    """Edge-case and boundary tests for CircuitBreaker."""

    def test_zero_equity_skips_daily_loss_check(self) -> None:
        """
        When equity=0 the daily-loss percentage cannot be computed safely.
        The implementation must skip that check to avoid ZeroDivisionError
        and must NOT trip the breaker based on daily_pnl alone.
        """
        breaker = _make_breaker(max_daily_loss_pct=0.05)
        # Would cause ZeroDivisionError if the check runs
        result = breaker.check(
            equity=0,
            daily_pnl=-100.0,
            drawdown=0.0,
            consecutive_losses=0,
        )
        assert result is False
        assert breaker.is_tripped is False

    @pytest.mark.parametrize(
        "drawdown,expected_tripped",
        [
            (0.15, True),   # exactly at threshold — must trip (>= semantics)
            (0.16, True),   # above threshold
            (0.1499, False),  # just below threshold — must NOT trip
            (0.14, False),  # clearly below threshold
        ],
    )
    def test_drawdown_threshold_boundary(
        self,
        drawdown: float,
        expected_tripped: bool,
    ) -> None:
        """
        The drawdown check uses >= semantics.  Verify exact-at-threshold
        trips the breaker and just-below-threshold does not.
        """
        breaker = _make_breaker(max_drawdown_pct=0.15)
        result = breaker.check(
            equity=10_000,
            daily_pnl=-100.0,
            drawdown=drawdown,
            consecutive_losses=0,
        )
        assert result is expected_tripped
        assert breaker.is_tripped is expected_tripped

    def test_exact_threshold_boundary_trips(self) -> None:
        """
        Drawdown exactly equal to max_drawdown_pct must trip the breaker
        because the check condition is >= (not strictly >).
        """
        breaker = _make_breaker(max_drawdown_pct=0.15)
        result = breaker.check(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.15,
            consecutive_losses=0,
        )
        assert result is True
        assert breaker.is_tripped is True

    def test_below_threshold_boundary_does_not_trip(self) -> None:
        """
        Drawdown just below max_drawdown_pct must NOT trip the breaker.
        Using 0.1499 is safely below 0.15.
        """
        breaker = _make_breaker(max_drawdown_pct=0.15)
        result = breaker.check(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.1499,
            consecutive_losses=0,
        )
        assert result is False
        assert breaker.is_tripped is False

    def test_positive_daily_pnl_never_trips_daily_loss_check(self) -> None:
        """
        A positive daily_pnl value must never trigger the daily-loss check,
        regardless of how large it is.  min(0, positive) == 0, so
        daily_loss_pct == 0.
        """
        breaker = _make_breaker(max_daily_loss_pct=0.05)
        result = breaker.check(
            equity=10_000,
            daily_pnl=500.0,
            drawdown=0.0,
            consecutive_losses=0,
        )
        assert result is False
        assert breaker.is_tripped is False

    def test_multiple_trip_reset_cycles_increment_trip_count(self) -> None:
        """
        Repeated trip/reset cycles must increment trip_count on each trip.
        After 3 cycles the count must be 3.
        """
        breaker = _make_breaker()
        for cycle in range(1, 4):
            breaker.trip(f"trip #{cycle}")
            breaker.reset()

        # trip_count is preserved across resets
        assert breaker.state.trip_count == 3
        assert breaker.is_tripped is False

    def test_reset_while_not_tripped_is_no_op(self) -> None:
        """
        Calling reset() on a breaker that has never been tripped must not
        raise any exception and must leave the state unchanged.
        """
        breaker = _make_breaker()
        # Should not raise
        breaker.reset()
        assert breaker.is_tripped is False
        assert breaker.state.trip_count == 0

    def test_custom_config_thresholds_are_respected(self) -> None:
        """
        A CircuitBreaker with non-default config values must trip using
        those custom thresholds, not the defaults.

        Custom config: max_daily_loss_pct=0.10, max_drawdown_pct=0.25,
        max_consecutive_losses=10.

        Values that would trip a default breaker (daily loss=6%, drawdown=16%,
        consecutive=5) must NOT trip the custom one.
        """
        breaker = _make_breaker(
            max_daily_loss_pct=0.10,
            max_drawdown_pct=0.25,
            max_consecutive_losses=10,
        )
        # Would trip defaults but must NOT trip this custom breaker
        result = breaker.check(
            equity=10_000,
            daily_pnl=-600.0,   # 6% loss — below 10% custom threshold
            drawdown=0.16,       # 16% — below 25% custom threshold
            consecutive_losses=5,  # 5 — below 10 custom threshold
        )
        assert result is False
        assert breaker.is_tripped is False

    @pytest.mark.parametrize(
        "consecutive_losses,expected_tripped",
        [
            (5, True),   # exactly at threshold
            (6, True),   # above threshold
            (4, False),  # one below threshold
            (0, False),  # zero losses
        ],
    )
    def test_consecutive_losses_threshold_boundary(
        self,
        consecutive_losses: int,
        expected_tripped: bool,
    ) -> None:
        """
        The consecutive-losses check uses >= semantics.  Verify boundary
        behaviour for values at, above, and below the threshold of 5.
        """
        breaker = _make_breaker(max_consecutive_losses=5)
        result = breaker.check(
            equity=10_000,
            daily_pnl=-50.0,
            drawdown=0.01,
            consecutive_losses=consecutive_losses,
        )
        assert result is expected_tripped
        assert breaker.is_tripped is expected_tripped

    @pytest.mark.parametrize(
        "daily_pnl,equity,expected_tripped",
        [
            (-500.0, 10_000, True),   # exactly 5% loss — at threshold
            (-600.0, 10_000, True),   # 6% loss — above threshold
            (-499.9, 10_000, False),  # just below 5% threshold
            (-400.0, 10_000, False),  # 4% loss — below threshold
        ],
    )
    def test_daily_loss_threshold_boundary(
        self,
        daily_pnl: float,
        equity: float,
        expected_tripped: bool,
    ) -> None:
        """
        The daily-loss check uses >= semantics on the computed percentage.
        Verify boundary behaviour for values at, above, and below 5%.
        """
        breaker = _make_breaker(max_daily_loss_pct=0.05)
        result = breaker.check(
            equity=equity,
            daily_pnl=daily_pnl,
            drawdown=0.01,
            consecutive_losses=0,
        )
        assert result is expected_tripped
        assert breaker.is_tripped is expected_tripped

    def test_trip_count_increments_on_automatic_trip_via_check(self) -> None:
        """
        trip_count must increment when the breaker is tripped automatically
        via check(), not only via manual trip().
        """
        breaker = _make_breaker(max_drawdown_pct=0.15)
        breaker.check(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.20,
            consecutive_losses=0,
        )
        assert breaker.state.trip_count == 1

    def test_config_property_returns_original_config(self) -> None:
        """
        The config property must return the CircuitBreakerConfig that was
        injected at construction, with all values preserved.
        """
        config = CircuitBreakerConfig(
            max_daily_loss_pct=0.08,
            max_drawdown_pct=0.20,
            max_consecutive_losses=7,
        )
        breaker = CircuitBreaker(config=config, run_id="cfg-test")
        returned = breaker.config
        assert returned.max_daily_loss_pct == 0.08
        assert returned.max_drawdown_pct == 0.20
        assert returned.max_consecutive_losses == 7

    def test_state_after_reset_has_no_trip_timestamp(self) -> None:
        """
        After reset() the state snapshot must have tripped_at == None.
        """
        breaker = _make_breaker()
        breaker.trip("temporary halt")
        breaker.reset()

        state = breaker.state
        assert state.tripped_at is None
        assert state.is_tripped is False

    def test_default_constructor_uses_default_config(self) -> None:
        """
        CircuitBreaker() with no config argument must use
        CircuitBreakerConfig() defaults (5%, 15%, 5 losses).
        """
        breaker = CircuitBreaker()
        cfg = breaker.config
        assert cfg.max_daily_loss_pct == 0.05
        assert cfg.max_drawdown_pct == 0.15
        assert cfg.max_consecutive_losses == 5

    def test_negative_equity_skips_daily_loss_check(self) -> None:
        """Negative equity must not trigger the daily loss check (guard: equity > 0)."""
        breaker = _make_breaker(max_daily_loss_pct=0.05)
        result = breaker.check(equity=-500.0, daily_pnl=-100.0, drawdown=0.0, consecutive_losses=0)
        assert result is False
        assert breaker.is_tripped is False


# ===========================================================================
# LiveTradingGate — layer checks
# ===========================================================================


class TestLiveTradingGate:
    """Tests for the three-layer live trading gate."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_all_three_layers_pass_gate_passes(self) -> None:
        """
        When all three layers are satisfied the gate returns a
        GateCheckResult with passed=True and an empty failures list.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(
            enable_live=True,
            api_key="key123",
            api_secret="secret456",
            confirm_token="tok",
        )
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is True
        assert result.failures == []
        assert result.layer_results["environment"] is True
        assert result.layer_results["api_keys"] is True
        assert result.layer_results["confirmation"] is True

    # ------------------------------------------------------------------
    # Layer 1 — environment
    # ------------------------------------------------------------------

    def test_environment_layer_fails_when_disabled(self) -> None:
        """
        Gate fails with passed=False when enable_live_trading is False,
        and the 'environment' layer is recorded as failed.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(
            enable_live=False,
            api_key="key123",
            api_secret="secret456",
            confirm_token="tok",
        )
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["environment"] is False
        assert len(result.failures) >= 1

    def test_environment_layer_fails_when_attribute_missing(self) -> None:
        """
        getattr(settings, 'enable_live_trading', False) defaults to False.
        A settings object without the attribute must fail Layer 1.
        """
        gate = LiveTradingGate()
        # SimpleNamespace with no enable_live_trading attribute
        settings = types.SimpleNamespace(
            exchange_api_key="key123",
            exchange_api_secret="secret456",
            live_trading_confirm_token="tok",
        )
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["environment"] is False

    # ------------------------------------------------------------------
    # Layer 2 — api_keys
    # ------------------------------------------------------------------

    def test_api_keys_layer_fails_when_key_empty(self) -> None:
        """
        Gate fails when exchange_api_key is an empty string.
        The 'api_keys' layer must be recorded as failed.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(api_key="", api_secret="secret456")
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["api_keys"] is False

    def test_api_keys_layer_fails_when_secret_empty(self) -> None:
        """
        Gate fails when exchange_api_secret is an empty string.
        The 'api_keys' layer must be recorded as failed.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(api_key="key123", api_secret="")
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["api_keys"] is False

    def test_api_keys_layer_fails_when_both_credentials_empty(self) -> None:
        """
        Gate fails when both exchange_api_key and exchange_api_secret are
        empty.  The 'api_keys' layer must be recorded as failed.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(api_key="", api_secret="")
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["api_keys"] is False

    @pytest.mark.parametrize("whitespace_key", ["   ", "\t", "\n"])
    def test_api_keys_layer_fails_when_key_is_whitespace_only(
        self, whitespace_key: str
    ) -> None:
        """
        Keys consisting only of whitespace characters are treated as empty
        (the implementation strips before checking truthiness).
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(api_key=whitespace_key, api_secret="secret456")
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        assert result.layer_results["api_keys"] is False

    @pytest.mark.parametrize("whitespace_secret", ["   ", "\t", "\n"])
    def test_api_keys_layer_fails_when_secret_is_whitespace_only(
        self, whitespace_secret: str
    ) -> None:
        """Whitespace-only exchange_api_secret must fail the API keys layer."""
        gate = LiveTradingGate()
        settings = _make_gate_settings(api_key="key123", api_secret=whitespace_secret)
        result = gate.check_gate(settings, confirm_token="tok")
        assert result.passed is False
        assert result.layer_results["api_keys"] is False

    # ------------------------------------------------------------------
    # Layer 3 — confirmation
    # ------------------------------------------------------------------

    def test_confirmation_layer_fails_when_token_missing(self) -> None:
        """
        Gate fails when no confirm_token is provided at call time, even
        though a token is configured in settings.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings()
        result = gate.check_gate(settings, confirm_token=None)

        assert result.passed is False
        assert result.layer_results["confirmation"] is False

    def test_confirmation_layer_fails_when_token_wrong(self) -> None:
        """
        Gate fails when a confirm_token is provided but does not match the
        stored token in settings.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(confirm_token="correct_token")
        result = gate.check_gate(settings, confirm_token="wrong_token")

        assert result.passed is False
        assert result.layer_results["confirmation"] is False

    def test_confirmation_layer_fails_when_no_stored_token_and_none_provided(
        self,
    ) -> None:
        """
        When neither stored token nor runtime token is provided, the
        confirmation layer must fail (belt-and-suspenders).
        """
        gate = LiveTradingGate()
        settings = types.SimpleNamespace(
            enable_live_trading=True,
            exchange_api_key="key123",
            exchange_api_secret="secret456",
            live_trading_confirm_token=None,
        )
        result = gate.check_gate(settings, confirm_token=None)

        assert result.passed is False
        assert result.layer_results["confirmation"] is False

    def test_confirmation_layer_fails_when_token_provided_but_not_configured(
        self,
    ) -> None:
        """
        When a runtime token is provided but no token is configured in
        settings, the confirmation layer must fail (cannot verify).
        """
        gate = LiveTradingGate()
        settings = types.SimpleNamespace(
            enable_live_trading=True,
            exchange_api_key="key123",
            exchange_api_secret="secret456",
            live_trading_confirm_token=None,
        )
        result = gate.check_gate(settings, confirm_token="some_token")

        assert result.passed is False
        assert result.layer_results["confirmation"] is False

    # ------------------------------------------------------------------
    # require_gate
    # ------------------------------------------------------------------

    def test_require_gate_raises_live_trading_gate_error_on_failure(self) -> None:
        """
        require_gate() must raise LiveTradingGateError when any layer fails.
        The exception must carry the gate_result attribute.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(enable_live=False)

        with pytest.raises(LiveTradingGateError) as exc_info:
            gate.require_gate(settings, confirm_token="tok")

        raised = exc_info.value
        assert isinstance(raised, LiveTradingGateError)
        assert hasattr(raised, "gate_result")
        assert raised.gate_result.passed is False

    def test_require_gate_does_not_raise_when_all_layers_pass(self) -> None:
        """
        require_gate() must return None (no exception) when all three
        layers pass.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings()

        # Must not raise
        result = gate.require_gate(settings, confirm_token="tok")
        assert result is None

    def test_require_gate_error_message_includes_failure_details(self) -> None:
        """
        The exception message raised by require_gate() must incorporate
        the human-readable failure detail from the failed layer.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(enable_live=False)

        with pytest.raises(LiveTradingGateError) as exc_info:
            gate.require_gate(settings, confirm_token="tok")

        message = str(exc_info.value)
        # The message must not be empty and must reference the failure
        assert len(message) > 0
        assert "ENABLE_LIVE_TRADING" in message or "environment" in message.lower() or "not set" in message.lower()

    # ------------------------------------------------------------------
    # GateCheckResult structure
    # ------------------------------------------------------------------

    def test_gate_check_result_failures_lists_only_failed_layer_details(
        self,
    ) -> None:
        """
        GateCheckResult.failures must contain the detail strings from only
        the layers that failed, not from passing layers.

        With enable_live=False and all other layers passing, failures
        must have exactly one entry corresponding to the environment layer.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(enable_live=False)
        result = gate.check_gate(settings, confirm_token="tok")

        # Only the environment layer fails
        assert len(result.failures) == 1
        # The failure detail must reference the environment misconfiguration
        assert "ENABLE_LIVE_TRADING" in result.failures[0]

    def test_gate_check_result_layer_results_maps_names_to_booleans(self) -> None:
        """
        GateCheckResult.layer_results must be a dict mapping the three
        layer names to booleans.  All three names must be present.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings()
        result = gate.check_gate(settings, confirm_token="tok")

        assert isinstance(result.layer_results, dict)
        assert "environment" in result.layer_results
        assert "api_keys" in result.layer_results
        assert "confirmation" in result.layer_results
        for name, value in result.layer_results.items():
            assert isinstance(value, bool), (
                f"layer_results[{name!r}] must be bool, got {type(value)}"
            )

    def test_gate_check_result_layers_list_has_three_gate_layer_objects(
        self,
    ) -> None:
        """
        GateCheckResult.layers must contain exactly three GateLayer objects,
        one per layer, regardless of pass/fail outcome.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings()
        result = gate.check_gate(settings, confirm_token="tok")

        assert len(result.layers) == 3
        for layer in result.layers:
            assert isinstance(layer, GateLayer)
            assert isinstance(layer.name, str)
            assert len(layer.name) > 0
            assert isinstance(layer.passed, bool)

    def test_gate_check_result_checked_at_is_utc_datetime(self) -> None:
        """
        GateCheckResult.checked_at must be a timezone-aware UTC datetime
        set at the time of the gate evaluation.
        """
        before = datetime.now(tz=UTC)
        gate = LiveTradingGate()
        settings = _make_gate_settings()
        result = gate.check_gate(settings, confirm_token="tok")
        after = datetime.now(tz=UTC)

        assert result.checked_at.tzinfo is not None
        assert before <= result.checked_at <= after

    def test_multiple_layer_failures_each_appear_in_failures_list(self) -> None:
        """
        When multiple layers fail simultaneously, all failure details must
        appear in GateCheckResult.failures.

        Using enable_live=False and empty keys causes Layer 1 and Layer 2
        to fail, producing two entries in failures.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(
            enable_live=False,
            api_key="",
            api_secret="",
        )
        result = gate.check_gate(settings, confirm_token="tok")

        assert result.passed is False
        # At least environment and api_keys layers must be in failures
        assert len(result.failures) >= 2
        assert result.layer_results["environment"] is False
        assert result.layer_results["api_keys"] is False

    def test_gate_is_fail_closed_on_all_layers_failing(self) -> None:
        """
        When every layer fails (disabled env, empty keys, wrong token),
        the gate must be fully closed: passed=False, three failures.
        """
        gate = LiveTradingGate()
        settings = _make_gate_settings(
            enable_live=False,
            api_key="",
            api_secret="",
            confirm_token="correct",
        )
        result = gate.check_gate(settings, confirm_token="wrong")

        assert result.passed is False
        assert len(result.failures) == 3
        assert all(v is False for v in result.layer_results.values())

    def test_failed_gate_layers_have_non_empty_detail(self) -> None:
        """Failed gate layers must provide non-empty operator-facing detail messages."""
        gate = LiveTradingGate()
        settings = _make_gate_settings(enable_live=False)
        result = gate.check_gate(settings, confirm_token="tok")
        for layer in result.layers:
            if not layer.passed:
                assert len(layer.detail) > 0, f"Layer {layer.name!r} has empty detail"
