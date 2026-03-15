"""
tests/unit/test_graduated_circuit_breaker.py
----------------------------------------------
Unit tests for the Sprint 32 graduated CircuitBreaker extensions.

Modules under test
------------------
packages/trading/safety.py

New Sprint 32 surface:
  CircuitBreakerResponse   -- OK / REDUCE / DAILY_LIMIT / HALT enum
  check_graduated()        -- soft threshold evaluation, no hard trip
  get_position_size_multiplier() -- multiplier per response level
  check_daily_limit_reset()      -- auto-reset DAILY_LIMIT on new UTC day
  CircuitBreakerConfig           -- reduce_drawdown_pct + ordering validator

Coverage groups (18 tests)
---------------------------
TestCheckGraduated       (6) -- OK, REDUCE, DAILY_LIMIT, HALT paths
TestGraduatedPriority    (3) -- HALT > DAILY_LIMIT > REDUCE precedence
TestPositionSizeMultiplier (4) -- multiplier per response level
TestDailyLimitReset      (3) -- auto-reset on new day, same day, disabled
TestConfigValidation     (2) -- ordering constraint + valid config

Design notes
------------
- TestDailyLimitReset.test_reset_on_next_day avoids datetime mocking by
  setting _daily_limit_date to date(2000, 1, 1) — guaranteed in the past
  relative to any real wall clock.  This is the deterministic pattern
  recommended for check_daily_limit_reset() which calls datetime.now(tz=UTC).date()
  and compares with the stored date.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from trading.safety import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResponse,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _breaker(
    *,
    max_daily_loss_pct: float = 0.05,
    max_drawdown_pct: float = 0.15,
    max_consecutive_losses: int = 5,
    reduce_drawdown_pct: float = 0.10,
    reduce_position_multiplier: float = 0.5,
    daily_loss_resume_next_day: bool = True,
    run_id: str = "test-run",
) -> CircuitBreaker:
    """Factory: build a CircuitBreaker with explicit Sprint 32 config."""
    config = CircuitBreakerConfig(
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        reduce_drawdown_pct=reduce_drawdown_pct,
        reduce_position_multiplier=reduce_position_multiplier,
        daily_loss_resume_next_day=daily_loss_resume_next_day,
    )
    return CircuitBreaker(config=config, run_id=run_id)


# ===========================================================================
# TestCheckGraduated
# ===========================================================================


class TestCheckGraduated:
    """Tests for check_graduated() return values."""

    def test_ok_normal_conditions(self) -> None:
        """All params within limits => CircuitBreakerResponse.OK."""
        breaker = _breaker()
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-100.0,   # 1% loss — below 5% limit
            drawdown=0.05,       # below reduce threshold of 10%
        )
        assert response == CircuitBreakerResponse.OK

    def test_reduce_drawdown(self) -> None:
        """drawdown=0.11 (above reduce=0.10, below halt=0.15) => REDUCE."""
        breaker = _breaker()
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-100.0,
            drawdown=0.11,
        )
        assert response == CircuitBreakerResponse.REDUCE

    def test_reduce_at_threshold_boundary(self) -> None:
        """drawdown exactly at reduce_drawdown_pct (0.10) => REDUCE."""
        breaker = _breaker()
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.10,
        )
        assert response == CircuitBreakerResponse.REDUCE

    def test_daily_limit(self) -> None:
        """daily_pnl=-600, equity=10000 (6% > 5% limit) => DAILY_LIMIT."""
        breaker = _breaker()
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-600.0,
            drawdown=0.02,
        )
        assert response == CircuitBreakerResponse.DAILY_LIMIT

    def test_halt_drawdown(self) -> None:
        """After a hard trip, check_graduated returns HALT."""
        breaker = _breaker()
        # Hard trip the breaker to put it into HALT state
        breaker.trip("drawdown breach")
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.16,
        )
        assert response == CircuitBreakerResponse.HALT
        assert breaker.is_tripped is True

    def test_halt_already_tripped(self) -> None:
        """A manually tripped breaker always returns HALT from check_graduated."""
        breaker = _breaker()
        breaker.trip("operator halt")
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=0.0,
            drawdown=0.0,
        )
        assert response == CircuitBreakerResponse.HALT


# ===========================================================================
# TestGraduatedPriority
# ===========================================================================


class TestGraduatedPriority:
    """Tests for graduated response level priority ordering."""

    def test_halt_beats_daily_limit(self) -> None:
        """A tripped breaker returns HALT even when daily loss also exceeds limit."""
        breaker = _breaker()
        breaker.trip("forced halt")
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-600.0,   # would normally be DAILY_LIMIT
            drawdown=0.20,
        )
        assert response == CircuitBreakerResponse.HALT

    def test_daily_limit_beats_reduce(self) -> None:
        """DAILY_LIMIT takes precedence over REDUCE when both conditions are met."""
        breaker = _breaker()
        # drawdown=0.11 would give REDUCE, but daily loss > limit gives DAILY_LIMIT
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-600.0,   # 6% daily loss > 5% limit
            drawdown=0.11,       # in REDUCE zone
        )
        assert response == CircuitBreakerResponse.DAILY_LIMIT

    def test_reduce_beats_ok(self) -> None:
        """REDUCE is returned (not OK) when drawdown is in the reduce zone."""
        breaker = _breaker()
        response = breaker.check_graduated(
            equity=10_000,
            daily_pnl=-50.0,   # well within daily limit
            drawdown=0.11,      # between reduce_drawdown_pct and max_drawdown_pct
        )
        assert response == CircuitBreakerResponse.REDUCE


# ===========================================================================
# TestPositionSizeMultiplier
# ===========================================================================


class TestPositionSizeMultiplier:
    """Tests for get_position_size_multiplier()."""

    def test_ok_returns_1(self) -> None:
        """When current_response == OK, multiplier is 1.0."""
        breaker = _breaker()
        # Ensure we are in OK state
        breaker.check_graduated(equity=10_000, daily_pnl=0.0, drawdown=0.0)
        assert breaker.get_position_size_multiplier() == 1.0

    def test_reduce_returns_config_multiplier(self) -> None:
        """When current_response == REDUCE, multiplier equals reduce_position_multiplier."""
        breaker = _breaker(reduce_position_multiplier=0.5)
        breaker.check_graduated(equity=10_000, daily_pnl=0.0, drawdown=0.11)
        assert breaker.get_position_size_multiplier() == pytest.approx(0.5)

    def test_daily_limit_returns_0(self) -> None:
        """When current_response == DAILY_LIMIT, multiplier is 0.0."""
        breaker = _breaker()
        breaker.check_graduated(equity=10_000, daily_pnl=-600.0, drawdown=0.0)
        assert breaker.current_response == CircuitBreakerResponse.DAILY_LIMIT
        assert breaker.get_position_size_multiplier() == 0.0

    def test_halt_returns_0(self) -> None:
        """When current_response == HALT, multiplier is 0.0."""
        breaker = _breaker()
        breaker.trip("halt for test")
        breaker.check_graduated(equity=10_000, daily_pnl=0.0, drawdown=0.0)
        assert breaker.current_response == CircuitBreakerResponse.HALT
        assert breaker.get_position_size_multiplier() == 0.0


# ===========================================================================
# TestDailyLimitReset
# ===========================================================================


class TestDailyLimitReset:
    """Tests for check_daily_limit_reset() auto-reset on new UTC day.

    Design note: test_reset_on_next_day avoids datetime mocking entirely.
    The implementation compares _daily_limit_date to datetime.now(tz=UTC).date().
    We set _daily_limit_date to date(2000, 1, 1) — a date guaranteed to be in
    the past relative to any real-world clock — so today > _daily_limit_date
    is always True without any patching.
    """

    def test_reset_on_next_day(self) -> None:
        """DAILY_LIMIT state clears when _daily_limit_date is in the past."""
        breaker = _breaker(daily_loss_resume_next_day=True)
        # Put breaker into DAILY_LIMIT state with a date far in the past
        breaker._current_response = CircuitBreakerResponse.DAILY_LIMIT
        breaker._daily_limit_date = date(2000, 1, 1)  # guaranteed in the past

        reset = breaker.check_daily_limit_reset()

        assert reset is True
        assert breaker._current_response == CircuitBreakerResponse.OK
        assert breaker._daily_limit_date is None

    def test_no_reset_same_day(self) -> None:
        """DAILY_LIMIT does not clear when _daily_limit_date is today."""
        breaker = _breaker(daily_loss_resume_next_day=True)
        breaker._current_response = CircuitBreakerResponse.DAILY_LIMIT
        # Use actual today so today > today is False
        today = datetime.now(tz=UTC).date()
        breaker._daily_limit_date = today

        reset = breaker.check_daily_limit_reset()
        assert reset is False
        assert breaker._current_response == CircuitBreakerResponse.DAILY_LIMIT

    def test_no_reset_when_disabled(self) -> None:
        """When daily_loss_resume_next_day=False, check_graduated never calls reset."""
        breaker = _breaker(daily_loss_resume_next_day=False)
        # Manually put into DAILY_LIMIT with a past date
        breaker._current_response = CircuitBreakerResponse.DAILY_LIMIT
        breaker._daily_limit_date = date(2000, 1, 1)

        # check_graduated with the flag disabled should NOT auto-reset
        response = breaker.check_graduated(equity=10_000, daily_pnl=0.0, drawdown=0.0)
        assert response == CircuitBreakerResponse.DAILY_LIMIT
        # _daily_limit_date still set (not reset)
        assert breaker._daily_limit_date == date(2000, 1, 1)


# ===========================================================================
# TestConfigValidation
# ===========================================================================


class TestConfigValidation:
    """Tests for CircuitBreakerConfig model validator."""

    def test_reduce_drawdown_must_be_less_than_max(self) -> None:
        """reduce_drawdown_pct == max_drawdown_pct must raise ValueError."""
        with pytest.raises(ValueError, match="reduce_drawdown_pct"):
            CircuitBreakerConfig(
                reduce_drawdown_pct=0.15,
                max_drawdown_pct=0.15,
            )

    def test_valid_config_no_error(self) -> None:
        """reduce=0.10 < max=0.15 must not raise."""
        cfg = CircuitBreakerConfig(
            reduce_drawdown_pct=0.10,
            max_drawdown_pct=0.15,
        )
        assert cfg.reduce_drawdown_pct == 0.10
        assert cfg.max_drawdown_pct == 0.15
