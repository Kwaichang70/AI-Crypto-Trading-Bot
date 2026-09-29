"""
tests/unit/test_smoke_guard.py
---------------------------------
WP-SMOKE (reports/vp2-smoke/synthesis-spec.md section 9) -- unit coverage
for ``packages/trading/smoke_guard.py`` (G-1..G-8) and the availability /
sizing integration points (SMK-T-06..08).
"""
from __future__ import annotations

from decimal import Decimal
from typing import ClassVar

import pytest

from common.types import RunMode
from trading.risk import RiskParameters
from trading.risk_manager import DefaultRiskManager
from trading.smoke_guard import (
    SMOKE_STRATEGY_NAMES,
    SmokeGuardError,
    validate_smoke_run,
)
from trading.strategy_availability import (
    StrategyStatus,
    get_availability,
    is_mode_allowed,
)

_OTHER_STRATEGY = "rsi_mean_reversion"


def _call(
    *,
    strategy_name: str = "smoke_roundtrip",
    mode: object = RunMode.LIVE,
    symbols: list[str] | None = None,
    timeframe: str = "5m",
    initial_capital: object = Decimal("65"),
    strategy_params: dict[str, object] | None = None,
    bracket: dict[str, object] | None = None,
    trailing_stop_pct: object = None,
    allow_pyramiding: bool | None = False,
    enable_adaptive_learning: bool | None = False,
) -> None:
    validate_smoke_run(
        strategy_name=strategy_name,
        mode=mode,
        symbols=symbols if symbols is not None else ["XRP/EUR"],
        timeframe=timeframe,
        initial_capital=initial_capital,
        strategy_params=strategy_params if strategy_params is not None else {"notional_quote": 9.0},
        bracket=bracket
        if bracket is not None
        else {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05},
        trailing_stop_pct=trailing_stop_pct,
        allow_pyramiding=allow_pyramiding,
        enable_adaptive_learning=enable_adaptive_learning,
    )


def _reasons(exc: SmokeGuardError) -> set[str]:
    return {i.reason for i in exc.issues}


# ===========================================================================
# SMK-T-06: validate_smoke_run table-driven over every G-1..G-8 reason.
# ===========================================================================


class TestValidateSmokeRunHappyPath:
    def test_valid_live_config_passes(self) -> None:
        _call()  # must not raise

    def test_non_smoke_strategy_is_a_noop(self) -> None:
        # Every rule violated at once -- still a no-op for another strategy.
        validate_smoke_run(
            strategy_name=_OTHER_STRATEGY,
            mode=RunMode.LIVE,
            symbols=["BTC/USD", "ETH/USD"],
            timeframe="15m",
            initial_capital=Decimal("10"),
            strategy_params={"unknown": 1},
            bracket=None,
            trailing_stop_pct=0.1,
            allow_pyramiding=True,
            enable_adaptive_learning=True,
        )  # must not raise

    def test_backtest_and_paper_with_capital_10000_pass_g4_and_g5(self) -> None:
        for mode in (RunMode.BACKTEST, RunMode.PAPER):
            _call(mode=mode, initial_capital=Decimal("10000"))


class TestG1Symbols:
    def test_two_symbols_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(symbols=["XRP/EUR", "LTC/EUR"])
        assert "too_many_symbols" in _reasons(exc.value)

    def test_zero_symbols_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(symbols=[])
        assert "too_many_symbols" in _reasons(exc.value)

    @pytest.mark.parametrize("mode", [RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE])
    def test_applies_in_every_mode(self, mode: RunMode) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(mode=mode, symbols=["A/EUR", "B/EUR"], initial_capital=Decimal("10000"))
        assert "too_many_symbols" in _reasons(exc.value)


class TestG2Timeframe:
    @pytest.mark.parametrize("timeframe", ["15m", "1h", "1d"])
    def test_disallowed_timeframe_rejected(self, timeframe: str) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(timeframe=timeframe)
        assert "timeframe_not_allowed" in _reasons(exc.value)

    @pytest.mark.parametrize("timeframe", ["1m", "5m"])
    def test_allowed_timeframes_pass(self, timeframe: str) -> None:
        _call(timeframe=timeframe)


class TestG3QuoteCurrency:
    def test_live_non_eur_quote_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(symbols=["XRP/USD"])
        assert "quote_not_allowed" in _reasons(exc.value)

    def test_paper_non_eur_quote_allowed(self) -> None:
        _call(mode=RunMode.PAPER, symbols=["XRP/USD"], initial_capital=Decimal("10000"))

    def test_backtest_non_eur_quote_allowed(self) -> None:
        _call(mode=RunMode.BACKTEST, symbols=["XRP/USD"], initial_capital=Decimal("10000"))


class TestG4CapitalBand:
    @pytest.mark.parametrize("capital", ["59.99", "66.01"])
    def test_out_of_band_rejected(self, capital: str) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(initial_capital=Decimal(capital))
        assert "capital_out_of_smoke_range" in _reasons(exc.value)

    @pytest.mark.parametrize("capital", ["60", "66"])
    def test_boundary_inclusive(self, capital: str) -> None:
        _call(initial_capital=Decimal(capital))

    def test_only_applies_live(self) -> None:
        _call(mode=RunMode.PAPER, initial_capital=Decimal("10000"))
        _call(mode=RunMode.BACKTEST, initial_capital=Decimal("10000"))


class TestG5RiskCeiling:
    def test_capital_10_notional_9_violates_ceiling(self) -> None:
        # 0.15 * 10 = 1.50 < 9.00 -- SMK-R-01's negative control, enforced
        # in EVERY mode (not just live).
        with pytest.raises(SmokeGuardError) as exc:
            _call(
                mode=RunMode.BACKTEST,
                initial_capital=Decimal("10"),
                strategy_params={"notional_quote": 9.0},
            )
        assert "notional_exceeds_risk_ceiling" in _reasons(exc.value)

    def test_notional_exactly_at_ceiling_passes(self) -> None:
        max_pct = Decimal(str(RiskParameters().max_position_size_pct))
        # capital=60 -> ceiling 9.00, comfortably inside the strategy's own
        # [5.00, 9.50] hard bound too (G-6), so only G-5 is exercised here.
        capital = Decimal("60")
        notional = max_pct * capital  # exactly at the ceiling
        _call(
            initial_capital=capital,
            strategy_params={"notional_quote": float(notional)},
        )

    def test_notional_one_cent_over_ceiling_fails(self) -> None:
        max_pct = Decimal(str(RiskParameters().max_position_size_pct))
        capital = Decimal("60")  # 0.15 * 60 = 9.00
        notional = max_pct * capital + Decimal("0.01")
        with pytest.raises(SmokeGuardError) as exc:
            _call(initial_capital=capital, strategy_params={"notional_quote": float(notional)})
        assert "notional_exceeds_risk_ceiling" in _reasons(exc.value)


class TestG6ParamBounds:
    @pytest.mark.parametrize(
        ("params", "expected_reason"),
        [
            ({"notional_quote": 4.99}, "param_out_of_range"),
            ({"notional_quote": 9.51}, "param_out_of_range"),
            ({"hold_bars": 0}, "param_out_of_range"),
            ({"hold_bars": 13}, "param_out_of_range"),
            ({"exit_retry_bars": 0}, "param_out_of_range"),
            ({"exit_retry_bars": 6}, "param_out_of_range"),
            ({"unknown_key": 1}, "unknown_param"),
        ],
    )
    def test_out_of_range_and_unknown_params(
        self, params: dict[str, object], expected_reason: str
    ) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params=params)
        assert expected_reason in _reasons(exc.value)

    def test_valid_bounds_pass(self) -> None:
        _call(strategy_params={"notional_quote": 5.00, "hold_bars": 12, "exit_retry_bars": 5})

    def test_collects_multiple_issues_at_once(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(
                symbols=["A/EUR", "B/EUR"],
                timeframe="15m",
                strategy_params={"notional_quote": 999.0, "unknown_key": 1},
            )
        reasons = _reasons(exc.value)
        assert {"too_many_symbols", "timeframe_not_allowed", "unknown_param"} <= reasons
        assert len(exc.value.issues) >= 3


class TestG7Bracket:
    def test_missing_stop_loss_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={})
        assert "stop_loss_required" in _reasons(exc.value)

    def test_blank_stop_loss_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={"bracket_stop_loss_pct": None})
        assert "stop_loss_required" in _reasons(exc.value)

    @pytest.mark.parametrize("sl", [0.0299, 0.0801, 0.02, 0.09])
    def test_out_of_range_stop_loss_rejected(self, sl: float) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": sl})
        assert "stop_loss_out_of_range" in _reasons(exc.value)

    @pytest.mark.parametrize("sl", [0.03, 0.08, 0.05])
    def test_boundary_and_default_stop_loss_pass(self, sl: float) -> None:
        _call(bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": sl})

    def test_bracket_mode_atr_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={"bracket_mode": "atr", "bracket_stop_loss_pct": 0.05})
        assert "bracket_mode_must_be_fixed" in _reasons(exc.value)

    def test_take_profit_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(
                bracket={
                    "bracket_mode": "fixed",
                    "bracket_stop_loss_pct": 0.05,
                    "bracket_take_profit_pct": 0.1,
                }
            )
        assert "take_profit_not_allowed" in _reasons(exc.value)

    def test_trailing_stop_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(trailing_stop_pct=0.02)
        assert "trailing_not_allowed" in _reasons(exc.value)

    def test_blank_trailing_stop_pct_passes(self) -> None:
        for blank in (None, "", 0, 0.0):
            _call(trailing_stop_pct=blank)


class TestG8PyramidingAndAdaptiveLearning:
    def test_allow_pyramiding_true_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(allow_pyramiding=True)
        assert "pyramiding_not_allowed" in _reasons(exc.value)

    def test_enable_adaptive_learning_true_rejected(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(enable_adaptive_learning=True)
        assert "adaptive_learning_not_allowed" in _reasons(exc.value)

    def test_false_and_none_pass(self) -> None:
        _call(allow_pyramiding=False, enable_adaptive_learning=False)
        _call(allow_pyramiding=None, enable_adaptive_learning=None)


class TestErrorShape:
    def test_error_code_is_fixed(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={})
        assert exc.value.code == "smoke_guardrail_violation"

    def test_issue_fields_present(self) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={})
        issue = exc.value.issues[0]
        assert issue.field
        assert issue.reason
        # value/min/max/message must all be safely accessible (may be None).
        _ = (issue.value, issue.min, issue.max, issue.message)


# ===========================================================================
# SMK-T-07: availability is DIAGNOSTIC with _ALL_MODES; names match.
# ===========================================================================


class TestAvailabilityConsistency:
    def test_smoke_roundtrip_is_diagnostic_all_modes(self) -> None:
        avail = get_availability("smoke_roundtrip")
        assert avail.status is StrategyStatus.DIAGNOSTIC
        for mode in (RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE):
            assert is_mode_allowed("smoke_roundtrip", mode) is True

    def test_diagnostic_names_equal_smoke_strategy_names(self) -> None:
        import trading.strategy_availability as sa

        diagnostic_names = {
            name
            for name, record in sa._AVAILABILITY.items()
            if record.status is StrategyStatus.DIAGNOSTIC
        }
        assert diagnostic_names == SMOKE_STRATEGY_NAMES


# ===========================================================================
# SMK-T-08: sizing through DefaultRiskManager defaults, via
# calculate_position_size + pre_trade_check -- proves the blast-radius
# ceiling independently of the guard (defence in depth).
# ===========================================================================


class TestSizingBlastRadius:
    def _size(
        self, *, capital: Decimal, target_notional: Decimal, price: Decimal = Decimal("100")
    ) -> Decimal:
        risk_manager = DefaultRiskManager(run_id="smk-t-08")
        # A stop-loss 5% below entry (D-SMK-3 default) determines the risk
        # distance for calculate_position_size's fixed-fractional sizing.
        stop_loss_price = price * Decimal("0.95")
        qty = risk_manager.calculate_position_size(
            equity=capital,
            entry_price=price,
            stop_loss_price=stop_loss_price,
            confidence=1.0,
        )
        # calculate_position_size sizes off risk distance, not the
        # strategy's requested notional -- cap at whichever is smaller,
        # mirroring _resolve_order_quantity's basis-then-cap flow closely
        # enough to prove the SAME blast-radius ceiling this guard's G-5
        # reads at runtime.
        requested_qty = target_notional / price
        return min(qty, requested_qty) * price

    def test_capital_10_yields_150_cents_not_9_euro(self) -> None:
        notional = self._size(capital=Decimal("10"), target_notional=Decimal("9"))
        assert notional <= Decimal("1.50")

    def test_capital_60_target_9_fills_at_9(self) -> None:
        notional = self._size(capital=Decimal("60"), target_notional=Decimal("9"))
        assert notional == Decimal("9") or abs(notional - Decimal("9")) < Decimal("0.5")

    def test_capital_65_target_9_fills_at_9(self) -> None:
        notional = self._size(capital=Decimal("65"), target_notional=Decimal("9"))
        assert abs(notional - Decimal("9")) < Decimal("0.5")

    def test_capital_66_target_1000_capped_at_990(self) -> None:
        notional = self._size(capital=Decimal("66"), target_notional=Decimal("1000"))
        assert notional <= Decimal("9.90")


# ===========================================================================
# WP-SMOKE fix F-2 (SMK-SEC-03) / SMK-T-38: non-finite Decimal input must
# raise SmokeGuardError, never decimal.InvalidOperation / ArithmeticError.
# ===========================================================================


class TestF2NonFiniteInputs:
    """SMK-T-38: every numeric-ish smoke_guard field must fail closed (a
    clean SmokeGuardError, 422 at the API layer) on NaN/sNaN/+-Infinity,
    never raise decimal.InvalidOperation (which would surface as a 500)."""

    _NON_FINITE: ClassVar[list[object]] = [float("nan"), "NaN", "sNaN", float("inf"), "-Infinity"]

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_notional_quote_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params={"notional_quote": bad})
        assert "param_out_of_range" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_hold_bars_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params={"hold_bars": bad})
        assert "param_out_of_range" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_exit_retry_bars_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params={"exit_retry_bars": bad})
        assert "param_out_of_range" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_bracket_stop_loss_pct_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": bad})
        assert "stop_loss_out_of_range" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_bracket_take_profit_pct_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(
                bracket={
                    "bracket_mode": "fixed",
                    "bracket_stop_loss_pct": 0.05,
                    "bracket_take_profit_pct": bad,
                }
            )
        assert "take_profit_not_allowed" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_bracket_atr_sl_multiplier_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(
                bracket={
                    "bracket_mode": "fixed",
                    "bracket_stop_loss_pct": 0.05,
                    "bracket_atr_sl_multiplier": bad,
                }
            )
        assert "take_profit_not_allowed" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_trailing_stop_pct_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(trailing_stop_pct=bad)
        assert "trailing_not_allowed" in _reasons(exc.value)

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_live_initial_capital_non_finite(self, bad: object) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(initial_capital=bad)
        assert "capital_out_of_smoke_range" in _reasons(exc.value)


# ===========================================================================
# WP-SMOKE fix F-3 (SMK-SEC-05) / SMK-T-40: bounded unknown-param
# reflection -- truncated field, static message, capped issue count.
# ===========================================================================


class TestF3BoundedUnknownParamReflection:
    def test_long_key_truncated_and_message_has_no_key(self) -> None:
        long_key = "x" * 500
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params={long_key: 1})
        issues = [i for i in exc.value.issues if i.reason == "unknown_param"]
        assert len(issues) == 1
        assert len(issues[0].field) == 64
        assert long_key not in issues[0].message
        assert "recognised" in issues[0].message

    def test_fifty_unknown_keys_capped_with_one_summary_issue(self) -> None:
        params = {f"unknown_{i}": i for i in range(50)}
        with pytest.raises(SmokeGuardError) as exc:
            _call(strategy_params=params)
        unknown_issues = [i for i in exc.value.issues if i.reason == "unknown_param"]
        assert len(unknown_issues) == 11
        summary = unknown_issues[-1]
        assert summary.field == "strategy_params"
        assert summary.value == "50"


# ===========================================================================
# WP-SMOKE fix F-4 (SMK-SEC-06) / SMK-T-41: fail closed on an unrecognised
# mode string; existing RunMode members are unaffected.
# ===========================================================================


class TestF4FailClosedOnUnknownMode:
    @pytest.mark.parametrize("bad_mode", ["live_sandbox", ""])
    def test_unrecognised_mode_rejected(self, bad_mode: str) -> None:
        with pytest.raises(SmokeGuardError) as exc:
            _call(mode=bad_mode)
        assert "mode_not_recognised" in _reasons(exc.value)

    def test_uppercase_live_string_treated_as_live(self) -> None:
        # G-4 fires for an out-of-band capital, proving is_live derived
        # "LIVE" -> True (case-insensitive normalisation).
        with pytest.raises(SmokeGuardError) as exc:
            _call(mode="LIVE", initial_capital=Decimal("1000"))
        assert "capital_out_of_smoke_range" in _reasons(exc.value)
        assert "mode_not_recognised" not in _reasons(exc.value)

    def test_runmode_members_unaffected(self) -> None:
        _call(mode=RunMode.LIVE)
        _call(mode=RunMode.PAPER, initial_capital=Decimal("10000"))
        _call(mode=RunMode.BACKTEST, initial_capital=Decimal("10000"))


# ===========================================================================
# WP-SMOKE fix F-7 (SMK-CR-01) / SMK-T-43: a stray, caller-supplied
# bracket_atr_period on an otherwise-valid fixed-mode config must NOT be
# rejected -- pins the corrected rationale (validate_run_exit_config
# normalises bracket_atr_period to 14 AFTER this guard runs at create
# time, and that normalised value is what every later G-13 call sees).
# ===========================================================================


class TestF7BracketAtrPeriodNotChecked:
    def test_bracket_atr_period_present_does_not_raise(self) -> None:
        _call(
            mode=RunMode.LIVE,
            bracket={
                "bracket_mode": "fixed",
                "bracket_atr_period": 14,
                "bracket_stop_loss_pct": 0.05,
            },
        )  # must not raise
