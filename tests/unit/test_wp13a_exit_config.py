"""
tests/unit/test_wp13a_exit_config.py
--------------------------------------
WP1.3a (reports/vp2-wp1.3a/synthesis-spec.md) -- unit coverage for the pure
validator module ``packages/trading/exit_config.py``.

Covers a representative slice of the mandatory ST-01..36 table: the core
bounds (E1-E11), zero-normalisation (SY-13a-02), `requires_exit_manager`
declarations on every registry strategy (ST-27), `resolve_allow_pyramiding`,
`salvage_exit_config` (protective-resume waiver), and
`validate_param_grid_exit_config` (optimizer pre-check).
"""
from __future__ import annotations

from typing import Any, ClassVar

import pytest

from common.types import RunMode
from trading.exit_config import (
    ExitConfigError,
    ExitCostModel,
    _trunc,  # WP13a-S-R4-03 unit test only
    parse_exit_config,
    require_exit_manager,
    resolve_allow_pyramiding,
    salvage_exit_config,
    strategy_requires_exit_manager,
    validate_param_grid_exit_config,
    validate_run_exit_config,
)
from trading.strategy import BaseStrategy

_COST = ExitCostModel.from_risk_parameters()
_C_SIDE = _COST.c_side  # 0.0065 with code defaults
_C_RT = _COST.c_rt  # 0.013


class _Requires(BaseStrategy):
    requires_exit_manager: ClassVar[bool] = True

    def on_bar(self, bars: Any, *, mtf_context: Any = None) -> list[Any]:
        return []

    @property
    def min_bars_required(self) -> int:
        return 1


class _NotRequires(BaseStrategy):
    requires_exit_manager: ClassVar[bool] = False

    def on_bar(self, bars: Any, *, mtf_context: Any = None) -> list[Any]:
        return []

    @property
    def min_bars_required(self) -> int:
        return 1


class _AccumulatesByDefault(BaseStrategy):
    requires_exit_manager: ClassVar[bool] = False
    default_allow_pyramiding: ClassVar[bool] = True

    def on_bar(self, bars: Any, *, mtf_context: Any = None) -> list[Any]:
        return []

    @property
    def min_bars_required(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# ST-01/02: not-a-number / not-finite / invalid_type
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_non_numeric_string_is_not_a_number(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(bracket={"bracket_stop_loss_pct": "abc"}, trailing_stop_pct=None)
        assert exc.value.code == "invalid_exit_config"
        assert exc.value.issues[0].reason == "not_a_number"

    def test_nan_and_inf_are_not_finite(self) -> None:
        for bad in ("nan", "inf", float("nan"), float("inf")):
            with pytest.raises(ExitConfigError) as exc:
                parse_exit_config(bracket={"bracket_stop_loss_pct": bad}, trailing_stop_pct=None)
            assert exc.value.issues[0].reason == "not_finite"

    def test_bool_is_invalid_type_not_unset(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(bracket={"bracket_stop_loss_pct": True}, trailing_stop_pct=None)
        assert exc.value.issues[0].reason == "invalid_type"

    def test_atr_period_integral_float_accepted_non_integral_rejected(self) -> None:
        cfg = parse_exit_config(
            bracket={
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 1.5,
                "bracket_atr_period": 12.0,
            },
            trailing_stop_pct=None,
        )
        assert cfg.bracket["bracket_atr_period"] == 12

        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(
                bracket={
                    "bracket_mode": "atr",
                    "bracket_atr_sl_multiplier": 1.5,
                    "bracket_atr_period": 12.5,
                },
                trailing_stop_pct=None,
            )
        assert exc.value.issues[0].field == "bracket_atr_period"
        assert exc.value.issues[0].reason == "invalid_type"

    def test_invalid_bracket_mode(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(bracket={"bracket_mode": "trailing"}, trailing_stop_pct=None)
        assert exc.value.issues[0].reason == "invalid_choice"


# ---------------------------------------------------------------------------
# SY-13a-02: zero/blank means unset (except bracket_atr_period)
# ---------------------------------------------------------------------------


class TestZeroMeansUnset:
    def test_zero_and_blank_sl_are_unset(self) -> None:
        for raw in (0, 0.0, "0", None, ""):
            cfg = parse_exit_config(
                bracket={"bracket_stop_loss_pct": raw}, trailing_stop_pct=None
            )
            assert cfg.bracket == {}

    def test_zero_trailing_is_unset(self) -> None:
        cfg = parse_exit_config(bracket=None, trailing_stop_pct=0.0)
        assert cfg.trailing_stop_pct is None

    def test_atr_period_zero_is_hard_422_not_unset(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(
                bracket={
                    "bracket_mode": "atr",
                    "bracket_atr_sl_multiplier": 1.5,
                    "bracket_atr_period": 0,
                },
                trailing_stop_pct=None,
            )
        assert exc.value.issues[0].field == "bracket_atr_period"
        assert exc.value.issues[0].reason == "out_of_range"

    def test_ui_shaped_momentum_payload_atr_active_fixed_zero_dropped(self) -> None:
        """ST-04: ATR 1.5/3.0 active plus fixed-mode 0/0 (UI-sent, inactive) --
        the normalised bracket keeps only the ATR keys, mode and period."""
        cfg = parse_exit_config(
            bracket={
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 1.5,
                "bracket_atr_tp_multiplier": 3.0,
                "bracket_atr_period": 14,
                "bracket_stop_loss_pct": 0,
                "bracket_take_profit_pct": 0,
            },
            trailing_stop_pct=None,
        )
        assert cfg.bracket == {
            "bracket_mode": "atr",
            "bracket_atr_period": 14,
            "bracket_atr_sl_multiplier": 1.5,
            "bracket_atr_tp_multiplier": 3.0,
        }

    def test_inactive_mode_non_zero_is_422(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(
                bracket={"bracket_mode": "atr", "bracket_stop_loss_pct": 0.02},
                trailing_stop_pct=None,
            )
        assert exc.value.issues[0].reason == "inactive_mode_value"

    def test_mode_and_period_only_counts_as_no_bracket(self) -> None:
        cfg = parse_exit_config(
            bracket={"bracket_mode": "atr", "bracket_atr_period": 14},
            trailing_stop_pct=None,
        )
        assert cfg.bracket == {}
        assert cfg.has_downside_exit is False


# ---------------------------------------------------------------------------
# Bounds: E4/E6/E7/E9/E10
# ---------------------------------------------------------------------------


class TestBounds:
    def test_fixed_sl_at_or_below_cost_is_stop_inside_entry_cost(self) -> None:
        for bad in (0.005, _C_SIDE):
            with pytest.raises(ExitConfigError) as exc:
                parse_exit_config(bracket={"bracket_stop_loss_pct": bad}, trailing_stop_pct=None)
            assert exc.value.issues[0].reason == "stop_inside_entry_cost"

    def test_fixed_sl_just_above_cost_passes_with_warning(self) -> None:
        cfg, issues, warnings = _parse_with_warnings(
            bracket={"bracket_stop_loss_pct": _C_SIDE + 0.0001}, trailing_stop_pct=None
        )
        assert not issues
        assert cfg.bracket["bracket_stop_loss_pct"] == pytest.approx(_C_SIDE + 0.0001)
        assert any(w.code == "sl_below_round_trip_cost" for w in warnings)

    def test_fixed_sl_above_policy_max_out_of_range(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(bracket={"bracket_stop_loss_pct": 0.51}, trailing_stop_pct=None)
        assert exc.value.issues[0].reason == "out_of_range"

    def test_fixed_tp_at_or_below_cost_is_take_profit_inside_exit_cost(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            parse_exit_config(bracket={"bracket_take_profit_pct": 0.006}, trailing_stop_pct=None)
        assert exc.value.issues[0].reason == "take_profit_inside_exit_cost"

    def test_atr_multiplier_out_of_range(self) -> None:
        for bad in (0.05, 25.0):
            with pytest.raises(ExitConfigError) as exc:
                parse_exit_config(
                    bracket={"bracket_mode": "atr", "bracket_atr_sl_multiplier": bad},
                    trailing_stop_pct=None,
                )
            assert exc.value.issues[0].reason == "out_of_range"

    def test_atr_period_upper_bound_99_uniform(self) -> None:
        parse_exit_config(
            bracket={
                "bracket_mode": "atr",
                "bracket_atr_sl_multiplier": 1.5,
                "bracket_atr_period": 99,
            },
            trailing_stop_pct=None,
        )
        with pytest.raises(ExitConfigError):
            parse_exit_config(
                bracket={
                    "bracket_mode": "atr",
                    "bracket_atr_sl_multiplier": 1.5,
                    "bracket_atr_period": 100,
                },
                trailing_stop_pct=None,
            )

    def test_trailing_out_of_range(self) -> None:
        for bad in (0.6, 1.0, 0.004):
            with pytest.raises(ExitConfigError) as exc:
                parse_exit_config(bracket=None, trailing_stop_pct=bad)
            assert exc.value.issues[0].reason == "out_of_range"

    def test_trailing_valid_with_below_cost_warning(self) -> None:
        cfg, issues, warnings = _parse_with_warnings(bracket=None, trailing_stop_pct=0.01)
        assert not issues
        assert cfg.trailing_stop_pct == 0.01
        assert any(w.code == "trailing_below_round_trip_cost" for w in warnings)

    def test_tp_only_does_not_satisfy_has_downside_exit(self) -> None:
        cfg = parse_exit_config(bracket={"bracket_take_profit_pct": 0.05}, trailing_stop_pct=None)
        assert cfg.has_downside_exit is False

    def test_trailing_only_satisfies_has_downside_exit(self) -> None:
        cfg = parse_exit_config(bracket=None, trailing_stop_pct=0.02)
        assert cfg.has_downside_exit is True

    def test_reward_risk_warning(self) -> None:
        _, _, warnings = _parse_with_warnings(
            bracket={"bracket_stop_loss_pct": 0.05, "bracket_take_profit_pct": 0.03},
            trailing_stop_pct=None,
        )
        assert any(w.code == "reward_risk_le_1" for w in warnings)

    def test_risk_budget_warning_not_hard_fail(self) -> None:
        # E8 was demoted to W3 (SY-13a-04) -- SL 0.12/0.13 must be 201, not 422.
        for sl in (0.12, 0.13):
            _cfg, issues, warnings = _parse_with_warnings(
                bracket={"bracket_stop_loss_pct": sl}, trailing_stop_pct=None
            )
            assert not issues
            assert any(w.code == "stop_exceeds_risk_budget" for w in warnings)


def _parse_with_warnings(*, bracket: dict[str, object] | None, trailing_stop_pct: object):
    from trading.exit_config import _parse_core

    return _parse_core(bracket, trailing_stop_pct, _COST)


# ---------------------------------------------------------------------------
# requires_exit_manager / registry declarations (ST-27)
# ---------------------------------------------------------------------------


class TestRequiresExitManager:
    def test_magicmock_is_not_required(self) -> None:
        from unittest.mock import MagicMock

        mock_strategy = MagicMock()
        assert strategy_requires_exit_manager(mock_strategy) is False

    def test_require_exit_manager_raises_for_requiring_strategy_with_no_exit(self) -> None:
        cfg = parse_exit_config(bracket=None, trailing_stop_pct=None)
        with pytest.raises(ExitConfigError) as exc:
            require_exit_manager([_Requires("s")], cfg)
        assert exc.value.code == "exit_manager_required"
        assert exc.value.requires_one_of is not None

    def test_require_exit_manager_passes_for_non_requiring_strategy(self) -> None:
        cfg = parse_exit_config(bracket=None, trailing_stop_pct=None)
        require_exit_manager([_NotRequires("s")], cfg)  # must not raise

    def test_registry_declarations(self) -> None:
        """ST-27 (WP-SMOKE, reports/vp2-smoke/synthesis-spec.md section 8):
        SmokeRoundtripStrategy declares requires_exit_manager=True
        explicitly, exactly like every other registry strategy."""
        from trading.strategies import (
            BreakoutStrategy,
            DCARSIHybridStrategy,
            GridTradingStrategy,
            MACrossoverStrategy,
            ModelStrategy,
            MomentumBreakoutStrategy,
            RSIMeanReversionStrategy,
            SLTPReversionStrategy,
            SmokeRoundtripStrategy,
        )

        requires_true = {MomentumBreakoutStrategy, SLTPReversionStrategy, SmokeRoundtripStrategy}
        pyramiding_true = {DCARSIHybridStrategy, GridTradingStrategy}
        every_cls = {
            MACrossoverStrategy,
            RSIMeanReversionStrategy,
            BreakoutStrategy,
            ModelStrategy,
            DCARSIHybridStrategy,
            GridTradingStrategy,
            SLTPReversionStrategy,
            MomentumBreakoutStrategy,
            SmokeRoundtripStrategy,
        }
        for cls in every_cls:
            assert "requires_exit_manager" in cls.__dict__, cls
            assert "default_allow_pyramiding" in cls.__dict__ or cls not in pyramiding_true
            expected_required = cls in requires_true
            assert cls.__dict__["requires_exit_manager"] is expected_required, cls
            expected_pyramiding = cls in pyramiding_true
            actual_pyramiding = cls.__dict__.get("default_allow_pyramiding", False)
            assert actual_pyramiding is expected_pyramiding, cls


# ---------------------------------------------------------------------------
# resolve_allow_pyramiding (SY-13a-08)
# ---------------------------------------------------------------------------


class TestResolveAllowPyramiding:
    def test_explicit_wins(self) -> None:
        assert resolve_allow_pyramiding(True, [_NotRequires]) is True
        assert resolve_allow_pyramiding(False, [_AccumulatesByDefault]) is False

    def test_default_false_for_ordinary_strategy(self) -> None:
        assert resolve_allow_pyramiding(None, [_NotRequires]) is False

    def test_default_true_only_for_accumulating_strategy(self) -> None:
        assert resolve_allow_pyramiding(None, [_AccumulatesByDefault]) is True

    def test_mixed_strategies_all_must_default_true(self) -> None:
        assert resolve_allow_pyramiding(None, [_AccumulatesByDefault, _NotRequires]) is False


# ---------------------------------------------------------------------------
# E11 / live pyramiding ban + W7/W8
# ---------------------------------------------------------------------------


class TestLivePyramidingBan:
    def test_live_resolved_true_raises(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            validate_run_exit_config(
                _AccumulatesByDefault,
                bracket=None,
                trailing_stop_pct=0.02,
                mode=RunMode.LIVE,
                allow_pyramiding=None,
            )
        assert exc.value.code == "live_pyramiding_forbidden"
        assert exc.value.hint is not None

    def test_live_explicit_false_gives_w7(self) -> None:
        verdict = validate_run_exit_config(
            _AccumulatesByDefault,
            bracket=None,
            trailing_stop_pct=0.02,
            mode=RunMode.LIVE,
            allow_pyramiding=False,
        )
        assert verdict.allow_pyramiding is False
        assert any(w.code == "accumulation_disabled" for w in verdict.warnings)

    def test_paper_resolved_true_is_fine(self) -> None:
        verdict = validate_run_exit_config(
            _AccumulatesByDefault,
            bracket=None,
            trailing_stop_pct=0.02,
            mode=RunMode.PAPER,
            allow_pyramiding=None,
        )
        assert verdict.allow_pyramiding is True

    def test_live_no_downside_exit_not_required_gives_w8(self) -> None:
        verdict = validate_run_exit_config(
            _NotRequires,
            bracket=None,
            trailing_stop_pct=None,
            mode=RunMode.LIVE,
            allow_pyramiding=None,
        )
        assert any(w.code == "no_downside_exit" for w in verdict.warnings)

    def test_backtest_no_downside_exit_no_w8(self) -> None:
        verdict = validate_run_exit_config(
            _NotRequires,
            bracket=None,
            trailing_stop_pct=None,
            mode=RunMode.BACKTEST,
            allow_pyramiding=None,
        )
        assert not any(w.code == "no_downside_exit" for w in verdict.warnings)

    def test_unrecognised_mode_string_raises_wp13a_s03(self) -> None:
        """WP13a-S-03 (security round 2): an unrecognised mode value must
        raise -- not silently skip E11 (case-sensitive: "LIVE"/"Live"/
        " live" are all rejected, not coerced to RunMode.LIVE)."""
        for bad_mode in ("LIVE", "Live", " live", "not_a_mode", ""):
            with pytest.raises(ExitConfigError) as exc:
                validate_run_exit_config(
                    _AccumulatesByDefault,
                    bracket=None,
                    trailing_stop_pct=0.02,
                    mode=bad_mode,
                    allow_pyramiding=True,
                )
            assert exc.value.code == "invalid_exit_config"
            assert exc.value.issues[0].field == "mode"
            assert exc.value.issues[0].reason == "invalid_choice"

    def test_recognised_mode_enum_instance_passes_through(self) -> None:
        """A real RunMode instance is accepted as-is (no re-parse needed)."""
        verdict = validate_run_exit_config(
            _NotRequires,
            bracket=None,
            trailing_stop_pct=0.02,
            mode=RunMode.LIVE,
            allow_pyramiding=None,
        )
        assert verdict.allow_pyramiding is False


# ---------------------------------------------------------------------------
# salvage_exit_config (protective resume waiver, SY-13a-16)
# ---------------------------------------------------------------------------


class TestSalvage:
    def test_invalid_bracket_valid_trailing_keeps_only_trailing(self) -> None:
        cfg, issues = salvage_exit_config(
            bracket={"bracket_mode": "atr", "bracket_stop_loss_pct": 0.02},  # inactive-mode 422
            trailing_stop_pct=0.02,
        )
        assert cfg.bracket == {}
        assert cfg.trailing_stop_pct == 0.02
        assert issues  # the dropped bracket component is listed
        assert cfg.has_downside_exit is True

    def test_both_invalid_gives_no_downside_exit(self) -> None:
        cfg, issues = salvage_exit_config(
            bracket={"bracket_mode": "atr", "bracket_stop_loss_pct": 0.02},
            trailing_stop_pct=0.6,  # out of range
        )
        assert cfg.bracket == {}
        assert cfg.trailing_stop_pct is None
        assert cfg.has_downside_exit is False
        assert len(issues) == 2

    def test_both_valid_no_issues(self) -> None:
        cfg, issues = salvage_exit_config(
            bracket={"bracket_stop_loss_pct": 0.05}, trailing_stop_pct=0.02
        )
        assert not issues
        assert cfg.bracket["bracket_stop_loss_pct"] == 0.05
        assert cfg.trailing_stop_pct == 0.02

    def test_huge_int_gives_not_finite_not_overflow_error(self) -> None:
        """WP13a-S-R5-03 (security round 5, AC3): salvage runs the same
        ``_coerce_numeric`` as every other validator path -- a Python int
        far outside float range must never raise an uncaught
        ``OverflowError`` here either (protective resume's salvage path
        has no exception handler around this call at all -- an uncaught
        OverflowError here would crash the resume endpoint)."""
        huge_int = 10**3999
        cfg, issues = salvage_exit_config(
            bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": huge_int},
            trailing_stop_pct=None,
        )
        assert cfg.bracket == {}
        assert cfg.has_downside_exit is False
        assert len(issues) == 1
        assert issues[0].reason == "not_finite"
        assert issues[0].field == "bracket_stop_loss_pct"
        assert len(issues[0].value) <= 64


# ---------------------------------------------------------------------------
# validate_param_grid_exit_config (optimizer pre-check, SY-13a-17)
# ---------------------------------------------------------------------------


class TestValidateParamGrid:
    def test_no_bracket_keys_for_requiring_strategy_raises(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            validate_param_grid_exit_config(_Requires, {"lookback": [10, 20]})
        assert exc.value.code == "exit_manager_required"

    def test_one_invalid_combination_reported(self) -> None:
        with pytest.raises(ExitConfigError) as exc:
            validate_param_grid_exit_config(
                _Requires,
                {
                    "bracket_stop_loss_pct": [0.02, 0.51],  # second combo out of range
                    "bracket_take_profit_pct": [0.1],
                },
            )
        assert exc.value.code == "invalid_exit_config"
        assert getattr(exc.value, "total_invalid", None) == 1
        # WP13a-C-01: combo_index/params first-class, field is the PLAIN
        # field name (never a composite "combo[i].field" string).
        combo_issues = exc.value.combo_issues
        assert len(combo_issues) == 1
        issue = combo_issues[0]
        assert issue.combo_index == 1
        assert issue.params == {
            "bracket_stop_loss_pct": 0.51,
            "bracket_take_profit_pct": 0.1,
        }
        assert issue.field == "bracket_stop_loss_pct"
        assert issue.reason == "out_of_range"

    def test_valid_grid_does_not_raise(self) -> None:
        validate_param_grid_exit_config(
            _Requires,
            {
                "bracket_stop_loss_pct": [0.02, 0.03],
                "bracket_take_profit_pct": [0.1, 0.2],
            },
        )

    def test_not_requiring_strategy_empty_grid_never_raises(self) -> None:
        validate_param_grid_exit_config(_NotRequires, {"lookback": [10, 20]})


class TestTruncBoundedCostWP13aSR403:
    """WP13a-S-R4-03 (security round 4): ``_trunc`` must never be
    proportional to an arbitrarily large input's real size -- previously
    it called the unbounded ``str()`` over the WHOLE value before
    slicing, so a single ~7MB list, re-validated across up to 1000
    uncacheable grid positions, blocked the event loop for ~43s."""

    def test_scalar_output_is_byte_identical_to_pre_fix_behaviour(self) -> None:
        """Normal scalars must render EXACTLY as the old
        ``str(value)[:64]`` did -- this fix must not change any existing
        test's expectations for str/int/float/bool/None."""
        assert _trunc(None) is None
        assert _trunc(True) == "True"
        assert _trunc(False) == "False"
        assert _trunc(42) == "42"
        assert _trunc(3.14) == "3.14"
        assert _trunc("hello") == "hello"
        long_string = "x" * 100
        assert _trunc(long_string) == long_string[:64]
        assert len(_trunc(long_string)) == 64

    @pytest.mark.parametrize(
        "huge_value",
        [
            list(range(1_000_000)),
            {i: i for i in range(1_000_000)},
            [list(range(1_000_000))],
        ],
    )
    def test_huge_container_is_fast_and_bounded(self, huge_value: object) -> None:
        """A huge list/dict (including nested inside another list, the
        exact shape of the real attack) must render in well under a
        second and never exceed 64 characters."""
        import time

        started = time.perf_counter()
        result = _trunc(huge_value)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5, f"_trunc took {elapsed:.3f}s for a huge container"
        assert result is not None
        assert len(result) <= 64


class TestIdentityKeyedDedupeCacheWP13aCR501:
    """WP13a-C-R5-01 (critic round 5, optional): pin the identity-keyed
    dedupe cache on its own -- 1000 full-grid positions that all share the
    SAME unhashable exit-key value (via an unrelated non-exit dimension)
    must call ``validate_run_exit_config`` exactly ONCE, not 1000 times."""

    def test_1000_positions_sharing_one_unhashable_value_gives_one_validator_call(
        self,
    ) -> None:
        from unittest.mock import patch

        import trading.exit_config as exit_config_module

        huge_value = list(range(1_000_000))
        grid = {
            "lookback": list(range(1000)),
            "bracket_stop_loss_pct": [huge_value],
        }

        spy = patch.object(
            exit_config_module,
            "validate_run_exit_config",
            wraps=exit_config_module.validate_run_exit_config,
        )
        with spy as mock_validate:
            with pytest.raises(ExitConfigError) as exc:
                validate_param_grid_exit_config(_Requires, grid)

        assert mock_validate.call_count == 1
        assert exc.value.total_invalid == 1000
