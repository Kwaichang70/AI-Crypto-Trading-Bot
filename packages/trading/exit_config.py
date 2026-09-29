"""
packages/trading/exit_config.py
--------------------------------
WP1.3a (Verbeterplan v2, ``reports/vp2-wp1.3a/synthesis-spec.md``).

Single, pure validator for the engine-level exit configuration (fixed/ATR
bracket + trailing stop) and the ``allow_pyramiding`` flag.  No engine, DB or
FastAPI imports here -- this module is imported by ``apps.api.routers.runs``,
``apps.api.routers.optimize``, ``apps.api.services.run_orchestrator``,
``trading.strategy_engine``, ``trading.backtest`` and
``scripts/wp13a_exit_config_precheck.py`` alike, so it must stay a leaf.

Design
~~~~~~
``bracket_exit.py`` and ``trailing_stop.py`` export their *mechanical*
constructor ranges (``FIXED_PCT_RANGE``, ``ATR_MULTIPLIER_RANGE``,
``ATR_PERIOD_MIN``, ``TRAILING_PCT_RANGE``).  This module layers the
*policy* bounds on top (SY-13a-04 §1a): the entry-cost floor (E6/E7), the
tightened stop-loss ceiling (E4, 0.50 rather than the constructor's 0.95),
and the ATR period ceiling (E4/E10, 99 rather than 500) -- see the module
constants below.  A config that fails these policy bounds is still
*mechanically* constructible; that is precisely why a shared validator, run
BEFORE construction, is required (a construction-time ``try/except`` can
only ever catch mechanical failures).

"0 means unset" (SY-13a-02): ``None``, ``""`` and exact numeric ``0`` all
mean "unset" for the four bracket pct/multiplier fields and for
``trailing_stop_pct``.  ``bracket_atr_period`` is the one exception: it is
never "off", so ``0`` is a hard 422 (E4), not a silent no-op.

Cost model (SY-13a-01): ``ExitCostModel`` is always built from
``RiskParameters()`` **defaults** (the paper/live cost model), in every run
mode including backtest, so a config that passes a backtest can never later
422 at promote.
"""
from __future__ import annotations

import math
import reprlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import structlog

from common.types import RunMode
from trading.bracket_exit import ATR_MULTIPLIER_RANGE, ATR_PERIOD_MIN, FIXED_PCT_RANGE
from trading.risk import RiskParameters
from trading.trailing_stop import TRAILING_PCT_RANGE

__all__ = [
    "POLICY_ATR_PERIOD_MAX",
    "POLICY_FIXED_SL_MAX",
    "ExitConfig",
    "ExitConfigError",
    "ExitConfigIssue",
    "ExitConfigVerdict",
    "ExitConfigWarning",
    "ExitCostModel",
    "OptimizeComboIssue",
    "parse_exit_config",
    "require_exit_manager",
    "resolve_allow_pyramiding",
    "salvage_exit_config",
    "strategy_requires_exit_manager",
    "validate_param_grid_exit_config",
    "validate_run_exit_config",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy bounds (SY-13a-04 §1a) -- layered on top of the mechanical ranges.
# ---------------------------------------------------------------------------
POLICY_FIXED_SL_MAX: float = 0.50
POLICY_FIXED_TP_MAX: float = FIXED_PCT_RANGE[1]  # 0.95, unchanged
POLICY_ATR_PERIOD_MAX: int = 99

_VALID_BRACKET_MODES: frozenset[str] = frozenset({"fixed", "atr"})
_FIXED_ACTIVE_KEYS: tuple[str, str] = ("bracket_stop_loss_pct", "bracket_take_profit_pct")
_ATR_ACTIVE_KEYS: tuple[str, str] = ("bracket_atr_sl_multiplier", "bracket_atr_tp_multiplier")

_LONG_TIMEFRAMES: frozenset[str] = frozenset({"4h", "1d", "1w"})

_REQUIRES_ONE_OF: tuple[str, str, str] = (
    "bracket_stop_loss_pct (bracket_mode=fixed)",
    "bracket_atr_sl_multiplier (bracket_mode=atr)",
    "trailing_stop_pct",
)


# WP13a-S-R4-03 (security round 4): a single, module-level ``reprlib.Repr``
# instance -- its own maxlevel/maxlist/maxdict/maxstring/maxother caps bound
# the cost of rendering ANY container independently of the container's real
# size, so ``_trunc`` below can never again call an unbounded ``str()`` over
# a huge user-supplied list/dict (previously: a single ~7MB grid value,
# revalidated across up to 1000 uncacheable positions, blocked the event
# loop for ~43s -- 99.7% of that time was ``str(value)`` inside ``_trunc``).
_SAFE_REPR = reprlib.Repr()
_SAFE_REPR.maxlevel = 2
_SAFE_REPR.maxlist = 4
_SAFE_REPR.maxdict = 4
_SAFE_REPR.maxstring = 64
_SAFE_REPR.maxother = 64


def _trunc(value: object) -> str | None:
    """Render ``value`` as a string truncated to 64 chars (SY-13a-18).

    WP13a-S-R4-03 (security round 4): bounded-cost by construction, for
    every input shape -- never proportional to an arbitrarily large
    input's real size:
    - ``None`` -> ``None`` (unchanged).
    - ``str`` -> sliced directly (no ``str()`` call needed; also avoids
      ``reprlib`` adding quotes, keeping scalar output byte-identical to
      the pre-fix behaviour).
    - ``int`` / ``float`` / ``bool`` -> ``str()`` (cheap, fixed-size for
      any real-world number; unchanged from before).
    - anything else (list, dict, tuple, set, nested combinations, ...) ->
      rendered through the bounded ``_SAFE_REPR`` instance above instead
      of the unbounded ``str()``/``repr()``, then sliced to 64 chars same
      as every other branch.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        text = _SAFE_REPR.repr(value)
    return text if len(text) <= 64 else text[:64]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExitConfigIssue:
    field: str
    reason: str
    value: str | None = None
    min: float | None = None
    max: float | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class ExitConfigWarning:
    code: str
    field: str | None
    message: str


@dataclass(frozen=True, slots=True)
class OptimizeComboIssue:
    """WP13a-C-01: one combination-scoped issue for the optimizer's 422
    envelope (``POST /api/v1/optimize``).  Deliberately a SEPARATE type
    from :class:`ExitConfigIssue` -- create/promote/resume's ``errors[]``
    is field-scoped (one run, one config); the optimizer's is
    combination-scoped (one grid, many combinations), and the UI's
    ``OptimizeExitConfigIssue`` type mirrors that split
    (``apps/ui/src/lib/types.ts``). ``field`` is the PLAIN field name --
    never a composite ``combo[i].field`` string."""

    combo_index: int
    params: dict[str, object]
    field: str | None
    reason: str
    value: str | None
    min: float | None
    max: float | None
    message: str


class ExitConfigError(ValueError):
    """Raised by every hard-fail path (SY-13a-01/18).

    ``code`` is one of ``"invalid_exit_config"``, ``"exit_manager_required"``
    or ``"live_pyramiding_forbidden"`` -- the precedence order used when more
    than one applies (SY-13a-18).
    """

    def __init__(
        self,
        code: Literal[
            "invalid_exit_config", "exit_manager_required", "live_pyramiding_forbidden"
        ],
        issues: Iterable[ExitConfigIssue] = (),
        *,
        strategy: str | None = None,
        requires_one_of: tuple[str, ...] | None = None,
        hint: str | None = None,
        warnings: tuple[ExitConfigWarning, ...] = (),
    ) -> None:
        self.code = code
        self.issues: tuple[ExitConfigIssue, ...] = tuple(issues)
        self.strategy = strategy
        self.requires_one_of = requires_one_of
        self.hint = hint
        self.warnings = warnings
        reasons = [i.reason for i in self.issues]
        super().__init__(f"{code}: {reasons or strategy or hint}")


@dataclass(frozen=True, slots=True)
class ExitCostModel:
    """Round-trip cost inputs, always derived from paper/live defaults
    (SY-13a-01) -- ``RiskParameters()`` with no overrides, even in backtest.
    """

    c_side: float
    c_rt: float
    pos_cap: float
    r: float

    @classmethod
    def from_risk_parameters(cls, params: RiskParameters | None = None) -> ExitCostModel:
        p = params if params is not None else RiskParameters()
        c_side = p.taker_fee_pct + p.slippage_bps / 1e4
        return cls(
            c_side=c_side,
            c_rt=2 * c_side,
            pos_cap=p.max_position_size_pct,
            r=p.per_trade_risk_pct,
        )


_DEFAULT_COST_MODEL = ExitCostModel.from_risk_parameters()


@dataclass(frozen=True, slots=True)
class ExitConfig:
    """Normalised, engine-ready exit configuration."""

    bracket: dict[str, object] = field(default_factory=dict)
    trailing_stop_pct: float | None = None

    @property
    def has_downside_exit(self) -> bool:
        """True iff there is an active-mode SL (fixed or ATR) or a trailing
        stop.  A TP-only bracket does NOT satisfy this (SY-13a-05)."""
        if self.trailing_stop_pct is not None:
            return True
        if not self.bracket:
            return False
        mode = self.bracket.get("bracket_mode", "fixed")
        if mode == "atr":
            return self.bracket.get("bracket_atr_sl_multiplier") is not None
        return self.bracket.get("bracket_stop_loss_pct") is not None


@dataclass(frozen=True, slots=True)
class ExitConfigVerdict:
    config: ExitConfig
    allow_pyramiding: bool
    warnings: tuple[ExitConfigWarning, ...] = ()


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_numeric(
    field_name: str, raw: object, *, integral: bool = False
) -> tuple[float | None, ExitConfigIssue | None]:
    """Coerce ``raw`` to a float.

    Returns ``(None, None)`` for "absent" (``None``/``""``).  Returns
    ``(None, issue)`` for anything that is not a finite real number, or
    (when ``integral``) not integral.  Never raises.
    """
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool):
        return None, ExitConfigIssue(
            field=field_name,
            reason="invalid_type",
            value=_trunc(raw),
            message=f"{field_name} must be numeric, not a boolean",
        )
    if isinstance(raw, (int, float)):
        try:
            num = float(raw)
        except OverflowError:
            # WP13a-S-R5-03 (security round 5, AC3): a Python ``int`` far
            # outside float range (e.g. a ~4000-digit JSON integer --
            # ``float(str)`` clamps to +/-inf, but ``float(int)`` RAISES
            # instead) must classify as a clean 422, not an uncaught 500.
            # Chosen reason: ``not_finite`` (documented here, per SY-13a's
            # own convention) -- semantically this value IS effectively
            # +/-infinity in float terms, matching the existing
            # ``math.isinf``/``math.isnan`` -> ``not_finite`` branch below.
            return None, ExitConfigIssue(
                field=field_name,
                reason="not_finite",
                value=_trunc(raw),
                message=f"{field_name} must be a finite number",
            )
    elif isinstance(raw, str):
        try:
            num = float(raw)
        except ValueError:
            return None, ExitConfigIssue(
                field=field_name,
                reason="not_a_number",
                value=_trunc(raw),
                message=f"{field_name} is not a number",
            )
    else:
        return None, ExitConfigIssue(
            field=field_name,
            reason="invalid_type",
            value=_trunc(raw),
            message=f"{field_name} has an unsupported type",
        )
    if math.isnan(num) or math.isinf(num):
        return None, ExitConfigIssue(
            field=field_name,
            reason="not_finite",
            value=_trunc(raw),
            message=f"{field_name} must be a finite number",
        )
    if integral and num != int(num):
        return None, ExitConfigIssue(
            field=field_name,
            reason="invalid_type",
            value=_trunc(raw),
            message=f"{field_name} must be an integer (e.g. 12.0, not 12.5)",
        )
    return num, None


def _strategy_class(obj: object) -> type:
    return obj if isinstance(obj, type) else type(obj)


def _strategy_name(obj: object) -> str:
    cls = _strategy_class(obj)
    meta = getattr(cls, "metadata", None)
    name = getattr(meta, "name", None) if meta is not None else None
    return str(name) if name else cls.__name__


# ---------------------------------------------------------------------------
# Core parse (shared by parse_exit_config / salvage_exit_config /
# validate_run_exit_config)
# ---------------------------------------------------------------------------


def _parse_core(
    bracket: Mapping[str, object] | None,
    trailing_stop_pct: object,
    cost: ExitCostModel,
) -> tuple[ExitConfig, list[ExitConfigIssue], list[ExitConfigWarning]]:
    issues: list[ExitConfigIssue] = []
    warnings: list[ExitConfigWarning] = []
    raw_bracket: Mapping[str, object] = bracket or {}

    # -- bracket_mode --------------------------------------------------
    raw_mode = raw_bracket.get("bracket_mode")
    if raw_mode is None or raw_mode == "":
        mode = "fixed"
    elif isinstance(raw_mode, str) and raw_mode in _VALID_BRACKET_MODES:
        mode = raw_mode
    else:
        issues.append(
            ExitConfigIssue(
                field="bracket_mode",
                reason="invalid_choice",
                value=_trunc(raw_mode),
                message="bracket_mode must be 'fixed' or 'atr' (case-sensitive)",
            )
        )
        mode = "fixed"  # keep parsing so every issue in the payload is collected

    # -- bracket_atr_period (checked whenever present, regardless of mode;
    #    SY-13a-02: never "off", so 0 is always a hard failure) -----------
    period: int | None = None
    period_num, period_issue = _coerce_numeric(
        "bracket_atr_period", raw_bracket.get("bracket_atr_period"), integral=True
    )
    if period_issue is not None:
        issues.append(period_issue)
    elif period_num is not None:
        if not (ATR_PERIOD_MIN <= period_num <= POLICY_ATR_PERIOD_MAX):
            issues.append(
                ExitConfigIssue(
                    field="bracket_atr_period",
                    reason="out_of_range",
                    value=_trunc(period_num),
                    min=ATR_PERIOD_MIN,
                    max=POLICY_ATR_PERIOD_MAX,
                    message=(
                        f"bracket_atr_period must be in [{ATR_PERIOD_MIN}, "
                        f"{POLICY_ATR_PERIOD_MAX}]"
                    ),
                )
            )
        else:
            period = int(period_num)

    active_keys = _FIXED_ACTIVE_KEYS if mode == "fixed" else _ATR_ACTIVE_KEYS
    inactive_keys = _ATR_ACTIVE_KEYS if mode == "fixed" else _FIXED_ACTIVE_KEYS

    # -- inactive-mode fields (SY-13a-03/E3): non-zero is a hard failure,
    #    zero/blank is silently dropped -----------------------------------
    for key in inactive_keys:
        num, issue = _coerce_numeric(key, raw_bracket.get(key))
        if issue is not None:
            issues.append(issue)
            continue
        if num is not None and num != 0.0:
            other_mode = "atr" if mode == "fixed" else "fixed"
            issues.append(
                ExitConfigIssue(
                    field=key,
                    reason="inactive_mode_value",
                    value=_trunc(num),
                    message=(
                        f"{key} belongs to bracket_mode={other_mode!r}, but "
                        f"the active mode is {mode!r}; clear it or set "
                        "bracket_mode accordingly"
                    ),
                )
            )
        elif num == 0.0:
            logger.debug(
                "exit_config.inactive_fields_ignored",
                field=key,
                active_mode=mode,
            )

    # -- active-mode fields ------------------------------------------------
    resolved: dict[str, float] = {}
    for key in active_keys:
        num, issue = _coerce_numeric(key, raw_bracket.get(key))
        if issue is not None:
            issues.append(issue)
            continue
        if num is None or num == 0.0:
            continue  # unset (SY-13a-02)

        if key == "bracket_stop_loss_pct":
            if num <= cost.c_side:
                issues.append(
                    ExitConfigIssue(
                        field=key,
                        reason="stop_inside_entry_cost",
                        value=_trunc(num),
                        min=cost.c_side,
                        message=(
                            f"fixed stop-loss {num} must exceed the round-trip "
                            f"entry cost ({cost.c_side:.4f}); it would fire on "
                            "the entry bar by construction"
                        ),
                    )
                )
                continue
            if num > POLICY_FIXED_SL_MAX:
                issues.append(
                    ExitConfigIssue(
                        field=key,
                        reason="out_of_range",
                        value=_trunc(num),
                        min=cost.c_side,
                        max=POLICY_FIXED_SL_MAX,
                        message=f"fixed stop-loss must be <= {POLICY_FIXED_SL_MAX}",
                    )
                )
                continue
            if num < cost.c_rt:
                warnings.append(
                    ExitConfigWarning(
                        code="sl_below_round_trip_cost",
                        field=key,
                        message=(
                            f"stop-loss {num} is below the round-trip cost "
                            f"{cost.c_rt:.4f}; fees make up at least half of "
                            "every stopped-out loss"
                        ),
                    )
                )
            resolved[key] = num
        elif key == "bracket_take_profit_pct":
            if num <= cost.c_side:
                issues.append(
                    ExitConfigIssue(
                        field=key,
                        reason="take_profit_inside_exit_cost",
                        value=_trunc(num),
                        min=cost.c_side,
                        message=(
                            f"fixed take-profit {num} must exceed the exit "
                            f"cost ({cost.c_side:.4f}); it would realise a "
                            "net loss"
                        ),
                    )
                )
                continue
            if num > POLICY_FIXED_TP_MAX:
                issues.append(
                    ExitConfigIssue(
                        field=key,
                        reason="out_of_range",
                        value=_trunc(num),
                        min=cost.c_side,
                        max=POLICY_FIXED_TP_MAX,
                        message=f"fixed take-profit must be <= {POLICY_FIXED_TP_MAX}",
                    )
                )
                continue
            resolved[key] = num
        else:  # ATR multipliers
            lo, hi = ATR_MULTIPLIER_RANGE
            if not (lo <= num <= hi):
                issues.append(
                    ExitConfigIssue(
                        field=key,
                        reason="out_of_range",
                        value=_trunc(num),
                        min=lo,
                        max=hi,
                        message=f"{key} must be in [{lo}, {hi}]",
                    )
                )
                continue
            resolved[key] = num

    # -- W2: reward:risk <= 1 -----------------------------------------
    if mode == "fixed":
        sl = resolved.get("bracket_stop_loss_pct")
        tp = resolved.get("bracket_take_profit_pct")
    else:
        sl = resolved.get("bracket_atr_sl_multiplier")
        tp = resolved.get("bracket_atr_tp_multiplier")
    if sl is not None and tp is not None and tp <= sl:
        warnings.append(
            ExitConfigWarning(
                code="reward_risk_le_1",
                field=None,
                message="take-profit distance is not greater than the stop-loss distance",
            )
        )

    # -- W3: stop_exceeds_risk_budget (fixed SL only; E8 demoted) --------
    sl_for_budget = resolved.get("bracket_stop_loss_pct")
    if sl_for_budget is not None:
        _warn_risk_budget(
            warnings, field="bracket_stop_loss_pct", distance=sl_for_budget, cost=cost
        )

    has_active_bracket = any(k in resolved for k in active_keys)
    normalized_bracket: dict[str, object] = {}
    if has_active_bracket:
        normalized_bracket["bracket_mode"] = mode
        normalized_bracket["bracket_atr_period"] = period if period is not None else 14
        normalized_bracket.update(resolved)

    # -- trailing_stop_pct (independent of bracket mode) ------------------
    trailing_num, trailing_issue = _coerce_numeric("trailing_stop_pct", trailing_stop_pct)
    trailing_resolved: float | None = None
    if trailing_issue is not None:
        issues.append(trailing_issue)
    elif trailing_num is not None and trailing_num != 0.0:
        lo, hi = TRAILING_PCT_RANGE
        if not (lo <= trailing_num <= hi):
            issues.append(
                ExitConfigIssue(
                    field="trailing_stop_pct",
                    reason="out_of_range",
                    value=_trunc(trailing_num),
                    min=lo,
                    max=hi,
                    message=f"trailing_stop_pct must be in [{lo}, {hi}]",
                )
            )
        else:
            if trailing_num < cost.c_rt:
                warnings.append(
                    ExitConfigWarning(
                        code="trailing_below_round_trip_cost",
                        field="trailing_stop_pct",
                        message=(
                            f"trailing stop {trailing_num} is below the "
                            f"round-trip cost {cost.c_rt:.4f}"
                        ),
                    )
                )
            _warn_risk_budget(
                warnings, field="trailing_stop_pct", distance=trailing_num, cost=cost
            )
            trailing_resolved = trailing_num

    config = ExitConfig(bracket=normalized_bracket, trailing_stop_pct=trailing_resolved)
    return config, issues, warnings


def _warn_risk_budget(
    warnings: list[ExitConfigWarning], *, field: str, distance: float, cost: ExitCostModel
) -> None:
    budget = cost.pos_cap * (distance + cost.c_side)
    if budget >= 0.8 * cost.r:
        exceeds = budget >= cost.r
        verb = "exceeds" if exceeds else "approaches"
        warnings.append(
            ExitConfigWarning(
                code="stop_exceeds_risk_budget",
                field=field,
                message=(
                    f"a stop at this distance implies a worst-case loss of "
                    f"{budget:.2%} of equity at the position cap, which {verb} "
                    f"the {cost.r:.0%} per-trade risk budget"
                ),
            )
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_exit_config(
    *,
    bracket: Mapping[str, object] | None,
    trailing_stop_pct: object,
    cost: ExitCostModel | None = None,
) -> ExitConfig:
    """Strictly parse and normalise a bracket + trailing config.

    Raises ``ExitConfigError(code="invalid_exit_config")`` with every
    collected issue on any failure (SY-13a-01).
    """
    config, issues, _warnings = _parse_core(bracket, trailing_stop_pct, cost or _DEFAULT_COST_MODEL)
    if issues:
        raise ExitConfigError("invalid_exit_config", issues)
    return config


def salvage_exit_config(
    *,
    bracket: Mapping[str, object] | None,
    trailing_stop_pct: object,
    cost: ExitCostModel | None = None,
) -> tuple[ExitConfig, tuple[ExitConfigIssue, ...]]:
    """Validate the bracket and the trailing stop **independently**, keeping
    only the valid component(s) (SY-13a-16, "salvage with explicit waiver").

    Never raises.  The returned issue tuple lists every dropped component so
    the caller can surface it (e.g. ``exitConfigWaived``).
    """
    cost = cost or _DEFAULT_COST_MODEL
    bracket_cfg, bracket_issues, _ = _parse_core(bracket, None, cost)
    trailing_cfg, trailing_issues, _ = _parse_core(None, trailing_stop_pct, cost)
    config = ExitConfig(
        bracket=bracket_cfg.bracket if not bracket_issues else {},
        trailing_stop_pct=trailing_cfg.trailing_stop_pct if not trailing_issues else None,
    )
    return config, tuple(bracket_issues) + tuple(trailing_issues)


def strategy_requires_exit_manager(obj: object) -> bool:
    """``getattr(obj, "requires_exit_manager", False) is True`` -- the
    ``is True`` guard means a MagicMock (whose attributes are truthy but not
    literally ``True``) is never treated as requiring an exit manager
    (SY-13a-06)."""
    return getattr(obj, "requires_exit_manager", False) is True


def require_exit_manager(strategies: Iterable[object], cfg: ExitConfig) -> None:
    """Raise ``ExitConfigError(code="exit_manager_required")`` iff any
    strategy in ``strategies`` requires a downside exit and ``cfg`` has
    none (SY-13a-05/06/07)."""
    if cfg.has_downside_exit:
        return
    requiring = [s for s in strategies if strategy_requires_exit_manager(s)]
    if not requiring:
        return
    raise ExitConfigError(
        "exit_manager_required",
        (),
        strategy=_strategy_name(requiring[0]),
        requires_one_of=_REQUIRES_ONE_OF,
    )


def resolve_allow_pyramiding(explicit: bool | None, strategies: Iterable[object]) -> bool:
    """An explicit value always wins.  Otherwise: True only if *every*
    strategy's ``default_allow_pyramiding`` is True (SY-13a-08)."""
    if explicit is not None:
        return bool(explicit)
    classes = [_strategy_class(s) for s in strategies]
    if not classes:
        return False
    return all(bool(getattr(c, "default_allow_pyramiding", False)) for c in classes)


def validate_run_exit_config(
    strategy_cls: type | object,
    *,
    bracket: Mapping[str, object] | None,
    trailing_stop_pct: object,
    mode: RunMode | str,
    allow_pyramiding: bool | None,
    timeframe: str | None = None,
    cost: ExitCostModel | None = None,
) -> ExitConfigVerdict:
    """The one-stop call used by every enforcement point (SY-13a-15/§1c).

    Order: parse (raises ``invalid_exit_config``) -> require_exit_manager
    (raises ``exit_manager_required``) -> resolve ``allow_pyramiding`` ->
    live-pyramiding ban (raises ``live_pyramiding_forbidden``) -> warnings.
    """
    # WP13a-S-03 (security round 2): normalise/validate mode FIRST, before
    # any other check.  ``RunMode`` is a plain ``StrEnum`` with no custom
    # ``_missing_`` -- this is an exact, case-sensitive match, so "LIVE",
    # "Live" and " live" all raise here instead of silently falling through
    # to the ``mode_value == RunMode.LIVE.value`` string comparison below
    # (which would have been fail-OPEN on an unrecognised mode, skipping
    # the E11 live-pyramiding ban entirely).
    try:
        mode = mode if isinstance(mode, RunMode) else RunMode(mode)
    except ValueError as exc:
        raise ExitConfigError(
            "invalid_exit_config",
            (
                ExitConfigIssue(
                    field="mode",
                    reason="invalid_choice",
                    value=_trunc(mode),
                    message=f"mode must be one of {[m.value for m in RunMode]}",
                ),
            ),
        ) from exc

    cost = cost or _DEFAULT_COST_MODEL
    config, issues, warnings = _parse_core(bracket, trailing_stop_pct, cost)
    if issues:
        raise ExitConfigError("invalid_exit_config", issues, warnings=tuple(warnings))

    require_exit_manager([strategy_cls], config)

    resolved_pyramiding = resolve_allow_pyramiding(allow_pyramiding, [strategy_cls])
    mode_value = mode.value
    default_true = bool(getattr(_strategy_class(strategy_cls), "default_allow_pyramiding", False))

    if mode_value == RunMode.LIVE.value:
        if resolved_pyramiding:
            hint = None
            if default_true and allow_pyramiding is None:
                hint = (
                    "strategy accumulates by design; pass allowPyramiding=false "
                    "to run single-entry (unvalidated), or use paper"
                )
            raise ExitConfigError(
                "live_pyramiding_forbidden",
                (),
                strategy=_strategy_name(strategy_cls),
                hint=hint,
                warnings=tuple(warnings),
            )
        if default_true:
            warnings.append(
                ExitConfigWarning(
                    code="accumulation_disabled",
                    field="allow_pyramiding",
                    message=(
                        f"{_strategy_name(strategy_cls)} accumulates by design "
                        "but pyramiding is disabled for this run (single-entry, "
                        "unvalidated)"
                    ),
                )
            )
        if not strategy_requires_exit_manager(strategy_cls) and not config.has_downside_exit:
            warnings.append(
                ExitConfigWarning(
                    code="no_downside_exit",
                    field=None,
                    message=(
                        f"{_strategy_name(strategy_cls)} has no bracket or "
                        "trailing stop and relies solely on its own SELL "
                        "signal for downside protection in LIVE"
                    ),
                )
            )
        if timeframe is not None and str(timeframe) in _LONG_TIMEFRAMES:
            warnings.append(
                ExitConfigWarning(
                    code="bar_close_exit_checks_only",
                    field=None,
                    message=(
                        f"timeframe {timeframe} is checked at bar close only; "
                        "intra-bar exit crossings are not detected until 1.3b"
                    ),
                )
            )
    elif default_true and allow_pyramiding is False:
        # Paper/backtest: also surface W7 when the operator explicitly
        # disabled accumulation for a strategy that defaults to it on.
        warnings.append(
            ExitConfigWarning(
                code="accumulation_disabled",
                field="allow_pyramiding",
                message=(
                    f"{_strategy_name(strategy_cls)} accumulates by design "
                    "but pyramiding was explicitly disabled for this run"
                ),
            )
        )

    return ExitConfigVerdict(
        config=config, allow_pyramiding=resolved_pyramiding, warnings=tuple(warnings)
    )


def _bounded_params_for_response(params: Mapping[str, object]) -> dict[str, object]:
    """WP13a-S-R4-03/S-R5-02 (security rounds 4-5): bound the
    per-combination ``params`` dict before it is echoed into the
    optimizer's 422 response body (up to 20 of these are embedded per
    response, per WP13a-C-01).

    - ``None`` / ``int`` / ``float`` / ``bool`` pass through UNCHANGED --
      preserving the existing response contract and tests exactly (e.g.
      ``params["bracket_stop_loss_pct"] == 0.51`` stays a float, never a
      string).
    - ``str`` passes through UNCHANGED when at most 64 characters (the
      overwhelming common case -- a normal ``bracket_mode`` string etc.);
      a LONGER ``str`` is routed through ``_trunc`` instead (WP13a-S-R5-02:
      previously a single ~7MB string grid value was echoed verbatim into
      each of up to 20 issues, producing a ~140MB response). It stays a
      ``str`` either way -- ``_trunc``'s ``str`` branch is a direct slice,
      never adding quotes/repr formatting.
    - Anything else (list, dict, nested combinations, ...) is replaced by
      its bounded ``_trunc`` preview string, so a single huge grid value
      can never make the response body itself an amplification vector,
      independent of how fast the validator that discovered it now is.
    """
    bounded: dict[str, object] = {}
    for k, v in params.items():
        if v is None or isinstance(v, (int, float, bool)):
            bounded[k] = v
        elif isinstance(v, str):
            bounded[k] = v if len(v) <= 64 else _trunc(v)
        else:
            bounded[k] = _trunc(v)
    return bounded


def validate_param_grid_exit_config(
    strategy_cls: type, param_grid: Mapping[str, Iterable[object]]
) -> None:
    """Pre-validate every combination in an optimizer parameter grid
    (SY-13a-17).  Raises the first ``ExitConfigError`` hit, with every
    failing combination reported as a combination-scoped
    :class:`OptimizeComboIssue` on ``.combo_issues`` (capped at 20;
    ``.total_invalid`` on the exception carries the real count,
    ``requires_one_of``/``strategy`` from the first exit-manager failure
    encountered) -- WP13a-C-01: ``field`` is always the PLAIN field name,
    never a composite ``combo[i].field`` string; ``combo_index`` and
    ``params`` are first-class, matching
    ``apps/ui/src/lib/types.ts::OptimizeExitConfigIssue`` exactly.

    ``param_grid`` maps a parameter name to its list of grid values, in the
    ``itertools.product``-style shape ``ParameterOptimizer`` builds
    combinations from.

    WP13a-S-R2-01 (security round 2): the caller (``apps.api.routers.
    optimize``) already rejects any grid whose FULL cross-product exceeds
    ``max_combinations`` (<=1000) BEFORE this function is ever called, so
    the product iterated here is always bounded.  Within that bound, this
    function still iterates the full grid's ``itertools.product`` LAZILY
    (never materialised as a list) so memory stays O(1) per step rather
    than O(grid size), and ``combo_index`` is the position within that
    real, full-grid product -- i.e. exactly what the UI's grid view shows
    -- rather than an index into a reduced "exit keys only" sub-grid (which
    would silently disagree with the UI whenever the grid also varies any
    non-exit-relevant param).

    Only ``bracket_*`` / ``trailing_stop_pct`` / ``allow_pyramiding`` keys
    affect the exit-config verdict; every other grid dimension is
    exit-irrelevant "noise" that the optimizer's own combination loop
    validates separately via the strategy's ``_validate_params``.  Many
    full-grid positions can therefore share the exact same *exit-relevant*
    sub-tuple of values (e.g. a grid that varies ``lookback`` 1..100 but
    holds ``bracket_stop_loss_pct`` fixed) -- ``validate_run_exit_config``
    is deterministic in those values alone, so this function caches its
    result per distinct exit-relevant sub-tuple and reuses it for every
    full-grid position that shares that sub-tuple, instead of re-running
    the same validation work over and over.  ``params`` on each reported
    issue still only ever carries the exit-relevant subset (never the
    exit-irrelevant dimensions), unchanged from before.
    """
    import itertools

    all_keys = list(param_grid)
    all_value_lists = [list(param_grid[k]) for k in all_keys]

    exit_keys = [
        k
        for k in all_keys
        if k.startswith("bracket_") or k in ("trailing_stop_pct", "allow_pyramiding")
    ]

    combo_issues: list[OptimizeComboIssue] = []
    first_error: ExitConfigError | None = None
    total_invalid = 0
    # Cache keyed by the exit-relevant sub-tuple only, so full-grid
    # positions that differ only in exit-irrelevant dimensions reuse the
    # same validation result instead of re-validating (WP13a-S-R2-01).
    cache: dict[tuple[object, ...], ExitConfigError | None] = {}

    for idx, values in enumerate(itertools.product(*all_value_lists)):
        full_combo = dict(zip(all_keys, values, strict=True))

        # WP13a-S-R3-02/S-R4-03 (security rounds 3-4): a list/dict grid
        # value on an exit key (e.g. ``bracket_stop_loss_pct: [[0.05]]``)
        # is unhashable, so it cannot be a raw cache-key element -- build
        # the key element-by-element, substituting ``("__id__", id(v))``
        # for any unhashable element instead of dropping the cache
        # entirely.  ``itertools.product`` reuses the SAME element objects
        # from ``all_value_lists`` on every position (it never copies
        # them), and every one of those objects stays referenced by
        # ``param_grid``/``all_value_lists`` for the whole duration of
        # this call, so ``id(v)`` cannot be recycled mid-call -- one
        # malformed (unhashable) value is therefore still validated only
        # ONCE, no matter how many full-grid positions it recurs across
        # via an unrelated non-exit dimension (e.g. ``lookback``). This
        # closes the amplification (a single large unhashable value,
        # revalidated at every one of up to 1000 uncacheable positions)
        # that made the round-3 fallback re-run ``_trunc`` -- and thus its
        # then-unbounded ``str()`` -- up to 1000 times over.
        cache_key_parts: list[object] = []
        for k in exit_keys:
            v = full_combo.get(k)
            try:
                hash(v)
            except TypeError:
                cache_key_parts.append(("__id__", id(v)))
            else:
                cache_key_parts.append(v)
        cache_key = tuple(cache_key_parts)

        if cache_key in cache:
            exc = cache[cache_key]
        else:
            exit_only_combo = {k: full_combo[k] for k in exit_keys}
            bracket = {k: v for k, v in exit_only_combo.items() if k.startswith("bracket_")}
            trailing = exit_only_combo.get("trailing_stop_pct")
            allow_pyramiding = exit_only_combo.get("allow_pyramiding")
            try:
                validate_run_exit_config(
                    strategy_cls,
                    bracket=bracket,
                    trailing_stop_pct=trailing,
                    mode=RunMode.BACKTEST,
                    allow_pyramiding=(
                        allow_pyramiding if isinstance(allow_pyramiding, bool) else None
                    ),
                )
                exc = None
            except ExitConfigError as e:
                exc = e
            cache[cache_key] = exc

        if exc is not None:
            total_invalid += 1
            if first_error is None:
                first_error = exc
            if len(combo_issues) < 20:
                # WP13a-S-R4-03 (security round 4): bound the ECHOED
                # params dict too, not just the field-scoped `_trunc`ed
                # ``value`` -- up to 20 of these are embedded verbatim in
                # the response body, and echoing a huge raw list/dict 20
                # times would re-open the same amplification (an
                # expensive-to-encode multi-megabyte response) even
                # though validation itself is now fast.
                exit_only_params = _bounded_params_for_response(
                    {k: full_combo[k] for k in exit_keys}
                )
                if exc.issues:
                    for issue in exc.issues:
                        combo_issues.append(
                            OptimizeComboIssue(
                                combo_index=idx,
                                params=exit_only_params,
                                field=issue.field,
                                reason=issue.reason,
                                value=issue.value,
                                min=issue.min,
                                max=issue.max,
                                message=issue.message,
                            )
                        )
                else:
                    # exit_manager_required / live_pyramiding_forbidden
                    # carry no per-field issues -- still one combo-scoped
                    # row so the UI has something to render per failure.
                    combo_issues.append(
                        OptimizeComboIssue(
                            combo_index=idx,
                            params=exit_only_params,
                            field=None,
                            reason=exc.code,
                            value=None,
                            min=None,
                            max=None,
                            message=str(exc),
                        )
                    )

    if first_error is not None:
        err = ExitConfigError(
            first_error.code,
            (),
            strategy=first_error.strategy,
            requires_one_of=first_error.requires_one_of,
            hint=first_error.hint,
        )
        # Consulted by apps.api.routers.optimize's
        # ``_optimize_exit_config_error_detail`` for the 422 envelope
        # (SY-13a-17, WP13a-C-01: combo_index/params, capped at 20, plus
        # the real total_invalid count).
        err.combo_issues = tuple(combo_issues)  # type: ignore[attr-defined]
        err.total_invalid = total_invalid  # type: ignore[attr-defined]
        raise err
