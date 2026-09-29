"""
packages/trading/smoke_guard.py
---------------------------------
Pure guardrail validator for WP-SMOKE (VP2 herstart-protocol step 2, D2).

Binding spec: ``reports/vp2-smoke/synthesis-spec.md`` section 7 (G-1..G-8).

This module performs **zero I/O** -- no DB, no exchange, no logging side
effects beyond what the caller does with the raised exception. It exists
so the create-run path, the resume path, the promote path and the
orchestrator's defence-in-depth call (G-13) can all share one
implementation of "is this a smoke_roundtrip config the operator is
allowed to run" without re-deriving the bounds independently.

Scope note (producer split, synthesis spec section 13)
--------------------------------------------------------
This module implements G-1..G-8 (the config-shape guardrails). G-9
(live exclusivity), G-10 (resume mode), G-11 (promotion ban), G-12 (paper
boot recovery) and G-13's call *sites* are wired by Producer B into
``apps/api/routers/runs.py`` / ``apps/api/services/run_orchestrator.py``
-- they key off :data:`SMOKE_STRATEGY_NAMES` directly (by name, per
synthesis spec C-4) and do not need a dedicated guard function here.

Deliberate scoping decision (G-7, ATR fields) -- corrected (WP-SMOKE fix
F-7, SMK-CR-01)
-----------------------------------------------
``bracket_atr_period`` is intentionally NOT checked for "must be absent".
The reason is NOT that ``_extract_bracket_config`` (``apps/api/routers/
runs.py``) injects a default -- it does not. It copies
``bracket_atr_period`` into the returned bracket dict only when the
caller literally supplied the key in ``strategyParams``; it injects no
default of its own.

The default ``bracket_atr_period=14`` (and a default ``bracket_mode``) is
injected later, by ``trading.exit_config.validate_run_exit_config``
(``exit_config.py:560``), which runs AFTER this guard's create-time call.
Its normalized output then REPLACES the create-time bracket dict in
``create_run``, is persisted verbatim in ``config_snapshot
["bracket_config"]``, and is exactly what G-13 (``run_paper_engine`` /
``run_live_engine``, and protective resume) passes back into
``validate_smoke_run`` on every subsequent guard call for the run's
lifetime.

An absence check on ``bracket_atr_period`` would therefore reject EVERY
valid smoke run at G-13, because by the time G-13 runs,
``bracket_atr_period`` is always present (normalized to 14). A stray,
caller-supplied ``bracket_atr_period`` on an otherwise-valid
``bracket_mode="fixed"`` config is inert in any case -- the ATR period is
never read for fixed-mode brackets -- and ``validate_run_exit_config``
still bounds it independently. This guard instead checks
``bracket_atr_sl_multiplier`` / ``bracket_atr_tp_multiplier`` (genuinely
blank unless the operator sets them) for "must be absent or blank".
See SMK-T-43.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from trading.exit_config import _trunc  # WP13a-S-R4-03: bounded-cost value preview, reused as-is
from trading.risk import RiskParameters

__all__ = [
    "SMOKE_STRATEGY_NAMES",
    "SmokeGuardError",
    "SmokeGuardIssue",
    "validate_smoke_run",
]

#: C-4 of the synthesis spec: keyed by strategy NAME, never by availability
#: status -- a status change must not silently disable this guard. A
#: consistency test (SMK-T-07) asserts this set equals
#: ``{names with StrategyStatus.DIAGNOSTIC}``.
SMOKE_STRATEGY_NAMES: frozenset[str] = frozenset({"smoke_roundtrip"})

# G-2: allowed timeframes (D-SMK-2 recommends 5m; 1m stays allowed).
_ALLOWED_TIMEFRAMES: frozenset[str] = frozenset({"1m", "5m"})

# G-4: live capital band (D-SMK-1).
_LIVE_CAPITAL_MIN = Decimal("60.00")
_LIVE_CAPITAL_MAX = Decimal("66.00")

# G-6: strategy_params bounds (mirrors _SmokeRoundtripParams; duplicated
# deliberately -- this module must stay import-independent of the concrete
# strategy class so the guard can run at the API layer before the strategy
# is ever constructed).
_NOTIONAL_MIN = Decimal("5.00")
_NOTIONAL_MAX = Decimal("9.50")
_HOLD_BARS_MIN = 1
_HOLD_BARS_MAX = 12
_EXIT_RETRY_BARS_MIN = 1
_EXIT_RETRY_BARS_MAX = 5
_ALLOWED_STRATEGY_PARAM_KEYS: frozenset[str] = frozenset(
    {"notional_quote", "hold_bars", "exit_retry_bars"}
)

# WP-SMOKE fix F-3 (SMK-SEC-05): bound the unknown-strategy-param
# reflection so a caller cannot grow the 422 body (or the "reasons" log
# list) without limit by sending many unknown keys, and so no single
# key's raw text -- however long -- reaches the response untruncated.
_MAX_UNKNOWN_PARAM_ISSUES = 10
_MAX_FIELD_LEN = 64

# G-7: mandatory fixed stop-loss bounds (C-8 of the synthesis spec).
_SL_MIN = Decimal("0.03")
_SL_MAX = Decimal("0.08")


@dataclass(frozen=True, slots=True)
class SmokeGuardIssue:
    """One field-scoped guardrail violation. Mirrors
    ``trading.exit_config.ExitConfigIssue`` field-for-field so the API
    layer's 422 envelope can be built identically to
    ``_exit_config_error_detail`` (synthesis spec section 7)."""

    field: str
    reason: str
    value: str | None = None
    min: float | None = None
    max: float | None = None
    message: str = ""


class SmokeGuardError(ValueError):
    """Raised by :func:`validate_smoke_run` when one or more G-1..G-8
    rules are violated. ``code`` is always ``"smoke_guardrail_violation"``
    -- every collected issue is distinguished by its own ``reason``, not
    by the exception's ``code``."""

    def __init__(
        self,
        *,
        code: Literal["smoke_guardrail_violation"] = "smoke_guardrail_violation",
        issues: Sequence[SmokeGuardIssue],
    ) -> None:
        self.code = code
        self.issues: tuple[SmokeGuardIssue, ...] = tuple(issues)
        reasons = [i.reason for i in self.issues]
        super().__init__(f"{code}: {reasons}")


def _to_decimal(value: object) -> Decimal | None:
    """Best-effort ``Decimal`` coercion. ``None`` on anything that does
    not cleanly represent a real number -- callers treat that as "cannot
    evaluate this bound", never as a crash (this module has no I/O and
    must never raise anything other than :class:`SmokeGuardError`).

    WP-SMOKE fix F-2 (SMK-SEC-03): a non-finite ``Decimal`` (NaN, sNaN or
    +/-Infinity) parses successfully but breaks every ordering comparison
    a caller does next (``Decimal('NaN') < x`` raises
    ``decimal.InvalidOperation``, an uncaught ``ArithmeticError`` that
    would surface as a 500). Treat it the same as an uncoercible value --
    ``None`` -- so every existing bound check below falls through to its
    normal "value is missing/invalid" branch and raises a clean
    ``SmokeGuardError`` (422) instead. See SMK-T-38."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not dec.is_finite():
        return None
    return dec


def _is_blank(value: object) -> bool:
    """SY-13a-02 style zero-normalisation: ``None`` / ``""`` / exact ``0``
    mean "unset" for every bracket field this guard treats as optional."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    dec = _to_decimal(value)
    return dec is not None and dec == Decimal("0")


def validate_smoke_run(
    *,
    strategy_name: str,
    mode: object,
    symbols: Sequence[str],
    timeframe: str,
    initial_capital: object,
    strategy_params: Mapping[str, Any],
    bracket: Mapping[str, Any] | None,
    trailing_stop_pct: object,
    allow_pyramiding: bool | None,
    enable_adaptive_learning: bool | None,
) -> None:
    """Validate a proposed run configuration for ``strategy_name``.

    No-op (returns immediately, never raises) when ``strategy_name`` is
    not in :data:`SMOKE_STRATEGY_NAMES` -- every other strategy is
    completely unaffected by this module.

    Collects **every** violated rule (G-1..G-8) before raising, so a
    caller can surface the full list in one 422 response rather than one
    rule at a time (synthesis spec section 7).

    Parameters
    ----------
    strategy_name:
        The run's strategy name, exactly as it appears in the registry
        (``"smoke_roundtrip"`` is the only name this guard currently acts
        on).
    mode:
        The run mode. Accepts a ``RunMode`` member or its string value
        (``"backtest"`` / ``"paper"`` / ``"live"``), compared
        case-insensitively by string value only, so this module never has
        to import ``common.types``. WP-SMOKE fix F-4 (SMK-SEC-06): any
        value that does not normalize to one of those three strings
        raises a dedicated ``mode_not_recognised`` issue (fail closed)
        rather than silently being treated as non-live.
    symbols:
        The run's symbol list (G-1: exactly one).
    timeframe:
        The run's timeframe string (G-2: ``"1m"`` or ``"5m"``).
    initial_capital:
        The run's starting capital (G-4 in live; G-5 in every mode).
    strategy_params:
        The raw ``strategyParams`` mapping, BEFORE
        ``_extract_bracket_config`` and BEFORE the strategy's own Pydantic
        model validates it (G-6 is the first layer).
    bracket:
        The bracket-exit config already extracted by the caller (the
        ``bracket_*``-keyed mapping ``_extract_bracket_config`` returns),
        or ``None`` if none was supplied at all (G-7).
    trailing_stop_pct:
        The run's top-level trailing-stop percentage, or a blank sentinel
        (``None``/``""``/``0``) (G-7: must be blank for smoke).
    allow_pyramiding:
        The run's resolved ``allowPyramiding`` flag (G-8).
    enable_adaptive_learning:
        The run's resolved ``enableAdaptiveLearning`` flag (G-8).

    Raises
    ------
    SmokeGuardError
        If any G-1..G-8 rule is violated. ``issues`` lists every
        violation found, never just the first.
    """
    if strategy_name not in SMOKE_STRATEGY_NAMES:
        return

    # WP-SMOKE fix F-4 (SMK-SEC-06): normalize a RunMode member (via its
    # .value) or a plain string identically, then fail CLOSED -- append a
    # guard issue -- on anything outside the three known mode strings,
    # rather than silently treating an unrecognised mode as non-live and
    # skipping G-3/G-4's capital-band enforcement. The other G-1..G-8
    # rules still run and collect regardless (see SMK-T-41).
    mode_str = str(getattr(mode, "value", mode)).strip().lower()
    is_live = mode_str == "live"
    issues: list[SmokeGuardIssue] = []
    if mode_str not in {"backtest", "paper", "live"}:
        issues.append(
            SmokeGuardIssue(
                field="mode",
                reason="mode_not_recognised",
                value=_trunc(mode),
                message="Unrecognised run mode for smoke_roundtrip.",
            )
        )

    # ------------------------------------------------------------------
    # G-1: exactly one symbol.
    # ------------------------------------------------------------------
    if len(symbols) != 1:
        issues.append(
            SmokeGuardIssue(
                field="symbols",
                reason="too_many_symbols",
                value=_trunc(list(symbols)),
                message="smoke_roundtrip runs exactly one symbol per run.",
            )
        )

    # ------------------------------------------------------------------
    # G-2: timeframe in {1m, 5m}.
    # ------------------------------------------------------------------
    if timeframe not in _ALLOWED_TIMEFRAMES:
        issues.append(
            SmokeGuardIssue(
                field="timeframe",
                reason="timeframe_not_allowed",
                value=_trunc(timeframe),
                message="smoke_roundtrip only allows the 1m or 5m timeframe.",
            )
        )

    # ------------------------------------------------------------------
    # G-3: symbol quote == EUR (live only).
    # ------------------------------------------------------------------
    if is_live:
        for symbol in symbols:
            quote = symbol.split("/", 1)[1].strip().upper() if "/" in symbol else ""
            if quote != "EUR":
                issues.append(
                    SmokeGuardIssue(
                        field="symbols",
                        reason="quote_not_allowed",
                        value=_trunc(symbol),
                        message="Live smoke runs are EUR-quote only.",
                    )
                )

    # ------------------------------------------------------------------
    # G-4: initial_capital in [60.00, 66.00] (live only).
    # ------------------------------------------------------------------
    capital_dec = _to_decimal(initial_capital)
    if is_live:
        _capital_in_band = (
            capital_dec is not None
            and _LIVE_CAPITAL_MIN <= capital_dec <= _LIVE_CAPITAL_MAX
        )
        if not _capital_in_band:
            issues.append(
                SmokeGuardIssue(
                    field="initial_capital",
                    reason="capital_out_of_smoke_range",
                    value=_trunc(initial_capital),
                    min=float(_LIVE_CAPITAL_MIN),
                    max=float(_LIVE_CAPITAL_MAX),
                    message="Live smoke initial_capital must be in [60.00, 66.00] EUR.",
                )
            )

    # ------------------------------------------------------------------
    # G-6: strategy_params bounds + unknown keys. Runs BEFORE the
    # strategy's own Pydantic model ever sees strategy_params (third
    # layer) and before _validate_params_against_schema (second layer).
    # ------------------------------------------------------------------
    _unknown_keys = [key for key in strategy_params if key not in _ALLOWED_STRATEGY_PARAM_KEYS]
    for key in _unknown_keys[:_MAX_UNKNOWN_PARAM_ISSUES]:
        issues.append(
            SmokeGuardIssue(
                field=str(key)[:_MAX_FIELD_LEN],
                reason="unknown_param",
                value=_trunc(strategy_params[key]),
                message="Not a recognised smoke_roundtrip parameter.",
            )
        )
    if len(_unknown_keys) > _MAX_UNKNOWN_PARAM_ISSUES:
        issues.append(
            SmokeGuardIssue(
                field="strategy_params",
                reason="unknown_param",
                value=str(len(_unknown_keys)),
                message="Additional unknown parameters omitted.",
            )
        )

    notional_raw = strategy_params.get("notional_quote")
    if notional_raw is not None:
        notional_dec = _to_decimal(notional_raw)
        if notional_dec is None or notional_dec < _NOTIONAL_MIN or notional_dec > _NOTIONAL_MAX:
            issues.append(
                SmokeGuardIssue(
                    field="notional_quote",
                    reason="param_out_of_range",
                    value=_trunc(notional_raw),
                    min=float(_NOTIONAL_MIN),
                    max=float(_NOTIONAL_MAX),
                    message="notional_quote must be in [5.00, 9.50].",
                )
            )

    hold_bars_raw = strategy_params.get("hold_bars")
    if hold_bars_raw is not None:
        hold_bars_dec = _to_decimal(hold_bars_raw)
        if (
            hold_bars_dec is None
            or hold_bars_dec != hold_bars_dec.to_integral_value()
            or hold_bars_dec < _HOLD_BARS_MIN
            or hold_bars_dec > _HOLD_BARS_MAX
        ):
            issues.append(
                SmokeGuardIssue(
                    field="hold_bars",
                    reason="param_out_of_range",
                    value=_trunc(hold_bars_raw),
                    min=float(_HOLD_BARS_MIN),
                    max=float(_HOLD_BARS_MAX),
                    message="hold_bars must be an integer in [1, 12].",
                )
            )

    exit_retry_raw = strategy_params.get("exit_retry_bars")
    if exit_retry_raw is not None:
        exit_retry_dec = _to_decimal(exit_retry_raw)
        if (
            exit_retry_dec is None
            or exit_retry_dec != exit_retry_dec.to_integral_value()
            or exit_retry_dec < _EXIT_RETRY_BARS_MIN
            or exit_retry_dec > _EXIT_RETRY_BARS_MAX
        ):
            issues.append(
                SmokeGuardIssue(
                    field="exit_retry_bars",
                    reason="param_out_of_range",
                    value=_trunc(exit_retry_raw),
                    min=float(_EXIT_RETRY_BARS_MIN),
                    max=float(_EXIT_RETRY_BARS_MAX),
                    message="exit_retry_bars must be an integer in [1, 5].",
                )
            )

    # ------------------------------------------------------------------
    # G-5: notional_quote <= RiskParameters().max_position_size_pct *
    # initial_capital, in EVERY mode. The default is read at runtime
    # (never hard-coded) so a future change to RiskParameters' default
    # concentration cap is picked up automatically.
    # ------------------------------------------------------------------
    if capital_dec is not None:
        notional_for_ceiling = (
            _to_decimal(notional_raw) if notional_raw is not None else _NOTIONAL_MIN
        )
        # Fall back to the strategy's own default (9.00) when the caller
        # did not supply notional_quote at all -- the ceiling still must
        # hold against whatever the strategy will actually default to.
        if notional_raw is None:
            notional_for_ceiling = Decimal("9.00")
        max_position_size_pct = Decimal(str(RiskParameters().max_position_size_pct))
        ceiling = max_position_size_pct * capital_dec
        if notional_for_ceiling is not None and notional_for_ceiling > ceiling:
            issues.append(
                SmokeGuardIssue(
                    field="strategy_params.notional_quote",
                    reason="notional_exceeds_risk_ceiling",
                    value=_trunc(notional_raw if notional_raw is not None else "9.00"),
                    max=float(ceiling),
                    message=(
                        "notional_quote exceeds max_position_size_pct * "
                        "initial_capital for this run."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # G-7: mandatory fixed stop-loss; no TP, no ATR, no trailing.
    # ------------------------------------------------------------------
    bracket_map: Mapping[str, Any] = bracket or {}
    sl_raw = bracket_map.get("bracket_stop_loss_pct")
    if _is_blank(sl_raw):
        issues.append(
            SmokeGuardIssue(
                field="bracket_stop_loss_pct",
                reason="stop_loss_required",
                value=_trunc(sl_raw),
                message="smoke_roundtrip requires a fixed bracket_stop_loss_pct.",
            )
        )
    else:
        sl_dec = _to_decimal(sl_raw)
        if sl_dec is None or sl_dec < _SL_MIN or sl_dec > _SL_MAX:
            issues.append(
                SmokeGuardIssue(
                    field="bracket_stop_loss_pct",
                    reason="stop_loss_out_of_range",
                    value=_trunc(sl_raw),
                    min=float(_SL_MIN),
                    max=float(_SL_MAX),
                    message="bracket_stop_loss_pct must be in [0.03, 0.08].",
                )
            )

    mode_raw = bracket_map.get("bracket_mode")
    if not _is_blank(mode_raw) and str(mode_raw) != "fixed":
        issues.append(
            SmokeGuardIssue(
                field="bracket_mode",
                reason="bracket_mode_must_be_fixed",
                value=_trunc(mode_raw),
                message="smoke_roundtrip only allows bracket_mode='fixed' (or unset).",
            )
        )

    tp_raw = bracket_map.get("bracket_take_profit_pct")
    if not _is_blank(tp_raw):
        issues.append(
            SmokeGuardIssue(
                field="bracket_take_profit_pct",
                reason="take_profit_not_allowed",
                value=_trunc(tp_raw),
                message="smoke_roundtrip has no take-profit; only the fixed SL is allowed.",
            )
        )

    # Genuinely-blank-unless-set ATR fields. bracket_atr_period is
    # deliberately excluded from this "must be absent" check: it is
    # always normalized to 14 by validate_run_exit_config AFTER this
    # guard runs at create time, and that normalized value is what every
    # later G-13 call sees -- an absence check would false-reject every
    # valid run (see the module docstring, WP-SMOKE fix F-7 / SMK-CR-01,
    # and SMK-T-43).
    for atr_field in ("bracket_atr_sl_multiplier", "bracket_atr_tp_multiplier"):
        atr_raw = bracket_map.get(atr_field)
        if not _is_blank(atr_raw):
            issues.append(
                SmokeGuardIssue(
                    field=atr_field,
                    reason="take_profit_not_allowed",
                    value=_trunc(atr_raw),
                    message="smoke_roundtrip does not support ATR-mode brackets.",
                )
            )

    if not _is_blank(trailing_stop_pct):
        issues.append(
            SmokeGuardIssue(
                field="trailing_stop_pct",
                reason="trailing_not_allowed",
                value=_trunc(trailing_stop_pct),
                message="smoke_roundtrip has no trailing stop; only the fixed SL is allowed.",
            )
        )

    # ------------------------------------------------------------------
    # G-8: no pyramiding, no adaptive learning.
    # ------------------------------------------------------------------
    if allow_pyramiding is True:
        issues.append(
            SmokeGuardIssue(
                field="allow_pyramiding",
                reason="pyramiding_not_allowed",
                value="True",
                message="smoke_roundtrip never allows pyramiding.",
            )
        )
    if enable_adaptive_learning is True:
        issues.append(
            SmokeGuardIssue(
                field="enable_adaptive_learning",
                reason="adaptive_learning_not_allowed",
                value="True",
                message="smoke_roundtrip never allows adaptive learning.",
            )
        )

    if issues:
        raise SmokeGuardError(issues=issues)
