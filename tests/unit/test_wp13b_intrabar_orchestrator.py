"""
tests/unit/test_wp13b_intrabar_orchestrator.py
------------------------------------------------
WP1.3b-b WP13B-T-65 (API / orchestrator wiring): ``run_live_engine`` resolves the live
intra-bar exit-check interval with the pure resolver and hands the result to the
``StrategyEngine`` config; it logs ``runs.intrabar_exit_checks`` with the interval or the
reason.  Same patch stack as ``tests/unit/test_live_engine_wiring.py``.

    smoke runs          -> off (forced), also on a protective resume
    timeframe <= I      -> off (``interval_ge_timeframe``)
    interval 0          -> off (``disabled``)
    no SL / trailing    -> off (``no_protective_exit``)
    waived exit config  -> off (a protective resume that salvaged nothing has no manager)
    otherwise           -> ``exit_check_interval_s`` in the engine config
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from api.routers.runs import _RUN_TASKS, _run_live_engine
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.unit.test_live_engine_wiring import (
    _PT_CCXT,
    _PT_FLUSH,
    _PT_FLUSH_LOOP,
    _PT_LIVE_ENGINE,
    _PT_MDS,
    _PT_PORTFOLIO,
    _PT_RISK_MGR,
    _PT_SESSION_FACTORY,
    _PT_SETTINGS,
    _PT_STRAT_ENGINE,
    _make_ccxt_module_mock,
    _make_db_session_factory,
    _make_settings_mock,
    _make_strategy_cls_mock,
    _make_strategy_engine_mock,
    _risk_mgr_cls,
)
from trading.costs import DEFAULT_COST_PROFILE, CostProfile, get_cost_profile

_RUN_ID = "bbbb0000-0000-0000-0000-0000000013bb"
_SL = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.05}
_TP_ONLY = {"bracket_mode": "fixed", "bracket_take_profit_pct": 0.10}


async def _run(
    *,
    interval: object = 300,
    timeframe: TimeFrame = TimeFrame.ONE_HOUR,
    strategy_name: str = "ma_crossover",
    bracket_config: dict[str, object] | None = _SL,
    trailing_stop_pct: float | None = None,
    protective_mode: bool = False,
    exit_config_waived: bool = False,
    cost_profile: CostProfile = DEFAULT_COST_PROFILE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``run_live_engine`` over mocks; return (engine config, intrabar log records)."""
    settings_mock = _make_settings_mock()
    settings_mock.live_exit_check_interval_s = interval
    get_sf_mock, _ = _make_db_session_factory()
    ccxt_module_mock, _ = _make_ccxt_module_mock()
    se_cls_mock, _ = _make_strategy_engine_mock()
    live_engine_cls = MagicMock()
    live_engine_cls.return_value = MagicMock()
    portfolio_cls = MagicMock()
    portfolio_cls.return_value = MagicMock()
    mds_cls = MagicMock()
    mds_cls.return_value = MagicMock()
    _RUN_TASKS.pop(_RUN_ID, None)
    with (
        capture_logs() as logs,
        patch(_PT_SETTINGS, return_value=settings_mock),
        patch(_PT_SESSION_FACTORY, get_sf_mock),
        patch(_PT_CCXT, ccxt_module_mock),
        patch(_PT_MDS, mds_cls),
        patch(_PT_LIVE_ENGINE, live_engine_cls),
        patch(_PT_PORTFOLIO, portfolio_cls),
        patch(_PT_RISK_MGR, _risk_mgr_cls()),
        patch(_PT_STRAT_ENGINE, se_cls_mock),
        patch(_PT_FLUSH, new_callable=AsyncMock),
        patch(_PT_FLUSH_LOOP, new_callable=AsyncMock),
    ):
        await _run_live_engine(
            run_id_str=_RUN_ID,
            strategy_cls=_make_strategy_cls_mock(),
            strategy_name=strategy_name,
            strategy_params={},
            symbols=["BTC/USDT"],
            timeframe=timeframe,
            initial_capital="10000",
            cost_profile=cost_profile,
            bracket_config=bracket_config,
            trailing_stop_pct=trailing_stop_pct,
            protective_mode=protective_mode,
            exit_config_waived=exit_config_waived,
        )
    se_cls_mock.assert_called_once()
    config = dict(se_cls_mock.call_args.kwargs["config"])
    return config, [e for e in logs if e["event"] == "runs.intrabar_exit_checks"]


async def test_t65_a_stop_loss_run_gets_the_default_interval_and_says_so() -> None:
    config, logs = await _run()
    assert config["exit_check_interval_s"] == 300
    assert len(logs) == 1 and logs[0]["interval_s"] == 300 and "reason" not in logs[0]


async def test_t65_a_trailing_stop_alone_is_enough() -> None:
    config, _logs = await _run(bracket_config=None, trailing_stop_pct=0.05)
    assert config["exit_check_interval_s"] == 300


async def test_t65_the_configured_interval_is_passed_through() -> None:
    config, logs = await _run(interval=120)
    assert config["exit_check_interval_s"] == 120 and logs[0]["interval_s"] == 120


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"interval": 0}, "disabled"),
        ({"timeframe": TimeFrame.FIVE_MINUTES}, "interval_ge_timeframe"),
        ({"timeframe": TimeFrame.ONE_MINUTE}, "interval_ge_timeframe"),
        ({"bracket_config": None}, "no_protective_exit"),
        ({"bracket_config": _TP_ONLY}, "no_protective_exit"),
        ({"strategy_name": "smoke_roundtrip", "protective_mode": True}, "smoke_run"),
        (
            {"protective_mode": True, "exit_config_waived": True, "bracket_config": None},
            "no_protective_exit",
        ),
    ],
)
async def test_t65_the_check_is_off_with_the_reason_logged(
    kwargs: dict[str, Any], reason: str
) -> None:
    config, logs = await _run(**kwargs)
    assert "exit_check_interval_s" not in config
    assert len(logs) == 1 and logs[0]["interval_s"] is None and logs[0]["reason"] == reason


async def test_t65_a_protective_resume_with_a_valid_exit_config_keeps_the_check() -> None:
    """Exits keep running in protective mode (entries are what it blocks), so does the check."""
    config, _logs = await _run(protective_mode=True)
    assert config["exit_check_interval_s"] == 300


async def test_t65_a_bad_setting_fails_the_run_instead_of_silently_running_unchecked() -> None:
    """Settings validates at startup, so this is defence in depth: the run ends in error."""
    with pytest.raises(AssertionError, match="called once"):
        await _run(interval=59)  # the run errored out: StrategyEngine was never constructed


# ---------------------------------------------------------------------------
# WP21A-CR-01 / SEC-01: the resolver parses at the RUN's own cost basis
# ---------------------------------------------------------------------------

_LEGACY_BAND = [
    {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.0068, "bracket_take_profit_pct": 0.05},
    {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.03, "bracket_take_profit_pct": 0.01},
]


@pytest.mark.parametrize("bracket", _LEGACY_BAND)
async def test_wp21a_cr01_legacy_run_in_the_band_keeps_the_intrabar_check(
    bracket: dict[str, object],
) -> None:
    """A legacy run (floor 0.0065) with a fixed SL/TP in (0.0065, 0.0130] stays protected."""
    config, logs = await _run(
        bracket_config=bracket, cost_profile=get_cost_profile("legacy_runtime_v0")
    )
    assert config["exit_check_interval_s"] == 300
    assert logs[0]["interval_s"] == 300 and "reason" not in logs[0]


async def test_wp21a_cr01_the_basis_is_the_runs_profile_not_the_default() -> None:
    """The same bracket at the default profile is below its floor: the basis is honoured."""
    config, logs = await _run(bracket_config=_LEGACY_BAND[0], cost_profile=DEFAULT_COST_PROFILE)
    assert "exit_check_interval_s" not in config
    assert logs[0]["reason"] == "no_protective_exit"
