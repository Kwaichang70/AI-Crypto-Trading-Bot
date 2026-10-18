"""
tests/integration/test_wp21h_resume_basis.py
----------------------------------------------
WP2.1-H (CF-21-SEC-A, SEC-01): the three paper / live callers of ``resolve_run_cost_profile``
in ``apps/api/routers/runs.py`` read the run's STORED profile.  The accepted rows are exactly
the ones a runtime writer produces: a LIVE-EQUIVALENT registry profile, or ``legacy_runtime_v0``
as ``legacy_inferred``.  Any other stored row (a persisted ``override``, a ``legacy_inferred``
row that is not the runtime family's legacy profile, a registry row that is not
live-equivalent -- ``frictionless_diagnostic``, a stress row, ``legacy_backtest_v0``, a registry
``legacy_runtime_v0`` -- ...) must give the LEGACY RUNTIME basis, so the basis is never below
``legacy_runtime_v0``.  Legitimate rows resolve exactly as before.  (A tampered row may still
name ANOTHER live-equivalent registry id: an accepted residual, WP21H-SEC-04.)
Hermetic: no Postgres, no network, no exchange.

    live resume                  POST /runs/{id}/resume -> ``_run_live_engine(cost_profile=)``
    live boot recovery (check)   ``recover_orphaned_runs``  -> ``validate_run_exit_config(cost=)``
    paper boot recovery          ``recover_orphaned_runs``  -> ``_run_paper_engine(cost_profile=)``
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import api.routers.runs as runs_module
from api.config import get_settings
from trading.costs import COST_PROFILES, CostProfile, get_cost_profile
from trading.exit_config import ExitCostModel

_LEGACY_RUNTIME = dataclasses.replace(COST_PROFILES["legacy_runtime_v0"], source="legacy_inferred")
_D6 = get_cost_profile("coinbase_advanced_d6")
_CHEAP = _D6.with_overrides(maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1"))


def _tampered_rows() -> dict[str, dict[str, Any]]:
    wrong_family = dataclasses.replace(
        COST_PROFILES["legacy_backtest_v0"], source="legacy_inferred"
    )
    return {
        "override": _CHEAP.to_metadata(),
        "legacy_inferred_legacy_false": dataclasses.replace(
            _LEGACY_RUNTIME, legacy=False
        ).to_metadata(),
        "legacy_inferred_backtest_family": wrong_family.to_metadata(),
        "legacy_inferred_cheaper_numbers": dataclasses.replace(
            _LEGACY_RUNTIME, maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1")
        ).to_metadata(),
        "oversized_exchange": {**_D6.to_metadata(), "exchange": "<b>" + "x" * 28_000},
        "registry_numbers_edited": {**_D6.to_metadata(), "taker_fee_bps": "1"},
        # SEC-01: genuine registry rows that are not live-equivalent (valid for backtest /
        # optimize, never stored on a paper / live run).  ``frictionless_diagnostic`` is the
        # 0-cost floor; the others cover every non-live-equivalent registry id.
        **{
            f"registry_{pid}": COST_PROFILES[pid].to_metadata()
            for pid in sorted(COST_PROFILES)
            if not COST_PROFILES[pid].live_equivalent
        },
    }


_TAMPERED = _tampered_rows()
_BRACKET = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.05,
    "bracket_take_profit_pct": 0.1,
}
#: Inside the entry cost of EVERY live-equivalent profile and of ``legacy_runtime_v0``
#: (0.0065), but outside a 0 floor: it passes only if the basis was lowered (SEC-01).
_TIGHT_BRACKET: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.004,
    "bracket_take_profit_pct": 0.005,
}
#: Outside the legacy runtime floor (0.0065), inside the default's (0.013).
_LEGACY_ONLY_BRACKET: dict[str, object] = {
    "bracket_mode": "fixed",
    "bracket_stop_loss_pct": 0.0068,
    "bracket_take_profit_pct": 0.05,
}
_NOT_LIVE_EQUIVALENT = tuple(
    sorted(pid for pid, p in COST_PROFILES.items() if not p.live_equivalent)
)


def _clear_run_tasks() -> None:
    getattr(runs_module, "_RUN_TASKS").clear()  # noqa: B009 -- private registry of the router


def _paper_config(
    cost_profile: object | None, bracket: dict[str, object] | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strategy_name": "grid_trading",
        "symbols": ["BTC/USD"],
        "timeframe": "1h",
        "initial_capital": "10000",
        "strategy_params": {},
        "bracket_config": dict(bracket or _BRACKET),
        "mode": "paper",
    }
    if cost_profile is not None:
        config["cost_profile"] = cost_profile
    return config


def _live_resume_app(monkeypatch: pytest.MonkeyPatch, resume_mod: Any) -> Any:
    monkeypatch.setenv("REQUIRE_API_AUTH", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM_TOKEN", resume_mod.CONFIRM_TOKEN)
    monkeypatch.setenv("ADMIN_API_KEY", resume_mod.ADMIN_KEY)
    get_settings.cache_clear()
    from api.main import create_app

    return create_app()


def _live_resume_response(
    monkeypatch: pytest.MonkeyPatch,
    stored: object | None,
    bracket: dict[str, object] | None = None,
) -> tuple[Any, AsyncMock]:
    """``POST /runs/{id}/resume`` of an orphaned live run whose stored ``cost_profile`` is
    ``stored``; return ``(response, the _run_live_engine spy)``."""
    from tests.integration import test_wp18a_resume_endpoint as resume_mod

    app = _live_resume_app(monkeypatch, resume_mod)
    monkeypatch.setattr(runs_module, "_STRATEGY_REGISTRY", {"grid_trading": MagicMock()})
    run_row = resume_mod._make_run_row(status="orphaned")
    run_row.config["bracket_config"] = dict(bracket or _BRACKET)
    if stored is not None:
        run_row.config["cost_profile"] = stored
    session = resume_mod._DispatchSession(run_row=run_row, runs_update_rowcounts=[1, 1])
    client = resume_mod._client(app, session)

    async def _scan(db: Any, run: Any, exchange: Any, *, fence: Any = None) -> Any:
        from api.services.run_recovery import ImportReport

        return ImportReport()

    import api.services.run_recovery as run_recovery_module

    monkeypatch.setattr(run_recovery_module, "scan_and_import", _scan)
    spy = AsyncMock()
    try:
        with patch("api.routers.runs._run_live_engine", spy):
            resp = client.post(
                f"/api/v1/runs/{resume_mod.RUN_ID}/resume",
                headers={
                    "X-Live-Confirm-Token": resume_mod.CONFIRM_TOKEN,
                    "X-Admin-Key": resume_mod.ADMIN_KEY,
                },
            )
    finally:
        get_settings.cache_clear()
        _clear_run_tasks()
    return resp, spy


def _live_resume(
    monkeypatch: pytest.MonkeyPatch,
    stored: object | None,
    bracket: dict[str, object] | None = None,
) -> CostProfile:
    """Resume an orphaned live run whose stored ``cost_profile`` is ``stored``; return the
    profile the live engine was started with."""
    resp, spy = _live_resume_response(monkeypatch, stored, bracket)
    assert resp.status_code == 200, resp.text
    profile = spy.call_args.kwargs["cost_profile"]
    assert isinstance(profile, CostProfile)
    return profile


async def _paper_boot(
    stored: object | None, bracket: dict[str, object] | None = None
) -> tuple[int, Any, AsyncMock, list[str]]:
    """Boot recovery of one orphaned paper run; return ``(recovered, the run row, the
    _run_paper_engine spy, the reasons the run was marked 'error' with)``.

    The reasons are recorded at the ``_mark_run_error`` call boundary, not from a log capture:
    the module-level logger of ``runs.py`` may have been cached by an earlier app lifespan."""
    from tests.unit import test_sprint24_run_recovery as rec

    orphan = rec._make_orphan(
        run_mode="paper", status="running", config=_paper_config(stored, bracket)
    )
    factory = rec._build_factory(
        rec._make_session_for_select([orphan]),
        rec._make_session_for_write(orphan),
        rec._make_session_for_write(orphan),
    )
    spy = AsyncMock()
    marked = AsyncMock(wraps=getattr(runs_module, "_mark_run_error"))  # noqa: B009
    with (
        patch("api.db.session.get_session_factory", return_value=factory),
        rec._patched_registry("grid_trading"),
        patch(
            "api.routers.runs._load_resume_snapshot",
            AsyncMock(return_value=rec._EMPTY_SNAPSHOT),
        ),
        patch("api.routers.runs._run_paper_engine", spy),
        patch("api.routers.runs._mark_run_error", marked),
    ):
        recovered = await runs_module.recover_orphaned_runs()
    _clear_run_tasks()
    return recovered, orphan, spy, [c.kwargs["reason"] for c in marked.call_args_list]


async def _paper_recovery(stored: object | None) -> CostProfile:
    recovered, _orphan, spy, _reasons = await _paper_boot(stored)
    assert recovered == 1
    profile = spy.call_args.kwargs["cost_profile"]
    assert isinstance(profile, CostProfile)
    return profile


async def _live_boot_check_basis(stored: object | None) -> ExitCostModel:
    """The exit-cost basis the live boot-recovery config check validated against."""
    from tests.unit import test_sprint24_run_recovery as rec

    config = _paper_config(stored)
    config["mode"] = "live"
    orphan = rec._make_orphan(run_mode="live", status="running", config=config)
    factory = rec._build_factory(
        rec._make_session_for_select([orphan]), rec._make_session_for_write(orphan)
    )
    seen: list[Any] = []

    def _recording_validate(*_a: Any, **kw: Any) -> Any:
        seen.append(kw["cost"])
        return MagicMock()

    with (
        patch("api.db.session.get_session_factory", return_value=factory),
        patch("api.services.audit_log.record_audit_event", AsyncMock()),
        rec._patched_registry("grid_trading"),
        patch("api.routers.runs.validate_run_exit_config", _recording_validate),
    ):
        await runs_module.recover_orphaned_runs()
    _clear_run_tasks()
    assert len(seen) == 1, "the live boot check must validate exactly once"
    basis = seen[0]
    assert isinstance(basis, ExitCostModel)
    return basis


@pytest.mark.parametrize("label", sorted(_TAMPERED))
class TestTamperedRowsGetTheLegacyRuntimeBasis:
    def test_live_resume(self, monkeypatch: pytest.MonkeyPatch, label: str) -> None:
        assert _live_resume(monkeypatch, _TAMPERED[label]) == _LEGACY_RUNTIME

    async def test_paper_boot_recovery(self, label: str) -> None:
        assert await _paper_recovery(_TAMPERED[label]) == _LEGACY_RUNTIME

    async def test_live_boot_recovery_check(self, label: str) -> None:
        basis = await _live_boot_check_basis(_TAMPERED[label])
        assert basis == ExitCostModel.from_cost_profile(_LEGACY_RUNTIME)
        assert basis != ExitCostModel.from_cost_profile(_CHEAP)


class TestLegitimateRowsAreUnchanged:
    @pytest.mark.parametrize("pid", ["coinbase_advanced_1", "coinbase_advanced_d6"])
    def test_live_resume_of_a_registry_row_uses_it(
        self, monkeypatch: pytest.MonkeyPatch, pid: str
    ) -> None:
        stored = get_cost_profile(pid).to_metadata(fee_source="exchange_fills")
        assert _live_resume(monkeypatch, stored) == get_cost_profile(pid)

    def test_live_resume_without_the_key_is_legacy_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _live_resume(monkeypatch, None) == _LEGACY_RUNTIME

    def test_live_resume_of_the_auto_retry_legacy_row_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _live_resume(monkeypatch, _LEGACY_RUNTIME.to_metadata()) == _LEGACY_RUNTIME

    async def test_paper_boot_recovery_of_a_registry_row_uses_it(self) -> None:
        assert await _paper_recovery(_D6.to_metadata()) == _D6

    async def test_paper_boot_recovery_without_the_key_is_legacy_runtime(self) -> None:
        assert await _paper_recovery(None) == _LEGACY_RUNTIME

    async def test_paper_boot_recovery_of_the_auto_retry_legacy_row_is_accepted(self) -> None:
        assert await _paper_recovery(_LEGACY_RUNTIME.to_metadata()) == _LEGACY_RUNTIME

    async def test_live_boot_check_of_a_registry_row_uses_it(self) -> None:
        basis = await _live_boot_check_basis(_D6.to_metadata())
        assert basis == ExitCostModel.from_cost_profile(_D6)


def _reasons(resp: Any) -> set[str]:
    return {e["reason"] for e in resp.json()["detail"]["errors"]}


class TestSec01ALoweredBasisCannotBypassTheFeeBandChecks:
    """WP21H-SEC-01 (AC-5): SL 0.004 / TP 0.005 sit inside the entry / exit cost of the
    legacy runtime basis, so the resume must REJECT them.  Under a 0 floor
    (``frictionless_diagnostic``) they would pass -- the harm CF-21-SEC-A closes."""

    @pytest.mark.parametrize("pid", _NOT_LIVE_EQUIVALENT)
    def test_live_resume_of_a_tampered_registry_row_is_422(
        self, monkeypatch: pytest.MonkeyPatch, pid: str
    ) -> None:
        resp, spy = _live_resume_response(
            monkeypatch, COST_PROFILES[pid].to_metadata(), _TIGHT_BRACKET
        )
        assert resp.status_code == 422, resp.text
        assert {"stop_inside_entry_cost", "take_profit_inside_exit_cost"} <= _reasons(resp)
        spy.assert_not_called()

    def test_the_rejection_is_the_one_of_a_row_without_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tampered, _ = _live_resume_response(
            monkeypatch, COST_PROFILES["frictionless_diagnostic"].to_metadata(), _TIGHT_BRACKET
        )
        absent, _ = _live_resume_response(monkeypatch, None, _TIGHT_BRACKET)
        assert (tampered.status_code, absent.status_code) == (422, 422)
        assert tampered.json()["detail"] == absent.json()["detail"]

    @pytest.mark.parametrize("pid", ["coinbase_advanced_1", "coinbase_advanced_d6"])
    def test_a_legitimate_row_is_still_judged_at_its_own_basis(
        self, monkeypatch: pytest.MonkeyPatch, pid: str
    ) -> None:
        resp, _ = _live_resume_response(
            monkeypatch, COST_PROFILES[pid].to_metadata(fee_source="exchange_fills"), _TIGHT_BRACKET
        )
        assert resp.status_code == 422, resp.text

    def test_a_tampered_row_falls_back_to_the_legacy_basis_not_to_a_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity: a bracket that is valid at the legacy runtime basis (0.0065) still
        resumes under a tampered row, with the engine on ``legacy_runtime_v0``."""
        stored = COST_PROFILES["frictionless_diagnostic"].to_metadata()
        assert _live_resume(monkeypatch, stored, _LEGACY_ONLY_BRACKET) == _LEGACY_RUNTIME

    @pytest.mark.parametrize("pid", _NOT_LIVE_EQUIVALENT)
    async def test_paper_boot_of_a_tampered_registry_row_marks_the_run_error(
        self, pid: str
    ) -> None:
        recovered, orphan, spy, reasons = await _paper_boot(
            COST_PROFILES[pid].to_metadata(), _TIGHT_BRACKET
        )
        assert recovered == 0
        assert orphan.status == "error"
        spy.assert_not_called()
        assert reasons == ["invalid_exit_config"]  # the exit-config code, nothing else

    async def test_paper_boot_of_a_tampered_row_falls_back_to_the_legacy_basis(self) -> None:
        recovered, orphan, spy, _ = await _paper_boot(
            COST_PROFILES["frictionless_diagnostic"].to_metadata(), _LEGACY_ONLY_BRACKET
        )
        assert recovered == 1 and orphan.status == "running"
        assert spy.call_args.kwargs["cost_profile"] == _LEGACY_RUNTIME
