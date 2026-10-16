"""
tests/unit/test_wp21_costs.py
------------------------------
WP2.1a (reports/vp2-wp2.1/synthesis-spec.md) -- the cost-model core.

    WP21-T-01  CostProfile validation
    WP21-T-02  registry numbers + pinned fingerprints + default id
    WP21-T-03  to_metadata <-> from_metadata strictness
    WP21-T-04  resolve_run_cost_profile (legacy by absence, never raises)
    WP21-T-05  new_run_cost_profile (U-2 per-mode rules)
    WP21-T-06  ExitCostModel.from_cost_profile / policy_cost_model
    WP21-T-07  AST pins (leaf module, no env, no cost fields, wiring)
    WP21-T-08  paper arithmetic from the profile
    WP21-T-09  C25 (risk_params reach the backtest risk manager unchanged)
    WP21-T-10  engine-side parse at the run's OWN profile
    WP21-T-15  resolution parity across consumers
    WP21-T-17  optimizer: one profile object for every combination
    WP21-T-18  ML gate passes DEFAULT and records it
    WP21-T-19  research scripts: --cost-profile / numeric overrides / header
    WP21-T-23  auto-retry copies the crashed run's resolved profile
    WP21-T-24  orchestrator wiring
    WP21-T-25  run_orm_to_response cost fields
    WP21-T-26  leaderboard eligibility helper
    WP21-T-29  DS-1 precheck --cost-profile
    WP21-T-30  idempotency fingerprint unchanged for bodies without costProfile
    WP21-T-31  StrategyEngine mock-safety + env var ignored with a name-only warning
    WP21-T-32  BacktestResult.total_slippage_cost / cost_profile_*
    WP21-T-38  match_fee_tier
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from structlog.testing import capture_logs

from common.types import OrderSide, RunMode, TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.fakes.scripted_strategy import ScriptedSequenceStrategy
from trading import costs as costs_module
from trading.backtest import BacktestRunner
from trading.costs import (
    COST_PROFILES,
    DEFAULT_COST_PROFILE,
    DEFAULT_COST_PROFILE_ID,
    CostProfile,
    CostProfileError,
    get_cost_profile,
    is_legacy_cost_config,
    match_fee_tier,
    new_run_cost_profile,
    policy_side_cost_profile,
    resolve_run_cost_profile,
)
from trading.engines.paper import PaperExecutionEngine
from trading.exit_config import (
    ExitConfigError,
    ExitCostModel,
    parse_exit_config,
    policy_cost_model,
)
from trading.risk import RiskParameters
from trading.strategy_engine import StrategyEngine

_REPO = Path(__file__).resolve().parents[2]

_FINGERPRINTS = {
    "coinbase_advanced_1": "9369517879a6af6b",
    "coinbase_advanced_d6": "ca183b87232bc5b9",
    "coinbase_stress_t80": "c29cfac01872964f",
    "coinbase_stress_t100": "3eed378a7cb7af19",
    "frictionless_diagnostic": "6cf8f3a0990171a9",
    "legacy_backtest_v0": "6541ca4a0be16bb6",
    "legacy_runtime_v0": "e42b8a5c31b4b1f8",
}


def _profile(**overrides: Any) -> CostProfile:
    base: dict[str, Any] = {
        "id": "test_profile",
        "exchange": "generic",
        "tier": "test_tier",
        "maker_fee_bps": Decimal("10"),
        "taker_fee_bps": Decimal("15"),
        "slippage_bps": Decimal("5"),
    }
    base.update(overrides)
    return CostProfile(**base)


# ---------------------------------------------------------------------------
# WP21-T-01 -- validation
# ---------------------------------------------------------------------------


class TestT01Validation:
    def test_valid_profile_builds(self) -> None:
        p = _profile()
        assert p.fee_rate("taker") == Decimal("0.0015")
        assert p.fee_rate("maker") == Decimal("0.0010")
        assert p.slippage_rate == Decimal("0.0005")
        assert p.side_cost_rate() == Decimal("0.0020")
        assert p.round_trip_cost_rate() == Decimal("0.0040")

    @pytest.mark.parametrize("bad", [15, 15.0, True, "15", None])
    def test_non_decimal_numeric_fields_rejected(self, bad: object) -> None:
        for field in ("maker_fee_bps", "taker_fee_bps", "slippage_bps", "impact_coefficient_bps"):
            overrides: dict[str, Any] = {field: bad}
            if field == "maker_fee_bps":
                overrides["maker_fee_bps"] = bad
            with pytest.raises(CostProfileError):
                _profile(**overrides)

    @pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
    def test_non_finite_rejected(self, bad: Decimal) -> None:
        with pytest.raises(CostProfileError):
            _profile(taker_fee_bps=bad)

    def test_more_than_two_decimal_places_rejected(self) -> None:
        with pytest.raises(CostProfileError, match="decimal places"):
            _profile(taker_fee_bps=Decimal("15.001"))
        assert _profile(taker_fee_bps=Decimal("15.25")).taker_fee_bps == Decimal("15.25")

    def test_negative_rejected(self) -> None:
        with pytest.raises(CostProfileError):
            _profile(slippage_bps=Decimal("-1"))
        with pytest.raises(CostProfileError):
            _profile(maker_fee_bps=Decimal("-1"))

    def test_maker_above_taker_rejected(self) -> None:
        with pytest.raises(CostProfileError, match="maker_fee_bps must be <= taker_fee_bps"):
            _profile(maker_fee_bps=Decimal("20"), taker_fee_bps=Decimal("15"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("taker_fee_bps", Decimal("251")),
            ("slippage_bps", Decimal("201")),
            ("impact_coefficient_bps", Decimal("101")),
        ],
    )
    def test_upper_bounds(self, field: str, value: Decimal) -> None:
        with pytest.raises(CostProfileError, match="out of range"):
            _profile(**{field: value})

    def test_upper_bounds_are_inclusive(self) -> None:
        p = _profile(
            maker_fee_bps=Decimal("250"),
            taker_fee_bps=Decimal("250"),
            slippage_bps=Decimal("200"),
            impact_coefficient_bps=Decimal("100"),
        )
        assert p.taker_fee_bps == Decimal("250")

    def test_at_most_one_class_flag(self) -> None:
        for a, b in [
            ("live_equivalent", "stress"),
            ("diagnostic", "legacy"),
            ("legacy", "stress"),
            ("live_equivalent", "diagnostic"),
        ]:
            with pytest.raises(CostProfileError, match="at most one"):
                _profile(**{a: True, b: True})
        assert _profile(stress=True).stress is True

    @pytest.mark.parametrize("bad", ["ab", "UPPER_case", "has-dash", "x" * 49, "", "white space"])
    def test_id_regex(self, bad: str) -> None:
        with pytest.raises(CostProfileError, match="id must match"):
            _profile(id=bad)

    def test_fill_policy_and_source_validated(self) -> None:
        with pytest.raises(CostProfileError, match="fill_policy"):
            _profile(fill_policy="limit_maker")
        with pytest.raises(CostProfileError, match="source"):
            _profile(source="registry2")

    def test_with_overrides_is_a_custom_override(self) -> None:
        base = get_cost_profile("coinbase_advanced_d6")
        custom = base.with_overrides(taker_fee_bps=Decimal("90"))
        assert (custom.id, custom.source, custom.base_id) == ("custom", "override", base.id)
        assert custom.live_equivalent is False
        assert custom.taker_fee_bps == Decimal("90")
        assert custom.maker_fee_bps == base.maker_fee_bps
        assert custom.slippage_bps == base.slippage_bps
        assert custom.fingerprint() != base.fingerprint()


# ---------------------------------------------------------------------------
# WP21-T-02 -- registry
# ---------------------------------------------------------------------------


class TestT02Registry:
    EXPECTED: ClassVar[dict[str, tuple[str, str, int, int, int, str]]] = {
        # id: (exchange, tier, maker, taker, slip, flag)
        "coinbase_advanced_1": (
            "coinbase",
            "advanced_1_usd_0_1k_30d",
            60,
            120,
            10,
            "live_equivalent",
        ),
        "coinbase_advanced_d6": (
            "coinbase",
            "advanced_d6_assumed_0_10k",
            40,
            60,
            10,
            "live_equivalent",
        ),
        "coinbase_stress_t80": ("coinbase", "stress_taker_80", 40, 80, 10, "stress"),
        "coinbase_stress_t100": ("coinbase", "stress_taker_100", 40, 100, 10, "stress"),
        "frictionless_diagnostic": ("generic", "diagnostic_zero_cost", 0, 0, 0, "diagnostic"),
        "legacy_backtest_v0": ("generic", "pre_wp21_backtest", 10, 15, 5, "legacy"),
        "legacy_runtime_v0": ("coinbase", "pre_wp21_runtime", 40, 60, 5, "legacy"),
    }

    def test_registry_is_exactly_the_spec_table(self) -> None:
        assert set(COST_PROFILES) == set(self.EXPECTED)
        for pid, (exchange, tier, maker, taker, slip, flag) in self.EXPECTED.items():
            p = COST_PROFILES[pid]
            assert (p.id, p.exchange, p.tier) == (pid, exchange, tier)
            assert (p.maker_fee_bps, p.taker_fee_bps, p.slippage_bps) == (
                Decimal(maker),
                Decimal(taker),
                Decimal(slip),
            )
            assert p.impact_coefficient_bps == Decimal(0)
            assert p.source == "registry" and p.base_id is None
            flags = {
                f: getattr(p, f) for f in ("live_equivalent", "diagnostic", "legacy", "stress")
            }
            assert flags == {k: (k == flag) for k in flags}, pid

    def test_every_fingerprint_is_pinned(self) -> None:
        # Appending an id REQUIRES adding its pin here (spec 6.1: append-only).
        assert set(_FINGERPRINTS) == set(COST_PROFILES)
        for pid, fp in _FINGERPRINTS.items():
            assert COST_PROFILES[pid].fingerprint() == fp, pid

    def test_fingerprints_are_unique(self) -> None:
        assert len({p.fingerprint() for p in COST_PROFILES.values()}) == len(COST_PROFILES)

    def test_default_is_coinbase_advanced_1_and_live_equivalent(self) -> None:
        assert DEFAULT_COST_PROFILE_ID == "coinbase_advanced_1"
        assert DEFAULT_COST_PROFILE is COST_PROFILES[DEFAULT_COST_PROFILE_ID]
        assert DEFAULT_COST_PROFILE.live_equivalent is True
        # U-1: 60 bps maker / 120 bps taker + 10 bps slippage.
        assert (
            DEFAULT_COST_PROFILE.maker_fee_bps,
            DEFAULT_COST_PROFILE.taker_fee_bps,
            DEFAULT_COST_PROFILE.slippage_bps,
        ) == (Decimal(60), Decimal(120), Decimal(10))

    def test_registry_is_immutable(self) -> None:
        with pytest.raises(TypeError):
            COST_PROFILES["x"] = DEFAULT_COST_PROFILE  # type: ignore[index]
        with pytest.raises(TypeError):
            del COST_PROFILES["coinbase_advanced_1"]  # type: ignore[attr-defined]
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_COST_PROFILE.taker_fee_bps = Decimal(1)  # type: ignore[misc]

    def test_get_cost_profile_unknown_has_stable_code(self) -> None:
        with pytest.raises(CostProfileError) as exc:
            get_cost_profile("nope")
        assert exc.value.code == "unknown_cost_profile"
        assert "nope" not in str(exc.value)  # the request value is never echoed


# ---------------------------------------------------------------------------
# WP21-T-03 -- metadata round trip + strict parse
# ---------------------------------------------------------------------------


class TestT03Metadata:
    @pytest.mark.parametrize("pid", sorted(COST_PROFILES))
    def test_round_trip_is_exact_for_every_registry_id(self, pid: str) -> None:
        p = COST_PROFILES[pid]
        for fee_source in ("model", "exchange_fills"):
            meta = p.to_metadata(fee_source=fee_source)  # type: ignore[arg-type]
            assert meta["fee_source"] == fee_source
            assert CostProfile.from_metadata(meta) == p

    def test_metadata_is_json_safe(self) -> None:
        import json

        meta = DEFAULT_COST_PROFILE.to_metadata()
        assert json.loads(json.dumps(meta)) == meta
        assert (
            meta["taker_fee_bps"] == "120"
            and meta["fingerprint"] == _FINGERPRINTS["coinbase_advanced_1"]
        )

    def test_override_profile_round_trips(self) -> None:
        custom = get_cost_profile("coinbase_advanced_d6").with_overrides(
            taker_fee_bps=Decimal("90")
        )
        assert CostProfile.from_metadata(custom.to_metadata()) == custom

    def _meta(self) -> dict[str, Any]:
        return dict(DEFAULT_COST_PROFILE.to_metadata())

    def test_unknown_key_rejected(self) -> None:
        meta = self._meta()
        meta["surprise"] = 1
        with pytest.raises(CostProfileError, match="keys"):
            CostProfile.from_metadata(meta)

    def test_missing_key_rejected(self) -> None:
        meta = self._meta()
        del meta["tier"]
        with pytest.raises(CostProfileError, match="keys"):
            CostProfile.from_metadata(meta)

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("taker_fee_bps", 120),
            ("taker_fee_bps", 120.0),
            ("taker_fee_bps", "abc"),
            ("slippage_bps", None),
            ("legacy", "no"),
            ("live_equivalent", 1),
            ("schema", 2),
            ("schema", True),
            ("fee_source", "wallet"),
            ("source", "elsewhere"),
            ("fill_policy", "limit_maker"),
            ("id", 7),
            ("base_id", 7),
        ],
    )
    def test_wrong_types_rejected(self, key: str, bad: object) -> None:
        meta = self._meta()
        meta[key] = bad
        with pytest.raises(CostProfileError):
            CostProfile.from_metadata(meta)

    def test_not_a_mapping_rejected(self) -> None:
        for bad in (None, [], "x", 3):
            with pytest.raises(CostProfileError):
                CostProfile.from_metadata(bad)

    def test_fingerprint_mismatch_rejected(self) -> None:
        meta = self._meta()
        meta["fingerprint"] = "0" * 16
        with pytest.raises(CostProfileError, match="fingerprint"):
            CostProfile.from_metadata(meta)

    def test_registry_id_with_different_numbers_rejected(self) -> None:
        # numbers changed AND fingerprint recomputed: only the registry cross-check catches it
        tampered = dataclasses.replace(DEFAULT_COST_PROFILE, taker_fee_bps=Decimal("100"))
        with pytest.raises(CostProfileError, match="registry"):
            CostProfile.from_metadata(tampered.to_metadata())

    def test_registry_source_with_unknown_id_rejected(self) -> None:
        ghost = dataclasses.replace(DEFAULT_COST_PROFILE, id="ghost_profile")
        with pytest.raises(CostProfileError, match="not in the registry"):
            CostProfile.from_metadata(ghost.to_metadata())

    def test_an_id_with_a_trailing_newline_is_rejected(self) -> None:
        """WP21A-SEC-04: ``$`` matches before a final newline; the id uses ``fullmatch``.

        ``source="override"`` so the id regex is the ONLY rule that can reject it.
        """
        with pytest.raises(CostProfileError, match="id must match"):
            dataclasses.replace(DEFAULT_COST_PROFILE, source="override", id="custom\n")

    def test_a_base_id_with_a_trailing_newline_is_rejected(self) -> None:
        with pytest.raises(CostProfileError, match="base_id must match"):
            dataclasses.replace(
                DEFAULT_COST_PROFILE,
                source="override",
                id="custom",
                base_id="coinbase_advanced_1\n",
            )

    def test_a_persisted_override_with_a_newline_id_falls_back_to_legacy(self) -> None:
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        cfg = {"cost_profile": {**override.to_metadata(), "id": "custom\n"}}
        resolved = resolve_run_cost_profile(cfg, mode="paper")  # never raises
        assert resolved.invalid is True and resolved.persisted is False
        assert resolved.profile.id == "legacy_runtime_v0"


# ---------------------------------------------------------------------------
# WP21-T-04 -- resolve_run_cost_profile
# ---------------------------------------------------------------------------


class TestT04Resolve:
    def test_valid_persisted_profile(self) -> None:
        cfg = {"cost_profile": get_cost_profile("coinbase_advanced_d6").to_metadata()}
        r = resolve_run_cost_profile(cfg, mode="paper")
        assert r.profile.id == "coinbase_advanced_d6"
        assert (r.persisted, r.invalid) == (True, False)

    def test_exchange_fills_fee_source_is_carried(self) -> None:
        cfg = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata(fee_source="exchange_fills")}
        assert resolve_run_cost_profile(cfg, mode="live").fee_source == "exchange_fills"

    @pytest.mark.parametrize("config", [None, {}, {"strategy_name": "x"}, {"cost_profile": None}])
    def test_absent_backtest_is_legacy_backtest(self, config: Any) -> None:
        r = resolve_run_cost_profile(config, mode="backtest")
        assert r.profile.id == "legacy_backtest_v0"
        assert r.profile.source == "legacy_inferred"
        assert (r.persisted, r.invalid) == (False, False)

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_absent_runtime_modes_are_legacy_runtime(self, mode: str) -> None:
        r = resolve_run_cost_profile({}, mode=mode)
        assert r.profile.id == "legacy_runtime_v0" and r.profile.source == "legacy_inferred"
        assert r.profile.legacy is True

    def test_optimize_mode_uses_the_backtest_legacy_profile(self) -> None:
        assert resolve_run_cost_profile({}, mode="optimize").profile.id == "legacy_backtest_v0"

    def test_malformed_is_legacy_invalid_with_exactly_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "SENTINEL-" + "z" * 12
        cfg = {"cost_profile": {"id": secret, "taker_fee_bps": secret}}
        with caplog.at_level(logging.WARNING, logger="trading.costs"):
            r = resolve_run_cost_profile(cfg, mode="paper")
        assert r.profile.id == "legacy_runtime_v0"
        assert (r.persisted, r.invalid) == (False, True)
        records = [rec for rec in caplog.records if "cost_profile.invalid" in rec.getMessage()]
        assert len(records) == 1
        assert secret not in caplog.text  # CF-36: the stored value is never echoed

    def test_log_invalid_false_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="trading.costs"):
            r = resolve_run_cost_profile({"cost_profile": "junk"}, mode="live", log_invalid=False)
        assert r.invalid is True and caplog.records == []

    def test_bad_mode_is_a_value_error(self) -> None:
        for bad in ("papr", "", "BACKTEST", None):
            with pytest.raises(ValueError, match="mode"):
                resolve_run_cost_profile({}, mode=bad)  # type: ignore[arg-type]

    def test_resolve_never_mutates_the_config(self) -> None:
        cfg: dict[str, Any] = {"a": 1}
        resolve_run_cost_profile(cfg, mode="paper")
        assert cfg == {"a": 1}

    def test_is_legacy_cost_config(self) -> None:
        assert is_legacy_cost_config(None) and is_legacy_cost_config({})
        assert is_legacy_cost_config({"cost_profile": {"junk": 1}})
        assert is_legacy_cost_config(
            {"cost_profile": get_cost_profile("legacy_backtest_v0").to_metadata()}
        )
        assert not is_legacy_cost_config({"cost_profile": DEFAULT_COST_PROFILE.to_metadata()})


# ---------------------------------------------------------------------------
# WP21-T-05 -- new_run_cost_profile
# ---------------------------------------------------------------------------


class TestT05NewRun:
    @pytest.mark.parametrize("mode", ["backtest", "optimize", "paper", "live"])
    def test_none_is_the_default(self, mode: str) -> None:
        assert new_run_cost_profile(None, mode=mode) is DEFAULT_COST_PROFILE

    @pytest.mark.parametrize("mode", ["backtest", "optimize"])
    @pytest.mark.parametrize("pid", sorted(COST_PROFILES))
    def test_backtest_and_optimize_accept_every_registry_id(self, mode: str, pid: str) -> None:
        assert new_run_cost_profile(pid, mode=mode) is COST_PROFILES[pid]

    def test_paper_accepts_only_live_equivalent_ids(self) -> None:
        for pid, p in COST_PROFILES.items():
            if p.live_equivalent:
                assert new_run_cost_profile(pid, mode="paper") is p
            else:
                with pytest.raises(CostProfileError) as exc:
                    new_run_cost_profile(pid, mode="paper")
                assert exc.value.code == "cost_profile_not_live_equivalent"

    def test_live_accepts_no_id_at_all(self) -> None:
        for pid in [*COST_PROFILES, "nope"]:
            with pytest.raises(CostProfileError) as exc:
                new_run_cost_profile(pid, mode="live")
            assert exc.value.code == "cost_profile_not_allowed_for_live"

    def test_unknown_id_code(self) -> None:
        for mode in ("backtest", "paper"):
            with pytest.raises(CostProfileError) as exc:
                new_run_cost_profile("nope", mode=mode)
            assert exc.value.code == "unknown_cost_profile"

    def test_bad_mode_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            new_run_cost_profile(None, mode="shadow")


# ---------------------------------------------------------------------------
# WP21-T-06 -- ExitCostModel
# ---------------------------------------------------------------------------


class TestT06ExitCostModel:
    @pytest.mark.parametrize(
        ("pid", "c_side", "c_rt"),
        [
            ("coinbase_advanced_1", 0.0130, 0.0260),
            ("coinbase_advanced_d6", 0.0070, 0.0140),
            ("legacy_runtime_v0", 0.0065, 0.0130),
            ("legacy_backtest_v0", 0.0020, 0.0040),
        ],
    )
    def test_exact_floats_of_the_decimal(self, pid: str, c_side: float, c_rt: float) -> None:
        m = ExitCostModel.from_cost_profile(get_cost_profile(pid))
        # exact: float(Decimal) ONCE, not 0.012 + 0.001 in float (0.013000000000000001)
        assert (m.c_side, m.c_rt) == (c_side, c_rt)

    def test_pos_cap_and_r_come_from_risk_parameters(self) -> None:
        m = ExitCostModel.from_cost_profile(DEFAULT_COST_PROFILE)
        d = RiskParameters()
        assert (m.pos_cap, m.r) == (d.max_position_size_pct, d.per_trade_risk_pct)
        custom = RiskParameters(max_position_size_pct=0.5, per_trade_risk_pct=0.01)
        m2 = ExitCostModel.from_cost_profile(DEFAULT_COST_PROFILE, custom)
        assert (m2.pos_cap, m2.r) == (0.5, 0.01)

    def test_policy_picks_the_stricter_of_default_and_run(self) -> None:
        d6 = get_cost_profile("coinbase_advanced_d6")
        legacy = get_cost_profile("legacy_runtime_v0")
        adv1 = get_cost_profile("coinbase_advanced_1")
        # default (adv1) is stricter than d6 / legacy -> the default wins
        assert policy_side_cost_profile(d6) is DEFAULT_COST_PROFILE
        assert policy_cost_model(legacy).c_side == 0.0130
        assert policy_cost_model(adv1).c_side == 0.0130
        # a run profile stricter than the default wins (default moved below it)
        with patch.object(costs_module, "DEFAULT_COST_PROFILE", d6):
            assert policy_side_cost_profile(adv1) is adv1
            assert policy_cost_model(adv1).c_side == 0.0130
            assert policy_cost_model(legacy).c_side == 0.0070

    def test_floor_boundary_is_exact_at_the_default(self) -> None:
        bracket = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.013}
        with pytest.raises(ExitConfigError):
            parse_exit_config(
                bracket=bracket,
                trailing_stop_pct=None,
                cost=policy_cost_model(DEFAULT_COST_PROFILE),
            )
        ok = parse_exit_config(
            bracket={"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.0131},
            trailing_stop_pct=None,
            cost=policy_cost_model(DEFAULT_COST_PROFILE),
        )
        assert ok.bracket["bracket_stop_loss_pct"] == 0.0131


# ---------------------------------------------------------------------------
# WP21-T-07 -- AST pins
# ---------------------------------------------------------------------------

_COST_KWARGS = frozenset(
    {
        "taker_fee_bps",
        "maker_fee_bps",
        "slippage_bps",
        "impact_coefficient_bps",
        "taker_fee_pct",
        "maker_fee_pct",
    }
)
_GUARDED_CALLS = frozenset(
    {
        "PaperExecutionEngine",
        "BacktestRunner",
        "ParameterOptimizer",
        "RiskParameters",
        "StrategyEngine",
        "LiveExecutionEngine",
    }
)


def _production_files() -> list[Path]:
    out: list[Path] = []
    for root in ("apps", "packages", "scripts"):
        out.extend(
            p
            for p in (_REPO / root).rglob("*.py")
            if "node_modules" not in p.parts and "__pycache__" not in p.parts
        )
    return sorted(out)


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class TestT07AstPins:
    def test_costs_module_is_a_leaf_that_reads_no_environment(self) -> None:
        src = (_REPO / "packages/trading/costs.py").read_text()
        tree = ast.parse(src)
        stdlib = sys.stdlib_module_names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in stdlib, alias.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top = (node.module or "").split(".")[0]
                assert top in stdlib or node.module == "common.types", node.module
        forbidden = {"environ", "getenv", "putenv", "os", "sys", "dotenv", "argv"}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert used.isdisjoint(forbidden), used & forbidden

    def test_risk_parameters_has_no_cost_fields(self) -> None:
        names = {f.name for f in dataclasses.fields(RiskParameters)}
        assert not {n for n in names if "fee" in n or "slippage" in n or "impact" in n}
        with pytest.raises(TypeError):
            RiskParameters(taker_fee_pct=0.001)  # type: ignore[call-arg]

    def test_no_cost_kwarg_reaches_a_guarded_constructor_in_production_code(self) -> None:
        offenders: list[str] = []
        for path in _production_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) in _GUARDED_CALLS:
                    for kw in node.keywords:
                        if kw.arg in _COST_KWARGS:
                            offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}:{kw.arg}")
        assert offenders == []

    def test_no_fee_pct_names_survive_in_production_code(self) -> None:
        offenders: list[str] = []
        for path in _production_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Attribute):
                    names.append(node.attr)
                elif isinstance(node, ast.Name):
                    names.append(node.id)
                elif isinstance(node, ast.arg):
                    names.append(node.arg)
                elif isinstance(node, ast.keyword) and node.arg:
                    names.append(node.arg)
                for n in names:
                    if n in {"taker_fee_pct", "maker_fee_pct"}:
                        offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}:{n}")
        assert offenders == []

    def test_every_production_engine_construction_passes_cost_profile(self) -> None:
        needed = {"PaperExecutionEngine", "StrategyEngine", "LiveExecutionEngine"}
        seen: dict[str, int] = dict.fromkeys(needed, 0)
        for path in _production_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) in needed:
                    kws = {kw.arg for kw in node.keywords}
                    assert "cost_profile" in kws, f"{path.relative_to(_REPO)}:{node.lineno}"
                    seen[_call_name(node)] += 1  # type: ignore[index]
        # paper: backtest + orchestrator; strategy: backtest + 2 orchestrator; live: orchestrator
        assert seen == {"PaperExecutionEngine": 2, "StrategyEngine": 3, "LiveExecutionEngine": 1}

    def test_every_production_runner_and_optimizer_construction_passes_cost_profile(self) -> None:
        count = 0
        for path in _production_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) in {
                    "BacktestRunner",
                    "ParameterOptimizer",
                }:
                    kws = {kw.arg for kw in node.keywords}
                    assert "cost_profile" in kws, f"{path.relative_to(_REPO)}:{node.lineno}"
                    count += 1
        # optimizer (BacktestRunner), create_run, ml gate, 2 scripts, optimize router,
        # and (WP2.1c) the re-baseline script
        assert count == 7

    def test_settings_has_no_cost_field(self) -> None:
        from api.config import Settings

        assert [n for n in Settings.model_fields if "cost" in n.lower()] == []

    def test_lifespan_calls_the_env_warning(self) -> None:
        src = (_REPO / "apps/api/main.py").read_text()
        assert "warn_ignored_cost_profile_env()" in src


# ---------------------------------------------------------------------------
# WP21-T-08 -- paper arithmetic
# ---------------------------------------------------------------------------


def _paper(profile: CostProfile) -> PaperExecutionEngine:
    rm = MagicMock()
    return PaperExecutionEngine(run_id="t08", risk_manager=rm, cost_profile=profile)


class TestT08PaperArithmetic:
    def test_buy_and_sell_slippage_and_fee_on_the_slipped_price(self) -> None:
        eng = _paper(get_cost_profile("coinbase_advanced_1"))
        buy = eng._apply_slippage(Decimal("100"), OrderSide.BUY)
        sell = eng._apply_slippage(Decimal("100"), OrderSide.SELL)
        assert buy == Decimal("100.10000000")  # 100 * (1 + 10/10000)
        assert sell == Decimal("99.90000000")  # 100 * (1 - 10/10000)
        fee = eng._calculate_fee(Decimal("2"), buy, is_maker=False)
        assert fee == Decimal("2.40240000")  # 2 * 100.10 * 0.012

    def test_fee_rounds_half_up_to_8dp(self) -> None:
        eng = _paper(_profile(taker_fee_bps=Decimal("1"), maker_fee_bps=Decimal("0")))
        # notional * 0.0001: a tie at the 9th decimal place rounds UP (ROUND_HALF_UP)
        assert eng._calculate_fee(Decimal("0.00005"), Decimal("1"), is_maker=False) == Decimal(
            "0.00000001"
        )  # 5e-9 -> 1e-8
        assert eng._calculate_fee(Decimal("0.00004"), Decimal("1"), is_maker=False) == Decimal(
            "0.00000000"
        )  # 4e-9 -> 0

    def test_limit_orders_use_the_maker_rate(self) -> None:
        eng = _paper(get_cost_profile("coinbase_advanced_1"))
        assert eng._calculate_fee(Decimal("1"), Decimal("100"), is_maker=True) == Decimal("0.6")
        assert eng._calculate_fee(Decimal("1"), Decimal("100"), is_maker=False) == Decimal("1.2")

    def test_impact_term_comes_from_the_profile(self) -> None:
        p = _profile(impact_coefficient_bps=Decimal("100"), slippage_bps=Decimal("5"))
        eng = _paper(p)
        eng.set_last_bar("BTC/USDT", Decimal("100"), Decimal("100"))
        out = eng._apply_slippage(
            Decimal("100"), OrderSide.BUY, symbol="BTC/USDT", quantity=Decimal("10")
        )
        assert Decimal("100.05") < out < Decimal("101")
        no_impact = _paper(_profile(slippage_bps=Decimal("5")))
        no_impact.set_last_bar("BTC/USDT", Decimal("100"), Decimal("100"))
        assert no_impact._apply_slippage(
            Decimal("100"), OrderSide.BUY, symbol="BTC/USDT", quantity=Decimal("10")
        ) == Decimal("100.05000000")

    def test_default_engine_profile_is_the_default(self) -> None:
        eng = PaperExecutionEngine(run_id="t08d", risk_manager=MagicMock())
        assert eng.cost_profile is DEFAULT_COST_PROFILE

    def test_risk_manager_params_are_not_read_for_fees(self) -> None:
        rm = MagicMock()
        rm.params.taker_fee_pct = 0.5  # a stale source that must be ignored
        eng = PaperExecutionEngine(
            run_id="t08r", risk_manager=rm, cost_profile=get_cost_profile("legacy_backtest_v0")
        )
        assert eng._calculate_fee(Decimal("1"), Decimal("100"), is_maker=False) == Decimal(
            "0.15000000"
        )


# ---------------------------------------------------------------------------
# WP21-T-09 -- C25
# ---------------------------------------------------------------------------


def _hold_strategy() -> ScriptedSequenceStrategy:
    return ScriptedSequenceStrategy("t09", {"script": {}})


class TestT09C25:
    def test_custom_risk_params_reach_the_backtest_risk_manager_unchanged(self) -> None:
        custom = RiskParameters(
            sizing_mode="atr",
            atr_risk_multiplier=Decimal("2.5"),
            max_cluster_exposure_pct=0.3,
            max_position_size_pct=0.2,
        )
        runner = BacktestRunner(
            strategies=[_hold_strategy()],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            risk_params=custom,
            seed=1,
        )
        _engine, _pf, rm, _ex = runner._build_engine("t09")
        assert rm.params is custom
        assert (rm.params.sizing_mode, rm.params.atr_risk_multiplier) == ("atr", Decimal("2.5"))
        assert rm.params.max_cluster_exposure_pct == 0.3

    def test_default_profile_and_default_risk_params(self) -> None:
        runner = BacktestRunner(
            strategies=[_hold_strategy()],
            symbols=["BTC/USDT"],
            timeframe=TimeFrame.ONE_HOUR,
            seed=1,
        )
        assert runner.cost_profile is DEFAULT_COST_PROFILE
        _e, _p, rm, ex = runner._build_engine("t09b")
        assert rm.params == RiskParameters()
        assert ex.cost_profile is DEFAULT_COST_PROFILE


# ---------------------------------------------------------------------------
# WP21-T-10 -- engine-side parse at the run's OWN profile
# ---------------------------------------------------------------------------

_SL_0068 = {"bracket_mode": "fixed", "bracket_stop_loss_pct": 0.0068}


def _engine(profile: CostProfile | None, bracket: dict[str, object]) -> StrategyEngine:
    config: dict[str, Any] = dict(bracket)
    return StrategyEngine(
        strategies=[_hold_strategy()],
        execution_engine=MagicMock(),
        risk_manager=MagicMock(),
        market_data=MagicMock(),
        portfolio=MagicMock(),
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        run_mode=RunMode.BACKTEST,
        config=config,
        cost_profile=profile,
    )


class TestT10EngineSideParse:
    def test_strategy_engine_accepts_sl_0068_under_legacy_runtime_only(self) -> None:
        legacy = get_cost_profile("legacy_runtime_v0")
        assert _engine(legacy, _SL_0068).cost_profile is legacy
        with pytest.raises(ExitConfigError):
            _engine(DEFAULT_COST_PROFILE, _SL_0068)
        with pytest.raises(ExitConfigError):
            _engine(None, _SL_0068)  # None -> the default

    def test_backtest_runner_accepts_sl_0068_under_legacy_runtime_only(self) -> None:
        def build(profile: CostProfile | None) -> BacktestRunner:
            return BacktestRunner(
                strategies=[_hold_strategy()],
                symbols=["BTC/USDT"],
                timeframe=TimeFrame.ONE_HOUR,
                bracket_config=dict(_SL_0068),
                cost_profile=profile,
                seed=1,
            )

        assert build(get_cost_profile("legacy_runtime_v0")).cost_profile.id == "legacy_runtime_v0"
        with pytest.raises(ExitConfigError):
            build(DEFAULT_COST_PROFILE)
        with pytest.raises(ExitConfigError):
            build(None)


# ---------------------------------------------------------------------------
# WP21-T-31 (a) -- mock-safety
# ---------------------------------------------------------------------------


class TestT31aMockSafety:
    def test_strategy_engine_with_a_mock_risk_manager_builds_its_exit_cost_model(self) -> None:
        eng = _engine(None, {})
        assert eng._exit_cost_model == ExitCostModel.from_cost_profile(DEFAULT_COST_PROFILE)
        assert eng.cost_profile is DEFAULT_COST_PROFILE


# ---------------------------------------------------------------------------
# WP21-T-31 (b) -- DEFAULT_COST_PROFILE in the environment is ignored
# ---------------------------------------------------------------------------


class TestT31bEnvIgnored:
    def test_env_var_is_ignored_and_warned_by_name_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from api.config import get_settings, warn_ignored_cost_profile_env

        marker = "coinbase_advanced_d6"
        monkeypatch.setenv("DEFAULT_COST_PROFILE", marker)
        get_settings.cache_clear()
        try:
            with capture_logs() as logs:
                names = warn_ignored_cost_profile_env()
            assert names == ["DEFAULT_COST_PROFILE"]
            events = [e for e in logs if e["event"] == "cost_profile.env_ignored"]
            assert len(events) == 1
            assert events[0]["log_level"] == "warning"
            assert events[0]["names"] == ["DEFAULT_COST_PROFILE"]
            assert marker not in repr(events[0])  # the NAME is logged, never the value
            # nothing moved: the default is the code constant
            assert costs_module.default_cost_profile().id == "coinbase_advanced_1"
            assert get_settings() is not None
            assert [n for n in get_settings().model_fields if "cost" in n.lower()] == []
        finally:
            get_settings.cache_clear()

    def test_any_cost_profile_named_variable_is_reported(self) -> None:
        from api.config import warn_ignored_cost_profile_env

        env = {"X_COST_PROFILE_ID": "v1", "PATH": "/bin", "cost_profile": "v2"}
        with capture_logs() as logs:
            names = warn_ignored_cost_profile_env(env)
        assert names == ["X_COST_PROFILE_ID", "cost_profile"]
        assert len([e for e in logs if e["event"] == "cost_profile.env_ignored"]) == 1
        assert "v1" not in repr(logs) and "v2" not in repr(logs)

    def test_no_warning_without_such_a_variable(self) -> None:
        from api.config import warn_ignored_cost_profile_env

        with capture_logs() as logs:
            assert warn_ignored_cost_profile_env({"PATH": "/bin"}) == []
        assert logs == []


# ---------------------------------------------------------------------------
# WP21-T-32 -- BacktestResult cost fields
# ---------------------------------------------------------------------------


async def _run_with(profile: CostProfile) -> Any:
    from tests.integration.fakes.bar_paths import scripted_path

    strat = ScriptedSequenceStrategy(
        "t32", {"script": {0: "buy", 4: "sell"}, "target_notional": "3000"}
    )
    runner = BacktestRunner(
        strategies=[strat],
        symbols=["BTC/USDT"],
        timeframe=TimeFrame.ONE_HOUR,
        seed=3,
        cost_profile=profile,
    )
    bars = scripted_path(
        [100, 101, 102, 103, 104, 103, 102, 101],
        first_open=datetime(2026, 3, 10, 10, 0, tzinfo=UTC),
        symbol="BTC/USDT",
    )
    result = await runner.run({"BTC/USDT": bars})
    return runner, result


class TestT32ResultFields:
    async def test_total_slippage_cost_is_sum_abs_fill_minus_expected_times_qty(self) -> None:
        runner, result = await _run_with(get_cost_profile("coinbase_advanced_1"))
        assert runner.last_execution_engine is not None
        fills = runner.last_execution_engine.get_all_fills()
        assert len(fills) == 2
        expected = sum(
            (abs(f.price - f.expected_price) * f.quantity for f in fills),  # type: ignore[operator]
            Decimal(0),
        )
        assert result.total_slippage_cost == expected > Decimal(0)
        # 10 bps on each of two legs: 15 units * (100 * 0.001 + 104 * 0.001)
        assert result.total_slippage_cost == Decimal("15") * (
            Decimal("100.10000000") - Decimal("100")
        ) + Decimal("15") * (Decimal("104") - Decimal("103.89600000"))
        assert result.cost_profile_id == "coinbase_advanced_1"
        assert result.cost_profile_fingerprint == _FINGERPRINTS["coinbase_advanced_1"]

    async def test_frictionless_has_zero_slippage_cost_and_zero_fees(self) -> None:
        _runner, result = await _run_with(get_cost_profile("frictionless_diagnostic"))
        assert result.total_slippage_cost == Decimal(0)
        assert result.total_fees_paid == Decimal(0)
        assert result.cost_profile_id == "frictionless_diagnostic"


# ---------------------------------------------------------------------------
# WP21-T-38 -- match_fee_tier
# ---------------------------------------------------------------------------


class TestT38MatchFeeTier:
    def test_known_tiers(self) -> None:
        assert match_fee_tier("0.012", "0.006") == "coinbase_advanced_1"
        assert match_fee_tier("0.0120", "0.0060") == "coinbase_advanced_1"
        assert match_fee_tier("0.006", "0.004") == "coinbase_advanced_d6"
        assert match_fee_tier(Decimal("0.006"), Decimal("0.004")) == "coinbase_advanced_d6"

    def test_unknown_tier_is_none(self) -> None:
        assert match_fee_tier("0.008", "0.006") is None
        assert match_fee_tier("0.012", "0.004") is None  # taker of one, maker of the other

    @pytest.mark.parametrize(
        "taker",
        [
            0.012,
            12,
            None,
            True,
            b"0.012",
            ["0.012"],
            "garbage",
            "",
            "NaN",
            "Infinity",
            object(),
            # WP21A-CR-03/SEC-03: finite but absurd exponents overflow ``rate * _BPS``
            "1e999999999",
            "-1e999999999",
            "1e9999999999",
            Decimal("1e999999999"),
        ],
    )
    def test_floats_ints_none_and_garbage_never_raise(self, taker: object) -> None:
        assert match_fee_tier(taker, "0.006") is None
        assert match_fee_tier("0.012", taker) is None

    def test_only_live_equivalent_ids_match(self) -> None:
        # stress / legacy / diagnostic rates never match, even though they exist
        assert match_fee_tier("0.008", "0.004") is None  # coinbase_stress_t80
        assert match_fee_tier("0.0015", "0.0010") is None  # legacy_backtest_v0
        assert match_fee_tier("0", "0") is None  # frictionless_diagnostic

    def test_the_default_matches_its_own_registry_rates(self) -> None:
        p = DEFAULT_COST_PROFILE
        assert (
            match_fee_tier(str(p.fee_rate("taker")), str(p.fee_rate("maker")))
            == DEFAULT_COST_PROFILE_ID
        )


# ---------------------------------------------------------------------------
# WP21-T-30 -- idempotency fingerprint
# ---------------------------------------------------------------------------


class TestT30IdempotencyFingerprint:
    #: Fixed vector of the PRE-2.1a fingerprint of this exact body (computed on pristine
    #: HEAD 5b1151f with ``compute_fingerprint(CREATE_ENDPOINT, {}, body.model_dump(mode="json",
    #: exclude={"confirm_token"}))``).
    PRE_2_1A_FINGERPRINT = "649351dc0e796c59d4512feb29c2e33c4ba24c326ba5f42ceee043cc2206dddc"

    @staticmethod
    def _body(**extra: Any) -> Any:
        from api.schemas import RunCreateRequest

        return RunCreateRequest.model_validate(
            {
                "strategyName": "momentum_breakout",
                "strategyParams": {"lookback": 5},
                "symbols": ["BTC/USD"],
                "timeframe": "1h",
                "mode": "backtest",
                "initialCapital": "10000.00",
                "backtestStart": "2024-01-01T00:00:00Z",
                "backtestEnd": "2024-06-01T00:00:00Z",
                **extra,
            }
        )

    @staticmethod
    def _fingerprint(body: Any) -> str:
        """Mirror of create_run's computation (the handler is the source of truth)."""
        from api.services.idempotency import CREATE_ENDPOINT, compute_fingerprint

        exclude = {"confirm_token"}
        if body.cost_profile is None:
            exclude.add("cost_profile")
        return compute_fingerprint(
            CREATE_ENDPOINT, {}, body.model_dump(mode="json", exclude=exclude)
        )

    def test_body_without_cost_profile_matches_the_pre_2_1a_fingerprint(self) -> None:
        assert self._fingerprint(self._body()) == self.PRE_2_1A_FINGERPRINT

    def test_body_with_cost_profile_differs(self) -> None:
        with_cp = self._fingerprint(self._body(costProfile="coinbase_advanced_d6"))
        assert with_cp != self.PRE_2_1A_FINGERPRINT
        assert with_cp != self._fingerprint(self._body(costProfile="coinbase_stress_t80"))
