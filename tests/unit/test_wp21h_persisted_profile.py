"""
tests/unit/test_wp21h_persisted_profile.py
-------------------------------------------
WP2.1-H hardening of the cost-profile reader (reports/vp2-wp2.1b/final-synthesis.md
section 5.5; security-r1 SEC-03; 2.1a security report SEC-05; WP21H-SEC-01).

    WP21H-T-01  ``CostProfile.exchange`` is bounded to 1..48 characters, like ``tier``
    WP21H-T-02  a STORED profile is valid only if it is a registry profile, or the
                mode family's own legacy profile (``legacy_inferred``); anything else
                reads as invalid -> the mode's legacy profile + ``invalid`` and is
                never echoed (CF-21-SEC-A, SEC-03).  For paper / live the registry
                profile must also be ``live_equivalent`` (SEC-01)
    WP21H-T-03  legitimate rows resolve exactly as before (registry, absent keys,
                legacy-by-absence, the auto-retry's legacy row); the registry matrix is
                split by mode family and ``live_equivalent``
    WP21H-T-04  the readers are closed: only ``from_metadata`` and ``_parse_persisted``
                call the schema parser; ``resolve_run_cost_profile`` is the only
                reader of a stored cost PROFILE in production code (it is also the only
                reader of its cost BASIS; ``run_persistence`` reads the stored
                *fingerprint* to compare it with the result's and logs it only when it
                is a well-formed digest, see WP21H-T-07)
    WP21H-T-05  the live / paper resume callers resolve with a runtime-family mode
                (source pin; the behavioural half is in
                ``tests/integration/test_wp21h_resume_basis.py``)
    WP21H-T-06  invariants (SEC-01): the paper / live basis is never below
                ``legacy_runtime_v0``, and every profile a runtime writer produces is
                inside the paper / live reader's allow-list
    WP21H-T-07  the stored fingerprint is logged only when well-formed (SEC-03)
                (see ``tests/unit/test_wp21h_fingerprint_log.py``)

Accepted residual (WP21H-SEC-04): a tampered paper / live row may still name another
LIVE-EQUIVALENT registry id; the stored id is not integrity-protected against a database
writer.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from api.services.cost_profiles import cost_response_fields
from trading import costs as costs_module
from trading.costs import (
    COST_PROFILES,
    DEFAULT_COST_PROFILE,
    CostProfile,
    CostProfileError,
    default_cost_profile,
    get_cost_profile,
    is_leaderboard_cost_eligible,
    is_legacy_cost_config,
    new_run_cost_profile,
    resolve_run_cost_profile,
)

_REPO = Path(__file__).resolve().parents[2]
_ALL_MODES = ("backtest", "optimize", "paper", "live")
_RUNTIME_MODES = ("paper", "live")
_BACKTEST_MODES = ("backtest", "optimize")
_SENTINEL = "SENTINEL-" + "q" * 14
_LIVE_EQUIVALENT_IDS = tuple(sorted(i for i, p in COST_PROFILES.items() if p.live_equivalent))
_NOT_LIVE_EQUIVALENT_IDS = tuple(
    sorted(i for i, p in COST_PROFILES.items() if not p.live_equivalent)
)
_FEE_SOURCES = ("model", "exchange_fills")
_LEGACY_RUNTIME = dataclasses.replace(COST_PROFILES["legacy_runtime_v0"], source="legacy_inferred")


def _legacy_inferred(legacy_id: str, **changes: Any) -> CostProfile:
    return dataclasses.replace(COST_PROFILES[legacy_id], source="legacy_inferred", **changes)


def _cfg(profile: CostProfile, **meta: Any) -> dict[str, Any]:
    return {"cost_profile": {**profile.to_metadata(), **meta}}


def _family_legacy_id(mode: str) -> str:
    return "legacy_backtest_v0" if mode in _BACKTEST_MODES else "legacy_runtime_v0"


def _assert_invalid_falls_back_to_the_mode_legacy(config: Any, mode: str) -> None:
    resolved = resolve_run_cost_profile(config, mode=mode, log_invalid=False)
    assert (resolved.persisted, resolved.invalid) == (False, True)
    assert resolved.profile == _legacy_inferred(_family_legacy_id(mode))
    assert resolved.profile.legacy is True and resolved.fee_source == "model"


# ---------------------------------------------------------------------------
# WP21H-T-01 -- exchange bound
# ---------------------------------------------------------------------------


class TestT01ExchangeBound:
    def _with_exchange(self, exchange: object) -> CostProfile:
        return dataclasses.replace(
            DEFAULT_COST_PROFILE,
            id="custom",
            source="override",
            exchange=exchange,  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize("length", [1, 8, 48])
    def test_one_to_forty_eight_characters_are_accepted(self, length: int) -> None:
        assert self._with_exchange("e" * length).exchange == "e" * length

    @pytest.mark.parametrize("bad", ["", "e" * 49, "e" * 28_000, None, 7, b"coinbase"])
    def test_out_of_bounds_or_non_string_is_rejected(self, bad: object) -> None:
        with pytest.raises(CostProfileError, match=r"exchange must be a string of 1\.\.48"):
            self._with_exchange(bad)

    def test_the_error_never_echoes_the_value(self) -> None:
        with pytest.raises(CostProfileError) as exc:
            self._with_exchange(_SENTINEL * 5)
        assert _SENTINEL not in str(exc.value)

    def test_the_sec_03_probe_is_rejected_by_the_strict_parser(self) -> None:
        """A 28k-character ``exchange`` (SEC-03) never builds a profile: the bound is the
        ONLY rule that can reject this metadata (override source, schema otherwise valid)."""
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        meta = {**override.to_metadata(), "exchange": "<b>" + "x" * 28_000}
        assert CostProfile.from_metadata(override.to_metadata()) == override  # control
        with pytest.raises(CostProfileError, match="exchange"):
            CostProfile.from_metadata(meta)

    def test_every_registry_exchange_is_within_the_bound(self) -> None:
        assert all(0 < len(p.exchange) <= 48 for p in COST_PROFILES.values())


# ---------------------------------------------------------------------------
# WP21H-T-02 -- tampered rows are invalid, never echoed
# ---------------------------------------------------------------------------


class TestT02TamperedRowsAreInvalid:
    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_a_persisted_override_is_invalid(self, mode: str) -> None:
        override = get_cost_profile("coinbase_advanced_d6").with_overrides(
            maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1")
        )
        assert override.source == "override"
        _assert_invalid_falls_back_to_the_mode_legacy(
            {"cost_profile": override.to_metadata()}, mode
        )

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_legacy_inferred_with_legacy_false_is_invalid(self, mode: str) -> None:
        """The SEC-03 probe: the legacy numbers and id, but ``legacy=false``."""
        for legacy_id in ("legacy_runtime_v0", "legacy_backtest_v0"):
            probe = _legacy_inferred(legacy_id, legacy=False)
            assert probe.legacy is False
            _assert_invalid_falls_back_to_the_mode_legacy(_cfg(probe), mode)

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    def test_the_backtest_legacy_row_is_invalid_for_paper_and_live(self, mode: str) -> None:
        _assert_invalid_falls_back_to_the_mode_legacy(
            _cfg(_legacy_inferred("legacy_backtest_v0")), mode
        )

    @pytest.mark.parametrize("mode", _BACKTEST_MODES)
    def test_the_runtime_legacy_row_is_invalid_for_backtest_and_optimize(self, mode: str) -> None:
        _assert_invalid_falls_back_to_the_mode_legacy(
            _cfg(_legacy_inferred("legacy_runtime_v0")), mode
        )

    @pytest.mark.parametrize("mode", _ALL_MODES)
    @pytest.mark.parametrize(
        "changes",
        [
            pytest.param(
                {"maker_fee_bps": Decimal("0"), "taker_fee_bps": Decimal("1")}, id="cheaper_fees"
            ),
            pytest.param({"slippage_bps": Decimal("0")}, id="no_slippage"),
            pytest.param({"exchange": "elsewhere"}, id="other_exchange"),
            pytest.param({"tier": "cheap_tier"}, id="other_tier"),
            pytest.param({"base_id": "coinbase_advanced_1"}, id="base_id_set"),
            pytest.param({"live_equivalent": True, "legacy": False}, id="flags_flipped"),
        ],
    )
    def test_an_edited_legacy_row_is_invalid_even_with_a_recomputed_fingerprint(
        self, mode: str, changes: dict[str, Any]
    ) -> None:
        # ``to_metadata`` recomputes the fingerprint, so only the equality with the
        # mode's own legacy profile can reject these.
        edited = _legacy_inferred(_family_legacy_id(mode), **changes)
        _assert_invalid_falls_back_to_the_mode_legacy(_cfg(edited), mode)

    @pytest.mark.parametrize("mode", _ALL_MODES)
    @pytest.mark.parametrize(
        "pid", sorted(set(COST_PROFILES) - {"legacy_runtime_v0", "legacy_backtest_v0"})
    )
    def test_a_registry_economics_row_relabelled_legacy_inferred_is_invalid(
        self, mode: str, pid: str
    ) -> None:
        relabelled = dataclasses.replace(COST_PROFILES[pid], source="legacy_inferred")
        _assert_invalid_falls_back_to_the_mode_legacy(_cfg(relabelled), mode)

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    def test_a_tampered_row_gives_the_runtime_legacy_basis_on_the_resume_paths(
        self, mode: str
    ) -> None:
        """What the three ``runs.py`` resume callers receive for a tampered row."""
        override = DEFAULT_COST_PROFILE.with_overrides(
            maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1")
        )
        profile = resolve_run_cost_profile(
            {"cost_profile": override.to_metadata()}, mode=mode
        ).profile
        assert (profile.id, profile.source, profile.legacy) == (
            "legacy_runtime_v0",
            "legacy_inferred",
            True,
        )
        assert (profile.maker_fee_bps, profile.taker_fee_bps, profile.slippage_bps) == (
            Decimal(40),
            Decimal(60),
            Decimal(5),
        )

    def test_an_invalid_value_is_never_echoed_and_logs_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        tampered = {
            "cost_profile": {
                **override.to_metadata(),
                "tier": _SENTINEL,
                "base_id": "coinbase_advanced_1",
            }
        }
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_run_cost_profile(tampered, mode="live")
        assert resolved.invalid is True
        assert _SENTINEL not in repr(resolved)
        assert _SENTINEL not in caplog.text
        records = [r for r in caplog.records if "cost_profile.invalid" in r.getMessage()]
        assert len(records) == 1

    def test_log_invalid_false_stays_silent_for_a_tampered_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_run_cost_profile(
                {"cost_profile": override.to_metadata()}, mode="paper", log_invalid=False
            )
        assert resolved.invalid is True and caplog.records == []

    def test_the_parser_error_text_carries_no_value(self) -> None:
        tampered = _legacy_inferred("legacy_runtime_v0", tier=_SENTINEL)
        with pytest.raises(CostProfileError) as exc:
            costs_module._parse_persisted(tampered.to_metadata(), mode="paper")
        assert _SENTINEL not in str(exc.value)

    @pytest.mark.parametrize("mode", _ALL_MODES)
    @pytest.mark.parametrize(
        "raw",
        [
            "x" * 28_000,
            ["a"] * 50,
            {"k": {"nested": [1, 2, {"deep": None}]}},
            {**DEFAULT_COST_PROFILE.to_metadata(), "exchange": "<script>" + "x" * 28_000},
            {**DEFAULT_COST_PROFILE.to_metadata(), "taker_fee_bps": "9" * 5_000},
            {**DEFAULT_COST_PROFILE.to_metadata(), "maker_fee_bps": "1E-999999999"},
            {**DEFAULT_COST_PROFILE.to_metadata(), "slippage_bps": "NaN"},
            {**DEFAULT_COST_PROFILE.to_metadata(), "source": "override"},
            b"bytes",
            7,
            True,
            0.5,
        ],
    )
    def test_hostile_values_never_raise(self, mode: str, raw: object) -> None:
        resolved = resolve_run_cost_profile({"cost_profile": raw}, mode=mode, log_invalid=False)
        assert resolved.profile.legacy is True
        assert is_legacy_cost_config({"cost_profile": raw}) in (True, False)  # never raises
        assert is_leaderboard_cost_eligible({"cost_profile": raw}) in (True, False)

    def test_the_response_fields_of_a_tampered_row_read_as_invalid_legacy(self) -> None:
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        config = {"cost_profile": override.to_metadata()}
        for run_mode in ("backtest", "paper", "live", "optimize"):
            fields = cost_response_fields(config, run_mode)
            assert fields["cost_profile_invalid"] is True
            assert fields["legacy_costs"] is True
            assert fields["live_equivalent_costs"] is False
            summary = fields["cost_profile"]
            assert summary.id.startswith("legacy_") and summary.source == "legacy_inferred"

    def test_is_legacy_cost_config_treats_tampered_rows_as_legacy(self) -> None:
        override = DEFAULT_COST_PROFILE.with_overrides(slippage_bps=Decimal("7"))
        assert is_legacy_cost_config({"cost_profile": override.to_metadata()}) is True
        probe = _legacy_inferred("legacy_runtime_v0", legacy=False)
        assert is_legacy_cost_config(_cfg(probe)) is True
        assert is_legacy_cost_config(_cfg(_legacy_inferred("legacy_backtest_v0"))) is True
        assert is_legacy_cost_config(_cfg(_legacy_inferred("legacy_runtime_v0"))) is True


# ---------------------------------------------------------------------------
# WP21H-T-03 -- legitimate rows are unchanged
# ---------------------------------------------------------------------------


class TestT03LegitimateRowsAreUnchanged:
    @pytest.mark.parametrize("mode", _BACKTEST_MODES)
    @pytest.mark.parametrize("pid", sorted(COST_PROFILES))
    @pytest.mark.parametrize("fee_source", _FEE_SOURCES)
    def test_every_registry_row_resolves_to_itself_for_backtest_and_optimize(
        self, mode: str, pid: str, fee_source: str
    ) -> None:
        cfg = {"cost_profile": COST_PROFILES[pid].to_metadata(fee_source=fee_source)}  # type: ignore[arg-type]
        resolved = resolve_run_cost_profile(cfg, mode=mode)
        assert resolved.profile == COST_PROFILES[pid]
        assert (resolved.persisted, resolved.invalid, resolved.fee_source) == (
            True,
            False,
            fee_source,
        )

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    @pytest.mark.parametrize("pid", _LIVE_EQUIVALENT_IDS)
    @pytest.mark.parametrize("fee_source", _FEE_SOURCES)
    def test_every_live_equivalent_registry_row_resolves_to_itself_for_paper_and_live(
        self, mode: str, pid: str, fee_source: str
    ) -> None:
        cfg = {"cost_profile": COST_PROFILES[pid].to_metadata(fee_source=fee_source)}  # type: ignore[arg-type]
        resolved = resolve_run_cost_profile(cfg, mode=mode)
        assert resolved.profile == COST_PROFILES[pid]
        assert (resolved.persisted, resolved.invalid, resolved.fee_source) == (
            True,
            False,
            fee_source,
        )

    def test_the_registry_split_is_not_vacuous(self) -> None:
        """Both halves of the paper / live split are populated, and the default is
        in the accepted half (a default switch to a stress row would fail here)."""
        assert {"coinbase_advanced_1", "coinbase_advanced_d6"} <= set(_LIVE_EQUIVALENT_IDS)
        assert {
            "frictionless_diagnostic",
            "legacy_backtest_v0",
            "legacy_runtime_v0",
            "coinbase_stress_t80",
            "coinbase_stress_t100",
        } == set(_NOT_LIVE_EQUIVALENT_IDS)
        assert default_cost_profile().id in _LIVE_EQUIVALENT_IDS

    @pytest.mark.parametrize("mode", _ALL_MODES)
    @pytest.mark.parametrize("config", [None, {}, {"strategy_name": "x"}, {"cost_profile": None}])
    def test_absent_keys_are_legacy_by_absence(self, mode: str, config: Any) -> None:
        resolved = resolve_run_cost_profile(config, mode=mode)
        assert resolved.profile == _legacy_inferred(_family_legacy_id(mode))
        assert (resolved.persisted, resolved.invalid) == (False, False)

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_the_mode_familys_own_legacy_row_is_accepted_as_persisted(self, mode: str) -> None:
        """The one ``legacy_inferred`` row a writer produces (the auto-retry copy of a
        legacy paper run) still resolves as a persisted, valid profile."""
        row = _legacy_inferred(_family_legacy_id(mode))
        for fee_source in ("model", "exchange_fills"):
            cfg = {"cost_profile": row.to_metadata(fee_source=fee_source)}
            resolved = resolve_run_cost_profile(cfg, mode=mode)
            assert resolved.profile == row
            assert (resolved.persisted, resolved.invalid, resolved.fee_source) == (
                True,
                False,
                fee_source,
            )

    def test_the_resolved_legacy_profile_round_trips_through_the_auto_retry_copy(self) -> None:
        """T-23: ``legacy_inferred`` resolved from an absent key, persisted by the
        auto-retry (``to_metadata``) and read back, is the same profile."""
        absent = resolve_run_cost_profile({}, mode="paper").profile
        again = resolve_run_cost_profile({"cost_profile": absent.to_metadata()}, mode="paper")
        assert again.profile == absent and again.persisted and not again.invalid

    def test_from_metadata_is_still_the_schema_level_inverse(self) -> None:
        """An ``override`` profile still round-trips through the library API; only the
        reader of STORED rows applies the source policy."""
        custom = get_cost_profile("coinbase_advanced_d6").with_overrides(
            taker_fee_bps=Decimal("90")
        )
        assert CostProfile.from_metadata(custom.to_metadata()) == custom

    def test_the_leaderboard_cost_half_is_unchanged(self) -> None:
        assert is_leaderboard_cost_eligible({"cost_profile": DEFAULT_COST_PROFILE.to_metadata()})
        for pid in ("coinbase_stress_t80", "frictionless_diagnostic", "legacy_backtest_v0"):
            assert not is_leaderboard_cost_eligible(
                {"cost_profile": COST_PROFILES[pid].to_metadata()}
            )


# ---------------------------------------------------------------------------
# WP21H-T-04 / T-05 -- the readers are closed (source pins)
# ---------------------------------------------------------------------------


def _calls(path: Path | ast.Module, name: str) -> list[ast.Call]:
    tree = path if isinstance(path, ast.Module) else ast.parse(path.read_text())
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called = (
                f.id
                if isinstance(f, ast.Name)
                else f.attr
                if isinstance(f, ast.Attribute)
                else None
            )
            if called == name:
                out.append(node)
    return out


def _enclosing_function(tree: ast.Module, target: ast.AST) -> str | None:
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) and any(
            n is target for n in ast.walk(fn)
        ):
            return fn.name
    return None


class TestT04ReadersAreClosed:
    def test_only_from_metadata_and_parse_persisted_call_the_schema_parser(self) -> None:
        path = _REPO / "packages/trading/costs.py"
        tree = ast.parse(path.read_text())
        callers = {_enclosing_function(tree, c) for c in _calls(tree, "_parse_metadata")}
        assert callers == {"from_metadata", "_parse_persisted"}

    def test_the_stored_row_reader_uses_the_persisted_parser(self) -> None:
        path = _REPO / "packages/trading/costs.py"
        tree = ast.parse(path.read_text())
        callers = {_enclosing_function(tree, c) for c in _calls(tree, "_parse_persisted")}
        assert callers == {"resolve_run_cost_profile"}

    def test_is_legacy_cost_config_reads_through_the_resolver(self) -> None:
        path = _REPO / "packages/trading/costs.py"
        tree = ast.parse(path.read_text())
        callers = {_enclosing_function(tree, c) for c in _calls(tree, "resolve_run_cost_profile")}
        assert {"is_legacy_cost_config", "is_leaderboard_cost_eligible"} <= callers

    def test_no_production_module_outside_costs_reads_the_schema_parser(self) -> None:
        offenders: list[str] = []
        for root in ("apps", "packages", "scripts"):
            for path in sorted((_REPO / root).rglob("*.py")):
                if path == _REPO / "packages/trading/costs.py" or "__pycache__" in path.parts:
                    continue
                for name in ("_parse_metadata", "_parse_persisted", "from_metadata"):
                    offenders += [
                        f"{path.relative_to(_REPO)}:{c.lineno}:{name}" for c in _calls(path, name)
                    ]
        assert offenders == []

    def test_a_stored_override_or_foreign_legacy_row_has_no_writer(self) -> None:
        """Writers of ``config["cost_profile"]`` / ``run_metadata["cost_profile"]`` /
        ``ModelVersionORM.extra["cost_profile"]`` all go through ``to_metadata``; the
        profiles they are handed are registry profiles (``new_run_cost_profile``,
        ``default_cost_profile``) or a RESOLVED runtime profile.  ``with_overrides`` (the
        only ``override`` constructor) is called by library / script code that never
        imports the database layer."""
        override_callers: list[str] = []
        for root in ("apps", "packages", "scripts"):
            for path in sorted((_REPO / root).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                if _calls(path, "with_overrides"):
                    override_callers.append(str(path.relative_to(_REPO)))
        assert override_callers == [
            "packages/trading/costs_cli.py",
            "scripts/wp21_rebaseline.py",
        ]
        for rel in override_callers:
            src = (_REPO / rel).read_text()
            tree = ast.parse(src)
            imported: set[str] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imported.update(a.name for a in n.names)
                elif isinstance(n, ast.ImportFrom):
                    imported.add(n.module or "")
            assert not {m for m in imported if m.split(".")[0] in {"api", "sqlalchemy"}}, rel


class TestT05ResumeCallersUseARuntimeFamilyMode:
    def test_every_resolve_call_in_runs_py_names_a_runtime_mode(self) -> None:
        path = _REPO / "apps/api/routers/runs.py"
        calls = _calls(path, "resolve_run_cost_profile")
        assert calls, "the resume callers are gone -- re-point this pin"
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            assert "mode" in kw, f"runs.py:{call.lineno}"
            mode = kw["mode"]
            assert isinstance(mode, ast.Constant) and mode.value in _RUNTIME_MODES, (
                f"runs.py:{call.lineno} must resolve with mode='paper'|'live' so a tampered "
                "row falls back to legacy_runtime_v0"
            )


# ---------------------------------------------------------------------------
# WP21H-SEC-01 -- paper / live accept live-equivalent registry rows only
# ---------------------------------------------------------------------------

_PERSISTED_REJECTION = "persisted cost_profile source is not accepted"


class TestSec01RuntimeModesAcceptLiveEquivalentRegistryRowsOnly:
    """AC-1 / AC-2: a registry row that is not ``live_equivalent`` is a tampered row on a
    paper / live run (no writer stores one there), and reads as the legacy runtime
    profile, flagged invalid, with one warning and no value in the log."""

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    @pytest.mark.parametrize("pid", _NOT_LIVE_EQUIVALENT_IDS)
    @pytest.mark.parametrize("fee_source", _FEE_SOURCES)
    def test_a_registry_row_that_is_not_live_equivalent_reads_as_invalid_legacy(
        self, mode: str, pid: str, fee_source: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = {"cost_profile": COST_PROFILES[pid].to_metadata(fee_source=fee_source)}  # type: ignore[arg-type]
        with caplog.at_level(logging.DEBUG):
            resolved = resolve_run_cost_profile(cfg, mode=mode)
        assert resolved.profile == _LEGACY_RUNTIME
        assert (resolved.persisted, resolved.invalid, resolved.fee_source) == (
            False,
            True,
            "model",
        )
        records = [r for r in caplog.records if "cost_profile.invalid" in r.getMessage()]
        assert len(records) == 1 and mode in records[0].getMessage()
        assert pid not in caplog.text, "the stored id must not be echoed"
        fields = cost_response_fields(cfg, mode)
        assert fields["cost_profile_invalid"] is True
        assert fields["legacy_costs"] is True
        assert fields["live_equivalent_costs"] is False
        assert fields["cost_profile"].id == "legacy_runtime_v0"
        assert fields["cost_profile"].source == "legacy_inferred"

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    @pytest.mark.parametrize("pid", _NOT_LIVE_EQUIVALENT_IDS)
    def test_the_persisted_parser_rejects_it_with_a_fixed_message(
        self, mode: str, pid: str
    ) -> None:
        with pytest.raises(CostProfileError) as exc:
            costs_module._parse_persisted(COST_PROFILES[pid].to_metadata(), mode=mode)
        assert str(exc.value) == _PERSISTED_REJECTION

    @pytest.mark.parametrize("mode", _BACKTEST_MODES)
    @pytest.mark.parametrize("pid", _NOT_LIVE_EQUIVALENT_IDS)
    def test_backtest_and_optimize_still_accept_every_registry_row(
        self, mode: str, pid: str
    ) -> None:
        profile, _ = costs_module._parse_persisted(COST_PROFILES[pid].to_metadata(), mode=mode)
        assert profile == COST_PROFILES[pid]

    @pytest.mark.parametrize("pid", sorted(COST_PROFILES))
    def test_is_legacy_cost_config_is_unchanged_for_every_registry_row(self, pid: str) -> None:
        """CR-06: the mode-agnostic default reads every registry id, as before."""
        cfg = {"cost_profile": COST_PROFILES[pid].to_metadata()}
        assert is_legacy_cost_config(cfg) is COST_PROFILES[pid].legacy

    def test_is_legacy_cost_config_has_the_old_truth_table(self) -> None:
        assert is_legacy_cost_config(None) and is_legacy_cost_config("not a mapping")  # type: ignore[arg-type]
        assert is_legacy_cost_config({}) and is_legacy_cost_config({"cost_profile": None})
        assert is_legacy_cost_config({"cost_profile": {"junk": 1}})
        assert is_legacy_cost_config({"cost_profile": _LEGACY_RUNTIME.to_metadata()})
        assert is_legacy_cost_config(
            {"cost_profile": _legacy_inferred("legacy_backtest_v0").to_metadata()}
        )
        override = get_cost_profile("coinbase_advanced_d6").with_overrides(
            maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1")
        )
        assert is_legacy_cost_config({"cost_profile": override.to_metadata()})
        assert not is_legacy_cost_config({"cost_profile": DEFAULT_COST_PROFILE.to_metadata()})
        assert not is_legacy_cost_config(
            {"cost_profile": COST_PROFILES["frictionless_diagnostic"].to_metadata()}
        )

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    def test_a_runtime_run_reads_the_row_in_its_own_mode(self, mode: str) -> None:
        cfg = {"cost_profile": COST_PROFILES["frictionless_diagnostic"].to_metadata()}
        assert is_legacy_cost_config(cfg, mode=mode) is True
        assert is_legacy_cost_config(cfg, mode="backtest") is False


# ---------------------------------------------------------------------------
# WP21H-T-06 -- invariants (AC-6)
# ---------------------------------------------------------------------------


def _cheaper_than_legacy_runtime() -> dict[str, dict[str, Any]]:
    d6 = get_cost_profile("coinbase_advanced_d6")
    cheap = d6.with_overrides(maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1"))
    return {
        "override": cheap.to_metadata(),
        "legacy_inferred_cheaper": _legacy_inferred(
            "legacy_runtime_v0", maker_fee_bps=Decimal("0"), taker_fee_bps=Decimal("1")
        ).to_metadata(),
        "legacy_inferred_legacy_false": _legacy_inferred(
            "legacy_runtime_v0", legacy=False
        ).to_metadata(),
        "registry_numbers_edited": {**d6.to_metadata(), "taker_fee_bps": "1"},
    }


class TestT06Invariants:
    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    @pytest.mark.parametrize("pid", sorted(COST_PROFILES))
    @pytest.mark.parametrize("fee_source", _FEE_SOURCES)
    def test_the_runtime_basis_is_never_below_legacy_runtime(
        self, mode: str, pid: str, fee_source: str
    ) -> None:
        """Whatever registry row is stored, the paper / live basis is at least
        ``legacy_runtime_v0``'s (taker: 65 bps; maker: 45 bps)."""
        floor = COST_PROFILES["legacy_runtime_v0"]
        cfg = {"cost_profile": COST_PROFILES[pid].to_metadata(fee_source=fee_source)}  # type: ignore[arg-type]
        profile = resolve_run_cost_profile(cfg, mode=mode, log_invalid=False).profile
        assert profile.side_cost_rate("taker") >= floor.side_cost_rate("taker")
        assert profile.side_cost_rate("maker") >= floor.side_cost_rate("maker")

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    @pytest.mark.parametrize("label", sorted(_cheaper_than_legacy_runtime()))
    def test_the_runtime_basis_is_never_below_legacy_runtime_for_other_tampered_rows(
        self, mode: str, label: str
    ) -> None:
        floor = COST_PROFILES["legacy_runtime_v0"]
        cfg = {"cost_profile": _cheaper_than_legacy_runtime()[label]}
        resolved = resolve_run_cost_profile(cfg, mode=mode, log_invalid=False)
        assert resolved.invalid is True
        assert resolved.profile.side_cost_rate("taker") >= floor.side_cost_rate("taker")
        assert resolved.profile.side_cost_rate("maker") >= floor.side_cost_rate("maker")

    @pytest.mark.parametrize("mode", _RUNTIME_MODES)
    def test_every_profile_a_runtime_writer_produces_is_inside_the_reader_allow_list(
        self, mode: str
    ) -> None:
        """Writer set <= reader allow-list: ``new_run_cost_profile`` (create) for every
        registry id and for ``None``, plus the auto-retry's copy of a resolved legacy
        profile.  If a writer starts producing a row the reader rejects, a legitimate
        run becomes 'invalid' on resume -- this test fails first."""
        produced: list[CostProfile] = []
        for requested in (None, *sorted(COST_PROFILES)):
            try:
                produced.append(new_run_cost_profile(requested, mode=mode))
            except CostProfileError:
                continue  # the writer refuses this id for this mode: nothing is stored
        produced.append(resolve_run_cost_profile({}, mode=mode).profile)  # auto-retry copy
        for profile in produced:
            for fee_source in _FEE_SOURCES:
                meta = profile.to_metadata(fee_source=fee_source)  # type: ignore[arg-type]
                resolved = resolve_run_cost_profile({"cost_profile": meta}, mode=mode)
                assert resolved.profile == profile, profile.id
                assert (resolved.persisted, resolved.invalid) == (True, False), profile.id
        # non-vacuous: the writer set is what the contract says it is
        accepted_ids = {p.id for p in produced}
        expected = {default_cost_profile().id, "legacy_runtime_v0"}
        if mode == "paper":
            expected |= set(_LIVE_EQUIVALENT_IDS)
        assert accepted_ids == expected
