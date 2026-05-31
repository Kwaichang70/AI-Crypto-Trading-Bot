"""
tests/unit/test_sprint51_cycle2_strategy_availability.py
---------------------------------------------------------
Sprint 51 Cycle 2 — strategy-availability lockdown.

Module under test
-----------------
packages/trading/strategy_availability.py
  - is_mode_allowed(strategy_name, mode) -> bool
  - get_availability(strategy_name) -> StrategyAvailability
  - StrategyStatus (StrEnum: active / demoted / experimental)
  - StrategyAvailability dataclass (allowed_modes, status, demotion_reason,
    promotion_requirements)

Positive lockdown coverage (TEST-S51C2-001 .. TEST-S51C2-0xx):
  1. is_mode_allowed truth table — 6 strategies x 3 modes exact matrix.
  2. Fail-closed default — unlisted name resolves to backtest-only / demoted.
  3. get_availability semantics — demoted have reason + requirements, active
     allow all 3 modes and carry no demotion reason.
  4. Normalization — is_mode_allowed does NOT normalize internally; the guards
     normalize at the call site. Documented and verified here.
  5. C4 keyset-consistency — strategies registry keys == runs registry keys ==
     availability registry keys (the 6), and ACTIVE/DEMOTED partition is
     disjoint + exhaustive over those 6.

All tests are pure, deterministic, no network, no DB.
"""

from __future__ import annotations

import pytest

from common.types import RunMode
from trading.strategy_availability import (
    StrategyAvailability,
    StrategyStatus,
    get_availability,
    is_mode_allowed,
)


# ---------------------------------------------------------------------------
# Reference matrices (the single hand-written expectation used for assertions)
# ---------------------------------------------------------------------------

# The strategies the system ships, partitioned by intended lockdown state.
_DEMOTED_STRATEGIES = {"ma_crossover", "breakout", "model_strategy"}
_ACTIVE_STRATEGIES = {"dca_rsi_hybrid", "grid_trading", "rsi_mean_reversion"}
# Sprint 51 Cycle 3: new strategies start EXPERIMENTAL (backtest-only) until
# they clear the walk-forward OOS profitability gate.
_EXPERIMENTAL_STRATEGIES = {"sl_tp_reversion"}
# Strategies restricted to backtest-only (demoted OR experimental).
_BACKTEST_ONLY_STRATEGIES = _DEMOTED_STRATEGIES | _EXPERIMENTAL_STRATEGIES
_ALL_STRATEGIES = _ACTIVE_STRATEGIES | _BACKTEST_ONLY_STRATEGIES

# Exact expected (strategy, mode) -> allowed boolean.
# Active -> all three modes True.  Backtest-only -> backtest True, paper/live False.
_TRUTH_TABLE: dict[tuple[str, RunMode], bool] = {}
for _name in _ACTIVE_STRATEGIES:
    _TRUTH_TABLE[(_name, RunMode.BACKTEST)] = True
    _TRUTH_TABLE[(_name, RunMode.PAPER)] = True
    _TRUTH_TABLE[(_name, RunMode.LIVE)] = True
for _name in _BACKTEST_ONLY_STRATEGIES:
    _TRUTH_TABLE[(_name, RunMode.BACKTEST)] = True
    _TRUTH_TABLE[(_name, RunMode.PAPER)] = False
    _TRUTH_TABLE[(_name, RunMode.LIVE)] = False


# ===========================================================================
# 1. is_mode_allowed truth table — TEST-S51C2-001 .. 018
# ===========================================================================


class TestIsModeAllowedTruthTable:
    """Exhaustive 6-strategy x 3-mode truth table for is_mode_allowed()."""

    @pytest.mark.parametrize(
        ("strategy_name", "mode", "expected"),
        [
            (name, mode, expected)
            for (name, mode), expected in _TRUTH_TABLE.items()
        ],
        ids=[
            f"{name}-{mode.value}"
            for (name, mode) in _TRUTH_TABLE
        ],
    )
    def test_mode_matrix(
        self, strategy_name: str, mode: RunMode, expected: bool
    ) -> None:
        """TEST-S51C2-001: each (strategy, mode) pair returns the exact expected bool."""
        assert is_mode_allowed(strategy_name, mode) is expected, (
            f"is_mode_allowed({strategy_name!r}, {mode}) expected {expected}"
        )

    def test_every_active_allows_all_three_modes(self) -> None:
        """TEST-S51C2-002: ACTIVE strategies allow backtest + paper + live."""
        for name in _ACTIVE_STRATEGIES:
            for mode in (RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE):
                assert is_mode_allowed(name, mode) is True

    def test_every_demoted_allows_only_backtest(self) -> None:
        """TEST-S51C2-003: backtest-only strategies allow backtest ONLY."""
        for name in _BACKTEST_ONLY_STRATEGIES:
            assert is_mode_allowed(name, RunMode.BACKTEST) is True
            assert is_mode_allowed(name, RunMode.PAPER) is False
            assert is_mode_allowed(name, RunMode.LIVE) is False


# ===========================================================================
# 2. Fail-closed default — TEST-S51C2-020 .. 023
# ===========================================================================


class TestFailClosedDefault:
    """Any unlisted strategy must resolve to backtest-only / demoted."""

    def test_unlisted_strategy_backtest_allowed(self) -> None:
        """TEST-S51C2-020: unlisted name allowed in backtest (fail-closed floor)."""
        assert is_mode_allowed("totally_made_up_strategy", RunMode.BACKTEST) is True

    def test_unlisted_strategy_paper_and_live_blocked(self) -> None:
        """TEST-S51C2-021: unlisted name blocked in paper AND live."""
        assert is_mode_allowed("totally_made_up_strategy", RunMode.PAPER) is False
        assert is_mode_allowed("totally_made_up_strategy", RunMode.LIVE) is False

    def test_unlisted_strategy_status_is_demoted(self) -> None:
        """TEST-S51C2-022: unlisted name resolves to DEMOTED status."""
        av = get_availability("totally_made_up_strategy")
        assert av.status is StrategyStatus.DEMOTED
        assert av.allowed_modes == frozenset({RunMode.BACKTEST})

    def test_empty_string_is_fail_closed(self) -> None:
        """TEST-S51C2-023: empty strategy name also fails closed to backtest-only."""
        assert is_mode_allowed("", RunMode.PAPER) is False
        assert is_mode_allowed("", RunMode.LIVE) is False
        assert is_mode_allowed("", RunMode.BACKTEST) is True


# ===========================================================================
# 3. get_availability semantics — TEST-S51C2-030 .. 034
# ===========================================================================


class TestGetAvailabilitySemantics:
    """Field-level guarantees of the StrategyAvailability records."""

    def test_returns_strategy_availability_instance(self) -> None:
        """TEST-S51C2-030: get_availability returns a StrategyAvailability."""
        assert isinstance(get_availability("grid_trading"), StrategyAvailability)

    def test_demoted_have_reason_and_requirements(self) -> None:
        """TEST-S51C2-031: every DEMOTED strategy has a non-empty reason + reqs."""
        for name in _DEMOTED_STRATEGIES:
            av = get_availability(name)
            assert av.status is StrategyStatus.DEMOTED
            assert av.demotion_reason.strip() != "", (
                f"{name} must carry a non-empty demotion_reason"
            )
            assert len(av.promotion_requirements) > 0, (
                f"{name} must carry at least one promotion requirement"
            )

    def test_active_have_no_reason_and_all_modes(self) -> None:
        """TEST-S51C2-032: every ACTIVE strategy has empty reason + all 3 modes."""
        for name in _ACTIVE_STRATEGIES:
            av = get_availability(name)
            assert av.status is StrategyStatus.ACTIVE
            assert av.demotion_reason == ""
            assert av.promotion_requirements == []
            assert av.allowed_modes == frozenset(
                {RunMode.BACKTEST, RunMode.PAPER, RunMode.LIVE}
            )

    def test_demoted_allowed_modes_is_backtest_only(self) -> None:
        """TEST-S51C2-033: DEMOTED allowed_modes is exactly {backtest}."""
        for name in _DEMOTED_STRATEGIES:
            assert get_availability(name).allowed_modes == frozenset(
                {RunMode.BACKTEST}
            )

    def test_availability_record_is_frozen(self) -> None:
        """TEST-S51C2-034: StrategyAvailability is immutable (frozen dataclass)."""
        av = get_availability("ma_crossover")
        with pytest.raises(Exception):
            av.status = StrategyStatus.ACTIVE  # type: ignore[misc]


# ===========================================================================
# 4. Normalization — TEST-S51C2-040 .. 042
# ===========================================================================


class TestNormalizationBehaviour:
    """
    is_mode_allowed does NOT normalize the name internally.

    The product guards (create_run, recover_orphaned_runs, promote_to_live)
    normalize with ``.lower().replace("-", "_")`` BEFORE calling
    is_mode_allowed.  We verify that contract here so the test suite documents
    where normalization happens (call site, not the registry function).
    The create_run / recovery integration tests exercise the end-to-end
    normalized path.
    """

    def test_non_normalized_name_fails_closed_in_paper(self) -> None:
        """TEST-S51C2-040: a non-normalized 'MA-Crossover' is not a registry key.

        Because is_mode_allowed does not normalize, the raw mixed-case/hyphen
        form is unknown -> fail-closed -> paper blocked.  (The product guards
        normalize first, so the *real* request path still resolves correctly.)
        """
        assert is_mode_allowed("MA-Crossover", RunMode.PAPER) is False
        # The hyphen lower-case form is likewise unknown to the raw function.
        assert is_mode_allowed("ma-crossover", RunMode.PAPER) is False

    def test_caller_normalized_name_resolves_correctly(self) -> None:
        """TEST-S51C2-041: after the caller normalizes, the demoted rule applies."""
        normalized = "MA-Crossover".lower().replace("-", "_")
        assert normalized == "ma_crossover"
        assert is_mode_allowed(normalized, RunMode.PAPER) is False
        assert is_mode_allowed(normalized, RunMode.BACKTEST) is True

    def test_caller_normalized_active_name_resolves_correctly(self) -> None:
        """TEST-S51C2-042: normalized active name allows paper/live."""
        normalized = "Grid-Trading".lower().replace("-", "_")
        assert normalized == "grid_trading"
        assert is_mode_allowed(normalized, RunMode.PAPER) is True
        assert is_mode_allowed(normalized, RunMode.LIVE) is True


# ===========================================================================
# 5. C4 keyset-consistency — TEST-S51C2-050 .. 054
# ===========================================================================


class TestKeysetConsistency:
    """The three registries must agree on the exact same 6 strategy keys."""

    def _availability_keys(self) -> set[str]:
        import trading.strategy_availability as sa

        return set(sa._AVAILABILITY.keys())

    def _runs_registry_keys(self) -> set[str]:
        import api.routers.runs as runs_module

        # Reset the lazy singleton so we read the true registry, then restore.
        original = runs_module._STRATEGY_REGISTRY
        runs_module._STRATEGY_REGISTRY = None
        try:
            keys = set(runs_module._get_strategy_registry().keys())
        finally:
            runs_module._STRATEGY_REGISTRY = original
        return keys

    def _strategies_registry_keys(self) -> set[str]:
        import api.routers.strategies as strategies_module

        original = strategies_module._REGISTRY
        strategies_module._REGISTRY = None
        try:
            keys = set(strategies_module._get_registry().keys())
        finally:
            strategies_module._REGISTRY = original
        return keys

    def test_availability_registry_has_exactly_six(self) -> None:
        """TEST-S51C2-050: the availability registry holds exactly the shipped strategies."""
        assert self._availability_keys() == _ALL_STRATEGIES

    def test_runs_registry_matches_availability(self) -> None:
        """TEST-S51C2-051: runs.py registry keys == availability keys."""
        assert self._runs_registry_keys() == self._availability_keys()

    def test_strategies_registry_matches_availability(self) -> None:
        """TEST-S51C2-052: strategies.py registry keys == availability keys."""
        assert self._strategies_registry_keys() == self._availability_keys()

    def test_all_three_registries_agree(self) -> None:
        """TEST-S51C2-053: strategies == runs == availability (transitively)."""
        avail = self._availability_keys()
        runs = self._runs_registry_keys()
        strategies = self._strategies_registry_keys()
        assert avail == runs == strategies == _ALL_STRATEGIES

    def test_active_demoted_partition_is_disjoint_and_exhaustive(self) -> None:
        """TEST-S51C2-054: status partition is disjoint + covers every strategy."""
        # Pairwise disjoint
        assert _ACTIVE_STRATEGIES.isdisjoint(_DEMOTED_STRATEGIES)
        assert _ACTIVE_STRATEGIES.isdisjoint(_EXPERIMENTAL_STRATEGIES)
        assert _DEMOTED_STRATEGIES.isdisjoint(_EXPERIMENTAL_STRATEGIES)
        # Exhaustive over the availability registry
        assert _ALL_STRATEGIES == self._availability_keys()
        # And the recorded status agrees with the partition for each key.
        for name in _ACTIVE_STRATEGIES:
            assert get_availability(name).status is StrategyStatus.ACTIVE
        for name in _DEMOTED_STRATEGIES:
            assert get_availability(name).status is StrategyStatus.DEMOTED
        for name in _EXPERIMENTAL_STRATEGIES:
            assert get_availability(name).status is StrategyStatus.EXPERIMENTAL
