"""
tests/unit/test_wp13a_promote_strategy_params.py
--------------------------------------------------
WP13a-C-R2-09 (critic round 2, Low): a dedicated, positive unit test for
``apps.api.routers.runs._build_promoted_strategy_params`` -- the helper
extracted from ``promote_to_live`` for WP13a-S-07 (security round 2).

Module under test
------------------
apps/api/routers/runs.py :: _build_promoted_strategy_params

Prior coverage only exercised this indirectly via regression suites that
assert no *breakage*; this file asserts the actual fixed *behaviour*
directly: the persisted ``trailing_stop_pct`` is the NORMALISED verdict
value (not the raw source value), and the source run's own
``strategy_params`` dict is never mutated in place and never aliased.
"""

from __future__ import annotations

from api.routers.runs import _build_promoted_strategy_params


class TestBuildPromotedStrategyParamsWP13aCR209:
    def test_trailing_stop_pct_normalised_value_is_persisted(self) -> None:
        """The source has the raw string ``"0.02"``; the verdict's
        NORMALISED value (a float) must be what ends up in the returned
        dict, not the raw source string."""
        source_strategy_params = {"lookback": 20, "trailing_stop_pct": "0.02"}
        normalised_trailing_stop_pct = 0.02  # what ExitConfigVerdict.config.trailing_stop_pct holds

        result = _build_promoted_strategy_params(
            source_strategy_params, normalised_trailing_stop_pct
        )

        assert result["trailing_stop_pct"] == normalised_trailing_stop_pct
        assert result["trailing_stop_pct"] is normalised_trailing_stop_pct
        assert isinstance(result["trailing_stop_pct"], float)

    def test_source_strategy_params_not_mutated_and_not_same_object(self) -> None:
        """The returned dict must be a fresh copy: mutating it (or its
        trailing_stop_pct key) must never be observable on the source
        run's own ``strategy_params`` dict, which is still attached to the
        live DB session for the SOURCE (unpromoted) run."""
        source_strategy_params = {"lookback": 20, "trailing_stop_pct": "0.02"}
        source_snapshot = dict(source_strategy_params)

        result = _build_promoted_strategy_params(source_strategy_params, 0.05)

        assert result is not source_strategy_params
        assert source_strategy_params == source_snapshot
        assert source_strategy_params["trailing_stop_pct"] == "0.02"
        assert result["trailing_stop_pct"] == 0.05

    def test_unset_trailing_stop_pct_is_popped_not_left_stale(self) -> None:
        """When the verdict resolves trailing_stop_pct to None (unset), the
        stale raw source value must be removed, not silently left in
        place."""
        source_strategy_params = {"lookback": 20, "trailing_stop_pct": "0.02"}

        result = _build_promoted_strategy_params(source_strategy_params, None)

        assert "trailing_stop_pct" not in result
        # Source is untouched even though the key was "removed" for the
        # promoted copy.
        assert source_strategy_params["trailing_stop_pct"] == "0.02"

    def test_none_source_strategy_params_returns_fresh_empty_dict(self) -> None:
        """A source run with no strategy_params at all (``None``) must not
        raise, and must still return an independent, mutable dict."""
        result = _build_promoted_strategy_params(None, 0.03)

        assert result == {"trailing_stop_pct": 0.03}
        result["extra"] = "safe-to-mutate"
        assert "extra" not in _build_promoted_strategy_params(None, 0.03)
