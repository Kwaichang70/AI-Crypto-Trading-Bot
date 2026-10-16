"""
tests/integration/test_wp21b_optimize_api.py
----------------------------------------------
WP2.1b API coverage for the cost model of an OPTIMIZATION run (hermetic: a mocked
``AsyncSession``, no Postgres, no network, no exchange).

    WP21-T-34 (API half)
        * POST /optimize writes ``run_metadata = {"cost_profile": ..., "engine_semantics": ...}``
        * GET /optimize/{id} and GET /optimize return ``costProfile`` (+ the SY-21-10 flags)
        * NULL / missing key / malformed ``run_metadata`` -> legacy, HTTP 200, never a 500 (AC-05)
    SY-21-03
        * the optimize request rules are unchanged: any registry id; unknown id -> 422 and
          NOTHING is persisted
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.db.models import OptimizationEntryORM, OptimizationRunORM
from api.services.cost_profiles import engine_semantics
from common.types import TimeFrame
from tests.conftest import make_bars
from trading.backtest import BacktestRunner
from trading.costs import DEFAULT_COST_PROFILE, get_cost_profile

_URL = "/api/v1/optimize"
_SYMBOL = "BTC/USD"
_BARS = {_SYMBOL: make_bars(300, symbol=_SYMBOL, timeframe=TimeFrame.ONE_HOUR)}
_BODY: dict[str, Any] = {
    "strategyName": "ma_crossover",
    "paramGrid": {"fast_period": [5, 10], "slow_period": [20, 30]},
    "symbols": [_SYMBOL],
    "timeframe": "1h",
    "backtestStart": "2024-01-01T00:00:00Z",
    "backtestEnd": "2024-06-01T00:00:00Z",
    "topN": 3,
    "maxCombinations": 10,
}
_COST_KEYS = {
    "costProfile",
    "legacyCosts",
    "costProfileInvalid",
    "legacyExitSemantics",
    "liveEquivalentCosts",
}
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Db:
    """A mocked ``AsyncSession`` whose reads are scripted per test."""

    def __init__(self) -> None:
        self.session = AsyncMock()
        self.session.add = MagicMock()
        self.session.flush = AsyncMock()
        self.session.commit = AsyncMock()
        self.session.rollback = AsyncMock()
        self.set_rows([])

    def set_rows(self, rows: list[Any]) -> None:
        """Rows served to ``GET /optimize`` (``.scalars().all()``) and ``GET /{id}``
        (``.scalar_one_or_none()``, the first row)."""
        scalars = MagicMock()
        scalars.all.return_value = rows
        result = MagicMock()
        result.scalars.return_value = scalars
        result.scalar_one_or_none.return_value = rows[0] if rows else None
        self.session.execute = AsyncMock(return_value=result)

    def added(self, kind: type) -> list[Any]:
        return [c.args[0] for c in self.session.add.call_args_list if isinstance(c.args[0], kind)]


@pytest.fixture()
def db(app_dev_mode: Any) -> Generator[_Db, None, None]:
    from api.db.session import get_db

    holder = _Db()

    async def _override() -> AsyncGenerator[AsyncMock, None]:
        yield holder.session

    app_dev_mode.dependency_overrides[get_db] = _override
    yield holder
    app_dev_mode.dependency_overrides.pop(get_db, None)


def _post(client: TestClient, **overrides: Any) -> Any:
    with patch("api.routers.optimize._fetch_bars_for_backtest", new=AsyncMock(return_value=_BARS)):
        return client.post(_URL, json={**_BODY, **overrides})


def _row(run_metadata: Any, *, n: int = 0) -> OptimizationRunORM:
    """A transient ORM row, as the SELECT would materialise it."""
    row = OptimizationRunORM(
        id=uuid.UUID(int=0x2100 + n),
        strategy_name="ma_crossover",
        symbols=[_SYMBOL],
        timeframe="1h",
        rank_by="sharpe_ratio",
        total_combinations=4,
        completed_combinations=4,
        failed_combinations=0,
        elapsed_seconds=1.5,
        created_at=_NOW,
    )
    # ``run_metadata`` is assigned only when given: a row from a pre-020 table / a NULL column
    # simply reads ``None``.
    row.run_metadata = run_metadata
    return row


def _stamped(profile_id: str) -> dict[str, Any]:
    profile = get_cost_profile(profile_id)
    return {
        "cost_profile": profile.to_metadata(),
        "engine_semantics": engine_semantics("optimize"),
    }


def _cost_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {k: body[k] for k in _COST_KEYS}


# ---------------------------------------------------------------------------
# POST: run_metadata is written (WP21-T-34)
# ---------------------------------------------------------------------------


class TestPostWritesRunMetadata:
    def test_the_default_profile_is_persisted_with_the_engine_semantics(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        resp = _post(client_dev)
        assert resp.status_code == 200, resp.text
        (run,) = db.added(OptimizationRunORM)
        assert run.run_metadata == {
            "cost_profile": DEFAULT_COST_PROFILE.to_metadata(),
            "engine_semantics": engine_semantics("optimize"),
        }
        assert run.run_metadata["cost_profile"]["fee_source"] == "model"
        # SY-21-11 / WP2.1c: optimization metadata carries the backtest breaker flag, now true
        assert run.run_metadata["engine_semantics"]["backtest_circuit_breaker"] is True
        assert str(run.id) == resp.json()["optimizationRunId"]
        db.session.commit.assert_awaited_once()

    def test_every_combination_runs_with_the_breaker_it_persists_wp21_t36(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        """WP2.1c (SY-21-13, GAP-05): ``run_metadata`` says what the optimizer ran with."""
        seen: list[bool] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw["enable_circuit_breaker"])
            original_init(self, *args, **kw)

        with patch.object(BacktestRunner, "__init__", _spy):
            resp = _post(client_dev)
        assert resp.status_code == 200, resp.text
        (run,) = db.added(OptimizationRunORM)
        assert run.run_metadata["engine_semantics"]["backtest_circuit_breaker"] is True
        assert seen == [True] * 4  # 2 x 2 combinations

    def test_there_is_no_api_toggle_for_the_optimizer_breaker_wp21_t36(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        seen: list[bool] = []
        original_init = BacktestRunner.__init__

        def _spy(self: BacktestRunner, *args: Any, **kw: Any) -> None:
            seen.append(kw["enable_circuit_breaker"])
            original_init(self, *args, **kw)

        with patch.object(BacktestRunner, "__init__", _spy):
            resp = _post(client_dev, enableCircuitBreaker=False)
        assert resp.status_code in (200, 422), resp.text
        assert set(seen) <= {True}
        if resp.status_code == 200:
            (run,) = db.added(OptimizationRunORM)
            assert run.run_metadata["engine_semantics"]["backtest_circuit_breaker"] is True

    @pytest.mark.parametrize(
        "profile_id",
        [
            "coinbase_advanced_d6",
            "coinbase_stress_t80",
            "frictionless_diagnostic",
            "legacy_backtest_v0",
            "legacy_runtime_v0",
        ],
    )
    def test_the_requested_registry_id_is_what_is_persisted(
        self, client_dev: TestClient, db: _Db, profile_id: str
    ) -> None:
        resp = _post(client_dev, costProfile=profile_id)
        assert resp.status_code == 200, resp.text
        (run,) = db.added(OptimizationRunORM)
        assert run.run_metadata["cost_profile"] == get_cost_profile(profile_id).to_metadata()
        assert resp.json()["costProfile"]["id"] == profile_id

    def test_the_persisted_fingerprint_is_the_one_the_optimizer_simulated_with(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        resp = _post(client_dev, costProfile="coinbase_stress_t100")
        (run,) = db.added(OptimizationRunORM)
        stress = get_cost_profile("coinbase_stress_t100")
        assert run.run_metadata["cost_profile"]["fingerprint"] == stress.fingerprint()
        assert resp.json()["costProfile"]["fingerprint"] == stress.fingerprint()

    def test_entries_are_still_persisted_next_to_the_run(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        resp = _post(client_dev)
        assert resp.status_code == 200, resp.text
        assert len(db.added(OptimizationEntryORM)) == len(resp.json()["entries"]) > 0

    @pytest.mark.parametrize("bad_id", ["nope", "", "COINBASE_ADVANCED_1", "custom"])
    def test_an_unknown_profile_is_a_422_and_persists_nothing(
        self, client_dev: TestClient, db: _Db, bad_id: str
    ) -> None:
        resp = _post(client_dev, costProfile=bad_id)
        assert resp.status_code == 422
        assert resp.json()["detail"] == {"code": "unknown_cost_profile"}
        assert db.added(OptimizationRunORM) == []
        db.session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST response == what a later GET derives from the persisted row
# ---------------------------------------------------------------------------


class TestPostAndGetShareOneShape:
    @pytest.mark.parametrize(
        "profile_id",
        [None, "coinbase_advanced_d6", "coinbase_stress_t80", "legacy_backtest_v0"],
    )
    def test_post_get_detail_and_list_agree_on_every_cost_field(
        self, client_dev: TestClient, db: _Db, profile_id: str | None
    ) -> None:
        overrides = {} if profile_id is None else {"costProfile": profile_id}
        posted = _post(client_dev, **overrides)
        assert posted.status_code == 200, posted.text
        (run,) = db.added(OptimizationRunORM)
        run.created_at = _NOW  # the server default the mocked flush does not apply
        db.set_rows([run])

        detail = client_dev.get(f"{_URL}/{run.id}")
        listed = client_dev.get(_URL)
        assert detail.status_code == 200 and listed.status_code == 200, (
            detail.text,
            listed.text,
        )
        expected = _cost_fields(posted.json())
        assert _cost_fields(detail.json()) == expected
        assert _cost_fields(listed.json()[0]) == expected
        assert expected["costProfile"]["id"] == (profile_id or DEFAULT_COST_PROFILE.id)


# ---------------------------------------------------------------------------
# GET: costProfile + SY-21-10 flags from the persisted run_metadata
# ---------------------------------------------------------------------------


class TestGetReturnsTheCostProfile:
    @pytest.mark.parametrize(
        ("profile_id", "legacy", "live_equivalent", "stress", "diagnostic"),
        [
            ("coinbase_advanced_1", False, True, False, False),
            ("coinbase_advanced_d6", False, True, False, False),
            ("coinbase_stress_t80", False, False, True, False),
            ("frictionless_diagnostic", False, False, False, True),
            ("legacy_backtest_v0", True, False, False, False),
        ],
    )
    def test_detail_and_list_expose_the_profile_and_the_flags(
        self,
        client_dev: TestClient,
        db: _Db,
        profile_id: str,
        legacy: bool,
        live_equivalent: bool,
        stress: bool,
        diagnostic: bool,
    ) -> None:
        row = _row(_stamped(profile_id))
        db.set_rows([row])
        for body in (client_dev.get(f"{_URL}/{row.id}").json(), client_dev.get(_URL).json()[0]):
            profile = body["costProfile"]
            assert profile["id"] == profile_id
            assert profile["source"] == "registry"
            assert profile["feeSource"] == "model"
            assert (profile["legacy"], profile["stress"], profile["diagnostic"]) == (
                legacy,
                stress,
                diagnostic,
            )
            assert profile["liveEquivalent"] is live_equivalent
            assert {"makerFeeBps", "takerFeeBps", "slippageBps", "roundTripBps"} <= set(profile)
            assert body["legacyCosts"] is legacy
            assert body["costProfileInvalid"] is False
            assert body["legacyExitSemantics"] is False
            assert body["liveEquivalentCosts"] is live_equivalent

    def test_the_default_profile_numbers_on_the_wire(self, client_dev: TestClient, db: _Db) -> None:
        row = _row(_stamped("coinbase_advanced_1"))
        db.set_rows([row])
        profile = client_dev.get(f"{_URL}/{row.id}").json()["costProfile"]
        assert (profile["makerFeeBps"], profile["takerFeeBps"], profile["slippageBps"]) == (
            "60",
            "120",
            "10",
        )
        assert profile["roundTripBps"] == "260"
        assert profile["fingerprint"] == DEFAULT_COST_PROFILE.fingerprint()

    def test_the_list_is_one_query_and_keeps_its_existing_fields(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        rows = [_row(_stamped("coinbase_advanced_1"), n=1), _row(None, n=2)]
        db.set_rows(rows)
        resp = client_dev.get(_URL)
        assert resp.status_code == 200
        assert db.session.execute.await_count == 1  # no N+1: run_metadata is a plain column
        body = resp.json()
        assert [b["optimizationRunId"] for b in body] == [str(r.id) for r in rows]
        assert {
            "strategyName",
            "symbols",
            "timeframe",
            "rankBy",
            "totalCombinations",
            "completedCombinations",
            "failedCombinations",
            "elapsedSeconds",
            "createdAt",
        } <= set(body[0])
        assert body[0]["legacyCosts"] is False and body[1]["legacyCosts"] is True


# ---------------------------------------------------------------------------
# NULL / missing / malformed run_metadata: legacy, never a 500 (AC-05)
# ---------------------------------------------------------------------------

_LEGACY_METADATA = [
    pytest.param(None, False, id="null"),
    pytest.param({}, False, id="empty-object"),
    pytest.param({"engine_semantics": {"schema": 1}}, False, id="cost-key-missing"),
    pytest.param({"cost_profile": None}, False, id="cost-key-null"),
    pytest.param({"cost_profile": "junk"}, True, id="string"),
    pytest.param({"cost_profile": 7}, True, id="int"),
    pytest.param({"cost_profile": []}, True, id="list"),
    pytest.param({"cost_profile": {}}, True, id="empty-profile"),
    pytest.param({"cost_profile": {"junk": 1}}, True, id="unknown-key"),
    pytest.param({"cost_profile": {"id": "coinbase_advanced_1"}}, True, id="id-only"),
    pytest.param("junk", False, id="metadata-is-a-string"),
    pytest.param(5, False, id="metadata-is-an-int"),
    pytest.param([], False, id="metadata-is-a-list"),
    pytest.param([{"cost_profile": 1}], False, id="metadata-is-a-list-of-objects"),
]


class TestLegacyAndMalformedMetadataNeverRaise:
    @pytest.mark.parametrize(("metadata", "invalid"), _LEGACY_METADATA)
    def test_detail_is_200_and_legacy(
        self, client_dev: TestClient, db: _Db, metadata: Any, invalid: bool
    ) -> None:
        row = _row(metadata)
        db.set_rows([row])
        resp = client_dev.get(f"{_URL}/{row.id}")
        assert resp.status_code == 200, resp.text
        self._assert_legacy(resp.json(), invalid)

    @pytest.mark.parametrize(("metadata", "invalid"), _LEGACY_METADATA)
    def test_list_is_200_and_legacy(
        self, client_dev: TestClient, db: _Db, metadata: Any, invalid: bool
    ) -> None:
        db.set_rows([_row(metadata, n=1), _row(_stamped("coinbase_advanced_1"), n=2)])
        resp = client_dev.get(_URL)
        assert resp.status_code == 200, resp.text
        bad, good = resp.json()
        self._assert_legacy(bad, invalid)
        # one bad row never contaminates its neighbour
        assert good["costProfile"]["id"] == "coinbase_advanced_1"
        assert good["legacyCosts"] is False and good["liveEquivalentCosts"] is True

    @staticmethod
    def _assert_legacy(body: dict[str, Any], invalid: bool) -> None:
        profile = body["costProfile"]
        assert profile is not None
        assert (profile["id"], profile["source"]) == ("legacy_backtest_v0", "legacy_inferred")
        assert profile["legacy"] is True and profile["liveEquivalent"] is False
        assert body["legacyCosts"] is True
        assert body["costProfileInvalid"] is invalid
        assert body["liveEquivalentCosts"] is False

    def test_a_missing_engine_semantics_block_only_flags_the_exit_semantics(
        self, client_dev: TestClient, db: _Db
    ) -> None:
        metadata = {"cost_profile": DEFAULT_COST_PROFILE.to_metadata()}
        row = _row(metadata)
        db.set_rows([row])
        body = client_dev.get(f"{_URL}/{row.id}").json()
        assert body["legacyExitSemantics"] is True
        assert body["legacyCosts"] is False and body["costProfile"]["id"] == "coinbase_advanced_1"

    def test_an_unknown_run_is_still_a_404(self, client_dev: TestClient, db: _Db) -> None:
        db.set_rows([])
        resp = client_dev.get(f"{_URL}/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_a_malformed_value_is_never_echoed(self, client_dev: TestClient, db: _Db) -> None:
        marker = "SENTINEL-do-not-echo"
        row = _row({"cost_profile": {"id": marker, "tier": marker}})
        db.set_rows([row])
        resp = client_dev.get(f"{_URL}/{row.id}")
        assert resp.status_code == 200
        assert marker not in resp.text
