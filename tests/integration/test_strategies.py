"""
tests/integration/test_strategies.py
--------------------------------------
Integration tests for strategy discovery endpoints.

Endpoints under test
--------------------
GET /api/v1/strategies
    Returns StrategyListResponse with ``strategies`` list and ``total`` count.
    No database access required; reads from a static in-process registry.

GET /api/v1/strategies/{name}/schema
    Returns StrategyInfoResponse for the named strategy, or 404 if unknown.
    The router normalises hyphenated names to underscores before lookup.

Design notes
------------
- These tests are DB-free. The strategy registry is built at import time
  from concrete strategy classes in ``packages/trading/strategies/``.
- All three registered strategies (ma_crossover, rsi_mean_reversion, breakout)
  are verified by name, schema structure, and auth behaviour.
- JSON response keys are camelCase (alias_generator=to_camel in schemas.py):
    - ``display_name``     -> ``displayName``
    - ``parameter_schema`` -> ``parameterSchema``
    - ``strategies``       -> ``strategies``  (unchanged)
    - ``total``            -> ``total``        (unchanged)
- Auth tests use ``client_prod`` (auth required) and ``client_dev`` (open).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIST_URL = "/api/v1/strategies"
_SCHEMA_URL_TEMPLATE = "/api/v1/strategies/{name}/schema"

# All three strategies registered in _build_registry() at router import time.
_KNOWN_STRATEGY_NAMES = {"ma_crossover", "rsi_mean_reversion", "breakout"}

# Required top-level fields on every StrategyInfoResponse item.
_REQUIRED_STRATEGY_FIELDS = {
    "name",
    "displayName",
    "version",
    "description",
    "tags",
    "parameterSchema",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _schema_url(name: str) -> str:
    """Return the full schema endpoint URL for a given strategy name."""
    return _SCHEMA_URL_TEMPLATE.format(name=name)


# ===========================================================================
# GET /api/v1/strategies — list all strategies
# ===========================================================================


@pytest.mark.integration
class TestListStrategies:
    """Integration tests for GET /api/v1/strategies."""

    def test_list_returns_200(self, client_dev: TestClient) -> None:
        """GET /api/v1/strategies should return HTTP 200 OK."""
        resp = client_dev.get(_LIST_URL)
        assert resp.status_code == 200

    def test_list_response_has_strategies_and_total(
        self, client_dev: TestClient
    ) -> None:
        """Response body must contain a ``strategies`` list and a ``total`` integer."""
        resp = client_dev.get(_LIST_URL)
        data = resp.json()
        assert "strategies" in data, "response missing 'strategies' key"
        assert "total" in data, "response missing 'total' key"
        assert isinstance(data["strategies"], list), "'strategies' must be a list"
        assert isinstance(data["total"], int), "'total' must be an integer"

    def test_list_total_at_least_three(self, client_dev: TestClient) -> None:
        """Registry must expose at least the three baseline strategies."""
        resp = client_dev.get(_LIST_URL)
        data = resp.json()
        assert data["total"] >= 3, (
            f"Expected at least 3 registered strategies, got {data['total']}"
        )

    def test_list_total_matches_strategies_length(
        self, client_dev: TestClient
    ) -> None:
        """``total`` must exactly equal the length of the ``strategies`` list."""
        resp = client_dev.get(_LIST_URL)
        data = resp.json()
        assert data["total"] == len(data["strategies"]), (
            f"total={data['total']} does not match "
            f"len(strategies)={len(data['strategies'])}"
        )

    def test_each_strategy_has_required_fields(
        self, client_dev: TestClient
    ) -> None:
        """Every strategy entry must have all required StrategyInfoResponse fields.

        The schema uses ``alias_generator=to_camel``, so snake_case Python
        fields serialise to camelCase JSON keys.
        """
        resp = client_dev.get(_LIST_URL)
        strategies = resp.json()["strategies"]
        for entry in strategies:
            missing = _REQUIRED_STRATEGY_FIELDS - entry.keys()
            assert not missing, (
                f"Strategy entry {entry.get('name', '<unknown>')!r} "
                f"missing required fields: {missing}"
            )

    def test_list_contains_ma_crossover(self, client_dev: TestClient) -> None:
        """``ma_crossover`` must appear in the strategy list."""
        resp = client_dev.get(_LIST_URL)
        names = {s["name"] for s in resp.json()["strategies"]}
        assert "ma_crossover" in names, (
            f"'ma_crossover' not found in strategy names: {names}"
        )

    def test_list_contains_rsi_mean_reversion(
        self, client_dev: TestClient
    ) -> None:
        """``rsi_mean_reversion`` must appear in the strategy list."""
        resp = client_dev.get(_LIST_URL)
        names = {s["name"] for s in resp.json()["strategies"]}
        assert "rsi_mean_reversion" in names, (
            f"'rsi_mean_reversion' not found in strategy names: {names}"
        )

    def test_list_contains_breakout(self, client_dev: TestClient) -> None:
        """``breakout`` must appear in the strategy list."""
        resp = client_dev.get(_LIST_URL)
        names = {s["name"] for s in resp.json()["strategies"]}
        assert "breakout" in names, (
            f"'breakout' not found in strategy names: {names}"
        )

    def test_all_three_known_strategies_present(
        self, client_dev: TestClient
    ) -> None:
        """All three baseline strategy identifiers must be present in the list."""
        resp = client_dev.get(_LIST_URL)
        names = {s["name"] for s in resp.json()["strategies"]}
        missing = _KNOWN_STRATEGY_NAMES - names
        assert not missing, (
            f"The following expected strategies were absent from the list: {missing}"
        )

    def test_strategy_tags_are_list(self, client_dev: TestClient) -> None:
        """Every strategy's ``tags`` field must be a list (may be empty)."""
        resp = client_dev.get(_LIST_URL)
        for entry in resp.json()["strategies"]:
            assert isinstance(entry["tags"], list), (
                f"Strategy {entry['name']!r}: 'tags' must be a list, "
                f"got {type(entry['tags']).__name__}"
            )

    def test_strategy_parameter_schema_is_dict(
        self, client_dev: TestClient
    ) -> None:
        """Every strategy's ``parameterSchema`` field must be a dict."""
        resp = client_dev.get(_LIST_URL)
        for entry in resp.json()["strategies"]:
            assert isinstance(entry["parameterSchema"], dict), (
                f"Strategy {entry['name']!r}: 'parameterSchema' must be a dict"
            )


# ===========================================================================
# GET /api/v1/strategies/{name}/schema — individual schema lookup
# ===========================================================================


@pytest.mark.integration
class TestGetStrategySchema:
    """Integration tests for GET /api/v1/strategies/{name}/schema."""

    # --- 200 per registered strategy ---

    def test_ma_crossover_returns_200(self, client_dev: TestClient) -> None:
        """GET /api/v1/strategies/ma_crossover/schema should return 200."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        assert resp.status_code == 200

    def test_ma_crossover_name_field_correct(
        self, client_dev: TestClient
    ) -> None:
        """The ``name`` field in the response must equal 'ma_crossover'."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        assert resp.json()["name"] == "ma_crossover"

    def test_rsi_mean_reversion_returns_200(
        self, client_dev: TestClient
    ) -> None:
        """GET /api/v1/strategies/rsi_mean_reversion/schema should return 200."""
        resp = client_dev.get(_schema_url("rsi_mean_reversion"))
        assert resp.status_code == 200

    def test_rsi_mean_reversion_name_field_correct(
        self, client_dev: TestClient
    ) -> None:
        """The ``name`` field in the response must equal 'rsi_mean_reversion'."""
        resp = client_dev.get(_schema_url("rsi_mean_reversion"))
        assert resp.json()["name"] == "rsi_mean_reversion"

    def test_breakout_returns_200(self, client_dev: TestClient) -> None:
        """GET /api/v1/strategies/breakout/schema should return 200."""
        resp = client_dev.get(_schema_url("breakout"))
        assert resp.status_code == 200

    def test_breakout_name_field_correct(self, client_dev: TestClient) -> None:
        """The ``name`` field in the response must equal 'breakout'."""
        resp = client_dev.get(_schema_url("breakout"))
        assert resp.json()["name"] == "breakout"

    # --- JSON Schema structural validation ---

    def test_parameter_schema_has_properties_key(
        self, client_dev: TestClient
    ) -> None:
        """``parameterSchema`` must contain a ``properties`` key (valid JSON Schema)."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        schema = resp.json()["parameterSchema"]
        assert "properties" in schema, (
            "parameterSchema is missing the required 'properties' key. "
            f"Got keys: {list(schema.keys())}"
        )

    def test_parameter_schema_type_is_object(
        self, client_dev: TestClient
    ) -> None:
        """``parameterSchema`` must declare ``type: object`` (root JSON Schema type)."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        schema = resp.json()["parameterSchema"]
        assert schema.get("type") == "object", (
            f"Expected parameterSchema['type'] == 'object', "
            f"got {schema.get('type')!r}"
        )

    def test_parameter_schema_properties_is_dict(
        self, client_dev: TestClient
    ) -> None:
        """``parameterSchema.properties`` must be a dict mapping param names to sub-schemas."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        schema = resp.json()["parameterSchema"]
        assert isinstance(schema.get("properties"), dict), (
            "parameterSchema['properties'] must be a dict"
        )

    @pytest.mark.parametrize("name", sorted(_KNOWN_STRATEGY_NAMES))
    def test_all_strategies_have_valid_json_schema(
        self, client_dev: TestClient, name: str
    ) -> None:
        """All three strategy schemas must be valid JSON Schema objects with 'properties'."""
        resp = client_dev.get(_schema_url(name))
        assert resp.status_code == 200
        schema = resp.json()["parameterSchema"]
        assert isinstance(schema, dict), f"parameterSchema for {name!r} is not a dict"
        assert "properties" in schema, (
            f"parameterSchema for {name!r} missing 'properties' key"
        )
        assert schema.get("type") == "object", (
            f"parameterSchema for {name!r}: expected type='object', "
            f"got {schema.get('type')!r}"
        )

    # --- 404 for unknown strategy ---

    def test_unknown_strategy_returns_404(self, client_dev: TestClient) -> None:
        """A name not in the registry must return HTTP 404."""
        resp = client_dev.get(_schema_url("nonexistent_strategy_xyz"))
        assert resp.status_code == 404

    def test_404_response_contains_detail(self, client_dev: TestClient) -> None:
        """The 404 response body should include a ``detail`` field with information.

        Note: HTTPException does not use ErrorResponse serialisation.
        The 'code' field declared in ErrorResponse is absent from these 404 responses.
        """
        resp = client_dev.get(_schema_url("nonexistent_strategy_xyz"))
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data, "404 response body missing 'detail' key"
        assert isinstance(data["detail"], str), "'detail' must be a string"
        assert len(data["detail"]) > 0, "'detail' must not be empty"
        assert "code" not in data, (
            "Unexpected 'code' key in 404 — if router now uses ErrorResponse, "
            "update this test to assert code is a non-empty string."
        )

    # --- Name normalisation: hyphen to underscore ---

    def test_hyphenated_ma_crossover_resolves(
        self, client_dev: TestClient
    ) -> None:
        """'ma-crossover' (hyphenated) should resolve to 'ma_crossover' and return 200.

        The router normalises ``name.lower().replace('-', '_')`` before
        registry lookup, so hyphenated variants must succeed.
        """
        resp = client_dev.get(_schema_url("ma-crossover"))
        assert resp.status_code == 200

    def test_hyphenated_rsi_resolves(self, client_dev: TestClient) -> None:
        """'rsi-mean-reversion' (fully hyphenated) should resolve correctly."""
        resp = client_dev.get(_schema_url("rsi-mean-reversion"))
        assert resp.status_code == 200

    def test_hyphenated_name_returns_correct_strategy_name(
        self, client_dev: TestClient
    ) -> None:
        """Hyphenated name lookup must return the canonical underscore name field."""
        resp = client_dev.get(_schema_url("ma-crossover"))
        assert resp.status_code == 200
        assert resp.json()["name"] == "ma_crossover", (
            "Hyphenated lookup should return the canonical 'ma_crossover' name"
        )

    def test_uppercased_known_strategy_resolves(
        self, client_dev: TestClient
    ) -> None:
        """'MA_CROSSOVER' should resolve because the router calls name.lower() before lookup."""
        resp = client_dev.get(_schema_url("MA_CROSSOVER"))
        assert resp.status_code == 200
        assert resp.json()["name"] == "ma_crossover"

    def test_completely_invalid_strategy_name_returns_404(
        self, client_dev: TestClient
    ) -> None:
        """Any path segment that does not map to a registered strategy returns 404."""
        resp = client_dev.get(_schema_url("not_a_real_strategy_at_all"))
        assert resp.status_code == 404


# ===========================================================================
# Authentication — production mode (require_api_auth=True)
# ===========================================================================


@pytest.mark.integration
class TestStrategiesAuthentication:
    """Strategy endpoint behaviour under production-mode API key authentication."""

    def test_list_without_auth_returns_401_in_prod_mode(
        self, client_prod: TestClient
    ) -> None:
        """GET /api/v1/strategies without an API key must return 401 in prod mode."""
        resp = client_prod.get(_LIST_URL)
        assert resp.status_code == 401

    def test_list_with_valid_auth_returns_200_in_prod_mode(
        self, client_prod: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET /api/v1/strategies with a valid API key must return 200 in prod mode."""
        resp = client_prod.get(_LIST_URL, headers=auth_headers)
        assert resp.status_code == 200

    def test_schema_without_auth_returns_401_in_prod_mode(
        self, client_prod: TestClient
    ) -> None:
        """GET /api/v1/strategies/ma_crossover/schema without a key returns 401."""
        resp = client_prod.get(_schema_url("ma_crossover"))
        assert resp.status_code == 401

    def test_schema_with_valid_auth_returns_200_in_prod_mode(
        self, client_prod: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET /api/v1/strategies/ma_crossover/schema with a valid key returns 200."""
        resp = client_prod.get(_schema_url("ma_crossover"), headers=auth_headers)
        assert resp.status_code == 200

    def test_list_with_invalid_key_returns_401_in_prod_mode(
        self, client_prod: TestClient
    ) -> None:
        """GET /api/v1/strategies with a wrong API key must return 401."""
        resp = client_prod.get(
            _LIST_URL,
            headers={"X-API-Key": "completely-wrong-key-that-does-not-match"},
        )
        assert resp.status_code == 401

    def test_authenticated_list_total_matches_dev_mode(
        self, client_prod: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Authenticated prod-mode list returns same strategies as dev mode.

        The strategy registry is static (3 strategies defined at code level),
        so we assert the count is at least 3 using only the prod client.
        Using both client_dev and client_prod in the same test causes
        settings cache contamination (get_settings is lru_cached).
        """
        resp = client_prod.get(_LIST_URL, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] >= 3, (
            "Strategy count in prod mode should be at least 3 (ma_crossover, rsi, breakout)"
        )


# ===========================================================================
# Response content correctness — metadata spot-checks
# ===========================================================================


@pytest.mark.integration
class TestStrategyMetadataContent:
    """Spot-check that the actual metadata values from strategy classes are surfaced.

    These tests are intentionally lightweight — they confirm the registry
    wires up to the real class attributes, not that every field is perfect.
    The authoritative values come from the ``metadata: ClassVar[StrategyMetadata]``
    declarations in each strategy class.
    """

    def test_ma_crossover_display_name_non_empty(
        self, client_dev: TestClient
    ) -> None:
        """MA Crossover strategy must have a non-empty displayName."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        assert resp.json()["displayName"], "displayName must not be empty"

    def test_ma_crossover_version_is_semver_like(
        self, client_dev: TestClient
    ) -> None:
        """MA Crossover version should look like a semantic version (x.y.z)."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        version = resp.json()["version"]
        assert version, "version must not be empty"
        parts = version.split(".")
        assert len(parts) >= 2, (
            f"version {version!r} does not look like a semver string"
        )

    def test_ma_crossover_description_non_empty(
        self, client_dev: TestClient
    ) -> None:
        """MA Crossover strategy must have a non-empty description string."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        assert resp.json()["description"], "description must not be empty"

    def test_rsi_mean_reversion_has_tags(self, client_dev: TestClient) -> None:
        """RSI Mean Reversion strategy must declare at least one tag."""
        resp = client_dev.get(_schema_url("rsi_mean_reversion"))
        tags = resp.json()["tags"]
        assert isinstance(tags, list) and len(tags) > 0, (
            "rsi_mean_reversion must have at least one tag"
        )

    def test_breakout_has_tags(self, client_dev: TestClient) -> None:
        """Breakout strategy must declare at least one tag."""
        resp = client_dev.get(_schema_url("breakout"))
        tags = resp.json()["tags"]
        assert isinstance(tags, list) and len(tags) > 0, (
            "breakout must have at least one tag"
        )

    def test_ma_crossover_schema_has_fast_period_property(
        self, client_dev: TestClient
    ) -> None:
        """MA Crossover parameterSchema properties must include 'fast_period'."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        properties = resp.json()["parameterSchema"]["properties"]
        assert "fast_period" in properties, (
            f"Expected 'fast_period' in ma_crossover schema properties. "
            f"Got: {list(properties.keys())}"
        )

    def test_ma_crossover_schema_has_slow_period_property(
        self, client_dev: TestClient
    ) -> None:
        """MA Crossover parameterSchema properties must include 'slow_period'."""
        resp = client_dev.get(_schema_url("ma_crossover"))
        properties = resp.json()["parameterSchema"]["properties"]
        assert "slow_period" in properties, (
            f"Expected 'slow_period' in ma_crossover schema properties. "
            f"Got: {list(properties.keys())}"
        )

    def test_rsi_schema_has_rsi_period_property(
        self, client_dev: TestClient
    ) -> None:
        """RSI Mean Reversion parameterSchema must include 'rsi_period'."""
        resp = client_dev.get(_schema_url("rsi_mean_reversion"))
        properties = resp.json()["parameterSchema"]["properties"]
        assert "rsi_period" in properties, (
            f"Expected 'rsi_period' in rsi_mean_reversion schema properties. "
            f"Got: {list(properties.keys())}"
        )

    def test_breakout_schema_has_lookback_period_property(
        self, client_dev: TestClient
    ) -> None:
        """Breakout parameterSchema must include 'lookback_period'."""
        resp = client_dev.get(_schema_url("breakout"))
        properties = resp.json()["parameterSchema"]["properties"]
        assert "lookback_period" in properties, (
            f"Expected 'lookback_period' in breakout schema properties. "
            f"Got: {list(properties.keys())}"
        )
