"""CF-97-37 patch 37-A: shadow auth check, production-only switches, peer/client logs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from fastapi import Request
from fastapi.testclient import TestClient

from api.auth import _AUTH_SHADOW
from api.config import get_settings
from common.metrics import metrics as metrics_collector
from tests._cf36_logging import SENTINEL, ConfigureFn

RAW = f"{SENTINEL}-dummy-api-key-0001"
RAW_HASH = hashlib.sha256(RAW.encode()).hexdigest()
OTHER = f"{SENTINEL}-dummy-other-key-0002"
KEYED = "/api/v1/runs"


def _count(result: str) -> float:
    return float(_AUTH_SHADOW.labels(result=result)._value.get())


def _counts() -> dict[str, float]:
    return {r: _count(r) for r in ("ok", "reject", "missing")}


def _client(monkeypatch: pytest.MonkeyPatch, **env: str) -> Iterator[TestClient]:
    base = {
        "PROMETHEUS_ENABLED": "false",
        "DATABASE_URL": "postgresql+asyncpg://x:x@localhost:5432/x",
        "RATE_LIMIT_ENABLED": "false",
    }
    for k in ("ENVIRONMENT", "REQUIRE_API_AUTH", "API_KEY_HASH", "TRUSTED_PROXY_COUNT", "DEBUG"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    from api.main import create_app

    # No ``with``: the lifespan (and its Postgres) is not started.
    yield TestClient(create_app(), raise_server_exceptions=False)
    get_settings.cache_clear()


@pytest.fixture()
def shadow(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(monkeypatch, REQUIRE_API_AUTH="false", API_KEY_HASH=RAW_HASH)


@pytest.fixture()
def default_dev(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(monkeypatch)


@pytest.fixture()
def prod(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(
        monkeypatch,
        ENVIRONMENT="production",
        REQUIRE_API_AUTH="true",
        API_KEY_HASH=RAW_HASH,
        RATE_LIMIT_ENABLED="true",
    )


@pytest.fixture()
def dev_auth_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(
        monkeypatch, ENVIRONMENT="development", REQUIRE_API_AUTH="true", API_KEY_HASH=RAW_HASH
    )


# -- shadow check ------------------------------------------------------------


def test_shadow_counts_ok_reject_missing_and_never_refuses(shadow: TestClient) -> None:
    before = _counts()
    for headers in ({"X-API-Key": RAW}, {"X-API-Key": OTHER}, {}):
        assert shadow.get(KEYED, headers=headers).status_code != 401
    after = _counts()
    assert {r: after[r] - before[r] for r in after} == {"ok": 1, "reject": 1, "missing": 1}


def test_shadow_does_not_count_unkeyed_routes(shadow: TestClient) -> None:
    before = _counts()
    # /health needs the lifespan's app.state, which these tests do not start.
    assert shadow.get("/api/v1/metrics").status_code == 200
    assert shadow.get("/api/v1/metrics", headers={"X-API-Key": OTHER}).status_code == 200
    assert _counts() == before


def test_shadow_is_silent_when_no_hash_is_configured(default_dev: TestClient) -> None:
    before = _counts()
    assert default_dev.get(KEYED).status_code != 401
    assert default_dev.get(KEYED, headers={"X-API-Key": OTHER}).status_code != 401
    assert _counts() == before


def test_shadow_does_not_run_while_auth_is_on(prod: TestClient) -> None:
    before = _counts()
    assert prod.get(KEYED).status_code == 401
    assert prod.get(KEYED, headers={"X-API-Key": RAW}).status_code != 401
    assert _counts() == before


def test_shadow_series_are_not_in_metrics_get_all(shadow: TestClient) -> None:
    shadow.get(KEYED, headers={"X-API-Key": OTHER})
    assert "api_auth_shadow" not in repr(metrics_collector.get_all())


def test_shadow_series_are_exported_at_zero_on_the_prometheus_registry() -> None:
    from prometheus_client import generate_latest

    from api.prometheus import PROMETHEUS_REGISTRY

    text = generate_latest(PROMETHEUS_REGISTRY).decode()
    for r in ("ok", "reject", "missing"):
        assert f'api_auth_shadow_total{{result="{r}"}}' in text


def test_shadow_fault_is_contained_counted_nowhere_and_logged_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR-37A-02: call ``_shadow_check`` directly; a fault must not escape."""
    import api.auth as auth
    from api.config import Settings

    def boom(*_a: Any, **_k: Any) -> bool:
        raise RuntimeError(f"shadow fault {OTHER}")

    monkeypatch.setattr(auth, "verify_api_key", boom)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": KEYED,
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
        }
    )
    settings = Settings(_env_file=None, api_key_hash=RAW_HASH)
    before = _counts()
    with structlog.testing.capture_logs() as logs:
        assert auth._shadow_check(request, settings, OTHER) is None
    assert _counts() == before
    assert logs == [{"event": "auth.shadow_check_failed", "log_level": "debug"}]


def test_would_reject_logs_peer_and_client_and_never_the_key(
    shadow: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
    get_settings.cache_clear()
    from api.main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    with structlog.testing.capture_logs() as logs:
        client.get(KEYED, headers={"X-API-Key": OTHER, "X-Forwarded-For": "6.6.6.6, 203.0.113.9"})
        client.get(KEYED)
    events = [e for e in logs if e["event"] == "auth.would_reject"]
    assert [e["reason"] for e in events] == ["reject", "missing"]
    assert events[0]["client"] == "203.0.113.9"
    assert events[0]["peer"] == "testclient"
    assert events[0]["path"] == KEYED and events[0]["method"] == "GET"
    # SE-37A-01: an exact field set on every would_reject event (reject and
    # missing), so no key or hash fragment can be added unnoticed.
    fields = {"event", "log_level", "reason", "path", "method", "peer", "client"}
    assert all(set(e) == fields for e in events), [sorted(e) for e in events]
    assert OTHER not in repr(logs) and RAW not in repr(logs)


# -- production-only switches (A-08) ------------------------------------------


def test_production_ignores_query_api_key(prod: TestClient) -> None:
    assert prod.get(f"{KEYED}?api_key={RAW}").status_code == 401
    assert prod.get(KEYED, headers={"X-API-Key": RAW}).status_code != 401


def test_development_still_honours_query_api_key(dev_auth_on: TestClient) -> None:
    assert dev_auth_on.get(f"{KEYED}?api_key={RAW}").status_code != 401
    assert dev_auth_on.get(KEYED).status_code == 401


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_production_has_no_docs(prod: TestClient, path: str) -> None:
    assert prod.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_development_serves_docs(default_dev: TestClient, path: str) -> None:
    assert default_dev.get(path).status_code == 200


# -- log hygiene --------------------------------------------------------------


def test_auth_logs_carry_peer_and_client(dev_auth_on: TestClient) -> None:
    with structlog.testing.capture_logs() as logs:
        dev_auth_on.get(KEYED)
        dev_auth_on.get(KEYED, headers={"X-API-Key": OTHER})
    events = {e["event"]: e for e in logs if e["event"].startswith("auth.")}
    for name in ("auth.missing_key", "auth.invalid_key"):
        assert events[name]["peer"] == "testclient" and events[name]["client"] == "testclient"


def test_key_never_reaches_the_production_log_stream(
    shadow: TestClient, production_logging: ConfigureFn
) -> None:
    cap = production_logging(json_output=True, log_level="DEBUG")
    shadow.get(KEYED, headers={"X-API-Key": OTHER})
    shadow.get(f"{KEYED}?api_key={OTHER}")
    cap.assert_present("auth.would_reject")
    assert OTHER not in cap.text and RAW not in cap.text
