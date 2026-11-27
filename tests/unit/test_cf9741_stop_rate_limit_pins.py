"""CF-97-41 characterization pins: stop-run is not in the ordinary-traffic bucket.

Real ``_rate_limit_middleware`` on a throwaway app, tiny limits, frozen clock.
"""

from __future__ import annotations

import types
import uuid
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter

from api import rate_limit
from api.config import get_settings

RUN = "/api/v1/runs"
KILL = "/api/v1/emergency/kill-switch"
OTHER = "/api/v1/other-resource"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    for k, v in {
        "REQUIRE_API_AUTH": "false",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_READ": "3/minute",
        "RATE_LIMIT_WRITE": "2/minute",
        "TRUSTED_PROXY_COUNT": "1",
    }.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    # Frozen clock for the in-memory storage; fresh limiter per test.
    monkeypatch.setattr("limits.storage.memory.time", types.SimpleNamespace(time=lambda: 1.0e6))
    monkeypatch.setattr(
        rate_limit, "limiter", Limiter(key_func=rate_limit.get_client_ip, storage_uri="memory://")
    )
    app = FastAPI()
    for path in (RUN, f"{RUN}/{{rid}}", f"{RUN}/{{rid}}/emergency-stop", KILL, OTHER):
        app.add_api_route(path, lambda: {}, methods=["GET", "POST", "DELETE"])
    rate_limit.setup_rate_limiting(app)
    yield TestClient(app)
    get_settings.cache_clear()


def _codes(c: TestClient, method: str, path: str, n: int, ip: str = "10.0.0.1") -> list[int]:
    return [c.request(method, path, headers={"X-Forwarded-For": ip}).status_code for _ in range(n)]


def test_stop_bucket_is_separate_from_get_and_post(client: TestClient) -> None:
    assert _codes(client, "GET", RUN, 4) == [200, 200, 200, 429]
    assert _codes(client, "POST", RUN, 3) == [200, 200, 429]
    assert _codes(client, "DELETE", OTHER, 3) == [200, 200, 429]
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 3) == [200, 200, 429]


def test_stop_calls_do_not_consume_get_or_post(client: TestClient) -> None:
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 3) == [200, 200, 429]
    assert _codes(client, "GET", RUN, 3) == [200, 200, 200]
    assert _codes(client, "POST", RUN, 2) == [200, 200]


def test_stop_bucket_is_shared_across_run_ids_and_per_client(client: TestClient) -> None:
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 2) == [200, 200]
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 1) == [429]
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 2, ip="10.0.0.2") == [200, 200]


def test_emergency_stop_and_kill_switch_post_are_exempt(client: TestClient) -> None:
    assert set(_codes(client, "POST", f"{RUN}/{uuid.uuid4()}/emergency-stop", 5)) == {200}
    assert set(_codes(client, "POST", KILL, 5)) == {200}


def test_kill_switch_get_is_not_exempt(client: TestClient) -> None:
    assert _codes(client, "GET", KILL, 4) == [200, 200, 200, 429]
