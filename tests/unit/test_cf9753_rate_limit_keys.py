"""CF-97-53: buckets key on the route template; emergency exemptions are exact.

Real ``_rate_limit_middleware``, frozen clock, no database.
"""

from __future__ import annotations

import types
import uuid
from collections.abc import Iterator
from typing import NoReturn

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from starlette.requests import Request
from structlog.testing import capture_logs

from api import rate_limit
from api.config import get_settings

RUN = "/api/v1/runs"
STOP = f"{RUN}/{{rid}}"
EMERGENCY = f"{RUN}/{{rid}}/emergency-stop"
KILL = "/api/v1/emergency/kill-switch"
RETRAIN = "/api/v1/ml/retrain"


def _env(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    base = {
        "REQUIRE_API_AUTH": "false",
        "RATE_LIMIT_ENABLED": "true",
        "RATE_LIMIT_READ": "3/minute",
        "RATE_LIMIT_WRITE": "2/minute",
        "TRUSTED_PROXY_COUNT": "1",
        "DATABASE_URL": "postgresql+asyncpg://x:x@localhost:5432/x",
        "DEBUG": "true",
    }
    for k, v in {**base, **extra}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    _env(monkeypatch)
    monkeypatch.setattr("limits.storage.memory.time", types.SimpleNamespace(time=lambda: 1.0e6))
    monkeypatch.setattr(
        rate_limit, "limiter", Limiter(key_func=rate_limit.get_client_ip, storage_uri="memory://")
    )
    app = FastAPI()
    for path in (RUN, STOP, EMERGENCY, KILL, f"{KILL}/clear", RETRAIN + "/{symbol:path}"):
        app.add_api_route(path, lambda: {}, methods=["GET", "POST", "DELETE"])
    rate_limit.setup_rate_limiting(app)
    yield TestClient(app)
    get_settings.cache_clear()


def _codes(c: TestClient, method: str, path: str, n: int = 1, ip: str = "10.0.0.1") -> list[int]:
    return [c.request(method, path, headers={"X-Forwarded-For": ip}).status_code for _ in range(n)]


def test_varying_the_path_does_not_mint_buckets(client: TestClient) -> None:
    got = [_codes(client, "POST", f"{RETRAIN}/S{i}")[0] for i in range(4)]
    assert got == [200, 200, 429, 429]
    got = [_codes(client, "POST", f"/nope/{i}")[0] for i in range(4)]
    assert got == [404, 404, 429, 429]  # one shared unmatched bucket per method and client
    assert _codes(client, "POST", "/nope/x", ip="10.0.0.2") == [404]


def test_flooding_unmatched_and_retrain_paths_never_starves_stop(client: TestClient) -> None:
    _codes(client, "DELETE", "/nope/x", 5)
    _codes(client, "POST", f"{RETRAIN}/A/B", 5)
    _codes(client, "GET", RUN, 5)
    assert _codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 3) == [200, 200, 429]


def test_read_and_write_tiers_differ(client: TestClient) -> None:
    assert _codes(client, "GET", RUN, 4) == [200, 200, 200, 429]
    assert _codes(client, "PATCH", RUN, 3) == [405, 405, 429]  # write tier counts before routing


@pytest.mark.parametrize("rid", [str(uuid.uuid4()), uuid.uuid4().hex])
def test_emergency_stop_post_is_never_limited(client: TestClient, rid: str) -> None:
    assert set(_codes(client, "POST", f"{RUN}/{rid}/emergency-stop", 6)) == {200}


@pytest.mark.parametrize(
    "path",
    [
        f"{RETRAIN}/X/emergency-stop",
        "/api/v1/other/emergency-stop",
        f"{RUN}/not-a-uuid/emergency-stop",
        f"{RUN}/{uuid.uuid4()}/emergency-stop/x",
        f"/x{RUN}/{uuid.uuid4()}/emergency-stop",
    ],
)
def test_lookalike_emergency_stop_paths_are_limited(client: TestClient, path: str) -> None:
    assert 429 in _codes(client, "POST", path, 6)


def test_kill_switch_post_exempt_only_on_the_canonical_path(client: TestClient) -> None:
    assert set(_codes(client, "POST", KILL, 6)) == {200}
    assert 429 in _codes(client, "POST", f"{KILL}/clear", 6)
    assert 429 in _codes(client, "GET", KILL, 6)


def test_template_table_for_representative_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, RATE_LIMIT_ENABLED="false")
    from api.main import create_app

    app = create_app()
    rid = str(uuid.uuid4())
    cases = {
        ("GET", RUN): RUN,
        ("DELETE", f"{RUN}/{rid}"): "/api/v1/runs/{run_id}",
        ("GET", f"{RUN}/{rid}"): "/api/v1/runs/{run_id}",
        ("GET", f"{RUN}/not-a-uuid"): "/api/v1/runs/{run_id}",
        ("POST", f"{RUN}/{rid}/emergency-stop"): "/api/v1/runs/{run_id}/emergency-stop",
        ("POST", KILL): KILL,
        ("POST", f"{KILL}/clear"): f"{KILL}/clear",
        ("POST", "/api/v1/ml/retrain/BTC/USDT"): "/api/v1/ml/retrain/{symbol:path}",
        ("POST", "/api/v1/ml/retrain/x/emergency-stop"): "/api/v1/ml/retrain/{symbol:path}",
        ("DELETE", RUN): "_unmatched_",
        ("GET", "/api/v1/nope"): "_unmatched_",
    }
    for (method, path), want in cases.items():
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "root_path": "",
            "headers": [],
            "query_string": b"",
            "app": app,
        }
        assert rate_limit._route_template(Request(scope)) == want, (method, path)


@pytest.mark.parametrize("symbol", ["BTC/USDT", "1INCH/EUR", "btc/usd"])
def test_retrain_accepts_spot_symbols(monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
    _env(monkeypatch, RATE_LIMIT_ENABLED="false")
    from api.main import create_app

    resp = TestClient(create_app()).post(f"/api/v1/ml/retrain/{symbol}")
    assert resp.status_code == 503  # no RetrainingService here: past validation


@pytest.mark.parametrize(
    "symbol",
    ["BTC", "BTC/USDT/emergency-stop", "BTC/USDT:USDT", "BTC/US%20T", "A/" + "B" * 30, "/", "BTC/"],
)
def test_retrain_rejects_other_paths_with_422(monkeypatch: pytest.MonkeyPatch, symbol: str) -> None:
    _env(monkeypatch, RATE_LIMIT_ENABLED="false")
    from api.main import create_app

    resp = TestClient(create_app()).post(f"/api/v1/ml/retrain/{symbol}")
    assert resp.status_code == 422


def test_emergency_stop_exemption_is_post_only(client: TestClient) -> None:
    assert 429 in _codes(client, "GET", f"{RUN}/{uuid.uuid4()}/emergency-stop", 6)


def test_stop_bucket_is_separate_from_other_write_methods_on_the_same_template(
    client: TestClient,
) -> None:
    path = f"{RUN}/{uuid.uuid4()}"
    assert _codes(client, "POST", path, 3) == [200, 200, 429]
    assert _codes(client, "PUT", path, 1) == [405]  # PUT is a write method too
    assert _codes(client, "DELETE", path, 3) == [200, 200, 429]


def _raw(c: TestClient, method: str, path: str, n: int = 1) -> list[int]:
    """Like _codes, but a redirect is reported, not followed (it would hit the canonical path)."""
    return [
        c.request(
            method, path, headers={"X-Forwarded-For": "10.0.0.1"}, follow_redirects=False
        ).status_code
        for _ in range(n)
    ]


def _boom(*_: object, **__: object) -> NoReturn:
    raise RuntimeError("limiter fault")


@pytest.mark.parametrize("fault", ["lookup", "storage"])
def test_limiter_fault_fails_open_and_is_logged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "lookup":
        monkeypatch.setattr(rate_limit, "_route_template", _boom)
    else:
        monkeypatch.setattr(rate_limit.limiter._limiter, "hit", _boom)
    with capture_logs() as logs:
        assert set(_codes(client, "GET", RUN, 5)) == {200}  # read limit is 3: nothing is counted
        assert set(_codes(client, "DELETE", f"{RUN}/{uuid.uuid4()}", 5)) == {200}
    failed = [e for e in logs if e["event"] == "rate_limit.check_failed"]
    assert len(failed) == 10


def test_emergency_paths_never_reach_the_lookup(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rate_limit, "_route_template", _boom)
    with capture_logs() as logs:
        assert set(_codes(client, "POST", f"{RUN}/{uuid.uuid4()}/emergency-stop", 5)) == {200}
        assert set(_codes(client, "POST", KILL, 5)) == {200}
    assert not [e for e in logs if e["event"] == "rate_limit.check_failed"]


def test_slash_form_stop_calls_do_not_use_up_the_canonical_stop_bucket(client: TestClient) -> None:
    path = f"{RUN}/{uuid.uuid4()}"
    assert _raw(client, "DELETE", path + "/", 3) == [307, 307, 429]  # unmatched bucket, limit 2
    assert _codes(client, "DELETE", path, 3) == [200, 200, 429]


def test_kill_switch_slash_form_is_limited_while_the_canonical_path_stays_exempt(
    client: TestClient,
) -> None:
    assert _raw(client, "POST", KILL + "/", 3) == [307, 307, 429]
    assert set(_codes(client, "POST", KILL, 6)) == {200}
