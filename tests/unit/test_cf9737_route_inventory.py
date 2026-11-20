"""CF-97-37 patch 37-A: route-inventory guard and the generated route list.

``tests/fixtures/api_routes_v1.json`` is the interface to the BFF patch (37-B): one
entry per ``/api/v1`` route with its method, path template and dependency class.
This test regenerates the list from ``app.routes`` and fails when the checked-in
file is stale.  Regenerate with ``CF9737_WRITE_ROUTES=1 pytest <this file>``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from starlette.routing import WebSocketRoute

from api.auth import require_api_key
from api.config import get_settings
from api.deps import require_admin

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "api_routes_v1.json"

#: Routes under /api/v1 that are deliberately unauthenticated, with the reason.
#: A new open route fails the guard until it is added here (or gets a dependency).
OPEN_ROUTES: dict[str, str] = {
    "/api/v1/metrics": "JSON metrics; unauthenticated by design, fate owned by CF-97-34",
    "/api/v1/signals": "public read-only market data cache; classification is FI-9737-B",
}


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for k in ("ENVIRONMENT", "REQUIRE_API_AUTH", "API_KEY_HASH", "DEBUG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost:5432/x")
    get_settings.cache_clear()
    from api.main import create_app

    yield create_app()
    get_settings.cache_clear()


def _calls(dep: Dependant) -> set[Any]:
    found = {dep.call}
    for sub in dep.dependencies:
        found |= _calls(sub)
    return found


def _dependency_class(route: APIRoute) -> str:
    calls = _calls(route.dependant)
    key, admin = require_api_key in calls, require_admin in calls
    return {(True, True): "key+admin", (True, False): "key", (False, True): "admin"}.get(
        (key, admin), "none"
    )


def _inventory(app: Any) -> list[dict[str, str]]:
    rows = [
        {"method": m, "path": r.path, "dependency": _dependency_class(r)}
        for r in app.routes
        if isinstance(r, APIRoute) and r.path.startswith("/api/v1")
        for m in sorted(r.methods or ())
        if m not in ("HEAD", "OPTIONS")
    ]
    return sorted(rows, key=lambda x: (x["path"], x["method"]))


def test_generated_route_list_is_current(app: Any) -> None:
    rows = _inventory(app)
    assert len(rows) >= 30  # non-vacuous: the real app (37 routes at a7c6b9e)
    text = json.dumps(rows, indent=1, sort_keys=True) + "\n"
    if os.environ.get("CF9737_WRITE_ROUTES") == "1":
        FIXTURE.write_text(text)
    assert FIXTURE.read_text() == text, "tests/fixtures/api_routes_v1.json is stale"


def test_every_api_v1_route_is_authenticated_or_listed_open(app: Any) -> None:
    open_found = {r["path"] for r in _inventory(app) if r["dependency"] == "none"}
    unexpected = {
        p for p in open_found if not any(p == o or p.startswith(o + "/") for o in OPEN_ROUTES)
    }
    assert not unexpected, f"unauthenticated routes not on the allow-list: {sorted(unexpected)}"
    assert "/api/v1/metrics" in open_found  # the allow-list is not stale either


def test_known_dependency_classes(app: Any) -> None:
    by = {(r["method"], r["path"]): r["dependency"] for r in _inventory(app)}
    assert by[("GET", "/api/v1/runs")] == "key"
    assert by[("GET", "/api/v1/emergency/kill-switch")] == "key"
    assert "key+admin" in by.values() and "admin" in by.values()


def test_there_is_no_websocket_route(app: Any) -> None:
    assert [r.path for r in app.routes if isinstance(r, WebSocketRoute)] == []
