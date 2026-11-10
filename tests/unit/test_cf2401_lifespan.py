"""
tests/unit/test_cf2401_lifespan.py
-----------------------------------
CF-24-01 P5: T-28 (lifespan order, no provider await, shutdown order), T-29 (restart: empty book,
no provider-cache seeding), T-33 (one worker, static), N-5 (done-callback on the supervisor) and
CF-2401-33 (the poller's clients are built without ``session=``).

The lifespan tests run the REAL ``lifespan`` through ``TestClient`` with the database-facing steps
replaced by recorders and every provider call blackholed (it never returns).
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
import structlog
import yaml
from fastapi.testclient import TestClient

import common.logging as clog
from api.config import get_settings
from common import redaction as R
from common.metrics import metrics
from data import live_context_policy as policy
from data.live_context import LIVE_BOOK
from tests._cf36_logging import isolated_logging_state

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "apps/api/main.py"
SENTINEL_KEY = "CF2401LIFESPANKEY0123456789"


def _lifespan_tree() -> ast.AsyncFunctionDef:
    tree = ast.parse(MAIN.read_text())
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan"
    )


def _first_line(fn: ast.AST, needle: str) -> int:
    leaf = (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)  # not Try/If: they start earlier
    hits = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, leaf)
        and needle in ast.unparse(n)
        and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]  # not a docstring
    assert hits, needle
    return min(hits)


# ---------------------------------------------------------------------------
# T-28 (static): order, removals, constructors, done-callback
# ---------------------------------------------------------------------------
def test_the_kill_switch_loads_before_the_poller_and_the_poller_starts_before_recovery() -> None:
    fn = _lifespan_tree()
    kill = _first_line(fn, "_kill_switch.load(")
    build = _first_line(fn, "ContextPoller(")
    start = _first_line(fn, "_context_poller.start()")
    recovery = _first_line(fn, "recover_orphaned_runs()")
    clients = [_first_line(fn, f"{c}(") for c in ("FearGreedClient", "CoinGeckoClient")]
    assert kill < min(clients) < build < start < recovery
    assert recovery < _first_line(fn, "RetrainingService(")  # SY-01-15 step 5: the rest, after


def test_the_warm_ups_and_the_warmer_are_gone() -> None:
    text = MAIN.read_text()
    for gone in (
        "warmed_up",
        "warmup_failed",
        "HistoryCacheWarmer",
        "history_cache_warmer",
        ".get_latest(",
    ):
        assert gone not in text, gone
    root = ROOT / "apps/api"
    assert not (root / "services/history_cache_warmer.py").exists()
    for path in (*(root).rglob("*.py"), *(ROOT / "packages").rglob("*.py")):
        assert "services.history_cache_warmer" not in path.read_text(), path


def test_the_lifespan_awaits_no_provider() -> None:
    awaited = [ast.unparse(n.value) for n in ast.walk(_lifespan_tree()) if isinstance(n, ast.Await)]
    assert not [
        a for a in awaited if any(w in a for w in ("get_latest", "fetch_validated", "get_history"))
    ]
    assert not [a for a in awaited if "_context_poller" in a and ".start" in a]  # start() is sync


def test_the_clients_are_built_without_a_session_and_the_supervisor_has_a_done_callback() -> None:
    """CF-2401-33(a) and N-5."""
    fn = _lifespan_tree()
    names = {"FearGreedClient", "CoinGeckoClient", "FREDClient", "WhaleAlertClient"}
    calls = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") in names
    ]
    assert {c.func.id for c in calls} == names  # type: ignore[attr-defined]
    assert all("session" not in {k.arg for k in c.keywords} and len(c.args) == 0 for c in calls)
    callbacks = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_done_callback"
    ]
    assert "_poller_task.add_done_callback(_context_poller.on_supervisor_done)" in callbacks


def test_the_p5_lifespan_and_container_log_sites_carry_no_exception_text() -> None:
    """P5-PR-09 / SEC-P5-03: the P5 sites in main.py and container.py log error_type only."""
    from tests.unit.test_cf2401_poller import exc_text_problems

    events = ("fgi_client.", "coingecko_client.", "fred_client.", "whale_alert_client.",
              "context_poller.")  # fmt: skip
    tag = ("background_tasks.cancel_all: context_poller",)
    main, box = MAIN.read_text(), (ROOT / "apps/api/container.py").read_text()
    assert exc_text_problems(_lifespan_tree(), events) == []
    assert exc_text_problems(ast.parse(box), tag) == []
    for new in ("exc_info=True)", "e=exc)", "e=str(exc))"):
        assert exc_text_problems(ast.parse(main.replace("**exc_log_fields(exc))", new)), events)
    assert exc_text_problems(ast.parse(box.replace("type(exc).__name__,", "exc,")), tag)


def test_the_shutdown_stops_the_poller_flushes_then_closes_the_clients() -> None:
    fn = _lifespan_tree()
    stop = _first_line(fn, "_poller.stop()")
    flush = _first_line(fn, "_poller.flush_sink(")
    container = _first_line(fn, "container.shutdown()")
    close = [
        _first_line(fn, f"{c}.close()")
        for c in ("_fgi_client", "_coingecko_client", "_fred_client", "_whale_alert_client")
    ]
    assert stop < flush < container < min(close)


# ---------------------------------------------------------------------------
# T-28 / T-29 (behavioural): the real lifespan with blackholed providers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # Settings reads .env from the CWD: never a real one


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "REQUIRE_API_AUTH": "false",
        "RATE_LIMIT_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
        "DEBUG": "false",
        "ML_AUTO_RETRAIN": "false",
        "ADMIN_API_KEY": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4",
        "DATABASE_URL": "postgresql+asyncpg://test:pw@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379/0",
        "FRED_API_KEY": SENTINEL_KEY + "F",
        "WHALE_ALERT_API_KEY": SENTINEL_KEY + "W",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)


def _boot(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the real lifespan; return (event order, constructor kwargs, recovery-time reads)."""
    from api.services import kill_switch
    from data.macro_data import FREDClient
    from data.market_signals import CoinGeckoClient
    from data.sentiment import FearGreedClient
    from data.whale_tracker import WhaleAlertClient
    from trading.mtf_signals import LIVE_SIGNALS

    order: list[str] = []
    kwargs: dict[str, dict[str, Any]] = {}
    seen: dict[str, Any] = {}

    async def blackhole(*_a: Any, **_k: Any) -> None:
        await asyncio.Event().wait()  # a provider that never answers

    for cls, name in (
        (FearGreedClient, "fgi"),
        (CoinGeckoClient, "cg"),
        (FREDClient, "fred"),
        (WhaleAlertClient, "whale"),
    ):
        original_init, original_close = cls.__init__, cls.close

        def init(self: Any, *a: Any, _o: Any = original_init, _n: str = name, **k: Any) -> None:
            kwargs[_n] = {"args": len(a), **k}
            _o(self, *a, **k)
            self._latest_cache = ("STALE-DISPLAY-CACHE", 0.0)  # a provider cache that must not seed

        async def close(self: Any, _o: Any = original_close, _n: str = name) -> None:
            order.append(f"close:{_n}")
            await _o(self)

        monkeypatch.setattr(cls, "__init__", init)
        monkeypatch.setattr(cls, "close", close)
        monkeypatch.setattr(cls, "fetch_validated", blackhole)

    async def load(*_a: Any, **_k: Any) -> None:
        order.append("kill_switch")

    async def recover() -> int:
        order.append("recovery")
        seen["values"] = LIVE_SIGNALS.values_at(None)  # type: ignore[arg-type]
        seen["book_empty"] = len(LIVE_BOOK.snapshot()) == 0
        return 0

    monkeypatch.setattr(kill_switch, "load", load)
    monkeypatch.setattr("api.routers.runs.recover_orphaned_runs", recover)
    from api.container import AppContainer
    from api.services.context_poller import ContextPoller

    start, stop, flush, shutdown = (
        ContextPoller.start,
        ContextPoller.stop,
        ContextPoller.flush_sink,
        AppContainer.shutdown,
    )

    def spy_start(self: Any) -> Any:
        order.append(f"poller_start:{len(self._providers)}")
        return start(self)

    async def spy_stop(self: Any) -> None:
        order.append("poller_stop")
        await stop(self)

    async def spy_flush(self: Any, timeout_s: float = 5.0) -> None:
        order.append(f"flush:{timeout_s}")
        await flush(self, timeout_s)

    async def spy_shutdown(self: Any) -> None:
        order.append("container_shutdown")
        await shutdown(self)

    monkeypatch.setattr(ContextPoller, "start", spy_start)
    monkeypatch.setattr(ContextPoller, "stop", spy_stop)
    monkeypatch.setattr(ContextPoller, "flush_sink", spy_flush)
    monkeypatch.setattr(AppContainer, "shutdown", spy_shutdown)
    monkeypatch.setattr(
        policy, "STARTUP_STAGGER_S", 0
    )  # every provider hangs in flight at shutdown

    with isolated_logging_state():
        structlog.reset_defaults()
        clog._LOGGING_CONFIGURED = False
        R._reset_registry_for_tests()
        get_settings.cache_clear()
        try:
            from api.main import create_app

            app = create_app()
            began = time.monotonic()
            with TestClient(app, raise_server_exceptions=False):
                seen["container"] = app.state.container
                time.sleep(0.2)  # the providers are now blackholed in flight
                poller = app.state.container.background_tasks.context_poller
                seen["poller_running"] = bool(poller and poller.running)
            seen["elapsed"] = time.monotonic() - began
        finally:
            get_settings.cache_clear()
    return order, kwargs, seen


def test_the_real_lifespan_orders_startup_and_shutdown_and_never_waits_for_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(monkeypatch)
    order, kwargs, seen = _boot(monkeypatch)
    assert order[:3] == ["kill_switch", "poller_start:4", "recovery"]  # T-28
    assert order.index("poller_stop") < order.index("flush:5.0") < order.index("container_shutdown")
    assert order.index("container_shutdown") < min(
        i for i, e in enumerate(order) if e.startswith("close:")
    )
    assert {e for e in order if e.startswith("close:")} == {
        "close:fgi",
        "close:cg",
        "close:fred",
        "close:whale",
    }
    assert seen["elapsed"] < 10 and seen["poller_running"] is True  # blackholed, yet no stall
    assert all(k.get("args") == 0 and "session" not in k for k in kwargs.values())  # CF-2401-33(a)
    assert seen["container"].background_tasks.context_poller.running is False  # stopped on exit


def test_a_recovered_consumer_reads_an_empty_book_not_the_provider_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-29: the display caches hold a value, the book is empty, so recovery sees all None."""
    _env(monkeypatch)
    _, _, seen = _boot(monkeypatch)
    from trading.mtf_signals import SignalValues

    assert seen["book_empty"] is True and seen["values"] == SignalValues()


def test_a_poller_that_cannot_start_leaves_the_api_up_with_a_neutral_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(monkeypatch)
    from api.services.context_poller import ContextPoller

    def boom(self: Any) -> Any:
        raise RuntimeError("no poller today")

    monkeypatch.setattr(ContextPoller, "__init__", lambda self, **k: boom(self))
    metrics.gauge("mtf_context_poller_alive", 1.0)  # a stale value from an earlier process
    order, _, seen = _boot(monkeypatch)  # _boot re-patches start; __init__ raises first
    assert order[:2] == ["kill_switch", "recovery"] and seen["book_empty"] is True
    assert metrics.get_all()["gauges"]["mtf_context_poller_alive"] == 0.0  # CR-P5-09


# ---------------------------------------------------------------------------
# T-33: one worker, static and read-only (CF-2401-05, SY-01-22g)
# ---------------------------------------------------------------------------
def compose_problems(api: dict[str, Any]) -> list[str]:
    found = [f"key:{k}" for k in ("command", "entrypoint", "scale") if k in api]
    deploy = api.get("deploy", {})
    found += ["replicas"] if "replicas" in deploy else []
    found += ["global"] if deploy.get("mode") == "global" else []
    found += ["env"] if "WEB_CONCURRENCY" in str(api.get("environment", {})) else []
    return found


def test_dockerfile_and_compose_pin_a_single_worker() -> None:
    dockerfile = (ROOT / "infra/Dockerfile.api").read_text()
    assert '"--workers", "1"' in dockerfile and "gunicorn" not in dockerfile.lower()
    assert "WEB_CONCURRENCY" not in dockerfile
    entrypoint = (ROOT / "infra/docker-entrypoint.sh").read_text()
    assert "--workers" not in entrypoint and "WEB_CONCURRENCY" not in entrypoint
    api = yaml.safe_load((ROOT / "infra/docker-compose.yml").read_text())["services"]["api"]
    assert compose_problems(api) == []
    assert "resources" in api["deploy"]  # the existing block stays allowed (SY-01-22g)


@pytest.mark.parametrize(
    "patch",
    [
        {"command": "x"},
        {"entrypoint": "x"},
        {"scale": 2},
        {"deploy": {"replicas": 2}},
        {"deploy": {"mode": "global"}},
        {"environment": {"WEB_CONCURRENCY": "4"}},
    ],
)
def test_the_compose_pin_catches_overrides(patch: dict[str, Any]) -> None:
    api = yaml.safe_load((ROOT / "infra/docker-compose.yml").read_text())["services"]["api"]
    assert compose_problems({**api, **patch}) != []
