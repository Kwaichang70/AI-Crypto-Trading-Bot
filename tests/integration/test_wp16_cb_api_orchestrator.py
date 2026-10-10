"""
tests/integration/test_wp16_cb_api_orchestrator.py
----------------------------------------------------
WP1.6 -- orchestrator wiring, the circuit-breaker endpoints, the HALT Telegram
alert and CF-24.

    WP16-T-27  run_paper_engine / run_live_engine (fresh, resume, protective) build the
               breaker from THAT run's risk parameters and register the alert listener
    WP16-T-28  GET /runs/{id}/circuit-breaker returns the typed state of a real breaker
    WP16-T-29  POST .../reset -> 410, no state change, no audit row, 401 without the key
    WP16-T-30  Telegram: HALT entered CRITICAL / cleared WARNING, nothing for REDUCE or
               DAILY_LIMIT, no secret/amount ever logged or sent
    WP16-T-31  CF-24: after a resume the persisted ``drawdown_pct`` is the gate-true one
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from api.db.models import EquitySnapshotORM
from common.types import TimeFrame
from tests._wp15_logging import _structlog_unfiltered  # noqa: F401
from tests.integration.test_wp15b_resume_flush_watermark import _run_until_final_flush
from tests.integration.test_wp15b_resume_seed_wiring import _session_factory
from tests.unit._wp15_support import make_fill
from tests.unit._wp16_support import HALT_ONLY, OTHER, bar, build_engine, buy, live_cb, snap
from trading.recovery import ResumeSnapshot
from trading.safety import CircuitBreaker, CircuitBreakerResponse, build_run_circuit_breaker

SYM = "BTC/USD"
RUN_ID = "ce9bea50-0000-0000-0000-0000000016aa"


# ---------------------------------------------------------------------------
# T-27  orchestrator wiring
# ---------------------------------------------------------------------------


class _CapturingEngine:
    """Stands in for ``StrategyEngine``: records its constructor kwargs, then is cancelled."""

    instances: ClassVar[list[_CapturingEngine]] = []

    def __init__(self, *args: object, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._stop_event = asyncio.Event()
        self.auto_stop_reason: str | None = None
        self.start = AsyncMock()
        # WP1.3b-a: the resume sites call ``prime_resume_exit_state`` before ``start``.
        self.prime_resume_exit_state = MagicMock()
        self.run_live_loop = AsyncMock(side_effect=asyncio.CancelledError())
        self.stop = AsyncMock()
        self.risk_manager = MagicMock()
        _CapturingEngine.instances.append(self)


@pytest.fixture(autouse=True)
def _quiet_side_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.services import run_orchestrator as orch

    monkeypatch.setattr(orch, "_flush_incremental", AsyncMock())
    monkeypatch.setattr(orch, "_incremental_flush_loop", AsyncMock())
    monkeypatch.setattr(orch, "_auto_stop_after", AsyncMock())
    monkeypatch.setattr(orch, "_SHUTDOWN_REQUESTED", False)
    _CapturingEngine.instances.clear()


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.exchange_api_key = None
    settings.exchange_api_secret = None
    settings.exchange_api_passphrase = None
    settings.max_run_duration_hours = 24.0
    settings.exchange_id = "coinbase"
    return settings


def _empty_resume() -> ResumeSnapshot:
    return ResumeSnapshot(
        run_id=str(uuid.uuid4()),
        initial_cash=Decimal("10000"),
        fills=[],
        orders=[],
        peak_equity_hint=None,
        max_bar_index=-1,
    )


async def _drive(mode: str, *, resume: bool, protective: bool = False) -> _CapturingEngine:
    from api.services import run_orchestrator as orch

    listener = MagicMock(name="alert-listener")
    common: dict[str, Any] = {
        "run_id_str": str(uuid.uuid4()),
        "strategy_cls": MagicMock,
        "strategy_name": "grid_trading",
        "strategy_params": {},
        "symbols": [SYM],
        "timeframe": TimeFrame.ONE_HOUR,
        "initial_capital": "10000",
        "resume": _empty_resume() if resume else None,
    }
    patches: list[Any] = [
        patch("api.db.session.get_session_factory", _session_factory()),
        patch("api.config.get_settings", return_value=_settings()),
        patch("data.services.ccxt_market_data.CCXTMarketDataService", MagicMock()),
        patch("trading.portfolio.PortfolioAccounting", MagicMock()),
        patch("trading.strategy_engine.StrategyEngine", _CapturingEngine),
        patch.object(orch, "make_circuit_breaker_alert_listener", MagicMock(return_value=listener)),
        patch.object(orch, "record_audit_event", AsyncMock()),
    ]
    if mode == "paper":
        patches.append(patch("trading.engines.paper.PaperExecutionEngine", MagicMock()))
        patches.append(patch.object(orch, "_auto_retry_paper_run", AsyncMock()))
        runner = orch.run_paper_engine
    else:
        patches.append(patch.object(orch, "build_live_ccxt_exchange", MagicMock()))
        patches.append(patch("trading.engines.live.LiveExecutionEngine", MagicMock()))
        runner = orch.run_live_engine
        common["protective_mode"] = protective
    with contextlib.ExitStack() as stack:  # unwinds every started patch even if one fails
        for p in patches:
            stack.enter_context(p)
        with pytest.raises(asyncio.CancelledError):
            await runner(**common)
    assert len(_CapturingEngine.instances) == 1, "the engine must have been constructed"
    engine = _CapturingEngine.instances[0]
    engine.alert_listener = listener  # type: ignore[attr-defined]
    engine.run_id = common["run_id_str"]  # type: ignore[attr-defined]
    return engine


_LIVE_BANDS = ("0.2", "0.18", "0.10", "0.08")  # entry / exit / reduce entry / reduce exit
_PAPER_BANDS = ("0.3", "0.27", "0.15", "0.12")


@pytest.mark.parametrize(
    ("mode", "resume", "protective", "bands"),
    [
        ("paper", False, False, _PAPER_BANDS),
        ("paper", True, False, _PAPER_BANDS),
        ("live", False, False, _LIVE_BANDS),
        ("live", True, False, _LIVE_BANDS),
        ("live", True, True, _LIVE_BANDS),
    ],
    ids=["paper-fresh", "paper-resume", "live-fresh", "live-resume", "live-protective"],
)
async def test_wp16_t27_the_engine_gets_the_runs_own_breaker_and_the_alert_listener(
    mode: str, resume: bool, protective: bool, bands: tuple[str, str, str, str]
) -> None:
    engine = await _drive(mode, resume=resume, protective=protective)
    cb = engine.kwargs["circuit_breaker"]
    assert isinstance(cb, CircuitBreaker)
    t = cb.thresholds
    assert t is not None
    assert (t.halt_entry, t.halt_exit, t.reduce_entry, t.reduce_exit) == tuple(
        Decimal(b) for b in bands
    )
    assert t.max_daily_loss_pct == Decimal("0.08")
    assert t.deescalation_evaluations == 3 and t.reduce_multiplier == Decimal("0.5")
    assert cb._run_id == engine.run_id  # type: ignore[attr-defined]
    assert cb._listeners == [engine.alert_listener]  # type: ignore[attr-defined]
    assert cb.state.evaluations == 0  # nothing is persisted: a resume builds a FRESH breaker


# ---------------------------------------------------------------------------
# T-28 / T-29  endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> Any:
    from api.routers.runs import _RUN_ENGINES

    _RUN_ENGINES.clear()
    yield _RUN_ENGINES
    _RUN_ENGINES.clear()


async def test_wp16_t28_get_returns_the_typed_state_of_a_real_breaker(
    client_dev: TestClient, registry: dict[str, Any]
) -> None:
    breaker = build_run_circuit_breaker("t28", HALT_ONLY)
    engine, _ex, _pf = build_engine(
        cb=breaker, signals=lambda i: [buy(OTHER)], params=HALT_ONLY
    )
    registry[RUN_ID] = engine
    await bar(engine, "800")  # 20% drawdown on the live limit -> HALT

    response = client_dev.get(f"/api/v1/runs/{RUN_ID}/circuit-breaker")
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "halt" and body["graduated_response"] == "halt"
    assert body["entries_blocked"] is True and body["entry_size_multiplier"] == "0"
    assert body["active_conditions"] == ["halt", "reduce"]
    assert body["evaluations"] == 1 and body["halt_count"] == 1
    assert body["halt_exit_progress"] == 0 and body["reduce_exit_progress"] == 0
    assert body["is_tripped"] is True and body["trip_count"] == 0
    assert Decimal(body["thresholds"]["halt_entry"]) == Decimal("0.2")
    assert Decimal(body["inputs"]["drawdown_pct"]) == Decimal("0.2")
    assert body["last_evaluated_at"] is not None and body["halt_since"] is not None
    assert body["last_halt_reason"].startswith("drawdown 0.2 >= limit 0.2")


def test_wp16_t28_get_is_404_when_the_run_is_not_in_process(
    client_dev: TestClient, registry: dict[str, Any]
) -> None:
    assert client_dev.get(f"/api/v1/runs/{RUN_ID}/circuit-breaker").status_code == 404
    registry[RUN_ID] = SimpleNamespace(circuit_breaker=None)
    assert client_dev.get(f"/api/v1/runs/{RUN_ID}/circuit-breaker").status_code == 404


def test_wp16_t29_reset_is_410_with_no_state_change_no_audit_row_and_no_db(
    client_dev: TestClient, registry: dict[str, Any]
) -> None:
    from api.routers import circuit_breaker as router_module

    breaker = live_cb("t29")
    for _ in range(3):
        breaker.evaluate(snap("0.25"))
    engine = SimpleNamespace(circuit_breaker=breaker)
    registry[RUN_ID] = engine
    before = breaker.state.model_dump(mode="json")

    with patch("api.services.audit_log.record_audit_event", new=AsyncMock()) as audit:
        response = client_dev.post(f"/api/v1/runs/{RUN_ID}/circuit-breaker/reset")

    assert response.status_code == 410
    assert response.json() == {"detail": router_module.RESET_RETIRED_DETAIL}
    assert breaker.state.model_dump(mode="json") == before
    assert breaker.current_response == CircuitBreakerResponse.HALT
    audit.assert_not_awaited()
    # the handler takes no DB session and no request: nothing to write an audit row with
    assert list(inspect.signature(router_module.reset_circuit_breaker).parameters) == ["run_id"]


def test_wp16_t29_reset_requires_the_api_key(
    client_prod: TestClient, auth_headers: dict[str, str]
) -> None:
    url = f"/api/v1/runs/{RUN_ID}/circuit-breaker/reset"
    assert client_prod.post(url).status_code == 401
    assert client_prod.get(f"/api/v1/runs/{RUN_ID}/circuit-breaker").status_code == 401
    assert client_prod.post(url, headers=auth_headers).status_code == 410
    # WP16-CR-04 (M20): the OpenAPI document also says 410
    responses = client_prod.app.openapi()["paths"][
        "/api/v1/runs/{run_id}/circuit-breaker/reset"
    ]["post"]["responses"]
    assert "410" in responses and "200" not in responses


# ---------------------------------------------------------------------------
# T-30  Telegram
# ---------------------------------------------------------------------------


class _FakeNotifier:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []
        self._raises = raises

    async def send_alert(
        self, alert_type: str, level: str, message: str, run_id: str | None = None
    ) -> bool:
        if self._raises is not None:
            raise self._raises
        self.calls.append((alert_type, level, message, run_id))
        return True


async def _settle() -> None:
    from api.services.run_orchestrator import _ALERT_TASKS

    await asyncio.gather(*list(_ALERT_TASKS))
    await asyncio.sleep(0)  # let the done-callbacks run


def _alert_cb(notifier: Any, monkeypatch: pytest.MonkeyPatch) -> CircuitBreaker:
    import api.main as main
    from api.services.run_orchestrator import make_circuit_breaker_alert_listener

    monkeypatch.setattr(main, "get_telegram_notifier", lambda: notifier)
    cb = live_cb("11111111-2222-3333-4444-555555555555")
    cb.evaluate(snap("0"))
    cb.add_transition_listener(make_circuit_breaker_alert_listener(cb._run_id))
    return cb


async def test_wp16_t30_halt_entered_and_cleared_alert_once_each_and_the_task_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.services.run_orchestrator import _ALERT_TASKS

    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)

    cb.evaluate(snap("0.25"))  # HALT entered
    assert len(_ALERT_TASKS) == 1, "the task must be referenced until it finishes"
    await _settle()
    assert len(notifier.calls) == 1 and not _ALERT_TASKS  # discarded on completion
    alert_type, level, message, run_id = notifier.calls[0]
    assert (alert_type, level) == ("circuit_breaker", "CRITICAL")
    assert run_id == "11111111-2222-3333-4444-555555555555"
    assert "HALT" in message and "NOT stopped" in message and "exits active" in message
    for needle in ("25.00%", "20.00%", "18.00%", "3 bars"):
        assert needle in message

    for _ in range(3):  # three bars below 0.9 L -> HALT cleared (-> REDUCE)
        cb.evaluate(snap("0.05"))
    await _settle()
    assert len(notifier.calls) == 2
    assert notifier.calls[1][1] == "WARNING" and "HALT cleared" in notifier.calls[1][2]
    assert "50% size" in notifier.calls[1][2]


async def test_wp16_t30_reduce_and_daily_limit_send_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.services.run_orchestrator import _ALERT_TASKS

    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)
    cb.evaluate(snap("0.12"))  # OK -> REDUCE
    cb.evaluate(snap("0.12", worst="-800"))  # REDUCE -> DAILY_LIMIT
    cb.evaluate(snap("0.0", day=snap().day.replace(day=11)))  # day roll: DAILY -> REDUCE
    assert not _ALERT_TASKS
    await _settle()
    assert notifier.calls == []


async def test_wp16_t30_halt_into_daily_limit_says_entries_stay_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)
    cb.evaluate(snap("0.25"))
    for _ in range(3):
        cb.evaluate(snap("0.05", worst="-800"))  # HALT clears while the day latch holds
    await _settle()
    assert [c[1] for c in notifier.calls] == ["CRITICAL", "WARNING"]
    assert "stay blocked by the daily loss limit" in notifier.calls[1][2]


async def test_wp16_t30_no_notifier_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = _alert_cb(None, monkeypatch)
    d = cb.evaluate(snap("0.25"))
    await _settle()
    assert d.level == CircuitBreakerResponse.HALT


async def test_wp16_t30_a_raising_notifier_is_swallowed_and_never_logs_a_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "123456789:AAFakeBotTokenValue-XYZ"  # noqa: S105 - fake value to prove it is never logged
    boom = RuntimeError(f"POST https://api.telegram.org/bot{secret}/sendMessage failed")
    cb = _alert_cb(_FakeNotifier(raises=boom), monkeypatch)
    with capture_logs() as logs:
        d = cb.evaluate(snap("0.25"))
        await _settle()
    assert d.level == CircuitBreakerResponse.HALT  # the engine is unaffected
    failed = [e for e in logs if e["event"] == "circuit_breaker.telegram_notify_failed"]
    assert len(failed) == 1 and failed[0]["error_type"] == "RuntimeError"
    assert secret not in repr(logs) and "AAFakeBotToken" not in repr(logs)
    assert "api.telegram.org" not in repr(failed)


def test_wp16_t30_without_a_running_loop_the_alert_is_skipped_and_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cb = _alert_cb(_FakeNotifier(), monkeypatch)  # evaluated from a plain (loop-less) test
    with capture_logs() as logs:
        d = cb.evaluate(snap("0.25"))
    assert d.level == CircuitBreakerResponse.HALT
    assert [e for e in logs if e["event"] == "circuit_breaker.alert_skipped_no_event_loop"]


async def test_wp16_t24_telegram_text_is_allow_listed_no_amounts_no_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)
    d = cb.evaluate(snap("0.25"))  # nav 75 000 000, peak 100 000 000 in this fixture
    for _ in range(3):
        cb.evaluate(snap("0.05"))
    await _settle()
    assert len(notifier.calls) == 2
    nav, peak = str(d.inputs.nav), str(d.inputs.peak_equity)
    for _t, _lvl, message, _rid in notifier.calls:
        assert nav not in message and peak not in message and "000000" not in message
        lowered = message.lower()
        for forbidden in ("balance", "api_key", "secret", "token", "account", "passphrase"):
            assert forbidden not in lowered
        # every number is a percentage or the bar count N
        numbers = re.findall(r"\d+(?:\.\d+)?%?", message)
        assert all(n.endswith("%") or n == "3" for n in numbers), numbers


async def test_wp16_t24_telegram_texts_are_html_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """WP16-CR-01 / SEC-09: ``send_alert`` uses ``parse_mode="HTML"``; a raw ``<``,
    ``>`` or ``&`` can make Telegram reject the only HALT push alert."""
    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)
    cb.evaluate(snap("0.25"))  # HALT entered (CRITICAL)
    for _ in range(3):
        cb.evaluate(snap("0.05"))  # HALT cleared -> REDUCE (WARNING)
    cb.evaluate(snap("0.25"))  # HALT entered again
    for _ in range(3):
        cb.evaluate(snap("0.05", worst="-800"))  # HALT cleared -> DAILY_LIMIT (WARNING)
    await _settle()
    texts = [c[2] for c in notifier.calls]
    assert len(texts) == 4
    assert "reached the limit 20.00%" in texts[0]
    assert "50% size" in texts[1] and "daily loss limit" in texts[3]
    for text in texts:
        assert not set(text) & set("<>&"), text


async def test_wp16_t24_restart_watch_halt_text_is_true_allow_listed_and_html_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1.6 fix2 (CF-35): a HALT entered by the restart watch (0.9 L <= dd < L) must
    not claim that the limit was reached; same allow-list and HTML rules as T-24."""
    notifier = _FakeNotifier()
    cb = _alert_cb(notifier, monkeypatch)  # one calm evaluation: the HALT watch is on
    assert cb.evaluate(snap("0.19")).level == CircuitBreakerResponse.HALT
    await _settle()
    assert [(c[0], c[1]) for c in notifier.calls] == [("circuit_breaker", "CRITICAL")]
    text = notifier.calls[0][2]
    assert "Drawdown 19.00% is at or above 18.00% during the restart watch (limit 20.00%)." in text
    assert "reached the limit" not in text
    assert not set(text) & set("<>&"), text
    numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
    assert all(n.endswith("%") or n == "3" for n in numbers), numbers


# ---------------------------------------------------------------------------
# T-31  CF-24
# ---------------------------------------------------------------------------


def _probe_a_snapshot(hint: Decimal | None) -> ResumeSnapshot:
    """SEC-201 probe-A shape: an 'orphan' fill priced above every later mark.

    25 BTC bought at 100 (NAV 10 000 at the fill price = the replay peak); the
    persisted peak hint is LOWER (9 000) or absent.  The first real mark of 60 puts
    the NAV at 9 000: the gate-true drawdown is 10%, the raw hint would say 0.
    """
    return ResumeSnapshot(
        run_id=str(uuid.uuid4()),
        initial_cash=Decimal("10000"),
        fills=[
            make_fill(
                quantity="25", price="100", symbol=SYM,
                executed_at=datetime.now(UTC) - timedelta(days=5),
            )
        ],
        orders=[],
        peak_equity_hint=hint,
        max_bar_index=3,
    )


async def _persisted_rows(snapshot: ResumeSnapshot, portfolio: Any, state: Any) -> list[Any]:
    from api.services import run_orchestrator

    added: list[Any] = []
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.merge = AsyncMock()
    db.flush = AsyncMock()
    db.add_all = MagicMock(side_effect=lambda rows: added.extend(rows))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=db)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch(
        "api.db.session.get_session_factory",
        MagicMock(return_value=MagicMock(return_value=ctx)),
    ):
        await run_orchestrator.flush_incremental(
            run_id_str=snapshot.run_id, portfolio=portfolio, execution_engine=None,
            state=state, log=MagicMock(),
        )
    return [r for r in added if isinstance(r, EquitySnapshotORM)]


@pytest.mark.parametrize(
    ("mode", "protective"),
    [("paper", False), ("live", False), ("live", True)],
    ids=["paper", "live", "live-protective"],
)
@pytest.mark.parametrize("hint", [Decimal("9000"), None], ids=["low-hint", "no-hint"])
async def test_wp16_t31_cf24_persisted_drawdown_is_the_gate_true_drawdown_after_a_resume(
    mode: str, protective: bool, hint: Decimal | None
) -> None:
    snapshot = _probe_a_snapshot(hint)
    captured = await _run_until_final_flush(mode, snapshot, protective=protective)
    portfolio, state = captured["portfolio"], captured["state"]

    portfolio.update_market_prices({SYM: Decimal("60")})  # NAV 9 000 vs the replay peak 10 000
    risk = portfolio.risk_snapshot()
    assert risk.peak_equity == Decimal("10000")
    expected = (risk.peak_equity - Decimal("9000")) / risk.peak_equity
    assert expected == Decimal("0.1")

    rows = await _persisted_rows(snapshot, portfolio, state)
    assert [r.equity for r in rows] == [Decimal("9000")]
    assert rows[0].drawdown_pct == expected  # HEAD persisted 0 here (peak := the raw hint)
