"""
tests/unit/test_cf2401_alerts.py
---------------------------------
CF-24-01 P6: T-34 (Telegram ``StatusListener``: transitions, gate, levels, dead, recorder, 6 h
re-notify, fixed strings, failures), T-35 (the three Grafana rules), CF-2401-96 (poller-dead rule
on the heartbeat timestamp), CF-2401-98(a) amended (open / announced rule), CF-2401-107 wiring.

No network: the notifier is a fake; the sleep and the monotonic clock are the poller test's
``FakeClock``.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from structlog.testing import capture_logs

import api.services.context_alerts as ca
from api.services.context_alerts import ContextAlertListener
from api.services.context_poller import ContextPoller
from data import live_context_policy as policy
from data.live_context import LIVE_CONSUMERS
from tests.unit.test_cf2401_lifespan import _first_line, _lifespan_tree
from tests.unit.test_cf2401_poller import FakeClock, exc_text_problems, poller_problems

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "apps/api/services/context_alerts.py"
RULES = ROOT / "infra/grafana/provisioning/alerting/trading-alerts.yml"
TOKEN, CHAT = "123456:CF2401TOKENSENTINEL", "-100CF2401CHATSENTINEL"
S1, S2, S3, S4 = list(ca._SERIES)


class Notifier:
    """Fake ``TelegramNotifier``: records ``send_alert``; can return False, raise or hang."""

    def __init__(self, result: bool = True, exc: Exception | None = None, hang: bool = False):
        self.calls: list[tuple[str, str, str]] = []
        self.result, self.exc, self.hang = result, exc, hang

    async def send_alert(self, *, alert_type: str, level: str, message: str) -> bool:
        self.calls.append((alert_type, level, message))
        if self.hang:
            await asyncio.Event().wait()
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
async def env():  # type: ignore[no-untyped-def]
    LIVE_CONSUMERS.reset()
    clock, box = FakeClock(), SimpleNamespace(n=Notifier())
    lst = ContextAlertListener(lambda: box.n, sleep=clock.sleep, clock=clock.monotonic)
    yield SimpleNamespace(lst=lst, box=box, clock=clock, n=box.n)
    await lst.aclose()
    LIVE_CONSUMERS.reset()


def levels(n: Notifier) -> list[str]:
    return [c[1] for c in n.calls]


async def dead_supervisor(poller: ContextPoller, how: str) -> None:
    async def boom() -> None:
        raise RuntimeError(f"https://api.telegram.org/bot{TOKEN}")

    async def quiet() -> None:
        return None

    task = asyncio.ensure_future(boom() if how == "raise" else quiet())
    await asyncio.wait({task})
    poller.on_supervisor_done(task)
    for _ in range(20):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# T-34: transitions (CF-2401-98(a) amended, cases i-vi) and the gate
# ---------------------------------------------------------------------------
async def test_the_open_announced_rule_covers_cases_i_to_vi(env: Any) -> None:
    lst, n = env.lst, env.n
    LIVE_CONSUMERS.started("paper")
    await lst.on_status_change(S1, "unavailable", "fresh")  # (i) the boot move is silent
    assert n.calls == []
    await lst.on_status_change(S2, "fresh", "stale")  # (ii) a first-ever fresh -> stale alerts
    assert [c[2] for c in n.calls] == [ca._STALE.format(ca._SERIES[S2])]
    await lst.on_status_change(S3, "stale", "fresh")  # (iii) a seeded-stale baseline is silent
    assert len(n.calls) == 1
    await lst.on_status_change(S1, "fresh", "stale")  # (iv) one stale, one recovered
    await lst.on_status_change(S1, "stale", "fresh")
    assert [c[2] for c in n.calls[1:]] == [
        ca._STALE.format(ca._SERIES[S1]),
        ca._FRESH.format(ca._SERIES[S1]),
    ]
    await lst.on_status_change(S2, "stale", "fresh")  # (vi) S2 is still open and announced
    assert n.calls[-1][2] == ca._FRESH.format(ca._SERIES[S2]) and len(n.calls) == 4
    LIVE_CONSUMERS.reset()
    await lst.on_status_change(S4, "fresh", "stale")  # (v) no consumers: nothing, not announced
    token = LIVE_CONSUMERS.started("paper")
    await lst.on_status_change(S4, "stale", "fresh")
    assert len(n.calls) == 4
    LIVE_CONSUMERS.stopped(token)


async def test_the_gate_and_the_levels(env: Any) -> None:
    lst, n = env.lst, env.n
    await lst.on_status_change(S1, "fresh", "stale")  # no consumer run: no message
    assert n.calls == []
    token = LIVE_CONSUMERS.started("paper")
    await lst.on_status_change(S2, "fresh", "stale")
    LIVE_CONSUMERS.started("live")
    await lst.on_status_change(S3, "fresh", "stale")
    assert levels(n) == ["WARNING", "CRITICAL"] and {c[0] for c in n.calls} == {"mtf_context"}
    LIVE_CONSUMERS.stopped(token)


async def test_a_series_never_fresh_since_boot_alerts_on_its_first_failure_once(env: Any) -> None:
    LIVE_CONSUMERS.started("paper")  # QR-P6-01: e.g. a bad FRED key at boot
    await env.lst.on_status_change(S3, "unavailable", "auth_failed")
    texts = [t.format(ca._SERIES[S3]) for t in (ca._STALE, ca._FRESH)]
    assert [c[2] for c in env.n.calls] == texts[:1]  # the first failure, at once
    await env.lst.on_status_change(S3, "auth_failed", "unavailable")  # still open: silent
    await env.lst.on_status_change(S3, "unavailable", "egress_blocked")
    await env.lst.on_status_change(S3, "egress_blocked", "fresh")
    assert [c[2] for c in env.n.calls] == texts


async def test_other_moves_and_unknown_series_send_nothing(env: Any) -> None:
    LIVE_CONSUMERS.started("live")
    for old, new in (("not_configured", "stale"), ("fresh", "not_configured"), ("fresh", "fresh")):
        await env.lst.on_status_change(S1, old, new)
    await env.lst.on_status_change("evil:<b>http://x", "fresh", "stale")
    assert env.n.calls == [] and env.lst._tasks == {}


# ---------------------------------------------------------------------------
# T-34 / N-5: poller dead, recorder failing, 6 h re-notify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["raise", "return"])
async def test_a_dead_supervisor_sends_exactly_one_critical_message(env: Any, how: str) -> None:
    poller = ContextPoller(listener=env.lst)  # no consumer run: dead is always sent
    await dead_supervisor(poller, how)
    assert env.n.calls == [("mtf_context", "CRITICAL", ca._DEAD)]
    await env.lst.on_poller_dead()  # a second call is not a second message
    assert len(env.n.calls) == 1


async def test_a_shutdown_cancel_is_not_a_death(env: Any) -> None:
    poller = ContextPoller(listener=env.lst)
    poller.start()
    await poller.stop()
    await dead_supervisor(poller, "return")
    assert env.n.calls == []


async def test_a_raising_notifier_never_reaches_the_poller(env: Any) -> None:
    env.box.n = Notifier(exc=RuntimeError(f"https://api.telegram.org/bot{TOKEN} {CHAT}"))
    poller = ContextPoller(listener=env.lst)
    with capture_logs() as logs:
        await dead_supervisor(poller, "return")
    sent = [x for x in logs if x["event"] == "context_alerts.send_failed"]
    assert [x["error_type"] for x in sent] == ["RuntimeError"]
    assert not [x for x in logs if x["event"] == "context_poller.listener_failed"]
    assert TOKEN[7:] not in repr(logs) and CHAT not in repr(logs)


async def test_the_dead_message_is_renotified_every_six_hours(env: Any) -> None:
    await env.lst.on_poller_dead()
    await env.clock.advance(policy.STATUS_REMINDER_S)
    await env.clock.advance(policy.STATUS_REMINDER_S)
    assert levels(env.n) == ["CRITICAL"] * 3 and env.clock.sleeps[0][1] == policy.STATUS_REMINDER_S
    assert {c[2] for c in env.n.calls} == {ca._DEAD}  # the reminder text is the same fixed string


async def test_a_death_before_the_notifier_exists_is_delivered_by_the_reminder(env: Any) -> None:
    late, env.box.n = env.box.n, None  # F1: step 7 has not run yet
    await env.lst.on_poller_dead()
    env.box.n = late
    await env.clock.advance(policy.STATUS_REMINDER_S)
    assert len(late.calls) == 1


async def test_a_stale_series_is_renotified_while_open_gated_and_stops_on_recovery(
    env: Any,
) -> None:
    token = LIVE_CONSUMERS.started("paper")
    await env.lst.on_status_change(S1, "fresh", "stale")
    await env.clock.advance(policy.STATUS_REMINDER_S)
    assert len(env.n.calls) == 2
    LIVE_CONSUMERS.stopped(token)
    await env.clock.advance(policy.STATUS_REMINDER_S)  # no consumer run: skipped, still open
    assert len(env.n.calls) == 2 and S1 in env.lst._tasks
    LIVE_CONSUMERS.started("paper")
    await env.lst.on_status_change(S1, "stale", "fresh")
    await env.clock.advance(policy.STATUS_REMINDER_S)
    assert len(env.n.calls) == 3 and env.lst._tasks == {}  # one recovered, then no reminder


async def test_a_reminder_announces_after_a_skipped_wake_and_the_recovery_is_gated(
    env: Any,
) -> None:
    await env.lst.on_status_change(S1, "fresh", "stale")  # no consumer run: open, not announced
    await env.lst.on_status_change(S2, "fresh", "stale")
    await env.clock.advance(policy.STATUS_REMINDER_S)  # a gated wake is skipped, not the end
    token = LIVE_CONSUMERS.started("live")
    await env.clock.advance(policy.STATUS_REMINDER_S)  # the reminders announce both
    await env.lst.on_status_change(S1, "stale", "fresh")  # announced by a reminder: recovered
    LIVE_CONSUMERS.stopped(token)
    await env.lst.on_status_change(S2, "stale", "fresh")  # announced, but the gate is closed
    stale = {("CRITICAL", ca._STALE.format(ca._SERIES[s])) for s in (S1, S2)}
    assert {c[1:] for c in env.n.calls[:2]} == stale and len(env.n.calls) == 3
    assert env.n.calls[2] == ("mtf_context", "CRITICAL", ca._FRESH.format(ca._SERIES[S1]))


async def test_recorder_failing_is_always_sent_once_per_window(env: Any) -> None:
    env.box.n = None
    await env.lst.on_recorder_failing()  # F1: no notifier yet; the window is not used up
    env.box.n = env.n
    for _ in range(3):
        await env.lst.on_recorder_failing()  # no consumer run, however often P7 calls it
    assert env.n.calls == [("mtf_context", "WARNING", ca._RECORDER)]
    await env.clock.advance(policy.STATUS_REMINDER_S)
    await env.lst.on_recorder_failing()
    assert len(env.n.calls) == 2


# ---------------------------------------------------------------------------
# T-34: fixed strings; failures (F1-F4); lifecycle
# ---------------------------------------------------------------------------
def test_every_message_is_a_fixed_string_with_no_markup_url_or_value() -> None:
    texts = [ca._DEAD, ca._RECORDER] + [
        t.format(name) for t in (ca._STALE, ca._FRESH) for name in ca._SERIES.values()
    ]
    for text in texts:
        assert re.fullmatch(r"[A-Za-z0-9 ;.]+", text), text  # no <, >, &, :, /, %, digits+units
    assert not [t for t in texts if "http" in t.lower() or "{" in t]
    assert "provider is failing" in ca._STALE and "as it ages out" in ca._DEAD  # RK-P6-03


async def test_send_failures_log_the_error_type_only(env: Any) -> None:
    LIVE_CONSUMERS.started("live")
    secret = RuntimeError(f"https://api.telegram.org/bot{TOKEN}/sendMessage chat {CHAT}")
    for bad in (Notifier(result=False), Notifier(exc=secret)):
        env.box.n = bad
        with capture_logs() as logs:
            await env.lst.on_status_change(S1, "fresh", "stale")  # never raises
            await env.lst.on_status_change(S1, "stale", "fresh")
        assert all(x["event"] == "context_alerts.send_failed" for x in logs) and logs
        assert TOKEN[7:] not in repr(logs) and CHAT not in repr(logs)
        assert {x.get("error_type") for x in logs} <= {None, "RuntimeError"}


async def test_a_hung_send_is_bounded(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ca, "SEND_TIMEOUT_S", 0.01)
    env.box.n = Notifier(hang=True)
    with capture_logs() as logs:
        await asyncio.wait_for(env.lst.on_poller_dead(), 5)
    assert [x["error_type"] for x in logs] == ["TimeoutError"]


async def test_a_missing_or_failing_getter_sends_nothing_and_keeps_the_state(env: Any) -> None:
    def broken() -> Any:
        raise RuntimeError(TOKEN)

    with capture_logs() as logs:  # CR-P6-05: a silent recovery never calls the getter
        await ContextAlertListener(broken).on_status_change(S2, "unavailable", "fresh")
    assert logs == []
    LIVE_CONSUMERS.started("live")
    for getter in (lambda: None, broken):
        lst = ContextAlertListener(getter, sleep=env.clock.sleep)
        with capture_logs() as logs:
            await lst.on_status_change(S1, "fresh", "stale")
            await lst.on_poller_dead()
        assert S1 in lst._tasks and ca._DEAD_KEY in lst._tasks and TOKEN[7:] not in repr(logs)
        assert {x["event"] for x in logs} <= {"context_alerts.notifier_unavailable"}  # no send
        await lst.aclose()


async def test_aclose_cancels_every_reminder_sends_nothing_and_is_idempotent(env: Any) -> None:
    LIVE_CONSUMERS.started("paper")
    await env.lst.on_status_change(S1, "fresh", "stale")
    await env.lst.on_poller_dead()
    tasks = list(env.lst._tasks.values())
    await env.lst.aclose()
    await env.lst.aclose()
    assert all(t.cancelled() for t in tasks) and env.lst._tasks == {} and len(env.n.calls) == 2


# ---------------------------------------------------------------------------
# CF-2401-107: wiring in main.py, and the module's own pins
# ---------------------------------------------------------------------------
def test_main_wires_the_listener_in_its_own_try_and_shuts_it_down_in_order() -> None:
    fn = _lifespan_tree()
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    own = [t for t in tries if "ContextAlertListener" in ast.unparse(t.body)]
    assert len(own) == 1 and "ContextPoller" not in ast.unparse(own[0])  # F5
    (start,) = [t for t in tries if "ContextPoller(" in ast.unparse(t.body)]
    assert "await _alerts.on_poller_dead()" in ast.unparse(start.handlers)  # RK-P6-02
    src = ast.unparse(fn)
    assert "lambda: container.services.telegram_notifier" in src
    assert "get_telegram_notifier" not in ast.unparse(own[0]) and "listener=_alerts" in src
    assert src.count("TelegramNotifier(") == 1  # no second transport
    flush, close = _first_line(fn, "_poller.flush_sink"), _first_line(fn, "_alerts.aclose")
    assert flush < close < _first_line(fn, "_telegram_notifier.close")  # F6
    assert exc_text_problems(fn, ("context_alerts.",)) == []
    leaky = ast.parse(MODULE.read_text().replace("**exc_log_fields(exc)", "e=str(exc)"))
    assert exc_text_problems(leaky)


def test_the_module_has_no_transport_no_exception_text_and_one_reminder_constant() -> None:
    src = MODULE.read_text()
    tree = ast.parse(src)
    mods = {a.module or "" for a in ast.walk(tree) if isinstance(a, ast.ImportFrom)}
    mods |= {x.name for a in ast.walk(tree) if isinstance(a, ast.Import) for x in a.names}
    assert not {m for m in mods if m.split(".")[0] in {"aiohttp", "httpx", "requests", "trading"}}
    assert exc_text_problems(tree) == [] and "21_600" not in src and "21600" not in src
    assert poller_problems(src) == [] and 10 < ca.SEND_TIMEOUT_S <= 15  # no env read; F4 range
    assert not any(isinstance(n, ast.Name) and n.id == "LIVE_BOOK" for n in ast.walk(tree))


# ---------------------------------------------------------------------------
# T-35 / CF-2401-96 / CF-2401-109: the three Grafana rules
# ---------------------------------------------------------------------------
_POLLER_SRC = (ROOT / "apps/api/services/context_poller.py").read_text()
_KEYS = sorted("uid title condition for noDataState execErrState labels annotations data".split())
_RULE_PINS = {  # uid: (severity, for, noDataState, evaluator)
    "mtf-context-poller-dead": ("critical", "5m", "Alerting", {"type": "gt", "params": [2700]}),
    "mtf-context-stale": ("warning", "15m", "OK", {"type": "gt", "params": [0.5]}),
    "mtf-context-recorder-failing": ("warning", "1m", "OK", {"type": "gt", "params": [0]}),
}


def _rules() -> dict[str, Any]:
    (group,) = yaml.safe_load(RULES.read_text())["groups"]
    return {r["uid"]: r for r in group["rules"]}


def _expr(rule: dict[str, Any]) -> str:
    return str(next(d for d in rule["data"] if d["refId"] == "A")["model"]["expr"])


@pytest.mark.parametrize("uid", sorted(_RULE_PINS))
def test_the_three_rules_are_pinned(uid: str) -> None:
    severity, hold, no_data, evaluator = _RULE_PINS[uid]
    rule = _rules()[uid]
    assert (rule["labels"], rule["for"], rule["noDataState"]) == (
        {"severity": severity},
        hold,
        no_data,
    )
    assert (rule["execErrState"], rule["condition"]) == ("Error", "C")
    assert sorted(rule) == _KEYS  # no notification_settings or contact routing
    (cond,) = [d for d in rule["data"] if d["refId"] == "C"]
    assert cond["model"]["conditions"] == [{"evaluator": evaluator}]
    assert list(_rules())[-3:] == list(_RULE_PINS)  # appended after trading-api-down


def test_the_poller_dead_rule_keys_on_the_heartbeat_and_45_minutes() -> None:
    expr = _expr(_rules()["mtf-context-poller-dead"])
    assert expr == "time() - max(trading_mtf_context_poller_heartbeat_timestamp_seconds)"
    assert "alive" not in expr  # CF-2401-96: a failing tick leaves alive at 1.0
    assert 2700 == 45 * 60 > policy.POLLER_DEAD_AFTER_S == 180


def test_the_rule_metrics_are_emitted_by_the_poller_and_the_recorder_name_is_bound() -> None:
    for uid in ("mtf-context-poller-dead", "mtf-context-stale"):
        names = re.findall(r"trading_(mtf_context_\w+)", _expr(_rules()[uid]))
        assert names and all(f'"{n}"' in _POLLER_SRC for n in names), (uid, names)
    stale = " ".join(_expr(_rules()["mtf-context-stale"]).split())
    assert "and on(series) (trading_mtf_context_series_configured == 1))" in stale  # RD-04.6
    assert stale.endswith("and on() (sum(trading_mtf_context_consumer_runs_active) > 0)")
    assert "trading_mtf_context_recorder_write_failures_total[15m]" in _expr(
        _rules()["mtf-context-recorder-failing"]
    )
