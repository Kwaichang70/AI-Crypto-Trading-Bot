"""
tests/unit/test_cf2696_api_memory_infra.py
------------------------------------------
CF-26-96 (RK-9703-52 items 2-3, SY-97-135): static pins for the API container
memory settings, the API-restart alert and the deploy note.

Pure file parsing (no docker binary, no network). A new file on purpose: the
existing infra-config suites are not edited by this change.

What is pinned and why:
  - ``memswap_limit`` on the api service equals ``deploy.resources.limits.memory``
    (equal means zero swap, so an over-limit process is killed, not swapped);
  - the memory limit itself is NOT changed here (768 MiB);
  - no ``oom_kill_disable``, ``restart: unless-stopped`` kept, no autoheal;
  - the restart alert keys on ``app_uptime_seconds``, which the app must keep
    exposing (a rename would turn the alert into a silent no-data rule);
  - the deploy note states the kill-instead-of-swap behaviour and only ever
    reads ``docker inspect`` through a narrow ``--format`` (a general inspect of
    the api container prints secrets).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "infra" / "docker-compose.yml"
_ALERTS = _REPO_ROOT / "infra" / "grafana" / "provisioning" / "alerting" / "api-restart-alerts.yml"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "api-container-memory.md"
_PROMETHEUS_PY = _REPO_ROOT / "apps" / "api" / "prometheus.py"

_UNITS = {"b": 1, "k": 1024, "kb": 1024, "m": 1024**2, "mb": 1024**2, "g": 1024**3, "gb": 1024**3}
_API_LIMIT_BYTES = 768 * 1024**2


def _to_bytes(value: Any) -> int:
    """Compose byte-value string (``768m``, ``1g``, ``805306368``) to bytes."""
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+)\s*([a-z]*)", str(value).strip().lower())
    assert match, f"not a compose byte value: {value!r}"
    unit = match.group(2) or "b"
    assert unit in _UNITS, f"unknown unit in {value!r}"
    return int(match.group(1)) * _UNITS[unit]


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def api(compose: dict[str, Any]) -> dict[str, Any]:
    return compose["services"]["api"]


class TestComposeMemory:
    def test_memory_limit_is_unchanged(self, api: dict[str, Any]) -> None:
        # Raising it is the user's decision (RK-9703-52 item 4), not this change.
        assert _to_bytes(api["deploy"]["resources"]["limits"]["memory"]) == _API_LIMIT_BYTES

    def test_memswap_limit_equals_memory_limit(self, api: dict[str, Any]) -> None:
        assert "memswap_limit" in api, "memswap_limit missing: the api container may use swap"
        assert _to_bytes(api["memswap_limit"]) == _to_bytes(
            api["deploy"]["resources"]["limits"]["memory"]
        )

    def test_mem_limit_if_present_is_consistent(self, api: dict[str, Any]) -> None:
        # The Compose spec requires mem_limit to agree with limits.memory when both are set.
        if "mem_limit" in api:
            assert _to_bytes(api["mem_limit"]) == _to_bytes(
                api["deploy"]["resources"]["limits"]["memory"]
            )

    def test_oom_kill_disable_is_never_set(self, compose: dict[str, Any]) -> None:
        for name, svc in compose["services"].items():
            assert "oom_kill_disable" not in svc, f"{name}: oom_kill_disable must not be set"

    def test_restart_policy_is_unchanged(self, api: dict[str, Any]) -> None:
        assert api["restart"] == "unless-stopped"

    def test_no_autoheal_service_or_label(self, compose: dict[str, Any]) -> None:
        text = _COMPOSE.read_text(encoding="utf-8").lower()
        assert "autoheal" not in text
        assert not any("autoheal" in name.lower() for name in compose["services"])

    def test_other_services_have_no_memswap_change(self, compose: dict[str, Any]) -> None:
        # Scope: only the api container is touched by CF-26-96.
        for name, svc in compose["services"].items():
            if name != "api":
                assert "memswap_limit" not in svc, f"{name}: out of scope for CF-26-96"


def _rules() -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    assert doc["apiVersion"] == 1
    (group,) = doc["groups"]
    assert (group["orgId"], group["name"], group["folder"], group["interval"]) == (
        1,
        "trading-infra",
        "Trading",
        "1m",
    )
    return {r["uid"]: r for r in group["rules"]}


class TestRestartAlert:
    def test_only_the_restart_rule_exists(self) -> None:
        assert list(_rules()) == ["trading-api-restarted"]

    def test_rule_shape(self) -> None:
        rule = _rules()["trading-api-restarted"]
        assert rule["condition"] == "C"
        assert rule["labels"] == {"severity": "critical"}
        assert rule["for"] == "0s"
        assert rule["noDataState"] == "OK"  # trading-api-down covers a missing target
        assert [d["refId"] for d in rule["data"]] == ["A", "C"]
        query, expr = rule["data"]
        assert query["datasourceUid"] == "prometheus"
        assert query["model"]["instant"] is True
        assert expr["model"]["conditions"] == [{"evaluator": {"type": "gt", "params": [0.5]}}]

    def test_query_is_the_uptime_reset_selector(self) -> None:
        query = _rules()["trading-api-restarted"]["data"][0]["model"]["expr"]
        assert query == 'max(resets(app_uptime_seconds{job="trading-bot-api"}[15m]))'

    def test_job_label_matches_the_prometheus_scrape_job(self) -> None:
        scrape = yaml.safe_load(
            (_REPO_ROOT / "infra" / "prometheus.yml").read_text(encoding="utf-8")
        )
        assert "trading-bot-api" in [j["job_name"] for j in scrape["scrape_configs"]]

    def test_scrape_path_is_metrics_and_the_app_serves_that_route(self) -> None:
        # SE-2696-03 (d): the rule reads the custom registry served at /metrics. If the
        # scrape path drifted, the series would vanish and noDataState OK would hide it.
        scrape = yaml.safe_load(
            (_REPO_ROOT / "infra" / "prometheus.yml").read_text(encoding="utf-8")
        )
        (job,) = [j for j in scrape["scrape_configs"] if j["job_name"] == "trading-bot-api"]
        assert job["metrics_path"] == "/metrics"
        source = _PROMETHEUS_PY.read_text(encoding="utf-8")
        assert re.search(r'@app\.get\(\s*"/metrics"', source), "the /metrics route moved"
        assert "generate_latest(PROMETHEUS_REGISTRY)" in source

    def test_the_app_still_exposes_app_uptime_seconds(self) -> None:
        # The rule is keyed on this gauge; a rename would leave it silent (noDataState OK).
        assert '"app_uptime_seconds"' in _PROMETHEUS_PY.read_text(encoding="utf-8")

    def test_the_pinned_trading_safety_group_is_not_touched(self) -> None:
        text = (_ALERTS.parent / "trading-alerts.yml").read_text(encoding="utf-8")
        assert "trading-api-restarted" not in text


class TestDeployNote:
    @pytest.fixture(scope="class")
    def note(self) -> str:
        return _RUNBOOK.read_text(encoding="utf-8")

    def test_states_kill_instead_of_swapping(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "killed (OOM, SIGKILL) instead of swapping" in flat

    def test_states_live_runs_stay_orphaned_until_resumed(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "live runs are set to `orphaned`" in flat
        assert "/resume" in flat

    def test_names_the_post_restart_checks(self, note: str) -> None:
        assert "OOMKilled" in note
        assert "recovery\\.(run_orphaned" in note
        assert "memory.swap.max" in note

    def test_every_docker_inspect_is_narrow(self, note: str) -> None:
        # A general docker inspect of the api container prints secrets (runbook rule).
        lines = [ln for ln in note.splitlines() if "docker inspect" in ln]
        narrow = [ln for ln in lines if "--format" in ln and "OOMKilled" in ln]
        assert narrow, "the OOMKilled check must be present"
        for ln in lines:
            assert "--format" in ln or "never" in ln.lower() or "general" in ln.lower(), ln

    def test_does_not_claim_the_kill_switch_stops_backtests(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "backtests included" not in flat  # create_run gates paper and live only

    def test_does_not_tell_anyone_to_disable_oom_kill(self, note: str) -> None:
        assert "must not be added" in note  # the only mention is the prohibition

    def test_no_command_prints_the_environment_or_the_full_config(self, note: str) -> None:
        # SE-2696-04: a full inspect, the container env and an interpolated compose
        # config all print secrets; only the prohibition line may name them.
        banned = re.compile(
            r"docker\s+(container\s+)?inspect(?!\s+--format)|\bprintenv\b"
            r"|exec\s+(-T\s+)?api\s+env\b|\$DC\s+config\b|docker\s+compose\s+config\b"
        )
        hits = [ln for ln in note.splitlines() if banned.search(ln)]
        assert all("never" in ln.lower() for ln in hits), hits


def test_the_alert_file_holds_rules_only_no_contact_point_or_secret() -> None:
    # SE-2696-05: a contact point needs a secret; it is operator-provisioned, never in the repo.
    doc = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    assert set(doc) == {"apiVersion", "groups"}
