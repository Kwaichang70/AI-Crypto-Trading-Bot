"""
tests/unit/test_cf9734_caddy_metrics_block.py
---------------------------------------------
CF-97-34 (SE-2696-06), option A: static pins for the Caddy block that hides the
unauthenticated JSON snapshot ``GET /api/v1/metrics`` at the edge.

Pure file/text parsing (the caddy binary is not installed in the sandbox), in the
same style as tests/unit/test_wp13a_infra_body_limit.py and
tests/unit/test_wp17b_infra_config.py, which are not edited by this change.

What is pinned:
  - the block exists in the primary site block (not the Grafana block);
  - it sits above the generic ``handle /api/*`` proxy block and the other /api blocks keep
    their order (a style rule: Caddy sorts single-path handle blocks by path length and runs
    the catch-all last, so this is not a routing claim);
  - it is a prefix match (trailing ``*``), answers 404 and does not proxy;
  - Prometheus still scrapes ``api:8000`` directly at ``/metrics`` (the other,
    Prometheus-text route), so the block cannot break monitoring;
  - no Caddy block routes the root ``/metrics`` to the API;
  - the runbook carries the verification commands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CADDYFILE = _REPO_ROOT / "infra" / "Caddyfile"
_PROMETHEUS = _REPO_ROOT / "infra" / "prometheus.yml"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "caddy-metrics-block.md"

_METRICS_HEADER = re.compile(r"^[ \t]*handle[ \t]+/api/v1/metrics\*[ \t]*\{[ \t]*$", re.MULTILINE)


def _primary_site_block(caddyfile_text: str) -> str:
    """Extract the primary `{{TAILSCALE_FQDN}} {` ... `}` block only (same approach as
    tests/unit/test_wp17b_infra_config.py)."""
    lines = caddyfile_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "{{TAILSCALE_FQDN}} {"), None)
    assert start is not None, "primary site block header not found in infra/Caddyfile"
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start : i + 1])
    raise AssertionError("unbalanced braces in the primary Caddyfile site block")


def _body_after(text: str, header_start: int) -> str:
    """Text strictly inside the braces of the block whose header starts at header_start."""
    open_idx = text.index("{", header_start)
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces in handle block")


def _without_comments(text: str) -> str:
    """Drop whole-line `#` comments, so a comment can neither shift a block position nor
    unbalance the brace count (CR-9734-04)."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _code_lines(body: str) -> list[str]:
    """Non-empty, non-comment lines of a block body."""
    return [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]


@pytest.fixture(scope="module")
def caddyfile_text() -> str:
    return _without_comments(_CADDYFILE.read_text())


@pytest.fixture(scope="module")
def primary(caddyfile_text: str) -> str:
    return _primary_site_block(caddyfile_text)


class TestMetricsBlock:
    def test_block_exists_once_in_the_primary_site_block(
        self, primary: str, caddyfile_text: str
    ) -> None:
        assert len(_METRICS_HEADER.findall(primary)) == 1
        # ... and nowhere else (the Grafana site block must not carry it).
        assert len(_METRICS_HEADER.findall(caddyfile_text)) == 1

    def test_block_precedes_the_generic_api_proxy_block(self, primary: str) -> None:
        metrics_idx = _METRICS_HEADER.search(primary).start()  # type: ignore[union-attr]
        generic_idx = primary.index("handle /api/* {")
        assert metrics_idx < generic_idx, (
            "style: the metrics block must sit above `handle /api/*` (Caddy sorts "
            "single-path handle blocks itself; this keeps the layout readable and is not a "
            "routing fault)"
        )

    def test_block_is_before_the_catch_all(self, primary: str) -> None:
        metrics_idx = _METRICS_HEADER.search(primary).start()  # type: ignore[union-attr]
        assert metrics_idx < primary.index("handle {")

    def test_existing_order_is_unchanged(self, primary: str) -> None:
        auth = primary.index("handle /api/auth/* {")
        admin = primary.index("handle /api/admin/* {")
        generic = primary.index("handle /api/* {")
        catch_all = primary.index("handle {")
        assert auth < admin < generic < catch_all

    def test_block_answers_404_and_does_not_proxy(self, primary: str) -> None:
        match = _METRICS_HEADER.search(primary)
        assert match
        body = _body_after(primary, match.start())
        assert _code_lines(body) == ["respond 404"], _code_lines(body)
        assert "reverse_proxy" not in body

    def test_matcher_is_a_prefix_match(self) -> None:
        # The trailing `*` covers the trailing-slash form and any sub-path.
        code = _without_comments(_CADDYFILE.read_text())
        assert _METRICS_HEADER.search(code)
        assert "handle /api/v1/metrics {" not in code

    def test_no_block_routes_the_root_metrics_path_to_the_api(self, primary: str) -> None:
        # The Prometheus text endpoint /metrics reaches the UI via the catch-all, not the API.
        assert not re.search(r"handle\s+/metrics", primary)
        catch_all = _body_after(primary, primary.index("handle {"))
        assert _code_lines(catch_all) == ["reverse_proxy ui:3000"]

    def test_the_generic_api_block_still_proxies_to_the_api(self, primary: str) -> None:
        body = _body_after(primary, primary.index("handle /api/* {"))
        assert "reverse_proxy api:8000" in body


class TestMonitoringIsUnaffected:
    def test_prometheus_scrapes_the_api_directly_at_metrics(self) -> None:
        scrape = yaml.safe_load(_PROMETHEUS.read_text(encoding="utf-8"))
        (job,) = [j for j in scrape["scrape_configs"] if j["job_name"] == "trading-bot-api"]
        assert job["metrics_path"] == "/metrics"
        targets = [t for cfg in job["static_configs"] for t in cfg["targets"]]
        assert targets == ["api:8000"], "the scrape must bypass Caddy"


class TestRunbook:
    @pytest.fixture(scope="class")
    def note(self) -> str:
        return _RUNBOOK.read_text(encoding="utf-8")

    def test_shows_the_block_and_the_outside_check(self, note: str) -> None:
        assert "handle /api/v1/metrics* {" in note
        assert "respond 404" in note
        assert "curl -s -o /dev/null -w '%{http_code}\\n' https://<host>/api/v1/metrics" in note

    def test_shows_the_monitoring_check_and_the_hand_applied_form(self, note: str) -> None:
        assert "trading-bot-api" in note
        assert "caddy validate" in note
        assert "caddy reload" in note
        assert "the line `handle /api/* {`" in note

    def test_live_check_covers_the_encoded_spellings_without_normalising(self, note: str) -> None:
        # SE-9734-01: the block holds only if Caddy matches the decoded path, which was
        # not verified; the live check must send the encoded forms as written.
        assert "--path-as-is" in note
        assert "%6detrics" in note
        assert "%2Fmetrics" in note
        for variant in ("'/api/v1/Metrics'", "'//api/v1/metrics'", "'/API/v1/metrics'"):
            assert variant in note, variant

    def test_any_200_is_a_failure_and_the_check_is_the_evidence(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "Expected: `404` on every line." in flat
        assert "**Any `200` is a failure:**" in flat
        assert "assume the stale bind-mount inode" in flat
        assert "The check, not the reload, is the evidence." in flat

    def test_discloses_that_caddy_was_not_run(self, note: str) -> None:
        # CR-9734-01: no caddy binary was available when this was written.
        flat = " ".join(note.split())
        assert "were not run against Caddy" in flat
        assert "no `caddy` binary was available" in flat
        assert "case, percent-encoding, duplicate slashes" in flat

    def test_steps_are_ordered_validate_reload_live_check(self, note: str) -> None:
        validate = note.index("caddy validate")
        reload_ = note.index("caddy reload")
        live = note.index("--path-as-is")
        assert validate < reload_ < live

    def test_describes_the_prefix_precisely(self, note: str) -> None:
        # CR-9734-03: the matcher is a prefix, not just sub-paths.
        flat = " ".join(note.split())
        assert "any path starting with `/api/v1/metrics`" in flat
        assert "any sub-path" not in flat

    def test_states_the_limits_of_the_block(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "does not change the application" in flat
        assert "firewall" in flat
