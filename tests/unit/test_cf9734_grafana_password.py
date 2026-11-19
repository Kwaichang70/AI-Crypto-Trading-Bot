"""
tests/unit/test_cf9734_grafana_password.py
------------------------------------------
CF-97-34 / SE-9734-04: the Grafana admin password must have no default.

Grafana is published through Caddy on port 3001. With the old
``${GRAFANA_PASSWORD:-admin}`` fallback, a deploy without the variable gave the admin
account a known password. These are static pins (no docker binary, no network):

  - the compose file requires the variable (``:?``) and has no default of any kind;
  - ``.env.example`` names ``GRAFANA_PASSWORD`` with an empty value (never a real or
    placeholder value);
  - the runbook carries the interim check and the facts an operator needs, and never a
    command that prints the password or passes it as an argument.

A new file on purpose: the existing infra-config suites are not edited by this change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "infra" / "docker-compose.yml"
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "grafana-admin-password.md"
_CADDY_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "caddy-metrics-block.md"
_DEPLOY_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "deploy-fase1-minimumset.md"

_REQUIRED_FORM = re.compile(r"^\$\{GRAFANA_PASSWORD:\?[^}\s][^}]*\}$")


@pytest.fixture(scope="module")
def compose_text() -> str:
    return _COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def grafana(compose_text: str) -> dict[str, Any]:
    return yaml.safe_load(compose_text)["services"]["grafana"]


class TestComposeRequiresThePassword:
    def test_admin_password_is_the_required_form_with_a_message(
        self, grafana: dict[str, Any]
    ) -> None:
        value = grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
        assert _REQUIRED_FORM.fullmatch(value), value

    def test_no_default_operator_on_the_variable_anywhere(self, compose_text: str) -> None:
        # `:-`, `-`, `:=`, `=` and `:+` all supply or alter a default; only `:?` may follow.
        for match in re.finditer(r"\$\{GRAFANA_PASSWORD([^}]*)\}", compose_text):
            assert match.group(1).startswith(":?"), match.group(0)

    def test_the_old_default_is_gone_from_the_file(self, compose_text: str) -> None:
        assert ":-admin" not in compose_text
        assert "admin / ${GRAFANA_PASSWORD" not in compose_text

    def test_the_password_variable_is_set_only_once(self, compose_text: str) -> None:
        assert compose_text.count("GF_SECURITY_ADMIN_PASSWORD") == 1

    def test_the_value_is_not_a_literal(self, grafana: dict[str, Any]) -> None:
        value = grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
        assert value.startswith("${") and value.endswith("}")


class TestEnvExample:
    @pytest.fixture(scope="class")
    def lines(self) -> list[str]:
        return _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    def test_names_the_variable_once_with_an_empty_value(self, lines: list[str]) -> None:
        hits = [ln for ln in lines if ln.startswith("GRAFANA_PASSWORD")]
        assert hits == ["GRAFANA_PASSWORD="], hits

    def test_no_commented_example_value(self, lines: list[str]) -> None:
        for ln in lines:
            if "GRAFANA_PASSWORD" in ln and ln.lstrip().startswith("#"):
                assert not re.search(r"GRAFANA_PASSWORD\s*=\s*\S", ln), ln


class TestRunbook:
    @pytest.fixture(scope="class")
    def note(self) -> str:
        return _RUNBOOK.read_text(encoding="utf-8")

    def test_has_the_interim_check_exactly(self, note: str) -> None:
        assert (
            "curl -s -o /dev/null -w '%{http_code}\\n' -u admin:admin https://<host>:3001/api/user"
            in note
        )
        assert "Expected: `401`." in note

    def test_states_the_volume_initialisation_fact(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "only when the `grafana_data` volume is first initialised" in flat
        assert "keeps the password `admin`" in flat

    def test_states_the_whole_project_behaviour_change(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "behaviour change for the whole project" in flat
        assert "could not be scoped to the Grafana service" in flat
        assert "before deploying this compose file, add `GRAFANA_PASSWORD`" in flat

    def test_the_default_password_probe_is_one_shot(self, note: str) -> None:
        # SE-9734-07: repeated failed logins can lock the account.
        assert "Run it once, not in a loop" in note

    def test_states_the_deploy_order(self, note: str) -> None:
        flat = " ".join(note.split())
        first = flat.index("**before** the new compose file reaches the host")
        second = flat.index("**before** any `down`, `up` or `stop`")
        third = flat.index("3. Only then deploy.")
        assert first < second < third
        assert "require exit code 0" in flat
        assert '`$DC config --quiet; echo "exit=$?"`' in flat

    def test_has_the_incident_escape_hatch(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "the plain Docker CLI still works" in flat
        assert "does not interpolate the compose file" in flat
        for command in ("`docker ps`", "`docker logs <container>`", "`docker stop <container>`"):
            assert command in flat, command
        assert "Never use a general `docker inspect`" in flat

    def test_records_the_firewall_rule(self, note: str) -> None:
        assert "443 and 3001" in note
        assert "100.64.0.0/10" in note

    def test_the_cli_flag_is_conditional_and_never_a_trailing_argument(self, note: str) -> None:
        flat = " ".join(note.split())
        assert "Only if the help lists a `--password-from-stdin` flag" in flat
        assert "could not be checked" in flat
        # No `reset-admin-password <value>` form: the next token must be an option.
        for match in re.finditer(r"reset-admin-password\s+(\S+)", note):
            assert match.group(1).startswith("-"), match.group(0)

    def test_no_command_prints_the_password_or_the_environment(self, note: str) -> None:
        banned = re.compile(
            r"docker\s+(container\s+)?inspect(?!\s+--format)|\bprintenv\b"
            r"|exec\s+(-T\s+)?\w+\s+env\b|\$DC\s+config\b(?!\s+--quiet)"
            r"|docker\s+compose\s+config\b(?!\s+--quiet)"
        )
        hits = [ln for ln in note.splitlines() if banned.search(ln)]
        assert all("never" in ln.lower() for ln in hits), hits


class TestDeployRunbookRow:
    @pytest.fixture(scope="class")
    def section(self) -> str:
        text = _DEPLOY_RUNBOOK.read_text(encoding="utf-8")
        start = text.index("### 3.3 Environment inventory")
        return text[start : text.index("### 3.4", start)]

    def test_the_variable_is_listed_in_the_required_table_as_new(self, section: str) -> None:
        rows = [ln for ln in section.splitlines() if ln.startswith("| **`GRAFANA_PASSWORD`**")]
        assert len(rows) == 1, rows
        assert "**New requirement (CF-97-34).**" in rows[0]

    def test_the_row_states_the_deploy_order(self, section: str) -> None:
        row = next(ln for ln in section.splitlines() if ln.startswith("| **`GRAFANA_PASSWORD`**"))
        assert "**before** the new compose file reaches the host" in row
        assert "**before** any `down`/`up`/`stop`" in row
        assert "`$DC config --quiet`" in row
        assert "exit code 0" in row

    def test_the_existing_required_rows_are_kept(self, section: str) -> None:
        for name in ("POSTGRES_PASSWORD", "ADMIN_API_KEY", "INTERNAL_ADMIN_API_KEY"):
            assert f"`{name}`" in section, name


def test_the_caddy_runbook_no_longer_advertises_the_default() -> None:
    text = _CADDY_RUNBOOK.read_text(encoding="utf-8")
    assert "compose default is `admin`" not in text
    assert "grafana-admin-password.md" in text
