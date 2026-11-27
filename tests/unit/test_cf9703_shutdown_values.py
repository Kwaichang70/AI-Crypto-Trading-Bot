"""CF-97-03 P7c (SY-9703S-03, A4): the graceful-shutdown values are pinned.

The API container drains for 25 s (``--timeout-graceful-shutdown`` in ``infra/Dockerfile.api``) and
compose waits 75 s (``stop_grace_period`` of the ``api`` service) before SIGKILL.  P7a's ML-training
shutdown measurement (25.3 s to exit) and the shutdown budget comment in the compose file depend on
both.  Static text checks, each with a planted control; a new file because the infra-config pin file
is reserved for another phase.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_INFRA = Path(__file__).resolve().parents[2] / "infra"


def drain_seconds(dockerfile: str) -> list[str]:
    return re.findall(r'"--timeout-graceful-shutdown",\s*"([^"]*)"', dockerfile)


def grace_period(compose: str) -> object:
    return yaml.safe_load(compose)["services"]["api"].get("stop_grace_period")


def test_the_api_image_drains_for_25_seconds() -> None:
    assert drain_seconds((_INFRA / "Dockerfile.api").read_text()) == ["25"]


def test_the_api_service_waits_75_seconds_before_sigkill() -> None:
    assert grace_period((_INFRA / "docker-compose.yml").read_text()) == "75s"


def test_the_pins_flag_planted_changes() -> None:
    cmd = '["python", "-m", "uvicorn", "--timeout-graceful-shutdown", "%s"]'
    assert drain_seconds(cmd % "25") == ["25"]
    assert drain_seconds(cmd % "24") != ["25"] and drain_seconds(cmd % "26") != ["25"]
    assert drain_seconds('["python", "-m", "uvicorn"]') == []  # the flag removed
    svc = "services:\n  api:\n    stop_grace_period: %s\n"
    assert grace_period(svc % "75s") == "75s" and grace_period(svc % "74s") != "75s"
    assert grace_period("services:\n  api: {}\n") is None  # the key removed
    assert grace_period("services:\n  ui:\n    stop_grace_period: 75s\n  api: {}\n") is None
