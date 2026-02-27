"""
packages/common/config.py
--------------------------
Base configuration class providing environment-variable loading conventions
shared across all packages. Each package sub-classing this gains:
  - Consistent .env loading
  - Structured log bootstrapping
  - Run-ID tracking
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.types import RunMode

__all__ = ["BaseAppConfig", "generate_run_id", "configure_structlog"]


def generate_run_id() -> str:
    """
    Generate a globally unique run identifier.

    Format: ``run_<ISO8601_compact>_<8-char UUID fragment>``
    Example: ``run_20260227T143022Z_a3f7c1b2``

    Deterministic seed for backtests is applied at the strategy layer,
    not here.
    """
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{uid}"


def configure_structlog(log_level: str = "INFO", *, json_logs: bool = True) -> None:
    """
    Bootstrap structlog with a consistent processor chain.

    Call once at application startup in the FastAPI lifespan handler.
    Subsequent calls are idempotent (structlog ignores re-configuration
    after the first call in the same process).

    Parameters
    ----------
    log_level:
        Minimum log level string (e.g. ``"INFO"``).
    json_logs:
        When True, renders logs as JSON suitable for log aggregators.
        When False, renders coloured console output for local development.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class BaseAppConfig(BaseSettings):
    """
    Minimal base configuration for non-API packages.

    API-specific settings (database URL, port, etc.) live in
    ``apps/api/config.py``. This class provides only the fields
    that every component needs regardless of deployment context.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field(
        default="INFO",
        description="Minimum log level for structured output",
    )
    json_logs: bool = Field(
        default=True,
        description="Emit JSON logs when True, coloured console when False",
    )
    run_mode: RunMode = Field(
        default=RunMode.PAPER,
        description="Active run mode",
    )
