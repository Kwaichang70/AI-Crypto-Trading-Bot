"""
apps/api/config.py
------------------
Application configuration loaded from environment variables and .env file.
Uses Pydantic Settings v2 for strict validation and type safety.
Environment variables always take priority over .env file values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """
    Central application settings.

    All values are read from environment variables first, then from the
    .env file at project root. SecretStr fields are never serialised to
    logs or JSON by default.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = Field(default="AI Crypto Trading Bot", description="Human-readable app name")
    app_version: str = Field(default="0.1.0", description="Semantic version string")
    debug: bool = Field(default=False, description="Enable debug mode — never True in production")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Structured log level"
    )

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Uvicorn bind host")  # noqa: S104
    port: int = Field(default=8000, ge=1024, le=65535, description="Uvicorn bind port")
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins for the Next.js frontend",
    )

    # ------------------------------------------------------------------
    # API authentication
    # ------------------------------------------------------------------
    api_key_hash: str = Field(
        default="",
        description=(
            "SHA-256 hex digest of the valid API key. "
            "Generate with: echo -n 'my-secret-key' | sha256sum | awk '{print $1}'. "
            "When empty and require_api_auth=False, auth is disabled (dev mode). "
            "NEVER store the raw API key here — only the hash."
        ),
    )
    require_api_auth: bool = Field(
        default=False,
        description=(
            "Master switch for API key authentication. "
            "False = all endpoints open (local dev). "
            "True = all non-public endpoints require a valid API key."
        ),
    )

    # ------------------------------------------------------------------
    # Rate limiting (SEC-S2-001)
    # ------------------------------------------------------------------
    rate_limit_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for API rate limiting. "
            "True = enforce per-IP rate limits on all non-exempt endpoints. "
            "False = disable rate limiting (only for local dev/testing). "
            "MUST be True in production."
        ),
    )
    rate_limit_auth_failures: str = Field(
        default="5/minute",
        description=(
            "Rate limit for authentication failures per IP. "
            "Uses limits library syntax: '5/minute', '10/hour', etc. "
            "Tight limit to prevent brute-force API key guessing."
        ),
    )
    rate_limit_write: str = Field(
        default="30/minute",
        description=(
            "Rate limit for write endpoints (POST, PUT, PATCH, DELETE) per IP. "
            "Uses limits library syntax: '30/minute', '60/hour', etc."
        ),
    )
    rate_limit_read: str = Field(
        default="120/minute",
        description=(
            "Rate limit for read endpoints (GET) per IP. "
            "Uses limits library syntax: '120/minute', '300/hour', etc."
        ),
    )
    trusted_proxy_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description=(
            "Number of trusted reverse proxy hops between the internet and this service. "
            "0 (default) = direct connection mode — X-Forwarded-For is ignored entirely. "
            "This is the safe default and prevents IP spoofing via forged XFF headers. "
            "Set to 1 if behind a single nginx/ALB/Cloudflare proxy. "
            "SECURITY: Never set higher than the actual number of controlled proxy hops. "
            "Clients can inject arbitrary entries at the left of XFF; only the "
            "proxy-appended rightmost entries are trustworthy. (CR-RL-002)"
        ),
    )

    # ------------------------------------------------------------------
    # Observability — Prometheus
    # ------------------------------------------------------------------
    prometheus_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for the Prometheus /metrics scrape endpoint. "
            "True = register GET /metrics with Prometheus text exposition format. "
            "False = endpoint is not mounted (useful for minimal deployments). "
            "Controlled by PROMETHEUS_ENABLED environment variable."
        ),
    )

    # ------------------------------------------------------------------
    # Database (PostgreSQL via asyncpg)
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="",
        description=(
            "Async PostgreSQL DSN. "
            "Format: postgresql+asyncpg://user:pass@host:5432/dbname. "
            "If empty, assembled from POSTGRES_* environment variables."
        ),
    )
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="trading_bot", description="PostgreSQL database name")
    postgres_user: str = Field(default="trading", description="PostgreSQL user")
    postgres_password: SecretStr = Field(
        default=SecretStr(""), description="PostgreSQL password -- never logged"
    )

    @model_validator(mode="before")
    @classmethod
    def assemble_database_url(cls, values: dict) -> dict:
        """Build database_url from POSTGRES_* vars if not provided directly."""
        db_url = values.get("database_url", "")
        if not db_url:
            host = values.get("postgres_host", "localhost")
            port = values.get("postgres_port", 5432)
            db = values.get("postgres_db", "trading_bot")
            user = values.get("postgres_user", "trading")
            pw_raw = values.get("postgres_password", "")
            # SecretStr arrives as-is in mode="before"; extract the plain value
            pw = pw_raw.get_secret_value() if isinstance(pw_raw, SecretStr) else pw_raw
            values["database_url"] = (
                f"postgresql+asyncpg://{user}:{pw}@{host}:{port}/{db}"
            )
        return values
    db_pool_size: int = Field(default=10, ge=1, le=50, description="SQLAlchemy connection pool size")
    db_max_overflow: int = Field(
        default=20, ge=0, le=100, description="SQLAlchemy max overflow connections"
    )
    db_pool_timeout: float = Field(
        default=30.0, gt=0, description="Seconds to wait for a pool connection"
    )

    # ------------------------------------------------------------------
    # Redis (optional caching / job queue)
    # ------------------------------------------------------------------
    redis_url: str | None = Field(
        default=None,
        description="Redis DSN. Omit to disable Redis integration.",
    )

    # ------------------------------------------------------------------
    # Exchange / CCXT
    # ------------------------------------------------------------------
    exchange_id: str = Field(
        default="kraken",
        description="CCXT exchange ID (e.g. 'kraken', 'binance')",
    )
    exchange_api_key: SecretStr | None = Field(
        default=None,
        description="Exchange API key — never logged",
    )
    exchange_api_secret: SecretStr | None = Field(
        default=None,
        description="Exchange API secret — never logged",
    )

    # ------------------------------------------------------------------
    # Live trading safety gates (all three must be satisfied)
    # ------------------------------------------------------------------
    enable_live_trading: bool = Field(
        default=False,
        description=(
            "Master switch for live order placement. Must be True to place real orders. "
            "Enforcement happens in ExecutionEngine, not here."
        ),
    )
    live_trading_confirm_token: SecretStr | None = Field(
        default=None,
        description="Extra safety token required alongside enable_live_trading=True",
    )

    # ------------------------------------------------------------------
    # Risk defaults (overridable per-run)
    # ------------------------------------------------------------------
    default_max_open_positions: int = Field(default=3, ge=1, le=20)
    default_per_trade_risk_pct: float = Field(
        default=0.01, gt=0.0, le=0.05, description="Fraction of equity risked per trade (0.01 = 1%)"
    )
    default_max_daily_loss_pct: float = Field(
        default=0.05, gt=0.0, le=0.25, description="Max daily loss as fraction of equity before halt"
    )
    default_max_drawdown_pct: float = Field(
        default=0.15, gt=0.0, le=0.50, description="Max drawdown before circuit breaker fires"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "database_url must use the 'postgresql+asyncpg://' scheme for async support"
            )
        return v

    @field_validator("api_key_hash")
    @classmethod
    def validate_api_key_hash(cls, v: str) -> str:
        """Validate that api_key_hash is either empty or a valid SHA-256 hex digest."""
        if v and len(v) != 64:
            raise ValueError(
                "api_key_hash must be a 64-character SHA-256 hex digest. "
                "Generate with: echo -n 'my-key' | sha256sum"
            )
        if v and not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(
                "api_key_hash must contain only hexadecimal characters (0-9, a-f)"
            )
        return v.lower()  # Normalise to lowercase for consistent comparison



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the .env file is parsed exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()
