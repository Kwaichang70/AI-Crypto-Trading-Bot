"""
apps/api/config.py
------------------
Application configuration loaded from environment variables and .env file.
Uses Pydantic Settings v2 for strict validation and type safety.
Environment variables always take priority over .env file values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

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
            "SHA-256 hex digest of the valid (primary) API key. "
            "Generate with: echo -n 'my-secret-key' | sha256sum | awk '{print $1}'. "
            "When empty and require_api_auth=False, auth is disabled (dev mode). "
            "NEVER store the raw API key here — only the hash."
        ),
    )
    # SEC-003 (Sprint 45): zero-downtime API key rotation.
    #
    # Setting api_key_hash_secondary to the SHA-256 of a SECOND valid key
    # lets that key authenticate alongside the primary during a rotation
    # window.  Typical workflow:
    #   1. Generate new key.  Move CURRENT hash to api_key_hash_secondary.
    #      Put NEW hash in api_key_hash.  Reload settings (or restart).
    #   2. Roll out the new raw key to all clients.
    #   3. Once every client is migrated, clear api_key_hash_secondary.
    #
    # Both fields are compared via hmac.compare_digest so timing attacks
    # cannot distinguish primary-match from secondary-match from invalid.
    api_key_hash_secondary: str = Field(
        default="",
        description=(
            "SHA-256 hex digest of a secondary (grace-period) API key.  "
            "SEC-003 rotation aid: leave the previous hash here while "
            "clients migrate to a new key, then clear after rollout."
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
    database_url: SecretStr = Field(
        default=SecretStr(""),
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
    def assemble_database_url(cls, values: dict[str, Any]) -> dict[str, Any]:
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

    @model_validator(mode="after")
    def _validate_cors_when_auth_enabled(self) -> "Settings":
        """SEC-007 (Sprint 43): refuse wildcard origins under production
        auth posture.

        ``allow_credentials=True`` is incompatible with ``allow_origins=["*"]``
        per the CORS spec — browsers will refuse the response — but a
        wildcard prefix/suffix or a literal ``"*"`` is a classic operator
        footgun that bypasses every layer of auth in dev mode reflections.
        When ``require_api_auth`` is on, reject the misconfiguration at
        startup so the API never boots into a vulnerable state.
        """
        if not self.require_api_auth:
            return self
        for origin in self.allowed_origins:
            stripped = origin.strip()
            if stripped == "*" or stripped.startswith("*") or stripped.endswith("*"):
                raise ValueError(
                    f"allowed_origins contains wildcard {origin!r} but "
                    f"require_api_auth=True — wildcards are forbidden under "
                    f"production auth posture (SEC-007)."
                )
        return self
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
        default="binance",
        description="CCXT exchange ID (e.g. 'binance', 'kraken')",
    )
    exchange_api_key: SecretStr | None = Field(
        default=None,
        description="Exchange API key — never logged",
    )
    exchange_api_secret: SecretStr | None = Field(
        default=None,
        description="Exchange API secret — never logged",
    )
    exchange_api_passphrase: SecretStr | None = Field(
        default=None,
        description="Exchange API passphrase (e.g. Coinbase legacy keys) — never logged",
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
    # Admin operations (Sprint 50 Cycle 3)
    # ------------------------------------------------------------------
    admin_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Plaintext admin key for privileged operations (global kill-switch).  "
            "Validated directly via hmac.compare_digest — NOT hashed like api_key_hash.  "
            "Generate with: openssl rand -hex 32.  "
            "When empty, all admin endpoints return 401 (indistinguishable from absent header).  "
            "Rotate independently of X-API-Key — rotation only affects kill-switch."
        ),
    )

    @field_validator("admin_api_key")
    @classmethod
    def _validate_admin_api_key(cls, v: SecretStr) -> SecretStr:
        """Reject placeholder and weak values at settings load time.

        Rules (empty string = "not configured" sentinel, always allowed):
        1. Value starting with 'REPLACE_ME' (any case) → rejected (placeholder)
        2. Non-empty value shorter than 32 characters → rejected (too short)
        3. Non-empty value that is entirely alphabetic → rejected (no entropy)
        4. Non-empty value that is entirely numeric → rejected (no entropy)

        Pattern mirrors `validate_api_key_hash` (config.py) + extends it
        with entropy checks analogous to password complexity requirements.
        """
        raw = v.get_secret_value()
        if not raw:
            # Empty = "admin key not configured" sentinel; 401 returned at runtime
            return v
        if raw.upper().startswith("REPLACE_ME"):
            raise ValueError(
                "admin_api_key appears to be a placeholder (starts with 'REPLACE_ME'). "
                "Generate a real key with: openssl rand -hex 32"
            )
        if len(raw) < 32:
            raise ValueError(
                f"admin_api_key must be at least 32 characters (got {len(raw)}). "
                "Generate a real key with: openssl rand -hex 32"
            )
        if raw.isalpha():
            raise ValueError(
                "admin_api_key must not be entirely alphabetic (no entropy). "
                "Generate a real key with: openssl rand -hex 32"
            )
        if raw.isdigit():
            raise ValueError(
                "admin_api_key must not be entirely numeric (no entropy). "
                "Generate a real key with: openssl rand -hex 32"
            )
        return v

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
    # ML Auto-Retraining (Sprint 23)
    # ------------------------------------------------------------------
    ml_auto_retrain: bool = Field(
        default=False,
        description=(
            "Master switch for automatic model retraining. "
            "false = RetrainingService does not start (safe default). "
            "true = RetrainingService polls trade counts and retrains when threshold met. "
            "Follows the same safety-first convention as enable_live_trading."
        ),
    )
    ml_min_trades_for_retrain: int = Field(
        default=50,
        ge=10,
        le=10000,
        description=(
            "Minimum number of new closed trades since last training required "
            "to trigger automatic retraining. 50 provides ~15-20 samples per class "
            "with a typical 40/30/30 label distribution."
        ),
    )
    ml_min_accuracy_threshold: float = Field(
        default=0.38,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum test-set accuracy for a retrained model to be activated. "
            "0.38 is 5% above the 0.33 random baseline for a 3-class classifier. "
            "Models below this threshold are saved but not activated."
        ),
    )
    ml_max_model_versions: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Maximum number of model versions kept on disk and in the DB "
            "per (symbol, timeframe) pair. Older versions are pruned after "
            "each successful retraining. The active model is never pruned."
        ),
    )
    ml_retrain_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description=(
            "How often (in minutes) RetrainingService polls the database for "
            "new trade counts. The service sleeps for this interval before each "
            "check. Default 60 minutes (once per hour)."
        ),
    )

    # ------------------------------------------------------------------
    # Data retention
    # ------------------------------------------------------------------
    equity_snapshot_retention_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days to keep raw equity snapshots before pruning. "
                    "Snapshots older than this are deleted daily at UTC midnight.",
    )
    max_run_duration_hours: int = Field(
        default=168,  # 7 days
        ge=1,
        le=8760,  # 1 year
        description=(
            "Maximum duration in hours that a paper or live run may remain active "
            "before it is automatically stopped.  When the timeout fires, the engine "
            "stop event is set and the run transitions to 'stopped' via the normal "
            "finally-block path (incremental flush + DB status update).  "
            "Default 168 h = 7 days.  Override with MAX_RUN_DURATION_HOURS env var."
        ),
    )
    # AR-006 (Sprint 45): concurrency cap on simultaneously-running paper +
    # live engines.  Each active run consumes DB connection-pool slots
    # (incremental flush every 30 s) plus CCXT rate-limit budget; an
    # unbounded number of runs can exhaust the pool and starve health
    # checks.  Default 20 leaves comfortable headroom above the typical
    # operator workload (~5 concurrent) while staying well below the
    # db_pool_size + db_max_overflow ceiling.
    max_concurrent_runs: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Hard cap on simultaneously-running paper + live engines. "
            "POST /api/v1/runs returns 503 when the active count is at "
            "or above this limit.  Backtest runs (which complete "
            "synchronously inside the POST handler) are NOT counted."
        ),
    )

    # ------------------------------------------------------------------
    # Paper->Live Promotion Gate (Sprint 50 Cycle 5)
    # ------------------------------------------------------------------
    min_paper_trades_for_promotion: int = Field(
        default=50,
        ge=1,
        le=10_000,
        description=(
            "Minimum number of closed trades a paper run must have accumulated "
            "before it is eligible for promotion to live trading.  "
            "This is a data-volume readiness threshold only -- performance "
            "acceptability (Sharpe, drawdown) is left to the operator.  "
            "Override with MIN_PAPER_TRADES_FOR_PROMOTION env var."
        ),
    )
    min_paper_runtime_days: float = Field(
        default=7.0,
        gt=0.0,
        le=365.0,
        description=(
            "Minimum runtime in fractional days a paper run must have completed "
            "before it is eligible for promotion to live trading.  "
            "7.0 days (one full calendar week) ensures the paper run has been "
            "exposed to weekend liquidity conditions and at least one full "
            "market-open/close cycle.  "
            "Override with MIN_PAPER_RUNTIME_DAYS env var."
        ),
    )

    # ------------------------------------------------------------------
    # Walk-Forward OOS Model Activation Gate (Sprint 50 Cycle 5)
    # NOTE: the gate metric is a directional z-score SKILL PROXY (2*acc-1)*sqrt(n),
    # NOT a trading Sharpe. It is magnitude-blind: does not account for fees,
    # slippage, or position sizing. See reports/sprint50-cycle5-quant-backlog.md
    # for the Cycle 6+ plan to replace this proxy with a real BacktestRunner OOS gate.
    # ------------------------------------------------------------------
    min_oos_skill_score: float = Field(
        default=1.64,
        ge=0.0,
        le=5.0,
        description=(
            "Minimum aggregate out-of-sample (OOS) directional z-score skill score "
            "required to activate a model version via PUT /ml/models/{id}/activate.  "
            "Default 1.64 ~= one-sided 95% confidence that the classifier beats a "
            "coin flip on the z-score scale; a 0.5 default would mean only ~69% "
            "confidence and is too weak to gate live model activation (cycle 5 "
            "quant caveat #4).  "
            "The gate uses (2*acc-1)*sqrt(n) -- a directional z-score proxy, NOT a "
            "trading Sharpe (magnitude-blind; no fees/slippage/sizing).  "
            "Computed as the DEFLATED MEDIAN OOS skill score across walk-forward folds "
            "during POST /ml/train.  Models trained before Sprint 50 Cycle 5 have "
            "walk_forward_oos_skill_score=NULL and pass the gate by default "
            "(operator is warned in the response).  "
            "Set to 0.0 to disable the gate (allow all models).  "
            "Override with MIN_OOS_SKILL_SCORE env var."
        ),
    )
    min_worst_fold_skill_score: float = Field(
        default=0.0,
        ge=-10.0,
        le=5.0,
        description=(
            "Minimum OOS directional z-score skill score for the WORST individual "
            "walk-forward fold.  A model may pass the median deflated skill score gate "
            "but still be blocked if even one fold is catastrophic (below this floor). "
            "Default 0.0 -- the worst fold must at least break even. "
            "Set to a negative value to allow some catastrophic folds. "
            "Override with MIN_WORST_FOLD_SKILL_SCORE env var."
        ),
    )
    min_trades_per_fold: int = Field(
        default=20,
        ge=1,
        le=1000,
        description=(
            "Minimum number of OOS trades required in EVERY walk-forward fold "
            "for the OOS skill score metrics to be considered statistically meaningful. "
            "If any fold produces fewer trades than this threshold, the model "
            "activation gate returns 'insufficient_oos_samples' (distinct from "
            "'oos_skill_below_min') so the operator knows to train on more bars "
            "or reduce num_wf_folds rather than improve the strategy. "
            "Override with MIN_TRADES_PER_FOLD env var."
        ),
    )
    wf_num_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description=(
            "Number of walk-forward folds used by POST /ml/train to compute "
            "the OOS Sharpe gate value.  Minimum 2 (degenerate single-fold "
            "would be in-sample only).  Default 5 gives a reasonable "
            "train_fraction=0.7 with 5 equal test windows.  "
            "Override with WF_NUM_FOLDS env var."
        ),
    )

    # ------------------------------------------------------------------
    # Telegram alerts (optional)
    # ------------------------------------------------------------------
    telegram_bot_token: str | None = Field(
        default=None,
        description="Telegram Bot API token for alerts (from @BotFather)",
    )
    telegram_chat_id: str | None = Field(
        default=None,
        description="Telegram chat/group ID to send alerts to",
    )

    # ------------------------------------------------------------------
    # External market signal API keys (Sprint 37)
    # ------------------------------------------------------------------
    # SEC-001 (Sprint 43): both keys use SecretStr so they don't leak into
    # settings.model_dump(), debug repr, or accidental logging — same
    # treatment as the other secrets in this Settings class (api_key_hash,
    # database_url, exchange_api_*).  Call-sites must use .get_secret_value().
    fred_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "FRED (Federal Reserve Economic Data) API key. "
            "Register for free at https://fred.stlouisfed.org/docs/api/api_key.html. "
            "When set, enables macro yield-curve signal injection into MultiTimeframeContext. "
            "When None (default), FRED signals are disabled."
        ),
    )
    whale_alert_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Whale Alert API key for on-chain large transaction monitoring. "
            "Register at https://whale-alert.io/. "
            "When set, enables whale net flow signal injection into MultiTimeframeContext. "
            "When None (default), Whale Alert signals are disabled."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        if not raw.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "database_url must use the 'postgresql+asyncpg://' scheme for async support"
            )
        return v

    @field_validator("api_key_hash", "api_key_hash_secondary")
    @classmethod
    def validate_api_key_hash(cls, v: str) -> str:
        """Validate that api_key_hash[_secondary] is empty or a valid SHA-256 hex digest."""
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

    @model_validator(mode="after")
    def _validate_api_key_hashes_differ(self) -> "Settings":
        """SEC-003: primary and secondary hashes must differ when both set.

        Identical hashes are a no-op rotation that just confuses the
        operator — fail loudly at startup so the misconfiguration cannot
        slip into production.
        """
        if (
            self.api_key_hash
            and self.api_key_hash_secondary
            and self.api_key_hash == self.api_key_hash_secondary
        ):
            raise ValueError(
                "api_key_hash and api_key_hash_secondary must be different "
                "hashes — identical values defeat the purpose of a rotation "
                "window (SEC-003)."
            )
        return self



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the .env file is parsed exactly once per process.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()
