# AI Crypto Trading Bot

A production-grade cryptocurrency trading platform with automated strategy execution, backtesting, paper trading, live trading, adaptive learning, and a full web dashboard.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)
![Tests](https://img.shields.io/badge/tests-1500%2B%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)

## Features

- **6 Trading Strategies** — DCA+RSI Hybrid, Grid Trading, RSI Mean Reversion, MA Crossover, Breakout, ML Model
- **3 Run Modes** — Backtest (historical), Paper (simulated live), Live (real orders)
- **Adaptive Learning** — Self-improving parameter optimization from trade outcomes
- **Risk Management** — Graduated circuit breaker, position sizing, daily loss limits
- **Web Dashboard** — Real-time equity curves, trade history, strategy leaderboard
- **Parameter Optimizer** — Grid search with ranked results and one-click run launch
- **ML Training Pipeline** — Train and hot-swap scikit-learn models from the UI
- **Fear & Greed Index** — Contrarian sentiment signal from alternative.me API
- **Multi-Timeframe Analysis** — Higher-timeframe context for strategy decisions
- **Telegram Alerts** — Trade notifications, error alerts, daily summaries to your phone
- **Prometheus + Grafana** — Production monitoring with auto-provisioned 12-panel dashboard
- **CSV Export** — Export trades, orders, fills, positions, and runs to CSV
- **Dark/Light Mode** — System-aware theme toggle with localStorage persistence

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, SQLAlchemy (async), Alembic |
| Trading | CCXT, Pandas, NumPy, scikit-learn |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Database | PostgreSQL (asyncpg) |
| Cache | Redis (optional) |
| Infra | Docker, Docker Compose, GitHub Actions CI |
| Monitoring | Prometheus, Grafana (auto-provisioned dashboard) |

---

## Quick Start

```bash
# Clone and configure
git clone https://github.com/Kwaichang70/AI-Crypto-Trading-Bot.git
cd AI-Crypto-Trading-Bot
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD at minimum

# Start all services
docker compose -f infra/docker-compose.yml --env-file .env up --build

# Access
# Dashboard:  http://localhost:3000
# API:        http://localhost:8000
# API Docs:   http://localhost:8000/docs
# Grafana:    http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## Dashboard

The web dashboard provides:

- **Home** — Aggregate portfolio stats, equity sparkline, recent runs
- **Runs** — List, filter, compare, and export runs with pagination
- **Run Detail** — Equity curve, trades, orders, fills, positions tabs with CSV export
- **New Run** — Strategy selector, parameter editor, adaptive learning toggles
- **Compare** — Side-by-side metrics and overlaid equity curves for 2-5 runs
- **Leaderboard** — Strategy performance ranking across all backtest runs
- **Optimize** — Parameter grid search with ranked results
- **Models** — ML model management (train, retrain, activate, version history)
- **Dark/Light Mode** — System-aware theme toggle
- **Mobile Responsive** — Hamburger menu navigation for tablet/phone
- **Toast Notifications** — Visual feedback for all user actions

---

## Trading Strategies

| Strategy | ID | Description |
|----------|-----|-------------|
| **DCA + RSI Hybrid** | `dca_rsi_hybrid` | Systematic buying every N bars with RSI-based dip boost and profit taking. Trades in ALL market conditions. |
| **Grid Trading** | `grid_trading` | Buy/sell at configurable grid levels above/below reference price. Profits from price oscillations. |
| RSI Mean Reversion | `rsi_mean_reversion` | RSI crossover signals with configurable oversold/overbought thresholds |
| MA Crossover | `ma_crossover` | Fast/slow moving average crossover with trend confirmation |
| Breakout | `breakout` | Donchian channel breakout with ATR-based position sizing |
| ML Model | `model_strategy` | RandomForest classifier trained on 10 OHLCV-derived features |

All strategies support:
- Configurable parameters via the UI
- Trailing stop-loss (optional, per-run)
- Fear & Greed Index confidence boost
- Multi-timeframe context (backtest mode)
- Adaptive learning integration

---

## Adaptive Learning System

The bot can learn from its own trades and improve over time:

1. **Trade Journal** — Tracks MAE/MFE excursion, exit reasons, skipped trades
2. **Performance Analyzer** — Regime analysis, indicator correlation, parameter effectiveness
3. **Adaptive Optimizer** — Conservative parameter tuning (max 20% change per cycle)
4. **Safeguards** — Rollback at -5% PnL, 72h cooldown, disable after 3 rollbacks
5. **Reporting** — Daily/weekly reports, 9 alert types (circuit breaker, rollback, ATH)

Enable via the New Run page: toggle "Enable Adaptive Learning" for paper/live runs.

---

## Telegram Alerts

Get notified on your phone for every trade, error, and circuit breaker event.

```bash
# 1. Create a bot via @BotFather on Telegram
# 2. Add to your .env:
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

You'll receive:
- 🟢 **BUY** / 🔴 **SELL** notifications with symbol, quantity, price, and PnL
- 🚨 **Circuit breaker** activations
- ↩️ **Rollback** alerts when adaptive learning reverts parameters
- 📋 **Daily summaries** with equity, PnL, and trade count

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/v1/runs` | List runs (paginated, filterable) |
| POST | `/api/v1/runs` | Create a new run |
| GET | `/api/v1/runs/{id}` | Run detail |
| DELETE | `/api/v1/runs/{id}` | Stop a running run |
| PATCH | `/api/v1/runs/{id}/archive` | Archive a stopped run |
| GET | `/api/v1/runs/{id}/portfolio` | Portfolio summary |
| GET | `/api/v1/runs/{id}/equity-curve` | Equity time series |
| GET | `/api/v1/runs/{id}/trades` | Completed trades |
| GET | `/api/v1/runs/{id}/orders` | Orders |
| GET | `/api/v1/runs/{id}/fills` | Execution fills |
| GET | `/api/v1/runs/{id}/positions` | Open positions |
| GET | `/api/v1/runs/{id}/diagnostics` | Live engine state |
| GET | `/api/v1/runs/{id}/learning` | Adaptive learning state |
| GET | `/api/v1/strategies` | Available strategies |
| GET | `/api/v1/strategies/{name}/schema` | Strategy parameter schema |
| POST | `/api/v1/optimize` | Run parameter grid search |
| GET | `/api/v1/optimize` | List optimization runs |
| GET | `/api/v1/optimize/{id}` | Optimization detail |
| GET | `/api/v1/portfolio/summary` | Cross-run aggregate portfolio |
| GET | `/api/v1/ml/models` | List ML model versions |
| POST | `/api/v1/ml/train` | Train new model from OHLCV data |
| POST | `/api/v1/ml/retrain/{symbol}` | Retrain from trade outcomes |
| PUT | `/api/v1/ml/models/{id}/activate` | Activate a model version |

All endpoints (except `/health` and `/metrics`) require `X-API-Key` header when `REQUIRE_API_AUTH=true`.

---

## Project Structure

```
.
├── apps/
│   ├── api/                     # FastAPI backend
│   │   ├── routers/             # runs, strategies, orders, portfolio, optimize, ml
│   │   ├── db/                  # SQLAlchemy models, session, migrations
│   │   ├── config.py            # Pydantic Settings (all env vars)
│   │   ├── auth.py              # SHA-256 API key middleware
│   │   ├── rate_limit.py        # Per-IP rate limiting
│   │   ├── prometheus.py        # /metrics endpoint
│   │   └── main.py              # App factory + lifespan
│   └── ui/                      # Next.js 14 dashboard
│       └── src/
│           ├── app/             # Pages (runs, optimize, leaderboard, models, compare)
│           ├── components/      # Reusable UI components
│           └── lib/             # API client, types, CSV export
├── packages/
│   ├── trading/                 # Core trading engine
│   │   ├── strategies/          # DCA+RSI, Grid, RSI, MA Crossover, Breakout, Model
│   │   ├── telegram.py          # Telegram Bot API notifications
│   │   ├── ccxt_retry.py        # Exponential backoff for exchange calls
│   │   ├── ccxt_errors.py       # User-friendly CCXT error translation
│   │   ├── engines/             # Paper + Live execution engines
│   │   ├── adaptive_learning.py # Background learning pipeline
│   │   ├── adaptive_optimizer.py # Conservative parameter tuning
│   │   ├── performance_analyzer.py # Trade outcome analysis
│   │   ├── reporting.py         # Daily/weekly reports + alerts
│   │   ├── trade_journal.py     # MAE/MFE tracking, exit reasons
│   │   ├── trailing_stop.py     # Trailing stop-loss manager
│   │   ├── risk.py              # Position sizing, circuit breakers
│   │   ├── portfolio.py         # Portfolio accounting
│   │   ├── backtest.py          # Backtesting runner
│   │   ├── optimizer.py         # Parameter grid search
│   │   ├── strategy_engine.py   # Central bar-loop orchestrator
│   │   └── safety.py            # Live trading gates, graduated circuit breaker
│   ├── data/                    # Market data layer
│   │   ├── services/            # CCXT market data service
│   │   ├── indicators.py        # 10 vectorized technical indicators
│   │   ├── sentiment.py         # Fear & Greed Index client
│   │   ├── ml_features.py       # ML feature engineering
│   │   └── ml_training.py       # Model trainer
│   └── common/                  # Shared types, events, logging, metrics
├── infra/
│   ├── docker-compose.yml       # Full stack (API, UI, Postgres, Redis, Prometheus, Grafana)
│   ├── Dockerfile.api / .ui     # Multi-stage production builds
│   ├── alembic/                 # Database migrations (001-007)
│   ├── grafana/                 # Auto-provisioned dashboard
│   └── prometheus.yml           # Scrape config
├── scripts/
│   ├── train_model.py           # CLI model training
│   ├── db_maintenance.sh        # Weekly VACUUM + cleanup
│   └── install_maintenance_cron.sh
├── tests/
│   ├── unit/                    # 1500+ unit tests
│   └── integration/             # API integration tests
└── docs/                        # Architecture, risk model, strategy guide
```

---

## Safety

Live trading is **disabled by default**. Three gates must be satisfied simultaneously:

1. `ENABLE_LIVE_TRADING=true` environment variable
2. Valid `EXCHANGE_API_KEY` + `EXCHANGE_API_SECRET`
3. `LIVE_TRADING_CONFIRM_TOKEN` matching the configured value

Additional safety mechanisms:
- **Graduated Circuit Breaker** — OK → REDUCE (halve sizes) → DAILY_LIMIT (no new entries) → HALT
- **Max Daily Loss** — Configurable (default 5%), halts trading when breached
- **Max Drawdown** — Configurable (default 15%), emergency stop
- **Coinbase Min Order Validation** — Checks exchange minimum amount/cost before placing
- **CCXT Retry with Backoff** — Exponential backoff + jitter for exchange API calls
- **Max Run Duration** — Auto-stop after configurable hours (default 168h / 7 days)
- **Rollback Mechanism** — Auto-revert parameter changes if PnL drops > 5%

---

## Development

```bash
# Install dependencies
pip install uv
uv sync --all-packages

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/apps:$(pwd)/packages"

# Run tests
pytest tests/ -v --tb=short

# Type checking
mypy packages/ apps/ --ignore-missing-imports

# Frontend
cd apps/ui && npm install && npm run dev
```

---

## Database Maintenance

Weekly maintenance runs automatically via cron (if installed):

```bash
# Install the cron job (Sundays 03:00 UTC)
bash scripts/install_maintenance_cron.sh

# Or run manually
bash scripts/db_maintenance.sh
```

This performs: VACUUM ANALYZE, archive old runs (>90 days), clean orphaned data, report table sizes.

---

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the complete list with documentation.

Key settings:
- `EXCHANGE_ID` — CCXT exchange (default: `binance`, also supports `coinbase`)
- `EQUITY_SNAPSHOT_RETENTION_DAYS` — Days to keep equity data (default: 90)
- `MAX_RUN_DURATION_HOURS` — Auto-stop limit (default: 168)
- `ML_AUTO_RETRAIN` — Enable automatic model retraining (default: false)

---

## License

MIT
