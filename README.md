# AI Crypto Trading Bot

A production-grade cryptocurrency trading platform with automated strategy execution, backtesting, paper trading, and a web dashboard.

## Tech Stack

| Layer      | Technology                                        |
|------------|---------------------------------------------------|
| Backend    | Python 3.11+, FastAPI, SQLAlchemy (async), Alembic |
| Trading    | CCXT, Pandas, NumPy                               |
| Frontend   | Next.js, TypeScript, Tailwind CSS                 |
| Database   | PostgreSQL (asyncpg driver)                       |
| Cache      | Redis (optional)                                  |
| Infra      | Docker, Docker Compose, GitHub Actions CI         |
| Observability | Prometheus metrics, structured JSON logging    |

---

## Quick Start (Docker Compose)

```bash
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD at minimum
docker compose -f infra/docker-compose.yml up --build
```

The API will be available at `http://localhost:8000` and the dashboard at `http://localhost:3000`.

---

## Development Setup

### Python Backend

```bash
# Install uv (https://github.com/astral-sh/uv)
pip install uv

# Install all packages in editable mode
uv pip install -e "packages/trading[dev]" -e "packages/data[dev]" -e "packages/common[dev]" -e "apps/api[dev]"

# Copy and configure environment
cp .env.example .env

# Run database migrations
alembic -c infra/alembic/alembic.ini upgrade head

# Start the API server
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Next.js Frontend

```bash
cd apps/ui
npm install
npm run dev
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running API + PostgreSQL)
pytest tests/integration/ -v

# Full suite with coverage
pytest --cov=apps --cov=packages --cov-report=term-missing
```

---

## Project Structure

```
.
├── apps/
│   ├── api/                  # FastAPI application
│   │   ├── routers/          # Endpoint modules (runs, strategies, orders, portfolio)
│   │   ├── config.py         # Pydantic Settings (all env vars)
│   │   ├── auth.py           # API key middleware
│   │   ├── rate_limit.py     # Per-IP rate limiting
│   │   ├── prometheus.py     # /metrics endpoint
│   │   └── main.py           # App factory
│   └── ui/                   # Next.js dashboard
│       └── src/              # Pages and components
├── packages/
│   ├── trading/              # Core trading engine
│   │   ├── strategies/       # MA Crossover, RSI, Breakout, Model
│   │   ├── engines/          # Paper and live execution engines
│   │   ├── risk_manager.py   # Position sizing, circuit breakers
│   │   ├── portfolio.py      # Portfolio accounting
│   │   └── backtest.py       # Backtesting runner
│   ├── data/                 # Market data and indicators
│   │   ├── market_data.py    # CCXT data fetching
│   │   └── indicators.py     # Vectorized technical indicators
│   └── common/               # Shared types and utilities
│       ├── types.py
│       ├── events.py
│       └── logging.py
├── infra/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── migrations/           # Alembic migration scripts
├── tests/
│   ├── unit/
│   └── integration/
└── docs/                     # Architecture and strategy documentation
```

---

## Available Strategies

| Strategy         | ID            | Description                                              |
|------------------|---------------|----------------------------------------------------------|
| MA Crossover     | `ma_crossover` | Fast/slow moving average crossover signals              |
| RSI Mean Reversion | `rsi`       | Overbought/oversold RSI-based entry and exit            |
| Breakout         | `breakout`    | Price breakout above/below rolling high-low range        |
| Model (ML)       | `model`       | ML model placeholder — returns HOLD until model trained |

---

## API Endpoints

| Method | Path                   | Description                              |
|--------|------------------------|------------------------------------------|
| POST   | `/runs`                | Create and launch a backtest or paper run |
| GET    | `/runs`                | List all runs                            |
| GET    | `/runs/{id}`           | Get run details and status               |
| GET    | `/strategies`          | List available strategies                |
| POST   | `/orders`              | Place a manual order (paper/live)        |
| GET    | `/orders`              | List orders for a run                    |
| GET    | `/portfolio`           | Current portfolio snapshot               |
| GET    | `/health`              | Liveness check                           |
| GET    | `/metrics`             | Prometheus text scrape endpoint          |

Protected endpoints require `X-API-Key` header when `REQUIRE_API_AUTH=true`.

---

## Safety

Live trading is disabled by default and requires all three gates to be satisfied simultaneously:

1. `ENABLE_LIVE_TRADING=true` environment variable
2. Valid `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET`
3. `LIVE_TRADING_CONFIRM_TOKEN` matching the configured secret

Paper trading mode is always available without exchange credentials.

---

## License

MIT
