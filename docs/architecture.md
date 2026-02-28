# AI Crypto Trading Bot -- Architecture

## Overview

The AI Crypto Trading Bot is a production-grade cryptocurrency trading platform
that supports automated strategy execution across three run modes: backtesting
on historical data, paper trading with simulated fills, and live trading with
real order placement via CCXT. It consists of a Python backend (FastAPI), a
Next.js dashboard frontend, PostgreSQL for persistence, and Redis for caching.

Key capabilities:

- Pluggable strategy interface with confidence scoring
- Deterministic backtesting with look-ahead bias prevention
- Paper trading with slippage, fee, and latency simulation
- Live trading through CCXT with a 3-layer safety gate
- 6-stage pre-trade risk gating and fixed-fractional position sizing
- Real-time portfolio accounting with equity curve tracking
- Full order state machine with idempotency enforcement
- RESTful API for run management, order queries, and portfolio monitoring
- Web dashboard for backtest results and live monitoring

## System Architecture

```
+-------------------------------------------------------------+
|                        Next.js Dashboard                     |
|                    (TypeScript, Tailwind CSS)                 |
|    [ Backtest UI ] [ Paper Monitor ] [ Equity Chart ]        |
+------------------------------+------------------------------+
                               |
                          HTTP / REST
                               |
+------------------------------v------------------------------+
|                        FastAPI Backend                        |
|  /health   /api/v1/runs   /api/v1/orders   /api/v1/portfolio |
|  /api/v1/strategies                                          |
+------+-----------+-----------+-----------+-------------------+
       |           |           |           |
       v           v           v           v
  +---------+ +---------+ +--------+ +----------+
  |Strategy | |Execution| | Risk   | | Market   |
  | Engine  | | Engine  | |Manager | | Data     |
  |(orchestr| |(paper/  | |(6-stage| | Service  |
  | ator)   | | live)   | | gate)  | | (CCXT)   |
  +----+----+ +----+----+ +---+----+ +----+-----+
       |           |           |           |
       v           v           v           v
  +---------+ +---------+ +---------+ +---------+
  |Portfolio | | Order   | | Circuit | | L1 Cache|
  |Accounting| | State   | | Breaker | | (TTL)   |
  |(equity,  | | Machine | | + Kill  | |         |
  | PnL,    | |         | | Switch  | |         |
  | drawdown)| |         | |         | |         |
  +---------+ +---------+ +---------+ +---------+
       |           |                       |
       v           v                       v
  +--------------------------------------------------+
  |              PostgreSQL 16                         |
  |  runs | orders | fills | trades | equity_snapshots |
  |  signals                                           |
  +--------------------------------------------------+
  |              Redis 7 (optional)                    |
  |  Cache layer, future job queue                     |
  +--------------------------------------------------+
```

## Component Descriptions

### 1. StrategyEngine (Orchestrator)

**File:** `packages/trading/strategy_engine.py`

The central orchestrator that ties together all trading core components for a
complete trading run. It manages the full lifecycle from `start(run_id)` through
bar-by-bar processing to `stop()`.

Responsibilities:

- Run lifecycle management with state machine (IDLE -> STARTING -> RUNNING -> STOPPING -> STOPPED)
- Bar-by-bar processing: fetch candles, feed to strategies, collect signals, execute, route fills
- Multi-strategy support: runs multiple strategies in parallel, each producing independent signals
- Fill routing: routes fills from execution engine to PortfolioAccounting and RiskManager
- Warmup management: skips strategy calls for the first N bars to allow indicator convergence
- Rolling bar window management for paper/live modes with configurable max history size

### 2. BaseStrategy (Strategy Interface)

**File:** `packages/trading/strategy.py`

Abstract base class defining the pluggable strategy contract. Every concrete
strategy implements `on_bar(bars) -> list[Signal]`, the single hook called on
each candle.

Key interface:

- `on_bar(bars)` -- process OHLCV history, return zero or more Signals
- `on_start(run_id)` / `on_stop()` -- lifecycle hooks
- `parameter_schema()` -- JSON Schema for API validation
- `min_bars_required` -- minimum history needed before producing signals
- `metadata` -- class-level StrategyMetadata for introspection

Strategies are stateless with respect to order placement. They emit Signals;
the ExecutionEngine and RiskManager handle the rest.

### 3. ExecutionEngine (Paper + Live)

**Files:** `packages/trading/execution.py`, `packages/trading/engines/paper.py`, `packages/trading/engines/live.py`

The execution layer converts Signals into Orders and drives them through the
order state machine. Two concrete implementations exist:

**PaperExecutionEngine** -- Simulated fills with:
- Configurable slippage model (basis-point spread)
- Fee calculation using risk manager parameters
- Full order state machine compliance
- Resting limit order support (fills when price crosses)
- Position tracking keyed by symbol

**LiveExecutionEngine** -- Real exchange orders via CCXT with:
- Explicit `enable_live_trading` gate (outermost safety layer)
- CCXT async exchange instance integration
- Error handling and structured logging
- All orders pass through RiskManager pre-trade check

### 4. RiskManager (Pre-Trade Gating)

**Files:** `packages/trading/risk.py`, `packages/trading/risk_manager.py`

Synchronous pre-trade risk gating and position sizing. The DefaultRiskManager
runs a 6-stage check pipeline that collects all violations before issuing a
verdict. See the [Risk Model](risk-model.md) for full details.

### 5. PortfolioAccounting

**File:** `packages/trading/portfolio.py`

Concrete class serving as the single source of truth for portfolio state within
a trading run. Updated on every fill event and on every bar (market price updates).

Tracks:

- Cash balance and total equity (cash + open position value)
- Peak equity and drawdown
- Daily PnL with automatic day-boundary reset
- Per-symbol position snapshots with unrealised PnL
- Equity curve as (timestamp, equity) time series
- Completed trade history with win/loss counters
- All-in cost basis including fees

### 6. MarketDataService (CCXT)

**Files:** `packages/data/market_data.py`, `packages/data/services/ccxt_market_data.py`

Async OHLCV candle fetching, normalisation, and caching via CCXT.

Architecture:

- Exchange access via `ccxt.async_support` (HTTP, not WebSocket)
- Token-bucket rate limiting with exponential backoff on 429 errors
- L1 in-process TTL cache per (symbol, timeframe, since, limit)
- Transparent pagination for `fetch_ohlcv_range`
- Timestamp normalisation to UTC, prices to Decimal
- Deduplication of overlapping page boundaries

### 7. Safety Layer

**File:** `packages/trading/safety.py`

Two independent safety mechanisms:

- **LiveTradingGate** -- 3-layer activation gate (environment flag + API keys + confirmation token)
- **CircuitBreaker** -- emergency stop on daily loss, drawdown, or consecutive loss thresholds

Both are fail-closed: ambiguity blocks orders. Breakers never auto-reset.

## Data Flow

How a bar flows through the system during a trading run:

```
1. Market Data
   MarketDataService.fetch_ohlcv() or .get_latest_bar()
          |
          v
2. Price Update
   StrategyEngine._update_engine_prices(current_bars)
   PortfolioAccounting.update_market_prices(prices)
          |
          v
3. Risk Check
   RiskManager.tick_cooldown()
   If LIVE mode: check kill_switch -- skip bar if active
          |
          v
4. Strategy Invocation
   For each strategy:
     signals = strategy.on_bar(history_by_symbol[symbol])
          |
          v
5. Signal Processing
   For each signal:
     orders = ExecutionEngine.process_signal(signal)
       internally: RiskManager.pre_trade_check(order, ...)
       internally: build Order, submit, generate Fill
          |
          v
6. Fill Routing
   For each fill:
     PortfolioAccounting.update_position(fill, current_price)
     If SELL fill closes position:
       RiskManager.update_after_fill(realised_pnl, is_loss)
          |
          v
7. Resting Order Check (paper mode)
   ExecutionEngine.check_resting_orders(symbol, price)
   Route any triggered fills to portfolio + risk manager
          |
          v
8. Equity Curve Update
   PortfolioAccounting records (timestamp, equity) point
   Peak equity updated if current > previous peak
```

## Run Modes

The system supports three run modes, each handled by the same StrategyEngine
with different component wiring:

### BACKTEST

- Bars are pre-fetched and passed to `StrategyEngine.run_backtest(bars_by_symbol)`
- Uses PaperExecutionEngine for simulated fills
- Growing window prevents look-ahead bias (bars[0:i+1] on step i)
- Deterministic execution via optional random seed
- No market data service calls (stub provided)
- Warmup period skips strategy calls but updates prices
- BacktestRunner computes all metrics (CAGR, Sharpe, Sortino, Calmar, etc.)

### PAPER

- Uses PaperExecutionEngine with real market data
- Polls MarketDataService on each timeframe interval
- Rolling bar window with configurable max history (default 500)
- Bar deduplication by timestamp
- Strategies see realistic data but fills are simulated
- Slippage and fees modeled per RiskParameters

### LIVE

- Uses LiveExecutionEngine with real CCXT exchange connection
- Requires all 3 safety gate layers to pass before run creation
- Kill switch checked on every bar -- bar skipped entirely if active
- All orders pass through 6-stage risk check before submission
- CircuitBreaker can halt trading on threshold breach
- Same polling loop as PAPER mode

### What Changes Between Modes

| Aspect               | BACKTEST                  | PAPER                    | LIVE                      |
|----------------------|---------------------------|--------------------------|---------------------------|
| Market data source   | Pre-fetched bars          | CCXT (real-time)         | CCXT (real-time)          |
| Execution engine     | PaperExecutionEngine      | PaperExecutionEngine     | LiveExecutionEngine       |
| Fills                | Simulated, immediate      | Simulated, immediate     | Real, exchange-reported   |
| Safety gate required | No                        | No                       | Yes (3-layer)             |
| Kill switch enforced | No                        | No (signals continue)    | Yes (skips bar)           |
| Deterministic        | Yes (with seed)           | No                       | No                        |
| Warmup               | Growing window slice      | Pre-fetch via API        | Pre-fetch via API         |

## Dependency Graph

Package dependencies flow inward. Outer layers depend on inner layers, never
the reverse.

```
                    +-------------------+
                    |   apps/api        |  <-- FastAPI endpoints, DB session
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+     +-------------v-----------+
    | packages/trading  |     |   packages/data         |
    | strategy_engine   |     |   market_data (abstract) |
    | execution (base)  |---->|   ccxt_market_data       |
    | engines/paper     |     +-------------+-----------+
    | engines/live      |                   |
    | risk / risk_mgr   |                   |
    | portfolio         |                   |
    | safety            |                   |
    | backtest          |                   |
    | metrics           |                   |
    | models            |                   |
    | strategy (base)   |                   |
    +--------+----------+                   |
             |                              |
             +-----------+------------------+
                         |
               +---------v---------+
               | packages/common   |
               | types.py (enums)  |
               | models.py (OHLCV) |
               | config.py         |
               +-------------------+
```

Rules:

- `packages/common` has zero internal dependencies
- `packages/data` depends only on `packages/common`
- `packages/trading` depends on `packages/common` and `packages/data`
- `apps/api` depends on all packages (it is the composition root)
- `apps/ui` depends only on the API (HTTP boundary)

## Database Schema

Six tables in PostgreSQL, all under the `public` schema. All monetary columns
use `Numeric(20, 8)` -- never Float.

```
+------------------+       +------------------+       +------------------+
|     runs         |       |    orders        |       |     fills        |
+------------------+       +------------------+       +------------------+
| id (UUID, PK)    |<------| run_id (FK)      |       | id (UUID, PK)    |
| run_mode         |       | id (UUID, PK)    |<------| order_id (FK)    |
| status           |       | client_order_id  |       | symbol           |
| config (JSONB)   |       | symbol           |       | side             |
| started_at       |       | side             |       | quantity          |
| stopped_at       |       | order_type       |       | price            |
| created_at       |       | quantity          |       | fee              |
| updated_at       |       | price            |       | fee_currency     |
+------------------+       | status           |       | is_maker         |
        |                  | filled_quantity   |       | executed_at      |
        |                  | average_fill_price|       +------------------+
        |                  | exchange_order_id |
        |                  | created_at       |
        |                  | updated_at       |
        |                  +------------------+
        |
        +---------------+------------------+
        |               |                  |
+-------v--------+ +---v-----------+ +----v-----------+
|    trades      | | equity_       | |   signals      |
+----------------+ | snapshots     | +----------------+
| id (UUID, PK) | +---------------+ | id (BIGSERIAL) |
| run_id (FK)   | | id (BIGSERIAL)| | run_id (FK)    |
| symbol        | | run_id (FK)   | | strategy_id    |
| side          | | equity        | | symbol         |
| entry_price   | | cash          | | direction      |
| exit_price    | | unrealised_pnl| | target_position|
| quantity      | | realised_pnl  | | confidence     |
| realised_pnl  | | drawdown_pct  | | metadata(JSONB)|
| total_fees    | | bar_index     | | generated_at   |
| entry_at      | | timestamp     | +----------------+
| exit_at       | +---------------+
| strategy_id   |
+----------------+
```

Key index strategies:

- `orders.client_order_id` UNIQUE -- idempotency at the DB layer
- `orders.status` PARTIAL index on active statuses only -- execution engine hot path
- `equity_snapshots(run_id, bar_index)` UNIQUE -- prevents duplicate snapshots
- `trades(run_id, symbol)` -- per-symbol PnL queries within a run
- `signals(strategy_id)` -- cross-run strategy attribution

## API Surface

All versioned endpoints live under `/api/v1`. The health endpoint is unversioned.

### Observability

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | Service health check (Docker, LB)    |

### Run Management

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| POST   | `/api/v1/runs`          | Start a new trading run              |
| GET    | `/api/v1/runs`          | List all runs (paginated)            |
| GET    | `/api/v1/runs/{run_id}` | Get a single run's details           |
| DELETE | `/api/v1/runs/{run_id}` | Stop a running run                   |

### Orders and Fills

| Method | Path                                     | Description                          |
|--------|------------------------------------------|--------------------------------------|
| GET    | `/api/v1/runs/{run_id}/orders`           | List orders for a run                |
| GET    | `/api/v1/runs/{run_id}/orders/{order_id}`| Get a single order                   |
| GET    | `/api/v1/runs/{run_id}/fills`            | List fills for a run                 |

### Portfolio

| Method | Path                                 | Description                          |
|--------|--------------------------------------|--------------------------------------|
| GET    | `/api/v1/runs/{run_id}/portfolio`    | Portfolio summary snapshot           |
| GET    | `/api/v1/runs/{run_id}/equity-curve` | Equity curve time series             |
| GET    | `/api/v1/runs/{run_id}/trades`       | Completed round-trip trades          |
| GET    | `/api/v1/runs/{run_id}/positions`    | Current open positions               |

### Strategy Discovery

| Method | Path                                    | Description                          |
|--------|-----------------------------------------|--------------------------------------|
| GET    | `/api/v1/strategies`                    | List all available strategies        |
| GET    | `/api/v1/strategies/{name}/schema`      | Get parameter schema for a strategy  |

## Security Model

### Three-Layer Live Trading Gate

Live trading requires all three layers to pass. Checked by LiveExecutionEngine
on startup and by the `/runs` endpoint before allowing a LIVE run.

```
Layer 1: Environment     ENABLE_LIVE_TRADING=true in env/.env
         |
         v (must pass)
Layer 2: API Keys        EXCHANGE_API_KEY and EXCHANGE_API_SECRET are non-empty
         |
         v (must pass)
Layer 3: Confirmation    Runtime confirm_token matches LIVE_TRADING_CONFIRM_TOKEN
         |
         v (all passed)
         Live trading enabled
```

Design principles:

- Fail-closed: any ambiguity defaults to blocking
- Layer 1 is OFF by default -- must be explicitly enabled
- API keys are stored as SecretStr -- never serialised to logs
- Confirmation token must be provided at runtime per run creation

### Circuit Breaker

Emergency stop mechanism evaluated on every bar tick:

- Triggers on: daily loss > threshold, drawdown > threshold, consecutive losses > threshold
- Once tripped, stays open until operator calls `reset()` -- never auto-resets
- Trip count preserved for audit trail
- Logs at CRITICAL severity with `alert="TRADING_HALTED"`

### Kill Switch

One-way latch on the RiskManager:

- `trigger_kill_switch(reason)` -- immediately blocks all new orders
- `reset_kill_switch()` -- requires explicit operator action
- In LIVE mode, the StrategyEngine skips entire bars when kill switch is active
- In PAPER mode, strategies continue executing (for monitoring) but orders are blocked

### Additional Security Measures

- PostgreSQL password is required (Docker Compose enforces via `?` modifier)
- Redis password support configured
- CORS restricted to configured origins (default: `localhost:3000`)
- OpenAPI/Swagger docs disabled in production (`debug=False`)
- Request timing middleware for latency monitoring
- All monetary values use Decimal -- never floating-point

## Technology Choices

| Technology  | Role                | Why                                                      |
|-------------|---------------------|----------------------------------------------------------|
| Python 3.11 | Backend runtime     | Async/await, type hints, ecosystem for finance/ML        |
| FastAPI     | API framework       | Async-native, auto OpenAPI docs, Pydantic integration    |
| SQLAlchemy 2| ORM                 | Async support, type-safe queries, Alembic migrations     |
| CCXT        | Exchange connector  | 100+ exchanges, unified API, async support               |
| Pydantic v2 | Domain models       | Validation, serialization, JSON Schema generation        |
| structlog   | Logging             | Structured JSON logs, context binding, production-ready  |
| PostgreSQL  | Primary database    | JSONB for flexible config, Numeric for money, ACID       |
| Redis       | Cache / queue       | L2 cache for candle data, future job queue for backtests  |
| Next.js     | Dashboard frontend  | Server components, TypeScript, Tailwind for rapid UI     |
| Docker      | Containerisation    | Reproducible builds, health checks, resource limits      |
| Pandas/NumPy| Data analysis       | Vectorised operations for indicators and metrics         |

## Infrastructure

Docker Compose orchestrates four services:

```
  postgres:16-alpine    port 5432   512MB limit   health: pg_isready
  redis:7-alpine        port 6379   192MB limit   health: redis-cli ping
  api (FastAPI)          port 8000   768MB limit   health: curl /health
  ui (Next.js)           port 3000   512MB limit   health: wget /api/health
```

Service dependencies: `ui -> api -> postgres + redis`

The API service passes all trading safety gate environment variables
(`ENABLE_LIVE_TRADING`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`,
`LIVE_TRADING_CONFIRM_TOKEN`) through Docker Compose, with live trading
defaulting to `false`.
