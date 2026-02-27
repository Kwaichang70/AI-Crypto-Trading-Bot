# Project: AI Crypto Trading Bot + App (MVP → Production-ready)

## 0) Rol & Output
Je bent **Claude Code** (senior full-stack + quant engineer). Bouw een complete repo met:
- Trading engine (live + paper)
- Backtesting module
- Strategy framework (rule-based + ML-ready)
- Risk management (position sizing, stops, max drawdown, circuit breakers)
- Exchange connector(s) (start met één: Kraken of Binance via CCXT)
- API server (FastAPI)
- Web UI (Next.js) voor dashboard + config + logs
- Observability (logging, metrics, alerts)
- Docker setup + duidelijke README
- Tests (unit + integration) + CI workflow

**Belangrijk:** Geen “financieel advies”. Focus op engineering, betrouwbaarheid, safety.

---

## 1) Product Scope (MVP)
### Must-have
1. **Modes**
   - Backtest mode
   - Paper trading mode (simulated fills + fees + slippage)
   - Live mode (echte orders) met strikte safety-gates

2. **Strategies (pluggable)**
   - Baseline: Moving Average Crossover
   - Baseline: RSI Mean Reversion
   - Baseline: Breakout (Donchian / ATR)
   - Framework zodat nieuwe strategies eenvoudig zijn toe te voegen

3. **Risk & Safety**
   - Max open positions
   - Max leverage = 1 (spot-only voor MVP)
   - Max daily loss / max drawdown → trading stopt
   - Per-trade risk cap (bijv. 0.5–1% equity)
   - Slippage + fee model
   - Cooldown na verliesreeks
   - Circuit breaker bij API errors / extreme volatility

4. **Data**
   - OHLCV candles (1m, 5m, 1h) via exchange (en caching lokaal)
   - Trades + orders opslaan in DB (Postgres)
   - Config/versioning: elke run krijgt een “run_id”

5. **UI**
   - Dashboard: equity curve, open positions, PnL, drawdown, trades
   - Strategie-config: parameters aanpassen + starten/stoppen
   - Logs & alerts pagina

6. **API (FastAPI)**
   - Start/stop strategy run
   - Get status
   - Get trades/orders
   - Get metrics (PnL, drawdown, exposure)
   - Update config (validated)

7. **Deployment**
   - Docker Compose: api + ui + postgres + redis (optioneel)
   - `.env.example` voor keys + config
   - README met setup en run instructions

### Nice-to-have (na MVP)
- Multi-exchange, multi-asset portfolio optimizer
- Websocket market data
- ML model training pipeline
- Walk-forward optimization
- Telegram/Discord alerts

---

## 2) Tech Stack (voorkeur)
- **Python 3.11+**
- **FastAPI** (API)
- **PostgreSQL** (persistency) via SQLAlchemy
- **Redis** (optioneel: caching + job queue)
- **CCXT** (exchange abstraction)
- **Pandas / NumPy** (data)
- **Backtesting.py** of eigen engine (kies pragmatisch)
- **Next.js (TypeScript)** + minimal UI (Tailwind)
- **Docker / Docker Compose**
- **Pytest** + GitHub Actions CI

---

## 3) Repo structuur (verwacht)
Maak een repo met minimaal:


/apps
/api # FastAPI
/ui # Next.js
/packages
/trading # engine, strategies, risk, execution
/data # data fetching, caching, indicators
/common # shared types, utils
/infra
docker-compose.yml
migrations/
/docs
architecture.md
risk-model.md
strategy-guide.md
README.md
.env.example


---

## 4) Architectuur-eisen
### Core componenten
1. **MarketDataService**
   - Fetch/caching candles
   - Normalise timestamps/timezones
   - Rate limiting + retries
   - Store raw + derived data

2. **StrategyEngine**
   - Strategy interface: `on_bar(data) -> signals`
   - Signals: `BUY/SELL/HOLD`, target position, confidence (float)
   - Parameter schema + validation

3. **ExecutionEngine**
   - Paper: simulated fills, partial fills, latency, slippage, fees
   - Live: order placement (limit/market), idempotency keys
   - Order state machine: NEW → PARTIAL → FILLED/CANCELED/REJECTED

4. **RiskManager**
   - Pre-trade checks (exposure, drawdown, daily loss, max size)
   - Position sizing (fixed fractional of equity)
   - Stop-loss / take-profit / trailing stop (configurable)
   - Kill-switch

5. **PortfolioAccounting**
   - Equity curve
   - Realized/unrealized PnL
   - Fees
   - Drawdown
   - Exposure by asset

6. **Persistence**
   - Tables: runs, configs, candles cache index, orders, fills, positions, metrics snapshots, events/logs

7. **Observability**
   - Structured JSON logs
   - Prometheus metrics endpoint (optioneel)
   - Alerts: “trading halted”, “API errors”, “daily loss hit”

---

## 5) Trading logic (MVP details)
### Timeframes
- Default: 5m candles
- Configurable per strategy

### Order types
- MVP: market orders + optional limit orders
- Spot only

### Fee/slippage
- Configure: taker fee %, maker fee %, slippage bps
- Paper mode moet dit meenemen

---

## 6) “AI” component (realistisch & veilig)
**Geen hype**: MVP gebruikt vooral rule-based strategies, maar maak de code **ML-ready**:
- Feature pipeline module (RSI, MACD, ATR, returns, volatility)
- “ModelStrategy” placeholder:
  - Inference interface: `predict(features) -> signal/confidence`
  - Start met simpel model: logistic regression of gradient boosting (sklearn)
  - Training buiten live loop (offline script)
- Duidelijke scheiding: training ≠ trading runtime
- Model versioning (file + metadata in DB)

---

## 7) Backtesting eisen
- Backtest op historische candles
- Metrics:
  - CAGR, Sharpe (approx), max drawdown, win rate, profit factor
  - Exposure, turnover, fees
- Output:
  - JSON report + opgeslagen in DB
  - Equity curve + trades list
- Walk-forward (optioneel), maar architectuur zo dat het later kan

---

## 8) Security & Safety gates (hard requirements)
1. Live trading staat standaard **UIT**
2. Live trading vereist:
   - `ENABLE_LIVE_TRADING=true`
   - API keys aanwezig
   - Extra “confirm token” in env (simple safeguard)
3. Secret handling:
   - nooit keys loggen
   - `.env` nooit committen
4. Rate limiting & retries met exponential backoff
5. Graceful shutdown: posities syncen, orders status checken
6. Idempotency voor order placement (waar mogelijk)

---

## 9) API contract (indicatief)
### Endpoints
- `POST /runs/start` (mode=backtest|paper|live, strategy, params, symbols, timeframe)
- `POST /runs/{run_id}/stop`
- `GET /runs/{run_id}/status`
- `GET /runs/{run_id}/trades`
- `GET /runs/{run_id}/orders`
- `GET /runs/{run_id}/metrics`
- `GET /health`
- `GET /metrics` (optioneel)

Gebruik Pydantic models met strikte validatie.

---

## 10) UI requirements (minimal maar bruikbaar)
- Home dashboard: status + equity + drawdown + open positions
- Runs page: lijst runs, detail run
- Config page: strategy dropdown + form fields
- Logs page: filter op run_id/level

---

## 11) Deliverables
1. Werkende Docker Compose: `docker compose up`
2. `make` of `justfile` scripts (optioneel) voor:
   - lint/test
   - start api/ui
   - run backtest
3. README met:
   - Setup
   - Paper trading demo
   - Backtest demo
   - Live trading safety checklist
4. Docs:
   - `docs/architecture.md` met diagram (ASCII/mermaid)
   - `docs/risk-model.md`
   - `docs/strategy-guide.md` met voorbeeld van nieuwe strategy

---

## 12) Kwaliteitslat
- Code: type hints, duidelijke modules, geen “god files”
- Tests: minimaal 10 unit tests + 3 integration tests (paper execution + risk stop + db persistence)
- Foutafhandeling: geen silent fails
- Deterministische backtests (seed control)
- Heldere logging per run

---

## 13) Werkwijze (hoe je moet uitvoeren)
1. Genereer repo structuur en basis tooling (Docker, env, CI)
2. Bouw trading core: data → strategy → risk → execution → persistence
3. Voeg backtesting toe
4. Voeg paper trading toe
5. Voeg UI toe
6. Voeg live trading toe achter safety gates
7. Voeg docs + tests toe
8. Lever een korte “Quickstart” en “MVP demo flow”

---

## 14) Niet doen
- Geen beloftes over winstgevendheid
- Geen high-frequency / latency-sensitive claims
- Geen leverage/derivatives in MVP
- Geen productie-sleutels hardcoded