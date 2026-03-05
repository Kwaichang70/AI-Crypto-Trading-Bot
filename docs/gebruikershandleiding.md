# Gebruikershandleiding — AI Crypto Trading Bot

> Tweetalige handleiding: Nederlands met Engelse technische termen.

---

## Inhoudsopgave

1. [Snel Starten](#1-snel-starten)
2. [Coinbase Configuratie](#2-coinbase-configuratie)
3. [Dashboard Overzicht](#3-dashboard-overzicht)
4. [Strategieen](#4-strategieen)
5. [Backtesting](#5-backtesting)
6. [Paper Trading](#6-paper-trading)
7. [Live Trading](#7-live-trading)
8. [ML Model Training](#8-ml-model-training)
9. [API Reference](#9-api-reference)
10. [Environment Variabelen](#10-environment-variabelen)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Snel Starten

### Vereisten

- Docker Desktop (Windows/Mac/Linux)
- Git
- Een Coinbase account met API keys (zie sectie 2)

### Installatie

```bash
# Clone het project
git clone <repo-url>
cd "AI Crypto Trading Bot"

# Kopieer en configureer environment variabelen
cp .env.example .env
# Bewerk .env met je eigen waarden (zie sectie 2 en 10)

# Start alle services
cd infra
docker compose --env-file ../.env up -d
```

### Services controleren

```bash
docker compose --env-file ../.env ps
```

Alle 4 services moeten `healthy` zijn:

| Service | Poort | Functie |
|---------|-------|---------|
| **postgres** | 5432 | Database (PostgreSQL 16) |
| **redis** | 6379 | Cache (Redis 7) |
| **api** | 8000 | Backend (FastAPI) |
| **ui** | 3000 | Dashboard (Next.js) |

### Openen

- **Dashboard:** http://localhost:3000
- **API Health:** http://localhost:8000/health
- **API Docs:** http://localhost:8000/docs (alleen als `DEBUG=true`)

---

## 2. Coinbase Configuratie

### API Key Aanmaken

1. Ga naar [Coinbase Developer Platform](https://portal.cdp.coinbase.com/projects/api-keys)
2. Klik **Create API Key**
3. Kies permissies: **`can_view`** + **`can_trade`** (NOOIT `can_transfer`)
4. Kies **ECDSA** als signature algorithm in Advanced Settings
5. Download het JSON bestand of kopieer de velden

### Wat je krijgt

Coinbase geeft je twee waarden:

| Veld | Formaat | Voorbeeld |
|------|---------|-----------|
| **API Key Name** | `organizations/{org_id}/apiKeys/{key_id}` | `organizations/a3f6d5.../apiKeys/b7e8c9...` |
| **Private Key** | EC Private Key (PEM) | `-----BEGIN EC PRIVATE KEY-----\nMIGk...` |

### Invullen in `.env`

```env
EXCHANGE_ID=coinbase

# Plak de volledige API Key Name (inclusief "organizations/")
EXCHANGE_API_KEY=organizations/a3f6d5xx-xxxx-xxxx/apiKeys/b7e8c9xx-xxxx-xxxx

# Plak de Private Key op een regel, vervang enters door \n
EXCHANGE_API_SECRET=-----BEGIN EC PRIVATE KEY-----\nMIGkAgEBBDA...\n-----END EC PRIVATE KEY-----

# Leeg laten voor nieuwe CDP keys (alleen nodig voor legacy keys)
EXCHANGE_API_PASSPHRASE=
```

### Ed25519 Keys (nieuw formaat sinds feb 2025)

Als je een Ed25519 key hebt (64 bytes, standaard sinds 2025), hoef je de PEM headers er **niet** uit te halen. De bot doet dit automatisch.

### Coinbase Beperkingen

| Feature | Coinbase | Opmerking |
|---------|----------|-----------|
| Trading pairs | BTC/**USD**, ETH/**USDC** | Geen USDT paren |
| Timeframes | 1m, 5m, 15m, 1h, 6h, 1d | Geen 3m, 30m, 4h, 1w |
| Candles per request | Max 300 | Bot handelt dit automatisch af |
| Rate limit | 30 req/s (private) | CCXT rate limiter ingebouwd |

---

## 3. Dashboard Overzicht

### Home Page (/)

- **Health indicator** — Groene stip als de API bereikbaar is
- **Portfolio kaarten** — Totaal Realized PnL, Win Rate, actieve runs, errors
- **Recente runs** — Tabel met de laatste runs en hun status

### Runs Page (/runs)

- **Filter pills** — Filter op mode (backtest/paper/live) en status (running/stopped/error)
- **Sorteerbare kolommen** — Return %, Trades, Sharpe ratio
- **Paginatie** — Server-side, 25 runs per pagina
- **Klik op een rij** om naar de run detail pagina te gaan

### Run Detail (/runs/{id})

- **Metrics kaarten** — Sharpe, Sortino, Calmar, Profit Factor, Exposure, CAGR, Max Drawdown
- **Tabs:**
  - **Equity Curve** — Lijndiagram van je portfolio waarde over tijd
  - **Trades** — Alle gesloten posities met entry/exit prijs, PnL, fees
  - **Orders** — Alle geplaatste orders (status, type, prijs, hoeveelheid)
  - **Fills** — Individuele trade executions
  - **Positions** — Open posities op het moment dat de run stopte

### Strategies Page (/strategies)

- Overzicht van alle beschikbare strategieen met parameters en beschrijvingen

---

## 4. Strategieen

### MA Crossover (`ma_crossover`)

Trend-following strategie gebaseerd op twee Simple Moving Averages.

| Parameter | Standaard | Bereik | Beschrijving |
|-----------|-----------|--------|--------------|
| `fast_period` | 10 | 2-500 | Snelle SMA venster |
| `slow_period` | 50 | 3-2000 | Langzame SMA venster |

**Signaal:** BUY wanneer fast SMA kruist boven slow SMA, SELL wanneer het omgekeerde.

### RSI Mean Reversion (`rsi_mean_reversion`)

Mean reversion strategie gebaseerd op de Relative Strength Index.

| Parameter | Standaard | Bereik | Beschrijving |
|-----------|-----------|--------|--------------|
| `rsi_period` | 14 | 2-100 | RSI lookback periode |
| `oversold` | 30 | 1-49 | Oversold drempel (BUY signaal) |
| `overbought` | 70 | 51-99 | Overbought drempel (SELL signaal) |

**Signaal:** BUY wanneer RSI < oversold, SELL wanneer RSI > overbought.

### Breakout (`breakout`)

Donchian Channel breakout strategie met ATR-gebaseerde stop-loss.

| Parameter | Standaard | Bereik | Beschrijving |
|-----------|-----------|--------|--------------|
| `period` | 20 | 5-200 | Lookback venster voor high/low kanaal |
| `atr_multiplier` | 2.0 | 0.5-5.0 | ATR vermenigvuldiger voor stop-loss |

**Signaal:** BUY bij doorbraak boven het kanaal, SELL bij doorbraak onder het kanaal.

### Model Strategy (`model_strategy`)

ML-aangedreven strategie met een getraind RandomForest classifier.

Geeft standaard HOLD terug totdat een model is getraind (zie sectie 8).

---

## 5. Backtesting

Een backtest simuleert een strategie op historische data. Resultaten zijn direct beschikbaar.

### Via het Dashboard

1. Ga naar **Runs** > **New Run**
2. Kies een strategie en parameters
3. Selecteer mode **Backtest**
4. Stel start- en einddatum in
5. Klik **Start Run**

### Via de API

```bash
curl -X POST "http://localhost:8000/api/v1/runs" \
  -H "X-API-Key: <jouw-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "strategyName": "breakout",
    "mode": "backtest",
    "symbols": ["BTC/USD"],
    "timeframe": "1h",
    "initial_capital": "5000",
    "backtest_start": "2026-02-01T00:00:00Z",
    "backtest_end": "2026-03-01T00:00:00Z",
    "strategy_params": {
      "period": 20,
      "atr_multiplier": 2.0
    }
  }'
```

### Resultaten Lezen

De response bevat `backtestMetrics` met:

| Metric | Beschrijving | Goede waarde |
|--------|--------------|--------------|
| `totalReturnPct` | Totaal rendement | > 0 (positief) |
| `sharpeRatio` | Risico-gecorrigeerd rendement | > 1.0 is goed, > 2.0 is excellent |
| `sortinoRatio` | Downside risk-gecorrigeerd | > 1.5 is goed |
| `maxDrawdownPct` | Maximale waardedaling | < 15% is acceptabel |
| `winRate` | Percentage winnende trades | > 40% voor trend-following |
| `profitFactor` | Bruto winst / bruto verlies | > 1.5 is goed |
| `totalTrades` | Aantal gesloten trades | Meer = betrouwbaarder statistiek |
| `exposurePct` | % van de tijd in een positie | Hangt af van strategie |

---

## 6. Paper Trading

Paper trading simuleert real-time handelen met echte marktdata, maar zonder echte orders te plaatsen. Ideaal om een strategie te testen voordat je echt geld inzet.

### Starten

```bash
curl -X POST "http://localhost:8000/api/v1/runs" \
  -H "X-API-Key: <jouw-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "strategyName": "breakout",
    "mode": "paper",
    "symbols": ["ETH/USD"],
    "timeframe": "1h",
    "initial_capital": "5000"
  }'
```

### Hoe het werkt

- De bot draait als **background task** op de server
- Elke candle interval (bijv. elk uur bij `1h`) worden nieuwe marktdata opgehaald
- De strategie genereert BUY/SELL/HOLD signalen
- Orders worden gesimuleerd met realistisch slippage en fees
- Equity curve en trades worden opgeslagen in de database

### Monitoren

- **Dashboard:** Ga naar de run detail pagina en bekijk de equity curve en trades
- **API:** `GET /api/v1/runs/{run_id}` voor status
- **Trades:** `GET /api/v1/runs/{run_id}/trades` voor gesloten posities

### Stoppen

```bash
curl -X DELETE "http://localhost:8000/api/v1/runs/{run_id}" \
  -H "X-API-Key: <jouw-api-key>"
```

Of via het dashboard: klik op de run en gebruik de stop-knop.

### Belangrijk

- Paper runs overleven een **server herstart** niet (ze draaien als asyncio tasks)
- Gebruik **niet** `--reload` met uvicorn — dit doodt background tasks bij elke file change
- De equity curve en trades worden pas naar de database geschreven wanneer de run stopt

---

## 7. Live Trading

> **WAARSCHUWING:** Live trading plaatst echte orders met echt geld. Zorg dat je paper trading hebt getest voordat je live gaat.

### Drie Safety Gates

De bot heeft een 3-laags beveiligingssysteem. **Alle drie** moeten actief zijn:

| Gate | Environment Variabele | Beschrijving |
|------|----------------------|--------------|
| **1. Master Switch** | `ENABLE_LIVE_TRADING=true` | Staat standaard op `false` |
| **2. API Credentials** | `EXCHANGE_API_KEY` + `EXCHANGE_API_SECRET` | Moeten niet-leeg zijn |
| **3. Confirm Token** | `LIVE_TRADING_CONFIRM_TOKEN` | Moet meegegeven worden in elk request |

### Configuratie

Voeg toe aan je `.env`:

```env
# Gate 1: Master switch
ENABLE_LIVE_TRADING=true

# Gate 2: Al geconfigureerd (zie sectie 2)

# Gate 3: Genereer een random token
LIVE_TRADING_CONFIRM_TOKEN=<genereer met: openssl rand -hex 32>
```

Herstart de API na wijzigingen (settings zijn gecacht):

```bash
cd infra && docker compose --env-file ../.env restart api
```

### Live Run Starten

```bash
curl -X POST "http://localhost:8000/api/v1/runs" \
  -H "X-API-Key: <jouw-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "strategyName": "breakout",
    "mode": "live",
    "symbols": ["ETH/USD"],
    "timeframe": "1h",
    "initial_capital": "5000",
    "confirm_token": "<jouw-live-trading-confirm-token>"
  }'
```

### Veiligheidsmaatregelen

- **Circuit Breaker:** Stopt automatisch als dagverlies > 5% of drawdown > 15%
- **Per-trade risk:** Maximaal 1% van je kapitaal per trade (standaard)
- **Max posities:** Maximaal 3 gelijktijdige open posities
- **Spot-only:** Geen leverage of derivatives
- **Geen withdrawals:** De bot kan NOOIT geld opnemen van je exchange

### Live Run Stoppen

```bash
curl -X DELETE "http://localhost:8000/api/v1/runs/{run_id}" \
  -H "X-API-Key: <jouw-api-key>"
```

Alle openstaande orders worden automatisch gecanceld bij het stoppen.

---

## 8. ML Model Training

De `model_strategy` gebruikt een RandomForest classifier om BUY/SELL/HOLD signalen te voorspellen. Voordat deze strategie werkt, moet je een model trainen.

### Via CLI

```bash
python scripts/train_model.py \
  --symbol BTC/USD \
  --exchange coinbase \
  --timeframe 1h \
  --bars 2000 \
  --n-estimators 100
```

Het model wordt opgeslagen in `models/BTC_USD_model.joblib`.

### Via de API

```bash
curl -X POST "http://localhost:8000/api/v1/ml/train?symbol=BTC/USD&exchange=coinbase&timeframe=1h&bars=2000" \
  -H "X-API-Key: <jouw-api-key>"
```

### Features

Het model gebruikt 10 technische indicatoren als input:

1. RSI (14 perioden)
2. MACD lijn
3. MACD signaal
4. Bollinger Band boven
5. Bollinger Band onder
6. Bollinger Band midden
7. ATR (14 perioden)
8. Volume/SMA ratio
9. Close/SMA ratio
10. Return over 1 bar

### Na Training

Start een backtest of paper run met `model_strategy`:

```bash
curl -X POST "http://localhost:8000/api/v1/runs" \
  -H "X-API-Key: <jouw-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "strategyName": "model_strategy",
    "mode": "paper",
    "symbols": ["BTC/USD"],
    "timeframe": "1h",
    "initial_capital": "5000"
  }'
```

---

## 9. API Reference

Alle endpoints vereisen `X-API-Key` header (tenzij anders vermeld).

### Health & Metrics

| Method | Endpoint | Auth | Beschrijving |
|--------|----------|------|--------------|
| GET | `/health` | Nee | Server status |
| GET | `/api/v1/metrics` | Nee | Prometheus metrics |

### Runs

| Method | Endpoint | Beschrijving |
|--------|----------|--------------|
| POST | `/api/v1/runs` | Start een nieuwe run (backtest/paper/live) |
| GET | `/api/v1/runs` | Lijst alle runs (paginatie: `?offset=0&limit=25`) |
| GET | `/api/v1/runs?mode=paper` | Filter op mode |
| GET | `/api/v1/runs?status=running` | Filter op status |
| GET | `/api/v1/runs/{id}` | Detail van een run |
| DELETE | `/api/v1/runs/{id}` | Stop een run |

### Run Data

| Method | Endpoint | Beschrijving |
|--------|----------|--------------|
| GET | `/api/v1/runs/{id}/equity-curve` | Equity curve datapunten |
| GET | `/api/v1/runs/{id}/trades` | Gesloten trades |
| GET | `/api/v1/runs/{id}/orders` | Orders |
| GET | `/api/v1/runs/{id}/fills` | Fill executions |
| GET | `/api/v1/runs/{id}/positions` | Position snapshots |
| GET | `/api/v1/runs/{id}/portfolio` | Portfolio snapshot |

### Portfolio

| Method | Endpoint | Beschrijving |
|--------|----------|--------------|
| GET | `/api/v1/portfolio/summary` | Aggregate portfolio over alle runs |

### Strategies

| Method | Endpoint | Beschrijving |
|--------|----------|--------------|
| GET | `/api/v1/strategies` | Lijst alle strategieen met parameter schema's |
| GET | `/api/v1/strategies/{name}/schema` | Parameter schema voor een strategie |

### ML Training

| Method | Endpoint | Beschrijving |
|--------|----------|--------------|
| POST | `/api/v1/ml/train` | Train een ML model (query params: symbol, exchange, timeframe, bars) |

---

## 10. Environment Variabelen

### Verplicht

| Variabele | Voorbeeld | Beschrijving |
|-----------|-----------|--------------|
| `POSTGRES_PASSWORD` | `mijn-veilig-wachtwoord` | Database wachtwoord |
| `EXCHANGE_ID` | `coinbase` | Exchange identifier |

### Exchange (voor paper en live)

| Variabele | Beschrijving |
|-----------|--------------|
| `EXCHANGE_API_KEY` | Coinbase API Key Name (volledige `organizations/...` string) |
| `EXCHANGE_API_SECRET` | Private Key (PEM formaat, `\n` als newline separator) |
| `EXCHANGE_API_PASSPHRASE` | Alleen voor legacy Coinbase Pro keys (leeg laten voor CDP keys) |

### Live Trading Gates

| Variabele | Standaard | Beschrijving |
|-----------|-----------|--------------|
| `ENABLE_LIVE_TRADING` | `false` | Master switch voor live orders |
| `LIVE_TRADING_CONFIRM_TOKEN` | leeg | Token dat meegestuurd moet worden bij live run requests |

### API Beveiliging

| Variabele | Standaard | Beschrijving |
|-----------|-----------|--------------|
| `REQUIRE_API_AUTH` | `false` | API key authenticatie vereisen |
| `API_KEY_HASH` | leeg | SHA-256 hash van je API key |
| `RATE_LIMIT_ENABLED` | `true` | Rate limiting aan/uit |

### Risk Management

| Variabele | Standaard | Beschrijving |
|-----------|-----------|--------------|
| `DEFAULT_MAX_OPEN_POSITIONS` | `3` | Max gelijktijdige posities |
| `DEFAULT_PER_TRADE_RISK_PCT` | `0.01` | Max 1% risico per trade |
| `DEFAULT_MAX_DAILY_LOSS_PCT` | `0.05` | Circuit breaker bij 5% dagverlies |
| `DEFAULT_MAX_DRAWDOWN_PCT` | `0.15` | Circuit breaker bij 15% drawdown |

### Database

| Variabele | Standaard | Beschrijving |
|-----------|-----------|--------------|
| `POSTGRES_HOST` | `localhost` | Database host (Docker: `postgres`) |
| `POSTGRES_PORT` | `5432` | Database poort |
| `POSTGRES_DB` | `trading_bot` | Database naam |
| `POSTGRES_USER` | `trading` | Database gebruiker |

---

## 11. Troubleshooting

### "Internal server error" bij backtest

**Oorzaak:** Verkeerde exchange of onbereikbare marktdata.

**Oplossing:**
1. Controleer `EXCHANGE_ID=coinbase` in `.env`
2. Gebruik Coinbase paren: `BTC/USD`, `ETH/USD` (niet `BTC/USDT`)
3. Controleer of er geen dubbele variabelen in `.env` staan (laatste waarde wint)

### "ExchangeNotAvailable" fout

**Oorzaak:** DNS probleem met de async HTTP client op Windows.

**Oplossing:** De bot patcht dit automatisch (aiodns -> ThreadedResolver). Als het toch optreedt, herstart de API container:

```bash
cd infra && docker compose --env-file ../.env restart api
```

### Paper run verdwijnt na herstart

**Oorzaak:** Paper runs draaien als asyncio background tasks en overleven geen herstart.

**Oplossing:** Dit is by design. De equity curve en trades worden opgeslagen bij het stoppen van de run. Start een nieuwe paper run na herstart.

### "Authentication error" bij Coinbase

**Oorzaak:** API key formaat incorrect.

**Controleer:**
1. `EXCHANGE_API_KEY` begint met `organizations/`
2. `EXCHANGE_API_SECRET` bevat de PEM key met `\n` newlines
3. Geen aanhalingstekens rondom de waarden in `.env`
4. Geen dubbele `EXCHANGE_API_KEY` regels in `.env`

### Dashboard toont geen data

**Oorzaak:** Geen runs gestart, of API onbereikbaar.

**Oplossing:**
1. Check API health: `curl http://localhost:8000/health`
2. Start een backtest om data te genereren (zie sectie 5)
3. Check `NEXT_PUBLIC_API_URL=http://localhost:8000` in UI environment

### Circuit breaker geactiveerd

**Oorzaak:** Dagverlies of drawdown limiet overschreden.

**Oplossing:** De circuit breaker moet handmatig gereset worden. Stop de run en start een nieuwe. Overweeg om risk parameters aan te passen in `.env`.

### Tijdzones

Alle timestamps in de API zijn **UTC**. De dashboard toont ze in je lokale tijdzone.

---

## Snelle Referentie

```bash
# Status checken
curl http://localhost:8000/health

# Strategieen bekijken
curl -H "X-API-Key: $KEY" http://localhost:8000/api/v1/strategies

# Backtest starten
curl -X POST http://localhost:8000/api/v1/runs \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"strategyName":"breakout","mode":"backtest","symbols":["BTC/USD"],"timeframe":"1h","initial_capital":"5000","backtest_start":"2026-02-01T00:00:00Z","backtest_end":"2026-03-01T00:00:00Z"}'

# Paper run starten
curl -X POST http://localhost:8000/api/v1/runs \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"strategyName":"breakout","mode":"paper","symbols":["ETH/USD"],"timeframe":"1h","initial_capital":"5000"}'

# Runs bekijken
curl -H "X-API-Key: $KEY" "http://localhost:8000/api/v1/runs?limit=10"

# Run stoppen
curl -X DELETE -H "X-API-Key: $KEY" http://localhost:8000/api/v1/runs/{id}

# Portfolio overzicht
curl -H "X-API-Key: $KEY" http://localhost:8000/api/v1/portfolio/summary

# ML model trainen
python scripts/train_model.py --symbol BTC/USD --exchange coinbase --timeframe 1h --bars 2000

# Docker containers herstarten
cd infra && docker compose --env-file ../.env restart api ui
```
