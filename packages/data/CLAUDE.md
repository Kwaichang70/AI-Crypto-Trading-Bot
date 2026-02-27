IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Data Package
Data acquisition, caching, and feature engineering.

### Components
- **MarketDataService** — OHLCV candle fetching and caching
  - Timeframes: 1m, 5m, 1h (configurable per strategy, default 5m)
  - Timestamp/timezone normalization
  - Rate limiting + retries with exponential backoff
  - Local caching with PostgreSQL cache index
- **Feature Pipeline** — Technical indicator computation
  - RSI, MACD, ATR, returns, volatility
  - Extensible for ML feature engineering
- **ModelStrategy Placeholder** — ML-ready inference interface
  - `predict(features) -> signal/confidence`
  - Training offline (separate from trading runtime)
  - Model versioning (file + metadata in DB)

### Data Sources
- Exchange APIs via CCXT abstraction
- Store raw + derived data in PostgreSQL
