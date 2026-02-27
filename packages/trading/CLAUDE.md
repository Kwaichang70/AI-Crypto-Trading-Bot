IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Trading Package
Core trading engine containing the heart of the system.

### Components
- **StrategyEngine** — Pluggable strategy interface: `on_bar(data) -> signals`
  - Signals: BUY/SELL/HOLD, target position, confidence (float)
  - Parameter schema + validation per strategy
- **ExecutionEngine** — Order placement and fill simulation
  - Paper mode: simulated fills, partial fills, latency, slippage, fees
  - Live mode: real order placement via CCXT (limit/market), idempotency keys
  - Order state machine: NEW -> PARTIAL -> FILLED/CANCELED/REJECTED
- **RiskManager** — Pre-trade checks and position management
  - Exposure limits, drawdown checks, daily loss limits, max position size
  - Position sizing (fixed fractional of equity)
  - Stop-loss / take-profit / trailing stop (configurable)
  - Kill-switch capability
- **Baseline Strategies** — MA Crossover, RSI Mean Reversion, Breakout (Donchian/ATR)

### Design Principles
- Spot-only for MVP (max leverage = 1)
- Deterministic backtests via seed control
- No silent failures — all errors must be logged and handled
- Fee/slippage model must be configurable (taker %, maker %, slippage bps)
