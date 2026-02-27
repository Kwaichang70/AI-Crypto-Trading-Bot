IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## FastAPI Backend
Python 3.11+ REST API server using FastAPI framework.

### Key Endpoints (from KickOff.md)
- `POST /runs/start` — Start backtest/paper/live run
- `POST /runs/{run_id}/stop` — Stop a run
- `GET /runs/{run_id}/status` — Get run status
- `GET /runs/{run_id}/trades` — Get trades for a run
- `GET /runs/{run_id}/orders` — Get orders for a run
- `GET /runs/{run_id}/metrics` — Get PnL, drawdown, exposure metrics
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics (optional)

### Technical Requirements
- Pydantic models with strict validation
- SQLAlchemy for database access
- Structured JSON logging
- Async where beneficial
- Type hints throughout
