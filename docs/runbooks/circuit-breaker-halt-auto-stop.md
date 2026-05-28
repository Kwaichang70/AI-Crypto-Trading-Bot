# Runbook — CircuitBreaker HALT Auto-Stop

**Audience:** Operator responding to a live or paper trading run that auto-stopped via the graduated CircuitBreaker HALT response.

**Triggered by:** Sprint 50 Cycle 4 — when `CircuitBreaker.check_graduated()` returns `HALT` during a live/paper run, the engine now sets `_stop_event` and the orchestrator writes an `event_type='circuit_breaker_halt_auto_stop'` row to `audit_events` before transitioning the run to `status='stopped'`.

**Related sprints:** Sprint 32 (graduated CB), Sprint 41 (SEC-002 audit), Sprint 50 cycle 3 (kill-switch + admin audit).

---

## 1. Detection

### 1a. Detect via dashboard
- Home page Active Runs card shows the run count drop.
- Open the run detail page — `status` will read `stopped`. Cycle 6 (run-detail logs sprint) will add a visual badge distinguishing auto-stop from operator-stop; until then, the audit_events query below is the authoritative source.

### 1b. Detect via audit query
```sql
SELECT
    timestamp,
    actor,
    resource_id AS run_id,
    payload->>'trigger' AS trigger
FROM audit_events
WHERE event_type = 'circuit_breaker_halt_auto_stop'
ORDER BY timestamp DESC
LIMIT 20;
```
`actor` will be `"system"` (engine-initiated, not operator).

### 1c. Detect via structured log
```bash
docker compose -f /opt/trading-bot/infra/docker-compose.yml logs api 2>&1 \
    | grep "circuit_breaker.halt_auto_stop_requested"
```
The log line carries `run_id`, `symbol`, `open_position_count`, and `reason="graduated_circuit_breaker_halt"`.

---

## 2. Immediate operator checklist

When you receive notice (dashboard, log, or — from cycle 8 — a Telegram alert):

1. **Identify the run.** Note `run_id`, `run_mode` (paper vs live), `strategy_name`, and the open position count from the log line.
2. **Live mode only — inspect exchange positions immediately.** The auto-stop disables Sprint 27 trailing-stops on all open positions. Positions remain on the exchange with **zero algorithmic protection**.
   - Coinbase: log into the exchange UI, check Active Positions for the symbols listed in the audit payload.
   - If the position is significantly underwater AND volatility is elevated: consider manual liquidation.
   - If the position is near break-even AND volatility is normal: consider holding and resetting the breaker (see §3).
3. **Paper mode only — no exchange action required.** Positions are simulated; the run is fully halted.
4. **Document the reason.** Write a one-line incident note covering: what triggered HALT (drawdown? consecutive losses? daily loss?), market conditions at the time, and the action taken on open positions.

---

## 3. Reset + restart procedure (TWO separate operations)

After deciding the situation is recoverable, the operator MUST perform these two operations **in order**:

### 3a. Reset the CircuitBreaker
The graduated CB retains its tripped state across run restarts. Resetting clears the `_tripped` flag so the next run does not immediately re-HALT.

```bash
curl -X POST https://trading-bot.tail51da62.ts.net/api/v1/runs/{RUN_ID}/circuit-breaker/reset \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json"
```

Expected response: `200 OK` with the reset confirmation. A `409` means the breaker is not tripped (no-op, safe to ignore).

### 3b. Restart the run (separate operation)
The auto-stopped run is in `status='stopped'`. It cannot be resumed in place — create a new run with the same config:

1. Open the run detail page → click "Duplicate Run".
2. Verify the strategy + symbols + sizing match the original.
3. Click "Start" to launch the new run.

The new run inherits a fresh CircuitBreaker instance via `Run.config.risk_parameters`. The previous breaker state is NOT carried over.

---

## 4. When NOT to restart

Do not immediately restart if any of the following holds:

- **Root cause unresolved.** If HALT was triggered by a flash crash and the market is still in distress, restarting will likely re-HALT within hours.
- **Strategy fundamentally broken.** If HALT was caused by a sequence of losing trades reflecting model drift (Sprint 23 RetrainingService) or regime change (Sprint 32 FGI), pause and consider:
  - Retraining the model (`POST /api/v1/ml/retrain`)
  - Switching to a different strategy temporarily
  - Reducing position sizing in `risk_parameters`
- **Open positions still unresolved.** Restarting before manually closing/holding the legacy positions creates a confused portfolio state — the new run instance will not see those positions.

---

## 5. Trailing-stop disablement — operational consequence

**Critical:** Sprint 27's trailing-stop manager runs *inside* the StrategyEngine bar loop. When HALT auto-stop terminates the engine, the trailing-stop manager stops with it. **Open positions on live exchanges are left without automatic stop-out.**

This is a deliberate design trade-off:
- **Pro:** Forced liquidation at HALT-trigger prices (typically post-flash-crash, post-drawdown) historically executes at worst-decile prices. The operator decides liquidation strategy.
- **Con:** A position could continue to bleed if the operator is asleep / unavailable. The audit row + log line ensure observability, and cycle 8 Telegram alerting closes the response-time gap.

Operators running live with substantial position sizes should consider:
- Setting exchange-side stop-loss orders as a fail-safe (independent of the bot)
- Configuring more conservative `max_drawdown_pct` thresholds to trigger HALT earlier (when positions are smaller)
- Cycle 8 Telegram alerting (planned) for sub-minute operator notification

---

## 6. Distinguishing auto-stop from operator-stop

| Source | `audit_events.event_type` | `actor` |
|---|---|---|
| Operator clicked Emergency Stop on a single run | `emergency_stop` | `api_key_<hex12>` |
| Operator clicked Emergency Stop All Runs (kill-switch) | `kill_switch` | `admin_key_<hex12>` |
| Engine auto-stopped via HALT | `circuit_breaker_halt_auto_stop` | `system` |
| Normal `DELETE /runs/{id}` (graceful stop) | (no audit row) | n/a |

Cycle 6 (run-detail logs sprint) will surface this distinction in the UI; until then, the table above is the operator's reference.

---

## 7. Related procedures

- **Operator-initiated single-run stop:** Sprint 45 SEC-006 — `POST /api/v1/runs/{id}/emergency-stop`
- **Operator-initiated global stop:** Sprint 50 cycle 3 — `POST /api/v1/emergency/kill-switch` (requires `X-Admin-Key`)
- **CircuitBreaker reset:** Sprint 41 — `POST /api/v1/runs/{id}/circuit-breaker/reset`
- **Live trading gate (initial activation):** Sprint 45 SEC-004 — `X-Live-Confirm-Token` header

---

## 8. Change log

- **2026-05-28** — Initial runbook drafted with Sprint 50 cycle 4 (CircuitBreaker HALT auto-stop wiring).
