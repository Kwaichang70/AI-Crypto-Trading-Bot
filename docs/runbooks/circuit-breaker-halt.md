# Runbook — CircuitBreaker HALT (entries blocked, auto-recovering)

**Audience:** Operator responding to a live or paper run whose graduated CircuitBreaker reached HALT, REDUCE or DAILY_LIMIT.

**Replaces:** `circuit-breaker-halt-auto-stop.md` (Sprint 50 Cycle 4). WP1.6 (decision D4, user decision U-1) removed the HALT auto-stop and the operator reset: **HALT blocks new entries only, the run keeps running, and HALT clears by itself.**

**Related:** `docs/risk-model.md` (Circuit Breaker), WP1.5 (drawdown / daily-loss gates), WP1.7 (kill switch, flatten), Sprint 32 (graduated CB), Sprint 50 cycle 3 (kill switch).

---

## 1. What the breaker does (WP1.6)

The breaker is evaluated **once per processed bar** from the portfolio's risk snapshot (mark-to-market NAV, peak, UTC-day start). It uses the run's own limits. **Every effect is BUY-only: strategy SELLs, bracket stop-loss / take-profit, trailing stops, resting orders and flatten are never blocked or scaled.** The breaker never stops the run and nothing is persisted: all levels are re-derived from data after a restart.

| Level | Enters (immediately) | Leaves | Effect |
|---|---|---|---|
| REDUCE | drawdown >= 0.5 x limit | drawdown < 0.4 x limit for 3 consecutive bars | BUY quantity x 0.5 (the final order quantity, after the risk ceiling) |
| DAILY_LIMIT | the WP1.5 daily-loss predicate (loss <= -8% of the day-start basis, or the day state is unknown) | at the first bar of the next UTC day (no debounce; same release as the gate) | BUYs dropped |
| HALT | drawdown >= limit (live 20%, paper 30%) | drawdown < 0.9 x limit for 3 consecutive bars, then it **steps down to REDUCE** (never straight to OK) | BUYs dropped; the run is NOT stopped |

| Run | HALT entry / exit | REDUCE entry / exit | Daily loss |
|---|---|---|---|
| LIVE | 20% / 18% | 10% / 8% | 8% |
| PAPER | 30% / 27% | 15% / 12% | 8% |

- Escalation takes one bar; de-escalation needs 3 consecutive bars (an in-band reading resets the count). In the HALT exit band (0.9 x limit up to the limit) the breaker still blocks although the WP1.5 gate would allow: it is deliberately stricter.
- The WP1.5 risk gate (`pre_trade_check`) stays the per-order enforcement of record. A BUY the breaker lets through is still checked by it.
- Backtests run **without** the breaker unless `BacktestRunner(enable_circuit_breaker=True)` (WP2.1 turns it on by default).
- **Restart watch.** A breaker is built fresh on every start and resume (nothing is persisted), so it cannot know the level it had before. For its first 3 evaluations it enters HALT already at 0.9 x limit (live 18%), and for its first 6 evaluations it enters REDUCE already at 0.4 x limit (live 8%). Recovery counters start at 0; a level entered during the watch is then held by the normal 3-bar hysteresis. The API field `restart_watch` shows `halt`, `reduce` or `none`. Effects after a restart, all expected:
  - Never laxer than the WP1.5 gate.
  - From the 6th evaluation on, never laxer than a breaker that was never restarted.
  - Before that, at most 5 bars laxer, only while the drawdown is below 0.9 x limit (full size only below 0.4 x limit): the recovery bars the old breaker was still counting. Usually one level (HALT becomes half size, or half size becomes full size); in the first 2 bars and only below 0.4 x limit it can be two levels (a breaker that was in HALT is at full size at once).
  - To keep entries blocked across a restart during a drawdown, use the kill switch (section 3) and clear it only after `restart_watch` shows `none`. Paper runs restart at API boot and keep an active kill switch, so engage it before the restart. A live run takes entries only after a normal resume, and a normal resume is refused with 409 `kill_switch_active` while the kill switch is active (see `docs/runbooks/deploy-fase1-minimumset.md`), so engage it right after the normal resume, once the run shows `running` (a press while the run is still `resuming` sends it back to `orphaned`). A protective resume blocks all entries of that run by itself; any later normal resume builds a new breaker with a new watch.
  - Stricter, possibly for a long time: after a restart at 18-20% drawdown entries stay blocked until the drawdown has been below 18% for 3 bars, and after a restart at 8-10% BUYs stay at half size until it has been below 8% for 3 bars, even if the breaker allowed more before the restart.
  - The watch also runs at the start of a new run; there the drawdown starts at 0, so it normally has no effect.

---

## 2. Detection

### 2a. Telegram (if configured)
A `CIRCUIT_BREAKER` message is sent when HALT is entered (`CRITICAL`: "entries blocked, exits active, run NOT stopped", the drawdown, the limit, N) and when HALT clears (`WARNING`). A HALT entered by the restart watch (section 1) reads "Drawdown X% is at or above 18.00% during the restart watch (limit 20.00%)" instead of "reached the limit". It carries percentages only (no amounts). REDUCE and DAILY_LIMIT do not send Telegram (logs and gauge only). If no notifier is configured there is **no push alert**; use the log/Grafana signals below. Never paste the bot token into tickets or logs.

### 2b. Grafana
Rule `trading-cb-halt` fires on `max(trading_circuit_breaker_state) >= 3` (OK=0, REDUCE=1, DAILY_LIMIT=2, HALT=3). The gauge is zeroed when a run stops.

### 2c. Structured log
```bash
docker compose -f /opt/trading-bot/infra/docker-compose.yml logs api 2>&1 \
    | grep -E "circuit_breaker\.(halt_entered|halt_cleared|transition|daily_limit_entered|reduce_entered)|engine\.circuit_breaker_(entries_blocked|eval_failed)|circuit_breaker\.listener_failed"
```
- `circuit_breaker.halt_entered` (critical, `alert="ENTRIES_BLOCKED"`) once per HALT entry; `circuit_breaker.halt_cleared` once per exit.
- `engine.circuit_breaker_entries_blocked` once per bar that dropped at least one BUY (`skip_reason`).
- `engine.circuit_breaker_eval_failed` (error): the breaker itself failed; that bar's BUYs were dropped (fail-closed for entries only). A loop of these means entries are blocked until fixed: open an incident.

### 2d. API
`GET /api/v1/runs/{run_id}/circuit-breaker` returns the typed state: `level`, `active_conditions`, `entries_blocked`, `halt_exit_progress` / `reduce_exit_progress` (k of 3), `restart_watch` (`halt` / `reduce` / `none`), `thresholds`, and the last evaluation's `inputs` (NAV, peak, drawdown, daily threshold, day-start source). 404 means the run is not in this process or has no breaker.

---

## 3. Operator checklist when HALT is entered

1. **The run is still running.** Exits (strategy SELL, bracket SL/TP, trailing stop) keep firing; new BUYs are dropped with `skip_reason=circuit_breaker_halt`.
2. Read the state: `GET .../circuit-breaker` (drawdown vs the limit, how many bars below the exit band).
3. **Decide** (the breaker does not decide for you):
   - Do nothing: HALT clears by itself after 3 bars below 90% of the limit, then entries resume at **half size** (REDUCE) until the drawdown is below 40% of the limit for 3 bars.
   - Block entries manually, independent of the breaker: the **kill switch** (`POST /api/v1/emergency/kill-switch`, admin).
   - Exit: **flatten** (`DELETE /api/v1/runs/{id}?flatten=true`, or the stop dialog in the UI), see `docs/runbooks/smoke-roundtrip-mechanics-test.md` section 6.
4. Document the reason: what triggered HALT, market conditions, the action taken.

### A flat run at the limit does not recover
Recovery needs the NAV to move. A run that is **flat** (stop-loss closed everything) while the drawdown is at or above the limit has a constant NAV, so neither the breaker nor the WP1.5 gate ever clears (WP1.5 "never auto-reset" property). Start a new run (WP1.9 will allow adjusting the parameters). This is not a defect.

---

## 4. The reset endpoint is retired

`POST /api/v1/runs/{id}/circuit-breaker/reset` now returns **410 Gone** (`circuit_breaker_reset_retired: ...`), changes nothing and writes no audit row. Update any script that called it. There is nothing to reset: HALT, REDUCE and DAILY_LIMIT are data-derived. (Route removal: CF-29.)

---

## 5. Historical audit rows

Runs stopped before WP1.6 may carry `audit_events.event_type = 'circuit_breaker_halt_auto_stop'` (actor `system`) and `circuit_breaker_reset`. Both event types stay valid for those historical rows; WP1.6 writes no new ones.

```sql
SELECT timestamp, actor, resource_id AS run_id, payload->>'trigger' AS trigger
FROM audit_events
WHERE event_type IN ('circuit_breaker_halt_auto_stop', 'circuit_breaker_reset')
ORDER BY timestamp DESC LIMIT 20;
```

---

## 6. Distinguishing the stop sources

| Source | `audit_events.event_type` | `actor` |
|---|---|---|
| Operator Emergency Stop on a single run | `emergency_stop` | `api_key_<hex12>` |
| Operator kill switch (all runs) | `kill_switch` | `admin_key_<hex12>` |
| Circuit-breaker HALT | none: HALT does not stop the run (WP1.6) | n/a |
| Historical HALT auto-stop (before WP1.6) | `circuit_breaker_halt_auto_stop` | `system` |

---

## 7. Change log

- **2026-05-28** — Initial runbook (Sprint 50 cycle 4, HALT auto-stop).
- **WP1.6** — Rewritten: HALT blocks entries only, auto-recovers (U-1), reset endpoint retired (410), Telegram alert on HALT, gauge zeroed on stop. Renamed from `circuit-breaker-halt-auto-stop.md`.
