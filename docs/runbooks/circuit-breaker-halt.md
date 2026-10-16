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
- Backtests run **with** the breaker by default since WP2.1c (`BacktestRunner(enable_circuit_breaker=True)`); `enable_circuit_breaker=False` (scripts: `--no-circuit-breaker`) switches it off. Runs created by 2.1a / 2.1b ran without it (`engine_semantics.backtest_circuit_breaker=false`).
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

## 7. Intra-bar exit check (WP1.3b-b)

**What it is.** A LIVE run with a stop-loss and/or a trailing stop checks the **bid** between bar closes
(default every 300 s) instead of waiting for the next close. It is a sibling task of the bar loop: a
bar pass is never delayed by it. Take-profit is **bar-close only**; the check never opens a position.

**Settings.** `LIVE_EXIT_CHECK_INTERVAL_S` (default `300`; `0` = off; otherwise an integer 60 to 3600).
It is effective only for a LIVE run whose timeframe is longer than the interval and that has a
bracket stop-loss or a trailing stop. It is always off for paper, backtest and `smoke_roundtrip`
runs. The run's start logs `runs.intrabar_exit_checks interval_s=300` or `interval_s=null reason=...`
(`disabled`, `interval_ge_timeframe`, `smoke_run`, `no_protective_exit`, `not_live`).

**Semantics.**
- Tick: read the bid of every held symbol that has a protective level. A symbol at or below its level
  gets a second bid reading about 10 s later; only if the higher of the two readings is still at or
  below the level is the exit sent (a normal market SELL through `process_signal`: the WP1.11a cap and
  the WP1.4b idempotent submit apply). `last` is never used.
- It never fires on a missing, empty, non-positive, non-finite or crossed-book reading, nor on one
  reading alone. Nothing is retried inside a reading; the confirming reading is the retry.
- Breach to SELL is at most: one interval + 10 s + fetch time + order latency. Gaps are gaps.
- The check is read-only on the portfolio (no marks, peak, day-start or breaker are touched) and never
  evaluates the circuit breaker. The kill switch, protective mode, HALT, DAILY_LIMIT and
  `reconcile_required` do **not** stop an intra-bar exit (they block entries only).
- Live stops are therefore touch-based (bid, confirmed), while the validated backtest is close-based.

**Logs and metrics.**

| Signal | Meaning | Action |
|---|---|---|
| `engine.intrabar_started` | the task runs (interval in the log) | none |
| `engine.intrabar_exit_signal` (warning) | an intra-bar exit was sent (`trigger`, both bids, both asks, `quote_age_s1/2` = age of the exchange's last-trade timestamp, logged only) | check the SELL fill; a wide spread or an old trade age is a thin-book or stale-ticker hint (L-14) |
| `engine.intrabar_breach_unconfirmed` (info) | the second reading was back above the level | none |
| `engine.intrabar_quote_failed` (warning) | one failed reading (`reason` = error type) | watch for `degraded` |
| `exit_check.degraded` (error) | 3 consecutive ticks without a usable bid for a held symbol; repeated every 15 min | **incident**: the bar close still protects; check the exchange / ticker; see G-LIVE L-14 |
| `exit_check.recovered` (info) | a valid reading again | none |
| `engine.intrabar_rate_limited` (warning) | 429: the interval doubled (cap 900 s) and is restored after one clean tick | none unless it persists |
| `engine.intrabar_lock_busy` (warning) | a bar pass / flatten held the lock for 5 s; one retry follows in 30 s | more than 6 per hour: incident |
| `engine.intrabar_tick_error` / `engine.intrabar_loop_died` (error) | a tick failed / the task ended; the bar loop is unaffected | `loop_died` (also `intrabar_loop_died_total`): plan a restart (bar close protects) |
| `engine.exit_not_submitted` | an exit reached no accepted order (`reason`, `streak`); `inflight_exit_pending` (info) is a SELL already in flight and not a failure | error level (streak 3, then every 10th) is an incident |

Counters: `intrabar_exit_checks_total`, `intrabar_exit_signals_total{trigger}`,
`intrabar_ticker_errors_total{reason}`, `intrabar_confirm_failed_total{reason}`,
`intrabar_breach_unconfirmed_total`, `intrabar_lock_busy_total`, `intrabar_tick_errors_total`,
`intrabar_loop_died_total`,
`exit_not_submitted_total{trigger,reason}`, `bracket_anchor_source_total{source,final}`.
No log line of this feature carries exception text, a URL or a raw ticker.

**G-LIVE checks (WP1.3b).**
- **L-14 (before the restart).** Read-only ticker sanity on the deploy host for every run symbol:
  `bid` finite and positive, `ask >= bid`, latency under 2 s, three calls 10 s apart. Also record
  whether `fetch_ohlcv(limit=N)` returns the still-forming candle on the real exchange (if it does,
  open the CR-09 fix). If the bid is not usable the check would never fire: fix it, or set
  `LIVE_EXIT_CHECK_INTERVAL_S=0` and record the operator's acceptance of bar-close-only exits.
- **L-15 (after the restart).** `runs.intrabar_exit_checks interval_s=300` for the live run; while it
  holds a position `intrabar_exit_checks_total` increases within 2 x the interval and
  `intrabar_ticker_errors_total` and `exit_check.degraded` stay at 0.
- **L-16 (after a resume).** For every held symbol `bracket_exit.anchored` with
  `source=resume_recomputed` (ATR) or exact (fixed), and `trailing_stop.peak_seeded`. Judge sources
  after 3 bar passes. `resume_fallback_current_atr` (final), `entry_unverified`, `entry_deferred` or
  `trailing_stop.seed_partial` are incidents: the position is protected, record the level difference.

**Limits.** No backtest or optimizer on the live API process while live holds positions (an
event-loop stall stops every exit). A dust position below the exchange minimum cannot be sold: the
exit retries every tick and logs `engine.exit_not_submitted` (error from the 3rd attempt).

**Shutdown and flatten (WP13BB-SEC-05, SEC-07).** On a stop or a graceful shutdown the run waits up
to 15 s for the check: at once when it is idle or in the confirm wait, at most one ticker reading
(10 s) when it is reading, and one `process_signal` per confirmed symbol when it is sending exits
(an exit in flight is never cancelled on purpose). A `flatten()` that starts while an intra-bar exit
holds the lock waits for it inside its own 30 s budget. With several confirmed symbols that wait can
use up the budget: the result is then `in_flight` with cause `lock_timeout` and the run latches
`flatten_incomplete` (entries blocked), while the check is selling those same positions. Re-read the
positions and retry the flatten; do not sell by hand while a bot SELL is still open.

---

## 8. Change log

- **2026-05-28** — Initial runbook (Sprint 50 cycle 4, HALT auto-stop).
- **WP1.6** — Rewritten: HALT blocks entries only, auto-recovers (U-1), reset endpoint retired (410), Telegram alert on HALT, gauge zeroed on stop. Renamed from `circuit-breaker-halt-auto-stop.md`.
- **WP1.3b-b** — Added section 7: the live intra-bar exit check (settings, semantics, logs, G-LIVE L-14..L-16).
- **WP1.3b-b fix** — Section 7: asks and trade age on `engine.intrabar_exit_signal`, `intrabar_loop_died_total`, shutdown and flatten notes.
