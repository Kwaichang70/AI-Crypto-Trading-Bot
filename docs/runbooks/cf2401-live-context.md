# Runbook — CF-24-01 live context (poller, book, recorder, shadow stage)

**Audience:** operator of the production stack (`/opt/trading-bot`) and the reviewers of release R1 (tag `cf2401-r1`).

**Scope:** what the live-context machinery of CF-24-01 does after R1, how to read it, what it deliberately does not do, and the checks around the R1 deploy. R1 starts **Stage 1** (`LIVE_CONTEXT_STAGE = "neutral_shadow"` in `packages/trading/mtf_signals.py`): the poller, the book and the recorder run, but consumer strategies (`rsi_mean_reversion`, `dca_rsi_hybrid`, `model_strategy`, ensembles that contain one) still receive an empty context. Stage 2 is a separate release (P9). The register with rulings, ledger and gate lists is `Documentation/CF-24-01-register.md`.

> [!CAUTION]
> Never paste secret values into a terminal, ticket or chat. This runbook names variables only. Provider keys travel in the query string of FRED and Whale Alert requests (CF-2401-16): never print a full URL, never run a provider command with `-v`, and paste only the header lines asked for. Conventions (`$DC`, `hdrs`, `env_set`) are those of `docs/runbooks/deploy-fase1-minimumset.md` section 0.

**Metric names.** The code and the ledger quotes use internal names (`mtf_context_fresh`). The exporter adds the prefix `trading_` (`apps/api/prometheus.py`), and Grafana and every query below use the exported form (`trading_mtf_context_fresh`). Counters keep their `_total` suffix.

---

## 1. Reading the state

`GET /api/v1/health/background`, same call pattern as deploy section 5: `curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/health/background`. Block `context_poller`:
- `configured`, `running`, `alive` (running and a supervisor heartbeat within 3 ticks = 180 s, monotonic clock), `heartbeat_age_seconds`, `regressed` (wall clock more than 5 min behind the book's high-water mark: no series is `fresh`, engines read an empty book).
- `series[]`: `series`, `status`, `last_outcome`, `last_reason`, `rows`, `age_seconds`, `failed_cycles`, `task_alive`.
- `status` is one of `fresh`, `stale`, `invalid`, `unavailable`, `not_configured`, `egress_blocked`, `auth_failed`. `last_outcome` is one of `ok`, `unchanged`, `http_3xx`, `http_4xx`, `http_429`, `http_5xx`, `timeout`, `connect_error`, `egress_blocked`, `invalid_payload`, `oversize`, `auth_failed`, `budget`. `last_reason` is a closed rejection code (`ErrReason`) or `unknown`.
- The key `history_cache_warmer` stays in the response as a deprecated compat key. It is always not configured; read `context_poller` instead.

Metrics (exported names, label `series` = `alternative_me:fgi`, `coingecko:btc_dominance`, `fred:T10Y2Y`, `whale:net_flow_1h`): `trading_mtf_context_fresh`, `trading_mtf_context_age_seconds`, `trading_mtf_context_series_configured`, `trading_mtf_context_poll_total{series,outcome}`, `trading_mtf_context_value_rejected_total{series,reason}`, `trading_mtf_context_poller_alive`, `trading_mtf_context_poller_heartbeat_timestamp_seconds`, `trading_mtf_context_consumer_runs_active{mode}`, `trading_mtf_context_clock_regressed_total`, `trading_mtf_context_regressed_reads`, `trading_mtf_context_task_restarts_total`, `trading_mtf_context_record_dropped_total{series}`, `trading_mtf_context_recorder_write_failures_total{series}`, `trading_mtf_context_record_flush_timeout_total`. `trading_mtf_context_regressed_reads` counts **engine reads** (consumers and non-consumers: momentum reads the book every bar), not consumer reads (CF-2401-69).

Log events (structured JSON, key `event`): `mtf_context.status_changed` (one WARNING per change, fields `series`, `old`, `new`), `mtf_context.status_reminder` (every 6 h while a series stays bad), `mtf_context.clock_regressed` / `mtf_context.clock_recovered`, `mtf_context.fgi_first_seen`, `context_poller.started`, `context_poller.dead`, `context_poller.task_restarted`, `context_poller.fetch_failed`, `context_poller.apply_failed`, `context_poller.startup_error`, `engine.mtf_context_shadow`.

## 2. What the stage does (T-39 points 1 to 3)

**(1) Display versus engines.** `/api/v1/signals/current` shows the latest validated provider value from the in-process cache, with **no age** (FGI = day D). Engines read the lagged book: during day D the FGI they see is the value of day D-1 (`estimated_lag`, observed 00:00 + 24 h), the same rule as backtests. The two differ by design until CF-2401-08 adds the book age and state to the display. Do not treat the difference as a fault.

**(2) Context-trained paper models never BUY in Stage 1.** `ModelStrategy.context_trained()` is true when the loaded artefact's importances over the context columns sum above 0, and also when the importances are missing or cannot be read. In Stage 1 every consumed field is `None`, so every BUY of such a paper model is dropped with skip reason `mtf_context_model_input_missing` (log event `engine.context_model_entries_blocked`); in LIVE it is `mtf_context_model_unverified`. Exits are never blocked. An inert model (importance 0) is never blocked. This is intended (SY-01-16); the block exists from P4b, not P4a. The per-run count of dropped BUYs is a preview of the Stage 2 change on neutral inputs only under UQ-1 (a) (register section 10; under (b) PAPER stays blocked); it is not model-skill evidence (CF-2401-83).

**(3) Snapshot-series stall (two branches, decided by the CF-2401-39 result in section 11).**
- *If CoinGecko `data.updated_at` is present as a number or a digit string:* a provider value older than 5 400 000 ms at fetch time is rejected `provider_stale` (`COINGECKO_PROVIDER_STALE_MS`, `packages/data/provider_bands.py`); watch `trading_mtf_context_value_rejected_total{series="coingecko:btc_dominance",reason="provider_stale"}`.
- *If it is absent:* a provider that serves a frozen dominance value is **not detectable**. A frozen value fetched in a new hour bucket refreshes the age by design (CF-2401-21). The only control is the daily check that `btcDominance` in `/api/v1/signals/current` moves (section 12, daily list).
- Whale is a snapshot series under both branches: a provider that keeps returning the same or an empty transaction list reads as repeated or zero flow and is not detectable either (a long run of identical `whale:net_flow_1h` shadow values is the only sign). FGI and T10Y2Y are provider-dated, so a stall does not refresh their age.

## 3. The clock (T-39 point 4)

The book has one frame: the wall clock. A guard compares every read with a high-water mark (`CLOCK_REGRESSION_TOLERANCE_MS` = 300 000 in `packages/data/live_context.py`).
- **Backward step larger than 5 min:** every engine read returns an empty context and `trading_mtf_context_clock_regressed_total` rises; `/health/background` shows `regressed: true`; one WARNING `mtf_context.clock_regressed`, later one INFO `mtf_context.clock_recovered`. Context stays neutral until the clock passes the mark.
- **Forward step of size S, later corrected by NTP:** the mark stays up to S in the future, so context is neutral until `now + 5 min` reaches it. At the correction, `recorded` rows (dominance, whale) stamped more than 5 min beyond the corrected clock are dropped (ER3-01). If the offset lasted longer than S minus 5 min (for example a timezone or RTC offset held for days), the earlier step rows are kept and read up to S too young (R3-RES-2, accepted residual, Low). A forward step can also reveal held FGI/T10Y2Y rows up to S early (ER3-04; the value was really fetched). A forward step longer than 7 d + the cap trims all real history: the 7d fields are `None` until refilled or reseeded. A far-future mark keeps context neutral until the clock catches up.
- **R3-RES-1 is closed** by the poller calling `heartbeat` immediately before each `publish` with the same cycle timestamp (CF-2401-59, P5).
- **Residual (RK-P3-R2-05, QR-P3-R2-06):** a correction that no guard call observes cannot be detected (only a wall-versus-monotonic jump detector could). After a backward step larger than 5 min inside one bucket, the book and the store can serve different values for that bucket (look-behind only, never look-ahead).
- Check time sync before every restart and during the soak (section 12, daily list). The `alive` flag is monotonic and survives a forward wall step; the heartbeat timestamp does not (97 f).

## 4. Poller residuals (CF-2401-97, recorded verbatim)

- **(a)** "One bad T10Y2Y row pauses FRED delivery until it leaves the 5-row page (≤ 4 business days); the served value lags but stays capped; see `mtf_context_value_rejected_total{series="fred:T10Y2Y"}`" (QR-P5-05). Operator line: `increase(trading_mtf_context_value_rejected_total{series="fred:T10Y2Y"}[1d])`.
- **(b)** "Recorded rows are stamped at cycle start, up to 30 s before receipt" (QR-P5-07).
- **(c)** The FRED outer bound is 2 × 10 s; a 9.9 s + 10 s pair can lose one cycle, with no PIT effect (P5-PR-07, QR-P5-06).
- **(d)** A future-dated FRED row shows as `reason=internal_error`; tell it from a code fault by the `apply_failed` `error_type` (N-3).
- **(e)** Dominance 7d deltas are `None` for 7 d after a boot until the P7 seed (N-2). **Since P7:** the boot seed restores it from the store (`context_recorder.seeded`). The seed does not restore it when: the table is missing or the load fails (`context_store_load_failed`, then `reason` is `store_unavailable`); the 3 s seed bound expires (`context_recorder.seed_timeout`, `SEED_TIMEOUT_S`); the recorder fails to build (`context_recorder.startup_error`); a series publish fails (`context_recorder.seed_failed`); or no recorded dominance row was available in the 6 h before `now − 7 d` (a store with less than about 7 d of history, or a gap of 6 h or more at that point; SY-01-17). Only rows with basis `recorded`, `available_at <= now` and inside the provider band are seeded.
- **(f)** A host forward wall step makes the heartbeat timestamp look recent by the step size; only `/health` `alive` (monotonic) catches it (RK-P5-04). Operator line: check `alive` in section 1, not the timestamp alone.
- **(g)** After a supervisor death `mtf_context_fresh` is zeroed (E02), but `mtf_context_age_seconds` freezes, so read the stale rule together with the heartbeat rule (RK-P5-01).
- **(h)** A hung event loop or stuck supervisor sends no Telegram message. It shows as `/health` `alive: false` and, after about 50 min, in the Grafana alert list, which notifies no one under UD-4 (a).
- **(i)** After a restart, or when a run starts while a series is already stale, Telegram may stay silent for up to 6 h. A series that never becomes fresh after boot is announced only if its provider fails with a classified status (auth, egress, invalid payload) or delivers an out-of-date value; one that only times out, is rate-limited or returns server errors stays silent. A poller that fails to start is reported at the first 6 h reminder. Check `/health/background` `context_poller` after every restart and before any gate lift. After a restart a series that **stays non-fresh from boot** (`unavailable`, or `stale` from a seed older than its cap) without a status change sends nothing; a fresh-seeded series whose provider never succeeds now fires `fresh -> stale` (CF-2401-111, P7).
- **(j)** T10Y2Y parity: live serves first-release values. A row is fixed once it becomes visible (`obs + 2 business days + 1 d`, ER3-02 rule 5). Backtests read `context_observations`. Backfill T10Y2Y only with basis `vintage` (ALFRED `realtime_start`), or with `estimated_lag` built from first-release values. Never backfill `estimated_lag` from a current FRED download: that stamps revised values at first-release time, which is look-ahead, and the append-only writer then keeps them. Remaining gap: live may miss a revision for as long as that row is the newest (usually 1-3 d, at most 7 d), and a vintage backtest sees each value about 1 business day earlier (QR-P3-04). Measure the revision sizes from ALFRED when the backfill is built. 'Fresh again' means a value is visible within its cap, possibly a frozen first print.

## 5. Alerts and thresholds

- **Telegram** (`apps/api/services/context_alerts.py`; needs `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, set by the user, UQ-2 in the register; without them nothing is sent):
  - Stale: one message when a series moves into a non-fresh status, and one "fresh again" when it recovers if the stale message was attempted. Only while a consumer run is active (`LIVE_CONSUMERS`). A move to `fresh` with no open condition (the boot case) sends nothing. A re-notify every 6 h (`STATUS_REMINDER_S`) while the condition holds. Level CRITICAL while a live consumer run is active (`trading_mtf_context_consumer_runs_active{mode="live"}` above 0), else WARNING (CF-2401-108).
  - Poller dead (CRITICAL, 6 h reminders) and recorder failing (WARNING, at most once per 6 h): sent whether or not a consumer run is active. All texts are fixed strings.
- **Grafana** (`infra/grafana/provisioning/alerting/trading-alerts.yml`, group `trading-safety`, 7 rules; **Grafana notifies no one under UD-4 (a)**, read the alert list):
  - `mtf-context-poller-dead`, critical, `time() - max(trading_mtf_context_poller_heartbeat_timestamp_seconds) > 2700` for 5 m, `noDataState: Alerting`. It is a backstop: a supervisor death is sent to Telegram at once (once `TELEGRAM_*` is set, UQ-2), and `/health/background` `alive` turns false after 180 s. The 45 min versus 180 s difference is deliberate (CF-2401-96).
  - `mtf-context-stale`, severity `warning` (static), fires on a configured series with `trading_mtf_context_fresh == 0` while `sum(trading_mtf_context_consumer_runs_active) > 0`, for 15 m.
  - `mtf-context-recorder-failing`, warning, `max(increase(trading_mtf_context_recorder_write_failures_total[15m])) > 0`, for 1 m.
- **Silence you must know:** Telegram may stay silent after a restart (97 i); a hung event loop sends nothing (97 h). Check section 1 after every restart.
- **Flap count (CF-2401-112):** per series per UTC day, count `mtf_context.status_changed` log lines with `new` not `fresh` (any `old`). More than 4 on any day blocks Stage 2 until the recovery hold-down lands.

## 6. Shadow evidence and log level (Stage 1)

- Consumer runs also log `engine.live_context` (fields `version`, `stage`) once at start. Keep `LOG_LEVEL` at INFO for all of Stage 1 (`$DC exec -T api printenv LOG_LEVEL` prints `info` or `INFO`). Above INFO `engine.mtf_context_shadow` is not written, and the G-E evidence is lost silently (CF-2401-70 a). Never set DEBUG on the server: the `.trace` events then carry tracebacks and the urllib3/httpcore loggers are no longer held at WARNING (operator rule L-19 in `packages/common/CLAUDE.md`, CF-2401-25). Verify the effective level on the server before the first start.
- **Keep the G-E evidence outside the container.** `engine.mtf_context_shadow`, `engine.live_context`, `mtf_context.status_changed` and `mtf_context.status_reminder` exist only in the api container log, and a recreate (`$DC up -d` with a new image, `--force-recreate`, a rollback) discards it. Export once a day and before every recreate: `$DC logs --no-log-prefix --since 25h api 2>&1 | grep -E 'engine\.mtf_context_shadow|engine\.live_context|mtf_context\.status_(changed|reminder)' > <directory outside the container>/cf2401-$(date -u +%F).jsonl`; de-duplicate overlapping exports on the whole line. A gap in the export is a gap in the G-E window for every run and series it covers (CF-2401-71, -112).
- `engine.mtf_context_shadow` (one per consumer bar; non-consumers emit none) carries `bar_ts_ms` (epoch ms of the bar), `values` (the fresh restricted values of the consumed fields) and `states` (`served` or `none` per field). It carries no ages; ages come from `trading_mtf_context_age_seconds`. The consumer's decisions use the empty context.
- A persistently raising book is silent by design: the engine reader returns all `None` without a log line. The signal is shadow states `none` while `/health/background` and the gauges say `fresh` (CF-2401-70 b). Treat it as a possible read fault and report it.
- Multi-symbol runs: join shadow events on `(run_id, bar_ts_ms, occurrence in log order)`; evidence runs are single-symbol (CF-2401-82).

## 7. Context-trained models, operator notes (CF-2401-88)

- (a) A half-finished hot-swap stays fail-closed (`context_trained()` true) until the next sidecar `version_id`; there is no retry for the same id. Remedy: re-publish the sidecar with a new `version_id`. Look for `model_strategy.hot_swap` and `model_strategy.hot_swap_failed`.
- (b) An artefact without `feature_importances_` (a calibrated or pipeline wrapper, even v1) classifies as true and is blocked in PAPER, with one WARNING `model_strategy.importances_failed`.

## 8. The recorder

Only `coingecko:btc_dominance` and `whale:net_flow_1h` are recorded (one row per hour bucket, basis `recorded`; FGI and T10Y2Y are reproducible elsewhere).
- Boot: `context_recorder.seeded` (fields `rows` = loaded count, `reason`: null after a successful load, `store_unavailable` after a failed one; it does not say how many rows were skipped as out of band, CF-2401-124 is open); after a seed timeout there is a WARNING `context_recorder.seed_timeout` and no `seeded` line. A missing migration gives a WARNING `context_store_load_failed` at the seed and, at the first write attempt, an INFO `context_recorder.disabled` with `reason="table_missing"`, then an hourly re-check and `context_recorder.enabled` when the table appears. Polling and freshness are unaffected.
- Failures: `context_recorder.write_failed` (WARNING, `rows`), `context_recorder.writer_error`, `context_recorder.startup_error` (ERROR; no Telegram, no Grafana: only Q-D and the log show it).
- Queue 256 with drop-oldest, batches of at most 64, daily budget 48 rows per series; every dropped row increments `trading_mtf_context_record_dropped_total{series}`.
- **Holes:** a lost row is never re-offered (the book holds the bucket). A stored row that is out of band is skipped at the seed but still marks its hour as held, so the real reading of that hour is not recorded: a one-hour gap. A lost row shows in Q-D (a day below 23); an out-of-band stored row shows in Q-C only (it still counts in Q-D). Backtests fall back to the previous bucket within the cap (never look-ahead). The no-retry decision is re-opened before recorded rows become a backtest input (CF-2401-122).
- After a host clock step, recorded rows with `available_at` between the step minus 1 h and the correction plus 1 h must be quarantined before any backtest reads them (CF-2401-121 b). The quarantine list is not built yet (CF-2401-11 is open): record the step time, its size, the correction time and that `available_at` interval in the register stage record. Never delete or update rows in `context_observations`: the table is append-only and `recorded` rows cannot be re-fetched.

## 9. Boot and shutdown order, budget

Boot (lifespan): kill switch; build the clients and the alert listener; build the recorder and **await its boot seed** (reads only the app's own database, bounded); build and start the poller (first round of each provider staggered 0 to 30 s) and then the recorder writer in its own `try`; `recover_orphaned_runs()`; the rest. The lifespan never awaits a provider.
Shutdown: `mark_shutdown_requested()` first (so runs are marked `orphaned`), then poller stop, recorder flush (5 s nominal, then every unwritten row is dropped and counted), alert listener close, `container.shutdown()`, engine-task cancel, client close.
**Budget.** The `api` `stop_grace_period` in `infra/docker-compose.yml` must satisfy: grace ≥ uvicorn `--timeout-graceful-shutdown` + 0.1 s + recorder flush (5 s, nominal) + background cancel (5 s) + run-registry cancel (5 s, currently empty) + run teardown (`asyncio.wait` in the lifespan, 30 s) + margin for the unbounded tail. Raise the grace, or lower the drain, whenever one of these terms grows; never shorten the grace on its own. The grace and drain values are set by the infra commit CF-2401-123 (`infra/docker-compose.yml`, `infra/Dockerfile.api`); read them there, not here.
- **Kill switch before a stop.** From SIGTERM on, the API is unreachable, and with it the kill switch, flatten and emergency stop. Engage the kill switch or flatten **before** `$DC stop api` or `$DC up -d`, not during the shutdown. A hung api can take up to the full `stop_grace_period` to stop, and no `emergency_stop` is possible during that time. `$DC kill api` stops it at once but skips `engine.stop()`: open orders stay open and the run row is marked `orphaned` only at the next boot. Use it only for an api confirmed hung.
- **After a SIGKILL.** If the api is stopped by SIGKILL (grace exceeded, `docker kill`, a second Ctrl-C on a foreground `docker compose up`, or a host/daemon shutdown), a live run may stay `running` with open exchange orders, and unflushed recorder rows are lost. Q-D coverage is the control for those rows. After the restart, check that the run is `orphaned` (boot recovery, never auto-resumed), review its open orders on the exchange, and resume it. Do not sell by hand while a bot SELL is open.
- **Teardown cut short.** A run still stopping when the lifespan's 30 s teardown wait ends is cancelled a second time when the event loop closes. That cuts `engine.stop()` short, so some of its open orders may stay open. The `orphaned` status and its audit row are still written. The live-resume order-cancel scan handles the remaining orders.
- **Unbounded tail.** After the lifespan has finished, the interpreter still waits, with no time limit, for leftover tasks and worker threads, including an in-flight retraining thread. A SIGKILL at that point loses only that retraining result. The runs are already stopped and marked, and the database pool is closed.
- **Host-level stop limits (CF-2401-141).** The Docker daemon's `shutdown-timeout` (default 15 s) and a host systemd `TimeoutStopSec` are outside the repo and can kill the api sooner than its grace. Before a host reboot or a Docker daemon restart, stop the api explicitly with `$DC stop api` and wait for it to exit.

## 10. Queries (run on the server; read-only; expected result in brackets)

Run each with `$DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' <<'SQL' ... SQL`. All four read `context_observations` (migration 021; columns `series_id`, `observed_at`, `available_at`, `available_at_basis`, `value`, `ingested_at`).

**Q-A duplicate bucket** [0 rows; a hit is a second vintage of one hour bucket: the same series and `observed_at` with a later `available_at` and possibly another value, typically written after a restart within that hour whose seed failed or timed out. A valid point-in-time row and a known parity residual, not corruption: a later seed keeps the earliest row, a backtest can read the later one from its `available_at`. Accepted for R1 (CF-2401-46); note each hit with series, `observed_at` and the restart in the register stage record]:
```sql
SELECT series_id, observed_at, count(*) FROM context_observations
WHERE available_at_basis = 'recorded' GROUP BY 1, 2 HAVING count(*) > 1;
```
**Q-B stamp sanity** [0 rows; cannot detect a host clock step]:
```sql
SELECT series_id, observed_at, available_at, ingested_at FROM context_observations
WHERE available_at_basis = 'recorded'
  AND (available_at > ingested_at + interval '5 minutes'
    OR available_at < ingested_at - interval '2 minutes');
```
**Q-C band** [0 rows; a hit is a tampered or foreign row that the seed skips silently]. The limits are **read from the code at the release sha**, never typed in: a later release that tightens a band leaves rows written under the old band, which the seed skips silently and a typed copy would not see. The heredoc is unquoted so the shell inserts the three values; empty values make psql fail (`ON_ERROR_STOP`), they do not pass:
```bash
read -r DMIN DMAX WMAX < <(python3 -I -c 'import runpy,sys; g=runpy.run_path(sys.argv[1]); print(g["DOMINANCE_MIN_EXCLUSIVE"], g["DOMINANCE_MAX"], g["WHALE_ABS_MAX"])' "$DEPLOY_DIR/packages/data/provider_bands.py")   # aea028b: 0 100 1E+12
$DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' <<SQL
SELECT series_id, observed_at, value FROM context_observations
WHERE (series_id = 'coingecko:btc_dominance' AND NOT (value > $DMIN AND value <= $DMAX))
   OR (series_id = 'whale:net_flow_1h' AND abs(value) > $WMAX);
SQL
```
**Q-D coverage** [each recorded series configured in the stack (whale only when `WHALE_ALERT_API_KEY` is set): at least 23 rows per complete UTC day; the first and last day are partial; a configured series or a day missing from the output has 0 rows and fails; an out-of-band stored row still counts here]:
```sql
SELECT series_id, (available_at AT TIME ZONE 'UTC')::date AS day_utc, count(*) AS n
FROM context_observations
WHERE available_at_basis = 'recorded' AND available_at >= now() - interval '8 days'
GROUP BY 1, 2 ORDER BY 1, 2;
```
Also in the soak: `increase(trading_mtf_context_record_dropped_total[1d])` is 0 outside a shutdown flush, and the log has no `context_recorder.disabled` and no `context_recorder.startup_error` line.

## 11. Provider facts (CF-2401-39, before the R1 deploy; operator, host with provider egress)

All provider numbers in the register are unverified until this is done. Paste only the header lines. FGI and CoinGecko need no key:
```bash
for u in 'https://api.alternative.me/fng/?limit=30' 'https://api.coingecko.com/api/v3/global'; do
  curl -sS -o /dev/null -D - -H 'Accept-Encoding: identity' "$u" | grep -i -E '^(HTTP/|content-type:|content-encoding:|location:)'
done
```
Expected per provider: status 200, a JSON content type, no `content-encoding` (identity honoured), no `location` (no redirect). For CoinGecko also fetch the body once and note only: is `data.updated_at` present, is it a number, how old is it against `date` (this picks the section 2 point 3 branch), are the 24h change fields present. For FRED and Whale Alert the key stays inside the api container (`FRED_API_KEY`, `WHALE_ALERT_API_KEY` are in its environment) and goes to curl on stdin, never in argv; only the header lines are printed, and a `location` value is replaced by `(present, value withheld)` because a redirect target can repeat the query string with the key:
```bash
$DC exec -T api sh -c 'printf "url = \"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&sort_order=desc&limit=5&file_type=json&api_key=%s\"\n" "$FRED_API_KEY" | curl -sS -o /dev/null -D - -H "Accept-Encoding: identity" --config - | LC_ALL=C grep -i -E "^(HTTP/|content-type:|content-encoding:|location:)" | sed -E "s/^[Ll][Oo][Cc][Aa][Tt][Ii][Oo][Nn]:.*/location: (present, value withheld)/"'
$DC exec -T api sh -c 'printf "url = \"https://api.whale-alert.io/v1/transactions?min_value=1000000&limit=100&start=%s&api_key=%s\"\n" "$(( $(date +%s) - 3600 ))" "$WHALE_ALERT_API_KEY" | curl -sS -o /dev/null -D - -H "Accept-Encoding: identity" --config - | LC_ALL=C grep -i -E "^(HTTP/|content-type:|content-encoding:|location:)" | sed -E "s/^[Ll][Oo][Cc][Aa][Tt][Ii][Oo][Nn]:.*/location: (present, value withheld)/"'
$DC exec -T api sh -c 'printf "url = \"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&sort_order=desc&limit=5&file_type=json&api_key=%s\"\n" "$FRED_API_KEY" | curl -sS -H "Accept-Encoding: identity" --config - | python -c "import json,sys; d=json.load(sys.stdin); o=d.get(\"observations\"); print(\"observations:\", type(o).__name__, len(o) if isinstance(o, list) else \"-\", \"dot_values:\", sum(1 for r in (o if isinstance(o, list) else []) if isinstance(r, dict) and r.get(\"value\") == \".\"))"'
$DC exec -T api sh -c 'printf "url = \"https://api.whale-alert.io/v1/transactions?min_value=1000000&limit=100&start=%s&api_key=%s\"\n" "$(( $(date +%s) - 3600 ))" "$WHALE_ALERT_API_KEY" | curl -sS -H "Accept-Encoding: identity" --config - | python -c "import json,sys; d=json.load(sys.stdin); t=d.get(\"transactions\"); print(\"result:\", d.get(\"result\") if isinstance(d.get(\"result\"), str) else type(d.get(\"result\")).__name__, \"transactions:\", type(t).__name__, \"count_type:\", type(d.get(\"count\")).__name__)"'
```
The two body commands print structure only (types and counts, never values or keys): FRED `observations` must be a list and `dot_values` counts rows whose `value` is `.`; Whale Alert `result` must be `success` and `transactions` a list. The decision (data-pipeline-engineer + security) and the result are recorded in the register at P9.

## 12. R1 deploy checks (G-D2, operator)

**Before the deploy (a failure means do not deploy):**
- [ ] RK-P0-05 query of `Documentation/CF-24-01-register.md` (R0 block): 0 rows. Run it also when R0 is already deployed: the R1 restart orphans every live run again, and a hit (for example a consumer run resumed with `mode=protective`) needs the same plan before the deploy (protective resume after it, or flatten and stop now).
- [ ] SEC-04 query of `Documentation/WP2.4-register.md` section 6: 0 rows.
- [ ] Section 11 done and judged.
- [ ] Migration 021 planned: it is applied with the new image after the api stop (deploy section 4.3 pattern; the image entrypoint also runs `alembic upgrade head`); after the start `alembic current` prints `021 (head)`.
- [ ] `LOG_LEVEL` is INFO, not DEBUG and not above INFO (`$DC exec -T api printenv LOG_LEVEL`, section 6). Time-sync service active (`timedatectl show -p NTPSynchronized --value` prints `yes`).
**First start (a failure means rollback per the deploy runbook):**
- [ ] Grafana log: no provisioning error; the alert list shows all 7 `trading-safety` rules (CF-2401-113). If not, the 4 older critical rules are affected too: release blocker.
- [ ] Section 1: `alive: true`, and every configured series reaches a successful poll: `last_outcome` `ok` or `unchanged` (both are success, `_SUCCESS` in `apps/api/services/context_poller.py`; dominance and whale show `unchanged` until the next hour bucket when the boot seed already holds the current one). Judge after the first round (it starts within 30 s of boot): `auth_failed`, `egress_blocked`, `invalid_payload`, `oversize`, `http_3xx` or `http_4xx` fails (except whale `invalid_payload` with `last_reason` `truncated`, a known rate, CF-2401-37); a transient outcome (`timeout`, `connect_error`, `http_5xx`, `http_429`, `budget`) is re-checked after that series' next cycle (FRED retries only after 6 h) before you decide. Write down the time from the `context_poller.started` line to the first successful poll per series from repeated section 1 calls (successful polls log only at DEBUG); after an unattended restart use the Prometheus measure below (CF-2401-111 b).
- [ ] A `context_recorder.seeded` line with `reason` null; `reason` `store_unavailable` (a failed load also logs a WARNING `context_store_load_failed`), or a `context_recorder.seed_timeout` line in place of `seeded` (a timeout logs no `seeded` line), fails this check. No `context_recorder.disabled` line once dominance or whale has had its first `ok` poll (the recorder writes only then; `disabled` comes at the first write attempt, not at boot).
**Daily during Stage 1** (feeds Stage 2 entry; a failure restarts the Stage 1 clock or blocks Stage 2 entry; record it in the register stage record): Q-A, Q-B, Q-C return 0 rows (a Q-A hit is noted as section 10 says; a Q-B or Q-C hit is a failure); Q-D at least 23 per series and complete UTC day; `trading_mtf_context_record_dropped_total` flat outside a shutdown flush; flap count (section 5); the whale `truncated` rate (query below, CF-2401-37); per series, the count and longest duration of non-fresh stretches with no `mtf_context.status_changed` line while consumer runs are active, and fresh coverage over active-run time (CF-2401-111 b); a stretch longer than 6 h makes the poller-side "never fresh after N cycles" signal mandatory before Stage 2; if section 2 point 3 took the absent branch, `btcDominance` in `/api/v1/signals/current` has moved since the day before; the time-sync journal (`journalctl -u chrony -u systemd-timesyncd --since yesterday`, look for step lines) shows no step above 300 s (else section 8 quarantine); section 1 after every restart, and the duration of every `$DC stop api` (one close to the `stop_grace_period` is a CF-2401-123 finding for devops, section 9).
**G-E measures (per configured series; Prometheus keeps 30 d).** Poll success = attempts with outcome `ok` or `unchanged` divided by all attempts (`_SUCCESS` in `apps/api/services/context_poller.py`; `budget`, a timeout before any request and every `invalid_payload`, including `truncated`, are failures): `sum by (series) (increase(trading_mtf_context_poll_total{outcome=~"ok|unchanged"}[7d])) / sum by (series) (increase(trading_mtf_context_poll_total[7d]))`, pass >= 0.99. The counters are not zero-initialised: `increase()` undercounts a `{series,outcome}` label set that first appears inside the window, so list the label sets present in the window and check each new failure outcome by its raw values.
Whale `truncated` rate: `sum(increase(trading_mtf_context_value_rejected_total{series="whale:net_flow_1h",reason="truncated"}[1d])) / sum(increase(trading_mtf_context_poll_total{series="whale:net_flow_1h"}[1d]))` (both sides summed so the label sets match).
**After an unattended restart** (CF-2401-111 b): the time to the first successful poll per series runs from the `context_poller.started` line to the first scrape at which `trading_mtf_context_poll_total{series="<id>",outcome=~"ok|unchanged"}` exists again (the counter restarts from 0 with the process and is created at the first success, so the resolution is one scrape interval). Successful polls log only at DEBUG, and the age gauge is no measure for a seeded series (a seed younger than one cadence reads 0 s).
