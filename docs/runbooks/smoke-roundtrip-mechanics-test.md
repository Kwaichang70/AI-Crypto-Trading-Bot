# Runbook — WP-SMOKE Real-Money Round-Trip Mechanics Test (`smoke_roundtrip`)

**Audience:** The single human operator who runs the real-money Coinbase mechanics test (VP2 herstart-protocol step 2, decision D2).

**What it proves:** The live order path works end to end at about €9 notional: a BUY is placed and filled, the fill and position are recorded, the engine places the SELL, the run ends flat, the ledger matches the exchange, the PnL is about −(2 × taker + spread), and "stop while holding" works with flatten.

**What it does NOT prove:** Any trading edge. `smoke_roundtrip` has none by design.

**Source of truth:** This runbook consolidates `reports/vp2-smoke/synthesis-spec.md` §10–§12, as amended by the operator conditions O-1..O-9 and deploy conditions D-1..D-6 in `reports/vp2-smoke/final-synthesis-smoke.md` (Round 2 addendum, R2-7/R2-8), and by `reports/vp2-docs/final-synthesis-docs.md` R-01 (O-3a), R-02 (AM-S12a), R-04 (live window) and R-09 (P12 interpretation). It closes carry-forward CF-SMK-S15. If this runbook and those three documents ever disagree, stop and escalate. Do not pick one yourself.

**Related:** WP1.7a/b (stop + flatten, stop dialog), WP1.8 (orphaned / resume), WP7.0 (Idempotency-Key), SEC-004 (`X-Live-Confirm-Token`), `docs/runbooks/deploy-fase1-minimumset.md` (deploy; must be complete first), `docs/runbooks/circuit-breaker-halt-auto-stop.md`.

---

## 0. Ground rules (read first)

- **Agents never start live runs.** Every live step below is done by you.
- **Secrets.** The confirm token, the API key and the admin key are never pasted into chat, tickets, files (other than the server `.env`, mode 600), screenshots or logs. Every command below reads them from shell variables and delivers them to `curl` through the `hdrs` helper (§2a), so they never appear in any process argument list. Do not run the commands with `set -x`. Do **not** `export` the secret variables. Do not run the session under `script` or any terminal recording.
- **Live window.** `ENABLE_LIVE_TRADING` is `false` except between §4c step 0 (open) and §4e (close). The window is never left open unattended. §4e runs on every exit path (Run B done, any abort, any escalation, a missed Run B window, INCONCLUSIVE unless the one permitted repeat follows in the same sitting, and in every case at the end of the sitting).
- **One operator, one client, one live run.** From the moment you create a smoke run until it is `stopped`, issue **no** other live create, promote or resume (O-4, CF-SMK-S11). The server refuses a smoke create while another live run exists, but it does **not** yet refuse other live creates while a smoke run exists.
- **Never** use `POST /runs/{id}/resume?mode=normal` for a smoke run. The server rejects it with 422 `smoke_resume_protective_only`.
- **Never** use "Max" when selling on Coinbase by hand if the account held the base coin before the test (`pre_total > 0`).
- `smoke_roundtrip` is hidden from the dashboard's strategy list (status `diagnostic`). Runs are created with `curl` only. The dashboard is used for the Run B stop dialog.

---

## 1. Purpose and scope

| Item | Value |
|---|---|
| Strategy | `smoke_roundtrip` (`packages/trading/strategies/smoke_roundtrip.py`) |
| Behaviour | BUY `notional_quote` on the first bar it processes, wait `hold_bars` bars, then emit SELL `target_position=0` on up to `exit_retry_bars` consecutive bars, then go silent for good. At most one BUY per strategy instance. |
| Safety net | Mandatory fixed stop-loss `bracket_stop_loss_pct` (0.05 in this procedure). No take-profit, no trailing stop. |
| Exchange / pair | Coinbase, one EUR pair. Default **XRP/EUR**; LTC/EUR is the alternative. |
| Timeframe | **5m** (1m is allowed by the guard but not used here). |
| Capital / notional | `initialCapital` **65** (must be backed by at least €65 free EUR on Coinbase); `notional_quote` **9.0**. |
| Runs | Paper rehearsal (SMK-T-34), then **Run A** (round trip, `hold_bars=1`), then **Run B** (stop while holding, `hold_bars=6`, stopped through the UI dialog with flatten). |

### 1a. Server-side guardrails you will run into (code: `packages/trading/smoke_guard.py`, `apps/api/routers/runs.py`)

A rejected create returns 422 `{"detail":{"code":"smoke_guardrail_violation","errors":[{"field","reason","value","min","max","message"}]}}` and writes nothing (no run row, no audit row, no idempotency claim).

| Rule | Bound | `reason` on violation |
|---|---|---|
| Symbols | exactly 1 | `too_many_symbols` |
| Timeframe | `1m` or `5m` | `timeframe_not_allowed` |
| Quote (live) | `EUR` | `quote_not_allowed` |
| `initialCapital` (live) | 60.00 – 66.00 | `capital_out_of_smoke_range` |
| `notional_quote` vs capital (all modes) | ≤ 0.15 × `initialCapital` | `notional_exceeds_risk_ceiling` |
| `notional_quote` | 5.00 – 9.50 | `param_out_of_range` |
| `hold_bars` | integer 1 – 12 | `param_out_of_range` |
| `exit_retry_bars` | integer 1 – 5 | `param_out_of_range` |
| Other strategy params | none allowed | `unknown_param` |
| `bracket_stop_loss_pct` | required, 0.03 – 0.08 | `stop_loss_required` / `stop_loss_out_of_range` |
| `bracket_mode` | absent or `fixed` | `bracket_mode_must_be_fixed` |
| TP / ATR multipliers | absent | `take_profit_not_allowed` |
| `trailing_stop_pct` | absent | `trailing_not_allowed` |
| `allowPyramiding` / `enableAdaptiveLearning` | not `true` | `pyramiding_not_allowed` / `adaptive_learning_not_allowed` |

Other smoke-specific responses:
- 409 `{"code":"smoke_requires_exclusive_live","conflicting_run_ids":[...]}`: another live run is `running`, `orphaned` or `resuming`.
- 422 `{"code":"smoke_resume_protective_only"}`: `resume?mode=normal` on a smoke run.
- 422 `{"code":"smoke_promotion_forbidden"}`: promote-to-live from a smoke paper run.

---

## 2. Prerequisites (checklist — every box must be ticked before Run A)

### 2a. Shell setup (on the server, via Tailscale SSH, one bash shell)

Every command in this runbook runs on the server, in one bash shell, as the deploy user who owns `.env`. Browser steps (Coinbase portal, the dashboard stop dialog) happen on your own machine. Redefine everything below in every new shell.

```bash
# Base URL of the API (placeholder -- use your own host)
export API_BASE="https://<your-host>/api/v1"
export DEPLOY_DIR=/opt/trading-bot
DC="docker compose -f $DEPLOY_DIR/infra/docker-compose.yml --env-file $DEPLOY_DIR/.env"

# The only secrets file is $DEPLOY_DIR/.env; never copy it.
[ "$(stat -c '%a %U' "$DEPLOY_DIR/.env")" = "600 $(id -un)" ] || echo "STOP: .env mode/owner is not 600 $(id -un) -- escalate"
[ ! -L "$DEPLOY_DIR/.env" ] || echo "STOP: .env is a symlink -- escalate"
find "$DEPLOY_DIR" -maxdepth 1 -name '.env.*' ! -name '.env.example'   # expect no output; any file listed may hold secrets: remove it
test ! -e "$DEPLOY_DIR/infra/.env" || echo "STOP: infra/.env exists -- escalate"

# Tools: all four must resolve; curl must be 7.55.0 or later (for -H @file)
command -v jq uuidgen openssl curl
curl --version | head -1

# Helper 1: change one .env value. The value arrives on stdin, never in argv, and is never printed.
# Writes through mktemp (O_EXCL, mode 600, never follows a symlink), then enforces mode 600 and owner.
env_set() {  # usage: printf '%s\n' VALUE | env_set NAME   -- on any problem prints "STOP: ..." and returns 1
  local name=$1 f="$DEPLOY_DIR/.env" val tmp
  IFS= read -r val && [ -n "$val" ] || { echo "STOP: no value for ${name}"; return 1; }
  [ -f "$f" ] && [ ! -L "$f" ] || { echo "STOP: $f missing or a symlink"; return 1; }
  [ "$(grep -c "^${name}=" "$f")" -le 1 ] || { echo "STOP: duplicate ${name} lines in .env"; return 1; }
  tmp=$(mktemp "$f.XXXXXX") || { echo "STOP: mktemp failed"; return 1; }
  { grep -v "^${name}=" "$f"; [ $? -le 1 ] && printf '%s=%s\n' "$name" "$val"; } > "$tmp" \
    && chmod 600 -- "$tmp" && mv -f -- "$tmp" "$f" \
    || { rm -f -- "$tmp"; echo "STOP: env_set ${name} failed"; return 1; }
  [ "$(stat -c '%a %U' "$f")" = "600 $(id -un)" ] || { echo "STOP: .env mode/owner is not 600 $(id -un)"; return 1; }
}

# Helper 2: print the requested auth headers. Values never reach any process argv.
hdrs() {  # usage: curl ... -H @<(hdrs api admin live)   (always inline; never store <(...) in a variable)
  local h
  for h in "$@"; do
    case $h in
      api)   [ -n "${API_KEY:-}" ]            && printf 'X-API-Key: %s\n'            "$API_KEY" ;;
      admin) [ -n "${ADMIN_KEY:-}" ]          && printf 'X-Admin-Key: %s\n'          "$ADMIN_KEY" ;;
      live)  [ -n "${LIVE_CONFIRM_TOKEN:-}" ] && printf 'X-Live-Confirm-Token: %s\n' "$LIVE_CONFIRM_TOKEN" ;;
    esac
  done
  return 0
}

# Secrets: typed, never echoed, never stored in a file or in shell history.
# Do not export them. The live confirm token is NOT typed: it is loaded from .env when the window opens (§4c step 0).
read -rs -p "X-API-Key (empty if auth is off): " API_KEY; echo
read -rs -p "X-Admin-Key: " ADMIN_KEY; echo
```

The admin key is mandatory: the kill switch (§6 step 0 and step 5) needs it. Empty variables emit no header, so a request without auth fails closed with 401/403.

> Any line that starts with `STOP:` ends the current section: do not run the next command. Follow the instruction on that line, or escalate.

**New shell while the window is open** (between §4c step 0 and §4e, for example after an SSH drop): redefine this block, then load the token with the lines below. **Never** re-run §4c step 0 while a live run exists: its recreate turns the run into `orphaned`.
```bash
LIVE_CONFIRM_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
[ -n "$LIVE_CONFIRM_TOKEN" ] || echo "STOP: token not loaded -- escalate"
```

`jq` is used below to filter JSON. Every `curl` also works without it.

### 2b. Deploy conditions (D-0..D-6)

- [ ] **D-0** The deploy runbook `docs/runbooks/deploy-fase1-minimumset.md` §5 passed and `alembic current` = `019 (head)` (deploy log, DEP §5.9).
- [ ] **D-1** The deployed code contains WP-SMOKE (commit `3db9148` or a descendant) and the acceptance suite is green (`reports/vp2-smoke/acceptance-report-smoke.md`).
- [ ] **D-2** SMK-T-37 is green on real Postgres (recorded in the acceptance report).
- [ ] **D-3** If D-2 is unmet, O-3 (§7) is the sole control for the create race. Record that in the evidence pack.
- [ ] **D-4** The API runs with `DEBUG=false` (checked by the pre-flight script in §2e).
- [ ] **D-5** No live run is `running`/`orphaned`/`resuming` at deploy time. `ENABLE_LIVE_TRADING` is `false` except inside the live window (§4c step 0 to §4e).
- [ ] **D-6** The deployed image contains the G-9 advisory lock:
  ```bash
  $DC exec -T api grep -c "_SMOKE_LIVE_CREATE_LOCK_KEY" /app/api/routers/runs.py   # expect >= 1
  ```
  The server has no git history (the sync excludes `.git`). On the **source machine** run `git rev-parse HEAD`; it must equal the commit recorded in DEP §5.8. Put both in the evidence pack.

### 2c. Operator decisions (O-1, O-2)

- [ ] **O-1 / D-SMK-1** You confirm: capital **€65**, notional **€9.00**, and at least **€65 free EUR** on Coinbase. If you will not fund €65, **stop here and escalate** (route via CF-SMK-S4), then §4e if the window is open. Never lower the guard.
- [ ] **O-2 / D-SMK-4** You confirm the pair (**XRP/EUR** default, LTC/EUR alternative). You are aware that the Coinbase taker fee for a small account may be **1.2%**, not the 0.6% the cost model assumes (SMK-R-03). This changes only the expected PnL band (§5), not pass/fail logic.

### 2d. Exchange keys, live OFF (phase 1 of the live window; edited by you, never by an agent)

The live gate has three layers; all must pass or `POST /runs` returns 403 `Live trading gate check failed. Failed layers: ...`. This step supplies the exchange keys only. The flag and the confirm token are set later, immediately before Run A (§4c step 0).

Precondition: run the "No other live run" query from the first bullet of §2e. It must be empty, because recreating the api container turns a live run into `orphaned`.

- [ ] The Coinbase key has view + trade permission only, never transfer (see `docs/gebruikershandleiding.md` §2).
- [ ] The downloaded `cdp_api_key*.json` is a secret: move it out of Downloads, never into a repo, and delete it after pasting into `.env`.
- [ ] If the CDP portal offers an IP allowlist, restrict the key to the server's egress IP.
- [ ] Disable the CDP key between test windows (§4e).
- [ ] Never screenshot the CDP API-key pages.

```bash
printf 'coinbase\n' | env_set EXCHANGE_ID
printf 'false\n'    | env_set ENABLE_LIVE_TRADING
# EXCHANGE_API_KEY / EXCHANGE_API_SECRET: paste with an editor that writes no backup file
# (nano; remove any .env~ / .env.swp afterwards). Never echo them and never pass them as arguments.
$DC up -d --no-deps --force-recreate api
$DC ps api                                                   # wait for "healthy" (start_period 60 s)
$DC exec -T api printenv ENABLE_LIVE_TRADING EXCHANGE_ID     # expect: false / coinbase
```

- [ ] `EXCHANGE_ID=coinbase` (the compose default is `binance`).
- [ ] Changing `.env` requires recreating the api container (done above). Do this **only while no live run exists**: a restart turns a live run into `orphaned`.
- [ ] Never send the confirm token in the request body (deprecated fallback). It goes in the `X-Live-Confirm-Token` header via `hdrs live`.

### 2e. Pre-flight measurements (read-only; record every value in the evidence pack)

- [ ] **No other live run** (O-3). The output must be empty:
  ```bash
  curl -sS -H @<(hdrs api) \
    "$API_BASE/runs?mode=live&include_archived=true&limit=500" \
    | jq '.items[] | select(.status=="running" or .status=="orphaned" or .status=="resuming") | {id, status}'
  ```
- [ ] **Kill switch not latched.** Expect `"latched": false` and `"source": "db"`:
  ```bash
  curl -sS -H @<(hdrs api) "$API_BASE/emergency/kill-switch"
  ```
- [ ] **Exchange limits, fees, spread and balances.** The script below runs inside the api container, uses the same exchange construction as the live engine, places **no** orders, and prints no secrets. It nulls every order, transfer and withdraw method (both the snake_case and the camelCase spelling that ccxt exposes) before the first network call. Paste it as is; the heredoc terminator `PY` must start in column 0:

```bash
$DC exec -T api python - <<'PY'
import asyncio
from api.config import get_settings
from api.services.run_orchestrator import build_live_ccxt_exchange

SYMBOL = "XRP/EUR"   # change to LTC/EUR if that is the chosen pair
BLOCKED = ["create_order", "create_market_order", "create_limit_order", "edit_order",
           "cancel_order", "cancel_all_orders", "withdraw", "transfer"]


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


async def main() -> None:
    ex = None
    try:
        s = get_settings()
        print("exchange_id:", s.exchange_id, "| enable_live_trading:", s.enable_live_trading, "| debug:", s.debug)
        ex = build_live_ccxt_exchange(s)
        # Read-only guard: any call to an order/transfer method now raises TypeError.
        for n in BLOCKED:
            for attr in (n, camel(n)):
                setattr(ex, attr, None)
        # ex.verbose must stay False (it would print request headers, i.e. the credentials).
        await ex.load_markets()
        m = ex.market(SYMBOL)
        print("limits.cost.min  :", m["limits"]["cost"]["min"])
        print("limits.amount.min:", m["limits"]["amount"]["min"])
        print("precision.amount :", m["precision"]["amount"], "(= 1 step)")
        print("taker / maker    :", m.get("taker"), "/", m.get("maker"))
        t = await ex.fetch_ticker(SYMBOL)
        bid, ask = t.get("bid"), t.get("ask")
        if bid and ask:
            print("bid / ask        :", bid, "/", ask, "| spread %:", round((ask - bid) / ((ask + bid) / 2) * 100, 4))
        else:
            print("bid/ask not in ticker -- read the spread from the Coinbase order book instead")
        b = await ex.fetch_balance()
        base = SYMBOL.split("/")[0]
        print("EUR free / total :", (b.get("EUR") or {}).get("free"), "/", (b.get("EUR") or {}).get("total"))
        print(base, "total (pre_total):", (b.get(base) or {}).get("total"))
    except Exception as exc:
        print("pre-flight check failed:", type(exc).__name__)
    finally:
        if ex is not None:
            await ex.close()


asyncio.run(main())
PY
```

  Required results:
  - [ ] `exchange_id: coinbase`, `debug: False`.
  - [ ] `limits.cost.min` ≤ **€2**.
  - [ ] Spread ≤ **0.5%**.
  - [ ] EUR free ≥ **€65**.
  - [ ] Base coin total = **0**. If it cannot be 0, record it as `pre_total` and never use "Max" in a manual sell.
  - [ ] Record `taker` (= `t` in §5), `maker`, `limits.amount.min`, `precision.amount` (= 1 step).

  If any required result fails, do not start. For a spread above 0.5%, wait or switch to the alternative pair (then re-run everything in §2e). If the pre-flight cannot be completed at all (O-1 unmet, or the script fails), stop here and escalate; if the window is open, go to §4e.

---

## 3. Paper rehearsal (SMK-T-34) — mandatory before any live run

Same body as Run A, but `"mode":"paper"` and no confirm token. `ENABLE_LIVE_TRADING` is `false` (phase 1).

```bash
KEY_P=$(uuidgen); echo "KEY_P=$KEY_P"      # record it; it is not a secret
curl -sS -i -X POST "$API_BASE/runs" \
  -H @<(hdrs api) \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $KEY_P" \
  -d '{"strategyName":"smoke_roundtrip","mode":"paper","symbols":["XRP/EUR"],"timeframe":"5m",
       "initialCapital":"65","allowPyramiding":false,"enableAdaptiveLearning":false,
       "strategyParams":{"notional_quote":9.0,"hold_bars":1,"exit_retry_bars":4,
                         "bracket_mode":"fixed","bracket_stop_loss_pct":0.05}}'
```

Expect 201, `Idempotent-Replay: false`, `"status":"running"`. Store the id: `RUN_P=<id>`.

Pass (O-7): **one** run shows exactly one BUY of about €9.00, then one SELL, then flat:
```bash
curl -sS -H @<(hdrs api) "$API_BASE/runs/$RUN_P/orders"    | jq '.items[] | {side, status, quantity, averageFillPrice, createdAt}'
curl -sS -H @<(hdrs api) "$API_BASE/runs/$RUN_P/positions" | jq '.'
```
Then stop it (paper needs no flatten decision): `curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN_P"`.

If the paper run ends in `error`, it is **not** retried automatically (by design). That is a failed rehearsal: root-cause it and do not go live.

---

## 4. Live runs

### 4a. Timeline to expect (5m timeframe, TF = 300 s)

| Event | Expected time after the create returns (T0) |
|---|---|
| `live.sync_positions_completed` at engine start | seconds |
| First strategy bar → BUY submitted | about **T0 + 10–15 min** (2nd–3rd bar close after start) |
| SELL (Run A, `hold_bars=1`) | one processed bar after the BUY bar: about **+5 min**, may be up to about +6 min |
| P6 limit (Run A) | SELL `createdAt` − BUY fill time ≤ (1 + 2) × 5 min = **15 min** |
| Strategy SELL window (Run B, `hold_bars=6`) | starts about **30 min** after the BUY bar. You stop Run B before that. |
| "No order" abort trigger | no BUY by the expected time + 3 × TF, i.e. about **T0 + 30 min** |
| "Engine silence" abort trigger | no engine activity for more than 2 × TF = **10 min** |

### 4b. Where to observe

| Surface | How | What for |
|---|---|---|
| Orders | `GET $API_BASE/runs/$RUN/orders` | side, `status` (`filled`), `clientOrderId` (starts with `<run_id>-`), `exchangeOrderId`, `createdAt` |
| Fills | `GET $API_BASE/runs/$RUN/fills` | `quantity`, `price`, `fee`, `feeCurrency` (expect `EUR`), `executedAt` |
| Positions | `GET $API_BASE/runs/$RUN/positions` | the held qty (only non-flat positions are listed; flat = empty list) |
| Portfolio | `GET $API_BASE/runs/$RUN/portfolio` | `currentCash`, `openPositions` (0 when flat), `totalRealisedPnl`, `totalFeesPaid` |
| Trades | `GET $API_BASE/runs/$RUN/trades` | `strategyId`, `realisedPnl`, `quantity`, `entryPrice`, `exitPrice` |
| Run | `GET $API_BASE/runs/$RUN` | `status` |
| Coinbase | Advanced Trade → Orders (filled / open) for the pair; Portfolio balances | the same orders (match on `exchangeOrderId`), commission, balances |
| API logs | see below | engine events, stop signals |

The trade's `exit_reason` is not exposed by the API. Read it from the `engine.trade_recorded` log line (field `exit_reason`).

Log follow command (event names are exact; run it in a second shell on the server, after redefining `DC`):
```bash
$DC logs -f --since 10m api 2>&1 | grep -E \
 'smoke_roundtrip\.|live\.(engine_started|sync_positions_completed|initial_capital_exceeds_free_quote|below_min_cost|below_min_amount|buy_blocked_|nav_unavailable|reconcile_required|run_cash_exceeds_exchange_free|external_holdings_ignored|sell_no_position|sell_inflight_pending|sell_capped_to_zero)|own_exceeds_exchange_total|engine\.trade_recorded|flatten\.incomplete|runs\.(stop|smoke|stopped)'
```

**Stop signals — any one of these means: go to §6 (abort) now:**
`live.below_min_cost`, `live.below_min_amount`, any `live.buy_blocked_*`, `live.nav_unavailable`, `own_exceeds_exchange_total`, `live.reconcile_required`, `live.initial_capital_exceeds_free_quote` (at start), `live.run_cash_exceeds_exchange_free`, `flatten.incomplete`, a **second BUY**, no order by about T0 + 30 min, or engine silence for more than 10 min.

Expected benign lines: `smoke_roundtrip.entry`, `smoke_roundtrip.exit_attempt` (attempt 1..4), `live.sell_no_position` on retry bars after the SELL filled, and one WARNING `smoke_roundtrip.exit_window_closed` at the end of the window. That warning is normal on a successful run.

### 4c. Run A — round trip (`hold_bars=1`)

0. **Open the live window.** Preconditions: `RUN_P` is `stopped`; the §2e live-run query is empty; the kill switch is unlatched; §2e and §3 passed. The operator never types or sees the confirm token: a fresh one is generated for this window and loaded from `.env`.
   ```bash
   PREV_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
   if openssl rand -hex 32 | env_set LIVE_TRADING_CONFIRM_TOKEN \
      && printf 'true\n' | env_set ENABLE_LIVE_TRADING; then
     $DC up -d --no-deps --force-recreate api
     LIVE_CONFIRM_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
     if [ -n "$LIVE_CONFIRM_TOKEN" ] && [ "$LIVE_CONFIRM_TOKEN" != "$PREV_TOKEN" ]; then
       echo "window token rotated: yes"
     else
       echo "STOP: token not loaded or not rotated -- run §4e, then escalate"
     fi
   else
     echo "STOP: window NOT opened -- run §4e, then escalate"
   fi
   unset PREV_TOKEN
   $DC ps api                                                   # wait for "healthy"
   $DC exec -T api printenv ENABLE_LIVE_TRADING EXCHANGE_ID     # evidence; MUST print: true / coinbase
   [ "$($DC exec -T api printenv ENABLE_LIVE_TRADING)" = true ] || echo "STOP: flag is not true in the container -- run §4e, then escalate"
   date -u +%FT%TZ                                              # evidence: "window opened at"
   ```
   From here until §4e the window is open. Do not leave the sitting without closing it. Continue to step 1 only if no `STOP:` line was printed.
1. Re-check §2e bullets 1 and 2 (no live run; kill switch not latched).
2. Record the request time, then create the run with a **fresh** key:
   ```bash
   T_REQ=$(date -u +%FT%TZ); echo "T_REQ=$T_REQ"   # before EVERY live create (Run A, Run B, an INCONCLUSIVE repeat)
   KEY_A=$(uuidgen); echo "KEY_A=$KEY_A"      # keep this shell open; reuse KEY_A on any same-key retry (§7)
   curl -sS -i -X POST "$API_BASE/runs" \
     -H @<(hdrs api live) \
     -H "Content-Type: application/json" \
     -H "Idempotency-Key: $KEY_A" \
     -d '{"strategyName":"smoke_roundtrip","mode":"live","symbols":["XRP/EUR"],"timeframe":"5m",
          "initialCapital":"65","allowPyramiding":false,"enableAdaptiveLearning":false,
          "strategyParams":{"notional_quote":9.0,"hold_bars":1,"exit_retry_bars":4,
                            "bracket_mode":"fixed","bracket_stop_loss_pct":0.05}}'
   ```
   Expect **201**, headers `Idempotency-Key: <KEY_A>` and `Idempotent-Replay: false`, body `"status":"running"`, `"runMode":"live"`. Store `RUN_A=<id>`. Note T0.

   | Response | Meaning | Action |
   |---|---|---|
   | 422 `smoke_guardrail_violation` | body outside the guard (§1a) | fix the body, use a **new** key |
   | 403 `Live trading gate check failed. Failed layers: ...` | env flag / API keys / token | check §2d and step 0 (then §4e if you stop), use a new key |
   | 409 `kill_switch_active` | global kill switch latched | investigate before clearing it |
   | 409 `smoke_requires_exclusive_live` | another live run exists | resolve that run first; new key |
   | 409 `idempotency_in_progress` | same key still being processed | wait, retry with the **same** key (§7) |
   | 422 `idempotency_key_reused` | same key, different body | use a new key |
   | 428 / 400 `idempotency_key_*` | header missing / not a UUID | fix the header |
   | 5xx or network error, or a 201 replay of an `error` run | unknown whether a run exists / no engine ever ran | **follow §7 exactly** (T_REQ, look-up before every retry) |

3. Watch the start logs: `live.sync_positions_completed` with no reconcile flags, and **no** `live.initial_capital_exceeds_free_quote`.
4. At about T0 + 10–15 min, the BUY:
   - `/orders`: exactly **one** `buy`, `status: "filled"`, `clientOrderId` starting with `<RUN_A>-`.
   - Coinbase → Orders → Filled: the same order (match `exchangeOrderId`). Screenshot it with the commission.
   - `/fills`: the buy fill with `quantity`, `price`, `fee`, `feeCurrency`.
   - `/positions` within 30 s: `XRP/EUR` with the filled qty.
   - Optional P9(b) mid-hold check: re-run the §2e script; base total − `pre_total` must equal the own qty ±1 step.
5. About one bar later, the SELL:
   - `/orders`: one `sell`, `filled`, `clientOrderId` starting with `<RUN_A>-`.
   - `/trades`: one trade, `strategyId` = `smoke_roundtrip-<first 8 hex chars of RUN_A without dashes>`.
   - Log `engine.trade_recorded ... exit_reason=signal_exit`.
   - `/positions` empty, `/portfolio` `openPositions: 0`.
6. Let the exit window run out (`smoke_roundtrip.exit_window_closed`, about 4 bars after the SELL). There must be **no** further order.
7. Post-run balances: re-run the §2e script and record EUR total and base total (`post_total`).
8. Stop Run A:
   ```bash
   curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN_A"
   ```
   Expect **422** `{"detail":{"code":"flatten_decision_required","held_symbols":[]}}`.
   - `held_symbols: []` → stop without flatten:
     ```bash
     curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN_A?flatten=false"
     ```
     Expect 200 with `"status":"stopped"`, `"flatten": null`, `"unprotectedPositions": []`.
   - `held_symbols: ["XRP/EUR"]` → check `/positions`. If qty ≤ 1 step (`precision.amount`), it is dust: P12 = PASS-DEGRADED (CF-SMK-S1). Stop with `?flatten=false` and record the dust. **Do not** use `?flatten=true` on dust (see §6 step 2). If qty > 1 step, it is a real position → §6.

### 4d. Run B — stop while holding (`hold_bars=6`, UI stop dialog with flatten)

Start Run B only after Run A **passed** and is `stopped`. O-8 before Run B:
- [ ] Run A `status` = `stopped`.
- [ ] Run A final ledger qty ≤ 1 step (P8).
- [ ] Re-run §2e and **re-record** base `pre_total` and EUR balances for Run B.

1. Create with a **new** key and `"hold_bars":6`:
   ```bash
   T_REQ=$(date -u +%FT%TZ); echo "T_REQ=$T_REQ"   # before every live create
   KEY_B=$(uuidgen); echo "KEY_B=$KEY_B"
   curl -sS -i -X POST "$API_BASE/runs" \
     -H @<(hdrs api live) \
     -H "Content-Type: application/json" \
     -H "Idempotency-Key: $KEY_B" \
     -d '{"strategyName":"smoke_roundtrip","mode":"live","symbols":["XRP/EUR"],"timeframe":"5m",
          "initialCapital":"65","allowPyramiding":false,"enableAdaptiveLearning":false,
          "strategyParams":{"notional_quote":9.0,"hold_bars":6,"exit_retry_bars":4,
                            "bracket_mode":"fixed","bracket_stop_loss_pct":0.05}}'
   ```
   Store `RUN_B=<id>`. Same response handling as Run A (including §7 for a 5xx, a network error or an `error` replay).
2. Verify the BUY exactly as in Run A step 4 (P1–P4).
3. As soon as `/positions` shows the qty, and well before the strategy's own SELL window (about 30 min after the BUY bar):
   1. Optional API evidence, which changes nothing:
      ```bash
      curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN_B"
      ```
      Expect 422 `flatten_decision_required` with `"held_symbols":["XRP/EUR"]`.
   2. In the dashboard, open `/runs/<RUN_B>` and click **Stop Run**. The dialog must say this is a running **LIVE** run and require a choice. The **Stop Run** button stays disabled until you choose. Screenshot this.
   3. Choose **"Flatten (sell everything) before stopping"**, then click **Stop Run**. Flatten can take up to about 35 s.
   4. Expect the flatten result view: outcome `flattened`, complete. The run shows `stopped`. Screenshot it.
4. Verify:
   - `/orders`: one extra `sell`, `filled`, `clientOrderId` starting with `<RUN_B>-`.
   - `/trades`: `strategyId` = `operator_flatten`. (The log `exit_reason` for this trade reads `signal_exit`, because the engine's flatten reason is not one of the stored exit reasons. That is expected.) If `strategyId` is not `operator_flatten` while the dialog shows `flattened` and the flatten SELL is in `/orders`, do not mark P13 FAIL on that alone. Record it and escalate for interpretation (CF-DOC-02).
   - `/positions` empty; Coinbase base total back to `pre_total` (±1 step).
   - No further orders on Coinbase or in `/orders`.
5. If the dialog shows **409 `flatten_incomplete`**, the run is still `running` with entries latched. Go to §6 step 2 or 3.

If you miss the window and the strategy SELL fires first, Run B has not demonstrated stop-while-holding (P13 not demonstrated). Stop the run as in Run A step 8, record it, escalate, and allow no third live run. Then §4e.

After Run B: go to §4e.

### 4e. Close the window (mandatory on every exit path)

Run this after Run B, after any abort (§6 step 6), on any escalation, after a missed Run B window, after INCONCLUSIVE (unless the one permitted repeat follows in the same sitting), and at the end of every sitting. Precondition: the §2e live-run query is empty. §6 always reaches that state, because emergency-stop always stops. (Recreating the api with a live run present would orphan it.) (Only exception: §6 step 5, where the run is already `orphaned`.)

```bash
PREV_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
if printf 'false\n' | env_set ENABLE_LIVE_TRADING \
   && openssl rand -hex 32 | env_set LIVE_TRADING_CONFIRM_TOKEN; then
  NEW_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
  if [ -n "$NEW_TOKEN" ] && [ "$NEW_TOKEN" != "$PREV_TOKEN" ]; then
    echo "token rotated: yes"
  else
    echo "STOP: token not rotated -- escalate"
  fi
  $DC up -d --no-deps --force-recreate api
else
  echo "STOP: .env not updated -- stopping the api, escalate"; $DC stop api
fi
unset PREV_TOKEN NEW_TOKEN
$DC ps api                                                   # healthy (or stopped after a STOP above)
$DC exec -T api printenv ENABLE_LIVE_TRADING                 # evidence; MUST print: false
[ "$($DC exec -T api printenv ENABLE_LIVE_TRADING 2>/dev/null)" = false ] \
  || { echo "STOP: flag is not false -- stopping the api, escalate"; $DC stop api; }
date -u +%FT%TZ                                              # evidence: "window closed at"
unset LIVE_CONFIRM_TOKEN
```

- The block stops the api by itself if `.env` cannot be updated or the flag is not `false`. No live run exists at this point (precondition; for the one exception see §6 step 5), so that is safe. Escalate on any `STOP:` line.
- Afterwards, disable the Coinbase CDP key in the portal until the next window.
- Evidence (§9): open and closed timestamps, both `printenv` outputs, the `token rotated: yes` lines printed by §4c step 0 and §4e, copied from the terminal (never the value), and "CDP key disabled: yes".

---

## 5. Pass criteria P1–P13 (sign-off table)

TF = 300 s. step = `precision.amount`. N = BUY notional. t = recorded taker. s = recorded spread (fraction). δ = mid-price change between the BUY and the SELL fill (fraction).

| ID | Criterion | FAIL / other outcome | Runs | A | B | Evidence |
|---|---|---|---|---|---|---|
| P1 | Exactly 1 BUY in `/runs/{id}/orders`: `filled`, `clientOrderId` prefix `<run_id>-`, the same order in Coinbase history | A second BUY → immediate FAIL plus abort | A, B | ☐ | ☐ | |
| P2 | BUY notional (filled qty × avg price) ∈ [€8.00, €9.60] | < €8 → FAIL-sizing; partial IOC → PASS-DEGRADED | A, B | ☐ | ☐ | |
| P3 | Σ BUY fills = Coinbase filled size ±1 step; fee = Coinbase commission ±€0.01; effective rate (fee ÷ notional) within ±5 bps of t | otherwise FAIL | A, B | ☐ | ☐ | |
| P4 | `/positions` shows the P3 qty during the hold (flush lag ≤ 30 s) | otherwise FAIL | A, B | ☐ | ☐ | |
| P5 | SELL placed by the engine: trade `strategyId` starts with `smoke_roundtrip-`, `exit_reason=signal_exit`, `clientOrderId` prefix `<run_id>-` | `stop_loss` → INCONCLUSIVE; manual → FAIL | A | ☐ | — | |
| P6 | SELL `createdAt` − BUY fill `executedAt` ≤ (hold_bars + 2) × TF (Run A: 15 min) | otherwise FAIL (timing) | A | ☐ | — | |
| P7 | SELL fill recorded; qty = floor_step(own) ±1 step | otherwise FAIL | A, B | ☐ | ☐ | |
| P8 | Flat: final ledger qty ≤ 1 step and value < €0.01; `/runs/{id}/portfolio` `openPositions: 0` | otherwise FAIL | A, B | ☐ | ☐ | |
| P9 | `sync_positions` matches the exchange: (a) start log `live.sync_positions_completed`, no `reconcile_required`/I8 flags during the run; (b) \|(post_total − pre_total) − ledger_final\| ≤ 1 step, and during the hold base total − pre_total = own qty ±1 step; (c) \|ΔEUR_exchange − (run `currentCash` final − 65)\| ≤ €0.02 | otherwise FAIL | A, B | ☐ | ☐ | |
| P10 | Identity from Coinbase fills: R_engine = q_s·p_s − (q_s/q_b)(q_b·p_b + f_b) − f_s, where R_engine = `/trades` `realisedPnl` | ±€0.01, else FAIL | A, B | ☐ | ☐ | |
| P11 | PnL band: centre −N(2t + s), band ±(N·\|δ\| + €0.02) | outside the band with P10 OK and \|δ\| > 1% → INCONCLUSIVE-MARKET; R < −€1.00 → FAIL | A, B | ☐ | ☐ | |
| P12 | Run A stop: 422 `flatten_decision_required` with `[]`, then `stopped` | non-empty list → PASS-DEGRADED (dust, CF-SMK-S1) | A | ☐ | — | |
| P13 | Run B: the UI dialog asks; `flattened`; `operator_flatten` exit; P7–P11 hold for the flatten SELL | otherwise FAIL | B | — | ☐ | |

**Verdict.** WP-SMOKE live acceptance is **PASS** when Run A satisfies P1–P12 and Run B satisfies P1–P4, P7–P11 and P13 (P12 is Run A only). PASS-DEGRADED items are recorded and do not block. INCONCLUSIVE means the run is repeated once in the same sitting with a new Idempotency-Key, after re-running §2e bullets 1–2 (otherwise §4e). Record `T_REQ` before the repeat.

*Footnote (P12):* interpretation of spec §11, which lists P12 for Run B literally; confirm at sign-off (`reports/vp2-docs/final-synthesis-docs.md` R-09).

### 5a. Expected PnL — worked example at N = €9.00

R ≈ −N·(2t + s) + N·δ. Realised PnL includes both fees: the entry fee sits in the all-in entry price, and the exit fee is subtracted.

| Taker t | Spread s | 2t + s | Centre −N(2t+s) | Band at δ = 0 (±€0.02) | Band at \|δ\| = 0.2% (±€0.038) |
|---|---|---|---|---|---|
| 0.006 (cost-model value) | 0.003 | 0.015 | **−€0.135** | [−€0.155, −€0.115] | [−€0.173, −€0.097] |
| 0.012 (likely, SMK-R-03) | 0.001 | 0.025 | **−€0.225** | [−€0.245, −€0.205] | [−€0.263, −€0.187] |
| 0.012 (likely, SMK-R-03) | 0.003 | 0.027 | **−€0.243** | [−€0.263, −€0.223] | [−€0.281, −€0.205] |

Arithmetic for the last row: 2 × 0.012 + 0.003 = 0.027; 9.00 × 0.027 = 0.243; band half-width 9.00 × 0.002 + 0.02 = 0.038.

Always compute the band from the **recorded** t, s and δ, not from this table. For reference, a stop-loss exit at 5% would cost about €0.59–€0.70, and the theoretical worst case (asset to zero while stuck) is bounded by the €9.50 notional ceiling plus fees.

---

## 6. Abort procedure

Trigger: any FAIL, any stop signal from §4b, any unexpected order, or engine silence for more than 2 × TF. Work top-down; stop at the first step that leaves you flat. The live window is open during an abort, so `ENABLE_LIVE_TRADING` is `true` and `LIVE_CONFIRM_TOKEN` is loaded; step 6 ends with §4e. This is spec §12 as amended by AM-S12a (step 0 is new; steps 1–6 keep their approved order).

0. **Trigger-specific only: kill switch first.** If the trigger is a second BUY on this run, or an order on the pair that is not from this run (no `<RUN>-` client-id prefix, or not in `/orders`): latch the global kill switch first, because step 1 cannot contain a foreign entry source. It stops nothing and exits keep running, so it cannot interfere with the flatten SELL in step 1.
   ```bash
   curl -sS -X POST -H @<(hdrs api admin) \
     -H 'X-Emergency-Reason: smoke abort: unexpected order' "$API_BASE/emergency/kill-switch"
   ```
   Expect `"latched": true`. Then go to step 1 for the smoke run. Never sell an order that is not from this run under this runbook; record it and escalate. For every other trigger, start at step 1.
1. **Engine first — stop with flatten.**
   ```bash
   curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN?flatten=true"
   ```
   This latches this run's entries (no new BUY) and sells the run's own capped qty. It never sells more than min(own, balance) and never touches coins the run did not buy. It takes up to about 35 s.
   - 200, `"status":"stopped"`, `flatten.outcome` `flattened` or `noop`, `complete: true` → done; go to step 6.
   - 409 `{"code":"flatten_incomplete","flatten":{...}}` → the run is still `running`, entries latched. Read `flatten.symbols[].remainingQty` and go to step 2 or 3.
   - 409 `flatten_requires_running_engine` → the run is `orphaned`/`resuming` or has no engine in this process: go to step 4.
2. **409 with dust only** (SMK-T-20 / SMK-R-05, CF-SMK-S1). Dust means `remainingQty` ≤ 1 step, or `remainingQty` × price < `limits.cost.min`. The engine cannot sell it and flatten will keep returning 409. Stop without flatten:
   ```bash
   curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN?flatten=false"
   ```
   Expect `stopped`, with the dust listed in `unprotectedPositions`. Record the qty and write it off. In the UI dialog, the same action is **"Stop without flatten"**.
3. **409 with a real remaining qty** (more than dust):
   1. Retry step 1 once.
   2. Then emergency stop with flatten. This endpoint **always** stops the run, whatever the flatten outcome:
      ```bash
      curl -sS -X POST -H @<(hdrs api) \
        -H "X-Emergency-Reason: smoke abort: flatten incomplete" \
        "$API_BASE/runs/$RUN/emergency-stop?flatten=true"
      ```
      Read `flatten` and `unprotectedPositions` in the response.
   3. If anything is still held, **manual sell on Coinbase (last resort)**:
      - Advanced Trade → open orders for the pair: cancel any order whose client id starts with `<run_id>-` (or whose order id matches an `exchangeOrderId` in `/orders`).
      - Market-sell **exactly** the ledger `remainingQty` (from the flatten response or `unprotectedPositions`). **Never** "Max" when `pre_total > 0`.
   4. **Run shows `error` and `/positions` or Coinbase shows a qty above dust** (a crashed engine also ends in `error`). Stop and emergency-stop reject `error` runs (409), so the API cannot close it. Do step 3.3 by hand: cancel the run's open orders, then market-sell **exactly** the ledger qty. Record it and escalate.
4. **Run is `orphaned`** (for example after an API restart mid-test). Never use `resume?mode=normal` (it is blocked anyway). Choose one:
   - Protective resume: new BUYs are dropped, the fresh strategy instance's SELL window or the SL closes the position. It requires the admin key **and** the confirm token (and the open window):
     ```bash
     curl -sS -X POST -H @<(hdrs api admin live) \
       "$API_BASE/runs/$RUN/resume?mode=protective"
     ```
     Then watch for the SELL (the fresh instance's window starts `hold_bars` bars after its first bar) and continue with step 1 once it is `running`, if needed.
   - Or stop it without flatten and sell by hand (step 3.3):
     ```bash
     curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN?flatten=false"
     ```
5. **Engine unresponsive, or unexpected orders keep appearing:** latch the global kill switch (skip if step 0 already did). This blocks new BUYs on every engine immediately. Exits keep running, and it stops nothing:
   ```bash
   curl -sS -X POST -H @<(hdrs api admin) \
     -H "X-Emergency-Reason: smoke abort" "$API_BASE/emergency/kill-switch"
   ```
   Then continue with steps 1–3. As an absolute last resort: disable or delete the API key in the Coinbase Developer Platform (removes trade permission), then `$DC stop api`. After that:
   - sell **exactly** the ledger qty by hand, as in step 3.3;
   - close the window **before** the api comes back: run the §4e block now. This is the one exception to the §4e precondition. The run is already `orphaned` and stays `orphaned` across the recreate (live runs are never auto-resumed), and the CDP key is disabled. The §4e recreate brings the api back with `ENABLE_LIVE_TRADING=false` and a rotated token;
   - then close the record: `curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN?flatten=false"` (expect `stopped`);
   - never re-enable the CDP key while the run is `orphaned`;
   - running §4e again in step 6 is harmless (it rotates again).
6. **Afterwards (always):**
   - Snapshot Coinbase balances and order history.
   - Export `/orders`, `/fills`, `/trades`, `/portfolio` and the api logs for the run (no secrets).
   - File an incident note.
   - **No new live run until the cause is found.**
   - **Close the live window: §4e.** It runs on every abort.
   - A latched global kill switch blocks every new paper/live create (409 `kill_switch_active`). Clear it only after the root cause is known. The reason is audited free text: no secrets and no personal data.
     ```bash
     curl -sS -X POST -H @<(hdrs api admin) \
       -H "Content-Type: application/json" -d '{"reason":"<3-500 chars: why it is safe>"}' \
       "$API_BASE/emergency/kill-switch/clear"
     ```

---

## 7. Retry rule for a 5xx or network error on create (O-3, O-3a, SMK-SEC-204)

A 5xx (or a dropped connection) on `POST /runs` does not tell you whether a run was created, and a same-key retry can also replay a run that never had an engine. Do **not** blindly retry, and never switch to a new key to get around a stuck **in-progress** request.

1. **Record `T_REQ`** (`T_REQ=$(date -u +%FT%TZ)`) immediately before every live create: Run A, Run B and an INCONCLUSIVE repeat (§4c step 2, §4d step 1).
2. **Look-up, before EVERY retry (not only the first).** It lists the five most recent live runs of any status, including `error`:
   ```bash
   curl -sS -H @<(hdrs api) "$API_BASE/runs?mode=live&include_archived=true&limit=500" \
     | jq '[.items[] | {id, status, createdAt}] | sort_by(.createdAt) | reverse | .[0:5]'
   ```
   Identify by eye the runs created at or after `T_REQ` (the API timestamp format may differ from `date`, so do not rely on string comparison).
3. Branches:
   - **(a)** A run since `T_REQ` in `running`/`orphaned`/`resuming`: do not retry. Continue with that run (set `RUN_A`/`RUN_B` to its id).
   - **(b)** A run since `T_REQ` in `error`, or a same-key retry that returns `201` + `Idempotent-Replay: true` with `"status":"error"`: this is the ambiguous-commit case (no engine ever ran). Proceed only if **all** of the following hold:
     - `/runs/$RUN/orders` is empty;
     - Coinbase shows no order on the pair since `T_REQ`;
     - the api log shows `idempotency.ambiguous_commit_marked_error` for **that** run id: `RUN=<id>; $DC logs --since "$T_REQ" api 2>&1 | grep -F 'idempotency.ambiguous_commit_marked_error' | grep -F "$RUN"`. The `..._mark_error_failed` variant does not qualify; it leaves the run `running`, which is branch (a).
     - §2e bullets 1–2 are re-checked.

     Then create again with **one** new key (`KEY_A2=$(uuidgen)` / `KEY_B2=$(uuidgen)`), after recording a new `T_REQ`. This remains the same Run A or Run B, not an additional live run. Record both keys. This key change is permitted **once per run**. A second ambiguous commit on the same run means §6 step 6 plus escalation (then §4e).

     **Otherwise** (any condition fails, or you are unsure): do **not** create again. Treat it as an abort: §6 step 3.4 if the run holds a qty, then §6 step 6, escalate, then §4e.
   - **(c)** Nothing since `T_REQ`: retry the **identical** request with the **same** key (`$KEY_A` / `$KEY_B`).
     - 201 with `Idempotent-Replay: true` → this is the original run; continue with it.
     - 201 with `Idempotent-Replay: false` → the first attempt created nothing; this is the run.
     - 409 `idempotency_in_progress` → the first claim has not expired yet. Wait (the default stale window is 240 s) and return to step 2.
4. A replayed `running` run that places no BUY by about T0 + 30 min is already an abort trigger (§4b). That also covers the `..._mark_error_failed` variant, where the run is stuck `running` without an engine.
5. **Never** use a new key for anything except: a corrected body after a 4xx (§4c table), and branch (b) once per run.

---

## 8. Known limitations and carry-forwards that affect you

| ID | Limitation | What you do |
|---|---|---|
| CF-SMK-S11 | Exclusivity is one-directional: a smoke create is refused while another live run exists, but other live creates, promotes and resumes are **not** refused while a smoke run exists. | O-4: no other live action from the smoke create until that run is `stopped`. |
| CF-SMK-S1 | Flatten treats a sub-step residual as not flat and returns a false 409 `flatten_incomplete` (pinned by SMK-T-20). | Dust handling in §6 step 2; P12 non-empty list = PASS-DEGRADED. |
| CF-SMK-S2 | `initialCapital` above free EUR only logs `live.initial_capital_exceeds_free_quote`; it does not block the run. | Pre-flight: free EUR ≥ €65. That log line is an abort trigger. |
| CF-SMK-S3 | No reconciliation endpoint. | P9 is checked by hand from logs and balance arithmetic. |
| CF-SMK-S4 | No €10-capital smoke variant; the guard requires €60–66 live. | If €65 cannot be funded, escalate. |
| CF-SMK-S6 | The dashboard does not know the `diagnostic` status; the strategy is hidden from the create form. | Create runs with `curl` (§3, §4). |
| CF-SMK-S7 | Warm-up and 1m timing quirks (1m effectively skips bars). | Use 5m. |
| CF-SMK-S9 | The cost model assumes a 0.6% taker; the real tier may be 1.2%. | Use the recorded taker for the PnL band. |
| CF-SMK-S15 | This runbook. | Record its git blob hash in the evidence pack. |
| — | The stop-loss is evaluated at bar close only (not intrabar). | Expected; the SL is a net, not the exit. |
| — | The global kill switch blocks entries only; it does not stop runs or exits. | Use stop/emergency-stop to end a run. |
| — | A restart of the api container turns a live run into `orphaned` (never auto-restarted). | Do not restart during a run; use §6 step 4 if it happens. |

---

## 9. Evidence pack checklist

Store the pack in `~/smoke-evidence/<YYYY-MM-DD>/`, created with `install -d -m 700`, outside any git working tree. **No secrets** in any file or screenshot (check the headers pane of any browser dev tools before screenshotting). Never screenshot the CDP API-key pages. `X-Emergency-Reason` and kill-switch clear reasons are logged or audited free text: no secrets and no personal data. Never run `printenv` or `env` without a variable name, and never `docker inspect` the api or ui containers.

```bash
EVID=~/smoke-evidence/$(date +%F); install -d -m 700 "$EVID"
```

**Setup**
- [ ] Commit hash of the deployed tree (source machine: `git rev-parse HEAD`; must equal DEP §5.8) and the D-6 grep result (server).
- [ ] Git blob hashes of the documents used (source machine): `git hash-object docs/runbooks/smoke-roundtrip-mechanics-test.md reports/vp2-smoke/synthesis-spec.md reports/vp2-smoke/final-synthesis-smoke.md reports/vp2-docs/final-synthesis-docs.md`.
- [ ] D-0 (`alembic current` = `019 (head)`) and, if D-2 was unmet, the D-3 note.
- [ ] O-1 and O-2 confirmations (date, chosen pair).
- [ ] Output of the §2e pre-flight script: `limits.cost.min`, `limits.amount.min`, `precision.amount`, `taker`, `maker`, bid/ask/spread, EUR free/total, base `pre_total`.
- [ ] Kill-switch status response and the empty "no live run" query output.

**Paper rehearsal**
- [ ] `KEY_P`, `RUN_P`, `/orders` output, final `/positions`.

**Live window**
- [ ] Window opened and closed timestamps, both `printenv` outputs (`true / coinbase` and `false`), the `token rotated: yes` lines printed by §4c step 0 and §4e, copied from the terminal (never the value), "CDP key disabled: yes".

**Per live run (A and B)**
- [ ] `KEY_*` (and the second key if §7 branch (b) was used), `RUN_*`, `T_REQ`, T0, the full create response headers (`Idempotency-Key`, `Idempotent-Replay`).
- [ ] Log excerpt: `live.sync_positions_completed` through `smoke_roundtrip.exit_window_closed` or the stop.
- [ ] `/orders`, `/fills`, `/trades`, `/positions` (during the hold and at the end), `/portfolio` (final) as JSON.
- [ ] Coinbase screenshots: filled BUY and SELL with size, price and commission; balances before, during (optional) and after.
- [ ] Balance arithmetic for P9(b)/(c) and the P10/P11 calculation with t, s, δ written out.
- [ ] Stop responses: Run A 422 `[]` + 200 `stopped`; Run B curl 422 `["XRP/EUR"]`, dialog screenshots (choice required; flatten result).
- [ ] The completed P1–P13 sign-off table and the verdict.

**If aborted**
- [ ] Every abort call and its response, the manual Coinbase actions (with exact qty), and the incident note.

**Before you finish (both scans must print nothing), then clean up the shell.** The window token was rotated and retired in §4e and is no longer in the shell, so the value scan covers the API and admin keys only. The pattern scan still catches any `X-Live-Confirm-Token:` header in the pack (HTTP/2 dumps use lower-case header names, hence `-i`).
```bash
grep -rIli -E 'BEGIN (EC )?PRIVATE KEY|X-(API|Admin)-Key:|X-Live-Confirm-Token:' "$EVID"
grep -rIlF -f <(printf '%s\n' "${API_KEY:-}" "${ADMIN_KEY:-}" | grep -v '^$') "$EVID"
unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit
```

---

## 10. Change log

- **2026-09-28 (r3)** — Round-2 fixes per `reports/vp2-docs/final-synthesis-docs.md` Round 2 addendum (WR2-01..WR2-11).
- **2026-09-28 (r2)** — Revised per `reports/vp2-docs/final-synthesis-docs.md` (WD-01..WD-22): server-side shell with `--env-file`, `hdrs`/`env_set` helpers, live window (§2d phase 1, §4c step 0, §4e), fixed pre-flight script, P12 Run A only, kill-switch step 0 (AM-S12a), error-replay branch (O-3a), evidence handling.
- **2026-09-28** — Initial runbook (CF-SMK-S15), consolidating `reports/vp2-smoke/synthesis-spec.md` §10–§12 with the Round 2 addendum O-1..O-9 / D-1..D-6 of `reports/vp2-smoke/final-synthesis-smoke.md`. Verified against commit `3db9148`.
