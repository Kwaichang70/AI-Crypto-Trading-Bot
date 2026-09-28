# Runbook — Deploy of the Phase-1 minimum set (Verbeterplan v2, Fase 1)

**Audience:** Operator merging and deploying the Phase-1 minimum-set branch to the Hetzner/Tailscale production stack (`/opt/trading-bot`).

**Scope:** branch `claude-session-20260924-vp2-fase1-minimumset-xsoezy` (HEAD `3db9148`, 16 commits ahead of `origin/main`): WP1.0, WP1.1, WP1.2, WP1.8a/b, WP1.4/1.4b, WP1.11a, WP1.7a/b, WP1.3a, WP7.0 and WP-SMOKE. These ship as **one deploy** (see section 1).

**This runbook does not enable live trading.** `ENABLE_LIVE_TRADING` stays `false` throughout; the only exception is the rollback alternative RB-2 step 3 (section 6.2), on the new image only. The live restart (herstart-protocol) and the EUR mechanics test are separate procedures; section 8 defines the hand-off to `docs/runbooks/smoke-roundtrip-mechanics-test.md`.

**Related:** `infra/CADDY-RUNBOOK.md` (STEP 2a, DC-1..DC-7 and DC-70-4), `docs/runbooks/circuit-breaker-halt-auto-stop.md`, `Documentation/Verbeterplan-v2-2026-09.md` (D12, herstart-protocol), and the final-synthesis/acceptance reports under `reports/vp2-*` (condition IDs are traced in Appendix A).

> [!CAUTION]
> Never paste secret values into a terminal transcript, ticket or chat. This runbook only names variables. Never run `docker compose config` without `--quiet`: the plain form prints every interpolated secret. Never run `printenv`/`env` without a variable name, and never `docker inspect` the api/ui containers or run `$DC exec api env`; all of these print secrets.

---

## 0. Conventions

All commands run on the server, as the deploy user who owns `.env`, unless marked "source machine".

```bash
export DEPLOY_DIR=/opt/trading-bot
DC="docker compose -f $DEPLOY_DIR/infra/docker-compose.yml --env-file $DEPLOY_DIR/.env"
[ "$(stat -c '%a %U' "$DEPLOY_DIR/.env")" = "600 $(id -un)" ] || echo "STOP: .env mode/owner is not 600 $(id -un) -- escalate"
[ ! -L "$DEPLOY_DIR/.env" ] || echo "STOP: .env is a symlink -- escalate"
find "$DEPLOY_DIR" -maxdepth 1 -name '.env.*' ! -name '.env.example'   # expect no output; any file listed may hold secrets: remove it

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
```

- Any line that starts with `STOP:` ends the current section: do not run the next command. Follow the instruction on that line, or escalate.
- `$DC` is used everywhere below. Redefine `DC` and both helpers in every new shell.
- If `REQUIRE_API_AUTH=true`, calls to `/api/v1/*` need `X-API-Key` (the raw key, not `API_KEY_HASH`). Set it without echoing, and leave it empty when auth is off:
  ```bash
  read -rs -p "X-API-Key (empty if auth is off): " API_KEY; echo
  ```
  Use `-H @<(hdrs api)` in every `curl` call below.
- Do not `export` the secret variables. Do not run the session under `script` or any terminal recording.
- `<FQDN>` is the Tailscale FQDN in the substituted Caddyfile.
- Keep a deploy log (a file outside the repo) and note each step's result. Section 5.9 lists the evidence to keep.
- Final step of the session (after 5.9): `unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit`.

---

## 1. What changes (read first)

| Area | Change | Source of truth |
|---|---|---|
| Migrations | Branch adds **017** (`orphaned`/`resuming` run status, 4 audit event types), **018** (`kill_switch_state`, per-run `entries_latch_*`, 3 audit event types), **019** (`idempotency_keys`). `origin/main` head is **016**; branch head is **019**. 018 to 019 is the WP7.0 migration. | `infra/alembic/versions/017_*.py`, `018_*.py`, `019_*.py` |
| Compose | `ADMIN_API_KEY` (api) and `INTERNAL_ADMIN_API_KEY` (ui) become **required** (`:?`). On `origin/main` neither reached its container. | `infra/docker-compose.yml` |
| Caddy | New `handle /api/admin/*` to `ui:3000`, ordered before `handle /api/*`. `request_body { max_size 1MB }` on `/api/auth/*`, `/api/admin/*` and `/api/*`. The catch-all `handle` is not capped. | `infra/Caddyfile` |
| API settings | New optional settings with defaults, not wired through compose: `MAX_REQUEST_BODY_BYTES` (1 MiB), `IDEMPOTENCY_KEY_TTL_HOURS` (24), `IDEMPOTENCY_STALE_AFTER_SECONDS` (240). No `.env` change needed. | `apps/api/config.py` |
| Not changed | `infra/Dockerfile.api`, `infra/Dockerfile.ui`, `infra/docker-entrypoint.sh` are identical to `origin/main`. The entrypoint still runs `alembic upgrade head` before uvicorn. | `git diff origin/main..HEAD` |

**One deploy, not several.** The pieces depend on each other:

- WP1.1 is not deployable to live without WP1.2 (side-aware risk gates). WP1.7a is not deployable without WP1.7b (the old UI reads the old kill-switch shape and cannot send `flatten`). A new API with an old UI returns 428 on every create and promote (WP7.0). Deploy the whole branch, api + ui + Caddy together.
- Single API worker: `Dockerfile.api` runs `--workers 1`. Do not add a `command:` override or raise the worker count (the kill-switch latch and the run-engine registry are per process).

---

## 2. NO-GO conditions (stop and escalate)

Do not deploy if any of these is true:

1. Any live run is `running`, `orphaned` or `resuming` (section 3.1). Policy D12: deploy only when no live run holds positions.
2. No verified database backup (section 3.2).
3. `$DC config --quiet` exits non-zero (section 3.4).
4. `ADMIN_API_KEY` and `INTERNAL_ADMIN_API_KEY` fingerprints differ (section 3.4). This fails closed (all admin actions return 403), so it is an availability gate.
5. The exit-config precheck exits `2`, or exits `1` for a live run without a documented plan (section 3.6).
6. `caddy validate` fails (section 3.5). Without a valid Caddyfile the `/api/admin/*` route and the edge body cap are missing.
7. The `ENABLE_LIVE_TRADING` value that will be deployed is not `false`.
8. A long-running transaction is open on the database at migration time (section 4.3). Never raise the migration `lock_timeout` (5 s) to get past it.

---

## 3. Pre-deploy checklist

Do 3.1 to 3.3 with the **old stack still running**.

### 3.1 No live run active, and what a restart does to runs

- [ ] Confirm no live run is active:
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "SELECT id, status FROM runs WHERE run_mode = '"'"'live'"'"' AND status IN ('"'"'running'"'"','"'"'orphaned'"'"','"'"'resuming'"'"');"'
  ```
  Expected: `(0 rows)`. Any row: stop (NO-GO 1). Resolve it through the still-running old API first (stop, or flatten where the old API supports it), then re-check.
- [ ] Know what the restart will do (the api container is stopped and recreated during this deploy; `stop_grace_period` is 30 s):
  - **Live runs** found `running` at boot are moved to `orphaned`; a live run is never restarted automatically. Only an operator resumes it: `POST /api/v1/runs/{id}/resume?mode=normal|protective`, admin key plus `X-Live-Confirm-Token` plus the live-trading gates.
  - A live resume **cancels every open order the bot placed for that run on the run's symbols** (up to 30 s per order; the scan starts 5 minutes before the run's start time). Any exchange or DB failure returns 409 with no automatic retry; the operator re-sends.
  - **Paper runs** (`running`/`orphaned`/`resuming`) are rebuilt in place under the same run id from persisted fills and restarted at boot, within a bounded resume-count budget. Since WP1.3a, a paper run whose persisted exit config is invalid is marked `error` at boot instead.
  - Because of NO-GO 1 no live run should exist. If one exists anyway, do not deploy over it.
- [ ] Note the paper runs that will be restarted (informational): `SELECT id, status FROM runs WHERE run_mode='paper' AND status IN ('running','orphaned','resuming');` (same `exec` form as above).

### 3.2 Database backup

There is no existing backup procedure in the repo; this is the minimum.

- [ ] Take and verify a dump (the `sh -c` keeps credentials inside the container; the dump goes outside the deploy tree, mode 600):
  ```bash
  BACKUP_DIR=/var/backups/trading-bot                      # outside the deploy tree and any repo
  sudo install -d -m 700 -o "$(id -un)" "$BACKUP_DIR"
  DUMP="$BACKUP_DIR/pre-fase1-$(date +%Y%m%d-%H%M%S).dump"
  if ( umask 077
       $DC exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB"' > "$DUMP" ) \
     && [ -s "$DUMP" ] && [ "$(stat -c %a "$DUMP")" = 600 ] \
     && $DC exec -T postgres pg_restore --list < "$DUMP" > /dev/null; then
    stat -c '%a %s %n' "$DUMP"                             # record: mode 600, size, name
    echo "BACKUP OK"
  else
    echo "STOP: BACKUP FAILED -- NO-GO 2; do not continue"; rm -f -- "$DUMP"
  fi
  ```
  Continue only after `BACKUP OK`. A failed dump, a 0-byte file, a wrong mode or an unreadable table of contents is a failed backup (NO-GO 2). If you copy the dump off the host, encrypt it first (e.g. `age`/`gpg`), never place it in a git working tree, and delete it at the end of retention.
- [ ] Record the current schema revision (expected `016` if the server tracks `origin/main`):
  ```bash
  $DC exec -T api sh -c 'cd /app/infra/alembic && python -m alembic -c alembic.ini current'
  ```
- Note: the downgrade path for 017 to 019 is data-lossy for latch and idempotency state and relabels rows (section 6). The dump is the only full undo.

### 3.3 Environment inventory (names only, never values)

Compare `.env` on the server against `.env.example` from the deployed tree. Required by compose with `:?` (deploy aborts if unset):

| Variable | Status |
|---|---|
| `POSTGRES_PASSWORD`, `NEXTAUTH_SECRET`, `NEXTAUTH_URL`, `ADMIN_EMAILS`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | Already required on `origin/main`. |
| **`ADMIN_API_KEY`** (api) | **New requirement.** Was in `.env.example` but never passed to the container. |
| **`INTERNAL_ADMIN_API_KEY`** (ui) | **New requirement.** Server-side only; must equal `ADMIN_API_KEY`. Never prefix with `NEXT_PUBLIC_`. |

- [ ] `NEXTAUTH_URL` is exactly the public origin, `https://<FQDN>`. The UI admin proxy's same-origin guard checks it first; a wrong value gives 403 to clients that do not send `Sec-Fetch-Site` (DC-10).
- [ ] `ENABLE_LIVE_TRADING` is `false` or unset (compose default `false`).
- [ ] `LIVE_TRADING_CONFIRM_TOKEN`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`: leave unchanged. They are not needed for this deploy.
- [ ] **`DEBUG=false`.** Verified behaviour: compose does **not** pass `DEBUG` to the api container and the api has no `env_file`, so the container runs with the code default (`debug: bool = False`, `apps/api/config.py`). A `DEBUG=` line in `.env` therefore has no effect on the container. Confirm the effective state in section 5.3. Do not add a `DEBUG` entry in this deploy (pinning `DEBUG: "false"` in compose is tracked as CF-DOC-01). `.env.example` documents `DEBUG=false`; keep `.env` consistent with it.
- [ ] Optional settings above (`MAX_REQUEST_BODY_BYTES` and the two idempotency settings) need no action; they are not passed through compose and the defaults apply.

### 3.4 Stage the new tree and validate config (no restart yet)

- [ ] **Source machine:** record the commit and check the tree is clean:
  ```bash
  git rev-parse HEAD && git status --porcelain     # status must print nothing
  ```
  Merging the branch is the user's decision. Deploy from the merged commit whose tests were run. There is no CI (workflow removed by commit `364bb99`), so no automated gate stands behind the merge.
- [ ] Sync the tree to `$DEPLOY_DIR` with the method in `infra/CADDY-RUNBOOK.md` STEP 3 (it excludes `.env` and `.git`, so **the server has no git history**; the commit must be recorded from the source machine).
- [ ] The repo Caddyfile carries the `{{TAILSCALE_FQDN}}` placeholder and the sync restores it. Re-substitute, then check:
  ```bash
  bash $DEPLOY_DIR/infra/caddy-config.sh
  grep -c 'TAILSCALE_FQDN' $DEPLOY_DIR/infra/Caddyfile     # expected: 0
  ```
- [ ] Compose renders with all required variables (DC-8 of WP1.7b; `--quiet` only):
  ```bash
  $DC config --quiet; echo "exit=$?"     # expected: exit=0 and no other output
  ```
- [ ] `ADMIN_API_KEY` and `INTERNAL_ADMIN_API_KEY` match (DC-9). This prints a short fingerprint only:
  ```bash
  key_fp() { grep -E "^$1=" "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"'" | sha256sum | cut -c1-8; }
  [ "$(key_fp ADMIN_API_KEY)" = "$(key_fp INTERNAL_ADMIN_API_KEY)" ] && echo "OK: keys match" || echo "MISMATCH: fix .env"
  ```
  (`infra/CADDY-RUNBOOK.md` S-09 has the full-hash variant; this one prints 8 hex characters, as carry-forward CF-N7 asks.)

### 3.5 Caddy config validation (DC-13a-2)

- [ ] Before touching the running Caddy:
  ```bash
  $DC run --rm --no-deps caddy caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
  ```
  Expected: exit 0 and "Valid configuration". See section 9: this exact command was never executed by the reviewers.
- [ ] Static expectations in the file to be deployed (all three must hold):
  ```bash
  grep -cE '^[[:space:]]*request_body \{' $DEPLOY_DIR/infra/Caddyfile      # expected: 3
  grep -nE '^[[:space:]]*handle /api/(auth/|admin/)?\*[[:space:]]*\{' $DEPLOY_DIR/infra/Caddyfile
  ```
  The anchored patterns ignore comment lines. The second command must print exactly three lines, in the order `/api/auth/*`, `/api/admin/*`, `/api/*` (Caddy takes the first match; `/api/admin/*` must precede `/api/*`). `request_body` must stay in block form; an inline `request_body max_size 1MB` is silently ignored.

### 3.6 Build images and run the exit-config precheck (DC-13a-1)

- [ ] Build both images without starting them, and keep the api image under a stable tag for rollback:
  ```bash
  $DC build api ui
  SHORT_COMMIT=REPLACE_ME    # edit: the short commit hash recorded on the source machine in 3.4 (7-12 hex characters)
  if [[ "$SHORT_COMMIT" =~ ^[0-9a-f]{7,12}$ ]]; then
    docker tag crypto-trading-bot/api:local "crypto-trading-bot/api:fase1-$SHORT_COMMIT" && echo "tagged fase1-$SHORT_COMMIT"
  else
    echo "STOP: set SHORT_COMMIT to the commit recorded in 3.4 -- the rollback tag was NOT created"
  fi
  ```
  Compose names the api image `crypto-trading-bot/api:local`. Rebuilding old code later overwrites that tag, and the rollback needs the new image (section 6). Continue only after `tagged fase1-...` was printed; note the tag in the deploy log.
- [ ] Run the read-only precheck against the production DB, with the old API still running. The API image does not ship `scripts/` and its default entrypoint runs `alembic upgrade head`, so `--entrypoint python` is mandatory:
  ```bash
  cd $DEPLOY_DIR
  install -d -m 700 ~/deploy-logs/$(date +%F)
  ( umask 077
    $DC run --rm --no-deps --entrypoint python -v "$PWD/scripts:/app/scripts:ro" api \
      /app/scripts/wp13a_exit_config_precheck.py | tee ~/deploy-logs/$(date +%F)/precheck-wp13a.txt; echo "exit=${PIPESTATUS[0]}" )
  ```
  Do not hand-build a `DATABASE_URL`; use the compose-provided `POSTGRES_*` variables (a wrong scheme echoes the password in the error). Attach the output to the deploy log after checking it contains no `input_value=` or DSN.
  - Exit `0`: proceed.
  - Exit `1`: every `INVALID` live run needs a documented plan (fix the config, or protective resume then `DELETE ...?flatten=true`). Every invalid paper run will be marked `error` at boot; record that you accept this.
  - Exit `2`: NO-GO.

---

## 4. Deploy

Order: migrations to **019** first, then api and ui together, then Caddy. The entrypoint would also run the migrations at api start; the explicit step below makes a failure visible and keeps the app from starting on a half-migrated schema.

### 4.1 Announce

- [ ] Tell operators the behaviour changes in section 7 before the restart (DC-7, DC-13a-5, DC-70-6).

### 4.1a Re-check immediately before stopping the api

- [ ] Re-run the section 3.1 live-run query. Expected: `(0 rows)`.
- [ ] `$DC exec -T api printenv ENABLE_LIVE_TRADING` (the old container's effective flag) must print `false` or nothing. Anything else: NO-GO 7.
- [ ] If any live run was ever stopped with `?flatten=false` or left `unprotectedPositions`, confirm on Coinbase that the position has been sold. Otherwise NO-GO 1.

### 4.2 Stop the api

- [ ] `$DC stop api` (graceful; up to 30 s). The old ui may show errors until 4.4.

### 4.3 Migrate to 019 with the new image

- [ ] No long transactions on the database (migration 019 briefly takes `SHARE ROW EXCLUSIVE` on `runs`; 017 takes `ACCESS EXCLUSIVE` on `runs` and `audit_events` to validate the new CHECKs):
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "SELECT pid, state, now()-xact_start AS age FROM pg_stat_activity
     WHERE datname = current_database() AND xact_start < now() - interval '"'"'5 seconds'"'"'
       AND pid <> pg_backend_pid();"'
  ```
  Expected: `(0 rows)`. Otherwise clear the blocker first.
- [ ] Migrate:
  ```bash
  $DC run --rm --no-deps --entrypoint sh api -c \
    'cd /app/infra/alembic && python -m alembic -c alembic.ini upgrade head && python -m alembic -c alembic.ini current'
  ```
  Expected: `017`, `018`, `019` applied in order, then `019 (head)`. Each migration sets `lock_timeout = '5s'`; if it fires the transaction rolls back and the command fails. Clear the blocker and repeat. Never raise the timeout.

### 4.4 Start api and ui together, then Caddy

- [ ] One command for both application images (never the api ahead of the ui: a new api with the old ui returns 428 on every create and promote):
  ```bash
  $DC up -d api ui
  ```
- [ ] Wait until both report `healthy` in `$DC ps` (api `start_period` is 60 s).
- [ ] Recreate Caddy so it reads the new Caddyfile:
  ```bash
  $DC up -d --force-recreate caddy
  ```
  `up -d` alone does not recreate Caddy (image and compose config unchanged), and a file replaced by the sync may leave a stale single-file bind mount inside the container. Recreating avoids both. `$DC exec caddy caddy reload --config /etc/caddy/Caddyfile` is the alternative only if 5.4 confirms the running container sees the new file. Watch `$DC logs -f caddy` for the certificate line.

---

## 5. Post-deploy verification

Do every step. Any failure means stop, and consider the rollback in section 6.

### 5.1 Health

- [ ] `$DC ps`: postgres, redis, api, ui, caddy all `healthy`/running.
- [ ] API liveness inside the container: `$DC exec -T api curl -fsS http://localhost:8000/health` returns JSON with `"status":"ok"`.
- [ ] Through Caddy, background tasks: `curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/health/background`. Expected: `orphan_repeater.running` is `true` and `active_runs.count` matches the paper runs that were restarted.
  - Note: `/health` (root) is not routed to the api by Caddy (it reaches the ui catch-all), and there is no `/api/v1/health` route. The stale probe in `infra/CADDY-RUNBOOK.md` STEP 5 and `infra/caddy-validate.sh` test 2 is fixed in this WP (both now use `/api/v1/health/background`).
- [ ] Schema: `alembic current` prints `019 (head)` and the table exists:
  ```bash
  $DC exec -T api sh -c 'cd /app/infra/alembic && python -m alembic -c alembic.ini current'
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tc "SELECT to_regclass('"'"'idempotency_keys'"'"');"'
  ```
  A wrong revision here comes before anything else: on 018 every create and promote fails at the idempotency claim; on 017 the kill-switch load fails and runs come up latched with reason `latch_state_unknown` (fail-closed but blocking).
- [ ] Logs show no boot failure: `$DC logs --tail 300 api | grep -c latch_state_unknown` prints `0`, and `$DC logs --tail 300 api | grep -i -E 'error|critical'` shows nothing unexplained (expected lines: `recovery.*` for restarted paper runs).

### 5.2 Single worker

- [ ] `$DC exec -T api ps aux | grep -c '[u]vicorn'` shows one uvicorn process tree, and `$DC top api` shows `--workers 1` on the uvicorn command line.

### 5.3 Effective configuration (non-secret values only)

- [ ] `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false`.
- [ ] `DEBUG` (D-4): `$DC exec -T api printenv DEBUG` prints nothing (variable not set; code default `False`), **or** prints `false`. Any other value: fix and redeploy. Corroboration: `$DC logs --tail 5 api` lines are JSON (`json_output = not settings.debug`).
- [ ] `$DC exec -T ui printenv NEXTAUTH_URL` equals `https://<FQDN>` exactly.
- [ ] `ADMIN_API_KEY` and `INTERNAL_ADMIN_API_KEY` match: repeat the check in 3.4.

### 5.4 Caddy: admin routing and 1 MB body cap

- [ ] The running container sees the new file:
  ```bash
  $DC exec -T caddy sh -c "grep -cE '^[[:space:]]*request_body \{' /etc/caddy/Caddyfile; grep -cE '^[[:space:]]*handle /api/admin/\*' /etc/caddy/Caddyfile"
  ```
  Expected `3` and `1`.
- [ ] Oversized body is rejected at the edge (DC-13a-4):
  ```bash
  head -c 2000000 /dev/zero > /tmp/oversized-body.bin
  curl -sS -o /dev/null -w '%{http_code}\n' -X POST -H 'Content-Type: application/json' \
    --data-binary @/tmp/oversized-body.bin https://<FQDN>/api/v1/optimize
  rm -f /tmp/oversized-body.bin
  ```
  Expected `413` (occasionally `502`, a known Caddy race; both mean rejected). `200`, `422` or a hang is a FAIL. Repeat against `/api/auth/x` and `/api/admin/kill-switch` (same rule). Caddy's cap is 1 MB (1,000,000 bytes); the API enforces its own 1 MiB limit independently.
- [ ] A normal create through the UI still works (covered by 5.7).
- [ ] `/api/admin/*` reaches the ui, not the api. An unauthenticated request must be answered by the UI layer (redirect to sign-in or a UI-format rejection), not by FastAPI:
  ```bash
  curl -sS -o /dev/null -w '%{http_code}\n' -X POST https://<FQDN>/api/admin/kill-switch
  ```
  The security review observed a `307` redirect for a request without session on a live standalone server. A FastAPI-style JSON `401`/`403` body about `X-Admin-Key` would mean the request went to `api:8000`, i.e. the route order is wrong.

### 5.5 Kill switch, end to end via Caddy

Kill-switch press blocks new entries on all running engines and stops nothing; exits keep running. Pressing it briefly blocks entries of the paper runs restarted at boot. Clear it in the same sitting.

- [ ] In a browser session as an **admin** user (`ADMIN_EMAILS`), open the dashboard: the kill-switch status reads unlatched.
- [ ] Press the kill switch from the UI (path: browser to Caddy to `/api/admin/kill-switch` to ui to api). Then confirm from the shell:
  ```bash
  curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch     # expected: "latched": true
  ```
- [ ] Clear it from the UI with a reason (3 to 500 characters). Then confirm again:
  ```bash
  curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch     # expected: "latched": false
  ```
  A clear can be overridden by a press that was still pending (DC-6): never assume a 200 means cleared. If `latched` is still `true`, re-issue the clear.
- [ ] Fallback if no browser session is available (tests Caddy to api only, not the ui proxy). Read the key without echoing it:
  ```bash
  read -rs -p "X-Admin-Key: " ADMIN_KEY; echo
  curl -sS -X POST -H @<(hdrs admin) -H 'X-Emergency-Reason: post-deploy check' \
    https://<FQDN>/api/v1/emergency/kill-switch                 # expect latched=true, latchPersisted=true
  curl -sS -X POST -H @<(hdrs admin) -H 'Content-Type: application/json' \
    -d '{"reason":"post-deploy check done"}' \
    https://<FQDN>/api/v1/emergency/kill-switch/clear
  ```
  The reason strings are logged or audited free text: no secrets and no personal data.
  Then repeat the GET check. The UI admin proxy path (5.4 route probe plus the browser run) is the acceptance evidence; the fallback alone is not.
- [ ] A viewer (non-admin) user gets 403 on the admin actions.

### 5.6 Idempotency key is required (WP7.0, DC-70-7)

- [ ] A valid create without the header is refused and creates no run:
  ```bash
  curl -sS -o /dev/null -w '%{http_code}\n' -X POST -H @<(hdrs api) -H 'Content-Type: application/json' \
    -d '{"strategyName":"grid_trading","strategyParams":{},"symbols":["BTC/USDT"],"timeframe":"1h","mode":"paper","initialCapital":"10000.00"}' \
    https://<FQDN>/api/v1/runs
  ```
  Expected `428` with body `detail.code == "idempotency_key_required"`. A `422` means the body was invalid, not a pass. Confirm no new run appeared in the run list.
- [ ] A create from the UI (backtest) returns `201` and the response header `Idempotent-Replay: false` (browser dev tools, or a valid request with a fresh `Idempotency-Key: <uuid>`).
- [ ] Within about 24 h, one `idempotency.pruned` log line appears (`$DC logs api | grep idempotency.pruned`). Check the next day.

### 5.7 `GET /strategies` hides `smoke_roundtrip`

```bash
curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/strategies | grep -o smoke_roundtrip | wc -l                                # expected: 0
curl -fsS -H @<(hdrs api) 'https://<FQDN>/api/v1/strategies?include_diagnostic=true' | grep -o smoke_roundtrip | wc -l     # expected: at least 1
```
- [ ] The run-creation form in the UI does not list the smoke strategy.

### 5.8 Advisory-lock constant present, deployed commit recorded (D-6)

- [ ] The constant is in the deployed api image:
  ```bash
  $DC exec -T api grep -n '_SMOKE_LIVE_CREATE_LOCK_KEY' /app/api/routers/runs.py
  ```
  Expected: the definition (`_SMOKE_LIVE_CREATE_LOCK_KEY: Final[int] = 6002536669374727473` at HEAD `3db9148`) plus at least one use. No output means the wrong tree was built: rollback.
- [ ] The image matches the recorded commit. Compare a file hash of the source tree with the one in the container:
  ```bash
  git rev-parse HEAD                                            # source machine, record it
  sha256sum apps/api/routers/runs.py                            # source machine
  $DC exec -T api sha256sum /app/api/routers/runs.py            # server: must be identical
  ```
- [ ] Record the commit hash (full 40 characters), the image id (`docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local`) and the tag from 3.6 in the deploy log.

### 5.9 Evidence to keep (input for the smoke hand-off)

Deploy log entries: commit hash; `alembic current` output (`019 (head)`); precheck output; backup file name, size and mode (never the dump itself); results of 5.1 to 5.8; effective `ENABLE_LIVE_TRADING` and `DEBUG` state; operator and timestamp.

Final step: `unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit`.

---

## 6. Rollback

### 6.1 Ordering rules (read before acting)

1. **Downgrade the database with the new image first, then deploy the older code.** An older image does not contain the newer migration scripts. `alembic upgrade head` in the old entrypoint against a database at a revision it does not know fails with `Can't locate revision identified by '019'` (reproduced in the WP7.0 acceptance, A3b). So an old container cannot even start on the newer schema.
2. **The new image must still exist.** Compose tags it `crypto-trading-bot/api:local`, and building old code overwrites that tag. Use the tag saved in 3.6: `docker tag "crypto-trading-bot/api:fase1-$SHORT_COMMIT" crypto-trading-bot/api:local` (set `SHORT_COMMIT` to the value used in 3.6; in a new shell list the saved tags with `docker image ls crypto-trading-bot/api`) before running the downgrade, and do not rebuild old code until after it.
3. **LIFO:** roll back WP7.0, then WP1.3a, then WP1.7a/b (and WP1.8). Reverting 1.7b alone recreates the DC-1 breach (1.7a with a UI that cannot send `flatten`).
4. **Never run the downgrade with the API running.** Stop the api first.
5. **Gate G-RB, before ANY downgrade and before deploying ANY older code (every target, including `018`).** All of the following must hold, each recorded in the deploy log:
   - (a) the section 3.1 live-run query returns `(0 rows)`;
   - (b) no bot-held position above dust remains on Coinbase for any pair a live run traded. Every flatten is confirmed, every `unprotectedPositions` entry was sold by hand with the **exact** ledger qty and recorded, and dust is written off;
   - (c) `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false` on the **running new image**.

   If any of these fails: NO-GO for the downgrade. Downgrade 018 drops the latch state and every `flatten_incomplete` marker; downgrade 017 relabels `orphaned`/`resuming` runs to `error`, which `stop_run`/`emergency_stop_run` reject, so a position held by such a run would become untrackable through the API. Positions are resolved on the new image only, as in RB-2 (section 6.2).
6. **Never raise `lock_timeout`.** If a downgrade fails on it, clear the blocker.
7. **`ENABLE_LIVE_TRADING=false` for the whole rollback, except RB-2 step 3 on the new image.** Never `true` on the old or a downgraded image (images below 017 have no protective resume at all). Reverting 1.3a re-enables pyramiding and the warn-and-disable exit behaviour.

### 6.2 Procedure

- [ ] **Record state** (new API still running): `GET /api/v1/emergency/kill-switch` (`curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch`); the runs holding a per-run latch (`SELECT id, entries_latch_reason FROM runs WHERE entries_latch_reason IS NOT NULL;`); every live run and any open positions.
- [ ] **Resolve positions (RB-2), through the running new API, in this order.** `<RUN>` is the run id; `API_BASE=https://<FQDN>/api/v1`.
  1. Live runs in `running`: `curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/<RUN>?flatten=true"`. On 409, follow `docs/runbooks/smoke-roundtrip-mechanics-test.md` section 6 steps 2-3 (dust: `?flatten=false` plus write-off; a real remainder: retry once, then emergency-stop with flatten, then an exact-qty manual sell).
  2. Live runs in `orphaned`/`resuming`, **preferred path, no flag change**: `curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/<RUN>?flatten=false"`, then a manual Coinbase sell of **exactly** the ledger qty (never "Max"), recorded.
  3. Live runs in `orphaned`, **alternative path, the only exception to the live-off rule** (use it when the operator wants the engine to sell; new image only). This step is self-contained: the section 0 helpers must be defined in this shell, and the preconditions of the smoke runbook's §4c step 0 do **not** apply here. Preconditions, each recorded in the deploy log:
     - no live run is `running` or `resuming`; the only live runs are `orphaned`. Recreating the api keeps them `orphaned`, because live runs are never auto-resumed (section 3.1);
     - the kill-switch state is recorded and left as it is. A protective resume and `DELETE ?flatten=true` are not blocked by a latched kill switch, so do not clear it for this step;
     - the Coinbase CDP key (trade permission, never transfer) is enabled in the portal for this step only. The engine cannot sell with it disabled.

     Load the admin key on its own (the `read` is the last line, so pasting this block cannot feed it another command):
     ```bash
     API_BASE="https://<FQDN>/api/v1"
     read -rs -p "X-Admin-Key: " ADMIN_KEY; echo
     ```
     Open the window (fresh token, flag `true`):
     ```bash
     PREV_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
     if openssl rand -hex 32 | env_set LIVE_TRADING_CONFIRM_TOKEN \
        && printf 'true\n' | env_set ENABLE_LIVE_TRADING; then
       $DC up -d --no-deps --force-recreate api
       LIVE_CONFIRM_TOKEN=$(grep -E '^LIVE_TRADING_CONFIRM_TOKEN=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2-)
       if [ -n "$LIVE_CONFIRM_TOKEN" ] && [ "$LIVE_CONFIRM_TOKEN" != "$PREV_TOKEN" ]; then
         echo "window token rotated: yes"
       else
         echo "STOP: token not loaded or not rotated -- run the close block below, then escalate"
       fi
     else
       echo "STOP: window NOT opened -- run the close block below, then escalate"
     fi
     unset PREV_TOKEN
     $DC ps api                                                   # wait for "healthy"
     $DC exec -T api printenv ENABLE_LIVE_TRADING EXCHANGE_ID     # evidence; MUST print: true / coinbase
     [ "$($DC exec -T api printenv ENABLE_LIVE_TRADING)" = true ] || echo "STOP: flag is not true -- run the close block below"
     ```
     Then, for each `orphaned` run in turn (`RUN=<id>`):
     ```bash
     curl -sS -X POST -H @<(hdrs api admin live) "$API_BASE/runs/$RUN/resume?mode=protective"   # 409: re-send once, else stop this path
     curl -sS -H @<(hdrs api) "$API_BASE/runs/$RUN" | jq -r .status                                # repeat until: running
     curl -sS -X DELETE -H @<(hdrs api) "$API_BASE/runs/$RUN?flatten=true"                        # expect 200 "stopped"
     ```
     On a 409 from the `DELETE`, follow `docs/runbooks/smoke-roundtrip-mechanics-test.md` section 6 steps 2-3. Every run touched here must end `stopped` (or be still `orphaned` if the resume never succeeded) before the window closes. **Never close the window while a live run is `running`.**

     Close the window, on **every** exit from this step (success, a failed resume, a 409, or any `STOP:`):
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
     $DC exec -T api printenv ENABLE_LIVE_TRADING                 # evidence; MUST print: false
     [ "$($DC exec -T api printenv ENABLE_LIVE_TRADING 2>/dev/null)" = false ] \
       || { echo "STOP: flag is not false -- stopping the api, escalate"; $DC stop api; }
     unset LIVE_CONFIRM_TOKEN ADMIN_KEY
     ```
     Then disable the CDP key in the portal. Record both toggles, both printed `token rotated: yes` lines and both `printenv` outputs **before** continuing to G-RB.
  4. **`stopped` or `error` live runs that still hold a residual above dust** (from `/positions` or Coinbase; `error` is also the final status of a crashed engine). The API cannot stop or flatten these: stop and emergency-stop return 409 for `error`. Cancel any open order whose client id starts with `<RUN>-`, market-sell **exactly** the ledger qty by hand (never "Max"), and record it. G-RB (b) fails until this is done.
- [ ] **Gate G-RB** (rule 5): record (a), (b) and (c) in the deploy log. Any failure: stop here.
- [ ] `$DC stop api` (and `$DC stop ui`).
- [ ] **Downgrade with the new image.** Choose the target:

  | Roll back | Target revision |
  |---|---|
  | WP7.0 only | `018` |
  | WP7.0 + WP1.7a/b (branch state before WP7.0 and 1.7) | `017` |
  | Everything in the branch (return to `origin/main` code) | `016` |

  ```bash
  TARGET=018     # 018, 017 or 016
  $DC run --rm --no-deps --entrypoint sh api -c \
    "cd /app/infra/alembic && python -m alembic -c alembic.ini downgrade $TARGET && python -m alembic -c alembic.ini current"
  ```
  Effects: `019 -> 018` drops `idempotency_keys` (dedup state only; a client retry within 24 h can then create a second run, so check the runs list). `018 -> 017` drops `kill_switch_state` and the per-run latch columns and relabels the three new audit event types to `emergency_stop` (original kept in `payload.original_event_type`). `017 -> 016` relabels `orphaned`/`resuming` runs to `error` and the four resume event types to `emergency_stop`.
- [ ] **Deploy the previous code.** Revert the api first or api and ui together (an old api works with the new ui; a new api with an old ui does not). On the source machine check out the previous commit, sync, re-run `caddy-config.sh`, then `$DC up -d --build api ui`.
- [ ] **Caddy:** validate as in 3.5 against the reverted Caddyfile, then `$DC up -d --force-recreate caddy`. If 1.7b is reverted, `/api/admin/*` and the edge body cap are gone with it.
- [ ] **Verify:** `alembic current` equals the target; the old api starts (its `upgrade head` is a no-op at its own head); `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false`; the G-RB results are recorded.
- [ ] Record what was rolled back, the target revision, the G-RB results and the reason in the deploy log.

---

## 7. Operator-visible behaviour changes (communicate before deploy)

Kill switch and stop:
- The kill switch no longer stops runs. It blocks **new entries** only; exits (stop-loss, take-profit, trailing) keep running.
- Creating a **paper** run while the switch is latched returns 409 `kill_switch_active` (previously only live).
- Clearing (global and per-run) needs the admin key, injected server-side by the ui; browsers never hold it. Clearing a per-run latch on a live run also needs `X-Live-Confirm-Token`. Protective resume also needs the token.
- Stopping a **live** run requires an explicit `flatten` decision (`true`/`false`); omitting it returns 422 `flatten_decision_required`. An emergency stop with `flatten=true` can take about 35 s (UI timeouts are 65 s).
- An incomplete flatten leaves the run `running` with a persisted `flatten_incomplete` latch and returns 409. Retry the stop, or stop with `flatten=false` to abandon tracking (the position stays on the exchange, unmanaged).

Run creation:
- **`Idempotency-Key` is required** on `POST /api/v1/runs` and `POST /api/v1/runs/{id}/promote-to-live`. A retry with the same key returns the same run. Missing key gives 428. The UI keeps the same key after a failed submit, and a page reload mints a new one.
- **Ambiguous commit:** a run in `error` with no engine, with the log line `idempotency.ambiguous_commit_marked_error` (or `..._mark_error_failed`, which leaves the run `running`), never placed an order. Verify on the exchange, reload the page or use a **new** key, and resubmit (see `infra/CADDY-RUNBOOK.md`, section "DC-70-4 -- WP7.0"). For a smoke live create follow `docs/runbooks/smoke-roundtrip-mechanics-test.md` section 7. Per-run stop/emergency-stop reject `error` runs; that is expected.
- **Exit configs are enforced** (WP1.3a): an invalid exit config returns 422 on create, promote, resume and optimize instead of starting silently without an exit. Momentum and sl_tp_reversion need a stop-loss in every mode. No pyramiding by default in any mode. Live `dca_rsi_hybrid`/`grid_trading` without `allowPyramiding=false` returns 422 `live_pyramiding_forbidden`; the UI always sends `false` for live (single-entry, unvalidated variant). Backtest, paper and ML-gate metrics shift because of no-pyramiding.
- New 413 limit: bodies over 1 MiB at the API, over 1 MB at Caddy on `/api/*`, `/api/auth/*`, `/api/admin/*`. The UI shows only a generic error.

Live engine (relevant when live is eventually enabled, not for this deploy):
- A SELL is capped at what the bot's own ledger shows it bought and still holds. After 300 s an unresolved SELL keeps its reserve, is flagged, and BUYs are blocked run-wide until the exchange reports a final status.
- Any in-flight BUY blocks all new BUYs in the run with no expiry; after 15 minutes `live.buy_inflight_block_stale` is logged at error level. A run with mixed or missing quote currencies cannot BUY.
- Boot recovery never auto-resumes a live run (section 3.1).

Log events worth an alert (all present in the code): `live.sell_reserve_stale`, `live.sell_blocked_ledger_doubt`, `live.sell_i8_mismatch`, `live.buy_inflight_block_stale`, `live.initial_capital_exceeds_free_quote`, `live.submit_lookup_cid_mismatch`, `recovery.orphan_holding_unprotected`, `recovery.resume_stuck` (a run in `resuming` for more than 600 s), `recovery.live_exit_config_invalid`, `idempotency.ambiguous_commit_marked_error`.

Residual risks the operator accepts by deploying (unchanged by this deploy): exits are evaluated once per bar (a daily run checks its stop-loss once a day); the circuit-breaker HALT auto-stop leaves positions unprotected (`docs/runbooks/circuit-breaker-halt-auto-stop.md`).

---

## 8. Hand-off to the smoke runbook

Next procedure: `docs/runbooks/smoke-roundtrip-mechanics-test.md`.

**This runbook is complete when** sections 5.1 to 5.8 pass and 5.9 is recorded. State handed over:

| Item | Expected at hand-off |
|---|---|
| Deployed commit and image | Recorded (5.8); `_SMOKE_LIVE_CREATE_LOCK_KEY` present (D-6) |
| Schema | `019 (head)` |
| `ENABLE_LIVE_TRADING` | `false` (D-5); the smoke runbook decides when the test window opens |
| `DEBUG` | Not set or `false` (D-4) |
| Live runs | None in `running`/`orphaned`/`resuming` (D-5) |
| Kill switch | Unlatched (5.5 ends cleared) |
| `smoke_roundtrip` | Hidden from `GET /strategies`; visible with `?include_diagnostic=true` |

**Owned by the smoke runbook, not repeated here:** operator conditions O-1..O-9, pre-flight, the SMK-T-34 paper rehearsal, Run A and Run B, the pass criteria (P1..P13) and the abort procedure.

**Points the smoke runbook must respect from this one:**
- The smoke runbook supplies the `EXCHANGE_*` credentials with live off (its section 2d, phase 1), opens the live window in its section 4c step 0 (`ENABLE_LIVE_TRADING=true` plus a fresh `LIVE_TRADING_CONFIRM_TOKEN`) and closes it in its section 4e. Each of these recreates the api container. That restart applies the orphan behaviour of section 3.1, so it may only happen with no live run present. Note the compose default `EXCHANGE_ID` is `binance`; the smoke plan uses Coinbase.
- Do not toggle live on a stack whose section 5 checks have not all passed, and do not `resume?mode=normal` a smoke run (blocked by design).
- D-1 (merge and tests) and D-2 (SMK-T-37 on a fresh real Postgres) are test-time conditions and are **not** re-proved by this runbook; D-3 applies only if D-2 was not met.

**Still open before any live restart (outside both runbooks):** herstart-protocol steps in `Documentation/Verbeterplan-v2-2026-09.md` (integration gate on the merged commit, EUR mechanics test, two weeks of shadow paper, Phase-2 revalidation), the operator decision on residual risks in section 7, and the WP1.4b item "operator confirms Coinbase accepts the 49-character client order id".

---

## 9. Known gaps in this runbook and the documents it relies on

Not verified (no Docker daemon, no exchange or server access in the authoring session):
- The commands in 3.2 (`pg_dump`/`pg_restore` inside the postgres container; assumes the image's default local trust for the unix socket), 3.5 (`caddy validate` via `compose run`: security review S-R6-06 records it was never run), 4.3, 4.4 and 6.2 have not been executed. The `alembic downgrade 018` path and the "old code cannot start on 019" failure were proven by WP7.0 acceptance A3b; downgrades to 017 and 016, and an old image against 018, are inferred from the migration code and the same alembic behaviour.
- `--force-recreate caddy` as the reload mechanism, and the stale bind-mount risk after an rsync rename, are derived from Docker behaviour, not from any report. No source specifies the reload command.
- The `307` on an unauthenticated `/api/admin/kill-switch` comes from security review probes of a standalone server, not from a Caddy-fronted stack.
- Whether the committed tree at `3db9148` equals the tree the smoke acceptance ran on (that report ran on `be2b027` plus uncommitted patches).

- The `env_set` and `hdrs` helpers, the fail-closed window blocks, the §3.2 `if` block and RB-2 steps 3-4 have not been executed (CF-DOC-06); the first real use is the verification.

Defects still open in existing documents (outside the scope of this file):
- The catch-all Caddy `handle` has no `request_body` cap (carry-forward CF-13a-13/14).
- The Caddy `1MB` cap is 1,000,000 bytes while the API cap is 1 MiB (1,048,576); intentional but easy to misread.

---

## 10. Change log

| Date | Change |
|---|---|
| 2026-09-28 | Initial proposal for branch HEAD `3db9148` (producer draft, pending critic review). |
| 2026-09-28 (r2) | Revised per `reports/vp2-docs/final-synthesis-docs.md` (WD-01..WD-22): anchored Caddyfile greps, secure backup, `env_set`/`hdrs` helpers, section 4.1a re-check, rollback gate G-RB and RB-2, DC-70-4 renaming, hand-off aligned with the smoke runbook. |
| 2026-09-28 (r3) | Round-2 fixes per `reports/vp2-docs/final-synthesis-docs.md` Round 2 addendum (WR2-01..WR2-11). |

---

## Appendix A. Condition traceability

| Condition | Where | Section here |
|---|---|---|
| DC-1/DC-2 (1.7a with 1.7b; live off), DC-2' | `reports/vp2-wp1.7/final-synthesis-1.7a.md`, `-1.7b.md`; `infra/CADDY-RUNBOOK.md` | 1, 3.3, 8 |
| DC-3 single worker | same; `infra/Dockerfile.api` | 1, 5.2 |
| DC-4 migration before app (018, amended to 019) | same; `final-synthesis-7.0.md` | 4.3, 5.1 |
| DC-5/DC-5' rollback | same | 6 (G-RB, RB-2) |
| DC-6 clear vs pending press | same; `infra/CADDY-RUNBOOK.md` | 5.5 |
| DC-7 operator behaviour changes | same | 7 |
| DC-8 (1.7b) `config --quiet` | `final-synthesis-1.7b.md` | 3.4 |
| DC-9 key match | same | 3.4, 5.3 |
| DC-10 `NEXTAUTH_URL` | same | 3.3, 5.3 |
| DC-11 api + ui + Caddy together | same | 1, 4.4 |
| DC-13a-1 precheck | `reports/vp2-wp1.3a/final-synthesis-1.3a.md` | 3.6 |
| DC-13a-2 `caddy validate` | same | 3.5 |
| DC-13a-3 redeploy together | same | 4.4 |
| DC-13a-4 413 edge smoke | same | 5.4 |
| DC-13a-5 communicate UV-1..12 | same | 4.1, 7 |
| DC-13a-6 resume plan after restart | same | 3.1, 3.6 |
| DC-13a-7 rollback (amended by DC-70-5) | same; `final-synthesis-7.0.md` | 6 |
| DC-13a-8 acceptance first | same | 3.4 |
| DC-70-1 migration 019 before app | `reports/vp2-wp7.0/final-synthesis-7.0.md` | 4.3, 5.1 |
| DC-70-2 SHARE ROW EXCLUSIVE lock, D12, no long tx | same | 2, 4.3 |
| DC-70-3 ui + api together | same | 4.4 |
| DC-70-4 error run after ambiguous commit | same; `infra/CADDY-RUNBOOK.md` "DC-70-4 -- WP7.0" | 7 |
| DC-70-5 rollback, downgrade 018 with new image | same | 6 |
| DC-70-6 communicate | same | 4.1, 7 |
| DC-70-7 post-deploy smoke (428, replay header, prune log) | same | 5.6 |
| D-1..D-6 (WP-SMOKE) | `reports/vp2-smoke/final-synthesis-smoke.md` R2-7; `acceptance-report-smoke.md` | 5.3, 5.8, 8 |
| O-1..O-9 | same | 8 (owned by the smoke runbook) |
| WP1.8a/b: migration 017 first; `ADMIN_API_KEY`; resume token; order cancel; orphan health | `reports/vp2-wp1.8/final-synthesis-1.8a.md`, `-1.8b.md` | 3.1, 3.3, 4.3 |
| WP1.1 not deployable before WP1.2 | git log `149097e`; `reports/vp2-wp1.1/executor-report.md` | 1 |
| WP1.4/1.4b/1.11a deployment notes, alert events | `reports/vp2-wp1.4*/final-synthesis.md`, `vp2-wp1.11a/final-synthesis.md` | 7, 8 |
| S-08 `config --quiet`; S-10 body limit | `infra/CADDY-RUNBOOK.md` STEP 2a | 3.4, 5.4 |
