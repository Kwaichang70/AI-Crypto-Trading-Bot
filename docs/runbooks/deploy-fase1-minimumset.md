# Runbook — Deploy of the Phase-1 minimum set (Verbeterplan v2, Fase 1)

**Audience:** Operator merging and deploying the Phase-1 minimum-set branch to the Hetzner/Tailscale production stack (`/opt/trading-bot`).

**Scope:** three releases of branch `claude-session-20260924-vp2-fase1-minimumset-xsoezy`.
- **Release 1 (sections 1 to 9):** WP1.0, WP1.1, WP1.2, WP1.8a/b, WP1.4/1.4b, WP1.11a, WP1.7a/b, WP1.3a, WP7.0 and WP-SMOKE (16 commits, last commit `3db9148`). Merged to `origin/main` by pull request #2 (merge commit `ea0d41b`). These ship as **one deploy** (see section 1).
- **Release 2 (section 10):** WP1.5a, WP1.5b, WP1.6, WP1.3b-a and WP1.3b-b (`6511af1`, `206eb12`, `bf2b27f`, `d7a9956`, `dc8cf83`) plus CF-36 log hygiene (cf36a `6047162`; cf36b with cf36b-fix, the commit that adds section 5.10 to this file), on top of `origin/main` `ea0d41b`. **No migration, no Caddy change, no UI change.** CF-36 is required for G-LIVE L-19. Release 2 is recommended, not minimum-set (`Documentation/Verbeterplan-v2-2026-09.md`, "Aanbevolen erbij: 1.3b, 1.5, 1.6").
- **Release 3 (section 11):** WP2.1a, WP2.1b, WP2.1c and WP2.1-H (`60d7eaf` to `a7ae1bf`, fourteen commits), on top of `origin/main` `d36984e` (pull request #3). **One migration (020) and a ui image change; no compose, Caddy or environment change.** The WP2.1c re-baseline (CF-26) is still pending, and nothing in this file reports its result.

**This runbook does not enable live trading.** `ENABLE_LIVE_TRADING` stays `false` throughout; the only exception is the rollback alternative RB-2 step 3 (section 6.2), on the new image only. The live restart (herstart-protocol) and the EUR mechanics test are separate procedures; section 8 defines the hand-off to `docs/runbooks/smoke-roundtrip-mechanics-test.md`. For Release 2, section 10.9 (G-LIVE) lists what must hold before any live restart.

**Related:** `infra/CADDY-RUNBOOK.md` (STEP 2a, DC-1..DC-7 and DC-70-4), `docs/runbooks/circuit-breaker-halt.md` (breaker; its section 7 covers the live intra-bar exit check), `Documentation/Verbeterplan-v2-2026-09.md` (D12, herstart-protocol), and the final-synthesis/acceptance reports under `reports/vp2-*` (condition IDs are traced in Appendix A and B). The `reports/` directory is git-ignored and is not on the server, so every check an operator needs is written out in this file.

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
- Final step of the session (after 5.9, and after 5.10 when the image carries CF-36): `unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit`.

---

## 1. What changes (read first)

| Area | Change | Source of truth |
|---|---|---|
| Migrations | Release 1 adds **017** (`orphaned`/`resuming` run status, 4 audit event types), **018** (`kill_switch_state`, per-run `entries_latch_*`, 3 audit event types), **019** (`idempotency_keys`). Before pull request #2 the `origin/main` head was **016**; the head was **019** after Release 1, and Release 2 adds no migration (section 10.2). Release 3 adds **020** (section 11.2); on the Release 3 tree read every `019 (head)` expectation below as `020 (head)`. 018 to 019 is the WP7.0 migration. | `infra/alembic/versions/017_*.py`, `018_*.py`, `019_*.py` |
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

Release 2 adds NO-GO 9 to 13 (section 10.4): the DS-1 precheck, the NaN equity-row check, the `LIVE_EXIT_CHECK_INTERVAL_S` value, the image content check and the schema revision. Release 3 adds NO-GO 14 to 16 (section 11.4): the starting state, the image content and the rollback images.

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
- [ ] Record the current schema revision (expected `016` if Release 1 has never been deployed, `019` if it has; section 10.3 decides the path):
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
- [ ] Release 2 only: `LIVE_EXIT_CHECK_INTERVAL_S` is optional and is passed through compose (section 10.2). Its value is checked in section 10.5, step P-4.

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
  Expected: the definition (`_SMOKE_LIVE_CREATE_LOCK_KEY: Final[int] = 6002536669374727473`, unchanged from `3db9148` through CF-36) plus at least one use. No output means the wrong tree was built: rollback.
- [ ] The image matches the recorded commit. Compare a file hash of the source tree with the one in the container:
  ```bash
  git rev-parse HEAD                                            # source machine, record it
  sha256sum apps/api/routers/runs.py                            # source machine
  $DC exec -T api sha256sum /app/api/routers/runs.py            # server: must be identical
  ```
- [ ] Record the commit hash (full 40 characters), the image id (`docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local`) and the tag from 3.6 in the deploy log.

### 5.9 Evidence to keep (input for the smoke hand-off)

Deploy log entries: commit hash; `alembic current` output (`019 (head)`); precheck output; backup file name, size and mode (never the dump itself); results of 5.1 to 5.8; the 5.10 test counts and the `CF36 log hygiene` line (if the image carries CF-36); effective `ENABLE_LIVE_TRADING` and `DEBUG` state; operator and timestamp.

### 5.10 Log hygiene (G-LIVE L-19, CF-36)

**This section applies only to an image built from a commit that contains cf36a and cf36b (Release 2 and later, section 10).** It is not part of a Release-1-only deploy; there L-19 stays open. The checks below do not enable live trading.

**L-19 = (a) + (b+) + (e).**
- (a) is the redaction processor and final pass on the structlog chain and the stdlib bridge, the uvicorn and `ccxt` logger handling, the secret registry wired to `Settings`, `hide_input_in_errors`, the two counters and the self-check.
- (b+) is the per-site fixes: `ccxt_retry.*`, `live.ticker_fetch_failed`, `live.invalid_ticker_price`, `network_error_retrying`, `engine.bracket_exit_error`, the eight `live.*` I/O sites, `recovery.scan_*`, `runs.backtest_*`, the FRED and Whale clients and the per-bar `engine.poll_fetch_failed`.
- (e) is `engine.warmup_fetch_failed`.
- CF-A (cf36b-fix): uvicorn writes a WebSocket handshake line to `uvicorn.error` for every Upgrade request, with or without a WebSocket route and also under `--no-access-log`; its query string is logged as `?[QUERY_REDACTED]`.

- [ ] **Operator rule:** the live image runs with `LOG_LEVEL` other than `DEBUG`, and with the image CMD unchanged (`--no-access-log`, no uvicorn `--log-level` and no `UVICORN_*` variable in the api environment (uvicorn reads `UVICORN_LOG_LEVEL` and the like as CLI options)). The `<event>.trace` events carry tracebacks (redacted, but not minimised), and the self-check fails a live config with `LOG_LEVEL=DEBUG`. At uvicorn `--log-level debug` the `websockets` library logs raw request lines and headers, which no filter covers.
- [ ] **Step 1, on the image commit, in CI or a clean clone on the source machine (never in `$DEPLOY_DIR`).** The tests run with an empty environment (`env -i`): a `.env` / `.env.local` in the working directory, or an exported secret, makes them false-red, and a failure diff can print a value. Record the commit and the counts:
  ```bash
  git rev-parse HEAD      # record; must equal the commit recorded in 5.8
  if [ -e .env ] || [ -e .env.local ]; then
    echo "STOP: .env or .env.local present -- run step 1 in CI or a clean clone, never in \$DEPLOY_DIR"
  else
    env -i PATH="$PATH" HOME="$HOME" .venv/bin/python -m pytest \
      tests/unit/test_cf36_*.py tests/integration/test_cf36_*.py -q -rs --no-cov -p no:cacheprovider
  fi
  ```
  Expected: exit 0 and `<N> passed` with no `failed` or `error` (record N; on the cf36b-fix tree without `wsproto` installed: `746 passed, 2 skipped`). A skip is allowed only with the reason `could not import 'wsproto'`. `--no-cov` is required: the repository-wide 80% coverage floor would turn a passing subset into exit 1.
- [ ] **Step 2, in the deployed container (paper config, baseline).** Prints one line and no value:
  ```bash
  $DC exec -T api python -m api.log_hygiene_check
  ```
  Expected: exit 0 and a single line beginning `CF36 log hygiene: PASS`. Record the `secrets_registered=<n>` value and any `warn=` field names. A `FAIL <check-id>` line, or any other output, is a NO-GO. This run sees the paper config only: the `log_level_debug` and `live_secret_unregistered` checks apply only while `ENABLE_LIVE_TRADING=true`. The L-19 gate run is the repeat on the live config in `smoke-roundtrip-mechanics-test.md` §4c step 0 (or the equivalent step of any procedure that switches live on); a `FAIL` there means STOP and close the window (smoke §4e). (The `exec -T` form is the convention of this runbook; it has not been run against the deploy host.)
- [ ] **Step 2b, cf36b in the image.** Step 2 proves cf36a only. This proves cf36b and cf36b-fix (read-only, prints no value):
  ```bash
  $DC exec -T api sh -c 'grep -q "market_data.raise.trace" /app/packages/data/services/ccxt_market_data.py \
    && python -c "from common.logging import bounded_scalar, _WsQueryFilter" && echo CF36B_MARKERS_OK'
  ```
  Expected: the single line `CF36B_MARKERS_OK`. Anything else is a NO-GO: the image lacks cf36b. Not run against the deploy host.
- [ ] **Watch (G-LIVE L-8).** Both series carry the `trading_` exposure prefix:
  - `trading_log_redaction_hits_total{rule="R-xx"}` above 0 means a site still emits credential-bearing text. Investigate the rule and the event. An R-08 hit on `http.request` with a client-supplied path is client input (pre-auth), not a site leak; check the event before escalating.
  - `trading_log_redaction_failures_total` above 0 means the redactor failed closed on an event. Investigate.
- [ ] **Access log.** `infra/Dockerfile.api` runs `--no-access-log`; keep it. If it is ever enabled, request lines show `?[QUERY_REDACTED]`, which is expected. WebSocket handshake lines show it in every case.

Pass requires steps 1, 2 and 2b here, and the step-2 repeat on the live config before the first live order. Step 1 verifies the site fixes on the source commit; steps 2 and 2b verify the image. If any of them fails, G-LIVE does not pass.

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

Residual risks the operator accepts by deploying (unchanged by this deploy): exits are evaluated once per bar (a daily run checks its stop-loss once a day); a circuit-breaker HALT blocks new entries only and exits continue; it auto-recovers (WP1.6, `docs/runbooks/circuit-breaker-halt.md`). Release 2 changes the first of these for live runs only: section 10.8.

---

## 8. Hand-off to the smoke runbook

Next procedure: `docs/runbooks/smoke-roundtrip-mechanics-test.md`.

**This runbook is complete when** sections 5.1 to 5.8 pass and 5.9 is recorded (and 5.10 passes when the image carries CF-36). State handed over:

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

**Release 2 hand-off:** after Release 2 the same table applies (schema `019 (head)`, live off, no live run, kill switch unlatched). In addition, the G-LIVE checklist of section 10.9 stays open until the operator has recorded every row.

**Release 3 hand-off:** after Release 3 the same table applies with schema `020 (head)` (live off, no live run, kill switch unlatched). The G-LIVE checklist stays open, and row L-21 (the fee-tier check, section 11.7 V3-7) gates the first live run on a WP2.1 image.

**Still open before any live restart (outside both runbooks):** herstart-protocol steps in `Documentation/Verbeterplan-v2-2026-09.md` (integration gate on the merged commit, EUR mechanics test, two weeks of shadow paper, Phase-2 revalidation), the operator decision on residual risks in section 7, and the WP1.4b item "operator confirms Coinbase accepts the 49-character client order id".

---

## 9. Known gaps in this runbook and the documents it relies on

Not verified (no Docker daemon, no exchange or server access in the authoring session):
- The commands in 3.2 (`pg_dump`/`pg_restore` inside the postgres container; assumes the image's default local trust for the unix socket), 3.5 (`caddy validate` via `compose run`: security review S-R6-06 records it was never run), 4.3, 4.4 and 6.2 have not been executed. The `alembic downgrade 018` path and the "old code cannot start on 019" failure were proven by WP7.0 acceptance A3b; downgrades to 017 and 016, and an old image against 018, are inferred from the migration code and the same alembic behaviour.
- `--force-recreate caddy` as the reload mechanism, and the stale bind-mount risk after an rsync rename, are derived from Docker behaviour, not from any report. No source specifies the reload command.
- The `307` on an unauthenticated `/api/admin/kill-switch` comes from security review probes of a standalone server, not from a Caddy-fronted stack.
- Whether the committed tree at `3db9148` equals the tree the smoke acceptance ran on (that report ran on `be2b027` plus uncommitted patches).

- The `env_set` and `hdrs` helpers, the fail-closed window blocks, the §3.2 `if` block and RB-2 steps 3-4 have not been executed (CF-DOC-06); the first real use is the verification.

Release 2 (section 10) has its own list of unverified items in section 10.11, and Release 3 (section 11) in section 11.11.

Defects still open in existing documents (outside the scope of this file):
- The catch-all Caddy `handle` has no `request_body` cap (carry-forward CF-13a-13/14).
- The Caddy `1MB` cap is 1,000,000 bytes while the API cap is 1 MiB (1,048,576); intentional but easy to misread.

---

## 10. Release 2 (WP1.5/1.6/1.3b): delta to sections 1 to 9

Commits: `6511af1` WP1.5a (daily-PnL and peak semantics), `206eb12` WP1.5b (seed the daily risk state on resume), `bf2b27f` WP1.6 (circuit breaker wiring), `d7a9956` WP1.3b-a (bracket levels anchored at entry), `dc8cf83` WP1.3b-b (live intra-bar stop-loss and trailing exit check), `6047162` CF-36a (central log redaction, self-check `api.log_hygiene_check`), then CF-36b with cf36b-fix (per-site log fixes, WebSocket query filter, section 5.10). This file cannot name the commit that adds it, so cf36b is proven by markers (P-7, Block L-1, 5.10 step 2b), not by hash.

### 10.1 Form of this change: delta section, not integration

This revision adds one section (this one), corrects six statements elsewhere that the merge of Release 1 made stale (the scope paragraph, the "Related" line, the migrations row in section 1, the schema expectation in 3.2, the lock-key note in 5.8, the residual-risk paragraph in section 7) and adds four pointers (sections 2, 3.3, 8 and 9). Reasons:

1. **Cross-references.** The smoke runbook cites this file by section (`§5`, `§5.9`, `section 6 steps 2-3`, `section 3.1`), and `circuit-breaker-halt.md` cites it too. Integrating Release 2 into sections 1 to 8 would renumber or rewrite steps those documents rely on. The change log moves from section 10 to 11; no document cites it.
2. **Most Release 1 steps do not apply.** Release 2 has no migration (4.3), no Caddy change (3.5, 4.4, 5.4), no UI change (4.4, 5.5 to 5.7) and no new required secret (3.3). Integrating would put an "if Release 1 / if Release 2" branch into nearly every step.
3. **Two starting states.** A server may still be on Release 1's predecessor (schema `016`) or already on Release 1 (schema `019`). Section 10.3 chooses the path; the delta form lets the first path reuse sections 3 to 5 unchanged.

### 10.2 What changes in Release 2

| Area | Change | Source of truth |
|---|---|---|
| Code (api image only) | WP1.5a: day-start basis, UTC day roll, live drawdown limit 20% (paper 30%), BUY-only gates. WP1.5b: a resumed run seeds its daily risk state from persisted equity snapshots. WP1.6: the circuit breaker is wired for paper and live (BUY-only effects, HALT auto-recovers, no auto-stop, reset endpoint returns 410). WP1.3b-a: bracket levels are frozen per lot at entry; ATR period capped at 49. WP1.3b-b: live intra-bar stop-loss / trailing check. | `git log origin/main..HEAD`; `packages/trading/CLAUDE.md` (WP1.5a, WP1.6, WP1.3b-a, WP1.3b-b) |
| Code: CF-36 log hygiene | cf36a: one redaction pass on every log path (structlog chain, stdlib bridge, uvicorn loggers; the `ccxt` logger is held at WARNING), a secret registry fed from `Settings`, the counters `trading_log_redaction_hits_total` / `trading_log_redaction_failures_total`, and the self-check `python -m api.log_hygiene_check`. cf36b: outbound-I/O log sites log `error_type`, `http_status`, `error_at` instead of exception text, tracebacks only in DEBUG `<event>.trace` events, market-data errors carry fixed text, and the assembled `DATABASE_URL` percent-encodes the password. cf36b-fix: the query string is stripped from uvicorn's WebSocket handshake lines. No new environment variable. | `packages/common/CLAUDE.md` (CF-36 rules); section 5.10 |
| Migrations | **None.** `git diff origin/main..HEAD -- infra/alembic` is empty; the newest revision is `019`. The WP1.5b resume code only adds read-only queries on `equity_snapshots`; `tests/migrations/test_wp15b_resume_hints.py` is a test. The entrypoint's `alembic upgrade head` is a no-op on a `019` database. | `infra/alembic/versions/`; `apps/api/services/run_persistence.py` (`load_resume_snapshot`) |
| Compose | One line in the `api` environment block: `LIVE_EXIT_CHECK_INTERVAL_S: ${LIVE_EXIT_CHECK_INTERVAL_S:-300}`. It is optional (`:-`, not `:?`), and an unset or empty `.env` value becomes `300`. Not passed to `ui`. | `infra/docker-compose.yml:197` |
| `.env.example` | `LIVE_EXIT_CHECK_INTERVAL_S=300` with a comment. | `.env.example:193` |
| API setting | `live_exit_check_interval_s`, default `300`: `0` (off) or an integer `60..3600`. Anything else (`59`, `3601`, `-1`, `abc`, `1.5`, `true`, empty string) fails Settings validation, so the api would not start with such a value in its environment. Effective only for a **live** run whose timeframe is longer than the interval and that has a stop-loss or trailing stop; always off for paper, backtest and `smoke_roundtrip`. | `apps/api/config.py` (`Settings.live_exit_check_interval_s`, `validate_live_exit_check_interval`); `packages/trading/exit_config.py` (`resolve_exit_check_interval`) |
| API surface | `GET /api/v1/runs/{id}/circuit-breaker` returns the typed breaker state (`level`, `evaluations`, `restart_watch`, `thresholds`, `inputs`, ...). `POST /api/v1/runs/{id}/circuit-breaker/reset` always returns `410`. The UI does not call either (no `circuit-breaker` string under `apps/ui`). | `apps/api/routers/circuit_breaker.py:60-79` |
| Grafana | `trading-cb-halt` annotation text only; the rule expression (`max(trading_circuit_breaker_state)`, threshold 2.5) is unchanged. **Current state (Release 3):** this row describes Release 2 as deployed. Since WP2.1c-delta (`b9bbfb8`) the expression is `max(trading_circuit_breaker_state{run_id!~"bt-.*"})`, and `trading-kill-switch` and `trading-high-drawdown` carry the same matcher; the threshold is still 2.5 (section 11.2, Grafana row; `circuit-breaker-halt.md` section 2b). | `infra/grafana/provisioning/alerting/trading-alerts.yml` |
| Metrics | New counters appear on `/metrics` as `trading_<name>` (unmapped names get the `trading_` prefix), for example `trading_intrabar_exit_checks_total`. A counter series exists only after its first increment. **No Grafana rule exists for any of them** (the file has four rules: `trading-cb-halt`, `trading-kill-switch`, `trading-high-drawdown`, `trading-api-down`). | `apps/api/prometheus.py` (`_collect_counters`); `packages/common/metrics.py:193,236` |
| Not changed | `infra/Dockerfile.api`, `infra/Dockerfile.ui`, `infra/docker-entrypoint.sh`, `infra/Caddyfile`, `apps/ui`, `scripts/`. `git diff --stat origin/main..HEAD -- infra apps/ui scripts` lists only `docker-compose.yml` and `trading-alerts.yml`. | `git diff` |
| Docs | `circuit-breaker-halt-auto-stop.md` is replaced by `circuit-breaker-halt.md` (HALT no longer stops the run). After a rollback to Release 1 code, the old runbook is `git show origin/main:docs/runbooks/circuit-breaker-halt-auto-stop.md`. Section 5.10 (CF-36) is new, and `smoke-roundtrip-mechanics-test.md` §4c step 0 now re-runs the CF-36 self-check on the live config. | `git diff --name-status` |

### 10.3 Which path (decide first)

- [ ] Read the schema revision and check that the running api image is Release 1:
  ```bash
  $DC exec -T api sh -c 'cd /app/infra/alembic && python -m alembic -c alembic.ini current'
  $DC exec -T api grep -c '_SMOKE_LIVE_CREATE_LOCK_KEY' /app/api/routers/runs.py
  $DC exec -T api grep -c '_quote_context' /app/packages/trading/strategy_engine.py   # Release 1: 0 (verified at ea0d41b); Release 2: 3
  ```
  - `019 (head)`, a lock-key count of at least `2` and a `_quote_context` count of `0`: **Case A, Release 2 only.** Follow 10.4 to 10.7, reusing sections 3.1, 3.2 and 3.4 where cited. Skip 3.5, 3.6, 4.3, 4.4 and 5.4 to 5.7 (nothing they test has changed).
  - A `_quote_context` count of `1` or more means "already Release 2": STOP and do not run P-3 or P-6 (P-3 would then tag a Release 2 image as `rel1-...`).
  - `016`: **Case B, Release 1 was never deployed.** Run sections 3 to 5 with the Release 2 tree (one deploy; the migrations 017 to 019 apply as written), then add the Release 2 steps: 10.4 gates before 4.2, the `LIVE_EXIT_CHECK_INTERVAL_S` check, 10.7 after 5.8, and 10.9 before any live restart. The rollback is section 6 (it covers the downgrade); 10.10 adds the code-layer note.
  - Anything else (for example `019` without the lock-key constant, or a revision below `016`): STOP and escalate.

### 10.4 NO-GO conditions added by Release 2

Do not deploy if any of NO-GO 1 to 8 holds, or if:

9. **DS-1 (hard gate, C-201).** The WP1.3a/DS-1 precheck (10.5, P-8) could not run, its output was not recorded, or it exited `2`. A precheck that cannot run fails the gate; it does not pass.
10. The precheck lists a **live** run in `running`, `orphaned` or `resuming` with `bracket_atr_period:out_of_range` (or any other `INVALID`). That also breaks NO-GO 1. Stop it with `DELETE /api/v1/runs/{id}?flatten=true` on the **current** image before deploying, or obtain the user's explicit written acceptance first, and report the finding to the user.
11. The NaN equity-row count (10.5, P-9) is not `0`. A NaN `peak_equity_hint` makes the resume of that run raise (`packages/trading/portfolio.py:699`), including at paper boot.
12. `LIVE_EXIT_CHECK_INTERVAL_S` in `.env` is set to anything other than `0` or an integer from `60` to `3600` (10.5, P-4).
13. The built image does not contain the Release 2 code (10.5, P-7), or the schema is not `019` (10.3, D-4).

### 10.5 Pre-deploy additions (old stack still running)

Do these in order. P-1 to P-3 and P-8 to P-10 are the Release 2 additions; P-5 to P-7 reuse sections 3.4 and 3.6 with the changes noted.

- [ ] **P-1 Path.** Section 10.3 done; Case A or B recorded in the deploy log.
- [ ] **P-2 Live runs and backup.** Run the section 3.1 live-run query (expected `(0 rows)`) and the section 3.2 backup. Change one line of the 3.2 block so the dump is named for this release: `DUMP="$BACKUP_DIR/pre-release2-$(date +%Y%m%d-%H%M%S).dump"`. Continue only after `BACKUP OK`.
- [ ] **P-3 Keep the running image for rollback, before building.** Section 3.6 builds first and tags only the new image. Release 2 needs the old image kept, because `crypto-trading-bot/api:local` is overwritten by the build:
  ```bash
  $DC images api                                                    # note the IMAGE ID of the running api
  docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local   # must be the same image; record the full id
  ROLLBACK_TAG="crypto-trading-bot/api:rel1-$(date +%Y%m%d-%H%M%S)"
  docker tag crypto-trading-bot/api:local "$ROLLBACK_TAG" && echo "rollback image tagged: $ROLLBACK_TAG"
  ```
  Continue only after `rollback image tagged:`. If the two ids differ, STOP: someone rebuilt `:local` since the last deploy, and the running image is no longer on a tag.
- [ ] **P-4 `LIVE_EXIT_CHECK_INTERVAL_S`.** The variable is optional. Read its value (it is not a secret) and validate it:
  ```bash
  v=$(grep -E '^LIVE_EXIT_CHECK_INTERVAL_S=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"' ")
  if [ -z "$v" ] || [ "$v" = 0 ] || { [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -ge 60 ] && [ "$v" -le 3600 ]; }; then
    echo "OK: LIVE_EXIT_CHECK_INTERVAL_S='${v:-<unset: compose default 300>}'"
  else
    echo "STOP: LIVE_EXIT_CHECK_INTERVAL_S is not 0 or 60..3600 -- NO-GO 12; fix .env"
  fi
  ```
  Decide and record the value that will apply at the first live run: `300` (default) means the intra-bar check is on for every live run with a stop-loss or trailing stop; `0` means bar-close exits only. The setting has no effect on this deploy (live stays off). To pin the choice in `.env` (the value goes through stdin): `printf '%s\n' 300 | env_set LIVE_EXIT_CHECK_INTERVAL_S`. If G-LIVE row L-14 (10.9) fails, set `0` before the live restart and record the acceptance of bar-close-only exits.
- [ ] **P-5 Stage the tree** as in section 3.4: record the commit on the source machine, sync with the `infra/CADDY-RUNBOOK.md` STEP 3 method, re-run `bash $DEPLOY_DIR/infra/caddy-config.sh` and check `grep -c 'TAILSCALE_FQDN' $DEPLOY_DIR/infra/Caddyfile` prints `0` (the sync restores the placeholder, and Caddy reads this file the next time it is recreated), then `$DC config --quiet; echo "exit=$?"` (`exit=0`) and the key-fingerprint check. Skip 3.5: Caddy is not touched.
- [ ] **P-6 Build and tag.** Build only the api image (the UI source is unchanged) and tag it for traceability:
  ```bash
  $DC build api
  SHORT_COMMIT=REPLACE_ME    # edit: the short commit hash recorded in P-5 (7-12 hex characters)
  if [[ "$SHORT_COMMIT" =~ ^[0-9a-f]{7,12}$ ]]; then
    docker tag crypto-trading-bot/api:local "crypto-trading-bot/api:rel2-$SHORT_COMMIT" && echo "tagged rel2-$SHORT_COMMIT"
  else
    echo "STOP: set SHORT_COMMIT to the commit recorded in P-5 -- the Release 2 tag was NOT created"
  fi
  ```
- [ ] **P-7 Image content (NO-GO 13; G-LIVE L-1, L-13).** Prove the image holds the synced tree, then prove the synced tree is the commit. Nothing is started by these commands.
  ```bash
  cd "$DEPLOY_DIR"
  FILES="apps/api/config.py apps/api/services/run_orchestrator.py apps/api/services/run_persistence.py \
         packages/trading/exit_config.py packages/trading/portfolio.py packages/trading/safety.py \
         packages/trading/strategy_engine.py packages/data/services/ccxt_market_data.py \
         apps/api/log_hygiene_check.py packages/common/logging.py packages/common/redaction.py"
  diff <(sha256sum $FILES | awk '{print substr($1,1,16), $2}') \
       <($DC run --rm --no-deps --entrypoint sha256sum api $(for f in $FILES; do echo "/app/${f#apps/}"; done) \
          | awk '{p=$2; sub("^/app/","",p); if (p ~ /^api\//) p="apps/" p; print substr($1,1,16), p}') \
    && echo "IMAGE MATCHES SYNCED TREE"
  sha256sum $FILES | awk '{print substr($1,1,16), $2}'        # record
  ```
  On the **source machine**, in the merged tree, print the same list and compare it by eye with the recorded one; also confirm the six commits are present (skip the loop if the merge was a squash or rebase; the hash comparison then carries the proof):
  ```bash
  sha256sum apps/api/config.py apps/api/services/run_orchestrator.py apps/api/services/run_persistence.py \
            packages/trading/exit_config.py packages/trading/portfolio.py packages/trading/safety.py \
            packages/trading/strategy_engine.py packages/data/services/ccxt_market_data.py \
            apps/api/log_hygiene_check.py packages/common/logging.py packages/common/redaction.py | awk '{print substr($1,1,16), $2}'
  for c in 6511af1 206eb12 bf2b27f d7a9956 dc8cf83 6047162; do git merge-base --is-ancestor "$c" HEAD && echo "ok $c" || echo "MISSING $c"; done
  ```
  Expected: `IMAGE MATCHES SYNCED TREE`, identical lists on both machines, six `ok` lines. cf36b has no hash here (see the start of section 10); the three CF-36 files in the list and the Block L-1 markers prove it.
- [ ] **P-8 DS-1: WP1.3a precheck on the Release 2 image (NO-GO 9, 10; G-LIVE L-20).** The Release 2 image caps the ATR period at 49 (it was 99), so a persisted run with `bracket_atr_period` 50 to 99 that was valid under Release 1 is now `INVALID`. The command is the one in section 3.6 with `--json`, and the result is a gate:
  ```bash
  cd "$DEPLOY_DIR"
  install -d -m 700 ~/deploy-logs/$(date +%F)
  DS1=~/deploy-logs/$(date +%F)/precheck-ds1.json
  ( umask 077
    $DC run --rm --no-deps --entrypoint python -v "$PWD/scripts:/app/scripts:ro" api \
      /app/scripts/wp13a_exit_config_precheck.py --json | tee "$DS1"; echo "exit=${PIPESTATUS[0]}" )
  tail -n1 "$DS1" | grep -Eq '^wp13a-precheck: [0-9]+ run\(s\) checked, [0-9]+ invalid\. No writes were made\.$' \
    && echo "precheck ran" || echo "STOP: precheck did not complete -- NO-GO 9"
  sed '$d' "$DS1" | jq -r '.[] | select(.valid != true) | "\(.mode) \(.status) \(.run_id) \(.codes|join(",")) \(.note // "")"'
  sed '$d' "$DS1" | jq -r '.[] | select((.note // "") | contains("bracket_atr_period:out_of_range")) | "ATR-OUT-OF-RANGE \(.mode) \(.status) \(.run_id)"'
  ```
  The script prints a summary line after the JSON, so `sed '$d'` drops it before `jq`. Do not hand-build a `DATABASE_URL`. Check that the saved file contains no `input_value=` or DSN before attaching it to the deploy log (see section 3.6).
  - `exit=0` and no listed run: proceed.
  - `exit=2`, an empty file or no `precheck ran`: NO-GO 9.
  - A **live** run listed: NO-GO 10.
  - A **paper** run listed: record it only. Every invalid paper run is marked `error` at boot of the new api (as in 3.6). The `ATR-OUT-OF-RANGE` lines name the ones caused by the new cap.
  - `exit=1` for another reason: handle as in section 3.6.
- [ ] **P-9 NaN equity rows (NO-GO 11; G-LIVE L-3).** Read-only:
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "SELECT count(*) FROM equity_snapshots WHERE equity = '"'"'NaN'"'"'::numeric OR drawdown_pct = '"'"'NaN'"'"'::numeric;"'
  ```
  Expected: `0`. Otherwise locate the runs and do not resume them (NO-GO 11):
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "SELECT run_id, count(*) FROM equity_snapshots WHERE equity = '"'"'NaN'"'"'::numeric OR drawdown_pct = '"'"'NaN'"'"'::numeric GROUP BY run_id;"'
  ```
- [ ] **P-10 Deploy-day note (G-LIVE L-5).** Paper runs restarted by this deploy read their daily-risk hints from the last persisted equity rows. If older code resumed a paper run earlier in the same UTC day, its rows for today may be stale, which can leave a fail-closed BUY latch until 00:00 UTC (exits are not affected). The effect expires at the first 00:00 UTC after this deploy. Deploying shortly after 00:00 UTC avoids it. Record the deploy time in UTC.

### 10.6 Deploy (code-only, the api container alone)

- [ ] **D-1 Announce** the changes in section 10.8 (as in 4.1).
- [ ] **D-2 Re-check** immediately before stopping: the section 3.1 live-run query returns `(0 rows)`, and `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false` or nothing (section 4.1a). Then read the flag in `.env` (NO-GO 7). The container check proves the old api; this line proves the value the new api will start with:
  ```bash
  v=$(grep -E '^ENABLE_LIVE_TRADING=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"' " | tr '[:upper:]' '[:lower:]')
  if [ -n "${ENABLE_LIVE_TRADING+x}" ]; then
    echo "STOP: ENABLE_LIVE_TRADING is set in this shell and would override .env -- unset it (NO-GO 7)"
  elif [ -z "$v" ] || [ "$v" = false ]; then
    echo "OK: .env ENABLE_LIVE_TRADING='${v:-<unset: compose default false>}'"
  else
    echo "STOP: .env ENABLE_LIVE_TRADING is not false -- NO-GO 7 (close the window per smoke §4e, or fix .env)"
  fi
  unset v
  ```
  Record `curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch` (expected `"latched": false`).
- [ ] **D-3 Stop the api:** `$DC stop api` (graceful, up to 30 s).
- [ ] **D-4 Confirm the schema with the new image.** This must change nothing:
  ```bash
  $DC run --rm --no-deps --entrypoint sh api -c \
    'cd /app/infra/alembic && python -m alembic -c alembic.ini upgrade head && python -m alembic -c alembic.ini current'
  ```
  Expected: `019 (head)` and **no** `Running upgrade` line. A `Running upgrade` line means the database was below `019` (contradicts 10.3): stop and escalate (NO-GO 13).
- [ ] **D-5 Start the api only:** `$DC up -d api`. The ui image is unchanged and stays up; Caddy is not recreated. Wait until `$DC ps` shows the api `healthy` (`start_period` 60 s). If the api exits right away, read `$DC logs --tail 50 api`: a Settings validation error names the field `live_exit_check_interval_s` (NO-GO 12).
- [ ] **D-6 Grafana (optional, cosmetic).** The `trading-cb-halt` annotation text is read from the provisioning file when Grafana starts: `$DC restart grafana` shows the new text. The rule itself is unchanged, so skipping this loses nothing.

### 10.7 Post-deploy verification (Release 2)

Repeat sections 5.1 (health, schema `019 (head)`, boot logs), 5.2 (single worker), 5.3 (`ENABLE_LIVE_TRADING` prints `false`; `DEBUG`) and 5.8 (lock-key constant, deployed commit recorded). Then:

- [ ] **R2-1 Setting reached the container.**
  ```bash
  $DC exec -T api printenv LIVE_EXIT_CHECK_INTERVAL_S      # expected: 300, or the value pinned in P-4
  ```
- [ ] **R2-2 Running image equals the built image.** Repeat the first `diff` of P-7 with `$DC exec -T api sha256sum` in place of `$DC run ... sha256sum`; expected `IMAGE MATCHES SYNCED TREE`. Record `docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local`, the tag from P-6 and the rollback tag from P-3.
- [ ] **R2-3 Boot log.** Expected lines for each restarted paper run: `resume.risk_state_seeded` with `source` one of `snapshot`, `run_start`, `roll`. A `resume.daily_state_unknown` with `reason` `no_day_start_hint` or `seed_from_the_future` is the designed fail-closed answer (BUYs blocked until 00:00 UTC, exits unaffected). **Incident:** `resume.daily_seed_invalid`, `resume.daily_state_unknown` with `reason="seed_invalid"`, `engine.circuit_breaker_eval_failed`, `recovery.run_failed`. `runs.recovery_exit_config_invalid` means a paper run was marked `error` at boot because its persisted exit config no longer validates (P-8 should have listed it):
  ```bash
  $DC logs --since 10m api 2>&1 | grep -E 'resume\.(risk_state_seeded|daily_seed_invalid|daily_state_unknown)|engine\.circuit_breaker_eval_failed|recovery\.run_failed|runs\.recovery_exit_config_invalid|runs\.intrabar_exit_checks'
  ```
  `runs.intrabar_exit_checks` appears only when a live run starts; there is none in this deploy.
- [ ] **R2-4 Breaker wiring on a restarted paper run (rehearsal of G-LIVE L-11).** Skip if no paper run was restarted; L-11 then happens at the live restart.
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "SELECT id FROM runs WHERE run_mode = '"'"'paper'"'"' AND status = '"'"'running'"'"' ORDER BY started_at LIMIT 3;"'
  RUN=REPLACE_ME    # edit: one id from the query above
  curl -fsS -H @<(hdrs api) "https://<FQDN>/api/v1/runs/$RUN/circuit-breaker" \
    | jq '{level, entries_blocked, evaluations, restart_watch, halt_entry: .thresholds.halt_entry, halt_exit: .thresholds.halt_exit, reduce_entry: .thresholds.reduce_entry, reduce_exit: .thresholds.reduce_exit, daily: .thresholds.max_daily_loss_pct}'
  ```
  Expected `200`. `restart_watch` is present (`halt` for the first 3 evaluations after the start, `reduce` up to the 6th, then `none`; one evaluation happens per processed bar, so a 1h run needs an hour to reach `evaluations >= 1`). Paper bands: `0.3`, `0.27`, `0.15`, `0.12`, daily `0.08`. A `404` means the run is not in this process or has no breaker. A paper breaker at `halt` or `reduce` in the first bars is expected, not a failure.
- [ ] **R2-5 Reset endpoint is retired.** A nil UUID is harmless; the route changes nothing and writes no audit row:
  ```bash
  curl -sS -o /dev/null -w '%{http_code}\n' -X POST -H @<(hdrs api) \
    https://<FQDN>/api/v1/runs/00000000-0000-0000-0000-000000000000/circuit-breaker/reset      # expected: 410
  ```
- [ ] **R2-6 Metrics endpoint serves the safety gauge.**
  ```bash
  $DC exec -T api curl -fsS http://localhost:8000/metrics | grep -E '^trading_circuit_breaker_state'
  ```
  Expected: one sample per run that has processed a bar, value `0.0` while the level is OK (`1` REDUCE, `2` DAILY_LIMIT, `3` HALT). The gauge is published per processed bar, so a 1h run shows no sample until its first bar.
- [ ] **R2-8 Log hygiene (G-LIVE L-19, CF-36).** Run section 5.10 once: steps 1, 2 and 2b, and start the L-8 watch of the two `trading_log_redaction_*` counters. In Case B it already ran as part of section 5; do not repeat it. Step 2 here is the paper-config baseline; the L-19 gate run on the live config is smoke §4c step 0.
- [ ] **R2-7 Evidence to keep** (adds to 5.9): Case A or B; the P-3 rollback tag and image id; the P-6 tag; the P-7 hash list; the P-8 file `precheck-ds1.json` (this is the L-20 / DS-1 record); the P-9 count; the `LIVE_EXIT_CHECK_INTERVAL_S` value and the decision behind it; R2-1 to R2-6 results; the R2-8 results (5.10 step-1 commit and counts, the `CF36 log hygiene` line, `CF36B_MARKERS_OK`); UTC deploy time (L-5).

Final step as in section 5.9: `unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit`.

### 10.8 Operator-visible behaviour changes (communicate before deploy)

Risk gates and resume (WP1.5a/b):
- The live drawdown limit is **20%**; paper and backtest keep 30%. Both the drawdown and the daily-loss gates block **BUYs only**; exits are never blocked.
- Daily loss: BUYs are blocked when the day's loss reaches 8% of the day-start basis (inclusive); a basis of zero or less blocks. The risk day rolls at 00:00 UTC by the wall clock in paper and live.
- A resumed run (paper at boot, live after `resume`) seeds its day state from the persisted equity rows. If it cannot (`resume.daily_state_unknown`), BUYs stay blocked until 00:00 UTC and exits work.
- The breaker endpoint reports the gate-true NAV, peak and drawdown in `inputs`; the dashboard drawdown must equal it within rounding (the L-9 replacement check).

Circuit breaker (WP1.6; `docs/runbooks/circuit-breaker-halt.md`):
- The breaker now runs on paper and live runs, once per processed bar. REDUCE halves BUY size, HALT and DAILY_LIMIT drop BUYs. **Nothing stops the run and exits keep firing.** HALT clears by itself after 3 bars below 0.9 x the limit and steps down to REDUCE first.
- **The HALT auto-stop is gone**, and with it the residual risk that HALT left positions unprotected. A flat run at the limit does not recover (runbook section 3).
- **Restart watch:** every new breaker (including the paper runs restarted by this deploy) is stricter for its first evaluations: HALT already at 0.9 x the limit for 3 evaluations, REDUCE already at 0.4 x for 6. Paper runs may show reduced size or blocked BUYs right after this deploy at a drawdown of 12% or more (paper REDUCE) or 27% or more (paper HALT). This is expected.
- A **normal** live resume is refused with 409 `kill_switch_active` while the kill switch is latched, and with 409 `entries_latched` while the run carries a persisted entries latch; a protective resume is exempt (`apps/api/routers/runs.py`, resume endpoint). `circuit-breaker-halt.md` section 1 relies on this for the restart-watch procedure.
- `POST .../circuit-breaker/reset` returns 410; update any script that called it. A Telegram `CIRCUIT_BREAKER` message is sent on HALT entered and cleared when the notifier is configured; otherwise there is no push (G-LIVE L-12).

Exits (WP1.3b-a/b):
- Bracket stop and take-profit levels are frozen at entry for each lot; the ATR is read once. Adds keep the first entry's levels. Paper (and backtest) results change for ATR brackets, trailing stops and add-to-position runs; fixed brackets without trailing or adds are byte-identical.
- **The ATR period is capped at 49** in every mode. Creating, promoting or resuming a run with `bracket_atr_period` 50 to 99 returns 422, and the `momentum_breakout` and `sl_tp_reversion` parameter schema now rejects it. A persisted paper run with a larger period is marked `error` at boot (P-8 lists them).
- **Live intra-bar check (live only, default on):** a live run with a stop-loss or trailing stop reads the **bid** every `LIVE_EXIT_CHECK_INTERVAL_S` seconds (default 300), confirms a breach with a second reading about 10 s later, and then sells through the normal SELL path. Take-profit stays bar-close only. Live stops are therefore touch-based while the validated backtest is close-based. The warning `bar_close_exit_checks_only` (W8) now also covers 1h and is shown only when the check is **not** effective. `smoke_roundtrip` is always off, so the smoke runbook's "stop-loss at bar close only" statement stays true. Details: `docs/runbooks/circuit-breaker-halt.md` section 7.
- Operator rules recorded in G-LIVE L-17: a flatten that starts while an intra-bar exit holds the lock can return `in_flight` / `lock_timeout` and latch `flatten_incomplete` (re-read positions, retry the flatten, do not sell by hand while a bot SELL is open); a stop or shutdown can wait up to 15 s for the check, so never shorten `stop_grace_period: 30s` of the `api` service.

Log hygiene (CF-36; section 5.10):
- Exchange and market-data errors are logged as `error_type`, `http_status` and `error_at` (file, line, function) instead of the exception text. Their tracebacks exist only at `LOG_LEVEL=DEBUG`, as separate `<event>.trace` events; a live image never runs at DEBUG (the self-check fails that config). An unmapped exchange error now reads `Unexpected exchange error (<Class>)` instead of `Exchange error: <text>`.
- Values that look like credentials appear as `[REDACTED...]`; query strings on uvicorn access lines and on WebSocket handshake lines appear as `?[QUERY_REDACTED]`. Market-data errors in an HTTP `detail` carry fixed text plus the exception class.
- A database password containing `/ # ? @ : % +` now works in the assembled `DATABASE_URL` (it is percent-encoded). D-4 (`alembic upgrade head` with the new image) is the first use.

Log events worth an alert in addition to those in section 7 (all present in the code): `resume.daily_state_unknown`, `resume.daily_seed_invalid`, `portfolio.clock_regression`, `circuit_breaker.halt_entered`, `engine.circuit_breaker_eval_failed`, `circuit_breaker.listener_failed`, `circuit_breaker.telegram_notify_failed`, `telegram.send_error`, `telegram.send_failed`, `bracket_exit.lot_unprotected`, `bracket_exit.resume_fallback_final`, `bracket_exit.entry_unverified`, `bracket_exit.anchor_late`, `bracket_exit.anchor_lot_mismatch`, `engine.exit_restore_failed`, `exit_check.degraded`, `engine.intrabar_tick_error`, `engine.intrabar_loop_died`, `engine.exit_not_submitted` (error level). **None of these is wired to an automatic alert in `infra/`** (10.2, Metrics row): until CF-27 / CF-13b-17 / CF-13b-30, "paged" in the G-LIVE list means a person watching the command in L-8.

### 10.9 G-LIVE checklist (before any live restart)

Source: the consolidated list L-1 to L-20 in `reports/vp2-wp1.3b/final-synthesis-1.3b-b.md` section 12, which supersedes 1.5b F.3, WP1.6 R3-7 and 1.3b-a R2-5 for an image that contains WP1.3b. **All rows must hold, or be recorded where the table allows it, before the live restart.** Release 2 itself is deployed with live off; this checklist does not gate that deploy except for the rows in table A.

**L-21 (WP2.1, Release 3 images only)** is the one-time fee-tier check: the default cost profile (`DEFAULT_COST_PROFILE_ID` in `packages/trading/costs.py`) assumes a Coinbase tier, and L-21 compares it with the account's real tier before the first live run on a WP2.1 image. Block L-21 is in 10.9.4; the 2.1 spec is `reports/vp2-wp2.1/synthesis-spec.md` section 8.

Gate in words: `6511af1`, `206eb12`, `bf2b27f` and `d7a9956` are in the image, and so are `6047162` (cf36a) and cf36b with cf36b-fix (L-19); if intra-bar exits are wanted, `dc8cf83` is too (1.3b-b without 1.3b-a, or without its fix patch, is not allowed); the acceptance reports are green (1.5b F.5, WP1.6 R3-5, 1.3b-a ACCEPT 13/13, 1.3b-b ACCEPT 13/13); L-1 to L-20 below are satisfied. 1.3b is recommended, not minimum-set: an image without 1.3b-b runs bar-close-only exits, and W8 says so.

Commands containing a pipe are in the numbered blocks of 10.9.4 (`Block L-n`), not in the table cells. Everything marked read-only changes nothing. Define `API_BASE="https://<FQDN>/api/v1"` and, for a live run, `RUN=<id>`.

#### 10.9.1 Table A: deploy-time gates (before D-3, Release 2 deploy)

| ID | Check | Exact command or query | Expected result | Owning WP |
|---|---|---|---|---|
| L-1 | Code gate WP1.5/1.6: the image contains the 1.5a, 1.5b and WP1.6 commits | Block L-1 (P-7 plus marker grep; read-only) | `IMAGE MATCHES SYNCED TREE`; eleven count lines, no `MISSING:` line; `ok` for `6511af1`, `206eb12`, `bf2b27f` | WP1.5a, WP1.5b, WP1.6 |
| L-3 | NaN equity rows (until CF-21) | P-9 (read-only SQL) | `0` | WP1.5b (SEC-C-1) |
| L-13 | Code gate 1.3b: `d7a9956`; if intra-bar is wanted, `dc8cf83` with its fix patch | Block L-1 (same markers) | `ok d7a9956`, `ok dc8cf83`; `_quote_context` and `intrabar_loop_died_total` found (they exist only with the fix patch) | WP1.3b-a, WP1.3b-b |
| L-20 | DS-1 record for the 1.3b-a image (C-201): precheck JSON recorded, zero unresolved `bracket_atr_period:out_of_range` live open positions, any user acceptance recorded before deploy | P-8 (read-only) | `precheck-ds1.json` exists and is complete; no live run listed | WP1.3b-a |

#### 10.9.2 Table B: after Release 2, before the live restart (live still off)

| ID | Check | Exact command or query | Expected result | Owning WP |
|---|---|---|---|---|
| L-2 | NTP slew (CF-19): `makestep` limited to startup; synchronised; no clock regression | Block L-2 (host, read-only) | `Leap status : Normal`; a `makestep` line with a small update limit such as `makestep 1 3`; `NTPSynchronized` `yes`; `0` `portfolio.clock_regression` lines | WP1.5b |
| L-4 | Kill switch and mode intent: decide normal or protective resume; with WP1.6, plan the L-11 kill-switch step if entries must stay blocked through the restart watch | `curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch` (read-only) | Decision recorded; state recorded | WP1.5b, WP1.6 |
| L-5 | Transitional stale rows: was the run resumed by pre-1.5b code earlier in the same UTC day? Self-expiring | Block L-5 | `L-5 expired`, or the restart is after 00:00 UTC | WP1.5b |
| L-12 | Alert channel. Branch A: send the rendered HALT alert through the deployed notifier. Branch B: no push exists until CF-27; record it and watch per L-8 | Block L-12 | Branch A: `sent: True` and the message arrives in the chat. Branch B: decision recorded | WP1.6 |
| L-14 | Coinbase ticker sanity (1.3b-b): three read-only `fetch_ticker_quote` calls per symbol, 10 s apart; record spread (bps) and trade age; record whether `fetch_ohlcv(limit=N)` returns the forming candle | Block L-14 (read-only, no key printed) | `exchange=coinbase`; bid finite and positive, `ask >= bid`, latency under 2 s on every call; `L-14 result: PASS`; spread and `trade_age_s` recorded (on Coinbase this is the last-trade age, not a book age) | WP1.3b-b |
| L-17 | Operator rules recorded: (a) D12, no backtest or optimizer on the live API process while live holds positions; (b) live stops are bid-touch with a 10 s confirm, the validated backtest is close-based; (c) TP is bar-close only; (d) flatten during an intra-bar exit can return `lock_timeout`: re-read positions, retry, do not sell by hand while a bot SELL is open; (e) a stop or shutdown can wait up to 15 s: keep `stop_grace_period: 30s` | `grep -n 'stop_grace_period' "$DEPLOY_DIR/infra/docker-compose.yml"` (read-only) | Rules (a) to (e) recorded in the deploy log; the first `stop_grace_period` (api) is `30s` | WP1.3b-a, WP1.3b-b |
| L-18 | Trailing-gap seed: CF-13b-20 is in the image; record that a resume can take up to 40 s per held trailing symbol before the loop starts | Block L-1 (markers) plus a recorded note | 1.3b-b in the image, or acceptance of `seed_partial` for downtimes above 300 bars recorded | WP1.3b-b (CF-13b-20) |
| L-19 | Log hygiene: CF-36 (a)+(b+)+(e) and CF-A are in the image and hold ((b+) includes `ccxt_retry.retrying`, `ccxt_retry.exhausted`, `live.ticker_fetch_failed`, `live.invalid_ticker_price`; (e) is `engine.warmup_fetch_failed`). With intra-bar enabled the four (b) sites are a prerequisite because ticks reach them every interval | Section 5.10 steps 1, 2 and 2b (R2-8); then the step-2 repeat on the live config when the flag is switched on (smoke §4c step 0, or the equivalent step of the live-restart procedure), before the first live order | Step 1 exit 0, commit and count recorded; `CF36 log hygiene: PASS` on the paper config and again on the live config; `CF36B_MARKERS_OK`; L-8 counters at 0 or each hit explained | CF-36 (cf36a `6047162`; cf36b with cf36b-fix; raised by WP1.6, 1.3b-a, 1.3b-b) |
| L-21 | Coinbase fee tier (WP2.1, U-1): one read-only check that the account's real spot fee tier matches `DEFAULT_COST_PROFILE_ID` | Block L-21 (read-only, no order, no secret printed) | `FEE_TIER: DEFAULT CONFIRMED`; the four `market … taker=` values equal `fee_tier.taker_fee_rate`; `has_promo_fee: False`. Any other result: §8.4 | WP2.1 |

#### 10.9.3 Table C: at and after the live restart (observed, not gating the Release 2 deploy)

| ID | Check | Exact command or query | Expected result | Owning WP |
|---|---|---|---|---|
| L-6 | Post-restart seed, within the first minutes | Block L-6 (read-only) | `resume.risk_state_seeded` with `source` in `snapshot`, `run_start`, `roll`; no `resume.daily_seed_invalid`. `unknown` with `seed_invalid` is a bug (BUYs blocked until 00:00 UTC): incident, do not restart repeatedly | WP1.5b |
| L-7 | Resume failure handling | Block L-7 (read-only) | Run is `running`; `0` `resume_internal_error`. A 500 `resume_internal_error` reverts the run to `orphaned` and `orphan_holding_repeater` alerts: protect positions on the exchange, check the database | WP1.5b (WP1.8) |
| L-8 | 24-hour log watch after the restart (no automatic page exists, see 10.8) | Block L-8, at +15 min, +1 h, +6 h, +24 h | No lines, or each line explained. Incident: an `eval_failed` loop, `bracket_exit.lot_unprotected` (PAPER/LIVE, stop-loss brackets), `intrabar_loop_died` (plan a restart; the bar close protects). A Telegram line with a URL, `/bot` path or token: rotate the bot token and open a security incident, and do not paste the line anywhere | WP1.5b, WP1.6, WP1.3b-a, WP1.3b-b |
| L-9 | **Retired** (WP1.6, CF-24 closed). Replacement: the breaker endpoint drawdown equals the dashboard drawdown within rounding | Block L-11 (field `inputs.drawdown_pct`) against the dashboard | Equal within rounding; otherwise an incident (CF-24 regression); do not restart to clear it | WP1.6 |
| L-10 | **Not issued** (ID not reused) | n/a | n/a | n/a |
| L-11 | Breaker wiring, within minutes of the restart | Block L-11 (read-only) | `jq -e` prints `true` (exit 0): `evaluations >= 1`; live bands `0.2`, `0.18`, `0.1`, `0.08`; daily `0.08`; N = 3; `restart_watch` present. Read the level per the restart-watch note in `circuit-breaker-halt.md` section 1: stricter right after a restart is expected. 404, wrong thresholds or no `restart_watch`: stop the run (flatten per the operator) and escalate | WP1.6 |
| L-15 | Intra-bar enabled, post-restart | Block L-15 (read-only) | `runs.intrabar_exit_checks` with `interval_s` 300 for the live run (1h/4h/1d); while holding, `trading_intrabar_exit_checks_total` rises within 2 x the interval; `trading_intrabar_ticker_errors_total` and `trading_intrabar_loop_died_total` have no series; `exit_check.degraded` count `0`. A `reason=...` log: check timeframe, smoke or settings | WP1.3b-b |
| L-16 | Anchor restore, post-resume; judge after 3 bar passes | Block L-16 (read-only) | `bracket_exit.anchored` with source `resume_recomputed` (ATR) or exact (fixed), and `trailing_stop.peak_seeded`. A final `resume_fallback_current_atr`, `entry_unverified`, `entry_deferred` or `trailing_stop.seed_partial` is an incident: the position is protected; record the level difference | WP1.3b-a |

#### 10.9.4 Command blocks

**Block L-1 (L-13, L-18).** P-7 first (image vs synced tree, source-machine list and commit loop). Then the markers, on the built image or the running container (read-only):
```bash
$DC run --rm --no-deps --entrypoint sh api -c '
  chk() { n=$(grep -cF -- "$1" "$2"); echo "$n  $1  $2"; [ "$n" -ge 1 ] || echo "MISSING: $1"; }
  chk _day_start_pending                 /app/packages/trading/portfolio.py
  chk day_start_equity_hint              /app/api/services/run_persistence.py
  chk restart_watch                      /app/packages/trading/safety.py
  chk "POLICY_ATR_PERIOD_MAX: int = 49"  /app/packages/trading/exit_config.py
  chk _quote_context                     /app/packages/trading/strategy_engine.py
  chk intrabar_loop_died_total           /app/packages/trading/strategy_engine.py
  chk live_exit_check_interval_s         /app/api/config.py
  chk "CF36 log hygiene: PASS"           /app/api/log_hygiene_check.py
  chk "def bounded_scalar"               /app/packages/common/logging.py
  chk market_data.raise.trace            /app/packages/data/services/ccxt_market_data.py
  chk "class _WsQueryFilter"             /app/packages/common/logging.py
'
```
Expected: eleven lines with a count of at least 1 and no `MISSING:` line. The markers stand for WP1.5a, WP1.5b, WP1.6, WP1.3b-a, WP1.3b-b (twice: the second proves the fix patch), the setting, and CF-36: cf36a (the self-check), cf36b (`bounded_scalar`, `market_data.raise.trace`) and cf36b-fix (`_WsQueryFilter`). The CF-36 markers make an image without CF-36 fail here at deploy time; the L-19 gate itself is section 5.10.

**Block L-2 (host, read-only).**
```bash
chronyc tracking
grep -rhE '^[[:space:]]*makestep' /etc/chrony/ /etc/chrony.conf 2>/dev/null
timedatectl show -p NTPSynchronized --value
$DC logs api 2>&1 | grep -c 'portfolio.clock_regression'
```
If the log count is not `0`, investigate every event before the restart (a clock step is the cause).

**Block L-5.**
```bash
DEPLOY_DATE=REPLACE_ME      # edit: UTC date of the Release 2 deploy, YYYY-MM-DD, from the deploy log
if [[ "$DEPLOY_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && [[ "$(date -u +%F)" > "$DEPLOY_DATE" ]]; then
  echo "L-5 expired"
else
  echo "L-5 open: restart after 00:00 UTC, or accept a possible fail-closed BUY latch until then"
fi
```

**Block L-12 (Branch A; sends one real Telegram message).** First check that the notifier is configured, and never paste the token anywhere:
```bash
$DC logs api 2>&1 | grep -m1 -E 'telegram\.(configured|disabled)'
```
`telegram.disabled` means Branch B: record that no push alert exists for HALT until CF-27 and watch per L-8. Note that `infra/grafana/provisioning` contains no contact point, so Grafana's `trading-cb-halt` rule alerts only if a contact point was configured by hand in Grafana. With `telegram.configured`, render the HALT text for a synthetic transition with the live thresholds and send it through the notifier; the script runs inside the api container and reads the credentials from its own environment, so nothing secret is in argv or in the output:
```bash
$DC exec -T api python - <<'PY'
from api.config import get_settings as _gs, register_settings_secrets as _rs
from common.logging import configure_logging as _cl
_cl(log_level="INFO", json_output=False, service_name="glive-check")  # CF-36 layer 1 on; DEBUG .trace dropped
_rs(_gs())
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from api.config import get_settings
from api.services.run_orchestrator import _circuit_breaker_alert_text
from trading.risk import live_risk_parameters
from trading.safety import CircuitBreakerResponse, CircuitBreakerThresholds, CircuitBreakerTransition
from trading.telegram import TelegramNotifier

t = CircuitBreakerThresholds.from_risk_parameters(live_risk_parameters())
tr = CircuitBreakerTransition(
    run_id="00000000-0000-0000-0000-000000000000",
    from_level=CircuitBreakerResponse.OK, to_level=CircuitBreakerResponse.HALT,
    at=datetime.now(UTC), reason="deploy alert-channel test",
    drawdown_pct=Decimal("0.20"), thresholds=t)
level, message = _circuit_breaker_alert_text(tr)
print(level, message)
s = get_settings()
if s.telegram_bot_token is None or not s.telegram_chat_id:
    raise SystemExit("STOP: telegram not configured -- Branch B")
token = s.telegram_bot_token.get_secret_value()   # in-process only; never printed

async def go():
    n = TelegramNotifier(bot_token=token, chat_id=s.telegram_chat_id)
    try:
        print("sent:", await n.send_alert("circuit_breaker", level, "[DEPLOY TEST] " + message, run_id=tr.run_id))
    finally:
        await n.close()

asyncio.run(go())
PY
```
Expected: `CRITICAL HALT: entries blocked, exits active, run NOT stopped. Drawdown 20.00% reached the limit 20.00%. Auto-recovers after 3 bars below 18.00%.`, then `sent: True`, and the message in the chat. `sent: False` or a `telegram.send_error` / `telegram.send_failed` line: treat as Branch B until fixed. `STOP: telegram not configured` means Branch B. The test text carries percentages only.

**Block L-14 (read-only; public market data; no key is printed).** The first lines turn on the CF-36 log redaction and the INFO level: without them a failing exchange call prints its raw traceback, including any credential the exchange echoes. Edit `L14_SYMBOLS` to the pairs the live runs will trade. The script builds the service the way the live path does, so it uses the exchange credentials when they are set; the first output line says whether it authenticated:
```bash
$DC exec -T -e L14_SYMBOLS="BTC/EUR,ETH/EUR" api python - <<'PY'
from api.config import get_settings as _gs, register_settings_secrets as _rs
from common.logging import configure_logging as _cl
_cl(log_level="INFO", json_output=False, service_name="glive-check")  # CF-36 layer 1 on; DEBUG .trace dropped
_rs(_gs())
import asyncio, os, sys, time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from api.config import get_settings
from api.services.run_orchestrator import _normalize_exchange_secret
from common.types import TimeFrame
from data.services.ccxt_market_data import CCXTMarketDataService

SYMBOLS = [s for s in os.environ.get("L14_SYMBOLS", "").split(",") if s]
CALLS, GAP_S = 3, 10

async def main() -> int:
    st = get_settings()
    key = st.exchange_api_key.get_secret_value() if st.exchange_api_key is not None else None
    sec = st.exchange_api_secret.get_secret_value() if st.exchange_api_secret is not None else None
    sec = _normalize_exchange_secret(sec) if sec is not None else None
    print(f"exchange={st.exchange_id} authenticated={key is not None and sec is not None} symbols={SYMBOLS}")
    svc = CCXTMarketDataService(exchange_id=st.exchange_id, api_key=key, api_secret=sec, cache_ttl_seconds=60)
    bad = 0
    try:
        await svc.connect()
        for sym in SYMBOLS:
            for i in range(CALLS):
                t0 = time.monotonic()
                try:
                    q = await svc.fetch_ticker_quote(sym)
                except Exception as exc:  # type only: the text could carry a URL
                    print(f"{sym} call={i + 1} FAIL error_type={type(exc).__name__}")
                    bad += 1
                else:
                    ms = (time.monotonic() - t0) * 1000
                    ok = q.bid is not None and (q.ask is None or q.ask >= q.bid) and ms < 2000
                    bps = ((q.ask - q.bid) / q.bid * 10000).quantize(Decimal("0.01")) if q.bid and q.ask else None
                    age = (datetime.now(UTC) - q.exchange_ts).total_seconds() if q.exchange_ts else None
                    print(f"{sym} call={i + 1} {'PASS' if ok else 'FAIL'} bid={q.bid} ask={q.ask} spread_bps={bps} trade_age_s={age} latency_ms={ms:.0f}")
                    bad += 0 if ok else 1
                if i < CALLS - 1:
                    await asyncio.sleep(GAP_S)
            try:
                bars = await svc.fetch_ohlcv(sym, TimeFrame.ONE_HOUR, limit=5)
                forming = bool(bars) and bars[-1].timestamp + timedelta(hours=1) > datetime.now(UTC)
                print(f"{sym} fetch_ohlcv_returns_forming_candle={forming} last_bar_open={bars[-1].timestamp.isoformat() if bars else None}")
            except Exception as exc:
                print(f"{sym} fetch_ohlcv FAIL error_type={type(exc).__name__}")
                bad += 1
    except Exception as exc:
        print(f"FAIL connect error_type={type(exc).__name__}")
        bad += 1
    finally:
        await svc.close()
    print(f"L-14 result: {'PASS' if bad == 0 else 'FAIL'} ({bad} failing readings)")
    return 1 if bad else 0

sys.exit(asyncio.run(main()))
PY
```
Expected: `exchange=coinbase` (the compose default is `binance`; a different exchange here is a configuration error to fix first), every call `PASS`, `L-14 result: PASS`. Record the spreads and `trade_age_s`. `fetch_ohlcv_returns_forming_candle=True` means the exchange returns the still-forming candle: open the 1.3b-a CR-09 fix before the live restart. **If the bid is unusable** the intra-bar check would never fire: fix the cause, or set `LIVE_EXIT_CHECK_INTERVAL_S=0` (P-4) and record the operator's acceptance of bar-close-only exits. A trade age above 2 x the interval on a held symbol is input to CF-13b-18, not a blocker. If `authenticated=False`, the probe used the public ticker only; repeat it after the `EXCHANGE_*` credentials are supplied (smoke runbook section 2d), because the live handle is authenticated.

**Block L-19.** No separate block: section 5.10 is the procedure (steps 1, 2 and 2b, the L-8 counters), and smoke §4c step 0 runs the live-config repeat. The former informational grep (`error=str(exc)` / `ticker=ticker`) is retired: after cf36b it still lists `live.fill_parse_failed` and `live.reconciliation_transition_failed`, which stay on layer 1 by design and are not a failure.

**Block L-21 (WP2.1, G-LIVE L-21; read-only; no order, no secret printed).** Prerequisite: the Coinbase API key is present in the api container's environment (smoke O-1). Without credentials, ccxt reports its hardcoded "Advanced 1" rate (taker 0.012) for every EUR market, so `markets[...]['taker']` alone proves nothing. (ccxt's stable pairs use 0.00001/0.) Do **not** use `fetch_trading_fees()`: ccxt 4.5.40 swaps maker and taker in its result. The script reads the raw `transaction_summary` response instead. The block refuses every non-GET request at ccxt's single HTTP choke point (`Exchange.fetch`), so it cannot place, cancel or transfer anything even with the trade-enabled container key.

```bash
$DC exec -T api python - <<'PY'
from api.config import get_settings as _gs, register_settings_secrets as _rs
from common.logging import configure_logging as _cl
_cl(log_level="INFO", json_output=False, service_name="glive-l21")  # CF-36 layer 1 on
_rs(_gs())
import asyncio
from api.services.run_orchestrator import build_live_ccxt_exchange

SYMBOLS = ["XRP/EUR", "BTC/EUR", "ETH/EUR", "SOL/EUR"]   # edit: the live run's symbols


async def main() -> None:
    ex = None
    try:
        s = _gs()
        print("exchange_id:", s.exchange_id)
        ex = build_live_ccxt_exchange(s)
        _orig_fetch = ex.fetch

        async def _get_only(url, method="GET", headers=None, body=None):
            # read-only guard: every ccxt REST call passes through Exchange.fetch;
            # any non-GET (order, cancel, edit, transfer, withdraw) raises before any I/O
            if str(method).upper() != "GET":
                raise PermissionError("L-21 is read-only")
            return await _orig_fetch(url, method, headers, body)

        ex.fetch = _get_only
        await ex.load_markets()
        for sym in SYMBOLS:
            m = ex.market(sym)
            print(f"market {sym}: taker={m.get('taker')} maker={m.get('maker')}")
        raw = await ex.v3PrivateGetBrokerageTransactionSummary({"product_type": "SPOT"})
        tier = (raw or {}).get("fee_tier") or {}
        promo = (raw or {}).get("has_promo_fee")
        print("fee_tier.pricing_tier  :", tier.get("pricing_tier"))
        print("fee_tier.taker_fee_rate:", tier.get("taker_fee_rate"))
        print("fee_tier.maker_fee_rate:", tier.get("maker_fee_rate"))
        print("fee_tier.usd_from/to   :", tier.get("usd_from"), "/", tier.get("usd_to"))
        print("has_promo_fee:", promo)
        try:
            tier_taker = float(tier.get("taker_fee_rate"))
        except (TypeError, ValueError):
            tier_taker = None
        if tier_taker is None or any(ex.market(sym).get("taker") != tier_taker for sym in SYMBOLS):
            print("FEE_TIER: MARKETS DO NOT CARRY THE TIER -- stop")
            return
        if promo:
            print("FEE_TIER: PROMO -- do not switch")
            return
        try:
            from trading.costs import DEFAULT_COST_PROFILE_ID, match_fee_tier
        except ImportError:
            print("FEE_TIER: REGISTRY NOT IN IMAGE (pre-WP2.1) -- compare by hand with the WP2.1 registry table")
            return
        match = match_fee_tier(tier.get("taker_fee_rate"), tier.get("maker_fee_rate"))
        print("default profile :", DEFAULT_COST_PROFILE_ID)
        print("matching profile:", match)
        if match == DEFAULT_COST_PROFILE_ID:
            print("FEE_TIER: DEFAULT CONFIRMED")
        elif match is not None:
            print("FEE_TIER: DIFFERS -- reviewed one-line change of DEFAULT_COST_PROFILE_ID to", match)
        else:
            print("FEE_TIER: NOT IN REGISTRY -- reviewed change: add a registry id, then switch the default")
    except Exception as exc:
        print("L-21 check failed:", type(exc).__name__)   # type only (CF-36): never the message
    finally:
        if ex is not None:
            await ex.close()


asyncio.run(main())
PY
```

**Expected result.**
- `exchange_id: coinbase`.
- `FEE_TIER: DEFAULT CONFIRMED`. The block prints it only when every listed market's taker equals `fee_tier.taker_fee_rate` and `has_promo_fee` is falsy.
- Other results:
  - `MARKETS DO NOT CARRY THE TIER`: stop and escalate.
  - `PROMO`: §8.4 step 4.
  - `DIFFERS` / `NOT IN REGISTRY`: §8.4.
  - `L-21 check failed: PermissionError`: the script attempted a non-GET request. This is a defect; stop.
- Save the output to `~/deploy-logs/<date>/glive-l21.txt` (`umask 077`). It contains rates only, no keys.

**If the tier differs (exact rule).**
1. **Never edit `.env`, and never set an environment variable, to change the default.** `DEFAULT_COST_PROFILE` in the environment is ignored and logged as `cost_profile.env_ignored` (SY-21-02).
2. If `FEE_TIER: DIFFERS`, open a reviewed change. Its only production edit is the one line `DEFAULT_COST_PROFILE_ID: Final[str] = "<matching id>"` in `packages/trading/costs.py`. It passes the normal chain (producer → code-critic + security → executor → testing) and ships in the next release. The per-profile goldens (WP21-T-13) need no re-pin.
3. If `FEE_TIER: NOT IN REGISTRY`, the reviewed change **adds** a `live_equivalent` registry id with the measured rates (and its fingerprint pin in WP21-T-02), and flips the same one line. Existing ids are never edited.
4. If `has_promo_fee: True`, do **not** switch. Record the rates and re-run L-21 after the promotion ends.
5. Until a switch ships, WP2.8's go/no-go is judged at **both** the current default and the measured tier.
6. Already-persisted runs keep their own profiles (resume never re-resolves).
7. Corroboration: the first real live fill logs `cost_profile.fill_fee_drift` if the fee is more than 5 bps off the profile, and smoke P3 records the effective rate.

**Block L-6.**
```bash
$DC logs --since 15m api 2>&1 | grep -E 'resume\.(risk_state_seeded|daily_seed_invalid|daily_state_unknown)'
```

**Block L-7.**
```bash
curl -sS -H @<(hdrs api) "$API_BASE/runs/$RUN" | jq -r .status          # expected: running
$DC logs api 2>&1 | grep -c 'resume_internal_error'                      # expected: 0
```

**Block L-8** (run at +15 min, +1 h, +6 h, +24 h; do not paste matching lines into tickets):
```bash
$DC logs --since 24h api 2>&1 | grep -E 'resume\.daily_state_unknown|portfolio\.clock_regression|circuit_breaker\.(halt_entered|listener_failed|telegram_notify_failed)|engine\.circuit_breaker_eval_failed|telegram\.send_(error|failed)|bracket_exit\.(lot_unprotected|resume_fallback_final|entry_unverified|anchor_finalized|anchor_late|anchor_lot_mismatch)|engine\.exit_restore_failed|exit_check\.degraded|engine\.intrabar_(tick_error|loop_died)|engine\.exit_not_submitted|engine\.intrabar_lock_busy'
$DC exec -T api curl -fsS http://localhost:8000/metrics | grep -E '^trading_(intrabar_loop_died_total|lot_unprotected_bars_total|exit_not_submitted_total|log_redaction_hits_total|log_redaction_failures_total)'
```
Treat `engine.exit_not_submitted` as an incident at error level (streak 3, 10, 20, ..., and every `process_signal_raised`), `engine.intrabar_lock_busy` as an incident above 6 per hour, and any `trading_intrabar_loop_died_total` series as an incident. A `trading_log_redaction_*` series is read as in section 5.10 (Watch). For `telegram.send_failed`, spot-check that the `body` field holds no token.

**Block L-11** (live run; paper bands are in 10.7 R2-4):
```bash
curl -fsS -H @<(hdrs api) "$API_BASE/runs/$RUN/circuit-breaker" | jq -e '
  (.evaluations >= 1)
  and ((.thresholds.halt_entry|tonumber) == 0.2) and ((.thresholds.halt_exit|tonumber) == 0.18)
  and ((.thresholds.reduce_entry|tonumber) == 0.1) and ((.thresholds.reduce_exit|tonumber) == 0.08)
  and ((.thresholds.max_daily_loss_pct|tonumber) == 0.08)
  and ((.thresholds.deescalation_evaluations|tonumber) == 3)
  and has("restart_watch")'
curl -fsS -H @<(hdrs api) "$API_BASE/runs/$RUN/circuit-breaker" | jq '{level, restart_watch, entries_blocked, drawdown: .inputs.drawdown_pct, peak: .inputs.peak_equity}'
```
The thresholds are compared numerically (`"0.2"` passes). To keep entries blocked through the restart watch, use the kill switch after the normal resume, as described in `circuit-breaker-halt.md` section 1.

**Block L-15** (while the run holds a position):
```bash
$DC logs api 2>&1 | grep 'runs.intrabar_exit_checks'
$DC exec -T api curl -fsS http://localhost:8000/metrics | grep -E '^trading_(intrabar_exit_checks_total|intrabar_ticker_errors_total|intrabar_loop_died_total)'
$DC logs api 2>&1 | grep -c 'exit_check.degraded'                         # expected: 0
```

**Block L-16** (after 3 bar passes):
```bash
$DC logs api 2>&1 | grep -E 'bracket_exit\.anchored|trailing_stop\.peak_seeded|resume_fallback_current_atr|entry_unverified|entry_deferred|trailing_stop\.seed_partial'
```

**Retirements and non-issues:** L-9 retired, L-10 not issued, L-5 expires on its own at the first 00:00 UTC after the Release 2 deploy.

### 10.10 Rollback note for Release 2

Release 2 has no migration, so its rollback is **code only**: the database stays at `019`, nothing is downgraded, and the Release 1 image runs against it (it contains 017 to 019). The section 6 downgrade machinery, G-RB (b) and the `SHORT_COMMIT` tag of 3.6 are not needed for this step. Rolling back **past Release 1** is section 6.

Rules that still apply (they come from section 6.1): stop the api first (rule 4); `ENABLE_LIVE_TRADING=false` for the whole rollback (rule 7); and gate G-RB (a) and (c), because the rollback restarts the api and a restart orphans every live run (section 3.1): the section 3.1 live-run query must return `(0 rows)` and `$DC exec -T api printenv ENABLE_LIVE_TRADING` must print `false`.

- [ ] **Record state** (Release 2 api still running): the kill switch, the running paper runs, and `restart_watch` of any paper run that is in a drawdown.
- [ ] **Gate:** G-RB (a) and (c), recorded in the deploy log, and the `.env` live flag (NO-GO 7; the same block as 10.6 D-2), before `$DC stop api`:
  ```bash
  v=$(grep -E '^ENABLE_LIVE_TRADING=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"' " | tr '[:upper:]' '[:lower:]')
  if [ -n "${ENABLE_LIVE_TRADING+x}" ]; then
    echo "STOP: ENABLE_LIVE_TRADING is set in this shell and would override .env -- unset it (NO-GO 7)"
  elif [ -z "$v" ] || [ "$v" = false ]; then
    echo "OK: .env ENABLE_LIVE_TRADING='${v:-<unset: compose default false>}'"
  else
    echo "STOP: .env ENABLE_LIVE_TRADING is not false -- NO-GO 7 (close the window per smoke §4e, or fix .env)"
  fi
  unset v
  ```
- [ ] `$DC stop api`.
- [ ] **Put the Release 1 image back** (the tag from P-3; list tags in a new shell with `docker image ls crypto-trading-bot/api`). Do not rebuild old code:
  ```bash
  ROLLBACK_TAG=REPLACE_ME      # edit: the crypto-trading-bot/api:rel1-... tag recorded in P-3
  if [[ "$ROLLBACK_TAG" =~ ^crypto-trading-bot/api:rel1-[0-9]{8}-[0-9]{6}$ ]] \
     && docker tag "$ROLLBACK_TAG" crypto-trading-bot/api:local; then
    echo "restored $ROLLBACK_TAG"
    $DC up -d api
  else
    echo "STOP: ROLLBACK_TAG not set to a rel1-... tag from P-3 (or the tag does not exist) -- the api was NOT restarted"
  fi
  ```
- [ ] **Verify:** the api is `healthy`; `alembic current` prints `019 (head)` (the Release 1 entrypoint's `upgrade head` is a no-op); `docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local` equals the id recorded in P-3; `$DC exec -T api grep -c _quote_context /app/packages/trading/strategy_engine.py` prints `0`; `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false`. Leave `LIVE_EXIT_CHECK_INTERVAL_S` in `.env`: the Release 1 code ignores it (`extra="ignore"`, `apps/api/config.py:33`), and the compose line is harmless.
- [ ] Record what was rolled back, why, and the G-RB results in the deploy log. **G-LIVE rows L-1 and L-13 no longer hold**, so a live restart is closed until Release 2 is deployed again.

What a rollback gives up, so it is a decision and not a reflex:
- It restores the WP1.5a daily-PnL and peak behaviour that Release 2 fixed, the Release 1 resume seeding, and the Sprint 50 **HALT auto-stop** (a HALT stops the run and leaves positions unprotected; its runbook is `git show origin/main:docs/runbooks/circuit-breaker-halt-auto-stop.md`).
- Paper runs are rebuilt from fills on restart; bracket anchors are not persisted, so there is nothing to convert back.
- No schema or persisted-format change in Release 2 needs reverting: `git diff --stat origin/main..HEAD -- infra/alembic apps/api/db` is empty, and the additions in `run_persistence.py` only read.

### 10.11 Not verified in section 10

Nothing in this section was executed against a server, a Docker daemon, a database or an exchange in the authoring session.
- **Shell blocks:** syntax-checked with `bash -n` and the `jq`, `sed '$d'`, `awk` and regular-expression logic was run against sample data; the `$DC`, `psql`, `docker` and `curl` calls were not run. `$DC images api` (P-3) and `docker compose logs --since` / `exec -e` flags depend on the installed Compose version.
- **L-12 and L-14 scripts:** compiled with `python -m py_compile`; imports and signatures were read in the source (`CCXTMarketDataService`, `TickerQuote`, `CircuitBreakerTransition`, `_circuit_breaker_alert_text`, `TelegramNotifier`) but not executed, because the dependencies are not installed in the authoring environment. Whether the public Coinbase ticker equals the authenticated one is not known (the scripts therefore print `authenticated=`). After cf36a the L-12 script was re-checked against `apps/api/main.py:443-457` (the token is a `SecretStr`, unwrapped in-process), and in the cf36b-fix review both scripts, with their CF-36 logging lines, were run inside the source tree against stubbed `aiohttp` / `ccxt` calls with fake secrets; neither was run against Telegram or Coinbase.
- **L-2:** the chrony commands and their expected text come from the 1.5b synthesis (CF-19) and chrony's usual output; no host was available. The repo has no NTP configuration.
- **Grafana:** whether the provisioning file is re-read on `restart grafana`, and whether a contact point exists in the running Grafana, are not known.
- **P-8 note format:** the `bracket_atr_period:out_of_range` text is built from the validator's issue list (`scripts/wp13a_exit_config_precheck.py` `entry["note"]`, `packages/trading/exit_config.py:553-565`); the script was not run.
- **Settings failure mode:** that an invalid `LIVE_EXIT_CHECK_INTERVAL_S` stops the api at startup follows from the validator (`validate_live_exit_check_interval` in `apps/api/config.py`) and the 1.3b-b acceptance probe, not from a boot test.
- **Merge identity:** whether the commits on `origin/main` after merge keep the ids `6511af1` to `dc8cf83` and `6047162` (a squash merge would change them; P-7 then relies on the hash comparison), and whether the merged tree equals the accepted one (the 1.3b-b acceptance ran on `d7a9956` plus the patches; `git diff --shortstat d7a9956 dc8cf83` is `24 files changed, 4989 insertions(+), 111 deletions(-)`, equal to the acceptance numbers).
- **Whether Release 1 is deployed on the server** is not known; 10.3 decides at run time.
- **CF-36 (5.10, R2-8, L-19, Block L-1 markers):** step 1 was run in an authoring worktree (`746 passed, 2 skipped`, exit 0, with a canary `.env` refused and with canary secrets exported); steps 2 and 2b, the smoke §4c step-0 repeat and the Block L-1 CF-36 `chk` lines were not run against an image or a container. The marker strings were checked against the source tree.

---

## 11. Release 3 (WP2.1a/b/c and WP2.1-H): delta to sections 1 to 10

Commits, on top of `origin/main` `d36984e` (pull request #3): `60d7eaf` (C25: `BacktestRunner` keeps all `RiskParameters` fields), `182cdf7` and `12616da` (WP2.1a: one cost model; a resume validates exits at the run's own cost profile), `926d7d2`, `0207a43`, `b8245a1`, `bbfbed1` and `4990974` (WP2.1b: optimization runs persist their cost model, UI cost labels, leaderboard filter and selectors, relaunch carry-over, migration test), `04c3d15`, `d13d454`, `83696c2` and `b9bbfb8` (WP2.1c: circuit breaker on by default in backtests, re-baseline harness, per-run alerts keep backtests out), `5e0a108` and `a7ae1bf` (WP2.1-H: hardened reads of the stored cost profile, test logging isolation). A squash merge changes these ids; the hash comparison in P3-5 then carries the proof, as in P-7.

> [!IMPORTANT]
> **The WP2.1c re-baseline (CF-26) has not run.** It needs network access to the exchange API, which is not available yet. This section reports no re-baseline numbers and does not treat CF-26 as closed. How far the new cost model and the breaker move backtest and paper results is therefore **not measured**.

### 11.1 Form of this change: delta section, not integration

Same form as section 10, for the same reasons: the smoke runbook and `circuit-breaker-halt.md` cite this file by section, and most Release 1 steps do not apply (no Caddy change, no compose change, no new environment variable, no new secret). The change log moves from section 11 to section 12; no document cites it (checked with `grep` over `docs/`, `infra/`, `scripts/` and `tests/`). Two differences from Release 2: Release 3 has **one migration (020)** and it changes the **ui image**.

Sections 1 to 10 name `019 (head)` as the schema head. On the Release 3 tree the head is `020`: when a step of sections 4, 5, 8 or 10 is run after this deploy, read `019 (head)` as `020 (head)`.

### 11.2 What changes in Release 3

| Area | Change | Source of truth |
|---|---|---|
| Code: cost model (api image) | WP2.1a: one cost model. `trading/costs.py` holds an append-only registry of seven profiles and the default `DEFAULT_COST_PROFILE_ID` (`coinbase_advanced_1`). `RiskParameters` has no fee or slippage field. Backtest, optimizer, paper engine, ML gate, exit validation and the precheck read the run's profile. A run's profile is frozen in `runs.config["cost_profile"]` at create; promote overwrites it with the default; resume, boot and salvage resolve it and never write it. The live ledger, sizing and orders still use the exchange's real fees; the profile is an estimate there. | `packages/trading/costs.py`; `apps/api/services/cost_profiles.py`; `packages/trading/CLAUDE.md` (WP2.1a) |
| Code: breaker default | WP2.1c: backtest, optimizer, ML gate and the research scripts run with the circuit breaker on (`BACKTEST_CIRCUIT_BREAKER_DEFAULT`, `True`). There is no API toggle (scripts: `--no-circuit-breaker`). New backtest runs and optimizations stamp `engine_semantics.backtest_circuit_breaker=true`. | `packages/trading/safety.py:539`; `packages/trading/backtest.py`; `packages/trading/optimizer.py`; `apps/api/services/cost_profiles.py` (`engine_semantics`) |
| Code: stored-profile hardening | WP2.1-H: a stored cost profile on a paper or live run is accepted only in the cases listed in 11.9; anything else reads as invalid legacy. | `packages/trading/costs.py` (`resolve_run_cost_profile`, `_parse_persisted`); `apps/api/services/run_persistence.py` (`_loggable_fingerprint`) |
| API surface | Run and optimization responses gain `costProfile`, `legacyCosts`, `costProfileInvalid` and `liveEquivalentCosts`; create and optimize requests accept `costProfile` (a registry id); the portfolio aggregate gains `legacyCostRunsExcluded`. | `apps/api/schemas.py`; `apps/api/routers/optimize.py`; `apps/api/routers/portfolio.py` |
| Code: ui image | WP2.1b: cost-profile labels, the leaderboard default filter "Live-equivalent costs", per-mode profile selectors in the new-run and optimize forms, and relaunch (Launch Run, Duplicate Run) carrying the source run's registry profile. **The ui image is rebuilt and restarted** (Release 2 did not touch it). | `apps/ui/src/lib/cost-profile.ts`; `apps/ui/src/app/leaderboard/page.tsx` |
| Migrations | **020** (`down_revision` `019`): one nullable JSONB column, `optimization_runs.run_metadata`, which holds the cost profile and `engine_semantics` of an optimization run. No default and no backfill: NULL means a legacy row. `SET LOCAL lock_timeout = '5s'`; `ADD COLUMN` takes a brief ACCESS EXCLUSIVE lock on `optimization_runs`. The api entrypoint runs `alembic upgrade head` at every start, so a new api container migrates by itself; D3-4 does it first, explicitly, so a failure is visible and the app never starts on a half-migrated schema. | `infra/alembic/versions/020_optimization_runs_run_metadata.py`; `infra/docker-entrypoint.sh` |
| Compose, Caddy, environment | **None.** At `a7ae1bf`, `git diff --stat d36984e..HEAD` for `infra/docker-compose.yml`, `infra/Dockerfile.api`, `infra/Dockerfile.ui`, `infra/docker-entrypoint.sh`, `infra/Caddyfile` and `.env.example` is empty. There is no new variable. The default cost profile is a code constant: `DEFAULT_COST_PROFILE`, or any environment variable with `COST_PROFILE` in its name, is ignored and logged by name only as `cost_profile.env_ignored`. | `git diff --stat d36984e..HEAD -- infra .env.example`; `apps/api/config.py` (`warn_ignored_cost_profile_env`) |
| Grafana | The three per-run alert rules `trading-cb-halt`, `trading-kill-switch` and `trading-high-drawdown` carry `{run_id!~"bt-.*"}` (thresholds, labels and `trading-api-down` are unchanged), so a backtest cannot fire them. The two `trading_drawdown_pct` dashboard panels carry the same matcher. The rule change needs D3-6. | `infra/grafana/provisioning/alerting/trading-alerts.yml`; `infra/grafana/dashboards/trading-bot.json`; `tests/unit/test_wp21c_alert_scope.py` |
| Scripts | `scripts/wp13a_exit_config_precheck.py` gains `--cost-profile <id>` (P3-6) and validates each run at its own resolved profile. `scripts/wp21_rebaseline.py` is the CF-26 harness: it is not part of this deploy and nothing here runs it. | the script docstrings |
| Docs | `docs/runbooks/circuit-breaker-halt.md` (sections 1 and 2b, the backtest and alert-scope notes). Block L-21 and the L-21 row in 10.9 were added by WP2.1a. | `git diff --name-status d36984e..HEAD -- docs` |

### 11.3 Which path (decide first)

This section is written for a server that runs **Release 2** (schema `019`).

- [ ] Read the schema revision and check the running api image:
  ```bash
  $DC exec -T api sh -c 'cd /app/infra/alembic && python -m alembic -c alembic.ini current'
  $DC exec -T api grep -c '_quote_context' /app/packages/trading/strategy_engine.py    # Release 2 and 3: 3 (verified at a7ae1bf)
  $DC exec -T api ls /app/packages/trading/costs.py                                     # Release 3: the path; Release 2: "No such file or directory"
  ```
  - `019 (head)`, a `_quote_context` count of `3` and no `costs.py`: **Case A, Release 3 only.** Follow 11.4 to 11.7. Skip 3.5 and 3.6 (Caddy is not touched; the precheck is P3-6).
  - `020 (head)`, or `costs.py` present: the server already runs Release 3, or a deploy stopped half-way. STOP and escalate. Do not run P3-3: it would tag a Release 3 image as the rollback image.
  - Anything else (schema `016`, or `019` with a `_quote_context` count below `3`, so Release 2 is not deployed): STOP and escalate. Deploying the Release 3 tree on a server that does not run Release 2 is not described here (section 10.3 Case B covers `016` with the Release 2 content).

### 11.4 NO-GO conditions added by Release 3

NO-GO 1 to 4 and 7 to 12 apply to this deploy unchanged: it restarts the api, runs a migration, and uses the same key check, precheck, NaN query and `LIVE_EXIT_CHECK_INTERVAL_S` check as Release 2. Do not deploy if any of them holds, or if:

14. The path is not Case A (11.3), or the schema is not `019` at D3-2.
15. The built image does not hold the synced tree (P3-5), or does not contain migration 020 (`infra/alembic/versions/020_optimization_runs_run_metadata.py`).
16. The Release 2 images were not tagged for rollback (P3-3). The rollback order of 11.10 needs the older image to exist; a rebuilt "older" image is not the image that was running.

### 11.5 Pre-deploy additions (old stack still running)

Do these in order. P3-3 must come before any build.

- [ ] **P3-1 Path.** Section 11.3 done; Case A recorded in the deploy log.
- [ ] **P3-2 Live runs and backup.** Run the section 3.1 live-run query (expected `(0 rows)`; NO-GO 1) and the section 3.2 backup. Change one line of the 3.2 block so the dump is named for this release: `DUMP="$BACKUP_DIR/pre-release3-$(date +%Y%m%d-%H%M%S).dump"`. Continue only after `BACKUP OK`. The dump predates column `run_metadata`, so it cannot restore optimization attribution created after this deploy (11.10).
- [ ] **P3-3 Keep the running images for rollback, before building.** `crypto-trading-bot/api:local` and `crypto-trading-bot/ui:local` are overwritten by the build, and both images change in Release 3:
  ```bash
  $DC images api ui                                                    # note the IMAGE IDs of the running containers
  docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local   # must be the running api image; record the full id
  docker image inspect --format '{{.Id}}' crypto-trading-bot/ui:local    # must be the running ui image; record the full id
  STAMP=$(date +%Y%m%d-%H%M%S)
  docker tag crypto-trading-bot/api:local "crypto-trading-bot/api:prerel3-$STAMP" \
    && docker tag crypto-trading-bot/ui:local "crypto-trading-bot/ui:prerel3-$STAMP" \
    && echo "rollback images tagged: prerel3-$STAMP"
  ```
  Continue only after `rollback images tagged:` (NO-GO 16). If an id differs from the running container's image, STOP: someone rebuilt `:local` since the last deploy, and the running image is no longer on a tag.
- [ ] **P3-4 Stage, build, tag.** Stage the tree as in P-5 (sync, `bash $DEPLOY_DIR/infra/caddy-config.sh` with the `TAILSCALE_FQDN` count of `0`, `$DC config --quiet; echo "exit=$?"` giving `exit=0`, and the key-fingerprint check). The sync also replaces the Grafana provisioning files, which are bind-mounted read-only into the grafana container (D3-6 and V3-5 cover what that means). Then build both images that changed and tag the api for traceability:
  ```bash
  $DC build api ui
  SHORT_COMMIT=REPLACE_ME    # edit: the short commit hash recorded in P-5 (7-12 hex characters)
  if [[ "$SHORT_COMMIT" =~ ^[0-9a-f]{7,12}$ ]]; then
    docker tag crypto-trading-bot/api:local "crypto-trading-bot/api:rel3-$SHORT_COMMIT" && echo "tagged rel3-$SHORT_COMMIT"
  else
    echo "STOP: set SHORT_COMMIT to the commit recorded in P-5 -- the Release 3 tag was NOT created"
  fi
  ```
  Continue only after `tagged rel3-…` was printed, and note the tag in the deploy log: rollback step 3 (11.10) uses it.
- [ ] **P3-5 Image content (NO-GO 15).** Prove the image holds the synced tree. This is the P-7 comparison with the Release 3 file list; nothing is started:
  ```bash
  cd "$DEPLOY_DIR"
  FILES="apps/api/config.py apps/api/db/models.py apps/api/main.py apps/api/routers/ml.py \
         apps/api/routers/optimize.py apps/api/routers/portfolio.py apps/api/routers/runs.py \
         apps/api/schemas.py apps/api/services/cost_profiles.py apps/api/services/model_activation_gate.py \
         apps/api/services/run_orchestrator.py apps/api/services/run_persistence.py \
         packages/trading/backtest.py packages/trading/costs.py packages/trading/costs_cli.py \
         packages/trading/engines/live.py packages/trading/engines/paper.py packages/trading/exit_config.py \
         packages/trading/metrics.py packages/trading/optimizer.py packages/trading/risk.py \
         packages/trading/safety.py packages/trading/strategy_engine.py \
         infra/alembic/versions/020_optimization_runs_run_metadata.py"
  diff <(sha256sum $FILES | awk '{print substr($1,1,16), $2}') \
       <($DC run --rm --no-deps --entrypoint sha256sum api $(for f in $FILES; do echo "/app/${f#apps/}"; done) \
          | awk '{p=$2; sub("^/app/","",p); if (p ~ /^api\//) p="apps/" p; print substr($1,1,16), p}') \
    && echo "IMAGE MATCHES SYNCED TREE"
  sha256sum $FILES | awk '{print substr($1,1,16), $2}'        # record
  ```
  On the **source machine**, in the merged tree, print the same list (`sha256sum $FILES | awk '{print substr($1,1,16), $2}'` with the same `FILES`) and compare it by eye with the recorded one; also confirm the fourteen commits are present (skip the loop if the merge was a squash or rebase; the hash comparison then carries the proof):
  ```bash
  for c in 60d7eaf 182cdf7 12616da 926d7d2 0207a43 b8245a1 bbfbed1 4990974 04c3d15 d13d454 83696c2 b9bbfb8 5e0a108 a7ae1bf; do
    git merge-base --is-ancestor "$c" HEAD && echo "ok $c" || echo "MISSING $c"
  done
  ```
  Expected: `IMAGE MATCHES SYNCED TREE` and identical lists on both machines. The `FILES` list is every non-test Python file that the Release 3 commits changed in `apps/api`, `packages/trading` and `infra/alembic`; the ui image and the scripts are not covered by it.
- [ ] **P3-6 DS-1 precheck with the promote basis (NO-GO 9 and 10).** Run the WP1.3a precheck on the **Release 3 image** (the script is mounted from the synced tree; its `--cost-profile` option needs the Release 3 `trading.costs`). Read-only, as in P-8, plus `--cost-profile coinbase_advanced_1`:
  ```bash
  cd "$DEPLOY_DIR"
  install -d -m 700 ~/deploy-logs/$(date +%F)
  DS1R3=~/deploy-logs/$(date +%F)/precheck-r3-ds1.json
  ( umask 077
    $DC run --rm --no-deps --entrypoint python -v "$PWD/scripts:/app/scripts:ro" api \
      /app/scripts/wp13a_exit_config_precheck.py --json --cost-profile coinbase_advanced_1 | tee "$DS1R3"; echo "exit=${PIPESTATUS[0]}" )
  tail -n1 "$DS1R3" | grep -Eq '^wp13a-precheck: [0-9]+ run\(s\) checked, [0-9]+ invalid(, [0-9]+ would fail at promote)?\. No writes were made\.$' \
    && echo "precheck ran" || echo "STOP: precheck did not complete -- NO-GO 9"
  sed '$d' "$DS1R3" | jq -r '.[] | select(.valid != true) | "INVALID \(.mode) \(.status) \(.run_id) \(.codes|join(",")) \(.note // "")"'
  sed '$d' "$DS1R3" | jq -r '.[] | select(.would_fail_at_promote == true) | "WOULD-FAIL-AT-PROMOTE \(.mode) \(.status) \(.run_id) \(.promote_codes|join(","))"'
  ```
  The script prints a summary line after the JSON, so `sed '$d'` drops it before `jq`. **The summary line differs from the one P-8 matches:** with `--cost-profile` and at least one affected run it reads `..., N invalid, M would fail at promote. No writes were made.`, and the P-8 pattern does not match it. Use the pattern above. Do not hand-build a `DATABASE_URL`, and check that the saved file contains no `input_value=` or DSN before attaching it to the deploy log (section 3.6).
  - `exit=0`, `precheck ran`, nothing listed as `INVALID`: proceed.
  - `exit=2`, an empty file or no `precheck ran`: NO-GO 9.
  - An `INVALID` **live** run: NO-GO 10. An `INVALID` **paper** run: record it only; every invalid paper run is marked `error` at boot (as in P-8).
  - A `WOULD-FAIL-AT-PROMOTE` line is **informational and does not change the exit code**. The script validates each run at its own cost profile (a run without a stored profile resolves to `legacy_runtime_v0`, side cost 0.0065), so these runs still resume. The line means the run has a fixed stop-loss or take-profit in (0.0065, 0.0130]: after this deploy the same exit config is refused with 422 `invalid_exit_config` when a run is created or promoted afresh (11.8). Record the run ids and tell their owners before the deploy.
- [ ] **P3-7 NaN equity rows and `LIVE_EXIT_CHECK_INTERVAL_S` (NO-GO 11 and 12).** Run P-9 unchanged (expected `0`): the api restart resumes paper runs, and a NaN `peak_equity_hint` makes that resume raise. Run the validation block of P-4 unchanged (an invalid value stops the api at start, after the old api is already stopped); no new decision is needed, because Release 3 does not touch the setting.

### 11.6 Deploy

- [ ] **D3-1 Announce** the changes in section 11.8 (as in 4.1).
- [ ] **D3-2 Re-check** immediately before stopping: the section 3.1 live-run query returns `(0 rows)`, `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false` or nothing (4.1a), and the `.env` live-flag block of 10.6 D-2 prints `OK` (NO-GO 7). Record `curl -fsS -H @<(hdrs api) https://<FQDN>/api/v1/emergency/kill-switch` (expected `"latched": false`). Confirm once more that the schema is `019 (head)` with the first command of 11.3 (NO-GO 14).
- [ ] **D3-3 Stop the api:** `$DC stop api` (graceful, up to 30 s). The old ui may show errors until D3-5.
- [ ] **D3-4 Migrate to 020 with the new image.** Check first that no transaction is open longer than 5 s (the query in 4.3). Then:
  ```bash
  $DC run --rm --no-deps --entrypoint sh api -c \
    'cd /app/infra/alembic && python -m alembic -c alembic.ini upgrade head && python -m alembic -c alembic.ini current'
  ```
  Expected: one `Running upgrade 019 -> 020` line, then `020 (head)`. Migration 020 sets `lock_timeout = '5s'`; if it fires, the transaction rolls back, the schema stays `019` and the command fails. Clear the blocker and repeat. Never raise the timeout. If the blocker cannot be cleared, or the output is anything else, stop here: the schema is still `019`, so no downgrade is needed. Before putting the older images back, apply gate G-RB (6.1 rule 5) and record it: (a) the section 3.1 live-run query returns `(0 rows)`; in place of (c), which needs a running api, the `.env` live-flag block of 10.6 D-2 prints `OK`; (b) is waived on this branch only, because the Release 3 code never started, so no position can exist that the Release 2 images did not already manage until D3-3. Then restore the two rollback images (11.10, step 4), confirm that `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false` on the restored api, and escalate. Do not start the new api, because its entrypoint would run the same migration.
- [ ] **D3-5 Start api and ui together:** `$DC up -d api ui`. Caddy is not recreated: its configuration did not change. Wait until `$DC ps` shows both `healthy` (api `start_period` 60 s). If the api exits right away, read `$DC logs --tail 50 api`.
- [ ] **D3-6 Grafana (not optional since WP2.1c).** The per-run alert rules are provisioned from `infra/grafana/provisioning/alerting/trading-alerts.yml`. A changed expression is not live until Grafana has re-read the file; this runbook does it by restarting the container: `$DC restart grafana`. Until then the old, unscoped expressions stay live, and a backtest that reaches HALT in the API process can fire `trading-cb-halt` as a false alarm. V3-5 checks the result. Skipping D3-6 does not affect trading; it leaves the false-alarm risk.

### 11.7 Post-deploy verification (Release 3)

Repeat sections 5.1 (health; the schema is now `020 (head)`), 5.2 (single worker), 5.3 (`ENABLE_LIVE_TRADING` prints `false`; `DEBUG`) and 5.8 (lock-key constant, deployed commit recorded), and the last bullet of 5.4 (`/api/admin/*` is answered by the ui layer; the ui container was recreated). Then:

- [ ] **V3-1 Column exists.** Read-only:
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "SELECT data_type FROM information_schema.columns WHERE table_name = '"'"'optimization_runs'"'"' AND column_name = '"'"'run_metadata'"'"';"'
  ```
  Expected: `jsonb`.
- [ ] **V3-2 Boot log: cost profile events.** Expected: no lines. An error from `docker compose` means the check did not run; empty output proves nothing in that case.
  ```bash
  $DC logs --since 10m api | grep -E 'cost_profile\.(invalid|env_ignored)|runs\.cost_profile_fingerprint_mismatch'
  ```
  `cost_profile.invalid mode=... action=fallback_to_legacy` on a resume or at boot is a **database-tamper indicator**: follow 11.9 (invalid stored profile). `cost_profile.env_ignored` lists variable names that contain `COST_PROFILE` and reached the api environment: remove them; they never had an effect. `runs.cost_profile_fingerprint_mismatch` is an error log written when a backtest result does not carry the profile fingerprint the run froze at create; the numbers of that run were not produced under the frozen profile (11.9, fingerprint in the mismatch log).
- [ ] **V3-3 Breaker stamp (optional, after the first backtest created on the new image).** Read-only:
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "SELECT config->'"'"'engine_semantics'"'"'->>'"'"'backtest_circuit_breaker'"'"' AS backtest_cb, count(*) FROM runs WHERE run_mode = '"'"'backtest'"'"' GROUP BY 1 ORDER BY 1;"'
  ```
  Expected: `true` for runs created after this deploy; `false` for runs created by WP2.1a or WP2.1b (never rewritten); an empty value (NULL) for older runs that have no `engine_semantics` block.
- [ ] **V3-4 Registry default.** `$DC exec -T api grep -n 'DEFAULT_COST_PROFILE_ID: Final' /app/packages/trading/costs.py` prints the definition. Record the id: it is the default the L-21 check (V3-7) compares with the account's real fee tier.
- [ ] **V3-5 Grafana picked up the change (after D3-6).**
  ```bash
  $DC exec -T grafana grep -c '^ *expr: .*run_id!~' /etc/grafana/provisioning/alerting/trading-alerts.yml        # expected: 3
  $DC exec -T grafana grep -c 'trading_drawdown_pct{run_id!~' /var/lib/grafana/dashboards/trading-bot.json       # expected: 2
  ```
  The first count proves the container sees the new rule file, the second the new dashboard file. Then open the three rules (Alerting, Alert rules, folder Trading) and confirm that the query of `trading-cb-halt` ("Circuit breaker HALT"), `trading-kill-switch` ("Kill switch active") and `trading-high-drawdown` ("Drawdown above 10 percent") shows `{run_id!~"bt-.*"}`; the Grafana rule list shows the titles. The dashboard file provider polls its directory (`updateIntervalSeconds: 30` in `infra/grafana/provisioning/dashboards/dashboards.yml`), so the dashboard panels need no restart. If a rule still shows the old expression, repeat D3-6 once and record the result; do not edit the rule in the Grafana UI.
- [ ] **V3-6 Log hygiene (CF-36).** The image was rebuilt from a new commit: run section 5.10 as in R2-8 (steps 1, 2 and 2b; `CF36 log hygiene: PASS` and `CF36B_MARKERS_OK`) and record the commit and counts of step 1.
- [ ] **V3-7 G-LIVE L-21: fee tier.** After this deploy and **before the first live run on a WP2.1 image**, run the one-time fee-tier check: G-LIVE row L-21 (10.9.2, Table B) and Block L-21 (10.9.4). It compares the default cost profile (`DEFAULT_COST_PROFILE_ID`, V3-4) with the account's real Coinbase fee tier through a read-only call, and it needs the Coinbase API key in the api container's environment (the prerequisite stated in the block). It does not gate this deploy, because live stays off. If the result is not `FEE_TIER: DEFAULT CONFIRMED`, follow the list "If the tier differs (exact rule)" under Block L-21; the default is changed only by a reviewed code change, never through `.env`.
- [ ] **V3-8 Evidence to keep** (adds to 5.9): Case A; the P3-3 rollback tags and image ids; the P3-4 tag; the P3-5 hash list; the P3-6 file `precheck-r3-ds1.json` and the run ids it listed as `WOULD-FAIL-AT-PROMOTE`; the P3-7 count; the D3-4 output (`Running upgrade 019 -> 020`, `020 (head)`); V3-1 to V3-6 results; UTC deploy time.

Final step as in section 5.9: `unset API_KEY ADMIN_KEY LIVE_CONFIRM_TOKEN; exit`.

### 11.8 Operator-visible behaviour changes (communicate before deploy)

Cost model (WP2.1a/b):
- **One cost model, default by id.** The default profile is `coinbase_advanced_1` (maker 60, taker 120, slippage 10 bps: 130 bps per side, 260 bps round trip). Name it by id; the number changes only through one reviewed code line (`DEFAULT_COST_PROFILE_ID`), never through `.env`.
- **The exit floor doubles at create and promote: from 0.0065 to 0.0130 per side.** A new or promoted run is validated at the stricter of the default and its own profile. A fixed stop-loss of 0.0130 or less is refused with 422 `invalid_exit_config` and the issue `stop_inside_entry_cost` (field `bracket_stop_loss_pct`); a fixed take-profit of 0.0130 or less gets the issue `take_profit_inside_exit_cost` (field `bracket_take_profit_pct`). 0.0131 is accepted, with a warning. Before Release 3 the limit was 0.0065. Runs that already exist are not touched: resume, boot and salvage use the run's own stored profile, and a run without one resolves to `legacy_runtime_v0` (0.0065), so no existing run becomes un-resumable.
- **Idempotent retries across the deploy (release note).** A client that retries a create or promote request with the same `Idempotency-Key` across this deploy, with a fixed stop-loss or take-profit in (0.0065, 0.0130], now gets a 422 instead of a replay of the original run. The exit validation runs before the replay check, so no duplicate run can be created; the client must pick a value above 0.0130 and use a new key. The operator note in the 2.1a security review (check the resume log for `runs.intrabar_exit_checks interval_s=None reason=no_protective_exit`) is obsolete: it applied only to a WP2.1a build without its resume fix, and Release 3 ships the fix together with WP2.1a.
- **New paper costs.** A new paper run takes the default profile, or `coinbase_advanced_d6` (maker 40, taker 60, slippage 10 bps); paper accepts only these two live-equivalent profiles. Before, paper used 40/60 bps plus 5 bps slippage (`legacy_runtime_v0`, 65 bps per side). Paper results are lower in proportion to turnover, bracket levels sit on the all-in basis (a stop at distance d is only about d minus 0.0130 below the entry close), and the breaker and the daily and drawdown gates fire earlier, because the paper NAV falls faster. Backtest and optimize accept any registry id and default to the same profile; before, backtests used `legacy_backtest_v0` (10/15 bps plus 5 bps slippage). **How large these shifts are is not measured: the re-baseline (CF-26) is pending.**
- **Live takes no profile.** A non-null `costProfile` on a live create is refused with 422 `cost_profile_not_allowed_for_live`. A live promote overwrites the inherited profile with the default. The live ledger books the exchange's real fees; the profile feeds only the log-only warnings `cost_profile.tier_mismatch` (at engine start) and `cost_profile.fill_fee_drift` (a real fill whose fee is more than 5 bps off the profile).
- **Leaderboard default filter.** The leaderboard shows "Live-equivalent costs" by default: only runs whose server flag `liveEquivalentCosts` is true (a valid stored live-equivalent registry profile). Legacy-cost, stress and diagnostic runs, **including every run created before WP2.1**, are hidden until the user chooses "All cost models", where they carry labels (`legacy costs`, `stress`, `diagnostic`). The portfolio aggregate's best and worst return use the same eligibility; `legacyCostRunsExcluded` counts the runs that were left out only because of their cost model.
- **Relaunch pins the source's profile.** Launch Run (from an optimization) and Duplicate Run (from a non-live run) carry the source's registry profile, including the default id and explicit `legacy_*` ids. A relaunched run keeps its source pricing even if the default changes later. A new-run form submitted by hand omits `costProfile` for the "Platform default" choice.

Backtests, optimizations and breaker (WP2.1c):
- **The circuit breaker is on by default** in backtests, optimizations, the ML gate and the research scripts (`BACKTEST_CIRCUIT_BREAKER_DEFAULT`). New rows record `engine_semantics.backtest_circuit_breaker=true` (`runs.config` for backtests, `optimization_runs.run_metadata` for optimizations). Rows created by WP2.1a or WP2.1b record `false` and are never rewritten; older rows have no block.
- **Backtest numbers created by Release 3 differ from rows created by WP2.1a/2.1b (and from older rows).** The breaker drops or halves BUYs. Compare only rows with the same `engine_semantics` and the same cost profile. The size of the difference is not measured (CF-26 pending).
- **Backtests emit breaker gauges and critical logs.** A backtest that runs inside the API process writes `trading_circuit_breaker_state` (2 and 3 included), `trading_kill_switch_active` and the other per-run gauges under `run_id` `bt-<12 hex>`, and logs `circuit_breaker.halt_entered` at critical. The alert rules and the two drawdown dashboard panels exclude `bt-` ids; the log filter that does the same is in `circuit-breaker-halt.md` section 2b. Paper and live ids are UUIDs, which never start with `bt-`.
- **The Grafana alert change needs D3-6** (a restart of the grafana container).
- **ML models gated before Release 3 ran without the breaker.** The walk-forward activation gate now runs its mini-backtests with the breaker and the default cost profile (the gate basis was 40/60 bps plus 5 bps slippage before). The breaker setting is **not** recorded on the model (carry-forward CF-21C-04, target Fase 6); the cost profile of a model gated on Release 3 is stored in `ModelVersionORM.extra["cost_profile"]`. Treat every model gated before this deploy as gated without the breaker and at the old cost basis, and do not compare its gate result with one produced on Release 3.

Log events worth watching in addition to those in sections 7 and 10.8 (all present in the code): `cost_profile.invalid`, `cost_profile.env_ignored`, `cost_profile.tier_mismatch`, `cost_profile.fill_fee_drift` and `runs.cost_profile_fingerprint_mismatch`. None of them is wired to an automatic alert in `infra/`.

### 11.9 Stored cost profile: operator notes and a known limitation (WP2.1-H)

**Invalid stored profile.** A stored cost profile on a **paper or live** run is accepted only if it is a live-equivalent registry profile (`coinbase_advanced_1`, `coinbase_advanced_d6`) or the inferred `legacy_runtime_v0` profile (the one the paper auto-retry writes). Anything else is read as invalid legacy: the run response shows `costProfileInvalid` and `legacyCosts` both true, the exit-cost basis is `legacy_runtime_v0` (side cost 0.0065, never lower), and a resume or the boot recovery logs one warning, `cost_profile.invalid mode=<paper or live> action=fallback_to_legacy`, with no value in it. A run with no stored profile is plain legacy, with no warning.

No legitimate writer produces an invalid row: create, promote and the paper auto-retry write only the profiles above; resume, boot recovery and backtest completion rewrite `runs.config` but copy the stored `cost_profile` block unchanged; and the request field accepts only registry ids. **Treat the warning as an indicator that the database was changed outside the application, and investigate it.** Keep the log line (it holds no secret). Find the run (read-only, not executed against a server; the list endpoint caps `limit` at 500, so page with `offset` if `.total` is larger):
```bash
for m in paper live; do
  curl -fsS -H @<(hdrs api) "https://<FQDN>/api/v1/runs?mode=$m&limit=500" \
    | jq -r '.items[] | select(.costProfileInvalid == true) | "\(.runMode) \(.status) \(.id)"'
done
```
Then, before anything else changes:
1. Preserve evidence: save the run's `config` (read-only `SELECT config FROM runs WHERE id = '<run id>'`, in the `exec` form of section 3.1) and `$DC logs postgres` to `~/deploy-logs/<date>/`, with `umask 077`.
2. Check scope: whoever can write `runs.config` can also write `kill_switch_state`, the `entries_latch_*` columns, `idempotency_keys` and any run's exit config. Check the kill-switch state (`GET /api/v1/emergency/kill-switch`) and the exit config (`bracket_*`, `trailing_stop_pct`) of every live run for unexplained changes.
3. If a live run is affected, press the kill switch (section 5.5) or stop the run until the cause is known.
4. Establish who had write access to the database and when the row changed. If the write cannot be attributed, treat the database credentials as compromised: rotate them and escalate.

Do not edit the row by hand to silence the warning, and do not restart repeatedly. The fallback is fail-safe for exit validation, so the cost profile itself is not an immediate trading fault; the database write behind it may be.

**Fingerprint in the mismatch log.** In `runs.cost_profile_fingerprint_mismatch`, the field `config_fingerprint` now shows `"<malformed>"` for any stored value that is not exactly 16 lowercase hex characters; a well-formed value is shown as is. A `<malformed>` value is itself a sign of a changed row (see the paragraph above).

**Known limitation.** The stored cost-profile id is **not integrity-protected** against someone who can write to the database. A tampered live row can swap between the live-equivalent tiers (`coinbase_advanced_1` and `coinbase_advanced_d6`) without any flag. The effect is bounded: the basis is always a real tier and never below the legacy floor, and the live ledger, sizing and orders do not use the profile at all. The log-only checks `cost_profile.tier_mismatch` and `cost_profile.fill_fee_drift` may show a swapped tier on a live run, but nothing guarantees it. Closing the gap would need an integrity check on the stored row (for example an HMAC), which is outside the current threat model. Keep database write access restricted.

### 11.10 Rollback note for Release 3

Release 3 has a migration, so unlike Release 2 the rollback has a database step. **An image older than 020 cannot start on a database at 020**: its entrypoint runs `alembic upgrade head` against a revision it does not know, which fails as in 6.1 rule 1 (reproduced for 019 in the WP7.0 acceptance; inferred for 020, not reproduced). So the order is fixed: **stop the api, downgrade to 019 with the new image, then deploy the older image.**

Rules from section 6.1 that apply: stop the api first (rule 4); never raise `lock_timeout` (rule 6); `ENABLE_LIVE_TRADING=false` for the whole rollback (rule 7); and **gate G-RB in full** (rule 5), because the rule covers every downgrade, 020 included: (a) the section 3.1 live-run query returns `(0 rows)`, (b) no bot-held position above dust remains on Coinbase for any pair a live run traded, (c) `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false` on the running new image. Release 2's note (10.10) needed only (a) and (c) because Release 2 had no downgrade. Add the `.env` live-flag block of 10.6 D-2 before `$DC stop api`.

- [ ] **1. Record state** (Release 3 api still running): the kill switch, the running paper runs, and the optimization attribution you may want to keep (the downgrade drops it):
  ```bash
  $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "SELECT count(*) FROM optimization_runs WHERE run_metadata IS NOT NULL;"'
  ```
  If you want to keep the attribution, export it before step 3 (read-only; the data holds no secret). The line count of the file equals the count above:
  ```bash
  install -d -m 700 ~/deploy-logs/$(date +%F)
  ( umask 077
    $DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
      "\copy (SELECT id, run_metadata FROM optimization_runs WHERE run_metadata IS NOT NULL) TO STDOUT"' \
      > ~/deploy-logs/$(date +%F)/optimization-run-metadata-pre-019.tsv )
  wc -l < ~/deploy-logs/$(date +%F)/optimization-run-metadata-pre-019.tsv
  ```
- [ ] **2. Gate G-RB** as above, recorded in the deploy log. Then `$DC stop api` (and `$DC stop ui`).
- [ ] **3. Downgrade to 019 with the new image.** `crypto-trading-bot/api:local` must still be the Release 3 image; if it was overwritten, re-tag it first with `docker tag "crypto-trading-bot/api:rel3-$SHORT_COMMIT" crypto-trading-bot/api:local` (6.1 rule 2; set `SHORT_COMMIT` to the P3-4 value) and do not rebuild old code until after the downgrade:
  ```bash
  $DC run --rm --no-deps --entrypoint sh api -c \
    "cd /app/infra/alembic && python -m alembic -c alembic.ini downgrade 019 && python -m alembic -c alembic.ini current"
  ```
  Expected: a `Running downgrade 020 -> 019` line and `019 (head)`. Effect: the column `optimization_runs.run_metadata` is dropped. Only the persisted cost-profile and `engine_semantics` attribution of optimization runs is lost; the rest of every row is untouched. Re-applying 020 later does not bring the attribution back, and the pre-deploy dump (P3-2) has no such column.
- [ ] **4. Put the Release 2 images back** (the tags from P3-3; list them in a new shell with `docker image ls crypto-trading-bot/api` and `docker image ls crypto-trading-bot/ui`). Do not rebuild old code:
  ```bash
  ROLLBACK_TAG=REPLACE_ME      # edit: the suffix recorded in P3-3, for example prerel3-20261021-093000
  if [[ "$ROLLBACK_TAG" =~ ^prerel3-[0-9]{8}-[0-9]{6}$ ]] \
     && docker tag "crypto-trading-bot/api:$ROLLBACK_TAG" crypto-trading-bot/api:local \
     && docker tag "crypto-trading-bot/ui:$ROLLBACK_TAG" crypto-trading-bot/ui:local; then
    echo "restored $ROLLBACK_TAG"
    $DC up -d api ui
  else
    echo "STOP: ROLLBACK_TAG not set to a prerel3-... suffix from P3-3 (or a tag does not exist) -- api and ui were NOT restarted"
  fi
  ```
- [ ] **5. Grafana:** nothing to undo. The scoped rules and panels use the `run_id` label, which the older code also writes, so they work with it.
- [ ] **6. Verify:** the api is `healthy`; `alembic current` prints `019 (head)` (the Release 2 entrypoint's `upgrade head` is a no-op); `docker image inspect --format '{{.Id}}' crypto-trading-bot/api:local` equals the api id recorded in P3-3; `$DC exec -T api ls /app/packages/trading/costs.py` fails with "No such file or directory"; `$DC exec -T api printenv ENABLE_LIVE_TRADING` prints `false`.
- [ ] **7.** Record what was rolled back, why, and the G-RB results in the deploy log. G-LIVE rows that depend on Release 3 no longer hold, so a live restart is closed until Release 3 is deployed again.

What a rollback gives up, so it is a decision and not a reflex:
- The optimization attribution created since the deploy (step 3).
- Backtests and optimizations return to the pre-2.1c semantics (no breaker) and the old cost numbers; the cost registry and its warnings disappear.
- Paper runs created on Release 3 carry a `cost_profile` block in `runs.config`. The Release 2 code has no reference to `cost_profile` (checked with `git grep` at `d36984e`), so it should ignore the block, but what it does with such a run on the next restart is **not verified**. Record the paper runs that exist at step 1 and check them after step 6.

### 11.11 Not verified in section 11

Nothing in this section was executed against a server, a Docker daemon, a database, a Grafana or an exchange in the authoring session.
- **Shell blocks:** syntax-checked with `bash -n`. The `grep -Eq` summary pattern and the `jq` filters of P3-6 were run against output that `scripts/wp13a_exit_config_precheck.py` produced from sample results (including a run that would fail at promote); the `FILES` comparison of P3-5 was run against a simulated image tree, with a negative control; the SQL quoting of V3-1 and V3-3 was run through `sh -c` with a stub `psql`. The `$DC`, `docker`, `psql` and `curl` calls themselves were not run. `$DC images api ui` (P3-3) and `docker compose logs --since` / `--no-log-prefix` depend on the installed Compose version.
- **TTY and `\r`:** `$DC run` allocates a TTY when stdin is a terminal, and its output may then end lines with `\r`. That would make the P3-5 compare or the P3-6 summary pattern print a STOP (fail-closed, never a false pass). If either stops with no visible cause, re-run that block with `$DC run -T` before escalating. This was not reproduced (no Docker in the authoring session); the same applies to 3.6, P-7 and P-8.
- **Grafana:** whether the alert provisioning file is re-read on `restart grafana` (D3-6), whether a contact point exists, and whether the dashboard provider reloads a changed file within its 30 s interval are not known. V3-5 is the check.
- **Old image on a 020 database:** the exact error text is inferred from the 019 case (6.1 rule 1).
- **Release 2 code on Release 3 data:** see 11.10 (paper runs with a `cost_profile` block).
- **Caddy and the recreated ui container:** Caddy is not recreated in D3-5. That it keeps reaching the new ui container is expected from how Caddy resolves upstream names, not tested here; the `/api/admin/*` probe in the repeat of 5.4 above would show a failure.
- **Invalid-profile query (11.9):** the `runs` list call and its `jq` filter follow the response schema (`costProfileInvalid`, `runMode`, `status`, `id`) and were not run.
- **Pre-existing, not changed here:** the L-21 table row and Block L-21 refer to "§8.4", which is the numbering of the 2.1 specification, not of this file; the rule it points to is the list "If the tier differs (exact rule)" under Block L-21.
- **CF-26:** the re-baseline has not run (see the start of this section), so no statement here depends on its result. Some code comments and `docs/risk-model.md` say "closes CF-26"; that refers to the code half only (the breaker default). The re-baseline, and with it the closure of CF-26, is pending.

---

## 12. Change log

| Date | Change |
|---|---|
| 2026-09-28 | Initial proposal for branch HEAD `3db9148` (producer draft, pending critic review). |
| 2026-09-28 (r2) | Revised per `reports/vp2-docs/final-synthesis-docs.md` (WD-01..WD-22): anchored Caddyfile greps, secure backup, `env_set`/`hdrs` helpers, section 4.1a re-check, rollback gate G-RB and RB-2, DC-70-4 renaming, hand-off aligned with the smoke runbook. |
| 2026-09-28 (r3) | Round-2 fixes per `reports/vp2-docs/final-synthesis-docs.md` Round 2 addendum (WR2-01..WR2-11). |
| 2026-10-12 (r4, Release 2) | Release 2 (WP1.5a/b, WP1.6, WP1.3b-a/b, HEAD `dc8cf83` plus CF-36 (`6047162`, cf36b)): new section 10 (delta, G-LIVE table, rollback note), Appendix B, six corrections of statements that the merge of Release 1 made stale (scope and Related line, section 1 migrations row, 3.2, 5.8, 7) and four pointers (sections 2, 3.3, 8, 9). The change log moved from section 10 to section 11. Producer draft `reports/vp2-docs-b/`; reconciled with CF-36 (sections 5.10, 10.2, 10.5 P-7, 10.7 R2-8, 10.8, 10.9 L-1/L-19 and Block L-8, Appendix B) per `reports/vp2-cf36/final-synthesis-cf36b.md` part B. Review fixes per its Round 2 addendum: Block L-12 unwraps the `SecretStr` token and stops when Telegram is not configured, Blocks L-12 and L-14 turn on CF-36 logging first, `.env` live-flag check in 10.6 D-2 and the 10.10 gate, fail-closed 10.10 restore, "already Release 2" branch in 10.3, symbol citations for the setting, `UVICORN_*` in the 5.10 operator rule. |
| 2026-10-12 | cf36b: added section 5.10 (G-LIVE L-19 log-hygiene verification, operator rule, L-8 counters with the `trading_` prefix, access-log note) per `reports/vp2-cf36/synthesis-spec.md` section 9 and `final-synthesis-cf36a.md` section 6. |
| 2026-10-12 (cf36b-fix) | Section 5.10 per `reports/vp2-cf36/final-synthesis-cf36b.md`: scoped to CF-36 images; step 1 hermetic (`env -i`, no `.env`, `--no-cov`, commit recorded); step 2 is the paper baseline and the L-19 gate run moves to the live config (smoke §4c step 0); new step 2b (cf36b markers in the image); CF-A note, uvicorn log-level operator rule, R-08 `http.request` note. |
| 2026-10-14 (WP2.1a) | Added G-LIVE row L-21 (Table B) and Block L-21 (10.9.4): the one-time Coinbase fee-tier verification for the WP2.1 default cost profile, per `reports/vp2-wp2.1/synthesis-spec.md` section 8 as amended by SA-01 (`reports/vp2-wp2.1/final-synthesis-2.1a.md` section 8; GET-only guard at `Exchange.fetch`, enforced verdict line). Reads the raw `transaction_summary` `fee_tier` (ccxt 4.5.40 `fetch_trading_fees()` swaps maker and taker; unauthenticated `load_markets()` reports the hardcoded 0.012/0.006). Not run against a container in this change. |
| 2026-10-20 (r5, Release 3) | Release 3 (WP2.1a/b/c and WP2.1-H, `60d7eaf` to `a7ae1bf`): new section 11 (delta, rollback note with the migration 020 order, operator-visible changes, stored-profile notes), Appendix C, and a Release 3 hand-off in section 8; pointers in the scope paragraph and sections 1, 2 and 9; a current-state note in the Grafana row of 10.2; the WP2.1a change-log row reworded (it now says "as amended by SA-01"); the change log moved from section 11 to section 12. R3-DOCS round per `reports/vp2-wp2.1h/final-synthesis.md` section 6.1; producer draft `reports/vp2-r3docs/`. The WP2.1c re-baseline (CF-26) is pending and not reported. Additions in this round: smoke runbook D-0 expects `020 (head)` on Release 3, Compose errors made visible in V3-2 and in the halt-runbook log filters, evidence-preservation and scope steps for an invalid stored profile (11.9), an optional attribution export (11.10 step 1), and gate G-RB on the D3-4 failure branch. |

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

## Appendix B. Release 2 condition traceability

| Condition / ID | Where | Section here |
|---|---|---|
| G-LIVE L-1..L-20 (consolidated) | `reports/vp2-wp1.3b/final-synthesis-1.3b-b.md` section 12 (supersedes `final-synthesis-1.5b.md` F.3, `final-synthesis-1.6.md` R3-7 and `final-synthesis-1.3b-a.md` R2-5) | 10.9 |
| L-2 CF-19 NTP slew; L-3 SEC-C-1 NaN rows (until CF-21); L-5 stale rows (self-expiring); L-6, L-7 | `reports/vp2-wp1.5/final-synthesis-1.5b.md` F.3 | 10.5 P-9, P-10; 10.9 |
| L-4 amended, L-8 amended, L-9 retired, L-10 not issued, L-11 CB wiring, L-12 alert channel | `reports/vp2-wp1.6/final-synthesis-1.6.md` R3-7 | 10.7 R2-4; 10.9 |
| DS-1 as hard gate (C-201); L-20 | `reports/vp2-wp1.3b/final-synthesis-1.3b-a.md` R2-5 | 10.4 (9, 10); 10.5 P-8 |
| L-13 code gate 1.3b; L-14 ticker sanity (amended by C-2, CR-10); L-15; L-16; L-17 (a)-(e); L-18 | `reports/vp2-wp1.3b/final-synthesis-1.3b-b.md` section 12; `docs/runbooks/circuit-breaker-halt.md` section 7 | 10.9 |
| L-19 log hygiene, CF-36 (a)+(b+)+(e) (relabelled by erratum A-1) plus CF-A | same; `reports/vp2-cf36/synthesis-spec.md` section 9; `final-synthesis-cf36a.md` A2.6/A2.7; `final-synthesis-cf36b.md` | 5.10; 10.7 R2-8; 10.9 (L-19, Block L-1); smoke §4c step 0 |
| `LIVE_EXIT_CHECK_INTERVAL_S`: default 300, 0 off, 60..3600 | `apps/api/config.py` (`Settings.live_exit_check_interval_s`); `infra/docker-compose.yml:197`; `.env.example:193`; `reports/vp2-wp1.3b/acceptance-report-1.3b-b.md` | 10.2, 10.5 P-4, 10.7 R2-1 |
| Acceptance (PASS / ACCEPT) | `reports/vp2-wp1.5/acceptance-report-1.5a.md`, `-1.5b.md`; `reports/vp2-wp1.6/acceptance-report.md`; `reports/vp2-wp1.3b/acceptance-report-1.3b-a.md`, `-1.3b-b.md` | 10.9 (gate in words) |
| No migration in WP1.5/1.6/1.3b | `reports/vp2-wp1.3b/final-synthesis-1.3b-b.md` section 10 row 10; `git diff origin/main..HEAD -- infra/alembic` | 10.2, 10.6 D-4, 10.10 |

## Appendix C. Release 3 condition traceability

| Condition / ID | Where | Section here |
|---|---|---|
| WP21B-CR-10 (2.1b C-05): Release 3 runbook section, migration 019 to 020, migrate-first via the entrypoint, the rollback order, 020 downgrade loses only optimization attribution | `reports/vp2-wp2.1b/final-synthesis.md` C-05 and section 5.6; `infra/alembic/versions/020_optimization_runs_run_metadata.py` (Deployment order, Downgrade path) | 11.2, 11.6 D3-4, 11.10 |
| CF-21-13: operator-visible changes (floor 0.0065 to 0.0130, new paper costs, leaderboard default filter on `liveEquivalentCosts`, SA-21B-02) | `reports/vp2-wp2.1/synthesis-spec.md` sections 6.2 and 15 (CF-21-13); `reports/vp2-wp2.1b/final-synthesis.md` section 5.7 | 11.8 |
| CF-21a-04 (WP21A-SEC-08): retry across the deploy gets a 422, not a replay; the SEC section 5.5 note is obsolete | `reports/vp2-wp2.1/final-synthesis-2.1a.md` (carry-forwards); `reports/vp2-wp2.1/security-report-2.1a.md` | 11.8 |
| WP21A-CR-202: change-log wording "as amended by SA-01" | `reports/vp2-wp2.1/critic-report-2.1a-r2.md` | 12 (change log) |
| X-7 (WP21-RK-09): DS-1 precheck with `--cost-profile coinbase_advanced_1`, informational "would fail at promote" | `reports/vp2-wp2.1/synthesis-spec.md` section 6.2; `scripts/wp13a_exit_config_precheck.py` | 11.5 P3-6 |
| L-21 pointer: after the deploy, before the first live run on a WP2.1 image | `reports/vp2-wp2.1/synthesis-spec.md` section 8; 10.9.2 and 10.9.4 | 11.7 V3-7, 8 |
| 2.1c breaker-on note (`BACKTEST_CIRCUIT_BREAKER_DEFAULT`, `engine_semantics.backtest_circuit_breaker=true`) | `reports/vp2-wp2.1c/final-synthesis.md`; `packages/trading/safety.py` | 11.2, 11.8, 11.7 V3-3 |
| CF-21C-06 (1) to (4): alert reload, `bt-` gauges and logs, differing backtest numbers, ML models gated without the breaker (CF-21C-04 deferred, G-H-02) | `reports/vp2-wp2.1c/final-synthesis.md` section 5.5; `reports/vp2-wp2.1c/delta-signoff.md` section 8; `reports/vp2-wp2.1h/final-synthesis.md` G-H-02 | 11.6 D3-6, 11.7 V3-5, 11.8 |
| WP21CD-SEC-04, WP21CD-CR-06, WP21CD-OPS-03, WP21CD-OPS-04 (2.1c-delta section 8) | same | `circuit-breaker-halt.md` section 2b (anchored log filter, timeless restart wording, dashboard note); 10.2 Grafana row (current-state note); `infra/grafana/dashboards/trading-bot.json` (two panel queries); R3-DOCS additions: smoke runbook D-0 expects `020 (head)` on Release 3, the `jq` filter passes non-JSON lines through as `unparsed`, Compose errors made visible (11.7 V3-2, `circuit-breaker-halt.md` sections 2b and 2c), tamper-evidence steps (11.9), attribution export (11.10 step 1), G-RB on the D3-4 failure branch (11.6) |
| WP21BD-CR-01, WP21BD-CR-02 | `reports/vp2-wp2.1b/delta-signoff.md` section 3 | `apps/ui/src/lib/cost-profile.ts` (comment above `costProfileField`); `infra/alembic/versions/020_optimization_runs_run_metadata.py` (docstring) |
| WP2.1-H, SA-21H-01 and SA-21H-03: invalid stored profile, `cost_profile.invalid`, tamper indicator | `reports/vp2-wp2.1h/final-synthesis.md` section 5.2 | 11.7 V3-2, 11.9 |
| WP2.1-H, WP21H-SEC-03: `config_fingerprint="<malformed>"` | same, "Contract records" | 11.7 V3-2, 11.9 |
| WP21H-SEC-04 (accepted residual): stored profile id not integrity-protected | same, section 5.3 | 11.9 |
| SA-21H-05, WP21H-CR-04 (developer note: one-process test runs, uncached structlog in the harness) | same | `README.md`, Development |
