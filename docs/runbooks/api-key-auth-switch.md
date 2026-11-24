# API key auth switch (CF-97-37): Part 1, release R2 (wiring)

**Audience:** the operator of the production stack (`/opt/trading-bot`).
**Source:** `reports/vp2-cf9737/design-ruling.md` (DR-9737-02, DR-9737-14, DR-9737-17), `synthesis-37a-r1.md` sections 6 and 7.
**Scope of this part:** release R2 only. Part 2 (the flip, release R4) is added with that release. Nothing in R2 changes who can call the API: `REQUIRE_API_AUTH` and `TRADING_ENVIRONMENT` keep their defaults.

## 1. What R2 changes

- `ui` gets `API_KEY` as a **required runtime variable** (compose `ui` environment). It is never a build argument and never `NEXT_PUBLIC_*`.
- `api` gets `API_KEY_HASH` as a **required** variable. It has no effect while auth is off, except that the shadow check from release R1 starts counting keyed requests.
- Caddy gets a `handle /bff/*` block with the same 1MB body cap as `/api/*`. It is inert: nothing calls `/bff` until release R3.
- `api` and `ui` get bounded Docker logs (`json-file`, 50m, 5 files). Reason: with `API_KEY_HASH` set and auth off, every keyless request (all dashboard polls until R3) logs one `auth.would_reject` WARNING, and compose set no log rotation before.

From this release the ui container receives `API_KEY`, and the four existing admin proxy routes (`/api/admin/kill-switch`, `kill-switch/clear`, `runs/[id]/resume`, `runs/[id]/entries-latch/clear`) start sending it as `X-API-Key` (an existing branch that was dead until now). With API auth still off the API ignores it; resume and entries-latch clear are counted by the 37-A shadow check, the kill-switch routes are admin-only and are not counted. Browser requests send no key until the BFF (R3) and are counted as `missing`; log rotation bounds the resulting WARNING lines.

R1 (patch 37-A, the shadow check) must be deployed first, so that the counter exists when the key starts to flow.

## 2. The hazard: the variables must be in `.env` before the new compose file arrives

Compose interpolates the whole file before it looks at any service. A missing or empty `API_KEY` or `API_KEY_HASH` makes **every** `$DC` command fail (`up`, `logs`, `ps`, `exec`, `stop`, also with one service selected). This was measured with Docker Compose v5.6.1 and dummy values (`reports/vp2-cf9737/verification-infra.md`). The error stops at the first missing variable, so fixing one can reveal the other.

Order, therefore: (1) pre-checks, (2) generate both variables in `.env`, (3) check them, (4) only then sync the new files, (5) `$DC config --quiet` and require exit code 0 **before any `down`, `up` or `stop`**, (6) deploy.

If compose refuses to load during an incident, the plain Docker CLI still works because it does not interpolate the compose file: `docker ps`, `docker logs <container>` (piped to `grep`) and `docker stop <container>`. Never use a general `docker inspect` of the api or ui container.

`docker run --env-file .env` (`Dockerfile.api:14`) is for development only, never with the production `.env`: it hands every value, including the raw `API_KEY`, to the container and bypasses the compose rename.

## 3. Steps

All commands run on the server as the deploy user. `DC`, `env_set` and `hdrs` are defined in `deploy-fase1-minimumset.md` section 0; redefine them in every new shell. Never `export` the key. Never run these steps under `set -x` (or `bash -x`): it prints every expanded value, including the key.

### 3.1 Pre-checks (counts only; any other value stops the deploy)

```bash
grep -c -E '^(TRADING_)?ENVIRONMENT=' "$DEPLOY_DIR/.env"                    # A1: total lines
grep -c -E '^(TRADING_)?ENVIRONMENT=development$' "$DEPLOY_DIR/.env"        # B1: must equal A1 (an empty value is not accepted)
grep -c -E '^REQUIRE_API_AUTH=' "$DEPLOY_DIR/.env"                          # A2: total lines
grep -c -E '^REQUIRE_API_AUTH=false$' "$DEPLOY_DIR/.env"                    # B2: must equal A2 (0 and 0, or 1 and 1)
test -e "$DEPLOY_DIR/.env.local"; echo "env_local=$?"                       # expected env_local=1 (absent)
printenv ENVIRONMENT TRADING_ENVIRONMENT REQUIRE_API_AUTH | wc -l           # deploying shell: expected 0
```

R2 changes no default, so a `.env` that already says `REQUIRE_API_AUTH=false` is normal (A2 and B2 both `1`). Any other value, including an empty one, makes A and B differ: stop and report it.

If compose runs from a systemd unit, also check `systemctl show <unit> -p Environment | grep -c ENVIRONMENT` (expected `0`). Compose takes the shell environment over `.env`, so the `printenv` line is not optional.

### 3.2 Generate the key and its hash (stdin only)

The key comes from 32 random bytes, goes straight into a helper through stdin, is never echoed and never in an argument list or shell history. The hash is computed inside the helper from the same stream. The key is independent of `ADMIN_API_KEY`.

```bash
env_set_key_pair() {  # usage: openssl rand -hex 32 | env_set_key_pair   (reads ONE line; needs 64 hex characters)
  local key
  IFS= read -r key && [ "${#key}" -ge 64 ] || { echo "STOP: need at least 32 random bytes (64 hex characters) on stdin"; return 1; }
  printf '%s\n' "$key" | env_set API_KEY \
    && printf '%s' "$key" | sha256sum | cut -d' ' -f1 | env_set API_KEY_HASH
}
openssl rand -hex 32 | env_set_key_pair
```

`env_set` stops on a duplicate line or a wrong mode; follow its `STOP:` line. Do not type or paste a key into a shell. To rotate later, see Part 2.

### 3.3 Check the pair (prints only OK or MISMATCH)

```bash
key_pair_check() {
  local k h a
  k=$(grep -E '^API_KEY=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"'")
  h=$(grep -E '^API_KEY_HASH=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"'")
  a=$(grep -E '^ADMIN_API_KEY=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"'")
  [ -n "$k" ] && [ "$k" != "$a" ] && [ "$(printf '%s' "$k" | sha256sum | cut -d' ' -f1)" = "$h" ] && echo OK || echo MISMATCH
}
key_pair_check
```

`OK` means the hash matches the key and the key differs from the admin key. `MISMATCH` covers every failure: run 3.2 again. Then count the lines (each must print `1`):

```bash
grep -c -E '^API_KEY=.+' "$DEPLOY_DIR/.env"
grep -c -E '^API_KEY_HASH=.+' "$DEPLOY_DIR/.env"
```

### 3.4 Sync the files, validate, deploy (D12 window: no live positions)

1. Sync the tree as in `deploy-fase1-minimumset.md` section 3.4, then re-substitute the Caddyfile hostname (`infra/caddy-config.sh`).
2. `$DC config --quiet; echo "exit=$?"` must print `exit=0` and nothing else. After each fix run it again.
3. `$DC up -d --build api ui` recreates the api and ui containers: it is an API restart, which orphans live runs (`recover_orphaned_runs`). With no live positions this is the accepted D12 case.
4. Caddy: follow `caddy-metrics-block.md` section 2 (validate, reload, then the live check, with the stale bind-mount fallback). The syntax of the new block was **not** run against Caddy when this was written (no `caddy` binary was available); `caddy validate` before the reload is the check.

### 3.5 Checks after the deploy

```bash
docker inspect --format '{{.HostConfig.LogConfig.Type}} {{json .HostConfig.LogConfig.Config}}' "$($DC ps -q api)"
docker inspect --format '{{.HostConfig.LogConfig.Type}} {{json .HostConfig.LogConfig.Config}}' "$($DC ps -q ui)"
```

Expected for both: `json-file` with `max-file` `5` and `max-size` `50m`. Both commands print the log settings only.

The `/bff/*` block with its body cap is in force (counts only, run through the caddy container). **Unverified when this was written:** no `caddy` binary was available, so the exact JSON strings could not be checked; the first host run records them under V-9737-05.

```bash
$DC exec -T caddy caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile | grep -o '"/bff/\*"' | wc -l          # expected: at least 1
$DC exec -T caddy caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile | grep -o '"max_size":1000000' | wc -l   # expected: 4 (auth, admin, api, bff)
```

A different count is a stop: report it, do not reinterpret it. No HTTP probe replaces this check: the UI's sign-in middleware covers `/bff/*` and answers an unauthenticated request with a redirect before the body is read, so a 413, a 502 or a 307 would not tell whether the block is in force.

The value compose resolves for the api's `ENVIRONMENT` (V-9737-10, at every release). Keep the filter inline: the full rendering contains secrets and is never printed.

```bash
$DC config --format json | jq -r '.services.api.environment.ENVIRONMENT'      # expected: development
```

The ui container holds the same key as `.env`. Only `OK` or `MISMATCH` is printed; the single quotes keep `$API_KEY` unexpanded on the host, and the expected hash goes through the pipe (stdin), never an argument. `sha256sum` in the ui image is BusyBox (alpine); that is unverified here and the first host run confirms it.

```bash
grep '^API_KEY_HASH=' "$DEPLOY_DIR/.env" | cut -d= -f2- \
  | $DC exec -T ui sh -c 'read h; [ "$(printf %s "$API_KEY" | sha256sum | cut -d" " -f1)" = "$h" ] && echo OK || echo MISMATCH'
```

Expected `OK`.

Prove the key pair at the API with one keyed request, then read the counter (Prometheus scrapes every 15 s, so wait about 30 s):

```bash
API_KEY=$(grep -E '^API_KEY=' "$DEPLOY_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d "\r\"'")
curl -s -o /dev/null -w '%{http_code}\n' -H @<(hdrs api) https://<host>/api/v1/runs
unset API_KEY
curl -s 'http://127.0.0.1:9090/api/v1/query?query=api_auth_shadow_total'
$DC logs --since 10m api 2>&1 | grep -c -E '"reason": ?"reject"'
```

Expected: the curl prints `200` (auth is off); `api_auth_shadow_total{result="ok"}` is above 0 and `{result="reject"}` is 0; the last command prints `0`. A `reject` above 0 means `API_KEY` and `API_KEY_HASH` do not match: stop and run 3.2 and 3.3 again. `{result="missing"}` is expected to rise during R2 (the dashboard still sends no key); it is not a failure until R3.

**What this gate proves, and what it does not.** It proves: (1) the `.env` pair is consistent (3.3); (2) the api process holds the matching hash (a keyed request raises `ok`, `reject` stays 0); (3) the ui container holds the same key (the container check prints `OK`); (4) log rotation is active on `api` and `ui`. It does **not** prove that every UI path attaches the key. That is the R3 gate (V-9737-12).

## 4. Rollback

Revert only the two `:?` lines in `infra/docker-compose.yml`: delete the `ui` `API_KEY` line, and set the `api` line back to `API_KEY_HASH: ${API_KEY_HASH:-}`. Keep the `logging:` blocks and the `/bff/*` Caddy block (the block is inert). Then `$DC up -d api ui`.

`API_KEY_HASH` may stay in `.env`: with auth off it only feeds the shadow check, and its WARNING lines stay bounded because the log rotation stays active. `API_KEY` may stay too; nothing reads it after the revert. Do not revert the whole patch: that would bring back unbounded logs while `API_KEY_HASH` is still set. If compose refuses to load, use the plain Docker CLI commands in section 2.

## 5. Not verified when this was written

- The Caddyfile syntax and the behaviour of the `/bff/*` block, and the exact strings of the `caddy adapt` count check (no `caddy` binary; static text pins only; the first host run records the strings under V-9737-05).
- `sha256sum` and `sh` inside the ui image for the container-side key check (BusyBox in an alpine image is assumed). The shell logic of that check was run outside a container against a scratch file.
- The `docker inspect --format` output on the production Docker version and the log rotation behaviour on the host (compose syntax was validated with Docker Compose v5.6.1 `config` only).
- The helpers in 3.2 and 3.3 were run only against a scratch file with dummy values, never against a real `.env`.
