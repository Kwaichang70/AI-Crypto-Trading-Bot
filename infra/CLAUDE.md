IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Infrastructure Folder
Deployment configuration and database migrations.

### Contents
- `docker-compose.yml` — Orchestrates: api + ui + postgres + redis
- `migrations/` — Database migration scripts (Alembic)
- Dockerfile definitions for each service

### Requirements
- `.env.example` for API keys + config (never commit actual `.env`)
- Graceful shutdown: sync positions, check order status before exit
- Health check endpoints for container orchestration

### Sandbox / CI validation notes (confirmed WP1.7b, C-04/INF-01)
- `docker compose config` (from `infra/`) works in this sandbox even though
  the Docker **daemon** is not running here (`docker info` fails to reach
  `/var/run/docker.sock`) — `config` only parses and interpolates
  `docker-compose.yml` plus `.env`, it does not talk to the daemon. Use it
  to validate compose syntax and `:?` fail-fast variables without needing a
  live stack.
- **Always add `--quiet`.** Plain `docker compose config` prints the fully
  interpolated file, including every secret substituted into `${VAR:?...}`
  placeholders (`ADMIN_API_KEY`, `POSTGRES_PASSWORD`, `NEXTAUTH_SECRET`,
  etc.) in cleartext. `docker compose config --quiet` validates the same
  thing (still exits non-zero with a stderr message on any missing
  required variable) without printing the rendered config. Never run the
  non-`--quiet` form where the output could land in a log, CI artifact, or
  screen-share.
- `caddy validate` / the `caddy` binary is not installed in this sandbox;
  `infra/caddy-validate.sh` is a post-deploy smoke test that needs a live
  Tailscale FQDN and running containers, not a syntax check. Caddyfile
  route-order changes are covered instead by the static
  `tests/unit/test_wp17b_infra_config.py` suite.
- pytest for these static infra-config tests needs the repo's `.venv`
  (`structlog` and friends are not on the bare system `python3` in this
  sandbox); run `.venv/bin/python -m pytest tests/unit/test_wp17b_infra_config.py
  tests/unit/test_wp13a_infra_body_limit.py --no-cov` rather than a bare
  `pytest` invocation.

### Caddyfile `request_body` / `max_size` gotchas (WP1.3a, confirmed against
### upstream Caddy docs/issue tracker — see `tests/unit/test_wp13a_infra_body_limit.py`)
- `request_body { max_size <size> }` **must use the block form**. An
  inline `request_body max_size 1MB` on one line is silently accepted by
  the Caddyfile parser but the `max_size` value is never applied — Caddy
  only reads it from inside the `{ }` block
  (caddyserver/caddy is the module; see rybbit-io/rybbit#1136 for a
  real-world instance of this exact mistake). Any future edit that
  "simplifies" this to one line is a silent regression, not a no-op.
- The `caddyhttp.requestbody` module (and its Caddyfile `request_body`
  directive) has shipped in Caddy's standard distribution since 2.0, so
  the `caddy:2-alpine` tag pinned in `docker-compose.yml` supports it with
  no image bump.
- Caddy documents 413 (Payload Too Large) as the response when `max_size`
  is exceeded, but with `reverse_proxy` in front of an upstream that reads
  the request body itself (as here — api:8000 / ui:3000), an oversized
  request can occasionally surface as 502 (Bad Gateway) instead, per a
  still-open upstream race (caddyserver/caddy#4558, #5652 as of Caddy
  2.11.x). Do not treat an observed 502 on an oversized-body probe as a
  broken limit — treat a plain 200 as the failure signal instead. This
  Caddy-level cap is defence in depth; the FastAPI app enforces its own
  independent ASGI-level body-size limit regardless of what Caddy returns.
