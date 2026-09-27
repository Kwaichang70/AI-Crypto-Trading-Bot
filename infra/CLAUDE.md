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
