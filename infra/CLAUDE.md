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
