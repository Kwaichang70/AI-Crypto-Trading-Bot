IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Apps Folder
Contains the two application entry points:
- `api/` — FastAPI backend server (Python 3.11+)
- `ui/` — Next.js frontend dashboard (TypeScript + Tailwind)

### Conventions
- Each app has its own dependency management
- API uses Pydantic models with strict validation
- UI communicates with API via REST endpoints
- Both apps are containerized via Docker Compose
