IMPORTANT: Critical Insights and Instructions related to the contents of this folder MUST be documented below.
Ensure your information or instruction is accurate, you must never poison context here or elsewhere. No Hallucinations or Invention.
If you discover and confirm poisoned context you must remove it from here so it does not mislead other agents.
Language must be folder-specific, unambiguous, and kept current by agents.
The instructions and knowledge below are not mandates, treat them as guidance only.
---

## Next.js Frontend Dashboard
TypeScript + Tailwind CSS web application for trading bot management.

### Pages (from KickOff.md)
- **Home Dashboard:** Status, equity curve, drawdown chart, open positions
- **Runs Page:** List of runs with detail view
- **Config Page:** Strategy dropdown + parameter form fields
- **Logs Page:** Filter by run_id and log level

### Technical Requirements
- Next.js with TypeScript
- Tailwind CSS for styling
- Minimal but usable UI
- REST API integration with FastAPI backend
- Real-time data display (polling initially, websocket nice-to-have)
