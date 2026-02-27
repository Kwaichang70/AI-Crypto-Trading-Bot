# AI Crypto Trading Bot - Primary Agent Instructions

## Project Overview
AI Crypto Trading Bot: A production-grade cryptocurrency trading platform with automated strategy execution, backtesting, paper trading, live trading, and a web dashboard.

**Tech Stack:** Python 3.11+ (FastAPI, SQLAlchemy, CCXT, Pandas/NumPy), Next.js (TypeScript, Tailwind), PostgreSQL, Redis, Docker.

## Code Modification Policy (MANDATORY)
**No direct code edits are permitted.** All code changes MUST follow the agent workflow:
1. A coding sub-agent produces a diff/patch in a report
2. A critic/reviewer agent reviews and approves or rejects the proposed changes
3. Only after approval does an executor agent apply the edits
4. Claude may present a diff to a reviewer, and the reviewer approves or rejects strictly following project documentation guidelines

## Workflow Execution Strategy (MANDATORY)
When performing tasks, Claude Code **MUST**:
1. Analyze the task to identify independent subtasks
2. Select appropriate specialized agents from `.claude/agents/` using:
   - Domain expertise match with the task
   - Required tools availability
   - Agent color diversity (when multiple agents with similar capabilities exist)
3. For complex advisory tasks, launch 2-5 multiple agents with different expertise to generate diverse perspectives
4. Always conclude with a Synthesis Agent to consolidate findings into a unified recommendation
5. Employ Git-based checkpoints: `git checkout -b claude-session-[timestamp]-[purpose]` for version control
6. **Critical:** Ensure agent outputs are trackable with unique IDs when issues are identified

## Available Workflows
Check `.claude/workflows/` for YAML workflow definitions. Choose workflows appropriate to the task complexity:
- `explore-plan-code.yaml` — General development workflow
- `tdd-workflow.yaml` — Test-driven development
- `review-and-fix.yaml` — Code review and remediation
- `feature-implementation.yaml` — Full feature lifecycle
- `security-audit.yaml` — Security-focused review
- `performance-optimization.yaml` — Performance analysis and improvement

## Available Sub-Agents
All agents are defined in `.claude/agents/`. Key agents include:
- **python-backend-specialist** — FastAPI, SQLAlchemy, async Python patterns
- **trading-engine-architect** — Strategy design, execution engine, CCXT integration
- **risk-management-expert** — Position sizing, circuit breakers, safety gates
- **data-pipeline-engineer** — OHLCV data, feature engineering, ML pipelines
- **nextjs-frontend-specialist** — Dashboard UI, TypeScript, Tailwind
- **database-architect** — PostgreSQL schema, migrations, query optimization
- **devops-infrastructure-specialist** — Docker, CI/CD, monitoring
- **security-audit-specialist** — API key safety, live trading gates, OWASP
- **testing-quality-specialist** — Pytest, integration tests, deterministic backtests
- **quant-strategy-analyst** — Backtesting metrics, strategy validation, financial analysis
- **code-critic** — Code quality review, actionable audit reports
- **architecture-critic** — System design review, pattern compliance
- **synthesis-arbiter** — Consolidates multi-agent findings, resolves conflicts
- **code-executor** — Applies approved diffs after critic approval

## R.A.C.R.S. Cycle (Mandatory for all significant work)
1. **Reason & Act:** Primary agent analyzes task and produces output
2. **Critique:** Output automatically reviewed by a Critic agent
3. **Reflect:** Primary agent refines work using Critic feedback
4. **Synthesize:** Arbiter agent consolidates all perspectives into final plan

## Escalation Protocol (MANDATORY)
After **2 consecutive rejections** on the same issue in any critic-producer loop, the workflow MUST escalate:
- Route to `synthesis-arbiter` with verdict `ESCALATE_TO_USER`
- Include full context: original task, both rejection reports, attempted fixes
- Do NOT continue the rejection loop beyond 2 iterations

## Workflow Chain Selection Guide
Match your task to the appropriate chain (see `memory/workflow-chains.md` for full blueprints):

| Task Type | Chain Pattern | Key Agents |
|-----------|--------------|------------|
| New strategy | quant → trading-arch → python → [critic + risk] → synthesis → executor → testing | 7 agents |
| Database schema | db-arch → python → critic → executor → testing | 5 agents |
| OHLCV data pipeline | [data-eng + db-arch] → python → critic → executor → testing | 6 agents |
| Risk management | [risk + trading-arch] → python → [critic + security] → synthesis → executor → testing | 8 agents |
| API endpoint | python → critic → executor → testing | 4 agents |
| Dashboard UI | nextjs → [python] → critic → executor → testing | 5 agents |
| Docker/CI | devops → security → critic → executor → testing | 5 agents |
| Backtest module | [quant + trading-arch] → synthesis → python → critic → executor → testing | 7 agents |
| Safety gates | [security + risk] → synthesis → python → [critic + security] → executor → testing | 8 agents |
| Major feature | [trading + python + db] → synthesis → arch-critic → impl → [critic + security] → synthesis → executor → testing | 10+ agents |

**Default fallback:** Any agent output → `code-critic` → `code-executor` → `primary`

## Critical Routing Rules
- **NEVER** route to `code-executor` without passing through `code-critic` first
- **NEVER** let a producer agent self-approve — always require external critic review
- **ALWAYS** use `synthesis-arbiter` when 2+ agents produce parallel outputs
- **ALWAYS** include `security-audit-specialist` for code touching exchange APIs or env vars
- **ALWAYS** include `testing-quality-specialist` after `code-executor` applies changes

## Report Format
All agent outputs must follow the template in `.claude/template/report.md`.

## Project Structure
```
/apps/api          # FastAPI backend
/apps/ui           # Next.js frontend
/packages/trading  # Trading engine, strategies, risk, execution
/packages/data     # Data fetching, caching, indicators
/packages/common   # Shared types, utilities
/infra             # Docker Compose, migrations
/docs              # Architecture docs
/reports           # Agent output reports
```

## Key Safety Rules
- Live trading is OFF by default; requires `ENABLE_LIVE_TRADING=true` + API keys + confirm token
- Never log API keys or secrets
- Never commit `.env` files
- All backtests must be deterministic (seed control)
- Spot-only for MVP (no leverage/derivatives)
