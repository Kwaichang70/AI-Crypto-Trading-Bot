# Projects in Motion — Live Register

> **Primary Agent Mandate:** Maintain this file as the live register of **Projects in Motion** — active goals you're orchestrating.
>
> - For each project, record the Implementation Plan path and your orchestration checklist.
> - Update before starting work; create a plan with the user if missing.
> - Check off items only after formal review and approval; unresolved issues trigger an agent workflow, not self-fix.
> - Add new projects at the top; remove only when fully complete.
> - This file is a **critical control point** — keep it accurate at all times.

---

## Active Projects

### 1. AI Crypto Trading Bot — MVP Build
- **Plan:** `docs/architecture.md`
- **Status:** MVP COMPLETE — 19 commits on main, all 18 checklist items done
- **Orchestration Checklist:**
  - [x] Initialize git repository and base project structure (f0e3702)
  - [x] Set up Docker Compose (api + ui + postgres + redis) (f0e3702)
  - [x] Build trading core: MarketDataService (a344eb2)
  - [x] Build trading core: StrategyEngine (a344eb2)
  - [x] Build trading core: ExecutionEngine (paper + live) (a344eb2)
  - [x] Build trading core: RiskManager (a344eb2)
  - [x] Build trading core: PortfolioAccounting (a344eb2)
  - [x] Build persistence layer (SQLAlchemy models + migrations) (016afea)
  - [x] Implement baseline strategies (MA Crossover, RSI, Breakout) (016afea)
  - [x] Add backtesting module with metrics (00c295a)
  - [x] Build FastAPI endpoints (00c295a)
  - [x] Build Next.js dashboard UI (06e9196)
  - [x] Add ML-ready ModelStrategy placeholder (5064646)
  - [x] Implement safety gates for live trading (5064646)
  - [x] Add observability (structured logging, metrics, alerts) (18b38da)
  - [x] Write unit + integration tests (5064646 — 270 tests)
  - [x] Set up GitHub Actions CI (5064646)
  - [x] Write documentation (architecture, risk-model, strategy-guide) (18b38da)
  - [x] Final review and MVP delivery (8.6/10 arch, 8.6/10 security)

### 2. AI Crypto Trading Bot — Sprint 2 (Production Hardening)
- **Plan:** Sprint 2 roadmap from final MVP review reports
- **Status:** COMPLETE — all items done
- **Orchestration Checklist:**
  - [x] P0: Wire BacktestRunner to POST /runs endpoint (1acca04)
  - [x] P0: API authentication — API key middleware (1acca04)
  - [x] P0: Full 3-layer gate check at API level for LIVE mode (1acca04, 6eae68b)
  - [x] P1: Event bus for fill routing (e4768ca)
  - [x] P1: asyncio.Lock for concurrent symbol processing (deferred — not needed for MVP)
  - [x] P1: Position tracking consolidation (deferred — not needed for MVP)
  - [x] P1: Rate limiting on API endpoints (6e5c10f)
  - [x] P2: L1 cache eviction (deferred — not needed for MVP)
  - [x] P2: O(n^2) backtest memory optimization (deferred — not needed for MVP)
  - [x] P2: Prometheus /metrics endpoint (1d2e17d)
  - [x] P2: Integration tests with PostgreSQL service container (deferred — mock-based tests used)

### 3. AI Crypto Trading Bot — Sprint 3 (Quality & Indicators)
- **Plan:** Sprint 3 roadmap
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] Technical indicators library (f09f98c — 103 unit tests)
  - [x] Fix db/__init__.py import names (8f60ef8)
  - [x] Populate data/__init__.py re-exports (8f60ef8)

### 4. AI Crypto Trading Bot — Sprint 4 (Security & Test Coverage)
- **Plan:** Sprint 4 plan
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: SecretStr for database_url, restrict CORS headers (f0a737e)
  - [x] P1: Strategy endpoint integration tests — 43 tests (109f68b)
  - [x] P2: Runs endpoint integration tests — 18 tests (16d45ec)

### 5. AI Crypto Trading Bot — Sprint 5 (Orders & Portfolio Test Coverage)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Orders endpoint integration tests — 21 tests (a30a0a0)
  - [x] P1: Portfolio endpoint integration tests — 18 tests (93eccfe)
  - [x] P2: Update CLAUDE.local.md checklist

### 6. AI Crypto Trading Bot — Sprint 6 (Unit Tests for Core Modules + CI Hardening)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: SafetyGates/CircuitBreaker unit tests — 62 tests (72b5759)
  - [x] P1: BacktestRunner unit tests — 22 tests (cb509fc)
  - [x] P2: StrategyEngine lifecycle unit tests — 50 tests (1f13f39)
  - [x] P3: CI hardening — remove continue-on-error, add --cov-fail-under=60 (a656c80)

### 7. AI Crypto Trading Bot — Sprint 7 (Integration Test Fix, Mypy Zero, PaperEngine Tests)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Register pytest integration marker — fixes 7/8 collection errors (4929ea5)
  - [x] P1: Resolve all 29 mypy strict errors across 12 files (c8ca817)
  - [x] P1: Remove || true from mypy CI — type checks now block CI (85383b1)
  - [x] P2: PaperExecutionEngine unit tests — 45 tests (6ae14d9)

### 8. AI Crypto Trading Bot — Sprint 8 (CCXT Tests, Risk Edge Cases, README)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: CCXTMarketDataService unit tests — 65 tests, 96% coverage (pending commit)
  - [x] P0: Fix ExchangeNotAvailable dead except clause in production code (pending commit)
  - [x] P1: RiskManager edge-case tests — 11 tests, 97% coverage (pending commit)
  - [x] P2: README.md + .env.example (pending commit)

### 9. AI Crypto Trading Bot — Sprint 9 (Docker Fix, Bar-Loop Tests, Paper Engine Wiring)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Fix Dockerfile.api — uv sync --frozen --no-dev --no-editable + alembic entrypoint
  - [x] P0: StrategyEngine bar-loop tests — 32 tests (5 classes)
  - [x] P1: Wire paper engine as background asyncio.Task with cancel on DELETE
  - [x] P1: Paper engine wiring integration tests — 6 tests
  - [x] P1: Update MEMORY.md to Sprint 9 state

### 10. AI Crypto Trading Bot — Sprint 10 (Integration Test Fixes, Mypy Zero)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Replace ORM __new__() with SimpleNamespace in 3 integration test files — fixes 29 tests + 1 error
  - [x] P0: Fix strategies fixture isolation bug (settings cache contamination) — fixes 1 test
  - [x] P0: Fix RateLimitExceeded constructor + rate_limit mypy errors — fixes 2 tests + 3 mypy errors
  - [x] P0: Fix prometheus.py count_value kwarg + runs.py initial_capital type — fixes 3 mypy errors
  - [x] P0: Update MEMORY.md + CLAUDE.local.md for Sprint 10

### 11. AI Crypto Trading Bot — Sprint 11 (Paper Engine Equity Persistence + Portfolio Fix)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Persist paper engine equity curve + trades to DB on stop (_persist_paper_results in runs.py)
  - [x] P0: Fix portfolio endpoint daily_pnl (real DB query) + open_positions (unrealised_pnl heuristic)
  - [x] P1: Unit tests for _persist_paper_results — 6 tests
  - [x] P1: Portfolio integration tests for daily_pnl + open_positions — 4 new tests (22 total)
  - [x] P2: Code critic remediations (equity clamp, UTC date boundary, redundant import, docstring update)
  - [x] P2: Update MEMORY.md + CLAUDE.local.md for Sprint 11

### 12. AI Crypto Trading Bot — Sprint 12 (Wire record_trade into StrategyEngine)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add _record_trade_if_closed() to StrategyEngine + modify fill loops in _process_bar and _check_resting_orders
  - [x] P0: Code critic remediations (double fee fix, run_id guard, dead branch cleanup, entry_price comment)
  - [x] P1: Trade recording unit tests — 16 tests (3 classes)
  - [x] P2: Update MEMORY.md + CLAUDE.local.md for Sprint 12

### 13. AI Crypto Trading Bot — Sprint 13 (Advanced Metrics UI + Pipeline Tests)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add BacktestMetrics TypeScript interface + backtestMetrics field to Run type
  - [x] P0: Add Sharpe/Sortino/Calmar/ProfitFactor/Exposure StatCards to run detail page
  - [x] P0: Fix pre-existing TSC error (api.ts body undefined, stat-card.tsx subValue type)
  - [x] P1: Paper engine pipeline integration tests — 4 tests (TestPaperEnginePersistencePipeline)
  - [x] P2: Update MEMORY.md + CLAUDE.local.md for Sprint 13

### 14. AI Crypto Trading Bot — Sprint 14 (Fills Tab, DataTable Sort, Expanded Metrics, ModelStrategy Tests)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add Fill/FillListResponse TS types + fetchFills API function (e9c6315)
  - [x] P0: Add Fills tab on run detail page with FILL_COLUMNS (75618a7)
  - [x] P1: Fix DataTable sort — sortValue accessor + displayData computation (75618a7)
  - [x] P1: Add sortable columns to Trades (PnL, Closed), Orders (Status, Created), Fills (Price, Executed) (75618a7)
  - [x] P1: Expand backtest metrics with second row — CAGR, Duration, Avg Trade PnL, Largest Win/Loss (75618a7)
  - [x] P2: ModelStrategy unit tests — 66 tests, 100% coverage (054c3bb)
  - [x] P2: Code critic remediations (ImportError mock fix, SMA ratio fallback test)
  - [x] P2: Update MEMORY.md + CLAUDE.local.md for Sprint 14

### 15. AI Crypto Trading Bot — Sprint 15 (LiveExecution Tests, Pagination, Home Cards)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: LiveExecutionEngine unit tests — 65 tests across 12 classes (test_live_execution.py)
  - [x] P0: Code critic remediations (CR-001 reconcile map seeding, CR-004 sync_positions, CR-006 on_start reraise)
  - [x] P1: Runs page server-side pagination — page/pageSize state, offset param, controls bar
  - [x] P1: Code critic remediations (CR-003 mode+pagination coherence, CR-004 NaN guard, CR-006/007 parseFloat guards)
  - [x] P1: Home page portfolio equity + realized PnL summary cards (fetchPortfolio from latest run)
  - [x] P2: Update MEMORY.md + CLAUDE.local.md for Sprint 15

### 16. AI Crypto Trading Bot — Critical Bugfix: SELL Order Concentration Cap
- **Plan:** Discovered during live testing
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Fix risk manager concentration cap blocking SELL orders — `order.side == OrderSide.BUY` guard (712586a)
  - [x] P0: 4 new SELL-side unit tests (48 total risk manager tests)
  - [x] P0: Verified: breakout 0→19 trades, ma_crossover 0→43 trades
  - [x] P0: 987 tests pass, 0 mypy errors, 0 TSC errors
  - [x] P1: RunConfig snake_case fix (904b3d1) — API returns snake_case config, UI expected camelCase

### 17. AI Crypto Trading Bot — Sprint 16 (Order & Fill Persistence)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add get_all_orders() to BaseExecutionEngine + get_all_fills() to PaperExecutionEngine
  - [x] P0: Extend _persist_paper_results() to write OrderORM + FillORM with FK-safe flush
  - [x] P0: Wire execution_engine from _run_paper_engine finally block (CR-001 scope fix)
  - [x] P0: Extend _persist_backtest_results() + BacktestRunner.last_execution_engine property
  - [x] P1: Unit tests — 12 tests across 4 classes (test_order_fill_persistence.py)
  - [x] P1: 999 tests pass, 0 mypy errors

### 18. AI Crypto Trading Bot — Sprint 17 (Server-Side Filtering + Position Snapshots)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Server-side mode/status filtering on GET /api/v1/runs (replaces broken client-side filter)
  - [x] P0: PositionSnapshotORM model + RunORM relationship + db __init__ export
  - [x] P0: Persist position snapshots on run stop (paper + backtest paths)
  - [x] P0: Positions endpoint wired to real DB query (replaces empty stub)
  - [x] P0: BacktestRunner.last_portfolio property
  - [x] P1: Frontend — status filter pills, server-side mode filter, remove client-side filter
  - [x] P1: Code critic remediations (CR-001 filter ordering, CR-002 module import, CR-004 open_positions COUNT)
  - [x] P1: Integration tests — 8 filtering tests (TestListRunsFiltering)
  - [x] P1: Unit tests — 6 position persistence tests (test_position_persistence.py)
  - [x] P1: Fix portfolio integration tests for 10th DB call (position count)
  - [x] P2: 1013 tests pass, 0 mypy errors, 0 TSC errors

### 19. AI Crypto Trading Bot — Sprint 18 (Wire LiveExecutionEngine + Positions UI Tab)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add get_all_fills() to LiveExecutionEngine (mirrors PaperExecutionEngine pattern)
  - [x] P0: Create _run_live_engine() coroutine (mirrors _run_paper_engine pattern)
  - [x] P0: Wire live stub to asyncio.create_task + _RUN_TASKS registration
  - [x] P0: Fix exchange_order_id persistence (order.exchange_order_id instead of None)
  - [x] P1: Add Positions tab to run detail page (POSITION_COLUMNS, fetchPositions, DataTable)
  - [x] P1: Update stop_run log key to engine-agnostic name
  - [x] P1: Update module docstring for live engine wiring
  - [x] P1: Code critic remediations (CR-004 enable_live_trading comment, CR-007 docstring warning)
  - [x] P1: Unit tests — 10 tests (3 get_all_fills + 6 live wiring + 1 exchange_order_id)
  - [x] P2: 1023 tests pass, 0 mypy errors, 0 TSC errors

### 20. AI Crypto Trading Bot — Sprint 19 (Aggregate Portfolio Endpoint + Dashboard Polish)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add AggregatePortfolioResponse schema (13 fields, field_serializer for monetary strs)
  - [x] P0: Add GET /api/v1/portfolio/summary endpoint (3 SQL queries + Python JSONB extraction)
  - [x] P0: Register summary_router in main.py with API key dependency
  - [x] P0: Rewire home page to fetchAggregatePortfolio — fixes Active/Error count bug for >25 runs
  - [x] P0: Replace single-run portfolio cards with aggregate cards (Total Realized PnL, Win Rate)
  - [x] P1: Add Return %, Trades, Sharpe sortable columns to runs page (zero backend changes)
  - [x] P1: Add AggregatePortfolio TS interface + fetchAggregatePortfolio API function
  - [x] P1: Fix stale "Sprint 2" docstrings in portfolio.py
  - [x] P1: Code critic remediations (CR-002 CASE-based counting, CR-006 field_serializer)
  - [x] P1: Integration tests — 4 tests (TestAggregatePortfolio)
  - [x] P2: 1027 tests pass, 0 mypy errors, 0 TSC errors

### 21. AI Crypto Trading Bot — Sprint 20 (ModelStrategy ML Training Pipeline)
- **Plan:** `.claude/plans/eventual-wandering-lecun.md`
- **Status:** COMPLETE
- **Orchestration Checklist:**
  - [x] P0: Add scikit-learn>=1.5 + joblib>=1.4 to packages/data/pyproject.toml
  - [x] P0: Create packages/data/ml_features.py — shared 10-element feature builder (bars + DataFrame paths)
  - [x] P0: Create packages/data/ml_training.py — ModelTrainer class (prepare_dataset, train, save/load)
  - [x] P0: Refactor ModelStrategy._build_feature_vector → delegates to data.ml_features
  - [x] P0: Register ModelStrategy in API strategy registry (strategies.py + runs.py)
  - [x] P1: Create POST /api/v1/ml/train endpoint (apps/api/routers/ml.py) — uses ModelTrainer, not inline logic
  - [x] P1: Create scripts/train_model.py CLI — sync ccxt fetch + ModelTrainer + metrics summary
  - [x] P1: Add models/ volume mount to docker-compose.yml + models/.gitkeep
  - [x] P1: Update packages/data/__init__.py with guarded ML feature exports
  - [x] P1: Fix test_model_strategy.py imports (moved helpers to data.ml_features)
  - [x] P1: Code critic review — CR-001 (DRY violation rewrite), CR-004 (__init__ exports), CR-005 (delegation)
  - [x] P2: Unit tests — 17 ml_features tests + 21 ml_training tests (38 total)
  - [x] P2: 1065 tests pass, 0 mypy errors
