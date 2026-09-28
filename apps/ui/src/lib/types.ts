/**
 * apps/ui/src/lib/types.ts
 * -------------------------
 * TypeScript interfaces mirroring all FastAPI Pydantic schemas in
 * apps/api/schemas.py.
 *
 * Rules:
 * - camelCase field names (the API uses alias_generator=to_camel)
 * - Monetary values as `string` (backend preserves Decimal precision)
 * - Timestamps as `string` (ISO-8601 from the wire; parse with new Date() at display time)
 * - UUIDs as `string`
 * - Use readonly arrays/objects for immutability
 */

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

export type RunMode = "backtest" | "paper" | "live";
export type RunStatus =
  | "running"
  | "stopped"
  | "error"
  | "archived"
  | "orphaned" // WP1.8a: engine task gone (API restart / graceful shutdown); needs an operator resume
  | "resuming"; // WP1.8a: short-lived compare-and-set lock held while POST /runs/{id}/resume is in flight

export interface RunConfig {
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  symbols: readonly string[];
  timeframe: string;
  mode: RunMode;
  initial_capital: string;
  backtest_start?: string;
  backtest_end?: string;
  /**
   * WP1.3a (SY-13a-08): resolved (never absent after create/promote/normal
   * resume/paper recovery) — an explicit `allowPyramiding` on the request
   * wins, otherwise the strategy's own `default_allow_pyramiding` ClassVar
   * (`True` only for dca_rsi_hybrid/grid_trading, `False` elsewhere).
   * Optional here purely for back-compat with pre-WP1.3a persisted configs.
   */
  allow_pyramiding?: boolean;
}

export interface Run {
  id: string;
  runMode: RunMode;
  status: RunStatus;
  config: RunConfig;
  startedAt: string;
  stoppedAt: string | null;
  createdAt: string;
  updatedAt: string;
  backtestMetrics?: BacktestMetrics | null;
  /**
   * Number of closed (round-trip) trades in this run.
   * null for pre-M5 run records or when no trades executed yet.
   * Added by M5 backend (Sprint 49). Optional with `?` so older records
   * deserialise without errors.
   */
  nClosedTrades?: number | null;
  /**
   * Three-tier confidence label derived from PSR and trade count.
   * "high" | "medium" | "low" | null (when PSR not computable).
   * Duplicates BacktestMetrics.confidenceFlag so the runs-list page can
   * render a badge without fetching full backtest metrics.
   * Added by M5 backend (Sprint 49).
   */
  confidenceFlag?: "high" | "medium" | "low" | null;
  /**
   * Probabilistic Sharpe Ratio surface value exposed at the run level
   * so it is sortable / filterable from the list endpoint.
   * null when fewer than 30 return observations.
   * Added by M5 backend (Sprint 49).
   */
  psr?: number | null;
  /**
   * True when this run meets the leaderboard eligibility criteria
   * (n_closed_trades >= 10 AND confidence_flag IN ("high", "medium")).
   * Defaults to false for pre-M5 records and runs with insufficient data.
   * Added by M5 backend (Sprint 49).
   */
  leaderboardEligible?: boolean;
  /**
   * WP1.3a (SY-13a §5 API contract): warnings emitted by the exit-config /
   * pyramiding validator on a SUCCESSFUL 201 create. Absent/empty on every
   * other read of a `Run` (GET /runs/{id} does not replay them) — this is a
   * create-response-only field, so callers must capture it from the
   * `createRun()` result at the moment of creation (see CF-13a-1 item 3).
   */
  configWarnings?: readonly ConfigWarning[];
  /**
   * WP1.3a (SY-13a-16): set on a PROTECTIVE resume (200) when one or more
   * exit-config components were dropped ("salvaged") to let the resume
   * proceed under the waiver. `null` when nothing was waived (including
   * every non-resume read and every normal-mode resume).
   */
  exitConfigWaived?: ExitConfigWaived | null;
  /**
   * WP1.3a (SY-13a-16): true when, after a protective-resume salvage, the
   * surviving config has NO downside exit at all (no SL/trailing) — the
   * critical "flatten recommended" case. `null`/absent outside a protective
   * resume response.
   */
  exitManagerMissing?: boolean | null;
}

/**
 * A single open position at backtest end, mark-to-market at the final bar.
 * Mirrors OpenPositionMTMResponse in apps/api/schemas.py.
 * All price/PnL fields are strings (Decimal precision preserved from backend).
 */
export interface OpenPositionMTM {
  symbol: string;
  /** Decimal string — e.g. "0.01234567" */
  quantity: string;
  /** Entry price as a Decimal string — e.g. "29500.00" */
  entryPrice: string;
  /** Last bar close price as a Decimal string — e.g. "31200.00" */
  lastPrice: string;
  /** Unrealised PnL = (lastPrice − entryPrice) × quantity, as a Decimal string */
  unrealisedPnl: string;
  /** ISO-8601 datetime when the position was opened */
  openedAt: string;
}

/**
 * Backtest performance metrics returned in RunDetailResponse.
 * Mirrors BacktestMetricsResponse in apps/api/schemas.py.
 * All percentage/ratio fields are decimal fractions (0.12 = 12%).
 * Monetary fields are strings for Decimal precision.
 */
export interface BacktestMetrics {
  totalReturnPct: number;
  cagr: number;
  initialCapital: string;
  finalEquity: string;
  totalFeesPaid: string;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdownPct: number;
  maxDrawdownDurationBars: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  /**
   * Gross profit / gross loss.
   * null  → no trades executed (profitFactorIsInfinite=false) OR all winners, zero losses (profitFactorIsInfinite=true).
   * 0.0   → all trades were losers (well-defined; render normally).
   * >0.0  → normal profitable/unprofitable ratio.
   */
  profitFactor: number | null;
  profitFactorIsInfinite: boolean;
  averageTradePnl: string;
  averageWin: string;
  averageLoss: string;
  largestWin: string;
  largestLoss: string;
  totalBars: number;
  barsInMarket: number;
  exposurePct: number;
  exposurePctPerSymbol?: Record<string, number>;
  startDate: string;
  endDate: string;
  durationDays: number;
  /**
   * Positions still open at the end of the backtest, valued at the final bar's
   * close price (mark-to-market). Empty list when all positions closed cleanly.
   * Added by M3 backend (Sprint 49). Optional with `?` so pre-M3 run records
   * that lack the field deserialise without errors.
   */
  openPositionsMtm?: readonly OpenPositionMTM[];
  /**
   * Probabilistic Sharpe Ratio per Bailey & López de Prado (2012).
   * Probability in [0, 1] that the true Sharpe exceeds zero.
   * null when fewer than 30 return observations.
   * Added by M4 backend (Sprint 49). Optional with `?` so pre-M4 run
   * records that lack the field deserialise without errors.
   */
  psr?: number | null;
  /**
   * Number of per-period return observations used to compute PSR.
   * Equals len(equity_curve) - 1.
   */
  nObservations?: number;
  /**
   * Three-tier confidence label derived from PSR and trade count.
   * "high" | "medium" | "low" | null (when PSR not computable).
   */
  confidenceFlag?: "high" | "medium" | "low" | null;
  /**
   * Quote currency detected from trade symbols ("USDT", "EUR", "MIXED" if heterogeneous, null pre-M6).
   * "MIXED" indicates heterogeneous quote currencies — PnL figures are not
   * directly comparable without FX conversion (M6b).
   */
  quoteCurrency?: string | null;
  /**
   * Reporting currency for displayed monetary values.
   * null = native quote, no FX conversion applied (M6 default).
   * M6b will populate "USD" for mixed runs after conversion.
   */
  reportingCurrency?: string | null;
  /**
   * Resolved RNG seed used for this backtest run.
   * Re-submit the run config with this seed to reproduce identical results.
   * null for pre-M7 records. Added by M7 backend (Sprint 49).
   */
  seed?: number | null;
}

export interface RunListResponse {
  total: number;
  offset: number;
  limit: number;
  items: readonly Run[];
}

export interface RunCreateRequest {
  strategyName: string;
  strategyParams: Record<string, unknown>;
  symbols: string[];
  timeframe: string;
  mode: RunMode;
  initialCapital: string;
  backtestStart?: string | null;
  backtestEnd?: string | null;
  /**
   * WP1.3a (SY-13a-08, CF-13a-1 item 4): `StrictBool` on the wire — never
   * send a truthy/falsy non-boolean. An explicit value always wins over the
   * strategy's `default_allow_pyramiding`. The UI always sends this
   * explicitly (never omitted) so the resolved value the operator saw in
   * the form is exactly what the backend persists:
   *   - live mode: always `false` (forced, not editable in the form).
   *   - paper/backtest: the checkbox value, which itself defaults to the
   *     strategy's own default (see `strategyDefaultAllowPyramiding` in
   *     `./exit-config`).
   */
  allowPyramiding?: boolean;
  /**
   * WP1.7a/SY-10 (S13): deprecated body-field fallback only. The UI never
   * populates this — a live-mode confirmation is sent EXCLUSIVELY via the
   * `X-Live-Confirm-Token` request header (see `createRun()` in `./api`),
   * so real callers pass the token as a `createRun(body, token)` argument,
   * never as part of this object. Kept typed (rather than removed) purely
   * so any not-yet-migrated caller/fixture that still sets it continues to
   * compile; the backend accepts the header only from this UI going forward.
   */
  confirmToken?: string | undefined;
}

// ---------------------------------------------------------------------------
// WP1.3a: exit-config / pyramiding validator error envelope + warnings
// -----------------------------------------------------------------------
// Mirrors packages/trading/exit_config.py's ExitConfigIssue/ExitConfigWarning
// dataclasses and apps/api's 422 `{"detail": {...}}` envelope for create,
// promote-to-live, resume and optimize (reports/vp2-wp1.3a/synthesis-spec.md
// §5/§18). These raw-dict `detail` bodies are NOT pydantic models (same
// precedent as `FlattenDecisionRequiredDetail.held_symbols` above), so their
// fields stay snake_case on the wire even though this project's pydantic
// response MODELS use camelCase — do not "fix" these to camelCase.
// ---------------------------------------------------------------------------

/** One field-level validation issue from the exit-config/pyramiding validator. */
export interface ExitConfigIssue {
  field: string | null;
  reason: string;
  /** Stringified, truncated to <=64 chars server-side. */
  value: string | null;
  min: number | null;
  max: number | null;
  message: string;
}

/** One non-blocking warning (W1-W5, W7, W8) returned alongside a 2xx. */
export interface ConfigWarning {
  code: string;
  field: string | null;
  message: string;
}

export type ExitConfigErrorCode =
  | "invalid_exit_config"
  | "exit_manager_required"
  | "live_pyramiding_forbidden";

/** 422 `detail` when a bracket/trailing value fails validation (E1-E4/E6/E7/E9/E10). */
export interface InvalidExitConfigDetail {
  code: "invalid_exit_config";
  errors: readonly ExitConfigIssue[];
  warnings?: readonly ConfigWarning[];
  hint?: string;
}

/** 422 `detail` when a `requires_exit_manager=True` strategy has no downside exit (E5). */
export interface ExitManagerRequiredDetail {
  code: "exit_manager_required";
  strategy: string;
  requires_one_of: readonly string[];
  errors: readonly ExitConfigIssue[];
  warnings?: readonly ConfigWarning[];
  hint?: string;
}

/** 422 `detail` for a resolved-True `allowPyramiding` in LIVE (E11, D-13a-1). */
export interface LivePyramidingForbiddenDetail {
  code: "live_pyramiding_forbidden";
  strategy: string;
  hint?: string;
  errors: readonly ExitConfigIssue[];
  warnings?: readonly ConfigWarning[];
}

/** Union of every structured 422 body create/promote/resume can return (SY-13a-18). */
export type ExitConfigErrorDetail =
  | InvalidExitConfigDetail
  | ExitManagerRequiredDetail
  | LivePyramidingForbiddenDetail;

/**
 * One grid-combination-scoped issue from `POST /api/v1/optimize`'s 422
 * (WP13a-C-01, round 2): the backend emits EXACTLY this flat shape --
 * `{combo_index, params, field, reason, value, min, max, message}`, no more
 * no less. Written out explicitly here (not `extends ExitConfigIssue`,
 * even though the field set is identical) so this contract is unambiguous
 * at a glance and does not silently drift if the base `ExitConfigIssue`
 * ever grows a field the optimize envelope doesn't carry.
 */
export interface OptimizeExitConfigIssue {
  combo_index: number;
  params: Record<string, unknown>;
  field: string | null;
  reason: string;
  /** Stringified, truncated to <=64 chars server-side. */
  value: string | null;
  min: number | null;
  max: number | null;
  message: string;
}

/**
 * `POST /api/v1/optimize` 422 `detail` — same top-level codes as create/
 * promote/resume, but `errors[]` items are combination-scoped (`combo_index`,
 * `params`) and the list is capped at 20, with the top-level `total_invalid`
 * always present (WP13a-C-01) giving the real count (SY-13a-17/§5).
 */
export interface OptimizeExitConfigErrorDetail {
  code: "invalid_exit_config" | "exit_manager_required";
  errors: readonly OptimizeExitConfigIssue[];
  total_invalid: number;
  warnings?: readonly ConfigWarning[];
}

/** `exitConfigWaived` on a protective-resume `Run` (SY-13a-16). */
export interface ExitConfigWaived {
  code: string;
  errors: readonly ExitConfigIssue[];
}

// ---------------------------------------------------------------------------
// WP1.7a/1.7b: kill switch, per-run entries latch, flatten result envelopes
// -----------------------------------------------------------------------
// Mirrors apps/api/routers/emergency.py (KillSwitch*) and the FlattenResult /
// UnprotectedPosition / RunStopResponse / RunEmergencyStopResponse models in
// apps/api/schemas.py -- every one of those shares the project's camelCase
// `API_MODEL_CONFIG`, so every field below is camelCase on the wire too.
// ---------------------------------------------------------------------------

export type FlattenSymbolStatus =
  | "no_position"
  | "flat"
  | "dust"
  | "partial"
  | "in_flight"
  | "failed";

export type FlattenCause =
  | "ledger_doubt"
  | "inflight_other"
  | "submit_unknown"
  | "rejected"
  | "timeout_open"
  | "lock_timeout"
  | "live_gate_closed"
  | "error";

/** One symbol's outcome from a single StrategyEngine.flatten() call. */
export interface FlattenSymbolResult {
  symbol: string;
  status: FlattenSymbolStatus;
  cause: FlattenCause | null;
  /** Decimal strings — precision preserved from the backend. */
  heldBefore: string;
  soldQty: string;
  remainingQty: string;
  orderIds: readonly string[];
  error: string | null;
}

export type FlattenOutcome = "noop" | "flattened" | "partial" | "failed";

/** Run-level result from a single StrategyEngine.flatten() call. */
export interface FlattenResult {
  runId: string;
  outcome: FlattenOutcome;
  complete: boolean;
  symbols: readonly FlattenSymbolResult[];
  latchPersisted: boolean;
}

/** One still-held symbol reported by emergency-stop / stop (SY-07). */
export interface UnprotectedPosition {
  symbol: string;
  /** Decimal string. */
  qty: string;
  source: "ledger" | "persisted";
}

/** DELETE /api/v1/runs/{id} 200 response. */
export interface RunStopResponse extends Run {
  flatten: FlattenResult | null;
  unprotectedPositions: readonly UnprotectedPosition[];
}

/** POST /api/v1/runs/{id}/emergency-stop 200 response. */
export interface RunEmergencyStopResponse extends Run {
  flatten: FlattenResult | null;
  unprotectedPositions: readonly UnprotectedPosition[];
  exposureUnknown: boolean;
}

/**
 * Error `detail` bodies for stop_run's 422/409s are raw dicts constructed
 * directly in apps/api/routers/runs.py (NOT pydantic models), so — unlike
 * every other shape on this page — they bypass `API_MODEL_CONFIG`'s
 * alias_generator and stay snake_case on the wire. `flatten` is the one
 * exception: it is a `FlattenResultResponse.model_dump(by_alias=True)`
 * embedded inside the raw dict, so ITS OWN fields are still camelCase.
 */
export interface FlattenDecisionRequiredDetail {
  code: "flatten_decision_required";
  held_symbols: readonly string[];
}

export interface FlattenIncompleteDetail {
  code: "flatten_incomplete";
  flatten: FlattenResult;
}

export interface FlattenRequiresRunningEngineDetail {
  code: "flatten_requires_running_engine";
}

export interface KillSwitchActiveDetail {
  code: "kill_switch_active";
}

export interface EntriesLatchedDetail {
  code: "entries_latched";
  reason: string;
}

export interface NotLatchedDetail {
  code: "not_latched";
}

/** GET /api/v1/emergency/kill-switch — read-only status for the UI badge. */
export interface KillSwitchStatus {
  latched: boolean;
  since: string | null;
  reason: string | null;
  /** "unknown" -- the latch could not be read at boot (fail-closed, latched). */
  source: "db" | "unknown";
}

export interface KillSwitchRunError {
  runId: string;
  error: string;
}

/** POST /api/v1/emergency/kill-switch 200 response (WP1.7a shape, SY-08). */
export interface KillSwitchPressResponse {
  latched: boolean;
  latchPersisted: boolean;
  since: string | null;
  runsLatched: readonly string[];
  orphanedLiveRunIds: readonly string[];
  resumingRunIds: readonly string[];
  flattenResults: Readonly<Record<string, FlattenResult>>;
  errors: readonly KillSwitchRunError[];
}

export interface KillSwitchRunKeptLatched {
  runId: string;
  reasons: readonly string[];
}

/** POST /api/v1/emergency/kill-switch/clear 200 response. */
export interface KillSwitchClearResponse {
  wasLatched: boolean;
  runsUnlatched: readonly string[];
  runsKeptLatched: readonly KillSwitchRunKeptLatched[];
}

/**
 * POST /api/v1/runs/{id}/entries-latch/clear 200 response.
 *
 * CF-B2 (shipped in this same WP) gave `clear_entries_latch`
 * (apps/api/routers/runs.py) a proper `API_MODEL_CONFIG` response model --
 * camelCase on the wire, matching this type field-for-field. `cleared` is
 * typed as the literal `"flatten_incomplete"` (WP17b-C-03) rather than a
 * plain `string`: the DB's own `ck_runs_entries_latch_reason` CHECK
 * constraint (apps/api/db/models.py) means this per-run latch reason can
 * only ever BE `'flatten_incomplete'` -- the backend's own
 * `EntriesLatchClearResponse.cleared: str` (apps/api/schemas.py) is looser
 * only because Pydantic has no equivalent DB-level constraint to mirror;
 * the UI can and should be stricter here for a stronger compile-time
 * guarantee against a typo'd comparison.
 */
export interface EntriesLatchClearResponse {
  runId: string;
  cleared: "flatten_incomplete";
  stillLatchedBy: readonly string[];
}

// ---------------------------------------------------------------------------
// Orders
// ---------------------------------------------------------------------------

export type OrderSide = "buy" | "sell";
export type OrderType = "market" | "limit" | "stop_limit" | "stop_market";
export type OrderStatus =
  | "new"
  | "pending_submit"
  | "open"
  | "partial"
  | "filled"
  | "canceled"
  | "rejected"
  | "expired";

export interface Order {
  id: string;
  clientOrderId: string;
  runId: string;
  symbol: string;
  side: OrderSide;
  orderType: OrderType;
  quantity: string;
  price: string | null;
  status: OrderStatus;
  filledQuantity: string;
  averageFillPrice: string | null;
  exchangeOrderId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface OrderListResponse {
  total: number;
  offset: number;
  limit: number;
  items: readonly Order[];
}

// ---------------------------------------------------------------------------
// Fills
// ---------------------------------------------------------------------------

export interface Fill {
  id: string;
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: string;
  price: string;
  fee: string;
  feeCurrency: string;
  isMaker: boolean;
  executedAt: string;
}

export interface FillListResponse {
  total: number;
  offset: number;
  limit: number;
  items: readonly Fill[];
}

// ---------------------------------------------------------------------------
// Trades
// ---------------------------------------------------------------------------

export interface Trade {
  id: string;
  runId: string;
  symbol: string;
  side: OrderSide;
  entryPrice: string;
  exitPrice: string;
  quantity: string;
  realisedPnl: string;
  totalFees: string;
  entryAt: string;
  exitAt: string;
  strategyId: string;
}

export interface TradeListResponse {
  total: number;
  offset: number;
  limit: number;
  items: readonly Trade[];
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

export interface Portfolio {
  runId: string;
  initialCash: string;
  currentCash: string;
  currentEquity: string;
  peakEquity: string;
  totalReturnPct: number;
  totalRealisedPnl: string;
  totalFeesPaid: string;
  dailyPnl: string;
  drawdownPct: number;
  maxDrawdownPct: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  openPositions: number;
  equityCurveLength: number;
}

export interface AggregatePortfolio {
  totalRuns: number;
  runningRuns: number;
  stoppedRuns: number;
  errorRuns: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalRealisedPnl: string;
  totalFeesPaid: string;
  bestRunReturnPct: number | null;
  worstRunReturnPct: number | null;
  totalInitialCapital: string;
  /**
   * Number of runs that meet leaderboard eligibility criteria
   * (n_closed_trades >= 10 AND confidence_flag IN ("high", "medium")).
   * Used by the home-page "Best Return (eligible)" card subtitle.
   * Added by M5 backend (Sprint 49). Optional with `?` for backward compat.
   */
  eligibleRunsCount?: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: string;
  cash: string;
  unrealisedPnl: string;
  realisedPnl: string;
  drawdownPct: number;
  barIndex: number;
}

export interface EquityCurveResponse {
  runId: string;
  totalPoints: number;
  points: readonly EquityPoint[];
}

export interface Position {
  symbol: string;
  runId: string;
  quantity: string;
  averageEntryPrice: string;
  currentPrice: string;
  realisedPnl: string;
  unrealisedPnl: string;
  totalFeesPaid: string;
  notionalValue: string;
  openedAt: string;
  updatedAt: string;
}

export interface PositionListResponse {
  runId: string;
  positions: readonly Position[];
  count: number;
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

export interface JsonSchemaProperty {
  /** Scalar type string (normalised backend form). */
  type?: "string" | "integer" | "number" | "boolean";
  /** Set by backend normaliser when the field is ``float | None``. */
  nullable?: boolean;
  /**
   * Raw Pydantic v2 anyOf form — present on un-normalised schemas or if a
   * future Pydantic version changes its emission shape.  Frontend resolvers
   * must handle this as a fallback.
   */
  anyOf?: Array<{ type: string; minimum?: number; maximum?: number }>;
  description?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  enum?: unknown[];
}

export interface JsonSchema {
  type: "object";
  title?: string;
  description?: string;
  properties: Record<string, JsonSchemaProperty>;
  required?: readonly string[];
  additionalProperties?: boolean;
}

/**
 * Promotion lifecycle status of a strategy (Sprint 51 Cycle 2).
 * - "active": cleared for all run modes (backtest, paper, live).
 * - "demoted": restricted to backtest only pending re-promotion.
 * - "experimental": newly added / under evaluation.
 */
export type StrategyStatus = "active" | "demoted" | "experimental";

export interface Strategy {
  name: string;
  displayName: string;
  version: string;
  description: string;
  tags: readonly string[];
  parameterSchema: JsonSchema;
  /**
   * Run modes this strategy may be launched in. Subset of RunMode.
   * OPTIONAL for back-compat with older API responses — when absent the UI
   * treats the strategy as allowing all three modes.
   */
  allowedModes?: readonly RunMode[];
  /** Promotion lifecycle status. Absent → treat as "active". */
  status?: StrategyStatus;
  /** Human-readable reason a strategy was demoted; null/absent when not demoted. */
  demotionReason?: string | null;
  /** Criteria that must be met before the strategy can be re-promoted. */
  promotionRequirements?: readonly string[];
  /**
   * WP1.3a (SY-13a-08): mirrors `strategy_cls.default_allow_pyramiding`
   * (`True` only for dca_rsi_hybrid/grid_trading, `False` on every other
   * registry strategy) IF the backend schema endpoint exposes it. This is
   * NOT part of the WP1.3a backend API contract (§4/§5 only add it as an
   * internal ClassVar) — absent on every API response as of this WP, so the
   * UI falls back to `PYRAMIDING_DEFAULT_TRUE_STRATEGIES` in
   * `./exit-config`. Typed here (optional) purely so a future backend
   * change that DOES expose it is picked up for free, with no UI change.
   */
  defaultAllowPyramiding?: boolean;
}

export interface StrategyListResponse {
  strategies: readonly Strategy[];
  total: number;
}

// ---------------------------------------------------------------------------
// ML Model Versions
// ---------------------------------------------------------------------------

/**
 * A single trained model version persisted by the ML training pipeline.
 * Mirrors ModelVersionResponse in apps/api/schemas.py.
 *
 * - accuracy is a decimal fraction (0.62 = 62 %).
 * - trainedAt is an ISO-8601 datetime string.
 */
export interface ModelVersion {
  /** UUID of the model version record. */
  id: string;
  /** Trading pair this model was trained on, e.g. "BTC/USD". */
  symbol: string;
  /** OHLCV timeframe used during training, e.g. "1h". */
  timeframe: string;
  /** ISO-8601 datetime when training completed. */
  trainedAt: string;
  /** Held-out accuracy as a decimal fraction (0.0 – 1.0). */
  accuracy: number;
  /** Number of completed trades used as training labels. */
  nTradesUsed: number;
  /** Number of OHLCV bars consumed during feature extraction. */
  nBarsUsed: number;
  /** Label generation method: "horizon" (fixed-lookahead) or "pnl" (trade PnL sign). */
  labelMethod: string;
  /** How training was initiated: "manual" (API/CLI) or "auto" (scheduled). */
  trigger: string;
  /** Filesystem path to the serialised model artefact. */
  modelPath: string;
  /** Whether this version is the active model used by ModelStrategy. */
  isActive: boolean;
  /** Arbitrary extra metadata stored by the trainer (precision, recall, etc.). */
  extra: Record<string, unknown> | null;
}

/** Response envelope for GET /api/v1/ml/models. */
export interface ModelVersionListResponse {
  items: readonly ModelVersion[];
  total: number;
}

// ---------------------------------------------------------------------------
// Parameter Optimization
// ---------------------------------------------------------------------------

export interface OptimizeRequest {
  strategyName: string;
  paramGrid: Record<string, unknown[]>;
  symbols: string[];
  timeframe: string;
  backtestStart: string;
  backtestEnd: string;
  initialCapital?: string;
  rankBy?: string;
  topN?: number;
  maxCombinations?: number;
}

export interface OptimizeEntry {
  rank: number;
  /** Parameter values for this combination (snake_case keys). */
  params: Record<string, unknown>;
  /**
   * Metric values keyed by snake_case name, e.g. "sharpe_ratio".
   * Note: these keys are NOT camelCased — the backend passes them through
   * as raw dict values, bypassing Pydantic's alias_generator.
   */
  metrics: Record<string, number>;
}

export interface OptimizeResponse {
  strategyName: string;
  symbols: readonly string[];
  timeframe: string;
  rankBy: string;
  totalCombinations: number;
  completedCombinations: number;
  failedCombinations: number;
  elapsedSeconds: number;
  entries: readonly OptimizeEntry[];
  /**
   * UUID of the persisted OptimizationRun record.
   * Optional until backend always returns this field (Sprint 31 backend persistence).
   * TODO(sprint-32): promote to required once GET /api/v1/optimize/{id} is deployed.
   * set after backend persistence is wired (Sprint 31)
   */
  optimizationRunId?: string;
}

// ---------------------------------------------------------------------------
// Optimization Run History
// ---------------------------------------------------------------------------

/**
 * A saved optimization run summary returned by GET /api/v1/optimize.
 * Mirrors OptimizationRunSummaryResponse in apps/api/schemas.py.
 *
 * The `entries` field is NOT included in the list response — only the full
 * detail response (GET /api/v1/optimize/{id}) carries the entries array.
 */
export interface OptimizationRunSummary {
  /** UUID of the saved optimization run (API returns as optimizationRunId). */
  optimizationRunId: string;
  /** Strategy name used in this optimization, e.g. "ma_crossover". */
  strategyName: string;
  /** OHLCV timeframe used, e.g. "1h". */
  timeframe: string;
  /** Trading symbols included in the backtest, e.g. ["BTC/USD"]. */
  symbols: readonly string[];
  /** Metric used to rank parameter combinations, e.g. "sharpe_ratio". */
  rankBy: string;
  /** Total number of parameter combinations attempted. */
  totalCombinations: number;
  /** Number of combinations that completed successfully. */
  completedCombinations: number;
  /** Number of combinations that failed (exception during backtest). */
  failedCombinations: number;
  /** Wall-clock seconds elapsed for the full grid search. */
  elapsedSeconds: number;
  /** ISO-8601 datetime when this optimization run was persisted. */
  createdAt: string;
}

/** Response envelope for GET /api/v1/optimize (paginated list). */
export interface OptimizationRunListResponse {
  items: readonly OptimizationRunSummary[];
  total: number;
  offset: number;
  limit: number;
}

// ---------------------------------------------------------------------------
// Adaptive Learning State
// ---------------------------------------------------------------------------

export interface ParameterChange {
  paramName: string;
  oldValue: unknown;
  newValue: unknown;
  changePct: number;
}

export interface LearningAdjustment {
  actionable: boolean;
  confidence: number;
  reason: string;
  changes: readonly ParameterChange[];
}

export interface LearningAnalysis {
  confidence: number;
  isActionable: boolean;
  totalTrades: number;
  totalSkipped: number;
  bestRegime?: string | null;
  worstRegime?: string | null;
  mostPredictiveIndicator?: string | null;
}

export interface OptimizerStateSummary {
  isEnabled: boolean;
  rollbackCount30d: number;
  cooldownUntil: string | null;
  disabledReason: string | null;
  preAdjustmentPnlPct: number | null;
}

export interface LearningState {
  enabled: boolean;
  autoApply: boolean;
  cycleCount: number;
  tradesIngested: number;
  skippedIngested: number;
  tradesAtLastCycle: number;
  minTradesPerCycle: number;
  optimizerState: OptimizerStateSummary;
  lastAdjustment: LearningAdjustment | null;
  lastAnalysis: LearningAnalysis | null;
}
