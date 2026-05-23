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
export type RunStatus = "running" | "stopped" | "error" | "archived";

export interface RunConfig {
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  symbols: readonly string[];
  timeframe: string;
  mode: RunMode;
  initial_capital: string;
  backtest_start?: string;
  backtest_end?: string;
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
  confirmToken?: string | undefined;
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

export interface Strategy {
  name: string;
  displayName: string;
  version: string;
  description: string;
  tags: readonly string[];
  parameterSchema: JsonSchema;
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
