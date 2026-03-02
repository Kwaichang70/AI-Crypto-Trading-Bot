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
export type RunStatus = "running" | "stopped" | "error";

export interface RunConfig {
  strategyName: string;
  strategyParams: Record<string, unknown>;
  symbols: readonly string[];
  timeframe: string;
  mode: RunMode;
  initialCapital: string;
  backtestStart?: string;
  backtestEnd?: string;
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
  profitFactor: number;
  averageTradePnl: string;
  averageWin: string;
  averageLoss: string;
  largestWin: string;
  largestLoss: string;
  totalBars: number;
  barsInMarket: number;
  exposurePct: number;
  startDate: string;
  endDate: string;
  durationDays: number;
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
  type: "string" | "integer" | "number" | "boolean";
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
