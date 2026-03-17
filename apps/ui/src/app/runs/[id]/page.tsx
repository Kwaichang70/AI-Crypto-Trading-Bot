"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  fetchRun,
  fetchPortfolio,
  fetchEquityCurve,
  fetchTrades,
  fetchOrders,
  fetchFills,
  fetchPositions,
  stopRun,
  formatCurrency,
  formatPct,
} from "@/lib/api";
import type { Run, Portfolio, EquityPoint, Trade, Order, Fill, Position } from "@/lib/types";
import type { CsvColumn } from "@/lib/csv-export";
import { ExportCsvButton } from "@/components/ui/export-csv-button";
import { Header } from "@/components/layout/header";
import { RunStatusBadge } from "@/components/ui/status-badge";
import { StatCard } from "@/components/ui/stat-card";
import { DataTable, type Column } from "@/components/ui/data-table";
import { Tabs } from "@/components/ui/tabs";
import { EquityCurveChart } from "@/components/charts/equity-curve";

// ---------------------------------------------------------------------------
// Trade columns
// ---------------------------------------------------------------------------

const TRADE_COLUMNS: Column<Trade>[] = [
  {
    key: "symbol",
    header: "Symbol",
    render: (t) => <span className="font-mono text-xs text-slate-300">{t.symbol}</span>,
  },
  {
    key: "side",
    header: "Side",
    render: (t) => (
      <span className={t.side === "buy" ? "text-profit text-xs font-medium" : "text-loss text-xs font-medium"}>
        {t.side.toUpperCase()}
      </span>
    ),
  },
  {
    key: "entryPrice",
    header: "Entry",
    render: (t) => <span className="font-mono text-xs">{formatCurrency(t.entryPrice)}</span>,
  },
  {
    key: "exitPrice",
    header: "Exit",
    render: (t) => <span className="font-mono text-xs">{formatCurrency(t.exitPrice)}</span>,
  },
  {
    key: "quantity",
    header: "Qty",
    render: (t) => <span className="font-mono text-xs">{t.quantity}</span>,
  },
  {
    key: "realisedPnl",
    header: "PnL",
    sortable: true,
    sortValue: (t) => parseFloat(t.realisedPnl),
    render: (t) => {
      const pnl = parseFloat(t.realisedPnl);
      return (
        <span className={`font-mono text-xs font-medium ${pnl >= 0 ? "text-profit" : "text-loss"}`}>
          {pnl >= 0 ? "+" : ""}{formatCurrency(t.realisedPnl)}
        </span>
      );
    },
  },
  {
    key: "exitAt",
    header: "Closed",
    sortable: true,
    sortValue: (t) => new Date(t.exitAt).getTime(),
    render: (t) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(t.exitAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Order columns
// ---------------------------------------------------------------------------

const ORDER_COLUMNS: Column<Order>[] = [
  {
    key: "symbol",
    header: "Symbol",
    render: (o) => <span className="font-mono text-xs text-slate-300">{o.symbol}</span>,
  },
  {
    key: "side",
    header: "Side",
    render: (o) => (
      <span className={o.side === "buy" ? "text-profit text-xs font-medium" : "text-loss text-xs font-medium"}>
        {o.side.toUpperCase()}
      </span>
    ),
  },
  {
    key: "orderType",
    header: "Type",
    render: (o) => <span className="text-xs text-slate-400">{o.orderType}</span>,
  },
  {
    key: "quantity",
    header: "Qty",
    render: (o) => <span className="font-mono text-xs">{o.quantity}</span>,
  },
  {
    key: "price",
    header: "Price",
    render: (o) => <span className="font-mono text-xs">{o.price ? formatCurrency(o.price) : "market"}</span>,
  },
  {
    key: "status",
    header: "Status",
    sortable: true,
    sortValue: (o) => o.status,
    render: (o) => <RunStatusBadge status={o.status} />,
  },
  {
    key: "createdAt",
    header: "Created",
    sortable: true,
    sortValue: (o) => new Date(o.createdAt).getTime(),
    render: (o) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(o.createdAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Fill columns
// ---------------------------------------------------------------------------

const FILL_COLUMNS: Column<Fill>[] = [
  {
    key: "symbol",
    header: "Symbol",
    render: (f) => <span className="font-mono text-xs text-slate-300">{f.symbol}</span>,
  },
  {
    key: "side",
    header: "Side",
    render: (f) => (
      <span className={f.side === "buy" ? "text-profit text-xs font-medium" : "text-loss text-xs font-medium"}>
        {f.side.toUpperCase()}
      </span>
    ),
  },
  {
    key: "quantity",
    header: "Qty",
    render: (f) => <span className="font-mono text-xs">{f.quantity}</span>,
  },
  {
    key: "price",
    header: "Price",
    sortable: true,
    sortValue: (f) => parseFloat(f.price),
    render: (f) => <span className="font-mono text-xs">{formatCurrency(f.price)}</span>,
  },
  {
    key: "fee",
    header: "Fee",
    render: (f) => (
      <span className="font-mono text-xs text-slate-400">
        {formatCurrency(f.fee)} {f.feeCurrency || ""}
      </span>
    ),
  },
  {
    key: "isMaker",
    header: "Maker",
    render: (f) => <span className="text-xs text-slate-400">{f.isMaker ? "Yes" : "No"}</span>,
  },
  {
    key: "executedAt",
    header: "Executed",
    sortable: true,
    sortValue: (f) => new Date(f.executedAt).getTime(),
    render: (f) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(f.executedAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Position columns
// ---------------------------------------------------------------------------

const POSITION_COLUMNS: Column<Position>[] = [
  {
    key: "symbol",
    header: "Symbol",
    render: (p) => <span className="font-mono text-xs text-slate-300">{p.symbol}</span>,
  },
  {
    key: "quantity",
    header: "Quantity",
    render: (p) => <span className="font-mono text-xs">{p.quantity}</span>,
  },
  {
    key: "averageEntryPrice",
    header: "Avg Entry",
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.averageEntryPrice)}</span>,
  },
  {
    key: "currentPrice",
    header: "Current Price",
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.currentPrice)}</span>,
  },
  {
    key: "unrealisedPnl",
    header: "Unrealised PnL",
    sortable: true,
    sortValue: (p) => parseFloat(p.unrealisedPnl),
    render: (p) => {
      const pnl = parseFloat(p.unrealisedPnl);
      return (
        <span className={`font-mono text-xs font-medium ${pnl >= 0 ? "text-profit" : "text-loss"}`}>
          {pnl >= 0 ? "+" : ""}{formatCurrency(p.unrealisedPnl)}
        </span>
      );
    },
  },
  {
    key: "notionalValue",
    header: "Notional Value",
    sortable: true,
    sortValue: (p) => parseFloat(p.notionalValue),
    render: (p) => <span className="font-mono text-xs">{formatCurrency(p.notionalValue)}</span>,
  },
  {
    key: "openedAt",
    header: "Opened At",
    sortable: true,
    sortValue: (p) => new Date(p.openedAt).getTime(),
    render: (p) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(p.openedAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// CSV column specs — plain scalar values for export
// ---------------------------------------------------------------------------

const TRADE_CSV_COLUMNS: CsvColumn<Trade>[] = [
  { header: "Symbol", value: (t) => t.symbol },
  { header: "Side", value: (t) => t.side },
  { header: "Entry Price", value: (t) => t.entryPrice },
  { header: "Exit Price", value: (t) => t.exitPrice },
  { header: "Quantity", value: (t) => t.quantity },
  { header: "PnL", value: (t) => t.realisedPnl },
  { header: "Total Fees", value: (t) => t.totalFees },
  { header: "Entry At", value: (t) => t.entryAt },
  { header: "Exit At", value: (t) => t.exitAt },
];

const ORDER_CSV_COLUMNS: CsvColumn<Order>[] = [
  { header: "Symbol", value: (o) => o.symbol },
  { header: "Side", value: (o) => o.side },
  { header: "Type", value: (o) => o.orderType },
  { header: "Quantity", value: (o) => o.quantity },
  { header: "Price", value: (o) => o.price ?? "" },
  { header: "Status", value: (o) => o.status },
  { header: "Filled Qty", value: (o) => o.filledQuantity },
  { header: "Avg Fill Price", value: (o) => o.averageFillPrice ?? "" },
  { header: "Created At", value: (o) => o.createdAt },
];

const FILL_CSV_COLUMNS: CsvColumn<Fill>[] = [
  { header: "Symbol", value: (f) => f.symbol },
  { header: "Side", value: (f) => f.side },
  { header: "Quantity", value: (f) => f.quantity },
  { header: "Price", value: (f) => f.price },
  { header: "Fee", value: (f) => f.fee },
  { header: "Fee Currency", value: (f) => f.feeCurrency },
  { header: "Maker", value: (f) => (f.isMaker ? "Yes" : "No") },
  { header: "Executed At", value: (f) => f.executedAt },
];

const POSITION_CSV_COLUMNS: CsvColumn<Position>[] = [
  { header: "Symbol", value: (p) => p.symbol },
  { header: "Quantity", value: (p) => p.quantity },
  { header: "Avg Entry Price", value: (p) => p.averageEntryPrice },
  { header: "Current Price", value: (p) => p.currentPrice },
  { header: "Unrealised PnL", value: (p) => p.unrealisedPnl },
  { header: "Notional Value", value: (p) => p.notionalValue },
  { header: "Opened At", value: (p) => p.openedAt },
];

const EQUITY_CSV_COLUMNS: CsvColumn<EquityPoint>[] = [
  { header: "Timestamp", value: (e) => e.timestamp },
  { header: "Equity", value: (e) => e.equity },
  { header: "Cash", value: (e) => e.cash },
  { header: "Unrealised PnL", value: (e) => e.unrealisedPnl },
  { header: "Realised PnL", value: (e) => e.realisedPnl },
  { header: "Drawdown %", value: (e) => (e.drawdownPct * 100).toFixed(4) },
  { header: "Bar Index", value: (e) => e.barIndex },
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [run, setRun] = useState<Run | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [equityPoints, setEquityPoints] = useState<readonly EquityPoint[]>([]);
  const [trades, setTrades] = useState<readonly Trade[]>([]);
  const [orders, setOrders] = useState<readonly Order[]>([]);
  const [fills, setFills] = useState<readonly Fill[]>([]);
  const [positions, setPositions] = useState<readonly Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isStopping, setIsStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    const [runRes, portRes, curveRes, tradesRes, ordersRes, fillsRes, posRes] = await Promise.all([
      fetchRun(id),
      fetchPortfolio(id),
      fetchEquityCurve(id, 500),
      fetchTrades(id, { limit: 100 }),
      fetchOrders(id, { limit: 100 }),
      fetchFills(id, { limit: 100 }),
      fetchPositions(id),
    ]);

    if (!runRes.ok) {
      setError(runRes.error.message);
      return;
    }

    setRun(runRes.data);
    if (portRes.ok) setPortfolio(portRes.data);
    if (curveRes.ok) setEquityPoints(curveRes.data.points);
    if (tradesRes.ok) setTrades(tradesRes.data.items);
    if (ordersRes.ok) setOrders(ordersRes.data.items);
    if (fillsRes.ok) setFills(fillsRes.data.items);
    if (posRes.ok) setPositions(posRes.data.positions);
  }, [id]);

  useEffect(() => {
    setIsLoading(true);
    void loadData().finally(() => setIsLoading(false));
  }, [loadData]);

  // Poll every 5 seconds for active runs, pausing when the browser tab is hidden.
  useEffect(() => {
    // Only poll while the run is in an active state.
    if (run?.status !== "running") return;

    const startPolling = () => setInterval(() => void loadData(), 5000);

    let intervalId = startPolling();

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        clearInterval(intervalId);
      } else {
        // Tab became visible — fetch immediately then restart the interval.
        void loadData();
        intervalId = startPolling();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearInterval(intervalId);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [run?.status, loadData]);

  async function handleStop() {
    if (!run) return;
    setIsStopping(true);
    const result = await stopRun(run.id);
    if (result.ok) {
      setRun(result.data);
    } else {
      setError(result.error.message);
    }
    setIsStopping(false);
  }

  function handleDuplicate() {
    if (!run?.config) return;
    const { strategy_name, symbols: configSymbols, timeframe, initial_capital } = run.config;
    const params = new URLSearchParams({
      strategy: strategy_name ?? "",
      symbols: Array.isArray(configSymbols) ? configSymbols.join(",") : "",
      timeframe: timeframe ?? "",
      initial_capital: initial_capital ?? "",
    });
    router.push(`/runs/new?${params.toString()}`);
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-64 animate-pulse rounded bg-slate-800" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 animate-pulse rounded-xl bg-slate-800" />
          ))}
        </div>
      </div>
    );
  }

  if (error ?? !run) {
    return (
      <div className="space-y-4">
        <Link href="/runs" className="text-sm text-indigo-400 hover:underline">
          Back to Runs
        </Link>
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          {error ?? "Run not found."}
        </div>
      </div>
    );
  }

  const strategyName = run.config?.strategy_name ?? "Unknown strategy";
  const symbols = run.config?.symbols?.join(", ") ?? "—";

  const isDone =
    run.status === "stopped" || run.status === "error";

  return (
    <div className="space-y-6">
      {/* Back link */}
      <Link href="/runs" className="text-sm text-slate-500 hover:text-slate-300 hover:underline">
        Back to Runs
      </Link>

      {/* Header */}
      <Header
        title={`Run ${run.id.slice(0, 8)}…`}
        subtitle={`${run.runMode} · ${strategyName} · ${symbols}`}
        actions={
          <div className="flex items-center gap-2">
            {run.status === "running" && (
              <button
                onClick={() => void handleStop()}
                disabled={isStopping}
                className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-2 text-sm font-medium text-red-400 transition-colors hover:bg-red-900/40 disabled:opacity-50"
              >
                {isStopping ? "Stopping…" : "Stop Run"}
              </button>
            )}
            {isDone && (
              <button
                onClick={handleDuplicate}
                className="rounded-lg border border-slate-700 bg-slate-800/60 px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:border-indigo-500 hover:bg-indigo-600/10 hover:text-indigo-300"
              >
                Duplicate Run
              </button>
            )}
          </div>
        }
      />

      {/* Status + metadata row */}
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
        <RunStatusBadge status={run.status} />
        <span>Started {new Date(run.startedAt).toLocaleString()}</span>
        {run.stoppedAt && (
          <span>Stopped {new Date(run.stoppedAt).toLocaleString()}</span>
        )}
      </div>

      {/* Tabs */}
      <Tabs
        tabs={[
          { id: "overview", label: "Overview" },
          { id: "trades", label: `Trades (${trades.length})` },
          { id: "orders", label: `Orders (${orders.length})` },
          { id: "fills", label: `Fills (${fills.length})` },
          { id: "positions", label: `Positions (${positions.length})` },
        ]}
      >
        {(activeTab) => (
          <>
            {activeTab === "overview" && (
              <div className="space-y-6">
                {/* Portfolio metric cards */}
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                  <StatCard
                    label="Current Equity"
                    value={portfolio ? `$${formatCurrency(portfolio.currentEquity)}` : "—"}
                    isLoading={!portfolio}
                  />
                  <StatCard
                    label="Total Return"
                    value={portfolio ? formatPct(portfolio.totalReturnPct) : "—"}
                    trend={
                      portfolio
                        ? portfolio.totalReturnPct >= 0 ? "up" : "down"
                        : "neutral"
                    }
                    isLoading={!portfolio}
                  />
                  <StatCard
                    label="Max Drawdown"
                    value={portfolio ? formatPct(portfolio.maxDrawdownPct) : "—"}
                    trend={portfolio && portfolio.maxDrawdownPct > 0.1 ? "down" : "neutral"}
                    isLoading={!portfolio}
                  />
                  <StatCard
                    label="Win Rate"
                    value={portfolio ? formatPct(portfolio.winRate) : "—"}
                    subValue={portfolio ? `${portfolio.winningTrades}W / ${portfolio.losingTrades}L` : undefined}
                    trend={portfolio && portfolio.winRate >= 0.5 ? "up" : "neutral"}
                    isLoading={!portfolio}
                  />
                  <StatCard
                    label="Total Trades"
                    value={portfolio?.totalTrades ?? "—"}
                    subValue={portfolio ? `$${formatCurrency(portfolio.totalRealisedPnl)} PnL` : undefined}
                    isLoading={!portfolio}
                  />
                </div>

                {/* Equity curve */}
                <div className="card">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-300">Equity Curve</h3>
                    <ExportCsvButton
                      filename={`run-${id.slice(0, 8)}-equity.csv`}
                      columns={EQUITY_CSV_COLUMNS}
                      data={equityPoints}
                    />
                  </div>
                  <EquityCurveChart points={equityPoints} height={260} />
                </div>

                {/* Backtest performance metrics — shown only for completed backtests */}
                {run.runMode === "backtest" && run.backtestMetrics && (
                  <div className="card">
                    <h3 className="mb-3 text-sm font-semibold text-slate-300">Backtest Performance</h3>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                      <StatCard
                        label="Sharpe Ratio"
                        value={run.backtestMetrics.sharpeRatio.toFixed(3)}
                        trend={run.backtestMetrics.sharpeRatio >= 1.0 ? "up" : run.backtestMetrics.sharpeRatio < 0 ? "down" : "neutral"}
                      />
                      <StatCard
                        label="Sortino Ratio"
                        value={run.backtestMetrics.sortinoRatio.toFixed(3)}
                        trend={run.backtestMetrics.sortinoRatio >= 1.5 ? "up" : run.backtestMetrics.sortinoRatio < 0 ? "down" : "neutral"}
                      />
                      <StatCard
                        label="Calmar Ratio"
                        value={run.backtestMetrics.calmarRatio.toFixed(3)}
                        trend={run.backtestMetrics.calmarRatio >= 1.0 ? "up" : "neutral"}
                      />
                      <StatCard
                        label="Profit Factor"
                        value={run.backtestMetrics.profitFactor.toFixed(2)}
                        trend={run.backtestMetrics.profitFactor >= 1.5 ? "up" : run.backtestMetrics.profitFactor < 1.0 ? "down" : "neutral"}
                      />
                      <StatCard
                        label="Exposure"
                        value={formatPct(run.backtestMetrics.exposurePct)}
                        subValue={`${run.backtestMetrics.barsInMarket} / ${run.backtestMetrics.totalBars} bars`}
                      />
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                      <StatCard
                        label="CAGR"
                        value={formatPct(run.backtestMetrics.cagr)}
                        trend={run.backtestMetrics.cagr > 0 ? "up" : run.backtestMetrics.cagr < 0 ? "down" : "neutral"}
                      />
                      <StatCard
                        label="Duration"
                        value={`${run.backtestMetrics.durationDays} days`}
                      />
                      <StatCard
                        label="Avg Trade PnL"
                        value={`$${formatCurrency(run.backtestMetrics.averageTradePnl)}`}
                        trend={parseFloat(run.backtestMetrics.averageTradePnl) >= 0 ? "up" : "down"}
                      />
                      <StatCard
                        label="Largest Win"
                        value={`$${formatCurrency(run.backtestMetrics.largestWin)}`}
                        trend="up"
                      />
                      <StatCard
                        label="Largest Loss"
                        value={`$${formatCurrency(run.backtestMetrics.largestLoss)}`}
                        trend="down"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === "trades" && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <ExportCsvButton
                    filename={`run-${id.slice(0, 8)}-trades.csv`}
                    columns={TRADE_CSV_COLUMNS}
                    data={trades}
                  />
                </div>
                <DataTable
                  columns={TRADE_COLUMNS}
                  data={trades}
                  keyExtractor={(t) => t.id}
                  emptyMessage="No completed trades for this run."
                />
              </div>
            )}

            {activeTab === "orders" && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <ExportCsvButton
                    filename={`run-${id.slice(0, 8)}-orders.csv`}
                    columns={ORDER_CSV_COLUMNS}
                    data={orders}
                  />
                </div>
                <DataTable
                  columns={ORDER_COLUMNS}
                  data={orders}
                  keyExtractor={(o) => o.id}
                  emptyMessage="No orders for this run."
                />
              </div>
            )}

            {activeTab === "fills" && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <ExportCsvButton
                    filename={`run-${id.slice(0, 8)}-fills.csv`}
                    columns={FILL_CSV_COLUMNS}
                    data={fills}
                  />
                </div>
                <DataTable
                  columns={FILL_COLUMNS}
                  data={fills}
                  keyExtractor={(f) => f.id}
                  emptyMessage="No fills for this run."
                />
              </div>
            )}

            {activeTab === "positions" && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <ExportCsvButton
                    filename={`run-${id.slice(0, 8)}-positions.csv`}
                    columns={POSITION_CSV_COLUMNS}
                    data={positions}
                  />
                </div>
                <DataTable
                  columns={POSITION_COLUMNS}
                  data={positions}
                  keyExtractor={(p) => p.symbol}
                  emptyMessage="No open positions for this run."
                />
              </div>
            )}
          </>
        )}
      </Tabs>
    </div>
  );
}
