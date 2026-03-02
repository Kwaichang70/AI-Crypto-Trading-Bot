"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  fetchRun,
  fetchPortfolio,
  fetchEquityCurve,
  fetchTrades,
  fetchOrders,
  stopRun,
  formatCurrency,
  formatPct,
} from "@/lib/api";
import type { Run, Portfolio, EquityPoint, Trade, Order } from "@/lib/types";
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
    render: (o) => <RunStatusBadge status={o.status} />,
  },
  {
    key: "createdAt",
    header: "Created",
    render: (o) => (
      <span className="font-mono text-xs text-slate-500">
        {new Date(o.createdAt).toLocaleString()}
      </span>
    ),
  },
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [run, setRun] = useState<Run | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [equityPoints, setEquityPoints] = useState<readonly EquityPoint[]>([]);
  const [trades, setTrades] = useState<readonly Trade[]>([]);
  const [orders, setOrders] = useState<readonly Order[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isStopping, setIsStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    const [runRes, portRes, curveRes, tradesRes, ordersRes] = await Promise.all([
      fetchRun(id),
      fetchPortfolio(id),
      fetchEquityCurve(id, 500),
      fetchTrades(id, { limit: 100 }),
      fetchOrders(id, { limit: 100 }),
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
  }, [id]);

  useEffect(() => {
    setIsLoading(true);
    void loadData().finally(() => setIsLoading(false));
  }, [loadData]);

  // Poll every 5 seconds for running runs
  useEffect(() => {
    if (!run || run.status !== "running") return;
    const interval = setInterval(() => void loadData(), 5000);
    return () => clearInterval(interval);
  }, [run, loadData]);

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

  const strategyName = run.config?.strategyName ?? "Unknown strategy";
  const symbols = run.config?.symbols?.join(", ") ?? "—";

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
          run.status === "running" ? (
            <button
              onClick={() => void handleStop()}
              disabled={isStopping}
              className="rounded-lg border border-red-700 bg-red-900/20 px-4 py-2 text-sm font-medium text-red-400 transition-colors hover:bg-red-900/40 disabled:opacity-50"
            >
              {isStopping ? "Stopping…" : "Stop Run"}
            </button>
          ) : null
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
                  <h3 className="mb-3 text-sm font-semibold text-slate-300">Equity Curve</h3>
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
                  </div>
                )}
              </div>
            )}

            {activeTab === "trades" && (
              <DataTable
                columns={TRADE_COLUMNS}
                data={trades}
                keyExtractor={(t) => t.id}
                emptyMessage="No completed trades for this run."
              />
            )}

            {activeTab === "orders" && (
              <DataTable
                columns={ORDER_COLUMNS}
                data={orders}
                keyExtractor={(o) => o.id}
                emptyMessage="No orders for this run."
              />
            )}
          </>
        )}
      </Tabs>
    </div>
  );
}
