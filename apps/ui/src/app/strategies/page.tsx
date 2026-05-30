/**
 * Strategies Page — Server Component.
 * Lists all available trading strategies with parameter schema detail.
 */

import type { Metadata } from "next";
import { fetchStrategies } from "@/lib/api";
import { Header } from "@/components/layout/header";
import { StrategyCard } from "@/components/strategies/strategy-card";

export const metadata: Metadata = { title: "Strategies" };
export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default async function StrategiesPage() {
  const result = await fetchStrategies();

  if (!result.ok) {
    return (
      <div className="space-y-6">
        <Header title="Strategies" subtitle="Available trading strategies" />
        <div className="rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          Could not load strategies: {result.error.message}
        </div>
      </div>
    );
  }

  const { strategies, total } = result.data;

  return (
    <div className="space-y-6">
      <Header
        title="Strategies"
        subtitle={`${total} available strategy${total !== 1 ? "ies" : "y"}`}
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {strategies.map((strategy) => (
          <StrategyCard key={strategy.name} strategy={strategy} />
        ))}
      </div>

      {strategies.length === 0 && (
        <p className="py-12 text-center text-slate-500">
          No strategies available. Ensure the API server is running.
        </p>
      )}
    </div>
  );
}
