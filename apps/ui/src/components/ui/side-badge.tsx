/**
 * SideBadge
 * ---------
 * Renders the BUY/SELL "side" label used by trades, orders and fills tables.
 * Extracted in Sprint 40 Stap 10 (CR-010) so the duplicated className ternary
 * across TRADE_COLUMNS, ORDER_COLUMNS and FILL_COLUMNS collapses to a single
 * visual source of truth — renames of the profit/loss design tokens stay in
 * one place.
 */

type Side = "buy" | "sell" | string;

interface SideBadgeProps {
  side: Side;
}

export function getSideClassName(side: Side): string {
  const base = "text-xs font-medium";
  return side === "buy" ? `text-profit ${base}` : `text-loss ${base}`;
}

export function SideBadge({ side }: SideBadgeProps) {
  return <span className={getSideClassName(side)}>{side.toUpperCase()}</span>;
}
