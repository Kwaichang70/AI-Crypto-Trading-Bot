/**
 * Quote-currency display helpers (AppAnalyse FASE 6: currency consistency).
 *
 * The UI used to hardcode "$" everywhere while runs may trade EUR (or other)
 * quote pairs. The display symbol is derived from the run's symbols, with an
 * explicit backend-provided quote currency (BacktestMetrics.quoteCurrency,
 * Sprint 49 M6) taking precedence when available.
 */

const SYMBOL_BY_CODE: Record<string, string> = {
  EUR: "€",
  USD: "$",
  USDC: "$",
  USDT: "$",
  GBP: "£",
};

/** Extract the quote code from a trading pair, e.g. "BTC/EUR" -> "EUR". */
function quoteOf(pair: string): string | null {
  const idx = pair.indexOf("/");
  if (idx < 0 || idx === pair.length - 1) return null;
  return pair.slice(idx + 1).toUpperCase();
}

/**
 * Resolve the display prefix for monetary values of a run.
 *
 * Precedence: explicit backend quote currency, else the (single) quote code
 * shared by all symbols. Mixed or unknown quotes return "" — no symbol is
 * better than a misleading one. Unknown-but-consistent codes render as
 * "CODE " (e.g. "CHF ").
 */
export function quoteCurrencyPrefix(
  symbols: readonly string[] | undefined,
  explicit?: string | null,
): string {
  let code: string | null = null;
  if (explicit) {
    code = explicit.toUpperCase();
  } else if (symbols && symbols.length > 0) {
    const quotes = new Set(
      symbols.map(quoteOf).filter((q): q is string => q !== null),
    );
    if (quotes.size !== 1) return "";
    code = [...quotes][0];
  }
  if (!code) return "";
  return SYMBOL_BY_CODE[code] ?? `${code} `;
}
