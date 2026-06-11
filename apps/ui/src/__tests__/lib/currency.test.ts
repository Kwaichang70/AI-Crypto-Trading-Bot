/**
 * Tests for quoteCurrencyPrefix (lib/currency.ts) — currency-consistent
 * monetary display on the run-detail page.
 */
import { quoteCurrencyPrefix } from "@/lib/currency";

describe("quoteCurrencyPrefix", () => {
  it("derives € from EUR pairs", () => {
    expect(quoteCurrencyPrefix(["BTC/EUR", "ETH/EUR"])).toBe("€");
  });

  it("derives $ from USD-family pairs", () => {
    expect(quoteCurrencyPrefix(["BTC/USD"])).toBe("$");
    expect(quoteCurrencyPrefix(["BTC/USDT", "ETH/USDT"])).toBe("$");
    expect(quoteCurrencyPrefix(["SOL/USDC"])).toBe("$");
  });

  it("returns empty string for mixed quote currencies", () => {
    expect(quoteCurrencyPrefix(["BTC/EUR", "ETH/USD"])).toBe("");
  });

  it("returns empty string for no symbols", () => {
    expect(quoteCurrencyPrefix([])).toBe("");
    expect(quoteCurrencyPrefix(undefined)).toBe("");
  });

  it("explicit backend currency takes precedence over symbols", () => {
    expect(quoteCurrencyPrefix(["BTC/EUR"], "USD")).toBe("$");
  });

  it("unknown but consistent code renders as 'CODE '", () => {
    expect(quoteCurrencyPrefix(["BTC/CHF"])).toBe("CHF ");
  });

  it("malformed pairs are ignored", () => {
    expect(quoteCurrencyPrefix(["BTCEUR", "BTC/"])).toBe("");
  });

  it("null explicit falls through to symbols", () => {
    expect(quoteCurrencyPrefix(["BTC/EUR"], null)).toBe("€");
  });
});
