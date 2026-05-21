Je bent een senior quant + full-stack engineer. Mijn trading-bot dashboard
(Next.js frontend + Python/FastAPI backend, draait op http://167.235.51.90:3000)
heeft over 1.517 trades een win-rate van 35,7% en -$2.212 PnL. Er zijn 6 strategieën
(ma_crossover, rsi_mean_reversion, breakout, model_strategy, dca_rsi_hybrid,
grid_trading) en een ML-pipeline met future_return labels die 72-80% accuracy
claimt maar 0 trades produceert. 14 runs hangen in error-state. Het dashboard
draait op HTTP zonder auth op een publiek IP.

Werk in deze volgorde en commit per fase met duidelijke messages.

=== FASE 0: Repo-audit ===
1. Map de repo-structuur, identificeer backend, frontend, strategy modules,
   model training pipeline, en data ingest.
2. Lever een ARCHITECTURE.md met diagram (mermaid) en een lijst van zorgen
   per laag (data, features, modeling, execution, risk, ops, UX).

=== FASE 1: Veiligheid (P0, MUST-HAVE) ===
3. Voeg authenticatie toe (NextAuth of FastAPI OAuth2 met JWT). Minimaal
   admin/viewer rollen. Force HTTPS via reverse proxy config (Caddy of Nginx
   voorbeeld in /deploy).
4. Implementeer global kill-switch endpoint + UI button die ALLE live runs
   stopt en pending orders cancelt.
5. Per run: daily_loss_limit, max_drawdown_pct, max_position_size,
   max_exposure_per_symbol. Trigger auto-stop bij overschrijding.
6. Promotion gate paper→live: min 200 trades, OOS Sharpe ≥ 1.0,
   max_dd ≤ 15%, statistisch significant t.o.v. buy&hold (bootstrap p<0.05).

=== FASE 2: Backtest- en validatie-engine ===
7. Voeg een realistisch cost model toe: maker/taker fees per exchange,
   slippage model (lineair in size × ADV-fractie), funding cost voor perps.
8. Implementeer walk-forward optimization (rolling + anchored varianten)
   met purged k-fold (Lopez de Prado) om leakage te voorkomen.
9. Rapporteer deflated Sharpe ratio, PSR, en bootstrap CI's op Sharpe/return.
10. Audit de `future_return` labeling op target leakage; introduceer
    triple-barrier labeling als alternatief.

=== FASE 3: ML-pipeline ===
11. Feature store met versioning (Parquet of DuckDB) en feature drift
    monitoring (PSI/KS).
12. Train/val/test split met embargo. Log AUC, PR-AUC, calibratie (Brier,
    reliability curve) i.p.v. alleen accuracy.
13. Model registry met champion/challenger en shadow trading vóór hot-swap.
14. Fix `model_strategy`: zorg dat actieve modellen daadwerkelijk signalen
    naar de executor sturen; voeg een end-to-end smoke test toe.

=== FASE 4: Risk & portfolio ===
15. Vol-targeting position sizing (annualized vol target, bv. 15%).
16. Correlation-aware allocator over symbolen (Ledoit-Wolf shrinkage covariance).
17. Regime-aware meta-allocator die strategie-gewichten dynamisch verschuift
    op basis van Fear&Greed, BTC dominance, en realized vol.
18. Voeg orderbook-, funding-rate- en on-chain (whale flow) features
    daadwerkelijk toe aan het feature pipeline.

=== FASE 5: Observability ===
19. Structured logging (JSON) + Prometheus metrics (latency, slippage,
    fill-rate, PnL per symbol).
20. Grafana dashboard template in /deploy/grafana.
21. Alerting via Telegram/Slack bij: error run, drawdown breach, stale
    market data >60s, exchange disconnect, model drift.
22. Auto-retry met exponential backoff voor de 14 hangende error runs;
    root-cause logger.

=== FASE 6: UX / Dashboard ===
23. Fix run-detail routing zodat ook short ID's werken (server-side lookup).
24. Run-detail: per-symbol PnL attributie, fee/slippage breakdown,
    trade-list met filter en CSV-export per symbol.
25. Tags, notes en favorieten op runs. Filter op symbol, date-range,
    return-bucket.
26. Currency consistency: detect base currency (EUR vs USD) en converteer
    equity correct met dagelijkse FX-rate.
27. Mobile-responsive sidebar (collapse onder 768px).

=== Leveringseisen ===
- Schrijf tests (pytest + Playwright) voor elke nieuwe feature.
- Update OpenAPI spec en de /api/docs pagina.
- Geen breaking changes zonder migration script.
- Geen secrets in repo; gebruik .env + .env.example.
- Maak per fase een PR met checklist en screenshots.

Begin met FASE 0 en wacht op mijn akkoord voordat je FASE 1 start.