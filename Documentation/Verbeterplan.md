Verbeterplan AI Crypto Trading Bot
Samenvatting
Eerste prioriteit: strategiebetrouwbaarheid. Probleemstrategieën worden tijdelijk backtest-only gemaakt voor nieuwe paper/live runs.
Belangrijkste technische oorzaak: Signal.target_position wordt nu niet leidend gebruikt door paper/live execution; daardoor hebben parameters zoals position_size, dca_amount, take_profit_pct en grid ordergrootte onvoldoende effect.
Security volgt daarna met credentials-login, admin/viewer rollen en JWT/API-token bescherming.
Fase 1: Strategy Lockdown + Sizing Contract
Voeg centrale strategy-availability metadata toe: allowed_modes, status, demotion_reason, promotion_requirements.
Zet ma_crossover, breakout en model_strategy op backtest only; dca_rsi_hybrid, grid_trading en rsi_mean_reversion blijven beschikbaar voor paper/live.
Valideer dit in run-creatie en orphan recovery: gedemote strategieën mogen niet als paper/live starten of automatisch herstarten.
UI toont statusbadges en disabled paper/live opties voor gedemote strategieën.
Maak Signal.target_position contractueel leidend:
BUY: target_position is gewenste quote-notional; execution converteert naar base quantity en cap’t met risk manager.
SELL: cap altijd op bestaande positie; target_position=0 betekent volledige close.
confidence blijft conviction/risk-scaling, niet de enige sizing-input.
Acceptatie: parameterwijzigingen in dca_amount, position_size, take_profit_pct en grid size leiden aantoonbaar tot andere ordernotionals.
Fase 2: Strategy Test Harness
Voeg synthetische OHLCV fixtures toe voor uptrend, downtrend, ranging, choppy, gap shock, bull-to-bear, bear-to-bull, low-vol en high-vol.
Maak backtests deterministisch: Python random is al geset; voeg waar nodig NumPy seeding toe en log seed in run metadata.
Voeg strategy-invariant tests toe:
Geen SELL zonder positie.
Geen exposure buiten [0, 1].
Geen paper/live run voor gedemote strategieën.
Zelfde seed + dataset geeft identieke orders/trades.
Voeg een Makefile of scripts toe voor test, test-strategy STRAT=... en een snelle e2e backtest.
Lokale noot: pytest staat nu niet op PATH; verificatie loopt via uv run pytest ... of CI.
Fase 3: Actieve Strategieën Fixen
dca_rsi_hybrid:
Laat DCA buy-notional via target_position werkelijk door execution lopen.
Maak take-profit inventory-aware: percentage of vaste notional mag nooit meer verkopen dan de actuele positie.
Voeg max_holding_bars, max_loss_per_position_pct en live-default trailing stop toe.
Voeg min-notional preflight en duidelijke skip/error logs toe.
grid_trading:
Maak per-grid notional echt uitvoerbaar via het nieuwe sizing contract.
Voeg recentering toe met anchor_method: first_fill, rolling_mid, vwap_24h.
Voeg trend-pause/kill toe bij sterke trend of max drawdown.
Verhoog default RSI-filtering zodat grid niet blind in trending selloffs blijft accumuleren.
rsi_mean_reversion:
Voeg time-stop en adverse-excursion stop toe.
Voeg regime-filter toe: alleen mean-reversion in range/low-trend omstandigheden.
Voorkom “open long forever” door exits niet alleen afhankelijk te maken van RSI > overbought.
Acceptatie per actieve strategie: synthetische tests tonen BUY en SELL fills, closed trades, parametergevoeligheid en bounded drawdown.
Fase 4: Gedemote Strategieën Herstellen + Promotiegate
ma_crossover:
Voeg signal_timeframe toe, MA-type keuze (ema default), expliciete cross-down close en live trailing stop default.
breakout:
Voeg ATR trailing stop, opposite-channel exit, confirmation bars en optioneel volume-filter toe.
model_strategy:
Geen model_path of ontbrekend actief model wordt fail-fast bij run-start, niet stil HOLD.
Laad standaard latest active model per symbol/timeframe.
Audit future_return labeling; voeg embargo/purged splits en triple-barrier labeling toe.
Rapporteer AUC, PR-AUC, Brier/calibratie naast accuracy.
Voeg smoke test toe: train tiny model, activate, strategy emit signal, executor maakt order.
Promotie naar paper/live vereist: minimaal 200 closed trades, OOS Sharpe >= 1.0, max drawdown <= 15%, bootstrap significantie tegenover buy-and-hold, en geen kritieke strategy-invariant failures.
Fase 5: Metrics, Dashboard Trust en Backfill
Verifieer bestaande fixes voor profit factor en exposure; behoud profit_factor_is_infinite.
Splits metrics in realized closed trades versus open MTM:
realized_return_pct
open_mtm_pnl
n_closed_trades
n_open_positions_mtm
Leaderboard mag “best return” alleen ranken bij voldoende closed trades; 0-trade MTM-runs krijgen lage confidence.
Voeg per-symbol PnL, fees, slippage en CSV-filtering toe.
Voeg backfill-script toe dat bestaande runs herberekent en legacy metrics tijdelijk bewaart.
Fix short-ID routing voor run detail.
Fase 6: Security en Operator Controls
Implementeer credentials-auth met admin/viewer rollen:
users tabel met password hash en role.
/api/v1/auth/login, /refresh, /logout, /me.
UI login met httpOnly cookies; backend accepteert JWT of bestaande API key voor service clients.
Viewer is read-only; admin mag runs starten/stoppen, modellen activeren en kill-switch gebruiken.
Voeg Caddy reverse-proxy config toe met HTTPS, security headers en env-gebaseerde domeinnaam.
Voeg global kill-switch toe: admin-only endpoint + UI button die alle running paper/live tasks stopt, pending orders cancelt en audit events schrijft.
Zet productie-defaults veilig: REQUIRE_API_AUTH=true, wildcard CORS verboden, docs alleen achter auth of alleen lokaal.
Testplan
Backend unit: metrics, execution sizing, risk gates, strategy invariants, model fail-fast.
Backend integration: run create blocked for demoted paper/live, allowed for backtest, auth roles, global kill-switch, model activation.
Strategy regression: elke strategie op synthetische regimes plus minimaal één realistische historical smoke backtest.
Frontend: Jest tests voor disabled strategy modes, login/session state, run detail metrics.
E2E: Playwright voor login, create backtest, blocked paper/live demoted strategy, kill-switch flow.
Validatiecommands: uv run pytest tests/unit tests/integration --no-cov, daarna volledige coverage-run; npm test, npm run type-check, npm run build.
Assumpties
Spot-only blijft de v1 scope; shorts en perpetual funding worden later toegevoegd.
Gedemote strategieën blijven zichtbaar en backtestbaar, maar niet startbaar als paper/live tot promotiegate slaagt.
Credentials + rollen is de gekozen auth-default; GitHub OAuth blijft buiten scope.
Er is nog geen concrete productiedomeinnaam nodig; HTTPS-config gebruikt env placeholders.