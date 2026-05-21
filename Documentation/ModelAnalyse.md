Je bent een senior quant-engineer. Ik heb mijn trading-bot dashboard
(http://167.235.51.90:3000) volledig geanalyseerd — alle 6 strategieën,
de optimizer, de metrics-pipeline, de ML-laag en de live-executor.
De observaties hieronder zijn ground truth: gebruik ze, gok niet.

Werk strikt in fases. Maak een PR per fase met:
  • probleembeschrijving (kopieer uit CONTEXT)
  • aanpak
  • test-evidence (CI-screenshot, optimizer-uitslag waar parameters
    spreiding produceren, backtest-comparison vóór/ná fix)
  • rollback-plan
Werk niet aan fase N+1 voordat fase N door tests + review is.
Gebruik conventional commits. Begin met FASE 0 en stop voor mijn akkoord.

============================================================
CONTEXT — geobserveerde feiten per strategie
============================================================

LEADERBOARD (huidige stand, misleidend):
  #1 grid_trading        20 runs  +6,71% avg  Sharpe 4,164  9 trades  PF 0,00
  #2 rsi_mean_reversion  60 runs  +1,71% avg  Sharpe 2,360  98 trades PF 0,07
  #3 dca_rsi_hybrid      28 runs  -1,12% avg  Sharpe 0,141  339 tr.  PF 1,03
  #4 model_strategy      15 runs   0,00% avg  Sharpe 0,000  0 trades PF 0,00
  #5 ma_crossover        55 runs  -4,02% avg  Sharpe -1,826 525 tr.  PF 0,34
  #6 breakout            36 runs  -6,97% avg  Sharpe -2,715 29 tr.   PF 0,08

  Totaal over alle runs: 1.517 trades, win-rate 35,7%, realized PnL -$2.212.

----- STRATEGIE 1: grid_trading (v1.0.0) -----
Params: grid_size_pct=0.01, num_grids=5, position_size=100,
        rsi_period=14, min_rsi_buy=0, max_rsi_sell=100,
        trailing_stop_pct=null  ← schema "type: null"
Bevindingen:
- Backtest `72e39a09` (118d, BTC/ETH/SOL/XRP):
    return +12,88%, Sharpe 1,944, PF 0,00, exposure 0,00%
    (0/11292 bars), 0 closed trades, 3 orders, 3 open longs.
    Rendement = puur mark-to-market van BTC 89k→99k, ETH 2045→3634,
    SOL 124→174. Geen alpha, alleen long-bias.
- Backtest `3eb149fb`: 9 trades, 100% wr, +€53 PnL, PF=0,00
  ondanks 9W/0L → metric-bug.
- Live `64f9f6fc` (€200, 2u15): 0 orders/fills, error-state,
  geen log in UI.
- Optimizer-run: 0/1 combinations, 1 failed, 0 sec wegens schema-
  bug `trailing_stop_pct.type: null` (moet `["number","null"]`).
- 0 SELL-fills in 118d backtest, geen recenter, geen kill-switch.

----- STRATEGIE 2: dca_rsi_hybrid (v1.0.0) -----
Params: dca_interval_bars=16, dca_amount=50, rsi_period=14,
        rsi_boost_threshold=40, rsi_boost_multiplier=2,
        rsi_skip_threshold=75, take_profit_rsi=70,
        take_profit_pct=0.5, position_size=1000,
        trailing_stop_pct=null
Bevindingen:
- Backtest `d6a1625e` (109d, $10k, 5 syms):
    +1,45% return, Sharpe 1,892, PF 1,36, 74,8% wr, 111 trades,
    exposure 1404/1112 bars = 126% (rekenkundig onmogelijk).
    ALLE 100 trades hebben SIDE=BUY. Alle trades "sluiten" exact
    op 21:17:08–21:17:10 = synthetische closes op laatste candle
    = endpoint-bias.
- Backtest `ce32480c` (118d, $2k, 4 syms):
    -3,69% return, Sharpe -1,611, PF 0,69, exposure 104,6%.
    Sharpe-swing +1,89→-1,61 tussen overlappende periodes =
    niet reproduceerbaar / geen seed.
- Optimizer-runs `02ffd031` (48 combs) en `eb6dff1a` (12 combs):
    top-rows hebben IDENTIEKE Sharpe/return/wr/trades terwijl
    `dca_amount`, `take_profit_pct`, `rsi_boost_threshold`
    verschillen. → DEZE 3 PARAMETERS ZIJN DODE CODE in de
    execution pipeline.
- Live `d2abe8ba` (€200, 40u): 1 order, 2 fills, 0 closed.
    Met dca_interval_bars=16 op 15m = 4u/buy → ~10 fills verwacht.
    Live-scheduler is dood OF margin-check faalt stil.

----- STRATEGIE 3: ma_crossover (v1.0.0) -----
Params: fast_period=10, slow_period=50, position_size=1000,
        trailing_stop_pct=null
Bevindingen:
- Backtest `f4e77789` (300d, 5 syms, EUR):
    -10,28% return, -20,58% max DD, 1 closed trade (loss -$2,40),
    51/10396 bars exposure = 0,49%, 3 open posities die
    cumulatief ~-$1.000 mark-to-market verloren.
- Backtest `22bb2e70` (151d, 4 syms, USD):
    -5,97% return, -12,34% max DD, 3 trades (alle 3 loss),
    exposure 0,56%, 3 open posities.
- Strategie staat 99,5% van de tijd flat. SMA(10)/SMA(50) op 15m
  produceert te weinig signalen voor crypto-cycli. Crossovers
  triggeren entry maar exit-signaal is óf nooit, óf te laat.
- Geen stop-loss (`trailing_stop_pct=null` default).
- Profit factor 0,00 bij alleen losses = metric M1 bug.

----- STRATEGIE 4: rsi_mean_reversion (v1.0.0) -----
Params: rsi_period=14, oversold=30, overbought=70,
        position_size=1000, trailing_stop_pct=null
Bevindingen:
- Optimizer-run `1847f823` (15m, USD-set, 1 combinatie):
    +13,82% return, Sharpe 1,769, 11,24% DD, **0 trades**,
    0% win rate. → "Best return" op leaderboard is een fantoom:
    pure mark-to-market van open longs.
- Backtest `2e6b9510` (59d, 5 syms):
    -0,26% return, Sharpe -2,465, 2 trades (beide loss),
    exposure 41,80% (474/1134 bars — proportioneel correct hier,
    inconsistent met andere runs!), PF 0,00.
- Backtest `55a4e86e` (5 syms USD):
    -5,89% return, max DD 14,24%, 1 win van +$8,66, 3 open
    posities verliezen ~$600 mark-to-market. Win-rate 100%,
    account verliest.
- Conclusie: entry op RSI<30 werkt, maar exit op RSI>70 wordt
  zelden gehaald in crypto-downtrends. Geen tijdelijke stop.
  Effectief: koop the dip + hodl forever.

----- STRATEGIE 5: breakout (v1.0.0) -----
Params: lookback_period=20, position_size=1000, atr_period=14,
        atr_multiplier=1.5, trailing_stop_pct=null
Bevindingen:
- Backtest `f6170c46` (151d): -7,16% return, -17,55% max DD,
    PF 0,00, **0 closed trades**, 3 orders/fills/open positions,
    exposure 0% (0/11013 bars) ← inconsistentie: 3 open posities
    maar 0% exposure (exposure-tracker erkent open posities niet).
- Backtest `3267310a` (59d): -9,68% return, -15,02% DD, Sharpe
    -3,501, 0 closed trades, 3 open, exposure 0%.
- Donchian-breakout-entries werken, maar de strategie heeft 0
    exits in 5 maanden trading. ATR-multiplier is geconfigureerd
    voor "confidence scaling" maar gebruikt nergens als stop.
- Identiek patroon aan grid_trading en ma_crossover:
    asymmetrische entries-only strategie.

----- STRATEGIE 6: model_strategy (v0.2.0) -----
Params: model_path="", feature_window=100, prediction_threshold=0.6,
        position_size=1000, model_dir="models/", trailing_stop_pct=null
Bevindingen:
- Backtest `df53df6a` (BTC/ETH/SOL/XRP USD, 7 sec wall-clock):
    0 trades, 0 orders, 0 fills, 0 positions.
    De ML-pipeline draait maar genereert geen signaal.
- /models toont 9 model-versies (BTC/ETH/SOL/XRP × EUR & USD),
    allemaal trained met method=future_return, trigger=manual,
    accuracy 72,7%–80,0%, **TRADES_USED = 0** voor elk model.
- Geen enkel model is ACTIVE (elk model heeft "Activate" knop,
    niet "Deactivate"). model_path default is leeg.
- Hot-swap reloader (RetrainingService) is bedraad maar nooit
    getriggerd → end-to-end smoke-test ontbreekt.
- 72-80% accuracy met label=future_return en 0 trades-used =
    klassieke red flag voor TARGET LEAKAGE (label maakt impliciet
    gebruik van toekomstige bars zonder embargo).
- prediction_threshold=0.6 is mogelijk te hoog voor 3-klasse
    (BUY/HOLD/SELL) softmax-output van een ondergetrained model.

----- INFRASTRUCTUUR-BUGS (cross-cutting) -----
INF-1: profit_factor = 0,00 bij 0 losses (moet inf/None) — minimaal
       7 runs aangedaan.
INF-2: exposure_bars > total_bars (>100%) — bevestigd in 2 dca-
       runs en inconsistent low (0%) bij grid/breakout met open
       posities.
INF-3: Trades-tab toont synthetische closes op laatste candle voor
       openstaande posities, vermengt met realized closed_trades.
       UI maakt geen onderscheid.
INF-4: Run-detail toont geen logs of error-stacks. 14 runs in
       error-state in totaal, oorzaken onzichtbaar.
INF-5: Optimizer-schema breekt op `trailing_stop_pct.type: null`.
       0/1 combinations gerapporteerd voor grid_trading.
INF-6: Dashboard draait op `http://167.235.51.90:3000` zonder auth,
       publiek IP, HTTP. "Stop Run" en "New Run live" zijn voor
       iedereen toegankelijk.
INF-7: Equity in $ getoond, trading pairs in EUR — geen FX-
       normalisatie in metrics.
INF-8: Leaderboard middelt regimes; "best return" kan een 0-trades
       fantoom-mark-to-market zijn.
INF-9: Backtests niet reproduceerbaar (geen seed pinning), zie
       dca_rsi_hybrid Sharpe-swing.
INF-10: Live-scheduler heartbeat ontbreekt. dca-cycle stopt stil.

============================================================
FASE 0 — REPO AUDIT & TEST INFRA  (PR: chore: audit + harness)
============================================================
1. Map de repo:
   - strategies/*  (6 strategie-klassen)
   - schemas/*     (param schemas, JSON/pydantic)
   - engine/       (backtest, paper, live executors)
   - metrics/      (sharpe, sortino, calmar, pf, exposure)
   - optimizer/    (grid search)
   - ml/           (RetrainingService, model loader, features,
                    labeling future_return)
   - api/, ui/
2. Lever STRATEGY_AUDIT.md met:
   - mermaid van data-flow params → strategy.on_bar → signal →
     order → fill → trade → metrics
   - per bevinding in CONTEXT: file:line waar het probleem zit
3. Test-harness:
   - pytest config, fixtures voor synthetische OHLCV-datasets:
     {uptrend, downtrend, ranging, choppy, gap_shock, bull_to_bear,
      bear_to_bull, low_vol, high_vol}
   - pin numpy + python random seed in backtest entrypoint
   - voeg `make test`, `make test-strategy STRAT=<name>`
   - voeg property-based tests (hypothesis) voor strategy invarianten
4. CI workflow: lint + pytest + één quick e2e backtest.
Acceptatie: STRATEGY_AUDIT.md gereviewd; groen pytest-skeleton.

============================================================
FASE 1 — METRICS PIPELINE  (PR: fix(metrics): trustworthy stats)
============================================================
Fixes INF-1, INF-2, INF-3, INF-7, INF-8.

M1 profit_factor:
  - profit_factor = sum_gains / sum_losses; sum_losses==0 → return
    None; sum_gains==0 → 0; geen losses én geen gains → None.
  - Test: 9W/0L → None; 5W/5L gelijk → ~1; 0W/5L → 0.

M2 exposure:
  - exposure = bars_with_open_position / (total_bars × n_symbols)
  - per-symbol-exposure in metrics-dict
  - open posities tellen mee, ook als entry niet geclosed is
    (fix breakout 0% bug, fix dca >100% bug)
  - schema-guard: 0 ≤ exposure ≤ 1, anders raise
  - Test: 5 syms, 100% van tijd in alle → 100%; 1 van 5 50% → 10%;
    3 open posities zonder close 60% van bars → 12%.

M3 trades-vs-mark:
  - Splits in `closed_trades` (echte round-trips, realized) en
    `open_positions_mtm` (open posities geprojecteerd naar last bar).
  - Win-rate, profit_factor, avg_trade_pnl UITSLUITEND over
    closed_trades.
  - Total return MAG mark-to-market bevatten maar moet flag
    `realized_return` apart tonen.
  - UI: aparte tabs "Closed Trades" en "Open Positions (MTM)".
  - Test: 5 buys + 2 closes → 2 closed_trades, 3 open_positions_mtm,
    realized_return < total_return als open posities winst hebben.

M4 sample-confidence + robust stats:
  - Voeg `n_observations` flag toe; trades<30 of duration<30d →
    confidence=low.
  - Implementeer Probabilistic Sharpe Ratio (PSR) + Deflated
    Sharpe (DSR) — toon in run-detail.
  - Test: bekende toy-cases (Lopez de Prado).

M5 leaderboard transparency:
  - Kolommen: bear_period_return, worst_decile_return,
    n_distinct_periods, n_closed_trades, n_open_mtm_only.
  - "Best return" alleen tonen als ≥10 closed_trades, anders "n/a*".
  - Tooltip op avg: "gemiddelde over heterogene regimes".

M6 FX/currency:
  - Detect quote currency per symbol. Equity & return berekenen
    in user-gekozen base currency met dagelijkse FX (CoinGecko of
    ECB).
  - Test: BTC/EUR positie + USD equity-display → consistent na FX.

M7 reproducibility:
  - Pin seed (numpy + python + torch indien aanwezig) per run, log
    in run-metadata.
  - Backtest identiek twee keer draaien → byte-identieke trades-CSV.

Backfill-script: hercompute metrics voor alle bestaande runs.
Sla `legacy_*` versies op gedurende 30 dagen voor vergelijking.

============================================================
FASE 2 — SECURITY & OPS  (PR: feat(ops): auth, kill-switch, logs)
============================================================
Fixes INF-4, INF-6, INF-10.

S1 auth + HTTPS:
  - NextAuth (frontend) + FastAPI OAuth2/JWT (backend), rollen
    admin/viewer.
  - Caddy of Nginx config in /deploy met automatic TLS.
  - IP-allowlist via env-var (default open, productie restrictief).

S2 global kill-switch:
  - Endpoint POST /api/kill-switch → cancelt alle pending orders,
    sluit alle open posities (live + paper), zet alle runs op
    stopped.
  - UI button rechtsboven (alleen admin).

S3 logs in run-detail:
  - Backend: structured JSON logs per run, append-only, last 10k
    entries beschikbaar via /api/runs/{id}/logs?level=&since=.
  - UI: nieuwe tab "Logs" met level-filter, search, tail-mode.
  - Backfill: voor 14 bestaande error-runs minimaal de stacktrace
    surfacen (best-effort uit bestaande logfiles).

S4 live-scheduler heartbeat:
  - Per running strategy-instance: `last_on_bar_ts`,
    `last_signal_ts`, `last_order_attempt_ts` (per symbol).
  - Alert (Slack/Telegram webhook) als `now - last_on_bar_ts >
    2 × timeframe_seconds`.
  - UI: heartbeat-indicator op /runs lijst.

S5 alerting:
  - Webhook-config in .env (SLACK_URL / TELEGRAM_TOKEN).
  - Triggers: error-status, kill-switch, drawdown-breach,
    stale-data, exchange-disconnect, model-drift.

============================================================
FASE 3 — STRATEGY FIXES (parallel-PRs, één per strategie)
============================================================

==== 3a grid_trading ====
G1 schema-bug: trailing_stop_pct.type = ["number","null"] of
   Optional[float]. Optimizer moet 4/4 combinaties draaien.
G2 recenter-logica: recenter_threshold_pct (def 0,05),
   anchor_method ∈ {"first_fill","rolling_mid","vwap_24h"}.
G3 trend-kill-switch: ADX(14)>35 → pause + close, plus
   max_drawdown_kill_switch_pct (def 15%).
G4 sizing: vervang position_size door risk_pct_per_grid (def 0,5%);
   respecteer exchange min_notional + min_qty; skip+log bij te klein.
G5 RSI-filter default ACTIEF: min_rsi_buy=30, max_rsi_sell=70
   (of verwijder tag "rsi" + spec).
G6 SELL-fills moeten in synthetische uptrend ≥1 keer voorkomen.
Acceptatie: optimizer 32 combinaties zonder fail; bear-backtest
stopt op kill-switch; live paper-run €200/4u ≥1 closed round-trip.

==== 3b dca_rsi_hybrid ====
D1 dca_amount LIVE maken: locate read in strategy class
   (vermoedelijk overschreven door position_size). Order-size =
   dca_amount × (rsi_boost_multiplier if rsi<rsi_boost_threshold
   else 1).
D2 SELL-pad bedraden: RSI≥take_profit_rsi & avg_entry<price →
   SELL take_profit_pct × position_qty. Smoke-test:
   synthetische RSI→80 → ≥1 SELL.
D3 parameter-sensitivity: optimizer met 8 combs → ≥3 distinct
   Sharpe-waarden. Geen identieke top-rows toegestaan.
D4 stop-loss: max_loss_per_position_pct (def 8%), trailing_stop_pct
   default 0,05 in live-mode.
D5 live-scheduler audit: 4u paper-run met dca_interval_bars=2 op
   15m → ≥3 buy-attempts in log.
D6 pre-flight check: per (symbol, dca_amount) ≥ exchange min_notional;
   fail-fast met duidelijke error in run-detail.
Acceptatie: optimizer toont spreiding over álle params; backtest
heeft BUY én SELL fills; live 24u/€200 ≥6 BUY-attempts.

==== 3c ma_crossover ====
MA1 timeframe-mismatch: SMA(10/50) op 15m → te traag. Voeg
    `signal_timeframe` toe (default 1h) onafhankelijk van execution
    timeframe.
MA2 exit-logica expliciet maken: cross-down moet position closen
    (niet alleen short flip). Test: bull crossover entry,
    bear crossover exit binnen 200 bars in trend-dataset.
MA3 stop-loss verplicht in live: trailing_stop_pct default 0,05.
MA4 multi-symbol independent positions (geen netting bug); voeg
    max_concurrent_positions toe.
MA5 add MA-type optie: {"sma","ema","wma","hma"}; default ema voor
    crypto (snellere reactie).
Acceptatie: 300d backtest produceert ≥30 closed trades (i.p.v. 1);
profit-factor en realized-return berekend.

==== 3d rsi_mean_reversion ====
RM1 hard exit toevoegen: time-stop max_holding_bars (def 96 op 15m
    = 24h), én cut-loss max_adverse_excursion_pct (def 5%).
    Endpoint-bias verdwijnt zo automatisch want geen oneindige
    open posities meer.
RM2 RSI-symmetrie: ook SHORT-side overwegen (entry RSI>overbought,
    exit RSI<50). Maak `enable_short` flag, default False.
RM3 regime-filter: rsi_mean_reversion alleen actief als
    bb_width < threshold of ADX < 25 (ranging market). Trending
    bear is doodlopend pad voor mean-reversion.
RM4 default `overbought=68, oversold=32` (iets minder extreem);
    test sensitivity.
Acceptatie: backtest van 100d in bear-regime → max DD < 10%, geen
"100% wr met -5,89% return"-paradox meer.

==== 3e breakout ====
BR1 exit-pad implementeren: breakout-exit op opposite Donchian-touch
    OF ATR-based trailing-stop (atr_multiplier × ATR onder entry).
    Spec belooft "ATR-scaled confidence" — bedraad dat in sizing
    EN in stop.
BR2 false-breakout filter: vereist `confirmation_bars` (def 2)
    close-above-channel; verwerp wicks.
BR3 volume-filter (optional): `min_volume_multiplier` (def 1,5×
    avg_vol_20).
BR4 trailing stop default ON in live (0,05).
Acceptatie: 150d backtest produceert ≥10 closed trades met BUY én
SELL fills; max DD < 12%; PF > 0,8.

==== 3f model_strategy ====
MS1 target-leakage audit op `future_return` labeling:
  - Verifieer dat label op bar t uitsluitend gebaseerd is op
    bars t..t+H zonder kennis van features uit bars t+1..t+H.
  - Implementeer embargo (h bars rondom split-grens) en purged
    k-fold cross-validation (Lopez de Prado).
  - Vervang/aanvullen met TRIPLE-BARRIER labeling: {up-touch,
    down-touch, time-stop} → 3-klasse classifier.
  - Test: shuffle labels → accuracy moet richting ~33% (3-klasse
    baseline). Als shuffle nog steeds >55% scoort = leakage.
MS2 model_path leeg = strategie produceert geen signaal:
  - Pre-flight check bij run-start: model_path bestaat én is
    geldig joblib/pickle; anders fail-fast.
  - Default: laad latest ACTIVE model uit /models registry voor
    het symbool.
MS3 Activate-flow:
  - "Activate" knop in UI moet actually een model active maken én
    de live-strategie hot-swappen.
  - Per (symbol, timeframe) max 1 actief model. UI toont
    "Deactivate" voor actief model.
  - Audit: in /models tonen accuracy + AUC + PR-AUC + calibration
    (Brier, reliability curve) i.p.v. alleen accuracy.
MS4 prediction_threshold sensitivity:
  - Standaard 0,6 is te restrictief voor 3-klasse softmax.
  - Voeg `class_thresholds` per klasse toe (kalibreerbaar via
    isotonic regression).
MS5 hot-swap end-to-end smoke-test:
  - Test in CI: train tiny model → publish via RetrainingService
    → running strategy-instance herlaadt binnen