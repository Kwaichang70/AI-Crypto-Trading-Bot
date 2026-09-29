# Kritische analyse + Verbeterplan v2 — AI Crypto Trading Bot

**Datum:** 2026-09-24 · **Status:** PLAN (niets uitgevoerd) · **Basis:** 3 parallelle code-verkenningen (backend/engine, quant/risk/backtest, frontend/infra/tests) + 1 ontwerp-pass, 207 commits op `main`, productie op Hetzner via Tailscale.
**Locatie:** `Documentation/Verbeterplan-v2-2026-09.md` (goedgekeurd door gebruiker 2026-09-24; alleen dit document en het register in `CLAUDE.local.md` zijn aangemaakt/bijgewerkt — geen codewijzigingen).

---

## 0. Context en waarom dit plan

Het vorige `Documentation/Verbeterplan.md` (mei 2026) is grotendeels uitgevoerd: sizing-contract, strategy-lockdown, metrics-pipeline, Caddy/NextAuth, kill-switch, promotion gates. Sindsdien is `momentum_breakout` gepromoveerd naar LIVE en draait sinds vandaag 14:13 UTC een echte live run (`f36fb44e`, €100 EUR, daily, ATR-bracket SL 1.5×/TP 3.0×, exits 100% gedelegeerd aan de engine-side bracket manager).

Deze analyse is gedaan met de vraag: **is de app veilig genoeg om echt geld te laten draaien, en zijn de bewijzen waarop de live-promotie rust betrouwbaar?** Het antwoord op beide is op dit moment **nee**, om redenen die hieronder met bestand en regelnummer zijn onderbouwd. De belangrijkste bevinding is geverifieerd door zelf de code te lezen, niet alleen via de agent-rapporten.

### Samenvatting in vijf zinnen
1. **De live-engine kan niet verkopen.** De interne positieadministratie van `LiveExecutionEngine` wordt alleen gevuld door `sync_positions()`, die nergens in productiecode wordt aangeroepen. Elke SELL, inclusief bracket stop-loss/take-profit en trailing stop, wordt in live geweigerd met `live.sell_no_position`. De lopende live run heeft dus **geen werkende exit**.
2. **De veiligheidslaag is grotendeels dood of verkeerd.** De graduated CircuitBreaker wordt in productie nooit aangemaakt, leest bovendien sleutels die niet bestaan, de kill-switch wordt nooit getriggerd, de "daily loss"-limiet is een permanente cumulatieve kill, en alle risk gates blokkeren ook beschermende SELLs.
3. **Het backtest-bewijs is te optimistisch.** Backtests rekenen met 10/15 bps fees waar live 40/60 bps geldt (round-trip ≈0,4% vs ≈1,3%), injecteren de *huidige* Fear&Greed/CoinGecko-waarden in historische bars (look-ahead), aligneren multi-symbol bars op index i.p.v. timestamp, en stempelen trades met wall-clock tijd.
4. **Ops en security zijn niet productie-waardig:** geen CI (workflow-bestand verwijderd in working tree), API-auth uit op de server terwijl live trading aan staat, Grafana-alerts bereiken niemand, geen database-backups, geen rollback-procedure, geen log-rotatie.
5. **Architectuurschuld remt alles:** god-modules (runs.py 2193 regels, strategy_engine.py 1826), drie positie-ledgers, drie handmatige strategy-registries, module-globale run-state zonder lock, ~1.400 regels dode code, en ML-track die feitelijk niet functioneert.

**Positief:** de discipline rond held-out validatie, het bracket-mechanisme (in paper bewezen), 2100+ unit tests, mypy strict, het 3-laags live gate ontwerp, audit events en het agent-workflow proces zijn solide fundamenten. Het probleem is niet gebrek aan features maar dat de *veiligheidsfeatures niet aangesloten zijn* en dat de *validatie-input niet klopt*.

---

## 1. Bevindingen (kritisch → structureel)

Ernst: **C** = kritiek (kapitaal-/veiligheidsrisico of foutief bewijs), **S** = structureel/hoog.

### 1.1 Live-veiligheid en risk-correctheid

| ID | Bevinding | Bewijs |
|---|---|---|
| **C1** | **Live-engine kan geen SELL uitvoeren.** `_positions` wordt alleen geschreven in `sync_positions()`; geen enkele productie-aanroep. BUY-fills schrijven het niet. SELL-guard geeft `[]` terug. Unit tests verbergen dit door `engine._positions[...]` direct te injecteren. Bracket manager leest de positie wél correct uit `PortfolioAccounting` en emitteert het SELL-signaal; het sterft één call later in de live engine. | `packages/trading/engines/live.py:578-587, 619, 1050-1108`; enige callers in `tests/unit/test_live_execution.py` |
| **C2** | **CircuitBreaker nooit geïnstantieerd in productie.** `circuit_breaker=None` in beide engine-starters. HALT/REDUCE/DAILY_LIMIT, HALT-auto-stop (Sprint 50 C4) en de Grafana HALT-alert zijn dood; `/runs/{id}/circuit-breaker` geeft altijd 404. | `apps/api/services/run_orchestrator.py:816-830, 1133-1185`; `apps/api/routers/circuit_breaker.py:38-40` |
| **C3** | **Verkeerde summary-sleutels** (verborgen achter C2): engine leest `realised_pnl`/`max_drawdown`; portfolio levert `total_realised_pnl`/`max_drawdown_pct`. Input altijd 0 → breaker kan nooit vuren, ook na aansluiten. | `strategy_engine.py:877-885` vs `portfolio.py:622-642` |
| **C4** | **Live equity = alleen cash.** `_fetch_equity` pakt de eerste quote-balance (EUR, dan USDT/…), negeert crypto-holdings. Kopen registreert als drawdown; twee 15%-posities = "30% drawdown" → alle orders geblokkeerd. `initial_capital` wordt in live genegeerd voor sizing (hele account is de basis). | `live.py:975-1007, 1009-1029, 1139` |
| **C5** | **"Daily" PnL is cumulatief sinds run-start** in paper én live; `_check_daily_loss` werkt daardoor als permanente kill na 8% cumulatief verlies, in alle modi. Paper-peak = `max(initial_cash, current)`. Blokkeert ook SELLs → posities blijven vastzitten. (Dit is de bug uit register #41 die eerder OOS-gates kan hebben vertekend.) | `paper.py:907-921`, `live.py:1031-1044`, `risk.py:469-484` |
| **C6** | **Risk gates zijn side-blind:** daily-loss, cooldown na 3 verliezen, drawdown ≥30%, max-open-positions blokkeren ook beschermende exits. Kill-switch in LIVE `return`t vóór de bracket/trailing-secties. `trigger_kill_switch` wordt nergens aangeroepen. | `risk_manager.py:88-122`, `risk.py:441-486`, `strategy_engine.py:853-867` vs `1063-1138` |
| **C7** | **Orphan-recovery herstart live runs automatisch bij API-boot** zonder confirm-token, met lege posities en verse `PortfolioAccounting`. Geen reconcile van balances/posities bij start. Paper-posities gaan verloren bij restart. `POST /runs/{id}/stop` flattent niet: crypto blijft op Coinbase staan. | `runs.py:1931-2173` (live 2094-2103, 2168), `runs.py:1161-1226`, `live.py:1117-1151` |
| **C8** | Config risk-defaults (`default_max_daily_loss_pct` etc.) worden nergens gelezen; `RiskParameters` niet per run instelbaar via API. | `config.py:327-337`, `risk.py:100-115` |
| **C14** | Exposure-cap negeert de nieuwe order (kan ~75% i.p.v. 60%); cluster-cap gebruikt `order.price` = `None` bij market orders → overgeslagen voor nieuwe symbolen; ATR-sizing onbereikbaar (niemand geeft `atr_value` door); risk-ceiling bindt nooit (1% default stop → 2× equity). | `risk.py:530, 557-563`, `risk_manager.py:31, 182-195, 212` |
| **C15** | Brackets alleen engine-side, op close, één keer per poll (daily = 1×/24u); geen exchange-native stops. ATR-levels worden **elke bar herberekend** (docstring zegt "anchored"). Ongeldige bracket-config → warning + brackets uit → BUY-only strategie heeft dan géén exit. `momentum_breakout` is niet positie-bewust (kan bij elke nieuwe breakout bijkopen). | `bracket_exit.py:378-387`, `strategy_engine.py:296-299, 1075-1079`, `momentum_breakout.py` |
| **C16** | Backtests/optimizer blokkeren de event loop (CPU-bound, geen yield) in de enige worker → paper/live loops én emergency-endpoints staan stil tijdens een backtest. Live `process_signal` doet `asyncio.sleep(2)` per order. | `runs.py:692`, `optimize.py:247`, `live.py:749-751` |
| **C17** | `run_live_loop` vangt elke iteratie-exception en gaat door → systematische live-fout stopt de run nooit. | `strategy_engine.py:753-756` |

### 1.2 Backtest-integriteit (het bewijs achter de live-promotie)

| ID | Bevinding | Bewijs |
|---|---|---|
| **C9** | **Fee-mismatch 4×.** Backtest default 10/15 bps + 5 bps slippage; live/paper 40/60 bps. Geldt voor API-backtests, optimizer en beide research-scripts. Alleen de ML-gate gebruikt 60/40. De momentum_breakout-validatie liep dus op ~¼ van de echte kosten. Alleen market orders → live altijd taker. | `backtest.py:187-189`, `runs.py:675-683`, `optimizer.py:138-140`, `research_momentum_brackets.py:179`, `validate_sl_tp_reversion.py:98`, `risk.py:112-115` |
| **C10** | **Look-ahead:** `_build_mtf_context` injecteert de *huidige* gecachte FGI/CoinGecko/FRED/Whale-waarden in elke historische bar van API-backtests. Raakt `dca_rsi_hybrid` en ModelStrategy v2. Scripts krijgen `None` → zelfde strategie backtest anders in API dan in script. | `strategy_engine.py:1274-1368`, `dca_rsi_hybrid.py:486-494, 533-607` |
| **C11** | Multi-symbol backtests aligneren op **bar-index, niet timestamp**; geen gap-detectie; ongelijke lengtes alleen een warning → symbolen stil verschoven in de tijd. O(n²) kopie van volledige historie per bar. | `strategy_engine.py:659-665`, `backtest.py:808-815`, `ccxt_market_data.py:310-432` |
| **C12** | Equity-punten, fills, posities en trade-tijden gebruiken `datetime.now()` i.p.v. bar-tijd en worden zo gepersisteerd. | `portfolio.py:230, 347`, `paper.py:343`, `strategy_engine.py:1513`, `run_persistence.py:268-319` |
| **C13** | Sharpe/PSR berekend over de equity-curve **inclusief warmup-punten zonder verandering** (206 bars bij daily momentum) én een extra punt per fill → Sharpe naar 0 getrokken, PSR-n opgeblazen, annualisatie klopt niet. | `strategy_engine.py:672-675`, `backtest.py:459`, `metrics.py:433, 894` |
| **S5** | Geen L2/Postgres OHLCV-cache (docstring belooft het); backtests refetchen elke keer van de exchange met `cache_ttl=0` → niet reproduceerbaar. Redis geconfigureerd, nergens gebruikt. CCXT rate-limiter uit. | `market_data.py:10-12`, `runs.py:~245`, `config.py:230`, `ccxt_market_data.py:191` |
| **S6** | Walk-forward validator optimaliseert niet op train-folds (rankt direct op test-folds), folds starten koud (warmup eet test-window op), niet-Sharpe metrics vallen terug op in-sample; API-optimizer zet WF nooit aan; DSR is een vereenvoudigde haircut. | `optimizer.py:320-458`, `walk_forward.py:317`, `optimize.py:225` |

### 1.3 Ops, security, CI

| ID | Bevinding | Bewijs |
|---|---|---|
| **C18** | Caddy stuurt `/api/*` rechtstreeks naar `api:8000`; browser stuurt geen `X-API-Key`; werkt alleen omdat `REQUIRE_API_AUTH=false` (server!). NextAuth-rollen alleen in UI afgedwongen; viewer kan live runs starten/modellen activeren. UI stuurt confirm-token in body (deprecated pad). `ADMIN_API_KEY`/`INTERNAL_ADMIN_API_KEY` niet in compose doorgegeven → UI kill-switch mogelijk 503 (op server verifiëren). `/metrics` publiek. | `infra/Caddyfile`, `apps/ui/src/lib/api.ts:115-119`, `lib/auth.ts:6-14`, `runs/new/page.tsx:259`, `docker-compose.yml` |
| **C19** | **Geen CI:** `.github/workflows/ci.yml` verwijderd (sinds 2026-09-25 gecommit in `364bb99`; zie §11). HEAD-versie waarschijnlijk toch stuk (`next build` vereist `NEXTAUTH_SECRET` sinds Sprint 50); geen jest-job, geen pip-audit/npm-audit/gitleaks. Deploy volledig handmatig (rsync + compose), geen rollback, **geen pg_dump-backup**, geen docker log-rotatie. | `git status`, `infra/docker-compose.yml`, `scripts/` |
| **C20** | Grafana alert-rules bestaan (4), maar geen contact points/notification policies → alerts bereiken niemand. | `infra/grafana/provisioning/alerting/` |
| **S10** | `whale_tracker` logt URL incl. `api_key` bij failure; Telegram-token is `str` i.p.v. `SecretStr`; `NEXTAUTH_SECRET` + Google-secret als build-ARG in UI-image-layers; `GRAFANA_PASSWORD` default `admin`. | `whale_tracker.py:267-291`, `config.py:552`, `Dockerfile.ui:72-84` |

### 1.4 Architectuur, tests, ML, frontend, docs

| ID | Bevinding | Bewijs |
|---|---|---|
| **S1** | God-modules: `runs.py` 2193 regels (`create_run` ~488, 4 engine-start-sites), `strategy_engine.py` 1826 (~15 verantwoordelijkheden, fill-routing 4× gekopieerd), `run_paper_engine`/`run_live_engine` bijna-duplicaten (~300 regels elk), `process_signal` gedupliceerd in paper/live, **drie positie-ledgers** (paper `_positions`, portfolio `_position_snapshots`, live `_positions`). | zie regelnummers in §1.1 |
| **S2** | Module-globale state `_RUN_TASKS`/`_RUN_ENGINES`/`_LEARNING_INSTANCES` gemuteerd vanuit 6 modules zonder lock; `RunRegistry` op `AppContainer` nooit gevuld (`/health` active runs altijd 0); `set_global_client`-singletons "sunset Sprint 41" zijn nog steeds het productiepad; 3 handmatige strategy-registries + 1 dode. Forceert `--workers 1`. | `run_orchestrator.py:152-158, 209`, `main.py:72-112, 485`, `health.py:154-158` |
| **S3** | Dode/niet-aangesloten code: `EnsembleStrategy` (461 regels), `EventBus` (725), `fx_service` (225) + `FxCacheWarmer`-stub die wél in `/health` staat, `RunRegistry`, `CircuitBreaker`, `trigger_kill_switch`, `sync_positions`, `_STRATEGY_REGISTRY`. | — |
| **S4** | ML feitelijk dood: `models/` bevat alleen `.gitkeep`, `ML_AUTO_RETRAIN=false`, ModelStrategy met leeg `model_path` → HOLD, DEMOTED. Training gebruikt geshufflede `train_test_split` op overlappende 5-bar forward labels (leakage → opgeblazen accuracy). ModelStrategy blokkeert in `on_bar` (sidecar-read per bar, `joblib.load` met `future.result(timeout=5)`); sidecar-pad vast op eerste symbool. AdaptiveLearning heeft 50 trades/cyclus nodig (onbereikbaar voor daily momentum). | `ml_training.py:334, 616`, `model_strategy.py:272-279, 323-338, 395-426`, `run_orchestrator.py:841` |
| **S7** | Tests: ~1963 unit + ~243 integratie; integratietests gebruiken `AsyncMock`-sessies, **geen echte Postgres**, migraties nooit getest; geen e2e; bestanden vernoemd naar sprint; `fail_under=80` vs 78 werkelijk; 134× `except Exception` met opvallende swallow-sites (externe signalen, shutdown-handlers). | `tests/integration/conftest.py:1-9`, `strategy_engine.py:915, 1290-1337`, `main.py:515-543` |
| **S8** | Frontend: raw fetch + 5s-polling van 8 endpoints; geen logs/events-tab; geen exchange-balance/reconciliatie-view; `error.tsx` rendert `<html>` (hoort in `global-error.tsx`, ontbreekt); `ErrorBoundary` ongebruikt; mobile zwak; max 100 trades geladen. | `runs/[id]/page.tsx:409-478, 685`, `app/error.tsx` |
| **S9** | Docs verouderd: README (6 i.p.v. 9 strategieën, geen Caddy/Tailscale/auth), ARCHITECTURE.md (noemt geshipte items "open"), docs/*.md uit februari, handleiding pre-auth. Geen deploy/backup/secret-rotatie/incident-runbook; live-readiness checklist staat niet in de repo. | `README.md`, `ARCHITECTURE.md:311-312` |
| **S11** | Strategie-kwaliteit: grid/dca/rsi ACTIVE+live-eligible zonder OOS-bewijs; forward-test bevestigde geen edge; grid 1% spacing < 1,3% live round-trip → verliest per constructie; grid recentert nooit. Verbeterplan Fase 2 (synthetische regime-harness + invarianten) nooit gebouwd. | `strategy_availability.py:84-167`, register #41 |

---

## 2. Kernconclusie en aanbeveling

**Live run `f36fb44e` heeft geen werkende exit (C1), foutieve equity/sizing (C4) en risk gates die de exit kunnen blokkeren (C6/C15).** Omdat het een daily strategie is die om 14:13 UTC is gestart en bij start geen breakout-conditie had, is er **tot ~2026-09-25 14:13 UTC nog geen BUY geweest**: de account is flat en stoppen kost nu niets. Dat venster sluit binnen 24 uur.

**Aanbeveling (beslissing D1, van de gebruiker):** stop de live run vóór de eerste bar-evaluatie, en herstart pas na Fase 1-minimumset + mechanics-test + Fase 2-hervalidatie. Alternatieven (hotfix binnen 24u; niets doen) zijn in §3 uitgewerkt en afgewezen. Stoppen/starten van live runs is en blijft een handeling van de gebruiker; het plan voert dit niet zelf uit.

---

## 3. Fase 0 — Directe live-veiligheidsrespons (vandaag)

| Optie | Betekenis | Voor | Tegen |
|---|---|---|---|
| **A. Nu stoppen (aanbevolen)** | `DELETE /runs/f36fb44e` (of UI-stop). Account is flat, niets af te wikkelen. Herstart na Fase 1-minimumset. | Nul kapitaal in gevaar; geen tijdsdruk op de agent-keten; vermijdt deploy onder een live run (C7). | ~1-2 weken forward-test-tijd verloren; paper-bewijs loopt gewoon door. |
| B. Hotfix sell-pad <24u | Patch `live.py` + deploy, run laten lopen. | Continuïteit. | Deploy herstart de API → orphan-recovery herstart de run met lege posities (C7). `sync_positions` heeft zelf twee latente bugs (base-asset→eerste markt met die base, dus mogelijk `BTC/USD` i.p.v. `BTC/EUR`; externe posities krijgen `average_entry_price=0` → ATR-TP ≈ 0+3·ATR → **onmiddellijke valse "take profit"** na elke restart). C4/C6 blijven. Niet veilig haalbaar in één cyclus. |
| C. Niets doen | Hopen dat geen breakout vuurt. | — | Elke BUY creëert een positie zonder exit, gesized op de hele EUR-balance. Onacceptabel. |

**Checklist Fase 0 (gebruiker + één read-only agent-pass, ~0,5 cyclus, geen codewijzigingen):**
1. Gebruiker stopt `f36fb44e` (eigen beslissing, eigen handeling).
2. Verifieer op Coinbase én via `GET /runs/f36fb44e/positions|orders` dat er geen crypto uit deze run en geen open orders zijn. Zo wel: handmatig verkopen op Coinbase; de bot kan het niet.
3. Op de server (Tailscale, read-only): aanwezigheid `ADMIN_API_KEY`/`INTERNAL_ADMIN_API_KEY` in de api-container-env (C18), Grafana contact points (C20), `REQUIRE_API_AUTH` waarde. Vastleggen; fixen in Fase 3.
4. Interim-regel tot Fase 1 DoD: **geen live runs**, alleen paper. Deploy-beleid: alleen deployen als er geen live run bestaat (D12).

**Herstart-protocol (na merge+deploy van de Fase 1-minimumset):**
1. **Integratietest-gate:** `tests/integration/test_live_protective_paths.py` groen (WP1.0/1.1): echte `LiveExecutionEngine` + echte `StrategyEngine` + `PortfolioAccounting`, ccxt vervangen door een in-memory fake exchange; scenario's BUY→SL, BUY→TP, BUY→trailing, BUY→kill-switch, BUY→API-restart→reconcile→SL.
2. **Mechanics-test op de echte exchange (D2):** dedicated `smoke_roundtrip`-strategie (BUY op eerste bar, SELL op de volgende; 1m/5m) op één pair (bv. XRP-EUR of LTC-EUR), `initial_capital=€10`, notional ≈ €8-10 (> Coinbase min-notional). Slaagt als: BUY zichtbaar op Coinbase en in `/orders`, fill vastgelegd, positie getoond, SELL geplaatst door de *engine* (niet handmatig), fill vastgelegd, portfolio flat, `sync_positions` == exchange-balance, realised PnL ≈ −(2×taker + spread). Daarna run stoppen en "stop terwijl holding"-gedrag nogmaals bevestigen.
3. **Shadow:** paper run met identieke momentum_breakout-config naast de live run; 2 weken dagelijks signalen vergelijken (divergentie = bug).
4. Live herstart op €100 pas na 1-3 én een positieve Fase 2 go/no-go (WP2.8) — of, als de gebruiker eerder wil (D14), op z'n vroegst na WP1.0-1.8 met de expliciete erkenning dat de edge op ¼ van de echte kosten is gevalideerd.

---

## 4. Fasen en werkpakketten

**Cyclus-definitie:** één volledige verplichte pass voor één WP (producer → code-critic [+ security-audit-specialist bij exchange/env-code] → code-executor → testing-quality-specialist), inclusief één rejection-ronde. Escalatie naar synthesis-arbiter na 2 rejections niet meegeteld. Git-checkpoint per WP: `claude-session-<timestamp>-<wp-id>`. Rapporten in `reports/vp2-<wp-id>-*.md` volgens `.claude/template/report.md`.

**Agent-ketens (uit CLAUDE.md):** Risk/safety-WP's = `[risk + trading-arch] → python → [critic + security] → synthesis → executor → testing`; backtest-WP's = `[quant + trading-arch] → synthesis → python → critic → executor → testing`; Docker/CI = `devops → security → critic → executor → testing`; UI = `nextjs → critic → executor → testing`; architectuur-WP's krijgen vooraf `architecture-critic`.

### Fase 1 — Live-veiligheid & risk-correctheid (~20 cycli)

**Doel:** elk beschermend pad (SL, TP, trailing, kill-switch, circuit breaker, restart) werkt end-to-end met de *echte* live engine; sizing en risk-getallen kloppen in live.

| WP | Titel | Bestanden | Acceptatie / tests | Cycli | Afh. | QW |
|---|---|---|---|---|---|---|
| **1.0** | FakeExchange-fixture + live protective-path harness | `tests/integration/conftest.py` (nieuw `FakeCCXTExchange`: balances, markets met meerdere quotes per base, `create_order`/`fetch_order`/`fetch_balance`/`fetch_ticker`/`fetch_ohlcv` met gescript prijspad, min-notional/precision-fouten), `tests/integration/test_live_protective_paths.py` | Harness draait `StrategyEngine(run_mode=LIVE)` met echte `LiveExecutionEngine`, zonder `_positions`-injectie. Geparametriseerde scenario-tabel; **faalt vandaag op elk SELL-scenario** (bewijst C1). | 1,5 | — | |
| **1.1** | Live positie-ledger fix | `packages/trading/engines/live.py` | (a) BUY/SELL-fills updaten `_positions`, óf `process_signal` leest held qty uit `PortfolioAccounting` via geïnjecteerde callback (keuze: portfolio = single source of truth, `_positions` wordt cache). (b) `sync_positions` beperkt tot run-symbolen + quote-currency; nooit posities met entry 0 — onbekende entry → vlag `reconcile_required`, niet bracket-geëvalueerd. (c) Startup-`sync_positions` in `run_live_engine`. (d) Balance-cache invalideren na fill. Tests: alle 1.0 SELL-scenario's groen; unit test base→market-mapping met BTC/USD+BTC/EUR aanwezig. | 2 | 1.0 | |
| **1.2** | Side-aware risk gates | `risk_manager.py:88-122`, `risk.py:441-486`, `strategy_engine.py:853-867` | Exposure-reducerende SELLs (qty ≤ held) omzeilen daily-loss, cooldown, drawdown, max-open-positions. Kill-switch in LIVE: blokkeert entries, bracket/trailing-secties draaien door (of flatten, per D3). Tests: SELL passeert elke gate terwijl BUY geblokkeerd; kill-switch+SL-scenario in harness; SELL-zonder-positie blijft geweigerd. | 2 | 1.0 | |
| **1.3a** | Bracket-config hard-fail + geen pyramiding | `runs.py` (create_run-validatie), `strategy_engine.py:285-300`, `momentum_breakout.py` of engine-config `allow_pyramiding=False` | Ongeldige bracket-config → 422 bij run-creatie (nooit "warning + uit"). BUY-only strategieën declareren `requires_exit_manager=True`; run-creatie weigert zonder bracket/trailing. Engine slaat BUY over voor reeds gehouden symbolen tenzij pyramiding aan. Tests: API-422-cases; harness "twee breakouts, één positie". | 1,5 | — | QW (validatiedeel) |
| **1.3b** | ATR-bracket anchoring + intra-bar exit-check | `bracket_exit.py`, `strategy_engine.py` (live loop) | Levels verankerd bij entry (zoals docstring zegt), opgeslagen bij positie-open, niet herberekend. Live loop: optionele `exit_check_interval_s` (bv. 300 s) die tussen bars tickers pollt voor gehouden symbolen en bracket/trailing checkt; bar-close-logica ongewijzigd voor backtest/paper. Tests: level-immutabiliteit; harness "SL mid-bar doorbroken → SELL binnen één interval". | 2 | 1.1, 1.2 | |
| **1.4** | Live equity = NAV; sizing-basis | `live.py:975-1007`, `_resolve_order_quantity` | Equity = quote-cash + Σ(held qty × last) voor run-symbolen; drawdown/peak op NAV; sizing-basis = `min(NAV, initial_capital)` (of hele account, per D5). Tests: twee 15%-posities ⇒ 0% drawdown; sizing gecapt op run-kapitaal. | 1,5 | 1.1 | |
| **1.5** | Daily-PnL & peak-semantiek | `paper.py:907-921`, `live.py:1031-1044`, `risk.py:469-484`, `portfolio.py` | Daily = realised (+ optioneel unrealised) sinds 00:00 UTC, reset op daggrens (bar-tijd in backtest); cumulatief verlies via drawdown-gate, niet daily-gate; paper-peak = running max van NAV. Tests: 8% verlies op dag 1 blokkeert dag 2 niet; peak volgt. | 1,5 | — | |
| **1.6** | CircuitBreaker aansluiten met getypeerd summary-contract | `run_orchestrator.py:816-830, 1133-1185`, `strategy_engine.py:877-885`, `portfolio.py:622-642`, `circuit_breaker.py`-router | `PortfolioSummary`-dataclass/TypedDict aan beide kanten (geen string-keys). Breaker geïnstantieerd voor paper+live. Live-semantiek (D4): REDUCE = halve sizing; HALT = entries blokkeren, exits behouden, alert, **geen auto-stop**; DAILY_LIMIT per 1.5. Debounce: level-wissel vereist N opeenvolgende polls. Endpoint geeft echte state. Tests: state-machine op echte summary; harness "HALT, daarna SL vuurt nog". | 2 | 1.2, 1.5 | C3-deel QW |
| **1.7** | Kill-switch + stop-met-flatten | `runs.py` (emergency-stop, stop_run), `risk_manager.py`, UI stop-dialoog | `trigger_kill_switch` aangeroepen door emergency-stop; `stop_run?flatten=true` plaatst market SELLs voor gehouden run-symbolen (live) vóór task-cancel; default bij live-stop = *vragen* in UI — nooit stil crypto laten staan. Tests: harness "emergency-stop terwijl holding → flat". | 1,5 | 1.1, 1.2 | |
| **1.8** | Orphan-recovery & reconcile | `runs.py:1931-2173`, `run_orchestrator.py`, `portfolio.py` (nieuw `PortfolioAccounting.from_fills`) | Bij boot: live runs **niet** auto-herstart; status → `orphaned`, vereist `POST /runs/{id}/resume` met confirm-token (header). Resume: portfolio herbouwen uit gepersisteerde fills, `sync_positions`, qty vergelijken; mismatch → `reconcile_required` (geen trading, alert). Paper: herbouw uit fills, auto-resume toegestaan. Tests: harness "BUY → restart → resume → SL vuurt met juiste entry-prijs"; mismatch-pad. | 2,5 | 1.1 | |
| **1.9** | RiskParameters per run + config-defaults | `config.py:327-337`, `risk.py`, `runs.py`-schemas, `run_orchestrator.py` | Defaults uit config; API accepteert `risk_params` (begrensd, gevalideerd); gepersisteerd in run-config; zichtbaar in UI. Tests: default-plumbing; bounds. | 1 | — | QW (defaults) |
| **1.10** | Sizing-wiskunde | `risk.py:530, 557-563`, callers in `paper.py`/`live.py` | Exposure-cap incl. kandidaat-order; cluster-cap gebruikt last price bij market orders; ATR-sizing óf aangesloten (`atr_value` vanuit engine) óf verwijderd; risk-ceiling gebruikt de werkelijke bracket-SL-afstand. Tests: tabelgedreven. | 1,5 | 1.3b | deels QW |
| **1.11** | Loop-resilience | `strategy_engine.py:715-756`, `live.py` order-wait | Exception-classificatie: transient → backoff + metric `engine_iteration_errors_total`; ≥N opeenvolgend → run-status `error` + alert; nooit stil. `asyncio.sleep(2)` vervangen door gepollde order-status (timeout, partial-fill). Backtest-loop: `await asyncio.sleep(0)` elke N bars als stopgap voor C16 (volledige offload in 4.6). Tests: fault-injection in harness. | 1,5 | 1.0 | sleep(0) QW |

**Minimumset vóór enige live-herstart:** 1.0, 1.1, 1.2, 1.3a, 1.4, 1.7, 1.8 (1.8 omdat elke deploy de API herstart). Aanbevolen erbij: 1.3b, 1.5, 1.6.
**Definition of Done Fase 1:** elke rij van de protective-path-scenariotabel groen met de echte live engine; 100% van de live SELL-paden gedekt; geen run aan te maken zonder exit-mechanisme; API-restart met gehouden positie kan niet handelen tot gereconcilieerd; mechanics-test (Fase 0 stap 2) geslaagd.

### Fase 2 — Backtest-integriteit & hervalidatie (~14 cycli)

**Producers:** quant-strategy-analyst (metrics, validatie), data-pipeline-engineer (cache, alignment, MTF), database-architect (OHLCV-schema), python-backend-specialist.

| WP | Titel | Bestanden | Acceptatie | Cycli | Afh. | QW |
|---|---|---|---|---|---|---|
| **2.1** | Eén kostenmodel | nieuw `packages/trading/costs.py` (`CostProfile`: maker/taker bps, slippage bps, per exchange/tier); `backtest.py:187-189`, `runs.py:675-683`, optimizer, research-scripts, ML-gate | Eén default (Coinbase Advanced: 40/60 bps + 10 bps slippage, D6); elke run persisteert `cost_profile` in metadata; leaderboard/compare tonen het; oude runs gevlagd "legacy costs". Tests: pariteitstest backtest-fees == live-fee-constanten voor hetzelfde profiel. | 1,5 | — | |
| **2.2** | Bar-tijd-klok | `portfolio.py:230, 347`, `paper.py:343`, `strategy_engine.py:1513` | Fills/equity/trades gestempeld met bar-timestamp in backtest (gesimuleerde klok geïnjecteerd); wall-clock alleen paper/live. Tests: gepersisteerde timestamps monotoon en gelijk aan bar-tijden. | 1,5 | — | |
| **2.3** | Timestamp-alignment multi-symbol | `strategy_engine.py:659` | Alignen op timestamp (inner join of expliciet beleid: bar overslaan / ffill met vlag); gap-detectie logt + metric; misalignment > tolerantie breekt backtest af. Tests: symbolen met ontbrekende bars. | 1,5 | — | |
| **2.4** | Point-in-time MTF-context | `strategy_engine.py:1274-1368`, data-services FGI/CoinGecko/FRED/Whale | Historische series opgeslagen met `as_of`; backtest doet as-of-join met lag; zonder historie is de feature in backtest uit en registreert de run `mtf_context=unavailable`. Tests: waarde op bar t hangt nooit af van data na t. | 2 | 2.6 | |
| **2.5** | Metric-integriteit | `strategy_engine.py:672-675`, `backtest.py:459`, `metrics.py` | Equity één keer per bar gesampled ná warmup; Sharpe geannualiseerd per timeframe; PSR/DSR-n = bars (niet punten); referentiewaarden uit bekende synthetische reeks. | 1 | 2.2 | |
| **2.6** | OHLCV-cache (Postgres) + rate-limiter | nieuwe migratie + tabel `ohlcv_bars`, cache-laag in `packages/data`, `CCXTMarketDataService`, `enable_rate_limit=True` | Backtests lezen uit DB, backfill bij miss, `data_hash` per run gepersisteerd → reproduceerbaar; Redis: verwijderen of gebruiken (D10). Tests: integratie met echte Postgres (vereist 4.8 of lokale compose-Postgres). | 2,5 | — | rate-limiter-flag QW |
| **2.7** | Walk-forward-correctheid | validator-module, `optimize.py`, DSR | Optimaliseren op train-folds, selecteren, evalueren op test; warmup pre-roll vóór elk test-window; anchored + rolling; DSR per Bailey/López de Prado met #trials; API-optimizer zet WF default aan. Tests: synthetische strategie met bekende OOS-decay. | 2,5 | 2.5 | |
| **2.8** | **Hervalidatie momentum_breakout** (analyse, geen code) | `Documentation/validation-momentum-breakout-2026-Q4.md` | Backtests op 40/60+10 bps, gealigneerde bars, WF-OOS, ≥2 jaar, 6 EUR-pairs, kostengevoeligheid (60/80/100 bps), vergelijking met 3-maanden-paper. **Go/no-go:** OOS PF ≥ 1,3, ≥ 30 OOS-trades, max DD ≤ 25%, positief in ≥ 3 van 4 folds, PF > 1,1 bij 80 bps, DSR > 0. No-go → run blijft paper; kapitaal blijft 0. | 1,5 | 2.1-2.3, 2.5, 2.7 | |

**DoD:** fee-pariteitstest groen; deterministische backtest (zelfde `data_hash` → zelfde resultaten); geen wall-clock-timestamps in backtest-tabellen; MTF-look-ahead-test groen; validatierapport met expliciete go/no-go, door gebruiker afgetekend.

### Fase 3 — Ops, security, CI (~10 cycli)

**Producers:** devops-infrastructure-specialist, python-backend-specialist, nextjs-frontend-specialist; security-audit-specialist op elk WP.

| WP | Titel | Bestanden | Acceptatie | Cycli | Afh. | QW |
|---|---|---|---|---|---|---|
| **3.1** | CI herstellen | `.github/workflows/ci.yml` (herstel + uitbreiden), `apps/ui/package.json` | Jobs: ruff, mypy, pytest unit (coverage-gate = huidige 78, daarna ratchet), pytest integratie met Postgres-service, jest, tsc, `next build` met dummy `NEXTAUTH_SECRET`, `pip-audit`, `npm audit --audit-level=high`, gitleaks, docker build. Required op main. | 1,5 | — | herstel-alleen QW |
| **3.2** | Server-side auth voor UI→API | Caddyfile, `apps/ui` route-handlers (proxy die API-key + door NextAuth gesigneerde rol-header injecteert), API auth-dependency met rollen, `REQUIRE_API_AUTH=true`, confirm-token alleen header | Browser bezit nooit de API-key; viewer kan geen live runs/modellen activeren (403 op API); `/metrics` beperkt tot Prometheus-netwerk; e2e-test via Caddy. | 3 | 3.1 | |
| **3.3** | Secrets-hygiëne | `whale_tracker`, Telegram-settings (`SecretStr`), UI-Dockerfile (runtime-env, geen build-arg-secrets), compose (`GRAFANA_PASSWORD` verplicht, `ADMIN_API_KEY`/`INTERNAL_ADMIN_API_KEY` doorgegeven), gitleaks-baseline | Geen secret in logs of image-layers (geverifieerd via `docker history` + gitleaks). | 1 | — | QW |
| **3.4** | Alerting | Grafana contact points (Telegram/e-mail), alert-rules: heartbeat ontbreekt, loop-errors, HALT, exchange-errors, orphaned run, `reconcile_required`, backup mislukt | Testalert ontvangen; runbook-links in alert. | 1 | 1.6, 1.11 | QW |
| **3.5** | Backups + log-rotatie | compose `pg_backup`-cron-container (dagelijkse dump, 14 dagen retentie, off-box kopie), `scripts/restore.sh`, docker `json-file` log-limits | Restore-oefening gedocumenteerd: RTO ≤ 30 min, RPO ≤ 24 u. | 1 | — | log-rotatie QW |
| **3.6** | Deploy + rollback | `scripts/deploy.sh` (getagde images in CI → pull op server, of rsync+build), pre-flight "geen live run met posities", health-gate, `rollback.sh` naar vorige tag, deploy-runbook | Rollback geoefend < 10 min. | 2 | 3.1, 1.8 | |

**DoD:** CI required en groen; `REQUIRE_API_AUTH=true` in prod; alert end-to-end bezorgd; restore en rollback geoefend met tijden vastgelegd.

### Fase 4 — Architectuurschuld (~18 cycli)

**Producers:** trading-engine-architect + architecture-critic (ontwerpreview vóór code-critic), python-backend-specialist, database-architect. **Pas starten als de Fase 1-harness bestaat** (dat is de regressie-gate voor elke refactor).

| WP | Titel | Bestanden | Acceptatie | Cycli | Afh. |
|---|---|---|---|---|---|
| **4.1** | `ExecutionEngineBase` + één positie-ledger | `engines/base.py` (nieuw), `paper.py`, `live.py`, `portfolio.py` | Gedeelde `process_signal` (gate → size → submit → record); abstracte `_submit`; `PortfolioAccounting` is de enige ledger (paper `_positions`, live `_positions`, `_position_snapshots` weg). Protective-path-harness ongewijzigd groen. | 4 | Fase 1 |
| **4.2** | Orchestrator/`create_run`-decompositie | `run_orchestrator.py`, `runs.py` | Eén `run_engine(mode, factory)`; `create_run` gesplitst in validate/build/start (<100 regels elk); 4 start-sites → 1. | 3 | 4.1 |
| **4.3** | Eén strategy-registry | `strategies/registry.py`; verwijder `_STRATEGY_REGISTRY` + 2 andere; API, UI, optimizer lezen het | Registry draagt status (ACTIVE/DEMOTED), live-eligible-vlag, param-schema, `requires_exit_manager`. | 1,5 | — |
| **4.4** | RunRegistry-adoptie | `AppContainer`; `_RUN_TASKS`/`_RUN_ENGINES`/`_LEARNING_INSTANCES` weg; lock; `/health` | `/health` toont echte actieve runs; geen module-globale mutable dicts. | 2 | 4.2 |
| **4.5** | Dode code verwijderen | EnsembleStrategy, EventBus, fx_service/FxCacheWarmer-stub, `set_global_client`-sunset | −1,4k regels; `/health` rapporteert geen nep-componenten meer. | 1 | 4.4 |
| **4.6** | Backtest/optimizer-offload | `runs.py:692`, `optimize.py:247`; `ProcessPoolExecutor` of aparte `worker`-compose-service (keuze) | Emergency-endpoints reageren < 200 ms tijdens lopende optimalisatie (load-test). | 2,5 | 4.2 |
| **4.7** | `strategy_engine.py`-decompositie | fill-routing-helper (4× gekopieerd), `ExitRunner` (bracket/trailing), `MtfContextBuilder`, metrics-recorder | Bestand < 800 regels; geen gedupliceerde fill-loop. | 2,5 | 4.1 |
| **4.8** | Echte-DB-integratietests + test-hygiëne | `tests/integration` op Postgres (compose/testcontainers), migraties in CI, sprint-bestandsnamen → feature-namen, swallow-site-audit (134× `except Exception`) | Migraties in CI uitgevoerd; elke brede except re-raist, logt met `exception`, of heeft een verantwoordend commentaar. | 2 | 3.1 |

**DoD:** één engine-base, één ledger, één registry, geen module-globals; `--workers 1` niet langer afgedwongen door state (wél nog single worker tot de 4.6 worker-service).

### Fase 5 — Strategie-onderzoek & harness (~9,5 cycli)

**Producer:** quant-strategy-analyst; testing-quality-specialist voor de harness.

| WP | Titel | Acceptatie | Cycli | Afh. |
|---|---|---|---|---|
| **5.1** | Synthetische regime-harness (Verbeterplan Fase 2) | Generators: trend op/neer, mean-revert, chop, crash+gap, low-vol drift; invarianten per strategie: nooit SELL zonder positie, geen NaN/inf, exit vuurt binnen bracket, signaalaantal begrensd, fee-bewust breakeven; verwacht-gedrag-matrix (bv. momentum wint in trend, verliest ≤ X in chop). | 2,5 | 2.1, 2.5 |
| **5.2** | grid/dca/rsi terugtrekken (D9) | Status DEMOTED, niet live-eligible, uit UI create-run voor live; code gearchiveerd of verwijderd. | 0,5 | 4.3 |
| **5.3** | Cross-sectioneel momentum-kandidaat | Rank EUR-pairs op 30/90-daags rendement, hold top-k, wekelijkse rebalance, vol-target, kostenbewuste turnover-cap; WF-validatie op 60/40 bps; promotie naar paper. | 3,5 | 2.7, 5.1 |
| **5.4** | momentum_breakout v2 (na hervalidatie) | Positie-bewustzijn (1.3a), BTC 200-daags regime-filter, time-stop; WF-vergelijking vs v1. | 2 | 2.8 |
| **5.5** | Promotiebeleid gecodificeerd | Paper→live gates in promotion-module: ≥ 30 trades, ≥ 3 maanden, PF ≥ 1,3 na kosten, DD ≤ 25%, harness groen. | 1 | 5.1 |

**DoD:** elke ACTIVE strategie doorstaat de harness en heeft WF-OOS-bewijs op echte kosten; geen live-eligible strategie zonder.

### Fase 6 — ML (beslissingspoort D8)

**Aanbeveling: bevriezen.** Er bestaat geen getraind model, ModelStrategy is DEMOTED, labels lekken, en AdaptiveLearning heeft 50 trades/cyclus nodig die een daily strategie nooit produceert. De cycli zijn beter besteed aan Fase 5.

| Optie | Werk | Cycli |
|---|---|---|
| **Bevriezen (aanbevolen)** | `ModelStrategy`, training-pipeline, learning-endpoints, `AdaptiveLearningTask` achter feature-flag (default uit); uit UI/health; code op branch bewaren; documenteren. | 1 |
| Fixen | Purged K-fold met embargo; triple-barrier-labels; reproduceerbaar train-script (seed, `data_hash`); per-symbol sidecar; async model-load bij start (geen I/O in `on_bar`); smoke-test op synthetische data; pas dan een paper run. | 5-6 |

### Fase 7 — Frontend & docs (~6,5 cycli)

| WP | Titel | Acceptatie | Cycli |
|---|---|---|---|
| **7.1** | Data-laag | SWR/React Query; één `/runs/{id}/snapshot`-endpoint of SSE i.p.v. 8×5s-polling; paginatie voor trades. | 2 |
| **7.2** | Run-detail: logs/events-tab, exchange-balance + reconciliatie-view (toont `reconcile_required`), circuit-breaker/kill-switch-status | Live-operator ziet engine-status zonder SSH. | 2 |
| **7.3** | Error-handling + mobile | `global-error.tsx`, `error.tsx` gefixt, `ErrorBoundary` gebruikt, responsive run-lijst/detail. | 1 |
| **7.4** | Docs | README (9 strategieën), ARCHITECTURE.md actueel, runbooks: deploy, rollback, backup/restore, secret-rotatie, live-incident (incl. "stop terwijl holding"), live-readiness-checklist in repo. | 1,5 |

---

## 5. Quick wins (< 1 cyclus elk; bundelen in 2-3 QW-cycli)

| # | Item | Bevinding |
|---|---|---|
| Q1 | Summary-sleutelnamen via getypeerd summary-object | C3 |
| Q2 | `ADMIN_API_KEY`/`INTERNAL_ADMIN_API_KEY` in compose doorgeven; op server verifiëren | C18 |
| Q3 | Config risk-defaults inlezen in `RiskParameters` | C8 |
| Q4 | Grafana contact point + één testalert | C20 |
| Q5 | Docker log-rotatie in compose | C19 |
| Q6 | `api_key` redacten in whale_tracker-log; `SecretStr` voor Telegram | S10 |
| Q7 | `ci.yml` herstellen met dummy `NEXTAUTH_SECRET` voor build | C19 |
| Q8 | Ongeldige bracket-config weigeren bij run-creatie (422) | C15 |
| Q9 | Exposure-cap incl. kandidaat-order; cluster-cap gebruikt last price | C14 |
| Q10 | `enable_rate_limit=True` op ccxt-clients | S5 |
| Q11 | `await asyncio.sleep(0)` elke N bars in backtest-loop | C16 |
| Q12 | `FxCacheWarmer`-stub uit `/health`; `/metrics` afschermen in Caddy | S3, C18 |
| Q13 | Coverage-gate naar 78 (later ratchet) zodat CI groen kan | S7 |

---

## 6. Beslissingen die alleen de gebruiker kan nemen

| ID | Beslissing | Aanbeveling |
|---|---|---|
| **D1** | Live run `f36fb44e` nu stoppen? | **Ja, vóór 2026-09-25 14:13 UTC** (account is nog flat). |
| D2 | Mechanics-test met €10 echt geld vóór €100-herstart? | Ja. |
| D3 | Kill-switch-semantiek in live: alleen entries blokkeren, of flatten? | Entries blokkeren + exits behouden; flatten als expliciete tweede actie. |
| D4 | Circuit-breaker HALT in live: run auto-stoppen of entries blokkeren + alert? | Entries blokkeren + alert (auto-stop zonder flatten laat crypto verweesd achter). |
| D5 | Live sizing-basis: hele account of `min(NAV, initial_capital)`? | Run-kapitaal. |
| D6 | Kostenprofiel: Coinbase Advanced (40/60 bps) + 10 bps slippage? Werkelijke 30-daagse tier bevestigen. | 40/60 + 10. |
| D7 | Exchange-native stop-loss-orders? | Nu niet: Coinbase Advanced via ccxt kent stop-limit maar geen OCO; TP blijft engine-side; ATR-herberekening geeft cancel/replace-churn. Herzien na 1.3b intra-bar-checks. |
| D8 | ML: bevriezen / fixen / verwijderen | Bevriezen. |
| D9 | grid/dca/rsi uit live-eligibility halen | Ja. |
| D10 | Redis: verwijderen of inzetten voor OHLCV/rate-limit-cache | Verwijderen tenzij 2.6 ervoor kiest. |
| D11 | UI→API-auth: Next.js server-side proxy (aanbevolen) vs API-key in browser | Proxy. |
| D12 | Deploy-beleid: alleen als geen live run posities houdt (tot 1.8 + 3.6) | Ja. |
| D13 | Vergelijkbaarheid historische runs na fee-wijziging: baselines herdraaien of legacy taggen | Legacy taggen, 3 relevante baselines herdraaien. |
| D14 | Live herstarten vóór Fase 2-hervalidatie (¼-kosten-validatie accepteren) of erna? | Erna. |

---

## 7. KPI's en totale Definition of Done

| KPI | Doel |
|---|---|
| Live protective paths gedekt door integratietests met echte live engine + fake ccxt | 100% (SL, TP, trailing, kill, HALT, restart/reconcile, stop-met-flatten) |
| Fee-pariteit | backtest/optimizer/ML-gate gebruiken hetzelfde `CostProfile` als live; pariteitstest in CI |
| Backtest-determinisme | zelfde `data_hash` + params → identieke metrics |
| CI | required op main; ruff, mypy, pytest unit+integratie (Postgres), jest, tsc, next build, pip-audit, npm audit, gitleaks |
| Auth | `REQUIRE_API_AUTH=true`; viewer-rol kan niets muteren via API |
| Alerting | live-run heartbeat/HALT/error-alerts bereiken een mens |
| Backup | dagelijkse pg_dump; restore geoefend; RTO ≤ 30 min, RPO ≤ 24 u |
| Deploy | gescript met rollback < 10 min |
| Code | geen module-globale run-state; één engine-base; één ledger; één registry; `strategy_engine.py` < 800 regels |
| Strategie | elke live-eligible strategie: WF-OOS op echte kosten + harness groen + ≥ 3 maanden paper |

---

## 8. Inspanning en volgorde

| Fase | Cycli |
|---|---|
| 0 | 0,5 (+ gebruikershandelingen) |
| 1 | ~20 |
| 2 | ~14 |
| 3 | ~10 |
| 4 | ~18 |
| 5 | ~9,5 |
| 6 | 1 (bevriezen) / 5-6 (fixen) |
| 7 | ~6,5 |
| QW-bundels | ~2,5 |
| **Totaal** | **~82 (ML bevriezen) – ~87 (ML fixen)** |

**Volgorde:** Fase 0 → Fase 1-minimumset → QW-bundel → rest Fase 1 ∥ 3.1/3.3/3.5 → Fase 2 → herstart-beslissing → rest Fase 3 → Fase 4 → Fase 5 → Fase 7 (runbooks landen eerder, met 3.5/3.6).

---

## 9. Risico's en afwegingen

| Risico | Mitigatie |
|---|---|
| Circuit breaker aansluiten op slechte/late data halt of stopt de live run | HALT stopt in live nooit automatisch; debounce N polls; alleen realised inputs; alert i.p.v. actie. |
| `sync_positions` mapt base-asset op niet-EUR-markt of entry-prijs 0 → onmiddellijke valse TP na restart | 1.1 beperkt tot run-symbolen/quote; onbekende entry ⇒ `reconcile_required`, geen bracket-evaluatie; 1.8 herbouwt entry uit fills. |
| Fee-defaults wijzigen maakt alle historische vergelijkingen ongeldig | `cost_profile` per run persisteren; legacy taggen; baselines herdraaien (D13). |
| Exchange-native stops op Coinbase Advanced: geen OCO, ccxt-param-eigenaardigheden, cancel/replace-churn bij ATR-levels | Uitstellen (D7); intra-bar engine-checks (1.3b). |
| Elke deploy herstart de API terwijl een live run posities houdt | D12-beleid + 1.8 (geen auto-resume, reconcile) + 3.6 pre-flight-check. |
| Side-aware gates laten een buggy SELL door | "Exposure-reducerend" = qty ≤ held; harness dekt SELL-zonder-positie-weigering. |
| Auth-hardening (3.2) breekt UI in prod | Achter `REQUIRE_API_AUTH`-toggle; e2e-test via Caddy vóór omzetten. |
| Grote refactors (4.1/4.2) regresseren live-gedrag | Protective-path-harness is de regressie-gate; refactors pas ná Fase 1-harness. |
| Hervalidatie geeft no-go voor momentum_breakout | Dan is er geen live-eligible strategie; kapitaal blijft paper — dat is de juiste uitkomst, geen falen van het plan. |
| Agent-keten-doorvoer (2 rejections → arbiter) blaast schattingen op | Schattingen bevatten één rejection-ronde; kleine WP's (≤ 2 cycli) verkleinen het rejection-oppervlak. |
| Repo zonder CI (verwijdering gecommit in `364bb99`) | Q7 herstelt het bestand vanuit `364bb99^`; tot die tijd lokaal `ruff`/`mypy`/`pytest` vóór elke merge. |

---

## 10. Verificatie van dit plan zelf

- Elke C-bevinding is met bestand+regel onderbouwd; C1 is bovendien door de primaire agent zelf geverifieerd (`grep` op alle schrijvers van `_positions` in `live.py`: alleen `sync_positions`; geen productie-caller).
- Aannames die bij uitvoering eerst te checken zijn: (a) er is inderdaad nog geen BUY geweest in `f36fb44e` (check `/runs/f36fb44e/orders`); (b) `ADMIN_API_KEY` ontbreekt echt in de api-container-env op de server (kan via een gitignored override staan); (c) Coinbase-fee-tier van de gebruiker (D6).
- Bij goedkeuring: dit document 1:1 kopiëren naar `Documentation/Verbeterplan-v2-2026-09.md` en register #42 toevoegen in `CLAUDE.local.md` met de D1-beslissing en de Fase 0-checklist. Verder **geen** codewijzigingen tot de gebruiker een fase/WP expliciet vrijgeeft.

### Kritieke bestanden
- `packages/trading/engines/live.py`
- `packages/trading/strategy_engine.py`
- `apps/api/services/run_orchestrator.py`
- `packages/trading/risk.py` + `packages/trading/risk_manager.py`
- `apps/api/routers/runs.py`
- `packages/trading/backtest.py`, `packages/trading/portfolio.py`, `packages/trading/bracket_exit.py`
- `infra/docker-compose.yml`, `infra/Caddyfile`, `.github/workflows/ci.yml`

---

## 11. Aanvulling 2026-09-25 — her-verificatie en nieuwe bevindingen

**Basis:** twee nieuwe onafhankelijke read-only passes: (1) her-verificatie van C1–C9 en de quant-bevindingen met bestand+regel, (2) frontend/infra/tests opnieuw. Geen codewijzigingen.

### 11.1 Her-verificatie
- **Opnieuw bevestigd:** C1–C11, C16 en S1 (`runs.py` 2193 regels, `create_run` `:349-818`, 3 mutable registries, `--workers 1`).
- **Q2 genuanceerd (C6):** portfolio-/cluster-exposure en concentratie gelden alleen voor BUY. Max-open-positions, daily loss, drawdown ≥ 30%, loss-streak-cooldown en de kill-switch blokkeren ook SELLs, en `max_order_size` kapt de SELL-hoeveelheid (`risk_manager.py:375`).
- **Correctie:** de verwijdering van `ci.yml` is gecommit (`364bb99`); zie C19.
- **Tijdkritisch:** de D1-deadline (2026-09-25 14:13 UTC) is vandaag. Controleer eerst de status van run `f36fb44e` (`/runs/f36fb44e/orders`, Coinbase-balance) voordat een ander WP start.

### 11.2 Nieuwe bevindingen

| ID | Bevinding | Bewijs | Opname in plan |
|---|---|---|---|
| **C21** | **Kill-switch werkt niet end-to-end.** Caddy stuurt alleen `/api/auth/*` naar de UI, dus `/api/admin/kill-switch` gaat naar FastAPI (geen route, 404). Daarbij ontbreken `INTERNAL_ADMIN_API_KEY` (UI → 503) en `ADMIN_API_KEY` (API → 401) in compose. Scherpt C18 aan van "mogelijk 503" naar "zeker kapot". | `infra/Caddyfile` (`handle /api/*`), `apps/ui/.../admin/kill-switch/route.ts:62-68`, `apps/api/deps.py:107` | **WP1.7** (Fase 1-minimumset): Caddy `handle /api/admin/*` → UI, env-vars in compose, e2e-test via Caddy. |
| **C22** | **`sync_positions` verkoopt meer dan de bot bezit.** De positiegrootte wordt de volledige exchange-balance van de base-asset, inclusief holdings die de bot niet kocht. Een full-close SELL (`held_quantity`) verkoopt dus alles. | `live.py:1083-1093`, `execution.py:349` | **WP1.1**: de bot-positie komt uit eigen fills. Exchange-balance dient alleen als bovengrens en reconcile-check (`min(eigen, exchange)`). Test: account met 0,5 BTC extern + 0,01 BTC via de bot → SELL = 0,01. |
| **C23** | Een aangesloten circuit breaker leegt óók SELL-signalen (`bar_signals = []`), terwijl het commentaar zegt dat alleen entries worden onderdrukt. | `strategy_engine.py:967` | **WP1.6**, extra acceptatie: HALT/REDUCE filteren alleen BUY; test "HALT + strategie-SELL gaat door". |
| **C24** | `max_order_size` kapt exits; drawdown en cooldown blokkeren de stop-loss juist wanneer die nodig is. | `risk_manager.py:375`, `risk.py:441-451, 486-503` | **WP1.2**: scope uitgebreid met `max_order_size`. |
| **C25** | `BacktestRunner` herbouwt `RiskParameters` zonder `sizing_mode`, `atr_risk_multiplier` en `max_cluster_exposure_pct`, waardoor custom risk in backtests deels wordt genegeerd. | `backtest.py:224-237` | **WP1.9** (één `RiskParameters`-pad voor alle modi) + pariteitstest in **WP2.1**. |
| **S12** | **Live-run-pagina is fragiel.** Eén mislukte poll vervangt de hele pagina door een foutmelding, omdat `error` nooit gewist wordt. Polls overlappen zonder in-flight guard, dus oudere responses kunnen nieuwere overschrijven. `createRun` heeft 120 s timeout zonder idempotency-key, dus een retry kan een tweede live run starten. | `runs/[id]/page.tsx:420-422, 445, 538`, `lib/api.ts` (`createRun`) | **Nieuw WP7.0** (~1 cyclus, vóór live-herstart): laatste goede data + "stale"-badge, in-flight guard, `Idempotency-Key`-header met backend-dedupe. |
| **S13** | **Live-UX mist veiligheidsstappen.** Geen getypte bevestiging voor LIVE-start; het confirm-token gaat in de body (deprecated pad) i.p.v. de `X-Live-Confirm-Token`-header. Geen LIVE-banner. Stop heeft geen bevestiging en geen waarschuwing over open posities. De kill-switch staat alleen op `/`. | `runs/new/page.tsx:259, 599-603`, `runs/[id]/page.tsx:566, 570-576`, `app/page.tsx:124` | **WP1.7** UI-deel (stop-dialoog, kill-switch in header) + **WP3.2** (token alleen via header). |
| **S14** | `/health` geeft altijd "ok" zonder DB-, exchange- of engine-task-check. De UI-healthcheck probeert `/api/health`, dat niet bestaat. Grafana's "Kill switch active"-alert kijkt naar de risk-manager-vlag, niet naar de globale kill-switch. | `main.py:654-658`, compose UI-healthcheck, `grafana/provisioning/alerting/trading-alerts.yml` | **WP3.4**: readiness-endpoint met DB-check en task-heartbeats; alerts op task-dood en globale kill-switch. |
| **S15** | Niets verhindert >1 worker of replica (dubbele runs door in-memory state). Migraties draaien bij elke start zonder lock en zonder dump vooraf. | `Dockerfile.api:150`, `docker-entrypoint.sh` | **WP4.4**: Postgres advisory/leader-lock bij startup. **WP3.5**: `pg_dump` vóór `alembic upgrade`. |
| **S16** | Infra-hygiëne:<br>• `REDIS_PASSWORD` default leeg<br>• images niet op digest gepind; uv via `curl \| sh`<br>• `/docs` en `/openapi.json` publiek<br>• CORS-wildcard-check alleen actief met auth aan<br>• `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.example` wordt in de productiebundle gebakken | compose, Dockerfiles, `main.py:622-624`, `config.py:210`, `.env.example:237` | **WP3.3** (QW Q14–Q16). |
| **S17** | Frontend-types zijn handgeschreven en drijven al af: `HealthResponse.components` bestaat niet in de backend; het UI-veld `error` heet in de backend `error_msg`. Jest draait niet in CI, en `src/app/**` valt buiten de coverage. | `apps/ui/src/lib/types.ts`, `route.ts:29` vs `emergency.py` | **WP7.1**: types genereren uit `/openapi.json`. **WP3.1**: jest-job. |
| **S18** | Flaky tests:<br>• timing: `test_adaptive_learning.py:612-654`<br>• datumgrens via `datetime.now()`: `test_graduated_circuit_breaker.py:258`<br>• ~39 tests op wall-clock<br>`filterwarnings=error` en `fail_under=80` worden zonder CI nergens afgedwongen. | zie kolom Bevinding | **WP4.8**: klok-injectie en fake timers. |

### 11.3 Gevolgen voor minimumset, quick wins en inspanning

**Minimumset vóór live-herstart (vervangt §4 Fase 1):**
- 1.0
- 1.1 + C22
- 1.2 + C24
- 1.3a
- 1.4
- 1.7 + C21/S13
- 1.8
- **7.0** (nieuw)

Aanbevolen erbij: 1.3b, 1.5 en 1.6 + C23.

**Extra quick wins:**

| # | Item | Bevinding |
|---|---|---|
| Q14 | `REDIS_PASSWORD` en `GRAFANA_PASSWORD` verplicht (`:?`) | S16 |
| Q15 | `/docs` en `openapi.json` alleen bij `DEBUG`; CORS-wildcard-check altijd | S16 |
| Q16 | `NEXT_PUBLIC_API_URL` leeg in `.env.example` | S16 |
| Q17 | Caddy `handle /api/admin/*` → UI | C21 |
| Q18 | `error` wissen bij succesvolle poll + in-flight guard | S12 |

**Inspanning:** +~3 cycli (7.0 ≈ 1; scope-uitbreiding 1.1/1.2/1.6/1.7 ≈ 2). **Totaal ~85 (ML bevriezen) – ~90 (ML fixen).** Volgorde ongewijzigd, met 7.0 parallel aan Fase 1.

**Aanvullende beslissing voor de gebruiker:**

| ID | Beslissing | Aanbeveling |
|---|---|---|
| D15 | Mag de bot crypto op de account aanraken die hij niet zelf kocht (C22)? | **Nee**: de positie is altijd `min(eigen fills, exchange-balance)`. |
