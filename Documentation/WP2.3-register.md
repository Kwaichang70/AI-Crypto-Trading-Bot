# WP2.3-register — Timestamp-alignment multi-symbol

**Datum:** 2026-11-06 · **Status:** P0–P4 gecommit (lokaal), gate A PASS op `b0586c0`; P4 = `2e2c79a` (`p4-v2.patch`, sha256 `c044f016…c120`, goedgekeurd); P5 is dit document; slotgate PASS op `2e2c79a` (P5 voegt alleen docs en één docstring toe) (unit 5550 + 1, integratie 1058 / 3 / 2 / 1) · **Branch:** `claude-session-20260924-vp2-fase1-minimumset-xsoezy` · **Basis:** `3d12187` (WP2.2-register inbegrepen) → `b0586c0` → `2e2c79a` → deze commit (P5).
**Plan:** `Documentation/Verbeterplan-v2-2026-09.md` §4, Fase 2, WP2.3 (bevinding C11).
**Bronnen:** de agent-rapporten in `reports/vp2-wp2.3/` (synthesis-spec, de drie ontwerpen, producer-, critic-, security-, risk-, quant- en arch-rapporten, `gateA-testing.md`). Die map is gitignored. Dit register is daarom zelfstandig leesbaar; ID's verwijzen naar die rapporten (prefix `WP23-` weggelaten).

---

## 1. Scope en resultaat

**Wat WP2.3 verandert:**
- **Pager (P0).** `fetch_ohlcv_range` stopt nooit meer op een korte of lege pagina. HEAD stopte bij de eerste korte pagina (bv. 299 van 2000 bars, synthesis-spec §2; in de P0-test 149 i.p.v. 1999, P0-04). De cursor wordt `last_ts + tf` na data en `+ page_limit·tf` na een lege pagina. Het stoppunt is `min(end, floor_tf(now) + tf)`, met `now` via de seam `_now_ms()`. Er komt geen bar buiten `[start, stop)` terug (SEC-03). De paginacap gebruikt een effectieve paginagrootte met een ondergrens van `page_limit // 4` (§3). Retry-After is begrensd op 60 s (SEC-05) en `page_limit ≥ 1` wordt afgedwongen (SEC-06).
- **Pure leaf (P1).** `packages/data/bar_alignment.py`: alleen stdlib, AST-gepind, richting `trading → data`. De module bevat `build_timeline`, `align_bars`, `check_alignment` en `fold_check`, plus `BarAlignmentError` met 8 gesloten codes, de tolerantieconstanten en `AlignmentReport` (`to_metadata`, `presence_digest`).
- **Engine (P2).** Een multi-symbol backtest loopt over de unie van de waargenomen timestamps, met een present-masker. Een afwezig symbool krijgt op die stap geen `on_bar`, mark, exit-check of fill. Er is geen ffill. Eén symbool neemt een fast path zonder validatie. Een full-row-guard (`not_trimmed`/`off_grid`) houdt `t0 == timeline[0]`.
- **Runner, optimizer, walk-forward (P3).**
  - Volgorde: align → telemetrie → check → warmup op de timeline → rebind.
  - Terminal MTM, start/end, `total_bars` en de OOS-range lezen het uitvoeringsvenster.
  - De optimizer aligneert één keer en doet een fold-check.
  - Walk-forward splitst op tijd.
  - De engine weigert dubbele symbolen (ARCH-04 (b)).
  - BACKTEST laat signalen voor een afwezig symbool vallen (ARCH-06).
- **API (P4).**
  - `BarAlignmentError` geeft 422 `{"code", "error_type"}`, met de branch vóór `except ValueError`.
  - Het venster is `requested_window = (start, min(end, floor_tf(now)))`, met `now` één keer gelezen vóór de fetch (SEC-01).
  - De request-grenzen van SEC-02 gelden: max. 20 symbolen, max. 500 000 bars × symbolen, start niet vóór 2009-01-03.
  - Dubbele symbolen geven 422 aan de request-rand; legacy-configs falen gesloten (CF-23-20).
  - Marker `engine_semantics.bar_alignment = "wp23-1"`; `timestamp_clock` blijft `"wp22-1"`.
  - `config["bar_alignment"]` wordt gepersisteerd, en run-responses krijgen `legacy_bar_alignment`.
- **Single-symbol-resultaten en goldens zijn byte-identiek** (I-1; quant P2-QA-05, P3-QA-07; gate A). De enige golden-bestandswijziging is de voorgeschreven pin-sleutel in P4 (§4).
- **CF-22-03 is deels gesloten:** bij een API-backtest en -optimize valt de vormende bar buiten het venster (SY-23-14). De fetch-side filter blijft voor WP2.6. Voor 1w geldt een uitzondering (CR-03, §5).

**Commits** (lokaal, in volgorde):

| Commit | Hash | Inhoud | Review (rondes, verdict) |
|---|---|---|---|
| P0 | `bf8e901` | Pager: geen stop op korte/lege pagina, `[start, stop)`, eff-cap v4, Retry-After-cap, `page_limit ≥ 1` | v2: critic, security en risk r1 elk APPROVE_WITH_NITS; v3: security r2 **REJECT** (R2-SEC-01); v4: security r3 APPROVE |
| P1 | `1c57626` | Leaf `bar_alignment.py` | critic r1 APPROVE_WITH_NITS; quant r1 MISMATCH (QA-04, vereiste fix); v2: critic r2 APPROVE_WITH_NITS |
| P2 | `669ac9f` | Engine-walk over de unietimeline | critic, arch: APPROVE_WITH_NITS; quant CONFIRMED_WITH_NOTES |
| P3 | `b0586c0` | Runner, optimizer, walk-forward; dubbele symbolen in de engine; ARCH-06 | critic r1 APPROVE_WITH_NITS; quant CONFIRMED_WITH_NOTES; v2: critic r2 APPROVE |
| P4 | `2e2c79a` | API-wiring, 422, venster-clamp, SEC-02-grenzen, CF-23-20, marker, legacy-vlag | critic en security r1 APPROVE_WITH_NITS; v2: delta-review (security r2, ook voor de critic) APPROVE |
| P5 | deze commit | `packages/trading/CLAUDE.md`, `packages/data/CLAUDE.md`, één docstringregel (P2-CR-01), dit register | code-critic r1 APPROVE_WITH_NITS |

- P0–P3 (`3d12187..b0586c0`): 15 bestanden, +5388/−182. Per commit:
  - P0: 3 bestanden, +750/−59;
  - P1: 2 bestanden, +2024;
  - P2: 4 bestanden, +760/−35;
  - P3: 8 bestanden, +1854/−88.
- P4 v2: 13 bestanden, +1414/−19. Alles inclusief tests; er is geen migratie.

**Beslissingen van de gebruiker** (opties in synthesis-spec §9; de keuzes zijn doorgegeven in de P5-opdracht):

| ID | Beslissing |
|---|---|
| U-01 | (a) QD-toleranties: 1% missing (warn 0,1%), max gap 4 (warn 2), head 0 / tail 1, per fold max(1, ⌊2%·n⌋), zero-volume telt als missing. Onder SY-23-05 gelden ze ook voor single-symbol API-runs. |
| U-02 | (a) Volledige scope, 2,25 cycli (plan 1,5), inclusief de walk-forward-tijdsplit en de fold-check. |
| U-03 | Eenmalige Coinbase-data-opname (egress naar `api.coinbase.com`) voor een sparse-candle-fixture, **vóór WP2.8**. Dit blokkeert WP2.3 niet. Raakt aan CF-26 uit het WP2.2-register (Coinbase-re-baseline, blokkeert PR #4): beide vragen toegang tot de Coinbase-API. |
| — | **"Lokaal tot PR #4 gemerged"**: de WP2.3-commits blijven lokaal op de branch tot PR #4 gemerged is. |

---

## 2. Kernbesluiten

| ID | Besluit | Motivatie |
|---|---|---|
| SY-23-01 | Unie-timeline met present-masker ("bar overslaan"). Inner join en `strict` zijn afgewezen. | Een inner join schrapt bars van complete symbolen (−8,75% tot −33%), besmet indicatoren over symbolen heen en verschuift de OOS. |
| SY-23-02 | Referentievenster R, kalendergrid G (head/tail/interior), uitvoeringsvenster E = eerste..laatste volle rij | Een marktbreed gat wordt zichtbaar, en E maakt `t0 == timeline[0]` exact. |
| SY-23-03 | QD-toleranties als code-constanten, alleen integers (U-01) | Dit zijn de enige kwantitatief afgeleide waarden (stop-overshoot ≤ 4,8 bars, ATR-bias ≤ 0,65%). |
| SY-23-04 | Bij een overtreding faalt de hele run; geen `drop_symbol` | Een symbool laten vallen wijzigt een vooraf geregistreerde hypothese en geeft survivorship-bias. |
| SY-23-05 | Afgedwongen met `requested_window` (elke API-run), anders alleen gerapporteerd | Goldens, scripts en de ML-gate blijven byte-identiek, en WP2.8 kan niet stil op afgekapte data draaien. |
| SY-23-06 | Warmup-gaten tellen mee in de tolerantie | Een gat in de warmup seedt ATR/RSI over dat gat. |
| SY-23-07 | `BarAlignmentError(ValueError)`, 8 codes, vaste tekst, vaste precedentie | CF-36: er mag geen data in een foutmelding staan. |
| SY-23-08 | 422 `{code}` vóór `except ValueError`; de optimizer re-raiset | De generieke branch echoot `str(exc)`; volgt het precedent van `cost_profile_http_error`. |
| SY-23-09 | Module in `packages/data`, alleen stdlib | Het bestaande pad is `trading → data`; `data → trading` is verboden. |
| SY-23-10 | Engine-diff op de `num_bars`-site plus de full-row-guard; WP2.2-pins onbewerkt | Dekt ook het directe engine-pad, dat geen ontwerp afdekte (G3). |
| SY-23-11 | Runner rebindt naar de gealigneerde dict vóór `t0`; `last_alignment`, geen veld op `BacktestResult` | Herstelt terminal MTM, start/end, `total_bars` en OOS; T-10 blijft triviaal controleerbaar. |
| SY-23-12 | P0-pager in scope en als eerste commit | Alle drie ontwerpen bevestigen de afkap (AD-04, QD-10, DP-12); zonder P0 zou de tail-check afgekapte fetches met 422 afwijzen (SY-23-22). |
| SY-23-13 | Live trailing-gap blijft acceptabel; risk-sign-off verplicht | De piek is nooit lager **zolang de gap-fetch slaagt** (aangescherpt door RISK-03). |
| SY-23-14 | De API geeft `requested_window = (start, min(end, floor_tf(now)))` mee | Anders faalt elke run met `end > now` op de tail-check; de vormende bar valt weg. |
| SY-23-15 | 3 counters zonder `run_id`, 3 logevents, geen gauges of alerts | Gauges zijn dubbelzinnig bij gelijktijdige backtests. |
| SY-23-16 | Marker `wp23-1`, `config.bar_alignment` met `presence_digest`, `legacy_bar_alignment` alleen voor backtests met niet precies één symbool (ontbrekende of misvormde symbolen vallen daaronder en falen gesloten) | De engine-semantiek van een legacy run met één symbool is ongewijzigd. |
| SY-23-17 | `oos_start_index` = ordinaal in de timeline; walk-forward-tijdsplit nu, de rest in WP2.7 | Anders gooit elke multi-symbol walk-forward met gaten een fout. |
| SY-23-18 | De optimizer aligneert en controleert één keer, inclusief folds | Anders vermenigvuldigen telling en logs zich met combinaties × folds. |
| SY-23-19 | Alleen backtest; paper/live → CF-23-01..03 | Geen van die punten wijzigt de backtest-semantiek. |
| SY-23-20 | 4h op Coinbase niet ondersteund (luide fout) | Coinbase heeft geen 4h; resampling → WP2.6. |
| SY-23-21 | Een afwijkend `bar.timeframe` wordt alleen gerapporteerd; geen `hypothesis` | `off_grid` vangt het inhoudelijke geval. |
| SY-23-22 | Volgorde P0 ∥ P1 → P2 → P3 (direct na elkaar) → P4 → P5 | P2 zonder P3 weigert ongetrimde input; de marker mag niet vóór P3 landen. |

Rulings uit de reviews:
- ARCH-04 (b): de engine weigert dubbele symbolen. Optie (a) brak een P2-pin.
- ARCH-06: geïmplementeerd in P3, dus CF-23-19 is niet geopend.
- P3-R1 ruling 1: geen veld `BacktestResult.bar_alignment`.
- P4-R1 ruling 3: de promote-check blijft.
- P4-R1 ruling 4: geen nieuwe CF voor de teruggerolde rij van een afgewezen backtest.
- P4-SEC-04: de fail-closed resume bij dubbele symbolen is correct.

---

## 3. Spec-wijzigingen ten opzichte van Verbeterplan v2, WP2.3

| Onderwerp | Verbeterplan v2 | Gerealiseerd | Bron |
|---|---|---|---|
| Beleid | "inner join of expliciet beleid: bar overslaan / ffill met vlag" | Bar overslaan (unie + masker); inner join en ffill afgewezen | SY-23-01 |
| Gap-detectie | "logt + metric" | Grid-gebaseerd (head/tail/interior); 3 counters en 3 logevents | SY-23-02/15 |
| Afbreken | "misalignment > tolerantie breekt af" | QD-waarden, de hele run, 422 met een gesloten code | SY-23-03/04/07/08 |
| `max_gap_bars` | — (spec: "longest interior run") | **Amendement QA-04:** langste run van afwezige **of** zero-volume slots, aaneengesloten samengevoegd; `gap_runs` blijven structureel | P1-QA-04, P1-R2-CR-04 |
| Pager | niet in het plan | **Toegevoegd** als P0, de eerste commit | SY-23-12 |
| P0-eff-cap | — | v2: cap `2·⌈span/(page_limit·tf)⌉+2`; dat kapt af als het exchange-max < `page_limit/2` (CR-01, SEC-04) | P0-CR-01 |
|  |  | v3: `eff` = `min(page_limit, len(eerste niet-lege pagina))`, waardoor de cap vacuüm werd (150× v2, R2-SEC-01, **REJECT**) | P0-R2-SEC-01 |
|  |  | v4: `eff = max(page_limit//4, min(page_limit, langste pagina tot nu))`, herberekend per niet-lege pagina; worst case 4× v2, eerlijke tellingen ongewijzigd | P0-R3-SEC-01 |
| Pager-hardening | — | **Toegevoegd** in v3: `[start, stop)` (SEC-03), Retry-After ≤ 60 s (SEC-05), `page_limit ≥ 1` (SEC-06), comment `page_limit ≤ exchange-max` (CR-04) | P0-v3 |
| Walk-forward | in WP2.7 | Minimale tijdsplit en fold-check **in WP2.3** | SY-23-17, U-02 |
| Venster-clamp | — | **Toegevoegd**; `now` vóór de fetch (P4-SEC-01) | SY-23-14 |
| API-grenzen | — | **Toegevoegd** in P4: 20 symbolen, 500k bars × symbolen, start ≥ 2009-01-03, `OverflowError` → 422 (SEC-05) | P0-SEC-02, P4-SEC-05 |
| Dubbele symbolen, signal-drop | — | **Toegevoegd**: engine (P3) en API (P4) | ARCH-04/06, CF-23-20 |
| SY-23-13 | "nooit losser" | "nooit losser zolang de gap-fetch slaagt" | P0-RISK-03 |
| T-01-tekst | "2000 gevraagd, k=150 → 1999" | Rekenfout; getest zijn 1 gat → 1999 en 154 gaten → alle bestaande bars | P0-07, P0-CR-07 |
| T-21 "run-rij `status=error`" | — | Alleen in-flight: de rollback maakt de rij niet duurzaam; log en counter zijn het duurzame spoor | P4-R1 ruling 4 |
| Inspanning | 1,5 cycli | Spec 2,25 cycli (incl. 0,2 voor reviewrondes) | synthesis-spec §5 |

---

## 4. Testbewijs

| Gate / meting | Basis | Unit | Integratie (pass/fail/skip/xfail) | Goldens | Overig | Verdict |
|---|---|---|---|---|---|---|
| Baseline | `3d12187` | 4983 + 1 | 1030 / 3 / 2 / 1 | 120 | — | — |
| Review P0 v2 | `3d12187` + v2 | 5008 + 1 | 1030 / 3 / 2 / 1 | — | 6/6 mutanten (critic); risk: 1680 resumes, piek nooit < HEAD | — |
| Review P0 v3 | `3d12187` + v3 | 5019 + 1 | 1030 / 3 / 2 / 1 | — | 3 mutanten gedood; R2-SEC-01 blokkerend | REJECT |
| Review P0 v4 | `3d12187` + v4 | 5022 + 1 | niet opnieuw gedraaid | — | 104 pagertests ×2; 6 mutanten; C5 28 034 (v3: 1 051 200) | APPROVE |
| Review P1 v2 | `3d12187` + P1 v2 | 5244 + 1 | — | — | 261 ×2; dead-run-oracle 3000 cases, 0 afwijkingen | — |
| Review P2 | + P2 | 5244 + 1 | 1058 / 3 / 2 / 1 | groen | 434 ×2; 5/5 critic-mutanten; quant: S1/S2 = prototype | — |
| Review P3 v2 | `1c57626` + P2 + P3 v2 | 5442 + 1 | 1058 / 3 / 2 / 1 | 120 | 158 ×2; quant: QA-06-waarden, 30-seed MC = prototype | — |
| **Gate A** | `b0586c0` | 5442 + 1 | 1058 / 3 / 2 / 1 | 120 | WP23-tests 485 (3× stabiel, ook omgekeerd); `python -O` 479; mypy --strict 0 (100 bestanden); ruff 829 = 829, 0 nieuw; leaf puur | PASS |
| Review P4 v1 | `b0586c0` + v1 | 5546 + 1 | 1058 / 3 / 2 / 1 | 120 | 104 ×2; M2 (bounds na claim) overleeft → CR-01 | — |
| P4 v2 | `b0586c0` + v2 (= `2e2c79a`) | 5550 + 1 | 1058 / 3 / 2 / 1 | 120 | 108 ×2 (producer en security r2); mutanten 41/41 (producer), 7/7 (security r2) | APPROVE (security r2) |
| **Slotgate** | `2e2c79a` (P5 voegt alleen docs en één docstring toe) | 5550 + 1 | 1058 / 3 / 2 / 1 | 120 | WP23-tests 593 (3× stabiel, ook omgekeerd), `python -O` 587, mypy --strict 0 fouten, ruff 829 = basis, geen schemawijziging | PASS |

- **Bekende failures, al aanwezig op de baseline** (dezelfde als bij WP2.2):
  - unit: `test_metrics.py::TestComputeSharpe::test_constant_returns_zero_std_returns_zero`;
  - integratie: `test_portfolio_endpoints::TestAggregatePortfolio::test_aggregate_with_backtest_metrics` en de twee OOS-tests in `test_sprint50_cycle5_integration`;
  - skips: `test_cf36_ws_realio.py` (`wsproto` ontbreekt); xfail: `test_smk_t_20` (CF-SMK-S1).
- **Telling-afstemming:**
  - De P3-producer rapporteerde 5395. Dat is op een basis zonder de 39 pagertests van P0; de critic reconcilieerde naar 5434 (P3 r1).
  - Gate A rekent 4983 + 39 + 261 + 0 + 159 = 5442. Er is één test meer dan de 158 nieuwe P3-tests, afkomstig uit bewerkte bestaande bestanden; die is niet geïsoleerd.
- **Quant:**
  - P3 levert S1/S2 `total_bars` 1450, S2 `exposure_pct` 0,6352 en OOS idx 1430 → −2,156 (P3-QA-01).
  - T-16: 0 misaligned stappen, 0 verschoven fills, 1450 bars. T-17: ΔOOS = 0.
  - De walk-forward-tijdsplit is zonder bias (P3-QA-04).
- **Goldens:**
  - Het is dezelfde set van 6 bestanden en 120 tests als in het WP2.2-register.
  - De golden-diff `3d12187..b0586c0` is leeg (gate A).
  - P4 voegt alleen de sleutel `"bar_alignment": "wp23-1"` toe aan de `engine_semantics`-pin in `test_wp21c_breaker_default.py` (SY-23-16).
- **Pins:** drie exact-dict-pins krijgen elk precies één sleutel. Een vierde pin (`test_wp21b_optimize_api.py:145`) popt `bar_alignment` en blijft de rest exact vergelijken (P4-02, P4-R1 ruling 2).
- **Andere bewerkte tests:** vijf integratiebestanden kregen alleen een `backtestEnd` dat past bij de gemockte bars. Zonder die edit falen 35 tests, allemaal op een echte `tail_coverage` (P4-01, P4-R1 ruling 1).

| Commit | Tests (WP23-T-NN) |
|---|---|
| P0 | T-01 pager, T-02 live trailing-gap (risk-sign-off) |
| P1 | T-03 AST-pin, T-04 QD-fixture, T-06 ppm/gap, T-07 coverage, T-08 off-grid, T-09 grenzen |
| P2 | T-05 engine-masker, T-08 engine-guard, T-12 random gaps, T-13 onafhankelijkheid, T-14 WP2.2-pins |
| P3 | T-10/T-11 byte-identiteit, T-15 boekhouding, T-16 oracle, T-17 OOS, T-18 walk-forward, T-19 optimizer, T-20 afwijzing, T-24 determinisme |
| P4 | T-21 422 en clamp, T-22 marker en legacy-tabel, T-23 Prometheus (met P3) |
| alle | T-25 volledige suite |

---

## 5. Carry-forwards en open punten

Een eigenaar of doel met "(v)" erachter is een voorstel van de opsteller. Zonder "(v)" komt hij uit het genoemde bronrapport.

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| CF-23-01 | Live draait een strategie opnieuw op een stale bar (`strategy_engine.py:4478-4482`, de tak "No new bar yet"; regelnummers na P4) | trading-engine-architect | Vóór een multi-symbol- of intraday-live-herstart |
| CF-23-02 | `get_latest_bar` kan de vormende candle teruggeven; samen met CF-22-03. Daarbij hoort **P4-CR-03**: de 1w-`floor_tf` is epoch-verankerd (donderdag), dus van donderdag tot zondag wordt de vormende weekbar niet getrimd. Geen regressie (HEAD deed hetzelfde). | data-pipeline-engineer (v) | WP2.6 |
| CF-23-03 | Ankerpariteit: backtest vraagt geen W aaneengesloten bars, paper/live wel | quant-strategy-analyst (v) | Zichtbaar in de paper-vergelijking van WP2.8 |
| CF-23-04 | O(n²)-history-slicing (C11); hangt samen met P4-SEC-02 | python-backend-specialist (v) | WP4.6 |
| CF-23-05 | WP2.5: kalender-as-of-equity-sampling, all-absent slots, SR en n uit dezelfde sampling (QD T-10-waarden overgedragen: waarheid SR 31,052536 / n 100; as-of SR 31,734025 / n 100; present-bar SR 32,273953 / n 97; eindequity 16284,270878) | quant-strategy-analyst (v) | WP2.5 |
| CF-23-06 | WP2.7: `oos_start_at`, train-fold-optimalisatie, pre-roll, fold-tolerantie inclusief pre-roll | quant-strategy-analyst (v) | WP2.7 |
| CF-23-07 | UI: badge voor `legacy_bar_alignment` en de alignment-stats (met CF-22-02), plus leesbare meldingen voor de 422-codes; de body bevat alleen een code (P4-R1) | nextjs-frontend-specialist (v) | UI-backlog, Fase 7 (v) |
| CF-23-08 | Niet-monotone fixture in test_wp13b_fix.py:505-512 | testing-quality-specialist (v) | WP4.8 (v) |
| CF-23-09 | Generieke `ValueError`-branches echoën `str(exc)` (runs.py, optimize.py); verwant: P4-SEC-09 (500's bij gemengde tz of vrije timeframe in optimize, 400 met echo bij een start in de toekomst) | security-audit-specialist | CF-36-vervolg |
| CF-23-10 | **WP2.8-gate:** research-scripts (`research_momentum_brackets.py`, `validate_sl_tp_reversion.py`, `train_model.py`) en de ML-gate (`ml.py:380`) hebben eigen, afkappende pagers. Haal data alleen via de post-P0-`fetch_ohlcv_range` of `wp21_rebaseline.fetch_candles`. | data-pipeline-engineer (v) | Vóór WP2.8 |
| CF-23-11 | 4h-resampling met een partial-bucket-vlag | data-pipeline-engineer (v) | WP2.6 |
| CF-23-12 | Opgenomen Coinbase-sparse-candle-fixture (weglating A1, end-inclusiviteit) | quant-strategy-analyst (v) | Vóór WP2.8 (U-03) |
| CF-23-13 | Researchmodus `drop_symbol`, alleen als die ooit gewenst is; nooit voor WP2.8 | — | Op verzoek |
| CF-23-14 | WP2.6: `data_hash` over de ruwe fetch naast `presence_digest`; gap-herverzoek (R2) op `GapRun`. Daarbij hoort de **QA-08-formulering**: gelijke `data_hash` voor alle symbolen + symboolset + R + tf + versie ⇒ gelijke `presence_digest`. Dit geldt niet per symbool afzonderlijk. | data-pipeline-engineer (v) | WP2.6 |
| CF-23-15 | WP2.8-kickoff: `series_provenance` meten voor 6 pairs × {1h, 1d}; universum en run-vorm vooraf registreren | quant-strategy-analyst (v) | WP2.8-kickoff |
| CF-23-16 | `seed_partial` blijft vuren op pairs met echte no-trade-buckets. Verwant: CF-23-RISK-B, `fetched_bars=len(gap)` op `trailing_stop.seed_partial` (RISK-08c). | risk-management-expert | Nog toe te wijzen |
| CF-23-17 | Legacy runs van vóór P0 kunnen op afgekapte data gedraaid hebben; dat is achteraf niet vast te stellen, dus alleen gedocumenteerd en niet gevlagd | — | Alleen documentatie |
| CF-23-18 | WP2.4 sleutelt zijn as-of-join op `close_i` (AD-15) | data-pipeline-engineer (v) | WP2.4 |
| CF-23-20 | Dubbele symbolen. **Opgelost in P4:** 422 aan de request-rand (run-create en optimize). Legacy-configs falen gesloten met `duplicate_symbols_in_config` bij resume (422 vóór de CAS), paper-bootrecovery en promote; nooit stil ontdubbelen. Rest → P4-SEC-04. | python-backend-specialist + security | Gesloten (P4) |
| P0-RISK-03 / P0-SEC-01 (CF-23-RISK-A) | Een `wait_for`-timeout (40 s) gooit alle verzamelde pagina's weg, zodat op een trage exchange de seed losser kan zijn dan HEAD. Oplossing: pager-`deadline` of `max_pages` met een gedeeltelijk resultaat. | risk-management-expert (+ engine-regel) | Nog toe te wijzen; vóór G-LIVE (v) |
| P0-R2-SEC-03 | Dekkingsgrens van de floor: een count-based exchange met max < ca. `page_limit/8` (38) wordt afgekapt, met een warning | primary (docs) | **Gedocumenteerd in P5** (`packages/data/CLAUDE.md`) |
| P0-SEC-07 | Paginagrens 10 (niet 8) en het verouderde engine-comment | — | **Gesloten**: comments in P2, CLAUDE.md in P5 |
| P0-RISK-08 | Runbook: rijen L-16/L-18 en de resume-sectie van `circuit-breaker-halt.md`, over hoe je `seed_partial` leest en over verwachte catch-up-trailing-SELLs bij een resume | devops-infrastructure-specialist (v) | Vóór de release van WP2.3 (v) |
| **P4-SEC-02 (High, pre-existing)** | Backtest en optimize blokkeren de event-loop van de enkele uvicorn-worker (O(n²): 13,8 s bij 8k bars, ca. 15 h geëxtrapoleerd op de grens). Zolang dat duurt staan de kill switch, emergency-stop en live-taken stil. Oplossing: off-loop (`asyncio.to_thread`) met een semaphore(1), of tot dan een CPU-budget of geen synchrone backtest naast een lopende live run. | risk-management-expert + python-backend-specialist | Vóór een live run naast API-backtests (v) |
| P4-SEC-03 | Geen procesbrede cap op gelijktijdige history-fetches: N requests delen het egress-IP van live trading | python-backend-specialist / devops | WP2.6 (rate-limiter) (v) |
| P4-SEC-04 | Live runs met een dubbel symbool zijn na de deploy niet meer te resumen; fail-closed is correct. Open: **deploy-gate** (zie Deploy-notities), een ruling over ontdubbelen in protective-mode (alleen in geheugen, met waiver `duplicate_symbols_deduped`) en een boot-log `recovery.live_duplicate_symbols`. | devops (gate) + risk-management-expert (ruling) | Gate vóór de rollout van WP2.3 |
| P4-SEC-06 | Geen symboolcap voor paper/live (`MAX_RUN_SYMBOLS`, bv. 20) | python-backend-specialist | Backlog (v) |
| P4-SEC-07/08/09 | 07: uniciteit is een exacte string-vergelijking (een canonieke-vorm-validator is een nit). 08: **opgelost in P4 v2** (de scan loopt door na een niet-string). 09: zie CF-23-09. | backlog | Backlog |
| P3-QA N1 | Rijen die aan een fold-rand weggesnapt worden, telt geen enkele fold-check. Optie: `fold_check` op het ongesnapte venster, of `rows_excluded_at_edges` per fold rapporteren. | quant-strategy-analyst (v) | WP2.7 |
| ARCH-09 / P3-QA-05 | Concentratienotitie: de globale tolerantie begrenst lokale concentratie niet (15 gaten in 50 bars = 30% lokaal). Een venstergebonden check hoort bij het pre-roll- en fold-werk. | quant-strategy-analyst (v) | WP2.7 |
| ARCH-10 | Geen package-brede pin "`packages/data` importeert nooit `trading`/`apps`"; nu alleen conventie plus de leaf-pin | testing-quality-specialist (v) | WP3.1 / WP4.8 (v) |
| Kleine nits | P1-R2-CR-02 (de degenerate-fold-check loopt na de off-grid-scan); P1-R2-CR-03 (`observed`/`limit`-asserts); P2-CR-02 (testnamen); P2-CR-04 (comment over de `present`-volgorde); P4-CR-05 (private `_TIMEFRAME_SECONDS`-import); P4-CR-07 (een unit-test importeert integratie-fixtures) | python-backend-specialist (v) | Follow-up / WP4.8 (v) |
| Gesloten of gevouwen (geen actie) | Gevouwen in een v2/v3: P0-CR-02/03/04 en SEC-03/05/06; P1-CR-01..05, CR-07, CR-09 (deels) en QA-04/05; P3-CR-01/03/04 en QA-N2; P4-SEC-01/05 en CR-01/02/04/06. Gesloten in P5: P0-RISK-04 (docs), P2-CR-01 (docstring), P2-CR-03 (Known limit). Gesloten door de commitvolgorde: P2-ARCH-07/08. Informatief, zonder actie: P0-CR-05/06 (de testbestand-mypy-bevindingen vallen buiten de gate), P1-CR-06/08, P2-ARCH-03 (residu), P2-CR-05/06, P3-R2-CR-02 (een `ValueError` na de hele-run-check telt als `ok`) | — | — |

### Deploy-notities
- **SEC-04-deploygate (verplicht vóór de rollout).** Draai deze query en flatten of stop elke treffer vóór de deploy:

```sql
SELECT id, status FROM runs
WHERE run_mode = 'live' AND status IN ('running','orphaned','resuming')
  AND jsonb_typeof(config->'symbols') = 'array'
  AND jsonb_array_length(config->'symbols')
      <> (SELECT count(DISTINCT s) FROM jsonb_array_elements_text(config->'symbols') s);
```

- **Nieuw 422-gedrag van API-backtests (P4-R1 ruling 1):**
  - Een venster dat vandaag of in de toekomst eindigt, geeft geen 422.
  - Een 422 komt alleen als de exchange meer dan 1 bar achterloopt, als een venster in het verleden meer dan 1 bar tail mist, of als het pair na de start gelist is (`head_coverage`).
  - Dunne pairs zonder trades in de laatste ≥ 2 buckets geven een 422 (CF-23-12/16).
  - Vanaf 5 aaneengesloten ontbrekende of zero-volume bars volgt `gap_too_long` (QA-04). Op 1h breekt een exchange-brede storing van 5 uur of meer elke run af. Dat treft vooral 1h-runs over incidenten, niet de 1d-runs van WP2.8 (P1-R2-CR-04, P1-QA-09).
  - 5m over 5 jaar (525k bars) valt nu boven de SEC-02-grens.
- **Afgewezen backtest:** de rij met `status="error"` wordt teruggerold. Het duurzame spoor is `runs.backtest_bar_alignment_rejected` (alleen de code), `backtest.bar_alignment_rejected` en `bar_alignment_rejected_total{reason}` (P4-R1 ruling 4).
- Er is geen migratie; oude rijen worden nooit herschreven (I-11).

---

## 6. Procesnotities

- **Escalatieregel (2 rejections):** niet geraakt.
  - Er was één REJECT: P0 v3, R2-SEC-01 (de eff-cap werd vacuüm). v4 nam de voorgestelde fix en 3 tests exact over en kreeg in r3 APPROVE.
  - P1 kreeg een quant-MISMATCH (QA-04, een vereiste fix, geen REJECT), opgelost in v2.
  - De v2's van P3 en P4 verwerkten niet-blokkerende nits.
- **Overtredingen van agentregels (geen ervan wijzigde code op de branch):**
  - De P4-producer maakte in de eigen wegwerp-worktree één lokale commit, `cf927fc` (P4-03). Geen branch bevat hem (P4-CR-08). Het object staat nog in de gedeelde object store (P4-v2-rapport). Volgens de critic ruimt `git gc` het op; er is niets aan gedaan.
  - Een `git stash list` of andere stash-actie is in de WP2.3-rapporten niet vastgelegd.
  - Het WP2.2-register meldt dat een stash uit WP2.2 (`refs/stash` = `5ebc033`) op een beslissing van de gebruiker wacht. Die beslissing is inmiddels genomen: op 2026-11-05 heeft de gebruiker gevraagd de stash te verwijderen, en `refs/stash` is toen verwijderd met `git update-ref -d` (zonder stash-commando). Het object `5ebc033` staat nog ongerefereerd in de object store.
- **Containerherstart midden in het WP:**
  - De agentrapporten leggen de herstart niet vast; de coördinator meldde hem.
  - De host-boottijd is 2026-11-06 07:21:53 UTC.
  - De drie ontwerpen, de synthesis-spec en de eerste `p0.patch`/`p1.patch` dateren van 2026-11-05 (11:19–12:03 UTC). Alle producer- en reviewrapporten zijn van na de herstart en reviewden die patches (sha's `54a18b3f…`, `382b2b84…`).
- **Rapportcorrecties:**
  - P0-07/CR-07: de rekenfout k=150.
  - P0-SEC-07: de paginagrens is 10, niet 8.
  - P3-QA-08: de kolom in P2-QA-02 bevat gemiddelden, geen maxima.
  - P2-QA-07: de opgeslagen QD-prototype vereist een hernoeming van `_pos`; de QD-getallen blijven staan.
  - P0 v3: ruff-telling 12, tegen 13 in critic r1 (een verschil in telmethode, P0-R2).
- **Worktrees:** elke agent verwijderde alleen de eigen worktrees en raakte die van anderen niet aan.
- **Hygiëne:**
  - Voor zover de rapporten het vermelden: geen `.env` gelezen, geen netwerk, en geen checkout/stash/reset/clean in de hoofd-repo. Niet elk rapport noemt het.
  - Serena, MCP memory, sequential-thinking en context7 waren niet beschikbaar in de rapporten die ze noemen; de andere rapporten zeggen er niets over.
