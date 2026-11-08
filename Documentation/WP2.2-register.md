# WP2.2-register — Bar-tijd-klok

**Datum:** 2026-11-03 · **Status:** code gecommit (lokaal), eindgate PASS (gate 3 op `7e61d13`) · **Branch:** `claude-session-20260924-vp2-fase1-minimumset-xsoezy` · **Basis:** `117236f` (WP2.1 inbegrepen) → `7e61d13`.
**Plan:** `Documentation/Verbeterplan-v2-2026-09.md` §4, Fase 2, WP2.2 (bevinding C12).
**Bronnen:** de agent-rapporten in `reports/vp2-wp2.2/` (synthesis-spec, synthesis-r2-guards, producer-, critic-, security-, quant- en gate-rapporten). Die map is gitignored. Dit register is daarom zelfstandig leesbaar; ID's verwijzen naar die rapporten.

---

## 1. Scope en resultaat

**Wat WP2.2 verandert:**
- **Backtests worden gestempeld op de bar-close.** Elk event van bar *i* krijgt `close_i = max_s(bars[s][i].timestamp) + tf`. Dat is dezelfde waarde als de WP1.5-risk-klok, en het geldt ook voor warmup-marks en voor CANCELED-transities in `stop()`. Het seed-punt is `t0 = close_0 − tf`. Het mechanisme is `packages/trading/clock.py` (`BarClock`, `WALL_CLOCK`, `use_clock`); `BacktestRunner` maakt de klok aan. Het raakt de tabellen `trades`, `fills`, `orders`, `equity_snapshots` en `position_snapshots`. De auditkolommen van `runs` blijven wall-time.
- **Paper en live blijven op wall-time, met een identiteitsguard.** Buiten BACKTEST moet de klok het singleton `WALL_CLOCK` zijn (of `None`, dat daarnaar resolvet); dat wordt op identiteit gecontroleerd, niet met `isinstance`. De guards zitten in `StrategyEngine.__init__` en `LiveExecutionEngine.attach_clock`. Daarnaast zijn er vijf fail-closed tripwires: `StrategyEngine.start`, `run_live_loop`, `LiveExecutionEngine.on_start`, `run_paper_engine` en `run_live_engine`. De enige test-seam is een monkeypatch op `trading.clock._wall_now`; `portfolio._utc_now` delegeert daarheen.
- **Prune-fix.** Dit is `prune_equity_snapshots` in `apps/api/services/retention.py`. Paper- en live-rijen worden verwijderd op `equity_snapshots.timestamp`, backtest-rijen op `runs.created_at`, en rijen van andere modi nooit. Zonder deze fix zou de 90-dagenprune herstempelde backtest-curves wissen.
- **Lot-close bij flatten.** Bij een volledige flatten roept de engine `close_lot(symbol)` aan op de trailing- en de bracket-manager. Dat voorkomt dat een volgende lot de oude lot-state erft, omdat lot-keys onder bar-tijd botsen.
- **Retraining-filter.** Beide queries in `apps/api/services/retraining.py` joinen `runs` met `run_mode IN ('paper','live')`. Anders worden backtest-trades na WP2.2 exact gealigneerde in-sample labels (leakage). Het pad is latent, want `ml_auto_retrain` staat standaard uit.
- **API-marker en `legacy_timestamps`.**
  - `engine_semantics.timestamp_clock = "wp22-1"` staat op backtest- en optimize-runs, niet op paper en live.
  - De run-response krijgt `legacy_timestamps: bool`. Een backtest zonder bekende marker telt als legacy (fail-closed).
  - `snapshot_at` van een backtest is de laatste bar-close. Een lege curve geeft `EmptyEquityCurveError` (HTTP 500).
  - `lastUpdated` gebruikt `bar_index` als tie-break.
- **Headline-metrics en goldens zijn ongewijzigd** (QA-05, zie §4).

**Commits** (lokaal, in volgorde):

| Commit | Hash | Inhoud | Review (rondes, verdict) |
|---|---|---|---|
| P0 | `4611b9d` | Prune per run-modus; index-gedreven batches; CF-36-schone logging | critic + security: r1 APPROVE_WITH_NITS; v2 verwerkt de nits; r2 APPROVE_WITH_NITS |
| P1 | `5efe612` | Klokmodule `trading/clock.py` | critic r1 **REJECT** (CR-01), r2 APPROVE_WITH_NITS; security r1+r2 APPROVE_WITH_NITS |
| P2a | `c5add9b` | `close_lot` in beide managers bij full flatten | critic + security APPROVE_WITH_NITS |
| P2a.1 | `4d53113` | Regressietests SEC-06 (a)–(d); commentaar dat `close_lot` niet mag raisen | critic + security APPROVE |
| P2b | `0d91d7e` | Atomaire stempelswitch, guards, tripwires, AST-pins | critic + security APPROVE_WITH_NITS; quant CONFIRMED_WITH_NOTES |
| P2b.1 | `e80905a` | Geen auto-retry na een paper-tripwire; 3 tripwire-testgaten gedicht; `_utc_now` → `_wall_now`; fallback-klok run-scoped | critic + security APPROVE_WITH_NITS |
| P3 | `0ba263f` | Marker, `legacy_timestamps`, `snapshot_at`, `lastUpdated`, twee pin-edits van één sleutel | critic r1 APPROVE_WITH_NITS; v2 verwerkt de nits; r2 APPROVE_WITH_NITS |
| P4 | `4be2313` | Retraining-filter paper/live | critic APPROVE_WITH_NITS |
| P2b.2 | `05eb63c` | Fallback-klok binnen de `with` (attach eerst), restore op het bij entry gevangen exec-engine, re-entrancy-guard, AST-pin op `._clock`-schrijfsites | critic APPROVE_WITH_NITS (met security-lens) |
| P5 | `7e61d13` | Apps-cleanup: AST-tripwire verbreed, pooled-PG-test tegen een `SET LOCAL`-lek, WARNING-log zonder traceback, `EmptyEquityCurveError`, regex-join-assert | critic APPROVE_WITH_NITS |

Totaal `117236f..7e61d13`: 32 bestanden, +5848/−144 (inclusief tests). Er is geen migratie.

**Beslissingen van de gebruiker:**

| ID | Beslissing |
|---|---|
| — | **"Lokaal tot PR #4 gemerged"**: de WP2.2-commits blijven lokaal op de branch tot PR #4 gemerged is. |
| — | **"Register in repo"**: per WP één beknopt register in `Documentation/`; dit document. |
| U1(a) | Backtest-equity blijft 90 dagen bewaard, gerekend vanaf `runs.created_at`. Dat is het effectieve gedrag van vóór WP2.2. |
| U2(a) | Volledige scope (~2,5 cycli), inclusief P4 en `legacy_timestamps`. |

---

## 2. Kernbesluiten

| ID | Besluit | Motivatie |
|---|---|---|
| SY-22-01 | Event-tijd = bar-close `close_i` voor elk backtest-event van bar *i* | Bar-open geeft één bar look-ahead, wijkt op 1d-bars af van de WP1.5-gatedag, en fills worden op `bar.close` geprijsd. |
| SY-22-02 | Seed `t0 = close_0 − tf = max_s(bars[s][0].timestamp)` | De curve blijft monotoon en gelijk verdeeld, ook bij misaligned multi-symbol. |
| SY-22-03 | Klokmodule met amendementen A–C (zie §3) | Eén klokbron en één injectiepad. |
| SY-22-04 | BACKTEST accepteert `None` of een `BarClock` | De PAPER/LIVE-helft is vervangen door SY-22-R2-01/02. |
| SY-22-05 | `use_clock` (ContextVar) alleen in `run_backtest`, met reset in `finally`. `utc_now` alleen als `default_factory`. Elk gepersisteerd veld wordt expliciet gestempeld. | Dekt de in-memory defaults zonder een uitzonderingslijst over negen strategiebestanden. |
| SY-22-06 | Geen `seq`-kolom, geen migratie | Eén fill per order (gepind door T-18), resume is alleen live, en `bar_index` ordent de curve al. |
| SY-22-07 | `close_lot` bij full flatten in beide managers; geen re-keying | Sluit de lot-key-botsing en het stale-bracket-anchor-pad. Re-keying raakt de resume-paden → CF-22-01. |
| SY-22-08 | Marker `timestamp_clock = "wp22-1"` op backtest en optimize | Volgt de repo-conventie voor versietags (`wp21-1`, `wp13b-1`). WP2.3 kan de versie ophogen. De legacy-regel faalt gesloten. |
| SY-22-09 | `legacy_timestamps` wordt server-side berekend; de UI volgt later | Minimale API nu. UI → CF-22-02. |
| SY-22-10 | Prune-fix als P0, een harde blocker voor P2b | Anders wist de prune herstempelde backtest-curves, omdat hun stamps in het verleden liggen. |
| SY-22-11 | Retraining-filter (P4) hoort in de scope | WP2.2 maakt van backtest-trades gealigneerde in-sample labels. |
| SY-22-12 | De per-run `daily_pnl`-API blijft ongewijzigd | Geen regressie: vandaag is de waarde ook al 0 op elke dag behalve de rundag. Het N/A-label → CF-22-02. |
| SY-22-13 | Geen deterministische id's | De DoD vraagt gelijke resultaten, geen gelijke uuid's. T-08 normaliseert uuid's naar ordinalen. Optioneel in WP4.8. |
| SY-22-14 | "Backtest-tabellen" in de DoD = `trades`, `fills`, `orders`, `equity_snapshots`, `position_snapshots` | De auditkolommen van `runs` leggen vast wanneer de run draaide; `ck_runs_stopped_after_started` heeft ze nodig. |
| SY-22-15 | Backtest-`snapshot_at` = `equity_curve[-1].timestamp` | Een pure functie van het resultaat; geen nieuwe portfolio-API. |
| SY-22-16 | `lastUpdated` ordent op `timestamp desc, bar_index desc` | Per stamp kunnen er meerdere curvepunten zijn. |
| SY-22-17 | WP2.5-contract C1–C5 als geteste invarianten, zonder nieuwe summary-velden | `trading_start_at` en `oos_start_at` zijn van WP2.5, dat de metric-inputs bezit. |
| SY-22-18 | Buiten scope: partial last bar → WP2.6 (CF-22-03); MTF-wall-reads (C10) → WP2.4; `start_date`/`end_date` blijven op bar-open | Elk punt hoort bij het WP dat het onderwerp bezit. |
| SY-22-19 | Goldens zijn onschendbaar | Een gewijzigde digest-constante betekent automatisch REJECT. |
| SY-22-20 | Volgorde P0 → P1 → P2a → P2b → P3 → P4; P2b is atomair | `ck_trades_exit_after_entry` vereist dat de stamps en `exit_at` samen landen. De marker mag niet vóór de stamps landen. |
| SY-22-21 | Circuit breaker ongewijzigd | Hij draait al op bar-tijd (WP1.5/1.6). |
| SY-22-R2-01 | LIVE: identiteits-allowlist `clock is WALL_CLOCK`. Geen `clock=` op de constructors van Base en Live. Class-default `_clock = WALL_CLOCK`. | Een duck-typed klok of `bar_clock.now` komt door een reject-check heen, en live `_transition` stempelt via `_clock`. Vervangt de LIVE-helft van SY-22-04. |
| SY-22-R2-02 | PAPER: dezelfde allowlist op `StrategyEngine`. `PaperExecutionEngine` blijft mode-agnostisch. | Paper is de poort naar live. Nul bestaande tests geraakt. S18 wordt bediend via de `_wall_now`-seam. Vervangt de PAPER-helft van SY-22-04. |
| SY-22-R2-03 | `PortfolioAccounting(clock=)` blijft een ongeguarde `Callable` | 26 legitieme fake-klok-sites in tests. R7 (nul `clock=` in `apps/`) en R10 zijn de controles. |
| SY-22-R2-04 | De P2a-nits gaan als aparte commit P2a.1 | Houdt P2a en P2b atomair, zonder extra reviewronde. |
| SY-22-R2-05 | Bindende P2b-eisen R1–R15: o.a. R6 type-gated tripwires op 5 sites, R4 een expliciete `raise` (geen `assert`), R12 en R13 | `python -O` stript asserts, en Mock-exec-engines zouden zonder type-gate breken. |
| SY-22-R2-06 | De finality-melding noemt de klasse letterlijk (R15, in P2b) | Track A bezit `clock.py`. |

Volgens de P2b-security-review zijn R1–R15 allemaal MET. R14 is MET sinds P2b.1.

---

## 3. Spec-wijzigingen ten opzichte van Verbeterplan v2, WP2.2

| Onderwerp | Verbeterplan v2 | Gerealiseerd | Bron |
|---|---|---|---|
| Stempelmoment | "bar-timestamp" | Bar-close `close_i`, seed `t0` | SY-22-01/02 |
| Klok-injectie | "gesimuleerde klok geïnjecteerd" | `trading/clock.py`. **A:** één `close_i`-object voedt zowel `BarClock.advance` als `portfolio.advance_clock`. **B:** het exec-engine krijgt de klok alleen via `StrategyEngine.attach_clock`. **C:** class-default `BaseExecutionEngine._clock = WALL_CLOCK`. | SY-22-03 |
| Paper/live | "wall-clock alleen paper/live" | Identiteits-allowlist plus 5 tripwires | SY-22-R2-01/02 |
| Prune-fix | niet in het plan | **Toegevoegd** als P0, blocker voor P2b | SY-22-10, U1(a) |
| Lot-close | niet in het plan | **Toegevoegd** als P2a | SY-22-07 |
| API-semantiek | niet in het plan | **Toegevoegd** als P3 | SY-22-08/09/15/16 |
| Retraining | niet in het plan | **P4 in scope** | SY-22-11, U2(a) |
| Sequence-key | — | **Geen `seq`-kolom en geen migratie** (quantvoorstel QD-05 afgewezen) | SY-22-06 |
| DoD "geen wall-clock in backtest-tabellen" | ongescoped | Vijf tabellen. De MTF-wall-reads bleven open tot WP2.4; voor nieuwe runs gesloten door WP2.4 P1 (`2697090`, G14): `regime_at_entry`/`signal_context` zijn point-in-time of `None`. Oude rijen blijven zoals opgeslagen. Zie `Documentation/WP2.4-register.md`. | SY-22-14/18 |
| Hardening | — | **Toegevoegd:** P2a.1, P2b.1, P2b.2 en P5 (reviewnits; zie de commit-tabel) | SY-22-R2-04, reviews |
| Inspanning | 1,5 cycli | Spec 2,5 cycli (incl. 0,35 voor review- en fixrondes en de PG-gate) | synthesis-spec §6 |

---

## 4. Testbewijs

| Gate / meting | Commit | Unit | Integratie (pass/fail/skip/xfail) | Goldens | Overig | Verdict |
|---|---|---|---|---|---|---|
| Gate 1 | `c5add9b` (P0–P2a) | 4864 + 1 bekend | 947 / 3 / 2 / 1 | 120 | Nieuwe WP22-bestanden: 90 tests (36/44/10). T-12 PG16 **10 un-skipped**. `tests/migrations` 90/0. Determinisme stabiel. | PASS |
| Gate 2 | `0d91d7e` (t/m P2b) | 4907 + 1 | 1017 / 3 / 2 / 1 | 120 | WP22-tests 167, stabiel. `python -O`: 103 passed. | PASS_WITH_NOTES |
| Review P2b.1 | `e80905a` | 4907 + 1 | 1024 / 3 / 2 / 1 | 120 | 7/7 mutanten gedood | — |
| Review P3 | `0ba263f` | 4960 + 1 | 1024 / 3 / 2 / 1 | 120 | 6/6 producer-mutanten gedood | — |
| Baseline P4 | `4be2313` | 4965 + 1 | 1024 / 3 / 2 / 1 | — | Geverifieerd door de P5-producer | — |
| Review P2b.2 | `05eb63c` (op `4be2313`) | 4967 + 1 | 1030 / 3 / 2 / 1 | 120 | 4/4 producer-mutanten gedood | — |
| Review P5 | `7e61d13` (op `4be2313`) | 4981 + 1 | 1024 / 3 / 2 / 1 | 120 | PG16 **12 un-skipped** (2 runs, door de P5-producer; de critic draaide geen PG) | — |
| **Eindgate (gate 3)** | `7e61d13` | 4983 + 1 | 1030 / 3 / 2 / 1 | 120 | WP22-bestanden 292 (3× stabiel, ook omgekeerd). `python -O`: 118 passed. T-12 PG16 **12 un-skipped**; `tests/migrations` 92/0. mypy --strict 0 → 0; ruff 830 → 829, geen nieuwe bevindingen. | PASS |

- **Bekende failures, al aanwezig op de baseline:**
  - unit: `test_metrics.py::TestComputeSharpe::test_constant_returns_zero_std_returns_zero`;
  - integratie: `test_portfolio_endpoints::TestAggregatePortfolio::test_aggregate_with_backtest_metrics`, plus de twee OOS-eligibility-tests in `test_sprint50_cycle5_integration`.
  - De skips zijn `test_cf36_ws_realio.py` (`wsproto` ontbreekt); de xfail is `test_smk_t_20` (CF-SMK-S1).
- **Quant (P2b):**
  - T-03, T-04, T-05 en T-15 kloppen exact met het quantontwerp.
  - Headline-metrics (CAGR, Sharpe, Sortino, Calmar, PSR, MDD, exposure) zijn byte-identiek op synthetische 1h- en 1d-runs (QA-05).
  - Twee vergiftigde wall-klokken (2031 en 1999) geven identieke rijen (QA-10).
- **T-12 Postgres-gate** is un-skipped uitgevoerd, dus WP2.2 is niet "done pending PG verification".
- **Testspecificatie:**

| Commit | Tests (WP22-T-NN) |
|---|---|
| P0 | T-12 prune (unit en echte PG) |
| P1 | T-01 klok-unit |
| P2a / P2a.1 | T-11 lot-botsing (AT-10: stop 102,82, geen exit op 105) |
| P2b | T-02 DoD-stamps, T-03 quantwaarden, T-04 dagattributie, T-05 multi-symbol, T-06 wall-tripwire, T-07 AST-pins, T-08 determinisme, T-09 paper/live ongewijzigd, T-15 WP2.5-contract, T-16 CF-26-record, T-18 één fill per order, T-19 runner-randen |
| P3 | T-13 marker en legacy-regel, T-14 `lastUpdated` |
| P4 | T-17 retraining-filter |
| alle | T-10 goldens, T-20 volledige suite |

**Golden-set (6 bestanden, 120 tests):**
- `tests/integration/test_wp13b_anchor_harness.py`
- `tests/integration/test_wp15a_backtest_day.py`
- `tests/integration/test_wp16_cb_backtest.py`
- `tests/integration/test_wp21_cost_parity.py`
- `tests/integration/test_wp21_cost_goldens.py`
- `tests/unit/test_wp21c_breaker_default.py`

Geen enkele digest- of golden-constante is gewijzigd (SY-22-19). Vijf bestanden zijn byte-identiek aan `117236f`. In `test_wp21c_breaker_default.py` is alleen de voorgeschreven sleutel `"timestamp_clock": "wp22-1"` toegevoegd aan de `engine_semantics`-pin (P3, SY-22-08).

---

## 5. Carry-forwards en open punten

Eigenaar en doel zijn voorstellen van de opsteller, tenzij een bron-ID is genoemd.

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| CF-22-01 | Lots her-keyen op het id van de openende order (raakt de resume-paden van paper en live) | trading-engine-architect | WP4.1 |
| CF-22-02 | UI: legacy-timestamps-badge, label "Simulated until", `daily_pnl` N/A voor backtests | nextjs-frontend-specialist | UI-backlog (Fase 7) |
| CF-22-03 | Nog lopende laatste bar bij de fetch laten vallen (geeft anders stamps in de toekomst) | data-pipeline-engineer | WP2.6 |
| CF-22-04 | Holding-time en exposure onder partial exits: gebruik de unie van `[entry_at, exit_at)` of weeg per lot/hoeveelheid (naïeve som 707 tegen 550 bars) | quant-strategy-analyst | WP2.5 |
| CF-22-P0-SEC-01 | De prune kan de resume-drawdown-peak verlagen van live runs die ouder zijn dan de retentie. Verwant: P0-CR-06, backtest-prune ongeacht run-status (een optionele status-guard). | risk-management-expert | Vóór een G-LIVE waarin een live run langer loopt dan het retentievenster |
| P0-SEC-04 | Exportregel voor `equity_snapshots` toevoegen aan runbook §11.9 (`docs/runbooks/deploy-fase1-minimumset.md`); nog niet gedaan | primary (docs) | Vóór de release van WP2.2 |
| P2B1-SEC-08 | `test_wp15a_contracts::test_wp15_t18…` retargeten op `trading.clock._wall_now`, daarna de letterlijke `datetime.now` uit de docstring van `portfolio._utc_now` halen. Het is een WP1.5-testbestand, dus goedkeuring is nodig. | python-backend-specialist | Vervolgcommit / WP4.8 |
| R13 | Re-review van `thread_inherit_context` vóór Python 3.14 (een test faalt zodra de vlag aan staat) | security-audit-specialist | Vóór adoptie van Python 3.14 |
| P2B2-CR-01..05, SEC-02 | Resterende P2b.2-nits:<br>• AST-pin mist `object.__setattr__`, tuple-targets en `__dict__`/`vars()`<br>• de engine-side `is bar_clock`-guard is ongetest (mutant M3b overleeft)<br>• de "Raises"-docstring van `run_backtest` mist de re-entrancy-`RuntimeError`<br>• `_restore_fallback_clock` hernoemen<br>• `match=` toevoegen aan de eerste scope-test<br>• "één exec-engine per `StrategyEngine`" documenteren (SEC-02) | python-backend-specialist | Follow-up (P2b.3) / WP4.8 |
| P5-CR-01..05 | Resterende P5-nits:<br>• het AST-importrule van de prune-loop is te omzeilen (`import api.db.models`, `from api import db`, relatieve imports, `__import__`/`importlib`, `.delete`/`.begin_nested`/`.connection`)<br>• de PG-rollback-test is redundant<br>• de CF-36-loop-test is te los (accepteert extra `**`-splats)<br>• een unit-test importeert de integratie-harness (+51 mypy-fouten transitief)<br>• docstring "same statistics → same plan" is te stellig | python-backend-specialist | Follow-up / WP4.8 |
| P5-SEC-01 | De DEBUG-trace van `equity_prune.error` kan SQL-parameters bevatten (cutoff, run-ids). Redaction dekt secrets, geen business-identifiers. | security-audit-specialist | Nog toe te wijzen |
| P2A-SEC-08 | `engine.fill_hook_error` logt met `.exception` (volledige traceback) en staat niet op de CF-36-lijst | security-audit-specialist | CF-36-backlog |
| P4-CR-02 | De retrain-trigger telt trades na `last_at`, maar de fetch heeft geen tijdsgrens (superset); al aanwezig vóór WP2.2 | python-backend-specialist | Fase 6 (ML) |
| P4-CR-07 | Het retraining-filter is alleen getest op gecompileerde SQL en een stub-sessie; T-12 dekt alleen de prune. Geen echte-PG-check van de retraining-queries. | testing-quality-specialist | Vóór `ml_auto_retrain` aan gaat |
| P0-R2-CR-03 | De lege prune-pass groeit lineair met oude, geleegde backtest-runs (95 ms bij 60k runs). Heroverwegen bij miljoenen backtests. | database-architect | Heroverwegen bij schaalgroei |
| P3-CR-03, P2B-QA-11, P2A-CR-06 | Kleine testgaten:<br>• `legacy_timestamps` alleen op converter-niveau getest (stop en emergency-stop `model_dump` niet direct)<br>• T-04 1d-test gebruikt `>=` waar T-15 `==` gebruikt<br>• engine-tests bewijzen niet dat warn-set-discards en pending-clearing de managers bereiken (eigenaar bevestigt of P2a.1/P2b dit al dekt) | python-backend-specialist | Follow-up / WP4.8 |
| — | Bestaande ruff- en format-schuld in o.a. `strategy_engine.py`, `portfolio.py`, `run_persistence.py`, `schemas.py`, `routers/runs.py`, `retraining.py`; mypy-fouten in testbestanden (`tests/` is uitgesloten in de config). Niet verergerd. | devops-infrastructure-specialist | WP3.1 / WP4.8 |
| CF-26 | Coinbase-re-baseline van WP2.1c (`scripts/wp21_rebaseline.py`) is nog niet gedraaid: de fetch naar de Coinbase-API wordt in deze omgeving door de proxy geweigerd (403). Oplossing: domein toestaan of fixture aanleveren. WP2.2 laat het record byte-identiek (T-16). **Blokkeert PR #4 (gebruikersbesluit "Wachten op CF-26").** | quant-strategy-analyst | Vóór de merge van PR #4 |
| QA-06/07/08 | Overdracht naar WP2.5:<br>• 1d-bars: de bar die op dag D opent, wordt geboekt op D+1 (gelijk aan de gate); verschuif open-gelabelde vendorreeksen één tf<br>• `result.end_date` ligt één tf vóór `curve[-1].timestamp`<br>• gebruik `curve[W].timestamp` als `trading_start_at`; de OOS-grens is `curve[oos_offset−1]`<br>• tel per stamp alleen het laatste punt voor annualisatie | quant-strategy-analyst | WP2.5 |

---

## 6. Procesnotities

- **Overtredingen van agentregels (geen ervan wijzigde code):**
  - `git stash list` is vier keer read-only uitgevoerd, door de P2a-producer, de P0-v2-producer, de register-producer en de register-critic.
  - Eén keer is per ongeluk `git stash` uitgevoerd (P2b.1-producer, in de eigen worktree). Dat liet `refs/stash` = `5ebc033` achter. **Die stash bestaat nog en wacht op een beslissing van de gebruiker** (droppen of bewaren).
  - Eén no-op `git checkout` zonder argumenten (P5-critic, in de eigen worktree).
- **Escalatieregel (2 rejections):** niet geraakt. P1 kreeg één REJECT (WP22-P1-CR-01: debug-logtests afhankelijk van globale structlog-state) en werd in ronde 2 goedgekeurd. P0 en P3 kregen een v2 om niet-blokkerende nits te verwerken; dat waren geen rejections.
- **Rapportcorrecties:** de unit-telling in de P2a-producer is gecorrigeerd naar 4784 + 1 (CR-07), en die in de P2b-producer naar 4907 + 1 (CR-04).
- **Overig:** tijdens P0 v2 bleef een scratch-PG-cluster ongeveer een minuut draaien na een timeout. Er waren geen scratch-objecten meer en het cluster is daarna gestopt. Bij alle PG-runs ging het wachtwoord via stdin; aan het eind waren er 0 scratch-rollen en 0 scratch-databases.
