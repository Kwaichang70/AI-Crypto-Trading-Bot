# WP2.5-register — Metric-integriteit (één equity-sample per bar)

**Datum:** 2026-11-06 · **Status:** P0–P3 gecommit (lokaal op de werkbranch; de back-upbranch staat in §1), gates P1, P2 en slotgate PASS; P3 = `b4ceb12` (inhoud gelijk aan `p3-v2.patch`, sha256 `a95c288b…287d`; critic r1 en security r1 beoordeelden `p3.patch`, critic r2 keurde de v2-delta inclusief SEC-12 goed); P4 is dit document · **Branch:** `claude-session-20260924-vp2-fase1-minimumset-xsoezy` · **Basis:** `01d393d` (WP2.3-register inbegrepen) → `606e9c1` → `2b19283` → `8ba7ace` → `b4ceb12` → deze commit (P4).
**Plan:** `Documentation/Verbeterplan-v2-2026-09.md` §4, Fase 2, WP2.5 (bevinding C13).
**Bronnen:** de agent-rapporten in `reports/vp2-wp2.5/` (synthesis-spec, de drie ontwerpen, producer-, critic-, quant-, security- en gate-rapporten, `user-decisions.md`). Die map is gitignored. Dit register is daarom zelfstandig leesbaar; ID's verwijzen naar die rapporten (prefix `WP25-` weggelaten waar dat kan).

---

## 1. Scope en resultaat

**Het defect (C13).** Het HEAD-rapport telde als n het aantal punten van de equity-curve: warmup-punten (vlak op startkapitaal), één punt per bar en één extra punt per fill. Daardoor was de Sharpe verdund, was de n van PSR opgeblazen en was de annualisatie niet gekoppeld aan het aantal bars (synthesis-spec §2: daily W=206 geeft 0,865–0,883×, W=406 geeft 0,733×).

**Wat WP2.5 verandert:**
- **Eén sample per bar (P0).** Nieuwe leaf `packages/trading/equity_sampling.py` (alleen stdlib, AST-gepind). Hij kapt de curve in tot het laatste punt per stamp, begint bij het basispunt `trading_start_at` en neemt per bar-slot de laatste waarde tot en met dat slot (calendar as-of). Een slot zonder eigen punt draagt de vorige waarde (`carried_slots`). Een ontbrekend basispunt faalt gesloten (`equity_sampling_no_base_point`).
- **Eén formule-ingang (P1).** `packages/trading/metrics.py` krijgt `METRICS_VERSION = "wp25-1"`, `KNOWN_METRICS_VERSIONS`, een strikte `periods_per_year`, `return_moments`, `ReturnMetrics` en `compute_return_metrics`. Sharpe, Sortino, PSR en n komen alleen daar vandaan. P1 is gedragsneutraal op de randen na: de zero-std-guard (`std <= 1e-9·|mean|`) en `compute_cagr` zonder `OverflowError` (`inf`).
- **De gedragsschakelaar (P2).**
  - De runner bouwt het sample ná de run uit de curve; engine, portfolio, curve, fills en trades blijven ongemoeid.
  - Sharpe, Sortino, PSR, n, skew/kurtosis, `confidence_flag`, `max_drawdown_duration_bars` en CAGR lezen het sample. `max_drawdown_pct` blijft over alle punten staan (SY-25-05).
  - OOS wordt op tijd gesneden vanaf `oos_start_at`.
  - `BacktestResult` krijgt 7 gedefaulte velden (§2, SY-25-11).
  - CF-26 is bevroren (SY-25-17): `scripts/wp21_rebaseline.py` rekent met twee eigen helpers de pre-WP2.5-formule.
- **API en persistentie (P3).**
  - Marker `engine_semantics.metrics = "wp25-1"` (alleen backtest en optimize).
  - `legacy_metrics` op run- en optimize-responses, voor elk aantal symbolen.
  - Vijf nieuwe velden in `BacktestMetricsResponse`.
  - Leaderboard krijgt een metrics-helft en de teller `legacy_metrics_runs_excluded`.
  - ML-gate: `walk_forward.metrics`-label en een waarschuwing voor oude modellen.
  - Optimize scrubt niet-eindige entry-metrics naar `None`.
  - `backfill_metrics_v2.py` slaat wp25-rijen zonder enige schrijfactie over.
- **Ongewijzigd:** `strategy_engine.py`, `portfolio.py`, `engines/`, `test_wp22_ast_pins.py`, `TestT15Wp25Contract`, T-16, elke golden-constante, Alembic (head 020), paper/live (geen marker, SY-25-19).

**Commits** (lokaal, in volgorde):

| Commit | Hash | Inhoud | Review (rondes, verdict) |
|---|---|---|---|
| P0 | `606e9c1` | Sampler-leaf, geen wiring (3 bestanden, +882) | critic r1 APPROVE_WITH_EDITS; quant r1 MATCH; v2 (`p0-v2.patch`, sha `e08137e1…f43f`): critic r2 APPROVE |
| P1 | `2b19283` | Formule-consolidatie, identieke getallen (5 bestanden, +822/−93) | critic r1 APPROVE_WITH_EDITS (CR-01..03, test-only); quant r1 MATCH_WITH_EDITS (QR-06, QR-11, test-only); v2 (`p1-v2.patch`, sha `8f46ba7a…9431`): critic r2 APPROVE |
| P2 | `8ba7ace` | Gedragsschakelaar + CF-26-freeze (9 bestanden, +1432/−60) | critic r1 APPROVE_WITH_EDITS (CR-01, CR-02, test-only); quant r1 MATCH; v2 (`p2-v2.patch`, sha `32080878…228f`): critic r2 APPROVE |
| P3 | `b4ceb12` | API en persistentie (18 bestanden, +1751/−28) | critic r1 APPROVE_WITH_EDITS (CR-01..04, test-only); security r1 APPROVE (SEC-12 optioneel); v2 (`p3-v2.patch`, sha `a95c288b…287d`) verwerkt CR-01..04 en SEC-12 |
| P4 | deze commit | `Documentation/WP2.5-register.md`, `packages/trading/CLAUDE.md` | — |

De P3-regels in dit register volgen `p3-v2.patch`; `b4ceb12` heeft dezelfde `git patch-id --stable` (`2284c5c2…`), dus de inhoud is gelijk.

**Beslissingen van de gebruiker** (AskUserQuestion, **2026-11-06**; `reports/vp2-wp2.5/user-decisions.md`). Alle vier zijn genomen met **optie (a)**, gelijk aan de default van de spec (§9).

| ID | Beslissing (genomen met optie (a), 2026-11-06) |
|---|---|
| UD-1 | Leaderboard: **uitsluiten.** Runs zonder metrics-marker zijn niet eligible; `/portfolio/summary` telt ze in `legacy_metrics_runs_excluded`. Het leaderboard is leeg tot strategieën opnieuw gedraaid zijn (SY-25-15). |
| UD-2 | Veranderde getallen en optimizer-rankings: **accepteren.** Alleen nieuwe waarden, geen legacy-Sharpe op nieuwe runs. Oude optimizations blijven zoals opgeslagen, gevlagd `legacyMetrics`, en worden niet opnieuw gerankt. |
| UD-3 | ML-activatiegate: **drempel houden** (`min_oos_skill_score = 1.0` en de worst-fold-drempel). Pre-WP2.5-modellen krijgen een waarschuwing; herijken later (CF-25-04). |
| UD-4 | CF-26-rebaseline: **bevriezen.** Het script houdt de pre-WP2.5-formule; T-16 blijft onbewerkt (SY-25-17). |

Daarnaast gelden twee procesbeslissingen van de gebruiker (niet in `user-decisions.md`):
- **"Lokaal tot PR #4 gemerged"** (genomen in WP2.2, blijft van kracht): ook de WP2.5-commits worden niet naar de werkbranch gepusht tot PR #4 gemerged is. `origin/claude-session-20260924-vp2-fase1-minimumset-xsoezy` staat nog op `117236f`.
- **Back-upbranch** `origin/claude-session-20260924-vp2-wp22-backup`: alleen op expliciet verzoek bijgewerkt ("Push naar back-upbranch en drop de stash" → `3d12187`; akkoord voor de WP2.3-commits → `01d393d`; "Zet P0 en P1 ook op de back-upbranch" → `2b19283`). P2 (`8ba7ace`), P3 (`b4ceb12`) en P4 staan er pas op als de gebruiker dat vraagt.

---

## 2. Kernbesluiten (SY-25)

| ID | Besluit | Motivatie |
|---|---|---|
| SY-25-01 | Pure sampling ná de run uit `portfolio.get_equity_curve()`; geen engine- of portfolio-wijziging | Variant (A) (punten onderdrukken) breekt C1/C2, de EquitySnapshot-byte-identiteit en paper/live; variant (C) (parallelle lijst) is een tweede equity-bron |
| SY-25-02 | Leaf `trading/equity_sampling.py` (stdlib, AST-gepind); `METRICS_VERSION`, `KNOWN_METRICS_VERSIONS` en alle formules in `metrics.py` | `metrics.py` importeert pydantic, structlog en `trading.walk_forward` en kan geen gepinde leaf zijn; de versie dekt ook CAGR-span, duur-eenheid en zero-std-guard. `cost_profiles.py` en `schemas.py` importeren uit `trading.metrics` (geen `schemas → services`-cyclus) |
| SY-25-03 | Marker `engine_semantics.metrics = "wp25-1"`, alleen backtest en optimize; `timestamp_clock` en `bar_alignment` niet opgehoogd | Volgt SY-23-16; `promote_to_live` herstempelt met modus `live`, dus de marker lekt niet |
| SY-25-04 | Samplingregel: calendar as-of, laatste punt per stamp, basispunt `_step_close(timeline, W, t0, tf_s)` (`k == 0` geeft `t0`, dus W = 0 werkt), fail-closed zonder basispunt, `present_steps`-terugval met waarschuwing `backtest.metric_sampling_fallback` bij off-grid | Calendar as-of voorkomt de 1/√(1−f)-bias van present-only; de terugval is alleen bereikbaar voor single-symbol bibliotheek-runs |
| SY-25-05 | Alle rendementsmetrics en de drawdown-duur lezen het sample; **`max_drawdown_pct` blijft over alle punten** (QD U-01 (a) overruled) | Anders spreekt `max_drawdown_pct` de `drawdown_pct` van elk punt in hetzelfde resultaat tegen, en de MDD van de portfolio-router (`max(EquitySnapshot.drawdown_pct)`). Verschil ten hoogste één exit-kost (ENG-1: 0,002115 tegen 0,001952), 0 op elke realistische probe |
| SY-25-06 | CAGR over het handelsvenster: `span_days = max(1.0, (trading_end_at − trading_start_at)/86400)`; `compute_cagr(days: float)` houdt de HEAD-rekenwijze en geeft `inf` bij `OverflowError`; `start_date`/`end_date`/`duration_days` ongewijzigd | Integer-aanroepers (ook de CF-26-freeze) blijven byte-identiek; kortere spans raken de 1-dagsvloer vaker (gap G2) |
| SY-25-07 | Strikte `periods_per_year(tf)`; de stille `.get(tf, 365.25)` in de runner is weg; de factor wordt één keer vóór de simulatie bepaald | Een toekomstig 2h/6h-lid moet luid falen, niet daily-annualisatie krijgen (QR-10(e): `run_backtest` wordt dan nooit bereikt) |
| SY-25-08 | PSR: n = `sample.n_returns`; momenten uit dezelfde returns via `compute_return_metrics`; `_PSR_MIN_OBSERVATIONS = 30` en de drempels 0,80/0,95 ongewijzigd; DSR ongewijzigd | Echte DSR is WP2.7 (CF-25-01) |
| SY-25-09 | Zero-std-guard: `std == 0` of `std <= 1e-9·|mean|` geeft Sharpe 0,0; `return_moments` geeft dan (0,0; 3,0) | De bekende faal `test_constant_returns_zero_std_returns_zero` wordt groen zonder testwijziging (P1) |
| SY-25-10 | OOS op tijd: `oos_start_at = _step_close(timeline, oos_start_index, t0, tf_s)`; `oos_sharpe` en `oos_n_returns` uit één `compute_return_metrics`-resultaat (QR-10(a)) | Gat-vrij: `oos_n_returns == N − oos_start_index`, onafhankelijk van het aantal OOS-fills. Het HEAD-bereikcheck en de `oos_offset is None`-tak behouden hun semantiek |
| SY-25-11 | `BacktestResult` krijgt 7 gedefaulte velden: `metrics_version`, `trading_start_at`, `trading_end_at`, `periods_per_year`, `metric_sampling`, `carried_slots`, `oos_start_at` | Directe constructies blijven werken; `metrics_span_days` afgewezen (afleidbaar) |
| SY-25-12 | API-velden: `trading_start_at`, `trading_end_at`, `periods_per_year`, `metric_sampling`, `carried_slots`; **niet** `oos_start_at`, **niet** `metrics_version` | De API-backtest geeft nooit `oos_start_index`; `metrics_version` is redundant met de marker |
| SY-25-13 | `metrics_response_fields(config, run_mode)` → `{"legacy_metrics": bool}`; True iff modus backtest/optimize en marker geen `str` in `KNOWN_METRICS_VERSIONS`; **geen single-symbol-uitzondering**; faalt nooit, logt nooit | C13 raakt elke run; geen mapping van onbekende modus naar backtest |
| SY-25-14 | Oude rijen: alleen labelen, nooit herrekenen of herschrijven | 90 dagen retentie, W niet gepersisteerd, wall-clock-stamps vóór WP2.2, index-gepaarde rijen vóór WP2.3 |
| SY-25-15 | Leaderboard: `leaderboard_eligible` = trades/confidence EN kosten EN metrics-marker; teller `legacy_metrics_runs_excluded` (kosten eerst, metrics tweede) | UD-1 (a) |
| SY-25-16 | ML: `extra["walk_forward"]["metrics"] = METRICS_VERSION`; een block zonder bekend label wordt normaal gepoort en geeft bij slagen een vaste waarschuwing; geen auto-reject; drempels ongewijzigd | UD-3 (a); de waarschuwing is een module-constante en echoot nooit het opgeslagen label |
| SY-25-17 | CF-26-record bevroren: `_pre_wp25_sharpe` en `_pre_wp25_cagr` in `scripts/wp21_rebaseline.py`; T-16 en `test_wp21c_rebaseline.py` onbewerkt | PR #4 (head `117236f`) wacht op een baseline van die engine; relabelen zou een ander getal geven dan de code van `117236f` kan maken. UD-4 (a) |
| SY-25-18 | `backfill_metrics_v2.py`: een rij met bekende marker wordt vóór elke herberekening of schrijfactie overgeslagen (`skipped_wp25`); legacy-rijen ongewijzigd | Zonder guard zou het script correcte wp25-waarden overschrijven met sample-stdev-momenten |
| SY-25-19 | Paper/live ongewijzigd; geen marker | Daar bestaat geen equity-curve-metric; de promotiegate leest alleen tellingen |
| SY-25-20 | Vier exact-dict-pins krijgen precies `"metrics": "wp25-1"`; `test_p3_03`: `oos_n_returns` 85 → 70 en 62 → 51; goldens, T-15, T-16 en `test_wp22_ast_pins.py` onbewerkt | Zie §4 |
| SY-25-21 | Referentiewaarden gepind op sample-momenten (de schatter van de runner); normale momenten alleen in één documenterende assertie | Verklaart de CF-23-05-overdrachtswaarden |
| SY-25-22 | Optimize scrubt niet-eindige entry-metrics naar `None` bij de ORM-write en in de POST- en GET-one-respons (`_scrub_non_finite`); `_extract_metrics` en ranking ongewijzigd | SY-25-06 kan nu `+inf` geven waar HEAD een `OverflowError` gaf |
| SY-25-23 | Geen nieuwe holding-time-metric (CF-22-04); een pin-test bewaakt `bars_in_market == |∪[entry_at, exit_at)|/tf` | `exposure_pct` is al de intervalunie |
| SY-25-24 | QA-06/07/08: gridstamps zijn bar-close (`close_i`); een open-gelabelde benchmarkreeks schuift +1 tf; `end_date` ongewijzigd, `trading_end_at` erbij | Alleen documentatie, geen code |
| SY-25-25 | Alleen backend; de UI-slice is CF-25-03 | SY-25-15 is het tussentijdse vangnet |
| SY-25-26 | Volgorde P0 → P1 → P2 (met CF-26-freeze) → P3 (marker) → P4 | De marker mag niet vóór de getallen landen; tussen P2 en P3 lezen API-runs als legacy (veilige richting) |

Rulings uit de reviews:
- **P2 quant QR-10:** `trading_end_at` is `None` bij een leeg sample (zie §4 voor de exacte formulering).
- **P3 critic, ruling 1:** `test_sprint50_cycle5_model_gate.py` (één fixture-regel, `extra={"walk_forward": {"metrics": "wp25-1"}}`) is een goedgekeurde edit buiten de schrijflijst. De gate behandelt "geen bekend label" uniform; de smallere regel ("alleen waarschuwen als er een block is") faalt open voor precies de rijen met de minste herkomst.
- **P3 critic, ruling 2:** `exc_info=True` op `runs.backtest_metrics_parse_error` is vervangen door `error_type` (Pydantic-tekst echoot de opgeslagen, te manipuleren waarde). Security SEC-08: voeg nooit `loc`-waarden uit `exc.errors()` aan die log toe.
- **P3 critic, ruling 3:** getypeerde readers (`_datetime_attr`, `_finite_float_attr`, `_sampling_mode_attr`, `_carried_slots_attr`) in plaats van kale `getattr`; een kale `MagicMock`-result wordt anders door Pydantic geweigerd.
- **P3 critic, ruling 4 (CR-08):** het legacy-pad van `backfill_metrics_v2.py` houdt zijn eigen `.get(tf, 8_766.0)`-terugval; die is alleen bereikbaar voor rijen zonder marker. De runner heeft geen terugval meer.
- **P2 producer, afwijking 1:** `tests/unit/test_wp25_metric_identity.py` is in P2 omgezet naar de T-17-allow-list-vorm (buiten de schrijflijst); `head_capture.json` is niet geregenereerd.

---

## 3. Spec-wijzigingen ten opzichte van Verbeterplan v2, WP2.5

| Onderwerp | Verbeterplan v2 | Gerealiseerd | Bron |
|---|---|---|---|
| Sampling | "equity één keer per bar gesampled ná warmup" | Post-run sample uit de curve, calendar as-of vanaf `curve[W].timestamp`; de curve zelf is ongewijzigd | SY-25-01/04 |
| Annualisatie | "per timeframe" | Strikte `periods_per_year`; de stille terugval is weg | SY-25-07 |
| PSR/DSR-n | "n = bars" | PSR: ja. **DSR is niet gewijzigd** (geen n-input); echte DSR is WP2.7 | SY-25-08, CF-25-01 |
| Referentiewaarden | "uit bekende synthetische reeks" | REF-A..D en ENG-1 uit gesloten vorm, `rel=1e-9` | SY-25-21 |
| Drawdown | niet genoemd | `max_drawdown_pct` blijft alle punten; alleen de duur gaat naar het sample | SY-25-05 |
| CAGR/Calmar | niet genoemd | Over het handelsvenster; `compute_cagr(days: float)` met overflow-guard | SY-25-06 |
| Zero-std-guard | niet in het plan | **Toegevoegd** in P1 | SY-25-09 |
| API-marker, legacy-vlag, leaderboard, ML-label, backfill-guard, optimize-scrub | niet in het plan | **Toegevoegd** in P3 | SY-25-13/15/16/18/22 |
| CF-26-freeze | niet in het plan | **Toegevoegd** in P2 | SY-25-17 |
| Inspanning | 1 cyclus | Spec 1,6 cycli (P0 0,2; P1 0,3; P2 0,45; P3 0,4; P4 0,1; reviewrondes 0,15) | synthesis-spec §5 |
| `trading_end_at` bij een leeg sample | — | `None` (de letterlijke regel gaf een omgekeerd venster) | P2 afwijking 2, QR-10 |
| P3-omvang | ca. 170 regels productiecode | ca. 320 (veldbeschrijvingen, getypeerde readers) | P3-producer, afwijking 4 |

**Verouderde regelverwijzingen in het plan (BD-02).** Verbeterplan C13 (regel 54) en rij 2.5 (regel 151) noemen `strategy_engine.py:672-675`, `backtest.py:459`, `metrics.py:433, 894`. Op `01d393d` staan op die regels exit-config-code (`strategy_engine.py:668-681`), een niet-gerelateerd `run()`-signatuurgebied (`backtest.py:461`), het eind van `compute_cagr` (`metrics.py:433`) en `compute_trade_statistics` (`metrics.py:894`). De plantekst is niet aangepast (CF-25-09). Live plaatsen:

| Wat | Op `01d393d` (BD-02) | Na P2/P3 (gelezen op `8ba7ace` + `p3-v2.patch`) |
|---|---|---|
| Annualisatie en returns in de runner | `backtest.py:694-698` | `backtest.py:582` (`ann_factor`), sample en metrics `:715-742` |
| OOS, n en PSR in de runner | `backtest.py:753-780` | `backtest.py:750-796` |
| Sharpe | `metrics.py:456-501` | `metrics.py:562` (`compute_sharpe`), `:605` (guard) |
| PSR-minimum | `metrics.py:909` (`_PSR_MIN_OBSERVATIONS`) | `metrics.py:1023` |
| Nieuwe ingang | — | `metrics.py:1168` (`return_moments`), `:1189` (`ReturnMetrics`), `:1204` (`compute_return_metrics`) |
| Sampler | — | `equity_sampling.py:181` (`sample_equity_per_bar`) |

---

## 4. Testbewijs

| Gate / meting | Basis | Unit | Integratie (pass/fail/skip/xfail) | Overig | Verdict |
|---|---|---|---|---|---|
| Baseline | `01d393d` (code gelijk aan `2e2c79a`) | 5550 + 1 (WP2.3-slotgate) | 1058 / 3 / 2 / 1 | 829 ruff-bevindingen | — |
| Review P0 v2 | `01d393d` + v2 | `test_wp25_sampling` + leaf-pin: 77 passed (3×, ook `-p no:randomly`) | — | 6/6 mutanten gedood (critic r2) | APPROVE |
| **Gate P1** (`p1-gate-testing.md`) | `2b19283` | 5737 passed, 0 failed | 1058 / 3 / 2 / 1 | WP25-tests 186 (3×; omgekeerde bestandsvolgorde; `python -O -o filterwarnings=` 186 + 1 warning); goldens: 591 passed in 9 bestanden, T-16 groen; `mypy --strict` 0 nieuwe; ruff 829 = 829 | PASS |
| **Gate P2** (`p2-gate-testing.md`) | `8ba7ace` | 5767 passed, 0 failed | 1106 / 3 / 2 / 1 | WP25-tests 263 (3×; omgekeerd; `-O`); goldens/rebaseline/bar-clock: 547 passed, `TestT16RebaselineUnaffected` 1 passed, `TestT15Wp25Contract` 3 passed; T-24 stabiel; `mypy --strict` 0 in 4 bestanden; ruff 829 = baseline; geen wijziging aan protected paths | PASS |
| Review P3 r1 (`p3-critic-r1.md`) | `8ba7ace` + `p3.patch` | 5944 passed | 1106 / 3 / 2 / 1 | 54 mutanten, 50 gedood, 4 overleefd (P1, O2, O6, R5) → CR-01..04; mypy 0; ruff 56 op basis en patch | APPROVE_WITH_EDITS |
| Review P3 security r1 (`p3-security-r1.md`) | `8ba7ace` + `p3.patch` | aangrenzende suites 6033 passed, 0 failed | — | 50 probes passed; mutanten A en B gedood | APPROVE |
| Producer P3 v2 (`p3-v2-producer-report.md`) | `8ba7ace` + `p3-v2.patch` | 5949 passed, 0 failed | 1106 / 3 / 2 / 1 | wp25 API- en backfill-tests 182 passed; mutanten P1, O2, O6, R5 en SEC-12-revert gedood; mypy 0 op 8 bestanden | — |
| **Gate P3 / slotgate** | `b4ceb12` | **PASS**: unit 5949/0; integratie 1106 passed, alleen de 3 bekende fouten met ongewijzigde foutmelding t.o.v. `01d393d`; goldens, T-16 en `TestT15Wp25Contract` groen en onbewerkt; 445 WP25-tests 3× groen, ook omgekeerd en onder `-O` (met `-o filterwarnings=`); mypy --strict 0 nieuw; ruff 829 = basislijn; T-24 byte-identiek over 3 processen; Alembic-head 020 | | | open |

- **Bekende failures, al aanwezig op de baseline:**
  - unit: `test_metrics.py::TestComputeSharpe::test_constant_returns_zero_std_returns_zero` is **sinds P1 groen** (SY-25-09, zonder testwijziging; P1-gate: 5737 passed, 0 failed);
  - integratie, nog steeds rood met ongewijzigde faalmodus:
    - `test_portfolio_endpoints::TestAggregatePortfolio::test_aggregate_with_backtest_metrics` (`assert None == 0.25`; de omgeving heeft geen Postgres);
    - `test_sprint50_cycle5_integration::...::test_check_oos_eligibility_passes_all_gates` (`warning == ''`: de legacy z-score-waarschuwing staat er);
    - `test_sprint50_cycle5_integration::...::test_check_oos_eligibility_with_worst_fold_floor` (`True is False`);
  - skips: `test_cf36_ws_realio.py` (`wsproto` ontbreekt); xfail: `test_smk_t_20` (CF-SMK-S1).
- **Goldens:** geen enkele digest- of golden-constante is gewijzigd (SY-25-20). T-16 (`TestT16RebaselineUnaffected`) en `TestT15Wp25Contract` zijn onbewerkt en groen. De diff `01d393d..8ba7ace` raakt `strategy_engine.py`, `portfolio.py`, `engines/` en `test_wp22_ast_pins.py` niet (P1- en P2-gate).
- **Pins (P3):** vier exact-dict-pins krijgen elk precies `"metrics": "wp25-1"` (`test_wp21_cost_api.py`, `test_wp21c_breaker_default.py`, `test_wp22_api_semantics.py`, `test_wp23_api.py`); de paper-pin is onaangeroerd.
- **Andere bewerkte tests (P2/P3):** `test_wp23_runner.py::test_p3_03` (1 regel: `(1430, -2.156, 85), (1449, -1.159, 62)` → `(1430, -2.3805, 70), (1449, -1.2842, 51)`; quant-geverifieerd: −2,380472646681639 en −1,284152401449698; de oude pins reproduceren met de oude methode); `test_sprint50_cycle6_oos_backtest.py` (3 docstrings plus nieuwe test C6-031, waarden C6-010/011/030 ongewijzigd); in P3 de fixtures `test_sprint49_m5_leaderboard_transparency.py`, `test_wp21_api_units.py`, `test_wp21b_optimize_cost.py` en `test_sprint50_cycle5_model_gate.py` (marker erbij; alleen de laatste staat niet in de spec-lijst).

**Testinventaris WP25** (telling met `pytest --collect-only` op `8ba7ace` + `p3-v2.patch`; totaal 445):

| Bestand | Tests | Commit | Dekt |
|---|---|---|---|
| `tests/unit/test_wp25_sampling.py` | 53 | P0 | T-01..T-06 |
| `tests/unit/test_wp25_ast_pins.py` | 43 | P0 (leaf-deel), P2 (runner-deel) | T-05, T-23 |
| `tests/unit/test_wp25_metrics.py` | 96 | P1 | T-07, T-08, T-10 |
| `tests/unit/test_wp25_metric_identity.py` | 23 | P1, in P2 omgezet naar T-17 | T-09, T-17 |
| `tests/integration/test_wp25_runner.py` | 48 | P2 | T-11..T-16, T-18..T-22, T-24, T-25, CF-22-04 |
| `tests/unit/test_wp25_api.py` | 160 | P3 | T-26..T-30, T-32..T-34 |
| `tests/unit/test_wp25_backfill_guard.py` | 22 | P3 | T-31 |
| `tests/fixtures/wp25/head_capture.json` | — | P1 | HEAD-capture van 10 scenario's, op de onbewerkte boom op `01d393d` |

Gate-tellingen van de WP25-bestanden: 186 bij P1, 263 bij P2. De 445 bij P3 is 263 + 182 (wp25 API- en backfill-tests uit de P3 v2-producer, `p3-v2-producer-report.md`); die som is nog niet in een gate herhaald.

**AST-pins** (`test_wp25_ast_pins.py`, `test_wp25_metrics.py`):
- leaf: exacte import-whitelist, geen env/klok/logging/I/O, float alleen in de returnformule, mutated-source-zelftests;
- runner: all-points-returns (`compute_returns_from_equity`) alleen op de allow-list `scripts/wp21_rebaseline.py::_pre_wp25_sharpe` en `scripts/backfill_metrics_v2.py::_recompute_psr_from_snapshots`; geen aanroep in `backtest.py`, `optimizer.py`, `walk_forward.py` of `apps/`; **geen productie-aanroep geeft `mode=` aan de sampler** (ook `**kwargs` telt, QR-03); alle rendementsmetrics via één `compute_return_metrics`-route; geen inline `** 3`/`** 4`-momenten; `oos_equity_curve_offset` alleen in de `is None`-check en de info-log;
- `backtest.py` heeft geen `TIMEFRAME_PERIODS_PER_YEAR` en geen `.get(` met `365.25` of `PERIODS_PER_YEAR`.

### Oud tegen nieuw, 10 harness-scenario's

Bron: `p2-producer-report.md` ("Observed metric values"; oud = capture op `01d393d`, nieuw = P2-patch). Alle tien zijn `calendar_asof` met `carried_slots` 0 en `confidence_flag` "low" vóór en na. Elke nieuwe waarde is bit-voor-bit gelijk aan het onafhankelijke orakel op de HEAD-curve; `max_drawdown_pct` is in alle tien gelijk aan HEAD. n is exact `N − W`.

| Sc | Sharpe oud → nieuw | n oud → nieuw | CAGR oud → nieuw | PSR oud → nieuw | MDD-duur oud → nieuw | OOS Sharpe / n oud → nieuw |
|---|---|---|---|---|---|---|
| s01 1h single legacy | −0,255181 → −0,272276 | 634 → 550 | −0,00173423 → −0,00181613 | 0,472682 → 0,472849 | 210 → 194 | - |
| s02 1d single coinbase | −0,95755 → −1,04741 | 445 → 370 | −0,00233735 → −0,00264648 | 0,158135 → 0,148895 | 351 → 306 | - |
| s03 1h frictionless | −0,850622 → −0,913167 | 634 → 550 | −0,00484821 → −0,00507681 | 0,4096 → 0,409623 | 364 → 345 | - |
| s04 1h OOS coinbase | −13,1651 → −13,3494 | 645 → 550 | −0,100693 → −0,105194 | 0,00021992 → 0,000233914 | 503 → **515** | −18,7375 / 196 → −18,0098 / 180 |
| s05 1d OOS legacy | 0,636093 → 0,69142 | 440 → 370 | 0,00109729 → 0,0012427 | 0,763531 → 0,760902 | 228 → 183 | 2,70786 / 126 → 2,80534 / 120 |
| s06 1h multi3 holes | −4,47289 → −5,08095 | 584 → 450 | −0,0567396 → −0,0604057 | 0,12565 → 0,125837 | 526 → 443 | - |
| s07 1h multi3 holes OOS | −22,6802 → −23,3022 | 597 → 450 | −0,286726 → −0,302614 | 3,52266e-09 → 5,79327e-09 | 531 → 435 | −10,8468 / 138 → −10,8294 / 120 |
| s08 1d multi2 holes | −2,49865 → −2,60843 | 451 → 350 | −0,00871136 → −0,00992488 | 0,00333177 → 0,00274707 | 385 → 317 | - |
| s09 1h W=0 | 2,12479 → 2,18627 | 322 → 300 | 0,0131883 → 0,0126574 | 0,657897 → 0,656509 | 158 → 145 | - |
| s10 1h multi2 W=0 holes | 3,78031 → 4,04175 | 326 → 300 | 0,0366399 → 0,0351488 | 0,773266 → 0,776762 | 203 → 187 | - |

De multi-symbol-gaten in deze scenario's worden door een ander symbool gedekt, dus `carried_slots` is overal 0; het all-absent-slot staat in T-14 (n 60 = `total_bars` 59 + `carried_slots` 1).

### Quant-correcties op de spec-formuleringen (`p2-quant-r1.md` QR-10/QR-11/QR-12)

Gebruik deze formuleringen; de spec (§9 UD-2, SY-25-16, QD-13) is op deze punten te stellig.
- **`trading_end_at`** is `None` **alleen als de run stopte vóór het handels-basispunt** (leeg sample: n 0, Sharpe 0,0, PSR `None`, CAGR 0,0). `trading_start_at` blijft gezet. Stop precies op W geeft `end == start` en CAGR 0,0.
- **PSR** verschuift in de tien scenario's met −0,0092 (s02, kurtosis 26,9 → 16,3) tot +0,0035 (s10); drie liggen boven 0,002 (s02, s05 −0,0026, s10), de andere zeven binnen 0,0014 (de quant noemt −0,0093 voor s02; de tabel geeft −0,0092). "±0,002" uit UD-2 is te smal.
- **Drawdown-duur** kan **langer** worden (s04: 503 → 515). Een pre-fee tussenmark binnen één bar kwam boven de end-of-bar-piek en zette de HEAD-telling terug; de spell begint 54 bars eerder en 42 fill-punten tellen niet meer mee (+54 − 42 = +12). Dat is de juiste end-of-bar-semantiek. De bewering "de duur krimpt met het aantal fill-punten" (QD-13) geldt niet universeel. In s04 meten `max_drawdown_pct` (alle punten) en de duur vanaf verschillende pieken; geaccepteerd onder SY-25-05 en CF-25-16.
- **OOS-Sharpe**: de richting hangt af van het kostenprofiel en het teken van de Sharpe (s04 magnitude −3,9%, s05 +3,6%). De "+2–5%" uit SY-25-16 geldt alleen voor modellen met positieve Sharpe.
- **Sharpe-magnitude** stijgt in 10 van 10 (single-symbol, lage kosten: oud/nieuw ≈ √(n_nieuw/n_oud)); het weglaten van warmup dunt altijd minder uit. Onder het coinbase-profiel verlaagt het weglaten van fill-punten |Sharpe| juist voor deze negatieve-Sharpe-runs (s04 −13,17 → −12,77 in de decompositie).
- **CAGR** verandert in deze scenario's puur door de span: `log(1+nieuw)/log(1+oud) = duration_days/span` exact in alle tien. Voor W = 0 (s09, s10) krimpt hij, omdat de HEAD-vloer van 12 dagen korter was dan het echte venster van 12,5 dag.
- **CAGR/Calmar over korte vensters** (QR-11): wiskundig juist, economisch een extrapolatie (ENG-1: CAGR 25,12 = 2512%; Calmar 67 058 en 4 619). Toon CAGR en Calmar als "geannualiseerd vanuit de span in dagen" of markeer ze onder ca. 90 dagen (voor CF-25-03). Zie CF-25-17 voor de optimizer-ranking.

---

## 5. Carry-forwards en open punten

Een eigenaar of doel met "(v)" erachter is een voorstel van de opsteller. Zonder "(v)" komt hij uit het genoemde bronrapport. CF-25-01..16 komen uit de synthesis-spec §8; CF-25-17..24 zijn door de P4-opsteller genummerd voor de review-toevoegingen (de inhoud komt uit de genoemde reviewrapporten).

### 5.1 Uit de spec (CF-25-01..16)

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| CF-25-01 | Bailey–López de Prado DSR per trial/fold: `sample_equity_per_bar(result.equity_curve, start_at=result.oos_start_at or result.trading_start_at, ...)`; SR_pp, n en momenten uit dat ene sample; folds samengevoegd op tijd. Ook: DSR persisteren/tonen (nu weggelaten in `optimize.py`), `oos_start_at` in de API, fold-warmup-pre-roll, en wat "DSR > 0" betekent in WP2.8 | quant-strategy-analyst (v) | WP2.7 |
| CF-25-02 | Paper-versus-backtest: een paper-Sharpe alleen via de leaf, as-of op het bar-close-grid, nooit uit wall-gestempelde fill-punten | quant-strategy-analyst (v) | WP2.8 |
| CF-25-03 | UI: types `legacyMetrics`, `tradingStartAt`, `tradingEndAt`, `periodsPerYear`, `metricSampling`; badge en tooltips; compare-banner bij gemengde versies; leaderboard-groepssleutel met metrics-versie; CSV-kolom; optimize-header-badge; ook `legacyTimestamps` en `legacyBarAlignment` (CF-23-07) tonen | nextjs-frontend-specialist | Vóór een WP2.8-getal gelezen wordt |
| CF-25-04 | Herijk `min_oos_skill_score` (1,0), `min_worst_fold_skill_score` en `min_trades_per_fold` op de gemeten wp25-1-foldverdeling; `fold_max_dd` bevat de 200-bar train-staart | quant-strategy-analyst (v) | Na meting; samen met CF-25-17 |
| CF-25-05 | Optionele offline herberekening voor `wp22-1` + `wp23-1`-rijen binnen de retentie (dry-run default, nieuwe markerwaarde, nooit in-place) | python-backend-specialist (v) | Op verzoek |
| CF-25-06 | Exposure-noemers (steps) en n (kalender) lopen uiteen bij runs met gaten; documenteer in WP2.8-rapportage | quant-strategy-analyst (v) | WP2.8 |
| CF-25-07 | De terminale MTM draagt geen liquidatiekosten; totaal rendement is ca. één exit-kost te optimistisch voor runs die in een positie eindigen (WP2.8-rapportregel) | quant-strategy-analyst (v) | WP2.8 |
| CF-25-08 | Toekomstige ulcer/omega/CVaR/exposure-adjusted/benchmark-metrics moeten `BarSample`-returns nemen; benchmark-joins op `close_i` (+1 tf voor open-gelabelde series) | quant-strategy-analyst (v) | Bij invoering |
| CF-25-09 | Verouderde regelverwijzingen in Verbeterplan C13 en rij 2.5; live plaatsen in §3 | primary (docs) | Plantekst nog niet aangepast |
| CF-25-10 | `apps/api/CLAUDE.md` noemt `GET /runs/{run_id}/metrics`, dat bestaat niet | primary (docs) | Follow-up |
| CF-25-11 | Backtest-equityrijen worden na `equity_snapshot_retention_days` (90) gepruned, dus controle achteraf uit rijen is niet gegarandeerd | — | Alleen documentatie (§6) |
| CF-25-12 | Een idempotente create die over de deploy heen wordt herhaald, geeft de opgeslagen pre-WP2.5-body (informatief) | — | Alleen documentatie |
| CF-25-13 | `backfill_metrics_v2.py` is legacy-only na SY-25-18; zijn sample-stdev-momenten verschillen van de populatiemomenten van de runner. Zie ook P3 critic CR-06 en security SEC-10 (§5.2) | python-backend-specialist (v) | Pensioneren na WP2.8 |
| CF-25-14 | `_PSR_MIN_OBSERVATIONS = 30` telt nu **bars** (zie §6) | — | Verwacht gedrag, gedocumenteerd |
| CF-25-15 | Sortino-LPM-over-N-variant, alleen als die ooit gewenst is (verschuift elke Sortino met √((n−1)/n)) | quant-strategy-analyst (v) | Op verzoek |
| CF-25-16 | End-of-bar-MDD (`max_drawdown_eob_pct`) als WP2.8 Calmar op één sample wil | quant-strategy-analyst (v) | Op verzoek |

### 5.2 Toegevoegd door de reviews

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| CF-25-17 | **P2 QR-11 (b):** `cagr` en `calmar_ratio` zijn kiesbare optimizer-rankingmetrics (`optimizer.py:66-75`). Op HEAD had elke trial op dezelfde data dezelfde `duration_days`; nu is de span `N − W` per trial. Dat is de bedoelde W-neutrale rate, maar op korte folds versterkt de convexiteit van `(1+R)^(365.25/d)` de ruis (1% op een 1h-fold: ca. 79% CAGR over 150 u, ca. 472% over 50 u). Sharpe is niet geraakt. Oplossing: een minimum-spanguard of folds op een gemeenschappelijk venster rangschikken. (QR-11 (a), de weergave van korte-venster-CAGR/Calmar, staat in §4 en CF-25-03.) | quant-strategy-analyst (v) | WP2.7 / CF-25-04 |
| CF-25-18 | **P0 QR-12:** de sampler heeft geen bovengrens-slice. Een in-run IS/OOS-split of in-run foldvensters hebben een `until_stamp(at)`-tegenhanger nodig, of aparte per-fold-runs. Folds samenvoegen op `stamps` en een gedeelde basis niet dubbel tellen. Hoort bij CF-25-01. | quant-strategy-analyst (v) | WP2.7 |
| CF-25-19 | **P3 SEC-04 (Info, bestaand):** `/portfolio/summary`-eligibility negeert de run-modus (`select(RunORM.config)` over alle modi, `portfolio.py:681` in de P3-boom). Alleen directe DB-manipulatie kan een paper-rij met `backtest_metrics` en marker maken. Optie: filter op `run_mode == "backtest"`. | python-backend-specialist (v) | Backlog |
| CF-25-20 | **P3 SEC-07 (Low, bestaand):** `GET /optimize/{id}` geeft 500 als een opgeslagen entry-metric niet numeriek is (`float(v)` op bijv. `"abc"` gooit vóór de scrub). Oplossing: `try/except (TypeError, ValueError)` die naar `None` mapt. Alleen DB-manipulatie kan dit veroorzaken. | security-audit-specialist / python-backend-specialist | Backlog |
| CF-25-21 | **P3 SEC-10 (Low, operationeel):** `metrics_v2_backfilled` staat standaard op `false`, dus elke nieuwe wp25-backtestrij is bij elke aanroep van het backfill-script kandidaat en wordt met zijn equity-snapshots geladen; de guard is de enige bescherming en de kosten groeien met het aantal runs. Als het script blijft leven: filter de kandidaatquery in SQL op de marker. Valt onder CF-25-13. | python-backend-specialist (v) | Met CF-25-13 |
| CF-25-22 | **P3 critic CR-05 (Info):** de regel "`str` in `KNOWN_METRICS_VERSIONS`" staat op vier plaatsen: `schemas.py` (`is_leaderboard_metrics_eligible`), `cost_profiles.py` (`metrics_response_fields`), `scripts/backfill_metrics_v2.py` (`_has_known_metrics_marker`) en `model_activation_gate.py` (label-check). Ze komen vandaag overeen en elk is getest; er is driftrisico. Overweeg één helper `has_known_metrics_marker(config)`. | python-backend-specialist (v) | Een later WP |
| CF-25-23 | **P3 critic CR-06 (Info):** de backfill-guard slaat alleen **bekende** markers over. Een toekomstige markerwaarde (bijv. `wp26-1`) die niet in `KNOWN_METRICS_VERSIONS` staat, zou als legacy worden teruggeschreven. Breid `KNOWN_METRICS_VERSIONS` uit **voordat** een nieuwe marker landt. | Het WP dat de volgende marker invoert | Bij de volgende marker |
| CF-25-24 | **P0 QR-11 (Info):** `BarSample.from_stamp` met een off-grid `at` in calendar-modus geeft `mode="calendar_asof"` terwijl de gridinvariant `n == (end − start)/tf` breekt. Alleen WP2.7-foldgrenzen moeten gridstempels zijn (afgeleid uit `_step_close`); de docstring zegt dat nu. | quant-strategy-analyst (v) | WP2.7 |

### 5.3 Nog open uit WP2.3 (blijven geldig)

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| WP2.3 P4-SEC-04 | Live runs met een dubbel symbool zijn na de WP2.3-deploy niet meer te resumen (fail-closed is correct). **De deploy-SQL staat hieronder (§6) en is nog open.** Open: de deploy-gate zelf, een ruling over ontdubbelen in protective-mode, en een boot-log `recovery.live_duplicate_symbols`. Zie `Documentation/WP2.3-register.md` §5 | devops (gate) + risk-management-expert (ruling) | Gate vóór de rollout |
| WP2.3 P4-SEC-02 (**High**, bestaand) | Backtest en optimize blokkeren de event-loop van de enkele uvicorn-worker (O(n²): 13,8 s bij 8k bars; ca. 15 u geëxtrapoleerd op de grens). Zolang dat duurt staan kill switch, emergency-stop en live-taken stil. **WP2.5 heeft hier niets aan veranderd; de waarschuwing blijft open.** Oplossing: off-loop (`asyncio.to_thread`) met een semaphore(1), of tot dan een CPU-budget of geen synchrone backtest naast een lopende live run. | risk-management-expert + python-backend-specialist | Vóór een live run naast API-backtests |
| CF-26 | Coinbase-re-baseline van WP2.1c is nog niet gedraaid (403 van de proxy) en **blokkeert PR #4**. Het record is bevroren op `117236f` (SY-25-17, UD-4 (a)): een CF-26-run op een latere commit levert nog steeds de baseline die PR #4 verwacht. | quant-strategy-analyst | Vóór de merge van PR #4 |

De overige open punten van het WP2.3-register (CF-23-xx, P4-SEC-03/06/07/09 enz.) blijven ongewijzigd; WP2.5 sluit er geen. CF-23-05 (as-of-equity-sampling) **is gesloten** door P0–P2: de overdrachtswaarden (SR 31,052536 / 31,734025 / 32,273953, n 100 / 100 / 97, eindequity 16284,270878) zijn gereproduceerd (SY-25-21).

---

## 6. Deploy-notities

- **Leaderboard leeg tot herruns (UD-1).** Een run zonder `engine_semantics.metrics` is niet eligible. `/portfolio/summary.legacy_metrics_runs_excluded` telt de uitgesloten runs die wel aan de trades-, confidence- en kostenhelft voldoen; `legacy_cost_runs_excluded` beweegt niet. Pas na het opnieuw draaien van strategieën vult het leaderboard zich.
- **ML-modellen.** Een model zonder bekend `walk_forward.metrics`-label wordt normaal gepoort; bij slagen krijgt het de waarschuwing `walk-forward Sharpe predates WP2.5 (metrics wp25-1); retrain via POST /ml/train`. Al actieve modellen worden niet opnieuw gepoort (spec SY-25-16). De drempels blijven 1,0 (UD-3). Een model net onder 1,0 kan na hertraining wel slagen; voor positieve-Sharpe-modellen is de verschuiving ca. +2–5% (zie de OOS-correctie in §4).
- **Oude optimizations** blijven zoals opgeslagen, gevlagd `legacyMetrics`, en worden niet opnieuw gerankt; draai opnieuw wat je gebruikt (UD-2).
- **CF-26 bevroren.** Het record staat op `117236f`; `scripts/wp21_rebaseline.py` rekent met `_pre_wp25_sharpe` en `_pre_wp25_cagr` de oude formule (SY-25-17). Verander ze niet; de T-16-digest bewaakt het (mutatietest: de nieuwe Sharpe of CAGR in het record laat T-16 falen).
- **Getallen veranderen op API-runs** (zie de tabel in §4): n daalt naar `N − W`, |Sharpe| stijgt, CAGR volgt de span, de drawdown-duur telt bars, `max_drawdown_pct` is ongewijzigd.
- **Geen migratie.** Alleen JSONB-sleutels (`runs.config`, `optimization_runs.run_metadata`, `model_versions.extra`); Alembic head blijft 020 (gepind in `test_wp25_api.py`). Oude rijen worden nooit herschreven (SY-25-14).
- **Retentienotitie (CF-25-11).** Backtest-equityrijen worden gepruned zodra `runs.created_at` ouder is dan `equity_snapshot_retention_days` (default 90; `apps/api/config.py`, `services/retention.py`). Een metric kan daarna niet uit de rijen worden gecontroleerd; W is bovendien niet gepersisteerd. Daarom is "alleen labelen" de enige eerlijke keuze voor oude runs.
- **Versie-bumpbeleid.** Elke latere wijziging aan een gepersisteerd, uit de curve afgeleid getal hoogt `METRICS_VERSION` op (SY-25-03). Breid `KNOWN_METRICS_VERSIONS` uit samen met elke nieuwe marker: de backfill-guard slaat alleen bekende markers over (CF-25-23), en `legacy_metrics`, de leaderboard-helft en de ML-gate lezen dezelfde set (CF-25-22). `timestamp_clock` en `bar_alignment` hebben hun eigen versies en worden door een metrics-bump niet geraakt.
- **PSR n ≥ 30 telt nu bars.** `_PSR_MIN_OBSERVATIONS = 30` (`metrics.py:1023`; `compute_psr` geeft `None` bij `n_observations < 30`) vergelijkt met het aantal per-bar-returns na warmup, niet meer met curve-punten. Een daily-run met minder dan 30 bars na de warmup heeft dus `psr = None` en `confidence_flag = None`. De spec (CF-25-14) schrijft "minder dan 31 bars"; de code vergelijkt `n < 30` met n = aantal returns. De drempel zelf is ongewijzigd.
- **Deploy-gate van WP2.3 (SEC-04), nog open.** Draai deze query vóór de rollout en flatten of stop elke treffer:

```sql
SELECT id, status FROM runs
WHERE run_mode = 'live' AND status IN ('running','orphaned','resuming')
  AND jsonb_typeof(config->'symbols') = 'array'
  AND jsonb_array_length(config->'symbols')
      <> (SELECT count(DISTINCT s) FROM jsonb_array_elements_text(config->'symbols') s);
```

- **Event-loop-waarschuwing (WP2.3 P4-SEC-02), nog open.** Zie §5.3: een synchrone backtest of optimize houdt de kill switch en de live-taken van de enkele worker stil. WP2.5 maakt de backtest niet duurder in O-termen, maar lost het ook niet op.
- **Idempotente create over de deploy (CF-25-12).** Een herhaald create-verzoek met dezelfde idempotency-sleutel geeft de opgeslagen pre-WP2.5-body.

---

## 7. Procesnotities

- **Escalatieregel (2 rejections):** niet geraakt. Er was geen enkele REJECT. P0, P2 en P3 kregen APPROVE_WITH_EDITS (test-only edits, v2); P1 kreeg MATCH_WITH_EDITS van de quant (test-only).
- **P3 delta-check:** `p3-v2.patch` verwerkt de edits van critic r1 en security r1, inclusief de productiewijziging SEC-12 (`_finite_float_attr`); critic r2 (`p3-critic-r2.md`) keurde die delta goed (APPROVE). Dit register is gebaseerd op de code van `8ba7ace` + `p3-v2.patch` (sha256 `a95c288b879c2dfc5a0ad826af804d046180b53223b7e011850db8b277ad287d`, 18 bestanden, +1751/−28). `b4ceb12` heeft dezelfde `git patch-id --stable` (`2284c5c2…`) als `p3-v2.patch`.
- **Rapportcorrecties:** de P1-producer meldde dat de QA-06-opmerking (gridstamps zijn bar-close) "niet is toegevoegd"; hij stond er al sinds P0 en P2 maakte de formulering expliciet (alleen docstring). De SY-25-23-tekst "tot `trading_end_at`" is in P2 gecorrigeerd naar `trading_end_at + tf` (de engine telt de laatste bar). `deflated_sharpe_ratio` heeft een vierde gedefaulte parameter `use_full_formula`; T-25 pint de eerste drie namen en dat de optimizer er precies drie keywords aan geeft.
- **Overtreding van een agentregel (wijzigde niets):**
  - De P4-producer voerde één keer `git stash list` uit, read-only, in de eigen wegwerp-worktree (`p4-producer-report.md`, punt 6). Dat overtreedt de regel "geen git stash, nergens". Er is geen stash-actie uitgevoerd.
  - Een worktree deelt `refs/stash` met de hoofd-repo; die ref bestaat niet (door de P4-critic read-only gecontroleerd met `git rev-parse --verify refs/stash`).
  - Precedent: het WP2.2-register meldt vier read-only `git stash list`-runs en één per ongeluk uitgevoerde `git stash`.
- **Hygiëne:** per rapport geen `.env` gelezen, geen netwerk, en geen checkout/stash/reset/clean in de hoofd-repo (met de uitzondering hierboven); elke agent verwijderde alleen de eigen worktrees. Serena, MCP memory, sequential-thinking en context7 waren niet beschikbaar in de rapporten die ze noemen.
