# WP2.4-register — Point-in-time MTF-context (Fase A)

**Datum:** 2026-11-08 · **Status:** Fase A (P0–P3) gecommit (lokaal op de werkbranch), Fase-A-slotgate (testing-deel) PASS op `b669894`; P4 is dit document. **Fase B (P5 store/migratie 021, P6 backfill-CLI, P7 API-PIT-loading) is nog open** (UD-2 (b)). · **Branch:** `claude-session-20260924-vp2-fase1-minimumset-xsoezy` · **Basis:** `bbe7142` (einde WP2.5) → `a9459c3` → `2697090` → `d18192b` → `b669894` → deze commit (P4).
**Plan:** `Documentation/Verbeterplan-v2-2026-09.md` §4, Fase 2, WP2.4 (bevinding C10).
**Bronnen:** de agent-rapporten in `reports/vp2-wp2.4/` (synthesis-spec, de vier ontwerpen, producer-, critic-, quant-, security-, risk-, synthesis- en gate-rapporten, `user-decisions.md`). Die map is gitignored. Dit register is daarom zelfstandig leesbaar; ID's verwijzen naar die rapporten. `SY-24-nn` komt uit `synthesis-spec.md`, `CF-24-nn` uit zijn §8, `CF-Pn-nn`, `CF-SYN-nn` en `QR-nn` uit de synthese- en reviewrapporten van de genoemde commit.

---

## 1. Scope en resultaat

**Het defect (C10).** `apps/api/main.py` haalt `get_latest()` alleen bij het opstarten op (regels 306, 469, 512, 538). Een API-backtest kreeg daardoor in elke bar dezelfde bevroren opstartwaarden (FGI, BTC-dominantie, yield-spread, whale-flow) plus een wall-clock-gebaseerde "7d geleden"-waarde. Research-scripts kregen `None`. Het resultaat hing af van het tijdstip van de laatste herstart en API en script weken af (synthesis-spec §2 en verificatielijst; `quant-design.md` QD-01).

**Wat Fase A verandert** (P0–P3, geen migratie, geen API-PIT-loading):
- **Leaf (P0).** `packages/data/point_in_time.py`: alleen stdlib, AST-gepind. Decimal-waarden, integer-milliseconden, as-of-join op `close_i` met `available_at <= close_i` inclusief, de lag zit in `available_at` (geen lag-parameter in de join), veroudering gemeten vanaf de eerste beschikbaarheid, "7d geleden" = `asof(close_i − 7d)`, dekking per veld met drempel `MIN_COVERAGE_PPM = 990_000`. `parse_jsonl_lines` heeft een lag-vloer en een basis-allowlist per serie.
- **Adapter en engine (P1).** `packages/trading/mtf_signals.py`: `UNAVAILABLE`, `PitSignals`, `LIVE_SIGNALS`. `StrategyEngine.__init__(..., mtf_signals=None)` met een guard per modus. Een BACKTEST-engine leest nooit meer een singleton of de wandklok. Het live-blok is verplaatst, niet herschreven (zie §3, correcties).
- **Selectie en rapport (P2).** `BacktestRunner` en `ParameterOptimizer` kiezen één keer per run welke velden aan staan (`select_mtf_signals`) en bewaren `last_mtf_context`. `BaseStrategy.consumes_mtf_signals` (frozenset, expliciet per strategie) en de statussen `not_used`/`unavailable`/`partial`/`pit`.
- **API (P3).** Marker `engine_semantics.mtf_context = "wp24-1"` (alleen backtest en optimize), blok `config["mtf_context"]` / `run_metadata["mtf_context"]`, velden `legacyMtfContext` en `mtfContextStatus`, en een vierde helft in de leaderboard-regel met de teller `legacy_mtf_context_runs_excluded`.
- **Effect in Fase A.** De API geeft geen bron mee (SY-24-12), dus elke API-backtest draait met `UNAVAILABLE`. Dat is gelijk aan het oude "geen clients"-pad: geen enkele golden-constante is gewijzigd en API = script.
- **Ongewijzigd** (gate-items 2 en 3, `phaseA-gate-testing.md` §2 en §3): `portfolio.py`, `engines/`, `run_orchestrator.py`, `data/{sentiment,market_signals,macro_data,whale_tracker,ml_features}.py`, elk Alembic-bestand (head blijft 020), `test_wp22_ast_pins.py`, `test_multi_timeframe.py` en `tests/integration/test_wp22_bar_clock.py`. Paper/live gedrag is byte-identiek en draagt geen marker.

**Fasering (SY-24-13, UD-2 (b)).**
- Fase A = P0–P4: klaar met dit document.
- **Fase B = P5–P7, nog open:** P5 `context_observations` plus migratie `021_add_context_observations.py`, P6 `scripts/backfill_context_observations.py` (FGI en FRED T10Y2Y, `--dry-run` default), P7 API-PIT-loading uit de store. Pas daarna is WP2.4 volledig klaar. De Fase-B-gate staat in synthesis-spec §7.
- Fase C (forward recorder) is op gebruikersbesluit verplaatst naar de live-safety-WP van CF-24-01.
- ML-trainingsjoin op PIT: uitgesteld (CF-24-03, bij WP2.7).

**Commits** (lokaal, in volgorde; elke commit heeft dezelfde `git patch-id --stable` als zijn v2-patch):

| Commit | Hash | Inhoud | Reviews (rondes, verdict) |
|---|---|---|---|
| P0 | `a9459c3` | Leaf `point_in_time.py` (479 regels) plus 2 testbestanden; 3 bestanden, +1701 | critic r1 APPROVE_WITH_EDITS (CR-01..07); quant r1 MATCH_WITH_EDITS (QR-01..14, QR-08 en QR-09 vereist); synthese `p0-synthesis.md` REVISE (E1..E13); v2 (`p0-v2.patch`, sha256 `a5f02053…a99b`): critic r2 APPROVE, quant r2 MATCH (nieuw: QR-15) |
| P1 | `2697090` | Adapter `mtf_signals.py` (288 regels; 549 sinds P2) en engine-swap; 6 bestanden, +2405/−97 | critic r1 APPROVE_WITH_EDITS (CR-01..07); security r1 APPROVE (SEC-03 Low, optioneel); risk r1 APPROVE (RISK-05..10); synthese REVISE (E1..E5); v2 (`p1-v2.patch`, sha256 `094be087…3db3508`, bestandsset gelijk): critic r2 APPROVE, security r2 APPROVE; `p1-closeout.md` (geen aparte synthese-ronde) |
| P2 | `d18192b` | Runner/optimizer-selectie, rapport, `consumes_mtf_signals`; 17 bestanden, +2366/−8 | critic r1 APPROVE_WITH_EDITS (CR-01..12); quant r1 MATCH_WITH_EDITS (QR-04 en QR-07 vereist, QR-08 aanbevolen); synthese REVISE (E1..E11); v2 (`p2-v2.patch`, sha256 `e7107ffe…dda150`): critic r2 APPROVE, quant r2 MATCH; `p2-closeout.md`. Security niet vereist (tripwire vrij: geen wijziging aan `LiveSignals`, `check_backtest_source`, allowlists) |
| P3 | `b669894` | API-marker, vlaggen, leaderboard; 17 bestanden, +1459/−18 | critic r1 APPROVE_WITH_EDITS (CR-01 Medium, CR-02 en CR-03 Low, CR-04..06 Info; alle edits test-only); security r1 APPROVE_WITH_EDITS (E1, test-only); synthese REVISE (E1..E5); v2 (`p3-v2.patch`, sha256 `13db6520…13017d`): critic r2 APPROVE, security r2 APPROVE; `p3-closeout.md` |
| P4 | deze commit | `Documentation/WP2.4-register.md`, regel in `WP2.2-register.md`, drie `CLAUDE.md`-bestanden | — |

- P0–P3 (`bbe7142..b669894`): 39 bestanden, +7925/−117 (gate §3), inclusief tests.
- Elke commit kreeg precies één r1-ronde met edits (de v2's bevatten ook productiewijzigingen: P0 E1/E3/E6/E12, P1 E4, P2 E1/E2, P3 E3/O1; zie §2) en daarna een r2-APPROVE. Er was geen REJECT.

**Beslissingen van de gebruiker** (`reports/vp2-wp2.4/user-decisions.md`):

| ID | Beslissing |
|---|---|
| UD-1 | **(a) WP2.4 vóór WP2.6.** Dit wijkt af van de afhankelijkheid in het plan. WP2.4 neemt migratie 021; WP2.6 rebaset (CF-24-14). |
| UD-2 | **(b) Fase A + B nu.** De forward recorder gaat naar de CF-24-01-WP. Gevolg: CoinGecko- en whale-geschiedenis begint pas als die WP landt en is niet achteraf te herstellen. **Status:** de recorder (P7) zit in R1; gesloten na G-D en de R1-tag (`CF-24-01-register.md`). |
| UD-4 | **(a) Inerte modellen houden;** context-getrainde modellen zijn geen bewijs en worden later opnieuw getraind (CF-24-03/07). Geen code in WP2.4. |
| CF-24-01 | Bevroren live-context: **aparte live-safety-WP** (risk + security), direct na Fase A. |
| UD-3 en UD-5 | **Niet apart gevraagd.** De spec-default (a) is genomen: UD-3 (a) = `legacy_mtf_context` alleen voor consumer-strategieën, leaderboard-uitsluiting en teller; UD-5 (a) = FGI-lag 24 uur. De gebruiker kan dit overrulen (`user-decisions.md`). |

Daarnaast gelden procesbeslissingen van de gebruiker (niet in `user-decisions.md`):
- **"Lokaal tot PR #4 gemerged"** (genomen in WP2.2, blijft van kracht): de WP2.4-commits worden niet naar de werkbranch gepusht. `origin/claude-session-20260924-vp2-fase1-minimumset-xsoezy` staat nog op `117236f` (gelezen bij het opstellen).
- **Back-upbranch** `origin/claude-session-20260924-vp2-wp22-backup`: alleen op expliciet verzoek van de gebruiker bijgewerkt (beleid sinds WP2.2). Voor WP2.4 vroeg de gebruiker "Zet P0 en P1 ook op de back-upbranch" (→ `2697090`) en daarna "Zet P2 ook op de back-upbranch" (→ `d18192b`); geverifieerd met reflog en `git ls-remote`. De branch bevat nu alles tot en met P2 (`d18192b`); P3 en P4 staan er nog niet op. **P3 (`b669894`) en P4 staan er nog niet op.**

---

## 2. Kernbesluiten (SY-24)

| ID | Besluit | Motivatie / bron |
|---|---|---|
| SY-24-01 | Leaf `packages/data/point_in_time.py`: stdlib (zoals gebouwd: `bisect`, `collections.abc`, `dataclasses`, `datetime`, `decimal`, `hashlib`, `json`, `types`, `typing`), geen logging, env, klok of I/O, AST-gepind. Waarden zijn `Decimal`, niet float. Afgewezen: `mtf_pit.py`, `ContextSeriesSet`, `canonical_hash.py` | Naam niet MTF-specifiek (ML-join en WP2.6 hergebruiken hem); een float-hash is platformafhankelijk |
| SY-24-02 | Adapter `packages/trading/mtf_signals.py`: `SignalValues`, `BacktestSignalSource` (`UnavailableSignals`/`UNAVAILABLE`, `PitSignals`), `LiveSignals`/`LIVE_SIGNALS` (**geen** `BacktestSignalSource`). Guards in `StrategyEngine.__init__` volgens het WP2.2-klokpatroon, alleen typenamen in meldingen. Orchestrator niet bewerkt. Zie de correcties in §3 | Een Protocol zonder guard laat een live-adapter in een backtest toe |
| SY-24-03 | De lag wordt één keer bij ingestie in `available_at` gezet; de join heeft **geen** lag-parameter. `available_at_estimated()` wordt alleen door ingestiecode aangeroepen (in de leaf: de vloercontrole van `parse_jsonl_lines`) | Recorded rijen hebben hun echte beschikbaarheid al; één leessemantiek; de digest pint de lag |
| SY-24-04 | Serieregels als code-constanten (één reviewd blok): FGI +24 u / max 48 u; T10Y2Y vintage `realtime_start` + 1 d of fallback + 2 werkdagen + 1 d / max 7 d; BTC-dominantie en whale alleen `recorded` (6 u / 2 u); FEDFUNDS en de 24h-veranderingen worden niet geserveerd (`not_served`). Veroudering vanaf de **eerste beschikbaarheid**, inclusief. "7d geleden" = `asof(close_i − 7d)`. Grens: `available_ms <= decision_ms` is zichtbaar; bij gelijke `observed_ms` wint de grootste `available_ms` | De fallback is bewust één werkdag conservatiever dan het quant-voorstel (feestdagen); verworpen: AD's 7d-tolerantieregel en DP's veroudering vanaf `observed_at` |
| SY-24-05 | Dekking: alles-of-niets **per veld per run** bij `>= 990_000` ppm over de handelsstappen `k ∈ [W, N)`. Nul stappen = UIT. Een AAN-veld geeft `None` op de ≤1% ongedekte bars. Optimizer en walk-forward: één keer over het volledige uitgelijnde venster; fold-lokale gaten zijn geaccepteerd | Per-bar-`None` geeft structuurbreuken binnen een run; alles-of-niets over alle series zou FGI voor altijd weggooien omdat whale geen geschiedenis heeft |
| SY-24-06 | `consumes_mtf_signals: ClassVar[frozenset[str]]` op `BaseStrategy`, expliciet per registerstrategie (test-afgedwongen), standaard alle negen velden (fail-closed). `EnsembleStrategy.mtf_signal_fields()` = unie van de subs. Waarden: rsi_mean_reversion en dca_rsi_hybrid `{fear_greed_index, btc_dominance, yield_curve_spread, whale_net_flow}`; model_strategy `{fear_greed_index, fear_greed_index_7d_ago, btc_dominance, btc_dominance_7d_ago}`; ma_crossover, breakout, grid_trading, sl_tp_reversion, momentum_breakout en smoke_roundtrip `frozenset()` | Partial versus pit hangt af van WELKE velden gelezen worden (gelezen in de code op `b669894`) |
| SY-24-07 | Marker `engine_semantics.mtf_context = "wp24-1"` binnen het bestaande `backtest/optimize`-blok; `timestamp_clock`, `bar_alignment` en `metrics` niet opgehoogd. `config["mtf_context"]`: `version`, `source` (`unavailable`/`pit`), `status` (`not_used`/`unavailable`/`partial`/`pit`), `fields`, `series`, `series_digest`, `ingest_cutoff`, `reason`. `FieldStatus` is gesloten: `pit`, `no_source`, `insufficient_coverage`, `not_served`. `not_used`-runs krijgen altijd `UNAVAILABLE`. `BacktestResult` krijgt geen veld. Zie de gloss-correctie in §3 | G8: `regime_at_entry` wordt voor elke strategie berekend; een PIT-bron zou anders de trades van niet-consumers wijzigen |
| SY-24-08 | `mtf_context_response_fields(config, run_mode, strategy_name=None) → {"legacy_mtf_context", "mtf_context_status"}`. `legacy_mtf_context` is True iff modus backtest/optimize, marker geen bekende `str`, en de naam is een consumer of ontbreekt/is geen bruikbare `str` (fail-closed). Het consumer-set is een bevroren historisch feit (`LEGACY_MTF_CONSUMER_STRATEGIES`). Faalt nooit, logt nooit | Bevestigd via UD-3 (a) als spec-default |
| SY-24-09 | Opslag (Fase B): tabel `context_observations`, PK `(series_id, observed_at, available_at)`, kolom `available_at_basis`, `NUMERIC(24,8)`, schrijven met `NOT EXISTS` + `ON CONFLICT DO NOTHING`, migratie `021` met downgrade-guard, store in `apps/api/services/context_store.py`. **Nog niet gebouwd** | Fase B |
| SY-24-10 | HTF-filter blijft op de bar-open-sleutel (byte-identiek). Verplaatsen naar `close_i` is CF-24-12 | Geen productie-aanroeper geeft `htf_bars` |
| SY-24-11 | Leaderboard: vierde helft in de ene regel (`is_leaderboard_mtf_eligible`); `/portfolio/summary.legacy_mtf_context_runs_excluded`; tellervolgorde kosten, metrics, mtf. `partial`/`unavailable`/`not_used` blijven eligible met label | UD-3 (a) |
| SY-24-12 | Fase A: geen PIT-loading in de API. Fase B: de store wordt vóór de runner geladen, store-fout geeft `UNAVAILABLE` met reden `store_unavailable`; geen request-veld, geen env-schakelaar, geen upload | Fase B |
| SY-24-13 | Fase A = P0–P4 apart uitrolbaar; B = P5–P7; recorder volgens UD-2; ML-trainingsjoin uitgesteld. **P1–P3 in één release** | Tussen P1 en P3 lezen nieuwe runs als legacy (veilige richting) |
| SY-24-14 | Bestaande ML-modellen: geen code; behandeling volgens UD-4 (a); classificatie is CF-24-07 | — |
| SY-24-15 | WP2.6-volgorde: technisch beide kanten op mogelijk; WP2.4 eerst (UD-1 (a)). `series_digest` komt naast `presence_digest` en de toekomstige `ohlcv_data_hash`, nooit samengevoegd | CF-24-14 |
| SY-24-16 | Paper/live byte-identiek; CF-24-01 gaat naar een eigen live-safety-WP | — |
| SY-24-17 | Determinisme: integer-ms, Decimal-tekst in de digest, `ingest_cutoff` genoteerd. Twee runs zijn vergelijkbaar iff `presence_digest` en `series_digest` gelijk zijn | — |
| SY-24-18 | Nieuwe code logt alleen `exc_log_fields`/`error_type`; de verplaatste `LiveSignals`-code logt niets. Bestaande `str(exc)`-plaatsen zijn CF-24-02 | Geen live-module-edits in de verplaatsing |

**Rulings uit de reviews:**
- **P0 synthese C1:** QR-08 wint van CR-03. `available_at_estimated` en `available_at_vintage` weigeren elke invoer die niet 00:00 UTC is (`pit_bad_value`), want een stempel oostelijk van UTC gaf tot circa 2 à 2,6 dagen look-ahead.
- **P0 synthese C2:** de fix voor CR-01 zit in `Observation.__post_init__` (`abs()` vervangen door `copy_abs()`), niet alleen in de `except`-takken.
- **P0 synthese C4/C5:** de JSONL-vloer valideert alleen en herschrijft `available_ms` nooit; plus een allowlist van toegestane bases per serie. De pin op ingestie-aanroepers is een exacte lijst (`parse_jsonl_lines -> available_at_estimated` en `-> available_at_vintage`) plus een alias-scan.
- **P1 synthese E4:** de BACKTEST-guard is fail-closed (zie §3).
- **P2 synthese E2 (QR-08):** een AAN-veld wordt alleen geserveerd als het geconsumeerd is, of de now-partner van een geserveerd geconsumeerd 7d-veld. QR-07: `reason` beschrijft alleen geconsumeerde velden.
- **P3 synthese E1:** één per-naam importer-pin voor `apps/` (helper, `apps/`-invariant en een 15-variant zelftest) die ook vier dynamische bypasses (ARB-1..4) sluit; E3/O1: een fail-closed regel in de naamcontrole.

---

## 3. Correcties op de spec (CF-P1-03, CF-P2-04, CF-P2-08)

| Onderwerp | Spec-tekst | Gerealiseerd en correcte formulering | Bron |
|---|---|---|---|
| Verplaatst live-blok | SY-24-02: "HEAD `:2129-2216` verbatim verplaatst" | **`:2129-2193` is verbatim verplaatst** (65 regels, sha256 `8c31b7d5…c5447`, gepind in `test_wp24_live_parity.py`). De non-None-regel van `:2195-2216` is herschreven als `SignalValues.to_context_kwargs` (dezelfde 9 tests, dezelfde volgorde) | CF-P1-03 (CR-05), CF-P2-08 |
| BACKTEST-guard | SY-24-02: "`isinstance(x, BacktestSignalSource)`" | Een **"echte subclass"-guard**: `ABCMeta.register()` wordt niet gehonoreerd en een `LiveSignals` wordt nooit geaccepteerd. Code: `isinstance(x, LiveSignals)` of `not isinstance(x, BacktestSignalSource)` of `BacktestSignalSource not in type(x).__mro__` geeft `ValueError` (alleen typenaam). De accept-set is een strikte subset van de spec, dus fail-closed | P1 E4 (SEC-03, RISK-05), CF-P1-03 |
| `FieldStatus.pit` | SY-24-07: "`pit` (served)" | **`pit` = "passed coverage; served iff consumed, or the now-partner of a served consumed 7d field"**. Een ongeconsumeerd veld met status `pit` is dus niet per se geserveerd. Afleiden: `source == "pit"` ⇔ `status ∈ {pit, partial}`; vlaggen en leaderboard lezen alleen `status` | CF-P2-04, CF-P2-02 |
| G8 (niet-consumers) | G8: een PIT-bron mag de trades van niet-consumers niet wijzigen | **Gegeneraliseerd G8 geldt per serie:** trades hangen alleen via de series die geconsumeerde velden voeden van de store af. Residu: een consumer van uitsluitend een 7d-veld krijgt ook het now-veld van die serie; voor FGI stuurt dat veld `regime_at_entry`. Binnen de geconsumeerde serie, PIT-veilig, geen registerstrategie geraakt (`test_every_consumed_7d_field_has_its_now_partner_in_the_set` slaagt voor alle 9). De ensemble houdt G8 op ensemble-niveau (de unie) | CF-P2-04, `p2-quant-r2.md` QR2-06 |
| Aantal exact-dict-pins | T-27: "4 exact-dict-pins"; "`test_wp25_api` pint alleen paper/live" | Vijf exact-dict-pins plus een pin op de hele `run_metadata`; de tweede bewering is onjuist | CF-P3-02 (3) |
| Optimize-status | — | De `mtfContextStatus` van een optimize-run kan afwijken van een enkele run-backtest van de beste combinatie, in beide richtingen (de dekking wordt één keer over het hele venster bepaald; grens `floor(0,01·N)/(N − W_c)`) | CF-P2-05 (QR-05, QR-04) |
| Inspanning | spec Fase A 1,35 cycli | niet achteraf gemeten in de rapporten | — |

**Geaccepteerde producent-afwijkingen in P3** (CF-P3-02, ter informatie): (1) `LEGACY_MTF_CONSUMER_CLASS_NAMES`, omdat optimize-rijen `strategy_cls.__name__` opslaan; (2) de ene regel staat in `schemas.is_legacy_mtf_context` en `cost_profiles` omhult hem, wegens een importcyclus; (3) de pin-aantallen hierboven; (4) `promote_to_live` verwijdert `mtf_context`; (5) `mtfContextStatus` is `None` voor paper en live; (6) een onbekende, niet-lege, niet-blanco naam wordt niet gevlagd; (7) de importer-allowlist is alleen met `schemas.py` verbreed, met vastgepinde namen.

---

## 4. Testbewijs

Bron: `reports/vp2-wp2.4/phaseA-gate-testing.md` (gate T-33, HEAD `b669894`, read-only, ini-filters actief behalve bij `-O`).

| Meting | Resultaat |
|---|---|
| Unit | **6623 passed, 0 failed** |
| Integratie | 1129 passed, 3 failed, 2 skipped, 1 xfailed; de 3 fouten zijn exact de bekende set |
| WP24-tests | 8 bestanden, **697 tests**: 674 unit (api 124, ast_pins 141, engine 72, live_parity 127, point_in_time 136, runner 74) en 23 integratie (c10 9, optimizer 14). Drie runs groen, omgekeerde bestandsvolgorde groen, `python -O -o filterwarnings=` groen |
| mypy `--strict packages apps/api` | 0 fouten (111 bronbestanden; basis 109, +2 nieuwe modules) |
| ruff | **824 tegen 829 op de basis**, geen nieuwe bevinding. Verdwenen: 4× S110 in `strategy_engine.py` (verplaatst naar `mtf_signals.py` onder file-level noqa) en 1× F401 `typing.Any` in `backtest.py` |
| Determinisme | byte-identiek over twee processen (`PYTHONHASHSEED` 1 en 4242) voor rsi (7 trades) en dca (14 trades), PIT-bron, `series_digest` `d7995979…5df5e1c6` |
| Goldens | geen enkele digest- of golden-constante bewerkt; T-16 (`TestT16RebaselineUnaffected`) groen; de drie beschermde testbestanden hebben een lege `git diff bbe7142` |
| Verboden paden | lege diff; Alembic head 020 |
| Paper/live | `engine_semantics("paper")` en `("live")` bevatten geen `mtf_context`; live-parity 127 passed; promotie-tests plus live-parity 147 passed |

- **Bekende failures, al aanwezig op de basis** (gate §1, identiek aan `bbe7142`, alleen logtijdstempels verschillen):
  - `test_portfolio_endpoints::TestAggregatePortfolio::test_aggregate_with_backtest_metrics` (`assert None == 0.25`; geen Postgres);
  - `test_sprint50_cycle5_integration::TestModelActivationGateServiceIntegration::test_check_oos_eligibility_passes_all_gates` (legacy z-score-waarschuwing);
  - dezelfde klasse, `test_check_oos_eligibility_with_worst_fold_floor` (`True is False`).
  - Skips: `test_cf36_ws_realio.py` (`wsproto`); xfail: `test_smk_t_20`.
- **Per-commit tellingen uit de r2-reviews** (uitgevoerd door de reviewers, niet allemaal op dezelfde basis gemeten): unit 6132 (P0 r2 critic), 6371 (P1 r2 critic), 6481 (P2 r2 critic); integratie bij P1 en P2 telkens alleen de 3 bekende fouten. Gebruik de gate-cijfers hierboven als eindwaarde.
- **Harnasnotitie (QR2-07):** gate-runs gebruiken de ini-warningfilters. Een CLI-`-W error` overschrijft het ini-`ignore` en laat 138 reeds bestaande AsyncMock-tests falen, ook op HEAD.
- **Gate-items die buiten testing vielen** (gate "Not covered here"): item 6 (sign-offs: quant P0/P2, security en risk P1, security P3) en item 7 (dit register). De sign-offs zijn de r1/r2-verdicts in §1; de gate zelf heeft ze niet beoordeeld.

### Gemeten effecten van C10 (voorstel- en ontwerpfase)

Bron: `quant-design.md` QD-04 en §6. Meting met de echte `BacktestRunner` op `bbe7142`: 730 dagelijkse GBM-bars, σ = 60%, 12 seeds per regime, `legacy_backtest_v0`-kosten, geen brackets, één symbool. Dit zijn synthetische ontwerpmetingen, geen metingen op productiedata.

| Strategie | Effect van de bevroren C10-context (t.o.v. geen context) |
|---|---|
| `dca_rsi_hybrid` | Rendement **×0,44 tot ×1,17** (all-bear tot all-bull in een bull-regime; FGI 80 alleen: ×0,69; FGI 10: ×1,13). Live-sizing-factor ×0,33..×1,83 (QD-02) |
| `rsi_mean_reversion` | Trades **verwijderd of omgezet in buy-and-hold:** bij all-bear 0 trades en 0 exposure; bij FGI 80 rendement ×0,08; bij all-bull verdwijnen de SELLs (×18, buy-and-hold) |
| ML (`ModelStrategy` v2) | Inert getrainde modellen: geen effect. Context-getrainde modellen: **10–24% van de beslissingen verandert**; status `partial`/`unavailable`, dus niet eligible |
| `momentum_breakout` (en sl_tp_reversion, grid, ma_crossover, breakout) | **Niet geraakt** (status `not_used`) |

De bias is niet eenzijdig; hij volgt het regime op het opstartmoment. Dat het getal bij een neutrale snapshot (FGI 50, geen FRED/whale-sleutels) identiek is aan de scriptuitkomst is de reden dat Fase A geen golden wijzigt.

---

## 5. Carry-forwards en open punten

Een eigenaar met "(v)" is een voorstel van de opsteller; zonder "(v)" komt hij uit het bronrapport.

### 5.1 Uit de spec (CF-24-01..22, synthesis-spec §8)

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| CF-24-01 | **LIVE SAFETY.** Paper/live-"latest"-waarden (FGI, CoinGecko, FRED, whale) zijn bevroren bij het opstarten; `get_latest` draait alleen op `main.py:306, 469, 512, 538`. De FGI-7d-delta mengt een bevroren "nu" met een meebewegend "7 dagen geleden". Live dca en rsi dimensioneren posities (×0,33..×1,83) en context-getrainde v2-modellen voorspellen op verouderde input. Fix: periodieke refresh, per-serie-verouderingsgrens in live, fail-closed bij verouderde context; de forward recorder sluit hier aan | risk-management-expert + security-audit-specialist, met trading-engine-architect | **Eigen WP, direct na Fase A** (UD-besluit). **Status:** opgelost in R1 (P2-P7), gesloten na G-D en de R1-tag; zie `CF-24-01-register.md` |
| CF-24-02 | Log-hygiëne: `error=str(exc)` op `sentiment.py:269, 320`, `market_signals.py:296, 410`, `retraining.py:164, 276, 309, 323, 345, 352, 646` → `exc_log_fields`; controleer of een CoinGecko-sleutel erin kan staan | security-audit-specialist | CF-36-vervolg. **Status:** de 11 sites zijn gedaan in CF-24-01 P1 (R0, tag `cf2401-r0`) |
| CF-24-03 | v2-training joint FGI op de bar-open-index zonder `available_at` (`ml_features.py:653-666`) en gebruikt maar 30 punten; herbouwen op `data.point_in_time` bij `close_i` met train/serve-pariteitstest | data-pipeline-engineer | WP2.7 |
| CF-24-04 | Latente HTF-`reindex(ffill)`-lek in `build_extended_feature_matrix` (`:636-650`) | data-pipeline-engineer (v) | WP2.7 |
| CF-24-05 | De live-7d-nearest-snapshot-regel plus `datetime.now()` (`sentiment.py:518`, `market_signals.py:542`) → as-of-semantiek | CF-24-01-WP | Met CF-24-01. **Status:** opgelost in R1 (P3 boek, P4a: `LiveSignals` leest het boek met as-of-semantiek); gesloten na G-D en de R1-tag |
| CF-24-06 | Per-bar sidecar-hot-swap van `ModelStrategy` in backtests (`model_strategy.py:514-516`) | trading-engine-architect (v) | Nog toe te wijzen |
| CF-24-07 | Bestaande modellen classificeren op `feature_importances`, context-getrainde hertrainen of labelen (UD-4); schema v3 met expliciete "missing"-codering | quant-strategy-analyst + data-pipeline-engineer (v) | WP2.7 / ML-WP |
| CF-24-08 | FRED-`.`-sentinel geeft `None` in plaats van de laatste geldige waarde | data-pipeline-engineer (v) | Backlog. **Status:** opgelost in R1 (P2, `limit=5` met observatiedatum); gesloten na G-D en de R1-tag |
| CF-24-09 | FGI-geschiedeniscache niet gesleuteld op `limit` | data-pipeline-engineer (v) | Backlog. **Status:** opgelost in R1 (P2, altijd `limit=30` gecachet); gesloten na G-D en de R1-tag |
| CF-24-10 | Legacy `regime_at_entry`/`signal_context` op trades van vóór de marker: `adaptive_learning.py` en `performance_analyzer.py` moeten ze uitsluiten | quant-strategy-analyst (v) | Nog toe te wijzen |
| CF-24-11 | `test_multi_timeframe` wist alleen de FGI-singleton | testing-quality-specialist (v) | WP4.8 (v) |
| CF-24-12 | HTF-sleutel naar `close_i` zodra een productie-aanroeper `htf_bars` meegeeft | data-pipeline-engineer (v) | Bij die aanroeper |
| CF-24-13 | `fed_funds_rate`, `market_cap_change_24h`, `total_volume_change_24h` hebben geen consumer: verwijderen of later serveren | data-pipeline-engineer (v) | Backlog |
| CF-24-14 | WP2.6: migratie rebasen op 021; `ohlcv_data_hash` naast `series_digest`, nooit samengevoegd; canonieke-tekstregel hergebruiken | database-architect | WP2.6 |
| CF-24-15 | De ML-gate meet v1-foldmodellen, niet het uitgerolde v2; geschud splitsen met overlappende labels; RetrainingService kan mogelijk activeren zonder WF-gate (verifiëren) | quant-strategy-analyst | WP2.7 |
| CF-24-16 | Providerfeiten verifiëren vóór de P6-`--apply`: FGI-publicatietijd en `limit=0`, ALFRED `realtime_start`/`output_type`, whale-geschiedenisdiepte, egress via de proxy. Een FGI-lag van 1 u pas na eigen opgenomen snapshots | operator + data-pipeline-engineer | Vóór P6 `--apply` |
| CF-24-17 | BTC-dominantie-backfill (CoinGecko PRO, of CSV met verplichte herkomst en granulariteitslag) | data-pipeline-engineer (v) | Nog toe te wijzen |
| CF-24-18 | Replay-filter `ingested_at <= ingest_cutoff` | data-pipeline-engineer (v) | Nog toe te wijzen |
| CF-24-19 | UI: badges `legacyMtfContext` en `mtfContextStatus`, compare-banner, "context off"-waarschuwing voor consumer-strategieën. Bind badges aan `mtfContextStatus`, nooit aan `config.mtf_context.*` (CF-P3-01) | nextjs-frontend-specialist | UI-backlog |
| CF-24-20 | Script-side store-loader (scripts staan standaard op `UNAVAILABLE`; JSONL werkt nu al) | database-architect (v) | Na P5 |
| CF-24-21 | Fold-lokale dekkingsgaten van `ParameterOptimizer` (SY-24-05) meten als WF-resultaten van consumers ertoe doen | quant-strategy-analyst (v) | WP2.7 |
| CF-24-22 | De "bekende marker"-regel bestaat nu voor metrics en mtf op meerdere plaatsen; houd ze in de pas met `KNOWN_*` | python-backend-specialist (v) | Een later WP |

R1-status: zie `Documentation/CF-24-01-register.md` (ledger, G-D, Stage 2) en `docs/runbooks/cf2401-live-context.md`.

### 5.2 Uit P0 (`p0-synthesis.md` §6, `p0-quant-r2.md`)

| ID | Omschrijving | Eigenaar / doel |
|---|---|---|
| CF-SYN-01 | DB-rijen omzeilen `parse_jsonl_lines`. De P6-backfill berekent `available_at` alleen via `available_at_estimated`/`available_at_vintage` en respecteert `_ALLOWED_BASES`; P5 moet store- of loader-side een rij onder de vloer of buiten de allowlist weigeren (of testen) | P5/P6 |
| CF-SYN-02 | `lag_problems` en `ingestion_references` uitbreiden naar `PitSignals.values_at` en `select`. **In P1/P2 geland** (P1 E2 voor `values_at`, P2 `pit_select_problems` voor `select`) | Afgehandeld |
| CF-SYN-03 | (Info) Leaf-Decimal-validatie en `series_digest` gebruiken de ambient `decimal`-context; engine- en ML-aanroepers mogen de leaf niet onder een gewijzigde context draaien | P1+ (Info) |
| CF-SYN-04 | (Info) JSONL-`Decimal(str)` accepteert Unicode-cijfers, `_` en whitespace; P6 `--from-jsonl` mag naar een ASCII-grammatica beperken | P6 |
| QR-11 | Kwantiseren op 8 decimalen met `ROUND_HALF_UP` vóór `Observation`; PostgreSQL NUMERIC rondt halve af van nul; nooit `Decimal(float)` | P5/P6 en de recorder |
| QR-12 | Laad minstens `back_ms + max_age_ms` aan geschiedenis vóór de eerste beslissing (voorbeeld: FGI vanaf venstersstart over 365 daagse bars geeft 980 821 ppm, dus 7d-veld UIT); test toevoegen | P5/P6 en P7 `load_window` |
| QR-13 | `values_at` serveert een 7d-veld als `asof(d − FIELD_MAP[f].back_ms)` met de `SERIES_RULES`-leeftijd; beslissingstijden zijn gewone `int` (numpy `int64` geeft `pit_bad_value`, fail-closed) | P1 (`values_at`; niet apart geverifieerd door de opsteller) |
| QR-14 | Mengen van FGI `recorded` en `estimated_lag` verschuift de effectieve intraday-lag midden in een run van 24 u naar circa 1 u; serveer alleen `estimated_lag`-FGI in backtests, of noteer de mix in `lag_rule` en metadata | Recorder-WP en metadata |
| QR-15 | (P0 r2) De T10Y2Y-`vintage`-vloer is alleen een ondergrens: een JSONL-rij heeft geen `realtime_start`, dus een fixture kan een rij vroeger stempelen dan een echte ALFRED-vintage | P6 en CF-SYN-01 |

### 5.3 Uit P1 (`p1-synthesis.md` §6, `p1-closeout.md`)

| ID | Omschrijving | Status |
|---|---|---|
| CF-P1-01 | RISK-10: valideer tz-aware bars en de bron vóór `run_id` en engine | In P2 gebouwd (`check_backtest_source` in beide constructors, runstap 1d, `TestCfP101ValidateBeforeTheRun`) |
| CF-P1-02 | `PitSignals.select`-pin met exacte signatuur en lag-constante- en ingestieban | In P2 gebouwd (`pit_select_problems`, `TestCfP102SelectPin`) |
| CF-P1-03 | Spec-formulering (CR-05 plus E4) | **In dit register** (§3) |
| CF-P1-04 | Namen-allowlist in productiecode, verbreed volgens SEC-R2-03: `LIVE_SIGNALS`/`LiveSignals`/`BacktestSignalSource` alleen in `trading/mtf_signals.py`, `trading/strategy_engine.py` en de runner/optimizer-plaatsen; geen `type()`/metaclass-constructie met `BacktestSignalSource`; geen `.register(` erop | In P2 gebouwd (`name_reference_problems`, `TestCfP104NameReferenceAllowlist`; per naam, strakker dan de spec). Open restpunten: CF-P2-07, CF-P3-07 |
| SEC-R2-04 (P1) | Producent-notitie: de `not isinstance`-clausule is load-bearing tijdens runtime (metaclass-`__mro__`-schaduw) | Genoteerd in de P1-closeout |
| RISK-08 | Release-constraint: **P1–P3 in één release** | **Nu voldaan** (P3 is gecommit; P1–P3 liggen als een reeks vóór dezelfde uitrol). Taggen tussen P1 en P3 mocht niet |

### 5.4 Uit P2 (`p2-synthesis.md` §6, `p2-closeout.md`)

| ID | Omschrijving | Doel |
|---|---|---|
| CF-P2-01 | `reason` is de gesloten set {`no_consumed_fields`, `no_signal_source`, `no_history`, `insufficient_coverage`, `partial_fields`, null} (plus `store_unavailable` in P7), over geconsumeerde velden; nooit eligibility eruit afleiden | P3, P7 |
| CF-P2-02 | `source == "pit"` ⇔ `status ∈ {pit, partial}`; nooit "geserveerd" afleiden uit `fields[f] == "pit"` van een ongeconsumeerd veld | P3; gloss in §3 |
| CF-P2-03 | `ingest_cutoff` is in Fase A altijd null; niet valideren als ISO-string | P3, P7 |
| CF-P2-04 | Gloss van `pit` en gegeneraliseerd G8 per serie; de ensemble houdt G8 op ensemble-niveau | **Dit register** (§3) |
| CF-P2-05 | Een optimize-`mtfContextStatus` kan afwijken van een enkele re-backtest (QR-05, QR-04) | **Dit register** (§3) |
| CF-P2-06 | (QR-12) Run-niveau-AAN/UIT hangt af van beschikbaarheid over het hele venster; T-19-invariantie geldt gegeven de AAN/UIT-set; restrisico: provider-uitval die met marktstress correleert | **Dit register** (hier genoteerd) |
| CF-P2-07 | (CR-12) De naam-pin ziet geen string-concatenatie en `scripts/` valt erbuiten | P7-brief |
| CF-P2-08 | CF-P1-03 overnemen in het register | **Dit register** (§3) |
| CF-P2-09 | (optioneel, testhygiëne) `_run` in `tests/unit/test_wp24_engine.py` herschrijven naar `BacktestRunner(mtf_signals=source)` met `select_mtf_context=False` | **Open; buiten P4-scope** |
| CF-P2-10 | (optioneel, testhygiëne) dezelfde `ast.alias`-term in `pit_join_problems` plus één zelftest | **Open; buiten P4-scope** |
| CF-P2-11 | (a) QR-05: voor optimize en walk-forward start `ContextStore.load_window` op `close_0 − back_ms − max_age` met `close_0 = timeline[0] + tf`; (b) de CF-P1-04-allowlists zijn strakke gelijkheden; `runs.py` en `optimize.py` mogen alleen `PitSignals` nodig hebben, elke wijziging via een geredigeerde allowlist-diff met security; (c) optioneel: laad per consumer alleen de voedende series | P7 |
| CF-P2-12 | (QR-04 → CF-24-21) optionele strikte per-`W_c`-check en meting van fold-lokale gaten | WP2.7 |
| CF-P2-13 | (optioneel) één gedeelde `decision_times(bars, timeline, tf)`-helper in plaats van drie kopieën van de `close_i`-formule | WP2.7 |
| QR2-06 | Residu-formulering voor het register: "trades depend on the store only through the series that feed consumed fields; a consumed 7d field implies its now partner is served, and `regime_at_entry` reads the FGI now field whenever the FGI series is served" | **Dit register** (§3) |

CF-P2-09/10 zijn niet in P4 uitgevoerd: P4 is docs-only (spec §5, P4-rij).

### 5.5 Uit P3 (`p3-synthesis.md` §6, `p3-closeout.md`)

| ID | Omschrijving | Doel |
|---|---|---|
| CF-P3-01 | (SEC-04, bestaand contract) `RunResponse.config` echoot de opgeslagen JSONB letterlijk; alleen `mtfContextStatus` en `legacyMtfContext` zijn vertrouwde labels (gesloten set, nooit geëchood). De UI bindt aan die twee, niet aan `config.mtf_context.*` | UI (CF-24-19) |
| CF-P3-02 | Geaccepteerde producent-afwijkingen | Genoteerd in §3 |
| CF-P3-03 | (CR-04/SEC-07 restpunt, **met SEC-R2-05 gevouwen**) De naamregel is fail-closed voor ontbrekend, `None`, niet-str, leeg en uitsluitend whitespace. Opgevulde of ongenormaliseerde consumer-namen (`" rsi_mean_reversion"`, `"rsi_mean_reversion\n"`) worden NIET gevlagd, en SEC-R2-05 voegt toe: tekens die `strip()` overleven (onzichtbare tekens). Onbereikbaar via de API (400 bij een naam die niet naar een registersleutel normaliseert; de optimize-kolom is NOT NULL en komt uit `__name__`); bewust niet gepind | Informatief |
| CF-P3-04 | (CR-03, SEC-08) Een vijandige niet-`str`-`run_mode` waarvan `__eq__` raised zou raisen; onbereikbaar (CHECK-kolom of literal). "Faalt nooit" geldt voor JSONB-vormige waarden en gewone `str`-ORM-kolommen | Informatief |
| CF-P3-05 | (CR-05) Het consumer-set heeft twee spellingen (registersleutels en klassenamen); p3_22 houdt ze in de pas en faalt luid bij een hernoeming | Informatief |
| CF-P3-06 | (SEC-11, SEC-12) `/portfolio/summary`-eligibility is modus-blind (het WP2.5-SEC-04-punt, ongewijzigd; de mtf-teller kan niet bewegen voor paper/live omdat de metrics-helft ze eerst uitsluit). De NULL-naam-fallback naar `run_metadata["strategy_name"]` is onbereikbaar zolang de kolom `nullable=False` blijft; wordt de kolom nullable, dan moet de fallback weg | Backlog |
| CF-P3-07 | (GAP-2; verlengt CF-P2-07) De boombrede scan ziet `import trading; trading.mtf_signals.X` en gesplitste `__import__`/`importlib`/`sys.modules` in `apps/`-bestanden buiten de allowlist niet; de naam-banen vangen de live- en bronnamen wel | P7-voorwaarde (CF-P3-10) |
| CF-P3-08 | (SEC-04 → P7) P7 schrijft store-afgeleide inhoud in het blok dat `RunResponse.config` letterlijk echoot: `reason` binnen de gesloten set plus `store_unavailable`; geen uitzonderingstekst, DSN, URL, tabelnaam of SQL in enig veld; P7-test dat elk string-blad een gesloten waarde, ISO-tijdstempel, hex of null is | P7 |
| CF-P3-09 | Elke nieuwe `apps/`-importeur van `trading.mtf_signals` krijgt een `_PINNED_IMPORT_NAMES`-ingang met exacte namen (verwacht `PitSignals`); `LIVE_SIGNALS`/`LiveSignals` nooit; allowlist-diff door security | P7 |
| CF-P3-10 | (GAP-2-sluiting, **met SEC-R2-04 gevouwen**) Breid `name_reference_problems` uit voor `apps/`-bestanden: vlag een `Attribute` met naam `mtf_signals` en dynamische-importprimitieven (`importlib`, `import_module`, `__import__`, `sys.modules`) in niet-gepinde bestanden, met zelftests voor ARB-1..4. SEC-R2-04 voegt de vermomde vormen X-3..X-9 toe (`getattr`, `vars`, `exec`, `pkgutil`, `sys.__dict__`-import) | In of vóór P7 |
| CF-P3-11 | (SEC-02-continuïteit) Marker en blok blijven alleen server-geschreven na P7; breid p3_11's single-writer-pin uit naar de P7-builder; p3_08 en T-41 blijven groen | P7 |
| CF-P3-12 | (CF-P2-01/02-continuïteit) Met echte `pit`/`partial`-blokken lezen vlaggen en leaderboard nog steeds nooit `source`, `reason`, `fields` of `ingest_cutoff`; p3_32 blijft groen en onbewerkt; `pit`/`partial`/`unavailable` veranderen de eligibility niet | P7 |

### 5.6 Nog open uit WP2.5 en WP2.3 (blijven geldig)

| ID | Omschrijving | Eigenaar | Doel |
|---|---|---|---|
| WP2.3 P4-SEC-04 | Live runs met een dubbel symbool zijn na de WP2.3-deploy niet meer te resumen (fail-closed is correct). **De deploy-SQL staat in §6 en is nog open** | devops (gate) + risk-management-expert (ruling) | Gate vóór de rollout |
| WP2.3 P4-SEC-02 (**High**, bestaand) | Backtest en optimize blokkeren de event-loop van de enkele uvicorn-worker; kill switch, emergency-stop en live-taken staan stil zolang het duurt. **WP2.4 heeft hier niets aan veranderd** | risk-management-expert + python-backend-specialist | Vóór een live run naast API-backtests |
| CF-26 | Coinbase-re-baseline van WP2.1c is nog niet gedraaid (geblokkeerd op Coinbase-egress; zie het WP2.5-register) en blokkeert PR #4. PR #4 is nog niet gemerged (bevestigd door de coördinator) | quant-strategy-analyst | Vóór de merge van PR #4 |
| CF-25-xx, CF-23-xx | De open punten van het WP2.5-register (`Documentation/WP2.5-register.md` §5, o.a. CF-25-19 `/portfolio/summary` modus-blind, CF-25-20 `GET /optimize/{id}` met niet-numerieke entry-metric) en WP2.3-register blijven ongewijzigd; WP2.4 sluit er geen | — | — |

---

## 6. Deploy-notities (voor de gebruiker)

- **Release-eis.** Rol P1, P2 en P3 samen uit (RISK-08). Rolt alleen P1 uit, dan lezen nieuwe runs als legacy omdat de marker ontbreekt.
- **API-backtests draaien nu zonder MTF-context.** Na de deploy draaien API-backtests van `rsi_mean_reversion`, `dca_rsi_hybrid` en `model_strategy` (en een ensemble met die subs) zonder MTF-context: `mtfContextStatus = "unavailable"`. Niet-consumers (o.a. `momentum_breakout`) krijgen `not_used` en zijn qua trades, inclusief `regime_at_entry`, onveranderd. Context-getrainde `model_strategy`-v2-modellen zijn daardoor geen bewijs (UD-4) en moeten later opnieuw getraind worden (CF-24-03/07).
- **Resultaten komen nu overeen met de research-scripts** (C10-pariteit; API = script). Dca- en rsi-uitkomsten zijn reproduceerbaar en hangen niet meer af van het tijdstip van de laatste herstart. De getallen van consumer-strategieën kunnen sterk veranderen ten opzichte van oude runs (§4, synthetische meting: dca ×0,44..×1,17 in een bull-regime en tot ×0,16 in een vlak regime; rsi-trades verwijderd of buy-and-hold). Dat is de bedoelde correctie, geen regressie.
- **Oude consumer-runs** (backtest/optimize zonder `engine_semantics.mtf_context`) krijgen `legacyMtfContext = true`, worden uit het leaderboard gehouden en geteld in `/portfolio/summary.legacy_mtf_context_runs_excluded`. Dit is grotendeels theoretisch: elke run van vóór WP2.5 is al uitgesloten door de metrics-helft (WP2.5 UD-1). De mtf-helft raakt alleen consumer-runs die tussen de WP2.5-P3-deploy en de WP2.4-P3-deploy zijn gemaakt. Oude rijen worden nooit herschreven.
- **Runs met status `unavailable`/`partial`** blijven eligible met label, maar zijn geen WP2.8-promotiebewijs voor consumer-strategieën (alleen een rapportageregel; de promotiegate leest geen metric).
- **Live en paper zijn ongewijzigd** (byte-identiek, geen marker). **CF-24-01:** was open (live-context bevroren op de opstartwaarden); opgelost in R1 van de live-safety-WP, zie `Documentation/CF-24-01-register.md`. Tot de R1-deploy blijft dit gedrag van kracht.
- **Geen migratie in Fase A.** Alembic head blijft 020. Alleen JSONB-sleutels (`runs.config`, `optimization_runs.run_metadata`). Migratie 021 komt met P5; deploy die vóór de P7-code.
- **Versie-bumpbeleid.** Een latere wijziging aan de PIT-semantiek hoogt `MTF_CONTEXT_VERSION` op en voegt de nieuwe waarde toe aan `KNOWN_MTF_CONTEXT_VERSIONS` (CF-24-22). `timestamp_clock`, `bar_alignment` en `metrics` hebben eigen versies.
- **CF-24-01 R0 (P0, live-context-gate): deploy-gate en runbook.** Na de R0-deploy weigert de API een live create, promote en `mode=normal`-resume van een consumer-strategie (`rsi_mean_reversion`, `dca_rsi_hybrid`, `model_strategy`) met 422 `{"detail": {"code": "live_context_consumer_blocked"}}`. Een `mode=protective`-resume blijft toegestaan (UD-6 a).
  - **Deploygate RK-P0-05 (verplicht vóór de R0-deploy).** Elke live run wordt bij de deploy `orphaned` (graceful shutdown of boot-recovery); een consumer-run kan daarna alleen nog via `mode=protective` terug. Draai deze query vóór de deploy. Verwacht: 0 rijen (de enige live run is `momentum_breakout`). De query is fail-closed: hij toont elke live run die géén bekende niet-consumer is, dus ook een onbekende naam. Maak voor elke treffer vóór de deploy een plan: na de deploy een protective resume, of nu flatten en stoppen. Deploy alleen als de oude instantie volledig gestopt is (geen rolling overlap, geen tweede proces met eigen live-taken).

```sql
SELECT id, status, config->>'strategy_name' AS strategy_name FROM runs
WHERE run_mode = 'live' AND status IN ('running','orphaned','resuming')
  AND replace(lower(coalesce(config->>'strategy_name','')), '-', '_') NOT IN
      ('ma_crossover','breakout','grid_trading','sl_tp_reversion',
       'momentum_breakout','smoke_roundtrip');
```

  - **Runbook RK-P0-03: `live_context_consumer_blocked` bij een resume.** Een `mode=normal`-resume van een consumer-run geeft 422 `live_context_consumer_blocked`; de run blijft `orphaned` en onaangeroerd. De bestaande weigeringen vóór de CAS (kill switch 409, `entries_latched` 409, live-gate 403, smoke, beschikbaarheid, registry, symbolen/duplicaten, timeframe, strategy-params, bracket-config, exit-config/pyramiding, kapitaal 422, run-cap 503) gaan vóór. De 422 valt vóór de CAS en maskeert daarom een latere `kill_switch_after_start` (409); in beide gevallen is de uitkomst een weigering. Herstel: resume met `mode=protective` (entries geblokkeerd op strategie- en risk-laag; SL/TP, trailing en intrabar-exits lopen door voor zover de exit-config dat na salvage toelaat, zie RK-P0-02). Een normale resume blijft geweigerd tot de gate per strategie is opgeheven (UD-2, G-F).
  - **Runbook RK-P0-02: protective resume van een consumer.** Onder `mode=protective` krijgt een consumer-strategie nog steeds (mogelijk verouderde) context, en die bepaalt de grootte van haar eigen discretionaire SELL's: bij rsi kan een SELL tot qty 0 krimpen (CF-2401-02), dca klemt op [0,1..1]. De positie kan alleen kleiner worden. Logt de resume `resume.protective_without_exit_manager` (critical; `exitManagerMissing: true` in de resume-response), dan is er geen downside-exit (geen stop-loss en geen trailing stop, hooguit een take-profit) en hangt verliesbeperking alleen van die SELL's af: flatten dan via `POST /api/v1/runs/{run_id}/emergency-stop?flatten=true` (`flatten` staat standaard op false) in plaats van op strategie-exits te wachten. Logt de resume alleen `resume.protective_exit_config_waived`, dan is een deel van de exit-config weggevallen maar kan een stop-loss of trailing stop nog actief zijn; controleer `exitConfigWaived` en `exitManagerMissing` in de resume-response. Dit residu sluit met P4 (velden per declaratie, neutraal in Stage 1).
- **Overgenomen, nog open (WP2.3, niet door WP2.4 opgelost):**
  - **SEC-04-deploygate.** Draai deze query vóór de rollout en flatten of stop elke treffer:

```sql
SELECT id, status FROM runs
WHERE run_mode = 'live' AND status IN ('running','orphaned','resuming')
  AND jsonb_typeof(config->'symbols') = 'array'
  AND jsonb_array_length(config->'symbols')
      <> (SELECT count(DISTINCT s) FROM jsonb_array_elements_text(config->'symbols') s);
```

  - **Event-loop-waarschuwing (SEC-02).** Een synchrone backtest of optimize houdt de kill switch en de live-taken van de enkele worker stil. WP2.4 maakt dit niet erger en lost het niet op.
  - **CF-26** blokkeert PR #4 (zie §5.6).

---

## 7. Procesnotities

- **Escalatieregel (2 rejections):** niet geraakt. Er was geen enkel REJECT. P0–P3 kregen allemaal APPROVE_WITH_EDITS (P1 security en risk: APPROVE) en daarna r2-APPROVE/MATCH. Na de r2-reviews van P1, P2 en P3 hield de primary geen aparte synthesis-arbiter-ronde, omdat twee parallelle reviewers beide APPROVE (P2: critic APPROVE, quant MATCH) gaven zonder edits; in plaats daarvan schreef de primary een closeout (`p1-closeout.md`, `p2-closeout.md`, `p3-closeout.md`). Dat is een gedocumenteerde proceskeuze en geen schending van CLAUDE.md: er was niets te consolideren. Ook de P0 r2-apply door de executor volgde op twee goedkeuringen (critic APPROVE, quant MATCH) zonder arbiter-ronde.
- **Rapporten die door de primary zijn opgeslagen:** veel agents konden geen rapportbestand schrijven en gaven de tekst inline terug. De primary heeft die letterlijk opgeslagen in `reports/vp2-wp2.4/`: `dp-design`, `arch-design`, `quant-design`, `synthesis-spec`, `p0-critic-r1`, `p0-quant-r1`, `p0-quant-r2`, `p1-critic-r1`, `p1-synthesis`, `p2-critic-r1`, `p2-quant-r1`, `p2-synthesis`, `p2-quant-r2`, `p3-security-r2` en `p4-producer-report`. De primary schreef zelf `user-decisions.md`, `p1-closeout.md`, `p2-closeout.md` en `p3-closeout.md`. De overige rapporten zijn door de agents zelf geschreven of niet in deze lijst vermeld.
- **Extractie door de primary:** de reviewrapporten en patches zijn door de primary uit de agent-uitvoer gehaald. `p0-synthesis.md` noemt dat de patch "re-extracted" is en dat de leaf-sha overeenkomt met die van de quant (`4aa718d8…`); zie ook "Extracted file hashes" in `p0-quant-r2.md`. Elke v2-patch heeft dezelfde `git patch-id --stable` als de bijbehorende commit.
- **Overtredingen van agentregels:** er zijn er geen bekend in WP2.4. De agent-rapporten leggen geen `git stash`, `git stash list`, checkout, reset of clean vast (de rapporten die ze noemen, verklaren juist dat ze niet zijn gebruikt) en voor de primary-sessie is er ook geen bekend (bevestigd door de coördinator). Precedent: het WP2.5-register meldt één `git stash list` van de WP2.5-P4-producent; het WP2.2-register meldt er meer.
- **Eerdere rapportcorrecties in de keten:** P0-synthese C2 (de CR-01-fix van de critic miste de echte raise-site), C3 (de CR-02-bewering over `__context__` zou op correcte code falen) en deviation 4 van de P0-producent ("alleen later") was onjuist oostelijk van UTC (QR-08). Spec-woordfouten: zie §3.
- **Hygiëne:** elke agent werkte in een eigen wegwerp-worktree, schreef nooit `.env`, deed geen netwerkaanroepen naar providers en ruimde de eigen worktrees op (zoals de rapporten melden). De gate-worktree op `bbe7142` is verwijderd en geprund. Serena, MCP memory en context7 worden in de rapporten niet systematisch vermeld.
- **Open vragen voor de volgende WP-keten:** Fase B (P5–P7) volgt `synthesis-spec.md` §5; de CF-24-01-WP en CF-P3-10 (vóór P7) zijn de eerstvolgende blokkades. De Fase-B-gate vereist security-sign-off op P6 en P7 en een door de operator uitgevoerde backfill na CF-24-16.
