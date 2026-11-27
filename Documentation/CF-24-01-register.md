# CF-24-01-register — live-context-WP (poller, boek, recorder, stage-hek)

Dit register sluit CF-24-01 (paper/live-context was bevroren bij het opstarten) en volgt de WP2.x-registers. Prose is Nederlands; identifiers, events, metrics en SQL blijven Engels. Basis: R0 = P0 + P1 (tag `cf2401-r0`, `ee619bd`, `1d269c9`); WP2.4-P5 (tag `wp24-p5`, `2ff2cf3`); R1 = P2 `e969dd8`, P3 `53b945c`, P4a `2c6b12d`, P4b `10298fa`, P5 `052fe30`, P6 `e882075`, P7 `aea028b` en P8 (dit document); R2 = P9 (stage-flip, apart).
**Leesregels.** De bronrapporten staan in `reports/` en staan niet op de server: elke open rij bevat daarom de actie zelf. Een status begint met `gesloten: <bron> [<ID>]` of met `open` (eventueel met een toelichting); er wordt niets afgeleid. "open (sluit bij P8-merge)" betekent: de tekst staat in dit document of in het runbook en sluit bij de merge van P8. Procedures en alle SQL behalve het R0-blok (§6) staan in `docs/runbooks/cf2401-live-context.md` (hierna runbook), met query-ID's Q-A..Q-D.

## 1. Besluiten (SY-01, synthesis-spec)

| ID | Besluit |
|---|---|
| SY-01-01 | Fail-closed (A + D): een veld voorbij zijn cap is `None`, per veld, in paper en live, plus alerts; geen BUY-blok (B alleen waar `None` zelf off-distribution is, SY-01-16) |
| SY-01-02 | Niet-consumers (momentum_breakout) blijven byte-identiek op beslispad, orders en equity |
| SY-01-03 | De poller publiceert naar het boek (de enige bron die de engine leest; naast de poller publiceert alleen de boot-seed van de recorder, P7) en is de enige runtime-schrijver van `_latest_cache`, via `adopt_validated` na `publish` (SY-01-21b; T-06: `fetch_validated` raakt geen cache; het verouderde `get_latest` schrijft hem ook maar heeft geen runtime-aanroeper, CF-2401-07); de FGI-`_history_cache` schrijft ook `get_history` (`packages/data/sentiment.py:328`, zonder `_validate_history`), voor de RetrainingService (`apps/api/services/retraining.py:317`) als die cache leeg of ouder dan 6 h is (CF-2401-06); legacy-lezers tonen de laatst gevalideerde waarde (G-01), geen staleness in `cached_value` |
| SY-01-04 | Gefaseerde uitrol: `LIVE_CONTEXT_STAGE = "neutral_shadow"` (Stage 1, >= 7 dagen), dan `"fresh"` (P9 na G-E) |
| SY-01-05 | Live-gate voor consumer-strategieën (`live_context_consumer_blocked`, 422), allowlist leeg, vanaf R0 |
| SY-01-06 | FGI in live als `estimated_lag` (dag D 00:00 + 24 h), gelijk aan backtests; `mtf_context.fgi_first_seen` als bewijs |
| SY-01-07 | Recorder voor `coingecko:btc_dominance` en `whale:net_flow_1h`; één converter voor boek en recorder; queue 256 drop-oldest |
| SY-01-08 | Cadans, vloeren, budgetten en backoff als één geteste constantenblok (`live_context_policy.py`) |
| SY-01-09 | CF-24-02: alle 11 logsites (P1) |
| SY-01-10 | Validatie, groottegrenzen, FRED `limit=5`, FGI-cache op `limit` (P2) |
| SY-01-11 | Pin-ledger PC-1..PC-12 (oude moved-block-pin geschrapt, nieuwe `_LIVE_BODY_SHA256`) |
| SY-01-12 | rsi SELL-naar-nul: carry-forward CF-2401-02 |
| SY-01-13 | Alerting: Telegram + drie Grafana-regels |
| SY-01-14 | Wandklok eerst, high-water-guard tegen klokregressie |
| SY-01-15 | Lifespan-volgorde (door P7 aangepast, runbook §9) |
| SY-01-16 | Context-getrainde ML: `context_trained()` + twee filterredenen |
| SY-01-17 | `packages/data/live_context.py` (boek, converters) |
| SY-01-18 | Observability: metrics, `/health/background` `context_poller`, gesloten vocabulaires |
| SY-01-19 | Alleen codeconstanten, geen env-variabele; SY-01-20: nieuwe code logt alleen `exc_log_fields` en gesloten velden |

Amendementen. SY-01-21a klokguard in het boek; 21b poller is de enige runtime-schrijver van `_latest_cache`, na publish; 21c in-cycle waits binnen de cycle-grens; 21d volledig pin-ledger; 21e FGI-cache zonder vormwijziging; 21f singleton-isolatie en writer-pins; 21g `context_trained()`-interface, P4a/P4b; 21h stage-hek op `LIVE_SIGNALS`; 21i stall-detectie alleen voor provider-gedateerde series, `provider_stale` op cap/4; 21j `LIVE_CONSUMERS` en `StatusListener`; 21k pure live-gate-helper; 21l shadow-event zonder leeftijden (ER2-01). 22a reproduceerbare G-B-baseline en releaseregel; 22b grenspins en `provider_bands.py`; 22c singleton-leespin; 22d bucket-gebonden polls voor dominantie en whale; 22e `expect` vervalt; 22f Stage-1-gedrag van context-getrainde paper-modellen; 22g compose-pin voor één worker; 22h `live_context_policy.py` apart.
Errata. ER2-01 (shadow zonder leeftijden), ER2-02 (consumer-count-lezers), ER2-03 (R1 niet releasebaar vóór P8 + G-D, tag `cf2401-r1` pas daarna), ER2-04 (UD-6), ER3-01..08 (regressed branch laat toekomstige `recorded`-rijen vallen, publish-regel, recorder-offer-gate, T-25, T-10/T-11, P5-write-set, `regressed_reads`-rij), r3.1 AR4-01..06 (R3-RES-2 tekst). Volledige tekst: spec-bronnen; het gedrag staat in runbook §3.

## 2. Gebruikersbesluiten

| UD | Uitkomst |
|---|---|
| UD-1 | (a) neutrale context + shadow >= 7 dagen, dan fresh (P9 na G-E) |
| UD-2 | (a) lift per strategie na Stage 2 + WP2.4-P7 PIT-backtest + quant/risk-sign-off |
| UD-3 | (a) `regime_at_entry` = `None` op nieuwe momentum_breakout-trades |
| UD-4 | (a) Telegram; gebruiker zet `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` en bevestigt een testbericht vóór Stage 2 en elke gate-lift |
| UD-5 | spec-default (a): R1 wacht op WP2.4-P5 (gemerged); niet apart gevraagd; de gebruiker kan overrulen |
| UD-6 | (a) een `mode=protective`-resume van een consumer-run blijft toegestaan |
WP2.4 UD-2 (b) (recorder): opgelost door P7 in R1, gesloten na G-D en de R1-tag. Open vragen aan de gebruiker staan in §10.

## 3. Stage-record

| Stage | Constante | Start | Einde | Evidentie |
|---|---|---|---|---|
| R0 (gate) | allowlist leeg | R0-deploy: _in te vullen bij de deploy_ | n.v.t. | RK-P0-05-query = 0 rijen (blok §6) |
| Stage 1 | `neutral_shadow` | R1-deploy: _datum + sha bij G-D2 eerste start_ | >= 7 dagen en G-E | G-D2-dagelijkse lijst (§8), shadow-divergentierapport |
| Stage 2 | `fresh` | P9 | | T-40, G-B-rerun, G-E-evidentie (P9 vult in) |

## 4. Size-waivers (alle hash-gebonden, geen precedent)

| Commit | Meting | Gebonden aan |
|---|---|---|
| P2 | bron 693 / tests 1516 (8 bestanden, +2209/−23) | `p2.patch` `0f5f1824…690ccff` |
| P3 | bron 348 (plafond 350), tests 1291 (1291) | `p3.patch` `b4a6767b…02bb` |
| P4a | tests en tooling 1570 (1575) | `p4a-r2.patch` `e09e72ab…a485` |
| P4b | bron 102 (105), tests 536 (550) | `p4b-r2.patch` `99ef3b42…6d54` |
| P5 | bron 970 (1000), tests 2307 (2320), 13 bestanden | `b9dcaa15…8bf9f` |
| P6 | Python 149 (165), yml 96 (105), tests 381 (480), `test_wp21c_alert_scope.py` +3/−0 | `p6-r2.patch` `91dbcb0f…c49d0` |
| P7 | bron 242 (244, geen waiver), tests 439 (440, CF-2401-115) | `p7-r2.patch` `eeb8fb69…d4dd2f0f` |
| P8 | totaal <= 662 ingevoegde regels over 9 bestanden (CF-2401-125, verhoogd door CF-2401-138; r2-plafond 720) | het uiteindelijke `p8.patch` |

## 5. Ledger CF-2401-01..148

Eigenaar tussen haakjes waar de bron er één geeft. "S2" = Stage-2-ingang (§9), "G-D" = §8, "G-F" = §10.

| ID | Status | Inhoud en actie |
|---|---|---|
| 01 | open | C14/Q9 (risk): exposure-check sluit de nieuwe order uit, 2%-plafond bindt nooit; context-upsizing maakt beide makkelijker te raken. Vóór elke gate-lift |
| 02 | open | rsi SELL conf → 0 laat de eigen exit vallen (quant). **G-F-voorwaarde voor rsi**: oplossen of expliciet accepteren |
| 03 | open | Providerfeiten (FGI-publicatietijd, CoinGecko `updated_at`, Whale, FRED ALFRED, egress). Procedure: runbook §11 (= CF-2401-39) |
| 04 | open | Run-detail `liveContext` + UI-badge, met CF-24-19 (nextjs + python) |
| 05 | open; pin gesloten: p5-synthesis T-33 | `--workers 1` in Dockerfile en compose gepind. Leader election vóór schalen: open |
| 06 | open | RetrainingService: activatie buiten de gate en dominantie-PRO-historie buiten het pollerbudget; zie ook 101 |
| 07 | open | `get_latest`, `value_at_offset_from_cache`, `btc_dominance_at_offset_from_cache` na één release verwijderen |
| 08 | open | `/signals` en run-diagnostiek tonen leeftijd/status uit `LIVE_BOOK.freshness` (runbook §2 punt 1) |
| 09 | open | CF-24-10: legacy `regime_at_entry` uitsluiten in `adaptive_learning` / `performance_analyzer` |
| 10 | open | WP2.3 P4-SEC-02: synchrone API-backtests blokkeren de loop, de poller stalt, context wordt neutraal. Voorwaarde voor live consumers |
| 11 | open | Quarantainelijst voor opgenomen rijen (U-SEC-05), gelezen door de WP2.4-P7-loader; zie 121 |
| 12, 15, 16, 18, 19 | open | 12 proxy-veld `SecretStr` alleen bij bewezen nood (nooit `trust_env`); 15 CoinGecko-sleutel als header; 16 Whale-sleutel van query naar header na verificatie (SEC-P2-10); 18 FGI-sightings onder eigen serie-id; 19 env-oppervlak (alleen uit/trager) |
| 13, 14, 20 | open | 13 never-served velden (CF-24-13); 14 WP2.3 `_extract_retry_after` leest `str(exc)` (WP2.6); 20 `context_trained`-classificatie naar CF-24-07 / schema v3 |
| 17 | open, vernauwd | Dode poller + achterwaartse klokstap: alleen nog zonder lezende engine; moot voor trading |
| 21 | open | Als CoinGecko `updated_at` ontbreekt zijn stalls van snapshot-series niet detecteerbaar (data-pipeline + operator). Hangt af van uitkomst 39; runbook §2 punt 3 |
| 22 | open | De orchestrator-klassecheck weigert elke ensemble; een live-kandidaat-ensemble vraagt een allowlist-regel of instantiecheck (risk + python, G-F) |
| 23, 24 | open | 23: test-only, 7 retraining-sites `.trace` geparametriseerd (`test(logging):`, na `cf2401-r0`); 24: `.trace`-regel joinen, gereduceerd tot de retraining-`fgi_history`-regel |
| 25 | open | Pre-existing (security + devops): bootwaarschuwing of weigering bij `LOG_LEVEL=DEBUG` buiten development |
| 26 | open | Eerste allowlist-lift: allowlist wordt `Mapping[str, frozenset[str]]` met per naam de goedgekeurde veldenset, ensembles pinnen hun subs (risk + python + security) |
| 27, 28 | open | 27: router en orchestrator houden hetzelfde predicaat (helper, constante, protective-uitzondering); 28: `pytest -n auto` valt om in collectie (pre-existing), gates draaien serieel |
| 29 | open | Postgres-run van `tests/migrations/test_wp18a_resume_races.py` (`MIGRATION_TEST_DATABASE_URL`). Sluit bij G-D1 stap 2c (testing-quality-specialist); het resultaat komt in het G-D-rapport en wordt bij P9 hier overgenomen |
| 30 | gesloten: p4a-synthesis §9 ("are met") | Never-served velden `None` met pin |
| 31 | gesloten: p3-risk-r1 RK-P3-06 | Plausibiliteit FGI-dagstempel en FRED-datum (hele conversie geweigerd bij een toekomstig stempel); CoinGecko `updated_at` zonder future-reject (risk tekent geen tolerantie). Vector: p3-synthesis-r2 P3S2-E01 |
| 32 | open | P5 geleverd; geen sluiting op ID. Basis: besluit p5-synthesis A5 (P5-PR-07: FRED blijft 2 × 10 s zonder marge), risk "accept as is" (p5-risk-r1) en de kostentest RK-P5-03. FEDFUNDS-poot laat een geldige T10Y2Y niet vallen. G-D-sign-off risk/security |
| 33, 34, 35, 36 | gesloten: p5-synthesis §9 (33, 34 closed per security; 35, 36 "met") | 33 poller zonder `session=` + geen legacy `get_latest` met redirects; 34 aparte labels `bad_content_type`/`bad_encoding`; 35 `egress_blocked` bij `trust_env=False`; 36 loopback-harnas incl. timeout-vector (E08) |
| 37 | open (S2) | Whale-afwijzingen `truncated`: als materieel faalt G-E voor whale (risk + quant; maatstaf: 147); remedie: cursor-paginering met budget-herplan (8 calls/h). Ratio en query: runbook §12 |
| 38 | open (S2) | Dominantie-ondergrens 0 → 5: constante, `_EXPECTED_BANDS`, SY-01-10-erratum en risk-hersign-off |
| 39 | open | Operatoractie vóór de R1-deploy (G-D2); beoordeeld door data-pipeline-engineer + security; resultaat wordt bij P9 hier vastgelegd; runbook §2 punt 3 draagt beide takken. Alle providerfeiten zijn tot dan niet-geverifieerd |
| 40, 41 | open | 40 CoinGecko-24h-velden Optional; 41 `updated_at` op `CoinGeckoSnapshot` + negatieve-lag-metric (RK-P3-06) in een Stage-1-kalibratiecommit vóór P9 (S2) |
| 42 | open, herrouteerd | Niet P8 en niet R1 (CF-2401-129): `packages/data/pyproject.toml` zegt `aiohttp>=3.9`, de image installeert `uv.lock` (3.13.3). Volgende commit die dependencies raakt; dan ook `connector._ssl`-read (3.13.x) hertoetsen |
| 43, 44 | open | 43 pin-hardening (`_TLS_NAMES`, `_BANNED_KEYWORDS`, CR-P2R2-02-waivers); 44 FRED-overflow-vector op INFO. Optioneel, test-only |
| 45, 49, 50, 59, 65 | gesloten: p5-closeout G-D-quant-sign-off | Recorder-offer-gate (`publish == 1` en niet `regressed`), meervoudige T10Y2Y-levering, één `fetch_ms` per cycle, `heartbeat` vlak vóór `publish` (sluit R3-RES-1), hele-pagina-reject |
| 46 | gesloten (P7-deel): p7-synthesis §9 (CF-2401-46) | Eén rij per `(series_id, observed_at)` aan recorder-zijde; de seed neemt alleen `available <= now` en de vroegste rij per bucket (DB-P7-01). Restant (open, 120 a; P7-PR-04 geaccepteerd voor R1): een tweede vintage (zelfde bucket, latere `available_at`, mogelijk andere waarde) na een herstart binnen het uur met mislukte of verlopen seed is een geldige PIT-rij, geen corruptie; live en backtest kunnen voor die bucket verschillen. Elke Q-A-treffer bij het stage-record (§3) noteren |
| 47 | gesloten (zonder code): p7-synthesis P7-PR-02 | Voorwaartse-stap-stempels: risk accepteert zonder code onder bindende controles Q-B (120 b), tijdsyncbewijs (121 b) en herbeslissing van de detector (121 c); die blijven open |
| 48 | open | WP2.4-P6: T10Y2Y-backfill met basis `estimated_lag` uit first-release-waarden of `vintage`; nooit `estimated_lag` uit een huidige FRED-download. G-E/G-F-evidentie voor `yield_curve_spread` gebruikt `estimated_lag`-rijen (quant + data-pipeline) |
| 51 | open | Optioneel: tel "zichtbare rij, waarde verschilt" als CF-24-16-bewijs |
| 52, 53, 54, 64 | gesloten: p5-risk-r1 RK-P5-C1, C2, C7, C10 (r2-sha: p5-risk-r2 CONFIRMED) | 52 clamp na jitter; 53 cap/4-gap tussen cyclusstarts en whale <= 50 min; 54 poller raakt de drie never-served velden niet; 64 dode-poller-zichtbaarheid op health en metric |
| 55 | open | P5 geleverd (policy-helpers in `live_context_policy.py`); geen sluiting op ID. G-D-sign-off risk/security |
| 56 | gesloten: p5-security-r1 SEC-P3-08 (CF-2401-56 "met") | Helper-invoercontracten; converter- en publish-uitzonderingen als `invalid_payload`, logs met `exc_log_fields` |
| 57, 61 | gesloten: p4a-synthesis §9 ("are met") | `LIVE_CONSUMERS.started` vóór de eerste bar; `_public`-verharding |
| 58 | open (sluit bij P8-merge) | Runbook §3 (klok) en §4, dit register §7 (residuen), CLAUDE.md-wijzigingen. Sluit bij P8-merge |
| 60 | gesloten: p5-synthesis §9 | `macro_data.py`-logsite-telling blijft 1 |
| 62 | open | Boek-helft: `freshness` roept eerst `_int(now_ms)`, `ConsumerRegistry.stopped` weigert niet-`int`. Wacht op de eerste commit die `live_context.py` opent |
| 63 | open | Pin-test "geen andere `conftest.py` met `_cf2401_live_context_isolated`" is code: test-only commit na R1, vóór P9 (CF-2401-130). Controle nu: de P5-grep, G-D1 stap 5 |
| 66 | gesloten: p4a-synthesis §9 + p5-closeout | Pariteitsvoorwaarden: wandklok-`now` in de lezer; één `fetch_ms` in de poller |
| 67 | open (residu) | Ontwerpregel P4a-P7: geen gebonden methoden van singletons als callback; een lambda in een gelist bestand dat aan een andere module wordt doorgegeven wordt door geen pin gevangen. Code-review is de controle |
| 68 | open (P9) | Gesloten stage-vocabulaire: pin `{"neutral_shadow","fresh"}` en weiger elke andere waarde, of erratum (met arch-critic) dat de SY-01-21h-guard fail-closed maakt |
| 69 | gesloten (P5-deel): p5-synthesis §9; runbook-deel open (sluit bij P8-merge) | `trading_mtf_context_regressed_reads` telt engine-reads; HELP-tekst = 95 |
| 70 | open (sluit bij P8-merge) | Runbook §6: `LOG_LEVEL` = INFO in heel Stage 1 (hoger: shadow-evidentie valt stil weg; nooit DEBUG: L-19, zie 25); stille leesfout herkennen aan shadow `none` met gezonde gauges |
| 71 | open (S2) | G-E-evidentieplan: (a) minstens één paper-rsi- en één paper-dca-run, actief over het hele venster van >= 7 dagen; (b) divergentierapport speelt boosts opnieuw af over alle shadow-bars, geeft de dienstfractie per veld en telt de hoeveelheid-wijzigende signalen; (c) `none` bij verse gauge is een mogelijke leesfout; (d) = 82; (e) = 83 |
| 72 | open | Devops-notitie: `import ccxt` draait vier read-only `git`-subprocessen en een urllib3-IPv6-probe (supply-chain/image-hygiëne) |
| 73, 74, 75 | gesloten: p4b-synthesis §9 | `context_gate_problems`-pin; `bar_ts_ms` in `engine.mtf_context_shadow` (aanwezig in `strategy_engine.py`); `_live_fields` uit `self._strategies` |
| 76 | open (sluit bij P8-merge) | `packages/trading/CLAUDE.md` en `packages/data/CLAUDE.md` beschrijven boek-lezer, stage-hek, narrowing en registratie |
| 77, 79 | open (residu) | 77 `_public` ziet geen klasse-niveau-toewijzingen binnen `if`/`try`/`for`/`with` of klasse-imports; 79 `engine_import_problems` scant `ast.alias.name` niet (defence in depth) |
| 78 | open | Optioneel: gedeelde testhelpers (`DAY`, `fresh_book`, fixture `wall`) naar een eigen helperbestand onder `tests/` |
| 80 | open, bindend | P4b is gemerged vóór elke R1-build; geen tag of deploy met P4a zonder P6; geen R1-tag vóór P8 + G-D. Controle: G-D1 stap 0 |
| 81 | open, ontwerpregel | Een later bestand dat `LIVE_CONTEXT_STAGE` of `_live_stage` noemt wijzigt de E07-lijst bewust, met security-review |
| 82, 83 | open (S2) | 82 multi-symbol shadow-events joinen op `(run_id, bar_ts_ms, volgorde in log)`; 83 `model_strategy` valt buiten G-E: het aantal `mtf_context_model_input_missing` is een preview op neutrale input, geen skill-bewijs |
| 84 | open (P9) | RD-12-label (UQ-1 a) of PAPER-blokkade onder `fresh` (UQ-1 b) voor context-getrainde paper-modellen moet in of vóór P9 bestaan, anders geen flip. Gebruikersoptie UQ-1: beide takken in §10 |
| 85 | open (P9) | `htf_trend` is een contextkolom maar geen geconsumeerd veld: P9 beslist of HTF-beschikbaarheid in de PAPER-B-voorwaarde komt |
| 86, 87 | open | 86 `context_trained` in backtest/optimize-metadata (CF-24-07); 87 nul-winst-splits (schema v3) |
| 88 | open (sluit bij P8-merge) | Runbook §7: hot-swap blijft fail-closed tot de volgende `version_id`; artefact zonder `feature_importances_` is `True` |
| 89 | open | Cross-process trained→inert-flip tussen twee symbolen van één bar; bindend vóór een multi-symbol live-modellift (trading-engine-architect + risk) |
| 90 | open | AST-pin: elke `Signal(` onder `packages/trading/strategies/` zet `strategy_id`; bij een lift ook de LIVE-only "onbekende eigenaar → geblokkeerd"-regel overwegen |
| 91 | open (sluit bij P8-merge) | `context_gate_problems` (incl. facet `fields-fail-closed`) in het pin-ledger van `packages/trading/CLAUDE.md`, met de grenzen B1..B6; optioneel: B1-verharding (toewijzingsdoel en eerste argument moeten `bar_signals` zijn) |
| 92 | open, bindend bij lift | Artefact-integriteit vóór een lift van de P0-weigering voor `model_strategy`: sha256 in sidecar geverifieerd vóór `joblib.load`, `model_path` onder `model_dir`, artefacten read-only (devops + risk + security) |
| 93 | open | Pre-existing `error=str(exc)` en `model_path` op drie `model_strategy.py`-logsites naar `error_type` (CF-36-sweep) |
| 94 | gesloten: p6-size-ruling; p7-synthesis §7 (CF-2401-94 +25 %); p8-scope-ruling §2 | Sizing vóór coderen: P6 en P8 via de arbiter; P7 binnen +25 % zonder arbiterbesluit |
| 95 | open | Optioneel: per-metric Prometheus-HELP-map in `apps/api/prometheus.py` (eigen commit) |
| 96 | gesloten: p6-closeout | Regel `mtf-context-poller-dead`: `time() - max(trading_mtf_context_poller_heartbeat_timestamp_seconds) > 2700`, `for: 5m`, `noDataState: Alerting`; waarom 45 min tegenover de 180 s `alive`-regel: runbook §5 |
| 97 | open (sluit bij P8-merge) | Runbook §4 (a)..(j), letterlijk, met de P7-aanvullingen (IC-02, IC-03) |
| 98 | gesloten: p6-synthesis §9 | (a) = 114; (b) gesloten door 97(j) |
| 99 | gesloten: p7-synthesis §9 | `mark_shutdown_requested()` vóór het poller-stopblok |
| 100 | open | `/api/v1/metrics` is zonder auth (tailnet via Caddy) en toont de nieuwe gauges; auth is een eigen item (security + devops) |
| 101 | open | `Dockerfile.api`-commentaar dat meer dan één worker de providerbudgetten vermenigvuldigt; retraining-fetches blijven buiten het budget (CF-2401-06) |
| 102 | open | Optioneel: gesloten-vocabulaire-test (`_label` → `"unknown"`), en `ContextPoller.health(book, now_ms)` gebruikt `now_ms` niet |
| 103 | open (code-helft sluit bij P8-merge; gebruikers-helft UQ-4) | `ARCHITECTURE.md` en de commentaren in `fx_cache_warmer.py` en de docstring van `signals.py` zijn bijgewerkt. **Gebruikers-helft open (UQ-4)**: 7 allow-regels in `.claude/settings.json` noemen de verwijderde warmer (regels 126, 127, 129, 130, 132, 134, 136); alleen de gebruiker wijzigt dat bestand |
| 104 | open | E11-pin (`test_cf2401_provider_fetch.py`) is logger-naam-specifiek: vervang door een AST-check (precies één `ExceptHandler`, geen logaanroep). Test-only commit vóór P9 (CF-2401-130) |
| 105, 106, 107, 108 | gesloten: p6-closeout | P6-size, drie regels in groep `trading-safety`, notifier-wiring, `mtf-context-stale` statisch `warning` met Telegram CRITICAL bij een live consumer-run |
| 109 | gesloten: p7-synthesis §9 | `trading_mtf_context_recorder_write_failures_total` (intern `mtf_context_recorder_write_failures_total`) op nul geïnitialiseerd en gecross-pind met de yml |
| 110 | open | Eigen commit (`main.py` stap 7 + `packages/trading/telegram.py` + tests; critic + security) vóór de UD-4 (a) Telegram-gate-lift in P9 (en vóór Telegram aan gaat, UQ-2). Plaatsen: `apps/api/main.py:539` `chat_id` op INFO en `:543` `exc_info=True` (op `aea028b`; `:520`/`:524` na P6), `packages/trading/telegram.py:100-104` `body[:200]`. Fix: alleen `status` en een integer `error_code` loggen, `chat_id` uit de INFO-regel, `exc_log_fields` in plaats van `exc_info` (SEC-P6-R2-01) |
| 111 | open (tekst sluit bij P8-merge; controles open) | Een serie die vanaf boot niet-fresh blijft (`unavailable`, of `stale` uit een seed ouder dan zijn cap) zonder statuswijziging stuurt geen Telegram. Controles: runbook §4 (i); per serie na elke herstart de tijd tot de eerste geslaagde poll (`ok` of `unchanged`; meting volgens 142, runbook §12); aantal en langste duur van stille niet-fresh-stukken tijdens actieve runs, en fresh-dekking over de actieve-runtijd; elke stille episode > 6 h tijdens actieve runs maakt de poller-zijdige "nooit fresh na N cycli"-melding verplicht vóór Stage 2. Besluit (c): zie 131 |
| 112 | open (S2) | Flap-criterium: tel `mtf_context.status_changed` met `new != fresh` (elke `old`) per serie per dag; > 4 op een dag: recovery-hold-down (~+12 prod/+30 test, eigen review) vóór Stage 2 |
| 113 | open (G-D2) | Eerste Grafana-start: geen provisioningfout en alle 7 `trading-safety`-regels zichtbaar, anders releaseblokker |
| 114, 115 | gesloten: p6-closeout, p7-closeout | 114 conditie opent bij elke stap naar niet-fresh (geval vii); 115 P7-testwaiver |
| 116 | open (sluit bij P8-merge) | Lifespan- en shutdownvolgorde zoals door P7 gewijzigd: runbook §9 |
| 117 | open | Publieke `MetricsCollector.ensure_counter(name, labels)` (`packages/common/metrics.py`; maakt de serie op 0 onder de lock, verlaagt nooit); de recorder stapt over en laat `_label_key`, `_lock`, `_counters` vallen. In de volgende commit die `packages/common/` opent en vóór elke refactor van de `MetricsCollector`-internals (code-critic + security). Bij de overstap: zero-init-pins opnieuw draaien en een "never lowers"-test toevoegen |
| 118, 119 | open, vóór G-D | 118: `tests/unit/test_run_recovery_scan_helpers.py:123` krijgt `cid = f"{prefix}A{uuid4().hex[:11].upper()}"`; 119: `test_cf2401_recorder_pg.py` onder `tests/integration/` (DSN-gated). Eigen commits |
| 120 | open (tekst sluit bij P8-merge; metingen open) | Q-A..Q-D en de verwachte logregels: runbook §8 en §10. Q-C leest zijn grenzen uit `data.provider_bands` op de release-sha |
| 121 | open (G-D, S2) | (b) tijdsync-bewijs in de soak (geen stap > 300 s); na een stap komen rijen met `available_at` in [stap − 1 h, correctie + 1 h] in quarantaine voordat een backtest ze leest (lijst nog niet gebouwd, CF-2401-11: stap, grootte, correctietijd en dat interval in het stage-record §3 noteren; rijen in `context_observations` nooit wissen of wijzigen); (c) herbeslissing over de wall-versus-monotonic-detector vóór de loader recorded rijen als backtestinput gebruikt of vóór een gate-lift |
| 122 | open | Geen retry in de recorder (DB-P7-03): een verloren rij wordt nooit opnieuw aangeboden (het boek houdt de bucket), dus het gat blijft. Herbeslissing met de Q-D-dekkingsdata vóór recorded rijen een backtestinput worden (zelfde trigger als 121 c). Kandidaat: één retry in de writer (~6 regels; veilig, want de transactie is atomair en idempotent). Eigenaar: risk + database-architect |
| 123 | gesloten: cf2401-123-testing-report, cf2401-123b-testing-report | Shutdownbudget: compose-commit `7b88606` (api-`stop_grace_period` volgens de formule in het commentaar van `infra/docker-compose.yml`) en deploy-runbook-commit `38b785a` (136). Formule en operatorregels: runbook §9 (138); nooit de grace alleen verkorten |
| 124 | open (S2) | `context_recorder.seeded` krijgt `skipped` (rijen niet gepubliceerd door band of vorm, zonder de duplicaten per bucket) + key-set-pin (`event`, `log_level`, `rows`, `reason`, `skipped`); tot dan is Q-C de controle |
| 125 | gesloten: p8-scope-ruling §11 | P8-plafonds (§4); verhoogd door 138 |
| 126, 127, 128 | open | 126 9-paden-write-set en T-P8-gates; 127 G-D = G-D1 + sign-offs + G-D2 en de volgorde (nu 137); 128 scratch-e2e-harnas E1-E8 met drie zelftest-mutanten, daarna optioneel een committed integratietest vóór P9 |
| 129, 130 | open | 129 = 42; 130 = 63 en 104 |
| 131 | open (P9) | Besluit over 111(c). Risk's maatstaf: het stuck-since-boot-geval moet pagen vóór live context een beslissing beïnvloedt. Standaard geldt die, tenzij Stage 1 geen never-fresh-from-boot-episode toont en risk dat accepteert (P9-synthese: risk + quant). De P6-closeout noemt "verplicht als een stille episode > 6 h optreedt" |
| 132 | open | WP2.4-register regel 3 ("Fase B ... nog open") is verouderd sinds WP2.4-P5; eigenaar WP2.4-keten, niet door P8 gewijzigd |
| 133 | gesloten: cf2401-123-synthesis §1-§2 | Devops-brief bij 123: de teardown-term (`asyncio.wait` in de lifespan, `main.py`) is genoemd, de grace gekozen en de formule geldt (risk bevestigd) |
| 134 | open (sluit bij P8-merge) | Zelfstandig-regel voor register en runbook (rapporten staan niet op de server) |
| 135 | open, optioneel | Een "Release N"-sectie voor R1 in `docs/runbooks/deploy-fase1-minimumset.md` wanneer de gebruiker de deploy voorbereidt (devops) |
| 136 | gesloten: cf2401-123b-testing-report | Docs-commit 123b (`38b785a`): de zes verouderde grace-plaatsen in `docs/runbooks/deploy-fase1-minimumset.md`, inclusief de L-17-controle, volgen de api-`stop_grace_period` |
| 137 | open (tot G-D1) | Volgorde (wijzigt 127): 118, 119, 123 en 123b in willekeurige volgorde, P8 als laatste, dan G-D1 → sign-offs → tag; lineaire geschiedenis, elke patch met `git apply --check` op de dan geldende HEAD, eigen testpass per commit. Raakt de definitieve 118 of 119 `infra/`, het deploy-runbook of een P8-pad: terug naar de arbiter. G-D1 stap 0 en 5: §8 |
| 138 | open (sluit bij P8-merge) | Runbook §9: shutdownformule en operatorregels (kill switch vóór de stop, SIGKILL-gevolgen, afgebroken teardown, onbegrensde staart, host-stoplimieten); geen grace- of drain-getal in P8; P8-plafonds verhoogd (§4) |
| 139 | open | Pin-test die de grace koppelt aan de drain-, flush-, bg-cancel-, registry- en teardown-constanten (faalt luid als de extractie niet meer matcht, één mutant, registry-term latent). Test-only slot na R1, vóór P9, met 130; houdt de tag niet op (testing-quality-specialist + code-critic) |
| 140 | open (residu, optioneel) | In-app-tijdslimieten op de lifespan-staart (`dispose_engine`, client-closes) en een harde limiet op de flush (G-4). Niet R1; de marge in de formule dekt het tot dan (python-backend-specialist) |
| 141 | open (residu) | Host-stoplimieten (Docker-daemon `shutdown-timeout`, systemd `TimeoutStopSec`) en een tweede Ctrl-C op een voorgrond-compose: buiten de repo, niet geverifieerd. Operatorregel: runbook §9. Herzien bij 135 (devops + gebruiker) |
| 142 | open (sluit bij P8-merge) | Erratum 111(b) (p8-synthesis; RK-P8-01, risk I-06, QR-P8-05): de meting is de tijd van herstart tot de eerste geslaagde poll (`ok` of `unchanged`, `_SUCCESS`), niet alleen `ok`; bij aanwezigheid via `/health/background` (runbook §12 eerste start), onbeheerd via Prometheus (runbook §12); de age-gauge-variant vervalt (RK-P7-05). Geldt ook voor G-D1 E8 (risk + quant) |
| 143 | open (sluit bij P8-merge) | RK-P8-09 (wijzigt de G-D2-tekst van 127): de RK-P0-05-query draait ook vóór de R1-deploy als R0 al is uitgerold; de herstart orphant elke live run opnieuw en een treffer vraagt hetzelfde plan (gebruiker/operator) |
| 144 | open (P9) | QR-P8-10: G-E toont beschikbaarheid, geen pariteit. De P9-ingangssynthese (quant + risk) beslist over een shadow-versus-PIT-replay: shadow-`values` tegen een PIT-lezing van dezelfde `decision_at` (dominantie en whale uit de store, FGI uit providerhistorie onder `estimated_lag`, T10Y2Y uitgesteld naar WP2.4-P6); elk verschil verklaard door QR-P5-07, een Q-A-vintage, het RK-P7-R2-02-uurgat, 121-quarantaine of R3-RES-2 |
| 145 | open (P9) | QR-P8-08: bij G-E is de shadow de enige T10Y2Y-evidentie (`context_observations` heeft geen T10Y2Y-rijen, er is geen backfill-CLI); de P9-synthese (quant) neemt dat mee in de G-E-beoordeling, zie 48 en §9 |
| 146 | open (P9, G-F) | QR-P8-11: Stage-1-paper-resultaten van rsi/dca komen uit neutrale context en zijn geen G-F- of UD-2-bewijs; runs die de P9-herstart overspannen worden gesplitst op het `stage`-veld van `engine.live_context` of weggelaten (quant + risk, bij P9 en elke G-F) |
| 147 | open (P9) | Maatstaf "materieel" voor 37 (quant-advies QR-P8-09, niet bindend): > 1 % van de whale-pogingen over het venster of > 5 % op een UTC-dag; de P9-synthese beslist (risk + quant) |
| 148 | open, optioneel | DB-P8-06 / security I-04: read-only-sessie voor de runbookqueries (`default_transaction_read_only`, `statement_timeout`) als verdediging in de diepte; bij de volgende wijziging van runbook §10 of bij 135 (database-architect + security) |

CF-24-01 zelf en CF-24-05, -08, -09 (R1) en CF-24-02 (P1, R0) zijn opgelost en worden gesloten na G-D en de R1-tag (statusregels in `WP2.4-register.md` §5.1).

## 6. R0-deploygate en runbook-punten 5 tot 7 (byte-gelijk, `p0-synthesis-r2.md` §4, sha256 `ef24c421…d027`)

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

## 7. Overige residuen en reviewer-ID's (niet onder een CF)

- **SEC-P2R2-02**: dubbele `Content-Encoding`-headers; gesloten door ontwerp (`auto_decompress=False`, runtime-gepind door MX5). Verwijdering van die kwarg heropent het.
- **P2-afwijkingen** (p2-closeout §6): `MacroSnapshot.yield_curve_observed_on: date | None = None` is toegevoegd (de P3-converter heeft de T10Y2Y-observatiedatum nodig; niets `model_dump`t het); de transportpin heeft één extra regel voor elke `Name` `ssl` (vangt `getattr(ssl, 'CERT_' + 'NONE')`); de 11 `provider_bands.Reason`-literals staan herhaald in `provider_http` (pin 1 verbiedt de import) en de exacte-gelijkheidstest houdt ze gelijk.
- **P3**: SEC-P3-11 R2 (leaf-re-exports en reflectieve builtins L1-L7), SEC-P3-12 N9 (`exec`; alleen ruff `S102` vangt het, er is geen CI in de repo), N10 en K5 (runtime-gebouwde namen), K4 (alleen via de `bare-module`-regel), CR-P3R2-07 (`input()`, `breakpoint()`, `from typing import *`, `getcontext()`), E19 (conftest-uniciteit, zie 63).
- **AR4-08**: `view` schrijft `_series`; "één event loop, geen locks" is dragend. Een LIVE/PAPER-engine buiten de loop vraagt een lock of moet geweigerd worden.
- **Klok**: R3-RES-2 (Low), RK-P3-R2-05 en QR-P3-R2-06: zie runbook §3.
- **P4a**: SEC-P4A-R2-01 (drie versluierde vormen in `apps/` omzeilen de stage-pin), CR-R2-02 (`stage_source_problems` ziet geen keyword-, functie- of aliasnamen; relevant voor 68), CR-R2-01 (formatschuld in `test_wp24_ast_pins.py`), QR-P4A-R2-01 (per-veld-powercontroles).
- **P4b**: SEC-P4B-R2-01 (blinde vlek bij overschreven bound-key); CR-P4B-r2 Info (twee logsites delen één PC-8-anker, gate-docstring niet bijgewerkt).
- **P5**: CR-P5-R2-02 (`exc_text_problems` naar een gedeelde helper), RK-P5-R2-01/02 (equivalente mutanten).
- **P7**: RK-P7-R2-02 (een overgeslagen out-of-band-rij markeert het uur als bezet: een uur-gat), SEC-P7-R2-02 (optionele test bij de volgende recorder-wijziging: `_missing_table` is False bij `InsufficientPrivilegeError` en `UndefinedColumnError`), CR-P7-R2-01/02 (een annulering-slikkende store laat `flush` hangen; `_task` blijft gezet na een timed-out flush), SEC-P7-07 (de INFO-reden is `table_missing`; de spec zegt `no_table`; de docs volgen de code).

## 8. G-D: gate voor de tag `cf2401-r1`

G-D heeft twee delen (CF-2401-127). Volgorde (137): 118, 119, 123 en 123b in willekeurige volgorde, P8 als laatste, dan G-D1 → sign-offs → tag → G-D2.

**G-D1 (testing-quality-specialist, wegwerp-worktree op de release-kandidaat, serieel):**
- [ ] 0 Voorwaarden: 118, 119, 123 en 123b zijn voorouders van de release-kandidaat; boom schoon; `git tag -l` = `cf2401-r0`, `wp24-p5`; `git tag --contains <P2-sha>` leeg; `alembic heads` één kop `021`; P4b gemerged vóór de build (80).
- [ ] 1 Suites: unit alles groen; `test_run_recovery_scan_helpers.py` 20 keer achter elkaar zonder falen (bewijst 118); integratie alleen de bekende falers met ongewijzigde faalwijze; mypy strict en ruff 0 nieuw; G-A (digests, tweeproces-PIT-determinisme, lege diff-lijst tegen `bd93c27`); G-B seeds 1 en 4242 = 6/6; G-C-soaktests.
- [ ] 2 Postgres op de scratch-cluster (scratch-rol, wachtwoord alleen in een shellvariabele, geen `pg_hba`-wijziging, trap die opruimt): (a) `test_cf2401_recorder_pg.py` (`CONTEXT_STORE_TEST_DATABASE_URL`), (b) `test_wp24_context_store_pg.py`, (c) `test_wp18a_resume_races.py` (`MIGRATION_TEST_DATABASE_URL`; sluit CF-2401-29), (d) runbook Q-A, Q-B, Q-C letterlijk na de recorder-rondreis (voor Q-C is `$DEPLOY_DIR` de RC-worktree): parsen en 0 rijen, (e) opruiming geverifieerd.
- [ ] 3 Scratch-e2e paper consumer-run (rsi, dca, context-getraind model, momentum als controle; boot → fresh → stale → recovered): E1 consumers krijgen `SignalValues()` in elke fase; E2 één `engine.mtf_context_shadow` per consumer-bar met `bar_ts_ms`, niet-consumer geen; E3 het getrainde model koopt nooit; E4 geen Telegram voor de bootovergang, precies één per serie voor `fresh → stale` en één voor `stale → fresh` terwijl een consumer-run actief is, vaste teksten; E5 health-statussen kloppen; E6 gauges bewegen; E7 recorder schrijft één rij per bucket, alleen dominantie en whale, seed herstelt zonder rij-toename; E8 tijd tot de eerste geslaagde poll (`ok` of `unchanged`) per serie (142). Zelftest: drie mutanten (`LIVE_CONTEXT_STAGE = "fresh"`, listener alert op de eerste overgang, logaanroep met uitzonderingstekst) laten het harnas falen.
- [ ] 4 Log-hygiënesweep over stap 3 op INFO en DEBUG met sentinel-sleutel, URL en uitzonderingstekst: 0 treffers voor de sentinel, `http://`, `https://`, `api_key`, `Traceback` en de uitzonderingstekst; bestaande hygiënetests (T-03, T-30, T-34, recorder).
- [ ] 5 Statisch: yml parseert (groep `trading-safety`, 7 regels, unieke uid's); de eerste `stop_grace_period` in `infra/docker-compose.yml` is die van `api` en voldoet met de echte waarden aan de formule van 123 (runbook §9); de T-123b-greps op het deploy-runbook zijn schoon (137); `--workers 1`; `grep -rl --include=conftest.py _cf2401_live_context_isolated .` noemt alleen `tests/conftest.py`; geen `history_cache_warmer`-import in `apps`, `packages`, `infra`, `scripts`.
- [ ] 6 Rapport met alle resultaten en sha's.

**Sign-offs (parallel, alleen lezen):** risk + security voor P0, P2-P7, elk met hun CONFIRMED-closeoutrecord per fase, en zij bevestigen het G-D1-harnas en de sweep; quant voor P3, P4a, P4b en voor P5 de tekst hieronder; database-architect voor P7 en Q-A..Q-D. Daarna het besluit van de synthesis-arbiter; dan maakt de code-executor de lokale tag `cf2401-r1` (nooit gepusht).
> "G-D quant sign-off, P5, for the ER3-07 scope only, at `b9dcaa15e81205e9f669f50ad8a03fbbcb4afec59e4b1fb85a81dfcd93c8bf9f`. The recorder is offered a row exactly per ER3-03: only for dominance and whale, and only when `publish` returned 1 and `freshness(fetch_ms).regressed` is False, evaluated after the call. FGI and T10Y2Y are never offered. FRED delivers every released, in-band T10Y2Y row with its observation date. `yield_curve_spread` stays the newest row. One bad or future-dated row rejects the whole page (fail-closed, look-behind only). One integer `fetch_ms` per cycle is shared by the fetch, the converter, the pre-publish heartbeat, `publish`, the freshness gate and the offered row. The r2 bucketed-wait cap (3900 s) changes scheduling only under a backward wall step and never changes a stamp or the offer gate. G-B 6/6 at seeds 1 and 4242. Residuals QR-P5-05/06/07 are recorded in the P8 register (T-39, CF-2401-97)."

**G-D2 (operator = de gebruiker, op de server; runbook §12; "Lokaal tot PR #4 gemerged" blijft gelden):**
- Vóór de deploy (falen = niet deployen): RK-P0-05-query (blok §6) 0 rijen, ook als R0 al is uitgerold (een treffer, bv. een protective consumer-run, vraagt hetzelfde plan; 143); SEC-04-query (`WP2.4-register.md` §6) 0 rijen; CF-2401-39-providercontrole (runbook §11); migratie 021 met de nieuwe image na de api-stop (deploy §4.3-patroon); `LOG_LEVEL` = INFO (niet DEBUG, L-19); tijdsync actief.
- Eerste start (falen = rollback volgens het deploy-runbook): 113 (7 regels, geen provisioningfout); `/health/background` `alive: true` en elke serie bereikt een geslaagde poll (`ok` of `unchanged`; tijd per serie noteren, ook na elke herstart; een tijdelijke uitkomst opnieuw beoordelen na de volgende cyclus van die serie; 142); een `context_recorder.seeded`-regel met `reason` null (geen `store_unavailable`, geen `context_recorder.seed_timeout`) en, na de eerste `ok`-poll van dominantie of whale, geen `context_recorder.disabled`.
- Dagelijks in Stage 1 (voedt G-E; falen herstart de Stage 1-klok of blokkeert G-E; noteren in het stage-record): Q-A..Q-C 0 rijen (een Q-A-treffer noteren, zie 46), Q-D >= 23 per serie en volledige UTC-dag, `trading_mtf_context_record_dropped_total` vlak buiten een shutdown-flush; flap-telling (112); aantal en langste duur van stille niet-fresh-stukken (111); geen tijdstap > 300 s, anders quarantaine (121); whale-`truncated`-ratio (37); `/health/background` na elke herstart; dagelijkse export van de G-E-logregels buiten de container, en vóór elke recreate (runbook §6).
- Alleen gebruiker/operator: RK-P0-05- en SEC-04-SQL, de CF-2401-39-run met echte egress, migratie 021 in productie, de deploy zelf, de Grafana-start, de tijdsync-journal, Telegram-setup (nodig vóór Stage 2 en elke lift, niet voor R1; UQ-2).

## 9. Stage 2-ingang (G-E, vóór P9)

Alle punten gelden per geconfigureerde serie; poll-succes wordt per poging gemeten (SY-01-22d):
- [ ] >= 99 % poll-succes (`ok` of `unchanged`, query in runbook §12) over 7 opeenvolgende dagen; geen poller-dead-alert en elke stale-episode verklaard; >= 7 d + 6 h opgenomen dominantie (en whale, indien geconfigureerd) volgens Q-D.
- [ ] Gebruiker bevestigt een Telegram-testbericht (UD-4); CF-2401-110 is gemerged en gedeployed vóór Telegram aan gaat of vóór de P9-lift (UQ-2).
- [ ] Quant + risk beoordelen het shadow-divergentierapport volgens 71 (incl. 82 en 83); `model_strategy` valt erbuiten.
- [ ] 37 whale-`truncated`-ratio niet materieel (maatstaf: 147); 38 dominantie-ondergrens beslist; 41 kalibratiecommit gedaan of bewust uitgesteld; 48: bij G-E is de shadow de enige T10Y2Y-evidentie (live = eerste release, `estimated_lag`); `context_observations` heeft nog geen T10Y2Y-rijen; een backtest als G-E/G-F-evidentie voor `yield_curve_spread` leest alleen `estimated_lag`-rijen uit first-release-waarden (97 j; 145).
- [ ] 68 stage-vocabulaire gepind; 84 RD-12-label gebouwd en door de P9-synthese geverifieerd (UQ-1 a), of de PAPER-blokkade onder `fresh` gebouwd en gepind (UQ-1 b); 85 `htf_trend` beslist.
- [ ] 111 per serie na elke herstart gemeten (142), 131 en 144 beslist; 112 flap-telling <= 4 per UTC-dag, anders eerst de hold-down; 121 (b) tijdsyncbewijs en (c) detectorbeslissing; 122 retry-beslissing met de Q-D-data; 124 `skipped` in de seed-logregel.
- [ ] Q-B en Q-C leeg over de hele Stage 1; Q-A leeg behalve genoteerde second-vintage-buckets (herstart binnen het uur met mislukte of verlopen seed, P7-PR-04), elk met serie, `observed_at` en de herstart; Q-D >= 23 van 24 rijen per geconfigureerde recorded serie per volledige UTC-dag (een ontbrekende serie of dag telt als 0) en `increase(trading_mtf_context_record_dropped_total)` = 0 buiten een shutdown-flush (120 d); 29 en 39 hier ingevuld met het resultaat uit het G-D-rapport en de operatorrun.
- [ ] P9 draait T-40 en de G-B-rerun tegen de gecommitte digests en voegt de G-E-evidentie in §3 toe.
Een lift van de live-gate per strategie (G-F) staat los hiervan, zie §10.

## 10. G-F-lijst, incidenten en gebruikersitems

- **G-F (live-gate-lift per strategie):** Stage 2 live; UD-2 (a) PIT-backtest van die strategie; voor rsi is 02 opgelost; quant + risk + security; een allowlist-commit volgens 26 (Mapping met veldenset) door de volledige keten, met 22, 27, 89, 90 en 92 meegewogen; Stage-1-paper-resultaten (neutrale context) en runs die de P9-flip overspannen zijn geen G-F-bewijs (146).
- **Incidenten (proces, geen productgevolg):** INC-P2-01 stray stash-entry `bd2d743`: gesloten, de primary onderzocht en verwijderde hem op uitdrukkelijke eenmalige toestemming van de gebruiker vóór P5 (`git stash list` daarna leeg); INC-P2-02..04 en P3 (alleen-lezen `git stash list`, een door de critic ongewild uitgevoerde `patch` in de hoofdboom, hersteld en geverifieerd schoon); P4a/P4b (alleen-lezen stash-lijst, `git reset` in een wegwerp-worktree, een `apply --check` zonder argument); P5 (alleen-lezen `git apply --numstat`, daarna deny-rule; usage-limit-hervatting); P6 (overschreven scratch-script, usage-limit); P7 (geblokkeerde compound-commando's). Geen enkele raakte een goedgekeurd artefact.
- **UQ-1 (CF-2401-84):** onder `fresh` kopen context-getrainde paper-modellen zodra alle geconsumeerde velden gediend worden, op CF-24-03-scheve input. (a) spec-pad (RD-12): ze mogen kopen; in of vóór P9 komt het label (run-diagnostiek of registermarkering: context-getraind, train/serve-scheef, geen bewijs, met CF-24-03/07) en de P9-synthese verifieert het vóór de flip; "B alleen" volstaat niet. (b) aanbevolen, fail-closed: PAPER blijft geblokkeerd zoals LIVE tot CF-24-03 is opgelost; dit wijzigt SY-01-16 binnen UD-4 (a) en vraagt in P9 een codewijziging in `_drop_context_model_buys` met pin (`context_gate_problems`) en review, geen label. In beide takken zijn hun paper-resultaten geen bewijs. Alleen de gebruiker beslist; blokkeert alleen de P9-brief.
- **UQ-2 (CF-2401-110, Telegram in Stage 1):** (a) 110 in R1 vóór G-D; (b) aparte 110-commit en deploy direct na R1, daarna zet de gebruiker de `TELEGRAM_*`-variabelen (aanbevolen); (c) geen Telegram tot Stage 2. Zonder Telegram vertrouwt Stage 1 op de dagelijkse controles, want Grafana meldt aan niemand (UD-4 a). Blokkeert niets nu.
- **UQ-4 (CF-2401-103):** de 7 allow-regels in `.claude/settings.json` (zie 103) verwijderen wanneer het uitkomt. **UQ-5:** taal (register Nederlands, runbook Engels) volgt de conventie; de gebruiker kan overrulen. UQ-3 is beantwoord (INC-P2-01 hierboven).

## 11. Verwijzing

Gedrag, drempels, metrics, logevents en de vier queries: `docs/runbooks/cf2401-live-context.md`. Tests: `tests/unit/test_cf2401_*.py`, `tests/integration/test_cf2401_momentum_identity.py` en `scripts/cf2401_gb_capture.py` (G-B). WP2.4-context: `Documentation/WP2.4-register.md`.
