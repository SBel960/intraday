# Arborescence — validée le 2026-09-25

Package Python : `qlab`, avec un volet intraday et un volet long terme qui partagent le socle.
Chaque module exécutable expose un `main()` lancé par `python -m qlab.<module> --config ...`, construit avec `core/cli.py`. Il n'y a pas de CLI centrale qui liste les commandes, car elle obligerait à revalider un fichier déjà validé à chaque étape.

## Règles d'architecture (contre le code spaghetti)

Chaque besoin transversal a **un seul endroit** ; un nouveau module s'y branche au lieu de réécrire sa version.

| Besoin | Seul endroit | Interdit ailleurs |
|---|---|---|
| Chemins sous `$DATA_ROOT` | `core/paths.py` (`DataPaths`) | construire `root / "meta" / …` à la main |
| Accès réseau | `core/http.py` (`get` : attente du réseau, 418 / 429) | `urllib` / `urlopen` directs, boucles de relance maison |
| Commande `python -m` | `core/cli.py` (`run_command` : config, chemins, journal, codes de sortie) | `argparse` + `load_config` + `run_cli` recopiés (seule exception : `config.py`) |
| Erreurs et codes de sortie | `core/errors.py` (`QlabError`, `run_cli`) | `except Exception` hors `run_cli`, `sys.exit` dans le code métier |
| Temps | `core/timeutils.py` (entiers ms / µs) | `datetime` pour stocker ou calculer des horodatages |
| Montants, prix, quantités | `core/money.py` (`Decimal`) ; filtres de lot dans `exchange/lot.py` | flottants pour de l'argent ; *exception documentée* : fractions statistiques (distributions de coûts, rendements) en `float64` numpy dans `costs/`, `research/`, `longterm/` |
| Téléchargements de fichiers | `core/downloads.py` (`download_all`) | boucles de téléchargement parallèle recopiées par source |
| Registres append-only | `core/records.py` (`append_record`, `read_records`) | verrou / relecture / fsync recopiés par registre |
| Écriture de fichiers | `core/files.py` (`write_atomic`) | `tempfile` + `os.replace` recopiés |
| Secrets (clés d'API) | `core/secrets.py` (`Secret`, `.env` chmod 600) | valeur de clé dans un message, un journal ou le code |
| Format des frais | `exchange/fees.py` (`pair_fees`) | lire `snapshot.fees[...]` à la main |
| Journal | `core/jsonlog.py` via `Context.journal` | `print` comme seule trace d'un événement important |
| Lecture d'un YAML | `core/yamlschema.py` (`build`, `read_yaml`) | `yaml.safe_load` + validation recopiés |
| Marché (calendrier, devise, frais, levier, règles d'ordre) | fourni par le marché (config / courtier) | « 365 », « Binance », « EUR » supposés dans le code (test anti-calendrier en dur) |
| Valeurs métier | `config/*.yaml` via `core/config.py` | seuils, frais, symboles ou chemins en dur |

Ces règles sont **vérifiées automatiquement** par `tests/test_architecture.py` (en local et en CI) : un module qui les enfreint fait échouer les tests. S'y ajoutent l'absence de cycle d'imports (typage compris) et une complexité ≤ 12 par fonction (ruff C901).

Dépendances : `core` ne dépend d'aucun autre paquet de `qlab` ; `exchange` dépend de `core` ; les paquets suivants dépendent de `core` et `exchange`, jamais l'inverse. Un module qui dépasse ~300 lignes ou mélange deux responsabilités (ex. réseau + stockage) est découpé.

Colonne « Statut » : `à faire` → `livré` → `validé` (ou `à revalider`).

## Dépôt

```
intraday/
├── pyproject.toml                 dépendances, config pytest / mypy --strict / ruff
├── .github/workflows/ci.yml       CI : ruff, mypy --strict, pytest à chaque push
├── ops/                           tâches automatiques systemd utilisateur (sans sudo) ; voir ops/README.md
│   ├── qlab-exchange-info.*       minuteur horaire → exchange_info fetch --if-due (actif)
│   └── (plus tard)                signe de vie → healthchecks.io → alerte téléphone si le PC décroche ; URL de ping dans .env
├── README.md                      installation (WSL2, données hors /mnt/c), commandes, état d'avancement
├── docs/
│   ├── SPEC_INTRADAY.md           spécification d'origine (référence)
│   ├── SPEC_LONG_TERME.md         addendum long terme
│   ├── ARBORESCENCE.md            ce fichier + statut de validation par fichier
│   └── SPEC_EVENEMENTS.md         (à écrire en début de phase 9) règles des informations hors marché
├── config/
│   ├── base.yaml                  racine des données, budget disque, sources, univers tradé (trade) et observé (observe), limites de risque, paliers de capital
│   ├── intraday.yaml              horizons du gate, latences, fenêtres de features, seuils (fraîcheur, ratio 3)
│   └── longterm.yaml              lookbacks, bandes de rééquilibrage, DCA (montant, période), vol cible
├── hypotheses/
│   ├── _template.yaml             modèle de fiche d'hypothèse (mécanisme, horizon, features, coût, edge min, abandon)
│   └── lt_*.yaml                  fiches de la vague 1, écrites AVANT tout test (voir « Hypothèses par vagues »)
├── src/qlab/
│   ├── core/
│   │   ├── config.py              chargement YAML → dataclasses figées et validées ; erreur si une clé manque
│   │   ├── timeutils.py           entiers UTC ms/µs, tranche 5 min, jour de semaine, conversions sans flottant
│   │   ├── jsonlog.py             journal structuré JSONL (événements, décisions, erreurs)
│   │   ├── errors.py              QlabError (ConfigError, DataError, ExchangeError), run_cli : codes de sortie 0 / 1 / 2
│   │   ├── hashing.py             hash déterministe de fichiers et de tables Parquet (tests d'idempotence)
│   │   ├── paths.py               SEUL endroit des chemins sous $DATA_ROOT (DataPaths)
│   │   ├── http.py                SEUL accès réseau : attente sans fin du retour du réseau, 429 (Retry-After), arrêt sur 418
│   │   ├── cli.py                 mécanique commune des commandes : --config, config, chemins, journal, codes de sortie
│   │   ├── yamlschema.py          SEULE lecture YAML → dataclass stricte (config, fiches d'hypothèse), classe d'erreur au choix
│   │   ├── downloads.py           SEUL moteur de téléchargement : parallèle, vérification propre à chaque source, atomique, reprise, bilan
│   │   ├── secrets.py             secrets (.env chmod 600) : Secret jamais affiché, format strict
│   │   ├── records.py             SEUL registre append-only JSONL : verrou, contrôle avant écriture, fsync, lecture stricte (apports, essais)
│   │   ├── files.py               écriture atomique (temporaire, fsync, renommage) ; jamais d'écrasement par défaut
│   │   ├── money.py               montants en Decimal : saisie bornée, écriture décimale, nombre de la config → Decimal exact
│   │   └── ledger.py              registre append-only des apports / retraits → capital apporté (palier), flux TWR / MWR
│   ├── exchange/
│   │   ├── exchange_info.py       récupère exchangeInfo, mesure l'horloge (/api/v3/time), alerte si une paire tradée change ; commande fetch [--if-due] / show
│   │   ├── account.py             compte Binance en lecture seule : signature Ed25519 à l'heure serveur, refus de toute clé non lecture seule, frais réels
│   │   ├── fees.py                SEUL format des frais dans un snapshot : repli config + frais réels par paire (by_symbol)
│   │   ├── snapshots.py           snapshots versionnés (zstd, hash, écriture atomique), point-in-time, différences entre versions
│   │   ├── effective_params.py    recalcul à chaque appel : palier (capital apporté), frais du snapshot en vigueur, limites en devise et δ_min (valeur du portefeuille), alertes
│   │   └── lot.py                 arrondi prix → tickSize, quantité → stepSize (Decimal), rejet sous minNotional
│   ├── costs/
│   │   ├── cost_model.py          formules pures vectorisées (fractions float64) : s̃, c_taker, c_maker, coût d'un ordre, p*, drag, coût de rééquilibrage (§5, LT.4)
│   │   ├── cost_gate.py           LE GATE : distribution de c, mouvement médian par horizon, ratio, horizon min > 3
│   │   └── gate_run.py            cost gate v1 sur données réelles : journées Tardis × frais réels du snapshot, rapport Markdown dans reports/
│   ├── data/
│   │   ├── cursor.py              curseur persistant et atomique (reprise après crash, ni trou ni doublon)
│   │   ├── raw_writer.py          RAW JSONL.zst append-only, écriture par lots, rotation horaire
│   │   ├── collector.py           websockets aggTrade + bookTicker, reconnexion, horodatage de réception, contrôle de l'écart d'horloge vs serveur
│   │   ├── bronze.py              RAW → Bronze Parquet ZSTD, dédup sur (source, symbol, ts, seq), idempotent
│   │   ├── gaps.py                détection des trous (seq et temps), table des trous, jamais d'interpolation
│   │   ├── binance_vision.py      data.binance.vision : listage S3 paginé, clés des jeux de données, lecture des .CHECKSUM (aucun téléchargement)
│   │   ├── archives.py            synchronisation vers raw/binance_vision/ : bougies de TOUTES les paires spot (retirées comprises), financement USDⓈ-M, métriques des paires listées ; SHA-256, parallèle, reprise
│   │   ├── tardis.py              book_ticker historique de Tardis.dev (1er du mois gratuit) : disponibilité officielle, gzip et en-têtes vérifiés, lecture en cotations (ts_ms, bid, ask)
│   │   └── storage.py             taille disque par couche, projection de remplissage, alerte vs budget 150 Go
│   ├── features/
│   │   ├── kernels.py             boucles Numba (OFI, buckets VPIN, fenêtres glissantes)
│   │   ├── book.py                microprice, déséquilibre top of book, OFI normalisé (§2)
│   │   ├── flow.py                signe Binance, flux signé, ratio, VPIN, lambda de Kyle (§2)
│   │   ├── spreads.py             spreads effectif / réalisé, impact, Roll, Amihud (§2)
│   │   ├── volatility.py          RV, BV, sauts, Parkinson, Garman-Klass, EWMA, signature plot (§3)
│   │   └── seasonality.py         profils 5 min × jour de semaine, désaisonnalisation causale (§3)
│   ├── sampling/
│   │   ├── bars.py                barres temps / volume / dollar / tick imbalance, comparaison des rendements (§4)
│   │   └── labels.py              triple barrière coûts inclus, meta-labels (§4)
│   ├── research/
│   │   ├── hypothesis.py          fiches d'hypothèse : lecture stricte, grille de paramètres (n_trials déclaré d'avance), empreinte du contenu
│   │   ├── trials.py              registre d'essais : combinaisons déclarées seulement, fiche figée (empreinte), N distinct par volet, stats pour le DSR
│   │   ├── stats.py               moments, Sharpe annualisé, Lo 2002, Newey–West, PSR, DSR (vérifié sur l'exemple publié), MinTRL, test binomial, Sharpe minimal détectable, Benjamini-Hochberg
│   │   ├── bootstrap.py           bootstrap stationnaire (Politis–Romano) : blocs géométriques, séries tirées aux mêmes dates, graine fixée, IC par percentiles
│   │   ├── cv.py                  K plis purgés + embargo, walk-forward (expansif ou glissant), vérificateur de fuite (leaks)
│   │   ├── ic.py                  IC Spearman par horizon et décroissance ; IC transversal (actifs comparés entre eux, date par date) moyenné avec t-stat Newey–West
│   │   └── report.py              critères d'acceptation (DSR, écart de Sharpe bootstrap, sous-périodes dont baissière, actifs, MinTRL, paper) → verdict + Markdown
│   ├── sizing/
│   │   └── sizing.py              volatilité cible (réduit le risque, jamais de levier, fenêtre complète exigée), Kelly (information, plafonné à 0,25 f* et à la limite)
│   ├── backtest/
│   │   ├── events.py              fusion ordonnée des flux, latences données / ordre paramétrables
│   │   ├── fills.py               taker (slippage selon la profondeur) ; maker (traversée stricte, position en queue)
│   │   ├── engine.py              boucle événementielle, portefeuille, frais, arrondis, journal
│   │   └── leakage.py             test anti-fuite (features décalées de +1 événement)
│   ├── live/
│   │   ├── risk.py                limites, kill switch (fraîcheur, perte journalière, drawdown par palier) ; accepte dès le départ des drapeaux de risque externes (phase 9)
│   │   ├── broker.py              interface d'ordres ; live impossible sans flag explicite + clé sans retrait vérifiée
│   │   └── paper.py               paper trading sur flux live, journal complet de chaque décision
│   ├── events/                    phase 9 : informations hors marché (annonces, calendrier, actualité)
│   │   ├── calendar.py            événements programmés (Fed, inflation US…) : fenêtres sans nouvelle position
│   │   ├── news.py                collecte d'annonces (Binance, GDELT…), horodatées à la RÉCEPTION, RAW append-only
│   │   └── classify.py            classement des annonces → drapeaux de risque ; modèle et prompt versionnés
│   └── longterm/
│       ├── klines.py              lecture des archives (ms/µs ligne par ligne), contrôle qualité, bougies tronquées marquées, trous
│       ├── klines_build.py        construction de lt/klines_{1d|1h}/{paire}.parquet, reprise, erreurs isolées par paire
│       ├── universe.py            univers point-in-time : tradé (chauffe) et observé (1 paire en dollar par actif, stablecoins et tokens à levier exclus, volume médian 30 j ≥ 1 M$)
│       ├── funding.py             taux de financement USDⓈ-M → lt/futures/funding/{contrat}.parquet (heure exacte, intervalle lu par ligne), somme par jour
│       ├── market_state.py        vue globale sur l'univers observé : largeur (part des actifs au-dessus de leur moyenne L j), dispersion, part de BTC dans les volumes (corrélations : quand une fiche en aura besoin)
│       ├── signals.py             signaux des 7 fiches → poids cibles (grille journalière complète, décidés à la clôture de t, long seulement, somme ≤ 1)
│       ├── allocation.py          poids cibles → échanges (fractions de V) : buy & hold, calendaire (dates de DCA comprises), bandes, δ_min compté, cash jamais négatif, dérive des poids
│       ├── strategies.py          table fiche → signal, données (Market), rééquilibrage ; commande du gate LT officiel (rapport reports/lt_costs_*.md)
│       ├── lt_costs.py            GATE LT : rejeu des poids sans performance → turnover, drag, rejets minNotional, coûts/edge de la fiche, par palier
│       ├── lt_backtest.py         barre à barre en Decimal : décision clôture t, exécution ouverture t+1 (impact, frais réels, lot.py), δ_min réel, apports, TWR ; peek = tricheur du test anti-fuite
│       └── lt_report.py           CAGR, vol, MaxDD, Calmar, Sortino, PSR / DSR / MinTRL, vs buy & hold et DCA
└── tests/
    ├── conftest.py                uniquement la fixture de config temporaire (tmp_path) ; les données de test sont dans chaque test
    └── test_<nom>.py              un fichier de test par module source (données construites à la main, valeurs attendues écrites)
```

## Données (hors dépôt, chemin racine défini dans `config/base.yaml`)

```
$DATA_ROOT/
├── raw/{source}/{stream}/{symbol}/date=YYYY-MM-DD/HH.jsonl.zst      collecteur : append-only, jamais modifié
├── raw/tardis/binance/book_ticker/{SYMBOL}/YYYY-MM-DD.csv.gz         cotations Tardis (1er du mois), telles que reçues
├── raw/binance_vision/{spot|futures}/…/*.zip                         archives téléchargées telles quelles (SHA-256 vérifié), jamais modifiées (~3 Go)
├── bronze/{source}/{stream}/{symbol}/date=YYYY-MM-DD/*.parquet      ZSTD, dédupliqué, idempotent
├── silver/{features|bars|labels}/{symbol}/date=YYYY-MM-DD/*.parquet features dérivées uniquement
├── events/{calendar|news|flags}/date=YYYY-MM-DD/*                   phase 9 : événements horodatés à la réception
├── lt/{klines_1d|klines_1h}/{symbol}.parquet                          toutes les paires spot observées, retirées comprises (~3–5 Go)
├── lt/futures/{funding|metrics}/{symbol}.parquet                      financement, positions ouvertes (contrats USDⓈ-M)
├── meta/
│   ├── exchange_info/{source}/{YYYYMMDDTHHMMSSZ}.json.zst             snapshots versionnés (zstd : ~17 Mo → ~110 Ko), écrits seulement si le contenu change
│   ├── cursors/                                                       curseurs de reprise
│   ├── ledger.jsonl                                                   registre des apports / retraits (append-only)
│   ├── gaps.parquet                                                   table des trous
│   └── trials.jsonl                                                   registre d'essais (N)
├── logs/{component}/YYYY-MM-DD.jsonl                                 journaux JSONL append-only (jsonlog.py)
└── reports/                                                           rapports générés
```

## Ordre de livraison (un fichier source + son test par tour)

Réordonné le 2026-09-25 après l'estimation préliminaire du gate (section suivante) : le long terme passe avant la collecte et l'intraday, qui devient conditionnel.

| # | Phase | Fichier | Statut |
|---|---|---|---|
| 1 | 0 · Socle | `core/config.py` (+ `pyproject.toml`, `config/*.yaml`, `conftest.py`) | validé |
| 2 | 0 · Socle | `core/timeutils.py` | validé |
| 3 | 0 · Socle | `core/jsonlog.py` | validé |
| 4 | 0 · Socle | `core/errors.py` | validé |
| 5 | 0 · Socle | `core/hashing.py` | validé |
| 6 | 0 · Socle | `core/paths.py` | validé |
| 7 | 0 · Socle | `core/http.py` | validé |
| 8 | 0 · Socle | `core/cli.py` | validé |
| 9 | 0 · Socle | `core/files.py` | validé |
| 10 | 0 · Socle | `core/records.py` | validé |
| 11 | 0 · Socle | `core/secrets.py` | validé |
| 12 | 0 · Socle | `core/downloads.py` | validé |
| 13 | 0 · Socle | `core/yamlschema.py` | validé |
| 14 | 0 · Socle | `core/money.py` | validé |
| 15 | 0 · Socle | `core/ledger.py` | validé |
| 16 | 0 · Socle | `exchange/lot.py` | validé |
| 17 | 0 · Socle | `exchange/snapshots.py` | validé |
| 18 | 0 · Socle | `exchange/account.py` | validé |
| 19 | 0 · Socle | `exchange/fees.py` | validé |
| 20 | 0 · Socle | `exchange/exchange_info.py` (+ `ops/qlab-exchange-info.*`) | validé |
| 21 | 0 · Socle | `exchange/effective_params.py` | validé |
| 22 | 0 · Données réelles | `data/binance_vision.py` | validé |
| 23 | 0 · Données réelles | `data/archives.py` | validé |
| 24 | 1 · Gate | `costs/cost_model.py` | validé |
| 25 | 1 · Gate | `costs/cost_gate.py` | validé |
| 26 | 1 · Gate | `data/tardis.py` | validé |
| 27 | 1 · Gate | `costs/gate_run.py` | validé |
| — | 1 · Gate | **Cost gate v1 sur données réelles** : book_ticker Tardis (1er de chaque mois) des paires tradées, frais réels du compte (le §5 n'utilise pas les aggTrades) | **fait le 2026-09-26 : aucun horizon ≤ 15 min ne passe** (BTCEUR, ETHEUR, SOLEUR) |
| 28 | 2 · Recherche | `research/hypothesis.py` (+ `hypotheses/_template.yaml`) | validé |
| 29 | 2 · Recherche | `hypotheses/lt_*.yaml` : 7 fiches de la vague 1, écrites avant tout test (17 essais ; `lt_xs_momentum` révisée le 2026-09-26 avant tout test : top_k=3 retiré, dégénéré) | validé (figées) |
| 30 | 2 · Recherche | `research/trials.py` | validé |
| 31 | 2 · Recherche | `research/stats.py` | validé |
| 32 | 2 · Recherche | `research/bootstrap.py` | validé |
| 33 | 2 · Recherche | `research/cv.py` | validé |
| 34 | 2 · Recherche | `research/ic.py` (remonté de l'intraday ; version transversale) | validé |
| 35 | 2 · Recherche | `research/report.py` | validé |
| 36 | 3 · Long terme | `longterm/klines.py` + `longterm/klines_build.py` (+ `core/paths.py` : `lt_klines`) | validé |
| 37 | 3 · Long terme | `longterm/universe.py` (+ `core/config.py` : `universe.*`, `core/paths.py` : `lt_klines_dir`) | validé |
| 38 | 3 · Long terme | `longterm/market_state.py` | validé |
| 38 bis | 3 · Long terme | `longterm/funding.py` (+ `core/paths.py` : `lt_futures`) | validé |
| 39 | 3 · Long terme | `longterm/signals.py` | validé |
| 40 | 3 · Long terme | `longterm/allocation.py` | validé |
| 41 | 3 · Long terme | `longterm/lt_costs.py` (règle des rejets : non réalisable au-delà de `max_rejected_share` du volume voulu) | validé |
| 41 bis | 3 · Long terme | `longterm/strategies.py` (+ `core/config.py` : `trading_days_per_year`) | validé |
| 42 | 3 · Long terme | `sizing/sizing.py` | validé |
| 43 | 3 · Long terme | `longterm/lt_backtest.py` | validé |
| 44 | 3 · Long terme | `longterm/lt_report.py` | à faire |
| 45 | 4 · Paper LT | `live/risk.py` | à faire |
| 46 | 4 · Paper LT | `live/broker.py` | à faire |
| 47 | 4 · Paper LT | `live/paper.py` | à faire |

**Phase 9 — Événements (informations hors marché), après le paper LT.** Décidée le 2026-09-25 : le projet ne regarde aujourd'hui que des chiffres de marché ; un tweet, une annonce de la Fed ou un piratage n'entrent dans aucun calcul, et le kill switch ne voit que leurs conséquences sur les prix. Usage prévu d'abord pour le **risque** (ne pas être exposé au mauvais moment), et seulement ensuite, éventuellement, comme signal (fiche d'hypothèse, essai compté dans le DSR, cost gate). Règles à respecter :

1. Horodater à la **réception** (`received_ms`), pas à la publication : sinon le backtest sait avant d'avoir pu apprendre.
2. Un modèle de langage qui juge une annonce **passée** connaît déjà la suite : fuite d'information. On n'évalue qu'en conditions réelles (paper), ou avec un modèle dont les connaissances s'arrêtent avant la période testée. Modèle et prompt versionnés et journalisés.
3. Sources d'abord gratuites (calendriers économiques, annonces Binance, GDELT) ; l'API X est très chère.
4. `live/risk.py` accepte dès sa livraison des drapeaux de risque externes, pour que la phase 9 se branche sans modifier un fichier validé.

| # | Phase | Fichier | Statut |
|---|---|---|---|
| 48 | 9 · Événements | `docs/SPEC_EVENEMENTS.md` (spécification, à valider avant le code) | à faire |
| 49 | 9 · Événements | `events/calendar.py` | à faire |
| 50 | 9 · Événements | `events/news.py` | à faire |
| 51 | 9 · Événements | `events/classify.py` | à faire |

**Volet intraday — conditionnel.** Il ne démarre que si le cost gate v1 trouve un horizon intraday franchissable (par exemple grâce à un palier de frais plus bas ou à l'exécution maker). Sinon, il reste en attente et le rapport du gate le dit.

| # | Phase | Fichier | Statut |
|---|---|---|---|
| 52 | 5 · Collecte | `data/cursor.py` | en attente du gate |
| 53 | 5 · Collecte | `data/raw_writer.py` | en attente du gate |
| 54 | 5 · Collecte | `data/collector.py` | en attente du gate |
| 55 | 5 · Collecte | `data/bronze.py` | en attente du gate |
| 56 | 5 · Collecte | `data/gaps.py` | en attente du gate |
| 57 | 5 · Collecte | `data/storage.py` | en attente du gate |
| — | 5 · Collecte | **Cost gate v2 sur bookTicker collecté** (≥ 7 jours) | en attente du gate |
| 58 | 6 · Features | `features/kernels.py` | en attente du gate |
| 59 | 6 · Features | `features/book.py` | en attente du gate |
| 60 | 6 · Features | `features/flow.py` | en attente du gate |
| 61 | 6 · Features | `features/spreads.py` | en attente du gate |
| 62 | 6 · Features | `features/volatility.py` | en attente du gate |
| 63 | 6 · Features | `features/seasonality.py` | en attente du gate |
| 64 | 7 · Recherche ID | `sampling/bars.py` | en attente du gate |
| 65 | 7 · Recherche ID | `sampling/labels.py` | en attente du gate |
| 66 | 8 · Backtest ID | `backtest/events.py` | en attente du gate |
| 67 | 8 · Backtest ID | `backtest/fills.py` | en attente du gate |
| 68 | 8 · Backtest ID | `backtest/engine.py` | en attente du gate |
| 69 | 8 · Backtest ID | `backtest/leakage.py` | en attente du gate |

Les petits fichiers sans logique (`__init__.py`, etc.) accompagnent le fichier source suivant au lieu d'occuper un tour.

## Détection de signal (décidé le 2026-09-25)

But : maximiser les chances de détecter un signal **réel**, pas le nombre de signaux. Tester beaucoup d'indicateurs fait toujours apparaître des « signaux » dus au hasard. Quatre leviers :

1. **Plus de données indépendantes : observer large, trader étroit.** L'univers observé couvre toutes les paires spot Binance, toutes devises, retirées comprises (3 713 référencées, 1 368 en cotation, 497 actifs le 2026-09-25), plus le financement et les positions ouvertes des contrats à terme. Un signal testé transversalement sur des centaines d'actifs a bien plus de puissance que sur BTC seul (`research/ic.py`, version transversale). L'univers tradé reste étroit : BTCEUR, ETHEUR, SOLEUR (29 paires EUR disponibles), car à 50 € le minimum de 5 € et les frais limitent le nombre de lignes.
2. **Hypothèses avec une raison économique, écrites avant les tests** (`hypotheses/lt_*.yaml`) : momentum temporel, momentum transversal, faible volatilité, retour à la moyenne court terme, taux de financement des contrats à terme, filtre d'état du marché (largeur). Peu d'essais bien choisis rendent le DSR moins sévère.
3. **Puissance connue d'avance** (`research/stats.py`) : Sharpe minimal détectable vu l'historique et le nombre d'essais. Si la détection est impossible, le rapport dit « données insuffisantes », pas « pas de signal ». Contrôle des fausses découvertes (Benjamini-Hochberg) pour les familles de tests, en plus du DSR.
4. **Coûts bas** : moins de transactions, ordres limites, remise BNB, cotation directe en EUR (cost gate, `lt_costs.py`).

Ce qui ne marche pas : multiplier les indicateurs et les paramètres, optimiser jusqu'à obtenir un beau backtest. La preuve finale reste le paper trading en conditions réelles. Verdict possible et acceptable : rien ne bat le buy & hold après frais.

**Plus tard, phase à part (optionnelle)** : marchés traditionnels (Nasdaq, dollar, taux, or), corrélés à la crypto ; sources gratuites moins fiables.

## Break test (2026-09-25)

Attaque des modules validés, hors code du projet : fuzz contre des références indépendantes (200 000 dates contre `datetime`, 100 000 ordres contre les propriétés des filtres), 8 processus concurrents, 20 `kill -9` en pleine écriture, config YAML piégée (alias récursif, « billion laughs », tag d'exécution Python, doublons), dossier en lecture seule. Résiste : dates, ordres, registre (concurrence et crash), config. Quatre défauts trouvés et corrigés, chacun avec son test de non-régression :

1. `timeutils` : date hors des années 1–9999 (typiquement des µs lues comme des ms) ⇒ `OverflowError` ; désormais `DataError` explicite.
2. `lot` : valeurs extrêmes ⇒ `InvalidOperation` (contexte `Decimal` à 28 chiffres) ; calcul à 60 chiffres, sinon `DataError`.
3. `jsonlog` : écrivains concurrents ⇒ lignes vides parasites ; chaque écriture sous verrou `flock`.
4. `errors.run_cli` : erreur d'environnement (`OSError` : disque plein, droits) ⇒ code 1 « ERREUR SYSTÈME », plus code 2 « bug ».

Mineur : demi-caractère UTF-16 isolé ⇒ message clair (`jsonlog`, `ledger`).

## Audit anti-spaghetti, crash test et pen test (2026-09-26)

Mesures indépendantes (radon, pylint, vulture, graphe d'imports) : complexité moyenne A (3,1) sur 253 fonctions, maintenabilité A pour chaque fichier, aucun code mort. Crash test : ~20 000 entrées aléatoires ou piégées sur les modules récents. Pen test : clé API et clé privée absentes de tout l'historique Git, des journaux, des snapshots et du journal systemd ; droits 600 / 700 ; bombe XML et clé d'archive `../` bloquées. Huit défauts corrigés, chacun avec son test :

1. `MARKET_LOT_SIZE.maxQty`, recalculé en continu par Binance : fausses alertes et nouvelle version à chaque récupération ⇒ `change_key` sans champs volatils (l'empreinte d'intégrité reste complète : anciens snapshots valides).
2. Réponses Binance factices recopiées dans 3 tests ⇒ `tests/fakes.py` (duplication : 3 blocs → 0).
3. `archives._action` (complexité 19) ⇒ sélection / synchronisation / bilan ; `notify` recopié dans 2 commandes ⇒ `Context.notify`.
4. `effective_params.compute` (16 → 5) ⇒ `_pairs` + `_warnings` ; lecture du format des frais à la main ⇒ `fees.fallback_fees`.
5. `config._convert` (20) ⇒ `_convert` (6) + `_scalar` (10).
6. `errors` importait `jsonlog` pour le typage (cycle) ⇒ contrat `Journal` (Protocol).
7. `fees.pair_fees` : `KeyError` sur un bloc mal formé ⇒ `DataError` explicite.
8. Clé d'archive piégée : toute la synchronisation plantait ⇒ archive écartée, signalée, les autres continuent.

Garde-fous ajoutés : test anti-cycles d'imports, complexité ≤ 12 (ruff C901).

## Connexions HTTP persistantes : testées, non retenues (2026-09-26)

Idée : réutiliser la connexion (keep-alive) pour économiser une poignée de main TLS par fichier lors des ~318 000 téléchargements d'archives. Mesure sur data.binance.vision (CloudFront, fichiers jamais en cache, origine à Tokyo), mêmes fichiers, méthodes en alternance, `TCP_NODELAY` actif :

| Méthode | Médiane par requête |
|---|---|
| nouvelle connexion à chaque requête (poignée TLS comprise) | **272 ms** |
| connexion réutilisée | 687 ms |

Réutiliser la connexion **ralentit** chaque requête d'environ 400 ms sur ce serveur (reproduit 4 fois, avec `urllib` et avec `http.client`). `core/http.py` garde donc une connexion par requête ; le débit s'obtient par le nombre de téléchargements simultanés (`archives.download_workers`). À retester seulement si le serveur change.

## Multi-marchés et levier (décidé le 2026-09-26)

**Règle multi-marchés** : le propriétaire veut trader aussi des actions et d'autres instruments. Le cœur de recherche (fiches, essais, statistiques, bootstrap, validation croisée, coûts) est déjà indépendant du marché ; les prochains modules (bougies, univers, backtest, rapports) s'écrivent de même : calendrier (périodes par an, séances, jours fériés), devise, frais (en % et minimum fixe par ordre), règles d'ordre et levier sont **fournis par le marché**, jamais supposés. Garde-fou : `test_no_hard_coded_market_calendar`.

**Marchés candidats, après le long terme crypto** :
1. actions et ETF, via un courtier avec API (ex. Interactive Brokers ; PEA ou compte-titres à trancher) — à 50 €, les minimums fixes par ordre (≈ 1-2 € = 2-4 % d'un aller-retour) les rendent impraticables ; il faut quelques milliers d'euros ou un courtier sans commission avec fractions d'action ;
2. forex, indices, matières premières en CFD via Vantage (MetaTrader 5, Windows), **d'abord en compte démo** ; coûts à intégrer au cost gate : spread, commission, swap de chaque nuit.

**Axe levier (conditionnel)** : mesuré sur BTC 2020-2026 (financement réel 11,8 %/an), le levier fait **perdre** sur un actif à 60 % de volatilité (1× : 50 € → 515 € ; 2× : 167 € ; 3× : ruine en un jour ; à 2×, 20 % des positions d'un an liquidées ; Kelly ≈ 1,1×). Il ne s'active que si : stratégie validée (DSR > 0,95, confirmée en paper trading), levier ≤ ¼ du Kelly de **cette** stratégie, swaps et financement mesurés, kill switch strict, jamais de compte sans protection contre le solde négatif. Réglementation UE (ESMA) : 30:1 devises majeures, 2:1 crypto pour un particulier.

## Hypothèses par vagues (décidé le 2026-09-26)

Tester beaucoup d'hypothèses d'un coup rend le Deflated Sharpe impossible à passer (il faut battre le meilleur résultat obtenu par hasard parmi tous les essais) : on avance par vagues, la suivante seulement après le verdict de la précédente. Un seul compteur d'essais pour tout le volet long terme (`research/trials.py`) ; les essais très corrélés sont regroupés pour ne pas pénaliser à tort.

- **Vague 1 (7 fiches, 17 essais)** : momentum temporel (3), momentum transversal (2 ; k=3 retiré avant tout test : avec 3 paires tradées, c'est le panier de référence), faible volatilité (2), retour à la moyenne court terme (2), excès de levier / financement (2), filtre de largeur du marché (4), saisonnalité de fin / début de mois (2).
- **Vague 2** : rotation BTC → altcoins, cassure de volatilité, choc de volume, proximité du plus haut sur 1 an ; momentum transversal appliqué à un univers tradé plus large (paires EUR liquides).
- **Plus tard** : nouvelles cotations (trop risqué à 50 €), excès de levier par positions ouvertes (historique trop court), offre de stablecoins et événements (sources à ajouter).

**Liens entre hypothèses — rôles, pas concurrence** : ① état du marché (faut-il être investi ?) → ② sélection des actifs → ③ timing par actif → ④ taille (volatilité cible, sans levier) → ⑤ contraintes (minNotional, bandes, frais). Chaque brique est testée seule contre le buy & hold et le DCA, après frais réels ; seules les survivantes sont combinées, et la combinaison est une nouvelle fiche dont les essais s'ajoutent au compteur. Validation finale sur une période jamais touchée, puis paper trading.

## Cost gate v1 sur données réelles (2026-09-26) — VERDICT OFFICIEL

Cotations Tardis du 1er de chaque mois (BTCEUR et ETHEUR : 80 journées 2020-02 → 2026-09, 93 et 74 millions de cotations ; SOLEUR : 64 journées, 30 millions ; aucune cotation invalide), frais taker **réels** du compte 0,095 %, grille d'une seconde. Ratio = mouvement médian du mid / coût aller-retour médian :

| Paire | Coût médian | 5 s | 60 s | 5 min | 15 min |
|---|---|---|---|---|---|
| BTCEUR | 0,195 % | 0,03 | 0,16 | 0,37 | 0,64 |
| ETHEUR | 0,195 % | 0,05 | 0,22 | 0,50 | 0,86 |
| SOLEUR | 0,215 % | 0,07 | 0,30 | 0,70 | 1,21 |

**Aucun horizon ≤ 15 min ne dépasse 3, sur aucune paire ni aucune tranche horaire : aucune stratégie intraday n'est testée (§5).** Le spread est négligeable (1 tick) : ce sont les frais qui bloquent. Pour passer le seuil à 15 min, il faudrait un aller-retour d'environ 0,042 % (BTCEUR), 0,056 % (ETHEUR) ou 0,086 % (SOLEUR), soit des frais par jambe 3 à 5 fois plus bas que tes 0,095 % actuels (BTCEUR 5,2×, ETHEUR 3,7×, SOLEUR 3,1×). Le volet intraday reste en attente ; le long terme est la priorité. Rapport complet : `reports/cost_gate_v1_2026-09-26.md`.

## Estimation préliminaire du gate (2026-09-25, jetable, hors code du projet)

Bougies 1 min d'août 2026 (`data.binance.vision`) et spreads `bookTicker` échantillonnés 1/s pendant 120 s. Frais VIP 0 : 0,1 % par jambe. Coût aller-retour taker `c = 2f + s̃` ; le spread vaut 1 tick sur les six paires, donc `c ≈ 0,20 %` (0,15 % avec la remise BNB).

| Paire | Volume 24 h (devise) | 1 min | 5 min | 15 min | 1 h | 4 h | 1 j | Horizon ratio > 3 |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 1,57 Md | ×0,08 | ×0,20 | ×0,35 | ×0,72 | ×1,50 | ×3,74 | 1 j |
| BTCUSDC | 0,41 Md | ×0,09 | ×0,20 | ×0,35 | ×0,71 | ×1,51 | ×3,73 | 1 j |
| BTCEUR | 8,6 M | ×0,08 | ×0,20 | ×0,35 | ×0,73 | ×1,54 | ×3,51 | 1 j |
| ETHUSDT | 0,69 Md | ×0,11 | ×0,26 | ×0,44 | ×0,89 | ×1,89 | ×4,12 | 1 j |
| ETHUSDC | 0,31 Md | ×0,12 | ×0,26 | ×0,44 | ×0,89 | ×1,89 | ×4,14 | 1 j |
| ETHEUR | 9,0 M | ×0,11 | ×0,26 | ×0,44 | ×0,90 | ×1,91 | ×4,12 | 1 j |

Ratio = mouvement absolu médian du mid / c (taker, sans BNB). Même avec BNB, aucun horizon ≤ 4 h ne dépasse 3. Limites : un seul mois, spreads mesurés sur 2 minutes seulement, 5 s et 30 s non mesurables avec des bougies 1 min (ils seraient encore plus bas). Le cost gate v1 (`costs/cost_gate.py`) refera ce calcul proprement.

## Sources de données réelles

| Besoin | Source | Disponible |
|---|---|---|
| Bougies 1d / 1h (volet long terme) | archives `data.binance.vision` | tout de suite, historique depuis 2017 |
| Trades réels (aggTrades) | archives `data.binance.vision` | tout de suite |
| Spread réel (book_ticker spot historique) | `datasets.tardis.dev` (1er jour de chaque mois gratuit, le reste payant) | tout de suite, échantillon mensuel |
| Spread réel continu (bookTicker) | notre collecteur websocket | après ≥ 7 jours de collecte sur une machine allumée en continu |
| tickSize, stepSize, minNotional | REST `exchangeInfo` | tout de suite |

La base de données, ce sont les fichiers Parquet interrogés avec DuckDB. Il n'y a pas de serveur de base à installer.

## Dépendances externes prévues

`polars`, `duckdb`, `numba`, `numpy`, `scipy` (loi normale, Spearman), `pyyaml`, `zstandard`, `websockets`.
Pour le développement : `pytest`, `mypy`, `ruff`.
Pas de pandas, statsmodels, ClickHouse, Kafka ni Redis.
