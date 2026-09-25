# Arborescence — validée le 2026-09-25

Package Python : `qlab`, avec un volet intraday et un volet long terme qui partagent le socle.
Chaque module exécutable expose un `main()` lancé par `python -m qlab.<module> --config ...`. Il n'y a pas de CLI centrale, car elle obligerait à revalider un fichier déjà validé à chaque étape.

Colonne « Statut » : `à faire` → `livré` → `validé` (ou `à revalider`).

## Dépôt

```
intraday/
├── pyproject.toml                 dépendances, config pytest / mypy --strict / ruff
├── .github/workflows/ci.yml       CI : ruff, mypy --strict, pytest à chaque push
├── README.md                      installation (WSL2, données hors /mnt/c), commandes, état d'avancement
├── docs/
│   ├── SPEC_INTRADAY.md           spécification d'origine (référence)
│   ├── SPEC_LONG_TERME.md         addendum long terme
│   └── ARBORESCENCE.md            ce fichier + statut de validation par fichier
├── config/
│   ├── base.yaml                  racine des données, budget disque, sources, symboles, limites de risque, paliers de capital
│   ├── intraday.yaml              horizons du gate, latences, fenêtres de features, seuils (fraîcheur, ratio 3)
│   └── longterm.yaml              lookbacks, bandes de rééquilibrage, DCA (montant, période), vol cible
├── hypotheses/
│   └── _template.yaml             modèle de fiche d'hypothèse (mécanisme, horizon, features, coût, edge min, abandon)
├── src/qlab/
│   ├── core/
│   │   ├── config.py              chargement YAML → dataclasses figées et validées ; erreur si une clé manque
│   │   ├── timeutils.py           entiers UTC ms/µs, tranche 5 min, jour de semaine, conversions sans flottant
│   │   ├── jsonlog.py             journal structuré JSONL (événements, décisions, erreurs)
│   │   ├── errors.py              QlabError (ConfigError, DataError, ExchangeError), run_cli : codes de sortie 0 / 1 / 2
│   │   ├── hashing.py             hash déterministe de fichiers et de tables Parquet (tests d'idempotence)
│   │   └── ledger.py              registre append-only des apports / retraits → capital apporté (palier), flux TWR / MWR
│   ├── exchange/
│   │   ├── exchange_info.py       récupère exchangeInfo (Binance) et les frais réels du compte, stocke des snapshots versionnés
│   │   ├── effective_params.py    recalcul à chaque appel : palier, frais effectifs (snapshot ou repli), limites en devise, δ_min
│   │   └── lot.py                 arrondi prix → tickSize, quantité → stepSize (Decimal), rejet sous minNotional
│   ├── costs/
│   │   ├── cost_model.py          c_taker, c_maker, p*, drag annuel (§5)
│   │   └── cost_gate.py           LE GATE : distribution de c, mouvement médian par horizon, ratio, horizon min > 3
│   ├── data/
│   │   ├── cursor.py              curseur persistant et atomique (reprise après crash, ni trou ni doublon)
│   │   ├── raw_writer.py          RAW JSONL.zst append-only, écriture par lots, rotation horaire
│   │   ├── collector.py           websockets aggTrade + bookTicker, reconnexion, horodatage de réception, contrôle de l'écart d'horloge vs serveur
│   │   ├── bronze.py              RAW → Bronze Parquet ZSTD, dédup sur (source, symbol, ts, seq), idempotent
│   │   ├── gaps.py                détection des trous (seq et temps), table des trous, jamais d'interpolation
│   │   ├── archives.py            téléchargement data.binance.vision (aggTrades, klines) + Tardis.dev (book_ticker), checksum
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
│   │   ├── hypothesis.py          schéma, validation et hash des fiches d'hypothèse YAML
│   │   ├── trials.py              registre d'essais append-only, compteur N par volet, persisté
│   │   ├── stats.py               Sharpe annualisé, Lo 2002, Newey–West, PSR, DSR, MinTRL, test binomial
│   │   ├── bootstrap.py           stationary bootstrap (Politis–Romano), graine fixée
│   │   ├── cv.py                  purged K-fold + embargo, walk-forward (fenêtre glissante ou expansive)
│   │   ├── ic.py                  IC Spearman par horizon, décroissance, comparaison à l'horizon du gate
│   │   └── report.py              rapport Markdown : critères d'acceptation, verdict, comparaison buy & hold
│   ├── sizing/
│   │   └── sizing.py              vol targeting, Kelly (information, plafonné à 0,25 f*), limites de risque
│   ├── backtest/
│   │   ├── events.py              fusion ordonnée des flux, latences données / ordre paramétrables
│   │   ├── fills.py               taker (slippage selon la profondeur) ; maker (traversée stricte, position en queue)
│   │   ├── engine.py              boucle événementielle, portefeuille, frais, arrondis, journal
│   │   └── leakage.py             test anti-fuite (features décalées de +1 événement)
│   ├── live/
│   │   ├── risk.py                limites, kill switch (fraîcheur, perte journalière, drawdown par palier)
│   │   ├── broker.py              interface d'ordres ; live impossible sans flag explicite + clé sans retrait vérifiée
│   │   └── paper.py               paper trading sur flux live, journal complet de chaque décision
│   └── longterm/
│       ├── klines.py              bougies 1d / 1h → Bronze, contrôle qualité, trous marqués
│       ├── universe.py            univers point-in-time (listing, délisting, chauffe) sans biais du survivant
│       ├── signals.py             momentum série temporelle, moyennes mobiles, momentum transversal, inverse vol
│       ├── allocation.py          poids cibles → ordres : buy & hold, DCA, rééquilibrage calendaire / bandes, δ_min
│       ├── lt_costs.py            GATE LT : turnover, drag, rejets minNotional par palier de capital
│       ├── lt_backtest.py         barre à barre (décision clôture t, exécution ouverture t+1), apports, TWR / MWR
│       └── lt_report.py           CAGR, vol, MaxDD, Calmar, Sortino, PSR / DSR / MinTRL, vs buy & hold et DCA
└── tests/
    ├── conftest.py                uniquement la fixture de config temporaire (tmp_path) ; les données de test sont dans chaque test
    └── test_<nom>.py              un fichier de test par module source (données construites à la main, valeurs attendues écrites)
```

## Données (hors dépôt, chemin racine défini dans `config/base.yaml`)

```
$DATA_ROOT/
├── raw/{source}/{stream}/{symbol}/date=YYYY-MM-DD/HH.jsonl.zst      append-only, jamais modifié
├── bronze/{source}/{stream}/{symbol}/date=YYYY-MM-DD/*.parquet      ZSTD, dédupliqué, idempotent
├── silver/{features|bars|labels}/{symbol}/date=YYYY-MM-DD/*.parquet features dérivées uniquement
├── lt/{klines_1d|klines_1h}/{symbol}.parquet                          long terme (< 1 Go)
├── meta/
│   ├── exchange_info/{source}/{YYYYMMDDTHHMMSSZ}.json                 snapshots versionnés
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
| 3 bis | 0 · Socle | `core/errors.py` | validé |
| 4 | 0 · Socle | `core/hashing.py` | validé |
| 5 | 0 · Socle | `core/ledger.py` | validé |
| 6 | 0 · Socle | `exchange/lot.py` | à faire |
| 7 | 0 · Socle | `exchange/exchange_info.py` | à faire |
| 8 | 0 · Socle | `exchange/effective_params.py` | à faire |
| 9 | 0 · Données réelles | `data/archives.py` | à faire |
| 10 | 1 · Gate | `costs/cost_model.py` | à faire |
| 11 | 1 · Gate | `costs/cost_gate.py` | à faire |
| — | 1 · Gate | **Cost gate v1 sur données réelles** : aggTrades (archives Binance) + book_ticker (jours gratuits Tardis.dev) | à faire |
| 12 | 2 · Recherche | `research/hypothesis.py` | à faire |
| 13 | 2 · Recherche | `research/trials.py` | à faire |
| 14 | 2 · Recherche | `research/stats.py` | à faire |
| 15 | 2 · Recherche | `research/bootstrap.py` | à faire |
| 16 | 2 · Recherche | `research/cv.py` | à faire |
| 17 | 2 · Recherche | `research/report.py` | à faire |
| 18 | 3 · Long terme | `longterm/klines.py` | à faire |
| 19 | 3 · Long terme | `longterm/universe.py` | à faire |
| 20 | 3 · Long terme | `longterm/signals.py` | à faire |
| 21 | 3 · Long terme | `longterm/allocation.py` | à faire |
| 22 | 3 · Long terme | `longterm/lt_costs.py` | à faire |
| 23 | 3 · Long terme | `sizing/sizing.py` | à faire |
| 24 | 3 · Long terme | `longterm/lt_backtest.py` | à faire |
| 25 | 3 · Long terme | `longterm/lt_report.py` | à faire |
| 26 | 4 · Paper LT | `live/risk.py` | à faire |
| 27 | 4 · Paper LT | `live/broker.py` | à faire |
| 28 | 4 · Paper LT | `live/paper.py` | à faire |

**Volet intraday — conditionnel.** Il ne démarre que si le cost gate v1 trouve un horizon intraday franchissable (par exemple grâce à un palier de frais plus bas ou à l'exécution maker). Sinon, il reste en attente et le rapport du gate le dit.

| # | Phase | Fichier | Statut |
|---|---|---|---|
| 29 | 5 · Collecte | `data/cursor.py` | en attente du gate |
| 30 | 5 · Collecte | `data/raw_writer.py` | en attente du gate |
| 31 | 5 · Collecte | `data/collector.py` | en attente du gate |
| 32 | 5 · Collecte | `data/bronze.py` | en attente du gate |
| 33 | 5 · Collecte | `data/gaps.py` | en attente du gate |
| 34 | 5 · Collecte | `data/storage.py` | en attente du gate |
| — | 5 · Collecte | **Cost gate v2 sur bookTicker collecté** (≥ 7 jours) | en attente du gate |
| 35 | 6 · Features | `features/kernels.py` | en attente du gate |
| 36 | 6 · Features | `features/book.py` | en attente du gate |
| 37 | 6 · Features | `features/flow.py` | en attente du gate |
| 38 | 6 · Features | `features/spreads.py` | en attente du gate |
| 39 | 6 · Features | `features/volatility.py` | en attente du gate |
| 40 | 6 · Features | `features/seasonality.py` | en attente du gate |
| 41 | 7 · Recherche ID | `sampling/bars.py` | en attente du gate |
| 42 | 7 · Recherche ID | `sampling/labels.py` | en attente du gate |
| 43 | 7 · Recherche ID | `research/ic.py` | en attente du gate |
| 44 | 8 · Backtest ID | `backtest/events.py` | en attente du gate |
| 45 | 8 · Backtest ID | `backtest/fills.py` | en attente du gate |
| 46 | 8 · Backtest ID | `backtest/engine.py` | en attente du gate |
| 47 | 8 · Backtest ID | `backtest/leakage.py` | en attente du gate |

Les petits fichiers sans logique (`__init__.py`, etc.) accompagnent le fichier source suivant au lieu d'occuper un tour.

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

Ratio = mouvement absolu médian du mid / c (taker, sans BNB). Même avec BNB, aucun horizon ≤ 4 h ne dépasse 3. Limites : un seul mois, spreads mesurés sur 2 minutes seulement, 5 s et 30 s non mesurables avec des bougies 1 min (ils seraient encore plus bas). Le cost gate v1 (étape 11) refera ce calcul proprement.

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
