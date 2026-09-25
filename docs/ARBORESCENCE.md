# Arborescence proposée — EN ATTENTE DE VALIDATION

Package Python : `qlab`, avec un volet intraday et un volet long terme qui partagent le socle.
Chaque module exécutable expose un `main()` lancé par `python -m qlab.<module> --config ...`. Il n'y a pas de CLI centrale, car elle obligerait à revalider un fichier déjà validé à chaque étape.

Colonne « Statut » : `à faire` → `livré` → `validé` (ou `à revalider`).

## Dépôt

```
intraday/
├── pyproject.toml                 dépendances, config pytest / mypy --strict / ruff
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
│   │   └── hashing.py             hash déterministe de fichiers et de tables Parquet (tests d'idempotence)
│   ├── exchange/
│   │   ├── exchange_info.py       récupère exchangeInfo (Binance / Bybit) et le barème de frais, stocke des snapshots versionnés
│   │   └── lot.py                 arrondi prix → tickSize, quantité → stepSize (Decimal), rejet sous minNotional
│   ├── costs/
│   │   ├── cost_model.py          c_taker, c_maker, p*, drag annuel (§5)
│   │   └── cost_gate.py           LE GATE : distribution de c, mouvement médian par horizon, ratio, horizon min > 3
│   ├── data/
│   │   ├── cursor.py              curseur persistant et atomique (reprise après crash, ni trou ni doublon)
│   │   ├── raw_writer.py          RAW JSONL.zst append-only, écriture par lots, rotation horaire
│   │   ├── collector.py           websockets aggTrade + bookTicker, reconnexion, horodatage de réception
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
│   ├── gaps.parquet                                                   table des trous
│   └── trials.jsonl                                                   registre d'essais (N)
└── reports/                                                           rapports générés
```

## Ordre de livraison (un fichier source + son test par tour)

| # | Phase | Fichier | Statut |
|---|---|---|---|
| 1 | 0 · Socle | `core/config.py` (+ `pyproject.toml`, `config/*.yaml`, `conftest.py`) | à faire |
| 2 | 0 · Socle | `core/timeutils.py` | à faire |
| 3 | 0 · Socle | `core/jsonlog.py` | à faire |
| 4 | 0 · Socle | `exchange/lot.py` | à faire |
| 5 | 0 · Socle | `exchange/exchange_info.py` | à faire |
| 6 | 0 · Données réelles | `data/archives.py` | à faire |
| 7 | 1 · Gate | `costs/cost_model.py` | à faire |
| 8 | 1 · Gate | `costs/cost_gate.py` | à faire |
| — | 1 · Gate | **Cost gate v1 sur données réelles** : aggTrades (archives Binance) + book_ticker (jours gratuits Tardis.dev) | à faire |
| 9 | 2 · Collecte | `core/hashing.py` | à faire |
| 10 | 2 · Collecte | `data/cursor.py` | à faire |
| 11 | 2 · Collecte | `data/raw_writer.py` | à faire |
| 12 | 2 · Collecte | `data/collector.py` | à faire |
| 13 | 2 · Collecte | `data/bronze.py` | à faire |
| 14 | 2 · Collecte | `data/gaps.py` | à faire |
| 15 | 2 · Collecte | `data/storage.py` | à faire |
| — | 2 · Collecte | **Cost gate v2 sur bookTicker collecté** (BTCUSDT, ETHUSDT, ≥ 7 jours de bookTicker) | à faire |
| 16 | 3 · Features | `features/kernels.py` | à faire |
| 17 | 3 · Features | `features/book.py` | à faire |
| 18 | 3 · Features | `features/flow.py` | à faire |
| 19 | 3 · Features | `features/spreads.py` | à faire |
| 20 | 3 · Features | `features/volatility.py` | à faire |
| 21 | 3 · Features | `features/seasonality.py` | à faire |
| 22 | 4 · Recherche | `research/hypothesis.py` | à faire |
| 23 | 4 · Recherche | `research/trials.py` | à faire |
| 24 | 4 · Recherche | `sampling/bars.py` | à faire |
| 25 | 4 · Recherche | `sampling/labels.py` | à faire |
| 26 | 4 · Recherche | `research/stats.py` | à faire |
| 27 | 4 · Recherche | `research/bootstrap.py` | à faire |
| 28 | 4 · Recherche | `research/cv.py` | à faire |
| 29 | 4 · Recherche | `research/ic.py` | à faire |
| 30 | 4 · Recherche | `research/report.py` | à faire |
| 31 | 5 · Long terme | `longterm/klines.py` | à faire |
| 32 | 5 · Long terme | `longterm/universe.py` | à faire |
| 33 | 5 · Long terme | `longterm/signals.py` | à faire |
| 34 | 5 · Long terme | `longterm/allocation.py` | à faire |
| 35 | 5 · Long terme | `longterm/lt_costs.py` | à faire |
| 36 | 5 · Long terme | `sizing/sizing.py` | à faire |
| 37 | 5 · Long terme | `longterm/lt_backtest.py` | à faire |
| 38 | 5 · Long terme | `longterm/lt_report.py` | à faire |
| 39 | 6 · Backtest ID | `backtest/events.py` | à faire |
| 40 | 6 · Backtest ID | `backtest/fills.py` | à faire |
| 41 | 6 · Backtest ID | `backtest/engine.py` | à faire |
| 42 | 6 · Backtest ID | `backtest/leakage.py` | à faire |
| 43 | 7 · Paper | `live/risk.py` | à faire |
| 44 | 7 · Paper | `live/broker.py` | à faire |
| 45 | 7 · Paper | `live/paper.py` | à faire |

Le long terme est placé après la phase 4 parce qu'il réutilise `stats`, `bootstrap`, `cv`, `trials` et `report`, mais pas les features de microstructure ni le backtester événementiel. Il peut donc avancer pendant que le collecteur accumule des données. À 50 €, c'est aussi le volet le plus susceptible de passer son gate.

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
