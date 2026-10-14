# Spécification — volet « multi-actifs » : acheter et vendre sur tous les marchés

Rédigée le 2026-10-14, à la demande du propriétaire : « élargir notre monde, ne pas se baser sur le BTC uniquement, être ouvert à toutes les sources monétaires envisageables ». La validation fichier par fichier est levée pour cette recherche (« pas besoin que je valide tous », 2026-10-14). Les règles statistiques restent celles du projet : fiches écrites avant les tests, essais comptés, DSR, hold-out unique.

## 0. Pourquoi

Le volet timing (perpétuels Binance, 105 essais) n'a rien trouvé qui batte BTC. Deux limites venaient du terrain de jeu, pas de la méthode :
1. **Un seul marché, très corrélé.** Presque toutes les cryptos montent et baissent ensemble. Un suivi de tendance n'a donc qu'un seul pari indépendant à faire.
2. **Une référence très dure.** BTC a un Sharpe de +1,1 sur 2020 → 2025. Le battre revient à battre la meilleure classe d'actifs de la période.

Le résultat le plus solide de la littérature sur le « bon moment pour acheter ou vendre » vient du suivi de tendance **multi-actifs** (Moskowitz, Ooi et Pedersen, 2012 ; Hurst, Ooi et Pedersen, un siècle de données) : chaque marché seul donne peu, mais 40 à 60 marchés peu corrélés (actions, taux, devises, matières premières) se diversifient. C'est ce que ce volet teste.

## 1. Univers (45 marchés, `config/markets.yaml`)

Ce sont des **ETF cotés aux États-Unis**, pris comme mesure du rendement réellement obtenu. Leur cours ajusté inclut les dividendes, et pour les matières premières le coût du roulement des contrats : pas de faux sauts au changement d'échéance.

| Classe | Marchés |
|---|---|
| Actions | États-Unis (SPY, QQQ, IWM), Europe (VGK, EWG, EWU), Japon (EWJ), émergents (EEM, FXI, EWZ, EWT, EWY), Canada (EWC), Australie (EWA), Inde (INDA) |
| Obligations | Trésor US 1-3 ans (SHY), 7-10 ans (IEF), 20 ans et plus (TLT), indexées (TIP), entreprises (LQD), haut rendement (HYG), émergentes (EMB), internationales (BWX) |
| Devises (intérêts compris) | euro (FXE), yen (FXY), livre (FXB), dollar australien (FXA), dollar canadien (FXC), franc suisse (FXF), dollar US (UUP) |
| Matières premières | or (GLD), argent (SLV), pétrole (USO), gaz (UNG), cuivre (CPER), métaux industriels (DBB), agriculture (DBA), blé (WEAT), maïs (CORN), panier (DBC) |
| Immobilier | États-Unis (VNQ), monde hors US (RWX) |
| Crypto | BTC-USD, ETH-USD |

- **Sur l'accès réel** : un particulier français ne peut pas acheter d'ETF américains, faute de document d'information clé (PRIIPs). Ces séries mesurent le rendement. En réel, on passerait par des équivalents UCITS ou des futures, par exemple chez Interactive Brokers. Les coûts du §3 sont prévus pour couvrir cet écart. Aucun ordre réel n'est passé dans ce volet.
- **Biais du survivant** : la liste est choisie aujourd'hui, mais ce sont des marchés larges, pas des titres individuels. Un ETF n'entre qu'à sa date de création, plus un an de recul (`warmup_bars`). Le biais restant est faible, et il est signalé dans le rapport.

## 2. Données

- **Source** : Yahoo Finance, graphique quotidien (`/v8/finance/chart`). Accès public, sans clé, **non documenté**. On utilise l'OHLC ajusté par le rapport `adjclose / close`.
- **Profondeur** (essayée le 2026-10-14) : 2000 à 2007 selon les ETF, 2014 pour BTC.
- **Grille** : jours ouvrés (lundi → vendredi, minuit UTC). Un jour férié reste une ligne à prix nul (trou marqué, jamais interpolé). Les cryptos tradent le week-end : leur rendement du vendredi au lundi tombe le lundi.
- Stockage : `markets/daily/{ticker}.parquet`, quelques Mo au total. Téléchargement par `core/http.py`, une requête par seconde.

## 3. Coûts (prudents, par classe, une jambe)

| Classe | Coût d'une jambe |
|---|---|
| Actions, obligations, devises (ETF très liquides) | 0,05 % |
| Matières premières, émergents | 0,10 % |
| Crypto | 0,15 % |

- Le coût d'une jambe couvre la commission, le ½ spread et le glissement.
- **Les ventes à découvert d'ETF ont un coût d'emprunt.** Il est d'environ 0,3 %/an pour les plus liquides et n'est pas modélisé en version 1. Le rapport donne l'exposition vendeuse moyenne, pour en chiffrer l'ordre de grandeur.

## 4. Portefeuille

C'est le moteur du volet timing (`timing/portfolio.py`), généralisé de l'heure à une **barre** quelconque (heure ou jour ouvré), avec deux options en plus :
- **pondération inverse de la volatilité** (`weighting: inverse_vol`) : poids ∝ 1 / σ (écart-type des rendements des `vol_lookback_bars` barres passées), avec Σ|w| = 1. Ainsi, un marché très agité (gaz) ne pèse pas 10 fois plus en risque qu'une obligation courte. **Levier 1× au plus**, jamais de ciblage de volatilité avec levier.
- **portefeuille signé** (déjà livré) : achat si le signal est > 0, vente s'il est < 0.

## 5. Référence

Un **panier qui achète tous les marchés éligibles, pondéré par l'inverse de la volatilité**, rééquilibré à chaque période et sans coûts. C'est le portefeuille passif diversifié (« risk parity » sans levier). Il remplace BTC : battre BTC n'a plus de sens quand BTC n'est qu'un marché parmi 45.

## 6. Fiches (volet `multi`, écrites avant les tests, puis figées)

| Fiche | Mécanisme | Essais |
|---|---|---|
| `mk_trend` | suivi de tendance signé, pondéré par le risque (TSMOM) | lookback ∈ {21, 63, 126, 252} j × horizon ∈ {5, 21} j = 8 |
| `mk_channel` | canal de Donchian signé (règle des turtles : 20 et 55 j) | {20, 55, 126} × {5, 21} = 6 |
| `mk_momentum` | momentum transversal entre marchés : 5 achetés, 5 vendus | {63, 126, 252} × {5, 21} = 6 |
| `mk_momentum_long_only` | les 8 plus forts achetés, sans vente | {126, 252} × {21} = 2 |
| `mk_reversal` | retour à la moyenne d'une semaine entre marchés | {5} × {5} = 1 |

Cela fait 23 essais ; budget `markets.max_trials` = 60. Le N est **propre au volet** : univers, données et fréquence sont différents du volet timing. Le rapport rappelle aussi les 105 essais du volet timing.

## 7. Validation

- **Hold-out** : depuis le **2024-10-01** (2 ans). Il n'est lu qu'une fois, pour un candidat qui passe tout le reste.
- **Critères** (inchangés) :
  - DSR ≥ 0,95 sur le N du volet ;
  - écart de Sharpe avec la référence du §5, avec un IC bootstrap strictement positif ;
  - années civiles ;
  - deux moitiés disjointes des marchés ;
  - MinTRL.
- **Contrôle anti-fuite** : le signal « tricheur » doit gagner nettement avec le même moteur.

## 8. Fichiers

| Fichier | Rôle |
|---|---|
| `data/yahoo.py` | téléchargement et mise en parquet des séries quotidiennes |
| `core/config.py` + `config/markets.yaml` | univers, classes, coûts, hold-out, budget |
| `markets/daily_panel.py` | panel quotidien (grille des jours ouvrés, barre, éligibilité) |
| `timing/panel.py`, `timing/portfolio.py`, `timing/scan.py`, `timing/verdict.py` | généralisés à la barre ; pondération inverse de la volatilité ; référence donnée par l'étude |
| `markets/multi_asset.py` | préparation de l'étude multi-actifs, commandes `scan` et `verdict` |
| `hypotheses/multi/*.yaml` | fiches du §6 |
