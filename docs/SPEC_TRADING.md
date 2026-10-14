# Spécification — volet « trading » : positions de quelques heures sur des marchés bon marché

Rédigée le 2026-10-14 à la demande du propriétaire, qui a choisi l'option 1. La validation fichier par fichier est levée pour cette recherche. Les règles statistiques restent celles du projet : fiches écrites avant les tests, essais comptés, DSR, hold-out unique.

## 0. Pourquoi

- **Le trading horaire sur la crypto perd à cause des frais.** Sur les perpétuels Binance (volet timing), un aller-retour coûte de 0,2 à 0,3 % (frais taker, spread, glissement). Les essais à 1 à 4 h y finissaient avec un Sharpe de −5 à −50.
- **Sur les grands marchés à terme et de devises, c'est 10 à 30 fois moins cher.** Un aller-retour sur l'euro-dollar, le S&P 500, l'or ou le pétrole coûte de 0,005 à 0,02 %. Ce volet reprend les mêmes idées à l'échelle de l'heure, là où les frais ne les condamnent pas d'avance.

## 1. Univers (`config/trading.yaml`)

21 instruments :

| Classe | Instruments |
|---|---|
| Devises | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, EURGBP, indice dollar |
| Indices | S&P 500, Nasdaq 100, DAX, FTSE 100, CAC 40, Euro Stoxx 50, Nikkei 225 |
| Matières premières | or, argent, pétrole WTI et Brent |

En réel, ce serait des futures ou du change au comptant chez un courtier à bas coût, par exemple Interactive Brokers, accessible depuis la France. **Aucun ordre réel n'est passé dans ce volet.** Les taux longs et le gaz, prévus au départ, ne sont pas chez la source retenue.

## 2. Données

- **Source** : HistData, gratuite et sans clé. Barres d'**une minute** depuis 2010 : un zip par année, ou par mois pour l'année en cours, avec un formulaire à jeton.
- **Changement de source** : Dukascopy, prévue au départ, a été abandonnée le 2026-10-14. Elle freinait trop (503, connexions coupées) : 7 fichiers en 25 minutes pour 7 788 à télécharger. Les 7 fichiers bruts déjà reçus restent sous `raw/dukascopy/` ; ils ne sont pas supprimés sans accord.
- **Heure** : la source donne l'heure de New York **sans heure d'été**, soit UTC − 5 toute l'année.
- **Barres** : les minutes sont regroupées en barres horaires UTC. Copies brutes sous `raw/histdata/` (jamais modifiées, reprise possible) ; barres sous `trading/hourly/{instrument}.parquet`.
- **Heures fermées** (week-ends, fériés, pauses des indices) : clôture reportée, aucune entrée ni sortie. C'est la même règle que le volet multi-actifs.

## 3. Coûts d'une jambe

- **½ spread complet supposé par instrument** (config). La source ne donne pas le bid / ask, donc on prend des valeurs prudentes : de 0,005 % (EURUSD) à 0,04 % (argent).
- **Heures de bascule exclues** : 21 h et 22 h UTC ne sont pas tradables, parce que les spreads s'y écartent.
- **Commission et glissement** par classe : de 0,004 à 0,008 %.
- **Non modélisé en version 1** : le portage d'une nuit (« swap » en CFD, intégré au prix en futures).

## 4. Portefeuille, référence, validation

- **Moteur** : celui des volets timing et multi-actifs (`timing/portfolio.py`), avec des poids inverses de la volatilité sur 20 jours et un levier de 1× au plus.
- **Référence** : panier de tous les instruments éligibles, avec les mêmes poids.
- **Hold-out** : à partir du **2024-10-01**, lu une seule fois.
- **Critères** : DSR sur le N du volet `trading`, écart de Sharpe avec la référence, années civiles, deux moitiés disjointes des instruments, MinTRL, contrôle anti-fuite.

## 5. Fiches (volet `trading`, écrites avant les tests, puis figées)

| Fiche | Idée | Essais |
|---|---|---|
| `tr_reversal` | retour à la moyenne après un excès de 1 à 24 h (fournisseurs de liquidité) | lookback {1, 4, 24} × horizon {1, 4} = 6 |
| `tr_momentum` | les instruments les plus forts de 4 h à 5 jours continuent (transversal) | {4, 24, 120} × {1, 4, 24} = 9 |
| `tr_trend` | suivi de tendance signé par instrument | {24, 120, 480} × {4, 24} = 6 |
| `tr_channel` | sortie de range de 1 à 5 jours (cassure, signé) | {24, 120} × {4, 24} = 4 |
| `tr_seasonality` | heures de la semaine : ouverture de Londres, de New York, fixing de 16 h | horizon {1, 4} = 2 |

Cela fait 27 essais ; budget `trading.max_trials` = 60.

## 6. Fichiers

| Fichier | Rôle |
|---|---|
| `data/histdata.py` | téléchargement brut (page à jeton + formulaire) et barres horaires UTC |
| `core/config.py` + `config/trading.yaml` | instruments, classes, coûts, hold-out, budget |
| `core/http.py` | envoi d'un formulaire (POST), même politique de relance |
| `timing/portfolio.py` | ½ spread de la barre du trade si le panel porte `spread_frac` |
| `timing/panel.py` | `carry_closed`, commun aux marchés qui ferment |
| `timing/verdict.py` | sous-commandes communes des volets (`scan`, `best`, `holdout`) |
| `trading/fx_panel.py` | panel horaire (heures fermées, spread, éligibilité) |
| `trading/intraday_study.py` | étude et commandes du volet |
| `hypotheses/trading/*.yaml` | fiches du §5 |
