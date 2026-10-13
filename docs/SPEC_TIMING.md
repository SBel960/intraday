# Spécification — volet « timing » : meilleur moment d'achat et de vente

Rédigée le 2026-10-14, à la demande du propriétaire. À valider avant tout code. Ce volet reprend les règles de `SPEC_INTRADAY.md` (§4 labels, §5 gate, §7 validation, §11 un fichier par tour) ; seules les différences sont écrites ici.

## 0. Décisions du propriétaire (2026-10-14)

1. **But** : trouver le couple *signal × horizon* qui donne le meilleur gain **net de frais**, en achetant **et** en vendant. L'horizon n'est pas fixé d'avance (pas seulement 2 à 3 h) : c'est un résultat de la recherche.
2. **Long et short.** Le short n'existe pas en spot. Ce volet étudie donc deux supports, sur Binance, pour tester :
   - **spot** : long ou à plat (euros) ;
   - **perpétuels USDⓈ-M** : long, short ou à plat, **levier ≤ 1×** (notionnel ≤ capital), marge isolée.
   C'est un écart assumé à `SPEC_INTRADAY.md` §6 (« spot uniquement, pas de levier »), limité à ce volet et à 1×.
3. **Recherche et paper uniquement.** Aucun ordre réel. Le passage en réel est hors champ tant que l'accès au compte n'est pas vérifié (selon la presse, Binance a cessé de servir les résidents français le 2026-07-01, faute d'agrément MiCA).
4. Le paper long terme (`lt_ts_momentum · lookback_days=30`) continue jusqu'à ses 60 jours, sans changement.
5. Capital de référence : 50 EUR, comme le reste du projet.

## 1. Ce qui est réutilisé, pas réécrit

Socle `core/*`, frais `exchange/fees.py`, filtres de lot `exchange/lot.py`, spreads mesurés `exchange/spreads.py`, téléchargements `data/binance_vision.py` + `core/downloads.py`, coût `costs/cost_model.py`, fiches `research/hypothesis.py`, essais `research/trials.py`, statistiques `research/stats.py`, `bootstrap.py`, `cv.py`, `ic.py`, `report.py`, garde-fous `live/risk.py`.

Changements à des fichiers validés (chacun signalé et revalidé à son tour) :
- `research/hypothesis.py` : volet `timing` ajouté à `VOLETS`.
- `data/binance_vision.py` : clés des bougies 1 min des perpétuels et des bougies d'indice de prime (`premiumIndexKlines`).
- `exchange/spreads.py` + minuteur : relevé du bookTicker des perpétuels en plus du spot.

## 2. Données (gratuites, Binance Vision)

| Jeu | Usage | Paires |
|---|---|---|
| bougies spot 1 min | prix, volume, **volume acheteur agressif** (`taker_buy_base`), nombre de trades | observées : ~20 USDT les plus liquides ; tradées : BTC, ETH, SOL (EUR et USDT) |
| bougies perpétuels 1 min | prix du contrat, flux acheteur / vendeur | mêmes bases, USDⓈ-M |
| `premiumIndexKlines` 1 min | base perpétuel − spot | idem |
| `fundingRate` | coût de portage du perpétuel, signal | idem |
| `metrics` (5 min) | positions ouvertes, ratios long / short | idem |

- Les **aggTrades ne sont pas téléchargés** au départ : plusieurs dizaines de Go pour BTC seul. Les bougies 1 min portent déjà le déséquilibre acheteur / vendeur. On ne les ajoute que si une fiche le justifie, après estimation de la taille.
- Volume estimé avant téléchargement et affiché ; plafond dans la config (`timing.data.max_gb`) dans le budget de 150 Go (8,8 Go utilisés au 2026-10-14).
- Trous marqués, jamais interpolés ; la période touchée est exclue.
- **Observer large, trader étroit** (comme le long terme) : l'IC se mesure sur toutes les paires observées, pour la puissance statistique ; on ne trade que BTC, ETH et SOL.

## 3. Gate v2 : à partir de quel horizon le mouvement couvre-t-il les frais ?

Même formule qu'au §5 de l'intraday, étendue :

$$c = 2f + \tilde s + 2\,\text{slip} + \text{funding}(h) \quad\text{(funding : perpétuels seulement)}$$

- Horizons : **5 min, 15 min, 30 min, 1 h, 2 h, 3 h, 4 h, 6 h, 8 h, 12 h, 24 h, 48 h, 72 h** (`timing.gate.horizons_s`).
- Mouvement = |ln P(t+h) / P(t)| sur les clôtures 1 min, sur tout l'historique disponible (pas seulement le 1er du mois).
- Frais : barème du compte (spot) et barème publié des perpétuels, en config, en variante sans et avec remise BNB.
- Spread : médiane mesurée (`exchange/spreads.py`), sinon `costs.fallback_spread_frac`, signalé.
- Sortie, par paire, support (spot / perpétuel), heure UTC et variante de frais : coût médian et p90, mouvement médian, ratio, **plus petit horizon où le ratio dépasse 3**.
- Les horizons sous ce seuil ne sont pas testés : le rapport le dit.

## 4. Fiches d'hypothèse (écrites avant tout test, puis figées)

Une fiche par mécanisme, `hypotheses/timing/*.yaml`, avec une petite grille. Familles prévues :

1. **Saisonnalité horaire et hebdomadaire** : ouverture des marchés américains, creux asiatique, week-end.
2. **BTC en avance** : BTC bouge, ETH / SOL suivent avec retard.
3. **Déséquilibre du flux** : part du volume acheteur agressif, chocs de volume.
4. **Excès puis retour ou continuation** : après un mouvement de plus de k σ sur 1 à 6 h.
5. **Positionnement des perpétuels** : funding, base, positions ouvertes, ratio long / short (foule à levier d'un côté ⇒ mouvement inverse ?).
6. **Régime de volatilité / cassure** : sortie d'une plage calme.

**Budget d'essais** : `timing.max_trials` dans la config (proposé : 60), signaux × horizons × paramètres compris. Une fois le budget atteint, on s'arrête. « Le plus optimisé » est le meilleur **après** pénalité du nombre d'essais (DSR), pas le meilleur backtest brut.

## 5. Labels et décisions

- Décision prise à la clôture d'une bougie 1 min ; exécution sur la bougie suivante (latence ≥ 1 min, `timing.latency_bars`).
- Label principal : rendement futur sur l'horizon h, **net** du coût aller-retour.
- Variante : triple barrière (prise de gain, stop, temps), barrières exprimées en multiples de σ, coûts inclus.
- Position : +1 / 0 / −1 (−1 seulement sur perpétuel), taille par vol targeting (`SPEC_INTRADAY.md` §6), plafonnée à 1×.

## 6. Validation (critères de `SPEC_INTRADAY.md` §7, tous requis)

- IC par horizon d'abord (`research/ic.py`) : un signal sans IC significatif à aucun horizon n'est pas backtesté.
- Purged K-fold avec embargo ≥ h, DSR > 0,95 sur le nombre total d'essais du volet.
- Edge net médian par trade > 0, IC bootstrap à 95 % strictement positif.
- Stable sur ≥ 3 sous-périodes (dont une baissière) et ≥ 2 actifs.
- **Hold-out** : les 12 derniers mois avant le gel (`timing.holdout_start`, proposé : 2025-10-01) ne sont touchés qu'une seule fois, pour le candidat final.
- Toujours comparé au buy & hold et à la stratégie long terme sur la même période.
- Verdict acceptable : « rien ne passe ».

## 7. Backtest

- Rejeu sur bougies 1 min ; un ordre taker s'exécute au prix d'ouverture de la bougie d'exécution ± demi-spread + glissement.
- Perpétuels : funding débité ou crédité aux heures de règlement ; contrôle de marge à 1× (liquidation impossible, mais vérifiée).
- Arrondis `tickSize` / `stepSize` et rejet sous `minNotional` (`exchange/lot.py`), frais à chaque exécution.
- Test anti-fuite (§8 intraday) : décaler les signaux d'une bougie vers le futur doit améliorer le résultat.

## 8. Paper trading

Données publiques en direct, horloge contrôlée contre le serveur (seuil en config), garde-fous `live/risk.py` (perte du jour, drawdown, fraîcheur), journal complet. Durée minimale avant de conclure : `timing.paper_min_days`.

## 9. Ordre de livraison (un fichier + son test par tour)

1. `docs/SPEC_TIMING.md` (ce fichier)
2. `data/binance_vision.py` (perpétuels 1 min, premiumIndex) puis `timing/minute_bars.py` (téléchargement + construction des bougies 1 min, contrôle qualité)
3. `costs/horizon_gate.py` + rapport **gate v2**
4. `timing/features.py`
5. `timing/labels.py`
6. `hypotheses/timing/*.yaml` (+ volet `timing` dans `research/hypothesis.py`)
7. `timing/ic_scan.py` (IC signal × horizon)
8. `timing/backtest.py`
9. `timing/evaluate.py` (verdict, rapport)
10. `live/paper_timing.py` + minuteur
