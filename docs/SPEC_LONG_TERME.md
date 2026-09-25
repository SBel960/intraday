# ADDENDUM — Volet long terme (horizon jours → mois)

Complète `SPEC_INTRADAY.md`. **Toutes les règles de l'intraday s'appliquent** : fiche d'hypothèse obligatoire, journal d'essais (compteur `N` pour le DSR), coûts d'abord, pas de levier, spot uniquement, aucune valeur métier codée en dur, protocole « un fichier à la fois » du §11.

Ce qui change : l'horizon (jours à mois), la source de données (bougies journalières, stockage négligeable) et l'adversaire principal. À 50 €, ce n'est plus le spread mais **`minNotional` et le turnover** : un rééquilibrage trop fin est tout simplement impossible, et un rééquilibrage trop fréquent coûte plus qu'il ne rapporte.

---

## LT.1 Données

- Bougies spot `1d` (et `1h` pour estimer la volatilité) : archives `data.binance.vision` (avec vérification du checksum) + endpoint REST `klines` pour les derniers jours.
- Colonnes conservées : `open_time_ms, open, high, low, close, volume_base, volume_quote, n_trades, taker_buy_base, taker_buy_quote`.
- Convention : une bougie `1d` Binance couvre `[00:00, 24:00[` UTC. Une bougie n'est **utilisable qu'une fois close**.
- **Univers point-in-time** : une paire n'entre dans l'univers qu'après sa première cotation + une période de chauffe (config). Les paires délistées restent dans l'historique, ce qui évite le biais du survivant.
- Trous : marqués, jamais interpolés. La période concernée est exclue.
- Rendements : $r_t = \ln(C_t / C_{t-1})$ sur la clôture. À l'horizon journalier, le rebond bid-ask est négligeable devant le mouvement. Le **coût**, lui, vient toujours des spreads mesurés (`bookTicker`, §5) ou, à défaut, d'une hypothèse conservatrice paramétrée et signalée dans le rapport.
- Stockage estimé : moins de 1 Go pour plusieurs centaines de paires en `1d` + `1h`.

## LT.2 Références obligatoires (dans chaque rapport)

- **Buy & hold** (achat unique en $t_0$) de chaque actif et du panier équipondéré.
- **DCA** : montant fixe $A$ tous les $\Delta$ jours (config).
  - La performance DCA est mesurée en **TWR** (time-weighted), pour neutraliser les apports.
  - Le rendement de l'investisseur est mesuré en **MWR** (taux interne de rentabilité).
  - On ne confond jamais apport et performance.

## LT.3 Signaux candidats (chacun exige sa fiche d'hypothèse)

Spot et sans short : un signal négatif signifie **cash (stablecoin ou EUR)**, jamais une vente à découvert.

**Momentum série temporelle** (Moskowitz, Ooi, Pedersen 2012) :
$$s_t = \ln\frac{C_t}{C_{t-L}}, \qquad \text{position} = \mathbb{1}_{\{s_t > 0\}}$$

**Croisement de moyennes mobiles** : long si $\mathrm{MA}_{L_1}(C)_t > \mathrm{MA}_{L_2}(C)_t$ avec $L_1 < L_2$.

**Momentum transversal** : les actifs de l'univers point-in-time sont classés par $\ln(C_{t-S}/C_{t-L})$ (on saute les $S$ derniers jours). On détient les $k$ premiers en équipondéré.

**Inverse volatilité** :
$$w_i \propto \frac{1}{\hat\sigma_{i,t}}$$

**Vol targeting du portefeuille** (reste en cash, pas de levier) :
$$w_t = \min\!\left(\frac{\sigma_{\text{cible}}}{\hat\sigma_{p,t}},\; w_{\max}\right), \qquad w_{\max} \le 1$$

**Allocation fixe rééquilibrée** : poids cibles $w^*$ (ex. BTC / ETH / cash), rééquilibrage calendaire ou à bandes.

Toute fenêtre ($L, L_1, L_2, S, k$) essayée est un essai journalisé.

## LT.4 Coûts et contraintes du petit capital — LE GATE LT

Coût relatif d'un ordre sur l'actif $i$ :
$$c_i = f_{\text{taker}} + \tfrac{1}{2}\tilde s_i + \text{slip}_i$$

Coût d'un rééquilibrage de valeur de portefeuille $V$ :
$$C = V \sum_i |\Delta w_i|\, c_i$$

Turnover annuel et frottement :
$$\text{TO}_{\text{an}} = \sum_{\text{rééquilibrages}} \sum_i |\Delta w_i|, \qquad \text{drag} = \text{TO}_{\text{an}} \times \bar c$$

**Contrainte `minNotional`** : un ordre de taille $|\Delta w_i|\,V < \text{minNotional}$ est rejeté. D'où une **bande minimale de non-transaction** :
$$\delta_{\min} = \frac{\text{minNotional}}{V}$$
Par exemple, si `minNotional` = 5 et $V$ = 50, alors $\delta_{\min}$ = 10 %. La valeur réelle vient toujours d'`exchangeInfo`.

**Rééquilibrage à bandes** : on ne trade l'actif $i$ que si $|w_i - w_i^*| > \delta$, avec $\delta \ge \delta_{\min}$.

**Conversion EUR → devise de cotation** : son coût est un paramètre de config. On peut aussi utiliser des paires EUR directes si le gate les montre moins chères.

**Livrable `lt_costs.py`**. Pour chaque stratégie candidate, à chaque palier de capital (config), il produit :
1. le turnover annuel attendu ;
2. le drag annuel ;
3. le nombre et la part d'ordres rejetés par `minNotional` ;
4. le rapport entre le drag et l'edge minimal requis inscrit dans la fiche d'hypothèse.

Si le drag consomme plus que la fraction autorisée de l'edge visé (config), la stratégie n'est pas testée. Le rapport le dit explicitement.

## LT.5 Backtest barre à barre

- Décision à la **clôture de la barre $t$**, avec des données $\le t$ seulement. Exécution au prix d'**ouverture de $t+1$**, plus le demi-spread, le slippage et les frais.
- Arrondis `tickSize` / `stepSize` et rejet sous `minNotional` à chaque ordre (réutilise `exchange/lot.py`).
- Apports DCA gérés comme des flux externes. TWR et MWR sont calculés séparément.
- **Test anti-fuite** : décaler les signaux d'une barre vers le futur doit **améliorer** artificiellement le résultat. Sinon, le pipeline est suspect.
- Limite assumée : `exchangeInfo` n'est pas disponible historiquement. On applique le premier snapshot versionné au passé, et le rapport le signale.

## LT.6 Validation (réutilise le §7)

- Sharpe annualisé sur 365 jours. PSR, DSR (avec le `N` d'essais du volet long terme), Lo (2002), Newey–West, stationary bootstrap.
- **Minimum Track Record Length** (Bailey & López de Prado 2012), en nombre d'observations, avec un SR non annualisé :
$$\mathrm{MinTRL} = 1 + \left[1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2\right]\left(\frac{z_{1-\alpha}}{\widehat{SR} - SR^*}\right)^2$$
  Si l'historique disponible est plus court que MinTRL, le rapport écrit **« non concluant »**.
- Walk-forward à fenêtre expansive. Purged K-fold avec un embargo ≥ lookback maximal + horizon de détention.
- Métriques : CAGR, vol annualisée, MaxDD et durée du drawdown, Calmar, Sortino, turnover, exposition moyenne, nombre de trades, drag.

**Critères d'acceptation long terme** (tous requis) :
- DSR > 0,95 ;
- différentiel de Sharpe **après coûts** face au buy & hold du même actif ou panier, avec un IC bootstrap à 95 % strictement positif ;
- stabilité sur au moins 3 sous-périodes, dont au moins un marché baissier, et sur au moins 2 actifs ;
- historique ≥ MinTRL ;
- résultats du paper trading dans l'intervalle de confiance du backtest.

## LT.7 Exploitation

- Paper trading long terme : au plus une décision par barre. Kill switch si la barre du jour est absente ou périmée au moment de la décision.
- Mêmes limites de risque, même journal, **live désactivé par défaut** (même flag explicite, clés sans droit de retrait).
- Paliers de capital en config. `lt_costs.py` est relancé à chaque changement de palier.

## À ne pas faire (en plus de la liste intraday)

- Pas de backtest sur l'univers **actuel** projeté dans le passé (biais du survivant).
- Pas de performance DCA présentée sans séparer TWR et MWR.
- Pas de stratégie long terme présentée sans le buy & hold et le DCA sur la même période.
