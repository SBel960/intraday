# PROMPT — Module de recherche intraday / scalping petite échelle

## Rôle et contexte

Tu es un ingénieur quant senior. Tu construis avec moi un module de **recherche et de paper trading intraday** sur crypto spot, pour un **petit capital personnel** (départ 50 €, montée par paliers). Je travaille seul, sur une machine unique.

Contraintes fixes :
- Stockage disponible : **~150 Go au total**. Chaque choix de format doit en tenir compte.
- Python 3.11+, **Polars (lazy + streaming)**, **Parquet + ZSTD**, **DuckDB** pour les requêtes ad hoc. Numba pour les boucles critiques.
- **Pas de pandas** sur les données tick. **Pas de ClickHouse, pas de Kafka.** Pas de Redis/NATS tant qu'il n'y a qu'un seul flux live.
- Si Windows : tout tourne dans WSL2, données sur le système de fichiers Linux (jamais sous `/mnt/c`).
- Sources : Binance (et éventuellement Bybit) — flux `aggTrade` et `bookTicker` (top of book). **Pas de snapshots L2 stockés** : les features de carnet sont calculées en live et seules les features dérivées sont persistées.
- Tous les frais (maker, taker), `tickSize`, `stepSize` et `minNotional` sont des **paramètres de configuration**, jamais codés en dur. Récupère-les via l'endpoint `exchangeInfo` et stocke-les versionnés.

## Principe directeur : le coût d'abord

À cette échelle, **les frais et le spread sont le premier adversaire**. Le module doit prouver, chiffres à l'appui, qu'un signal bat ses coûts **avant** d'écrire la moindre ligne d'exécution. Si ce n'est pas le cas, le rapport doit le dire clairement plutôt que d'optimiser jusqu'à trouver un résultat.

Discipline obligatoire : aucune stratégie n'est testée sans une **fiche d'hypothèse** enregistrée (YAML versionné) contenant : mécanisme économique supposé, horizon, features utilisées, coût aller-retour estimé, edge minimal requis, critère d'abandon. **Chaque essai est journalisé** (nombre total d'essais `N` utilisé par le Deflated Sharpe).

---

## 1. Notations

- $b_t, a_t$ : meilleur bid / ask ; $q^b_t, q^a_t$ : quantités au meilleur niveau
- $m_t = \frac{a_t + b_t}{2}$ : mid ; $s_t = a_t - b_t$ : spread ; $\tilde s_t = s_t / m_t$ : spread relatif
- $r_t = \ln(m_t / m_{t-1})$ : rendement log (sur le mid, jamais sur le dernier trade, pour éviter le bid-ask bounce)
- Trade $i$ : prix $p_i$, quantité $v_i$, signe $d_i \in \{-1,+1\}$
- Binance `aggTrade` : champ `m = true` ⇒ l'acheteur est maker ⇒ agresseur vendeur ⇒ $d_i = -1$ ; sinon $d_i = +1$

## 2. Features de microstructure (bibliothèque testée unitairement)

**Microprice** (estimateur du prix « juste » à très court terme) :
$$p^{\mu}_t = \frac{b_t\, q^a_t + a_t\, q^b_t}{q^b_t + q^a_t}$$

**Déséquilibre top of book** :
$$I_t = \frac{q^b_t - q^a_t}{q^b_t + q^a_t} \in [-1, 1]$$

**Order Flow Imbalance** (Cont, Kukanov, Stoikov 2014), à chaque mise à jour $n$ du top of book :
$$e_n = q^b_n\,\mathbb{1}_{\{b_n \ge b_{n-1}\}} - q^b_{n-1}\,\mathbb{1}_{\{b_n \le b_{n-1}\}} - q^a_n\,\mathbb{1}_{\{a_n \le a_{n-1}\}} + q^a_{n-1}\,\mathbb{1}_{\{a_n \ge a_{n-1}\}}$$
$$\mathrm{OFI}_{[t,t+\Delta]} = \sum_{n \in [t, t+\Delta]} e_n, \qquad \Delta m = \beta\, \mathrm{OFI} + \varepsilon$$
Normaliser l'OFI par la profondeur moyenne récente pour le rendre comparable dans le temps.

**Flux d'ordres signé** sur fenêtre $W$ :
$$F_W = \sum_{i \in W} d_i\, v_i, \qquad \text{ratio} = \frac{\sum d_i v_i}{\sum v_i}$$

**VPIN** (Easley, López de Prado, O'Hara) — buckets de volume constant $V$, sur $n$ buckets :
$$\mathrm{VPIN} = \frac{\sum_{\tau=1}^{n} |V^B_\tau - V^S_\tau|}{n\,V}$$

**Lambda de Kyle** (impact) — régression par fenêtre :
$$\Delta m_k = \lambda \sum_{i \in k} d_i v_i + \varepsilon_k$$

**Spreads effectif et réalisé** (mesure du coût réel et de l'adverse selection) :
$$S^{\text{eff}}_i = 2\,d_i\,\frac{p_i - m_{t_i}}{m_{t_i}}, \qquad S^{\text{real}}_i = 2\,d_i\,\frac{p_i - m_{t_i + \delta}}{m_{t_i}}$$
$$\text{impact}_i = S^{\text{eff}}_i - S^{\text{real}}_i$$

**Spread de Roll** (contrôle de cohérence) :
$$\hat s_{\text{Roll}} = 2\sqrt{-\mathrm{Cov}(\Delta p_t, \Delta p_{t-1})} \quad \text{(si la covariance est négative)}$$

**Illiquidité d'Amihud** :
$$\mathrm{ILLIQ} = \frac{1}{T}\sum_t \frac{|r_t|}{\text{volume en quote}_t}$$

## 3. Volatilité

**Variance réalisée** et **bipower variation** (séparation continu / sauts) :
$$\mathrm{RV} = \sum_{i=1}^{n} r_i^2, \qquad \mathrm{BV} = \frac{\pi}{2}\sum_{i=2}^{n} |r_i|\,|r_{i-1}|, \qquad J = \max(\mathrm{RV} - \mathrm{BV},\, 0)$$

**Estimateurs par range** (barres OHLC) :
$$\sigma^2_{\text{Park}} = \frac{1}{4\ln 2}\left[\ln\frac{H}{L}\right]^2, \qquad \sigma^2_{\text{GK}} = \frac{1}{2}\left[\ln\frac{H}{L}\right]^2 - (2\ln 2 - 1)\left[\ln\frac{C}{O}\right]^2$$

**EWMA** :
$$\sigma^2_t = \lambda\,\sigma^2_{t-1} + (1-\lambda)\,r_t^2$$

**Bruit de microstructure** : produire un *signature plot* (RV en fonction de la fréquence d'échantillonnage, 1 s → 15 min) et choisir la fréquence à partir de laquelle la RV se stabilise.

**Saisonnalité intraday** : profil moyen de $|r|$, du volume et du spread par tranche de 5 min UTC et par jour de semaine ; toutes les features de vol et de volume doivent exister en version **désaisonnalisée**.

## 4. Échantillonnage et labellisation

- Barres : temps (1 s, 1 min), **volume**, **dollar**, et *tick imbalance bars* (López de Prado). Comparer la normalité et l'autocorrélation des rendements selon le type de barre.
- **Triple barrier** : pour une entrée en $t_0$, barrière haute $m_{t_0}(1 + k_{\text{up}}\,\hat\sigma_{t_0})$, basse $m_{t_0}(1 - k_{\text{dn}}\,\hat\sigma_{t_0})$, verticale $t_0 + h$. Le label est la première barrière touchée. **Les barrières doivent inclure le coût aller-retour.**
- **Meta-labeling** optionnel : un modèle primaire donne le sens, un modèle secondaire décide de trader ou non.

## 5. Modèle de coûts — LE GATE

Coût aller-retour relatif :
$$c_{\text{taker}} = 2 f_{\text{taker}} + \tilde s + 2\,\text{slip}, \qquad c_{\text{maker}} = 2 f_{\text{maker}} + \text{adverse selection mesurée}$$

**Seuil de rentabilité** : il faut $\mathbb{E}[g] > c$, où $g$ est le rendement brut par trade.

**Taux de réussite minimal** avec gain moyen net $W$ et perte moyenne nette $L$ :
$$p^* = \frac{L}{W + L}$$

**Frottement annuel** :
$$\text{drag} = N_{\text{trades/an}} \times c$$

**Premier livrable (avant tout le reste) : `cost_gate.py`**, qui à partir des spreads et frais **mesurés** produit par paire et par tranche horaire :
1. la distribution de $c$ ;
2. le mouvement absolu médian du mid à horizons 5 s, 30 s, 1 min, 5 min, 15 min ;
3. le ratio $\text{mouvement}/c$ ;
4. le **plus petit horizon** où ce ratio dépasse 3.

Sous ce seuil, aucune stratégie n'est testée : le rapport le dit explicitement.

## 6. Sizing et contraintes de lot

- Quantité arrondie à `stepSize`, prix arrondi à `tickSize`, rejet si notional < `minNotional`.
- **Vol targeting** : $w_t = \min\!\left(\frac{\sigma_{\text{cible}}}{\hat\sigma_t},\, w_{\max}\right)$
- **Kelly** (information seulement, jamais appliqué plein) :
$$f^* = \frac{\mu}{\sigma^2} \;\text{(continu)}, \qquad f^* = p - \frac{1-p}{b} \;\text{(discret)}$$
Plafonner à $0{,}25\,f^*$ et à la limite de risque de la config.
- Pas de levier. Spot uniquement.

## 7. Validation statistique (rapport automatique)

**Sharpe annualisé** (marché 24/7) :
$$\widehat{SR}_{\text{an}} = \frac{\bar r}{\hat\sigma_r}\sqrt{N_{\text{périodes/an}}}$$
Appliquer l'ajustement de **Lo (2002)** si les rendements sont autocorrélés ; t-stats avec erreurs **Newey–West**.

**Probabilistic Sharpe Ratio** ($\hat\gamma_3$ = skewness, $\hat\gamma_4$ = kurtosis non centrée en excès, $n$ = nombre d'observations) :
$$\mathrm{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{n-1}}{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right)$$

**Deflated Sharpe Ratio** — $SR^*$ = Sharpe maximal attendu par hasard après $N$ essais, $\gamma \approx 0{,}5772$ (Euler–Mascheroni) :
$$SR^* = \sqrt{\mathbb{V}[\widehat{SR}_k]}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right], \qquad \mathrm{DSR} = \mathrm{PSR}(SR^*)$$

**Information Coefficient** par horizon $h$ :
$$\mathrm{IC}_h = \rho_{\text{Spearman}}\big(\text{signal}_t,\; r_{t \to t+h}\big)$$
Tracer la **décroissance de l'IC** avec $h$, et la comparer à l'horizon minimal donné par le cost gate.

**Test du taux de réussite** : test binomial de $H_0 : p = p^*$.

**Intervalles de confiance** : *stationary bootstrap* par blocs (pas de bootstrap i.i.d.).

**Validation croisée** : **purged K-fold avec embargo** (purge des observations dont le label chevauche le fold de test, embargo ≥ horizon du label), plus un walk-forward final sur une période jamais touchée.

**Critères d'acceptation d'une stratégie** (tous requis) :
- DSR > 0,95
- edge net médian par trade > 0 **après** coûts mesurés, IC bootstrap à 95 % strictement positif
- stabilité sur au moins 3 sous-périodes et 2 paires
- résultats paper trading dans l'intervalle de confiance du backtest

## 8. Backtester événementiel

- Rejeu des événements dans l'ordre des timestamps d'échange, avec **latence paramétrable** : une décision prise à $t$ n'utilise que des données $\le t - \ell_{\text{données}}$ et s'exécute à $t + \ell_{\text{ordre}}$.
- **Ordres taker** : exécution au meilleur prix opposé au moment $t + \ell$, plus slippage si la quantité dépasse $q$ au meilleur niveau.
- **Ordres maker** : exécution **seulement si le prix traverse** le niveau (pas s'il le touche), hypothèse de file d'attente conservatrice (position en queue).
- Frais et arrondis de lot appliqués à chaque exécution.
- **Test anti-fuite obligatoire** : décaler toutes les features de +1 événement vers le futur doit **améliorer** artificiellement le résultat. Si ce n'est pas le cas, le pipeline de features est suspect.

## 9. Risque et exploitation (paper trading)

- Perte max journalière, drawdown max par palier, nombre max de positions, taille max par ordre : tous en config.
- **Kill switch** : arrêt et mise à plat si une limite est atteinte ou si une donnée est périmée (fraîcheur > seuil) ⇒ on ne trade jamais sur des features figées.
- Journal complet : signal, features à l'instant de décision, ordre, exécution simulée, coût, PnL.
- **Mode live désactivé par défaut**, activable seulement via un flag explicite, avec des clés API sans droit de retrait.

## 10. Ordre de livraison

1. `cost_gate.py` + rapport sur 2 paires liquides (ex. BTCUSDT, ETHUSDT)
2. Collecteur live `aggTrade` + `bookTicker` → RAW (JSONL compressé zstd, append-only) → Bronze Parquet dédupliqué sur la clé naturelle `(source, symbol, ts, seq)`, idempotent (rejouer deux fois donne un hash identique)
3. Bibliothèque de features (§2–3) avec tests unitaires sur des cas construits à la main (valeurs attendues connues)
4. Labellisation (§4) + validation (§7)
5. Backtester événementiel (§8) avec le test anti-fuite
6. Paper trading (§9)

Ne passe pas à l'étape suivante tant que les tests de l'étape courante ne passent pas. À chaque étape, donne-moi : ce qui a été fait, les tests, la taille disque consommée, et les limites connues.

## 11. Protocole de vérification — UN FICHIER À LA FOIS

Cette règle prime sur tout le reste du prompt.

**Déroulé imposé :**
1. Avant d'écrire du code, propose l'**arborescence complète** du projet (fichiers + rôle d'une ligne chacun) et attends ma validation.
2. Ensuite, tu livres **un seul fichier source par tour**, accompagné de **son fichier de test** (`tests/test_<nom>.py`).
3. Tu lances les tests, tu montres la sortie réelle, tu remplis la checklist ci-dessous, puis **tu t'arrêtes et tu attends mon « validé »**.
4. Sans « validé » explicite de ma part, tu ne passes pas au fichier suivant.
5. Si un fichier suivant oblige à modifier un fichier déjà validé, tu le signales **avant** de le faire, tu montres le diff, et tu relances les tests des deux fichiers. Ce fichier repasse alors en « à revalider ».

**Rapport à fournir pour chaque fichier :**
- Rôle du fichier en 2 lignes et interface publique (fonctions / classes, signatures, types).
- Dépendances : fichiers internes importés, bibliothèques externes.
- Commande exacte pour lancer ses tests + sortie complète.
- Checklist générale et checklist spécifique, **chaque point coché ou justifié**, jamais sauté en silence.
- Hypothèses faites et limites connues.
- Nombre de lignes du fichier. Si > 300 lignes, propose de le découper.

**Checklist générale (tous les fichiers) :**
- [ ] Typage complet, docstrings avec unités (secondes, ms, prix, quantité, quote vs base)
- [ ] Aucun frais, seuil, chemin ou symbole codé en dur : tout vient de la config
- [ ] Tous les timestamps en UTC, en entiers (ms ou µs), unité explicite dans le nom de variable
- [ ] Pas de pandas sur les données tick
- [ ] Erreurs gérées explicitement, pas de `except:` nu, pas d'échec silencieux
- [ ] Déterministe : même entrée ⇒ même sortie (graine fixée si aléatoire)
- [ ] Tests sur au moins un cas nominal, un cas limite (données vides, une seule ligne, valeurs nulles) et un cas d'erreur

**Checklists spécifiques :**

*Collecteur / ingestion*
- [ ] Idempotence testée : ingérer deux fois le même lot donne un hash de sortie identique
- [ ] Reprise après crash testée : curseur persistant, ni trou ni doublon
- [ ] Écriture par lots, jamais ligne par ligne
- [ ] RAW jamais modifié ni supprimé
- [ ] Détection de trous : marqués, jamais interpolés

*Features (§2–3)*
- [ ] Chaque formule testée contre une valeur **calculée à la main** sur un petit jeu construit (3 à 10 lignes), valeur attendue écrite dans le test
- [ ] Causalité : la feature en $t$ n'utilise que des données $\le t$ (test qui modifie une donnée future et vérifie que la feature en $t$ ne change pas)
- [ ] Division par zéro gérée (profondeur nulle, volume nul, variance nulle)
- [ ] Signe des trades Binance vérifié : `m = true` ⇒ $d = -1$

*Cost gate / labels / validation (§4, §5, §7)*
- [ ] Les coûts sont bien inclus dans les barrières et dans le PnL
- [ ] PSR et DSR vérifiés contre un exemple numérique de référence
- [ ] La purge et l'embargo sont testés : aucune observation de train ne chevauche un label de test
- [ ] Le compteur d'essais $N$ est bien incrémenté et persisté

*Backtester / paper trading (§8–9)*
- [ ] Test anti-fuite (§8) présent et passant
- [ ] Ordre maker non exécuté si le prix touche seulement le niveau, exécuté s'il le traverse
- [ ] Arrondis `tickSize` / `stepSize` et rejet sous `minNotional` testés
- [ ] Kill switch testé : donnée périmée ⇒ arrêt ; limite de perte ⇒ arrêt
- [ ] Mode live impossible à activer sans le flag explicite

## À ne pas faire

- Pas d'interpolation des trous de données : on les marque et on exclut la période.
- Pas de rendements calculés sur le dernier trade pour les features de court terme.
- Pas d'optimisation de paramètres sans journaliser chaque essai.
- Pas de résultat présenté sans frais, ni sans comparaison au buy & hold sur la même période.
- Pas de pandas sur le tick, pas de ClickHouse, pas de levier.
