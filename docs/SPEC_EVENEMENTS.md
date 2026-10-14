# Spécification — volet « événements » : informations hors marché (phase 9)

Rédigée le 2026-10-14 (étape 48). **À valider avant tout code.** Elle reprend les quatre règles de la phase 9 (`docs/ARBORESCENCE.md`, décision du 2026-09-25) et les précise. Les règles communes (§7 validation, §11 un fichier par tour) restent celles de `SPEC_INTRADAY.md`.

## 0. Décisions

**Déjà prises (2026-09-25) :**
1. **Horodater à la réception** (`received_ms`), jamais seulement à la publication. Sinon, le backtest sait avant d'avoir pu apprendre.
2. **Pas de jugement d'une annonce passée par un modèle de langage.** Le modèle connaît déjà la suite.
3. **Sources gratuites d'abord.** L'API X est exclue (trop chère).
4. **Usage d'abord pour le risque** (ne pas être exposé au mauvais moment), ensuite seulement comme signal. Un signal passe par une fiche, un essai compté, et les critères habituels.

**Ajoutées le 2026-10-14 :**
5. Les événements servent aussi de **signaux au volet timing**. Contrats perpétuels, long et short à 1× : un événement ouvre une position, par exemple vendre à découvert un jeton dont Binance annonce le retrait.
6. Les essais « événements » comptent dans le **même N que le volet timing** : c'est la même recherche du meilleur moment pour acheter ou vendre. N = 105 au 2026-10-14.

## 1. Sources (essayées depuis le PC le 2026-10-14)

| Source | Accès | Contenu | Horodatage | Usage |
|---|---|---|---|---|
| Annonces Binance (`/bapi/composite/v1/public/cms/article/list/query`, `catalogId` 48 = cotations, 161 = retraits) | 200, public, **non documenté** | titres ; historique : 2 283 cotations, 441 retraits (au moins depuis 2022) | `releaseDate` en ms | risque + signal |
| État de la plateforme (`/sapi/v1/system/status`) | 200, public | maintenance oui / non | réception | risque |
| Calendrier du FOMC (federalreserve.gov) | 200, page HTML | 8 réunions par an, communiqué à 14:00 heure de New York | date et heure connues des mois à l'avance | risque + signal |
| Calendrier BLS (inflation CPI, emploi NFP) | **403** pour un robot | 12 + 12 publications par an, à 08:30 heure de New York | connues à l'avance | risque |
| Piratages (DefiLlama `/hacks`) | 200, public | date, montant, cible | **jour seulement** | risque, jamais en backtest (horodatage trop grossier, règle 1) |
| GDELT (presse mondiale) | 429 au-delà d'1 requête / 5 s | articles | publication | plus tard, hors première livraison |

**Calendriers programmés (FOMC, CPI, NFP) : un fichier versionné écrit à la main**, `config/events_calendar.yaml`, avec l'heure en UTC et l'URL de la source. Il y a une trentaine de dates par an. Le site du BLS refuse les robots, et la page de la Fed est du HTML qui change. Une commande compare le fichier à la page de la Fed et signale les écarts ; elle ne corrige rien toute seule.

## 2. Modèle d'un événement

Les enregistrements sont en JSONL, **ajout seulement** (RAW), sous `events/{calendar|news|flags}/date=YYYY-MM-DD/`.

| Champ | Sens |
|---|---|
| `source`, `source_id` | origine et identifiant chez la source ; la paire (`source`, `source_id`) sert au dédoublonnage |
| `kind` | `delisting`, `pair_removal`, `perp_delisting`, `listing`, `perp_launch`, `fomc`, `cpi`, `nfp`, `maintenance`, `hack`, `other` |
| `bases` | actifs concernés (`["EPIC", "QUICK"]`), vide si l'événement touche tout le marché |
| `published_ms` | heure annoncée par la source |
| `received_ms` | heure de réception par qlab ; **absente** pour un événement rechargé de l'historique (`backfilled: true`) |
| `title`, `url`, `sha256` | texte brut et empreinte (preuve que rien n'a été réécrit) |
| `rules_version` | version des règles de classement appliquées (§4) |

- **Horloge.** `received_ms` vient de l'horloge du PC, qui a déjà dérivé de +1,35 s. Le collecteur mesure l'écart avec l'heure serveur de Binance et refuse de tourner au-delà de `events.max_clock_offset_ms`, comme l'exige CLAUDE.md.
- **Taille.** Quelques Mo par an, négligeable dans le budget de 150 Go.

## 3. Historique et temps réel

- **En backtest**, un événement rechargé n'a pas d'heure de réception. Son heure effective est `published_ms + events.latency_margin_ms` (5 min proposées). L'entrée se fait **à l'ouverture de l'heure suivante** (bougies horaires du volet timing). C'est prudent : en réel, on peut entrer quelques minutes après l'annonce, pas 5 à 65 minutes après.
- **En paper**, on mesure la latence réelle, `received_ms − published_ms`. Si son p95 dépasse la marge, les résultats de backtest des fiches `ev_*` sont marqués « marge optimiste » dans les rapports.
- **Interrogation des annonces** toutes les 60 s (minuteur `qlab-events`), une requête par catalogue. Tout passe par `core/http.py`. Le point d'accès n'est pas documenté : une réponse de forme inattendue arrête la collecte avec une erreur claire. On ne devine jamais.

## 4. Classement (`events/classify.py`)

**Version 1 : des règles déterministes sur le titre**, versionnées (`rules_version`), testées sur une liste de vrais titres écrits à la main dans le test. Exemples relevés le 2026-10-14 :

| Titre | `kind` | `bases` |
|---|---|---|
| « Binance Will Delist EPIC, QUICK on 2026-10-28 » | `delisting` | EPIC, QUICK |
| « Notice of Removal of Spot Trading Pairs - 2026-10-02 » | `pair_removal` (paires seulement : l'actif reste) | lues dans le corps de l'article |
| « Binance Futures Will Delist Multiple USDⓈ-M Perpetual Contracts » | `perp_delisting` | lues dans le corps |
| « Binance Will List Hyperliquid (HYPE) with Seed Tag Applied » | `listing` | HYPE |
| « Binance Futures Will Launch … Perpetual Contract » | `perp_launch` | — |
| « Binance Alpha Will Remove … » | `other` (hors Binance principal, ignoré) | — |

- Un titre qu'aucune règle ne reconnaît devient `other`. Il est compté dans le rapport, pour que les règles s'améliorent sans jamais deviner.
- **Modèle de langage : pas en version 1.** S'il est ajouté un jour, il ne sert qu'en temps réel (paper), avec le modèle et le prompt versionnés et journalisés (règle 2). Il n'est jamais utilisé pour reclasser l'historique.

## 5. Usage A — risque (drapeaux)

`live/risk.py` accepte déjà des drapeaux globaux (`RiskFlag` : `suspend` ou `flatten`, actifs jusqu'à `until_ms`). Règles proposées (`config/events.yaml`) :

| Événement | Drapeau | Fenêtre |
|---|---|---|
| FOMC, CPI, NFP | `suspend` (aucune nouvelle position) | de 30 min avant à 2 h après l'heure de publication |
| maintenance Binance | `suspend` | tant qu'elle dure, puis 30 min |
| piratage d'une plateforme ou d'un pont > 100 M$ | `suspend` | 24 h à partir de la **réception** |

**Exclusion par actif** (pas un drapeau global) : un actif visé par un `delisting` ou un `perp_delisting` sort de l'univers dès la réception de l'annonce. On ne l'achète plus, et une position déjà ouverte est fermée au rééquilibrage suivant. Ça demande une petite modification du paper ou du futur `paper_timing.py`, qui sera signalée et revalidée.

Honnêtement, le paper long terme est peu concerné. Il se rééquilibre à 00:05 UTC, alors que le FOMC tombe à 18:00 ou 19:00 UTC et les chiffres américains à 12:30 ou 13:30 UTC. Les drapeaux servent surtout au futur paper timing.

## 6. Usage B — signaux (fiches `ev_*`, volet timing)

Moteur prévu, `timing/event_book.py` :
- chaque événement ouvre une ligne de poids `1 / events.max_lines`, à l'heure effective du §3, pour `horizon_h` heures ;
- si toutes les lignes sont prises, l'événement est ignoré : premier arrivé, premier servi ;
- mêmes coûts, financement, liquidation à 1× et sortie au dernier prix (contrat retiré) que `timing/portfolio.py`.

Fiches prévues, écrites et figées **avant** tout test (§7) :

| Fiche | Mécanisme | Grille |
|---|---|---|
| `ev_delisting_short` | Le retrait du spot coupe la liquidité et force les ventes : vendre le perpétuel de l'actif après l'annonce | `horizon_h` ∈ {24, 72, 168} |
| `ev_listing_long` | La cotation au spot d'un actif qui a déjà un perpétuel attire une nouvelle demande : achat après l'annonce | `horizon_h` ∈ {4, 24, 72} |
| `ev_pre_fomc` | Dérive haussière avant le communiqué de la Fed, documentée sur les actions (Lucca et Moench, 2015) : acheter BTC h heures avant, sortir à l'heure du communiqué | `horizon_h` ∈ {12, 24} |

Cela fait 8 essais, et N passe de 105 à 113, sous le budget de 120.

**Puissance statistique.** Le nombre d'événements avant le hold-out est affiché avec chaque essai. En dessous de **30 événements**, l'essai est marqué « puissance insuffisante » et n'est pas jugé ; c'est probablement le cas d'`ev_pre_fomc`, avec une quarantaine de réunions. Le MinTRL le dira aussi.

**Critères inchangés** : DSR sur le N du volet, écart de Sharpe avec le buy & hold BTC, années civiles, deux moitiés disjointes des actifs touchés, MinTRL. Le hold-out (depuis 2025-10-01) reste unique pour tout le volet timing.

## 7. Fichiers et ordre de livraison

| # | Fichier | Contenu |
|---|---|---|
| 48 | `docs/SPEC_EVENEMENTS.md` | cette spécification |
| 49 | `events/calendar.py` + `config/events_calendar.yaml` + `config/events.yaml` | dates FOMC, CPI et NFP en UTC (2020 → 2026), fenêtres de risque, comparaison avec la page de la Fed |
| 50 | `events/news.py` + `ops/qlab-events.{service,timer}` | collecte des annonces Binance (temps réel + recharge de l'historique), RAW, contrôle d'horloge, `core/paths.py` : `events_*` |
| 51 | `events/classify.py` | règles versionnées titre → `kind`, `bases` ; drapeaux et exclusions |
| 51 bis | `timing/event_book.py` + fiches `hypotheses/timing/ev_*.yaml` + `timing/scan.py` | portefeuille piloté par événements, balayage |
| 51 ter | `live/timing_state.py` / cockpit | événements récents et drapeaux actifs dans le cockpit |

## 8. Hors champ

X / Twitter, sentiment calculé par un modèle sur l'historique, GDELT (plus tard, si une fiche le justifie), données on-chain, ordres réels.

## 9. À trancher par le propriétaire

1. Accepter le point d'accès **non documenté** des annonces Binance : lecture publique seulement, sans clé, mais sa forme peut changer.
2. Tenir les calendriers FOMC, CPI et NFP **à la main** dans un fichier versionné : le site du BLS bloque les robots.
3. Marge de latence en backtest de **5 min**, puis entrée à l'heure pleine suivante.
4. Les essais `ev_*` comptent dans le N du volet timing (recommandé), et non dans un volet séparé qui repartirait de zéro.
