# Fiches d'événements en attente de puissance

`tm_delisting_short` et `tm_listing_long` ont été écrites et figées le 2026-10-14, avant tout test. Elles exigent au moins 30 annonces exploitables avant le hold-out (2025-10-01) : contrat perpétuel existant et liquide à l'annonce.

Les comptes faits le 2026-10-14, sur les annonces seulement et sans regarder un seul rendement :

| Seuil de liquidité (volume quotidien médian) | Retraits avant hold-out | Cotations avant hold-out |
|---|---|---|
| 20 M USDT (univers du volet timing) | 2 | 7 |
| 5 M USDT | 20 | 7 |
| 1 M USDT | 24 | 8 |

**La puissance est insuffisante.** Selon leur propre règle d'abandon, ces fiches ne sont pas lancées. Elles sont rangées hors de `hypotheses/timing` pour qu'aucun balayage ne les exécute. Elles ne comptent donc pas dans le N du volet.

**Pour avancer honnêtement :**
- un **paper en temps réel** sur les nouvelles annonces, relevées par `events.news poll`, qui accumule des événements vraiment hors échantillon ;
- ou, décision du propriétaire, consacrer le hold-out du volet timing à ces fiches. Le hold-out ne se rejoue qu'une fois.
