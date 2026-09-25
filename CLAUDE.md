# Consignes pour Claude — projet qlab

## Règles du propriétaire (prioritaires)

- La machine d'exécution est un **PC de test** : Windows + WSL2 (Ubuntu). Tu as tous les droits sur ce PC (installation de paquets, configuration, accès réseau, API d'échange) **sauf** :
  - **demander avant de supprimer des données** (fichiers, dossiers, tables, archives, RAW compris) ;
  - **demander avant de désinstaller ou supprimer une application**.
- Tu gères le stockage sur ce PC : dossier des données, budget de ~150 Go, suivi de l'espace disque.
- Les données vont sur le système de fichiers Linux de WSL2, **jamais sous `/mnt/c`**. La racine est fixée dans `config/base.yaml`.
- Clés API : aucune n'est nécessaire pour la recherche ni pour le paper trading. Si un jour il en faut, elles doivent être **sans droit de retrait**, rangées dans un `.env` hors dépôt (dans `.gitignore`), et ne sont jamais affichées ni commitées.

## Protocole de travail

- Spécifications : `docs/SPEC_INTRADAY.md` et `docs/SPEC_LONG_TERME.md`.
- Arborescence et statut de chaque fichier : `docs/ARBORESCENCE.md`. Mets le statut à jour à chaque livraison.
- Règle du §11 : **un fichier source + son test par tour**. Lance les tests, montre la sortie réelle, remplis la checklist, puis attends « validé » avant le fichier suivant.
- Données réelles uniquement pour les rapports. Les tests unitaires utilisent de petits jeux construits à la main, avec les valeurs attendues écrites dans le test.
