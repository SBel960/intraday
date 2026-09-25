# qlab — recherche intraday et long terme (crypto spot, petit capital)

Recherche et paper trading sur Binance spot, en **EUR** (compte dans l'EEE), à partir de 50 €.
Principe : le coût d'abord. Aucune stratégie n'est testée tant qu'elle ne bat pas ses frais.

- Spécification intraday : [`docs/SPEC_INTRADAY.md`](docs/SPEC_INTRADAY.md)
- Addendum long terme : [`docs/SPEC_LONG_TERME.md`](docs/SPEC_LONG_TERME.md)
- Arborescence, ordre de livraison et statut de chaque fichier : [`docs/ARBORESCENCE.md`](docs/ARBORESCENCE.md)
- Consignes de travail : [`CLAUDE.md`](CLAUDE.md)

## État d'avancement

Phase 0 (socle) en cours. Le volet **long terme passe avant l'intraday** : l'estimation
préliminaire du gate montre qu'à 0,1 % de frais par jambe, aucun horizon ≤ 4 h ne franchit le
ratio mouvement / coût de 3 (détails dans `docs/ARBORESCENCE.md`).

## Installation (Windows + WSL2 Ubuntu)

Les données vont sur le système de fichiers Linux de WSL2, **jamais sous `/mnt/c`**
(la config le refuse). Racine : `data.root` dans `config/base.yaml`.

```bash
uv venv ~/.venvs/qlab --python 3.12
VIRTUAL_ENV=~/.venvs/qlab uv pip install -e ".[dev]"
source ~/.venvs/qlab/bin/activate
```

Horloge : l'horloge de WSL suit celle de Windows. Si elle dérive, dans un PowerShell
administrateur : `w32tm /resync /force` (ou `net start w32time` d'abord).

## Commandes

Toutes sortent avec le code 0 (succès), 1 (erreur attendue : config, saisie, données, réseau, disque), 2 (bug) ou 130 (Ctrl-C).

```bash
# Vérifier la configuration et l'afficher
python -m qlab.core.config --config config

# Registre des apports / retraits (fixe le palier de capital)
python -m qlab.core.ledger --config config deposit 50 --note "départ"
python -m qlab.core.ledger --config config withdrawal 20 --date 2026-10-01
python -m qlab.core.ledger --config config show

# Règles de toutes les paires Binance (snapshot versionné, horloge, alertes)
python -m qlab.exchange.exchange_info --config config fetch
python -m qlab.exchange.exchange_info --config config show --symbol BTCEUR
```

Toutes les commandes journalisent dans `$DATA_ROOT/logs/{commande}/AAAA-MM-JJ.jsonl`.

## Tâches automatiques

Un minuteur systemd utilisateur (sans sudo) vérifie chaque heure si un nouveau snapshot
`exchangeInfo` est dû : voir [`ops/README.md`](ops/README.md).

## Contrôles

```bash
ruff check . && ruff format --check . && mypy && pytest -q
```

Les mêmes contrôles tournent sur GitHub à chaque push (`.github/workflows/ci.yml`).
