# ops — tâches automatiques (systemd utilisateur, sans sudo)

Ces unités tournent tant que WSL tourne. Elles n'écrivent que dans `$DATA_ROOT` et journalisent
dans `$DATA_ROOT/logs/`.

| Unité | Rôle |
|---|---|
| `qlab-exchange-info.timer` → `.service` | chaque heure : `exchange_info fetch --if-due` (nouveau snapshot si le dernier a plus de `snapshot_refresh_hours`) |
| `qlab-spreads.timer` → `.service` | chaque heure : `spreads sample` (un relevé `bookTicker` des paires EUR et tradées, dans `meta/spreads.jsonl`) |
| `qlab-cockpit.service` | permanent : cockpit local sur http://localhost:8765 (`live.cockpit`, visible de ce PC seulement), relancé en cas d'échec |
| `qlab-paper.timer` → `.service` | chaque jour à 00:05 UTC : `live.paper run` (une décision sur la barre close, journal `meta/paper/{book}.jsonl`) ; portefeuille ouvert une fois à la main par `start` |

## Installer / activer

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/intraday/ops/qlab-exchange-info.service ~/intraday/ops/qlab-exchange-info.timer ~/.config/systemd/user/
ln -sf ~/intraday/ops/qlab-spreads.service ~/intraday/ops/qlab-spreads.timer ~/.config/systemd/user/
ln -sf ~/intraday/ops/qlab-paper.service ~/intraday/ops/qlab-paper.timer ~/.config/systemd/user/
ln -sf ~/intraday/ops/qlab-cockpit.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qlab-exchange-info.timer qlab-spreads.timer qlab-paper.timer qlab-cockpit.service
```

## Alertes téléphone (healthchecks.io)

`hc-ping.sh VARIABLE UNITÉ`, appelé en `ExecStopPost` : signale le succès ou l'échec du passage à
healthchecks.io, avec les 30 dernières lignes de sortie en cas d'échec. L'adresse de ping est lue
dans `.env` (seule cette ligne) ; variable absente ⇒ rien n'est envoyé. Un passage absent à l'heure
prévue (PC éteint, WSL fermé) est signalé par healthchecks lui-même (check en mode Cron).

| Unité | Variable dans `.env` | Check healthchecks |
|---|---|---|
| `qlab-paper.service` | `HC_PAPER_URL` | Cron `5 0 * * *` UTC, grâce 3 h |

## Vérifier / désactiver

```bash
systemctl --user list-timers qlab-exchange-info.timer      # prochain passage
journalctl --user -u qlab-exchange-info.service -n 20      # sorties récentes
systemctl --user disable --now qlab-exchange-info.timer    # désactiver
```

## À faire plus tard

Signe de vie vers healthchecks.io → alerte téléphone si le PC décroche (voir `docs/ARBORESCENCE.md`).
