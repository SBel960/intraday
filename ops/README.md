# ops — tâches automatiques (systemd utilisateur, sans sudo)

Ces unités tournent tant que WSL tourne. Elles n'écrivent que dans `$DATA_ROOT` et journalisent
dans `$DATA_ROOT/logs/`.

| Unité | Rôle |
|---|---|
| `qlab-exchange-info.timer` → `.service` | chaque heure : `exchange_info fetch --if-due` (nouveau snapshot si le dernier a plus de `snapshot_refresh_hours`) |

## Installer / activer

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/intraday/ops/qlab-exchange-info.service ~/intraday/ops/qlab-exchange-info.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qlab-exchange-info.timer
```

## Vérifier / désactiver

```bash
systemctl --user list-timers qlab-exchange-info.timer      # prochain passage
journalctl --user -u qlab-exchange-info.service -n 20      # sorties récentes
systemctl --user disable --now qlab-exchange-info.timer    # désactiver
```

## À faire plus tard

Signe de vie vers healthchecks.io → alerte téléphone si le PC décroche (voir `docs/ARBORESCENCE.md`).
