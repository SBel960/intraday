#!/bin/sh
# Signal à healthchecks.io après le passage d'un service (ExecStopPost) : succès, ou échec avec
# les dernières lignes de sortie (motif visible dans la notification).
# Usage : hc-ping.sh NOM_DE_VARIABLE UNITÉ ; l'adresse est lue dans .env (seule cette ligne :
# le service n'a jamais accès aux autres secrets). Variable absente : rien n'est envoyé.
var="$1"; unit="$2"
env_file="$(dirname "$0")/../.env"
url=$(grep -m1 "^${var}=" "$env_file" 2>/dev/null | cut -d= -f2-)
[ -n "$url" ] || exit 0
if [ "${SERVICE_RESULT:-success}" = "success" ] && [ "${EXIT_STATUS:-0}" = "0" ]; then
  target="$url"
else
  target="$url/fail"
fi
journalctl --user -u "$unit" -n 30 --no-pager -o cat 2>/dev/null | tail -c 10000 |
  curl -fsS -m 10 --retry 5 -o /dev/null --data-binary @- "$target" || true
