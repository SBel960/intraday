"""Mécanique commune des commandes ``python -m qlab.<module> --config config ...``.

Chaque module exécutable décrit seulement ses arguments et son action ; ``run_command`` fait
le reste, toujours de la même façon :

1. analyse ``--config`` et les arguments propres au module ;
2. charge et valide la configuration (erreur ⇒ code 1, sans journal : il n'existe pas encore) ;
3. ouvre le journal ``$DATA_ROOT/logs/{component}/`` et y trace le début de la commande ;
4. exécute l'action ; toute erreur est journalisée et convertie en code de sortie (``run_cli``).

Exception : ``core/config.py`` garde sa propre commande (il ne peut pas importer ce module).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qlab.core.config import QlabConfig, load_config
from qlab.core.errors import run_cli
from qlab.core.jsonlog import JsonLog
from qlab.core.paths import DataPaths


@dataclass(frozen=True, slots=True)
class Context:
    """Ce dont une action a besoin : config validée, chemins, journal, arguments analysés."""

    config: QlabConfig
    paths: DataPaths
    journal: JsonLog
    args: argparse.Namespace

    def notify(self, message: str) -> None:
        """Message d'attente (réseau, limite de requêtes) : affiché et journalisé."""
        print(message, file=sys.stderr, flush=True)
        self.journal.warning("fetch.retry", {"message": message})


def run_command(
    argv: Sequence[str] | None,
    *,
    prog: str,
    description: str,
    component: str,
    add_arguments: Callable[[argparse.ArgumentParser], None],
    action: Callable[[Context], int],
) -> int:
    """Analyse ``argv``, prépare le contexte et exécute ``action`` ; renvoie le code de sortie."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--config", type=Path, required=True, help="dossier des YAML")
    add_arguments(parser)
    args = parser.parse_args(argv)

    def entry() -> int:
        config = load_config(args.config)
        paths = DataPaths(config.base.data.root)
        with JsonLog(paths.logs, component) as journal:
            command = {k: v for k, v in vars(args).items() if k != "config"}
            journal.info("run.start", {k: str(v) for k, v in command.items()})
            return run_cli(lambda: action(Context(config, paths, journal, args)), journal=journal)

    return run_cli(entry)
