"""Exceptions du projet et point d'entrée commun des modules exécutables.

Deux familles d'erreurs :

- ``QlabError`` et ses sous-classes : erreurs *attendues* (config fausse, données invalides,
  réponse d'échange inutilisable). Elles portent un message qui suffit à corriger le problème.
- Toute autre exception : un *bug*. On affiche la trace complète.

Les erreurs d'environnement (``OSError`` : disque plein, droits, fichier absent) sont aussi des
erreurs attendues : le code n'y peut rien, le message suffit à corriger.

``run_cli`` applique cette distinction aux ``main()`` : code de sortie 0 si tout va bien,
1 pour une ``QlabError`` ou une ``OSError``, 2 pour un bug. Chaque échec est écrit sur stderr
et, si un journal est fourni, dans le journal JSONL.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qlab.core.jsonlog import JsonLog

EXIT_OK = 0
EXIT_EXPECTED_ERROR = 1
EXIT_BUG = 2


class QlabError(Exception):
    """Erreur attendue du projet : entrée invalide, pas un défaut du code."""


class DataError(QlabError):
    """Données invalides : trou, unité incohérente, fichier corrompu, schéma inattendu."""


class ExchangeError(QlabError):
    """Réponse d'échange inutilisable : statut HTTP, limite de requêtes, format inattendu."""


def run_cli(entry: Callable[[], int], *, journal: JsonLog | None = None) -> int:
    """Exécute ``entry`` et convertit l'issue en code de sortie (0, 1 ou 2).

    ``entry`` renvoie son propre code en cas de succès. Une ``QlabError`` ou une ``OSError``
    donne 1 avec un message d'une ligne ; toute autre exception donne 2 avec la trace complète.
    Les ``BaseException`` hors ``Exception`` (Ctrl-C, ``SystemExit``) ne sont pas interceptées.
    """
    try:
        return entry()
    except QlabError as exc:
        name = type(exc).__name__
        print(f"ERREUR ({name}) : {exc}", file=sys.stderr)
        if journal is not None:
            journal.error("run.failed", {"error": name, "message": str(exc), "bug": False})
        return EXIT_EXPECTED_ERROR
    except OSError as exc:
        name = type(exc).__name__
        print(f"ERREUR SYSTÈME ({name}) : {exc}", file=sys.stderr)
        if journal is not None:
            journal.error(
                "run.failed", {"error": name, "message": str(exc), "bug": False, "system": True}
            )
        return EXIT_EXPECTED_ERROR
    except Exception as exc:  # noqa: BLE001 — frontière du processus : bug signalé, pas masqué
        trace = traceback.format_exc()
        print(f"BUG ({type(exc).__name__}) :\n{trace}", file=sys.stderr)
        if journal is not None:
            journal.error(
                "run.failed",
                {"error": type(exc).__name__, "message": str(exc), "bug": True, "trace": trace},
            )
        return EXIT_BUG
