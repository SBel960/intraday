"""Exceptions du projet et point d'entrée commun des modules exécutables.

Deux familles d'erreurs :

- ``QlabError`` et ses sous-classes : erreurs *attendues* (config fausse, données invalides,
  réponse d'échange inutilisable). Elles portent un message qui suffit à corriger le problème.
- Toute autre exception : un *bug*. On affiche la trace complète.

Les erreurs d'environnement (``OSError`` : disque plein, droits, fichier absent) sont aussi des
erreurs attendues : le code n'y peut rien, le message suffit à corriger.

``run_cli`` applique cette distinction aux ``main()`` : code de sortie 0 si tout va bien,
1 pour une ``QlabError`` ou une ``OSError``, 2 pour un bug, 130 pour Ctrl-C. Chaque échec
est écrit sur stderr et, si un journal est fourni, dans le journal JSONL.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class Journal(Protocol):
    """Ce dont ``run_cli`` a besoin pour tracer un échec (``core/jsonlog.JsonLog`` convient).

    Contrat plutôt qu'import : le module le plus bas du socle ne dépend pas du journal.
    """

    def error(self, kind: str, data: Mapping[str, Any] | None = None) -> None: ...
    def warning(self, kind: str, data: Mapping[str, Any] | None = None) -> None: ...


EXIT_OK = 0
EXIT_EXPECTED_ERROR = 1
EXIT_BUG = 2
EXIT_INTERRUPTED = 130  # convention Unix : 128 + SIGINT


class QlabError(Exception):
    """Erreur attendue du projet : entrée invalide, pas un défaut du code."""


class DataError(QlabError):
    """Données invalides : trou, unité incohérente, fichier corrompu, schéma inattendu."""


class ExchangeError(QlabError):
    """Réponse d'échange inutilisable : statut HTTP, limite de requêtes, format inattendu."""


def run_cli(entry: Callable[[], int], *, journal: Journal | None = None) -> int:
    """Exécute ``entry`` et convertit l'issue en code de sortie (0, 1 ou 2).

    ``entry`` renvoie son propre code en cas de succès. Une ``QlabError`` ou une ``OSError``
    donne 1 avec un message d'une ligne ; toute autre exception donne 2 avec la trace complète.
    Ctrl-C donne 130 avec « Interrompu » (sans trace) ; ``SystemExit`` n'est pas intercepté.
    """
    try:
        return entry()
    except KeyboardInterrupt:
        print("Interrompu (Ctrl-C).", file=sys.stderr)
        if journal is not None:
            journal.warning("run.interrupted")
        return EXIT_INTERRUPTED
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
