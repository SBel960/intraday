"""Fichiers YAML → dataclasses figées, validées **strictement**.

Utilisé par la configuration (``core/config.py``) et les fiches d'hypothèse
(``research/hypothesis.py``) : une seule façon de lire un YAML dans le projet.

Règles : aucune valeur par défaut ; clé manquante ou inconnue, type incorrect (un booléen n'est
pas un entier), nombre non fini, chaîne vide ⇒ erreur avec le chemin de la clé
(``base.data.root``). Les validateurs des dataclasses (``__post_init__``) lèvent une
``SchemaError`` (ou une sous-classe) : elle est relancée préfixée de son chemin, même classe.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from qlab.core.errors import QlabError


class SchemaError(QlabError, ValueError):
    """Fichier YAML absent, illisible, incomplet, mal typé ou hors domaine."""


def _scalar(tp: type, value: object, path: str, error: type[SchemaError]) -> object:
    """Valeur scalaire stricte : un booléen n'est pas un entier, un entier devient un float."""
    ok = {
        bool: isinstance(value, bool),
        int: isinstance(value, int) and not isinstance(value, bool),
        float: isinstance(value, int | float) and not isinstance(value, bool),
        str: isinstance(value, str),
        Path: isinstance(value, str),
    }
    if tp not in ok:
        raise TypeError(f"{path} : type d'annotation non géré {tp!r}")
    if not ok[tp]:
        raise error(f"{path} : {tp.__name__} attendu, reçu {type(value).__name__} ({value!r})")
    if tp is float:
        number = float(value)  # type: ignore[arg-type]  # vérifié ci-dessus
        if not math.isfinite(number):
            raise error(f"{path} : nombre fini attendu, reçu {value}")
        return number
    if isinstance(value, str) and not value.strip():
        raise error(f"{path} : chaîne vide")
    return Path(value).expanduser() if tp is Path else value  # type: ignore[arg-type]


def _convert(tp: Any, value: object, path: str, error: type[SchemaError]) -> object:
    """Liste → tuple, table → dataclass, sinon scalaire strict."""
    if get_origin(tp) is tuple:
        if not isinstance(value, list):
            raise error(f"{path} : liste attendue, reçu {type(value).__name__}")
        item_tp = get_args(tp)[0]
        return tuple(_convert(item_tp, v, f"{path}[{i}]", error) for i, v in enumerate(value))
    if isinstance(tp, type) and is_dataclass(tp):
        return build(tp, value, path, error=error)
    return _scalar(tp, value, path, error)


def build[T](cls: type[T], raw: object, path: str, *, error: type[SchemaError]) -> T:
    """Construit la dataclass ``cls`` depuis une table YAML ; erreurs de la classe ``error``."""
    if not isinstance(raw, dict):
        raise error(f"{path} : table attendue, reçu {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    keys = {str(k) for k in raw}
    missing = [n for n in names if n not in keys]
    unknown = sorted(keys - set(names))
    if missing:
        raise error(f"{path} : clé(s) manquante(s) {missing}")
    if unknown:
        raise error(f"{path} : clé(s) inconnue(s) {unknown}")
    kwargs = {n: _convert(hints[n], raw[n], f"{path}.{n}", error) for n in names}
    try:
        return cls(**kwargs)
    except SchemaError as exc:
        raise type(exc)(f"{path} : {exc}") from exc


def read_yaml(path: Path, *, error: type[SchemaError]) -> object:
    """Contenu YAML de ``path`` (chargeur sûr : aucune exécution de code)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"lecture impossible de {path} : {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise error(f"YAML invalide dans {path} : {exc}") from exc
