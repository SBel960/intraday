"""Fiches d'hypothèse : l'hypothèse est écrite **avant** tout test, puis figée par son empreinte.

Aucune stratégie n'est testée sans fiche (SPEC_INTRADAY, « Discipline obligatoire »). Une fiche
YAML (``hypotheses/{id}.yaml``, modèle : ``hypotheses/_template.yaml``) déclare :

- le **mécanisme économique** supposé (pourquoi ça marcherait) ;
- l'horizon de détention (``horizon_s``), l'univers, les données utilisées ;
- la **grille des paramètres candidats** : chaque combinaison essayée est un essai compté pour
  le Deflated Sharpe ; la fiche fixe donc d'avance le nombre d'essais (``n_trials``) ;
- le coût aller-retour estimé, l'edge minimal requis, le **critère d'abandon**.

``fingerprint`` : empreinte SHA-256 du contenu **validé** (pas des octets : un commentaire ou un
espace ne la change pas). Le registre d'essais la garde : une fiche modifiée après coup se voit.
Lecture stricte par ``core/yamlschema`` (clé manquante ou inconnue ⇒ ``HypothesisError``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from qlab.core.cli import Context, run_command
from qlab.core.timeutils import date_to_ms
from qlab.core.yamlschema import SchemaError, build, read_yaml

VOLETS = ("longterm", "intraday")
UNIVERSES = ("trade", "observe")
EDGE_BASES = ("per_trade", "per_year")
MIN_MECHANISM_CHARS = 40  # une vraie phrase, pas un mot-clé
_ID_RE = re.compile(r"^[a-z0-9_]+$")


class HypothesisError(SchemaError):
    """Fiche d'hypothèse absente, incomplète ou incohérente."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise HypothesisError(msg)


@dataclass(frozen=True, slots=True)
class Parameter:
    """Un paramètre et ses valeurs candidates (toutes seront essayées)."""

    name: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _check(bool(_ID_RE.match(self.name)), f"nom de paramètre invalide : {self.name!r}")
        _check(len(self.values) > 0, f"{self.name} : au moins une valeur")
        _check(len(set(self.values)) == len(self.values), f"{self.name} : valeurs en double")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    title: str
    volet: str
    created: str
    mechanism: str
    horizon_s: int
    universe: str
    features: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    cost_round_trip_frac: float
    min_edge_frac: float
    edge_basis: str
    abandon: str

    def __post_init__(self) -> None:
        _check(bool(_ID_RE.match(self.id)), f"id invalide (attendu [a-z0-9_]+) : {self.id!r}")
        _check(self.volet in VOLETS, f"volet doit être {VOLETS} (reçu {self.volet!r})")
        try:
            date_to_ms(self.created)
        except ValueError as exc:
            raise HypothesisError(f"created : {exc}") from exc
        _check(
            len(self.mechanism.strip()) >= MIN_MECHANISM_CHARS,
            f"mechanism : expliquer le mécanisme en une phrase (≥ {MIN_MECHANISM_CHARS} car.)",
        )
        _check(self.horizon_s > 0, "horizon_s doit être > 0")
        _check(self.universe in UNIVERSES, f"universe doit être {UNIVERSES}")
        _check(len(self.features) > 0, "features : au moins une donnée utilisée")
        names = [p.name for p in self.parameters]
        _check(len(set(names)) == len(names), f"parameters : noms en double {names}")
        _check(0 <= self.cost_round_trip_frac < 1, "cost_round_trip_frac doit être dans [0, 1[")
        _check(0 < self.min_edge_frac < 1, "min_edge_frac doit être dans ]0, 1[")
        _check(self.edge_basis in EDGE_BASES, f"edge_basis doit être {EDGE_BASES}")
        if self.edge_basis == "per_trade":
            _check(
                self.min_edge_frac > self.cost_round_trip_frac,
                "min_edge_frac (par trade) doit dépasser le coût aller-retour estimé",
            )

    @property
    def n_trials(self) -> int:
        """Nombre de combinaisons de paramètres = essais déclarés d'avance."""
        return math.prod(len(p.values) for p in self.parameters)

    def grid(self) -> Iterator[dict[str, float]]:
        """Combinaisons de paramètres, dans un ordre déterministe."""
        names = [p.name for p in self.parameters]
        for combo in itertools.product(*(p.values for p in self.parameters)):
            yield dict(zip(names, combo, strict=True))

    @property
    def fingerprint(self) -> str:
        """SHA-256 du contenu validé (indépendant de la mise en forme du YAML)."""
        text = json.dumps(dataclasses.asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hypothesis(path: Path) -> Hypothesis:
    """Lit et valide une fiche ; son ``id`` doit être le nom du fichier (sans ``.yaml``)."""
    raw = read_yaml(path, error=HypothesisError)
    hyp = build(Hypothesis, raw, f"hypothèse {path.name}", error=HypothesisError)
    _check(hyp.id == path.stem, f"{path.name} : id {hyp.id!r} ≠ nom du fichier {path.stem!r}")
    return hyp


def load_all(directory: Path) -> list[Hypothesis]:
    """Toutes les fiches d'un dossier (``*.yaml``, sauf celles qui commencent par ``_``)."""
    return [
        load_hypothesis(p) for p in sorted(directory.glob("*.yaml")) if not p.name.startswith("_")
    ]


# --- commande ----------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["check"])
    parser.add_argument("files", nargs="+", type=Path, help="fiches YAML à valider")


def _action(ctx: Context) -> int:
    for path in ctx.args.files:
        h = load_hypothesis(path)
        print(f"{h.id:<28} {h.volet:<9} {h.n_trials:>5} essai(s)  empreinte {h.fingerprint[:12]}")
        ctx.journal.info(
            "hypothesis.checked", {"id": h.id, "fingerprint": h.fingerprint, "n_trials": h.n_trials}
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.research.hypothesis",
        description="Valide des fiches d'hypothèse",
        component="hypothesis",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
