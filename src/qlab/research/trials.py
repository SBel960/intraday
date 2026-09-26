"""Registre des essais : chaque test d'une combinaison de paramètres est compté, pour toujours.

Le Deflated Sharpe compare le meilleur Sharpe obtenu au meilleur Sharpe **attendu par hasard**
après N essais : N doit donc compter *tout* ce qui a été essayé, succès comme échecs. Le registre
(``DataPaths.trials``, append-only via ``core/records``) impose :

- un essai porte sur une **combinaison déclarée** dans la fiche d'hypothèse (``Hypothesis.grid``) :
  impossible d'essayer « un paramètre de plus » sans l'avoir écrit avant ;
- la fiche n'a **pas changé** depuis ses premiers essais (même empreinte) ; sinon refus : il faut
  une nouvelle fiche (nouvel id), dont les essais s'ajoutent ;
- ``n_trials(volet)`` : nombre de combinaisons **distinctes** essayées dans tout le volet (relancer
  la même combinaison, ex. sur des données corrigées, ne crée pas un nouvel essai ; la dernière
  mesure fait foi).

Chaque essai garde les statistiques nécessaires au PSR / DSR : Sharpe **par période** (non
annualisé), nombre d'observations, asymétrie, kurtosis (de Pearson : 3 pour une loi normale).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.records import Record, append_record, read_records
from qlab.core.timeutils import ms_to_iso
from qlab.research.hypothesis import VOLETS, Hypothesis

_FIELDS = frozenset(
    {
        "ts_ms",
        "volet",
        "hypothesis_id",
        "fingerprint",
        "params",
        "sharpe",
        "n_obs",
        "skew",
        "kurtosis",
        "periods_per_year",
        "note",
    }
)


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Statistiques d'un essai. ``sharpe`` : moyenne / écart-type des rendements par période."""

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float
    periods_per_year: int

    def __post_init__(self) -> None:
        for name in ("sharpe", "skew", "kurtosis"):
            if not math.isfinite(getattr(self, name)):
                raise DataError(f"essai : {name} non fini")
        if self.n_obs < 2:
            raise DataError("essai : au moins 2 observations")
        if self.kurtosis < 1:  # toute distribution a une kurtosis de Pearson ≥ 1
            raise DataError(f"essai : kurtosis de Pearson ≥ 1 attendue (reçu {self.kurtosis})")
        if self.periods_per_year <= 0:
            raise DataError("essai : periods_per_year doit être > 0")

    @property
    def annual_sharpe(self) -> float:
        return self.sharpe * math.sqrt(self.periods_per_year)


@dataclass(frozen=True, slots=True)
class Trial:
    ts_ms: int
    volet: str
    hypothesis_id: str
    fingerprint: str
    params: tuple[tuple[str, float], ...]  # trié par nom : clé stable
    result: TrialResult
    note: str

    @property
    def key(self) -> tuple[str, str, tuple[tuple[str, float], ...]]:
        """Identité d'un essai : volet, hypothèse, combinaison de paramètres."""
        return self.volet, self.hypothesis_id, self.params

    def to_record(self) -> Record:
        r = self.result
        return {
            "ts_ms": self.ts_ms,
            "volet": self.volet,
            "hypothesis_id": self.hypothesis_id,
            "fingerprint": self.fingerprint,
            "params": dict(self.params),
            "sharpe": r.sharpe,
            "n_obs": r.n_obs,
            "skew": r.skew,
            "kurtosis": r.kurtosis,
            "periods_per_year": r.periods_per_year,
            "note": self.note,
        }


def _params(raw: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(k), float(v)) for k, v in raw.items()))


def _int(obj: Record, key: str) -> int:
    v = obj[key]
    if not isinstance(v, int) or isinstance(v, bool):
        raise DataError(f"{key} : entier attendu, reçu {v!r}")
    return v


def _num(obj: Record, key: str) -> float:
    v = obj[key]
    if not isinstance(v, int | float) or isinstance(v, bool):
        raise DataError(f"{key} : nombre attendu, reçu {v!r}")
    return float(v)


def _parse(obj: Record, n: int) -> Trial:
    """Lecture stricte d'un essai : types exacts, sinon ``DataError`` avec la ligne."""
    try:
        if set(obj) != _FIELDS or not isinstance(obj["params"], dict):
            raise DataError(f"champs attendus {sorted(_FIELDS)}")
        volet, hyp_id, fingerprint, note = (
            obj[k] for k in ("volet", "hypothesis_id", "fingerprint", "note")
        )
        if not all(isinstance(t, str) for t in (volet, hyp_id, fingerprint, note)):
            raise DataError("volet, hypothesis_id, fingerprint et note : texte attendu")
        params = {str(k): _num(obj["params"], k) for k in obj["params"]}
        result = TrialResult(
            _num(obj, "sharpe"),
            _int(obj, "n_obs"),
            _num(obj, "skew"),
            _num(obj, "kurtosis"),
            _int(obj, "periods_per_year"),
        )
        return Trial(_int(obj, "ts_ms"), volet, hyp_id, fingerprint, _params(params), result, note)
    except DataError as exc:
        raise DataError(f"essai, ligne {n} : {exc}") from exc


class TrialRegistry:
    """Registre des essais stocké dans ``path`` (``DataPaths.trials``)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def trials(self, volet: str | None = None) -> tuple[Trial, ...]:
        """Essais enregistrés (tous, ou ceux d'un volet), dans l'ordre du registre."""
        all_trials = [_parse(obj, n) for n, obj in enumerate(read_records(self.path), start=1)]
        return tuple(t for t in all_trials if volet is None or t.volet == volet)

    def latest(self, volet: str) -> dict[tuple[str, str, tuple[tuple[str, float], ...]], Trial]:
        """Dernière mesure de chaque essai distinct du volet."""
        return {t.key: t for t in self.trials(volet)}

    def n_trials(self, volet: str) -> int:
        """N du Deflated Sharpe : combinaisons distinctes essayées dans tout le volet."""
        return len(self.latest(volet))

    def sharpes(self, volet: str) -> list[float]:
        """Sharpe (par période) de chaque essai distinct : sert à estimer V[SR] du DSR."""
        return [t.result.sharpe for t in self.latest(volet).values()]

    def record(
        self,
        hypothesis: Hypothesis,
        params: Mapping[str, float],
        result: TrialResult,
        *,
        ts_ms: int,
        note: str = "",
    ) -> Trial:
        """Enregistre un essai après contrôle : combinaison déclarée, fiche inchangée."""
        combo = _params(params)
        declared = {_params(g) for g in hypothesis.grid()}
        if combo not in declared:
            raise DataError(
                f"{hypothesis.id} : combinaison {dict(combo)} non déclarée dans la "
                f"fiche ({hypothesis.n_trials} déclarée(s))"
            )
        trial = Trial(
            ts_ms, hypothesis.volet, hypothesis.id, hypothesis.fingerprint, combo, result, note
        )

        def same_fiche(existing: list[Record]) -> None:
            for n, obj in enumerate(existing, start=1):
                old = _parse(obj, n)
                if old.hypothesis_id == hypothesis.id and old.fingerprint != trial.fingerprint:
                    raise DataError(
                        f"{hypothesis.id} : fiche modifiée après des essais "
                        "(empreinte différente) ; créer une nouvelle fiche"
                    )

        append_record(self.path, trial.to_record(), precheck=same_fiche)
        return trial


# --- commande ----------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["show"])
    parser.add_argument("--volet", choices=VOLETS, default="longterm")


def _action(ctx: Context) -> int:
    registry = TrialRegistry(ctx.paths.trials)
    latest = registry.latest(ctx.args.volet)
    print(f"Volet {ctx.args.volet} : N = {len(latest)} essai(s) distinct(s)")
    by_hyp: dict[str, int] = {}
    for t in latest.values():
        by_hyp[t.hypothesis_id] = by_hyp.get(t.hypothesis_id, 0) + 1
    for hyp_id, count in sorted(by_hyp.items()):
        print(f"  {hyp_id:<28} {count} essai(s)")
    for t in sorted(latest.values(), key=lambda t: t.ts_ms)[-10:]:
        print(
            f"  {ms_to_iso(t.ts_ms)}  {t.hypothesis_id}  {json.dumps(dict(t.params))}  "
            f"Sharpe annuel {t.result.annual_sharpe:+.2f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.research.trials",
        description="Registre des essais (N du Deflated Sharpe)",
        component="trials",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
