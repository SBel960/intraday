"""Registre append-only des apports et retraits : ``$DATA_ROOT/meta/ledger.jsonl``.

Sert à trois choses :
- le **capital apporté** (dépôts − retraits), qui fixe le palier (``BaseConfig.tier_for``) ;
- les **flux externes** datés, pour séparer apport et performance (TWR / MWR du DCA) ;
- la trace de chaque mouvement d'argent, jamais modifiée ni supprimée.

Une ligne = un flux ``{"ts_ms", "kind", "amount_quote", "note", "amount_fiat", "fiat"}``.
``amount_quote`` est un ``Decimal`` sérialisé en texte (jamais de flottant pour de l'argent), en
devise de cotation, strictement positif ; le signe vient de ``kind``. ``amount_fiat`` / ``fiat``
(optionnels, les deux ou aucun, ``null`` sinon) gardent le montant réellement versé ou reçu en
monnaie fiat (ex. ``50.00`` ``EUR``) : trace fiscale et coût réel de conversion.

Les flux sont en ordre chronologique (``ts_ms`` croissant au sens large). Chaque ajout se fait
sous verrou exclusif (``flock``) : relecture, contrôle chronologique, écriture, ``fsync``.

Lecture stricte : une ligne illisible ou hors schéma lève ``DataError`` avec son numéro. Pour de
l'argent, on ne saute jamais une ligne ; la réparation est manuelle.

Commande : ``python -m qlab.core.ledger --config config {show|deposit|withdrawal} ...``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from qlab.core.config import load_config
from qlab.core.errors import DataError, run_cli
from qlab.core.timeutils import date_to_ms, ms_to_iso, now_ms

FlowKind = Literal["deposit", "withdrawal"]
KINDS: tuple[FlowKind, ...] = ("deposit", "withdrawal")
LEDGER_RELPATH = Path("meta") / "ledger.jsonl"  # sous la racine des données de la config
_FIELDS = frozenset({"ts_ms", "kind", "amount_quote", "note", "amount_fiat", "fiat"})
_FIAT_RE = re.compile(r"^[A-Z]{3}$")  # code ISO 4217


MAX_INT_DIGITS = 12  # < 10¹² en devise de cotation : au-delà, c'est une faute de frappe
MAX_DECIMALS = 18
_AMOUNT_RE = re.compile(rf"^\d{{1,{MAX_INT_DIGITS}}}(\.\d{{1,{MAX_DECIMALS}}})?$")


def check_amount(amount: Decimal) -> Decimal:
    """Vérifie un montant : fini, > 0, < 10¹², au plus 18 décimales ; sinon ``DataError``."""
    if not amount.is_finite() or amount <= 0:
        raise DataError(f"montant doit être fini et > 0 : {amount}")
    if amount.adjusted() >= MAX_INT_DIGITS:
        raise DataError(f"montant trop grand (≥ 10^{MAX_INT_DIGITS}) : {amount}")
    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -MAX_DECIMALS:
        raise DataError(f"montant avec plus de {MAX_DECIMALS} décimales : {amount}")
    return amount


def parse_amount(text: str) -> Decimal:
    """Montant en notation décimale simple (``50``, ``12.34`` ; ni signe ni exposant)."""
    if not _AMOUNT_RE.match(text):
        raise DataError(f"montant illisible (attendu ex. 50 ou 12.34) : {text!r}")
    return check_amount(Decimal(text))


def format_amount(amount: Decimal) -> str:
    """Notation décimale, jamais scientifique : ``0.00000001`` et non ``1E-8``."""
    return format(amount, "f")


@dataclass(frozen=True, slots=True)
class Flow:
    """Un apport (``deposit``) ou un retrait (``withdrawal``) à l'instant ``ts_ms`` (epoch ms)."""

    ts_ms: int
    kind: FlowKind
    amount_quote: Decimal
    note: str = ""
    amount_fiat: Decimal | None = None
    fiat: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ms, int) or isinstance(self.ts_ms, bool):
            raise DataError(f"ts_ms doit être un entier (epoch ms), reçu {self.ts_ms!r}")
        if self.kind not in KINDS:
            raise DataError(f"kind doit être deposit ou withdrawal, reçu {self.kind!r}")
        if not isinstance(self.amount_quote, Decimal):
            raise DataError(f"amount_quote doit être un Decimal, reçu {self.amount_quote!r}")
        check_amount(self.amount_quote)
        if not isinstance(self.note, str) or "\n" in self.note:
            raise DataError("note doit être une chaîne sur une seule ligne")
        if (self.amount_fiat is None) != (self.fiat is None):
            raise DataError("amount_fiat et fiat vont ensemble : les deux ou aucun")
        if self.amount_fiat is not None:
            if not isinstance(self.amount_fiat, Decimal):
                raise DataError(f"amount_fiat doit être un Decimal, reçu {self.amount_fiat!r}")
            check_amount(self.amount_fiat)
            if not isinstance(self.fiat, str) or not _FIAT_RE.match(self.fiat):
                raise DataError(
                    f"fiat doit être un code ISO à 3 lettres (ex. EUR), reçu {self.fiat!r}"
                )

    @property
    def signed_quote(self) -> Decimal:
        """+montant pour un apport, −montant pour un retrait."""
        return self.amount_quote if self.kind == "deposit" else -self.amount_quote

    def to_line(self) -> bytes:
        record = {
            "ts_ms": self.ts_ms,
            "kind": self.kind,
            "amount_quote": format_amount(self.amount_quote),
            "note": self.note,
            "amount_fiat": None if self.amount_fiat is None else format_amount(self.amount_fiat),
            "fiat": self.fiat,
        }
        text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return (text + "\n").encode("utf-8")


def _parse_line(raw: bytes, n: int, path: Path) -> Flow:
    where = f"{path}, ligne {n}"
    if not raw.endswith(b"\n"):
        raise DataError(f"{where} : ligne tronquée (pas de fin de ligne)")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataError(f"{where} : JSON illisible") from exc
    if not isinstance(obj, dict) or set(obj) != _FIELDS:
        raise DataError(f"{where} : champs attendus {sorted(_FIELDS)}")
    fiat_text = obj["amount_fiat"]
    if not isinstance(obj["amount_quote"], str) or not isinstance(fiat_text, str | None):
        raise DataError(f"{where} : montants en texte attendus (pas de flottant)")
    try:
        amount_fiat = None if fiat_text is None else parse_amount(fiat_text)
        return Flow(
            obj["ts_ms"],
            obj["kind"],
            parse_amount(obj["amount_quote"]),
            obj["note"],
            amount_fiat,
            obj["fiat"],
        )
    except DataError as exc:
        raise DataError(f"{where} : {exc}") from exc


class Ledger:
    """Registre stocké dans ``path`` (créé au premier ajout, dossiers parents compris)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_data_root(cls, data_root: Path) -> Ledger:
        return cls(data_root / LEDGER_RELPATH)

    def flows(self) -> tuple[Flow, ...]:
        """Tous les flux, validés, dans l'ordre du fichier ; ``()`` si le registre n'existe pas."""
        if not self.path.exists():
            return ()
        flows: list[Flow] = []
        with self.path.open("rb") as f:
            for n, raw in enumerate(f, start=1):
                flow = _parse_line(raw, n, self.path)
                if flows and flow.ts_ms < flows[-1].ts_ms:
                    raise DataError(f"{self.path}, ligne {n} : flux antérieur au précédent")
                flows.append(flow)
        return tuple(flows)

    def append(self, flow: Flow) -> None:
        """Ajoute ``flow`` sous verrou exclusif : relecture et validation, écriture, ``fsync``.

        Un second écrivain (autre commande lancée en même temps) attend la fin du premier, ce
        qui garantit le contrôle chronologique. À la création du fichier, le dossier est aussi
        ``fsync`` pour que l'entrée de répertoire survive à une coupure.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        line = flow.to_line()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # libéré par os.close
            existing = self.flows()
            if existing and flow.ts_ms < existing[-1].ts_ms:
                raise DataError(
                    f"flux au {ms_to_iso(flow.ts_ms)} antérieur au dernier "
                    f"({ms_to_iso(existing[-1].ts_ms)}) : le registre est chronologique"
                )
            if os.write(fd, line) != len(line):
                raise OSError(f"écriture partielle dans {self.path}")
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def net_deposits_quote(self, at_ms: int | None = None) -> Decimal:
        """Apports − retraits jusqu'à ``at_ms`` inclus (tout le registre si ``None``).

        Peut être ≤ 0 si l'on a retiré plus que déposé (gains retirés) : c'est à l'appelant de
        décider quoi faire, ``tier_for`` refuse un capital apporté ≤ 0.
        """
        return sum(
            (f.signed_quote for f in self.flows() if at_ms is None or f.ts_ms <= at_ms),
            Decimal(0),
        )


# --- commande ----------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qlab.core.ledger", description=__doc__.split("\n")[0]
    )
    p.add_argument("--config", type=Path, required=True, help="dossier des YAML")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("show", help="liste les flux, le capital apporté et le palier")
    for kind in KINDS:
        s = sub.add_parser(kind, help=f"enregistre un {kind} (montant en devise de cotation)")
        s.add_argument("amount", help="montant, ex. 50 ou 12.34")
        s.add_argument("--date", help="YYYY-MM-DD (minuit UTC) ; défaut : maintenant")
        s.add_argument("--note", default="", help="commentaire sur une ligne")
        s.add_argument("--fiat-amount", help="montant réellement versé en fiat, ex. 50.00")
        s.add_argument("--fiat", help="code ISO de la monnaie fiat, ex. EUR (avec --fiat-amount)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def run() -> int:
        cfg = load_config(args.config).base
        ledger = Ledger.from_data_root(cfg.data.root)
        quote = cfg.symbols.quote_asset
        if args.action != "show":
            try:
                ts_ms = date_to_ms(args.date) if args.date else now_ms()
            except ValueError as exc:
                raise DataError(str(exc)) from exc
            if (args.fiat_amount is None) != (args.fiat is None):
                raise DataError("--fiat-amount et --fiat vont ensemble : les deux ou aucun")
            fiat_amount = None if args.fiat_amount is None else parse_amount(args.fiat_amount)
            kind = cast(FlowKind, args.action)
            ledger.append(
                Flow(ts_ms, kind, parse_amount(args.amount), args.note, fiat_amount, args.fiat)
            )
        for f in ledger.flows():
            fiat = "" if f.amount_fiat is None else f" ({format_amount(f.amount_fiat)} {f.fiat})"
            print(
                f"{ms_to_iso(f.ts_ms)}  {f.kind:<10} {f.signed_quote:>+14f} {quote}{fiat}  {f.note}"
            )
        net = ledger.net_deposits_quote()
        print(f"Capital apporté : {format_amount(net)} {quote}")
        if net > 0:
            tier = cfg.tier_for(net)
            print(f"Palier : {tier.name} (drawdown max {tier.max_drawdown_frac:.0%})")
        else:
            print("Palier : aucun (capital apporté ≤ 0)")
        return 0

    return run_cli(run)


if __name__ == "__main__":
    raise SystemExit(main())
