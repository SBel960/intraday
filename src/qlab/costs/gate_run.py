"""Cost gate v1 sur données réelles : cotations Tardis (1er de chaque mois) × frais réels.

Pour chaque paire (défaut : ``symbols.intraday``) :

1. frais taker **réels** de la paire, lus dans le dernier snapshot (``exchange/fees``) ; s'ils
   ne sont que le barème de repli, le rapport le dit ;
2. chaque journée Tardis est échantillonnée séparément (``cost_gate.sample_segment`` : pas de
   grille entre deux journées), puis le gate est calculé sur l'ensemble (``summarize``) ;
3. rapport Markdown dans ``reports/`` et résumé dans le journal.

Limites écrites dans le rapport : échantillon d'un jour par mois (1er du mois, gratuit) ; frais
actuels appliqués à tout l'historique (Binance ne publie pas l'historique des frais d'un compte).

Commande : ``python -m qlab.costs.gate_run --config config [--symbols A,B]``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import date_str, ms_to_iso, now_ms
from qlab.costs.cost_gate import ALL_SLOTS, GateParams, GateResult, sample_segment, summarize
from qlab.data import tardis
from qlab.exchange.fees import pair_fees
from qlab.exchange.snapshots import SnapshotStore


@dataclass(frozen=True, slots=True)
class PairGate:
    symbol: str
    fee_taker: float
    fee_origin: str
    days: tuple[str, ...]
    result: GateResult


def gate_pair(ctx: Context, symbol: str, fee_taker: float, fee_origin: str) -> PairGate:
    """Gate d'une paire sur toutes ses journées Tardis locales."""
    files = tardis.local_files(ctx.paths, symbol)
    if not files:
        raise DataError(
            f"{symbol} : aucune journée Tardis ; lancer « python -m qlab.data.tardis "
            "--config config sync »"
        )
    params = GateParams.from_config(ctx.config, fee_taker)
    segments = [sample_segment(tardis.read_quotes(f.read_bytes()), params) for f in files]
    days = tuple(f.name.removesuffix(".csv.gz") for f in files)
    return PairGate(symbol, fee_taker, fee_origin, days, summarize(segments, params))


def _overall_table(result: GateResult) -> list[str]:
    rows = [
        "| Horizon | Échantillons | Coût médian | Coût p90 | Mouvement médian | Ratio |",
        "|---|---|---|---|---|---|",
    ]
    overall = result.table.filter(pl.col("slot") == ALL_SLOTS)
    for r in overall.iter_rows(named=True):
        rows.append(
            f"| {r['horizon_s']} s | {r['n']} | {r['cost_median']:.4%} | "
            f"{r['cost_p90']:.4%} | {r['move_median']:.4%} | {r['ratio']:.2f} |"
        )
    return rows


def _slots_summary(result: GateResult) -> str:
    by_slot = {s: h for s, h in result.min_horizon_s.items() if s != ALL_SLOTS}
    passing = {s: h for s, h in by_slot.items() if h is not None}
    if not passing:
        return f"Aucune des {len(by_slot)} tranches horaires ne passe."
    detail = ", ".join(f"{s:02d} h → {h} s" for s, h in sorted(passing.items()))
    return f"{len(passing)} tranche(s) sur {len(by_slot)} passent : {detail}."


def render(gates: Sequence[PairGate], generated_ms: int) -> str:
    """Rapport Markdown (déterministe pour des entrées données)."""
    lines = [
        f"# Cost gate v1 — {ms_to_iso(generated_ms)}",
        "",
        "Coût aller-retour taker c = 2 f + s̃ + 2 slip ; mouvement = |ln m(t+h)/m(t)| ; "
        "horizon retenu : premier où mouvement médian / coût médian > seuil.",
        "",
    ]
    for g in gates:
        r = g.result
        origin = "frais réels du compte" if g.fee_origin == "account" else "barème de REPLI"
        lines += [
            f"## {g.symbol}",
            "",
            f"- Frais taker : {g.fee_taker:.4%} ({origin})",
            f"- Journées : {len(g.days)} (1er du mois, {g.days[0]} → {g.days[-1]})",
            f"- Cotations : {r.quotes_used} utilisées, {r.quotes_excluded} exclues",
            f"- **Verdict : {r.verdict()}**",
            f"- Tranches horaires : {_slots_summary(r)}",
            "",
            *_overall_table(r),
            "",
        ]
    lines += [
        "## Limites",
        "",
        "- Un jour par mois (1er du mois, gratuit chez Tardis) : échantillon, pas l'historique.",
        "- Frais actuels appliqués à tout l'historique (pas d'historique des frais du compte).",
        "- Glissement supposé : `intraday.gate.slippage_frac` (config).",
        "",
    ]
    return "\n".join(lines)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", help="paires, ex. BTCEUR ; défaut : symbols.intraday")
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    base = ctx.config.base
    symbols = (
        list(base.symbols.intraday)
        if ctx.args.symbols is None
        else [s.strip() for s in ctx.args.symbols.split(",")]
    )
    snapshot = SnapshotStore(ctx.paths, base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    gates = []
    for symbol in symbols:
        fees = pair_fees(snapshot.fees, symbol)
        print(f"{symbol} : frais taker {fees.taker:.4%} ({fees.origin}) — calcul…", flush=True)
        gates.append(gate_pair(ctx, symbol, fees.taker, fees.origin))
        print(f"  {gates[-1].result.verdict()}")
    now = now_ms()
    report = ctx.paths.reports / f"cost_gate_v1_{date_str(now)}.md"
    write_atomic(report, render(gates, now).encode("utf-8"), overwrite=True)
    print(f"Rapport : {report}")
    ctx.journal.info(
        "gate.v1",
        {
            g.symbol: {
                "fee_taker": g.fee_taker,
                "days": len(g.days),
                "min_horizon_s": g.result.overall_min_horizon_s,
            }
            for g in gates
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.costs.gate_run",
        description="Cost gate v1 sur données réelles (Tardis × frais réels)",
        component="gate_run",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
