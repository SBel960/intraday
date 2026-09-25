"""Synchronise les archives data.binance.vision vers ``raw/binance_vision/`` (RAW immuable).

Pour chaque archive publiée : si le fichier local existe avec la bonne taille, il est déjà
là (il n'est écrit qu'après vérification) ; sinon on télécharge l'archive **et** son
``.CHECKSUM``, on vérifie SHA-256 et taille, puis écriture atomique (``core/files.py``). Une
archive republiée avec une autre taille n'écrase jamais l'ancienne : elle est signalée.

Relancer reprend là où on s'était arrêté (idempotent). Le réseau passe par ``core/http.py``
(attente du retour de la connexion). Téléchargements en parallèle : ``archives.download_workers``.

Ce qui est synchronisé (config) :
- bougies spot mensuelles de **toutes** les paires (``observe.kline_intervals``), retirées
  comprises si ``observe.include_delisted`` ; + quotidiennes du mois en cours ;
- si ``observe.futures_metrics`` : financement USDⓈ-M mensuel de toutes les paires, métriques
  quotidiennes des paires ``archives.futures_metrics_symbols``.

Commande : ``python -m qlab.data.archives --config config sync [--dataset …] [--symbols A,B]
[--dry-run]``
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError, ExchangeError
from qlab.core.files import write_atomic
from qlab.core.paths import DataPaths
from qlab.core.timeutils import date_str, now_ms
from qlab.data import binance_vision as bv
from qlab.exchange.snapshots import SnapshotStore

SOURCE = "binance_vision"
DATASETS = ("klines", "funding", "metrics")
Fetch = Callable[[str], bytes]


@dataclass
class SyncReport:
    """Bilan d'une synchronisation (compteurs en nombre d'archives, octets téléchargés)."""

    listed: int = 0
    present: int = 0
    downloaded: int = 0
    downloaded_bytes: int = 0
    republished: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def archive_prefixes(
    fetch: Fetch,
    list_url: str,
    *,
    datasets: Sequence[str],
    intervals: Sequence[str],
    symbols: Sequence[str] | None,
    metrics_symbols: Sequence[str],
    today: str,
    trading: set[str] | None,
    active: set[str] | None,
) -> list[str]:
    """Dossiers (préfixes S3) à lister pour les jeux demandés — 2 listages seulement.

    ``symbols`` : restriction (``None`` = toutes) ; ``trading`` : si fourni, seules ces paires
    (``include_delisted`` faux). Mois en cours : archives quotidiennes (pas encore de mensuelle),
    listées seulement pour les paires ``active`` (en cotation d'après le dernier snapshot ; une
    paire retirée n'en publie plus) ; ``None`` = pour toutes.
    """

    def keep(sym: str) -> bool:
        return (symbols is None or sym in symbols) and (trading is None or sym in trading)

    prefixes: list[str] = []
    if "klines" in datasets:
        month = today[:7]
        for sym in filter(keep, bv.list_subdirs(fetch, list_url, bv.SPOT_KLINES_MONTHLY)):
            for iv in intervals:
                prefixes.append(f"{bv.SPOT_KLINES_MONTHLY}{sym}/{iv}/")
                if active is None or sym in active:
                    prefixes.append(f"{bv.SPOT_KLINES_DAILY}{sym}/{iv}/{sym}-{iv}-{month}")
    if "funding" in datasets:
        for sym in filter(keep, bv.list_subdirs(fetch, list_url, bv.UM_FUNDING_MONTHLY)):
            prefixes.append(f"{bv.UM_FUNDING_MONTHLY}{sym}/")
    if "metrics" in datasets:
        prefixes += [
            f"{bv.UM_METRICS_DAILY}{s}/" for s in metrics_symbols if symbols is None or s in symbols
        ]
    return prefixes


def list_all(
    fetch: Fetch, list_url: str, prefixes: Sequence[str], *, workers: int
) -> list[bv.ArchiveFile]:
    """Liste tous les ``prefixes`` en parallèle ; résultat trié par clé (déterministe)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = pool.map(lambda p: bv.list_archives(fetch, list_url, p), prefixes)
        return sorted((f for files in found for f in files), key=lambda f: f.key)


def download_one(archive: bv.ArchiveFile, base_url: str, paths: DataPaths, fetch: Fetch) -> int:
    """Télécharge, vérifie (SHA-256, taille) et écrit une archive ; renvoie sa taille."""
    url = bv.file_url(base_url, archive.key)
    data = fetch(url)
    name = archive.key.rsplit("/", 1)[-1]
    expected = bv.parse_checksum(fetch(url + ".CHECKSUM"), name)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DataError(f"SHA-256 incorrect pour {name} : {actual[:12]}… ≠ {expected[:12]}…")
    if len(data) != archive.size:
        raise DataError(f"taille {len(data)} ≠ {archive.size} annoncée pour {name}")
    write_atomic(paths.raw_archive(SOURCE, archive.relative_key), data)
    return len(data)


def sync(
    archives: Sequence[bv.ArchiveFile],
    *,
    base_url: str,
    paths: DataPaths,
    fetch: Fetch,
    workers: int,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> SyncReport:
    """Télécharge les archives absentes, en parallèle ; une archive en échec n'arrête pas les
    autres (listée dans ``failed``, retentée au prochain passage)."""
    report = SyncReport(listed=len(archives))
    todo = []
    for a in archives:
        local = paths.raw_archive(SOURCE, a.relative_key)
        if not local.exists():
            todo.append(a)
        elif local.stat().st_size == a.size:
            report.present += 1
        else:
            report.republished.append(a.key)
    if dry_run or not todo:
        return report
    step = max(1, len(todo) // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, a, base_url, paths, fetch): a for a in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                report.downloaded_bytes += fut.result()
                report.downloaded += 1
            except (DataError, ExchangeError) as exc:
                report.failed.append((futures[fut].key, str(exc)))
            if i % step == 0 or i == len(todo):
                progress(
                    f"  {i}/{len(todo)} archives traitées ({report.downloaded_bytes / 1e6:.1f} Mo)"
                )
    return report


# --- commande ----------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["sync"])
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--symbols", help="restreindre à ces paires, ex. BTCEUR,ETHEUR")
    parser.add_argument("--dry-run", action="store_true", help="compter sans télécharger")


def _action(ctx: Context) -> int:
    cfg, a = ctx.config.base, ctx.args
    observe, archives_cfg = cfg.observe, cfg.archives

    def notify(message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        ctx.journal.warning("fetch.retry", {"message": message})

    def fetch(url: str) -> bytes:
        return http.get(url, notify=notify).body

    datasets: tuple[str, ...] = DATASETS if a.dataset == "all" else (a.dataset,)
    if not observe.futures_metrics:
        datasets = tuple(d for d in datasets if d == "klines")
    latest = SnapshotStore(ctx.paths, cfg.exchange(a.exchange).name).latest()
    active = None if latest is None else {s["symbol"] for s in latest.symbols()}
    if latest is None:
        print("Pas de snapshot exchangeInfo : mois en cours listé pour toutes les paires (lent)")
    if not observe.include_delisted and active is None:
        raise DataError("include_delisted faux : un snapshot exchangeInfo est nécessaire")
    trading = None if observe.include_delisted else active
    symbols = None if a.symbols is None else [s.strip() for s in a.symbols.split(",")]
    print("Listage des archives publiées…")
    list_url = archives_cfg.binance_vision_list_url
    prefixes = archive_prefixes(
        fetch,
        list_url,
        datasets=datasets,
        intervals=observe.kline_intervals,
        symbols=symbols,
        metrics_symbols=archives_cfg.futures_metrics_symbols,
        today=date_str(now_ms()),
        trading=trading,
        active=active,
    )
    print(f"{len(prefixes)} dossiers à lister…", flush=True)
    remote = list_all(fetch, list_url, prefixes, workers=archives_cfg.list_workers)
    total_mb = sum(r.size for r in remote) / 1e6
    print(f"{len(remote)} archives publiées ({total_mb:.1f} Mo)")
    report = sync(
        remote,
        base_url=archives_cfg.binance_vision_url,
        paths=ctx.paths,
        fetch=fetch,
        workers=archives_cfg.download_workers,
        dry_run=a.dry_run,
    )
    missing = report.listed - report.present - len(report.republished)
    print(
        f"Déjà présentes : {report.present} ; à télécharger : {missing}"
        + (
            " (simulation)"
            if a.dry_run
            else f" ; téléchargées : {report.downloaded} ({report.downloaded_bytes / 1e6:.1f} Mo)"
        )
    )
    for key in report.republished:
        print(
            f"ATTENTION : republiée avec une autre taille, ancienne gardée : {key}", file=sys.stderr
        )
    for key, msg in report.failed:
        print(f"ÉCHEC : {key} : {msg}", file=sys.stderr)
    ctx.journal.info(
        "archives.sync",
        {
            "datasets": list(datasets),
            "listed": report.listed,
            "present": report.present,
            "downloaded": report.downloaded,
            "bytes": report.downloaded_bytes,
            "republished": list(report.republished[:100]),
            "failed": list(k for k, _ in report.failed[:100]),
            "dry_run": a.dry_run,
        },
    )
    return 1 if report.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.data.archives",
        description="Synchronise les archives data.binance.vision",
        component="archives",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
