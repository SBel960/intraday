"""Téléchargements parallèles vérifiés vers les données brutes (RAW immuable).

Chaque source (archives Binance, Tardis…) décrit ses fichiers comme des ``Job`` : où les ranger
et comment les **télécharger et vérifier** (empreinte, format…). ``download_all`` fait le reste,
de la même façon pour toutes :

- fichier local déjà présent (taille égale à celle annoncée, si connue) : rien à faire ; un
  fichier n'est écrit qu'après vérification, sa présence suffit donc ;
- taille différente de celle annoncée : **jamais réécrit**, signalé (``changed``) ;
- sinon téléchargement en parallèle, écriture atomique (``core/files.py``) ;
- un échec (``DataError`` / ``ExchangeError``) n'arrête pas les autres : il est listé et le
  fichier sera retenté au prochain passage. Toute autre exception est un bug et remonte.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from qlab.core.errors import DataError, ExchangeError
from qlab.core.files import write_atomic


@dataclass(frozen=True, slots=True)
class Job:
    """Un fichier à obtenir. ``fetch_verified`` télécharge **et** vérifie, ou lève
    ``DataError`` / ``ExchangeError`` ; ``expected_size`` : taille annoncée (``None`` : inconnue).
    """

    key: str
    dest: Path
    fetch_verified: Callable[[], bytes]
    expected_size: int | None = None


@dataclass
class DownloadReport:
    """Bilan : nombres de fichiers, octets téléchargés, fichiers changés et échecs (clé, motif)."""

    listed: int = 0
    present: int = 0
    downloaded: int = 0
    downloaded_bytes: int = 0
    changed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def missing(self) -> int:
        """Fichiers à télécharger (ni présents, ni changés, ni écartés d'avance)."""
        return self.listed - self.present - len(self.changed)


def _get(job: Job) -> int:
    data = job.fetch_verified()
    if job.expected_size is not None and len(data) != job.expected_size:
        raise DataError(f"taille {len(data)} ≠ {job.expected_size} annoncée pour {job.key}")
    write_atomic(job.dest, data)
    return len(data)


def download_all(
    jobs: Sequence[Job],
    *,
    workers: int,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> DownloadReport:
    """Obtient les fichiers absents, ``workers`` à la fois ; ``dry_run`` : compte seulement."""
    report = DownloadReport(listed=len(jobs))
    todo = []
    for job in jobs:
        if not job.dest.exists():
            todo.append(job)
        elif job.expected_size is None or job.dest.stat().st_size == job.expected_size:
            report.present += 1
        else:
            report.changed.append(job.key)
    if dry_run or not todo:
        return report
    step = max(1, len(todo) // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_get, job): job for job in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                report.downloaded_bytes += fut.result()
                report.downloaded += 1
            except (DataError, ExchangeError) as exc:
                report.failed.append((futures[fut].key, str(exc)))
            if i % step == 0 or i == len(todo):
                progress(
                    f"  {i}/{len(todo)} fichiers traités ({report.downloaded_bytes / 1e6:.1f} Mo)"
                )
    return report
