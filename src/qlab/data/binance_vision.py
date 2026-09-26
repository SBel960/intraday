"""data.binance.vision : ce qui est publié, où, et comment le vérifier. Aucun téléchargement.

Le serveur d'archives est un bucket S3 public. Le listage (``?prefix=…&delimiter=/``) renvoie
du XML, par pages de 1 000 entrées (``IsTruncated`` / ``NextMarker``). Chaque archive ``.zip``
a un fichier ``.zip.CHECKSUM`` voisin : ``<sha256 hex>  <nom du zip>``.

Jeux de données (clés sous ``data/``) :

- bougies spot mensuelles : ``spot/monthly/klines/{SYM}/{iv}/{SYM}-{iv}-YYYY-MM.zip``
- bougies spot quotidiennes (mois en cours) : ``spot/daily/klines/{SYM}/{iv}/…-YYYY-MM-DD.zip``
- financement USDⓈ-M mensuel : ``futures/um/monthly/fundingRate/{SYM}/…-YYYY-MM.zip``
- métriques USDⓈ-M quotidiennes : ``futures/um/daily/metrics/{SYM}/…-YYYY-MM-DD.zip``
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

from qlab.core.errors import ExchangeError

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_SHA256_RE = re.compile(r"^([0-9a-f]{64})\s+(\S+)\s*$")

SPOT_KLINES_MONTHLY = "data/spot/monthly/klines/"
SPOT_KLINES_DAILY = "data/spot/daily/klines/"
UM_FUNDING_MONTHLY = "data/futures/um/monthly/fundingRate/"
UM_METRICS_DAILY = "data/futures/um/daily/metrics/"

Fetch = Callable[[str], bytes]  # GET → corps (``core/http.get`` en production)


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """Une archive publiée : clé S3 complète (``data/…/X.zip``) et taille en octets."""

    key: str
    size: int

    @property
    def relative_key(self) -> str:
        """Clé sans le préfixe ``data/`` (chemin local sous ``raw/binance_vision/``)."""
        return self.key.removeprefix("data/")


def _page(fetch: Fetch, list_url: str, prefix: str, marker: str, delimiter: bool) -> ET.Element:
    query = {"prefix": prefix, "marker": marker}
    if delimiter:
        query["delimiter"] = "/"
    body = fetch(f"{list_url}?{urllib.parse.urlencode(query)}")
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise ExchangeError(f"listage S3 illisible pour {prefix!r}") from exc


def _pages(fetch: Fetch, list_url: str, prefix: str, *, delimiter: bool) -> list[ET.Element]:
    pages, marker = [], ""
    while True:
        root = _page(fetch, list_url, prefix, marker, delimiter)
        pages.append(root)
        if root.findtext(f"{_S3_NS}IsTruncated") != "true":
            return pages
        entries = [e.text or "" for e in root.iter(f"{_S3_NS}Key")] + [
            e.text or "" for e in root.iter(f"{_S3_NS}Prefix")
        ][1:]
        next_marker = root.findtext(f"{_S3_NS}NextMarker") or (max(entries) if entries else "")
        if not next_marker or next_marker <= marker:
            raise ExchangeError(f"pagination S3 bloquée pour {prefix!r}")
        marker = next_marker


def list_subdirs(fetch: Fetch, list_url: str, prefix: str) -> list[str]:
    """Noms des sous-dossiers directs de ``prefix`` (ex. symboles), triés."""
    names = []
    for root in _pages(fetch, list_url, prefix, delimiter=True):
        for cp in root.iter(f"{_S3_NS}CommonPrefixes"):
            p = cp.findtext(f"{_S3_NS}Prefix") or ""
            names.append(p.removeprefix(prefix).rstrip("/"))
    return sorted(n for n in names if n)


def list_archives(fetch: Fetch, list_url: str, prefix: str) -> list[ArchiveFile]:
    """Archives ``.zip`` sous ``prefix`` (récursif), triées par clé. Les ``.CHECKSUM`` sont
    exclus : ils sont téléchargés avec leur archive."""
    files = []
    for root in _pages(fetch, list_url, prefix, delimiter=False):
        for c in root.iter(f"{_S3_NS}Contents"):
            key = c.findtext(f"{_S3_NS}Key") or ""
            size = c.findtext(f"{_S3_NS}Size") or ""
            if key.endswith(".zip"):
                if not size.isdigit():
                    raise ExchangeError(f"taille absente dans le listage pour {key}")
                files.append(ArchiveFile(key, int(size)))
    return sorted(files, key=lambda f: f.key)


def parse_checksum(text: bytes, zip_name: str) -> str:
    """SHA-256 attendu d'après un ``.CHECKSUM`` ; vérifie qu'il vise bien ``zip_name``."""
    m = _SHA256_RE.match(text.decode("utf-8", errors="replace").strip())
    if m is None:
        raise ExchangeError(f"CHECKSUM illisible pour {zip_name}")
    if m.group(2) != zip_name:
        raise ExchangeError(f"CHECKSUM pour {m.group(2)!r}, attendu {zip_name!r}")
    return m.group(1)


def file_url(base_url: str, key: str) -> str:
    """URL de téléchargement ; la clé est encodée (des paires ont un nom non ASCII, ex.
    ``币安人生USDT`` : sans encodage, la requête HTTP échoue)."""
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(key, safe='/')}"
