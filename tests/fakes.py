"""Doublures de test partagées (aucun réseau) : faux bucket S3 de data.binance.vision, archives
de bougies construites à la main."""

from __future__ import annotations

import hashlib
import urllib.parse
import zipfile
from pathlib import Path

from qlab.core.paths import DataPaths

NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def s3_xml(
    prefix: str,
    keys: list[tuple[str, int]],
    subdirs: list[str],
    truncated: bool,
    next_marker: str | None = None,
) -> bytes:
    parts = [
        f'<ListBucketResult xmlns="{NS}"><Prefix>{prefix}</Prefix>',
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
    ]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts += [f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in keys]
    parts += [f"<CommonPrefixes><Prefix>{d}</Prefix></CommonPrefixes>" for d in subdirs]
    return ("".join(parts) + "</ListBucketResult>").encode()


class FakeS3:
    """Bucket factice : ``files`` (clé → taille), pages de ``page_size`` entrées."""

    def __init__(self, files: dict[str, int], page_size: int = 1000) -> None:
        self.files, self.page_size, self.calls = files, page_size, 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))
        prefix, marker, delim = q["prefix"], q.get("marker", ""), "delimiter" in q
        keys = sorted(k for k in self.files if k.startswith(prefix))
        if delim:
            entries = sorted(
                {
                    prefix + k[len(prefix) :].split("/")[0] + "/" if "/" in k[len(prefix) :] else k
                    for k in keys
                }
            )
        else:
            entries = keys
        entries = [e for e in entries if e > marker]
        page, rest = entries[: self.page_size], entries[self.page_size :]
        subdirs = [e for e in page if e.endswith("/")]
        files = [(e, self.files[e]) for e in page if not e.endswith("/")]
        nxt = page[-1] if rest and delim else None
        return s3_xml(prefix, files, subdirs, bool(rest), nxt)


class FakeVision:
    """Faux data.binance.vision complet : listage S3 (``list_url``), archives et ``.CHECKSUM``
    (``base_url``). ``content`` : clé → octets ; ``bad_checksum`` : clés à empreinte fausse ;
    ``failing`` : clés dont le téléchargement lève l'exception donnée."""

    def __init__(
        self,
        content: dict[str, bytes],
        *,
        list_url: str = "https://s3.example/bucket",
        base_url: str = "https://vision.example",
        page_size: int = 1000,
        bad_checksum: set[str] | None = None,
        failing: dict[str, Exception] | None = None,
    ) -> None:
        self.content, self.list_url, self.base_url = content, list_url, base_url
        self.bad_checksum, self.failing = bad_checksum or set(), failing or {}
        self.s3 = FakeS3({k: len(v) for k, v in content.items()}, page_size)
        self.downloads: list[str] = []

    def __call__(self, url: str) -> bytes:
        if url.startswith(self.list_url + "?"):
            return self.s3(url)
        key = urllib.parse.unquote(url.removeprefix(self.base_url + "/"))  # comme le serveur
        if key.endswith(".CHECKSUM"):
            zip_key = key.removesuffix(".CHECKSUM")
            sha = hashlib.sha256(self.content[zip_key]).hexdigest()
            if zip_key in self.bad_checksum:
                sha = "0" * 64
            return f"{sha}  {zip_key.rsplit('/', 1)[-1]}\n".encode()
        if key in self.failing:
            raise self.failing[key]
        self.downloads.append(key)
        return self.content[key]


# --- réponses exchangeInfo construites à la main ------------------------------------------

T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z
EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
FALLBACK_FEES: dict[str, object] = {
    "origin": "config",
    "maker_frac": 0.001,
    "taker_frac": 0.001,
    "bnb_discount_frac": 0.25,
    "pay_in_bnb": False,
    "effective_maker_frac": 0.001,
    "effective_taker_frac": 0.001,
}


def binance_filters(
    tick: str = "0.01", step: str = "0.00001", min_notional: str = "5"
) -> list[dict[str, object]]:
    """Filtres PRICE_FILTER / LOT_SIZE / NOTIONAL au format exchangeInfo."""
    return [
        {"filterType": "PRICE_FILTER", "minPrice": tick, "maxPrice": "1000000", "tickSize": tick},
        {"filterType": "LOT_SIZE", "minQty": step, "maxQty": "9000", "stepSize": step},
        {
            "filterType": "NOTIONAL",
            "minNotional": min_notional,
            "applyMinToMarket": True,
            "maxNotional": "9000000",
            "applyMaxToMarket": False,
        },
    ]


def binance_symbol(symbol: str, quote: str, status: str = "TRADING") -> dict[str, object]:
    """Une paire (``BTCEUR``, cotée ``EUR``) avec les filtres par défaut."""
    return {
        "symbol": symbol,
        "status": status,
        "baseAsset": symbol.removesuffix(quote),
        "quoteAsset": quote,
        "filters": binance_filters(),
    }


def exchange_info(symbols: list[dict[str, object]], server_time: int = T0) -> dict[str, object]:
    """Réponse exchangeInfo minimale contenant ``symbols``."""
    return {
        "timezone": "UTC",
        "serverTime": server_time,
        "rateLimits": [],
        "exchangeFilters": [],
        "symbols": symbols,
    }


# --- bougies Binance (archives spot) --------------------------------------------------------

KLINE_DAY_MS = 86_400_000


def kline_row(t_ms: int, close: float = 100.0, *, us: bool = False, **kw: float) -> str:
    """Bougie 1d cohérente (open 100, high 110, low 90) ; ``kw`` écrase un champ ; ``us`` :
    horodatages en µs comme les archives depuis 2025."""
    v = {"open": 100.0, "high": 110.0, "low": 90.0, "close": close, "vol": 2.0, **kw}
    k = 1000 if us else 1
    return (
        f"{t_ms * k},{v['open']},{v['high']},{v['low']},{v['close']},{v['vol']},"
        f"{(t_ms + KLINE_DAY_MS) * k - 1},{v['vol'] * close},7,1.0,{close},0"
    )


def kline_csv(*rows: str) -> bytes:
    return ("\n".join(rows) + "\n").encode()


def kline_archive(
    paths: DataPaths, kind: str, name: str, data: bytes, symbol: str = "BTCEUR"
) -> Path:
    """Archive ``spot/{kind}/klines/{symbol}/1d/{name}.zip`` de ``raw/binance_vision``."""
    path = paths.raw_archive("binance_vision", f"spot/{kind}/klines/{symbol}/1d/{name}.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}.csv", data)
    return path
