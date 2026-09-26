"""Tests de qlab.data.binance_vision : listage S3 paginé, CHECKSUM, sans réseau."""

from __future__ import annotations

import pytest
from fakes import NS, FakeS3, s3_xml

from qlab.core.errors import ExchangeError
from qlab.data import binance_vision as bv

LIST = "https://s3.example/bucket"


FILES = {
    "data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-01.zip": 100,
    "data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-01.zip.CHECKSUM": 88,
    "data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-02.zip": 110,
    "data/spot/monthly/klines/BTCEUR/1h/BTCEUR-1h-2024-01.zip": 900,
    "data/spot/monthly/klines/ETHEUR/1d/ETHEUR-1d-2024-01.zip": 95,
    "data/spot/monthly/klines/OLDEUR/1d/OLDEUR-1d-2019-05.zip": 50,
}


def test_list_subdirs() -> None:
    s3 = FakeS3(FILES)
    assert bv.list_subdirs(s3, LIST, bv.SPOT_KLINES_MONTHLY) == ["BTCEUR", "ETHEUR", "OLDEUR"]


def test_list_archives_excludes_checksums() -> None:
    files = bv.list_archives(FakeS3(FILES), LIST, "data/spot/monthly/klines/BTCEUR/1d/")
    assert files == [
        bv.ArchiveFile("data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-01.zip", 100),
        bv.ArchiveFile("data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-02.zip", 110),
    ]
    assert files[0].relative_key == "spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-01.zip"


def test_filename_prefix_selects_one_month() -> None:
    files = bv.list_archives(
        FakeS3(FILES), LIST, "data/spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-02"
    )
    assert [f.size for f in files] == [110]


@pytest.mark.parametrize("page_size", [1, 2, 3])
def test_pagination(page_size: int) -> None:
    """Pages de 1, 2 ou 3 entrées : même résultat qu'une seule page."""
    many = {f"data/spot/monthly/klines/S{i:02d}EUR/1d/x.zip": i for i in range(7)}
    s3 = FakeS3(many, page_size)
    assert bv.list_subdirs(s3, LIST, bv.SPOT_KLINES_MONTHLY) == [f"S{i:02d}EUR" for i in range(7)]
    assert len(bv.list_archives(FakeS3(many, page_size), LIST, bv.SPOT_KLINES_MONTHLY)) == 7
    assert s3.calls == -(-7 // page_size)  # ceil(7 / page_size) pages


def test_empty_prefix() -> None:
    assert bv.list_archives(FakeS3(FILES), LIST, "data/nothing/") == []
    assert bv.list_subdirs(FakeS3(FILES), LIST, "data/nothing/") == []


def test_bad_listing() -> None:
    with pytest.raises(ExchangeError, match="illisible"):
        bv.list_archives(lambda url: b"<html>", LIST, "data/")
    stuck = s3_xml("data/", [("data/a.zip", 1)], [], truncated=True, next_marker="")
    with pytest.raises(ExchangeError, match="bloquée"):
        bv.list_subdirs(lambda url: stuck, LIST, "data/")
    no_size = (
        f'<ListBucketResult xmlns="{NS}"><IsTruncated>false</IsTruncated><Contents>'
        "<Key>data/a.zip</Key></Contents></ListBucketResult>"
    )
    with pytest.raises(ExchangeError, match="taille absente"):
        bv.list_archives(lambda url: no_size.encode(), LIST, "data/")


def test_parse_checksum() -> None:
    sha = "474c1ce6fbb09e42cfc7231fee249aecc58af2fb5918570ffeba37998926b4a4"
    assert (
        bv.parse_checksum(f"{sha}  BTCUSDT-1d-2024-01.zip\n".encode(), "BTCUSDT-1d-2024-01.zip")
        == sha
    )
    with pytest.raises(ExchangeError, match="attendu"):
        bv.parse_checksum(f"{sha}  AUTRE.zip".encode(), "BTCUSDT-1d-2024-01.zip")
    with pytest.raises(ExchangeError, match="illisible"):
        bv.parse_checksum(b"<html>404</html>", "x.zip")


def test_file_url() -> None:
    assert bv.file_url("https://data.binance.vision/", "data/a.zip") == (
        "https://data.binance.vision/data/a.zip"
    )


def test_file_url_encodes_non_ascii_symbols() -> None:
    """Paire réelle au nom chinois (audit 2026-09-26 : le téléchargement complet plantait)."""
    url = bv.file_url("https://data.binance.vision", "data/spot/币安人生USDT/币安人生USDT-1d.zip")
    assert url.isascii()
    assert url == (
        "https://data.binance.vision/data/spot/%E5%B8%81%E5%AE%89%E4%BA%BA%E7%94%9FUSDT/"
        "%E5%B8%81%E5%AE%89%E4%BA%BA%E7%94%9FUSDT-1d.zip"
    )
