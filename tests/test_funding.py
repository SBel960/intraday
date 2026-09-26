"""Tests de qlab.longterm.funding : CSV construits à la main, contrôles, somme par jour, CLI."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.longterm import funding as fu

T0 = date_to_ms("2024-01-01")
H = MS_PER_HOUR
HEADER = "calc_time,funding_interval_hours,last_funding_rate"


def _csv(*rows: tuple[int, int, str]) -> bytes:
    return "\n".join([HEADER, *(f"{t},{h},{r}" for t, h, r in rows)]).encode() + b"\n"


def _archive(paths: DataPaths, symbol: str, name: str, data: bytes) -> Path:
    key = f"futures/um/monthly/fundingRate/{symbol}/{name}.zip"
    path = paths.raw_archive("binance_vision", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}.csv", data)
    return path


def test_parse_keeps_exact_time_and_interval() -> None:
    """Heure de calcul gardée à la ms près ; « 0E-8 » (notation Binance) = 0."""
    df = fu.parse_csv(_csv((T0 + 3, 8, "0.0001"), (T0 + 8 * H + 1, 8, "0E-8")))
    assert df.columns == list(fu.COLUMNS)
    assert df.rows() == [(T0 + 3, 8, 0.0001), (T0 + 8 * H + 1, 8, 0.0)]
    assert fu.parse_csv(b"").columns == list(fu.COLUMNS)


def test_two_settlements_a_few_ms_apart_are_both_kept() -> None:
    """Action tokenisée TSMUSDT, 2026-06-11 : règlements à …001 et …012 ms, taux différents."""
    df = fu.merge([fu.parse_csv(_csv((T0 + 1, 1, "0E-8"), (T0 + 12, 1, "-0.0023391")))])
    assert df.height == 2


def test_merge_drops_identical_duplicates_and_rejects_contradictions() -> None:
    a = fu.parse_csv(_csv((T0, 8, "0.0001"), (T0 + 8 * H, 8, "0.0002")))
    b = fu.parse_csv(_csv((T0 + 8 * H, 8, "0.0002"), (T0 + 16 * H, 8, "0.0003")))
    assert fu.merge([a, b])["rate"].to_list() == [0.0001, 0.0002, 0.0003]
    c = fu.parse_csv(_csv((T0, 8, "0.0009")))
    with pytest.raises(DataError, match="contradictoires"):
        fu.merge([a, c])
    assert fu.merge([]).height == 0


def test_daily_sum_by_hand() -> None:
    """Jour 1 : 3 échéances 0,0001 + 0,0002 − 0,0001 = 0,0002 ; jour 2 : passage à 4 h,
    6 échéances de 0,00005 = 0,0003."""
    day1 = [(T0 + k * 8 * H, 8, r) for k, r in enumerate(("0.0001", "0.0002", "-0.0001"))]
    day2 = [(T0 + 24 * H + k * 4 * H, 4, "0.00005") for k in range(6)]
    out = fu.daily(fu.parse_csv(_csv(*day1, *day2)))
    assert out["date_ms"].to_list() == [T0, T0 + 24 * H]
    assert out["funding_1d"].to_list() == pytest.approx([0.0002, 0.0003])
    assert out["n_events"].to_list() == [3, 6]


@pytest.mark.parametrize(
    ("row", "msg"),
    [
        ((5, 8, "0.0001"), "horodatage"),  # 1970
        (((T0 * 1000), 8, "0.0001"), "horodatage"),  # µs : non attendu ici
        ((T0, 0, "0.0001"), "intervalle"),
        ((T0, 8, "nan"), "non fini"),
    ],
)
def test_invalid_rows(row: tuple[int, int, str], msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        fu.parse_csv(_csv(row))


def test_read_archive_errors(tmp_path: Path) -> None:
    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as z:
        z.writestr("a.csv", _csv((T0, 8, "0")))
        z.writestr("b.csv", _csv((T0, 8, "0")))
    with pytest.raises(DataError, match="un seul fichier"):
        fu.read_archive(two)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("x.csv", _csv((T0, 8, "beaucoup")))
    with pytest.raises(DataError, match=r"bad\.zip : CSV illisible"):
        fu.read_archive(bad)
    with pytest.raises(DataError, match="absent"):
        fu.load(DataPaths(tmp_path), "BTCUSDT")


def test_cli_builds_and_isolates_errors(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = DataPaths(config_dir.parent / "data")
    assert fu.main(["--config", str(config_dir), "build"]) == 0  # aucune archive : rien à faire
    _archive(paths, "BTCUSDT", "BTCUSDT-fundingRate-2024-01", _csv((T0, 8, "0.0001")))
    _archive(paths, "BTCUSDT", "BTCUSDT-fundingRate-2024-02", _csv((T0 + 8 * H, 8, "0.0002")))
    _archive(paths, "BADUSDT", "BADUSDT-fundingRate-2024-01", _csv((T0, 8, "beaucoup")))
    assert fu.main(["--config", str(config_dir), "build"]) == 1
    out = capsys.readouterr().out
    assert "Financement : 1 contrats, 2 échéances" in out
    assert "ERREUR BADUSDT : BADUSDT-fundingRate-2024-01.zip : CSV illisible" in out
    assert fu.load(paths, "BTCUSDT")["rate"].to_list() == [0.0001, 0.0002]
