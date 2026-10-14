"""Tests de qlab.events.classify : vrais titres relevés le 2026-10-14, heure effective."""

from __future__ import annotations

import pytest

from qlab.events.classify import RULES_VERSION, Classified, classify, events_table


@pytest.mark.parametrize(
    ("title", "catalog", "expected"),
    [
        (
            "Binance Will Delist EPIC, QUICK on 2026-10-28",
            "delisting",
            ("delisting", ("EPIC", "QUICK")),
        ),
        (
            "Binance Will Delist BCD, CND, MTH, NCASH and YOYO on 2022-03-08",
            "delisting",
            ("delisting", ("BCD", "CND", "MTH", "NCASH", "YOYO")),
        ),
        ("Binance Will Delist BEAM on 2023-01-26", "delisting", ("delisting", ("BEAM",))),
        (
            "Binance Will Delist FTT Margin Pairs, USDⓈ-M Futures Contracts "
            "and Binance Flexible Products",
            "delisting",
            ("other", ()),
        ),
        ("Notice of Removal of Spot Trading Pairs - 2026-10-02", "delisting", ("pair_removal", ())),
        (
            "Binance Futures Will Delist USDⓈ-M ALPACAUSDT Perpetual Contract",
            "delisting",
            ("perp_delisting", ("ALPACA",)),
        ),
        (
            "Binance Will List Hyperliquid (HYPE) with Seed Tag Applied",
            "listing",
            ("listing", ("HYPE",)),
        ),
        ("Binance Will List OMG Market", "listing", ("listing", ("OMG",))),
        ("Binance Lists LINK", "listing", ("listing", ("LINK",))),
        (
            "Binance Futures Will Launch USDⓈ-Margined CTUSDT Perpetual Contract (2026-10-01)",
            "listing",
            ("perp_launch", ()),
        ),
        ("Binance Alpha Will Remove SUP, CUDIS (2026-10-09)", "delisting", ("other", ())),
        ("Binance Will Delist X on 2026-01-01", "autre_rubrique", ("other", ())),
    ],
)
def test_real_titles(title: str, catalog: str, expected: tuple[str, tuple[str, ...]]) -> None:
    assert classify(title, catalog) == Classified(*expected)


def test_effective_time_is_reception_else_publication_plus_margin() -> None:
    records = [
        {
            "title": "Binance Will Delist AA, BB on 2024-01-01",
            "catalog": "delisting",
            "published_ms": 1_000_000,
            "received_ms": None,
            "backfilled": True,
        },
        {
            "title": "Binance Lists CC",
            "catalog": "listing",
            "published_ms": 2_000_000,
            "received_ms": 2_000_500,
            "backfilled": False,
        },
        {
            "title": "Notice of Removal of Spot Trading Pairs",
            "catalog": "delisting",
            "published_ms": 3_000_000,
            "received_ms": None,
            "backfilled": True,
        },
    ]
    t = events_table(records, 300)
    assert t.select("kind", "base", "effective_ms").rows() == [
        ("delisting", "AA", 1_300_000),
        ("delisting", "BB", 1_300_000),
        ("listing", "CC", 2_000_500),
    ]
    assert t["rules_version"].to_list() == [RULES_VERSION] * 3
    assert events_table([], 300).is_empty()
