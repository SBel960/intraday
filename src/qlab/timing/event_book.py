"""Événements → colonnes du panel horaire des perpétuels (signaux d'événements, phase 9).

Pour chaque type d'annonce (``events/classify.py`` : ``delisting``, ``listing``), le contrat
``{actif}{quote}`` reçoit une colonne ``{type}_age`` : nombre de barres depuis la **barre de
signal** de la dernière annonce, nulle avant toute annonce. La barre de signal est la première
barre horaire dont la clôture suit l'heure effective de l'annonce (réception, ou publication
+ marge pour l'historique) : l'entrée se fait à l'ouverture suivante, jamais avant que
l'annonce soit connue.

Une annonce ne compte que si le contrat est **éligible** à sa barre de signal (univers liquide
point-in-time, ``timing/panel.py``) : pas d'entrée sur un contrat illiquide, même si la
position est ensuite tenue par un portefeuille persistant.

Les signaux ``delisting_short`` / ``listing_long`` (``timing/hourly_signals.py``) valent −1 / +1
pendant les ``k`` barres qui suivent la barre de signal, 0 sinon.
"""

from __future__ import annotations

import polars as pl

from qlab.core.timeutils import MS_PER_HOUR

KINDS = ("delisting", "listing")


def age_column(kind: str) -> str:
    return f"{kind}_age"


def signal_times(panel: pl.DataFrame, events: pl.DataFrame, kind: str, quote: str) -> pl.DataFrame:
    """(``symbol``, ``signal_ms``) des annonces ``kind`` dont le contrat est éligible à la
    barre de signal (première barre dont la clôture est postérieure à l'heure effective)."""
    ev = events.filter(pl.col("kind") == kind).select(
        (pl.col("base") + quote).alias("symbol"),
        (
            (pl.col("effective_ms") + MS_PER_HOUR - 1) // MS_PER_HOUR * MS_PER_HOUR - MS_PER_HOUR
        ).alias("signal_ms"),
    )
    ok = panel.filter(pl.col("eligible")).select("symbol", pl.col("t_ms").alias("signal_ms"))
    return ev.join(ok, on=["symbol", "signal_ms"], how="semi").unique().sort("signal_ms")


def with_event_ages(panel: pl.DataFrame, events: pl.DataFrame, quote: str) -> pl.DataFrame:
    """Ajoute ``{type}_age`` (barres depuis la dernière barre de signal) pour chaque type."""
    out = panel.with_row_index("_row")
    for kind in KINDS:
        times = signal_times(panel, events, kind, quote)
        ages = (
            out.select("_row", "symbol", "t_ms")
            .sort("t_ms")
            .join_asof(
                times, left_on="t_ms", right_on="signal_ms", by="symbol", check_sortedness=False
            )  # les deux côtés sont triés par temps juste au-dessus
            .select(
                "_row",
                ((pl.col("t_ms") - pl.col("signal_ms")) // MS_PER_HOUR).alias(age_column(kind)),
            )
        )
        out = out.join(ages, on="_row", how="left")
    return out.sort("_row").drop("_row")
