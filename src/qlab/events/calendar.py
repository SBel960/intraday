"""Calendrier des communiqués du FOMC (Fed) : dates versionnées, contrôlées, en heure UTC.

Les dates des réunions **programmées** sont dans ``config/events.yaml`` (``fomc_statements``),
relevées sur la page officielle de la Fed et versionnées : elles sont connues des mois à
l'avance, donc utilisables en backtest sans fuite. Les réunions non programmées (mars 2020),
annulées ou les votes par écrit n'en font pas partie (rien n'était connu d'avance). Le
communiqué tombe à ``fomc_statement_time_et`` (14:00), heure de New York, heure d'été comprise
(``zoneinfo``).

``check`` relit la page de la Fed (calendrier courant + pages historiques des années plus
anciennes) et signale tout écart avec la config ; il ne la corrige jamais (spec §1).
``risk_windows`` : fenêtres sans nouvelle position autour de chaque communiqué.

Commande : ``python -m qlab.events.calendar --config config check``
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import EventsConfig

NEW_YORK = ZoneInfo("America/New_York")
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
}
MONTH_ALIASES = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}
_CURRENT = re.compile(
    r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", re.S
)
_HISTORICAL = re.compile(r"<h5[^>]*>([A-Z][a-z]+) (\d{1,2})-(\d{1,2}) Meeting - (\d{4})</h5>")
_SKIPPED = ("unscheduled", "notation", "cancelled")
HEADERS = {"User-Agent": "Mozilla/5.0 (qlab research)"}


def statement_ms(day: str, time_et: str) -> int:
    """Heure UTC (ms) d'un communiqué publié le ``day`` à ``time_et``, heure de New York."""
    hours, minutes = (int(x) for x in time_et.split(":"))
    local = datetime.combine(date.fromisoformat(day), time(hours, minutes), tzinfo=NEW_YORK)
    return int(local.timestamp() * 1000)


def _last_month(label: str) -> int:
    name = label.rsplit("/", maxsplit=1)[-1].strip()
    return MONTHS[MONTH_ALIASES.get(name, name)]


def parse_current(html: str) -> list[str]:
    """Jours des communiqués (dernier jour de chaque réunion programmée) de la page courante."""
    out = []
    years = re.split(r"(\d{4}) FOMC Meetings", html)
    for year, section in zip(years[1::2], years[2::2], strict=True):
        for month, days in _CURRENT.findall(section):
            if any(s in days for s in _SKIPPED) or not re.match(r"^\d{1,2}-\d{1,2}\*?$", days):
                continue
            last = int(days.rstrip("*").split("-")[1])
            out.append(date(int(year), _last_month(month), last).isoformat())
    return sorted(out)


def parse_historical(html: str) -> list[str]:
    """Jours des communiqués des réunions programmées d'une page historique (une année)."""
    return sorted(
        date(int(y), MONTHS[m], int(d2)).isoformat() for m, _, d2, y in _HISTORICAL.findall(html)
    )


def risk_windows(cfg: EventsConfig) -> list[tuple[int, int, str]]:
    """(début, fin, raison) : de ``fomc_risk_before_min`` avant à ``fomc_risk_after_min``
    après chaque communiqué."""
    out = []
    for day in cfg.fomc_statements:
        t = statement_ms(day, cfg.fomc_statement_time_et)
        out.append(
            (
                t - cfg.fomc_risk_before_min * 60_000,
                t + cfg.fomc_risk_after_min * 60_000,
                f"communiqué du FOMC du {day}",
            )
        )
    return out


def _check(ctx: Context) -> int:
    cfg = ctx.config.events
    page = http.get(cfg.fomc_page_url, headers=HEADERS).body.decode("utf-8", "replace")
    official = set(parse_current(page))
    covered = {d[:4] for d in official}
    for year in sorted({d[:4] for d in cfg.fomc_statements} - covered):
        url = cfg.fomc_history_url.format(year=year)
        official |= set(
            parse_historical(http.get(url, headers=HEADERS).body.decode("utf-8", "replace"))
        )
    mine = set(cfg.fomc_statements)
    span = {d for d in official if min(mine) <= d <= max(mine)}
    missing, extra = sorted(span - mine), sorted(mine - official)
    for d in missing:
        print(f"absent de la config : communiqué du {d}")
    for d in extra:
        print(f"inconnu de la Fed : {d}")
    print(f"{len(mine)} communiqués en config ; {len(missing)} manquant(s), {len(extra)} en trop")
    return 1 if missing or extra else 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="comparer la config à la page de la Fed (sans la modifier)")


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.events.calendar",
        description="Calendrier des communiqués du FOMC",
        component="events_calendar",
        add_arguments=_add_arguments,
        action=_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
