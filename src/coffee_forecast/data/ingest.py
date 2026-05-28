import argparse
import logging
import os
import sqlite3
import traceback
from datetime import date, timedelta

import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.data.providers import TICKERS, PriceProvider, YahooProvider
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)

DEFAULT_START = date(2000, 1, 1)
_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


def _latest_date(conn: sqlite3.Connection, symbol: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(date) FROM prices WHERE symbol = ?", (symbol,)
    ).fetchone()
    return date.fromisoformat(row[0]) if row[0] else None


def ingest(
    conn: sqlite3.Connection,
    start_override: date | None = None,
    tickers: list[str] | None = None,
    provider: PriceProvider | None = None,
) -> None:
    _tickers = tickers if tickers is not None else TICKERS
    _provider: PriceProvider = provider if provider is not None else YahooProvider()
    today = date.today()

    for sym in _tickers:
        latest = _latest_date(conn, sym)
        if start_override is not None:
            fetch_start = start_override
        elif latest:
            fetch_start = latest + timedelta(days=1)
        else:
            fetch_start = DEFAULT_START

        if fetch_start > today:
            log.info("%s: already up to date", sym)
            continue

        log.info("%s: fetching %s → %s", sym, fetch_start, today)
        df = _provider.fetch([sym], fetch_start, today)

        if df.empty:
            log.info("%s: no data returned", sym)
            continue

        records = (
            df.astype(object).where(pd.notna(df), other=None)[_COLS].values.tolist()
        )
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO prices"
            " (date, symbol, open, high, low, close, volume, adj_close)"
            " VALUES (?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
        log.info("%s: inserted %d rows", sym, conn.total_changes - before)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest daily prices into SQLite")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Override fetch start date YYYY-MM-DD")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)
    ingest(conn, start_override=args.start)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
