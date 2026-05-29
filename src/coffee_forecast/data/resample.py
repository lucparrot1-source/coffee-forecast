import argparse
import logging
import os
import sqlite3
import traceback
from datetime import date

import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)


def resample(conn: sqlite3.Connection, as_of: date | None = None) -> None:
    cutoff = (as_of or date.today()).replace(day=1).isoformat()[:7]  # 'YYYY-MM'

    rows = conn.execute(
        "SELECT date, symbol, adj_close FROM prices"
        " WHERE adj_close IS NOT NULL AND strftime('%Y-%m', date) < ?",
        (cutoff,),
    ).fetchall()

    if not rows:
        log.info("No daily prices found — skipping resample")
        return

    df = pd.DataFrame(rows, columns=["date", "symbol", "adj_close"])
    df["year_month"] = df["date"].str[:7]

    if df.empty:
        log.info("No complete months to resample")
        return

    monthly = df.groupby(["year_month", "symbol"], as_index=False)["adj_close"].mean()
    monthly["date"] = monthly["year_month"] + "-01"

    records = monthly[["date", "symbol", "adj_close"]].values.tolist()
    conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)"
        " ON CONFLICT(date, symbol) DO UPDATE SET adj_close = excluded.adj_close",
        records,
    )
    conn.commit()

    for sym, count in monthly.groupby("symbol").size().items():
        log.info("%s: %d monthly rows upserted", sym, count)
    log.info("Total: %d monthly rows upserted", len(records))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Resample daily prices to monthly")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)
    resample(conn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
