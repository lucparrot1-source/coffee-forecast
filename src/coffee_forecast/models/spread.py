import argparse
import logging
import os
import sqlite3
import traceback

import numpy as np
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)


def compute_spread(wide: pd.DataFrame) -> pd.Series:
    """Return log(KC=F) - log(RM=F) as a monthly Series."""
    result: pd.Series = np.log(wide["KC=F"]) - np.log(wide["RM=F"])
    return result


def compute_zscore(s: pd.Series) -> pd.Series:
    """Expanding z-score: (s - expanding_mean) / expanding_std.

    Index 0 is NaN (need >= 2 points for std).
    """
    exp = s.expanding(min_periods=2)
    return (s - exp.mean()) / exp.std()


def fit_ar1(s: pd.Series) -> tuple[float, float]:
    """OLS AR(1) fit on spread series s. Returns (rho, half_life_months).

    half_life is nan when |rho| == 0 or |rho| >= 1 (non-stationary / explosive).
    """
    arr = np.asarray(s.values, dtype=float)
    y = arr[1:]
    x = arr[:-1]
    X = np.column_stack([np.ones_like(x), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = float(coefs[1])
    if 0.0 < abs(rho) < 1.0:
        half_life = -np.log(2) / np.log(abs(rho))
    else:
        half_life = float("nan")
    return rho, half_life


def generate_signal(z: pd.Series, entry: float = 2.0, exit_thresh: float = 0.5) -> pd.Series:
    """Stateful trading signal from z-score series.

    +1 = long spread, -1 = short spread, 0 = flat.
    NaN z-scores leave the current position unchanged.
    """
    signals: list[int] = []
    current = 0
    for zi in z:
        if not np.isnan(zi):
            if zi > entry:
                current = -1
            elif zi < -entry:
                current = 1
            elif abs(zi) < exit_thresh:
                current = 0
        signals.append(current)
    return pd.Series(signals, index=z.index, dtype=int)


def build_spread_df(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute spread, z-score, signal, and expanding half-life for all months."""
    spread = compute_spread(wide)
    z = compute_zscore(spread)
    sig = generate_signal(z)

    half_lives: list[float] = []
    for i in range(len(spread)):
        s_slice = spread.iloc[: i + 1]
        if len(s_slice) < 3:
            half_lives.append(float("nan"))
        else:
            _, hl = fit_ar1(s_slice)
            half_lives.append(hl)

    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(spread.index).strftime("%Y-%m-%d"),
            "spread": spread.values,
            "z_score": z.values,
            "signal": sig.values.astype(int),
            "half_life": half_lives,
        }
    )


def run_spread_model(conn: sqlite3.Connection) -> int:
    """Load monthly prices from DB, compute spread signals, upsert into spread_signals.

    Returns the number of rows written.
    """
    df = pd.read_sql(
        "SELECT date, symbol, adj_close FROM prices_monthly"
        " WHERE symbol IN ('KC=F', 'RM=F')"
        " ORDER BY date",
        conn,
    )
    if df.empty:
        log.warning("No monthly price data found for KC=F / RM=F — skipping spread model")
        return 0

    wide = df.pivot(index="date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna(subset=["KC=F", "RM=F"]).sort_index()

    if wide.empty:
        log.warning("No rows with both KC=F and RM=F present — skipping spread model")
        return 0

    result = build_spread_df(wide)

    records = [
        (
            row["date"],
            None if pd.isna(row["spread"]) else row["spread"],
            None if pd.isna(row["z_score"]) else row["z_score"],
            int(row["signal"]),
            None if pd.isna(row["half_life"]) else row["half_life"],
        )
        for _, row in result.iterrows()
    ]

    conn.executemany(
        "INSERT OR REPLACE INTO spread_signals (date, spread, z_score, signal, half_life)"
        " VALUES (?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    log.info("spread_signals: wrote %d rows", len(records))
    return len(records)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fit spread model and write signals to DB")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)
    run_spread_model(conn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
