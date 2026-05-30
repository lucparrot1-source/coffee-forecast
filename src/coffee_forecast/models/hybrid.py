import argparse
import json
import logging
import os
import sqlite3
import traceback
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging
from coffee_forecast.models.gamlss import compute_regime_labels

log = logging.getLogger(__name__)


def load_vecm_forecasts(conn: sqlite3.Connection, vecm_run_id: int) -> pd.DataFrame:
    """Load point forecasts from a VECM run.

    Returns DataFrame with columns: horizon, symbol, point_forecast.
    """
    return pd.read_sql(
        "SELECT horizon, symbol, point_forecast FROM forecasts WHERE run_id = ?",
        conn,
        params=(vecm_run_id,),
    )


def load_gamlss_quantiles(conn: sqlite3.Connection, gamlss_run_id: int) -> pd.DataFrame:
    """Load SHASH residual quantiles from a GAMLSS run.

    Returns DataFrame with columns: symbol, regime, q10, q25, q50, q75, q90.
    Quantiles are in log space (residual scale, not price scale).
    """
    return pd.read_sql(
        "SELECT symbol, regime, q10, q25, q50, q75, q90"
        " FROM gamlss_params WHERE run_id = ?",
        conn,
        params=(gamlss_run_id,),
    )


def get_current_regime(conn: sqlite3.Connection) -> str:
    """Return the volatility regime label for the latest available KC=F month.

    Reuses the same 12-month rolling-vol logic as GAMLSS training so that
    the regime used at inference time is consistent with the one used at fit time.
    Raises ValueError if no KC=F price data is present.
    """
    regime_series = compute_regime_labels(conn)
    if regime_series.empty:
        raise ValueError("Cannot determine current regime: no KC=F price data")
    return str(regime_series.iloc[-1])


def combine_forecasts(
    vecm_df: pd.DataFrame,
    gamlss_df: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    """Combine VECM point forecasts with GAMLSS quantile offsets.

    GAMLSS quantiles are log-space residuals. Because the VECM point forecast
    is exp(log_vecm_forecast), the combined price quantile is:
        price_qX = point_forecast * exp(gamlss_qX)

    Returns DataFrame: horizon, symbol, point_forecast, p10, p25, p50, p75, p90.
    Raises ValueError if the given regime or any required symbol is absent from gamlss_df.
    """
    regime_params = gamlss_df[gamlss_df["regime"] == regime]
    if regime_params.empty:
        raise ValueError(f"No GAMLSS params for regime '{regime}'")
    regime_params = regime_params.set_index("symbol")

    rows = []
    for _, row in vecm_df.iterrows():
        sym = str(row["symbol"])
        if sym not in regime_params.index:
            raise ValueError(f"No GAMLSS params for symbol '{sym}' in regime '{regime}'")
        pf = float(row["point_forecast"])
        q = regime_params.loc[sym]
        rows.append({
            "horizon": int(row["horizon"]),
            "symbol": sym,
            "point_forecast": pf,
            "p10": pf * np.exp(float(q["q10"])),
            "p25": pf * np.exp(float(q["q25"])),
            "p50": pf * np.exp(float(q["q50"])),
            "p75": pf * np.exp(float(q["q75"])),
            "p90": pf * np.exp(float(q["q90"])),
        })
    return pd.DataFrame(rows)


def write_hybrid_forecasts(
    conn: sqlite3.Connection,
    run_id: int,
    combined_df: pd.DataFrame,
    forecast_date: str,
) -> None:
    """Write combined forecast rows (point + quantiles) to the forecasts table."""
    records = []
    for _, row in combined_df.iterrows():
        h = int(row["horizon"])
        target_date = (
            pd.Timestamp(forecast_date) + pd.DateOffset(months=h)
        ).strftime("%Y-%m-%d")
        records.append((
            run_id, forecast_date, target_date, h, str(row["symbol"]),
            float(row["point_forecast"]),
            float(row["p10"]), float(row["p25"]), float(row["p50"]),
            float(row["p75"]), float(row["p90"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO forecasts"
        " (run_id, forecast_date, target_date, horizon, symbol,"
        "  point_forecast, p10, p25, p50, p75, p90)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    log.info("Wrote %d hybrid forecast rows for run_id=%d", len(records), run_id)


def run_hybrid_model(conn: sqlite3.Connection, vecm_run_id: int, gamlss_run_id: int) -> int:
    raise NotImplementedError
