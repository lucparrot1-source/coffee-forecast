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


def run_hybrid_model(
    conn: sqlite3.Connection,
    vecm_run_id: int,
    gamlss_run_id: int,
) -> int:
    """Orchestrate: load VECM forecasts + GAMLSS quantiles → combine → write DB.

    Returns the model_runs.id of the hybrid run, or -1 if skipped due to
    missing data.
    """
    vecm_df = load_vecm_forecasts(conn, vecm_run_id)
    if vecm_df.empty:
        log.warning("No VECM forecasts for run_id=%d — skipping hybrid", vecm_run_id)
        return -1

    gamlss_df = load_gamlss_quantiles(conn, gamlss_run_id)
    if gamlss_df.empty:
        log.warning("No GAMLSS quantiles for run_id=%d — skipping hybrid", gamlss_run_id)
        return -1

    regime = get_current_regime(conn)
    log.info("Current volatility regime: %s", regime)

    combined_df = combine_forecasts(vecm_df, gamlss_df, regime)

    quantile_cols = ["p10", "p25", "p50", "p75", "p90"]
    nan_count = int(combined_df[quantile_cols].isna().sum().sum())
    if nan_count > 0:
        log.warning(
            "Combined forecasts contain %d NaN quantile values — "
            "GAMLSS may have failed to converge for some symbol/regime combinations",
            nan_count,
        )

    run_meta = conn.execute(
        "SELECT train_end FROM model_runs WHERE id = ?", (vecm_run_id,)
    ).fetchone()
    if run_meta is None:
        log.error("VECM run_id=%d not found in model_runs — cannot proceed", vecm_run_id)
        return -1
    forecast_date = run_meta[0]

    cur = conn.execute(
        "INSERT INTO model_runs"
        " (run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            "hybrid",
            forecast_date,
            forecast_date,
            json.dumps({
                "vecm_run_id": vecm_run_id,
                "gamlss_run_id": gamlss_run_id,
                "regime": regime,
            }),
            "{}",
            "pending",
        ),
    )
    conn.commit()
    assert cur.lastrowid is not None
    run_id = int(cur.lastrowid)

    try:
        write_hybrid_forecasts(conn, run_id, combined_df, forecast_date)
    except Exception:
        conn.execute("UPDATE model_runs SET status = 'failed' WHERE id = ?", (run_id,))
        conn.commit()
        raise

    conn.execute("UPDATE model_runs SET status = 'success' WHERE id = ?", (run_id,))
    conn.commit()
    log.info(
        "Hybrid run complete: run_id=%d, regime=%s, rows=%d",
        run_id, regime, len(combined_df),
    )
    return run_id


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Combine VECM point forecasts with GAMLSS distributions"
    )
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    parser.add_argument(
        "--vecm-run-id",
        type=int,
        default=None,
        help="VECM model_runs.id to use (default: latest successful VECM run)",
    )
    parser.add_argument(
        "--gamlss-run-id",
        type=int,
        default=None,
        help="GAMLSS model_runs.id to use (default: latest successful GAMLSS run)",
    )
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)

    def _latest_run(model_type: str) -> int:
        row = conn.execute(
            "SELECT id FROM model_runs WHERE model_type = ? AND status = 'success'"
            " ORDER BY id DESC LIMIT 1",
            (model_type,),
        ).fetchone()
        if row is None:
            # Raise so the __main__ guard fires the Resend alert and exits non-zero.
            raise RuntimeError(
                f"No successful {model_type} run found — run that model first"
            )
        return int(row[0])

    vecm_run_id = args.vecm_run_id if args.vecm_run_id is not None else _latest_run("vecm")
    gamlss_run_id = args.gamlss_run_id if args.gamlss_run_id is not None else _latest_run("gamlss")

    log.info("Using vecm_run_id=%d, gamlss_run_id=%d", vecm_run_id, gamlss_run_id)
    run_hybrid_model(conn, vecm_run_id, gamlss_run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
