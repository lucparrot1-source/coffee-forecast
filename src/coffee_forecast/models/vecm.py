import argparse
import json
import logging
import os
import sqlite3
import traceback
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    VECMResults,
    select_order,
)

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)

ENDOG_SYMBOLS = ["KC=F", "RM=F"]
EXOG_SYMBOLS = ["BRL=X", "VND=X", "IDR=X", "DX-Y.NYB"]


def load_aligned_data(
    conn: sqlite3.Connection, max_date: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices_monthly, inner-join on common dates, log-transform.

    Returns (endog_df, exog_df) both on log scale, or (empty, empty) if no data.
    """
    all_symbols = ENDOG_SYMBOLS + EXOG_SYMBOLS
    date_filter = " AND date <= ?" if max_date is not None else ""
    extra_param: tuple[str, ...] = (max_date,) if max_date is not None else ()
    df = pd.read_sql(
        "SELECT date, symbol, adj_close FROM prices_monthly"
        f" WHERE symbol IN ({','.join('?' * len(all_symbols))}){date_filter}"
        " ORDER BY date",
        conn,
        params=tuple(all_symbols) + extra_param,
    )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna().sort_index()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    log_wide = wide.apply(np.log)
    endog_out = log_wide[ENDOG_SYMBOLS]
    exog_out = log_wide[EXOG_SYMBOLS]
    return endog_out, exog_out


def select_lag_order(endog: pd.DataFrame, exog: pd.DataFrame, maxlags: int = 12) -> int:
    """Return AIC-optimal VAR lag order (minimum 1) for VECM pre-selection."""
    res = select_order(endog.values, maxlags=maxlags, deterministic="co", exog=exog.values)
    return max(1, int(res.aic))


def fit_vecm(endog: pd.DataFrame, exog: pd.DataFrame, lag_order: int) -> VECMResults:
    """Fit a VECM with coint_rank=1 and exogenous drivers.

    k_ar_diff = lag_order - 1: number of lagged-difference terms.
    deterministic='co': constant restricted to cointegration relation.
    """
    model = VECM(
        endog.values,
        k_ar_diff=lag_order - 1,
        coint_rank=1,
        exog=exog.values,
        deterministic="co",
    )
    return model.fit()


def extract_residuals(result: VECMResults, endog: pd.DataFrame) -> pd.DataFrame:
    """Return long-format DataFrame of in-sample residuals (log scale).

    VECM residuals have fewer rows than endog: the first k_ar_diff rows are consumed
    by lagged differences. Dates are aligned from the tail of endog.index.
    """
    n_resid = result.resid.shape[0]
    dates = endog.index[-n_resid:]
    rows = []
    for i, dt in enumerate(dates):
        for j, sym in enumerate(ENDOG_SYMBOLS):
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": sym,
                "residual": float(result.resid[i, j]),
            })
    return pd.DataFrame(rows)


def generate_forecasts(
    result: VECMResults,
    endog_cols: list[str],
    exog_last_row: "np.ndarray[object, np.dtype[np.float64]]",
    steps: int = 3,
) -> pd.DataFrame:
    """Produce point forecasts at horizons 1..steps.

    Naïve exog assumption: exchange rates stay at their last observed level for all
    forecast steps. statsmodels VECM takes exog levels (not changes), so we tile the
    last observed row. Passing zeros here would imply rates collapse to zero.
    Forecasts are on log scale; point_forecast = exp(log_forecast).
    """
    exog_fc = np.tile(exog_last_row, (steps, 1))
    fc = result.predict(steps=steps, exog_fc=exog_fc)  # shape (steps, n_endog)
    rows = []
    for h in range(steps):
        for j, sym in enumerate(endog_cols):
            log_fc = float(fc[h, j])
            rows.append({
                "horizon": h + 1,
                "symbol": sym,
                "log_forecast": log_fc,
                "point_forecast": float(np.exp(log_fc)),
            })
    return pd.DataFrame(rows)


def write_run(
    conn: sqlite3.Connection, params: dict[str, object], metrics: dict[str, object]
) -> int:
    """Insert a model run record and return its id."""
    cur = conn.execute(
        "INSERT INTO model_runs"
        " (run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            "vecm",
            params["train_start"],
            params["train_end"],
            json.dumps(params),
            json.dumps(metrics),
            "success",
        ),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return int(cur.lastrowid)


def write_forecasts(
    conn: sqlite3.Connection, run_id: int, forecasts_df: pd.DataFrame, forecast_date: str
) -> None:
    """Insert forecast rows, computing target_date from forecast_date + horizon."""
    records = []
    for _, row in forecasts_df.iterrows():
        h = int(row["horizon"])
        target_date = (
            pd.Timestamp(forecast_date) + pd.DateOffset(months=h)
        ).strftime("%Y-%m-%d")
        records.append(
            (run_id, forecast_date, target_date, h, row["symbol"], row["point_forecast"])
        )
    conn.executemany(
        "INSERT OR REPLACE INTO forecasts"
        " (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()


def write_residuals(conn: sqlite3.Connection, run_id: int, residuals_df: pd.DataFrame) -> None:
    """Insert residual rows for a given run."""
    records = [
        (run_id, r["date"], r["symbol"], r["residual"])
        for _, r in residuals_df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO vecm_residuals (run_id, date, symbol, residual)"
        " VALUES (?, ?, ?, ?)",
        records,
    )
    conn.commit()


def run_vecm_model(conn: sqlite3.Connection, max_date: str | None = None) -> int:
    """Orchestrate load → lag-select → fit → residuals → forecasts → write DB.

    Returns the model_runs.id of the completed run, or -1 if no data found.
    """
    endog, exog = load_aligned_data(conn, max_date=max_date)
    if endog.empty:
        log.warning("No aligned monthly data found for all 6 symbols — skipping VECM")
        return -1

    lag_order = select_lag_order(endog, exog)
    log.info("Selected lag order: %d (AIC)", lag_order)

    result = fit_vecm(endog, exog, lag_order)
    log.info("VECM fitted: llf=%.4f, n_obs=%d", result.llf, len(endog))

    residuals_df = extract_residuals(result, endog)
    forecasts_df = generate_forecasts(result, endog.columns.tolist(), exog.values[-1])

    train_start = endog.index[0].strftime("%Y-%m-%d")
    train_end = endog.index[-1].strftime("%Y-%m-%d")

    params: dict[str, object] = {
        "lag_order": lag_order,
        "coint_rank": 1,
        "exog_symbols": exog.columns.tolist(),
        "train_start": train_start,
        "train_end": train_end,
        "n_obs": len(endog),
    }
    metrics: dict[str, object] = {"log_likelihood": float(result.llf)}

    run_id = write_run(conn, params, metrics)
    write_forecasts(conn, run_id, forecasts_df, train_end)
    write_residuals(conn, run_id, residuals_df)

    log.info(
        "VECM run complete: run_id=%d, forecasts=%d, residuals=%d",
        run_id, len(forecasts_df), len(residuals_df),
    )
    return run_id


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fit VECM and write forecasts to DB")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    args = parser.parse_args()
    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db
    conn = get_connection()
    ensure_schema(conn)
    run_vecm_model(conn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
