import argparse
import json
import logging
import os
import sqlite3
import traceback
from datetime import UTC, datetime

import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging
from coffee_forecast.models.gamlss import run_gamlss_model
from coffee_forecast.models.hybrid import run_hybrid_model
from coffee_forecast.models.vecm import run_vecm_model

log = logging.getLogger(__name__)


def get_validation_dates(
    conn: sqlite3.Connection, min_train_months: int = 36
) -> list[str]:
    """Return sorted list of train_end dates for walk-forward validation.

    Each date is a month-start string (YYYY-MM-01). The first valid train_end
    is at index (min_train_months - 1) in the KC=F price series. The last valid
    train_end is one month before the most recent KC=F price, so that the h=1
    forecast always has an actual price to compare against.
    """
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices_monthly WHERE symbol = 'KC=F' ORDER BY date"
    ).fetchall()
    dates = [r[0] for r in rows]
    if len(dates) < min_train_months + 1:
        return []
    return dates[min_train_months - 1 : -1]


def load_actuals(
    conn: sqlite3.Connection, symbols: list[str], dates: list[str]
) -> pd.DataFrame:
    """Load actual prices from prices_monthly for given symbols and dates.

    Returns DataFrame with columns: date, symbol, actual.
    """
    if not dates or not symbols:
        return pd.DataFrame(columns=["date", "symbol", "actual"])
    placeholders_sym = ",".join("?" * len(symbols))
    placeholders_dt = ",".join("?" * len(dates))
    return pd.read_sql(
        f"SELECT date, symbol, adj_close AS actual FROM prices_monthly"
        f" WHERE symbol IN ({placeholders_sym}) AND date IN ({placeholders_dt})",
        conn,
        params=tuple(symbols) + tuple(dates),
    )


def collect_backtest_rows(
    conn: sqlite3.Connection,
    run_id: int,
    train_end: str,
) -> list[dict]:
    """Load forecasts for run_id and join with actuals from prices_monthly.

    Returns a list of dicts, one per (horizon, symbol). actual=None when the
    target date is not yet in the DB (forecast is for a future month).
    """
    forecasts_df = pd.read_sql(
        "SELECT id AS forecast_id, target_date, horizon, symbol,"
        " point_forecast, p10, p25, p50, p75, p90"
        " FROM forecasts WHERE run_id = ?",
        conn,
        params=(run_id,),
    )
    if forecasts_df.empty:
        return []

    target_dates = forecasts_df["target_date"].unique().tolist()
    symbols = forecasts_df["symbol"].unique().tolist()
    actuals_df = load_actuals(conn, symbols, target_dates)

    # Both columns are YYYY-MM-DD text strings from SQLite — no datetime coercion needed for merge.
    actuals_df = actuals_df.rename(columns={"date": "target_date"})

    merged = forecasts_df.merge(actuals_df, on=["target_date", "symbol"], how="left")

    rows = []
    for _, row in merged.iterrows():
        actual_val = row.get("actual")
        rows.append({
            "forecast_id": int(row["forecast_id"]),
            "train_end": train_end,
            "target_date": str(row["target_date"]),
            "horizon": int(row["horizon"]),
            "symbol": str(row["symbol"]),
            "actual": float(actual_val) if pd.notna(actual_val) else None,
            "point_forecast": float(row["point_forecast"]) if pd.notna(row["point_forecast"]) else None,
            "p10": float(row["p10"]) if pd.notna(row["p10"]) else None,
            "p25": float(row["p25"]) if pd.notna(row["p25"]) else None,
            "p50": float(row["p50"]) if pd.notna(row["p50"]) else None,
            "p75": float(row["p75"]) if pd.notna(row["p75"]) else None,
            "p90": float(row["p90"]) if pd.notna(row["p90"]) else None,
        })
    return rows


def write_backtest_results(
    conn: sqlite3.Connection,
    run_id: int,
    backtest_date: str,
    rows: list[dict],
) -> None:
    """Insert backtest result rows into backtest_results table."""
    records = [
        (
            run_id, backtest_date, r["train_end"], r["target_date"],
            r["horizon"], r["symbol"], r["actual"],
            r["point_forecast"], r["p10"], r["p25"], r["p50"], r["p75"], r["p90"],
        )
        for r in rows
    ]
    conn.executemany(
        "INSERT INTO backtest_results"
        " (run_id, backtest_date, train_end, target_date, horizon, symbol, actual,"
        "  point_forecast, p10, p25, p50, p75, p90)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    log.info("Wrote %d backtest_results rows for run_id=%d", len(records), run_id)


def write_accuracy_log_entries(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> None:
    """Write per-forecast accuracy rows to accuracy_log for rows that have an actual."""
    records = []
    now = datetime.now(UTC).isoformat()
    for r in rows:
        if r["actual"] is None:
            continue
        actual = r["actual"]
        pf = r["point_forecast"]
        p50 = r["p50"]
        p10 = r["p10"]
        p90 = r["p90"]

        mae = abs(actual - pf) if pf is not None else None
        mape = abs(actual - pf) / actual * 100.0 if pf is not None else None
        pinball_50 = 0.5 * abs(actual - p50) if p50 is not None else None
        if p10 is not None and p90 is not None:
            coverage_80: int | None = 1 if p10 <= actual <= p90 else 0
        else:
            coverage_80 = None
        records.append((
            now,
            r["forecast_id"],
            actual,
            r["horizon"],
            r["symbol"],
            mae,
            mape,
            pinball_50,
            coverage_80,
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO accuracy_log"
        " (logged_at, forecast_id, actual, horizon, symbol, mae, mape, pinball_50, coverage_80)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    log.info("Wrote %d accuracy_log rows", len(records))


def compute_summary_metrics(conn: sqlite3.Connection, run_id: int) -> dict:
    """Compute aggregate metrics from backtest_results for a given backtest run_id.

    Returns a flat dict keyed by "h{horizon}_{symbol}_{metric}". Only rows
    with non-null actual values are included.
    """
    df = pd.read_sql(
        "SELECT horizon, symbol, actual, point_forecast, p10, p25, p50, p75, p90"
        " FROM backtest_results WHERE run_id = ? AND actual IS NOT NULL",
        conn,
        params=(run_id,),
    )
    if df.empty:
        return {}

    metrics: dict = {}
    for (horizon, symbol), group in df.groupby(["horizon", "symbol"]):
        key = f"h{horizon}_{symbol}"
        valid_pf = group[group["point_forecast"].notna()]
        mae = float((valid_pf["actual"] - valid_pf["point_forecast"]).abs().mean()) if len(valid_pf) else None
        mape = float(((valid_pf["actual"] - valid_pf["point_forecast"]).abs() / valid_pf["actual"] * 100.0).mean()) if len(valid_pf) else None

        valid_80 = group[group["p10"].notna() & group["p90"].notna()]
        if len(valid_80):
            coverage_80 = float(
                ((valid_80["actual"] >= valid_80["p10"]) & (valid_80["actual"] <= valid_80["p90"])).mean()
            )
        else:
            coverage_80 = None

        valid_50 = group[group["p25"].notna() & group["p75"].notna()]
        if len(valid_50):
            coverage_50 = float(
                ((valid_50["actual"] >= valid_50["p25"]) & (valid_50["actual"] <= valid_50["p75"])).mean()
            )
        else:
            coverage_50 = None

        valid_p50 = group[group["p50"].notna()]
        pinball_50 = float((0.5 * (valid_p50["actual"] - valid_p50["p50"]).abs()).mean()) if len(valid_p50) else None

        metrics[f"{key}_mae"] = mae
        metrics[f"{key}_mape"] = mape
        metrics[f"{key}_coverage_80"] = coverage_80
        metrics[f"{key}_coverage_50"] = coverage_50
        metrics[f"{key}_pinball_50"] = pinball_50
        # n_windows = rows with a realized actual (regardless of whether point_forecast or quantiles are present)
        metrics[f"{key}_n_windows"] = int(len(group))

    return metrics


def run_backtest(
    conn: sqlite3.Connection,
    min_train_months: int = 36,
    max_windows: int | None = None,
    skip_gamlss: bool = False,
) -> int:
    """Walk-forward expanding-window backtest.

    For each validation date, fits VECM (and optionally GAMLSS + hybrid),
    collects forecast vs actual comparisons, and writes to backtest_results and
    accuracy_log. Returns the model_runs.id for this backtest run, or -1 if
    no validation dates are found.
    """
    validation_dates = get_validation_dates(conn, min_train_months=min_train_months)
    if not validation_dates:
        log.warning("No validation dates found — need > %d months of KC=F data", min_train_months)
        return -1

    if max_windows is not None:
        validation_dates = validation_dates[:max_windows]

    log.info(
        "Starting backtest: %d windows, min_train_months=%d, skip_gamlss=%s",
        len(validation_dates), min_train_months, skip_gamlss,
    )

    cur = conn.execute(
        "INSERT INTO model_runs"
        " (run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            "backtest",
            validation_dates[0],
            validation_dates[-1],
            json.dumps({
                "min_train_months": min_train_months,
                "n_windows": len(validation_dates),
                "skip_gamlss": skip_gamlss,
            }),
            "{}",
            "pending",
        ),
    )
    conn.commit()
    run_id = int(cur.lastrowid)
    backtest_date = datetime.now(UTC).strftime("%Y-%m-%d")

    try:
        for i, train_end in enumerate(validation_dates):
            log.info("Window %d/%d: train_end=%s", i + 1, len(validation_dates), train_end)

            vecm_run_id = run_vecm_model(conn, max_date=train_end)
            if vecm_run_id == -1:
                log.warning("VECM skipped for train_end=%s — no data", train_end)
                continue

            if skip_gamlss:
                forecast_run_id = vecm_run_id
            else:
                gamlss_run_id = run_gamlss_model(conn, vecm_run_id, max_date=train_end)
                if gamlss_run_id == -1:
                    log.warning("GAMLSS skipped for train_end=%s — using VECM-only", train_end)
                    forecast_run_id = vecm_run_id
                else:
                    hybrid_run_id = run_hybrid_model(conn, vecm_run_id, gamlss_run_id, max_date=train_end)
                    if hybrid_run_id == -1:
                        log.warning("Hybrid skipped for train_end=%s — using VECM-only", train_end)
                        forecast_run_id = vecm_run_id
                    else:
                        forecast_run_id = hybrid_run_id

            rows = collect_backtest_rows(conn, forecast_run_id, train_end)
            write_backtest_results(conn, run_id, backtest_date, rows)
            write_accuracy_log_entries(conn, rows)

    except Exception:
        conn.execute("UPDATE model_runs SET status = 'failed' WHERE id = ?", (run_id,))
        conn.commit()
        raise

    summary = compute_summary_metrics(conn, run_id)
    log.info("Backtest summary: %s", summary)

    conn.execute(
        "UPDATE model_runs SET status = 'success', metrics = ? WHERE id = ?",
        (json.dumps(summary), run_id),
    )
    conn.commit()
    log.info("Backtest complete: run_id=%d, windows=%d", run_id, len(validation_dates))
    return run_id


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Walk-forward expanding-window backtest")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=36,
        help="Minimum training months before first validation window (default: 36)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit backtest to first N windows (default: all)",
    )
    parser.add_argument(
        "--skip-gamlss",
        action="store_true",
        help="Skip GAMLSS/hybrid step — run VECM-only backtest (no interval calibration)",
    )
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)

    result = run_backtest(
        conn,
        min_train_months=args.min_train_months,
        max_windows=args.max_windows,
        skip_gamlss=args.skip_gamlss,
    )
    if result == -1:
        raise RuntimeError(
            "Backtest skipped — not enough data. "
            f"Need > {args.min_train_months} months of KC=F prices for all 6 symbols."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
