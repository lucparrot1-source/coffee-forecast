import argparse
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from coffee_forecast.alerts import send_pipeline_alert
from coffee_forecast.db import get_connection
from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.logging_config import configure_logging

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # …/coffee_stats_modelling
_R_SCRIPT = _PROJECT_ROOT / "r" / "gamlss_fit.R"
RSCRIPT_TIMEOUT = 120  # seconds

# Candidate Rscript binary paths — checked in order when not on PATH
_RSCRIPT_CANDIDATES = [
    "Rscript",  # on PATH (Linux/Mac or Windows after PATH is set)
    r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
]


def _find_rscript() -> str:
    """Return the first usable Rscript binary path, or raise FileNotFoundError."""
    import shutil

    for candidate in _RSCRIPT_CANDIDATES:
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Rscript not found. Install R from https://cran.r-project.org/ "
        "and ensure it is on your PATH (or restart your terminal after install)."
    )


# ---------------------------------------------------------------------------
# Regime labelling
# ---------------------------------------------------------------------------


def compute_regime_labels(conn: sqlite3.Connection) -> pd.Series:
    """Return Low/Medium/High rolling-vol regime for KC=F monthly prices.

    Uses a 12-month rolling window on log-returns; 33rd/67th percentile
    thresholds computed over the full available history (same method as EDA).

    Returns a pd.Series indexed by DatetimeIndex with values Low/Medium/High.
    An empty Series is returned when no KC=F data is present.
    """
    df = pd.read_sql(
        "SELECT date, adj_close FROM prices_monthly WHERE symbol = 'KC=F' ORDER BY date",
        conn,
    )
    if df.empty:
        return pd.Series(dtype=str, name="regime")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["log_ret"] = np.log(df["adj_close"]).diff()
    df["rolling_vol"] = df["log_ret"].rolling(12).std()
    df = df.dropna(subset=["rolling_vol"])

    q33 = df["rolling_vol"].quantile(1 / 3)
    q67 = df["rolling_vol"].quantile(2 / 3)

    df["regime"] = pd.cut(
        df["rolling_vol"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low", "Medium", "High"],
    )
    result: pd.Series = df["regime"].astype(str)
    result.index.name = "date"
    result.name = "regime"
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_residuals(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Load vecm_residuals for a given run_id.

    Returns DataFrame with columns: date, symbol, residual.
    """
    return pd.read_sql(
        "SELECT date, symbol, residual FROM vecm_residuals WHERE run_id = ?",
        conn,
        params=(run_id,),
    )


def build_gamlss_input(residuals_df: pd.DataFrame, regime_series: pd.Series) -> pd.DataFrame:
    """Join residuals with regime labels on date.

    residuals_df : columns date (str YYYY-MM-DD), symbol, residual
    regime_series: DatetimeIndex → regime label

    Returns DataFrame with columns: date, symbol, residual, regime.
    Rows with no matching regime label are dropped with a warning.
    """
    regime_df = regime_series.reset_index()
    regime_df["date"] = regime_df["date"].dt.strftime("%Y-%m-%d")

    merged = residuals_df.merge(regime_df, on="date", how="inner")
    dropped = len(residuals_df) - len(merged)
    if dropped:
        log.warning("Dropped %d residual rows with no matching regime label", dropped)

    return merged[["date", "symbol", "residual", "regime"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# R subprocess
# ---------------------------------------------------------------------------


def call_rscript(
    input_csv: str,
    output_csv: str,
    r_script: Path = _R_SCRIPT,
) -> None:
    """Run gamlss_fit.R as a subprocess, blocking until completion.

    Raises FileNotFoundError  if Rscript or the R script is missing.
    Raises RuntimeError       if Rscript exits with a non-zero return code.
    Raises subprocess.TimeoutExpired if R takes longer than RSCRIPT_TIMEOUT seconds.
    """
    if not r_script.exists():
        raise FileNotFoundError(
            f"R script not found: {r_script}\n"
            "Check that the 'r/' directory is present in the project root."
        )

    rscript_bin = _find_rscript()
    result = subprocess.run(
        [rscript_bin, str(r_script), "--input", input_csv, "--output", output_csv],
        capture_output=True,
        text=True,
        timeout=RSCRIPT_TIMEOUT,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log.info("R: %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log.warning("R stderr: %s", line)

    if result.returncode != 0:
        raise RuntimeError(
            f"Rscript exited with code {result.returncode}.\n"
            "If R is not installed: https://cran.r-project.org/\n"
            "If gamlss is missing, run in R: install.packages('gamlss')\n"
            f"--- R stderr ---\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def parse_gamlss_output(output_csv: str) -> pd.DataFrame:
    """Read and validate the CSV written by gamlss_fit.R.

    Raises ValueError if any required column is absent.
    """
    df = pd.read_csv(output_csv)
    required = {
        "symbol", "regime",
        "mu", "sigma", "nu", "tau",
        "q10", "q25", "q50", "q75", "q90",
        "n_obs",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GAMLSS output CSV missing columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def _float_or_none(val: object) -> "float | None":
    """Convert to float, returning None for NaN or unconvertible values."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def write_gamlss_params(
    conn: sqlite3.Connection, run_id: int, params_df: pd.DataFrame
) -> None:
    """Upsert GAMLSS parameters into gamlss_params table."""
    records = [
        (
            run_id,
            str(row["symbol"]),
            str(row["regime"]),
            _float_or_none(row["mu"]),
            _float_or_none(row["sigma"]),
            _float_or_none(row["nu"]),
            _float_or_none(row["tau"]),
            _float_or_none(row["q10"]),
            _float_or_none(row["q25"]),
            _float_or_none(row["q50"]),
            _float_or_none(row["q75"]),
            _float_or_none(row["q90"]),
            int(row["n_obs"]),
        )
        for _, row in params_df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO gamlss_params"
        " (run_id, symbol, regime, mu, sigma, nu, tau, q10, q25, q50, q75, q90, n_obs)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()
    log.info("Wrote %d gamlss_params rows for run_id=%d", len(records), run_id)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_gamlss_model(conn: sqlite3.Connection, vecm_run_id: int) -> int:
    """Orchestrate: load residuals → regimes → CSV → Rscript → store params.

    Returns the model_runs.id for this GAMLSS run, or -1 if skipped due to
    missing data.
    """
    residuals_df = load_residuals(conn, vecm_run_id)
    if residuals_df.empty:
        log.warning("No residuals found for vecm_run_id=%d — skipping GAMLSS", vecm_run_id)
        return -1

    regime_series = compute_regime_labels(conn)
    if regime_series.empty:
        log.warning("No KC=F price data for regime labels — skipping GAMLSS")
        return -1

    gamlss_input = build_gamlss_input(residuals_df, regime_series)
    if gamlss_input.empty:
        log.warning("No overlapping dates between residuals and regime labels — skipping GAMLSS")
        return -1

    log.info(
        "GAMLSS input: %d rows, symbols=%s, regimes=%s",
        len(gamlss_input),
        list(gamlss_input["symbol"].unique()),
        list(gamlss_input["regime"].unique()),
    )

    # Register the run as pending before calling R (so a crash is traceable)
    cur = conn.execute(
        "INSERT INTO model_runs (run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            "gamlss",
            gamlss_input["date"].min(),
            gamlss_input["date"].max(),
            json.dumps({
                "vecm_run_id": vecm_run_id,
                "family": "SHASH",
                "conditioning": "rolling_vol_regime",
                "symbols": sorted(gamlss_input["symbol"].unique().tolist()),
            }),
            "{}",
            "pending",
        ),
    )
    conn.commit()
    run_id = int(cur.lastrowid)  # type: ignore[arg-type]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "gamlss_input.csv")
            output_csv = os.path.join(tmpdir, "gamlss_output.csv")
            gamlss_input.to_csv(input_csv, index=False)
            call_rscript(input_csv, output_csv)
            params_df = parse_gamlss_output(output_csv)

        write_gamlss_params(conn, run_id, params_df)
    except Exception:
        conn.execute("UPDATE model_runs SET status = 'failed' WHERE id = ?", (run_id,))
        conn.commit()
        raise

    conn.execute("UPDATE model_runs SET status = 'success' WHERE id = ?", (run_id,))
    conn.commit()
    log.info("GAMLSS run complete: run_id=%d, params_rows=%d", run_id, len(params_df))
    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fit GAMLSS SHASH and write params to DB")
    parser.add_argument("--db", default=None, help="Path to SQLite DB (overrides COFFEE_DB_PATH)")
    parser.add_argument(
        "--vecm-run-id",
        type=int,
        default=None,
        help="VECM model_runs.id to use (default: latest successful VECM run)",
    )
    args = parser.parse_args()

    if args.db:
        os.environ["COFFEE_DB_PATH"] = args.db

    conn = get_connection()
    ensure_schema(conn)

    vecm_run_id = args.vecm_run_id
    if vecm_run_id is None:
        row = conn.execute(
            "SELECT id FROM model_runs"
            " WHERE model_type = 'vecm' AND status = 'success'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            # Raise so the __main__ guard fires the Resend alert and exits non-zero.
            raise RuntimeError("No successful VECM run found — run the VECM model first")
        vecm_run_id = int(row[0])
        log.info("Using latest VECM run_id=%d", vecm_run_id)

    run_gamlss_model(conn, vecm_run_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send_pipeline_alert(__file__, traceback.format_exc())
        raise
