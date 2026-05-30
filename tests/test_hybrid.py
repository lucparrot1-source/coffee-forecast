import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from coffee_forecast.db.migrations import ensure_schema
from coffee_forecast.models.hybrid import (
    combine_forecasts,
    get_current_regime,
    load_gamlss_quantiles,
    load_vecm_forecasts,
    run_hybrid_model,
    write_hybrid_forecasts,
)


@pytest.fixture
def mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _seed_model_runs(conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert VECM and GAMLSS model run rows, return (vecm_run_id, gamlss_run_id)."""
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (1, '2024-01-01T00:00:00', 'vecm', '2014-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    conn.execute(
        "INSERT INTO model_runs (id, run_at, model_type, train_start, train_end, params, metrics, status)"
        " VALUES (2, '2024-01-01T00:00:00', 'gamlss', '2014-01-01', '2024-01-01', '{}', '{}', 'success')"
    )
    conn.commit()
    return 1, 2


def _seed_vecm_forecasts(conn: sqlite3.Connection, vecm_run_id: int) -> None:
    conn.executemany(
        "INSERT INTO forecasts (run_id, forecast_date, target_date, horizon, symbol, point_forecast)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (vecm_run_id, "2024-01-01", "2024-02-01", 1, "KC=F", 180.0),
            (vecm_run_id, "2024-01-01", "2024-03-01", 2, "KC=F", 185.0),
            (vecm_run_id, "2024-01-01", "2024-04-01", 3, "KC=F", 190.0),
            (vecm_run_id, "2024-01-01", "2024-02-01", 1, "RM=F", 100.0),
            (vecm_run_id, "2024-01-01", "2024-03-01", 2, "RM=F", 102.0),
            (vecm_run_id, "2024-01-01", "2024-04-01", 3, "RM=F", 104.0),
        ],
    )
    conn.commit()


def _seed_gamlss_params(conn: sqlite3.Connection, gamlss_run_id: int) -> None:
    conn.executemany(
        "INSERT INTO gamlss_params"
        " (run_id, symbol, regime, mu, sigma, nu, tau, q10, q25, q50, q75, q90, n_obs)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (gamlss_run_id, "KC=F", "Low",    0.01, 0.05, -0.1, 2.0, -0.07, -0.03, 0.01, 0.05, 0.09, 45),
            (gamlss_run_id, "RM=F", "Low",    0.00, 0.06, -0.1, 2.0, -0.08, -0.04, 0.00, 0.04, 0.08, 45),
            (gamlss_run_id, "KC=F", "Medium", 0.00, 0.08, -0.05, 1.8, -0.11, -0.05, 0.00, 0.05, 0.11, 42),
            (gamlss_run_id, "RM=F", "Medium", 0.00, 0.09, -0.05, 1.8, -0.12, -0.06, 0.00, 0.06, 0.12, 42),
            (gamlss_run_id, "KC=F", "High",   0.00, 0.12, -0.02, 1.5, -0.15, -0.08, 0.00, 0.08, 0.15, 30),
            (gamlss_run_id, "RM=F", "High",   0.00, 0.13, -0.02, 1.5, -0.16, -0.09, 0.00, 0.09, 0.16, 30),
        ],
    )
    conn.commit()


def _seed_kc_prices(conn: sqlite3.Connection, n: int = 48) -> None:
    """Insert synthetic KC=F monthly prices into prices_monthly."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    prices = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    conn.executemany(
        "INSERT INTO prices_monthly (date, symbol, adj_close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "KC=F", float(p)) for d, p in zip(dates, prices)],
    )
    conn.commit()


# --- Tests for Task 1 ---

def test_load_vecm_forecasts_returns_six_rows(mem_conn: sqlite3.Connection) -> None:
    vecm_run_id, _ = _seed_model_runs(mem_conn)
    _seed_vecm_forecasts(mem_conn, vecm_run_id)
    df = load_vecm_forecasts(mem_conn, vecm_run_id)
    assert len(df) == 6
    assert set(df.columns) >= {"horizon", "symbol", "point_forecast"}


def test_load_vecm_forecasts_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    df = load_vecm_forecasts(mem_conn, vecm_run_id=999)
    assert df.empty


def test_load_gamlss_quantiles_returns_six_rows(mem_conn: sqlite3.Connection) -> None:
    _, gamlss_run_id = _seed_model_runs(mem_conn)
    _seed_gamlss_params(mem_conn, gamlss_run_id)
    df = load_gamlss_quantiles(mem_conn, gamlss_run_id)
    assert len(df) == 6
    assert set(df.columns) >= {"symbol", "regime", "q10", "q25", "q50", "q75", "q90"}


def test_load_gamlss_quantiles_empty_for_unknown_run(mem_conn: sqlite3.Connection) -> None:
    df = load_gamlss_quantiles(mem_conn, gamlss_run_id=999)
    assert df.empty
